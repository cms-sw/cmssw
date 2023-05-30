#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include <algorithm>
#include <fstream>
#include <regex>
#include <unordered_map>

#ifdef CMSSW_GIT_HASH
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#endif
#include "L1Trigger/DemonstratorTools/interface/BoardData.h"

namespace {

  l1t::demo::BoardData createBoardDataFromRows(const std::string& id,
                                               const std::vector<size_t>& channels,
                                               const std::vector<std::vector<l1t::demo::Frame>>& dataRows) {
    l1t::demo::BoardData boardData(id);

    for (size_t i = 0; i < channels.size(); i++) {
      std::vector<l1t::demo::Frame> channelData(dataRows.size());
      for (size_t j = 0; j < dataRows.size(); j++)
        channelData.at(j) = dataRows.at(j).at(i);
      boardData.add(channels.at(i), channelData);
    }

    return boardData;
  }

  std::vector<std::string> searchAndTokenize(std::istream& file, const std::string& linePrefix) {
    std::string line;

    while (getline(file, line)) {
      // Strip leading spaces
      size_t startIndex = line.find_first_not_of(" \t");
      if (startIndex != std::string::npos)
        line = line.substr(startIndex);

      if (line.empty())
        continue;
      if (line[0] == '#')
        continue;

      if (line.rfind(linePrefix, 0) != std::string::npos) {
        std::vector<std::string> tokens;

        // Split the line into tokens
        const std::regex delimiterRegex("\\s+");
        std::sregex_token_iterator it(line.begin() + linePrefix.size(), line.end(), delimiterRegex, -1);

        for (; it != std::sregex_token_iterator(); it++) {
          const std::string token(it->str());
          if (token.empty())
            continue;
          tokens.push_back(token);
        }

        return tokens;
      } else
        throw std::logic_error("Found unexpected line found when searching for \"" + linePrefix + "\": \"" + line +
                               "\"");
    }
    throw std::logic_error("Couldn't find any line starting with \"" + linePrefix + "\"");
  }

}  // namespace

namespace l1t::demo {

  FileFormat parseFileFormat(const std::string& s) {
    static const std::unordered_map<std::string, FileFormat> kFormatStringMap({{"EMP", FileFormat::EMPv1},
                                                                               {"emp", FileFormat::EMPv1},
                                                                               {"EMPv1", FileFormat::EMPv1},
                                                                               {"EMPv2", FileFormat::EMPv2},
                                                                               {"APx", FileFormat::APx},
                                                                               {"apx", FileFormat::APx},
                                                                               {"X2O", FileFormat::X2O},
                                                                               {"x2O", FileFormat::X2O}});

    const auto it = kFormatStringMap.find(s);
    if (it == kFormatStringMap.end())
      throw std::runtime_error("Could not convert '" + s + "' to FileFormat enum value");

    return it->second;
  }

  BoardData readAPxFile(std::istream&);

  BoardData readEMPFileV1(std::istream&);

  BoardData readEMPFileV2(std::istream&);

  BoardData readX2OFile(std::istream&);

  BoardData read(const std::string& filePath, const FileFormat format) {
    std::ifstream file(filePath);

    if (not file.is_open())
      throw std::runtime_error("Could not open file '" + filePath + "'");

    return read(file, format);
  }

  BoardData read(std::istream& file, const FileFormat format) {
    switch (format) {
      case FileFormat::APx:
        return readAPxFile(file);
      case FileFormat::EMPv1:
        return readEMPFileV1(file);
      case FileFormat::EMPv2:
        return readEMPFileV2(file);
      case FileFormat::X2O:
        return readX2OFile(file);
    }

    std::ostringstream messageStream;
    messageStream << "No read function registered for format " << format;
    throw std::runtime_error(messageStream.str());
  }

  BoardData readAPxFile(std::istream& file) {
    std::string line;

    // Complain if file is empty
    if (not std::getline(file, line))
      throw std::runtime_error("Specified file is empty!");

    // If first line is sideband, skip it and move onto 2nd line
    if (line.find("#Sideband") == 0) {
      if (not std::getline(file, line))
        throw std::runtime_error("APx file has incorrect format: Link labels and data missing!");
    }

    // Parse link labels
    if (line.find("#LinkLabel") != 0)
      throw std::runtime_error(
          "APx file has incorrect format: Link header does not start with '#LinkLabel' (line is '" + line + "')");

    std::vector<size_t> indices;
    const std::regex delimiterRegex("\\s+");
    std::sregex_token_iterator it(line.begin() + 10, line.end(), delimiterRegex, -1);
    for (; it != std::sregex_token_iterator(); it++) {
      const std::string token(it->str());
      if (token.empty())
        continue;

      if (token.find("LINK_") != 0)
        throw std::runtime_error("Link column name '" + token + "' (does not start with 'LINK_')");
      if (token.size() == 5)
        throw std::runtime_error("Link column name '" + token + "' is too short");
      if (not std::all_of(token.begin() + 5, token.end(), ::isdigit))
        throw std::runtime_error("Link column name '" + token + "' does not end with a number");

      indices.push_back(std::stoul(token.substr(5)));
    }

    // Check for '#BeginData' line
    if (not std::getline(file, line))
      throw std::runtime_error("APx file has incorrect format: Data missing!");
    if (line != "#BeginData")
      throw std::runtime_error("APx file has incorrect format: '#BeginData' line missing (found '" + line + "')");

    // Parse link data
    std::vector<std::vector<l1t::demo::Frame>> dataRows;
    while (std::getline(file, line)) {
      it = std::sregex_token_iterator(line.begin(), line.end(), delimiterRegex, -1);
      size_t count = 0;
      for (; it != std::sregex_token_iterator(); it++, count++) {
        const std::string token(it->str());

        if ((token.find("0x") != 0) or (not std::all_of(token.begin() + 2, token.end(), ::isxdigit)))
          throw std::runtime_error("APx file has incorrect format: Data token '" + token +
                                   "' is not hexadecimal number");

        if (count == 0) {
          size_t rowIndex = std::stoul(token, nullptr, 16);
          if (rowIndex != dataRows.size())
            throw std::runtime_error("APx file has incorrect format: Expected data row " +
                                     std::to_string(dataRows.size()) + ", but found row " + std::to_string(rowIndex));
          dataRows.push_back(std::vector<l1t::demo::Frame>(indices.size()));
        }
        // Sideband info
        else if ((count % 2) == 1) {
          uint16_t sbValue = std::stoul(token, nullptr, 16);
          dataRows.back().at((count - 1) / 2).valid = (sbValue & 0x1);
          dataRows.back().at((count - 1) / 2).startOfPacket = ((sbValue >> 1) & 0x1);
          dataRows.back().at((count - 1) / 2).endOfPacket = ((sbValue >> 3) & 0x1);
        }
        // Data word
        else
          dataRows.back().at((count - 1) / 2).data = ap_uint<64>(std::stoull(token, nullptr, 16));
      }

      if (count != (2 * indices.size() + 1))
        throw std::runtime_error("APx file has incorrect format: Line has incorrect number of tokens (expected " +
                                 std::to_string(2 * indices.size() + 1) + ", found " + std::to_string(count) + "!");
    }

    return createBoardDataFromRows("", indices, dataRows);
  }

  BoardData readEMPFileV1(std::istream& file) {
    // 1) Search for ID string
    std::string id, line;
    while (getline(file, line)) {
      if (line.empty())
        continue;
      if (line[0] == '#')
        continue;

      if (line.rfind("Board ", 0) != std::string::npos) {
        id = line.substr(6);
        break;
      } else
        throw std::logic_error("Found unexpected line found when searching for board ID: \"" + line + "\"");
    }

    // 2) Search for column labels (i.e. list of channels/links)
    searchAndTokenize(file, "Quad/Chan :");
    const auto tokens = searchAndTokenize(file, "Link :");
    std::vector<size_t> channels;
    std::transform(tokens.begin(), tokens.end(), std::back_inserter(channels), [](const std::string& s) {
      return std::stoull(s);
    });

    // 3) Read the main data rows
    const std::regex delimiterRegex("\\s+");
    static const std::regex frameRegex("([01]s)?([01]v)([0-9a-fA-F]{16})");
    std::vector<std::vector<Frame>> dataRows;
    while (file.good() and getline(file, line)) {
      if (line.empty() or line[0] == '#')
        continue;

      std::ostringstream prefixStream;
      prefixStream << "Frame ";
      prefixStream << std::setw(4) << std::setfill('0') << dataRows.size();
      prefixStream << " :";

      const std::string prefix(prefixStream.str());
      if (line.rfind(prefix, 0) == std::string::npos)
        throw std::logic_error("Found unexpected line found when searching for \"" + prefix + "\": \"" + line + "\"");

      std::vector<l1t::demo::Frame> row;
      std::sregex_token_iterator it(line.begin() + prefix.size(), line.end(), delimiterRegex, -1);
      for (; it != std::sregex_token_iterator(); it++) {
        const std::string token(it->str());
        if (token.empty())
          continue;

        std::smatch what;
        if (not std::regex_match(token, what, frameRegex))
          throw std::logic_error("Token '" + token + "' doesn't match the valid format");

        l1t::demo::Frame value;
        // Import strobe if the strobe group is matched
        if (what[1].matched) {
          value.strobe = (what[1] == "1s");
        }

        value.valid = (what[2] == "1v");
        value.data = ap_uint<64>(std::stoull(what[3].str(), nullptr, 16));

        row.push_back(value);
      }

      dataRows.push_back(row);
    }

    return createBoardDataFromRows(id, channels, dataRows);
  }

  BoardData readEMPFileV2(std::istream& file) {
    // 1) Search for ID string
    std::string id, line;
    while (getline(file, line)) {
      if (line.empty())
        continue;
      if (line[0] == '#')
        continue;

      if (line.rfind("ID: ", 0) != std::string::npos) {
        id = line.substr(4);
        break;
      } else
        throw std::logic_error("Found unexpected line found when searching for board ID: \"" + line + "\"");
    }

    // 2) Check that next line states metadata formatting
    getline(file, line);
    if (line.find("Metadata: (strobe,) start of orbit, start of packet, end of packet, valid") != 0)
      throw std::logic_error("Expected metadata line following 'ID' line. Instead found:" + line);

    // 3) Search for column labels (i.e. list of channels/links)
    const auto tokens = searchAndTokenize(file, "Link  ");
    std::vector<size_t> channels;
    std::transform(tokens.begin(), tokens.end(), std::back_inserter(channels), [](const std::string& s) {
      return std::stoull(s);
    });

    // 4) Read the main data rows
    const std::regex delimiterRegex("\\s\\s+");
    static const std::regex frameRegex("([01])?([01])([01])([01])([01]) ([0-9a-fA-F]{16})");
    std::vector<std::vector<Frame>> dataRows;
    while (file.good() and getline(file, line)) {
      if (line.empty() or line[0] == '#')
        continue;

      std::ostringstream prefixStream;
      prefixStream << "Frame ";
      prefixStream << std::setw(4) << std::setfill('0') << dataRows.size();
      prefixStream << "  ";

      const std::string prefix(prefixStream.str());
      if (line.rfind(prefix, 0) == std::string::npos)
        throw std::logic_error("Found unexpected line found when searching for \"" + prefix + "\": \"" + line + "\"");

      std::vector<l1t::demo::Frame> row;
      std::sregex_token_iterator it(line.begin() + prefix.size(), line.end(), delimiterRegex, -1);
      for (; it != std::sregex_token_iterator(); it++) {
        const std::string token(it->str());
        if (token.empty())
          continue;

        std::smatch what;
        if (not std::regex_match(token, what, frameRegex))
          throw std::logic_error("Token '" + token + "' doesn't match the valid format");

        l1t::demo::Frame value;
        // Import strobe if the strobe group is matched
        if (what[1].matched) {
          value.strobe = (what[1] == "1");
        }

        value.startOfOrbit = (what[2] == "1");
        value.startOfPacket = (what[3] == "1");
        value.endOfPacket = (what[4] == "1");
        value.valid = (what[5] == "1");
        value.data = ap_uint<64>(std::stoull(what[6].str(), nullptr, 16));

        row.push_back(value);
      }

      dataRows.push_back(row);
    }

    return createBoardDataFromRows(id, channels, dataRows);
  }

  BoardData readX2OFile(std::istream& file) {
    throw std::runtime_error("Reading X2O file format not yet implemented. Will be done ASAP.");
  }

  void writeAPxFile(const BoardData&, std::ostream&);

  void writeEMPFileV1(const BoardData&, std::ostream&);

  void writeEMPFileV2(const BoardData&, std::ostream&);

  void writeX2OFile(const BoardData&, std::ostream&);

  void write(const BoardData& data, const std::string& filePath, const FileFormat format) {
    // Open file
#ifdef CMSSW_GIT_HASH
    edm::LogInfo("L1TDemonstratorTools")
#else
    std::cout
#endif
        << "Writing board data (" << std::distance(data.begin(), data.end()) << " channels, "
        << data.begin()->second.size() << " frames) to file '" << filePath << "' (format: " << format << ")"
        << std::endl;
    std::ofstream file(filePath);

    if (not file.is_open())
      throw std::runtime_error("Could not open file '" + filePath + "'");

    write(data, file, format);
  }

  void write(const BoardData& data, std::ostream& file, const FileFormat format) {
    // Check that number of frames is same for every channel
    const auto firstChannel = data.begin();

    for (const auto& channel : data) {
      const auto i = channel.first;
      const auto channelData = channel.second;
      if (channelData.size() != firstChannel->second.size())
        throw std::runtime_error("Cannot write board data to file - channels do not all have the same length (" +
                                 std::to_string(channelData.size()) + " words on channel " + std::to_string(i) +
                                 ", but " + std::to_string(firstChannel->second.size()) + " words on channel " +
                                 std::to_string(firstChannel->first) + ")");
    }

    // Call relevant write function
    switch (format) {
      case FileFormat::APx:
        writeAPxFile(data, file);
        return;
      case FileFormat::EMPv1:
        writeEMPFileV1(data, file);
        return;
      case FileFormat::EMPv2:
        writeEMPFileV2(data, file);
        return;
      case FileFormat::X2O:
        writeX2OFile(data, file);
        return;
    }
  }

  void writeAPxFile(const BoardData& data, std::ostream& file) {
    // Note: APx sideband encoding
    //   Short-term, simulation only:
    //     0 -> Valid
    //     1 -> EOF
    //   Planned (from ~ May 2021)
    //     0 -> Valid
    //     1 -> SOF (Start Of Frame)
    //     2 -> FFO (First Frame of Orbit)
    //     3 -> EOF (End Of Frame)
    //     4 -> FERR (Frame Error)
    //     5 -> RSV1
    //     6 -> RSV2
    //     7 -> RSV3

    file << std::setfill('0');
    file << "#Sideband ON" << std::endl;

    // Channel header
    file << "#LinkLabel";
    for (const auto& channel : data) {
      const auto i = channel.first;
      file << "                LINK_" << std::setw(2) << i << "    ";
    }
    file << std::endl;

    file << "#BeginData" << std::endl;

    // Frames
    file << std::hex;
    const auto firstChannel = data.begin();
    for (size_t i = 0; i < firstChannel->second.size(); i++) {
      file << "0x" << std::setw(4) << i;
      for (const auto& channel : data) {
        //const auto j = channel.first;
        const auto channelData = channel.second;
        uint16_t sideband = channelData.at(i).valid;
        sideband |= channelData.at(i).startOfPacket << 1;
        sideband |= channelData.at(i).endOfPacket << 3;
        file << "    0x" << std::setw(2) << sideband;
        file << " 0x" << std::setw(16) << uint64_t(channelData.at(i).data);
      }
      file << std::endl;
    }
  }

  void writeEMPFileV1(const BoardData& data, std::ostream& file) {
    file << std::setfill('0');

    // Board name/id
    file << "Board CMSSW" << std::endl;

    // Quad/chan header
    file << " Quad/Chan :";
    for (const auto& channel : data) {
      const auto i = channel.first;
      file << "         q" << std::setw(2) << i / 4 << 'c' << std::setw(1) << i % 4 << "       ";
    }
    file << std::endl;

    // Link header
    file << "      Link :";
    for (const auto& channel : data) {
      const auto i = channel.first;
      file << "          " << std::setw(3) << i << "        ";
    }
    file << std::endl;

    // Frames
    const auto firstChannel = data.begin();
    for (size_t i = 0; i < firstChannel->second.size(); i++) {
      file << "Frame " << std::setw(4) << i << " :";
      for (const auto& channel : data) {
        //const auto j = channel.first;
        const auto channelData = channel.second;
        file << " ";
        //TODO: Add strobe if zero anywhere on channel
        file << "  ";
        file << std::setw(1) << channelData.at(i).valid << "v" << std::setw(16) << std::hex
             << uint64_t(channelData.at(i).data);
      }
      file << std::endl << std::dec;
    }
  }

  void writeEMPFileV2(const BoardData& data, std::ostream& file) {
    file << std::setfill('0');

    // Board name/id
    file << "ID: " << data.name() << std::endl;
    file << "Metadata: (strobe,) start of orbit, start of packet, end of packet, valid" << std::endl;
    file << std::endl;

    // Link header
    file << "      Link  ";
    std::map<size_t, bool> strobedLinkMap;
    for (const auto& channel : data) {
      const auto i = channel.first;
      strobedLinkMap[i] =
          std::any_of(channel.second.begin(), channel.second.end(), [](const Frame& x) { return not x.strobe; });
      if (strobedLinkMap.at(i))
        file << " ";
      file << "            " << std::setw(3) << i << "        ";
    }
    file << std::endl;

    // Frames
    const auto firstChannel = data.begin();
    for (size_t i = 0; i < firstChannel->second.size(); i++) {
      file << "Frame " << std::setw(4) << i << "  ";
      for (const auto& channel : data) {
        //const auto j = channel.first;
        const auto channelData = channel.second;
        file << "  ";
        if (strobedLinkMap.at(channel.first))
          file << std::setw(1) << channelData.at(i).strobe;
        file << std::setw(1) << channelData.at(i).startOfOrbit << channelData.at(i).startOfPacket
             << channelData.at(i).endOfPacket << channelData.at(i).valid;
        file << " " << std::setw(16) << std::hex << uint64_t(channelData.at(i).data);
      }
      file << std::endl << std::dec;
    }
  }

  void writeX2OFile(const BoardData& data, std::ostream& file) {
    throw std::runtime_error("Writing X2O file format not yet implemented. Will be done ASAP.");
  }

}  // namespace l1t::demo
