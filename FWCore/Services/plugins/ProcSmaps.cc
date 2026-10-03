#include "FWCore/Services/plugins/ProcSmaps.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <fstream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string_view>

namespace {
  struct Mapping {
    edm::service::SmapsSection section = edm::service::SmapsSection::kOther;
    std::string path;
    bool executable = false;
    std::uint64_t sizeKB = 0;
    std::uint64_t rssKB = 0;
    std::uint64_t pssKB = 0;
  };

  bool isSharedLibrary(std::string_view path) {
    auto const position = path.rfind(".so");
    if (position == std::string_view::npos) {
      return false;
    }
    auto const suffix = path.substr(position + 3);
    return suffix.empty() or suffix.front() == '.';
  }

  bool isMappingHeader(std::string_view line) {
    auto const dash = line.find('-');
    auto const space = line.find(' ');
    if (dash == std::string_view::npos or space == std::string_view::npos or dash > space) {
      return false;
    }
    return std::all_of(line.begin(), line.begin() + dash, [](unsigned char c) { return std::isxdigit(c); }) and
           std::all_of(line.begin() + dash + 1, line.begin() + space, [](unsigned char c) { return std::isxdigit(c); });
  }

  Mapping parseHeader(std::string const& line) {
    Mapping mapping;
    std::istringstream stream(line);
    std::string addresses;
    std::string permissions;
    std::string offset;
    std::string device;
    std::string inode;
    stream >> addresses >> permissions >> offset >> device >> inode;
    mapping.executable = permissions.size() > 2 and permissions[2] == 'x';
    std::getline(stream, mapping.path);
    auto const first = mapping.path.find_first_not_of(' ');
    if (first == std::string::npos) {
      mapping.path.clear();
      mapping.section = edm::service::SmapsSection::kMmap;
    } else {
      mapping.path.erase(0, first);
      if (mapping.path.front() == '/') {
        if (mapping.path.ends_with(".pcm")) {
          mapping.section = edm::service::SmapsSection::kPcm;
        } else if (isSharedLibrary(mapping.path)) {
          mapping.section = edm::service::SmapsSection::kSharedObject;
        } else {
          mapping.section = edm::service::SmapsSection::kOtherFile;
        }
      } else if (mapping.path == "[stack]") {
        mapping.section = edm::service::SmapsSection::kStack;
      }
    }
    return mapping;
  }

  std::uint64_t valueKB(std::string_view line) {
    auto const colon = line.find(':');
    if (colon == std::string_view::npos) {
      return 0;
    }
    auto const first = line.find_first_not_of(' ', colon + 1);
    if (first == std::string_view::npos) {
      return 0;
    }
    std::uint64_t value = 0;
    std::from_chars(line.data() + first, line.data() + line.size(), value);
    return value;
  }
}  // namespace

namespace edm::service {
  SmapsInfo parseProcSmaps(std::istream& input) {
    SmapsInfo result;
    std::map<std::string, SmapsLibraryInfo> libraries;
    Mapping mapping;
    bool haveMapping = false;

    auto finishMapping = [&]() {
      if (not haveMapping) {
        return;
      }
      auto const index = static_cast<unsigned int>(mapping.section);
      result.sectionVSize_[index] += static_cast<double>(mapping.sizeKB) / 1024.;
      result.sectionRss_[index] += static_cast<double>(mapping.rssKB) / 1024.;
      if (mapping.section == SmapsSection::kSharedObject) {
        auto& library = libraries[mapping.path];
        library.path = mapping.path;
        library.sizeKB += mapping.sizeKB;
        library.rssKB += mapping.rssKB;
        library.pssKB += mapping.pssKB;
        if (mapping.executable) {
          library.executableSizeKB += mapping.sizeKB;
        }
      }
    };

    std::string line;
    while (std::getline(input, line)) {
      if (isMappingHeader(line)) {
        finishMapping();
        mapping = parseHeader(line);
        haveMapping = true;
      } else if (line.starts_with("Private_Clean:") or line.starts_with("Private_Dirty:")) {
        result.private_ += static_cast<double>(valueKB(line)) / 1024.;
      } else if (line.starts_with("Pss:")) {
        auto const value = valueKB(line);
        result.pss_ += static_cast<double>(value) / 1024.;
        mapping.pssKB += value;
      } else if (line.starts_with("AnonHugePages:")) {
        result.anonHugePages_ += static_cast<double>(valueKB(line)) / 1024.;
      } else if (line.starts_with("Rss:")) {
        mapping.rssKB += valueKB(line);
      } else if (line.starts_with("Size:")) {
        mapping.sizeKB += valueKB(line);
      }
    }
    finishMapping();

    result.libraries.reserve(libraries.size());
    for (auto& [path, library] : libraries) {
      result.libraries.emplace_back(std::move(library));
    }
    return result;
  }

  SmapsInfo readProcSmaps(std::string const& path) {
    std::ifstream input(path);
    if (not input) {
      throw std::runtime_error("Failed to open smaps file " + path);
    }
    return parseProcSmaps(input);
  }
}  // namespace edm::service
