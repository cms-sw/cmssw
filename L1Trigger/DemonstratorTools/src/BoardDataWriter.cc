#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"

#include <fstream>

#include "L1Trigger/DemonstratorTools/interface/Frame.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

namespace l1t::demo {

  BoardDataWriter::BoardDataWriter(FileFormat format,
                                   const std::string& path,
                                   const size_t framesPerBX,
                                   const size_t tmux,
                                   const size_t maxFramesPerFile,
                                   const ChannelMap_t& channelSpecs)
      : fileFormat_(format),
        boardDataFileID_("CMSSW"),
        filePathGen_([=](const size_t i) { return path + "_" + std::to_string(i) + ".txt"; }),
        framesPerBX_(framesPerBX),
        boardTMUX_(tmux),
        maxFramesPerFile_(maxFramesPerFile),
        maxEventsPerFile_(maxFramesPerFile_),
        eventIndex_(0),
        pendingEvents_(0),
        channelMap_(channelSpecs) {
    if (channelMap_.empty())
      throw std::runtime_error("BoardDataWriter channel map cannnot be empty");

    for (const auto& [id, value] : channelMap_) {
      const auto& [spec, indices] = value;
      for (const auto i : indices)
        boardData_.add(i);

      if ((spec.tmux % boardTMUX_) != 0)
        throw std::runtime_error("BoardDataWriter, link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "]: Specified TMUX period, " + std::to_string(spec.tmux) +
                                 ", is not a multiple of the board TMUX, " + std::to_string(boardTMUX_));

      const size_t tmuxRatio(spec.tmux / boardTMUX_);
      if (indices.size() != tmuxRatio)
        throw std::runtime_error("BoardDataWriter, link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "]: Number of channel indices specified, " + std::to_string(indices.size()) +
                                 ", does not match link:board TMUX ratio, " + std::to_string(tmuxRatio));

      maxEventsPerFile_ = std::min(maxEventsPerFile_,
                                   ((maxFramesPerFile_ - spec.offset) / (framesPerBX_ * boardTMUX_)) - (tmuxRatio - 1));
    }

    resetBoardData();
  }

  BoardDataWriter::BoardDataWriter(FileFormat format,
                                   const std::string& path,
                                   const size_t framesPerBX,
                                   const size_t tmux,
                                   const size_t maxFramesPerFile,
                                   const std::map<LinkId, std::vector<size_t>>& channelMap,
                                   const std::map<std::string, ChannelSpec>& channelSpecs)
      : BoardDataWriter(format, path, framesPerBX, tmux, maxFramesPerFile, mergeMaps(channelMap, channelSpecs)) {}

  void BoardDataWriter::setBoardDataFileID(const std::string& aId) {
    boardDataFileID_ = aId;
  }

  void BoardDataWriter::addEvent(const EventData& eventData) {
    // Check that data is supplied for each channel
    for (const auto& [id, info] : channelMap_) {
      if (not eventData.has(id))
        throw std::runtime_error("Event data for link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "] is missing.");
    }

    for (const auto& [id, channelData] : eventData) {
      // Check that each channel was declared to constructor
      if (channelMap_.count(id) == 0)
        throw std::runtime_error("Event data for link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "] was given to BoardDataWriter, but its structure was not defined");

      const auto& [spec, indices] = channelMap_.at(id);
      const size_t chanIndex(indices.at(pendingEvents_ % (spec.tmux / boardTMUX_)));

      // Check that that expected amount of data has been provided
      if (channelData.size() > (spec.tmux * framesPerBX_ - spec.interpacketGap))
        throw std::runtime_error("Event data for link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "] (TMUX " + std::to_string(spec.tmux) + ", " + std::to_string(spec.interpacketGap) +
                                 " cycles between packets) is too long (" + std::to_string(channelData.size()) +
                                 " 64-bit words)");

      if (channelData.empty())
        throw std::runtime_error("Event data for link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "] is empty");

      // Copy event data for this channel to board data object
      boardData_.at(chanIndex).insert(boardData_.at(chanIndex).end(), channelData.begin(), channelData.end());

      // Override flags for start & end of event
      BoardData::Channel::iterator it(boardData_.at(chanIndex).end() - 1);
      it->endOfPacket = true;
      it -= (channelData.size() - 1);
      it->startOfPacket = true;

      // Pad link with non-valid frames
      boardData_.at(chanIndex).insert(
          boardData_.at(chanIndex).end(), spec.tmux * framesPerBX_ - channelData.size(), Frame());
    }

    eventIndex_++;
    pendingEvents_++;

    if (pendingEvents_ == maxEventsPerFile_)
      flush();
  }

  void BoardDataWriter::flush() {
    if (pendingEvents_ == 0)
      return;

    // Pad any channels that aren't full with invalid frames
    for (auto& x : boardData_)
      x.second.resize(maxFramesPerFile_);

    // For each channel: Assert start_of_orbit for first clock cycle that start is asserted
    for (auto& x : boardData_) {
      for (auto& frame : x.second) {
        if (frame.startOfPacket) {
          frame.startOfOrbit = true;
          break;
        }
      }
    }

    // Set ID field for board data files
    boardData_.name(boardDataFileID_);

    // Write board data object to file
    const std::string filePath = filePathGen_(fileNames_.size());
    write(boardData_, filePath, fileFormat_);
    fileNames_.push_back(filePath);

    // Clear board data to be ready for next event
    resetBoardData();
  }

  BoardDataWriter::ChannelMap_t BoardDataWriter::mergeMaps(const std::map<LinkId, std::vector<size_t>>& indexMap,
                                                           const std::map<std::string, ChannelSpec>& specMap) {
    ChannelMap_t channelMap;
    for (const auto& x : indexMap)
      channelMap[x.first] = {specMap.at(x.first.interface), x.second};
    return channelMap;
  }

  void BoardDataWriter::resetBoardData() {
    for (auto& x : boardData_)
      x.second.clear();

    for (const auto& [id, value] : channelMap_) {
      const auto& [spec, indices] = value;
      for (size_t tmuxIndex = 0; tmuxIndex < indices.size(); tmuxIndex++)
        boardData_.at(indices.at(tmuxIndex)).resize(tmuxIndex * boardTMUX_ * framesPerBX_ + spec.offset);
    }

    pendingEvents_ = 0;
  }

}  // namespace l1t::demo