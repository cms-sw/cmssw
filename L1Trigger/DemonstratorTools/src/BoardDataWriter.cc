#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"

#include <fstream>

#include "L1Trigger/DemonstratorTools/interface/Frame.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

namespace l1t::demo {

  BoardDataWriter::BoardDataWriter(FileFormat format,
                                   const std::string& path,
                                   const std::string& fileExt,
                                   const size_t framesPerBX,
                                   const size_t tmux,
                                   const size_t maxFramesPerFile,
                                   const ChannelMap_t& channelSpecs,
                                   const bool staggerTmuxSlices)
      : fileFormat_(format),
        boardDataFileID_("CMSSW"),
        filePathGen_([=](const size_t i) { return path + "_" + std::to_string(i) + "." + fileExt; }),
        framesPerBX_(framesPerBX),
        boardTMUX_(tmux),
        maxFramesPerFile_(maxFramesPerFile),
        maxEventsPerFile_(maxFramesPerFile_),
        eventIndex_(0),
        pendingEvents_(0),
        channelMap_(channelSpecs),
        staggerTmuxSlices_(staggerTmuxSlices) {
    if (channelMap_.empty())
      throw std::runtime_error("BoardDataWriter channel map cannnot be empty");
    if (fileExt != "txt" && fileExt != "txt.gz" && fileExt != "txt.xz")
      throw std::runtime_error("BoardDataWriter fileExt must be one of txt, txt.gz, txt.xz");

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

      const size_t maxEventsPerFileStaggered =
          ((maxFramesPerFile_ - spec.offset) / (framesPerBX_ * boardTMUX_)) - (tmuxRatio - 1);
      const size_t maxEventsPerFileUnstaggered =
          tmuxRatio * ((maxFramesPerFile_ - spec.offset) / (framesPerBX_ * boardTMUX_ * tmuxRatio));
      maxEventsPerFile_ =
          std::min(maxEventsPerFile_, staggerTmuxSlices ? maxEventsPerFileStaggered : maxEventsPerFileUnstaggered);
    }

    resetBoardData();
  }

  BoardDataWriter::BoardDataWriter(FileFormat format,
                                   const std::string& path,
                                   const std::string& fileExt,
                                   const size_t framesPerBX,
                                   const size_t tmux,
                                   const size_t maxFramesPerFile,
                                   const std::map<LinkId, std::vector<size_t>>& channelMap,
                                   const std::map<std::string, ChannelSpec>& channelSpecs,
                                   const bool staggerTmuxSlices)
      : BoardDataWriter(format,
                        path,
                        fileExt,
                        framesPerBX,
                        tmux,
                        maxFramesPerFile,
                        mergeMaps(channelMap, channelSpecs),
                        staggerTmuxSlices) {}

  BoardDataWriter::~BoardDataWriter() {
    // Print a warning if there are events that have been processed but not written to file.
    // Note we don't call flush() in the destructor because any exceptions would not be properly handled.
    // BoardDataWriter::flush() should be called in the endJob() function, e.g. GTTFileWriter::endJob().
    if (pendingEvents_ > 0) {
      std::cerr << "BoardDataWriter: WARNING: The last " << pendingEvents_
                << " events were not written to file. Please remember to flush() in the endJob() function of your file "
                   "writer."
                << std::endl;
      if (!fileNames_.empty()) {
        std::cerr << " The name of the last complete output buffer file was " << fileNames_.back() << std::endl;
      }
    }
  }

  void BoardDataWriter::setBoardDataFileID(const std::string& aId) { boardDataFileID_ = aId; }

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

    if (staggerTmuxSlices_) {
      for (const auto& [id, value] : channelMap_) {
        const auto& [spec, indices] = value;
        for (size_t tmuxIndex = 0; tmuxIndex < indices.size(); tmuxIndex++)
          boardData_.at(indices.at(tmuxIndex)).resize(tmuxIndex * boardTMUX_ * framesPerBX_ + spec.offset);
      }
    }

    pendingEvents_ = 0;
  }

  void BoardDataWriter::checkNumEventsPerFile(const std::vector<BoardDataWriter*>& fileWriters) {
    // Check that all given file writers have the same maxEventsPerFile_
    if (fileWriters.empty())
      return;

    // Create a vector of pointers to all unstaggered file writers
    std::vector<BoardDataWriter*> fileWritersUnstaggered;
    for (const auto& fileWriter : fileWriters) {
      if (!fileWriter->staggerTmuxSlices_) {
        fileWritersUnstaggered.push_back(fileWriter);
      }
    }

    const size_t maxEventsPerFile = fileWritersUnstaggered.empty() ? fileWriters.front()->maxEventsPerFile_
                                                                   : fileWritersUnstaggered.front()->maxEventsPerFile_;

    // Print a warning if a staggered file writer has a different maxEventsPerFile_ (only a warning because staggered file writers
    // are not expected to all have the same maxEventsPerFile_ when they don't share the same link:board TMUX ratio)
    for (const auto& fileWriter : fileWriters) {
      if (fileWriter->staggerTmuxSlices_ && fileWriter->maxEventsPerFile_ != maxEventsPerFile) {
        std::cerr << "\nBoardDataWriter: WARNING: A staggered BoardDataWriter has a different maxEventsPerFile_.\n"
                  << " The first file writer has maxEventsPerFile_ = " << maxEventsPerFile
                  << ", but a staggered file writer has maxEventsPerFile_ = " << fileWriter->maxEventsPerFile_
                  << ".\n This is expected only if they are using different link:board TMUX ratios"
                  << " (or if you're using a mixture of staggered and unstaggered file writers).\n"
                  << std::endl;
        break;
      }
    }

    // Throw an error if the maxEventsPerFile_ is not the same for all unstaggered file writers
    for (const auto& fileWriter : fileWritersUnstaggered) {
      if (fileWriter->maxEventsPerFile_ != maxEventsPerFile) {
        throw std::runtime_error(
            "BoardDataWriter: All unstaggered BoardDataWriters must have the same maxEventsPerFile_.\n"
            " The first file writer has maxEventsPerFile_ = " +
            std::to_string(maxEventsPerFile) +
            ", but another file writer has maxEventsPerFile_ = " + std::to_string(fileWriter->maxEventsPerFile_) +
            ".\n Please set the maxFramesPerFile parameter to use a multiple of the link TMUX (after accounting for "
            "any offset).");
      }
    }
  }

}  // namespace l1t::demo
