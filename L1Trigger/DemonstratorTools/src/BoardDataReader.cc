#include "L1Trigger/DemonstratorTools/interface/BoardDataReader.h"

#include <fstream>

#include "L1Trigger/DemonstratorTools/interface/Frame.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

namespace l1t::demo {

  BoardDataReader::BoardDataReader(FileFormat format,
                                   const std::vector<std::string>& fileNames,
                                   const size_t framesPerBX,
                                   const size_t tmux,
                                   const size_t emptyFramesAtStart,
                                   const ChannelMap_t& channelMap)
      : fileFormat_(format),
        fileNames_(fileNames),
        framesPerBX_(framesPerBX),
        boardTMUX_(tmux),
        emptyFramesAtStart_(emptyFramesAtStart),
        channelMap_(channelMap),
        events_() {
    // TODO (long term): Move much of this to separate function, and only read files on demand

    // Verify that channel map/spec is self-consistent
    for (const auto& [id, value] : channelMap_) {
      const auto& [spec, indices] = value;
      if ((spec.tmux % boardTMUX_) != 0)
        throw std::runtime_error("Link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "]: Specified TMUX period, " + std::to_string(spec.tmux) +
                                 ", is not a multiple of the board TMUX, " + std::to_string(boardTMUX_));

      const size_t tmuxRatio(spec.tmux / boardTMUX_);
      if (indices.size() != tmuxRatio)
        throw std::runtime_error("Link [" + id.interface + ", " + std::to_string(id.channel) +
                                 "]: Number of channel indices specified, " + std::to_string(indices.size()) +
                                 ", does not match link:board TMUX ratio");
    }

    // Loop over input files
    for (const auto& path : fileNames_) {
      BoardData boardData(read(path, fileFormat_));

      // 1) Verify that all expected channels are present
      for (const auto& [id, value] : channelMap_) {
        const auto& [spec, indices] = value;
        for (const auto i : indices) {
          if (not boardData.has(i))
            throw std::runtime_error("Channel " + std::to_string(i) + " was declared but is missing from file '" +
                                     path + "'");
        }
      }

      // 2) Verify that packet structure is as expected
      for (const auto& [id, value] : channelMap) {
        const auto& [spec, indices] = value;
        for (size_t tmuxIndex = 0; tmuxIndex < indices.size(); tmuxIndex++) {
          const auto& chanData = boardData.at(indices.at(tmuxIndex));

          const size_t framesBeforeFirstPacket(emptyFramesAtStart_ + tmuxIndex * boardTMUX_ * framesPerBX_ +
                                               spec.offset);
          const size_t eventLength(spec.tmux * framesPerBX_);
          const size_t packetLength(eventLength - spec.interpacketGap);

          for (size_t j = 0; j < framesBeforeFirstPacket; j++) {
            if (chanData.at(j).valid)
              throw std::runtime_error("Frame " + std::to_string(j) + " on channel " +
                                       std::to_string(indices.at(tmuxIndex)) + " is valid, but first " +
                                       std::to_string(framesBeforeFirstPacket) + "frames should be invalid");
          }

          for (size_t j = framesBeforeFirstPacket; j < chanData.size(); j++) {
            if ((j + (framesPerBX_ * spec.tmux)) >= chanData.size())
              continue;

            bool expectValid(((j - framesBeforeFirstPacket) % eventLength) < packetLength);

            if (expectValid) {
              if (not chanData.at(j).valid)
                throw std::runtime_error("Frame " + std::to_string(j) + " on channel " +
                                         std::to_string(indices.at(tmuxIndex)) +
                                         " is invalid, but expected valid frame");
            } else if (chanData.at(j).valid)
              throw std::runtime_error("Frame " + std::to_string(j) + " on channel " +
                                       std::to_string(indices.at(tmuxIndex)) + " is valid, but expected invalid frame");
          }
        }
      }

      // 3) Extract the data for each event
      bool eventIncomplete(false);
      for (size_t eventIndex = 0;; eventIndex++) {
        EventData eventData;

        for (const auto& [id, value] : channelMap) {
          const auto& [spec, indices] = value;
          const auto& chanData = boardData.at(indices.at(eventIndex % (spec.tmux / boardTMUX_)));

          // Extract the frames for this event
          const size_t framesBeforeEvent(eventIndex * boardTMUX_ * framesPerBX_ + emptyFramesAtStart_ + spec.offset);
          const size_t packetLength(spec.tmux * framesPerBX_ - spec.interpacketGap);

          if (chanData.size() < (framesBeforeEvent + spec.tmux * framesPerBX_)) {
            eventIncomplete = true;
            break;
          }

          std::vector<ap_uint<64>> chanEventData(packetLength);
          for (size_t j = 0; j < packetLength; j++)
            chanEventData.at(j) = chanData.at(framesBeforeEvent + j).data;
          eventData.add(id, chanEventData);
        }

        if (eventIncomplete)
          break;

        events_.push_back(eventData);
      }
    }

    eventIt_ = events_.begin();
  }

  BoardDataReader::BoardDataReader(FileFormat format,
                                   const std::vector<std::string>& fileNames,
                                   const size_t framesPerBX,
                                   const size_t tmux,
                                   const size_t emptyFramesAtStart,
                                   const std::map<LinkId, std::vector<size_t>>& channelMap,
                                   const std::map<std::string, ChannelSpec>& channelSpecs)
      : BoardDataReader(format, fileNames, framesPerBX, tmux, emptyFramesAtStart, mergeMaps(channelMap, channelSpecs)) {
  }

  EventData BoardDataReader::getNextEvent() {
    if (eventIt_ == events_.end())
      throw std::runtime_error("Board data reader ran out of events");

    return *(eventIt_++);
  }

  BoardDataReader::ChannelMap_t BoardDataReader::mergeMaps(const std::map<LinkId, std::vector<size_t>>& indexMap,
                                                           const std::map<std::string, ChannelSpec>& specMap) {
    ChannelMap_t channelMap;
    for (const auto& x : indexMap)
      channelMap.at(x.first) = {specMap.at(x.first.interface), x.second};
    return channelMap;
  }

}  // namespace l1t::demo
