
#ifndef L1Trigger_DemonstratorTools_BoardDataWriter_h
#define L1Trigger_DemonstratorTools_BoardDataWriter_h

#include <functional>

#include "L1Trigger/DemonstratorTools/interface/BoardData.h"
#include "L1Trigger/DemonstratorTools/interface/EventData.h"
#include "L1Trigger/DemonstratorTools/interface/ChannelSpec.h"
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"

namespace l1t::demo {

  // Writes I/O buffer files created from hardware/firmware tests, ensuring that
  // the data conforms to the declared packet structure (by inserting invalid
  // frames automatically), concatenating data from each event (accounting for
  // different TM periods of specific links, and of the data-processor itself),
  // and transparently switching to new output files when the data would overrun
  // the length of the board's I/O buffers
  class BoardDataWriter {
  public:
    // map of logical channel ID -> [TMUX period, interpacket-gap & offset; channel indices]
    typedef std::map<LinkId, std::pair<ChannelSpec, std::vector<size_t>>> ChannelMap_t;

    BoardDataWriter(FileFormat,
                    const std::string& filePath,
                    const std::string& fileExt,
                    const size_t framesPerBX,
                    const size_t tmux,
                    const size_t maxFramesPerFile,
                    const ChannelMap_t&);

    BoardDataWriter(FileFormat,
                    const std::string& filePath,
                    const std::string& fileExt,
                    const size_t framesPerBX,
                    const size_t tmux,
                    const size_t maxFramesPerFile,
                    const std::map<LinkId, std::vector<size_t>>&,
                    const std::map<std::string, ChannelSpec>&);

    // Set ID string that's written at start of board data files
    void setBoardDataFileID(const std::string&);

    void addEvent(const EventData& data);

    // If there are events that have not been written to file, forces creation of a board data file containing them
    void flush();

  private:
    static ChannelMap_t mergeMaps(const std::map<LinkId, std::vector<size_t>>&,
                                  const std::map<std::string, ChannelSpec>&);

    void resetBoardData();

    FileFormat fileFormat_;

    // ID string that's written at start of board data files
    std::string boardDataFileID_;

    std::function<std::string(const size_t)> filePathGen_;

    std::vector<std::string> fileNames_;

    size_t framesPerBX_;

    size_t boardTMUX_;

    size_t maxFramesPerFile_;

    size_t maxEventsPerFile_;

    size_t eventIndex_;

    // Number of events stored in boardData_
    size_t pendingEvents_;

    BoardData boardData_;

    // map of logical channel ID -> [TMUX period, interpacket-gap & offset; channel indices]
    ChannelMap_t channelMap_;
  };

}  // namespace l1t::demo

#endif