
#ifndef L1Trigger_DemonstratorTools_BoardDataReader_h
#define L1Trigger_DemonstratorTools_BoardDataReader_h

#include <map>
#include <string>
#include <vector>

#include "L1Trigger/DemonstratorTools/interface/BoardData.h"
#include "L1Trigger/DemonstratorTools/interface/ChannelSpec.h"
#include "L1Trigger/DemonstratorTools/interface/EventData.h"
#include "L1Trigger/DemonstratorTools/interface/FileFormat.h"

namespace l1t::demo {

  // Reads I/O buffer files created from hardware/firmware tests, verifying that
  // received packets conform to the declared structure, separating out each
  // event (accounting for different TM periods of specific links and of the
  // data-processor itself), and transparently switching to data from new buffer
  // files as needed
  class BoardDataReader {
  public:
    // map of logical channel ID -> [TMUX period, interpacket-gap & offset; channel indices]
    typedef std::map<LinkId, std::pair<ChannelSpec, std::vector<size_t>>> ChannelMap_t;

    BoardDataReader(FileFormat,
                    const std::vector<std::string>&,
                    const size_t framesPerBX,
                    const size_t tmux,
                    const size_t emptyFramesAtStart,
                    const ChannelMap_t&);

    BoardDataReader(FileFormat,
                    const std::vector<std::string>&,
                    const size_t framesPerBX,
                    const size_t tmux,
                    const size_t emptyFramesAtStart,
                    const std::map<LinkId, std::vector<size_t>>&,
                    const std::map<std::string, ChannelSpec>&);

    EventData getNextEvent();

  private:
    static ChannelMap_t mergeMaps(const std::map<LinkId, std::vector<size_t>>&,
                                  const std::map<std::string, ChannelSpec>&);

    FileFormat fileFormat_;

    std::vector<std::string> fileNames_;

    size_t framesPerBX_;

    size_t boardTMUX_;

    size_t emptyFramesAtStart_;

    // map of logical channel ID -> [TMUX period, interpacket-gap & offset; channel indices]
    ChannelMap_t channelMap_;

    std::vector<EventData> events_;

    std::vector<EventData>::const_iterator eventIt_;
  };

}  // namespace l1t::demo

#endif