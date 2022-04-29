
#ifndef L1Trigger_DemonstratorTools_EventData_h
#define L1Trigger_DemonstratorTools_EventData_h

#include <map>
#include <string>
#include <vector>

#include "ap_int.h"

#include "L1Trigger/DemonstratorTools/interface/LinkId.h"

namespace l1t::demo {

  /*!
   * \brief Class representing information phase-2 ATCA I/O data corresponding to a single event, 
   *   with logical channel IDs (essentially string-uint pairs, e.g. tracks-0 to tracks-17).
   *    
   *   This class is used to provide an event-index-independent interface to the BoardDataWriter &
   *   BoardDataReader classes - i.e. to avoid any need to keep track of  `eventIndex % tmux` when
   *   using that class for boards whose TMUX period is less than any of their upstream systems. 
   *   One logical channel ID corresponds to different I/O channel indices from one event to the
   *   next for the input channels of a board have a higher TMUX period than the board (e.g. for
   *   tracks sent to the correlator/GMT/GTT, or for the GMT, GTT and correlator links into GT); the
   *   mapping of logical channel IDs to I/O channel indices is implemented in the BoardDataWriter
   *   and BoardDataReader classes.
   */
  class EventData {
  public:
    typedef std::map<LinkId, std::vector<ap_uint<64>>>::const_iterator const_iterator;

    EventData();

    EventData(const std::map<LinkId, std::vector<ap_uint<64>>>&);

    const_iterator begin() const;

    const_iterator end() const;

    void add(const LinkId&, const std::vector<ap_uint<64>>&);

    void add(const EventData&);

    const std::vector<ap_uint<64>>& at(const LinkId&) const;

    bool has(const LinkId&) const;

    // Returns number of channels
    size_t size();

  private:
    // Map of channel IDs to data
    std::map<LinkId, std::vector<ap_uint<64>>> data_;
  };

}  // namespace l1t::demo

#endif