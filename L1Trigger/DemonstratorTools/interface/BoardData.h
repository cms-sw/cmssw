
#ifndef L1Trigger_DemonstratorTools_BoardData_h
#define L1Trigger_DemonstratorTools_BoardData_h

#include <map>
#include <vector>

#include "L1Trigger/DemonstratorTools/interface/Frame.h"

namespace l1t::demo {

  //! Class representing information that's stored in the input or output buffers on a phase-2 board
  class BoardData {
  public:
    typedef std::vector<Frame> Channel;

    BoardData();

    BoardData(const std::string& name);

    BoardData(const std::string& name, const std::vector<size_t>& channels, size_t length);

    const std::string& name() const;

    void name(const std::string& aName);

    std::map<size_t, Channel>::const_iterator begin() const;

    std::map<size_t, Channel>::iterator begin();

    std::map<size_t, Channel>::const_iterator end() const;

    std::map<size_t, Channel>::iterator end();

    Channel& add(size_t);

    Channel& add(size_t, const Channel&);

    Channel& at(size_t);

    const Channel& at(size_t) const;

    bool has(size_t) const;

    // Returns number of channels
    size_t size();

  private:
    std::string name_;

    // Map of channel indices to data
    std::map<size_t, Channel> data_;
  };

}  // namespace l1t::demo

#endif