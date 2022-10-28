
#include "L1Trigger/DemonstratorTools/interface/EventData.h"

namespace l1t::demo {

  EventData::EventData() {}

  EventData::EventData(const std::map<LinkId, std::vector<ap_uint<64>>>& data) : data_(data) {}

  EventData::const_iterator EventData::begin() const { return data_.begin(); }

  EventData::const_iterator EventData::end() const { return data_.end(); }

  void EventData::add(const LinkId& i, const std::vector<ap_uint<64>>& data) { data_[i] = data; }

  void EventData::add(const EventData& data) {
    for (const auto& x : data)
      add(x.first, x.second);
  }

  const std::vector<ap_uint<64>>& EventData::at(const LinkId& i) const { return data_.at(i); }

  bool EventData::has(const LinkId& i) const { return data_.count(i) > 0; }

  size_t EventData::size() { return data_.size(); }

}  // namespace l1t::demo