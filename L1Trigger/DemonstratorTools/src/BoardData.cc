
#include "L1Trigger/DemonstratorTools/interface/BoardData.h"

namespace l1t::demo {

  BoardData::BoardData() {}

  BoardData::BoardData(const std::string& name) : name_(name) {}

  BoardData::BoardData(const std::string& name, const std::vector<size_t>& channels, size_t length) : name_(name) {
    for (const auto i : channels)
      data_[i] = Channel(length);
  }

  const std::string& BoardData::name() const { return name_; }

  void BoardData::name(const std::string& aName) { name_ = aName; }

  std::map<size_t, BoardData::Channel>::const_iterator BoardData::begin() const { return data_.begin(); }

  std::map<size_t, BoardData::Channel>::iterator BoardData::begin() { return data_.begin(); }

  std::map<size_t, BoardData::Channel>::const_iterator BoardData::end() const { return data_.end(); }

  std::map<size_t, BoardData::Channel>::iterator BoardData::end() { return data_.end(); }

  BoardData::Channel& BoardData::add(size_t i) {
    data_[i] = Channel();
    return data_.at(i);
  }

  BoardData::Channel& BoardData::add(size_t i, const Channel& data) {
    data_[i] = data;
    return data_.at(i);
  }

  BoardData::Channel& BoardData::at(size_t i) { return data_.at(i); }

  const BoardData::Channel& BoardData::at(size_t i) const { return data_.at(i); }

  bool BoardData::has(size_t i) const { return data_.count(i) > 0; }

  size_t BoardData::size() { return data_.size(); }

}  // namespace l1t::demo