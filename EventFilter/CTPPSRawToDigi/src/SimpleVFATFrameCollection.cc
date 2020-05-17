/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/

#include "EventFilter/CTPPSRawToDigi/interface/SimpleVFATFrameCollection.h"

using namespace std;

SimpleVFATFrameCollection::SimpleVFATFrameCollection() {}

SimpleVFATFrameCollection::~SimpleVFATFrameCollection() { data.clear(); }

const VFATFrame* SimpleVFATFrameCollection::GetFrameByID(unsigned int ID) const {
  // first convert ID to 12bit form
  ID = ID & 0xFFF;

  for (const auto& it : data)
    if (it.second.getChipID() == ID)
      if (it.second.checkFootprint() && it.second.checkCRC())
        return &(it.second);

  return nullptr;
}

const VFATFrame* SimpleVFATFrameCollection::GetFrameByIndex(TotemFramePosition index) const {
  auto it = data.find(index);
  if (it != data.end())
    return &(it->second);
  else
    return nullptr;
}

VFATFrameCollection::value_type SimpleVFATFrameCollection::BeginIterator() const {
  auto it = data.begin();
  return (it == data.end()) ? value_type(TotemFramePosition(), nullptr) : value_type(it->first, &it->second);
}

VFATFrameCollection::value_type SimpleVFATFrameCollection::NextIterator(const value_type& value) const {
  if (!value.second)
    return value;

  auto it = data.find(value.first);
  it++;

  return (it == data.end()) ? value_type(TotemFramePosition(), nullptr) : value_type(it->first, &it->second);
}

bool SimpleVFATFrameCollection::IsEndIterator(const value_type& value) const { return (value.second == nullptr); }
