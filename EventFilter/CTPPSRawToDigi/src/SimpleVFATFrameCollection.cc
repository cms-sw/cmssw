/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com) 
*    
****************************************************************************/


#include "EventFilter/CTPPSRawToDigi/interface/SimpleVFATFrameCollection.h"

//----------------------------------------------------------------------------------------------------

using namespace std;

SimpleVFATFrameCollection::SimpleVFATFrameCollection()
{
}

//----------------------------------------------------------------------------------------------------

SimpleVFATFrameCollection::~SimpleVFATFrameCollection()
{
  data.clear();
}

//----------------------------------------------------------------------------------------------------

const VFATFrame* SimpleVFATFrameCollection::GetFrameByID(unsigned int ID) const
{
  // first convert ID to 12bit form
  ID = ID & 0xFFF;

  for (MapType::const_iterator it = data.begin(); it != data.end(); ++it)
    if (it->second.getChipID() == ID)
      if (it->second.checkFootprint() && it->second.checkCRC())
        return &(it->second);

  return NULL;
}

//----------------------------------------------------------------------------------------------------

const VFATFrame* SimpleVFATFrameCollection::GetFrameByIndex(TotemFramePosition index) const
{
  MapType::const_iterator it = data.find(index);
  if (it != data.end())
    return &(it->second);
  else
    return NULL;
}

//----------------------------------------------------------------------------------------------------

VFATFrameCollection::value_type SimpleVFATFrameCollection::BeginIterator() const
{
  MapType::const_iterator it = data.begin();
  return (it == data.end()) ? value_type(TotemFramePosition(), NULL) : value_type(it->first, &it->second);
}

//----------------------------------------------------------------------------------------------------

VFATFrameCollection::value_type SimpleVFATFrameCollection::NextIterator(const value_type &value) const
{
  if (!value.second)
    return value;

  MapType::const_iterator it = data.find(value.first);
  it++;

  return (it == data.end()) ? value_type(TotemFramePosition(), NULL) : value_type(it->first, &it->second);
}

//----------------------------------------------------------------------------------------------------

bool SimpleVFATFrameCollection::IsEndIterator(const value_type &value) const
{
  return (value.second == NULL);
}
