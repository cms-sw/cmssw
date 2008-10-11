#ifndef HcalCondObjectContainer_h
#define HcalCondObjectContainer_h


#include <iostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

template<class Item>
class HcalCondObjectContainer
{
 public:
  // default constructor
  HcalCondObjectContainer();

  // destructor:
  ~HcalCondObjectContainer();

  // get the object back
  const Item* getValues(DetId fId) const;

  // does the object exist ?
  const bool exists(DetId fId) const;

  // set the object/fill it in:
  bool addValues(const Item& myItem);

  // list of available channels:
  std::vector<DetId> getAllChannels() const;


 private:
  void initContainer(int container);

  std::vector<Item> HBcontainer;
  std::vector<Item> HEcontainer;
  std::vector<Item> HOcontainer;
  std::vector<Item> HFcontainer;
  std::vector<Item> HTcontainer;
  std::vector<Item> ZDCcontainer;
  std::vector<Item> CALIBcontainer;
  std::vector<Item> CASTORcontainer;
};


template<class Item>
HcalCondObjectContainer<Item>::HcalCondObjectContainer()
{
}

template<class Item>
HcalCondObjectContainer<Item>::~HcalCondObjectContainer()
{
}

template<class Item> void
HcalCondObjectContainer<Item>::initContainer(int container)
{
  Item emptyItem;

  switch (container) 
    {
    case HcalGenericDetId::HcalGenBarrel: for (int i=0; i<2592; i++) HBcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenEndcap: for (int i=0; i<2592; i++) HEcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenOuter: for (int i=0; i<2160; i++) HOcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenForward: for (int i=0; i<1728; i++) HFcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenTriggerTower: for (int i=0; i<4176; i++) HTcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenZDC: for (int i=0; i<22; i++) ZDCcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenCalibration: for (int i=0; i<1386; i++) CALIBcontainer.push_back(emptyItem); break;
    case HcalGenericDetId::HcalGenCastor: for (int i=0; i<1; i++) CASTORcontainer.push_back(emptyItem); break;
    default: break;
    }
}


template<class Item> const Item*
HcalCondObjectContainer<Item>::getValues(DetId fId) const
{
  HcalGenericDetId myId(fId);
  int index = myId.hashedId();
  //  std::cout << "::::: getting values at index " << index  << ", DetId " << myId << std::endl;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int

  const Item* cell = NULL;
  if (index >= 0)
    switch (myId.genericSubdet() ) {
    case HcalGenericDetId::HcalGenBarrel: 
      if (index1 < HBcontainer.size()) 
	cell = &(HBcontainer.at(index1) );  
      break;
    case HcalGenericDetId::HcalGenEndcap: 
      if (index1 < HEcontainer.size()) 
	cell = &(HEcontainer.at(index1) ); 
      break;
    case HcalGenericDetId::HcalGenOuter: 
      if (index1 < HOcontainer.size())
	cell = &(HOcontainer.at(index1) ); 
      break;
    case HcalGenericDetId::HcalGenForward:
      if (index1 < HFcontainer.size()) 
	cell = &(HFcontainer.at(index1) ); 
      break;
    case HcalGenericDetId::HcalGenTriggerTower: 
      if (index1 < HTcontainer.size()) 
	cell = &(HTcontainer.at(index1) ); 
      break;
    case HcalGenericDetId::HcalGenZDC:    
      if (index1 < ZDCcontainer.size()) 
	cell = &(ZDCcontainer.at(index1) ); 
      break;
    case HcalGenericDetId::HcalGenCastor:
      if (index1 < CASTORcontainer.size()) 
	cell = &(CASTORcontainer.at(index1) ); 
      break;
    case HcalGenericDetId::HcalGenCalibration:
      if (index1 < CALIBcontainer.size())
	cell = &(CALIBcontainer.at(index1) ); 
      break;
    default: break;
    }
  
  //  Item emptyItem;
  //  if (cell->rawId() == emptyItem.rawId() ) 
  if ((!cell) || (cell->rawId() != fId ) )
    throw cms::Exception ("Conditions not found") 
      << "Unavailable Conditions for cell " << myId;
  return cell;
}

template<class Item> const bool
HcalCondObjectContainer<Item>::exists(DetId fId) const
{
  HcalGenericDetId myId(fId);
  int index = myId.hashedId();
  if (index < 0) return false;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int
  const Item* cell = NULL;
  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenBarrel: 
    if (index1 < HBcontainer.size()) cell = &(HBcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenEndcap: 
    if (index1 < HEcontainer.size()) cell = &(HEcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenOuter: 
    if (index1 < HOcontainer.size()) cell = &(HOcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenForward: 
    if (index1 < HFcontainer.size()) cell = &(HFcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenTriggerTower: 
    if (index1 < HTcontainer.size()) cell = &(HTcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenZDC: 
    if (index1 < ZDCcontainer.size()) cell = &(ZDCcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenCastor: 
    if (index1 < CASTORcontainer.size()) cell = &(CASTORcontainer.at(index1) );  
    break;
  case HcalGenericDetId::HcalGenCalibration: 
    if (index1 < CALIBcontainer.size()) cell = &(CALIBcontainer.at(index1) );  
    break;
  default: return false; break;
  }
  
  //  Item emptyItem;
  if (cell)
    //    if (cell->rawId() != emptyItem.rawId() ) 
    if (cell->rawId() == fId ) 
      return true;

  return false;
}

template<class Item> bool
HcalCondObjectContainer<Item>::addValues(const Item& myItem)
{
  unsigned long myRawId = myItem.rawId();
  HcalGenericDetId myId(myRawId);
  int index = myId.hashedId();
  if (index < 0) return false;
  bool success = false;

  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenBarrel:
    if (!HBcontainer.size() ) initContainer(myId.genericSubdet() );
    HBcontainer.at(index)  = myItem;
    success = true;
    break;
  case HcalGenericDetId::HcalGenEndcap: 
    if (!HEcontainer.size() ) initContainer(myId.genericSubdet() );
    HEcontainer.at(index)  = myItem; 
    success = true;
    break;
  case HcalGenericDetId::HcalGenOuter:  
    if (!HOcontainer.size() ) initContainer(myId.genericSubdet() );
    HOcontainer.at(index)  = myItem; 
    success = true;
    break;
  case HcalGenericDetId::HcalGenForward: 
    if (!HFcontainer.size() ) initContainer(myId.genericSubdet() );
    HFcontainer.at(index)  = myItem;
    success = true;
    break;
  case HcalGenericDetId::HcalGenTriggerTower: 
    if (!HTcontainer.size() ) initContainer(myId.genericSubdet() );
    HTcontainer.at(index)  = myItem;
    success = true;
    break;
  case HcalGenericDetId::HcalGenZDC: 
    if (!ZDCcontainer.size() ) initContainer(myId.genericSubdet() );
    ZDCcontainer.at(index)  = myItem; 
    success = true;
    break;
  case HcalGenericDetId::HcalGenCastor: 
    if (!CASTORcontainer.size() ) initContainer(myId.genericSubdet() );
    CASTORcontainer.at(index)  = myItem; 
    success = true;
    break;
  case HcalGenericDetId::HcalGenCalibration: 
    if (!CALIBcontainer.size() ) initContainer(myId.genericSubdet() );
    CALIBcontainer.at(index)  = myItem;  
    success = true;
    break;
  default: break;
  }

  return success;
}

template<class Item> std::vector<DetId>
HcalCondObjectContainer<Item>::getAllChannels() const
{
  std::vector<DetId> channels;
  Item emptyItem;
  for (unsigned int i=0; i<HBcontainer.size(); i++)
    {
      if (emptyItem.rawId() != HBcontainer.at(i).rawId() )
	channels.push_back( DetId(HBcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HEcontainer.size(); i++)
    {
      if (emptyItem.rawId() != HEcontainer.at(i).rawId() )
	channels.push_back( DetId(HEcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HOcontainer.size(); i++)
    {
      if (emptyItem.rawId() != HOcontainer.at(i).rawId() )
	channels.push_back( DetId(HOcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HFcontainer.size(); i++)
    {
      if (emptyItem.rawId() != HFcontainer.at(i).rawId() )
	channels.push_back( DetId(HFcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HTcontainer.size(); i++)
    {
      if (emptyItem.rawId() != HTcontainer.at(i).rawId() )
	channels.push_back( DetId(HTcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<ZDCcontainer.size(); i++)
    {
      if (emptyItem.rawId() != ZDCcontainer.at(i).rawId() )
	channels.push_back( DetId(ZDCcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<CALIBcontainer.size(); i++)
    {
      if (emptyItem.rawId() != CALIBcontainer.at(i).rawId() )
	channels.push_back( DetId(CALIBcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<CASTORcontainer.size(); i++)
    {
      if (emptyItem.rawId() != CASTORcontainer.at(i).rawId() )
	channels.push_back( DetId(CASTORcontainer.at(i).rawId()) );
    }

  return channels;
}


#endif
