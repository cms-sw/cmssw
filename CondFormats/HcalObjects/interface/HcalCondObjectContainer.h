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
  void addValues(const Item& myItem);

  // list of available channels:
  std::vector<DetId> getAllChannels() const;


 private:
  std::vector<Item> HBcontainer;
  std::vector<Item> HEcontainer;
  std::vector<Item> HOcontainer;
  std::vector<Item> HFcontainer;
  //  std::vector<Item> HTcontainer;
  std::vector<Item> ZDCcontainer;
  std::vector<Item> CALIBcontainer;
  std::vector<Item> CASTORcontainer;
};


template<class Item>
HcalCondObjectContainer<Item>::HcalCondObjectContainer()
{
  Item emptyItem;
  for (int i=0; i<2592; i++)
    {
      HBcontainer.push_back(emptyItem);
    }
  for (int i=0; i<2592; i++)
    {
      HEcontainer.push_back(emptyItem);
    }
  for (int i=0; i<2160; i++)
    {
      HOcontainer.push_back(emptyItem);
    }
  for (int i=0; i<1728; i++)
    {
      HFcontainer.push_back(emptyItem);
    }
//  for (int i=0; i<4176; i++)
//    {
//      HTcontainer.push_back(emptyItem);
//    }
  for (int i=0; i<22; i++)
    {
      ZDCcontainer.push_back(emptyItem);
    }
  for (int i=0; i<1386; i++)
    {
      CALIBcontainer.push_back(emptyItem);
    }
  for (int i=0; i<1; i++)
    {
      CASTORcontainer.push_back(emptyItem);
    }

}

template<class Item>
HcalCondObjectContainer<Item>::~HcalCondObjectContainer()
{
}

template<class Item> const Item*
HcalCondObjectContainer<Item>::getValues(DetId fId) const
{
  Item emptyItem;
  const Item* cell;
  HcalGenericDetId myId(fId);
  int index = myId.hashedId();
  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenBarrel: cell = &(HBcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenEndcap: cell = &(HEcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenOuter: cell = &(HOcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenForward: cell = &(HFcontainer.at(index) );  break;
    //  case HcalGenericDetId::HcalGenTriggerTower: cell = &(HTcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenZDC: cell = &(ZDCcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenCastor: cell = &(CASTORcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenCalibration: cell = &(CALIBcontainer.at(index) );  break;
  default: cell = NULL; break;
  }
  
  if (cell->rawId() == emptyItem.rawId() ) 
    throw cms::Exception ("Conditions not found") 
      << "Unavailable Conditions for cell " << myId;
  return cell;
}

template<class Item> const bool
HcalCondObjectContainer<Item>::exists(DetId fId) const
{
  Item emptyItem;
  const Item* cell;
  HcalGenericDetId myId(fId);
  int index = myId.hashedId();
  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenBarrel: cell = &(HBcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenEndcap: cell = &(HEcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenOuter: cell = &(HOcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenForward: cell = &(HFcontainer.at(index) );  break;
    //  case HcalGenericDetId::HcalGenTriggerTower: cell = &(HTcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenZDC: cell = &(ZDCcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenCastor: cell = &(CASTORcontainer.at(index) );  break;
  case HcalGenericDetId::HcalGenCalibration: cell = &(CALIBcontainer.at(index) );  break;
  default: return false; break;
  }
  
  if (cell->rawId() == emptyItem.rawId() ) 
    return false;

  return true;
}

template<class Item> void
HcalCondObjectContainer<Item>::addValues(const Item& myItem)
{
  unsigned long myRawId = myItem.rawId();
  HcalGenericDetId myId(myRawId);
  int index = myId.hashedId();
  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenBarrel: HBcontainer.at(index)  = myItem;  break;
  case HcalGenericDetId::HcalGenEndcap: HEcontainer.at(index)  = myItem;  break;
  case HcalGenericDetId::HcalGenOuter:  HOcontainer.at(index)  = myItem;  break;
  case HcalGenericDetId::HcalGenForward: HFcontainer.at(index)  = myItem;  break;
    //  case HcalGenericDetId::HcalGenTriggerTower: HTcontainer.at(index)  = myItem;  break;
  case HcalGenericDetId::HcalGenZDC: ZDCcontainer.at(index)  = myItem;  break;
  case HcalGenericDetId::HcalGenCastor: CASTORcontainer.at(index)  = myItem;  break;
  case HcalGenericDetId::HcalGenCalibration: CALIBcontainer.at(index)  = myItem;  break;
  default: break;
  }


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
//  for (unsigned int i=0; i<HTcontainer.size(); i++)
//    {
//      if (emptyItem.rawId() != HTcontainer.at(i).rawId() )
//	channels.push_back( DetId(HTcontainer.at(i).rawId()) );
//    }
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
