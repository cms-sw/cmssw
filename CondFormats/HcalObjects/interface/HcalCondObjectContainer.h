#ifndef HcalCondObjectContainer_h
#define HcalCondObjectContainer_h


#include <iostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <cstdlib>

template<class Item>
class HcalCondObjectContainer
{
 public:
  // default constructor
  HcalCondObjectContainer();

  // destructor:
  virtual ~HcalCondObjectContainer();

  // get the object back
  const Item* getValues(DetId fId) const;

  // does the object exist ?
  const bool exists(DetId fId) const;

  // set the object/fill it in:
  bool addValues(const Item& myItem);

  // list of available channels:
  std::vector<DetId> getAllChannels() const;

  virtual std::string myname() const {return (std::string)"Hcal Undefined";}

  // setting types for easier work for getAllContainers()
  typedef std::pair< std::string, std::vector<Item> > tHcalCont;
  typedef std::vector< tHcalCont > tAllContWithNames;

  const tAllContWithNames getAllContainers() const{
    tAllContWithNames allContainers;
    allContainers.push_back(tHcalCont("HB",HBcontainer));
    allContainers.push_back(tHcalCont("HE",HEcontainer));
    allContainers.push_back(tHcalCont("HO",HOcontainer));
    allContainers.push_back(tHcalCont("HF",HFcontainer));
    allContainers.push_back(tHcalCont("HT",HTcontainer));
    allContainers.push_back(tHcalCont("ZDC",ZDCcontainer));
    allContainers.push_back(tHcalCont("CALIB",CALIBcontainer));
    allContainers.push_back(tHcalCont("CASTOR",CASTORcontainer));
    return allContainers;
  }

  // set slow mode
  void setSlowMode(bool slow) { slowMode_=slow; }	

private:
  void initContainer(int container);

  //  bool m_h2mode;

  std::vector<Item> HBcontainer;
  std::vector<Item> HEcontainer;
  std::vector<Item> HOcontainer;
  std::vector<Item> HFcontainer;
  std::vector<Item> HTcontainer;
  std::vector<Item> ZDCcontainer;
  std::vector<Item> CALIBcontainer;
  std::vector<Item> CASTORcontainer;
  std::map<uint32_t,Item> SLOWcontainer;
  bool slowMode_;
};


template<class Item>
HcalCondObjectContainer<Item>::HcalCondObjectContainer(): slowMode_(false)
//: m_h2mode(false)
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
  const Item* cell = NULL;
  if (slowMode_) {
    typename std::map<uint32_t,Item>::const_iterator i=SLOWcontainer.find(fId.rawId());
    if (i!=SLOWcontainer.end()) cell=&(i->second);	
  } else {

    int index = myId.hashedId();
    //  std::cout << "::::: getting values at index " << index  << ", DetId " << myId << std::endl;
    unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int

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
  }
  
  //  Item emptyItem;
  //  if (cell->rawId() == emptyItem.rawId() ) 
  if ((!cell) || (cell->rawId() != fId ) )
    throw cms::Exception ("Conditions not found") 
      << "Unavailable Conditions of type " << myname() << " for cell " << myId;
  return cell;
}

template<class Item> const bool
HcalCondObjectContainer<Item>::exists(DetId fId) const
{
  HcalGenericDetId myId(fId);
  if (slowMode_) {
    typename std::map<uint32_t,Item>::const_iterator i;
    i=SLOWcontainer.find(fId.rawId());
    return (i!=SLOWcontainer.end());
  } else {
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
      if (index1 < CASTORcontainer.size()) cell = &(CASTORcontainer.at(index1));
      break;
    case HcalGenericDetId::HcalGenCalibration: 
      if (index1 < CALIBcontainer.size()) cell = &(CALIBcontainer.at(index1));
      break;
    default: return false; break;
    }
  
    //  Item emptyItem;
    if (cell)
      //    if (cell->rawId() != emptyItem.rawId() ) 
      if (cell->rawId() == fId ) 
	return true;
  }
  return false;
}

template<class Item> bool
HcalCondObjectContainer<Item>::addValues(const Item& myItem)
{
  unsigned long myRawId = myItem.rawId();
  HcalGenericDetId myId(myRawId);
  int index = myId.hashedId();
  if (slowMode_) {
    SLOWcontainer.insert(typename std::pair<uint32_t,Item>(myItem.rawId(),myItem));
    //    std::cout << "Added for " << myId << std::endl;
    return true;
  } else {
    bool success = false;
    if (index < 0) success = false;
    unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int

    switch (myId.genericSubdet() ) {
    case HcalGenericDetId::HcalGenBarrel:
      if (!HBcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < HBcontainer.size())
	{
	  HBcontainer.at(index1)  = myItem;
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenEndcap: 
      if (!HEcontainer.size() ) initContainer(myId.genericSubdet());
      if (index1 < HEcontainer.size())
	{
	  HEcontainer.at(index1)  = myItem; 
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenOuter:  
      if (!HOcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < HOcontainer.size())
	{
	  HOcontainer.at(index1)  = myItem; 
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenForward: 
      if (!HFcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < HFcontainer.size())
	{
	  HFcontainer.at(index1)  = myItem;
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenTriggerTower: 
      if (!HTcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < HTcontainer.size())
	{
	  HTcontainer.at(index1)  = myItem;
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenZDC: 
      if (!ZDCcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < ZDCcontainer.size())
	{
	  ZDCcontainer.at(index1)  = myItem; 
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenCastor: 
      if (!CASTORcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < CASTORcontainer.size())
	{
	  CASTORcontainer.at(index1)  = myItem; 
	  success = true;
	}
      break;
    case HcalGenericDetId::HcalGenCalibration: 
      if (!CALIBcontainer.size() ) initContainer(myId.genericSubdet() );
      if (index1 < CALIBcontainer.size())
	{
	  CALIBcontainer.at(index1)  = myItem;  
	  success = true;
	}
      break;
    default: break;
    }

    if (!success) 
      throw cms::Exception ("Filling of conditions failed") 
	<< " no valid filling possible for Conditions of type " << myname() << " for DetId " << myId;
    return success;
  }
}

template<class Item> std::vector<DetId>
HcalCondObjectContainer<Item>::getAllChannels() const
{
  std::vector<DetId> channels;
	
  if (slowMode_) {
    typename std::map<uint32_t,Item>::const_iterator i;
    for (i=SLOWcontainer.begin(); i!=SLOWcontainer.end(); i++)
      channels.push_back(i->first);    
  } else {
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
  }
  return channels;
}


#endif
