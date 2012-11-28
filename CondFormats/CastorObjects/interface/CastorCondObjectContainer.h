#ifndef CastorCondObjectContainer_h
#define CastorCondObjectContainer_h
//
//Adapted for CASTOR by L. Mundim
//
#include <iostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cstdlib>

template<class Item>
class CastorCondObjectContainer
{
 public:
  // default constructor
  CastorCondObjectContainer();

  // destructor:
  ~CastorCondObjectContainer();

  // get the object back
  const Item* getValues(DetId fId, bool throwOnFail=true) const;

  // does the object exist ?
  const bool exists(DetId fId) const;

  // set the object/fill it in:
  bool addValues(const Item& myItem);
  //bool addValues(const Item& myItem, bool h2mode_=false);

  // list of available channels:
  std::vector<DetId> getAllChannels() const;

  std::string myname() const {return (std::string)"Castor Undefined";}

 private:
  void initContainer();
  unsigned int hashed_id(DetId fId) const;

  std::vector<Item> CASTORcontainer;
};


template<class Item>
//CastorCondObjectContainer<Item>::CastorCondObjectContainer(): m_h2mode(false)
CastorCondObjectContainer<Item>::CastorCondObjectContainer()
{
}

template<class Item>
CastorCondObjectContainer<Item>::~CastorCondObjectContainer()
{
}

template<class Item> void
CastorCondObjectContainer<Item>::initContainer()
{
  Item emptyItem;

  if (CASTORcontainer.empty())
    for (int i=0; i<HcalCastorDetId:: kSizeForDenseIndexing; i++)
      CASTORcontainer.push_back(emptyItem);
  
}


template<class Item> const Item*
CastorCondObjectContainer<Item>::getValues(DetId fId, bool throwOnFail) const
{
  const Item* cell = NULL;
  HcalCastorDetId myId(fId);

  if (fId.det()==DetId::Calo && fId.subdetId()==HcalCastorDetId::SubdetectorId) {
    unsigned int index = hashed_id(fId);
    
    if (index < CASTORcontainer.size()) 
	cell = &(CASTORcontainer.at(index) ); 
  }

 
  if ((!cell) || (cell->rawId() != fId ) ) {
    if (throwOnFail) {
      throw cms::Exception ("Conditions not found") 
	<< "Unavailable Conditions of type " << myname() << " for cell " << myId;
    } else {
      cell=0;
    }
  }
  return cell;
}

template<class Item> const bool
CastorCondObjectContainer<Item>::exists(DetId fId) const
{
  const Item* cell = getValues(fId,false);
  if (cell)
    //    if (cell->rawId() != emptyItem.rawId() ) 
    if (cell->rawId() == fId ) 
      return true;
  return false;
}

template<class Item> bool
CastorCondObjectContainer<Item>::addValues(const Item& myItem)
{
  unsigned long myRawId = myItem.rawId();
  HcalCastorDetId myId(myRawId);
  unsigned int index = hashed_id(myId);
  bool success = false;


  if (CASTORcontainer.empty() ) initContainer();
  if (index < CASTORcontainer.size())
    {
      CASTORcontainer.at(index)  = myItem; 
      success = true;
    }
  

  if (!success) 
    throw cms::Exception ("Filling of conditions failed") 
      << " no valid filling possible for Conditions of type " << myname() << " for DetId " << myId;
  
  return success;
}

template<class Item> std::vector<DetId>
CastorCondObjectContainer<Item>::getAllChannels() const
{
  std::vector<DetId> channels;
  Item emptyItem;
  for (unsigned int i=0; i<CASTORcontainer.size(); i++)
    {
      if (emptyItem.rawId() != CASTORcontainer.at(i).rawId() )
	channels.push_back( DetId(CASTORcontainer.at(i).rawId()) );
    }

  return channels;
}


template<class Item>
unsigned int CastorCondObjectContainer<Item>::hashed_id(DetId fId) const {
  // the historical packing from HcalGeneric is different from HcalCastorDetId, so we clone the old packing here.
  HcalCastorDetId tid(fId); 
  int zside = tid.zside();
  int sector = tid.sector();
  int module = tid.module(); 
  static const int CASTORhalf=224;
  
  int index = 14*(sector-1) + (module-1);
  if (zside == 1) index += CASTORhalf;
  
  return index;
}

#endif
