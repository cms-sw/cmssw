#ifndef CastorCondObjectContainer_h
#define CastorCondObjectContainer_h
//
//Adapted for CASTOR by L. Mundim
//
#include <iostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
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
  const Item* getValues(DetId fId) const;

  // does the object exist ?
  const bool exists(DetId fId) const;

  // set the object/fill it in:
  bool addValues(const Item& myItem);
  //bool addValues(const Item& myItem, bool h2mode_=false);

  // list of available channels:
  std::vector<DetId> getAllChannels() const;

  std::string myname() const {return (std::string)"Castor Undefined";}

 private:
  //void initContainer(int container, bool h2mode_ = false);
  void initContainer(int container);

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
//CastorCondObjectContainer<Item>::initContainer(int container, bool h2mode_)
CastorCondObjectContainer<Item>::initContainer(int container)
{
  //m_h2mode = h2mode_;

  Item emptyItem;

  switch (container) 
    {
    case HcalGenericDetId::HcalGenCastor:
      for (int i=0; i<(2*HcalGenericDetId::CASTORhalf); i++) CASTORcontainer.push_back(emptyItem); break;
    default: break;
    }
}


template<class Item> const Item*
CastorCondObjectContainer<Item>::getValues(DetId fId) const
{
  HcalGenericDetId myId(fId);
  //int index = myId.hashedId(m_h2mode);
  int index = myId.hashedId();
  //  std::cout << "::::: getting values at index " << index  << ", DetId " << myId << std::endl;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int

  const Item* cell = NULL;
  if (index >= 0)
    switch (myId.genericSubdet() ) {
    case HcalGenericDetId::HcalGenCastor:
      if (index1 < CASTORcontainer.size()) 
	cell = &(CASTORcontainer.at(index1) ); 
      break;
    default: break;
    }
  
  //  Item emptyItem;
  //  if (cell->rawId() == emptyItem.rawId() ) 
  if ((!cell) || (cell->rawId() != fId ) )
    throw cms::Exception ("Conditions not found") 
      << "Unavailable Conditions of type " << myname() << " for cell " << myId;
  return cell;
}

template<class Item> const bool
CastorCondObjectContainer<Item>::exists(DetId fId) const
{
  HcalGenericDetId myId(fId);
  //int index = myId.hashedId(m_h2mode);
  int index = myId.hashedId();
  if (index < 0) return false;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int
  const Item* cell = NULL;
  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenCastor: 
    if (index1 < CASTORcontainer.size()) cell = &(CASTORcontainer.at(index1) );  
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
//CastorCondObjectContainer<Item>::addValues(const Item& myItem, bool h2mode_)
CastorCondObjectContainer<Item>::addValues(const Item& myItem)
{
  unsigned long myRawId = myItem.rawId();
  HcalGenericDetId myId(myRawId);
  //int index = myId.hashedId(h2mode_);
  int index = myId.hashedId();
  bool success = false;
  if (index < 0) success = false;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int


  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenCastor: 
    if (!CASTORcontainer.size() ) initContainer(myId.genericSubdet() );
    if (index1 < CASTORcontainer.size())
      {
	CASTORcontainer.at(index1)  = myItem; 
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


#endif
