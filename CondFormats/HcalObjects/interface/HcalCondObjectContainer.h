#ifndef HcalCondObjectContainer_h
#define HcalCondObjectContainer_h

#include <vector>
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

class HcalTopology;

//#define HCAL_COND_SUPPRESS_DEFAULT

class HcalCondObjectContainerBase {
public:
  const HcalTopology* topo() const { return topo_; }
  int getCreatorPackedIndexVersion() const { return packedIndexVersion_; }
  void setTopo(const HcalTopology* topo) const;
  void setTopo(const HcalTopology* topo);
protected:
  HcalCondObjectContainerBase(const HcalTopology*);
  unsigned int indexFor(DetId) const;
  unsigned int sizeFor(DetId) const;
  int packedIndexVersion_;
  inline HcalOtherSubdetector extractOther(const DetId& id) const { return HcalOtherSubdetector((id.rawId()>>20)&0x1F); }
  std::string textForId(const DetId& id) const;
private:
  mutable const HcalTopology* topo_;
};

template<class Item>
class HcalCondObjectContainer : public HcalCondObjectContainerBase {
public:
  // default constructor
  HcalCondObjectContainer(const HcalTopology* topo) : HcalCondObjectContainerBase(topo) { }

  // destructor:
  virtual ~HcalCondObjectContainer();

  // get the object back
  const Item* getValues(DetId fId, bool throwOnFail=true) const;

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

 private:
  void initContainer(DetId container);

  std::vector<Item> HBcontainer;
  std::vector<Item> HEcontainer;
  std::vector<Item> HOcontainer;
  std::vector<Item> HFcontainer;
  std::vector<Item> HTcontainer;
  std::vector<Item> ZDCcontainer;
  std::vector<Item> CALIBcontainer;
  std::vector<Item> CASTORcontainer;
  
  //volatile const HcalTopology* topo_; // This needs to not be in the DB


};


template<class Item>
HcalCondObjectContainer<Item>::~HcalCondObjectContainer()
{
}

template<class Item> void
HcalCondObjectContainer<Item>::initContainer(DetId fId)
{
  Item emptyItem;

  if (fId.det()==DetId::Hcal) {
    switch (HcalSubdetector(fId.subdetId())) {
    case(HcalBarrel) : for (unsigned int i=0; i<sizeFor(fId); i++) HBcontainer.push_back(emptyItem); break;
    case(HcalEndcap) : for (unsigned int i=0; i<sizeFor(fId); i++) HEcontainer.push_back(emptyItem); break;
    case(HcalOuter) : for (unsigned int i=0; i<sizeFor(fId); i++) HOcontainer.push_back(emptyItem); break;
    case(HcalForward) : for (unsigned int i=0; i<sizeFor(fId); i++) HFcontainer.push_back(emptyItem); break;
    case(HcalTriggerTower) : for (unsigned int i=0; i<sizeFor(fId); i++) HTcontainer.push_back(emptyItem); break;
    case(HcalOther) : if (extractOther(fId)==HcalCalibration) 
	for (unsigned int i=0; i<sizeFor(fId); i++) CALIBcontainer.push_back(emptyItem); break;
      break; 
    default: break;
    }
  } else if (fId.det()==DetId::Calo) {
    if (fId.subdetId()==HcalCastorDetId::SubdetectorId) {
      for (unsigned int i=0; i<sizeFor(fId); i++) CASTORcontainer.push_back(emptyItem); 
    } else if (fId.subdetId()==HcalZDCDetId::SubdetectorId) {
      for (unsigned int i=0; i<sizeFor(fId); i++) ZDCcontainer.push_back(emptyItem); 
    }
  }
}


template<class Item> const Item*
HcalCondObjectContainer<Item>::getValues(DetId fId, bool throwOnFail) const
{
  unsigned int index=indexFor(fId);
  
  const Item* cell = NULL;

  if (index<0xFFFFFFFu) {
    if (fId.det()==DetId::Hcal) {
      switch (HcalSubdetector(fId.subdetId())) {
      case(HcalBarrel) : if (index < HBcontainer.size()) cell = &(HBcontainer.at(index) );  break;
      case(HcalEndcap) : if (index < HEcontainer.size()) cell = &(HEcontainer.at(index) );  break;
      case(HcalForward) : if (index < HFcontainer.size()) cell = &(HFcontainer.at(index) );   break; 
      case(HcalOuter) : if (index < HOcontainer.size()) cell = &(HOcontainer.at(index) );    break;
      case(HcalTriggerTower) : if (index < HTcontainer.size()) cell = &(HTcontainer.at(index) );    break;
      case(HcalOther) : if (extractOther(fId)==HcalCalibration) 
	  if (index < CALIBcontainer.size()) cell = &(CALIBcontainer.at(index) );  
	break; 
      default: break;
      }
    } else if (fId.det()==DetId::Calo) {
      if (fId.subdetId()==HcalCastorDetId::SubdetectorId) {
	if (index < CASTORcontainer.size()) cell = &(CASTORcontainer.at(index) );
      } else if (fId.subdetId()==HcalZDCDetId::SubdetectorId) {
	if (index < ZDCcontainer.size()) cell = &(ZDCcontainer.at(index) );
      }
    }
  }
  
  //  Item emptyItem;
  //  if (cell->rawId() == emptyItem.rawId() ) 
  if ((!cell)) {
    if (throwOnFail) {
      throw cms::Exception ("Conditions not found") 
	<< "Unavailable Conditions of type " << myname() << " for cell " << textForId(fId);
    } 
  } else if (cell->rawId() != fId) {
    if (throwOnFail) {
      throw cms::Exception ("Conditions mismatch") 
	<< "Requested conditions of type " << myname() << " for cell " << textForId(fId) << " got conditions for cell " << textForId(DetId(cell->rawId()));
    } 
    cell=0;
  }

  return cell;
}

template<class Item> const bool
HcalCondObjectContainer<Item>::exists(DetId fId) const
{
  const Item* cell = getValues(fId,false);

  if (cell)
    if (cell->rawId() == fId ) 
      return true;
  
  return false;
}

template<class Item> bool
HcalCondObjectContainer<Item>::addValues(const Item& myItem)
{
  bool success = false;
  DetId fId(myItem.rawId());
  unsigned int index=indexFor(fId);
  
  Item* cell = NULL;

  if (index<0xFFFFFFFu) {
    if (fId.det()==DetId::Hcal) {
      switch (HcalSubdetector(fId.subdetId())) {
      case(HcalBarrel) : if (!HBcontainer.size() ) initContainer(fId);
      	if (index < HBcontainer.size()) cell = &(HBcontainer.at(index) );  break;
      case(HcalEndcap) : if (!HEcontainer.size() ) initContainer(fId);
      	if (index < HEcontainer.size()) cell = &(HEcontainer.at(index) );  break;
      case(HcalForward) : if (!HFcontainer.size() ) initContainer(fId);
      	if (index < HFcontainer.size()) cell = &(HFcontainer.at(index) );  break;
      case(HcalOuter) : if (!HOcontainer.size() ) initContainer(fId);
      	if (index < HOcontainer.size()) cell = &(HOcontainer.at(index) );  break;
      case(HcalTriggerTower) : if (!HTcontainer.size() ) initContainer(fId);
      	if (index < HTcontainer.size()) cell = &(HTcontainer.at(index) );  break;  
      case(HcalOther) : if (extractOther(fId)==HcalCalibration) {
	  if (!CALIBcontainer.size() ) initContainer(fId);
	  if (index < CALIBcontainer.size()) cell = &(CALIBcontainer.at(index) );  
	}
	break; 
      default: break;
      }
    } else if (fId.det()==DetId::Calo) {
      if (fId.subdetId()==HcalCastorDetId::SubdetectorId) {
	if (!CASTORcontainer.size() ) initContainer(fId);
	if (index < CASTORcontainer.size()) cell = &(CASTORcontainer.at(index) );
      } else if (fId.subdetId()==HcalZDCDetId::SubdetectorId) {	
	if (!ZDCcontainer.size() ) initContainer(fId);
	if (index < ZDCcontainer.size()) cell = &(ZDCcontainer.at(index) );
      }
    }
  }

  if (cell!=0) {
    (*cell)=myItem;
    success=true;
  }

  if (!success) 
    throw cms::Exception ("Filling of conditions failed") 
      << " no valid filling possible for Conditions of type " << myname() << " for DetId " << textForId(fId);
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
