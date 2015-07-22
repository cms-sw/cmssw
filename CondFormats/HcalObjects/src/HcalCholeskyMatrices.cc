#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrices.h"
#include "CondFormats/HcalObjects/interface/HcalDetIdRelationship.h"

HcalCholeskyMatrices::HcalCholeskyMatrices(const HcalTopology* topo) : HcalCondObjectContainerBase(topo) 
{
}

HcalCholeskyMatrices::~HcalCholeskyMatrices()
{
}

void
HcalCholeskyMatrices::initContainer(DetId fId)
{
  HcalCholeskyMatrix emptyItem;

  if (fId.det()==DetId::Hcal) {
    switch (HcalSubdetector(fId.subdetId())) {
    case(HcalBarrel) : for (unsigned int i=0; i<sizeFor(fId); i++) HBcontainer.push_back(emptyItem); break;
    case(HcalEndcap) : for (unsigned int i=0; i<sizeFor(fId); i++) HEcontainer.push_back(emptyItem); break;
    case(HcalOuter) : for (unsigned int i=0; i<sizeFor(fId); i++) HOcontainer.push_back(emptyItem); break;
    case(HcalForward) : for (unsigned int i=0; i<sizeFor(fId); i++) HFcontainer.push_back(emptyItem); break;
    default: break;
    }
  }
}


const HcalCholeskyMatrix*
HcalCholeskyMatrices::getValues(DetId fId, bool throwOnFail) const
{
  unsigned int index = indexFor(fId);
  const HcalCholeskyMatrix* cell = NULL;
  if (index<0xFFFFFFFFu) {
    if (fId.det()==DetId::Hcal) {
      switch (HcalSubdetector(fId.subdetId())) {
      case(HcalBarrel) : if (index < HBcontainer.size()) cell = &(HBcontainer.at(index) );  
      case(HcalEndcap) : if (index < HEcontainer.size()) cell = &(HEcontainer.at(index) );  
      case(HcalForward) : if (index < HFcontainer.size()) cell = &(HFcontainer.at(index) );  
      case(HcalOuter) : if (index < HOcontainer.size()) cell = &(HOcontainer.at(index) );  
    default: break;
      }
    }
  }
  
  //  HcalCholeskyMatrix emptyHcalCholeskyMatrix;
  //  if (cell->rawId() == emptyHcalCholeskyMatrix.rawId() ) 
  if ((!cell) || (!hcalEqualDetId(cell,fId))) {
//     (fId.det()==DetId::Hcal && HcalDetId(cell->rawId()) != HcalDetId(fId)) ||
//     (fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(cell->rawId()) != HcalZDCDetId(fId)) ||
//     (fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (cell->rawId() != fId))) {
    if (throwOnFail) {
      throw cms::Exception ("Conditions not found") 
	<< "Unavailable Conditions of type " << myname() << " for cell " << fId.rawId();
    } else {
      cell=0;
    }
  }
  return cell;
}

const bool
HcalCholeskyMatrices::exists(DetId fId) const
{

  const HcalCholeskyMatrix* cell = getValues(fId,false);
  
  //  HcalCholeskyMatrix emptyHcalCholeskyMatrix;
  if (cell)
    if (!hcalEqualDetId(cell,fId))
//	(fId.det()==DetId::Hcal && HcalDetId(cell->rawId()) == HcalDetId(fId)) ||
//	(fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(cell->rawId()) == HcalZDCDetId(fId)) ||
//	(fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (cell->rawId() == fId)))
      return true;
  
  return false;
}

bool
HcalCholeskyMatrices::addValues(const HcalCholeskyMatrix& myItem)
{
  bool success = false;
  DetId fId(myItem.rawId());
  unsigned int index=indexFor(fId);

  HcalCholeskyMatrix* cell = NULL;

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
      default: break;
      }
    }
  }

  if (cell!=0) {
    (*cell)=myItem;
    success = true;
  }

  if (!success) 
    throw cms::Exception ("Filling of conditions failed") 
      << " no valid filling possible for Conditions of type " << myname() << " for DetId " << fId.rawId();
  return success;
}

std::vector<DetId>
HcalCholeskyMatrices::getAllChannels() const
{
  std::vector<DetId> channels;
  HcalCholeskyMatrix emptyHcalCholeskyMatrix;
  for (unsigned int i=0; i<HBcontainer.size(); i++) {
    if (emptyHcalCholeskyMatrix.rawId() != HBcontainer.at(i).rawId() )
      channels.push_back( DetId(HBcontainer.at(i).rawId()) );
  }
  for (unsigned int i=0; i<HEcontainer.size(); i++) {
    if (emptyHcalCholeskyMatrix.rawId() != HEcontainer.at(i).rawId() )
      channels.push_back( DetId(HEcontainer.at(i).rawId()) );
  }
  for (unsigned int i=0; i<HOcontainer.size(); i++) {
    if (emptyHcalCholeskyMatrix.rawId() != HOcontainer.at(i).rawId() )
      channels.push_back( DetId(HOcontainer.at(i).rawId()) );
  }
  for (unsigned int i=0; i<HFcontainer.size(); i++) {
    if (emptyHcalCholeskyMatrix.rawId() != HFcontainer.at(i).rawId() )
      channels.push_back( DetId(HFcontainer.at(i).rawId()) );
  }
  return channels;
}

