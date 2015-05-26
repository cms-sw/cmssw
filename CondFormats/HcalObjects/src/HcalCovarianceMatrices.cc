#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrices.h"


HcalCovarianceMatrices::HcalCovarianceMatrices(const HcalTopology* topo) : HcalCondObjectContainerBase(topo) 
{
}

HcalCovarianceMatrices::~HcalCovarianceMatrices()
{
}

void
HcalCovarianceMatrices::initContainer(DetId fId)
{
  HcalCovarianceMatrix emptyItem;


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


const HcalCovarianceMatrix*
HcalCovarianceMatrices::getValues(DetId fId, bool throwOnFail) const
{
  unsigned int index=indexFor(fId);

  const HcalCovarianceMatrix* cell = NULL;

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
  
  //  HcalCovarianceMatrix emptyHcalCovarianceMatrix;
  //  if (cell->rawId() == emptyHcalCovarianceMatrix.rawId() ) 
  if ((!cell) || (!hcalEqualDetId(cell,fId))) {
//    (fId.det()==DetId::Hcal && HcalDetId(cell->rawId()) != HcalDetId(fId)) ||
//    (fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(cell->rawId()) != HcalZDCDetId(fId)) ||
//    (fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (cell->rawId() != fId))) {
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
HcalCovarianceMatrices::exists(DetId fId) const
{
  const HcalCovarianceMatrix* cell = getValues(fId,false);
  
  //  HcalCovarianceMatrix emptyHcalCovarianceMatrix;
  if (cell)
    if (hcalEqualDetId(cell,fId))
//	(fId.det()==DetId::Hcal && HcalDetId(cell->rawId()) == HcalDetId(fId)) ||
//	(fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(cell->rawId()) == HcalZDCDetId(fId)) ||
//	(fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (cell->rawId() == fId)))
      return true;

  return false;
}

bool
HcalCovarianceMatrices::addValues(const HcalCovarianceMatrix& myHcalCovarianceMatrix)
{
  unsigned int index=0;
  bool success = false;
  HcalCovarianceMatrix* cell=0;
  DetId fId(myHcalCovarianceMatrix.rawId());

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
  if (cell) {
    *cell=myHcalCovarianceMatrix;
    success = true;
  }


  if (!success) 
    throw cms::Exception ("Filling of conditions failed") 
      << " no valid filling possible for Conditions of type " << myname() << " for DetId " << fId.rawId();
  return success;
}

std::vector<DetId>
HcalCovarianceMatrices::getAllChannels() const
{
  std::vector<DetId> channels;
  HcalCovarianceMatrix emptyHcalCovarianceMatrix;
  for (unsigned int i=0; i<HBcontainer.size(); i++) {
    if (emptyHcalCovarianceMatrix.rawId() != HBcontainer.at(i).rawId() )
      channels.push_back( DetId(HBcontainer.at(i).rawId()) );
  }
  for (unsigned int i=0; i<HEcontainer.size(); i++) {
    if (emptyHcalCovarianceMatrix.rawId() != HEcontainer.at(i).rawId() )
      channels.push_back( DetId(HEcontainer.at(i).rawId()) );
  }
  for (unsigned int i=0; i<HOcontainer.size(); i++) {
    if (emptyHcalCovarianceMatrix.rawId() != HOcontainer.at(i).rawId() )
      channels.push_back( DetId(HOcontainer.at(i).rawId()) );
  }
  for (unsigned int i=0; i<HFcontainer.size(); i++) {
    if (emptyHcalCovarianceMatrix.rawId() != HFcontainer.at(i).rawId() )
      channels.push_back( DetId(HFcontainer.at(i).rawId()) );
  }
  return channels;
}
