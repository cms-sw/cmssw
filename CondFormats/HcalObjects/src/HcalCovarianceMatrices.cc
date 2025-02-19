#include "CondFormats/HcalObjects/interface/HcalCovarianceMatrices.h"


HcalCovarianceMatrices::HcalCovarianceMatrices()
{
}

HcalCovarianceMatrices::~HcalCovarianceMatrices()
{
}

void
HcalCovarianceMatrices::initContainer(int container, bool h2mode_)
{
  HcalCovarianceMatrix emptyHcalCovarianceMatrix;

  switch (container) 
    {
    case HcalGenericDetId::HcalGenBarrel: 
      for (int i=0; i<(2*HcalGenericDetId::HBhalf); i++) HBcontainer.push_back(emptyHcalCovarianceMatrix); break;
    case HcalGenericDetId::HcalGenEndcap: 
      if (!h2mode_) for (int i=0; i<(2*HcalGenericDetId::HEhalf); i++) HEcontainer.push_back(emptyHcalCovarianceMatrix); 
      else for (int i=0; i<(2*HcalGenericDetId::HEhalfh2mode); i++) HEcontainer.push_back(emptyHcalCovarianceMatrix); 
      break;
    case HcalGenericDetId::HcalGenOuter: 
      for (int i=0; i<(2*HcalGenericDetId::HOhalf); i++) HOcontainer.push_back(emptyHcalCovarianceMatrix); break;
    case HcalGenericDetId::HcalGenForward: 
      for (int i=0; i<(2*HcalGenericDetId::HFhalf); i++) HFcontainer.push_back(emptyHcalCovarianceMatrix); break;
    default: break;
    }
}


const HcalCovarianceMatrix*
HcalCovarianceMatrices::getValues(DetId fId) const
{
  HcalGenericDetId myId(fId);
  bool h2mode_ = (HEcontainer.size()==(2*HcalGenericDetId::HEhalfh2mode));

  int index = myId.hashedId(h2mode_);
  //  std::cout << "::::: getting values at index " << index  << ", DetId " << myId << std::endl;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int

  const HcalCovarianceMatrix* cell = NULL;
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
    default: break;
    }
  
  //  HcalCovarianceMatrix emptyHcalCovarianceMatrix;
  //  if (cell->rawId() == emptyHcalCovarianceMatrix.rawId() ) 
  if ((!cell) || (cell->rawId() != fId ) )
    throw cms::Exception ("Conditions not found") 
      << "Unavailable Conditions of type " << myname() << " for cell " << myId;
  return cell;
}

const bool
HcalCovarianceMatrices::exists(DetId fId) const
{
  HcalGenericDetId myId(fId);
  bool h2mode_ = (HEcontainer.size()==(2*HcalGenericDetId::HEhalfh2mode));

  int index = myId.hashedId(h2mode_);
  if (index < 0) return false;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int
  const HcalCovarianceMatrix* cell = NULL;
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
  default: return false; break;
  }
  
  //  HcalCovarianceMatrix emptyHcalCovarianceMatrix;
  if (cell)
    //    if (cell->rawId() != emptyHcalCovarianceMatrix.rawId() ) 
    if (cell->rawId() == fId ) 
      return true;

  return false;
}

bool
HcalCovarianceMatrices::addValues(const HcalCovarianceMatrix& myHcalCovarianceMatrix, bool h2mode_)
{
  unsigned long myRawId = myHcalCovarianceMatrix.rawId();
  HcalGenericDetId myId(myRawId);
  int index = myId.hashedId(h2mode_);
  bool success = false;
  if (index < 0) success = false;
  unsigned int index1 = abs(index); // b/c I'm fed up with compiler warnings about comparison betw. signed and unsigned int

  switch (myId.genericSubdet() ) {
  case HcalGenericDetId::HcalGenBarrel:
    if (!HBcontainer.size() ) initContainer(myId.genericSubdet() );
    if (index1 < HBcontainer.size())
      {
        HBcontainer.at(index1)  = myHcalCovarianceMatrix;
        success = true;
      }
    break;
  case HcalGenericDetId::HcalGenEndcap: 
    if (!HEcontainer.size() ) initContainer(myId.genericSubdet(), h2mode_ );
    if (index1 < HEcontainer.size())
      {
        HEcontainer.at(index1)  = myHcalCovarianceMatrix; 
        success = true;
      }
    break;
  case HcalGenericDetId::HcalGenOuter:  
    if (!HOcontainer.size() ) initContainer(myId.genericSubdet() );
    if (index1 < HOcontainer.size())
      {
        HOcontainer.at(index1)  = myHcalCovarianceMatrix; 
        success = true;
      }
    break;
  case HcalGenericDetId::HcalGenForward: 
    if (!HFcontainer.size() ) initContainer(myId.genericSubdet() );
    if (index1 < HFcontainer.size())
      {
        HFcontainer.at(index1)  = myHcalCovarianceMatrix;
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

std::vector<DetId>
HcalCovarianceMatrices::getAllChannels() const
{
  std::vector<DetId> channels;
  HcalCovarianceMatrix emptyHcalCovarianceMatrix;
  for (unsigned int i=0; i<HBcontainer.size(); i++)
    {
      if (emptyHcalCovarianceMatrix.rawId() != HBcontainer.at(i).rawId() )
        channels.push_back( DetId(HBcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HEcontainer.size(); i++)
    {
      if (emptyHcalCovarianceMatrix.rawId() != HEcontainer.at(i).rawId() )
        channels.push_back( DetId(HEcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HOcontainer.size(); i++)
    {
      if (emptyHcalCovarianceMatrix.rawId() != HOcontainer.at(i).rawId() )
        channels.push_back( DetId(HOcontainer.at(i).rawId()) );
    }
  for (unsigned int i=0; i<HFcontainer.size(); i++)
    {
      if (emptyHcalCovarianceMatrix.rawId() != HFcontainer.at(i).rawId() )
        channels.push_back( DetId(HFcontainer.at(i).rawId()) );
    }
  return channels;
}
