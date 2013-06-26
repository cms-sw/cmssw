#ifndef HSCPDeDxInfo_H
#define HSCPDeDxInfo_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <vector>
#include "DataFormats/Common/interface/ValueMap.h"

namespace susybsm {


 class HSCPDeDxInfo
  {
   public:
     std::vector<float> charge;
     std::vector<float> chargeUnSat;
     std::vector<float> probability;
     std::vector<float> pathlength;
     std::vector<float> cosine;
     std::vector<uint32_t> detIds;
     std::vector<bool> shapetest;
     std::vector<float> modwidth;
     std::vector<float> modlength;
     std::vector<float> localx;
     std::vector<float> localy;
     HSCPDeDxInfo(){}
  };

  typedef  std::vector<HSCPDeDxInfo>    HSCPDeDxInfoCollection;
  typedef  edm::ValueMap<HSCPDeDxInfo>  HSCPDeDxInfoValueMap;  
  typedef  edm::Ref<HSCPDeDxInfoCollection> HSCPDeDxInfoRef;
  typedef  edm::RefProd<HSCPDeDxInfoCollection> HSCPDeDxInfoRefProd;
  typedef  edm::RefVector<HSCPDeDxInfoCollection> HSCPDeDxInfoRefVector;
}

#endif
