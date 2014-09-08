#ifndef HSCPDeDxInfo_H
#define HSCPDeDxInfo_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include <vector>

namespace susybsm {

 class HSCPDeDxInfo
  {
   public:
     std::vector<float> charges;
     std::vector<float> pathlengths;
     std::vector<uint32_t> detIds;
     std::vector<float> localPosXs;
     std::vector<float> localPosYs;
     std::vector<uint32_t> clusterIndices;
     HSCPDeDxInfo(){}
  };

  typedef  std::vector<HSCPDeDxInfo>    HSCPDeDxInfoCollection;
  typedef  edm::ValueMap<HSCPDeDxInfo>  HSCPDeDxInfoValueMap;  
  typedef  edm::Ref<HSCPDeDxInfoCollection> HSCPDeDxInfoRef;
  typedef  edm::RefProd<HSCPDeDxInfoCollection> HSCPDeDxInfoRefProd;
  typedef  edm::RefVector<HSCPDeDxInfoCollection> HSCPDeDxInfoRefVector;
}

#endif
