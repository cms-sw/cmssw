#ifndef HSCPCaloInfo_H
#define HSCPCaloInfo_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <vector>
#include "DataFormats/Common/interface/ValueMap.h"

namespace susybsm {


 class HSCPCaloInfo
  {
   public:
     float hcalCrossedEnergy, ecalCrossedEnergy, hoCrossedEnergy;
     float ecal3by3dir, ecal5by5dir;
     float hcal3by3dir, hcal5by5dir;
     float ecalBeta, ecalBetaError;
     float ecalDeDx;
     float trkIsoDr;
     float ecalTime, ecalTimeError;
     float ecalInvBetaError;
     int ecalCrysCrossed;
     std::vector<float> ecalSwissCrossKs;
     std::vector<float> ecalE1OverE9s;
     std::vector<float> ecalTrackLengths;
     std::vector<float> ecalEnergies;
     std::vector<float> ecalTimes;
     std::vector<float> ecalTimeErrors;
     std::vector<float> ecalChi2s;
     std::vector<float> ecalOutOfTimeChi2s;
     std::vector<float> ecalOutOfTimeEnergies;
     std::vector<DetId> ecalDetIds;
     std::vector<GlobalPoint> ecalTrackExitPositions;

     HSCPCaloInfo()
     {
       hcalCrossedEnergy = -9999;
       ecalCrossedEnergy = -9999;
       hoCrossedEnergy = -9999;
       ecal3by3dir = -9999;
       ecal5by5dir = -9999;
       hcal3by3dir = -9999;
       hcal5by5dir = -9999;
       ecalBeta = -9999;
       ecalBetaError = -9999;
       ecalDeDx = -9999;
       trkIsoDr = -9999;
       ecalTime = -9999;
       ecalTimeError = -9999;
       ecalInvBetaError = -9999;
       ecalCrysCrossed = 0;
     }
  };

  typedef  std::vector<HSCPCaloInfo>    HSCPCaloInfoCollection;
  typedef  edm::ValueMap<HSCPCaloInfo>  HSCPCaloInfoValueMap;  
  typedef  edm::Ref<HSCPCaloInfoCollection> HSCPCaloInfoRef;
  typedef  edm::RefProd<HSCPCaloInfoCollection> HSCPCaloInfoRefProd;
  typedef  edm::RefVector<HSCPCaloInfoCollection> HSCPCaloInfoRefVector;

}

#endif
