#ifndef ZToMuMuGammaAnalyzer_H
#define ZToMuMuGammaAnalyzer_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
// DataFormats
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

/// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"



#include <vector>
#include <string>

/** \class ZToMuMuGammaAnalyzer
 **  
 **
 **  $Id: ZToMuMuGammaAnalyzer
 **  $Date: 2011/05/23 15:01:13 $ 
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
 **     
 ***/


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;


class ZToMuMuGammaAnalyzer  : public edm::EDAnalyzer
{


 public:
   
  //
  explicit ZToMuMuGammaAnalyzer( const edm::ParameterSet& ) ;
  virtual ~ZToMuMuGammaAnalyzer();
                                   
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& ) ;
  
      
 private:
  std::string photonProducer_;       
  std::string photonCollection_;
  std::string barrelRecHitProducer_;
  std::string barrelRecHitCollection_;
  std::string endcapRecHitProducer_;
  std::string endcapRecHitCollection_;
  std::string muonProducer_;       
  std::string muonCollection_;
  //
  std::string fName_;
  int verbosity_;
  edm::InputTag triggerEvent_;
  bool useTriggerFiltering_;
  unsigned int prescaleFactor_;
  bool standAlone_;
  std::string outputFileName_;
  edm::ParameterSet parameters_;
  bool isHeavyIon_;
  DQMStore *dbe_;
  std::stringstream currentFolder_;
  int nEvt_;
  int nEntry_;


  // muon selection
  float muonMinPt_;
  int minPixStripHits_;
  float muonMaxChi2_;
  float muonMaxDxy_;
  int muonMatches_;
  int validPixHits_;
  int validMuonHits_;
  float muonTrackIso_;
  float muonTightEta_;
  // dimuon selection
  float minMumuInvMass_;
  float maxMumuInvMass_;
  // photon selection
  float photonMinEt_;
  float photonMaxEta_;
  float photonTrackIso_;
  
  // mu mu gamma selection
  float nearMuonDr_;
  float nearMuonHcalIso_;
  float farMuonEcalIso_;
  float farMuonTrackIso_;
  float farMuonMinPt_;
  float minMumuGammaInvMass_;
  float maxMumuGammaInvMass_;

  float mumuInvMass( const reco::Muon & m1,const reco::Muon & m2 ) ;
  float mumuGammaInvMass(const reco::Muon & mu1,const reco::Muon & mu2, const reco::Photon& pho );
  bool  basicMuonSelection (  const reco::Muon & m );
  bool  muonSelection (  const reco::Muon & m,  const reco::BeamSpot& bs );
  bool  photonSelection (  const reco::Photon & p );

  MonitorElement*  h1_mumuInvMass_;
  MonitorElement*  h1_mumuGammaInvMass_;

};





#endif




