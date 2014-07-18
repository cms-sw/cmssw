#ifndef PhotonAnalyzer_H
#define PhotonAnalyzer_H

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
#include "DataFormats/VertexReco/interface/VertexFwd.h"
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

//

#include <vector>
#include <string>

/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
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


class PhotonAnalyzer : public edm::EDAnalyzer
{


 public:
   
  //
  explicit PhotonAnalyzer( const edm::ParameterSet& ) ;
  virtual ~PhotonAnalyzer();
                                   
      
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& ) ;

 
 private:
  //
  bool  photonSelection (  const reco::PhotonRef & p );
  float  phiNormalization( float& a);

  MonitorElement* bookHisto(std::string histoName, std::string title, int bin, double min, double max);

  void book2DHistoVector(std::vector<std::vector<MonitorElement*> > & toFill,
			 std::string histoType, std::string histoName, std::string title,			 
							       int xbin, double xmin, double xmax,
							       int ybin=1,double ymin=1, double ymax=2);

  void book3DHistoVector( std::vector<std::vector<std::vector<MonitorElement*> > > & toFill,
			  std::string histoType, std::string histoName, std::string title, 
							       int xbin, double xmin, double xmax,
							       int ybin=1,double ymin=1, double ymax=2);


  void fill2DHistoVector(std::vector<std::vector<MonitorElement*> >& histoVector,double x, int cut, int type);
  void fill2DHistoVector(std::vector<std::vector<MonitorElement*> >& histoVector,double x, double y, int cut, int type);

  void fill3DHistoVector(std::vector<std::vector<std::vector<MonitorElement*> > >& histoVector,double x, int cut, int type, int part);
  void fill3DHistoVector(std::vector<std::vector<std::vector<MonitorElement*> > >& histoVector,double x, double y, int cut, int type, int part);



  //////////

  std::string fName_;
  int verbosity_;

  unsigned int prescaleFactor_;

  edm::EDGetTokenT<std::vector<reco::Photon> > photon_token_;

  edm::EDGetTokenT<edm::ValueMap<bool> > PhotonIDLoose_token_;

  edm::EDGetTokenT<edm::ValueMap<bool> > PhotonIDTight_token_;
  
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > barrelRecHit_token_;

  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > endcapRecHit_token_;
  
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEvent_token_;

  edm::EDGetTokenT<reco::VertexCollection> offline_pvToken_;
  
  double minPhoEtCut_;
  double photonMaxEta_;
  double invMassEtCut_;

  double cutStep_;
  int numberOfSteps_;

  bool useBinning_;
  bool useTriggerFiltering_;
  bool standAlone_;
  std::string outputFileName_;
  
  bool minimalSetOfHistos_;
  bool excludeBkgHistos_;

  int isolationStrength_; 

  bool isHeavyIon_;

  edm::ParameterSet parameters_;
           
  ////////

  DQMStore *dbe_;
  std::stringstream currentFolder_;

  int histo_index_photons_;
  int histo_index_conversions_;
  int histo_index_efficiency_;
  int histo_index_invMass_;


  int nEvt_;
  int nEntry_;

  std::vector<std::string> types_;
  std::vector<std::string> parts_;

  //////////

  MonitorElement* totalNumberOfHistos_efficiencyFolder;
  MonitorElement* totalNumberOfHistos_invMassFolder;
  MonitorElement* totalNumberOfHistos_photonsFolder;
  MonitorElement* totalNumberOfHistos_conversionsFolder;

  MonitorElement* h_nRecoVtx_;

  MonitorElement* h_phoEta_Loose_;
  MonitorElement* h_phoEta_Tight_;
  MonitorElement* h_phoEt_Loose_;
  MonitorElement* h_phoEt_Tight_;

  MonitorElement* h_phoEta_preHLT_;
  MonitorElement* h_phoEta_postHLT_;
  MonitorElement* h_phoEt_preHLT_;
  MonitorElement* h_phoEt_postHLT_;

  MonitorElement* h_convEta_Loose_;
  MonitorElement* h_convEta_Tight_;
  MonitorElement* h_convEt_Loose_;
  MonitorElement* h_convEt_Tight_;


  MonitorElement* h_phoEta_Vertex_;

  MonitorElement* h_invMassTwoWithTracks_;
  MonitorElement* h_invMassOneWithTracks_;
  MonitorElement* h_invMassZeroWithTracks_;
  MonitorElement* h_invMassAllPhotons_;
  MonitorElement* h_invMassPhotonsEBarrel_;
  MonitorElement* h_invMassPhotonsEEndcap_;

 ////////2D vectors of histograms


  std::vector<std::vector<MonitorElement*> > h_nTrackIsolSolidVsEta_;
  std::vector<std::vector<MonitorElement*> > h_trackPtSumSolidVsEta_;
  std::vector<std::vector<MonitorElement*> > h_nTrackIsolHollowVsEta_;
  std::vector<std::vector<MonitorElement*> > h_trackPtSumHollowVsEta_;
  std::vector<std::vector<MonitorElement*> > h_ecalSumVsEta_;
  std::vector<std::vector<MonitorElement*> > h_hcalSumVsEta_;


  std::vector<std::vector<MonitorElement*> > h_nTrackIsolSolidVsEt_;
  std::vector<std::vector<MonitorElement*> > h_trackPtSumSolidVsEt_;
  std::vector<std::vector<MonitorElement*> > h_nTrackIsolHollowVsEt_;
  std::vector<std::vector<MonitorElement*> > h_trackPtSumHollowVsEt_;
  std::vector<std::vector<MonitorElement*> > h_ecalSumVsEt_;
  std::vector<std::vector<MonitorElement*> > h_hcalSumVsEt_;


  std::vector<std::vector<MonitorElement*> > h_nTrackIsolSolid_;
  std::vector<std::vector<MonitorElement*> > h_trackPtSumSolid_;
  std::vector<std::vector<MonitorElement*> > h_nTrackIsolHollow_;
  std::vector<std::vector<MonitorElement*> > h_trackPtSumHollow_;
  std::vector<std::vector<MonitorElement*> > h_ecalSum_;
  std::vector<std::vector<MonitorElement*> > h_ecalSumEBarrel_;
  std::vector<std::vector<MonitorElement*> > h_ecalSumEEndcap_;
  std::vector<std::vector<MonitorElement*> > h_hcalSum_;
  std::vector<std::vector<MonitorElement*> > h_hcalSumEBarrel_;
  std::vector<std::vector<MonitorElement*> > h_hcalSumEEndcap_;

  std::vector<std::vector<MonitorElement*> > h_phoIsoBarrel_;
  std::vector<std::vector<MonitorElement*> > h_phoIsoEndcap_;
  std::vector<std::vector<MonitorElement*> > h_chHadIsoBarrel_;
  std::vector<std::vector<MonitorElement*> > h_chHadIsoEndcap_;
  std::vector<std::vector<MonitorElement*> > h_nHadIsoBarrel_;
  std::vector<std::vector<MonitorElement*> > h_nHadIsoEndcap_;


  std::vector<std::vector<MonitorElement*> > p_nTrackIsolSolidVsEta_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumSolidVsEta_;
  std::vector<std::vector<MonitorElement*> > p_nTrackIsolHollowVsEta_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumHollowVsEta_;
  std::vector<std::vector<MonitorElement*> > p_ecalSumVsEta_;
  std::vector<std::vector<MonitorElement*> > p_hcalSumVsEta_;

  std::vector<std::vector<MonitorElement*> > p_nTrackIsolSolidVsEt_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumSolidVsEt_;
  std::vector<std::vector<MonitorElement*> > p_nTrackIsolHollowVsEt_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumHollowVsEt_;

  


  std::vector<std::vector<MonitorElement*> > p_r9VsEt_;
  std::vector<std::vector<MonitorElement*> > p_r9VsEta_;

  std::vector<std::vector<MonitorElement*> > p_e1x5VsEt_;
  std::vector<std::vector<MonitorElement*> > p_e1x5VsEta_;

  std::vector<std::vector<MonitorElement*> > p_e2x5VsEt_;
  std::vector<std::vector<MonitorElement*> > p_e2x5VsEta_;

  std::vector<std::vector<MonitorElement*> > p_maxEXtalOver3x3VsEt_;
  std::vector<std::vector<MonitorElement*> > p_maxEXtalOver3x3VsEta_;

  std::vector<std::vector<MonitorElement*> > p_r1x5VsEt_;
  std::vector<std::vector<MonitorElement*> > p_r1x5VsEta_;

  std::vector<std::vector<MonitorElement*> > p_r2x5VsEt_;
  std::vector<std::vector<MonitorElement*> > p_r2x5VsEta_;

  std::vector<std::vector<MonitorElement*> > p_sigmaIetaIetaVsEta_;

  std::vector<std::vector<MonitorElement*> > p_dCotTracksVsEta_;

  std::vector<std::vector<MonitorElement*> > p_hOverEVsEta_;
  std::vector<std::vector<MonitorElement*> > p_hOverEVsEt_;

  std::vector<std::vector<MonitorElement*> > h_phoEta_;
  std::vector<std::vector<MonitorElement*> > h_scEta_;


  std::vector<std::vector<MonitorElement*> > h_phoConvEtaForEfficiency_;

  std::vector<std::vector<MonitorElement*> > h_phoEta_BadChannels_;
  std::vector<std::vector<MonitorElement*> > h_phoEt_BadChannels_;
  std::vector<std::vector<MonitorElement*> > h_phoPhi_BadChannels_;

  std::vector<std::vector<MonitorElement*> > h_phoConvEta_;

  std::vector<std::vector<MonitorElement*> > h_convVtxRvsZ_;
  std::vector<std::vector<MonitorElement*> > h_convVtxZEndcap_;
  std::vector<std::vector<MonitorElement*> > h_convVtxZ_;
  std::vector<std::vector<MonitorElement*> > h_convVtxYvsX_;
  std::vector<std::vector<MonitorElement*> > h_convVtxR_;

  std::vector<std::vector<MonitorElement*> > h_r9VsEt_;
  std::vector<std::vector<MonitorElement*> > h_r9VsEta_;


  std::vector<std::vector<MonitorElement*> > h_e1x5VsEt_;
  std::vector<std::vector<MonitorElement*> > h_e1x5VsEta_;

  std::vector<std::vector<MonitorElement*> > h_e2x5VsEt_;
  std::vector<std::vector<MonitorElement*> > h_e2x5VsEta_;

  std::vector<std::vector<MonitorElement*> > h_maxEXtalOver3x3VsEt_;
  std::vector<std::vector<MonitorElement*> > h_maxEXtalOver3x3VsEta_;

  std::vector<std::vector<MonitorElement*> > h_r1x5VsEt_;
  std::vector<std::vector<MonitorElement*> > h_r1x5VsEta_;

  std::vector<std::vector<MonitorElement*> > h_r2x5VsEt_;
  std::vector<std::vector<MonitorElement*> > h_r2x5VsEta_;

  std::vector<std::vector<MonitorElement*> > h_sigmaIetaIetaVsEta_;

  std::vector<std::vector<MonitorElement*> > h_tkChi2_;

  std::vector<std::vector<MonitorElement*> > h_vertexChi2Prob_;

  std::vector<std::vector<MonitorElement*> > p_nHitsVsEta_;

  std::vector<std::vector<MonitorElement*> > p_tkChi2VsEta_;


  ////////3D std::vectors of histograms

  std::vector<std::vector<std::vector<MonitorElement*> > > p_ecalSumVsEt_;
  std::vector<std::vector<std::vector<MonitorElement*> > > p_hcalSumVsEt_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoE_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoSigmaEoverE_;
  std::vector<std::vector<std::vector<MonitorElement*> > > p_phoSigmaEoverEvsNVtx_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoEt_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_r9_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoPhi_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_scPhi_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoConvPhiForEfficiency_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoConvPhi_;



  std::vector<std::vector<std::vector<MonitorElement*> > > h_hOverE_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_h1OverE_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_h2OverE_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoSigmaIetaIeta_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_nPho_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoConvE_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoConvEt_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoConvR9_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_nConv_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_eOverPTracks_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_pOverETracks_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_dCotTracks_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_dPhiTracksAtVtx_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_dPhiTracksAtEcal_;

  std::vector<std::vector<std::vector<MonitorElement*> > > h_dEtaTracksAtEcal_;

};





#endif




