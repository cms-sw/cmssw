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

using std::vector;
using std::string;

/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date: 2010/06/03 15:46:38 $ 
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
 
 private:
  //

  float  phiNormalization( float& a);

  MonitorElement* bookHisto(string histoName, string title, int bin, double min, double max);

  vector<vector<MonitorElement*> > book2DHistoVector(string histoType, string histoName, string title, 
							       int xbin, double xmin, double xmax,
							       int ybin=1,double ymin=1, double ymax=2);

  vector<vector<vector<MonitorElement*> > > book3DHistoVector(string histoType, string histoName, string title, 
							       int xbin, double xmin, double xmax,
							       int ybin=1,double ymin=1, double ymax=2);


  void fill2DHistoVector(vector<vector<MonitorElement*> >& histoVector,double x, int cut, int type);
  void fill2DHistoVector(vector<vector<MonitorElement*> >& histoVector,double x, double y, int cut, int type);

  void fill3DHistoVector(vector<vector<vector<MonitorElement*> > >& histoVector,double x, int cut, int type, int part);
  void fill3DHistoVector(vector<vector<vector<MonitorElement*> > >& histoVector,double x, double y, int cut, int type, int part);



  //////////

  string fName_;
  int verbosity_;

  unsigned int prescaleFactor_;

  string photonProducer_;       
  string photonCollection_;

  string barrelRecHitProducer_;
  string barrelRecHitCollection_;

  string endcapRecHitProducer_;
  string endcapRecHitCollection_;

  edm::InputTag triggerEvent_;

  double minPhoEtCut_;
  double invMassEtCut_;

  double cutStep_;
  int numberOfSteps_;

  bool useBinning_;
  bool useTriggerFiltering_;
  bool standAlone_;
  string outputFileName_;
  

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

  vector<string> types_;
  vector<string> parts_;

  //////////

  MonitorElement* totalNumberOfHistos_efficiencyFolder;
  MonitorElement* totalNumberOfHistos_invMassFolder;
  MonitorElement* totalNumberOfHistos_photonsFolder;
  MonitorElement* totalNumberOfHistos_conversionsFolder;


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

 ////////2D vectors of histograms


  vector<vector<MonitorElement*> > h_nTrackIsolSolidVsEta_;
  vector<vector<MonitorElement*> > h_trackPtSumSolidVsEta_;
  vector<vector<MonitorElement*> > h_nTrackIsolHollowVsEta_;
  vector<vector<MonitorElement*> > h_trackPtSumHollowVsEta_;
  vector<vector<MonitorElement*> > h_ecalSumVsEta_;
  vector<vector<MonitorElement*> > h_hcalSumVsEta_;


  vector<vector<MonitorElement*> > h_nTrackIsolSolidVsEt_;
  vector<vector<MonitorElement*> > h_trackPtSumSolidVsEt_;
  vector<vector<MonitorElement*> > h_nTrackIsolHollowVsEt_;
  vector<vector<MonitorElement*> > h_trackPtSumHollowVsEt_;
  vector<vector<MonitorElement*> > h_ecalSumVsEt_;
  vector<vector<MonitorElement*> > h_hcalSumVsEt_;


  vector<vector<MonitorElement*> > h_nTrackIsolSolid_;
  vector<vector<MonitorElement*> > h_trackPtSumSolid_;
  vector<vector<MonitorElement*> > h_nTrackIsolHollow_;
  vector<vector<MonitorElement*> > h_trackPtSumHollow_;
  vector<vector<MonitorElement*> > h_ecalSum_;
  vector<vector<MonitorElement*> > h_hcalSum_;


  vector<vector<MonitorElement*> > p_nTrackIsolSolidVsEta_;
  vector<vector<MonitorElement*> > p_trackPtSumSolidVsEta_;
  vector<vector<MonitorElement*> > p_nTrackIsolHollowVsEta_;
  vector<vector<MonitorElement*> > p_trackPtSumHollowVsEta_;
  vector<vector<MonitorElement*> > p_ecalSumVsEta_;
  vector<vector<MonitorElement*> > p_hcalSumVsEta_;

  vector<vector<MonitorElement*> > p_nTrackIsolSolidVsEt_;
  vector<vector<MonitorElement*> > p_trackPtSumSolidVsEt_;
  vector<vector<MonitorElement*> > p_nTrackIsolHollowVsEt_;
  vector<vector<MonitorElement*> > p_trackPtSumHollowVsEt_;

  vector<vector<MonitorElement*> > p_r9VsEt_;
  vector<vector<MonitorElement*> > p_r9VsEta_;

  vector<vector<MonitorElement*> > p_e1x5VsEt_;
  vector<vector<MonitorElement*> > p_e1x5VsEta_;

  vector<vector<MonitorElement*> > p_e2x5VsEt_;
  vector<vector<MonitorElement*> > p_e2x5VsEta_;

  vector<vector<MonitorElement*> > p_maxEXtalOver3x3VsEt_;
  vector<vector<MonitorElement*> > p_maxEXtalOver3x3VsEta_;

  vector<vector<MonitorElement*> > p_r1x5VsEt_;
  vector<vector<MonitorElement*> > p_r1x5VsEta_;

  vector<vector<MonitorElement*> > p_r2x5VsEt_;
  vector<vector<MonitorElement*> > p_r2x5VsEta_;

  vector<vector<MonitorElement*> > p_sigmaIetaIetaVsEta_;

  vector<vector<MonitorElement*> > p_dCotTracksVsEta_;

  vector<vector<MonitorElement*> > p_hOverEVsEta_;
  vector<vector<MonitorElement*> > p_hOverEVsEt_;

  vector<vector<MonitorElement*> > h_phoEta_;
  vector<vector<MonitorElement*> > h_scEta_;


  vector<vector<MonitorElement*> > h_phoConvEtaForEfficiency_;

  vector<vector<MonitorElement*> > h_phoEta_BadChannels_;
  vector<vector<MonitorElement*> > h_phoEt_BadChannels_;
  vector<vector<MonitorElement*> > h_phoPhi_BadChannels_;

  vector<vector<MonitorElement*> > h_phoConvEta_;

  vector<vector<MonitorElement*> > h_convVtxRvsZ_;
  vector<vector<MonitorElement*> > h_convVtxZEndcap_;
  vector<vector<MonitorElement*> > h_convVtxZ_;
  vector<vector<MonitorElement*> > h_convVtxYvsX_;
  vector<vector<MonitorElement*> > h_convVtxR_;

  vector<vector<MonitorElement*> > h_r9VsEt_;
  vector<vector<MonitorElement*> > h_r9VsEta_;


  vector<vector<MonitorElement*> > h_e1x5VsEt_;
  vector<vector<MonitorElement*> > h_e1x5VsEta_;

  vector<vector<MonitorElement*> > h_e2x5VsEt_;
  vector<vector<MonitorElement*> > h_e2x5VsEta_;

  vector<vector<MonitorElement*> > h_maxEXtalOver3x3VsEt_;
  vector<vector<MonitorElement*> > h_maxEXtalOver3x3VsEta_;

  vector<vector<MonitorElement*> > h_r1x5VsEt_;
  vector<vector<MonitorElement*> > h_r1x5VsEta_;

  vector<vector<MonitorElement*> > h_r2x5VsEt_;
  vector<vector<MonitorElement*> > h_r2x5VsEta_;

  vector<vector<MonitorElement*> > h_sigmaIetaIetaVsEta_;

  vector<vector<MonitorElement*> > h_tkChi2_;

  vector<vector<MonitorElement*> > h_vertexChi2Prob_;

  vector<vector<MonitorElement*> > p_nHitsVsEta_;

  vector<vector<MonitorElement*> > p_tkChi2VsEta_;


  ////////3D vectors of histograms

  vector<vector<vector<MonitorElement*> > > p_ecalSumVsEt_;
  vector<vector<vector<MonitorElement*> > > p_hcalSumVsEt_;

  vector<vector<vector<MonitorElement*> > > h_phoE_;
  vector<vector<vector<MonitorElement*> > > h_phoEt_;
  vector<vector<vector<MonitorElement*> > > h_r9_;
  vector<vector<vector<MonitorElement*> > > h_phoPhi_;
  vector<vector<vector<MonitorElement*> > > h_scPhi_;
  vector<vector<vector<MonitorElement*> > > h_phoConvPhiForEfficiency_;
  vector<vector<vector<MonitorElement*> > > h_phoConvPhi_;



  vector<vector<vector<MonitorElement*> > > h_hOverE_;
  vector<vector<vector<MonitorElement*> > > h_h1OverE_;
  vector<vector<vector<MonitorElement*> > > h_h2OverE_;

  vector<vector<vector<MonitorElement*> > > h_phoSigmaIetaIeta_;

  vector<vector<vector<MonitorElement*> > > h_nPho_;

  vector<vector<vector<MonitorElement*> > > h_phoConvE_;
  vector<vector<vector<MonitorElement*> > > h_phoConvEt_;
  vector<vector<vector<MonitorElement*> > > h_phoConvR9_;

  vector<vector<vector<MonitorElement*> > > h_nConv_;

  vector<vector<vector<MonitorElement*> > > h_eOverPTracks_;
  vector<vector<vector<MonitorElement*> > > h_pOverETracks_;

  vector<vector<vector<MonitorElement*> > > h_dCotTracks_;

  vector<vector<vector<MonitorElement*> > > h_dPhiTracksAtVtx_;
  vector<vector<vector<MonitorElement*> > > h_dPhiTracksAtEcal_;

  vector<vector<vector<MonitorElement*> > > h_dEtaTracksAtEcal_;

};





#endif




