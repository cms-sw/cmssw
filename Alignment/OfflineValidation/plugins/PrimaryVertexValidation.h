#ifndef PrimaryVertexValidation_h
#define PrimaryVertexValidation_h

// system include files
#include <string>
#include <sstream>
#include <vector>
#include <map>

// ROOT Included
#include "TFile.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TTree.h"

// CMSSW includes
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h" 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFindingBase.h"
#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

//
// class decleration
//

class PrimaryVertexValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {

 public:
  explicit PrimaryVertexValidation(const edm::ParameterSet&);
  ~PrimaryVertexValidation();

 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  bool isHit2D(const TrackingRecHit &hit) const;
  bool hasFirstLayerPixelHits(const reco::TransientTrack track);
  std::pair<Double_t,Double_t> getMedian(TH1F *histo);
  std::pair<Double_t,Double_t> getMAD(TH1F *histo);
  std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t> > fitResiduals(TH1 *hist);
  void fillTrendPlot(TH1F* trendPlot, TH1F *residualsPlot[100], TString fitPar_, TString var_);
  static bool vtxSort( const reco::Vertex &  a, const reco::Vertex & b );
  bool passesTrackCuts(const reco::Track & track, const reco::Vertex & vertex,std::string qualityString_, double dxyErrMax_,double dzErrMax_, double ptErrMax_);
  std::map<std::string, TH1*> bookVertexHistograms(TFileDirectory dir);
  void fillTrackHistos(std::map<std::string, TH1*> & h, const std::string & ttype, const reco::TransientTrack *tt, const reco::Vertex & v,const reco::BeamSpot & beamSpot, double fBfield);
  void add(std::map<std::string, TH1*>& h, TH1* hist);
  void fill(std::map<std::string, TH1*>& h, std::string s, double x);
  void fill(std::map<std::string, TH1*>& h, std::string s, double x, double y);
  void fillMap(TH2F* trendMap, TH1F* residualsMapPlot[100][100], TString fitPar_);
  
  inline double square(double x){
    return x*x;
  }

  // ----------member data ---------------------------
  edm::ParameterSet theConfig;
  int Nevt_;
 
  TrackFilterForPVFindingBase* theTrackFilter_; 
  TrackClusterizerInZ* theTrackClusterizer_;

  // setting of the number of plots 
  static const int nMaxBins_ = 100; // maximum number of bookable histograms

  // Output 
  bool storeNtuple_;
  bool lightNtupleSwitch_;   // switch to keep only info for daily validation     
  bool useTracksFromRecoVtx_; 
  
  // requirements on the vertex
  double vertexZMax_;

  // requirements on the probe
  bool    askFirstLayerHit_;  // ask hit in the first layer of pixels 
  double  ptOfProbe_;
  double  etaOfProbe_; 
  int nBins_;                 // actual number of histograms     
  std::vector<unsigned int> runControlNumbers_;

  bool debug_;
  bool runControl_;

  edm::EDGetTokenT<reco::TrackCollection>  theTrackCollectionToken; 
  edm::EDGetTokenT<reco::VertexCollection> theVertexCollectionToken; 
  edm::EDGetTokenT<reco::BeamSpot>         theBeamspotToken;

  TTree* rootTree_;
  
  // Root-Tuple variables :
  //=======================
  void SetVarToZero();  

  static const int nMaxtracks_ = 1000;
  static const int cmToum = 10000;

  float phiSect_;
  float etaSect_;

  // event-related quantities
  int nTracks_;
  int nTracksPerClus_;
  int nClus_;
  int nOfflineVertices_;
  unsigned int RunNumber_;
  unsigned int EventNumber_;
  unsigned int LuminosityBlockNumber_;
  double  xOfflineVertex_;
  double  yOfflineVertex_;
  double  zOfflineVertex_;

  double xErrOfflineVertex_; 
  double yErrOfflineVertex_;
  double zErrOfflineVertex_;

  double BSx0_;
  double BSy0_;
  double BSz0_;
  double Beamsigmaz_;
  double Beamdxdz_;   
  double BeamWidthX_;
  double BeamWidthY_;
  double wxy2_;

  // track-related quantities
  double pt_[nMaxtracks_];   
  double p_[nMaxtracks_];    
  int nhits_[nMaxtracks_];
  int nhits1D_[nMaxtracks_];
  int nhits2D_[nMaxtracks_];
  int nhitsBPIX_[nMaxtracks_]; 
  int nhitsFPIX_[nMaxtracks_];
  int nhitsTIB_[nMaxtracks_];
  int nhitsTID_[nMaxtracks_];
  int nhitsTOB_[nMaxtracks_];
  int nhitsTEC_[nMaxtracks_];    
  int isHighPurity_[nMaxtracks_];
  double eta_[nMaxtracks_];
  double theta_[nMaxtracks_];
  double phi_[nMaxtracks_];
  double chi2_[nMaxtracks_];
  double chi2ndof_[nMaxtracks_];
  int    charge_[nMaxtracks_];
  double qoverp_[nMaxtracks_];
  double dz_[nMaxtracks_];
  double dxy_[nMaxtracks_];
  double dxyBs_[nMaxtracks_]; 
  double dzBs_[nMaxtracks_];  
  double xPCA_[nMaxtracks_];
  double yPCA_[nMaxtracks_];
  double zPCA_[nMaxtracks_];
  double xUnbiasedVertex_[nMaxtracks_];
  double yUnbiasedVertex_[nMaxtracks_];
  double zUnbiasedVertex_[nMaxtracks_];
  float  chi2normUnbiasedVertex_[nMaxtracks_]; 
  float  chi2UnbiasedVertex_[nMaxtracks_];
  float  chi2ProbUnbiasedVertex_[nMaxtracks_];
  float  DOFUnbiasedVertex_[nMaxtracks_];
  float  sumOfWeightsUnbiasedVertex_[nMaxtracks_];
  int    tracksUsedForVertexing_[nMaxtracks_];
  
  double dxyFromMyVertex_[nMaxtracks_];
  double dzFromMyVertex_[nMaxtracks_];
  double d3DFromMyVertex_[nMaxtracks_];

  double dxyErrorFromMyVertex_[nMaxtracks_];
  double dzErrorFromMyVertex_[nMaxtracks_];
  double d3DErrorFromMyVertex_[nMaxtracks_];

  double IPTsigFromMyVertex_[nMaxtracks_];
  double IPLsigFromMyVertex_[nMaxtracks_];
  double IP3DsigFromMyVertex_[nMaxtracks_];

  int   hasRecVertex_[nMaxtracks_];
  int   isGoodTrack_[nMaxtracks_];

  // ---- directly histograms // ===> unbiased residuals
  
  // absolute residuals

  TH1F* a_dxyPhiResiduals[nMaxBins_];
  TH1F* a_dxyEtaResiduals[nMaxBins_];
  
  TH1F* a_dzPhiResiduals[nMaxBins_];
  TH1F* a_dzEtaResiduals[nMaxBins_];
  
  TH1F* a_IP2DPhiResiduals[nMaxBins_];
  TH1F* a_IP2DEtaResiduals[nMaxBins_];
  
  TH1F* a_IP3DPhiResiduals[nMaxBins_];
  TH1F* a_IP3DEtaResiduals[nMaxBins_];

  TH1F* a_reszPhiResiduals[nMaxBins_];
  TH1F* a_reszEtaResiduals[nMaxBins_];

  TH1F* a_d3DPhiResiduals[nMaxBins_];
  TH1F* a_d3DEtaResiduals[nMaxBins_];

  // normalized residuals

  TH1F* n_dxyPhiResiduals[nMaxBins_];
  TH1F* n_dxyEtaResiduals[nMaxBins_];
  
  TH1F* n_dzPhiResiduals[nMaxBins_];
  TH1F* n_dzEtaResiduals[nMaxBins_];
  
  TH1F* n_IP2DPhiResiduals[nMaxBins_];
  TH1F* n_IP2DEtaResiduals[nMaxBins_];
  
  TH1F* n_IP3DPhiResiduals[nMaxBins_];
  TH1F* n_IP3DEtaResiduals[nMaxBins_];

  TH1F* n_reszPhiResiduals[nMaxBins_];
  TH1F* n_reszEtaResiduals[nMaxBins_];

  TH1F* n_d3DPhiResiduals[nMaxBins_];
  TH1F* n_d3DEtaResiduals[nMaxBins_];

  // for the maps

  TH1F* a_dxyResidualsMap[nMaxBins_][nMaxBins_];
  TH1F* a_dzResidualsMap[nMaxBins_][nMaxBins_];
  TH1F* a_d3DResidualsMap[nMaxBins_][nMaxBins_];
      				 				    
  TH1F* n_dxyResidualsMap[nMaxBins_][nMaxBins_];  				 
  TH1F* n_dzResidualsMap[nMaxBins_][nMaxBins_];
  TH1F* n_d3DResidualsMap[nMaxBins_][nMaxBins_];
  
  // ---- trends as function of phi
  
  TH1F* a_dxyPhiMeanTrend;
  TH1F* a_dxyPhiWidthTrend;
  TH1F* a_dzPhiMeanTrend;
  TH1F* a_dzPhiWidthTrend;

  TH1F* a_dxyEtaMeanTrend;
  TH1F* a_dxyEtaWidthTrend;
  TH1F* a_dzEtaMeanTrend;
  TH1F* a_dzEtaWidthTrend;

  TH1F* n_dxyPhiMeanTrend;
  TH1F* n_dxyPhiWidthTrend;
  TH1F* n_dzPhiMeanTrend;
  TH1F* n_dzPhiWidthTrend;

  TH1F* n_dxyEtaMeanTrend;
  TH1F* n_dxyEtaWidthTrend;
  TH1F* n_dzEtaMeanTrend;
  TH1F* n_dzEtaWidthTrend;

  // ---- medians and MAD

  TH1F* a_dxyPhiMedianTrend;
  TH1F* a_dxyPhiMADTrend;
  TH1F* a_dzPhiMedianTrend;
  TH1F* a_dzPhiMADTrend;

  TH1F* a_dxyEtaMedianTrend;
  TH1F* a_dxyEtaMADTrend;
  TH1F* a_dzEtaMedianTrend;
  TH1F* a_dzEtaMADTrend;

  TH1F* n_dxyPhiMedianTrend;
  TH1F* n_dxyPhiMADTrend;
  TH1F* n_dzPhiMedianTrend;
  TH1F* n_dzPhiMADTrend;

  TH1F* n_dxyEtaMedianTrend;
  TH1F* n_dxyEtaMADTrend;
  TH1F* n_dzEtaMedianTrend;
  TH1F* n_dzEtaMADTrend;

  // 2D residuals
  
  TH2F* a_dxyVsPhi;
  TH2F* a_dzVsPhi;

  TH2F* n_dxyVsPhi;
  TH2F* n_dzVsPhi;

  TH2F* a_dxyVsEta;
  TH2F* a_dzVsEta;

  TH2F* n_dxyVsEta;
  TH2F* n_dzVsEta;

  // 2D maps

  TH2F* a_dxyMeanMap;
  TH2F* a_dzMeanMap;

  TH2F* n_dxyMeanMap;
  TH2F* n_dzMeanMap;

  TH2F* a_dxyWidthMap;
  TH2F* a_dzWidthMap;

  TH2F* n_dxyWidthMap;
  TH2F* n_dzWidthMap;

  // ---- directly histograms =================> biased residuals
  
  // absolute residuals

  TH1F* a_dxyPhiBiasResiduals[nMaxBins_];
  TH1F* a_dxyEtaBiasResiduals[nMaxBins_];
  
  TH1F* a_dzPhiBiasResiduals[nMaxBins_];
  TH1F* a_dzEtaBiasResiduals[nMaxBins_];
  
  // normalized BiasResiduals

  TH1F* n_dxyPhiBiasResiduals[nMaxBins_];
  TH1F* n_dxyEtaBiasResiduals[nMaxBins_];
  
  TH1F* n_dzPhiBiasResiduals[nMaxBins_];
  TH1F* n_dzEtaBiasResiduals[nMaxBins_];
  
  // for the maps

  TH1F* a_dxyBiasResidualsMap[nMaxBins_][nMaxBins_];
  TH1F* a_dzBiasResidualsMap[nMaxBins_][nMaxBins_];
        				 				    
  TH1F* n_dxyBiasResidualsMap[nMaxBins_][nMaxBins_];  				 
  TH1F* n_dzBiasResidualsMap[nMaxBins_][nMaxBins_];

  // ---- trends as function of phi
  
  TH1F* a_dxyPhiMeanBiasTrend;
  TH1F* a_dxyPhiWidthBiasTrend;
  TH1F* a_dzPhiMeanBiasTrend;
  TH1F* a_dzPhiWidthBiasTrend;

  TH1F* a_dxyEtaMeanBiasTrend;
  TH1F* a_dxyEtaWidthBiasTrend;
  TH1F* a_dzEtaMeanBiasTrend;
  TH1F* a_dzEtaWidthBiasTrend;

  TH1F* n_dxyPhiMeanBiasTrend;
  TH1F* n_dxyPhiWidthBiasTrend;
  TH1F* n_dzPhiMeanBiasTrend;
  TH1F* n_dzPhiWidthBiasTrend;

  TH1F* n_dxyEtaMeanBiasTrend;
  TH1F* n_dxyEtaWidthBiasTrend;
  TH1F* n_dzEtaMeanBiasTrend;
  TH1F* n_dzEtaWidthBiasTrend;

  // ---- medians and MAD

  TH1F* a_dxyPhiMedianBiasTrend;
  TH1F* a_dxyPhiMADBiasTrend;
  TH1F* a_dzPhiMedianBiasTrend;
  TH1F* a_dzPhiMADBiasTrend;

  TH1F* a_dxyEtaMedianBiasTrend;
  TH1F* a_dxyEtaMADBiasTrend;
  TH1F* a_dzEtaMedianBiasTrend;
  TH1F* a_dzEtaMADBiasTrend;

  TH1F* n_dxyPhiMedianBiasTrend;
  TH1F* n_dxyPhiMADBiasTrend;
  TH1F* n_dzPhiMedianBiasTrend;
  TH1F* n_dzPhiMADBiasTrend;

  TH1F* n_dxyEtaMedianBiasTrend;
  TH1F* n_dxyEtaMADBiasTrend;
  TH1F* n_dzEtaMedianBiasTrend;
  TH1F* n_dzEtaMADBiasTrend;

  // 2D maps

  TH2F* a_dxyMeanBiasMap;
  TH2F* a_dzMeanBiasMap;

  TH2F* n_dxyMeanBiasMap;
  TH2F* n_dzMeanBiasMap;

  TH2F* a_dxyWidthBiasMap;
  TH2F* a_dzWidthBiasMap;

  TH2F* n_dxyWidthBiasMap;
  TH2F* n_dzWidthBiasMap;

  // check event
  TH1F* h_nTracks;
  TH1F* h_nClus;
  TH1F* h_nOfflineVertices;
  TH1F* h_runNumber;
  TH1F* h_xOfflineVertex;
  TH1F* h_yOfflineVertex;
  TH1F* h_zOfflineVertex;
  TH1F* h_xErrOfflineVertex;
  TH1F* h_yErrOfflineVertex;
  TH1F* h_zErrOfflineVertex;
  TH1F* h_BSx0;    
  TH1F* h_BSy0;    
  TH1F* h_BSz0;    
  TH1F* h_Beamsigmaz; 
  TH1F* h_BeamWidthX;
  TH1F* h_BeamWidthY;          

  // check probe 

  TH2F* h2_probeEtaPhi_;
  TH2F* h2_probeEtaPt_;

  TH1F* h_probeP_;
  TH1F* h_probePt_;
  TH1F* h_probeEta_;
  TH1F* h_probePhi_;
  TH1F* h_probeChi2_;
  TH1F* h_probeNormChi2_;
  TH1F* h_probeCharge_;
  TH1F* h_probeQoverP_;
  
  TH1F* h_probedzRecoV_;
  TH1F* h_probedxyRecoV_;

  TH1F* h_probedzRefitV_;
  TH1F* h_probedxyRefitV_;

  TH1F* h_probed0RefitV_;
  TH1F* h_probez0RefitV_;

  TH1F* h_probesignIP2DRefitV_;
  TH1F* h_probed3DRefitV_;
  TH1F* h_probereszRefitV_;
  
  TH1F* h_probeRecoVSigZ_;
  TH1F* h_probeRecoVSigXY_;
  TH1F* h_probeRefitVSigZ_;
  TH1F* h_probeRefitVSigXY_;
  TH1F* h_probeRefitVSig3D_;
  TH1F* h_probeRefitVLogSig3D_;
  TH1F* h_probeRefitVSigResZ_;

  TH1F* h_probeHits_;  
  TH1F* h_probeHits1D_; 
  TH1F* h_probeHits2D_; 
  TH1F* h_probeHitsInTIB_;  
  TH1F* h_probeHitsInTOB_;  
  TH1F* h_probeHitsInTID_;  
  TH1F* h_probeHitsInTEC_;  
  TH1F* h_probeHitsInBPIX_; 
  TH1F* h_probeHitsInFPIX_; 

  // check vertex

  TH1F* h_fitVtxNdof_;
  TH1F* h_fitVtxChi2_;
  TH1F* h_fitVtxNtracks_;  
  TH1F* h_fitVtxChi2ndf_;  
  TH1F* h_fitVtxChi2Prob_; 
  TH1F* h_fitVtxTrackWeights_;
  TH1F* h_fitVtxTrackAverageWeight_;

  TH1F* h_recoVtxNtracks_; 
  TH1F* h_recoVtxChi2ndf_; 
  TH1F* h_recoVtxChi2Prob_;
  TH1F* h_recoVtxSumPt_;   

  std::map<std::string, TH1*> hDA;

};

#endif
