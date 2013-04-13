#ifndef PrimaryVertexValidation_h
#define PrimaryVertexValidation_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TH1D.h"
#include "TH1I.h"
#include "TH2D.h"
#include "TTree.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"


#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFindingBase.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"
#include "RecoVertex/PrimaryVertexProducer/interface/TrackClusterizerInZ.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

// system include files
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <map>

//
// class decleration
//

class PrimaryVertexValidation : public edm::EDAnalyzer {

 public:
  explicit PrimaryVertexValidation(const edm::ParameterSet&);
  ~PrimaryVertexValidation();

 private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  bool isHit2D(const TrackingRecHit &hit) const;
  bool hasFirstLayerPixelHits(const reco::TransientTrack track);
  AlgebraicVector3 displacementFromTrack(const GlobalPoint& pv, const GlobalPoint& dcaPosition_,  const GlobalVector& tangent_);
  double approximateTrackError(const GlobalPoint& refPoint, const GlobalPoint& dcaPosition,const GlobalVector& tangent, const AlgebraicMatrix33& covMatrix);
  std::pair<Double_t,Double_t> getMedian(TH1F *histo);
  std::pair<Double_t,Double_t> getMAD(TH1F *histo);
  std::pair<std::pair<Double_t,Double_t>, std::pair<Double_t,Double_t> > fitResiduals(TH1 *hist);
  void FillTrendPlot(TH1F* trendPlot, TH1F *residualsPlot[100], TString fitPar_, TString var_);

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
  
  // requirements on the probe
  bool    askFirstLayerHit_;  // ask hit in the first layer of pixels 
  double  ptOfProbe_;
  double  etaOfProbe_; 
  int nBins_;                 // actual number of histograms     

  bool debug_;
  edm::InputTag  TrackCollectionTag_;

  TTree* rootTree_;
  
  // Root-Tuple variables :
  //=======================
  void SetVarToZero();  

  static const int nMaxtracks_ = 1000;
  static const int cmToum = 10000;

  float phipitch_;
  float etapitch_;

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
  double BSx0_;
  double BSy0_;
  double BSz0_;
  double Beamsigmaz_;
  double Beamdxdz_;   
  double BeamWidthX_;
  double BeamWidthY_;

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
  float  DOFUnbiasedVertex_[nMaxtracks_];
  float  sumOfWeightsUnbiasedVertex_[nMaxtracks_];
  int    tracksUsedForVertexing_[nMaxtracks_];
  
  double dxyFromMyVertex_[nMaxtracks_];
  double dzFromMyVertex_[nMaxtracks_];

  double dxyErrorFromMyVertex_[nMaxtracks_];
  double dzErrorFromMyVertex_[nMaxtracks_];

  double IPTsigFromMyVertex_[nMaxtracks_];
  double IPLsigFromMyVertex_[nMaxtracks_];

  double dszFromMyVertex_[nMaxtracks_];
  int   hasRecVertex_[nMaxtracks_];
  int   isGoodTrack_[nMaxtracks_];

  // ---- directly histograms // ===> unbiased residuals
  
  // absolute residuals

  TH1F* a_dxyPhiResiduals[nMaxBins_];
  TH1F* a_dxyEtaResiduals[nMaxBins_];
  
  TH1F* a_dzPhiResiduals[nMaxBins_];
  TH1F* a_dzEtaResiduals[nMaxBins_];
  
  // normalized residuals

  TH1F* n_dxyPhiResiduals[nMaxBins_];
  TH1F* n_dxyEtaResiduals[nMaxBins_];
  
  TH1F* n_dzPhiResiduals[nMaxBins_];
  TH1F* n_dzEtaResiduals[nMaxBins_];
  
  // for the maps

  TH1F* a_dxyResidualsMap[nMaxBins_][nMaxBins_];
  TH1F* a_dzResidualsMap[nMaxBins_][nMaxBins_];
        				 				    
  TH1F* n_dxyResidualsMap[nMaxBins_][nMaxBins_];  				    
  TH1F* n_dzResidualsMap[nMaxBins_][nMaxBins_];

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

  // check probe 

  TH1F* h_probePt_;
  TH1F* h_probeEta_;
  TH1F* h_probePhi_;
  TH1F* h_probeChi2_;
  TH1F* h_probeNormChi2_;
  TH1F* h_probeCharge_;
  TH1F* h_probeQoverP_;
  TH1F* h_probedz_;
  TH1F* h_probedxy_;
  
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
  		    
  TH1F* h_recoVtxNtracks_; 
  TH1F* h_recoVtxChi2ndf_; 
  TH1F* h_recoVtxChi2Prob_;
  TH1F* h_recoVtxSumPt_;   

};

#endif
