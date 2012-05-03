#ifndef ggPFPhotons_h
#define ggPFPhotons_h
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoEgamma/EgammaTools/interface/ggPFClusters.h"
#include "RecoEgamma/EgammaTools/interface/ggPFESClusters.h"
#include "RecoEgamma/EgammaTools/interface/ggPFTracks.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include <memory>
using namespace edm;
using namespace std;
using namespace reco;
//PFPHOTON CLASS BY RISHI PATEL FOR HiggsToGammaGamma rpatel@cern.ch
class ggPFPhotons  {

 public:
  
  explicit ggPFPhotons(
		       reco::Photon phot,
		       edm::Handle<reco::PhotonCollection>& pfPhotons,
		       edm::Handle<reco::GsfElectronCollection>& pfElectrons,
		       edm::Handle<PFCandidateCollection>& pfCandidates,
		       edm::Handle<EcalRecHitCollection>& EBReducedRecHits,
		       edm::Handle<EcalRecHitCollection>& EEReducedRecHits,
		       edm::Handle<EcalRecHitCollection>& ESRecHits,
		       const CaloSubdetectorGeometry* geomBar,
		       const CaloSubdetectorGeometry* geomEnd,
		       edm::Handle<BeamSpot>& beamSpotHandle
		       );
  //add Geometry for Cluster Shape Calculation??
  virtual ~ggPFPhotons();
  bool MatchPFReco(){ return matchPFReco_;}
  bool isConv(){ return isConv_;}
  bool hasSLConv(){return hasSLConv_;}
  bool isPFEle(){return isPFEle_;}
  float PFPS1(){return PFPreShower1_;}
  float PFPS2(){return PFPreShower2_;}
  float MustE(){return EinMustache_;}
  float MustEOut(){return MustacheEOut_;}
  float MustEtOut(){return MustacheEtOut_;}
  float PFLowE(){return PFLowClusE_;}
  double PFdEta(){return dEtaLowestC_;}
  double PFdPhi(){return dPhiLowestC_;}
  double PFClusRMSTot(){return PFClPhiRMS_;}
  double PFClusRMSMust(){return PFClPhiRMSMust_;}
  std::vector<reco::CaloCluster>PFClusters(){return PFClusters_;}
  std::vector<reco::CaloCluster>PFClustersSCFP(){return PFSCFootprintClusters_;}
  //for Vertex
  std::pair<float, float> SLPoint();
  //fill PFCLuster Variables
  void fillPFClusters();
  //Local Containment Correction
  double getPFPhoECorr( std::vector<reco::CaloCluster>PFClusters, const GBRForest *ReaderLCEB, const GBRForest *ReaderLCEE);
  std::pair<double, double>CalcRMS(vector<reco::CaloCluster> PFClust, reco::Photon PFPhoton);
  std::vector<reco::CaloCluster>recoPhotonClusterLink( 		
						      reco::Photon phot, 
						      edm::Handle<PFCandidateCollection>& pfCandidates
						      );
  std::pair<double, double>SuperClusterSize(reco::Photon phot
					      );
  static void recoPhotonClusterLink(
							     reco::SuperCluster sc, 
							     std::vector<reco::PFCandidatePtr>&insideMust, 
							     std::vector<reco::PFCandidatePtr>&outsideMust,
							     edm::Handle<PFCandidateCollection>& pfCandidates,
							     double etabound,
							     double phibound
							     );
  static std::pair<double, double>SuperClusterSize(reco::SuperCluster sc,
						   Handle<EcalRecHitCollection>&   EBReducedRecHits,
						   Handle<EcalRecHitCollection>&   EEReducedRecHits,
						   const CaloSubdetectorGeometry* geomBar,
						   const CaloSubdetectorGeometry* geomEnd
					    );
  //for filling PFCluster Variables
 private:
  reco::Photon matchedPhot_;
  Handle<reco::PhotonCollection> pfPhotons_;
  Handle<reco::GsfElectronCollection> pfElectrons_;
  Handle<EcalRecHitCollection>  EBReducedRecHits_;
  Handle<EcalRecHitCollection>  EEReducedRecHits_;
  Handle<EcalRecHitCollection> ESRecHits_;
  const CaloSubdetectorGeometry* geomBar_;
  const CaloSubdetectorGeometry* geomEnd_;
  Handle<BeamSpot> beamSpotHandle_;
  reco::Photon PFPhoton_;
  reco::GsfElectron PFElectron_;
  bool matchPFReco_;
  bool isPFEle_;
  bool isConv_;
  bool hasSLConv_;
  std::vector<reco::CaloCluster>PFClusters_;
  std::vector<reco::CaloCluster>PFSCFootprintClusters_;
  float EinMustache_;
  float MustacheEOut_;
  float MustacheEtOut_;
  float PFPreShower1_;
  float PFPreShower2_;
  float PFLowClusE_;
  double dEtaLowestC_;
  double dPhiLowestC_;
  double PFClPhiRMS_;
  double PFClPhiRMSMust_;  
  double PFPhoLocallyCorrE_;
};
#endif
