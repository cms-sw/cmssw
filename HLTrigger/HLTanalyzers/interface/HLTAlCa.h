#ifndef HLTALCA_H 
#define HLTALCA_H 
 
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h" 
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h" 
#include "DataFormats/L1Trigger/interface/L1EmParticle.h" 
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h" 
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"  
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h" 
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h" 
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h" 

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h" 
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h" 
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h" 
#include <DataFormats/FEDRawData/interface/FEDNumbering.h> 

/// EgammaCoreTools 
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h" 

typedef std::vector<std::string> MyStrings;

/** \class HLTAlCa
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */

typedef std::map<DetId, EcalRecHit> RecHitsMap; 
 
// Less than operator for sorting EcalRecHits according to energy. 
class eecalRecHitLess : public std::binary_function<EcalRecHit, EcalRecHit, bool>  
{ 
 public: 
  bool operator()(EcalRecHit x, EcalRecHit y)  
  {  
    return (x.energy() > y.energy());  
  } 
}; 


class HLTAlCa {
public:
  HLTAlCa(); 

  void setup(const edm::ParameterSet& pSet, TTree* tree); 

  /** Analyze the Data */
  void analyze(const edm::Handle<EBRecHitCollection>              & ebrechits,
	       const edm::Handle<EERecHitCollection>              & eerechits,
               const edm::Handle<HBHERecHitCollection>            & hbherechits, 
               const edm::Handle<HORecHitCollection>              & horechits, 
               const edm::Handle<HFRecHitCollection>              & hfrechits, 
	       const edm::Handle<EBRecHitCollection>              & pi0ebrechits, 
               const edm::Handle<EERecHitCollection>              & pi0eerechits, 
	       const edm::Handle<l1extra::L1EmParticleCollection> & l1extemi,  
	       const edm::Handle<l1extra::L1EmParticleCollection> & l1extemn,  
	       const edm::Handle<l1extra::L1JetParticleCollection> & l1extjetc,   
	       const edm::Handle<l1extra::L1JetParticleCollection> & l1extjetf, 
	       const edm::Handle<l1extra::L1JetParticleCollection> & l1exttaujet, 
	       const edm::ESHandle< EcalElectronicsMapping >      & ecalmapping, 
	       const edm::ESHandle<CaloGeometry>                  & geoHandle, 
	       const edm::ESHandle<CaloTopology>                  & pTopology,  
	       const edm::ESHandle<L1CaloGeometry>                & l1CaloGeom,
	       TTree* tree);

private:

  // Internal variables for AlCa pi0
  edm::InputTag barrelHits_;
  edm::InputTag endcapHits_;
  std::string pi0BarrelHits_;
  std::string pi0EndcapHits_;
  int gammaCandEtaSize_;
  int gammaCandPhiSize_;
  double clusSeedThr_;
  int clusEtaSize_;
  int clusPhiSize_;
  double clusSeedThrEndCap_;
  double selePtGammaOne_;
  double selePtGammaTwo_;
  double selePtPi0_;
  double seleMinvMaxPi0_;
  double seleMinvMinPi0_;
  double seleXtalMinEnergy_;
  double selePtGammaEndCap_;
  double selePtPi0EndCap_;
  double seleMinvMaxPi0EndCap_;
  double seleMinvMinPi0EndCap_;
  int seleNRHMax_;
  double seleS4S9GammaOne_;
  double seleS4S9GammaTwo_;
  double seleS4S9GammaEndCap_;
  double selePi0BeltDR_;
  double selePi0BeltDeta_;
  double selePi0Iso_;
  bool ParameterLogWeighted_;
  double ParameterX0_;
  double ParameterT0_barl_;
  double ParameterT0_endc_;
  double ParameterT0_endcPresh_;
  double ParameterW0_;
  double selePi0IsoEndCap_;
  edm::InputTag l1IsolatedTag_;
  edm::InputTag l1NonIsolatedTag_;
  edm::InputTag l1SeedFilterTag_;
  std::vector<EBDetId> detIdEBRecHits; 
  std::vector<EcalRecHit> EBRecHits; 
  std::vector<EEDetId> detIdEERecHits; 
  std::vector<EcalRecHit> EERecHits; 
  double ptMinForIsolation_; 
  bool storeIsoClusRecHit_; 
  double ptMinForIsolationEndCap_; 
  bool useEndCapEG_;
  bool Jets_; 
  edm::InputTag CentralSource_;
  edm::InputTag ForwardSource_;
  edm::InputTag TauSource_;
  bool JETSdoCentral_ ;
  bool JETSdoForward_ ;
  bool JETSdoTau_ ;
  double Ptmin_jets_; 
  double Ptmin_taujets_; 
  double JETSregionEtaMargin_;
  double JETSregionPhiMargin_;
  int debug_; 
  bool first_; 
  double EMregionEtaMargin_;
  double EMregionPhiMargin_;
  std::map<std::string,double> providedParameters;  
  std::vector<int> FEDListUsed; //by regional objects.  
  std::vector<int> FEDListUsedBarrel; 
  std::vector<int> FEDListUsedEndcap; 
  bool RegionalMatch_;
  double ptMinEMObj_ ; 
  bool doSelForEtaBarrel_; 
  double selePtGammaEta_;
  double selePtEta_;
  double seleS4S9GammaEta_; 
  double seleMinvMaxEta_; 
  double seleMinvMinEta_; 
  double ptMinForIsolationEta_; 
  double seleIsoEta_; 
  double seleEtaBeltDR_; 
  double seleEtaBeltDeta_; 
  bool storeIsoClusRecHitEta_;
  bool removePi0CandidatesForEta_; 
  double massLowPi0Cand_; 
  double massHighPi0Cand_; 

  EcalElectronicsMapping* TheMapping;

  const CaloSubdetectorGeometry *geometry_eb;
  const CaloSubdetectorGeometry *geometry_ee;
  const CaloSubdetectorGeometry *geometry_es;
  const CaloSubdetectorTopology *topology_eb;
  const CaloSubdetectorTopology *topology_ee;
  PositionCalc posCalculator_;
  static const int MAXCLUS = 2000;
  static const int MAXPI0S = 200;
  long int nEBRHSavedTotal ; 
  long int nEERHSavedTotal ; 
  long int nEvtPassedTotal; 
  long int nEvtPassedEETotal; 
  long int nEvtPassedEBTotal; 
  long int nEvtProcessedTotal; 
  int nClusAll;  
  
  // Tree variables
  float ohHighestEnergyEERecHit, ohHighestEnergyEBRecHit;
  float ohHighestEnergyHBHERecHit, ohHighestEnergyHORecHit, ohHighestEnergyHFRecHit; 

  int Nalcapi0clusters;
  float *ptClusAll, *etaClusAll, *phiClusAll, *s4s9ClusAll;

  // input variables
  bool _Monte,_Debug;

  int evtCounter;

  std::vector<int> ListOfFEDS(double etaLow, double etaHigh, double phiLow,  
			      double phiHigh, double etamargin, double phimargin); 

  int convertSmToFedNumbBarrel(int ieta, int smId);

  void convxtalid(Int_t &nphi,Int_t &neta);

  int diff_neta_s(Int_t neta1, Int_t neta2);

  int diff_nphi_s(Int_t nphi1,Int_t nphi2);

};

#endif
