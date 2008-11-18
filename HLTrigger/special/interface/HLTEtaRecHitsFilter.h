/**

 Description: Producer for EcalRecHits to be used for eta ECAL calibration. ECAL Barrel RecHits only.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yong Yang


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "TrackingTools/TrackAssociator/interface/TimerStack.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
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


#include <vector>
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"


#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"

#include "TLorentzVector.h"

//
// class declaration
//

typedef std::map<DetId, EcalRecHit> RecHitsMap;

// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitSort : public std::binary_function<EcalRecHit, EcalRecHit, bool> 
{
 public:
  bool operator()(EcalRecHit x, EcalRecHit y) 
    { 
      return (x.energy() > y.energy()); 
    }
};



class HLTEtaRecHitsFilter : public HLTFilter {
   public:


  explicit HLTEtaRecHitsFilter(const edm::ParameterSet&);
   	

      ~HLTEtaRecHitsFilter();

      
      int convertSmToFedNumbBarrel(int, int); 
      
      

      std::vector<int> ListOfFEDS(double etaLow, double etaHigh, double phiLow,
                                    double phiHigh, double etamargin, double phimargin);
	
	
      void convxtalid(int &, int &);
      int diff_neta_s(int, int);
      int diff_nphi_s(int, int);
         

      virtual bool filter(edm::Event &, const edm::EventSetup&);



   private:
      // ----------member data ---------------------------

 // replace the 2 strings with 1 InputTag of form label:instance
   edm::InputTag barrelHits_;
  edm::InputTag endcapHits_;
    

 std::string etaBarrelHits_;
 std::string etaEndcapHits_;

  
 

 int gammaCandEtaSize_;
 int gammaCandPhiSize_;

 double clusSeedThr_;
 int clusEtaSize_;
 int clusPhiSize_;


 double seleXtalMinEnergy_;
 int seleNRHMax_;

 bool ParameterLogWeighted_;
 double ParameterX0_;
 double ParameterT0_barl_;
 double ParameterT0_endc_;
 double ParameterT0_endcPresh_;
 double ParameterW0_;

  edm::InputTag l1IsolatedTag_;
  edm::InputTag l1NonIsolatedTag_;
  edm::InputTag l1SeedFilterTag_;



 int debug_; 
 bool first_; 
 double EMregionEtaMargin_;
 double EMregionPhiMargin_;
 
 double ptMinForIsolation_; 
 
 bool storeIsoClusRecHit_; 
 


 std::vector<int> FEDListUsed; ///by EM objects. 
 std::vector<int> FEDListUsedBarrel; 
 std::vector<int> FEDListUsedEndcap; 

 


 double ptMinEMObj_ ; 
 
 EcalElectronicsMapping* TheMapping;
 

 bool doMatchRegionL1EG_; 
 

 const CaloSubdetectorGeometry *geometry_eb;
 const CaloSubdetectorGeometry *geometry_ee;
 const CaloSubdetectorGeometry *geometry_es;
 const CaloSubdetectorTopology *topology_eb;
 const CaloSubdetectorTopology *topology_ee;
 
 std::map<std::string,double> providedParameters;  
 
 PositionCalc posCalculator_;
 
 static const int MAXCLUS = 2000;
 static const int MAXPI0S = 200;



 EBRecHitCollection::const_iterator itb;
 


 std::vector<EBDetId> detIdEBRecHits; 
 std::vector<EcalRecHit> EBRecHits; 
  
 // std::vector<EEDetId> detIdEERecHits; 
 // std::vector<EcalRecHit> EERecHits; 

 
 

 bool RegionalMatch_; 
 bool EGamma_; 
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
 
 
 bool store5x5SelectedEta_; 
 
 
 


};
