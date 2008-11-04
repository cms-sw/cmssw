/**

Description: Producer for EcalRecHits to be used for pi0 ECAL calibration. ECAL Barrel RecHits only.


 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vladimir Litvine


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


//
// class declaration
//

typedef std::map<DetId, EcalRecHit> RecHitsMap;

// Less than operator for sorting EcalRecHits according to energy.
class ecalRecHitLess : public std::binary_function<EcalRecHit, EcalRecHit, bool> 
{
 public:
  bool operator()(EcalRecHit x, EcalRecHit y) 
    { 
      return (x.energy() > y.energy()); 
    }
};



class HLTPi0RecHitsFilter : public HLTFilter {
   public:
      explicit HLTPi0RecHitsFilter(const edm::ParameterSet&);
      ~HLTPi0RecHitsFilter();

      
      int convertSmToFedNumbBarrel(int, int); 
      void convxtalid(int & , int &);
      int diff_neta_s(int,int);
      int diff_nphi_s(int,int);
      
      

      std::vector<int> ListOfFEDS(double etaLow, double etaHigh, double phiLow,
                                    double phiHigh, double etamargin, double phimargin);
	
	
	

      virtual bool filter(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 // replace the 2 strings with 1 InputTag of form label:instance
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
 //New criteria
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
 // double detaL1_;
 // double dphiL1_;
 // bool UseMatchedL1Seed_;
 double selePi0IsoEndCap_;


  edm::InputTag l1IsolatedTag_;
  edm::InputTag l1NonIsolatedTag_;
  edm::InputTag l1SeedFilterTag_;


  /// std::map<DetId, EcalRecHit> *recHitsEB_map;
  ///replace by two vectors. 

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
 
 

 std::vector<int> FEDListUsed; ///by regional objects.  ( em, jet, etc)

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
 






 EcalElectronicsMapping* TheMapping;
 

 const CaloSubdetectorGeometry *geometry_eb;
 const CaloSubdetectorGeometry *geometry_ee;
 const CaloSubdetectorGeometry *geometry_es;
 const CaloSubdetectorTopology *topology_eb;
 const CaloSubdetectorTopology *topology_ee;

 
 PositionCalc posCalculator_;
 
 static const int MAXCLUS = 2000;
 static const int MAXPI0S = 200;

};
