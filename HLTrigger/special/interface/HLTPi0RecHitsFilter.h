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
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"


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


      virtual bool filter(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 // replace the 2 strings with 1 InputTag of form label:instance
   edm::InputTag barrelHits_;


 std::string pi0BarrelHits_;

 int gammaCandEtaSize_;
 int gammaCandPhiSize_;

 double clusSeedThr_;
 int clusEtaSize_;
 int clusPhiSize_;

 double selePtGammaOne_;
 double selePtGammaTwo_;
 double selePtPi0_;
 double seleMinvMaxPi0_;
 double seleMinvMinPi0_;
 double seleXtalMinEnergy_;
 int seleNRHMax_;
 //New criteria
 double seleS4S9GammaOne_;
 double seleS4S9GammaTwo_;
 double selePi0BeltDR_;
 double selePi0BeltDeta_;
 double selePi0Iso_;
 bool ParameterLogWeighted_;
 double ParameterX0_;
 double ParameterT0_barl_;
 double ParameterW0_;



 std::map<DetId, EcalRecHit> *recHitsEB_map;



};
