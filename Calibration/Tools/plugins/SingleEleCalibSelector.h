#ifndef SingleEleCalibSelector_h
#define SingleEleCalibSelector_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include <functional>
#include <vector>
#include <map>


class SingleEleCalibSelector{

 public:
  
 typedef reco::GsfElectronCollection collection ;
 typedef reco::GsfElectronRef electron ;
 typedef std::vector<const reco::GsfElectron *> container;
 typedef container::const_iterator const_iterator;
 
 //! ctor
 SingleEleCalibSelector (const edm::ParameterSet& iConfig) ;
 //!dtor
 ~SingleEleCalibSelector () ;
 
 const_iterator begin() const { return selected_.begin(); }
 const_iterator end() const { return selected_.end(); }
  
 void select (edm::Handle<collection>, const edm::Event&, const edm::EventSetup&) ;
 
 private:

 container selected_ ; //! the selected collection

 edm::ESHandle<CaloTopology> theCaloTopology;   

 DetId findMaxHit (const std::vector<std::pair<DetId,float> > & v1,
		   const EBRecHitCollection* EBhits,
		   const EERecHitCollection* EEhits);
 
 double EnergyNxN(const std::vector<DetId> & vNxN,
		  const EBRecHitCollection* EBhits,
		  const EERecHitCollection* EEhits);
 
 double ESCOPinMin_, ESeedOPoutMin_, PinMPoutOPinMin_, E5x5OPoutMin_, E3x3OPinMin_, E3x3OE5x5Min_;
 double ESCOPinMax_, ESeedOPoutMax_, PinMPoutOPinMax_, E5x5OPoutMax_, E3x3OPinMax_, E3x3OE5x5Max_;
 edm::InputTag EBrecHitLabel_;
 edm::InputTag EErecHitLabel_;
};  

#endif


