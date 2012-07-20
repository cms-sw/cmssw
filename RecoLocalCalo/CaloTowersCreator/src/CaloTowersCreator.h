#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATOR_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"


/** \class CaloTowersCreator
  *  
  * $Date: 2011/03/18 19:12:05 $
  * $Revision: 1.8 $
  * Original author: J. Mans - Minnesota
  */

// Now we allow for the creation of towers from 
// rejected hists as well: requested by the MET group
// for studies of the effect of noise clean up.

class CaloTowersCreator : public edm::EDProducer {
public:
  explicit CaloTowersCreator(const edm::ParameterSet& ps);
  virtual ~CaloTowersCreator() { }
  virtual void produce(edm::Event& e, const edm::EventSetup& c);
  double EBEScale, EEEScale, HBEScale, HESEScale;
  double HEDEScale, HOEScale, HF1EScale, HF2EScale;

private:

  static const std::vector<double>& getGridValues();
  template<typename COLL>
  void process(edm::Event& e, const edm::InputTag & label);

  CaloTowersCreationAlgo algo_;
  edm::InputTag hbheLabel_,hoLabel_,hfLabel_, hcalUpgradeLabel_;
  std::vector<edm::InputTag> ecalLabels_;
  bool allowMissingInputs_;


  // more compact flags: all HCAL are combined
  
  unsigned int theHcalAcceptSeverityLevel_;
  unsigned int theEcalAcceptSeverityLevel_;
  
  // flag to use recovered hits
  bool theRecoveredHcalHitsAreUsed_;
  bool theRecoveredEcalHitsAreUsed_;


  // paramaters for creating towers from rejected hits

  bool useRejectedHitsOnly_;
  unsigned int theHcalAcceptSeverityLevelForRejectedHit_;
  unsigned int theEcalAcceptSeverityLevelForRejectedHit_;
  bool useRejectedRecoveredHcalHits_;
  bool useRejectedRecoveredEcalHits_;

  edm::ESWatcher<HcalSeverityLevelComputerRcd> hcalSevLevelWatcher_;
  edm::ESWatcher<HcalChannelQualityRcd> hcalChStatusWatcher_;
  edm::ESWatcher<IdealGeometryRecord> caloTowerConstituentsWatcher_;
  bool upgrade_;
};

#endif
