#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATOR_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"


/** \class CaloTowersCreator
  *  
  * $Date: 2011/05/20 18:43:38 $
  * $Revision: 1.10 $
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

  CaloTowersCreationAlgo algo_;
  edm::InputTag hbheLabel_,hoLabel_,hfLabel_;
  std::vector<edm::InputTag> ecalLabels_;
  bool allowMissingInputs_;


  // more compact flags: all HCAL are combined
  
  unsigned int theHcalAcceptSeverityLevel_;
  std::vector<int> theEcalSeveritiesToBeExcluded_;
  
  // flag to use recovered hits
  bool theRecoveredHcalHitsAreUsed_;
  bool theRecoveredEcalHitsAreUsed_;


  // paramaters for creating towers from rejected hits

  bool useRejectedHitsOnly_;
  unsigned int theHcalAcceptSeverityLevelForRejectedHit_;
  //  for ECAL we have a list of problem flags
   std::vector<int> theEcalSeveritiesToBeUsedInBadTowers_; 



  // Flags wheteher to use recovered hits for production of
  // "bad towers". 
  // If the recoverd hits were already used for good towers,
  // these flags have no effect. 
  // Note: These flags have no effect on the default tower reconstruction.
  bool useRejectedRecoveredHcalHits_;
  bool useRejectedRecoveredEcalHits_;

  edm::ESWatcher<HcalSeverityLevelComputerRcd> hcalSevLevelWatcher_;
  edm::ESWatcher<HcalChannelQualityRcd> hcalChStatusWatcher_;
  edm::ESWatcher<IdealGeometryRecord> caloTowerConstituentsWatcher_;

};

#endif
