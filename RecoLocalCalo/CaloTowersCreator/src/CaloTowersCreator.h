#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATOR_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"


/** \class CaloTowersCreator
  *  
  * $Date: 2007/09/25 16:19:05 $
  * $Revision: 1.3 $
  * \author J. Mans - Minnesota
  */
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

  // For treatmaent of bad/anomalous cells
  // Values set in the configuration file and passed
  // to CaloTowersCreationAlgo
  //
  // from DB
  uint theHbheAcceptSevLevelDb_;
  uint theHfAcceptSevLevelDb_;
  uint theHoAcceptSevLevelDb_;
  uint theEcalAcceptSevLevelDb_;
  // from the RecHit
  uint theHbheAcceptSevLevelRecHit_;
  uint theHfAcceptSevLevelRecHit_;
  uint theHoAcceptSevLevelRecHit_;
  uint theEcalAcceptSevLevelRecHit_;
  // flag to use recovered hits
  bool theRecovHbheIsUsed_;
  bool theRecovHoIsUsed_;
  bool theRecovHfIsUsed_;
  bool theRecovEcalIsUsed_;



};

#endif
