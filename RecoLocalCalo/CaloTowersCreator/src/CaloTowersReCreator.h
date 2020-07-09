#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSRECREATOR_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSRECREATOR_H 1

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTowerTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoLocalCalo/CaloTowersCreator/interface/CaloTowersCreationAlgo.h"

/** \class CaloTowersReCreator
  *  
  */
class CaloTowersReCreator : public edm::stream::EDProducer<> {
public:
  explicit CaloTowersReCreator(const edm::ParameterSet& ps);
  ~CaloTowersReCreator() override {}
  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  double EBEScale, EEEScale, HBEScale, HESEScale;
  double HEDEScale, HOEScale, HF1EScale, HF2EScale;

private:
  CaloTowersCreationAlgo algo_;
  edm::EDGetTokenT<CaloTowerCollection> tok_calo_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_topo_;
  edm::ESGetToken<CaloTowerTopology, HcalRecNumberingRecord> tok_cttopo_;
  edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> tok_ctmap_;
  bool allowMissingInputs_;
};

#endif
