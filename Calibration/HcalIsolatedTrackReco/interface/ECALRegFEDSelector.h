#ifndef Calibration_ECALRegFEDSelector_h
#define Calibration_ECALRegFEDSelector_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"

class ECALRegFEDSelector : public edm::EDProducer {
public:
  ECALRegFEDSelector(const edm::ParameterSet&);
  ~ECALRegFEDSelector() override;
  std::unique_ptr<const EcalElectronicsMapping> ec_mapping;

  double delta_;
  bool fedSaved[1200];

  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_seed_;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
};

#endif
