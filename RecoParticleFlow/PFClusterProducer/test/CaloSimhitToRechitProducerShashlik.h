#ifndef ExtraCode_CaloSimhitToRechit_CaloSimhitToRechitProducerShashlik_h
#define ExtraCode_CaloSimhitToRechit_CaloSimhitToRechitProducerShashlik_h


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"


#include <vector>

class CaloCellGeometry;

class CaloSimhitToRechitProducerShashlik : public edm::EDProducer
{
public:
  explicit CaloSimhitToRechitProducerShashlik(const edm::ParameterSet& iConfig);
  virtual ~CaloSimhitToRechitProducerShashlik();

  virtual void  produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
  
  double scaleEnergy (DetId id, double energy, const CaloCellGeometry& cell) const;

private:
  edm::InputTag mSource;                       // input PCaloHits source
  double mEnergyScale;
};





#endif
