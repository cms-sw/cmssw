#ifndef ExtraCode_CaloSimhitToRechit_CaloSimhitToRechitProducer_h
#define ExtraCode_CaloSimhitToRechit_CaloSimhitToRechitProducer_h


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"


#include <vector>

class CaloCellGeometry;

class CaloSimhitToRechitProducer : public edm::EDProducer
{
public:
  explicit CaloSimhitToRechitProducer(const edm::ParameterSet& iConfig);
  virtual ~CaloSimhitToRechitProducer();

  virtual void  produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
  
  double scaleEnergy (DetId id, double energy, const CaloCellGeometry& cell) const;

private:
  edm::InputTag mSource;                       // input PCaloHits source
  double mEnergyScale;
};





#endif
