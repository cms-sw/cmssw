#ifndef CSCTriggerPrimitives_CSCDigiSuppressor_h
#define CSCTriggerPrimitives_CSCDigiSuppressor_h
#include <list>

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"


class CSCDigiSuppressor : public edm::EDProducer
{
public:
  explicit CSCDigiSuppressor(const edm::ParameterSet& ps);
  ~CSCDigiSuppressor() {}

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  void fillDigis(const CSCDetId & id, const std::list<int> & keyStrips,
                 const CSCStripDigiCollection & oldStripDigis,
                 CSCStripDigiCollection & newStripDigis);

  std::list<int>
  cfebsToRead(const CSCDetId & id, const std::list<int> & keyStrips) const;

  edm::InputTag theLCTTag;
  edm::InputTag theStripDigiTag;
};

#endif

