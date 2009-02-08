
#ifndef RPCDigiFilter_h
#define RPCDigiFilter_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class RPCGeometry;

class RPCDigiFilter : public edm::EDProducer
{
public:

  explicit RPCDigiFilter(const edm::ParameterSet& ps);
  virtual ~RPCDigiFilter();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

  bool acceptDigiDB(std::pair<uint32_t, int> detStripPair,  const edm::EventSetup& iSetup);
  bool acceptDigiGeom(std::pair<uint32_t, int> detStripPair);


};

#endif

