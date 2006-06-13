#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h"

L1RCTProducer::L1RCTProducer(const edm::ParameterSet& conf) 
  : src(conf.getParameter<string>("src"))
{
  //produces<JSCOutput>();
}

L1RCTProducer::~L1RCTProducer(){}

void L1RCTProducer::produce(edm::Event& e, const edm::EventSetup&)
{
  vector<vector<vector<unsigned short> > > barrel;
  vector<vector<unsigned short> > hf;
  
  /*
  edm::Handle<L1RCTEcal> ecal;
  edm::Handle<L1RCTHcal> hcal;
  emd::Handle<L1RCTHF> hf;
  e.getByLabel(ecal,src);
  e.getByLabel(hcal,src);
  e.getByLabel(hf,src);
  */
}
