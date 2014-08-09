#include "GeneratorInterface/Core/interface/GenFilterEfficiencyProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GenFilterEfficiencyProducer::GenFilterEfficiencyProducer(const edm::ParameterSet& iConfig) :
  filterPath(iConfig.getParameter<std::string>("filterPath")),
  tns_(),
  thisProcess(),pathIndex(100000),
  numEventsTotal(0),numEventsPassed(0),
  sumpass_pw(0),
  sumpass_pw2(0),
  sumpass_nw(0),
  sumpass_nw2(0),
  sumfail_pw(0),
  sumfail_pw2(0),
  sumfail_nw(0),
  sumfail_nw2(0)

{
   //now do what ever initialization is needed
  if (edm::Service<edm::service::TriggerNamesService>().isAvailable()) {
    // get tns pointer
    tns_ = edm::Service<edm::service::TriggerNamesService>().operator->();
    if (tns_!=0) {
      thisProcess = tns_->getProcessName();
      std::vector<std::string> theNames = tns_->getTrigPaths();
      for ( unsigned int i = 0; i < theNames.size(); i++ ) {
        if ( theNames[i] == filterPath ) { pathIndex = i; continue; }
      }
    }
    else
      edm::LogError("ServiceNotAvailable") << "TriggerNamesServive not available, no filter information stored";
  }

  produces<GenFilterInfo, edm::InLumi>(); 

}


GenFilterEfficiencyProducer::~GenFilterEfficiencyProducer()
{

} 


//
// member functions
//

// ------------ method called to for each event  ------------
void
GenFilterEfficiencyProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::InputTag theTrig("TriggerResults","",thisProcess);
  edm::Handle<edm::TriggerResults> trigR;
  iEvent.getByLabel(theTrig,trigR); 
  edm::Handle<GenEventInfoProduct>    genEventScale;
  double weight = 1;
  if (iEvent.getByLabel("generator", genEventScale)) 
    weight = genEventScale->weight();
  

  unsigned int nSize = (*trigR).size();
  // std::cout << "Number of paths in TriggerResults = " << nSize  << std::endl;
  if ( nSize >= pathIndex ) {
    if ( trigR->wasrun(pathIndex) ) { numEventsTotal++; }
    if ( trigR->accept(pathIndex) ) { 
      numEventsPassed++; 
      if(weight > 0)
	{
	  sumpass_pw  += weight;
	  sumpass_pw2 += weight*weight;
	}
      else
	{
	  sumpass_nw  += weight;
	  sumpass_nw2 += weight*weight;
	}

    }
    else // if fail the filter
      {
	if(weight > 0)
	  {
	    sumfail_pw  += weight;
	    sumfail_pw2 += weight*weight;
	  }
	else
	  {
	    sumfail_nw  += weight;
	    sumfail_nw2 += weight*weight;
	  }
      }
    //    std::cout << "Total events = " << numEventsTotal << " passed = " << numEventsPassed << std::endl;
  }
}

void
GenFilterEfficiencyProducer::beginLuminosityBlock(edm::LuminosityBlock const&, const edm::EventSetup&) {

  numEventsTotal = 0;
  numEventsPassed = 0;  
  sumpass_pw = 0;
  sumpass_pw2 = 0;
  sumpass_nw = 0;
  sumpass_nw2 = 0;
  sumfail_pw = 0;
  sumfail_pw2 = 0;
  sumfail_nw = 0;
  sumfail_nw2 = 0;
}
void
GenFilterEfficiencyProducer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, const edm::EventSetup&) {
}

void
GenFilterEfficiencyProducer::endLuminosityBlockProduce(edm::LuminosityBlock & iLumi, const edm::EventSetup&) {

  std::auto_ptr<GenFilterInfo> thisProduct(new GenFilterInfo(numEventsTotal,
							     numEventsPassed,
							     sumpass_pw,
							     sumpass_pw2,
							     sumpass_nw,
							     sumpass_nw2,
							     sumfail_pw,
							     sumfail_pw2,
							     sumfail_nw,
							     sumfail_nw2
							     ));
  iLumi.put(thisProduct);
}
