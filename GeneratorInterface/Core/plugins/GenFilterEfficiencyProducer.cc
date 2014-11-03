#include "GeneratorInterface/Core/interface/GenFilterEfficiencyProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

GenFilterEfficiencyProducer::GenFilterEfficiencyProducer(const edm::ParameterSet& iConfig) :
  filterPath(iConfig.getParameter<std::string>("filterPath")),
  tns_(),
  thisProcess(),pathIndex(100000),
  numEventsPassPos_(0),
  numEventsPassNeg_(0),
  numEventsTotalPos_(0),
  numEventsTotalNeg_(0),
  sumpass_w_(0.),
  sumpass_w2_(0.),
  sumtotal_w_(0.),
  sumtotal_w2_(0.)
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

  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","",thisProcess));
  genEventInfoToken_ = consumes<GenEventInfoProduct>(edm::InputTag("generator",""));
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

  edm::Handle<edm::TriggerResults> trigR;
  iEvent.getByToken(triggerResultsToken_,trigR); 
  edm::Handle<GenEventInfoProduct>    genEventScale;
  iEvent.getByToken(genEventInfoToken_,genEventScale);
  if (!genEventScale.isValid()) return;
  double weight = genEventScale->weight();
  
  unsigned int nSize = (*trigR).size();
  // std::cout << "Number of paths in TriggerResults = " << nSize  << std::endl;
  if ( nSize >= pathIndex ) {

    if (!trigR->wasrun(pathIndex))return;
    if ( trigR->accept(pathIndex) ) { 
      sumpass_w_ += weight;
      sumpass_w2_+= weight*weight;
      
      sumtotal_w_ += weight;
      sumtotal_w2_+= weight*weight;

      if(weight > 0)
	{
	  numEventsPassPos_++;
	  numEventsTotalPos_++;
	}
      else
	{
	  numEventsPassNeg_++;
	  numEventsTotalNeg_++;
	}

    }
    else // if fail the filter
      {
	sumtotal_w_ += weight;
	sumtotal_w2_+= weight*weight;

	if(weight > 0)
	  numEventsTotalPos_++;
	else
	  numEventsTotalNeg_++;
      }
    //    std::cout << "Total events = " << numEventsTotal << " passed = " << numEventsPassed << std::endl;

  }
  
}

void
GenFilterEfficiencyProducer::beginLuminosityBlock(edm::LuminosityBlock const&, const edm::EventSetup&) {

  numEventsPassPos_=0;
  numEventsPassNeg_=0;
  numEventsTotalPos_=0;
  numEventsTotalNeg_=0;
  sumpass_w_=0;
  sumpass_w2_=0;
  sumtotal_w_=0;
  sumtotal_w2_=0;
 
}
void
GenFilterEfficiencyProducer::endLuminosityBlock(edm::LuminosityBlock const& iLumi, const edm::EventSetup&) {
}

void
GenFilterEfficiencyProducer::endLuminosityBlockProduce(edm::LuminosityBlock & iLumi, const edm::EventSetup&) {

  std::auto_ptr<GenFilterInfo> thisProduct(new GenFilterInfo(
							     numEventsPassPos_,
							     numEventsPassNeg_,
							     numEventsTotalPos_,
							     numEventsTotalNeg_,
							     sumpass_w_,
							     sumpass_w2_,
							     sumtotal_w_,
							     sumtotal_w2_
							     ));
  iLumi.put(thisProduct);
}
