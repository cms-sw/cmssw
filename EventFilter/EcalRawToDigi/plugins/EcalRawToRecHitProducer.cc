#include "EventFilter/EcalRawToDigi/plugins/EcalRawToRecHitProducer.h"

EcalRawToRecHitProducer::EcalRawToRecHitProducer(const edm::ParameterSet& iConfig)
{
  sourceTag_=iConfig.getParameter<edm::InputTag>("sourceTag");

  splitOutput_=iConfig.getParameter<bool>("splitOutput");
  if (splitOutput_){
    EBrechitCollection_=iConfig.getParameter<std::string>("EBrechitCollection");
    EErechitCollection_=iConfig.getParameter<std::string>("EErechitCollection");
    produces<EBRecHitCollection>(EBrechitCollection_);
    produces<EERecHitCollection>(EErechitCollection_);
    LogDebug("EcalRawToRecHit|Producer")<<"ready to create rechits from lazy getter: "<<sourceTag_
					<<"\n "<<((global_)?" global":"regional")<<" RAW->RecHit"
					<<"\n splitting in two collections"
					<<"\n EB instance: "<<EBrechitCollection_
					<<"\n EE instance: "<<EErechitCollection_;
  }
  else{
    produces<EcalRecHitCollection>();
    LogDebug("EcalRawToRecHit|Producer")<<"ready to create rechits from lazy getter: "<<sourceTag_
					<<"\n "<<((global_)?" global":"regional")<<" RAW->RecHit";
  }
}


EcalRawToRecHitProducer::~EcalRawToRecHitProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalRawToRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  MyWatcher watcher("Producer");
  LogDebug("EcalRawToRecHit|Producer")<<watcher.lap();

  //retrieve a refgetter
  edm::Handle<EcalRecHitRefGetter> rgetter;
  iEvent.getByLabel(sourceTag_ ,rgetter);
  LogDebug("EcalRawToRecHit|Producer")<<"ref getter retreived."
				      <<watcher.lap();
 
  if (splitOutput_){
    //prepare the output collection
    std::auto_ptr<EBRecHitCollection> EBrechits( new EBRecHitCollection );
    std::auto_ptr<EERecHitCollection> EErechits( new EERecHitCollection );
    //loop the refgetter
    uint iR=0;
    EcalRecHitRefGetter::const_iterator iRegion=rgetter->begin();
    for (;iRegion!=rgetter->end();++iRegion){
      LogDebug("EcalRawToRecHit|Producer")<<"looping over refgetter region: "<<iR<<watcher.lap();
      std::vector<EcalRecHit>::const_iterator iRecHit=iRegion->begin();
      for (;iRecHit!=iRegion->end();iRecHit++){
	DetId detid =iRecHit->id();
	//split barrel and endcap
	int EcalNum=detid.subdetId(); //1 stands for Barrel, 2 for endcaps
	LogDebug("EcalRawToRecHit|Producer")<<"subdetId is: "<<EcalNum;
	if (EcalNum==1) EBrechits->push_back(*iRecHit);
	else if (EcalNum==2) EErechits->push_back(*iRecHit);
	else {
	  edm::LogError("EcalRawToRecHit|Producer")<<" a subdetid is not recognized. recHit on :"<< iRecHit->id().rawId() 
						   <<" is lost.";
	}//subdetid
      }//loop over things in region
      LogDebug("EcalRawToRecHit|Producer")<<"looping over refgetter region: "<<iR++<<" done"
					  <<watcher.lap();
    }//loop over regions

    LogDebug("EcalRawToRecHit|Producer")<<EBrechits->size()<<" EB recHits to be put with instance: "<<EBrechitCollection_
					<<"\n"<<EErechits->size()<<" EE recHits to be put with instance: "<<EErechitCollection_
					<< watcher.lap();
    iEvent.put(EBrechits, EBrechitCollection_);
    iEvent.put(EErechits, EErechitCollection_);
    LogDebug("EcalRawToRecHit|Producer")<<"collections uploaded."
					<< watcher.lap();
  }
  else{
    //prepare the output collection
    std::auto_ptr< EcalRecHitCollection > rechits( new EcalRecHitCollection);
    //loop the refgetter
    EcalRecHitRefGetter::const_iterator iRegion=rgetter->begin();
    for (;iRegion!=rgetter->end();++iRegion){
      std::vector<EcalRecHit>::const_iterator iRecHit=iRegion->begin();
      for (;iRecHit!=iRegion->end();iRecHit++){
	rechits->push_back(*iRecHit);
      }//loop over things in region
    }//loop over regions
    LogDebug("EcalRawToRecHit|Producer")<<rechits->size()<<" rechits to be put."
					<< watcher.lap();
    iEvent.put(rechits);
    LogDebug("EcalRawToRecHit|Producer")<<"collections uploaded."
					<< watcher.lap();
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalRawToRecHitProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalRawToRecHitProducer::endJob() {
}
