#include "EventFilter/EcalRawToDigi/plugins/EcalRawToRecHitProducer.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"

EcalRawToRecHitProducer::EcalRawToRecHitProducer(const edm::ParameterSet& iConfig)
{
  lsourceTag_=iConfig.getParameter<edm::InputTag>("lazyGetterTag");
  sourceTag_=iConfig.getParameter<edm::InputTag>("sourceTag");

  splitOutput_=iConfig.getParameter<bool>("splitOutput");
  if (splitOutput_){
    EBrechitCollection_=iConfig.getParameter<std::string>("EBrechitCollection");
    EErechitCollection_=iConfig.getParameter<std::string>("EErechitCollection");
    produces<EBRecHitCollection>(EBrechitCollection_);
    produces<EERecHitCollection>(EErechitCollection_);
    LogDebug("EcalRawToRecHit|Producer")<<"ready to create rechits from lazy getter: "<<lsourceTag_
					<<"\n using region ref from: "<<sourceTag_
					<<"\n splitting in two collections"
					<<"\n EB instance: "<<EBrechitCollection_
					<<"\n EE instance: "<<EErechitCollection_;
  }
  else{
    rechitCollection_=iConfig.getParameter<std::string>("rechitCollection");
    produces<EcalRecHitCollection>(rechitCollection_);
    LogDebug("EcalRawToRecHit|Producer")<<"ready to create rechits from lazy getter: "<<lsourceTag_
					<<"\n using region ref from: "<<sourceTag_
					<<"\n not splitting the output collection.";
  }

  cleaningAlgo_=0;
  if (iConfig.exists("cleaningConfig")){
    const edm::ParameterSet & cleaning=iConfig.getParameter<edm::ParameterSet>("cleaningConfig");
    if (!cleaning.empty())
      cleaningAlgo_ = new EcalCleaningAlgo(cleaning);
  }
}


EcalRawToRecHitProducer::~EcalRawToRecHitProducer()
{
  if (cleaningAlgo_) delete cleaningAlgo_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalRawToRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

//  MyWatcher watcher("Producer");
//  LogDebug("EcalRawToRecHit|Producer")<<watcher.lap();

  //retrieve a lazygetter
  edm::Handle<EcalRecHitLazyGetter> lgetter;
  iEvent.getByLabel(lsourceTag_, lgetter);
  LogDebug("EcalRawToRecHit|Producer")<<"lazy getter retreived from: "<<lsourceTag_<<(lgetter.failedToGet()?" not valid ":"valid") ;
//				      <<watcher.lap();
  
  //retrieve a refgetter
  edm::Handle<EcalRecHitRefGetter> rgetter;
  iEvent.getByLabel(sourceTag_ ,rgetter);
  LogDebug("EcalRawToRecHit|Producer")<<"ref getter retreived from: "<<sourceTag_<<(rgetter.failedToGet()?" not valid ":"valid");
//				      <<watcher.lap();

 
  if (splitOutput_){
    //prepare the output collection
    std::auto_ptr<EBRecHitCollection> EBrechits( new EBRecHitCollection );
    std::auto_ptr<EERecHitCollection> EErechits( new EERecHitCollection );
    //loop the refgetter
    unsigned int iR=0;
    EcalRecHitRefGetter::const_iterator iRegion=rgetter->begin();
    EcalRecHitRefGetter::const_iterator iRegionEnd=rgetter->end();
    for (;iRegion!=iRegionEnd;++iRegion){
      LogDebug("EcalRawToRecHit|Producer")<<"looping over refgetter region: "<<iR;
//<<watcher.lap();
      lgetter->setEvent(iEvent);
      std::vector<EcalRecHit>::const_iterator iRecHit=lgetter->begin_record()+iRegion->start();
      std::vector<EcalRecHit>::const_iterator iRecHitEnd =lgetter->begin_record()+iRegion->finish();
      for (;iRecHit!=iRecHitEnd;iRecHit++){
	DetId detid =iRecHit->id();
	//split barrel and endcap
	int EcalNum=detid.subdetId(); //1 stands for Barrel, 2 for endcaps
	LogDebug("EcalRawToRecHit|Producer")<<"subdetId is: "<<EcalNum;
	if (EcalNum==1) EBrechits->push_back(*iRecHit);
	else if (EcalNum==2) EErechits->push_back(*iRecHit);
	else {
	  edm::LogError("IncorrectRecHit")<<" a subdetid is not recognized. recHit on :"<< iRecHit->id().rawId() 
						   <<" is lost.";
	}//subdetid
      }//loop over things in region
      LogDebug("EcalRawToRecHit|Producer")<<"looping over refgetter region: "<<iR++<<" done";
//					  <<watcher.lap();
    }//loop over regions

    LogDebug("EcalRawToRecHit|Producer")<<EBrechits->size()<<" EB recHits to be put with instance: "<<EBrechitCollection_
					<<"\n"<<EErechits->size()<<" EE recHits to be put with instance: "<<EErechitCollection_ ;
//					<< watcher.lap();
    
    // cleaning of anomalous signals, aka spikes
    // only doable once we have a "global" collection of hits
    if (cleaningAlgo_){
      EBrechits->sort();
      EErechits->sort();
      cleaningAlgo_->setFlags(*EBrechits);
      cleaningAlgo_->setFlags(*EErechits);
    }

    iEvent.put(EBrechits, EBrechitCollection_);
    iEvent.put(EErechits, EErechitCollection_);
    LogDebug("EcalRawToRecHit|Producer")<<"collections uploaded.";
//					<< watcher.lap();
  }
  else{
    //prepare the output collection
    std::auto_ptr< EcalRecHitCollection > rechits( new EcalRecHitCollection);
    //loop the refgetter
    unsigned int iR=0;
    EcalRecHitRefGetter::const_iterator iRegion=rgetter->begin();
    EcalRecHitRefGetter::const_iterator iRegionEnd=rgetter->end();
    for (;iRegion!=iRegionEnd;++iRegion){
      LogDebug("EcalRawToRecHit|Producer")<<"looping over refgetter region: "<<iR ;
//<<watcher.lap();
      lgetter->setEvent(iEvent);
      std::vector<EcalRecHit>::const_iterator iRecHit=lgetter->begin_record()+iRegion->start();
      std::vector<EcalRecHit>::const_iterator iRecHitEnd=lgetter->begin_record()+iRegion->finish();
      for (;iRecHit!=iRecHitEnd;iRecHit++){
	LogDebug("EcalRawToRecHit|Producer")<<"dereferencing rechit ref.";
	DetId detid =iRecHit->id();
	int EcalNum=detid.subdetId(); //1 stands for Barrel, 2 for endcaps
	LogDebug("EcalRawToRecHit|Producer")<<"subdetId is: "<<EcalNum;
	rechits->push_back(*iRecHit);
      }//loop over things in region
      LogDebug("EcalRawToRecHit|Producer")<<"looping over refgetter region: "<<iR++<<" done" ;
//<<watcher.lap();
    }//loop over regions

    if (cleaningAlgo_){
      rechits->sort();
      cleaningAlgo_->setFlags(*rechits);
    }
    LogDebug("EcalRawToRecHit|Producer")<<rechits->size()<<" rechits to be put." ;
//<< watcher.lap();
    iEvent.put(rechits,rechitCollection_);
    LogDebug("EcalRawToRecHit|Producer")<<"collections uploaded." ;
//					<< watcher.lap();
  }

}

