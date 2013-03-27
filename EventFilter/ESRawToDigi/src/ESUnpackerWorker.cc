#include "EventFilter/ESRawToDigi/interface/ESUnpackerWorker.h"

#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"


ESUnpackerWorker::ESUnpackerWorker(const edm::ParameterSet & conf){
  
  edm::ParameterSet DCCpset = conf.getParameter<edm::ParameterSet>("DCCDataUnpacker");
  ESUnpacker_ = new ESUnpacker(DCCpset);
  
  edm::ParameterSet  RH = conf.getParameter<edm::ParameterSet>("RHAlgo");
  std::string componentType =  RH.getParameter<std::string>("Type");
  RHWorker_ = ESRecHitWorkerFactory::get()->create(componentType, RH);

}

ESUnpackerWorker::~ESUnpackerWorker(){
  delete ESUnpacker_;
  delete RHWorker_;
}

void ESUnpackerWorker::setHandles(const EcalUnpackerWorkerRecord & iRecord) {

}

void ESUnpackerWorker::set(const edm::EventSetup & es) const {
  RHWorker_->set(es);
}

void ESUnpackerWorker::write(edm::Event & e) const{

}

void ESUnpackerWorker::update(const edm::Event & e)const{
}


std::auto_ptr< EcalRecHitCollection > ESUnpackerWorker::work(const uint32_t & index, const FEDRawDataCollection & rawdata)const{
//  MyWatcher watcher("Worker");
  LogDebug("ESRawToRecHit|Worker")<<"is going to work on index: "<<index ;
//				  <<watcher.lap();

  int fedIndex = EcalRegionCabling::esFedIndex(index);


  const FEDRawData & fedData = rawdata.FEDData(fedIndex);

  //###### get the digi #######
  ESRawDataCollection productDCC;
  ESLocalRawDataCollection productKCHIP;
  ESDigiCollection productdigis;

  ESUnpacker_->interpretRawData(fedIndex, fedData, productDCC, productKCHIP, productdigis);
  
  LogDebug("ESRawToRecHit|Worker")<<"unpacked "<<productdigis.size()<<" digis" ;
//				  <<watcher.lap();


  //then make rechits
  ESDigiCollection::const_iterator beginDigiES = productdigis.begin();
  ESDigiCollection::const_iterator endDigiES = productdigis.end();

  std::auto_ptr< EcalRecHitCollection > ecalrechits( new EcalRecHitCollection );

  ecalrechits->reserve(productdigis.size());
  
  ESDigiCollection::const_iterator esIt=beginDigiES;
  for (;esIt!=endDigiES;++esIt){
    RHWorker_->run( esIt, *ecalrechits );
  }

  LogDebug("ESRawToRecHit|Worker")<<" made : "<<ecalrechits->size()<<" es rechits" ;
//				  <<watcher.lap();

  return ecalrechits;
}

										   
