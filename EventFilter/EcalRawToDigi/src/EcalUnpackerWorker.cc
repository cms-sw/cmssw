#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorker.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"


#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactoryOld.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactoryOld.h"

EcalUnpackerWorker::EcalUnpackerWorker(const edm::ParameterSet & conf){
  ///DCCDataUnpacker
  edm::ParameterSet DCCpset = conf.getParameter<edm::ParameterSet>("DCCDataUnpacker");
  edm::ParameterSet EEMpset = conf.getParameter<edm::ParameterSet>("ElectronicsMapper");

  unsigned int numbXtalTSamples_ = EEMpset.getParameter<unsigned int>("numbXtalTSamples");
  unsigned int numbTriggerTSamples_ = EEMpset.getParameter<unsigned int>("numbTriggerTSamples");
  
  if( numbXtalTSamples_ <6 || numbXtalTSamples_>64 || (numbXtalTSamples_-2)%4 ){
    edm::LogError("IncorrectConfiguration")<<"Unsuported number of xtal time samples : "<<numbXtalTSamples_
					   <<"\n Valid Number of xtal time samples are : 6,10,14,18,...,62"; 
  }
  
  if( numbTriggerTSamples_ !=1 && numbTriggerTSamples_ !=4 && numbTriggerTSamples_ !=8  ){
    edm::LogError("IncorrectConfiguration")<<"Unsuported number of trigger time samples : "<<numbTriggerTSamples_
					   <<"\n Valid number of trigger time samples are :  1, 4 or 8"; 
  }
  
  myMap_ = new EcalElectronicsMapper(numbXtalTSamples_, numbTriggerTSamples_);
  std::vector<int> oFl = DCCpset.getParameter<std::vector<int> >("orderedFedList");
  std::vector<int> oDl = DCCpset.getParameter<std::vector<int> >("orderedDCCIdList");
  bool readResult = myMap_->makeMapFromVectors(oFl,oDl);
  
  if(!readResult){
    edm::LogError("IncorrectConfiguration")<<"\n unable to configure EcalElectronicsMapper from vectors.";
  }
    
  unpacker_ = new DCCDataUnpacker(myMap_,
				  DCCpset.getParameter<bool>("headerUnpacking"),
				  DCCpset.getParameter<bool>("srpUnpacking"),
				  DCCpset.getParameter<bool>("tccUnpacking"),
				  DCCpset.getParameter<bool>("feUnpacking"),
				  DCCpset.getParameter<bool>("memUnpacking"),
				  DCCpset.getParameter<bool>("syncCheck"),
				  DCCpset.getParameter<bool>("feIdCheck"),
                                  DCCpset.getParameter<bool>("forceKeepFRData") 
                                  );
  edm::LogInfo("EcalRawToRecHit")
    <<"\n ECAL RawToDigi within RawToRecHit configuration:"
    <<"\n Header  unpacking is "<<DCCpset.getParameter<bool>("headerUnpacking")
    <<"\n SRP Bl. unpacking is "<<DCCpset.getParameter<bool>("srpUnpacking")
    <<"\n TCC Bl. unpacking is "<<DCCpset.getParameter<bool>("tccUnpacking")
    <<"\n FE  Bl. unpacking is "<<DCCpset.getParameter<bool>("feUnpacking")
    <<"\n MEM Bl. unpacking is "<<DCCpset.getParameter<bool>("memUnpacking")
    <<"\n sync check is "<<DCCpset.getParameter<bool>("syncCheck")
    <<"\n feID check is "<<DCCpset.getParameter<bool>("feIdCheck")
    <<"\n force keep FR data is "<<DCCpset.getParameter<bool>("forceKeepFRData")
    <<"\n";
  
  unpacker_->setEBDigisCollection(&productDigisEB);
  unpacker_->setEEDigisCollection(&productDigisEE);
  unpacker_->setDccHeadersCollection(&productDccHeaders); 
  unpacker_->setInvalidGainsCollection(&productInvalidGains); 
  unpacker_->setInvalidGainsSwitchCollection(&productInvalidGainsSwitch);
  unpacker_->setInvalidChIdsCollection(&productInvalidChIds);
  unpacker_->setInvalidEEGainsCollection(&productInvalidEEGains);
  unpacker_->setInvalidEEGainsSwitchCollection(&productInvalidEEGainsSwitch);
  unpacker_->setInvalidEEChIdsCollection(&productInvalidEEChIds);
  unpacker_->setEBSrFlagsCollection(&productEBSrFlags);
  unpacker_->setEESrFlagsCollection(&productEESrFlags);
  unpacker_->setEcalTpsCollection(&productTps);
  unpacker_->setEcalPSsCollection(&productPSs);
  unpacker_->setInvalidTTIdsCollection(&productInvalidTTIds);  
  unpacker_->setInvalidZSXtalIdsCollection(&productInvalidZSXtalIds);
  unpacker_->setInvalidBlockLengthsCollection(&productInvalidBlockLengths);
  unpacker_->setPnDiodeDigisCollection(&productPnDiodeDigis);
  unpacker_->setInvalidMemTtIdsCollection(& productInvalidMemTtIds);
  unpacker_->setInvalidMemBlockSizesCollection(& productInvalidMemBlockSizes);
  unpacker_->setInvalidMemChIdsCollection(& productInvalidMemChIds);
  unpacker_->setInvalidMemGainsCollection(& productInvalidMemGains);

  DCCDataUnpacker::silentMode_ = conf.getUntrackedParameter<bool> ("silentMode",true);

  edm::ParameterSet UncalibPSet = conf.getParameter<edm::ParameterSet>("UncalibRHAlgo");
  std::string UncaliComponentName = UncalibPSet.getParameter<std::string>("Type");
  UncalibWorker_ = EcalUncalibRecHitWorkerFactoryOld::get()->create(UncaliComponentName, UncalibPSet);

  edm::ParameterSet CalibPSet = conf.getParameter<edm::ParameterSet>("CalibRHAlgo");
  std::string CaliComponentName = CalibPSet.getParameter<std::string>("Type");
  CalibWorker_ = EcalRecHitWorkerFactoryOld::get()->create(CaliComponentName, CalibPSet);

}

EcalUnpackerWorker::~EcalUnpackerWorker(){
  //delete everything that is own by the EcalUnpackerWorker
  delete myMap_;
  delete unpacker_;
  delete UncalibWorker_;
  delete CalibWorker_;

}

void EcalUnpackerWorker::setHandles(const EcalUnpackerWorkerRecord & iRecord) {
  iRecord.getRecord<EcalRegionCablingRecord>().get(cabling);  
  //the mapping is set as long as the mapping is valid. 
  myMap_->setEcalElectronicsMapping(cabling->mapping()); 
}

void EcalUnpackerWorker::set(const edm::EventSetup & es) const {
  UncalibWorker_->set(es);
  CalibWorker_->set(es);
}

void EcalUnpackerWorker::write(edm::Event & e) const{
  //write the collection in the event as requested.
}

void EcalUnpackerWorker::update(const edm::Event & e)const{
  /// keep the event
  evt=&e;

  const bool reserveMem =true;

  /// DCCDataUnpacker
  productDigisEB.reset(new EBDigiCollection);
  productDigisEE.reset(new EEDigiCollection);
  productDccHeaders.reset(new EcalRawDataCollection);
  productInvalidGains.reset(new EBDetIdCollection);
  productInvalidGainsSwitch.reset(new EBDetIdCollection);
  productInvalidChIds.reset(new EBDetIdCollection);
  productInvalidEEGains.reset(new EEDetIdCollection);
  productInvalidEEGainsSwitch.reset(new EEDetIdCollection);
  productInvalidEEChIds.reset(new EEDetIdCollection);
  productEBSrFlags.reset(new EBSrFlagCollection);
  productEESrFlags.reset(new EESrFlagCollection);
  productTps.reset(new EcalTrigPrimDigiCollection);
  productPSs.reset(new EcalPSInputDigiCollection);
  productInvalidTTIds.reset(new EcalElectronicsIdCollection);
  productInvalidZSXtalIds.reset(new EcalElectronicsIdCollection);
  productInvalidBlockLengths.reset(new EcalElectronicsIdCollection);
  productPnDiodeDigis.reset(new EcalPnDiodeDigiCollection);
  productInvalidMemTtIds.reset(new EcalElectronicsIdCollection);
  productInvalidMemBlockSizes.reset(new EcalElectronicsIdCollection);
  productInvalidMemChIds.reset(new EcalElectronicsIdCollection);
  productInvalidMemGains.reset(new EcalElectronicsIdCollection);

  if (reserveMem){
    productDigisEB->reserve(1700);
    productDigisEE->reserve(1700); 
  }
    
 }


std::auto_ptr< EcalRecHitCollection > EcalUnpackerWorker::work(const uint32_t & index, const FEDRawDataCollection & rawdata)const{
//  MyWatcher watcher("Worker");
  LogDebug("EcalRawToRecHit|Worker")<<"is going to work on index: "<<index
				    <<"for fed Id: "<<EcalRegionCabling::fedIndex(index);
//<<watcher.lap();
//std::cout<<watcher.lap() <<std::endl;
  int fedIndex = EcalRegionCabling::fedIndex(index);

  const FEDRawData & fedData = rawdata.FEDData(fedIndex);

  //remember where the iterators were before unpacking
   LogDebug("EcalRawToRecHit|Worker")
    <<"size of digi collections before unpacking: "
    <<(*unpacker_->ebDigisCollection())->size()
    <<" "<<(*unpacker_->eeDigisCollection())->size();
  //  <<watcher.lap();

  EcalDigiCollection::const_iterator beginDigiEB =  (*unpacker_->ebDigisCollection())->end();
  EcalDigiCollection::const_iterator beginDigiEE =  (*unpacker_->eeDigisCollection())->end();
  
  //###### get the digi #######
  // unpack first
  int smId =0;
  int length = fedData.size();
  if ( length >= EMPTYEVENTSIZE ){
    if(myMap_->setActiveDCC(fedIndex)){
      smId = myMap_->getActiveSM();
      uint64_t * pData = (uint64_t *)(fedData.data());
       LogDebug("EcalRawToRecHit|Worker")<<"calling the unpacker: "<<length<<" "<<smId<<" "<<fedIndex ;
//					      <<watcher.lap();
      unpacker_->unpack( pData, static_cast<unsigned int>(length),smId,fedIndex);
       LogDebug("EcalRawToRecHit|Worker")<<"unpacking done." ;
//					      <<watcher.lap();
    }
    else{
      edm::LogInfo("EcalUnpackerWorker")<<"cannot set: "<<fedIndex<<" to be an active DCC.";
       LogDebug("EcalRawToRecHit|Worker")<<"cannot set: "<<fedIndex<<" to be an active DCC.";
      return std::auto_ptr< EcalRecHitCollection >(new EcalRecHitCollection);
    }
  }
  else {
    edm::LogInfo("EcalUnpackerWorker")<<"empty event on this FED: "<<fedIndex<<" length: "<<length;
     LogDebug("EcalRawToRecHit|Worker")<<"empty event on this FED: "<<fedIndex<<" length: "<<length;
    return std::auto_ptr< EcalRecHitCollection >(new EcalRecHitCollection);
  }

   LogDebug("EcalRawToRecHit|Worker")
    <<"size of digi collections after unpacking: "
    <<(*unpacker_->ebDigisCollection())->size()
    <<" "<<(*unpacker_->eeDigisCollection())->size() ;
//    <<watcher.lap();
  EcalDigiCollection::const_iterator endDigiEB = (*unpacker_->ebDigisCollection())->end();
  EcalDigiCollection::const_iterator endDigiEE = (*unpacker_->eeDigisCollection())->end();

  //collection for the rechits: uncalib and final
  std::auto_ptr< EcalRecHitCollection > ecalrechits( new EcalRecHitCollection );
  std::auto_ptr< EcalUncalibratedRecHitCollection > uncalibRecHits( new EcalUncalibratedRecHitCollection );
  
   LogDebug("EcalRawToRecHit|Worker")<<"going to work on EE rechits from: "<<endDigiEE-beginDigiEE<<" digis."
					    <<"\ngoing to work on EB rechits from: "<<endDigiEB-beginDigiEB<<" digis." ;
//					  <<watcher.lap();
  // EB
  //make the uncalibrated rechits on the fly
  if (beginDigiEB!=endDigiEB){
    work<EBDetId>(beginDigiEB, endDigiEB, uncalibRecHits, ecalrechits);
  }
   LogDebug("EcalRawToRecHit|Worker")<<uncalibRecHits->size()<<" uncalibrated rechit created so far\n"
					  <<ecalrechits->size()<<" rechit created so far." ;
//					  <<watcher.lap();
  
  // EE
  if (beginDigiEE!=endDigiEE){
    work<EEDetId>(beginDigiEE, endDigiEE, uncalibRecHits, ecalrechits);
  } 
   LogDebug("EcalRawToRecHit|Worker")<<uncalibRecHits->size()<<" uncalibrated rechit created eventually\n"
					  <<ecalrechits->size()<<" rechit created eventually" ;
//					  <<watcher.lap();

  return ecalrechits;
}
