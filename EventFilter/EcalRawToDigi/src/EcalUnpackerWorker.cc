#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"


#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


EcalUnpackerWorker::EcalUnpackerWorker(const edm::ParameterSet & conf){
  ///DCCDataUnpacker
  edm::ParameterSet DCCpset = conf.getParameter<edm::ParameterSet>("DCCDataUnpacker");
  edm::ParameterSet EEMpset = conf.getParameter<edm::ParameterSet>("ElectronicsMapper");

  uint numbXtalTSamples_ = EEMpset.getParameter<uint>("numbXtalTSamples");
  uint numbTriggerTSamples_ = EEMpset.getParameter<uint>("numbTriggerTSamples");
  
  if( numbXtalTSamples_ <6 || numbXtalTSamples_>64 || (numbXtalTSamples_-2)%4 ){
    edm::LogError("EcalRawToRecHit|Worker")<<"Unsuported number of xtal time samples : "<<numbXtalTSamples_
					   <<"\n Valid Number of xtal time samples are : 6,10,14,18,...,62"; 
  }
  
  if( numbTriggerTSamples_ !=1 && numbTriggerTSamples_ !=4 && numbTriggerTSamples_ !=8  ){
    edm::LogError("EcalRawToRecHit|Worker")<<"Unsuported number of trigger time samples : "<<numbTriggerTSamples_
					   <<"\n Valid number of trigger time samples are :  1, 4 or 8"; 
  }
  
  myMap_ = new EcalElectronicsMapper(numbXtalTSamples_, numbTriggerTSamples_);
  std::vector<int> oFl = DCCpset.getParameter<std::vector<int> >("orderedFedList");
  std::vector<int> oDl = DCCpset.getParameter<std::vector<int> >("orderedDCCIdList");
  bool readResult = myMap_->makeMapFromVectors(oFl,oDl);
  
  if(!readResult){
    edm::LogError("EcalRawToRecHit|Worker")<<"\n unable to read file : "
					   <<conf.getParameter<std::string>("DCCMapFile");
  }
    
  unpacker_ = new DCCDataUnpacker(myMap_,
				  DCCpset.getParameter<bool>("headerUnpacking"),
				  DCCpset.getParameter<bool>("srpUnpacking"),
				  DCCpset.getParameter<bool>("tccUnpacking"),
				  DCCpset.getParameter<bool>("feUnpacking"),
				  DCCpset.getParameter<bool>("memUnpacking"),
				  DCCpset.getParameter<bool>("syncCheck"));
  
  unpacker_->setEBDigisCollection(&productDigisEB);
  unpacker_->setEEDigisCollection(&productDigisEE);
  unpacker_->setDccHeadersCollection(&productDccHeaders); 
  unpacker_->setInvalidGainsCollection(&productInvalidGains); 
  unpacker_->setInvalidGainsSwitchCollection(&productInvalidGainsSwitch);
  unpacker_->setInvalidGainsSwitchStayCollection(&productInvalidGainsSwitch);
  unpacker_->setInvalidChIdsCollection(&productInvalidChIds);
  unpacker_->setEBSrFlagsCollection(&productEBSrFlags);
  unpacker_->setEESrFlagsCollection(&productEESrFlags);
  unpacker_->setEBTpsCollection(&productEBTps);
  unpacker_->setEETpsCollection(&productEETps);
  unpacker_->setInvalidTTIdsCollection(&productInvalidTTIds);
  unpacker_->setInvalidBlockLengthsCollection(&productInvalidBlockLengths);
  unpacker_->setPnDiodeDigisCollection(&productPnDiodeDigis);
  unpacker_->setInvalidMemTtIdsCollection(& productInvalidMemTtIds);
  unpacker_->setInvalidMemBlockSizesCollection(& productInvalidMemBlockSizes);
  unpacker_->setInvalidMemChIdsCollection(& productInvalidMemChIds);
  unpacker_->setInvalidMemGainsCollection(& productInvalidMemGains);

  /// EcalUncalibRecHitRecWeightsAlgo
  uncalibMaker_ = new EcalUncalibRecHitRecWeightsAlgo<EBDataFrame>();
    
  /// EcalRecHitAbsAlgo
  rechitMaker_ = new EcalRecHitSimpleAlgo();
      
}

EcalUnpackerWorker::~EcalUnpackerWorker(){
  //free all the memory
  //wil matter if worker is re-created by eventsetup
}

void EcalUnpackerWorker::setHandles(const EcalUnpackerWorkerRecord & iRecord){
  
  iRecord.getRecord<EcalPedestalsRcd>().get(peds);
  iRecord.getRecord<EcalGainRatiosRcd>().get(gains);
  iRecord.getRecord<EcalWeightXtalGroupsRcd>().get(grps);
  iRecord.getRecord<EcalTBWeightsRcd>().get(wgts);

  iRecord.getRecord<EcalIntercalibConstantsRcd>().get(ical);
  iRecord.getRecord<EcalADCToGeVConstantRcd>().get(agc);
  iRecord.getRecord<EcalLaserDbRecord>().get(laser);

  iRecord.getRecord<EcalRegionCablingRecord>().get(cabling);

  //just to be sure it is done each event
  myMap_->setEcalElectronicsMapping(cabling->mapping());
}

void EcalUnpackerWorker::write(edm::Event & e) const{
  //write the collection in the event as requested.
}

void EcalUnpackerWorker::update(const edm::Event & e)const{
  /// keep the event
  evt=&e;

  const bool reserveMem =false;

  
  /// DCCDataUnpacker
  productDigisEB.reset(new EBDigiCollection);
  productDigisEE.reset(new EEDigiCollection);
  productDccHeaders.reset(new EcalRawDataCollection);
  productInvalidGains.reset(new EBDetIdCollection);
  productInvalidGainsSwitch.reset(new EBDetIdCollection);
  productInvalidGainsSwitchStay.reset(new EBDetIdCollection);
  productInvalidChIds.reset(new EBDetIdCollection);
  productEBSrFlags.reset(new EBSrFlagCollection);
  productEESrFlags.reset(new EESrFlagCollection);
  productEBTps.reset(new EcalTrigPrimDigiCollection);
  productEETps.reset(new EcalTrigPrimDigiCollection);
  //  productInvalidTTIds.reset(new EcalTrigTowerDetIdCollection);
  //  productInvalidBlockLengths.reset(new EcalTrigTowerDetIdCollection);
  productInvalidTTIds.reset(new EcalElectronicsIdCollection);
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
    
  /// EcalUncalibRecHitRecWeightsAlgo
    /*
      peds->update();
      gains->update();
      grps->update();
    */

 }


std::auto_ptr< EcalRecHitCollection > EcalUnpackerWorker::work(const uint32_t & index, const FEDRawDataCollection & rawdata)const{
  MyWatcher watcher("Worker");
  LogDebug("EcalRawToRecHit|Worker")<<"is going to work on index: "<<index
				    <<"for fed Id: "<<EcalRegionCabling::fedIndex(index)<<watcher.lap();

  int fedIndex = EcalRegionCabling::fedIndex(index);

  const FEDRawData & fedData = rawdata.FEDData(fedIndex);

  //remember where the iterators were before unpacking
  /*R*/ LogDebug("EcalRawToRecHit|Worker")
    <<"size of digi collections before unpacking: "
    <<(*unpacker_->ebDigisCollection())->size()
    <<" "<<(*unpacker_->eeDigisCollection())->size()
    <<watcher.lap();
  
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
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"calling the unpacker: "<<length<<" "<<smId<<" "<<fedIndex
					      <<watcher.lap();
      unpacker_->unpack( pData, static_cast<uint>(length),smId,fedIndex);
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"unpacking done."
					      <<watcher.lap();
    }
    else{
      edm::LogInfo("EcalUnpackerWorker")<<"cannot set: "<<fedIndex<<" to be an active DCC.";
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"cannot set: "<<fedIndex<<" to be an active DCC.";
      return std::auto_ptr< EcalRecHitCollection >(new EcalRecHitCollection);
    }
  }
  else {
    edm::LogInfo("EcalUnpackerWorker")<<"empty event on this FED: "<<fedIndex<<" length: "<<length;
    /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"empty event on this FED: "<<fedIndex<<" length: "<<length;
    return std::auto_ptr< EcalRecHitCollection >(new EcalRecHitCollection);
  }

  /*R*/ LogDebug("EcalRawToRecHit|Worker")
    <<"size of digi collections before unpacking: "
    <<(*unpacker_->ebDigisCollection())->size()
    <<" "<<(*unpacker_->eeDigisCollection())->size()
    <<watcher.lap();
  EcalDigiCollection::const_iterator endDigiEB = (*unpacker_->ebDigisCollection())->end();
  EcalDigiCollection::const_iterator endDigiEE = (*unpacker_->eeDigisCollection())->end();

  //collection for the rechits: uncalib and final
  std::auto_ptr< EcalRecHitCollection > ecalrechits( new EcalRecHitCollection );
  std::auto_ptr< EcalUncalibratedRecHitCollection > uncalibRecHits( new EcalUncalibratedRecHitCollection );
  
  /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"going to work on EE rechits from: "<<endDigiEE-beginDigiEE<<" digis."
					    <<"\ngoing to work on EB rechits from: "<<endDigiEB-beginDigiEB<<" digis."
					  <<watcher.lap();
  // EB
  //make the uncalibrated rechits on the fly
  if (beginDigiEB!=endDigiEB){
    work<EBDetId>(beginDigiEB, endDigiEB, uncalibRecHits, ecalrechits);
  }
  /*R*/ LogDebug("EcalRawToRecHit|Worker")<<uncalibRecHits->size()<<" uncalibrated rechit created so far\n"
					    <<ecalrechits->size()<<" rechit created so far."
					  <<watcher.lap();
  
  // EE
  if (beginDigiEE!=endDigiEE){
    work<EEDetId>(beginDigiEE, endDigiEE, uncalibRecHits, ecalrechits);
  } 
  /*R*/ LogDebug("EcalRawToRecHit|Worker")<<uncalibRecHits->size()<<" uncalibrated rechit created eventually\n"
					    <<ecalrechits->size()<<" rechit created eventually"
					  <<watcher.lap();

  return ecalrechits;
}



/*

void EcalUnpackerWorker::workoutEB(EBDigiCollection::const_iterator & beginDigi, 
EBDigiCollection::const_iterator & endDigi,
std::auto_ptr< EBUncalibratedRecHitCollection > & EBuncalibRecHits,
std::auto_ptr< EcalRecHitCollection > & EBrechits)const {
EBDigiCollection::const_iterator itdg = beginDigi;
//######### get the uncalibrated rechit #######  

EcalTBWeights::EcalTBWeightMap const & wgtsMap = wgts->getMap();
for(; itdg != endDigi; ++itdg) 
{
EBDetId detid(itdg->id());
unsigned int hashedIndex = detid.hashedIndex();
// ### pedestal first 
const EcalPedestals::Item& aped =  peds->barrel(hashedIndex);
double pedVec[3];
pedVec[0]=aped.mean_x12;pedVec[1]=aped.mean_x6;pedVec[2]=aped.mean_x1;

// ### then gains
const EcalMGPAGainRatio& aGain = gains->barrel(hashedIndex);
double gainRatios[3];
gainRatios[0]=1.;gainRatios[1]=aGain.gain12Over6();gainRatios[2]=aGain.gain6Over1()*aGain.gain12Over6();

// ### then groupId
// lookup group ID for this channel
const EcalXtalGroupId& gid = grps->barrel(hashedIndex);

// use a fake TDC iD for now until it become available in raw data
EcalTBWeights::EcalTDCId tdcid(1);

// now lookup the correct weights in the map
EcalTBWeights::EcalTBWeightMap::const_iterator wit;
wit = wgtsMap.find( std::make_pair(gid,tdcid) );
if( wit == wgtsMap.end() ) {
edm::LogError("EcalRawToRecHit|Worker") << "No weights found for EcalGroupId: " << gid.id() 
  << " and  EcalTDCId: " << tdcid
			    
			    << "\n  skipping digi with id: " << detid;
continue;
}
								const EcalWeightSet& wset = wit->second; // this is the EcalWeightSet
	
const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
const EcalWeightSet::EcalChi2WeightMatrix& mat3 = wset.getChi2WeightsBeforeGainSwitch();
const EcalWeightSet::EcalChi2WeightMatrix& mat4 = wset.getChi2WeightsAfterGainSwitch();

const EcalWeightSet::EcalWeightMatrix* weights[2];
weights[0]=&mat1;
weights[1]=&mat2;

const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];
chi2mat[0]=&mat3;
chi2mat[1]=&mat4;

EBuncalibRecHits->push_back( uncalibMaker_->makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat));
}
			    
			    
			    //######### get the rechit #########
			    const EcalIntercalibConstants::EcalIntercalibConstantMap& icalMap=ical->getMap();  
for(EBUncalibratedRecHitCollection::const_iterator it  = EBuncalibRecHits->begin();it != EBuncalibRecHits->end(); ++it) {

EBDetId detid(it->id());

// first intercalibration constants
EcalIntercalibConstants::EcalIntercalibConstantMap::const_iterator icalit=icalMap.find(detid);
EcalIntercalibConstants::EcalIntercalibConstant icalconst = 1;
if( icalit!=icalMap.end() ){
icalconst = icalit->second;
} else {
edm::LogError("EcalRawToRecHit|Worker") << "No intercalib const found for xtal " 
					   << detid
					      << "! something wrong with EcalIntercalibConstants in your DB? ";
}
						 
						 // get laser coefficient
						 float lasercalib = laser->getLaserCorrection( detid, evt->time());

// make the rechit and put in the output collection
EBrechits->push_back(EcalRecHit( rechitMaker_->makeRecHit(*it, icalconst * lasercalib) ));
}
}
*/
										   
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
EVENTSETUP_DATA_REG(EcalUnpackerWorker);
