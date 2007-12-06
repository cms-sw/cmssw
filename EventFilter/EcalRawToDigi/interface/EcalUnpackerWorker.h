#ifndef EcalUnpackerWorker_H
#define EcalUnpackerWorker_H

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/EcalRawToDigiDev/interface/DCCDataUnpacker.h"
#include "EventFilter/EcalRawToDigiDev/interface/EcalElectronicsMapper.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/EcalRawToDigi/interface/MyWatcher.h"

class EcalUnpackerWorker {
 public:

  EcalUnpackerWorker(const edm::ParameterSet & conf);
  
  ~EcalUnpackerWorker();
  
  // the method that does it all
  //  std::auto_ptr<EcalRecHitCollection> work(const uint32_t & i, const FEDRawData&) const;
  std::auto_ptr<EcalRecHitCollection> work(const uint32_t & i, const FEDRawDataCollection&) const;
  
  // method to set things up once per event
  void update(const edm::Event & e) const;
  
  void write(edm::Event &e) const;

  void setHandles(const EcalUnpackerWorkerRecord & iRecord);
  
 private:

  mutable const edm::Event * evt;

  DCCDataUnpacker * unpacker_;

  EcalElectronicsMapper * myMap_;
  mutable std::auto_ptr<EBDigiCollection> productDigisEB;
  mutable std::auto_ptr<EEDigiCollection> productDigisEE;
  mutable std::auto_ptr<EcalRawDataCollection> productDccHeaders;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidGains;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidGainsSwitch;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidGainsSwitchStay;
  mutable std::auto_ptr< EBDetIdCollection> productInvalidChIds;
  mutable std::auto_ptr<EBSrFlagCollection> productEBSrFlags;
  mutable std::auto_ptr<EESrFlagCollection> productEESrFlags;
  mutable std::auto_ptr<EcalTrigPrimDigiCollection> productEBTps;
  mutable std::auto_ptr<EcalTrigPrimDigiCollection> productEETps;
  //  mutable std::auto_ptr<EcalTrigTowerDetIdCollection> productInvalidTTIds;
  //  mutable std::auto_ptr<EcalTrigTowerDetIdCollection> productInvalidBlockLengths;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidTTIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidBlockLengths;
  mutable std::auto_ptr<EcalPnDiodeDigiCollection> productPnDiodeDigis;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemTtIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemBlockSizes;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemChIds;
  mutable std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemGains;

  edm::ESHandle<EcalRegionCabling> cabling;  

  EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> * uncalibMaker_;
  edm::ESHandle<EcalPedestals> peds;
  edm::ESHandle<EcalGainRatios>  gains;
  edm::ESHandle<EcalWeightXtalGroups>  grps;
  edm::ESHandle<EcalTBWeights> wgts;

  EcalRecHitAbsAlgo * rechitMaker_;

  edm::ESHandle<EcalIntercalibConstants> ical;
  edm::ESHandle<EcalADCToGeVConstant> agc;
  edm::ESHandle<EcalLaserDbService> laser;

 public:

  
  template <class DID> void work(EcalDigiCollection::const_iterator & beginDigi,
				 EcalDigiCollection::const_iterator & endDigi,
				 std::auto_ptr<EcalUncalibratedRecHitCollection> & uncalibRecHits,
				 std::auto_ptr< EcalRecHitCollection > & EBrechits)const{
    MyWatcher watcher("<Worker>");
    LogDebug("EcalRawToRecHit|Worker")<<"ready to work on digis."<<watcher.lap();

    EcalDigiCollection::const_iterator itdg = beginDigi;
    /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"iterator check." ;
    EcalTBWeights::EcalTBWeightMap const & wgtsMap = wgts->getMap();
    /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"weight map check."<<watcher.lap();
    for(; itdg != endDigi; ++itdg) 
      {
	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"starting dealing with one digi." 
						<<watcher.lap();
	DID detid(itdg->id());
	unsigned int hashedIndex = detid.hashedIndex();
	// ### pedestal and gain first and groupid
	const EcalPedestals::Item* aped = 0;
	const EcalMGPAGainRatio* aGain = 0;
	const EcalXtalGroupId * gid = 0;
	if (detid.subdet()==EcalEndcap){
	  /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"EndCap id, getting pedestals, gains and group id.\n"
						    <<"detid: "<<detid<<"\n has hashed index: "<<hashedIndex
						  <<watcher.lap();
	  aped=&peds->endcap(hashedIndex);
	  aGain=&gains->endcap(hashedIndex);
	  gid=&grps->endcap(hashedIndex);}
	else {
	  /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"Barrel id, getting pedestals, gains and group id.\n"
						    <<"detid: "<<detid<<"\n has hashed index: "<<hashedIndex
						  <<watcher.lap();
	  aped=&peds->barrel(hashedIndex);
	  aGain=&gains->barrel(hashedIndex);
	  gid=&grps->barrel(hashedIndex);}
	
	double pedVec[3];
	pedVec[0]=aped->mean_x12;pedVec[1]=aped->mean_x6;pedVec[2]=aped->mean_x1;
	double gainRatios[3];
	gainRatios[0]=1.;gainRatios[1]=aGain->gain12Over6();gainRatios[2]=aGain->gain6Over1()*aGain->gain12Over6();
	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"peds and gains loaded.";

	// use a fake TDC iD for now until it become available in raw data
	EcalTBWeights::EcalTDCId tdcid(1);
	
	// now lookup the correct weights in the map
	EcalTBWeights::EcalTBWeightMap::const_iterator wit;
	wit = wgtsMap.find( std::make_pair(*gid,tdcid) );
	if( wit == wgtsMap.end() ) {
	  edm::LogError("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " << gid->id() << " and  EcalTDCId: " << tdcid
						  << "\n  skipping digi with id: " << detid
						  <<watcher.lap();
	  /*R*/ LogDebug("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " << gid->id() << " and  EcalTDCId: " << tdcid
						     << "\n  skipping digi with id: " << detid
						   <<watcher.lap();
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
	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"weights loaded.";

	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"creating an unaclibrated rechit."
						<<watcher.lap();
	uncalibRecHits->push_back( uncalibMaker_->makeRecHit(*itdg, pedVec, gainRatios, weights, chi2mat));
	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"created."
						<<watcher.lap();
      }
    
    LogDebug("EcalRawToRecHit|Worker")<<"looping over uncalib rechits created."
				      <<watcher.lap();

    if (DID::subdet()==EcalEndcap){ 
      rechitMaker_->setADCToGeVConstant(float(agc->getEEValue())); 
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"ADCtoGeV constant set in EE: "<<agc->getEEValue() 
					      <<watcher.lap(); 
    } 
    else{ 
      rechitMaker_->setADCToGeVConstant(float(agc->getEBValue())); 
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"ADCtoGeV constant set in EB: "<<agc->getEBValue() 
					      <<watcher.lap(); 
    } 

    //######### get the rechit #########
    const EcalIntercalibConstantMap& icalMap=ical->getMap();  
    for(EBUncalibratedRecHitCollection::const_iterator it  = uncalibRecHits->begin();it != uncalibRecHits->end(); ++it) {
      DID detid(it->id());
      //      if (detid.subdet()==EcalEndcap){
      //	rechitMaker_->setADCToGeVConstant(float(agc->getEEValue()));
      //	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"ADCtoGeV constant set in EE: "<<agc->getEEValue()
      //						<<watcher.lap();
      //      }
      //      else{
      //	rechitMaker_->setADCToGeVConstant(float(agc->getEBValue()));
      //	/*R*/ LogDebug("EcalRawToRecHit|Worker")<<"ADCtoGeV constant set in EB: "<<agc->getEBValue()
      //						<<watcher.lap();
      //      }
      // first intercalibration constants
      EcalIntercalibConstantMap::const_iterator icalit=icalMap.find(detid);
      EcalIntercalibConstant icalconst = 1;
      if( icalit!=icalMap.end() ){
	icalconst = (*icalit);
      } else {
	edm::LogError("EcalRecHitError") << "No intercalib const found for xtal " << detid<< "! something wrong with EcalIntercalibConstants in your DB? ";
	LogDebug("EcalRecHitError") << "No intercalib const found for xtal " << detid<< "! something wrong with EcalIntercalibConstants in your DB? ";
      }
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"intercalibration constant loaded."
					      <<watcher.lap();

      // get laser coefficient
      float lasercalib = laser->getLaserCorrection( detid, evt->time());
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"laser correction diode."
					      <<watcher.lap();
      
      // make the rechit and put in the output collection
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"creating a rechit."
					      <<watcher.lap();
      EBrechits->push_back(EcalRecHit( rechitMaker_->makeRecHit(*it, icalconst * lasercalib) ));
      /*R*/ LogDebug("EcalRawToRecHit|Worker")<<"created."
					      <<watcher.lap();
    }
  }

};


#endif
