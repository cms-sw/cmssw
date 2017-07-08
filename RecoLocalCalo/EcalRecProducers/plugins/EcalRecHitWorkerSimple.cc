#include "RecoLocalCalo/EcalRecProducers/plugins/EcalRecHitWorkerSimple.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

EcalRecHitWorkerSimple::EcalRecHitWorkerSimple(const edm::ParameterSet&ps, edm::ConsumesCollector& c) :
  EcalRecHitWorkerBaseClass(ps,c)
{
    rechitMaker_ = new EcalRecHitSimpleAlgo();
    v_chstatus_ = 
        StringToEnumValue<EcalChannelStatusCode::Code>(ps.getParameter<std::vector<std::string> >("ChannelStatusToBeExcluded"));
    killDeadChannels_ = ps.getParameter<bool>("killDeadChannels");
    laserCorrection_ = ps.getParameter<bool>("laserCorrection");
	EBLaserMIN_ = ps.getParameter<double>("EBLaserMIN");
	EELaserMIN_ = ps.getParameter<double>("EELaserMIN");
	EBLaserMAX_ = ps.getParameter<double>("EBLaserMAX");
	EELaserMAX_ = ps.getParameter<double>("EELaserMAX");

	skipTimeCalib_=ps.getParameter<bool>("skipTimeCalib");

	// Traslate string representation of flagsMapDBReco into enum values 
	const edm::ParameterSet & p=ps.getParameter< edm::ParameterSet >("flagsMapDBReco");
	std::vector<std::string> recoflagbitsStrings = p.getParameterNames();
	v_DB_reco_flags_.resize(32); 

	for (unsigned int i=0;i!=recoflagbitsStrings.size();++i){
	  EcalRecHit::Flags recoflagbit = (EcalRecHit::Flags)
	    StringToEnumValue<EcalRecHit::Flags>(recoflagbitsStrings[i]);
	  std::vector<std::string> dbstatus_s =  
	    p.getParameter<std::vector<std::string> >(recoflagbitsStrings[i]);
	  std::vector<uint32_t> dbstatuses;
	  for (unsigned int j=0; j!= dbstatus_s.size(); ++j){
	    EcalChannelStatusCode::Code  dbstatus  = (EcalChannelStatusCode::Code)
	      StringToEnumValue<EcalChannelStatusCode::Code>(dbstatus_s[j]);
	    dbstatuses.push_back(dbstatus);
	  }

	  v_DB_reco_flags_[recoflagbit]=dbstatuses;
	}  

    flagmask_=0;
    flagmask_|=    0x1<<EcalRecHit::kNeighboursRecovered;
    flagmask_|=    0x1<<EcalRecHit::kTowerRecovered;
    flagmask_|=    0x1<<EcalRecHit::kDead;
    flagmask_|=    0x1<<EcalRecHit::kKilled;
    flagmask_|=    0x1<<EcalRecHit::kTPSaturated;
    flagmask_|=    0x1<<EcalRecHit::kL1SpikeFlag; 
}


void EcalRecHitWorkerSimple::set(const edm::EventSetup& es)
{
        es.get<EcalIntercalibConstantsRcd>().get(ical);

        if (!skipTimeCalib_){
            es.get<EcalTimeCalibConstantsRcd>().get(itime);
            es.get<EcalTimeOffsetConstantRcd>().get(offtime);
        }

        es.get<EcalADCToGeVConstantRcd>().get(agc);
        es.get<EcalChannelStatusRcd>().get(chStatus);
        if ( laserCorrection_ ) es.get<EcalLaserDbRecord>().get(laser);
}


bool
EcalRecHitWorkerSimple::run( const edm::Event & evt,
                const EcalUncalibratedRecHit& uncalibRH,
                EcalRecHitCollection & result )
{
    DetId detid=uncalibRH.id();
    
    EcalChannelStatusMap::const_iterator chit = chStatus->find(detid);
    EcalChannelStatusCode::Code  dbstatus = chit->getStatusCode();
    
    // check for channels to be excluded from reconstruction
    if ( !v_chstatus_.empty()) {
        
        std::vector<int>::const_iterator res = 
		    std::find( v_chstatus_.begin(), v_chstatus_.end(), dbstatus );
        if ( res != v_chstatus_.end() ) return false;
        
    }

	uint32_t flagBits = setFlagBits(v_DB_reco_flags_, dbstatus);

	float offsetTime = 0; // the global time phase
	const EcalIntercalibConstantMap& icalMap = ical->getMap();  
    if ( detid.subdetId() == EcalEndcap ) {
        rechitMaker_->setADCToGeVConstant( float(agc->getEEValue()) );
		if (!skipTimeCalib_) offsetTime = offtime->getEEValue();
    } else {
        rechitMaker_->setADCToGeVConstant( float(agc->getEBValue()) );
		if (!skipTimeCalib_) offsetTime = offtime->getEBValue();
    }

    // first intercalibration constants
    EcalIntercalibConstantMap::const_iterator icalit = icalMap.find(detid);
    EcalIntercalibConstant icalconst = 1;
    if( icalit!=icalMap.end() ) {
        icalconst = (*icalit);
    } else {
        edm::LogError("EcalRecHitError") << "No intercalib const found for xtal "
                                         << detid.rawId()
                                         << "! something wrong with EcalIntercalibConstants in your DB? ";
    }

    // get laser coefficient
    float lasercalib = 1.;
    if ( laserCorrection_ )	lasercalib = laser->getLaserCorrection( detid, evt.time());
	

    // get time calibration coefficient
    EcalTimeCalibConstant itimeconst = 0;

    if (!skipTimeCalib_){
        const EcalTimeCalibConstantMap & itimeMap = itime->getMap();  
        EcalTimeCalibConstantMap::const_iterator itime = itimeMap.find(detid);

        if( itime!=itimeMap.end() ) {
            itimeconst = (*itime);
        } else {
            edm::LogError("EcalRecHitError") << "No time calib const found for xtal "
                                             << detid.rawId()
                                             << "! something wrong with EcalTimeCalibConstants in your DB? ";
        }
          
    } 

    // make the rechit and put in the output collection, unless recovery has to take care of it
    if (! (flagmask_ & flagBits ) || !killDeadChannels_) {
        EcalRecHit myrechit( rechitMaker_->makeRecHit(uncalibRH, 
                                                      icalconst * lasercalib, 
                                                      (itimeconst + offsetTime), 
                                                      /*recoflags_ 0*/ 
                                                      flagBits) );
	
        if (detid.subdetId() == EcalBarrel && (lasercalib < EBLaserMIN_ || lasercalib > EBLaserMAX_)) 
            myrechit.setFlag(EcalRecHit::kPoorCalib);
        if (detid.subdetId() == EcalEndcap && (lasercalib < EELaserMIN_ || lasercalib > EELaserMAX_)) 
            myrechit.setFlag(EcalRecHit::kPoorCalib);
        result.push_back(myrechit);
	}

    return true;
}

// Take our association map of dbstatuses-> recHit flagbits and return the apporpriate flagbit word
uint32_t EcalRecHitWorkerSimple::setFlagBits(const std::vector<std::vector<uint32_t> >& map, 
					     const uint32_t& status  ){
  
  for (unsigned int i = 0; i!=map.size(); ++i){
    if (std::find(map[i].begin(), map[i].end(),status)!= map[i].end()) 
      return 0x1 << i;
  }

  return 0;
}


EcalRecHitWorkerSimple::~EcalRecHitWorkerSimple(){

  delete rechitMaker_;
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalRecHitWorkerFactory, EcalRecHitWorkerSimple, "EcalRecHitWorkerSimple" );
