//
// $Id: EcalTrivialConditionRetriever.cc,v 1.47 2010/01/29 10:45:27 fra Exp $
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include <iostream>
#include <fstream>

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

//#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace edm;

EcalTrivialConditionRetriever::EcalTrivialConditionRetriever( const edm::ParameterSet&  ps)
{

  // initilize parameters used to produce cond DB objects
  adcToGeVEBConstant_ = ps.getUntrackedParameter<double>("adcToGeVEBConstant",0.035);
  adcToGeVEEConstant_ = ps.getUntrackedParameter<double>("adcToGeVEEConstant",0.060);

  intercalibConstantMeanMC_ = ps.getUntrackedParameter<double>("intercalibConstantMeanMC",1.0);
  intercalibConstantSigmaMC_ = ps.getUntrackedParameter<double>("intercalibConstantSigmaMC",0.0);

  intercalibConstantMean_ = ps.getUntrackedParameter<double>("intercalibConstantMean",1.0);
  intercalibConstantSigma_ = ps.getUntrackedParameter<double>("intercalibConstantSigma",0.0);
  intercalibErrorMean_ = ps.getUntrackedParameter<double>("IntercalibErrorMean",0.0);

  timeCalibConstantMean_ = ps.getUntrackedParameter<double>("timeCalibConstantMean",0.0);
  timeCalibConstantSigma_ = ps.getUntrackedParameter<double>("timeCalibConstantSigma",0.0);
  timeCalibErrorMean_ = ps.getUntrackedParameter<double>("timeCalibErrorMean",0.0);

  laserAlphaMean_  = ps.getUntrackedParameter<double>("laserAlphaMean",1.55);
  laserAlphaSigma_ = ps.getUntrackedParameter<double>("laserAlphaSigma",0);

  laserAPDPNTime1_ = (unsigned long)atoi( ps.getUntrackedParameter<std::string>("laserAPDPNTime1","1").c_str());
  laserAPDPNTime2_= (unsigned long)atoi( ps.getUntrackedParameter<std::string>("laserAPDPNTime2","0").c_str());
  laserAPDPNTime3_= (unsigned long)atoi( ps.getUntrackedParameter<std::string>("laserAPDPNTime3","0").c_str());

  laserAPDPNRefMean_ = ps.getUntrackedParameter<double>("laserAPDPNRefMean",1.0);
  laserAPDPNRefSigma_ = ps.getUntrackedParameter<double>("laserAPDPNRefSigma",0.0);

  laserAPDPNMean_ = ps.getUntrackedParameter<double>("laserAPDPNMean",1.0);
  laserAPDPNSigma_ = ps.getUntrackedParameter<double>("laserAPDPNSigma",0.0);

  localContCorrParameters_ = ps.getUntrackedParameter< std::vector<double> >("localContCorrParameters", std::vector<double>(0) );
  crackCorrParameters_ = ps.getUntrackedParameter< std::vector<double> >("crackCorrParameters", std::vector<double>(0) );
  energyCorrectionParameters_ = ps.getUntrackedParameter< std::vector<double> >("energyCorrectionParameters", std::vector<double>(0) );
  energyUncertaintyParameters_ = ps.getUntrackedParameter< std::vector<double> >("energyUncertaintyParameters", std::vector<double>(0) );

  EBpedMeanX12_ = ps.getUntrackedParameter<double>("EBpedMeanX12", 200.);
  EBpedRMSX12_  = ps.getUntrackedParameter<double>("EBpedRMSX12",  1.10);
  EBpedMeanX6_  = ps.getUntrackedParameter<double>("EBpedMeanX6",  200.);
  EBpedRMSX6_   = ps.getUntrackedParameter<double>("EBpedRMSX6",   0.90);
  EBpedMeanX1_  = ps.getUntrackedParameter<double>("EBpedMeanX1",  200.);
  EBpedRMSX1_   = ps.getUntrackedParameter<double>("EBpedRMSX1",   0.62);

  EEpedMeanX12_ = ps.getUntrackedParameter<double>("EEpedMeanX12", 200.);
  EEpedRMSX12_  = ps.getUntrackedParameter<double>("EEpedRMSX12",  2.50);
  EEpedMeanX6_  = ps.getUntrackedParameter<double>("EEpedMeanX6",  200.);
  EEpedRMSX6_   = ps.getUntrackedParameter<double>("EEpedRMSX6",   2.00);
  EEpedMeanX1_  = ps.getUntrackedParameter<double>("EEpedMeanX1",  200.);
  EEpedRMSX1_   = ps.getUntrackedParameter<double>("EEpedRMSX1",   1.40);

  gainRatio12over6_ = ps.getUntrackedParameter<double>("gainRatio12over6", 2.0);
  gainRatio6over1_  = ps.getUntrackedParameter<double>("gainRatio6over1",  6.0);

  getWeightsFromFile_ = ps.getUntrackedParameter<bool>("getWeightsFromFile",false);

  nTDCbins_ = 1;

  weightsForAsynchronousRunning_ = ps.getUntrackedParameter<bool>("weightsForTB",false);

  if (weightsForAsynchronousRunning_)
    {
      getWeightsFromFile_ = true; //override user request 
      //nTDCbins_ = 25;
      nTDCbins_ = 50; //modif Alex-21-07-2006
    }

  std::string path="CalibCalorimetry/EcalTrivialCondModules/data/";
  std::string weightType;
  std::ostringstream str;

  if (!weightsForAsynchronousRunning_)
    str << "_CMS.txt" ;
  else
    str << "_TB.txt" ;

  weightType = str.str();

  amplWeightsFile_ = ps.getUntrackedParameter<std::string>("amplWeightsFile",path+"ampWeights"+weightType);
  amplWeightsAftFile_ = ps.getUntrackedParameter<std::string>("amplWeightsAftFile",path+"ampWeightsAfterGainSwitch"+weightType);
  pedWeightsFile_ = ps.getUntrackedParameter<std::string>("pedWeightsFile",path+"pedWeights"+weightType);
  pedWeightsAftFile_ = ps.getUntrackedParameter<std::string>("pedWeightsAftFile",path+"pedWeightsAfterGainSwitch"+weightType);
  jittWeightsFile_ = ps.getUntrackedParameter<std::string>("jittWeightsFile",path+"timeWeights"+weightType);
  jittWeightsAftFile_ = ps.getUntrackedParameter<std::string>("jittWeightsAftFile",path+"timeWeightsAfterGainSwitch"+weightType);
  chi2MatrixFile_ = ps.getUntrackedParameter<std::string>("chi2MatrixFile",path+"chi2Matrix"+weightType);
  chi2MatrixAftFile_ = ps.getUntrackedParameter<std::string>("chi2MatrixAftFile",path+"chi2MatrixAfterGainSwitch"+weightType);

  amplWeights_.resize(nTDCbins_);  
  amplWeightsAft_.resize(nTDCbins_); 
  pedWeights_.resize(nTDCbins_);  
  pedWeightsAft_.resize(nTDCbins_); 
  jittWeights_.resize(nTDCbins_);  
  jittWeightsAft_.resize(nTDCbins_); 
  chi2Matrix_.resize(nTDCbins_);
  chi2MatrixAft_.resize(nTDCbins_);

  // default weights for MGPA shape after pedestal subtraction
  getWeightsFromConfiguration(ps);

  producedEcalPedestals_ = ps.getUntrackedParameter<bool>("producedEcalPedestals",true);
  producedEcalWeights_ = ps.getUntrackedParameter<bool>("producedEcalWeights",true);

  producedEcalGainRatios_ = ps.getUntrackedParameter<bool>("producedEcalGainRatios",true);
  producedEcalADCToGeVConstant_ = ps.getUntrackedParameter<bool>("producedEcalADCToGeVConstant",true);

  producedEcalMappingElectronics_ = ps.getUntrackedParameter<bool>("producedEcalMappingElectronics",true);
  mappingFile_ = ps.getUntrackedParameter<std::string>("mappingFile","");

  if ( producedEcalMappingElectronics_ ) {
    if ( mappingFile_ != "" ) { // if file provided read channel map
      setWhatProduced( this, &EcalTrivialConditionRetriever::getMappingFromConfiguration );
    } else { 
      setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalMappingElectronics );
    }
    findingRecord<EcalMappingElectronicsRcd>();
  }

  verbose_ = ps.getUntrackedParameter<int>("verbose", 0);

  //Tell Producer what we produce
  //setWhatproduce(this);
  if (producedEcalPedestals_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalPedestals );

  if (producedEcalWeights_) {
      setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalWeightXtalGroups );
      setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalTBWeights );
    }

  if (producedEcalGainRatios_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalGainRatios );

  if (producedEcalADCToGeVConstant_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalADCToGeVConstant );

  // intercalibration constants
  producedEcalIntercalibConstants_ = ps.getUntrackedParameter<bool>("producedEcalIntercalibConstants",true);
  intercalibConstantsFile_ = ps.getUntrackedParameter<std::string>("intercalibConstantsFile","") ;

  if (producedEcalIntercalibConstants_) { // user asks to produce constants
    if(intercalibConstantsFile_ != "") {  // if file provided read constants
        setWhatProduced (this, &EcalTrivialConditionRetriever::getIntercalibConstantsFromConfiguration ) ;
    } else { // set all constants to 1. or smear as specified by user
        setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalIntercalibConstants ) ;
    }
    findingRecord<EcalIntercalibConstantsRcd> () ;
  }
  // MC intercalibrations
  producedEcalIntercalibConstantsMC_ = ps.getUntrackedParameter<bool>("producedEcalIntercalibConstantsMC",true);

  if (producedEcalIntercalibConstantsMC_) { // user asks to produce constants
    setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalIntercalibConstantsMC ) ;
    findingRecord<EcalIntercalibConstantsMCRcd> () ;
  }

  // intercalibration constants
  producedEcalIntercalibErrors_ = ps.getUntrackedParameter<bool>("producedEcalIntercalibErrors",true);
  intercalibErrorsFile_ = ps.getUntrackedParameter<std::string>("intercalibErrorsFile","") ;

  if (producedEcalIntercalibErrors_) { // user asks to produce constants
    if(intercalibErrorsFile_ != "") {  // if file provided read constants
        setWhatProduced (this, &EcalTrivialConditionRetriever::getIntercalibErrorsFromConfiguration ) ;
    } else { // set all constants to 1. or smear as specified by user
        setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalIntercalibErrors ) ;
    }
    findingRecord<EcalIntercalibErrorsRcd> () ;
  }

  // time calibration constants
  producedEcalTimeCalibConstants_ = ps.getUntrackedParameter<bool>("producedEcalTimeCalibConstants",true);
  timeCalibConstantsFile_ = ps.getUntrackedParameter<std::string>("timeCalibConstantsFile","") ;

  if (producedEcalTimeCalibConstants_) { // user asks to produce constants
    if(timeCalibConstantsFile_ != "") {  // if file provided read constants
        setWhatProduced (this, &EcalTrivialConditionRetriever::getTimeCalibConstantsFromConfiguration ) ;
    } else { // set all constants to 1. or smear as specified by user
        setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalTimeCalibConstants ) ;
    }
    findingRecord<EcalTimeCalibConstantsRcd> () ;
  }

  // time calibration constants
  producedEcalTimeCalibErrors_ = ps.getUntrackedParameter<bool>("producedEcalTimeCalibErrors",true);
  timeCalibErrorsFile_ = ps.getUntrackedParameter<std::string>("timeCalibErrorsFile","") ;

  if (producedEcalTimeCalibErrors_) { // user asks to produce constants
    if(timeCalibErrorsFile_ != "") {  // if file provided read constants
        setWhatProduced (this, &EcalTrivialConditionRetriever::getTimeCalibErrorsFromConfiguration ) ;
    } else { // set all constants to 1. or smear as specified by user
        setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalTimeCalibErrors ) ;
    }
    findingRecord<EcalTimeCalibErrorsRcd> () ;
  }

  // cluster corrections
  producedEcalClusterLocalContCorrParameters_ = ps.getUntrackedParameter<bool>("producedEcalClusterLocalContCorrParameters", true);
  producedEcalClusterCrackCorrParameters_ = ps.getUntrackedParameter<bool>("producedEcalClusterCrackCorrParameters", true);
  producedEcalClusterEnergyCorrectionParameters_ = ps.getUntrackedParameter<bool>("producedEcalClusterEnergyCorrectionParameters", true);
  producedEcalClusterEnergyUncertaintyParameters_ = ps.getUntrackedParameter<bool>("producedEcalClusterEnergyUncertaintyParameters", true);
  if ( producedEcalClusterLocalContCorrParameters_ ) {
          setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalClusterLocalContCorrParameters );
          findingRecord<EcalClusterLocalContCorrParametersRcd>();
  }
  if ( producedEcalClusterCrackCorrParameters_ ) {
          setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalClusterCrackCorrParameters );
          findingRecord<EcalClusterCrackCorrParametersRcd>();
  }
  if ( producedEcalClusterEnergyCorrectionParameters_ ) {
          setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalClusterEnergyCorrectionParameters );
          findingRecord<EcalClusterEnergyCorrectionParametersRcd>();
  }
  if ( producedEcalClusterEnergyUncertaintyParameters_ ) {
          setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalClusterEnergyUncertaintyParameters );
          findingRecord<EcalClusterEnergyUncertaintyParametersRcd>();
  }

  // laser correction
  producedEcalLaserCorrection_ = ps.getUntrackedParameter<bool>("producedEcalLaserCorrection",true);
  if (producedEcalLaserCorrection_) { // user asks to produce constants
    // set all constants to 1. or smear as specified by user
    setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalLaserAlphas ) ;
    findingRecord<EcalLaserAlphasRcd> () ;
    setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalLaserAPDPNRatiosRef ) ;
    findingRecord<EcalLaserAPDPNRatiosRefRcd> () ;
    setWhatProduced (this, &EcalTrivialConditionRetriever::produceEcalLaserAPDPNRatios) ;
    findingRecord<EcalLaserAPDPNRatiosRcd> () ;
  }

  // channel status
  producedEcalChannelStatus_ = ps.getUntrackedParameter<bool>("producedEcalChannelStatus",true);
  channelStatusFile_ = ps.getUntrackedParameter<std::string>("channelStatusFile","");

  if ( producedEcalChannelStatus_ ) {
          if ( channelStatusFile_ != "" ) { // if file provided read channel map
                  setWhatProduced( this, &EcalTrivialConditionRetriever::getChannelStatusFromConfiguration );
          } else { // set all channels to working -- FIXME might be changed
                  setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalChannelStatus );
          }
          findingRecord<EcalChannelStatusRcd>();
  }
  // DQM channel status
  producedEcalDQMChannelStatus_ = ps.getUntrackedParameter<bool>("producedEcalDQMChannelStatus",true);
  if ( producedEcalDQMChannelStatus_ ) {
    setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalDQMChannelStatus );
    findingRecord<EcalDQMChannelStatusRcd>();
  }
  // DCS Tower status
  producedEcalDCSTowerStatus_ = ps.getUntrackedParameter<bool>("producedEcalDCSTowerStatus",true);
  if ( producedEcalDCSTowerStatus_ ) {
    setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalDCSTowerStatus );
    findingRecord<EcalDCSTowerStatusRcd>();
  }
  // DAQ Tower status
  producedEcalDAQTowerStatus_ = ps.getUntrackedParameter<bool>("producedEcalDAQTowerStatus",true);
  if ( producedEcalDAQTowerStatus_ ) {
    setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalDAQTowerStatus );
    findingRecord<EcalDAQTowerStatusRcd>();
  }
  // DQM Tower status
  producedEcalDQMTowerStatus_ = ps.getUntrackedParameter<bool>("producedEcalDQMTowerStatus",true);
  if ( producedEcalDQMTowerStatus_ ) {
    setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalDQMTowerStatus );
    findingRecord<EcalDQMTowerStatusRcd>();
  }

  // trigger channel status
  producedEcalTrgChannelStatus_ = ps.getUntrackedParameter<bool>("producedEcalTrgChannelStatus",true);
  trgChannelStatusFile_ = ps.getUntrackedParameter<std::string>("trgChannelStatusFile","");

  if ( producedEcalTrgChannelStatus_ ) {
          if ( trgChannelStatusFile_ != "" ) { // if file provided read channel map
                  setWhatProduced( this, &EcalTrivialConditionRetriever::getTrgChannelStatusFromConfiguration );
          } else { // set all channels to working -- FIXME might be changed
                  setWhatProduced( this, &EcalTrivialConditionRetriever::produceEcalTrgChannelStatus );
          }
          findingRecord<EcalTPGCrystalStatusRcd>();
  }

  //Tell Finder what records we find
  if (producedEcalPedestals_)  findingRecord<EcalPedestalsRcd>();

  if (producedEcalWeights_) {
      findingRecord<EcalWeightXtalGroupsRcd>();
      findingRecord<EcalTBWeightsRcd>();
    }

  if (producedEcalGainRatios_)  findingRecord<EcalGainRatiosRcd>();

  if (producedEcalADCToGeVConstant_)  findingRecord<EcalADCToGeVConstantRcd>();

}

EcalTrivialConditionRetriever::~EcalTrivialConditionRetriever()
{
}

//
// member functions
//
void
EcalTrivialConditionRetriever::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& rk,
                                               const edm::IOVSyncValue& iTime,
                                               edm::ValidityInterval& oValidity)
{
  if(verbose_>=1) std::cout << "EcalTrivialConditionRetriever::setIntervalFor(): record key = " << rk.name() << "\ttime: " << iTime.time().value() << std::endl;
  //For right now, we will just use an infinite interval of validity
  oValidity = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime() );
}

//produce methods
std::auto_ptr<EcalPedestals>
EcalTrivialConditionRetriever::produceEcalPedestals( const EcalPedestalsRcd& ) {
  std::auto_ptr<EcalPedestals>  peds = std::auto_ptr<EcalPedestals>( new EcalPedestals() );
  EcalPedestals::Item EBitem;
  EcalPedestals::Item EEitem;
  
  EBitem.mean_x1  = EBpedMeanX1_;
  EBitem.rms_x1   = EBpedRMSX1_;
  EBitem.mean_x6  = EBpedMeanX6_;
  EBitem.rms_x6   = EBpedRMSX6_;
  EBitem.mean_x12 = EBpedMeanX12_;
  EBitem.rms_x12  = EBpedRMSX12_;

  EEitem.mean_x1  = EEpedMeanX1_;
  EEitem.rms_x1   = EEpedRMSX1_;
  EEitem.mean_x6  = EEpedMeanX6_;
  EEitem.rms_x6   = EEpedRMSX6_;
  EEitem.mean_x12 = EEpedMeanX12_;
  EEitem.rms_x12  = EEpedRMSX12_;
  
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(iEta,iPhi))
	{
	  EBDetId ebdetid(iEta,iPhi);
	  peds->insert(std::make_pair(ebdetid.rawId(),EBitem));
	}
    }
  }
  
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eedetidpos(iX,iY,1);
	  peds->insert(std::make_pair(eedetidpos.rawId(),EEitem));
	}
      if(EEDetId::validDetId(iX,iY,-1))
	{
	  EEDetId eedetidneg(iX,iY,-1);
	  peds->insert(std::make_pair(eedetidneg.rawId(),EEitem));
	}
    }
  }

  //return std::auto_ptr<EcalPedestals>( peds );
  return peds;
}

std::auto_ptr<EcalWeightXtalGroups>
EcalTrivialConditionRetriever::produceEcalWeightXtalGroups( const EcalWeightXtalGroupsRcd& )
{
  std::auto_ptr<EcalWeightXtalGroups> xtalGroups = std::auto_ptr<EcalWeightXtalGroups>( new EcalWeightXtalGroups() );
  EcalXtalGroupId defaultGroupId(1);
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  //	  xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId(ieta) ); // define rings in eta
	  xtalGroups->setValue(ebid.rawId(), defaultGroupId ); // define rings in eta
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eedetidpos(iX,iY,1);
	  xtalGroups->setValue(eedetidpos.rawId(), defaultGroupId ); 
	}
      if(EEDetId::validDetId(iX,iY,-1))
	{
	  EEDetId eedetidneg(iX,iY,-1);
	  xtalGroups->setValue(eedetidneg.rawId(), defaultGroupId ); 
	}
    }
  }
  return xtalGroups;
}

std::auto_ptr<EcalIntercalibConstants>
EcalTrivialConditionRetriever::produceEcalIntercalibConstants( const EcalIntercalibConstantsRcd& )
{
  std::auto_ptr<EcalIntercalibConstants>  ical = std::auto_ptr<EcalIntercalibConstants>( new EcalIntercalibConstants() );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  ical->setValue( ebid.rawId(), intercalibConstantMean_ + r*intercalibConstantSigma_ );
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), intercalibConstantMean_ + r*intercalibConstantSigma_ );
	}
      if(EEDetId::validDetId(iX,iY,-1))
        {
	  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), intercalibConstantMean_ + r1*intercalibConstantSigma_ );
	}
    }
  }
  
  return ical;
}

std::auto_ptr<EcalIntercalibConstantsMC>
EcalTrivialConditionRetriever::produceEcalIntercalibConstantsMC( const EcalIntercalibConstantsMCRcd& )
{
  std::auto_ptr<EcalIntercalibConstantsMC>  ical = std::auto_ptr<EcalIntercalibConstantsMC>( new EcalIntercalibConstantsMC() );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  ical->setValue( ebid.rawId(), intercalibConstantMeanMC_ + r*intercalibConstantSigmaMC_ );
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), intercalibConstantMeanMC_ + r*intercalibConstantSigmaMC_ );
	}
      if(EEDetId::validDetId(iX,iY,-1))
        {
	  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), intercalibConstantMeanMC_ + r1*intercalibConstantSigmaMC_ );
	}
    }
  }
  
  return ical;
}

std::auto_ptr<EcalIntercalibErrors>
EcalTrivialConditionRetriever::produceEcalIntercalibErrors( const EcalIntercalibErrorsRcd& )
{
  std::auto_ptr<EcalIntercalibErrors>  ical = std::auto_ptr<EcalIntercalibErrors>( new EcalIntercalibErrors() );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  ical->setValue( ebid.rawId(), intercalibErrorMean_);
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), intercalibErrorMean_ );
	}
      if(EEDetId::validDetId(iX,iY,-1))
        {
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), intercalibErrorMean_ );
	}
    }
  }
  
  return ical;
}

std::auto_ptr<EcalTimeCalibConstants>
EcalTrivialConditionRetriever::produceEcalTimeCalibConstants( const EcalTimeCalibConstantsRcd& )
{
  std::auto_ptr<EcalTimeCalibConstants>  ical = std::auto_ptr<EcalTimeCalibConstants>( new EcalTimeCalibConstants() );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  ical->setValue( ebid.rawId(), timeCalibConstantMean_ + r*timeCalibConstantSigma_ );
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), timeCalibConstantMean_ + r*timeCalibConstantSigma_ );
	}
      if(EEDetId::validDetId(iX,iY,-1))
        {
	  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), timeCalibConstantMean_ + r1*timeCalibConstantSigma_ );
	}
    }
  }
  
  return ical;
}

std::auto_ptr<EcalTimeCalibErrors>
EcalTrivialConditionRetriever::produceEcalTimeCalibErrors( const EcalTimeCalibErrorsRcd& )
{
  std::auto_ptr<EcalTimeCalibErrors>  ical = std::auto_ptr<EcalTimeCalibErrors>( new EcalTimeCalibErrors() );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA ;++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  ical->setValue( ebid.rawId(), timeCalibErrorMean_);
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), timeCalibErrorMean_ );
	}
      if(EEDetId::validDetId(iX,iY,-1))
        {
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), timeCalibErrorMean_ );
	}
    }
  }
  
  return ical;
}

std::auto_ptr<EcalGainRatios>
EcalTrivialConditionRetriever::produceEcalGainRatios( const EcalGainRatiosRcd& )
{
  std::auto_ptr<EcalGainRatios> gratio = std::auto_ptr<EcalGainRatios>( new EcalGainRatios() );
  EcalMGPAGainRatio gr;
  gr.setGain12Over6( gainRatio12over6_ );
  gr.setGain6Over1( gainRatio6over1_ );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi))
	{
	  EBDetId ebid(ieta,iphi);
	  gratio->setValue( ebid.rawId(), gr );
	}
    }
  }
  
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eedetidpos(iX,iY,1);
	  gratio->setValue( eedetidpos.rawId(), gr );
	}
      if (EEDetId::validDetId(iX,iY,-1))
	{
	  EEDetId eedetidneg(iX,iY,-1);
	  gratio->setValue( eedetidneg.rawId(), gr );
	}
    }
  }
  
  return gratio;
}

std::auto_ptr<EcalADCToGeVConstant>
EcalTrivialConditionRetriever::produceEcalADCToGeVConstant( const EcalADCToGeVConstantRcd& )
{
  return std::auto_ptr<EcalADCToGeVConstant>( new EcalADCToGeVConstant(adcToGeVEBConstant_,adcToGeVEEConstant_) );
}

std::auto_ptr<EcalTBWeights>
EcalTrivialConditionRetriever::produceEcalTBWeights( const EcalTBWeightsRcd& )
{
  // create weights for the test-beam
  std::auto_ptr<EcalTBWeights> tbwgt = std::auto_ptr<EcalTBWeights>( new EcalTBWeights() );

  // create weights for each distinct group ID
  //  int nMaxTDC = 10;
//   for(int igrp=-EBDetId::MAX_IETA; igrp<=EBDetId::MAX_IETA; ++igrp) {
//     if(igrp==0) continue; 
  int igrp=1;
  for(int itdc=1; itdc<=nTDCbins_; ++itdc) {
    // generate random number
    //    double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
    
    // make a new set of weights
    EcalWeightSet wgt;
    //typedef std::vector< std::vector<EcalWeight> > EcalWeightSet::EcalWeightMatrix;
    EcalWeightSet::EcalWeightMatrix& mat1 = wgt.getWeightsBeforeGainSwitch();
    EcalWeightSet::EcalWeightMatrix& mat2 = wgt.getWeightsAfterGainSwitch();
    
//     if(verbose_>=1) {
//       std::cout << "initial size of mat1: " << mat1.size() << std::endl;
//       std::cout << "initial size of mat2: " << mat2.size() << std::endl;
//     }
    
    // generate random numbers to use as weights
    /**
       for(size_t i=0; i<3; ++i) {
       std::vector<EcalWeight> tv1, tv2;
       for(size_t j=0; j<10; ++j) {
       double ww = igrp*itdc*r + i*10. + j;
       //std::cout << "row: " << i << " col: " << j << " -  val: " << ww  << std::endl;
       tv1.push_back( EcalWeight(ww) );
       tv2.push_back( EcalWeight(100+ww) );
       }
       mat1.push_back(tv1);
       mat2.push_back(tv2);
       }
    **/
    
    // use values provided by user
    mat1.Place_in_row(amplWeights_[itdc-1],0,0);
    mat1.Place_in_row(pedWeights_[itdc-1],1,0);
    mat1.Place_in_row(jittWeights_[itdc-1],2,0);
    
    // wdights after gain switch 
    mat2.Place_in_row(amplWeightsAft_[itdc-1],0,0);
    mat2.Place_in_row(pedWeightsAft_[itdc-1],1,0);
    mat2.Place_in_row(jittWeightsAft_[itdc-1],2,0);
    
    // fill the chi2 matrcies with random numbers
    //    r = (double)std::rand()/( double(RAND_MAX)+double(1) );
    EcalWeightSet::EcalChi2WeightMatrix& mat3 = wgt.getChi2WeightsBeforeGainSwitch();
    EcalWeightSet::EcalChi2WeightMatrix& mat4 = wgt.getChi2WeightsAfterGainSwitch();
    mat3=chi2Matrix_[itdc-1];
    mat4=chi2MatrixAft_[itdc-1];
    
    //     for(size_t i=0; i<10; ++i) 
    //       {
    // 	mat3.push_back(chi2Matrix_[itdc-1][i]);
    // 	mat4.push_back(chi2MatrixAft_[itdc-1][i]);
    //       }
    //       std::vector<EcalWeight> tv1, tv2;
    //       for(size_t j=0; j<10; ++j) {
    // 	double ww = igrp*itdc*r + i*10. + j;
    // 	tv1.push_back( EcalWeight(1000+ww) );
    // 	tv2.push_back( EcalWeight(1000+100+ww) );
    //       }
    
  
    
    
//     if(verbose_>=1) {
//       std::cout << "group: " << igrp << " TDC: " << itdc 
// 		<< " mat1: " << mat1.size() << " mat2: " << mat2.size()
// 		<< " mat3: " << mat3.size() << " mat4: " << mat4.size()
// 		<< std::endl;
//     }
    
    // put the weight in the container
    tbwgt->setValue(std::make_pair(igrp,itdc), wgt);
  } 
  //   }
  return tbwgt;
}


// cluster functions/corrections
std::auto_ptr<EcalClusterLocalContCorrParameters>
EcalTrivialConditionRetriever::produceEcalClusterLocalContCorrParameters( const EcalClusterLocalContCorrParametersRcd &)
{
        std::auto_ptr<EcalClusterLocalContCorrParameters> ipar = std::auto_ptr<EcalClusterLocalContCorrParameters>( new EcalClusterLocalContCorrParameters() );
        for (size_t i = 0; i < localContCorrParameters_.size(); ++i ) {
                ipar->params().push_back( localContCorrParameters_[i] );
        }
        return ipar;
}
std::auto_ptr<EcalClusterCrackCorrParameters>
EcalTrivialConditionRetriever::produceEcalClusterCrackCorrParameters( const EcalClusterCrackCorrParametersRcd &)
{
        std::auto_ptr<EcalClusterCrackCorrParameters> ipar = std::auto_ptr<EcalClusterCrackCorrParameters>( new EcalClusterCrackCorrParameters() );
        for (size_t i = 0; i < crackCorrParameters_.size(); ++i ) {
                ipar->params().push_back( crackCorrParameters_[i] );
        }
        return ipar;
}
std::auto_ptr<EcalClusterEnergyCorrectionParameters>
EcalTrivialConditionRetriever::produceEcalClusterEnergyCorrectionParameters( const EcalClusterEnergyCorrectionParametersRcd &)
{
        std::auto_ptr<EcalClusterEnergyCorrectionParameters> ipar = std::auto_ptr<EcalClusterEnergyCorrectionParameters>( new EcalClusterEnergyCorrectionParameters() );
        for (size_t i = 0; i < energyCorrectionParameters_.size(); ++i ) {
                ipar->params().push_back( energyCorrectionParameters_[i] );
        }
        return ipar;
}
std::auto_ptr<EcalClusterEnergyUncertaintyParameters>
EcalTrivialConditionRetriever::produceEcalClusterEnergyUncertaintyParameters( const EcalClusterEnergyUncertaintyParametersRcd &)
{
        std::auto_ptr<EcalClusterEnergyUncertaintyParameters> ipar = std::auto_ptr<EcalClusterEnergyUncertaintyParameters>( new EcalClusterEnergyUncertaintyParameters() );
        for (size_t i = 0; i < energyUncertaintyParameters_.size(); ++i ) {
                ipar->params().push_back( energyUncertaintyParameters_[i] );
        }
        return ipar;
}


// laser records
std::auto_ptr<EcalLaserAlphas>
EcalTrivialConditionRetriever::produceEcalLaserAlphas( const EcalLaserAlphasRcd& )
{
  std::auto_ptr<EcalLaserAlphas>  ical = std::auto_ptr<EcalLaserAlphas>( new EcalLaserAlphas() );
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi)) {
            EBDetId ebid(ieta,iphi);
            double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
            ical->setValue( ebid, laserAlphaMean_ + r*laserAlphaSigma_ );
      }
    }
  }

   for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
     for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
       // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
       if (EEDetId::validDetId(iX,iY,1)) {
             double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
             EEDetId eedetidpos(iX,iY,1);
             ical->setValue( eedetidpos, laserAlphaMean_ + r*laserAlphaSigma_ );
       }

       if (EEDetId::validDetId(iX,iY,-1)) {
             double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
             EEDetId eedetidneg(iX,iY,-1);
             ical->setValue( eedetidneg, laserAlphaMean_ + r1*laserAlphaSigma_ );
       }
     }
   }
  
  return ical;
}


std::auto_ptr<EcalLaserAPDPNRatiosRef>
EcalTrivialConditionRetriever::produceEcalLaserAPDPNRatiosRef( const EcalLaserAPDPNRatiosRefRcd& )
{
  std::auto_ptr<EcalLaserAPDPNRatiosRef>  ical = std::auto_ptr<EcalLaserAPDPNRatiosRef>( new EcalLaserAPDPNRatiosRef() );
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi)) {
 	  EBDetId ebid(ieta,iphi);
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
 	  ical->setValue( ebid, laserAPDPNRefMean_ + r*laserAPDPNRefSigma_ );
      }
    }
  }

   for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
     for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
       // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
       if (EEDetId::validDetId(iX,iY,1)) {
 	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
 	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos, laserAPDPNRefMean_ + r*laserAPDPNRefSigma_ );
       }

       if (EEDetId::validDetId(iX,iY,-1)) {
 	  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
 	  EEDetId eedetidneg(iX,iY,-1);
 	  ical->setValue( eedetidneg, laserAPDPNRefMean_ + r1*laserAPDPNRefSigma_ );
       }
     }
   }
  
  return ical;
}


std::auto_ptr<EcalLaserAPDPNRatios>
EcalTrivialConditionRetriever::produceEcalLaserAPDPNRatios( const EcalLaserAPDPNRatiosRcd& )
{

  std::auto_ptr<EcalLaserAPDPNRatios>  ical = std::auto_ptr<EcalLaserAPDPNRatios>( new EcalLaserAPDPNRatios() );
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi)) {
  	  EBDetId ebid(ieta,iphi);
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );

	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair pairAPDPN;
	  pairAPDPN.p1 = laserAPDPNMean_ + r*laserAPDPNSigma_;
	  pairAPDPN.p2 = laserAPDPNMean_ + r*laserAPDPNSigma_;
	  pairAPDPN.p3 = laserAPDPNMean_ + r*laserAPDPNSigma_;
	  ical->setValue( ebid, pairAPDPN );
      }
     }
  }

   for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
     for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
       // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
       if (EEDetId::validDetId(iX,iY,1)) {
 	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
 	  EEDetId eedetidpos(iX,iY,1);

 	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair pairAPDPN;
 	  pairAPDPN.p1 = laserAPDPNMean_ + r*laserAPDPNSigma_;
 	  pairAPDPN.p2 = laserAPDPNMean_ + r*laserAPDPNSigma_;
 	  pairAPDPN.p3 = laserAPDPNMean_ + r*laserAPDPNSigma_;
 	  ical->setValue( eedetidpos, pairAPDPN );
       }

       if (EEDetId::validDetId(iX,iY,-1)) {
 	  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
 	  EEDetId eedetidneg(iX,iY,-1);

 	  EcalLaserAPDPNRatios::EcalLaserAPDPNpair pairAPDPN;
 	  pairAPDPN.p1 = laserAPDPNMean_ + r1*laserAPDPNSigma_;
 	  pairAPDPN.p2 = laserAPDPNMean_ + r1*laserAPDPNSigma_;
 	  pairAPDPN.p3 = laserAPDPNMean_ + r1*laserAPDPNSigma_;
 	  ical->setValue( eedetidneg, pairAPDPN );
       }
     }
   }

  EcalLaserAPDPNRatios::EcalLaserTimeStamp TimeStamp;
  //  for(int i=1; i<=92; i++){
  for(int i=0; i<92; i++){
    TimeStamp.t1 = Timestamp(laserAPDPNTime1_);
    if(laserAPDPNTime2_ == 0 ){ 
      TimeStamp.t2 = Timestamp(edm::Timestamp::endOfTime().value());
    } else {
      TimeStamp.t2 = Timestamp(laserAPDPNTime2_);
    }
    if(laserAPDPNTime3_ == 0 ){ 
      TimeStamp.t3 = Timestamp(edm::Timestamp::endOfTime().value());
    } else {
      TimeStamp.t3 = Timestamp(laserAPDPNTime3_);
    }

    ical->setTime( i, TimeStamp );
  }
  
  return ical;

}

  
void EcalTrivialConditionRetriever::getWeightsFromConfiguration(const edm::ParameterSet& ps)
{

  std::vector < std::vector<double> > amplwgtv(nTDCbins_);

  if (!getWeightsFromFile_ && nTDCbins_ == 1)
    {
      std::vector<double> vampl;
      //As default using simple 3+1 weights
      vampl.push_back( -0.33333 );
      vampl.push_back( -0.33333 );
      vampl.push_back( -0.33333 );
      vampl.push_back(  0. );
      vampl.push_back(  0. );
      vampl.push_back(  1. );
      vampl.push_back(  0. );
      vampl.push_back(  0. );
      vampl.push_back(  0. );
      vampl.push_back(  0. );
      amplwgtv[0]= ps.getUntrackedParameter< std::vector<double> >("amplWeights", vampl);
    }
  else if (getWeightsFromFile_)
    {
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading amplitude weights from file " << edm::FileInPath(amplWeightsFile_).fullPath().c_str() ;
      std::ifstream amplFile(edm::FileInPath(amplWeightsFile_).fullPath().c_str());
      int tdcBin=0;
      while (!amplFile.eof() && tdcBin < nTDCbins_) 
	{
	  for(int j = 0; j < 10; ++j) {
	    float ww;
	    amplFile >> ww;
	    amplwgtv[tdcBin].push_back(ww);
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
      //Read from file
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }

  
  for (int i=0;i<nTDCbins_;i++)
    {
      assert(amplwgtv[i].size() == 10);
      int j=0;
      for(std::vector<double>::const_iterator it = amplwgtv[i].begin(); it != amplwgtv[i].end(); ++it) 
	{
	  (amplWeights_[i])[j]=*it;
	  j++;
	}
    }
  

  std::vector < std::vector<double> > amplwgtvAftGain(nTDCbins_);

  if (!getWeightsFromFile_ && nTDCbins_ == 1 )
    {
      std::vector<double> vamplAftGain;
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  1. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      vamplAftGain.push_back(  0. );
      amplwgtvAftGain[0] = ps.getUntrackedParameter< std::vector<double> >("amplWeightsAftGain", vamplAftGain);
    }
  else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading amplitude weights aftre gain switch from file " << edm::FileInPath(amplWeightsAftFile_).fullPath().c_str() ;
      std::ifstream amplFile(edm::FileInPath(amplWeightsAftFile_).fullPath().c_str());
      int tdcBin=0;
      while (!amplFile.eof() && tdcBin < nTDCbins_) 
	{
	  for(int j = 0; j < 10; ++j) {
	    float ww;
	    amplFile >> ww;
	    amplwgtvAftGain[tdcBin].push_back(ww);
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }

  for (int i=0;i<nTDCbins_;i++)
    {
      assert(amplwgtvAftGain[i].size() == 10);
      int j=0;
      for(std::vector<double>::const_iterator it = amplwgtvAftGain[i].begin(); it != amplwgtvAftGain[i].end(); ++it) {
	(amplWeightsAft_[i])[j]=*it;
	j++;
      }
    }
      
  // default weights to reco amplitude w/o pedestal subtraction

  std::vector< std::vector<double> > pedwgtv(nTDCbins_);

  if (!getWeightsFromFile_ && nTDCbins_ == 1)
    {
      std::vector<double> vped;
      vped.push_back( 0.33333 );
      vped.push_back( 0.33333 );
      vped.push_back( 0.33333 );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      pedwgtv[0] = ps.getUntrackedParameter< std::vector<double> >("pedWeights", vped);
    }
  else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading pedestal weights from file " << edm::FileInPath(pedWeightsFile_).fullPath().c_str() ;
      std::ifstream pedFile(edm::FileInPath(pedWeightsFile_).fullPath().c_str());
      int tdcBin=0;
      while (!pedFile.eof() && tdcBin < nTDCbins_) 
	{
	  for(int j = 0; j < 10; ++j) {
	    float ww;
	    pedFile >> ww;
	    pedwgtv[tdcBin].push_back(ww);
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }

  for (int i=0;i<nTDCbins_;i++)
    {
      assert(pedwgtv[i].size() == 10);
      int j=0;
      for(std::vector<double>::const_iterator it = pedwgtv[i].begin(); it != pedwgtv[i].end(); ++it) {
	(pedWeights_[i])[j] = *it;
	j++;
      }
    }
  
  std::vector< std::vector<double> > pedwgtvaft(nTDCbins_);

  if (!getWeightsFromFile_ && nTDCbins_ == 1)
    {
      std::vector<double> vped;
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      vped.push_back( 0. );
      pedwgtvaft[0] = ps.getUntrackedParameter< std::vector<double> >("pedWeightsAft", vped);
    }
  else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading pedestal after gain switch weights from file " << edm::FileInPath(pedWeightsAftFile_).fullPath().c_str() ;
      std::ifstream pedFile(edm::FileInPath(pedWeightsAftFile_).fullPath().c_str());
      int tdcBin=0;
      while (!pedFile.eof() && tdcBin < nTDCbins_) 
	{
	  for(int j = 0; j < 10; ++j) {
	    float ww;
	    pedFile >> ww;
	    pedwgtvaft[tdcBin].push_back(ww);
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }
  
  for (int i=0;i<nTDCbins_;i++)
    {
      assert(pedwgtvaft[i].size() == 10);
      int j=0;
      for(std::vector<double>::const_iterator it = pedwgtvaft[i].begin(); it != pedwgtvaft[i].end(); ++it) {
	(pedWeightsAft_[i])[j]=*it;
	j++;
      }
    }
  
  
  
  // default weights to reco jitter

  std::vector< std::vector<double> > jittwgtv(nTDCbins_);

  if (!getWeightsFromFile_  && nTDCbins_ == 1 )
    {
      std::vector<double> vjitt;
      vjitt.push_back( 0.04066309 );
      vjitt.push_back( 0.04066309 );
      vjitt.push_back( 0.04066309 );
      vjitt.push_back( 0.000 );
      vjitt.push_back( 1.325176 );
      vjitt.push_back( -0.04997078 );
      vjitt.push_back( -0.504338 );
      vjitt.push_back( -0.5024844 );
      vjitt.push_back( -0.3903718 );
      vjitt.push_back( 0.000 );
      jittwgtv[0] = ps.getUntrackedParameter< std::vector<double> >("jittWeights", vjitt);
    }
  else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading jitter weights from file " << edm::FileInPath(jittWeightsFile_).fullPath().c_str() ;
      std::ifstream jittFile(edm::FileInPath(jittWeightsFile_).fullPath().c_str());
      int tdcBin=0;
      while (!jittFile.eof() && tdcBin < nTDCbins_) 
	{
	  for(int j = 0; j < 10; ++j) {
	    float ww;
	    jittFile >> ww;
	    jittwgtv[tdcBin].push_back(ww);
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }
  
  for (int i=0;i<nTDCbins_;i++)
    {
      assert(jittwgtv[i].size() == 10);
      int j=0;
      for(std::vector<double>::const_iterator it = jittwgtv[i].begin(); it != jittwgtv[i].end(); ++it) {
	(jittWeights_[i])[j]= *it;
	j++;
      }
    }
  
  std::vector< std::vector<double> > jittwgtvaft(nTDCbins_);

  if (!getWeightsFromFile_ && nTDCbins_ == 1)
    {
      std::vector<double> vjitt;
      vjitt.push_back( 0. );
      vjitt.push_back( 0. );
      vjitt.push_back( 0. );
      vjitt.push_back( 0. );
      vjitt.push_back( 1.097871 );
      vjitt.push_back( -0.04551035 );
      vjitt.push_back( -0.4159156 );
      vjitt.push_back( -0.4185352 );
      vjitt.push_back( -0.3367127 );
      vjitt.push_back( 0. );
      jittwgtvaft[0] = ps.getUntrackedParameter< std::vector<double> >("jittWeightsAft", vjitt);
    }
  else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading jitter after gain switch weights from file " << edm::FileInPath(jittWeightsAftFile_).fullPath().c_str() ;
      std::ifstream jittFile(edm::FileInPath(jittWeightsAftFile_).fullPath().c_str());
      int tdcBin=0;
      while (!jittFile.eof() && tdcBin < nTDCbins_) 
	{
	  for(int j = 0; j < 10; ++j) {
	    float ww;
	    jittFile >> ww;
	    jittwgtvaft[tdcBin].push_back(ww);
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }
  
  for (int i=0;i<nTDCbins_;i++)
    {
      assert(jittwgtvaft[i].size() == 10);
      int j=0;
      for(std::vector<double>::const_iterator it = jittwgtvaft[i].begin(); it != jittwgtvaft[i].end(); ++it) {
	(jittWeightsAft_[i])[j]= *it;
	j++;
      }
    }

   
   std::vector<  EcalWeightSet::EcalChi2WeightMatrix > chi2Matrix(nTDCbins_);
   if (!getWeightsFromFile_  && nTDCbins_ == 1 )
     {
       //        chi2Matrix[0].resize(10);
       //        for (int i=0;i<10;i++)
	 //	 chi2Matrix[0][i].resize(10);

       chi2Matrix[0](0,0) = 0.694371;
       chi2Matrix[0](0,1) = -0.305629;  
       chi2Matrix[0](0,2) = -0.305629;
       chi2Matrix[0](0,3) = 0.;
       chi2Matrix[0](0,4) = 0.;
       chi2Matrix[0](0,5) = 0.;
       chi2Matrix[0](0,6) = 0.;
       chi2Matrix[0](0,7) = 0.;
       chi2Matrix[0](0,8) = 0.;
       chi2Matrix[0](0,9) = 0.;
       chi2Matrix[0](1,0) = -0.305629;
       chi2Matrix[0](1,1) = 0.694371;
       chi2Matrix[0](1,2) = -0.305629;
       chi2Matrix[0](1,3) = 0.;
       chi2Matrix[0](1,4) = 0.;
       chi2Matrix[0](1,5) = 0.;
       chi2Matrix[0](1,6) = 0.;
       chi2Matrix[0](1,7) = 0.;
       chi2Matrix[0](1,8) = 0.;
       chi2Matrix[0](1,9) = 0.;
       chi2Matrix[0](2,0) = -0.305629;
       chi2Matrix[0](2,1) = -0.305629;
       chi2Matrix[0](2,2) = 0.694371;
       chi2Matrix[0](2,3) = 0.;
       chi2Matrix[0](2,4) = 0.;
       chi2Matrix[0](2,5) = 0.;
       chi2Matrix[0](2,6) = 0.;
       chi2Matrix[0](2,7) = 0.;
       chi2Matrix[0](2,8) = 0.;
       chi2Matrix[0](2,9) = 0.;
       chi2Matrix[0](3,0) = 0.;
       chi2Matrix[0](3,1) = 0.;
       chi2Matrix[0](3,2) = 0.;
       chi2Matrix[0](3,3) = 0.;
       chi2Matrix[0](3,4) = 0.;
       chi2Matrix[0](3,5) = 0.;
       chi2Matrix[0](3,6) = 0.;
       chi2Matrix[0](3,7) = 0.;
       chi2Matrix[0](3,8) = 0.;
       chi2Matrix[0](3,9) = 0.; 
       chi2Matrix[0](4,0) = 0.;
       chi2Matrix[0](4,1) = 0.;
       chi2Matrix[0](4,2) = 0.;
       chi2Matrix[0](4,3) = 0.;
       chi2Matrix[0](4,4) = 0.8027116;
       chi2Matrix[0](4,5) = -0.2517103;
       chi2Matrix[0](4,6) = -0.2232882;
       chi2Matrix[0](4,7) = -0.1716192;
       chi2Matrix[0](4,8) = -0.1239006;
       chi2Matrix[0](4,9) = 0.; 
       chi2Matrix[0](5,0) = 0.;
       chi2Matrix[0](5,1) = 0.;
       chi2Matrix[0](5,2) = 0.;
       chi2Matrix[0](5,3) = 0.;
       chi2Matrix[0](5,4) = -0.2517103;
       chi2Matrix[0](5,5) = 0.6528964;
       chi2Matrix[0](5,6) = -0.2972839;
       chi2Matrix[0](5,7) = -0.2067162;
       chi2Matrix[0](5,8) = -0.1230729;
       chi2Matrix[0](5,9) = 0.;
       chi2Matrix[0](6,0) = 0.;
       chi2Matrix[0](6,1) = 0.;
       chi2Matrix[0](6,2) = 0.;
       chi2Matrix[0](6,3) = 0.;
       chi2Matrix[0](6,4) = -0.2232882;
       chi2Matrix[0](6,5) = -0.2972839;
       chi2Matrix[0](6,6) = 0.7413607;
       chi2Matrix[0](6,7) = -0.1883866;
       chi2Matrix[0](6,8) = -0.1235052;
       chi2Matrix[0](6,9) = 0.; 
       chi2Matrix[0](7,0) = 0.;
       chi2Matrix[0](7,1) = 0.;
       chi2Matrix[0](7,2) = 0.;
       chi2Matrix[0](7,3) = 0.;
       chi2Matrix[0](7,4) = -0.1716192;
       chi2Matrix[0](7,5) = -0.2067162;
       chi2Matrix[0](7,6) = -0.1883866;
       chi2Matrix[0](7,7) = 0.844935;
       chi2Matrix[0](7,8) = -0.124291;
       chi2Matrix[0](7,9) = 0.; 
       chi2Matrix[0](8,0) = 0.;
       chi2Matrix[0](8,1) = 0.;
       chi2Matrix[0](8,2) = 0.;
       chi2Matrix[0](8,3) = 0.;
       chi2Matrix[0](8,4) = -0.1239006;
       chi2Matrix[0](8,5) = -0.1230729;
       chi2Matrix[0](8,6) = -0.1235052;
       chi2Matrix[0](8,7) = -0.124291;
       chi2Matrix[0](8,8) = 0.8749833;
       chi2Matrix[0](8,9) = 0.;
       chi2Matrix[0](9,0) = 0.;
       chi2Matrix[0](9,1) = 0.;
       chi2Matrix[0](9,2) = 0.;
       chi2Matrix[0](9,3) = 0.;
       chi2Matrix[0](9,4) = 0.;
       chi2Matrix[0](9,5) = 0.;
       chi2Matrix[0](9,6) = 0.;
       chi2Matrix[0](9,7) = 0.;
       chi2Matrix[0](9,8) = 0.;
       chi2Matrix[0](9,9) = 0.; 
     }
   else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading chi2Matrix from file " << edm::FileInPath(chi2MatrixFile_).fullPath().c_str() ;
      std::ifstream chi2MatrixFile(edm::FileInPath(chi2MatrixFile_).fullPath().c_str());
      int tdcBin=0;
      while (!chi2MatrixFile.eof() && tdcBin < nTDCbins_) 
	{
	  //	  chi2Matrix[tdcBin].resize(10);
	  for(int j = 0; j < 10; ++j) {
	    for(int l = 0; l < 10; ++l) {
	      float ww;
	      chi2MatrixFile >> ww;
	      chi2Matrix[tdcBin](j,l)=ww;
	    }
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }
   
//    for (int i=0;i<nTDCbins_;i++)
//      {
       //       assert(chi2Matrix[i].size() == 10);
   chi2Matrix_ =  chi2Matrix;
//      }

   std::vector< EcalWeightSet::EcalChi2WeightMatrix > chi2MatrixAft(nTDCbins_);
   if (!getWeightsFromFile_  && nTDCbins_ == 1 )
     {
       //       chi2MatrixAft[0].resize(10);
       //       for (int i=0;i<10;i++)
	 //	 chi2MatrixAft[0][i].resize(10);

       chi2MatrixAft[0](0,0) = 0.;
       chi2MatrixAft[0](0,1) = 0.;  
       chi2MatrixAft[0](0,2) = 0.;
       chi2MatrixAft[0](0,3) = 0.;
       chi2MatrixAft[0](0,4) = 0.;
       chi2MatrixAft[0](0,5) = 0.;
       chi2MatrixAft[0](0,6) = 0.;
       chi2MatrixAft[0](0,7) = 0.;
       chi2MatrixAft[0](0,8) = 0.;
       chi2MatrixAft[0](0,9) = 0.;
       chi2MatrixAft[0](1,0) = 0.;
       chi2MatrixAft[0](1,1) = 0.;
       chi2MatrixAft[0](1,2) = 0.;
       chi2MatrixAft[0](1,3) = 0.;
       chi2MatrixAft[0](1,4) = 0.;
       chi2MatrixAft[0](1,5) = 0.;
       chi2MatrixAft[0](1,6) = 0.;
       chi2MatrixAft[0](1,7) = 0.;
       chi2MatrixAft[0](1,8) = 0.;
       chi2MatrixAft[0](1,9) = 0.;
       chi2MatrixAft[0](2,0) = 0.;
       chi2MatrixAft[0](2,1) = 0.;
       chi2MatrixAft[0](2,2) = 0.;
       chi2MatrixAft[0](2,3) = 0.;
       chi2MatrixAft[0](2,4) = 0.;
       chi2MatrixAft[0](2,5) = 0.;
       chi2MatrixAft[0](2,6) = 0.;
       chi2MatrixAft[0](2,7) = 0.;
       chi2MatrixAft[0](2,8) = 0.;
       chi2MatrixAft[0](2,9) = 0.;
       chi2MatrixAft[0](3,0) = 0.;
       chi2MatrixAft[0](3,1) = 0.;
       chi2MatrixAft[0](3,2) = 0.;
       chi2MatrixAft[0](3,3) = 0.;
       chi2MatrixAft[0](3,4) = 0.;
       chi2MatrixAft[0](3,5) = 0.;
       chi2MatrixAft[0](3,6) = 0.;
       chi2MatrixAft[0](3,7) = 0.;
       chi2MatrixAft[0](3,8) = 0.;
       chi2MatrixAft[0](3,9) = 0.; 
       chi2MatrixAft[0](4,0) = 0.;
       chi2MatrixAft[0](4,1) = 0.;
       chi2MatrixAft[0](4,2) = 0.;
       chi2MatrixAft[0](4,3) = 0.;
       chi2MatrixAft[0](4,4) = 0.8030884;
       chi2MatrixAft[0](4,5) = -0.2543541;
       chi2MatrixAft[0](4,6) = -0.2243544;
       chi2MatrixAft[0](4,7) = -0.1698177;
       chi2MatrixAft[0](4,8) = -0.1194506;
       chi2MatrixAft[0](4,9) = 0.; 
       chi2MatrixAft[0](5,0) = 0.;
       chi2MatrixAft[0](5,1) = 0.;
       chi2MatrixAft[0](5,2) = 0.;
       chi2MatrixAft[0](5,3) = 0.;
       chi2MatrixAft[0](5,4) = -0.2543541;
       chi2MatrixAft[0](5,5) = 0.6714465;
       chi2MatrixAft[0](5,6) = -0.2898025;
       chi2MatrixAft[0](5,7) = -0.2193564;
       chi2MatrixAft[0](5,8) = -0.1542964;
       chi2MatrixAft[0](5,9) = 0.;
       chi2MatrixAft[0](6,0) = 0.;
       chi2MatrixAft[0](6,1) = 0.;
       chi2MatrixAft[0](6,2) = 0.;
       chi2MatrixAft[0](6,3) = 0.;
       chi2MatrixAft[0](6,4) = -0.2243544;
       chi2MatrixAft[0](6,5) = -0.2898025;
       chi2MatrixAft[0](6,6) = 0.7443781;
       chi2MatrixAft[0](6,7) = -0.1934846;
       chi2MatrixAft[0](6,8) = -0.136098;
       chi2MatrixAft[0](6,9) = 0.; 
       chi2MatrixAft[0](7,0) = 0.;
       chi2MatrixAft[0](7,1) = 0.;
       chi2MatrixAft[0](7,2) = 0.;
       chi2MatrixAft[0](7,3) = 0.;
       chi2MatrixAft[0](7,4) = -0.1698177;
       chi2MatrixAft[0](7,5) = -0.2193564;
       chi2MatrixAft[0](7,6) = -0.1934846;
       chi2MatrixAft[0](7,7) = 0.8535482;
       chi2MatrixAft[0](7,8) = -0.1030149;
       chi2MatrixAft[0](7,9) = 0.; 
       chi2MatrixAft[0](8,0) = 0.;
       chi2MatrixAft[0](8,1) = 0.;
       chi2MatrixAft[0](8,2) = 0.;
       chi2MatrixAft[0](8,3) = 0.;
       chi2MatrixAft[0](8,4) = -0.1194506;
       chi2MatrixAft[0](8,5) = -0.1542964;
       chi2MatrixAft[0](8,6) = -0.136098;
       chi2MatrixAft[0](8,7) = -0.1030149;
       chi2MatrixAft[0](8,8) = 0.9275388;
       chi2MatrixAft[0](8,9) = 0.;
       chi2MatrixAft[0](9,0) = 0.;
       chi2MatrixAft[0](9,1) = 0.;
       chi2MatrixAft[0](9,2) = 0.;
       chi2MatrixAft[0](9,3) = 0.;
       chi2MatrixAft[0](9,4) = 0.;
       chi2MatrixAft[0](9,5) = 0.;
       chi2MatrixAft[0](9,6) = 0.;
       chi2MatrixAft[0](9,7) = 0.;
       chi2MatrixAft[0](9,8) = 0.;
       chi2MatrixAft[0](9,9) = 0.; 
     }
   else if (getWeightsFromFile_)
    {
      //Read from file
      edm::LogInfo("EcalTrivialConditionRetriever") << "Reading chi2MatrixAft from file " << edm::FileInPath(chi2MatrixAftFile_).fullPath().c_str() ;
      std::ifstream chi2MatrixAftFile(edm::FileInPath(chi2MatrixAftFile_).fullPath().c_str());
      int tdcBin=0;
      while (!chi2MatrixAftFile.eof() && tdcBin < nTDCbins_) 
	{
	  //	  chi2MatrixAft[tdcBin].resize(10);
	  for(int j = 0; j < 10; ++j) {
	    for(int l = 0; l < 10; ++l) {
	      float ww;
	      chi2MatrixAftFile >> ww;
	      chi2MatrixAft[tdcBin](j,l)=ww;
	    }
	  }
	  ++tdcBin;
	}
      assert (tdcBin == nTDCbins_);
    }
  else
    {
      //Not supported
      edm::LogError("EcalTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }
   
//    for (int i=0;i<nTDCbins_;i++)
//      {
       //       assert(chi2MatrixAft[i].size() == 10);
   chi2MatrixAft_ =  chi2MatrixAft;
   //      }

}


// --------------------------------------------------------------------------------

std::auto_ptr<EcalChannelStatus>
EcalTrivialConditionRetriever::getChannelStatusFromConfiguration (const EcalChannelStatusRcd&)
{
  std::auto_ptr<EcalChannelStatus> ecalStatus = std::auto_ptr<EcalChannelStatus>( new EcalChannelStatus() );
  

  // start by setting all statuses to 0
  
  // barrel
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi)) {
	EBDetId ebid(ieta,iphi);
	ecalStatus->setValue( ebid, 0 );
      }
    }
  }
  // endcap
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1)) {
	EEDetId eedetidpos(iX,iY,1);
	ecalStatus->setValue( eedetidpos, 0 );
      }
      if (EEDetId::validDetId(iX,iY,-1)) {
	EEDetId eedetidneg(iX,iY,-1);
	ecalStatus->setValue( eedetidneg, 0 );
      }
    }
  }
  
  
  
  // overwrite the statuses which are in the file
  
  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading channel statuses from file " << edm::FileInPath(channelStatusFile_).fullPath().c_str() ;
  std::ifstream statusFile(edm::FileInPath(channelStatusFile_).fullPath().c_str());
  if ( !statusFile.good() ) {
    edm::LogError ("EcalTrivialConditionRetriever") 
      << "*** Problems opening file: " << channelStatusFile_ ;
    throw cms::Exception ("Cannot open ECAL channel status file") ;
  }
  
  std::string EcalSubDet;
  std::string str;
  int hashIndex(0);
  int status(0);
  
  while (!statusFile.eof()) 
    {
      statusFile >> EcalSubDet;
      if (EcalSubDet!=std::string("EB") && EcalSubDet!=std::string("EE"))
	{
	  std::getline(statusFile,str);
	  continue;
	}
      else
	{
	  statusFile>> hashIndex >> status;
	}
      // std::cout << EcalSubDet << " " << hashIndex << " " << status;
      
      if(EcalSubDet==std::string("EB"))
	{
	  EBDetId ebid = EBDetId::unhashIndex(hashIndex);
	  ecalStatus->setValue( ebid, status );
	}
      else if(EcalSubDet==std::string("EE"))
	{
	  EEDetId eedetid = EEDetId::unhashIndex(hashIndex);
	  ecalStatus->setValue( eedetid, status );
	}
      else
	{
	  edm::LogError ("EcalTrivialConditionRetriever") 
	    << " *** " << EcalSubDet << " is neither EB nor EE ";
	}
      
    }
  // the file is supposed to be in the form  -- FIXME
  
  
  statusFile.close();
  return ecalStatus;
}



std::auto_ptr<EcalChannelStatus>
EcalTrivialConditionRetriever::produceEcalChannelStatus( const EcalChannelStatusRcd& )
{

        std::auto_ptr<EcalChannelStatus>  ical = std::auto_ptr<EcalChannelStatus>( new EcalChannelStatus() );
        // barrel
        for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
                if(ieta==0) continue;
                for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
                        if (EBDetId::validDetId(ieta,iphi)) {
                                EBDetId ebid(ieta,iphi);
                                ical->setValue( ebid, 0 );
                        }
                }
        }
        // endcap
        for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
                for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
                        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
                        if (EEDetId::validDetId(iX,iY,1)) {
                                EEDetId eedetidpos(iX,iY,1);
                                ical->setValue( eedetidpos, 0 );
                        }
                        if (EEDetId::validDetId(iX,iY,-1)) {
                                EEDetId eedetidneg(iX,iY,-1);
                                ical->setValue( eedetidneg, 0 );
                        }
                }
        }
        return ical;
}

// --------------------------------------------------------------------------------
std::auto_ptr<EcalDQMChannelStatus>
EcalTrivialConditionRetriever::produceEcalDQMChannelStatus( const EcalDQMChannelStatusRcd& )
{
  uint32_t sta(0);

        std::auto_ptr<EcalDQMChannelStatus>  ical = std::auto_ptr<EcalDQMChannelStatus>( new EcalDQMChannelStatus() );
        // barrel
        for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
                if(ieta==0) continue;
                for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
                        if (EBDetId::validDetId(ieta,iphi)) {
                                EBDetId ebid(ieta,iphi);
                                ical->setValue( ebid, sta );
                        }
                }
        }
        // endcap
        for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
                for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
                        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
                        if (EEDetId::validDetId(iX,iY,1)) {
                                EEDetId eedetidpos(iX,iY,1);
                                ical->setValue( eedetidpos, sta );
                        }
                        if (EEDetId::validDetId(iX,iY,-1)) {
                                EEDetId eedetidneg(iX,iY,-1);
                                ical->setValue( eedetidneg, sta );
                        }
                }
        }
        return ical;
}
// --------------------------------------------------------------------------------

std::auto_ptr<EcalDQMTowerStatus>
EcalTrivialConditionRetriever::produceEcalDQMTowerStatus( const EcalDQMTowerStatusRcd& )
{

        std::auto_ptr<EcalDQMTowerStatus>  ical = std::auto_ptr<EcalDQMTowerStatus>( new EcalDQMTowerStatus() );

	uint32_t sta(0);
	
        // barrel
	int iz=0;
        for(int k=0 ; k<2; k++ ) {
	  if(k==0) iz=-1;
	  if(k==1) iz=+1;
	  for(int i=1 ; i<73; i++) {
	    for(int j=1 ; j<18; j++) {
	      if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,i )){
		EcalTrigTowerDetId ebid(iz,EcalBarrel,j,i);

		ical->setValue( ebid, sta );
	      }
	    }
	  }
	}


        // endcap
        for(int k=0 ; k<2; k++ ) {
	  if(k==0) iz=-1;
	  if(k==1) iz=+1;
	  for(int i=1 ; i<21; i++) {
	    for(int j=1 ; j<21; j++) {
	      if (EcalScDetId::validDetId(i,j,iz )){
		EcalScDetId eeid(i,j,iz);
		ical->setValue( eeid, sta );
	      }
	    }
	  }
	}

        return ical;
}

// --------------------------------------------------------------------------------
std::auto_ptr<EcalDCSTowerStatus>
EcalTrivialConditionRetriever::produceEcalDCSTowerStatus( const EcalDCSTowerStatusRcd& )
{

        std::auto_ptr<EcalDCSTowerStatus>  ical = std::auto_ptr<EcalDCSTowerStatus>( new EcalDCSTowerStatus() );

	int status(0);
	
        // barrel
	int iz=0;
        for(int k=0 ; k<2; k++ ) {
	  if(k==0) iz=-1;
	  if(k==1) iz=+1;
	  for(int i=1 ; i<73; i++) {
	    for(int j=1 ; j<18; j++) {
	      if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,i )){
		EcalTrigTowerDetId ebid(iz,EcalBarrel,j,i);

		ical->setValue( ebid, status );
	      }
	    }
	  }
	}


        // endcap
        for(int k=0 ; k<2; k++ ) {
	  if(k==0) iz=-1;
	  if(k==1) iz=+1;
	  for(int i=1 ; i<21; i++) {
	    for(int j=1 ; j<21; j++) {
	      if (EcalScDetId::validDetId(i,j,iz )){
		EcalScDetId eeid(i,j,iz);
		ical->setValue( eeid, status );
	      }
	    }
	  }
	}

        return ical;
}
// --------------------------------------------------------------------------------

std::auto_ptr<EcalDAQTowerStatus>
EcalTrivialConditionRetriever::produceEcalDAQTowerStatus( const EcalDAQTowerStatusRcd& )
{

        std::auto_ptr<EcalDAQTowerStatus>  ical = std::auto_ptr<EcalDAQTowerStatus>( new EcalDAQTowerStatus() );

	int status(0);
	
        // barrel
	int iz=0;
        for(int k=0 ; k<2; k++ ) {
	  if(k==0) iz=-1;
	  if(k==1) iz=+1;
	  for(int i=1 ; i<73; i++) {
	    for(int j=1 ; j<18; j++) {
	      if (EcalTrigTowerDetId::validDetId(iz,EcalBarrel,j,i )){
		EcalTrigTowerDetId ebid(iz,EcalBarrel,j,i);

		ical->setValue( ebid, status );
	      }
	    }
	  }
	}


        // endcap
        for(int k=0 ; k<2; k++ ) {
	  if(k==0) iz=-1;
	  if(k==1) iz=+1;
	  for(int i=1 ; i<21; i++) {
	    for(int j=1 ; j<21; j++) {
	      if (EcalScDetId::validDetId(i,j,iz )){
		EcalScDetId eeid(i,j,iz);
		ical->setValue( eeid, status );
	      }
	    }
	  }
	}

        return ical;
}
// --------------------------------------------------------------------------------

std::auto_ptr<EcalTPGCrystalStatus>
EcalTrivialConditionRetriever::getTrgChannelStatusFromConfiguration (const EcalTPGCrystalStatusRcd&)
{
  std::auto_ptr<EcalTPGCrystalStatus> ecalStatus = std::auto_ptr<EcalTPGCrystalStatus>( new EcalTPGCrystalStatus() );
  

  // start by setting all statuses to 0
  
  // barrel
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta,iphi)) {
	EBDetId ebid(ieta,iphi);
	ecalStatus->setValue( ebid, 0 );
      }
    }
  }
  // endcap
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX,iY,1)) {
	EEDetId eedetidpos(iX,iY,1);
	ecalStatus->setValue( eedetidpos, 0 );
      }
      if (EEDetId::validDetId(iX,iY,-1)) {
	EEDetId eedetidneg(iX,iY,-1);
	ecalStatus->setValue( eedetidneg, 0 );
      }
    }
  }
  
  
  
  // overwrite the statuses which are in the file
  
  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading channel statuses from file " << edm::FileInPath(channelStatusFile_).fullPath().c_str() ;
  std::ifstream statusFile(edm::FileInPath(channelStatusFile_).fullPath().c_str());
  if ( !statusFile.good() ) {
    edm::LogError ("EcalTrivialConditionRetriever") 
      << "*** Problems opening file: " << channelStatusFile_ ;
    throw cms::Exception ("Cannot open ECAL channel status file") ;
  }
  
  std::string EcalSubDet;
  std::string str;
  int hashIndex(0);
  int status(0);
  
  while (!statusFile.eof()) 
    {
      statusFile >> EcalSubDet;
      if (EcalSubDet!=std::string("EB") && EcalSubDet!=std::string("EE"))
	{
	  std::getline(statusFile,str);
	  continue;
	}
      else
	{
	  statusFile>> hashIndex >> status;
	}
      // std::cout << EcalSubDet << " " << hashIndex << " " << status;
      
      if(EcalSubDet==std::string("EB"))
	{
	  EBDetId ebid = EBDetId::unhashIndex(hashIndex);
	  ecalStatus->setValue( ebid, status );
	}
      else if(EcalSubDet==std::string("EE"))
	{
	  EEDetId eedetid = EEDetId::unhashIndex(hashIndex);
	  ecalStatus->setValue( eedetid, status );
	}
      else
	{
	  edm::LogError ("EcalTrivialConditionRetriever") 
	    << " *** " << EcalSubDet << " is neither EB nor EE ";
	}
      
    }
  // the file is supposed to be in the form  -- FIXME
  
  
  statusFile.close();
  return ecalStatus;
}



std::auto_ptr<EcalTPGCrystalStatus>
EcalTrivialConditionRetriever::produceEcalTrgChannelStatus( const EcalTPGCrystalStatusRcd& )
{

        std::auto_ptr<EcalTPGCrystalStatus>  ical = std::auto_ptr<EcalTPGCrystalStatus>( new EcalTPGCrystalStatus() );
        // barrel
        for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
                if(ieta==0) continue;
                for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
                        if (EBDetId::validDetId(ieta,iphi)) {
                                EBDetId ebid(ieta,iphi);
                                ical->setValue( ebid, 0 );
                        }
                }
        }
        // endcap
        for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
                for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
                        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
                        if (EEDetId::validDetId(iX,iY,1)) {
                                EEDetId eedetidpos(iX,iY,1);
                                ical->setValue( eedetidpos, 0 );
                        }
                        if (EEDetId::validDetId(iX,iY,-1)) {
                                EEDetId eedetidneg(iX,iY,-1);
                                ical->setValue( eedetidneg, 0 );
                        }
                }
        }
        return ical;
}



// --------------------------------------------------------------------------------


std::auto_ptr<EcalIntercalibConstants> 
EcalTrivialConditionRetriever::getIntercalibConstantsFromConfiguration 
( const EcalIntercalibConstantsRcd& )
{
  std::auto_ptr<EcalIntercalibConstants>  ical = 
      std::auto_ptr<EcalIntercalibConstants>( new EcalIntercalibConstants() );

  // Read the values from a txt file
  // -------------------------------

  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading intercalibration constants from file "
                                                << intercalibConstantsFile_.c_str() ;

  FILE *inpFile ;
  inpFile = fopen (intercalibConstantsFile_.c_str (),"r") ;
  if (!inpFile) 
    {
      edm::LogError ("EcalTrivialConditionRetriever") 
         << "*** Can not open file: " << intercalibConstantsFile_ ;
      throw cms::Exception ("Cannot open inter-calibration coefficients txt file") ;
    }

  char line[256] ;
  std::ostringstream str ;
  fgets (line,255,inpFile) ;
  int sm_number=atoi (line) ;
  str << "sm: " << sm_number ;  

  fgets (line,255,inpFile) ;
  //int nevents=atoi (line) ; // not necessary here just for online conddb

  fgets (line,255,inpFile) ;
  std::string gen_tag = line ;
  str << "gen tag: " << gen_tag ;  // should I use this? 

  fgets (line,255,inpFile) ;
  std::string cali_method = line ;
  str << "cali method: " << cali_method << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_version = line ;
  str << "cali version: " << cali_version << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_type = line ;
  str << "cali type: " << cali_type ; // not important

  edm::LogInfo("EcalTrivialConditionRetriever")
            << "[PIETRO] Intercalibration file - " 
            << str.str () << std::endl ;

  float calib[1700]={1} ;
  float calib_rms[1700]={0} ;
  int calib_nevents[1700]={0} ;
  int calib_status[1700]={0} ;

  int ii = 0 ;

  while (fgets (line,255,inpFile)) 
    {
      ii++;
      int dmy_num = 0 ;
      float dmy_calib = 0. ;
      float dmy_RMS = 0. ;
      int dmy_events = 0 ;
      int dmy_status = 0 ;
      sscanf (line, "%d %f %f %d %d", &dmy_num, &dmy_calib,
                                      &dmy_RMS, &dmy_events,
                                      &dmy_status) ;
      assert (dmy_num >= 1) ;
      assert (dmy_num <= 1700) ;
      calib[dmy_num-1] = dmy_calib ; 
      calib_rms[dmy_num-1] = dmy_RMS  ;
      calib_nevents[dmy_num-1] = dmy_events ;
      calib_status[dmy_num-1] = dmy_status ;

//       edm::LogInfo ("EcalTrivialConditionRetriever")
//                 << "[PIETRO] cry = " << dmy_num 
//                 << " calib = " << calib[dmy_num-1] 
//                 << " RMS = " << calib_rms[dmy_num-1] 
//                 << " events = " << calib_nevents[dmy_num-1] 
//                 << " status = " << calib_status[dmy_num-1] 
//                 << std::endl ;
    }

  fclose (inpFile) ;           // close inp. file
  edm::LogInfo ("EcalTrivialConditionRetriever") << "Read intercalibrations for " << ii << " xtals " ; 
  if (ii!=1700) edm::LogWarning ("StoreEcalCondition") 
                << "Some crystals missing, set to 1" << std::endl ;

  // Transfer the data to the inter-calibration coefficients container
  // -----------------------------------------------------------------

  // DB supermodule always set to 1 for the TestBeam FIXME
  int sm_db=1 ;
  // loop over channels 
  for (int i=0 ; i<1700 ; i++)
    {
      //if (EBDetId::validDetId(iEta,iPhi)) {
      // CANNOT be used -- validDetId only checks ETA PHI method
      // doing a check by hand, here is the only place in CMSSW 
      // outside TB code and EcalRawToDigi where needed
      // => no need for changing the DetId interface
      //
      // checking only sm_db -- guess can change with the above FIXME
      if (sm_db >= EBDetId::MIN_SM && sm_db <= EBDetId::MAX_SM) {
        EBDetId ebid (sm_db,i+1,EBDetId::SMCRYSTALMODE) ;
        if (calib_status[i]) ical->setValue (ebid.rawId (), calib[i]) ;
        else ical->setValue (ebid.rawId (), 1.) ;
      }
      //}
    } // loop over channels 
	
//  edm::LogInfo ("EcalTrivialConditionRetriever") << "INTERCALIBRATION DONE" ; 
  return ical;
}


std::auto_ptr<EcalIntercalibErrors> 
EcalTrivialConditionRetriever::getIntercalibErrorsFromConfiguration 
( const EcalIntercalibErrorsRcd& )
{
  std::auto_ptr<EcalIntercalibErrors>  ical = 
      std::auto_ptr<EcalIntercalibErrors>( new EcalIntercalibErrors() );

  // Read the values from a txt file
  // -------------------------------

  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading intercalibration constants from file "
                                                << intercalibErrorsFile_.c_str() ;

  FILE *inpFile ;
  inpFile = fopen (intercalibErrorsFile_.c_str (),"r") ;
  if (!inpFile) 
    {
      edm::LogError ("EcalTrivialConditionRetriever") 
         << "*** Can not open file: " << intercalibErrorsFile_ ;
      throw cms::Exception ("Cannot open inter-calibration coefficients txt file") ;
    }

  char line[256] ;
  std::ostringstream str ;
  fgets (line,255,inpFile) ;
  int sm_number=atoi (line) ;
  str << "sm: " << sm_number ;  

  fgets (line,255,inpFile) ;
  //int nevents=atoi (line) ; // not necessary here just for online conddb

  fgets (line,255,inpFile) ;
  std::string gen_tag = line ;
  str << "gen tag: " << gen_tag ;  // should I use this? 

  fgets (line,255,inpFile) ;
  std::string cali_method = line ;
  str << "cali method: " << cali_method << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_version = line ;
  str << "cali version: " << cali_version << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_type = line ;
  str << "cali type: " << cali_type ; // not important

  edm::LogInfo("EcalTrivialConditionRetriever")
            << "[PIETRO] Intercalibration file - " 
            << str.str () << std::endl ;

  float calib[1700]={1} ;
  float calib_rms[1700]={0} ;
  int calib_nevents[1700]={0} ;
  int calib_status[1700]={0} ;

  int ii = 0 ;

  while (fgets (line,255,inpFile)) 
    {
      ii++;
      int dmy_num = 0 ;
      float dmy_calib = 0. ;
      float dmy_RMS = 0. ;
      int dmy_events = 0 ;
      int dmy_status = 0 ;
      sscanf (line, "%d %f %f %d %d", &dmy_num, &dmy_calib,
                                      &dmy_RMS, &dmy_events,
                                      &dmy_status) ;
      assert (dmy_num >= 1) ;
      assert (dmy_num <= 1700) ;
      calib[dmy_num-1] = dmy_calib ; 
      calib_rms[dmy_num-1] = dmy_RMS  ;
      calib_nevents[dmy_num-1] = dmy_events ;
      calib_status[dmy_num-1] = dmy_status ;

//       edm::LogInfo ("EcalTrivialConditionRetriever")
//                 << "[PIETRO] cry = " << dmy_num 
//                 << " calib = " << calib[dmy_num-1] 
//                 << " RMS = " << calib_rms[dmy_num-1] 
//                 << " events = " << calib_nevents[dmy_num-1] 
//                 << " status = " << calib_status[dmy_num-1] 
//                 << std::endl ;
    }

  fclose (inpFile) ;           // close inp. file
  edm::LogInfo ("EcalTrivialConditionRetriever") << "Read intercalibrations for " << ii << " xtals " ; 
  if (ii!=1700) edm::LogWarning ("StoreEcalCondition") 
                << "Some crystals missing, set to 1" << std::endl ;

  // Transfer the data to the inter-calibration coefficients container
  // -----------------------------------------------------------------

  // DB supermodule always set to 1 for the TestBeam FIXME
  int sm_db=1 ;
  // loop over channels 
  for (int i=0 ; i<1700 ; i++)
    {
      //if (EBDetId::validDetId(iEta,iPhi)) {
      // CANNOT be used -- validDetId only checks ETA PHI method
      // doing a check by hand, here is the only place in CMSSW 
      // outside TB code and EcalRawToDigi where needed
      // => no need for changing the DetId interface
      //
      // checking only sm_db -- guess can change with the above FIXME
      if (sm_db >= EBDetId::MIN_SM && sm_db <= EBDetId::MAX_SM) {
        EBDetId ebid (sm_db,i+1,EBDetId::SMCRYSTALMODE) ;
        if (calib_status[i]) ical->setValue (ebid.rawId (), calib[i]) ;
        else ical->setValue (ebid.rawId (), 1.) ;
      }
      //}
    } // loop over channels 
	
//  edm::LogInfo ("EcalTrivialConditionRetriever") << "INTERCALIBRATION DONE" ; 
  return ical;
}

// --------------------------------------------------------------------------------


std::auto_ptr<EcalTimeCalibConstants> 
EcalTrivialConditionRetriever::getTimeCalibConstantsFromConfiguration 
( const EcalTimeCalibConstantsRcd& )
{
  std::auto_ptr<EcalTimeCalibConstants>  ical = 
      std::auto_ptr<EcalTimeCalibConstants>( new EcalTimeCalibConstants() );

  // Read the values from a txt file
  // -------------------------------

  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading time calibration constants from file "
                                                << timeCalibConstantsFile_.c_str() ;

  FILE *inpFile ;
  inpFile = fopen (timeCalibConstantsFile_.c_str (),"r") ;
  if (!inpFile) 
    {
      edm::LogError ("EcalTrivialConditionRetriever") 
         << "*** Can not open file: " << timeCalibConstantsFile_ ;
      throw cms::Exception ("Cannot open inter-calibration coefficients txt file") ;
    }

  char line[256] ;
  std::ostringstream str ;
  fgets (line,255,inpFile) ;
  int sm_number=atoi (line) ;
  str << "sm: " << sm_number ;  

  fgets (line,255,inpFile) ;
  //int nevents=atoi (line) ; // not necessary here just for online conddb

  fgets (line,255,inpFile) ;
  std::string gen_tag = line ;
  str << "gen tag: " << gen_tag ;  // should I use this? 

  fgets (line,255,inpFile) ;
  std::string cali_method = line ;
  str << "cali method: " << cali_method << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_version = line ;
  str << "cali version: " << cali_version << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_type = line ;
  str << "cali type: " << cali_type ; // not important

  edm::LogInfo("EcalTrivialConditionRetriever")
            << "TimeCalibration file - " 
            << str.str () << std::endl ;

  float calib[1700]={1} ;
  float calib_rms[1700]={0} ;
  int calib_nevents[1700]={0} ;
  int calib_status[1700]={0} ;

  int ii = 0 ;

  while (fgets (line,255,inpFile)) 
    {
      ii++;
      int dmy_num = 0 ;
      float dmy_calib = 0. ;
      float dmy_RMS = 0. ;
      int dmy_events = 0 ;
      int dmy_status = 0 ;
      sscanf (line, "%d %f %f %d %d", &dmy_num, &dmy_calib,
                                      &dmy_RMS, &dmy_events,
                                      &dmy_status) ;
      assert (dmy_num >= 1) ;
      assert (dmy_num <= 1700) ;
      calib[dmy_num-1] = dmy_calib ; 
      calib_rms[dmy_num-1] = dmy_RMS  ;
      calib_nevents[dmy_num-1] = dmy_events ;
      calib_status[dmy_num-1] = dmy_status ;

//       edm::LogInfo ("EcalTrivialConditionRetriever")
//                 << "[PIETRO] cry = " << dmy_num 
//                 << " calib = " << calib[dmy_num-1] 
//                 << " RMS = " << calib_rms[dmy_num-1] 
//                 << " events = " << calib_nevents[dmy_num-1] 
//                 << " status = " << calib_status[dmy_num-1] 
//                 << std::endl ;
    }

  fclose (inpFile) ;           // close inp. file
  edm::LogInfo ("EcalTrivialConditionRetriever") << "Read timeCalibrations for " << ii << " xtals " ; 
  if (ii!=1700) edm::LogWarning ("StoreEcalCondition") 
                << "Some crystals missing, set to 1" << std::endl ;

  // Transfer the data to the inter-calibration coefficients container
  // -----------------------------------------------------------------

  // DB supermodule always set to 1 for the TestBeam FIXME
  int sm_db=1 ;
  // loop over channels 
  for (int i=0 ; i<1700 ; i++)
    {
      //if (EBDetId::validDetId(iEta,iPhi)) {
      // CANNOT be used -- validDetId only checks ETA PHI method
      // doing a check by hand, here is the only place in CMSSW 
      // outside TB code and EcalRawToDigi where needed
      // => no need for changing the DetId interface
      //
      // checking only sm_db -- guess can change with the above FIXME
      if (sm_db >= EBDetId::MIN_SM && sm_db <= EBDetId::MAX_SM) {
        EBDetId ebid (sm_db,i+1,EBDetId::SMCRYSTALMODE) ;
        if (calib_status[i]) ical->setValue (ebid.rawId (), calib[i]) ;
        else ical->setValue (ebid.rawId (), 1.) ;
      }
      //}
    } // loop over channels 
	
//  edm::LogInfo ("EcalTrivialConditionRetriever") << "INTERCALIBRATION DONE" ; 
  return ical;
}


std::auto_ptr<EcalTimeCalibErrors> 
EcalTrivialConditionRetriever::getTimeCalibErrorsFromConfiguration 
( const EcalTimeCalibErrorsRcd& )
{
  std::auto_ptr<EcalTimeCalibErrors>  ical = 
      std::auto_ptr<EcalTimeCalibErrors>( new EcalTimeCalibErrors() );

  // Read the values from a txt file
  // -------------------------------

  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading timeCalibration constants from file "
                                                << timeCalibErrorsFile_.c_str() ;

  FILE *inpFile ;
  inpFile = fopen (timeCalibErrorsFile_.c_str (),"r") ;
  if (!inpFile) 
    {
      edm::LogError ("EcalTrivialConditionRetriever") 
         << "*** Can not open file: " << timeCalibErrorsFile_ ;
      throw cms::Exception ("Cannot open inter-calibration coefficients txt file") ;
    }

  char line[256] ;
  std::ostringstream str ;
  fgets (line,255,inpFile) ;
  int sm_number=atoi (line) ;
  str << "sm: " << sm_number ;  

  fgets (line,255,inpFile) ;
  //int nevents=atoi (line) ; // not necessary here just for online conddb

  fgets (line,255,inpFile) ;
  std::string gen_tag = line ;
  str << "gen tag: " << gen_tag ;  // should I use this? 

  fgets (line,255,inpFile) ;
  std::string cali_method = line ;
  str << "cali method: " << cali_method << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_version = line ;
  str << "cali version: " << cali_version << std::endl ; // not important 

  fgets (line,255,inpFile) ;
  std::string cali_type = line ;
  str << "cali type: " << cali_type ; // not important

  edm::LogInfo("EcalTrivialConditionRetriever")
            << "TimeCalibration file - " 
            << str.str () << std::endl ;

  float calib[1700]={1} ;
  float calib_rms[1700]={0} ;
  int calib_nevents[1700]={0} ;
  int calib_status[1700]={0} ;

  int ii = 0 ;

  while (fgets (line,255,inpFile)) 
    {
      ii++;
      int dmy_num = 0 ;
      float dmy_calib = 0. ;
      float dmy_RMS = 0. ;
      int dmy_events = 0 ;
      int dmy_status = 0 ;
      sscanf (line, "%d %f %f %d %d", &dmy_num, &dmy_calib,
                                      &dmy_RMS, &dmy_events,
                                      &dmy_status) ;
      assert (dmy_num >= 1) ;
      assert (dmy_num <= 1700) ;
      calib[dmy_num-1] = dmy_calib ; 
      calib_rms[dmy_num-1] = dmy_RMS  ;
      calib_nevents[dmy_num-1] = dmy_events ;
      calib_status[dmy_num-1] = dmy_status ;

//       edm::LogInfo ("EcalTrivialConditionRetriever")
//                 << "[PIETRO] cry = " << dmy_num 
//                 << " calib = " << calib[dmy_num-1] 
//                 << " RMS = " << calib_rms[dmy_num-1] 
//                 << " events = " << calib_nevents[dmy_num-1] 
//                 << " status = " << calib_status[dmy_num-1] 
//                 << std::endl ;
    }

  fclose (inpFile) ;           // close inp. file
  edm::LogInfo ("EcalTrivialConditionRetriever") << "Read time calibrations for " << ii << " xtals " ; 
  if (ii!=1700) edm::LogWarning ("StoreEcalCondition") 
                << "Some crystals missing, set to 1" << std::endl ;

  // Transfer the data to the inter-calibration coefficients container
  // -----------------------------------------------------------------

  // DB supermodule always set to 1 for the TestBeam FIXME
  int sm_db=1 ;
  // loop over channels 
  for (int i=0 ; i<1700 ; i++)
    {
      //if (EBDetId::validDetId(iEta,iPhi)) {
      // CANNOT be used -- validDetId only checks ETA PHI method
      // doing a check by hand, here is the only place in CMSSW 
      // outside TB code and EcalRawToDigi where needed
      // => no need for changing the DetId interface
      //
      // checking only sm_db -- guess can change with the above FIXME
      if (sm_db >= EBDetId::MIN_SM && sm_db <= EBDetId::MAX_SM) {
        EBDetId ebid (sm_db,i+1,EBDetId::SMCRYSTALMODE) ;
        if (calib_status[i]) ical->setValue (ebid.rawId (), calib[i]) ;
        else ical->setValue (ebid.rawId (), 1.) ;
      }
      //}
    } // loop over channels 
	
//  edm::LogInfo ("EcalTrivialConditionRetriever") << "INTERCALIBRATION DONE" ; 
  return ical;
}

// --------------------------------------------------------------------------------

std::auto_ptr<EcalMappingElectronics>
EcalTrivialConditionRetriever::getMappingFromConfiguration (const EcalMappingElectronicsRcd&)
{
  std::auto_ptr<EcalMappingElectronics> mapping = std::auto_ptr<EcalMappingElectronics>( new EcalMappingElectronics() );
  edm::LogInfo("EcalTrivialConditionRetriever") << "Reading mapping from file " << edm::FileInPath(mappingFile_).fullPath().c_str() ;
  
  std::ifstream f(edm::FileInPath(mappingFile_).fullPath().c_str());
  if (!f.good())
    {
      edm::LogError("EcalTrivialConditionRetriever") << "File not found";
      throw cms::Exception("FileNotFound");
    }
  
  // uint32_t detid, elecid, triggerid;
  
  int ix, iy, iz, CL;
  // int dccid, towerid, stripid, xtalid;
  // int tccid, tower, ipseudostrip, xtalinps;
  int dccid, towerid, pseudostrip_in_SC, xtal_in_pseudostrip;
  int tccid, tower, pseudostrip_in_TCC, pseudostrip_in_TT;
  
  while ( ! f.eof()) 
    {
      // f >> detid >> elecid >> triggerid; 
      f >> ix >> iy >> iz >> CL >> dccid >> towerid >> pseudostrip_in_SC >> xtal_in_pseudostrip >> tccid >> tower >> 
	pseudostrip_in_TCC >> pseudostrip_in_TT ;
      
//       if (!EEDetId::validDetId(ix,iy,iz))
// 	  continue;
	  
      EEDetId detid(ix,iy,iz,EEDetId::XYMODE);
      // std::cout << " dcc tower ps_in_SC xtal_in_ps " << dccid << " " << towerid << " " << pseudostrip_in_SC << " " << xtal_in_pseudostrip << std::endl;
      EcalElectronicsId elecid(dccid,towerid, pseudostrip_in_SC, xtal_in_pseudostrip);
      // std::cout << " tcc tt ps_in_TT xtal_in_ps " << tccid << " " << tower << " " << pseudostrip_in_TT << " " << xtal_in_pseudostrip << std::endl;
      EcalTriggerElectronicsId triggerid(tccid, tower, pseudostrip_in_TT, xtal_in_pseudostrip);
      EcalMappingElement aElement;
      aElement.electronicsid = elecid.rawId();
      aElement.triggerid = triggerid.rawId();
      (*mapping).setValue(detid, aElement);
    }
  
  f.close();
  return mapping;
}



std::auto_ptr<EcalMappingElectronics>
EcalTrivialConditionRetriever::produceEcalMappingElectronics( const EcalMappingElectronicsRcd& )
{

        std::auto_ptr<EcalMappingElectronics>  ical = std::auto_ptr<EcalMappingElectronics>( new EcalMappingElectronics() );
        return ical;
}
