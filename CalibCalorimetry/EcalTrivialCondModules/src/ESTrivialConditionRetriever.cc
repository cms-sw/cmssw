#include <iostream>
#include <fstream>
#include <vector>


#include "CalibCalorimetry/EcalTrivialCondModules/interface/ESTrivialConditionRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/EcalDetId/interface/ESDetId.h"

//#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace edm;

ESTrivialConditionRetriever::ESTrivialConditionRetriever( const edm::ParameterSet&  ps)
{

  // initilize parameters used to produce cond DB objects
  adcToGeVLowConstant_ = ps.getUntrackedParameter<double>("adcToGeVLowConstant",1.0);
  adcToGeVHighConstant_ = ps.getUntrackedParameter<double>("adcToGeVHighConstant",1.0);

  intercalibConstantMean_ = ps.getUntrackedParameter<double>("intercalibConstantMean",1.0);
  intercalibConstantSigma_ = ps.getUntrackedParameter<double>("intercalibConstantSigma",0.0);

  //  intercalibErrorMean_ = ps.getUntrackedParameter<double>("IntercalibErrorMean",0.0);

  ESpedMean_ = ps.getUntrackedParameter<double>("ESpedMean", 200.);
  ESpedRMS_  = ps.getUntrackedParameter<double>("ESpedRMS",  1.00);

  getWeightsFromFile_ = ps.getUntrackedParameter<bool>("getWeightsFromFile",false);


  std::string path="CalibCalorimetry/EcalTrivialCondModules/data/";
  std::string weightType;
  std::ostringstream str;

  weightType = str.str();

  amplWeightsFile_ = ps.getUntrackedParameter<std::string>("amplWeightsFile",path+"ampWeightsES"+weightType);

  // default weights for MGPA shape after pedestal subtraction
  getWeightsFromConfiguration(ps);

  producedESPedestals_ = ps.getUntrackedParameter<bool>("producedESPedestals",true);
  producedESWeights_ = ps.getUntrackedParameter<bool>("producedESWeights",true);

  producedESADCToGeVConstant_ = ps.getUntrackedParameter<bool>("producedESADCToGeVConstant",true);

  verbose_ = ps.getUntrackedParameter<int>("verbose", 0);

  //Tell Producer what we produce
  //setWhatproduce(this);
  if (producedESPedestals_)
    setWhatProduced(this, &ESTrivialConditionRetriever::produceESPedestals );

  if (producedESWeights_) {
      setWhatProduced(this, &ESTrivialConditionRetriever::produceESWeightStripGroups );
      setWhatProduced(this, &ESTrivialConditionRetriever::produceESTBWeights );
    }

  if (producedESADCToGeVConstant_)
    setWhatProduced(this, &ESTrivialConditionRetriever::produceESADCToGeVConstant );

  // intercalibration constants
  producedESIntercalibConstants_ = ps.getUntrackedParameter<bool>("producedESIntercalibConstants",true);
  intercalibConstantsFile_ = ps.getUntrackedParameter<std::string>("intercalibConstantsFile","") ;

  if (producedESIntercalibConstants_) { // user asks to produce constants
    if(intercalibConstantsFile_ == "") {  // if file provided read constants
      //    setWhatProduced (this, &ESTrivialConditionRetriever::getIntercalibConstantsFromConfiguration ) ;
      //  } else { // set all constants to 1. or smear as specified by user
        setWhatProduced (this, &ESTrivialConditionRetriever::produceESIntercalibConstants ) ;
    }
    findingRecord<ESIntercalibConstantsRcd> () ;
  }

  // intercalibration constants errors
  /*  producedESIntercalibErrors_ = ps.getUntrackedParameter<bool>("producedESIntercalibErrors",true);
      intercalibErrorsFile_ = ps.getUntrackedParameter<std::string>("intercalibErrorsFile","") ;
      
      if (producedESIntercalibErrors_) { // user asks to produce constants
      if(intercalibErrorsFile_ != "") {  // if file provided read constants
      setWhatProduced (this, &ESTrivialConditionRetriever::getIntercalibErrorsFromConfiguration ) ;
      } else { // set all constants to 1. or smear as specified by user
      setWhatProduced (this, &ESTrivialConditionRetriever::produceESIntercalibErrors ) ;
      }
      findingRecord<ESIntercalibErrorsRcd> () ;
      }
  */

  // channel status
  producedESChannelStatus_ = ps.getUntrackedParameter<bool>("producedESChannelStatus",true);
  channelStatusFile_ = ps.getUntrackedParameter<std::string>("channelStatusFile","");

  if ( producedESChannelStatus_ ) {
          if ( channelStatusFile_ != "" ) { // if file provided read channel map
                  setWhatProduced( this, &ESTrivialConditionRetriever::getChannelStatusFromConfiguration );
          } else { // set all channels to working -- FIXME might be changed
                  setWhatProduced( this, &ESTrivialConditionRetriever::produceESChannelStatus );
          }
          findingRecord<ESChannelStatusRcd>();
  }

  //Tell Finder what records we find
  if (producedESPedestals_)  findingRecord<ESPedestalsRcd>();

  if (producedESWeights_) {
      findingRecord<ESWeightStripGroupsRcd>();
      findingRecord<ESTBWeightsRcd>();
    }

  if (producedESADCToGeVConstant_)  findingRecord<ESADCToGeVConstantRcd>();

}

ESTrivialConditionRetriever::~ESTrivialConditionRetriever()
{
}

//
// member functions
//
void
ESTrivialConditionRetriever::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& rk,
                                               const edm::IOVSyncValue& iTime,
                                               edm::ValidityInterval& oValidity)
{
  if(verbose_>=1) std::cout << "ESTrivialConditionRetriever::setIntervalFor(): record key = " << rk.name() << "\ttime: " << iTime.time().value() << std::endl;
  //For right now, we will just use an infinite interval of validity
  oValidity = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime() );
}

//produce methods
std::auto_ptr<ESPedestals>
ESTrivialConditionRetriever::produceESPedestals( const ESPedestalsRcd& ) {
  std::cout<< " producing pedestals"<< std::endl;
  std::auto_ptr<ESPedestals>  peds = std::auto_ptr<ESPedestals>( new ESPedestals() );
  ESPedestals::Item ESitem;
  ESitem.mean  = ESpedMean_;
  ESitem.rms   = ESpedRMS_;
  
  for (int istrip=ESDetId::ISTRIP_MIN;istrip<=ESDetId::ISTRIP_MAX;istrip++) {
    for (int ix=ESDetId::IX_MIN;ix<=ESDetId::IX_MAX;ix++){
      for (int iy=ESDetId::IY_MIN;iy<=ESDetId::IY_MAX;iy++){
	for ( int iplane=1; iplane<=2; iplane++){
	  for(int izeta=-1; izeta<=1 ;++izeta) {
	    if(izeta==0) continue;
	    try {
	      //ESDetId Plane iplane Zside izeta 
	      ESDetId aPositiveId(istrip,ix,iy,iplane,izeta);
	      peds->insert(std::make_pair(aPositiveId.rawId(),ESitem));
	    }
	    catch ( cms::Exception &e ) 
	      { 
	      }
	  }
	}
      }
    }
  }
  //return std::auto_ptr<ESPedestals>( peds );
  std::cout<< " produced pedestals"<< std::endl;
  return peds;
}

std::auto_ptr<ESWeightStripGroups>
ESTrivialConditionRetriever::produceESWeightStripGroups( const ESWeightStripGroupsRcd& )
{
  std::auto_ptr<ESWeightStripGroups> xtalGroups = std::auto_ptr<ESWeightStripGroups>( new ESWeightStripGroups() );
  ESStripGroupId defaultGroupId(1);
  std::cout << "entering produce weight groups"<< std::endl;
  for (int istrip=ESDetId::ISTRIP_MIN;istrip<=ESDetId::ISTRIP_MAX;istrip++) {
    for (int ix=ESDetId::IX_MIN;ix<=ESDetId::IX_MAX;ix++){
      for (int iy=ESDetId::IY_MIN;iy<=ESDetId::IY_MAX;iy++){
	for ( int iplane=1; iplane<=2; iplane++){
	  for(int izeta=-1; izeta<=1 ;++izeta) {
	    if(izeta==0) continue;
	    try {
	      //ESDetId Plane iplane Zside izeta 
	      ESDetId anESId(istrip,ix,iy,iplane,izeta);
	      //	  xtalGroups->setValue(ebid.rawId(), ESStripGroupId(ieta) ); // define rings in eta
	      xtalGroups->setValue(anESId.rawId(), defaultGroupId ); // define rings in eta
	    }
	    catch ( cms::Exception &e ) 
	      { 
	      }
	  }
	}
      }
    }
  }
  std::cout << "done with produce weight groups"<< std::endl;

  return xtalGroups;
}

std::auto_ptr<ESIntercalibConstants>
ESTrivialConditionRetriever::produceESIntercalibConstants( const ESIntercalibConstantsRcd& )
{
  std::auto_ptr<ESIntercalibConstants>  ical = std::auto_ptr<ESIntercalibConstants>( new ESIntercalibConstants() );
  std::cout << "entring produce intercalib "<< std::endl;

  for (int istrip=ESDetId::ISTRIP_MIN;istrip<=ESDetId::ISTRIP_MAX;istrip++) {
    for (int ix=ESDetId::IX_MIN;ix<=ESDetId::IX_MAX;ix++){
      for (int iy=ESDetId::IY_MIN;iy<=ESDetId::IY_MAX;iy++){
	for ( int iplane=1; iplane<=2; iplane++){
	  for(int izeta=-1; izeta<=1 ;++izeta) {
	    if(izeta==0) continue;
	    try {
	      //ESDetId Plane iplane Zside izeta 
              ESDetId anESId(istrip,ix,iy,iplane,izeta);
	      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	      ical->setValue( anESId.rawId(), intercalibConstantMean_ + r*intercalibConstantSigma_ );
	    }
            catch ( cms::Exception &e )
              {
              }
          }
        }
      }
    }
  }
  std::cout << "done produce intercalib"<< std::endl;

  return ical;
}


std::auto_ptr<ESADCToGeVConstant>
ESTrivialConditionRetriever::produceESADCToGeVConstant( const ESADCToGeVConstantRcd& )
{
  return std::auto_ptr<ESADCToGeVConstant>( new ESADCToGeVConstant(adcToGeVLowConstant_,adcToGeVHighConstant_) );
}

std::auto_ptr<ESTBWeights>
ESTrivialConditionRetriever::produceESTBWeights( const ESTBWeightsRcd& )
{
  // create weights for the test-beam
  std::auto_ptr<ESTBWeights> tbwgt = std::auto_ptr<ESTBWeights>( new ESTBWeights() );

  int igrp=1;
  ESWeightSet wgt = ESWeightSet(amplWeights_);
  //  ESWeightSet::ESWeightMatrix& mat1 = wgt.getWeights();
  
  tbwgt->setValue(igrp,wgt);


  return tbwgt;
}


  
void ESTrivialConditionRetriever::getWeightsFromConfiguration(const edm::ParameterSet& ps)
{

  ESWeightSet::ESWeightMatrix vampl;

  if (!getWeightsFromFile_ )
    {
      
      //      vampl.set(1.);

      // amplwgtv[0]= ps.getUntrackedParameter< std::vector<double> >("amplWeights", vampl);
    }
  else if (getWeightsFromFile_)
    {
      edm::LogInfo("ESTrivialConditionRetriever") << "Reading amplitude weights from file " << edm::FileInPath(amplWeightsFile_).fullPath().c_str() ;
      std::ifstream amplFile(edm::FileInPath(amplWeightsFile_).fullPath().c_str());
      while (!amplFile.eof() ) 
	{
	  for(int j = 0; j < 2; ++j) {
	    std::vector<float> vec(3) ;
	    for(int k = 0; k < 3; ++k) {
	      float ww;
	      amplFile >> ww;
	            vec[k]=ww;
	    }
	    // vampl.putRow(vec);
	  }
	}
    }
  else
    {
      //Not supported
      edm::LogError("ESTrivialConditionRetriever") << "Configuration not supported. Exception is raised ";
      throw cms::Exception("WrongConfig");
    }

  
  amplWeights_=ESWeightSet(vampl);

}

// --------------------------------------------------------------------------------

std::auto_ptr<ESChannelStatus>
ESTrivialConditionRetriever::getChannelStatusFromConfiguration (const ESChannelStatusRcd&)
{
  std::auto_ptr<ESChannelStatus> ecalStatus = std::auto_ptr<ESChannelStatus>( new ESChannelStatus() );
  

  // start by setting all statuses to 0
  

  for (int istrip=ESDetId::ISTRIP_MIN;istrip<=ESDetId::ISTRIP_MAX;istrip++) {
    for (int ix=ESDetId::IX_MIN;ix<=ESDetId::IX_MAX;ix++){
      for (int iy=ESDetId::IY_MIN;iy<=ESDetId::IY_MAX;iy++){
	for ( int iplane=1; iplane<=2; iplane++){
	  for(int izeta=-1; izeta<=1 ;++izeta) {
	    if(izeta==0) continue;
	    try {
	      //ESDetId Plane iplane Zside izeta 
	      ESDetId anESId(istrip,ix,iy,iplane,izeta);
	      //	  xtalGroups->setValue(ebid.rawId(), ESStripGroupId(ieta) ); // define rings in eta
	      ecalStatus->setValue( anESId, 0 );
	    }
	    catch ( cms::Exception &e ) 
	      { 
	      }
	  }
	}
      }
    }
  }
  
  // overwrite the statuses which are in the file
  
  edm::LogInfo("ESTrivialConditionRetriever") << "Reading channel statuses from file " << edm::FileInPath(channelStatusFile_).fullPath().c_str() ;
  std::ifstream statusFile(edm::FileInPath(channelStatusFile_).fullPath().c_str());
  if ( !statusFile.good() ) {
    edm::LogError ("ESTrivialConditionRetriever") 
      << "*** Problems opening file: " << channelStatusFile_ ;
    throw cms::Exception ("Cannot open ECAL channel status file") ;
  }
  
  std::string ESSubDet;
  std::string str;
  int hashIndex(0);
  int status(0);
  
  while (!statusFile.eof()) 
    {
      statusFile >> ESSubDet;
      if (ESSubDet!=std::string("ES") )
	{
	  std::getline(statusFile,str);
	  continue;
	}
      else
	{
	  statusFile>> hashIndex >> status;
	}
      // std::cout << ESSubDet << " " << hashIndex << " " << status;
      
      if(ESSubDet==std::string("ES"))
	{
	  ESDetId esid = ESDetId::unhashIndex(hashIndex);
	  ecalStatus->setValue( esid, status );
	}
      else
	{
	  edm::LogError ("ESTrivialConditionRetriever") 
	    << " *** " << ESSubDet << " is not ES ";
	}
    }
  // the file is supposed to be in the form 
  // ES hashed_index status 
  // ES 132332 1  --> higher than 0  means bad 
  
  statusFile.close();
  return ecalStatus;
}



std::auto_ptr<ESChannelStatus>
ESTrivialConditionRetriever::produceESChannelStatus( const ESChannelStatusRcd& )
{

  std::auto_ptr<ESChannelStatus>  ical = std::auto_ptr<ESChannelStatus>( new ESChannelStatus() );
  for (int istrip=ESDetId::ISTRIP_MIN;istrip<=ESDetId::ISTRIP_MAX;istrip++) {
    for (int ix=ESDetId::IX_MIN;ix<=ESDetId::IX_MAX;ix++){
      for (int iy=ESDetId::IY_MIN;iy<=ESDetId::IY_MAX;iy++){
	for ( int iplane=1; iplane<=2; iplane++){
	  for(int izeta=-1; izeta<=1 ;++izeta) {
	    if(izeta==0) continue;
	    try {
	      //ESDetId Plane iplane Zside izeta 
	      ESDetId anESId(istrip,ix,iy,iplane,izeta);
	      //	  xtalGroups->setValue(ebid.rawId(), ESStripGroupId(ieta) ); // define rings in eta
	      ical->setValue( anESId, 0 );
	    }
	    catch ( cms::Exception &e ) 
	      { 
	      }
	  }
	}
      }
    }
  }
  
  return ical;
}



// --------------------------------------------------------------------------------



