/** \class AlcaBeamSpotManager
 *  No description available.
 *
 *  $Date: 2010/06/18 14:25:44 $
 *  $Revision: 1.1 $
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>
#include <math.h>

using namespace edm;
using namespace reco;
using namespace std;

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotManager::AlcaBeamSpotManager(void){
}

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotManager::AlcaBeamSpotManager(const ParameterSet& iConfig) :
  beamSpotOutputBase_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("BeamSpotOutputBase"))
{
  LogInfo("AlcaBeamSpotManager") 
    << "Output base: " << beamSpotOutputBase_ 
    << std::endl;
}

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotManager::~AlcaBeamSpotManager(void){
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotManager::readLumi(const LuminosityBlock& iLumi){

  Handle<BeamSpot> beamSpotHandle;
  iLumi.getByLabel("alcaBeamSpotProducer","alcaBeamSpot", beamSpotHandle);


  if(beamSpotHandle.isValid()) { // check the product
    beamSpotMap_[iLumi.luminosityBlock()] = *beamSpotHandle;
    const BeamSpot* aBeamSpot =  &beamSpotMap_[iLumi.luminosityBlock()];
    aBeamSpot = beamSpotHandle.product();
    LogInfo("AlcaBeamSpotManager")
      << "Lumi: " << iLumi.luminosityBlock() << std::endl;
    LogInfo("AlcaBeamSpotManager")
      << *aBeamSpot << std::endl;
  }
  else {
    LogInfo("AlcaBeamSpotManager")
        << "Lumi: " << iLumi.luminosityBlock() << std::endl;
    LogInfo("AlcaBeamSpotManager")
        << "   BS is not valid!" << std::endl;
  }

}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotManager::createWeightedPayloads(void){
  for(bsMap_iterator it=beamSpotMap_.begin(); it!=beamSpotMap_.end();it++){
    if(it->second.type() != 2){
      beamSpotMap_.erase(it);
      --it;
    }
  }
  if(beamSpotMap_.size() <= 1){
    return;
  }
  else if(beamSpotMap_.size() == 2){
    return;
  }
  if(beamSpotOutputBase_ == "lumibased"){
    bsMap_iterator referenceBS = beamSpotMap_.begin();
    bsMap_iterator firstBS     = beamSpotMap_.begin();
    bsMap_iterator lastBS      = beamSpotMap_.begin();
    bsMap_iterator currentBS   = beamSpotMap_.begin();
    bsMap_iterator nextBS      = ++beamSpotMap_.begin();
    bsMap_iterator nextNextBS  = ++(++(beamSpotMap_.begin()));

    int iteration = 0;
    while(nextNextBS!=beamSpotMap_.end()){
      LogInfo("AlcaBeamSpotManager")
    	<< "Iteration: " << iteration++
    	<< endl << currentBS->first  << " : " << currentBS->second
//  	  << endl << nextBS->first	     << " : " << nextBS->second  
//  	  << endl << nextNextBS->first << " : " << nextNextBS->second 
    	<< endl;  
      
      currentBS = nextBS;
      nextBS	= nextNextBS;
      nextNextBS++;
    }
  }
  else if(beamSpotOutputBase_ == "runbased"){
    BeamSpot aBeamSpot = weight(beamSpotMap_.begin(),beamSpotMap_.end());
    LuminosityBlockNumber_t firstLumi = beamSpotMap_.begin()->first;
    beamSpotMap_.clear();
    beamSpotMap_[firstLumi] = aBeamSpot;
  }
  else{
    LogInfo("AlcaBeamSpotManager")
      << "Unrecognized BeamSpotOutputBase parameter: " << beamSpotOutputBase_
      << endl;    
  }
}

//--------------------------------------------------------------------------------------------------
BeamSpot AlcaBeamSpotManager::weight(const bsMap_iterator& begin,
                                     const bsMap_iterator& end){
  double x,xError = 0;
  double y,yError = 0;
  double z,zError = 0;
  double sigmaZ,sigmaZError = 0;
  double dxdz,dxdzError = 0;
  double dydz,dydzError = 0;
  double widthX,widthXError = 0;
  double widthY,widthYError = 0;

  BeamSpot::BeamType type = BeamSpot::Unknown;
  for(bsMap_iterator it=begin; it!=end; it++){
    weight(x     , xError     , it->second.x0()        , it->second.x0Error());
    weight(y     , yError     , it->second.y0()        , it->second.y0Error());
    weight(z     , zError     , it->second.z0()        , it->second.z0Error());
    weight(sigmaZ, sigmaZError, it->second.sigmaZ()    , it->second.sigmaZ0Error());
    weight(dxdz  , dxdzError  , it->second.dxdz()      , it->second.dxdzError());
    weight(dydz  , dydzError  , it->second.dydz()      , it->second.dydzError());
    weight(widthX, widthXError, it->second.BeamWidthX(), it->second.BeamWidthXError());
    weight(widthY, widthYError, it->second.BeamWidthY(), it->second.BeamWidthYError());
    if(it->second.type() == BeamSpot::Tracker){
      type = BeamSpot::Tracker;
    }
  }
  BeamSpot::Point bsPosition(x,y,z);
  BeamSpot::CovarianceMatrix error;
  error(0,0) = xError*xError;
  error(1,1) = yError*yError;
  error(2,2) = zError*zError;
  error(3,3) = sigmaZError*sigmaZError;
  error(4,4) = dxdzError*dxdzError;
  error(5,5) = dydzError*dydzError;
  error(6,6) = widthXError*widthXError;
  BeamSpot weightedBeamSpot(bsPosition,sigmaZ,dxdz,dydz,widthX,error,type);
  weightedBeamSpot.setBeamWidthY(widthY);
  LogInfo("AlcaBeamSpotManager")
    << weightedBeamSpot
    << endl;
  return weightedBeamSpot;
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotManager::weight(double& mean,double& meanError,const double& val,const double& valError){
    double tmpError = 0;
    if (meanError < 1e-8){
	tmpError = 1/(valError*valError);
	mean = val*tmpError;
    }
    else{
	tmpError = 1/(meanError*meanError) + 1/(valError*valError);
	mean = mean/(meanError*meanError) + val/(valError*valError);
    }
    mean = mean/tmpError;
    meanError = sqrt(1/tmpError);
}
