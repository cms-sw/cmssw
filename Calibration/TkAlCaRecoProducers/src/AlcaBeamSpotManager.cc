/** \class AlcaBeamSpotManager
 *  No description available.
 *
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>
#include <math.h>
#include <limits.h>

using namespace edm;
using namespace reco;
using namespace std;

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotManager::AlcaBeamSpotManager(void){
}

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotManager::AlcaBeamSpotManager(const ParameterSet& iConfig, edm::ConsumesCollector&& iC) :
  beamSpotOutputBase_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("BeamSpotOutputBase")),
  beamSpotModuleName_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("BeamSpotModuleName")),
  beamSpotLabel_     (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("BeamSpotLabel")),
  sigmaZCut_    (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<double>("SigmaZCut"))
{
  edm::InputTag beamSpotTag_(beamSpotModuleName_, beamSpotLabel_);
  beamSpotToken_ = iC.consumes<reco::BeamSpot,edm::InLumi>(beamSpotTag_);
  LogInfo("AlcaBeamSpotManager") 
    << "Output base: " << beamSpotOutputBase_ 
    << std::endl;
  reset();
}

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotManager::~AlcaBeamSpotManager(void){
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotManager::reset(void){
  beamSpotMap_.clear();
}
//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotManager::readLumi(const LuminosityBlock& iLumi){

  Handle<BeamSpot> beamSpotHandle;
  iLumi.getByToken(beamSpotToken_, beamSpotHandle);

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
  vector<bsMap_iterator> listToErase;
  for(bsMap_iterator it=beamSpotMap_.begin(); it!=beamSpotMap_.end();it++){
    if(it->second.type() != BeamSpot::Tracker || it->second.sigmaZ()<sigmaZCut_ ) {
      listToErase.push_back(it);
    }
  }
  for(vector<bsMap_iterator>::iterator it=listToErase.begin(); it !=listToErase.end(); it++){
    beamSpotMap_.erase(*it);
  }
  if(beamSpotMap_.size() <= 1){
    return;
  }
  //Return only if lumibased since the collapsing alghorithm requires the next and next to next lumi sections
  else if(beamSpotMap_.size() == 2 && beamSpotOutputBase_ == "lumibased"){
    return;
  }
  if(beamSpotOutputBase_ == "lumibased"){
//    bsMap_iterator referenceBS = beamSpotMap_.begin();
    bsMap_iterator firstBS     = beamSpotMap_.begin();
//    bsMap_iterator lastBS      = beamSpotMap_.begin();
    bsMap_iterator currentBS   = beamSpotMap_.begin();
    bsMap_iterator nextBS      = ++beamSpotMap_.begin();
    bsMap_iterator nextNextBS  = ++(++(beamSpotMap_.begin()));

    map<LuminosityBlockNumber_t,BeamSpot> tmpBeamSpotMap_;
    bool docreate = true;
    bool endOfRun = false;//Added
    bool docheck = true;//Added
    bool foundShift = false;
    long countlumi = 0;//Added
    string tmprun = "";//Added
    long maxNlumis = 60;//Added
//    if weighted:
//        maxNlumis = 999999999


    unsigned int iteration = 0;
    //while(nextNextBS!=beamSpotMap_.end()){
    while(nextBS!=beamSpotMap_.end()){
      LogInfo("AlcaBeamSpotManager")
        << "Iteration: " << iteration << " size: " << beamSpotMap_.size() << "\n"
        << "Lumi: " << currentBS->first  << "\n" 
        << currentBS->second
        << "\n" << nextBS->first     << "\n" << nextBS->second  
        << endl;  
      if (nextNextBS!=beamSpotMap_.end())
	LogInfo("AlcaBeamSpotManager")
	  << nextNextBS->first << "\n" << nextNextBS->second
	  << endl;


      if(docreate){																																    
	firstBS = currentBS;																												    
        docreate = false;//Added																															    
      }
      //if(iteration >= beamSpotMap_.size()-3){
      if(iteration >= beamSpotMap_.size()-2){
        LogInfo("AlcaBeamSpotManager")
	  << "Reached lumi " << currentBS->first
	  << " now close payload because end of data has been reached.";																										    
          docreate = true;
	  endOfRun = true;																															    
      }    
      // check we run over the same run 																													    
//      if (ibeam->first.Run() != inextbeam->first.Run()){																											    
//        LogInfo("AlcaBeamSpotManager")																													    
//          << "close payload because end of run.";																												    
//        docreate = true;																															    
//      }																																	    
      // check maximum lumi counts
      if (countlumi == maxNlumis -1){																														    
        LogInfo("AlcaBeamSpotManager")																														    
          << "close payload because maximum lumi sections accumulated within run ";																								    
        docreate = true;																															    
        countlumi = 0;																																    
      }
//      if weighted:																																    
//          docheck = False																															    
      // check offsets																																    
      if(docheck){																																    
	foundShift = false;
	LogInfo("AlcaBeamSpotManager")																											  		    
          << "Checking checking!" << endl;
	float limit = 0;																														  	    
        pair<float,float> adelta1;																												  		    
        pair<float,float> adelta2;																												  		    
        pair<float,float> adelta;																												  
        pair<float,float> adelta1dxdz;
        pair<float,float> adelta2dxdz;
        pair<float,float> adelta1dydz;
        pair<float,float> adelta2dydz;
        pair<float,float> adelta1widthX;
        pair<float,float> adelta2widthX;
        pair<float,float> adelta1widthY;
        pair<float,float> adelta2widthY;
        pair<float,float> adelta1z0;
        pair<float,float> adelta1sigmaZ;

        // define minimum limit
        float min_limit = 0.0025;
        
        // limit for x and y
        limit = currentBS->second.BeamWidthX()/2.;																										  		    
        if(limit < min_limit){
	  limit = min_limit;
        }
	
	//check movements in X																														  		    
	adelta1 = delta(currentBS->second.x0(), currentBS->second.x0Error(), nextBS->second.x0(), nextBS->second.x0Error());																	  		    
        adelta2 = pair<float,float>(0.,1.e9);																													    
        if (nextNextBS->second.type() != -1){																											  		    
            adelta2 = delta(nextBS->second.x0(), nextBS->second.x0Error(), nextNextBS->second.x0(), nextNextBS->second.x0Error());																  		    
        }																															  		    
        bool deltaX = (deltaSig(adelta1.first,adelta1.second) > 3.5 && adelta1.first >= limit)?true:false;																			  		    
        if(iteration < beamSpotMap_.size()-2){  																										  		    
            if( !deltaX && adelta1.first*adelta2.first > 0. &&  fabs(adelta1.first+adelta2.first) >= limit){																			  		    
              LogInfo("AlcaBeamSpotManager")																											  		    
        	<< " positive, " << (adelta1.first+adelta2.first) << " limit=" << limit << endl;																				  		    
        	deltaX = true;  																												  		    
            }
            else if( deltaX && adelta1.first*adelta2.first < 0 && adelta2.first != 0 && fabs(adelta1.first/adelta2.first) > 0.33 && fabs(adelta1.first/adelta2.first) < 3){ 													  		    
              LogInfo("AlcaBeamSpotManager")																											  		    
        	<< " negative, " << adelta1.first/adelta2.first << endl;																							  		    
        	deltaX = false; 																												  		    
            }
        }																															  

        //calculating all deltas																														  		    
        adelta1dxdz = delta(currentBS->second.dxdz(), currentBS->second.dxdzError(), nextBS->second.dxdz(), nextBS->second.dxdzError());
        adelta2dxdz = pair<float,float>(0.,1.e9);
        adelta1dydz = delta(currentBS->second.dydz(), currentBS->second.dydzError(), nextBS->second.dydz(), nextBS->second.dydzError());
        adelta2dydz = pair<float,float>(0.,1.e9);
        adelta1widthX = delta(currentBS->second.BeamWidthX(), currentBS->second.BeamWidthXError(), nextBS->second.BeamWidthX(), nextBS->second.BeamWidthXError());
        adelta2widthX = pair<float,float>(0.,1.e9);
        adelta1widthY = delta(currentBS->second.BeamWidthY(), currentBS->second.BeamWidthYError(), nextBS->second.BeamWidthY(), nextBS->second.BeamWidthYError());
        adelta2widthY = pair<float,float>(0.,1.e9);
	adelta1z0 = delta(currentBS->second.z0(), currentBS->second.z0Error(), nextBS->second.z0(), nextBS->second.z0Error());
	adelta1sigmaZ = delta(currentBS->second.sigmaZ(), currentBS->second.sigmaZ0Error(), nextBS->second.sigmaZ(), nextBS->second.sigmaZ0Error());
	
	//check movements in Y																														  		    
        adelta1 = delta(currentBS->second.y0(), currentBS->second.y0Error(), nextBS->second.y0(), nextBS->second.y0Error());																			    
        adelta2 = pair<float,float>(0.,1.e9);																													    
        if( nextNextBS->second.type() != BeamSpot::Unknown){																										  	    
          adelta2 = delta(nextBS->second.y0(), nextBS->second.y0Error(), nextNextBS->second.y0(), nextNextBS->second.y0Error());																  		    
          adelta2dxdz = delta(nextBS->second.dxdz(), nextBS->second.dxdzError(), nextNextBS->second.dxdz(), nextNextBS->second.dxdzError());
          adelta2dydz = delta(nextBS->second.dydz(), nextBS->second.dydzError(), nextNextBS->second.dydz(), nextNextBS->second.dydzError());
          adelta2widthX = delta(nextBS->second.BeamWidthX(), nextBS->second.BeamWidthXError(), nextNextBS->second.BeamWidthX(), nextNextBS->second.BeamWidthXError());
          adelta2widthY = delta(nextBS->second.BeamWidthY(), nextBS->second.BeamWidthYError(), nextNextBS->second.BeamWidthY(), nextNextBS->second.BeamWidthYError());

        }																															  		    
        bool deltaY = (deltaSig(adelta1.first,adelta1.second) > 3.5 && adelta1.first >= limit)?true:false;																			  		    
        if(iteration < beamSpotMap_.size()-2){  																										  		    
          if( !deltaY && adelta1.first*adelta2.first > 0. &&  fabs(adelta1.first+adelta2.first) >= limit){																					  
            LogInfo("AlcaBeamSpotManager")																													  
              << " positive, " << (adelta1.first+adelta2.first) << " limit=" << limit << endl;  																						  
              deltaY = true;																															  
          }
          else if( deltaY && adelta1.first*adelta2.first < 0 && adelta2.first != 0 && fabs(adelta1.first/adelta2.first) > 0.33 && fabs(adelta1.first/adelta2.first) < 3){																  
            LogInfo("AlcaBeamSpotManager")																													  
              << " negative, " << adelta1.first/adelta2.first << endl;  																									  
              deltaY = false;																															  
          }
	}

	limit = currentBS->second.sigmaZ()/2.;														    														
	bool deltaZ = (deltaSig(adelta1z0.first,adelta1z0.second) > 3.5 && fabs(adelta1z0.first) >= limit)?true:false;						      
	adelta = delta(currentBS->second.sigmaZ(), currentBS->second.sigmaZ0Error(), nextBS->second.sigmaZ(), nextBS->second.sigmaZ0Error()); 		    														
	bool deltasigmaZ = (deltaSig(adelta.first,adelta.second) > 5.0)?true:false;										    														
	bool deltadxdz = false;
        bool deltadydz = false;
        bool deltawidthX = false;
        bool deltawidthY = false;

	if(iteration < beamSpotMap_.size()-2){                                                                                                                                                              

          adelta = delta(currentBS->second.dxdz(), currentBS->second.dxdzError(), nextBS->second.dxdz(), nextBS->second.dxdzError());				    														      
          deltadxdz   = (deltaSig(adelta.first,adelta.second) > 5.0)?true:false;										    														
          if(deltadxdz && (adelta1dxdz.first*adelta2dxdz.first) < 0 && adelta2dxdz.first != 0 && fabs(adelta1dxdz.first/adelta2dxdz.first) > 0.33 && fabs(adelta1dxdz.first/adelta2dxdz.first) < 3){
            deltadxdz = false;
	  }

          adelta = delta(currentBS->second.dydz(), currentBS->second.dydzError(), nextBS->second.dydz(), nextBS->second.dydzError());				    														      
          deltadydz   = (deltaSig(adelta.first,adelta.second) > 5.0)?true:false;										    														
          if(deltadydz && (adelta1dydz.first*adelta2dydz.first) < 0 && adelta2dydz.first != 0 && fabs(adelta1dydz.first/adelta2dydz.first) > 0.33 && fabs(adelta1dydz.first/adelta2dydz.first) < 3){
            deltadydz = false;
	  }

          adelta = delta(currentBS->second.BeamWidthX(), currentBS->second.BeamWidthXError(), nextBS->second.BeamWidthX(), nextBS->second.BeamWidthXError());	    														
          deltawidthX = (deltaSig(adelta.first,adelta.second) > 5.0)?true:false;										    														
          if(deltawidthX && (adelta1widthX.first*adelta2widthX.first) < 0 && adelta2widthX.first != 0 && fabs(adelta1widthX.first/adelta2widthX.first) > 0.33 && fabs(adelta1widthX.first/adelta2widthX.first) < 3){
            deltawidthX = false;
	  }

          adelta = delta(currentBS->second.BeamWidthY(), currentBS->second.BeamWidthYError(), nextBS->second.BeamWidthY(), nextBS->second.BeamWidthYError());	    														
          deltawidthY = (deltaSig(adelta.first,adelta.second) > 5.0)?true:false;										    														
          if(deltawidthY && (adelta1widthY.first*adelta2widthY.first) < 0 && adelta2widthY.first != 0 && fabs(adelta1widthY.first/adelta2widthY.first) > 0.33 && fabs(adelta1widthY.first/adelta2widthY.first) < 3){
            deltawidthY = false;
	  }

	}
	if (deltaX || deltaY || deltaZ || deltasigmaZ || deltadxdz || deltadydz || deltawidthX || deltawidthY){						    														
	  docreate = true;
	  foundShift = true;
	  LogInfo("AlcaBeamSpotManager")															    														
	    << "close payload because of movement in" 													    														
	    <<  " X=" << deltaX 
	    << ", Y=" << deltaY 
	    << ", Z=" << deltaZ 
	    << ", sigmaZ=" << deltasigmaZ 
	    << ", dxdz=" << deltadxdz 
	    << ", dydz=" << deltadydz
	    << ", widthX=" << deltawidthX
	    << ", widthY=" << deltawidthY
	    << endl;  
	}

      }
      if(docreate){
	if(foundShift){
	  tmpBeamSpotMap_[firstBS->first] = weight(firstBS,nextBS);
	  if (endOfRun){ 
	    //if we're here, then we need to found a shift in the last LS
	    //We already created a new IOV, now create one just for the last LS
	    tmpBeamSpotMap_[nextBS->first] = nextBS->second;
	  }
	}
	else if(!foundShift && !endOfRun){ //maxLS reached
          tmpBeamSpotMap_[firstBS->first] = weight(firstBS,nextBS);
	}
	else { // end of run with no shift detectred in last LS
	  tmpBeamSpotMap_[firstBS->first] = weight(firstBS,beamSpotMap_.end());
	}
	firstBS = nextBS;
          countlumi = 0;
	  																															    
      }
      //tmprun = currentBS->second.Run
      // increase the counter by one only if the IOV hasn't been closed																														    
      if (!docreate) ++countlumi;																																    
      
      currentBS = nextBS;
      nextBS	= nextNextBS;
      nextNextBS++;
      ++iteration;
    }
    beamSpotMap_.clear();
    beamSpotMap_ = tmpBeamSpotMap_;
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
  LogInfo("AlcaBeamSpotManager")
    << "Weighted BeamSpot will span lumi " 
    << begin->first << " to " << end->first
    << endl;

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
    << "Weighted BeamSpot will be:" <<'\n'
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

//--------------------------------------------------------------------------------------------------
pair<float,float> AlcaBeamSpotManager::delta(const float& x, const float& xError, const float& nextX, const float& nextXError){
  return pair<float,float>(x - nextX, sqrt(pow(xError,2) + pow(nextXError,2)) );
}

//--------------------------------------------------------------------------------------------------
float AlcaBeamSpotManager::deltaSig(const float& num, const float& den){
  if(den != 0){
    return fabs(num/den);
  }
  else{
    return float(LONG_MAX);
  }
}

