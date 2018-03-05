/** \class SiPixelStatusManager
 *  helper class that set up IOV strcutre of SiPixelDetectorStatus
 *
 *  \author 
 */

#include "CalibTracker/SiPixelQuality/interface/SiPixelStatusManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>
#include <cmath>
#include <climits>

#include <iostream>

using namespace edm;
using namespace std;

//--------------------------------------------------------------------------------------------------
SiPixelStatusManager::SiPixelStatusManager(){
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusManager::SiPixelStatusManager(const ParameterSet& iConfig, edm::ConsumesCollector&& iC) :
  outputBase_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("outputBase")),
  aveDigiOcc_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("aveDigiOcc")),
  nLumi_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<int>("resetEveryNLumi")),
  moduleName_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("moduleName")),
  label_     (iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters").getUntrackedParameter<std::string>("label"))
{

  edm::InputTag siPixelStatusTag_(moduleName_, label_);
  siPixelStatusToken_ = iC.consumes<SiPixelDetectorStatus,edm::InLumi>(siPixelStatusTag_);

  LogInfo("SiPixelStatusManager") 
    << "Output base: " << outputBase_ 
    << std::endl;
  reset();
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusManager::~SiPixelStatusManager(){
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusManager::reset(){
     siPixelStatusMap_.clear();
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusManager::readLumi(const LuminosityBlock& iLumi){

  edm::Handle<SiPixelDetectorStatus> siPixelStatusHandle;
  iLumi.getByToken(siPixelStatusToken_, siPixelStatusHandle);

  if(siPixelStatusHandle.isValid()) { // check the product
    siPixelStatusMap_[iLumi.luminosityBlock()] = *siPixelStatusHandle;
  }
  else {
    LogInfo("SiPixelStatusManager")
        << "Lumi: " << iLumi.luminosityBlock() << std::endl;
    LogInfo("SiPixelStatusManager")
        << " SiPixelDetectorStatus is not valid!" << std::endl;
  }

}



//--------------------------------------------------------------------------------------------------
void SiPixelStatusManager::createBadComponents(){

  siPixelStatusMap_iterator firstStatus    = siPixelStatusMap_.begin();
  siPixelStatusMap_iterator lastStatus     = siPixelStatusMap_.end();

  std::map<LuminosityBlockNumber_t,SiPixelDetectorStatus> tmpSiPixelStatusMap_;

  if(outputBase_ == "nLumibased" && nLumi_>1){ // can't be equal to 1

    int iterationLumi = 0;
    // number of IOV 
    // if the total number of Lumi Blocks can't be completely divided by nLumi_,
    // the residual Lumi Blocks will be as the last IOV
    int nIOV = siPixelStatusMap_.size()/nLumi_;
    if(siPixelStatusMap_.size()%nLumi_!=0) nIOV = nIOV + 1;

    LuminosityBlockNumber_t tmpLumi;
    SiPixelDetectorStatus tmpSiPixelStatus;
    for (siPixelStatusMap_iterator it = firstStatus; it != lastStatus; it++) {
  
        // this is the begining of an IOV
        if(iterationLumi%nLumi_==0){
           tmpLumi = it->first;
           tmpSiPixelStatus = it->second; 
	}

        // keep update detector status up to nLumi_ lumi sections
        if(iterationLumi%nLumi_>0){
          tmpSiPixelStatus.updateDetectorStatus(it->second);
        }

        siPixelStatusMap_iterator currentIt = it;
        siPixelStatusMap_iterator nextIt = (++currentIt);
        // wirte out if current lumi is the last lumi-section in the IOV
        if(iterationLumi%nLumi_==nLumi_-1 || nextIt==lastStatus) 
        {
             // fill it into a new map (with IOV structured)
             tmpSiPixelStatusMap_[tmpLumi] = tmpSiPixelStatus;
        }

        iterationLumi=iterationLumi+1;
    }

    // check if not enough number of Lumi in the last IOV
    // if it is, combine last IOV with the IOV before it
    if(siPixelStatusMap_.size()%nLumi_!=0){

       siPixelStatusMap_iterator iterEnd = tmpSiPixelStatusMap_.end();
       siPixelStatusMap_iterator iterLastIOV = std::prev(iterEnd);
       siPixelStatusMap_iterator iterBeforeLastIOV = std::prev(iterLastIOV);

       (iterBeforeLastIOV->second).updateDetectorStatus(iterLastIOV->second);
       tmpSiPixelStatusMap_.erase(iterLastIOV);

    }

    siPixelStatusMap_.clear();
    siPixelStatusMap_ = tmpSiPixelStatusMap_;

  }
  else if(outputBase_ == "dynamicLumibased"){

    double aveDigiOcc = 1.0*aveDigiOcc_;
  
    LuminosityBlockNumber_t tmpLumi;
    SiPixelDetectorStatus tmpSiPixelStatus;
    bool isNewIOV = true;

    int counter = 0;
    for (siPixelStatusMap_iterator it = firstStatus; it != lastStatus; it++) {

         if(isNewIOV){ // if it is new IOV, init with the current data
               tmpLumi = it->first;
               tmpSiPixelStatus = it->second;
         }
         else{ // if it is not new IOV, append current data
               tmpSiPixelStatus.updateDetectorStatus(it->second);
         }

         // if reaching the end of data, write the last IOV to the map whatsoevec
         siPixelStatusMap_iterator currentIt = it;
         siPixelStatusMap_iterator nextIt = (++currentIt);
         if(tmpSiPixelStatus.perRocDigiOcc()<aveDigiOcc && nextIt!=lastStatus){
            isNewIOV = false; // if digi occ is not enough, next data will not belong to new IOV
         }
         else{ // if (accunumated) digi occ is enough, write the data to the map
           isNewIOV = true;
           tmpSiPixelStatusMap_[tmpLumi]=tmpSiPixelStatus;
           // so next loop is the begining of a new IOV
         }
         counter++;

   } // end of siPixelStatusMap

   // check whether last IOV has enough statistics
   // if not, combine with previous IOV
   siPixelStatusMap_iterator iterEnd = tmpSiPixelStatusMap_.end();
   siPixelStatusMap_iterator iterLastIOV = std::prev(iterEnd);
   siPixelStatusMap_iterator iterBeforeLastIOV = std::prev(iterLastIOV);
   if((iterLastIOV->second).perRocDigiOcc()<aveDigiOcc){
      (iterBeforeLastIOV->second).updateDetectorStatus(iterLastIOV->second);
      tmpSiPixelStatusMap_.erase(iterLastIOV);
   }

   siPixelStatusMap_.clear();
   siPixelStatusMap_ = tmpSiPixelStatusMap_;

  }
  else if(outputBase_ == "runbased" || ( (int(siPixelStatusMap_.size()) <= nLumi_ && outputBase_ == "nLumibased")) ){

    siPixelStatusMap_iterator firstStatus    = siPixelStatusMap_.begin();
    siPixelStatusMap_iterator lastStatus     = siPixelStatusMap_.end();

    LuminosityBlockNumber_t tmpLumi = firstStatus->first;
    SiPixelDetectorStatus tmpSiPixelStatus = firstStatus->second;

    siPixelStatusMap_iterator nextStatus = ++siPixelStatusMap_.begin();
    for (siPixelStatusMap_iterator it = nextStatus; it != lastStatus; it++) {
          tmpSiPixelStatus.updateDetectorStatus(it->second);
    }

    siPixelStatusMap_.clear();
    siPixelStatusMap_[tmpLumi] = tmpSiPixelStatus;

  }
  else{
    LogInfo("SiPixelStatusManager")
      << "Unrecognized payload outputBase parameter: " << outputBase_
      << endl;    
  }


}


void SiPixelStatusManager::createStuckTBMs(){

    // initialize the first IOV and SiPixelDetector status (in the first IOV)
    siPixelStatusMap_iterator firstStatus = siPixelStatusMap_.begin();
    LuminosityBlockNumber_t   firstLumi = firstStatus->first;
    SiPixelDetectorStatus     firstStuckTBMs = firstStatus->second;
    stuckTBMsMap_[firstLumi] = firstStuckTBMs.getStuckTBMsRocs();   

    siPixelStatusMap_iterator lastStatus     = siPixelStatusMap_.end();

    ///////////
    bool sameAsLastIOV = true;
    LuminosityBlockNumber_t previousLumi = firstLumi;
    siPixelStatusMap_iterator secondStatus = ++siPixelStatusMap_.begin();
    for (siPixelStatusMap_iterator it = secondStatus; it != lastStatus; it++) {
       
        LuminosityBlockNumber_t tmpLumi = it->first;
        SiPixelDetectorStatus tmpStuckTBMs = it->second;
        std::map<int,std::vector<int> >tmpBadRocLists = tmpStuckTBMs.getStuckTBMsRocs();

        std::map<int, SiPixelModuleStatus>::iterator itModEnd = tmpStuckTBMs.end();
        for (std::map<int, SiPixelModuleStatus>::iterator itMod = tmpStuckTBMs.begin(); itMod != itModEnd; ++itMod) 
        {
            int detid = itMod->first;
            // if the badroc list differs for any detid, update the payload
            if(tmpBadRocLists[detid]!=(stuckTBMsMap_[previousLumi])[detid]){
               sameAsLastIOV = false;
               return;
            }       
        }

        if(sameAsLastIOV==false){
            //only write new IOV when this Lumi is not equal to the previous one

            stuckTBMsMap_[tmpLumi] = tmpBadRocLists; 
            previousLumi = tmpLumi;

        }

    }

}
