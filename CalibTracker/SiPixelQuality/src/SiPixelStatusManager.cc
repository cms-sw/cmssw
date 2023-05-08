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
SiPixelStatusManager::SiPixelStatusManager() {}

//--------------------------------------------------------------------------------------------------
SiPixelStatusManager::SiPixelStatusManager(const ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : outputBase_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters")
                      .getUntrackedParameter<std::string>("outputBase")),
      aveDigiOcc_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters")
                      .getUntrackedParameter<int>("aveDigiOcc")),
      nLumi_(iConfig.getParameter<edm::ParameterSet>("SiPixelStatusManagerParameters")
                 .getUntrackedParameter<int>("resetEveryNLumi")),
      moduleName_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters")
                      .getUntrackedParameter<std::string>("moduleName")),
      label_(iConfig.getParameter<ParameterSet>("SiPixelStatusManagerParameters")
                 .getUntrackedParameter<std::string>("label")) {
  edm::InputTag siPixelStatusTag_(moduleName_, label_);
  siPixelStatusToken_ = iC.consumes<SiPixelDetectorStatus, edm::InLumi>(siPixelStatusTag_);

  LogInfo("SiPixelStatusManager") << "Output base: " << outputBase_ << std::endl;
  reset();
}

//--------------------------------------------------------------------------------------------------
SiPixelStatusManager::~SiPixelStatusManager() {}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusManager::reset() {
  siPixelStatusMap_.clear();
  siPixelStatusVtr_.clear();
}

//--------------------------------------------------------------------------------------------------
bool SiPixelStatusManager::rankByLumi(SiPixelDetectorStatus status1, SiPixelDetectorStatus status2) {
  return (status1.getLSRange().first < status2.getLSRange().first);
}

void SiPixelStatusManager::createPayloads() {
  //only create std::map payloads when the number of non-zero DIGI lumi sections is greater than ZERO otherwise segmentation fault
  if (!siPixelStatusVtr_.empty()) {
    // sort the vector according to lumi
    std::sort(siPixelStatusVtr_.begin(), siPixelStatusVtr_.end(), SiPixelStatusManager::rankByLumi);

    // create FEDerror25 ROCs and bad ROCs from PCL
    SiPixelStatusManager::createFEDerror25();
    SiPixelStatusManager::createBadComponents();

    // realse the cost of siPixelStatusVtr_ since it is not needed anymore
    siPixelStatusVtr_.clear();
  }
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusManager::readLumi(const LuminosityBlock& iLumi) {
  edm::Handle<SiPixelDetectorStatus> siPixelStatusHandle;
  iLumi.getByToken(siPixelStatusToken_, siPixelStatusHandle);

  if (siPixelStatusHandle.isValid()) {  // check the product
    SiPixelDetectorStatus tmpStatus = (*siPixelStatusHandle);
    if (tmpStatus.digiOccDET() > 0) {  // only put in SiPixelDetectorStatus with non zero digi (pixel hit)
      siPixelStatusVtr_.push_back(tmpStatus);
    }
  } else {
    edm::LogWarning("SiPixelStatusManager") << " SiPixelDetectorStatus is not valid for run " << iLumi.run() << " lumi "
                                            << iLumi.luminosityBlock() << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
void SiPixelStatusManager::createBadComponents() {
  siPixelStatusVtr_iterator firstStatus = siPixelStatusVtr_.begin();
  siPixelStatusVtr_iterator lastStatus = siPixelStatusVtr_.end();

  siPixelStatusMap_.clear();

  // doesn't work for nLumi_=1 cos any integer can be completely divided by 1
  if (outputBase_ == "nLumibased" && nLumi_ > 1) {
    // if the total number of Lumi Blocks can't be completely divided by nLumi_,
    // the residual Lumi Blocks will be as the last IOV
    int iterationLumi = 0;

    LuminosityBlockNumber_t tmpLumi;
    SiPixelDetectorStatus tmpSiPixelStatus;
    for (siPixelStatusVtr_iterator it = firstStatus; it != lastStatus; it++) {
      // this is the begining of an IOV
      if (iterationLumi % nLumi_ == 0) {
        tmpLumi = edm::LuminosityBlockNumber_t(it->getLSRange().first);
        tmpSiPixelStatus = (*it);
      }

      // keep update detector status up to nLumi_ lumi sections
      if (iterationLumi % nLumi_ > 0) {
        tmpSiPixelStatus.updateDetectorStatus((*it));
        tmpSiPixelStatus.setLSRange(int(tmpLumi), (*it).getLSRange().second);
      }

      siPixelStatusVtr_iterator currentIt = it;
      siPixelStatusVtr_iterator nextIt = std::next(currentIt);
      // wirte out if current lumi is the last lumi-section in the IOV
      if (iterationLumi % nLumi_ == nLumi_ - 1 || nextIt == lastStatus) {
        // fill it into a new map (with IOV structured)
        siPixelStatusMap_[tmpLumi] = tmpSiPixelStatus;
      }

      iterationLumi = iterationLumi + 1;
    }

    // check whether there is not enough number of Lumi in the last IOV
    // (only when siPixelStatusVtr_.size() > nLumi_ or equivalently current siPixelStatusMap_.size()>1
    //            (otherwise there will be only one IOV, and not previous IOV before the last IOV)
    //            and the number of lumi can not be completely divided by the nLumi_.
    //                (then the number of lumis in the last IOV is equal to the residual, which is less than nLumi_)
    // if it is, combine last IOV with the IOV before it
    if (siPixelStatusVtr_.size() % nLumi_ != 0 && siPixelStatusMap_.size() > 1) {
      // start from the iterator of the end of std::map
      siPixelStatusMap_iterator iterEnd = siPixelStatusMap_.end();
      // the last IOV
      siPixelStatusMap_iterator iterLastIOV = std::prev(iterEnd);
      // the IOV before the last IOV
      siPixelStatusMap_iterator iterBeforeLastIOV = std::prev(iterLastIOV);

      // combine the last IOV data to the IOV before the last IOV
      (iterBeforeLastIOV->second).updateDetectorStatus(iterLastIOV->second);
      (iterBeforeLastIOV->second)
          .setLSRange((iterBeforeLastIOV->second).getLSRange().first, (iterLastIOV->second).getLSRange().second);

      // delete the last IOV, so the IOV before the last IOV becomes the new last IOV
      siPixelStatusMap_.erase(iterLastIOV);
    }

  } else if (outputBase_ == "dynamicLumibased") {
    double aveDigiOcc = 1.0 * aveDigiOcc_;

    edm::LuminosityBlockNumber_t tmpLumi;
    SiPixelDetectorStatus tmpSiPixelStatus;
    bool isNewIOV = true;

    for (siPixelStatusVtr_iterator it = firstStatus; it != lastStatus; it++) {
      if (isNewIOV) {  // if it is new IOV, init with the current data
        tmpLumi = edm::LuminosityBlockNumber_t(it->getLSRange().first);
        tmpSiPixelStatus = (*it);
      } else {  // if it is not new IOV, append current data
        tmpSiPixelStatus.updateDetectorStatus((*it));
        tmpSiPixelStatus.setLSRange(int(tmpLumi), (*it).getLSRange().second);
      }

      // if reaching the end of data, write the last IOV to the map whatsoevec
      siPixelStatusVtr_iterator currentIt = it;
      siPixelStatusVtr_iterator nextIt = std::next(currentIt);
      if (tmpSiPixelStatus.perRocDigiOcc() < aveDigiOcc && nextIt != lastStatus) {
        isNewIOV = false;  // if digi occ is not enough, next data will not belong to new IOV
      } else {             // if (accunumated) digi occ is enough, write the data to the map
        isNewIOV = true;
        siPixelStatusMap_[tmpLumi] = tmpSiPixelStatus;
        // so next loop is the begining of a new IOV
      }

    }  // end of siPixelStatusMap

    // check whether last IOV has enough statistics
    // (ONLY when there are more than oneIOV(otherwise there is NO previous IOV before the last IOV) )
    // if not, combine with previous IOV
    if (siPixelStatusMap_.size() > 1) {
      // start from the end iterator of the std::map
      siPixelStatusMap_iterator iterEnd = siPixelStatusMap_.end();
      // the last IOV
      siPixelStatusMap_iterator iterLastIOV = std::prev(iterEnd);
      // if the statistics of the last IOV is not enough
      if ((iterLastIOV->second).perRocDigiOcc() < aveDigiOcc) {
        // the IOV before the last IOV of the map
        siPixelStatusMap_iterator iterBeforeLastIOV = std::prev(iterLastIOV);
        // combine the last IOV data to the IOV before the last IOV
        (iterBeforeLastIOV->second).updateDetectorStatus(iterLastIOV->second);
        (iterBeforeLastIOV->second)
            .setLSRange((iterBeforeLastIOV->second).getLSRange().first, (iterLastIOV->second).getLSRange().second);
        // erase the last IOV, so the IOV before the last IOV becomes the new last IOV
        siPixelStatusMap_.erase(iterLastIOV);
      }
    }

  } else if (outputBase_ == "runbased" || ((int(siPixelStatusVtr_.size()) <= nLumi_ && outputBase_ == "nLumibased"))) {
    edm::LuminosityBlockNumber_t tmpLumi = edm::LuminosityBlockNumber_t(firstStatus->getLSRange().first);
    SiPixelDetectorStatus tmpSiPixelStatus = (*firstStatus);

    siPixelStatusVtr_iterator nextStatus = ++siPixelStatusVtr_.begin();
    for (siPixelStatusVtr_iterator it = nextStatus; it != lastStatus; it++) {
      tmpSiPixelStatus.updateDetectorStatus((*it));
      tmpSiPixelStatus.setLSRange(int(tmpLumi), (*it).getLSRange().second);
    }

    siPixelStatusMap_[tmpLumi] = tmpSiPixelStatus;

  } else {
    LogInfo("SiPixelStatusManager") << "Unrecognized payload outputBase parameter: " << outputBase_ << endl;
  }
}

void SiPixelStatusManager::createFEDerror25() {
  // initialize the first IOV and SiPixelDetector status (in the first IOV)
  siPixelStatusVtr_iterator firstStatus = siPixelStatusVtr_.begin();
  edm::LuminosityBlockNumber_t firstLumi = edm::LuminosityBlockNumber_t(firstStatus->getLSRange().first);
  SiPixelDetectorStatus firstFEDerror25 = (*firstStatus);
  FEDerror25Map_[firstLumi] = firstFEDerror25.getFEDerror25Rocs();

  siPixelStatusVtr_iterator lastStatus = siPixelStatusVtr_.end();

  ///////////
  bool sameAsLastIOV = true;
  edm::LuminosityBlockNumber_t previousLumi = firstLumi;

  siPixelStatusVtr_iterator secondStatus = std::next(siPixelStatusVtr_.begin());
  for (siPixelStatusVtr_iterator it = secondStatus; it != lastStatus; it++) {
    // init for each lumi section (iterator)
    edm::LuminosityBlockNumber_t tmpLumi = edm::LuminosityBlockNumber_t(it->getLSRange().first);
    SiPixelDetectorStatus tmpFEDerror25 = (*it);

    std::map<int, std::vector<int> > tmpBadRocLists = tmpFEDerror25.getFEDerror25Rocs();

    std::map<int, SiPixelModuleStatus>::iterator itModEnd = tmpFEDerror25.end();
    for (std::map<int, SiPixelModuleStatus>::iterator itMod = tmpFEDerror25.begin(); itMod != itModEnd; ++itMod) {
      int detid = itMod->first;
      // if the badroc list differs for any detid, update the payload
      if (tmpBadRocLists[detid] != (FEDerror25Map_[previousLumi])[detid]) {
        sameAsLastIOV = false;
        break;  // jump out of the loop once a new payload is found
      }
    }

    if (sameAsLastIOV == false) {
      //only write new IOV when this Lumi's FEDerror25 ROC list is not equal to the previous one
      FEDerror25Map_[tmpLumi] = tmpBadRocLists;
      // and reset
      previousLumi = tmpLumi;
      sameAsLastIOV = true;
    }
  }
}
