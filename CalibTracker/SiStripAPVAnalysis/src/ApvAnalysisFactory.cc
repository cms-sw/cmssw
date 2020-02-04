#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"
//#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

#include "CalibTracker/SiStripAPVAnalysis/interface/TT6NTPedestalCalculator.h"

using namespace std;
ApvAnalysisFactory::ApvAnalysisFactory(string theAlgorithmType,
                                       int theNumCMstripsInGroup,
                                       int theMaskCalcFlag,
                                       float theMaskNoiseCut,
                                       float theMaskDeadCut,
                                       float theMaskTruncCut,
                                       float theCutToAvoidSignal,
                                       int theEventInitNumber,
                                       int theEventIterNumber) {
  theAlgorithmType_ = theAlgorithmType;
  theNumCMstripsInGroup_ = theNumCMstripsInGroup;
  theMaskCalcFlag_ = theMaskCalcFlag;
  theMaskNoiseCut_ = theMaskNoiseCut;
  theMaskDeadCut_ = theMaskDeadCut;
  theMaskTruncCut_ = theMaskTruncCut;
  theCutToAvoidSignal_ = theCutToAvoidSignal;
  theEventInitNumber_ = theEventInitNumber;
  theEventIterNumber_ = theEventIterNumber;
}

ApvAnalysisFactory::ApvAnalysisFactory(const edm::ParameterSet& pset) {
  theCMType_ = pset.getParameter<string>("CMType");
  useDB_ = pset.getParameter<bool>("useDB");

  theAlgorithmType_ = pset.getParameter<string>("CalculatorAlgorithm");
  theNumCMstripsInGroup_ = pset.getParameter<int>("NumCMstripsInGroup");
  theMaskCalcFlag_ = pset.getParameter<int>("MaskCalculationFlag");

  theMaskNoiseCut_ = pset.getParameter<double>("MaskNoiseCut");
  theMaskDeadCut_ = pset.getParameter<double>("MaskDeadCut");
  theMaskTruncCut_ = pset.getParameter<double>("MaskTruncationCut");
  theCutToAvoidSignal_ = pset.getParameter<double>("CutToAvoidSignal");

  theEventInitNumber_ = pset.getParameter<int>("NumberOfEventsForInit");
  theEventIterNumber_ = pset.getParameter<int>("NumberOfEventsForIteration");
  apvMap_.clear();
}
//----------------------------------
ApvAnalysisFactory::~ApvAnalysisFactory() {
  ApvAnalysisFactory::ApvAnalysisMap::iterator it = apvMap_.begin();
  for (; it != apvMap_.end(); it++) {
    vector<ApvAnalysis*>::iterator myApv = (*it).second.begin();
    for (; myApv != (*it).second.end(); myApv++)
      deleteApv(*myApv);
  }
  apvMap_.clear();
}

//----------------------------------

bool ApvAnalysisFactory::instantiateApvs(uint32_t detId, int numberOfApvs) {
  ApvAnalysisFactory::ApvAnalysisMap::iterator CPos = apvMap_.find(detId);
  if (CPos != apvMap_.end()) {
    cout << " APVs for Detector Id " << detId << " already created !!!" << endl;
    ;
    return false;
  }
  vector<ApvAnalysis*> temp;
  for (int i = 0; i < numberOfApvs; i++) {
    ApvAnalysis* apvTmp = new ApvAnalysis(theEventIterNumber_);
    //      constructAuxiliaryApvClasses(apvTmp);
    constructAuxiliaryApvClasses(apvTmp, detId, i);
    temp.push_back(apvTmp);
  }
  apvMap_.insert(pair<uint32_t, vector<ApvAnalysis*> >(detId, temp));
  return true;
}

std::vector<ApvAnalysis*> ApvAnalysisFactory::getApvAnalysis(const uint32_t nDET_ID) {
  ApvAnalysisMap::const_iterator _apvAnalysisIter = apvMap_.find(nDET_ID);

  return apvMap_.end() != _apvAnalysisIter ? _apvAnalysisIter->second : std::vector<ApvAnalysis*>();
}

void ApvAnalysisFactory::constructAuxiliaryApvClasses(ApvAnalysis* theAPV, uint32_t detId, int thisApv) {
  //----------------------------------------------------------------
  // Create the ped/noise/CMN calculators, zero suppressors etc.
  // (Is called by addDetUnitAndConstructApvs()).
  //
  // N.B. Don't call this twice for the same APV!
  //-----------------------------------------------------------------
  // cout<<"VirtualApvAnalysisFactory::constructAuxiliaryApvClasses"<<endl;
  TkPedestalCalculator* thePedestal = nullptr;
  TkNoiseCalculator* theNoise = nullptr;
  TkApvMask* theMask = nullptr;
  TkCommonModeCalculator* theCM = nullptr;

  TkCommonMode* theCommonMode = new TkCommonMode();
  TkCommonModeTopology* theTopology = new TkCommonModeTopology(128, theNumCMstripsInGroup_);
  theCommonMode->setTopology(theTopology);

  // Create desired algorithms.
  if (theAlgorithmType_ == "TT6") {
    theMask = new TT6ApvMask(theMaskCalcFlag_, theMaskNoiseCut_, theMaskDeadCut_, theMaskTruncCut_);
    theNoise = new TT6NoiseCalculator(theEventInitNumber_, theEventIterNumber_, theCutToAvoidSignal_);
    thePedestal = new TT6PedestalCalculator(theEventInitNumber_, theEventIterNumber_, theCutToAvoidSignal_);
    theCM = new TT6CommonModeCalculator(theNoise, theMask, theCutToAvoidSignal_);
  } else if ("TT6NT" == theAlgorithmType_) {
    theMask = new TT6ApvMask(theMaskCalcFlag_, theMaskNoiseCut_, theMaskDeadCut_, theMaskTruncCut_);
    theNoise = new TT6NoiseCalculator(theEventInitNumber_, theEventIterNumber_, theCutToAvoidSignal_);
    thePedestal = new TT6NTPedestalCalculator;
    theCM = new TT6CommonModeCalculator(theNoise, theMask, theCutToAvoidSignal_);
  } else if (theAlgorithmType_ == "MIX") {
    // the mask as to be defined also for SimplePedCalculator
    theMask = new TT6ApvMask(theMaskCalcFlag_, theMaskNoiseCut_, theMaskDeadCut_, theMaskTruncCut_);

    thePedestal = new SimplePedestalCalculator(theEventInitNumber_);

    theNoise = new SimpleNoiseCalculator(theEventInitNumber_, useDB_);

    if (theCMType_ == "Median") {
      theCM = new MedianCommonModeCalculator();
    } else {
      cout << "Sorry Only Median is available for now, Mean and FastLinear are coming soon" << endl;
    }
  }

  if (theCommonMode)
    theCM->setCM(theCommonMode);
  if (thePedestal)
    theAPV->setPedestalCalculator(*thePedestal);
  if (theNoise)
    theAPV->setNoiseCalculator(*theNoise);
  if (theMask)
    theAPV->setMask(*theMask);
  if (theCM)
    theAPV->setCommonModeCalculator(*theCM);
}

void ApvAnalysisFactory::updatePair(uint32_t detId, size_t pairNumber, const edm::DetSet<SiStripRawDigi>& in) {
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    size_t iter = 0;

    for (vector<ApvAnalysis*>::const_iterator apvIt = (apvAnalysisIt->second).begin();
         apvIt != (apvAnalysisIt->second).end();
         apvIt++) {
      if (iter == pairNumber * 2 || iter == (2 * pairNumber + 1)) {
        //      cout << "ApvAnalysisFactory::updatePair pair number " << pairNumber << endl;
        //      cout << "ApvAnlysis will be updated for the apv # " << iter << endl;

        edm::DetSet<SiStripRawDigi> tmpRawDigi;
        tmpRawDigi.data.reserve(128);

        size_t startStrip = 128 * (iter % 2);
        size_t stopStrip = startStrip + 128;

        for (size_t istrip = startStrip; istrip < stopStrip; istrip++) {
          if (in.data.size() <= istrip)
            tmpRawDigi.data.push_back(SiStripRawDigi(0));
          else
            tmpRawDigi.data.push_back(in.data[istrip]);  //maybe dangerous
        }

        (*apvIt)->newEvent();
        (*apvIt)->updateCalibration(tmpRawDigi);
      }

      iter++;
    }
  }

}  // void

void ApvAnalysisFactory::update(uint32_t detId, const edm::DetSet<SiStripRawDigi>& in) {
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    size_t i = 0;
    for (vector<ApvAnalysis*>::const_iterator apvIt = (apvAnalysisIt->second).begin();
         apvIt != (apvAnalysisIt->second).end();
         apvIt++) {
      edm::DetSet<SiStripRawDigi> tmpRawDigi;
      //it is missing the detId ...
      tmpRawDigi.data.reserve(128);
      size_t startStrip = 128 * i;
      size_t stopStrip = startStrip + 128;

      for (size_t istrip = startStrip; istrip < stopStrip; istrip++) {
        if (in.data.size() <= istrip)
          tmpRawDigi.data.push_back(SiStripRawDigi(0));
        else
          tmpRawDigi.data.push_back(in.data[istrip]);  //maybe dangerous
      }

      (*apvIt)->newEvent();
      (*apvIt)->updateCalibration(tmpRawDigi);
      i++;
    }
  }
}

void ApvAnalysisFactory::getPedestal(uint32_t detId, int apvNumber, ApvAnalysis::PedestalType& peds) {
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> myApvs = apvAnalysisIt->second;
    peds = myApvs[apvNumber]->pedestalCalculator().pedestal();
  }
}

void ApvAnalysisFactory::getPedestal(uint32_t detId, ApvAnalysis::PedestalType& peds) {
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = apvAnalysisIt->second;
    for (vector<ApvAnalysis*>::const_iterator it = theApvs.begin(); it != theApvs.end(); it++) {
      ApvAnalysis::PedestalType tmp = (*it)->pedestalCalculator().pedestal();
      for (ApvAnalysis::PedestalType::const_iterator pit = tmp.begin(); pit != tmp.end(); pit++)
        peds.push_back(*pit);
    }
  }
}
float ApvAnalysisFactory::getStripPedestal(uint32_t detId, int stripNumber) {
  //Get the pedestal for a given apv
  ApvAnalysis::PedestalType temp;
  int apvNumb = int(stripNumber / 128.);
  int stripN = (stripNumber - apvNumb * 128);

  getPedestal(detId, apvNumb, temp);
  return temp[stripN];
}
void ApvAnalysisFactory::getNoise(uint32_t detId, int apvNumber, ApvAnalysis::PedestalType& noise) {
  //Get the pedestal for a given apv
  noise.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = apvAnalysisIt->second;

    noise = theApvs[apvNumber]->noiseCalculator().noise();
  }
}

float ApvAnalysisFactory::getStripNoise(uint32_t detId, int stripNumber) {
  //Get the pedestal for a given apv
  ApvAnalysis::PedestalType temp;
  int apvNumb = int(stripNumber / 128.);
  int stripN = (stripNumber - apvNumb * 128);

  getNoise(detId, apvNumb, temp);
  return temp[stripN];
}

void ApvAnalysisFactory::getNoise(uint32_t detId, ApvAnalysis::PedestalType& peds) {
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator theApvs_map = apvMap_.find(detId);
  if (theApvs_map != apvMap_.end()) {
    vector<ApvAnalysis*>::const_iterator theApvs = (theApvs_map->second).begin();
    for (; theApvs != (theApvs_map->second).end(); theApvs++) {
      ApvAnalysis::PedestalType tmp = (*theApvs)->noiseCalculator().noise();
      for (ApvAnalysis::PedestalType::const_iterator pit = tmp.begin(); pit != tmp.end(); pit++)
        peds.push_back(*pit);
    }
  }
}

void ApvAnalysisFactory::getRawNoise(uint32_t detId, int apvNumber, ApvAnalysis::PedestalType& noise) {
  //Get the pedestal for a given apv
  noise.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = apvAnalysisIt->second;

    noise = theApvs[apvNumber]->pedestalCalculator().rawNoise();
  }
}

float ApvAnalysisFactory::getStripRawNoise(uint32_t detId, int stripNumber) {
  //Get the pedestal for a given apv
  ApvAnalysis::PedestalType temp;
  int apvNumb = int(stripNumber / 128.);
  int stripN = (stripNumber - apvNumb * 128);

  getRawNoise(detId, apvNumb, temp);
  return temp[stripN];
}

void ApvAnalysisFactory::getRawNoise(uint32_t detId, ApvAnalysis::PedestalType& peds) {
  //Get the pedestal for a given apv
  peds.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator theApvs_map = apvMap_.find(detId);
  if (theApvs_map != apvMap_.end()) {
    vector<ApvAnalysis*>::const_iterator theApvs = (theApvs_map->second).begin();
    for (; theApvs != (theApvs_map->second).end(); theApvs++) {
      ApvAnalysis::PedestalType tmp = (*theApvs)->pedestalCalculator().rawNoise();
      for (ApvAnalysis::PedestalType::const_iterator pit = tmp.begin(); pit != tmp.end(); pit++)
        peds.push_back(*pit);
    }
  }
}

vector<float> ApvAnalysisFactory::getCommonMode(uint32_t detId, int apvNumber) {
  vector<float> tmp;
  tmp.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator theApvs_map = apvMap_.find(detId);
  if (theApvs_map != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = theApvs_map->second;

    tmp = theApvs[apvNumber]->commonModeCalculator().commonMode()->returnAsVector();
  }
  return tmp;
}
void ApvAnalysisFactory::getCommonMode(uint32_t detId, ApvAnalysis::PedestalType& tmp) {
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = apvAnalysisIt->second;
    for (unsigned int i = 0; i < theApvs.size(); i++) {
      //To be fixed. We return only the first one in the vector.
      vector<float> tmp_cm = theApvs[i]->commonModeCalculator().commonMode()->returnAsVector();
      for (unsigned int it = 0; it < tmp_cm.size(); it++)
        tmp.push_back(tmp_cm[it]);
    }
  }
}

void ApvAnalysisFactory::getMask(uint32_t det_id, TkApvMask::MaskType& tmp) {
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(det_id);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = apvAnalysisIt->second;
    for (unsigned int i = 0; i < theApvs.size(); i++) {
      TkApvMask::MaskType theMaskType = (theApvs[i]->mask()).mask();
      //cout <<"theMaskType size "<<theMaskType.size()<<endl;

      for (unsigned int ii = 0; ii < theMaskType.size(); ii++) {
        tmp.push_back(theMaskType[ii]);
        //cout <<"The Value "<<theMaskType[ii]<<" "<<ii<<endl;
      }
    }
  }
}
bool ApvAnalysisFactory::isUpdating(uint32_t detId) {
  bool updating = true;
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    for (vector<ApvAnalysis*>::const_iterator apvIt = (apvAnalysisIt->second).begin();
         apvIt != (apvAnalysisIt->second).end();
         apvIt++) {
      if (!((*apvIt)->pedestalCalculator().status()->isUpdating()))
        updating = false;
    }
  }
  return updating;
}

void ApvAnalysisFactory::deleteApv(ApvAnalysis* apv) {
  delete &(apv->pedestalCalculator());
  delete &(apv->noiseCalculator());
  delete &(apv->mask());
  delete &(apv->commonModeCalculator().commonMode()->topology());
  delete (apv->commonModeCalculator().commonMode());
  delete &(apv->commonModeCalculator());
  delete apv;
}
//
// -- Get Common Mode Slope
//
float ApvAnalysisFactory::getCommonModeSlope(uint32_t detId, int apvNumber) {
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator theApvs_map = apvMap_.find(detId);
  float tmp = -100.0;
  if (theApvs_map != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = theApvs_map->second;
    tmp = theApvs[apvNumber]->commonModeCalculator().getCMSlope();
    return tmp;
  }
  return tmp;
}
void ApvAnalysisFactory::getCommonModeSlope(uint32_t detId, ApvAnalysis::PedestalType& tmp) {
  tmp.clear();
  map<uint32_t, vector<ApvAnalysis*> >::const_iterator apvAnalysisIt = apvMap_.find(detId);
  if (apvAnalysisIt != apvMap_.end()) {
    vector<ApvAnalysis*> theApvs = apvAnalysisIt->second;
    for (unsigned int i = 0; i < theApvs.size(); i++) {
      tmp.push_back(theApvs[i]->commonModeCalculator().getCMSlope());
    }
  }
}
