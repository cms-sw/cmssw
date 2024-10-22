#ifndef ApvAnalysisFactory_h
#define ApvAnalysisFactory_h
//#define DEBUG_INSTANCE_COUNTING

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysis.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkStateMachine.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonMode.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TT6ApvMask.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TT6NoiseCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TT6PedestalCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/TT6CommonModeCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/MedianCommonModeCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/SimplePedestalCalculator.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/SimpleNoiseCalculator.h"
#include <vector>
#include <iostream>
#include <map>
#include <algorithm>
#include <string>

class ApvAnalysisFactory {
public:
  typedef std::vector<ApvAnalysis*> ApvAnalysisVector;

  ApvAnalysisFactory(std::string theAlgorithmType,
                     int theNumCMstripsInGroup,
                     int theMaskCalcFlag,
                     float theMaskNoiseCut,
                     float theMaskDeadCut,
                     float theMaskTruncCut,
                     float theCutToAvoidSignal,
                     int theEventInitNumber,
                     int theEventIterNumber);
  ApvAnalysisFactory(const edm::ParameterSet& pset);
  ~ApvAnalysisFactory();

  bool instantiateApvs(uint32_t det_id, int numberOfApvs);

  ApvAnalysisVector getApvAnalysis(const uint32_t nDET_ID);

  void getPedestal(uint32_t det_id, int apvNumber, ApvAnalysis::PedestalType& peds);
  void getPedestal(uint32_t det_id, ApvAnalysis::PedestalType& peds);
  float getStripPedestal(uint32_t det_id, int stripNumber);

  void getMask(uint32_t det_id, TkApvMask::MaskType& tmp);
  void getCommonMode(uint32_t det_id, ApvAnalysis::PedestalType& tmp);
  std::vector<float> getCommonMode(uint32_t det_id, int apvNumber);

  void getNoise(uint32_t det_id, int apvNumber, ApvAnalysis::PedestalType& noise);
  void getNoise(uint32_t det_id, ApvAnalysis::PedestalType& noise);
  float getStripNoise(uint32_t det_id, int stripNumber);

  void getRawNoise(uint32_t det_id, int apvNumber, ApvAnalysis::PedestalType& noise);
  void getRawNoise(uint32_t det_id, ApvAnalysis::PedestalType& noise);
  float getStripRawNoise(uint32_t det_id, int stripNumber);

  void getCommonModeSlope(uint32_t det_id, ApvAnalysis::PedestalType& tmp);
  float getCommonModeSlope(uint32_t det_id, int apvNumber);

  void update(uint32_t det_id, const edm::DetSet<SiStripRawDigi>& in);
  void updatePair(uint32_t det_id, size_t apvPair, const edm::DetSet<SiStripRawDigi>& in);

  std::string getStatus(uint32_t det_id);

  bool isUpdating(uint32_t detId);

  typedef std::map<uint32_t, ApvAnalysisVector> ApvAnalysisMap;
  typedef std::map<ApvAnalysis*, int> MapNumberingPosition;
  typedef std::map<ApvAnalysis*, uint32_t> InverseMap;

private:
  //  void  constructAuxiliaryApvClasses (ApvAnalysis* thisApv);
  void constructAuxiliaryApvClasses(ApvAnalysis* theApv, uint32_t det_id, int thisApv);
  void deleteApv(ApvAnalysis* apv);
  ApvAnalysisMap apvMap_;
  std::string theAlgorithmType_;
  int theNumCMstripsInGroup_;
  int theMaskCalcFlag_;
  float theMaskNoiseCut_;
  float theMaskDeadCut_;
  float theMaskTruncCut_;
  float theCutToAvoidSignal_;
  int theEventInitNumber_;
  int theEventIterNumber_;
  std::string theCMType_;
  bool useDB_;
};
#endif
