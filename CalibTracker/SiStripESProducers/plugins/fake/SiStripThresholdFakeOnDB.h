#ifndef CalibTracker_SiStripESProducers_SiStripThresholdFakeOnDB_h
#define CalibTracker_SiStripESProducers_SiStripThresholdFakeOnDB_h

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include <vector>


class SiStripThresholdFakeOnDB : public ConditionDBWriter<SiStripThreshold> {

public:

  explicit SiStripThresholdFakeOnDB(const edm::ParameterSet&);
  ~SiStripThresholdFakeOnDB();

private:

  void algoAnalyze(const edm::Event &, const edm::EventSetup &);

  SiStripThreshold * getNewObject();

private:
  
  SiStripThreshold* threshold_;


};
#endif
