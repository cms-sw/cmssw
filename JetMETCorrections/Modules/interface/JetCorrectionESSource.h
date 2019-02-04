#ifndef JetCorrectionESSource_h
#define JetCorrectionESSource_h

//
// Original Author:  Fedor Ratnikov
// Created:  Dec. 28, 2006 (originally JetCorrectionService, renamed in 2011)
//

#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/SourceFactory.h"

class JetCorrector;
class JetCorrectionsRecord;

namespace edm {
  namespace eventsetup {
    class EventSetupRecordKey;
  }
}

#define DEFINE_JET_CORRECTION_ESSOURCE(corrector_, name_ ) \
typedef JetCorrectionESSource <corrector_>  name_; \
DEFINE_FWK_EVENTSETUP_SOURCE(name_)

template <class Corrector>
class JetCorrectionESSource : public edm::ESProducer,
			      public edm::EventSetupRecordIntervalFinder
{
private:
  edm::ParameterSet mParameterSet;
  std::string mLevel;
  std::string mEra;
  std::string mAlgo;
  std::string mSection;
  bool mDebug;

public:
  JetCorrectionESSource(edm::ParameterSet const& fConfig) : mParameterSet(fConfig) 
  {
    std::string label = fConfig.getParameter<std::string>("@module_label");
    mLevel            = fConfig.getParameter<std::string>("level");
    mEra              = fConfig.getParameter<std::string>("era");
    mAlgo             = fConfig.getParameter<std::string>("algorithm");
    mSection          = fConfig.getParameter<std::string>("section");
    mDebug            = fConfig.getUntrackedParameter<bool>("debug",false);

    setWhatProduced(this, label);
    findingRecord<JetCorrectionsRecord>();
  }

  ~JetCorrectionESSource() override {}

  std::unique_ptr<JetCorrector> produce(JetCorrectionsRecord const& iRecord) 
  {
    std::string fileName("CondFormats/JetMETObjects/data/");
    if (!mEra.empty())
      fileName += mEra;
    if (!mLevel.empty())
      fileName += "_"+mLevel;
    if (!mAlgo.empty())
      fileName += "_"+mAlgo;
    fileName += ".txt";
    if (mDebug)
      std::cout << "Parameter File: " << fileName << std::endl;
    edm::FileInPath fip(fileName);
    JetCorrectorParameters *tmpJetCorPar = new JetCorrectorParameters(fip.fullPath(), mSection);
    return std::make_unique<Corrector>(*tmpJetCorPar, mParameterSet);
  }

  void setIntervalFor(edm::eventsetup::EventSetupRecordKey const&, edm::IOVSyncValue const&, edm::ValidityInterval& fIOV) override
  {
    fIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); // anytime
  }
};
#endif
