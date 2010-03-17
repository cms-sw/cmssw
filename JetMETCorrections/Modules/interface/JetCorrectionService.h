#ifndef JetCorrectionService_h
#define JetCorrectionService_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: JetCorrectionService.h,v 1.6 2010/03/15 20:24:23 kkousour Exp $
//
//

// system include files
#include <memory>
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

// macro to define instance of JetCorrectionService
#define DEFINE_JET_CORRECTION_SERVICE(corrector_, name_ ) \
typedef JetCorrectionService <corrector_>  name_; \
DEFINE_FWK_EVENTSETUP_SOURCE(name_)


// Correction Service itself
template <class Corrector>
class JetCorrectionService : public edm::ESProducer,
			     public edm::EventSetupRecordIntervalFinder
{
  private:
    edm::ParameterSet mParameterSet;
    std::string mLevel;
    std::string mAlgo;
    std::string mSection;
    std::string mPayloadName;
    bool mDebug;

  public:
    //------------- construction ---------------------------------------
    JetCorrectionService(const edm::ParameterSet& fConfig) : mParameterSet(fConfig) 
      {
        std::string label = fConfig.getParameter<std::string>("@module_label"); 
        mLevel            = fConfig.getParameter<std::string>("level");
        mAlgo             = fConfig.getParameter<std::string>("algorithm");
        mSection          = fConfig.getParameter<std::string>("section");
        mDebug            = fConfig.getUntrackedParameter<bool>("debug",false);
        
        mPayloadName = mLevel;
        if (!mAlgo.empty())
          mPayloadName += "_"+mAlgo;
        if (!mSection.empty())
          mPayloadName += "_"+mSection; 
        if (mDebug)
          std::cout<<"Payload: "<<mPayloadName<<std::endl;
        setWhatProduced(this,label);
        findingRecord<JetCorrectionsRecord>();      
      }
    //------------- destruction ----------------------------------------
    ~JetCorrectionService () {}
    //------------- member functions -----------------------------------
    boost::shared_ptr<JetCorrector> produce(const JetCorrectionsRecord& iRecord) 
      {
        edm::ESHandle<JetCorrectorParameters> JetCorPar;
        iRecord.get(mPayloadName,JetCorPar); 
        boost::shared_ptr<JetCorrector> mCorrector(new Corrector(*JetCorPar,mParameterSet));
        return mCorrector;
      }
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,const edm::IOVSyncValue&,edm::ValidityInterval& fIOV)
      {
        fIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime()); // anytime
      }
};

#endif
