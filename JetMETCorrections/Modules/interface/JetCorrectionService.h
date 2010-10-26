#ifndef JetCorrectionService_h
#define JetCorrectionService_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: JetCorrectionService.h,v 1.9 2010/10/12 13:50:38 srappocc Exp $
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
    //boost::shared_ptr<JetCorrector> mCorrector;
    edm::ParameterSet mParameterSet;
    std::string mLevel;
    std::string mEra;
    std::string mAlgo;
    std::string mSection;
    std::string mPayloadName;
    bool mUseCondDB;
    bool mDebug;

  public:
    //------------- construction ---------------------------------------
    JetCorrectionService(const edm::ParameterSet& fConfig) : mParameterSet(fConfig) 
      {
        std::string label = fConfig.getParameter<std::string>("@module_label"); 
        mLevel            = fConfig.getParameter<std::string>("level");
        mEra              = fConfig.getParameter<std::string>("era");
        mAlgo             = fConfig.getParameter<std::string>("algorithm");
        mSection          = fConfig.getParameter<std::string>("section");
        mUseCondDB        = fConfig.getUntrackedParameter<bool>("useCondDB",false);
        mDebug            = fConfig.getUntrackedParameter<bool>("debug",false);
	mPayloadName = mAlgo;
        
        setWhatProduced(this, label);
        findingRecord <JetCorrectionsRecord> ();

      }
    //------------- destruction ----------------------------------------
    ~JetCorrectionService () {}
    //------------- member functions -----------------------------------
    boost::shared_ptr<JetCorrector> produce(const JetCorrectionsRecord& iRecord) 
      {
        if (mUseCondDB)
          {
            edm::ESHandle<JetCorrectorParametersCollection> JetCorParColl;
            iRecord.get(mPayloadName,JetCorParColl); 
	    JetCorrectorParameters const & JetCorPar = (*JetCorParColl)[ mLevel ];
            boost::shared_ptr<JetCorrector> mCorrector(new Corrector(JetCorPar,mParameterSet));
            return mCorrector;
          }
        else
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
              std::cout<<"Parameter File: "<<fileName<<std::endl;
            edm::FileInPath fip(fileName);
            JetCorrectorParameters *tmpJetCorPar = new JetCorrectorParameters(fip.fullPath(),mSection);
            boost::shared_ptr<JetCorrector> mCorrector(new Corrector(*tmpJetCorPar,mParameterSet));
            return mCorrector;
          }
      }
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,const edm::IOVSyncValue&,edm::ValidityInterval& fIOV)
      {
        fIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime()); // anytime
      }
};

#endif
