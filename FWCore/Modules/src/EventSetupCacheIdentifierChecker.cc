// -*- C++ -*-
//
// Package:    EventSetupCacheIdentifierChecker
// Class:      EventSetupCacheIdentifierChecker
// 
/**\class EventSetupCacheIdentifierChecker EventSetupCacheIdentifierChecker.cc FWCore/Modules/src/EventSetupCacheIdentifierChecker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Chris Jones
//         Created:  Wed May 30 14:42:16 CDT 2012
//
//


// system include files
#include <memory>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// class declaration
//

namespace edm {
  class EventSetupCacheIdentifierChecker : public edm::EDAnalyzer {
   public:
    explicit EventSetupCacheIdentifierChecker(const edm::ParameterSet&);
    ~EventSetupCacheIdentifierChecker();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
    //virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    //virtual void endJob() ;

    virtual void beginRun(edm::Run const&, edm::EventSetup const&);
    //virtual void endRun(edm::Run const&, edm::EventSetup const&);
    virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

    void check(edm::EventSetup const&);
    void initialize();
      // ----------member data ---------------------------
    ParameterSet m_pset;
    std::map<eventsetup::EventSetupRecordKey,std::vector<unsigned int> > m_recordKeysToExpectedCacheIdentifiers;
    unsigned int m_index;
  };
}
//
// constants, enums and typedefs
//
using namespace edm;

//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupCacheIdentifierChecker::EventSetupCacheIdentifierChecker(const edm::ParameterSet& iConfig):
m_pset(iConfig),
m_index(0)
{
   //now do what ever initialization is needed

}


EventSetupCacheIdentifierChecker::~EventSetupCacheIdentifierChecker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
EventSetupCacheIdentifierChecker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  check(iSetup);
}


// ------------ method called once each job just before starting event loop  ------------
//void 
//EventSetupCacheIdentifierChecker::beginJob()
//{
//}

// ------------ method called once each job just after ending the event loop  ------------
//void 
//EventSetupCacheIdentifierChecker::endJob() 
//{
//}

// ------------ method called when starting to processes a run  ------------
void 
EventSetupCacheIdentifierChecker::beginRun(edm::Run const&, edm::EventSetup const& iSetup)
{
  check(iSetup);
}

// ------------ method called when ending the processing of a run  ------------
//void 
//EventSetupCacheIdentifierChecker::endRun(edm::Run const&, edm::EventSetup const&)
//{
//}

// ------------ method called when starting to processes a luminosity block  ------------
void 
EventSetupCacheIdentifierChecker::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& iSetup)
{
  check(iSetup);
}

// ------------ method called when ending the processing of a luminosity block  ------------
//void 
//EventSetupCacheIdentifierChecker::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
//{
//}

void
EventSetupCacheIdentifierChecker::check(edm::EventSetup const& iSetup)
{
  if(0==m_recordKeysToExpectedCacheIdentifiers.size()) {
    initialize();
  }
  using namespace edm::eventsetup;

  
  for(auto it = m_recordKeysToExpectedCacheIdentifiers.begin(), itEnd = m_recordKeysToExpectedCacheIdentifiers.end();
      it != itEnd;
      ++it) {
    EventSetupRecord const* pRecord = iSetup.find(it->first);
    if(0 == pRecord) {
      edm::LogWarning("RecordNotInIOV") <<"The EventSetup Record '"<<it->first.name()<<"' is not available for this IOV.";
    }
    if(it->second.size() <= m_index) {
      throw cms::Exception("TooFewCacheIDs")<<"The vector of cacheIdentifiers for the record "<<it->first.name()<<" is too short";
    }
    if(0 != pRecord && pRecord->cacheIdentifier() != it->second[m_index]) {
      throw cms::Exception("IncorrectCacheID")<<"The Record "<<it->first.name()<<" was supposed to have cacheIdentifier: "<<it->second[m_index]<<" but instead has "<<pRecord->cacheIdentifier();
    }
  }
  ++m_index;
}

void
EventSetupCacheIdentifierChecker::initialize()
{
  std::vector<std::string> recordNames{m_pset.getParameterNamesForType<std::vector<unsigned int> >(false)};

  for(auto const& name: recordNames) {
    eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType(name));
    if(recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogWarning("DataGetter") <<"Record \""<< name <<"\" does not exist "<<std::endl;
      
      continue;
    }
    
    m_recordKeysToExpectedCacheIdentifiers.insert(std::make_pair(recordKey, m_pset.getUntrackedParameter<std::vector<unsigned int> >(name)));
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EventSetupCacheIdentifierChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addWildcardUntracked<std::vector<unsigned int> >("*")->setComment("The label is the name of an EventSetup Record while the vector contains the expected cacheIdentifier values for each beginRun, beginLuminosityBlock and event transition");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EventSetupCacheIdentifierChecker);
