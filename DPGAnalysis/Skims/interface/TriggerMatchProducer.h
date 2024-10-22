#ifndef PhysicsTools_TagAndProbe_TriggerMatchProducer_h
#define PhysicsTools_TagAndProbe_TriggerMatchProducer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

#include <string>
#include <cmath>
#include <TString.h>
#include <TRegexp.h>

// forward declarations
template <class object>
class TriggerMatchProducer : public edm::stream::EDProducer<> {
public:
  explicit TriggerMatchProducer(const edm::ParameterSet&);
  ~TriggerMatchProducer() override;

private:
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data --------------------------

  edm::InputTag _inputProducer;
  edm::EDGetTokenT<edm::View<object> > _inputProducerToken;
  edm::InputTag triggerEventTag_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;
  edm::InputTag triggerResultsTag_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  std::string hltTag_;
  double delRMatchingCut_;
  std::string filterName_;
  bool storeRefCollection_;
  //  bool isFilter_;
  //  bool printIndex_;
  bool changed_;
  HLTPrescaleProvider hltPrescaleProvider_;
};

template <class object>
TriggerMatchProducer<object>::TriggerMatchProducer(const edm::ParameterSet& iConfig)
    : hltPrescaleProvider_(iConfig, consumesCollector(), *this) {
  _inputProducer = iConfig.template getParameter<edm::InputTag>("InputProducer");
  _inputProducerToken = consumes<edm::View<object> >(_inputProducer);

  // **************** Trigger ******************* //
  const edm::InputTag dTriggerEventTag("hltTriggerSummaryAOD", "", "HLT");
  triggerEventTag_ = iConfig.getUntrackedParameter<edm::InputTag>("triggerEventTag", dTriggerEventTag);
  triggerEventToken_ = consumes<trigger::TriggerEvent>(triggerEventTag_);

  const edm::InputTag dTriggerResults("TriggerResults", "", "HLT");
  // By default, trigger results are labeled "TriggerResults" with process name "HLT" in the event.
  triggerResultsTag_ = iConfig.getUntrackedParameter<edm::InputTag>("triggerResultsTag", dTriggerResults);
  triggerResultsToken_ = consumes<edm::TriggerResults>(triggerResultsTag_);

  //  const edm::InputTag dHLTTag("HLT_Ele15_LW_L1R", "","HLT8E29");
  hltTag_ = iConfig.getUntrackedParameter<std::string>("hltTag", "HLT_Ele*");

  delRMatchingCut_ = iConfig.getUntrackedParameter<double>("triggerDelRMatch", 0.30);
  // ******************************************** //
  //Trigger path VS l1 trigger filter. Trigger Path is default.
  //   isFilter_ = iConfig.getUntrackedParameter<bool>("isTriggerFilter",false);
  //   printIndex_ = iConfig.getUntrackedParameter<bool>("verbose",false);

  produces<edm::PtrVector<object> >();
  produces<edm::RefToBaseVector<object> >("R");

  filterName_ = "";
}

template <class object>
TriggerMatchProducer<object>::~TriggerMatchProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <class object>
void TriggerMatchProducer<object>::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

  // Create the output collection
  std::unique_ptr<edm::RefToBaseVector<object> > outColRef(new edm::RefToBaseVector<object>);
  std::unique_ptr<edm::PtrVector<object> > outColPtr(new edm::PtrVector<object>);

  // Get the input collection
  edm::Handle<edm::View<object> > candHandle;
  try {
    event.getByToken(_inputProducerToken, candHandle);
  } catch (cms::Exception& ex) {
    edm::LogError("TriggerMatchProducer") << "Error! Can't get collection: " << _inputProducer;
    throw ex;
  }

  // Trigger Info
  edm::Handle<trigger::TriggerEvent> trgEvent;
  event.getByToken(triggerEventToken_, trgEvent);
  edm::Handle<edm::TriggerResults> pTrgResults;
  event.getByToken(triggerResultsToken_, pTrgResults);

  //gracefully choose the single appropriate HLT path from the list of desired paths
  std::vector<std::string> activeHLTPathsInThisEvent = hltConfig.triggerNames();
  std::map<std::string, bool> triggerInMenu;
  std::map<std::string, bool> triggerUnprescaled;
  //    for (std::vector<edm::InputTag>::const_iterator iMyHLT = hltTags_.begin();
  //         iMyHLT != hltTags_.end(); ++iMyHLT) {
  //      if ( triggerInMenu.find((*iMyHLT).label()) == triggerInMenu.end() )
  //        triggerInMenu[(*iMyHLT).label()] = false;
  //      if ( triggerUnprescaled.find((*iMyHLT).label()) == triggerUnprescaled.end() )
  //        triggerUnprescaled[(*iMyHLT).label()] = false;
  //    }
  for (std::vector<std::string>::const_iterator iHLT = activeHLTPathsInThisEvent.begin();
       iHLT != activeHLTPathsInThisEvent.end();
       ++iHLT) {
    //        cout << "######## " << *iHLT << endl;

    if (TString(*iHLT).Contains(TRegexp(hltTag_))) {
      triggerInMenu[*iHLT] = true;
      if (hltPrescaleProvider_.prescaleValue<double>(event, eventSetup, *iHLT) == 1)
        triggerUnprescaled[*iHLT] = true;
    }
  }

  // Some sanity checks
  if (not trgEvent.isValid()) {
    edm::LogInfo("info") << "******** Following Trigger Summary Object Not Found: " << triggerEventTag_;

    event.put(std::move(outColRef), "R");
    event.put(std::move(outColPtr));
    return;
  }

  //---------------------------------------------------------------------------

  edm::InputTag filterTag;
  // loop over these objects to see whether they match
  const trigger::TriggerObjectCollection& TOC(trgEvent->getObjects());

  std::vector<int> index;
  std::vector<std::string> filters;
  //    if(isFilter_){
  //-----------------------------------------------------------------------
  for (std::map<std::string, bool>::const_iterator iMyHLT = triggerInMenu.begin(); iMyHLT != triggerInMenu.end();
       ++iMyHLT) {
    if (!(iMyHLT->second && triggerUnprescaled[iMyHLT->first]))
      continue;

    int triggerIndex = -1;
    edm::InputTag filterTag;
    try {
      filters = hltConfig.moduleLabels(iMyHLT->first);
      triggerIndex = hltConfig.triggerIndex(iMyHLT->first);
    } catch (std::exception const&) {
      cout << "bad trigger\n";
    }
    // Results from TriggerResults product
    if (triggerIndex == -1 || !(pTrgResults->wasrun(triggerIndex)) || !(pTrgResults->accept(triggerIndex)) ||
        (pTrgResults->error(triggerIndex))) {
      continue;
    }

    for (std::vector<std::string>::iterator filter = filters.begin(); filter != filters.end(); ++filter) {
      edm::InputTag testTag(*filter, "", triggerEventTag_.process());
      int testindex = trgEvent->filterIndex(testTag);
      if (!(testindex >= trgEvent->sizeFilters())) {
        filterName_ = *filter;
        filterTag = testTag;
      }
    }

    //using last filter tag
    index.push_back(trgEvent->filterIndex(filterTag));
    //        std::cout << "TrgPath" << iMyHLT->first << "hltTag_.label() " <<
    // 	 filterTag.label() << "   filter name " <<
    // 	 filterName_ << "  sizeFilters " <<
    // 	 trgEvent->sizeFilters() << std::endl;
  }

  // Loop over the candidate collection
  edm::PtrVector<object> ptrVect;
  edm::RefToBaseVector<object> refs;
  for (size_t i = 0; i < candHandle->size(); ++i) {
    ptrVect.push_back(candHandle->ptrAt(i));
    refs.push_back(candHandle->refAt(i));
  }
  // find how many objects there are
  unsigned int counter = 0;
  for (typename edm::View<object>::const_iterator j = candHandle->begin(); j != candHandle->end(); ++j, ++counter) {
    bool hltTrigger = false;
    for (unsigned int idx = 0; idx < index.size(); ++idx) {
      if (hltTrigger)
        continue;
      const trigger::Keys& KEYS(trgEvent->filterKeys(index[idx]));
      const size_type nK(KEYS.size());
      // Get cut decision for each candidate
      // Did this candidate cause a HLT trigger?

      for (int ipart = 0; ipart != nK; ++ipart) {
        const trigger::TriggerObject& TO = TOC[KEYS[ipart]];
        double dRval = deltaR(j->eta(), j->phi(), TO.eta(), TO.phi());
        hltTrigger = dRval < delRMatchingCut_;
        if (hltTrigger)
          break;
      }
    }

    if (hltTrigger) {
      outColRef->push_back(refs[counter]);
      outColPtr->push_back(ptrVect[counter]);
    }
  }
  event.put(std::move(outColRef), "R");
  event.put(std::move(outColPtr));
}

// ---- method called once each job just before starting event loop  ---
template <class object>
void TriggerMatchProducer<object>::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  //   std::cout << "calling init(" << "iRun" << ", " << "iSetup" << ", " << triggerEventTag_.process() << ", " << "changed_" << ") in beginRun()" << std::endl;
  if (!hltPrescaleProvider_.init(iRun, iSetup, triggerEventTag_.process(), changed_)) {
    edm::LogError("TriggerMatchProducer") << "Error! Can't initialize HLTConfigProvider";
    throw cms::Exception("HLTConfigProvider::init() returned non 0");
  }
  // HLTConfigProvider const&  hltConfig = hltPrescaleProvider_.hltConfigProvider();
  //   if(printIndex_ && changed_)
  //     std::cout << "HLT configuration changed !" << std::endl;
  //  std::vector<std::string> filters = hltConfig.moduleLabels( hltTag_.label() );
}
#endif
