#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TString.h"
#include "TRegexp.h"

#include <cassert>

#include "CommonTools/TriggerUtils/interface/CandidateTriggerObjectProducer.h"

//
// constructors and destructor
//
CandidateTriggerObjectProducer::CandidateTriggerObjectProducer(const edm::ParameterSet& ps)
    : triggerResultsToken_(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("triggerResults"))),
      triggerEventToken_(consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("triggerEvent"))),
      processName_(ps.getParameter<edm::InputTag>("triggerEvent").process()),
      triggerName_(ps.getParameter<std::string>("triggerName")),
      hltPrescaleProvider_(ps, consumesCollector(), *this) {
  //   cout << "Trigger Object Producer:" << endl
  //        << "   TriggerResultsTag = " << triggerResultsTag_.encode() << endl
  //        << "   TriggerEventTag = " << triggerEventTag_.encode() << endl;

  produces<reco::CandidateCollection>();
}

CandidateTriggerObjectProducer::~CandidateTriggerObjectProducer() {}

//
// member functions
//
void CandidateTriggerObjectProducer::beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup) {
  using namespace edm;

  bool changed(false);
  if (!hltPrescaleProvider_.init(iRun, iSetup, processName_, changed)) {
    edm::LogError("CandidateTriggerObjectProducer") << "Error! Can't initialize HLTConfigProvider";
    throw cms::Exception("HLTConfigProvider::init() returned non 0");
  }

  return;
}

// ------------ method called to produce the data  ------------
void CandidateTriggerObjectProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  std::unique_ptr<reco::CandidateCollection> coll(new reco::CandidateCollection);

  // get event products
  edm::Handle<edm::TriggerResults> triggerResultsHandle;
  iEvent.getByToken(triggerResultsToken_, triggerResultsHandle);
  if (!triggerResultsHandle.isValid()) {
    edm::LogError("CandidateTriggerObjectProducer")
        << "CandidateTriggerObjectProducer::analyze: Error in getting TriggerResults product from Event!";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(triggerEventToken_, triggerEventHandle);
  if (!triggerEventHandle.isValid()) {
    edm::LogError("CandidateTriggerObjectProducer")
        << "CandidateTriggerObjectProducer::analyze: Error in getting TriggerEvent product from Event!";
    return;
  }

  HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

  // sanity check
  //   std::cout << hltConfig.size() << std::endl;
  //   std::cout << triggerResultsHandle->size() << std::endl;
  assert(triggerResultsHandle->size() == hltConfig.size());

  const unsigned int n(hltConfig.size());
  std::vector<std::string> activeHLTPathsInThisEvent = hltConfig.triggerNames();
  std::map<std::string, bool> triggerInMenu;
  std::map<std::string, bool> triggerUnprescaled;

  for (std::vector<std::string>::const_iterator iHLT = activeHLTPathsInThisEvent.begin();
       iHLT != activeHLTPathsInThisEvent.end();
       ++iHLT) {
    //matching with regexp filter name. More than 1 matching filter is allowed
    if (TString(*iHLT).Contains(TRegexp(TString(triggerName_)))) {
      triggerInMenu[*iHLT] = true;
      auto const prescales = hltPrescaleProvider_.prescaleValues<double>(iEvent, iSetup, *iHLT);
      if (prescales.first == 1 and prescales.second == 1)
        triggerUnprescaled[*iHLT] = true;
    }
  }

  for (std::map<std::string, bool>::const_iterator iMyHLT = triggerInMenu.begin(); iMyHLT != triggerInMenu.end();
       ++iMyHLT) {
    //using only unprescaled triggers
    if (!(iMyHLT->second && triggerUnprescaled[iMyHLT->first]))
      continue;
    const unsigned int triggerIndex(hltConfig.triggerIndex(iMyHLT->first));

    assert(triggerIndex == iEvent.triggerNames(*triggerResultsHandle).triggerIndex(iMyHLT->first));

    // abort on invalid trigger name
    if (triggerIndex >= n) {
      edm::LogError("CandidateTriggerObjectProducer")
          << "CandidateTriggerObjectProducer::analyzeTrigger: path " << triggerName_ << " - not found!";
      return;
    }

    // modules on this trigger path
    //      const unsigned int m(hltConfig.size(triggerIndex));
    const std::vector<std::string>& moduleLabels(hltConfig.saveTagsModules(triggerIndex));

    // Results from TriggerResults product
    if (!(triggerResultsHandle->wasrun(triggerIndex)) || !(triggerResultsHandle->accept(triggerIndex)) ||
        (triggerResultsHandle->error(triggerIndex))) {
      continue;
    }

    //       const unsigned int moduleIndex(triggerResultsHandle->index(triggerIndex));

    //       assert (moduleIndex<m);

    // Results from TriggerEvent product - Looking only on last filter since trigger is accepted
    for (unsigned int imodule = 0; imodule < moduleLabels.size(); ++imodule) {
      const std::string& moduleLabel(moduleLabels[imodule]);
      const std::string moduleType(hltConfig.moduleType(moduleLabel));
      //Avoiding L1 seeds
      if (moduleType.find("Level1GTSeed") != std::string::npos)
        continue;
      // check whether the module is packed up in TriggerEvent product
      const unsigned int filterIndex(triggerEventHandle->filterIndex(InputTag(moduleLabel, "", processName_)));
      if (filterIndex < triggerEventHandle->sizeFilters()) {
        //	    std::cout << " 'L3' filter in slot " << imodule
        //	              << " - label/type " << moduleLabel << "/" << moduleType << std::endl;
        const Vids& VIDS(triggerEventHandle->filterIds(filterIndex));
        const Keys& KEYS(triggerEventHandle->filterKeys(filterIndex));
        const size_type nI(VIDS.size());
        const size_type nK(KEYS.size());
        assert(nI == nK);
        const size_type n(std::max(nI, nK));
        //	    std::cout << "   " << n  << " accepted 'L3' objects found: " << std::endl;
        const TriggerObjectCollection& TOC(triggerEventHandle->getObjects());
        for (size_type i = 0; i != n; ++i) {
          const TriggerObject& TO(TOC[KEYS[i]]);
          coll->push_back(reco::LeafCandidate(0, TO.particle().p4(), reco::Particle::Point(0., 0., 0.), TO.id()));
          // 	      std::cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "
          //                      << TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass()
          //                      << std::endl;
        }
      }
    }
  }

  iEvent.put(std::move(coll));
  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CandidateTriggerObjectProducer);
