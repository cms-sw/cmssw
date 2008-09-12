/** \class HLTEventAnalyzerRAW
 *
 * See header file for documentation
 *
 *  $Date: 2008/09/11 13:19:18 $
 *  $Revision: 1.3 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerRAW.h"
#include <cassert>

//
// constructors and destructor
//
HLTEventAnalyzerRAW::HLTEventAnalyzerRAW(const edm::ParameterSet& ps) : 
  processName_(ps.getParameter<std::string>("processName")),
  triggerName_(ps.getParameter<std::string>("triggerName")),
  triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
  triggerEventWithRefsTag_(ps.getParameter<edm::InputTag>("triggerEventWithRefs"))
{
  using namespace std;
  using namespace edm;

  cout << "HLTEventAnalyzerRAW configuration: " << endl
       << "   ProcessName = " << processName_ << endl
       << "   TriggerName = " << triggerName_ << endl
       << "   TriggerResultsTag = " << triggerResultsTag_.encode() << endl
       << "   TriggerEventWithRefsTag = " << triggerEventWithRefsTag_.encode() << endl;

}

HLTEventAnalyzerRAW::~HLTEventAnalyzerRAW()
{
}

//
// member functions
//
void
HLTEventAnalyzerRAW::beginRun(edm::Run const &, edm::EventSetup const&)
{
  using namespace std;
  using namespace edm;
  
  // HLT config does not change within runs!
  if (hltConfig_.init(processName_)) {
    // check if trigger name in (new) config
    if (triggerName_!="@") { // "@" means: analyze all triggers in config
      const unsigned int n(hltConfig_.size());
      const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName_));
      if (triggerIndex>=n) {
	cout << "HLTEventAnalyzerRAW::beginRun:"
	     << " TriggerName " << triggerName_ 
	     << " not available in (new) config!" << endl;
	cout << "Available TriggerNames are: " << endl;
	hltConfig_.dump("Triggers");
      }
    }
  } else {
    cout << "HLTEventAnalyzerRAW::beginRun:"
	 << " config extraction failure with process name "
	 << processName_ << endl;
  }
  return;

}

// ------------ method called to produce the data  ------------
void
HLTEventAnalyzerRAW::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  
  // get event products
  iEvent.getByLabel(triggerResultsTag_,triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()) {
    cout << "HLTEventAnalyzerRAW::analyze: Error in getting TriggerResults product from Event!" << endl;
    return;
  }
  iEvent.getByLabel(triggerEventWithRefsTag_,triggerEventWithRefsHandle_);
  if (!triggerEventWithRefsHandle_.isValid()) {
    cout << "HLTEventAnalyzerRAW::analyze: Error in getting TriggerEventWithRefs product from Event!" << endl;
    return;
  }
  // sanity check
  assert(triggerResultsHandle_->size()==hltConfig_.size());
  
  // analyze this event for the triggers requested
  if (triggerName_=="@") {
    const unsigned int n(hltConfig_.size());
    for (unsigned int i=0; i!=n; ++i) {
      analyzeTrigger(hltConfig_.triggerName(i));
    }
  } else {
    analyzeTrigger(triggerName_);
  }
  
  return;
  
}

void HLTEventAnalyzerRAW::analyzeTrigger(const std::string& triggerName) {
  
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  
  const unsigned int n(hltConfig_.size());
  const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName));
  
  // abort on invalid trigger name
  if (triggerIndex>=n) {
    cout << "HLTEventAnalyzerRAW::analyzeTrigger: path "
	 << triggerName << " - not found!" << endl;
    return;
  }
  
  cout << "HLTEventAnalyzerRAW::analyzeTrigger: path "
       << triggerName << " [" << triggerIndex << "]" << endl;
  // modules on this trigger path
  const unsigned int m(hltConfig_.size(triggerIndex));
  const vector<string> moduleLabels(hltConfig_.moduleLabels(triggerIndex));

  // Results from TriggerResults product
  cout << " Trigger path status:"
       << " WasRun=" << triggerResultsHandle_->wasrun(triggerIndex)
       << " Accept=" << triggerResultsHandle_->accept(triggerIndex)
       << " Error =" << triggerResultsHandle_->error(triggerIndex)
       << endl;
  const unsigned int moduleIndex(triggerResultsHandle_->index(triggerIndex));
  cout << " Last active module - label/type: "
       << moduleLabels[moduleIndex] << "/" << hltConfig_.moduleType(moduleLabels[moduleIndex])
       << " [" << moduleIndex << " out of 0-" << (m-1) << " on this path]"
       << endl;
  assert (moduleIndex<m);

  // Results from TriggerEventWithRefs product - Attention: must look only for
  // modules actually run in this path for this event!
  for (unsigned int j=0; j<=moduleIndex; ++j) {
    const string& moduleLabel(moduleLabels[j]);
    const string  moduleType(hltConfig_.moduleType(moduleLabel));
    // check whether the module is packed up in TriggerEventWithRef product
    const unsigned int filterIndex(triggerEventWithRefsHandle_->filterIndex(InputTag(moduleLabel,"",processName_)));
    if (filterIndex<triggerEventWithRefsHandle_->size()) {
      cout << " Filter in slot " << j << " - label/type " << moduleLabel << "/" << moduleType << endl;
      cout << "   Accepted objects:";

      VRphoton photons;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,photons);
      const unsigned int nPhotons(photons.size());
      if (nPhotons>0) {
	cout << "  Photons: " << nPhotons;
      }

      VRelectron electrons;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,electrons);
      const unsigned int nElectrons(electrons.size());
      if (nElectrons>0) {
	cout << "  Electrons: " << nElectrons;
      }

      VRmuon muons;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,muons);
      const unsigned int nMuons(muons.size());
      if (nMuons>0) {
	cout << "  Muons: " << nMuons;
      }

      VRjet jets;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,jets);
      const unsigned int nJets(jets.size());
      if (nJets>0) {
	cout << "  Jets: " << nJets;
      }

      VRcomposite composites;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,composites);
      const unsigned int nComposites(composites.size());
      if (nComposites>0) {
	cout << "  Composites: " << nComposites;
      }

      VRmet mets;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,mets);
      const unsigned int nMETs(mets.size());
      if (nMETs>0) {
	cout << "  METs: " << nMETs;
      }

      VRht hts;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,hts);
      const unsigned int nHTs(hts.size());
      if (nHTs>0) {
	cout << "  HTs: " << nHTs;
      }

      VRpixtrack pixtracks;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,pixtracks);
      const unsigned int nPixTracks(pixtracks.size());
      if (nPixTracks>0) {
	cout << "  PixTracks: " << nPixTracks;
      }

      VRl1em l1em;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,l1em);
      const unsigned int nL1EM(l1em.size());
      if (nL1EM>0) {
	cout << "  L1EM: " << nL1EM;
      }

      VRl1muon l1muon;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,l1muon);
      const unsigned int nL1Muon(l1muon.size());
      if (nL1Muon>0) {
	cout << "  L1Muon: " << nL1Muon;
      }

      VRl1jet l1jet;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,l1jet);
      const unsigned int nL1Jet(l1jet.size());
      if (nL1Jet>0) {
	cout << "  L1Jet: " << nL1Jet;
      }

      VRl1etmiss l1etmiss;
      triggerEventWithRefsHandle_->getObjects(filterIndex,0,l1etmiss);
      const unsigned int nL1EtMiss(l1etmiss.size());
      if (nL1EtMiss>0) {
	cout << "  L1EtMiss: " << nL1EtMiss;
      }

      cout << endl;

    }
  }

   return;
}
