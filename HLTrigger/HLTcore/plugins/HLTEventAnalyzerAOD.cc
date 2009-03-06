/** \class HLTEventAnalyzerAOD
 *
 * See header file for documentation
 *
 *  $Date: 2008/09/06 12:01:52 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerAOD.h"
#include <cassert>

//
// constructors and destructor
//
HLTEventAnalyzerAOD::HLTEventAnalyzerAOD(const edm::ParameterSet& ps) : 
  processName_(ps.getParameter<std::string>("processName")),
  triggerName_(ps.getParameter<std::string>("triggerName")),
  triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
  triggerEventTag_(ps.getParameter<edm::InputTag>("triggerEvent"))
{
  using namespace std;
  using namespace edm;

  cout << "HLTEventAnalyzerAOD configuration: " << endl
       << "   ProcessName = " << processName_ << endl
       << "   TriggerName = " << triggerName_ << endl
       << "   TriggerResultsTag = " << triggerResultsTag_.encode() << endl
       << "   TriggerEventTag = " << triggerEventTag_.encode() << endl;

}

HLTEventAnalyzerAOD::~HLTEventAnalyzerAOD()
{
}

//
// member functions
//
void
HLTEventAnalyzerAOD::beginRun(edm::Run const &, edm::EventSetup const&)
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
	cout << "HLTEventAnalyzerAOD::beginRun:"
	     << " TriggerName " << triggerName_ 
	     << " not available in (new) config!" << endl;
	cout << "Available TriggerNames are: " << endl;
	hltConfig_.dump("Triggers");
      }
    }
  } else {
    cout << "HLTEventAnalyzerAOD::beginRun:"
	 << " config extraction failure with process name "
	 << processName_ << endl;
  }
  return;

}

// ------------ method called to produce the data  ------------
void
HLTEventAnalyzerAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  
  // get event products
  iEvent.getByLabel(triggerResultsTag_,triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()) {
    cout << "HLTEventAnalyzerAOD::analyze: Error in getting TriggerResults product from Event!" << endl;
    return;
  }
  iEvent.getByLabel(triggerEventTag_,triggerEventHandle_);
  if (!triggerEventHandle_.isValid()) {
    cout << "HLTEventAnalyzerAOD::analyze: Error in getting TriggerEvent product from Event!" << endl;
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

void HLTEventAnalyzerAOD::analyzeTrigger(const std::string& triggerName) {
  
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  
  const unsigned int n(hltConfig_.size());
  const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName));
  
  // abort on invalid trigger name
  if (triggerIndex>=n) {
    cout << "HLTEventAnalyzerAOD::analyzeTrigger: path "
	 << triggerName << " - not found!" << endl;
    return;
  }
  
  cout << "HLTEventAnalyzerAOD::analyzeTrigger: path "
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

  // Results from TriggerEvent product - Attention: must look only for
  // modules actually run in this path for this event!
  for (unsigned int j=0; j<=moduleIndex; ++j) {
    const string& moduleLabel(moduleLabels[j]);
    const string  moduleType(hltConfig_.moduleType(moduleLabel));
    // check whether the module is packed up in TriggerEvent product
    const unsigned int filterIndex(triggerEventHandle_->filterIndex(InputTag(moduleLabel,"",processName_)));
    if (filterIndex<triggerEventHandle_->sizeFilters()) {
      cout << " 'L3' filter in slot " << j << " - label/type " << moduleLabel << "/" << moduleType << endl;
      const Vids& VIDS (triggerEventHandle_->filterIds(filterIndex));
      const Keys& KEYS(triggerEventHandle_->filterKeys(filterIndex));
      const size_type nI(VIDS.size());
      const size_type nK(KEYS.size());
      assert(nI==nK);
      const size_type n(max(nI,nK));
      cout << "   " << n  << " accepted 'L3' objects found: " << endl;
      const TriggerObjectCollection& TOC(triggerEventHandle_->getObjects());
      for (size_type i=0; i!=n; ++i) {
	const TriggerObject& TO(TOC[KEYS[i]]);
	cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "
	     << TO.id() << " " << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass()
	     << endl;
      }
    }
  }

   return;
}
