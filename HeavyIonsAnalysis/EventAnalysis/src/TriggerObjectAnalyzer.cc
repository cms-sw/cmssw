// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TNtuple.h"

using namespace std;
using namespace edm;

//
// class declaration
//

class TriggerObjectAnalyzer : public edm::EDAnalyzer {
public:
  explicit TriggerObjectAnalyzer(const edm::ParameterSet&);
  ~TriggerObjectAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  // ----------member data ---------------------------

  std::string   processName_;
  std::vector<std::string>   triggerNames_;
  edm::InputTag triggerResultsTag_;
  edm::InputTag triggerEventTag_;

  edm::Handle<edm::TriggerResults>   triggerResultsHandle_;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle_;

  HLTConfigProvider hltConfig_;

  unsigned int triggerIndex_;
  unsigned int moduleIndex_;
  string moduleLabel_;
  vector<string> moduleLabels_;

  edm::Service<TFileService> fs;
  vector<TNtuple*> nt_;
  int verbose_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TriggerObjectAnalyzer::TriggerObjectAnalyzer(const edm::ParameterSet& ps):
  processName_(ps.getParameter<std::string>("processName")),
  triggerNames_(ps.getParameter<std::vector<std::string> >("triggerNames")),
  triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
  triggerEventTag_(ps.getParameter<edm::InputTag>("triggerEvent"))
{
  //now do what ever initialization is needed
  nt_.reserve(triggerNames_.size());
  nt_[0] = fs->make<TNtuple>("jetObjTree","HLT_HIJet*_Triggers","id:pt:eta:phi:mass");
  verbose_ = 0;
}


TriggerObjectAnalyzer::~TriggerObjectAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
TriggerObjectAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if(hltConfig_.size() > 0){

    float id = -99,pt=-99,eta=-99,phi=-99,mass=-99;

    using namespace edm;
    iEvent.getByLabel(triggerEventTag_,triggerEventHandle_);
    iEvent.getByLabel(triggerResultsTag_,triggerResultsHandle_);

    for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){

      triggerIndex_ = hltConfig_.triggerIndex(triggerNames_[itrig]);

      const unsigned int mIndex = triggerResultsHandle_->index(triggerIndex_);

      // Results from TriggerEvent product - Attention: must look only for
      // modules actually run in this path for this event!
      for (unsigned int j=0; j<=mIndex; ++j) {
	// check whether the module is packed up in TriggerEvent product
	string trigFilterIndex = hltConfig_.moduleLabels(triggerIndex_).at(j); //this is simple to put into a loop to get all triggers...
	const unsigned int filterIndex(triggerEventHandle_->filterIndex(InputTag(trigFilterIndex,"",processName_)));
	if (filterIndex<triggerEventHandle_->sizeFilters()) {
	  const trigger::Vids& VIDS (triggerEventHandle_->filterIds(filterIndex));
	  const trigger::Keys& KEYS(triggerEventHandle_->filterKeys(filterIndex));
	  const unsigned int nI(VIDS.size());
	  const unsigned int nK(KEYS.size());
	  assert(nI==nK);
	  const unsigned int n(max(nI,nK));

	  const trigger::TriggerObjectCollection& TOC(triggerEventHandle_->getObjects());
	  for (unsigned int i=0; i!=n; ++i) {
	    const trigger::TriggerObject& TO(TOC[KEYS[i]]);
	    //This check prevents grabbing the L1 trigger object (VIDS < 0), and finds the max trigger pt within all trigger collections
	    if(VIDS[i]>0 && pt<TO.pt()){
	      //cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "
	      //              << TO.id() << " " << TO.pt() << " " << TO.et() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass()
	      //                        << endl;
	      id = TO.id();
	      pt = TO.pt();
	      eta = TO.eta();
	      phi = TO.phi();
	      mass = TO.mass();
	    }
	  }
	}
      }
    }

    nt_[0]->Fill(id,pt,eta,phi,mass);
  }
}


// ------------ method called once each job just before starting event loop  ------------
void
TriggerObjectAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
TriggerObjectAnalyzer::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
void
TriggerObjectAnalyzer::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  bool changed(true);
  if (hltConfig_.init(iRun,iSetup,processName_,changed)) {
    if (changed) {
      const unsigned int n(hltConfig_.size());
      for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){
	if (triggerNames_[itrig]!="@") { // "@" means: analyze all triggers in config

	  //This functionality is currently not working for me (kjung - mar 24, 2014)... no trigger passes the condition
	  if (triggerIndex_>=n) {
	    //cout << "HLTEventAnalyzerAOD::analyze:"
	    //    << " TriggerName " << triggerNames_[0]
	    //    << " not available in (new) config!" << endl;
	    //cout << "Available TriggerNames are: " << endl;
	    //hltConfig_.dump("Triggers");
	  }
	}
      }
      if(verbose_){
	hltConfig_.dump("ProcessName");
	hltConfig_.dump("GlobalTag");
	hltConfig_.dump("TableName");
	hltConfig_.dump("Streams");
	hltConfig_.dump("Datasets");
	hltConfig_.dump("PrescaleTable");
	hltConfig_.dump("ProcessPSet");
      }
    }
  } else {
    cout << "HLTEventAnalyzerAOD::analyze:"
	 << " config extraction failure with process name "
	 << processName_ << endl;
  }

}

// ------------ method called when ending the processing of a run  ------------
void
TriggerObjectAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
TriggerObjectAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
TriggerObjectAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TriggerObjectAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TriggerObjectAnalyzer);
