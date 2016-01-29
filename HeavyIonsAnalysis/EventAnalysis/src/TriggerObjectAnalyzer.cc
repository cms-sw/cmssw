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
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"

#include "TNtuple.h"
#include "TRegexp.h"

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
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  edm::Handle<edm::TriggerResults>   triggerResultsHandle_;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle_;

  HLTConfigProvider hltConfig_;

  unsigned int triggerIndex_;
  unsigned int moduleIndex_;
  string moduleLabel_;
  vector<string> moduleLabels_;

  edm::Service<TFileService> fs;
  vector<TTree*> nt_;
  int verbose_;

  std::map<std::string, bool> triggerInMenu;

  vector<double> id[500];
  vector<double> pt[500];
  vector<double> eta[500];
  vector<double> phi[500];
  vector<double> mass[500];
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
  triggerEventTag_(ps.getParameter<edm::InputTag>("triggerEvent")),
  triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
  triggerEventToken_(consumes<trigger::TriggerEvent>(triggerEventTag_))
{
  //now do what ever initialization is needed
  nt_.reserve(triggerNames_.size());
  for(unsigned int isize=0; isize<triggerNames_.size(); isize++){
  nt_[isize] = fs->make<TTree>(triggerNames_.at(isize).c_str(),Form("trigger %d",isize));
  }


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

    //float id = -99,pt=-99,eta=-99,phi=-99,mass=-99;

    using namespace edm;
    iEvent.getByLabel(triggerEventTag_,triggerEventHandle_);
    iEvent.getByLabel(triggerResultsTag_,triggerResultsHandle_);

    for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){
	std::map<std::string,bool>::iterator inMenu = triggerInMenu.find(triggerNames_[itrig]);
        if (inMenu==triggerInMenu.end()){ continue; }
      
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
	    if(VIDS[i]>0){ // && pt<TO.pt()){
	      if(verbose_){
		cout << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": "
	                    << TO.id() << " " << TO.pt() << " " << TO.et() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass()
	                              << endl;
	      }
	      id[itrig].push_back(TO.id());
	      pt[itrig].push_back(TO.pt());
	      eta[itrig].push_back(TO.eta());
	      phi[itrig].push_back(TO.phi());
	      mass[itrig].push_back(TO.mass());
	    }
	  }
	}
      }
    }

    //nt_[0]->Fill(id,pt,eta,phi,mass);
  }
  for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){
	  nt_[itrig]->Fill();
	  id[itrig].clear();
	  pt[itrig].clear();
	  eta[itrig].clear();
	  phi[itrig].clear();
	  mass[itrig].clear();
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
      std::vector<std::string> activeHLTPathsInThisEvent = hltConfig_.triggerNames();
     
      triggerInMenu.clear(); 
      for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){
	for (std::vector<std::string>::const_iterator iHLT = activeHLTPathsInThisEvent.begin(); iHLT != activeHLTPathsInThisEvent.end(); ++iHLT){
              //matching with regexp filter name. More than 1 matching filter is allowed so trig versioning is transparent to analyzer
              if (TString(*iHLT).Contains(TRegexp(TString(triggerNames_[itrig])))){
                  triggerInMenu[*iHLT] = true;
                  triggerNames_[itrig] = TString(*iHLT);
              }
          }
      }
      for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){
        std::map<std::string,bool>::iterator inMenu = triggerInMenu.find(triggerNames_[itrig]);
	if (inMenu==triggerInMenu.end()) {
            cout << "<HLT Object Analyzer> Warning! Trigger " << triggerNames_[itrig] << " not found in HLTMenu. Skipping..." << endl;
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
    cout << "HLTObjectAnalyzer::analyze:"
	 << " config extraction failure with process name "
	 << processName_ << endl;
  }

  for(unsigned int itrig=0; itrig<triggerNames_.size(); itrig++){
    nt_[itrig]->Branch("TriggerObjID",&(id[itrig]));
    nt_[itrig]->Branch("pt",&(pt[itrig]));
    nt_[itrig]->Branch("eta",&(eta[itrig]));
    nt_[itrig]->Branch("phi",&(phi[itrig]));
    nt_[itrig]->Branch("mass",&(mass[itrig]));
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
