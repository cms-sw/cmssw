#include "DQMOffline/Ecal/interface/EcalOfflineCosmicClient.h"

using namespace std;

EcalOfflineCosmicClient::EcalOfflineCosmicClient(const edm::ParameterSet& iConfig) :
   fileName_ (iConfig.getUntrackedParameter<string>("fileName",string("EcalOfflineCosmicClient.root"))),
   saveFile_ (iConfig.getUntrackedParameter<bool>("saveFile",false)),
   endFunction_ (iConfig.getParameter<string>("endFunction")),
   rootDir_ (iConfig.getParameter<string>("rootDir")),
   subDetDirs_ (iConfig.getParameter<vector<string> >("subDetDirs")),
   l1TriggerDirs_ (iConfig.getParameter<vector<string> >("l1TriggerDirs")),
   timingDir_ (iConfig.getParameter<string>("timingDir")),
   timingVsAmp_ (iConfig.getParameter<string>("timingVsAmp")),
   timingTTBinned_ (iConfig.getParameter<string>("timingTTBinned")),
   timingModBinned_ (iConfig.getParameter<string>("timingModBinned")),
   clusterDir_ (iConfig.getParameter<string>("clusterDir")),
   clusterPlots_ (iConfig.getParameter<vector<string> >("clusterPlots"))
{
   dbe_ = 0;
   dbe_ = edm::Service<DQMStore>().operator->();
}

EcalOfflineCosmicClient::~EcalOfflineCosmicClient() {
}

void EcalOfflineCosmicClient::beginJob(const edm::EventSetup& iSetup) {
}

void EcalOfflineCosmicClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
}

void EcalOfflineCosmicClient::endJob() {
   if(endFunction_ == "endJob")
      end();
}

void 
EcalOfflineCosmicClient::endLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& iSetup)
{
   if(endFunction_ == "endLuminosityBlock")
      end();
}

void EcalOfflineCosmicClient::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
   if(endFunction_ == "endRun")
      end();
}

void EcalOfflineCosmicClient::end() {
   for(vector<string>::const_iterator subDetItr=subDetDirs_.begin(); 
	 subDetItr!=subDetDirs_.end();++subDetItr) {

      string clusterPath=rootDir_+"/"+(*subDetItr)+"/AllEvents/"+clusterDir_;

      dbe_->setCurrentFolder(clusterPath);
      if((*subDetItr) == "EB") {
	 doProfile(clusterPath,"NumBCinSCphi2D");
      }
      doProfile(clusterPath,"NumXtalsVsEnergy2D");

      for(vector<string>::const_iterator l1TrigItr=l1TriggerDirs_.begin(); 
	    l1TrigItr!=l1TriggerDirs_.end();++l1TrigItr) {

	 string timingPath=rootDir_+"/"+(*subDetItr)+"/"+(*l1TrigItr)+"/"+timingDir_;
	 dbe_->setCurrentFolder(timingPath);

	 if((*l1TrigItr) == "AllEvents") {
	    doProfile(timingPath,timingVsAmp_);
	 }

	 doProfile(timingPath,timingTTBinned_);

	 if((*subDetItr) == "EB") {
	    doProfile(timingPath,timingModBinned_);
	 }
      }
   }
   if(saveFile_)
      dbe_->save(fileName_);
}

void EcalOfflineCosmicClient::doProfile3D(MonitorElement* me, string name) {
   TProfile2D* p2f = (TProfile2D*) me->getTH3F()->Project3DProfile("yx");
   MonitorElement* meProfile = dbe_->bookProfile2D(name,p2f);
   string title = meProfile->getTProfile2D()->GetTitle();
   if(title.substr(title.size()-4) == "_pyx")
      meProfile->getTProfile2D()->SetTitle(title.substr(0,title.size()-4).c_str());
}

void EcalOfflineCosmicClient::doProfileX(MonitorElement* me, string name) {
   TProfile* p1f = (TProfile*) me->getTH2F()->ProfileX();
   dbe_->bookProfile(name,p1f);
}

void EcalOfflineCosmicClient::doProfile(string path, string name) {
   MonitorElement* me = dbe_->get(path+"/"+name);
   if(me != NULL) {
      string oldName = name;
      name = oldName;
      name.erase(name.end()-2,name.end());

      if(me->kind() == MonitorElement::DQM_KIND_TH3F) 
	 doProfile3D(me,name);

      else if(me->kind() == MonitorElement::DQM_KIND_TH2F) {
	 doProfileX(me,name);
      }

      dbe_->removeElement(oldName);
   }
}
