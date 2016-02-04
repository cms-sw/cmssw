#include "DQMOffline/JetMET/interface/JetMETDQMOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

JetMETDQMOfflineClient::JetMETDQMOfflineClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{

  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogError("JetMETDQMOfflineClient") 
    << "unable to get DQMStore service, upshot is no client histograms will be made";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    if(dbe_) dbe_->setVerbose(0);
  }
 
  verbose_ = conf_.getUntrackedParameter<int>("Verbose");

  dirName_=iConfig.getUntrackedParameter<std::string>("DQMDirName");
  if(dbe_) dbe_->setCurrentFolder(dirName_);

  dirNameJet_=iConfig.getUntrackedParameter<std::string>("DQMJetDirName");
  dirNameMET_=iConfig.getUntrackedParameter<std::string>("DQMMETDirName");
 
}


JetMETDQMOfflineClient::~JetMETDQMOfflineClient()
{ 
  
}

void JetMETDQMOfflineClient::beginJob(void)
{
 

}

void JetMETDQMOfflineClient::endJob() 
{

}

void JetMETDQMOfflineClient::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
 
}


void JetMETDQMOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  runClient_();
}

//dummy analysis function
void JetMETDQMOfflineClient::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{
  
}

void JetMETDQMOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c)
{ 

}

void JetMETDQMOfflineClient::runClient_()
{
  
  if(!dbe_) return; //we dont have the DQMStore so we cant do anything

  LogDebug("JetMETDQMOfflineClient") << "runClient" << std::endl;
  if (verbose_) std::cout << "runClient" << std::endl; 

  std::vector<MonitorElement*> MEs;

  ////////////////////////////////////////////////////////
  // Total number of lumi sections and time for this run
  ///////////////////////////////////////////////////////

  TH1F* tlumisec;
  MonitorElement *meLumiSec = dbe_->get("JetMET/lumisec");

  int    totlsec=0;
  double totltime=0;

  if ( meLumiSec->getRootObject() ){

    tlumisec = meLumiSec->getTH1F();    
    for (int i=0; i<500; i++){
      if (tlumisec->GetBinContent(i+1)) totlsec++;
    }

  }
  totltime = totlsec * 90.;

  /////////////////////////
  // MET
  /////////////////////////

  dbe_->setCurrentFolder(dirName_+"/"+dirNameMET_);

  MonitorElement *me;
  TH1F *tMET;
  TH1F *tMETRate;
  MonitorElement *hMETRate;

  //
  // --- Producing MET rate plots
  //

  // Look at all folders (JetMET/MET/CaloMET, JetMET/MET/CaloMETNoHF, etc)
  std::vector<std::string> fullPathDQMFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathDQMFolders.size();i++) {

    if (verbose_) std::cout << fullPathDQMFolders[i] << std::endl;      
    dbe_->setCurrentFolder(fullPathDQMFolders[i]);

    // Look at all subfolders (JetMET/MET/CaloMET/{All,Cleaup,BeamHaloIDLoosePass, etc})
    std::vector<std::string> fullPathDQMSubFolders = dbe_->getSubdirs();
    for(unsigned int j=0;j<fullPathDQMSubFolders.size();j++) {

      if (verbose_) std::cout << fullPathDQMSubFolders[j] << std::endl;      
      dbe_->setCurrentFolder(fullPathDQMSubFolders[j]);
      if (verbose_) std::cout << "setCurrentFolder done" << std::endl;      

      // Look at all MonitorElements in this folder
      std::string METMEName="METTask_CaloMET";
      if ( dbe_->get(fullPathDQMSubFolders[j]+"/"+"METTask_MET") )  METMEName="METTask_MET";
      if ( dbe_->get(fullPathDQMSubFolders[j]+"/"+"METTask_PfMET")) METMEName="METTask_PfMET";

      me = dbe_->get(fullPathDQMSubFolders[j]+"/"+METMEName);
      if (verbose_) std::cout << "get done" << std::endl;      
      
      if ( me ) {
	if (verbose_) std::cout << "me true" << std::endl;      
	if ( me->getRootObject() ) {
	  if (verbose_) std::cout << "me getRootObject true" << std::endl;      

	  MonitorElement *metest = dbe_->get(fullPathDQMSubFolders[j]+"/"+METMEName+"Rate");
	  if (metest) dbe_->removeElement(METMEName+"Rate");

	  if (verbose_) std::cout << "metest done" << std::endl;      
	  
	  tMET     = me->getTH1F();
	  if (verbose_) std::cout << "getTH1F done" << std::endl;      

	  // Integral plot 
	  tMETRate = (TH1F*) tMET->Clone((METMEName+"Rate").c_str());
	  for (int i = tMETRate->GetNbinsX()-1; i>=0; i--){
	    tMETRate->SetBinContent(i+1,tMETRate->GetBinContent(i+2)+tMET->GetBinContent(i+1));
	  }
	  if (verbose_) std::cout << "making integral plot done" << std::endl;      
      
	  // Convert number of events to Rate (Hz)
	  if (totltime){
	    for (int i=tMETRate->GetNbinsX()-1;i>=0;i--){
	      tMETRate->SetBinContent(i+1,tMETRate->GetBinContent(i+1)/double(totltime));
	    }
	  }
	  if (verbose_) std::cout << "making rate plot done" << std::endl;      

	  hMETRate      = dbe_->book1D(METMEName+"Rate",tMETRate);
 	  hMETRate->setTitle(METMEName+" Rate");
 	  hMETRate->setAxisTitle("MET Threshold [GeV]",1);
	  if (verbose_) std::cout << "booking rate plot ME done" << std::endl;      

	} // me->getRootObject()
      } // me
      if (verbose_) std::cout << "end of subfolder loop" << std::endl;
    }   // fullPathDQMSubFolders-loop - All, Cleanup, ...
    if (verbose_) std::cout << "end of folder loop" << std::endl;
  }   // fullPathDQMFolders-loop - CaloMET, CaloMETNoHF, ...

  /////////////////////////
  // Jet
  /////////////////////////
   
  dbe_->setCurrentFolder(dirName_+"/"+dirNameJet_);

  MonitorElement *hPassFraction;

  // Look at all folders (JetMET/Jet/AntiKtJets,JetMET/Jet/CleanedAntiKtJets, etc)
  fullPathDQMFolders.clear();
  fullPathDQMFolders = dbe_->getSubdirs();
  for(unsigned int i=0;i<fullPathDQMFolders.size();i++) {

    if (verbose_) std::cout << fullPathDQMFolders[i] << std::endl;      
    dbe_->setCurrentFolder(fullPathDQMFolders[i]);

    std::vector<std::string> getMEs = dbe_->getMEs();

    std::vector<std::string>::const_iterator cii;
    for(cii=getMEs.begin(); cii!=getMEs.end(); cii++) {
      if ((*cii).find("_binom")!=std::string::npos) continue;
      if ((*cii).find("JIDPassFractionVS")!=std::string::npos){  // Look for MEs with "JIDPassFractionVS"
	me = dbe_->get(fullPathDQMFolders[i]+"/"+(*cii));

	if ( me ) {
	if ( me->getRootObject() ) {
	  TProfile *tpro = (TProfile*) me->getRootObject();
	  TH1F *tPassFraction = new TH1F(((*cii)+"_binom").c_str(),((*cii)+"_binom").c_str(),
	    tpro->GetNbinsX(),tpro->GetBinLowEdge(1),tpro->GetBinLowEdge(tpro->GetNbinsX()+1));
	  for (int ibin=0; ibin<tpro->GetNbinsX(); ibin++){
	    double nentries = tpro->GetBinEntries(ibin+1);
	    double epsilon  = tpro->GetBinContent(ibin+1);
	    if (epsilon>1. || epsilon<0.) continue;
	    tPassFraction->SetBinContent(ibin+1,epsilon);                        // 
	    if(nentries>0) tPassFraction->SetBinError(ibin+1,pow(epsilon*(1.-epsilon)/nentries,0.5));
	  }
	  hPassFraction      = dbe_->book1D(tPassFraction->GetName(),tPassFraction);
	  delete tPassFraction;
	} // me->getRootObject()
	} // me
      } // if find
    }   // cii-loop

  } // i-loop 

}

