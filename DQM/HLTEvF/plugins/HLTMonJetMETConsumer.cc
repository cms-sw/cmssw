#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/HLTMonJetMETConsumer.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

/*
Todo: 
- convert print statements into LogXXX printouts
- option to restrict to certain eta/phi regions
- 
 */
using namespace edm;

HLTMonJetMETConsumer::HLTMonJetMETConsumer(const edm::ParameterSet& iConfig)
{
  
  //  LogDebug("HLTMonJetMETConsumer") << "constructor...." ;
  
  logFile_.open("HLTMonJetMETConsumer.log");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }
  
  outputFile_ =
    iConfig.getUntrackedParameter <std::string>("outputFile", "");
  if (outputFile_.size() != 0) {
    LogInfo("HLTMonJetMETConsumer") << "L1T Monitoring histograms will be saved to " 
			      << outputFile_ ;
  }
  else {
    outputFile_ = "L1TDQM.root";
  }
  
  bool disable =
    iConfig.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
  

  // read in the list of reference and probe filters
  std::vector<edm::ParameterSet> reffilters = iConfig.getParameter<std::vector<edm::ParameterSet> >("reffilters");
  std::vector<edm::ParameterSet> probefilters = iConfig.getParameter<std::vector<edm::ParameterSet> >("probefilters");

  for(std::vector<edm::ParameterSet>::iterator filterconf = reffilters.begin() ; filterconf != reffilters.end() ; filterconf++){
    theHLTRefLabels.push_back(filterconf->getParameter<std::string>("HLTRefLabels"));
//     std::cout << "[==> DEBUG] Ref: " << filterconf->getParameter<std::string>("HLTRefLabels") << std::endl;
  }
  RefLabelSize = theHLTRefLabels.size();

  for(std::vector<edm::ParameterSet>::iterator filterconf = probefilters.begin() ; filterconf != probefilters.end() ; filterconf++){
    theHLTProbeLabels.push_back(filterconf->getParameter<std::string>("HLTProbeLabels"));
//     std::cout << "[==> DEBUG] Probe: " << filterconf->getParameter<std::string>("HLTProbeLabels") << std::endl;
  }
  ProbeLabelSize = theHLTProbeLabels.size();


  if (RefLabelSize != ProbeLabelSize){
    LogError("HLTMonJetMETConsumer") << "Number of reference and probe filters must be the same\n. Please check your configuration.";
    return;
  }
  if (ProbeLabelSize == 0){
    LogError("HLTMonJetMETConsumer") << "No filter for monitoring specified. You need to configure at least one filter.";
    return;
  }

  
  dirname_="HLT/HLTMonhltMonJetMET/"+iConfig.getParameter<std::string>("@module_label");

//   std::string thislabel_ = iConfig.getParameter<std::string>("@module_label");
//   printf("[DEBUG] label = %s\n",thislabel_.c_str());
//   printf("[DEBUG] dirname = %s\n",dirname_.c_str());

  if (dbe != NULL) {
    dbe->setCurrentFolder(dirname_);
  }
  
}


HLTMonJetMETConsumer::~HLTMonJetMETConsumer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
HLTMonJetMETConsumer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   if(!runConsumer) return;  
//   printf("[DEBUG]: HLTMonJetMETConsumer (analyze)\n");
  if (ProbeLabelSize == 0){
    LogError("HLTMonJetMETConsumer") << "0 filters for monitoring specified. Nothing will be done.";
    return;
  }

//   printf("[DEBUG] working on event %i\n",ievt);
  ievt++;

  TH1F *num_Et=NULL;
  TH1F *num_Eta=NULL;
  TH1F *num_Phi=NULL;
  TH1F *denom_Et=NULL;
  TH1F *denom_Eta=NULL;
  TH1F *denom_Phi=NULL;
  for(int i=0; i < RefLabelSize;i++ ){
    if(injetmet_probe_Et[i]) num_Et = injetmet_probe_Et[i]->getTH1F();
    if(injetmet_ref_Et[i]) denom_Et = injetmet_ref_Et[i]->getTH1F();
    if(injetmet_probe_Eta[i]) num_Eta = injetmet_probe_Eta[i]->getTH1F();
    if(injetmet_ref_Eta[i]) denom_Eta = injetmet_ref_Eta[i]->getTH1F();
    if(injetmet_probe_Phi[i]) num_Phi = injetmet_probe_Phi[i]->getTH1F();
    if(injetmet_ref_Phi[i]) denom_Phi = injetmet_ref_Phi[i]->getTH1F();

    if (!num_Et || !num_Eta || !num_Phi) {
      std::cout << "Can't find probe " << theHLTProbeLabels[i] << std::endl;
      return;
    }
    if (!denom_Et || !denom_Eta || !denom_Phi) {
      std::cout << "Can't find reference " << theHLTProbeLabels[i] << std::endl;
      return;
    }

    for(int j=1; j <= num_Et->GetXaxis()->GetNbins();j++ ){
      double y1 = num_Et->GetBinContent(j);
      double y2 = denom_Et->GetBinContent(j);
      double eff = y2 > 0. ? y1/y2 : 0.;
      outjetmet_Et[i]->setBinContent(j, y1);
      outjetmet_Eff_Et[i]->setBinContent(j, eff);
//       printf("  [DEBUG] Path %i, Bin %i, y1 = %f, y2 = %f, eff = %f\n",i,j,y1,y2,eff);
    }
    for(int j=1; j <= num_Eta->GetXaxis()->GetNbins();j++ ){
      double y1 = num_Eta->GetBinContent(j);
      double y2 = denom_Eta->GetBinContent(j);
      double eff = y2 > 0. ? y1/y2 : 0.;
      outjetmet_Eta[i]->setBinContent(j, y1);
      outjetmet_Eff_Eta[i]->setBinContent(j, eff);
//       printf("  [DEBUG] Path %i, Bin %i, y1 = %f, y2 = %f, eff = %f\n",i,j,y1,y2,eff);
    }
    for(int j=1; j <= num_Phi->GetXaxis()->GetNbins();j++ ){
      double y1 = num_Phi->GetBinContent(j);
      double y2 = denom_Phi->GetBinContent(j);
      double eff = y2 > 0. ? y1/y2 : 0.;
      outjetmet_Phi[i]->setBinContent(j, y1);
      outjetmet_Eff_Phi[i]->setBinContent(j, eff);
//       printf("  [DEBUG] Path %i, Bin %i, y1 = %f, y2 = %f, eff = %f\n",i,j,y1,y2,eff);
    }

  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
HLTMonJetMETConsumer::beginJob(const edm::EventSetup&)
{
  ievt = 0;
  runConsumer=false;
//   printf("[DEBUG]: HLTMonJetMETConsumer (beginJob)\n");
  
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    
    
    std::string histdir_ = "HLT/HLTMonhltMonJetMET";
    
     std::string MEname;
     TH1F *htemp = NULL;

     MEname = histdir_ + "/" + theHLTRefLabels[0] + "et";
//      printf("[DEBUG] MEname = %s\n",MEname.c_str());
     MEtemp = dbe->get(MEname);
//      if (!MEtemp) {printf("Error in HLTMonJetMETConsumer: %s not found.\n",MEname.c_str()); }
     if (!MEtemp) {LogError("HLTMonJetMETConsumer") << "Error! Filter " << MEname << " not found.\n";}
     if (!MEtemp) {std::cout<<"HLTMonJetMETConsumer: Error! Filter " << MEname << " not found.\n";}
     if(!MEtemp) return;
     htemp = MEtemp->getTH1F();
     int nbin_Et = htemp->GetNbinsX();
     double xmin_Et = htemp->GetXaxis()->GetXmin();
     double xmax_Et = htemp->GetXaxis()->GetXmax();
//      printf("[DEBUG] nbin = %i, xmin = %f, xmax = %f\n",nbin_Et, xmin_Et, xmax_Et);

     MEname = histdir_ + "/" + theHLTRefLabels[0] + "eta";
//      printf("[DEBUG] MEname = %s\n",MEname.c_str());
     MEtemp = dbe->get(MEname);
     if(!MEtemp) return;
     htemp = MEtemp->getTH1F();
     int nbin_Eta = htemp->GetNbinsX();
     double xmin_Eta = htemp->GetXaxis()->GetXmin();
     double xmax_Eta = htemp->GetXaxis()->GetXmax();
//      printf("[DEBUG] nbin = %i, xmin = %f, xmax = %f\n",nbin_Eta, xmin_Eta, xmax_Eta);

     MEname = histdir_ + "/" + theHLTRefLabels[0] + "phi";
//      printf("[DEBUG] MEname = %s\n",MEname.c_str());
     MEtemp = dbe->get(MEname);
     if(!MEtemp) return;
     htemp = MEtemp->getTH1F();
     int nbin_Phi = htemp->GetNbinsX();
     double xmin_Phi = htemp->GetXaxis()->GetXmin();
     double xmax_Phi = htemp->GetXaxis()->GetXmax();
//      printf("[DEBUG] nbin = %i, xmin = %f, xmax = %f\n",nbin_Phi, xmin_Phi, xmax_Phi);

     for (int i=0; i<ProbeLabelSize; i++){
       // read the MEs from the source
       MEname = histdir_ + "/" + theHLTRefLabels[i] + "et";
       injetmet_ref_Et[i] = dbe->get(MEname);
       MEname = histdir_ + "/" + theHLTRefLabels[i] + "eta";
       injetmet_ref_Eta[i] = dbe->get(MEname);
       MEname = histdir_ + "/" + theHLTRefLabels[i] + "phi";
       injetmet_ref_Phi[i] = dbe->get(MEname);

       MEname = histdir_ + "/" + theHLTProbeLabels[i] + "et";
       injetmet_probe_Et[i] = dbe->get(MEname);
       MEname = histdir_ + "/" + theHLTProbeLabels[i] + "eta";
       injetmet_probe_Eta[i] = dbe->get(MEname);
       MEname = histdir_ + "/" + theHLTProbeLabels[i] + "phi";
       injetmet_probe_Phi[i] = dbe->get(MEname);

       if (!injetmet_ref_Et[i]) {
	 std::cout << "Can't find reference " << std::endl;
       }
       if (!injetmet_probe_Et[i]) {
	 std::cout << "Can't find probe " << std::endl;
       }

       // book outgoing MEs (probes)
       //       outjetmet_Et[i] = dbe->book1D(Form("%s_Et_%i",i),theHLTProbeLabels[i],Form("Et %i",i),nbin_Et, xmin_Et, xmax_Et);
       outjetmet_Et[i]  = dbe->book1D(Form("%s_Et",   theHLTProbeLabels[i].c_str()),
				      Form("Et (%s)", theHLTProbeLabels[i].c_str()),nbin_Et, xmin_Et, xmax_Et);
       outjetmet_Eta[i] = dbe->book1D(Form("%s_Eta",  theHLTProbeLabels[i].c_str()),
				      Form("Eta (%s)",theHLTProbeLabels[i].c_str()),nbin_Eta, xmin_Eta, xmax_Eta);
       outjetmet_Phi[i] = dbe->book1D(Form("%s_Phi",  theHLTProbeLabels[i].c_str()),
				      Form("Phi (%s)",theHLTProbeLabels[i].c_str()),nbin_Phi, xmin_Phi, xmax_Phi);
       
       outjetmet_Eff_Et[i]  = dbe->book1D(Form("Eff_%s_Et",theHLTProbeLabels[i].c_str()),
					  Form("Eff vs Et %s / %s",theHLTProbeLabels[i].c_str(),theHLTRefLabels[i].c_str()),
					  nbin_Et, xmin_Et, xmax_Et);
       outjetmet_Eff_Eta[i] = dbe->book1D(Form("Eff_%s_Eta",theHLTProbeLabels[i].c_str()),
					  Form("Eff vs Eta %s / %s",theHLTProbeLabels[i].c_str(),theHLTRefLabels[i].c_str()),
					  nbin_Eta, xmin_Eta, xmax_Eta);
       outjetmet_Eff_Phi[i] = dbe->book1D(Form("Eff_%s_Phi",theHLTProbeLabels[i].c_str()),
					  Form("Eff vs Phi %s / %s",theHLTProbeLabels[i].c_str(),theHLTRefLabels[i].c_str()),
					  nbin_Phi, xmin_Phi, xmax_Phi);
     }
     
    LogDebug("HLTMonJetMETConsumer") << " reading histo: "  << histdir_;    

  } // end "if(dbe)"
  runConsumer=true;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTMonJetMETConsumer::endJob() {

   if (outputFile_.size() != 0 && dbe)
     dbe->save(outputFile_);
 
   return;
}

//DEFINE_FWK_MODULE(HLTMonJetMETConsumer);
