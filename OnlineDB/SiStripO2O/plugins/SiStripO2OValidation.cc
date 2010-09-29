// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "TBranch.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TIterator.h"
#include "TKey.h"
#include "TLegend.h"
#include "TObject.h"
#include "TObjString.h"
#include "TString.h"
#include "TTree.h"


//
// class decleration
//

class SiStripO2OValidation : public edm::EDAnalyzer {
public:
  explicit SiStripO2OValidation(const edm::ParameterSet&);
  ~SiStripO2OValidation();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void ValidateNoise(int treenr1, int treenr2);
  void ValidatePedestal(int treenr1, int treenr2);
  void ValidateThreshold(int treenr1, int treenr2);
  void ValidateFedCabling(int treenr1, int treenr2);
  void ValidateQuality(int treenr1, int treenr2);
  void ValidateTiming(int treenr1, int treenr2);
  void ValidateLatency(int treenr1, int treenr2);

   virtual void endJob() ;
  
  //Tree variables
  TTree *val_tree[24];
  TString tree_name[24];
  TFile *inputfile;
  int nrtrees;
   
  //cfg input parameters
  TString cfg_rootfile;
  TString cfg_fileextension;
  bool cfg_FEDCabling;
  bool cfg_Threshold;
  bool cfg_Quality;
  bool cfg_Noise;
  bool cfg_Pedestal;
  bool cfg_APVLatency;
  bool cfg_APVTiming;
  bool cfg_debug;     

  //CMSSW Version
  TString cmssw1;
  TString cmssw2;
  
  
  // ----------member data ---------------------------
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
SiStripO2OValidation::SiStripO2OValidation(const edm::ParameterSet& iConfig):
  cfg_rootfile(static_cast<TString>(iConfig.getUntrackedParameter<std::string>("RootFile","SiStripO2OValidation.root"))),
  cfg_fileextension(static_cast<TString>(iConfig.getUntrackedParameter<std::string>("FileExtension","gif"))),
  cfg_FEDCabling(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateFEDCabling",false))),
  cfg_Threshold(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateThreshold",false))),
  cfg_Quality(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateQuality",false))),
  cfg_Noise(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateNoise",false))),
  cfg_Pedestal(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidatePedestal",false))),
  cfg_APVLatency(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateAPVLatency",false))),
  cfg_APVTiming(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateAPVTiming",false))),
  cfg_debug(static_cast<bool>(iConfig.getUntrackedParameter<bool>("DebugMode",false)))
{
  inputfile=new TFile(cfg_rootfile,"update");
  inputfile->cd();
  TIter nextkey(gDirectory->GetListOfKeys());
  TKey *key;
  while ((key = (TKey*)nextkey())) {
    TObject *obj = key->ReadObj();
    if ( obj->IsA()->InheritsFrom( "TTree" )){
        tree_name[nrtrees]=obj->GetName();
        val_tree[nrtrees]=static_cast<TTree*>(gDirectory->Get(tree_name[nrtrees]));
	++nrtrees;
    } 
  }

  cmssw1="CMSSW";
  cmssw2="CMSSW";
  for(int j=0;j<nrtrees;++j){
    TObjArray *array  = tree_name[j].Tokenize("_");
    TObjString *tmpstr[100];
    // TObjArray *tmparray;
    
    bool foundtree=false;
    for (int i=0; i<array->GetSize();i++){
      tmpstr[i]=((TObjString*)(array->At(i)));
      if(tmpstr[i]==0) break;
      if((tmpstr[i]->GetString()).Contains("CMSSW")){foundtree=true; continue;}
      
      if(j==0 && foundtree==true){
	cmssw1=cmssw1+"_"+tmpstr[i]->GetString();
      }
      if(j!=0 && foundtree==true){
	cmssw2=cmssw2+"_"+tmpstr[i]->GetString();
      }
    }
    if(cmssw1 !=cmssw2 && cmssw2!="CMSSW") {std::cout << "Version1: " << cmssw1 << " Version2: "<< cmssw2<< std::endl; break;}else{cmssw2="CMSSW";}
    
  }
}

SiStripO2OValidation::~SiStripO2OValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripO2OValidation::beginJob()
{
  
}

// ------------ method called to for each event  ------------
void
SiStripO2OValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  TString noisetree1="nothing";        int noi_t1=4711;
    TString noisetree2="nothing";      int noi_t2=4711;
    TString thresholdtree1="nothing";  int thr_t1=4711;
    TString thresholdtree2="nothing";  int thr_t2=4711;
    TString fedcablingtree1="nothing"; int fed_t1=4711;
    TString fedcablingtree2="nothing"; int fed_t2=4711;
    TString qualitytree1="nothing";    int qua_t1=4711;
    TString qualitytree2="nothing";    int qua_t2=4711;
    TString pedestaltree1="nothing";   int ped_t1=4711;
    TString pedestaltree2="nothing";   int ped_t2=4711;
    TString timingtree1="nothing";     int tim_t1=4711;
    TString timingtree2="nothing";     int tim_t2=4711;
    TString latencytree1="nothing";    int lat_t1=4711;
    TString latencytree2="nothing";    int lat_t2=4711;

  for(int i=0;i<nrtrees;++i){
    TString tree_name;

    tree_name=val_tree[i]->GetTitle();
    //Noise
    if(tree_name.Contains("Noise") && noisetree1.Contains("nothing") && noisetree2.Contains("nothing")){
      noisetree1=tree_name;
      noi_t1=i;
      continue;
    } 
    if(tree_name.Contains("Noise") && !(noisetree1.Contains("nothing")) && noisetree2.Contains("nothing")){
      noisetree2=tree_name;
      std::cout << noisetree2 << std::endl;
      noi_t2=i;
      if(cfg_Noise){ValidateNoise(noi_t1, noi_t2);}
      continue;
    } 
    //Pedestal
    if(tree_name.Contains("Pedestal") && pedestaltree1.Contains("nothing") && pedestaltree2.Contains("nothing")){
      pedestaltree1=tree_name;
      ped_t1=i;
      continue;
    } 
    if(tree_name.Contains("Pedestal") && !(pedestaltree1.Contains("nothing")) && pedestaltree2.Contains("nothing")){
      pedestaltree2=tree_name;
      ped_t2=i;
      if(cfg_Pedestal) ValidatePedestal(ped_t1, ped_t2);
      continue;
    } 
    //Threshold
        if(tree_name.Contains("Threshold") && thresholdtree1.Contains("nothing") && thresholdtree2.Contains("nothing")){
      thresholdtree1=tree_name;
      thr_t1=i;
      continue;
    } 
    if(tree_name.Contains("Threshold") && !(thresholdtree1.Contains("nothing")) && thresholdtree2.Contains("nothing")){
      thresholdtree2=tree_name;
      thr_t2=i;
      if(cfg_Threshold) ValidateThreshold(thr_t1, thr_t2);
      continue;
    } 
    //FedCabling
    if(tree_name.Contains("FEDCabling") && fedcablingtree1.Contains("nothing") && fedcablingtree2.Contains("nothing")){
      fedcablingtree1=tree_name;
      fed_t1=i;
      continue;
    } 
    if(tree_name.Contains("FEDCabling") && !(fedcablingtree1.Contains("nothing")) && fedcablingtree2.Contains("nothing")){
      fedcablingtree2=tree_name;
      fed_t2=i;
      if(cfg_FEDCabling) ValidateFedCabling(fed_t1, fed_t2);
      continue;
    } 
    //Quality
    if(tree_name.Contains("BadStrip") && qualitytree1.Contains("nothing") && qualitytree2.Contains("nothing")){
      qualitytree1=tree_name;
      qua_t1=i;
      continue;
    } 
    if(tree_name.Contains("BadStrip") && !(qualitytree1.Contains("nothing")) && qualitytree2.Contains("nothing")){
      qualitytree2=tree_name;
      qua_t2=i;
      if(cfg_Quality) ValidateQuality(qua_t1, qua_t2);
      continue;
    } 
    //Timing
    if(tree_name.Contains("APVTiming") && timingtree1.Contains("nothing") && timingtree2.Contains("nothing")){
      timingtree1=tree_name;
      tim_t1=i;
      continue;
    } 
    if(tree_name.Contains("APVTiming") && !(timingtree1.Contains("nothing")) && timingtree2.Contains("nothing")){
      timingtree2=tree_name;
      tim_t2=i;
      if(cfg_APVTiming) ValidateTiming(tim_t1, tim_t2);
      continue;
    } 
    //Latency
    if(tree_name.Contains("APVLatency") && latencytree1.Contains("nothing") && latencytree2.Contains("nothing")){
      latencytree1=tree_name;
      lat_t1=i;
      continue;
    } 
    if(tree_name.Contains("APVLatency") && !(latencytree1.Contains("nothing")) && latencytree2.Contains("nothing")){
      latencytree2=tree_name;
      lat_t2=i;
      if(cfg_APVLatency) ValidateLatency(lat_t1, lat_t2);
      continue;
    } 


  }


  if(cfg_debug){
    std::cout <<"  noisetree1: "<<  noisetree1 << std::endl;
    std::cout <<"  noisetree2:" <<  noisetree2 << std::endl;
    std::cout <<"  thresholdtree1: "<<  thresholdtree1 << std::endl;
    std::cout <<"  thresholdtree2: "<<  thresholdtree2 << std::endl;
    std::cout <<"  fedcablingtree1: "<<  fedcablingtree1 << std::endl;
    std::cout <<"  fedcablingtree2: "<<  fedcablingtree2 << std::endl;
    std::cout <<"  qualitytree1: "<<  qualitytree1 << std::endl;
    std::cout <<"  qualitytree2: "<<  qualitytree2 << std::endl;
    std::cout <<"  pedestaltree1: "<<  pedestaltree1 << std::endl;
    std::cout <<"  pedestaltree2: "<<  pedestaltree2 << std::endl;
    std::cout <<"  timingtree1: "<<  timingtree1 << std::endl;
    std::cout <<"  timingtree2: "<<  timingtree2 << std::endl;
    std::cout <<"  latencytree1: "<<  latencytree1 << std::endl;
    std::cout <<"  latencytree2: "<<  latencytree2 << std::endl;
  }
 
   
}

///////////////////////////////////////////////////
//
//
//
void SiStripO2OValidation::ValidateNoise(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidateNoise]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

  TString name1="Noise_"+cmssw1;
  TString name2="Noise_"+cmssw2;
  TString nameval="Noise_Difference_"+cmssw1+"_"+cmssw2;
  TString mapval="SiStripO2ONoiseValidationMap_"+cmssw1+"_"+cmssw2;
  TString map1="SiStripO2ONoiseMap_"+cmssw1;
  TString map2="SiStripO2ONoiseMap_"+cmssw2;

   TH1F  *h_noiseval=new TH1F(nameval, nameval,100,-1000.,1000.);
   TH1F  *h_noise1=new TH1F(name1, name1, 100,0.,25.);
   TH1F  *h_noise2=new TH1F(name2, name2, 100,0.,25.);
   TrackerMap *tkmp_noiseval= new TrackerMap(mapval.Data());
   TrackerMap *tkmp_noise1= new TrackerMap(map1.Data());
   TrackerMap *tkmp_noise2= new TrackerMap(map2.Data());
     
   Int_t tree1_detid;
   float tree1_integrated_noise;
   TBranch        *b_tree1_detid;   //!
   TBranch *b_tree1_integrated_noise;
   val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
   val_tree[treenr1]->SetBranchAddress("INTEGRATEDNOISE",&tree1_integrated_noise, &b_tree1_integrated_noise);
   
   Int_t tree2_detid;
   float tree2_integrated_noise;
   TBranch        *b_tree2_detid;   //!
   TBranch *b_tree2_integrated_noise;
   val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
   val_tree[treenr2]->SetBranchAddress("INTEGRATEDNOISE",&tree2_integrated_noise, &b_tree2_integrated_noise);

   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
   Long64_t nbytes2 = 0, nb2 = 0;
   bool firstloop=true;
   int nrmatching=0;
   int nrnonmatching=0; 
   Long64_t lastmatch =0;

   for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
       h_noise1->Fill(tree1_integrated_noise);
       tkmp_noise1->fill(tree1_detid,tree1_integrated_noise);
       for (Long64_t jentry2=lastmatch; jentry2 < nentries2; jentry2++) {
       nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
        if(tree1_detid > tree2_detid ){
         h_noiseval->Fill(tree1_integrated_noise); 
         tkmp_noiseval->fill(tree1_detid,tree1_integrated_noise);
	 std::cout << "[SiStripO2OValidation::" << __func__
		   << "] DetId: "<< tree1_detid
		   << " is only present in tree1: "<< name1
		   << std::endl; 
	 //break;
       }
       if(firstloop){h_noise2->Fill(tree2_integrated_noise);
                     tkmp_noise2->fill(tree2_detid,tree2_integrated_noise);
                    }
       if(tree1_detid == tree2_detid){h_noiseval->Fill(tree1_integrated_noise-tree2_integrated_noise);
                                      tkmp_noiseval->fill(tree1_detid,tree1_integrated_noise-tree2_integrated_noise);
				      lastmatch=jentry2+1;
				      if(tree2_integrated_noise != tree1_integrated_noise){
					std::cout << "[SiStripO2OValidation::" << __func__
                                                  << "] Diff in IntNoise for DetId: "<< tree1_detid
                                                  << " tree1: "<< tree1_integrated_noise 
                                                  << " tree2: " << tree2_integrated_noise
                                                  << std::endl; 
					tkmp_noise2->fill(tree2_detid,tree2_integrated_noise);
					nrnonmatching++;
				      }else{
					nrmatching++;
					tkmp_noise2->fill(tree2_detid,tree2_integrated_noise);
				      }
				      break;
                                     }
       }
       firstloop=false;
   }

   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree1: " << nentries1 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree2: " << nentries2 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;

  if(cfg_debug){
    inputfile->cd();
    h_noiseval->Write();
    h_noise1->Write();
    h_noise2->Write();
    
    TCanvas *c_noiseval=new TCanvas(nameval,nameval,640,480);
    c_noiseval->cd();
    h_noiseval->Draw();
    TString c_valname=nameval+"."+cfg_fileextension;   
    c_noiseval->Print(c_valname);
    
    TCanvas *c_noise1=new TCanvas(name1,name1,640,480);
    c_noise1->cd();
    h_noise1->Draw();
    TString c_name1=name1+"."+cfg_fileextension;   
    c_noise1->Print(c_name1);
    
    TCanvas *c_noise2=new TCanvas(name2,name2,640,480);
    c_noise2->cd();
    h_noise2->Draw();
    TString c_name2=name2+"."+cfg_fileextension;   
    c_noise2->Print(c_name2);
  }

  TString mapval_name=mapval+"."+cfg_fileextension;
  tkmp_noiseval->save(true,-100.,100.,mapval_name.Data());
  TString map1_name=map1+"."+cfg_fileextension;
  tkmp_noise1->save(true,2000.,7500.,map1_name.Data());
  TString map2_name=map2+"."+cfg_fileextension;
  tkmp_noise2->save(true,2000.,7500.,map2_name.Data());


}

///////////////////////////////////////////////////
void SiStripO2OValidation::ValidatePedestal(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidatePedestal]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

  TString name1="Pedestal_"+cmssw1;
  TString name2="Pedestal_"+cmssw2;
  TString nameval="Pedestal_Difference_"+cmssw1+"_"+cmssw2;
  TString mapval="SiStripO2OPedestalValidationMap_"+cmssw1+"_"+cmssw2;
  TString map1="SiStripO2OPedestalMap_"+cmssw1;
  TString map2="SiStripO2OPedestalMap_"+cmssw2;

   TH1F  *h_pedval=new TH1F(nameval, nameval,100,-1000.,1000.);
   TH1F  *h_ped1=new TH1F(name1, name1, 100,0.,25.);
   TH1F  *h_ped2=new TH1F(name2, name2, 100,0.,25.);
   TrackerMap *tkmp_pedval= new TrackerMap(mapval.Data());
   TrackerMap *tkmp_ped1= new TrackerMap(map1.Data());
   TrackerMap *tkmp_ped2= new TrackerMap(map2.Data());
     
   Int_t tree1_detid;
   float tree1_integrated_pedestal;
   TBranch        *b_tree1_detid;   //!
   TBranch *b_tree1_integrated_pedestal;
   val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
   val_tree[treenr1]->SetBranchAddress("INTPEDESTAL",&tree1_integrated_pedestal, &b_tree1_integrated_pedestal);
 
  Int_t tree2_detid;
   float tree2_integrated_pedestal;
   TBranch        *b_tree2_detid;   //!
   TBranch *b_tree2_integrated_pedestal;
   val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
   val_tree[treenr2]->SetBranchAddress("INTPEDESTAL",&tree2_integrated_pedestal, &b_tree2_integrated_pedestal);
   
 
   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
   Long64_t nbytes2 = 0, nb2 = 0;
   bool firstloop=true;
   Long64_t lastmatch=0;
   int nrmatching =0;
   int nrnonmatching=0;
   for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
       h_ped1->Fill(tree1_integrated_pedestal);
       tkmp_ped1->fill(tree1_detid,tree1_integrated_pedestal);
       for (Long64_t jentry2=lastmatch; jentry2 < nentries2; jentry2++) {
       nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
       if(tree1_detid > tree2_detid ){
         h_pedval->Fill(tree1_integrated_pedestal); 
         tkmp_pedval->fill(tree1_detid,tree1_integrated_pedestal);
	 std::cout << " No Match found for Tree2 DetId: " << tree2_detid <<" "<< tree1_detid << " " <<lastmatch <<std::endl;
	 continue;
       }
       if(firstloop){h_ped2->Fill(tree2_integrated_pedestal);
                     tkmp_ped2->fill(tree2_detid,tree2_integrated_pedestal);
                    }
       if(tree1_detid == tree2_detid){h_pedval->Fill(tree1_integrated_pedestal - tree2_integrated_pedestal);
                                      tkmp_pedval->fill(tree1_detid,tree1_integrated_pedestal-tree2_integrated_pedestal);
				      tkmp_ped2->fill(tree2_detid,tree2_integrated_pedestal);
				      lastmatch=jentry2+1;
				      if(tree2_integrated_pedestal != tree1_integrated_pedestal){
					std::cout << "[SiStripO2OValidation::" << __func__ 
                                                  << "] Diff in IntPedestal for DetId: " << tree1_detid
                                                  << " tree1: " << tree1_integrated_pedestal
                                                  << " tree2: " << tree2_integrated_pedestal
                                                  << std::endl;
					nrnonmatching++;
					}else{
					nrmatching++;
				      }
				      break;
                                     }
       }
       firstloop=false;
   }

       std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree1: " << nentries1 << std::endl;
       std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree2: " << nentries2 << std::endl;
       std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
       std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;

 if(cfg_debug){
   inputfile->cd();
   h_pedval->Write();
   h_ped1->Write();
   h_ped2->Write();
 
   TCanvas *c_pedval=new TCanvas(nameval,nameval,640,480);
   c_pedval->cd();
   h_pedval->Draw();
   TString c_valname=nameval+"."+cfg_fileextension;   
   c_pedval->Print(c_valname);

   TCanvas *c_ped1=new TCanvas(name1,name1,640,480);
   c_ped1->cd();
   h_ped1->Draw();
   TString c_name1=name1+"."+cfg_fileextension;   
   c_ped1->Print(c_name1);

   TCanvas *c_ped2=new TCanvas(name2,name2,640,480);
   c_ped2->cd();
   h_ped2->Draw();
   TString c_name2=name2+"."+cfg_fileextension;   
   c_ped2->Print(c_name2);
 }

   TString mapval_name=mapval+"."+cfg_fileextension;
   tkmp_pedval->save(true,-100.,100.,mapval_name.Data());
   TString map1_name=map1+"."+cfg_fileextension;
   tkmp_ped1->save(true,25000.,125000.,map1_name.Data());
   TString map2_name=map2+"."+cfg_fileextension;
   tkmp_ped2->save(true,25000.,125000.,map2_name.Data());


}
///////////////////////////////////////////////////
void SiStripO2OValidation::ValidateThreshold(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidateThreshold]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

   TString nameh1="ThresholdH_"+cmssw1;
   TString nameh2="ThresholdH_"+cmssw2;
   TString namevalh="ThresholdH_Difference_"+cmssw1+"_"+cmssw2;
   TString mapvalh="SiStripO2OThresholdHValidationMap_"+cmssw1+"_"+cmssw2;
   TString maph1="SiStripO2OThresholdHMap_"+cmssw1;
   TString maph2="SiStripO2OThresholdHMap_"+cmssw2;

   TString namel1="ThresholdL_"+cmssw1;
   TString namel2="ThresholdL_"+cmssw2;
   TString namevall="ThresholdL_Difference_"+cmssw1+"_"+cmssw2;
   TString mapvall="SiStripO2OThresholdLValidationMap_"+cmssw1+"_"+cmssw2;
   TString mapl1="SiStripO2OThresholdLMap_"+cmssw1;
   TString mapl2="SiStripO2OThresholdLMap_"+cmssw2;

   TString namec1="ThresholdC_"+cmssw1;
   TString namec2="ThresholdC_"+cmssw2;
   TString namevalc="ThresholdC_Difference_"+cmssw1+"_"+cmssw2;
   TString mapvalc="SiStripO2OThresholdCValidationMap_"+cmssw1+"_"+cmssw2;
   TString mapc1="SiStripO2OThresholdCMap_"+cmssw1;
   TString mapc2="SiStripO2OThresholdCMap_"+cmssw2;

   TH1F  *h_thrvalh=new TH1F(namevalh, namevalh,100,-1000.,1000.);
   TH1F  *h_thrh1=new TH1F(nameh1, nameh1,100,-1000.,1000.);
   TH1F  *h_thrh2=new TH1F(nameh2, nameh2,100,-1000.,1000.);
   TH1F  *h_thrvall=new TH1F(namevall, namevall, 100,0.,25.);
   TH1F  *h_thrl1=new TH1F(namel1, namel1, 100,0.,25.);
   TH1F  *h_thrl2=new TH1F(namel2, namel2, 100,0.,25.);
   TH1F  *h_thrvalc=new TH1F(namevalc, namevalc, 100,0.,25.);
   TH1F  *h_thrc1=new TH1F(namec1, namec1, 100,0.,25.);
   TH1F  *h_thrc2=new TH1F(namec2, namec2, 100,0.,25.);
   TrackerMap *tkmp_thrvalh= new TrackerMap(mapvalh.Data());
   TrackerMap *tkmp_thrh1= new TrackerMap(maph1.Data());
   TrackerMap *tkmp_thrh2= new TrackerMap(maph2.Data());
   TrackerMap *tkmp_thrvall= new TrackerMap(mapvall.Data());
   TrackerMap *tkmp_thrl1= new TrackerMap(mapl1.Data());
   TrackerMap *tkmp_thrl2= new TrackerMap(mapl2.Data());
   TrackerMap *tkmp_thrvalc= new TrackerMap(mapvalc.Data());
   TrackerMap *tkmp_thrc1= new TrackerMap(mapc1.Data());
   TrackerMap *tkmp_thrc2= new TrackerMap(mapc2.Data());

  

  Int_t tree1_detid;
   float tree1_inththreshold;
   float tree1_intlthreshold;
   float tree1_intcthreshold;
   TBranch        *b_tree1_detid;   //!
   TBranch *b_tree1_inththreshold;
   TBranch *b_tree1_intlthreshold;
   TBranch *b_tree1_intcthreshold;
   val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
   val_tree[treenr1]->SetBranchAddress("INTHTHRESHOLD",&tree1_inththreshold, &b_tree1_inththreshold);
   val_tree[treenr1]->SetBranchAddress("INTLTHRESHOLD",&tree1_intlthreshold, &b_tree1_intlthreshold);
   val_tree[treenr1]->SetBranchAddress("INTCTHRESHOLD",&tree1_intcthreshold, &b_tree1_intcthreshold);

   Int_t tree2_detid;
   float tree2_inththreshold;
   float tree2_intlthreshold;
   float tree2_intcthreshold;
   TBranch        *b_tree2_detid;   //!
   TBranch *b_tree2_inththreshold;
   TBranch *b_tree2_intlthreshold;
   TBranch *b_tree2_intcthreshold;
   val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
   val_tree[treenr2]->SetBranchAddress("INTHTHRESHOLD",&tree2_inththreshold, &b_tree2_inththreshold);
   val_tree[treenr2]->SetBranchAddress("INTLTHRESHOLD",&tree2_intlthreshold, &b_tree2_intlthreshold);
   val_tree[treenr2]->SetBranchAddress("INTCTHRESHOLD",&tree2_intcthreshold, &b_tree2_intcthreshold);

   
 
   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
   Long64_t nbytes2 = 0, nb2 = 0;
   bool firstloop=true;
   Long64_t lastmatch=0;
   int nrmatching=0;
   int nrnonmatching=0;

   for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
        h_thrh1->Fill(tree1_inththreshold);
	h_thrl1->Fill(tree1_intlthreshold);
	h_thrc1->Fill(tree1_intcthreshold);
	tkmp_thrh1->fill(tree1_detid,tree1_inththreshold);
	tkmp_thrl1->fill(tree1_detid,tree1_intlthreshold);
	tkmp_thrc1->fill(tree1_detid,tree1_intcthreshold);
				     
       for (Long64_t jentry2=lastmatch; jentry2 < nentries2; jentry2++) {
       nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
      
       if(firstloop){h_thrh2->Fill(tree2_inththreshold);
                     h_thrl2->Fill(tree2_intlthreshold);
		     h_thrc2->Fill(tree2_intcthreshold);
		     tkmp_thrh2->fill(tree2_detid,tree2_inththreshold);
		     tkmp_thrl2->fill(tree2_detid,tree2_intlthreshold);
		     tkmp_thrc2->fill(tree2_detid,tree2_intcthreshold);
                    }
       if(tree1_detid > tree2_detid ){
         h_thrvalh->Fill(tree1_inththreshold); 
         h_thrvall->Fill(tree1_intlthreshold); 
         h_thrvalc->Fill(tree1_intcthreshold); 
         tkmp_thrvalh->fill(tree1_detid,tree1_inththreshold);
	 tkmp_thrvall->fill(tree1_detid,tree1_intlthreshold);
	 tkmp_thrvalc->fill(tree1_detid,tree1_intcthreshold);
	 std::cout << " No Match found for Tree2 DetId: " << tree2_detid <<" "<< tree1_detid << " " <<lastmatch <<std::endl;
	 continue;
       }
       if(tree1_detid == tree2_detid){h_thrvalh->Fill(tree1_inththreshold - tree2_inththreshold);
                                      h_thrvall->Fill(tree1_intlthreshold - tree2_intlthreshold);
                                      h_thrvalc->Fill(tree1_intcthreshold - tree2_intcthreshold);
				      tkmp_thrvalh->fill(tree1_detid,tree1_inththreshold - tree2_inththreshold);
				      tkmp_thrvall->fill(tree1_detid,tree1_intlthreshold - tree2_intlthreshold);
				      tkmp_thrvalc->fill(tree1_detid,tree1_intcthreshold - tree2_intcthreshold);
				        tkmp_thrh2->fill(tree2_detid,tree2_inththreshold);
					tkmp_thrl2->fill(tree2_detid,tree2_intlthreshold);
					tkmp_thrc2->fill(tree2_detid,tree2_intcthreshold);

				      lastmatch=jentry2+1;
				       if(tree2_inththreshold != tree1_inththreshold ||
					  tree2_intlthreshold != tree1_intlthreshold ||
					  tree2_intcthreshold != tree1_intcthreshold){
					 std::cout << "[SiStripO2OValidation::" << __func__ 
                                                   << "] Diff in IntThreshold for DetId: " << tree1_detid
                                                   << " tree1: " <<   tree1_inththreshold 
                                                   << " tree2: " << tree2_inththreshold << " || " 
                                                   << " tree1: " <<   tree1_intlthreshold  
                                                   << " tree2: " << tree2_intlthreshold << " || "
                                                   << " tree1: " <<   tree1_intcthreshold 
                                                   << " tree2: " << tree2_intcthreshold << " || " 
                                                   << std::endl;
					nrnonmatching++;
					}else{
					nrmatching++;
				       } 
				       break;
                }
       }
       firstloop=false;
   }
   
     std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree1: " << nentries1 << std::endl;
     std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree2: " << nentries2 << std::endl;
     std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
     std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;

     if(cfg_debug){ 
       inputfile->cd();
       h_thrvalh->Write();
       h_thrvall->Write();
       h_thrvalc->Write();
       h_thrh1->Write();
       h_thrl1->Write();
       h_thrc1->Write();
       h_thrh2->Write();
       h_thrl2->Write();
       h_thrc2->Write();
 
       TCanvas *c_thrvalh=new TCanvas(namevalh,namevalh,640,480);
       c_thrvalh->cd();
       h_thrvalh->Draw();
       TString c_valnameh=namevalh+"."+cfg_fileextension;   
       c_thrvalh->Print(c_valnameh);

       TCanvas *c_thrvall=new TCanvas(namevall,namevall,640,480);
       c_thrvall->cd();
       h_thrvall->Draw();
       TString c_valnamel=namevall+"."+cfg_fileextension;   
       c_thrvall->Print(c_valnamel);

       TCanvas *c_thrvalc=new TCanvas(namevalc,namevalc,640,480);
       c_thrvalc->cd();
       h_thrvalc->Draw();
       TString c_valnamec=namevalc+"."+cfg_fileextension;   
       c_thrvalc->Print(c_valnamec);

       TCanvas *c_thrh1=new TCanvas(nameh1,nameh1,640,480);
       c_thrh1->cd();
       h_thrh1->Draw();
       TString c_nameh1=nameh1+"."+cfg_fileextension;   
       c_thrh1->Print(c_nameh1);

       TCanvas *c_thrl1=new TCanvas(namel1,namel1,640,480);
       c_thrl1->cd();
       h_thrl1->Draw();
       TString c_namel1=namel1+"."+cfg_fileextension;   
       c_thrl1->Print(c_namel1);

       TCanvas *c_thrc1=new TCanvas(namec1,namec1,640,480);
       c_thrc1->cd();
       h_thrc1->Draw();
       TString c_namec1=namec1+"."+cfg_fileextension;   
       c_thrc1->Print(c_namec1);
 
       TCanvas *c_thrh2=new TCanvas(nameh2,nameh2,640,480);
       c_thrh2->cd();
       h_thrh2->Draw();
       TString c_nameh2=nameh2+"."+cfg_fileextension;   
       c_thrh2->Print(c_nameh2);

       TCanvas *c_thrl2=new TCanvas(namel2,namel2,640,480);
       c_thrl2->cd();
       h_thrl2->Draw();
       TString c_namel2=namel2+"."+cfg_fileextension;   
       c_thrl2->Print(c_namel2);

       TCanvas *c_thrc2=new TCanvas(namec2,namec2,640,480);
       c_thrc2->cd();
       h_thrc2->Draw();
       TString c_namec2=namec2+"."+cfg_fileextension;   
       c_thrc2->Print(c_namec2);
     }

   TString mapvalh_name=mapvalh+"."+cfg_fileextension;
   tkmp_thrvalh->save(true,-100.,100.,mapvalh_name.Data());
   TString mapvall_name=mapvall+"."+cfg_fileextension;
   tkmp_thrvall->save(true,-100.,100.,mapvall_name.Data());
   TString mapvalc_name=mapvalc+"."+cfg_fileextension;
   tkmp_thrvalc->save(true,-100.,100.,mapvalc_name.Data());

   TString maph1_name=maph1+"."+cfg_fileextension;
   tkmp_thrh1->save(true,1000.,5000.,maph1_name.Data());
   TString mapl1_name=mapl1+"."+cfg_fileextension;
   tkmp_thrl1->save(true,1000.,5000.,mapl1_name.Data());
   TString mapc1_name=mapc1+"."+cfg_fileextension;
   tkmp_thrc1->save(true,1000.,5000.,mapc1_name.Data());

   TString maph2_name=maph2+"."+cfg_fileextension;
   tkmp_thrh2->save(true,1000.,5000.,maph2_name.Data());
   TString mapl2_name=mapl2+"."+cfg_fileextension;
   tkmp_thrl2->save(true,1000.,5000.,mapl2_name.Data());
   TString mapc2_name=mapc2+"."+cfg_fileextension;
   tkmp_thrc2->save(true,1000.,5000.,mapc2_name.Data());

}

///////////////////////////////////////////////////
void SiStripO2OValidation::ValidateQuality(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidateQuality]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

  TString name1="Quality_"+cmssw1;
  TString name2="Quality_"+cmssw2;
  TString nameval="Quality_Difference_"+cmssw1+"_"+cmssw2;
  TString mapval="SiStripO2OQualityValidationMap_"+cmssw1+"_"+cmssw2;
  TString map1="SiStripO2OQualityMap_"+cmssw1;
  TString map2="SiStripO2OQualityMap_"+cmssw2;

  TH1F  *h_quaval=new TH1F(nameval, nameval,100,-1000.,1000.);
  TH1F  *h_qua1=new TH1F(name1, name1, 100,0.,25.);
  TH1F  *h_qua2=new TH1F(name2, name2, 100,0.,25.);
  TrackerMap *tkmp_quaval= new TrackerMap(mapval.Data());
  TrackerMap *tkmp_qua1= new TrackerMap(map1.Data());
  TrackerMap *tkmp_qua2= new TrackerMap(map2.Data());

   Int_t tree1_detid;
   Int_t tree1_nrbadstrips;
   TBranch        *b_tree1_detid;   //!
   TBranch *b_tree1_nrbadstrips;
   val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
   val_tree[treenr1]->SetBranchAddress("NRBADSTRIPS",&tree1_nrbadstrips, &b_tree1_nrbadstrips);

   Int_t tree2_detid;
   Int_t tree2_nrbadstrips;
   TBranch        *b_tree2_detid;   //!
   TBranch *b_tree2_nrbadstrips;
   val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
   val_tree[treenr2]->SetBranchAddress("NRBADSTRIPS",&tree2_nrbadstrips, &b_tree2_nrbadstrips);
 
 
   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
  
   Long64_t nbytes2 = 0, nb2 = 0;
   int nrmatching=0;
   int nrnonmatching=0;
   bool firstloop=true;

   for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
       h_qua1->Fill(tree1_nrbadstrips);
       tkmp_qua1->fill(tree1_detid,tree1_nrbadstrips);
       for (Long64_t jentry2=0; jentry2 < nentries2; jentry2++) {
	 nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
	 if(firstloop){h_qua2->Fill(tree2_nrbadstrips);
	               tkmp_qua2->fill(tree2_detid,tree2_nrbadstrips);
		      
	              }
	 if(tree1_detid == tree2_detid){h_quaval->Fill(tree1_nrbadstrips - tree2_nrbadstrips);
	                                tkmp_quaval->fill(tree1_detid,tree1_nrbadstrips - tree2_nrbadstrips);
					tkmp_qua2->fill(tree2_detid,tree2_nrbadstrips);
					if(tree1_nrbadstrips == tree2_nrbadstrips) {
					  nrmatching++;
					}else{
					  std::cout << "[SiStripO2OValidation::" << __func__ 
						    << "] Diff in NrBadStrips for DetId: " << tree1_detid
						    << " tree1: " <<   tree1_nrbadstrips
						    << " tree2: " << tree2_nrbadstrips << " || " 
						    << std::endl;
					  nrnonmatching++;
					}
					break;
	 }
       }
       firstloop=false;
   }
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree1: " << nentries1 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree2: " << nentries2 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;

     if(cfg_debug){ 
       inputfile->cd();
       h_quaval->Write();
       h_qua1->Write();
       h_qua2->Write();
 
       TCanvas *c_quaval=new TCanvas(nameval,nameval,640,480);
       c_quaval->cd();
       h_quaval->Draw();
       TString c_valname=nameval+"."+cfg_fileextension;   
       c_quaval->Print(c_valname);

       TCanvas *c_qua1=new TCanvas(name1,name1,640,480);
       c_qua1->cd();
       h_qua1->Draw();
       TString c_name1=name1+"."+cfg_fileextension;   
       c_qua1->Print(c_name1);

       TCanvas *c_qua2=new TCanvas(name2,name2,640,480);
       c_qua2->cd();
       h_qua2->Draw();
       TString c_name2=name2+"."+cfg_fileextension;   
       c_qua2->Print(c_name2);
     }

   TString mapval_name=mapval+"."+cfg_fileextension;
   tkmp_quaval->save(true,-10.,10.,mapval_name.Data());
   TString map1_name=map1+"."+cfg_fileextension;
   tkmp_qua1->save(true,0.,10.,map1_name.Data());
   TString map2_name=map2+"."+cfg_fileextension;
   tkmp_qua2->save(true,0.,10.,map2_name.Data());
  
}
///////////////////////////////////////////////////
void SiStripO2OValidation::ValidateTiming(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidateTiming]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

  TString name1="Timing_"+cmssw1;
  TString name2="Timing_"+cmssw2;
  TString nameval="Timing_Difference_"+cmssw1+"_"+cmssw2;
  TString mapval="SiStripO2OTimingValidationMap_"+cmssw1+"_"+cmssw2;
  TString map1="SiStripO2OTimingMap_"+cmssw1;
  TString map2="SiStripO2OTimingMap_"+cmssw2;

  TH1F  *h_timval=new TH1F(nameval, nameval,100,-1000.,1000.);
  TH1F  *h_tim1=new TH1F(name1, name1, 100,0.,25.);
  TH1F  *h_tim2=new TH1F(name2, name2, 100,0.,25.);
  TrackerMap *tkmp_timval= new TrackerMap(mapval.Data());
  TrackerMap *tkmp_tim1= new TrackerMap(map1.Data());
  TrackerMap *tkmp_tim2= new TrackerMap(map2.Data());
     
   Int_t tree1_detid;
   Int_t tree1_apvnr;
   float tree1_tickheight;
   TBranch        *b_tree1_detid;   //!
   TBranch *b_tree1_apvnr;
   TBranch *b_tree1_tickheight;
   val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
   val_tree[treenr1]->SetBranchAddress("APVNR",&tree1_apvnr, &b_tree1_apvnr);
   val_tree[treenr1]->SetBranchAddress("TICKHEIGHT",&tree1_tickheight, &b_tree1_tickheight);

   Int_t tree2_detid;
   Int_t tree2_apvnr;
   float tree2_tickheight;
   TBranch        *b_tree2_detid;   //!
   TBranch *b_tree2_apvnr;
   TBranch *b_tree2_tickheight;
   val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
   val_tree[treenr2]->SetBranchAddress("APVNR",&tree2_apvnr, &b_tree2_apvnr);
   val_tree[treenr2]->SetBranchAddress("TICKHEIGHT",&tree2_tickheight, &b_tree2_tickheight);
 
   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
   Long64_t nbytes2 = 0, nb2 = 0;
   bool firstloop=true;
   Long64_t lastmatch=0;
   int nrmatching=0;
     int nrnonmatching=0;

   for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
       h_tim1->Fill(tree1_tickheight); 
       tkmp_tim1->fill(tree1_detid,tree1_tickheight);

       for (Long64_t jentry2=lastmatch; jentry2 < nentries2; jentry2++) {
       nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
       if(firstloop){h_tim2->Fill(tree2_tickheight); 
	             tkmp_tim2->fill(tree1_detid,tree2_tickheight);
                    }
       if(tree1_detid > tree2_detid ){
         h_timval->Fill(tree1_tickheight); 
	 tkmp_timval->fill(tree1_detid,tree1_tickheight); 
	 nrnonmatching++; 
       }
       if(tree1_detid == tree2_detid &&
          (tree2_apvnr>6))              {std::cout   << "tree1_detid "<<tree1_detid 
						     << " tree1_apvnr " << tree1_apvnr
						     << " tree2_detid "<<tree2_detid
						     << " tree2_apvnr " << tree2_apvnr
						     <<  std::endl;}
       if(tree1_detid == tree2_detid &&
          tree1_apvnr == tree2_apvnr)  {h_timval->Fill(tree1_tickheight - tree2_tickheight);
	                                tkmp_timval->fill(tree1_detid,tree1_tickheight - tree2_tickheight);
					lastmatch=jentry2+1;
					if(tree2_tickheight != tree1_tickheight){
					  std::cout << "[SiStripO2OValidation::" << __func__ 
						    << "] Diff in TickHeight for DetId: " << tree1_detid 
						    << " tree1: " << tree1_tickheight  
						    << " tree2: " << tree2_tickheight
						    << std::endl; 
					  nrnonmatching++;
					}else{
					  nrmatching++;
					}
					break;
       }
       firstloop=false;
       }
   }

       std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree1: " << nentries1 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree2: " << nentries2 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;

 if(cfg_debug){ 
   inputfile->cd();
   h_timval->Write();
   h_tim1->Write();
   h_tim2->Write();
 
    TCanvas *c_timval=new TCanvas(nameval,nameval,640,480);
   c_timval->cd();
   h_timval->Draw();
   TString c_valname=nameval+"."+cfg_fileextension;   
   c_timval->Print(c_valname);

   TCanvas *c_tim1=new TCanvas(name1,name1,640,480);
   c_tim1->cd();
   h_tim1->Draw();
   TString c_name1=name1+"."+cfg_fileextension;   
   c_tim1->Print(c_name1);

   TCanvas *c_tim2=new TCanvas(name2,name2,640,480);
   c_tim2->cd();
   h_tim2->Draw();
   TString c_name2=name2+"."+cfg_fileextension;   
   c_tim2->Print(c_name2);
 }

   TString mapval_name=mapval+"."+cfg_fileextension;
   tkmp_timval->save(true,-0.2,0.2,mapval_name.Data());
   TString map1_name=map1+"."+cfg_fileextension;
   tkmp_tim1->save(true,0.,8.,map1_name.Data());
   TString map2_name=map2+"."+cfg_fileextension;
   tkmp_tim2->save(true,0.,8.,map2_name.Data());

    
}
///////////////////////////////////////////////////
// since the latency object just stores 1 value for all detids
// that share this one value, no plots are made, they can easily
// be implented by uncommenting the histogramms an TKmaps
void SiStripO2OValidation::ValidateLatency(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidateLatency]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

  TString namelat1="Latency_"+cmssw1;
  TString namelat2="Latency_"+cmssw2;
  TString namelatval="Latency_Difference_"+cmssw1+"_"+cmssw2;
  TString maplatval="SiStripO2OLatencyValidationMap_"+cmssw1+"_"+cmssw2;
  TString maplat1="SiStripO2OLatencyMap_"+cmssw1;
  TString maplat2="SiStripO2OLatencyMap_"+cmssw2;
  
//   TH1F  *h_latval=new TH1F(namelatval, namelatval,100,-1000.,1000.);
//   TH1F  *h_lat1=new TH1F(namelat1, namelat1, 100,0.,25.);
//   TH1F  *h_lat2=new TH1F(namelat2, namelat2, 100,0.,25.);
//   TrackerMap *tkmp_latval= new TrackerMap(maplatval.Data());
//   TrackerMap *tkmp_lat1= new TrackerMap(maplat1.Data());
//   TrackerMap *tkmp_lat2= new TrackerMap(maplat2.Data());

//   TString namemod1="APVMode_"+cmssw1;
//   TString namemod2="APVMode_"+cmssw2;
//   TString namemodval="APVMode_Difference_"+cmssw1+"_"+cmssw2;
//   TString mapmodval="SiStripO2OAPVModeValidationMap_"+cmssw1+"_"+cmssw2;
//   TString mapmod1="SiStripO2OAPVModeMap_"+cmssw1;
//   TString mapmod2="SiStripO2OAPVModeMap_"+cmssw2;

//   TH1F  *h_modval=new TH1F(namemodval, namemodval,100,-1000.,1000.);
//   TH1F  *h_mod1=new TH1F(namemod1, namemod1, 100,0.,25.);
//   TH1F  *h_mod2=new TH1F(namemod2, namemod2, 100,0.,25.);
//   TrackerMap *tkmp_modval= new TrackerMap(mapmodval.Data());
//   TrackerMap *tkmp_mod1= new TrackerMap(mapmod1.Data());
//   TrackerMap *tkmp_mod2= new TrackerMap(mapmod2.Data());


   Int_t tree1_detid;
   Int_t tree1_apvnr;
   Int_t tree1_latency;
   Int_t tree1_apvmode;
   TBranch        *b_tree1_detid;   //!
   TBranch *b_tree1_apvnr;
   TBranch *b_tree1_latency;
   TBranch *b_tree1_apvmode;
   val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
   val_tree[treenr1]->SetBranchAddress("APVNR",&tree1_apvnr, &b_tree1_apvnr);
   val_tree[treenr1]->SetBranchAddress("APVLATENCY",&tree1_latency, &b_tree1_latency);
   val_tree[treenr1]->SetBranchAddress("APVMODE",&tree1_apvmode, &b_tree1_apvmode);


   Int_t tree2_detid;
   Int_t tree2_apvnr;
   Int_t tree2_latency;
   Int_t tree2_apvmode;
   TBranch        *b_tree2_detid;   //!
   TBranch *b_tree2_apvnr;
   TBranch *b_tree2_latency;
   TBranch *b_tree2_apvmode;
   val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
   val_tree[treenr2]->SetBranchAddress("APVNR",&tree2_apvnr, &b_tree2_apvnr);
   val_tree[treenr2]->SetBranchAddress("APVLATENCY",&tree2_latency, &b_tree2_latency);
   val_tree[treenr2]->SetBranchAddress("APVMODE",&tree2_apvmode, &b_tree2_apvmode);

   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
   Long64_t nbytes2 = 0, nb2 = 0;
   int nrmatching=0;
   int nrnonmatching=0;
   
     for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
      
         for (Long64_t jentry2=0; jentry2 < nentries2; jentry2++) {
	   nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
	   if(tree1_detid != tree2_detid || tree1_latency != tree2_latency || tree1_apvmode != tree2_apvmode){
	     std::cout << " Tree1 Latency Range[" << jentry1 <<"]: Starting with DetId:" << tree1_detid
		       << " Latency: "<< tree1_latency 
		       << " APVMode: " << tree1_apvmode
		       << std::endl;
	
	     std::cout << " Tree2 Latency Range[" << jentry2 <<"]: Starting with DetId:" << tree2_detid
		       << " Latency: "<< tree2_latency 
		       << " APVMode: " << tree2_apvmode
		       << std::endl;
	     nrnonmatching++;
	   }else{
	      if(tree1_detid == tree2_detid && tree1_latency == tree2_latency && tree1_apvmode == tree2_apvmode){
		nrmatching++;
		break;
	      }
	   }
	 }
   }
   std::cout << "SiStripO2OValidation::" << __func__ <<"] Latency ranges Tree1: " << nentries1 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] Latency ranges Tree2: " << nentries2 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;
}

///////////////////////////////////////////////////
// Cabling Histo is filled in the following way:
// 1 if cabling is exact match
// 999 otherwise
// 
//
// TkMap is filled:
// 1 entry is added per APV in the individual CMSSW maps
// The comparing TkMap is only filled when something is not found in the cabling of the other version. 
// If the cabling of both CMSSW releases is identical, the TkMap will be empty (all white modules)
void SiStripO2OValidation::ValidateFedCabling(int treenr1, int treenr2){
  std::cout << "Entering [SiStripO2OValidation::ValidateFedCabling]" << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr1]->GetTitle() << std::endl;
  std::cout << "Tree in use: " <<val_tree[treenr2]->GetTitle() << std::endl;

 TString name1="FedCabling_"+cmssw1;
  TString name2="FedCabling_"+cmssw2;
  TString nameval="FedCabling_Difference_"+cmssw1+"_"+cmssw2;
  TString mapval="SiStripO2OFedCablingValidationMap_"+cmssw1+"_"+cmssw2;
  TString map1="SiStripO2OFedCablingMap_"+cmssw1;
  TString map2="SiStripO2OFedCablingMap_"+cmssw2;

  TH1F  *h_fedval=new TH1F(nameval, nameval,100,-1000.,1000.);
  TH1F  *h_fed1=new TH1F(name1, name1, 100,0.,25.);
  TH1F  *h_fed2=new TH1F(name2, name2, 100,0.,25.);
  TrackerMap *tkmp_fedval= new TrackerMap(mapval.Data());
  TrackerMap *tkmp_fed1= new TrackerMap(map1.Data());
  TrackerMap *tkmp_fed2= new TrackerMap(map2.Data());

     
 //Tree 1
 Int_t tree1_detid;
 int	tree1_fedcr;
 int	tree1_fedsl;
 int	tree1_fedid;
 int   tree1_feunit;
 int   tree1_fechan;
 int   tree1_fedchan;
 int   tree1_feccr;
 int   tree1_fecsl;
 int   tree1_fecring;
 int	tree1_ccuaddr;
 int	tree1_module;
 int   tree1_apv1;
 int	tree1_apv2;
 int	tree1_apvpair;
 int   tree1_nrapvpairs;
 int   tree1_dcuid;
 
 TBranch *b_tree1_detid;   //!
 TBranch *b_tree1_fedcr;
 TBranch *b_tree1_fedsl;
 TBranch *b_tree1_fedid;
 TBranch *b_tree1_feunit;
 TBranch *b_tree1_fechan;
 TBranch *b_tree1_fedchan;
 TBranch *b_tree1_feccr;
 TBranch *b_tree1_fecsl;
 TBranch *b_tree1_fecring;
 TBranch *b_tree1_ccuaddr;
 TBranch *b_tree1_module;
 TBranch *b_tree1_apv1;
 TBranch *b_tree1_apv2;
 TBranch *b_tree1_apvpair;
 TBranch *b_tree1_nrapvpairs;
 TBranch *b_tree1_dcuid;
 
 val_tree[treenr1]->SetBranchAddress("DETID", &tree1_detid, &b_tree1_detid);
 val_tree[treenr1]->SetBranchAddress("DCUID",&tree1_dcuid, &b_tree1_dcuid);
 val_tree[treenr1]->SetBranchAddress("FEDCR",&tree1_fedcr, &b_tree1_fedcr);
 val_tree[treenr1]->SetBranchAddress("FEDSL",&tree1_fedsl, &b_tree1_fedsl);
 val_tree[treenr1]->SetBranchAddress("FEDID",&tree1_fedid, &b_tree1_fedid);
 val_tree[treenr1]->SetBranchAddress("FEUNIT",&tree1_feunit,&b_tree1_feunit);
 val_tree[treenr1]->SetBranchAddress("FECHAN",&tree1_fechan,&b_tree1_fechan);
 val_tree[treenr1]->SetBranchAddress("FEDCHAN",&tree1_fedchan,&b_tree1_fedchan);
 val_tree[treenr1]->SetBranchAddress("FECCR",&tree1_feccr,&b_tree1_feccr);
 val_tree[treenr1]->SetBranchAddress("FECSL",&tree1_fecsl,&b_tree1_fecsl);
 val_tree[treenr1]->SetBranchAddress("FECRING",&tree1_fecring,&b_tree1_fecring);
 val_tree[treenr1]->SetBranchAddress("CCUADDR",&tree1_ccuaddr,&b_tree1_ccuaddr);
 val_tree[treenr1]->SetBranchAddress("MODULE",&tree1_module,&b_tree1_module);
 val_tree[treenr1]->SetBranchAddress("APV1",&tree1_apv1,&b_tree1_apv1);
 val_tree[treenr1]->SetBranchAddress("APV2",&tree1_apv2,&b_tree1_apv2);
 val_tree[treenr1]->SetBranchAddress("APVPAIR",&tree1_apvpair,&b_tree1_apvpair);
 val_tree[treenr1]->SetBranchAddress("NRAPVPAIRS",&tree1_nrapvpairs,&b_tree1_nrapvpairs);

 //Tree 2
 Int_t tree2_detid;
 int	tree2_fedcr;
 int	tree2_fedsl;
 int	tree2_fedid;
 int   tree2_feunit;
 int   tree2_fechan;
 int   tree2_fedchan;
 int   tree2_feccr;
 int   tree2_fecsl;
 int   tree2_fecring;
 int	tree2_ccuaddr;
 int	tree2_module;
 int   tree2_apv1;
 int	tree2_apv2;
 int	tree2_apvpair;
 int   tree2_nrapvpairs;
 int   tree2_dcuid;
 
 TBranch *b_tree2_detid;   //!
 TBranch *b_tree2_fedcr;
 TBranch *b_tree2_fedsl;
 TBranch *b_tree2_fedid;
 TBranch *b_tree2_feunit;
 TBranch *b_tree2_fechan;
 TBranch *b_tree2_fedchan;
 TBranch *b_tree2_feccr;
 TBranch *b_tree2_fecsl;
 TBranch *b_tree2_fecring;
 TBranch *b_tree2_ccuaddr;
 TBranch *b_tree2_module;
 TBranch *b_tree2_apv1;
 TBranch *b_tree2_apv2;
 TBranch *b_tree2_apvpair;
 TBranch *b_tree2_nrapvpairs;
 TBranch *b_tree2_dcuid;
 
 val_tree[treenr2]->SetBranchAddress("DETID", &tree2_detid, &b_tree2_detid);
 val_tree[treenr2]->SetBranchAddress("DCUID",&tree2_dcuid, &b_tree2_dcuid);
 val_tree[treenr2]->SetBranchAddress("FEDCR",&tree2_fedcr, &b_tree2_fedcr);
 val_tree[treenr2]->SetBranchAddress("FEDSL",&tree2_fedsl, &b_tree2_fedsl);
 val_tree[treenr2]->SetBranchAddress("FEDID",&tree2_fedid, &b_tree2_fedid);
 val_tree[treenr2]->SetBranchAddress("FEUNIT",&tree2_feunit,&b_tree2_feunit);
 val_tree[treenr2]->SetBranchAddress("FECHAN",&tree2_fechan,&b_tree2_fechan);
 val_tree[treenr2]->SetBranchAddress("FEDCHAN",&tree2_fedchan,&b_tree2_fedchan);
 val_tree[treenr2]->SetBranchAddress("FECCR",&tree2_feccr,&b_tree2_feccr);
 val_tree[treenr2]->SetBranchAddress("FECSL",&tree2_fecsl,&b_tree2_fecsl);
 val_tree[treenr2]->SetBranchAddress("FECRING",&tree2_fecring,&b_tree2_fecring);
 val_tree[treenr2]->SetBranchAddress("CCUADDR",&tree2_ccuaddr,&b_tree2_ccuaddr);
 val_tree[treenr2]->SetBranchAddress("MODULE",&tree2_module,&b_tree2_module);
 val_tree[treenr2]->SetBranchAddress("APV1",&tree2_apv1,&b_tree2_apv1);
 val_tree[treenr2]->SetBranchAddress("APV2",&tree2_apv2,&b_tree2_apv2);
 val_tree[treenr2]->SetBranchAddress("APVPAIR",&tree2_apvpair,&b_tree2_apvpair);
 val_tree[treenr2]->SetBranchAddress("NRAPVPAIRS",&tree2_nrapvpairs,&b_tree2_nrapvpairs);

 
   Long64_t nentries1 = val_tree[treenr1]->GetEntriesFast();
   Long64_t nbytes1 = 0, nb1 = 0;
   Long64_t nentries2 = val_tree[treenr2]->GetEntriesFast();
   Long64_t nbytes2 = 0, nb2 = 0;
   
  
   int nrmatching=0;
    int nrnonmatching=0;
   bool firstloop=true;
   Long64_t lastmatch=0;
   for (Long64_t jentry1=0; jentry1 < nentries1; jentry1++) {
       nb1 = val_tree[treenr1]->GetEntry(jentry1); nbytes1 +=nb1;
       h_fed1->Fill(1.);
       tkmp_fed1->fill(tree1_detid,1.);
       for (Long64_t jentry2=lastmatch; jentry2 < nentries2; jentry2++) {
	 nb2 = val_tree[treenr2]->GetEntry(jentry2); nbytes2 +=nb2;
	 if(    tree1_detid > tree2_detid ){
	   h_fedval->Fill(999); //999 if cabling is not identical
	   tkmp_fedval->fill(tree1_detid,1.); 
	   nrnonmatching++;
	   break;
	 }
	if(    tree1_detid == tree2_detid
           && tree1_dcuid == tree2_dcuid
           && tree1_fedcr == tree2_fedcr
           && tree1_fedsl == tree2_fedsl
           && tree1_fedid == tree2_fedid
           && tree1_feunit == tree2_feunit
           && tree1_fechan == tree2_fechan
           && tree1_fedchan == tree2_fedchan
           && tree1_feccr == tree2_feccr
           && tree1_fecsl == tree2_fecsl
           && tree1_fecring == tree2_fecring
           && tree1_ccuaddr == tree2_ccuaddr
           && tree1_module == tree2_module
           && tree1_apv1 == tree2_apv1
           && tree1_apv2 == tree2_apv2
           && tree1_apvpair == tree2_apvpair
	   && tree1_nrapvpairs == tree2_nrapvpairs){h_fedval->Fill(1); //1 if cabling is identical
	                                            lastmatch=jentry2+1;
						    h_fed2->Fill(1.);
						    tkmp_fed2->fill(tree2_detid,1.);
						    nrmatching++;
						    break;
	                                           }
	if(tree1_detid !=tree2_detid){
	  h_fedval->Fill(999); //1 if cabling is identical
	  tkmp_fedval->fill(tree1_detid,1.);   
	  std::cout << "Cabling Information for DetId: " << tree1_detid << " not in second tree!" << std::endl;
	  nrnonmatching++;
	}
       }
       firstloop=false;
   }
   
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree1: " << nentries1 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrEntries Tree2: " << nentries2 << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrMatching: " << nrmatching << std::endl;
   std::cout << "SiStripO2OValidation::" << __func__ <<"] NrNonMatching: " << nrnonmatching << std::endl;

   if(cfg_debug){
   inputfile->cd();
   h_fedval->Write();
   h_fed1->Write();
   h_fed2->Write();

   
     TCanvas *c_fedval=new TCanvas(nameval,nameval,640,480);
   c_fedval->cd();
   h_fedval->Draw();
   TString c_valname=nameval+"."+cfg_fileextension;   
   c_fedval->Print(c_valname);

   TCanvas *c_fed1=new TCanvas(name1,name1,640,480);
   c_fed1->cd();
   h_fed1->Draw();
   TString c_name1=name1+"."+cfg_fileextension;   
   c_fed1->Print(c_name1);

   TCanvas *c_fed2=new TCanvas(name2,name2,640,480);
   c_fed2->cd();
   h_fed2->Draw();
   TString c_name2=name2+"."+cfg_fileextension;   
   c_fed2->Print(c_name2);

   }

   TString mapval_name=mapval+"."+cfg_fileextension;
   tkmp_fedval->save(true,0.,10.,mapval_name.Data());
   TString map1_name=map1+"."+cfg_fileextension;
   tkmp_fed1->save(true,0.,10.,map1_name.Data());
   TString map2_name=map2+"."+cfg_fileextension;
   tkmp_fed2->save(true,0.,10.,map2_name.Data());

}


// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripO2OValidation::endJob() {

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripO2OValidation);
