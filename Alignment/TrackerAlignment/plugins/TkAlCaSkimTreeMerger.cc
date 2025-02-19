
#ifndef TrackerAlignment_TkAlCaSkimTreeMerger_H
#define TrackerAlignment_TkAlCaSkimTreeMerger_H

#include <Riostream.h>
#include <string>
#include <fstream>
#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "TFile.h"
#include "TString.h"
#include "TChain.h"
#include "TStopwatch.h"

class TkAlCaSkimTreeMerger : public edm::EDAnalyzer{

 public:
  TkAlCaSkimTreeMerger(const edm::ParameterSet &iConfig);
  ~TkAlCaSkimTreeMerger();
  void beginJob();
  void endJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  TTree *out_;//TTree containing the merged result
  TTree *firsttree_;//first tree of the list; this gives the structure to all the others 
  TChain *ch_;//chain containing all the tree you want to merge
  std::string filelist_;//text file containing the list of input files
  std::string firstfilename_;
  std::string treename_;//name of the tree you want to merge (contained in the file)
  std::string outfilename_;//name of the file where you want to save the output
 
  //Hit Population
  typedef map<uint32_t,uint32_t>DetHitMap;
  DetHitMap hitmap_;
  DetHitMap overlapmap_;
  int maxhits_;//above this number, the hit population is prescaled. Configurable for each subdet 
  edm::ParameterSet maxhitsSet_;
  int maxPXBhits_, maxPXFhits_, maxTIBhits_, maxTIDhits_, maxTOBhits_, maxTEChits_;
 

  TStopwatch myclock;

};




#endif


TkAlCaSkimTreeMerger::TkAlCaSkimTreeMerger(const edm::ParameterSet &iConfig) :
    filelist_(iConfig.getParameter<string>("FileList")), 
    treename_(iConfig.getParameter<string>("TreeName")),
    outfilename_(iConfig.getParameter<string>("OutputFile")),
    // maxhits_(iConfig.getParameter<vint>("NhitsMaxLimit"))
    maxhits_(iConfig.getParameter<int32_t>("NhitsMaxLimit")),
    maxhitsSet_(iConfig.getParameter<edm::ParameterSet>("NhitsMaxSet"))
{
  maxPXBhits_=maxhitsSet_.getParameter<int32_t>("PXBmaxhits");
  maxPXFhits_=maxhitsSet_.getParameter<int32_t>("PXFmaxhits");
  maxTIBhits_=maxhitsSet_.getParameter<int32_t>("TIBmaxhits");
  maxTIDhits_=maxhitsSet_.getParameter<int32_t>("TIDmaxhits");
  maxTOBhits_=maxhitsSet_.getParameter<int32_t>("TOBmaxhits");
  maxTEChits_=maxhitsSet_.getParameter<int32_t>("TECmaxhits");
  //anything you want to do for initializing
  cout<<"\n\n*** MAX N HITS = "<<maxhits_<<endl<<endl;
  out_=0;
  firsttree_=0;
  ch_=0;
}


TkAlCaSkimTreeMerger::~TkAlCaSkimTreeMerger(){
  //default destructor
  // delete out_;
  // delete firsttree_;

  delete ch_;
  cout<<"finished."<<endl;
}

// ------------ method called before analyzing the first event  ------------
void TkAlCaSkimTreeMerger::beginJob(){
 
  myclock.Start();

  //prepare the chain
  ch_=new TChain(treename_.c_str());
  cout<<"The chain contains "<<ch_->GetNtrees()<<" trees"<<endl;
  
  //load the trees into the chain
  ifstream flist(filelist_.c_str(),ios::in);
  std::string filename;
  std::string firstfilename;
  bool first=true;
  while(!flist.eof()){
    filename="";
    flist>>filename;
    if(filename.empty())continue;
    //cout<<"Adding "<<filename<<endl;
    ch_->Add(filename.c_str());
    if(first){
      firstfilename_=filename;
      first=false;
    }
   
  }
  cout<<"Now the chain contains "<<ch_->GetNtrees()<<" trees ("<<ch_->GetEntries()<<" entries)"<<endl;

 
  unsigned int id_ch=0;
  uint32_t nhits_ch=0,noverlaps_ch=0;
  ch_->SetBranchAddress("DetId",    &id_ch);
  ch_->SetBranchAddress("Nhits",    &nhits_ch);
  ch_->SetBranchAddress("Noverlaps",&noverlaps_ch);

  ch_->SetBranchStatus("SubDet",0);
  ch_->SetBranchStatus("Layer",0);
  ch_->SetBranchStatus("is2D",0);
  ch_->SetBranchStatus("isStereo",0);
  ch_->SetBranchStatus("posX",0);
  ch_->SetBranchStatus("posY",0);
  ch_->SetBranchStatus("posZ",0);
  ch_->SetBranchStatus("posR",0);
  ch_->SetBranchStatus("posEta",0);
  ch_->SetBranchStatus("posPhi",0);//now only id, nhits and noverlaps are on...


  int totnhits(0),totnoverlaps(0);


  //look if you find this detid in the map
  DetHitMap::iterator mapiter;
  DetHitMap::iterator overlapiter;
  
  for(int ent=0;ent<ch_->GetEntries();++ent){
  //  for(int ent=0;ent<100;++ent){
    ch_->GetEntry(ent);
    totnhits+=nhits_ch;
    totnoverlaps+=noverlaps_ch;

    mapiter= hitmap_.find(id_ch);
    if(mapiter!=hitmap_.end() ){//present, increase its value
      hitmap_[id_ch]=hitmap_[id_ch]+nhits_ch;
    }
    else{//not present, let's add this key to the map with value=1
      hitmap_.insert(pair<uint32_t, uint32_t>(id_ch, nhits_ch));
    }
    //do the same for overlaps
    overlapiter= overlapmap_.find(id_ch);
    if(overlapiter!=overlapmap_.end() ){
      overlapmap_[id_ch]=overlapmap_[id_ch]+noverlaps_ch;
    }
    else{
      overlapmap_.insert(pair<uint32_t, uint32_t>(id_ch, noverlaps_ch));
    }
    
  }//end loop on ent - entries in the chain


  cout<<"Nhits in the chain: "<<totnhits<<endl;
  cout<<"NOverlaps in the chain: "<<totnoverlaps<<endl;


  myclock.Stop();
  cout<<"Finished beginJob after "<<myclock.RealTime()<<" s (real time) / "<<myclock.CpuTime()<<" s (cpu time)"<<endl;
  myclock.Continue();
}//end beginJob


// ------------ method called to for each event  ------------
void TkAlCaSkimTreeMerger::analyze(const edm::Event&, const edm::EventSetup&){
    // cout<<firsttree_->GetEntries()<<endl;
}//end analyze

// ------------ method called after having analyzed all the events  ------------
void TkAlCaSkimTreeMerger::endJob(){


  //address variables in the first tree and in the chain
  TFile *firstfile=new TFile(firstfilename_.c_str(),"READ");
  firsttree_=(TTree*)firstfile->Get(treename_.c_str());
  cout<<"the first tree has "<<firsttree_->GetEntries() <<" entries"<<endl; 
  unsigned int id=0;
  uint32_t nhits=0,noverlaps=0;
  float posX(-99999.0),posY(-77777.0),posZ(-88888.0);
  float posEta(-6666.0),posPhi(-5555.0),posR(-4444.0);
  int subdet=0;
  unsigned int layer=0; 
  // bool is2D=false,isStereo=false;
  firsttree_->SetBranchAddress("DetId",    &id);
  firsttree_->SetBranchAddress("Nhits",    &nhits);
  firsttree_->SetBranchAddress("Noverlaps",&noverlaps);
  firsttree_->SetBranchAddress("SubDet",   &subdet);
  firsttree_->SetBranchAddress("Layer",    &layer);
  //  firsttree_->SetBranchAddress("is2D" ,    &is2D);
  // firsttree_->SetBranchAddress("isStereo", &isStereo);
  firsttree_->SetBranchAddress("posX",     &posX);
  firsttree_->SetBranchAddress("posY",     &posY);
  firsttree_->SetBranchAddress("posZ",     &posZ);
  firsttree_->SetBranchAddress("posR",     &posR);
  firsttree_->SetBranchAddress("posEta",   &posEta);
  firsttree_->SetBranchAddress("posPhi",   &posPhi);


  //create and book the output
 
 
  TFile *outfile=new TFile(outfilename_.c_str(),"RECREATE");
  out_=new TTree(treename_.c_str(),"AlignmentHitMapsTOTAL");
  unsigned int id_out=0;
  uint32_t nhits_out=0,noverlaps_out=0;
  float posX_out(-99999.0),posY_out(-77777.0),posZ_out(-88888.0);
  float posEta_out(-6666.0),posPhi_out(-5555.0),posR_out(-4444.0);
  int subdet_out=0;
  unsigned int layer_out=0; 
  bool is2D_out=false,isStereo_out=false;
  float prescfact_out=1.0;
  float prescfact_overlap_out=1.0;

  out_->Branch("DetId",    &id_out ,      "DetId/i");
  out_->Branch("Nhits",    &nhits_out ,   "Nhits/i");
  out_->Branch("Noverlaps",&noverlaps_out,"Noverlaps/i");
  out_->Branch("SubDet",   &subdet_out,   "SubDet/I");
  out_->Branch("Layer",    &layer_out,    "Layer/i");
  out_->Branch("is2D" ,    &is2D_out,     "is2D/B");
  out_->Branch("isStereo", &isStereo_out, "isStereo/B");
  out_->Branch("posX",     &posX_out,     "posX/F");
  out_->Branch("posY",     &posY_out,     "posY/F");
  out_->Branch("posZ",     &posZ_out,     "posZ/F");
  out_->Branch("posR",     &posR_out,     "posR/F");
  out_->Branch("posEta",   &posEta_out,   "posEta/F");
  out_->Branch("posPhi",   &posPhi_out,   "posPhi/F");
  out_->Branch("PrescaleFactor",&prescfact_out,"PrescaleFact/F");  
  out_->Branch("PrescaleFactorOverlap",&prescfact_overlap_out,"PrescaleFactOverlap/F"); 


  //look if you find this detid in the map
  DetHitMap::iterator mapiter;
  
   for(int mod=0;mod<firsttree_->GetEntries();mod++){
     //for(int mod=0;mod<100;++mod){
    //   nhits_out=0;
    // noverlaps_out=0;
 
    firsttree_->GetEntry(mod);
    nhits_out=hitmap_[id];
    noverlaps_out=overlapmap_[id];
    // if(mod<25)cout<<"Nhits 1st tree: "<<nhits<<"\tTotal nhits chain: "<<nhits_out<<endl;
    id_out=id;
    subdet_out=subdet;
    layer_out=layer;
    posX_out=posX;
    posY_out=posY;
    posZ_out=posZ;
    posR_out=posR;
    posEta_out=posEta;
    posPhi_out=posPhi;

    //calculate prescaling factor
    int subdetmax=-1;
    if(subdet_out==1)subdetmax=maxPXBhits_;
    else if(subdet_out==2)subdetmax=maxPXFhits_;
    else if(subdet_out==3)subdetmax=maxTIBhits_;
    else if(subdet_out==4)subdetmax=maxTIDhits_;
    else if(subdet_out==5)subdetmax=maxTOBhits_;
    else if(subdet_out==6)subdetmax=maxTEChits_;
    else subdetmax=-9999;

    if(maxhits_>-1){
      if(int(nhits_out)>maxhits_){
	prescfact_out=float(maxhits_)/float(nhits_out);
      }
      if(int(noverlaps_out)>maxhits_){
	prescfact_overlap_out=float(maxhits_)/float(noverlaps_out);
      }
    }
    else if(subdetmax>0){//calculate different prescaling factors for each subdet
      if(int(nhits_out)>subdetmax){
	prescfact_out=float(subdetmax/nhits_out);
      }
     if(int(noverlaps_out)>subdetmax){
	prescfact_overlap_out=float(subdetmax)/float(noverlaps_out);
      }
    }
    else{
      prescfact_out=1.0;
      prescfact_overlap_out=1.0;
    }
    out_->Fill();

  }//end loop on mod - first tree modules


  myclock.Stop();
  cout<<"Finished endJob after "<<myclock.RealTime()<<" s (real time) / "<<myclock.CpuTime()<<" s (cpu time)"<<endl;
  cout<<"Ending the tree merging."<<endl;
  out_->Write();
  cout<<"Deleting..."<<flush;
  delete firstfile;
  delete outfile;
 
  
}//end endJob


// ========= MODULE DEF ==============
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TkAlCaSkimTreeMerger);

