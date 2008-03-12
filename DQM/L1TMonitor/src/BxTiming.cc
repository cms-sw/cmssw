#include "DQM/L1TMonitor/interface/BxTiming.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"


BxTiming::BxTiming(const edm::ParameterSet& iConfig) {

  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag",0);
  if(verbose())
    std::cout << "BxTiming::BxTiming()...\n" << std::flush;

  fedRef_ = iConfig.getUntrackedParameter<int>("ReferenceFedId",813);
  fedSource_ = iConfig.getUntrackedParameter<edm::InputTag>
    ("FedSource",edm::InputTag("source"));
  gtSource_ = iConfig.getUntrackedParameter<edm::InputTag>
    ("GtSource",edm::InputTag("gtUnpack"));
  histFile_ = iConfig.getUntrackedParameter<std::string>
    ("HistFile","");
  histFolder_ = iConfig.getUntrackedParameter<std::string>
    ("HistFolder", "L1T/BXSynch/");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) { 
    dbe = edm::Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }
  
  if(dbe!=NULL)
    dbe->setCurrentFolder(histFolder_);
  
  nEvt_ = 0;
  
  if(verbose())
    std::cout << "BxTiming::BxTiming constructor...done.\n" << std::flush;
}

BxTiming::~BxTiming() {}

void 
BxTiming::beginJob(const edm::EventSetup&) {

  if(verbose())
    std::cout << "BxTiming::beginJob()  start\n" << std::flush;

  DQMStore* dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if(dbe) {
    dbe->setCurrentFolder(histFolder_);
    dbe->rmdir(histFolder_);
  }

  std::string lbl("");
  nfed_ = FEDNumbering::lastFEDId()+1;
  std::string SysLabel[NSYS] = {
    "ECAL", "HCAL", "GCT", "CSCTPG", "CSCTF", "DTTPG", "DTTF", "RPC", "GT"
  };
  
  std::pair<int,int> fedRange[NSYS] = {
    FEDNumbering::getEcalFEDIds(),      //600..670
    FEDNumbering::getHcalFEDIds(),      //700..731
    FEDNumbering::getTriggerGCTFEDIds(),//745..749
    FEDNumbering::getCSCFEDIds(),       //750..757
    FEDNumbering::getCSCTFFEDIds(),     //760..760
    FEDNumbering::getDTFEDIds(),        //770..775
    FEDNumbering::getDTTFFEDIds(),      //780..780
    FEDNumbering::getRPCFEDIds(),       //790..795
    FEDNumbering::getTriggerGTPFEDIds() //812..813
  };
  for(int i=0; i<NSYS; i++) fedRange_[i]=fedRange[i];


  int fedRefSys=-1;
  for(int i=0; i<NSYS; i++)
    if(fedRef_>=fedRange_[i].first && fedRef_<=fedRange_[i].second)
      {fedRefSys=i; break;}
  std::string refName("");
  if(fedRefSys>=0)
    refName+=SysLabel[fedRefSys];
  else
    refName+=fedRef_;

  /// book the histograms
  if(dbe) {

    dbe->setCurrentFolder(histFolder_);

    const int dbx = 100;
    hBxDiffAllFed = dbe->bookProfile("BxDiffAllFed", "BxDiffAllFed", 
				     nfed_, -0.5, nfed_+0.5, 
                                     2*dbx+1, -1*dbx-0.5,dbx+0.5);

    for(int i=0; i<NSYS; i++) {
      lbl.clear();lbl+=SysLabel[i];lbl+="FedBxDiff"; 
      int nfeds = fedRange_[i].second - fedRange_[i].first + 1;
      nfeds = (nfeds>0)? nfeds:1;
      hBxDiffSysFed[i] = dbe->bookProfile(lbl.data(),lbl.data(), nfeds, 
					  fedRange_[i].first-0.5, fedRange_[i].second+0.5,
					  2*dbx+1,-1*dbx-0.5,dbx+0.5);
    }

    const int norb = 3565;
    lbl.clear();lbl+="BxOccyAllFed";
    hBxOccyAllFed = dbe->book1D(lbl.data(),lbl.data(),norb+1,-0.5,norb+0.5);
    hBxOccyOneFed = new MonitorElement*[nfed_];
    dbe->setCurrentFolder(histFolder_+"SingleFed/");
    for(int i=0; i<nfed_; i++) {
      lbl.clear(); lbl+="BxOccyOneFed";
      char *ii = new char[1000]; std::sprintf(ii,"%d",i);lbl+=ii;
      hBxOccyOneFed[i] = dbe->book1D(lbl.data(),lbl.data(),norb+1,-0.5,norb+0.5);
      delete ii;
    }
    
  }
  
  /// labeling (cosmetics added here)
  hBxDiffAllFed->setAxisTitle("FED ID",1);
  lbl.clear(); lbl+="BX(fed)-BX("; lbl+=refName; lbl+=")";
  hBxDiffAllFed->setAxisTitle(lbl,2);
  for(int i=0; i<NSYS; i++) {
    lbl.clear(); lbl+=SysLabel[i]; lbl+=" FED ID";
    hBxDiffSysFed[i]->setAxisTitle(lbl,1);
    lbl.clear(); lbl+="BX("; lbl+=SysLabel[i]; lbl+=")-BX(";lbl+=refName; lbl+=")";
    hBxDiffSysFed[i]->setAxisTitle(lbl,2);
  }
  hBxOccyAllFed->setAxisTitle("bx",1);
  lbl.clear(); lbl+="Combined FED occupancy";
  hBxOccyAllFed->setAxisTitle(lbl,1);
  for(int i=0; i<nfed_; i++) {
    hBxOccyOneFed[i] ->setAxisTitle("bx",1);
    lbl.clear(); lbl+=" FED "; char *ii = new char[1000]; std::sprintf(ii,"%d",i);lbl+=ii; lbl+=" occupancy";
    hBxOccyOneFed[i] ->setAxisTitle(lbl,2);
  }
  
  if(verbose())
    std::cout << "BxTiming::beginJob()  end.\n" << std::flush;
}

void 
BxTiming::endJob() {

  if(verbose())
    std::cout << "BxTiming::endJob Nevents: " << nEvt_ << "\n" << std::flush;

  if(histFile_.size()!=0  && dbe) 
    dbe->save(histFile_);
  
  if(verbose())
    std::cout << "BxTiming::endJob()  end.\n" << std::flush;
}


// ------------ method called to for each event  ------------
void
BxTiming::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  if(verbose())
    std::cout << "BxTiming::analyze()  start\n" << std::flush;

  nEvt_++;

  /// get the raw data
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByLabel(fedSource_, rawdata);
  //iEvent.getByType(rawdata);

  // get the GT bits
  edm::Handle<L1GlobalTriggerReadoutRecord> gtdata;
  iEvent.getByLabel(gtSource_, gtdata);
  if(gtdata.isValid())
    std::vector<bool> gtbits = gtdata->decisionWord();

  // get reference bx
  int bxRef = FEDHeader(rawdata->FEDData(fedRef_).data()).bxID();

  // loop over feds
  for (int i = 0; i<FEDNumbering::lastFEDId(); i++){
    const FEDRawData& data = rawdata->FEDData(i);
    size_t size=data.size();
    if(!size) continue;
    FEDHeader header(data.data());
    //int lvl1id = header.lvl1ID();//Level-1 event number generated by the TTC system
    int bx = header.bxID(); // The bunch crossing number

    hBxDiffAllFed->Fill(i,bx-bxRef);
    for(int j=0; j<NSYS; j++)
      if(i>=fedRange_[j].first && i<=fedRange_[j].second)
	hBxDiffSysFed[j]->Fill(i,bx-bxRef);

    hBxOccyAllFed->Fill(bx);
    hBxOccyOneFed[i]->Fill(bx);

  }
  
  if(verbose())
    std::cout << "BxTiming::analyze() end.\n" << std::flush;
}
