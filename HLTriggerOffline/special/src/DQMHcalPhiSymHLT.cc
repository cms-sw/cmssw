// -*- C++ -*-
//
/*
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory Safronov
//         Created:  Thu Sep 10 08:28:14 CEST 2009
// $Id: DQMHcalPhiSymHLT.cc,v 1.1 2009/09/10 10:18:14 safronov Exp $
//
//


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

class DQMHcalPhiSymHLT : public edm::EDAnalyzer {
public:
  explicit DQMHcalPhiSymHLT(const edm::ParameterSet&);
  ~DQMHcalPhiSymHLT();
  
  MonitorElement* hFEDsize;
  MonitorElement* hHCALsize;
  MonitorElement* hFULLsize;
  MonitorElement* hHCALvsLumiSec;


  std::string folderName_;
  std::string outRootFileName_;
  edm::InputTag rawInLabel_;
  bool saveToRootFile_;
  
  DQMStore* dbe_;  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  int firstLumiSec;  
  int iEvt;
};


DQMHcalPhiSymHLT::DQMHcalPhiSymHLT(const edm::ParameterSet& iConfig)
{
  folderName_ = iConfig.getParameter<std::string>("folderName");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");
  rawInLabel_=iConfig.getParameter<edm::InputTag>("rawInputLabel");
  saveToRootFile_=iConfig.getParameter<bool>("SaveToRootFile");
 
  iEvt=0;
}

DQMHcalPhiSymHLT::~DQMHcalPhiSymHLT()
{
}


void 
DQMHcalPhiSymHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (iEvt==0) firstLumiSec=iEvent.luminosityBlock();
  iEvt++;
  
  std::auto_ptr<FEDRawDataCollection> producedData(new FEDRawDataCollection);
  
  edm::Handle<FEDRawDataCollection> rawIn;
  iEvent.getByLabel(rawInLabel_,rawIn);
 
  std::vector<int> selFEDs;

  //get HCAL FEDs:
  for (int i=FEDNumbering::MINHCALFEDID; i<=FEDNumbering::MAXHCALFEDID; i++)
    {
      selFEDs.push_back(i);
    }

  const FEDRawDataCollection *rdc=rawIn.product();
  
  //calculate full HCAL data size:
  size_t hcalSize=0;
  for (unsigned int k=0; k<selFEDs.size(); k++)
    {
      const FEDRawData & fedData = rdc->FEDData(selFEDs[k]);
      hcalSize+=fedData.size();
      hFEDsize->Fill(fedData.size()*pow(1024,-1),1);
    }
  hHCALsize->Fill(hcalSize*pow(1024,-1),1);
  hHCALvsLumiSec->Fill(iEvent.luminosityBlock()-firstLumiSec,hcalSize*pow(1024,-1),1);
  
  //calculate full data size:
  size_t fullSize=0;
  for (int j=0; j<FEDNumbering::MAXFEDID; ++j )
    {
      const FEDRawData & fedData = rdc->FEDData(j);
      fullSize+=fedData.size();
    }
  hFULLsize->Fill(fullSize*pow(1024,-1),1);
}


// ------------ method called once each job just before starting event loop  ------------
void 
DQMHcalPhiSymHLT::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);
  
  hFEDsize=dbe_->book1D("hFEDsize","HCAL FED size (kB)",1000,0,100);
  hFEDsize->setAxisTitle("kB",1);

  hHCALsize=dbe_->book1D("hHCALsize","HCAL data size (kB)",1000,0,1000);
  hHCALsize->setAxisTitle("kB",1);

  hFULLsize=dbe_->book1D("hFULLsize","Full data size (kB)",1000,0,2000);
  hFULLsize->setAxisTitle("kB",1);

  hHCALvsLumiSec=dbe_->book2D("hHCALvsLumiSec","HCAL data size (kB) vs. internal lumi block number",10000,0,10000,1000,0,1000);
  hHCALvsLumiSec->setAxisTitle("kB",2);
  hHCALvsLumiSec->setAxisTitle("internal luminosity block number",1);
}

void 
DQMHcalPhiSymHLT::endJob() 

{
  if(dbe_&&saveToRootFile_) 
    {  
      dbe_->save(outRootFileName_);
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMHcalPhiSymHLT);
