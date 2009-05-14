#include "DQM/RPCMonitorClient/interface/RPCMonitorRaw.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <bitset>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/RPCRawToDigi/interface/RPCRawDataCounts.h"


#include "TH1D.h"
#include "TH2D.h"
#include "TFile.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


using namespace std;
using namespace edm;


RPCMonitorRaw::~RPCMonitorRaw() { LogTrace("") << "RPCMonitorRaw destructor"; }

void RPCMonitorRaw::beginJob( const edm::EventSetup& )
{

// Get DQM interface
  DQMStore* theDMBE = edm::Service<DQMStore>().operator->();
  
  theDMBE->setCurrentFolder(theConfig.getUntrackedParameter<std::string>("PrefixDir","RPC" )+"/FEDIntegrity" );
  
  me_h[0]=theDMBE->book1D("recordType_790",theCounts.recordTypeHisto(790));
  me_h[1]=theDMBE->book1D("recordType_791",theCounts.recordTypeHisto(791));
  me_h[2]=theDMBE->book1D("recordType_792",theCounts.recordTypeHisto(792));
  
  me_e =theDMBE->book1D("readoutErrors",theCounts.readoutErrorHisto());
}

void RPCMonitorRaw::endJob()
{

  bool writeHistos = theConfig.getUntrackedParameter<bool>("writeHistograms");
  if (writeHistos) {
    std::string histoFile = theConfig.getUntrackedParameter<std::string>("histoFileName"); 
    TFile f(histoFile.c_str(),"RECREATE");
    me_h[0]->getTH1F()->Write();
    me_h[1]->getTH1F()->Write();
    me_h[2]->getTH1F()->Write();
    me_e->getTH1F()->Write();
    edm::LogInfo(" END JOB, histos saved!");
    f.Close();
  }
}


void RPCMonitorRaw::analyze(const  edm::Event& ev, const edm::EventSetup& es) 
{
  edm::Handle<RPCRawDataCounts> rawCounts;
  ev.getByType( rawCounts);

  const RPCRawDataCounts * aCounts = rawCounts.product();
  theCounts += *aCounts;

  me_e->Reset();
  me_h[0]->Reset();
  me_h[1]->Reset();
  me_h[2]->Reset();

  vector<double> v1;

  theCounts.readoutErrorVector(v1);
  me_e->getTH1F()->FillN(v1.size()/2,&v1[0],&v1[1],2);

  for (int i=0;i<3;i++) {
    theCounts.recordTypeVector(790+i,v1);
    me_h[i]->getTH1F()->FillN(v1.size()/2,&v1[0],&v1[1],2);
  }

  me_h[0]->update();
  me_h[1]->update();
  me_h[2]->update();
  me_e->update();
}
