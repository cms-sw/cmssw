//#######################
// Author Marcin Konecki
//######################

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
#include "DQM/RPCMonitorClient/interface/RPCRawDataCountsHistoMaker.h"


#include "TH1F.h"
#include "TH2F.h"
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
  
  theDMBE->setCurrentFolder("RPC/FEDIntegrity/");
  
  me_t[0]=theDMBE->book1D("recordType_790",RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(790));
  me_t[1]=theDMBE->book1D("recordType_791",RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(791));
  me_t[2]=theDMBE->book1D("recordType_792",RPCRawDataCountsHistoMaker::emptyRecordTypeHisto(792));
  for (int i=0;i<3;++i)me_t[i]->getTH1F()->SetStats(0);
  
  me_e[0]=theDMBE->book1D("readoutErrors_790",RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(790));
  me_e[1]=theDMBE->book1D("readoutErrors_791",RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(791));
  me_e[2]=theDMBE->book1D("readoutErrors_792",RPCRawDataCountsHistoMaker::emptyReadoutErrorHisto(792));
  for (int i=0;i<3;++i)me_e[i]->getTH1F()->SetStats(0);

  me_mapGoodEvents=theDMBE->book2D("mapGoodEvents","mapGoodEvents",36,-0.5,35.5, 3, 789.5,792.5);
  me_mapGoodEvents->getTH2F()->SetNdivisions(3,"y");
  me_mapGoodEvents->getTH2F()->SetStats(0);
  me_mapBadEvents =theDMBE->book2D("mapBadEvents", "mapBadEvents", 36,-0.5,35.5, 3, 789.5,792.5);
  me_mapBadEvents->getTH2F()->SetNdivisions(3,"y");
  me_mapBadEvents->getTH2F()->SetStats(0);

}
void RPCMonitorRaw::endJob()
{
bool writeHistos = theConfig.getUntrackedParameter<bool>("writeHistograms", false);
  if (writeHistos) {
    std::string histoFile = theConfig.getUntrackedParameter<std::string>("histoFileName"); 
    TFile f(histoFile.c_str(),"RECREATE");
    for (int i=0; i<3; ++i) {
      me_t[i]->getTH1F()->Write();
      me_e[i]->getTH1F()->Write();
    }
    me_mapGoodEvents->getTH2F()->Write();
    me_mapBadEvents->getTH2F()->Write();
    edm::LogInfo(" END JOB, histos saved!");
    f.Close();
  }

 
}


void RPCMonitorRaw::analyze(const  edm::Event& ev, const edm::EventSetup& es) 
{

  edm::Handle<RPCRawDataCounts> rawCounts;
  ev.getByType( rawCounts);
  RPCRawDataCountsHistoMaker histoMaker(*rawCounts.product());

  for (int i=0; i<3; i++) {
    histoMaker.fillRecordTypeHisto(790+i, me_t[i]->getTH1F());
    histoMaker.fillReadoutErrorHisto(790+i, me_e[i]->getTH1F() );
  }
  histoMaker.fillGoodEventsHisto(me_mapGoodEvents->getTH2F());
  histoMaker.fillBadEventsHisto(me_mapBadEvents->getTH2F());


  for (int i=0; i<3; ++i) {
    me_t[i]->update();
    me_e[i]->update();
  }
  me_mapGoodEvents->update();
  me_mapBadEvents->update();

 

}
