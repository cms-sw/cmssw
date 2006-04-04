#include <DQM/HcalMonitorModule/interface/HcalMonitorModule.h>

/*
 * \file HcalMonitorModule.cc
 * 
 * $Date: 2006/01/03 19:49:47 $
 * $Revision: 1.6 $
 * \author W Fisher
 *
*/

HcalMonitorModule::HcalMonitorModule(const edm::ParameterSet& ps){

  m_logFile.open("HcalMonitorModule.log");

  m_dbe = 0;
  if ( ps.getUntrackedParameter<bool>("DaqMonitorBEInterface", false) ) {
    m_dbe = edm::Service<DaqMonitorBEInterface>().operator->();
    m_dbe->setVerbose(0);
  }
  m_monitorDaemon = false;
  if ( ps.getUntrackedParameter<bool>("MonitorDaemon", false) ) {
    edm::Service<MonitorDaemon> daemon;
    daemon.operator->();
    m_monitorDaemon = true;
  }

  m_outputFile = ps.getUntrackedParameter<string>("outputFile", "");
  if ( m_outputFile.size() != 0 ) {
    cout << "Hcal Monitoring histograms will be saved to " << m_outputFile.c_str() << endl;
  }
  
  m_runNum = 0;
  m_meStatus=0;
  m_meRunNum=0;
  m_meRunType=0;
  m_meEvtNum=0;
  m_meEvtMask=0;
  
  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal");
    m_meStatus  = m_dbe->bookInt("STATUS");
    m_meRunNum  = m_dbe->bookInt("RUN NUMBER");
    m_meRunType = m_dbe->bookInt("RUN TYPE");
    m_meEvtNum  = m_dbe->bookInt("EVT NUMBER");
    m_meEvtMask = m_dbe->bookInt("EVT MASK");

    m_meStatus->Fill(-1);
    m_meRunNum->Fill(0);
    m_meRunType->Fill(-1);
    m_meEvtNum->Fill(-1);
    m_meEvtMask->Fill(-1);
  }
  
  m_evtSel = new HcalMonitorSelector(ps);
  m_digiMon = NULL;
  m_dfMon = NULL;
  m_rhMon = NULL;
  m_pedMon = NULL;

  if ( ps.getUntrackedParameter<bool>("RecHitMonitor", false) ) {
    m_rhMon = new HcalRecHitMonitor();
    m_rhMon->setup(ps, m_dbe);
  }

  if ( ps.getUntrackedParameter<bool>("DigiMonitor", false) ) {
    m_digiMon = new HcalDigiMonitor();
    m_digiMon->setup(ps, m_dbe);
  }

  if ( ps.getUntrackedParameter<bool>("DataFormatMonitor", false) ) {
    m_dfMon = new HcalDataFormatMonitor();
    m_dfMon->setup(ps, m_dbe);
  }

  if ( ps.getUntrackedParameter<bool>("PedestalMonitor", false) ) {
    m_pedMon = new HcalPedestalMonitor();
    m_pedMon->setup(ps, m_dbe);
  }

  if ( m_dbe ) m_dbe->showDirStructure();
  
}

HcalMonitorModule::~HcalMonitorModule(){
  if(m_digiMon!=NULL) { delete m_digiMon; m_digiMon=NULL; }
  if(m_dfMon!=NULL) { delete m_dfMon; m_dfMon=NULL; }
  if(m_rhMon!=NULL) { delete m_rhMon; m_rhMon=NULL; }
  if(m_pedMon!=NULL) { delete m_pedMon; m_pedMon=NULL; }
  delete m_evtSel;

  m_logFile.close();
}

void HcalMonitorModule::beginJob(const edm::EventSetup& c){
  m_ievt = 0;
  if ( m_meStatus ) m_meStatus->Fill(0);
  if ( m_meEvtNum ) m_meEvtNum->Fill(m_ievt);
  if ( m_monitorDaemon ) sleep(15);

}

void HcalMonitorModule::endJob(void) {

  cout << "HcalMonitorModule: analyzed " << m_ievt << " events" << endl;

  if ( m_meStatus ) m_meStatus->Fill(2);
  if ( m_meRunNum) m_meRunNum->Fill(m_runNum); //???
  if ( m_meEvtNum ) m_meEvtNum->Fill(m_ievt);
  if ( m_monitorDaemon ) sleep(45);

  if(m_digiMon!=NULL) m_digiMon->done();
  if(m_dfMon!=NULL) m_dfMon->done();
  if(m_rhMon!=NULL) m_rhMon->done();
  if(m_pedMon!=NULL) m_pedMon->done();

  if ( m_outputFile.size() != 0  && m_dbe ) m_dbe->save(m_outputFile);
}

void HcalMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  m_ievt++;
  m_evtSel->processEvent(e);
  int evtMask = m_evtSel->getEventMask();
  m_runNum = m_evtSel->getRunNumber();
  
  // To get information from the event setup, you must request the "Record"
  // which contains it and then extract the object you need
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<IdealGeometryRecord>().get(geometry);

  if ( m_dbe ){ 
    m_meStatus->Fill(1);
    m_meRunNum->Fill(m_runNum);  ///????
    m_meEvtNum->Fill(m_ievt);
    m_meEvtMask->Fill(evtMask);
  }

  /////Digi monitor stuff
  if((m_digiMon != NULL) && (evtMask&DO_HCAL_DIGIMON)){
    edm::Handle<HBHEDigiCollection> hbhe;
    edm::Handle<HODigiCollection> ho;
    edm::Handle<HFDigiCollection> hf;
    e.getByType(hbhe);
    e.getByType(hf);
    e.getByType(ho);
    m_digiMon->processEvent(*hbhe, *ho, *hf);
  }

  if((m_pedMon != NULL) && (evtMask&DO_HCAL_PED_CALIBMON)){
    edm::Handle<HBHEDigiCollection> hbhe;
    edm::Handle<HODigiCollection> ho;
    edm::Handle<HFDigiCollection> hf;
    e.getByType(hbhe);
    e.getByType(hf);
    e.getByType(ho);
    // get conditions
    edm::ESHandle<HcalDbService> conditions;
    eventSetup.get<HcalDbRecord>().get(conditions);
    m_pedMon->processEvent(*hbhe, *ho, *hf, *conditions);
  }

    
  /////Daata Format monitor stuff
  if((m_dfMon != NULL) && (evtMask&DO_HCAL_DFMON)){
    edm::Handle<FEDRawDataCollection> rawraw;  
    e.getByType(rawraw);           
    m_dfMon->processEvent(*rawraw);
  }

  /////Rec Hit monitor stuff
  if((m_rhMon != NULL) && (evtMask&DO_HCAL_RECHITMON)){
    edm::Handle<HBHERecHitCollection> hb_hits;
    edm::Handle<HORecHitCollection> ho_hits;
    edm::Handle<HFRecHitCollection> hf_hits;
    e.getByType(hb_hits);
    e.getByType(ho_hits);
    e.getByType(hf_hits);
    m_rhMon->processEvent(*hb_hits,*ho_hits,*hf_hits);
  }

  if(m_ievt%1000 == 0)
    cout << "HcalMonitorModule: analyzed " << m_ievt << " events" << endl;

  return;
}

// Here are the necessary incantations to declare your module to the
// framework, so it can be referenced in a cmsRun file.
//
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalMonitorModule)

