#include <DQM/HcalMonitorModule/interface/HcalMonitorModule.h>

/*
 * \file HcalMonitorModule.cc
 * 
 * $Date: 2005/11/13 17:20:53 $
 * $Revision: 1.1 $
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

  if ( ps.getUntrackedParameter<bool>("MonitorDaemon", false) ) {
    edm::Service<MonitorDaemon> daemon;
    daemon.operator->();
  }

  m_outputFile = ps.getUntrackedParameter<string>("outputFile", "");
  if ( m_outputFile.size() != 0 ) {
    cout << "Hcal Monitoring histograms will be saved to " << m_outputFile.c_str() << endl;
  }
  
  if ( m_dbe ) {
    m_dbe->setCurrentFolder("Hcal");
    m_meStatus  = m_dbe->bookInt("STATUS");
    m_meRun     = m_dbe->bookInt("RUN");
    m_meEvt     = m_dbe->book1D("EVT","EVT", 100,0,1000);
  }
  
  m_digiMon = new HcalDigiMonitor();
  m_digiMon->setup(ps, m_dbe);

  m_dfMon = new HcalDataFormatMonitor();
  m_dfMon->setup(ps, m_dbe);

  if ( m_dbe ) m_dbe->showDirStructure();
  
}

HcalMonitorModule::~HcalMonitorModule(){
  delete m_digiMon;
  delete m_dfMon;
  m_logFile.close();
}

void HcalMonitorModule::beginJob(const edm::EventSetup& c){
  m_ievt = 0;
  if ( m_meStatus ) m_meStatus->Fill(0);
}

void HcalMonitorModule::endJob(void) {

  cout << "HcalMonitorModule: analyzed " << m_ievt << " events" << endl;
  m_digiMon->done();
  m_dfMon->done();
  if ( m_meStatus ) m_meStatus->Fill(2);
  if ( m_outputFile.size() != 0  && m_dbe ) m_dbe->save(m_outputFile);

  sleep(5);
}

void HcalMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  m_ievt++;

  if ( m_meStatus ) m_meStatus->Fill(1);
  if ( m_meRun ) m_meRun->Fill(14316);
  if ( m_meEvt ) m_meEvt->Fill(m_ievt,m_ievt);

  edm::Handle<HBHEDigiCollection> hbhe;
  edm::Handle<HODigiCollection> ho;
  edm::Handle<HFDigiCollection> hf;
  e.getByType(hbhe);
  e.getByType(hf);
  e.getByType(ho);
  m_digiMon->processEvent(*hbhe, *ho, *hf);

  edm::Handle<FEDRawDataCollection> rawraw;  
  e.getByType(rawraw);           
  m_dfMon->processEvent(*rawraw);

  sleep(5);
  return;
}

// Here are the necessary incantations to declare your module to the
// framework, so it can be referenced in a cmsRun file.
//
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HcalMonitorModule)

