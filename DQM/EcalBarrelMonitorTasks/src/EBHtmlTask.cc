/*
 * \file EBHtmlTask.cc
 * 
 * $Date: 2005/10/07 08:47:46 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBHtmlTask.h>

EBHtmlTask::EBHtmlTask(const edm::ParameterSet& ps, TFile* rootFile){

  logFile.open("EBHtmlTask.log");

  localRootFile = rootFile;
}

EBHtmlTask::~EBHtmlTask(){

  logFile.close();

}

void EBHtmlTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  cout << "EBHtmlTask::analyze" << endl;
  localRootFile->cd();
  localRootFile->cd("EBPedestalTask/Gain01");
  TObject* obj = gDirectory->Get("EBPT pedestal SM01 G01");
  if ( obj ) {
    cout << " Found histogram 'EBPT pedestal SM01 G01' :" << obj << endl;
  }

}

