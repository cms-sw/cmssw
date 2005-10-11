/*
 * \file EBHtmlTask.cc
 * 
 * $Date: 2005/10/08 08:55:06 $
 * $Revision: 1.3 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBHtmlTask.h>

EBHtmlTask::EBHtmlTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

  logFile.open("EBHtmlTask.log");

}

EBHtmlTask::~EBHtmlTask(){

  logFile.close();

}

void EBHtmlTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  cout << "EBHtmlTask::analyze" << endl;
  gROOT->cd();
  gROOT->cd("EcalBarrel/EBPedestalTask/Gain01");
  TObject* obj = gDirectory->Get("EBPT pedestal SM01 G01");
  if ( obj ) {
    cout << " Found histogram 'EBPT pedestal SM01 G01' :" << obj << endl;
  }

}

