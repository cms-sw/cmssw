#include <CalibCalorimetry/HcalStandardModules/interface/HcalPedestalAnalyzer.h>
#include "CalibCalorimetry/HcalAlgos/interface/HcalAlgoUtils.h"
#include <iostream>
#include <fstream>

/*
 * \file HcalPedestalAnalyzer.cc
 * 
 * $Date: 2006/01/05 19:55:32 $
 * $Revision: 1.1 $
 * \author W Fisher
 *
*/

HcalPedestalAnalyzer::HcalPedestalAnalyzer(const edm::ParameterSet& ps){

  m_logFile.open("HcalPedestalAnalyzer.log");
  m_pedAnal = new HcalPedestalAnalysis(ps);

}

HcalPedestalAnalyzer::~HcalPedestalAnalyzer(){
  m_logFile.close();
  delete m_pedAnal;
}

void HcalPedestalAnalyzer::beginJob(const edm::EventSetup& c){
  m_ievt = 0;
}

void HcalPedestalAnalyzer::endJob(void) {
  m_pedAnal->done();
}

void HcalPedestalAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& eventSetup){

  m_ievt++;

  ///get digis
  edm::Handle<HBHEDigiCollection> hbhe; e.getByType(hbhe);
  edm::Handle<HODigiCollection> ho;     e.getByType(ho);
  edm::Handle<HFDigiCollection> hf;     e.getByType(hf);
  // get conditions
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  m_pedAnal->processEvent(*hbhe, *ho, *hf, *conditions);
  
  if(m_ievt%1000 == 0)
    cout << "HcalPedestalAnalyzer: analyzed " << m_ievt << " events" << endl;

  return;
}
