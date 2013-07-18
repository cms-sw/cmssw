#ifndef HcalPedestalAnalyzer_H
#define HcalPedestalAnalyzer_H

/*
 * \file HcalPedestalAnalyzer.h
 *
 * $Date: 2012/10/09 14:57:58 $
 * $Revision: 1.4 $
 * \author S. Stoynev / W. Fisher
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPedestalAnalysis.h"

/*
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class HcalPedestalAnalyzer: public edm::EDAnalyzer{

public:

/// Constructor
HcalPedestalAnalyzer(const edm::ParameterSet& ps);

/// Destructor
~HcalPedestalAnalyzer();

protected:
//void SampleAnalyzer_in(const edm::ParameterSet& ps);
//void SampleAnalyzer_out();

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob();

// EndJob
void endJob(void);

private:
 
  int m_ievt;
  int ped_sample;
  //  string m_outputFileROOT;
  //string m_outputFileMean;
  //string m_outputFileWidth;
  HcalPedestalAnalysis* m_pedAnal;

  const HcalTopology* m_topo; 

  int m_startSample;
  int m_endSample;

  std::string m_inputPedestals_source;
  std::string m_inputPedestals_tag;
  int m_inputPedestals_run;
  std::string m_inputPedestalWidths_source;
  std::string m_inputPedestalWidths_tag;
  int m_inputPedestalWidths_run;
  std::string m_outputPedestals_dest;
  std::string m_outputPedestals_tag;
  int m_outputPedestals_run;
  std::string m_outputPedestalWidths_dest;
  std::string m_outputPedestalWidths_tag;
  int m_outputPedestalWidths_run;

  edm::InputTag hbheDigiCollectionTag_;
  edm::InputTag hoDigiCollectionTag_;
  edm::InputTag hfDigiCollectionTag_;
};

#endif
