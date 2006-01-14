#ifndef HcalPedestalAnalysis_H
#define HcalPedestalAnalysis_H

/*
 * \file HcalPedestalAnalysis.h
 *
 * $Date: 2006/01/05 19:55:32 $
 * $Revision: 1.1 $
 * \author W. Fisher
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "TH1F.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

class HcalPedestalAnalysis{
  
public:
  
  /// Constructor
  HcalPedestalAnalysis(const edm::ParameterSet& ps);
  
  /// Destructor
  ~HcalPedestalAnalysis();
  
  void done(void);  
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);

protected:
  
  
private:
  
  void perChanHists(int id, const HcalDetId detid, const HcalQIESample& qie, map<HcalDetId, map<int, TH1F*> > &tool);
  
  string m_outputFileROOT;
  string m_outputFileMean;
  string m_outputFileWidth;
  
  int m_startSample;
  int m_endSample;
  
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;
  
  struct{
    map<HcalDetId,map<int, TH1F*> > PEDVALS;
    TH1F* ALLPEDS;
    TH1F* PEDRMS;
    TH1F* PEDMEAN;
    TH1F* CAPIDRMS;
    TH1F* CAPIDMEAN;
  } hbHists, hfHists, hoHists;
  map<HcalDetId, map<int,TH1F*> >::iterator _meo;

};

#endif
