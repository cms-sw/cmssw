#ifndef CALIBMUON_CSCCALIBRATION_CSCAFEBTHRANALYSIS_H
#define CALIBMUON_CSCCALIBRATION_CSCAFEBTHRANALYSIS_H 1

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "CalibMuon/CSCCalibration/interface/CSCToAFEB.h"
#include <map>
#include <string>
#include <sstream>
#include <stdio.h>

/** \class CSCAFEBThrAnalysis
  *  
  * $Date: $
  * $Revision: $
  * \author 
  */
class CSCAFEBThrAnalysis {
public:
  CSCAFEBThrAnalysis(); 
  void setup(const std::string& histoFileName);
  void analyze(const CSCWireDigiCollection& wirecltn); 
  void done();

private:
  void bookForId(int flag, const int& idint,const std::string& ids);

  /// Statistics
  int nmbev;
  int nmbev_no_wire;

  /// DAC info
  int npulses; 
  int unsigned indDac;
  int BegDac;
  int EvDac;
  int StepDac;
  std::vector<float> vecDac;
  
  /// Maps - per event, threshold curve, fit results
  std::map<int, std::vector<int> >                 m_wire_ev;
  std::map<int, std::vector<std::vector<int> > >   m_wire_dac;
  std::map<int, std::vector<std::vector<float> > > m_res_for_db;

  /// Layer, wire to AFEB, channel conversion
  const CSCToAFEB csctoafeb;

  /// ROOT hist file
  TFile* hist_file; 

  /// Histogram maps
  std::map<int, TH1*> mh_ChanEff;

  std::map<int, TH2*> mh_FirstTime;
  std::map<int, TH2*> mh_AfebDac;
  std::map<int, TH2*> mh_AfebThrPar;
  std::map<int, TH2*> mh_AfebNoisePar;
  std::map<int, TH2*> mh_AfebNDF;
  std::map<int, TH2*> mh_AfebChi2perNDF;
};

#endif
