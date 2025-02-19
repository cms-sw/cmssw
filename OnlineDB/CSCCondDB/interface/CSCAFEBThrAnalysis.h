#ifndef CSCAFEBTHRANALYSIS_H
#define CSCAFEBTHRANALYSIS_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "OnlineDB/CSCCondDB/interface/CSCToAFEB.h"
#include <map>
#include <string>
#include <sstream>
#include <stdio.h>

/** \class CSCAFEBThrAnalysis
  *  
  * $Date: 2006/08/02 20:19:51 $
  * $Revision: 1.2 $
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
  void hf1ForId(std::map<int, TH1*>& mp, int flag, const int& id, 
                                         float& x, float w);
  void hf2ForId(std::map<int, TH2*>& mp, int flag, const int& id,
                                         float& x, float& y, float w);
  /// Statistics
  int nmbev;
  int nmbev_no_wire;

  /// DAC info
  int npulses; 
  int unsigned indDac;
  int BegDac;
  int EndDac;
  int EvDac;
  int StepDac;
  std::vector<float> vecDac;
  std::vector<int> vecDacOccup;
  
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
