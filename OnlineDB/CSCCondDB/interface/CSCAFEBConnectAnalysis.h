#ifndef CSCAFEBCONNECTANALYSIS_H
#define CSCAFEBCONNECTANALYSIS_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "OnlineDB/CSCCondDB/interface/CSCToAFEB.h"
#include <map>
#include <string>
#include <sstream>
#include <stdio.h>

/** \class CSCAFEBConnectAnalysis
  *  
  * $Date: 2006/08/02 20:17:49 $
  * $Revision: 1.1 $
  * \author 
  */
class CSCAFEBConnectAnalysis {
public:
  CSCAFEBConnectAnalysis(); 
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
  int npulses; 
  int nmblayers;
  int pulsed_layer;
  std::vector<int>  nmbpulses;
  
  /// Maps 

  std::map<int, int>                               m_csc_list;
  std::map<int, std::vector<int> >                 m_wire_ev;
  std::map<int, std::vector<std::vector<float> > > m_res_for_db;

  /// Layer, wire to AFEB, channel conversion
  const CSCToAFEB csctoafeb;

  /// ROOT hist file
  TFile* hist_file; 

  /// Histogram maps
  std::map<int, TH1*> mh_LayerNmbPulses;
  std::map<int, TH1*> mh_WireEff;
  std::map<int, TH1*> mh_Eff;
  std::map<int, TH1*> mh_WirePairCrosstalk;
  std::map<int, TH1*> mh_PairCrosstalk;
  std::map<int, TH1*> mh_WireNonPairCrosstalk;
  std::map<int, TH1*> mh_NonPairCrosstalk;
  std::map<int, TH2*> mh_FirstTime;
};

#endif
