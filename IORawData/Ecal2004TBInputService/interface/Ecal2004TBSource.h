/* -*- C++ -*- */
#ifndef Ecal2004TBSource_h_included
#define Ecal2004TBSource_h_included 1

#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ExternalInputSource.h"

class TFile;
class TBranch;
class TTree;
class TRawCrystal;
class TRawHeader;
class TRawPn;
class TRawTower;
class TRawTriggerChannel;
class TRunInfo;
class TRawTdcTriggers;
class TRawTdcInfo;
class TRawHodo;
class TRawLaserPulse;
class TRawPattern;
class TRawAdc2249;
class TRawScaler;

class Ecal2004TBSource : public edm::ExternalInputSource {
public:
explicit Ecal2004TBSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc);
protected:
  virtual void setRunAndEventInfo();
  virtual bool produce(edm::Event & e);
private:
  //  void unpackSetup(const std::vector<std::string>& params);
  void openFile(const std::string& filename);
  TRunInfo* m_runInfo;
  TTree* m_tree;
  TFile* m_file;
  int m_i, m_fileCounter;
  int m_imax;
  int n_towers;
  int n_pns;
  int n_pnIEs;

  static const int maxTowers=68; // MAX number of towers
  static const int maxXtals=25;  // MAX number of xtals
  static const int maxPns=10; // MAX number of Pns
  static const int maxPnIEs=10; // MAX number of Pns

  static const int nHodoFibres = 64;    // Number of fibers per hodoscope plane   
  static const int nHodoscopes = 2;     // Number of different mappings between fiber and electronics     
  static const int nHodoPlanes = 4;     // Number of hodoscopes along the beam
  static const int hodoRawLen  = 4;     // The raw data is stored as 4 integers for each hodo plane

  // hodoscopes info
  int nHodoHits[nHodoPlanes];
  int hodoHits[nHodoPlanes][nHodoFibres];
  int hodoAll[nHodoPlanes*nHodoFibres];


  TRawTower* m_towers[maxTowers];
  int towerNumbers[maxTowers];
  TRawPn* m_pns[maxPns];
  int pnNumbers[maxPns];
  TRawPn* m_pnIEs[maxPnIEs];
  int pnIENumbers[maxPnIEs];
  
  TRawHeader* m_eventHeader;
  TRawTdcInfo* m_tdcInfo;
  TRawHodo* m_hodo;

  int m_burstNum;
  int m_xtalInBeam;

  TBranch* b_burstNum;
  TBranch* b_xtalInBeam;
};



#endif // Ecal2004TBSource_h_included
