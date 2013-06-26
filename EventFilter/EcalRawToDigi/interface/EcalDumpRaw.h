/*
 * $Id: EcalDumpRaw.h,v 1.6 2013/04/22 15:48:17 wmtan Exp $
 *
 * Author: Ph Gras. CEA/IRFU - Saclay
 */

#ifndef ECALDUMPRAW_H
#define ECALDUMPRAW_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <inttypes.h>
//#include "pgras/PGUtilities/interface/PGHisto.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

/**
 * Utility to dump ECAL Raw data. Hexadecimal dump is accompagned with a side by
 * data interpretention.
 *
 * The script test/dumpRaw can be used to run this module. E. g.:
 *  dumpRaw /store/..../data_file.root
 * Run dumpRaw -h to get help on this script.
 *
 * Author: Ph. Gras CEA/IRFU Saclay
 *
 */
class EcalDumpRaw : public edm::EDAnalyzer {
  //ctors
public:
  explicit EcalDumpRaw(const edm::ParameterSet&);
  ~EcalDumpRaw();


  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  void analyzeEB(const edm::Event&, const edm::EventSetup&) const;
  void analyzeEE(const edm::Event&, const edm::EventSetup&) const;
  void endJob();  
  //methods
public:
private:
  void analyzeFed(int fedId);
  void analyzeApd();
  std::string toNth(int n);
  bool decode(const uint32_t* data, int iWord32, std::ostream& out);
  double max(const std::vector<double>& a, unsigned& pos){
    pos = 0;
    double m = a[pos];
    for(unsigned i = 1; i < a.size(); ++i){
      if(a[i]>m){ m = a[i]; pos = i;}
    }
    return m;
  }
  double min(const std::vector<double>& a){
    double m = a[0];
    for(unsigned i = 1; i < a.size(); ++i){
      if(a[i]<m) m = a[i];
    }
    return m;
  }
  //static int lme(int dccId1, int side);

  template<class T>
  std::string toString(T val){
    std::stringstream s;
    s << val;
    return s.str();
  }

  static int sideOfRu(int ru1);
  
  static int modOfRu(int ru1);

  static int lmodOfRu(int ru1);

  std::string srRange(int offset) const;

  std::string ttfTag(int tccType, unsigned iSeq) const;

  std::string tpgTag(int tccType, unsigned iSeq) const;
  
  //fields
private:
  int      verbosity_;
  bool     writeDcc_;
  int      beg_fed_id_;
  int      end_fed_id_;
  int      first_event_;
  int      last_event_;
  std::string   filename_;
  int      iEvent_;

  unsigned iTowerWord64_;
  unsigned iSrWord64_;
  unsigned iTccWord64_;
  enum {inDaqHeader, inDccHeader, inTccBlock, inSrBlock, inTowerBlock}
  decodeState_;
  size_t towerBlockLength_;

  std::vector<double> adc_;

  static const int nSamples = 10;
  double amplCut_;
  bool dump_;
  bool dumpAdc_;
  bool l1aHistory_;
  //  bool doHisto_;
  int maxEvt_;
  int profileFedId_;
  int profileRuId_;
  int l1aMinX_;
  int l1aMaxX_;
  int dccCh_;
  std::vector<uint32_t> lastOrbit_;
  static const unsigned nDccs_ = 54;
  static const unsigned fedStart_ = 601;
  static const int maxTpgsPerTcc_ = 68;
  static const int maxTccsPerDcc_ = 4;


  //@{
  /** TCC types
   */
  static const int ebmTcc_    = 0;
  static const int ebpTcc_    = 1;
  static const int eeInnerTcc_   = 2;
  static const int eeOuterTcc_  = 3;
  static const int nTccTypes_ = 4;
  //@}

  /** TT ID in the order the TPG appears in the data
   */
  static const int ttId_[nTccTypes_][maxTpgsPerTcc_];
  
  unsigned fedId_;
  unsigned dccId_;
  unsigned side_;
  unsigned eventId_;
  std::vector<unsigned> eventList_;
  unsigned minEventId_;
  unsigned maxEventId_;
  unsigned orbit0_;
  uint32_t orbit_;
  bool orbit0Set_;
  int bx_;
  int l1a_;
  int l1amin_;
  int l1amax_;
  int simpleTrigType_;
  int detailedTrigType_;
  //  PGHisto histo_;
  std::vector<std::vector<uint32_t> > l1as_;
  std::vector<std::vector<uint32_t> > orbits_;
  std::vector<std::vector<int> > tpg_;
  std::vector<int> nTpgs_;
  std::vector<int> dccChStatus_;
  int iRu_;
  int srpL1a_;
  int tccL1a_;
  //Number of TPGs in TCC block currently parsed:
  int nTts_;
  //Length of TCC block currently parsed:
  int tccBlockLen64_;
  static const int nRu_ = 70;
  std::vector<int> feL1a_;
  int srpBx_;
  int tccBx_;
  ///type of TCC currently parsed
  int tccType_;
  std::vector<int> feBx_;
  std::vector<int> feRuId_;
  int iTow_;
  std::ofstream dumpFile_;
  bool pulsePerRu_;
  bool pulsePerLmod_;
  bool pulsePerLme_;
  int tccId_;
  //tcc sequence number of currenlty parsed tower block of one DCC
  int iTcc_;
  edm::InputTag fedRawDataCollectionTag_;
  edm::InputTag l1AcceptBunchCrossingCollectionTag_;
};

#endif //ECALDUMPRAW_H not defined
