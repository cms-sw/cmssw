//Emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-

#ifndef ECALDATAREADER_H
#define ECALDATAREADER_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <inttypes.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

//forward declaration
class EcalElectronicsMapping;

/** EcalDataReader class
 *
 * CMSSW module to decode Ecal raw data. It can both produce
 * ECAL digis and dump data in hexadecimal with a side-by-side
 * human readable data interpretation. It replaces former
 * EcalDumpRaw and EcalRawToDigi modules providing the
 * functionalities of both.
 *
 * Original author: Ph. Gras CEA/Saclay
 */
class EcalDataReader : public edm::EDProducer {
  //ctors
public:
  explicit EcalDataReader(const edm::ParameterSet&);
  ~EcalDataReader();


  virtual void produce(edm::Event&, const edm::EventSetup&);

  void analyzeEB(const edm::Event&, const edm::EventSetup&) const;
  void analyzeEE(const edm::Event&, const edm::EventSetup&) const;
  void beginJob();
  void endJob();
  //methods
public:
private:
  void analyzeFed(int fedId);
  void analyzeApd();
  std::string toNth(int n);
  bool decode(const uint32_t* data, int iWord32, std::ostream& out);

  static int lme(int dccId1, int side);

  static int lmodOfRu(int ru1);

  double max(std::vector<double> a, unsigned& pos){
    pos = 0;
    double m = a[pos];
    for(unsigned i = 1; i < a.size(); ++i){
      if(a[i]>m){ m = a[i]; pos = i;}
    }
    return m;
  }
  double min(std::vector<double> a){
    double m = a[0];
    for(unsigned i = 1; i < a.size(); ++i){
      if(a[i]<m) m = a[i];
    }
    return m;
  }
  static int modOfRu(int ru1);

  void reset();

  void setDccHeader();

  static int sideOfRu(int ru1);

  std::string srRange(int offset) const;

  template<class T>
  std::string toString(T val){
    std::stringstream s;
    s << val;
    return s.str();
  }

  EcalTrigTowerDetId ttSeq2ttDetId(int iSeq) const;

  EcalTrigTowerDetId ebSrfSeq2ttDetId(int iSeq) const;

  std::string ttfTag(int tccType, unsigned iSeq) const;

  std::string tpgTag(int tccType, unsigned iSeq) const;

  std::string prettyTp(int tp) const;


  //@{
  /** Methods to fill digi collecions
   */
  void newTpg(int iTcc, int iSeq, int val);

  void newSrf(int iSeq, int val);

  void newDataFrame();

  void initCollections();

  void putCollections(edm::Event& event);
  //@}

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
  //  bool doHisto_;
  int maxEvt_;
  int profileFedId_;
  int profileRuId_;
  int l1aMinX_;
  int l1aMaxX_;

  /** Input collection name
   */
  edm::InputTag ecalRawDataCollection_;

  //@{
  /** Output collection names
   */
  std::string ebDigiCollection_;
  std::string eeDigiCollection_;
  std::string ebSrFlagCollection_;
  std::string eeSrFlagCollection_;
  std::string tpgCollection_;
  std::string dccHeaderCollection_;
  //@}

  //@{
  /** Buffers for produced digi collection
   */
  std::auto_ptr<EBDigiCollection> ebDigiColl_;
  std::auto_ptr<EEDigiCollection> eeDigiColl_;
  std::auto_ptr<EBSrFlagCollection> ebSrfColl_;
  std::auto_ptr<EESrFlagCollection> eeSrfColl_;
  std::auto_ptr<EcalTrigPrimDigiCollection> tpgColl_;
  std::auto_ptr<EcalRawDataCollection> dccHeaderColl_;
  //@}

  //@{
  /** Switches for digi production
   */
  bool produceDigis_;
  bool produceSrfs_;
  bool produceTps_;
  bool produceDccHeaders_;
  bool producePnDiodeDigis_;
  bool producePseudoStripInputs_;
  //@}

  /** Switch for production of in-error channel DetdId list
   */
  bool produceBadChannelList_;

  int dccCh_;
  std::vector<uint32_t> lastOrbit_;
  static const unsigned nDccs_ = 54;
  static const unsigned nTccs_ = 108;
  static const unsigned fedStart_ = 601;
  static const unsigned maxTpsPerTcc_ = 68;
  static const unsigned maxTccsPerDcc_ = 4;
  /** Number of DCC channels (readout units) including MEMs
   */
  static const int nRu_ = 70;

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
  static const int ttId_[nTccTypes_][maxTpsPerTcc_];

  //  static const int tccTtOffset_[nTccs_];

  /** EtaAbs and Phi offset of TCC
   * first index: TCC ID starting at 0
   * second index: 0 for etaAbs, 1 for etaPhi
   */
  static const int tccEtaPhi_[nTccs_][2];

  /** TT index in TCC in the order the TPG appears in the data.
   * The TT index starts from 0, runs along phi in trigonometric
   * way and then runs along eta.
   */
  static const int seq2iTt0_[nTccTypes_][maxTpsPerTcc_];

  unsigned fedId_;
  unsigned dccId_;
  unsigned side_;
  unsigned eventId_;
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
  int color_;
  //  std::vector<std::vector<uint32_t> > l1as_;
  //  std::vector<std::vector<uint32_t> > orbits_;
  std::vector<std::vector<int> > tpg_;
  //Number of TPs in TCC block currently parsed:
  std::vector<unsigned> nTps_;
  int iRu_;
  int srpL1a_;
  std::vector<short> tccL1a_;
  //Length of TCC block currently parsed:
  int tccBlockLen64_;
  std::vector<short> feL1a_;
  int srpBx_;
  std::vector<short> tccBx_;
  ///type of TCC currently parsed
  int tccType_;
  std::vector<short> feBx_;
  std::vector<int> feRuId_;
  int iTow_;
  std::ofstream dumpFile_;
  bool pulsePerRu_;
  bool pulsePerLmod_;
  bool pulsePerLme_;
  int thisTccId_;
  int tccId_[maxTccsPerDcc_];
  //tcc sequence number of currently parsed tower block of one DCC
  unsigned iTcc_;
  //Number of signal samples in data frame currently parsed.
  int nSamples_;
  int runNumber_;
  int dccErrors_;
  int eventLengthFromHeader_;
  int runType_;
  int tccStatus_;
  int srStatus_;
  int mf_;
  int tzs_;
  int zs_;
  int sr_;
  std::vector<short> feStatus_;
  EcalDCCHeaderBlock dccHeader_;

  const uint16_t* pDataFrame_;
  int strip_;
  int xtalInStrip_;

  const EcalElectronicsMapping* elecMap_;
};

#endif //ANALYZER_MODULE_H not defined
