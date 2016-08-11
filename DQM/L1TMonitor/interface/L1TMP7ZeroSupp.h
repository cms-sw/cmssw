#ifndef DQM_L1TMonitor_L1TMP7ZeroSupp_h
#define DQM_L1TMonitor_L1TMP7ZeroSupp_h

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class L1TMP7ZeroSupp : public DQMEDAnalyzer {

 public:

  L1TMP7ZeroSupp(const edm::ParameterSet& ps);
  virtual ~L1TMP7ZeroSupp();

 protected:

  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

 private:

  // Add additional bins only before NBINLABELS
  enum binlabels {EVTS=0, EVTSGOOD, EVTSBAD, BLOCKS, ZSBLKSGOOD, ZSBLKSBAD, ZSBLKSBADFALSEPOS, ZSBLKSBADFALSENEG, NBINLABELS};

  edm::EDGetTokenT<FEDRawDataCollection> fedDataToken_;
  std::vector<int> fedIds_;
  std::vector<std::vector<int>> masks_;

  // header and trailer sizes in chars
  int slinkHeaderSize_;
  int slinkTrailerSize_;
  int amc13HeaderSize_;
  int amc13TrailerSize_;
  int amcHeaderSize_;
  int amcTrailerSize_;
  int zsFlagMask_;

  int maxFedReadoutSize_;

  std::string monitorDir_;
  bool verbose_;

  MonitorElement* zeroSuppVal_;
  MonitorElement* readoutSizeNoZS_;
  MonitorElement* readoutSizeZS_;
  MonitorElement* readoutSizeZSExpected_;
};

#endif
