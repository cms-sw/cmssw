#ifndef L1COMPARATOR_H
#define L1COMPARATOR_H

/*\class L1Comparator
 *\description L1 trigger data|emulation comparison and validation
 *\author Nuno Leonardo (CERN)
 *\date 07.02
 */

// common/system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <atomic>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

// l1 dataformats, d|e record includes
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

// comparator template
#include "L1Trigger/HardwareValidation/interface/DEcompare.h"

// db trigger/subsystem key access
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
//#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

template <class T>
class DEcompare;

class L1Comparator : public edm::global::EDProducer<edm::RunCache<std::array<bool, dedefs::DEnsys>>> {
public:
  explicit L1Comparator(const edm::ParameterSet&);

private:
  using RunCache = std::array<bool, dedefs::DEnsys>;
  std::shared_ptr<RunCache> globalBeginRun(edm::Run const&, const edm::EventSetup&) const final;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const final {}
  void endJob() override;

  struct EventInfo {
    L1DEDigiCollection m_dedigis;
    std::array<bool, dedefs::DEnsys> DEmatchEvt = {{true}};
    std::array<std::array<int, 2>, dedefs::DEnsys> DEncand = {{{{0, 0}}}};
    std::ostringstream dumpToFile_;
    int nevt_;
    int evtNum_;
    int runNum_;
    //flag whether event id has already been written to dumpFile
    bool dumpEvent_ = true;
  };

  template <class T>
  void process(T const*, T const*, const int, const int, EventInfo& eventInfo) const;
  template <class T>
  void process(
      const edm::Handle<T> data, const edm::Handle<T> emul, const int sys, const int cid, EventInfo& eventInfo) const {
    if (data.isValid() && emul.isValid())
      process(data.product(), emul.product(), sys, cid, eventInfo);
  }

  template <class T>
  bool CompareCollections(edm::Handle<T> data, edm::Handle<T> emul, std::ostream&) const;
  template <class T>
  bool dumpCandidate(const T& dt, const T& em, std::ostream& s) const;

  int verbose() const { return verbose_; }
  const bool m_stage1_layer2_;

private:
  mutable std::atomic<int> nevt_;
  const int verbose_;

  edm::EDGetTokenT<L1CaloEmCollection> tokenCaloEm_[2];
  edm::EDGetTokenT<L1CaloRegionCollection> tokenCaloRegion_[2];
  edm::EDGetTokenT<L1GctEmCandCollection> tokenGctEmCand_isoEm_[2];
  edm::EDGetTokenT<L1GctEmCandCollection> tokenGctEmCand_nonIsoEm_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_cenJets_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_forJets_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_tauJets_[2];
  edm::EDGetTokenT<L1GctJetCandCollection> tokenGctJetCand_isoTauJets_[2];
  edm::EDGetTokenT<L1GctEtTotalCollection> tokenGctEtTotal_[2];
  edm::EDGetTokenT<L1GctEtHadCollection> tokenGctEtHad_[2];
  edm::EDGetTokenT<L1GctEtMissCollection> tokenGctEtMiss_[2];
  edm::EDGetTokenT<L1GctHFRingEtSumsCollection> tokenGctHFRingEtSums_[2];
  edm::EDGetTokenT<L1GctHFBitCountsCollection> tokenGctHFBitCounts_[2];
  edm::EDGetTokenT<L1GctHtMissCollection> tokenGctHtMiss_[2];
  edm::EDGetTokenT<L1GctJetCountsCollection> tokenGctJetCounts_[2];
  edm::EDGetTokenT<L1MuDTChambPhContainer> tokenMuDTChambPh_[2];
  edm::EDGetTokenT<L1MuDTChambThContainer> tokenMuDTChambTh_[2];
  edm::EDGetTokenT<LTCDigiCollection> tokenLTCDigi_[2];
  edm::EDGetTokenT<L1MuDTTrackContainer> tokenMuDTTrack_[2];
  edm::EDGetTokenT<L1MuRegionalCandCollection> tokenMuRegionalCandRPCb_[2];
  edm::EDGetTokenT<L1MuRegionalCandCollection> tokenMuRegionalCandRPCf_[2];
  edm::EDGetTokenT<L1MuGMTCandCollection> tokenMuGMTCand_[2];
  edm::EDGetTokenT<L1MuGMTReadoutCollection> tokenMuReadoutCand_[2];

  const edm::ESGetToken<L1TriggerKey, L1TriggerKeyRcd> tokenTriggerKey_;

  const std::array<bool, dedefs::DEnsys> m_doSys;
  const std::string m_dumpFileName;
  CMS_THREAD_GUARD(m_fileGuard) mutable std::ofstream m_dumpFile;
  const int m_dumpMode;
  mutable std::mutex m_fileGuard;
  mutable std::atomic<bool> m_match;
};

#endif
