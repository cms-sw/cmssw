#include "DQM/RPCMonitorDigi/interface/RPCTTUMonitor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fmt/format.h>

//
RPCTTUMonitor::RPCTTUMonitor(const edm::ParameterSet& iConfig) {
  ttuFolder = iConfig.getUntrackedParameter<std::string>("TTUFolder", "RPC/TTU");
  outputFile = iConfig.getUntrackedParameter<std::string>("OutPutFile", "");

  m_gtReadoutLabel = consumes<L1GlobalTriggerReadoutRecord>(iConfig.getParameter<edm::InputTag>("GTReadoutRcd"));
  m_gmtReadoutLabel = consumes<L1MuGMTReadoutCollection>(iConfig.getParameter<edm::InputTag>("GMTReadoutRcd"));
  m_rpcTechTrigEmu = consumes<L1GtTechnicalTriggerRecord>(iConfig.getParameter<edm::InputTag>("L1TTEmuBitsLabel"));

  m_ttBits = iConfig.getParameter<std::vector<unsigned> >("BitNumbers");
  m_maxttBits = m_ttBits.size();
}

// ------------ method called to for each event  ------------
void RPCTTUMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //..............................................................................................
  // Data .
  edm::Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  iEvent.getByToken(m_gtReadoutLabel, gtRecord);

  if (!gtRecord.isValid()) {
    edm::LogError("RPCTTUMonitor") << "can nout find L1GlobalTriggerRecord \n";
    return;
  }

  // Emulator .
  edm::Handle<L1GtTechnicalTriggerRecord> emuTTRecord;
  iEvent.getByToken(m_rpcTechTrigEmu, emuTTRecord);

  if (!emuTTRecord.isValid()) {
    edm::LogError("RPCTTUMonitor") << "can not find L1GtTechnicalTriggerRecord (emulator) \n";
    return;
  }

  //..............................................................................................
  //
  //Timing difference between RPC-PAT and DT

  const int dGMT = discriminateGMT(iEvent, iSetup);
  if (dGMT < 0)
    return;

  std::map<int, bool> ttuDec;

  const int bxX = iEvent.bunchCrossing();  // ... 1 to 3564

  for (int k = 0; k < m_maxttBits; ++k) {
    for (int iebx = 0; iebx <= 2; iebx++) {
      const TechnicalTriggerWord gtTTWord = gtRecord->technicalTriggerWord(iebx - 1);
      ttuDec[iebx - 1] = gtTTWord[24 + k];
    }

    //. RPC
    if (m_rpcTrigger) {
      const int bx1 = (bxX - m_GMTcandidatesBx[0]);
      for (const auto& dec : ttuDec) {
        if (dec.second) {
          const int bx2 = dec.first;
          const float bxdiffPacTT = 1.0 * (bx1 - bx2);
          m_bxDistDiffPac[k]->Fill(bxdiffPacTT);
        }
      }
    }

    //.. DT
    if (m_dtTrigger) {
      const int bx1 = (bxX - m_DTcandidatesBx[0]);
      for (const auto& dec : ttuDec) {
        if (dec.second) {
          const int bx2 = dec.first;
          const float bxdiffDtTT = 1.0 * (bx1 - bx2);
          m_bxDistDiffDt[k]->Fill(bxdiffDtTT);
        }
      }
    }
    ttuDec.clear();
  }

  m_GMTcandidatesBx.clear();
  m_DTcandidatesBx.clear();

  //..............................................................................................
  //
  //... For Data Emulator comparison

  const TechnicalTriggerWord gtTTWord = gtRecord->technicalTriggerWord();

  std::vector<L1GtTechnicalTrigger> ttVec = emuTTRecord->gtTechnicalTrigger();

  int k = 0;
  //int m_BxWindow = 0;
  bool hasDataTrigger = false;
  bool hasEmulatorTrigger = false;

  if (ttVec.empty())
    return;

  for (const auto& bits : m_ttBits) {
    hasDataTrigger = gtTTWord.at(bits);
    m_ttBitsDecisionData->Fill(bits, (int)hasDataTrigger);

    hasEmulatorTrigger = ttVec[k].gtTechnicalTriggerResult();
    m_ttBitsDecisionEmulator->Fill(ttVec[k].gtTechnicalTriggerBitNumber(), (int)hasEmulatorTrigger);

    discriminateDecision(hasDataTrigger, hasEmulatorTrigger, k);

    ++k;
  }
}

int RPCTTUMonitor::discriminateGMT(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  iEvent.getByToken(m_gmtReadoutLabel, pCollection);

  if (!pCollection.isValid()) {
    edm::LogError("discriminateGMT") << "can't find L1MuGMTReadoutCollection with label \n";

    return -1;
  }

  int gmtDec(0);

  bool rpcBar_l1a = false;
  bool dtBar_l1a = false;

  m_dtTrigger = false;
  m_rpcTrigger = false;

  // get GMT readout collection
  const L1MuGMTReadoutCollection* gmtRC = pCollection.product();

  // get record vector
  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtRC->getRecords();

  edm::LogInfo("DiscriminateGMT") << "nRecords: " << gmt_records.size() << '\n';

  for (const auto& rr : gmt_records) {
    const int BxInEvent = rr.getBxInEvent();
    const int BxInEventNew = rr.getBxNr();

    // RPC barrel muon candidates
    int nrpcB = 0;
    int ndtB = 0;

    std::vector<L1MuRegionalCand> BrlRpcCands = rr.getBrlRPCCands();
    std::vector<L1MuRegionalCand> BrlDtCands = rr.getDTBXCands();

    for (const auto& rc : BrlRpcCands) {
      if (!rc.empty()) {
        m_GMTcandidatesBx.push_back(BxInEventNew);

        nrpcB++;
      }
    }

    for (const auto& rc : BrlDtCands) {
      if (!rc.empty()) {
        m_DTcandidatesBx.push_back(BxInEventNew);
        ndtB++;
      }
    }

    if (BxInEvent == 0 && nrpcB > 0)
      rpcBar_l1a = true;
    if (BxInEvent == 0 && ndtB > 0)
      dtBar_l1a = true;
  }

  if (rpcBar_l1a) {
    gmtDec = 1;
    m_rpcTrigger = true;
  }

  if (dtBar_l1a) {
    gmtDec = 2;
    m_dtTrigger = true;
  }

  return gmtDec;
}

void RPCTTUMonitor::discriminateDecision(bool data, bool emu, int indx) {
  if (data == 1 && emu == 1) {
    m_dataVsemulator[indx]->Fill(1);
  } else if (data == 1 && emu == 0) {
    m_dataVsemulator[indx]->Fill(3);
  } else if (data == 0 && emu == 1) {
    m_dataVsemulator[indx]->Fill(5);
  } else if (data == 0 && emu == 0) {
    m_dataVsemulator[indx]->Fill(7);
  }
}

void RPCTTUMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& r, edm::EventSetup const& iSetup) {
  ibooker.setCurrentFolder(ttuFolder);
  std::string hname;

  m_ttBitsDecisionData = ibooker.book1D("TechTrigger.Bits.Data", "Technical Trigger bits : Summary", 10, 23, 33);

  m_ttBitsDecisionEmulator =
      ibooker.book1D("TechTrigger.Bits.Emulator", "Technical Trigger bits : Summary", 10, 23, 33);
  for (int k = 0; k < m_maxttBits; ++k) {
    hname = fmt::format("BX.diff.PAC-TTU.bit.{}", m_ttBits[k]);
    m_bxDistDiffPac[k] = ibooker.book1D(hname, "Timing difference between PAC and TTU", 7, -3, 3);

    hname = fmt::format("BX.diff.DT-TTU.bit.{}", m_ttBits[k]);
    m_bxDistDiffDt[k] = ibooker.book1D(hname, "Timing difference between DT and TTU", 7, -3, 3);

    hname = fmt::format("Emu.Ttu.Compare.bit.{}", m_ttBits[k]);
    m_dataVsemulator[k] = ibooker.book1D(hname, "Comparison between emulator and TT decisions", 10, 0, 10);
  }
}
