/*
 * \file DTLocalTriggerBaseTask.cc
 *
 * \author C. Battilana - CIEMAT
 *
*/

#include "DQM/DTMonitorModule/src/DTLocalTriggerBaseTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// DT DQM
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

// DT trigger
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

//Root
#include "TH1.h"
#include "TAxis.h"

#include <sstream>
#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

class DTTPGCompareUnit {
public:
  DTTPGCompareUnit() { m_qual = -1; }
  ~DTTPGCompareUnit(){};

  void setTM(int qual, int bx) {
    m_qual = qual;
    m_BX = bx;
  }

  int qualTM() const { return m_qual; }

private:
  int m_qual;
  int m_BX;
};

DTLocalTriggerBaseTask::DTLocalTriggerBaseTask(const edm::ParameterSet& ps)
    : m_nEvents(0), m_nLumis(0), m_trigGeomUtils(nullptr) {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") << "[DTLocalTriggerBaseTask]: Constructor" << endl;

  m_tpMode = ps.getUntrackedParameter<bool>("testPulseMode");
  m_detailedAnalysis = ps.getUntrackedParameter<bool>("detailedAnalysis");

  m_targetBXTM = ps.getUntrackedParameter<int>("targetBXTM");
  m_bestAccRange = ps.getUntrackedParameter<int>("bestTrigAccRange");

  m_processTM = ps.getUntrackedParameter<bool>("processTM");
  m_processAB7 = ps.getUntrackedParameter<bool>("processAB7");

  m_tm_phiIn_Token = consumes<L1MuDTChambPhContainer>(ps.getUntrackedParameter<InputTag>("inputTagTMphIn"));
  m_tm_phiOut_Token = consumes<L1MuDTChambPhContainer>(ps.getUntrackedParameter<InputTag>("inputTagTMphOut"));
  m_tm_theta_Token = consumes<L1MuDTChambThContainer>(ps.getUntrackedParameter<InputTag>("inputTagTMth"));
  m_ab7_phi_Token = consumes<L1Phase2MuDTPhContainer>(ps.getUntrackedParameter<InputTag>("inputTagAB7"));

  if (m_processTM)
    m_types.push_back("TM");

  if (m_processAB7)
    m_types.push_back("AB7");

  if (m_tpMode) {
    topFolder("TM") = "DT/11-LocalTriggerTP-TM/";
    topFolder("AB7") = "DT/12-LocalTriggerTP-SliceTest/";
  } else {
    topFolder("TM") = "DT/03-LocalTrigger-TM/";
    topFolder("AB7") = "DT/04-LocalTrigger-SliceTest/";
  }

  m_params = ps;
}

DTLocalTriggerBaseTask::~DTLocalTriggerBaseTask() {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
      << "[DTLocalTriggerBaseTask]: analyzed " << m_nEvents << " events" << endl;
  if (m_trigGeomUtils)
    delete m_trigGeomUtils;
}

void DTLocalTriggerBaseTask::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& context) {
  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  m_nEventMonitor = ibooker.bookFloat("nProcessedEventsTrigger");
  for (int wh = -2; wh < 3; ++wh) {
    for (int stat = 1; stat < 5; ++stat) {
      for (int sect = 1; sect < 13; ++sect) {
        bookHistos(ibooker, DTChamberId(wh, stat, sect));
      }
    }
  }
}

void DTLocalTriggerBaseTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  m_nEventsInLS = 0;
  m_nLumis++;
  int resetCycle = m_params.getUntrackedParameter<int>("ResetCycle");

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
      << "[DTLocalTriggerBaseTask]: Begin of LS transition" << endl;

  if (m_nLumis % resetCycle == 0)
    for (auto& histosInChamb : m_chamberHistos)
      for (auto& histo : histosInChamb.second)
        histo.second->Reset();
}

void DTLocalTriggerBaseTask::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") << "[DTLocalTriggerBaseTask]: End of LS transition" << endl;

  for (auto& trendHisto : m_trendHistos)
    trendHisto.second->updateTimeSlot(lumiSeg.luminosityBlock(), m_nEventsInLS);
}

void DTLocalTriggerBaseTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") << "[DTLocalTriggerBaseTask]: BeginRun" << endl;

  ESHandle<DTGeometry> geom;
  context.get<MuonGeometryRecord>().get(geom);
  m_trigGeomUtils = new DTTrigGeomUtils(geom);
}

void DTLocalTriggerBaseTask::analyze(const edm::Event& e, const edm::EventSetup& c) {
  m_nEvents++;
  m_nEventsInLS++;
  m_nEventMonitor->Fill(m_nEvents);

  m_compMapIn.clear();
  m_compMapOut.clear();

  Handle<L1MuDTChambPhContainer> phiInTrigsTM;
  Handle<L1MuDTChambPhContainer> phiOutTrigsTM;
  Handle<L1MuDTChambThContainer> thetaTrigsTM;
  Handle<L1Phase2MuDTPhContainer> phiTrigsAB7;

  if (m_processTM) {
    e.getByToken(m_tm_phiIn_Token, phiInTrigsTM);
    e.getByToken(m_tm_phiOut_Token, phiOutTrigsTM);
    e.getByToken(m_tm_theta_Token, thetaTrigsTM);

    if (phiInTrigsTM.isValid() && phiOutTrigsTM.isValid() && thetaTrigsTM.isValid()) {
      runTMAnalysis(phiInTrigsTM->getContainer(), phiOutTrigsTM->getContainer(), thetaTrigsTM->getContainer());
    } else {
      LogVerbatim("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
          << "[DTLocalTriggerBaseTask]: one or more TM tokens not valid!" << endl;
      return;
    }
  }

  if (m_processAB7) {
    e.getByToken(m_ab7_phi_Token, phiTrigsAB7);

    if (phiTrigsAB7.isValid()) {
      runAB7Analysis(phiTrigsAB7->getContainer());
    } else {
      LogVerbatim("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
          << "[DTLocalTriggerBaseTask]: AB7 token not valid!" << endl;
    }
  }
}

void DTLocalTriggerBaseTask::bookHistos(DQMStore::IBooker& ibooker, const DTChamberId& dtCh) {
  uint32_t rawId = dtCh.rawId();

  stringstream wheel;
  wheel << dtCh.wheel();
  stringstream station;
  station << dtCh.station();
  stringstream sector;
  sector << dtCh.sector();

  map<string, int> minBX;
  map<string, int> maxBX;

  minBX["TM"] = m_params.getUntrackedParameter<int>("minBXTM");
  maxBX["TM"] = m_params.getUntrackedParameter<int>("maxBXTM");
  minBX["AB7"] = m_params.getUntrackedParameter<int>("minBXAB7");
  maxBX["AB7"] = m_params.getUntrackedParameter<int>("maxBXAB7");

  string chTag = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
  string labelInOut = "";

  for (const auto& type : m_types) {
    LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
        << "[DTLocalTriggerBaseTask]: booking histos for " << topFolder(type) << "Wheel" << wheel.str() << "/Sector"
        << sector.str() << "/Station" << station.str() << endl;

    if (type == "AB7" && (dtCh.wheel() != 2 || dtCh.sector() != 12))
      continue;

    vector<string> plotLabels;
    vector<string> folderLabels;

    if (type == "TM") {
      plotLabels.push_back("_In");
      plotLabels.push_back("_Out");
      folderLabels.push_back("/LocalTriggerPhiIn");
      folderLabels.push_back("/LocalTriggerPhiOut");
    }
    if (type == "AB7") {
      plotLabels.push_back("");
      folderLabels.push_back("/LocalTriggerPhi");
    }

    for (size_t iLabel = 0; iLabel < plotLabels.size(); ++iLabel) {
      // Book Phi View Related Plots

      auto plotLabel = plotLabels.at(iLabel);
      ibooker.setCurrentFolder(topFolder(type) + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" +
                               station.str() + folderLabels.at(iLabel));

      int nQualities = type == "AB7" ? 11 : 7;

      string histoTag = type + "_BXvsQual" + plotLabel;
      m_chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                        "BX vs trigger quality",
                                                        nQualities,
                                                        -0.5,
                                                        nQualities - 0.5,
                                                        (int)(maxBX[type] - minBX[type] + 1),
                                                        minBX[type] - .5,
                                                        maxBX[type] + .5);
      if (type == "AB7")
        setQLabelsPh2((m_chamberHistos[rawId])[histoTag], 1);
      else
        setQLabels((m_chamberHistos[rawId])[histoTag], 1);

      if (!m_tpMode && !(type == "AB7")) {
        histoTag = type + "_BestQual" + plotLabel;
        m_chamberHistos[rawId][histoTag] =
            ibooker.book1D(histoTag + chTag, "Trigger quality of best primitives", 7, -0.5, 6.5);
        setQLabels(m_chamberHistos[rawId][histoTag], 1);

        histoTag = type + "_Flag1stvsQual" + plotLabel;
        m_chamberHistos[dtCh.rawId()][histoTag] =
            ibooker.book2D(histoTag + chTag, "1st/2nd trig flag vs quality", 7, -0.5, 6.5, 2, -0.5, 1.5);
        setQLabels(m_chamberHistos[rawId][histoTag], 1);

        histoTag = type + "_FlagUpDownvsQual" + plotLabel;
        m_chamberHistos[dtCh.rawId()][histoTag] =
            ibooker.book2D(histoTag + chTag, "Up/Down trig flag vs quality", 7, -0.5, 6.5, 2, -0.5, 1.5);
        setQLabels(m_chamberHistos[rawId][histoTag], 1);
      }

      if (type == "TM") {
        float minPh, maxPh;
        int nBinsPh;
        m_trigGeomUtils->phiRange(dtCh, minPh, maxPh, nBinsPh);

        histoTag = type + "_QualvsPhirad" + plotLabel;
        m_chamberHistos[rawId][histoTag] =
            ibooker.book2D(histoTag + chTag, "Trigger quality vs local position", nBinsPh, minPh, maxPh, 7, -0.5, 6.5);
        setQLabels(m_chamberHistos[rawId][histoTag], 2);

        if (plotLabel == "_Out" && !m_tpMode) {
          histoTag = type + "_RPCBitvsQual" + plotLabel;
          m_chamberHistos[rawId][histoTag] =
              ibooker.book2D(histoTag + chTag, "RPC bit vs DT trigger quality", 9, -1.5, 7.5, 3, -0.5, 2.5);
          //setQLabels((m_chamberHistos[dtCh.rawId()])[histoTag], 2);
        }

        if (m_detailedAnalysis && !m_tpMode) {
          histoTag = type + "_QualvsPhibend" + plotLabel;
          m_chamberHistos[rawId][histoTag] =
              ibooker.book2D(histoTag + chTag, "Trigger quality vs local direction", 200, -40., 40., 7, -0.5, 6.5);
          setQLabels((m_chamberHistos[dtCh.rawId()])[histoTag], 2);
        }
      }
    }

    // Book Theta View Related Plots
    ibooker.setCurrentFolder(topFolder(type) + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" +
                             station.str() + "/LocalTriggerTheta");

    string histoTag = "";
    if (type == "TM" && dtCh.station() != 4) {
      histoTag = type + "_PositionvsBX";
      m_chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                        "Theta trigger position vs BX",
                                                        (int)(maxBX[type] - minBX[type] + 1),
                                                        minBX[type] - .5,
                                                        maxBX[type] + .5,
                                                        7,
                                                        -0.5,
                                                        6.5);
      histoTag = type + "_PositionvsQual";
      m_chamberHistos[rawId][histoTag] =
          ibooker.book2D(histoTag + chTag, "Theta trigger position vs quality", 3, 0.5, 3.5, 7, -0.5, 6.5);
      setQLabelsTheta(m_chamberHistos[rawId][histoTag], 1);
      histoTag = type + "_ThetaBXvsQual";
      m_chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                        "BX vs trigger quality",
                                                        3,
                                                        0.5,
                                                        3.5,
                                                        (int)(maxBX[type] - minBX[type] + 1),
                                                        minBX[type] - .5,
                                                        maxBX[type] + .5);
      setQLabelsTheta(m_chamberHistos[rawId][histoTag], 1);
    }
  }
}

void DTLocalTriggerBaseTask::runTMAnalysis(std::vector<L1MuDTChambPhDigi> const* phInTrigs,
                                           std::vector<L1MuDTChambPhDigi> const* phOutTrigs,
                                           std::vector<L1MuDTChambThDigi> const* thTrigs) {
  vector<L1MuDTChambPhDigi>::const_iterator iph = phInTrigs->begin();
  vector<L1MuDTChambPhDigi>::const_iterator iphe = phInTrigs->end();

  for (; iph != iphe; ++iph) {
    int wh = iph->whNum();
    int sec = iph->scNum() + 1;  // B(O)MTF->DT Convention
    int st = iph->stNum();
    int qual = iph->code();
    int is1st = iph->Ts2Tag() ? 1 : 0;
    int bx = iph->bxNum() - is1st;
    int updown = iph->UpDownTag();

    if (qual < 0 || qual > 6)
      continue;  // Check that quality is in a valid range

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    float pos = m_trigGeomUtils->trigPos(&(*iph));
    float dir = m_trigGeomUtils->trigDir(&(*iph));

    if (abs(bx - m_targetBXTM) <= m_bestAccRange && m_compMapIn[rawId].qualTM() <= qual)
      m_compMapIn[rawId].setTM(qual, bx);

    map<string, MonitorElement*>& innerME = m_chamberHistos[rawId];
    if (m_tpMode) {
      innerME["TM_BXvsQual_In"]->Fill(qual, bx);       // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_QualvsPhirad_In"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view
    } else {
      innerME["TM_BXvsQual_In"]->Fill(qual, bx);              // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_Flag1stvsQual_In"]->Fill(qual, is1st);      // SM Qual 1st/2nd track flag Phi view
      innerME["TM_FlagUpDownvsQual_In"]->Fill(qual, updown);  // SM Qual Up/Down track flag Phi view
      if (!is1st)
        innerME["TM_QualvsPhirad_In"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view ONLY for 1st tracks
      if (m_detailedAnalysis) {
        innerME["TM_QualvsPhibend_In"]->Fill(dir, qual);  // SM Qual vs bending Phi view
      }
    }
  }

  iph = phOutTrigs->begin();
  iphe = phOutTrigs->end();

  for (; iph != iphe; ++iph) {
    int wh = iph->whNum();
    int sec = iph->scNum() + 1;  // B(O)MTF->DT Convention
    int st = iph->stNum();
    int qual = iph->code();
    int is1st = iph->Ts2Tag() ? 1 : 0;
    int rpcBit = iph->RpcBit();
    int bx = iph->bxNum() - is1st;
    int updown = iph->UpDownTag();
    if (qual < 0 || qual > 6)
      continue;  // Check that quality is in a valid range

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    float pos = m_trigGeomUtils->trigPos(&(*iph));
    float dir = m_trigGeomUtils->trigDir(&(*iph));

    if (abs(bx - m_targetBXTM) <= m_bestAccRange && m_compMapOut[rawId].qualTM() <= qual)
      m_compMapOut[rawId].setTM(qual, bx);

    map<string, MonitorElement*>& innerME = m_chamberHistos[rawId];
    if (m_tpMode) {
      innerME["TM_BXvsQual_Out"]->Fill(qual, bx);       // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_QualvsPhirad_Out"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view
    } else {
      innerME["TM_BXvsQual_Out"]->Fill(qual, bx);              // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_RPCBitvsQual_Out"]->Fill(qual, rpcBit);      // SM RPC bitvs Qual Phi view
      innerME["TM_Flag1stvsQual_Out"]->Fill(qual, is1st);      // SM Qual 1st/2nd track flag Phi view
      innerME["TM_FlagUpDownvsQual_Out"]->Fill(qual, updown);  // SM Qual Up/Down track flag Phi view

      if (!is1st)
        innerME["TM_QualvsPhirad_Out"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view ONLY for 1st tracks
      if (m_detailedAnalysis) {
        innerME["TM_QualvsPhibend_Out"]->Fill(dir, qual);  // SM Qual vs bending Phi view
      }
    }
  }

  vector<L1MuDTChambThDigi>::const_iterator ith = thTrigs->begin();
  vector<L1MuDTChambThDigi>::const_iterator ithe = thTrigs->end();

  for (; ith != ithe; ++ith) {
    int wh = ith->whNum();
    int sec = ith->scNum() + 1;  // B(O)MTF -> DT Convention
    int st = ith->stNum();
    int bx = ith->bxNum();

    int thcode[7];

    for (int pos = 0; pos < 7; pos++) {
      thcode[pos] = ith->code(pos);
      if (ith->position(pos) == 0 && ith->quality(pos) == 1)
        thcode[pos] = 3;
    }

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    map<string, MonitorElement*>& innerME = m_chamberHistos[rawId];

    for (int pos = 0; pos < 7; pos++)
      if (thcode[pos] > 0) {                                   //Fired
        innerME["TM_PositionvsBX"]->Fill(bx, pos);             // SM BX vs Position Theta view
        innerME["TM_PositionvsQual"]->Fill(thcode[pos], pos);  //code = pos + qual; so 0, 1, 2 for 0, L, H resp.
        innerME["TM_ThetaBXvsQual"]->Fill(thcode[pos], bx);    //code = pos + qual; so 0, 1, 2 for 0, L, H resp.
      }
  }
  // Fill Quality plots with best TM triggers (phi view In)
  if (!m_tpMode) {
    for (auto& comp : m_compMapIn) {
      int bestQual = comp.second.qualTM();
      if (bestQual > -1)
        m_chamberHistos[comp.first]["TM_BestQual_In"]->Fill(bestQual);  // SM Best Qual Trigger Phi view
    }
  }

  // Fill Quality plots with best TM triggers (phi view Out)
  if (!m_tpMode) {
    for (auto& comp : m_compMapOut) {
      int bestQual = comp.second.qualTM();
      if (bestQual > -1)
        m_chamberHistos[comp.first]["TM_BestQual_Out"]->Fill(bestQual);  // SM Best Qual Trigger Phi view
    }
  }
}

void DTLocalTriggerBaseTask::runAB7Analysis(std::vector<L1Phase2MuDTPhDigi> const* phTrigs) {
  vector<L1Phase2MuDTPhDigi>::const_iterator iph = phTrigs->begin();
  vector<L1Phase2MuDTPhDigi>::const_iterator iphe = phTrigs->end();
  for (; iph != iphe; ++iph) {
    int wh = iph->whNum();
    int sec = iph->scNum() + 1;  // B(O)MTF->DT Convention
    int st = iph->stNum();
    int qual = iph->quality();
    int bx = iph->bxNum();

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    map<string, MonitorElement*>& innerME = m_chamberHistos[rawId];
    innerME["AB7_BXvsQual"]->Fill(qual, bx);
  }
}

void DTLocalTriggerBaseTask::setQLabels(MonitorElement* me, short int iaxis) {
  TH1* histo = me->getTH1();
  if (!histo)
    return;

  TAxis* axis = nullptr;
  if (iaxis == 1) {
    axis = histo->GetXaxis();
  } else if (iaxis == 2) {
    axis = histo->GetYaxis();
  }
  if (!axis)
    return;

  string labels[7] = {"LI", "LO", "HI", "HO", "LL", "HL", "HH"};
  int istart = axis->GetXmin() < -1 ? 2 : 1;
  for (int i = 0; i < 7; i++) {
    axis->SetBinLabel(i + istart, labels[i].c_str());
  }
}

void DTLocalTriggerBaseTask::setQLabelsTheta(MonitorElement* me, short int iaxis) {
  TH1* histo = me->getTH1();
  if (!histo)
    return;

  TAxis* axis = nullptr;
  if (iaxis == 1) {
    axis = histo->GetXaxis();
  } else if (iaxis == 2) {
    axis = histo->GetYaxis();
  }
  if (!axis)
    return;

  string labels[3] = {"L", "H", "err"};
  int istart = axis->GetXmin() < -1 ? 2 : 1;
  for (int i = 0; i < 3; i++) {
    axis->SetBinLabel(i + istart, labels[i].c_str());
  }
}

void DTLocalTriggerBaseTask::setQLabelsPh2(MonitorElement* me, short int iaxis) {
  TH1* histo = me->getTH1();
  if (!histo)
    return;

  TAxis* axis = nullptr;
  if (iaxis == 1) {
    axis = histo->GetXaxis();
  } else if (iaxis == 2) {
    axis = histo->GetYaxis();
  }
  if (!axis)
    return;

  string labels[11] = {"", "L only", "L multiple", "H only", "H multiple", "3+2", "LL", "4+2", "HL", "HH", ""};
  int istart = axis->GetXmin() < -1 ? 2 : 1;
  for (int i = 0; i < 11; i++) {
    axis->SetBinLabel(i + istart, labels[i].c_str());
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
