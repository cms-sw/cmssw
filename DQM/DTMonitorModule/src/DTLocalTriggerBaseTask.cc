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
  DTTPGCompareUnit() { theQual = -1; }
  ~DTTPGCompareUnit(){};

  void setTM(int qual, int bx) {
    theQual = qual;
    theBX = bx;
  }

  int qualTM() const { return theQual; }

private:
  int theQual;
  int theBX;
};

DTLocalTriggerBaseTask::DTLocalTriggerBaseTask(const edm::ParameterSet& ps)
    : nEvents(0), nLumis(0), theTrigGeomUtils(nullptr) {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") << "[DTLocalTriggerBaseTask]: Constructor" << endl;

  tpMode = ps.getUntrackedParameter<bool>("testPulseMode");
  detailedAnalysis = ps.getUntrackedParameter<bool>("detailedAnalysis");

  targetBXTM = ps.getUntrackedParameter<int>("targetBXTM");
  bestAccRange = ps.getUntrackedParameter<int>("bestTrigAccRange");

  processTM = ps.getUntrackedParameter<bool>("processTM");

  tm_phiIn_Token_ = consumes<L1MuDTChambPhContainer>(ps.getUntrackedParameter<InputTag>("inputTagTMphIn"));
  tm_phiOut_Token_ = consumes<L1MuDTChambPhContainer>(ps.getUntrackedParameter<InputTag>("inputTagTMphOut"));
  tm_theta_Token_ = consumes<L1MuDTChambThContainer>(ps.getUntrackedParameter<InputTag>("inputTagTMth"));

  if (processTM)
    theTypes.push_back("TM");

  if (tpMode) {
    topFolder("TM") = "DT/11-LocalTriggerTP-TM/";
  } else {
    topFolder("TM") = "DT/03-LocalTrigger-TM/";
  }

  theParams = ps;
}

DTLocalTriggerBaseTask::~DTLocalTriggerBaseTask() {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
      << "[DTLocalTriggerBaseTask]: analyzed " << nEvents << " events" << endl;
  if (theTrigGeomUtils) {
    delete theTrigGeomUtils;
  }
}

void DTLocalTriggerBaseTask::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& context) {
  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsTrigger");
  for (int wh = -2; wh < 3; ++wh) {
    for (int stat = 1; stat < 5; ++stat) {
      for (int sect = 1; sect < 13; ++sect) {
        bookHistos(ibooker, DTChamberId(wh, stat, sect));
      }
    }
  }
}

void DTLocalTriggerBaseTask::beginLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  nEventsInLS = 0;
  nLumis++;
  int resetCycle = theParams.getUntrackedParameter<int>("ResetCycle");

  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
      << "[DTLocalTriggerBaseTask]: Begin of LS transition" << endl;

  if (nLumis % resetCycle == 0) {
    map<uint32_t, map<string, MonitorElement*> >::const_iterator chambIt = chamberHistos.begin();
    map<uint32_t, map<string, MonitorElement*> >::const_iterator chambEnd = chamberHistos.end();
    for (; chambIt != chambEnd; ++chambIt) {
      map<string, MonitorElement*>::const_iterator histoIt = chambIt->second.begin();
      map<string, MonitorElement*>::const_iterator histoEnd = chambIt->second.end();
      for (; histoIt != histoEnd; ++histoIt) {
        histoIt->second->Reset();
      }
    }
  }
}

void DTLocalTriggerBaseTask::endLuminosityBlock(const LuminosityBlock& lumiSeg, const EventSetup& context) {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") << "[DTLocalTriggerBaseTask]: End of LS transition" << endl;

  map<uint32_t, DTTimeEvolutionHisto*>::const_iterator chambIt = trendHistos.begin();
  map<uint32_t, DTTimeEvolutionHisto*>::const_iterator chambEnd = trendHistos.end();
  for (; chambIt != chambEnd; ++chambIt) {
    cout << "updating time slot in DTLocalTriggerBaseTask" << endl;
    chambIt->second->updateTimeSlot(lumiSeg.luminosityBlock(), nEventsInLS);
    cout << "updated time slot in DTLocalTriggerBaseTask" << endl;
  }
}

void DTLocalTriggerBaseTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {
  LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask") << "[DTLocalTriggerBaseTask]: BeginRun" << endl;

  ESHandle<DTGeometry> theGeom;
  context.get<MuonGeometryRecord>().get(theGeom);
  theTrigGeomUtils = new DTTrigGeomUtils(theGeom);
}

void DTLocalTriggerBaseTask::analyze(const edm::Event& e, const edm::EventSetup& c) {
  nEvents++;
  nEventsInLS++;
  nEventMonitor->Fill(nEvents);

  theCompMapIn.clear();
  theCompMapOut.clear();

  Handle<L1MuDTChambPhContainer> phiInTrigsTM;
  Handle<L1MuDTChambPhContainer> phiOutTrigsTM;
  Handle<L1MuDTChambThContainer> thetaTrigsTM;

  if (processTM) {
    InputTag inputTagTM = theParams.getUntrackedParameter<InputTag>("inputTagTM");

    e.getByToken(tm_phiIn_Token_, phiInTrigsTM);
    e.getByToken(tm_phiOut_Token_, phiOutTrigsTM);
    e.getByToken(tm_theta_Token_, thetaTrigsTM);

    if (phiInTrigsTM.isValid() && phiOutTrigsTM.isValid() && thetaTrigsTM.isValid()) {
      runTMAnalysis(phiInTrigsTM->getContainer(), phiOutTrigsTM->getContainer(), thetaTrigsTM->getContainer());
    } else {
      LogVerbatim("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
          << "[DTLocalTriggerBaseTask]: one or more TM handles for Input Tag " << inputTagTM << " not found!" << endl;
      return;
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

  minBX["TM"] = theParams.getUntrackedParameter<int>("minBXTM");
  maxBX["TM"] = theParams.getUntrackedParameter<int>("maxBXTM");

  string chTag = "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();
  string labelInOut = "";

  vector<string>::const_iterator typeIt = theTypes.begin();
  vector<string>::const_iterator typeEnd = theTypes.end();

  for (; typeIt != typeEnd; ++typeIt) {
    LogTrace("DTDQM|DTMonitorModule|DTLocalTriggerBaseTask")
        << "[DTLocalTriggerBaseTask]: booking histos for " << topFolder((*typeIt)) << "Wheel" << wheel.str()
        << "/Sector" << sector.str() << "/Station" << station.str() << endl;

    for (int InOut = 0; InOut < 2; InOut++) {
      // Book Phi View Related Plots

      if (InOut == 0) {
        ibooker.setCurrentFolder(topFolder(*typeIt) + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" +
                                 station.str() + "/LocalTriggerPhiIn");
        labelInOut = "_In";
      } else if (InOut == 1) {
        ibooker.setCurrentFolder(topFolder(*typeIt) + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" +
                                 station.str() + "/LocalTriggerPhiOut");
        labelInOut = "_Out";
      }

      string histoTag = (*typeIt) + "_BXvsQual" + labelInOut;
      chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                      "BX vs trigger quality",
                                                      7,
                                                      -0.5,
                                                      6.5,
                                                      (int)(maxBX[(*typeIt)] - minBX[*typeIt] + 1),
                                                      minBX[*typeIt] - .5,
                                                      maxBX[*typeIt] + .5);
      setQLabels((chamberHistos[rawId])[histoTag], 1);

      if (!tpMode) {
        histoTag = (*typeIt) + "_BestQual" + labelInOut;
        chamberHistos[rawId][histoTag] =
            ibooker.book1D(histoTag + chTag, "Trigger quality of best primitives", 7, -0.5, 6.5);
        setQLabels(chamberHistos[rawId][histoTag], 1);

        histoTag = (*typeIt) + "_Flag1stvsQual" + labelInOut;
        chamberHistos[dtCh.rawId()][histoTag] =
            ibooker.book2D(histoTag + chTag, "1st/2nd trig flag vs quality", 7, -0.5, 6.5, 2, -0.5, 1.5);
        setQLabels(chamberHistos[rawId][histoTag], 1);

        histoTag = (*typeIt) + "_FlagUpDownvsQual" + labelInOut;
        chamberHistos[dtCh.rawId()][histoTag] =
            ibooker.book2D(histoTag + chTag, "Up/Down trig flag vs quality", 7, -0.5, 6.5, 2, -0.5, 1.5);
        setQLabels(chamberHistos[rawId][histoTag], 1);
      }

      if (*typeIt == "TM") {
        float minPh, maxPh;
        int nBinsPh;
        theTrigGeomUtils->phiRange(dtCh, minPh, maxPh, nBinsPh);

        histoTag = (*typeIt) + "_QualvsPhirad" + labelInOut;
        chamberHistos[rawId][histoTag] =
            ibooker.book2D(histoTag + chTag, "Trigger quality vs local position", nBinsPh, minPh, maxPh, 7, -0.5, 6.5);
        setQLabels(chamberHistos[rawId][histoTag], 2);

        if (detailedAnalysis && !tpMode) {
          histoTag = (*typeIt) + "_QualvsPhibend" + labelInOut;
          chamberHistos[rawId][histoTag] =
              ibooker.book2D(histoTag + chTag, "Trigger quality vs local direction", 200, -40., 40., 7, -0.5, 6.5);
          setQLabels((chamberHistos[dtCh.rawId()])[histoTag], 2);
        }
      }
    }  //InOut loop

    // Book Theta View Related Plots
    ibooker.setCurrentFolder(topFolder(*typeIt) + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" +
                             station.str() + "/LocalTriggerTheta");

    string histoTag = "";
    if ((*typeIt) == "TM" && dtCh.station() != 4) {
      histoTag = (*typeIt) + "_PositionvsBX";
      chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                      "Theta trigger position vs BX",
                                                      (int)(maxBX[(*typeIt)] - minBX[*typeIt] + 1),
                                                      minBX[*typeIt] - .5,
                                                      maxBX[*typeIt] + .5,
                                                      7,
                                                      -0.5,
                                                      6.5);
      histoTag = (*typeIt) + "_PositionvsQual";
      chamberHistos[rawId][histoTag] =
          ibooker.book2D(histoTag + chTag, "Theta trigger position vs quality", 2, 0.5, 2.5, 7, -0.5, 6.5);
      setQLabelsTheta(chamberHistos[rawId][histoTag], 1);
      histoTag = (*typeIt) + "_ThetaBXvsQual";
      chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                      "BX vs trigger quality",
                                                      2,
                                                      0.5,
                                                      2.5,
                                                      (int)(maxBX[(*typeIt)] - minBX[*typeIt] + 1),
                                                      minBX[*typeIt] - .5,
                                                      maxBX[*typeIt] + .5);
      setQLabelsTheta(chamberHistos[rawId][histoTag], 1);
      //      histoTag = (*typeIt) + "_ThetaBestQual";
      //      chamberHistos[rawId][histoTag] = ibooker.book1D(histoTag+chTag,
      //                              "Trigger quality of best primitives (theta)",2,0.5,2.5); // 0 = not fired, 1 = L, 2 = H
      //      setQLabelsTheta(chamberHistos[rawId][histoTag],1);
    } else {
      if (dtCh.station() != 4) {
        histoTag = (*typeIt) + "_ThetaBXvsQual";
        chamberHistos[rawId][histoTag] = ibooker.book2D(histoTag + chTag,
                                                        "BX vs trigger quality",
                                                        7,
                                                        -0.5,
                                                        6.5,
                                                        (int)(maxBX[(*typeIt)] - minBX[*typeIt] + 1),
                                                        minBX[*typeIt] - .5,
                                                        maxBX[*typeIt] + .5);
        setQLabels((chamberHistos[dtCh.rawId()])[histoTag], 1);

        histoTag = (*typeIt) + "_ThetaBestQual";
        chamberHistos[rawId][histoTag] =
            ibooker.book1D(histoTag + chTag, "Trigger quality of best primitives (theta)", 7, -0.5, 6.5);
        setQLabels((chamberHistos[dtCh.rawId()])[histoTag], 1);
      }
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
    int sec = iph->scNum() + 1;  // DTTF->DT Convention
    int st = iph->stNum();
    int qual = iph->code();
    int is1st = iph->Ts2Tag() ? 1 : 0;
    int bx = iph->bxNum() - is1st;
    int updown = iph->UpDownTag();

    if (qual < 0 || qual > 6)
      continue;  // Check that quality is in a valid range

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    float pos = theTrigGeomUtils->trigPos(&(*iph));
    float dir = theTrigGeomUtils->trigDir(&(*iph));

    if (abs(bx - targetBXTM) <= bestAccRange && theCompMapIn[rawId].qualTM() <= qual)
      theCompMapIn[rawId].setTM(qual, bx);

    map<string, MonitorElement*>& innerME = chamberHistos[rawId];
    if (tpMode) {
      innerME["TM_BXvsQual_In"]->Fill(qual, bx);       // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_QualvsPhirad_In"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view
    } else {
      innerME["TM_BXvsQual_In"]->Fill(qual, bx);              // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_Flag1stvsQual_In"]->Fill(qual, is1st);      // SM Qual 1st/2nd track flag Phi view
      innerME["TM_FlagUpDownvsQual_In"]->Fill(qual, updown);  // SM Qual Up/Down track flag Phi view
      if (!is1st)
        innerME["TM_QualvsPhirad_In"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view ONLY for 1st tracks
      if (detailedAnalysis) {
        innerME["TM_QualvsPhibend_In"]->Fill(dir, qual);  // SM Qual vs bending Phi view
      }
    }
  }

  iph = phOutTrigs->begin();
  iphe = phOutTrigs->end();

  for (; iph != iphe; ++iph) {
    int wh = iph->whNum();
    int sec = iph->scNum() + 1;  // DTTF->DT Convention
    int st = iph->stNum();
    int qual = iph->code();
    int is1st = iph->Ts2Tag() ? 1 : 0;
    int bx = iph->bxNum() - is1st;
    int updown = iph->UpDownTag();
    if (qual < 0 || qual > 6)
      continue;  // Check that quality is in a valid range

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    float pos = theTrigGeomUtils->trigPos(&(*iph));
    float dir = theTrigGeomUtils->trigDir(&(*iph));

    if (abs(bx - targetBXTM) <= bestAccRange && theCompMapOut[rawId].qualTM() <= qual)
      theCompMapOut[rawId].setTM(qual, bx);

    map<string, MonitorElement*>& innerME = chamberHistos[rawId];
    if (tpMode) {
      innerME["TM_BXvsQual_Out"]->Fill(qual, bx);       // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_QualvsPhirad_Out"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view
    } else {
      innerME["TM_BXvsQual_Out"]->Fill(qual, bx);              // SM BX vs Qual Phi view (1st tracks)
      innerME["TM_Flag1stvsQual_Out"]->Fill(qual, is1st);      // SM Qual 1st/2nd track flag Phi view
      innerME["TM_FlagUpDownvsQual_Out"]->Fill(qual, updown);  // SM Qual Up/Down track flag Phi view

      if (!is1st)
        innerME["TM_QualvsPhirad_Out"]->Fill(pos, qual);  // SM Qual vs radial angle Phi view ONLY for 1st tracks
      if (detailedAnalysis) {
        innerME["TM_QualvsPhibend_Out"]->Fill(dir, qual);  // SM Qual vs bending Phi view
      }
    }
  }

  vector<L1MuDTChambThDigi>::const_iterator ith = thTrigs->begin();
  vector<L1MuDTChambThDigi>::const_iterator ithe = thTrigs->end();

  for (; ith != ithe; ++ith) {
    int wh = ith->whNum();
    int sec = ith->scNum() + 1;  // DTTF -> DT Convention
    int st = ith->stNum();
    int bx = ith->bxNum();

    int thcode[7];

    for (int pos = 0; pos < 7; pos++)
      thcode[pos] = ith->code(pos);

    DTChamberId dtChId(wh, st, sec);
    uint32_t rawId = dtChId.rawId();

    map<string, MonitorElement*>& innerME = chamberHistos[rawId];

    for (int pos = 0; pos < 7; pos++)
      if (thcode[pos] > 0) {                                   //Fired
        innerME["TM_PositionvsBX"]->Fill(bx, pos);             // SM BX vs Position Theta view
        innerME["TM_PositionvsQual"]->Fill(thcode[pos], pos);  //code = pos + qual; so 0, 1, 2 for 0, L, H resp.
        innerME["TM_ThetaBXvsQual"]->Fill(thcode[pos], bx);    //code = pos + qual; so 0, 1, 2 for 0, L, H resp.
      }
  }
  // Fill Quality plots with best TM triggers (phi view In)
  if (!tpMode) {
    map<uint32_t, DTTPGCompareUnit>::const_iterator compIt = theCompMapIn.begin();
    map<uint32_t, DTTPGCompareUnit>::const_iterator compEnd = theCompMapIn.end();
    for (; compIt != compEnd; ++compIt) {
      int bestQual = compIt->second.qualTM();
      if (bestQual > -1)
        chamberHistos[compIt->first]["TM_BestQual_In"]->Fill(bestQual);  // SM Best Qual Trigger Phi view
    }
  }

  // Fill Quality plots with best TM triggers (phi view Out)
  if (!tpMode) {
    map<uint32_t, DTTPGCompareUnit>::const_iterator compIt = theCompMapOut.begin();
    map<uint32_t, DTTPGCompareUnit>::const_iterator compEnd = theCompMapOut.end();
    for (; compIt != compEnd; ++compIt) {
      int bestQual = compIt->second.qualTM();
      if (bestQual > -1)
        chamberHistos[compIt->first]["TM_BestQual_Out"]->Fill(bestQual);  // SM Best Qual Trigger Phi view
    }
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

  string labels[2] = {"L", "H"};
  int istart = axis->GetXmin() < -1 ? 2 : 1;
  for (int i = 0; i < 2; i++) {
    axis->SetBinLabel(i + istart, labels[i].c_str());
  }
}
// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
