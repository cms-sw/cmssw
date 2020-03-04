/*
 *  See header file for a description of this class.
 *
 *  \author C. Battilana - CIEMAT
 *  \author P. Bellan - INFN PD
 *  \author A. Branca = INFN PD
 */

#include "DQM/DTMonitorClient/src/DTDCSByLumiSummary.h"
#include "DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <string>

using namespace std;
using namespace edm;

DTDCSByLumiSummary::DTDCSByLumiSummary(const ParameterSet& pset) { bookingdone = false; }

DTDCSByLumiSummary::~DTDCSByLumiSummary() {}

void DTDCSByLumiSummary::beginRun(const edm::Run& r, const edm::EventSetup& setup) {}

void DTDCSByLumiSummary::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                               DQMStore::IGetter& igetter,
                                               edm::LuminosityBlock const& lumi,
                                               edm::EventSetup const& setup) {
  if (!bookingdone) {
    ibooker.setCurrentFolder("DT/EventInfo/DCSContents");

    globalHVSummary = ibooker.book2D("HVGlbSummary", "HV Status Summary", 1, 1, 13, 5, -2, 3);
    globalHVSummary->setAxisTitle("Sectors", 1);
    globalHVSummary->setAxisTitle("Wheel", 2);

    {
      auto scope = DQMStore::IBooker::UseLumiScope(ibooker);
      totalDCSFraction = ibooker.bookFloat("DTDCSSummary");
      for (int wh = -2; wh <= 2; wh++) {
        stringstream wheel_str;
        wheel_str << wh;

        MonitorElement* FractionWh = ibooker.bookFloat("DT_Wheel" + wheel_str.str());

        totalDCSFractionWh.push_back(FractionWh);
      }
    }

    globalHVSummary->Reset();

    // CB LumiFlag marked products are reset on LS boundaries
    totalDCSFraction->Reset();

    for (int wh = -2; wh <= 2; wh++) {
      totalDCSFractionWh[wh + 2]->Reset();
    }
  }
  bookingdone = true;

  // Get the by lumi product plot from the task
  int lumiNumber = lumi.id().luminosityBlock();

  bool null_pointer_histo(false);

  std::vector<float> wh_activeFrac;

  for (int wh = -2; wh <= 2; wh++) {
    stringstream wheel_str;
    wheel_str << wh;

    string hActiveUnitsPath = "DT/EventInfo/DCSContents/hActiveUnits" + wheel_str.str();

    MonitorElement* hActiveUnits = igetter.get(hActiveUnitsPath);

    if (hActiveUnits) {
      float activeFrac = static_cast<float>(hActiveUnits->getBinContent(2)) /  // CB 2nd bin is # of active channels
                         hActiveUnits->getBinContent(1);  // first bin is overall number of channels

      if (activeFrac < 0.)
        activeFrac = -1;

      wh_activeFrac.push_back(activeFrac);

      // Fill by lumi Certification ME
      totalDCSFraction->Fill(activeFrac);
      totalDCSFractionWh[wh + 2]->Fill(activeFrac);

    } else {
      LogTrace("DTDQM|DTMonitorClient|DTDCSByLumiSummary")
          << "[DTDCSByLumiSummary]: got null pointer retrieving histo at :" << hActiveUnitsPath << " for lumi # "
          << lumiNumber << "client operation not performed." << endl;

      null_pointer_histo = true;
    }

  }  // end loop on wheels

  if (!null_pointer_histo)
    dcsFracPerLumi[lumiNumber] = wh_activeFrac;  // Fill map to be used to compute trend plots

  // CB LumiFlag marked products are reset on LS boundaries
  totalDCSFraction->Reset();

  for (int wh = -2; wh <= 2; wh++) {
    totalDCSFractionWh[wh + 2]->Reset();
  }
}

void DTDCSByLumiSummary::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  // Book trend plots ME & loop on map to fill it with by lumi info
  map<int, std::vector<float> >::const_iterator fracPerLumiIt = dcsFracPerLumi.begin();
  map<int, std::vector<float> >::const_iterator fracPerLumiEnd = dcsFracPerLumi.end();

  if (fracPerLumiIt != fracPerLumiEnd) {
    int fLumi = dcsFracPerLumi.begin()->first;
    int lLumi = dcsFracPerLumi.rbegin()->first;

    ibooker.setCurrentFolder("DT/EventInfo/DCSContents");

    int nLumis = lLumi - fLumi + 1.;

    // trend plots
    for (int wh = -2; wh <= 2; wh++) {
      stringstream wheel_str;
      wheel_str << wh;

      DTTimeEvolutionHisto* trend;

      trend = new DTTimeEvolutionHisto(ibooker,
                                       "hDCSFracTrendWh" + wheel_str.str(),
                                       "Fraction of DT-HV ON Wh" + wheel_str.str(),
                                       nLumis,
                                       fLumi,
                                       1,
                                       false,
                                       2);

      hDCSFracTrend.push_back(trend);
    }
  }

  float goodLSperWh[5] = {0, 0, 0, 0, 0};
  float badLSperWh[5] = {0, 0, 0, 0, 0};

  // fill trend plots and save infos for summaryPlot
  for (; fracPerLumiIt != fracPerLumiEnd; ++fracPerLumiIt) {
    for (int wh = -2; wh <= 2; wh++) {
      std::vector<float> activeFracPerWh;
      activeFracPerWh = fracPerLumiIt->second;

      hDCSFracTrend[wh + 2]->setTimeSlotValue(activeFracPerWh[wh + 2], fracPerLumiIt->first);

      if (activeFracPerWh[wh + 2] > 0) {  // we do not count the lumi were the DTs are off (no real problem),
        // even if this can happen in the middle of a run (real problem: to be fixed)
        if (activeFracPerWh[wh + 2] > 0.9)
          goodLSperWh[wh + 2]++;
        else {
          badLSperWh[wh + 2]++;
        }
      } else {  // there is no HV value OR all channels OFF
        if (activeFracPerWh[wh + 2] < 0)
          badLSperWh[wh + 2] = -1;  // if there were no HV values, activeFrac returning -1
      }
    }
  }

  // fill summaryPlot
  for (int wh = -2; wh <= 2; wh++) {
    if (goodLSperWh[wh + 2] != 0 || badLSperWh[wh + 2] == -1) {
      float r = badLSperWh[wh + 2] / fabs(goodLSperWh[wh + 2] + badLSperWh[wh + 2]);
      if (r > 0.5)
        globalHVSummary->Fill(1, wh, 0);
      else
        globalHVSummary->Fill(1, wh, 1);
      if (r == -1)
        globalHVSummary->Fill(1, wh, -1);

    } else {
      globalHVSummary->Fill(1, wh, 0);
    }
  }
}
