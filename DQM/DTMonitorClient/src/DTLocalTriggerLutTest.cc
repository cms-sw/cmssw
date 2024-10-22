/*
 *  See header file for a description of this class.
 *
 *  \author C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 */

// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerLutTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// Root
#include "TF1.h"
#include "TProfile.h"

//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;

DTLocalTriggerLutTest::DTLocalTriggerLutTest(const edm::ParameterSet& ps) {
  setConfig(ps, "DTLocalTriggerLut");
  baseFolderTM = "DT/03-LocalTrigger-TM/";
  thresholdPhiMean = ps.getUntrackedParameter<double>("thresholdPhiMean", 1.5);
  thresholdPhiRMS = ps.getUntrackedParameter<double>("thresholdPhiRMS", .5);
  thresholdPhibMean = ps.getUntrackedParameter<double>("thresholdPhibMean", 1.5);
  thresholdPhibRMS = ps.getUntrackedParameter<double>("thresholdPhibRMS", .8);
  doCorrStudy = ps.getUntrackedParameter<bool>("doCorrelationStudy", false);

  bookingdone = false;
}

DTLocalTriggerLutTest::~DTLocalTriggerLutTest() {}

void DTLocalTriggerLutTest::Bookings(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  vector<string>::const_iterator iTr = trigSources.begin();
  vector<string>::const_iterator trEnd = trigSources.end();
  vector<string>::const_iterator iHw = hwSources.begin();
  vector<string>::const_iterator hwEnd = hwSources.end();

  //Booking
  if (parameters.getUntrackedParameter<bool>("staticBooking", true)) {
    for (; iTr != trEnd; ++iTr) {
      trigSource = (*iTr);
      for (; iHw != hwEnd; ++iHw) {
        hwSource = (*iHw);
        // Loop over the TriggerUnits
        for (int wh = -2; wh <= 2; ++wh) {
          bookWheelHistos(ibooker, wh, "PhiResidualMean");
          bookWheelHistos(ibooker, wh, "PhiResidualRMS");
          bookWheelHistos(ibooker, wh, "PhibResidualMean");
          bookWheelHistos(ibooker, wh, "PhibResidualRMS");
          if (doCorrStudy) {
            bookWheelHistos(ibooker, wh, "PhiTkvsTrigSlope");
            bookWheelHistos(ibooker, wh, "PhiTkvsTrigIntercept");
            bookWheelHistos(ibooker, wh, "PhiTkvsTrigCorr");
            bookWheelHistos(ibooker, wh, "PhibTkvsTrigSlope");
            bookWheelHistos(ibooker, wh, "PhibTkvsTrigIntercept");
            bookWheelHistos(ibooker, wh, "PhibTkvsTrigCorr");
          }
        }
      }
    }
  }

  // Summary test histo booking (only static)
  for (iTr = trigSources.begin(); iTr != trEnd; ++iTr) {
    trigSource = (*iTr);
    for (iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw) {
      hwSource = (*iHw);
      // Loop over the TriggerUnits
      for (int wh = -2; wh <= 2; ++wh) {
        bookWheelHistos(ibooker, wh, "PhiLutSummary");
        bookWheelHistos(ibooker, wh, "PhibLutSummary");
      }

      bookCmsHistos(ibooker, "PhiLutSummary");
      bookCmsHistos(ibooker, "PhibLutSummary");
    }
  }

  bookingdone = true;
}

void DTLocalTriggerLutTest::beginRun(const edm::Run& r, const edm::EventSetup& c) {
  DTLocalTriggerBaseTest::beginRun(r, c);
}

void DTLocalTriggerLutTest::runClientDiagnostic(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (!bookingdone)
    Bookings(ibooker, igetter);

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr) {
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw) {
      hwSource = (*iHw);
      vector<const DTChamber*>::const_iterator chIt = muonGeom->chambers().begin();
      vector<const DTChamber*>::const_iterator chEnd = muonGeom->chambers().end();
      for (; chIt != chEnd; ++chIt) {
        DTChamberId chId((*chIt)->id());
        int wh = chId.wheel();
        int sect = chId.sector();
        int stat = chId.station();

        if (doCorrStudy) {
          // Perform Correlation Plots analysis (TM + segment Phi)

          TH2F* TrackPhitkvsPhitrig = getHisto<TH2F>(igetter.get(getMEName("PhitkvsPhitrig", "Segment", chId)));

          if (TrackPhitkvsPhitrig && TrackPhitkvsPhitrig->GetEntries() > 10) {
            // Fill client histos
            if (whME[wh].find(fullName("PhiTkvsTrigCorr")) == whME[wh].end()) {
              bookWheelHistos(ibooker, wh, "PhiTkvsTrigSlope");
              bookWheelHistos(ibooker, wh, "PhiTkvsTrigIntercept");
              bookWheelHistos(ibooker, wh, "PhiTkvsTrigCorr");
            }

            TProfile* PhitkvsPhitrigProf = TrackPhitkvsPhitrig->ProfileX();
            double phiInt = 0;
            double phiSlope = 0;
            double phiCorr = 0;
            try {
              TF1 ffPhi("mypol1", "pol1");
              PhitkvsPhitrigProf->Fit(&ffPhi, "CQO");
              phiInt = ffPhi.GetParameter(0);
              phiSlope = ffPhi.GetParameter(1);
              phiCorr = TrackPhitkvsPhitrig->GetCorrelationFactor();
            } catch (cms::Exception& iException) {
              edm::LogError(category()) << "[" << testName << "Test]: Error fitting PhitkvsPhitrig for Wheel " << wh
                                        << " Sector " << sect << " Station " << stat;
            }

            std::map<std::string, MonitorElement*>& innerME = whME[wh];
            fillWhPlot(innerME.find(fullName("PhiTkvsTrigSlope"))->second, sect, stat, phiSlope - 1);
            fillWhPlot(innerME.find(fullName("PhiTkvsTrigIntercept"))->second, sect, stat, phiInt);
            fillWhPlot(innerME.find(fullName("PhiTkvsTrigCorr"))->second, sect, stat, phiCorr, false);
          }

          // Perform Correlation Plots analysis (TM + segment Phib)
          TH2F* TrackPhibtkvsPhibtrig = getHisto<TH2F>(igetter.get(getMEName("PhibtkvsPhibtrig", "Segment", chId)));

          if (stat != 3 && TrackPhibtkvsPhibtrig &&
              TrackPhibtkvsPhibtrig->GetEntries() > 10) {  // station 3 has no meaningful MB3 phi bending information

            // Fill client histos
            if (whME[wh].find(fullName("PhibTkvsTrigCorr")) == whME[wh].end()) {
              bookWheelHistos(ibooker, wh, "PhibTkvsTrigSlope");
              bookWheelHistos(ibooker, wh, "PhibTkvsTrigIntercept");
              bookWheelHistos(ibooker, wh, "PhibTkvsTrigCorr");
            }

            TProfile* PhibtkvsPhibtrigProf = TrackPhibtkvsPhibtrig->ProfileX();
            double phibInt = 0;
            double phibSlope = 0;
            double phibCorr = 0;
            try {
              TF1 ffPhib("ffPhib", "pol1");
              PhibtkvsPhibtrigProf->Fit(&ffPhib, "CQO");
              phibInt = ffPhib.GetParameter(0);
              phibSlope = ffPhib.GetParameter(1);
              phibCorr = TrackPhibtkvsPhibtrig->GetCorrelationFactor();
            } catch (cms::Exception& iException) {
              edm::LogError(category()) << "[" << testName << "Test]: Error fitting PhibtkvsPhibtrig for Wheel " << wh
                                        << " Sector " << sect << " Station " << stat;
            }

            std::map<std::string, MonitorElement*>& innerME = whME[wh];
            fillWhPlot(innerME.find(fullName("PhibTkvsTrigSlope"))->second, sect, stat, phibSlope - 1);
            fillWhPlot(innerME.find(fullName("PhibTkvsTrigIntercept"))->second, sect, stat, phibInt);
            fillWhPlot(innerME.find(fullName("PhibTkvsTrigCorr"))->second, sect, stat, phibCorr, false);
          }
        }

        // Make Phi Residual Summary

        TH1F* PhiResidual = getHisto<TH1F>(igetter.get(getMEName("PhiResidual", "Segment", chId)));
        int phiSummary = 1;

        if (PhiResidual && PhiResidual->GetEffectiveEntries() > 10) {
          // Fill client histos
          if (whME[wh].find(fullName("PhiResidualMean")) == whME[wh].end()) {
            bookWheelHistos(ibooker, wh, "PhiResidualMean");
            bookWheelHistos(ibooker, wh, "PhiResidualRMS");
          }

          double peak = PhiResidual->GetBinCenter(PhiResidual->GetMaximumBin());
          double phiMean = 0;
          double phiRMS = 0;
          try {
            TF1 ffPhi("ffPhi", "gaus");
            PhiResidual->Fit(&ffPhi, "CQO", "", peak - 5, peak + 5);
            phiMean = ffPhi.GetParameter(1);
            phiRMS = ffPhi.GetParameter(2);
          } catch (cms::Exception& iException) {
            edm::LogError(category()) << "[" << testName << "Test]: Error fitting PhiResidual for Wheel " << wh
                                      << " Sector " << sect << " Station " << stat;
          }

          std::map<std::string, MonitorElement*>& innerME = whME[wh];
          fillWhPlot(innerME.find(fullName("PhiResidualMean"))->second, sect, stat, phiMean);
          fillWhPlot(innerME.find(fullName("PhiResidualRMS"))->second, sect, stat, phiRMS);

          phiSummary = performLutTest(phiMean, phiRMS, thresholdPhiMean, thresholdPhiRMS);
        }
        fillWhPlot(whME[wh].find(fullName("PhiLutSummary"))->second, sect, stat, phiSummary);

        // Make Phib Residual Summary
        TH1F* PhibResidual = getHisto<TH1F>(igetter.get(getMEName("PhibResidual", "Segment", chId)));
        int phibSummary = stat == 3 ? 0 : 1;  // station 3 has no meaningful MB3 phi bending information

        if (stat != 3 && PhibResidual &&
            PhibResidual->GetEffectiveEntries() > 10) {  // station 3 has no meaningful MB3 phi bending information

          // Fill client histos
          if (whME[wh].find(fullName("PhibResidualMean")) == whME[wh].end()) {
            bookWheelHistos(ibooker, wh, "PhibResidualMean");
            bookWheelHistos(ibooker, wh, "PhibResidualRMS");
          }

          double peak = PhibResidual->GetBinCenter(PhibResidual->GetMaximumBin());
          double phibMean = 0;
          double phibRMS = 0;
          try {
            TF1 ffPhib("ffPhib", "gaus");
            PhibResidual->Fit(&ffPhib, "CQO", "", peak - 5, peak + 5);
            phibMean = ffPhib.GetParameter(1);
            phibRMS = ffPhib.GetParameter(2);
          } catch (cms::Exception& iException) {
            edm::LogError(category()) << "[" << testName << "Test]: Error fitting PhibResidual for Wheel " << wh
                                      << " Sector " << sect << " Station " << stat;
          }

          std::map<std::string, MonitorElement*>& innerME = whME[wh];
          fillWhPlot(innerME.find(fullName("PhibResidualMean"))->second, sect, stat, phibMean);
          fillWhPlot(innerME.find(fullName("PhibResidualRMS"))->second, sect, stat, phibRMS);

          phibSummary = performLutTest(phibMean, phibRMS, thresholdPhibMean, thresholdPhibRMS);
        }
        fillWhPlot(whME[wh].find(fullName("PhibLutSummary"))->second, sect, stat, phibSummary);
      }
    }
  }

  // Barrel Summary Plots
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr) {
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw) {
      hwSource = (*iHw);
      for (int wh = -2; wh <= 2; ++wh) {
        std::map<std::string, MonitorElement*>* innerME = &(whME[wh]);

        TH2F* phiWhSummary = getHisto<TH2F>(innerME->find(fullName("PhiLutSummary"))->second);
        TH2F* phibWhSummary = getHisto<TH2F>(innerME->find(fullName("PhibLutSummary"))->second);
        for (int sect = 1; sect <= 12; ++sect) {
          int phiErr = 0;
          int phibErr = 0;
          int phiNoData = 0;
          int phibNoData = 0;
          for (int stat = 1; stat <= 4; ++stat) {
            switch (static_cast<int>(phiWhSummary->GetBinContent(sect, stat))) {
              case 1:
                phiNoData++;
                [[fallthrough]];
              case 2:
              case 3:
                phiErr++;
            }
            switch (static_cast<int>(phibWhSummary->GetBinContent(sect, stat))) {
              case 1:
                phibNoData++;
                [[fallthrough]];
              case 2:
              case 3:
                phibErr++;
            }
          }
          if (phiNoData == 4)
            phiErr = 5;
          if (phibNoData == 3)
            phibErr = 5;  // MB3 has no phib information
          cmsME.find(fullName("PhiLutSummary"))->second->setBinContent(sect, wh + wheelArrayShift, phiErr);
          cmsME.find(fullName("PhibLutSummary"))->second->setBinContent(sect, wh + wheelArrayShift, phibErr);
        }
      }
    }
  }
}

int DTLocalTriggerLutTest::performLutTest(double mean, double RMS, double thresholdMean, double thresholdRMS) {
  bool meanErr = fabs(mean) > thresholdMean;
  bool rmsErr = RMS > thresholdRMS;

  return (meanErr || rmsErr) ? 2 + (meanErr != rmsErr) : 0;
}

void DTLocalTriggerLutTest::fillWhPlot(MonitorElement* plot, int sect, int stat, float value, bool lessIsBest) {
  if (sect > 12) {
    int scsect = sect == 13 ? 4 : 10;
    if ((fabs(value) > fabs(plot->getBinContent(scsect, stat))) == lessIsBest) {
      plot->setBinContent(scsect, stat, value);
    }
  } else {
    plot->setBinContent(sect, stat, value);
  }

  return;
}
