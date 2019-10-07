/*
 *  See header file for a description of this class.
 *
 *  \author C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */

// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerTest.h"

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

DTLocalTriggerTest::DTLocalTriggerTest(const edm::ParameterSet& ps) {
  setConfig(ps, "DTLocalTrigger");
  baseFolderTM = "DT/03-LocalTrigger-TM/";
  nMinEvts = ps.getUntrackedParameter<int>("nEventsCert", 5000);

  bookingdone = false;
}

DTLocalTriggerTest::~DTLocalTriggerTest() {}

void DTLocalTriggerTest::Bookings(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
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
          for (int sect = 1; sect <= 12; ++sect) {
            bookSectorHistos(ibooker, wh, sect, "BXDistribPhiIn");
            bookSectorHistos(ibooker, wh, sect, "QualDistribPhiIn");
            bookSectorHistos(ibooker, wh, sect, "BXDistribPhiOut");
            bookSectorHistos(ibooker, wh, sect, "QualDistribPhiOut");
          }

          bookWheelHistos(ibooker, wh, "CorrectBXPhiIn");
          bookWheelHistos(ibooker, wh, "ResidualBXPhiIn");
          bookWheelHistos(ibooker, wh, "CorrFractionPhiIn");
          bookWheelHistos(ibooker, wh, "2ndFractionPhiIn");
          bookWheelHistos(ibooker, wh, "TriggerInclusivePhiIn");

          bookWheelHistos(ibooker, wh, "CorrectBXPhiOut");
          bookWheelHistos(ibooker, wh, "ResidualBXPhiOut");
          bookWheelHistos(ibooker, wh, "CorrFractionPhiOut");
          bookWheelHistos(ibooker, wh, "2ndFractionPhiOut");
          bookWheelHistos(ibooker, wh, "TriggerInclusivePhiOut");
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
        bookWheelHistos(ibooker, wh, "CorrFractionSummaryIn", "Summaries");
        bookWheelHistos(ibooker, wh, "2ndFractionSummaryIn", "Summaries");
        bookWheelHistos(ibooker, wh, "CorrFractionSummaryOut", "Summaries");
        bookWheelHistos(ibooker, wh, "2ndFractionSummaryOut", "Summaries");
      }
      bookCmsHistos(ibooker, "CorrFractionSummaryIn");
      bookCmsHistos(ibooker, "2ndFractionSummaryIn");
      bookCmsHistos(ibooker, "CorrFractionSummaryOut");
      bookCmsHistos(ibooker, "2ndFractionSummaryOut");

      if (hwSource == "TM") {
        bookCmsHistos(ibooker, "TrigGlbSummary", "", true);
      }
    }
  }

  bookingdone = true;
}

void DTLocalTriggerTest::beginRun(const edm::Run& r, const edm::EventSetup& c) {
  DTLocalTriggerBaseTest::beginRun(r, c);
}

void DTLocalTriggerTest::runClientDiagnostic(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  if (!bookingdone)
    Bookings(ibooker, igetter);

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr) {
    trigSource = (*iTr);

    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw) {
      hwSource = (*iHw);
      // Loop over the TriggerUnits
      for (int stat = 1; stat <= 4; ++stat) {
        for (int wh = -2; wh <= 2; ++wh) {
          for (int sect = 1; sect <= 12; ++sect) {
            DTChamberId chId(wh, stat, sect);
            int sector_id = (wh + wheelArrayShift) + (sect - 1) * 5;

            // IN part
            TH2F* BXvsQual = getHisto<TH2F>(igetter.get(getMEName("BXvsQual_In", "LocalTriggerPhiIn", chId)));
            TH1F* BestQual = getHisto<TH1F>(igetter.get(getMEName("BestQual_In", "LocalTriggerPhiIn", chId)));
            TH2F* Flag1stvsQual = getHisto<TH2F>(igetter.get(getMEName("Flag1stvsQual_In", "LocalTriggerPhiIn", chId)));
            if (BXvsQual && Flag1stvsQual && BestQual) {
              int corrSummary = 1;
              int secondSummary = 1;
              //default values for histograms
              double BX_OK = 51.;
              double BXMean = 51.;
              double corrFrac = 0.;
              double secondFrac = 0.;
              double besttrigs = 0.;
              if (BestQual->GetEntries() > 1) {
                TH1D* BXHH = BXvsQual->ProjectionY("", 6, 7, "");
                TH1D* Flag1st = Flag1stvsQual->ProjectionY();
                int BXOK_bin = BXHH->GetEntries() >= 1 ? BXHH->GetMaximumBin() : 51;
                BXMean = BXHH->GetEntries() >= 1 ? BXHH->GetMean() : 51;
                BX_OK = BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
                double trigsFlag2nd = Flag1st->GetBinContent(2);
                double trigs = Flag1st->GetEntries();
                besttrigs = BestQual->GetEntries();
                double besttrigsCorr = BestQual->Integral(5, 7, "");
                delete BXHH;
                delete Flag1st;

                corrFrac = besttrigsCorr / besttrigs;
                secondFrac = trigsFlag2nd / trigs;
                if (corrFrac < parameters.getUntrackedParameter<double>("corrFracError", .5)) {
                  corrSummary = 2;
                } else if (corrFrac < parameters.getUntrackedParameter<double>("corrFracWarning", .6)) {
                  corrSummary = 3;
                } else {
                  corrSummary = 0;
                }
                if (secondFrac > parameters.getUntrackedParameter<double>("secondFracError", .2)) {
                  secondSummary = 2;
                } else if (secondFrac > parameters.getUntrackedParameter<double>("secondFracWarning", .1)) {
                  secondSummary = 3;
                } else {
                  secondSummary = 0;
                }

                if (secME[sector_id].find(fullName("BXDistribPhiIn")) == secME[sector_id].end()) {
                  bookSectorHistos(ibooker, wh, sect, "QualDistribPhiIn");
                  bookSectorHistos(ibooker, wh, sect, "BXDistribPhiIn");
                }

                TH1D* BXDistr = BXvsQual->ProjectionY();
                TH1D* QualDistr = BXvsQual->ProjectionX();
                std::map<std::string, MonitorElement*>* innerME = &(secME[sector_id]);

                int nbinsBX = BXDistr->GetNbinsX();
                int firstBinCenter = static_cast<int>(BXDistr->GetBinCenter(1));
                int lastBinCenter = static_cast<int>(BXDistr->GetBinCenter(nbinsBX));
                int iMin = firstBinCenter > -4 ? firstBinCenter : -4;
                int iMax = lastBinCenter < 20 ? lastBinCenter : 20;
                for (int ibin = iMin + 5; ibin <= iMax + 5; ++ibin) {
                  innerME->find(fullName("BXDistribPhiIn"))
                      ->second->setBinContent(ibin, stat, BXDistr->GetBinContent(ibin - 5 - firstBinCenter + 1));
                }
                for (int ibin = 1; ibin <= 7; ++ibin) {
                  innerME->find(fullName("QualDistribPhiIn"))
                      ->second->setBinContent(ibin, stat, QualDistr->GetBinContent(ibin));
                }

                delete BXDistr;
                delete QualDistr;
              }

              std::map<std::string, MonitorElement*>* innerME = &(secME[sector_id]);

              if (whME[wh].find(fullName("CorrectBXPhiIn")) == whME[wh].end()) {
                bookWheelHistos(ibooker, wh, "ResidualBXPhiIn");
                bookWheelHistos(ibooker, wh, "CorrectBXPhiIn");
                bookWheelHistos(ibooker, wh, "CorrFractionPhiIn");
                bookWheelHistos(ibooker, wh, "2ndFractionPhiIn");
                bookWheelHistos(ibooker, wh, "TriggerInclusivePhiIn");
              }

              innerME = &(whME[wh]);
              innerME->find(fullName("CorrectBXPhiIn"))->second->setBinContent(sect, stat, BX_OK + 0.00001);
              innerME->find(fullName("ResidualBXPhiIn"))
                  ->second->setBinContent(sect, stat, round(25. * (BXMean - BX_OK)) + 0.00001);
              innerME->find(fullName("CorrFractionPhiIn"))->second->setBinContent(sect, stat, corrFrac);
              innerME->find(fullName("TriggerInclusivePhiIn"))->second->setBinContent(sect, stat, besttrigs);
              innerME->find(fullName("2ndFractionPhiIn"))->second->setBinContent(sect, stat, secondFrac);

              whME[wh].find(fullName("CorrFractionSummaryIn"))->second->setBinContent(sect, stat, corrSummary);
              whME[wh].find(fullName("2ndFractionSummaryIn"))->second->setBinContent(sect, stat, secondSummary);

            }  // closes BXvsQual && Flag1stvsQual && BestQual

            if (hwSource == "TM") {
              //Out part

              TH2F* BXvsQual = getHisto<TH2F>(igetter.get(getMEName("BXvsQual_Out", "LocalTriggerPhiOut", chId)));
              TH1F* BestQual = getHisto<TH1F>(igetter.get(getMEName("BestQual_Out", "LocalTriggerPhiOut", chId)));
              TH2F* Flag1stvsQual =
                  getHisto<TH2F>(igetter.get(getMEName("Flag1stvsQual_Out", "LocalTriggerPhiOut", chId)));
              if (BXvsQual && Flag1stvsQual && BestQual) {
                int corrSummary = 1;
                int secondSummary = 1;
                //default values for histograms
                double BX_OK = 51.;
                double BXMean = 51.;
                double corrFrac = 0.;
                double secondFrac = 0.;
                double besttrigs = 0.;

                if (BestQual->GetEntries() > 1) {
                  TH1D* BXHH = BXvsQual->ProjectionY("", 6, 7, "");
                  TH1D* Flag1st = Flag1stvsQual->ProjectionY();
                  int BXOK_bin = BXHH->GetEntries() >= 1 ? BXHH->GetMaximumBin() : 51;
                  BXMean = BXHH->GetEntries() >= 1 ? BXHH->GetMean() : 51;
                  BX_OK = BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
                  double trigsFlag2nd = Flag1st->GetBinContent(2);
                  double trigs = Flag1st->GetEntries();
                  besttrigs = BestQual->GetEntries();
                  double besttrigsCorr = BestQual->Integral(5, 7, "");
                  delete BXHH;
                  delete Flag1st;

                  corrFrac = besttrigsCorr / besttrigs;
                  secondFrac = trigsFlag2nd / trigs;
                  if (corrFrac < parameters.getUntrackedParameter<double>("corrFracError", .5)) {
                    corrSummary = 2;
                  } else if (corrFrac < parameters.getUntrackedParameter<double>("corrFracWarning", .6)) {
                    corrSummary = 3;
                  } else {
                    corrSummary = 0;
                  }
                  if (secondFrac > parameters.getUntrackedParameter<double>("secondFracError", .2)) {
                    secondSummary = 2;
                  } else if (secondFrac > parameters.getUntrackedParameter<double>("secondFracWarning", .1)) {
                    secondSummary = 3;
                  } else {
                    secondSummary = 0;
                  }

                  if (secME[sector_id].find(fullName("BXDistribPhiOut")) == secME[sector_id].end()) {
                    bookSectorHistos(ibooker, wh, sect, "QualDistribPhiOut");
                    bookSectorHistos(ibooker, wh, sect, "BXDistribPhiOut");
                  }

                  TH1D* BXDistr = BXvsQual->ProjectionY();
                  TH1D* QualDistr = BXvsQual->ProjectionX();
                  std::map<std::string, MonitorElement*>* innerME = &(secME[sector_id]);

                  int nbinsBX = BXDistr->GetNbinsX();
                  int firstBinCenter = static_cast<int>(BXDistr->GetBinCenter(1));
                  int lastBinCenter = static_cast<int>(BXDistr->GetBinCenter(nbinsBX));
                  int iMin = firstBinCenter > -4 ? firstBinCenter : -4;
                  int iMax = lastBinCenter < 20 ? lastBinCenter : 20;
                  for (int ibin = iMin + 5; ibin <= iMax + 5; ++ibin) {
                    innerME->find(fullName("BXDistribPhiOut"))
                        ->second->setBinContent(ibin, stat, BXDistr->GetBinContent(ibin - 5 - firstBinCenter + 1));
                  }
                  for (int ibin = 1; ibin <= 7; ++ibin) {
                    innerME->find(fullName("QualDistribPhiOut"))
                        ->second->setBinContent(ibin, stat, QualDistr->GetBinContent(ibin));
                  }

                  delete BXDistr;
                  delete QualDistr;
                }

                std::map<std::string, MonitorElement*>* innerME = &(secME[sector_id]);

                if (whME[wh].find(fullName("CorrectBXPhiOut")) == whME[wh].end()) {
                  bookWheelHistos(ibooker, wh, "ResidualBXPhiOut");
                  bookWheelHistos(ibooker, wh, "CorrectBXPhiOut");
                  bookWheelHistos(ibooker, wh, "CorrFractionPhiOut");
                  bookWheelHistos(ibooker, wh, "2ndFractionPhiOut");
                  bookWheelHistos(ibooker, wh, "TriggerInclusivePhiOut");
                }

                innerME = &(whME[wh]);
                innerME->find(fullName("CorrectBXPhiOut"))->second->setBinContent(sect, stat, BX_OK + 0.00001);
                innerME->find(fullName("ResidualBXPhiOut"))
                    ->second->setBinContent(sect, stat, round(25. * (BXMean - BX_OK)) + 0.00001);
                innerME->find(fullName("CorrFractionPhiOut"))->second->setBinContent(sect, stat, corrFrac);
                innerME->find(fullName("TriggerInclusivePhiOut"))->second->setBinContent(sect, stat, besttrigs);
                innerME->find(fullName("2ndFractionPhiOut"))->second->setBinContent(sect, stat, secondFrac);

                whME[wh].find(fullName("CorrFractionSummaryOut"))->second->setBinContent(sect, stat, corrSummary);
                whME[wh].find(fullName("2ndFractionSummaryOut"))->second->setBinContent(sect, stat, secondSummary);

              }  // closes BXvsQual && Flag1stvsQual && BestQual

            }  // Check on TM source
               //Theta part
            if (hwSource == "TM") {
              // Perform TM plot analysis (Theta ones)
              TH2F* ThetaPosvsBX = getHisto<TH2F>(igetter.get(getMEName("PositionvsBX", "LocalTriggerTheta", chId)));
              double BX_OK = 48;
              // no theta triggers in stat 4!
              if (ThetaPosvsBX && stat < 4 && ThetaPosvsBX->GetEntries() > 1) {
                TH1D* BX = ThetaPosvsBX->ProjectionX();
                int BXOK_bin = BX->GetEffectiveEntries() >= 1 ? BX->GetMaximumBin() : 10;
                BX_OK = ThetaPosvsBX->GetXaxis()->GetBinCenter(BXOK_bin);
                delete BX;

                if (whME[wh].find(fullName("CorrectBXTheta")) == whME[wh].end()) {
                  bookWheelHistos(ibooker, wh, "CorrectBXTheta");
                }
                std::map<std::string, MonitorElement*>* innerME = &(whME.find(wh)->second);
                innerME->find(fullName("CorrectBXTheta"))->second->setBinContent(sect, stat, BX_OK + 0.00001);
              }
              // Adding trigger info to compute H fraction (11/10/2016) M.C.Fouz
              TH2F* ThetaBXvsQual = getHisto<TH2F>(igetter.get(getMEName("ThetaBXvsQual", "LocalTriggerTheta", chId)));
              TH1F* ThetaBestQual = getHisto<TH1F>(igetter.get(getMEName("ThetaBestQual", "LocalTriggerTheta", chId)));
              if (ThetaBXvsQual && ThetaBestQual && stat < 4 && ThetaBestQual->GetEntries() > 1) {
                double trigs = ThetaBestQual->GetEntries();
                double trigsH = ThetaBestQual->GetBinContent(
                    2);  // Note that for the new plots H is at bin=2 and not 4 as in DDU!!!!
                if (whME[wh].find(fullName("HFractionTheta")) == whME[wh].end()) {
                  bookWheelHistos(ibooker, wh, "HFractionTheta");
                }
                std::map<std::string, MonitorElement*>* innerME = &(whME.find(wh)->second);
                innerME->find(fullName("HFractionTheta"))->second->setBinContent(sect, stat, trigsH / trigs);
              }
              // END ADDING H Fraction info
            }
          }
        }
      }
    }
  }

  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr) {
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw) {
      hwSource = (*iHw);
      for (int wh = -2; wh <= 2; ++wh) {
        std::map<std::string, MonitorElement*>* innerME = &(whME[wh]);
        // In part
        TH2F* corrWhSummaryIn = getHisto<TH2F>(innerME->find(fullName("CorrFractionSummaryIn"))->second);
        TH2F* secondWhSummaryIn = getHisto<TH2F>(innerME->find(fullName("2ndFractionSummaryIn"))->second);
        for (int sect = 1; sect <= 12; ++sect) {
          int corrErr = 0;
          int secondErr = 0;
          int corrNoData = 0;
          int secondNoData = 0;
          for (int stat = 1; stat <= 4; ++stat) {
            switch (static_cast<int>(corrWhSummaryIn->GetBinContent(sect, stat))) {
              case 1:
                corrNoData++;
                [[fallthrough]];
              case 2:
                corrErr++;
            }
            switch (static_cast<int>(secondWhSummaryIn->GetBinContent(sect, stat))) {
              case 1:
                secondNoData++;
                [[fallthrough]];
              case 2:
                secondErr++;
            }
          }
          if (corrNoData == 4)
            corrErr = 5;
          if (secondNoData == 4)
            secondErr = 5;
          cmsME.find(fullName("CorrFractionSummaryIn"))->second->setBinContent(sect, wh + wheelArrayShift, corrErr);
          cmsME.find(fullName("2ndFractionSummaryIn"))->second->setBinContent(sect, wh + wheelArrayShift, secondErr);
        }
        // Out part
        TH2F* corrWhSummaryOut = getHisto<TH2F>(innerME->find(fullName("CorrFractionSummaryOut"))->second);
        TH2F* secondWhSummaryOut = getHisto<TH2F>(innerME->find(fullName("2ndFractionSummaryOut"))->second);
        for (int sect = 1; sect <= 12; ++sect) {
          int corrErr = 0;
          int secondErr = 0;
          int corrNoData = 0;
          int secondNoData = 0;
          for (int stat = 1; stat <= 4; ++stat) {
            switch (static_cast<int>(corrWhSummaryOut->GetBinContent(sect, stat))) {
              case 1:
                corrNoData++;
                [[fallthrough]];
              case 2:
                corrErr++;
            }
            switch (static_cast<int>(secondWhSummaryOut->GetBinContent(sect, stat))) {
              case 1:
                secondNoData++;
                [[fallthrough]];
              case 2:
                secondErr++;
            }
          }
          if (corrNoData == 4)
            corrErr = 5;
          if (secondNoData == 4)
            secondErr = 5;
          cmsME.find(fullName("CorrFractionSummaryOut"))->second->setBinContent(sect, wh + wheelArrayShift, corrErr);
          cmsME.find(fullName("2ndFractionSummaryOut"))->second->setBinContent(sect, wh + wheelArrayShift, secondErr);
        }
      }
    }
  }
  fillGlobalSummary(igetter);
}

void DTLocalTriggerTest::fillGlobalSummary(DQMStore::IGetter& igetter) {
  float glbPerc[5] = {1., 0.9, 0.6, 0.3, 0.01};
  trigSource = "";
  hwSource = "TM";

  int nSecReadout = 0;

  for (int wh = -2; wh <= 2; ++wh) {
    for (int sect = 1; sect <= 12; ++sect) {
      float maxErr = 8.;
      int corr = cmsME.find(fullName("CorrFractionSummaryIn"))->second->getBinContent(sect, wh + wheelArrayShift);
      int second = cmsME.find(fullName("2ndFractionSummaryIn"))->second->getBinContent(sect, wh + wheelArrayShift);
      int lut = 0;
      MonitorElement* lutsME = igetter.get(topFolder() + "Summaries/TrigLutSummary");
      if (lutsME) {
        lut = lutsME->getBinContent(sect, wh + wheelArrayShift);
        maxErr += 4;
      } else {
        LogTrace(category()) << "[" << testName << "Test]: TM Lut test Summary histo not found." << endl;
      }
      (corr < 5 || second < 5) && nSecReadout++;
      int errcode = ((corr < 5 ? corr : 4) + (second < 5 ? second : 4) + (lut < 5 ? lut : 4));
      errcode = min(int((errcode / maxErr + 0.01) * 5), 5);
      cmsME.find("TrigGlbSummary")->second->setBinContent(sect, wh + wheelArrayShift, glbPerc[errcode]);
    }
  }

  if (!nSecReadout)
    cmsME.find("TrigGlbSummary")->second->Reset();  // white histo id TM is not RO

  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsTrigger";
  MonitorElement* meProcEvts = igetter.get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    cmsME.find("TrigGlbSummary")->second->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    cmsME.find("TrigGlbSummary")->second->setEntries(nMinEvts + 1);
    LogVerbatim(category()) << "[" << testName << "Test]: ME: " << nEvtsName << " not found!" << endl;
  }
}
