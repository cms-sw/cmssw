/*
 *  See header file for a description of this class.
 *
 *  \author C. Battilana S. Marcellini - INFN Bologna
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */

// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerTPTest.h"

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

DTLocalTriggerTPTest::DTLocalTriggerTPTest(const edm::ParameterSet& ps) {
  setConfig(ps, "DTLocalTriggerTP");
  baseFolderTM = "DT/11-LocalTriggerTP-TM/";

  bookingdone = false;
}

DTLocalTriggerTPTest::~DTLocalTriggerTPTest() {}

void DTLocalTriggerTPTest::Bookings(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
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
          bookWheelHistos(ibooker, wh, "CorrectBXPhi");
          bookWheelHistos(ibooker, wh, "ResidualBXPhi");
        }
      }
    }
  }

  bookingdone = true;
}

void DTLocalTriggerTPTest::beginRun(const edm::Run& r, const edm::EventSetup& c) {
  DTLocalTriggerBaseTest::beginRun(r, c);
}

void DTLocalTriggerTPTest::runClientDiagnostic(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
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

            // Perform TM common plot analysis (Phi ones)
            TH2F* BXvsQual = getHisto<TH2F>(igetter.get(getMEName("BXvsQual_In", "LocalTriggerPhiIn", chId)));
            if (BXvsQual) {
              if (BXvsQual->GetEntries() > 1) {
                TH1D* BX = BXvsQual->ProjectionY();
                int BXOK_bin = BX->GetMaximumBin();
                double BXMean = BX->GetMean();
                double BX_OK = BXvsQual->GetYaxis()->GetBinCenter(BXOK_bin);
                delete BX;

                if (whME[wh].find(fullName("CorrectBXPhi")) == whME[wh].end()) {
                  bookWheelHistos(ibooker, wh, "ResidualBXPhi");
                  bookWheelHistos(ibooker, wh, "CorrectBXPhi");
                }

                std::map<std::string, MonitorElement*>* innerME = &(whME[wh]);
                innerME->find(fullName("CorrectBXPhi"))->second->setBinContent(sect, stat, BX_OK + 0.00001);
                innerME->find(fullName("ResidualBXPhi"))
                    ->second->setBinContent(sect, stat, round(25. * (BXMean - BX_OK)) + 0.00001);
              }
            }
          }
        }
      }
    }
  }
}
