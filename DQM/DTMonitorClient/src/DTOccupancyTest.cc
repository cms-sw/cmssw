/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - University and INFN Torino
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah -ncpp-um-my
 *
 */

#include "DQM/DTMonitorClient/src/DTOccupancyTest.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"

using namespace edm;
using namespace std;

DTOccupancyTest::DTOccupancyTest(const edm::ParameterSet& ps)
    : muonGeomToken_(esConsumes<edm::Transition::BeginRun>()) {
  LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest]: Constructor";

  // Get the DQM service

  lsCounter = 0;

  writeRootFile = ps.getUntrackedParameter<bool>("writeRootFile", false);
  if (writeRootFile) {
    rootFile = new TFile("DTOccupancyTest.root", "RECREATE");
    ntuple = new TNtuple("OccupancyNtuple",
                         "OccupancyNtuple",
                         "ls:wh:st:se:lay1MeanCell:lay1RMS:lay2MeanCell:lay2RMS:lay3MeanCell:lay3RMS:lay4MeanCell:"
                         "lay4RMS:lay5MeanCell:lay5RMS:lay6MeanCell:lay6RMS:lay7MeanCell:lay7RMS:lay8MeanCell:lay8RMS:"
                         "lay9MeanCell:lay9RMS:lay10MeanCell:lay10RMS:lay11MeanCell:lay11RMS:lay12MeanCell:lay12RMS");
  }

  // switch on the mode for running on test pulses (different top folder)
  tpMode = ps.getUntrackedParameter<bool>("testPulseMode", false);

  runOnAllHitsOccupancies = ps.getUntrackedParameter<bool>("runOnAllHitsOccupancies", true);
  runOnNoiseOccupancies = ps.getUntrackedParameter<bool>("runOnNoiseOccupancies", false);
  runOnInTimeOccupancies = ps.getUntrackedParameter<bool>("runOnInTimeOccupancies", false);
  nMinEvts = ps.getUntrackedParameter<int>("nEventsCert", 5000);
  nMinEvtsPC = ps.getUntrackedParameter<int>("nEventsMinPC", 2200);
  nZeroEvtsPC = ps.getUntrackedParameter<int>("nEventsZeroPC", 30);

  bookingdone = false;

  // Event counter
  nevents = 0;
}

DTOccupancyTest::~DTOccupancyTest() {
  LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest") << " destructor called" << endl;
}

void DTOccupancyTest::beginRun(const edm::Run& run, const EventSetup& context) {
  LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest]: BeginRun";

  // Get the geometry
  muonGeom = &context.getData(muonGeomToken_);
}

void DTOccupancyTest::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker,
                                            DQMStore::IGetter& igetter,
                                            edm::LuminosityBlock const& lumiSeg,
                                            edm::EventSetup const& context) {
  if (!bookingdone) {
    // Book the summary histos
    //   - one summary per wheel
    for (int wh = -2; wh <= 2; ++wh) {  // loop over wheels
      bookHistos(ibooker, wh, string("Occupancies"), "OccupancySummary");
    }

    ibooker.setCurrentFolder(topFolder());
    string title = "Occupancy Summary";
    if (tpMode) {
      title = "Test Pulse Occupancy Summary";
    }
    //   - global summary with alarms
    summaryHisto = ibooker.book2D("OccupancySummary", title.c_str(), 12, 1, 13, 5, -2, 3);
    summaryHisto->setAxisTitle("sector", 1);
    summaryHisto->setAxisTitle("wheel", 2);

    //   - global summary with percentages
    glbSummaryHisto = ibooker.book2D("OccupancyGlbSummary", title.c_str(), 12, 1, 13, 5, -2, 3);
    glbSummaryHisto->setAxisTitle("sector", 1);
    glbSummaryHisto->setAxisTitle("wheel", 2);

    // assign the name of the input histogram
    if (runOnAllHitsOccupancies) {
      nameMonitoredHisto = "OccupancyAllHits_perCh";
    } else if (runOnNoiseOccupancies) {
      nameMonitoredHisto = "OccupancyNoise_perCh";
    } else if (runOnInTimeOccupancies) {
      nameMonitoredHisto = "OccupancyInTimeHits_perCh";
    } else {  // default is AllHits histo
      nameMonitoredHisto = "OccupancyAllHits_perCh";
    }
  }
  bookingdone = true;

  LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest")
      << "[DTOccupancyTest]: End of LS transition, performing the DQM client operation";
  lsCounter++;

  // Reset the global summary
  summaryHisto->Reset();
  glbSummaryHisto->Reset();

  nChannelTotal = 0;
  nChannelDead = 0;

  // Get all the DT chambers
  vector<const DTChamber*> chambers = muonGeom->chambers();

  for (vector<const DTChamber*>::const_iterator chamber = chambers.begin(); chamber != chambers.end();
       ++chamber) {  // Loop over all chambers
    DTChamberId chId = (*chamber)->id();

    MonitorElement* chamberOccupancyHisto = igetter.get(getMEName(nameMonitoredHisto, chId));

    // Run the tests on the plot for the various granularities
    if (chamberOccupancyHisto != nullptr) {
      // Get the 2D histo
      TH2F* histo = chamberOccupancyHisto->getTH2F();
      float chamberPercentage = 1.;
      int result = runOccupancyTest(histo, chId, chamberPercentage);
      int sector = chId.sector();

      if (sector == 13) {
        sector = 4;
        float resultSect4 = wheelHistos[chId.wheel()]->getBinContent(sector, chId.station());
        if (resultSect4 > result) {
          result = (int)resultSect4;
        }
      } else if (sector == 14) {
        sector = 10;
        float resultSect10 = wheelHistos[chId.wheel()]->getBinContent(sector, chId.station());
        if (resultSect10 > result) {
          result = (int)resultSect10;
        }
      }

      // the 2 MB4 of Sect 4 and 10 count as half a chamber
      if ((sector == 4 || sector == 10) && chId.station() == 4)
        chamberPercentage = chamberPercentage / 2.;

      wheelHistos[chId.wheel()]->setBinContent(sector, chId.station(), result);
      if (result > summaryHisto->getBinContent(sector, chId.wheel() + 3)) {
        summaryHisto->setBinContent(sector, chId.wheel() + 3, result);
      }
      glbSummaryHisto->Fill(sector, chId.wheel(), chamberPercentage * 1. / 4.);
    } else {
      LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest")
          << "[DTOccupancyTest] ME: " << getMEName(nameMonitoredHisto, chId) << " not found!" << endl;
    }
  }

  string nEvtsName = "DT/EventInfo/Counters/nProcessedEventsDigi";

  MonitorElement* meProcEvts = igetter.get(nEvtsName);

  if (meProcEvts) {
    int nProcEvts = meProcEvts->getFloatValue();
    glbSummaryHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
    summaryHisto->setEntries(nProcEvts < nMinEvts ? 10. : nProcEvts);
  } else {
    glbSummaryHisto->setEntries(nMinEvts + 1);
    summaryHisto->setEntries(nMinEvts + 1);
    LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest")
        << "[DTOccupancyTest] ME: " << nEvtsName << " not found!" << endl;
  }

  // Fill the global summary
  // Check for entire sectors off and report them on the global summary
  //FIXME: TODO

  if (writeRootFile)
    ntuple->AutoSave("SaveSelf");
}

void DTOccupancyTest::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest") << "[DTOccupancyTest] endjob called!";
  if (writeRootFile) {
    rootFile->cd();
    ntuple->Write();
    rootFile->Close();
  }
}

// --------------------------------------------------

void DTOccupancyTest::bookHistos(DQMStore::IBooker& ibooker, const int wheelId, string folder, string histoTag) {
  // Set the current folder
  stringstream wheel;
  wheel << wheelId;

  ibooker.setCurrentFolder(topFolder());

  // build the histo name
  string histoName = histoTag + "_W" + wheel.str();

  LogVerbatim("DTDQM|DTMonitorClient|DTOccupancyTest")
      << "[DTOccupancyTest]: booking wheel histo:" << histoName << " (tag " << histoTag
      << ") in: " << topFolder() + "Wheel" + wheel.str() + "/" + folder << endl;

  string histoTitle = "Occupancy summary WHEEL: " + wheel.str();
  if (tpMode) {
    histoTitle = "TP Occupancy summary WHEEL: " + wheel.str();
  }

  wheelHistos[wheelId] = ibooker.book2D(histoName, histoTitle, 12, 1, 13, 4, 1, 5);
  wheelHistos[wheelId]->setBinLabel(1, "MB1", 2);
  wheelHistos[wheelId]->setBinLabel(2, "MB2", 2);
  wheelHistos[wheelId]->setBinLabel(3, "MB3", 2);
  wheelHistos[wheelId]->setBinLabel(4, "MB4", 2);
  wheelHistos[wheelId]->setAxisTitle("sector", 1);
}

string DTOccupancyTest::getMEName(string histoTag, const DTChamberId& chId) {
  stringstream wheel;
  wheel << chId.wheel();
  stringstream station;
  station << chId.station();
  stringstream sector;
  sector << chId.sector();

  string folderRoot = topFolder() + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str() + "/";

  string folder = "Occupancies/";

  // build the histo name
  string histoName = histoTag + "_W" + wheel.str() + "_St" + station.str() + "_Sec" + sector.str();

  string histoname = folderRoot + histoName;

  return histoname;
}

int DTOccupancyTest::getIntegral(TH2F* histo, int firstBinX, int lastBinX, int firstBinY, int lastBinY, bool doall) {
  int sum = 0;
  for (Int_t i = firstBinX; i < lastBinX + 1; i++) {
    for (Int_t j = firstBinY; j < lastBinY + 1; j++) {
      if (histo->GetBinContent(i, j) > 0) {
        if (!doall)
          return 1;
        sum += histo->GetBinContent(i, j);
      }
    }
  }

  return sum;
}

// Run a test on the occupancy of the chamber
// Return values:
// 0 -> all ok
// 1 -> # consecutive dead channels > N
// 2 -> dead layer
// 3 -> dead SL
// 4 -> dead chamber

int DTOccupancyTest::runOccupancyTest(TH2F* histo, const DTChamberId& chId, float& chamberPercentage) {
  int nBinsX = histo->GetNbinsX();

  LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "--- Occupancy test for chamber: " << chId << endl;

  int compDeadCell = 0;
  int totCell = 0;
  int totOccup = 0;

  for (int slay = 1; slay <= 3; ++slay) {  // loop over SLs
    int binYlow = ((slay - 1) * 4) + 1;

    if (chId.station() == 4 && slay == 2)
      continue;
    for (int lay = 1; lay <= 4; ++lay) {  // loop over layers
      DTLayerId layID(chId, slay, lay);
      int firstWire = muonGeom->layer(layID)->specificTopology().firstChannel();
      int nWires = muonGeom->layer(layID)->specificTopology().channels();
      int binY = binYlow + (lay - 1);
      int totalDeadCells = 0;
      int nDeadCellsInARow = 1;
      int nDeadCellsInARowMax = 0;
      bool previousIsDead = false;

      totCell += nWires;

      for (int cell = firstWire; cell != (nWires + firstWire); ++cell) {  // loop over cells
        double cellOccup = histo->GetBinContent(cell, binY);
        LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "       cell occup: " << cellOccup;
        totOccup += cellOccup;

        if (cellOccup == 0) {
          totalDeadCells++;
          if (previousIsDead) {
            nDeadCellsInARow++;
          } else {
            if (nDeadCellsInARow > nDeadCellsInARowMax)
              nDeadCellsInARowMax = nDeadCellsInARow;
            nDeadCellsInARow = 1;
          }
          previousIsDead = true;
          LogTrace("DTDQM|DTMonitorClient|DTOccupancyTest") << "       below reference" << endl;
        } else {
          previousIsDead = false;
        }
        //   // 3 cells not dead between a group of dead cells don't break the count
        if (nDeadCellsInARow > nDeadCellsInARowMax)
          nDeadCellsInARowMax = nDeadCellsInARow;
      }
      compDeadCell += totalDeadCells;
      if (nDeadCellsInARowMax >= 7.) {
        histo->SetBinContent(nBinsX + 1, binY, -1.);
      }
    }
  }

  nChannelTotal += totCell;
  nChannelDead += compDeadCell;
  chamberPercentage = 1. - (float(compDeadCell) / totCell);

  int min_occup = nZeroEvtsPC * 20;
  if (chId.station() == 3)
    min_occup = nZeroEvtsPC * 3;
  if (chId.station() == 2)
    min_occup = nZeroEvtsPC * 8;
  if ((chId.station() == 4) && (chId.sector() == 9))
    min_occup = nZeroEvtsPC * 3;
  if ((chId.station() == 4) && (chId.sector() == 10))
    min_occup = nZeroEvtsPC * 3;
  if ((chId.station() == 4) && (chId.sector() == 11))
    min_occup = nZeroEvtsPC * 3;
  if ((chId.station() == 4) && (chId.sector() == 14))
    min_occup = nZeroEvtsPC * 3;

  if (totOccup < min_occup)
    return 4;
  if (totOccup < nMinEvtsPC)
    chamberPercentage = 1.;

  if (chamberPercentage < 0.2)
    return 4;
  if (chamberPercentage < 0.5)
    return 3;
  if (chamberPercentage < 0.75)
    return 2;
  if (chamberPercentage < 0.9)
    return 1;

  return 0;
}

string DTOccupancyTest::topFolder() const {
  if (tpMode)
    return string("DT/10-TestPulses/");
  return string("DT/01-Digi/");
}
