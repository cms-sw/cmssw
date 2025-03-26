#include "L1Trigger/DTTriggerPhase2/interface/MPCoincidenceFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

// ============================================================================
// Constructors and destructor
// ============================================================================
MPCoincidenceFilter::MPCoincidenceFilter(const ParameterSet &pset)
    : MPFilter(pset),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      co_option_(pset.getParameter<int>("co_option")),
      co_quality_(pset.getParameter<int>("co_quality")),
      co_wh2option_(pset.getParameter<int>("co_wh2option")),
      scenario_(pset.getParameter<int>("scenario")) {}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPCoincidenceFilter::initialise(const edm::EventSetup &iEventSetup) {}

void MPCoincidenceFilter::run(edm::Event &iEvent,
                              const edm::EventSetup &iEventSetup,
                              std::vector<metaPrimitive> &allMPaths,
                              std::vector<metaPrimitive> &inMPaths,
                              std::vector<metaPrimitive> &outMPaths) {
  if (debug_)
    LogDebug("MPCoincidenceFilter") << "MPCoincidenceFilter: run";

  double shift_back = 0;  // Needed for t0 (TDC) calculation, taken from main algo
  if (scenario_ == MC)
    shift_back = 400;
  else if (scenario_ == DATA)
    shift_back = 0;
  else if (scenario_ == SLICE_TEST)
    shift_back = 400;

  auto filteredMPs = filter(inMPaths, allMPaths, co_option_, co_quality_, co_wh2option_, shift_back);
  for (auto &mp : filteredMPs)
    outMPaths.push_back(mp);
}

void MPCoincidenceFilter::finish() {}

///////////////////////////
///  OTHER METHODS

std::vector<metaPrimitive> MPCoincidenceFilter::filter(std::vector<metaPrimitive> inMPs,
                                                       std::vector<metaPrimitive> allMPs,
                                                       int co_option,
                                                       int co_quality,
                                                       int co_wh2option,
                                                       double shift_back) {
  std::vector<metaPrimitive> outMPs;

  for (auto &mp : inMPs) {
    DTChamberId chId(mp.rawId);
    DTSuperLayerId slId(mp.rawId);

    bool PhiMP = false;
    if (slId.superLayer() != 2)
      PhiMP = true;

    int sector = chId.sector();
    int wheel = chId.wheel();
    int station = chId.station();
    if (sector == 13)
      sector = 4;
    if (sector == 14)
      sector = 10;

    bool wh2pass = false;
    if (abs(wheel) == 2 && station == 1 && co_wh2option == 1)
      wh2pass = true;
    if (abs(wheel) == 2 && station < 3 && co_wh2option == 2)
      wh2pass = true;

    if (co_option == -1 || mp.quality > 5 || wh2pass == 1)
      outMPs.push_back(mp);
    else {
      int sector_p1 = sector + 1;
      int sector_m1 = sector - 1;
      if (sector_p1 == 13)
        sector_p1 = 1;
      if (sector_m1 == 0)
        sector_m1 = 12;

      string whch = "wh" + std::to_string(wheel) + "ch" + std::to_string(station) + "";
      float t0_mean, t0_width;
      if (PhiMP == 1) {
        t0_mean = mphi_mean.find("" + whch + "")->second;
        t0_width = mphi_width.find("" + whch + "")->second;
      } else {
        t0_mean = mth_mean.find("" + whch + "")->second;
        t0_width = mth_width.find("" + whch + "")->second;
      }

      float t0 = (mp.t0 - shift_back * LHC_CLK_FREQ) * ((float)TIME_TO_TDC_COUNTS / (float)LHC_CLK_FREQ);
      t0 = t0 - t0_mean;

      bool co_found = false;

      for (auto &mp2 : allMPs) {
        DTChamberId chId2(mp2.rawId);
        DTSuperLayerId slId2(mp2.rawId);

        bool PhiMP2 = false;
        if (slId2.superLayer() != 2)
          PhiMP2 = true;

        int qcut = co_quality;  // Tested for 0,1,5
        if (mp.quality > qcut)
          qcut = 0;  // Filter High Quality WRT any Quality
        if (PhiMP2 == 0 && qcut > 2)
          qcut = 2;  // For Th TP max quality is 3 (4-hit)

        if (!(mp2.quality > qcut))
          continue;

        int sector2 = chId2.sector();
        int wheel2 = chId2.wheel();
        int station2 = chId2.station();
        if (sector2 == 13)
          sector2 = 4;
        if (sector2 == 14)
          sector2 = 10;

        bool SectorSearch = false;
        if (sector2 == sector || sector2 == sector_p1 || sector2 == sector_m1)
          SectorSearch = true;
        if (SectorSearch != 1)
          continue;

        bool WheelSearch = false;
        if (wheel2 == wheel || wheel2 == wheel - 1 || wheel2 == wheel + 1)
          WheelSearch = true;
        if (WheelSearch != 1)
          continue;

        string whch2 = "wh" + std::to_string(wheel2) + "ch" + std::to_string(station2) + "";
        float t0_mean2, t0_width2;
        if (PhiMP2 == 1) {
          t0_mean2 = mphi_mean.find("" + whch2 + "")->second;
          t0_width2 = mphi_width.find("" + whch2 + "")->second;
        } else {
          t0_mean2 = mth_mean.find("" + whch2 + "")->second;
          t0_width2 = mth_width.find("" + whch2 + "")->second;
        }

        float t02 = (mp2.t0 - shift_back * LHC_CLK_FREQ) * ((float)TIME_TO_TDC_COUNTS / (float)LHC_CLK_FREQ);
        t02 = t02 - t0_mean2;

        float thres = t0_width + t0_width2;

        bool SameCh = false;
        if (station2 == station && sector2 == sector && wheel2 == wheel)
          SameCh = true;

        bool Wh2Exc = false;
        if (abs(wheel) == 2 && station < 3 && SameCh == 1)
          Wh2Exc = true;  // exception for WH2 MB1/2

        if (Wh2Exc == 1 && PhiMP != PhiMP2) {  // pass if Phi-Th(large k) pair in same chamber
          float k = 0;
          if (PhiMP == 0)
            k = mp.phiB;
          else
            k = mp2.phiB;

          if (wheel == 2 && k > 0.9) {
            co_found = true;
            break;
          } else if (wheel == -2 && k < -0.9) {
            co_found = true;
            break;
          }
        }

        if (co_option == 1 && PhiMP2 == 0)
          continue;  // Phi Only
        else if (co_option == 2 && PhiMP2 == 1)
          continue;  // Theta Only

        if (station2 == station)
          continue;  // Different chambers + not adjacent chambers (standard)

        if (abs(t02 - t0) < thres) {
          co_found = true;
          break;
        }
      }  // loop over all MPs and look for co

      if (co_found == 1)
        outMPs.push_back(mp);

    }  // co check decision
  }  // input MP iterator

  return outMPs;
}
