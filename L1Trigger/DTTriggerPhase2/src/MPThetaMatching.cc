#include "L1Trigger/DTTriggerPhase2/interface/MPThetaMatching.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

using namespace edm;
using namespace std;
using namespace cmsdt;

namespace {
  struct {
    bool operator()(const metaPrimitive &mp1, const metaPrimitive &mp2) const {
      DTChamberId chId1(mp1.rawId);

      int sector1 = chId1.sector();
      int wheel1 = chId1.wheel();
      int station1 = chId1.station();

      DTChamberId chId2(mp2.rawId);
      DTSuperLayerId slId2(mp2.rawId);

      int sector2 = chId2.sector();
      int wheel2 = chId2.wheel();
      int station2 = chId2.station();

      // First, compare by chamber
      if (sector1 != sector2)
        return sector1 < sector2;
      if (wheel1 != wheel2)
        return wheel1 < wheel2;
      if (station1 != station2)
        return station1 < station2;

      // If they are in the same category, sort by the value (4th index)
      return mp1.quality > mp2.quality;
    }
  } const compareMPs;
}  //namespace

// ============================================================================
// Constructors and destructor
// ============================================================================
MPThetaMatching::MPThetaMatching(const ParameterSet &pset)
    : MPFilter(pset),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      th_option_(pset.getParameter<int>("th_option")),
      th_quality_(pset.getParameter<int>("th_quality")),
      scenario_(pset.getParameter<int>("scenario")) {}

MPThetaMatching::~MPThetaMatching() { finish(); }
// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MPThetaMatching::initialise(const edm::EventSetup &iEventSetup) {}

void MPThetaMatching::run(edm::Event &iEvent,
                          const edm::EventSetup &iEventSetup,
                          std::vector<metaPrimitive> &inMPaths,
                          std::vector<metaPrimitive> &outMPaths) {
  if (debug_)
    LogDebug("MPThetaMatching") << "MPThetaMatching: run";

  double shift_back = 0;  // Needed for t0 (TDC) calculation, taken from main algo
  if (scenario_ == MC)
    shift_back = 400;
  else if (scenario_ == DATA)
    shift_back = 0;
  else if (scenario_ == SLICE_TEST)
    shift_back = 400;

  if (th_option_ > 0) {
    auto filteredMPs = filter(inMPaths, shift_back);
    for (auto &mp : filteredMPs)
      outMPaths.push_back(mp);
  } else {
    if (th_option_ < 0)
      LogDebug("MPThetaMatching")
          << "MPThetaMatching: th_option can NOT be negative!!! Check settings. Saving all MPs for the moment" << endl;
    outMPaths = inMPaths;  //no filter at all
  }
}

void MPThetaMatching::finish(){};

///////////////////////////
///  OTHER METHODS

std::vector<metaPrimitive> MPThetaMatching::filter(std::vector<metaPrimitive> inMPs, double shift_back) {
  std::vector<metaPrimitive> outMPs;
  std::vector<metaPrimitive> thetaMPs;
  std::vector<metaPrimitive> phiMPs;

  //survey theta and phi MPs
  for (auto &mp : inMPs) {
    DTChamberId chId(mp.rawId);
    DTSuperLayerId slId(mp.rawId);

    if (slId.superLayer() == 2)
      thetaMPs.push_back(mp);
    else if (chId.station() == 4)  //No theta matching for MB4, save MP
      outMPs.push_back(mp);
    else if (mp.quality > th_quality_)  //don't do theta matching for q > X, save
    {
      outMPs.push_back(mp);
      phiMPs.push_back(mp);  //keep them in the loop to filter in Theta for cases where there are no other phiMPs
    } else
      phiMPs.push_back(mp);
  }

  //Order Theta MPs by quality
  std::sort(thetaMPs.begin(), thetaMPs.end(), compareMPs);

  //Use only best quality theta MP in chamber
  thetaMPs = getBestThetaMPInChamber(thetaMPs);

  // Loop on phi, save those at chamber without Theta MPs
  for (auto &mp : phiMPs) {
    DTChamberId chId(mp.rawId);
    DTSuperLayerId slId(mp.rawId);

    int sector = chId.sector();
    int wheel = chId.wheel();
    int station = chId.station();

    if (!isThereThetaMPInChamber(sector, wheel, station, thetaMPs)) {
      if (mp.quality <= th_quality_)  //Has it been saved already?
        outMPs.push_back(mp);         // No theta MPs in chamber to match, save MP
    }
  }

  // Loop on theta (already ordered)
  int oldSector = 0;
  int oldStation = 0;
  int oldWheel = -999;
  // container to decide which theta-phi pair to save
  std::vector<std::tuple<metaPrimitive, metaPrimitive, float>>
      deltaTimePosPhiCands;  //thetaMP, phiMP, difference in TimePosition
  std::vector<metaPrimitive> savedThetaMPs;

  for (metaPrimitive &mpTheta : thetaMPs) {
    DTChamberId chId(mpTheta.rawId);
    DTSuperLayerId slId(mpTheta.rawId);

    int sector = chId.sector();
    int wheel = chId.wheel();
    int station = chId.station();

    if (station == 4) {  //not possible
      LogDebug("MPThetaMatching") << "MPThetaMatching: station 4 does NOT have Theta SL 2";
      continue;
    }

    if (sector != oldSector || wheel != oldWheel || station != oldStation) {  //new chamber
      if (deltaTimePosPhiCands.size() > 0)
        orderAndSave(deltaTimePosPhiCands, &outMPs, &savedThetaMPs);

      deltaTimePosPhiCands.clear();
      oldSector = sector;
      oldWheel = wheel;
      oldStation = station;
    }

    //    float t0 = (mpTheta.t0 - shift_back * LHC_CLK_FREQ) * ((float) TIME_TO_TDC_COUNTS / (float) LHC_CLK_FREQ);
    float t0 = ((int)round(mpTheta.t0 / (float)LHC_CLK_FREQ)) - shift_back;
    float posRefZ = zFE[wheel + 2];

    if (wheel == 0 && (sector == 1 || sector == 4 || sector == 5 || sector == 8 || sector == 9 || sector == 12))
      posRefZ = -posRefZ;
    float posZ = abs(mpTheta.phi);

    // Loop in Phis
    for (metaPrimitive &mpPhi : phiMPs) {
      DTChamberId chId2(mpPhi.rawId);
      DTSuperLayerId slId2(mpPhi.rawId);

      int sector2 = chId2.sector();
      int wheel2 = chId2.wheel();
      int station2 = chId2.station();

      if (station2 != station || sector2 != sector || wheel2 != wheel)
        continue;

      float t02 = ((int)round(mpPhi.t0 / (float)LHC_CLK_FREQ)) - shift_back;

      float tphi = t02 - abs(posZ / ZRES_CONV - posRefZ) / vwire;

      int LR = -1;
      if (wheel == 0 && (sector == 3 || sector == 4 || sector == 7 || sector == 8 || sector == 11 || sector == 12))
        LR = +1;
      else if (wheel > 0)
        LR = pow(-1, wheel + sector + 1);
      else if (wheel < 0)
        LR = pow(-1, -wheel + sector);

      float posRefX = LR * xFE[station - 1];
      float ttheta = t0 - (mpPhi.x / 1000 - posRefX) / vwire;

      deltaTimePosPhiCands.push_back({mpTheta, mpPhi, abs(tphi - ttheta)});
    }  //loop in phis

    if (deltaTimePosPhiCands.size() == 0) {
      outMPs.push_back(mpTheta);  //save ThetaMP when there is no phi TPs
      savedThetaMPs.push_back(mpTheta);
    }
  }  // loop in thetas

  if (deltaTimePosPhiCands.size() > 0)
    orderAndSave(deltaTimePosPhiCands, &outMPs, &savedThetaMPs);  //do once more for last theta TP in loop

  return outMPs;
};

bool MPThetaMatching::isThereThetaMPInChamber(int sector2,
                                              int wheel2,
                                              int station2,
                                              std::vector<metaPrimitive> thetaMPs) {
  for (auto &mp1 : thetaMPs) {
    DTChamberId chId(mp1.rawId);
    DTSuperLayerId slId(mp1.rawId);

    int sector = chId.sector();
    int wheel = chId.wheel();
    int station = chId.station();
    if (sector == sector2 && wheel == wheel2 && station == station2)
      return true;
  }
  return false;
};

std::vector<metaPrimitive> MPThetaMatching::getBestThetaMPInChamber(std::vector<metaPrimitive> thetaMPs) {
  std::vector<metaPrimitive> bestThetaMPs;
  for (const auto &mp1 : thetaMPs) {
    DTChamberId chId1(mp1.rawId);

    //if there are more than 1 theta TPs in chamber, use and save only the one with highest quality
    int sector1 = chId1.sector();
    int wheel1 = chId1.wheel();
    if (wheel1 == 0) {  //Exception for Wheel 0 ONLY, to take into account background in other chambers
      int station1 = chId1.station();
      // Theta TPs (SL2) can be only q=1 (3hits) or q=3 (4 hits)
      if (mp1.quality > 1) {
        bestThetaMPs.push_back(mp1);
        continue;
      }

      int nTPs = 0;
      bool saved = false;
      // if q=1
      for (const auto &mp2 : thetaMPs) {
        DTChamberId chId2(mp2.rawId);
        DTSuperLayerId slId2(mp2.rawId);

        int sector2 = chId2.sector();
        int wheel2 = chId2.wheel();
        int station2 = chId2.station();

        if (sector1 == sector2 && wheel1 == wheel2 && station1 == station2) {
          if (mp2.quality > mp1.quality && wheel1 == 0) {
            saved = true;
            break;  //there is a q=3 and it was already saved
          } else if (mp2.quality == mp1.quality && mp2.t0 != mp1.t0) {
            saved = true;
            bestThetaMPs.push_back(mp1);
            break;  //if there are more than 1 with same q=1, save both
          } else if (abs(wheel1) == 2 && station1 < 3) {
            saved = true;
            bestThetaMPs.push_back(mp1);
            break;
          }

          nTPs++;
        }
      }
      if (nTPs == 1 && !saved)
        bestThetaMPs.push_back(mp1);  //only one Theta TP in chamber and it is q=1
    }                                 //wheel ==0

    else
      bestThetaMPs.push_back(mp1);
  }  // loop in thetaMPs

  return bestThetaMPs;
};

void MPThetaMatching::orderAndSave(std::vector<std::tuple<metaPrimitive, metaPrimitive, float>> deltaTimePosPhiCands,
                                   std::vector<metaPrimitive> *outMPs,
                                   std::vector<metaPrimitive> *savedThetaMPs) {
  //reorder deltaTimePosPhiCands according to tphi-ttheta distance
  std::sort(deltaTimePosPhiCands.begin(), deltaTimePosPhiCands.end(), comparePairs);
  int count = 0;

  for (std::tuple<metaPrimitive, metaPrimitive, float> &p :
       deltaTimePosPhiCands) {  //save up to nth nearest Theta-Phi pair candidate
    if (std::get<1>(p).quality > th_quality_)
      continue;
    DTChamberId chId(std::get<1>(p).rawId);

    if ((abs(chId.wheel()) == 2 && chId.station() < 3 && count < th_option_ + 1) ||  //save an extra pair for WH+-2MB1/2
        count < th_option_) {
      std::get<0>(p).t0 = std::get<1>(p).t0;     //replace t0 by associated phi t0
      outMPs->push_back(std::get<1>(p));         //add PhiMP
      outMPs->push_back(std::get<0>(p));         //add ThetaMP
      savedThetaMPs->push_back(std::get<0>(p));  //for accounting
      count++;
    } else
      break;  //avoid Theta duplicates when saving more than one pair
  }

  for (std::tuple<metaPrimitive, metaPrimitive, float> &p :
       deltaTimePosPhiCands) {  // save theta TP when paired Phi was above th_quality_
    DTChamberId chId(std::get<1>(p).rawId);
    if (count < th_option_ || (abs(chId.wheel()) == 2 && chId.station() < 3 && count < (th_option_ + 1))) {
      if (std::get<1>(p).quality > th_quality_) {
        //if (abs(chId.wheel())!=0)
        std::get<0>(p).t0 = std::get<1>(p).t0;  //replace t0 by associated phi t0
        outMPs->push_back(std::get<0>(p));      //add ThetaMP
        savedThetaMPs->push_back(std::get<0>(p));
        count++;
      }
    } else
      break;
  }
}
