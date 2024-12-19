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
void MPThetaMatching::initialise(const edm::EventSetup &iEventSetup) {
  totINphi = 0;
  totOUTphi = 0;
  totINtheta = 0;
  totOUTtheta = 0;
  st4OUTphi = 0;
  hqOUTphi = 0;
  nothetaOUTphi = 0;
  nophiOUTtheta = 0;
}

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

  //cout<<"############NEW EVENT####"<< "inMPaths: "<<inMPaths.size()<<endl;

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

  //if(doLog) cout << "Before filter #TPs:" << inMPaths.size() << " after: " << outMPaths.size() << endl;
}

void MPThetaMatching::finish() {
  totOUTphi = totOUTphi - st4OUTphi - nothetaOUTphi - nophiOUTtheta;
  totINphi = totINphi - nothetaOUTphi - nophiOUTtheta;
  std::cout << "Rejected rate Phi: " << 1 - float(totOUTphi) / totINphi << endl;
  std::cout << "Rejected rate Theta: " << 1 - float(totOUTtheta) / totINtheta << endl;
  std::cout << "Rejected rate total: " << 1 - float(totOUTphi + totOUTtheta) / (totINtheta + totINphi) << endl;

  totOUTphi = totOUTphi - hqOUTphi;
  totINphi = totINphi - hqOUTphi;
  std::cout << "Rejected rate Phi (excluding HQ): " << 1 - float(totOUTphi) / totINphi << endl;
  std::cout << "Rejected rate total (excluding HQ): " << 1 - float(totOUTphi + totOUTtheta) / (totINtheta + totINphi)
            << endl;
};

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
    {
      outMPs.push_back(mp);
      st4OUTphi += 1;
    } else if (mp.quality > th_quality_)  //don't do theta matching for q > X, save
    {
      outMPs.push_back(mp);
      phiMPs.push_back(mp);  //keep them in the loop to filter in Theta for cases where there are no other phiMPs
      hqOUTphi += 1;
    } else
      phiMPs.push_back(mp);
  }

  bool doLog = false;
  if (thetaMPs.size() > 0 || phiMPs.size() > 0) {
    cout << "Initial #TPs Theta: " << thetaMPs.size() << " Phi: " << phiMPs.size() << endl;
    doLog = true;
  }

  totINphi += phiMPs.size();
  totINtheta += thetaMPs.size();

  //Order Theta MPs by quality
  std::sort(thetaMPs.begin(), thetaMPs.end(), compareMPs);

  //Use only best quality theta MP in chamber
  //thetaMPs = getBestThetaMPInChamber(thetaMPs);
  if (doLog)
    cout << "Intermediate #TPs Theta: " << thetaMPs.size() << endl;

  // Loop on phi, save those at chamber without Theta MPs
  for (auto &mp : phiMPs) {
    DTChamberId chId(mp.rawId);
    DTSuperLayerId slId(mp.rawId);

    int sector = chId.sector();
    int wheel = chId.wheel();
    int station = chId.station();

    if (!isThereThetaMPInChamber(sector, wheel, station, thetaMPs)) {
      if (mp.quality <= th_quality_) {  //Has it been saved already?
        outMPs.push_back(mp);           // No theta MPs in chamber to match, save MP
        nothetaOUTphi += 1;
      }
    }
  }

  // Loop on theta (already ordered)
  int oldSector = 0;
  int oldStation = 0;
  int oldWheel = 0;
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
    if (doLog)
      cout << "Theta wh se st q t0: " << wheel << " " << sector << " " << station << " " << mpTheta.quality << " "
           << mpTheta.t0 << endl;

    if (wheel == 0 && (sector == 1 || sector == 4 || sector == 5 || sector == 8 || sector == 9 || sector == 12))
      posRefZ = -posRefZ;
    float posZ = abs(mpTheta.phi);
    //cout << "t0: " << t0 << endl;
    //cout << "posZ: " << posZ << endl;

    // Loop in Phis
    for (metaPrimitive &mpPhi : phiMPs) {
      DTChamberId chId2(mpPhi.rawId);
      DTSuperLayerId slId2(mpPhi.rawId);

      int sector2 = chId2.sector();
      int wheel2 = chId2.wheel();
      int station2 = chId2.station();
      if (doLog)
        cout << "Phi wheel: " << wheel2 << " sector " << sector2 << " station: " << station2 << " q: " << mpPhi.quality
             << " t0: " << mpPhi.t0 << endl;

      if (station2 != station || sector2 != sector || wheel2 != wheel)
        continue;

      //     float t02 = (mpPhi.t0 - shift_back * LHC_CLK_FREQ) * ((float) TIME_TO_TDC_COUNTS / (float) LHC_CLK_FREQ);
      float t02 = ((int)round(mpPhi.t0 / (float)LHC_CLK_FREQ)) - shift_back;

      //cout << "posRefZ: " << posRefZ << " Z: " << posZ / ZRES_CONV << endl;
      float tphi = t02 - abs(posZ / ZRES_CONV - posRefZ) / vwire;
      //cout << "tphi: " << tphi << endl;

      int LR = -1;
      if (wheel == 0 && (sector == 3 || sector == 4 || sector == 7 || sector == 8 || sector == 11 || sector == 12))
        LR = +1;
      else if (wheel > 0)
        LR = pow(-1, wheel + sector + 1);
      else if (wheel < 0)
        LR = pow(-1, -wheel + sector);
      //cout << "wh st se: " << wheel << " " << station << " " << sector << " LR: " << LR << endl;

      float posRefX = LR * xFE[station - 1];
      float ttheta = t0 - (mpPhi.x / 1000 - posRefX) / vwire;
      //cout << "posRefX: " << posRefX << " X: " << mp2.x / 1000 << endl;
      //cout << "ttheta: " << ttheta << endl;

      deltaTimePosPhiCands.push_back({mpTheta, mpPhi, abs(tphi - ttheta)});
      //cout << "deltaTimePosPhiCands: " << deltaTimePosPhiCands.size() << endl;
    }  //loop in phis

    if (deltaTimePosPhiCands.size() == 0) {
      outMPs.push_back(mpTheta);  //save ThetaMP when there is no phi TPs
      savedThetaMPs.push_back(mpTheta);
      nophiOUTtheta += 1;
    }
  }  // loop in thetas

  if (deltaTimePosPhiCands.size() > 0)
    orderAndSave(deltaTimePosPhiCands, &outMPs, &savedThetaMPs);  //do once more for last theta TP in loop

  if (doLog)
    cout << "After #TPs Theta: " << savedThetaMPs.size() << " Phi: " << outMPs.size() - savedThetaMPs.size() << endl;

  if (inMPs.size() > outMPs.size())
    cout << " ########### TP size reduced from: " << inMPs.size() << " to: " << outMPs.size()
         << " rate: " << float(outMPs.size()) / inMPs.size() << endl;
  else if (outMPs.size() > inMPs.size())
    cout << " $$$$$$$$$$$$$ TP size enlarged from: " << inMPs.size() << " to: " << outMPs.size() << endl;

  if (phiMPs.size() > (outMPs.size() - savedThetaMPs.size()))
    cout << " ########### TP Phi size reduced from: " << phiMPs.size()
         << " to: " << outMPs.size() - savedThetaMPs.size()
         << " rate: " << float((outMPs.size() - savedThetaMPs.size())) / phiMPs.size() << endl;

  if (thetaMPs.size() > savedThetaMPs.size())
    cout << " ########### TP Theta size reduced from: " << thetaMPs.size() << " to: " << savedThetaMPs.size()
         << " rate: " << float(savedThetaMPs.size()) / thetaMPs.size() << endl;

  totOUTtheta = totOUTtheta + savedThetaMPs.size();
  totOUTphi = totOUTphi + (outMPs.size() - savedThetaMPs.size());

  for (const auto &mp1 : outMPs) {
    cout << "FINAL OUTPUT OF WHAT IT IS SAVED" << endl;
    DTChamberId chId1(mp1.rawId);
    DTSuperLayerId slId1(mp1.rawId);

    int sector1 = chId1.sector();
    int wheel1 = chId1.wheel();
    int station1 = chId1.station();
    if (slId1.superLayer() == 2)
      cout << "Theta wheel: " << wheel1 << " sector " << sector1 << " station1: " << station1 << " q: " << mp1.quality
           << " t0: " << mp1.t0 << endl;
    else
      cout << "Phi  wheel: " << wheel1 << " sector " << sector1 << " station1: " << station1 << " q: " << mp1.quality
           << " t0: " << mp1.t0 << endl;
    cout << endl;
  }

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
    int station1 = chId1.station();
    cout << "Theta wheel: " << wheel1 << " sector " << sector1 << " station1: " << station1 << " q: " << mp1.quality
         << " t0: " << mp1.t0 << endl;
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
        if (mp2.quality > mp1.quality && (abs(wheel1) != 2 || station1 > 2)) {  //exception for WH+-2 MB1/2
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
  }

  return bestThetaMPs;
};

void MPThetaMatching::orderAndSave(std::vector<std::tuple<metaPrimitive, metaPrimitive, float>> deltaTimePosPhiCands,
                                   std::vector<metaPrimitive> *outMPs,
                                   std::vector<metaPrimitive> *savedThetaMPs) {
  //reorder deltaTimePosPhiCands according to tphi-ttheta distance
  std::sort(deltaTimePosPhiCands.begin(), deltaTimePosPhiCands.end(), comparePairs);
  int count = 0;

  for ( std::tuple<metaPrimitive, metaPrimitive, float> &p :
       deltaTimePosPhiCands) {  //save up to nth nearest Theta-Phi pair candidate
    cout << "Count: " << count << " abs(tphi-ttheta): " << std::get<2>(p) << " qTheta: " << std::get<0>(p).quality
         << " qPhi: " << std::get<1>(p).quality << endl;
    if (std::get<1>(p).quality > th_quality_)
      continue;
    DTChamberId chId(std::get<1>(p).rawId);

    //if ((abs(chId.wheel()) == 2 && chId.station() < 3 && count < th_option_ + 1) ||  //save an extra pair for WH+-2MB1/2
        if( count < th_option_) {
      std::get<0>(p).t0 = std::get<1>(p).t0;
      outMPs->push_back(std::get<1>(p));         //add PhiMP
      outMPs->push_back(std::get<0>(p));         //add ThetaMP
      savedThetaMPs->push_back(std::get<0>(p));  //for accounting
      count++;
    } else
      break;  //FIXME: avoid Theta duplicates when saving more than one pair
  }

  for (std::tuple<metaPrimitive, metaPrimitive, float> &p : deltaTimePosPhiCands) {
    DTChamberId chId(std::get<1>(p).rawId);
    if (count < th_option_ ){ //|| (abs(chId.wheel()) == 2 && chId.station() < 3 && count < (th_option_ + 1))) {
      if (std::get<1>(p).quality > th_quality_) {
        std::get<0>(p).t0 = std::get<1>(p).t0;
        outMPs->push_back(std::get<0>(p));  //add ThetaMP
        savedThetaMPs->push_back(std::get<0>(p));
        cout << "Count: " << count << " abs(tphi-ttheta): " << std::get<2>(p) << " qTheta: " << std::get<0>(p).quality
             << " qPhi: " << std::get<1>(p).quality << " no Phis below qcut to match" << endl;
        count++;
      }
    } else
      break;
  }
}
