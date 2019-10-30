#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/ME.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEChannel.h"

//GHM ClassImp(ME)

using namespace std;

const ME::TimeStamp ME::kLowMask = 0xFFFFFFFF;

TString ME::granularity[ME::iSizeG] = {"R", "SM", "LMR", "LMM", "SC", "C"};

TString ME::APDPrimVar[ME::iSizeAPD] = {"FLAG",
                                        "MEAN",
                                        "RMS",
                                        "M3",
                                        "APD_OVER_PNA_MEAN",
                                        "APD_OVER_PNA_RMS",
                                        "APD_OVER_PNA_M3",
                                        "APD_OVER_PNB_MEAN",
                                        "APD_OVER_PNB_RMS",
                                        "APD_OVER_PNB_M3",
                                        "APD_OVER_PN_MEAN",
                                        "APD_OVER_PN_RMS",
                                        "APD_OVER_PN_M3",
                                        "SHAPE_COR",
                                        "ALPHA",
                                        "BETA",
                                        "TIME_MEAN",
                                        "TIME_RMS",
                                        "TIME_M3",
                                        "TIME_NEVT"};

TString ME::PNPrimVar[ME::iSizePN] = {
    "FLAG",
    "MEAN",
    "RMS",
    "M3",
    "PNA_OVER_PNB_MEAN",
    "PNA_OVER_PNB_RMS",
    "PNA_OVER_PNB_M3",
};

TString ME::MTQPrimVar[ME::iSizeMTQ] = {
    "FIT_METHOD", "MTQ_AMPL", "MTQ_TIME", "MTQ_RISE", "MTQ_FWHM", "MTQ_FW20", "MTQ_FW80", "MTQ_SLIDING"};

TString ME::TPAPDPrimVar[ME::iSizeTPAPD] = {"FLAG", "MEAN", "RMS", "M3", "NEVT"};

TString ME::TPPNPrimVar[ME::iSizeTPPN] = {"GAIN", "MEAN", "RMS", "M3"};

TString ME::type[ME::iSizeT] = {"Laser", "TestPulse"};

TString ME::color[ME::iSizeC] = {"Blue", "Green", "Red", "IRed", "LED1", "LED2"};

std::vector<MEChannel*> ME::_trees = std::vector<MEChannel*>(4, (MEChannel*)nullptr);

bool ME::useElectronicNumbering = false;

TString ME::lmdataPath(int lmr) {
  TString out_(std::getenv("MELMDAT"));
  out_ += "/";
  out_ += ME::smName(lmr);
  out_ += "/";
  return out_;
}

TString ME::primPath(int lmr) {
  TString out_(std::getenv("MESTORE"));
  out_ += "/";
  out_ += ME::smName(lmr);
  out_ += "/";
  return out_;
}

TString ME::path() { return TString(std::getenv("MUSECAL")) + "/"; }

TString ME::rootFileName(ME::Header header, ME::Settings settings) {
  // get the laser monitoring region and super-module
  int lmr_ = ME::lmr(header.dcc, header.side);
  TString outfile_ = primPath(lmr_);
  outfile_ += "LMF_";
  outfile_ += ME::smName(lmr_);
  outfile_ += "_";
  outfile_ += header.side;
  if (settings.type == ME::iLaser) {
    switch (settings.wavelength) {
      case iBlue:
        outfile_ += "_BlueLaser";
        break;
      case iGreen:
        outfile_ += "_GreenLaser";
        break;
      case iRed:
        outfile_ += "_RedLaser";
        break;
      case iIRed:
        outfile_ += "_IRedLaser";
        break;
      default:
        break;
    }
  } else if (settings.type == ME::iTestPulse) {
    outfile_ += "_testPulse";
  }
  outfile_ += "_";
  outfile_ += header.rundir.c_str();
  outfile_ += "_TS";
  outfile_ += header.ts_beg;
  outfile_ += ".root";
  return outfile_;
}

TString ME::runListName(int lmr, int type, int color) {
  TString outfile_ = primPath(lmr);
  if (type == iLaser) {
    outfile_ += "runlist_";
    switch (color) {
      case ME::iBlue:
        outfile_ += "Blue_";
        break;
      case ME::iGreen:
        outfile_ += "Red_";
        break;
      case ME::iRed:
        outfile_ += "Red_";
        break;
      case ME::iIRed:
        outfile_ += "IRed_";
        break;
      default:
        abort();
    }
    outfile_ += "Laser";
  } else if (type == iTestPulse) {
    outfile_ += "runlist_Test_Pulse";
  }
  return outfile_;
}

std::vector<ME::Time> ME::timeDiff(Time t1, Time t2, short int& sign) {
  sign = 1;
  //  Time t1 = time_high( T1 );
  //  Time t2 = time_high( T2 );
  Time dt_s(0);
  if (t1 > t2) {
    dt_s = t1 - t2;
  } else {
    sign = -1;
    dt_s = t2 - t1;
  }
  Time dt_min = dt_s / 60;
  Time n_s = dt_s - dt_min * 60;
  Time dt_h = dt_min / 60;
  Time n_min = dt_min - dt_h * 60;
  Time dt_day = dt_h / 24;
  Time n_h = dt_h - dt_day * 24;
  Time n_day = dt_day;

  std::vector<Time> vec_;
  vec_.push_back(n_day);
  vec_.push_back(n_h);
  vec_.push_back(n_min);
  vec_.push_back(n_s);

  return vec_;
}

float ME::timeDiff(Time t1, Time t0, int tunit) {
  float sign = 1.;
  Time dt(0);
  if (t1 > t0) {
    dt = t1 - t0;
  } else {
    sign = -1.;
    dt = t0 - t1;
  }
  float dt_f = ((float)dt) * sign;
  switch (tunit) {
    case iDay:
      return dt_f / 86400.;
      break;
    case iHour:
      return dt_f / 3600.;
      break;
    case iMinute:
      return dt_f / 60.;
      break;
    default:
      return dt_f;
  };
  return 0;
}

ME::Time ME::time(float dt, Time t0, int tunit) {
  short int sign = 1;
  if (dt < 0)
    sign = -1;
  float t_ = sign * dt;
  switch (tunit) {
    case iDay:
      t_ *= 86400;
      break;
    case iHour:
      t_ *= 3600;
      break;
    case iMinute:
      t_ *= 60;
      break;
  };
  ME::Time it_ = static_cast<ME::Time>(t_);
  std::cout << "dt/it/t0/ " << dt << "/" << it_ << "/" << t0 << std::endl;
  if (sign == 1)
    return t0 + it_;
  else
    return t0 - it_;
}

ME::Time ME::time_low(TimeStamp t) { return static_cast<Time>(kLowMask & t); }

ME::Time ME::time_high(TimeStamp t) { return static_cast<Time>(t >> 32); }

TString ME::region[ME::iSizeE] = {"EE-", "EB-", "EB+", "EE+"};

int ME::ecalRegion(int ilmr) {
  assert(ilmr > 0 && ilmr <= 92);
  if (ilmr <= 36)
    return iEBM;
  ilmr -= 36;
  if (ilmr <= 36)
    return iEBP;
  ilmr -= 36;
  if (ilmr <= 10)
    return iEEP;
  return iEEM;
}

int ME::lmr(int idcc, int side) {
  int ilmr = 0;

  assert(side == 0 || side == 1);

  if (idcc > 600)
    idcc -= 600;
  assert(idcc >= 1 && idcc <= 54);
  int ireg;
  if (idcc <= 9)
    ireg = iEEM;
  else {
    idcc -= 9;
    if (idcc <= 18)
      ireg = iEBM;
    else {
      idcc -= 18;
      if (idcc <= 18)
        ireg = iEBP;
      else {
        idcc -= 18;
        if (idcc <= 9)
          ireg = iEEP;
        else
          abort();
      }
    }
  }
  if (ireg == iEEM || ireg == iEEP) {
    if (side == 1 && idcc != 8) {
      return -1;
    }
    ilmr = idcc;
    if (idcc == 9)
      ilmr++;
    if (idcc == 8 && side == 1)
      ilmr++;
  } else if (ireg == iEBM || ireg == iEBP) {
    ilmr = 2 * (idcc - 1) + side + 1;
  } else
    abort();

  if (ireg == iEBP)
    ilmr += 36;
  else if (ireg == iEEP)
    ilmr += 72;
  else if (ireg == iEEM)
    ilmr += 82;

  return ilmr;
}

std::pair<int, int> ME::dccAndSide(int ilmr) {
  int idcc = 0;
  int side = 0;

  int ireg = ecalRegion(ilmr);
  if (ireg == iEEM)
    ilmr -= 82;
  else if (ireg == iEBP)
    ilmr -= 36;
  else if (ireg == iEEP)
    ilmr -= 72;

  if (ireg == iEEM || ireg == iEEP) {
    assert(ilmr >= 1 && ilmr <= 10);
    side = 0;
    idcc = ilmr;
    if (ilmr >= 9)
      idcc--;
    if (ilmr == 9)
      side = 1;
  } else {
    assert(ilmr >= 1 && ilmr <= 36);
    idcc = (ilmr - 1) / 2 + 1;
    side = (ilmr - 1) % 2;
  }

  if (ireg > iEEM)
    idcc += 9;
  if (ireg > iEBM)
    idcc += 18;
  if (ireg > iEBP)
    idcc += 18;

  //  idcc += 600;

  return std::pair<int, int>(idcc, side);
}

void ME::regionAndSector(int ilmr, int& ireg, int& ism, int& idcc, int& side) {
  ireg = ecalRegion(ilmr);

  std::pair<int, int> ipair_ = dccAndSide(ilmr);
  idcc = ipair_.first;
  side = ipair_.second;

  ism = 0;
  if (ireg == iEEM || ireg == iEEP) {
    if (idcc > 600)
      idcc -= 600;  // also works with FEDids
    if (idcc >= 1 && idcc <= 9) {
      ism = 6 + idcc;
      if (ism > 9)
        ism -= 9;
      ism += 9;
    } else if (idcc >= 46 && idcc <= 54) {
      ism = idcc - 46 + 7;
      if (ism > 9)
        ism -= 9;
    } else
      abort();
  } else if (ireg == iEBM || ireg == iEBP) {
    if (idcc > 600)
      idcc -= 600;  // also works with FEDids
    assert(idcc >= 10 && idcc <= 45);
    ism = idcc - 9;
    if (ism > 18)
      ism -= 18;
    else
      ism += 18;
  } else
    abort();
}

TString ME::smName(int ireg, int ism) {
  TString out;
  if (ireg == ME::iEEM || ireg == ME::iEEP) {
    assert(ism >= 1 && ism <= 18);
    out = "EE+";
    if (ireg == ME::iEEM)
      out = "EE-";
    if (ism > 9)
      ism -= 9;
    out += ism;
  } else if (ireg == ME::iEBM || ireg == ME::iEBP) {
    assert(ism >= 1 && ism <= 36);
    out = "EB+";
    if (ism > 18) {
      out = "EB-";
      ism -= 18;
    }
    out += ism;
  } else
    abort();
  return out;
}

TString ME::smName(int ilmr) {
  TString out;
  int reg_(0);
  int sm_(0);
  int dcc_(0);
  int side_(0);
  ME::regionAndSector(ilmr, reg_, sm_, dcc_, side_);
  out = smName(reg_, sm_);
  return out;
}

TString ME::smNameFromDcc(int idcc) {
  int ilmr = lmr(idcc, 0);
  return smName(ilmr);
}

MEChannel* ME::lmrTree(int ilmr) { return regTree(ecalRegion(ilmr))->getDescendant(iLMRegion, ilmr); }

MEChannel* ME::regTree(int ireg) {
  assert(ireg >= iEEM && ireg <= iEEP);
  if (_trees[ireg] != nullptr)
    return _trees[ireg];

  int iEcalRegion_ = ireg;
  int iSector_ = 0;
  int iLMRegion_ = 0;
  int iLMModule_ = 0;
  int iSuperCrystal_ = 0;
  int iCrystal_ = 0;
  MEChannel* leaf_(nullptr);
  MEChannel* tree_(nullptr);

  if (iEcalRegion_ == iEBM || iEcalRegion_ == iEBP) {
    for (int isect = 1; isect <= 18; isect++) {
      iSector_ = isect;
      if (iEcalRegion_ == iEBM)
        iSector_ += 18;
      if (_trees[iEcalRegion_] == nullptr) {
        //	      std::cout << "Building the tree of crystals -- "
        //		   << ME::region[iEcalRegion_];
        _trees[iEcalRegion_] = new MEChannel(0, 0, iEcalRegion_, nullptr);
      }
      tree_ = _trees[iEcalRegion_];
      for (int iX = 0; iX < 17; iX++) {
        for (int iY = 0; iY < 4; iY++) {
          iSuperCrystal_ = MEEBGeom::tt_channel(iX, iY);
          iLMModule_ = MEEBGeom::lm_channel(iX, iY);
          for (int jx = 0; jx < 5; jx++) {
            for (int jy = 0; jy < 5; jy++) {
              int ix = 5 * iX + jx;
              int iy = 5 * iY + jy;
              if (useElectronicNumbering) {
                iCrystal_ = MEEBGeom::electronic_channel(ix, iy);
              } else {
                iCrystal_ = MEEBGeom::crystal_channel(ix, iy);
              }
              MEEBGeom::EtaPhiCoord globalCoord = MEEBGeom::globalCoord(iSector_, ix, iy);
              int ieta = globalCoord.first;
              int iphi = globalCoord.second;
              iLMRegion_ = MEEBGeom::lmr(ieta, iphi);
              leaf_ = tree_;
              leaf_ = leaf_->getDaughter(ieta, iphi, iSector_);
              leaf_ = leaf_->getDaughter(ieta, iphi, iLMRegion_);
              leaf_ = leaf_->getDaughter(ieta, iphi, iLMModule_);
              leaf_ = leaf_->getDaughter(ieta, iphi, iSuperCrystal_);
              leaf_ = leaf_->getDaughter(ieta, iphi, iCrystal_);
            }
          }
        }
      }
    }
  } else if (iEcalRegion_ == iEEM || iEcalRegion_ == iEEP) {
    int iz = 1;
    if (iEcalRegion_ == iEEM)
      iz = -1;
    if (_trees[iEcalRegion_] == nullptr) {
      //	  std::cout << "Building the tree of crystals -- "
      //	       << ME::region[iEcalRegion_];
      _trees[iEcalRegion_] = new MEChannel(0, 0, iEcalRegion_, nullptr);
    }
    tree_ = _trees[iEcalRegion_];

    for (int ilmr = 72; ilmr <= 92; ilmr++)  // force the order of Monitoring Regions
    {
      if (ecalRegion(ilmr) != iEcalRegion_)
        continue;
      for (int ilmm = 1; ilmm <= 19; ilmm++)  // force the order of Monitoring Modules
      {
        for (int iXX = 1; iXX <= 10; iXX++) {
          for (int iside = 1; iside <= 2; iside++)  // symmetrize wrt y-axis
          {
            int iX = iXX;
            if (iside == 2)
              iX = 20 - iXX + 1;
            for (int iY = 1; iY <= 20; iY++) {
              //int iSector_   = MEEEGeom::sector( iX, iY );
              int iSector_ = MEEEGeom::sm(iX, iY, iz);
              if (iSector_ < 0)
                continue;
              iLMRegion_ = MEEEGeom::lmr(iX, iY, iz);
              if (iLMRegion_ != ilmr)
                continue;
              iLMModule_ = MEEEGeom::lmmod(iX, iY);
              if (iLMModule_ != ilmm)
                continue;
              iSuperCrystal_ = MEEEGeom::sc(iX, iY);

              for (int jxx = 1; jxx <= 5; jxx++) {
                int jx = jxx;  // symmetrize...
                if (iside == 2)
                  jx = 5 - jxx + 1;

                for (int jy = 1; jy <= 5; jy++) {
                  int ix = 5 * (iX - 1) + jx;
                  int iy = 5 * (iY - 1) + jy;
                  iCrystal_ = MEEEGeom::crystal(ix, iy);
                  if (iCrystal_ < 0)
                    continue;
                  leaf_ = tree_;
                  leaf_ = leaf_->getDaughter(ix, iy, iSector_);
                  leaf_ = leaf_->getDaughter(ix, iy, iLMRegion_);
                  leaf_ = leaf_->getDaughter(ix, iy, iLMModule_);
                  leaf_ = leaf_->getDaughter(ix, iy, iSuperCrystal_);
                  leaf_ = leaf_->getDaughter(ix, iy, iCrystal_);
                }
              }
            }
          }
        }
      }
    }
  }
  //  std::cout << ".... done" << std::endl;
  return _trees[iEcalRegion_];
}

bool ME::isBarrel(int ilmr) {
  int reg_ = ecalRegion(ilmr);
  if (reg_ == iEEM || reg_ == iEEP)
    return false;
  else if (reg_ == iEBM || reg_ == iEBP)
    return true;
  else
    abort();
  return true;
}

std::pair<int, int> ME::memFromLmr(int ilmr) {
  if (isBarrel(ilmr))
    return MEEBGeom::memFromLmr(ilmr);
  else
    return MEEEGeom::memFromLmr(ilmr);
  return std::pair<int, int>();
}
std::vector<int> ME::apdRefChannels(int ilmmod, int ilmr) {
  if (isBarrel(ilmr))
    return MEEBGeom::apdRefChannels(ilmmod);
  else
    return MEEEGeom::apdRefChannels(ilmmod);
  return std::vector<int>();
}

std::vector<int> ME::lmmodFromLmr(int ilmr) {
  if (isBarrel(ilmr))
    return MEEBGeom::lmmodFromLmr(ilmr);
  else
    return MEEEGeom::lmmodFromLmr(ilmr);
  return std::vector<int>();
}

std::vector<int> ME::memFromDcc(int idcc) {
  std::vector<int> vec;
  for (int iside = 0; iside <= 1; iside++) {
    int ilmr = lmr(idcc, iside);
    if (ilmr < 0)
      continue;
    std::pair<int, int> mem_ = memFromLmr(ilmr);
    vec.push_back(mem_.first);
    vec.push_back(mem_.second);
  }
  return vec;
}

std::vector<int> ME::lmmodFromDcc(int idcc) {
  std::vector<int> vec;
  for (int iside = 0; iside <= 1; iside++) {
    int ilmr = lmr(idcc, iside);
    if (ilmr < 0)
      continue;
    bool isBarrel_ = isBarrel(ilmr);
    std::vector<int> vec_ = lmmodFromLmr(ilmr);
    for (unsigned ii = 0; ii < vec_.size(); ii++) {
      int ilmmod_ = vec_[ii];
      if (!isBarrel_) {
        // special case for Julie
        if (ilmmod_ == 18 && iside == 1)
          ilmmod_ = 20;
        if (ilmmod_ == 19 && iside == 1)
          ilmmod_ = 21;
      }
      vec.push_back(ilmmod_);
    }
  }
  return vec;
}

std::pair<int, int> ME::pn(int ilmr, int ilmmod, ME::PN ipn) {
  std::pair<int, int> pnpair_(0, 0);
  std::pair<int, int> mempair_ = memFromLmr(ilmr);
  if (isBarrel(ilmr)) {
    pnpair_ = MEEBGeom::pn(ilmmod);
  } else {
    int dee_ = MEEEGeom::dee(ilmr);
    pnpair_ = MEEEGeom::pn(dee_, ilmmod);
  }
  int mem_(0);
  int pn_(0);
  if (ipn == iPNA) {
    mem_ = mempair_.first;
    pn_ = pnpair_.first;
  } else {
    mem_ = mempair_.second;
    pn_ = pnpair_.second;
  }
  return std::pair<int, int>(mem_, pn_);
}
