#include "RecoEgamma/EgammaTools/interface/EnergyScaleCorrection.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <iterator>

EnergyScaleCorrection::EnergyScaleCorrection(const std::string& correctionFileName, unsigned int genSeed)
    : smearingType_(ECALELF) {
  if (!correctionFileName.empty()) {
    std::string filename = correctionFileName + "_scales.dat";
    readScalesFromFile(filename);
    if (scales_.empty()) {
      throw cms::Exception("EnergyScaleCorrection") << "scale correction map empty";
    }
  }

  if (!correctionFileName.empty()) {
    std::string filename = correctionFileName + "_smearings.dat";
    readSmearingsFromFile(filename);
    if (smearings_.empty()) {
      throw cms::Exception("EnergyScaleCorrection") << "smearing correction map empty";
    }
  }
}

float EnergyScaleCorrection::scaleCorr(unsigned int runNumber,
                                       double et,
                                       double eta,
                                       double r9,
                                       unsigned int gainSeed,
                                       std::bitset<kErrNrBits> uncBitMask) const {
  const ScaleCorrection* scaleCorr = getScaleCorr(runNumber, et, eta, r9, gainSeed);
  if (scaleCorr != nullptr)
    return scaleCorr->scale();
  else
    return kDefaultScaleVal_;
}

float EnergyScaleCorrection::scaleCorrUncert(unsigned int runNumber,
                                             double et,
                                             double eta,
                                             double r9,
                                             unsigned int gainSeed,
                                             std::bitset<kErrNrBits> uncBitMask) const {
  const ScaleCorrection* scaleCorr = getScaleCorr(runNumber, et, eta, r9, gainSeed);
  if (scaleCorr != nullptr)
    return scaleCorr->scaleErr(uncBitMask);
  else
    return 0.;
}

float EnergyScaleCorrection::smearingSigma(
    int runnr, double et, double eta, double r9, unsigned int gainSeed, ParamSmear par, float nSigma) const {
  if (par == kRho)
    return smearingSigma(runnr, et, eta, r9, gainSeed, nSigma, 0.);
  if (par == kPhi)
    return smearingSigma(runnr, et, eta, r9, gainSeed, 0., nSigma);
  return smearingSigma(runnr, et, eta, r9, gainSeed, 0., 0.);
}

float EnergyScaleCorrection::smearingSigma(
    int runnr, double et, double eta, double r9, unsigned int gainSeed, float nrSigmaRho, float nrSigmaPhi) const {
  const SmearCorrection* smearCorr = getSmearCorr(runnr, et, eta, r9, gainSeed);

  if (smearCorr != nullptr)
    return smearCorr->sigma(nrSigmaRho, nrSigmaPhi);
  else
    return kDefaultSmearVal_;
}

const EnergyScaleCorrection::ScaleCorrection* EnergyScaleCorrection::getScaleCorr(
    unsigned int runnr, double et, double eta, double r9, unsigned int gainSeed) const {
  // buld the category based on the values of the object
  CorrectionCategory category(runnr, et, eta, r9, gainSeed);
  auto result = scales_.find(category);

  if (result == scales_.end()) {
    edm::LogInfo("EnergyScaleCorrection")
        << "Scale category not found: " << category << " Returning uncorrected value.";
    return nullptr;
  }

  //validate the result, just to be sure
  if (!result->first.inCategory(runnr, et, eta, r9, gainSeed)) {
    throw cms::Exception("LogicError") << " error found scale category " << result->first
                                       << " that does not contain run " << runnr << " et " << et << " eta " << eta
                                       << " r9 " << r9 << " gain seed " << gainSeed;
  }
  return &result->second;
}

const EnergyScaleCorrection::SmearCorrection* EnergyScaleCorrection::getSmearCorr(
    unsigned int runnr, double et, double eta, double r9, unsigned int gainSeed) const {
  // buld the category based on the values of the object
  CorrectionCategory category(runnr, et, eta, r9, gainSeed);
  auto result = smearings_.find(category);

  if (result == smearings_.end()) {
    edm::LogInfo("EnergyScaleCorrection")
        << "Smear category not found: " << category << " Returning uncorrected value.";
    return nullptr;
  }

  //validate the result, just to be sure
  if (!result->first.inCategory(runnr, et, eta, r9, gainSeed)) {
    throw cms::Exception("LogicError") << " error found smear category " << result->first
                                       << " that does not contain run " << runnr << " et " << et << " eta " << eta
                                       << " r9 " << r9 << " gain seed " << gainSeed;
  }
  return &result->second;
}

void EnergyScaleCorrection::addScale(const std::string& category,
                                     int runMin,
                                     int runMax,
                                     double energyScale,
                                     double energyScaleErrStat,
                                     double energyScaleErrSyst,
                                     double energyScaleErrGain) {
  CorrectionCategory cat(category, runMin, runMax);  // build the category from the string
  auto result = scales_.equal_range(cat);
  if (result.first != result.second) {
    throw cms::Exception("ConfigError") << "Category already defined! " << cat;
  }

  ScaleCorrection corr(energyScale, energyScaleErrStat, energyScaleErrSyst, energyScaleErrGain);
  scales_.insert(result.first, {cat, corr});  //use a hint where to insert
}

void EnergyScaleCorrection::addScale(int runMin,
                                     int runMax,
                                     double etaMin,
                                     double etaMax,
                                     double r9Min,
                                     double r9Max,
                                     double etMin,
                                     double etMax,
                                     unsigned int gain,
                                     double energyScale,
                                     double energyScaleErrStat,
                                     double energyScaleErrSyst,
                                     double energyScaleErrGain) {
  CorrectionCategory cat(runMin, runMax, etaMin, etaMax, r9Min, r9Max, etMin, etMax, gain);

  auto result = scales_.equal_range(cat);
  if (result.first != result.second) {
    throw cms::Exception("ConfigError") << "Category already defined! " << cat;
  }

  ScaleCorrection corr(energyScale, energyScaleErrStat, energyScaleErrSyst, energyScaleErrGain);
  scales_.insert(result.first, {cat, corr});
}

void EnergyScaleCorrection::addSmearing(const std::string& category,
                                        int runMin,
                                        int runMax,
                                        double rho,
                                        double errRho,
                                        double phi,
                                        double errPhi,
                                        double eMean,
                                        double errEMean) {
  CorrectionCategory cat(category);
  auto res = smearings_.equal_range(cat);

  if (res.first != res.second) {
    throw cms::Exception("EnergyScaleCorrection") << "Smearing category already defined " << cat;
  }

  SmearCorrection corr(rho, errRho, phi, errPhi, eMean, errEMean);
  smearings_.insert(res.first, {cat, corr});  //use a hint from res
}

void EnergyScaleCorrection::setSmearingType(FileFormat value) {
  if (value <= 1) {
    smearingType_ = value;
  } else {
    smearingType_ = UNKNOWN;
  }
}

void EnergyScaleCorrection::readScalesFromFile(const std::string& filename) {
  std::ifstream file(edm::FileInPath(filename).fullPath().c_str());

  if (!file.good()) {
    throw cms::Exception("EnergyScaleCorrection") << "file " << filename << " not readable.";
  }

  int runMin, runMax;
  std::string category, region2;
  double energyScale, energyScaleErr, energyScaleErrStat, energyScaleErrSyst, energyScaleErrGain;

  double etaMin;      ///< Min eta value for the bin
  double etaMax;      ///< Max eta value for the bin
  double r9Min;       ///< Min R9 vaule for the bin
  double r9Max;       ///< Max R9 value for the bin
  double etMin;       ///< Min Et value for the bin
  double etMax;       ///< Max Et value for the bin
  unsigned int gain;  ///< 12, 6, 1, 61 (double gain switch)

  // TO count the #columns in that txt file and decide based on that the version to read
  std::string line;
  std::stringstream stream;
  getline(file, line);
  stream.clear();
  stream << line;

  int ncols = std::distance(std::istream_iterator<std::string>(stream), std::istream_iterator<std::string>());

  file.seekg(0, std::ios::beg);

  if (ncols == 9) {
    for (file >> category; file.good(); file >> category) {
      file >> region2 >> runMin >> runMax >> energyScale >> energyScaleErr >> energyScaleErrStat >>
          energyScaleErrSyst >> energyScaleErrGain;
      addScale(category, runMin, runMax, energyScale, energyScaleErrStat, energyScaleErrSyst, energyScaleErrGain);
    }
  } else {
    if (file.peek() == 'r')
      file.ignore(1000, 10);

    for (file >> runMin; file.good(); file >> runMin) {
      file >> runMax >> etaMin >> etaMax >> r9Min >> r9Max >> etMin >> etMax >> gain >> energyScale >> energyScaleErr;
      file.ignore(1000, 10);
      energyScaleErrStat = energyScaleErr;
      energyScaleErrSyst = 0;
      energyScaleErrGain = 0;

      addScale(runMin,
               runMax,
               etaMin,
               etaMax,
               r9Min,
               r9Max,
               etMin,
               etMax,
               gain,
               energyScale,
               energyScaleErrStat,
               energyScaleErrSyst,
               energyScaleErrGain);
    }
  }

  file.close();
  return;
}

//also more or less untouched function from the orginal package
void EnergyScaleCorrection::readSmearingsFromFile(const std::string& filename) {
  std::ifstream file(edm::FileInPath(filename).fullPath().c_str());
  if (!file.good()) {
    throw cms::Exception("EnergyScaleCorrection") << "file " << filename << " not readable";
  }

  int runMin = 0;
  int runMax = 900000;
  int unused = 0;
  std::string category, region2;
  double rho, phi, eMean, errRho, errPhi, errEMean;
  double etaMin, etaMax, r9Min, r9Max;
  std::string phiString, errPhiString;

  while (file.peek() != EOF && file.good()) {
    if (file.peek() == 10) {  // 10 = \n
      file.get();
      continue;
    }

    if (file.peek() == 35) {  // 35 = #
      file.ignore(1000, 10);  // ignore the rest of the line until \n
      continue;
    }

    if (smearingType_ == UNKNOWN) {  // trying to guess: not recommended
      throw cms::Exception("ConfigError") << "unknown smearing type";

    } else if (smearingType_ == GLOBE) {
      file >> category >> unused >> etaMin >> etaMax >> r9Min >> r9Max >> runMin >> runMax >> eMean >> errEMean >>
          rho >> errRho >> phi >> errPhi;

      addSmearing(category, runMin, runMax, rho, errRho, phi, errPhi, eMean, errEMean);

    } else if (smearingType_ == ECALELF) {
      file >> category >> eMean >> errEMean >> rho >> errRho >> phiString >> errPhiString;

      if (phiString == "M_PI_2")
        phi = M_PI_2;
      else
        phi = std::stod(phiString);

      if (errPhiString == "M_PI_2")
        errPhi = M_PI_2;
      else
        errPhi = std::stod(errPhiString);

      addSmearing(category, runMin, runMax, rho, errRho, phi, errPhi, eMean, errEMean);

    } else {
      file >> category >> rho >> phi;
      errRho = errPhi = eMean = errEMean = 0;
      addSmearing(category, runMin, runMax, rho, errRho, phi, errPhi, eMean, errEMean);
    }
  }

  file.close();
  return;
}

std::ostream& EnergyScaleCorrection::ScaleCorrection::print(std::ostream& os) const {
  os << "( " << scale_ << " +/- " << scaleErrStat_ << " +/- " << scaleErrSyst_ << " +/- " << scaleErrGain_ << ")";
  return os;
}

float EnergyScaleCorrection::ScaleCorrection::scaleErr(const std::bitset<kErrNrBits>& uncBitMask) const {
  double totErr(0);
  auto pow2 = [](const double& x) { return x * x; };

  if (uncBitMask.test(kErrStatBitNr))
    totErr += pow2(scaleErrStat_);
  if (uncBitMask.test(kErrSystBitNr))
    totErr += pow2(scaleErrSyst_);
  if (uncBitMask.test(kErrGainBitNr))
    totErr += pow2(scaleErrGain_);

  return std::sqrt(totErr);
}

std::ostream& EnergyScaleCorrection::SmearCorrection::print(std::ostream& os) const {
  os << rho_ << " +/- " << rhoErr_ << "\t" << phi_ << " +/- " << phiErr_ << "\t" << eMean_ << " +/- " << eMeanErr_;
  return os;
}

//here be dragons
//this function is nasty and needs to be replaced
EnergyScaleCorrection::CorrectionCategory::CorrectionCategory(const std::string& category, int runnrMin, int runnrMax)
    : runMin_(runnrMin),
      runMax_(runnrMax),
      etaMin_(0),
      etaMax_(3),
      r9Min_(-1),
      r9Max_(999),
      etMin_(0),
      etMax_(9999999),
      gain_(0) {
  size_t p1, p2;  // boundary

  // eta region
  p1 = category.find("absEta_");
  if (category.find("absEta_0_1") != std::string::npos) {
    etaMin_ = 0;
    etaMax_ = 1;
  } else if (category.find("absEta_1_1.4442") != std::string::npos) {
    etaMin_ = 1;
    etaMax_ = 1.479;
  } else if (category.find("absEta_1.566_2") != std::string::npos) {
    etaMin_ = 1.479;
    etaMax_ = 2;
  } else if (category.find("absEta_2_2.5") != std::string::npos) {
    etaMin_ = 2;
    etaMax_ = 3;
  } else {
    if (p1 != std::string::npos) {
      p1 = category.find('_', p1);
      p2 = category.find('_', p1 + 1);
      etaMin_ = std::stof(category.substr(p1 + 1, p2 - p1 - 1));
      p1 = p2;
      p2 = category.find('-', p1);
      etaMax_ = std::stof(category.substr(p1 + 1, p2 - p1 - 1));
    }
  }

  if (category.find("EBlowEta") != std::string::npos) {
    etaMin_ = 0;
    etaMax_ = 1;
  };
  if (category.find("EBhighEta") != std::string::npos) {
    etaMin_ = 1;
    etaMax_ = 1.479;
  };
  if (category.find("EElowEta") != std::string::npos) {
    etaMin_ = 1.479;
    etaMax_ = 2;
  };
  if (category.find("EEhighEta") != std::string::npos) {
    etaMin_ = 2;
    etaMax_ = 7;
  };

  // Et region
  p1 = category.find("-Et_");

  if (p1 != std::string::npos) {
    p1 = category.find('_', p1);
    p2 = category.find('_', p1 + 1);
    etMin_ = std::stof(category.substr(p1 + 1, p2 - p1 - 1));
    p1 = p2;
    p2 = category.find('-', p1);
    etMax_ = std::stof(category.substr(p1 + 1, p2 - p1 - 1));
  }

  if (category.find("gold") != std::string::npos || category.find("Gold") != std::string::npos ||
      category.find("highR9") != std::string::npos) {
    r9Min_ = 0.94;
    r9Max_ = std::numeric_limits<float>::max();
  } else if (category.find("bad") != std::string::npos || category.find("Bad") != std::string::npos ||
             category.find("lowR9") != std::string::npos) {
    r9Min_ = -1;
    r9Max_ = 0.94;
  };
  // R9 region
  p1 = category.find("-R9");
  if (p1 != std::string::npos) {
    p1 = category.find('_', p1);
    p2 = category.find('_', p1 + 1);
    r9Min_ = std::stof(category.substr(p1 + 1, p2 - p1 - 1));
    // If there is one value, just set lower bound
    if (p2 != std::string::npos) {
      p1 = p2;
      p2 = category.find('-', p1);
      r9Max_ = std::stof(category.substr(p1 + 1, p2 - p1 - 1));
      if (r9Max_ >= 1.0)
        r9Max_ = std::numeric_limits<float>::max();
    }
  }
  //------------------------------
  p1 = category.find("gainEle_");  // Position of first character
  if (p1 != std::string::npos) {
    p1 += 8;                      // Position of character after _
    p2 = category.find('-', p1);  // Position of - or end of string
    gain_ = std::stoul(category.substr(p1, p2 - p1), nullptr);
  }
  //so turns out the code does an inclusive X<=Y<=Z search for bins
  //which is what we want for run numbers
  //however then the problem is when we get a value exactly at the bin boundary
  //for the et/eta/r9 which then gives multiple bins
  //so we just decrement the maxValues ever so slightly to ensure that they are different
  //from the next bins min value
  etMax_ = std::nextafterf(etMax_, std::numeric_limits<float>::min());
  etaMax_ = std::nextafterf(etaMax_, std::numeric_limits<float>::min());
  r9Max_ = std::nextafterf(r9Max_, std::numeric_limits<float>::min());
}

///for the new file format
EnergyScaleCorrection::CorrectionCategory::CorrectionCategory(unsigned int runMin,
                                                              unsigned int runMax,
                                                              float etaMin,
                                                              float etaMax,
                                                              float r9Min,
                                                              float r9Max,
                                                              float etMin,
                                                              float etMax,
                                                              unsigned int gainSeed)
    : runMin_(runMin),
      runMax_(runMax),
      etaMin_(etaMin),
      etaMax_(etaMax),
      r9Min_(r9Min),
      r9Max_(r9Max),
      etMin_(etMin),
      etMax_(etMax),
      gain_(gainSeed) {
  ///Same logic as the above constructor to avoid problems at the bin
  ///boundary of et/eta/R9 - just decrement the maxValues
  ///ever so slightly to ensure that they are different
  ///from the next bins min value
  etMax_ = std::nextafterf(etMax_, std::numeric_limits<float>::min());
  etaMax_ = std::nextafterf(etaMax_, std::numeric_limits<float>::min());
  r9Max_ = std::nextafterf(r9Max_, std::numeric_limits<float>::min());
};

bool EnergyScaleCorrection::CorrectionCategory::inCategory(
    const unsigned int runnr, const float et, const float eta, const float r9, const unsigned int gainSeed) const {
  return runnr >= runMin_ && runnr <= runMax_ && et >= etMin_ && et <= etMax_ && eta >= etaMin_ && eta <= etaMax_ &&
         r9 >= r9Min_ && r9 <= r9Max_ && (gain_ == 0 || gainSeed == gain_);
}

bool EnergyScaleCorrection::CorrectionCategory::operator<(const EnergyScaleCorrection::CorrectionCategory& b) const {
  if (runMin_ < b.runMin_ && runMax_ < b.runMax_)
    return true;
  if (runMax_ > b.runMax_ && runMin_ > b.runMin_)
    return false;

  if (etaMin_ < b.etaMin_ && etaMax_ < b.etaMax_)
    return true;
  if (etaMax_ > b.etaMax_ && etaMin_ > b.etaMin_)
    return false;

  if (r9Min_ < b.r9Min_ && r9Max_ < b.r9Max_)
    return true;
  if (r9Max_ > b.r9Max_ && r9Min_ > b.r9Min_)
    return false;

  if (etMin_ < b.etMin_ && etMax_ < b.etMax_)
    return true;
  if (etMax_ > b.etMax_ && etMin_ > b.etMin_)
    return false;

  if (gain_ == 0 || b.gain_ == 0)
    return false;  // if corrections are not categorized in gain then default gain value should always return false in order to have a match with the category
  if (gain_ < b.gain_)
    return true;
  else
    return false;
  return false;
}

std::ostream& EnergyScaleCorrection::CorrectionCategory::print(std::ostream& os) const {
  os << runMin_ << " " << runMax_ << "\t" << etaMin_ << " " << etaMax_ << "\t" << r9Min_ << " " << r9Max_ << "\t"
     << etMin_ << " " << etMax_ << "\t" << gain_;
  return os;
}
