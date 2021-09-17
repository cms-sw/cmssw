#ifndef RecoEgamma_EgammaTools_EnergyScaleCorrection_h
#define RecoEgamma_EgammaTools_EnergyScaleCorrection_h

//author: Alan Smithee
//description:
//  A port of Shervin Nourbakhsh's EnergyScaleCorrection_class in EgammaAnalysis/ElectronTools
//  this reads the scale & smearing corrections in from a text file for given categories
//  it then allows these values to be accessed

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <bitset>

class EnergyScaleCorrection {
public:
  enum FileFormat { UNKNOWN = 0, GLOBE, ECALELF_TOY, ECALELF };

  enum ParamSmear { kNone = 0, kRho, kPhi, kNParamSmear };

  enum ScaleNuisances {
    kErrStatBitNr = 0,
    kErrSystBitNr = 1,
    kErrGainBitNr = 2,
    kErrNrBits = 3,
    kErrNone = 0,
    kErrStat = 1,
    kErrSyst = 2,
    kErrGain = 4,
    kErrStatSyst = 3,
    kErrStatGain = 5,
    kErrSystGain = 6,
    kErrStatSystGain = 7
  };

  class ScaleCorrection {
  public:
    ScaleCorrection() : scale_(1.), scaleErrStat_(0.), scaleErrSyst_(0.), scaleErrGain_(0.) {}
    ScaleCorrection(float iScale, float iScaleErrStat, float iScaleErrSyst, float iScaleErrGain)
        : scale_(iScale), scaleErrStat_(iScaleErrStat), scaleErrSyst_(iScaleErrSyst), scaleErrGain_(iScaleErrGain) {}

    float scale() const { return scale_; }
    float scaleErr(const std::bitset<kErrNrBits>& uncBitMask) const;
    float scaleErrStat() const { return scaleErrStat_; }
    float scaleErrSyst() const { return scaleErrSyst_; }
    float scaleErrGain() const { return scaleErrGain_; }

    friend std::ostream& operator<<(std::ostream& os, const ScaleCorrection& a) { return a.print(os); }
    std::ostream& print(std::ostream& os) const;

  private:
    float scale_, scaleErrStat_, scaleErrSyst_, scaleErrGain_;
  };

  struct SmearCorrection {
  public:
    SmearCorrection() : rho_(0.), rhoErr_(0.), phi_(0.), phiErr_(0.), eMean_(0.), eMeanErr_(0.) {}
    SmearCorrection(float iRho, float iRhoErr, float iPhi, float iPhiErr, float iEMean, float iEMeanErr)
        : rho_(iRho), rhoErr_(iRhoErr), phi_(iPhi), phiErr_(iPhiErr), eMean_(iEMean), eMeanErr_(iEMeanErr) {}

    friend std::ostream& operator<<(std::ostream& os, const SmearCorrection& a) { return a.print(os); }
    std::ostream& print(std::ostream& os) const;

    float sigma(const float et, const float nrSigmaRho = 0., const float nrSigmaPhi = 0.) const {
      const float rhoVal = rho_ + rhoErr_ * nrSigmaRho;
      const float phiVal = phi_ + phiErr_ * nrSigmaPhi;
      const float constTerm = rhoVal * std::sin(phiVal);
      const float alpha = rhoVal * eMean_ * std::cos(phiVal);
      return std::sqrt(constTerm * constTerm + alpha * alpha / et);
    }

  private:
    float rho_, rhoErr_;
    float phi_, phiErr_;
    float eMean_, eMeanErr_;
  };

  class CorrectionCategory {
  public:
    CorrectionCategory(const std::string& category, int runnrMin = 0, int runnrMax = 999999);
    CorrectionCategory(
        const unsigned int runnr, const float et, const float eta, const float r9, const unsigned int gainSeed)
        : runMin_(runnr),
          runMax_(runnr),
          etaMin_(std::abs(eta)),
          etaMax_(std::abs(eta)),
          r9Min_(r9),
          r9Max_(r9),
          etMin_(et),
          etMax_(et),
          gain_(gainSeed) {}

    CorrectionCategory(unsigned int runMin,
                       unsigned int runMax,
                       float etaMin,
                       float etaMax,
                       float r9Min,
                       float r9Max,
                       float etMin,
                       float etMax,
                       unsigned int gainSeed);

    bool operator<(const CorrectionCategory& b) const;
    bool inCategory(
        const unsigned int runnr, const float et, const float eta, const float r9, const unsigned int gainSeed) const;

    friend std::ostream& operator<<(std::ostream& os, const CorrectionCategory& a) { return a.print(os); }
    std::ostream& print(std::ostream& os) const;

  private:
    //all boundaries are inclusive (X<=Y<=Z)
    unsigned int runMin_;
    unsigned int runMax_;
    float etaMin_;       ///< min eta value for the bin
    float etaMax_;       ///< max eta value for the bin
    float r9Min_;        ///< min R9 vaule for the bin
    float r9Max_;        ///< max R9 value for the bin
    float etMin_;        ///< min Et value for the bin
    float etMax_;        ///< max Et value for the bin
    unsigned int gain_;  ///< 12, 6, 1, 61 (double gain switch)
  };

public:
  EnergyScaleCorrection(const std::string& correctionFileName, unsigned int genSeed = 0);
  EnergyScaleCorrection(){};
  ~EnergyScaleCorrection() {}

  float scaleCorr(unsigned int runnr,
                  double et,
                  double eta,
                  double r9,
                  unsigned int gainSeed = 12,
                  std::bitset<kErrNrBits> uncBitMask = kErrNone) const;

  float scaleCorrUncert(unsigned int runnr,
                        double et,
                        double eta,
                        double r9,
                        unsigned int gainSeed,
                        std::bitset<kErrNrBits> uncBitMask = kErrNone) const;

  float smearingSigma(
      int runnr, double et, double eta, double r9, unsigned int gainSeed, ParamSmear par, float nSigma = 0.) const;
  float smearingSigma(
      int runnr, double et, double eta, double r9, unsigned int gainSeed, float nSigmaRho, float nSigmaPhi) const;

  void setSmearingType(FileFormat value);

  const ScaleCorrection* getScaleCorr(unsigned int runnr, double et, double eta, double r9, unsigned int gainSeed) const;
  const SmearCorrection* getSmearCorr(unsigned int runnr, double et, double eta, double r9, unsigned int gainSeed) const;

private:
  void addScale(const std::string& category,
                int runMin,
                int runMax,
                double deltaP,
                double errDeltaP,
                double errSystDeltaP,
                double errDeltaPGain);

  void addScale(int runMin,
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
                double energyScaleErrGain);

  void addSmearing(const std::string& category,
                   int runMin,
                   int runMax,
                   double rho,
                   double errRho,
                   double phi,
                   double errPhi,
                   double eMean,
                   double errEMean);

  void readScalesFromFile(const std::string& filename);
  void readSmearingsFromFile(const std::string& filename);

  //static data members
  static constexpr float kDefaultScaleVal_ = 1.0;
  static constexpr float kDefaultSmearVal_ = 0.0;

  //data members
  FileFormat smearingType_;
  std::map<CorrectionCategory, ScaleCorrection> scales_;
  std::map<CorrectionCategory, SmearCorrection> smearings_;

  template <typename T1, typename T2>
  class Sorter {
  public:
    bool operator()(const std::pair<T1, T2>& lhs, const T1& rhs) const { return lhs.first < rhs; }
    bool operator()(const std::pair<T1, T2>& lhs, const std::pair<T1, T2>& rhs) const { return lhs.first < rhs.first; }
    bool operator()(const T1& lhs, const std::pair<T1, T2>& rhs) const { return lhs < rhs.first; }
    bool operator()(const T1& lhs, const T1& rhs) const { return lhs < rhs; }
  };
};

#endif
