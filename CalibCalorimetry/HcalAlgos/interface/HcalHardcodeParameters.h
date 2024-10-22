#ifndef HcalAlgos_HcalHardcodeParameters_h
#define HcalAlgos_HcalHardcodeParameters_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMRadiationDamage.h"

#include <vector>

class HcalHardcodeParameters {
public:
  //default constructor
  HcalHardcodeParameters() {}

  //construct from values
  HcalHardcodeParameters(double pedestal,
                         double pedestalWidth,
                         const std::vector<double>& gain,
                         const std::vector<double>& gainWidth,
                         int zsThreshold,
                         int qieType,
                         const std::vector<double>& qieOffset,
                         const std::vector<double>& qieSlope,
                         int mcShape,
                         int recoShape,
                         double photoelectronsToAnalog,
                         const std::vector<double>& darkCurrent,
                         const std::vector<double>& noiseCorrelation,
                         double noiseThreshold,
                         double seedThreshold);

  //construct from pset
  HcalHardcodeParameters(const edm::ParameterSet& p);

  //destructor
  inline virtual ~HcalHardcodeParameters() {}

  //accessors
  //note: all vector accessors use at() in order to throw exceptions for malformed conditions
  inline double pedestal() const { return pedestal_; }
  inline double pedestalWidth() const { return pedestalWidth_; }
  inline double gain(unsigned index) const { return gain_.at(index); }
  inline double gainWidth(unsigned index) const { return gainWidth_.at(index); }
  inline int zsThreshold() const { return zsThreshold_; }
  inline int qieType() const { return qieType_; }
  inline double qieOffset(unsigned range) const { return qieOffset_.at(range); }
  inline double qieSlope(unsigned range) const { return qieSlope_.at(range); }
  inline int mcShape() const { return mcShape_; }
  inline int recoShape() const { return recoShape_; }
  inline double photoelectronsToAnalog() const { return photoelectronsToAnalog_; }
  double darkCurrent(unsigned index, double intlumi) const;
  double noiseCorrelation(unsigned index) const;
  inline double noiseThreshold() const { return noiseThreshold_; }
  inline double seedThreshold() const { return seedThreshold_; }

private:
  //member variables
  double pedestal_, pedestalWidth_;
  std::vector<double> gain_, gainWidth_;
  int zsThreshold_;
  int qieType_;
  std::vector<double> qieOffset_, qieSlope_;
  int mcShape_, recoShape_;
  double photoelectronsToAnalog_;
  std::vector<double> darkCurrent_;
  std::vector<double> noiseCorrelation_;
  bool doSipmRadiationDamage_;
  HcalSiPMRadiationDamage sipmRadiationDamage_;
  double noiseThreshold_, seedThreshold_;
};

#endif
