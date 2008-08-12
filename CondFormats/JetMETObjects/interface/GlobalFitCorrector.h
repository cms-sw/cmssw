#ifndef GlobalFitCorrector_h
#define GlobalFitCorrector_h

#include <map>
#include <string>
#include <vector>
#include <math.h>


namespace reco{
  class CaloJet;
}

namespace CaloBoundaries{
  static const double ecalRadius   =  129;
  static const double ecalFront    =  314;
  static const double ecalBack     = -314;
  static const double hcalRadius   =  177;
  static const double hcalFront    =  500;
  static const double hcalBack     = -500;
}

class Parametrization;
class CaloSubdetectorGeometry;
class SimpleJetCorrectorParameters;

class GlobalFitCorrector{
  
 public:
  
  GlobalFitCorrector(const CaloSubdetectorGeometry* geom) : geom_(geom), params_(0) {};
  GlobalFitCorrector(const CaloSubdetectorGeometry* geom, const std::string& file);
  virtual ~GlobalFitCorrector();
  
  /// apply correction using CaloJet information
  virtual double correction(const reco::CaloJet&) const;

 private:
  
  GlobalFitCorrector(const GlobalFitCorrector&);
  GlobalFitCorrector& operator=(const GlobalFitCorrector&);

  /// get tower index from eta
  int indexEta(double eta);

  /// get parametrization
  const Parametrization& parametrization() {return *parametrization_; };

 private:

  typedef std::pair<int,int>  CalibKey;
  typedef std::vector<double> CalibVal;
  typedef std::map<CalibKey,CalibVal> CalibMap;

  CalibMap jetParams_;
  CalibMap towerParams_; 

  Parametrization* parametrization_;
  const CaloSubdetectorGeometry* geom_;
  SimpleJetCorrectorParameters* params_;
};

#endif
