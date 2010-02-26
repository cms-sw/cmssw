#ifndef GlobalFitCorrector_h
#define GlobalFitCorrector_h

#include <map>
#include <string>
#include <vector>
#include <math.h>


namespace reco{
  class CaloJet;
}
class Parametrization;
class SimpleJetCorrectorParameters;


class GlobalFitCorrector{
  
 public:
  
  GlobalFitCorrector() : parametrization_(0), params_(0) {};
  GlobalFitCorrector(const std::string& file);
  virtual ~GlobalFitCorrector();
  
  /// apply correction using CaloJet information
  virtual double correction(const reco::CaloJet& jet) const;

 private:
  
  GlobalFitCorrector(const GlobalFitCorrector&);
  GlobalFitCorrector& operator=(const GlobalFitCorrector&);

  /// get tower index from eta
  int indexEta(double eta) const;
  /// get parametrization
  const Parametrization& parametrization() {return *parametrization_; };

 private:

  typedef std::pair<int,int>  CalibKey;
  typedef std::vector<double> CalibVal;
  typedef std::map<CalibKey,CalibVal> CalibMap;

  CalibMap jetParams_;
  CalibMap towerParams_;

  Parametrization* parametrization_;
  SimpleJetCorrectorParameters* params_;
};

#endif
