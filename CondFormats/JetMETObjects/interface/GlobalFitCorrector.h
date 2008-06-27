#ifndef GlobalFitCorrector_h
#define GlobalFitCorrector_h

#include <map>
#include <string>
#include <vector>

namespace reco{
  class CaloJet;
}

class GlobalFitCorrectorParameters;

class GlobalFitCorrector{
 public:
  GlobalFitCorrector() : params_(0) {};
  GlobalFitCorrector(const std::string&);
  virtual ~GlobalFitCorrector();

  /// apply correction using CaloJet information
  virtual double correction(const reco::CaloJet&) const;

 private:
  GlobalFitCorrector(const GlobalFitCorrector&);
  GlobalFitCorrector& operator=(const GlobalFitCorrector&);

 private:
  typedef std::pair<int,int>  CalibKey;
  typedef std::vector<double> CalibVal;
  typedef std::map<CalibKey,CalibVal> CalibMap;

  GlobalFitCorrectorParameters* params_;
};

#endif
