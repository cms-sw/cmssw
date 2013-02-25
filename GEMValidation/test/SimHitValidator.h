#ifndef _SimHitValidator_h_
#define _SimHitValidator_h_

#include "BaseValidator.h"

class SimHitValidator : public BaseValidator
{
 public:
  enum Selection{Muon, NonMuon, All};

  SimHitValidator();
  ~SimHitValidator();

  void makeValidationPlots(const Selection& );
  void makeTrackValidationPlots();
  void makeValidationReport();
  void setEtaBinLabels(const TH1D* h);
  
 private:
  
};

#endif
