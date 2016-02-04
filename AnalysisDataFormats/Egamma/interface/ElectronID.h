#ifndef ElectronID_h
#define ElectronID_h

#include "AnalysisDataFormats/Egamma/interface/ElectronIDFwd.h"

namespace reco {

class ElectronID
{
 public:

  ElectronID(bool cutBasedDecision=-1,
	     double likelihood=-1.,
	     double neuralNetOutput=-1.) :
    cutBasedDecision_(cutBasedDecision),
    likelihood_(likelihood), neuralNetOutput_(neuralNetOutput) {}

  bool cutBasedDecision() const {return cutBasedDecision_;}

  double likelihood() const {return likelihood_;}

  double neuralNetOutput() const {return neuralNetOutput_;}

 private:

  bool cutBasedDecision_;
  double likelihood_;
  double neuralNetOutput_;

};

}

#endif
