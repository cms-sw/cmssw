#ifndef gen_HEPMC3FILTERDRIVER_H
#define gen_HEPMC3FILTERDRIVER_H

//class to select a HepMC3Filter to run with multiple hadronization attempts
//inside HadronizerFilter

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"
#include "GeneratorInterface/Core/interface/BaseHepMC3Filter.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"

//
// class declaration
//

class HepMC3FilterDriver {
public:
  HepMC3FilterDriver(const edm::ParameterSet&);
  ~HepMC3FilterDriver();

  bool filter(const HepMC3::GenEvent* evt, double weight = 1.);
  void statistics() const;
  void resetStatistics();

  unsigned int numEventsPassPos() const { return numEventsPassPos_; }
  unsigned int numEventsPassNeg() const { return numEventsPassNeg_; }
  unsigned int numEventsTotalPos() const { return numEventsTotalPos_; }
  unsigned int numEventsTotalNeg() const { return numEventsTotalNeg_; }
  double sumpass_w() const { return sumpass_w_; }
  double sumpass_w2() const { return sumpass_w2_; }
  double sumtotal_w() const { return sumtotal_w_; }
  double sumtotal_w2() const { return sumtotal_w2_; }

private:
  BaseHepMC3Filter* filter_;
  unsigned int numEventsPassPos_;
  unsigned int numEventsPassNeg_;
  unsigned int numEventsTotalPos_;
  unsigned int numEventsTotalNeg_;
  double sumpass_w_;
  double sumpass_w2_;
  double sumtotal_w_;
  double sumtotal_w2_;
};
#endif
