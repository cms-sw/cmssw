#ifndef gen_HEPMCFILTERDRIVER_H
#define gen_HEPMCFILTERDRIVER_H

// J.Bendavid
//class to select a HepMCFilter to run with multiple hadronization attempts
//inside HadronizerFilter

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"

//
// class declaration
//

class BaseHepMCFilter;

class HepMCFilterDriver {
public:
  HepMCFilterDriver(const edm::ParameterSet&);
  ~HepMCFilterDriver();

  bool filter(const HepMC::GenEvent* evt, double weight = 1.);
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
  BaseHepMCFilter* filter_;
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
