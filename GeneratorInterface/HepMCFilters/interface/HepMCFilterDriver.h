#ifndef gen_HEPMCFILTERDRIVER_H
#define gen_HEPMCFILTERDRIVER_H

// J.Bendavid
//class to select a HepMCFilter to run with multiple hadronization attempts
//inside HadronizerFilter

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class BaseHepMCFilter;

class HepMCFilterDriver {
  public:
    HepMCFilterDriver(const edm::ParameterSet&);
    ~HepMCFilterDriver();

    bool filter(const HepMC::GenEvent* evt, double weight=1.);
    void statistics() const;
    void resetStatistics();
    
  private:
    BaseHepMCFilter *filter_;
    unsigned int ntried_;
    unsigned int naccepted_;
    double weighttried_;
    double weightaccepted_;
  
};
#endif
