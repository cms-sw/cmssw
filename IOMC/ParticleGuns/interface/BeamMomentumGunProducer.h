#ifndef IOMC_ParticleGuns_BeamMomentumGunProducer_H
#define IOMC_ParticleGuns_BeamMomentumGunProducer_H

#include "IOMC/ParticleGuns/interface/FlatBaseThetaGunProducer.h"

#include <iostream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"

namespace edm {

  class BeamMomentumGunProducer : public FlatBaseThetaGunProducer {
  public:
    BeamMomentumGunProducer(const ParameterSet &);
    ~BeamMomentumGunProducer() override {}

    void produce(Event &e, const EventSetup &es) override;

  private:
    // data members
    double xoff_, yoff_, zpos_;
    TFile *fFile_;
    TTree *fTree_;
    long int nentries_;

    // Declaration of leaf types
    int npar_, eventId_;
    std::vector<int> *parPDGId_;
    std::vector<float> *parX_, *parY_, *parZ_;
    std::vector<float> *parPx_, *parPy_, *parPz_;

    // List of branches
    TBranch *b_npar_, *b_eventId_, *b_parPDGId_;
    TBranch *b_parX_, *b_parY_, *b_parZ_;
    TBranch *b_parPx_, *b_parPy_, *b_parPz_;

    static constexpr double mm2cm_ = 0.1, cm2mm_ = 10.0;
    static constexpr double MeV2GeV_ = 0.001;
  };
}  // namespace edm

#endif
