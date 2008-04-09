#ifndef PTYDISTRIBUTOR_H
#define PTYDISTRIBUTOR_H

#include <fstream>
#include <iostream>
#include <sstream>

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandGeneral.h"

namespace edm
{
  class PtYDistributor {
  public:
    PtYDistributor() {};
    PtYDistributor(std::string inputfile, CLHEP::HepRandomEngine& fRandomEngine);
    virtual ~PtYDistributor() {};

    double fireY(double ymin=0, double ymax=100);
    double firePt(double ptmin=0, double ptmax=999);
    
  private:
    int theProbSize1;
    int theProbSize2;
    
    double aProbFunc1[1000];
    double aProbFunc2[1000];
    
    std::string file;

    RandGeneral* fYGenerator;
    RandGeneral* fPtGenerator;

  };
}
#endif
