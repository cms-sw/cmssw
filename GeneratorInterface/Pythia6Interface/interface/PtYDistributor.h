#ifndef gen_PTYDISTRIBUTOR_H
#define gen_PTYDISTRIBUTOR_H

#include <fstream>
#include <iostream>
#include <sstream>

#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandGeneral.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

namespace gen
{
  class PtYDistributor {
  public:
    PtYDistributor() {};
    //PtYDistributor(std::string inputfile, CLHEP::HepRandomEngine& fRandomEngine, double ptmax, double ptmin, double ymax, double ymin, int ptbins, int ybins);
    PtYDistributor(const edm::FileInPath& fip, CLHEP::HepRandomEngine& fRandomEngine, 
                   double ptmax, double ptmin, double ymax, double ymin, 
		   int ptbins, int ybins);
    virtual ~PtYDistributor() {};

    double fireY();
    double firePt();
    double fireY(double ymin, double ymax);
    double firePt(double ptmin, double ptmax);
    
  private:
    double ptmax_;
    double ptmin_;
    double ymax_;
    double ymin_;

    int ptbins_;
    int ybins_;

    CLHEP::RandGeneral* fYGenerator;
    CLHEP::RandGeneral* fPtGenerator;

  };
}

#endif

