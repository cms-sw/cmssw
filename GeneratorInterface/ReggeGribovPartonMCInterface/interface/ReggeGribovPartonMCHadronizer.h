#ifndef REGGEGRIBOVPARTONMCHADRONIZER_H
#define REGGEGRIBOVPARTONMCHADRONIZER_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

#include "boost/scoped_ptr.hpp"

#include <map>
#include <string>
#include <vector>
#include <math.h>

boost::scoped_ptr<CLHEP::RandFlat> gFlatDistribution_; // yes, this must be global...

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
}

extern "C"
{
  void crmc_f_( int&, int&, int&, double&, int&, double&, double&, double&,
                double&, double&);
}

extern "C"
{
  void crmc_init_f_( int&, int&, double&, double&, int&, int&, int&, int&, const char*);
}

extern "C"
{
  extern struct
  {
    float sigtot; // ........ total cross section
    float sigcut;
    float sigela;
    float sloela;
    float sigsd;
    float sigine;
    float sigdif;
    float sigineaa;
    float sigtotaa;
    float sigelaaa;
    float sigcutaa;
  } hadr5_; //crmc-aaa.f
}

extern "C"
{
  extern struct
  {
    float phievt; // ........ angle of impact parameter
    int   nevt;   // ........ error code. 1=valid event, 0=invalid event
    float bimevt; // ........ absolute value of impact parameter
    int   kolevt; // ........ number of collisions
    int   koievt; // ........ number of inelastic collisions
    float pmxevt; // ........ reference momentum
    float egyevt; // ........ pp cm energy (hadron) or string energy (lepton)
    int   npjevt; // ........ number of primary projectile participants
    int   ntgevt; // ........ number of primary target participants
    int   npnevt; // ........ number of primary projectile neutron spectators
    int   nppevt; // ........ number of primary projectile proton spectators
    int   ntnevt; // ........ number of primary target neutron spectators
    int   ntpevt; // ........ number of primary target proton spectators
    int   jpnevt; // ........ number of absolute projectile neutron spectators
    int   jppevt; // ........ number of absolute projectile proton spectators
    int   jtnevt; // ........ number of absolute target neutron spectators
    int   jtpevt; // ........ number of absolute target proton spectators
    float xbjevt; // ........ bjorken x for dis
    float qsqevt; // ........ q**2 for dis
    int   nglevt; // ........ number of collisions acc to  Glauber
    float zppevt; // ........ average Z-parton-proj
    float zptevt; // ........ average Z-parton-targ
    int   minfra; // 
    int   maxfra; // 
  } cevt_; //epos.inc
}

extern "C"
{
  extern struct
  {
    int   ng1evt; // ........ number of collisions acc to  Glauber
    int   ng2evt; // ........ number of Glauber participants with at least two IAs
    float rglevt; // ........
    float sglevt; // ........
    float eglevt; // ........
    float fglevt; // ........
    int   ikoevt; // ........ number of elementary parton-parton scatterings
    float typevt; // ........ type of event (1=Non Diff, 2=Double Diff, 3=Single Diff)
  } c2evt_; //epos.inc
}

namespace gen
{
  extern "C" {
    float rangen_();
  }

  extern "C" {
    double drangen_(int*);
  }


  class ReggeGribovPartonMCHadronizer : public BaseHadronizer
  {

  public:
    ReggeGribovPartonMCHadronizer(const edm::ParameterSet &);
    virtual ~ReggeGribovPartonMCHadronizer(); 

    bool generatePartonsAndHadronize();
    bool hadronize();
    bool decay();
    bool residualDecay();
    bool readSettings( int ) { return true; }
    bool initializeForExternalPartons() { return true; }
    bool initializeForInternalPartons();
    bool declareStableParticles( const std::vector<int> );
    bool declareSpecialSettings( const std::vector<std::string> ) { return true; }
    
    void finalizeEvent();
    void statistics();
    const char* classname() const;

  private:
    edm::ParameterSet pset_;
    double  fBeamMomentum;
    double  fTargetMomentum;
    int     fBeamID;
    int     fTargetID;
    int     fHEModel;
    edm::FileInPath fParamFileName;
    int     fNEvent;

    int     fNParticles;
    double  fImpactParameter;
    int     fPartID[99990];
    double  fPartPx[99990];
    double  fPartPy[99990];
    double  fPartPz[99990];
    double  fPartEnergy[99990];
    double  fPartMass[99990];
  };

} /*end namespace*/

#endif //ifndef REGGEGRIBOVPARTONMCHADRONIZER_H
