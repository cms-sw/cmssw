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
  //iout,ievent,noutpart,impactpar,outpart,outpx,outpy,outpz,oute,outm,outstat
  void crmc_f_(int&, int&, int&, double&, int&, double&, double&, double&,
                double&, double&, int&);

  //iEvent,iSeed,pproj,ptarg,ipart,itarg,imodel,itab,iout,output,param
  void crmc_set_f_(int&, int&, double&, double&, int&, int&, int&, int&,
                   int&, const char*, const char*);

  //iEvent,iSeed,pproj,ptarg,ipart,itarg,imodel,itab,iout,output,param
  void crmc_init_f_();
}

#define nmxhep 99990

extern "C"
{
  extern struct
  {
    int nevhep;
    int nhep;

    int isthep[nmxhep];
    int idhep[nmxhep];
    int jmohep[nmxhep][2];
    int jdahep[nmxhep][2];
    double phep[nmxhep][5]; 
    double vhep[nmxhep][5];
  } hepcom_; //epos.inc
  
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
  } hadr5_; //epos.inc

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
    int   kohevt; // ........ number of hard collisions
  } cevt_; //epos.inc

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

  extern struct
  {
    float bmaxim;
    float bminim;
    float phimax;
    float phimin;
  } nucl2_; //epos.inc
}


extern "C"
{
  extern struct
  {
    char fnch[500];
    char fnhi[500];
    char fndt[500];
    char fnii[500];
    char fnid[500];
    char fnie[500];
    char fnrj[500];
    char fnmt[500];
    char fngrv[500];
    char fncp[500];
    char fnnx[500];
    char fncs[500];
    char fndr[500];
    char fnhpf[500];
  } fname_; //epos.inc

  extern struct
  {
    int nfnch;
    int nfnhi;
    int nfndt;
    int nfnii;
    int nfnid;
    int nfnie;
    int nfnrj;
    int nfnmt;
    int nfngrv;
    int nfncp;
    int nfnnx;
    int nfncs;
    int nfndr;
    int nfnhpf;
  } nfname_; //epos.inc

  extern struct
  {
    char fndat[500];
    char fnncs[500];
    int ifdat;
    int ifncs;
  } qgsfname_; //epos-bas.f

  extern struct
  {
    int nfndat;
    int nfnncs;
  } qgsnfname_; //epos-bas.f

  extern struct
  {
    char fniidat[500];
    char fniincs[500];
    int ifiidat;
    int ifiincs;
  } qgsiifname_; //epos-bas.f

  extern struct
  {
    int nfniidat;
    int nfniincs;
  } qgsiinfname_; //epos-bas.f
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
    bool initializeTablePaths();

    void finalizeEvent();
    void statistics();
    const char* classname() const;

  private:
    edm::ParameterSet pset_;
    double  m_BeamMomentum;
    double  m_TargetMomentum;
    int     m_BeamID;
    int     m_TargetID;
    int     m_HEModel;
    double  m_bMin;
    double  m_bMax;
    edm::FileInPath m_ParamFileName;
    int     m_NEvent;

    int     m_NParticles;
    double  m_ImpactParameter;
    int     m_PartID[nmxhep];
    double  m_PartPx[nmxhep];
    double  m_PartPy[nmxhep];
    double  m_PartPz[nmxhep];
    double  m_PartEnergy[nmxhep];
    double  m_PartMass[nmxhep];
    int     m_PartStatus[nmxhep];
  };

} /*end namespace*/

#endif //ifndef REGGEGRIBOVPARTONMCHADRONIZER_H
