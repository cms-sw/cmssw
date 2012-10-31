#include <iostream>
#include <cmath>
#include <string>

#include "GeneratorInterface/ReggeGribovPartonMCInterface/interface/ReggeGribovPartonMCHadronizer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

#include <HepMC/GenCrossSection.h>
#include <HepMC/GenEvent.h>
#include <HepMC/HeavyIon.h>
#include <HepMC/PdfInfo.h>
#include <HepMC/Units.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
using namespace edm;
using namespace std;
using namespace gen;

extern "C"
{
  float gen::rangen_()
  {
    float a = float(gFlatDistribution_->fire());
    return a;
  }
}

extern "C"
{
  double gen::drangen_(int *idummy)
  {
    double a = gFlatDistribution_->fire();
    return a;
  }
}

ReggeGribovPartonMCHadronizer::ReggeGribovPartonMCHadronizer(const ParameterSet &pset) :
  BaseHadronizer(pset),
  pset_(pset),
  m_BeamMomentum(pset.getParameter<double>("beammomentum")),
  m_TargetMomentum(pset.getParameter<double>("targetmomentum")),
  m_BeamID(pset.getParameter<int>("beamid")),
  m_TargetID(pset.getParameter<int>("targetid")),
  m_HEModel(pset.getParameter<int>("model")),
  m_bMin(pset.getParameter<double>("bmin")),
  m_bMax(pset.getParameter<double>("bmax")),
  m_ParamFileName(pset.getUntrackedParameter<string>("paramFileName")),
  m_NEvent(0),
  m_ImpactParameter(0.)
{
  // Default constructor

  Service<RandomNumberGenerator> rng;
  if ( ! rng.isAvailable())
    {
      throw cms::Exception("Configuration")
        << "ReggeGribovPartonMCHadronizer requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
    }

  gFlatDistribution_.reset(new CLHEP::RandFlat(rng->getEngine(), 0., 1.));

  int  nevet       = 1; //needed for CS
  int  noTables    = 0; //don't calculate tables
  int  LHEoutput   = 0; //no lhe
  int  dummySeed   = 123;
  char dummyName[] = "dummy";
  crmc_init_f_(nevet,dummySeed,m_BeamMomentum,m_TargetMomentum,m_BeamID,
               m_TargetID,m_HEModel,noTables,LHEoutput,dummyName,
               m_ParamFileName.fullPath().c_str());

  //additionally initialise tables
  //epos
  FileInPath path_fnii =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/epos.initl");
  FileInPath path_fnie =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/epos.iniev");
  FileInPath path_fnrj =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/epos.inirj");
  FileInPath path_fncs =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/epos.inics");
  cout << "!!!! " << path_fnii.fullPath().length() << endl;
  strcpy(fname_.fnii, path_fnii.fullPath().c_str());
  strcpy(fname_.fnie, path_fnie.fullPath().c_str());
  strcpy(fname_.fnrj, path_fnrj.fullPath().c_str());
  strcpy(fname_.fncs, path_fncs.fullPath().c_str());

  //qgsjet
  FileInPath path_fndat =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/qgsjet.dat");
  FileInPath path_fnncs =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/qgsjet.ncs");
  strcpy(qgsfname_.fndat, path_fndat.fullPath().c_str());
  strcpy(qgsfname_.fnncs, path_fnncs.fullPath().c_str());

  //qgsjetII
  FileInPath path_fnIIdat =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/qgsjet-II-04.lzma");
  FileInPath path_fnIIncs =
    FileInPath("GeneratorInterface/ReggeGribovPartonMCInterface/data/sectnu-II-04");
  strcpy(qgsiifname_.fnIIdat, path_fnIIdat.fullPath().c_str());
  strcpy(qgsiifname_.fnIIncs, path_fnIIncs.fullPath().c_str());

  //change impact paramter
  nucl2_.bminim = float(m_bMin);
  nucl2_.bmaxim = float(m_bMax);
}


//_____________________________________________________________________
ReggeGribovPartonMCHadronizer::~ReggeGribovPartonMCHadronizer()
{
  // destructor
  gFlatDistribution_.reset();
}

//_____________________________________________________________________
bool ReggeGribovPartonMCHadronizer::generatePartonsAndHadronize()
{
  int iout=0,ievent=0;
  crmc_f_(iout,ievent,m_NParticles,m_ImpactParameter,
         m_PartID[0],m_PartPx[0],m_PartPy[0],m_PartPz[0],
          m_PartEnergy[0],m_PartMass[0],m_PartStatus[0]);
  LogDebug("ReggeGribovPartonMCInterface") << "event generated" << endl;

  HepMC::GenEvent* evt = new HepMC::GenEvent();

  evt->set_event_number(m_NEvent++);
  evt->set_signal_process_id(c2evt_.typevt); //an integer ID uniquely specifying the signal process (i.e. MSUB in Pythia)

  //create event structure;
  HepMC::GenVertex* theVertex = new HepMC::GenVertex();
  evt->add_vertex(theVertex);

  //number of beam particles
  for(int i = 0; i < m_NParticles; i++)
    {
      if (m_PartEnergy[i]*m_PartEnergy[i] + 1e-9 < m_PartPy[i]*m_PartPy[i] + m_PartPx[i]*m_PartPx[i] + m_PartPz[i]*m_PartPz[i])
        LogWarning("ReggeGribovPartonMCInterface")
          << "momentum off  Id:" << m_PartID[i] 
          << "(" << i << ") " 
          << sqrt(fabs(m_PartEnergy[i]*m_PartEnergy[i] - (m_PartPy[i]*m_PartPy[i] + m_PartPx[i]*m_PartPx[i] + m_PartPz[i]*m_PartPz[i])))
          << endl;
      theVertex->add_particle_out(new HepMC::GenParticle(HepMC::FourVector(m_PartPx[i],
                                                                           m_PartPy[i],
                                                                           m_PartPz[i],
                                                                           m_PartEnergy[i]),
                                                         m_PartID[i],
                                                         m_PartStatus[i]));
    }

  if (m_TargetID + m_BeamID > 2)
    {
      HepMC::HeavyIon ion(-1, //cevt_.koievt, // FIXME // Number of hard scatterings
                          cevt_.npjevt,                // Number of projectile participants
                          cevt_.ntgevt,                // Number of target participants
                          cevt_.kolevt,                // Number of NN (nucleon-nucleon) collisions
                          cevt_.npnevt + cevt_.ntnevt, // Number of spectator neutrons
                          cevt_.nppevt + cevt_.ntpevt, // Number of spectator protons
                          -1, //c2evt_.ng1evt, //FIXME // Number of N-Nwounded collisions
                          -1, //c2evt_.ng2evt, //FIXME // Number of Nwounded-N collisons
                          -1, //c2evt_.ikoevt, //FIXME // Number of Nwounded-Nwounded collisions
                          cevt_.bimevt,                // Impact Parameter(fm) of collision
                          cevt_.phievt,                // Azimuthal angle of event plane
                          c2evt_.fglevt,               // eccentricity of participating nucleons
                          hadr5_.sigineaa);            // nucleon-nucleon inelastic
      evt->set_heavy_ion(ion);
    }
  LogDebug("ReggeGribovPartonMCInterface") << "HepEvt and vertex constructed" << endl;
  //evt->print();
  event().reset(evt);
  return true;
}

//_____________________________________________________________________
bool ReggeGribovPartonMCHadronizer::hadronize()
{
   return false;
}

bool ReggeGribovPartonMCHadronizer::decay()
{
   return true;
}

bool ReggeGribovPartonMCHadronizer::residualDecay()
{
   return true;
}

void ReggeGribovPartonMCHadronizer::finalizeEvent(){
    return;
}

void ReggeGribovPartonMCHadronizer::statistics(){
    return;
}

const char* ReggeGribovPartonMCHadronizer::classname() const
{
   return "gen::ReggeGribovPartonMCHadronizer";
}

bool ReggeGribovPartonMCHadronizer::declareStableParticles ( const std::vector<int> )
{
 return true;
 }

bool ReggeGribovPartonMCHadronizer::initializeForInternalPartons()
{
 return true;
 }
