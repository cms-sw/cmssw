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
  fBeamMomentum(pset.getParameter<double>("beammomentum")),
  fTargetMomentum(pset.getParameter<double>("targetmomentum")),
  fBeamID(pset.getParameter<int>("beamid")),
  fTargetID(pset.getParameter<int>("targetid")),
  fHEModel(pset.getParameter<int>("model")),
  fParamFileName(pset.getParameter<string>("paramFileName")),
  fNEvent(0),
  fImpactParameter(0)
{
  // Default constructor

  setbuf(stdout,NULL); /* set output to unbuffered */

  Service<RandomNumberGenerator> rng;
  if ( ! rng.isAvailable())
    {
      throw cms::Exception("Configuration")
        << "XXXXXXX requires the RandomNumberGeneratorService\n"
        "which is not present in the configuration file.  You must add the service\n"
        "in the configuration file or remove the modules that require it.";
    }

  gFlatDistribution_.reset(new CLHEP::RandFlat(rng->getEngine(),  0., 1.));

  int nevet = 0;
  int dummySeed = 123;
  crmc_init_f_(nevet,dummySeed,fBeamMomentum,fTargetMomentum,fBeamID,
               fTargetID,fHEModel,nevet,fParamFileName.fullPath().c_str());

  //produces<HepMCProduct>();
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
  crmc_f_(iout,ievent,fNParticles,fImpactParameter,
         fPartID[0],fPartPx[0],fPartPy[0],fPartPz[0],
         fPartEnergy[0],fPartMass[0]);

  HepMC::GenEvent* evt = new HepMC::GenEvent();

  evt->set_event_number(fNEvent++);
  evt->set_signal_process_id(c2evt_.typevt); //an integer ID uniquely specifying the signal process (i.e. MSUB in Pythia)

  //create event structure;
  HepMC::GenVertex* theVertex = new HepMC::GenVertex();
  evt->add_vertex(theVertex);

  //number of beam particles
  int nBeam = fTargetID + fBeamID;
  if (fTargetID == 207) // fix for lead wrong ID
    ++nBeam;
  if (fBeamID == 207)
    ++nBeam;

  for(int i = 0; i < fNParticles; i++)
    {
     if (fPartEnergy[i]*fPartEnergy[i] + 1e-9 < fPartPy[i]*fPartPy[i] + fPartPx[i]*fPartPx[i] + fPartPz[i]*fPartPz[i])
       edm::LogInfo("ReggeGribovPartonMCInterface") << "momentum off  Id:" << fPartID[i] << "(" << i << ") " << sqrt(fabs(fPartEnergy[i]*fPartEnergy[i] - (fPartPy[i]*fPartPy[i] + fPartPx[i]*fPartPx[i] + fPartPz[i]*fPartPz[i]))) << endl;
       //status 1 means final particle
      theVertex->add_particle_out( new HepMC::GenParticle( HepMC::FourVector(fPartPx[i], fPartPy[i], fPartPz[i], fPartEnergy[i]), fPartID[i], (i < nBeam)?4:1)); //beam particles status = 4, final status =1;
    }

  if (nBeam > 2)
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
