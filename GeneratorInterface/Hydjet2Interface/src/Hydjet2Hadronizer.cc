/*
  Hydjet2
  Interface to the HYDJET++ generator, produces HepMC events

  Author: Andrey Belyaev (Andrey.Belyaev@cern.ch)

  Hydjet2Hadronizer is the modified InitialStateHydjet

  HYDJET++
  version 2.2:
  InitialStateHydjet is the modified InitialStateBjorken
  The high-pt part related with PYTHIA-PYQUEN is included
  InitialStateBjorken (FASTMC) was used.

         
  InitialStateBjorken           
  version 2.0: 
  Ludmila Malinina  malinina@lav01.sinp.msu.ru,   SINP MSU/Moscow and JINR/Dubna
  Ionut Arsene  i.c.arsene@fys.uio.no,            Oslo University
  June 2007
        
  version 1.0:                                                            
  Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
  amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
  November. 2, 2005                     
*/

//expanding localy equilibated fireball with volume hadron radiation
//thermal part: Blast wave model, Bjorken-like parametrization
//high-pt: PYTHIA + jet quenching model PYQUEN

#include <TLorentzVector.h>
#include <TVector3.h>
#include <TMath.h>

#include "GeneratorInterface/Core/interface/FortranInstance.h"
#include "GeneratorInterface/Hydjet2Interface/interface/Hydjet2Hadronizer.h"
#include "GeneratorInterface/Hydjet2Interface/interface/RandArrayFunction.h"
#include "GeneratorInterface/Hydjet2Interface/interface/HadronDecayer.h"
#include "GeneratorInterface/Hydjet2Interface/interface/GrandCanonical.h"
#include "GeneratorInterface/Hydjet2Interface/interface/StrangePotential.h"
#include "GeneratorInterface/Hydjet2Interface/interface/EquationSolver.h"
#include "GeneratorInterface/Hydjet2Interface/interface/Particle.h"
#include "GeneratorInterface/Hydjet2Interface/interface/ParticlePDG.h"
#include "GeneratorInterface/Hydjet2Interface/interface/UKUtility.h"

#include <iostream> 
#include <fstream>
#include <cmath>
#include "boost/lexical_cast.hpp"

#include "FWCore/Concurrency/interface/SharedResourceNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "HepMC/PythiaWrapper6_4.h"
#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/SimpleVector.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#include "GeneratorInterface/Hydjet2Interface/interface/HYJET_COMMONS.h"

extern "C" void  hyevnt_();
extern "C" void  myini_();

extern HYIPARCommon HYIPAR;
extern HYFPARCommon HYFPAR;
extern HYJPARCommon HYJPAR;
extern HYPARTCommon HYPART;

using namespace edm;
using namespace std;
using namespace gen;

TString RunInputHYDJETstr;

// definition of the static member fLastIndex
int Particle::fLastIndex;
bool ev=0;
namespace {
  int convertStatus(int st){
    if(st<= 0) return 0;
    if(st<=10) return 1;
    if(st<=20) return 2;
    if(st<=30) return 3;
    else return st;
  }
}

const std::vector<std::string> Hydjet2Hadronizer::theSharedResources = { edm::SharedResourceNames::kPythia6, gen::FortranInstance::kFortranInstance };

//____________________________________________________________________________________________
Hydjet2Hadronizer::Hydjet2Hadronizer(const edm::ParameterSet& pset):
  BaseHadronizer(pset),
  fSqrtS(pset.getParameter<double>("fSqrtS")),	// C.m.s. energy per nucleon pair
  fAw(pset.getParameter<double>("fAw")),		// Atomic weigth of nuclei, fAw
  fIfb(pset.getParameter<int>("fIfb")),		// Flag of type of centrality generation, fBfix (=0 is fixed by fBfix, >0 distributed [fBfmin, fBmax])
  fBmin(pset.getParameter<double>("fBmin")),	// Minimum impact parameter in units of nuclear radius, fBmin
  fBmax(pset.getParameter<double>("fBmax")),	// Maximum impact parameter in units of nuclear radius, fBmax
  fBfix(pset.getParameter<double>("fBfix")), 	// Fixed impact parameter in units of nuclear radius, fBfix
  fT(pset.getParameter<double>("fT")),		// Temperature at chemical freeze-out, fT [GeV]
  fMuB(pset.getParameter<double>("fMuB")), 	// Chemical baryon potential per unit charge, fMuB [GeV]
  fMuS(pset.getParameter<double>("fMuS")),	// Chemical strangeness potential per unit charge, fMuS [GeV]  
  fMuC(pset.getParameter<double>("fMuC")),	// Chemical charm potential per unit charge, fMuC [GeV] (used if charm production is turned on)  
  fMuI3(pset.getParameter<double>("fMuI3")),	// Chemical isospin potential per unit charge, fMuI3 [GeV]      
  fThFO(pset.getParameter<double>("fThFO")),	// Temperature at thermal freeze-out, fThFO [GeV]
  fMu_th_pip(pset.getParameter<double>("fMu_th_pip")),// Chemical potential of pi+ at thermal freeze-out, fMu_th_pip [GeV] 
  fTau(pset.getParameter<double>("fTau")), 	// Proper time proper at thermal freeze-out for central collisions, fTau [fm/c]
  fSigmaTau(pset.getParameter<double>("fSigmaTau")), 	// Duration of emission at thermal freeze-out for central collisions, fSigmaTau [fm/c]
  fR(pset.getParameter<double>("fR")),		// Maximal transverse radius at thermal freeze-out for central collisions, fR [fm]
  fYlmax(pset.getParameter<double>("fYlmax")),	// Maximal longitudinal flow rapidity at thermal freeze-out, fYlmax
  fUmax(pset.getParameter<double>("fUmax")),	// Maximal transverse flow rapidity at thermal freeze-out for central collisions, fUmax
  fDelta(pset.getParameter<double>("fDelta")),	// Momentum azimuthal anizotropy parameter at thermal freeze-out, fDelta
  fEpsilon(pset.getParameter<double>("fEpsilon")),	// Spatial azimuthal anisotropy parameter at thermal freeze-out, fEpsilon
  fIfDeltaEpsilon(pset.getParameter<double>("fIfDeltaEpsilon")),	// Flag to specify fDelta and fEpsilon values, fIfDeltaEpsilon (=0 user's ones, >=1 calculated) 
  fDecay(pset.getParameter<int>("fDecay")),	// Flag to switch on/off hadron decays, fDecay (=0 decays off, >=1 decays on)
  fWeakDecay(pset.getParameter<double>("fWeakDecay")),// Low decay width threshold fWeakDecay[GeV]: width<fWeakDecay decay off, width>=fDecayWidth decay on; can be used to switch off weak decays
  fEtaType(pset.getParameter<double>("fEtaType")),	// Flag to choose longitudinal flow rapidity distribution, fEtaType (=0 uniform, >0 Gaussian with the dispersion Ylmax)
  fTMuType(pset.getParameter<double>("fTMuType")),	// Flag to use calculated T_ch, mu_B and mu_S as a function of fSqrtS, fTMuType (=0 user's ones, >0 calculated) 
  fCorrS(pset.getParameter<double>("fCorrS")),	// Strangeness supression factor gamma_s with fCorrS value (0<fCorrS <=1, if fCorrS <= 0 then it is calculated)  
  fCharmProd(pset.getParameter<int>("fCharmProd")),	// Flag to include thermal charm production, fCharmProd (=0 no charm production, >=1 charm production) 
  fCorrC(pset.getParameter<double>("fCorrC")),	// Charmness enhancement factor gamma_c with fCorrC value (fCorrC >0, if fCorrC<0 then it is calculated)
  fNhsel(pset.getParameter<int>("fNhsel")),	//Flag to include jet (J)/jet quenching (JQ) and hydro (H) state production, fNhsel (0 H on & J off, 1 H/J on & JQ off, 2 H/J/HQ on, 3 J on & H/JQ off, 4 H off & J/JQ on)
  fPyhist(pset.getParameter<int>("fPyhist")),	// Flag to suppress the output of particle history from PYTHIA, fPyhist (=1 only final state particles; =0 full particle history from PYTHIA)
  fIshad(pset.getParameter<int>("fIshad")),	// Flag to switch on/off nuclear shadowing, fIshad (0 shadowing off, 1 shadowing on)
  fPtmin(pset.getParameter<double>("fPtmin")),	// Minimal pt of parton-parton scattering in PYTHIA event, fPtmin [GeV/c] 
  fT0(pset.getParameter<double>("fT0")),		// Initial QGP temperature for central Pb+Pb collisions in mid-rapidity, fT0 [GeV] 
  fTau0(pset.getParameter<double>("fTau0")),	// Proper QGP formation time in fm/c, fTau0 (0.01<fTau0<10)
  fNf(pset.getParameter<int>("fNf")),		// Number of active quark flavours in QGP, fNf (0, 1, 2 or 3)
  fIenglu(pset.getParameter<int>("fIenglu")),	// Flag to fix type of partonic energy loss, fIenglu (0 radiative and collisional loss, 1 radiative loss only, 2 collisional loss only)
  fIanglu(pset.getParameter<int>("fIanglu")),	// Flag to fix type of angular distribution of in-medium emitted gluons, fIanglu (0 small-angular, 1 wide-angular, 2 collinear).

  embedding_(pset.getParameter<bool>("embeddingMode")),
  rotate_(pset.getParameter<bool>("rotateEventPlane")),
  evt(0),
  nsub_(0),
  nhard_(0), 
  nsoft_(0),
  phi0_(0.),
  sinphi0_(0.),
  cosphi0_(1.),
  pythia6Service_(new Pythia6Service(pset))
 
{  
  // constructor 
  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_;
  //Max number of events printed on verbosity level
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_;
  if(embedding_) src_ = pset.getParameter<edm::InputTag>("backgroundLabel");
}

//__________________________________________________________________________________________
Hydjet2Hadronizer::~Hydjet2Hadronizer(){
  // destructor 
  call_pystat(1);
  delete pythia6Service_;
}


//_____________________________________________________________________
void Hydjet2Hadronizer::doSetRandomEngine(CLHEP::HepRandomEngine* v)
{
  pythia6Service_->setRandomEngine(v);
  hjRandomEngine = v;
}

//______________________________________________________________________________________________________
bool Hydjet2Hadronizer::readSettings( int )  {     

  Pythia6Service::InstanceWrapper guard(pythia6Service_);
  pythia6Service_->setGeneralParams();

  SERVICE.iseed_fromC=hjRandomEngine->CLHEP::HepRandomEngine::getSeed(); 
  LogInfo("Hydjet2Hadronizer|GenSeed") << "Seed for random number generation: "<<hjRandomEngine->CLHEP::HepRandomEngine::getSeed(); 

  fNPartTypes = 0;         //counter of hadron species
   
  return kTRUE;
}

//______________________________________________________________________________________________________
bool Hydjet2Hadronizer::initializeForInternalPartons(){

  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // the input impact parameter (bxx_) is in [fm]; transform in [fm/RA] for hydjet usage
  const float ra = nuclear_radius();
  LogInfo("Hydjet2Hadronizer|RAScaling")<<"Nuclear radius(RA) =  "<<ra;
  fBmin     /= ra;
  fBmax     /= ra;
  fBfix   /= ra;

  //check and redefine input parameters
  if(fTMuType>0 &&  fSqrtS > 2.24) {

    if(fSqrtS < 2.24){
      LogError("Hydjet2Hadronizer|sqrtS") << "SqrtS<2.24 not allowed with fTMuType>0";
      return 0;
    }
    
    //sqrt(s) = 2.24 ==> T_kin = 0.8 GeV
    //see J. Cleymans, H. Oeschler, K. Redlich,S. Wheaton, Phys Rev. C73 034905 (2006)
    fMuB = 1.308/(1. + fSqrtS*0.273);
    fT = 0.166 - 0.139*fMuB*fMuB - 0.053*fMuB*fMuB*fMuB*fMuB;
    fMuI3 = 0.;
    fMuS = 0.;

    //create strange potential object and set strangeness density 0
    NAStrangePotential* psp = new NAStrangePotential(0., fDatabase);
    psp->SetBaryonPotential(fMuB);
    psp->SetTemperature(fT);

    //compute strangeness potential
    if(fMuB > 0.01) fMuS = psp->CalculateStrangePotential();
    LogInfo("Hydjet2Hadronizer|Strange") << "fMuS = " << fMuS;  

    //if user choose fYlmax larger then allowed by kinematics at the specified beam energy sqrt(s)     
    if(fYlmax > TMath::Log(fSqrtS/0.94)){
      LogError("Hydjet2Hadronizer|Ylmax") << "fYlmax more then TMath::Log(fSqrtS vs 0.94)!!! ";
      return 0;
    }
      
    if(fCorrS <= 0.) {
      //see F. Becattini, J. Mannien, M. Gazdzicki, Phys Rev. C73 044905 (2006)
      fCorrS = 1. - 0.386* TMath::Exp(-1.23*fT/fMuB);
      LogInfo("Hydjet2Hadronizer|Strange") << "The phenomenological f-la F. Becattini et al. PRC73 044905 (2006) for CorrS was used."<<endl
      <<"Strangeness suppression parameter = "<<fCorrS;
    }
    LogInfo("Hydjet2Hadronizer|Strange") << "The phenomenological f-la J. Cleymans et al. PRC73 034905 (2006) for Tch mu_B was used." << endl
    <<"The simulation will be done with the calculated parameters:" << endl
    <<"Baryon chemical potential = "<<fMuB<< " [GeV]" << endl
    <<"Strangeness chemical potential = "<<fMuS<< " [GeV]" << endl
    <<"Isospin chemical potential = "<<fMuI3<< " [GeV]" << endl
    <<"Strangeness suppression parameter = "<<fCorrS << endl
    <<"Eta_max = "<<fYlmax;
  }
  
  LogInfo("Hydjet2Hadronizer|Param") << "Used eta_max = "<<fYlmax<<  endl
  <<"maximal allowed eta_max TMath::Log(fSqrtS/0.94)=  "<<TMath::Log(fSqrtS/0.94);

  //initialisation of high-pt part                                                                                                     
  HYJPAR.nhsel = fNhsel;
  HYJPAR.ptmin = fPtmin;
  HYJPAR.ishad = fIshad;
  HYJPAR.iPyhist = fPyhist;
  HYIPAR.bminh = fBmin;
  HYIPAR.bmaxh = fBmax;
  HYIPAR.AW = fAw;
  
  HYPYIN.ifb = fIfb;
  HYPYIN.bfix = fBfix;
  HYPYIN.ene = fSqrtS;

  PYQPAR.T0 = fT0;
  PYQPAR.tau0 = fTau0;
  PYQPAR.nf = fNf;
  PYQPAR.ienglu = fIenglu;
  PYQPAR.ianglu = fIanglu;
  myini_();
 
  // calculation of  multiplicities of different particle species
  // according to the grand canonical approach
  GrandCanonical gc(15, fT, fMuB, fMuS, fMuI3, fMuC);
  GrandCanonical gc_ch(15, fT, fMuB, fMuS, fMuI3, fMuC);
  GrandCanonical gc_pi_th(15, fThFO, 0., 0., fMu_th_pip, fMuC);
  GrandCanonical gc_th_0(15, fThFO, 0., 0., 0., 0.);
  
  // std::ofstream outMult("densities.txt");
  //    outMult<<"encoding    particle density      chemical potential "<<std::endl;

  double Nocth=0; //open charm
  double NJPsith=0; //JPsi
    
  //effective volume for central     
  double dYl= 2 * fYlmax; //uniform distr. [-Ylmax; Ylmax]  
  if (fEtaType >0) dYl = TMath::Sqrt(2 * TMath::Pi()) * fYlmax ;  //Gaussian distr.                                                                            
  fVolEff = 2 * TMath::Pi() * fTau * dYl * (fR * fR)/TMath::Power((fUmax),2) * 
    ((fUmax)*TMath::SinH((fUmax))-TMath::CosH((fUmax))+ 1);
  LogInfo("Hydjet2Hadronizer|Param") << "central Effective volume = " << fVolEff << " [fm^3]";
  
  double particleDensity_pi_ch=0;
  double particleDensity_pi_th=0;
  //  double particleDensity_th_0=0;

  if(fThFO != fT && fThFO > 0){
    GrandCanonical gc_ch(15, fT, fMuB, fMuS, fMuI3, fMuC);
    GrandCanonical gc_pi_th(15, fThFO, 0., 0., fMu_th_pip, fMuC);
    GrandCanonical gc_th_0(15, fThFO, 0., 0., 0., 0.);
    particleDensity_pi_ch = gc_ch.ParticleNumberDensity(fDatabase->GetPDGParticle(211));
    particleDensity_pi_th = gc_pi_th.ParticleNumberDensity(fDatabase->GetPDGParticle(211));
  }

  for(int particleIndex = 0; particleIndex < fDatabase->GetNParticles(); particleIndex++) {
    ParticlePDG *currParticle = fDatabase->GetPDGParticleByIndex(particleIndex);
    int encoding = currParticle->GetPDG();

    //strangeness supression
    double gammaS = 1;
    int S = int(currParticle->GetStrangeness());
    if(encoding == 333)S = 2;
    if(fCorrS < 1. && S != 0)gammaS = TMath::Power(fCorrS,-TMath::Abs(S));


    //average densities      
    double particleDensity = gc.ParticleNumberDensity(currParticle)/gammaS;
    
    //compute chemical potential for single f.o. mu==mu_ch
    double mu = fMuB  * int(currParticle->GetBaryonNumber()) + 
      fMuS  * int(currParticle->GetStrangeness()) +
      fMuI3 * int(currParticle->GetElectricCharge()) +
      fMuC * int(currParticle->GetCharmness());

    //thermal f.o.
    if(fThFO != fT && fThFO > 0){
      double particleDensity_ch = gc_ch.ParticleNumberDensity(currParticle);
      double particleDensity_th_0 = gc_th_0.ParticleNumberDensity(currParticle);
      double numb_dens_bolt = particleDensity_pi_th*particleDensity_ch/particleDensity_pi_ch;               
      mu = fThFO*TMath::Log(numb_dens_bolt/particleDensity_th_0);
      if(abs(encoding)==211 || encoding==111)mu= fMu_th_pip; 
      particleDensity = numb_dens_bolt;         
    }
    
    // set particle densities to zero for some particle codes
    // pythia quark codes
    if(abs(encoding)<=9) {
      particleDensity=0;        
    }
    // leptons
    if(abs(encoding)>10 && abs(encoding)<19) {
      particleDensity=0;   
    }
    // exchange bosons
    if(abs(encoding)>20 && abs(encoding)<30) {
      particleDensity=0;
    }
    // pythia special codes (e.g. strings, clusters ...)
    if(abs(encoding)>80 && abs(encoding)<100) {
      particleDensity=0;
    }
    // pythia di-quark codes
    // Note: in PYTHIA all diquark codes have the tens digits equal to zero
    if(abs(encoding)>1000 && abs(encoding)<6000) {
      int tens = ((abs(encoding)-(abs(encoding)%10))/10)%10;
      if(tens==0) {             // its a diquark;
	particleDensity=0;
      }
    }
    // K0S and K0L
    if(abs(encoding)==130 || abs(encoding)==310) {
      particleDensity=0;
    }
    // charmed particles     

    if(encoding==443)NJPsith=particleDensity*fVolEff/dYl; 	
    
    // We generate thermo-statistically only J/psi(443), D_+(411), D_-(-411), D_0(421), 
    //Dbar_0(-421), D1_+(413), D1_-(-413), D1_0(423), D1bar_0(-423)
    //Dcs(431) Lambdac(4122)
    if(currParticle->GetCharmQNumber()!=0 || currParticle->GetCharmAQNumber()!=0) {
      //ml if(abs(encoding)!=443 &&  
      //ml	 abs(encoding)!=411 && abs(encoding)!=421 && 
      //ml	 abs(encoding)!=413 && abs(encoding)!=423 && abs(encoding)!=4122 && abs(encoding)!=431) {
      //ml	particleDensity=0; } 

      if(abs(encoding)==441 ||  
	 abs(encoding)==10441 || abs(encoding)==10443 || 
	 abs(encoding)==20443 || abs(encoding)==445 || abs(encoding)==4232 || abs(encoding)==4322 ||
	 abs(encoding)==4132 || abs(encoding)==4312 || abs(encoding)==4324 || abs(encoding)==4314 ||
	 abs(encoding)==4332 || abs(encoding)==4334 
	 ) {
	particleDensity=0; } 
      else
        {
          if(abs(encoding)!=443){ //only open charm
            Nocth=Nocth+particleDensity*fVolEff/dYl;       
            LogInfo("Hydjet2Hadronizer|Charam") << encoding<<" Nochth "<<Nocth;
            //      particleDensity=particleDensity*fCorrC;
            //      if(abs(encoding)==443)particleDensity=particleDensity*fCorrC;
          }
        }
      
    }
    

    // bottom mesons
    if((abs(encoding)>500 && abs(encoding)<600) ||
       (abs(encoding)>10500 && abs(encoding)<10600) ||
       (abs(encoding)>20500 && abs(encoding)<20600) ||
       (abs(encoding)>100500 && abs(encoding)<100600)) {
      particleDensity=0;
    }
    // bottom baryons
    if(abs(encoding)>5000 && abs(encoding)<6000) {
      particleDensity=0;
    }
    ////////////////////////////////////////////////////////////////////////////////////////


    if(particleDensity > 0.) {
      fPartEnc[fNPartTypes] = encoding;
      fPartMult[2 * fNPartTypes] = particleDensity;
      fPartMu[2 * fNPartTypes] = mu;
      ++fNPartTypes;
      if(fNPartTypes > 1000)
	LogError("Hydjet2Hadronizer") << "fNPartTypes is too large" << fNPartTypes;

      //outMult<<encoding<<" "<<particleDensity*fVolEff/dYl <<" "<<mu<<std::endl;

    }
  }

  //put open charm number and cc number in Params
  fNocth = Nocth;
  fNccth = NJPsith;


  return kTRUE;
}


//__________________________________________________________________________________________
bool Hydjet2Hadronizer::generatePartonsAndHadronize(){


  Pythia6Service::InstanceWrapper guard(pythia6Service_);

  // Initialize the static "last index variable"
  Particle::InitIndexing();
    
  //----- high-pt part------------------------------
  TLorentzVector partJMom, partJPos, zeroVec;

  // generate single event
  if(embedding_){
    fIfb = 0;
    const edm::Event& e = getEDMEvent();
    Handle<HepMCProduct> input;
    e.getByLabel(src_,input);
    const HepMC::GenEvent * inev = input->GetEvent();
    const HepMC::HeavyIon* hi = inev->heavy_ion();
    if(hi){
      fBfix = hi->impact_parameter();
      phi0_ = hi->event_plane_angle();
      sinphi0_ = sin(phi0_);
      cosphi0_ = cos(phi0_);
    }else{
      LogWarning("EventEmbedding")<<"Background event does not have heavy ion record!";
    }
  }else if(rotate_) rotateEvtPlane();

  nsoft_ = 0;
  nhard_ = 0;
  /*
    edm::LogInfo("HYDJET2mode") << "##### HYDJET2 fNhsel = " << fNhsel;
    edm::LogInfo("HYDJET2fpart") << "##### HYDJET2 fpart = " << hyflow.fpart;??
    edm::LogInfo("HYDJET2tf") << "##### HYDJET2 hadron freez-out temp, Tf = " << hyflow.Tf;??
    edm::LogInfo("HYDJET2tf") << "##### HYDJET2 hadron freez-out temp, Tf = " << hyflow.Tf;??
    edm::LogInfo("HYDJET2inTemp") << "##### HYDJET2: QGP init temperature, fT0 ="<<fT0;
    edm::LogInfo("HYDJET2inTau") << "##### HYDJET2: QGP formation time, fTau0 ="<<fTau0;
  */
  // generate a HYDJET event
  int ntry = 0;
  while(nsoft_ == 0 && nhard_ == 0){
    if(ntry > 100){
      LogError("Hydjet2EmptyEvent") << "##### HYDJET2: No Particles generated, Number of tries ="<<ntry;
      // Throw an exception. Use the EventCorruption exception since it maps onto SkipEvent
      // which is what we want to do here.
      std::ostringstream sstr;
      sstr << "Hydjet2HadronizerProducer: No particles generated after " << ntry << " tries.\n";
      edm::Exception except(edm::errors::EventCorruption, sstr.str());
      throw except;
    } else {
      
  
      //generate non-equilibrated part event
      hyevnt_();

      ////////-------------HARD & SOFT particle list ----------begin------- //////////////// --->to separete func.
      if(fNhsel != 0){   
        //get number of particles in jets
        int numbJetPart = HYPART.njp;
    
        for(int i = 0; i <numbJetPart; ++i) {
          int pythiaStatus    = int(HYPART.ppart[i][0]);   // PYTHIA status code
          int pdg             = int(HYPART.ppart[i][1]);   // PYTHIA species code
          double px           = HYPART.ppart[i][2];          // px
          double py           = HYPART.ppart[i][3];          // py
          double pz           = HYPART.ppart[i][4];          // pz
          double e            = HYPART.ppart[i][5];          // E
          double vx           = HYPART.ppart[i][6];          // x
          double vy           = HYPART.ppart[i][7];          // y
          double vz           = HYPART.ppart[i][8];          // z
          double vt           = HYPART.ppart[i][9];          // t
          // particle line number in pythia are 1 based while we use a 0 based numbering
          int mother_index    = int(HYPART.ppart[i][10])-1;  //line number of parent particle
          int daughter_index1 = int(HYPART.ppart[i][11])-1;  //line number of first daughter
          int daughter_index2 = int(HYPART.ppart[i][12])-1;  //line number of last daughter

          // For status codes 3, 13, 14 the first and last daughter indexes have a different meaning
          // used for color flow in PYTHIA. So these indexes will be reset to zero.
          if(TMath::Abs(daughter_index1)>numbJetPart || TMath::Abs(daughter_index2)>numbJetPart ||
             TMath::Abs(daughter_index1)>TMath::Abs(daughter_index2)) {
            daughter_index1 = -1;
            daughter_index2 = -1;
          }
      
          ParticlePDG *partDef = fDatabase->GetPDGParticle(pdg);
            
          int type=1; //from jet
          if(partDef) {
            int motherPdg = int(HYPART.ppart[mother_index][1]);
            if(motherPdg==0) motherPdg = -1;
            partJMom.SetXYZT(px, py, pz, e); 
            partJPos.SetXYZT(vx, vy, vz, vt);
            Particle particle(partDef, partJPos, partJMom, 0, 0, type, motherPdg, zeroVec, zeroVec);
            int index = particle.SetIndex();
            if(index!=i) {
              LogWarning("Hydjet2Hadronizer") << " Allocated HYDJET++ index is not synchronized with the PYTHIA index!" << endl
                   << " Collision history information is destroyed! It happens when a PYTHIA code is not" << endl
                   << " implemented in HYDJET++ particle list particles.data! Check it out!";
            }
            particle.SetPythiaStatusCode(pythiaStatus);
            particle.SetMother(mother_index);
            particle.SetFirstDaughterIndex(daughter_index1);
            particle.SetLastDaughterIndex(daughter_index2);
            if(pythiaStatus!=1) particle.SetDecayed();
            allocator.AddParticle(particle, source);
          }
          else {
            LogWarning("Hydjet2Hadronizer") << " PYTHIA particle of specie " << pdg << " is not in HYDJET++ particle list" << endl
                 <<" Please define it in particles.data, otherwise the history information will be de-synchronized and lost!";
          }
        }
      } //nhsel !=0 not only hydro!        


      //----------HYDRO part--------------------------------------
      
      // get impact parameter    
      double impactParameter = HYFPAR.bgen;

      // Sergey psiforv3
      double psiforv3 = 0.;	//AS-ML Nov2012  epsilon3 //
      double e3 = (0.2/5.5)*TMath::Power(impactParameter,1./3.);
      psiforv3 = TMath::TwoPi() *  (-0.5 + CLHEP::RandFlat::shoot(hjRandomEngine))  / 3.;
      SERVICEEV.psiv3 = -psiforv3;

      if(fNhsel < 3){
        const double  weightMax = 2*TMath::CosH(fUmax);
        const int nBins = 100;
        double probList[nBins];
        RandArrayFunction arrayFunctDistE(nBins);
        RandArrayFunction arrayFunctDistR(nBins);
        TLorentzVector partPos, partMom, n1, p0;
        TVector3 vec3;
        const TLorentzVector zeroVec;
        //set maximal hadron energy
        const double eMax = 5.;
        //-------------------------------------
        // get impact parameter

        double Delta =fDelta;
        double Epsilon =fEpsilon;

        if(fIfDeltaEpsilon>0){
          double Epsilon0 = 0.5*impactParameter; //e0=b/2Ra
          double coeff = (HYIPAR.RA/fR)/12.;//phenomenological coefficient
          Epsilon = Epsilon0 * coeff;
          double C=5.6;
          double A = C*Epsilon*(1-Epsilon*Epsilon);
          if(TMath::Abs(Epsilon)<0.0001 || TMath::Abs(A)<0.0001 )Delta=0.0;          if(TMath::Abs(Epsilon)>0.0001 && TMath::Abs(A)>0.0001)Delta = 0.5*(TMath::Sqrt(1+4*A*(Epsilon+A))-1)/A;

        }
        
        //effective volume for central
        double dYl= 2 * fYlmax; //uniform distr. [-Ylmax; Ylmax]
        if (fEtaType >0) dYl = TMath::Sqrt(2 * TMath::Pi()) * fYlmax ;  //Gaussian distr.

        double VolEffcent = 2 * TMath::Pi() * fTau * dYl * (fR * fR)/TMath::Power((fUmax),2) * ((fUmax)*TMath::SinH((fUmax))-TMath::CosH((fUmax))+ 1);

        //effective volume for non-central Simpson2
        double VolEffnoncent = fTau * dYl * SimpsonIntegrator2(0., 2.*TMath::Pi(), Epsilon, Delta);

        fVolEff = VolEffcent * HYFPAR.npart/HYFPAR.npart0;

        double coeff_RB = TMath::Sqrt(VolEffcent * HYFPAR.npart/HYFPAR.npart0/VolEffnoncent);
        double coeff_R1 = HYFPAR.npart/HYFPAR.npart0;
        coeff_R1 = TMath::Power(coeff_R1, 0.333333);

        double Veff=fVolEff;
        //------------------------------------
        //cycle on particles types

        double Nbcol = HYFPAR.nbcol;
        double NccNN = SERVICE.charm;
        double Ncc = Nbcol * NccNN/dYl;
        double Nocth = fNocth;
        double NJPsith = fNccth;

        double gammaC=1.0;
        if(fCorrC<=0){
          gammaC=CharmEnhancementFactor(Ncc, Nocth, NJPsith, 0.001);
        }else{
          gammaC=fCorrC;
        }

        LogInfo("Hydjet2Hadronizer|Param") <<" gammaC = " <<gammaC;

        for(int i = 0; i < fNPartTypes; ++i) {
          double Mparam = fPartMult[2 * i] * Veff;
          const int encoding = fPartEnc[i];

          //ml  if(abs(encoding)==443)Mparam = Mparam * gammaC * gammaC;  
          //ml  if(abs(encoding)==411 || abs(encoding)==421 ||abs(encoding)==413 || abs(encoding)==423
          //ml   || abs(encoding)==4122 || abs(encoding)==431)

          ParticlePDG *partDef0 = fDatabase->GetPDGParticle(encoding);

          if(partDef0->GetCharmQNumber()!=0 || partDef0->GetCharmAQNumber()!=0)Mparam = Mparam * gammaC;
          if(abs(encoding)==443)Mparam = Mparam * gammaC;  

          LogInfo("Hydjet2Hadronizer|Param") <<encoding<<" "<<Mparam/dYl;

          int multiplicity = CLHEP::RandPoisson::shoot(hjRandomEngine, Mparam);

          LogInfo("Hydjet2Hadronizer|Param") <<"specie: " << encoding << "; average mult: = " << Mparam << "; multiplicity = " << multiplicity;

          if (multiplicity > 0) {
            ParticlePDG *partDef = fDatabase->GetPDGParticle(encoding);
            if(!partDef) {
              LogError("Hydjet2Hadronizer") << "No particle with encoding "<< encoding;
              continue;
            }

            
              if(fCharmProd<=0 && (partDef->GetCharmQNumber()!=0 || partDef->GetCharmAQNumber()!=0)){
              LogInfo("Hydjet2Hadronizer|Param") <<"statistical charmed particle not allowed ! "<<encoding;
              continue;
              }
              if(partDef->GetCharmQNumber()!=0 || partDef->GetCharmAQNumber()!=0)
              LogInfo("Hydjet2Hadronizer|Param") <<" charm pdg generated "<< encoding;
            

            //compute chemical potential for single f.o. mu==mu_ch
            //compute chemical potential for thermal f.o.                
            double mu = fPartMu[2 * i];

            //choose Bose-Einstein or Fermi-Dirac statistics
            const double d    = !(int(2*partDef->GetSpin()) & 1) ? -1 : 1;
            const double mass = partDef->GetMass();                	 
	
            //prepare histogram to sample hadron energy: 
            double h = (eMax - mass) / nBins;
            double x = mass + 0.5 * h;
            int i;        
            for(i = 0; i < nBins; ++i) {
              if(x>=mu && fThFO>0)probList[i] = x * TMath::Sqrt(x * x - mass * mass) / (TMath::Exp((x - mu) / (fThFO)) + d);
              if(x>=mu && fThFO<=0)probList[i] = x * TMath::Sqrt(x * x - mass * mass) / (TMath::Exp((x - mu) / (fT)) + d);
              if(x<mu)probList[i] = 0.; 
              x += h;
            }
            arrayFunctDistE.PrepareTable(probList);

            //prepare histogram to sample hadron transverse radius: 
            h = (fR) / nBins;
            x =  0.5 * h;
            double param = (fUmax) / (fR);
            for (i = 0; i < nBins; ++i) {
              probList[i] = x * TMath::CosH(param*x);
              x += h;
            }
            arrayFunctDistR.PrepareTable(probList);

            //loop over hadrons, assign hadron coordinates and momenta
            double weight = 0., yy = 0., px0 = 0., py0 = 0., pz0 = 0.;
            double e = 0., x0 = 0., y0 = 0., z0 = 0., t0 = 0., etaF = 0.; 
            double r, RB, phiF;

            RB = fR * coeff_RB * coeff_R1 * TMath::Sqrt((1+e3)/(1-e3));

            for(int j = 0; j < multiplicity; ++j) {               
              do {
                fEtaType <=0 ? etaF = fYlmax * (2. * CLHEP::RandFlat::shoot(hjRandomEngine) - 1.) : etaF = (fYlmax) * (CLHEP::RandGauss::shoot(hjRandomEngine));
                n1.SetXYZT(0.,0.,TMath::SinH(etaF),TMath::CosH(etaF));  

                if(TMath::Abs(etaF)>5.)continue;

                //old
                //double RBold = fR * TMath::Sqrt(1-fEpsilon);

                //RB = fR * coeff_RB * coeff_R1;

                //double impactParameter =HYFPAR.bgen;
                //double e0 = 0.5*impactParameter;
                //double RBold1 = fR * TMath::Sqrt(1-e0);                                                                                                   

                double rho = TMath::Sqrt(CLHEP::RandFlat::shoot(hjRandomEngine));
                double phi = TMath::TwoPi() * CLHEP::RandFlat::shoot(hjRandomEngine);
                double Rx =  TMath::Sqrt(1-Epsilon)*RB; 
                double Ry =  TMath::Sqrt(1+Epsilon)*RB;

                x0 = Rx * rho * TMath::Cos(phi);
                y0 = Ry * rho * TMath::Sin(phi);
                r = TMath::Sqrt(x0*x0+y0*y0);
                phiF = TMath::Abs(TMath::ATan(y0/x0));

                if(x0<0&&y0>0)phiF = TMath::Pi()-phiF;
                if(x0<0&&y0<0)phiF = TMath::Pi()+phiF;
                if(x0>0&&y0<0)phiF = 2.*TMath::Pi()-phiF;

                //new Nov2012 AS-ML
                if(r>RB*(1+e3*TMath::Cos(3*(phiF+psiforv3)))/(1+e3))continue;

                //proper time with emission duration                                                               
                double tau = coeff_R1 * fTau +  sqrt(2.) * fSigmaTau * coeff_R1 * (CLHEP::RandGauss::shoot(hjRandomEngine));
                z0 = tau  * TMath::SinH(etaF);
                t0 = tau  * TMath::CosH(etaF);
                double rhou = fUmax * r / RB; 
                double rhou3 = 0.063*TMath::Sqrt((0.5*impactParameter)/0.67);	    double rhou4 = 0.023*((0.5*impactParameter)/0.67);
                double rrcoeff = 1./TMath::Sqrt(1. + Delta*TMath::Cos(2*phiF));
	    //AS-ML Nov.2012 
	    rhou3=0.; 	    //rhou4=0.; 
                                                                                     	    rhou = rhou * (1 + rrcoeff*rhou3*TMath::Cos(3*(phiF+psiforv3)) + rrcoeff*rhou4*TMath::Cos(4*phiF) );	    //ML new suggestion of AS mar2012
                double delta1 = 0.;	    Delta = Delta * (1.0 + delta1 * TMath::Cos(phiF) - delta1 * TMath::Cos(3*phiF));

                double uxf = TMath::SinH(rhou)*TMath::Sqrt(1+Delta)*TMath::Cos(phiF); 
                double uyf = TMath::SinH(rhou)*TMath::Sqrt(1-Delta)*TMath::Sin(phiF);
                double utf = TMath::CosH(etaF) * TMath::CosH(rhou) * 
                  TMath::Sqrt(1+Delta*TMath::Cos(2*phiF)*TMath::TanH(rhou)*TMath::TanH(rhou));
                double uzf = TMath::SinH(etaF) * TMath::CosH(rhou) * 
                  TMath::Sqrt(1+Delta*TMath::Cos(2*phiF)*TMath::TanH(rhou)*TMath::TanH(rhou));

                vec3.SetXYZ(uxf / utf, uyf / utf, uzf / utf);
                n1.Boost(-vec3); 

                yy = weightMax * CLHEP::RandFlat::shoot(hjRandomEngine);        

                double php0 = TMath::TwoPi() * CLHEP::RandFlat::shoot(hjRandomEngine);
                double ctp0 = 2. * CLHEP::RandFlat::shoot(hjRandomEngine) - 1.;
                double stp0 = TMath::Sqrt(1. - ctp0 * ctp0); 
                e = mass + (eMax - mass) * arrayFunctDistE(); 
                double pp0 = TMath::Sqrt(e * e - mass * mass);
                px0 = pp0 * stp0 * TMath::Sin(php0); 
                py0 = pp0 * stp0 * TMath::Cos(php0);
                pz0 = pp0 * ctp0;
                p0.SetXYZT(px0, py0, pz0, e);

                //weight for rdr          
                weight = (n1 * p0) /e;  // weight for rdr gammar: weight = (n1 * p0) / n1[3] / e; 

              } while(yy >= weight); 

              if(abs(z0)>1000 || abs(x0)>1000) LogInfo("Hydjet2Hadronizer|Param") <<" etaF = "<<etaF<<std::endl;

              partMom.SetXYZT(px0, py0, pz0, e);
              partPos.SetXYZT(x0, y0, z0, t0);
              partMom.Boost(vec3);

              int type =0; //hydro
              Particle particle(partDef, partPos, partMom, 0., 0, type, -1, zeroVec, zeroVec);
              particle.SetIndex();
              allocator.AddParticle(particle, source);
            } //nhsel==4 , no hydro part
          }
        }
      }

      ////////-------------HARD & SOFT particle list ----------end------- ///////////////////////////

      Npart = (int)HYFPAR.npart;      
      Bgen = HYFPAR.bgen;
      Njet = (int)HYJPAR.njet;
      Nbcol = (int)HYFPAR.nbcol;

      if(source.empty()) {
        LogError("Hydjet2Hadronizer") << "Source is not initialized!!";
        //return ;
      }
      //Run the decays
      if(RunDecays()) Evolve(source, allocator, GetWeakDecayLimit());

      LPIT_t it;
      LPIT_t e;

      //Fill the decayed arrays
      Ntot = 0; Nhyd=0; Npyt=0;      
      for(it = source.begin(), e = source.end(); it != e; ++it) {
        TVector3 pos(it->Pos().Vect());
        TVector3 mom(it->Mom().Vect());
        float m1 = it->TableMass();
        pdg[Ntot] = it->Encoding();
        Mpdg[Ntot] = it->GetLastMotherPdg();
        Px[Ntot] = mom[0];
        Py[Ntot] = mom[1];
        Pz[Ntot] = mom[2];
        E[Ntot] =  TMath::Sqrt(mom.Mag2() + m1*m1);
        X[Ntot] = pos[0];
        Y[Ntot] = pos[1];
        Z[Ntot] = pos[2];
        T[Ntot] = it->T();
        type[Ntot] = it->GetType();
        pythiaStatus[Ntot] = it->GetPythiaStatusCode();
        Index[Ntot] = it->GetIndex();
        MotherIndex[Ntot] = it->GetMother();
        NDaughters[Ntot] = it->GetNDaughters();
        FirstDaughterIndex[Ntot] = -1; LastDaughterIndex[Ntot] = -1;
        //index of first daughter
        FirstDaughterIndex[Ntot] = it->GetFirstDaughterIndex();
        //index of last daughter
        LastDaughterIndex[Ntot] = it->GetLastDaughterIndex();
        if(type[Ntot]==1) {     // jets
          if(pythiaStatus[Ntot]==1 && NDaughters[Ntot]==0)  // code for final state particle in pythia
            final[Ntot]=1;
          else
            final[Ntot]=0;
        }
        if(type[Ntot]==0) {     // hydro
          if(NDaughters[Ntot]==0)
            final[Ntot]=1;
          else
            final[Ntot]=0;
        }

        if(type[Ntot]==0)Nhyd++;
        if(type[Ntot]==1)Npyt++;

        Ntot++;
        if(Ntot > kMax)
          LogError("Hydjet2Hadronizer") << "Ntot is too large" << Ntot;
      }

      nsoft_    = Nhyd;
      nsub_     = Njet;
      nhard_    = Npyt;

      //100 trys

      ++ntry;
    }
  }

  if(ev==0) { 
    Sigin=HYJPAR.sigin;
    Sigjet=HYJPAR.sigjet;
  }
  ev=1;

  if(fNhsel < 3) nsub_++;

  // event information
  HepMC::GenEvent *evt = new HepMC::GenEvent();
  if(nhard_>0 || nsoft_>0) get_particles(evt); 

  evt->set_signal_process_id(pypars.msti[0]); // type of the process
  evt->set_event_scale(pypars.pari[16]); // Q^2
  add_heavy_ion_rec(evt);


  event().reset(evt);

  allocator.FreeList(source);

  return kTRUE;
}


//________________________________________________________________
bool Hydjet2Hadronizer::declareStableParticles(const std::vector<int>& _pdg )
{
  std::vector<int> pdg = _pdg;
  for ( size_t i=0; i < pdg.size(); i++ ) {
    int pyCode = pycomp_( pdg[i] );
    std::ostringstream pyCard ;
    pyCard << "MDCY(" << pyCode << ",1)=0";
    std::cout << pyCard.str() << std::endl;
    call_pygive( pyCard.str() );
  }
  return true;
}
//________________________________________________________________
bool Hydjet2Hadronizer::hadronize()
{
  return false;
}
bool Hydjet2Hadronizer::decay()
{
  return true;
}
bool Hydjet2Hadronizer::residualDecay()
{
  return true;
}
void Hydjet2Hadronizer::finalizeEvent()
{
}
void Hydjet2Hadronizer::statistics()
{
}
const char* Hydjet2Hadronizer::classname() const
{
  return "gen::Hydjet2Hadronizer";
}

//----------------------------------------------------------------------------------------------

//______________________________________________________________________________________________
//f2=f(phi,r)
double Hydjet2Hadronizer::f2(double x, double y, double Delta) {
  LogDebug("f2") <<"in f2: "<<"delta"<<Delta; 
  double RsB = fR; //test: podstavit' *coefff_RB
  double rhou =  fUmax * y / RsB;
  double ff = y*TMath::CosH(rhou)*
    TMath::Sqrt(1+Delta*TMath::Cos(2*x)*TMath::TanH(rhou)*TMath::TanH(rhou));
  //n_mu u^mu f-la 20
  return ff;
}

//____________________________________________________________________________________________
double Hydjet2Hadronizer::SimpsonIntegrator(double a, double b, double phi, double Delta) {
  LogDebug("SimpsonIntegrator") <<"in SimpsonIntegrator"<<"delta - "<<Delta; 
  int nsubIntervals=100;
  double h = (b - a)/nsubIntervals;
  double s = f2(phi,a + 0.5*h,Delta);
  double t = 0.5*(f2(phi,a,Delta) + f2(phi,b,Delta));
  double x = a;
  double y = a + 0.5*h;
  for(int i = 1; i < nsubIntervals; i++) {
    x += h;
    y += h;
    s += f2(phi,y,Delta);
    t += f2(phi,x,Delta);
  }	
  t += 2.0*s;
  return t*h/3.0;
}

//______________________________________________________________________________________________
double Hydjet2Hadronizer::SimpsonIntegrator2(double a, double b, double Epsilon, double Delta) {

  LogInfo("SimpsonIntegrator2") <<"in SimpsonIntegrator2: epsilon - "<<Epsilon<<" delta - "<<Delta; 
  int nsubIntervals=10000;
  double h = (b - a)/nsubIntervals; //-1-pi, phi
  double s=0;
  //  double h2 = (fR)/nsubIntervals; //0-R maximal RB ?

  double x = 0; //phi
  for(int j = 1; j < nsubIntervals; j++) {
    x += h; // phi
    double e = Epsilon;
    double RsB = fR; //test: podstavit' *coefff_RB
    double RB = RsB *(TMath::Sqrt(1-e*e)/TMath::Sqrt(1+e*TMath::Cos(2*x))); //f-la7 RB    
    double sr = SimpsonIntegrator(0,RB,x,Delta);
    s += sr;
  }
  return s*h;

}

//___________________________________________________________________________________________________
double Hydjet2Hadronizer::MidpointIntegrator2(double a, double b, double Delta, double Epsilon) {

  int nsubIntervals=2000; 
  int nsubIntervals2=1; 
  double h = (b - a)/nsubIntervals; //0-pi , phi
  double h2 = (fR)/nsubIntervals; //0-R maximal RB ?

  double x = a + 0.5*h;
  double y = 0;
      
  double t = f2(x,y,Delta);                    
 
  double e = Epsilon;

  for(int j = 1; j < nsubIntervals; j++) {
    x += h; // integr  phi

    double RsB = fR; //test: podstavit' *coefff_RB
    double  RB = RsB *(TMath::Sqrt(1-e*e)/TMath::Sqrt(1+e*TMath::Cos(2*x))); //f-la7 RB

    nsubIntervals2 = int(RB / h2)+1;
    // integr R 
    y=0;
    for(int i = 1; i < nsubIntervals2; i++) 
      t += f2(x,(y += h2),Delta);
  }
  return t*h*h2;
}

//__________________________________________________________________________________________________________
double Hydjet2Hadronizer::CharmEnhancementFactor(double Ncc, double Ndth, double NJPsith, double Epsilon) {

  double gammaC=100.;
  double x1 = gammaC*Ndth; 
  double var1 = Ncc-0.5*gammaC*Ndth*TMath::BesselI1(x1)/TMath::BesselI0(x1)-gammaC*gammaC*NJPsith;
  LogInfo("Charam") << "gammaC 20"<<" var "<<var1<<endl;
  gammaC=1.;
  double x0 = gammaC*Ndth;
  double var0 = Ncc-0.5*gammaC*Ndth*TMath::BesselI1(x0)/TMath::BesselI0(x0)-gammaC*gammaC*NJPsith;
  LogInfo("Charam") << "gammaC 1"<<" var "<<var0;
  
  for(int i=1; i<1000; i++){ 
    if(var1 * var0<0){
      gammaC=gammaC+0.01*i;
      double x = gammaC*Ndth;  
      var0 = Ncc-0.5*gammaC*Ndth*TMath::BesselI1(x)/TMath::BesselI0(x)-gammaC*gammaC*NJPsith;
    }
    else
      {
        LogInfo("Charam") << "gammaC "<<gammaC<<" var0 "<<var0;
        return gammaC;
      } 

  }
  LogInfo("Charam") << "gammaC not found ? "<<gammaC<<" var0 "<<var0;
  return -100;
}
//----------------------------------------------------------------------------------------------



//_____________________________________________________________________




//________________________________________________________________
void Hydjet2Hadronizer::rotateEvtPlane()
{
  const double pi = 3.14159265358979;
  phi0_ = 2.*pi*gen::pyr_(0) - pi;
  sinphi0_ = sin(phi0_);
  cosphi0_ = cos(phi0_);
}

//_____________________________________________________________________
bool Hydjet2Hadronizer::get_particles(HepMC::GenEvent *evt )
{
  // Hard particles. The first nhard_ lines from hyjets array.
  // Pythia/Pyquen sub-events (sub-collisions) for a given event
  // Return T/F if success/failure
  // Create particles from lujet entries, assign them into vertices and
  // put the vertices in the GenEvent, for each SubEvent
  // The SubEvent information is kept by storing indeces of main vertices
  // of subevents as a vector in GenHIEvent.
  LogDebug("SubEvent")<< "Number of sub events "<<nsub_;
  LogDebug("Hydjet2")<<"Number of hard events "<<Njet;
  LogDebug("Hydjet2")<<"Number of hard particles "<<nhard_;
  LogDebug("Hydjet2")<<"Number of soft particles "<<nsoft_;

  vector<HepMC::GenVertex*> sub_vertices(nsub_);

  int ihy = 0;
  for(int isub=0;isub<nsub_;isub++){
    LogDebug("SubEvent") <<"Sub Event ID : "<<isub;

    int sub_up = (isub+1)*50000; // Upper limit in mother index, determining the range of Sub-Event
    vector<HepMC::GenParticle*> particles;
    vector<int> mother_ids;
    vector<HepMC::GenVertex*> prods;

    sub_vertices[isub] = new HepMC::GenVertex(HepMC::FourVector(0,0,0,0),isub);
    evt->add_vertex(sub_vertices[isub]);

    if(!evt->signal_process_vertex()) evt->set_signal_process_vertex(sub_vertices[isub]);

    while(ihy<nhard_+nsoft_ && (MotherIndex[ihy] < sub_up || ihy > nhard_ )){
      particles.push_back(build_hyjet2(ihy,ihy+1));
      prods.push_back(build_hyjet2_vertex(ihy,isub));
      mother_ids.push_back(MotherIndex[ihy]);
      LogDebug("DecayChain")<<"Mother index : "<<MotherIndex[ihy];
      ihy++;
    }
    //Produce Vertices and add them to the GenEvent. Remember that GenParticles are adopted by
    //GenVertex and GenVertex is adopted by GenEvent.
    LogDebug("Hydjet2")<<"Number of particles in vector "<<particles.size();

    for (unsigned int i = 0; i<particles.size(); i++) {

      HepMC::GenParticle* part = particles[i];
      //The Fortran code is modified to preserve mother id info, by seperating the beginning
      //mother indices of successive subevents by 5000
      int mid = mother_ids[i]-isub*50000-1;
      LogDebug("DecayChain")<<"Particle "<<i;
      LogDebug("DecayChain")<<"Mother's ID "<<mid;
      LogDebug("DecayChain")<<"Particle's PDG ID "<<part->pdg_id();

      if(mid <= 0){

        sub_vertices[isub]->add_particle_out(part);
        continue;
      }

      if(mid > 0){
        HepMC::GenParticle* mother = particles[mid];
        LogDebug("DecayChain")<<"Mother's PDG ID "<<mother->pdg_id();
        HepMC::GenVertex* prod_vertex = mother->end_vertex();
        if(!prod_vertex){
          prod_vertex = prods[i];
          prod_vertex->add_particle_in(mother);
          evt->add_vertex(prod_vertex);
          prods[i]=0; // mark to protect deletion
        }

        prod_vertex->add_particle_out(part);

      }
    }

    // cleanup vertices not assigned to evt
    for (unsigned int i = 0; i<prods.size(); i++) {
      if(prods[i]) delete prods[i];
    }
  }

  return kTRUE;
}


//___________________________________________________________________     
HepMC::GenParticle* Hydjet2Hadronizer::build_hyjet2(int index, int barcode)
{
  // Build particle object corresponding to index in hyjets (soft+hard)  
 
  double px0 = Px[index];
  double py0 = Py[index];
 
  double px = px0*cosphi0_-py0*sinphi0_;
  double py = py0*cosphi0_+px0*sinphi0_;
 
  HepMC::GenParticle* p = new HepMC::GenParticle(
                                                 HepMC::FourVector(
					px,	// px
					py,	// py
					Pz[index],	// pz
					E[index]),	// E
					pdg[index],	// id
                                                 	convertStatus(final[index]) // status
                       
                                                 );
 
  p->suggest_barcode(barcode);
  return p;
}

//___________________________________________________________________     
HepMC::GenVertex* Hydjet2Hadronizer::build_hyjet2_vertex(int i,int id)
{
  // build verteces for the hyjets stored events                        
 
  double x0=X[i];
  double y0=Y[i];
  double x = x0*cosphi0_-y0*sinphi0_;
  double y = y0*cosphi0_+x0*sinphi0_;
  double z=Z[i];
  double t=T[i];
 
  HepMC::GenVertex* vertex = new HepMC::GenVertex(HepMC::FourVector(x,y,z,t),id);
  return vertex;
}

//_____________________________________________________________________
void Hydjet2Hadronizer::add_heavy_ion_rec(HepMC::GenEvent *evt)
{
  // heavy ion record in the final CMSSW Event
  double npart = Npart; 
  int nproj = static_cast<int>(npart / 2);
  int ntarg = static_cast<int>(npart - nproj);
 
  HepMC::HeavyIon* hi = new HepMC::HeavyIon(
                                            nsub_,                	// Ncoll_hard/N of SubEvents
                                            nproj,          	// Npart_proj
                                            ntarg,          	// Npart_targ
                                            Nbcol,    		// Ncoll
                                            0,                        	// spectator_neutrons
                                            0,                      	// spectator_protons
                                            0,                      	// N_Nwounded_collisions
                                            0,                       	// Nwounded_N_collisions
                                            0,                       	// Nwounded_Nwounded_collisions
                                            Bgen * nuclear_radius(), 	// impact_parameter in [fm]
                                            phi0_,                   	// event_plane_angle
                                            0,                       	// eccentricity
                                            Sigin                    	// sigma_inel_NN
                                            );

  evt->set_heavy_ion(*hi);
  delete hi;
}




