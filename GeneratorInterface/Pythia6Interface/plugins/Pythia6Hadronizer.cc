   
// -*- C++ -*-

#include "Pythia6Hadronizer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"
#include "HepMC/PdfInfo.h"
#include "HepMC/PythiaWrapper6_2.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "HepMC/IO_HEPEVT.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "GeneratorInterface/Core/interface/FortranCallback.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/Core/interface/RNDMEngineAccess.h"

#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"

 
HepMC::IO_HEPEVT conv;

// NOTE: here a number of Pythia6 routines are declared,
// plus some functionalities to pass around Pythia6 params
//
#include "Pythia6Service.h"
#include "Pythia6Declarations.h"

namespace gen
{

extern "C" {
   void upinit_() { FortranCallback::getInstance()->fillHeader(); return; }
   void upevnt_() { 
      FortranCallback::getInstance()->fillEvent(); 
      if ( !Pythia6Hadronizer::getJetMatching() ) return;
      
      Pythia6Hadronizer::getJetMatching()->beforeHadronisationExec();
      return ; 
   }
   
   void upveto_(int* veto) { 
         
      if ( !Pythia6Hadronizer::getJetMatching() )
      {
         *veto=0;
	 return;
      }
      
      if ( !hepeup_.nup || Pythia6Hadronizer::getJetMatching()->isMatchingDone() )
      { 
         *veto=1;
         return;
      }
      
      // NOTE: I'm passing NULL pointers, instead of HepMC::GenEvent, etc.
      //   
      *veto = Pythia6Hadronizer::getJetMatching()->match(0,0,true); 
   
      return; 
   }

} // extern "C"


JetMatching* Pythia6Hadronizer::fJetMatching = 0;

Pythia6Hadronizer::Pythia6Hadronizer(edm::ParameterSet const& ps) 
   : fPy6Service( new Pythia6Service(ps) ), // this will store py6 params for further settings
     fCOMEnergy(ps.getParameter<double>("comEnergy")),
     fEventCounter(0),
     fRunInfo(0),
     fEventInfo(0),
     fHepMCVerbosity(ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
     fMaxEventsToPrint(ps.getUntrackedParameter<int>("maxEventsToPrint", 0)),
     fPythiaListVerbosity(ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0))
{ 

   if ( ps.exists("jetMatching") )
   {
      edm::ParameterSet jmParams =
			ps.getUntrackedParameter<edm::ParameterSet>(
								"jetMatching");
      std::string scheme = jmParams.getParameter<std::string>("scheme");
      if ( scheme == "Madgraph" )
      {
         if ( !fJetMatching) fJetMatching = new JetMatchingMadgraph(jmParams);
      }
      else if ( scheme == "MLM" )
      {
         throw cms::Exception("InvalidJetMatching")
			<< "Port of " << scheme << "scheme \"" << "\""
			   " for parton-shower matching is still in progress."
			<< std::endl;
      }
      else
      {
         throw cms::Exception("InvalidJetMatching")
			<< "Unknown scheme \"" << scheme << "\""
			   " specified for parton-shower matching."
			<< std::endl;
      }
   }
   
   fGenRunInfo.setFilterEfficiency(
      ps.getUntrackedParameter<double>("filterEfficiency", -1.) );
   //
   // fill up later 
   //
   //fGenRunInfo.setsetExternalXSecLO(
   //   GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("...", -1.)) );
   //fGenRunInfo.setsetExternalXSecNLO(
   //    GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("...", -1.)) );


/* old stuff 
   edm::Service<edm::RandomNumberGenerator> rng;
   int seed = rng->mySeed();
   std::cout << " seed = " << seed << std::endl;
   std::ostringstream ss;
   ss << "MRPY(1)=" << rng->mySeed();
   paramGeneral.push_back(ss.str());
*/
   // Initialize the random engine unconditionally
   //
   randomEngine = &getEngineReference();

   // first of all, silence Pythia6 banner printout
   if (!call_pygive("MSTU(12)=12345")) 
   {
      throw edm::Exception(edm::errors::Configuration,"PythiaError") 
          <<" Pythia did not accept MSTU(12)=12345";
   }
   
}

Pythia6Hadronizer::~Pythia6Hadronizer()
{
   if ( fPy6Service != 0 ) delete fPy6Service;
   if ( fRunInfo != 0 ) delete fRunInfo ;
   if ( fJetMatching != 0 ) delete fJetMatching;
}

void Pythia6Hadronizer::finalizeEvent()
{
      
   // convert to HEPEVT
   //
   //call_pyhepc(1);
      
   // convert to HepMC
   //
   //fGenEvent = conv.read_next_event();
      
   fGenEvent->set_signal_process_id(pypars.msti[0]);
   fGenEvent->set_event_scale(pypars.pari[16]);
   // evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
   fGenEvent->set_event_number( fEventCounter );
   
   // get pdf info directly from Pythia6 and set it up into HepMC::GenEvent
   //
   int id1 = pyint1.mint[14];
   int id2 = pyint1.mint[15];
   if ( id1 == 21 ) id1 = 0;
   if ( id2 == 21 ) id2 = 0; 
   double x1 = pyint1.vint[40];
   double x2 = pyint1.vint[41];  
   double Q  = pyint1.vint[50];
   double pdf1 = pyint1.vint[38];
   pdf1 /= x1 ;
   double pdf2 = pyint1.vint[39];
   pdf2 /= x2 ;
   fGenEvent->set_pdf_info( HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2) ) ;
    
   fGenEvent->weights().push_back( pyint1.vint[96] );
   
   // service printouts, if requested
   //
   if (fMaxEventsToPrint > 0) 
   {
      fMaxEventsToPrint--;
      if (fPythiaListVerbosity) call_pylist(fPythiaListVerbosity);
      if (fHepMCVerbosity) 
      {
         std::cout << "Event process = " << pypars.msti[0] << std::endl 
	      << "----------------------" << std::endl;
         fGenEvent->print();
      }
   }
   
   return;
}

bool Pythia6Hadronizer::generatePartonsAndHadronize()
{

   FortranCallback::getInstance()->resetIterationsPerEvent();
   
   // generate event with Pythia6
   //
   call_pyevnt();
   
   //formEvent();
   call_pyhepc(1);
   fGenEvent.reset( conv.read_next_event() );
   
   fEventCounter++;
      
   return true;
}

bool Pythia6Hadronizer::hadronize()
{
   
   FortranCallback::getInstance()->resetIterationsPerEvent();
   if ( fJetMatching != NULL) 
   {
      fJetMatching->resetMatchingStatus() ;
      fJetMatching->beforeHadronisation(fEventInfo);
   }
      
   // generate event with Pythia6
   //
   call_pyevnt();
      
   if ( FortranCallback::getInstance()->getIterationsPerEvent() > 1 || 
        hepeup_.nup <= 0 || pypars.msti[0] == 1 )
   {
      fGenEvent.reset();
/*
      std::cout << " terminating loop inside event because of : " << 
      FortranCallback::getInstance()->getIterationsPerEvent() << " " <<
      hepeup_.nup << " " << pypars.msti[0] << std::endl;
*/
      return false;
   }
      
   //formEvent();
   call_pyhepc(1);
   fGenEvent.reset( conv.read_next_event() );
   
   fEventCounter++;
      
   return true;
}

bool Pythia6Hadronizer::decay()
{
   return true;
}

bool Pythia6Hadronizer::residualDecay()
{
   return true;
}

bool Pythia6Hadronizer::initializeForExternalPartons()
{
     
   // note: CSA mode is NOT supposed to woirk with external partons !!!
   
   fPy6Service->setGeneralParams();

      
   call_pyinit("USER", "", "", 0.0);
      
   std::vector<std::string> slha = fRunInfo->findHeader("slha");
   if (!slha.empty()) {
		edm::LogInfo("Generator|LHEInterface")
			<< "Pythia6 hadronisation found an SLHA header, "
			<< "will be passed on to Pythia." << std::endl;
      fPy6Service->setSLHAFromHeader(slha);   
      fPy6Service->closeSLHA();
   }


   if ( fJetMatching != NULL ) 
   {
      fJetMatching->init( fRunInfo );
      call_pygive("MSTP(143)=1");
/*
   call_pygive(std::string("MSTP(143)=") +
	      (wantsShoweredEvent() ? "1" : "0"));
*/
   }

   return true;
}

bool Pythia6Hadronizer::initializeForInternalPartons()
{
    
   fPy6Service->setGeneralParams();   
   fPy6Service->setCSAParams();
   fPy6Service->setSLHAParams();
   
   call_pyinit("CMS", "p", "p", fCOMEnergy);
   
   fPy6Service->closeSLHA();
   
   return true;
}

bool Pythia6Hadronizer::declareStableParticles( std::vector<int> pdg )
{
   
   for ( size_t i=0; i<pdg.size(); i++ )
   {
      int pyCode = pycomp_( pdg[i] );
      std::ostringstream pyCard ;
      pyCard << "MDCY(" << pyCode << ",1)=0";
      std::cout << pyCard.str() << std::endl;
      call_pygive( pyCard.str() );
   }
   
   return true;
}

void Pythia6Hadronizer::statistics()
{

  double cs = pypars.pari[0]; // cross section in mb
  fGenRunInfo.setInternalXSec(GenRunInfoProduct::XSec(cs));

  call_pystat(1);
  
  return;

}

const char* Pythia6Hadronizer::classname() const
{
   return "gen::Pythia6Hadronizer";
}

void Pythia6Hadronizer::setLHERunInfo( lhef::LHERunInfo* lheri )
{

   fRunInfo = lheri;
   
   FortranCallback::getInstance()->setLHERunInfo(lheri);

   return;

}

void Pythia6Hadronizer::setLHEEventProd( LHEEventProduct* lheep )
{

   fEventInfo = lheep;
   
   FortranCallback::getInstance()->setLHEEventProd(lheep);

   return;

}

void Pythia6Hadronizer::resetEvent( HepMC::GenEvent* e )
{

   // here I have to reset contents of HEPEVT, for consistency, 
   // and also of PYJETS !!!
   //
   fGenEvent.reset(e);

   return;
}

} // namespace gen
