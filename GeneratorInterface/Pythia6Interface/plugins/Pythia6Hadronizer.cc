   
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

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"

 
HepMC::IO_HEPEVT conv;

#include "HepPID/ParticleIDTranslations.hh"

// NOTE: here a number of Pythia6 routines are declared,
// plus some functionalities to pass around Pythia6 params
//
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"
#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Declarations.h"

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
      *veto = Pythia6Hadronizer::getJetMatching()->match(0, 0, true);

      return; 
   }
   
   //
   // these two are NOT part of Pythi6 core code but are "custom" add-ons
   // we keep them interfaced here, rather than in GenExtensions, because
   // they tweak not at the ME level, but a step further, at the framgmentation
   //
   // stop-hadrons
   //
   void pystrhad_();   // init stop-hadrons (id's, names, charges...)
   void pystfr_(int&); // tweaks fragmentation, fragments the string near to a stop, 
                       // to form stop-hadron by producing a new q-qbar pair
   // gluino/r-hadrons
   void pyglrhad_();
   void pyglfr_();   // tweaks fragmentation, fragment the string near to a gluino,
                     // to form gluino-hadron, either by producing a new g-g pair,
		     // or two new q-qbar ones


} // extern "C"


JetMatching* Pythia6Hadronizer::fJetMatching = 0;

Pythia6Hadronizer::Pythia6Hadronizer(edm::ParameterSet const& ps) 
   : fPy6Service( new Pythia6Service(ps) ), // this will store py6 params for further settings
     fCOMEnergy(ps.getParameter<double>("comEnergy")),
     fHepMCVerbosity(ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
     fMaxEventsToPrint(ps.getUntrackedParameter<int>("maxEventsToPrint", 0)),
     fPythiaListVerbosity(ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0))
{ 

   // J.Y.: the following 3 params are "hacked", in the sense 
   // that they're tracked but get in optionally;
   // this will be fixed once we update all applications
   //

   fStopHadronsEnabled = false;
   if ( ps.exists( "stopHadrons" ) )
      fStopHadronsEnabled = ps.getParameter<bool>("stopHadrons") ;

   fGluinoHadronsEnabled = false;
   if ( ps.exists( "gluinoHadrons" ) )
      fGluinoHadronsEnabled = ps.getParameter<bool>("gluinoHadrons");
   
   fImposeProperTime = false;
   if ( ps.exists( "imposeProperTime" ) )
   {
      fImposeProperTime = ps.getParameter<bool>("imposeProperTime");
   }
   
   fConvertToPDG = false;
   if ( ps.exists( "doPDGConvert" ) )
   {
      fConvertToPDG = ps.getParameter<bool>("doPDGConvert");
   }
   
   if ( ps.exists("jetMatching") )
   {
      edm::ParameterSet jmParams =
			ps.getUntrackedParameter<edm::ParameterSet>(
								"jetMatching");

      fJetMatching = JetMatching::create(jmParams).release();
   }
   
   runInfo().setFilterEfficiency(
      ps.getUntrackedParameter<double>("filterEfficiency", -1.) );
   //
   // fill up later, note C.S. -> for all generators via BaseHadronizer?
   //
   //runInfo().setsetExternalXSecLO(
   //   GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("...", -1.)) );
   //runInfo().setsetExternalXSecNLO(
   //    GenRunInfoProduct::XSec(ps.getUntrackedParameter<double>("...", -1.)) );


   // Initialize the random engine unconditionally
   //
   randomEngine = &getEngineReference();

   // first of all, silence Pythia6 banner printout
   //
   if (!call_pygive("MSTU(12)=12345")) 
   {
      throw edm::Exception(edm::errors::Configuration,"PythiaError") 
          <<" Pythia did not accept MSTU(12)=12345";
   }
   
}

Pythia6Hadronizer::~Pythia6Hadronizer()
{
   if ( fPy6Service != 0 ) delete fPy6Service;
   if ( fJetMatching != 0 ) delete fJetMatching;
}

void Pythia6Hadronizer::finalizeEvent()
{

   bool lhe = lheEvent() != 0;

   HepMC::PdfInfo pdf;

   // if we are in hadronizer mode, we can pass on information from
   // the LHE input
   if (lhe)
   {
      lheEvent()->fillEventInfo( event().get() );
      lheEvent()->fillPdfInfo( &pdf );
   }

   if (!lhe || event()->signal_process_id() < 0) event()->set_signal_process_id( pypars.msti[0] );
   if (!lhe || event()->event_scale() < 0)       event()->set_event_scale( pypars.pari[16] );
   
// FIXME event scale is *not* pthat.  We now have the binningValue() for that
//     -> the event scale Q would be pypars.pari[22]
//        If we change this, we need to announce it!!!
// FIXME: alpha qcd, alpha qed
// FIXME: signal_process_vertex

   // get pdf info directly from Pythia6 and set it up into HepMC::GenEvent
   // S. Mrenna: Prefer vint block
   //
   if (!lhe || pdf.id1() < 0)      pdf.set_id1( pyint1.mint[14] == 21 ? 0 : pyint1.mint[14] );
   if (!lhe || pdf.id2() < 0)      pdf.set_id2( pyint1.mint[15] == 21 ? 0 : pyint1.mint[15] );
   if (!lhe || pdf.x1() < 0)       pdf.set_x1( pyint1.vint[40] );
   if (!lhe || pdf.x2() < 0)       pdf.set_x2( pyint1.vint[41] );
   if (!lhe || pdf.pdf1() < 0)     pdf.set_pdf1( pyint1.vint[38] / pyint1.vint[40] );
   if (!lhe || pdf.pdf2() < 0)     pdf.set_pdf2( pyint1.vint[39] / pyint1.vint[41] );
   if (!lhe || pdf.scalePDF() < 0) pdf.set_scalePDF( pyint1.vint[50] );

   event()->set_pdf_info( pdf ) ;

   event()->weights().push_back( pyint1.vint[96] );

   // here we treat long-lived particles
   //
   if ( fImposeProperTime || pydat1.mstj[21]==3 || pydat1.mstj[21]==4 ) imposeProperTime();

   // final touch - convert Py6->PDG, if requested
   //
   if ( fConvertToPDG )
   {
      // do conversion here
/* this comes from example by Todd Adams, see talk at the Gen meeting on 03/16/09

      for ( HepMC::GenEvent::particle_iterator part = event()->particles_begin();
         part != event()->particles_end(); ++part) {
         if ((*part)->pdg_id() != HepPID::translatePythiatoPDT((*part)->pdg_id())) {
           std::cout << " found a change orig=" << (*part)->pdg_id() << " new="
                     << HepPID::translatePythiatoPDT((*part)->pdg_id()) << std::endl;
         }
         (*part)->set_pdg_id(HepPID::translatePythiatoPDT((*part)->pdg_id()));
      }

*/
   }
   
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
         event()->print();
      }
   }
   
   return;
}

bool Pythia6Hadronizer::generatePartonsAndHadronize()
{

   FortranCallback::getInstance()->resetIterationsPerEvent();
   
   // generate event with Pythia6
   //
   
   if ( fStopHadronsEnabled || fGluinoHadronsEnabled )
   {
      // call_pygive("MSTJ(1)=-1");
      call_pygive("MSTJ(14)=-1");
   }
   
   call_pyevnt();
   
   if ( fStopHadronsEnabled || fGluinoHadronsEnabled )
   {
      // call_pygive("MSTJ(1)=1");
      call_pygive("MSTJ(14)=1");
      int ierr=0;
      if ( fStopHadronsEnabled ) pystfr_(ierr);
      if ( fGluinoHadronsEnabled ) pyglfr_();
   }
   
   //formEvent();
   call_pyhepc(1);
   event().reset( conv.read_next_event() );
   
   return true;
}

bool Pythia6Hadronizer::hadronize()
{
   FortranCallback::getInstance()->setLHEEvent( lheEvent() );
   FortranCallback::getInstance()->resetIterationsPerEvent();
   if ( fJetMatching )
   {
      fJetMatching->resetMatchingStatus() ;
      fJetMatching->beforeHadronisation( lheEvent() );
   }

   // generate event with Pythia6
   //
   if ( fStopHadronsEnabled || fGluinoHadronsEnabled )
   {
      call_pygive("MSTJ(1)=-1");
      call_pygive("MSTJ(14)=-1");
   }

   call_pyevnt();

   if ( FortranCallback::getInstance()->getIterationsPerEvent() > 1 || 
        hepeup_.nup <= 0 || pypars.msti[0] == 1 )
   {
      // update LHE matching statistics
      lheEvent()->count( lhef::LHERunInfo::kSelected );

      event().reset();
      return false;
   }

   // update LHE matching statistics
   //
   lheEvent()->count( lhef::LHERunInfo::kAccepted );

   if ( fStopHadronsEnabled || fGluinoHadronsEnabled )
   {
      call_pygive("MSTJ(1)=1");
      call_pygive("MSTJ(14)=1");
      int ierr = 0;
      if ( fStopHadronsEnabled ) pystfr_(ierr);
      if ( fGluinoHadronsEnabled ) pyglfr_();
   }

   call_pyhepc(1);
   event().reset( conv.read_next_event() );
   
   return true;
}

bool Pythia6Hadronizer::decay()
{
   return true;
}

bool Pythia6Hadronizer::residualDecay()
{
   
   // int nDocLines = pypars.msti[3];
   
   // because the counter in HEPEVT might have been reset already, 
   // get the Npart directly from pyjets
   
   int NPartsBeforeDecays = pyjets.n ;
   int NPartsAfterDecays = event()->particles_size();
   
   std::vector<int> part_idx_to_decay;
   
   //
   // here put additional info back to pyjets BY HANDS
   //
   
   for ( int ipart=NPartsBeforeDecays+1; ipart<=NPartsAfterDecays; ipart++ )
   {
      HepMC::GenParticle* part = event()->barcode_to_particle( ipart );
      int status = part->status();
      int pdgid = part->pdg_id();
      // add part to pyjets, with proper links/pointers
      if ( status == 1 )
      {
         pyjets.k[0][ipart-1] = 1;
      }
      else if ( status = 2 )
      {
         pyjets.k[0][ipart-1] = 11;
      }
      else if ( status == 3 )
      {
         pyjets.k[0][ipart-1] = 21;
      }
      int py6id = pycomp_( pdgid );
      pyjets.k[1][ipart-1] = pdgid;
      HepMC::GenVertex* prod_vtx = part->production_vertex();
      assert ( prod_vtx->particles_in_size() == 1 );
      HepMC::GenParticle* mother = (*prod_vtx->particles_in_const_begin());      
      int mother_id = mother->barcode();
      pyjets.k[2][ipart-1] = mother_id;
      //
      // here also reset status & dauthters for this mother, if needs be
      //
      if ( mother->end_vertex() )
      {
         if ( pyjets.k[0][mother_id-1] == 1 )  pyjets.k[0][mother_id-1] = 11;
	 pyjets.k[3][mother_id-1] = ipart;
         pyjets.k[4][mother_id-1] = ipart + mother->end_vertex()->particles_out_size();
      }
      //
      HepMC::GenVertex* end_vtx = part->end_vertex();      
      if ( end_vtx )
      {
        pyjets.k[3][ipart-1] = (*end_vtx->particles_out_const_begin())->barcode();
        pyjets.k[4][ipart-1] = pyjets.k[3][ipart-1] + end_vtx->particles_out_size();
      }
      else
      {
         pyjets.k[3][ipart-1] = 0;
         pyjets.k[4][ipart-1] = 0;
      }
      pyjets.p[0][ipart-1] = part->momentum().x();
      pyjets.p[1][ipart-1] = part->momentum().y();
      pyjets.p[2][ipart-1] = part->momentum().z();
      pyjets.p[3][ipart-1] = part->momentum().t();
      pyjets.p[4][ipart-1] = part->generated_mass();
      //
      // should I make any vtx adjustments, like in pyhepc(2) ???
      //
      pyjets.v[0][ipart-1] = prod_vtx->position().x();
      pyjets.v[1][ipart-1] = prod_vtx->position().y();
      pyjets.v[2][ipart-1] = prod_vtx->position().z();
      pyjets.v[3][ipart-1] = prod_vtx->position().t();
      pyjets.v[4][ipart-1] = 0.;
      //
      // here also fill in missing info on colour connection in jet systems
      // see pyhep(2) as an example !!!
      //
/*
          IF(ISTHEP(I).EQ.2.AND.PHEP(4,I).GT.PHEP(5,I)) THEN
            I1=JDAHEP(1,I)
            IF(I1.GT.0.AND.I1.LE.NHEP) V(I,5)=(VHEP(4,I1)-VHEP(4,I))*
     &      PHEP(5,I)/PHEP(4,I)
          ENDIF

*/
      if ( status == 2 && part->momentum().t() > part->generated_mass() )
      {
	 HepMC::GenParticle* daughter1 = event()->barcode_to_particle( pyjets.k[3][ipart-1] );
	 if ( daughter1 ) 
	 {
	    pyjets.v[4][ipart-1] = (daughter1->production_vertex()->position()).t() 
	                         - pyjets.v[3][ipart-1];
	    pyjets.v[4][ipart-1] *= ( pyjets.p[4][ipart-1] / pyjets.p[3][ipart-1] );
	 }
      }
      // check particle status and whether should be further decayed
      pyjets.n++;
      if ( status == 2 || status == 3 ) continue;
// these 2 lines below are from the old code
//-->      if ( abs(pdgid) < 100 ) continue;
//-->      if ( abs(pdgid) == 211 || abs(pdgid) == 321 || abs(pdgid) == 130 || abs(pdgid) == 310 ) continue;
      if ( pydat3.mdcy[0][py6id-1] != 1 ) continue; // particle is not expected to decay
      // 
      // now mark for  decay
      //
      part_idx_to_decay.push_back(ipart);
      // etc.
   }
   
   for ( size_t ip=0; ip<part_idx_to_decay.size(); ip++ )
   {         
      pydecy_(part_idx_to_decay[ip]);
   }
        
   call_pyhepc(1);
   
   event().reset( conv.read_next_event() );
   
   return true;
}

bool Pythia6Hadronizer::initializeForExternalPartons()
{

   // note: CSA mode is NOT supposed to woirk with external partons !!!
   
   fPy6Service->setGeneralParams();

   FortranCallback::getInstance()->setLHERunInfo( lheRunInfo() );

   if ( fStopHadronsEnabled )
   {
      // overwrite mstp(111), no matter what
      call_pygive("MSTP(111)=0");
      pystrhad_();
      call_pygive("MWID(302)=0");   // I don't know if this is specific to processing ME/LHE only,
      call_pygive("MDCY(302,1)=0"); // or this should also be the case for full event...
                                    // anyway, this comes from experience of processing MG events
   }
   
   if ( fGluinoHadronsEnabled )
   {
      // overwrite mstp(111), no matter what
      call_pygive("MSTP(111)=0");
      pyglrhad_();
      //call_pygive("MWID(309)=0");   
      //call_pygive("MDCY(309,1)=0"); 
   }
   
   call_pyinit("USER", "", "", 0.0);

   std::vector<std::string> slha = lheRunInfo()->findHeader("slha");
   if (!slha.empty()) {
		edm::LogInfo("Generator|LHEInterface")
			<< "Pythia6 hadronisation found an SLHA header, "
			<< "will be passed on to Pythia." << std::endl;
      fPy6Service->setSLHAFromHeader(slha);   
      fPy6Service->closeSLHA();
   }

   if ( fJetMatching )
   {
      fJetMatching->init( lheRunInfo() );
// FIXME: the jet matching routine might not be interested in PS callback
      call_pygive("MSTP(143)=1");
   }

   return true;
}

bool Pythia6Hadronizer::initializeForInternalPartons()
{
    
   fPy6Service->setGeneralParams();   
   fPy6Service->setCSAParams();
   fPy6Service->setSLHAParams();
   
   if ( fStopHadronsEnabled )
   {
      // overwrite mstp(111), no matter what
      call_pygive("MSTP(111)=0");
      pystrhad_();
   }
   
   if ( fGluinoHadronsEnabled )
   {
      // overwrite mstp(111), no matter what
      call_pygive("MSTP(111)=0");
      pyglrhad_();
   }
   
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

void Pythia6Hadronizer::imposeProperTime()
{

   // this is practically a copy/paste of the original code by J.Alcaraz, 
   // taken directly from PythiaSource
    
   int dumm=0;
   HepMC::GenEvent::vertex_const_iterator vbegin = event()->vertices_begin();
   HepMC::GenEvent::vertex_const_iterator vend = event()->vertices_end();
   HepMC::GenEvent::vertex_const_iterator vitr = vbegin;
   for (; vitr != vend; ++vitr ) 
   {
      HepMC::GenVertex::particle_iterator pbegin = (*vitr)->particles_begin(HepMC::children);
      HepMC::GenVertex::particle_iterator pend = (*vitr)->particles_end(HepMC::children);
      HepMC::GenVertex::particle_iterator pitr = pbegin;
      for (; pitr != pend; ++pitr) 
      {
         if ((*pitr)->end_vertex()) continue;
         if ((*pitr)->status()!=1) continue;
         
	 int pdgcode= abs((*pitr)->pdg_id());
         // Do nothing if the particle is not expected to decay
         if ( pydat3.mdcy[0][pycomp_(pdgcode)-1] !=1 ) continue;

         double ctau = pydat2.pmas[3][pycomp_(pdgcode)-1];
         HepMC::FourVector mom = (*pitr)->momentum();
         HepMC::FourVector vin = (*vitr)->position();
         double x = 0.;
         double y = 0.;
         double z = 0.;
         double t = 0.;
         bool decayInRange = false;
         while (!decayInRange) 
	 {
            double unif_rand = pyr_(&dumm);
            // Value of 0 is excluded, so following line is OK
            double proper_length = - ctau * log(unif_rand);
            double factor = proper_length/mom.m();
            x = vin.x() + factor * mom.px();
            y = vin.y() + factor * mom.py();
            z = vin.z() + factor * mom.pz();
            t = vin.t() + factor * mom.e();
            // Decay must be happen outside a cylindrical region
            if (pydat1.mstj[21]==4) {
               if (std::sqrt(x*x+y*y)>pydat1.parj[72] || fabs(z)>pydat1.parj[73]) decayInRange = true;
               // Decay must be happen outside a given sphere
               } 
	       else if (pydat1.mstj[21]==3) {
                  if (std::sqrt(x*x+y*y+z*z)>pydat1.parj[71]) decayInRange = true;
               } 
               // Decay is always OK otherwise
	       else {
                  decayInRange = true;
               }
         }
                  
         HepMC::GenVertex* vdec = new HepMC::GenVertex(HepMC::FourVector(x,y,z,t));
         event()->add_vertex(vdec);
         vdec->add_particle_in((*pitr));
      }
   }   

   return;
   
}

void Pythia6Hadronizer::statistics()
{

  if ( !runInfo().internalXSec() )
  {
     // set xsec if not already done (e.g. from LHE cross section collector)
     double cs = pypars.pari[0]; // cross section in mb
     runInfo().setInternalXSec( cs );
// FIXME: can we get the xsec statistical error somewhere?
  }

  call_pystat(1);
  
  return;

}

const char* Pythia6Hadronizer::classname() const
{
   return "gen::Pythia6Hadronizer";
}

} // namespace gen
