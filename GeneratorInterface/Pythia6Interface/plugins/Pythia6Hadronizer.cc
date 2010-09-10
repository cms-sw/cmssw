   
// -*- C++ -*-

#include "Pythia6Hadronizer.h"

#include "HepMC/GenEvent.h"
#include "HepMC/PdfInfo.h"
#include "HepMC/PythiaWrapper6_2.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "HepMC/IO_HEPEVT.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/Core/interface/FortranCallback.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

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

class Pythia6ServiceWithCallback : public Pythia6Service {
  public:
     Pythia6ServiceWithCallback( const edm::ParameterSet& ps ) : Pythia6Service(ps) {}

  private:
    void upInit()
    { FortranCallback::getInstance()->fillHeader(); }

    void upEvnt()
    {
      FortranCallback::getInstance()->fillEvent(); 
      if ( Pythia6Hadronizer::getJetMatching() )
        Pythia6Hadronizer::getJetMatching()->beforeHadronisationExec();    
    }

    bool upVeto()
    { 
      if ( !Pythia6Hadronizer::getJetMatching() )
        return false;

      if ( !hepeup_.nup || Pythia6Hadronizer::getJetMatching()->isMatchingDone() )
         return true;

      // NOTE: I'm passing NULL pointers, instead of HepMC::GenEvent, etc.
      return Pythia6Hadronizer::getJetMatching()->match(0, 0, true);
    }
};

struct {
	int n, npad, k[5][pyjets_maxn];
	double p[5][pyjets_maxn], v[5][pyjets_maxn];
} pyjets_local;

JetMatching* Pythia6Hadronizer::fJetMatching = 0;

Pythia6Hadronizer::Pythia6Hadronizer(edm::ParameterSet const& ps) 
   : BaseHadronizer(ps),
     fPy6Service( new Pythia6ServiceWithCallback(ps) ), // this will store py6 params for further settings
     fCOMEnergy(ps.getParameter<double>("comEnergy")),
     fHepMCVerbosity(ps.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
     fMaxEventsToPrint(ps.getUntrackedParameter<int>("maxEventsToPrint", 0)),
     fPythiaListVerbosity(ps.getUntrackedParameter<int>("pythiaPylistVerbosity", 0)),
     fDisplayPythiaBanner(ps.getUntrackedParameter<bool>("displayPythiaBanner",false)),
     fDisplayPythiaCards(ps.getUntrackedParameter<bool>("displayPythiaCards",false))
{ 

   // J.Y.: the following 4 params are "hacked", in the sense 
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
      fConvertToPDG = ps.getParameter<bool>("doPDGConvert");
   
   if ( ps.exists("jetMatching") )
   {
      edm::ParameterSet jmParams =
			ps.getUntrackedParameter<edm::ParameterSet>(
								"jetMatching");

      fJetMatching = JetMatching::create(jmParams).release();
   }
   
   // first of all, silence Pythia6 banner printout, unless display requested
   //
   if ( !fDisplayPythiaBanner )
   {
      if (!call_pygive("MSTU(12)=12345")) 
      {
         throw edm::Exception(edm::errors::Configuration,"PythiaError") 
             <<" Pythia did not accept MSTU(12)=12345";
      }
   }
   
// silence printouts from PYGIVE, unless display requested
//
   if ( ! fDisplayPythiaCards )
   {
      if (!call_pygive("MSTU(13)=0")) 
      {
         throw edm::Exception(edm::errors::Configuration,"PythiaError") 
             <<" Pythia did not accept MSTU(13)=0";
      }
   }

   // tmp stuff to deal with EvtGen corrupting pyjets
   // NPartsBeforeDecays = 0;
   flushTmpStorage();
   
}

Pythia6Hadronizer::~Pythia6Hadronizer()
{
   if ( fPy6Service != 0 ) delete fPy6Service;
   if ( fJetMatching != 0 ) delete fJetMatching;
}

void Pythia6Hadronizer::flushTmpStorage()
{

   pyjets_local.n = 0 ;
   pyjets_local.npad = 0 ;
   for ( int ip=0; ip<pyjets_maxn; ip++ )
   {
      for ( int i=0; i<5; i++ )
      {
         pyjets_local.k[i][ip] = 0;
	 pyjets_local.p[i][ip] = 0.;
	 pyjets_local.v[i][ip] = 0.;
      }
   }
   return;

}

void Pythia6Hadronizer::fillTmpStorage()
{

   pyjets_local.n = pyjets.n;
   pyjets_local.npad = pyjets.npad;
   for ( int ip=0; ip<pyjets_maxn; ip++ )
   {
      for ( int i=0; i<5; i++ )
      {
         pyjets_local.k[i][ip] = pyjets.k[i][ip];
	 pyjets_local.p[i][ip] = pyjets.p[i][ip];
	 pyjets_local.v[i][ip] = pyjets.v[i][ip];
      }
   }
   
   return ;

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
   else
     {
       // filling in factorization "Q scale" now! pthat moved to binningValues()
       //
            
       if ( event()->signal_process_id() <= 0) event()->set_signal_process_id( pypars.msti[0] );
       if ( event()->event_scale() <=0 )       event()->set_event_scale( pypars.pari[22] );
       if ( event()->alphaQED() <= 0)          event()->set_alphaQED( pyint1.vint[56] );
       if ( event()->alphaQCD() <= 0)          event()->set_alphaQCD( pyint1.vint[57] );
   
       // get pdf info directly from Pythia6 and set it up into HepMC::GenEvent
       // S. Mrenna: Prefer vint block
       //
       if ( pdf.id1() <= 0)      pdf.set_id1( pyint1.mint[14] == 21 ? 0 : pyint1.mint[14] );
       if ( pdf.id2() <= 0)      pdf.set_id2( pyint1.mint[15] == 21 ? 0 : pyint1.mint[15] );
       if ( pdf.x1() <= 0)       pdf.set_x1( pyint1.vint[40] );
       if ( pdf.x2() <= 0)       pdf.set_x2( pyint1.vint[41] );
       if ( pdf.pdf1() <= 0)     pdf.set_pdf1( pyint1.vint[38] / pyint1.vint[40] );
       if ( pdf.pdf2() <= 0)     pdf.set_pdf2( pyint1.vint[39] / pyint1.vint[41] );
       if ( pdf.scalePDF() <= 0) pdf.set_scalePDF( pyint1.vint[50] );   
     }
   
   /* 9/9/2010 - JVY: This is the old piece of code - I can't remember why we implemented it this way.
      However, it's causing problems with pdf1 & pdf2 when processing LHE samples,
      specifically, because both are set to -1, it tries to fill with Py6 numbers that
      are NOT valid/right at this point !
      In general, for LHE/ME event processing we should implement the correct calculation
      of the pdf's, rather than using py6 ones.
      
      // filling in factorization "Q scale" now! pthat moved to binningValues()

      if (!lhe || event()->signal_process_id() < 0) event()->set_signal_process_id( pypars.msti[0] );
      if (!lhe || event()->event_scale() < 0)       event()->set_event_scale( pypars.pari[22] );
      if (!lhe || event()->alphaQED() < 0)          event()->set_alphaQED( pyint1.vint[56] );
      if (!lhe || event()->alphaQCD() < 0)          event()->set_alphaQCD( pyint1.vint[57] );
      
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
   */

   event()->set_pdf_info( pdf ) ;

   // this is "standard" Py6 event weight (corresponds to PYINT1/VINT(97)
   //
   if (lhe && std::abs(lheRunInfo()->getHEPRUP()->IDWTUP) == 4)
     // translate mb to pb (CMS/Gen "convention" as of May 2009)
     event()->weights().push_back( pyint1.vint[96] * 1.0e9 );
   else
     event()->weights().push_back( pyint1.vint[96] );
   //
   // this is event weight as 1./VINT(99) (PYINT1/VINT(99) is returned by the PYEVWT) 
   //
   event()->weights().push_back( 1./(pyint1.vint[98]) );

   // now create the GenEventInfo product from the GenEvent and fill
   // the missing pieces

   eventInfo().reset( new GenEventInfoProduct( event().get() ) );

   // in Pythia6 pthat is used to subdivide samples into different bins
   // in LHE mode the binning is done by the external ME generator
   // which is likely not pthat, so only filling it for Py6 internal mode
   if (!lhe)
   {
     eventInfo()->setBinningValues( std::vector<double>(1, pypars.pari[16]) );
   }

   // here we treat long-lived particles
   //
   if ( fImposeProperTime || pydat1.mstj[21]==3 || pydat1.mstj[21]==4 ) imposeProperTime();

   // convert particle IDs Py6->PDG, if requested
   if ( fConvertToPDG ) {
      for ( HepMC::GenEvent::particle_iterator part = event()->particles_begin(); 
                                               part != event()->particles_end(); ++part) {
         (*part)->set_pdg_id(HepPID::translatePythiatoPDT((*part)->pdg_id()));
      }
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
   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance

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
      if ( fStopHadronsEnabled ) 
      {
         pystfr_(ierr);
	 if ( ierr != 0 ) // skip failed events
	 {
	    event().reset();
	    return false;
	 }
      }
      if ( fGluinoHadronsEnabled ) pyglfr_();
   }
   
   if ( pyint1.mint[50] != 0 ) // skip event if py6 considers it bad
   {
      event().reset();
      return false;
   }
   
   //formEvent();
   call_pyhepc(1);
   event().reset( conv.read_next_event() );
   
   // tmp stuff to deal with EvtGen confusing pyjets
   //NPartsBeforeDecays = event()->particles_size();
   flushTmpStorage();
   fillTmpStorage();
      
   return true;
}

bool Pythia6Hadronizer::hadronize()
{
   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance

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
      if ( fStopHadronsEnabled ) 
      {
         pystfr_(ierr);
	 if ( ierr != 0 ) // skip failed events
	 {
	    event().reset();
	    return false;
	 }
      }
            
      if ( fGluinoHadronsEnabled ) pyglfr_();
   }

   if ( pyint1.mint[50] != 0 ) // skip event if py6 considers it bad
   {
      event().reset();
      return false;
   }
   
   call_pyhepc(1);
   event().reset( conv.read_next_event() );

   // tmp stuff to deal with EvtGen confusing pyjets
   // NPartsBeforeDecays = event()->particles_size();
   flushTmpStorage();
   fillTmpStorage();
      
   return true;
}

bool Pythia6Hadronizer::decay()
{
   return true;
}

bool Pythia6Hadronizer::residualDecay()
{
   
   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance
   
   if ( pyjets_local.n != pyjets.n )
   {
      // restore pyjets to its state as it was before external decays 
      // - it might have been jammed by py1ent calls in EvtGen
      pyjets.n = pyjets_local.n;
      pyjets.npad = pyjets_local.npad;
      for ( int ip=0; ip<pyjets_local.n; ip++ )
      {
         for ( int i=0; i<5; i++ )
	 {
	    pyjets.k[i][ip] = pyjets_local.k[i][ip];
	    pyjets.p[i][ip] = pyjets_local.p[i][ip];
	    pyjets.v[i][ip] = pyjets_local.v[i][ip];
	 }
      }
   }
   
   // int nDocLines = pypars.msti[3];
   
   // because the counter in HEPEVT might have been reset already, 
   // get the Npart directly from pyjets
   
   // this is currently done (tmp) via setting it as data member,
   // at the end of py6 evfent generation
   int NPartsBeforeDecays = pyjets.n ;
   int NPartsAfterDecays = event()->particles_size();
   
   std::vector<int> part_idx_to_decay;
   part_idx_to_decay.clear(); // just a safety measure, shouldn't be necessary but...
   
   //
   // here put additional info back to pyjets BY HANDS
   //
   
   for ( int ipart=NPartsBeforeDecays+1; ipart<=NPartsAfterDecays; ipart++ )
   {
      HepMC::GenParticle* part = event()->barcode_to_particle( ipart );
      int status = part->status();
      int pdgid = part->pdg_id();
      // add part to pyjets, with proper links/pointers
      if ( 1 == status )
      {
         pyjets.k[0][ipart-1] = 1;
      }
      else if ( 2 == status )
      {
         pyjets.k[0][ipart-1] = 11;
      }
      else if ( 3 == status )
      {
         pyjets.k[0][ipart-1] = 21;
      }
      int py6id = pycomp_( pdgid );
      pyjets.k[1][ipart-1] = pdgid;
      HepMC::GenVertex* prod_vtx = part->production_vertex();
      // in principle, this should never happen but...
      if ( ! prod_vtx ) continue;
      // in principle, it should always be 1 but...EvtGen confuses things sometimes
      //assert ( prod_vtx->particles_in_size() == 1 );
      if ( prod_vtx->particles_in_size() != 1 ) continue;
      HepMC::GenParticle* mother = (*prod_vtx->particles_in_const_begin());      
      int mother_id = mother->barcode();
      pyjets.k[2][ipart-1] = mother_id;
      //
      // here also reset status & dauthters for this mother, if needs be
      //
      if ( mother->end_vertex() )
      {
         if ( pyjets.k[0][mother_id-1] >= 1 && pyjets.k[0][mother_id-1] <= 10 )  pyjets.k[0][mother_id-1] = 11;
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
   
   // FIXME:
   // the if-statement below is an extra protection
   // put in place following problem report from Roberto Covarelli, 
   // that some events were inconsistent and even crashed (rarely)
   // in principle, it should NOT matter but...
   //
   if ( part_idx_to_decay.size() > 0 )
   {
      for ( size_t ip=0; ip<part_idx_to_decay.size(); ip++ )
      {         
         pydecy_(part_idx_to_decay[ip]);
      }
        
      call_pyhepc(1);
   
      event().reset( conv.read_next_event() );
   }
   
   return true;
}

bool Pythia6Hadronizer::initializeForExternalPartons()
{
   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance

   // note: CSA mode is NOT supposed to woirk with external partons !!!
   
   fPy6Service->setGeneralParams();
   fPy6Service->setPYUPDAParams(false);

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

   fPy6Service->setPYUPDAParams(true);

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
   Pythia6Service::InstanceWrapper guard(fPy6Service);	// grab Py6 instance
    
   fPy6Service->setGeneralParams();   
   fPy6Service->setCSAParams();
   fPy6Service->setSLHAParams();
   fPy6Service->setPYUPDAParams(false);
   
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

   fPy6Service->setPYUPDAParams(true);
   
   fPy6Service->closeSLHA();
   
   return true;
}

bool Pythia6Hadronizer::declareStableParticles( std::vector<int> pdg )
{
   
   for ( size_t i=0; i<pdg.size(); i++ )
   {
      int PyID = HepPID::translatePDTtoPythia( pdg[i] );
      // int PyID = pdg[i]; 
      int pyCode = pycomp_( PyID );
      if ( pyCode > 0 )
      {
         std::ostringstream pyCard ;
         pyCard << "MDCY(" << pyCode << ",1)=0";
/* this is a test printout... 
         std::cout << "pdg= " << pdg[i] << " " << pyCard.str() << std::endl; 
*/
         call_pygive( pyCard.str() );
      }
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
            double unif_rand = fPy6Service->call(pyr_, &dumm);
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
     cs *= 1.0e9; // translate to pb (CMS/Gen "convention" as of May 2009)
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
