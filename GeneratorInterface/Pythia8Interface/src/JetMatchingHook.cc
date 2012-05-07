#include "GeneratorInterface/Pythia8Interface/interface/JetMatchingHook.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HepMC/HEPEVT_Wrapper.h"

using namespace gen;
using namespace Pythia8;


JetMatchingHook::JetMatchingHook( const edm::ParameterSet& ps, Info* info )
   : UserHooks(), fRunBlock(0), fEventBlock(0), fEventNumber(0), fInfoPtr(info), fJetMatching(0)
{

   assert( fInfoPtr );
   
   std::string scheme = ps.getParameter<std::string>("scheme");

   if ( scheme == "Madgraph" )
   {
      fJetMatching = new JetMatchingMadgraph(ps);
   }
   else if ( scheme == "MLM" || scheme == "Alpgen" )
   {
	throw cms::Exception("JetMatching")
	<< "Port of " << scheme << "scheme \"" << "\""
	" for parton-shower matching is still in progress."
	<< std::endl;
   }
   else
      throw cms::Exception("InvalidJetMatching")
      << "Unknown scheme \"" << scheme << "\""
      " specified for parton-shower matching."
      << std::endl;
 
}

JetMatchingHook::~JetMatchingHook()
{
   if ( fJetMatching ) delete fJetMatching;
}

void JetMatchingHook::init ( lhef::LHERunInfo* runInfo )
{

   setLHERunInfo( runInfo );
   if ( !fRunBlock )
   {
      throw cms::Exception("JetMatching")
      << "Invalid RunInfo" << std::endl;
   
   }
   fJetMatching->init( runInfo );
   return; 

}

void JetMatchingHook::beforeHadronization( lhef::LHEEvent* lhee )
{

   setLHEEvent( lhee );   
   fJetMatching->beforeHadronisation( lhee );
      
   fJetMatching->beforeHadronisationExec();  

   return;   

}

bool JetMatchingHook::doVetoPartonLevel( const Event& event )
{
            
   omitResonanceDecays(event); 
   
   subEvent(event,true);
   setHEPEVT( workEvent );
   
   if ( !hepeup_.nup || fJetMatching->isMatchingDone() )
   {
      return true;
   }
   
   // Note from JY:
   // match(...)input agrs here are reserved for future development and are irrelevat at this point
   // just for future references, they're: 
   // const HepMC::GenEvent* partonLevel,
   // const HepMC::GenEvent *finalState,
   // bool showeredFinalState
   //
   bool jmtch = fJetMatching->match( 0, 0, true ); // true if veto-ed, false if accepted (not veto-ed)
   if ( jmtch )
   {
      return true;
   }
         
   // Do not veto events that got this far
   //
   return false;

}

void JetMatchingHook::setHEPEVT( const Event& event )
{

   int index = 0;
   for ( int iprt=1; iprt<event.size(); iprt++ ) // from 0-entry (system) or from 1st entry ???
   {
   
      const Particle& part = event[iprt];
      if ( part.status() != 62 ) continue;
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
      HepMC::HEPEVT_Wrapper::set_status( index, 1 ); 
      //
      // Please note that we do NOT boost along Z (unlike in Py6) because we get to matching 
      // later in the event development, so boost isn't necessary
      //
      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), part.pz(), part.e() );
      HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
      HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      //
      // at this point we do NOT attepmt to figure out and set mother-daughter links (although in Py we do)
      //
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); 
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 );
      
   }   

   HepMC::HEPEVT_Wrapper::set_number_entries( index );
   
   HepMC::HEPEVT_Wrapper::set_event_number( fEventNumber ); // well, if you know it... well, it's one of the counters...
   
   // HepMC::HEPEVT_Wrapper::print_hepevt();

   return;

}

/* Oct.2011 - Note from JY:
   This is an attempt to mimic Py6 routine PYVETO, including boost along Z and mother-daughter links.
   It's unclear if we'll ever need thsoe details, but for now I keep the commented code here, just in case...

void JetMatchingHook::setHEPEVT( const Event& event )
{

   HepMC::HEPEVT_Wrapper::zero_everything();
   
   int index = 0;
   std::vector<int> IndexContainer;
   IndexContainer.clear();
   int status = 0;
   for ( int iprt=1; iprt<event.size(); iprt++ ) // from 0-entry (system) or from 1st entry ???
   {
   
      const Particle& part = event[iprt];
      
      if ( abs(part.status()) < 22 ) continue; // below 10 is "service"
                                               // 11-19 are beam particles; below 10 is "service"
					       // 21 is incoming partons
      if ( part.status() < -23 ) continue; // intermediate steps in the event development 
                                           // BUT already decayed/branched/fragmented/...
      // so we get here if status=+/-22 or +/-23, or remaining particles with status >30 
      //
      IndexContainer.push_back( iprt );
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
      // HepMC::HEPEVT_Wrapper::set_status( index, event.statusHepMC(iprt) ); 
      status = 1;
      if (abs(part.status()) == 22 || abs(part.status()) == 23 )
      {
         // HepMC::HEPEVT_Wrapper::set_status( index, 2 );
	 status = 2;
      }
      HepMC::HEPEVT_Wrapper::set_status( index, status );
      // needs to be boosted along z-axis !!!
      // this is from Py6/pyveto code:
      // C...Define longitudinal boost from initiator rest frame to cm frame
      // - need to replicate !!!
      // GAMMA=0.5D0*(VINT(141)+VINT(142))/SQRT(VINT(141)*VINT(142))
      // GABEZ=0.5D0*(VINT(141)-VINT(142))/SQRT(VINT(141)*VINT(142))
      double x1 = fInfoPtr->x1();
      double x2 = fInfoPtr->x2();
      double dot = x1*x2;
      assert( dot );
      double gamma = 0.5 * ( x1+x2 ) / std::sqrt( dot );
      double gabez = 0.5 * ( x1-x2 ) / std::sqrt( dot );
      double pz = gamma*part.pz() + gabez*part.e();
      double e  = gamma*part.e()  + gabez*part.pz();
      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), pz, e );
      HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
      HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); // for some, mother will need to be re-set properly !
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 );
      if ( status == 2 ) continue; 
      int parentId = part.mother1();
      int parentPrevId = 0;
      
      while ( parentId > 0 )
      {               
         if ( parentId == part.mother2() ) // carbon copy
	 {
	    parentPrevId = parentId;
	    parentId = event[parentPrevId].mother1();
	    continue;
	 }
	 
	 // we get here if not a carbon copy, but a result of a process
	 if ( parentId < parentPrevId ) // normal process
	 {
	    parentPrevId = parentId;
	    parentId = event[parentPrevId].mother1();
	 }
	 else if ( parentId > parentPrevId || parentId > iprt ) // "circular"/"forward-pointing parent" - intermediate process
	 {
	    parentId = -1;
	    break;
	 }
	 
	 if ( parentId < part.mother2() ) // normal process
	 {
	    parentPrevId = parentId;
	    parentId = event[parentPrevId].mother1();
	 }
	 
         if ( abs(event[parentId].status()) == 22 || abs(event[parentId].status())== 23 ) // hard block
	 {
	    break;
	 } 
	 
	 if ( abs(event[parentId].status()) < 22 ) // incoming
	 {
	    parentId = -1;
	    break;
	 } 
      }

      if ( parentId > 0 )
      {
         for ( int idx=0; idx<(int)IndexContainer.size(); idx++ )
         {
            if ( parentId == IndexContainer[idx] )
	    {
	       HepMC::HEPEVT_Wrapper::set_parents( index, idx+1, idx+1 ); // probably need to check status of index-particle in HEPEVT - has to be 2 !!!
	       break;
	    }
         }
      }
        
   }
   
   HepMC::HEPEVT_Wrapper::set_number_entries( index );
   
   HepMC::HEPEVT_Wrapper::set_event_number( fEventNumber ); // well, if you know it... well, it's one of the counters...
   
   HepMC::HEPEVT_Wrapper::print_hepevt();
   
   return;

}
*/
