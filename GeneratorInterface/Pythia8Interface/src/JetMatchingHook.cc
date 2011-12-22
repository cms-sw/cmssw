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
            
   // not necessary because later on we select only "stable partons"
   //
   // omitResonanceDecays(event); 
      
   // extract "hardest" event - the output will go into workEvent, 
   // which is a data mamber of base class UserHooks
   //
   subEvent(event,true);

   setHEPEVT( event ); // here we pass in "full" event, not workEvent !
   // setHEPEVT();

   // std::cout << " NPartons= " << hepeup_.nup << std::endl;
   
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

/* earlier, simplified version

void JetMatchingHook::setHEPEVT()
{

   HepMC::HEPEVT_Wrapper::zero_everything();
   
   int index = 0;
   
   for ( int iprt=1; iprt<workEvent.size(); iprt++ ) // from 0-entry (system) or from 1st entry ???
   {
   
      const Particle& part = workEvent[iprt];
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
*/

void JetMatchingHook::setHEPEVT( const Event& event )
{
        
   HepMC::HEPEVT_Wrapper::zero_everything();   
      
   // service container for further mother-daughters links
   //
   std::vector<int> IndexContainer;
   IndexContainer.clear();   

   // fisrt of all, add "system particle" - to avoid 0-mother and prevent hiccups in ME2pythia
   //
   int index = 1;
   const Particle& part = event[0];
   HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
   HepMC::HEPEVT_Wrapper::set_status( index, 2 );
   HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), part.pz(), part.e() );
   HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
   HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
   HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); // for some, mother will need to be re-set properly !
   HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 );      

   // now select the original partons from LHE recpord
   //
   for ( int iprt=1; iprt<event.size(); iprt++ )
   {
      const Particle& part = event[iprt];

      if ( abs(part.status()) < 22 ) continue; // below 10 is "service"
                                               // 11-19 are beam particles; below 10 is "service"
					       // 21 is incoming partons
      if ( abs(part.status()) > 23 ) break; // intermediate steps in the event development 
                                               // BUT already decayed/branched/fragmented/...
					       // Here we have to remember that, in princle, LHE record may contain decayed resonances, 
					       // i.e. the resonance and its decay products, like Z->e+e- or Z->tautau, or the likes,
					       // but in this case Py8 will place decay products much father in the Py8::Event,
					       // although they'll appear with status=23
					       // Thus, if we break on the fisrt record with abs(status)>23, we will NOT take in
					       // decay products, which is what we want
      //
      // so we get here if status=+/-22 or +/-23
      //

      IndexContainer.push_back( iprt );

      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
      // HepMC::HEPEVT_Wrapper::set_status( index, event.statusHepMC(iprt) ); 
      HepMC::HEPEVT_Wrapper::set_status( index, 2 );
      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), part.pz(), part.e() );
      HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
      HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 1, 1 ); // for now, point it back to "system particle"
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 ); 
   }
      
   HepMC::HEPEVT_Wrapper::set_number_entries( index );   
      
   // do NOT reset index as we need to *add* more particles sequentially
   
   // now that the initial partons are in, attach parton-level from Pythia8
   //
   for ( int iprt=1; iprt<workEvent.size(); iprt++ ) // from 0-entry (system) or from 1st entry ???
   {
   
      const Particle& part = workEvent[iprt];
      

      if ( part.status() != 62 ) continue;
                  
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
      
      // HepMC::HEPEVT_Wrapper::set_status( index, event.statusHepMC(iprt) ); 
      HepMC::HEPEVT_Wrapper::set_status( index, 1 );
      
      // it used to need a boost along z-axis !!!
      // this is from Py6/pyveto code:
      //
      // C...Define longitudinal boost from initiator rest frame to cm frame
      // - need to replicate !!!
      // GAMMA=0.5D0*(VINT(141)+VINT(142))/SQRT(VINT(141)*VINT(142))
      // GABEZ=0.5D0*(VINT(141)-VINT(142))/SQRT(VINT(141)*VINT(142))
      //
      // however, since we call it in py8 later in the event than in py6,
      // the boost is no longer necessary...
      //
//      double x1 = fInfoPtr->x1();
//      double x2 = fInfoPtr->x2();
//      double dot = x1*x2;
//      assert( dot );
//      double gamma = 0.5 * ( x1+x2 ) / std::sqrt( dot );
//      double gabez = 0.5 * ( x1-x2 ) / std::sqrt( dot );
//      double pz = gamma*part.pz() + gabez*part.e();
//      double e  = gamma*part.e()  + gabez*part.pz();
//      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), pz, e );
//
      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), part.pz(), part.e() );
      HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
      HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 1, 1 ); // by default, point back to the "system particle:
                                                         // although for some, mother will need to be re-set properly !
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 );

      // now refine mother-daughters links, where applicable
      
      int parentId = getAncestor( part.daughter1(), event );
      
      if ( parentId <= 0 ) continue;

      for ( int idx=0; idx<(int)IndexContainer.size(); idx++ )
      {
         if ( parentId == IndexContainer[idx] )
	 {
	    HepMC::HEPEVT_Wrapper::set_parents( index, idx+2, idx+2 ); 
	    break;
	 }
      }

   }  
     
   HepMC::HEPEVT_Wrapper::set_number_entries( index );
   
   HepMC::HEPEVT_Wrapper::set_event_number( fEventNumber ); // well, if you know it... well, it's one of the counters...
   
   // HepMC::HEPEVT_Wrapper::print_hepevt();
   
   return;

}

int JetMatchingHook::getAncestor( int pos, const Event& fullEvent )
{

   int parentId = fullEvent[pos].mother1();
   int parentPrevId = 0;
   int counter = pos;
   
   while ( parentId > 0 )
   {               
         if ( parentId == fullEvent[counter].mother2() ) // carbon copy, keep walking up
	 {
	    parentPrevId = parentId;
	    counter = parentId;
	    parentId = fullEvent[parentPrevId].mother1();
	    continue;
	 }
	 
	 // we get here if not a carbon copy
	 
	 // let's check if it's a normal process, etc.
	 //
	 if ( (parentId < parentPrevId) || parentId < fullEvent[counter].mother2() ) // normal process
	 {
	    
	    // first of all, check if hard block
	    if ( abs(fullEvent[counter].status()) == 22 || abs(fullEvent[counter].status()) == 23 )
	    {
	       // yes, it's the hard block
	       // we got what we want, and can exit now !
	       parentId = counter;
	       break;
	    }
	    else
	    {
	       parentPrevId = parentId;
	       parentId = fullEvent[parentPrevId].mother1();
	    }
	 }
	 else if ( parentId > parentPrevId || parentId > pos ) // "circular"/"forward-pointing parent" - intermediate process
	 {
	    parentId = -1;
	    break;
	 }

         // additional checks... although we shouldn't be geeting here all that much...
	 //	 
	 if ( abs(fullEvent[parentId].status()) == 22 || abs(fullEvent[parentId].status())== 23 ) // hard block
	 {
	    break;
	 } 	 
	 if ( abs(fullEvent[parentId].status()) < 22 ) // incoming
	 {
	    parentId = -1;
	    break;
	 } 
   }
   
   return parentId;

}
