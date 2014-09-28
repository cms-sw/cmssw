#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HepMC/HEPEVT_Wrapper.h"

extern "C" {
// this is patchup for Py6 common block because 
// several elements of the VINT array are used in the matching process

    extern struct {
	int mint[400];
	double vint[400];
    } pyint1_;

}


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
      
   // here we'll have to adjust, if needed, for "massless" particles
   // from earlier Madgraph version(s)
   // also, we'll have to setup elements of the Py6 fortran array 
   // VINT(357), VINT(358), VINT(360) and VINT(390)
   // if ( fJetMatching->getMatchingScheme() == "Madgraph" )
   // {
   //    
   // }
   
   fJetMatching->beforeHadronisationExec();  
   
   return;   

}

bool JetMatchingHook::doVetoPartonLevel( const Event& event )
{
                  
   // extract "hardest" event - the output will go into workEvent, 
   // which is a data mamber of base class UserHooks
   //
   subEvent(event,true);

   setHEPEVT( event ); // here we pass in "full" event, not workEvent !

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

void JetMatchingHook::setHEPEVT( const Event& event )
{
           
   HepMC::HEPEVT_Wrapper::zero_everything();   
      
   // service container for further mother-daughters links
   //
   std::vector<int> Py8PartonIdx; // position of original (LHE) partons in Py8::Event
   Py8PartonIdx.clear(); 
   std::vector<int> HEPEVTPartonIdx; // position of LHE partons in HEPEVT (incl. ME-generated decays)
   HEPEVTPartonIdx.clear(); 

   // general counter
   //
   int index = 0;

   int Py8PartonCounter = 0;
   int HEPEVTPartonCounter = 0;
   
   // find the fisrt parton that comes from LHE (ME-generated)
   // skip the incoming particles/partons
   for ( int iprt=1; iprt<event.size(); iprt++ )
   {
      const Particle& part = event[iprt];
      if ( abs(part.status()) < 22 ) continue; // below 10 is "service"
                                               // 11-19 are beam particles; below 10 is "service"
					       // 21 is incoming partons      
      Py8PartonCounter = iprt;
      break;
   }

   const lhef::HEPEUP& hepeup = *fEventBlock->getHEPEUP();
   // start the counter from 2, because we don't want the incoming particles/oartons !
   for ( int iprt=2; iprt<hepeup.NUP; iprt++ )
   {
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, hepeup.IDUP[iprt] );
      HepMC::HEPEVT_Wrapper::set_status( index, 2 );
      HepMC::HEPEVT_Wrapper::set_momentum( index, hepeup.PUP[iprt][0], hepeup.PUP[iprt][1], hepeup.PUP[iprt][2], hepeup.PUP[iprt][4] );
      HepMC::HEPEVT_Wrapper::set_mass( index, hepeup.PUP[iprt][4] );
      // --> FIXME HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); // NO, not anymore to the "system particle"
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 ); 
      if (  hepeup.MOTHUP[iprt].first > 2 && hepeup.MOTHUP[iprt].second > 2 ) // decay from LHE, will NOT show at the start of Py8 event !!!
      {
         HEPEVTPartonCounter++;
	 continue;
      }
      Py8PartonIdx.push_back( Py8PartonCounter );
      Py8PartonCounter++;
      HEPEVTPartonIdx.push_back( HEPEVTPartonCounter);
      HEPEVTPartonCounter++;   
   }
      
   HepMC::HEPEVT_Wrapper::set_number_entries( index );   
         
   // now that the initial partons are in, attach parton-level from Pythia8
   // do NOT reset index as we need to *add* more particles sequentially
   //
   for ( int iprt=1; iprt<workEvent.size(); iprt++ ) // from 0-entry (system) or from 1st entry ???
   {
   
      const Particle& part = workEvent[iprt];
      

      if ( part.status() != 62 ) continue;
                  
      index++;
      HepMC::HEPEVT_Wrapper::set_id( index, part.id() );
      
      // HepMC::HEPEVT_Wrapper::set_status( index, event.statusHepMC(iprt) ); 
      HepMC::HEPEVT_Wrapper::set_status( index, 1 );      
      HepMC::HEPEVT_Wrapper::set_momentum( index, part.px(), part.py(), part.pz(), part.e() );
      HepMC::HEPEVT_Wrapper::set_mass( index, part.m() );
      HepMC::HEPEVT_Wrapper::set_position( index, part.xProd(), part.yProd(), part.zProd(), part.tProd() );
      HepMC::HEPEVT_Wrapper::set_parents( index, 0, 0 ); // just set to 0 like in Py6...
                                                         // although for some, mother will need to be re-set properly !
      HepMC::HEPEVT_Wrapper::set_children( index, 0, 0 );

      // now refine mother-daughters links, where applicable
      
      int parentId = getAncestor( part.daughter1(), event );
      
      if ( parentId <= 0 ) continue;

      for ( int idx=0; idx<(int)Py8PartonIdx.size(); idx++ )
      {
         if ( parentId == Py8PartonIdx[idx] )
	 {
            int idx1 = HEPEVTPartonIdx[idx]; 
	    HepMC::HEPEVT_Wrapper::set_parents( index, idx1+1, idx1+1 ); 
	    break;
	 }
      }

   }  
     
   HepMC::HEPEVT_Wrapper::set_number_entries( index );
   
   HepMC::HEPEVT_Wrapper::set_event_number( fEventNumber ); // well, if you know it... well, it's one of the counters...
   
//   HepMC::HEPEVT_Wrapper::print_hepevt();
   
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
