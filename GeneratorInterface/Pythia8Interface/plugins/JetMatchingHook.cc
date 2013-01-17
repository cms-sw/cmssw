#include "GeneratorInterface/Pythia8Interface/plugins/JetMatchingHook.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"

#include "FWCore/Utilities/interface/Exception.h"

//#include "HepMC/HEPEVT_Wrapper.h"
#include <cassert>

#include "GeneratorInterface/Pythia8Interface/plugins/Py8toJetInput.h"

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
   : UserHooks(), fRunBlock(0), fEventBlock(0), fEventNumber(0), fInfoPtr(info), 
     fJetMatching(0), fJetInputFill(0),
     fIsInitialized(false)
{

   assert( fInfoPtr );
   
   std::string scheme = ps.getParameter<std::string>("scheme");

   if ( scheme == "Madgraph" )
   {
      fJetMatching = new JetMatchingMadgraph(ps);
      fJetInputFill = new Py8toJetInputHEPEVT();
   }
   else if ( scheme == "MadgraphFastJet" )
   {
      fJetMatching = new JetMatchingMG5(ps);
      fJetInputFill = new Py8toJetInput();
   }
   else if ( scheme == "MadgraphSlowJet" )
   {
      fJetMatching = new JetMatchingMG5(ps);
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
   double etaMax = fJetMatching->getJetEtaMax();
   fJetInputFill->setJetEtaMax( etaMax );
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

bool 
JetMatchingHook::doVetoPartonLevel( const Event& event )
// JetMatchingHook::doVetoPartonLevelEarly( const Event& event )
{
                  
   // event.list();
   
   // extract "hardest" event - the output will go into workEvent, 
   // which is a data mamber of base class UserHooks
   //
   subEvent(event,true);
      
   if ( !hepeup_.nup || fJetMatching->isMatchingDone() )
   {
      return true;
   }
   
   //
   // bool jmtch = fJetMatching->match( 0, 0, true ); // true if veto-ed, false if accepted (not veto-ed)
   std::vector<fastjet::PseudoJet> jetInput = fJetInputFill->fillJetAlgoInput( event, workEvent,
                                                                               fEventBlock,
									       fJetMatching->getPartonList() );
   bool jmtch = fJetMatching->match( fEventBlock, &jetInput );
   if ( jmtch )
   {
      return true;
   }
         
   // Do not veto events that got this far
   //
   return false;

}

/*
void JetMatchingHook::setJetAlgoInput( const Event& event )
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
      

//      if ( part.status() != 62 ) continue;
      if ( part.status() < 51 ) continue;
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
*/

#include <iomanip>
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
//#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
//#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
//#include "GeneratorInterface/Pythia8Interface/interface/JetMatchingMG5.h"

namespace gen
{ 

/*
extern "C" {

	extern struct UPPRIV {
		int	lnhin, lnhout;
		int	mscal, ievnt;
		int	ickkw, iscale;
	} uppriv_;

	extern struct MEMAIN {
		double 	etcjet, rclmax, etaclmax, qcut, showerkt, clfact;
		int	maxjets, minjets, iexcfile, ktsche;
		int	mektsc,nexcres, excres[30];
      		int	nqmatch,excproc,iexcproc[1000],iexcval[1000];
                bool    nosingrad,jetprocs;
	} memain_;

}
*/

bool JetMatchingMG5::initAfterBeams()
{

      doMerge        = uppriv_.ickkw;
      qCut           = memain_.qcut;
      nQmatch        = memain_.nqmatch;
      clFact         = 0;


  // Read in parameters
  nJet           = memain_.minjets;
  nJetMax        = memain_.maxjets;

  // Now Jet Algorithm -
  // for now, it's hardcoded as the kT algorithm !!!
  // (in SlowJets it's defined by the slowJetPower=1)
  //
  jetAlgorithm   = 2;
  etaJetMax      = memain_.etaclmax;
  coneRadius     = 1.0;
  slowJetPower   = 1;
  eTjetMin       = 20;


  // Matching procedure
  jetMatch = 0;
  exclusiveMode = 1;
  ktScheme       = memain_.mektsc;

  qCutSq         = pow(qCut,2);
  etaJetMaxAlgo  = etaJetMax;


  // If not merging, then done
  if (!doMerge) return true;

  // Exclusive mode; if set to 2, then set based on nJet/nJetMax
  if (exclusiveMode == 2) 
  {

    // No nJet or nJetMax, so default to exclusive mode
    if (nJet < 0 || nJetMax < 0) 
    {

      exclusive = true;

    // Inclusive if nJet == nJetMax, exclusive otherwise
    } 
    else 
    {
      exclusive = (nJet == nJetMax) ? false : true;
    }

  // Otherwise, just set as given
  } 
  else 
  {
    exclusive = (exclusiveMode == 0) ? false : true;
  }

  // FIXME !!!
  // (commented out) usage of CellJet and SlowJet is from Steve's ofiginal example, 
  // written for native Py8 - later on, I'll have to implement similar options in
  // FastJet terms, based on proper settings !!!
  //
  // Initialise chosen jet algorithm: CellJet.
  if (jetAlgorithm == 1) 
  {

    // Extra options for CellJet. nSel = 1 means that all final-state
    // particles are taken and we retain control of what to select.
    // smear/resolution/upperCut are not used and are set to default values.
//    int    nSel = 2, smear = 0;
//    double resolution = 0.5, upperCut = 2.;
//    cellJet = new CellJet(etaJetMaxAlgo, nEta, nPhi, nSel,
//                          smear, resolution, upperCut, eTthreshold);

  // SlowJet
  } 
  else if (jetAlgorithm == 2) 
  {

    // this is basically the MadGraph one !
    //
//    slowJet = new SlowJet(slowJetPower, coneRadius, eTjetMin, etaJetMaxAlgo);
    //
    //   in principle, we can even use "power" as the 1st input arg,
    //   because kt_algorithm corresponds to slowJetPower=1
    //
    fJetFinder = new fastjet::JetDefinition( fastjet::kt_algorithm, coneRadius );
    fInclusiveJets.clear();
    fExclusiveJets.clear();
    fPtSortedJets.clear();
        
  }

  // Check the jetMatch parameter; option 2 only works with SlowJet
  if (jetAlgorithm == 1 && jetMatch == 2) 
  {
    // FIXME !
    // Will need to print a warning of override here !
    //
    jetMatch = 1;
  }

  // Print information
  std::string jetStr  = (jetAlgorithm ==  1) ? "CellJet" :
                   (slowJetPower == -1) ? "anti-kT" :
                   (slowJetPower ==  0) ? "C/A"     :
                   (slowJetPower ==  1) ? "kT"      : "unknown";
  std::string modeStr = (exclusive)         ? "exclusive" : "inclusive";

  std::cout << std::endl
       << " *-------  MG5 matching parameters  -------*" << std::endl
       << " |  qCut                |  " << std::setw(14)
       << qCut << "  |" << std::endl
       << " |  nQmatch             |  " << std::setw(14)
       << nQmatch << "  |" << std::endl
       << " |  clFact              |  " << std::setw(14)
       << clFact << "  |" << std::endl
       << " |  Jet algorithm       |  " << std::setw(14)
       << jetStr << "  |" << std::endl
       << " |  eTjetMin            |  " << std::setw(14)
       << eTjetMin << "  |" << std::endl
       << " |  etaJetMax           |  " << std::setw(14)
       << etaJetMax << "  |" << std::endl
       << " |  jetAllow            |  " << std::setw(14)
       << jetAllow << "  |" << std::endl
       << " |  jetMatch            |  " << std::setw(14)
       << jetMatch << "  |" << std::endl
       << " |  Mode                |  " << std::setw(14)
       << modeStr << "  |" << std::endl
       << " *-----------------------------------------*" << std::endl;   

   
   
   return true;

}

void JetMatchingMG5::beforeHadronisation( const lhef::LHEEvent* lhee )
{

   if (!runInitialized)
		throw cms::Exception("Generator|PartonShowerVeto")
			<< "Run not initialized in JetMatchingMadgraph"
			<< std::endl;
   
   for (int i = 0; i < 3; i++) 
   {
    typeIdx[i].clear();
   }

   // Sort original process final state into light/heavy jets and 'other'.
   // Criteria:
   //   1 <= ID <= 5 and massless, or ID == 21 --> light jet (typeIdx[0])
   //   4 <= ID <= 6 and massive               --> heavy jet (typeIdx[1])
   //   All else                               --> other     (typeIdx[2])
   //
   const lhef::HEPEUP& hepeup = *lhee->getHEPEUP(); 
   int idx = 2; 
   for ( int i=0; i < hepeup.NUP; i++ )
   {
      if ( hepeup.ISTUP[i] < 0 ) continue;
      if ( hepeup.MOTHUP[i].first != 1 && hepeup.MOTHUP[i].second !=2 ) continue; // this way we skip entries that come
                                                                                  // from resonance decays;
										  // we only take those that descent
										  // directly from "incoming partons"
      idx = 2;
      if ( hepeup.IDUP[i] == ID_GLUON || (fabs(hepeup.IDUP[i]) <= nQmatch) ) // light jet
         // light jet
	 idx = 0;
      else if ( fabs(hepeup.IDUP[i]) > nQmatch && fabs(hepeup.IDUP[i]) <= ID_TOP) // heavy jet
         idx = 1;
      // Store
      typeIdx[idx].push_back(i); 
   } 
      
  return;

}

int JetMatchingMG5::match( const lhef::LHEEvent* partonLevel, 
                           const std::vector<fastjet::PseudoJet>* jetInput )
{
      
   runJetAlgo( jetInput );
   
   // Number of hard partons
   //
   int nPartons = typeIdx[0].size();
   
   int nCLjets = fExclusiveJets.size();
   
   if ( nCLjets < nPartons ) return LESS_JETS;
   
   double localQcutSq = qCutSq;
   
   if ( exclusive ) 
   {
      if( nCLjets > nPartons ) return MORE_JETS;
   }
   else // inclusive ???
   {
      // FIXME !!!
   }
   
  if( clFact != 0 ) localQcutSq *= pow(clFact,2);
  
  std::vector<fastjet::PseudoJet> fMatchingInput;
  
  std::vector<bool> jetAssigned;
  jetAssigned.assign( fExclusiveJets.size(), false );

  int iNow = 0;
  
  const lhef::HEPEUP& hepeup = *partonLevel->getHEPEUP();
  
/* --->
  int proc = hepeup.IDPRUP;
  
  if ( proc == 2 || proc == 3 )
  {
     std::cout << " process = " << proc << std::endl;
  }
*/
  
  while ( iNow < nPartons )
  {

     fMatchingInput.clear();

     for ( int i=0; i<nCLjets; i++ )
     {
        if ( jetAssigned[i] ) continue;
        if ( i == nPartons )  break;
	//
	// this looks "awkward" but this way we do NOT pass in cluster_hist_index
	// which would actually indicate position in "history" of the "master" ClusSeq
	//
	fastjet::PseudoJet exjet = fExclusiveJets[i];
        fMatchingInput.push_back( fastjet::PseudoJet( exjet.px(), exjet.py(), exjet.pz(), exjet.e() ) );
	fMatchingInput.back().set_user_index(i);
     }

     int idx = typeIdx[0][iNow];
     fMatchingInput.push_back( fastjet::PseudoJet( hepeup.PUP[idx][0],
                                                   hepeup.PUP[idx][1],
						   hepeup.PUP[idx][2],
						   hepeup.PUP[idx][3]) );

     //
     // in principle, one can use ClusterSequence::n_particles() 
     //
     int nBefore = fMatchingInput.size();

     // Create new clustering object - which includes the 1st clustering run !!!
     // NOTE-1: it better be a new object for each try, or history will be "too crowded".
     // NOTE-2: when created, the it ALWAYS makes at least 1 clustering step, so in most
     //         cases at least 1 new jet object will be added; although in some cases 
     //         no jet is added, if the system doesn't cluster to anything (for example,
     //         input jet(s) and parton(s) are going opposite directions)
     //
     fastjet::ClusterSequence ClusSeq( fMatchingInput, *fJetFinder );
     
     const std::vector<fastjet::PseudoJet>&        output = ClusSeq.jets();
     int nClusJets = output.size() - nBefore;
     
     // JVY:
     // I think the right idea would be this:
     // at least 1 (one) new jet needs to be added
     //
     // 
     if ( nClusJets < 1 )
     {
        return UNMATCHED_PARTON;
     }
     //
     // very unlikely case but let's do it just to be safe
     //
     if ( nClusJets >= nBefore )
     {
        return MORE_JETS;
     }
      
     // Now browse history and see how close the clustering distance
     //
     // NOTE: Remember, there maybe more than one new jet in the list (for example,
     //       for process=2,3,...);
     //       in this case we take the ** first ** one that satisfies the distance/cut,
     //       which is ** typically ** also the best one 
     //
     bool matched = false;
     const std::vector<fastjet::ClusterSequence::history_element>&  history = ClusSeq.history();
              
     // for ( unsigned int i=nBefore; i<history.size(); i++ )
     for ( unsigned int i=nBefore; i<output.size(); i++ )
     {

        int hidx = output[i].cluster_sequence_history_index();
	double dNext = history[hidx].dij;
	if ( dNext < localQcutSq )
	{
	   //
	   // the way we form input, parent1 is always jet, and parent2 can be any,
	   // but if it's a good match/cluster, it should point at the parton
	   //
	   int parent1 = history[hidx].parent1;
	   int parent2 = history[hidx].parent2;
	   if ( parent1 < 0 || parent2 < 0 ) break; // bad cluster, no match
	   //
	   // pull up jet's "global" index 
	   //
	   int pidx = fMatchingInput[parent1].user_index();
	   jetAssigned[pidx] = true; 
	   matched = true;
	   break;
	}
     }
     if ( !matched )
     {
        return UNMATCHED_PARTON;
     }
     
     iNow++;   
  }
  
  return NONE;

}

void JetMatchingMG5::runJetAlgo(const std::vector<fastjet::PseudoJet>* jetInput )
{

   // --> FIXME !!!
   // in principle, this needs to be done based on exclusive/inclusive settings
   
   fInclusiveJets.clear();
   fExclusiveJets.clear();
   
   // clustering algorithm 
   // NOTE: It includes not only init, but it ALSO RUNS !!!
   //       The list of jets will include the initial input AND the reclustered jets. 
   //
   fastjet::ClusterSequence ClusSeq( *jetInput, *fJetFinder );
         
   // fInclusiveJets = ClusSeq.inclusive_jets(); // in principle, it has input argument ptmin (D=0)
   fExclusiveJets = ClusSeq.exclusive_jets( qCutSq ); 
   
   return;

}

} // end namespace
