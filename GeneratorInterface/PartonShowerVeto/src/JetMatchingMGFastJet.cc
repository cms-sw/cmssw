
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMGFastJet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iomanip>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

namespace gen
{ 

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

bool JetMatchingMGFastJet::initAfterBeams()
{
  
  if ( fIsInit ) return true;
  
  //
  doMerge = uppriv_.ickkw; 
  // doMerge = true;
  qCut    = memain_.qcut; // note: main32 overrides it from LHE but as XQCUT !!!
  nQmatch        = memain_.nqmatch;
  clFact         = 1.; // Default value specified in JetMatchningStyle.xml (py8.173-cand)
                      // NOTE: ME2pythia seems to default to 1.5 - need to check !!!
		      // in general, needs to read key ALPSFACT from LHE file - fix CMSSW code !!!
  nJetMin        = memain_.minjets;
  nJetMax        = memain_.maxjets;



  // Read in parameters
  nJetMin        = memain_.minjets;
  nJetMax        = memain_.maxjets;

  //
  // this is the kT algorithm !!!
  // (in SlowJets it's defined by the slowJetPower=1)
  //
  jetAlgorithm   = 2; // Default (in py8.173-cand, (xmldoc); settingsPtr->mode("JetMatching:jetAlgorithm"); )
                      // in fact, this MUST be set to 2, as it means Madgraph matching
  etaJetMax      = memain_.etaclmax; // note: main32 does NOT pick it from LHE !!!
  coneRadius     = 1.0; // Default (in py8-173-cand (xmldoc); settingsPtr->parm("JetMatching:coneRadius"); )
  slowJetPower   = 1;   // this meant kT algorithm (echos jetAlgoritm=2 ???); it MUST be set to 1.
  // slowJetPower = -1; // Default (py8-173-cand; xmldoc/JetMatchingStyle.xml)
  eTjetMin       = 20;  // Defualt (py8.173-cand, (xmldoc); settingsPtr->parm("JetMatching:etaJetMax"); )




  // Matching procedure

  // Matching procedure
  jetAllow = 1; // Default value specifid in JetMatchingStyle.xml (py8.173-cand; settingsPtr->mode("JetMatching:jetAllow"); )
  exclusiveMode = 2; // Default value specified in JetMatchingStyle.xml (py8.173-cand; settingsPtr->mode("JetMatching:exclusive"); )
  qCutSq         = pow(qCut,2);
  etaJetMaxAlgo  = etaJetMax;
  
  // this should be something like memaev_.iexc
  fExcLocal = true;

  // ktScheme       = memain_.mektsc;

  qCutSq         = pow(qCut,2);
  etaJetMaxAlgo  = etaJetMax;


  // If not merging, then done
  if (!doMerge) return true;

  // Exclusive mode; if set to 2, then set based on nJet/nJetMax
  //

  if (exclusiveMode == 2) {

    // No nJet or nJetMax, so default to exclusive mode
    if (nJetMax < 0) {
      //infoPtr->errorMsg("Warning in py8base::JetMatchingMadgraph:init: "
      //    "missing jet multiplicity information; running in exclusive mode");
      exclusiveMode = 1;
    }
  }
  
  // Initialise chosen jet algorithm. 
  //
/*  I think this is not even necessary as jetAlgorithm MUST be set to 2 for Madgraph
  if (jetAlgorithm != 2) {
      //infoPtr->errorMsg("Warning in py8base::JetMatchingMadgraph:init: "
	//		"CellJet not supported.  Setting jet algorithm to SlowJet");
      jetAlgorithm = 2;
  }
*/  
  // Initialise chosen jet algorithm. SlowJet.
  //
  if (jetAlgorithm == 2) {
/* I think slowJetPower MUST be set to 1, and no overrides !
    if(slowJetPower != 1) {
      //infoPtr->errorMsg("Warning in MatchingMadgraph:init: "
	//		"anti-kT and CA not supported.  Setting SlowJet for kT clustering");
      slowJetPower = 1;
    }
*/
    fJetFinder = new fastjet::JetDefinition( fastjet::kt_algorithm, coneRadius );
    fClusJets.clear();
    fPtSortedJets.clear();

  } 

  // Print information
  std::string jetStr  = (jetAlgorithm ==  1) ? "CellJet" :
                        (slowJetPower == -1) ? "anti-kT" :
                        (slowJetPower ==  0) ? "C/A"     :
                        (slowJetPower ==  1) ? "kT"      : "unknown";
  std::string modeStr =      (exclusiveMode)         ? "exclusive" : "inclusive";
  std::cout << std::endl
       << " *-----  Madgraph matching parameters  -----*" << std::endl
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
       << " |  Mode                |  " << std::setw(14)
       << modeStr << "  |" << std::endl
       << " *-----------------------------------------*" << std::endl;
   
   
   fIsInit = true;
   
   return true;

}

void JetMatchingMGFastJet::beforeHadronisation( const lhef::LHEEvent* lhee )
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

   // Exclusive mode; if set to 2, then set based on nJet/nJetMax
   
   // NOTE: In principle, I should use exclusive, inclusive, or soup !!!
   
/*
   if (exclusiveMode == 2) 
   {
      // Inclusive if nJet == nJetMax, exclusive otherwise
      //
      int nParton = typeIdx[0].size();
      fExcLocal = (nParton == nJetMax) ? false : true;
   } 
   else 
   {
       // Otherwise, just set as given
       fExcLocal = (exclusiveMode == 0) ? false : true;
   }
*/   
   // should be like this:

   if ( soup )
   {
      int nParton = typeIdx[0].size();
      fExcLocal = ( nParton < nJetMax );
   }
   else
      fExcLocal = exclusive;
   
      
  return;

}

int JetMatchingMGFastJet::match( const lhef::LHEEvent* partonLevel, 
                           const std::vector<fastjet::PseudoJet>* jetInput )
{
      
   // runJetAlgo( jetInput );
   
   // Number of hard partons
   //
   int NPartons = typeIdx[0].size();
   
   fClusJets.clear();
   fPtSortedJets.clear();
   
   int ClusSeqNJets = 0;

   fastjet::ClusterSequence ClusSequence( *jetInput, *fJetFinder );
   
   if ( fExcLocal )
   {
      fClusJets = ClusSequence.exclusive_jets( qCutSq );
   }
   else
   {
      fClusJets = ClusSequence.inclusive_jets( qCut );
   }
   
   ClusSeqNJets = fClusJets.size(); 
   
   
   if ( ClusSeqNJets < NPartons ) return LESS_JETS;
   
   double localQcutSq = qCutSq;
   
   if ( fExcLocal ) // exclusive
   {
      if( ClusSeqNJets > NPartons ) return MORE_JETS;
   }
   else // inclusive 
   {
      fPtSortedJets = fastjet::sorted_by_pt( *jetInput );
      localQcutSq = std::max( qCutSq, fPtSortedJets[0].pt2() );
      fClusJets = ClusSequence.exclusive_jets( NPartons ); // override
      ClusSeqNJets = NPartons;
   }
   
  if( clFact != 0 ) localQcutSq *= pow(clFact,2);
  
  std::vector<fastjet::PseudoJet> MatchingInput;
  
  std::vector<bool> jetAssigned;
  jetAssigned.assign( fClusJets.size(), false );

  int iNow = 0;
  
  const lhef::HEPEUP& hepeup = *partonLevel->getHEPEUP();
  
/* --->
  int proc = hepeup.IDPRUP;
  
  if ( proc == 2 || proc == 3 )
  {
     std::cout << " process = " << proc << std::endl;
  }
*/
  
  while ( iNow < NPartons )
  {

     MatchingInput.clear();

     for ( int i=0; i<ClusSeqNJets; i++ )
     {
        if ( jetAssigned[i] ) continue;
        if ( i == NPartons )  break;
	//
	// this looks "awkward" but this way we do NOT pass in cluster_hist_index
	// which would actually indicate position in "history" of the "master" ClusSeq
	//
	fastjet::PseudoJet exjet = fClusJets[i];
        MatchingInput.push_back( fastjet::PseudoJet( exjet.px(), exjet.py(), exjet.pz(), exjet.e() ) );
	MatchingInput.back().set_user_index(i);
     }

     int idx = typeIdx[0][iNow];
     MatchingInput.push_back( fastjet::PseudoJet( hepeup.PUP[idx][0],
                                                  hepeup.PUP[idx][1],
						  hepeup.PUP[idx][2],
						  hepeup.PUP[idx][3]) );

     //
     // in principle, one can use ClusterSequence::n_particles() 
     //
     int NBefore = MatchingInput.size();

     // Create new clustering object - which includes the 1st clustering run !!!
     // NOTE-1: it better be a new object for each try, or history will be "too crowded".
     // NOTE-2: when created, the it ALWAYS makes at least 1 clustering step, so in most
     //         cases at least 1 new jet object will be added; although in some cases 
     //         no jet is added, if the system doesn't cluster to anything (for example,
     //         input jet(s) and parton(s) are going opposite directions)
     //
     fastjet::ClusterSequence ClusSeq( MatchingInput, *fJetFinder );
     
     const std::vector<fastjet::PseudoJet>&        output = ClusSeq.jets();
     int NClusJets = output.size() - NBefore;
     
     // JVY:
     // I think the right idea would be this:
     // at least 1 (one) new jet needs to be added
     //
     // 
     if ( NClusJets < 1 )
     {
        return UNMATCHED_PARTON;
     }
     //
     // very unlikely case but let's do it just to be safe
     //
     if ( NClusJets >= NBefore )
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
     for ( unsigned int i=NBefore; i<output.size(); i++ )
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
	   int pidx = MatchingInput[parent1].user_index();
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

   // Now save some info for DJR analysis (if requested).
   // This mimics what is done in ME2pythia.f
   // Basically, NJets and matching scale for these 4 cases:
   // 1->0, 2->1, 3->2, and 4->3
   //
   if ( fDJROutFlag > 0 )
   {   
      std::vector<double> MergingScale;
      MergingScale.clear();
      for ( int nj=0; nj<4; nj++ )
      {
         double dmscale2 = ClusSequence.exclusive_dmerge( nj );
         double dmscale = sqrt( dmscale2 );
         MergingScale.push_back( dmscale );
      }
      fDJROutput.open( "events.tree", std::ios_base::app );  
      double dNJets = (double)ClusSeqNJets;
      fDJROutput << " " << dNJets << " " << MergingScale[0] << " " 
                                         << MergingScale[1] << " "
                                         << MergingScale[2] << " "
				         << MergingScale[3] << std::endl;  
      fDJROutput.close();
   }
 
  return NONE;

}

} // end namespace
