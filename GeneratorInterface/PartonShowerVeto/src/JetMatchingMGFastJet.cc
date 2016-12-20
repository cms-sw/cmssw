#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMGFastJet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iomanip>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/trim.hpp>


namespace gen
{ 

extern "C" {

        #define PARAMLEN 20
        namespace {
                struct Param {
                        Param(const std::string &str)
                        {
                                int len = std::min(PARAMLEN,
                                                   (int)str.length());
                                std::memcpy(value, str.c_str(), len);
                                std::memset(value + len, ' ', PARAMLEN - len);
                        }

                        char value[PARAMLEN];
                };
        }

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

        extern struct OUTTREE {
                int flag;  
        } outtree_;

}


template<typename T>
T JetMatchingMGFastJet::parseParameter(const std::string &value)
{
        std::istringstream ss(value);
        T result;
        ss >> result;
        return result;
}

template<>
std::string JetMatchingMGFastJet::parseParameter(const std::string &value)
{
        std::string result;
        if (!result.empty() && result[0] == '\'')
                result = result.substr(1);
        if (!result.empty() && result[result.length() - 1] == '\'')
                result.resize(result.length() - 1);
        return result;
}

template<>
bool JetMatchingMGFastJet::parseParameter(const std::string &value_)
{
        std::string value(value_);
        std::transform(value.begin(), value.end(),
                       value.begin(), (int(*)(int))std::toupper);
        return value == "T" || value == "Y" || value=="True" ||
               value == "1" || value == ".TRUE.";
}


template<typename T>  
T JetMatchingMGFastJet::getParameter(
                        const std::map<std::string, std::string> &params,
                        const std::string &var, const T &defValue)
{
        std::map<std::string, std::string>::const_iterator pos =
                                                        params.find(var);
        if (pos == params.end())
                return defValue;
        return parseParameter<T>(pos->second);
}

template<typename T>
T JetMatchingMGFastJet::getParameter(const std::string &var,
                                    const T &defValue) const
{
        return getParameter(mgParams, var, defValue);
}


static std::map<std::string, std::string>
parseHeader(const std::vector<std::string> &header)
{
        std::map<std::string, std::string> params;

        for(std::vector<std::string>::const_iterator iter = header.begin();
            iter != header.end(); ++iter) {
                std::string line = *iter;
                if (line.empty() || line[0] == '#')
                        continue;

                std::string::size_type pos = line.find('!');
                if (pos != std::string::npos)
                        line.resize(pos);

                pos = line.find('=');
                if (pos == std::string::npos)
                        continue;

                std::string var =
                        boost::algorithm::trim_copy(line.substr(pos + 1));
                std::string value =
                        boost::algorithm::trim_copy(line.substr(0, pos));

                params[var] = value;
        }

        return params;
}

template<typename T>
void JetMatchingMGFastJet::updateOrDie(
                        const std::map<std::string, std::string> &params,
                        T &param, const std::string &name)
{
        if (param < 0){
                param = getParameter(params, name, param);
        }
        if (param < 0)
                throw cms::Exception("Generator|PartonShowerVeto")
                        << "The MGParamCMS header does not specify the jet "
                           "matching parameter \"" << name << "\", but it "
                           "is requested by the CMSSW configuration."
                        << std::endl;
}


JetMatchingMGFastJet::JetMatchingMGFastJet(const edm::ParameterSet &params) :
        JetMatching(params),
        runInitialized(false),
        fJetFinder(0),
        fIsInit(false)
{
        std::string mode = params.getParameter<std::string>("mode");
        if (mode == "inclusive") {
                soup = false;
                exclusive = false;
        } else if (mode == "exclusive") {
                soup = false;
                exclusive = true;
        } else if (mode == "auto")
                soup = true;
        else
                throw cms::Exception("Generator|LHEInterface")
                        << "MGFastJet jet matching scheme requires \"mode\" "
                           "parameter to be set to either \"inclusive\", "
                           "\"exclusive\" or \"auto\"." << std::endl;
        
        memain_.etcjet = 0.;
        memain_.rclmax = 0.0;
        memain_.clfact = 0.0;
        memain_.ktsche = 0.0;
        memain_.etaclmax = params.getParameter<double>("MEMAIN_etaclmax");
        memain_.qcut = params.getParameter<double>("MEMAIN_qcut");
        memain_.minjets = params.getParameter<int>("MEMAIN_minjets");
        memain_.maxjets = params.getParameter<int>("MEMAIN_maxjets");
        memain_.showerkt = params.getParameter<double>("MEMAIN_showerkt");
        memain_.nqmatch = params.getParameter<int>("MEMAIN_nqmatch");
        outtree_.flag = params.getParameter<int>("outTree_flag");
        std::string list_excres = params.getParameter<std::string>("MEMAIN_excres");
        std::vector<std::string> elems;
        std::stringstream ss(list_excres);
        std::string item;
        int index=0;
        while(std::getline(ss, item, ',')) {
                elems.push_back(item);
                memain_.excres[index]=std::atoi(item.c_str());
                index++;
        }
        memain_.nexcres=index;
        
        fDJROutFlag = params.getParameter<int>("outTree_flag"); 
}


double JetMatchingMGFastJet::getJetEtaMax() const
{
  return memain_.etaclmax;
}


bool JetMatchingMGFastJet::initAfterBeams()
{
  
  if ( fIsInit ) return true;
  
  //
  doMerge = uppriv_.ickkw; 
  // doMerge = true;
  qCut    = memain_.qcut; // 
  nQmatch        = memain_.nqmatch;
  clFact         = 1.; // Default value 
                      // NOTE: ME2pythia seems to default to 1.5 - need to check !!!
		      // in general, needs to read key ALPSFACT from LHE file - fix CMSSW code !!!
  nJetMin        = memain_.minjets;
  nJetMax        = memain_.maxjets;

  etaJetMax      = memain_.etaclmax; 

  coneRadius     = 1.0; 
  jetAlgoPower   = 1; //   this is the kT algorithm !!! 

  // Matching procedure
  //
  qCutSq         = pow(qCut,2);  
  // this should be something like memaev_.iexc
  fExcLocal = true;


   // If not merging, then done (?????)
   //
   // if (!doMerge) return true;
  
   // Initialise chosen jet algorithm. 
   //
   fJetFinder = new fastjet::JetDefinition( fastjet::kt_algorithm, coneRadius );
   fClusJets.clear();
   fPtSortedJets.clear();  
   
   fIsInit = true;
   
   return true;

}

void JetMatchingMGFastJet::beforeHadronisation( const lhef::LHEEvent* lhee )
{

   if (!runInitialized)
		throw cms::Exception("Generator|PartonShowerVeto")
			<< "Run not initialized in JetMatchingMGFastJet"
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
   
   // NOTE: In principle, I should use exclusive, inclusive, or soup !!!   
   // should be like this:
   if ( soup )
   {
      int NPartons = typeIdx[0].size(); 
      fExcLocal = ( NPartons < nJetMax );
   }
   else
      fExcLocal = exclusive;
        
   return;

}


void JetMatchingMGFastJet::init(const lhef::LHERunInfo* runInfo)
{
        if (fIsInit) return;

        // read MGFastJet run card

        std::map<std::string, std::string> parameters;

        std::vector<std::string> header = runInfo->findHeader("MGRunCard");
        if (header.empty())
                throw cms::Exception("Generator|PartonShowerVeto")
                        << "In order to use MGFastJet jet matching, "
                           "the input file has to contain the corresponding "
                           "MadGraph headers." << std::endl;

        mgParams = parseHeader(header);

        // set variables in common block

        std::vector<Param> params;
        std::vector<Param> values;
        for(std::map<std::string, std::string>::const_iterator iter =
                        mgParams.begin(); iter != mgParams.end(); ++iter) {
                params.push_back(" " + iter->first);
                values.push_back(iter->second);

        }

        // set MG matching parameters

        uppriv_.ickkw = getParameter<int>("ickkw", 0);
        memain_.mektsc = getParameter<int>("ktscheme", 0);

        header = runInfo->findHeader("MGParamCMS");

        std::map<std::string, std::string> mgInfoCMS = parseHeader(header);

        for(std::map<std::string, std::string>::const_iterator iter =
                        mgInfoCMS.begin(); iter != mgInfoCMS.end(); ++iter) {
                        std::cout<<"mgInfoCMS: "<<iter->first<<" "<<iter->second<<std::endl;

        }

        updateOrDie(mgInfoCMS, memain_.etaclmax, "etaclmax");
        updateOrDie(mgInfoCMS, memain_.qcut, "qcut");
        updateOrDie(mgInfoCMS, memain_.minjets, "minjets");
        updateOrDie(mgInfoCMS, memain_.maxjets, "maxjets");
        updateOrDie(mgInfoCMS, memain_.showerkt, "showerkt");
        updateOrDie(mgInfoCMS, memain_.nqmatch, "nqmatch");
                
        runInitialized = true;

        initAfterBeams();

        fIsInit=true;

        return;
}


int JetMatchingMGFastJet::match( const lhef::LHEEvent* partonLevel, 
                                 const std::vector<fastjet::PseudoJet>* jetInput )
{
         
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

  int counter = 0;
  
  const lhef::HEPEUP& hepeup = *partonLevel->getHEPEUP();
    
  while ( counter < NPartons )
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

     int idx = typeIdx[0][counter];
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
     
     // 
     // JVY - I think this is the right idea:
     // at least 1 (one) new jet needs to be added
     // however, need to double check details and refine, especially for inclusive mode
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
     
     counter++;   
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
      double dNJets = (double)NPartons;
      fDJROutput << " " << dNJets << " " << MergingScale[0] << " " 
                                         << MergingScale[1] << " "
                                         << MergingScale[2] << " "
				         << MergingScale[3] << std::endl;  
      fDJROutput.close();
   }
 
  return NONE;

}

} // end namespace
