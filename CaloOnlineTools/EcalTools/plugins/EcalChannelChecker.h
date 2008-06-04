// -*- C++ -*-
//
// Package:   EcalChannelChecker 
// Class:     EcalChannelChecker 
// 
/**\class EcalChannelChecker EcalChannelChecker.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
// Original Author:  Caterina DOGLIONI
//         Created:  Tu Apr 22 5:46:22 CEST 2008
// $Id: EcalChannelChecker.h,v 1.7 2008/05/05 13:31:42 doglioni Exp $
//
//

// system include files
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <string>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TEventList.h"
#include "TCut.h"

#define MAX_XTALS 61200

//
// class declaration
//

class EcalChannelChecker : public edm::EDAnalyzer {
	public:
		explicit EcalChannelChecker(const edm::ParameterSet&);
		~EcalChannelChecker();

	private:

		//enum for graph type
		enum n_h1Type { H1_AMPLI=0 , H1_PED , H1_JITTER, PROF_PULSE , NTYPES}; 


		virtual void beginJob(const edm::EventSetup&) ;
		virtual void analyze(const edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;

		//helpers
		TEventList * getEventListFromCut (const TCut& cut);
		std::string intToString(int num);
		std::string printBitmask(std::vector<bool> * bitmask);
		std::string printBitmaskCuts(std::vector<bool> * bitmask);
		std::string makeCutFromMaskedVectorInt(const std::vector<int> & v_masked, const std::string & type);
		std::string makeCutFromMaskedVectorString(const std::vector<std::string> & v_masked, const std::string & type);
		void initHistTypeMaps();
		void fillEventListVector(const std::vector<std::string>& v_cuts, const std::vector<std::string> & v_masked);   	
		void printLogEventList(const TEventList & eventList);
		void printMaskedHi();
		void printMaskedSlices();
		void writeHistFromFile(int hashedIndex, const char* slice, int ic, n_h1Type H1TYPE); 

		// ----------member data --------------------------

		//tree
		TTree * t_;

		//filenames
		std::string inputTreeFileName_;
		std::string inputHistoFileName_;
		std::string outputFileName_;

		//vectors for cuts/event lists
		std::vector<std::string> v_cuts_;
		std::vector <TEventList> v_eventList_;
		std::vector <int> v_maskedHi_;
		std::vector <std::string> v_maskedSlices_;

		//total TEventList (no duplicate crystals)
		TEventList totalEventList_;

		//vector for bitmasks (vector<bool> bitmask[i] = 1 ->  crystal selected for v_cuts_[i])
		std::vector<bool> * xtalBitmask_[MAX_XTALS];
		//number of cuts
		int nCuts_;

		//maps
		//map for directories:hist type
		std::map < EcalChannelChecker::n_h1Type , std::string > h1TypeToDirectoryMap_;
		//map for hist name:hist type
		std::map < EcalChannelChecker::n_h1Type , std::string > h1TypeToNameMap_;

		//files
		TFile * fin_tree_;
		TFile * fin_histo_;
		TFile * fout_;

		//tree variables
		int	ic;
                char    slice[5];
		int     ieta;
		int     iphi;
		int     hashedIndex;
		float   ped_avg;
		float   ped_rms;
		float   ampli_avg;
		float   ampli_rms;
		float   jitter_avg;
		float   jitter_rms;
		float   ampli_fracBelowThreshold;
		int     entries;
		float   entriesOverAvg;

		//branches
		TBranch        *b_ic;   //!
		TBranch        *b_slice;   //!
		TBranch        *b_ieta;   //!
		TBranch        *b_iphi;   //!
		TBranch        *b_hashedIndex;   //!
		TBranch        *b_ped_avg;   //!
		TBranch        *b_ped_rms;   //!
		TBranch        *b_ampli_avg;   //!
		TBranch        *b_ampli_rms;   //!
		TBranch        *b_jitter_avg;   //!
		TBranch        *b_jitter_rms;   //!
		TBranch        *b_ampli_fracBelowThreshold;   //!
		TBranch        *b_entries;   //!
		TBranch        *b_entriesOverAvg;

};
