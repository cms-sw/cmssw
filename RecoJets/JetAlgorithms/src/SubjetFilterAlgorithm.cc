////////////////////////////////////////////////////////////////////////////////
//
// SubjetFilterAlgorithm
// ---------------------
//
// This algorithm implements the Subjet/Filter jet reconstruction
// which was first proposed here: http://arXiv.org/abs/0802.2470
//
// The implementation is largely based on fastjet_boosted_higgs.cc provided
// with the fastjet package.
//
// see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSubjetFilterJetProducer
//
//                       David Lopes-Pegna           <david.lopes-pegna@cern.ch>
//            25/11/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "RecoJets/JetAlgorithms/interface/SubjetFilterAlgorithm.h"

#include <fastjet/ClusterSequenceArea.hh>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>


using namespace std;


ostream & operator<<(ostream & ostr, fastjet::PseudoJet & jet);


////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
SubjetFilterAlgorithm::SubjetFilterAlgorithm(const std::string& moduleLabel,
					     const std::string& jetAlgorithm,
					     unsigned nFatMax,
					     double   rParam,
					     double   rFilt,
					     double   jetPtMin, 
					     double   massDropCut,
					     double   asymmCut,
					     bool     asymmCutLater,
					     bool     doAreaFastjet,
					     double   ghostEtaMax,
					     int      activeAreaRepeats,
					     double   ghostArea,
					     bool     verbose)
  : moduleLabel_(moduleLabel)
  , jetAlgorithm_(jetAlgorithm)
  , nFatMax_(nFatMax)
  , rParam_(rParam)
  , rFilt_(rFilt)
  , jetPtMin_(jetPtMin)
  , massDropCut_(massDropCut)
  , asymmCut2_(asymmCut*asymmCut)
  , asymmCutLater_(asymmCutLater)
  , doAreaFastjet_(doAreaFastjet)
  , ghostEtaMax_(ghostEtaMax)
  , activeAreaRepeats_(activeAreaRepeats)
  , ghostArea_(ghostArea)
  , verbose_(verbose)
  , nevents_(0)
  , ntotal_(0)
  , nfound_(0)
  , fjJetDef_(0)
  , fjAreaDef_(0)
{
  // FASTJET JET DEFINITION
  if (jetAlgorithm=="CambridgeAachen"||jetAlgorithm_=="ca")
    fjJetDef_=new fastjet::JetDefinition(fastjet::cambridge_algorithm,rParam_);
  else if (jetAlgorithm=="AntiKt"||jetAlgorithm_=="ak")
    fjJetDef_=new fastjet::JetDefinition(fastjet::antikt_algorithm,rParam_);
  else if (jetAlgorithm=="Kt"||jetAlgorithm_=="kt")
    fjJetDef_=new fastjet::JetDefinition(fastjet::kt_algorithm,rParam_);
  else
    throw cms::Exception("InvalidJetAlgo")
      <<"Jet Algorithm for SubjetFilterAlgorithm is invalid: "
      <<jetAlgorithm_<<", use (ca|CambridgeAachen)|(Kt|kt)|(AntiKt|ak)"<<endl;

  // FASTJET AREA DEFINITION
  if (doAreaFastjet_) {
    fastjet::GhostedAreaSpec ghostSpec(ghostEtaMax_,activeAreaRepeats_,ghostArea_);
    fjAreaDef_=new fastjet::AreaDefinition(fastjet::active_area_explicit_ghosts,
					   ghostSpec); 
  }
}


//______________________________________________________________________________
SubjetFilterAlgorithm::~SubjetFilterAlgorithm()
{
  if (0!=fjJetDef_)  delete fjJetDef_;
  if (0!=fjAreaDef_) delete fjAreaDef_;
}
  

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void SubjetFilterAlgorithm::run(const std::vector<fastjet::PseudoJet>& fjInputs, 
				std::vector<CompoundPseudoJet>& fjJets,
				const edm::EventSetup& iSetup)
{
  nevents_++;
  
  if (verbose_) cout<<endl<<nevents_<<". EVENT"<<endl;
  
  fastjet::ClusterSequence* cs = (doAreaFastjet_) ?
    new fastjet::ClusterSequenceArea(fjInputs,*fjJetDef_,*fjAreaDef_) :
    new fastjet::ClusterSequence    (fjInputs,*fjJetDef_);
  
  vector<fastjet::PseudoJet> fjFatJets =
    fastjet::sorted_by_pt(cs->inclusive_jets(jetPtMin_));
  
  size_t nFat =
    (nFatMax_==0) ? fjFatJets.size() : std::min(fjFatJets.size(),(size_t)nFatMax_);
  
  for (size_t iFat=0;iFat<nFat;iFat++) {
    
    if (verbose_) cout<<endl<<iFat<<". FATJET: "<<fjFatJets[iFat]<<endl;
    
    fastjet::PseudoJet fjFatJet = fjFatJets[iFat];
    fastjet::PseudoJet fjCurrentJet(fjFatJet);
    fastjet::PseudoJet fjSubJet1,fjSubJet2;
    bool hadSubJets;
    
    vector<CompoundPseudoSubJet> subJets;
    
    
    // FIND SUBJETS PASSING MASSDROP [AND ASYMMETRY] CUT(S)
    while ((hadSubJets = cs->has_parents(fjCurrentJet,fjSubJet1,fjSubJet2))) {
      
      if (fjSubJet1.m() < fjSubJet2.m()) swap(fjSubJet1,fjSubJet2);
      
      if (verbose_) cout<<"SUBJET CANDIDATES: "<<fjSubJet1<<endl
			<<"                   "<<fjSubJet2<<endl;
      if (verbose_) cout<<"md="<<fjSubJet1.m()/fjCurrentJet.m()
			<<" y="<<fjSubJet1.kt_distance(fjSubJet2)/fjCurrentJet.m2()
			<<endl;
      
      if (fjSubJet1.m()<massDropCut_*fjCurrentJet.m() &&
	  (asymmCutLater_||
	   fjSubJet1.kt_distance(fjSubJet2)>asymmCut2_*fjCurrentJet.m2())) {
	break;
      }
      else {
	fjCurrentJet = fjSubJet1;
      }
    }
    
    if (!hadSubJets) {
      if (verbose_) cout<<"FAILED TO SELECT SUBJETS"<<endl;
    }
    // FOUND TWO GOOD SUBJETS PASSING MASSDROP CUT
    else {
      if (verbose_) cout<<"SUBJETS selected"<<endl;
      
      if (asymmCutLater_&&
	  fjSubJet1.kt_distance(fjSubJet2)<=asymmCut2_*fjCurrentJet.m2()) {
	if (verbose_) cout<<"FAILED y-cut"<<endl;
      }
      // PASSED ASYMMETRY (Y) CUT
      else {
	if (verbose_) cout<<"PASSED y cut"<<endl;
	
	vector<fastjet::PseudoJet> fjFilterJets;
	double       Rbb   = std::sqrt(fjSubJet1.squared_distance(fjSubJet2));
	double       Rfilt = std::min(0.5*Rbb,rFilt_);
	double       dcut  = Rfilt*Rfilt/rParam_/rParam_;
	fjFilterJets=fastjet::sorted_by_pt(cs->exclusive_subjets(fjCurrentJet,dcut));
	
	if (verbose_) {
	  cout<<"Rbb="<<Rbb<<", Rfilt="<<Rfilt<<endl;
	  cout<<"FILTER JETS: "<<flush;
	  for (size_t i=0;i<fjFilterJets.size();i++) {
	    if (i>0) cout<<"             "<<flush; cout<<fjFilterJets[i]<<endl;
	  }
	}
	
	vector<fastjet::PseudoJet> fjSubJets;
	fjSubJets.push_back(fjSubJet1);
	fjSubJets.push_back(fjSubJet2);
	size_t nFilter = fjFilterJets.size();
	for (size_t iFilter=0;iFilter<nFilter;iFilter++)
	  fjSubJets.push_back(fjFilterJets[iFilter]);
	
	for (size_t iSub=0;iSub<fjSubJets.size();iSub++) {
	  vector<fastjet::PseudoJet> fjConstituents=
	    cs->constituents(fjSubJets[iSub]);
	  vector<int> constituents;
	  for (size_t iConst=0;iConst<fjConstituents.size();iConst++) {
	    int userIndex = fjConstituents[iConst].user_index();
	    if (userIndex>=0)  constituents.push_back(userIndex);
	  }

	  double subJetArea=(doAreaFastjet_) ?
	    ((fastjet::ClusterSequenceArea*)cs)->area(fjSubJets[iSub]) : 0.0;
	  
	  if (iSub<2||constituents.size()>0)
	    subJets.push_back(CompoundPseudoSubJet(fjSubJets[iSub],
						   subJetArea,
						   constituents));
	}
	
      } // PASSED Y CUT
      
    } // PASSED MASSDROP CUT
    
    if (verbose_) cout<<"write fatjet with "<<subJets.size()<<" sub+filter jets"
		      <<endl;
    
    double fatJetArea = (doAreaFastjet_) ?
      ((fastjet::ClusterSequenceArea*)cs)->area(fjFatJet) : 0.0;

    fjJets.push_back(CompoundPseudoJet(fjFatJet,fatJetArea,subJets));
    
    ntotal_++;
    if (subJets.size()>3) nfound_++;
    
  } // LOOP OVER FATJETS
  
  if (verbose_) cout<<endl<<fjJets.size()<<" FATJETS written\n"<<endl;
  
  delete cs;
  
  return;
}


//______________________________________________________________________________
string SubjetFilterAlgorithm::summary() const
{
  double eff = (ntotal_>0) ? nfound_/(double)ntotal_ : 0;
  std::stringstream ss;
  ss<<"************************************************************\n"
    <<"* "<<moduleLabel_<<" (SubjetFilterAlgorithm) SUMMARY:\n"
    <<"************************************************************\n"
    <<"nevents = "<<nevents_<<endl
    <<"ntotal  = "<<ntotal_<<endl
    <<"nfound  = "<<nfound_<<endl
    <<"eff     = "<<eff<<endl
    <<"************************************************************\n";
  return ss.str();
}



/// does the actual work for printing out a jet
ostream & operator<<(ostream & ostr, fastjet::PseudoJet & jet) {
  ostr << "pt="  <<setw(10)<<jet.perp() 
       << " eta="<<setw(6) <<jet.eta()  
       << " m="  <<setw(10)<<jet.m();
  return ostr;
}
