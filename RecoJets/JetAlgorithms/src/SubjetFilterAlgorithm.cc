////////////////////////////////////////////////////////////////////////////////
//
// SubjetFilterAlgorithm
// ---------------------
//
//            25/11/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "RecoJets/JetAlgorithms/interface/SubjetFilterAlgorithm.h"

#include <fastjet/ClusterSequence.hh>

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
					     double rParam, double jetPtMin,
					     double massDropCut,double asymmCut,
					     bool asymmCutLater)
  : moduleLabel_(moduleLabel)
  , jetAlgorithm_(jetAlgorithm)
  , nFatMax_(nFatMax)
  , rParam_(rParam)
  , jetPtMin_(jetPtMin)
  , massDropCut_(massDropCut)
  , asymmCut2_(asymmCut*asymmCut)
  , asymmCutLater_(asymmCutLater)
  , ntotal_(0)
  , nfound_(0)
  , fjJetDef_(0)
{
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
}


//______________________________________________________________________________
SubjetFilterAlgorithm::~SubjetFilterAlgorithm()
{
  if (0!=fjJetDef_) delete fjJetDef_;
}
  

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void SubjetFilterAlgorithm::run(const std::vector<fastjet::PseudoJet>& fjInputs, 
				std::vector<CompoundPseudoJet>& fjJets,
				const edm::EventSetup& iSetup)
{
  ntotal_++;

  // DEBUG
  //cout<<endl<<ntotal_<<". EVENT:"<<endl;
  
  fastjet::ClusterSequence cs(fjInputs,*fjJetDef_);
  vector<fastjet::PseudoJet> fjFatJets =
    fastjet::sorted_by_pt(cs.inclusive_jets(jetPtMin_));
  
  unsigned int nFat = std::min((unsigned int)fjFatJets.size(),nFatMax_);
  for (unsigned iFat=0;iFat<nFat;iFat++) {
    fastjet::PseudoJet fjFatJet = fjFatJets[iFat];
    fastjet::PseudoJet fjCurrentJet(fjFatJet);
    fastjet::PseudoJet fjSubJet1,fjSubJet2;
    bool hadSubJets;
    
    // DEBUG
    //cout<<iFat+1<<". FAT JET ("<<rParam_<<"):\n"<<fjFatJet<<endl;
    
    while ((hadSubJets = cs.has_parents(fjCurrentJet,fjSubJet1,fjSubJet2))) {
      if (fjSubJet1.m() < fjSubJet2.m()) swap(fjSubJet1,fjSubJet2);
      if (fjSubJet1.m()<massDropCut_*fjCurrentJet.m() &&
	  (asymmCutLater_||
	   fjSubJet1.kt_distance(fjSubJet2)>asymmCut2_*fjCurrentJet.m2())) {
	break;
      }
      else {
	fjCurrentJet = fjSubJet1;
      }
    }
    
    if (!hadSubJets) break;
    
    if (asymmCutLater_&&
	fjSubJet1.kt_distance(fjSubJet2)<=asymmCut2_*fjCurrentJet.m2()) break;
    
    
    vector<fastjet::PseudoJet> fjFilterJets;
    double       Rbb   = std::sqrt(fjSubJet1.squared_distance(fjSubJet2));
    double       Rfilt = std::min(0.5*Rbb,0.3);
    double       dcut  = Rfilt*Rfilt/rParam_/rParam_;
    fjFilterJets = fastjet::sorted_by_pt(cs.exclusive_subjets(fjCurrentJet,dcut));

    // DEBUG
    //cout<<"SUB JETS ("<<Rbb<<"):\n"<<fjSubJet1<<endl<<fjSubJet2<<endl;
    
    // DEBUG
    //cout<<"FILTER JETS:\n"
    //<<fjFilterJets[0]<<endl
    //<<fjFilterJets[1]<<endl
    //<<fjFilterJets[2]<<endl
    //<<endl;
    
    vector<fastjet::PseudoJet> fjSubJets;
    fjSubJets.push_back(fjSubJet1);
    fjSubJets.push_back(fjSubJet2);
    unsigned nFilter = std::min(3,(int)fjFilterJets.size());
    for (unsigned iFilter=0;iFilter<nFilter;iFilter++)
      fjSubJets.push_back(fjFilterJets[iFilter]);
    
    vector<CompoundPseudoSubJet> subJets;
    for (unsigned iSub=0;iSub<fjSubJets.size();iSub++) {
      vector<fastjet::PseudoJet> fjConstituents = cs.constituents(fjSubJets[iSub]);
      vector<int> constituents;
      for (unsigned iConst=0;iConst<fjConstituents.size();iConst++)
	constituents.push_back(fjConstituents[iConst].user_index());
      subJets.push_back(CompoundPseudoSubJet(fjSubJets[iSub],constituents));
    }
    
    fjJets.push_back(CompoundPseudoJet(fjFatJet,subJets));
  }
  
  if (fjJets.size()>0) nfound_++;
  
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
    <<"ntotal = "<<ntotal_<<endl
    <<"nfound = "<<nfound_<<endl
    <<"eff    = "<<eff<<endl
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
