#include <iostream>

#include "TMath.h"

#include "DQM/Physics/interface/JetCombinatorics.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


void 
Combo::Print() 
{
  edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] jet Wp  : px = " << Wp_.Px() << " py = " <<  Wp_.Py() << " pz = " << Wp_.Pz() << " e = " << Wp_.E() << std::endl;
  edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] jet Wq  : px = " << Wq_.Px() << " py = " <<  Wq_.Py() << " pz = " << Wq_.Pz() << " e = "<< Wq_.E() << std::endl;
  edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] jet Hadb: px = " << Hadb_.Px() << " py = " <<  Hadb_.Py() <<" pz = " << Hadb_.Pz() <<" e = "<< Hadb_.E() << std::endl;
  edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] jet Lepb: px = " << Lepb_.Px() << " py = " <<  Lepb_.Py() <<" pz = " << Lepb_.Pz() <<" e = "<< Lepb_.E() << std::endl;
  edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] chi-squared = " << chi2_ << " sumEt = " << SumEt_ << std::endl;
}

std::string itoa(int i) {
  char temp[20];
  sprintf(temp,"%d",i);
  return((std::string)temp);
}

JetCombinatorics::JetCombinatorics() {

  this->Clear();
		
  minMassLepW_ = -999999.;
  maxMassLepW_ = 999999.;
  minMassHadW_ = -999999.;
  maxMassHadW_ = 999999.;
  minMassLepTop_ = -999999.;
  maxMassLepTop_ = 999999.;
	
  minPhi_ = -1.;
  removeDuplicates_ = true;
  maxNJets_ = 9999;
  verbosef = false;
  UsebTagging_ = false;
  UseMtop_ = true;
  SigmasTypef = 0;
  UseFlv_ = false;
	
  Template4jCombos_ = NestedCombinatorics(); // 12 combinations
  Template5jCombos_ = Combinatorics(4,5); // 5 combinations of 4 combos
  Template6jCombos_ = Combinatorics(4,6); // 15 combinations of 4 combos
  Template7jCombos_ = Combinatorics(4,7); // 35 combinations of 4 combos
  
}

JetCombinatorics::~JetCombinatorics() {
  this->Clear();
}

void JetCombinatorics::Clear() {

  allCombos_.clear();
  allCombosSumEt_.clear();
  Template4jCombos_.clear();
  Template5jCombos_.clear();
  Template6jCombos_.clear();
  Template7jCombos_.clear();
  cand1_.clear();
	
}
	
std::map< int, std::string > JetCombinatorics::Combinatorics(int n, int max) {

  // find a combinatorics template
  // This is a simple stupid function to make algebratic combinatorics

  int kcombos = n;
  int maxcombos = max;

  std::string list;

  for ( int m=0; m<maxcombos; m++) { list = list + (itoa(m));}

  std::string seed;
  for ( int m=0; m<kcombos; m++) { seed = seed + (itoa(m));}

	
  std::map< int, std::string > aTemplateCombos;
  aTemplateCombos.clear();
	
  aTemplateCombos[0] = seed;

  int i = 0;
  int totalmatches = seed.size();
  int totalIte = list.size();

  for ( int ite = 0; ite < ((int)totalIte); ite++) {

    for ( i=0; i< (int) totalmatches; i++) {

      std::string newseed = aTemplateCombos[ite];
      std::string newseed2;
      for ( int itemp=0; itemp<(int)newseed.size(); itemp++) {
	if (itemp!=i) newseed2 = newseed2 + (newseed[itemp]);
      }
      for ( int j=0; j<(int) list.size(); j++) {
	bool Isnewelement = true;
	std::string newelement = "0";
	for (int k=0; k< (int)newseed2.size(); k++) {
	  if ( list[j] == newseed2[k] ) Isnewelement = false;
	}
	if (Isnewelement) {

	  newelement = list[j];

	  std::string candseed = newseed2;
	  candseed = candseed + newelement;

	  bool IsnewCombo = true;
	  for (int ic=0; ic<(int)aTemplateCombos.size(); ++ic ) {

	    int nmatch = 0;
	    for ( int ij=0; ij<(int)(aTemplateCombos[ic]).size(); ij++) {

	      for (int ik=0; ik<(int)candseed.size(); ik++) {
		if ( candseed[ik] == aTemplateCombos[ic][ij] ) nmatch++;
	      }
	    }
	    if (nmatch == (int)totalmatches)
	      IsnewCombo = false;

	  }
	  if (IsnewCombo) {
	    aTemplateCombos[(int)aTemplateCombos.size()] = candseed;
	  }
	}
      }
    }
  }//close iterations
  
  return aTemplateCombos;
  
}

std::map< int, std::string > JetCombinatorics::NestedCombinatorics() {

  std::map< int, std::string > aTemplateCombos;
  aTemplateCombos.clear();
  
  aTemplateCombos[0] = "0123";
  aTemplateCombos[1] = "0132";
  aTemplateCombos[2] = "0213";
  aTemplateCombos[3] = "0231";
  aTemplateCombos[4] = "0312";
  aTemplateCombos[5] = "0321";
  aTemplateCombos[6] = "1203";
  aTemplateCombos[7] = "1230";
  aTemplateCombos[8] = "1302";
  aTemplateCombos[9] = "1320";
  aTemplateCombos[10] = "2301";
  aTemplateCombos[11] = "2310";
  
  return aTemplateCombos;

}

void JetCombinatorics::FourJetsCombinations(std::vector<TLorentzVector> jets, std::vector<double> bdiscriminators ) {

  int n = 0; // total number of combos
  std::map< Combo, int, minChi2 > allCombos;
  std::map< Combo, int, maxSumEt > allCombosSumEt;
  
  std::map< int, std::string > aTemplateCombos;
  aTemplateCombos.clear();
  
  if ( jets.size() == 4 ) aTemplateCombos[0] = std::string("0123");
  if ( jets.size() == 5 ) aTemplateCombos = Template5jCombos_;
  if ( jets.size() == 6 ) aTemplateCombos = Template6jCombos_;
  if ( jets.size() == 7 ) aTemplateCombos = Template7jCombos_;	
  
  // force to use only 4 jets
  if ( maxNJets_ == 4 ) aTemplateCombos[0] = std::string("0123");
  
  edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] size of vector of jets = " << jets.size() << std::endl;
  
  for (size_t ic=0; ic != aTemplateCombos.size(); ++ic) {
    
    edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] get 4 jets from the list, cluster # " << ic << "/"<< aTemplateCombos.size()-1 << std::endl;
    
    // get a template
    std::string aTemplate = aTemplateCombos[ic];
    
    edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] template of 4 jets = " << aTemplate << std::endl;
    
    // make a list of 4 jets
    std::vector< TLorentzVector > the4jets;
    std::vector< int > the4Ids;
    std::vector< double > thebdisc;
    std::vector< double > theFlvCorr;
    
    for (int ij=0; ij<4; ij++) {
      int tmpi = atoi((aTemplate.substr(ij,1)).c_str());
      the4jets.push_back(jets[tmpi]);
      the4Ids.push_back(tmpi);
      if ( UsebTagging_ ) thebdisc.push_back( bdiscriminators[tmpi] );
      if ( UseFlv_ ) theFlvCorr.push_back( flavorCorrections_[tmpi] );
    }
    
    edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] with these 4 jets, make 12 combinations: " <<std::endl;
    
    for (size_t itemplate=0; itemplate!= Template4jCombos_.size(); ++itemplate) {
      
      std::string a4template = Template4jCombos_[itemplate];
      
      edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] ==> combination: " << a4template << " is # " << itemplate << "/"<<  Template4jCombos_.size()-1 << std::endl;
      
      Combo acombo;
      
      acombo.SetWp( the4jets[atoi((a4template.substr(0,1)).c_str())] );
      acombo.SetWq( the4jets[atoi((a4template.substr(1,1)).c_str())] );
      acombo.SetHadb( the4jets[atoi((a4template.substr(2,1)).c_str())] );
      acombo.SetLepb( the4jets[atoi((a4template.substr(3,1)).c_str())] );
      acombo.SetLepW( theLepW_ );
      
      acombo.SetIdWp( the4Ids[atoi((a4template.substr(0,1)).c_str())] );
      acombo.SetIdWq( the4Ids[atoi((a4template.substr(1,1)).c_str())] );
      acombo.SetIdHadb( the4Ids[atoi((a4template.substr(2,1)).c_str())] );
      acombo.SetIdLepb( the4Ids[atoi((a4template.substr(3,1)).c_str())] );
      
      if ( UseFlv_ ) {
	acombo.SetFlvCorrWp( theFlvCorr[atoi((a4template.substr(0,1)).c_str())] );
	acombo.SetFlvCorrWq( theFlvCorr[atoi((a4template.substr(1,1)).c_str())] );
	acombo.SetFlvCorrHadb( theFlvCorr[atoi((a4template.substr(2,1)).c_str())] );
	acombo.SetFlvCorrLepb( theFlvCorr[atoi((a4template.substr(3,1)).c_str())] );
	acombo.ApplyFlavorCorrections();
      }
      if ( UsebTagging_ ) {
	acombo.Usebtagging();
	acombo.SetbDiscPdf(bTagPdffilename_);
	acombo.SetWp_disc( thebdisc[atoi((a4template.substr(0,1)).c_str())] );
	acombo.SetWq_disc( thebdisc[atoi((a4template.substr(1,1)).c_str())] );
	acombo.SetHadb_disc( thebdisc[atoi((a4template.substr(2,1)).c_str())] );
	acombo.SetLepb_disc( thebdisc[atoi((a4template.substr(3,1)).c_str())] );
      }
      
      acombo.UseMtopConstraint(UseMtop_);
      // choose value of sigmas
      acombo.SetSigmas(SigmasTypef);
      
      acombo.analyze();
      
      if (verbosef) {
	edm::LogInfo("JetCombinatorics") << "[JetCombinatorics] ==> combination done:" << std::endl;
	acombo.Print();
      }
      
      // invariant mass cuts
      TLorentzVector aHadWP4 = acombo.GetHadW();
      TLorentzVector aLepWP4 = acombo.GetLepW();
      TLorentzVector aLepTopP4=acombo.GetLepTop();
      
      if ( ( aHadWP4.M() > minMassHadW_ && aHadWP4.M() < maxMassHadW_ ) &&
	   ( aLepWP4.M() > minMassLepW_ && aLepWP4.M() < maxMassLepW_ ) &&
	   ( aLepTopP4.M() > minMassLepTop_ && aLepTopP4.M() < maxMassLepTop_) ) {
	
	allCombos[acombo] = n;
	allCombosSumEt[acombo] = n;
	
	n++;
      }
    }
  }
  
  allCombos_ = allCombos;
  allCombosSumEt_ = allCombosSumEt;
  
}

Combo JetCombinatorics::GetCombination(int n) {
  int j = 0;
  Combo a;
  for ( std::map<Combo,int,minChi2>::const_iterator ite=allCombos_.begin();
	ite!=allCombos_.end(); ++ite) {
    if (j == n) a = ite->first;
    j++;
  }
  return a;
}

Combo JetCombinatorics::GetCombinationSumEt(int n) {
  int j = 0;
  Combo a;
  for ( std::map<Combo,int,maxSumEt>::const_iterator ite=allCombosSumEt_.begin();
	ite!=allCombosSumEt_.end(); ++ite) {
    if (j == n) a = ite->first;
    j++;
  }
  return a;
}
