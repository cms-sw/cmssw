//  ProtJet2.cc
//  Revision History:  R. Harris 10/19/05  Modified to work with real CaloTowers from Jeremy Mans
//

#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"

#include <vector>
#include <algorithm>               // include STL algorithm implementations
#include <numeric>		   // For the use of numeric

ProtoJet::ProtoJet(const Candidates& fConstituents) 
  : mConstituents (fConstituents)
{
  calculateLorentzVector(); 
}//end of constructor


void ProtoJet::calculateLorentzVector() {
  mP4 = LorentzVector (0,0,0,0);
  for(Candidates::const_iterator i = mConstituents.begin(); i !=  mConstituents.end(); ++i) {
    const reco::Candidate* c = *i;
    mP4 += c->p4();
  } //end of loop over the jet constituents
}



