#include "DQMOffline/RecoB/interface/AcceptJet.h"

#include<iostream>

using namespace std;

AcceptJet::AcceptJet()
{
  // cut parameters
  // (meaningless ones to make sure that no event processed if not set by user)
  etaMin     = 0.0;
  etaMax     = 2.4;

//   pPartonMin = 9999.9;
//   pPartonMax = 0.0;
// 
//   ptPartonMin = 9999.9;
//   ptPartonMax = 0.0;

  ptRecJetMin = 9999.9;
  ptRecJetMax = 0.0;

  pRecJetMin  = 9999.9;
  pRecJetMax  = 0.0;
}
// 'global' event selection based on basic variables


bool AcceptJet::operator() (const reco::Jet & jet, const int & jetFlavour) const
{

  bool accept = true;

  // temporary fudge to correct for double loop error
  //  jetPartonMomentum /= 2.0;

  if ( fabs(jet.eta()) < etaMin  ||
       fabs(jet.eta()) > etaMax  ) accept = false;

//   if ( jetFlavour.underlyingParton4Vec().P() < pPartonMin  ||
//        jetFlavour.underlyingParton4Vec().P() > pPartonMax  ) accept = false;
// 
//   if ( jetFlavour.underlyingParton4Vec().Pt() < ptPartonMin  ||
//        jetFlavour.underlyingParton4Vec().Pt() > ptPartonMax  ) accept = false;

  if ( jet.pt() < ptRecJetMin ||
       jet.pt() > ptRecJetMax ) accept = false;

  if ( jet.p() < pRecJetMin ||
       jet.p() > pRecJetMax ) accept = false;

  return accept;
}
