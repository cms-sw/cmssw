#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

  ratioMin = -1.0;
  ratioMax = 9999.9;
}
// 'global' event selection based on basic variables


bool AcceptJet::operator() (const reco::Jet & jet, const int & jetFlavour, const edm::Handle<reco::SoftLeptonTagInfoCollection> & infos) const
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

  if ( !infos.isValid() ) {
    edm::LogWarning("infos not valid") << "A valid SoftLeptonTagInfoCollection was not found!"
                                     << " Skipping ratio check.";
  }
  else {
    double pToEratio = ratio( jet, infos );
    if ( pToEratio < ratioMin ||
         pToEratio > ratioMax ) accept = false;
  }

  return accept;
}

double AcceptJet::ratio(const reco::Jet & jet, const edm::Handle<reco::SoftLeptonTagInfoCollection>& infos) const
{
  double jetRatio = 0.0;
  reco::SoftLeptonTagInfoCollection::const_iterator infoiter = infos->begin();
  for( ; infoiter != infos->end(); ++infoiter)
  {
    if( reco::deltaR(jet.eta(), jet.phi(), infoiter->jet()->eta(), infoiter->jet()->phi()) > 1e-4 )
      continue;

    if( infoiter->leptons() == 0 )
      break;

    for( unsigned int k = 0; k != infoiter->leptons(); ++k )
    {
      double tempRatio = infoiter->properties(k).ratio;
      if( tempRatio > jetRatio )
        jetRatio = tempRatio;
    }
  }

  return jetRatio;
}
