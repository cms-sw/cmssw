#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<iostream>

using namespace std;

AcceptJet::AcceptJet(const double& etaMin_, const double& etaMax_, const double& ptMin_, const double& ptMax_,
          const double& pMin_, const double& pMax_, const double& ratioMin_, const double& ratioMax_) :
  etaMin(etaMin_), etaMax(etaMax_), ptRecJetMin(ptMin_), ptRecJetMax(ptMax_), pRecJetMin(pMin_),
  pRecJetMax(pMax_), ratioMin(ratioMin_), ratioMax(ratioMax_) {}


bool AcceptJet::operator() (const reco::Jet & jet, const int & jetFlavour, const edm::Handle<reco::SoftLeptonTagInfoCollection> & infos) const
{

  // temporary fudge to correct for double loop error
  //  jetPartonMomentum /= 2.0;

  if ( fabs(jet.eta()) < etaMin  ||
       fabs(jet.eta()) > etaMax  ) return false;

//   if ( jetFlavour.underlyingParton4Vec().P() < pPartonMin  ||
//        jetFlavour.underlyingParton4Vec().P() > pPartonMax  ) accept = false;
// 
//   if ( jetFlavour.underlyingParton4Vec().Pt() < ptPartonMin  ||
//        jetFlavour.underlyingParton4Vec().Pt() > ptPartonMax  ) accept = false;

  if ( jet.pt() < ptRecJetMin ||
       jet.pt() > ptRecJetMax ) return false;

  if ( jet.p() < pRecJetMin ||
       jet.p() > pRecJetMax ) return false;

  if ( !infos.isValid() ) {
    edm::LogWarning("infos not valid") << "A valid SoftLeptonTagInfoCollection was not found!"
                                     << " Skipping ratio check.";
  }
  else {
    double pToEratio = ratio( jet, infos );
    if ( pToEratio < ratioMin ||
         pToEratio > ratioMax ) return false;
  }

  return true;
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
