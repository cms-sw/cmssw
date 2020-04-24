#include "DQMOffline/RecoB/interface/AcceptJet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include<iostream>

using namespace std;

AcceptJet::AcceptJet(const double& etaMin_, const double& etaMax_, const double& ptMin_, const double& ptMax_,
		     const double& pMin_, const double& pMax_, const double& ratioMin_, const double& ratioMax_, const bool& doJetID_) :
  etaMin(etaMin_), etaMax(etaMax_), ptRecJetMin(ptMin_), ptRecJetMax(ptMax_), pRecJetMin(pMin_),
  pRecJetMax(pMax_), ratioMin(ratioMin_), ratioMax(ratioMax_), doJetID(doJetID_) {}


bool AcceptJet::operator() (const reco::Jet & jet, const int & jetFlavour, const edm::Handle<reco::SoftLeptonTagInfoCollection> & infos, const double jec) const
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

  if ( jet.pt()*jec < ptRecJetMin ||
       jet.pt()*jec > ptRecJetMax ) return false;

  if ( jet.p()*jec < pRecJetMin ||
       jet.p()*jec > pRecJetMax ) return false;

  if ( !infos.isValid() ) {
    LogDebug("infos not valid") << "A valid SoftLeptonTagInfoCollection was not found!"
                                     << " Skipping ratio check.";
  }
  else {
    double pToEratio = ratio( jet, infos );
    if ( pToEratio/jec < ratioMin ||
         pToEratio/jec > ratioMax ) return false;
  }

  if(doJetID){
    const reco::PFJet * pfjet = dynamic_cast<const reco::PFJet *>(&jet);
    if(pfjet) {
      double neutralHadronEnergyFraction = pfjet->neutralHadronEnergy()/(jet.energy()*jec);
      double neutralEmEnergyFraction = pfjet->neutralEmEnergy()/(jet.energy()*jec);
      int nConstituents = pfjet->getPFConstituents().size();
      double chargedHadronEnergyFraction = pfjet->chargedHadronEnergy()/(jet.energy()*jec);
      int chargedMultiplicity = pfjet->chargedMultiplicity();
      double chargedEmEnergyFraction = pfjet->chargedEmEnergy()/(jet.energy()*jec);
      if(!(neutralHadronEnergyFraction < 0.99 
	   && neutralEmEnergyFraction < 0.99 
	   && nConstituents > 1 
	   && chargedHadronEnergyFraction > 0.0 
	   && chargedMultiplicity > 0.0 
	   && chargedEmEnergyFraction < 0.99)) 
	return false; //2012 values
    }
    //Looks to not work -> to be fixed
    //else std::cout<<"Jets are not PF jets, put 'doJetID' to False."<<std::endl;
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
