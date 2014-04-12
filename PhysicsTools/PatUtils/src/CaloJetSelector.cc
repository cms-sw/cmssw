
#include "PhysicsTools/PatUtils/interface/CaloJetSelector.h"
#include "DataFormats/Math/interface/deltaR.h"
 
using pat::CaloJetSelector;

//______________________________________________________________________________
const pat::ParticleStatus
CaloJetSelector::filter( //const unsigned int&        index, 
                         // const edm::View<reco::CaloJet>& Jets
			  const reco::CaloJet& Jet
                          ) const
{
  ParticleStatus result = GOOD;

  ///Retrieve information
  ///Pt Jet
  if (Jet.p4().Pt()<config_.Ptmin) result = BAD;
  
  ///eta region
  double eta = fabs(Jet.p4().Eta());
  if (eta>config_.Etamax) result = BAD;

  ///electromagnetic fraction
  double EMF = Jet.emEnergyFraction();              
  if (EMF<config_.EMFmin ||
      EMF>config_.EMFmax    ) result = BAD;

  ///(EMCalEnergyFraction + HadCalEnergyFraction) / (EMCalEnergyFraction - HadCalEnergyFraction) 
  double HadF = Jet.emEnergyFraction();              
  double EMvsHadF = 0.;
  if (EMF-HadF!=0.) EMvsHadF = (EMF+HadF)/(EMF-HadF);
  if (EMvsHadF<config_.EMvsHadFmin ||
      EMvsHadF>config_.EMvsHadFmax    ) result = BAD;

  //ratio Energy over Momentum (both from calorimeter)
  //Useful? Both come from a lorentz-vector
  //double EoverP = 0.;
  //if (Jet.p4().P()!=0.) EoverP = Jet.energy() / Jet.p4().P();
  //if (EoverP > config_.EoverPmax) result = BAD;
  
  ///n90: number of towers containing 90% of the jet's energy
  double n90 = Jet.n90();
  if (n90<config_.N90min ||
      n90>config_.N90max    ) result = BAD;

  ///Tower Number
  std::vector<CaloTowerPtr> jetTowers = Jet.getCaloConstituents();
  if (jetTowers.size()<config_.NCaloTowersmin ||
      jetTowers.size()>config_.NCaloTowersmax  ) result = BAD;

  //calculate tower related variables:
  double MaxEnergyTower = 0., SumTowPt=0., SumTowPtR=0.;
  for(std::vector<CaloTowerPtr>::const_iterator tow = jetTowers.begin(),
      towend = jetTowers.end(); tow != towend; ++tow){

    SumTowPt  += (*tow)->et();
    SumTowPtR += (*tow)->et()*deltaR( Jet.p4().Eta(), Jet.p4().Phi(),
                                      (*tow)->eta(),  (*tow)->phi()     );
    if ( (*tow)->et() > MaxEnergyTower )
       MaxEnergyTower = (*tow)->et();
  }

  ///Highest Et Tower / Et Jet
  double EtTowerMaxOverEtJet = 0.;
  if (Jet.p4().Et()!=0.) EtTowerMaxOverEtJet = MaxEnergyTower /Jet.p4().Et();
  if (EtTowerMaxOverEtJet < config_.HighestTowerOverJetmin ||
      EtTowerMaxOverEtJet > config_.HighestTowerOverJetmax  ) result = BAD;

  ///Sum(E Twr * DeltaR(Twr-Jet)) / Sum(E Twr)
  double RWidth = 0.;
  if (SumTowPt!=0.) RWidth = SumTowPtR /SumTowPt;
  if (RWidth < config_.RWidthmin ||
      RWidth > config_.RWidthmax  ) result = BAD;
  
  ///Pt Jet / Towers Area 		  
  double PtJetoverArea = 0.;
  if (Jet.towersArea() !=0.) PtJetoverArea = Jet.p4().Pt() / Jet.towersArea();
  if (PtJetoverArea < config_.PtJetOverArea_min ||
      PtJetoverArea > config_.PtJetOverArea_max  ) result = BAD;

  ///Highest Et Tower / Towers Area
  double PtToweroverArea = 0.;
  if (Jet.towersArea() !=0.) PtToweroverArea = MaxEnergyTower / Jet.towersArea();
  if (PtToweroverArea < config_.PtTowerOverArea_min ||
      PtToweroverArea > config_.PtTowerOverArea_max  ) result = BAD;

  return result;
}
