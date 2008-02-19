
#include "PhysicsTools/PatUtils/interface/CaloJetSelector.h"
#include "DataFormats/Math/interface/deltaR.h"
 
using pat::CaloJetSelector;

//______________________________________________________________________________
CaloJetSelector::CaloJetSelector( const edm::ParameterSet& config ) 
{

  // Retrieve configuration config.ters only once
    EMFmin_		    = config.getParameter<double>("EMFmin");
    EMFmax_                 = config.getParameter<double>("EMFmax");
    Etamax_                 = config.getParameter<double>("Etamax");
    PTmin_		    = config.getParameter<double>("PTmin");
    EMvsHadFmin_ 	    = config.getParameter<double>("EMvsHadFmin");		
    EMvsHadFmax_ 	    = config.getParameter<double>("EMvsHadFmax");
    HadFmin_ 		    = config.getParameter<double>("HadFmin");			 
    HadFmax_ 		    = config.getParameter<double>("HadFmax");
    N90min_ 		    = config.getParameter<double>("N90min");			 
    N90max_ 		    = config.getParameter<double>("N90max");
    NCaloTowersmin_ 	    = config.getParameter<double>("NCaloTowersmin");		 
    NCaloTowersmax_ 	    = config.getParameter<double>("NCaloTowersmax");
    HighestTowerOverJetmin_ = config.getParameter<double>("HighestTowerOverJetmin");	 
    HighestTowerOverJetmax_ = config.getParameter<double>("HighestTowerOverJetmax");
    RWidthmin_ 		    = config.getParameter<double>("RWidthmin"); 		 
    RWidthmax_ 		    = config.getParameter<double>("RWidthmax");
    PTjetOverArea_min_ 	    = config.getParameter<double>("PTjetOverAreamin"); 	 
    PTjetOverArea_max_ 	    = config.getParameter<double>("PTjetOverAreamax");
    PTtowerOverArea_min_    = config.getParameter<double>("PTtowerOverAreamin");	 
    PTtowerOverArea_max_    = config.getParameter<double>("PTtowerOverAreamax");
}


//______________________________________________________________________________
const unsigned int 
CaloJetSelector::filter( //const unsigned int&        index, 
                         // const edm::View<reco::CaloJet>& Jets
			  const reco::CaloJet& Jet
                          ) const
{
  bool result = GOOD;

  ///Retrieve information
  ///Pt Jet
  if (Jet.p4().Pt()<PTmin_) result = BAD;
  
  ///eta region
  double eta = fabs(Jet.p4().Eta());
  if (eta>Etamax_) result = BAD;

  ///electromagnetic fraction
  double EMF = Jet.emEnergyFraction();              
  if (EMF<EMFmin_ ||
      EMF>EMFmax_    ) result = BAD;

  ///(EMCalEnergyFraction + HadCalEnergyFraction) / (EMCalEnergyFraction - HadCalEnergyFraction) 
  double HadF = Jet.emEnergyFraction();              
  double EMvsHadF = 0.;
  if (EMF-HadF!=0.) EMvsHadF = (EMF+HadF)/(EMF-HadF);
  if (EMvsHadF<EMvsHadFmin_ ||
      EMvsHadF>EMvsHadFmax_    ) result = BAD;

  //ratio Energy over Momentum (both from calorimeter)
  //Useful? Both come from a lorentz-vector
  //double EoverP = 0.;
  //if (Jet.p4().P()!=0.) EoverP = Jet.energy() / Jet.p4().P();
  //if (EoverP > EoverPmax_) result = BAD;
  
  ///n90: number of towers containing 90% of the jet's energy
  double n90 = Jet.n90();
  if (n90<N90min_ ||
      n90>N90max_    ) result = BAD;

  ///Tower Number
  std::vector <CaloTowerRef> jetTowers = Jet.getConstituents();
  if (jetTowers.size()<NCaloTowersmin_ ||
      jetTowers.size()>NCaloTowersmax_  ) result = BAD;

  //calculate tower related variables:
  double MaxEnergyTower = 0., SumTowPt=0., SumTowPtR=0.;
  for(std::vector<CaloTowerRef>::const_iterator tow = jetTowers.begin(),
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
  if (EtTowerMaxOverEtJet < HighestTowerOverJetmin_ ||
      EtTowerMaxOverEtJet > HighestTowerOverJetmax_  ) result = BAD;

  ///Sum(E Twr * DeltaR(Twr-Jet)) / Sum(E Twr)
  double RWidth = 0.;
  if (SumTowPt!=0.) RWidth = SumTowPtR /SumTowPt;
  if (RWidth < RWidthmin_ ||
      RWidth > RWidthmax_  ) result = BAD;
  
  ///Pt Jet / Towers Area 		  
  double PtJetoverArea = 0.;
  if (Jet.towersArea() !=0.) PtJetoverArea = Jet.p4().Pt() / Jet.towersArea();
  if (PtJetoverArea < PTjetOverArea_min_ ||
      PtJetoverArea > PTjetOverArea_max_  ) result = BAD;

  ///Highest Et Tower / Towers Area
  double PtToweroverArea = 0.;
  if (Jet.towersArea() !=0.) PtToweroverArea = MaxEnergyTower / Jet.towersArea();
  if (PtToweroverArea < PTtowerOverArea_min_ ||
      PtToweroverArea > PTtowerOverArea_max_  ) result = BAD;

  return result;
}
