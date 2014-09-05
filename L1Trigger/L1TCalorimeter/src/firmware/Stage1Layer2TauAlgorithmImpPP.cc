/// \class l1t::Stage1Layer2TauAlgorithm
///
/// Description: interface for MP firmware
///
/// Implementation:
///
/// \author: Kalanand Mishra - Fermilab
///
/// Tau definition: 4x8 towers.


#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2TauAlgorithmImp.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/JetFinderMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"


using namespace std;
using namespace l1t;


Stage1Layer2TauAlgorithmImpPP::Stage1Layer2TauAlgorithmImpPP(CaloParamsStage1* params) : params_(params)
{
}

Stage1Layer2TauAlgorithmImpPP::~Stage1Layer2TauAlgorithmImpPP(){};




void l1t::Stage1Layer2TauAlgorithmImpPP::processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
						      const std::vector<l1t::CaloRegion> & regions,
						      const std::vector<l1t::Jet> * jets,
						      std::vector<l1t::Tau> * taus) {

  double towerLsb = params_->towerLsbSum();

  std::string regionPUSType = params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();
  int tauSeedThreshold= floor( params_->tauSeedThreshold()/towerLsb + 0.5); // convert GeV to HW units
  int tauNeighbourThreshold= floor( params_->tauNeighbourThreshold()/towerLsb + 0.5); // convert GeV to HW units
  int jetSeedThreshold= floor( params_->jetSeedThreshold()/towerLsb + 0.5); // convert GeV to HW units
  int switchOffTauVeto = floor( params_->switchOffTauVeto()/towerLsb + 0.5);
  int switchOffTauIso = floor( params_->switchOffTauIso()/towerLsb + 0.5);
  int tauRelativeJetIsolationLimit = params_->tauRelativeJetIsolationLimit();
  double tauRelativeJetIsolationCut = params_->tauRelativeJetIsolationCut();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();



  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  RegionCorrection(regions, EMCands, subRegions, regionPUSParams, regionPUSType);



  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  //slidingWindowJetFinder(jetSeedThreshold, subRegions, unCorrJets);
  TwelveByTwelveFinder(jetSeedThreshold, subRegions, unCorrJets);

  std::vector<l1t::Tau> *preGtTaus = new std::vector<l1t::Tau>();


  for(CaloRegionBxCollection::const_iterator region = subRegions->begin();
      region != subRegions->end(); region++) {

    int regionEt = region->hwPt();
    if(regionEt < tauSeedThreshold) continue;

    int regionEta = region->hwEta();
    int regionPhi = region->hwPhi();

    //int associatedSecondRegionEt =
    //  AssociatedSecondRegionEt(region->hwEta(), region->hwPhi(),
    //			       *subRegions);

    int tauEt=regionEt;
    int isoFlag=0;

    int highestNeighborEt=0;
    int highestNeighborTauVeto=1;
    int isEast=0;
    int isSouth=0;
    int isWest=0;
    int isNorth=0;
    int EastEt=0;
    int SouthEt=0;
    int WestEt=0;
    int NorthEt=0;
    int NEEt=0;
    int NWEt=0;
    int SEEt=0;
    int SWEt=0;

    //Find neighbor with highest Et and find energies of all neighboring regions
    for(CaloRegionBxCollection::const_iterator neighbor = regions.begin();
	neighbor != regions.end(); neighbor++) {
      
      int neighborPhi = neighbor->hwPhi();
      int neighborEta = neighbor->hwEta();
      int deltaPhi = regionPhi - neighborPhi;
      if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1)
	deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi
      
      int deltaEta = regionEta - neighborEta;
      
      if ((std::abs(deltaPhi) + std::abs(deltaEta)) > 0 && std::abs(deltaPhi) < 2 && std::abs(deltaEta) < 2) {
	
	int neighborEt = neighbor->hwPt();

	if (deltaEta==-1) {
	  if (deltaPhi==-1) NEEt=neighborEt;
	  else if (deltaPhi==0) {
	    isEast=1;
	    EastEt=neighborEt;
	  }
	  else SEEt=neighborEt;
	}
	else if (deltaEta==0) {
	  if (deltaPhi==-1) {
	    isNorth=1;
	    NorthEt=neighborEt;
	  }
	  if (deltaPhi==1) {
	    isSouth=1;
	    SouthEt=neighborEt;
	  }	
	}
	else {
	  if (deltaPhi==-1) NWEt=neighborEt;
	  else if (deltaPhi==0) {
	    isWest=1;
	    WestEt=neighborEt;
	  }
	  else SWEt=neighborEt;
	}
	
	if (!(std::abs(deltaPhi)==1 && std::abs(deltaEta==1))) {
	  
	  if (neighborEt > highestNeighborEt) {
	    highestNeighborEt = neighborEt;
	    highestNeighborTauVeto = neighbor->tauVeto();
	  }
	}
      }
    }
    
    if((tauEt > highestNeighborEt && (isEast==0 || isNorth==0)) || (tauEt >= highestNeighborEt && (isSouth==0 || isWest==0))) {

      if (highestNeighborEt >= tauNeighbourThreshold) tauEt += highestNeighborEt;
      
      if ((highestNeighborTauVeto == 0 && region->tauVeto() == 0) || tauEt > switchOffTauVeto) {

	double jetIsolation = JetIsolation(tauEt, region->hwEta(), region->hwPhi(), *unCorrJets);

	if (jetIsolation < tauRelativeJetIsolationCut || (tauEt >= switchOffTauIso && jetIsolation < tauRelativeJetIsolationLimit)
	    || (std::abs(jetIsolation - 999) < 0.1) ) isoFlag=1;
	
	ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > tauLorentz(0,0,0,0);
	
	l1t::Tau theTau(*&tauLorentz, tauEt, region->hwEta(), region->hwPhi(), isoFlag);
	
	if( tauEt >0) preGtTaus->push_back(theTau);
      }
    }
  }
  TauToGtScales(params_, preGtTaus, taus);

  delete subRegions;
  delete unCorrJets;
  delete preGtTaus;

  //the taus should be sorted, highest pT first.
  // do not truncate the tau list, GT converter handles that
  auto comp = [&](l1t::Tau i, l1t::Tau j)-> bool {
    return (i.hwPt() < j.hwPt() );
  };

  std::sort(taus->begin(), taus->end(), comp);
  std::reverse(taus->begin(), taus->end());
}





//  Compute jet isolation.
double l1t::Stage1Layer2TauAlgorithmImpPP::JetIsolation(int et, int ieta, int iphi,
							const std::vector<l1t::Jet> & jets) const {

  for(JetBxCollection::const_iterator jet = jets.begin();
      jet != jets.end(); jet++) {

    if (ieta==jet->hwEta() && iphi==jet->hwPhi()){

      //if (et >200)
      //  cout << "ISOL:  tauET: " << et << " jetET: " << jet->hwPt() << endl;

      double isolation = (double) (jet->hwPt() - et);
      return isolation/et;
    }
  }

  // set output
  return 999.;
}
