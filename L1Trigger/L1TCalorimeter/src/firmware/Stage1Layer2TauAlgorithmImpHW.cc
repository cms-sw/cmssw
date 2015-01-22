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
#include "L1Trigger/L1TCalorimeter/interface/HardwareSortingMethods.h"


using namespace std;
using namespace l1t;


Stage1Layer2TauAlgorithmImpHW::Stage1Layer2TauAlgorithmImpHW(CaloParamsStage1* params) : params_(params)
{
}

Stage1Layer2TauAlgorithmImpHW::~Stage1Layer2TauAlgorithmImpHW(){};




void l1t::Stage1Layer2TauAlgorithmImpHW::processEvent(const std::vector<l1t::CaloEmCand> & EMCands,
						      const std::vector<l1t::CaloRegion> & regions,
						      std::vector<l1t::Tau> * isoTaus,
						      std::vector<l1t::Tau> * taus) {

  double towerLsb = params_->towerLsbSum();

  std::string regionPUSType = params_->regionPUSType();
  std::vector<double> regionPUSParams = params_->regionPUSParams();
  int tauSeedThreshold= floor( params_->tauSeedThreshold()/towerLsb + 0.5); // convert GeV to HW units
  int tauNeighbourThreshold= floor( params_->tauNeighbourThreshold()/towerLsb + 0.5); // convert GeV to HW units
  int tauMaxPtTauVeto = floor( params_->tauMaxPtTauVeto()/towerLsb + 0.5);
  int tauMinPtJetIsolationB = floor( params_->tauMinPtJetIsolationB()/towerLsb + 0.5);
  int isoTauEtaMin = params_->isoTauEtaMin();
  int isoTauEtaMax = params_->isoTauEtaMax();
  double tauMaxJetIsolationB = params_->tauMaxJetIsolationB();
  double tauMaxJetIsolationA = params_->tauMaxJetIsolationA();

  std::vector<l1t::CaloRegion> *subRegions = new std::vector<l1t::CaloRegion>();


  //Region Correction will return uncorrected subregions if
  //regionPUSType is set to None in the config
  RegionCorrection(regions, subRegions, regionPUSParams, regionPUSType);



  // ----- need to cluster jets in order to compute jet isolation ----
  std::vector<l1t::Jet> *unCorrJets = new std::vector<l1t::Jet>();
  TwelveByTwelveFinder(0, subRegions, unCorrJets);

  std::vector<l1t::Tau> *preGtTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *preSortTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *sortedTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *preGtIsoTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *preSortIsoTaus = new std::vector<l1t::Tau>();
  std::vector<l1t::Tau> *sortedIsoTaus = new std::vector<l1t::Tau>();

  for(CaloRegionBxCollection::const_iterator region = subRegions->begin();
      region != subRegions->end(); region++) {
    if(region->hwEta() < 4 || region->hwEta() > 17)
      continue;

    int regionEt = region->hwPt();
    if(regionEt < tauSeedThreshold) continue;

    int regionEta = region->hwEta();
    int regionPhi = region->hwPhi();

    int tauEt = regionEt;
    int isoFlag = 0;  // is 1 if it passes the relative jet iso requirement
    int quality = 0;  //doesn't really mean anything and isn't used

    int highestNeighborEt=-1;
    int highestNeighborEta=-1;
    int highestNeighborPhi=-1;
    int highestNeighborTauVeto=-1;

    //Find neighbor with highest Et
    for(CaloRegionBxCollection::const_iterator neighbor = subRegions->begin();
	neighbor != subRegions->end(); neighbor++) {

      int neighborPhi = neighbor->hwPhi();
      int neighborEta = neighbor->hwEta();
      if(neighborEta < 4 || neighborEta > 17)
	continue;

      int deltaPhi = regionPhi - neighborPhi;
      if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1)
	deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

      deltaPhi = std::abs(deltaPhi);
      int deltaEta = std::abs(regionEta - neighborEta);

      if (deltaPhi + deltaEta > 0 && deltaPhi + deltaEta < 2) {  //nondiagonal neighbors
	if (neighbor->hwPt() > highestNeighborEt) {
	  highestNeighborEt = neighbor->hwPt();
	  highestNeighborEta = neighbor->hwEta();
	  highestNeighborPhi = neighbor->hwPhi();
	  highestNeighborTauVeto = neighbor->hwQual() & 0x1; // tauVeto should be the first bit of quality integer
	}
      }
    }

    string NESW = findNESW(regionEta, regionPhi, highestNeighborEta, highestNeighborPhi);

    if((tauEt > highestNeighborEt && (NESW=="isEast" || NESW=="isNorth"))
       || (tauEt >= highestNeighborEt && (NESW=="isSouth" || NESW=="isWest"))
       || highestNeighborEt == 0 ) {

      if (highestNeighborEt >= tauNeighbourThreshold) tauEt += highestNeighborEt;

      int regionTauVeto = region->hwQual() & 0x1;  // tauVeto should be the first bit of quality integer

      double jetIsolation = JetIsolation(tauEt, region->hwEta(), region->hwPhi(), *unCorrJets);
      if (region->hwEta() >= isoTauEtaMin && region->hwEta() <= isoTauEtaMax ){
	if ((highestNeighborTauVeto == 0 && regionTauVeto == 0) || tauEt > tauMaxPtTauVeto) {
	  if (jetIsolation < tauMaxJetIsolationA || (tauEt >= tauMinPtJetIsolationB && jetIsolation < tauMaxJetIsolationB)
	      || (std::abs(jetIsolation - 999.) < 0.1) ) isoFlag=1;
	}
      }
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > tauLorentz(0,0,0,0);

      l1t::Tau theTau(*&tauLorentz, tauEt, region->hwEta(), region->hwPhi(), quality, isoFlag);

      preGtTaus->push_back(theTau);
      if(isoFlag)
	preGtIsoTaus->push_back(theTau);
    }
  }

  TauToGtPtScales(params_, preGtTaus, preSortTaus);
  TauToGtPtScales(params_, preGtIsoTaus, preSortIsoTaus);

  SortTaus(preSortTaus, sortedTaus);
  SortTaus(preSortIsoTaus, sortedIsoTaus);

  TauToGtEtaScales(params_, sortedTaus, taus);
  TauToGtEtaScales(params_, sortedIsoTaus, isoTaus);

  // set all filler taus to have isolation bit set
  for(std::vector<l1t::Tau>::iterator iTau = isoTaus->begin(); iTau != isoTaus->end(); ++iTau)
    iTau->setHwIso(1);

  delete subRegions;
  delete unCorrJets;
  delete preGtTaus;
  delete preSortTaus;
  delete sortedTaus;
  delete preGtIsoTaus;
  delete preSortIsoTaus;
  delete sortedIsoTaus;

  const bool verbose = true;
  if(verbose)
  {
    std::cout << "Taus" << std::endl;
    for(std::vector<l1t::Tau>::const_iterator iTau = taus->begin(); iTau != taus->end(); ++iTau)
    {
      unsigned int packed = pack15bits(iTau->hwPt(), iTau->hwEta(), iTau->hwPhi());
      std::cout << bitset<15>(packed).to_string() << std::endl;
    }
    std::cout << "Isolated Taus" << std::endl;
    for(std::vector<l1t::Tau>::const_iterator iTau = isoTaus->begin(); iTau != isoTaus->end(); ++iTau)
    {
      unsigned int packed = pack15bits(iTau->hwPt(), iTau->hwEta(), iTau->hwPhi());
      std::cout << bitset<15>(packed).to_string() << std::endl;
    }
  }
}





//  Compute jet isolation.
double l1t::Stage1Layer2TauAlgorithmImpHW::JetIsolation(int et, int ieta, int iphi,
							const std::vector<l1t::Jet> & jets) const {

  for(JetBxCollection::const_iterator jet = jets.begin();
      jet != jets.end(); jet++) {

    if (ieta==jet->hwEta() && iphi==jet->hwPhi()){

      double isolation = (double) (jet->hwPt() - et);
      return isolation/et;
    }
  }

  // set output
  return 999.;
}


//  Find if the neighbor with the highest Et is N, E, S, or W
string l1t::Stage1Layer2TauAlgorithmImpHW::findNESW(int ieta, int iphi, int neta, int nphi) const {

  int deltaPhi = iphi - nphi;
  if (std::abs(deltaPhi) == L1CaloRegionDetId::N_PHI-1)
    deltaPhi = -deltaPhi/std::abs(deltaPhi); //18 regions in phi

  int deltaEta = ieta - neta;

  if ((std::abs(deltaPhi) +  std::abs(deltaEta)) < 2) {
    if (deltaEta==-1) {
      return "isEast";
    }
    else if (deltaEta==0) {
      if (deltaPhi==-1) {
	return "isNorth";
      }
      if (deltaPhi==1) {
	return "isSouth";
      }
    }
    else {
      return "isWest";
    }
  }

  return "999";

}
