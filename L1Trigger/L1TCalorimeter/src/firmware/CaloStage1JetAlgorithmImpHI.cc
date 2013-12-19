///
/// \class l1t::CaloStage1JetAlgorithmImpHI
///
///
/// \author: R. Alex Barbieri MIT
///

// This example implements algorithm version 1 and 2.

#include "CaloStage1JetAlgorithmImp.h"

// Taken from UCT code. Might not be appropriate. Refers to legacy L1 objects.
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//#include "DataFormats/Candidate/interface/LeafCandidate.h"

using namespace std;
using namespace l1t;

CaloStage1JetAlgorithmImpHI::CaloStage1JetAlgorithmImpHI(/*const CaloParams & dbPars*/)/* : db(dbPars)*/
  : regionLSB_(0.5) {}

CaloStage1JetAlgorithmImpHI::~CaloStage1JetAlgorithmImpHI(){};

// double regionPhysicalEt(const l1t::CaloRegion& cand) {
//   return regionLSB_*cand.hwPt();
// }
// int deltaGctPhi(const CaloRegion & region, const CaloRegion & neighbor)
// {
//   int phi1 = region.hwPhi();
//   int phi2 = neighbor.hwPhi();
//   int diff = phi1 - phi2;
//   if (std::abs(phi1 - phi2) == 18-1) { //18 regions in phi
//     diff = -diff/std::abs(diff);
//   }
//   return diff;
// }

void puSubtraction(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> subRegions);
// void makeJets(const std::vector<l1t::CaloRegion> & regions, int puLevelHI[],
// 	      std::vector<l1t::Jet> & jets);


void CaloStage1JetAlgorithmImpHI::processEvent(const std::vector<l1t::CaloRegion> & regions,
					       std::vector<l1t::Jet> & jets){

  std::vector<l1t::CaloRegion> subRegions;
  puSubtraction(regions, subRegions);
  //makeJets(regions, puLevelHI, jets);
  findJets(subRegions, jets);

  // std::vector<l1t::CaloRegion>::const_iterator incell;
  // for (incell = regions.begin(); incell != regions.end(); ++incell){
  //   //do nothing for now
  // }

}

// NB PU is not in the physical scale!!  Needs to be multiplied by regionLSB
void puSubtraction(const std::vector<l1t::CaloRegion> & regions, std::vector<l1t::CaloRegion> subRegions)
{
  int puLevelHI[L1CaloRegionDetId::N_ETA];
  double r_puLevelHI[L1CaloRegionDetId::N_ETA];
  int etaCount[L1CaloRegionDetId::N_ETA];
  for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
  {
    puLevelHI[i] = 0;
    r_puLevelHI[i] = 0.0;
    etaCount[i] = 0;
  }

  for(vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++){
    r_puLevelHI[region->hwEta()] += region->hwPt();
    etaCount[region->hwEta()]++;
  }

  for(unsigned i = 0; i < L1CaloRegionDetId::N_ETA; ++i)
  {
    puLevelHI[i] = floor(r_puLevelHI[i]/etaCount[i] + 0.5);
  }

  for(vector<CaloRegion>::const_iterator region = regions.begin(); region!= regions.end(); region++){
    int subPt = std::max(0, region->hwPt() - puLevelHI[region->hwEta()]);
    int subEta = region->hwEta();
    int subPhi = region->hwPhi();

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *lorentz =
      new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    CaloRegion newSubRegion(*lorentz, 0, 0, subPt, subEta, subPhi, 0, 0, 0);
    subRegions.push_back(newSubRegion);
  }
}

// void makeJets(const std::vector<l1t::CaloRegion> & regions, int puLevelHI[],
// 	      std::vector<l1t::Jet> & jets)
// {
//   for(vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {
//     double regionET = regionPhysicalEt(*region);
//     regionET = std::max(0.,regionET -
// 			(puLevelHI[region->hwEta()]*regionLSB_));
//     double neighborN_et = 0;
//     double neighborS_et = 0;
//     double neighborE_et = 0;
//     double neighborW_et = 0;
//     double neighborNE_et = 0;
//     double neighborSW_et = 0;
//     double neighborNW_et = 0;
//     double neighborSE_et = 0;
//     unsigned int nNeighbors = 0;
//     for(vector<CaloRegion>::const_iterator neighbor = regions.begin(); neighbor != regions.end(); neighbor++) {
//       double neighborET = regionPhysicalEt(*neighbor);
//       if(deltaGctPhi(*region, *neighbor) == 1 &&
// 	 (region->hwEta()    ) == neighbor->hwEta()) {
// 	neighborN_et = std::max(0.,neighborET -
// 				(puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == -1 &&
// 	      (region->hwEta()    ) == neighbor->hwEta()) {
// 	neighborS_et = std::max(0.,neighborET -
// 				(puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == 0 &&
// 	      (region->hwEta() + 1) == neighbor->hwEta()) {
// 	neighborE_et = std::max(0.,neighborET -
// 				(puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == 0 &&
// 	      (region->hwEta() - 1) == neighbor->hwEta()) {
// 	neighborW_et = std::max(0.,neighborET -
// 				(puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == 1 &&
// 	      (region->hwEta() + 1) == neighbor->hwEta()) {
// 	neighborNE_et = std::max(0.,neighborET -
// 				 (puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == -1 &&
// 	      (region->hwEta() - 1) == neighbor->hwEta()) {
// 	neighborSW_et = std::max(0.,neighborET -
// 				 (puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == 1 &&
// 	      (region->hwEta() - 1) == neighbor->hwEta()) {
// 	neighborNW_et = std::max(0.,neighborET -
// 				 (puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//       else if(deltaGctPhi(*region, *neighbor) == -1 &&
// 	      (region->hwEta() + 1) == neighbor->hwEta()) {
// 	neighborSE_et = std::max(0.,neighborET -
// 				 (puLevelHI[neighbor->hwEta()]*regionLSB_));
// 	nNeighbors++;
// 	continue;
//       }
//     }
//     if(regionET > neighborN_et &&
//        regionET > neighborNW_et &&
//        regionET > neighborW_et &&
//        regionET > neighborSW_et &&
//        regionET >= neighborNE_et &&
//        regionET >= neighborE_et &&
//        regionET >= neighborSE_et &&
//        regionET >= neighborS_et) {
//       unsigned int jetET = regionET +
// 	neighborN_et + neighborS_et + neighborE_et + neighborW_et +
// 	neighborNE_et + neighborSW_et + neighborSE_et + neighborNW_et;
//       /*
// 	int jetPhi = region->hwPhi() * 4 +
// 	( - 2 * (neighborS_et + neighborSE_et + neighborSW_et)
// 	+ 2 * (neighborN_et + neighborNE_et + neighborNW_et) ) / jetET;
// 	if(jetPhi < 0) {

// 	}
// 	else if(jetPhi >= ((int) N_JET_PHI)) {
// 	jetPhi -= N_JET_PHI;
// 	}
// 	int jetEta = region->hwEta() * 4 +
// 	( - 2 * (neighborW_et + neighborNW_et + neighborSW_et)
// 	+ 2 * (neighborE_et + neighborNE_et + neighborSE_et) ) / jetET;
// 	if(jetEta < 0) jetEta = 0;
// 	if(jetEta >= ((int) N_JET_ETA)) jetEta = N_JET_ETA - 1;
//       */
//       // Temporarily use the region granularity -- we will try to improve as above when code is debugged
//       int jetPhi = region->hwPhi();
//       int jetEta = region->hwEta();

//       bool neighborCheck = (nNeighbors == 8);
//       // On the eta edge we only expect 5 neighbors
//       if (!neighborCheck && (jetEta == 0 || jetEta == 21) && nNeighbors == 5)
// 	neighborCheck = true;

//       if (!neighborCheck) {
// 	std::cout << "phi: " << jetPhi << " eta: " << jetEta << " n: " << nNeighbors << std::endl;
// 	assert(false);
//       }

//       ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *jetLorentz = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
//       l1t::Jet theJet(*jetLorentz, jetET, jetEta, jetPhi);
//       //l1t::Jet theJet(0, jetET, jetEta, jetPhi);

//       jets.push_back(theJet);
//     }
//   }
//   //jets.sort();
//   //jets.reverse();
// }
