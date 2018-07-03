///
/// \class l1t::Stage1Layer2FlowAlgorithm
///
/// \authors: Maxime Guilbaud
///           R. Alex Barbieri
///
/// Description: Flow Algorithm HI

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFRingSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

l1t::Stage1Layer2FlowAlgorithm::Stage1Layer2FlowAlgorithm(CaloParamsHelper* params) : params_(params)
{
 //now do what ever initialization is needed
 //Converting phi to be as it is define at GCT (-pi to pi instead of 0 to 2*pi)
 for(unsigned int i = 0; i < L1CaloRegionDetId::N_PHI; i++) {
   if(i < 10){
       sinPhi.push_back(sin(2. * 3.1415927 * i * 1.0 / L1CaloRegionDetId::N_PHI));
       cosPhi.push_back(cos(2. * 3.1415927 * i * 1.0 / L1CaloRegionDetId::N_PHI));
   }
   else {
       sinPhi.push_back(sin(-3.1415927 + 2. * 3.1415927 * (i-9) * 1.0 / L1CaloRegionDetId::N_PHI));
       cosPhi.push_back(cos(-3.1415927 + 2. * 3.1415927 * (i-9) * 1.0 / L1CaloRegionDetId::N_PHI));
   }
 }
}


l1t::Stage1Layer2FlowAlgorithm::~Stage1Layer2FlowAlgorithm() {}


void l1t::Stage1Layer2FlowAlgorithm::processEvent(const std::vector<l1t::CaloRegion> & regions,
						  const std::vector<l1t::CaloEmCand> & EMCands,
						  const std::vector<l1t::Tau> * taus,
						  l1t::CaloSpare * spare) {
  double q2x = 0;
  double q2y = 0;
  double regionET=0.;

  for(std::vector<CaloRegion>::const_iterator region = regions.begin(); region != regions.end(); region++) {

    int ieta=region->hwEta();
    if (ieta > 3 && ieta < 18) {
      continue;
    }

    int iphi=region->hwPhi();
    regionET=region->hwPt();

    q2x+= regionET * cosPhi[iphi];
    q2y+= regionET * sinPhi[iphi];
  }
  int HFq2 = q2x*q2x+q2y*q2y;
  //double psi2 = 0.5 * atan(q2y/q2x);

  // ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > dummy(0,0,0,0);

  // l1t::CaloSpare V2 (*&dummy,CaloSpare::CaloSpareType::V2,(int)HFq2,0,0,0);

  // spares->push_back(V2);
  spare->SetRing(1, HFq2&0x7);
}
