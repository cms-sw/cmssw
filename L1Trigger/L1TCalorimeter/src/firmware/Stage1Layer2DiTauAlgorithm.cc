///
/// \class l1t::Stage1Layer2DiTauAlgorithm
///
/// \authors:
///           R. Alex Barbieri
///
/// Description: DiTau Algorithm

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2HFRingSumAlgorithmImp.h"
//#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
//#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"

l1t::Stage1Layer2DiTauAlgorithm::Stage1Layer2DiTauAlgorithm(CaloParamsStage1* params) : params_(params)
{
}


l1t::Stage1Layer2DiTauAlgorithm::~Stage1Layer2DiTauAlgorithm() {}


void l1t::Stage1Layer2DiTauAlgorithm::processEvent(const std::vector<l1t::CaloRegion> & regions,
						   const std::vector<l1t::CaloEmCand> & EMCands,
						   const std::vector<l1t::Tau> * taus,
						   std::vector<l1t::CaloSpare> * spares) {

  int outputBits =0; // we have 12 bits to fill as output
  int overThreshold[32] = {0};

  for(std::vector<l1t::Tau>::const_iterator itTau = taus->begin();
      itTau != taus->end(); ++itTau){
    if( !itTau->hwIso() ) continue;

    for(int i = 0; i < 32; ++i)
    {
      if( itTau->hwPt() > i ) //taus have 4GeV LSB right now, assumed!!
	overThreshold[i]++;
    }
  }

  for(int i = 0; i < 32; ++i)
    if(overThreshold[i] > 1) outputBits = i;
  
  ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > dummy(0,0,0,0);

  l1t::CaloSpare ditau (*&dummy,CaloSpare::CaloSpareType::Tau,outputBits,0,0,0);

  spares->push_back(ditau);
}
