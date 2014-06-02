///
/// \class l1t::Stage1Layer2CentralityAlgorithm
///
/// \authors: Gian Michele Innocenti
///           R. Alex Barbieri
///
/// Description: Centrality Algorithm HI

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage1Layer2EtSumAlgorithmImp.h"
#include "L1Trigger/L1TCalorimeter/interface/PUSubtractionMethods.h"
#include "L1Trigger/L1TCalorimeter/interface/legacyGtHelper.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

l1t::Stage1Layer2CentralityAlgorithm::Stage1Layer2CentralityAlgorithm(CaloParamsStage1* params) : params_(params)
{

}


l1t::Stage1Layer2CentralityAlgorithm::~Stage1Layer2CentralityAlgorithm() {


}


void l1t::Stage1Layer2CentralityAlgorithm::processEvent(const std::vector<l1t::CaloRegion> & regions,
							const std::vector<l1t::CaloEmCand> & EMCands,
							      std::vector<l1t::EtSum> * etsums) {

  // ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > etLorentz(0,0,0,0);

  // // convert back to hardware ET
  // l1t::EtSum etMiss(*&etLorentz,EtSum::EtSumType::kMissingEt,MET/jetLsb ,0,iPhiET,0);
  // l1t::EtSum htMiss(*&etLorentz,EtSum::EtSumType::kMissingHt,MHT/jetLsb ,0,iPhiHT,0);
  // l1t::EtSum etTot (*&etLorentz,EtSum::EtSumType::kTotalEt,sumET/jetLsb,0,0,0);
  // l1t::EtSum htTot (*&etLorentz,EtSum::EtSumType::kTotalHt,sumHT/jetLsb ,0,0,0);

  // std::vector<l1t::EtSum> *preGtEtSums = new std::vector<l1t::EtSum>();

  // preGtEtSums->push_back(etMiss);
  // preGtEtSums->push_back(htMiss);
  // preGtEtSums->push_back(etTot);
  // preGtEtSums->push_back(htTot);

  // // All algorithms
  // EtSumToGtScales(params_, preGtEtSums, etsums);

  // delete subRegions;
  // delete preGtEtSums;

}
