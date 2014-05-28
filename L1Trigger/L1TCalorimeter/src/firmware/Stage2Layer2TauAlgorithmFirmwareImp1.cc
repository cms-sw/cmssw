///
/// \class l1t::Stage2Layer2TauAlgorithmFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 jet algo

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2TauAlgorithmFirmware.h"



l1t::Stage2Layer2TauAlgorithmFirmwareImp1::Stage2Layer2TauAlgorithmFirmwareImp1(CaloParams* params) :
  params_(params)
{


}


l1t::Stage2Layer2TauAlgorithmFirmwareImp1::~Stage2Layer2TauAlgorithmFirmwareImp1() {


}


void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::processEvent(const std::vector<l1t::CaloCluster> & clusters,
							      std::vector<l1t::Tau> & taus) {

  for ( auto itr = clusters.cbegin(); itr != clusters.end(); ++itr ) {
    if( itr->isValid() ){
        math::XYZTLorentzVector p4;
        l1t::Tau tau( p4, itr->hwPt(), itr->hwEta(), itr->hwPhi(), 0);
        taus.push_back(tau);
    }
  }

}


void l1t::Stage2Layer2TauAlgorithmFirmwareImp1::merging(std::vector<Tau>& taus){
  //for ( auto itr = taus.begin(); itr != taus.end(); ++itr ) {
  //  l1t::Tau& tau = *itr;
  //}
}
