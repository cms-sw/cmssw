///
/// \class l1t::Stage2PreProcessorFirmwareImp1
///
/// \author: Jim Brooke
///
/// Description: first iteration of stage 2 processing

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2PreProcessorFirmware.h"
#include "L1Trigger/L1TCalorimeter/interface/Stage2TowerCompressAlgorithmFirmware.h"

#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace std;

l1t::Stage2PreProcessorFirmwareImp1::Stage2PreProcessorFirmwareImp1(unsigned fwv, CaloParams* params) :
  m_params(params)
{

  m_towerAlgo = new Stage2TowerCompressAlgorithmFirmwareImp1(m_params);
  
}

l1t::Stage2PreProcessorFirmwareImp1::~Stage2PreProcessorFirmwareImp1()
{ 

};


//void l1t::Stage2PreProcessorFirmwareImp1::setParams(CaloParams* params) {
//  m_params = params;
//}


//need to switch to BXVector
void l1t::Stage2PreProcessorFirmwareImp1::processEvent(const std::vector<l1t::CaloTower> & inTowers,
						       std::vector<l1t::CaloTower> & outTowers) {

  m_towerAlgo->processEvent( inTowers, outTowers );

}


void l1t::Stage2PreProcessorFirmwareImp1::print(std::ostream& out) const {

  out << "Stage 2 Pre Processor" << std::endl;

  out << "  Tower compress algo  : " << (m_towerAlgo?1:0) << std::endl;

}


