//
// $Id: $
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include <iostream>

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"


using namespace edm;

EcalTrivialConditionRetriever::EcalTrivialConditionRetriever( const edm::ParameterSet&  pset)
{
  //  std::cout<<"PedestalRetriever::PedestalRetriever"<<std::endl;
  //Tell Producer what we produce
  //setWhatProduced(this);
  setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalPedestals );
  setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalWeightXtalGroups );
  setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalIntercalibConstants );
  setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalGainRatios );
  setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalADCToGeVConstant );
  setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalTBWeights );

  //Tell Finder what records we find
  findingRecord<EcalPedestalsRcd>();
  findingRecord<EcalWeightXtalGroupsRcd>();
  findingRecord<EcalIntercalibConstantsRcd>();
  findingRecord<EcalGainRatiosRcd>();
  findingRecord<EcalADCToGeVConstantRcd>();
  findingRecord<EcalTBWeightsRcd>();
}

EcalTrivialConditionRetriever::~EcalTrivialConditionRetriever()
{
}

//
// member functions
//
void
EcalTrivialConditionRetriever::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& rk,
				   const edm::IOVSyncValue& iTime, 
				   edm::ValidityInterval& oValidity)
{
  std::cout << "EcalTrivialConditionRetriever::setIntervalFor(): record key = " << rk.name() << "\ttime: " << iTime.time().value() << std::endl;
  //For right now, we will just use an infinite interval of validity
  oValidity = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(),
				     edm::IOVSyncValue::endOfTime() );
}

//produce methods
std::auto_ptr<EcalPedestals>
EcalTrivialConditionRetriever::produceEcalPedestals( const EcalPedestalsRcd& )
{
  std::auto_ptr<EcalPedestals>  peds = std::auto_ptr<EcalPedestals>( new EcalPedestals() );
  EcalPedestals::Item item;
  int channelId;
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      channelId= (iEta-1)*EBDetId::MAX_IPHI+iPhi;
      int tt = channelId;
      if(tt%2 == 0) {
        item.mean_x1  =0.91;
        item.rms_x1   =0.17;
        item.mean_x6  =0.52;
        item.rms_x6   =0.03;
        item.mean_x12 =0.16;
        item.rms_x12  =0.05;
      } else {
        item.mean_x1  =0.50;
        item.rms_x1   =0.94;
        item.mean_x6  =0.72;
        item.rms_x6   =0.07;
        item.mean_x12 =0.87;
        item.rms_x12  =0.07;
      }
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      EBDetId ebdetid(iEta,iPhi);
      peds->m_pedestals.insert(std::make_pair(ebdetid.rawId(),item));
    }
  }
  //std::cout << "EcalTrivialConditionRetriever: EcalPedetsals with " << peds->m_pedestals.size() << " entries." << std::endl;

  //return std::auto_ptr<EcalPedestals>( peds );
  return peds;
}

std::auto_ptr<EcalWeightXtalGroups>
EcalTrivialConditionRetriever::produceEcalWeightXtalGroups( const EcalWeightXtalGroupsRcd& )
{
  std::auto_ptr<EcalWeightXtalGroups> xtalGroups = std::auto_ptr<EcalWeightXtalGroups>( new EcalWeightXtalGroups() );
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta,iphi);
      xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId(ieta) ); // define rings in eta
    }
  }
  return xtalGroups;
}

std::auto_ptr<EcalIntercalibConstants>
EcalTrivialConditionRetriever::produceEcalIntercalibConstants( const EcalIntercalibConstantsRcd& )
{
  std::auto_ptr<EcalIntercalibConstants>  ical = std::auto_ptr<EcalIntercalibConstants>( new EcalIntercalibConstants() );
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta,iphi);
      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
      ical->setValue( ebid.rawId(), 0.95 + r*0.1 );
    }
  }


  return ical;
}

std::auto_ptr<EcalGainRatios>
EcalTrivialConditionRetriever::produceEcalGainRatios( const EcalGainRatiosRcd& )
{
  std::auto_ptr<EcalGainRatios> gratio = std::auto_ptr<EcalGainRatios>( new EcalGainRatios() );
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta,iphi);
      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
      EcalMGPAGainRatio gr;
      gr.setGain12Over6( 1.9 + r*0.2 );
      gr.setGain6Over1( 5.9 + r*0.2 );

      gratio->setValue( ebid.rawId(), gr );
    }
  }

  return gratio;
}

std::auto_ptr<EcalADCToGeVConstant>
EcalTrivialConditionRetriever::produceEcalADCToGeVConstant( const EcalADCToGeVConstantRcd& )
{
  return std::auto_ptr<EcalADCToGeVConstant>( new EcalADCToGeVConstant(0.037) );
}

std::auto_ptr<EcalTBWeights>
EcalTrivialConditionRetriever::produceEcalTBWeights( const EcalTBWeightsRcd& )
{
  // create weights for the test-beam
  std::auto_ptr<EcalTBWeights> tbwgt = std::auto_ptr<EcalTBWeights>( new EcalTBWeights() );

  // create weights for each distinct group ID
  int nMaxTDC = 10;
  for(int igrp=-EBDetId::MAX_IETA; igrp<=EBDetId::MAX_IETA; ++igrp) {
    if(igrp==0) continue; 
    for(int itdc=1; itdc<=nMaxTDC; ++itdc) {
      // generate random number
      double r = (double)std::rand()/( double(RAND_MAX)+double(1) );

      // make a new set of weights
      EcalWeightSet wgt;
      //typedef std::vector< std::vector<EcalWeight> > EcalWeightSet::EcalWeightMatrix;
      EcalWeightSet::EcalWeightMatrix& mat1 = wgt.getWeightsBeforeGainSwitch();
      EcalWeightSet::EcalWeightMatrix& mat2 = wgt.getWeightsAfterGainSwitch();

      //std::cout << "initial size of mat1: " << mat1.size() << std::endl;
      //std::cout << "initial size of mat2: " << mat2.size() << std::endl;

      for(size_t i=0; i<3; ++i) {
      std::vector<EcalWeight> tv1, tv2;
        for(size_t j=0; j<10; ++j) {
          double ww = igrp*itdc*r + i*10. + j;
          //std::cout << "row: " << i << " col: " << j << " -  val: " << ww  << std::endl;
          tv1.push_back( EcalWeight(ww) );
          tv2.push_back( EcalWeight(100+ww) );
        }
        mat1.push_back(tv1);
        mat2.push_back(tv2);
      }

      // fill the chi2 matrcies
      r = (double)std::rand()/( double(RAND_MAX)+double(1) );
      EcalWeightSet::EcalWeightMatrix& mat3 = wgt.getChi2WeightsBeforeGainSwitch();
      EcalWeightSet::EcalWeightMatrix& mat4 = wgt.getChi2WeightsAfterGainSwitch();
      for(size_t i=0; i<10; ++i) {
        std::vector<EcalWeight> tv1, tv2;
        for(size_t j=0; j<10; ++j) {
          double ww = igrp*itdc*r + i*10. + j;
          tv1.push_back( EcalWeight(1000+ww) );
          tv2.push_back( EcalWeight(1000+100+ww) );
        }
        mat3.push_back(tv1);
        mat4.push_back(tv2);
      }

      /**
      cout << "group: " << igrp << " TDC: " << itdc 
           << " mat1: " << mat1.size() << " mat2: " << mat2.size()
           << " mat3: " << mat3.size() << " mat4: " << mat4.size()
           << endl;
      **/

      // put the weight in the container
      //tbwgt->setValue(igrp,itdc, wgt);
      tbwgt->setValue(std::make_pair(igrp,itdc), wgt);
      //cout << "size of EcalTBWeightsMap: " << tbwgt->getMap().size() << endl;
    }
  }
  return tbwgt;
}
