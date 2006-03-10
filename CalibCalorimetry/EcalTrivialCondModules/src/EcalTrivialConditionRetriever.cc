//
// $Id: EcalTrivialConditionRetriever.cc,v 1.1 2006/03/02 17:03:44 rahatlou Exp $
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include <iostream>

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"


using namespace edm;

EcalTrivialConditionRetriever::EcalTrivialConditionRetriever( const edm::ParameterSet&  ps)
{

  // initilize parameters used to produce cond DB objects
  adcToGeVEBConstant_ = ps.getUntrackedParameter<double>("adcToGeVEBConstant",0.037);

  intercalibConstantMean_ = ps.getUntrackedParameter<double>("intercalibConstantMean",1.0);
  intercalibConstantSigma_ = ps.getUntrackedParameter<double>("intercalibConstantSigma",0.0);

  pedMeanX12_ = ps.getUntrackedParameter<double>("pedMeanX12", 198.80);
  pedRMSX12_  = ps.getUntrackedParameter<double>("pedRMSX12",  1.10);
  pedMeanX6_  = ps.getUntrackedParameter<double>("pedMeanX6",  199.40);
  pedRMSX6_   = ps.getUntrackedParameter<double>("pedRMSX6",   0.90);
  pedMeanX1_  = ps.getUntrackedParameter<double>("pedMeanX1",  201.00);
  pedRMSX1_   = ps.getUntrackedParameter<double>("pedRMSX1",   0.62);

  gainRatio12over6_ = ps.getUntrackedParameter<double>("gainRatio12over6", 2.0);
  gainRatio6over1_  = ps.getUntrackedParameter<double>("gainRatio6over1",  6.0);


  // default weights to reco amplitude after pedestal subtraction
  std::vector<double> vampl;
  vampl.push_back( -0.333 );
  vampl.push_back( -0.333 );
  vampl.push_back( -0.333 );
  vampl.push_back(  0.025 );
  vampl.push_back(  0.147 );
  vampl.push_back(  0.221 );
  vampl.push_back(  0.216 );
  vampl.push_back(  0.176 );
  vampl.push_back(  0.127 );
  vampl.push_back(  0.088 );
  std::vector<double> amplwgtv = ps.getUntrackedParameter< std::vector<double> >("amplWeights", vampl);
  assert(amplwgtv.size() == 10);
  for(std::vector<double>::const_iterator it = amplwgtv.begin(); it != amplwgtv.end(); ++it) {
    amplWeights_.push_back( EcalWeight(*it) );
  }

  // default weights to reco amplitude w/o pedestal subtraction
  std::vector<double> vped;
  vped.push_back( 0.333 );
  vped.push_back( 0.333 );
  vped.push_back( 0.333 );
  vped.push_back( 0.000 );
  vped.push_back( 0.000 );
  vped.push_back( 0.000 );
  vped.push_back( 0.000 );
  vped.push_back( 0.000 );
  vped.push_back( 0.000 );
  vped.push_back( 0.000 );
  std::vector<double> pedwgtv = ps.getUntrackedParameter< std::vector<double> >("pedWeights", vped);
  assert(pedwgtv.size() == 10);
  for(std::vector<double>::const_iterator it = pedwgtv.begin(); it != pedwgtv.end(); ++it) {
    pedWeights_.push_back( EcalWeight(*it) );
  }

  // default weights to reco jitter
  std::vector<double> vjitt;
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.800 );
  vjitt.push_back( 0.800 );
  vjitt.push_back( 0.800 );
  vjitt.push_back( 0.200 );
  vjitt.push_back( 0.300 );
  vjitt.push_back( 0.300 );
  vjitt.push_back( 0.300 );
  std::vector<double> jittwgtv = ps.getUntrackedParameter< std::vector<double> >("jittWeights", vjitt);
  assert(jittwgtv.size() == 10);
  for(std::vector<double>::const_iterator it = jittwgtv.begin(); it != jittwgtv.end(); ++it) {
    jittWeights_.push_back( EcalWeight(*it) );
  }

  verbose_  = ps.getUntrackedParameter<int>("verbose", 0);

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
  if(verbose_>=1) std::cout << "EcalTrivialConditionRetriever::setIntervalFor(): record key = " << rk.name() << "\ttime: " << iTime.time().value() << std::endl;
  //For right now, we will just use an infinite interval of validity
  oValidity = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime() );
}

//produce methods
std::auto_ptr<EcalPedestals>
EcalTrivialConditionRetriever::produceEcalPedestals( const EcalPedestalsRcd& ) {
  std::auto_ptr<EcalPedestals>  peds = std::auto_ptr<EcalPedestals>( new EcalPedestals() );
  EcalPedestals::Item item;
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      item.mean_x1  = pedMeanX1_;
      item.rms_x1   = pedRMSX1_;
      item.mean_x6  = pedMeanX6_;
      item.rms_x6   = pedRMSX6_;
      item.mean_x12 = pedMeanX12_;
      item.rms_x12  = pedRMSX1_;
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      EBDetId ebdetid(iEta,iPhi);
      peds->m_pedestals.insert(std::make_pair(ebdetid.rawId(),item));
    }
  }

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
      ical->setValue( ebid.rawId(), intercalibConstantMean_ + r*intercalibConstantSigma_ );
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
      EcalMGPAGainRatio gr;
      gr.setGain12Over6( gainRatio12over6_ );
      gr.setGain6Over1( gainRatio6over1_ );
      gratio->setValue( ebid.rawId(), gr );
    }
  }

  return gratio;
}

std::auto_ptr<EcalADCToGeVConstant>
EcalTrivialConditionRetriever::produceEcalADCToGeVConstant( const EcalADCToGeVConstantRcd& )
{
  return std::auto_ptr<EcalADCToGeVConstant>( new EcalADCToGeVConstant(adcToGeVEBConstant_) );
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

      if(verbose_>=1) {
        std::cout << "initial size of mat1: " << mat1.size() << std::endl;
        std::cout << "initial size of mat2: " << mat2.size() << std::endl;
      }

      // generate random numbers to use as weights
      /**
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
      **/

      // use values provided by user
      mat1.push_back(amplWeights_);
      mat1.push_back(pedWeights_);
      mat1.push_back(jittWeights_);

      // use same wights after gain switch for now
      mat2.push_back(amplWeights_);
      mat2.push_back(pedWeights_);
      mat2.push_back(jittWeights_);

      // fill the chi2 matrcies with random numbers
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

      if(verbose_>=1) {
        std::cout << "group: " << igrp << " TDC: " << itdc 
             << " mat1: " << mat1.size() << " mat2: " << mat2.size()
             << " mat3: " << mat3.size() << " mat4: " << mat4.size()
             << std::endl;
      }

      // put the weight in the container
      tbwgt->setValue(std::make_pair(igrp,itdc), wgt);
    }
  }
  return tbwgt;
}
