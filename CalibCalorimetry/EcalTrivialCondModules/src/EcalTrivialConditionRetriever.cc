//
// $Id: EcalTrivialConditionRetriever.cc,v 1.6 2006/06/23 14:36:42 meridian Exp $
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#include <iostream>

#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"


using namespace edm;

EcalTrivialConditionRetriever::EcalTrivialConditionRetriever( const edm::ParameterSet&  ps)
{

  // initilize parameters used to produce cond DB objects
  adcToGeVEBConstant_ = ps.getUntrackedParameter<double>("adcToGeVEBConstant",0.035);
  adcToGeVEEConstant_ = ps.getUntrackedParameter<double>("adcToGeVEEConstant",0.060);

  intercalibConstantMean_ = ps.getUntrackedParameter<double>("intercalibConstantMean",1.0);
  intercalibConstantSigma_ = ps.getUntrackedParameter<double>("intercalibConstantSigma",0.0);

  EBpedMeanX12_ = ps.getUntrackedParameter<double>("EBpedMeanX12", 200.);
  EBpedRMSX12_  = ps.getUntrackedParameter<double>("EBpedRMSX12",  1.10);
  EBpedMeanX6_  = ps.getUntrackedParameter<double>("EBpedMeanX6",  200.);
  EBpedRMSX6_   = ps.getUntrackedParameter<double>("EBpedRMSX6",   0.90);
  EBpedMeanX1_  = ps.getUntrackedParameter<double>("EBpedMeanX1",  200.);
  EBpedRMSX1_   = ps.getUntrackedParameter<double>("EBpedRMSX1",   0.62);

  EEpedMeanX12_ = ps.getUntrackedParameter<double>("EEpedMeanX12", 200.);
  EEpedRMSX12_  = ps.getUntrackedParameter<double>("EEpedRMSX12",  2.50);
  EEpedMeanX6_  = ps.getUntrackedParameter<double>("EEpedMeanX6",  200.);
  EEpedRMSX6_   = ps.getUntrackedParameter<double>("EEpedRMSX6",   2.00);
  EEpedMeanX1_  = ps.getUntrackedParameter<double>("EEpedMeanX1",  200.);
  EEpedRMSX1_   = ps.getUntrackedParameter<double>("EEpedRMSX1",   1.40);

  gainRatio12over6_ = ps.getUntrackedParameter<double>("gainRatio12over6", 2.0);
  gainRatio6over1_  = ps.getUntrackedParameter<double>("gainRatio6over1",  6.0);


  // default weights to reco amplitude after pedestal subtraction
  std::vector<double> vampl;
  vampl.push_back( -0.333 );
  vampl.push_back( -0.333 );
  vampl.push_back( -0.333 );
  vampl.push_back(  0. );
  vampl.push_back(  1. );
  vampl.push_back(  0. );
  vampl.push_back(  0. );
  vampl.push_back(  0. );
  vampl.push_back(  0. );
  vampl.push_back(  0. );
  std::vector<double> amplwgtv = ps.getUntrackedParameter< std::vector<double> >("amplWeights", vampl);
  assert(amplwgtv.size() == 10);
  for(std::vector<double>::const_iterator it = amplwgtv.begin(); it != amplwgtv.end(); ++it) {
    amplWeights_.push_back( EcalWeight(*it) );
  }

  std::vector<double> vamplAftGain;
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  1. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  vamplAftGain.push_back(  0. );
  std::vector<double> amplwgtvaft = ps.getUntrackedParameter< std::vector<double> >("amplWeightsAftGain", vamplAftGain);
  assert(amplwgtvaft.size() == 10);
  for(std::vector<double>::const_iterator it = amplwgtvaft.begin(); it != amplwgtvaft.end(); ++it) {
    amplWeightsAft_.push_back( EcalWeight(*it) );
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
  vjitt.push_back( 0.000 );
  vjitt.push_back( 1.000 );
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.000 );
  vjitt.push_back( 0.000 );
  std::vector<double> jittwgtv = ps.getUntrackedParameter< std::vector<double> >("jittWeights", vjitt);
  assert(jittwgtv.size() == 10);
  for(std::vector<double>::const_iterator it = jittwgtv.begin(); it != jittwgtv.end(); ++it) {
    jittWeights_.push_back( EcalWeight(*it) );
  }


  producedEcalPedestals_ = ps.getUntrackedParameter<bool>("producedEcalPedestals",true);
  producedEcalWeights_ = ps.getUntrackedParameter<bool>("producedEcalWeights",true);
  producedEcalIntercalibConstants_ = ps.getUntrackedParameter<bool>("producedEcalIntercalibConstants",true);
  producedEcalGainRatios_ = ps.getUntrackedParameter<bool>("producedEcalGainRatios",true);
  producedEcalADCToGeVConstant_ = ps.getUntrackedParameter<bool>("producedEcalADCToGeVConstant",true);

  verbose_  = ps.getUntrackedParameter<int>("verbose", 0);

  //Tell Producer what we produce
  //setWhatProduced(this);
  if (producedEcalPedestals_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalPedestals );
  if (producedEcalWeights_)
    {
      setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalWeightXtalGroups );
      setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalTBWeights );
    }
  if (producedEcalIntercalibConstants_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalIntercalibConstants );
  if (producedEcalGainRatios_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalGainRatios );
  if (producedEcalADCToGeVConstant_)
    setWhatProduced(this, &EcalTrivialConditionRetriever::produceEcalADCToGeVConstant );

  //Tell Finder what records we find
  if (producedEcalPedestals_)
    findingRecord<EcalPedestalsRcd>();
  if (producedEcalWeights_)
    {
      findingRecord<EcalWeightXtalGroupsRcd>();
      findingRecord<EcalTBWeightsRcd>();
    }
  if (producedEcalIntercalibConstants_)
    findingRecord<EcalIntercalibConstantsRcd>();
  if (producedEcalGainRatios_)
    findingRecord<EcalGainRatiosRcd>();
  if (producedEcalADCToGeVConstant_)
    findingRecord<EcalADCToGeVConstantRcd>();

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
  EcalPedestals::Item EBitem;
  EcalPedestals::Item EEitem;
  
  EBitem.mean_x1  = EBpedMeanX1_;
  EBitem.rms_x1   = EBpedRMSX1_;
  EBitem.mean_x6  = EBpedMeanX6_;
  EBitem.rms_x6   = EBpedRMSX6_;
  EBitem.mean_x12 = EBpedMeanX12_;
  EBitem.rms_x12  = EBpedRMSX12_;

  EEitem.mean_x1  = EEpedMeanX1_;
  EEitem.rms_x1   = EEpedRMSX1_;
  EEitem.mean_x6  = EEpedMeanX6_;
  EEitem.rms_x6   = EEpedRMSX6_;
  EEitem.mean_x12 = EEpedMeanX12_;
  EEitem.rms_x12  = EEpedRMSX12_;
  
  for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;
    for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      try 
	{
	  EBDetId ebdetid(iEta,iPhi);
	  peds->m_pedestals.insert(std::make_pair(ebdetid.rawId(),EBitem));
	}
      catch (...)
	{
	}
    }
  }
  
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try 
	{
	  EEDetId eedetidpos(iX,iY,1);
	  peds->m_pedestals.insert(std::make_pair(eedetidpos.rawId(),EEitem));
	  EEDetId eedetidneg(iX,iY,-1);
	  peds->m_pedestals.insert(std::make_pair(eedetidneg.rawId(),EEitem));
	}
      catch (...)
	{
	}
    }
  }

  //return std::auto_ptr<EcalPedestals>( peds );
  return peds;
}

std::auto_ptr<EcalWeightXtalGroups>
EcalTrivialConditionRetriever::produceEcalWeightXtalGroups( const EcalWeightXtalGroupsRcd& )
{
  std::auto_ptr<EcalWeightXtalGroups> xtalGroups = std::auto_ptr<EcalWeightXtalGroups>( new EcalWeightXtalGroups() );
  EcalXtalGroupId defaultGroupId(1);
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      try
	{
	  EBDetId ebid(ieta,iphi);
	  //	  xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId(ieta) ); // define rings in eta
	  xtalGroups->setValue(ebid.rawId(), defaultGroupId ); // define rings in eta
	}
      catch (...)
	{
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try 
	{
	  EEDetId eedetidpos(iX,iY,1);
	  xtalGroups->setValue(eedetidpos.rawId(), defaultGroupId ); 
	  EEDetId eedetidneg(iX,iY,-1);
	  xtalGroups->setValue(eedetidneg.rawId(), defaultGroupId ); 
	}
      catch (...)
	{
	}
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
      try
	{
	  EBDetId ebid(ieta,iphi);
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  ical->setValue( ebid.rawId(), intercalibConstantMean_ + r*intercalibConstantSigma_ );
	}
      catch (...)
	{
	}
    }
  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try 
	{
	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidpos(iX,iY,1);
	  ical->setValue( eedetidpos.rawId(), intercalibConstantMean_ + r*intercalibConstantSigma_ );
	  double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
	  EEDetId eedetidneg(iX,iY,-1);
	  ical->setValue( eedetidneg.rawId(), intercalibConstantMean_ + r1*intercalibConstantSigma_ );
	}
      catch (...)
	{
	}
    }
  }
  
  return ical;
}

std::auto_ptr<EcalGainRatios>
EcalTrivialConditionRetriever::produceEcalGainRatios( const EcalGainRatiosRcd& )
{
  std::auto_ptr<EcalGainRatios> gratio = std::auto_ptr<EcalGainRatios>( new EcalGainRatios() );
  EcalMGPAGainRatio gr;
  gr.setGain12Over6( gainRatio12over6_ );
  gr.setGain6Over1( gainRatio6over1_ );

  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      try
	{
	  EBDetId ebid(ieta,iphi);
	  gratio->setValue( ebid.rawId(), gr );
	}
      catch (...)
	{
	}
    }
  }
  
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try 
	{
	  EEDetId eedetidpos(iX,iY,1);
	  gratio->setValue( eedetidpos.rawId(), gr );
	  EEDetId eedetidneg(iX,iY,-1);
	  gratio->setValue( eedetidneg.rawId(), gr );
	}
      catch (...)
	{
	}
    }
  }

  return gratio;
}

std::auto_ptr<EcalADCToGeVConstant>
EcalTrivialConditionRetriever::produceEcalADCToGeVConstant( const EcalADCToGeVConstantRcd& )
{
  return std::auto_ptr<EcalADCToGeVConstant>( new EcalADCToGeVConstant(adcToGeVEBConstant_,adcToGeVEEConstant_) );
}

std::auto_ptr<EcalTBWeights>
EcalTrivialConditionRetriever::produceEcalTBWeights( const EcalTBWeightsRcd& )
{
  // create weights for the test-beam
  std::auto_ptr<EcalTBWeights> tbwgt = std::auto_ptr<EcalTBWeights>( new EcalTBWeights() );

  // create weights for each distinct group ID
  int nMaxTDC = 10;
//   for(int igrp=-EBDetId::MAX_IETA; igrp<=EBDetId::MAX_IETA; ++igrp) {
//     if(igrp==0) continue; 
  int igrp=1;
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
    mat2.push_back(amplWeightsAft_);
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
  //   }
  return tbwgt;
}
