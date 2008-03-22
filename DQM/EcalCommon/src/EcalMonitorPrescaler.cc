// $Id: EcalMonitorPrescaler.cc,v 1.6 2007/12/18 08:43:40 dellaric Exp $

/*!
  \file EcalMonitorPrescaler.cc
  \brief Ecal specific Prescaler
  \author G. Della Ricca
  \version $Revision: 1.6 $
  \date $Date: 2007/12/18 08:43:40 $
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include <DQM/EcalCommon/interface/EcalMonitorPrescaler.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalMonitorPrescaler::EcalMonitorPrescaler(ParameterSet const& ps) {

  count_ = 0;

  EcalRawDataCollection_ = ps.getParameter<InputTag>("EcalRawDataCollection");

  occupancyPrescaleFactor_ = ps.getUntrackedParameter<int>("occupancyPrescaleFactor" , 0);
  integrityPrescaleFactor_ = ps.getUntrackedParameter<int>("integrityPrescaleFactor", 0);

  cosmicPrescaleFactor_ = ps.getUntrackedParameter<int>("cosmicPrescaleFactor", 0);
  laserPrescaleFactor_ = ps.getUntrackedParameter<int>("laserPrescaleFactor", 0);
  pedestalonlinePrescaleFactor_ = ps.getUntrackedParameter<int>("pedestalonlinePrescaleFactor", 0);
  pedestalPrescaleFactor_ = ps.getUntrackedParameter<int>("pedestalPrescaleFactor", 0);
  testpulsePrescaleFactor_ = ps.getUntrackedParameter<int>("testpulsePrescaleFactor", 0);

  triggertowerPrescaleFactor_ = ps.getUntrackedParameter<int>("triggertowerPrescaleFactor" , 0);
  timingPrescaleFactor_ = ps.getUntrackedParameter<int>("timingPrescaleFactor" , 0);

  clusterPrescaleFactor_ = ps.getUntrackedParameter<int>("clusterPrescaleFactor", 0);

}
    
EcalMonitorPrescaler::~EcalMonitorPrescaler() { }

bool EcalMonitorPrescaler::filter(Event & e, EventSetup const&) {

  count_++;

  bool status = false;

  if ( occupancyPrescaleFactor_ ) {
    if ( count_ % occupancyPrescaleFactor_ == 0 ) status = true;
  }
  if ( integrityPrescaleFactor_ ) {
    if ( count_ % integrityPrescaleFactor_ == 0 ) status = true;
  }

  if ( pedestalonlinePrescaleFactor_ ) {
    if ( count_ % pedestalonlinePrescaleFactor_ == 0 ) status = true;
  }

  if ( triggertowerPrescaleFactor_ ) {
    if ( count_ % triggertowerPrescaleFactor_ == 0 ) status = true;
  }
  if ( timingPrescaleFactor_ ) {
    if ( count_ % timingPrescaleFactor_ == 0 ) status = true;
  }

  if ( clusterPrescaleFactor_ ) {
    if ( count_ % clusterPrescaleFactor_ == 0 ) status = true;
  }

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( dcch.getRunType() == EcalDCCHeaderBlock::COSMIC ) {
        if ( cosmicPrescaleFactor_ ) {
          if ( count_ % cosmicPrescaleFactor_ == 0 ) status = true;
        }
      }
      if ( dcch.getRunType() == EcalDCCHeaderBlock::LASER_STD ) {
        if ( laserPrescaleFactor_ ) { 
          if ( count_ % laserPrescaleFactor_ == 0 ) status = true;
        }
      }
      if ( dcch.getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ) {
        if ( pedestalPrescaleFactor_ ) { 
          if ( count_ % pedestalPrescaleFactor_ == 0 ) status = true;
        }
      }
      if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ) {
        if ( testpulsePrescaleFactor_ ) { 
          if ( count_ % testpulsePrescaleFactor_ == 0 ) status = true;
        }
      }

    }

  } else {

    LogWarning("EcalMonitorPrescaler") << EcalRawDataCollection_ << " not available";

  }

  return status;

}

void EcalMonitorPrescaler::endJob() { }

