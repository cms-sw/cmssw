// $Id: EcalMonitorPrescaler.cc,v 1.2 2007/04/11 06:50:38 dellaric Exp $

/*!
  \file EcalMonitorPrescaler.cc
  \brief Ecal specific Prescaler
  \author G. Della Ricca
  \version $Revision: 1.2 $
  \date $Date: 2007/04/11 06:50:38 $
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include <DQM/EcalCommon/interface/EcalMonitorPrescaler.h>

using namespace cms;
using namespace edm;
using namespace std;

EcalMonitorPrescaler::EcalMonitorPrescaler(edm::ParameterSet const& ps) {

  count_ = 0;

    EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");

  occupancyPrescaleFactor_ = ps.getUntrackedParameter<int>("occupancyPrescaleFactor" , 1);
  integrityPrescaleFactor_ = ps.getUntrackedParameter<int>("integrityPrescaleFactor", 1);

  cosmicPrescaleFactor_ = ps.getUntrackedParameter<int>("cosmicPrescaleFactor", 1);
  laserPrescaleFactor_ = ps.getUntrackedParameter<int>("laserPrescaleFactor", 1);
  pedestalonlinePrescaleFactor_ = ps.getUntrackedParameter<int>("pedestalonlinePrescaleFactor", 1);
  pedestalPrescaleFactor_ = ps.getUntrackedParameter<int>("pedestalPrescaleFactor", 1);
  testpulsePrescaleFactor_ = ps.getUntrackedParameter<int>("testpulsePrescaleFactor", 1);

  triggertowerPrescaleFactor_ = ps.getUntrackedParameter<int>("triggertowerPrescaleFactor" , 1);
  timingPrescaleFactor_ = ps.getUntrackedParameter<int>("timingPrescaleFactor" , 1);

  clusterPrescaleFactor_ = ps.getUntrackedParameter<int>("clusterPrescaleFactor", 1);

}
    
EcalMonitorPrescaler::~EcalMonitorPrescaler() { }

bool EcalMonitorPrescaler::filter(edm::Event & e,edm::EventSetup const&) {

  count_++;

  bool status = false;

  if ( count_ % occupancyPrescaleFactor_ == 0 ) status = true;
  if ( count_ % integrityPrescaleFactor_ == 0 ) status = true;

  if ( count_ % pedestalonlinePrescaleFactor_ == 0 ) status = true;

  if ( count_ % triggertowerPrescaleFactor_ == 0 ) status = true;
  if ( count_ % timingPrescaleFactor_       == 0 ) status = true;

  if ( count_ % clusterPrescaleFactor_ == 0 ) status = true;

  try {

    Handle<EcalRawDataCollection> dcchs;
    e.getByLabel(EcalRawDataCollection_, dcchs);

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      EcalDCCHeaderBlock dcch = (*dcchItr);

      if ( dcch.getRunType() == EcalDCCHeaderBlock::COSMIC ) {
        if ( count_ % cosmicPrescaleFactor_    == 0 ) status = true;
      }
      if ( dcch.getRunType() == EcalDCCHeaderBlock::LASER_STD ) {
        if ( count_ % laserPrescaleFactor_     == 0 ) status = true;
      }
      if ( dcch.getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ) {
        if ( count_ % pedestalPrescaleFactor_  == 0 ) status = true;
      }
      if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ) {
        if ( count_ % testpulsePrescaleFactor_ == 0 ) status = true;
      }

    }

  } catch ( exception& ex) {

    LogWarning("EcalMonitorPrescaler") << EcalRawDataCollection_ << " not available";

  }

  return status;

}

void EcalMonitorPrescaler::endJob() { }

