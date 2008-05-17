// $Id: EcalMonitorPrescaler.h,v 1.1 2007/04/27 16:53:27 dellaric Exp $

/*!
  \file EcalMonitorPrescaler.h
  \brief Ecal specific Prescaler 
  \author G. Della Ricca
  \version $Revision: 1.1 $
  \date $Date: 2007/04/27 16:53:27 $
*/

#ifndef EcalMonitorPrescaler_H
#define EcalMonitorPrescaler_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalMonitorPrescaler: public edm::EDFilter {

public:

explicit EcalMonitorPrescaler(edm::ParameterSet const&);
virtual ~EcalMonitorPrescaler();

virtual bool filter(edm::Event& e, edm::EventSetup const& c);
void endJob();

private:

edm::InputTag EcalRawDataCollection_;

int count_;

// accept one in n

int occupancyPrescaleFactor_;
int integrityPrescaleFactor_;
int statusflagsPrescaleFactor_;

int pedestalonlinePrescaleFactor_;

int laserPrescaleFactor_;
int ledPrescaleFactor_;
int pedestalPrescaleFactor_;
int testpulsePrescaleFactor_;

int triggertowerPrescaleFactor_;
int timingPrescaleFactor_;

int cosmicPrescaleFactor_;
int clusterPrescaleFactor_;

};

#endif // EcalMonitorPrescaler_H
