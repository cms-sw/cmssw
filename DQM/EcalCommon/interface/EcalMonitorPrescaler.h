#ifndef EcalMonitorPrescaler_H
#define EcalMonitorPrescaler_H

/*!
  \file EcalMonitorPrescaler.h
  \brief Ecal specific Prescaler 
  \author G. Della Ricca
  \version $Revision: 1.7 $
  \date $Date: 2010/03/27 20:44:49 $
*/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalMonitorPrescaler: public edm::EDFilter {

public:

EcalMonitorPrescaler(const edm::ParameterSet& ps);
virtual ~EcalMonitorPrescaler();

bool filter(edm::Event& e, const edm::EventSetup& c);

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

int pedestaloffsetPrescaleFactor_;

int triggertowerPrescaleFactor_;
int timingPrescaleFactor_;

int cosmicPrescaleFactor_;

int physicsPrescaleFactor_;

int clusterPrescaleFactor_;

};

#endif // EcalMonitorPrescaler_H
