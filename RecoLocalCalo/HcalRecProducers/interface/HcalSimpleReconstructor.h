#ifndef HCALSIMPLERECONSTRUCTOR_H
#define HCALSIMPLERECONSTRUCTOR_H 1

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"


    /** \class HcalSimpleReconstructor
	
    $Date: 2005/09/12 20:00:43 $
    $Revision: 1.2 $
    \author J. Mans - Minnesota
    */
    class HcalSimpleReconstructor : public edm::EDProducer {
    public:
      explicit HcalSimpleReconstructor(const edm::ParameterSet& ps);
      virtual ~HcalSimpleReconstructor();
      virtual void produce(edm::Event& e, const edm::EventSetup& c);
    private:
      HcalSimpleRecAlgo reco_;
      HcalSubdetector subdet_;
    };

#endif
