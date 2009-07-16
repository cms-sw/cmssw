#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitWorker.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

ESRecHitWorker::ESRecHitWorker(const edm::ParameterSet& ps) :
        ESRecHitWorkerBaseClass( ps )
{
  //These should be taken from a DB
  int ESGain = ps.getParameter<int>("ESGain");
  int ESBaseline = ps.getParameter<int>("ESBaseline");
  double ESMIPADC = ps.getParameter<double>("ESMIPADC");
  double ESMIPkeV = ps.getParameter<double>("ESMIPkeV");
  
  algo_ = new ESRecHitSimAlgo(ESGain, ESBaseline, ESMIPADC, ESMIPkeV); 
}


ESRecHitWorker::~ESRecHitWorker()
{
        delete algo_;
}


void
ESRecHitWorker::set(const edm::EventSetup& es)
{
}

bool
ESRecHitWorker::run( const edm::Event & evt, 
                const ESDigiCollection::const_iterator & itdg, 
                ESRecHitCollection & result )
{
        result.push_back( algo_->reconstruct(*itdg) );
        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( ESRecHitWorkerFactory, ESRecHitWorker, "ESRecHitWorker" );
