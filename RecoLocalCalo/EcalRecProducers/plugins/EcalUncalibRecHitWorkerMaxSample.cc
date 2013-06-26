/** \class EcalMaxSampleUncalibRecHitProducer
 *   produce ECAL uncalibrated rechits from dataframes 
 *
 *  $Id: EcalUncalibRecHitWorkerMaxSample.cc,v 1.1 2008/12/10 01:52:42 ferriff Exp $
 *  $Date: 2008/12/10 01:52:42 $
 *  $Revision: 1.1 $
 *  \author G. Franzoni, E. Di Marco
 *
 */
#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerMaxSample.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

EcalUncalibRecHitWorkerMaxSample::EcalUncalibRecHitWorkerMaxSample(const edm::ParameterSet& ps) :
        EcalUncalibRecHitWorkerBaseClass( ps )
{
}


void
EcalUncalibRecHitWorkerMaxSample::set(const edm::EventSetup& es)
{
}

bool
EcalUncalibRecHitWorkerMaxSample::run( const edm::Event & evt, 
                const EcalDigiCollection::const_iterator & itdg, 
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

        if ( detid.subdetId() == EcalBarrel ) {
                result.push_back( ebAlgo_.makeRecHit(*itdg, 0, 0, 0, 0 ) );
        } else {
                result.push_back( eeAlgo_.makeRecHit(*itdg, 0, 0, 0, 0 ) );
        }

        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMaxSample, "EcalUncalibRecHitWorkerMaxSample" );
