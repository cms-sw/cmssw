/** \class EcalUncalibRecHitWorkerAnalFit
 *   produce ECAL uncalibrated rechits from dataframes with the analytical fit method
 *
 *  $Id: EcalUncalibRecHitWorkerAnalFit.cc,v 1.1 2008/12/10 01:52:42 ferriff Exp $
 *  $Date: 2008/12/10 01:52:42 $
 *  $Revision: 1.1 $
 *  \author Shahram Rahatlou, University of Rome & INFN, Sept 2005
 *
 */
#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerAnalFit.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
//#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "CLHEP/Matrix/Matrix.h"
//#include "CLHEP/Matrix/SymMatrix.h"
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

EcalUncalibRecHitWorkerAnalFit::EcalUncalibRecHitWorkerAnalFit(const edm::ParameterSet& ps) :
        EcalUncalibRecHitWorkerBaseClass( ps )
{
}


void
EcalUncalibRecHitWorkerAnalFit::set(const edm::EventSetup& es)
{
        // Gain Ratios
        LogDebug("EcalUncalibRecHitDebug") << "fetching gainRatios....";
        es.get<EcalGainRatiosRcd>().get(pRatio);
        LogDebug("EcalUncalibRecHitDebug") << "done." ;

        // fetch the pedestals from the cond DB via EventSetup
        LogDebug("EcalUncalibRecHitDebug") << "fetching pedestals....";
        es.get<EcalPedestalsRcd>().get( pedHandle );
        LogDebug("EcalUncalibRecHitDebug") << "done." ;
}

bool
EcalUncalibRecHitWorkerAnalFit::run( const edm::Event& evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result)
{
        using namespace edm;

        const EcalGainRatioMap & gainMap = pRatio.product()->getMap(); // map of gain ratios
        const EcalPedestalsMap & pedMap = pedHandle.product()->getMap(); // map of pedestals

        EcalPedestalsMapIterator pedIter; // pedestal iterator
        EcalPedestals::Item aped; // pedestal object for a single xtal

        EcalGainRatioMap::const_iterator gainIter; // gain iterator
        EcalMGPAGainRatio aGain; // gain object for a single xtal

        DetId detid( itdg->id() );

        // find pedestals for this channel
        //LogDebug("EcalUncalibRecHitDebug") << "looking up pedestal for crystal: " << itdg->id(); // FIXME
        pedIter = pedMap.find( detid );
        if( pedIter != pedMap.end() ) {
                aped = (*pedIter);
        } else {
                edm::LogError("EcalUncalibRecHitWorkerAnalFit") << "error!! could not find pedestals for channel: ";
                if ( detid.subdetId() == EcalBarrel ) {
                        edm::LogError("EcalUncalibRecHitWorkerAnalFit") << EBDetId( detid );
                } else {
                        edm::LogError("EcalUncalibRecHitWorkerAnalFit") << EEDetId( detid );
                }
                edm::LogError("EcalUncalibRecHitWorkerAnalFit") << "\n  no uncalib rechit will be made for this digi!";
                return false;
        }
        double pedVec[3];
        pedVec[0] = aped.mean_x12;
        pedVec[1] = aped.mean_x6;
        pedVec[2] = aped.mean_x1;


        // find gain ratios
        //LogDebug("EcalUncalibRecHitDebug") << "looking up gainRatios for crystal: " << itdg->id(); // FIXME
        gainIter = gainMap.find( detid );
        if( gainIter != gainMap.end() ) {
                aGain = (*gainIter);
        } else {
                edm::LogError("EcalUncalibRecHitWorkerAnalFit") << "error!! could not find gain ratios for channel: ";
                if ( detid.subdetId() == EcalBarrel ) {
                        edm::LogError("EcalUncalibRecHitWorkerAnalFit") << EBDetId( detid );
                } else {
                        edm::LogError("EcalUncalibRecHitWorkerAnalFit") << EEDetId( detid );
                }
                edm::LogError("EcalUncalibRecHitWorkerAnalFit") << "\n  no uncalib rechit will be made for this digi!";
                return false;
        }
        double gainRatios[3];
        gainRatios[0] = 1.;
        gainRatios[1] = aGain.gain12Over6();
        gainRatios[2] = aGain.gain6Over1()*aGain.gain12Over6();

        if ( detid.subdetId() == EcalBarrel ) {
                EcalUncalibratedRecHit aHit = algoEB_.makeRecHit(*itdg, pedVec, gainRatios, 0 ,0);
                result.push_back( aHit );
                if(aHit.amplitude()>0.) {
                        LogDebug("EcalUncalibRecHitInfo") << "EcalUncalibRecHitWorkerAnalFit: processed EBDataFrame with id: "
                                << EBDetId( detid )
                                << "\n" << "uncalib rechit amplitude: " << aHit.amplitude();
                }
        } else {
                EcalUncalibratedRecHit aHit = algoEE_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                result.push_back( aHit );
                if(aHit.amplitude()>0.) {
                        LogDebug("EcalUncalibRecHitInfo") << "EcalUncalibRecHitWorkerAnalFit: processed EEDataFrame with id: "
                                << EEDetId( detid ) << "\n"
                                << "uncalib rechit amplitude: " << aHit.amplitude();
                }
        }
        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerAnalFit, "EcalUncalibRecHitWorkerAnalFit" );
