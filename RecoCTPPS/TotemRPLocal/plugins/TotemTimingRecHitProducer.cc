/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingRecHitProducerAlgorithm.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

/// TOTEM/PPS timing detectors digi-to-rechits conversion module
class TotemTimingRecHitProducer : public edm::stream::EDProducer<>
{
  public:
    explicit TotemTimingRecHitProducer( const edm::ParameterSet& );

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    void produce( edm::Event&, const edm::EventSetup& ) override;

    /// Input digi collection
    edm::EDGetTokenT<edm::DetSetVector<TotemTimingDigi> > digiToken_;
    /// Digi-to-rechits transformation algorithm
    TotemTimingRecHitProducerAlgorithm algo_;
};

TotemTimingRecHitProducer::TotemTimingRecHitProducer( const edm::ParameterSet& iConfig ) :
  digiToken_( consumes<edm::DetSetVector<TotemTimingDigi> >( iConfig.getParameter<edm::InputTag>( "digiTag" ) ) ),
  algo_( iConfig )
{
  produces<edm::DetSetVector<TotemTimingRecHit> >();
}

void
TotemTimingRecHitProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr<edm::DetSetVector<TotemTimingRecHit> > pOut( new edm::DetSetVector<TotemTimingRecHit> );

  // get the digi collection
  edm::Handle<edm::DetSetVector<TotemTimingDigi> > digis;
  iEvent.getByToken( digiToken_, digis );

  // get the geometry
  edm::ESHandle<CTPPSGeometry> geometry;
  iSetup.get<VeryForwardRealGeometryRecord>().get( geometry );

  // produce the rechits collection
  algo_.build( geometry.product(), *digis, *pOut );

  iEvent.put( std::move( pOut ) );
}

void
TotemTimingRecHitProducer::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>( "digiTag", edm::InputTag( "totemTimingRawToDigi", "TotemTiming" ) )
    ->setComment( "input digis collection to retrieve" );
  desc.add<edm::FileInPath>( "calibrationFile", edm::FileInPath( "RecoCTPPS/TotemRPLocal/data/timing_offsets_ufsd_2018.dec18.cal.json" ) )
    ->setComment( "file with SAMPIC calibrations, ADC and INL; if empty or corrupted, no calibration will be applied" );
  desc.add<int>( "baselinePoints", 8 )
    ->setComment( "number of points to be used for the baseline" );
  desc.add<double>( "saturationLimit", 0.85 )
    ->setComment( "all signals with max > saturationLimit will be considered as saturated" );
  desc.add<double>( "cfdFraction", 0.3 )
    ->setComment( "fraction of the CFD" );
  desc.add<int>( "smoothingPoints", 20 )
    ->setComment( "number of points to be used for the smoothing using sinc (lowpass)" );
  desc.add<double>( "lowPassFrequency", 0.7 )
    ->setComment( "Frequency (in GHz) for CFD smoothing, 0 for disabling the filter" );
  desc.add<double>( "hysteresis", 5.e-3 )
    ->setComment( "hysteresis of the discriminator" );
  desc.add<bool>( "mergeTimePeaks", true )
      ->setComment( "if time peaks schould be merged" );

  descr.add( "totemTimingRecHits", desc );
}

DEFINE_FWK_MODULE( TotemTimingRecHitProducer );

