/****************************************************************************
 *
 * This is a part of PPS offline software.
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
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondRecHitProducerAlgorithm.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

class CTPPSDiamondRecHitProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSDiamondRecHitProducer( const edm::ParameterSet& );

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    void produce( edm::Event&, const edm::EventSetup& ) override;

    edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondDigi> > digiToken_;

    /// Label to timing calibration tag
    edm::ESInputTag timingCalibrationTag_;
    /// A watcher to detect timing calibration changes.
    edm::ESWatcher<PPSTimingCalibrationRcd> calibWatcher_;

    CTPPSDiamondRecHitProducerAlgorithm algo_;
};

CTPPSDiamondRecHitProducer::CTPPSDiamondRecHitProducer( const edm::ParameterSet& iConfig ) :
  digiToken_( consumes<edm::DetSetVector<CTPPSDiamondDigi> >( iConfig.getParameter<edm::InputTag>( "digiTag" ) ) ),
  timingCalibrationTag_( iConfig.getParameter<std::string>( "timingCalibrationTag" ) ),
  algo_( iConfig )
{
  produces<edm::DetSetVector<CTPPSDiamondRecHit> >();
}

void
CTPPSDiamondRecHitProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  auto pOut = std::make_unique<edm::DetSetVector<CTPPSDiamondRecHit> >();

  // get the digi collection
  edm::Handle<edm::DetSetVector<CTPPSDiamondDigi> > digis;
  iEvent.getByToken( digiToken_, digis );

  if ( !digis->empty() ) {
    if ( calibWatcher_.check( iSetup ) ) {
      edm::ESHandle<PPSTimingCalibration> hTimingCalib;
      iSetup.get<PPSTimingCalibrationRcd>().get( timingCalibrationTag_, hTimingCalib );
      algo_.setCalibration( *hTimingCalib );
    }
    // get the geometry
    edm::ESHandle<CTPPSGeometry> geometry;
    iSetup.get<VeryForwardRealGeometryRecord>().get( geometry );

    // produce the rechits collection
    algo_.build( *geometry, *digis, *pOut );
  }

  iEvent.put( std::move( pOut ) );
}

void
CTPPSDiamondRecHitProducer::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>( "digiTag", edm::InputTag( "ctppsDiamondRawToDigi", "TimingDiamond" ) )
    ->setComment( "input digis collection to retrieve" );
  desc.add<std::string>( "timingCalibrationTag", "GlobalTag:PPSDiamondTimingCalibration" )
    ->setComment( "input tag for timing calibrations retrieval" );
  desc.add<double>( "timeSliceNs", 25.0/1024.0 )
    ->setComment( "conversion constant between HPTDC timing bin size and nanoseconds" );
  desc.add<int>( "timeShift", 0 ) // to be determined at calibration level, will be replaced by a map channel id -> time shift
    ->setComment( "overall time offset to apply on all hits in all channels" );

  descr.add( "ctppsDiamondRecHits", desc );
}

DEFINE_FWK_MODULE( CTPPSDiamondRecHitProducer );
