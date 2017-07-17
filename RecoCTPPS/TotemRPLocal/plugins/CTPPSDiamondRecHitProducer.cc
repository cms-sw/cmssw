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

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondRecHitProducerAlgorithm.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"

class CTPPSDiamondRecHitProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSDiamondRecHitProducer( const edm::ParameterSet& );
    ~CTPPSDiamondRecHitProducer();

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;

    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondDigi> > digiToken_;

    /// A watcher to detect geometry changes.
    //edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher_;

    CTPPSDiamondRecHitProducerAlgorithm algo_;
};

CTPPSDiamondRecHitProducer::CTPPSDiamondRecHitProducer( const edm::ParameterSet& iConfig ) :
  digiToken_( consumes< edm::DetSetVector<CTPPSDiamondDigi> >( iConfig.getParameter<edm::InputTag>( "digiTag" ) ) ),
  algo_( iConfig )
{
  produces< edm::DetSetVector<CTPPSDiamondRecHit> >();
}

CTPPSDiamondRecHitProducer::~CTPPSDiamondRecHitProducer()
{}

void
CTPPSDiamondRecHitProducer::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< edm::DetSetVector<CTPPSDiamondRecHit> > pOut( new edm::DetSetVector<CTPPSDiamondRecHit> );

  // get the digi collection
  edm::Handle< edm::DetSetVector<CTPPSDiamondDigi> > digis;
  iEvent.getByToken( digiToken_, digis );

  // get the geometry
  edm::ESHandle<TotemRPGeometry> geometry;
  iSetup.get<VeryForwardRealGeometryRecord>().get( geometry );

  // produce the rechits collection
  algo_.build( geometry.product(), *( digis ), *( pOut ) );

  iEvent.put( std::move( pOut ) );
}

void
CTPPSDiamondRecHitProducer::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descr.addDefault( desc );
}

DEFINE_FWK_MODULE( CTPPSDiamondRecHitProducer );
