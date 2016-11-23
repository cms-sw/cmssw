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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

class CTPPSDiamondLocalTrackFitter : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSDiamondLocalTrackFitter( const edm::ParameterSet& );
    ~CTPPSDiamondLocalTrackFitter();

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;

    edm::EDGetTokenT< edm::View<CTPPSDiamondRecHit> > recHitsToken_;
};

CTPPSDiamondLocalTrackFitter::CTPPSDiamondLocalTrackFitter( const edm::ParameterSet& iConfig ) :
  recHitsToken_( consumes< edm::View<CTPPSDiamondRecHit> >( iConfig.getParameter<edm::InputTag>( "recHitsTag" ) ) )
{
  produces< std::vector<CTPPSDiamondLocalTrack> >();
}

CTPPSDiamondLocalTrackFitter::~CTPPSDiamondLocalTrackFitter()
{
}

void
CTPPSDiamondLocalTrackFitter::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< std::vector<CTPPSDiamondLocalTrack> > pOut( new std::vector<CTPPSDiamondLocalTrack> );

  edm::Handle< edm::View<CTPPSDiamondRecHit> > recHits;
  iEvent.getByToken( recHitsToken_, recHits );

  for ( unsigned short i = 0; i < recHits->size(); i++ ) {
    const edm::Ptr<CTPPSDiamondRecHit> recHit = recHits->ptrAt( i );
    // ...
    // if you want to insert a new CTPPSDiamondLocalTrack object to the output collection, simply use:
    //   pOut->push_back( CTPPSDiamondLocalTrack(...) );
  }


  iEvent.put( std::move( pOut ) );
}

void
CTPPSDiamondLocalTrackFitter::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descr.addDefault( desc );
}

DEFINE_FWK_MODULE( CTPPSDiamondLocalTrackFitter );
