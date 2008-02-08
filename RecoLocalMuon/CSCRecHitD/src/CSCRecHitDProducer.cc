#include <RecoLocalMuon/CSCRecHitD/src/CSCRecHitDProducer.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCRecHitDBuilder.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripGainAvg.h>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBCrosstalk.h>
#include <CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h>
#include <CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>



CSCRecHitDProducer::CSCRecHitDProducer( const edm::ParameterSet& ps ) : iRun( 0 ), CSCstripGainAvg( 1 ) {
	
  stripDigiProducer_ = ps.getParameter<std::string>("CSCStripDigiProducer");
  wireDigiProducer_  = ps.getParameter<std::string>("CSCWireDigiProducer");
  useCalib           = ps.getUntrackedParameter<bool>("CSCUseCalibrations");
  debug              = ps.getUntrackedParameter<bool>("CSCDebug");
  
  recHitBuilder_     = new CSCRecHitDBuilder( ps ); // pass on the Parameter Settings
  stripGainAvg_      = new CSCStripGainAvg( ps ); // pass on the Parameter Settings
  
  // register what this produces
  produces<CSCRecHit2DCollection>();

}

CSCRecHitDProducer::~CSCRecHitDProducer()
{
  delete recHitBuilder_;
}


void  CSCRecHitDProducer::produce( edm::Event& ev, const edm::EventSetup& setup )
{

	
  // find the geometry & calibrations for this event & cache it in the builder

  // Geometry
  edm::ESHandle<CSCGeometry> h;
  setup.get<MuonGeometryRecord>().get( h );
  const CSCGeometry* pgeom = &*h;
  recHitBuilder_->setGeometry( pgeom );

  if ( useCalib ) {  
    // Strip gains
    edm::ESHandle<CSCDBGains> hGains;
    setup.get<CSCDBGainsRcd>().get( hGains );
    const CSCDBGains* pGains = hGains.product(); 
    // Strip X-talk
    edm::ESHandle<CSCDBCrosstalk> hCrosstalk;
    setup.get<CSCDBCrosstalkRcd>().get( hCrosstalk );
    const CSCDBCrosstalk* pCrosstalk = hCrosstalk.product();
    // Strip autocorrelation noise matrix
    edm::ESHandle<CSCDBNoiseMatrix> hNoiseMatrix;
    setup.get<CSCDBNoiseMatrixRcd>().get(hNoiseMatrix);
    const CSCDBNoiseMatrix* pNoiseMatrix = hNoiseMatrix.product();

    // Pass set of calibrations to builder all at once
    stripGainAvg_->setCalibration( pGains );
    if (ev.id().run() != iRun) {
      CSCstripGainAvg = stripGainAvg_->getStripGainAvg();
      iRun = ev.id().run();
    }
    recHitBuilder_->setCalibration( CSCstripGainAvg, pGains, pCrosstalk, pNoiseMatrix );
  }
	
  // Get the collections of strip & wire digis from event
  edm::Handle<CSCStripDigiCollection> stripDigis;
  edm::Handle<CSCWireDigiCollection> wireDigis;
  ev.getByLabel(stripDigiProducer_, "MuonCSCStripDigi", stripDigis);
  ev.getByLabel(wireDigiProducer_,  "MuonCSCWireDigi",  wireDigis);

  // Create empty collection of rechits
  std::auto_ptr<CSCRecHit2DCollection> oc( new CSCRecHit2DCollection );


  // Fill the CSCRecHit2DCollection
  recHitBuilder_->build( stripDigis.product(), wireDigis.product(),*oc);


  // Put collection in event
  if (debug) std::cout << "Will output rechits collection to event" << std::endl;
  ev.put( oc );

}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCRecHitDProducer);

