// This is CSCRecHitDBuilder.cc

// Copied from RecHitB. Possible changes

#include <RecoLocalMuon/CSCRecHitD/interface/CSCRecHitDBuilder.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCHitFromStripOnly.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCHitFromWireOnly.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCMake2DRecHit.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCWireHitCollection.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCStripHitCollection.h>
#include <RecoLocalMuon/CSCRecHitD/interface/CSCRangeMapForRecHit.h>
 
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <CondFormats/CSCObjects/interface/CSCDBGains.h>
#include <CondFormats/DataRecord/interface/CSCDBGainsRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBCrosstalk.h>
#include <CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h>
#include <CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h>
#include <CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h>

#include <FWCore/Utilities/interface/Exception.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h> 

#include <iostream>


/* Constructor
 *
 */
CSCRecHitDBuilder::CSCRecHitDBuilder( const edm::ParameterSet& ps ) : geom_(0) {
  
  // Receives ParameterSet percolated down from EDProducer	


  useCalib               = ps.getUntrackedParameter<bool>("CSCUseCalibrations");  
  isData                 = ps.getUntrackedParameter<bool>("CSCIsRunningOnData"); 
  debug                  = ps.getUntrackedParameter<bool>("CSCDebug");
  stripWireDeltaT        = ps.getUntrackedParameter<int>("CSCstripWireDeltaTime");
  makePseudo2DHits       = ps.getUntrackedParameter<bool>("CSCproduce1DHits");
  
  HitsFromStripOnly_     = new CSCHitFromStripOnly( ps ); 
  HitsFromWireOnly_      = new CSCHitFromWireOnly( ps );  
  //HitsFromWireSegments_  = new CSCWireSegments( ps );
  //HitsFromStripSegments_ = new CSCStripSegments( ps );
  Make2DHits_            = new CSCMake2DRecHit( ps );
}

/* Destructor
 *
 */
CSCRecHitDBuilder::~CSCRecHitDBuilder() {
  delete HitsFromStripOnly_;
  delete HitsFromWireOnly_;
  //delete HitsFromWireSegments_;
  //delete HitsFromStripSegments_;
  delete Make2DHits_;   
}


/* Member function build
 *
 */
void CSCRecHitDBuilder::build( const CSCStripDigiCollection* stripdc, const CSCWireDigiCollection* wiredc,
                               CSCRecHit2DCollection& oc ) {

  if ( useCalib ) {
    // Pass gain constants to strip hit reconstruction package
    HitsFromStripOnly_->setCalibration( gAvg_, gains_ );
    // Pass X-talks and noise matrix to 2-D hit builder 
    Make2DHits_->setCalibration( gAvg_, gains_, xtalk_, noise_ );
  }



  // Clean hit collections sorted by layer    
  std::vector<CSCDetId> stripLayer;
  std::vector<CSCDetId>::const_iterator sIt;
  std::vector<CSCDetId> wireLayer;
  std::vector<CSCDetId>::const_iterator wIt;

  
  // Make collection of wire only hits !  
  CSCWireHitCollection clean_woc;
  
  for ( CSCWireDigiCollection::DigiRangeIterator it = wiredc->begin(); it != wiredc->end(); ++it ){
    const CSCDetId& id = (*it).first;
    const CSCLayer* layer = getLayer( id );
    const CSCWireDigiCollection::Range rwired = wiredc->get( id );
    
    // Skip if no wire digis in this layer
    if ( rwired.second == rwired.first ) continue;
    
    std::vector<CSCWireHit> rhv = HitsFromWireOnly_->runWire( id, layer, rwired);

    if ( rhv.size() > 0 ) wireLayer.push_back( id );
    
    // Add the wire hits to master collection
    clean_woc.put( id, rhv.begin(), rhv.end() );
  }

  if ( debug ) std::cout << "Done producing wire hits " << std::endl;

  // Make collection of strip only hits
  
  CSCStripHitCollection clean_soc;  
  for ( CSCStripDigiCollection::DigiRangeIterator it = stripdc->begin(); it != stripdc->end(); ++it ){
    const CSCDetId& id = (*it).first;
    const CSCLayer* layer = getLayer( id );
    const CSCStripDigiCollection::Range& rstripd = (*it).second;
    
    // Skip if no strip digis in this layer
    if ( rstripd.second == rstripd.first ) continue;
    
    std::vector<CSCStripHit> rhv = HitsFromStripOnly_->runStrip( id, layer, rstripd);

    if ( rhv.size() > 0 ) stripLayer.push_back( id );
    
    // Add the strip hits to master collection
    clean_soc.put( id, rhv.begin(), rhv.end() );
  }

  if ( debug ) std::cout << "Done producing strip hits " << std::endl;


  // Now create 2-D hits by looking at superposition of strip and wire hit in a layer
  //
  // N.B.  I've sorted the hits from layer 1-6 always, so can test if there are "holes", 
  // that is layers without hits for a given chamber.

  // Vector to store rechit within layer
  std::vector<CSCRecHit2D> hitsInLayer;

  int layer_idx     = 0;
  int hits_in_layer = 0;
  CSCDetId old_id; 

  // Now loop over each layer containing strip hits
  for ( sIt=stripLayer.begin(); sIt != stripLayer.end(); ++sIt ) {

    bool foundMatch = false;
    hitsInLayer.clear();
    hits_in_layer = 0;
   
    std::vector<CSCStripHit> cscStripHit;
    
    CSCRangeMapForRecHit acc;
    CSCStripHitCollection::range range = clean_soc.get(acc.cscDetLayer(*sIt));

    // Create vector of strip hits for this layer    
    for ( CSCStripHitCollection::const_iterator clean_soc = range.first; clean_soc != range.second; ++clean_soc)
      cscStripHit.push_back(*clean_soc);

    const CSCDetId& sDetId = (*sIt);
    const CSCLayer* layer  = getLayer( sDetId );

    // This is used to test for gaps in layers and needs to be initialized here 
    if ( layer_idx == 0 ) {
      old_id = sDetId;
    }

    // This is the id I'll compare the wire digis against because of the ME1a confusion in data
    // i.e. ME1a == ME11 == ME1b for wire in data
    CSCDetId compId;

    // For ME11, real data wire digis are labelled as belonging to ME1b, 
    // so that's where ME1a must look
    if ((      isData          ) && 
        (sDetId.station() == 1 ) && 
        (sDetId.ring()    == 4 )) {
      int sendcap  = sDetId.endcap();
      int schamber = sDetId.chamber();
      int slayer   = sDetId.layer();
      CSCDetId testId( sendcap, 1, 1, schamber, slayer );
      compId = testId;
    } else {
      compId = sDetId;
    }

    // Now loop over wire hits
    for ( wIt=wireLayer.begin(); wIt != wireLayer.end(); ++wIt ) {
        
      const CSCDetId& wDetId  = (*wIt);
        
      // Because of ME1a, use the compId to make a comparison between strip and wire hit CSCDetId
      if ((wDetId.endcap()  == compId.endcap() ) &&
          (wDetId.station() == compId.station()) &&
          (wDetId.ring()    == compId.ring()   ) &&
          (wDetId.chamber() == compId.chamber()) &&
          (wDetId.layer()   == compId.layer()  )) {
          
        // Create vector of wire hits for this layer
        std::vector<CSCWireHit> cscWireHit;
       
        CSCRangeMapForRecHit acc2;
        CSCWireHitCollection::range range = clean_woc.get(acc2.cscDetLayer(*wIt));
      
        for ( CSCWireHitCollection::const_iterator clean_woc = range.first; clean_woc != range.second; ++clean_woc)
          cscWireHit.push_back(*clean_woc);

        // Build 2D hit for all possible strip-wire pairs 
        // overlapping within this layer
        if (debug) std::cout << "# strip hits in layer: " << cscStripHit.size() << "  " 
                             << "# wire hits in layer: "  << cscWireHit.size()  << std::endl;

        for (unsigned i = 0; i != cscStripHit.size(); ++i ) {
          const CSCStripHit s_hit = cscStripHit[i];
          for (unsigned j = 0; j != cscWireHit.size(); ++j ) {
            const CSCWireHit w_hit = cscWireHit[j];
            CSCRecHit2D rechit = Make2DHits_->hitFromStripAndWire(sDetId, layer, w_hit, s_hit);
            bool isInFiducial = Make2DHits_->isHitInFiducial( layer, rechit );
            if ( isInFiducial ) {
              foundMatch = true;  
              hitsInLayer.push_back( rechit );
              hits_in_layer++;
            }
          }
        }
      }
    }

    // output vector of 2D rechits to collection
    if (hits_in_layer > 0) {
      oc.put( sDetId, hitsInLayer.begin(), hitsInLayer.end() );
      hitsInLayer.clear();
    }
    hits_in_layer = 0;
    layer_idx++;
    old_id = sDetId;
  }

  if ( debug ) std::cout << "Done producing 2D-hits " << std::endl;

}


/* getLayer
 *
 */
const CSCLayer* CSCRecHitDBuilder::getLayer( const CSCDetId& detId )  {
  if ( !geom_ ) throw cms::Exception("MissingGeometry") << "[CSCRecHitDBuilder::getLayer] Missing geometry" << std::endl;
  return geom_->layer(detId);
}

