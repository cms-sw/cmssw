// This is CSCRecHitDBuilder.cc

#include <RecoLocalMuon/CSCRecHitD/src/CSCRecHitDBuilder.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCHitFromStripOnly.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCHitFromWireOnly.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCMake2DRecHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHitCollection.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHitCollection.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCRangeMapForRecHit.h>
 
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


CSCRecHitDBuilder::CSCRecHitDBuilder( const edm::ParameterSet& ps ) : geom_(0) {
  
  // Receives ParameterSet percolated down from EDProducer	

  useCalib               = ps.getParameter<bool>("CSCUseCalibrations");  
  stripWireDeltaT        = ps.getParameter<int>("CSCstripWireDeltaTime");
  
  hitsFromStripOnly_     = new CSCHitFromStripOnly( ps ); 
  hitsFromWireOnly_      = new CSCHitFromWireOnly( ps );  
  make2DHits_            = new CSCMake2DRecHit( ps );
}


CSCRecHitDBuilder::~CSCRecHitDBuilder() {
  delete hitsFromStripOnly_;
  delete hitsFromWireOnly_;
  delete make2DHits_;   
}


void CSCRecHitDBuilder::build( const CSCStripDigiCollection* stripdc, const CSCWireDigiCollection* wiredc,
                               CSCRecHit2DCollection& oc ) {
  LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] build entered";
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
    if ( rwired.second == rwired.first ) {
	    continue; 
    }
          
    std::vector<CSCWireHit> rhv = hitsFromWireOnly_->runWire( id, layer, rwired);

    if ( rhv.size() > 0 ) wireLayer.push_back( id );
    
    // Add the wire hits to master collection
    clean_woc.put( id, rhv.begin(), rhv.end() );
  }

  LogTrace("CSCRecHitBuilder") << "[CSCRecHitDBuilder] wire hits created";

  // Make collection of strip only hits
  
  CSCStripHitCollection clean_soc;  
  for ( CSCStripDigiCollection::DigiRangeIterator it = stripdc->begin(); it != stripdc->end(); ++it ){
    const CSCDetId& id = (*it).first;
    const CSCLayer* layer = getLayer( id );
    const CSCStripDigiCollection::Range& rstripd = (*it).second;
    
    // Skip if no strip digis in this layer
    if ( rstripd.second == rstripd.first ) continue;
    
    std::vector<CSCStripHit> rhv = hitsFromStripOnly_->runStrip( id, layer, rstripd);

    if ( rhv.size() > 0 ) stripLayer.push_back( id );
    
    // Add the strip hits to master collection
    clean_soc.put( id, rhv.begin(), rhv.end() );
  }

  LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] strip hits created";


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

    CSCDetId compId = sDetId;
    CSCWireDigiCollection::Range rwired = wiredc->get( sDetId );
    // Skip if no wire digis in this layer
    // But for ME11, real wire digis are labelled as belonging to ME1b, so that's where ME1a must look
    // (We try ME1a - above - anyway, because simulated wire digis are labelled as ME1a.)
    if ( rwired.second == rwired.first ) {
      if ( sDetId.station()!=1 || sDetId.ring()!=4 ){
        continue; // not ME1a, skip to next layer
      }
      // So if ME1a has no wire digis (always the case for data) make the
      // wire digi ID point to ME1b. This is what is compared to the
      // strip digi ID below (and not used anywhere else). 
      // Later, rechits use the strip digi ID for construction.
   
      // It is ME1a but no wire digis there, so try ME1b...
      int endcap  = sDetId.endcap();
      int chamber = sDetId.chamber();
      int layer   = sDetId.layer();
      CSCDetId idw( endcap, 1, 1, chamber, layer ); // Set idw to same layer in ME1b
      compId = idw;
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

        LogTrace("CSCRecHitBuilder")<< "[CSCRecHitDBuilder] found " << cscStripHit.size() << " strip and " 
                             << cscWireHit.size()  << " wire hits in layer " << sDetId;

        for (unsigned i = 0; i != cscStripHit.size(); ++i ) {
          const CSCStripHit s_hit = cscStripHit[i];
          for (unsigned j = 0; j != cscWireHit.size(); ++j ) {
            const CSCWireHit w_hit = cscWireHit[j];
            CSCRecHit2D rechit = make2DHits_->hitFromStripAndWire(sDetId, layer, w_hit, s_hit);

            bool isInFiducial = make2DHits_->isHitInFiducial( layer, rechit );
            if ( isInFiducial ) {
              hitsInLayer.push_back( rechit );
              hits_in_layer++;
            }
          }
        }
      }
    }

    LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] " << hits_in_layer << " rechits found in layer " << sDetId;

    // output vector of 2D rechits to collection
    if (hits_in_layer > 0) {
      oc.put( sDetId, hitsInLayer.begin(), hitsInLayer.end() );
      hitsInLayer.clear();
    }
    hits_in_layer = 0;
    layer_idx++;
    old_id = sDetId;
  }

  LogTrace("CSCRecHitDBuilder") << "[CSCRecHitDBuilder] " << oc.size() << " 2d rechits created in this event.";

}


const CSCLayer* CSCRecHitDBuilder::getLayer( const CSCDetId& detId )  {
  if ( !geom_ ) throw cms::Exception("MissingGeometry") << "[CSCRecHitDBuilder::getLayer] Missing geometry" << std::endl;
  return geom_->layer(detId);
}


void CSCRecHitDBuilder::setConditions( const CSCRecoConditions* reco ) {
  hitsFromStripOnly_->setConditions( reco );
  hitsFromWireOnly_->setConditions( reco );
  make2DHits_->setConditions( reco );  
}
