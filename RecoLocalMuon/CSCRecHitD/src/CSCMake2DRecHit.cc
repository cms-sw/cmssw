// This is CSCMake2DRecHit
//---- Taken from RecHitB. Possible changes
 
#include <RecoLocalMuon/CSCRecHitD/src/CSCMake2DRecHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCXonStrip_MatchGatti.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>


/* Constructor
 *
 */
CSCMake2DRecHit::CSCMake2DRecHit(const edm::ParameterSet& ps){
    
  useCalib                   = ps.getUntrackedParameter<bool>("CSCUseCalibrations");
  stripWireDeltaTime         = ps.getUntrackedParameter<int>("CSCstripWireDeltaTime");

  xMatchGatti_             = new CSCXonStrip_MatchGatti( ps );

}   


/* Destructor
 *
 */
CSCMake2DRecHit::~CSCMake2DRecHit() {
  delete xMatchGatti_;
}


/* hitFromStripAndWire
 *
 */
CSCRecHit2D CSCMake2DRecHit::hitFromStripAndWire(const CSCDetId& id, const CSCLayer* layer,
                                                 const CSCWireHit& wHit, const CSCStripHit& sHit){
  
  
  // Cache layer info for ease of access
  layer_        = layer;
  layergeom_    = layer_->geometry();
  specs_        = layer->chamber()->specs();
  id_           = id;
  
  const float sqrt_12 = 3.4641;
  
  float tpeak = -99.;
  
  CSCRecHit2D::ADCContainer adcMap;
  CSCRecHit2D::ChannelContainer wgroups;
  
  
  // Find wire hit position and wire properties
  wgroups = wHit.wgroups();
 


  int wg_left = wgroups[0];;
  int wg_right = wgroups[wgroups.size()-1];
  
  int Nwires1 = layergeom_->numberOfWiresPerGroup( wg_left );
  int Nwires2 = layergeom_->numberOfWiresPerGroup( wg_right );
  
  float Mwire1 = layergeom_->middleWireOfGroup( wg_left );
  float Mwire2 = layergeom_->middleWireOfGroup( wg_right );
  
  int centerWire_left = (int) (Mwire1 - Nwires1 / 2. + 0.5);
  int centerWire_right = (int) (Mwire2 + Nwires2 / 2.);
  
  float centerWire = (centerWire_left + centerWire_right) / 2.;

  //---- WGs around dead HV segment regions may need special treatment...
  //---- This is not addressed here.
    
  float sigmaWire = 0.;
  if(1==wgroups.size()){
    //---- simple - just 1 WG
    sigmaWire  = layergeom_->yResolution( wgroups[0]);
  }
  else if(2==wgroups.size()){
    //---- 2 WGs - get the larger error (overestimation if a single track is passing
    //---- between the WGs; underestimation if there are two separate signal sources)
    if(layergeom_->yResolution( wgroups[0] ) > layergeom_->yResolution( wgroups[1] )){
      sigmaWire  = layergeom_->yResolution( wgroups[0]);
    }    
    else{
      sigmaWire  = layergeom_->yResolution( wgroups[1]);
    }
  }  
  else{
    //---- worst possible case; take most conservative approach
    for(unsigned int iWG=0;iWG<wgroups.size();iWG++){
      sigmaWire+=layergeom_->yResolution( wgroups[iWG] );
    }
  }
  
  // Find strips position and properties
  
  CSCRecHit2D::ChannelContainer strips = sHit.strips();
  int tmax = sHit.tmax();
  int nStrip = strips.size();
  int idCenterStrip = nStrip/2;
  int ch = strips[idCenterStrip];
  int centerStrip = ch;
  //float strip_pos = sHit.sHitPos(); // centroid, in units of strip #
  
  
  // Setup ADCs
  std::vector<float> adcs = sHit.s_adc();
  std::vector<float> adc2;
  for ( int iStrip = 0; iStrip < nStrip; ++iStrip) {
    adc2.clear();
    for ( int t = 0; t < 4; ++t ) adc2.push_back(adcs[t+iStrip*4]);    
    adcMap.put( strips[iStrip], adc2.begin(), adc2.end() ); 
    if (iStrip == nStrip/2 ) 
      tpeak = 50. * ( adc2[0]*(tmax-1) + adc2[1]*tmax + adc2[2]*(tmax+1) ) / (adc2[0]+adc2[1]+adc2[2]);
  }

  float positionWithinTheStrip= -99.;
  float sigmaWithinTheStrip = -99.;
  int quality = -1;
  LocalPoint lp0(0., 0.);
  
  
  // If at the edge, then used 1 strip cluster only :
  if ( ch == 1 || ch == specs_->nStrips() || nStrip < 2 ) {
    lp0 = layergeom_->stripWireIntersection( centerStrip, centerWire);
    positionWithinTheStrip = 0.;
    sigmaWithinTheStrip = layergeom_->stripPitch(lp0) / sqrt_12;
    quality = 2;
  }
  else {
    // If not at the edge, used cluster of size ClusterSize:
    int ch0 = strips[idCenterStrip];
    LocalPoint lp11  = layergeom_->stripWireIntersection( ch0, centerWire);
    float stripWidth = layergeom_->stripPitch( lp11 );
    
    //---- Calculate local position within the strip
    float xWithinChamber = lp11.x();
    quality = 0;
    xMatchGatti_->findXOnStrip( id, layer_, sHit, centerStrip, 
				xWithinChamber,
				stripWidth, tpeak, positionWithinTheStrip, 
				sigmaWithinTheStrip, quality);
    lp0 = LocalPoint( xWithinChamber, layergeom_->yOfWire(centerWire, xWithinChamber) );
  }
  
  // compute the errors in local x and y
  LocalError localerr = layergeom_->localError( centerStrip, 
						sigmaWithinTheStrip, sigmaWire );
  
  // store rechit
  CSCRecHit2D rechit( id, lp0, localerr, strips,
		      adcMap, wgroups, tpeak, positionWithinTheStrip, 
		      sigmaWithinTheStrip, quality);
  return rechit;
}

/* isHitInFiducial
 *
 * Only useful for ME11 chambers.
 */
bool CSCMake2DRecHit::isHitInFiducial( const CSCLayer* layer, const CSCRecHit2D& rh ) {

  bool isInFiducial = true;
  
  // Allow extra margin for future tuning etc.
  float marginAtEdge = 0.1; 
  
  const CSCLayerGeometry* layergeom = layer->geometry();
  
  float y = rh.localPosition().y();
  float apothem = layergeom->length()/2.;
  
  if ( fabs(y) > (apothem+marginAtEdge) ) isInFiducial=false;
  
  return isInFiducial;
}
 
void CSCMake2DRecHit::setConditions( const CSCRecoConditions* reco ) {
  xMatchGatti_->setConditions( reco );
} 

