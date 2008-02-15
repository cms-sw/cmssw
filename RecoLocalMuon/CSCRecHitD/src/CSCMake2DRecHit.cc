// This is CSCMake2DRecHit
//---- Taken from RecHitB. Possible changes
 
#include <RecoLocalMuon/CSCRecHitD/src/CSCMake2DRecHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCXonStrip_MatchGatti.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>

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
    
  debug                      = ps.getUntrackedParameter<bool>("CSCDebug");
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
  
  float sqrt_12 = 3.4641;
  
  double sigma, chisq, prob;
  sigma = 0.00;
  chisq = -90.00;
  prob  = -91.00;
  float tpeak = -90.;
  
  CSCRecHit2D::ChannelContainer channels;
  CSCRecHit2D::ADCContainer adcMap;
  CSCRecHit2D::ChannelContainer wgroups;
  
  
  // Find wire hit position and wire properties
  wgroups = wHit.wgroups();
  int nWG = wgroups.size();
  int wireg1 = wgroups[0];
  int wireg2 = wgroups[nWG-1];
  
  int Nwires1 = layergeom_->numberOfWiresPerGroup( wireg1 );
  int Nwires2 = layergeom_->numberOfWiresPerGroup( wireg2 );
  
  float Mwire1 = layergeom_->middleWireOfGroup( wireg1 );
  float Mwire2 = layergeom_->middleWireOfGroup( wireg2 );
  
  int wire1 = (int) (Mwire1 - Nwires1 / 2. + 0.5);
  int wire2 = (int) (Mwire2 + Nwires2 / 2.);
  
  float centerWire = (wire1 + wire2) / 2.;
  
  float sigmaWire  = (layergeom_->yResolution( wireg1 ) + layergeom_->yResolution( wireg2 )) / 2.;
  
  
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
  
  //CSCRecHit2D *rechit_;  
  // If at the edge, then used 1 strip cluster only :
  if ( ch == 1 || ch == specs_->nStrips() || nStrip < 2 ) {
    
    LocalPoint lp1 = layergeom_->stripWireIntersection( centerStrip, centerWire);
    
    float x = lp1.x();
    float y = lp1.y();
    
    LocalPoint lp0(x, y);
    
    sigma =  layergeom_->stripPitch(lp0) / sqrt_12; 
    
    // Now compute the errors properly on local x and y
    LocalError localerr = layergeom_->localError( centerStrip, sigma, sigmaWire );
    
    CSCRecHit2D rechit( id, lp0, localerr, strips, adcMap, wgroups, tpeak, chisq, prob );
    return rechit;  
  } 
  else{
    
    // If not at the edge, used cluster of size ClusterSize:
    
    int ch0 = strips[idCenterStrip];
    LocalPoint lp11  = layergeom_->stripWireIntersection( ch0, centerWire);
    
    float x = lp11.x();
    float y = lp11.y();
    
    LocalPoint lpTest(x, y);
    
    float stripWidth = layergeom_->stripPitch( lpTest );  
    sigma =  stripWidth / sqrt_12;
    
    
    // 
    
    if (useCalib){
      xMatchGatti_->setCalibration( globalGainAvg, gains_, xtalk_, noise_ );
    }
    
    //---- Calculate local position within the strip
    float x_to_gatti = x;   
    double PositionWithinTheStrip;
    double SigmaWithinTheStrip;
    float chisq_fit;//---- meaningless here
    xMatchGatti_->findXOnStrip( id, layer_, sHit, centerStrip, x_to_gatti, stripWidth, PositionWithinTheStrip, tpeak, SigmaWithinTheStrip, chisq_fit);
    
    x     = PositionWithinTheStrip;
    sigma = SigmaWithinTheStrip;
    chisq = chisq_fit; //---- meaningless here
    prob  = 1.;//---- meaningless here
    
    y = layergeom_->yOfWire(centerWire, x);
    
    LocalPoint lp0(x, y);
    
    // Now compute the errors properly on local x and y
    LocalError localerr = layergeom_->localError( centerStrip, sigma, sigmaWire );
    
    // store rechit    
    CSCRecHit2D rechit( id, lp0, localerr, strips, adcMap, wgroups, tpeak, chisq, prob );
    return rechit;
  }
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
 
 

