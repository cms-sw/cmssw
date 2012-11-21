// This is CSCMake2DRecHit
 
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


CSCMake2DRecHit::CSCMake2DRecHit(const edm::ParameterSet& ps):
  peakTimeFinder_( new CSCFindPeakTime( ps ) ){
    
  useCalib            = ps.getParameter<bool>("CSCUseCalibrations");
  stripWireDeltaTime  = ps.getParameter<int>("CSCstripWireDeltaTime"); //@@ Non-standard  CSC*s*trip...
  useTimingCorrections= ps.getParameter<bool>("CSCUseTimingCorrections");
  useGasGainCorrections = ps.getParameter<bool>("CSCUseGasGainCorrections");

  xMatchGatti_        = new CSCXonStrip_MatchGatti( ps );

}   


CSCMake2DRecHit::~CSCMake2DRecHit() {
  delete xMatchGatti_;
}


CSCRecHit2D CSCMake2DRecHit::hitFromStripAndWire(const CSCDetId& id, const CSCLayer* layer,
                                                 const CSCWireHit& wHit, const CSCStripHit& sHit)
{
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
  if(wHit.deadWG()>0 || wgroups.size()>2){
    //---- worst possible case; take most conservative approach
    for(unsigned int iWG=0;iWG<wgroups.size();iWG++){
      sigmaWire+=layergeom_->yResolution( wgroups[iWG] );
    }
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
  else if(1==wgroups.size()){
    //---- simple - just 1 WG
    sigmaWire  = layergeom_->yResolution( wgroups[0]);
  }
  
  // Find strips position and properties
  
  CSCRecHit2D::ChannelContainer strips = sHit.strips();
  int tmax = sHit.tmax();
  int nStrip = strips.size();
  int idCenterStrip = nStrip/2;
  int centerStrip = strips[idCenterStrip];
  
  // Retrieve strip pulseheights from the CSCStripHit
  const std::vector<float>& adc    = sHit.s_adc();
  const std::vector<float>& adcRaw = sHit.s_adcRaw();

  std::vector<float> adc2;
  std::vector<float> adc2Raw;

  LogTrace("CSCMake2DRecHit") << "[CSCMake2DRecHit] dump of adc values to be added to rechit follows...";

  for ( int iStrip = 0; iStrip < nStrip; ++iStrip) {

    adc2.clear();
    adc2Raw.clear();
    adc2.reserve(4);
    adc2Raw.reserve(4);
    for ( int t = 0; t < 4; ++t ){
      adc2.push_back(adc[t+iStrip*4]);
      adc2Raw.push_back(adcRaw[t+iStrip*4]);
    }
    //After CMSSW_5_0: ADC value is pedestal-subtracted and electronics-gain corrected
    adcMap.put( strips[iStrip], adc2.begin(), adc2.end() ); 
    // Up to CMSSW_5_0, Rechit takes _raw_ adc values
    // adcMap.put( strips[iStrip], adc2Raw.begin(), adc2Raw.end() ); 

    LogTrace("CSCMake2DRecHit") << "[CSCMake2DRecHit] strip = " << strips[iStrip] << 
      " adcs= " << adc2Raw[0] << " " << adc2Raw[1] << " " << adc2Raw[2] << " " << adc2Raw[3];

  }

  //The tpeak finding for both edge and non-edge strips has been moved to here
  //tpeak will be a const argument for xMatchGatti_->findXOnStrip
  float adcArray[4];
  for ( int t = 0; t < 4; ++t ) {
    int k = t+4*(idCenterStrip);
    adcArray[t] = adc[k];
  }
  tpeak = peakTimeFinder_->peakTime( tmax, adcArray, tpeak ); 
  // Just for completeness, the start time of the pulse is 133 ns earlier, according to Stan :)
  float t_zero = tpeak - 133.;
  LogTrace("CSCRecHit") << "[CSCMake2DRecHit] " << 
    id << " strip=" << centerStrip << ", t_zero=" << t_zero << ", tpeak=" << tpeak;


  float positionWithinTheStrip= -99.;
  float sigmaWithinTheStrip = -99.;
  int quality = -1;
  LocalPoint lp0(0., 0.);
  
  float stripWidth = -99.;
  // If at the edge, then used 1 strip cluster only
  if ( centerStrip == 1 || centerStrip == specs_->nStrips() || nStrip < 2 ) {
    lp0 = layergeom_->stripWireIntersection( centerStrip, centerWire);
    positionWithinTheStrip = 0.;
    stripWidth = layergeom_->stripPitch(lp0);
    sigmaWithinTheStrip = stripWidth / sqrt_12;
    quality = 2;
  }
  else {
    // If not at the edge, used cluster of size ClusterSize:
    LocalPoint lp11  = layergeom_->stripWireIntersection( centerStrip, centerWire);
    stripWidth = layergeom_->stripPitch( lp11 );
    
    //---- Calculate local position within the strip
    float xWithinChamber = lp11.x();
    quality = 0;
    if(layergeom_->inside(lp11 )){// save time; this hit is to be discarded anyway - see isHitInFiducial(...)

      xMatchGatti_->findXOnStrip( id, layer_, sHit, centerStrip, 
				  xWithinChamber,
				  stripWidth,  tpeak, positionWithinTheStrip, 
				  sigmaWithinTheStrip, quality);
    }				
    lp0 = LocalPoint( xWithinChamber, layergeom_->yOfWire(centerWire, xWithinChamber) );
  }

  
  
  // compute the errors in local x and y
  LocalError localerr = layergeom_->localError( centerStrip, 
						sigmaWithinTheStrip, sigmaWire );

  // Before storing the recHit, take the opportunity to change its time
  if (useTimingCorrections){
    float chipCorrection = recoConditions_->chipCorrection(id,centerStrip);
    float phaseCorrection = (sHit.stripsl1a()[0]>> (15-0) & 0x1)*25.;
    float chamberCorrection = recoConditions_->chamberTimingCorrection(id);

    GlobalPoint gp0 = layer_->toGlobal(lp0);
    float tofCorrection = gp0.mag()/29.9792458;
    float signalPropagationSpeed[11] = {0.0, -78, -76, -188, -262, -97, -99, -90, -99, -99, -113};
    float position = lp0.y()/sin(layergeom_->stripAngle(centerStrip));
    float yCorrection = position/signalPropagationSpeed[id_.iChamberType()];
    //printf("RecHit in e:%d s:%d r:%d c:%d l:%d strip:%d \n",id.endcap(),id.station(), id.ring(),id.chamber(),id.layer(),centerStrip);
    //printf("\t tpeak before = %5.2f \t chipCorr %5.2f phaseCorr %5.2f chamberCorr %5.2f tofCorr %5.2f \n",
    //	   tpeak,chipCorrection, phaseCorrection,chamberCorrection,tofCorrection);
    //printf("localy = %5.2f, yCorr = %5.2f \n",lp0.y(),yCorrection);
    tpeak = tpeak + chipCorrection + phaseCorrection + chamberCorrection-tofCorrection+yCorrection;
    //printf("\t tpeak after = %5.2f\n",tpeak);
  }

  // Calculate wire time to the half bx level using time bins on
  // Store wire time with a precision of 0.01 as an int (multiply by 100)
  // Convert from bx to ns (multiply by 25)
  int scaledWireTime = 100*findWireBx(wHit.timeBinsOn(), tpeak,id)*25; 


  //Get the gas-gain correction for this rechit
  float gasGainCorrection = -999.;   
  if (useGasGainCorrections) {
    gasGainCorrection = recoConditions_->gasGainCorrection(id,centerStrip,centerWire);
  } 

  /// Correct the 3x3 ADC sum into the energy deposited in the layer.  Note:  this correction will 
  /// give you dE.  In order to get the dE/dX, you will need to divide by the path length...
  /// If the algorithm to compute the corrected energy fails, flag it by a specific nonsense value:
  /// If the user has chosen not to use the gas gain correction --->  -998.
  /// If the gas gain correction from the database is a bad value ->  -997.
  /// If it is an edge strip -------------------------------------->  -996.
  /// If gas-gain is OK, but the ADC vector is the wrong size  ---->  -999.
  /// If the user has created the Rechit without the energy deposit>  -995.
  float energyDeposit = -999.;
  if (gasGainCorrection < -998.) {
    // if the user has chosen not to use the gas gain correction, set the energy to a different nonsense value
    energyDeposit = -998.;

  } else if (gasGainCorrection < 0.) {
    // if the gas gain correction from the database is a bad value, set the energy to yet another nonsense value
    energyDeposit = -997.;

  } else {
    // gas-gain correction is OK, correct the 3x3 ADC sum
    if (adcMap.size()==12) {
      energyDeposit =
	adcMap[0] * gasGainCorrection + adcMap[1] * gasGainCorrection + adcMap[2] * gasGainCorrection +
	adcMap[4] * gasGainCorrection + adcMap[5] * gasGainCorrection + adcMap[6] * gasGainCorrection +
	adcMap[8] * gasGainCorrection + adcMap[9] * gasGainCorrection + adcMap[10]* gasGainCorrection ;	

    } else if (adcMap.size()==4) {
      // if this is an edge strip, set the energy to yet another nonsense value
      energyDeposit = -996.;
    }

  }

  /// store rechit

   /// Retrive the L1APhase+strips combination
   CSCRecHit2D::ChannelContainer L1A_and_strips = sHit.stripsTotal();        /// L1A
   /// Retrive the Bx + wgroups combination
   CSCRecHit2D::ChannelContainer BX_and_wgroups = wHit.wgroupsBXandWire();   /// BX
  // (sigmaWithinTheStrip/stripWidth) is in strip widths just like positionWithinTheStrip is!
     CSCRecHit2D rechit( id, lp0, localerr, L1A_and_strips,                  /// L1A;
     //adcMap, wgroups, tpeak, positionWithinTheStrip,
      adcMap, BX_and_wgroups, tpeak, positionWithinTheStrip,        /// BX
      sigmaWithinTheStrip/stripWidth, quality, sHit.deadStrip(), wHit.deadWG(), scaledWireTime,
      energyDeposit);

  /// To see RecHit content (L1A feature included) (to be commented out)
  // rechit.print();

  LogTrace("CSCRecHit") << "[CSCMake2DRecHit] rechit created in layer " << id << "... \n" << rechit;

  return rechit;

}


bool CSCMake2DRecHit::isHitInFiducial( const CSCLayer* layer, const CSCRecHit2D& rh ) {

  bool isInFiducial = true;
  const CSCLayerGeometry* layergeom_ = layer->geometry();
  LocalPoint rhPosition =  rh.localPosition();
  // is the rechit within the chamber? 
  //(the problem occurs in ME11a/b otherwise it is OK)
  // we could use also 
  //bool inside( const Local3DPoint&, const LocalError&, float scale=1.f ) const;
  if(!layergeom_->inside(rhPosition)){
    isInFiducial = false;
  }
  
  return isInFiducial;
}
 

void CSCMake2DRecHit::setConditions( const CSCRecoConditions* reco ) {
  xMatchGatti_->setConditions( reco );
  // And cache for use here
  recoConditions_ = reco;
} 

float CSCMake2DRecHit::findWireBx(std::vector <int> timeBinsOn, float tpeak ,const CSCDetId& id) {
  // Determine the wire Bx from the vector of time bins on for the wire digi with peak time as an intial estimate.
  // Assumes that a single hit should create either one time bin on or two consecutive time bins on
  // so algorithm looks for bin on nearest to peak time and checks if it has a bin on consecutive with it
  float anode_bx_offset = recoConditions_->anodeBXoffset(id);
  float wireBx=-1;
  float timeGuess=tpeak/25.+ anode_bx_offset;  
  float diffMin=9999.;
  int bestMatch=-9;
  for (int j=0; j<(int)timeBinsOn.size(); j++) {
    double diff=timeGuess-timeBinsOn[j];
    // Find bin on closest to peak time
    if (fabs(diff)<fabs(diffMin)) {
      diffMin=diff;
      bestMatch=j;
      wireBx=timeBinsOn[j];
    }
  }
  int side=diffMin/fabs(diffMin);
  bool unchanged=true;
  // First check if bin on the same side as peak time is on
  if ((bestMatch+side)>-1 && (bestMatch+side)<(int)timeBinsOn.size()) {      // Make sure one next to it within vector limits
    if (timeBinsOn[bestMatch]==(timeBinsOn[bestMatch+side]-side)) {      // See if next bin on in vector is consecutive in time
      // Set time to the average of the two bins
      wireBx=wireBx+(float)side/2.;
      unchanged=false;
    }
  }
  // If no match is found then check the other side
  if ((bestMatch-side)>-1 && (bestMatch-side)<(int)timeBinsOn.size() && unchanged) {       // Make sure one next to it exists
    if (timeBinsOn[bestMatch]==(timeBinsOn[bestMatch-side]+side)) {    // See if nextbin on is consecutive in time
      wireBx=wireBx-(double)side/2.;
      unchanged=false;
    }
  }
  return wireBx - anode_bx_offset; // expect collision muons to be centered near 0 bx
}
