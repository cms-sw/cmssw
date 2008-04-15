// This is  CSCHitFromStripOnly.cc
// Taken from RecHitB. Possible changes 

#include <RecoLocalMuon/CSCRecHitD/src/CSCHitFromStripOnly.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCPeakBinOfStripPulse.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripData.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHitData.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include <Geometry/CSCGeometry/interface/CSCLayerGeometry.h>

#include <CondFormats/CSCObjects/interface/CSCGains.h>
#include <CondFormats/DataRecord/interface/CSCGainsRcd.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <cmath>
#include <string>
#include <vector>
#include <iostream>


CSCHitFromStripOnly::CSCHitFromStripOnly( const edm::ParameterSet& ps ) : recoConditions_(0) {
  
  useCalib                   = ps.getUntrackedParameter<bool>("CSCUseCalibrations");
  //theClusterSize             = ps.getUntrackedParameter<int>("CSCStripClusterSize");
  theThresholdForAPeak       = ps.getUntrackedParameter<double>("CSCStripPeakThreshold");
  theThresholdForCluster     = ps.getUntrackedParameter<double>("CSCStripClusterChargeCut");

  pulseheightOnStripFinder_  = new CSCPeakBinOfStripPulse( ps );

}


CSCHitFromStripOnly::~CSCHitFromStripOnly() {
  delete pulseheightOnStripFinder_;
}


/* runStrip
 *
 * Search for strip with ADC output exceeding theThresholdForAPeak.  For each of these strips,
 * build a cluster of strip of size theClusterSize (typically 5 strips).  Finally, make
 * a Strip Hit out of these clusters by finding the center-of-mass position of the hit
 */
std::vector<CSCStripHit> CSCHitFromStripOnly::runStrip( const CSCDetId& id, const CSCLayer* layer,
                                                        const CSCStripDigiCollection::Range& rstripd 
) {	

  std::vector<CSCStripHit> hitsInLayer;
  
  // cache layer info for ease of access
  id_        = id;
  layer_     = layer;
  layergeom_ = layer_->geometry();
  specs_     = layer->chamber()->specs();  
  Nstrips    = specs_->nStrips();

  TmaxOfCluster = 5;

  // Get gain correction weights for all strips in layer, and cache in gainWeight.
  // They're used in fillPulseHeights below.
  if ( useCalib ) {
    recoConditions_->stripWeights( id, gainWeight );
  }
  
  // Fill adc map and find maxima (potential hits)
  fillPulseHeights( rstripd );
  findMaxima();      

  // Make a Strip Hit out of each strip local maximum
  for ( unsigned imax = 0; imax < theMaxima.size(); ++imax ) {

    // Initialize parameters entering the CSCStripHit
    clusterSize = theClusterSize;
    theStrips.clear();
    strips_adc.clear();
    strips_adcRaw.clear();

    // This is where centroid position is determined
    // The strips_adc vector is also filled here
    // Remember, the array starts at 0, but the stripId starts at 1...
    float strippos = makeCluster( theMaxima[imax]+1 );    
    
    if ( strippos < 0 || TmaxOfCluster < 3 ) continue;

    //---- If two maxima are too close the error assigned will be width/sqrt(12) - see CSCXonStrip_MatchGatti.cc
    int maximum_to_left = 99; //---- If there is one maximum - the distance is set to 99 (strips)
    int maximum_to_right = 99;
    if(imax<theMaxima.size()-1){
      maximum_to_right = theMaxima.at(imax+1) - theMaxima.at(imax);
    }
    if(imax>0 ){
      maximum_to_left =  theMaxima.at(imax-1) - theMaxima.at(imax);
    }
    if(fabs(maximum_to_right) < fabs(maximum_to_left)){
      theClosestMaximum.push_back(maximum_to_right);
    }
    else{
      theClosestMaximum.push_back(maximum_to_left);
    }
    
    //---- Check if a neighbouring strip is a dead strip
    bool deadStrip = isNearDeadStrip(id, theMaxima.at(imax)); 
    
    CSCStripHit striphit( id, strippos, TmaxOfCluster, theStrips, strips_adc, strips_adcRaw,
			  theConsecutiveStrips.at(imax), theClosestMaximum.at(imax), deadStrip);
    hitsInLayer.push_back( striphit ); 
  }

  return hitsInLayer;
}


/* makeCluster
 *
 */
float CSCHitFromStripOnly::makeCluster( int centerStrip ) {
  
  float strippos = -1.;
  clusterSize = theClusterSize;
  std::vector<CSCStripHitData> stripDataV;
 
  // We only want to use strip position in terms of strip # for the strip hit.
    
  // If the cluster size is such that you go beyond the edge of detector, shrink cluster appropriatly
  for ( int i = 1; i < theClusterSize/2 + 1; ++i) {
 
    if ( centerStrip - i < 1 || centerStrip + i > specs_->nStrips() ) {

      // Shrink cluster size, but keep it an odd number of strips.
      clusterSize = 2*i - 1;  
    }
  }

  for ( int i = -clusterSize/2; i <= clusterSize/2; ++i ) {
    CSCStripHitData data = makeStripData(centerStrip, i);
    stripDataV.push_back( data );
    theStrips.push_back( centerStrip + i );
  }
  
  strippos = findHitOnStripPosition( stripDataV, centerStrip );
  
  return strippos;
}


/** makeStripData
 *
 */
CSCStripHitData CSCHitFromStripOnly::makeStripData(int centerStrip, int offset) {
  
  CSCStripHitData prelimData(-1.,0.,0.,0.,0.,0.,0.,0.,0.,0);
  int thisStrip = centerStrip+offset;

  int tmax      = thePulseHeightMap[centerStrip-1].t();
  TmaxOfCluster = tmax;

  float adc[12];
  float adcRaw[12];

  if ( tmax == 3 ) {
    adc[0] = thePulseHeightMap[thisStrip-1].y2();
    adc[1] = thePulseHeightMap[thisStrip-1].y3();
    adc[2] = thePulseHeightMap[thisStrip-1].y4();
    adc[3] = thePulseHeightMap[thisStrip-1].y5();
    adcRaw[0] = thePulseHeightMap[thisStrip-1].y2Raw();
    adcRaw[1] = thePulseHeightMap[thisStrip-1].y3Raw();
    adcRaw[2] = thePulseHeightMap[thisStrip-1].y4Raw();
    adcRaw[3] = thePulseHeightMap[thisStrip-1].y5Raw();
  } else if ( tmax == 4 ) {
    adc[0] = thePulseHeightMap[thisStrip-1].y3();
    adc[1] = thePulseHeightMap[thisStrip-1].y4();
    adc[2] = thePulseHeightMap[thisStrip-1].y5();
    adc[3] = thePulseHeightMap[thisStrip-1].y6();
    adcRaw[0] = thePulseHeightMap[thisStrip-1].y3Raw();
    adcRaw[1] = thePulseHeightMap[thisStrip-1].y4Raw();
    adcRaw[2] = thePulseHeightMap[thisStrip-1].y5Raw();
    adcRaw[3] = thePulseHeightMap[thisStrip-1].y6Raw();
  } else if ( tmax == 5 ) {
    adc[0] = thePulseHeightMap[thisStrip-1].y4();
    adc[1] = thePulseHeightMap[thisStrip-1].y5();
    adc[2] = thePulseHeightMap[thisStrip-1].y6();
    adc[3] = thePulseHeightMap[thisStrip-1].y7();
    adcRaw[0] = thePulseHeightMap[thisStrip-1].y4Raw();
    adcRaw[1] = thePulseHeightMap[thisStrip-1].y5Raw();
    adcRaw[2] = thePulseHeightMap[thisStrip-1].y6Raw();
    adcRaw[3] = thePulseHeightMap[thisStrip-1].y7Raw();
  } else if ( tmax == 6 ) {
    adc[0] = thePulseHeightMap[thisStrip-1].y5();
    adc[1] = thePulseHeightMap[thisStrip-1].y6();
    adc[2] = thePulseHeightMap[thisStrip-1].y7();
    adc[3] = 0.1;
    adcRaw[0] = thePulseHeightMap[thisStrip-1].y5Raw();
    adcRaw[1] = thePulseHeightMap[thisStrip-1].y6Raw();
    adcRaw[2] = thePulseHeightMap[thisStrip-1].y7Raw();
    adcRaw[3] = 0.1;
  } else {
    adc[0] = 0.1;
    adc[1] = 0.1;
    adc[2] = 0.1;
    adc[3] = 0.1;
    adcRaw[0] = 0.1;
    adcRaw[1] = 0.1;
    adcRaw[2] = 0.1;
    adcRaw[3] = 0.1;
    LogTrace("CSCRecHit")  << "[CSCHitFromStripOnly::makeStripData] Tmax out of range: contact CSC expert!!!" << "\n";
  }
  
  if ( offset == 0 ) {
    prelimData = CSCStripHitData(thisStrip, adc[0], adc[1], adc[2], adc[3], 
				 adcRaw[0], adcRaw[1], adcRaw[2], adcRaw[3],TmaxOfCluster);
  } else {
    int sign = offset>0 ? 1 : -1;
    // If there's another maximum that would like to use part of this cluster, 
    // it gets shared in proportion to the height of the maxima
    for ( int i = 1; i <= clusterSize/2; ++i ) {

      // Find the direction of the offset
      int testStrip = thisStrip + sign*i;
      std::vector<int>::iterator otherMax = find(theMaxima.begin(), theMaxima.end(), testStrip-1);

      // No other maxima found, so just store
      if ( otherMax == theMaxima.end() ) {
        prelimData = CSCStripHitData(thisStrip, adc[0], adc[1], adc[2], adc[3], adcRaw[0], adcRaw[1], adcRaw[2], adcRaw[3],TmaxOfCluster);
      } else {
        if ( tmax == 3 ) {
          adc[4] = thePulseHeightMap[testStrip-1].y2();
          adc[5] = thePulseHeightMap[testStrip-1].y3();
          adc[6] = thePulseHeightMap[testStrip-1].y4();
          adc[7] = thePulseHeightMap[testStrip-1].y5();
          adc[8] = thePulseHeightMap[centerStrip-1].y2();
          adc[9] = thePulseHeightMap[centerStrip-1].y3();
          adc[10] = thePulseHeightMap[centerStrip-1].y4();
          adc[11] = thePulseHeightMap[centerStrip-1].y5();
          adcRaw[4] = thePulseHeightMap[testStrip-1].y2Raw();
          adcRaw[5] = thePulseHeightMap[testStrip-1].y3Raw();
          adcRaw[6] = thePulseHeightMap[testStrip-1].y4Raw();
          adcRaw[7] = thePulseHeightMap[testStrip-1].y5Raw();
          adcRaw[8] = thePulseHeightMap[centerStrip-1].y2Raw();
          adcRaw[9] = thePulseHeightMap[centerStrip-1].y3Raw();
          adcRaw[10] = thePulseHeightMap[centerStrip-1].y4Raw();
          adcRaw[11] = thePulseHeightMap[centerStrip-1].y5Raw();
        } else if ( tmax == 4 ) {
          adc[4] = thePulseHeightMap[testStrip-1].y3();
          adc[5] = thePulseHeightMap[testStrip-1].y4();
          adc[6] = thePulseHeightMap[testStrip-1].y5();
          adc[7] = thePulseHeightMap[testStrip-1].y6();
          adc[8] = thePulseHeightMap[centerStrip-1].y3();
          adc[9] = thePulseHeightMap[centerStrip-1].y4();
          adc[10] = thePulseHeightMap[centerStrip-1].y5();
          adc[11] = thePulseHeightMap[centerStrip-1].y6();
	  adcRaw[4] = thePulseHeightMap[testStrip-1].y3Raw();
	  adcRaw[5] = thePulseHeightMap[testStrip-1].y4Raw();
          adcRaw[6] = thePulseHeightMap[testStrip-1].y5Raw();
          adcRaw[7] = thePulseHeightMap[testStrip-1].y6Raw();
          adcRaw[8] = thePulseHeightMap[centerStrip-1].y3Raw();
          adcRaw[9] = thePulseHeightMap[centerStrip-1].y4Raw();
          adcRaw[10] = thePulseHeightMap[centerStrip-1].y5Raw();
          adcRaw[11] = thePulseHeightMap[centerStrip-1].y6Raw();
	} else if ( tmax == 5 ) {
          adc[4] = thePulseHeightMap[testStrip-1].y4();
          adc[5] = thePulseHeightMap[testStrip-1].y5();
          adc[6] = thePulseHeightMap[testStrip-1].y6();
          adc[7] = thePulseHeightMap[testStrip-1].y7();
          adc[8] = thePulseHeightMap[centerStrip-1].y4();
          adc[9] = thePulseHeightMap[centerStrip-1].y5();
          adc[10] = thePulseHeightMap[centerStrip-1].y6();
          adc[11] = thePulseHeightMap[centerStrip-1].y7();
          adcRaw[4] = thePulseHeightMap[testStrip-1].y4Raw();
          adcRaw[5] = thePulseHeightMap[testStrip-1].y5Raw();
          adcRaw[6] = thePulseHeightMap[testStrip-1].y6Raw();
          adcRaw[7] = thePulseHeightMap[testStrip-1].y7Raw();
          adcRaw[8] = thePulseHeightMap[centerStrip-1].y4Raw();
          adcRaw[9] = thePulseHeightMap[centerStrip-1].y5Raw();
          adcRaw[10] = thePulseHeightMap[centerStrip-1].y6Raw();
          adcRaw[11] = thePulseHeightMap[centerStrip-1].y7Raw();
	} else if ( tmax == 6 ) {
          adc[4] = thePulseHeightMap[testStrip-1].y5();
          adc[5] = thePulseHeightMap[testStrip-1].y6();
          adc[6] = thePulseHeightMap[testStrip-1].y7();
          adc[7] = 0.1;
          adc[8] = thePulseHeightMap[centerStrip-1].y5();
          adc[9] = thePulseHeightMap[centerStrip-1].y6();
          adc[10] = thePulseHeightMap[centerStrip-1].y7();
          adc[11] = 0.1;
          adcRaw[4] = thePulseHeightMap[testStrip-1].y5Raw();
          adcRaw[5] = thePulseHeightMap[testStrip-1].y6Raw();
          adcRaw[6] = thePulseHeightMap[testStrip-1].y7Raw();
          adcRaw[7] = 0.1;
          adcRaw[8] = thePulseHeightMap[centerStrip-1].y5Raw();
          adcRaw[9] = thePulseHeightMap[centerStrip-1].y6Raw();
          adcRaw[10] = thePulseHeightMap[centerStrip-1].y7Raw();
          adcRaw[11] = 0.1;
        } else {
          for (int k = 4; k < 12; ++k ){
	    adc[k] = adcRaw[k] = 0.1;
	  }
        }
        // Scale shared strip B by ratio of peak of ADC counts from central strip A
        // and neighbouring maxima C
	for (int k = 0; k < 4; ++k){
	  if(adc[k+4]>0 && adc[k+8]>0){
	    adc[k] = adc[k] * adc[k+8] / (adc[k+4]+adc[k+8]);
	  }
	  if(adcRaw[k+4]>0 && adcRaw[k+8]>0){
	    adcRaw[k] = adcRaw[k] * adcRaw[k+8] / (adcRaw[k+4]+adcRaw[k+8]);
	  }
	}
        prelimData = CSCStripHitData(thisStrip, adc[0], adc[1], adc[2], adc[3], 
				     adcRaw[0], adcRaw[1], adcRaw[2], adcRaw[3], TmaxOfCluster);
      }
    }
  }
  return prelimData;
}
 

/* fillPulseHeights
 *
 */
void CSCHitFromStripOnly::fillPulseHeights( const CSCStripDigiCollection::Range& rstripd ) {
  
  // Loop over strip digis and fill the pulseheight map
  
  thePulseHeightMap.clear();
  thePulseHeightMap.resize(100);

  //float maxADC = 0;  
  //float maxADCCluster = 0;  
  //float adc_cluster[3];  
  //adc_cluster[0] = 0.;  
  //adc_cluster[1] = 0.;  
  //adc_cluster[2] = 0.;  

  for ( CSCStripDigiCollection::const_iterator it = rstripd.first; it != rstripd.second; ++it ) {
    bool fill            = true;
    int  thisChannel     = (*it).getStrip(); 
    std::vector<int> sca = (*it).getADCCounts();
    float height[6];
    float hmax = 0.;
    int   tmax = -1;

    fill = pulseheightOnStripFinder_->peakAboveBaseline( (*it), hmax, tmax, height );
    //float maxCluster = 0;

    //adc_cluster[2] = adc_cluster[1];
    //adc_cluster[1] = adc_cluster[0];
    //adc_cluster[0] = hmax;  

    //for (int j = 0; j < 3; ++j ) maxCluster += adc_cluster[j];   
 
    //if (hmax > maxADC) maxADC = hmax;
    //if (maxCluster > maxADCCluster ) maxADCCluster = maxCluster;

    // Don't forget that the ME_11/a strips are ganged !!!
    // Have to loop 2 more times to populate strips 17-48.
    
    if ( id_.station() == 1 && id_.ring() == 4 ) {
      for ( int j = 0; j < 3; ++j ) {
        thePulseHeightMap[thisChannel-1+16*j] = CSCStripData( float(thisChannel+16*j), hmax, tmax, height[0], height[1], height[2], height[3], height[4], height[5], height[0], height[1], height[2], height[3], height[4], height[5]);
        if ( useCalib ){
           thePulseHeightMap[thisChannel-1+16*j] *= gainWeight[thisChannel-1];
        }
      }
    } else {
      thePulseHeightMap[thisChannel-1] = CSCStripData( float(thisChannel), hmax, tmax, height[0], height[1], height[2], height[3], height[4], height[5], height[0], height[1], height[2], height[3], height[4], height[5]);
      if ( useCalib ){
        thePulseHeightMap[thisChannel-1] *= gainWeight[thisChannel-1];
      }
    }
  }
}


/* findMaxima
 *
 * fills vector theMaxima with the local maxima in the pulseheight distribution
 * of the strips. The threshold defining a maximum is a configurable parameter.
 * A typical value is 30.
 */
void CSCHitFromStripOnly::findMaxima() {
  
  theMaxima.clear();
  theConsecutiveStrips.clear();
  theClosestMaximum.clear();
  for ( size_t i = 0; i < thePulseHeightMap.size(); ++i ) {

    float heightPeak = thePulseHeightMap[i].ymax();

    // sum 3 strips so that hits between strips are not suppressed
    float heightCluster;

    bool maximumFound = false;
    // Left edge of chamber
    if ( i == 0 ) {
      heightCluster = thePulseHeightMap[i].ymax()+thePulseHeightMap[i+1].ymax();
      // Have found a strip Hit if...
      if (( heightPeak                   > theThresholdForAPeak         ) &&  
          ( heightCluster                > theThresholdForCluster       ) && 
	  ( thePulseHeightMap[i].ymax() >= thePulseHeightMap[i+1].ymax()) &&
          ( thePulseHeightMap[i].t()     > 2                            ) &&
          ( thePulseHeightMap[i].t()     < 7                            )) {
        theMaxima.push_back(i);
	maximumFound = true;
      }
    // Right edge of chamber
    } else if ( i == thePulseHeightMap.size()-1) {  
      heightCluster = thePulseHeightMap[i-1].ymax()+thePulseHeightMap[i].ymax();
      // Have found a strip Hit if...
      if (( heightPeak                   > theThresholdForAPeak         ) &&  
          ( heightCluster                > theThresholdForCluster       ) && 
          ( thePulseHeightMap[i].ymax()  > thePulseHeightMap[i-1].ymax()) &&
          ( thePulseHeightMap[i].t()     > 2                            ) &&
          ( thePulseHeightMap[i].t()     < 7                            )) {
        theMaxima.push_back(i);
	maximumFound = true;
      }
    // Any other strips
    } else {
      heightCluster = thePulseHeightMap[i-1].ymax()+thePulseHeightMap[i].ymax()+thePulseHeightMap[i+1].ymax();
      // Have found a strip Hit if...
      if (( heightPeak                   > theThresholdForAPeak         ) &&  
          ( heightCluster                > theThresholdForCluster       ) && 
          ( thePulseHeightMap[i].ymax()  > thePulseHeightMap[i-1].ymax()) &&
	  ( thePulseHeightMap[i].ymax() >= thePulseHeightMap[i+1].ymax()) &&
          ( thePulseHeightMap[i].t()     > 2                            ) &&
          ( thePulseHeightMap[i].t()     < 7                            )) {
        theMaxima.push_back(i);
	maximumFound = true;
      }
    }
    //---- Consecutive strips with charge (real cluster); if too wide - measurement is not accurate 
    if(maximumFound){
      int numberOfConsecutiveStrips = 1;
      float testThreshold = 10.;//---- ADC counts; 
                                //---- this is not XTalk corrected so it is correct in first approximation only
      int j = 0;
      for(int l = 0;l<8 ;l++){
        if(j<0){
          std::cout<<" CSCRecHitD: This should never occur!!! Contact CSC expert!"<<std::endl;
        }
        j++;
        bool signalPresent = false;
        for(int k =0;k<2;k++){
          j*= -1;//---- check from left and right
          int anotherConsecutiveStrip = i+j;
          if(anotherConsecutiveStrip>=0 && anotherConsecutiveStrip<int(thePulseHeightMap.size() )){
            if(thePulseHeightMap[anotherConsecutiveStrip].ymax()>testThreshold){
              numberOfConsecutiveStrips++;
              signalPresent = true;
            }
          }
        }
        if(!signalPresent){
          break;
        }
      }
      theConsecutiveStrips.push_back(numberOfConsecutiveStrips);
    }
  }
}



/* findHitOnStripPosition
 *
 */
float CSCHitFromStripOnly::findHitOnStripPosition( const std::vector<CSCStripHitData>& data, const int& centerStrip ) {
  
  float strippos = -1.;
  
  int nstrips = data.size();
  if ( nstrips < 1 ) return strippos;
  
  // biggestStrip is strip with largest pulse height 
  // Use pointer subtraction

  int biggestStrip = max_element(data.begin(), data.end()) - data.begin();
  strippos = data[biggestStrip].x() * 1.;
  
  // If more than one strip:  use centroid to find center of cluster
  // but only use time bin == tmax (otherwise, bias centroid).
  float sum  = 0.;
  float sum_w= 0.;

  float stripLeft = 0.;
  float stripRight = 0.;
  
  for ( unsigned i = 0; i != data.size(); ++i ) {
    float w0 = data[i].y0();
    float w1 = data[i].y1();
    float w2 = data[i].y2();
    float w3 = data[i].y3();
    float w0Raw = data[i].y0Raw();
    float w1Raw = data[i].y1Raw();
    float w2Raw = data[i].y2Raw();
    float w3Raw = data[i].y3Raw();

    // Require ADC to be > 0.
    if (w0 < 0.) w0 = 0.001;
    if (w1 < 0.) w1 = 0.001;
    if (w2 < 0.) w2 = 0.001;
    if (w3 < 0.) w3 = 0.001;
    

    if (i == data.size()/2 -1) stripLeft = w1;
    if (i == data.size()/2 +1) stripRight = w1;


     // Fill the adcs to the strip hit --> needed for Gatti fitter
    strips_adc.push_back( w0 );
    strips_adc.push_back( w1 );
    strips_adc.push_back( w2 );
    strips_adc.push_back( w3 );
    strips_adcRaw.push_back( w0Raw );
    strips_adcRaw.push_back( w1Raw );
    strips_adcRaw.push_back( w2Raw );
    strips_adcRaw.push_back( w3Raw );

    if ( data[i].x() < 1 ){
      LogTrace("CSCRecHit") << "problem in indexing of strip, strip id is: " << data[i].x() << "\n";
    } 
    sum_w += w1;
    sum   += w1 * data[i].x();
  }

  if ( sum_w > 0.) strippos = sum / sum_w;    

  return strippos;
}

bool CSCHitFromStripOnly::isNearDeadStrip(const CSCDetId& id, int centralStrip){

  const std::bitset<80> & deadStrips = recoConditions_->badStripWord( id );
  bool isDead = false;
  if(centralStrip>0 && centralStrip<79){
    isDead = (deadStrips.test(centralStrip+1) || deadStrips.test(centralStrip-1));
  }
  return isDead;
} 
