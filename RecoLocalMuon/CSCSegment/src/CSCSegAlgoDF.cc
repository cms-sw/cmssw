/**
 * \file CSCSegAlgoDF.cc
 *
 *  \author Dominique Fortin - UCR
 *
 * Last update: 17.02.2015 - Tim Cox - use CSCSegFit for least-squares fit
 * @@ Seems to find very few segments so many candidates must be being rejected. 
 *
 */
 
#include "CSCSegAlgoDF.h"
#include "CSCSegFit.h"

#include "Geometry/CSCGeometry/interface/CSCLayer.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "CSCSegAlgoPreClustering.h"
#include "CSCSegAlgoShowering.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>


/* Constructor
 *
 */
CSCSegAlgoDF::CSCSegAlgoDF(const edm::ParameterSet& ps) 
  : CSCSegmentAlgorithm(ps), myName("CSCSegAlgoDF"), sfit_(0) {
	
  debug                  = ps.getUntrackedParameter<bool>("CSCSegmentDebug");
  minLayersApart         = ps.getParameter<int>("minLayersApart");
  minHitsPerSegment      = ps.getParameter<int>("minHitsPerSegment");
  dRPhiFineMax           = ps.getParameter<double>("dRPhiFineMax");
  dPhiFineMax            = ps.getParameter<double>("dPhiFineMax");
  tanThetaMax            = ps.getParameter<double>("tanThetaMax");
  tanPhiMax              = ps.getParameter<double>("tanPhiMax");	
  chi2Max                = ps.getParameter<double>("chi2Max");	
  preClustering          = ps.getUntrackedParameter<bool>("preClustering");
  minHitsForPreClustering= ps.getParameter<int>("minHitsForPreClustering");
  nHitsPerClusterIsShower= ps.getParameter<int>("nHitsPerClusterIsShower");
  Pruning                = ps.getUntrackedParameter<bool>("Pruning");
  maxRatioResidual       = ps.getParameter<double>("maxRatioResidualPrune");

  preCluster_            = new CSCSegAlgoPreClustering( ps );
  showering_             = new CSCSegAlgoShowering( ps );
}


/* Destructor
 *
 */
CSCSegAlgoDF::~CSCSegAlgoDF() {
  delete preCluster_;
  delete showering_;
}


/* run
 *
 */
std::vector<CSCSegment> CSCSegAlgoDF::run(const CSCChamber* aChamber, const ChamberHitContainer& rechits) {

  // Store chamber info in temp memory
  theChamber = aChamber; 

  int nHits = rechits.size();

  // Segments prior to pruning
  std::vector<CSCSegment> segments_temp;  

  if ( preClustering && nHits > minHitsForPreClustering ) {
    // This is where the segment origin is in the chamber on avg.
    std::vector<CSCSegment> testSegments;
    std::vector<ChamberHitContainer> clusteredHits = preCluster_->clusterHits(theChamber, rechits);
    // loop over the found clusters:
    for (std::vector<ChamberHitContainer>::iterator subrechits = clusteredHits.begin(); subrechits != clusteredHits.end(); ++subrechits ) {
      // build the subset of segments:
      std::vector<CSCSegment> segs = buildSegments( (*subrechits) );
      // add the found subset of segments to the collection of all segments in this chamber:
      segments_temp.insert( segments_temp.end(), segs.begin(), segs.end() );
    }
  } else {
    std::vector<CSCSegment> segs = buildSegments( rechits );
    // add the found subset of segments to the collection of all segments in this chamber:
    segments_temp.insert( segments_temp.end(), segs.begin(), segs.end() ); 
  }

  return segments_temp; 
}


/* This builds segments by first creating proto-segments from at least 3 hits.
 * We intend to try all possible pairs of hits to start segment building. 'All possible' 
 * means each hit lies on different layers in the chamber.  Once a hit has been assigned 
 * to a segment, we don't consider it again, THAT IS, FOR THE FIRST PASS ONLY !
 * In fact, this is one of the possible flaw with the SK algorithms as it sometimes manages
 * to build segments with the wrong starting points.  In the DF algorithm, the endpoints
 * are tested as the best starting points in a 2nd loop.
 *
 * Also, a  maximum of   5   segments is allowed in the chamber (and then it just gives up.)
 *
 * @@ There are 7 return's in this function!
 *
 */
std::vector<CSCSegment> CSCSegAlgoDF::buildSegments(const ChamberHitContainer& _rechits) {

  ChamberHitContainer rechits = _rechits;
  // Clear buffer for segment vector
  std::vector<CSCSegment> segmentInChamber;
  segmentInChamber.clear();

  unsigned nHitInChamber = rechits.size();

  //  std::cout << "[CSCSegAlgoDF::buildSegments] address of chamber = " << theChamber << std::endl;
  //  std::cout << "[CSCSegAlgoDF::buildSegments] starting in " << theChamber->id()
  //         << " with " << nHitInChamber << " rechits" << std::endl;

  // Return #1 - OK, there aren't enough rechits to build a segment
  if ( nHitInChamber < 3 ) return segmentInChamber;

  LayerIndex layerIndex( nHitInChamber );

  size_t nLayers = 0;
  size_t old_layer = 0;
  for ( size_t i = 0; i < nHitInChamber; ++i ) {    
    size_t this_layer = rechits[i]->cscDetId().layer();
    //    std::cout << "[CSCSegAlgoDF::buildSegments] this_layer = " << this_layer << std::endl;
    layerIndex[i] = this_layer;
    //    std::cout << "[CSCSegAlgoDF::buildSegments] layerIndex[" << i << "] = " << layerIndex[i] << std::endl;
    if ( this_layer != old_layer ) {
      old_layer = this_layer;
      ++nLayers;   
    }
  }

  //  std::cout << "[CSCSegAlgoDF::buildSegments] layers with rechits = " << nLayers << std::endl;  
  
  // Return #2 - OK, there aren't enough hit layers to build a segment
  if ( nLayers < 3 ) return segmentInChamber;

  double z1 = theChamber->layer(1)->position().z();
  double z6 = theChamber->layer(6)->position().z();
  
  if ( z1 > 0. ) {
    if ( z1 > z6 ) { 
      reverse( layerIndex.begin(), layerIndex.end() );
      reverse( rechits.begin(),    rechits.end() );
    }    
  }
  else if ( z1 < 0. ) {
    if ( z1 < z6 ) {
      reverse( layerIndex.begin(), layerIndex.end() );
      reverse( rechits.begin(),    rechits.end() );
    }    
  }

  //  std::cout << "[CSCSegAlgoDF::buildSegments] rechits have been ordered" << std::endl;  

  // Showering muon
  if ( preClustering && int(nHitInChamber) > nHitsPerClusterIsShower && nLayers > 2 ) {

    // std::cout << "[CSCSegAlgoDF::buildSegments] showering block" << std::endl;  

    CSCSegment segShower = showering_->showerSeg(theChamber, rechits);

    // Return #3 - OK, this is now 'effectve' rechits 
    // Make sure have at least 3 hits...
    if ( segShower.nRecHits() < 3 ) return segmentInChamber;

    segmentInChamber.push_back(segShower);
    if (debug) dumpSegment( segShower );

    // Return #4 - OK, only get one try at building a segment from shower
    return segmentInChamber;
  }

  // Initialize flags that a given hit has been allocated to a segment
  BoolContainer used_ini(rechits.size(), false);
  usedHits = used_ini;
  
  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();

  //  std::cout << "[CSCSegAlgoDF::buildSegments] entering rechit loop" << std::endl;
	
  // Now Loop over hits within the chamber to find 1st seed for segment building
  for ( ChamberHitContainerCIt i1 = ib; i1 < ie; ++i1 ) {
    if ( usedHits[i1-ib] ) continue;

    const CSCRecHit2D* h1 = *i1;
    int layer1 = layerIndex[i1-ib];
    const CSCLayer* l1 = theChamber->layer(layer1);
    GlobalPoint gp1 = l1->toGlobal(h1->localPosition());
    LocalPoint lp1 = theChamber->toLocal(gp1);  
           
    // Loop over hits backward to find 2nd seed for segment building
    for ( ChamberHitContainerCIt i2 = ie-1; i2 > ib; --i2 ) {	

      if ( usedHits[i2-ib] ) continue;   // Hit has been used already

      int layer2 = layerIndex[i2-ib];
      if ( (layer2 - layer1) < minLayersApart ) continue;

      const CSCRecHit2D* h2 = *i2;
      const CSCLayer* l2 = theChamber->layer(layer2);
      GlobalPoint gp2 = l2->toGlobal(h2->localPosition());
      LocalPoint lp2 = theChamber->toLocal(gp2);  

      // Clear proto segment so it can be (re)-filled 
      protoSegment.clear();

      // We want hit wrt chamber (and local z will be != 0)
      float dz = gp2.z()-gp1.z();
      float slope_u = (lp2.x() - lp1.x())/dz ;
      float slope_v = (lp2.y() - lp1.y())/dz ;    

      // Test if entrance angle is roughly pointing towards IP
      if (fabs(slope_v) > tanThetaMax) continue;
      if (fabs(slope_u) > tanPhiMax ) continue;
     
      protoSegment.push_back(h1);
      protoSegment.push_back(h2);

      //      std::cout << "[CSCSegAlgoDF::buildSegments] about to fit 2 hits on layers "
      //		<< layer1 << " and " << layer2 << std::endl;

      // protoSegment has just 2 hits - but need to create a CSCSegFit to hold it in case 
      // we fail to add any more hits
      updateParameters();

      // Try adding hits to proto segment
      tryAddingHitsToSegment(rechits, i1, i2, layerIndex); 
	
      // Check no. of hits on segment to see if segment is large enough
      bool segok = true;
      unsigned iadd = 0;

      if (protoSegment.size() < minHitsPerSegment+iadd) segok = false;
  
      if ( Pruning && segok ) pruneFromResidual();

      // Check if segment satisfies chi2 requirement
      if ( sfit_->chi2() > chi2Max) segok = false;

      if ( segok ) {

        // Create an actual CSCSegment - retrieve all info from the current fit
        CSCSegment temp(sfit_->hits(), sfit_->intercept(), sfit_->localdir(), 
                              sfit_->covarianceMatrix(), sfit_->chi2());
	//	std::cout << "[CSCSegAlgoDF::buildSegments] about to delete sfit= = " << sfit_ << std::endl;
	delete sfit_;
        sfit_ = 0; // avoid possibility of attempting a second delete later

        segmentInChamber.push_back(temp); 
	if (debug) dumpSegment( temp );

	// Return #5 - OK, fewer than 3 rechits not on this segment left in chamber
        if (nHitInChamber-protoSegment.size() < 3) return segmentInChamber; 
	// Return $6 - already have more than 4 segments in this chamber
        if (segmentInChamber.size() > 4) return segmentInChamber;

        // Flag used hits
        flagHitsAsUsed(rechits);
      } 
    } 
  }
  // Return #7
  return segmentInChamber;
}


/* Method tryAddingHitsToSegment
 *
 * Look at left over hits and try to add them to proto segment by looking how far they
 * are from the segment in terms of the hit error matrix (so how many sigmas away).
 *
 */
void CSCSegAlgoDF::tryAddingHitsToSegment( const ChamberHitContainer& rechits, 
                                           const ChamberHitContainerCIt i1, 
                                           const ChamberHitContainerCIt i2,
                                           const LayerIndex& layerIndex) {
  
/* Iterate over the layers with hits in the chamber
 * Skip the layers containing the segment endpoints on first pass, but then
 * try hits on layer containing the segment starting points on 2nd pass
 * if segment has >2 hits.  Once a hit is added to a layer, don't replace it 
 * until 2nd iteration
 */  


//  std::cout << "[CSCSegAlgoDF::tryAddingHitsToSegment] entering"  
//	    << " with rechits.size() = " << rechits.size() << std::endl;
  
  ChamberHitContainerCIt ib = rechits.begin();
  ChamberHitContainerCIt ie = rechits.end();
  closeHits.clear();


  //  int counter1 = 0;
  //  int counter2 = 0;

  for ( ChamberHitContainerCIt i = ib; i != ie; ++i ) {
    //    std::cout << "counter1 = " << ++counter1 << std::endl;
    if (i == i1 || i == i2 ) continue;
    if ( usedHits[i-ib] ) continue;   // Don't use hits already part of a segment.

    //    std::cout << "counter2 = " << ++counter2 << std::endl;
    const CSCRecHit2D* h = *i;
    int layer = layerIndex[i-ib];
    int layer1 = layerIndex[i1-ib];
    int layer2 = layerIndex[i2-ib];

    //    std::cout << "[CSCSegAlgoDF::tryAddingHitsToSegment] layer, layer1, layer2 = " 
    //	      << layer << ", " << layer1 << ", " << layer2 << std::endl;

    // Low multiplicity case
    // only adds hit to protoSegment if no hit on layer already; otherwise adds to closeHits
    if (rechits.size() < 9) {
      //      std::cout << "low mult" << std::endl;
      if ( isHitNearSegment( h ) ) {
        if ( !hasHitOnLayer(layer) ) {
          addHit(h, layer);
        } else {
          closeHits.push_back(h);
        }
      }

    // High multiplicity case
    // only adds hit to protoSegment if no hit on layer already AND then refits; otherwise adds to closeHits
    } else { 
      //      std::cout << "high mult" << std::endl;
      if ( isHitNearSegment( h ) ) {
	//	std::cout << "near seg" << std::endl;
        if ( !hasHitOnLayer(layer) ) {
	  //	  std::cout << "no hit on layer" << std::endl;
          if ( addHit(h, layer) ) {
	    //	    std::cout << "update fit" << std::endl;
             updateParameters();
	  }
        // Don't change the starting points at this stage !!!
        } else {
	  //	  std::cout << "already hit on layer" << std::endl;
          closeHits.push_back(h);
          if (layer != layer1 && layer != layer2 ) compareProtoSegment(h, layer);
        }
      }
    }
  }
 
  if ( int(protoSegment.size()) < 3) return;
  //  std::cout << "final fit" << std::endl;
  updateParameters();

  // 2nd pass to remove biases 
  // This time, also consider changing the endpoints
  for ( ChamberHitContainerCIt i = closeHits.begin() ; i != closeHits.end(); ++i ) {      
    //    std::cout << "2nd pass" << std::endl;
    const CSCRecHit2D* h = *i;      
    int layer = (*i)->cscDetId().layer();     
    compareProtoSegment(h, layer); 
  } 

}


/* isHitNearSegment
 *
 * Compare rechit with expected position from protosegment
 */
bool CSCSegAlgoDF::isHitNearSegment( const CSCRecHit2D* hit ) const {

  const CSCLayer* layer = theChamber->layer(hit->cscDetId().layer());

  // hit phi position in global coordinates
  GlobalPoint Hgp = layer->toGlobal(hit->localPosition());
  float Hphi = Hgp.phi();                                
  if (Hphi < 0.) Hphi += 2.*M_PI;
  LocalPoint Hlp = theChamber->toLocal(Hgp);
  float z = Hlp.z();  

  float LocalX = sfit_->xfit(z); 
  float LocalY = sfit_->yfit(z); 

  LocalPoint Slp(LocalX, LocalY, z);
  GlobalPoint Sgp = theChamber->toGlobal(Slp); 
  float Sphi = Sgp.phi();
  if (Sphi < 0.) Sphi += 2.*M_PI;
  float R = sqrt(Sgp.x()*Sgp.x() + Sgp.y()*Sgp.y());
  
  float deltaPhi = Sphi - Hphi;
  if (deltaPhi >  2.*M_PI) deltaPhi -= 2.*M_PI;
  if (deltaPhi < -2.*M_PI) deltaPhi += 2.*M_PI;
  if (deltaPhi < 0.) deltaPhi = -deltaPhi; 

  float RdeltaPhi = R * deltaPhi;

  if (RdeltaPhi < dRPhiFineMax && deltaPhi < dPhiFineMax ) return true;

  return false;
}


/* Method addHit
 *
 * Test if can add hit to proto segment. If so, try to add it.
 *
 */
bool CSCSegAlgoDF::addHit(const CSCRecHit2D* aHit, int layer) {


  //  std::cout << "[CSCSegAlgoDF::addHit] on layer " << layer << " to protoSegment.size() = " 
  //	    << protoSegment.size() << std::endl;
  
  // Return true if hit was added successfully and then parameters are updated.
  // Return false if there is already a hit on the same layer, or insert failed.
  
  if ( protoSegment.size() > 5 ) return false; //@@ can only have 6 hits at most
  
  // Test that we are not trying to add the same hit again
  for ( ChamberHitContainer::const_iterator it = protoSegment.begin(); it != protoSegment.end(); ++it ) 
    if ( aHit == (*it)  ) return false;
  
  protoSegment.push_back(aHit);

  return true;
}    


/* Method updateParameters
 *      
 * Perform a simple Least Square Fit on proto segment to determine slope and intercept
 *
 */   
void CSCSegAlgoDF::updateParameters() {

  // Delete existing CSCSegFit, create a new one and make the fit
  // Uses internal variables - theChamber & protoSegment

  //  std::cout << "[CSCSegAlgoDF::updateParameters] about to delete sfit_ = " << sfit_ << std::endl;
  delete sfit_;
  //  std::cout << "[CSCSegAlgoDF::updateParameters] protoSegment.size() = " 
  //                                 << protoSegment.size() << std::endl;
  //  std::cout  << "[CSCSegAlgoDF::updateParameters] theChamber = " << theChamber << std::endl;
  sfit_ = new CSCSegFit( theChamber, protoSegment );
  //  std::cout << "[CSCSegAlgoDF::updateParameters] new sfit_ = " << sfit_ << std::endl;
  sfit_->fit();
}

/* hasHitOnLayer
 *
 * Just make sure hit to be added to layer comes from different layer than those in proto segment   
 */
bool CSCSegAlgoDF::hasHitOnLayer(int layer) const {
  

  //  std::cout << "[CSCSegAlgoDF::hasHitOnLayer] on layer " << layer << std::endl;


  // Is there already a hit on this layer?
  for ( ChamberHitContainerCIt it = protoSegment.begin(); it != protoSegment.end(); it++ )
    if ( (*it)->cscDetId().layer() == layer ) return true;
  
  return false;
}


/* Method compareProtoSegment
 *      
 * For hit coming from the same layer of an existing hit within the proto segment
 * test if achieve better chi^2 by using this hit than the other
 *
 */ 
void CSCSegAlgoDF::compareProtoSegment(const CSCRecHit2D* h, int layer) {


  //  std::cout << "[CSCSegAlgoDF::compareProtoSegment] for hit on layer " << layer 
  //	    << " with protoSegment.size() = " << protoSegment.size() << std::endl;

  // Try adding the hit to existing segment, and remove old one existing in same layer
  ChamberHitContainer::iterator it;
  for ( it = protoSegment.begin(); it != protoSegment.end(); ) {
    if ( (*it)->cscDetId().layer() == layer ) {
      it = protoSegment.erase(it);
    } else {
      ++it;
    }
  }

  //  std::cout << "[CSCSegAlgoDF::compareProtoSegment] about to add hit on layer " << layer 
  //	    << " with protoSegment.size() = " << protoSegment.size() << std::endl;

  bool ok = addHit(h, layer);

  CSCSegFit* newfit = 0;
  if ( ok ) {
    newfit = new CSCSegFit( theChamber, protoSegment );
    //    std::cout << "[CSCSegAlgoDF::compareProtoSegment] newfit = " << newfit << std::endl;
    newfit->fit();
  }
  if ( !ok || (newfit->chi2() > sfit_->chi2()) ) {
    //    std::cout << "[CSCSegAlgoDF::compareProtoSegment] about to delete newfit = " << newfit << std::endl;
    delete newfit;   // failed to add a hit or new fit is worse
  }
  else {
    //    std::cout << "[CSCSegAlgoDF::compareProtoSegment] about to delete sfit_ = " << sfit_ << std::endl;
    delete sfit_;  // new fit is better
    sfit_ = newfit;
    //    std::cout << "[CSCSegAlgoDF::compareProtoSegment] reset sfit_ = " << sfit_ << std::endl;
  }
}


/* Method flagHitsAsUsed
 *
 * Flag hits which have entered segment building so we don't reuse them.
 * Also flag does which were very close to segment to reduce combinatorics
 */
void CSCSegAlgoDF::flagHitsAsUsed(const ChamberHitContainer& rechitsInChamber) {
  
  // Flag hits on segment as used
  ChamberHitContainerCIt ib = rechitsInChamber.begin();
  ChamberHitContainerCIt hi, iu;
  
  for ( hi = protoSegment.begin(); hi != protoSegment.end(); ++hi ) {
    for ( iu = ib; iu != rechitsInChamber.end(); ++iu ) {
      if (*hi == *iu) usedHits[iu-ib] = true;
    }
  }
  // Don't reject hits marked as "nearby" for now.
  // So this is bypassed at all times for now !!!
  // Perhaps add back to speed up algorithm some more
  if (closeHits.size() > 0) return;  
  for ( hi = closeHits.begin(); hi != closeHits.end(); ++hi ) {
    for ( iu = ib; iu != rechitsInChamber.end(); ++iu ) {
      if (*hi == *iu) usedHits[iu-ib] = true;
    }
  }

}


// Try to clean up segments by quickly looking at residuals
void CSCSegAlgoDF::pruneFromResidual(void){

  // Only prune if have at least 5 hits 
  if ( protoSegment.size() < 5 ) return ;

  // Now Study residuals
      
  float maxResidual = 0.;
  float sumResidual = 0.;
  int nHits = 0;
  int badIndex = -1;
  int j = 0;


  ChamberHitContainer::const_iterator ih;

  for ( ih = protoSegment.begin(); ih != protoSegment.end(); ++ih ) {
    const CSCRecHit2D& hit = (**ih);
    const CSCLayer* layer  = theChamber->layer(hit.cscDetId().layer());
    GlobalPoint gp         = layer->toGlobal(hit.localPosition());
    LocalPoint lp          = theChamber->toLocal(gp);

    float residual = sfit_->Rdev(lp.x(), lp.y(), lp.z());

    sumResidual += residual;
    nHits++;
    if ( residual > maxResidual ) {
      maxResidual = residual;
      badIndex = j;
    }
    j++;
  }

  float corrAvgResidual = (sumResidual - maxResidual)/(nHits -1);

  // Keep all hits 
  if ( maxResidual/corrAvgResidual < maxRatioResidual ) return;


  // Drop worse hit and recompute segment properties + fill

  ChamberHitContainer newProtoSegment;

  j = 0;
  for ( ih = protoSegment.begin(); ih != protoSegment.end(); ++ih ) {
    if ( j != badIndex ) newProtoSegment.push_back(*ih);
    j++;
  }
  
  protoSegment.clear();

  for ( ih = newProtoSegment.begin(); ih != newProtoSegment.end(); ++ih ) {
    protoSegment.push_back(*ih);
  }

  // Compute segment parameters
  updateParameters();

}

void CSCSegAlgoDF::dumpSegment( const CSCSegment& seg ) const {

  edm::LogVerbatim("CSCSegment") << "CSCSegment in " << theChamber->id()
                                 << "\nlocal position = " << seg.localPosition()
                                 << "\nerror = " << seg.localPositionError()
                                 << "\nlocal direction = " << seg.localDirection()
                                 << "\nerror =" << seg.localDirectionError()
                                 << "\ncovariance matrix"
                                 << seg.parametersError()
                                 << "chi2/ndf = " << seg.chi2() << "/" << seg.degreesOfFreedom()
                                 << "\n#rechits = " << seg.specificRecHits().size()
                                 << "\ntime = " << seg.time();
}
