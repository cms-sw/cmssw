/**
 *  See header file for a description of this class.
 *  
 *  All the code is under revision
 *
 *  $Date: 2009/02/07 01:31:56 $
 *  $Revision: 1.6 $
 *
 *  \author A. Vitelli - INFN Torino, V.Palichik
 *  \author ported by: R. Bellan - INFN Torino
 */


#include "RecoMuon/MuonSeedGenerator/src/MuonSeedOrcaPatternRecognition.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"

#include "DataFormats/Common/interface/Handle.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// C++
#include <vector>

using namespace std;

    const std::string metname = "Muon|RecoMuon|MuonSeedOrcaPatternRecognition";

// Constructor
MuonSeedOrcaPatternRecognition::MuonSeedOrcaPatternRecognition(const edm::ParameterSet& pset)
: MuonSeedVPatternRecognition(pset),
  theCrackEtas(pset.getParameter<std::vector<double> >("crackEtas")),
  theCrackWindow(pset.getParameter<double>("crackWindow"))
{
}


// reconstruct muon's seeds
void MuonSeedOrcaPatternRecognition::produce(const edm::Event& event, const edm::EventSetup& eSetup,
                                             std::vector<MuonRecHitContainer> & result)
{
  // divide the RecHits by DetLayer, in order to fill the
  // RecHitContainer like it was in ORCA
  
  // Muon Geometry - DT, CSC and RPC 
  edm::ESHandle<MuonDetLayerGeometry> muonLayers;
  eSetup.get<MuonRecoGeometryRecord>().get(muonLayers);

  // get the DT layers
  vector<DetLayer*> dtLayers = muonLayers->allDTLayers();

  // get the CSC layers
  vector<DetLayer*> cscForwardLayers = muonLayers->forwardCSCLayers();
  vector<DetLayer*> cscBackwardLayers = muonLayers->backwardCSCLayers();
    
  // Backward (z<0) EndCap disk
  const DetLayer* ME4Bwd = cscBackwardLayers[4];
  const DetLayer* ME3Bwd = cscBackwardLayers[3];
  const DetLayer* ME2Bwd = cscBackwardLayers[2];
  const DetLayer* ME12Bwd = cscBackwardLayers[1];
  const DetLayer* ME11Bwd = cscBackwardLayers[0];
  
  // Forward (z>0) EndCap disk
  const DetLayer* ME11Fwd = cscForwardLayers[0];
  const DetLayer* ME12Fwd = cscForwardLayers[1];
  const DetLayer* ME2Fwd = cscForwardLayers[2];
  const DetLayer* ME3Fwd = cscForwardLayers[3];
  const DetLayer* ME4Fwd = cscForwardLayers[4];
     
  // barrel
  const DetLayer* MB4DL = dtLayers[3];
  const DetLayer* MB3DL = dtLayers[2];
  const DetLayer* MB2DL = dtLayers[1];
  const DetLayer* MB1DL = dtLayers[0];
  
  // instantiate the accessor
  // Don not use RPC for seeding
  MuonDetLayerMeasurements muonMeasurements(theDTRecSegmentLabel.label(),theCSCRecSegmentLabel,edm::InputTag(),
					    enableDTMeasurement,enableCSCMeasurement,false);

  MuonRecHitContainer list9 = muonMeasurements.recHits(MB4DL,event);
  MuonRecHitContainer list6 = muonMeasurements.recHits(MB3DL,event);
  MuonRecHitContainer list7 = muonMeasurements.recHits(MB2DL,event);
  MuonRecHitContainer list8 = muonMeasurements.recHits(MB1DL,event);

  if(false) {
    dumpLayer("MB4 ", list9);
    dumpLayer("MB3 ", list6);
    dumpLayer("MB2 ", list7);
    dumpLayer("MB1 ", list8);
  }


  bool* MB1 = zero(list8.size());
  bool* MB2 = zero(list7.size());
  bool* MB3 = zero(list6.size());

  endcapPatterns(muonMeasurements.recHits(ME11Bwd,event),
                 muonMeasurements.recHits(ME12Bwd,event),
                 muonMeasurements.recHits(ME2Bwd,event),
                 muonMeasurements.recHits(ME3Bwd,event),
                 muonMeasurements.recHits(ME4Bwd,event),
                 list8, list7, list6,
                 MB1, MB2, MB3, result);

  endcapPatterns(muonMeasurements.recHits(ME11Fwd,event),
                 muonMeasurements.recHits(ME12Fwd,event),
                 muonMeasurements.recHits(ME2Fwd,event),
                 muonMeasurements.recHits(ME3Fwd,event),
                 muonMeasurements.recHits(ME4Fwd,event),
                 list8, list7, list6,
                 MB1, MB2, MB3, result);


  // ----------    Barrel only
  
  unsigned int counter = 0;
  if ( list9.size() < 100 ) {   // +v
    for (MuonRecHitContainer::iterator iter=list9.begin(); iter!=list9.end(); iter++ ){
      MuonRecHitContainer seedSegments;
      seedSegments.push_back(*iter);
      complete(seedSegments, list6, MB3);
      complete(seedSegments, list7, MB2);
      complete(seedSegments, list8, MB1);
      if(check(seedSegments)) result.push_back(seedSegments);
    }
  }


  if ( list6.size() < 100 ) {   // +v
    for ( counter = 0; counter<list6.size(); counter++ ){
      if ( !MB3[counter] ) { 
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list6[counter]);
	complete(seedSegments, list7, MB2);
	complete(seedSegments, list8, MB1);
	complete(seedSegments, list9);
        if(check(seedSegments)) result.push_back(seedSegments);
      }
    }
  }


  if ( list7.size() < 100 ) {   // +v
    for ( counter = 0; counter<list7.size(); counter++ ){
      if ( !MB2[counter] ) { 
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list7[counter]);
	complete(seedSegments, list8, MB1);
	complete(seedSegments, list9);
	complete(seedSegments, list6, MB3);
	if (seedSegments.size()>1 || 
           (seedSegments.size()==1 && seedSegments[0]->dimension()==4) )
        {
          result.push_back(seedSegments);
	}
      }
    }
  }


  if ( list8.size() < 100 ) {   // +v
    for ( counter = 0; counter<list8.size(); counter++ ){
      if ( !MB1[counter] ) { 
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list8[counter]);
	complete(seedSegments, list9);
	complete(seedSegments, list6, MB3);
	complete(seedSegments, list7, MB2);
        if (seedSegments.size()>1 ||
           (seedSegments.size()==1 && seedSegments[0]->dimension()==4) )
        {
          result.push_back(seedSegments);
	}
      }
    }
  }

  if ( MB3 ) delete [] MB3;
  if ( MB2 ) delete [] MB2;
  if ( MB1 ) delete [] MB1;


  if(result.empty()) 
  {
    MuonRecHitContainer all = muonMeasurements.recHits(ME4Bwd,event);
    MuonRecHitContainer tmp = muonMeasurements.recHits(ME3Bwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME2Bwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME12Bwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME11Bwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME11Fwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME12Fwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME2Fwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME3Fwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(ME4Fwd,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(MB4DL,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(MB3DL,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(MB2DL,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    tmp = muonMeasurements.recHits(MB1DL,event);
    copy(tmp.begin(),tmp.end(),back_inserter(all));

    LogTrace(metname)<<"Number of segments: "<<all.size();

    for(MuonRecHitContainer::const_iterator segmentItr = all.begin();
        segmentItr != all.end(); ++segmentItr)
    {
      MuonRecHitContainer singleSegmentContainer;
      singleSegmentContainer.push_back(*segmentItr);
      result.push_back(singleSegmentContainer);
    }
  }

}


bool * MuonSeedOrcaPatternRecognition::zero(unsigned listSize)
{
  bool * result = 0;
  if (listSize) {
    result = new bool[listSize]; 
    for ( size_t i=0; i<listSize; i++ ) result[i]=false;
  }
  return result;
}


void MuonSeedOrcaPatternRecognition::endcapPatterns(
  const MuonRecHitContainer & me11, const MuonRecHitContainer & me12,
  const MuonRecHitContainer & me2,  const MuonRecHitContainer & me3,
  const MuonRecHitContainer & me4,  const  MuonRecHitContainer & mb1,
  const MuonRecHitContainer & mb2,  const  MuonRecHitContainer & mb3,
  bool * MB1, bool * MB2, bool * MB3,
  std::vector<MuonRecHitContainer> & result)
{
  if(false) {
    dumpLayer("ME4 ", me4);
    dumpLayer("ME3 ", me3);
    dumpLayer("ME2 ", me2);
    dumpLayer("ME12 ", me12);
    dumpLayer("ME11 ", me11);
  }

  std::vector<MuonRecHitContainer> patterns;
  MuonRecHitContainer crackSegments;
  rememberCrackSegments(me11, crackSegments);
  rememberCrackSegments(me12, crackSegments);
  rememberCrackSegments(me2,  crackSegments);
  rememberCrackSegments(me3,  crackSegments);
  rememberCrackSegments(me4,  crackSegments);


  MuonRecHitContainer list24 = me4;
  MuonRecHitContainer list23 = me3;

  MuonRecHitContainer list12 = me2;

  MuonRecHitContainer list22 = me12;
  MuonRecHitContainer list21 = me11;

  MuonRecHitContainer list11 = list21;
  MuonRecHitContainer list5 = list22;
  MuonRecHitContainer list13 = list23;
  MuonRecHitContainer list4 = list24;

  if ( list21.size() == 0 )  {
    list11 = list22; list5 = list21;
  }

  if ( list24.size() < list23.size() && list24.size() > 0 )  {
    list13 = list24; list4 = list23;
  }

  if ( list23.size() == 0 )  {
    list13 = list24; list4 = list23;
  }

  MuonRecHitContainer list1 = list11;
  MuonRecHitContainer list2 = list12;
  MuonRecHitContainer list3 = list13;


  if ( list12.size() == 0 )  {
    list3 = list12;
    if ( list11.size() <= list13.size() && list11.size() > 0 ) {
      list1 = list11; list2 = list13;}
    else { list1 = list13; list2 = list11;}
  }

  if ( list13.size() == 0 )  {
    if ( list11.size() <= list12.size() && list11.size() > 0 ) {
      list1 = list11; list2 = list12;}
    else { list1 = list12; list2 = list11;}
  }

  if ( list12.size() != 0 &&  list13.size() != 0 )  {
    if ( list11.size()<=list12.size() && list11.size()<=list13.size() && list11.size()>0 ) {   // ME 1
      if ( list12.size() > list13.size() ) {
        list2 = list13; list3 = list12;}
    }
    else if ( list12.size() <= list13.size() ) {                                   //  start with ME 2
      list1 = list12;
      if ( list11.size() <= list13.size() && list11.size() > 0 ) {
        list2 = list11; list3 = list13;}
      else { list2 = list13; list3 = list11;}
    }
    else {                                                                         //  start with ME 3
      list1 = list13;
      if ( list11.size() <= list12.size() && list11.size() > 0 ) {
        list2 = list11; list3 = list12;}
      else { list2 = list12; list3 = list11;}
    }
  }


  bool* ME2 = zero(list2.size());
  bool* ME3 = zero(list3.size());
  bool* ME4 = zero(list4.size());
  bool* ME5 = zero(list5.size());


  // creates list of compatible track segments

  for (MuonRecHitContainer::iterator iter = list1.begin(); iter!=list1.end(); iter++ ){
    if ( (*iter)->recHits().size() < 4 && list3.size() > 0 ) continue; // 3p.tr-seg. are not so good for starting
    MuonRecHitContainer seedSegments;
    seedSegments.push_back(*iter);
    complete(seedSegments, list2, ME2);
    complete(seedSegments, list3, ME3);
    complete(seedSegments, list4, ME4);
    complete(seedSegments, list5, ME5);
    complete(seedSegments, mb3, MB3);
    complete(seedSegments, mb2, MB2);
    complete(seedSegments, mb1, MB1);
    if(check(seedSegments)) patterns.push_back(seedSegments);
  }


  unsigned int counter;

  for ( counter = 0; counter<list2.size(); counter++ ){

    if ( !ME2[counter] ) {
      MuonRecHitContainer seedSegments;
      seedSegments.push_back(list2[counter]);
      complete(seedSegments, list3, ME3);
      complete(seedSegments, list4, ME4);
      complete(seedSegments, list5, ME5);
      complete(seedSegments, mb3, MB3);
      complete(seedSegments, mb2, MB2);
      complete(seedSegments, mb1, MB1);
      if(check(seedSegments)) patterns.push_back(seedSegments);
    }
  }


  if ( list3.size() < 20 ) {   // +v
    for ( counter = 0; counter<list3.size(); counter++ ){
      if ( !ME3[counter] ) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list3[counter]);
        complete(seedSegments, list4, ME4);
        complete(seedSegments, list5, ME5);
        complete(seedSegments, mb3, MB3);
        complete(seedSegments, mb2, MB2);
        complete(seedSegments, mb1, MB1);
        if(check(seedSegments)) patterns.push_back(seedSegments);
      }
    }
  }

  if ( list4.size() < 20 ) {   // +v
    for ( counter = 0; counter<list4.size(); counter++ ){
      if ( !ME4[counter] ) {
        MuonRecHitContainer seedSegments;
        seedSegments.push_back(list4[counter]);
        complete(seedSegments, list5, ME5);
        complete(seedSegments, mb3, MB3);
        complete(seedSegments, mb2, MB2);
        complete(seedSegments, mb1, MB1);
        if(check(seedSegments)) patterns.push_back(seedSegments);
      }
    }
  }

  if ( ME5 ) delete [] ME5;
  if ( ME4 ) delete [] ME4;
  if ( ME3 ) delete [] ME3;
  if ( ME2 ) delete [] ME2;

  if(!patterns.empty())
  {
    result.insert(result.end(), patterns.begin(), patterns.end());
  }
  else
  {
    if(!crackSegments.empty())
    {
       // make some single-segment seeds
       for(MuonRecHitContainer::const_iterator crackSegmentItr = crackSegments.begin();
           crackSegmentItr != crackSegments.end(); ++crackSegmentItr)
       {
          MuonRecHitContainer singleSegmentPattern;
          singleSegmentPattern.push_back(*crackSegmentItr);
          result.push_back(singleSegmentPattern);
       }
    }
  }
}



void MuonSeedOrcaPatternRecognition::complete(MuonRecHitContainer& seedSegments,
                                 const MuonRecHitContainer &recHits, bool* used) const {

  MuonRecHitContainer good_rhit;

  //+v get all rhits compatible with the seed on dEta/dPhi Glob.

  ConstMuonRecHitPointer first = seedSegments[0]; // first rechit of seed

  GlobalPoint ptg2 = first->globalPosition(); // its global pos +v

  int nr=0; // count rechits we have checked against seed

  for (MuonRecHitContainer::const_iterator iter=recHits.begin(); iter!=recHits.end(); iter++){

    GlobalPoint ptg1 = (*iter)->globalPosition();  //+v global pos of rechit
    float deta = fabs (ptg1.eta()-ptg2.eta());
    // Geom::Phi should keep it in the range [-pi, pi]
    float dphi = fabs (ptg1.phi()-ptg2.phi());
    float eta2 = fabs( ptg2.eta() );
    // be a little more lenient in cracks
    bool crack = isCrack(*iter) || isCrack(first);
    float detaWindow = crack ? 0.25 : 0.2;
    if ( deta > detaWindow || dphi > .1 ) {
      nr++;
      continue;
    }   // +vvp!!!

    if( eta2 < 1.0 ) {     //  barrel only
      LocalPoint pt1 = first->det()->toLocal(ptg1); // local pos of rechit in seed's det

      LocalVector dir1 = first->localDirection();

      LocalPoint pt2 = first->localPosition();

      float m = dir1.z()/dir1.x();   // seed's slope in local xz
      float yf = pt1.z();            // local z of rechit
      float yi = pt2.z();            // local z of seed
      float xi = pt2.x();            // local x of seed
      float xf = (yf-yi)/m + xi;     // x of linear extrap alone seed direction to z of rechit
      float dist = fabs ( xf - pt1.x() ); // how close is actual to predicted local x ?

      float d_cut = sqrt((yf-yi)*(yf-yi)+(pt1.x()-pt2.x())*(pt1.x()-pt2.x()))/10.;


      //@@ Tim asks: what is the motivation for this cut?
      //@@ It requires (xpred-xrechit)< 0.1 * distance between rechit and seed in xz plane
      if ( dist < d_cut ) {
	good_rhit.push_back(*iter);
	if (used) used[nr]=true;
      }

    }  // eta  < 1.0

    else {    //  endcap & overlap.
      // allow a looser dphi cut where bend is greatest, so we get those little 5-GeV muons
      // watch out for ghosts from ME1/A, below 2.0.
      float dphicut = (eta2 > 1.6 && eta2 < 2.0) ? 0.1 : 0.07;
      // segments at the edge of the barrel may not have a good eta measurement
      float detacut = (first->isDT() || (*iter)->isDT()) ? 0.2 : 0.1;

      if ( deta < detacut && dphi < dphicut ) {
	good_rhit.push_back(*iter);
	if (used) used[nr]=true;
      }

    }  // eta > 1.0


    nr++;

  }  // recHits iter

  // select the best rhit among the compatible ones (based on Dphi Glob & Dir)

  MuonRecHitPointer best=0;

  float best_dphiG = M_PI;
  float best_dphiD = M_PI;

  if( fabs ( ptg2.eta() ) > 1.0 ) {    //  endcap & overlap.
      
    // select the best rhit among the compatible ones (based on Dphi Glob & Dir)
      
    GlobalVector dir2 =  first->globalDirection();
   
    GlobalPoint  pos2 =  first->globalPosition();  // +v
      
    for (MuonRecHitContainer::iterator iter=good_rhit.begin(); iter!=good_rhit.end(); iter++){

      GlobalPoint pos1 = (*iter)->globalPosition();  // +v
 
      float dphi = pos1.phi()-pos2.phi();       //+v

      if (dphi < 0.) dphi = -dphi;             //+v
      if (dphi > M_PI) dphi = 2.*M_PI - dphi;  //+v

      if (  dphi < best_dphiG*1.5 ) {  


	if (  dphi < best_dphiG*.67  && best_dphiG > .005 )  best_dphiD = M_PI;  // thresh. of strip order

	GlobalVector dir1 = (*iter)->globalDirection();
	
	float  dphidir = fabs ( dir1.phi()-dir2.phi() );

	if (dphidir > M_PI) dphidir = 2.*M_PI - dphidir;
	if (dphidir > M_PI*.5) dphidir = M_PI - dphidir;  // +v  [0,pi/2]
	if (  dphidir < best_dphiD ) {

	  best_dphiG = dphi;
	  if ( dphi < .002 )  best_dphiG =  .002;                          // thresh. of half-strip order
	  best_dphiD = dphidir;
	  best = (*iter);

	}

      }


    }   //  rhit iter

  }  // eta > 1.0

  if( fabs ( ptg2.eta() ) < 1.0 ) {     //  barrel only

    // select the best rhit among the compatible ones (based on Dphi)

    float best_dphi = M_PI;

    for (MuonRecHitContainer::iterator iter=good_rhit.begin(); iter!=good_rhit.end(); iter++){
      GlobalVector dir1 = (*iter)->globalDirection();

      //@@ Tim: Why do this again? 'first' hasn't changed, has it?
      //@@ I comment it out.
      //    RecHit first = seed.rhit();
      
      GlobalVector dir2 = first->globalDirection();
      
      float dphi = dir1.phi()-dir2.phi();

      if (dphi < 0.) dphi = -dphi;
      if (dphi > M_PI) dphi = 2.*M_PI - dphi;

      if (  dphi < best_dphi ) {

	best_dphi = dphi;
	best = (*iter);
      }

    }   //  rhit iter

  }  // eta < 1.0


  // add the best Rhit to the seed 
  if(best)
    if ( best->isValid() ) seedSegments.push_back(best);

}  //   void complete.



bool MuonSeedOrcaPatternRecognition::check(const MuonRecHitContainer & segments)
{
  return (segments.size() > 1);
}


bool MuonSeedOrcaPatternRecognition::isCrack(const ConstMuonRecHitPointer & segment) const
{
  bool result = false;
  double absEta = fabs(segment->globalPosition().eta());
  for(std::vector<double>::const_iterator crackItr = theCrackEtas.begin();
      crackItr != theCrackEtas.end(); ++crackItr)
  {
    if(fabs(absEta-*crackItr) < theCrackWindow) {
      result = true;
    }
  }
  return result;
}


void MuonSeedOrcaPatternRecognition::rememberCrackSegments(const MuonRecHitContainer & segments,
                                                           MuonRecHitContainer & crackSegments) const
{
  for(MuonRecHitContainer::const_iterator segmentItr = segments.begin(); 
      segmentItr != segments.end(); ++segmentItr)
  {
    if((**segmentItr).hit()->dimension() == 4 && isCrack(*segmentItr)) 
    {
       crackSegments.push_back(*segmentItr);
    }
  }
}


#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"


void MuonSeedOrcaPatternRecognition::dumpLayer(const char * name, const MuonRecHitContainer & segments) const
{
  MuonPatternRecoDumper theDumper;

  LogTrace(metname) << name << std::endl;
  for(MuonRecHitContainer::const_iterator segmentItr = segments.begin();
      segmentItr != segments.end(); ++segmentItr)
  {
    LogTrace(metname) << theDumper.dumpMuonId((**segmentItr).geographicalId());
  }
}


