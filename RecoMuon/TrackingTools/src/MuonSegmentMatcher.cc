// -*- C++ -*- // // Package:  MuonSegmentMatcher // Class:  MuonSegmentMatcher // /**\class MuonSegmentMatcher MuonSegmentMatcher.cc
// Description: <one line class summary>
// Implementation:
//     <Notes on implementation>
//*/
//
// Original Author:  Alan Tua
//         Created:  Wed Jul  9 21:40:17 CEST 2008
//
//

#include "RecoMuon/TrackingTools/interface/MuonSegmentMatcher.h"


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// system include files
#include <vector>
#include <iostream>

class MuonServiceProxy;
class MuonSegmentMatcher;

using namespace std;

// constructors and destructor

MuonSegmentMatcher::MuonSegmentMatcher(const edm::ParameterSet& matchParameters, MuonServiceProxy* service,edm::ConsumesCollector& iC)
  :
  theService(service),
  DTSegmentTags_(matchParameters.getParameter<edm::InputTag>("DTsegments")),
  CSCSegmentTags_(matchParameters.getParameter<edm::InputTag>("CSCsegments")),
  dtRadius_(matchParameters.getParameter<double>("DTradius")),
  dtTightMatch(matchParameters.getParameter<bool>("TightMatchDT")),
  cscTightMatch(matchParameters.getParameter<bool>("TightMatchCSC"))
{
  dtRecHitsToken = iC.consumes<DTRecSegment4DCollection>(DTSegmentTags_);
  allSegmentsCSCToken = iC.consumes<CSCSegmentCollection>(CSCSegmentTags_) ;

}

MuonSegmentMatcher::~MuonSegmentMatcher()
{
}

// member functions
vector<const DTRecSegment4D*> MuonSegmentMatcher::matchDT(const reco::Track &muon, const edm::Event& event)
{
  using namespace edm;

  edm::Handle<DTRecSegment4DCollection> dtRecHits;
  event.getByToken(dtRecHitsToken, dtRecHits);  
  
  vector<const DTRecSegment4D*> pointerTo4DSegments;

  TrackingRecHitRefVector dtHits;

  bool segments = false;

  // Loop and select DT recHits
  for(trackingRecHit_iterator hit = muon.recHitsBegin(); hit != muon.recHitsEnd(); ++hit) {
    if (!(*hit)->isValid()) continue; 
    if ( (*hit)->geographicalId().det() != DetId::Muon ) continue; 
    if ( (*hit)->geographicalId().subdetId() != MuonSubdetId::DT ) continue;
    if ((*hit)->recHits().size()) 
      if ((*(*hit)->recHits().begin())->recHits().size()>1) segments = true;
    dtHits.push_back(*hit);
  }
  
  //  cout << "Muon DT hits found: " << dtHits.size() << " segments " << segments << endl;
  
  double PhiCutParameter=dtRadius_;
  double ZCutParameter=dtRadius_;
  double matchRatioZ=0;
  double matchRatioPhi=0;

  for (DTRecSegment4DCollection::const_iterator rechit = dtRecHits->begin(); rechit!=dtRecHits->end();++rechit) {
  
    LocalPoint pointLocal = rechit->localPosition();

    if (segments) {
      // Loop over muon recHits
      for(trackingRecHit_iterator hit = dtHits.begin(); hit != dtHits.end(); ++hit) {
					
	// Pick the one in the same DT Chamber as the muon
	DetId idT = (*hit)->geographicalId();
	if(!(rechit->geographicalId().rawId()==idT.rawId())) continue; 

        // and compare the local positions
        LocalPoint segLocal = (*hit)->localPosition();
        if ((fabs(pointLocal.x()-segLocal.x())<ZCutParameter) && 
            (fabs(pointLocal.y()-segLocal.y())<ZCutParameter)) 
          pointerTo4DSegments.push_back(&(*rechit));
      }
      continue;
    }

    double nhitsPhi = 0;
    double nhitsZ = 0;
	  
    if(rechit->hasZed()) {
      double countMuonDTHits = 0;
      double countAgreeingHits=0;

      const DTRecSegment2D* segmZ;
      segmZ = dynamic_cast<const DTRecSegment2D*>(rechit->zSegment());
      nhitsZ = segmZ->recHits().size(); 
		
      const vector<DTRecHit1D> hits1d = segmZ->specificRecHits();
      DTChamberId chamberSegIdT((segmZ->geographicalId()).rawId());
		
      // Loop over muon recHits
      for(trackingRecHit_iterator hit = dtHits.begin(); hit != dtHits.end(); ++hit) {

	if ( !(*hit)->isValid()) continue; 
	
	DetId idT = (*hit)->geographicalId();
	DTChamberId dtDetIdHitT(idT.rawId());
	DTSuperLayerId dtDetLayerIdHitT(idT.rawId());

	LocalPoint  pointLocal = (*hit)->localPosition();
					
	if ((chamberSegIdT==dtDetIdHitT) && (dtDetLayerIdHitT.superlayer()==2)) countMuonDTHits++;

        for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {

	  if ( !hiti->isValid()) continue; 

	  // Pick the one in the same DT Layer as the 1D hit
	  if(!(hiti->geographicalId().rawId()==idT.rawId())) continue; 

          // and compare the local positions
	  LocalPoint segLocal = hiti->localPosition();
//	  cout << "Zed Segment Point = "<<pointLocal<<"    Muon Point = "<<segLocal<<"  Dist:  "
//	       << (fabs(pointLocal.x()-segLocal.x()))+(fabs(pointLocal.y()-segLocal.y()))<< endl;
	  if ((fabs(pointLocal.x()-segLocal.x())<ZCutParameter) && 
	      (fabs(pointLocal.y()-segLocal.y())<ZCutParameter)) 
	    countAgreeingHits++;
	} //End Segment Hit Iteration
      } //End Muon Hit Iteration
		
      matchRatioZ = countMuonDTHits == 0 ? 0 : countAgreeingHits/countMuonDTHits;
      if (nhitsZ)
        if (countAgreeingHits/nhitsZ>matchRatioZ) matchRatioZ=countAgreeingHits/nhitsZ;
    } //End HasZed Check
			
    if(rechit->hasPhi()) {
      double countMuonDTHits = 0;
      double countAgreeingHits=0;

      //PREPARE PARAMETERS FOR SEGMENT DETECTOR GEOMETRY
      const DTRecSegment2D* segmPhi;
      segmPhi = dynamic_cast<const DTRecSegment2D*>(rechit->phiSegment());
      nhitsPhi = segmPhi->recHits().size();
		
      const vector<DTRecHit1D> hits1d = segmPhi->specificRecHits();
      DTChamberId chamberSegIdT((segmPhi->geographicalId()).rawId());
		
      // Loop over muon recHits
      for(trackingRecHit_iterator hit = dtHits.begin(); hit != dtHits.end(); ++hit) {

	if ( !(*hit)->isValid()) continue; 

	DetId idT = (*hit)->geographicalId();
	DTChamberId dtDetIdHitT(idT.rawId());
	DTSuperLayerId dtDetLayerIdHitT(idT.rawId());

	LocalPoint pointLocal = (*hit)->localPosition(); //Localposition is in DTLayer http://cmslxr.fnal.gov/lxr/source/DataFormats/DTRecHit/interface/DTRecHit1D.h

	if ((chamberSegIdT==dtDetIdHitT)&&((dtDetLayerIdHitT.superlayer()==1)||(dtDetLayerIdHitT.superlayer()==3))) 
	  countMuonDTHits++;

	for (vector<DTRecHit1D>::const_iterator hiti=hits1d.begin(); hiti!=hits1d.end(); hiti++) {

	  if ( !hiti->isValid()) continue; 

	  // Pick the one in the same DT Layer as the 1D hit
	  if(!(hiti->geographicalId().rawId()==idT.rawId())) continue; 
					 
          // and compare the local positions
	  LocalPoint segLocal = hiti->localPosition();
//	  cout << "     Phi Segment Point = "<<pointLocal<<"    Muon Point = "<<segLocal<<"  Dist:   " 
//	       << (fabs(pointLocal.x()-segLocal.x()))+(fabs(pointLocal.y()-segLocal.y()))<< endl;

	  if ((fabs(pointLocal.x()-segLocal.x())<PhiCutParameter) && 
	      (fabs(pointLocal.y()-segLocal.y())<PhiCutParameter))
	    countAgreeingHits++; 
	} // End Segment Hit Iteration
      } // End Muon Hit Iteration

      matchRatioPhi = countMuonDTHits != 0 ? countAgreeingHits/countMuonDTHits : 0;
      if (nhitsPhi)
        if (countAgreeingHits/nhitsPhi>matchRatioPhi) matchRatioPhi=countAgreeingHits/nhitsPhi;
    } // End HasPhi Check
//    DTChamberId chamberSegId2((rechit->geographicalId()).rawId());
    if (dtTightMatch && nhitsPhi && nhitsZ) {
      if((matchRatioPhi>0.9)&&(matchRatioZ>0.9)) {
//	cout<<"Making a tight match in Chamber "<<chamberSegId2<<endl;
	pointerTo4DSegments.push_back(&(*rechit));
      }
    } else {
      if((matchRatioPhi>0.9 && nhitsPhi)||(matchRatioZ>0.9 && nhitsZ)) {
//	cout<<"Making a loose match in Chamber "<<chamberSegId2<<endl;
	pointerTo4DSegments.push_back(&(*rechit));
      }
    }
    
  } //End DT Segment Iteration

  return pointerTo4DSegments;
}



vector<const CSCSegment*> MuonSegmentMatcher::matchCSC(const reco::Track& muon, const edm::Event& event)
{

  using namespace edm;

  edm::Handle<CSCSegmentCollection> allSegmentsCSC;
  event.getByToken(allSegmentsCSCToken, allSegmentsCSC);

  vector<const CSCSegment*> pointerToCSCSegments;

  double matchRatioCSC=0;
  int numCSC = 0;
  double CSCXCut = 0.001;
  double CSCYCut = 0.001;
  double countMuonCSCHits = 0;

  for(CSCSegmentCollection::const_iterator segmentCSC = allSegmentsCSC->begin(); segmentCSC != allSegmentsCSC->end(); segmentCSC++) {
    double CSCcountAgreeingHits=0;

    if ( !segmentCSC->isValid()) continue; 

    numCSC++;
    const vector<CSCRecHit2D>& CSCRechits2D = segmentCSC->specificRecHits();
    countMuonCSCHits = 0;
    CSCDetId myChamber((*segmentCSC).geographicalId().rawId());

    bool segments = false;

    for(trackingRecHit_iterator hitC = muon.recHitsBegin(); hitC != muon.recHitsEnd(); ++hitC) {
      if (!(*hitC)->isValid()) continue; 
      if ( (*hitC)->geographicalId().det() != DetId::Muon ) continue; 
      if ( (*hitC)->geographicalId().subdetId() != MuonSubdetId::CSC ) continue;
      if (!(*hitC)->isValid()) continue;
      if ( (*hitC)->recHits().size()>1) segments = true;

      //DETECTOR CONSTRUCTION
      DetId id = (*hitC)->geographicalId();
      CSCDetId cscDetIdHit(id.rawId());

      if (segments) {
	if(!(myChamber.rawId()==cscDetIdHit.rawId())) continue; 

        // and compare the local positions
        LocalPoint positionLocalCSC = (*hitC)->localPosition();
	LocalPoint segLocalCSC = segmentCSC->localPosition();
	if ((fabs(positionLocalCSC.x()-segLocalCSC.x())<CSCXCut) && 
	    (fabs(positionLocalCSC.y()-segLocalCSC.y())<CSCYCut)) 
	  pointerToCSCSegments.push_back(&(*segmentCSC)); 
        continue;
      }

      if(!(cscDetIdHit.ring()==myChamber.ring())) continue;
      if(!(cscDetIdHit.station()==myChamber.station())) continue;
      if(!(cscDetIdHit.endcap()==myChamber.endcap())) continue;
      if(!(cscDetIdHit.chamber()==myChamber.chamber())) continue;

      countMuonCSCHits++;

      LocalPoint positionLocalCSC = (*hitC)->localPosition();
	
      for (vector<CSCRecHit2D>::const_iterator hiti=CSCRechits2D.begin(); hiti!=CSCRechits2D.end(); hiti++) {

	if ( !hiti->isValid()) continue; 
	CSCDetId cscDetId((hiti->geographicalId()).rawId());
		
	if ((*hitC)->geographicalId().rawId()!=(hiti->geographicalId()).rawId()) continue;

	LocalPoint segLocalCSC = hiti->localPosition();
	//		cout<<"Layer Id (MuonHit) =  "<<cscDetIdHit<<" Muon Local Position (det frame) "<<positionLocalCSC <<endl;
	//		cout<<"Layer Id  (CSCHit) =  "<<cscDetId<<"  Hit Local Position (det frame) "<<segLocalCSC <<endl;
	if((fabs(positionLocalCSC.x()-segLocalCSC.x())<CSCXCut) && 
	   (fabs(positionLocalCSC.y()-segLocalCSC.y())<CSCYCut)) {
	  CSCcountAgreeingHits++;
	  //		  cout << "   Matched." << endl;
	}  
      }//End 2D rechit iteration
    }//End muon hit iteration
    
    matchRatioCSC = countMuonCSCHits == 0 ? 0 : CSCcountAgreeingHits/countMuonCSCHits;
		
    if ((matchRatioCSC>0.9) && ((countMuonCSCHits>1) || !cscTightMatch)) pointerToCSCSegments.push_back(&(*segmentCSC));

  } //End CSC Segment Iteration 

  return pointerToCSCSegments;

}

//define this as a plug-in
//DEFINE_FWK_MODULE(MuonSegmentMatcher);
