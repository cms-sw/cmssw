/**
 *  Class: ChamberSegmentUtility
 *
 *  Description:
 *  utility class for the dynamical truncation algorithm
 *
 *
 *  Authors :
 *  D. Pagano & G. Bruno - UCL Louvain
 *
 **/

#include "RecoMuon/GlobalTrackingTools/interface/ChamberSegmentUtility.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <iostream>
#include <map>
#include <vector>
#include <iostream>

using namespace edm;
using namespace std;
using namespace reco;

ChamberSegmentUtility::ChamberSegmentUtility()
{}

void ChamberSegmentUtility::initCSU(const edm::Handle<DTRecSegment4DCollection>& DTSegProd,
				    const edm::Handle<CSCSegmentCollection>& CSCSegProd) {
  
  all4DSegments = DTSegProd;
  CSCSegments   = CSCSegProd;

  unsigned int index = 0;
  for (  CSCSegmentCollection::id_iterator chamberId = CSCSegments->id_begin();
         chamberId != CSCSegments->id_end(); ++chamberId, ++index ) {
    
    CSCSegmentCollection::range  range = CSCSegments->get((*chamberId));
    
    for (CSCSegmentCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment) {
      if ((*chamberId).station() == 1) cscsegMap[1].push_back(*segment);
      if ((*chamberId).station() == 2) cscsegMap[2].push_back(*segment);
      if ((*chamberId).station() == 3) cscsegMap[3].push_back(*segment);
      if ((*chamberId).station() == 4) cscsegMap[4].push_back(*segment);
    }
  }
  
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){
    
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));

    for (DTRecSegment4DCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment){
      if ((*chamberIdIt).station() == 1) dtsegMap[1].push_back(*segment);
      if ((*chamberIdIt).station() == 2) dtsegMap[2].push_back(*segment);
      if ((*chamberIdIt).station() == 3) dtsegMap[3].push_back(*segment);
      if ((*chamberIdIt).station() == 4) dtsegMap[4].push_back(*segment);
    }
  }
}



vector<CSCSegment> ChamberSegmentUtility::getCSCSegmentsInChamber(CSCDetId sel)
{
  
  // loop on segments 4D   
  unsigned int index = 0;
  for (  CSCSegmentCollection::id_iterator chamberId = CSCSegments->id_begin();
	 chamberId != CSCSegments->id_end(); ++chamberId, ++index ) {
    
    if ((*chamberId).chamber() != sel.chamber()) continue;
    
    // Get the range for the corresponding ChamberId                                                                                                                     
    CSCSegmentCollection::range  range = CSCSegments->get((*chamberId));
    
    // Loop over the rechits of this DetUnit                                                                                                                            
    for (CSCSegmentCollection::const_iterator segment = range.first;
	 segment!=range.second; ++segment) {
      cscseg.push_back(*segment);
    }
  }
  return cscseg;
}



vector<DTRecSegment4D> ChamberSegmentUtility::getDTSegmentsInChamber(DTChamberId sel)
{

  // loop on segments 4D                                                                                                                                                      
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){
    
    if (*chamberIdIt != sel) continue;
    
    // Get the range for the corresponding ChamberId              
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));
    
    // Loop over the rechits of this DetUnit         
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment){
      dtseg.push_back(*segment);
    }
  }
  return dtseg;
}



vector<CSCRecHit2D> ChamberSegmentUtility::getCSCRHmap(const CSCSegment& selected)
{

  vector<CSCRecHit2D> allchRH;
  
  // loop on segments 4D                                                       
  unsigned int index = 0;
  for (  CSCSegmentCollection::id_iterator chamberId = CSCSegments->id_begin();
         chamberId != CSCSegments->id_end(); ++chamberId, ++index ) {
    
    // Get the range for the corresponding ChamberId                    
    CSCSegmentCollection::range  range = CSCSegments->get((*chamberId));
    
    // Loop over the rechits of this DetUnit                        
    for (CSCSegmentCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment) {
      
      if((*segment).parameters() == selected.parameters()) {
	allchRH = (*segment).specificRecHits();
      }
    }  
  }
  return allchRH;
}

  
vector<DTRecHit1D> ChamberSegmentUtility::getDTRHmap(const DTRecSegment4D& selected)
{
  
  vector<DTRecHit1D> allchRH;
  phiSegRH.clear();
  zSegRH.clear();
  
  // loop on segments 4D                                                                                                                                                      
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){
    
    // Get the range for the corresponding ChamberId              
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));
    
    // Loop over the rechits of this DetUnit         
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
         segment!=range.second; ++segment){
      
      if((*segment).parameters() == selected.parameters()) {
	if((*segment).hasPhi()){
	  const DTChamberRecSegment2D* phiSeg = (*segment).phiSegment();
	  phiSegRH = phiSeg->specificRecHits();
	}
	if((*segment).hasZed()){
	  const DTSLRecSegment2D* zSeg = (*segment).zSegment();
	  zSegRH = zSeg->specificRecHits();
	}

	// RecHits will be ordered later
	for (vector<DTRecHit1D>::const_iterator itphi = phiSegRH.begin(); itphi != phiSegRH.end(); itphi++) allchRH.push_back(*itphi);
	for (vector<DTRecHit1D>::iterator itz = zSegRH.begin(); itz < zSegRH.end(); itz++) allchRH.push_back(*itz);
	
      }
    }
  }
  return allchRH;
}



