// -*- C++ -*-
//
// Package:    CombinedBTagInfo
// Class:      CombinedBTagInfo
// 
/**\class CombinedBTagInfo CombinedBTagInfo.cc DataFormats/BTauReco/src/CombinedBTagInfo.cc

 Description: Extended information for combined b-jet tagging

 Implementation:
     <Notes on implementation>
*/


// this class header

#include "DataFormats/BTauReco/interface/CombinedBTagInfo.h"

//------------------------------------------------------------------------------
// Constructors
//------------------------------------------------------------------------------
reco::CombinedBTagInfo::CombinedBTagInfo() {

  // reset variables
  reco::CombinedBTagInfo::vertexType_                       = reco::CombinedBTagInfo::NotDefined;
						            
  reco::CombinedBTagInfo::jetPt_                            = -999;
  reco::CombinedBTagInfo::jetEta_                           = -999;
  						            
  reco::CombinedBTagInfo::pAll_.set(-999,-999,-999);        
  reco::CombinedBTagInfo::pB_.set(-999,-999,-999);          
  reco::CombinedBTagInfo::bPLong_                           = -999;
  reco::CombinedBTagInfo::bPt_                              = -999;
  reco::CombinedBTagInfo::vertexMass_                       = -999;
  reco::CombinedBTagInfo::vertexMultiplicity_               = -999;
  reco::CombinedBTagInfo::eSVXOverE_                        = -999;
  reco::CombinedBTagInfo::meanTrackY_                       = -999;
  						            
  reco::CombinedBTagInfo::angleGeomKinJet_                  = -999;
  reco::CombinedBTagInfo::angleGeomKinVertex_               = -999;  

  reco::CombinedBTagInfo::flightDistance2DMin_              = -999;
  reco::CombinedBTagInfo::flightDistanceSignificance2DMin_  = -999;
  reco::CombinedBTagInfo::flightDistance3DMin_              = -999;
  reco::CombinedBTagInfo::flightDistanceSignificance3DMin_  = -999;

  reco::CombinedBTagInfo::flightDistance2DMax_              = -999;
  reco::CombinedBTagInfo::flightDistanceSignificance2DMax_  = -999;
  reco::CombinedBTagInfo::flightDistance3DMax_              = -999;
  reco::CombinedBTagInfo::flightDistanceSignificance3DMax_  = -999;

  reco::CombinedBTagInfo::flightDistance2DMean_             = -999;
  reco::CombinedBTagInfo::flightDistanceSignificance2DMean_ = -999;
  reco::CombinedBTagInfo::flightDistance3DMean_             = -999;
  reco::CombinedBTagInfo::flightDistanceSignificance3DMean_ = -999;

 
  // reset also maps?

} // constructor


//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
reco::CombinedBTagInfo::~CombinedBTagInfo() {

}


// -------------------------------------------------------------------------------
// -------------------------------------------------------------------------------

//
// map related
//

bool reco::CombinedBTagInfo::existTrackData(TRACKREF trackRef) {

  bool returnValue = false;

  // create an iterator
  std::map <TRACKREF, reco::CombinedBTagInfo::TrackData>::const_iterator iter;

  // try to find element
  iter = reco::CombinedBTagInfo::trackDataMap_.find(trackRef);
  if (iter != reco::CombinedBTagInfo::trackDataMap_.end())
    returnValue = true;

  return returnValue;

} // bool exitTrackData
// -------------------------------------------------------------------------------

void reco::CombinedBTagInfo::flushTrackData() {
  reco::CombinedBTagInfo::trackDataMap_.clear();
  
} // void flushTrackData
// -------------------------------------------------------------------------------

void reco::CombinedBTagInfo::storeTrackData(TRACKREF trackRef,
					    const reco::CombinedBTagInfo::TrackData& trackData) {
  
  reco::CombinedBTagInfo::trackDataMap_[trackRef] = trackData;

} //void storeTrackData
// -------------------------------------------------------------------------------

int reco::CombinedBTagInfo::sizeTrackData() {
  
  int size = reco::CombinedBTagInfo::trackDataMap_.size();

  return size;

} // int sizeTrackData
// -------------------------------------------------------------------------------

const reco::CombinedBTagInfo::TrackData* reco::CombinedBTagInfo::getTrackData(TRACKREF trackRef) {

  // create an iterator
  std::map <TRACKREF, reco::CombinedBTagInfo::TrackData>::const_iterator iter;

  // try to find element
  iter = reco::CombinedBTagInfo::trackDataMap_.find(trackRef);

  if (iter != reco::CombinedBTagInfo::trackDataMap_.end()) {

    // found element
    return &reco::CombinedBTagInfo::trackDataMap_[trackRef];

  } else {
    
    // element not found
    return 0;

  } //if iter != end

} // TrackData* getTrackData
// -------------------------------------------------------------------------------
