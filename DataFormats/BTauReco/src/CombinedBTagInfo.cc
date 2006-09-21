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

using namespace std;

void reco::CombinedBTagInfo::VertexData::print() const
{
  cout << "****** print VertexData from extended bTag information (combined bTag) " << endl;
  cout << "chi2                         " << chi2                         << endl;
  cout << "ndof                         " << ndof                         << endl;
  cout << "nTracks                      " << nTracks                      << endl; 
  cout << "mass                         " << mass                         << endl;   
  cout << "isV0                         " << isV0                         << endl;     
  cout << "fracPV                       " << fracPV                       << endl;    
  cout << "flightDistance2D             " << flightDistance2D             << endl;
  cout << "flightDistance2DError        " << flightDistance2DError        << endl;
  cout << "flightDistance2DSignificance " << flightDistance2DSignificance << endl;
  cout << "flightDistance3D             " << flightDistance3D             << endl;
  cout << "flightDistance3DError        " << flightDistance3DError        << endl;
  cout << "flightDistance3DSignificance " << flightDistance3DSignificance << endl;    
}

void reco::CombinedBTagInfo::VertexData::init()
{
  chi2                         = -999;
  ndof                         = -999;
  nTracks                      = -999;
  mass                         = -999;
  isV0                         = -999;
  fracPV                       = -999;
  flightDistance2D             = -999;
  flightDistance2DError        = -999;
  flightDistance2DSignificance = -999;
  flightDistance3D             = -999;
  flightDistance3DError        = -999;
  flightDistance3DSignificance = -999;
}

reco::CombinedBTagInfo::TrackData::TrackData() 
{
  init();
}

reco::CombinedBTagInfo::TrackData::TrackData(
           const reco::TrackRef & mref, bool musedInSVX, double mpt, double mrapidity, 
           double meta, double md0, double md0Sign, double md0Error, double mjetDistance,
           int mnHitsTotal, int mnHitsPixel, bool mfirstHitPixel, double mchi2,
           double mip2D, double mip2Derror, double mip2DSignificance, double mip3D,
           double mip3DError, double mip3DSignificance, bool maboveCharmMass ) :
  trackRef(mref),usedInSVX(musedInSVX),pt(mpt),rapidity(mrapidity),eta(meta),d0(md0),d0Sign(md0Sign),
  d0Error(md0Error),jetDistance(mjetDistance),nHitsTotal(mnHitsTotal),nHitsPixel(mnHitsPixel),
  chi2(mchi2),ip2D(mip2D),ip2DError(mip2Derror),ip2DSignificance(mip2DSignificance),ip3D(mip3D),
  ip3DError(mip3DError),ip3DSignificance(mip3DSignificance), aboveCharmMass(maboveCharmMass),
  isValid(true)
{}

reco::CombinedBTagInfo::VertexData::VertexData()
{
  init();
}

void reco::CombinedBTagInfo::TrackData::init()
{
  usedInSVX        = false;
  aboveCharmMass   = false;
  pt               = -999;
  rapidity         = -999;
  eta              = -999;
  d0               = -999;
  d0Sign           = -999;
  d0Error          = -999;
  nHitsTotal       = -999;
  nHitsPixel       = -999;
  firstHitPixel    = false;
  chi2             = -999;
  ip2D             = -999;
  ip2DError        = -999;
  ip2DSignificance = -999;
  ip3D             = -999;
  ip3DError        = -999;
  ip3DSignificance = -999;
  isValid=false;
}

void reco::CombinedBTagInfo::TrackData::print() const
{
  cout << "*** printing trackData for combined b-tag info " << endl;
  cout << "    usedInSVX        " << usedInSVX        << endl;
  cout << "    aboveCharmMass   " << aboveCharmMass   << endl;
  cout << "    pt               " << pt               << endl;
  cout << "    rapidity         " << rapidity         << endl;
  cout << "    eta              " << eta              << endl;
  cout << "    d0               " << d0               << endl;
  cout << "    d0Sign           " << d0Sign           << endl;
  cout << "    d0Error          " << d0Error          << endl;
  cout << "    jetDistance      " << jetDistance      << endl;
  cout << "    nHitsTotal       " << nHitsTotal       << endl;
  cout << "    nHitsPixel       " << nHitsPixel       << endl;
  cout << "    firstHitPixel    " << firstHitPixel    << endl;
  cout << "    chi2             " << chi2             << endl;
  cout << "    ip2D             " << ip2D             << endl;
  cout << "    ip2DError        " << ip2DError        << endl;
  cout << "    ip2DSignificance " << ip2DSignificance << endl;
  cout << "    ip3D             " << ip3D             << endl;
  cout << "    ip3DError        " << ip3DError        << endl;
  cout << "    ip3DSignificance " << ip3DSignificance << endl;
}

reco::CombinedBTagInfo::CombinedBTagInfo() {
  // reset everything
  reset();
}

reco::CombinedBTagInfo::~CombinedBTagInfo() { }

//
// map related
//

bool reco::CombinedBTagInfo::existTrackData( const reco::TrackRef & trackRef)
{
  return ( trackDataMap_.find(trackRef) != trackDataMap_.end() );
  /*
  bool returnValue = false;
  TrackDataAssociation::const_iterator iter = trackDataMap_.find(trackRef);
  if (iter != trackDataMap_.end()) {
    returnValue = true;
  }
  return returnValue;*/
} // bool exitTrackData

void reco::CombinedBTagInfo::flushTrackData() {
  //  trackDataMap_.clear();
} // void flushTrackData

void reco::CombinedBTagInfo::storeTrackData( reco::TrackRef trackRef,
                                             const TrackData& trackData)
{
  //  cout << "*** trackData to store " << endl;
  //  trackData.print();
  trackDataMap_.insert(trackRef, trackData);
//   TrackDataAssociation::const_iterator iter;
//   iter = trackDataMap_.find(trackRef);
//   if (iter != trackDataMap_.end())
//     (iter->val).print();
} //void storeTrackData

int reco::CombinedBTagInfo::sizeTrackData()
{
  return trackDataMap_.size();
} // int sizeTrackData

const reco::CombinedBTagInfo::TrackData * reco::CombinedBTagInfo::getTrackData(reco::TrackRef trackRef)
{
 TrackDataAssociation::const_iterator iter = trackDataMap_.find(trackRef);
 if (iter != trackDataMap_.end()) {
   return &(iter->val);
 } else {
   return 0;
 } //if iter != end
} // TrackData* getTrackData

void reco::CombinedBTagInfo::printTrackData() {
  for ( TrackDataAssociation::const_iterator mapIter = trackDataMap_.begin(); 
        mapIter != trackDataMap_.end(); mapIter++)
  {
    const TrackData& trackData = mapIter->val;
    trackData.print();
  } // for mapIter
} // void printTrackData

bool reco::CombinedBTagInfo::existVertexData(vector<reco::Vertex>::const_iterator vertexRef)
{
  return ( vertexDataMap_.find(vertexRef) != vertexDataMap_.end() );
  /*
  bool returnValue = false;

  // try to find element
  map <vector<reco::Vertex>::const_iterator, VertexData>::const_iterator iter = 
     vertexDataMap_.find(vertexRef);
  if (iter != vertexDataMap_.end())
    returnValue = true;
  return returnValue;*/
} // bool exitVertexData

void reco::CombinedBTagInfo::flushVertexData() {
  //  vertexDataMap_.clear();
} // void flushVertexData
// -------------------------------------------------------------------------------

void reco::CombinedBTagInfo::storeVertexData( vector<reco::Vertex>::const_iterator vertexRef,
                                              const VertexData& vertexData) {
  vertexDataMap_[vertexRef] = vertexData;
} //void storeVertexData

int reco::CombinedBTagInfo::sizeVertexData() const
{
  return vertexDataMap_.size();
} // int sizeVertexData

reco::CombinedBTagInfo::VertexData * 
    reco::CombinedBTagInfo::getVertexData(vector<reco::Vertex>::const_iterator vertexRef) const
{
  // try to find element
  map <vector<reco::Vertex>::const_iterator, VertexData>::const_iterator iter = 
    vertexDataMap_.find(vertexRef);

  if (iter != vertexDataMap_.end()) {
    // found element
    return &vertexDataMap_[vertexRef];
  } else {
    // element not found
    return 0;
  } //if iter != end
} // VertexData* getVertexData

void reco::CombinedBTagInfo::reset()
{
  //
  // reset all information
  //
  GlobalVector resetVector (-999.0,-999.0,-999.0);

  // flush maps and vectors
  flushTrackData();
  flushVertexData();
  secondaryVertices_.clear();
  tracksAboveCharm_.clear();
  tracksAtSecondaryVertex_.clear();
  
  // reset variables
  vertexType_                       = reco::CombinedBTagEnums::NotDefined;
  jetPt_                            = -999;
  jetEta_                           = -999;
  pB_                               = resetVector;
  pAll_                             = resetVector;   
  bPLong_                           = -999;
  bPt_                              = -999;
  vertexMass_                       = -999;
  vertexMultiplicity_               = -999;
  eSVXOverE_                        = -999;
  meanTrackY_                       = -999;
  energyBTracks_                    = -999;
  energyAllTracks_                  = -999;
  angleGeomKinJet_                  = -999;
  angleGeomKinVertex_               = -999;  
  flightDistance2DMin_              = -999;
  flightDistanceSignificance2DMin_  = -999;
  flightDistance3DMin_              = -999;
  flightDistanceSignificance3DMin_  = -999;
  flightDistance2DMax_              = -999;
  flightDistanceSignificance2DMax_  = -999;
  flightDistance3DMax_              = -999;
  flightDistanceSignificance3DMax_  = -999;
  flightDistance2DMean_             = -999;
  flightDistanceSignificance2DMean_ = -999;
  flightDistance3DMean_             = -999;
  flightDistanceSignificance3DMean_ = -999;
  first2DSignedIPSigniAboveCut_     = -999;
} //reset

double reco::CombinedBTagInfo::jetPt() const
{
  return jetPt_;
}

double reco::CombinedBTagInfo::jetEta() const
{
  return jetEta_;
}

reco::Vertex reco::CombinedBTagInfo::primaryVertex() const
{
  return primaryVertex_;
}

vector<reco::Vertex> reco::CombinedBTagInfo::secVertices() const
{
  return secondaryVertices_;
}

vector<reco::TrackRef> reco::CombinedBTagInfo::tracksAboveCharm() const
{
  return tracksAboveCharm_;
}

vector<reco::TrackRef> reco::CombinedBTagInfo::tracksAtSecondaryVertex() const
{
  return tracksAtSecondaryVertex_;
}

int reco::CombinedBTagInfo::nSecVertices() const
{
  return secondaryVertices_.size();
}

reco::CombinedBTagEnums::VertexType reco::CombinedBTagInfo::vertexType() const
{
  return vertexType_;
}

double reco::CombinedBTagInfo::vertexMass() const
{return vertexMass_;}

int reco::CombinedBTagInfo::vertexMultiplicity() const
{return vertexMultiplicity_;}

double reco::CombinedBTagInfo::eSVXOverE() const {return eSVXOverE_;}

GlobalVector reco::CombinedBTagInfo::pAll() const {return pAll_;}

GlobalVector reco::CombinedBTagInfo::pB() const {return pB_;}

double reco::CombinedBTagInfo::pBLong() const {return bPLong_;}

double reco::CombinedBTagInfo::pBPt() const {return bPt_;}

double reco::CombinedBTagInfo::meanTrackRapidity() const {return meanTrackY_;}

double reco::CombinedBTagInfo::angleGeomKinJet() const {return angleGeomKinJet_;}

double reco::CombinedBTagInfo::angleGeomKinVertex() const {return angleGeomKinVertex_;} 

double reco::CombinedBTagInfo::flightDistance2DMin() const {return flightDistance2DMin_; }

double reco::CombinedBTagInfo::flightDistanceSignificance2DMin() const
  {return flightDistanceSignificance2DMin_;}

double reco::CombinedBTagInfo::flightDistance3DMin() const 
  {return flightDistance3DMin_; }

double reco::CombinedBTagInfo::flightDistanceSignificance3DMin() const 
  {return flightDistanceSignificance3DMin_; }

double reco::CombinedBTagInfo::flightDistance2DMax() const
  {return flightDistance2DMax_; }

double reco::CombinedBTagInfo::flightDistanceSignificance2DMax() const
  {return flightDistanceSignificance2DMax_; }

double reco::CombinedBTagInfo::flightDistance3DMax() const
  {return flightDistance3DMax_; } 

double reco::CombinedBTagInfo::flightDistanceSignificance3DMax() const
  {return flightDistanceSignificance3DMax_  ;}

double reco::CombinedBTagInfo::flightDistance2DMean() const
  {return flightDistance2DMean_             ;}

double reco::CombinedBTagInfo::flightDistance3DMean() const
  {return flightDistance3DMean_             ;}

double reco::CombinedBTagInfo::flightDistanceSignificance3DMean() const
  {return flightDistanceSignificance3DMean_ ;}

double reco::CombinedBTagInfo::first2DSignedIPSigniAboveCut() const
  {return first2DSignedIPSigniAboveCut_;}

void reco::CombinedBTagInfo::setJetPt (double pt) {jetPt_ = pt;}

void reco::CombinedBTagInfo::setJetEta(double eta) {jetEta_ = eta;} 

void reco::CombinedBTagInfo::setPrimaryVertex( const reco::Vertex & pv) {primaryVertex_ = pv;}

void reco::CombinedBTagInfo::addSecondaryVertex( const reco::Vertex & sv) {secondaryVertices_.push_back(sv);}

void reco::CombinedBTagInfo::addTrackAtSecondaryVertex(reco::TrackRef trackRef)
  {tracksAtSecondaryVertex_.push_back(trackRef);}

void reco::CombinedBTagInfo::setVertexType( reco::CombinedBTagEnums::VertexType type) {vertexType_ = type;}
void reco::CombinedBTagInfo::setVertexMass( double mass) {vertexMass_ = mass;}
void reco::CombinedBTagInfo::setVertexMultiplicity( int mult ) {vertexMultiplicity_ = mult;}
void reco::CombinedBTagInfo::setESVXOverE( double e) {eSVXOverE_ = e; }
void reco::CombinedBTagInfo::setEnergyBTracks(double energy) {energyBTracks_ = energy;}
void reco::CombinedBTagInfo::setEnergyAllTracks(double energy) {energyAllTracks_ = energy;}
void reco::CombinedBTagInfo::setPAll( const GlobalVector & p) { pAll_ = p;}
void reco::CombinedBTagInfo::setPB( const GlobalVector & p ) { pB_ = p;}
void reco::CombinedBTagInfo::setBPLong(double pLong) { bPLong_ = pLong;} 
void reco::CombinedBTagInfo::setBPt(double pt) {bPt_ = pt;}
void reco::CombinedBTagInfo::setMeanTrackRapidity(double meanY) {meanTrackY_ = meanY;}
void reco::CombinedBTagInfo::setAngleGeomKinJet(double angle) {angleGeomKinJet_ = angle;}
void reco::CombinedBTagInfo::setAngleGeomKinVertex(double angle) {angleGeomKinVertex_ = angle;}
void reco::CombinedBTagInfo::addTrackAboveCharm(reco::TrackRef trackRef) {tracksAboveCharm_.push_back(trackRef);}
void reco::CombinedBTagInfo::setFlightDistance2DMin(double value) {flightDistance2DMin_ = value;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance2DMin (double value) {flightDistanceSignificance2DMin_ = value;}
void reco::CombinedBTagInfo::setFlightDistance3DMin(double value) {flightDistance3DMin_ = value;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance3DMin(double value) {flightDistanceSignificance3DMin_ = value;}
void reco::CombinedBTagInfo::setFlightDistance2DMax(double value) {flightDistance2DMax_ = value;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance2DMax(double value) {flightDistanceSignificance2DMax_ = value;}
void reco::CombinedBTagInfo::setFlightDistance3DMax (double value) {flightDistance3DMax_ = value;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance3DMax(double value) {flightDistanceSignificance3DMax_ = value;}
void reco::CombinedBTagInfo::setFlightDistance2DMean(double value) {flightDistance2DMean_ = value;} 
void reco::CombinedBTagInfo::setFlightDistanceSignificance2DMean(double value) {flightDistanceSignificance2DMean_ = value;} 
void reco::CombinedBTagInfo::setFlightDistance3DMean(double value) {flightDistance3DMean_ = value;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance3DMean (double value) {flightDistanceSignificance3DMean_ = value;}
void reco::CombinedBTagInfo::setFirst2DSignedIPSigniAboveCut(double ipSignificance) {first2DSignedIPSigniAboveCut_ = ipSignificance;} 

std::string reco::CombinedBTagInfo::getVertexTypeName() const
{
  return reco::CombinedBTagEnums::typeOfVertex ( vertexType_ );
}
