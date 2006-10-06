#include "DataFormats/BTauReco/interface/CombinedBTagInfo.h"
#include <limits>

using namespace std;

namespace {
  typedef std::numeric_limits<double> num;
  typedef std::numeric_limits<int> numi;
}

reco::CombinedBTagInfo::CombinedBTagInfo() {
  reset();
}

reco::CombinedBTagInfo::~CombinedBTagInfo() { }

bool reco::CombinedBTagInfo::existTrackData( const reco::TrackRef & trackRef)
{
  return ( trackDataMap_.find(trackRef) != trackDataMap_.end() );
}

void reco::CombinedBTagInfo::flushTrackData() {
  //  trackDataMap_.clear();
}

void reco::CombinedBTagInfo::storeTrackData( const reco::TrackRef & trackRef,
                                             const reco::CombinedBTagTrack & trackData)
{
  // trackDataMap_[trackRef]=trackData;
  trackDataMap_.erase(trackRef );
  trackDataMap_.insert(trackRef, trackData);
}

int reco::CombinedBTagInfo::sizeTrackData() const
{
  return trackDataMap_.size();
}

const reco::CombinedBTagTrack * reco::CombinedBTagInfo::getTrackData(
    const reco::TrackRef & trackRef ) const
{
  TrackDataAssociation::const_iterator iter = trackDataMap_.find(trackRef);
  if (iter != trackDataMap_.end()) {
    return &(iter->val);
  } else {
    return 0;
  }
}

void reco::CombinedBTagInfo::printTrackData() const
{
  for ( TrackDataAssociation::const_iterator mapIter = trackDataMap_.begin();
        mapIter != trackDataMap_.end(); mapIter++)
  {
    const reco::CombinedBTagTrack & trackData = mapIter->val;
    trackData.print();
  }
}

bool reco::CombinedBTagInfo::existVertexData(vector<reco::Vertex>::const_iterator vertexRef)
{
  return ( vertexDataMap_.find(vertexRef) != vertexDataMap_.end() );
}

void reco::CombinedBTagInfo::flushVertexData() {
  //  vertexDataMap_.clear();
}

void reco::CombinedBTagInfo::storeVertexData( vector<reco::Vertex>::const_iterator vertexRef,
                                              const reco::CombinedBTagVertex & vertexData) {
  vertexDataMap_[vertexRef] = vertexData;
}

int reco::CombinedBTagInfo::sizeVertexData() const
{
  return vertexDataMap_.size();
}

reco::CombinedBTagVertex *
    reco::CombinedBTagInfo::getVertexData(vector<reco::Vertex>::const_iterator vertexRef) const
{
  // try to find element
  map <vector<reco::Vertex>::const_iterator, reco::CombinedBTagVertex>::const_iterator iter =
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
  // reset all information
  GlobalVector resetVector (num::quiet_NaN(), num::quiet_NaN(), num::quiet_NaN());

  // flush maps and vectors
  flushTrackData();
  flushVertexData();
  secondaryVertices_.clear();
  tracksAboveCharm_.clear();
  tracksAtSecondaryVertex_.clear();

  // reset variables
  vertexType_                       = reco::CombinedBTagEnums::NotDefined;
  jetPt_                            = num::quiet_NaN();
  jetEta_                           = num::quiet_NaN();
  pB_                               = resetVector;
  pAll_                             = resetVector;
  bPLong_                           = num::quiet_NaN();
  bPt_                              = num::quiet_NaN();
  vertexMass_                       = num::quiet_NaN();
  vertexMultiplicity_               = numi::quiet_NaN();
  eSVXOverE_                        = num::quiet_NaN();
  meanTrackY_                       = num::quiet_NaN();
  energyBTracks_                    = num::quiet_NaN();
  energyAllTracks_                  = num::quiet_NaN();
  angleGeomKinJet_                  = num::quiet_NaN();
  angleGeomKinVertex_               = num::quiet_NaN();
  flightDistance2D_                 = MinMeanMax();
  flightDistanceSignificance2D_     = MinMeanMax();
  flightDistance3D_                 = MinMeanMax();
  flightDistanceSignificance3D_     = MinMeanMax();
  first2DSignedIPSigniAboveCut_     = num::quiet_NaN();
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

reco::MinMeanMax reco::CombinedBTagInfo::flightDistance2D() const {return flightDistance2D_; }
reco::MinMeanMax reco::CombinedBTagInfo::flightDistanceSignificance2D() const 
    {return flightDistanceSignificance2D_; }
reco::MinMeanMax reco::CombinedBTagInfo::flightDistance3D() const {return flightDistance3D_; }
reco::MinMeanMax reco::CombinedBTagInfo::flightDistanceSignificance3D() const 
    {return flightDistanceSignificance3D_; }

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
void reco::CombinedBTagInfo::setFlightDistance2D( const MinMeanMax & v ) {flightDistance2D_ = v;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance2D ( const MinMeanMax & v )
    {flightDistanceSignificance2D_ = v;}
void reco::CombinedBTagInfo::setFlightDistance3D( const MinMeanMax & v ) {flightDistance3D_ = v;}
void reco::CombinedBTagInfo::setFlightDistanceSignificance3D ( const MinMeanMax & v )
    {flightDistanceSignificance3D_ = v;}

void reco::CombinedBTagInfo::setFirst2DSignedIPSigniAboveCut(double ipSignificance) {first2DSignedIPSigniAboveCut_ = ipSignificance;}

std::string reco::CombinedBTagInfo::getVertexTypeName() const
{
  return reco::CombinedBTagEnums::typeOfVertex ( vertexType_ );
}
