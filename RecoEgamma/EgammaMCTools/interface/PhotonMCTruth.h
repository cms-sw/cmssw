#ifndef PhotonMCTruth_h
#define PhotonMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>

/** \class PhotonMCTruth
 *       
 *  This class stores all the MC truth information needed about the
 *  conversion
 * 
 *  $Date: 2007/04/13 12:28:01 $
 *  $Revision: 1.1 $
 *  \author N. Marinelli  IASA-Athens
 *
 */




class PhotonMCTruth {
public:
  PhotonMCTruth() : isAConversion_(0),thePhoton_(0.,0.,0.), theR_(0.), theZ_(0.), 
                       theConvVertex_(0.,0.,0.) {};

  PhotonMCTruth(HepLorentzVector v) : thePhoton_(v) {};


  PhotonMCTruth(int isAConversion,HepLorentzVector v, float rconv, float zconv,
			 HepLorentzVector convVertex, HepLorentzVector pV, std::vector<const SimTrack *> tracks );


 HepLorentzVector primaryVertex() const {return thePrimaryVertex_;}
 // const vector<const TkSimTrack*> &convSimTracks()  { return theTracks_;}
 int isAConversion() const { return isAConversion_;}
 float radius() const {return theR_;}
 float z() const {return theZ_;}
 HepLorentzVector fourMomentum() const {return thePhoton_;}
 HepLorentzVector vertex() const {return theConvVertex_;}
 std::vector<const SimTrack *> simTracks() const {return tracks_;} 
 float ptTrack1() const;
 float ptTrack2() const;
 float invMass() const;
 

 /*
 vector<GlobalPoint> track1Hits() const {
   if(theTracks_.size() > 0) {
     return getHitPositions(theTracks_[0]);
   } else {
     vector<GlobalPoint> theHitPositions;
     return theHitPositions;
   }
 };
 vector<GlobalPoint> track2Hits() const {
   if(theTracks_.size() > 1) {
     return getHitPositions(theTracks_[1]);
   } else {
     vector<GlobalPoint> theHitPositions;
     return theHitPositions;
   }
 };
 */
 
 //Get nearest basic cluster to point where SimTrack impacts ECAL
 // const EgammaBasicCluster * bcFromTrack1() const{
 // return (theBasicClusters_.size() > 0) ? theBasicClusters_[0] : 0;};

 // const EgammaBasicCluster * bcFromTrack2() const{
 // return (theBasicClusters_.size() > 1) ? theBasicClusters_[1] : 0;};
 
 GlobalPoint ecalImpactPosTrack1() const {return ecalPosTrack1_;};
 GlobalPoint ecalImpactPosTrack2() const {return ecalPosTrack2_;};

 // vector<GlobalPoint> getHitPositions(const TkSimTrack*  const &theTkSimTrack) ;
 
 private:

 //  void basicClusters();
  //Find nearest basic cluster to point where each SimTrack impacts ECAL
 // const EgammaBasicCluster * bcFromTrack(const TkSimTrack * theTkSimTrack) const;
 // void printTrackHits (const TkSimTrack * theTkSimTrack) const;
  //  vector<GlobalPoint> getHitPositions(const TkSimTrack*  &theTkSimTrack) const ;

  //Find point where each SimTrack impacts ECAL
 //  void ecalImpactPositions ();


  int isAConversion_;
  HepLorentzVector thePhoton_;
  float theR_;
  float theZ_;
  HepLorentzVector theConvVertex_;
  HepLorentzVector thePrimaryVertex_;
  std::vector<const SimTrack *> tracks_;


  GlobalPoint ecalPosTrack1_;
  GlobalPoint ecalPosTrack2_;
};

#endif

