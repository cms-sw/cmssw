// -*- C++ -*-
//
// Package:    TrackerOnlyConversionProducer
// Class:      TrackerOnlyConversionProducer
//
/**\class TrackerOnlyConversionProducer TrackerOnlyConversionProducer.cc MyCode/TrackerOnlyConversionProducer/src/TrackerOnlyConversionProducer.cc

 Description: Produces converted photon objects using default track collections

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Hongliang Liu
//         Created:  Thu Mar 13 17:40:48 CDT 2008
// $Id: TrackerOnlyConversionProducer.cc,v 1.4 2008/09/29 22:17:03 hlliu Exp $
//
//


// system include files
#include <memory>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//ECAL clusters
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//Tracker tracks
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
//#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
//#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
//#include "DataFormats/V0Candidate/interface/V0Candidate.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"

using namespace edm;
using namespace reco;
using namespace std;

//
// class decleration
//

inline const GeomDet * recHitDet( const TrackingRecHit & hit, const TrackingGeometry * geom ) {
    return geom->idToDet( hit.geographicalId() );
}

inline const BoundPlane & recHitSurface( const TrackingRecHit & hit, const TrackingGeometry * geom ) {
    return recHitDet( hit, geom )->surface();
}

inline LocalVector toLocal( const Track::Vector & v, const Surface & s ) {
    return s.toLocal( GlobalVector( v.x(), v.y(), v.z() ) );
}

inline double map_phi2(double phi) {
    // map phi to [-pi,pi]
    double result = phi;
    if ( result < 1.0*Geom::pi() ) result = result + Geom::twoPi();
    if ( result >= Geom::pi())  result = result - Geom::twoPi();
    return result;
}

class TrackerOnlyConversionProducer : public edm::EDProducer {
    public:
      explicit TrackerOnlyConversionProducer(const edm::ParameterSet&);
      ~TrackerOnlyConversionProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      virtual void endRun(const edm::Run&, const edm::EventSetup&);

      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef std::vector<Point> PointCollection;

      std::vector<edm::InputTag>  src_; 

      edm::InputTag bcBarrelCollection_;
      edm::InputTag bcEndcapCollection_;
      std::string ConvertedPhotonCollection_;

      double halfWayEta_, halfWayPhi_;//halfway open angle to search in basic clusters

      double energyBC_;//1.5GeV
      double dEtaTkBC_, dPhiTkBC_;//0.06 0.6

      double maxChi2Left_, maxChi2Right_;//30 30
      double maxHitsLeft_, maxHitsRight_;//5 2

      double deltaCotTheta_, deltaPhi_;//0.02 0.2

      bool allowSingleLeg_;//if single track conversion ?

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
TrackerOnlyConversionProducer::TrackerOnlyConversionProducer(const edm::ParameterSet& iConfig)
{
    src_ = iConfig.getParameter<std::vector<edm::InputTag> >("src");


    bcBarrelCollection_     = iConfig.getParameter<edm::InputTag>("bcBarrelCollection");
    bcEndcapCollection_     = iConfig.getParameter<edm::InputTag>("bcEndcapCollection");

    halfWayEta_ = iConfig.getParameter<double>("HalfwayEta");//open angle to search track matches with BC

    //Track-cluster matching eta and phi cuts
    dEtaTkBC_ = iConfig.getParameter<double>("dEtaTrackBC");//TODO research on cut endcap/barrel
    dPhiTkBC_ = iConfig.getParameter<double>("dPhiTrackBC");

    energyBC_ = iConfig.getParameter<double>("EnergyBC");//BC energy cut

    //Track cuts on left right track: at least one leg reaches ECAL
    //Left track: must exist, must reach Ecal and match BC, so loose cut on Chi2 and tight on hits
    //Right track: not necessary to exist (if allowSingleLeg_), not necessary to reach ECAL or match BC, so tight cut on Chi2 and loose on hits
    maxChi2Left_ =  iConfig.getParameter<double>("MaxChi2Left");
    maxChi2Right_ =  iConfig.getParameter<double>("MaxChi2Right");
    maxHitsLeft_ = iConfig.getParameter<int>("MaxHitsLeft");
    maxHitsRight_ = iConfig.getParameter<int>("MaxHitsRight");

    //Track Open angle cut on delta cot(theta) and delta phi
    deltaCotTheta_ = iConfig.getParameter<double>("DeltaCotTheta");
    deltaPhi_ = iConfig.getParameter<double>("DeltaPhi");

    // if allow single track collection, by default False
    allowSingleLeg_ = iConfig.getParameter<bool>("AllowSingleLeg");

    //output
    ConvertedPhotonCollection_     = iConfig.getParameter<std::string>("convertedPhotonCollection");

    produces< reco::ConversionCollection >(ConvertedPhotonCollection_);

}


TrackerOnlyConversionProducer::~TrackerOnlyConversionProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackerOnlyConversionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   reco::ConversionCollection outputConvPhotonCollection;
   std::auto_ptr<reco::ConversionCollection> outputConvPhotonCollection_p(new reco::ConversionCollection);

   //Read multiple track input collections
   std::vector<const reco::TrackCollection*>  trackCollections;
   std::vector<edm::Handle<reco::TrackCollection> > trackCollectionHandles;
   for (unsigned ii = 0; ii<src_.size(); ++ii){
       edm::Handle<reco::TrackCollection> temp_handle;
       if(iEvent.getByLabel(src_[ii],temp_handle)){//starting from 170
	   trackCollectionHandles.push_back(temp_handle);
       } else {
	   edm::LogWarning("TrackerOnlyConversionProducer") << "Collection reco::TrackCollection with label " << src_[ii] << " cannot be found, using empty collection of same type";
       }
   }
   edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;//TODO error handling if no collection found
   iEvent.getByLabel( bcBarrelCollection_, bcBarrelHandle);

   edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;//TODO check cluster type if BasicCluster or PFCluster
   iEvent.getByLabel( bcEndcapCollection_, bcEndcapHandle);

   edm::ESHandle<TrackerGeometry> trackerGeomHandle;
   edm::ESHandle<MagneticField> magFieldHandle;

   iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeomHandle );
   iSetup.get<IdealMagneticFieldRecord>().get( magFieldHandle );

   const TrackerGeometry* trackerGeom = trackerGeomHandle.product();
   const MagneticField* magField = magFieldHandle.product();;

   //pair up tracks and record unpaired but matched tracks
   //
   //1. combining all track collections
   reco::TrackRefVector allTracks;
   int total_tracks = 0;
   for (unsigned int ii = 0; ii<trackCollectionHandles.size(); ++ii)
       total_tracks += trackCollectionHandles[ii]->size();
   allTracks.reserve(total_tracks);//for many tracks to save relocation time

   for (unsigned int ii = 0; ii<trackCollectionHandles.size(); ++ii){
       if (!trackCollectionHandles[ii]->empty()){
	   for (unsigned int jj = 0; jj<trackCollectionHandles[ii]->size(); ++jj){
	       edm::Ref<reco::TrackCollection>  ref(trackCollectionHandles[ii], jj);
	       if (ref.isNonnull()){
		   allTracks.push_back(ref);//TODO find a way to get vector directly from handle to avoid loop
	       }
	   }
       }
   }
   //2. select track by propagating
   //2.0 build Basic cluster geometry map to search in eta bounds for clusters
   std::multimap<double, reco::CaloClusterPtr> basicClusterPtrs;
   edm::Handle<edm::View<reco::CaloCluster> > bcHandle = bcBarrelHandle;
   for (unsigned jj = 0; jj < 2; ++jj ){
       for (unsigned ii = 0; ii < bcHandle->size(); ++ii ) {
	   if (bcHandle->ptrAt(ii)->energy()>energyBC_)
	       basicClusterPtrs.insert(std::make_pair(bcHandle->ptrAt(ii)->position().eta(), bcHandle->ptrAt(ii)));
       }
       bcHandle = bcEndcapHandle;
   }

   std::vector<math::XYZPoint> trackImpactPosition;
   trackImpactPosition.reserve(allTracks.size());//track impact position at ECAL
   std::vector<bool> trackValidECAL;//Does this track reach ECAL basic cluster (reach ECAL && match with BC)
   trackValidECAL.assign(allTracks.size(), false);

   std::vector<reco::CaloClusterPtr> trackMatchedBC;
   reco::CaloClusterPtr empty_bc;
   trackMatchedBC.assign(allTracks.size(), empty_bc);//TODO find a better way to avoid copy constructor

   std::multimap<double, int> trackInnerEta;//Track innermost state Eta map to TrackRef index, to be used in track pair sorting

   //2.1 propagate all tracks into ECAL, record its eta and phi
   //for (std::vector<edm::Ref<reco::TrackCollection> >::const_iterator ref = allTracks.begin(); ref != allTracks.end(); ++ref){
   for (reco::TrackRefVector::const_iterator ref = allTracks.begin(); ref != allTracks.end(); ++ref){
       const TrackRef& tk_ref = *ref;

       if (tk_ref->normalizedChi2() > maxChi2Left_ || tk_ref->found() < maxHitsLeft_ //track quality cut
	  ) continue;

       //map TrackRef to Eta
       trackInnerEta.insert(std::make_pair(tk_ref->innerMomentum().eta(), ref-allTracks.begin()));

       PropagatorWithMaterial propag( alongMomentum, 0.000511, magField );
       TrajectoryStateTransform transformer;
       ReferenceCountingPointer<Surface> ecalWall(
	       new  BoundCylinder( GlobalPoint(0.,0.,0.), TkRotation<float>(),
		   SimpleCylinderBounds( 129, 129, -320.5, 320.5 ) ) );
       const float epsilon = 0.001;
       Surface::RotationType rot; // unit rotation matrix
       const float barrelRadius = 129.f;
       const float barrelHalfLength = 270.9f;
       const float endcapRadius = 171.1f;
       const float endcapZ = 320.5f;
       ReferenceCountingPointer<BoundCylinder>  theBarrel_(new BoundCylinder( Surface::PositionType(0,0,0), rot,
		   SimpleCylinderBounds( barrelRadius-epsilon, barrelRadius+epsilon, -barrelHalfLength, barrelHalfLength)));
       ReferenceCountingPointer<BoundDisk>      theNegativeEtaEndcap_(
	       new BoundDisk( Surface::PositionType( 0, 0, -endcapZ), rot,
		   SimpleDiskBounds( 0, endcapRadius, -epsilon, epsilon)));
       ReferenceCountingPointer<BoundDisk>      thePositiveEtaEndcap_(
	       new BoundDisk( Surface::PositionType( 0, 0, endcapZ), rot,
		   SimpleDiskBounds( 0, endcapRadius, -epsilon, epsilon)));

       //const TrajectoryStateOnSurface myTSOS = transformer.innerStateOnSurface(*(*ref), *trackerGeom, magField);
       const TrajectoryStateOnSurface myTSOS = transformer.outerStateOnSurface(*tk_ref, *trackerGeom, magField);
       TrajectoryStateOnSurface  stateAtECAL;
       stateAtECAL = propag.propagate(myTSOS, *theBarrel_);
       if (!stateAtECAL.isValid() || ( stateAtECAL.isValid() && fabs(stateAtECAL.globalPosition().eta() ) >1.479 )  ) {
	   //endcap propagator
	   if (myTSOS.globalPosition().eta() > 0.) {
	       stateAtECAL = propag.propagate(myTSOS, *thePositiveEtaEndcap_);
	   } else {
	       stateAtECAL = propag.propagate(myTSOS, *theNegativeEtaEndcap_);
	   }
       }       
       math::XYZPoint ew;
       if (stateAtECAL.isValid()){
	   ew  = stateAtECAL.globalPosition();
       }
       trackImpactPosition.push_back(ew);//for invalid state, it will be count as invalid, so before read it out, check trackValidECAL[]
       if (!stateAtECAL.isValid()){ continue;}

       const double track_eta = ew.eta();
       const double track_phi = ew.phi();

       //2.2 check matching with BC
       reco::CaloClusterPtr closest_bc;
       double min_eta = 999., min_phi = 999.;
       for (std::multimap<double, reco::CaloClusterPtr>::iterator bc = basicClusterPtrs.lower_bound(track_eta - halfWayEta_); 
	       bc != basicClusterPtrs.upper_bound(track_eta + halfWayEta_); ++bc){//use eta map to select possible BC collection then loop in
	   const reco::CaloClusterPtr& ebc = bc->second;
	   const double delta_eta = track_eta-(ebc->position().eta());
	   const double delta_phi = map_phi2(track_phi-(ebc->position().phi()));
	   if (fabs(delta_eta)<dEtaTkBC_ && fabs(delta_phi)<dPhiTkBC_){
	       if (fabs(min_eta)>fabs(delta_eta) && fabs(min_phi)>fabs(delta_phi)){//take the closest to track BC
		   min_eta = delta_eta;
		   min_phi = delta_phi;
		   closest_bc = bc->second;
		   //TODO check if min_eta>delta_eta but min_phi<delta_phi
	       }
	   }
       }
       if (min_eta < 999.){//this track matched a BC
	   trackMatchedBC[ref-allTracks.begin()] = closest_bc;
	   trackValidECAL[ref-allTracks.begin()] = true;
       }
   }
   //3. pair up : to select one track from matched EBC, and select the other track to fit cot theta and phi open angle cut
   //TODO it is k-Closest pair of point problem
   reco::VertexCollection vertexs;
   std::vector<bool> isPaired;
   isPaired.assign(allTracks.size(), false);
   ///for( std::vector<edm::Ref<reco::TrackCollection> >::const_iterator ll = allTracks.begin(); ll != allTracks.end(); ++ ll ) {
   for( reco::TrackRefVector::const_iterator ll = allTracks.begin(); ll != allTracks.end(); ++ ll ) {
       //Level 1 loop, in all tracks matched with ECAL
       if (!trackValidECAL[ll-allTracks.begin()] //this Left leg should have valid BC
	       || (*ll)->d0()*(*ll)->charge()<0. //d0*charge for converted photons
	       || isPaired[ll-allTracks.begin()]) //this track should not be in another pair
	   continue;
       const double left_eta = (*ll)->innerMomentum().eta();
       bool found_right = false;//check if right leg found, if no but allowSingleLeg_, go build a conversion with left leg
       std::map<double, int> right_candidates;//store all right legs passed the cut (theta and ref pair)

       //select right leg candidates, which passed the cuts
       for (std::multimap<double, int>::iterator rr = trackInnerEta.lower_bound(left_eta - halfWayEta_);
	       rr != trackInnerEta.upper_bound(left_eta + halfWayEta_); ++rr){//select neighbor tracks by eta
	   //Level 2 loop
	   //TODO find the closest one rather than the first matching
	   const edm::Ref<reco::TrackCollection> & right = allTracks[rr->second];

	   if (right->normalizedChi2() > maxChi2Right_ || right->found() < maxHitsRight_ //track quality cut
		   || isPaired[rr->second] //this track should not be in another pair
		   //|| right == (*ll) //no double counting (dummy if require opposite charge)
		   || right->d0()*right->charge()<0. //d0*charge for converted photons
		   || ((*ll)->charge())*(right->charge()) > 0) //opposite charge
	       continue;

	   const double theta_l = (*ll)->innerMomentum().Theta();
	   const double theta_r = right->innerMomentum().Theta();
	   const double dCotTheta =  1./tan(theta_l) - 1./tan(theta_r) ;

	   if (fabs(dCotTheta) > deltaCotTheta_) continue;//Delta Cot(Theta) cut for pair

	   const double phi_l = (*ll)->innerMomentum().phi();
	   const double phi_r = right->innerMomentum().phi();
	   const double dPhi = phi_l - phi_r;

	   if (fabs(dPhi) > deltaPhi_) continue;//Delta Phi cut for pair

	   //TODO INSERT MORE CUTS HERE!

	   right_candidates.insert(std::pair<double, int>(theta_r, rr->second));
       }
       //take the closest to left as right
       double min_theta = 999.;
       edm::Ref<reco::TrackCollection> right;
       int right_index = -1;
       for (std::map<double, int>::iterator rr = right_candidates.begin(); rr != right_candidates.end(); ++rr){
	   const double theta_l = (*ll)->innerMomentum().Theta();
	   const double dCotTheta = 1./tan(theta_l) - 1./tan(rr->first);
	   if (fabs(min_theta) > fabs(dCotTheta)){
	       min_theta = dCotTheta;
	       right_index = rr->second;
	       right = allTracks[right_index];
	   }
       }
       if (min_theta <999.){//find a good right track
	   //if all cuts passed, go ahead to make conversion candidates
	   std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
	   trackPairRef.push_back((*ll));//left track
	   trackPairRef.push_back(right);//right track

	   reco::CaloClusterPtrVector scPtrVec;
	   reco::Vertex  theConversionVertex;//Dummy vertex, validity false by default
	   std::vector<math::XYZPoint> trkPositionAtEcal;
	   std::vector<reco::CaloClusterPtr> matchingBC;

	   trkPositionAtEcal.push_back(trackImpactPosition[ll-allTracks.begin()]);//left track
	   if (trackValidECAL[right_index])//second track ECAL position may be invalid
	       trkPositionAtEcal.push_back(trackImpactPosition[right_index]);

	   matchingBC.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
	   scPtrVec.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
	   if (trackValidECAL[right_index]){//second track ECAL position may be invalid
	       matchingBC.push_back(trackMatchedBC[right_index]);
	       scPtrVec.push_back(trackMatchedBC[right_index]);
	   }

	   //TODO: currently, scPtrVec is assigned as matching BC; no Kalman vertex fit, so theConversionVertex validity is false by default
	   //      for first track (called left), trkPositionAtEcal and matchingBC must be valid
	   //      for second track (called right), trkPositionAtEcal and matchingBC is not necessary valid
	   //      so, BE CAREFUL check number of elements before reading them out
	   reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
	   outputConvPhotonCollection.push_back(newCandidate);

	   found_right = true;
	   isPaired[ll-allTracks.begin()] = true;//mark this track is used in pair
	   isPaired[right_index] = true;
	   break; // to get another left leg and start new conversion
       }
       if (!found_right && allowSingleLeg_){
	   std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
	   trackPairRef.push_back((*ll));//left track

	   reco::CaloClusterPtrVector scPtrVec;
	   reco::Vertex  theConversionVertex;//Dummy vertex, validity false by default
	   std::vector<math::XYZPoint> trkPositionAtEcal;
	   std::vector<reco::CaloClusterPtr> matchingBC;

	   trkPositionAtEcal.push_back(trackImpactPosition[ll-allTracks.begin()]);//left track

	   matchingBC.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
	   scPtrVec.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track

	   reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC);
	   outputConvPhotonCollection.push_back(newCandidate);

	   isPaired[ll-allTracks.begin()] = true;//mark this track is used in pair
       }
   } 

   //output sorted as track pair then single track
   outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(), outputConvPhotonCollection.end());
   //outputConvPhotonTrackCollection_p->assign(outputConvPhotonTrackCollection.begin(), outputConvPhotonTrackCollection.end());
   iEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);
   //iEvent.put( outputConvPhotonTrackCollection_p, ConvertedPhotonCollection_);
   //TODO add photon object to reco::Photon

}

// ------------ meth(e called once each job just before starting event loop  ------------
    void
TrackerOnlyConversionProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
TrackerOnlyConversionProducer::endJob() {
}

void
TrackerOnlyConversionProducer::beginRun(const edm::Run& run, const edm::EventSetup& iSetup ) {
}

void
TrackerOnlyConversionProducer::endRun(const edm::Run & run, const edm::EventSetup & iSetup){
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackerOnlyConversionProducer);
