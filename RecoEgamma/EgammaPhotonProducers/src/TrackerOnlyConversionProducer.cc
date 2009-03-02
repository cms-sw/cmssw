// -*- C++ -*-
//
// Package:    TrackerOnlyConversionProducer
// Class:      TrackerOnlyConversionProducer
//
/**\class TrackerOnlyConversionProducer 

 Description: Produces converted photon objects using default track collections

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Hongliang Liu
//         Created:  Thu Mar 13 17:40:48 CDT 2008
// $Id: TrackerOnlyConversionProducer.cc,v 1.3 2009/02/05 17:55:43 nancy Exp $
//
//


// system include files
#include <memory>
#include <map>


#include "RecoEgamma/EgammaPhotonProducers/interface/TrackerOnlyConversionProducer.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

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


TrackerOnlyConversionProducer::TrackerOnlyConversionProducer(const edm::ParameterSet& iConfig)
{
    src_ = iConfig.getParameter<std::vector<edm::InputTag> >("src");

    allowTrackBC_ = iConfig.getParameter<bool>("AllowTrackBC");
    allowD0_ = iConfig.getParameter<bool>("AllowD0");
    allowDeltaCot_ = iConfig.getParameter<bool>("AllowDeltaCot");
    allowMinApproach_ = iConfig.getParameter<bool>("AllowMinApproach");
    allowOppCharge_ = iConfig.getParameter<bool>("AllowOppCharge");

    halfWayEta_ = iConfig.getParameter<double>("HalfwayEta");//open angle to search track matches with BC

    if (allowD0_)
	d0Cut_ = iConfig.getParameter<double>("d0");

    if (allowTrackBC_) {
	//Track-cluster matching eta and phi cuts
	dEtaTkBC_ = iConfig.getParameter<double>("dEtaTrackBC");//TODO research on cut endcap/barrel
	dPhiTkBC_ = iConfig.getParameter<double>("dPhiTrackBC");

	bcBarrelCollection_     = iConfig.getParameter<edm::InputTag>("bcBarrelCollection");
	bcEndcapCollection_     = iConfig.getParameter<edm::InputTag>("bcEndcapCollection");

	energyBC_ = iConfig.getParameter<double>("EnergyBC");//BC energy cut
	energyTotalBC_ = iConfig.getParameter<double>("EnergyTotalBC");//BC energy cut

    }
    //Track cuts on left right track: at least one leg reaches ECAL
    //Left track: must exist, must reach Ecal and match BC, so loose cut on Chi2 and tight on hits
    //Right track: not necessary to exist (if allowSingleLeg_), not necessary to reach ECAL or match BC, so tight cut on Chi2 and loose on hits
    maxChi2Left_ =  iConfig.getParameter<double>("MaxChi2Left");
    maxChi2Right_ =  iConfig.getParameter<double>("MaxChi2Right");
    maxHitsLeft_ = iConfig.getParameter<int>("MaxHitsLeft");
    maxHitsRight_ = iConfig.getParameter<int>("MaxHitsRight");

    //Track Open angle cut on delta cot(theta) and delta phi
    deltaPhi_ = iConfig.getParameter<double>("DeltaPhi");
    if (allowDeltaCot_)
	deltaCotTheta_ = iConfig.getParameter<double>("DeltaCotTheta");
    if (allowMinApproach_)
	minApproach_ = iConfig.getParameter<double>("MinApproach");

    // if allow single track collection, by default False
    allowSingleLeg_ = iConfig.getParameter<bool>("AllowSingleLeg");
    rightBC_ = iConfig.getParameter<bool>("AllowRightBC");

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

//calculate the center of track circle in transverse plane
//muon hypothesis uses AOD and circle based on muon (no brems)
//non-muon hypothesis needs TrackExtra 
void TrackerOnlyConversionProducer::getCircleCenter(const reco::TrackRef& tk, const double r, double& x0, double& y0,  bool muon){
    if (muon){// muon hypothesis
	double x1, y1, phi;
	x1 = tk->vx();//inner position and inner momentum need track Extra!
	y1 = tk->vy();
	phi = tk->phi();
	const int charge = tk->charge();
	x0 = x1 + r*sin(phi)*charge;
	y0 = y1 - r*cos(phi)*charge;
    } else {// works for electron and muon
	double x1, y1, phi;
	x1 = tk->innerPosition().x();//inner position and inner momentum need track Extra!
	y1 = tk->innerPosition().y();
	phi = tk->innerMomentum().phi();
	const int charge = tk->charge();
	x0 = x1 + r*sin(phi)*charge;
	y0 = y1 - r*cos(phi)*charge;
    }
}
void TrackerOnlyConversionProducer::getCircleCenter(const edm::RefToBase<reco::Track>& tk, const double r, double& x0, double& y0,  bool muon){
    if (muon){// muon hypothesis
	double x1, y1, phi;
	x1 = tk->vx();//inner position and inner momentum need track Extra!
	y1 = tk->vy();
	phi = tk->phi();
	const int charge = tk->charge();
	x0 = x1 + r*sin(phi)*charge;
	y0 = y1 - r*cos(phi)*charge;
    } else {// works for electron and muon
	double x1, y1, phi;
	x1 = tk->innerPosition().x();//inner position and inner momentum need track Extra!
	y1 = tk->innerPosition().y();
	phi = tk->innerMomentum().phi();
	const int charge = tk->charge();
	x0 = x1 + r*sin(phi)*charge;
	y0 = y1 - r*cos(phi)*charge;
    }
}
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
   edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;//TODO check cluster type if BasicCluster or PFCluster
   if (allowTrackBC_){
       iEvent.getByLabel( bcBarrelCollection_, bcBarrelHandle);
       iEvent.getByLabel( bcEndcapCollection_, bcEndcapHandle);
   }

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

   if (allowTrackBC_){
       for (unsigned jj = 0; jj < 2; ++jj ){
	   for (unsigned ii = 0; ii < bcHandle->size(); ++ii ) {
	       if (bcHandle->ptrAt(ii)->energy()>energyBC_)
		   basicClusterPtrs.insert(std::make_pair(bcHandle->ptrAt(ii)->position().eta(), bcHandle->ptrAt(ii)));
	   }
	   bcHandle = bcEndcapHandle;
       }
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

       //check if this track matches ECAL or not
       if (allowTrackBC_){
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
   }
   //3. pair up : to select one track from matched EBC, and select the other track to fit cot theta and phi open angle cut
   //TODO it is k-Closest pair of point problem
   reco::VertexCollection vertexs;
   std::vector<bool> isPaired;
   isPaired.assign(allTracks.size(), false);
   for( reco::TrackRefVector::const_iterator ll = allTracks.begin(); ll != allTracks.end(); ++ ll ) {
       //Level 1 loop, in all tracks matched with ECAL
       if ((allowTrackBC_ && !trackValidECAL[ll-allTracks.begin()]) //this Left leg should have valid BC
	       || (allowD0_ && (*ll)->d0()*(*ll)->charge()/(*ll)->d0Error()<d0Cut_) //d0*charge signicifance for converted photons
	       || isPaired[ll-allTracks.begin()]) //this track should not be used in another pair
	   continue;
       const double left_eta = (*ll)->innerMomentum().eta();
       bool found_right = false;//check if right leg found, if no but allowSingleLeg_, go build a conversion with left leg
       std::vector<int> right_candidates;//store all right legs passed the cut (theta/approach and ref pair)
       std::vector<double> right_candidate_theta, right_candidate_approach;

       //select right leg candidates, which passed the cuts
       for (std::multimap<double, int>::iterator rr = trackInnerEta.lower_bound(left_eta - halfWayEta_);
	       rr != trackInnerEta.upper_bound(left_eta + halfWayEta_); ++rr){//select neighbor tracks by eta
	   //Level 2 loop
	   //TODO find the closest one rather than the first matching
	   const edm::Ref<reco::TrackCollection> & right = allTracks[rr->second];

	   if (right->normalizedChi2() > maxChi2Right_ || right->found() < maxHitsRight_ //track quality cut
		   || isPaired[rr->second] //this track should not be in another pair
		   //|| right == (*ll) //no double counting (dummy if require opposite charge)
		   || (allowD0_ && right->d0()*right->charge()/(right->d0Error())<d0Cut_) //d0*charge significance for converted photons
		   || (allowTrackBC_ && !trackValidECAL[rr->second] && rightBC_) // if right track matches ECAL
		   || (allowOppCharge_ && (*ll)->charge())*(right->charge()) > 0) //opposite charge
	       continue;

	   //DeltaPhi as preselection cut
	   const double phi_l = (*ll)->innerMomentum().phi();
	   const double phi_r = right->innerMomentum().phi();
	   const double dPhi = phi_l - phi_r;

	   if (fabs(dPhi) > deltaPhi_) continue;//Delta Phi cut for pair

	   if (allowTrackBC_){
	       //check energy of BC
	       double total_e_bc = trackMatchedBC[ll-allTracks.begin()]->energy();
	       if (rightBC_) total_e_bc += trackMatchedBC[rr->second]->energy();

	       if (total_e_bc < energyTotalBC_) continue;
	   }

	   double theta_r = 999.;
	   if (allowDeltaCot_){
	       const double theta_l = (*ll)->innerMomentum().Theta();
	       theta_r = right->innerMomentum().Theta();
	       const double dCotTheta =  1./tan(theta_l) - 1./tan(theta_r) ;

	       if (fabs(dCotTheta) > deltaCotTheta_) continue;//Delta Cot(Theta) cut for pair
	   }

	   double distance = -9999.;
	   if (allowMinApproach_){
	       double x_l, x_r, y_l, y_r;

	       const double xx_l = (*ll)->innerPosition().x(), yy_l = (*ll)->innerPosition().y(), zz_l = (*ll)->innerPosition().z();
	       const double xx_r = right->innerPosition().x(), yy_r = right->innerPosition().y(), zz_r = right->innerPosition().z();
	       const double radius_l = (*ll)->innerMomentum().Rho()/(.3*(magField->inTesla(GlobalPoint(xx_l, yy_l, zz_l)).z()))*100;
	       const double radius_r = right->innerMomentum().Rho()/(.3*(magField->inTesla(GlobalPoint(xx_r, yy_r, zz_r)).z()))*100;
	       getCircleCenter((*ll), radius_l, x_l, y_l);
	       getCircleCenter(right, radius_r, x_r, y_r);
	       distance = sqrt((x_l-x_r)*(x_l-x_r)+(y_l-y_r)*(y_l-y_r)) - radius_l - radius_r;

	       if (distance < minApproach_) continue;
	   }
	   //TODO INSERT MORE CUTS HERE!

	   right_candidates.push_back(rr->second);
	   right_candidate_theta.push_back(theta_r);
	   right_candidate_approach.push_back(distance);
       }
       //take the closest to left as right
       double min_theta = 999., min_approach = -999;;
       edm::Ref<reco::TrackCollection> right;
       int right_index = -1;
       for (unsigned int ii = 0; ii< right_candidates.size(); ++ii){
	   const double theta_l = (*ll)->innerMomentum().Theta();
	   const double dCotTheta = 1./tan(theta_l) - 1./tan(right_candidate_theta[ii]);
	   const double distance = right_candidate_approach[ii];
	   if (fabs(min_theta) > fabs(dCotTheta) 
		   &&  min_approach <= distance){
	       min_theta = dCotTheta;
	       min_approach = distance;
	       right_index = right_candidates[ii];;
	       right = allTracks[right_index];
	   }
       }
       if (right_index > -1){//find a good right track
	   //if all cuts passed, go ahead to make conversion candidates
	   std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
	   trackPairRef.push_back((*ll));//left track
	   trackPairRef.push_back(right);//right track

	   std::vector<math::XYZVector> trackPin;
	   std::vector<math::XYZVector> trackPout;

	   trackPin.push_back((*ll)->innerMomentum());
	   trackPin.push_back(right->innerMomentum());
	   trackPout.push_back((*ll)->outerMomentum());
	   trackPout.push_back(right->outerMomentum());

	   reco::CaloClusterPtrVector scPtrVec;
	   reco::Vertex  theConversionVertex;//Dummy vertex, validity false by default
	   std::vector<math::XYZPoint> trkPositionAtEcal;
	   std::vector<reco::CaloClusterPtr> matchingBC;

	   if (allowTrackBC_){//TODO find out the BC ptrs if not doing matching, otherwise, leave it empty
	       trkPositionAtEcal.push_back(trackImpactPosition[ll-allTracks.begin()]);//left track
	       if (trackValidECAL[right_index])//second track ECAL position may be invalid
		   trkPositionAtEcal.push_back(trackImpactPosition[right_index]);

	       matchingBC.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
	       scPtrVec.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
	       if (trackValidECAL[right_index]){//second track ECAL position may be invalid
		   matchingBC.push_back(trackMatchedBC[right_index]);
		   if (!(trackMatchedBC[right_index] == trackMatchedBC[ll-allTracks.begin()]))//avoid 1 BC 2 tk
		       scPtrVec.push_back(trackMatchedBC[right_index]);
	       }
	   }
	   const float minAppDist = min_approach;

	   //TODO: currently, scPtrVec is assigned as matching BC; no Kalman vertex fit, so theConversionVertex validity is false by default
	   //      for first track (called left), trkPositionAtEcal and matchingBC must be valid
	   //      for second track (called right), trkPositionAtEcal and matchingBC is not necessary valid
	   //      so, BE CAREFUL check number of elements before reading them out
	   reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC, minAppDist, trackPin, trackPout );
	   outputConvPhotonCollection.push_back(newCandidate);

	   found_right = true;
	   isPaired[ll-allTracks.begin()] = true;//mark this track is used in pair
	   isPaired[right_index] = true;
	   break; // to get another left leg and start new conversion
       }
       if (!found_right && allowSingleLeg_){
	   /*
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
	   */
       }
   } 

   //output sorted as track pair then single track
   outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(), outputConvPhotonCollection.end());
   iEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);

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

