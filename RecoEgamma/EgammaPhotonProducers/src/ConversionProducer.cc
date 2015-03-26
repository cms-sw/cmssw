// -*- C++ -*-
//
// Package:    ConversionProducer
// Class:      ConversionProducer
//
/**\class ConversionProducer 

Description: Produces converted photon objects using default track collections

Implementation:
<Notes on implementation>
*/
//
// Original Authors:  Hongliang Liu
//         Created:  Thu Mar 13 17:40:48 CDT 2008
//
//


// system include files
#include <memory>
#include <map>


#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionProducer.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h"


#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionVertexFinder.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/TangentApproachInRPhi.h"
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionHitChecker.h"


//Kinematic constraint vertex fitter
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include <RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h>
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/ColinearityKinematicConstraint.h"



ConversionProducer::ConversionProducer(const edm::ParameterSet& iConfig):
  theVertexFinder_(0)

{
  algoName_ = iConfig.getParameter<std::string>( "AlgorithmName" );

  src_ = 
    consumes<edm::View<reco::ConversionTrack> >(iConfig.getParameter<edm::InputTag>("src"));

  maxNumOfTrackInPU_ = iConfig.getParameter<int>("maxNumOfTrackInPU");
  maxTrackRho_ = iConfig.getParameter<double>("maxTrackRho");
  maxTrackZ_ = iConfig.getParameter<double>("maxTrackZ");

  allowTrackBC_ = iConfig.getParameter<bool>("AllowTrackBC");
  allowD0_ = iConfig.getParameter<bool>("AllowD0");
  allowDeltaPhi_ = iConfig.getParameter<bool>("AllowDeltaPhi");
  allowDeltaCot_ = iConfig.getParameter<bool>("AllowDeltaCot");
  allowMinApproach_ = iConfig.getParameter<bool>("AllowMinApproach");
  allowOppCharge_ = iConfig.getParameter<bool>("AllowOppCharge");

  allowVertex_ = iConfig.getParameter<bool>("AllowVertex");

  bypassPreselGsf_ = iConfig.getParameter<bool>("bypassPreselGsf");
  bypassPreselEcal_ = iConfig.getParameter<bool>("bypassPreselEcal");
  bypassPreselEcalEcal_ = iConfig.getParameter<bool>("bypassPreselEcalEcal");
  
  deltaEta_ = iConfig.getParameter<double>("deltaEta");
  
  halfWayEta_ = iConfig.getParameter<double>("HalfwayEta");//open angle to search track matches with BC

  d0Cut_ = iConfig.getParameter<double>("d0");
    
  usePvtx_ = iConfig.getParameter<bool>("UsePvtx");//if use primary vertices

  vertexProducer_   = 
    consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertexProducer"));
  

  //Track-cluster matching eta and phi cuts
  dEtaTkBC_ = iConfig.getParameter<double>("dEtaTrackBC");//TODO research on cut endcap/barrel
  dPhiTkBC_ = iConfig.getParameter<double>("dPhiTrackBC");
  
  bcBarrelCollection_ = 
    consumes<edm::View<reco::CaloCluster> >(iConfig.getParameter<edm::InputTag>("bcBarrelCollection"));
  bcEndcapCollection_ = 
    consumes<edm::View<reco::CaloCluster> >(iConfig.getParameter<edm::InputTag>("bcEndcapCollection"));
  
  scBarrelProducer_   = 
    consumes<edm::View<reco::CaloCluster> >(iConfig.getParameter<edm::InputTag>("scBarrelProducer"));
  scEndcapProducer_   = 
    consumes<edm::View<reco::CaloCluster> >(iConfig.getParameter<edm::InputTag>("scEndcapProducer"));
  
  energyBC_               = iConfig.getParameter<double>("EnergyBC");//BC energy threshold
  energyTotalBC_          = iConfig.getParameter<double>("EnergyTotalBC");//BC energy threshold
  minSCEt_                = iConfig.getParameter<double>("minSCEt");//super cluster energy threshold
  dEtacutForSCmatching_     = iConfig.getParameter<double>("dEtacutForSCmatching");// dEta between conversion momentum direction and SC position
  dPhicutForSCmatching_     = iConfig.getParameter<double>("dPhicutForSCmatching");// dPhi between conversion momentum direction and SC position

  
   

  //Track cuts on left right track: at least one leg reaches ECAL
  //Left track: must exist, must reach Ecal and match BC, so loose cut on Chi2 and tight on hits
  //Right track: not necessary to exist (if allowSingleLeg_), not necessary to reach ECAL or match BC, so tight cut on Chi2 and loose on hits
  maxChi2Left_ =  iConfig.getParameter<double>("MaxChi2Left");
  maxChi2Right_ =  iConfig.getParameter<double>("MaxChi2Right");
  minHitsLeft_ = iConfig.getParameter<int>("MinHitsLeft");
  minHitsRight_ = iConfig.getParameter<int>("MinHitsRight");

  //Track Open angle cut on delta cot(theta) and delta phi
  deltaCotTheta_ = iConfig.getParameter<double>("DeltaCotTheta");
  deltaPhi_ = iConfig.getParameter<double>("DeltaPhi");
  minApproachLow_ = iConfig.getParameter<double>("MinApproachLow");
  minApproachHigh_ = iConfig.getParameter<double>("MinApproachHigh");
  

  // if allow single track collection, by default False
  allowSingleLeg_ = iConfig.getParameter<bool>("AllowSingleLeg");
  rightBC_ = iConfig.getParameter<bool>("AllowRightBC");

  //track inner position dz cut, need RECO
  dzCut_ = iConfig.getParameter<double>("dz");
  //track analytical cross cut
  r_cut = iConfig.getParameter<double>("rCut");
  vtxChi2_ = iConfig.getParameter<double>("vtxChi2");


  theVertexFinder_ = new ConversionVertexFinder ( iConfig );

  thettbuilder_ = 0;

  //output
  ConvertedPhotonCollection_     = iConfig.getParameter<std::string>("convertedPhotonCollection");

  produces< reco::ConversionCollection >(ConvertedPhotonCollection_);

}


ConversionProducer::~ConversionProducer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete theVertexFinder_;
}


// ------------ method called to produce the data  ------------
void
ConversionProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  reco::ConversionCollection outputConvPhotonCollection;
  std::auto_ptr<reco::ConversionCollection> outputConvPhotonCollection_p(new reco::ConversionCollection);

  //std::cout << " ConversionProducer::produce " << std::endl;
  //Read multiple track input collections

  edm::Handle<edm::View<reco::ConversionTrack> > trackCollectionHandle;
  iEvent.getByToken(src_,trackCollectionHandle);    

  //build map of ConversionTracks ordered in eta
  std::multimap<float, edm::Ptr<reco::ConversionTrack> > convTrackMap;
  edm::PtrVector<reco::ConversionTrack> trackPtrVector;
  for (size_t i = 0; i < trackCollectionHandle->size(); ++i)
    trackPtrVector.push_back(trackCollectionHandle->ptrAt(i));

  for (edm::PtrVector<reco::ConversionTrack>::const_iterator tk_ref = trackPtrVector.begin(); tk_ref != trackPtrVector.end(); ++tk_ref ){
    convTrackMap.insert(std::make_pair((*tk_ref)->track()->eta(),*tk_ref));
  }

  edm::Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  if (usePvtx_){
    iEvent.getByToken(vertexProducer_, vertexHandle);
    if (!vertexHandle.isValid()) {
      edm::LogError("ConversionProducer") 
	<< "Error! Can't get the product primary Vertex Collection "<< "\n";
      usePvtx_ = false;
    }
    if (usePvtx_)
      vertexCollection = *(vertexHandle.product());
  }
    
  edm::ESHandle<TransientTrackBuilder> hTransientTrackBuilder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",hTransientTrackBuilder);
  thettbuilder_ = hTransientTrackBuilder.product();
    
  reco::Vertex the_pvtx;
  //because the priamry vertex is sorted by quality, the first one is the best
  if (!vertexCollection.empty())
    the_pvtx = *(vertexCollection.begin());
    
  if (trackCollectionHandle->size()> maxNumOfTrackInPU_){
    iEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);
    return;
  }
    
    
  // build Super and Basic cluster geometry map to search in eta bounds for clusters
  std::multimap<double, reco::CaloClusterPtr> basicClusterPtrs;
  std::multimap<double, reco::CaloClusterPtr> superClusterPtrs;

  
  buildSuperAndBasicClusterGeoMap(iEvent,basicClusterPtrs,superClusterPtrs);
      
  buildCollection( iEvent, iSetup, convTrackMap,  superClusterPtrs, basicClusterPtrs, the_pvtx, outputConvPhotonCollection);//allow empty basicClusterPtrs
    
  outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(), outputConvPhotonCollection.end());
  iEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);
    
}


void ConversionProducer::buildSuperAndBasicClusterGeoMap(const edm::Event& iEvent,  
							 std::multimap<double, reco::CaloClusterPtr>& basicClusterPtrs,
							 std::multimap<double, reco::CaloClusterPtr>& superClusterPtrs){

  // Get the Super Cluster collection in the Barrel
  edm::Handle<edm::View<reco::CaloCluster> > scBarrelHandle;
  iEvent.getByToken(scBarrelProducer_,scBarrelHandle);
  if (!scBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") 
      << "Error! Can't get the barrel superclusters!";
  }
    
  // Get the Super Cluster collection in the Endcap
  edm::Handle<edm::View<reco::CaloCluster> > scEndcapHandle;
  iEvent.getByToken(scEndcapProducer_,scEndcapHandle);
  if (!scEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") 
      << "Error! Can't get the endcap superclusters!";
  }
    
    
  edm::Handle<edm::View<reco::CaloCluster> > bcBarrelHandle;
  edm::Handle<edm::View<reco::CaloCluster> > bcEndcapHandle;//TODO check cluster type if BasicCluster or PFCluster

  iEvent.getByToken( bcBarrelCollection_, bcBarrelHandle);
  if (!bcBarrelHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") 
      << "Error! Can't get the barrel basic clusters!";
  }

  iEvent.getByToken( bcEndcapCollection_, bcEndcapHandle);
  if (! bcEndcapHandle.isValid()) {
    edm::LogError("ConvertedPhotonProducer") 
      << "Error! Can't get the endcap basic clusters!";
  }

  edm::Handle<edm::View<reco::CaloCluster> > bcHandle = bcBarrelHandle;
  edm::Handle<edm::View<reco::CaloCluster> > scHandle = scBarrelHandle;

  if ( bcHandle.isValid()  ) {    
    for (unsigned jj = 0; jj < 2; ++jj ){
      for (unsigned ii = 0; ii < bcHandle->size(); ++ii ) {
	if (bcHandle->ptrAt(ii)->energy()>energyBC_)
	  basicClusterPtrs.insert(std::make_pair(bcHandle->ptrAt(ii)->position().eta(), bcHandle->ptrAt(ii)));
      }
      bcHandle = bcEndcapHandle;
    }
  }


  if ( scHandle.isValid()  ) {
    for (unsigned jj = 0; jj < 2; ++jj ){
      for (unsigned ii = 0; ii < scHandle->size(); ++ii ) {
	if (scHandle->ptrAt(ii)->energy()>minSCEt_)
	  superClusterPtrs.insert(std::make_pair(scHandle->ptrAt(ii)->position().eta(), scHandle->ptrAt(ii)));
      }
      scHandle = scEndcapHandle;
    } 
  }


}


void ConversionProducer::buildCollection(edm::Event& iEvent, const edm::EventSetup& iSetup,
                                         const std::multimap<float, edm::Ptr<reco::ConversionTrack> >& allTracks,
                                         const std::multimap<double, reco::CaloClusterPtr>& superClusterPtrs,
                                         const std::multimap<double, reco::CaloClusterPtr>& basicClusterPtrs,
                                         const reco::Vertex& the_pvtx,
                                         reco::ConversionCollection & outputConvPhotonCollection){
  
  
  edm::ESHandle<TrackerGeometry> trackerGeomHandle;
  edm::ESHandle<MagneticField> magFieldHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeomHandle );
  iSetup.get<IdealMagneticFieldRecord>().get( magFieldHandle );
  
  const TrackerGeometry* trackerGeom = trackerGeomHandle.product();
  const MagneticField* magField = magFieldHandle.product();
  
//   std::vector<math::XYZPointF> trackImpactPosition;
//   trackImpactPosition.reserve(allTracks.size());//track impact position at ECAL
//   std::vector<bool> trackValidECAL;//Does this track reach ECAL basic cluster (reach ECAL && match with BC)
//   trackValidECAL.assign(allTracks.size(), false);
//   
//   std::vector<reco::CaloClusterPtr> trackMatchedBC;
//   reco::CaloClusterPtr empty_bc;
//   trackMatchedBC.assign(allTracks.size(), empty_bc);//TODO find a better way to avoid copy constructor
//   
//   std::vector<int> bcHandleId;//the associated BC handle id, -1 invalid, 0 barrel 1 endcap
//   bcHandleId.assign(allTracks.size(), -1);
  
  // not used    std::multimap<double, int> trackInnerEta;//Track innermost state Eta map to TrackRef index, to be used in track pair sorting
  
  std::map<edm::Ptr<reco::ConversionTrack>, math::XYZPointF> trackImpactPosition;
  std::map<edm::Ptr<reco::ConversionTrack>, reco::CaloClusterPtr> trackMatchedBC;
  
  ConversionHitChecker hitChecker;
  
  
  //2 propagate all tracks into ECAL, record its eta and phi
 
  for (std::multimap<float, edm::Ptr<reco::ConversionTrack> >::const_iterator tk_ref = allTracks.begin(); tk_ref != allTracks.end(); ++tk_ref ){
    const reco::Track* tk = tk_ref->second->trackRef().get()  ;
    
    
    //check impact position then match with BC
    math::XYZPointF ew;
    if ( getTrackImpactPosition(tk, trackerGeom, magField, ew) ){
      trackImpactPosition[tk_ref->second] = ew;
      
      reco::CaloClusterPtr closest_bc;//the closest matching BC to track
      
      if ( getMatchedBC(basicClusterPtrs, ew, closest_bc) ){
	trackMatchedBC[tk_ref->second] = closest_bc;
      }
    }    
  }
  
  
  
  //3. pair up tracks: 
  //TODO it is k-Closest pair of point problem
  //std::cout << " allTracks.size() " <<  allTracks.size() << std::endl;
  for(std::multimap<float, edm::Ptr<reco::ConversionTrack> >::const_iterator ll = allTracks.begin(); ll != allTracks.end();  ++ll ) {
    bool track1HighPurity=true;
    //std::cout << " Loop on allTracks " << std::endl;
    const  edm::RefToBase<reco::Track> & left = ll->second->trackRef();
    

    //TODO: This is a workaround, should be fixed with a proper function in the TTBuilder
    //(Note that the TrackRef and GsfTrackRef versions of the constructor are needed
    // to properly get refit tracks in the output vertex)
    reco::TransientTrack ttk_l;
    if (dynamic_cast<const reco::GsfTrack*>(left.get())) {
      ttk_l = thettbuilder_->build(left.castTo<reco::GsfTrackRef>());
    }
    else {
      ttk_l = thettbuilder_->build(left.castTo<reco::TrackRef>());
    }
    
    ////  Remove BC matching from track selection 
    //      if ((allowTrackBC_ && !trackValidECAL[ll-allTracks.begin()]) )//this Left leg should have valid BC
    // continue;
    
    
    if (the_pvtx.isValid()){
      if (!(trackD0Cut(left, the_pvtx)))   track1HighPurity=false;
    } else {
      if (!(trackD0Cut(left)))  track1HighPurity=false;
    }
    
    std::vector<int> right_candidates;//store all right legs passed the cut (theta/approach and ref pair)
    std::vector<double> right_candidate_theta, right_candidate_approach;
    std::vector<std::pair<bool, reco::Vertex> > vertex_candidates;
    
    //inner loop only over tracks between eta and eta + deltaEta of the first track
    float etasearch = ll->first + deltaEta_;
    std::multimap<float, edm::Ptr<reco::ConversionTrack> >::const_iterator rr = ll;
    ++rr;    
    for (; rr != allTracks.lower_bound(etasearch); ++rr ) {
      bool track2HighPurity = true;
      bool highPurityPair = true;
      
      const  edm::RefToBase<reco::Track> & right = rr->second->trackRef();
      
      
      //TODO: This is a workaround, should be fixed with a proper function in the TTBuilder
      reco::TransientTrack ttk_r;
      if (dynamic_cast<const reco::GsfTrack*>(right.get())) {
        ttk_r = thettbuilder_->build(right.castTo<reco::GsfTrackRef>());
      }
      else {
        ttk_r = thettbuilder_->build(right.castTo<reco::TrackRef>());
      }
      //std::cout << " This track is " <<  right->algoName() << std::endl;
      
      
      //all vertexing preselection should go here
      
      //check for opposite charge
      if (  allowOppCharge_ && (left->charge()*right->charge() > 0) )  
        continue; //same sign, reject pair
          
      ////  Remove BC matching from track selection 
      //if ( (allowTrackBC_ && !trackValidECAL[rr-allTracks.begin()] && rightBC_) )// if right track matches ECAL
      //  continue;
          
          
      double approachDist = -999.;
      //apply preselection to track pair, overriding preselection for gsf+X or ecalseeded+X pairs if so configured
      bool preselected = preselectTrackPair(ttk_l,ttk_r, approachDist);
      preselected = preselected || (bypassPreselGsf_ && (left->algo()==reco::TrackBase::gsf || right->algo()==reco::TrackBase::gsf));
      preselected = preselected || (bypassPreselEcal_ && (left->algo()==reco::TrackBase::outInEcalSeededConv || right->algo()==reco::TrackBase::outInEcalSeededConv || left->algo()==reco::TrackBase::inOutEcalSeededConv || right->algo()==reco::TrackBase::inOutEcalSeededConv));
      preselected = preselected || (bypassPreselEcalEcal_ && (left->algo()==reco::TrackBase::outInEcalSeededConv || left->algo()==reco::TrackBase::inOutEcalSeededConv) && (right->algo()==reco::TrackBase::outInEcalSeededConv || right->algo()==reco::TrackBase::inOutEcalSeededConv));
      
      if (!preselected) {
        continue;
      }
          
      //do the actual vertex fit
      reco::Vertex theConversionVertex;//by default it is invalid          
      bool goodVertex = checkVertex(ttk_l, ttk_r, magField, theConversionVertex);
          
      //bail as early as possible in case the fit didn't return a good vertex
      if (!goodVertex) {
        continue;
      }

          
          
      //track pair pass the quality cut
      if (   !( (trackQualityFilter(left, true) && trackQualityFilter(right, false))
                || (trackQualityFilter(left, false) && trackQualityFilter(right, true)) ) ) {
        highPurityPair=false;
      }

      if (the_pvtx.isValid()){
        if (!(trackD0Cut(right, the_pvtx))) track2HighPurity=false; 
      } else {
        if (!(trackD0Cut(right))) track2HighPurity=false; 
      }
        

      //if all cuts passed, go ahead to make conversion candidates
      std::vector<edm::RefToBase<reco::Track> > trackPairRef;
      trackPairRef.push_back(left);//left track
      trackPairRef.push_back(right);//right track

      std::vector<math::XYZVectorF> trackPin;
      std::vector<math::XYZVectorF> trackPout;
      std::vector<math::XYZPointF> trackInnPos;
      std::vector<uint8_t> nHitsBeforeVtx;
      std::vector<Measurement1DFloat> dlClosestHitToVtx;

      if (left->extra().isNonnull() && right->extra().isNonnull()){//only available on TrackExtra
        trackInnPos.push_back(  toFConverterP(left->innerPosition()));
        trackInnPos.push_back(  toFConverterP(right->innerPosition()));
        trackPin.push_back(toFConverterV(left->innerMomentum()));
        trackPin.push_back(toFConverterV(right->innerMomentum()));
        trackPout.push_back(toFConverterV(left->outerMomentum()));
	trackPout.push_back(toFConverterV(right->outerMomentum()));
      }
          
      if (ll->second->trajRef().isNonnull() && rr->second->trajRef().isNonnull()) {
        std::pair<uint8_t,Measurement1DFloat> leftWrongHits = hitChecker.nHitsBeforeVtx(*ll->second->trajRef().get(),theConversionVertex);
        std::pair<uint8_t,Measurement1DFloat> rightWrongHits = hitChecker.nHitsBeforeVtx(*rr->second->trajRef().get(),theConversionVertex);
        nHitsBeforeVtx.push_back(leftWrongHits.first);
        nHitsBeforeVtx.push_back(rightWrongHits.first);
        dlClosestHitToVtx.push_back(leftWrongHits.second);
        dlClosestHitToVtx.push_back(rightWrongHits.second);            
      }
          
      uint8_t nSharedHits = hitChecker.nSharedHits(*left.get(),*right.get());


      //if using kinematic fit, check with chi2 post cut
      if (theConversionVertex.isValid()){
        const float chi2Prob = ChiSquaredProbability(theConversionVertex.chi2(), theConversionVertex.ndof());
        if (chi2Prob<vtxChi2_)  highPurityPair=false;
      }

      //std::cout << "  highPurityPair after vertex cut " <<  highPurityPair << std::endl;
      std::vector<math::XYZPointF> trkPositionAtEcal;
      std::vector<reco::CaloClusterPtr> matchingBC;

      if (allowTrackBC_){//TODO find out the BC ptrs if not doing matching, otherwise, leave it empty
        //const int lbc_handle = bcHandleId[ll-allTracks.begin()],
        //	      rbc_handle = bcHandleId[rr-allTracks.begin()];

        std::map<edm::Ptr<reco::ConversionTrack>, math::XYZPointF>::const_iterator trackImpactPositionLeft = trackImpactPosition.find(ll->second);
        std::map<edm::Ptr<reco::ConversionTrack>, math::XYZPointF>::const_iterator trackImpactPositionRight = trackImpactPosition.find(rr->second);
        std::map<edm::Ptr<reco::ConversionTrack>, reco::CaloClusterPtr>::const_iterator trackMatchedBCLeft = trackMatchedBC.find(ll->second);        
        std::map<edm::Ptr<reco::ConversionTrack>, reco::CaloClusterPtr>::const_iterator trackMatchedBCRight = trackMatchedBC.find(rr->second);        
        
        if (trackImpactPositionLeft!=trackImpactPosition.end()) {
          trkPositionAtEcal.push_back(trackImpactPositionLeft->second);//left track
        }
        else {
          trkPositionAtEcal.push_back(math::XYZPointF());//left track
        }
        if (trackImpactPositionRight!=trackImpactPosition.end()) {//second track ECAL position may be invalid
          trkPositionAtEcal.push_back(trackImpactPositionRight->second);
        }

        double total_e_bc = 0.;
        if (trackMatchedBCLeft!=trackMatchedBC.end()) {
          matchingBC.push_back(trackMatchedBCLeft->second);//left track
          total_e_bc += trackMatchedBCLeft->second->energy();
        }
        else {
          matchingBC.push_back( reco::CaloClusterPtr() );//left track
        }
        if (trackMatchedBCRight!=trackMatchedBC.end()) {//second track ECAL position may be invalid
          matchingBC.push_back(trackMatchedBCRight->second);
          total_e_bc += trackMatchedBCRight->second->energy();
        }
        
        if (total_e_bc<energyTotalBC_) {
          highPurityPair = false;
        }


      }
      //signature cuts, then check if vertex, then post-selection cuts
      highPurityPair = highPurityPair && track1HighPurity && track2HighPurity && goodVertex && checkPhi(left, right, trackerGeom, magField, theConversionVertex) ;


      /// match the track pair with a SC. If at least one track matches, store the SC
      /*
        for ( std::vector<edm::RefToBase<reco::Track> >::iterator iTk=trackPairRef.begin();  iTk!=trackPairRef.end(); iTk++) {
        math::XYZPointF impPos;
        if ( getTrackImpactPosition(*iTk, trackerGeom, magField, impPos) ) {
	       
        }

        }
      */

      const float minAppDist = approachDist;
      reco::Conversion::ConversionAlgorithm algo = reco::Conversion::algoByName(algoName_);
      float dummy=0;
      reco::CaloClusterPtrVector scPtrVec;
      reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC, minAppDist,  trackInnPos, trackPin, trackPout, nHitsBeforeVtx, dlClosestHitToVtx, nSharedHits, dummy, algo );
      // Fill in scPtrVec with the macthing SC
      if ( matchingSC ( superClusterPtrs, newCandidate, scPtrVec) ) 
        newCandidate.setMatchingSuperCluster( scPtrVec);
          
      //std::cout << " ConversionProducer  scPtrVec.size " <<  scPtrVec.size() << std::endl;
          
      newCandidate.setQuality(reco::Conversion::highPurity,  highPurityPair);
      bool generalTracksOnly = ll->second->isTrackerOnly() && rr->second->isTrackerOnly() && !dynamic_cast<const reco::GsfTrack*>(ll->second->trackRef().get()) && !dynamic_cast<const reco::GsfTrack*>(rr->second->trackRef().get());
      bool arbitratedEcalSeeded = ll->second->isArbitratedEcalSeeded() && rr->second->isArbitratedEcalSeeded();
      bool arbitratedMerged = ll->second->isArbitratedMerged() && rr->second->isArbitratedMerged();
      bool arbitratedMergedEcalGeneral = ll->second->isArbitratedMergedEcalGeneral() && rr->second->isArbitratedMergedEcalGeneral();          
          
      newCandidate.setQuality(reco::Conversion::generalTracksOnly,  generalTracksOnly);
      newCandidate.setQuality(reco::Conversion::arbitratedEcalSeeded,  arbitratedEcalSeeded);
      newCandidate.setQuality(reco::Conversion::arbitratedMerged,  arbitratedMerged);
      newCandidate.setQuality(reco::Conversion::arbitratedMergedEcalGeneral,  arbitratedMergedEcalGeneral);          
          
      outputConvPhotonCollection.push_back(newCandidate);

    }
      
  }






}





//
// member functions
//

inline bool ConversionProducer::trackQualityFilter(const  edm::RefToBase<reco::Track>&   ref, bool isLeft){
  bool pass = true;
  if (isLeft){
    pass = (ref->normalizedChi2() < maxChi2Left_ && ref->found() >= minHitsLeft_);
  } else {
    pass = (ref->normalizedChi2() < maxChi2Right_ && ref->found() >= minHitsRight_);
  }

  return pass;
}

inline bool ConversionProducer::trackD0Cut(const  edm::RefToBase<reco::Track>&  ref){
  //NOTE if not allow d0 cut, always true
  return ((!allowD0_) || !(ref->d0()*ref->charge()/ref->d0Error()<d0Cut_));
}

inline bool ConversionProducer::trackD0Cut(const edm::RefToBase<reco::Track>&  ref, const reco::Vertex& the_pvtx){
  //
  return ((!allowD0_) || !(-ref->dxy(the_pvtx.position())*ref->charge()/ref->dxyError()<d0Cut_));
}


bool ConversionProducer::getTrackImpactPosition(const reco::Track* tk_ref,
                                                const TrackerGeometry* trackerGeom, const MagneticField* magField,
                                                math::XYZPointF& ew){

  PropagatorWithMaterial propag( alongMomentum, 0.000511, magField );
  
  ReferenceCountingPointer<Surface> ecalWall(
                                             new  BoundCylinder(129.f, GlobalPoint(0.,0.,0.), TkRotation<float>(),
                                                                 new SimpleCylinderBounds( 129, 129, -320.5, 320.5 ) ) );
  const float epsilon = 0.001;
  Surface::RotationType rot; // unit rotation matrix
  const float barrelRadius = 129.f;
  const float barrelHalfLength = 270.9f;
  const float endcapRadius = 171.1f;
  const float endcapZ = 320.5f;
  ReferenceCountingPointer<BoundCylinder>  theBarrel_(new BoundCylinder(barrelRadius, Surface::PositionType(0,0,0), rot,
                                                                         new SimpleCylinderBounds( barrelRadius-epsilon, barrelRadius+epsilon, 
-barrelHalfLength, barrelHalfLength)));
  ReferenceCountingPointer<BoundDisk>      theNegativeEtaEndcap_(
                                                                 new BoundDisk( Surface::PositionType( 0, 0, -endcapZ), rot,
                                                                                new SimpleDiskBounds( 0, endcapRadius, -epsilon, epsilon)));
  ReferenceCountingPointer<BoundDisk>      thePositiveEtaEndcap_(
                                                                 new BoundDisk( Surface::PositionType( 0, 0, endcapZ), rot,
                                                                                new SimpleDiskBounds( 0, endcapRadius, -epsilon, epsilon)));

  //const TrajectoryStateOnSurface myTSOS = trajectoryStateTransform::innerStateOnSurface(*(*ref), *trackerGeom, magField);
  const TrajectoryStateOnSurface myTSOS = trajectoryStateTransform::outerStateOnSurface(*tk_ref, *trackerGeom, magField);
  TrajectoryStateOnSurface  stateAtECAL;
  stateAtECAL = propag.propagate(myTSOS, *theBarrel_);
  if (!stateAtECAL.isValid() || ( stateAtECAL.isValid() && fabs(stateAtECAL.globalPosition().eta() ) >1.479f )  ) {
    //endcap propagator
    if (myTSOS.globalPosition().z() > 0.) {
	    stateAtECAL = propag.propagate(myTSOS, *thePositiveEtaEndcap_);
    } else {
	    stateAtECAL = propag.propagate(myTSOS, *theNegativeEtaEndcap_);
    }
  }       
  if (stateAtECAL.isValid()){
    ew = stateAtECAL.globalPosition();
    return true;
  }
  else
    return false;
}




bool ConversionProducer::matchingSC(const std::multimap<double, reco::CaloClusterPtr>& scMap, 
                                    reco::Conversion& aConv,
                                    // reco::CaloClusterPtr& mSC){
                                    reco::CaloClusterPtrVector& mSC) {

  //  double dRMin=999.;
  double detaMin=999.;
  double dphiMin=999.;                 
  reco::CaloClusterPtr match;
  for (std::multimap<double, reco::CaloClusterPtr>::const_iterator scItr = scMap.begin();  scItr != scMap.end(); scItr++) {
    const reco::CaloClusterPtr& sc = scItr->second; 
    const double delta_phi = reco::deltaPhi( aConv.refittedPairMomentum().phi(), sc->phi());
    double sceta = sc->eta();
    double conveta = etaTransformation(aConv.refittedPairMomentum().eta(), aConv.zOfPrimaryVertexFromTracks() );
    const double delta_eta = fabs(conveta - sceta);
    if ( fabs(delta_eta) < fabs(detaMin) && fabs(delta_phi) < fabs(dphiMin) ) {
      detaMin=  fabs(delta_eta);
      dphiMin=  fabs(delta_phi);
      match=sc;
    }
  }
  
  if ( fabs(detaMin) < dEtacutForSCmatching_ && fabs(dphiMin) < dPhicutForSCmatching_ ) {
    mSC.push_back(match);
    return true;
  } else 
    return false;
}

bool ConversionProducer::getMatchedBC(const std::multimap<double, reco::CaloClusterPtr>& bcMap, 
                                      const math::XYZPointF& trackImpactPosition,
                                      reco::CaloClusterPtr& closestBC){
  const double track_eta = trackImpactPosition.eta();
  const double track_phi = trackImpactPosition.phi();

  double min_eta = 999., min_phi = 999.;
  reco::CaloClusterPtr closest_bc;
  for (std::multimap<double, reco::CaloClusterPtr>::const_iterator bc = bcMap.lower_bound(track_eta - halfWayEta_);
       bc != bcMap.upper_bound(track_eta + halfWayEta_); ++bc){//use eta map to select possible BC collection then loop in
    const reco::CaloClusterPtr& ebc = bc->second;
    const double delta_eta = track_eta-(ebc->position().eta());
    const double delta_phi = reco::deltaPhi(track_phi, (ebc->position().phi()));
    if (fabs(delta_eta)<dEtaTkBC_ && fabs(delta_phi)<dPhiTkBC_){
	    if (fabs(min_eta)>fabs(delta_eta) && fabs(min_phi)>fabs(delta_phi)){//take the closest to track BC
        min_eta = delta_eta;
        min_phi = delta_phi;
        closest_bc = bc->second;
        //TODO check if min_eta>delta_eta but min_phi<delta_phi
	    }
    }
  }

  if (min_eta < 999.){
    closestBC = closest_bc;
    return true;
  } else
    return false;
}






//check track open angle of phi at vertex
bool ConversionProducer::checkPhi(const edm::RefToBase<reco::Track>& tk_l, const edm::RefToBase<reco::Track>& tk_r,
                                  const TrackerGeometry* trackerGeom, const MagneticField* magField,
                                  const reco::Vertex& vtx){
  if (!allowDeltaPhi_)
    return true;
  //if track has innermost momentum, check with innermost phi
  //if track also has valid vertex, propagate to vertex then calculate phi there
  //if track has no innermost momentum, just return true, because track->phi() makes no sense
  if (tk_l->extra().isNonnull() && tk_r->extra().isNonnull()){
    double iphi1 = tk_l->innerMomentum().phi(), iphi2 = tk_r->innerMomentum().phi();
    if (vtx.isValid()){
	    PropagatorWithMaterial propag( anyDirection, 0.000511, magField );
	    
	    double recoPhoR = vtx.position().Rho();
	    Surface::RotationType rot;
	    ReferenceCountingPointer<BoundCylinder>  theBarrel_(new BoundCylinder(recoPhoR, Surface::PositionType(0,0,0), rot,
                                                                 new SimpleCylinderBounds( recoPhoR-0.001, recoPhoR+0.001, 
                                                                 -fabs(vtx.position().z()), fabs(vtx.position().z()))));
	    ReferenceCountingPointer<BoundDisk>      theDisk_(
                                                        new BoundDisk( Surface::PositionType( 0, 0, vtx.position().z()), rot,
                                                                       new SimpleDiskBounds( 0, recoPhoR, -0.001, 0.001)));

	    const TrajectoryStateOnSurface myTSOS1 = trajectoryStateTransform::innerStateOnSurface(*tk_l, *trackerGeom, magField);
	    const TrajectoryStateOnSurface myTSOS2 = trajectoryStateTransform::innerStateOnSurface(*tk_r, *trackerGeom, magField);
	    TrajectoryStateOnSurface  stateAtVtx1, stateAtVtx2;
	    stateAtVtx1 = propag.propagate(myTSOS1, *theBarrel_);
	    if (!stateAtVtx1.isValid() ) {
        stateAtVtx1 = propag.propagate(myTSOS1, *theDisk_);
	    }
	    if (stateAtVtx1.isValid()){
        iphi1 = stateAtVtx1.globalDirection().phi();
	    }
	    stateAtVtx2 = propag.propagate(myTSOS2, *theBarrel_);
	    if (!stateAtVtx2.isValid() ) {
        stateAtVtx2 = propag.propagate(myTSOS2, *theDisk_);
	    }
	    if (stateAtVtx2.isValid()){
        iphi2 = stateAtVtx2.globalDirection().phi();
	    }
    }
    const double dPhi = reco::deltaPhi(iphi1, iphi2);
    return (fabs(dPhi) < deltaPhi_);
  } else {
    return true;
  }
}

bool ConversionProducer::preselectTrackPair(const reco::TransientTrack &ttk_l, const reco::TransientTrack &ttk_r,
                                            double& appDist) {
  

  double dCotTheta =  1./tan(ttk_l.track().innerMomentum().theta()) - 1./tan(ttk_r.track().innerMomentum().theta());
  if (allowDeltaCot_ && (std::abs(dCotTheta) > deltaCotTheta_)) {
    return false;
  }
  
  //non-conversion hypothesis, reject prompt track pairs
  ClosestApproachInRPhi closest;
  closest.calculate(ttk_l.innermostMeasurementState(),ttk_r.innermostMeasurementState());
  if (!closest.status()) {
    return false;
  }
  
  if (closest.crossingPoint().perp() < r_cut) {
    return false;
  }

  
  //compute tangent point btw tracks (conversion hypothesis)
  TangentApproachInRPhi tangent;
  tangent.calculate(ttk_l.innermostMeasurementState(),ttk_r.innermostMeasurementState());
  if (!tangent.status()) {
    return false;
  }
  
  GlobalPoint tangentPoint = tangent.crossingPoint();
  double rho = tangentPoint.perp();
  
  //reject candidates well outside of tracker bounds
  if (rho > maxTrackRho_) {
    return false;
  }
  
  if (std::abs(tangentPoint.z()) > maxTrackZ_) {
    return false;
  }
  
  std::pair<GlobalTrajectoryParameters,GlobalTrajectoryParameters> trajs = tangent.trajectoryParameters();
  
  //very large separation in z, no hope
  if (std::abs(trajs.first.position().z() - trajs.second.position().z()) > dzCut_) {
    return false;
  }
  
  
  float minApproach = tangent.perpdist();
  appDist = minApproach;
  
  if (allowMinApproach_ && (minApproach < minApproachLow_ || minApproach > minApproachHigh_) ) {
    return false;
  }
  
  return true;
  
  
}

bool ConversionProducer::checkTrackPair(const std::pair<edm::RefToBase<reco::Track>, reco::CaloClusterPtr>& ll, 
                                        const std::pair<edm::RefToBase<reco::Track>, reco::CaloClusterPtr>& rr){

  const reco::CaloClusterPtr& bc_l = ll.second;//can be null, so check isNonnull()
  const reco::CaloClusterPtr& bc_r = rr.second;
    
  //The cuts should be ordered by considering if takes time and if cuts off many fakes
  if (allowTrackBC_){
    //check energy of BC
    double total_e_bc = 0;
    if (bc_l.isNonnull()) total_e_bc += bc_l->energy();
    if (rightBC_) 
	    if (bc_r.isNonnull())
        total_e_bc += bc_r->energy();

    if (total_e_bc < energyTotalBC_) return false;
  }

  return true;
}



//because reco::vertex uses track ref, so have to keep them
bool ConversionProducer::checkVertex(const reco::TransientTrack &ttk_l, const reco::TransientTrack &ttk_r, 
                                     const MagneticField* magField,
                                     reco::Vertex& the_vertex){
  bool found = false;

  std::vector<reco::TransientTrack>  pair;
  pair.push_back(ttk_l);
  pair.push_back(ttk_r);
   
  found = theVertexFinder_->run(pair, the_vertex);



  return found;
}



double ConversionProducer::etaTransformation(  float EtaParticle , float Zvertex)  {

  //---Definitions
  const float PI    = 3.1415927;

  //---Definitions for ECAL
  const float R_ECAL           = 136.5;
  const float Z_Endcap         = 328.0;
  const float etaBarrelEndcap  = 1.479; 
   
  //---ETA correction

  float Theta = 0.0  ; 
  float ZEcal = R_ECAL*sinh(EtaParticle)+Zvertex;

  if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
  if(Theta<0.0) Theta = Theta+PI ;
  double ETA = - log(tan(0.5*Theta));
         
  if( fabs(ETA) > etaBarrelEndcap )
    {
      float Zend = Z_Endcap ;
      if(EtaParticle<0.0 )  Zend = -Zend ;
      float Zlen = Zend - Zvertex ;
      float RR = Zlen/sinh(EtaParticle); 
      Theta = atan(RR/Zend);
      if(Theta<0.0) Theta = Theta+PI ;
      ETA = - log(tan(0.5*Theta));		      
    } 
  //---Return the result
  return ETA;
  //---end
}

