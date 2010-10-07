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
// $Id: TrackerOnlyConversionProducer.cc,v 1.21 2010/06/03 14:34:47 nancy Exp $
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

#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Math/interface/deltaPhi.h"

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

using namespace edm;
using namespace reco;
using namespace std;


TrackerOnlyConversionProducer::TrackerOnlyConversionProducer(const edm::ParameterSet& iConfig)
{
    algoName_ = iConfig.getParameter<std::string>( "AlgorithmName" );

    src_ = iConfig.getParameter<std::vector<edm::InputTag> >("src");

    allowTrackBC_ = iConfig.getParameter<bool>("AllowTrackBC");
    allowD0_ = iConfig.getParameter<bool>("AllowD0");
    allowDeltaPhi_ = iConfig.getParameter<bool>("AllowDeltaPhi");
    allowDeltaCot_ = iConfig.getParameter<bool>("AllowDeltaCot");
    allowMinApproach_ = iConfig.getParameter<bool>("AllowMinApproach");
    allowOppCharge_ = iConfig.getParameter<bool>("AllowOppCharge");

    allowVertex_ = iConfig.getParameter<bool>("AllowVertex");

    halfWayEta_ = iConfig.getParameter<double>("HalfwayEta");//open angle to search track matches with BC

    if (allowD0_)
	d0Cut_ = iConfig.getParameter<double>("d0");
    
    usePvtx_ = iConfig.getParameter<bool>("UsePvtx");//if use primary vertices

    if (usePvtx_){
	vertexProducer_   = iConfig.getParameter<std::string>("primaryVertexProducer");
    }

    if (allowTrackBC_) {
	//Track-cluster matching eta and phi cuts
	dEtaTkBC_ = iConfig.getParameter<double>("dEtaTrackBC");//TODO research on cut endcap/barrel
	dPhiTkBC_ = iConfig.getParameter<double>("dPhiTrackBC");

	bcBarrelCollection_     = iConfig.getParameter<edm::InputTag>("bcBarrelCollection");
	bcEndcapCollection_     = iConfig.getParameter<edm::InputTag>("bcEndcapCollection");

	energyBC_ = iConfig.getParameter<double>("EnergyBC");//BC energy cut
	energyTotalBC_ = iConfig.getParameter<double>("EnergyTotalBC");//BC energy cut

    }

    if (allowVertex_){
	maxDistance_ = iConfig.getParameter<double>("maxDistance");
	maxOfInitialValue_ = iConfig.getParameter<double>("maxOfInitialValue");
	maxNbrOfIterations_ = iConfig.getParameter<int>("maxNbrOfIterations");
    }
    //Track cuts on left right track: at least one leg reaches ECAL
    //Left track: must exist, must reach Ecal and match BC, so loose cut on Chi2 and tight on hits
    //Right track: not necessary to exist (if allowSingleLeg_), not necessary to reach ECAL or match BC, so tight cut on Chi2 and loose on hits
    maxChi2Left_ =  iConfig.getParameter<double>("MaxChi2Left");
    maxChi2Right_ =  iConfig.getParameter<double>("MaxChi2Right");
    minHitsLeft_ = iConfig.getParameter<int>("MinHitsLeft");
    minHitsRight_ = iConfig.getParameter<int>("MinHitsRight");

    //Track Open angle cut on delta cot(theta) and delta phi
    if (allowDeltaCot_)
	deltaCotTheta_ = iConfig.getParameter<double>("DeltaCotTheta");
    if (allowDeltaPhi_)
	deltaPhi_ = iConfig.getParameter<double>("DeltaPhi");
    if (allowMinApproach_){
	minApproachLow_ = iConfig.getParameter<double>("MinApproachLow");
	minApproachHigh_ = iConfig.getParameter<double>("MinApproachHigh");
    }

    // if allow single track collection, by default False
    allowSingleLeg_ = iConfig.getParameter<bool>("AllowSingleLeg");
    rightBC_ = iConfig.getParameter<bool>("AllowRightBC");

    //track inner position dz cut, need RECO
    dzCut_ = iConfig.getParameter<double>("dz");
    //track analytical cross cut
    r_cut = iConfig.getParameter<double>("rCut");
    vtxChi2_ = iConfig.getParameter<double>("vtxChi2");

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

inline bool TrackerOnlyConversionProducer::trackQualityFilter(const edm::Ref<reco::TrackCollection>&  ref, bool isLeft){
    bool pass = true;
    if (isLeft){
	pass = (ref->normalizedChi2() < maxChi2Left_ && ref->found() >= minHitsLeft_);
    } else {
	pass = (ref->normalizedChi2() < maxChi2Right_ && ref->found() >= minHitsRight_);
    }

    return pass;
}

inline bool TrackerOnlyConversionProducer::trackD0Cut(const edm::Ref<reco::TrackCollection>&  ref){
    //NOTE if not allow d0 cut, always true
    return ((!allowD0_) || !(ref->d0()*ref->charge()/ref->d0Error()<d0Cut_));
}

inline bool TrackerOnlyConversionProducer::trackD0Cut(const edm::Ref<reco::TrackCollection>&  ref, const reco::Vertex& the_pvtx){
    //
    return ((!allowD0_) || !(-ref->dxy(the_pvtx.position())*ref->charge()/ref->dxyError()<d0Cut_));
}

double TrackerOnlyConversionProducer::getMinApproach(const TrackRef& ll, const TrackRef& rr, const MagneticField* magField){
    double x_l, x_r, y_l, y_r;

    const double xx_l = ll->innerPosition().x(), yy_l = ll->innerPosition().y(), zz_l = ll->innerPosition().z();
    const double xx_r = rr->innerPosition().x(), yy_r = rr->innerPosition().y(), zz_r = rr->innerPosition().z();
    const double radius_l = ll->innerMomentum().Rho()/(.29979*(magField->inTesla(GlobalPoint(xx_l, yy_l, zz_l)).z()))*100;
    const double radius_r = rr->innerMomentum().Rho()/(.29979*(magField->inTesla(GlobalPoint(xx_r, yy_r, zz_r)).z()))*100;
    getCircleCenter(ll, radius_l, x_l, y_l);
    getCircleCenter(rr, radius_r, x_r, y_r);

    return sqrt((x_l-x_r)*(x_l-x_r)+(y_l-y_r)*(y_l-y_r)) - radius_l - radius_r;
}

bool TrackerOnlyConversionProducer::getTrackImpactPosition(const TrackRef& tk_ref,
	const TrackerGeometry* trackerGeom, const MagneticField* magField,
	math::XYZPoint& ew){

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
    if (stateAtECAL.isValid()){
	ew = stateAtECAL.globalPosition();
	return true;
    }
    else
	return false;
}

bool TrackerOnlyConversionProducer::getMatchedBC(const std::multimap<double, reco::CaloClusterPtr>& bcMap, 
	const math::XYZPoint& trackImpactPosition,
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

bool TrackerOnlyConversionProducer::getMatchedBC(const reco::CaloClusterPtrVector& bcMap,
	const math::XYZPoint& trackImpactPosition,
	reco::CaloClusterPtr& closestBC){
    const double track_eta = trackImpactPosition.eta();
    const double track_phi = trackImpactPosition.phi();

    double min_eta = 999., min_phi = 999.;
    reco::CaloClusterPtr closest_bc;
    for (reco::CaloClusterPtrVector::const_iterator bc = bcMap.begin();
	    bc != bcMap.end(); ++bc){//use eta map to select possible BC collection then loop in
	const reco::CaloClusterPtr& ebc = (*bc);
	const double delta_eta = track_eta-(ebc->position().eta());
	const double delta_phi = reco::deltaPhi(track_phi, (ebc->position().phi()));
	if (fabs(delta_eta)<dEtaTkBC_ && fabs(delta_phi)<dPhiTkBC_){
	    if (fabs(min_eta)>fabs(delta_eta) && fabs(min_phi)>fabs(delta_phi)){//take the closest to track BC
		min_eta = delta_eta;
		min_phi = delta_phi;
		closest_bc = ebc;
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
bool TrackerOnlyConversionProducer::checkPhi(const reco::TrackRef& tk_l, const reco::TrackRef& tk_r,
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
	    TrajectoryStateTransform transformer;
	    double recoPhoR = vtx.position().Rho();
	    Surface::RotationType rot;
	    ReferenceCountingPointer<BoundCylinder>  theBarrel_(new BoundCylinder( Surface::PositionType(0,0,0), rot,
			SimpleCylinderBounds( recoPhoR-0.001, recoPhoR+0.001, -fabs(vtx.position().z()), fabs(vtx.position().z()))));
	    ReferenceCountingPointer<BoundDisk>      theDisk_(
		    new BoundDisk( Surface::PositionType( 0, 0, vtx.position().z()), rot,
			SimpleDiskBounds( 0, recoPhoR, -0.001, 0.001)));

	    const TrajectoryStateOnSurface myTSOS1 = transformer.innerStateOnSurface(*tk_l, *trackerGeom, magField);
	    const TrajectoryStateOnSurface myTSOS2 = transformer.innerStateOnSurface(*tk_r, *trackerGeom, magField);
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

bool TrackerOnlyConversionProducer::checkTrackPair(const std::pair<reco::TrackRef, reco::CaloClusterPtr>& ll, 
	const std::pair<reco::TrackRef, reco::CaloClusterPtr>& rr, 
	const MagneticField* magField,
	double & appDist){

    const reco::TrackRef& tk_l = ll.first;
    const reco::TrackRef& tk_r = rr.first;
    const TransientTrack ttk_l(tk_l, magField);
    const TransientTrack ttk_r(tk_r, magField);
    const reco::CaloClusterPtr& bc_l = ll.second;//can be null, so check isNonnull()
    const reco::CaloClusterPtr& bc_r = rr.second;
    
    //DeltaPhi as preselection cut
    /*
    if (allowDeltaPhi_){
	const double phi_l = tk_l->innerMomentum().phi();
	const double phi_r = tk_r->innerMomentum().phi();
	double dPhi = reco::deltaPhi(phi_l, phi_r);

	if (fabs(dPhi) > deltaPhi_) return false;//Delta Phi cut for pair
    }
    */

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

    if (allowDeltaCot_){
	const double theta_l = tk_l ->innerMomentum().Theta();
	double theta_r = tk_r->innerMomentum().Theta();
	const double dCotTheta =  1./tan(theta_l) - 1./tan(theta_r) ;

	if (fabs(dCotTheta) > deltaCotTheta_) return false;//Delta Cot(Theta) cut for pair
    }

    //check with tracks innermost position z, except TEC
    if (tk_l->extra().isNonnull() && tk_r->extra().isNonnull()){//inner position delta Z cut
	const double inner_z_l = tk_l->innerPosition().z();
	const double inner_z_r = tk_r->innerPosition().z();
	if (fabs(inner_z_l)<120. && fabs(inner_z_r)<120.) {//not using delta z cut in TEC
	    if (fabs(inner_z_l-inner_z_r) > dzCut_)
		return false;
	}
    }

    if (allowMinApproach_){
	const double distance = getMinApproach(tk_l, tk_r, magField);

	if (distance < minApproachLow_ || distance >minApproachHigh_) return false;
	else
	    appDist = distance;
    }

    //check with analytical track cross, if cross too early, it is not conversion
    TwoTrackMinimumDistance md;
    md.calculate  (  ttk_l.initialFreeState(),  ttk_r.initialFreeState() );
    GlobalPoint thecross = md.crossingPoint();
    const double cross_r = sqrt(thecross.x()*thecross.x()+thecross.y()*thecross.y());

    if (cross_r<r_cut) return false;
    
    //TODO INSERT MORE CUTS HERE!

    return true;
}



//because reco::vertex uses track ref, so have to keep them
bool TrackerOnlyConversionProducer::checkVertex(const reco::TrackRef& tk_l, const reco::TrackRef& tk_r, 
	const MagneticField* magField,
	reco::Vertex& the_vertex){
    bool found = false;

    TransientTrack ttk_l(tk_l, magField);
    TransientTrack ttk_r(tk_r, magField);

    float sigma = 0.00000000001;
    float chi = 0.;
    float ndf = 0.;
    float mass = 0.000000511;

    edm::ParameterSet pSet;
    pSet.addParameter<double>("maxDistance", maxDistance_);//0.001
    pSet.addParameter<double>("maxOfInitialValue",maxOfInitialValue_) ;//1.4
    pSet.addParameter<int>("maxNbrOfIterations", maxNbrOfIterations_);//40

    KinematicParticleFactoryFromTransientTrack pFactory;

    vector<RefCountedKinematicParticle> particles;

    particles.push_back(pFactory.particle (ttk_l,mass,chi,ndf,sigma));
    particles.push_back(pFactory.particle (ttk_r,mass,chi,ndf,sigma));

    MultiTrackKinematicConstraint *  constr = new ColinearityKinematicConstraint(ColinearityKinematicConstraint::PhiTheta);

    KinematicConstrainedVertexFitter kcvFitter;
    kcvFitter.setParameters(pSet);
    RefCountedKinematicTree myTree = kcvFitter.fit(particles, constr);
    if( myTree->isValid() ) {
	myTree->movePointerToTheTop();                                                                                
	RefCountedKinematicParticle the_photon = myTree->currentParticle();                                           
	if (the_photon->currentState().isValid()){                                                                    
	    //const ParticleMass photon_mass = the_photon->currentState().mass();                                       
	    RefCountedKinematicVertex gamma_dec_vertex;                                                               
	    gamma_dec_vertex = myTree->currentDecayVertex();                                                          
	    if( gamma_dec_vertex->vertexIsValid() ){                                                                  
		const float chi2Prob = ChiSquaredProbability(gamma_dec_vertex->chiSquared(), gamma_dec_vertex->degreesOfFreedom());
		if (chi2Prob>0.){// no longer cut here, only ask positive probability here 
		    //const math::XYZPoint vtxPos(gamma_dec_vertex->position());                                           
		  the_vertex = *gamma_dec_vertex;
		  found = true;
		}
	    }
	}
    }
    delete constr;                                                                                                    

    return found;
}

//calculate the center of track circle in transverse plane
//muon hypothesis uses AOD and circle based on muon (no brems)
//non-muon hypothesis needs TrackExtra 
inline void TrackerOnlyConversionProducer::getCircleCenter(const reco::TrackRef& tk, const double r, double& x0, double& y0,  bool muon){
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

inline void TrackerOnlyConversionProducer::getCircleCenter(const edm::RefToBase<reco::Track>& tk, const double r, double& x0, double& y0,  bool muon){
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

void TrackerOnlyConversionProducer::buildCollection(edm::Event& iEvent, const edm::EventSetup& iSetup,
	const reco::TrackRefVector& allTracks,
	const std::multimap<double, reco::CaloClusterPtr>& basicClusterPtrs,
	const reco::Vertex& the_pvtx,
	reco::ConversionCollection & outputConvPhotonCollection){

    edm::ESHandle<TrackerGeometry> trackerGeomHandle;
    edm::ESHandle<MagneticField> magFieldHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeomHandle );
    iSetup.get<IdealMagneticFieldRecord>().get( magFieldHandle );

    const TrackerGeometry* trackerGeom = trackerGeomHandle.product();
    const MagneticField* magField = magFieldHandle.product();;

    std::vector<math::XYZPoint> trackImpactPosition;
    trackImpactPosition.reserve(allTracks.size());//track impact position at ECAL
    std::vector<bool> trackValidECAL;//Does this track reach ECAL basic cluster (reach ECAL && match with BC)
    trackValidECAL.assign(allTracks.size(), false);

    std::vector<reco::CaloClusterPtr> trackMatchedBC;
    reco::CaloClusterPtr empty_bc;
    trackMatchedBC.assign(allTracks.size(), empty_bc);//TODO find a better way to avoid copy constructor

    std::vector<int> bcHandleId;//the associated BC handle id, -1 invalid, 0 barrel 1 endcap
    bcHandleId.assign(allTracks.size(), -1);

    std::multimap<double, int> trackInnerEta;//Track innermost state Eta map to TrackRef index, to be used in track pair sorting

    //2 propagate all tracks into ECAL, record its eta and phi
    for (reco::TrackRefVector::const_iterator ref = allTracks.begin(); ref != allTracks.end(); ++ref){
	const TrackRef& tk_ref = *ref;

	//if ( !(trackQualityFilter(tk_ref, true)) ) continue;

	//map TrackRef to Eta
	trackInnerEta.insert(std::make_pair(tk_ref->innerMomentum().eta(), ref-allTracks.begin()));

	if (allowTrackBC_){
	    //check impact position then match with BC
	    math::XYZPoint ew;
	    if ( getTrackImpactPosition(tk_ref, trackerGeom, magField, ew) ){
		trackImpactPosition.push_back(ew);

		reco::CaloClusterPtr closest_bc;//the closest matching BC to track

		if ( getMatchedBC(basicClusterPtrs, ew, closest_bc) ){
		    trackMatchedBC[ref-allTracks.begin()] = closest_bc;
		    trackValidECAL[ref-allTracks.begin()] = true;
		    bcHandleId[ref-allTracks.begin()] = (fabs(closest_bc->position().eta())>1.479)?1:0;
		}
	    } else {
		trackImpactPosition.push_back(ew);
		continue;
	    }

	}
    }
    //3. pair up : to select one track from matched EBC, and select the other track to fit cot theta and phi open angle cut
    //TODO it is k-Closest pair of point problem
    reco::VertexCollection vertexs;
    std::vector<bool> isPaired;
    isPaired.assign(allTracks.size(), false);
    for( reco::TrackRefVector::const_iterator ll = allTracks.begin(); ll != allTracks.end(); ++ ll ) {
	if ((allowTrackBC_ && !trackValidECAL[ll-allTracks.begin()]) //this Left leg should have valid BC
		//|| !(trackD0Cut(*ll)) //d0*charge signicifance for converted photons
		|| isPaired[ll-allTracks.begin()]) //this track should not be used in another pair
	    continue;

	if (the_pvtx.isValid()){
	    if (!(trackD0Cut(*ll, the_pvtx)))
		continue;
	} else {
	    if (!(trackD0Cut(*ll)))
		continue;
	}
	const double left_eta = (*ll)->innerMomentum().eta();
	bool found_right = false;//check if right leg found, if no but allowSingleLeg_, go build a conversion with left leg
	std::vector<int> right_candidates;//store all right legs passed the cut (theta/approach and ref pair)
	std::vector<double> right_candidate_theta, right_candidate_approach;
	reco::VertexCollection vertex_candidates;//store the candiate vertex to candidate right

	//select right leg candidates, which passed the cuts
	////TODO translate it!
	for (std::multimap<double, int>::iterator rr = trackInnerEta.lower_bound(left_eta - halfWayEta_);
		rr != trackInnerEta.upper_bound(left_eta + halfWayEta_); ++rr){//select neighbor tracks by eta
	    //Level 2 loop
	    //TODO find the closest one rather than the first matching
	    const edm::Ref<reco::TrackCollection> & right = allTracks[rr->second];

	    if ( //!(trackQualityFilter(right, false) ) || 
		    isPaired[rr->second] //this track should not be in another pair
		    //|| right == (*ll) //no double counting (dummy if require opposite charge)
		    //|| !(trackD0Cut(right)) //d0*charge significance for converted photons
		    || (allowTrackBC_ && !trackValidECAL[rr->second] && rightBC_) // if right track matches ECAL
		    || (allowOppCharge_ && ( (*ll)->charge()*right->charge() > 0 ) )  ) //opposite charge
		continue;

	    //track pair pass the quality cut
	    if (!( (trackQualityFilter((*ll), true) && trackQualityFilter(right, false))
		|| (trackQualityFilter((*ll), false) && trackQualityFilter(right, true)) ) )
		continue;

	    if (the_pvtx.isValid()){
		if (!(trackD0Cut(right, the_pvtx)))
		    continue;
	    } else {
		if (!(trackD0Cut(right)))
		    continue;
	    }
	    const std::pair<reco::TrackRef, reco::CaloClusterPtr> the_left = std::make_pair<reco::TrackRef, reco::CaloClusterPtr>((*ll), trackMatchedBC[ll-allTracks.begin()]);
	    const std::pair<reco::TrackRef, reco::CaloClusterPtr> the_right = std::make_pair<reco::TrackRef, reco::CaloClusterPtr>(right, trackMatchedBC[rr->second]);

	    double app_distance = -999.;
	    //signature cuts, then check if vertex, then post-selection cuts
	    if ( checkTrackPair(the_left, the_right, magField, app_distance) ){
		reco::Vertex the_vertex;//by default it is invalid
		//if allow vertex and there is a vertex, go vertex finding, otherwise
		if (allowVertex_) {
		    //const bool found_vertex = checkVertex((*ll), right, magField, the_vertex);
		    checkVertex((*ll), right, magField, the_vertex);
		}
		if (checkPhi((*ll), right, trackerGeom, magField, the_vertex)){
		    right_candidates.push_back(rr->second);
		    right_candidate_theta.push_back(right->innerMomentum().Theta());
		    right_candidate_approach.push_back(app_distance);
		    vertex_candidates.push_back(the_vertex);//this vertex can be valid or not
		}
	    }
	}
	//fill collection with all right candidates
	if (!right_candidates.empty()){//find a good right track
	    for (unsigned int ii = 0; ii< right_candidates.size(); ++ii){
		const int right_index = right_candidates[ii];
		edm::Ref<reco::TrackCollection> right = allTracks[right_index];;
		const double min_approach = right_candidate_approach[ii];
		//if all cuts passed, go ahead to make conversion candidates
		std::vector<edm::Ref<reco::TrackCollection> > trackPairRef;
		trackPairRef.push_back((*ll));//left track
		trackPairRef.push_back(right);//right track

		std::vector<math::XYZVector> trackPin;
		std::vector<math::XYZVector> trackPout;
		std::vector<math::XYZPoint> trackInnPos;

		if ((*ll)->extra().isNonnull() && right->extra().isNonnull()){//only available on TrackExtra
		  trackInnPos.push_back(  (*ll)->innerPosition());
		  trackInnPos.push_back(  right->innerPosition());
		  trackPin.push_back((*ll)->innerMomentum());
		  trackPin.push_back(right->innerMomentum());
		  trackPout.push_back((*ll)->outerMomentum());
		  trackPout.push_back(right->outerMomentum());
		}

		reco::CaloClusterPtrVector scPtrVec;
		reco::Vertex  theConversionVertex;//Dummy vertex, validity false by default
		//if using kinematic fit, check with chi2 post cut
		if (allowVertex_) {
		    theConversionVertex = vertex_candidates[ii];
		    if (theConversionVertex.isValid()){
			const float chi2Prob = ChiSquaredProbability(theConversionVertex.chi2(), theConversionVertex.ndof());
			if (chi2Prob<vtxChi2_)
			    continue;
		    }
		}
		std::vector<math::XYZPoint> trkPositionAtEcal;
		std::vector<reco::CaloClusterPtr> matchingBC;

		if (allowTrackBC_){//TODO find out the BC ptrs if not doing matching, otherwise, leave it empty
		    const int lbc_handle = bcHandleId[ll-allTracks.begin()],
			  rbc_handle = bcHandleId[right_index];

		    trkPositionAtEcal.push_back(trackImpactPosition[ll-allTracks.begin()]);//left track
		    if (trackValidECAL[right_index])//second track ECAL position may be invalid
			trkPositionAtEcal.push_back(trackImpactPosition[right_index]);

		    matchingBC.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
		    scPtrVec.push_back(trackMatchedBC[ll-allTracks.begin()]);//left track
		    if (trackValidECAL[right_index]){//second track ECAL position may be invalid
			matchingBC.push_back(trackMatchedBC[right_index]);
			if (!(trackMatchedBC[right_index] == trackMatchedBC[ll-allTracks.begin()])//avoid 1 bc 2 tk
				&& lbc_handle == rbc_handle )//avoid ptr from different collection
			    scPtrVec.push_back(trackMatchedBC[right_index]);
		    }
		}
		const float minAppDist = min_approach;

		reco::Conversion::ConversionAlgorithm algo = reco::Conversion::algoByName(algoName_);
		float dummy=0;
		reco::Conversion  newCandidate(scPtrVec,  trackPairRef, trkPositionAtEcal, theConversionVertex, matchingBC, minAppDist,  trackInnPos, trackPin, trackPout, dummy, algo );
		outputConvPhotonCollection.push_back(newCandidate);

		found_right = true;
		isPaired[ll-allTracks.begin()] = true;//mark this track is used in pair
		isPaired[right_index] = true;
	    }
	}
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

    edm::Handle<reco::VertexCollection> vertexHandle;
    reco::VertexCollection vertexCollection;
    if (usePvtx_){
	iEvent.getByLabel(vertexProducer_, vertexHandle);
	if (!vertexHandle.isValid()) {
	    edm::LogError("TrackerOnlyConversionProducer") << "Error! Can't get the product primary Vertex Collection "<< "\n";
	    usePvtx_ = false;
	}
	if (usePvtx_)
	    vertexCollection = *(vertexHandle.product());
    }

    reco::Vertex the_pvtx;
    //because the priamry vertex is sorted by quality, the first one is the best
    if (!vertexCollection.empty())
      the_pvtx = *(vertexCollection.begin());
    
    reco::TrackRefVector allTracks;
    int total_tracks = 0;
    for (unsigned int ii = 0; ii<trackCollectionHandles.size(); ++ii)
      total_tracks += trackCollectionHandles[ii]->size();
    
    if (total_tracks>150){
	iEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);
	return;
    }
	    
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

    buildCollection( iEvent, iSetup, allTracks, basicClusterPtrs, the_pvtx, outputConvPhotonCollection);//allow empty basicClusterPtrs

    outputConvPhotonCollection_p->assign(outputConvPhotonCollection.begin(), outputConvPhotonCollection.end());
    iEvent.put( outputConvPhotonCollection_p, ConvertedPhotonCollection_);

}



