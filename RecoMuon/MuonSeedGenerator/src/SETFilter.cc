/** \class SETFilter
   I. Bloch, E. James, S. Stoynev
 */
#include "RecoMuon/MuonSeedGenerator/src/SETFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// FIXME: remove this
#include "FWCore/Framework/interface/Event.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

#include <iostream>
#include <fstream>

// there is an existing sorter somewhere in the CMSSW code (I think) - delete that
struct sorter {
  //bigger first!
  bool operator() (TransientTrackingRecHit::ConstRecHitPointer hit_1,
                   TransientTrackingRecHit::ConstRecHitPointer hit_2) const {
    if(hit_1->det()->subDetector() != GeomDetEnumerators::CSC ||
       hit_2->det()->subDetector() != GeomDetEnumerators::CSC){
      // this is a piculiar "fix" for CSCs
      return (hit_1->globalPosition().mag2()>hit_2->globalPosition().mag2());
    }
    else{
      return (fabs(hit_1->globalPosition().z())>fabs(hit_2->globalPosition().z()));
    }
  }
} const sortRadius;// bigger first



using namespace edm;
using namespace std;

SETFilter::SETFilter(const ParameterSet& par,
		     const MuonServiceProxy* service)
  :theService(service)//,
 //theOverlappingChambersFlag(true)
{
  thePropagatorName = par.getParameter<string>("Propagator");
  useSegmentsInTrajectory = par.getUntrackedParameter<bool>("UseSegmentsInTrajectory");
}

SETFilter::~SETFilter(){

  LogTrace("Muon|RecoMuon|SETFilter")
    <<"SETFilter destructor called"<<endl;
  
}

void SETFilter::setEvent(const Event& event){
}

void SETFilter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  
  theLastUpdatedTSOS =  theLastButOneUpdatedTSOS = TrajectoryStateOnSurface();

  theDetLayers.clear();
}


const Propagator* SETFilter::propagator() const {
  return &*theService->propagator(thePropagatorName);
}


void SETFilter::incrementChamberCounters(const DetLayer *layer){

  if(layer->subDetector()==GeomDetEnumerators::DT) dtChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::CSC) cscChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::RPCBarrel || layer->subDetector()==GeomDetEnumerators::RPCEndcap) rpcChambers++; 
  else 
    LogError("Muon|RecoMuon|SETFilter")
      << "Unrecognized module type in incrementChamberCounters";
  // FIXME:
  //   << layer->module() << " " <<layer->Part() << endl;
  
  totalChambers++;
}

//---- the SET FW-fitter within a cluster 
bool SETFilter::fwfit_SET(std::vector < SeedCandidate> & validSegmentsSet_in,
				   std::vector < SeedCandidate> & validSegmentsSet_out){
  // this is the SET algorithm fit
  validSegmentsSet_out.clear();

  //---- It is supposed to be called within a loop over "segment clusters"; 
  //---- so std::vector < SeedCandidate> consists of "valid" combinations (sets) within a "cluster"
  //---- "seed" above has nothing to do with the "Seed" in the STA code
 
  // a trajectory is not really build but a TSOS is build and is checked (below)
  bool validStep = true;
  std::vector <double> chi2AllCombinations(validSegmentsSet_in.size());
  std::vector < TrajectoryStateOnSurface > lastUpdatedTSOS_Vect(validSegmentsSet_in.size());
  // loop over all valid sets
  for(unsigned int iSet = 0; iSet<validSegmentsSet_in.size(); ++iSet){
    //std::cout<<"   iSET = "<<iSet<<std::endl;
    //---- start fit from the origin
    CLHEP::Hep3Vector origin (0.,0.,0.);
    Trajectory::DataContainer trajectoryMeasurementsInTheSet_tmp;
    //---- Find minimum chi2 (corresponding to a specific 3D-momentum)    
    chi2AllCombinations[iSet]  = findMinChi2(iSet, origin, validSegmentsSet_in[iSet], lastUpdatedTSOS_Vect,
                                             trajectoryMeasurementsInTheSet_tmp);
  }
  //---- Find the best muon candidate (min chi2) in the cluster; find more candidates?
  std::vector < double >::iterator itMin = min_element(chi2AllCombinations.begin(),chi2AllCombinations.end());

  int positionMin = itMin - chi2AllCombinations.begin();

  // "the best" set; have to find reasonable conditions to include more than one set   
  validSegmentsSet_out.push_back(validSegmentsSet_in[positionMin]);

  return validStep;
}

//---- the SET FW-fitter
bool SETFilter::buildTrajectoryMeasurements(SeedCandidate * finalMuon, Trajectory::DataContainer & finalCandidate){
  // this is the SET algorithm fit
  bool validTrajectory = true;
  // reset the fitter 
  reset(); // the layer counters
  finalCandidate.clear();

  //---- Check if (only last?) TSOS is valid and build a trajectory (for the backward filter) 

  if(finalMuon->trajectoryMeasurementsInTheSet.size() &&
     finalMuon->trajectoryMeasurementsInTheSet.back().forwardPredictedState().isValid()){
    // loop over all measurements in the set
    for(unsigned int iMeas =0; iMeas<finalMuon->trajectoryMeasurementsInTheSet.size();++iMeas){
      // strore the measurements 
      finalCandidate.push_back(finalMuon->trajectoryMeasurementsInTheSet[iMeas]);
      const DetLayer *layer = finalMuon->trajectoryMeasurementsInTheSet[iMeas].layer();

      incrementChamberCounters(layer);

      theDetLayers.push_back(layer);

    }
    theLastUpdatedTSOS = finalMuon->trajectoryMeasurementsInTheSet.at(finalMuon->trajectoryMeasurementsInTheSet.size()-1).forwardPredictedState();
    //std::cout<<"  THE OUTPUT FROM FW FILTER: |P| = "<<finalMuon->momentum.mag()<<
    //" theta = "<<finalMuon->momentum.theta()<<" phi = "<<finalMuon->momentum.phi()<<std::endl;
  }
  else{
    validTrajectory = false;
    //std::cout<<" TSOS not valid; no trajectory build"<<std::endl;
  }
  return validTrajectory;
}

//
bool SETFilter::transform(Trajectory::DataContainer &measurements_segments, 
			  TransientTrackingRecHit::ConstRecHitContainer & hitContainer, 
			  TrajectoryStateOnSurface & firstTSOS){
  // transforms "segment trajectory" to "rechit container"
  //sort(measurements_segments.begin(),measurements_segments.end(),sortRadius);
  bool success = true;
  // loop over all segments in the trajectory
  for(int iMeas = measurements_segments.size() - 1; iMeas>-1;--iMeas){
    TransientTrackingRecHit ::ConstRecHitContainer sortedHits;
    // loop over the rechits contained in the segments
    for(unsigned int jMeas = 0; jMeas<measurements_segments[iMeas].recHit()->transientHits().size();++jMeas){
      if(measurements_segments[iMeas].recHit()->transientHits().at(jMeas)->transientHits().size()>1){
	// loop over the rechits contained in the rechits contained in the segments (OK, OK - this is for DT only;
	// the convention there is a bit different from the CSCs)
	for(unsigned int kMeas = 0;kMeas<measurements_segments[iMeas].recHit()->transientHits().at(jMeas)->transientHits().size();
	    ++kMeas){
	  sortedHits.push_back( 
			       measurements_segments[iMeas].recHit()->transientHits().at(jMeas)->transientHits().at(kMeas));
	}
      }
      else{
        sortedHits = measurements_segments[iMeas].recHit()->transientHits();
      }
    }
    // sort the rechits by radius (or z) and put them in a container
    sort(sortedHits.begin(),sortedHits.end(),sortRadius);
    hitContainer.insert(hitContainer.end(),sortedHits.begin(),sortedHits.end());    
  }

  // this is the last segment state
  FreeTrajectoryState ftsStart = *(measurements_segments.at(measurements_segments.size()-1).forwardPredictedState().freeState());

  // this is the last (from the IP) rechit
  TransientTrackingRecHit::ConstRecHitPointer muonRecHit =  hitContainer[0];
  DetId detId_last = hitContainer[0]->hit()->geographicalId();
  const GeomDet* layer_last = theService->trackingGeometry()->idToDet(detId_last);

  // get the last rechit TSOS
  TrajectoryStateOnSurface tSOSDest = propagator()->propagate(ftsStart, layer_last->surface());
  firstTSOS = tSOSDest;
  // ftsStart should be at the last rechit surface
  if (!tSOSDest.isValid()){
    success = false;
    //     ftsStart = *tSOSDest.freeState();
  }
  return success; 
}

bool SETFilter::transformLight(Trajectory::DataContainer &measurements_segments,
			       TransientTrackingRecHit::ConstRecHitContainer & hitContainer,
			       TrajectoryStateOnSurface & firstTSOS){
  // transforms "segment trajectory" to "segment container"

  bool success = true;
  // loop over all segments in the trajectory
  if(useSegmentsInTrajectory){// if segments the "backword fit" (rechits)
                              // performed later is actually a forward one (?!) 
    for(unsigned int iMeas = 0; iMeas<measurements_segments.size();++iMeas){
      hitContainer.push_back(measurements_segments[iMeas].recHit());
    }
  }
  else{
    for(int iMeas = measurements_segments.size() - 1; iMeas>-1;--iMeas){
      hitContainer.push_back(measurements_segments[iMeas].recHit());
    }

  }
  // this is the last segment state
  firstTSOS = measurements_segments.at(0).forwardPredictedState();
  return success;
}


double SETFilter::findChi2(double pX, double pY, double pZ,
			   const CLHEP::Hep3Vector& r3T,
			   SeedCandidate & muonCandidate,
			   TrajectoryStateOnSurface  &lastUpdatedTSOS,
			   Trajectory::DataContainer & trajectoryMeasurementsInTheSet,
			   bool detailedOutput){
  //---- actual chi2 calculations; only the measurement error is taken into accout!
  //---- chi2 is to compare an extrapolated point to various measurements so
  //---- the extrapolation error is not an issue (is it?)

  if(detailedOutput){
    trajectoryMeasurementsInTheSet.clear();
  }

  int charge =  muonCandidate.charge;
  GlobalVector p3GV(pX,pY,pZ);
  GlobalPoint r3GP(r3T.x(), r3T.y(), r3T.z());
  //---- how to disable error propagation?
  // VI: just not set it!
  FreeTrajectoryState ftsStart(r3GP, p3GV, charge, &*(theService->magneticField()));
  // VI let's be backward compatible...
  if(detailedOutput) {
    AlgebraicSymMatrix55 cov; cov*=1e-20;
    ftsStart.setCurvilinearError(cov);
  }
  TrajectoryStateOnSurface tSOSDest;
    
  double chi2_loc = 0.;
  for(unsigned int iMeas = 0; iMeas <muonCandidate.theSet.size(); ++iMeas){
    MuonTransientTrackingRecHit::MuonRecHitPointer muonRecHit =  muonCandidate.theSet[iMeas];
    DetId detId = muonRecHit->hit()->geographicalId();
    const GeomDet* layer = theService->trackingGeometry()->idToDet(detId);

    //---- propagate the muon starting from position r3T and momentum p3T 

    //    bool radX0CorrectionMode_ = false; // not needed here?
    //if (radX0CorrectionMode_ ){
    //} else {

    tSOSDest = propagator()->propagate(ftsStart, layer->surface());
    lastUpdatedTSOS = tSOSDest;
    if (tSOSDest.isValid()){
      //---- start next step ("adding" measurement) from the last TSOS
      ftsStart = *tSOSDest.freeState();
    } else{
      //std::cout<<"... not valid TSOS"<<std::endl;
      chi2_loc = 9999999999.;
      break;
    }
    //}

    LocalPoint locHitPos = muonRecHit->localPosition();
    LocalError locHitErr = muonRecHit->localPositionError();
    const GlobalPoint globPropPos = ftsStart.position();
    LocalPoint locPropPos = layer->toLocal(globPropPos);

    //
    //---- chi2 calculated in local system; correlation taken into accont
    CLHEP::HepMatrix dist(1,2);//, distT(2,1);
    double  chi2_intermed = -9;
    int ierr = 0;
    dist(1,1) = locPropPos.x() - locHitPos.x();
    dist(1,2) = locPropPos.y() - locHitPos.y();
    CLHEP::HepMatrix IC(2,2);
    IC(1,1) = locHitErr.xx();
    IC(2,1) = locHitErr.xy();
    IC(2,2) = locHitErr.yy();
    IC(1,2) = IC(2,1);

    //---- Special care is needed for "1-dim measurements" (RPCs, outer DT(?))
    if(4!=muonRecHit->hit()->dimension()){
      for(int iE = 1;iE<3;++iE){
	// this is bellow is a DT fix; hopefully it will not be needed
        if ( fabs(IC(iE,iE)) < 0.00000001){ 
          IC(iE,iE) = 62500./12.;// error squared - ( 250 cm /sqrt(12) )^2; large 
        }
      }
    }
    //---- Invert covariance matrix
    IC.invert(ierr);
    //if (ierr != 0) {
    //std::cout << "failed to invert covariance matrix (2x2) =\n" << IC << std::endl;;
    //}
    chi2_intermed = pow(dist(1,1),2)*IC(1,1) + 2.*dist(1,1)*dist(1,2)*IC(1,2) + pow(dist(1,2),2)*IC(2,2);
    if(chi2_intermed<0){// should we check?
       chi2_intermed = 9999999999.;
    }
    chi2_loc += chi2_intermed;

    // that's for the last call; we don't need to construct a TrajectoryMeasurement at every chi2 step
    if(detailedOutput){
      DetId detId = muonRecHit->hit()->geographicalId();
      const DetLayer *layer = theService->detLayerGeometry()->idToLayer( detId);
      //std::cout<<"    seg pos in traj : "<<lastUpdatedTSOS.globalPosition()<<std::endl;
      // put the measurement into the set
      // VI set the error as the fit needs it... (it is nonsense anyhow...)   
      // (do it on the tsos)
      /*
      if (!lastUpdatedTSOS.hasError()){
	AlgebraicSymMatrix55 cov; cov*=1e6;
	lastUpdatedTSOS.freeTrajectoryState().setCurvilinearError(cov);
      }
      */
      trajectoryMeasurementsInTheSet.push_back( TrajectoryMeasurement
						( lastUpdatedTSOS,
						  muonRecHit,
						  chi2_intermed,
						  layer ) );
    }
  }
  return chi2_loc;
}

double SETFilter::findMinChi2(unsigned int iSet, const CLHEP::Hep3Vector& r3T,
                                         SeedCandidate & muonCandidate,
                                         std::vector < TrajectoryStateOnSurface > &lastUpdatedTSOS_Vect,// delete 
                                         Trajectory::DataContainer & trajectoryMeasurementsInTheSet){
  // a chi2 minimization procedure 

  //---- Which three variables to use? 
  //---- (1/|P|, theta, phi) ? in that case many sin() and cos() operations :-/
  //---- maybe vary directly sin() and cos()?
  bool detailedOutput = false;

  double cosTheta = muonCandidate.momentum.cosTheta();
  double theta = acos(cosTheta);
  double phi = muonCandidate.momentum.phi();
  double pMag = muonCandidate.momentum.mag();

  double minChi2 = -999.;
  TrajectoryStateOnSurface  lastUpdatedTSOS;

  //---- Fit Parameters

  if(pMag<5.){// hardcoded - remove it! same in SETSeedFinder
    pMag = 5.;// GeV
  }
  //---- This offset helps the minimization to go faster (in the specific case)
  pMag *=1.2;
  double invP = 1./pMag;
  //std::cout<<"    INIT pMag = "<<pMag<<" invP = "<<invP<<" theta = "<<theta<<" phi = "<<phi<<std::endl;

  //---- apply downhill SIMPLEX minimization (also "amoeba" method; thus the "feet" below are  amoeba's feet)

  //std::cout<<"    SIMPLEX minimization"<<std::endl;
  //---- parameters ; the should be hardcoded   
  const double reflect = 1;
  const double expand = -0.5;
  const double contract = 0.25;

  const int nDim = 3; // invP, theta, phi
  //---- Now choose nDim + 1 points
  std::vector <CLHEP::Hep3Vector> feet(nDim+1);
  std::vector <double> chi2Feet(nDim+1);
  std::vector <TrajectoryStateOnSurface*> lastUpdatedTSOS_pointer(nDim+1);// probably not needed; to be deleted
   
  //---- The minimization procedure strats from these nDim+1 points (feet)

  CLHEP::Hep3Vector foot1(invP, theta, phi);// well obviuosly it is not a real Hep3Vector; better change it to a simple vector
  feet[0] = foot1;
  chi2Feet[0] = chi2AtSpecificStep(feet[0], r3T, muonCandidate, lastUpdatedTSOS,
                                       trajectoryMeasurementsInTheSet, detailedOutput);
  lastUpdatedTSOS_pointer[0] = &lastUpdatedTSOS;

  std::vector <CLHEP::Hep3Vector> morePoints =
    find3MoreStartingPoints(feet[0], r3T, muonCandidate);
  feet[1] = morePoints[0];
  feet[2] = morePoints[1];
  feet[3] = morePoints[2];

  //--- SIMPLEX initial step(s)
  for(unsigned int iFoot = 1; iFoot<feet.size();++iFoot){

    chi2Feet[iFoot] = chi2AtSpecificStep(feet[iFoot], r3T, muonCandidate, lastUpdatedTSOS,
                                         trajectoryMeasurementsInTheSet, detailedOutput);
    lastUpdatedTSOS_pointer[iFoot] = &lastUpdatedTSOS;
  }

  unsigned int high, second_high, low;  
  //const unsigned int iterations = 1;//---- to be replaced by something better
  int iCalls = 0;
  //for(unsigned int iIt = 0;iIt<iterations;++iIt){
  while(iCalls<3.){
    //---- SIMPLEX step 1
    pickElements(chi2Feet, high, second_high, low);
    ++iCalls;
    feet[high] = reflectFoot(feet, high, reflect );
    chi2Feet[high] = chi2AtSpecificStep(feet[high], r3T, muonCandidate, lastUpdatedTSOS,
                                        trajectoryMeasurementsInTheSet, detailedOutput);
    lastUpdatedTSOS_pointer[high] = &lastUpdatedTSOS;
    //---- SIMPLEX step 2.1
    if(chi2Feet[high] <chi2Feet[low]){
      ++iCalls; 
      feet[high] = reflectFoot(feet, high, expand);
      chi2Feet[high] = chi2AtSpecificStep(feet[high], r3T, muonCandidate, lastUpdatedTSOS,
                                          trajectoryMeasurementsInTheSet, detailedOutput);
      lastUpdatedTSOS_pointer[high] = &lastUpdatedTSOS;
    }
    //---- SIMPLEX step 2.2
    else if( chi2Feet[high] > chi2Feet[second_high]){
      double worstChi2 = chi2Feet[high];
      ++iCalls;
      feet[high] =  reflectFoot(feet, high, contract);
      chi2Feet[high] = chi2AtSpecificStep(feet[high], r3T, muonCandidate, lastUpdatedTSOS,
                                          trajectoryMeasurementsInTheSet, detailedOutput);
      lastUpdatedTSOS_pointer[high] = &lastUpdatedTSOS;
      //---- SIMPLEX step 2.2.1
      if(chi2Feet[high] <worstChi2){
        nDimContract(feet, low);
        for(unsigned int iFoot = 0; iFoot<feet.size();++iFoot){
	  ++iCalls; 
          chi2Feet[iFoot] = chi2AtSpecificStep(feet[iFoot], r3T, muonCandidate, lastUpdatedTSOS,
                                               trajectoryMeasurementsInTheSet, detailedOutput);
          lastUpdatedTSOS_pointer[iFoot] = &lastUpdatedTSOS;
        }
	--iCalls;// one of the above is not changed
      }
    }
  }
  //---- Here the SIMPLEX minimization ends

  // this is the minimum found
  int bestFitElement = min_element(chi2Feet.begin(),chi2Feet.end()) - chi2Feet.begin();

  //---- repeat to get the trajectoryMeasurementsInTheSet (optimize?)
  detailedOutput = true;
  chi2Feet[bestFitElement] = chi2AtSpecificStep(feet[bestFitElement], r3T, muonCandidate, lastUpdatedTSOS,
                                                trajectoryMeasurementsInTheSet, detailedOutput);
  minChi2 = chi2Feet[bestFitElement];

  double pMag_updated = 1./feet[bestFitElement].x();
  double sin_theta = sin(feet[bestFitElement].y());
  double cos_theta = cos(feet[bestFitElement].y());
  double sin_phi = sin(feet[bestFitElement].z());
  double cos_phi = cos(feet[bestFitElement].z());

  double best_pX = pMag_updated*sin_theta*cos_phi;
  double best_pY = pMag_updated*sin_theta*sin_phi;
  double best_pZ = pMag_updated*cos_theta;
  CLHEP::Hep3Vector bestP(best_pX, best_pY, best_pZ);
  //---- update the best momentum estimate  
  //if(minChi2<999999. && pMag_updated>0.){//fit failed?! check
  muonCandidate.momentum = bestP;
  //}
  //---- update the trajectory
  muonCandidate.trajectoryMeasurementsInTheSet = trajectoryMeasurementsInTheSet;
  // do we need that?
  lastUpdatedTSOS_Vect[iSet]= *(lastUpdatedTSOS_pointer[bestFitElement]);

  //std::cout<<"   FINAL:  P = "<<muonCandidate.momentum.mag()<<" theta = "<<muonCandidate.momentum.theta()<<
  //" phi = "<<muonCandidate.momentum.phi()<<"   chi = "<<chi2Feet[bestFitElement]<<std::endl;
  return minChi2;
}

double SETFilter::
chi2AtSpecificStep(CLHEP::Hep3Vector &foot,
                   const CLHEP::Hep3Vector& r3T,
                   SeedCandidate & muonCandidate,
                   TrajectoryStateOnSurface  &lastUpdatedTSOS,
                   Trajectory::DataContainer & trajectoryMeasurementsInTheSet,
                   bool detailedOutput){
  // specific input parameters - find chi2
  double chi2 = 999999999999.;
  if(foot.x()>0){ // this is |P|; maybe return a flag too?
    double pMag_updated = 1./foot.x();
    double sin_theta = sin(foot.y());
    double cos_theta = cos(foot.y());
    double sin_phi = sin(foot.z());
    double cos_phi = cos(foot.z());
      
    double pX = pMag_updated*sin_theta*cos_phi;
    double pY = pMag_updated*sin_theta*sin_phi;
    double pZ = pMag_updated*cos_theta;
    chi2 = findChi2(pX, pY, pZ, r3T, 
                    muonCandidate, lastUpdatedTSOS,
                    trajectoryMeasurementsInTheSet, detailedOutput);
  }
  return chi2;
}   

std::vector <CLHEP::Hep3Vector> SETFilter::
find3MoreStartingPoints(CLHEP::Hep3Vector &key_foot,
                   const CLHEP::Hep3Vector& r3T,
                   SeedCandidate & muonCandidate){
  // SIMPLEX uses nDim + 1 starting points; 
  // so here we need 3 more (one we already have)
  std::vector <CLHEP::Hep3Vector> morePoints;// again - CLHEP::Hep3Vector is not a good choice here
  double invP = key_foot.x();
  double theta = key_foot.y();
  double phi = key_foot.z();


  double deltaPhi_init = 0.005;
  double deltaTheta_init = 0.005;
  //double deltaInvP_init = 1.1 * invP;
  double deltaInvP_init = 0.1 * invP;
  //deltaInvP_init = 0.5 * invP;

  // try to chose better point to start with
  bool optimized = true;
  if(optimized){
    //---- Find a minimum chi2 for every variable (others are fixed) by supposing chi2 is a parabola
    //---- Then these points ("minima") are probably better starting points for the real minimization

    TrajectoryStateOnSurface  lastUpdatedTSOS;// fake here
    Trajectory::DataContainer trajectoryMeasurementsInTheSet;// fake here
    bool detailedOutput = false;// fake here


    std::vector < double > pInv_init(3);
    std::vector < double > theta_init(3);
    std::vector < double > phi_init(3);
    std::vector < double > chi2_init(3);

    pInv_init[0] = invP - deltaInvP_init;
    pInv_init[1] = invP;
    pInv_init[2] = invP + deltaInvP_init;
    //pInv_init[2] = invP +  0.1 * invP;

    theta_init[0] = theta-deltaTheta_init;
    theta_init[1] = theta;
    theta_init[2] = theta+deltaTheta_init;

    phi_init[0] = phi-deltaPhi_init;
    phi_init[1] = phi;
    phi_init[2] = phi+deltaPhi_init;

    double sin_theta_nom = sin(theta_init[1]);
    double cos_theta_nom = cos(theta_init[1]);
    double sin_phi_nom = sin(phi_init[1]);
    double cos_phi_nom = cos(phi_init[1]);
    double pMag_updated_nom =  1./pInv_init[1];

    //--- invP
    for(int i=0;i<3;++i){
      double pMag_updated = 1./pInv_init[i];
      double pX = pMag_updated*sin_theta_nom*cos_phi_nom;
      double pY = pMag_updated*sin_theta_nom*sin_phi_nom;
      double pZ = pMag_updated*cos_theta_nom;
      chi2_init[i] = findChi2(pX, pY, pZ, r3T,
                         muonCandidate, lastUpdatedTSOS,
                         trajectoryMeasurementsInTheSet, detailedOutput);
    }
    std::pair <double,double> result_pInv =
      findParabolaMinimum(pInv_init, chi2_init);

    //---- theta
    for(int i=0;i<3;++i){
      double sin_theta = sin(theta_init[i]);
      double cos_theta = cos(theta_init[i]);
      double pX = pMag_updated_nom*sin_theta*cos_phi_nom;
      double pY = pMag_updated_nom*sin_theta*sin_phi_nom;
      double pZ = pMag_updated_nom*cos_theta;
        chi2_init[i] =  findChi2(pX, pY, pZ, r3T,
                         muonCandidate, lastUpdatedTSOS,
                         trajectoryMeasurementsInTheSet, detailedOutput);
    }
    std::pair <double,double> result_theta =
      findParabolaMinimum(theta_init, chi2_init);

    //---- phi
    for(int i=0;i<3;++i){
      double sin_phi = sin(phi_init[i]);
      double cos_phi = cos(phi_init[i]);
      double pX = pMag_updated_nom*sin_theta_nom*cos_phi;
      double pY = pMag_updated_nom*sin_theta_nom*sin_phi;
      double pZ = pMag_updated_nom*cos_theta_nom;
      chi2_init[i] =  findChi2(pX, pY, pZ, r3T,
                         muonCandidate, lastUpdatedTSOS,
                         trajectoryMeasurementsInTheSet, detailedOutput);
    }
    std::pair <double,double> result_phi =
      findParabolaMinimum(phi_init, chi2_init);
    // should we use that?
    double newPhi = result_phi.first;
    if(fabs(newPhi - phi)<0.02){// too close?
      newPhi = phi + 0.02;
    }
    CLHEP::Hep3Vector foot2(invP, theta, result_phi.first);
    CLHEP::Hep3Vector foot3(invP, result_theta.first , phi);
    double newInvP = result_pInv.first;
    if(fabs(newInvP - invP)<0.001){//too close
      newInvP = invP + 0.001;
    }
    CLHEP::Hep3Vector foot4(result_pInv.first, theta, phi);
    morePoints.push_back(foot2);
    morePoints.push_back(foot3);
    morePoints.push_back(foot4);
  }
  else{
    // the points
    CLHEP::Hep3Vector foot2(invP, theta, phi-deltaPhi_init);
    CLHEP::Hep3Vector foot3(invP, theta-deltaTheta_init, phi);
    CLHEP::Hep3Vector foot4(invP-deltaInvP_init, theta, phi);
    morePoints.push_back(foot2);
    morePoints.push_back(foot3);
    morePoints.push_back(foot4);
  }
  return morePoints;
}

std::pair <double,double> SETFilter::findParabolaMinimum(std::vector <double> &quadratic_var,
                                                                    std::vector <double> &quadratic_chi2){

  // quadratic equation minimization

  double paramAtMin = -99.;
  std::vector <double> quadratic_param(3);

  CLHEP::HepMatrix denominator(3,3);
  CLHEP::HepMatrix enumerator_1(3,3);
  CLHEP::HepMatrix enumerator_2(3,3);
  CLHEP::HepMatrix enumerator_3(3,3);

  for(int iCol=1;iCol<4;++iCol){
    denominator(1,iCol) = 1;
    denominator(2,iCol) = quadratic_var.at(iCol-1);
    denominator(3,iCol) = pow(quadratic_var.at(iCol-1),2);

    enumerator_1(1,iCol) = quadratic_chi2.at(iCol-1);
    enumerator_1(2,iCol) = denominator(2,iCol);
    enumerator_1(3,iCol) = denominator(3,iCol);

    enumerator_2(1,iCol) = denominator(1,iCol);
    enumerator_2(2,iCol) = quadratic_chi2.at(iCol-1);
    enumerator_2(3,iCol) = denominator(3,iCol);

    enumerator_3(1,iCol) = denominator(1,iCol);
    enumerator_3(2,iCol) = denominator(2,iCol);
    enumerator_3(3,iCol) = quadratic_chi2.at(iCol-1);
  }
  const double mult = 5.;// "save" accuracy"; result is independent on "mult"
  denominator *=mult;
  enumerator_1*=mult;
  enumerator_2*=mult;
  enumerator_3*=mult;

  std::vector <CLHEP::HepMatrix> enumerator;
  enumerator.push_back(enumerator_1);
  enumerator.push_back(enumerator_2);
  enumerator.push_back(enumerator_3);

  double determinant = denominator.determinant();
  if(fabs(determinant)>0.00000000001){
    for(int iPar=0;iPar<3;++iPar){
      quadratic_param.at(iPar) = enumerator.at(iPar).determinant()/determinant;
    }
  }
  else{
    //std::cout<<" determinant 0? Check the code..."<<std::endl;
  }
  if(quadratic_param.at(2)>0){
    paramAtMin = - quadratic_param.at(1)/quadratic_param.at(2)/2;
  }
  else {
    //std::cout<<" parabola has a maximum or division by zero... Using initial value. "<<std::endl;
    paramAtMin = quadratic_var.at(1);
  }
  double chi2_min = quadratic_param.at(0) + quadratic_param.at(1)*paramAtMin + quadratic_param.at(2)*pow(paramAtMin,2);
  std::pair <double, double> result;
  result =  std::make_pair ( paramAtMin, chi2_min);
  return result;
}

void SETFilter::pickElements(std::vector <double> &chi2Feet,
                                       unsigned int & high, unsigned int & second_high, unsigned int & low){
  // a SIMPLEX function
  std::vector <double> chi2Feet_tmp = chi2Feet;
  std::vector <double>::iterator minEl = min_element(chi2Feet.begin(), chi2Feet.end());
  std::vector <double>::iterator maxEl = max_element(chi2Feet.begin(), chi2Feet.end());
  high = maxEl - chi2Feet.begin();
  low = minEl - chi2Feet.begin();
  chi2Feet_tmp[high] = chi2Feet_tmp[low]; 
  std::vector <double>::iterator second_maxEl = max_element(chi2Feet_tmp.begin(), chi2Feet_tmp.end());
  second_high = second_maxEl - chi2Feet_tmp.begin();

  return;
}

CLHEP::Hep3Vector SETFilter::reflectFoot(std::vector <CLHEP::Hep3Vector> & feet,
                                                 unsigned int key_foot, double scale ){
  // a SIMPLEX function
  CLHEP::Hep3Vector newPosition(0.,0.,0.);
  if(scale==0.5){
    //std::cout<<" STA muon: scale parameter for simplex method incorrect : "<<scale<<std::endl;
    return newPosition;
  }
  CLHEP::Hep3Vector centroid(0,0,0);
  for(unsigned int iFoot = 0; iFoot<feet.size();++iFoot){
    if(iFoot==key_foot){
      continue;
    }
    centroid += feet[iFoot];
  }
  centroid/=(feet.size()-1);
  CLHEP::Hep3Vector displacement = 2.*(centroid - feet[key_foot]);
  newPosition = feet[key_foot] + scale * displacement;
  return newPosition;
}

void SETFilter::nDimContract(std::vector <CLHEP::Hep3Vector> & feet, unsigned int low){
  for(unsigned int iFoot = 0; iFoot<feet.size();++iFoot){
    // a SIMPLEX function
    if(iFoot==low){
      continue;
    }
    feet[iFoot] +=  feet[low];
    feet[iFoot] /=2.;
  }
  return;
}
