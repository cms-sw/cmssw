/** \class DirectMuonTrajectoryBuilder
 *  Class which takes a trajectory seed and fit its hits, returning a Trajectory container
 *
 *  \author
 */

#include "RecoMuon/TrackingTools/interface/DirectMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TrackRefitter/interface/SeedTransformer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;


DirectMuonTrajectoryBuilder::DirectMuonTrajectoryBuilder(const ParameterSet& par, 
							 const MuonServiceProxy* service):theService(service){
  
  // The seed transformer (used to refit the seed and get the seed transient state)
  //  ParameterSet seedTransformerPSet = par.getParameter<ParameterSet>("SeedTransformerParameters");
  ParameterSet seedTransformerParameters = par.getParameter<ParameterSet>("SeedTransformerParameters");
  theSeedTransformer = new SeedTransformer(seedTransformerParameters);
}

DirectMuonTrajectoryBuilder::~DirectMuonTrajectoryBuilder(){

  LogTrace("Muon|RecoMuon|DirectMuonTrajectoryBuilder") 
    << "DirectMuonTrajectoryBuilder destructor called" << endl;
  
  if(theSeedTransformer) delete theSeedTransformer;
}

MuonTrajectoryBuilder::TrajectoryContainer 
DirectMuonTrajectoryBuilder::trajectories(const TrajectorySeed& seed){
  
  // Set the services for the seed transformer
  theSeedTransformer->setServices(theService->eventSetup());

  const string metname = "Muon|RecoMuon|DirectMuonTrajectoryBuilder";
  
  MuonTrajectoryBuilder::TrajectoryContainer trajectoryContainer;
  
  vector<Trajectory> seedTrajectories = theSeedTransformer->seedTransform(seed);
  
  if(!seedTrajectories.empty())
    for(vector<Trajectory>::const_iterator trajectory = seedTrajectories.begin(); trajectory!=seedTrajectories.end(); ++trajectory)
      trajectoryContainer.push_back(new Trajectory(*trajectory));
  else LogTrace(metname) << "Seed not refitted";
    
  
  return  trajectoryContainer;


// std::pair<bool, Trajectory> 
// SETFilter::bwfit_SET(const TrajectorySeed &trajectorySeed  , 
// 				  const TransientTrackingRecHit::ConstRecHitContainer & trajRH,
// 				  const TrajectoryStateOnSurface & firstTsos) {
//   // get the actual fitter - Kalman fit
//   theService->eventSetup().get<TrajectoryFitter::Record>().get(theBWLightFitterName, theBWLightFitter);
//   vector<Trajectory> refitted;
//   Trajectory trajectory;
//   // the actual Kalman Fit
//   refitted = theBWLightFitter->fit(trajectorySeed, trajRH, firstTsos);                                  
//   if(!refitted.empty()){
//     // under tests...
//     bool applyPruning = false;
//     if(applyPruning){
//       double previousTheta = trajRH[0]->globalPosition().theta();
//       double previousWeight = 0.;
//       std::vector <double> weights(trajRH.size());
//       std::vector <double> weight_diff(trajRH.size());
//       for(unsigned int iRH = 0; iRH<trajRH.size();++iRH){
// 	double weight = trajRH[iRH]->globalPosition().theta() - previousTheta;
// 	weights.at(iRH)= weight;
// 	double weightDiff = weight + previousWeight;
// 	weight_diff.at(iRH) = weightDiff;
// 	std::cout<<" iRH = "<<iRH<<" globPos"<< trajRH[iRH]->globalPosition()<<" weight = "<<weight<<" weightDiff = "<<weightDiff<<std::endl;
// 	previousTheta = trajRH[iRH]->globalPosition().theta();
// 	previousWeight = weight;

//       }
//       Trajectory::DataContainer measurements_segments = refitted.front().measurements();
//       if(measurements_segments.size() != trajRH.size()){
// 	std::cout<<" measurements_segments.size() = "<<measurements_segments.size()<<
// 	  " trajRH.size() = "<<trajRH.size()<<std::endl;
// 	std::cout<<" THIS IS NOT SUPPOSED TO HAPPEN! CHECK THE CODE (pruning)"<<std::endl;
//       }
//       std::vector <int> badHits;
//       TransientTrackingRecHit::ConstRecHitContainer trajRH_pruned;
//       for(unsigned int iMeas = 0; iMeas<measurements_segments.size();++iMeas){
// 	// we have to apply pruning on the base of intermed. chi2 of measurements
// 	// and then refit again!
// 	std::cout<<" after refitter : iMeas = "<<iMeas<<"  estimate() = "<< measurements_segments[iMeas].estimate()<<
// 	  " globPos = "<< measurements_segments[iMeas].recHit()->globalPosition()<<std::endl;
// 	//const TransientTrackingRecHit::ConstRecHitContainer trajRH_pruned;
// 	bool pruningCondition = fabs(weights[iMeas])>0.0011 && fabs(weight_diff[iMeas])>0.0011;
// 	std::cout<<" weights[iMeas] = "<<weights[iMeas]<<" weight_diff[iMeas] = "<<weight_diff[iMeas]<<" pruningCondition = "<<pruningCondition<<std::endl;
// 	//bool pruningCondition = (measurements_segments[iMeas].estimate()>50);
// 	if(iMeas && pruningCondition && measurements_segments.size() == trajRH.size()){// first is kept for technical reasons (for now)
// 	  badHits.push_back(iMeas);
// 	}
// 	else{
// 	  trajRH_pruned.push_back(trajRH[iMeas]);
// 	}
//       }
//       if(float(measurements_segments.size())/float(badHits.size()) >0.5 &&
// 	 measurements_segments.size() - badHits.size() > 6){
// 	std::cout<<" this is pruning ; badHits.size() = "<<badHits.size()<<std::endl;
// 	refitted = theBWLightFitter->fit(trajectorySeed, trajRH_pruned, firstTsos);  
//       }
//     }
//     std::pair<bool, Trajectory> refitResult = make_pair(true,refitted.front());
//     //return RefitResult(true,refitted.front());
//     return refitResult;
//   }
//   else{
//     //    std::cout<<" refitted.empty() = "<<refitted.empty()<<std::endl;
//     std::pair<bool, Trajectory> refitResult = make_pair(false,trajectory);
//     //return RefitResult(false,trajectory);
//     return refitResult;
//   }
}


MuonTrajectoryBuilder::CandidateContainer 
DirectMuonTrajectoryBuilder::trajectories(const TrackCand&)
{
  return MuonTrajectoryBuilder::CandidateContainer();
}



void DirectMuonTrajectoryBuilder::setEvent(const edm::Event& event){
}


