#include "RecoMuon/TrackingTools/plugins/MuonErrorMatrixAdjuster.h"

#include "TString.h"
#include "TMath.h"
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>
#include <MagneticField/Engine/interface/MagneticField.h>

#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include "RecoMuon/TrackingTools/interface/MuonErrorMatrix.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h>
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

MuonErrorMatrixAdjuster::MuonErrorMatrixAdjuster(const edm::ParameterSet& iConfig)
{
  theCategory="MuonErrorMatrixAdjuster";
  theInstanceName = iConfig.getParameter<std::string>("instanceName");
  //register your products
  produces<reco::TrackCollection>( theInstanceName);
  produces<TrackingRecHitCollection>(); 
  produces<reco::TrackExtraCollection>();
          
  theTrackLabel = iConfig.getParameter<edm::InputTag>("trackLabel");
  theRescale = iConfig.getParameter<bool>("rescale");

  theMatrixProvider_pset = iConfig.getParameter<edm::ParameterSet>("errorMatrix_pset");
}


MuonErrorMatrixAdjuster::~MuonErrorMatrixAdjuster()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

//take the error matrix and rescale it or just replace it
reco::TrackBase::CovarianceMatrix MuonErrorMatrixAdjuster::fix_cov_matrix(const reco::TrackBase::CovarianceMatrix& error_matrix,const GlobalVector& momentum)
{   
  //CovarianceMatrix is template for SMatrix
  reco::TrackBase::CovarianceMatrix revised_matrix( theMatrixProvider->get(momentum));
   
  if( theRescale){
    //rescale old error matrix up by a factor
    multiply(revised_matrix,error_matrix); 
  }
  return revised_matrix;
}

void MuonErrorMatrixAdjuster::multiply(reco::TrackBase::CovarianceMatrix & revised_matrix, const reco::TrackBase::CovarianceMatrix & scale_matrix){
  //scale term by term the matrix
  // the true type of the matrix is such that [i][j] is the same memory object as [j][i]: looping i:0-5, j:0-5 double multiply the terms
  // need to loop only on i:0-5, j:i-5
  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      revised_matrix(i,j)*=scale_matrix(i,j);
    }}
}
bool MuonErrorMatrixAdjuster::divide(reco::TrackBase::CovarianceMatrix & num_matrix, const reco::TrackBase::CovarianceMatrix & denom_matrix){
  //divide term by term the matrix
  // the true type of the matrix is such that [i][j] is the same memory object as [j][i]: looping i:0-5, j:0-5 double multiply the terms
  // need to loop only on i:0-5, j:i-5
  for(int i = 0;i!=5;i++){for(int j = i;j!=5;j++){
      if (denom_matrix(i,j)==0) return false;
      num_matrix(i,j)/=denom_matrix(i,j);
   }}
  return true;
}

reco::Track MuonErrorMatrixAdjuster::makeTrack(const reco::Track & recotrack_orig,
					  const FreeTrajectoryState & PCAstate){
  //get the parameters of the track so I can reconstruct it
  double chi2 = recotrack_orig.chi2();
  double ndof = recotrack_orig.ndof();
  math::XYZPoint refpos = recotrack_orig.referencePoint();
  math::XYZVector mom = recotrack_orig.momentum();
  int charge = recotrack_orig.charge();
  
  reco::TrackBase::CovarianceMatrix covariance_matrix = fix_cov_matrix(recotrack_orig.covariance(),PCAstate.momentum());

  LogDebug(theCategory)<<"chi2: "<<chi2
		       <<"\n ndof: "<<ndof
		       <<"\n refpos: "<<refpos
		       <<"\n mom: "<<mom
		       <<"\n charge: "<<charge
		       <<"\n covariance:\n"<<recotrack_orig.covariance()
		       <<"\n replaced by:\n"<<covariance_matrix;

  return reco::Track(chi2,ndof,refpos,mom,charge,covariance_matrix);
}

reco::TrackExtra * MuonErrorMatrixAdjuster::makeTrackExtra(const reco::Track & recotrack_orig,
						      reco::Track & recotrack,
						      reco::TrackExtraCollection& TEcol){ 
  //get the 5x5 matrix of recotrack/recotrack_orig
  reco::TrackBase::CovarianceMatrix scale_matrix(recotrack.covariance());
  if (!divide(scale_matrix,recotrack_orig.covariance())){
    edm::LogError( theCategory)<<"original track error matrix has term ==0... skipping.";
    return 0; }
  
  const reco::TrackExtraRef & trackExtra_orig = recotrack_orig.extra();
  if (trackExtra_orig.isNull()) {
    edm::LogError( theCategory)<<"original track has no track extra... skipping.";
    return 0;}

  //copy the outer state. rescaling the error matrix
  reco::TrackBase::CovarianceMatrix outerCov(trackExtra_orig->outerStateCovariance());
  multiply(outerCov,scale_matrix);

  //copy the inner state, rescaling the error matrix
  reco::TrackBase::CovarianceMatrix innerCov(trackExtra_orig->innerStateCovariance());
  multiply(innerCov,scale_matrix);

  //put the trackExtra
  TEcol.push_back(reco::TrackExtra (trackExtra_orig->outerPosition(), trackExtra_orig->outerMomentum(), true,
				    trackExtra_orig->innerPosition(), trackExtra_orig->innerMomentum(), true,
				    outerCov, trackExtra_orig->outerDetId(),
				    innerCov, trackExtra_orig->innerDetId(),
				    trackExtra_orig->seedDirection()));

  //add a reference to the trackextra on the track
  recotrack.setExtra(edm::Ref<reco::TrackExtraCollection>( theRefprodTE, theTEi++));
  
  //return the reference to the last inserted then
  return &(TEcol.back());
}


bool MuonErrorMatrixAdjuster::attachRecHits(const reco::Track & recotrack_orig,
				       reco::Track & recotrack,
				       reco::TrackExtra & trackextra,
                                       TrackingRecHitCollection& RHcol,
                                       const TrackerTopology& ttopo){
  //loop over the hits of the original track
  trackingRecHit_iterator recHit = recotrack_orig.recHitsBegin();
  auto const firstHitIndex = theRHi;
  for (; recHit!=recotrack_orig.recHitsEnd();++recHit){
    //clone it. this is meandatory
    TrackingRecHit * hit = (*recHit)->clone();
    
    //put it on the new track
    recotrack.appendHitPattern(*hit, ttopo);
    //copy them in the new collection
    RHcol.push_back(hit);
    ++theRHi;

  }//loop over original rechits
  //do something with the trackextra 
  trackextra.setHits(theRefprodRH, firstHitIndex, theRHi-firstHitIndex);
  
  return true; //if nothing fails
}

bool MuonErrorMatrixAdjuster::selectTrack(const reco::Track & recotrack_orig){return true;}

// ------------ method called to produce the data  ------------
void
MuonErrorMatrixAdjuster::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  //open a collection of track
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel( theTrackLabel,tracks);
  LogDebug( theCategory)<<"considering: "<<tracks->size()<<" uncorrected reco::Track from the event.("<< theTrackLabel<<")";
  
  //get the mag field
  iSetup.get<IdealMagneticFieldRecord>().get( theField);

  edm::ESHandle<TrackerTopology> httopo;
  iSetup.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology& ttopo = *httopo;

  //prepare the output collection
  std::auto_ptr<reco::TrackCollection> Toutput(new reco::TrackCollection());
  std::auto_ptr<TrackingRecHitCollection> TRHoutput(new TrackingRecHitCollection());
  std::auto_ptr<reco::TrackExtraCollection> TEoutput(new reco::TrackExtraCollection());
  theRefprodTE = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  theTEi=0;
  theRefprodRH =iEvent.getRefBeforePut<TrackingRecHitCollection>();
  theRHi=0;
  
  
  
  for(unsigned int it=0;it!=tracks->size();it++)
    {
      const reco::Track & recotrack_orig = (*tracks)[it];
      FreeTrajectoryState PCAstate = trajectoryStateTransform::initialFreeState(recotrack_orig, theField.product());
      if (PCAstate.position().mag()==0)
	{edm::LogError( theCategory)<<"invalid state from track initial state in "<< theTrackLabel <<". skipping.";continue;}

      //create a reco::Track
      reco::Track recotrack = makeTrack(recotrack_orig,PCAstate);
      
      //make a selection on the create reco::Track
      if (!selectTrack(recotrack)) continue;
      
      Toutput->push_back(recotrack);
      reco::Track & recotrackref = Toutput->back();
      
      //build the track extra
      reco::TrackExtra * extra = makeTrackExtra(recotrack_orig,recotrackref,*TEoutput);
      if (!extra){
	edm::LogError( theCategory)<<"cannot create the track extra for this track.";
	//pop the inserted track
	Toutput->pop_back();
	continue;}
      
      //attach the collection of rechits
      if (!attachRecHits(recotrack_orig,recotrackref,*extra,*TRHoutput, ttopo)){
	edm::LogError( theCategory)<<"cannot attach any rechits on this track";
	//pop the inserted track
	Toutput->pop_back();
	//pop the track extra
	TEoutput->pop_back();
	theTEi--;
	continue;}
      
    }//loop over the original reco tracks
  
  LogDebug( theCategory)<<"writing: "<<Toutput->size()<<" corrected reco::Track to the event.";
  
  iEvent.put(Toutput, theInstanceName);
  iEvent.put(TEoutput);
  iEvent.put(TRHoutput);

}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonErrorMatrixAdjuster::beginJob()
{
  theMatrixProvider = new MuonErrorMatrix(theMatrixProvider_pset);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonErrorMatrixAdjuster::endJob() {
  if ( theMatrixProvider) delete theMatrixProvider;
}
