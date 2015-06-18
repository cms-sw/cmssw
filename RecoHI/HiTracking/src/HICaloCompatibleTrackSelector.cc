/*
 
 Based on analytical track selector   
 
 - This track selector assigns a quality bit to tracks deemed compatible with matching calo info
 - The default mode is to use the matching provided by particle flow,
   but a delta R matching to calorimeter towers is also supported
 - No selection is done other then selecting calo-compatible tracks.
 - The code always keeps all tracks in the input collection (should make configurable) 
 - Note that matching by PF candidate only works on the same track collection used as input to PF
 - Tower code not up to data
 
   Authors:  Matthew Nguyen, Andre Yoon, Frank Ma (November 4th, 2011)

 */


// Basic inclusion
#include "RecoHI/HiTracking/interface/HICaloCompatibleTrackSelector.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/deltaR.h"
#include <Math/DistFunc.h>
#include "TMath.h"


using reco::modules::HICaloCompatibleTrackSelector;

HICaloCompatibleTrackSelector::HICaloCompatibleTrackSelector( const edm::ParameterSet & cfg ) :
  srcTracks_(consumes<TrackCollection>(cfg.getParameter<edm::InputTag>("srcTracks"))),
  srcPFCands_(consumes<PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCands"))),
  srcTower_(consumes<CaloTowerCollection>(cfg.getParameter<edm::InputTag>("srcTower"))),
  usePFCandMatching_(cfg.getUntrackedParameter<bool>("usePFCandMatching", true)),
  trkMatchPtMin_(cfg.getUntrackedParameter<double>("trkMatchPtMin",10.0)),
  trkCompPtMin_(cfg.getUntrackedParameter<double>("trkCompPtMin",35.0)),
  trkEtaMax_(cfg.getUntrackedParameter<double>("trkEtaMax",2.4)),
  towerPtMin_(cfg.getUntrackedParameter<double>("towerPtMin",5.0)),
  matchConeRadius_(cfg.getUntrackedParameter<double>("matchConeRadius",0.087)),
  keepAllTracks_(cfg.getUntrackedParameter<bool>("keepAllTracks", true)),
  copyExtras_(cfg.getUntrackedParameter<bool>("copyExtras", true)),
  copyTrajectories_(cfg.getUntrackedParameter<bool>("copyTrajectories", true)),
  qualityToSet_(cfg.getParameter<std::string>("qualityToSet")),
  qualityToSkip_(cfg.getParameter<std::string>("qualityToSkip")),
  qualityToMatch_(cfg.getParameter<std::string>("qualityToMatch")),
  minimumQuality_(cfg.getParameter<std::string>("minimumQuality")),
  resetQuality_(cfg.getUntrackedParameter<bool>("resetQuality", true)),
  passMuons_(cfg.getUntrackedParameter<bool>("passMuons", true)),
  passElectrons_(cfg.getUntrackedParameter<bool>("passElectrons", false)),
  funcDeltaRTowerMatch_(cfg.getParameter<std::string>("funcDeltaRTowerMatch")),
  funcCaloComp_(cfg.getParameter<std::string>("funcCaloComp"))
{
  std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
  produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks");
  if (copyExtras_) {
    produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras");
    produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits");
  }
  if (copyTrajectories_) {
    produces< std::vector<Trajectory> >().setBranchAlias( alias + "Trajectories");
    produces< TrajTrackAssociationCollection >().setBranchAlias( alias + "TrajectoryTrackAssociations");
    srcTrackTrajs_ = (consumes<std::vector<Trajectory> >(cfg.getParameter<edm::InputTag>("srcTracks")));
    srcTrackTrajAssoc_ = (consumes<TrajTrackAssociationCollection>(cfg.getParameter<edm::InputTag>("srcTracks")));
  }

  // pt dependence of delta R matching requirement
  fDeltaRTowerMatch = new TF1("fDeltaRTowerMatch",funcDeltaRTowerMatch_.c_str(),0,200); 
  // pt dependance of calo compatibility, i.e., minimum sum Calo Et vs. track pT
  fCaloComp = new TF1("fCaloComp",funcCaloComp_.c_str(),0,200); // a parameterization of pt dependent cut


}

HICaloCompatibleTrackSelector::~HICaloCompatibleTrackSelector() {
}


void HICaloCompatibleTrackSelector::produce( edm::Event& evt, const edm::EventSetup& es ) 
{
  using namespace std; 
  using namespace edm;
  using namespace reco;
  
  LogDebug("HICaloCompatibleTrackSelector")<<"min pt for selection = "<< trkMatchPtMin_<<endl;
  
  
  Handle<TrackCollection> hSrcTrack;
  Handle< vector<Trajectory> > hTraj;
  Handle< vector<Trajectory> > hTrajP;
  Handle< TrajTrackAssociationCollection > hTTAss;

  evt.getByToken(srcTracks_,hSrcTrack);
  
  selTracks_ = auto_ptr<TrackCollection>(new TrackCollection());
  rTracks_ = evt.getRefBeforePut<TrackCollection>();      
  if (copyExtras_) {
    selTrackExtras_ = auto_ptr<TrackExtraCollection>(new TrackExtraCollection());
    selHits_ = auto_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
    rHits_ = evt.getRefBeforePut<TrackingRecHitCollection>();
    rTrackExtras_ = evt.getRefBeforePut<TrackExtraCollection>();
  }


  if (copyTrajectories_) trackRefs_.resize(hSrcTrack->size());


  Handle<PFCandidateCollection> pfCandidates;
  Handle<CaloTowerCollection> towers;

  bool isPFThere = false;
  bool isTowerThere = false;
  
  if(usePFCandMatching_) isPFThere = evt.getByToken(srcPFCands_, pfCandidates);  
  else isTowerThere = evt.getByToken(srcTower_, towers);
  
  size_t current = 0;
  for (TI ti = hSrcTrack->begin(), ed = hSrcTrack->end(); ti != ed; ++ti, ++current) {
    

    const reco::Track& trk = *ti;
    
    bool isSelected;
    if(usePFCandMatching_) isSelected = selectByPFCands(ti, hSrcTrack, pfCandidates, isPFThere);
    else isSelected = selectByTowers(ti, hSrcTrack, towers, isTowerThere);
    
    if(!keepAllTracks_ && !isSelected) continue; 

    // Add all tracks to output collection, the rest of the code only sets the quality bit
    selTracks_->push_back( reco::Track( trk ) ); // clone and store
    
    
    if(isSelected) selTracks_->back().setQuality(reco::TrackBase::qualityByName(qualityToSet_.c_str()));
    
        
    if (copyExtras_) {
      // TrackExtras
      selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
					      trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
					      trk.outerStateCovariance(), trk.outerDetId(),
					      trk.innerStateCovariance(), trk.innerDetId(),
					      trk.seedDirection(), trk.seedRef() ) );
      selTracks_->back().setExtra( TrackExtraRef( rTrackExtras_, selTrackExtras_->size() - 1) );
      TrackExtra & tx = selTrackExtras_->back();
      tx.setResiduals(trk.residuals());
      // TrackingRecHits
      auto const firstHitIndex = selHits_->size();
      for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	selHits_->push_back( (*hit)->clone() );
      }
      tx.setHits( rHits_, firstHitIndex, selHits_->size() - firstHitIndex );
    }
    if (copyTrajectories_) {
      trackRefs_[current] = TrackRef(rTracks_, selTracks_->size() - 1);
    }

    
  }  // close track loop
  
  if ( copyTrajectories_ ) {
    Handle< vector<Trajectory> > hTraj;
    Handle< TrajTrackAssociationCollection > hTTAss;
    evt.getByToken(srcTrackTrajs_, hTTAss);
    evt.getByToken(srcTrackTrajAssoc_, hTraj);
    selTrajs_ = auto_ptr< vector<Trajectory> >(new vector<Trajectory>()); 
    rTrajectories_ = evt.getRefBeforePut< vector<Trajectory> >();
    selTTAss_ = auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
    for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
      Ref< vector<Trajectory> > trajRef(hTraj, i);
      TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
      if (match != hTTAss->end()) {
	const Ref<TrackCollection> &trkRef = match->val; 
	short oldKey = static_cast<short>(trkRef.key());
	if (trackRefs_[oldKey].isNonnull()) {
	  selTrajs_->push_back( Trajectory(*trajRef) );
	  selTTAss_->insert ( Ref< vector<Trajectory> >(rTrajectories_, selTrajs_->size() - 1), trackRefs_[oldKey] );
	}
      }
    }
  }
  
  static const string emptyString;
  evt.put(selTracks_);
  if (copyExtras_ ) {
    evt.put(selTrackExtras_); 
    evt.put(selHits_);
  }
  if ( copyTrajectories_ ) {
    evt.put(selTrajs_);
    evt.put(selTTAss_);
  }
}



bool 
HICaloCompatibleTrackSelector::selectByPFCands(TI ti, edm::Handle<TrackCollection> hSrcTrack, edm::Handle<PFCandidateCollection> pfCandidates, bool isPFThere)
{

  const reco::Track& trk = *ti;

  // If it passes this quality threshold or is under the minimum match pT, automatically save it
  if(trk.quality(reco::TrackBase::qualityByName(qualityToSkip_))){
    return true;
  }
  else if(!trk.quality(reco::TrackBase::qualityByName(minimumQuality_))){
    return false;
  }
  else
    {      
      
      double trkPt = trk.pt();
      //if(trkPt < trkMatchPtMin_ ) return false;
      
      double caloEt = 0.0;
      if(usePFCandMatching_){
	if(isPFThere){
	  unsigned int trackKey = ti - hSrcTrack->begin();
	  caloEt = matchPFCandToTrack(pfCandidates, trackKey, trkPt);      
	}
      }
      
      // Set quality bit based on calo matching
      if(!(caloEt>0.)) return false;
      
      
      if(trkPt<=trkCompPtMin_){
	if(trk.quality(reco::TrackBase::qualityByName(qualityToMatch_))) return true;
	else return false;
      }
      else{
	// loose cuts are implied in selectors, make configurable?
	float compPt = (fCaloComp->Eval(trkPt)!=fCaloComp->Eval(trkPt)) ? 0 : fCaloComp->Eval(trkPt); // protect agains NaN
	if(caloEt>compPt) return true;
	else return false;
      }            
    } // else above trkMatchPtMin_

  throw cms::Exception("Undefined case in HICaloCompatibleTrackSelector") << "Undefined case in HICaloCompatibleTrackSelector";
}


bool 
HICaloCompatibleTrackSelector::selectByTowers(TI ti, edm::Handle<TrackCollection> hSrcTrack, edm::Handle<CaloTowerCollection> towers, bool isTowerThere)
{

  // Not up to date! use PF towers instead

  const reco::Track& trk = *ti;

  // If it passes the high purity cuts, then consider it confirmed
  if(trk.quality(reco::TrackBase::qualityByName(qualityToSkip_))){
    return true;
  }
  else{
    if(trk.pt() <  trkMatchPtMin_ || !trk.quality(reco::TrackBase::qualityByName(qualityToMatch_))) return false;
    
    double caloEt = 0.0;
    if(isTowerThere){      
      double matchDr;
      matchByDrAllowReuse(trk,towers,matchDr,caloEt);
      float matchConeRadius_pt = (fDeltaRTowerMatch->Eval(trk.pt())!=fDeltaRTowerMatch->Eval(trk.pt())) ? 0 : fDeltaRTowerMatch->Eval(trk.pt()); // protect agains NaN
      if (caloEt>0 && matchDr>matchConeRadius_pt) caloEt=0.;      
    }
    
    if(trk.pt()<trkCompPtMin_||caloEt>0.75*(trk.pt()-trkCompPtMin_)) return true;
    else return false;
  }
}

double
HICaloCompatibleTrackSelector::matchPFCandToTrack( const edm::Handle<reco::PFCandidateCollection> & pfCandidates, unsigned it, double trkPt)
{

  // This function returns the sum of the calo energy in the block containing the track
  // There is another piece of information which could be useful which is the pT assigned to the charged hadron by PF
  

  double sum_ecal=0.0, sum_hcal=0.0;
    
  int candType = 0;
  reco::PFCandidate matchedCand; 

  // loop over the PFCandidates until you find the one whose trackRef points to the track

  for( CI ci  = pfCandidates->begin(); ci!=pfCandidates->end(); ++ci)  {

    const reco::PFCandidate& cand = *ci;

    int type = cand.particleId();
    
    // only charged hadrons and leptons can be asscociated with a track
    if(!(type == PFCandidate::h ||     //type1
	 type == PFCandidate::e ||     //type2
	 type == PFCandidate::mu      //type3
	 )
       ) continue;
    
    
    unsigned candTrackRefKey = cand.trackRef().key();      
    
    if(it==candTrackRefKey) {
      matchedCand = cand;
      candType = type;
      break;      
    }
  }
  // take all muons as compatible, extend to electrons when validataed
  if(passMuons_ && candType==3) return 9999.;
  if(passElectrons_ && candType==2) return 9999.;

  if(trkPt < trkMatchPtMin_ ) return 0.;

  if(candType>0){
    
    // Now that we found the matched PF candidate, loop over the elements in the block summing the calo Et
    for(unsigned ib=0; ib<matchedCand.elementsInBlocks().size(); ib++) {
      
      PFBlockRef blockRef = matchedCand.elementsInBlocks()[ib].first;
            
      unsigned indexInBlock = matchedCand.elementsInBlocks()[ib].second;
      const edm::OwnVector<  reco::PFBlockElement>&  elements = (*blockRef).elements();
      
      //This tells you what type of element it is:
      //cout<<" block type"<<elements[indexInBlock].type()<<endl;
      
      switch (elements[indexInBlock].type()) {
	
      case PFBlockElement::ECAL: {
	reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
	sum_ecal += clusterRef->energy()/cosh(clusterRef->eta());
	break;
      }
	
      case PFBlockElement::HCAL: {
	reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
	sum_hcal += clusterRef->energy()/cosh(clusterRef->eta());
	break; 
      }       
      case PFBlockElement::TRACK: {
	//Do nothing since track are not normally linked to other tracks
	break; 
      }       
      default:
	break;
      }
      
      // We could also add in the PS, HO, ..

    } // end of elementsInBlocks()
  }  // end of isCandFound	
  
  

  return sum_ecal+sum_hcal;
  
}

void HICaloCompatibleTrackSelector::matchByDrAllowReuse(const reco::Track & trk, const edm::Handle<CaloTowerCollection> & towers, double & bestdr, double & bestpt)
{
  // loop over towers
  bestdr=1e10;
  bestpt=0.;
  for(unsigned int i = 0; i < towers->size(); ++i){
    const CaloTower & tower= (*towers)[i];
    double pt = tower.pt();
    if (pt<towerPtMin_) continue;
    if (fabs(tower.eta())>trkEtaMax_) continue;
    double dr = reco::deltaR(tower,trk);
    if (dr<bestdr) {
      bestdr = dr;
      bestpt = pt;
    }
  }
}
