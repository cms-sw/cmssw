#include "HeavyFlavorAnalysis/Onia2MuMu/interface/OniaPhotonConversionProducer.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include <TMath.h>
#include <vector>


// to order from high to low ProbChi2
bool ConversionLessByChi2(const reco::Conversion& c1, const reco::Conversion& c2){
  return TMath::Prob(c1.conversionVertex().chi2(),c1.conversionVertex().ndof()) > TMath::Prob(c2.conversionVertex().chi2(),c2.conversionVertex().ndof());
}


bool ConversionEqualByTrack(const reco::Conversion& c1, const reco::Conversion& c2){
  bool atLeastOneInCommon=false;
  for(auto const& tk1 : c1.tracks()){
    for(auto const& tk2 : c2.tracks()){
      if(tk1==tk2){
	atLeastOneInCommon=true;
	break;
      }
    } 
  } 
  return atLeastOneInCommon;
}

bool lt_(std::pair<double,short> a, std::pair<double,short> b) { 
     return a.first < b.first; }

// define operator== for conversions, those with at least one track in common
namespace reco {
  bool operator==(const reco::Conversion& c1, const reco::Conversion& c2) {
    return c1.tracks()[0] == c2.tracks()[0] ||
           c1.tracks()[1] == c2.tracks()[1] ||
	   c1.tracks()[1] == c2.tracks()[0] ||
           c1.tracks()[0] == c2.tracks()[1] ;
  }
}

OniaPhotonConversionProducer:: OniaPhotonConversionProducer(const edm::ParameterSet& ps) {
  convCollectionToken_     = consumes<reco::ConversionCollection>(ps.getParameter<edm::InputTag>("conversions"));
  thePVsToken_             = consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("primaryVertexTag"));

  wantTkVtxCompatibility_  = ps.getParameter<bool>("wantTkVtxCompatibility");
  sigmaTkVtxComp_          = ps.getParameter<uint32_t>("sigmaTkVtxComp");
  wantCompatibleInnerHits_ = ps.getParameter<bool>("wantCompatibleInnerHits");
  TkMinNumOfDOF_           = ps.getParameter<uint32_t>("TkMinNumOfDOF");

  wantHighpurity_          = ps.getParameter<bool>("wantHighpurity");
  _vertexChi2ProbCut       = ps.getParameter<double>("vertexChi2ProbCut");
  _trackchi2Cut            = ps.getParameter<double>("trackchi2Cut");
  _minDistanceOfApproachMinCut = ps.getParameter<double>("minDistanceOfApproachMinCut");
  _minDistanceOfApproachMaxCut = ps.getParameter<double>("minDistanceOfApproachMaxCut");

  pfCandidateCollectionToken_  = consumes<reco::PFCandidateCollection>(ps.getParameter<edm::InputTag>("pfcandidates"));

  pi0OnlineSwitch_        = ps.getParameter<bool>("pi0OnlineSwitch");
  pi0SmallWindow_         = ps.getParameter<std::vector<double> >("pi0SmallWindow");
  pi0LargeWindow_         = ps.getParameter<std::vector<double> >("pi0LargeWindow");

  std::string algo = ps.getParameter<std::string>("convAlgo");
  convAlgo_ = StringToEnumValue<reco::Conversion::ConversionAlgorithm>(algo);

  std::vector<std::string> qual = ps.getParameter<std::vector<std::string> >("convQuality"); 
  if( !qual[0].empty() ) convQuality_ =StringToEnumValue<reco::Conversion::ConversionQuality>(qual);

  convSelectionCuts_ = ps.getParameter<std::string>("convSelection");
  convSelection_ = std::make_unique<StringCutObjectSelector<reco::Conversion>>(convSelectionCuts_);
  produces<pat::CompositeCandidateCollection>("conversions");
}

void OniaPhotonConversionProducer::produce(edm::Event& event, const edm::EventSetup& esetup){

  std::unique_ptr<reco::ConversionCollection> outCollection(new reco::ConversionCollection);
  std::unique_ptr<pat::CompositeCandidateCollection> patoutCollection(new pat::CompositeCandidateCollection);

  edm::Handle<reco::VertexCollection> priVtxs;
  event.getByToken(thePVsToken_, priVtxs);
    
  edm::Handle<reco::ConversionCollection> pConv;
  event.getByToken(convCollectionToken_,pConv);

  edm::Handle<reco::PFCandidateCollection> pfcandidates;
  event.getByToken(pfCandidateCollectionToken_,pfcandidates);

  const reco::PFCandidateCollection pfphotons = selectPFPhotons(*pfcandidates);  

  for(reco::ConversionCollection::const_iterator conv = pConv->begin(); conv != pConv->end(); ++conv){

    if (! ( *convSelection_)(*conv)){
	continue; // selection string
    }
    if (convAlgo_ != 0 && conv->algo()!= convAlgo_){
	continue; // select algorithm
    }
    if(!convQuality_.empty()){
	bool flagsok=true;
	for (std::vector<int>::const_iterator flag = convQuality_.begin(); flag!=convQuality_.end(); ++flag){
	reco::Conversion::ConversionQuality q = (reco::Conversion::ConversionQuality)(*flag);
           if (!conv->quality(q)) {
	      flagsok=false;
	      break;
           }
        }
	if (!flagsok){
	   continue;
        }
    }
    outCollection->push_back(*conv);
  }
    
  removeDuplicates(*outCollection);
  
  for (reco::ConversionCollection::const_iterator conv = outCollection->begin(); conv != outCollection->end(); ++conv){

     bool flag1 = true;
     bool flag2 = true;
     bool flag3 = true;
     bool flag4 = true;

    // The logic implies that by default this flags are true and if the check are not wanted conversions are saved.
    // If checks are required and failed then don't save the conversion.

     bool flagTkVtxCompatibility  = true;
     if (! checkTkVtxCompatibility(*conv,*priVtxs.product())) {
       flagTkVtxCompatibility = false;
       if (wantTkVtxCompatibility_) {
          flag1 = false;
       }
     }
     bool flagCompatibleInnerHits = false;
     if (conv->tracks().size()==2) {
       reco::HitPattern hitPatA=conv->tracks().at(0)->hitPattern();
       reco::HitPattern hitPatB=conv->tracks().at(1)->hitPattern();
       if ( foundCompatibleInnerHits(hitPatA,hitPatB) && foundCompatibleInnerHits(hitPatB,hitPatA) ) flagCompatibleInnerHits = true;
     }
     if (wantCompatibleInnerHits_ && ! flagCompatibleInnerHits) {
       flag2 = false;
     }
     bool flagHighpurity = true;
     if (!HighpuritySubset(*conv,*priVtxs.product())) {
       flagHighpurity = false;
       if (wantHighpurity_) {
          flag3 = false;
       }
     }
     bool pizero_rejected = false;
     bool large_pizero_window = CheckPi0(*conv, pfphotons, pizero_rejected);
     if (pi0OnlineSwitch_ && pizero_rejected) {
       flag4 = false;
     }

     int flags = 0;
     if (flag1 && flag2 && flag3 && flag4){
        flags = PackFlags(*conv,flagTkVtxCompatibility,flagCompatibleInnerHits,flagHighpurity,pizero_rejected,large_pizero_window);
        std::unique_ptr<pat::CompositeCandidate> pat_conv(makePhotonCandidate(*conv));
        pat_conv->addUserInt("flags",flags);
        patoutCollection->push_back(*pat_conv);
     }
  }
  event.put(std::move(patoutCollection),"conversions");
}

int OniaPhotonConversionProducer::PackFlags(const reco::Conversion& conv, bool flagTkVtxCompatibility, 
                                            bool flagCompatibleInnerHits, bool flagHighpurity,
                                            bool pizero_rejected, bool large_pizero_window ) {
   int flags = 0;
   if ( flagTkVtxCompatibility ) flags += 1;
   if ( flagCompatibleInnerHits ) flags += 2;
   if ( flagHighpurity ) flags += 4;
   if ( pizero_rejected ) flags += 8;
   if ( large_pizero_window ) flags += 16;

   flags += (conv.algo()*32);
   int q_mask = 0;
   std::vector<std::string> s_quals;
   s_quals.push_back("generalTracksOnly");
   s_quals.push_back("arbitratedEcalSeeded");
   s_quals.push_back("arbitratedMerged");
   s_quals.push_back("arbitratedMergedEcalGeneral");
   s_quals.push_back("highPurity");
   s_quals.push_back("highEfficiency");
   s_quals.push_back("ecalMatched1Track");
   s_quals.push_back("ecalMatched2Track");
   std::vector<int>  i_quals = StringToEnumValue<reco::Conversion::ConversionQuality>(s_quals);
   for (std::vector<int>::const_iterator qq = i_quals.begin(); qq!=i_quals.end(); ++qq) {
      reco::Conversion::ConversionQuality q = (reco::Conversion::ConversionQuality)(*qq);
      if (conv.quality(q)) q_mask = *qq;
   }
   flags += (q_mask*32*8);
   return flags;
}

/** Put in out collection only those conversion candidates that are not sharing tracks.
    If sharing, keep the one with the best chi2.
 */
void OniaPhotonConversionProducer::removeDuplicates(reco::ConversionCollection& c){
  // first sort from high to low chi2 prob
  std::sort(c.begin(),c.end(),ConversionLessByChi2);
  int iter1 = 0;
  // Cycle over all the elements of the collection and compare to all the following, 
  // if two elements have at least one track in common delete the element with the lower chi2
  while(iter1 < (( (int) c.size() ) - 1) ){
     int iter2 = iter1+1;
     while( iter2 < (int) c.size()) if(ConversionEqualByTrack( c[iter1], c[iter2] ) ){
        c.erase( c.begin() + iter2 );
        }else{
        iter2++;	// Increment index only if this element is no duplicate. 
			// If it is duplicate check again the same index since the vector rearranged elements index after erasing
        }
     iter1++;
  }
}

bool OniaPhotonConversionProducer::checkTkVtxCompatibility(const reco::Conversion& conv, const reco::VertexCollection& priVtxs) {
  std::vector< std::pair< double, short> > idx[2];
  short ik=-1;
  for(auto const& tk : conv.tracks()){
    ik++;
    short count=-1;
    for(auto const& vtx : priVtxs){
      count++;
    
      double dz_= tk->dz(vtx.position());
      double dzError_=tk->dzError();
      dzError_=sqrt(dzError_*dzError_+vtx.covariance(2,2));
      if(fabs(dz_)/dzError_ > sigmaTkVtxComp_) continue;
      idx[ik].push_back(std::pair<double,short>(fabs(dz_),count));
    }
    if (idx[ik].empty()) return false;
    if (idx[ik].size()>1) std::stable_sort(idx[ik].begin(),idx[ik].end(),lt_);
  }
  if (ik!=1) return false;
  if (idx[0][0].second==idx[1][0].second) return true;
  if (idx[0].size()>1 && idx[0][1].second==idx[1][0].second) return true;
  if (idx[1].size()>1 && idx[0][0].second==idx[1][1].second) return true;

  return false;
}

bool OniaPhotonConversionProducer::foundCompatibleInnerHits(const reco::HitPattern& hitPatA, const reco::HitPattern& hitPatB) {
  size_t count=0;
  uint32_t oldSubStr=0;
  for (int i=0; i<hitPatA.numberOfAllHits(reco::HitPattern::HitCategory::TRACK_HITS) && count<2; i++) {
    uint32_t hitA = hitPatA.getHitPattern(reco::HitPattern::HitCategory::TRACK_HITS,i);
    if (!hitPatA.validHitFilter(hitA) || !hitPatA.trackerHitFilter(hitA)) continue;
    
    if(hitPatA.getSubStructure(hitA)==oldSubStr && hitPatA.getLayer(hitA)==oldSubStr)
      continue;

    if(hitPatB.getTrackerMonoStereo(reco::HitPattern::HitCategory::TRACK_HITS,hitPatA.getSubStructure(hitA),hitPatA.getLayer(hitA)) != 0)
      return true;
    
    oldSubStr=hitPatA.getSubStructure(hitA);
    count++;
  } 
  return false;  
}

bool OniaPhotonConversionProducer::
HighpuritySubset(const reco::Conversion& conv, const reco::VertexCollection& priVtxs){
  // select high purity conversions our way:
  // vertex chi2 cut
  if(ChiSquaredProbability(conv.conversionVertex().chi2(),conv.conversionVertex().ndof())< _vertexChi2ProbCut) return false;

  // d0 cut
  // Find closest primary vertex
  int closest_pv_index = 0;
  int i=0;
  for(auto const& vtx : priVtxs){
    if( conv.zOfPrimaryVertexFromTracks( vtx.position() ) < conv.zOfPrimaryVertexFromTracks( priVtxs[closest_pv_index].position() ) ) closest_pv_index = i;
    i++;
  }
  // Now check impact parameter wtr with the just found closest primary vertex
  for(auto const& tk : conv.tracks()) if(-tk->dxy(priVtxs[closest_pv_index].position())*tk->charge()/tk->dxyError()<0) return false;
  
  // chi2 of single tracks
  for(auto const& tk : conv.tracks()) if(tk->normalizedChi2() > _trackchi2Cut) return false;

  // dof for each track  
  for(auto const& tk : conv.tracks()) if(tk->ndof()< TkMinNumOfDOF_) return false;
  
  // distance of approach cut
  if (conv.distOfMinimumApproach() < _minDistanceOfApproachMinCut || conv.distOfMinimumApproach() > _minDistanceOfApproachMaxCut ) return false;
  
  return true;
}

pat::CompositeCandidate *OniaPhotonConversionProducer::makePhotonCandidate(const reco::Conversion& conv){

  pat::CompositeCandidate *photonCand = new pat::CompositeCandidate();
  photonCand->setP4(convertVector(conv.refittedPair4Momentum()));
  photonCand->setVertex(conv.conversionVertex().position());

  photonCand->addUserData<reco::Track>( "track0", *conv.tracks()[0] );
  photonCand->addUserData<reco::Track>( "track1", *conv.tracks()[1] );

  return photonCand;

}

// create a collection of PF photons
const reco::PFCandidateCollection  OniaPhotonConversionProducer::selectPFPhotons(const reco::PFCandidateCollection& pfcandidates) {
  reco::PFCandidateCollection pfphotons;
  for (reco::PFCandidateCollection::const_iterator cand =   pfcandidates.begin(); cand != pfcandidates.end(); ++cand){
    if (cand->particleId() == reco::PFCandidate::gamma) pfphotons.push_back(*cand);
  }
  return  pfphotons;
}

bool OniaPhotonConversionProducer::CheckPi0( const reco::Conversion& conv, const reco::PFCandidateCollection& photons,
				            bool &pizero_rejected ) {
  // 2 windows are defined for Pi0 rejection, Conversions that, paired with others photons from the event, have an
  // invariant mass inside the "small" window will be pizero_rejected and those that falls in the large window will
  // be CheckPi0.
  bool check_small = false;
  bool check_large = false;

  float small1 = pi0SmallWindow_[0];
  float small2 = pi0SmallWindow_[1];
  float large1 = pi0LargeWindow_[0];
  float large2 = pi0LargeWindow_[1];
  for (reco::PFCandidateCollection::const_iterator photon = photons.begin(); photon!=photons.end(); ++photon) {
    float inv = (conv.refittedPair4Momentum() + photon->p4()).M(); 
    if (inv > large1 && inv < large2) {
      check_large = true;
      if (inv > small1 && inv < small2) { 
         check_small = true;
         break;
      }
    }
  }
  pizero_rejected = check_small;
  return check_large;
}

reco::Candidate::LorentzVector OniaPhotonConversionProducer::convertVector(const math::XYZTLorentzVectorF& v){
  return reco::Candidate::LorentzVector(v.x(),v.y(), v.z(), v.t());
}


void OniaPhotonConversionProducer::endStream() {
}
//define this as a plug-in
DEFINE_FWK_MODULE(OniaPhotonConversionProducer);
