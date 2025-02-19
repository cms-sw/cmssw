/** \class HLTMuonL3PreFilter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz, J-R Vlimant
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL3PreFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// constructors and destructor
//
using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;

HLTMuonL3PreFilter::HLTMuonL3PreFilter(const ParameterSet& iConfig) : HLTFilter(iConfig),
   beamspotTag_   (iConfig.getParameter< edm::InputTag > ("BeamSpotTag")),
   candTag_   (iConfig.getParameter<InputTag > ("CandTag")),
   previousCandTag_   (iConfig.getParameter<InputTag > ("PreviousCandTag")),
   min_N_     (iConfig.getParameter<int> ("MinN")),
   max_Eta_   (iConfig.getParameter<double> ("MaxEta")),
   min_Nhits_ (iConfig.getParameter<int> ("MinNhits")),
   max_Dr_    (iConfig.getParameter<double> ("MaxDr")),
   min_Dr_    (iConfig.getParameter<double> ("MinDr")),
   max_Dz_    (iConfig.getParameter<double> ("MaxDz")),
   min_DxySig_(iConfig.getParameter<double> ("MinDxySig")),
   min_Pt_    (iConfig.getParameter<double> ("MinPt")),
   nsigma_Pt_  (iConfig.getParameter<double> ("NSigmaPt")), 
   max_NormalizedChi2_ (iConfig.getParameter<double> ("MaxNormalizedChi2")),
   max_DXYBeamSpot_ (iConfig.getParameter<double> ("MaxDXYBeamSpot")),
   min_NmuonHits_ (iConfig.getParameter<int> ("MinNmuonHits")),
   max_PtDifference_ (iConfig.getParameter<double> ("MaxPtDifference")),
   min_TrackPt_ (iConfig.getParameter<double> ("MinTrackPt")),
   devDebug_ (false)
{
   LogDebug("HLTMuonL3PreFilter")
      << " CandTag/MinN/MaxEta/MinNhits/MaxDr/MinDr/MaxDz/MinDxySig/MinPt/NSigmaPt : "
      << candTag_.encode()
      << " " << min_N_ 
      << " " << max_Eta_
      << " " << min_Nhits_
      << " " << max_Dr_
      << " " << min_Dr_
      << " " << max_Dz_
      << " " << min_DxySig_
      << " " << min_Pt_
      << " " << nsigma_Pt_;
}

HLTMuonL3PreFilter::~HLTMuonL3PreFilter()
{
}

void
HLTMuonL3PreFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL3MuonCandidates"));
  //  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag("hltDiMuonL2PreFiltered0"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  desc.add<int>("MinN",1);
  desc.add<double>("MaxEta",2.5);
  desc.add<int>("MinNhits",0);
  desc.add<double>("MaxDr",2.0);
  desc.add<double>("MinDr",-1.0);
  desc.add<double>("MaxDz",9999.0);
  desc.add<double>("MinDxySig",-1.0);
  desc.add<double>("MinPt",3.0);
  desc.add<double>("NSigmaPt",0.0);
  desc.add<double>("MaxNormalizedChi2",9999.0);
  desc.add<double>("MaxDXYBeamSpot",9999.0);
  desc.add<int>("MinNmuonHits",0);
  desc.add<double>("MaxPtDifference",9999.0);
  desc.add<double>("MinTrackPt",0.0);
  descriptions.add("hltMuonL3PreFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonL3PreFilter::hltFilter(Event& iEvent, const EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // get hold of trks
   Handle<RecoChargedCandidateCollection> mucands;
   if (saveTags()) filterproduct.addCollectionTag(candTag_);
   iEvent.getByLabel (candTag_,mucands);
   // sort them by L2Track
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;
   unsigned int maxI = mucands->size();
   for (unsigned int i=0;i!=maxI;++i){
     TrackRef tk = (*mucands)[i].track();
     edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
     TrackRef staTrack = l3seedRef->l2Track();
     LogDebug("HLTMuonL3PreFilter") <<"L2 from: "<<iEvent.getProvenance(staTrack.id()).moduleLabel() <<" index: "<<staTrack.key();
     L2toL3s[staTrack].push_back(RecoChargedCandidateRef(mucands,i));
   }

   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByLabel (previousCandTag_,previousLevelCands);
   BeamSpot beamSpot;
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByLabel(beamspotTag_,recoBeamSpotHandle);
   beamSpot = *recoBeamSpotHandle;


   //needed to compare to L2
   vector<RecoChargedCandidateRef> vl2cands;
   previousLevelCands->getObjects(TriggerMuon,vl2cands);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_it = L2toL3s.begin();
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_end = L2toL3s.end();
   LogDebug("HLTMuonL3PreFilter")<<"looking at: "<<L2toL3s.size()<<" L2->L3s from: "<<mucands->size();
   for (; L2toL3s_it!=L2toL3s_end; ++L2toL3s_it){

     if (!triggeredByLevel2(L2toL3s_it->first,vl2cands)) continue;
     
     //loop over the L3Tk reconstructed for this L2.
     unsigned int iTk=0;
     unsigned int maxItk=L2toL3s_it->second.size();
     for (; iTk!=maxItk; iTk++){
       
       RecoChargedCandidateRef & cand=L2toL3s_it->second[iTk];
       TrackRef tk = cand->track();

       LogDebug("HLTMuonL3PreFilter") << " Muon in loop, q*pt= " << tk->charge()*tk->pt() <<" (" << cand->charge()*cand->pt() << ") " << ", eta= " << tk->eta() << " (" << cand->eta() << ") " << ", hits= " << tk->numberOfValidHits() << ", d0= " << tk->d0() << ", dz= " << tk->dz();
       
       // eta cut
       if (fabs(cand->eta())>max_Eta_) continue;
       
       // cut on number of hits
       if (tk->numberOfValidHits()<min_Nhits_) continue;
       
       //max dr cut
       //if (fabs(tk->d0())>max_Dr_) continue;
       if (fabs( (- (cand->vx()-beamSpot.x0()) * cand->py() + (cand->vy()-beamSpot.y0()) * cand->px() ) / cand->pt() ) >max_Dr_) continue;

       //min dr cut
       if (fabs( (- (cand->vx()-beamSpot.x0()) * cand->py() + (cand->vy()-beamSpot.y0()) * cand->px() ) / cand->pt() ) <min_Dr_) continue;

       //dz cut
       if (fabs((cand->vz()-beamSpot.z0()) - ((cand->vx()-beamSpot.x0())*cand->px()+(cand->vy()-beamSpot.y0())*cand->py())/cand->pt() * cand->pz()/cand->pt())>max_Dz_) continue;

       // dxy significance cut (safeguard against bizarre values)
       if (min_DxySig_ > 0 && (tk->dxyError() <= 0 || fabs(tk->dxy(beamSpot.position())/tk->dxyError()) < min_DxySig_)) continue;

       //normalizedChi2 cut
       if (tk->normalizedChi2() > max_NormalizedChi2_ ) continue;

       //dxy beamspot cut
       if (fabs(tk->dxy(beamSpot.position())) > max_DXYBeamSpot_ ) continue;

       //min muon hits cut
       reco::HitPattern trackHits = tk->hitPattern();
       if (trackHits.numberOfValidMuonHits() < min_NmuonHits_ ) continue;
       
       //pt difference cut
       double candPt = cand->pt();
       double trackPt = tk->pt();

       if (fabs(candPt - trackPt) > max_PtDifference_ ) continue;

       //track pt cut
       if (trackPt < min_TrackPt_ ) continue;
       
       // Pt threshold cut
       double pt = cand->pt();
       double err0 = tk->error(0);
       double abspar0 = fabs(tk->parameter(0));
       double ptLx = pt;
       // convert 50% efficiency threshold to 90% efficiency threshold
       if (abspar0>0) ptLx += nsigma_Pt_*err0/abspar0*pt;
       LogTrace("HLTMuonL3PreFilter") << " ...Muon in loop, trackkRef pt= "
				      << tk->pt() << ", ptLx= " << ptLx 
				      << " cand pT " << cand->pt();
      if (ptLx<min_Pt_) continue;
      
      filterproduct.addObject(TriggerMuon,cand);
      n++;
      break; // and go on with the next L2 association
     }
   }////loop over L2s from L3 grouping

   vector<RecoChargedCandidateRef> vref;
   filterproduct.getObjects(TriggerMuon,vref);
   for (unsigned int i=0; i<vref.size(); i++ ) {
     RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
     TrackRef tk = candref->get<TrackRef>();
     LogDebug("HLTMuonL3PreFilter")
       << " Track passing filter: trackRef pt= " << tk->pt() << " (" << candref->pt() << ") " << ", eta: " << tk->eta() << " (" << candref->eta() << ") ";
   }
   
   // filter decision
   const bool accept (n >= min_N_);
   
   LogDebug("HLTMuonL3PreFilter") << " >>>>> Result of HLTMuonL3PreFilter is " << accept << ", number of muons passing thresholds= " << n; 

   return accept;
}
bool
HLTMuonL3PreFilter::triggeredByLevel2(const TrackRef& staTrack,vector<RecoChargedCandidateRef>& vcands)
{
  bool ok=false;
  for (unsigned int i=0; i<vcands.size(); i++) {
    if ( vcands[i]->get<TrackRef>() == staTrack ) {
      ok=true;
      LogDebug("HLTMuonL3PreFilter") << "The L2 track triggered";
      break;
    }
  }
  return ok;
}

