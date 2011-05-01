/** \class HLTMuonL3PreFilter
 *
 * See header file for documentation
 *
 *  \author J-R Vlimant
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL1toL3TkPreFilter.h"

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

//
// constructors and destructor
//
using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;

HLTMuonL1toL3TkPreFilter::HLTMuonL1toL3TkPreFilter(const ParameterSet& iConfig) :
   beamspotTag_   (iConfig.getParameter< edm::InputTag > ("BeamSpotTag")),
   candTag_   (iConfig.getParameter<InputTag > ("CandTag")),
   previousCandTag_   (iConfig.getParameter<InputTag > ("PreviousCandTag")),
   min_N_     (iConfig.getParameter<int> ("MinN")),
   max_Eta_   (iConfig.getParameter<double> ("MaxEta")),
   min_Nhits_ (iConfig.getParameter<int> ("MinNhits")),
   max_Dr_    (iConfig.getParameter<double> ("MaxDr")),
   max_Dz_    (iConfig.getParameter<double> ("MaxDz")),
   min_Pt_    (iConfig.getParameter<double> ("MinPt")),
   nsigma_Pt_  (iConfig.getParameter<double> ("NSigmaPt")), 
   saveTags_  (iConfig.getParameter<bool>("saveTags")) 
{

   LogDebug("HLTMuonL1toL3TkPreFilter")
      << " CandTag/MinN/MaxEta/MinNhits/MaxDr/MaxDz/MinPt/NSigmaPt : " 
      << candTag_.encode()
      << " " << min_N_ 
      << " " << max_Eta_
      << " " << min_Nhits_
      << " " << max_Dr_
      << " " << max_Dz_
      << " " << min_Pt_
      << " " << nsigma_Pt_;

   //register your products
   produces<TriggerFilterObjectWithRefs>();
}

HLTMuonL1toL3TkPreFilter::~HLTMuonL1toL3TkPreFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonL1toL3TkPreFilter::filter(Event& iEvent, const EventSetup& iSetup)
{

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterproduct (new TriggerFilterObjectWithRefs(path(),module()));

   // get hold of trks
   //   Handle<reco::TrackCollection> mucands;
   Handle<RecoChargedCandidateCollection> mucands;
   iEvent.getByLabel(candTag_,mucands);
   if(saveTags_)filterproduct->addCollectionTag(candTag_);
   // sort them by L2Track
   std::map<l1extra::L1MuonParticleRef, std::vector<RecoChargedCandidateRef> > L1toL3s;
   unsigned int n = 0;
   unsigned int maxN = mucands->size();
   for (;n!=maxN;n++){
     TrackRef tk = (*mucands)[n].track();
     edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
     l1extra::L1MuonParticleRef l1mu = l3seedRef->l1Particle();
       L1toL3s[l1mu].push_back(RecoChargedCandidateRef(mucands,n));
   }

   // additionnal objects needed
   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByLabel (previousCandTag_,previousLevelCands);
   BeamSpot beamSpot;
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByLabel(beamspotTag_,recoBeamSpotHandle);
   beamSpot = *recoBeamSpotHandle;


   //needed to compare to L1 
   vector<l1extra::L1MuonParticleRef> vl1cands;
   previousLevelCands->getObjects(TriggerL1Mu,vl1cands);

   std::map<l1extra::L1MuonParticleRef, std::vector<RecoChargedCandidateRef> > ::iterator L1toL3s_it = L1toL3s.begin();
   std::map<l1extra::L1MuonParticleRef, std::vector<RecoChargedCandidateRef> > ::iterator L1toL3s_end = L1toL3s.end();
   for (; L1toL3s_it!=L1toL3s_end; ++L1toL3s_it){
     
     if (!triggeredAtL1(L1toL3s_it->first,vl1cands)) continue;
     
     //loop over the L3Tk reconstructed for this L1.
     unsigned int iTk=0;
     unsigned int maxItk=L1toL3s_it->second.size();
     for (; iTk!=maxItk; iTk++){
       
       RecoChargedCandidateRef & cand=L1toL3s_it->second[iTk];
       TrackRef tk = cand->track();

      if (fabs(tk->eta())>max_Eta_) continue;

      // cut on number of hits
      if (tk->numberOfValidHits()<min_Nhits_) continue;

      //dr cut
      //if (fabs(tk->d0())>max_Dr_) continue;
      if (fabs(tk->dxy(beamSpot.position()))>max_Dr_) continue;

      //dz cut
      if (fabs(tk->dz())>max_Dz_) continue;

      // Pt threshold cut
      double pt = tk->pt();
      double err0 = tk->error(0);
      double abspar0 = fabs(tk->parameter(0));
      double ptLx = pt;
      // convert 50% efficiency threshold to 90% efficiency threshold
      if (abspar0>0) ptLx += nsigma_Pt_*err0/abspar0*pt;
      LogTrace("HLTMuonL1toL3TkPreFilter") << " ...Muon in loop, pt= "
            << pt << ", ptLx= " << ptLx;
      if (ptLx<min_Pt_) continue;

      //one good L3Tk
      filterproduct->addObject(TriggerMuon,cand);      
      break; // and go on with the next L1 association
     }

   }//loop over L1s from L3 grouping


   vector<RecoChargedCandidateRef> vref;
   filterproduct->getObjects(TriggerMuon,vref);
   for (unsigned int i=0; i<vref.size(); i++ ) {
     TrackRef tk = vref[i]->track();
     LogDebug("HLTMuonL1toL3TkPreFilter")
       << " Track passing filter: pt= " << tk->pt() << ", eta: " 
       << tk->eta();
   }
   
   // filter decision
   const bool accept ((int)n >= min_N_);
   
   // put filter object into the Event
   iEvent.put(filterproduct);
   
   LogDebug("HLTMuonL1toL3TkPreFilter") << " >>>>> Result of HLTMuonL1toL3TkPreFilter is " << accept << ", number of muons passing thresholds= " << n; 
   
   return accept;
}
bool
HLTMuonL1toL3TkPreFilter::triggeredAtL1(const l1extra::L1MuonParticleRef & l1mu,std::vector<l1extra::L1MuonParticleRef>& vcands)
{
  bool ok=false;

  // compare to previously triggered L1
  for (unsigned int i=0; i<vcands.size(); i++) {
    //    l1extra::L1MuonParticleRef candref =  L1MuonParticleRef(vcands[i]);
    if (vcands[i] == l1mu){
      ok=true;
      LogDebug("HLTMuonL1toL3TkPreFilter") << "The L1 mu triggered";
      break;}
  }
  return ok;
}

