/** \class HLTMuonL3PreFilter
 *
 * See header file for documentation
 *
 *  \author J-R Vlimant
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL3TkPreFilter.h"

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

HLTMuonL3TkPreFilter::HLTMuonL3TkPreFilter(const ParameterSet& iConfig) :
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
   saveTag_  (iConfig.getUntrackedParameter<bool> ("SaveTag",true)) 
{

   LogDebug("HLTMuonL3TkPreFilter")
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

HLTMuonL3TkPreFilter::~HLTMuonL3TkPreFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonL3TkPreFilter::filter(Event& iEvent, const EventSetup& iSetup)
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
   if(saveTag_)filterproduct->addCollectionTag(candTag_);
   // sort them by L2Track
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;
   uint n = 0;
   uint maxN = mucands->size();
   for (;n!=maxN;n++){
     TrackRef tk = (*mucands)[n].track();
     edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
     TrackRef staTrack = l3seedRef->l2Track();
     L2toL3s[staTrack].push_back(RecoChargedCandidateRef(mucands,n));
   }

   // additionnal objects needed
   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByLabel (previousCandTag_,previousLevelCands);
   BeamSpot beamSpot;
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByLabel(beamspotTag_,recoBeamSpotHandle);
   beamSpot = *recoBeamSpotHandle;


   //needed to compare to L2
   vector<RecoChargedCandidateRef> vl2cands;
   previousLevelCands->getObjects(TriggerMuon,vl2cands);

   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_it = L2toL3s.begin();
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_end = L2toL3s.end();
   for (; L2toL3s_it!=L2toL3s_end; ++L2toL3s_it){
     
     if (!triggeredAtL2(L2toL3s_it->first,vl2cands)) continue;
     
     //loop over the L3Tk reconstructed for this L2.
     uint iTk=0;
     uint maxItk=L2toL3s_it->second.size();
     for (; iTk!=maxItk; iTk++){
       
       RecoChargedCandidateRef & cand=L2toL3s_it->second[iTk];
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
      LogTrace("HLTMuonL3TkPreFilter") << " ...Muon in loop, pt= "
            << pt << ", ptLx= " << ptLx;
      if (ptLx<min_Pt_) continue;

      //one good L3Tk
      filterproduct->addObject(TriggerMuon,cand);      
      break; // and go on with the next L2 association
     }

   }//loop over L2s from L3 grouping


   vector<RecoChargedCandidateRef> vref;
   filterproduct->getObjects(TriggerMuon,vref);
   for (unsigned int i=0; i<vref.size(); i++ ) {
     TrackRef tk = vref[i]->track();
     LogDebug("HLTMuonL3TkPreFilter")
       << " Track passing filter: pt= " << tk->pt() << ", eta: " 
       << tk->eta();
   }
   
   // filter decision
   const bool accept ((int)n >= min_N_);
   
   // put filter object into the Event
   iEvent.put(filterproduct);
   
   LogDebug("HLTMuonL3TkPreFilter") << " >>>>> Result of HLTMuonL3TkPreFilter is " << accept << ", number of muons passing thresholds= " << n; 
   
   return accept;
}
bool
HLTMuonL3TkPreFilter::triggeredAtL2(const TrackRef& staTrack,std::vector<RecoChargedCandidateRef>& vcands)
{
  //FIXME: L1 seeding cannot be filtered with this.
  bool ok=false;

  // compare to previously triggered L2
  for (unsigned int i=0; i<vcands.size(); i++) {
    RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vcands[i]);
    TrackRef tk = candref->get<TrackRef>();
    if ( tk == staTrack ) {
      ok=true;
      LogDebug("HLTMuonL3TkPreFilter") << "The L2 track triggered";
      break;
    }
  }
  return ok;
}

