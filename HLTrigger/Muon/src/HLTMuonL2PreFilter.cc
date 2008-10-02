/** \class HLTMuonL2PreFilter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL2PreFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// constructors and destructor
//
using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;

HLTMuonL2PreFilter::HLTMuonL2PreFilter(const ParameterSet& iConfig) :
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
   saveTag_  (iConfig.getUntrackedParameter<bool> ("SaveTag",false)) 
{

   LogDebug("HLTMuonL2PreFilter")
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

HLTMuonL2PreFilter::~HLTMuonL2PreFilter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonL2PreFilter::filter(Event& iEvent, const EventSetup& iSetup)
{

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterproduct (new TriggerFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RecoChargedCandidateRef ref;

   // get hold of trks
   Handle<RecoChargedCandidateCollection> mucands;
   if(saveTag_)filterproduct->addCollectionTag(candTag_);
   iEvent.getByLabel (candTag_,mucands);
   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByLabel (previousCandTag_,previousLevelCands);
   vector<L1MuonParticleRef> vl1cands;
   previousLevelCands->getObjects(TriggerL1Mu,vl1cands);

   BeamSpot beamSpot;
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByLabel(beamspotTag_,recoBeamSpotHandle);
   beamSpot = *recoBeamSpotHandle;
  
   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   RecoChargedCandidateCollection::const_iterator cand;
   for (cand=mucands->begin(); cand!=mucands->end(); cand++) {
      TrackRef tk = cand->get<TrackRef>();

      LogDebug("HLTMuonL2PreFilter") << " Muon in loop, q*pt= " << tk->charge()*tk->pt() << ", eta= " << tk->eta() << ", hits= " << tk->numberOfValidHits() << ", d0= " << tk->d0() << ", dz= " << tk->dz();


      // find the L1 Particle corresponding to the L2 Track
      if (!triggeredByLevel1(tk,vl1cands)) continue;
      // eta cut

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
      LogTrace("HLTMuonL2PreFilter") << " ...Muon in loop, pt= "
            << pt << ", ptLx= " << ptLx;
      if (ptLx<min_Pt_) continue;

      n++;
      ref= RecoChargedCandidateRef(Ref<RecoChargedCandidateCollection>
                     (mucands,distance(mucands->begin(),cand)));
      filterproduct->addObject(TriggerMuon,ref);
   }

   vector<RecoChargedCandidateRef> vref;
   filterproduct->getObjects(TriggerMuon,vref);
   for (unsigned int i=0; i<vref.size(); i++ ) {
     RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
     TrackRef tk = candref->get<TrackRef>();
     LogDebug("HLTMuonL2PreFilter")
       << " Track passing filter: pt= " << tk->pt() << ", eta: " 
       << tk->eta();
   }
   
   // filter decision
   const bool accept (n >= min_N_);
   
   // put filter object into the Event
   iEvent.put(filterproduct);
   
   LogDebug("HLTMuonL2PreFilter") << " >>>>> Result of HLTMuonL2PreFilter is " << accept << ", number of muons passing thresholds= " << n; 

   return accept;
}

bool
HLTMuonL2PreFilter::triggeredByLevel1(TrackRef& tk,vector<L1MuonParticleRef>& vcands)
{
  bool ok=false;
  edm::Ref<L2MuonTrajectorySeedCollection> l2seedRef = tk->seedRef().castTo<edm::Ref<L2MuonTrajectorySeedCollection> >();
  l1extra::L1MuonParticleRef l1FromSeed = l2seedRef->l1Particle();
  for (unsigned int i=0; i<vcands.size(); i++) {
    if (l1FromSeed == vcands[i]){
      ok=true;
      LogTrace("HLTMuonL2PreFilter") << "The L1 stub triggered";
      break;
    }
  }
  return ok;
}
