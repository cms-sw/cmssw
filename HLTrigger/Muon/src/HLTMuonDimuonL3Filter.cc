/** \class HLTMuonDimuonL2Filter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz, P. Garcia
 *
 */


#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonL3Filter.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

//
// constructors and destructor
//
HLTMuonDimuonL3Filter::HLTMuonDimuonL3Filter(const edm::ParameterSet& iConfig) :   beamspotTag_   (iConfig.getParameter< edm::InputTag > ("BeamSpotTag")),
   candTag_     (iConfig.getParameter< edm::InputTag > ("CandTag")),
   previousCandTag_   (iConfig.getParameter<InputTag > ("PreviousCandTag")),
    fast_Accept_ (iConfig.getParameter<bool> ("FastAccept")),
   max_Eta_     (iConfig.getParameter<double> ("MaxEta")),
   min_Nhits_   (iConfig.getParameter<int> ("MinNhits")),
   max_Dr_      (iConfig.getParameter<double> ("MaxDr")),
   max_Dz_      (iConfig.getParameter<double> ("MaxDz")),
   chargeOpt_   (iConfig.getParameter<int> ("ChargeOpt")),
   min_PtPair_  (iConfig.getParameter<double> ("MinPtPair")),
   min_PtMax_   (iConfig.getParameter<double> ("MinPtMax")),
   min_PtMin_   (iConfig.getParameter<double> ("MinPtMin")),
   min_InvMass_ (iConfig.getParameter<double> ("MinInvMass")),
   max_InvMass_ (iConfig.getParameter<double> ("MaxInvMass")),
   min_Acop_    (iConfig.getParameter<double> ("MinAcop")),
   max_Acop_    (iConfig.getParameter<double> ("MaxAcop")),
   min_PtBalance_ (iConfig.getParameter<double> ("MinPtBalance")),
   max_PtBalance_ (iConfig.getParameter<double> ("MaxPtBalance")),
   nsigma_Pt_   (iConfig.getParameter<double> ("NSigmaPt")), 
   max_DzMuMu_  (iConfig.getParameter<double>("MaxDzMuMu")),
   max_YPair_   (iConfig.getParameter<double>("MaxRapidityPair")),
   saveTag_  (iConfig.getUntrackedParameter<bool> ("SaveTag",false)) 
{

   LogDebug("HLTMuonDimuonL3Filter")
      << " CandTag/MinN/MaxEta/MinNhits/MaxDr/MaxDz/MinPt1/MinPt2/MinInvMass/MaxInvMass/MinAcop/MaxAcop/MinPtBalance/MaxPtBalance/NSigmaPt/MaxDzMuMu/MaxRapidityPair : " 
      << candTag_.encode()
      << " " << fast_Accept_
      << " " << max_Eta_
      << " " << min_Nhits_
      << " " << max_Dr_
      << " " << max_Dz_
      << " " << chargeOpt_ << " " << min_PtPair_
      << " " << min_PtMax_ << " " << min_PtMin_
      << " " << min_InvMass_ << " " << max_InvMass_
      << " " << min_Acop_ << " " << max_Acop_
      << " " << min_PtBalance_ << " " << max_PtBalance_
      << " " << nsigma_Pt_
      << " " << max_DzMuMu_
      << " " << max_YPair_;

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonDimuonL3Filter::~HLTMuonDimuonL3Filter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonDimuonL3Filter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   double const MuMass = 0.106;
   double const MuMass2 = MuMass*MuMass;
   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterproduct (new TriggerFilterObjectWithRefs(path(),module()));

   // get hold of trks
   Handle<RecoChargedCandidateCollection> mucands;
   if(saveTag_)filterproduct->addCollectionTag(candTag_);
   iEvent.getByLabel (candTag_,mucands);
   // sort them by L2Track
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;
   unsigned int maxI = mucands->size();
   for (unsigned int i=0;i!=maxI;i++){
     TrackRef tk = (*mucands)[i].track();
     edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
     TrackRef staTrack = l3seedRef->l2Track();
     L2toL3s[staTrack].push_back(RecoChargedCandidateRef(mucands,i));
   }

   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByLabel (previousCandTag_,previousLevelCands);
   BeamSpot beamSpot;
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByLabel(beamspotTag_,recoBeamSpotHandle);
   beamSpot = *recoBeamSpotHandle;
  
   // needed to compare to L2
   vector<RecoChargedCandidateRef> vl2cands;
   previousLevelCands->getObjects(TriggerMuon,vl2cands);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   double e1,e2;
   Particle::LorentzVector p,p1,p2;

   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_it1 = L2toL3s.begin();
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_end = L2toL3s.end();
   bool atLeastOnePair=false;
   for (; L2toL3s_it1!=L2toL3s_end; ++L2toL3s_it1){

     if (!triggeredByLevel2(L2toL3s_it1->first,vl2cands)) continue;
     
     //loop over the L3Tk reconstructed for this L2.
     unsigned int iTk1=0;
     unsigned int maxItk1=L2toL3s_it1->second.size();
     for (; iTk1!=maxItk1; iTk1++){
       bool thisL3Index1isDone=false;
       RecoChargedCandidateRef & cand1=L2toL3s_it1->second[iTk1];
       TrackRef tk1 = cand1->get<TrackRef>();
       // eta cut
       LogDebug("HLTMuonDimuonL3Filter") << " 1st muon in loop: q*pt= "
					 << tk1->charge()*tk1->pt() << ", eta= " << tk1->eta() << ", hits= " << tk1->numberOfValidHits();

       if (fabs(tk1->eta())>max_Eta_) continue;
       
       // cut on number of hits
       if (tk1->numberOfValidHits()<min_Nhits_) continue;
       
       //dr cut
       //      if (fabs(tk1->d0())>max_Dr_) continue;
       if (fabs(tk1->dxy(beamSpot.position()))>max_Dr_) continue;
       
       //dz cut
       if (fabs(tk1->dz())>max_Dz_) continue;
       
       // Pt threshold cut
       double pt1 = tk1->pt();
       double err1 = tk1->error(0);
       double abspar1 = fabs(tk1->parameter(0));
       double ptLx1 = pt1;
       // convert 50% efficiency threshold to 90% efficiency threshold
       if (abspar1>0) ptLx1 += nsigma_Pt_*err1/abspar1*pt1;
       LogDebug("HLTMuonDimuonL3Filter") << " ... 1st muon in loop, pt1= "
					 << pt1 << ", ptLx1= " << ptLx1;
       std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_it2 = L2toL3s_it1;
       L2toL3s_it2++;
       for (; L2toL3s_it2!=L2toL3s_end; ++L2toL3s_it2){
	 if (!triggeredByLevel2(L2toL3s_it2->first,vl2cands)) continue;
	 
	    //loop over the L3Tk reconstructed for this L2.
	    unsigned int iTk2=0;
	    unsigned int maxItk2=L2toL3s_it2->second.size();
	    for (; iTk2!=maxItk2; iTk2++){
	      RecoChargedCandidateRef & cand2=L2toL3s_it2->second[iTk2];
	      TrackRef tk2 = cand2->get<TrackRef>();
	      
	      // eta cut
	      LogDebug("HLTMuonDimuonL3Filter") << " 2nd muon in loop: q*pt= " << tk2->charge()*tk2->pt() << ", eta= " << tk2->eta() << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0();
	      if (fabs(tk2->eta())>max_Eta_) continue;
	      
	      // cut on number of hits
	      if (tk2->numberOfValidHits()<min_Nhits_) continue;
	      
	      //dr cut
	      // if (fabs(tk2->d0())>max_Dr_) continue;
	      if (fabs(tk2->dxy(beamSpot.position()))>max_Dr_) continue;
	      
	      //dz cut
	      if (fabs(tk2->dz())>max_Dz_) continue;
	      
	      // Pt threshold cut
	      double pt2 = tk2->pt();
	      double err2 = tk2->error(0);
	      double abspar2 = fabs(tk2->parameter(0));
	      double ptLx2 = pt2;
	      // convert 50% efficiency threshold to 90% efficiency threshold
	      if (abspar2>0) ptLx2 += nsigma_Pt_*err2/abspar2*pt2;
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 2nd muon in loop, pt2= "
						<< pt2 << ", ptLx2= " << ptLx2;
	      
	      if (ptLx1>ptLx2) {
		if (ptLx1<min_PtMax_) continue;
		if (ptLx2<min_PtMin_) continue;
	      } else {
		if (ptLx2<min_PtMax_) continue;
		if (ptLx1<min_PtMin_) continue;
	      }
	      
	      if (chargeOpt_<0) {
		if (tk1->charge()*tk2->charge()>0) continue;
	      } else if (chargeOpt_>0) {
		if (tk1->charge()*tk2->charge()<0) continue;
	      }
	      
	      // Acoplanarity
	      double acop = fabs(tk1->phi()-tk2->phi());
	      if (acop>M_PI) acop = 2*M_PI - acop;
	      acop = M_PI - acop;
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 acop= " << acop;
	      if (acop<min_Acop_) continue;
	      if (acop>max_Acop_) continue;

	      // Pt balance
	      double ptbalance = fabs(tk1->pt()-tk2->pt());
	      if (ptbalance<min_PtBalance_) continue;
	      if (ptbalance>max_PtBalance_) continue;

	      // Combined dimuon system
	      e1 = sqrt(tk1->momentum().Mag2()+MuMass2);
	      e2 = sqrt(tk2->momentum().Mag2()+MuMass2);
	      p1 = Particle::LorentzVector(tk1->px(),tk1->py(),tk1->pz(),e1);
	      p2 = Particle::LorentzVector(tk2->px(),tk2->py(),tk2->pz(),e2);
	      p = p1+p2;
	      
	      double pt12 = p.pt();
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 pt12= " << pt12;
	      if (pt12<min_PtPair_) continue;
	      
	      double invmass = abs(p.mass());
	      // if (invmass>0) invmass = sqrt(invmass); else invmass = 0;
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 invmass= " << invmass;
	      if (invmass<min_InvMass_) continue;
	      if (invmass>max_InvMass_) continue;

              // Delta Z between the two muons
              double DeltaZMuMu = fabs(tk2->dz(beamSpot.position())-tk1->dz(beamSpot.position()));
              if ( DeltaZMuMu > max_DzMuMu_) continue;
              
              // Max dimuon |rapidity|
              double rapidity = fabs(p.Rapidity());
              if ( rapidity > max_YPair_) continue;
              
	      // Add this pair
	      n++;
	      LogDebug("HLTMuonDimuonL3Filter") << " Track1 passing filter: pt= " << tk1->pt() << ", eta: " << tk1->eta();
	      LogDebug("HLTMuonDimuonL3Filter") << " Track2 passing filter: pt= " << tk2->pt() << ", eta: " << tk2->eta();
	      LogDebug("HLTMuonDimuonL3Filter") << " Invmass= " << invmass;

	      bool i1done = false;
	      bool i2done = false;
	      vector<RecoChargedCandidateRef> vref;
	      filterproduct->getObjects(TriggerMuon,vref);
	      for (unsigned int i=0; i<vref.size(); i++) {
		RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
		TrackRef tktmp = candref->get<TrackRef>();
		if (tktmp==tk1) {
		  i1done = true;
		} else if (tktmp==tk2) {
		  i2done = true;
		}
		if (i1done && i2done) break;
	      }
	    
	      if (!i1done) { 
		filterproduct->addObject(TriggerMuon,cand1);
	      }
	      if (!i2done) { 
		filterproduct->addObject(TriggerMuon,cand2);
	      }
	      
	      //break anyway since a L3 track pair has been found matching the criteria
	      thisL3Index1isDone=true;
	      atLeastOnePair=true;
	      break;
	    }//loop on the track of the second L2
	    //break the loop if fast accept.
	    if (atLeastOnePair && fast_Accept_) break;
       }//loop on the second L2

       
       //break the loop if fast accept.
       if (atLeastOnePair && fast_Accept_) break;
       if (thisL3Index1isDone) break;
     }//loop on tracks for first L2
     //break the loop if fast accept.
     if (atLeastOnePair && fast_Accept_) break;
   }//loop on the first L2
   
   // filter decision
   const bool accept (n >= 1);

   // put filter object into the Event
   iEvent.put(filterproduct);

   LogDebug("HLTMuonDimuonL3Filter") << " >>>>> Result of HLTMuonDimuonL3Filter is "<< accept << ", number of muon pairs passing thresholds= " << n; 

   return accept;
}


bool
HLTMuonDimuonL3Filter::triggeredByLevel2(const TrackRef& staTrack,vector<RecoChargedCandidateRef>& vcands)
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
