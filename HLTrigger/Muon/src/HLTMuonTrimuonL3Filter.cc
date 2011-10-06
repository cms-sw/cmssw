/** \class HLTMuonTrimuonL3Filter
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
#include "HLTrigger/Muon/interface/HLTMuonTrimuonL3Filter.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"

#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

//
// constructors and destructor
//
HLTMuonTrimuonL3Filter::HLTMuonTrimuonL3Filter(const edm::ParameterSet& iConfig) :   beamspotTag_   (iConfig.getParameter< edm::InputTag > ("BeamSpotTag")),
   candTag_     (iConfig.getParameter< edm::InputTag > ("CandTag")),
   previousCandTag_   (iConfig.getParameter<InputTag > ("PreviousCandTag")),
    fast_Accept_ (iConfig.getParameter<bool> ("FastAccept")),
   max_Eta_     (iConfig.getParameter<double> ("MaxEta")),
   min_Nhits_   (iConfig.getParameter<int> ("MinNhits")),
   max_Dr_      (iConfig.getParameter<double> ("MaxDr")),
   max_Dz_      (iConfig.getParameter<double> ("MaxDz")),
   chargeOpt_   (iConfig.getParameter<int> ("ChargeOpt")),
   min_PtTriplet_  (iConfig.getParameter<double> ("MinPtTriplet")),
   min_PtMax_   (iConfig.getParameter<double> ("MinPtMax")),
   min_PtMin_   (iConfig.getParameter<double> ("MinPtMin")),
   min_InvMass_ (iConfig.getParameter<double> ("MinInvMass")),
   max_InvMass_ (iConfig.getParameter<double> ("MaxInvMass")),
   min_Acop_    (iConfig.getParameter<double> ("MinAcop")),
   max_Acop_    (iConfig.getParameter<double> ("MaxAcop")),
   min_PtBalance_ (iConfig.getParameter<double> ("MinPtBalance")),
   max_PtBalance_ (iConfig.getParameter<double> ("MaxPtBalance")),
   nsigma_Pt_   (iConfig.getParameter<double> ("NSigmaPt")), 
   max_DCAMuMu_  (iConfig.getParameter<double>("MaxDCAMuMu")),
   max_YTriplet_   (iConfig.getParameter<double>("MaxRapidityTriplet")),
   saveTags_  (iConfig.getParameter<bool>("saveTags"))
{

   LogDebug("HLTMuonTrimuonL3Filter")
      << " CandTag/MinN/MaxEta/MinNhits/MaxDr/MaxDz/MinPt1/MinPt2/MinInvMass/MaxInvMass/MinAcop/MaxAcop/MinPtBalance/MaxPtBalance/NSigmaPt/MaxDzMuMu/MaxRapidityTriplet : " 
      << candTag_.encode()
      << " " << fast_Accept_
      << " " << max_Eta_
      << " " << min_Nhits_
      << " " << max_Dr_
      << " " << max_Dz_
      << " " << chargeOpt_ << " " << min_PtTriplet_
      << " " << min_PtMax_ << " " << min_PtMin_
      << " " << min_InvMass_ << " " << max_InvMass_
      << " " << min_Acop_ << " " << max_Acop_
      << " " << min_PtBalance_ << " " << max_PtBalance_
      << " " << nsigma_Pt_
      << " " << max_DCAMuMu_
      << " " << max_YTriplet_;

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonTrimuonL3Filter::~HLTMuonTrimuonL3Filter()
{
}

void
HLTMuonTrimuonL3Filter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL3MuonCandidates"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  desc.add<bool>("FastAccept",false);
  desc.add<double>("MaxEta",2.5);
  desc.add<int>("MinNhits",0);
  desc.add<double>("MaxDr",2.0);
  desc.add<double>("MaxDz",9999.0);
  desc.add<int>("ChargeOpt",0);
  desc.add<double>("MinPtTriplet",0.0);
  desc.add<double>("MinPtMax",3.0);
  desc.add<double>("MinPtMin",3.0);
  desc.add<double>("MinInvMass",2.8);
  desc.add<double>("MaxInvMass",3.4);
  desc.add<double>("MinAcop",-1.0);
  desc.add<double>("MaxAcop",3.15);
  desc.add<double>("MinPtBalance",-1.0);
  desc.add<double>("MaxPtBalance",999999.0);
  desc.add<double>("NSigmaPt",0.0);
  desc.add<bool>("saveTags",false);
  desc.add<double>("MaxDCAMuMu",99999.9);
  desc.add<double>("MaxRapidityTriplet",999999.0);
  descriptions.add("hltMuonTrimuonL3Filter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonTrimuonL3Filter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
   if(saveTags_)filterproduct->addCollectionTag(candTag_);
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

   // Needed for DCA calculation
   ESHandle<MagneticField> bFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  
   // needed to compare to L2
   vector<RecoChargedCandidateRef> vl2cands;
   previousLevelCands->getObjects(TriggerMuon,vl2cands);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   double e1,e2,e3;
   Particle::LorentzVector p,p1,p2,p3;

   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_it1 = L2toL3s.begin();
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_end = L2toL3s.end();
   bool atLeastOneTriplet=false;
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
       LogDebug("HLTMuonTrimuonL3Filter") << " 1st muon in loop: q*pt= "
					 << tk1->charge()*tk1->pt() << " (" << cand1->charge()*cand1->pt()<< ") " << ", eta= " << tk1->eta() << " (" << cand1->eta() << ") " << ", hits= " << tk1->numberOfValidHits();

       if (fabs(cand1->eta())>max_Eta_) continue;
       
       // cut on number of hits
       if (tk1->numberOfValidHits()<min_Nhits_) continue;
       
       //dr cut
       //      if (fabs(tk1->d0())>max_Dr_) continue;
       if (fabs( (- (cand1->vx()-beamSpot.x0()) * cand1->py() + (cand1->vy()-beamSpot.y0()) * cand1->px() ) / cand1->pt() ) >max_Dr_) continue;
       
       //dz cut
       if (fabs((cand1->vz()-beamSpot.z0()) - ((cand1->vx()-beamSpot.x0())*cand1->px()+(cand1->vy()-beamSpot.y0())*cand1->py())/cand1->pt() * cand1->pz()/cand1->pt())>max_Dz_) continue;
       
       // Pt threshold cut
       double pt1 = cand1->pt();
       //       double err1 = tk1->error(0);
       //       double abspar1 = fabs(tk1->parameter(0));
       double ptLx1 = pt1;
       // Don't convert to 90% efficiency threshold
       LogDebug("HLTMuonTrimuonL3Filter") << " ... 1st muon in loop, pt1= "
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
	      LogDebug("HLTMuonTrimuonL3Filter") << " 2nd muon in loop: q*pt= " << tk2->charge()*tk2->pt() << " (" << cand2->charge()*cand2->pt() << ") " << ", eta= " << tk2->eta() << " (" << cand2->eta() << ") " << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0() ;
	      if (fabs(cand2->eta())>max_Eta_) continue;
	      
	      // cut on number of hits
	      if (tk2->numberOfValidHits()<min_Nhits_) continue;
	      
	      //dr cut
	      // if (fabs(tk2->d0())>max_Dr_) continue;
	      if (fabs( (- (cand2->vx()-beamSpot.x0()) * cand2->py() + (cand2->vy()-beamSpot.y0()) * cand2->px() ) / cand2->pt() ) >max_Dr_) continue;
	      
	      //dz cut
	      if (fabs((cand2->vz()-beamSpot.z0()) - ((cand2->vx()-beamSpot.x0())*cand2->px()+(cand2->vy()-beamSpot.y0())*cand2->py())/cand2->pt() * cand2->pz()/cand2->pt())>max_Dz_) continue;	      
	      
	      // Pt threshold cut
	      double pt2 = cand2->pt();
	      //	      double err2 = tk2->error(0);
	      //	      double abspar2 = fabs(tk2->parameter(0));
	      double ptLx2 = pt2;
	      // Don't convert to 90% efficiency threshold
	      LogDebug("HLTMuonTrimuonL3Filter") << " ... 2nd muon in loop, pt2= "
						 << pt2 << ", ptLx2= " << ptLx2;
	      
	      std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > ::iterator L2toL3s_it3 = L2toL3s_it2;
	      L2toL3s_it3++;
	      for (; L2toL3s_it3!=L2toL3s_end; ++L2toL3s_it3){
		
		if (!triggeredByLevel2(L2toL3s_it3->first,vl2cands)) continue;
		
		//loop over the L3Tk reconstructed for this L2.
		unsigned int iTk3=0;
		unsigned int maxItk3=L2toL3s_it3->second.size();
		for (; iTk3!=maxItk3; iTk3++){
		  RecoChargedCandidateRef & cand3=L2toL3s_it3->second[iTk3];
		  TrackRef tk3 = cand3->get<TrackRef>();
		  // eta cut
		  LogDebug("HLTMuonTrimuonL3Filter") << " 3rd muon in loop: q*pt= "
						     << tk3->charge()*tk3->pt() << " (" << cand3->charge()*cand3->pt()<< ") " << ", eta= " << tk3->eta() << " (" << cand3->eta() << ") " << ", hits= " << tk3->numberOfValidHits();
		  
		  if (fabs(cand3->eta())>max_Eta_) continue;
		  
		  // cut on number of hits
		  if (tk3->numberOfValidHits()<min_Nhits_) continue;
		  
		  //dr cut
		  //      if (fabs(tk1->d0())>max_Dr_) continue;
		  if (fabs( (- (cand3->vx()-beamSpot.x0()) * cand3->py() + (cand3->vy()-beamSpot.y0()) * cand3->px() ) / cand3->pt() ) >max_Dr_) continue;
		  
		  //dz cut
		  if (fabs((cand3->vz()-beamSpot.z0()) - ((cand3->vx()-beamSpot.x0())*cand3->px()+(cand3->vy()-beamSpot.y0())*cand3->py())/cand3->pt() * cand3->pz()/cand3->pt())>max_Dz_) continue;
		  
		  // Pt threshold cut
		  double pt3 = cand3->pt();
		  //       double err3 = tk3->error(0);
		  //       double abspar3 = fabs(tk3->parameter(0));
		  double ptLx3 = pt3;
		  // Don't convert to 90% efficiency threshold
		  LogDebug("HLTMuonTrimuonL3Filter") << " ... 3rd muon in loop, pt3= "
						     << pt3 << ", ptLx3= " << ptLx3;
		  
		  if (ptLx1>ptLx2 && ptLx1>ptLx3 && ptLx1<min_PtMax_) continue;
		  else if (ptLx2>ptLx1 && ptLx2>ptLx3 && ptLx2<min_PtMax_) continue;
                  else if (ptLx3<ptLx2 && ptLx3>ptLx1 && ptLx3<min_PtMax_) continue;

		  if (ptLx1<ptLx2 && ptLx1<ptLx3 && ptLx1<min_PtMin_) continue;
		  else if (ptLx2<ptLx1 && ptLx2<ptLx3 && ptLx2<min_PtMin_) continue;
                  else if (ptLx3<ptLx2 && ptLx3<ptLx1 && ptLx3<min_PtMin_) continue;
		  
		  if (chargeOpt_>0) {
		    if (abs (cand1->charge()+cand2->charge()+cand3->charge()) != chargeOpt_) continue;
		  }
		  
		  // Acoplanarity
		  double acop = fabs(cand1->phi()-cand2->phi());
		  if (acop>M_PI) acop = 2*M_PI - acop;
		  acop = M_PI - acop;
		  LogDebug("HLTMuonTrimuonL3Filter") << " ... 1-2 acop= " << acop;
		  if (acop<min_Acop_) continue;
		  if (acop>max_Acop_) continue;

		  acop = fabs(cand1->phi()-cand3->phi());
		  if (acop>M_PI) acop = 2*M_PI - acop;
		  acop = M_PI - acop;
		  LogDebug("HLTMuonTrimuonL3Filter") << " ... 1-3 acop= " << acop;
		  if (acop<min_Acop_) continue;
		  if (acop>max_Acop_) continue;

		  acop = fabs(cand3->phi()-cand2->phi());
		  if (acop>M_PI) acop = 2*M_PI - acop;
		  acop = M_PI - acop;
		  LogDebug("HLTMuonTrimuonL3Filter") << " ... 3-2 acop= " << acop;
		  if (acop<min_Acop_) continue;
		  if (acop>max_Acop_) continue;
		  
		  // Pt balance
		  double ptbalance = fabs(cand1->pt()-cand2->pt());
		  if (ptbalance<min_PtBalance_) continue;
		  if (ptbalance>max_PtBalance_) continue;
		  ptbalance = fabs(cand1->pt()-cand3->pt());
		  if (ptbalance<min_PtBalance_) continue;
		  if (ptbalance>max_PtBalance_) continue;
		  ptbalance = fabs(cand3->pt()-cand2->pt());
		  if (ptbalance<min_PtBalance_) continue;
		  if (ptbalance>max_PtBalance_) continue;
		  
		  // Combined trimuon system
		  e1 = sqrt(cand1->momentum().Mag2()+MuMass2);
		  e2 = sqrt(cand2->momentum().Mag2()+MuMass2);
		  e3 = sqrt(cand3->momentum().Mag2()+MuMass2);
		  p1 = Particle::LorentzVector(cand1->px(),cand1->py(),cand1->pz(),e1);
		  p2 = Particle::LorentzVector(cand2->px(),cand2->py(),cand2->pz(),e2);
		  p3 = Particle::LorentzVector(cand3->px(),cand3->py(),cand3->pz(),e3);
		  p = p1+p2+p3;
		  
		  double pt123 = p.pt();
		  LogDebug("HLTMuonTrimuonL3Filter") << " ... 1-2 pt123= " << pt123;
		  if (pt123<min_PtTriplet_) continue;
		  
		  double invmass = abs(p.mass());
		  // if (invmass>0) invmass = sqrt(invmass); else invmass = 0;
		  LogDebug("HLTMuonTrimuonL3Filter") << " ... 1-2 invmass= " << invmass;
		  if (invmass<min_InvMass_) continue;
		  if (invmass>max_InvMass_) continue;
		  
		  // Delta Z between the two muons
		  //double DeltaZMuMu = fabs(tk2->dz(beamSpot.position())-tk1->dz(beamSpot.position()));
		  //if ( DeltaZMuMu > max_DzMuMu_) continue;
		  
		  // DCA between the three muons
		  TransientTrack mu1TT(*tk1, &(*bFieldHandle));
		  TransientTrack mu2TT(*tk2, &(*bFieldHandle));
		  TransientTrack mu3TT(*tk3, &(*bFieldHandle));
		  TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
		  TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
		  TrajectoryStateClosestToPoint mu3TS = mu3TT.impactPointTSCP();
		  if (mu1TS.isValid() && mu2TS.isValid() && mu3TS.isValid()) {
		    ClosestApproachInRPhi cApp;
		    cApp.calculate(mu1TS.theState(), mu2TS.theState());
		    if (!cApp.status()
			|| cApp.distance() > max_DCAMuMu_) continue;
		    cApp.calculate(mu1TS.theState(), mu3TS.theState());
		    if (!cApp.status()
			|| cApp.distance() > max_DCAMuMu_) continue;
		    cApp.calculate(mu3TS.theState(), mu2TS.theState());
		    if (!cApp.status()
			|| cApp.distance() > max_DCAMuMu_) continue;
		  }
		  
		  // Max dimuon |rapidity|
		  double rapidity = fabs(p.Rapidity());
		  if ( rapidity > max_YTriplet_) continue;
		  
		  // Add this triplet
		  n++;
		  LogDebug("HLTMuonTrimuonL3Filter") << " Track1 passing filter: pt= " << cand1->pt() << ", eta: " << cand1->eta();
		  LogDebug("HLTMuonTrimuonL3Filter") << " Track2 passing filter: pt= " << cand2->pt() << ", eta: " << cand2->eta();
		  LogDebug("HLTMuonTrimuonL3Filter") << " Track2 passing filter: pt= " << cand3->pt() << ", eta: " << cand3->eta();
		  LogDebug("HLTMuonTrimuonL3Filter") << " Invmass= " << invmass;
		  
		  bool i1done = false;
		  bool i2done = false;
		  bool i3done = false;
		  vector<RecoChargedCandidateRef> vref;
		  filterproduct->getObjects(TriggerMuon,vref);
		  for (unsigned int i=0; i<vref.size(); i++) {
		    RecoChargedCandidateRef candref =  RecoChargedCandidateRef(vref[i]);
		    TrackRef tktmp = candref->get<TrackRef>();
		    if (tktmp==tk1) {
		      i1done = true;
		    } else if (tktmp==tk2) {
		      i2done = true;
		    } else if (tktmp==tk3) {
		      i3done = true;
		    }
		    if (i1done && i2done && i3done) break;
		  }
		  
		  if (!i1done) { 
		    filterproduct->addObject(TriggerMuon,cand1);
		  }
		  if (!i2done) { 
		    filterproduct->addObject(TriggerMuon,cand2);
		  }
		  if (!i3done) { 
		    filterproduct->addObject(TriggerMuon,cand3);
		  }
		  
		  //break anyway since a L3 track triplet has been found matching the criteria
		  thisL3Index1isDone=true;
		  atLeastOneTriplet=true;
		  break;
		}//loop on the track of the third L2
		//break the loop if fast accept.
		if (atLeastOneTriplet && fast_Accept_) break;
	      }//loop on the third L2
	    }//loop on the track of the second L2
	    //break the loop if fast accept.
	    if (atLeastOneTriplet && fast_Accept_) break;
       }//loop on the second L2
       
       
       //break the loop if fast accept.
       if (atLeastOneTriplet && fast_Accept_) break;
       if (thisL3Index1isDone) break;
     }//loop on tracks for first L2
     //break the loop if fast accept.
     if (atLeastOneTriplet && fast_Accept_) break;
   }//loop on the first L2
   
   // filter decision
   const bool accept (n >= 1);
   
   // put filter object into the Event
   iEvent.put(filterproduct);
   
   LogDebug("HLTMuonTrimuonL3Filter") << " >>>>> Result of HLTMuonTrimuonL3Filter is "<< accept << ", number of muon triplets passing thresholds= " << n; 
   
   return accept;
}


bool
HLTMuonTrimuonL3Filter::triggeredByLevel2(const TrackRef& staTrack,vector<RecoChargedCandidateRef>& vcands)
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
