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
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

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
HLTMuonDimuonL3Filter::HLTMuonDimuonL3Filter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
   beamspotTag_       (iConfig.getParameter< edm::InputTag > ("BeamSpotTag")),
   beamspotToken_     (consumes<reco::BeamSpot>(beamspotTag_)),
   candTag_           (iConfig.getParameter< edm::InputTag > ("CandTag")),
   candToken_         (consumes<reco::RecoChargedCandidateCollection>(candTag_)),
   previousCandTag_   (iConfig.getParameter<InputTag > ("PreviousCandTag")),
   previousCandToken_ (consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
   previousCandIsL2_(iConfig.getParameter<bool> ("PreviousCandIsL2")),
   fast_Accept_ (iConfig.getParameter<bool> ("FastAccept")),
   min_N_       (iConfig.getParameter<int> ("MinN")),
   max_Eta_     (iConfig.getParameter<double> ("MaxEta")),
   min_Nhits_   (iConfig.getParameter<int> ("MinNhits")),
   max_Dr_      (iConfig.getParameter<double> ("MaxDr")),
   max_Dz_      (iConfig.getParameter<double> ("MaxDz")),
   chargeOpt_   (iConfig.getParameter<int> ("ChargeOpt")),
   min_PtPair_  (iConfig.getParameter< vector<double> > ("MinPtPair")),
   max_PtPair_  (iConfig.getParameter< vector<double> > ("MaxPtPair")),
   min_PtMax_   (iConfig.getParameter< vector<double> > ("MinPtMax")),
   min_PtMin_   (iConfig.getParameter< vector<double> > ("MinPtMin")),
   max_PtMin_   (iConfig.getParameter< vector<double> > ("MaxPtMin")),
   min_InvMass_ (iConfig.getParameter< vector<double> > ("MinInvMass")),
   max_InvMass_ (iConfig.getParameter< vector<double> > ("MaxInvMass")),
   min_Acop_    (iConfig.getParameter<double> ("MinAcop")),
   max_Acop_    (iConfig.getParameter<double> ("MaxAcop")),
   min_PtBalance_ (iConfig.getParameter<double> ("MinPtBalance")),
   max_PtBalance_ (iConfig.getParameter<double> ("MaxPtBalance")),
   nsigma_Pt_   (iConfig.getParameter<double> ("NSigmaPt")),
   max_DCAMuMu_  (iConfig.getParameter<double>("MaxDCAMuMu")),
   max_YPair_   (iConfig.getParameter<double>("MaxRapidityPair")),
   cutCowboys_(iConfig.getParameter<bool>("CutCowboys")),
   theL3LinksLabel (iConfig.getParameter<InputTag>("InputLinks")),
   linkToken_ (consumes<reco::MuonTrackLinksCollection>(theL3LinksLabel))
{

   LogDebug("HLTMuonDimuonL3Filter")
      << " CandTag/MinN/MaxEta/MinNhits/MaxDr/MaxDz/MinPt1/MinPt2/MinInvMass/MaxInvMass/MinAcop/MaxAcop/MinPtBalance/MaxPtBalance/NSigmaPt/MaxDzMuMu/MaxRapidityPair : "
      << candTag_.encode()
      << " " << fast_Accept_
      << " " << min_N_
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
      << " " << max_DCAMuMu_
      << " " << max_YPair_;
}

HLTMuonDimuonL3Filter::~HLTMuonDimuonL3Filter() = default;

void
HLTMuonDimuonL3Filter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("BeamSpotTag",edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL3MuonCandidates"));
  //  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag("hltDiMuonL2PreFiltered0"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  desc.add<bool>("PreviousCandIsL2",true);
  desc.add<bool>("FastAccept",false);
  desc.add<int>("MinN",1);
  desc.add<double>("MaxEta",2.5);
  desc.add<int>("MinNhits",0);
  desc.add<double>("MaxDr",2.0);
  desc.add<double>("MaxDz",9999.0);
  desc.add<int>("ChargeOpt",0);
  vector<double> v1; v1.push_back(0.0);
  vector<double> v2; v2.push_back(1e125);
  vector<double> v3; v3.push_back(3.0);
  vector<double> v4; v4.push_back(3.0);
  vector<double> v5; v5.push_back(1e125);
  vector<double> v6; v6.push_back(2.8);
  vector<double> v7; v7.push_back(3.4);
  desc.add<vector<double> >("MinPtPair",v1);
  desc.add<vector<double> >("MaxPtPair",v2);
  desc.add<vector<double> >("MinPtMax",v3);
  desc.add<vector<double> >("MinPtMin",v4);
  desc.add<vector<double> >("MaxPtMin",v5);
  desc.add<vector<double> >("MinInvMass",v6);
  desc.add<vector<double> >("MaxInvMass",v7);
  desc.add<double>("MinAcop",-1.0);
  desc.add<double>("MaxAcop",3.15);
  desc.add<double>("MinPtBalance",-1.0);
  desc.add<double>("MaxPtBalance",999999.0);
  desc.add<double>("NSigmaPt",0.0);
  desc.add<double>("MaxDCAMuMu",99999.9);
  desc.add<double>("MaxRapidityPair",999999.0);
  desc.add<bool>("CutCowboys",false);
  desc.add<edm::InputTag>("InputLinks",edm::InputTag(""));
  descriptions.add("hltMuonDimuonL3Filter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonDimuonL3Filter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{

   if (min_InvMass_.size() != min_PtPair_.size()) {cout << "ERROR!!! Vector sizes don't match!" << endl; return false;}
   if (min_InvMass_.size() != max_PtPair_.size()) {cout << "ERROR!!! Vector sizes don't match!" << endl; return false;}
   if (min_InvMass_.size() != min_PtMax_.size()) {cout << "ERROR!!! Vector sizes don't match!" << endl; return false;}
   if (min_InvMass_.size() != min_PtMin_.size()) {cout << "ERROR!!! Vector sizes don't match!" << endl; return false;}
   if (min_InvMass_.size() != max_PtMin_.size()) {cout << "ERROR!!! Vector sizes don't match!" << endl; return false;}
   if (min_InvMass_.size() != max_InvMass_.size()) {cout << "ERROR!!! Vector sizes don't match!" << endl; return false;}

   double const MuMass = 0.106;
   double const MuMass2 = MuMass*MuMass;
   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // get hold of trks
   Handle<RecoChargedCandidateCollection> mucands;
   if (saveTags()) filterproduct.addCollectionTag(candTag_);
   iEvent.getByToken(candToken_,mucands);

   // Test to see if we can use L3MuonTrajectorySeeds:
   if (mucands->empty()) return false;
   auto const &tk = (*mucands)[0].track();
   bool useL3MTS=false;

   if (tk->seedRef().isNonnull()){
	   auto a = dynamic_cast<const L3MuonTrajectorySeed*>(tk->seedRef().get());
	   useL3MTS = a != nullptr;
   }

   // sort them by L2Track
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;

   // If we can use L3MuonTrajectory seeds run the older code:
   if (useL3MTS){
     unsigned int maxI = mucands->size();
     for (unsigned int i=0;i!=maxI;i++){
       const TrackRef &tk = (*mucands)[i].track();
       if (previousCandIsL2_) {
           edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
           TrackRef staTrack = l3seedRef->l2Track();
           L2toL3s[staTrack].push_back(RecoChargedCandidateRef(mucands,i));
       } else {
           L2toL3s[tk].push_back(RecoChargedCandidateRef(mucands,i));
       }
     }
   }
   // Using normal TrajectorySeeds:
   else{
     // Read Links collection:
     edm::Handle<reco::MuonTrackLinksCollection> links;
     iEvent.getByToken(linkToken_, links);

     // Loop over RecoChargedCandidates:
     for(unsigned int i(0); i < mucands->size(); ++i){
	RecoChargedCandidateRef cand(mucands,i);
	for(auto const & link : *links){
	  TrackRef tk = cand->track();

	  // Using the same method that was used to create the links between L3 and L2
	  // ToDo: there should be a better way than dR,dPt matching
	  const reco::Track& globalTrack = *link.globalTrack();
	  float dR2 = deltaR2(tk->eta(),tk->phi(),globalTrack.eta(),globalTrack.phi());
	  float dPt = std::abs(tk->pt() - globalTrack.pt())/tk->pt();
          const TrackRef staTrack = link.standAloneTrack();
	  if (dR2 < 0.02*0.02 and dPt < 0.001 and previousCandIsL2_) {
	      L2toL3s[staTrack].push_back(RecoChargedCandidateRef(cand));
	  }
	  else if (not previousCandIsL2_){
	      L2toL3s[tk].push_back(RecoChargedCandidateRef(cand));
	  }
        } //MTL loop
     } //RCC loop
   } //end of using normal TrajectorySeeds


   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByToken(previousCandToken_,previousLevelCands);
   BeamSpot beamSpot;
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByToken(beamspotToken_,recoBeamSpotHandle);
   beamSpot = *recoBeamSpotHandle;

   // Needed for DCA calculation
   ESHandle<MagneticField> bFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

   // needed to compare to L2
   vector<RecoChargedCandidateRef> vl2cands;
   previousLevelCands->getObjects(TriggerMuon,vl2cands);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   double e1,e2;
   Particle::LorentzVector p,p1,p2;

   auto L2toL3s_it1 = L2toL3s.begin();
   auto L2toL3s_end = L2toL3s.end();
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
       LogDebug("HLTMuonDimuonL3Filter") << " ... 1st muon in loop, pt1= "
					 << pt1 << ", ptLx1= " << ptLx1;
       auto L2toL3s_it2 = L2toL3s_it1;
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
	      LogDebug("HLTMuonDimuonL3Filter") << " 2nd muon in loop: q*pt= " << tk2->charge()*tk2->pt() << " (" << cand2->charge()*cand2->pt() << ") " << ", eta= " << tk2->eta() << " (" << cand2->eta() << ") " << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0() ;
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
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 2nd muon in loop, pt2= "
						<< pt2 << ", ptLx2= " << ptLx2;
	
	      if (chargeOpt_<0) {
		if (cand1->charge()*cand2->charge()>0) continue;
	      } else if (chargeOpt_>0) {
		if (cand1->charge()*cand2->charge()<0) continue;
	      }
	
	      // Acoplanarity
	      double acop = fabs(cand1->phi()-cand2->phi());
	      if (acop>M_PI) acop = 2*M_PI - acop;
	      acop = M_PI - acop;
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 acop= " << acop;
	      if (acop<min_Acop_) continue;
	      if (acop>max_Acop_) continue;

	      // Pt balance
	      double ptbalance = fabs(cand1->pt()-cand2->pt());
	      if (ptbalance<min_PtBalance_) continue;
	      if (ptbalance>max_PtBalance_) continue;

	      // Combined dimuon system
	      e1 = sqrt(cand1->momentum().Mag2()+MuMass2);
	      e2 = sqrt(cand2->momentum().Mag2()+MuMass2);
	      p1 = Particle::LorentzVector(cand1->px(),cand1->py(),cand1->pz(),e1);
	      p2 = Particle::LorentzVector(cand2->px(),cand2->py(),cand2->pz(),e2);
	      p = p1+p2;
	
	      double pt12 = p.pt();
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 pt12= " << pt12;
	
	      double invmass = abs(p.mass());
	      // if (invmass>0) invmass = sqrt(invmass); else invmass = 0;
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 invmass= " << invmass;
	      bool proceed=false;
	      for (unsigned int iv=0 ; iv<min_InvMass_.size(); iv++) {
		if (invmass<min_InvMass_[iv]) continue;
		if (invmass>max_InvMass_[iv]) continue;
		if (ptLx1>ptLx2) {
		  if (ptLx1<min_PtMax_[iv]) continue;
		  if (ptLx2<min_PtMin_[iv]) continue;
		  if (ptLx2>max_PtMin_[iv]) continue;
		} else {
		  if (ptLx2<min_PtMax_[iv]) continue;
		  if (ptLx1<min_PtMin_[iv]) continue;
		  if (ptLx1>max_PtMin_[iv]) continue;
		}
		if (pt12<min_PtPair_[iv]) continue;
		if (pt12>max_PtPair_[iv]) continue;
		proceed=true;
	      }
	      if (!proceed) continue;

              // Delta Z between the two muons
              //double DeltaZMuMu = fabs(tk2->dz(beamSpot.position())-tk1->dz(beamSpot.position()));
              //if ( DeltaZMuMu > max_DzMuMu_) continue;

	      // DCA between the two muons
	      TransientTrack mu1TT(*tk1, &(*bFieldHandle));
	      TransientTrack mu2TT(*tk2, &(*bFieldHandle));
	      TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
	      TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
	      if (mu1TS.isValid() && mu2TS.isValid()) {
		ClosestApproachInRPhi cApp;
		cApp.calculate(mu1TS.theState(), mu2TS.theState());
		if (!cApp.status()
		    || cApp.distance() > max_DCAMuMu_) continue;
	      }

              // Max dimuon |rapidity|
              double rapidity = fabs(p.Rapidity());
              if ( rapidity > max_YPair_) continue;

	      ///
	      // if cutting on cowboys reject muons that bend towards each other
	      if(cutCowboys_ && (cand1->charge()*deltaPhi(cand1->phi(), cand2->phi()) > 0.)) continue;

	      // Add this pair
	      n++;
	      LogDebug("HLTMuonDimuonL3Filter") << " Track1 passing filter: pt= " << cand1->pt() << ", eta: " << cand1->eta();
	      LogDebug("HLTMuonDimuonL3Filter") << " Track2 passing filter: pt= " << cand2->pt() << ", eta: " << cand2->eta();
	      LogDebug("HLTMuonDimuonL3Filter") << " Invmass= " << invmass;

	      bool i1done = false;
	      bool i2done = false;
	      vector<RecoChargedCandidateRef> vref;
	      filterproduct.getObjects(TriggerMuon,vref);
	      for (auto & i : vref) {
		RecoChargedCandidateRef candref =  RecoChargedCandidateRef(i);
		TrackRef tktmp = candref->get<TrackRef>();
		if (tktmp==tk1) {
		  i1done = true;
		} else if (tktmp==tk2) {
		  i2done = true;
		}
		if (i1done && i2done) break;
	      }
	
	      if (!i1done) {
		filterproduct.addObject(TriggerMuon,cand1);
	      }
	      if (!i2done) {
		filterproduct.addObject(TriggerMuon,cand2);
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
   const bool accept (n >= min_N_);

   LogDebug("HLTMuonDimuonL3Filter") << " >>>>> Result of HLTMuonDimuonL3Filter is "<< accept << ", number of muon pairs passing thresholds= " << n;

   return accept;
}


bool
HLTMuonDimuonL3Filter::triggeredByLevel2(TrackRef const & staTrack,vector<RecoChargedCandidateRef> const & vcands)
{
  bool ok=false;
  for (auto const & vcand : vcands) {
    if ( vcand->get<TrackRef>() == staTrack ) {
      ok=true;
      LogDebug("HLTMuonL3PreFilter") << "The L2 track triggered";
      break;
    }
  }
  return ok;
}
