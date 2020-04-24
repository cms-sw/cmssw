/** \class HLTMuonDimuonL2Filter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz, P. Garcia
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "HLTrigger/Muon/interface/HLTMuonDimuonL3Filter.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

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
   l1CandTag_   (iConfig.getParameter<InputTag > ("L1CandTag")),
   l1CandToken_ (consumes<trigger::TriggerFilterObjectWithRefs>(l1CandTag_)),
   recoMuTag_   (iConfig.getParameter<InputTag > ("inputMuonCollection")),
   recoMuToken_ (consumes<reco::MuonCollection>(recoMuTag_)),   
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
   linkToken_ (consumes<reco::MuonTrackLinksCollection>(theL3LinksLabel)),
   L1MatchingdR_ (iConfig.getParameter<double> ("L1MatchingdR")),
   matchPreviousCand_ (iConfig.getParameter<bool>("MatchToPreviousCand") ),
   MuMass2_(0.106*0.106)
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
  desc.add<edm::InputTag>("L1CandTag",edm::InputTag(""));
  desc.add<edm::InputTag>("inputMuonCollection",edm::InputTag(""));
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
  desc.add<double>("L1MatchingdR",0.3);
  desc.add<bool>("MatchToPreviousCand", true);
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

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // Read RecoChargedCandidates from L3MuonCandidateProducer:
   Handle<RecoChargedCandidateCollection> mucands;
   if (saveTags()) filterproduct.addCollectionTag(candTag_);	//?
   iEvent.getByToken(candToken_,mucands);

   // Read L2 triggered objects:
   Handle<TriggerFilterObjectWithRefs> previousLevelCands;
   iEvent.getByToken(previousCandToken_,previousLevelCands);
   vector<RecoChargedCandidateRef> vl2cands;
   previousLevelCands->getObjects(TriggerMuon,vl2cands);

   // Read BeamSpot information:
   Handle<BeamSpot> recoBeamSpotHandle;
   iEvent.getByToken(beamspotToken_,recoBeamSpotHandle);
   const BeamSpot& beamSpot = *recoBeamSpotHandle;

   // sort them by L2Track
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;
   // map the L3 cands matched to a L1 to their position in the recoMuon collection
   std::map<unsigned int, RecoChargedCandidateRef > MuonToL3s;

   // Test to see if we can use L3MuonTrajectorySeeds:
   if (mucands->empty()) return false;
   auto const &tk = (*mucands)[0].track();
   bool useL3MTS=false;

   if (tk->seedRef().isNonnull()){
	   auto a = dynamic_cast<const L3MuonTrajectorySeed*>(tk->seedRef().get());
	   useL3MTS = a != nullptr;
   }

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

     edm::Handle<trigger::TriggerFilterObjectWithRefs> level1Cands;
     std::vector<l1t::MuonRef> vl1cands;
     std::vector<l1t::MuonRef>::iterator vl1cands_begin;
     std::vector<l1t::MuonRef>::iterator vl1cands_end;

     bool check_l1match = true;

     // Loop over RecoChargedCandidates:
     for(unsigned int i(0); i < mucands->size(); ++i){
	RecoChargedCandidateRef cand(mucands,i);
        TrackRef tk = cand->track(); // is inner track

	if (!matchPreviousCand_){
	    MuonToL3s[i] = RecoChargedCandidateRef(cand);
	}
	else{

	  check_l1match = true;
	  for(auto const & link : *links){

	    // Using the same method that was used to create the links between L3 and L2
	    // ToDo: there should be a better way than dR,dPt matching
	    const reco::Track& trackerTrack = *link.trackerTrack();
            if (tk->pt()==0 or trackerTrack.pt()==0) continue;

	    float dR2 = deltaR2(tk->eta(),tk->phi(),trackerTrack.eta(),trackerTrack.phi());
	    float dPt = std::abs(tk->pt() - trackerTrack.pt())/tk->pt();

	    if (dR2 < 0.02*0.02 and dPt < 0.001) {
	        const TrackRef staTrack = link.standAloneTrack();
	        L2toL3s[staTrack].push_back(RecoChargedCandidateRef(cand));
	        check_l1match = false;
	    }
          } //MTL loop

          if (not l1CandTag_.label().empty() and check_l1match){
              iEvent.getByToken(l1CandToken_,level1Cands);
              level1Cands->getObjects(trigger::TriggerL1Mu,vl1cands);
              const unsigned int nL1Muons(vl1cands.size());
	      for (unsigned int il1=0; il1!=nL1Muons; ++il1) {
                  if (deltaR(cand->eta(), cand->phi(), vl1cands[il1]->eta(), vl1cands[il1]->phi()) < L1MatchingdR_) { //was muon, non cand
  	            MuonToL3s[i] = RecoChargedCandidateRef(cand);
                  }
	      }
          }
        }  
     } //RCC loop
   } //end of using normal TrajectorySeeds

   // Needed for DCA calculation
   ESHandle<MagneticField> bFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;

   // look at all mucands,  check cuts and add to filter object
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

       LogDebug("HLTMuonDimuonL3Filter") << " 1st muon in loop: q*pt= " << tk1->charge()*tk1->pt() 
		<< " (" << cand1->charge()*cand1->pt()<< ") " << ", eta= " << tk1->eta() 
		<< " (" << cand1->eta() << ") " << ", hits= " << tk1->numberOfValidHits();

       // Run muon selection on first muon:
       if (!applyMuonSelection(cand1, beamSpot)) continue;

       // Pt threshold cut
       // Don't convert to 90% efficiency threshold
       LogDebug("HLTMuonDimuonL3Filter") << " ... 1st muon in loop, pt1= " << cand1->pt();

       // Loop on 2nd muon cand
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
	
	      LogDebug("HLTMuonDimuonL3Filter") << " 2nd muon in loop: q*pt= " << tk2->charge()*tk2->pt()
		<< " (" << cand2->charge()*cand2->pt() << ") " << ", eta= " << tk2->eta() 
		<< " (" << cand2->eta() << ") " << ", hits= " << tk2->numberOfValidHits() << ", d0= " << tk2->d0() ;

              // Run muon selection on second muon:
              if (!applyMuonSelection(cand2, beamSpot)) continue;

	      // Pt threshold cut
	      // Don't convert to 90% efficiency threshold
	      LogDebug("HLTMuonDimuonL3Filter") << " ... 2nd muon in loop, pt2= " << cand2->pt();

              // Run dimuon selection:
              if (!applyDiMuonSelection(cand1, cand2, beamSpot, bFieldHandle)) continue;
	
	      // Add this pair
	      n++;
	      LogDebug("HLTMuonDimuonL3Filter") << " Track1 passing filter: pt= " << cand1->pt() << ", eta: " << cand1->eta();
	      LogDebug("HLTMuonDimuonL3Filter") << " Track2 passing filter: pt= " << cand2->pt() << ", eta: " << cand2->eta();

	      bool i1done = false;
	      bool i2done = false;
	      vector<RecoChargedCandidateRef> vref;
	      filterproduct.getObjects(TriggerMuon,vref);
	      for (auto & i : vref) {
		RecoChargedCandidateRef candref =  RecoChargedCandidateRef(i);
		TrackRef tktmp = candref->get<TrackRef>();
		if (tktmp==tk1) i1done = true;
		else if (tktmp==tk2) i2done = true;	//why is this an elif?
		if (i1done && i2done) break;
	      }
	      if (!i1done) filterproduct.addObject(TriggerMuon,cand1);
	      if (!i2done) filterproduct.addObject(TriggerMuon,cand2);
	
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

       //Loop over L3FromL1 collection see if we get a pair that way
       auto MuonToL3s_it1  = MuonToL3s.begin();
       auto MuonToL3s_end = MuonToL3s.end();
       for (; MuonToL3s_it1!=MuonToL3s_end; ++MuonToL3s_it1){
         const RecoChargedCandidateRef& cand2=MuonToL3s_it1->second;
         if (!applyMuonSelection(cand2, beamSpot)) continue;
         TrackRef tk2 = cand2->get<TrackRef>();

         // Run dimuon selection:
         if (!applyDiMuonSelection(cand1, cand2, beamSpot, bFieldHandle)) continue;
         n++;
         LogDebug("HLTMuonDimuonL3Filter") << " L3FromL2 Track1 passing filter: pt= " << cand1->pt() << ", eta: " << cand1->eta();
         LogDebug("HLTMuonDimuonL3Filter") << " L3FromL1 Track2 passing filter: pt= " << cand2->pt() << ", eta: " << cand2->eta();

         bool i1done = false;
         bool i2done = false;
         vector<RecoChargedCandidateRef> vref;
         filterproduct.getObjects(TriggerMuon,vref);
         for (auto & i : vref) {
           RecoChargedCandidateRef candref =  RecoChargedCandidateRef(i);
           TrackRef tktmp = candref->get<TrackRef>();
           if (tktmp==tk1) i1done = true;
           else if (tktmp==tk2) i2done = true;     //why is this an elif?
           if (i1done && i2done) break;
         }
         if (!i1done) filterproduct.addObject(TriggerMuon,cand1);
         if (!i2done) filterproduct.addObject(TriggerMuon,cand2);
       
         //break anyway since a L3 track pair has been found matching the criteria
         thisL3Index1isDone=true;
         atLeastOnePair=true;
         break;
       }//L3FromL1 loop
       //break the loop if fast accept.
       if (atLeastOnePair && fast_Accept_) break;
       if (thisL3Index1isDone) break;

     }//loop on tracks for first L2
     //break the loop if fast accept.
     if (atLeastOnePair && fast_Accept_) break;
   }//loop on the first L2


   // now loop on 1st L3 from L1
   auto MuonToL3s_it1  = MuonToL3s.begin();
   auto MuonToL3s_end = MuonToL3s.end();
   for (; MuonToL3s_it1!=MuonToL3s_end; ++MuonToL3s_it1){
     bool thisL3Index1isDone=false;
     const RecoChargedCandidateRef& cand1=MuonToL3s_it1->second;
     if (!applyMuonSelection(cand1, beamSpot)) continue;
     TrackRef tk1 = cand1->get<TrackRef>();

     // Loop on 2nd L3 from L1
     auto MuonToL3s_it2 = MuonToL3s_it1;
     for (; MuonToL3s_it2!=MuonToL3s_end; ++MuonToL3s_it2){
       const RecoChargedCandidateRef& cand2=MuonToL3s_it2->second;
       if (!applyMuonSelection(cand2, beamSpot)) continue;
       TrackRef tk2 = cand2->get<TrackRef>();

       // Run dimuon selection:
       if (!applyDiMuonSelection(cand1, cand2, beamSpot, bFieldHandle)) continue;

       n++;
       LogDebug("HLTMuonDimuonL3Filter") << " L3FromL1 Track1 passing filter: pt= " << cand1->pt() << ", eta: " << cand1->eta();
       LogDebug("HLTMuonDimuonL3Filter") << " L3FromL1 Track2 passing filter: pt= " << cand2->pt() << ", eta: " << cand2->eta();

       bool i1done = false;
       bool i2done = false;
       vector<RecoChargedCandidateRef> vref;
       filterproduct.getObjects(TriggerMuon,vref);
       for (auto & i : vref) {
         RecoChargedCandidateRef candref =  RecoChargedCandidateRef(i);
         TrackRef tktmp = candref->get<TrackRef>();
         if (tktmp==tk1) i1done = true;
         else if (tktmp==tk2) i2done = true;	//why is this an elif?
         if (i1done && i2done) break;
       }
       if (!i1done) filterproduct.addObject(TriggerMuon,cand1);
       if (!i2done) filterproduct.addObject(TriggerMuon,cand2);
       
       //break anyway since a L3 track pair has been found matching the criteria
       thisL3Index1isDone=true;
       atLeastOnePair=true;
       break;
     } //loop on 2nd muon

     //break the loop if fast accept
     if (atLeastOnePair && fast_Accept_) break;
     if (thisL3Index1isDone) break;
   } //loop on 1st muon


   // filter decision
   const bool accept (n >= min_N_);

   LogDebug("HLTMuonDimuonL3Filter") << " >>>>> Result of HLTMuonDimuonL3Filter is "<< accept << ", number of muon pairs passing thresholds= " << n;

   return accept;
}


bool HLTMuonDimuonL3Filter::triggeredByLevel2(TrackRef const & staTrack,vector<RecoChargedCandidateRef> const & vcands){
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

bool HLTMuonDimuonL3Filter::applyMuonSelection(const RecoChargedCandidateRef& cand, const BeamSpot& beamSpot) const{
	// eta cut
	if (std::abs(cand->eta())>max_Eta_) return false;
	
	// cut on number of hits
	TrackRef tk = cand->track();
	if (tk->numberOfValidHits()<min_Nhits_) return false;
	
	//dr cut
	if (std::abs( (- (cand->vx()-beamSpot.x0()) * cand->py() + (cand->vy()-beamSpot.y0()) * cand->px() ) / cand->pt() ) >max_Dr_) return false;
	
	//dz cut
	if (std::abs((cand->vz()-beamSpot.z0()) - ((cand->vx()-beamSpot.x0())*cand->px()+(cand->vy()-beamSpot.y0())*cand->py())/cand->pt() * cand->pz()/cand->pt())>max_Dz_) return false;

	return true;
}


bool HLTMuonDimuonL3Filter::applyDiMuonSelection(const RecoChargedCandidateRef& cand1, const RecoChargedCandidateRef& cand2, const BeamSpot& beamSpot, const ESHandle<MagneticField>& bFieldHandle) const{
	// Opposite Charge
	if (chargeOpt_<0 and (cand1->charge()*cand2->charge()>0)) return false;
	else if (chargeOpt_>0 and (cand1->charge()*cand2->charge()<0)) return false;
	
	// Acoplanarity
	double acop = std::abs(cand1->phi()-cand2->phi());
	if (acop>M_PI) acop = 2*M_PI - acop;
	acop = M_PI - acop;
	LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 acop= " << acop;
	if (acop<min_Acop_) return false;
	if (acop>max_Acop_) return false;
	
	// Pt balance
	double ptbalance = std::abs(cand1->pt()-cand2->pt());
	if (ptbalance<min_PtBalance_) return false;
	if (ptbalance>max_PtBalance_) return false;
	
	// Combined dimuon syste
	double e1,e2;
        Particle::LorentzVector p,p1,p2;
	e1 = sqrt(cand1->momentum().Mag2()+MuMass2_);
	e2 = sqrt(cand2->momentum().Mag2()+MuMass2_);
	p1 = Particle::LorentzVector(cand1->px(),cand1->py(),cand1->pz(),e1);
	p2 = Particle::LorentzVector(cand2->px(),cand2->py(),cand2->pz(),e2);
	p = p1+p2;
	
	double pt12 = p.pt();
	LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 pt12= " << pt12;
	
	double ptLx1 = cand1->pt();
	double ptLx2 = cand2->pt();
	double invmass = abs(p.mass());
	// if (invmass>0) invmass = sqrt(invmass); else invmass = 0;
	LogDebug("HLTMuonDimuonL3Filter") << " ... 1-2 invmass= " << invmass;
	bool proceed=false;
	for (unsigned int iv=0 ; iv<min_InvMass_.size(); iv++) {
	  if (invmass<min_InvMass_[iv]) return false;
	  if (invmass>max_InvMass_[iv]) return false;
	  if (ptLx1>ptLx2) {
	    if (ptLx1<min_PtMax_[iv]) return false;
	    if (ptLx2<min_PtMin_[iv]) return false;
	    if (ptLx2>max_PtMin_[iv]) return false;
	  } else {
	    if (ptLx2<min_PtMax_[iv]) return false;
	    if (ptLx1<min_PtMin_[iv]) return false;
	    if (ptLx1>max_PtMin_[iv]) return false;
	  }
	  if (pt12<min_PtPair_[iv]) return false;
	  if (pt12>max_PtPair_[iv]) return false;
	  proceed=true;
          break;
	}
	if (!proceed) return false;
	
	// Delta Z between the two muons
	//double DeltaZMuMu = std::abs(tk2->dz(beamSpot.position())-tk1->dz(beamSpot.position()));
	//if ( DeltaZMuMu > max_DzMuMu_) return false;
	
	// DCA between the two muons
	TrackRef tk1 = cand1->track();
	TrackRef tk2 = cand2->track();
	TransientTrack mu1TT(*tk1, &(*bFieldHandle));
	TransientTrack mu2TT(*tk2, &(*bFieldHandle));
	TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
	TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
	if (mu1TS.isValid() && mu2TS.isValid()) {
	  ClosestApproachInRPhi cApp;
	  cApp.calculate(mu1TS.theState(), mu2TS.theState());
	  if (!cApp.status()
	      || cApp.distance() > max_DCAMuMu_) return false;
	}
	
	// Max dimuon |rapidity|
	double rapidity = std::abs(p.Rapidity());
	if ( rapidity > max_YPair_) return false;
	
	// if cutting on cowboys reject muons that bend towards each other
	if(cutCowboys_ && (cand1->charge()*deltaPhi(cand1->phi(), cand2->phi()) > 0.)) return false;
	return true;
}
