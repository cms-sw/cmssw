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
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

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
   beamspotToken_ (consumes<reco::BeamSpot>(beamspotTag_)),
   candTag_   (iConfig.getParameter<InputTag > ("CandTag")),
   candToken_ (consumes<reco::RecoChargedCandidateCollection>(candTag_)),
   previousCandTag_   (iConfig.getParameter<InputTag > ("PreviousCandTag")),
   previousCandToken_ (consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
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
   min_DXYBeamSpot_ (iConfig.getParameter<double> ("MinDXYBeamSpot")),
   min_NmuonHits_ (iConfig.getParameter<int> ("MinNmuonHits")),
   max_PtDifference_ (iConfig.getParameter<double> ("MaxPtDifference")),
   min_TrackPt_ (iConfig.getParameter<double> ("MinTrackPt")),
   devDebug_ (false),
   theL3LinksLabel (iConfig.getParameter<InputTag>("InputLinks")),
   linkToken_ (consumes<reco::MuonTrackLinksCollection>(theL3LinksLabel))
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
  desc.add<double>("MinDXYBeamSpot",-1.0);
  desc.add<int>("MinNmuonHits",0);
  desc.add<double>("MaxPtDifference",9999.0);
  desc.add<double>("MinTrackPt",0.0);
  desc.add<edm::InputTag>("InputLinks",edm::InputTag(""));
  descriptions.add("hltMuonL3PreFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTMuonL3PreFilter::hltFilter(Event& iEvent, const EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const{

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   if (saveTags()) filterproduct.addCollectionTag(candTag_);

   // Read RecoChargedCandidates from L3MuonCandidateProducer:
   Handle<RecoChargedCandidateCollection> mucands;
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

   // Number of objects passing the L3 Trigger:
   int n = 0;

   // sort them by L2Track
   std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;
   
   // Test to see if we can use L3MuonTrajectorySeeds:
   if (mucands->size()<1) return false;
   auto tk = (*mucands)[0].track();
   bool useL3MTS=false;

   if (tk->seedRef().isNonnull()){
	   auto a = dynamic_cast<const L3MuonTrajectorySeed*>(tk->seedRef().get());
	   useL3MTS = a != nullptr;
   }

   // If we can use L3MuonTrajectory seeds run the older code:
   if (useL3MTS){
     LogDebug("HLTMuonL3PreFilter") << "HLTMuonL3PreFilter::hltFilter is in mode: useL3MTS";

     unsigned int maxI = mucands->size();
     for (unsigned int i=0;i!=maxI;++i){
       TrackRef tk = (*mucands)[i].track();
       edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
       TrackRef staTrack = l3seedRef->l2Track();
       LogDebug("HLTMuonL3PreFilter") <<"L2 from: "<<iEvent.getProvenance(staTrack.id()).moduleLabel() <<" index: "<<staTrack.key();
       L2toL3s[staTrack].push_back(RecoChargedCandidateRef(mucands,i));
     }

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
         if (std::abs(cand->eta())>max_Eta_) continue;
  
         // cut on number of hits
         if (tk->numberOfValidHits()<min_Nhits_) continue;
  
         //max dr cut
         auto dr = std::abs( (- (cand->vx()-beamSpot.x0()) * cand->py() + (cand->vy()-beamSpot.y0()) * cand->px() ) / cand->pt() );
         if (dr >max_Dr_) continue;
  
         //min dr cut
         if (dr <min_Dr_) continue;
  
         //dz cut
         if (std::abs((cand->vz()-beamSpot.z0()) - ((cand->vx()-beamSpot.x0())*cand->px()+(cand->vy()-beamSpot.y0())*cand->py())/cand->pt() * cand->pz()/cand->pt())>max_Dz_) continue;
  
         // dxy significance cut (safeguard against bizarre values)
         if (min_DxySig_ > 0 && (tk->dxyError() <= 0 || std::abs(tk->dxy(beamSpot.position())/tk->dxyError()) < min_DxySig_)) continue;
  
         //normalizedChi2 cut
         if (tk->normalizedChi2() > max_NormalizedChi2_ ) continue;
  
         //dxy beamspot cut
         float absDxy = std::abs(tk->dxy(beamSpot.position()));
         if (absDxy > max_DXYBeamSpot_ || absDxy < min_DXYBeamSpot_ ) continue;
  
         //min muon hits cut
         const reco::HitPattern& trackHits = tk->hitPattern();
         if (trackHits.numberOfValidMuonHits() < min_NmuonHits_ ) continue;
  
         //pt difference cut
         double candPt = cand->pt();
         double trackPt = tk->pt();
  
         if (std::abs(candPt - trackPt) > max_PtDifference_ ) continue;
  
         //track pt cut
         if (trackPt < min_TrackPt_ ) continue;
  
         // Pt threshold cut
         double pt = cand->pt();
         double err0 = tk->error(0);
         double abspar0 = std::abs(tk->parameter(0));
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
  } //end of useL3MTS

  // Using normal TrajectorySeeds:
  else{
    LogDebug("HLTMuonL3PreFilter") << "HLTMuonL3PreFilter::hltFilter is in mode: not useL3MTS";

    // Read Links collection:
    edm::Handle<reco::MuonTrackLinksCollection> links;
    iEvent.getByToken(linkToken_, links);

    // Loop over RecoChargedCandidates:
    for(unsigned int i(0); i < mucands->size(); ++i){
      RecoChargedCandidateRef cand(mucands,i);
      for(unsigned int l(0); l <links->size(); ++l){
        const reco::MuonTrackLinks* link = &links->at(l);
	bool useThisLink=false;
	TrackRef tk = cand->track();
	reco::TrackRef trkTrack = link->trackerTrack();

	// Using the same method that was used to create the links
	// ToDo: there should be a better way than dR,dPt matching
	const reco::Track& globalTrack = *link->globalTrack();
	float dR2 = deltaR2(tk->eta(),tk->phi(),globalTrack.eta(),globalTrack.phi());
	float dPt = std::abs(tk->pt() - globalTrack.pt())/tk->pt();
	if (dR2 < 0.02*0.02 and dPt < 0.001) {
		useThisLink=true;
	}

	if (useThisLink){
          const TrackRef staTrack = link->standAloneTrack();
	  if (!triggeredByLevel2(staTrack,vl2cands)) continue;

           // eta cut
           if (std::abs(cand->eta())>max_Eta_) continue;
    
           // cut on number of hits
           if (tk->numberOfValidHits()<min_Nhits_) continue;
    
           //max dr cut
           //if (std::abs(tk->d0())>max_Dr_) continue;
           auto dr = std::abs( (- (cand->vx()-beamSpot.x0()) * cand->py() + (cand->vy()-beamSpot.y0()) * cand->px() ) / cand->pt() );
           if (dr >max_Dr_) continue;
    
           //min dr cut
           if (dr <min_Dr_) continue;
    
           //dz cut
           if (std::abs((cand->vz()-beamSpot.z0()) - ((cand->vx()-beamSpot.x0())*cand->px()+(cand->vy()-beamSpot.y0())*cand->py())/cand->pt() * cand->pz()/cand->pt())>max_Dz_) continue;
    
           // dxy significance cut (safeguard against bizarre values)
           if (min_DxySig_ > 0 && (tk->dxyError() <= 0 || std::abs(tk->dxy(beamSpot.position())/tk->dxyError()) < min_DxySig_)) continue;
    
           //normalizedChi2 cut
           if (tk->normalizedChi2() > max_NormalizedChi2_ ) continue;
    
           //dxy beamspot cut
           float absDxy = std::abs(tk->dxy(beamSpot.position()));
           if (absDxy > max_DXYBeamSpot_ || absDxy < min_DXYBeamSpot_ ) continue;
    
           //min muon hits cut
           const reco::HitPattern& trackHits = tk->hitPattern();
           if (trackHits.numberOfValidMuonHits() < min_NmuonHits_ ) continue;
    
           //pt difference cut
           double candPt = cand->pt();
           double trackPt = tk->pt();
    
           if (std::abs(candPt - trackPt) > max_PtDifference_ ) continue;
    
           //track pt cut
           if (trackPt < min_TrackPt_ ) continue;
    
           // Pt threshold cut
           double pt = cand->pt();
           double err0 = tk->error(0);
           double abspar0 = std::abs(tk->parameter(0));
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
	} //end of useThisLink
      } //end of muons in links collection
    } //end of RecoCand collection

    // filter decision:
    const bool accept (n >= min_N_);
    return accept;
  } //not useL3MTS
}

bool
HLTMuonL3PreFilter::triggeredByLevel2(const TrackRef& staTrack,vector<RecoChargedCandidateRef>& vcands) const
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

