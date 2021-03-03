/** \class HLTMuonL3PreFilter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz, J-R Vlimant
 *
 */

#include "HLTMuonL3PreFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/Muon.h"

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

HLTMuonL3PreFilter::HLTMuonL3PreFilter(const ParameterSet& iConfig)
    : HLTFilter(iConfig),
      beamspotTag_(iConfig.getParameter<edm::InputTag>("BeamSpotTag")),
      beamspotToken_(consumes<reco::BeamSpot>(beamspotTag_)),
      candTag_(iConfig.getParameter<InputTag>("CandTag")),
      candToken_(consumes<reco::RecoChargedCandidateCollection>(candTag_)),
      previousCandTag_(iConfig.getParameter<InputTag>("PreviousCandTag")),
      previousCandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
      l1CandTag_(iConfig.getParameter<InputTag>("L1CandTag")),
      l1CandToken_(consumes<trigger::TriggerFilterObjectWithRefs>(l1CandTag_)),
      recoMuTag_(iConfig.getParameter<InputTag>("inputMuonCollection")),
      recoMuToken_(consumes<reco::MuonCollection>(recoMuTag_)),
      min_N_(iConfig.getParameter<int>("MinN")),
      max_Eta_(iConfig.getParameter<double>("MaxEta")),
      min_Nhits_(iConfig.getParameter<int>("MinNhits")),
      max_Dr_(iConfig.getParameter<double>("MaxDr")),
      min_Dr_(iConfig.getParameter<double>("MinDr")),
      max_Dz_(iConfig.getParameter<double>("MaxDz")),
      min_DxySig_(iConfig.getParameter<double>("MinDxySig")),
      min_Pt_(iConfig.getParameter<double>("MinPt")),
      nsigma_Pt_(iConfig.getParameter<double>("NSigmaPt")),
      max_NormalizedChi2_(iConfig.getParameter<double>("MaxNormalizedChi2")),
      max_DXYBeamSpot_(iConfig.getParameter<double>("MaxDXYBeamSpot")),
      min_DXYBeamSpot_(iConfig.getParameter<double>("MinDXYBeamSpot")),
      min_NmuonHits_(iConfig.getParameter<int>("MinNmuonHits")),
      max_PtDifference_(iConfig.getParameter<double>("MaxPtDifference")),
      min_TrackPt_(iConfig.getParameter<double>("MinTrackPt")),
      min_MuonStations_L3fromL1_(iConfig.getParameter<int>("minMuonStations")),
      allowedTypeMask_L3fromL1_(iConfig.getParameter<unsigned int>("allowedTypeMask")),
      requiredTypeMask_L3fromL1_(iConfig.getParameter<unsigned int>("requiredTypeMask")),
      maxNormalizedChi2_L3fromL1_(iConfig.getParameter<double>("MaxNormalizedChi2_L3FromL1")),
      trkMuonId_(muon::SelectionType(iConfig.getParameter<unsigned int>("trkMuonId"))),
      L1MatchingdR_(iConfig.getParameter<double>("L1MatchingdR")),
      matchPreviousCand_(iConfig.getParameter<bool>("MatchToPreviousCand")),

      devDebug_(false),
      theL3LinksLabel(iConfig.getParameter<InputTag>("InputLinks")),
      linkToken_(consumes<reco::MuonTrackLinksCollection>(theL3LinksLabel)) {
  LogDebug("HLTMuonL3PreFilter") << " CandTag/MinN/MaxEta/MinNhits/MaxDr/MinDr/MaxDz/MinDxySig/MinPt/NSigmaPt : "
                                 << candTag_.encode() << " " << min_N_ << " " << max_Eta_ << " " << min_Nhits_ << " "
                                 << max_Dr_ << " " << min_Dr_ << " " << max_Dz_ << " " << min_DxySig_ << " " << min_Pt_
                                 << " " << nsigma_Pt_;
}

HLTMuonL3PreFilter::~HLTMuonL3PreFilter() = default;

void HLTMuonL3PreFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("BeamSpotTag", edm::InputTag("hltOfflineBeamSpot"));
  desc.add<edm::InputTag>("CandTag", edm::InputTag("hltL3MuonCandidates"));
  //  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag("hltDiMuonL2PreFiltered0"));
  desc.add<edm::InputTag>("PreviousCandTag", edm::InputTag(""));
  desc.add<edm::InputTag>("L1CandTag", edm::InputTag(""));
  desc.add<edm::InputTag>("inputMuonCollection", edm::InputTag(""));
  desc.add<int>("MinN", 1);
  desc.add<double>("MaxEta", 2.5);
  desc.add<int>("MinNhits", 0);
  desc.add<double>("MaxDr", 2.0);
  desc.add<double>("MinDr", -1.0);
  desc.add<double>("MaxDz", 9999.0);
  desc.add<double>("MinDxySig", -1.0);
  desc.add<double>("MinPt", 3.0);
  desc.add<double>("NSigmaPt", 0.0);
  desc.add<double>("MaxNormalizedChi2", 9999.0);
  desc.add<double>("MaxDXYBeamSpot", 9999.0);
  desc.add<double>("MinDXYBeamSpot", -1.0);
  desc.add<int>("MinNmuonHits", 0);
  desc.add<double>("MaxPtDifference", 9999.0);
  desc.add<double>("MinTrackPt", 0.0);
  desc.add<int>("minMuonStations", -1);
  desc.add<int>("minTrkHits", -1);
  desc.add<int>("minMuonHits", -1);
  desc.add<unsigned int>("allowedTypeMask", 255);
  desc.add<unsigned int>("requiredTypeMask", 0);
  desc.add<double>("MaxNormalizedChi2_L3FromL1", 9999.);
  desc.add<unsigned int>("trkMuonId", 0);
  desc.add<double>("L1MatchingdR", 0.3);
  desc.add<bool>("MatchToPreviousCand", true);
  desc.add<edm::InputTag>("InputLinks", edm::InputTag(""));
  descriptions.add("hltMuonL3PreFilter", desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTMuonL3PreFilter::hltFilter(Event& iEvent,
                                   const EventSetup& iSetup,
                                   trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  if (saveTags())
    filterproduct.addCollectionTag(candTag_);

  // Read RecoChargedCandidates from L3MuonCandidateProducer:
  Handle<RecoChargedCandidateCollection> mucands;
  iEvent.getByToken(candToken_, mucands);

  // Read L2 triggered objects:
  Handle<TriggerFilterObjectWithRefs> previousLevelCands;
  iEvent.getByToken(previousCandToken_, previousLevelCands);
  vector<RecoChargedCandidateRef> vl2cands;
  previousLevelCands->getObjects(TriggerMuon, vl2cands);

  // Read BeamSpot information:
  Handle<BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamspotToken_, recoBeamSpotHandle);
  const BeamSpot& beamSpot = *recoBeamSpotHandle;

  // Number of objects passing the L3 Trigger:
  int n = 0;

  // sort them by L2Track
  std::map<reco::TrackRef, std::vector<RecoChargedCandidateRef> > L2toL3s;
  // map the L3 cands matched to a L1 to their position in the recoMuon collection
  std::map<unsigned int, RecoChargedCandidateRef> MuonToL3s;

  // Test to see if we can use L3MuonTrajectorySeeds:
  if (mucands->empty())
    return false;
  auto const& tk = (*mucands)[0].track();
  bool useL3MTS = false;

  if (tk->seedRef().isNonnull()) {
    auto a = dynamic_cast<const L3MuonTrajectorySeed*>(tk->seedRef().get());
    useL3MTS = a != nullptr;
  }

  // If we can use L3MuonTrajectory seeds run the older code:
  if (useL3MTS) {
    LogDebug("HLTMuonL3PreFilter") << "HLTMuonL3PreFilter::hltFilter is in mode: useL3MTS";

    unsigned int maxI = mucands->size();
    for (unsigned int i = 0; i != maxI; ++i) {
      const TrackRef& tk = (*mucands)[i].track();
      edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef =
          tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
      TrackRef staTrack = l3seedRef->l2Track();
      LogDebug("HLTMuonL3PreFilter") << "L2 from: " << iEvent.getStableProvenance(staTrack.id()).moduleLabel()
                                     << " index: " << staTrack.key();
      L2toL3s[staTrack].push_back(RecoChargedCandidateRef(mucands, i));
    }
  }  //end of useL3MTS

  // Using normal TrajectorySeeds:
  else {
    LogDebug("HLTMuonL3PreFilter") << "HLTMuonL3PreFilter::hltFilter is in mode: not useL3MTS";

    // Read Links collection:
    edm::Handle<reco::MuonTrackLinksCollection> links;
    iEvent.getByToken(linkToken_, links);

    edm::Handle<trigger::TriggerFilterObjectWithRefs> level1Cands;
    std::vector<l1t::MuonRef> vl1cands;

    bool check_l1match = true;

    // Loop over RecoChargedCandidates:
    for (unsigned int i(0); i < mucands->size(); ++i) {
      RecoChargedCandidateRef cand(mucands, i);
      TrackRef tk = cand->track();

      if (!matchPreviousCand_) {
        MuonToL3s[i] = RecoChargedCandidateRef(cand);
      } else {
        check_l1match = true;
        int nlink = 0;
        for (auto const& link : *links) {
          nlink++;

          // Using the same method that was used to create the links between L3 and L2
          // ToDo: there should be a better way than dR,dPt matching
          const reco::Track& trackerTrack = *link.trackerTrack();

          float dR2 = deltaR2(tk->eta(), tk->phi(), trackerTrack.eta(), trackerTrack.phi());
          float dPt = std::abs(tk->pt() - trackerTrack.pt());
          if (tk->pt() != 0)
            dPt = dPt / tk->pt();

          if (dR2 < 0.02 * 0.02 and dPt < 0.001) {
            const TrackRef staTrack = link.standAloneTrack();
            L2toL3s[staTrack].push_back(RecoChargedCandidateRef(cand));
            check_l1match = false;
          }
        }  //MTL loop

        if (!l1CandTag_.label().empty() && check_l1match) {
          iEvent.getByToken(l1CandToken_, level1Cands);
          level1Cands->getObjects(trigger::TriggerL1Mu, vl1cands);
          const unsigned int nL1Muons(vl1cands.size());
          for (unsigned int il1 = 0; il1 != nL1Muons; ++il1) {
            if (deltaR(cand->eta(), cand->phi(), vl1cands[il1]->eta(), vl1cands[il1]->phi()) < L1MatchingdR_) {
              MuonToL3s[i] = RecoChargedCandidateRef(cand);
            }
          }
        }
      }
    }  //RCC loop
  }    //end of using normal TrajectorySeeds

  // look at all mucands,  check cuts and add to filter object
  LogDebug("HLTMuonL3PreFilter") << "looking at: " << L2toL3s.size() << " L2->L3s from: " << mucands->size();
  for (const auto& L2toL3s_it : L2toL3s) {
    if (!triggeredByLevel2(L2toL3s_it.first, vl2cands))
      continue;

    //loop over the L3Tk reconstructed for this L2.
    unsigned int iTk = 0;
    unsigned int maxItk = L2toL3s_it.second.size();
    for (; iTk != maxItk; iTk++) {
      const RecoChargedCandidateRef& cand = L2toL3s_it.second[iTk];
      if (!applySelection(cand, beamSpot))
        continue;

      filterproduct.addObject(TriggerMuon, cand);
      n++;
      break;  // and go on with the next L2 association
    }
  }  ////loop over L2s from L3 grouping

  // now loop on L3 from L1
  edm::Handle<reco::MuonCollection> recomuons;
  iEvent.getByToken(recoMuToken_, recomuons);

  for (const auto& MuonToL3s_it : MuonToL3s) {
    const reco::Muon& muon(recomuons->at(MuonToL3s_it.first));

    // applys specific cuts for TkMu
    if ((muon.type() & allowedTypeMask_L3fromL1_) == 0)
      continue;
    if ((muon.type() & requiredTypeMask_L3fromL1_) != requiredTypeMask_L3fromL1_)
      continue;
    if (muon.numberOfMatchedStations() < min_MuonStations_L3fromL1_)
      continue;
    if (!muon.globalTrack().isNull()) {
      if (muon.globalTrack()->normalizedChi2() > maxNormalizedChi2_L3fromL1_)
        continue;
    }
    if (muon.isTrackerMuon() && !muon::isGoodMuon(muon, trkMuonId_))
      continue;

    const RecoChargedCandidateRef& cand = MuonToL3s_it.second;
    // apply common selection
    if (!applySelection(cand, beamSpot))
      continue;
    filterproduct.addObject(TriggerMuon, cand);
    n++;

    break;  // and go on with the next L3 from L1
  }

  vector<RecoChargedCandidateRef> vref;
  filterproduct.getObjects(TriggerMuon, vref);
  for (auto& i : vref) {
    RecoChargedCandidateRef candref = RecoChargedCandidateRef(i);
    TrackRef tk = candref->get<TrackRef>();
    LogDebug("HLTMuonL3PreFilter") << " Track passing filter: trackRef pt= " << tk->pt() << " (" << candref->pt()
                                   << ") "
                                   << ", eta: " << tk->eta() << " (" << candref->eta() << ") ";
  }

  // filter decision
  const bool accept(n >= min_N_);

  LogDebug("HLTMuonL3PreFilter") << " >>>>> Result of HLTMuonL3PreFilter is " << accept
                                 << ", number of muons passing thresholds= " << n;

  return accept;
}

bool HLTMuonL3PreFilter::triggeredByLevel2(const TrackRef& staTrack, vector<RecoChargedCandidateRef>& vcands) const {
  bool ok = false;
  for (auto& vcand : vcands) {
    if (vcand->get<TrackRef>() == staTrack) {
      ok = true;
      LogDebug("HLTMuonL3PreFilter") << "The L2 track triggered";
      break;
    }
  }
  return ok;
}

bool HLTMuonL3PreFilter::applySelection(const RecoChargedCandidateRef& cand, const BeamSpot& beamSpot) const {
  // eta cut
  if (std::abs(cand->eta()) > max_Eta_)
    return false;

  TrackRef tk = cand->track();
  LogDebug("HLTMuonL3PreFilter") << " Muon in loop, q*pt= " << tk->charge() * tk->pt() << " ("
                                 << cand->charge() * cand->pt() << ") "
                                 << ", eta= " << tk->eta() << " (" << cand->eta() << ") "
                                 << ", hits= " << tk->numberOfValidHits() << ", d0= " << tk->d0()
                                 << ", dz= " << tk->dz();

  // cut on number of hits
  if (tk->numberOfValidHits() < min_Nhits_)
    return false;

  //max dr cut
  auto dr =
      std::abs((-(cand->vx() - beamSpot.x0()) * cand->py() + (cand->vy() - beamSpot.y0()) * cand->px()) / cand->pt());
  if (dr > max_Dr_)
    return false;

  //min dr cut
  if (dr < min_Dr_)
    return false;

  //dz cut
  if (std::abs((cand->vz() - beamSpot.z0()) -
               ((cand->vx() - beamSpot.x0()) * cand->px() + (cand->vy() - beamSpot.y0()) * cand->py()) / cand->pt() *
                   cand->pz() / cand->pt()) > max_Dz_)
    return false;

  // dxy significance cut (safeguard against bizarre values)
  if (min_DxySig_ > 0 && (tk->dxyError() <= 0 || std::abs(tk->dxy(beamSpot.position()) / tk->dxyError()) < min_DxySig_))
    return false;

  //normalizedChi2 cut
  if (tk->normalizedChi2() > max_NormalizedChi2_)
    return false;

  //dxy beamspot cut
  float absDxy = std::abs(tk->dxy(beamSpot.position()));
  if (absDxy > max_DXYBeamSpot_ || absDxy < min_DXYBeamSpot_)
    return false;

  //min muon hits cut
  const reco::HitPattern& trackHits = tk->hitPattern();
  if (trackHits.numberOfValidMuonHits() < min_NmuonHits_)
    return false;

  //pt difference cut
  double candPt = cand->pt();
  double trackPt = tk->pt();

  if (std::abs(candPt - trackPt) > max_PtDifference_)
    return false;

  //track pt cut
  if (trackPt < min_TrackPt_)
    return false;

  // Pt threshold cut
  double pt = cand->pt();
  double err0 = tk->error(0);
  double abspar0 = std::abs(tk->parameter(0));
  double ptLx = pt;
  // convert 50% efficiency threshold to 90% efficiency threshold
  if (abspar0 > 0)
    ptLx += nsigma_Pt_ * err0 / abspar0 * pt;
  LogTrace("HLTMuonL3PreFilter") << " ...Muon in loop, trackkRef pt= " << tk->pt() << ", ptLx= " << ptLx << " cand pT "
                                 << cand->pt();
  if (ptLx < min_Pt_)
    return false;

  return true;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTMuonL3PreFilter);
