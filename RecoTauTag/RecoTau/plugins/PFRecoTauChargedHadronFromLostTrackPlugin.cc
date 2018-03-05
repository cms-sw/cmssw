/*
 * PFRecoTauChargedHadronFromLostTrackPlugin
 *
 * Build PFRecoTauChargedHadron objects
 * using lostTracks, i.e. pat::PackedCandidates built for tracks not used 
 * by PFlow algorithm as input
 *
 * Author: Michal Bluj, NCBJ, Poland
 * based on PFRecoTauChargedHadronFromTrackPlugin by Christian Veelken
 *
 */

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include <TMath.h>

#include <memory>
#include <cmath>

namespace reco { namespace tau {

class PFRecoTauChargedHadronFromLostTrackPlugin : public PFRecoTauChargedHadronBuilderPlugin 
{
 public:
  explicit PFRecoTauChargedHadronFromLostTrackPlugin(const edm::ParameterSet&, edm::ConsumesCollector && iC);
  ~PFRecoTauChargedHadronFromLostTrackPlugin() override;
  // Return type is auto_ptr<ChargedHadronVector>
  return_type operator()(const reco::Jet&) const override;
  // Hook to update PV information
  void beginEvent() override;
  
 private:
  typedef std::vector<reco::CandidatePtr> CandPtrs;

  RecoTauVertexAssociator vertexAssociator_;

  RecoTauQualityCuts* qcuts_;

  edm::InputTag srcLostTracks_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> lostTracks_token;
  double dRcone_;
  bool dRconeLimitedToJetArea_;

  double dRmergeNeutralHadron_;
  double dRmergePhoton_;

  math::XYZVector magneticFieldStrength_;

  mutable int numWarnings_;
  int maxWarnings_;

  int verbosity_;
};

  PFRecoTauChargedHadronFromLostTrackPlugin::PFRecoTauChargedHadronFromLostTrackPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector && iC)
    : PFRecoTauChargedHadronBuilderPlugin(pset,std::move(iC)),
      vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"),std::move(iC)),
    qcuts_(nullptr)
{
  edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
  qcuts_ = new RecoTauQualityCuts(qcuts_pset);

  srcLostTracks_ = pset.getParameter<edm::InputTag>("srcLostTracks");
  lostTracks_token = iC.consumes<pat::PackedCandidateCollection>(srcLostTracks_);
  dRcone_ = pset.getParameter<double>("dRcone");
  dRconeLimitedToJetArea_ = pset.getParameter<bool>("dRconeLimitedToJetArea");

  dRmergeNeutralHadron_ = pset.getParameter<double>("dRmergeNeutralHadron");
  dRmergePhoton_ = pset.getParameter<double>("dRmergePhoton");

  numWarnings_ = 0;
  maxWarnings_ = 3;

  verbosity_ = ( pset.exists("verbosity") ) ?
    pset.getParameter<int>("verbosity") : 0;
}
  
PFRecoTauChargedHadronFromLostTrackPlugin::~PFRecoTauChargedHadronFromLostTrackPlugin()
{
  delete qcuts_;
}

// Update the primary vertex
void PFRecoTauChargedHadronFromLostTrackPlugin::beginEvent() 
{
  vertexAssociator_.setEvent(*this->evt());

  edm::ESHandle<MagneticField> magneticField;
  evtSetup()->get<IdealMagneticFieldRecord>().get(magneticField);
  magneticFieldStrength_ = magneticField->inTesla(GlobalPoint(0.,0.,0.));
}

namespace
{
  struct Candidate_withDistance 
  {
    reco::CandidatePtr pfCandidate_;
    double distance_;
  };

  bool isSmallerDistance(const Candidate_withDistance& cand1, const Candidate_withDistance& cand2)
  {
    return (cand1.distance_ < cand2.distance_);
  }
}

PFRecoTauChargedHadronFromLostTrackPlugin::return_type PFRecoTauChargedHadronFromLostTrackPlugin::operator()(const reco::Jet& jet) const 
{
  if ( verbosity_ ) {
    edm::LogPrint("TauChHFromLostTrack") << "<PFRecoTauChargedHadronFromLostTrackPlugin::operator()>:" ;
    edm::LogPrint("TauChHFromLostTrack") << " pluginName = " << name() ;
  }

  ChargedHadronVector output;

  const edm::Event& evt = (*this->evt());

  edm::Handle<pat::PackedCandidateCollection> lostTracks;
  evt.getByToken(lostTracks_token, lostTracks);

  qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
  float jEta=jet.eta();
  float jPhi=jet.phi();
  size_t numTracks = lostTracks->size();
  for ( size_t iTrack = 0; iTrack < numTracks; ++iTrack ) {
    // ignore lostTracks without detailed information
    if ( !(*lostTracks)[iTrack].hasTrackDetails() ) continue;
    const reco::Track *track = &(*lostTracks)[iTrack].pseudoTrack();

    // consider tracks in vicinity of tau-jet candidate only
    double dR = deltaR((*lostTracks)[iTrack].eta(), (*lostTracks)[iTrack].phi(), jEta,jPhi);
    double dRmatch = dRcone_;
    if ( dRconeLimitedToJetArea_ ) {
      double jetArea = jet.jetArea();
      if ( jetArea > 0. ) {
	dRmatch = TMath::Min(dRmatch, TMath::Sqrt(jetArea/TMath::Pi()));
      } else {
	if ( numWarnings_ < maxWarnings_ ) {
	  edm::LogInfo("PFRecoTauChargedHadronFromLostTrackPlugin::operator()") 
	    << "Jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << " has area = " << jetArea << " !!" << std::endl;
	  ++numWarnings_;
	}
	dRmatch = 0.1;
      }
    }
    if ( dR > dRmatch ) continue;

    // ignore tracks which fail quality cuts
    if ( (*lostTracks)[iTrack].charge() == 0 || !qcuts_->filterChargedCand((*lostTracks)[iTrack]) ) continue;

    reco::Candidate::Charge trackCharge_int = 0;
    if ( (*lostTracks)[iTrack].charge() > 0. ) trackCharge_int = +1;
    else if ( (*lostTracks)[iTrack].charge() < 0. ) trackCharge_int = -1;

    const double chargedPionMass = 0.13957; // GeV
    double chargedPionP  = (*lostTracks)[iTrack].p();
    double chargedPionEn = TMath::Sqrt(chargedPionP*chargedPionP + chargedPionMass*chargedPionMass);
    reco::Candidate::LorentzVector chargedPionP4((*lostTracks)[iTrack].px(), (*lostTracks)[iTrack].py(),(*lostTracks)[iTrack].pz(), chargedPionEn);

    reco::Vertex::Point vtx(0.,0.,0.);
    if ( vertexAssociator_.associatedVertex(jet).isNonnull() ) vtx = vertexAssociator_.associatedVertex(jet)->position();

    std::auto_ptr<PFRecoTauChargedHadron> chargedHadron(new PFRecoTauChargedHadron(trackCharge_int, chargedPionP4, vtx, 0, true, PFRecoTauChargedHadron::kTrack));
    // MB: Not possible to save track for a lostTrack(PackedCandidate)/miniAOD,
    // need to get the track later using the pseudoTrack method (downstream)
    //chargedHadron->track_ = edm::Ptr<reco::Track>(tracks, iTrack);
    chargedHadron->lostTrackCandidate_ = edm::Ptr<pat::PackedCandidate>(lostTracks,iTrack);

    // CV: Take code for propagating track to ECAL entrance 
    //     from RecoParticleFlow/PFTracking/src/PFTrackTransformer.cc
    //     to make sure propagation is done in the same way as for charged PFCandidates.
    //     
    //     The following replacements need to be made
    //       outerMomentum -> momentum
    //       outerPosition -> referencePoint
    //     in order to run on AOD input
    //    (outerMomentum and outerPosition require access to reco::TrackExtra objects, which are available in RECO only)
    //
    XYZTLorentzVector chargedPionPos((*lostTracks)[iTrack].vertex().x(), (*lostTracks)[iTrack].vertex().y(), (*lostTracks)[iTrack].vertex().z(), 0.);
    BaseParticlePropagator trackPropagator(RawParticle(chargedPionP4, chargedPionPos), 0., 0., magneticFieldStrength_.z());
    trackPropagator.setCharge((*lostTracks)[iTrack].charge());
    trackPropagator.propagateToEcalEntrance(false);
    if ( trackPropagator.getSuccess() != 0 ) { 
      chargedHadron->positionAtECALEntrance_ = trackPropagator.vertex();
    } else {
      if ( chargedPionP4.pt() > 2. ) {
	edm::LogWarning("PFRecoTauChargedHadronFromLostTrackPlugin::operator()") 
	  << "Failed to propagate track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << " to ECAL entrance !!" << std::endl;
      }
      chargedHadron->positionAtECALEntrance_ = math::XYZPointF(0.,0.,0.);
    }

    std::vector<Candidate_withDistance> neutralJetConstituents_withDistance;
    std::vector<reco::CandidatePtr> jetConstituents = jet.daughterPtrVector();
    for ( const auto& jetConstituent : jetConstituents ) {
      int pdgId = jetConstituent->pdgId();
      if ( !(pdgId == 130 || pdgId == 22) ) continue;
      double dR = deltaR(atECALEntrance(&*jetConstituent, magneticFieldStrength_.z()), chargedHadron->positionAtECALEntrance_);
      double dRmerge = -1.;      
      if      ( pdgId == 130 ) dRmerge = dRmergeNeutralHadron_;
      else if ( pdgId == 22 ) dRmerge = dRmergePhoton_;
      if ( dR < dRmerge ) {
	Candidate_withDistance jetConstituent_withDistance;
	jetConstituent_withDistance.pfCandidate_ = jetConstituent;
	jetConstituent_withDistance.distance_ = dR;
	neutralJetConstituents_withDistance.push_back(jetConstituent_withDistance);
	chargedHadron->addDaughter(jetConstituent);
      }
    }
    std::sort(neutralJetConstituents_withDistance.begin(), neutralJetConstituents_withDistance.end(), isSmallerDistance);

    const double caloResolutionCoeff = 1.0; // CV: approximate ECAL + HCAL calorimeter resolution for hadrons by 100%*sqrt(E)
    double trackPtError = 0.06; // MB: Approximate avarage track PtError by 2.5% (barrel), 4% (transition), 6% (endcaps) lostTracks w/o detailed track information available (after TRK-11-001)
    if( std::abs((*lostTracks)[iTrack].eta()) < 0.9 )
      trackPtError = 0.025;
    else if( std::abs((*lostTracks)[iTrack].eta()) < 1.4 )
      trackPtError = 0.04;
    if(track != nullptr)
      trackPtError = track->ptError();
    double resolutionTrackP =(*lostTracks)[iTrack].p()*(trackPtError/(*lostTracks)[iTrack].pt());
    double neutralEnSum = 0.;
    for ( std::vector<Candidate_withDistance>::const_iterator nextNeutral = neutralJetConstituents_withDistance.begin();
	  nextNeutral != neutralJetConstituents_withDistance.end(); ++nextNeutral ) {
      double nextNeutralEn = nextNeutral->pfCandidate_->energy();      
      double resolutionCaloEn = caloResolutionCoeff*sqrt(neutralEnSum + nextNeutralEn);
      double resolution = sqrt(resolutionTrackP*resolutionTrackP + resolutionCaloEn*resolutionCaloEn);
      if ( (neutralEnSum + nextNeutralEn) < ((*lostTracks)[iTrack].p() + 2.*resolution) ) {
	chargedHadron->neutralPFCandidates_.push_back(nextNeutral->pfCandidate_);
	neutralEnSum += nextNeutralEn;
      } else {
	break;
      }
    }

    setChargedHadronP4(*chargedHadron);

    if ( verbosity_ ) {
      edm::LogPrint("TauChHFromLostTrack") << *chargedHadron;
    }

    output.push_back(chargedHadron);
  }

  return output.release();
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronBuilderPluginFactory, reco::tau::PFRecoTauChargedHadronFromLostTrackPlugin, "PFRecoTauChargedHadronFromLostTrackPlugin");
