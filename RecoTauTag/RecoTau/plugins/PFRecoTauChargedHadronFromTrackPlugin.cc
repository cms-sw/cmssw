/*
 * PFRecoTauChargedHadronFromTrackPlugin
 *
 * Build PFRecoTauChargedHadron objects
 * using charged PFCandidates as input
 *
 * Author: Christian Veelken, LLR
 *
 * $Id $
 */

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/JetReco/interface/PFJet.h"
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
#include <math.h>

namespace reco { namespace tau {

class PFRecoTauChargedHadronFromTrackPlugin : public PFRecoTauChargedHadronBuilderPlugin 
{
 public:
  explicit PFRecoTauChargedHadronFromTrackPlugin(const edm::ParameterSet&);
  virtual ~PFRecoTauChargedHadronFromTrackPlugin();
  // Return type is auto_ptr<ChargedHadronVector>
  return_type operator()(const reco::PFJet&) const;
  // Hook to update PV information
  virtual void beginEvent();
  
 private:
  typedef std::vector<reco::PFCandidatePtr> PFCandPtrs;

  RecoTauVertexAssociator vertexAssociator_;

  RecoTauQualityCuts* qcuts_;

  edm::InputTag srcTracks_;
  double dRcone_;
  bool dRconeLimitedToJetArea_;

  double dRmergeNeutralHadron_;
  double dRmergePhoton_;

  math::XYZVector magneticFieldStrength_;

  mutable int numWarnings_;
  int maxWarnings_;

  int verbosity_;
};

PFRecoTauChargedHadronFromTrackPlugin::PFRecoTauChargedHadronFromTrackPlugin(const edm::ParameterSet& pset)
  : PFRecoTauChargedHadronBuilderPlugin(pset),
    vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts")),
    qcuts_(0)
{
  edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
  qcuts_ = new RecoTauQualityCuts(qcuts_pset);

  srcTracks_ = pset.getParameter<edm::InputTag>("srcTracks");
  dRcone_ = pset.getParameter<double>("dRcone");
  dRconeLimitedToJetArea_ = pset.getParameter<bool>("dRconeLimitedToJetArea");

  dRmergeNeutralHadron_ = pset.getParameter<double>("dRmergeNeutralHadron");
  dRmergePhoton_ = pset.getParameter<double>("dRmergePhoton");

  numWarnings_ = 0;
  maxWarnings_ = 3;

  verbosity_ = ( pset.exists("verbosity") ) ?
    pset.getParameter<int>("verbosity") : 0;
}
  
PFRecoTauChargedHadronFromTrackPlugin::~PFRecoTauChargedHadronFromTrackPlugin()
{
  delete qcuts_;
}

// Update the primary vertex
void PFRecoTauChargedHadronFromTrackPlugin::beginEvent() 
{
  vertexAssociator_.setEvent(*this->evt());

  edm::ESHandle<MagneticField> magneticField;
  evtSetup()->get<IdealMagneticFieldRecord>().get(magneticField);
  magneticFieldStrength_ = magneticField->inTesla(GlobalPoint(0.,0.,0.));
}

namespace
{
  struct PFCandidate_withDistance 
  {
    reco::PFCandidatePtr pfCandidate_;
    double distance_;
  };

  bool isSmallerDistance(const PFCandidate_withDistance& cand1, const PFCandidate_withDistance& cand2)
  {
    return (cand1.distance_ < cand2.distance_);
  }
}

PFRecoTauChargedHadronFromTrackPlugin::return_type PFRecoTauChargedHadronFromTrackPlugin::operator()(const reco::PFJet& jet) const 
{
  //if ( verbosity_ ) {
  //  std::cout << "<PFRecoTauChargedHadronFromTrackPlugin::operator()>:" << std::endl;
  //  std::cout << " pluginName = " << name() << std::endl;
  //}

  ChargedHadronVector output;

  const edm::Event& evt = (*this->evt());

  edm::Handle<reco::TrackCollection> tracks;
  evt.getByLabel(srcTracks_, tracks);

  qcuts_->setPV(vertexAssociator_.associatedVertex(jet));

  size_t numTracks = tracks->size();
  for ( size_t iTrack = 0; iTrack < numTracks; ++iTrack ) {
    reco::TrackRef track(tracks, iTrack);

    // consider tracks in vicinity of tau-jet candidate only
    double dR = deltaR(track->eta(), track->phi(), jet.eta(), jet.phi());
    double dRmatch = dRcone_;
    if ( dRconeLimitedToJetArea_ ) {
      double jetArea = jet.jetArea();
      if ( jetArea > 0. ) {
	dRmatch = TMath::Min(dRmatch, TMath::Sqrt(jetArea/TMath::Pi()));
      } else {
	if ( numWarnings_ < maxWarnings_ ) {
	  edm::LogWarning("PFRecoTauChargedHadronFromTrackPlugin::operator()") 
	    << "Jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << " has area = " << jetArea << " !!" << std::endl;
	  ++numWarnings_;
	}
	dRmatch = 0.1;
      }
    }
    if ( dR > dRmatch ) continue;

    // ignore tracks which fail quality cuts
    if ( !qcuts_->filterTrack(track) ) continue;

    reco::Candidate::Charge trackCharge_int = 0;
    if ( track->charge() > 0. ) trackCharge_int = +1;
    else if ( track->charge() < 0. ) trackCharge_int = -1;

    const double chargedPionMass = 0.13957; // GeV
    reco::Candidate::PolarLorentzVector p4_polar(track->pt(), track->eta(), track->phi(), chargedPionMass);
    reco::Candidate::LorentzVector p4(p4_polar.px(), p4_polar.py(), p4_polar.pz(), p4_polar.E());

    reco::Vertex::Point vtx(0.,0.,0.);
    if ( vertexAssociator_.associatedVertex(jet).isNonnull() ) vtx = vertexAssociator_.associatedVertex(jet)->position();

    std::auto_ptr<PFRecoTauChargedHadron> chargedHadron(new PFRecoTauChargedHadron(trackCharge_int, p4, vtx, 0, true, PFRecoTauChargedHadron::kTrack));
    chargedHadron->track_ = edm::Ptr<reco::Track>(tracks, iTrack);

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
    double chargedPionEn = sqrt(chargedPionMass*chargedPionMass + track->momentum().Mag2());
    reco::Candidate::LorentzVector chargedPionP4(track->momentum().x(), track->momentum().y(), track->momentum().z(), chargedPionEn);
    XYZTLorentzVector chargedPionPos(track->referencePoint().x(), track->referencePoint().y(), track->referencePoint().z(), 0.);
    BaseParticlePropagator trackPropagator(RawParticle(chargedPionP4, chargedPionPos), 0., 0., magneticFieldStrength_.z());
    trackPropagator.setCharge(track->charge());
    trackPropagator.propagateToEcalEntrance(false);
    if ( trackPropagator.getSuccess() != 0 ) { 
      chargedHadron->positionAtECALEntrance_ = trackPropagator.vertex();
    } else {
      if ( chargedPionP4.pt() > 2. ) {
	edm::LogWarning("PFRecoTauChargedHadronFromTrackPlugin::operator()") 
	  << "Failed to propagate track: Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << " to ECAL entrance !!" << std::endl;
      }
      chargedHadron->positionAtECALEntrance_ = math::XYZPointF(0.,0.,0.);
    }

    std::vector<PFCandidate_withDistance> neutralJetConstituents_withDistance;
    std::vector<reco::PFCandidatePtr> jetConstituents = jet.getPFConstituents();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator jetConstituent = jetConstituents.begin();
	  jetConstituent != jetConstituents.end(); ++jetConstituent ) {
      reco::PFCandidate::ParticleType jetConstituentType = (*jetConstituent)->particleId();
      if ( !(jetConstituentType == reco::PFCandidate::h0 || jetConstituentType == reco::PFCandidate::gamma) ) continue;
      double dR = deltaR((*jetConstituent)->positionAtECALEntrance(), chargedHadron->positionAtECALEntrance_);
      double dRmerge = -1.;      
      if      ( jetConstituentType == reco::PFCandidate::h0    ) dRmerge = dRmergeNeutralHadron_;
      else if ( jetConstituentType == reco::PFCandidate::gamma ) dRmerge = dRmergePhoton_;
      if ( dR < dRmerge ) {
	PFCandidate_withDistance jetConstituent_withDistance;
	jetConstituent_withDistance.pfCandidate_ = (*jetConstituent);
	jetConstituent_withDistance.distance_ = dR;
	neutralJetConstituents_withDistance.push_back(jetConstituent_withDistance);
	chargedHadron->addDaughter(*jetConstituent);
      }
    }
    std::sort(neutralJetConstituents_withDistance.begin(), neutralJetConstituents_withDistance.end(), isSmallerDistance);

    const double caloResolutionCoeff = 1.0; // CV: approximate ECAL + HCAL calorimeter resolution for hadrons by 100%*sqrt(E)
    double resolutionTrackP = track->p()*(track->ptError()/track->pt());
    double neutralEnSum = 0.;
    for ( std::vector<PFCandidate_withDistance>::const_iterator nextNeutral = neutralJetConstituents_withDistance.begin();
	  nextNeutral != neutralJetConstituents_withDistance.end(); ++nextNeutral ) {
      double nextNeutralEn = nextNeutral->pfCandidate_->energy();      
      double resolutionCaloEn = caloResolutionCoeff*sqrt(neutralEnSum + nextNeutralEn);
      double resolution = sqrt(resolutionTrackP*resolutionTrackP + resolutionCaloEn*resolutionCaloEn);
      if ( (neutralEnSum + nextNeutralEn) < (track->p() + 2.*resolution) ) {
	chargedHadron->neutralPFCandidates_.push_back(nextNeutral->pfCandidate_);
	neutralEnSum += nextNeutralEn;
      } else {
	break;
      }
    }

    setChargedHadronP4(*chargedHadron);

    //if ( verbosity_ ) {
    //  chargedHadron->print(std::cout);
    //}

    output.push_back(chargedHadron);
  }

  return output.release();
}

}} // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronBuilderPluginFactory, reco::tau::PFRecoTauChargedHadronFromTrackPlugin, "PFRecoTauChargedHadronFromTrackPlugin");
