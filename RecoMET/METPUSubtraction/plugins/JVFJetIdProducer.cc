#include "RecoMET/METPUSubtraction/plugins/JVFJetIdProducer.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"


enum { kNeutralJetPU, kNeutralJetNoPU };

JVFJetIdProducer::JVFJetIdProducer(const edm::ParameterSet& cfg)
{
  srcJets_ = consumes<reco::PFJetCollection>(cfg.getParameter<edm::InputTag>("srcJets"));

  srcPFCandidates_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCandidates"));
  srcPFCandToVertexAssociations_ = consumes<PFCandToVertexAssMap>(cfg.getParameter<edm::InputTag>("srcPFCandToVertexAssociations") );
  srcHardScatterVertex_ = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("srcHardScatterVertex") );
  minTrackPt_ = cfg.getParameter<double>("minTrackPt");
  dZcut_ = cfg.getParameter<double>("dZcut");

  JVFcut_ = cfg.getParameter<double>("JVFcut");

  std::string neutralJetOption_string = cfg.getParameter<std::string>("neutralJetOption");
  if      ( neutralJetOption_string == "PU"   ) neutralJetOption_ = kNeutralJetPU;
  else if ( neutralJetOption_string == "noPU" ) neutralJetOption_ = kNeutralJetNoPU;
  else throw cms::Exception("JVFJetIdProducer")
    << "Invalid Configuration Parameter 'neutralJetOption' = " << neutralJetOption_string << " !!\n";

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<edm::ValueMap<double> >("Discriminant");
  produces<edm::ValueMap<int> >("Id");
}

JVFJetIdProducer::~JVFJetIdProducer()
{
// nothing to be done yet...
}

namespace
{
  double computeJVF(const reco::PFJet& jet, 
		    const PFCandToVertexAssMap& pfCandToVertexAssociations,
		    const reco::VertexCollection& vertices, double dZ, double minTrackPt,
		    int verbosity) {

    LogDebug ("computeJVF")
      << "<computeJVF>:" << std::endl
      << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << std::endl;
    
    double trackSum_isVtxAssociated    = 0.;
    double trackSum_isNotVtxAssociated = 0.;
    
    std::vector<reco::PFCandidatePtr> pfConsts = jet.getPFConstituents();
    for ( std::vector<reco::PFCandidatePtr>::const_iterator jetConstituent = pfConsts.begin(); jetConstituent != pfConsts.end(); ++jetConstituent ) {
      
      if ( (*jetConstituent)->charge() == 0 ) continue;
	
	double trackPt = 0.;
	if( (*jetConstituent)->gsfTrackRef().isNonnull() && (*jetConstituent)->gsfTrackRef().isAvailable() ) trackPt = (*jetConstituent)->gsfTrackRef()->pt();
	else if ( (*jetConstituent)->trackRef().isNonnull() && (*jetConstituent)->trackRef().isAvailable() ) trackPt =(*jetConstituent)->trackRef()->pt();
	else trackPt = (*jetConstituent)->pt();

	if ( trackPt > minTrackPt ) {
	  int jetConstituent_vtxAssociationType = noPuUtils::isVertexAssociated( (*jetConstituent), pfCandToVertexAssociations, vertices, dZ);
	    //isVertexAssociated_fast(pfCandidateRef, pfCandToVertexAssociations_reversed, *hardScatterVertex, dZcut_, numWarnings_, maxWarnings_);
	  bool jetConstituent_isVtxAssociated = (jetConstituent_vtxAssociationType == noPuUtils::kChHSAssoc ); 
	  double jetConstituentPt = (*jetConstituent)->pt();
	  if ( jetConstituent_isVtxAssociated ) {
	    LogDebug ("computeJVF")
	      << "associated track: Pt = " << (*jetConstituent)->pt() << ", eta = " << (*jetConstituent)->eta() << ", phi = " << (*jetConstituent)->phi() << std::endl
	      << " (vtxAssociationType = " << jetConstituent_vtxAssociationType << ")" << std::endl;
	    
	    trackSum_isVtxAssociated += jetConstituentPt;
	  } else {
	    LogDebug ("computeJVF")
	      << "unassociated track: Pt = " << (*jetConstituent)->pt() << ", eta = " << (*jetConstituent)->eta() << ", phi = " << (*jetConstituent)->phi() << std::endl
	      << " (vtxAssociationType = " << jetConstituent_vtxAssociationType << ")" << std::endl;
	   
	    trackSum_isNotVtxAssociated += jetConstituentPt;
	  }
	}
    }

    double trackSum = trackSum_isVtxAssociated + trackSum_isNotVtxAssociated;

    double jvf = -1.;
    if ( std::abs(jet.eta()) < 2.5 && trackSum > 5. ) {
      jvf = trackSum_isVtxAssociated/trackSum;
    }

    LogDebug ("computeJVF")
      << "trackSum: associated = " << trackSum_isVtxAssociated << ", unassociated = " << trackSum_isNotVtxAssociated << std::endl
      << " --> JVF = " << jvf << std::endl;

    return jvf;
  }
}

void JVFJetIdProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
// get jets 
  edm::Handle<reco::PFJetCollection> jets;
  evt.getByToken(srcJets_, jets);
  
  // get PFCandidates
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByToken(srcPFCandidates_, pfCandidates);
 
  // get PFCandidate-to-vertex associations and "the" hard-scatter vertex
  edm::Handle<PFCandToVertexAssMap> pfCandToVertexAssociations;
  evt.getByToken(srcPFCandToVertexAssociations_, pfCandToVertexAssociations);
 
  edm::Handle<reco::VertexCollection> hardScatterVertex;
  evt.getByToken(srcHardScatterVertex_, hardScatterVertex);
 
  std::vector<double> jetIdDiscriminants;
  std::vector<int> jetIdFlags;
 
  size_t numJets = jets->size();
  for ( size_t iJet = 0; iJet < numJets; ++iJet ) {
    reco::PFJetRef jet(jets, iJet);
 
    double jetJVF = computeJVF(*jet, *pfCandToVertexAssociations, *hardScatterVertex, dZcut_, minTrackPt_, verbosity_ && jet->pt() > 20.);
    jetIdDiscriminants.push_back(jetJVF);
 
    int jetIdFlag = 0;
    if ( jetJVF > JVFcut_ ) jetIdFlag = 255;
    else if ( jetJVF < -0.5 && neutralJetOption_ == kNeutralJetNoPU ) jetIdFlag = 255;
    jetIdFlags.push_back(jetIdFlag);
  }
 
  std::auto_ptr<edm::ValueMap<double> > jetIdDiscriminants_ptr(new edm::ValueMap<double>());
  edm::ValueMap<double>::Filler jetIdDiscriminantFiller(*jetIdDiscriminants_ptr);
  jetIdDiscriminantFiller.insert(jets, jetIdDiscriminants.begin(), jetIdDiscriminants.end());
  jetIdDiscriminantFiller.fill();
 
  std::auto_ptr<edm::ValueMap<int> > jetIdFlags_ptr(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler jetIdFlagFiller(*jetIdFlags_ptr);
  jetIdFlagFiller.insert(jets, jetIdFlags.begin(), jetIdFlags.end());
  jetIdFlagFiller.fill();

  evt.put(jetIdDiscriminants_ptr, "Discriminant");
  evt.put(jetIdFlags_ptr, "Id");
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(JVFJetIdProducer);
