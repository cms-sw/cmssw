#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include <memory>
#include <vector>

// Introduce these two types for brevity of casting 
typedef edm::Ptr<reco::PFCandidate> recoCandPtr;
typedef edm::Ptr<pat::PackedCandidate> patCandPtr;

// This template function finds whether theCandidate is in thefootprint 
// collection. It is templated to be able to handle both reco and pat
// photons (from AOD and miniAOD, respectively).
template <class T, class U>
bool isInFootprint(const T& thefootprint, const U& theCandidate) {
  for ( auto itr = thefootprint.begin(); itr != thefootprint.end(); ++itr ) {
    if( itr->key() == theCandidate.key() ) return true;
  }
  return false;
}

class PhotonIDValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit PhotonIDValueMapProducer(const edm::ParameterSet&);
  ~PhotonIDValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void writeValueMap(edm::Event &iEvent,
		     const edm::Handle<edm::View<reco::Photon> > & handle,
		     const std::vector<float> & values,
		     const std::string    & label) const ;
  
  noZS::EcalClusterLazyTools *lazyToolnoZS;

  edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollection_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > pfCandidatesToken_;
  edm::EDGetToken src_;

  std::string dataFormat_;
 
  constexpr static char phoFull5x5SigmaIEtaIEta_[] = "phoFull5x5SigmaIEtaIEta";
  constexpr static char phoChargedIsolation_[] = "phoChargedIsolation";
  constexpr static char phoNeutralHadronIsolation_[] = "phoNeutralHadronIsolation";
  constexpr static char phoPhotonIsolation_[] = "phoPhotonIsolation";

};

constexpr char PhotonIDValueMapProducer::phoFull5x5SigmaIEtaIEta_[];
constexpr char PhotonIDValueMapProducer::phoChargedIsolation_[];
constexpr char PhotonIDValueMapProducer::phoNeutralHadronIsolation_[];
constexpr char PhotonIDValueMapProducer::phoPhotonIsolation_[];

PhotonIDValueMapProducer::PhotonIDValueMapProducer(const edm::ParameterSet& iConfig) {

  dataFormat_ = iConfig.getParameter<std::string>("dataFormat");
  if( dataFormat_ != "RECO" &&  dataFormat_ != "PAT") {
    throw cms::Exception("InvalidConfiguration") 
      << "PhotonIDValueMapProducer runs in \"RECO\" or \"PAT\" mode!";
  }

  //
  // Declare consummables
  //
  ebReducedRecHitCollection_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
							      ("ebReducedRecHitCollection"));
  eeReducedRecHitCollection_ = consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
							      ("eeReducedRecHitCollection"));

  vtxToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));

  // reco photons are castable into pat photons, so no need to handle reco/pat seprately
  src_ = consumes<edm::View<reco::Photon> >(iConfig.getParameter<edm::InputTag>("src"));

  if( dataFormat_ == "RECO" ){

    // The particleBasedIsolation object is relevant only for AOD, RECO format
    particleBasedIsolationToken_ = consumes<edm::ValueMap<std::vector<reco::PFCandidateRef > > >
      (iConfig.getParameter<edm::InputTag>("particleBasedIsolation"));

  }

  // AOD has reco::PFCandidate vector, and miniAOD has pat::PackedCandidate vector.
  // Both inherit from reco::Candidate, so we go to the base class. We can cast into
  // the full type later if needed.
  pfCandidatesToken_ = consumes< edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("pfCandidates")); 

  //
  // Declare producibles
  //
  produces<edm::ValueMap<float> >(phoFull5x5SigmaIEtaIEta_);  
  produces<edm::ValueMap<float> >(phoChargedIsolation_);  
  produces<edm::ValueMap<float> >(phoNeutralHadronIsolation_);  
  produces<edm::ValueMap<float> >(phoPhotonIsolation_);  

}

PhotonIDValueMapProducer::~PhotonIDValueMapProducer() {
}

void PhotonIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  // Constants 
  const float coneSizeDR = 0.3;
  const float dxyMax = 0.1;
  const float dzMax  = 0.2;

  lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_); 
  
  edm::Handle<edm::View<reco::Photon> > src;
  
  iEvent.getByToken(src_, src);

  // Get PV
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);
  if (vertices->empty()) return; // skip the event if no PV found
  const reco::Vertex &pv = vertices->front();
  
  edm::Handle< edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationMap;
  if( dataFormat_ == "RECO" ){
    // this exists only in AOD
    iEvent.getByToken(particleBasedIsolationToken_, particleBasedIsolationMap);
  }

  edm::Handle< edm::View<reco::Candidate>> pfCandidatesHandle;
  iEvent.getByToken(pfCandidatesToken_, pfCandidatesHandle);
  
  if( dataFormat_ == "PAT" && src->size() ) {
    edm::Ptr<pat::Photon> test(src->ptrVector()[0]);
    if( test.isNull() || !test.isAvailable() ) {
      throw cms::Exception("InvalidConfiguration")
	<<"DataFormat set to \"PAT\" but cannot cast to pat::Photon!";
    }
  }

  // size_t n = src->size();
  std::vector<float> phoFull5x5SigmaIEtaIEta;
  std::vector<float> phoChargedIsolation;
  std::vector<float> phoNeutralHadronIsolation;
  std::vector<float> phoPhotonIsolation;
  
  // reco::Photon::superCluster() is virtual so we can exploit polymorphism
  for (unsigned idxpho = 0; idxpho < src->size(); ++idxpho) {
    const auto& iPho = src->ptrAt(idxpho);

    //    
    // Compute full 5x5 quantities
    //
    const auto& theseed = *(iPho->superCluster()->seed());
    
    // For full5x5_sigmaIetaIeta, for 720 we use: lazy tools for AOD,
    // and userFloats for miniAOD. Since some point in 72X, one can
    // retrieve the full5x5 directly from the object with ->full5x5_sigmaIetaIeta()
    // for both formats.
    float see = -999;
    std::vector<float> vCov = lazyToolnoZS->localCovariances( theseed );
    see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
    phoFull5x5SigmaIEtaIEta.push_back(see);

    // 
    // Compute absolute uncorrected isolations with footprint removal
    //
    
    // First, find photon direction with respect to the good PV
    math::XYZVector photon_directionWrtVtx(iPho->superCluster()->x() - pv.x(),
                                           iPho->superCluster()->y() - pv.y(),
                                           iPho->superCluster()->z() - pv.z());

    // Zero the isolation sums
    float chargedIsoSum = 0;
    float neutralHadronIsoSum = 0;
    float photonIsoSum = 0;

    // Loop over all PF candidates
    for (unsigned idxcand = 0; idxcand < pfCandidatesHandle->size(); ++idxcand ){

      // Here, the type will be a simple reco::Candidate. We cast it
      // for full PFCandidate or PackedCandidate below as necessary
      const auto& iCand = pfCandidatesHandle->ptrAt(idxcand);

      // One would think that we should check that this iCand from 
      // the generic PF collection is not identical to the iPho photon
      // for which we are computing the isolations. However, it turns out
      // to be unnecessary. Below, in the function isInFootprint(), we drop
      // this iCand if it is in the footprint, and this always removes
      // the iCand if it matches the iPho.
      //     The explicit check at this point is not totally trivial because
      // of non-triviality of implementation of this check for miniAOD (PackedCandidates
      // of the PF collection do not contain the supercluser link, so can't use that).
      // if( dataFormat_ == "RECO" ){
      //  	if( ((const recoCandPtr)iCand)->superClusterRef() == iPho->superCluster() ) continue;
      // }


      // Check if this candidate is within the isolation cone
      float dR=deltaR(photon_directionWrtVtx.Eta(),photon_directionWrtVtx.Phi(), 
		      iCand->eta(), iCand->phi());
      if( dR > coneSizeDR ) continue;

      // Check if this candidate is not in the footprint
      bool inFootprint = false;
      if(dataFormat_=="RECO") {
      	inFootprint = isInFootprint( (*particleBasedIsolationMap)[iPho], iCand );
      }

      if(dataFormat_=="PAT"){
	
      	edm::Ptr<pat::Photon> patPhotonPtr(src->ptrVector()[idxpho]);
      	inFootprint = isInFootprint(patPhotonPtr->associatedPackedPFCandidates(), iCand);
      }

      if( inFootprint ) continue;
      // debug if( inFootprint ) printf(".");

      // Find candidate type
      reco::PFCandidate::ParticleType thisCandidateType = reco::PFCandidate::X;
      if( dataFormat_ == "RECO" )
	thisCandidateType = ( (const recoCandPtr)iCand)->particleId();
      if( dataFormat_ == "PAT" ){
	// the neutral hadrons and charged hadrons can be of pdgId types
	// only 130 (K0L) and +-211 (pi+-) in packed candidates
	const int pdgId = ( (const patCandPtr)iCand)->pdgId();
	if( pdgId == 22 )
	  thisCandidateType = reco::PFCandidate::gamma;
	else if( abs(pdgId) == 130) // PDG ID for K0L
	  thisCandidateType = reco::PFCandidate::h0;
	else if( abs(pdgId) == 211) // PDG ID for pi+-
	  thisCandidateType = reco::PFCandidate::h;
      }

      // Increment the appropriate isolation sum
      if( thisCandidateType == reco::PFCandidate::h ){
	// for charged hadrons, additionally check consistency
	// with the PV
	float dxy = -999, dz = -999;
	if( dataFormat_ == "RECO" ){
	  dz = ( (const recoCandPtr)iCand)->trackRef()->dz(pv.position());
	  dxy =( (const recoCandPtr)iCand)->trackRef()->dxy(pv.position());
	}
	if( dataFormat_ == "PAT" ){
	  dz = ( (const patCandPtr)iCand)->pseudoTrack().dz(pv.position());
	  dxy =( (const patCandPtr)iCand)->pseudoTrack().dxy(pv.position());
	}
	if (fabs(dz) > dzMax) continue;
	if(fabs(dxy) > dxyMax) continue;
	// The candidate is eligible, increment the isolaiton
	chargedIsoSum += iCand->pt();
      }

      if( thisCandidateType == reco::PFCandidate::h0 )
	neutralHadronIsoSum += iCand->pt();

      if( thisCandidateType == reco::PFCandidate::gamma )
	photonIsoSum += iCand->pt();
    }

    phoChargedIsolation      .push_back( chargedIsoSum       );
    phoNeutralHadronIsolation.push_back( neutralHadronIsoSum );
    phoPhotonIsolation       .push_back( photonIsoSum        );
  }
  
  writeValueMap(iEvent, src, phoFull5x5SigmaIEtaIEta, phoFull5x5SigmaIEtaIEta_);  
  writeValueMap(iEvent, src, phoChargedIsolation, phoChargedIsolation_);  
  writeValueMap(iEvent, src, phoNeutralHadronIsolation, phoNeutralHadronIsolation_);  
  writeValueMap(iEvent, src, phoPhotonIsolation, phoPhotonIsolation_);  
  
  delete lazyToolnoZS;
}

void PhotonIDValueMapProducer::writeValueMap(edm::Event &iEvent,
					     const edm::Handle<edm::View<reco::Photon> > & handle,
					     const std::vector<float> & values,
					     const std::string    & label) const 
{
  using namespace edm; 
  using namespace std;
  auto_ptr<ValueMap<float> > valMap(new ValueMap<float>());
  edm::ValueMap<float>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(valMap, label);
}

void PhotonIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(PhotonIDValueMapProducer);
