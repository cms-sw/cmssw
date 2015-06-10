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
  
  // This function computes charged hadron isolation 
  // with respect to multiple PVs and returns the worst
  // of the found isolation values.
  //   The function implements the computation method taken directly
  // from Run 1 code of H->gamma gamma, specifically from
  // the class CiCPhotonID of the HiggsTo2photons anaysis code.
  //    Template is introduced to handle reco/pat photons and aod/miniAOD
  // PF candidates collections
  template <class T, class U>
  float computeWorstPFChargedIsolation(const T& photon,
				       const U& pfCandidates,
				       const edm::Handle<reco::VertexCollection> vertices,
				       bool isAOD,
				       float dRmax, float dxyMax, float dzMax);

  // Some helper functions that are needed to access info in
  // AOD vs miniAOD
  reco::PFCandidate::ParticleType
  candidatePdgId(const edm::Ptr<reco::Candidate> candidate, bool isAOD);

  const reco::Track* getTrackPointer(const edm::Ptr<reco::Candidate> candidate, bool isAOD);


  // The object that will compute 5x5 quantities  
  noZS::EcalClusterLazyTools *lazyToolnoZS;

  // for AOD case
  edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollection_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > pfCandidatesToken_;
  edm::EDGetToken src_;

  // for miniAOD case
  edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollectionMiniAOD_;
  edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollectionMiniAOD_;
  edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollectionMiniAOD_;
  edm::EDGetTokenT<reco::VertexCollection> vtxTokenMiniAOD_;
  edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationTokenMiniAOD_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > pfCandidatesTokenMiniAOD_;
  edm::EDGetToken srcMiniAOD_;

  // Cluster shapes
  constexpr static char phoFull5x5SigmaIEtaIEta_[] = "phoFull5x5SigmaIEtaIEta";
  constexpr static char phoFull5x5SigmaIEtaIPhi_[] = "phoFull5x5SigmaIEtaIPhi";
  constexpr static char phoFull5x5E1x3_[] = "phoFull5x5E1x3";
  constexpr static char phoFull5x5E2x2_[] = "phoFull5x5E2x2";
  constexpr static char phoFull5x5E2x5Max_[] = "phoFull5x5E2x5Max";
  constexpr static char phoFull5x5E5x5_[] = "phoFull5x5E5x5";
  constexpr static char phoESEffSigmaRR_[] = "phoESEffSigmaRR";
  // Isolations
  constexpr static char phoChargedIsolation_[] = "phoChargedIsolation";
  constexpr static char phoNeutralHadronIsolation_[] = "phoNeutralHadronIsolation";
  constexpr static char phoPhotonIsolation_[] = "phoPhotonIsolation";
  constexpr static char phoWorstChargedIsolation_[] = "phoWorstChargedIsolation";

};

// Cluster shapes
constexpr char PhotonIDValueMapProducer::phoFull5x5SigmaIEtaIEta_[];
constexpr char PhotonIDValueMapProducer::phoFull5x5SigmaIEtaIPhi_[];
constexpr char PhotonIDValueMapProducer::phoFull5x5E1x3_[];
constexpr char PhotonIDValueMapProducer::phoFull5x5E2x2_[];
constexpr char PhotonIDValueMapProducer::phoFull5x5E2x5Max_[];
constexpr char PhotonIDValueMapProducer::phoFull5x5E5x5_[];
constexpr char PhotonIDValueMapProducer::phoESEffSigmaRR_[];
// Isolations
constexpr char PhotonIDValueMapProducer::phoChargedIsolation_[];
constexpr char PhotonIDValueMapProducer::phoNeutralHadronIsolation_[];
constexpr char PhotonIDValueMapProducer::phoPhotonIsolation_[];
constexpr char PhotonIDValueMapProducer::phoWorstChargedIsolation_[];

PhotonIDValueMapProducer::PhotonIDValueMapProducer(const edm::ParameterSet& iConfig) {

  //
  // Declare consummables, handle both AOD and miniAOD case
  //
  ebReducedRecHitCollection_        = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("ebReducedRecHitCollection"));
  ebReducedRecHitCollectionMiniAOD_ = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("ebReducedRecHitCollectionMiniAOD"));

  eeReducedRecHitCollection_        = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("eeReducedRecHitCollection"));
  eeReducedRecHitCollectionMiniAOD_ = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("eeReducedRecHitCollectionMiniAOD"));

  esReducedRecHitCollection_        = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("esReducedRecHitCollection"));
  esReducedRecHitCollectionMiniAOD_ = mayConsume<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>
								       ("esReducedRecHitCollectionMiniAOD"));

  vtxToken_        = mayConsume<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));
  vtxTokenMiniAOD_ = mayConsume<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("verticesMiniAOD"));

  // reco photons are castable into pat photons, so no need to handle reco/pat seprately
  src_        = mayConsume<edm::View<reco::Photon> >(iConfig.getParameter<edm::InputTag>("src"));
  srcMiniAOD_ = mayConsume<edm::View<reco::Photon> >(iConfig.getParameter<edm::InputTag>("srcMiniAOD"));

  // The particleBasedIsolation object is relevant only for AOD, RECO format
  particleBasedIsolationToken_ = mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef > > >
    (iConfig.getParameter<edm::InputTag>("particleBasedIsolation"));
  
  // AOD has reco::PFCandidate vector, and miniAOD has pat::PackedCandidate vector.
  // Both inherit from reco::Candidate, so we go to the base class. We can cast into
  // the full type later if needed. Since the collection names are different, we
  // introduce both collections
  pfCandidatesToken_        = mayConsume< edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("pfCandidates")); 
  pfCandidatesTokenMiniAOD_ = mayConsume< edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("pfCandidatesMiniAOD")); 

  //
  // Declare producibles
  //
  // Cluster shapes
  produces<edm::ValueMap<float> >(phoFull5x5SigmaIEtaIEta_);  
  produces<edm::ValueMap<float> >(phoFull5x5SigmaIEtaIPhi_);  
  produces<edm::ValueMap<float> >(phoFull5x5E1x3_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x2_);  
  produces<edm::ValueMap<float> >(phoFull5x5E2x5Max_);  
  produces<edm::ValueMap<float> >(phoFull5x5E5x5_);  
  produces<edm::ValueMap<float> >(phoESEffSigmaRR_);  
  // Isolations
  produces<edm::ValueMap<float> >(phoChargedIsolation_);  
  produces<edm::ValueMap<float> >(phoNeutralHadronIsolation_);  
  produces<edm::ValueMap<float> >(phoPhotonIsolation_);  
  produces<edm::ValueMap<float> >(phoWorstChargedIsolation_);  

}

PhotonIDValueMapProducer::~PhotonIDValueMapProducer() {
}

void PhotonIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using namespace edm;

  // Constants 
  const float coneSizeDR = 0.3;
  const float dxyMax = 0.1;
  const float dzMax  = 0.2;

  edm::Handle<edm::View<reco::Photon> > src;

  bool isAOD = true;
  iEvent.getByToken(src_, src);
  if( !src.isValid() ){
    isAOD = false;
    iEvent.getByToken(srcMiniAOD_,src);
  }
  if( !src.isValid() ) {
    throw cms::Exception("IllDefinedDataTier")
      << "DataFormat does not contain a photon source!";
  }

  // Configure Lazy Tools
  if( isAOD )
    lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, 
						  ebReducedRecHitCollection_, 
						  eeReducedRecHitCollection_,
						  esReducedRecHitCollection_); 
  else
    lazyToolnoZS = new noZS::EcalClusterLazyTools(iEvent, iSetup, 
						  ebReducedRecHitCollectionMiniAOD_, 
						  eeReducedRecHitCollectionMiniAOD_,
						  esReducedRecHitCollectionMiniAOD_); 
  
  // Get PV
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);
  if( !vertices.isValid() )
    iEvent.getByToken(vtxTokenMiniAOD_, vertices);
  if (vertices->empty()) return; // skip the event if no PV found
  const reco::Vertex &pv = vertices->front();
  
  edm::Handle< edm::ValueMap<std::vector<reco::PFCandidateRef > > > particleBasedIsolationMap;
  if( isAOD ){
    // this exists only in AOD
    iEvent.getByToken(particleBasedIsolationToken_, particleBasedIsolationMap);
  }

  edm::Handle< edm::View<reco::Candidate> > pfCandidatesHandle;

  iEvent.getByToken(pfCandidatesToken_, pfCandidatesHandle);
  if( !pfCandidatesHandle.isValid() )
    iEvent.getByToken(pfCandidatesTokenMiniAOD_, pfCandidatesHandle);

  if( !isAOD && src->size() ) {
    edm::Ptr<pat::Photon> test(src->ptrAt(0));
    if( test.isNull() || !test.isAvailable() ) {
      throw cms::Exception("InvalidConfiguration")
	<<"DataFormat is detected as miniAOD but cannot cast to pat::Photon!";
    }
  }

  // size_t n = src->size();
  // Cluster shapes
  std::vector<float> phoFull5x5SigmaIEtaIEta;
  std::vector<float> phoFull5x5SigmaIEtaIPhi;
  std::vector<float> phoFull5x5E1x3;
  std::vector<float> phoFull5x5E2x2;
  std::vector<float> phoFull5x5E2x5Max;
  std::vector<float> phoFull5x5E5x5;
  std::vector<float> phoESEffSigmaRR;
  // Isolations
  std::vector<float> phoChargedIsolation;
  std::vector<float> phoNeutralHadronIsolation;
  std::vector<float> phoPhotonIsolation;
  std::vector<float> phoWorstChargedIsolation;
  
  // reco::Photon::superCluster() is virtual so we can exploit polymorphism
  for (unsigned idxpho = 0; idxpho < src->size(); ++idxpho) {
    const auto& iPho = src->ptrAt(idxpho);

    //    
    // Compute full 5x5 quantities
    //
    const auto& theseed = *(iPho->superCluster()->seed());
    
    // For full5x5_sigmaIetaIeta, for 720 we use: lazy tools for AOD,
    // and userFloats or lazy tools for miniAOD. From some point in 72X and on, one can
    // retrieve the full5x5 directly from the object with ->full5x5_sigmaIetaIeta()
    // for both formats.
    float see = -999;
    std::vector<float> vCov = lazyToolnoZS->localCovariances( theseed );
    see = (isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
    float sep = vCov[1];
    phoFull5x5SigmaIEtaIEta.push_back(see);
    phoFull5x5SigmaIEtaIPhi.push_back(sep);

    phoFull5x5E1x3   .push_back(lazyToolnoZS-> e1x3   (theseed) );
    phoFull5x5E2x2   .push_back(lazyToolnoZS-> e2x2   (theseed) );
    phoFull5x5E2x5Max.push_back(lazyToolnoZS-> e2x5Max(theseed) );
    phoFull5x5E5x5   .push_back(lazyToolnoZS-> e5x5   (theseed) );

    phoESEffSigmaRR  .push_back(lazyToolnoZS->eseffsirir( *(iPho->superCluster()) ) );

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
      // if( isAOD ){
      //  	if( ((const recoCandPtr)iCand)->superClusterRef() == iPho->superCluster() ) continue;
      // }


      // Check if this candidate is within the isolation cone
      float dR2 = deltaR2(photon_directionWrtVtx.Eta(),photon_directionWrtVtx.Phi(), 
			iCand->eta(), iCand->phi());
      if( dR2 > coneSizeDR*coneSizeDR ) continue;

      // Check if this candidate is not in the footprint
      bool inFootprint = false;
      if(isAOD) {
      	inFootprint = isInFootprint( (*particleBasedIsolationMap)[iPho], iCand );
      } else {	
      	edm::Ptr<pat::Photon> patPhotonPtr(src->ptrAt(idxpho));
      	inFootprint = isInFootprint(patPhotonPtr->associatedPackedPFCandidates(), iCand);
      }

      if( inFootprint ) continue;

      // Find candidate type
      reco::PFCandidate::ParticleType thisCandidateType = candidatePdgId(iCand, isAOD);

      // Increment the appropriate isolation sum
      if( thisCandidateType == reco::PFCandidate::h ){
	// for charged hadrons, additionally check consistency
	// with the PV
	const reco::Track *theTrack = getTrackPointer( iCand, isAOD );

	float dxy = theTrack->dxy(pv.position());
	if(fabs(dxy) > dxyMax) continue;

	float dz  = theTrack->dz(pv.position());
	if (fabs(dz) > dzMax) continue;

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

    float worstChargedIso =
      computeWorstPFChargedIsolation(iPho, pfCandidatesHandle, vertices, 
				     isAOD, coneSizeDR, dxyMax, dzMax);
    phoWorstChargedIsolation .push_back( worstChargedIso );
    

  }
  
  // Cluster shapes
  writeValueMap(iEvent, src, phoFull5x5SigmaIEtaIEta, phoFull5x5SigmaIEtaIEta_);  
  writeValueMap(iEvent, src, phoFull5x5SigmaIEtaIPhi, phoFull5x5SigmaIEtaIPhi_);  
  writeValueMap(iEvent, src, phoFull5x5E1x3   , phoFull5x5E1x3_);  
  writeValueMap(iEvent, src, phoFull5x5E2x2   , phoFull5x5E2x2_);  
  writeValueMap(iEvent, src, phoFull5x5E2x5Max, phoFull5x5E2x5Max_);  
  writeValueMap(iEvent, src, phoFull5x5E5x5   , phoFull5x5E5x5_);  
  writeValueMap(iEvent, src, phoESEffSigmaRR  , phoESEffSigmaRR_);  
  // Isolations
  writeValueMap(iEvent, src, phoChargedIsolation, phoChargedIsolation_);  
  writeValueMap(iEvent, src, phoNeutralHadronIsolation, phoNeutralHadronIsolation_);  
  writeValueMap(iEvent, src, phoPhotonIsolation, phoPhotonIsolation_);  
  writeValueMap(iEvent, src, phoWorstChargedIsolation, phoWorstChargedIsolation_);  
  
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

// Charged isolation with respect to the worst vertex. See more
// comments above at the function declaration.
template <class T, class U>
float PhotonIDValueMapProducer
::computeWorstPFChargedIsolation(const T& photon, const U& pfCandidates,
				 const edm::Handle<reco::VertexCollection> vertices,
				 bool isAOD,
				 float dRmax, float dxyMax, float dzMax){

  float worstIsolation = 999;
  std::vector<float> allIsolations;

  // Constants below: there are no vetos and no min pt requirement,
  // just like in the original H->gamma gamma code.
  const float dRvetoBarrel = 0.0;
  const float dRvetoEndcap = 0.0;
  const float ptMin = 0.0;
  
  float dRveto;
  if (photon->isEB())
    dRveto = dRvetoBarrel;
  else
    dRveto = dRvetoEndcap;

  //Calculate isolation sum separately for each vertex
  for(unsigned int ivtx=0; ivtx<vertices->size(); ++ivtx) {
    
    // Shift the photon according to the vertex
    reco::VertexRef vtx(vertices, ivtx);
    math::XYZVector photon_directionWrtVtx(photon->superCluster()->x() - vtx->x(),
					   photon->superCluster()->y() - vtx->y(),
					   photon->superCluster()->z() - vtx->z());
    
    float sum = 0;
    // Loop over the PFCandidates
    for(unsigned i=0; i<pfCandidates->size(); i++) {
      
      const auto& iCand = pfCandidates->ptrAt(i);

      //require that PFCandidate is a charged hadron
      reco::PFCandidate::ParticleType thisCandidateType = candidatePdgId(iCand, isAOD);
      if (thisCandidateType != reco::PFCandidate::h) 
	continue;

      if (iCand->pt() < ptMin)
	continue;
      
      const reco::Track *theTrack = getTrackPointer( iCand, isAOD );
      float dxy = theTrack->dxy(vtx->position());
      if( fabs(dxy) > dxyMax) continue;
      
      float dz = theTrack->dz(vtx->position());
      if ( fabs(dz) > dzMax) continue;
      
      float dR2 = deltaR2(photon_directionWrtVtx.Eta(), photon_directionWrtVtx.Phi(), 
                          iCand->eta(),      iCand->phi());
      if(dR2 > dRmax*dRmax || dR2 < dRveto*dRveto) continue;
      
      sum += iCand->pt();
    }

    allIsolations.push_back(sum);
  }

  if( allIsolations.size()>0 )
    worstIsolation = * std::max_element( allIsolations.begin(), allIsolations.end() );
  
  return worstIsolation;
}

reco::PFCandidate::ParticleType
PhotonIDValueMapProducer::candidatePdgId(const edm::Ptr<reco::Candidate> candidate, 
					 bool isAOD){
  
  reco::PFCandidate::ParticleType thisCandidateType = reco::PFCandidate::X;
  if( isAOD )
    thisCandidateType = ( (const recoCandPtr)candidate)->particleId();
  else {
    // the neutral hadrons and charged hadrons can be of pdgId types
    // only 130 (K0L) and +-211 (pi+-) in packed candidates
    const int pdgId = ( (const patCandPtr)candidate)->pdgId();
    if( pdgId == 22 )
      thisCandidateType = reco::PFCandidate::gamma;
    else if( abs(pdgId) == 130) // PDG ID for K0L
      thisCandidateType = reco::PFCandidate::h0;
    else if( abs(pdgId) == 211) // PDG ID for pi+-
      thisCandidateType = reco::PFCandidate::h;
  }
  return thisCandidateType;
}

const reco::Track* 
PhotonIDValueMapProducer::getTrackPointer(const edm::Ptr<reco::Candidate> candidate, bool isAOD){

  const reco::Track* theTrack = nullptr;
  if( isAOD )
    theTrack = &*( ((const recoCandPtr) candidate)->trackRef());
  else
    theTrack = &( ((const patCandPtr) candidate)->pseudoTrack());

  return theTrack;
}

DEFINE_FWK_MODULE(PhotonIDValueMapProducer);
