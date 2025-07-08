// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


class MuonExtendedTableProducer : public edm::global::EDProducer<> {
  public:
    explicit MuonExtendedTableProducer(const edm::ParameterSet &iConfig) :
      name_(iConfig.getParameter<std::string>("name")),
      rhoTag_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))),
      muonTag_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      dsaMuonTag_(consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("dsaMuons"))),
      vtxTag_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
      bsTag_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
      generalTrackTag_(consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("generalTracks"))),
      jetTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      jetFatTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetsFat"))),
      jetSubTag_(consumes<std::vector<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jetsSub"))),
      transientTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder")))
    {
      produces<nanoaod::FlatTable>();
    }

    ~MuonExtendedTableProducer() override {};

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("rho")->setComment("input rho parameter");
      desc.add<edm::InputTag>("muons")->setComment("input muon collection");
      desc.add<edm::InputTag>("dsaMuons")->setComment("input displaced standalone muon collection");
      desc.add<edm::InputTag>("primaryVertex")->setComment("input primary vertex collection");
      desc.add<edm::InputTag>("beamspot")->setComment("input beamspot collection");
      desc.add<edm::InputTag>("generalTracks")->setComment("input generalTracks collection");
      desc.add<edm::InputTag>("jets")->setComment("input jet collection");
      desc.add<edm::InputTag>("jetsFat")->setComment("input fat jet collection");
      desc.add<edm::InputTag>("jetsSub")->setComment("input sub jet collection");
      desc.add<std::string>("name")->setComment("name of the muon nanoaod::FlatTable we are extending");
      descriptions.add("muonTable", desc);
    }

  private:
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

    int getMatches(const pat::Muon& muon, const reco::Track& dsaMuon, const float minPositionDiff) const;

    float getPFIso(const pat::Muon& muon) const;
    int findMatchedJet(const reco::Candidate& lepton, const edm::Handle< std::vector< pat::Jet > >& jets) const;
    void fillLeptonJetVariables(const reco::Muon *mu, edm::Handle< std::vector< pat::Jet > >& jets, const reco::Vertex& vertex, const double rho, std::vector<int> *jetIdx, std::vector<float> relIso0p4, std::vector<float> *jetPtRatio, std::vector<float> *jetPtRel, std::vector<int> *jetSelectedChargedMultiplicity) const;

    std::string name_;
    edm::EDGetTokenT<double> rhoTag_;
    edm::EDGetTokenT<std::vector<pat::Muon>> muonTag_;
    edm::EDGetTokenT<std::vector<reco::Track>> dsaMuonTag_;
    edm::EDGetTokenT<reco::VertexCollection> vtxTag_;
    edm::EDGetTokenT<reco::BeamSpot> bsTag_;
    edm::EDGetTokenT<std::vector<reco::Track>> generalTrackTag_;
    edm::EDGetTokenT<std::vector<pat::Jet> > jetTag_;
    edm::EDGetTokenT<std::vector<pat::Jet> > jetFatTag_;
    edm::EDGetTokenT<std::vector<pat::Jet> > jetSubTag_;
    edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;  
};

void MuonExtendedTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const
{

  float minPositionDiffForMatching = 1e-6;

  edm::Handle<double> rhoHandle;
  iEvent.getByToken(rhoTag_, rhoHandle);
  edm::Handle<std::vector<pat::Muon>> muons;
  iEvent.getByToken(muonTag_, muons);
  edm::Handle<std::vector<reco::Track>> dsaMuons;
  iEvent.getByToken(dsaMuonTag_, dsaMuons);
  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vtxTag_, primaryVertices);
  edm::Handle<reco::BeamSpot> beamspots;
  iEvent.getByToken(bsTag_, beamspots);
  edm::Handle<std::vector<reco::Track>> generalTracks;
  iEvent.getByToken(generalTrackTag_, generalTracks);
  edm::Handle<std::vector<pat::Jet> > jetHandle;
  iEvent.getByToken(jetTag_, jetHandle);
  edm::Handle<std::vector<pat::Jet> > jetFatHandle;
  iEvent.getByToken(jetFatTag_, jetFatHandle);
  edm::Handle<std::vector<pat::Jet> > jetSubHandle;
  iEvent.getByToken(jetSubTag_, jetSubHandle);

  const auto& pv = primaryVertices->at(0);
  GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

  const auto& bs = beamspots->position();
  GlobalPoint beamSpot(bs.x(), bs.y(), bs.z());
  reco::Vertex beamSpotVertex(beamspots->position(), beamspots->covariance3D());

  edm::ESHandle<TransientTrackBuilder> builder = iSetup.getHandle(transientTrackBuilderToken_);

  unsigned int nMuons = muons->size();
  unsigned int nDsaMuons = dsaMuons->size();  

  std::vector<float> idx, charge, trkPt, trkPtErr;

  std::vector<float> innerTrackValidFraction, globalTrackNormalizedChi2, CQChi2Position, CQTrackKink;
  std::vector<int> numberOfMatchedStation, numberOfValidPixelHits, numberOfValidTrackerHits, numberInnerHitsMissing, trackerLayersWithMeasurement, numberInnerHits;
  std::vector<float> relIso0p4;
  std::vector<float> jetPtRatio, jetPtRel;
  std::vector<int> jetIdx;
  std::vector<int> jetFatIdx, jetSubIdx;
  std::vector<int> jetSelectedChargedMultiplicity;
  
  std::vector<float> dzPV,dzPVErr,dxyPVTraj,dxyPVTrajErr,dxyPVSigned,dxyPVSignedErr,ip3DPVSigned,ip3DPVSignedErr;

  std::vector<float> trkNumPlanes,trkNumHits,trkNumDTHits,trkNumCSCHits,trkNumPixelHits(nMuons,-1),trkNumTrkLayers(nMuons,-1),normChi2;
  std::vector<float> outerEta(nMuons,-5),outerPhi(nMuons,-5);
  std::vector<float> innerVx(nMuons,-1),innerVy(nMuons,-1),innerVz(nMuons,-1),innerPt(nMuons,-1),innerEta(nMuons,-5),innerPhi(nMuons,-5);

  std::vector<std::vector<float>> nMatchesPerDSA;
  std::vector<float> dsaMatch1,dsaMatch1idx,dsaMatch2,dsaMatch2idx,dsaMatch3,dsaMatch3idx,dsaMatch4,dsaMatch4idx,dsaMatch5,dsaMatch5idx;

  for (unsigned int i = 0; i < nMuons; i++) {
    const pat::Muon & muon = (*muons)[i];
    const pat::MuonRef muonRef(muons,i);

    reco::TrackRef trackRef;

    if(muon.isGlobalMuon()) {
      trackRef = muon.combinedMuon();
    }
    else if (muon.isStandAloneMuon()) {
      trackRef = muon.standAloneMuon();
    }
    else {
      trackRef = muon.tunePMuonBestTrack();
    }

    idx.push_back(i);

    const auto& track = trackRef.get();
    reco::TransientTrack transientTrack = builder->build(track);

    charge.push_back(muon.charge());

    trkPt.push_back(track->pt());
    trkPtErr.push_back(track->ptError());

    relIso0p4.push_back(getPFIso(muon));

    fillLeptonJetVariables(&muon, jetHandle, pv, *rhoHandle, &jetIdx, relIso0p4, &jetPtRatio, &jetPtRel, &jetSelectedChargedMultiplicity);

    const reco::Candidate *mu_cand = dynamic_cast<const reco::Candidate*>(&muon);
    jetFatIdx.push_back(findMatchedJet(*mu_cand, jetFatHandle));
    jetSubIdx.push_back(findMatchedJet(*mu_cand, jetSubHandle));

    innerTrackValidFraction.push_back((!muon.innerTrack().isNull()) ? muon.innerTrack()->validFraction() : -1);
    globalTrackNormalizedChi2.push_back((!muon.globalTrack().isNull()) ? muon.globalTrack()->normalizedChi2() : -1);
    CQChi2Position.push_back(muon.combinedQuality().chi2LocalPosition);
    CQTrackKink.push_back(muon.combinedQuality().trkKink);
    numberOfMatchedStation.push_back(muon.numberOfMatchedStations());
    numberOfValidPixelHits.push_back((!muon.innerTrack().isNull()) ? muon.innerTrack()->hitPattern().numberOfValidPixelHits() : 0);
    numberOfValidTrackerHits.push_back((!muon.innerTrack().isNull()) ? muon.innerTrack()->hitPattern().numberOfValidTrackerHits() : 0);
    numberInnerHitsMissing.push_back(!muon.innerTrack().isNull() ? muon.innerTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) : 0);
    trackerLayersWithMeasurement.push_back((!muon.innerTrack().isNull()) ? muon.innerTrack()->hitPattern().trackerLayersWithMeasurement() : 0);
    numberInnerHits.push_back((!muon.globalTrack().isNull()) ? muon.globalTrack()->hitPattern().numberOfValidMuonHits() : (!muon.outerTrack().isNull() ? muon.outerTrack()->hitPattern().numberOfValidMuonHits() : 0));

    dzPV.push_back(track->dz(pv.position()));
    dzPVErr.push_back(std::hypot(track->dzError(), pv.zError()));
    TrajectoryStateClosestToPoint trajectoryPV = transientTrack.trajectoryStateClosestToPoint(primaryVertex);
    dxyPVTraj.push_back(trajectoryPV.perigeeParameters().transverseImpactParameter());
    dxyPVTrajErr.push_back(trajectoryPV.perigeeError().transverseImpactParameterError());
    GlobalVector muonRefTrackDir(muon.px(),muon.py(),muon.pz());
    dxyPVSigned.push_back(IPTools::signedTransverseImpactParameter(transientTrack, muonRefTrackDir, pv).second.value());
    dxyPVSignedErr.push_back(IPTools::signedTransverseImpactParameter(transientTrack, muonRefTrackDir, pv).second.error());
    
    ip3DPVSigned.push_back(IPTools::signedImpactParameter3D(transientTrack, muonRefTrackDir, beamSpotVertex).second.value());
    ip3DPVSignedErr.push_back(IPTools::signedImpactParameter3D(transientTrack, muonRefTrackDir, beamSpotVertex).second.error());  

    trkNumPlanes.push_back(track->hitPattern().muonStationsWithValidHits());
    trkNumHits.push_back(track->hitPattern().numberOfValidMuonHits());
    trkNumDTHits.push_back(track->hitPattern().numberOfValidMuonDTHits());
    trkNumCSCHits.push_back(track->hitPattern().numberOfValidMuonCSCHits());

    normChi2.push_back(track->normalizedChi2());

    // Cannot get outer track for tracker muons
    if (track->extra().isNonnull() && track->extra().isAvailable() && track->outerOk()) {
      outerEta[i] = track->outerEta();
      outerPhi[i] = track->outerPhi();
    }

    if(muon.innerTrack().isNonnull() && muon.innerTrack().isAvailable()){
      innerVx[i] = muon.innerTrack()->vx();
      innerVy[i] = muon.innerTrack()->vy();
      innerVz[i] = muon.innerTrack()->vz();
      innerPt[i] = muon.innerTrack()->pt();
      innerEta[i] = muon.innerTrack()->eta();
      innerPhi[i] = muon.innerTrack()->phi();
      trkNumPixelHits[i] = muon.innerTrack()->hitPattern().numberOfValidPixelHits();
      trkNumTrkLayers[i] = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement();
    }

    std::vector<std::pair<float, float>> dsaMatches(5, std::make_pair(-1.0,-1.0));
    std::vector<float> nDsaMatches;
    for (unsigned int j = 0; j < nDsaMuons; j++){
      if (j > 4) break;
      const reco::Track & dsaMuon = (*dsaMuons)[j];
      // Muon-DSA Matches Table
      int nMatches = getMatches(muon, dsaMuon, minPositionDiffForMatching);
      dsaMatches[j] = std::make_pair(nMatches, j);
      nDsaMatches.push_back(nMatches);
    }
    nMatchesPerDSA.push_back(nDsaMatches);
    std::sort(dsaMatches.rbegin(), dsaMatches.rend());
    dsaMatch1.push_back(dsaMatches[0].first);
    dsaMatch1idx.push_back(dsaMatches[0].second);
    dsaMatch2.push_back(dsaMatches[1].first);
    dsaMatch2idx.push_back(dsaMatches[1].second);
    dsaMatch3.push_back(dsaMatches[2].first);
    dsaMatch3idx.push_back(dsaMatches[2].second);
    dsaMatch4.push_back(dsaMatches[3].first);
    dsaMatch4idx.push_back(dsaMatches[3].second);
    dsaMatch5.push_back(dsaMatches[4].first);
    dsaMatch5idx.push_back(dsaMatches[4].second);
  }

  auto tab  = std::make_unique<nanoaod::FlatTable>(nMuons, name_, false, true);
  tab->addColumn<float>("idx", idx, "EXOnanoAOD muon index");
  
  tab->addColumn<float>("trkPt", trkPt, "");
  tab->addColumn<float>("trkPtErr", trkPtErr, "");

  tab->addColumn<float>("relIso0p4", relIso0p4, "");
  tab->addColumn<float>("jetPtRatio", jetPtRatio, "");
  tab->addColumn<float>("jetPtRel", jetPtRel, "");
  tab->addColumn<int>("jetSelectedChargedMultiplicity", jetSelectedChargedMultiplicity, "");
  tab->addColumn<int>("jetIdx", jetIdx, "");
  tab->addColumn<int>("jetFatIdx", jetFatIdx, "");
  tab->addColumn<int>("jetSubIdx", jetSubIdx, "");
  
  tab->addColumn<float>("innerTrackValidFraction", innerTrackValidFraction, "");
  tab->addColumn<float>("globalTrackNormalizedChi2", globalTrackNormalizedChi2, "");
  tab->addColumn<float>("CQChi2Position", CQChi2Position, "");
  tab->addColumn<float>("CQTrackKink", CQTrackKink, "");
  tab->addColumn<int>("numberOfMatchedStation", numberOfMatchedStation, "");
  tab->addColumn<int>("numberOfValidPixelHits", numberOfValidPixelHits, "");
  tab->addColumn<int>("numberOfValidTrackerHits", numberOfValidTrackerHits, "");
  tab->addColumn<int>("numberInnerHitsMissing", numberInnerHitsMissing, "");
  tab->addColumn<int>("trackerLayersWithMeasurement", trackerLayersWithMeasurement, "");
  tab->addColumn<int>("numberInnerHits", numberInnerHits, "");

  tab->addColumn<float>("dzPV", dzPV, "");
  tab->addColumn<float>("dzPVErr", dzPVErr, "");
  tab->addColumn<float>("dxyPVTraj", dxyPVTraj, "");
  tab->addColumn<float>("dxyPVTrajErr", dxyPVTrajErr, "");
  tab->addColumn<float>("dxyPVSigned", dxyPVSigned, "");
  tab->addColumn<float>("dxyPVSignedErr", dxyPVSignedErr, "");
  tab->addColumn<float>("ip3DPVSigned", ip3DPVSigned, "");
  tab->addColumn<float>("ip3DPVSignedErr", ip3DPVSignedErr, "");

  tab->addColumn<float>("trkNumPlanes", trkNumPlanes, "");
  tab->addColumn<float>("trkNumHits", trkNumHits, "");
  tab->addColumn<float>("trkNumDTHits", trkNumDTHits, "");
  tab->addColumn<float>("trkNumCSCHits", trkNumCSCHits, "");
  tab->addColumn<float>("normChi2", normChi2, "");
  tab->addColumn<float>("trkNumPixelHits", trkNumPixelHits, "");
  tab->addColumn<float>("trkNumTrkLayers", trkNumTrkLayers, "");

  tab->addColumn<float>("outerEta", outerEta, "");
  tab->addColumn<float>("outerPhi", outerPhi, "");

  tab->addColumn<float>("innerVx", innerVx, "");
  tab->addColumn<float>("innerVy", innerVy, "");
  tab->addColumn<float>("innerVz", innerVz, "");
  tab->addColumn<float>("innerPt", innerPt, "");
  tab->addColumn<float>("innerEta", innerEta, "");
  tab->addColumn<float>("innerPhi", innerPhi, "");

  tab->addColumn<float>("dsaMatch1", dsaMatch1, "");
  tab->addColumn<float>("dsaMatch1idx", dsaMatch1idx, "");
  tab->addColumn<float>("dsaMatch2", dsaMatch2, "");
  tab->addColumn<float>("dsaMatch2idx", dsaMatch2idx, "");
  tab->addColumn<float>("dsaMatch3", dsaMatch3, "");
  tab->addColumn<float>("dsaMatch3idx", dsaMatch3idx, "");
  tab->addColumn<float>("dsaMatch4", dsaMatch4, "");
  tab->addColumn<float>("dsaMatch4idx", dsaMatch4idx, "");
  tab->addColumn<float>("dsaMatch5", dsaMatch5, "");
  tab->addColumn<float>("dsaMatch5idx", dsaMatch5idx, "");

  iEvent.put(std::move(tab));
}

int MuonExtendedTableProducer::getMatches(const pat::Muon& muon, const reco::Track& dsaMuon, const float minPositionDiff=1e-6) const {

  int nMatches = 0;

  if( dsaMuon.extra().isNonnull() && dsaMuon.extra().isAvailable() ) {
    
    for (auto& hit : dsaMuon.recHits()){

      if (!hit->isValid()) continue;
      DetId id = hit->geographicalId();
      if (id.det() != DetId::Muon) continue;
      
      if (id.subdetId() == MuonSubdetId::DT || id.subdetId() == MuonSubdetId::CSC){
	
	for (auto& chamber : muon.matches()) {

	  if (chamber.id.rawId() != id.rawId()) continue;
	  
	  for (auto& segment : chamber.segmentMatches) {
	    
	    if (fabs(segment.x - hit->localPosition().x()) < minPositionDiff &&
		fabs(segment.y - hit->localPosition().y()) < minPositionDiff) {
              nMatches++;
              break;
	    }
	  }
	}
      }
    }
  }
  return nMatches;
}

float MuonExtendedTableProducer::getPFIso(const pat::Muon& muon) const {
  return (muon.pfIsolationR04().sumChargedHadronPt +
          std::max(0.,
                   muon.pfIsolationR04().sumNeutralHadronEt + muon.pfIsolationR04().sumPhotonEt -
                   0.5 * muon.pfIsolationR04().sumPUPt)) / muon.pt();
}

template< typename T1, typename T2 > bool isSourceCandidatePtrMatch( const T1& lhs, const T2& rhs ) {
  
  for( size_t lhsIndex = 0; lhsIndex < lhs.numberOfSourceCandidatePtrs(); ++lhsIndex ) {
    auto lhsSourcePtr = lhs.sourceCandidatePtr( lhsIndex );
    for( size_t rhsIndex = 0; rhsIndex < rhs.numberOfSourceCandidatePtrs(); ++rhsIndex ) {
      auto rhsSourcePtr = rhs.sourceCandidatePtr( rhsIndex );
      if( lhsSourcePtr == rhsSourcePtr ) {
	return true;
      }
    }
  }
  
  return false;
}

int MuonExtendedTableProducer::findMatchedJet(const reco::Candidate& lepton, const edm::Handle< std::vector< pat::Jet > >& jets) const {

  int iJet = -1;
  
  unsigned int nJets = jets->size();
  
  for(unsigned int i = 0; i < nJets; i++) {
    const pat::Jet & jet = (*jets)[i];
    if( isSourceCandidatePtrMatch( lepton, jet ) ) {
      return i;
    }
  }
  
  return iJet;
}

void MuonExtendedTableProducer::fillLeptonJetVariables(const reco::Muon *mu, edm::Handle< std::vector< pat::Jet > >& jets, const reco::Vertex& vertex, const double rho, std::vector<int> *jetIdx, std::vector<float> relIso0p4, std::vector<float> *jetPtRatio, std::vector<float> *jetPtRel, std::vector<int> *jetSelectedChargedMultiplicity) const {
   
  const reco::Candidate *cand = dynamic_cast<const reco::Candidate*>(mu);
  int matchedJetIdx = findMatchedJet( *cand, jets );
  
  jetIdx->push_back(matchedJetIdx);

  if( matchedJetIdx < 0 ) {
    float ptRatio = ( 1. / ( 1. + relIso0p4.back() ) );
    jetPtRatio->push_back(ptRatio);	 
    jetPtRel->push_back(0);
    jetSelectedChargedMultiplicity->push_back(0);
  } else {
    const pat::Jet& jet = (*jets)[matchedJetIdx];
    auto rawJetP4 = jet.correctedP4("Uncorrected");
    auto leptonP4 = cand->p4();
    
    bool leptonEqualsJet = ( ( rawJetP4 - leptonP4 ).P() < 1e-4 );
    
    if( leptonEqualsJet ) {
      jetPtRatio->push_back(1);
      jetPtRel->push_back(0);
      jetSelectedChargedMultiplicity->push_back(0);	    
    } else {
      auto L1JetP4 = jet.correctedP4("L1FastJet");
      double L2L3JEC = jet.pt()/L1JetP4.pt();
      auto lepAwareJetP4 = ( L1JetP4 - leptonP4 )*L2L3JEC + leptonP4;
      
      float ptRatio = cand->pt() / lepAwareJetP4.pt();
      float ptRel = leptonP4.Vect().Cross( (lepAwareJetP4 - leptonP4 ).Vect().Unit() ).R();
      jetPtRatio->push_back(ptRatio);
      jetPtRel->push_back(ptRel);
      jetSelectedChargedMultiplicity->push_back(0);
      
      for( const auto &daughterPtr : jet.daughterPtrVector() ) {
	const pat::PackedCandidate& daughter = *( (const pat::PackedCandidate*) daughterPtr.get() );
	
	if( daughter.charge() == 0 ) continue;
	if( daughter.fromPV() < 2 ) continue;
	if( reco::deltaR( daughter, *cand ) > 0.4 ) continue;
	if( !daughter.hasTrackDetails() ) continue;
	
	auto daughterTrack = daughter.pseudoTrack();
	    
	if( daughterTrack.pt() <= 1 ) continue;
	if( daughterTrack.hitPattern().numberOfValidHits() < 8 ) continue;
	if( daughterTrack.hitPattern().numberOfValidPixelHits() < 2 ) continue;
	if( daughterTrack.normalizedChi2() >= 5 ) continue;
	if( std::abs( daughterTrack.dz( vertex.position() ) ) >= 17 ) continue;
	if( std::abs( daughterTrack.dxy( vertex.position() ) ) >= 0.2 ) continue;
	++jetSelectedChargedMultiplicity->back();
      }
    }      
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(MuonExtendedTableProducer);
