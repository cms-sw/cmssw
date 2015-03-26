#include "RecoMET/METPUSubtraction/plugins/PFMETProducerMVA.h"

using namespace reco;

const double dR2Min = 0.01*0.01;
const double dR2Max = 0.5*0.5;
const double dPtMatch = 0.1;

PFMETProducerMVA::PFMETProducerMVA(const edm::ParameterSet& cfg) 
  : mvaMEtAlgo_(cfg),
    mvaMEtAlgo_isInitialized_(false)
{
  srcCorrJets_     = consumes<reco::PFJetCollection>(cfg.getParameter<edm::InputTag>("srcCorrJets"));
  srcUncorrJets_   = consumes<reco::PFJetCollection>(cfg.getParameter<edm::InputTag>("srcUncorrJets"));
  srcJetIds_       = consumes<edm::ValueMap<float> >(cfg.getParameter<edm::InputTag>("srcMVAPileupJetId"));
  srcPFCandidatesView_ = consumes<reco::CandidateView>(cfg.getParameter<edm::InputTag>("srcPFCandidates"));
  srcVertices_     = consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("srcVertices"));
  vInputTag srcLeptonsTags = cfg.getParameter<vInputTag>("srcLeptons");
  for(vInputTag::const_iterator it=srcLeptonsTags.begin();it!=srcLeptonsTags.end();it++) {
    srcLeptons_.push_back( consumes<reco::CandidateView >( *it ) );
  }

  minNumLeptons_   = cfg.getParameter<int>("minNumLeptons");
  srcRho_          = consumes<edm::Handle<double> >(cfg.getParameter<edm::InputTag>("srcRho"));

  globalThreshold_ = cfg.getParameter<double>("globalThreshold");

  minCorrJetPt_    = cfg.getParameter<double>     ("minCorrJetPt");
  useType1_        = cfg.getParameter<bool>       ("useType1");
  correctorLabel_  = cfg.getParameter<std::string>("corrector");
   
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<reco::PFMETCollection>();
}

PFMETProducerMVA::~PFMETProducerMVA(){}

void PFMETProducerMVA::produce(edm::Event& evt, const edm::EventSetup& es) 
{ 
  // CV: check if the event is to be skipped
  if ( minNumLeptons_ > 0 ) {
    int numLeptons = 0;
    for ( std::vector<edm::EDGetTokenT<reco::CandidateView> >::const_iterator srcLeptons_i = srcLeptons_.begin();
	  srcLeptons_i != srcLeptons_.end(); ++srcLeptons_i ) {
      edm::Handle<reco::CandidateView> leptons;
      evt.getByToken(*srcLeptons_i, leptons);
      numLeptons += leptons->size();
    }
    if ( !(numLeptons >= minNumLeptons_) ) {
      LogDebug( "produce" )
	<< "<PFMETProducerMVA::produce>:" << std::endl
	<< "Run: " << evt.id().run() << ", LS: " << evt.luminosityBlock()  << ", Event: " << evt.id().event() << std::endl
	<< " numLeptons = " << numLeptons << ", minNumLeptons = " << minNumLeptons_ << " --> skipping !!" << std::endl;
      
      reco::PFMET pfMEt;
      std::auto_ptr<reco::PFMETCollection> pfMEtCollection(new reco::PFMETCollection());
      pfMEtCollection->push_back(pfMEt);
      evt.put(pfMEtCollection);
      return;
    }
  }

  //get jet IDs
  edm::Handle<edm::ValueMap<float> > jetIds;
  evt.getByToken(srcJetIds_, jetIds);

  // get jets (corrected and uncorrected)
  edm::Handle<reco::PFJetCollection> corrJets;
  evt.getByToken(srcCorrJets_, corrJets);

  edm::Handle<reco::PFJetCollection> uncorrJets;
  evt.getByToken(srcUncorrJets_, uncorrJets);

  const JetCorrector* corrector = nullptr;
  if( useType1_ ) corrector = JetCorrector::getJetCorrector(correctorLabel_, es);
  
  edm::Handle<reco::CandidateView> pfCandidates_view;
  evt.getByToken(srcPFCandidatesView_, pfCandidates_view);

  // get vertices
  edm::Handle<reco::VertexCollection> vertices;
  evt.getByToken(srcVertices_, vertices); 
  // take vertex with highest sum(trackPt) as the vertex of the "hard scatter" interaction
  // (= first entry in vertex collection)
  const reco::Vertex* hardScatterVertex = ( vertices->size() >= 1 ) ?
    &(vertices->front()) : nullptr;
  
  // get leptons
  // (excluded from sum over PFCandidates when computing hadronic recoil)
  int  lId         = 0;
  bool lHasPhotons = false;
  std::vector<reco::PUSubMETCandInfo> leptonInfo = computeLeptonInfo(srcLeptons_,*pfCandidates_view,hardScatterVertex, lId, lHasPhotons,
								     evt);

  // initialize MVA MET algorithm
  // (this will load the BDTs, stored as GBRForrest objects;
  //  either in input ROOT files or in SQL-lite files/the Conditions Database) 
  if ( !mvaMEtAlgo_isInitialized_ ) {
    mvaMEtAlgo_.initialize(es);
    mvaMEtAlgo_isInitialized_ = true;
  }

  // reconstruct "standard" particle-flow missing Et
  CommonMETData pfMEt_data = metAlgo_.run( (*pfCandidates_view), globalThreshold_);
  SpecificPFMETData specificPfMET = pfMEtSpecificAlgo_.run( (*pfCandidates_view) );
  const reco::Candidate::LorentzVector p4( pfMEt_data.mex, pfMEt_data.mey, 0.0, pfMEt_data.met);
  const reco::Candidate::Point vtx(0.0, 0.0, 0.0 );
  reco::PFMET pfMEt(specificPfMET,pfMEt_data.sumet, p4, vtx);
  reco::Candidate::LorentzVector pfMEtP4_original = pfMEt.p4();
   
  // compute objects specific to MVA based MET reconstruction
  std::vector<reco::PUSubMETCandInfo> pfCandidateInfo = computePFCandidateInfo(*pfCandidates_view, hardScatterVertex);
  std::vector<reco::PUSubMETCandInfo>    jetInfo         = computeJetInfo(*uncorrJets, corrJets, *jetIds, *vertices, hardScatterVertex, *corrector,evt,es,leptonInfo,pfCandidateInfo);
  std::vector<reco::Vertex::Point>         vertexInfo      = computeVertexInfo(*vertices);
  // compute MVA based MET and estimate of its uncertainty
  mvaMEtAlgo_.setInput(leptonInfo, jetInfo, pfCandidateInfo, vertexInfo);
  mvaMEtAlgo_.setHasPhotons(lHasPhotons);
  mvaMEtAlgo_.evaluateMVA();
  pfMEt.setP4(mvaMEtAlgo_.getMEt());
  pfMEt.setSignificanceMatrix(mvaMEtAlgo_.getMEtCov());
  
  LogDebug("produce")
    << "Run: " << evt.id().run() << ", LS: " << evt.luminosityBlock()  << ", Event: " << evt.id().event() << std::endl
    << " PFMET: Pt = " << pfMEtP4_original.pt() << ", phi = " << pfMEtP4_original.phi() << " "
    << "(Px = " << pfMEtP4_original.px() << ", Py = " << pfMEtP4_original.py() << ")" << std::endl
    << " MVA MET: Pt = " << pfMEt.pt() << " phi = " << pfMEt.phi() << " (Px = " << pfMEt.px() << ", Py = " << pfMEt.py() << ")" << std::endl
    << " Cov:" << std::endl
    <<(mvaMEtAlgo_.getMEtCov())(0,0)<<"  "<<(mvaMEtAlgo_.getMEtCov())(0,1)<<std::endl
    <<(mvaMEtAlgo_.getMEtCov())(1,0)<<"  "<<(mvaMEtAlgo_.getMEtCov())(1,1)<<std::endl  << std::endl;
 
  // add PFMET object to the event
  std::auto_ptr<reco::PFMETCollection> pfMEtCollection(new reco::PFMETCollection());
  pfMEtCollection->push_back(pfMEt);
  evt.put(pfMEtCollection);
}

std::vector<reco::PUSubMETCandInfo>
PFMETProducerMVA::computeLeptonInfo(const std::vector<edm::EDGetTokenT<reco::CandidateView > >& srcLeptons_,
				    const reco::CandidateView& pfCandidates_view,
				    const reco::Vertex* hardScatterVertex,
				    int& lId, bool& lHasPhotons, edm::Event& evt ) {

  std::vector<reco::PUSubMETCandInfo> leptonInfo;

  for ( std::vector<edm::EDGetTokenT<reco::CandidateView > >::const_iterator srcLeptons_i = srcLeptons_.begin();
	srcLeptons_i != srcLeptons_.end(); ++srcLeptons_i ) {
    edm::Handle<reco::CandidateView> leptons;
    evt.getByToken(*srcLeptons_i, leptons);
    for ( reco::CandidateView::const_iterator lepton1 = leptons->begin();
	  lepton1 != leptons->end(); ++lepton1 ) {
      bool pMatch = false;
      for ( std::vector<edm::EDGetTokenT<reco::CandidateView> >::const_iterator srcLeptons_j = srcLeptons_.begin();
	    srcLeptons_j != srcLeptons_.end(); ++srcLeptons_j ) {
	edm::Handle<reco::CandidateView> leptons2;
	evt.getByToken(*srcLeptons_j, leptons2);
	for ( reco::CandidateView::const_iterator lepton2 = leptons2->begin();
	      lepton2 != leptons2->end(); ++lepton2 ) {
	  if(&(*lepton1) == &(*lepton2)) { continue; }
	  if(deltaR2(lepton1->p4(),lepton2->p4()) < dR2Max) { pMatch = true; }
	  if(pMatch &&     !istau(&(*lepton1)) &&  istau(&(*lepton2))) { pMatch = false; }
	  if(pMatch &&    ( (istau(&(*lepton1)) && istau(&(*lepton2))) || (!istau(&(*lepton1)) && !istau(&(*lepton2)))) 
	     &&     lepton1->pt() > lepton2->pt()) { pMatch = false; }
	  if(pMatch && lepton1->pt() == lepton2->pt()) {
	    pMatch = false;
	    for(unsigned int i0 = 0; i0 < leptonInfo.size(); i0++) {
	      if(std::abs(lepton1->pt() - leptonInfo[i0].p4().pt()) < dPtMatch) { pMatch = true; break; }
	    }
	  }
	  if(pMatch) break;
	}
	if(pMatch) break;
      }
      if(pMatch) continue;
      reco::PUSubMETCandInfo pLeptonInfo;
      pLeptonInfo.setP4( lepton1->p4() );
      pLeptonInfo.setChargedEnFrac( chargedEnFrac(&(*lepton1),pfCandidates_view,hardScatterVertex) );
      leptonInfo.push_back(pLeptonInfo); 
      if(lepton1->isPhoton()) { lHasPhotons = true; }
    }
    lId++;
  }
 
  return leptonInfo;
}


std::vector<reco::PUSubMETCandInfo> 
PFMETProducerMVA::computeJetInfo(const reco::PFJetCollection& uncorrJets, 
				 const edm::Handle<reco::PFJetCollection>& corrJets, 
				 const edm::ValueMap<float>& jetIds,
				 const reco::VertexCollection& vertices,
				 const reco::Vertex* hardScatterVertex,
				 const JetCorrector &iCorrector,edm::Event &iEvent,const edm::EventSetup &iSetup,
				 std::vector<reco::PUSubMETCandInfo> &iLeptons,std::vector<reco::PUSubMETCandInfo> &iCands)
{
  const L1FastjetCorrector* lCorrector = dynamic_cast<const L1FastjetCorrector*>(&iCorrector);
  std::vector<reco::PUSubMETCandInfo> retVal;
  for ( reco::PFJetCollection::const_iterator uncorrJet = uncorrJets.begin();
	uncorrJet != uncorrJets.end(); ++uncorrJet ) {
    // for ( reco::PFJetCollection::const_iterator corrJet = corrJets.begin();
    // 	  corrJet != corrJets.end(); ++corrJet ) {
    auto corrJet = corrJets->begin();
    for( size_t cjIdx=0;cjIdx<corrJets->size();++cjIdx, ++corrJet) {
      reco::PFJetRef corrJetRef( corrJets, cjIdx );

      // match corrected and uncorrected jets
      if ( uncorrJet->jetArea() != corrJet->jetArea() ) continue;
      if ( deltaR2(corrJet->p4(),uncorrJet->p4()) > dR2Min ) continue;

      // check that jet passes loose PFJet id.
      if(!passPFLooseId(&(*uncorrJet))) continue;

      // compute jet energy correction factor
      // (= ratio of corrected/uncorrected jet Pt)
      //double jetEnCorrFactor = corrJet->pt()/uncorrJet->pt();
      reco::PUSubMETCandInfo jetInfo;
      
      // PH: apply jet energy corrections for all Jets ignoring recommendations
      jetInfo.setP4( corrJet->p4() );
      double lType1Corr = 0;
      if(useType1_) { //Compute the type 1 correction ===> This code is crap 
	double pCorr = lCorrector->correction(*uncorrJet,iEvent,iSetup);
	lType1Corr = std::abs(corrJet->pt()-pCorr*uncorrJet->pt());
	TLorentzVector pVec; pVec.SetPtEtaPhiM(lType1Corr,0,corrJet->phi(),0); 
	reco::Candidate::LorentzVector pType1Corr; pType1Corr.SetCoordinates(pVec.Px(),pVec.Py(),pVec.Pz(),pVec.E());
	//Filter to leptons
	bool pOnLepton = false;
	for(unsigned int i0 = 0; i0 < iLeptons.size(); i0++) {
	  if(deltaR2(iLeptons[i0].p4(),corrJet->p4()) < dR2Max) {pOnLepton = true; break;}
	}	
 	//Add it to PF Collection
	if(corrJet->pt() > 10 && !pOnLepton) {
	  reco::PUSubMETCandInfo pfCandidateInfo;
	  pfCandidateInfo.setP4( pType1Corr );
	  pfCandidateInfo.setDZ( -999 );
	  iCands.push_back(pfCandidateInfo);
	}
	//Scale
	lType1Corr = (pCorr*uncorrJet->pt()-uncorrJet->pt());
	lType1Corr /=corrJet->pt();
      }
      
      // check that jet Pt used to compute MVA based jet id. is above threshold
      if ( !(jetInfo.p4().pt() > minCorrJetPt_) ) continue;

    
      jetInfo.setMvaVal( jetIds[ corrJetRef ] );
      float chEnF = (uncorrJet->chargedEmEnergy() + uncorrJet->chargedHadronEnergy() + uncorrJet->chargedMuEnergy() )/uncorrJet->energy();
      if(useType1_) chEnF += lType1Corr*(1-jetInfo.chargedEnFrac() );
      jetInfo.setChargedEnFrac( chEnF ); 
      retVal.push_back(jetInfo);
      break;
    }
  }

  //order jets per pt
  std::sort( retVal.begin(), retVal.end() );

  return retVal;
}

std::vector<reco::PUSubMETCandInfo> PFMETProducerMVA::computePFCandidateInfo(const reco::CandidateView& pfCandidates,
										  const reco::Vertex* hardScatterVertex)
{
  std::vector<reco::PUSubMETCandInfo> retVal;
  for ( reco::CandidateView::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    double dZ = -999.; // PH: If no vertex is reconstructed in the event
                       //     or PFCandidate has no track, set dZ to -999
    if ( hardScatterVertex ) {
      const reco::PFCandidate* pfc = dynamic_cast<const reco::PFCandidate* >( &(*pfCandidate) );
      if( pfc != nullptr ) { //PF candidate for RECO and PAT levels
	if      ( pfc->trackRef().isNonnull()    ) dZ = std::abs(pfc->trackRef()->dz(hardScatterVertex->position()));
	else if ( pfc->gsfTrackRef().isNonnull() ) dZ = std::abs(pfc->gsfTrackRef()->dz(hardScatterVertex->position()));
      }
      else { //if not, then packedCandidate for miniAOD level
	const pat::PackedCandidate* pfc = dynamic_cast<const pat::PackedCandidate* >( &(*pfCandidate) );
	dZ = std::abs( pfc->dz( hardScatterVertex->position() ) );
	//exact dz=zero corresponds to the -999 case for pfcandidate
	if(dZ==0) {dZ=-999;}
      }
    }
    reco::PUSubMETCandInfo pfCandidateInfo;
    pfCandidateInfo.setP4( pfCandidate->p4() );
    pfCandidateInfo.setDZ( dZ );
    retVal.push_back(pfCandidateInfo);
  }
  return retVal;
}

std::vector<reco::Vertex::Point> PFMETProducerMVA::computeVertexInfo(const reco::VertexCollection& vertices)
{
  std::vector<reco::Vertex::Point> retVal;
  for ( reco::VertexCollection::const_iterator vertex = vertices.begin();
	vertex != vertices.end(); ++vertex ) {
    if(std::abs(vertex->z())           > 24.) continue;
    if(vertex->ndof()              <  4.) continue;
    if(vertex->position().Rho()    >  2.) continue;
    retVal.push_back(vertex->position());
  }
  return retVal;
}
double PFMETProducerMVA::chargedEnFrac(const reco::Candidate *iCand,
				     const reco::CandidateView& pfCandidates,const reco::Vertex* hardScatterVertex) { 
  if(iCand->isMuon())     {
    return 1;
  }
  if(iCand->isElectron())   {
    return 1.;
  }
  if(iCand->isPhoton()  )   {return chargedFracInCone(iCand, pfCandidates,hardScatterVertex);}
  double lPtTot = 0; double lPtCharged = 0;
  const reco::PFTau *lPFTau = 0; 
  lPFTau = dynamic_cast<const reco::PFTau*>(iCand);
  if(lPFTau != nullptr) { 
    for (UInt_t i0 = 0; i0 < lPFTau->signalPFCands().size(); i0++) { 
      lPtTot += (lPFTau->signalPFCands())[i0]->pt(); 
      if((lPFTau->signalPFCands())[i0]->charge() == 0) continue;
      lPtCharged += (lPFTau->signalPFCands())[i0]->pt(); 
    }
  } 
  else { 
    const pat::Tau *lPatPFTau = nullptr; 
    lPatPFTau = dynamic_cast<const pat::Tau*>(iCand);
    if(lPatPFTau != nullptr) { 
      for (UInt_t i0 = 0; i0 < lPatPFTau->signalCands().size(); i0++) { 
	lPtTot += (lPatPFTau->signalCands())[i0]->pt(); 
	if((lPatPFTau->signalCands())[i0]->charge() == 0) continue;
	lPtCharged += (lPatPFTau->signalCands())[i0]->pt(); 
      }
    }
  }
  if(lPtTot == 0) lPtTot = 1.;
  return lPtCharged/lPtTot;
}
//Return tau id by process of elimination
bool PFMETProducerMVA::istau(const reco::Candidate *iCand) { 
  if(iCand->isMuon())     return false;
  if(iCand->isElectron()) return false;
  if(iCand->isPhoton())   return false;
  return true;
}
bool PFMETProducerMVA::passPFLooseId(const PFJet *iJet) { 
  if(iJet->energy()== 0)                                  return false;
  if(iJet->neutralHadronEnergy()/iJet->energy() > 0.99)   return false;
  if(iJet->neutralEmEnergy()/iJet->energy()     > 0.99)   return false;
  if(iJet->nConstituents() <  2)                          return false;
  if(iJet->chargedHadronEnergy()/iJet->energy() <= 0 && std::abs(iJet->eta()) < 2.4 ) return false;
  if(iJet->chargedEmEnergy()/iJet->energy() >  0.99  && std::abs(iJet->eta()) < 2.4 ) return false;
  if(iJet->chargedMultiplicity()            < 1      && std::abs(iJet->eta()) < 2.4 ) return false;
  return true;
}

double PFMETProducerMVA::chargedFracInCone(const reco::Candidate *iCand,
					   const reco::CandidateView& pfCandidates,
					   const reco::Vertex* hardScatterVertex,double iDRMax)
{
  double iDR2Max = iDRMax*iDRMax;
  reco::Candidate::LorentzVector lVis(0,0,0,0);
  for ( reco::CandidateView::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    if(deltaR2(iCand->p4(),pfCandidate->p4()) > iDR2Max)  continue;
    double dZ = -999.; // PH: If no vertex is reconstructed in the event
                       //     or PFCandidate has no track, set dZ to -999
    if ( hardScatterVertex ) {
      const reco::PFCandidate* pfc = dynamic_cast<const reco::PFCandidate* >( (&(*pfCandidate)) );
      if( pfc != nullptr ) { //PF candidate for RECO and PAT levels
	if      ( pfc->trackRef().isNonnull()    ) dZ = std::abs(pfc->trackRef()->dz(hardScatterVertex->position()));
	else if ( pfc->gsfTrackRef().isNonnull() ) dZ = std::abs(pfc->gsfTrackRef()->dz(hardScatterVertex->position()));
      }
      else { //if not, then packedCandidate for miniAOD level
	const pat::PackedCandidate* pfc = dynamic_cast<const pat::PackedCandidate* >( &(*pfCandidate) );
	dZ = std::abs( pfc->dz( hardScatterVertex->position() ) );
      }
    }
    if(std::abs(dZ) > 0.1) continue; 
    lVis += pfCandidate->p4();
  }
  return lVis.pt()/iCand->pt();
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFMETProducerMVA);
