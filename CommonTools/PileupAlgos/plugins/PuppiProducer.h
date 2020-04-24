#ifndef CommonTools_Puppi_PuppiProducer_h_
#define CommonTools_Puppi_PuppiProducer_h_
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"

#include "CommonTools/PileupAlgos/interface/PuppiContainer.h"

// ------------------------------------------------------------------------------------------
class PuppiProducer : public edm::stream::EDProducer<> {

public:
	explicit PuppiProducer(const edm::ParameterSet&);
	~PuppiProducer() override;

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	typedef math::XYZTLorentzVector LorentzVector;
	typedef std::vector<LorentzVector> LorentzVectorCollection;
	typedef reco::VertexCollection VertexCollection;
	typedef edm::View<reco::Candidate> CandidateView;
	typedef std::vector< reco::PFCandidate >               PFInputCollection;
	typedef std::vector< reco::PFCandidate >  PFOutputCollection;
	typedef std::vector< pat::PackedCandidate >            PackedOutputCollection;
	typedef edm::View<reco::PFCandidate>                   PFView;

private:
	virtual void beginJob() ;
	void produce(edm::Event&, const edm::EventSetup&) override;
	virtual void endJob() ;
      
	edm::EDGetTokenT< CandidateView > tokenPFCandidates_;
	edm::EDGetTokenT< VertexCollection > tokenVertices_;
	std::string     fPuppiName;
	std::string     fPFName;	
	std::string     fPVName;
	bool 			fPuppiDiagnostics;
	bool 			fPuppiForLeptons;
	bool            fUseDZ;
	float           fDZCut;
	float           fPtMax;
	bool fUseExistingWeights;
	bool fUseWeightsNoLep;
	bool fClonePackedCands;
	int fVtxNdofCut;
	double fVtxZCut;
	std::unique_ptr<PuppiContainer> fPuppiContainer;
	std::vector<RecoObj> fRecoObjCollection;
        std::unique_ptr< PFOutputCollection >          fPuppiCandidates;
	std::unique_ptr< PackedOutputCollection >      fPackedPuppiCandidates;
};
#endif
