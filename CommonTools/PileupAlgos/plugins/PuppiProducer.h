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
	~PuppiProducer();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	typedef math::XYZTLorentzVector LorentzVector;
	typedef std::vector<LorentzVector> LorentzVectorCollection;
	typedef reco::VertexCollection VertexCollection;
	typedef edm::View<reco::Candidate> CandidateView;
	typedef std::vector< reco::PFCandidate >               PFInputCollection;
	typedef std::vector< reco::PFCandidate >  PFOutputCollection;
	typedef edm::View<reco::PFCandidate>                   PFView;

private:
	virtual void beginJob() ;
	virtual void produce(edm::Event&, const edm::EventSetup&);
	virtual void endJob() ;
      
	virtual void beginRun(edm::Run&, edm::EventSetup const&);
	virtual void endRun(edm::Run&, edm::EventSetup const&);
	virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
	virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

	edm::EDGetTokenT< CandidateView > tokenPFCandidates_;
	edm::EDGetTokenT< VertexCollection > tokenVertices_;
	std::string     fPuppiName;
	std::string     fPFName;	
	std::string     fPVName;
	bool 			fPuppiDiagnostics;
	bool 			fPuppiForLeptons;
	bool            fUseDZ;
	float           fDZCut;
	bool fUseExistingWeights;
	bool fUseWeightsNoLep;
	int fVtxNdofCut;
	double fVtxZCut;
	std::unique_ptr<PuppiContainer> fPuppiContainer;
	std::vector<RecoObj> fRecoObjCollection;
        std::auto_ptr< PFOutputCollection >          fPuppiCandidates;
};
#endif
