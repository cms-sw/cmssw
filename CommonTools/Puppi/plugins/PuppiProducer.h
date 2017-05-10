#ifndef CommonTools_Puppi_PuppiProducer_h_
#define CommonTools_Puppi_PuppiProducer_h_
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"

#include "CommonTools/Puppi/interface/PuppiContainer.h"

// ------------------------------------------------------------------------------------------
class PuppiProducer : public edm::EDProducer {

public:
	explicit PuppiProducer(const edm::ParameterSet&);
	~PuppiProducer();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
	typedef math::XYZTLorentzVector LorentzVector;

private:
	virtual void beginJob() ;
	virtual void produce(edm::Event&, const edm::EventSetup&);
	virtual void endJob() ;
      
	virtual void beginRun(edm::Run&, edm::EventSetup const&);
	virtual void endRun(edm::Run&, edm::EventSetup const&);
	virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
	virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
	reco::PFCandidate::ParticleType translatePdgIdToType(int pdgid) const;
        
	  edm::InputTag fPFCands;
        edm::InputTag fVertices;
	std::string     fPuppiName;
	std::string     fPFName;	
	std::string     fPVName;
	bool            fUseDZ;
        float           fDZCut;
        PuppiContainer *fPuppiContainer;
	std::vector<RecoObj> fRecoObjCollection;
        std::auto_ptr< reco::PFCandidateCollection >          fPuppiCandidates;
};
#endif
