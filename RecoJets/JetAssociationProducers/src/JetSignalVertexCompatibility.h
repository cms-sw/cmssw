#ifndef JetSignalVertexCompatibility_h
#define JetSignalVertexCompatibility_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetSignalVertexCompatibilityAlgo.h"

class JetSignalVertexCompatibility : public edm::EDProducer {
    public:
	JetSignalVertexCompatibility(const edm::ParameterSet &params);
	~JetSignalVertexCompatibility();

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	reco::JetSignalVertexCompatibilityAlgo	algo;

	const edm::InputTag			jetTracksAssocLabel;
	const edm::InputTag			primaryVerticesLabel;
};

#endif // JetSignalVertexCompatibility_h
