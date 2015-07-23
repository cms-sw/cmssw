#ifndef RecoJets_JetProducers_plugins_PileupJetIDProducer_h
#define RecoJets_JetProducers_plugins_PileupJetIDProducer_h

// -*- C++ -*-
//
// Package:    PileupJetIdProducer
// Class:      PileupJetIdProducer
// 
/**\class PileupJetIdProducer PileupJetIdProducer.cc CMGTools/PileupJetIdProducer/src/PileupJetIdProducer.cc

Description: Produces a value map of jet --> pileup jet ID

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Pasquale Musella,40 2-A12,+41227671706,
//         Created:  Wed Apr 18 15:48:47 CEST 2012
//
//


// system include files
//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoJets/JetProducers/interface/PileupJetIdAlgo.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

// ------------------------------------------------------------------------------------------
class PileupJetIdProducer : public edm::stream::EDProducer<> {
public:
	explicit PileupJetIdProducer(const edm::ParameterSet&);
	~PileupJetIdProducer();

	static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
	virtual void produce(edm::Event&, const edm::EventSetup&) override;
      

	void initJetEnergyCorrector(const edm::EventSetup &iSetup, bool isData);

	edm::InputTag jets_, vertexes_, jetids_, rho_;
	std::string jec_;
	bool runMvas_, produceJetIds_, inputIsCorrected_, applyJec_;
	std::vector<std::pair<std::string, PileupJetIdAlgo *> > algos_;
	
	bool residualsFromTxt_;
	edm::FileInPath residualsTxt_;
	FactorizedJetCorrector *jecCor_;
	std::vector<JetCorrectorParameters> jetCorPars_;

        edm::EDGetTokenT<edm::View<reco::Jet> > input_jet_token_;
        edm::EDGetTokenT<reco::VertexCollection> input_vertex_token_;
        edm::EDGetTokenT<edm::ValueMap<StoredPileupJetIdentifier> > input_vm_pujetid_token_;
        edm::EDGetTokenT<double> input_rho_token_;

};

#endif
