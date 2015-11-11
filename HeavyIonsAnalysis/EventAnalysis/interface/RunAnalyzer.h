#pragma once

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

#include "TTree.h"

class RunAnalyzer: public edm::EDAnalyzer {
    public:
        explicit RunAnalyzer(const edm::ParameterSet& iConfig);
        virtual ~RunAnalyzer();

        virtual void beginJob() override;
        virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
        virtual void analyze(edm::Event const&, edm::EventSetup const&) override {};

    private:
        edm::EDGetTokenT<GenRunInfoProduct> genRunInfoToken_;

    private:
        // tree
        edm::Service<TFileService> fs_;
        TTree * fTree;
        float fXsec;
};
