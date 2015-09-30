// -*- C++ -*-
//
// Author: Sébastien Brochet 
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"

//
// class declaration
//

class JetResolutionDBReader : public edm::EDAnalyzer {
    public:
        explicit JetResolutionDBReader(const edm::ParameterSet&);
        ~JetResolutionDBReader();


    private:
        virtual void beginJob() override ;
        virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
        virtual void endJob() override ;

        std::string m_era;
        std::string m_label;

        bool m_save_file;
        bool m_print;
};

class JetResolutionScaleFactorDBReader : public edm::EDAnalyzer {
    public:
        explicit JetResolutionScaleFactorDBReader(const edm::ParameterSet&);


    private:
        virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

        std::string m_era;
        std::string m_label;

        bool m_save_file;
        bool m_print;
};


JetResolutionDBReader::JetResolutionDBReader(const edm::ParameterSet& iConfig)
{
    m_era           = iConfig.getUntrackedParameter<std::string>("era");
    m_label         = iConfig.getUntrackedParameter<std::string>("label");
    m_print         = iConfig.getUntrackedParameter<bool>("dump", true);
    m_save_file     = iConfig.getUntrackedParameter<bool>("saveFile", false);
}


JetResolutionDBReader::~JetResolutionDBReader()
{

}

void JetResolutionDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::ESHandle<JME::JetResolutionObject> jerObjectHandle;
    std::cout << "Inspecting JER payload for label: "<< m_label << std::endl;

    iSetup.get<JetResolutionRcd>().get(m_label, jerObjectHandle);

    if (m_print) {
        jerObjectHandle->dump();
    }

    if (m_save_file) {
        std::string f = m_era + "_" + m_label + ".txt";
        jerObjectHandle->saveToFile(f);
        std::cout << "JER payload saved as " << f << std::endl;
    }
}

void JetResolutionDBReader::beginJob()
{
}

void JetResolutionDBReader::endJob() 
{
}

JetResolutionScaleFactorDBReader::JetResolutionScaleFactorDBReader(const edm::ParameterSet& iConfig)
{
    m_era           = iConfig.getUntrackedParameter<std::string>("era");
    m_label         = iConfig.getUntrackedParameter<std::string>("label");
    m_print         = iConfig.getUntrackedParameter<bool>("dump", true);
    m_save_file     = iConfig.getUntrackedParameter<bool>("saveFile", false);
}

void JetResolutionScaleFactorDBReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::ESHandle<JME::JetResolutionObject> jerObjectHandle;
    std::cout << "Inspecting JER SF payload for label: "<< m_label << std::endl;

    iSetup.get<JetResolutionScaleFactorRcd>().get(m_label, jerObjectHandle);

    if (m_print) {
        jerObjectHandle->dump();
    }

    if (m_save_file) {
        std::string f = m_era + "_" + m_label + ".txt";
        jerObjectHandle->saveToFile(f);
        std::cout << "JER SF payload saved as " << f << std::endl;
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetResolutionDBReader);
DEFINE_FWK_MODULE(JetResolutionScaleFactorDBReader);
