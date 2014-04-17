// -*- C++ -*-
//
// Package:    JetMETCorrections/FFTJetModules
// Class:      OOTPileupCorrectionDBWriter
// 
/**\class OOTPileupCorrectionDBWriter OOTPileupCorrectionDBWriter.cc JetMETCorrections/FFTJetModules/plugins/OOTPileupCorrectionDBWriter.cc

 Description: writes a blob from a file into a database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed Aug  1 20:59:12 CDT 2012
//
//

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/HcalOOTPileupCorrectionData.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class OOTPileupCorrectionDBWriter : public edm::EDAnalyzer
{
public:
    explicit OOTPileupCorrectionDBWriter(const edm::ParameterSet&);
    virtual ~OOTPileupCorrectionDBWriter() {}

private:
    OOTPileupCorrectionDBWriter();
    OOTPileupCorrectionDBWriter(const OOTPileupCorrectionDBWriter&);
    OOTPileupCorrectionDBWriter& operator=(const OOTPileupCorrectionDBWriter&);

    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

    std::string inputFile;
    std::string record;
};

OOTPileupCorrectionDBWriter::OOTPileupCorrectionDBWriter(const edm::ParameterSet& ps)
    : init_param(std::string, inputFile),
      init_param(std::string, record)
{
}

void OOTPileupCorrectionDBWriter::analyze(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup)
{
    std::auto_ptr<HcalOOTPileupCorrectionData> fcp;

    {
        std::ifstream input(inputFile.c_str(), std::ios_base::binary);
        if (!input.is_open())
            throw cms::Exception("InvalidArgument")
                << "Failed to open file \"" << inputFile << '"' << std::endl;

        struct stat st;
        if (stat(inputFile.c_str(), &st))
            throw cms::Exception("SystemError")
                << "Failed to stat file \"" << inputFile << '"' << std::endl;

        const std::size_t len = st.st_size;
        fcp = std::auto_ptr<HcalOOTPileupCorrectionData>(
            new HcalOOTPileupCorrectionData(len));
        assert(fcp->length() == len);
        if (len)
            input.read(fcp->getBuffer(), len);
        if (input.fail())
            throw cms::Exception("SystemError")
                << "Input stream failure while reading file \""
                << inputFile << '"' << std::endl;
    }

    edm::Service<cond::service::PoolDBOutputService> poolDbService;
    if (poolDbService.isAvailable())
        poolDbService->writeOne(fcp.release(),
                                poolDbService->currentTime(),
                                record);
    else
        throw cms::Exception("ConfigurationError")
            << "PoolDBOutputService is not available, "
            << "please configure it properly" << std::endl;
}

DEFINE_FWK_MODULE(OOTPileupCorrectionDBWriter);
