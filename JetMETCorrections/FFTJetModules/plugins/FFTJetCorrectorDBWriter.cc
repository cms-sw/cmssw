// -*- C++ -*-
//
// Package:    JetMETCorrections/FFTJetModules
// Class:      FFTJetCorrectorDBWriter
// 
/**\class FFTJetCorrectorDBWriter FFTJetCorrectorDBWriter.cc JetMETCorrections/FFTJetModules/plugins/FFTJetCorrectorDBWriter.cc

 Description: writes a blob from a file into a database

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Wed Aug  1 20:59:12 CDT 2012
// $Id: FFTJetCorrectorDBWriter.cc,v 1.1 2012/11/14 22:34:57 igv Exp $
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

#include "CondFormats/JetMETObjects/interface/FFTJetCorrectorParameters.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#define init_param(type, varname) varname (ps.getParameter< type >( #varname ))

//
// class declaration
//
class FFTJetCorrectorDBWriter : public edm::EDAnalyzer
{
public:
    explicit FFTJetCorrectorDBWriter(const edm::ParameterSet&);
    virtual ~FFTJetCorrectorDBWriter() {}

private:
    FFTJetCorrectorDBWriter();
    FFTJetCorrectorDBWriter(const FFTJetCorrectorDBWriter&);
    FFTJetCorrectorDBWriter& operator=(const FFTJetCorrectorDBWriter&);

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

    std::string inputFile;
    std::string record;
};

FFTJetCorrectorDBWriter::FFTJetCorrectorDBWriter(const edm::ParameterSet& ps)
    : init_param(std::string, inputFile),
      init_param(std::string, record)
{
}

void FFTJetCorrectorDBWriter::analyze(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup)
{
    std::auto_ptr<FFTJetCorrectorParameters> fcp;

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
        fcp = std::auto_ptr<FFTJetCorrectorParameters>(
            new FFTJetCorrectorParameters(len));
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

DEFINE_FWK_MODULE(FFTJetCorrectorDBWriter);
