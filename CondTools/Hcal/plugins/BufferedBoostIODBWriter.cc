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

#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionBuffer.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

//
// class declaration
//
class BufferedBoostIODBWriter : public edm::EDAnalyzer
{
public:
    explicit BufferedBoostIODBWriter(const edm::ParameterSet&);
    ~BufferedBoostIODBWriter() override {}

private:
    BufferedBoostIODBWriter() = delete;
    BufferedBoostIODBWriter(const BufferedBoostIODBWriter&) = delete;
    BufferedBoostIODBWriter& operator=(const BufferedBoostIODBWriter&) = delete;

    void analyze(const edm::Event&, const edm::EventSetup&) override;

    std::string inputFile;
    std::string record;
};

BufferedBoostIODBWriter::BufferedBoostIODBWriter(const edm::ParameterSet& ps)
    : inputFile(ps.getParameter<std::string>("inputFile")),
      record(ps.getParameter<std::string>("record"))
{
}

void BufferedBoostIODBWriter::analyze(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup)
{
    std::unique_ptr<OOTPileupCorrectionBuffer> fcp;

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
        fcp = std::unique_ptr<OOTPileupCorrectionBuffer>(
            new OOTPileupCorrectionBuffer(len));
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

DEFINE_FWK_MODULE(BufferedBoostIODBWriter);
