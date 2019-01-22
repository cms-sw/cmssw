#include "CommonTools/MVAUtils/plugins/GBRForestWriter.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

#include <TFile.h>

GBRForestWriter::GBRForestWriter(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
    edm::VParameterSet cfgJobs = cfg.getParameter<edm::VParameterSet>("jobs");
    for (edm::VParameterSet::const_iterator cfgJob = cfgJobs.begin(); cfgJob != cfgJobs.end(); ++cfgJob) {
        jobEntryType* job = new jobEntryType(*cfgJob);
        jobs_.push_back(job);
    }
}

GBRForestWriter::~GBRForestWriter()
{
    for (std::vector<jobEntryType*>::iterator it = jobs_.begin(); it != jobs_.end(); ++it) {
        delete (*it);
    }
}

void GBRForestWriter::analyze(const edm::Event&, const edm::EventSetup&)
{

    for (std::vector<jobEntryType*>::iterator job = jobs_.begin(); job != jobs_.end(); ++job) {
        std::map<std::string, const GBRForest*> gbrForests; // key = name
        for (std::vector<categoryEntryType*>::iterator category = (*job)->categories_.begin();
             category != (*job)->categories_.end(); ++category) {
            const GBRForest* gbrForest = nullptr;
            if ((*category)->inputFileType_ == categoryEntryType::kXML) {
                gbrForest = createGBRForest((*category)->inputFileName_).release();
            } else if ((*category)->inputFileType_ == categoryEntryType::kGBRForest) {
                TFile* inputFile = new TFile((*category)->inputFileName_.data());
                // gbrForest = dynamic_cast<GBRForest*>(inputFile->Get((*category)->gbrForestName_.data())); // CV:
                // dynamic_cast<GBRForest*> fails for some reason ?!
                gbrForest = (GBRForest*)inputFile->Get((*category)->gbrForestName_.data());
                delete inputFile;
            }
            if (!gbrForest)
                throw cms::Exception("GBRForestWriter")
                    << " Failed to load GBRForest = " << (*category)->gbrForestName_.data()
                    << " from file = " << (*category)->inputFileName_ << " !!\n";
            gbrForests[(*category)->gbrForestName_] = gbrForest;
        }
        if ((*job)->outputFileType_ == jobEntryType::kGBRForest) {
            TFile* outputFile = new TFile((*job)->outputFileName_.data(), "RECREATE");

            for (std::map<std::string, const GBRForest*>::iterator gbrForest = gbrForests.begin();
                 gbrForest != gbrForests.end(); ++gbrForest) {
                outputFile->WriteObject(gbrForest->second, gbrForest->first.data());
            }
            delete outputFile;
        } else if ((*job)->outputFileType_ == jobEntryType::kSQLLite) {
            edm::Service<cond::service::PoolDBOutputService> dbService;
            if (!dbService.isAvailable())
                throw cms::Exception("GBRForestWriter") << " Failed to access PoolDBOutputService !!\n";

            for (std::map<std::string, const GBRForest*>::iterator gbrForest = gbrForests.begin();
                 gbrForest != gbrForests.end(); ++gbrForest) {
                std::string outputRecord = (*job)->outputRecord_;
                if (gbrForests.size() > 1)
                    outputRecord.append("_").append(gbrForest->first);
                dbService->writeOne(gbrForest->second, dbService->beginOfTime(), outputRecord);
            }
        }

        // gbrforest deletion
        for (std::map<std::string, const GBRForest*>::iterator gbrForest = gbrForests.begin();
             gbrForest != gbrForests.end(); ++gbrForest) {
            delete gbrForest->second;
        }
    }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GBRForestWriter);
