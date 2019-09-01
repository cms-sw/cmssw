/** \class GBRForestWriter
 *
 * Read GBRForest objects from ROOT file input
 * and store it in SQL-lite output file
 *
 * \authors Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <TFile.h>
#include <string>
#include <vector>

class GBRForestWriter : public edm::EDAnalyzer {
public:
  GBRForestWriter(const edm::ParameterSet&);
  ~GBRForestWriter() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string moduleLabel_;

  bool hasRun_;

  typedef std::vector<std::string> vstring;

  struct categoryEntryType {
    categoryEntryType(const edm::ParameterSet& cfg) {
      if (cfg.existsAs<edm::FileInPath>("inputFileName")) {
        edm::FileInPath inputFileName_fip = cfg.getParameter<edm::FileInPath>("inputFileName");
        inputFileName_ = inputFileName_fip.fullPath();
      } else if (cfg.existsAs<std::string>("inputFileName")) {
        inputFileName_ = cfg.getParameter<std::string>("inputFileName");
      } else
        throw cms::Exception("GBRForestWriter") << " Undefined Configuration Parameter 'inputFileName !!\n";
      std::string inputFileType_string = cfg.getParameter<std::string>("inputFileType");
      if (inputFileType_string == "XML")
        inputFileType_ = kXML;
      else if (inputFileType_string == "GBRForest")
        inputFileType_ = kGBRForest;
      else
        throw cms::Exception("GBRForestWriter")
            << " Invalid Configuration Parameter 'inputFileType' = " << inputFileType_string << " !!\n";
      if (inputFileType_ == kXML) {
        inputVariables_ = cfg.getParameter<vstring>("inputVariables");
        spectatorVariables_ = cfg.getParameter<vstring>("spectatorVariables");
        methodName_ = cfg.getParameter<std::string>("methodName");
        gbrForestName_ =
            (cfg.existsAs<std::string>("gbrForestName") ? cfg.getParameter<std::string>("gbrForestName") : methodName_);
      } else {
        gbrForestName_ = cfg.getParameter<std::string>("gbrForestName");
      }
    }
    ~categoryEntryType() {}
    std::string inputFileName_;
    enum { kXML, kGBRForest };
    int inputFileType_;
    vstring inputVariables_;
    vstring spectatorVariables_;
    std::string gbrForestName_;
    std::string methodName_;
  };
  struct jobEntryType {
    jobEntryType(const edm::ParameterSet& cfg) {
      if (cfg.exists("categories")) {
        edm::VParameterSet cfgCategories = cfg.getParameter<edm::VParameterSet>("categories");
        for (edm::VParameterSet::const_iterator cfgCategory = cfgCategories.begin(); cfgCategory != cfgCategories.end();
             ++cfgCategory) {
          categoryEntryType* category = new categoryEntryType(*cfgCategory);
          categories_.push_back(category);
        }
      } else {
        categoryEntryType* category = new categoryEntryType(cfg);
        categories_.push_back(category);
      }
      std::string outputFileType_string = cfg.getParameter<std::string>("outputFileType");
      if (outputFileType_string == "GBRForest")
        outputFileType_ = kGBRForest;
      else if (outputFileType_string == "SQLLite")
        outputFileType_ = kSQLLite;
      else
        throw cms::Exception("GBRForestWriter")
            << " Invalid Configuration Parameter 'outputFileType' = " << outputFileType_string << " !!\n";
      if (outputFileType_ == kGBRForest) {
        outputFileName_ = cfg.getParameter<std::string>("outputFileName");
      }
      if (outputFileType_ == kSQLLite) {
        outputRecord_ = cfg.getParameter<std::string>("outputRecord");
      }
    }
    ~jobEntryType() {
      for (std::vector<categoryEntryType*>::iterator it = categories_.begin(); it != categories_.end(); ++it) {
        delete (*it);
      }
    }
    std::vector<categoryEntryType*> categories_;
    enum { kGBRForest, kSQLLite };
    int outputFileType_;
    std::string outputFileName_;
    std::string outputRecord_;
  };
  std::vector<jobEntryType*> jobs_;
};

GBRForestWriter::GBRForestWriter(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
  edm::VParameterSet cfgJobs = cfg.getParameter<edm::VParameterSet>("jobs");
  for (edm::VParameterSet::const_iterator cfgJob = cfgJobs.begin(); cfgJob != cfgJobs.end(); ++cfgJob) {
    jobEntryType* job = new jobEntryType(*cfgJob);
    jobs_.push_back(job);
  }
}

GBRForestWriter::~GBRForestWriter() {
  for (std::vector<jobEntryType*>::iterator it = jobs_.begin(); it != jobs_.end(); ++it) {
    delete (*it);
  }
}

void GBRForestWriter::analyze(const edm::Event&, const edm::EventSetup&) {
  for (std::vector<jobEntryType*>::iterator job = jobs_.begin(); job != jobs_.end(); ++job) {
    std::map<std::string, const GBRForest*> gbrForests;  // key = name
    for (std::vector<categoryEntryType*>::iterator category = (*job)->categories_.begin();
         category != (*job)->categories_.end();
         ++category) {
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
        throw cms::Exception("GBRForestWriter") << " Failed to load GBRForest = " << (*category)->gbrForestName_.data()
                                                << " from file = " << (*category)->inputFileName_ << " !!\n";
      gbrForests[(*category)->gbrForestName_] = gbrForest;
    }
    if ((*job)->outputFileType_ == jobEntryType::kGBRForest) {
      TFile* outputFile = new TFile((*job)->outputFileName_.data(), "RECREATE");

      for (std::map<std::string, const GBRForest*>::iterator gbrForest = gbrForests.begin();
           gbrForest != gbrForests.end();
           ++gbrForest) {
        outputFile->WriteObject(gbrForest->second, gbrForest->first.data());
      }
      delete outputFile;
    } else if ((*job)->outputFileType_ == jobEntryType::kSQLLite) {
      edm::Service<cond::service::PoolDBOutputService> dbService;
      if (!dbService.isAvailable())
        throw cms::Exception("GBRForestWriter") << " Failed to access PoolDBOutputService !!\n";

      for (std::map<std::string, const GBRForest*>::iterator gbrForest = gbrForests.begin();
           gbrForest != gbrForests.end();
           ++gbrForest) {
        std::string outputRecord = (*job)->outputRecord_;
        if (gbrForests.size() > 1)
          outputRecord.append("_").append(gbrForest->first);
        dbService->writeOne(gbrForest->second, dbService->beginOfTime(), outputRecord);
      }
    }

    // gbrforest deletion
    for (std::map<std::string, const GBRForest*>::iterator gbrForest = gbrForests.begin();
         gbrForest != gbrForests.end();
         ++gbrForest) {
      delete gbrForest->second;
    }
  }
}

DEFINE_FWK_MODULE(GBRForestWriter);
