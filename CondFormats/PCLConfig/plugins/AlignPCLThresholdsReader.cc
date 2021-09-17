#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsRcd.h"

namespace edmtest {
  class AlignPCLThresholdsReader : public edm::one::EDAnalyzer<> {
  public:
    explicit AlignPCLThresholdsReader(edm::ParameterSet const& p);
    ~AlignPCLThresholdsReader() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    // ----------member data ---------------------------
    const edm::ESGetToken<AlignPCLThresholds, AlignPCLThresholdsRcd> thresholdToken_;
    const bool printdebug_;
    const std::string formatedOutput_;
  };

  AlignPCLThresholdsReader::AlignPCLThresholdsReader(edm::ParameterSet const& p)
      : thresholdToken_(esConsumes()),
        printdebug_(p.getUntrackedParameter<bool>("printDebug", true)),
        formatedOutput_(p.getUntrackedParameter<std::string>("outputFile", "")) {
    edm::LogInfo("AlignPCLThresholdsReader") << "AlignPCLThresholdsReader" << std::endl;
  }

  AlignPCLThresholdsReader::~AlignPCLThresholdsReader() {
    edm::LogInfo("AlignPCLThresholdsReader") << "~AlignPCLThresholdsReader " << std::endl;
  }

  void AlignPCLThresholdsReader::analyze(const edm::Event& e, const edm::EventSetup& context) {
    edm::LogInfo("AlignPCLThresholdsReader") << "### AlignPCLThresholdsReader::analyze  ###" << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << " ---EVENT NUMBER " << e.id().event() << std::endl;

    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType("AlignPCLThresholdsRcd"));

    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogInfo("AlignPCLThresholdsReader") << "Record \"AlignPCLThresholdsRcd"
                                               << "\" does not exist " << std::endl;
    }

    //this part gets the handle of the event source and the record (i.e. the Database)
    edm::ESHandle<AlignPCLThresholds> thresholdHandle = context.getHandle(thresholdToken_);
    edm::LogInfo("AlignPCLThresholdsReader") << "got eshandle" << std::endl;

    if (!thresholdHandle.isValid()) {
      edm::LogError("AlignPCLThresholdsReader") << " Could not get Handle" << std::endl;
      return;
    }

    const AlignPCLThresholds* thresholds = thresholdHandle.product();
    edm::LogInfo("AlignPCLThresholdsReader") << "got AlignPCLThresholds* " << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << "print  pointer address : ";
    edm::LogInfo("AlignPCLThresholdsReader") << thresholds << std::endl;

    edm::LogInfo("AlignPCLThresholdsReader") << "Size " << thresholds->size() << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << "Content of myThresholds " << std::endl;
    // use built-in method in the CondFormat to print the content
    if (printdebug_) {
      thresholds->printAll();
    }

    FILE* pFile = nullptr;
    if (!formatedOutput_.empty())
      pFile = fopen(formatedOutput_.c_str(), "w");
    if (pFile) {
      fprintf(pFile, "AlignPCLThresholds::printAll() \n");
      fprintf(pFile,
              " ======================================================================================================="
              "============ \n");
      fprintf(pFile, "N records cut: %i \n", thresholds->getNrecords());

      AlignPCLThresholds::threshold_map m_thresholds = thresholds->getThreshold_Map();

      for (auto it = m_thresholds.begin(); it != m_thresholds.end(); ++it) {
        fprintf(pFile,
                " ====================================================================================================="
                "============== \n");
        fprintf(pFile, "key : %s \n ", (it->first).c_str());
        fprintf(pFile, "- Xcut             : %8.3f   um   ", (it->second).getXcut());
        fprintf(pFile, "| sigXcut          : %8.3f        ", (it->second).getSigXcut());
        fprintf(pFile, "| maxMoveXcut      : %8.3f   um   ", (it->second).getMaxMoveXcut());
        fprintf(pFile, "| ErrorXcut        : %8.3f   um\n ", (it->second).getErrorXcut());

        fprintf(pFile, "- thetaXcut        : %8.3f urad   ", (it->second).getThetaXcut());
        fprintf(pFile, "| sigThetaXcut     : %8.3f        ", (it->second).getSigThetaXcut());
        fprintf(pFile, "| maxMoveThetaXcut : %8.3f urad   ", (it->second).getMaxMoveThetaXcut());
        fprintf(pFile, "| ErrorThetaXcut   : %8.3f urad\n ", (it->second).getErrorThetaXcut());

        fprintf(pFile, "- Ycut             : %8.3f   um   ", (it->second).getYcut());
        fprintf(pFile, "| sigYcut          : %8.3f        ", (it->second).getSigXcut());
        fprintf(pFile, "| maxMoveYcut      : %8.3f   um   ", (it->second).getMaxMoveYcut());
        fprintf(pFile, "| ErrorYcut        : %8.3f   um\n ", (it->second).getErrorYcut());

        fprintf(pFile, "- thetaYcut        : %8.3f urad   ", (it->second).getThetaYcut());
        fprintf(pFile, "| sigThetaYcut     : %8.3f        ", (it->second).getSigThetaYcut());
        fprintf(pFile, "| maxMoveThetaYcut : %8.3f urad   ", (it->second).getMaxMoveThetaYcut());
        fprintf(pFile, "| ErrorThetaYcut   : %8.3f urad\n ", (it->second).getErrorThetaYcut());

        fprintf(pFile, "- Zcut             : %8.3f   um   ", (it->second).getZcut());
        fprintf(pFile, "| sigZcut          : %8.3f        ", (it->second).getSigZcut());
        fprintf(pFile, "| maxMoveZcut      : %8.3f   um   ", (it->second).getMaxMoveZcut());
        fprintf(pFile, "| ErrorZcut        : %8.3f   um\n ", (it->second).getErrorZcut());

        fprintf(pFile, "- thetaZcut        : %8.3f urad   ", (it->second).getThetaZcut());
        fprintf(pFile, "| sigThetaZcut     : %8.3f        ", (it->second).getSigThetaZcut());
        fprintf(pFile, "| maxMoveThetaZcut : %8.3f urad   ", (it->second).getMaxMoveThetaZcut());
        fprintf(pFile, "| ErrorThetaZcut   : %8.3f urad\n ", (it->second).getErrorThetaZcut());

        if ((it->second).hasExtraDOF()) {
          for (unsigned int j = 0; j < (it->second).extraDOFSize(); j++) {
            std::array<float, 4> extraDOFCuts = thresholds->getExtraDOFCutsForAlignable(it->first, j);
            fprintf(pFile,
                    "Extra DOF: %i with label %s \n ",
                    j,
                    thresholds->getExtraDOFLabelForAlignable(it->first, j).c_str());
            fprintf(pFile, "- cut              : %8.3f        ", extraDOFCuts.at(0));
            fprintf(pFile, "| sigCut           : %8.3f        ", extraDOFCuts.at(1));
            fprintf(pFile, "| maxMoveCut       : %8.3f        ", extraDOFCuts.at(2));
            fprintf(pFile, "| maxErrorCut      : %8.3f     \n ", extraDOFCuts.at(3));
          }
        }
      }
    }
  }

  void AlignPCLThresholdsReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Reads payloads of type AlignPCLThresholds");
    desc.addUntracked<bool>("printDebug", true);
    desc.addUntracked<std::string>("outputFile", "");
    descriptions.add("AlignPCLThresholdsReader", desc);
  }

  DEFINE_FWK_MODULE(AlignPCLThresholdsReader);
}  // namespace edmtest
