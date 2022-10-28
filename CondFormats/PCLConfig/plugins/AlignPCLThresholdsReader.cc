#include <array>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholds.h"
#include "CondFormats/PCLConfig/interface/AlignPCLThresholdsHG.h"
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/AlignPCLThresholdsHGRcd.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

namespace edmtest {
  template <typename T, typename R>
  class AlignPCLThresholdsReader : public edm::one::EDAnalyzer<> {
  public:
    explicit AlignPCLThresholdsReader(edm::ParameterSet const& p);
    ~AlignPCLThresholdsReader() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

    // ----------member data ---------------------------
    const edm::ESGetToken<T, R> thresholdToken_;
    const bool printdebug_;
    const std::string formatedOutput_;
  };

  template <typename T, typename R>
  AlignPCLThresholdsReader<T, R>::AlignPCLThresholdsReader(edm::ParameterSet const& p)
      : thresholdToken_(esConsumes()),
        printdebug_(p.getUntrackedParameter<bool>("printDebug", true)),
        formatedOutput_(p.getUntrackedParameter<std::string>("outputFile", "")) {
    edm::LogInfo("AlignPCLThresholdsReader") << "AlignPCLThresholdsReader" << std::endl;
  }

  template <typename T, typename R>
  AlignPCLThresholdsReader<T, R>::~AlignPCLThresholdsReader() {
    edm::LogInfo("AlignPCLThresholdsReader") << "~AlignPCLThresholdsReader " << std::endl;
  }

  template <typename T, typename R>
  void AlignPCLThresholdsReader<T, R>::analyze(const edm::Event& e, const edm::EventSetup& context) {
    edm::LogInfo("AlignPCLThresholdsReader") << "### AlignPCLThresholdsReader::analyze  ###" << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << " ---EVENT NUMBER " << e.id().event() << std::endl;

    edm::eventsetup::EventSetupRecordKey inputKey = edm::eventsetup::EventSetupRecordKey::makeKey<R>();
    edm::eventsetup::EventSetupRecordKey recordKey(
        edm::eventsetup::EventSetupRecordKey::TypeTag::findType(inputKey.type().name()));

    if (recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      edm::LogInfo("AlignPCLThresholdsReader")
          << "Record \"" << inputKey.type().name() << "\" does not exist " << std::endl;
    }

    //this part gets the handle of the event source and the record (i.e. the Database)
    edm::ESHandle<T> thresholdHandle = context.getHandle(thresholdToken_);
    edm::LogInfo("AlignPCLThresholdsReader") << "got eshandle" << std::endl;

    if (!thresholdHandle.isValid()) {
      edm::LogError("AlignPCLThresholdsReader") << " Could not get Handle" << std::endl;
      return;
    }

    const T* thresholds = thresholdHandle.product();
    edm::LogInfo("AlignPCLThresholdsReader") << "got AlignPCLThresholds* " << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << "print  pointer address : ";
    edm::LogInfo("AlignPCLThresholdsReader") << thresholds << std::endl;

    edm::LogInfo("AlignPCLThresholdsReader") << "Size " << thresholds->size() << std::endl;
    edm::LogInfo("AlignPCLThresholdsReader") << "Content of myThresholds " << std::endl;
    // use built-in method in the CondFormat to print the content
    if (thresholds && printdebug_) {
      thresholds->printAll();
    }

    FILE* pFile = nullptr;
    if (!formatedOutput_.empty())
      pFile = fopen(formatedOutput_.c_str(), "w");
    if (pFile) {
      fprintf(pFile, "AlignPCLThresholds::printAll() \n");
      fprintf(pFile,
              " ======================================================================================================="
              "============\n");
      fprintf(pFile, "N records cut: %i \n", thresholds->getNrecords());

      AlignPCLThresholds::threshold_map m_thresholds = thresholds->getThreshold_Map();
      AlignPCLThresholdsHG::param_map m_floatMap{};

      if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
        m_floatMap = thresholds->getFloatMap();
      }

      for (auto it = m_thresholds.begin(); it != m_thresholds.end(); ++it) {
        bool hasFractionCut = (m_floatMap.find(it->first) != m_floatMap.end());

        fprintf(pFile,
                " ====================================================================================================="
                "==============\n");
        fprintf(pFile, "key : %s \n", (it->first).c_str());
        fprintf(pFile, "- Xcut             : %8.3f   um   ", (it->second).getXcut());
        fprintf(pFile, "| sigXcut          : %8.3f        ", (it->second).getSigXcut());
        fprintf(pFile, "| maxMoveXcut      : %8.3f   um   ", (it->second).getMaxMoveXcut());
        fprintf(pFile, "| ErrorXcut        : %8.3f   um   ", (it->second).getErrorXcut());
        if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
          if (hasFractionCut) {
            fprintf(pFile,
                    "| X_fractionCut      : %8.3f    \n",
                    thresholds->getFractionCut(it->first, AlignPCLThresholds::coordType::X));
          } else {
            fprintf(pFile, "\n");
          }
        } else {
          fprintf(pFile, "\n");
        }

        fprintf(pFile, "- thetaXcut        : %8.3f urad   ", (it->second).getThetaXcut());
        fprintf(pFile, "| sigThetaXcut     : %8.3f        ", (it->second).getSigThetaXcut());
        fprintf(pFile, "| maxMoveThetaXcut : %8.3f urad   ", (it->second).getMaxMoveThetaXcut());
        fprintf(pFile, "| ErrorThetaXcut   : %8.3f urad   ", (it->second).getErrorThetaXcut());
        if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
          if (hasFractionCut) {
            fprintf(pFile,
                    "| thetaX_fractionCut : %8.3f    \n",
                    thresholds->getFractionCut(it->first, AlignPCLThresholds::coordType::theta_X));
          } else {
            fprintf(pFile, "\n");
          }
        } else {
          fprintf(pFile, "\n");
        }

        fprintf(pFile, "- Ycut             : %8.3f   um   ", (it->second).getYcut());
        fprintf(pFile, "| sigYcut          : %8.3f        ", (it->second).getSigXcut());
        fprintf(pFile, "| maxMoveYcut      : %8.3f   um   ", (it->second).getMaxMoveYcut());
        fprintf(pFile, "| ErrorYcut        : %8.3f   um   ", (it->second).getErrorYcut());
        if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
          if (hasFractionCut) {
            fprintf(pFile,
                    "| Y_fractionCut      : %8.3f    \n",
                    thresholds->getFractionCut(it->first, AlignPCLThresholds::coordType::Y));
          } else {
            fprintf(pFile, "\n");
          }
        } else {
          fprintf(pFile, "\n");
        }

        fprintf(pFile, "- thetaYcut        : %8.3f urad   ", (it->second).getThetaYcut());
        fprintf(pFile, "| sigThetaYcut     : %8.3f        ", (it->second).getSigThetaYcut());
        fprintf(pFile, "| maxMoveThetaYcut : %8.3f urad   ", (it->second).getMaxMoveThetaYcut());
        fprintf(pFile, "| ErrorThetaYcut   : %8.3f urad   ", (it->second).getErrorThetaYcut());
        if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
          if (hasFractionCut) {
            fprintf(pFile,
                    "| thetaY_fractionCut : %8.3f    \n",
                    thresholds->getFractionCut(it->first, AlignPCLThresholds::coordType::theta_Y));
          } else {
            fprintf(pFile, "\n");
          }
        } else {
          fprintf(pFile, "\n");
        }

        fprintf(pFile, "- Zcut             : %8.3f   um   ", (it->second).getZcut());
        fprintf(pFile, "| sigZcut          : %8.3f        ", (it->second).getSigZcut());
        fprintf(pFile, "| maxMoveZcut      : %8.3f   um   ", (it->second).getMaxMoveZcut());
        fprintf(pFile, "| ErrorZcut        : %8.3f   um   ", (it->second).getErrorZcut());
        if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
          if (hasFractionCut) {
            fprintf(pFile,
                    "| Z_fractionCut      : %8.3f    \n",
                    thresholds->getFractionCut(it->first, AlignPCLThresholds::coordType::Z));
          } else {
            fprintf(pFile, "\n");
          }
        } else {
          fprintf(pFile, "\n");
        }

        fprintf(pFile, "- thetaZcut        : %8.3f urad   ", (it->second).getThetaZcut());
        fprintf(pFile, "| sigThetaZcut     : %8.3f        ", (it->second).getSigThetaZcut());
        fprintf(pFile, "| maxMoveThetaZcut : %8.3f urad   ", (it->second).getMaxMoveThetaZcut());
        fprintf(pFile, "| ErrorThetaZcut   : %8.3f urad   ", (it->second).getErrorThetaZcut());
        if constexpr (std::is_same_v<T, AlignPCLThresholdsHG>) {
          if (hasFractionCut) {
            fprintf(pFile,
                    "| thetaZ_fractionCut : %8.3f    \n",
                    thresholds->getFractionCut(it->first, AlignPCLThresholds::coordType::theta_Z));
          } else {
            fprintf(pFile, "\n");
          }
        } else {
          fprintf(pFile, "\n");
        }

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

  template <typename T, typename R>
  void AlignPCLThresholdsReader<T, R>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("Reads payloads of type AlignPCLThresholds");
    desc.addUntracked<bool>("printDebug", true);
    desc.addUntracked<std::string>("outputFile", "");
    descriptions.add(defaultModuleLabel<AlignPCLThresholdsReader<T, R>>(), desc);
  }

  typedef AlignPCLThresholdsReader<AlignPCLThresholds, AlignPCLThresholdsRcd> AlignPCLThresholdsLGReader;
  typedef AlignPCLThresholdsReader<AlignPCLThresholdsHG, AlignPCLThresholdsHGRcd> AlignPCLThresholdsHGReader;

  DEFINE_FWK_MODULE(AlignPCLThresholdsLGReader);
  DEFINE_FWK_MODULE(AlignPCLThresholdsHGReader);
}  // namespace edmtest
