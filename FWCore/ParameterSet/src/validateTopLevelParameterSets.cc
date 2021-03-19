#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ThreadsInfo.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <sstream>
#include <vector>
#include <string>

namespace edm {

  void fillOptionsDescription(ParameterSetDescription& description) {
    description.addUntracked<unsigned int>("numberOfThreads", s_defaultNumberOfThreads)
        ->setComment("If zero, let TBB use its default which is normally the number of CPUs on the machine");
    description.addUntracked<unsigned int>("numberOfStreams", 0)
        ->setComment("If zero, then set the number of streams to be the same as the number of threads");
    description.addUntracked<unsigned int>("numberOfConcurrentRuns", 1);
    description.addUntracked<unsigned int>("numberOfConcurrentLuminosityBlocks", 1)
        ->setComment("If zero, then set the same as the number of runs");

    edm::ParameterSetDescription eventSetupDescription;
    eventSetupDescription.addUntracked<unsigned int>("numberOfConcurrentIOVs", 1)
        ->setComment(
            "If zero, set to 1. Can be overridden by hard coded static in record C++ definition or by "
            "forceNumberOfConcurrentIOVs");
    edm::ParameterSetDescription nestedDescription;
    nestedDescription.addWildcardUntracked<unsigned int>("*")->setComment(
        "Parameter names should be record names and the values are the number of concurrent IOVS for each record."
        " Overrides all other methods of setting number of concurrent IOVs.");
    eventSetupDescription.addUntracked<edm::ParameterSetDescription>("forceNumberOfConcurrentIOVs", nestedDescription);
    description.addUntracked<edm::ParameterSetDescription>("eventSetup", eventSetupDescription);

    description.addUntracked<bool>("wantSummary", false)
        ->setComment("Set true to print a report on the trigger decisions and timing of modules");
    description.addUntracked<std::string>("fileMode", "FULLMERGE")
        ->setComment("Legal values are 'NOMERGE' and 'FULLMERGE'");
    description.addUntracked<bool>("forceEventSetupCacheClearOnNewRun", false);
    description.addUntracked<bool>("throwIfIllegalParameter", true)
        ->setComment("Set false to disable exception throws when configuration validation detects illegal parameters");
    description.addUntracked<bool>("printDependencies", false)->setComment("Print data dependencies between modules");
    description.addUntracked<bool>("deleteNonConsumedUnscheduledModules", true)
        ->setComment(
            "Delete modules that are unscheduled, i.e. only in Tasks, whose products are not consumed by any other "
            "otherwise-running module");

    // No default for this one because the parameter value is
    // actually used in the main function in cmsRun.cpp before
    // the parameter set is validated here.
    description.addOptionalUntracked<unsigned int>("sizeOfStackForThreadsInKB");

    std::vector<std::string> emptyVector;

    description.addUntracked<std::vector<std::string>>("Rethrow", emptyVector);
    description.addUntracked<std::vector<std::string>>("SkipEvent", emptyVector);
    description.addUntracked<std::vector<std::string>>("FailPath", emptyVector);
    description.addUntracked<std::vector<std::string>>("IgnoreCompletely", emptyVector);

    description.addUntracked<std::vector<std::string>>("canDeleteEarly", emptyVector)
        ->setComment("Branch names of products that the Framework can try to delete before the end of the Event");

    description.addOptionalUntracked<bool>("allowUnscheduled")
        ->setComment(
            "Obsolete. Has no effect. Allowed only for backward compatibility for old Python configuration files.");
    description.addOptionalUntracked<std::string>("emptyRunLumiMode")
        ->setComment(
            "Obsolete. Has no effect. Allowed only for backward compatibility for old Python configuration files.");
    description.addOptionalUntracked<bool>("makeTriggerResults")
        ->setComment(
            "Obsolete. Has no effect. Allowed only for backward compatibility for old Python configuration files.");
  }

  void fillMaxEventsDescription(ParameterSetDescription& description) {
    description.addUntracked<int>("input", -1)->setComment("Default of -1 implies no limit.");

    ParameterSetDescription nestedDescription;
    nestedDescription.addWildcardUntracked<int>("*");
    description.addOptionalNode(ParameterDescription<int>("output", false) xor
                                    ParameterDescription<ParameterSetDescription>("output", nestedDescription, false),
                                false);
  }

  void fillMaxLuminosityBlocksDescription(ParameterSetDescription& description) {
    description.addUntracked<int>("input", -1)->setComment("Default of -1 implies no limit.");
  }

  void fillMaxSecondsUntilRampdownDescription(ParameterSetDescription& description) {
    description.addUntracked<int>("input", -1)->setComment("Default of -1 implies no limit.");
  }

  void validateTopLevelParameterSets(ParameterSet* processParameterSet) {
    std::string processName = processParameterSet->getParameter<std::string>("@process_name");

    std::vector<std::string> psetNames{"options", "maxEvents", "maxLuminosityBlocks", "maxSecondsUntilRampdown"};

    for (auto const& psetName : psetNames) {
      bool isTracked{false};
      ParameterSet* pset = processParameterSet->getPSetForUpdate(psetName, isTracked);
      if (pset == nullptr) {
        ParameterSet emptyPset;
        processParameterSet->addUntrackedParameter<ParameterSet>(psetName, emptyPset);
        pset = processParameterSet->getPSetForUpdate(psetName, isTracked);
      }
      if (isTracked) {
        throw Exception(errors::Configuration) << "In the configuration the top level parameter set named \'"
                                               << psetName << "\' in process \'" << processName << "\' is tracked.\n"
                                               << "It must be untracked";
      }

      ParameterSetDescription description;
      if (psetName == "options") {
        fillOptionsDescription(description);
      } else if (psetName == "maxEvents") {
        fillMaxEventsDescription(description);
      } else if (psetName == "maxLuminosityBlocks") {
        fillMaxLuminosityBlocksDescription(description);
      } else if (psetName == "maxSecondsUntilRampdown") {
        fillMaxSecondsUntilRampdownDescription(description);
      }

      try {
        description.validate(*pset);
      } catch (cms::Exception& ex) {
        std::ostringstream ost;
        ost << "Validating top level \'" << psetName << "\' ParameterSet for process \'" << processName << "\'";
        ex.addContext(ost.str());
        throw;
      }
    }
  }

}  // namespace edm
