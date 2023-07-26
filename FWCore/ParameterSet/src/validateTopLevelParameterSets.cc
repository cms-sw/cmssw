#include "FWCore/ParameterSet/interface/validateTopLevelParameterSets.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ThreadsInfo.h"

#include "FWCore/Utilities/interface/EDMException.h"

#include <sstream>
#include <vector>
#include <string>

namespace edm {

  // NOTE: The defaults given here are not actually used when running cmsRun
  // Those come from hard coded values in the Python code in Config.py
  // The defaults here are used when running the edmPluginHelp utility so
  // it is important the defaults in both places are consistent.

  void fillOptionsDescription(ParameterSetDescription& description) {
    description.addUntracked<unsigned int>("numberOfThreads", s_defaultNumberOfThreads)
        ->setComment("If zero, let TBB use its default which is normally the number of CPUs on the machine");
    description.addUntracked<unsigned int>("numberOfStreams", 0)
        ->setComment(
            "If zero, then set the number of streams to be the same as the number of "
            "Threads (except always 1 if there is a looper)");
    description.addUntracked<unsigned int>("numberOfConcurrentLuminosityBlocks", 0)
        ->setComment(
            "If zero, use Framework default (currently 2 when the number of streams >= 2, otherwise 1). "
            "In all cases, the number of concurrent luminosity blocks will be reset to "
            "be the same as the number of streams if it is greater than the "
            "numbers of streams.");
    description.addUntracked<unsigned int>("numberOfConcurrentRuns", 1)
        ->setComment(
            "If zero or greater than the number of concurrent luminosity blocks, this will be reset to "
            "be the same as the number of concurrent luminosity blocks.");

    edm::ParameterSetDescription eventSetupDescription;
    eventSetupDescription.addUntracked<unsigned int>("numberOfConcurrentIOVs", 0)
        ->setComment(
            "If zero, use the Framework default which currently means the same as the "
            "number of concurrent luminosity blocks. Can be overridden by a hard coded "
            "static in a record C++ definition or by forceNumberOfConcurrentIOVs. "
            "In all cases, the number of concurrent IOVs will be reset to be the "
            "same as the number of concurrent luminosity blocks if greater than the "
            "number of concurrent luminosity blocks.");
    edm::ParameterSetDescription nestedDescription;
    nestedDescription.addWildcardUntracked<unsigned int>("*")->setComment(
        "Parameter names should be record names and the values are the number of concurrent IOVS for each record."
        " Overrides all other methods of setting number of concurrent IOVs.");
    eventSetupDescription.addUntracked<edm::ParameterSetDescription>("forceNumberOfConcurrentIOVs", nestedDescription);
    description.addUntracked<edm::ParameterSetDescription>("eventSetup", eventSetupDescription);

    description.addUntracked<std::vector<std::string>>("accelerators", {"*"})
        ->setComment(
            "Specify the set of compute accelerator(s) the job is allowed to use. The values can contain the direct "
            "names of accelerators supported by the ProcessAccelerators defined in the configuration, or patterns "
            "matching to them (patterns use '*' and '?' wildcards similar to shell). The actual set of accelerators to "
            "be used is determined on the worker node based on the available hardware. A CPU fallback with the name "
            "'cpu' is always included in the set of available accelerators. If no accelerator matching to the patterns "
            "are available on the worker node, the job is terminated with a specific error code. Same happens if an "
            "empty value is given in the configuration. Default value is pattern '*', which implies use of any "
            "supported and available hardware (including the CPU fallback).");
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
    description.addUntracked<std::vector<std::string>>("TryToContinue", emptyVector);
    description.addUntracked<std::vector<std::string>>("IgnoreCompletely", emptyVector);

    description.addUntracked<std::vector<std::string>>("canDeleteEarly", emptyVector)
        ->setComment("Branch names of products that the Framework can try to delete before the end of the Event");

    {
      edm::ParameterSetDescription validator;
      validator.add<std::string>("product");
      validator.add<std::vector<std::string>>("references")
          ->setComment("All the branch names for products to which 'product' contains a reference.");
      description.addVPSetUntracked("holdsReferencesToDeleteEarly", validator, std::vector<edm::ParameterSet>{})
          ->setComment(
              "The 'product' branch name of product which internally hold references to data in another product");
    }
    description.addUntracked<std::vector<std::string>>("modulesToIgnoreForDeleteEarly", emptyVector)
        ->setComment(
            "labels of modules whose consumes information will be ingored when determing lifetime for delete early "
            "data products");
    description.addUntracked<bool>("dumpOptions", false)
        ->setComment(
            "Print values of selected Framework parameters. The Framework might modify the values "
            "in the options parameter set and this prints the values after that modification.");

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

  void dumpOptionsToLogFile(unsigned int nThreads,
                            unsigned int nStreams,
                            unsigned int nConcurrentLumis,
                            unsigned int nConcurrentRuns) {
    LogAbsolute("Options") << "Number of Threads = " << nThreads << "\nNumber of Streams = " << nStreams
                           << "\nNumber of Concurrent Lumis = " << nConcurrentLumis
                           << "\nNumber of Concurrent Runs = " << nConcurrentRuns;
  }
}  // namespace edm
