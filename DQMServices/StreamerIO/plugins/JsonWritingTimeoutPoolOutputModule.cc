#include "JsonWritingTimeoutPoolOutputModule.h"

#include "DQMServices/Components/interface/fillJson.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <filesystem>

#include <boost/property_tree/json_parser.hpp>

#include <fmt/printf.h>

namespace dqmservices {

  JsonWritingTimeoutPoolOutputModule::JsonWritingTimeoutPoolOutputModule(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase::OutputModuleBase(ps),
        edm::TimeoutPoolOutputModule(ps),
        runNumber_(ps.getUntrackedParameter<uint32_t>("runNumber")),
        streamLabel_(ps.getUntrackedParameter<std::string>("streamLabel")),
        outputPath_(ps.getUntrackedParameter<std::string>("outputPath")),
        sequence_(0) {}

  std::pair<std::string, std::string> JsonWritingTimeoutPoolOutputModule::physicalAndLogicalNameForNewFile() {
    sequence_++;

    std::string base = fmt::sprintf("run%06d_ls%04d_%s", runNumber_, sequence_, streamLabel_);

    std::filesystem::path p(outputPath_);

    currentFileName_ = (p / base).string() + ".root";
    currentJsonName_ = (p / base).string() + ".jsn";

    return std::make_pair(currentFileName_, currentFileName_);
  }

  void JsonWritingTimeoutPoolOutputModule::doExtrasAfterCloseFile() {
    std::string json_tmp_ = currentJsonName_ + ".open";
    std::string transferDest = "";
    std::string mergeType = "ROOT";
    auto pt = dqmfilesaver::fillJson(runNumber_, sequence_, currentFileName_, transferDest, mergeType, nullptr);
    write_json(json_tmp_, pt);
    rename(json_tmp_.c_str(), currentJsonName_.c_str());
  }

  void JsonWritingTimeoutPoolOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TimeoutPoolOutputModule::fillDescription(desc);

    desc.setComment(
        "Almost same as TimeoutPoolOutputModule, but the output files names "
        "follow the FFF naming convention. Additionally a json 'description' "
        "file is emitted for every .root file written.");

    desc.addUntracked<uint32_t>("runNumber", 0)
        ->setComment(
            "The run number, only used for file prefix: "
            "'run000001_lumi0000_...'.");

    desc.addUntracked<std::string>("outputPath", "./")
        ->setComment(
            "Output path for the root and json files, usually the run "
            "directory.");

    desc.addUntracked<std::string>("streamLabel", "streamEvDOutput")->setComment("Stream label, used for file suffix.");

    descriptions.add("jsonWriting", desc);
  }

}  // namespace dqmservices

#include "FWCore/Framework/interface/MakerMacros.h"
using dqmservices::JsonWritingTimeoutPoolOutputModule;
DEFINE_FWK_MODULE(JsonWritingTimeoutPoolOutputModule);
