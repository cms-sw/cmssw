#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace dqmservices {
  class DQMStreamerWriteJsonAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    DQMStreamerWriteJsonAnalyzer(edm::ParameterSet const& iPSet);

    void analyze(edm::Event const&, edm::EventSetup const&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& iDesc);

  private:
    void writeJson() const;
    void writeEndJob() const;

    boost::filesystem::path writePath_;
    unsigned int const eventsPerLumi_;
    unsigned int const runNumber_;
    std::string const streamName_;
    std::vector<std::string> const dataFileForEachLumi_;
    unsigned int nEventsSeenSinceWrite_;
    unsigned int fileIndex_;
  };

  DQMStreamerWriteJsonAnalyzer::DQMStreamerWriteJsonAnalyzer(edm::ParameterSet const& iPSet)
      : eventsPerLumi_(iPSet.getUntrackedParameter<unsigned int>("eventsPerLumi")),
        runNumber_(iPSet.getUntrackedParameter<unsigned int>("runNumber")),
        streamName_(iPSet.getUntrackedParameter<std::string>("streamName")),
        dataFileForEachLumi_(iPSet.getUntrackedParameter<std::vector<std::string>>("dataFileForEachLumi")),
        nEventsSeenSinceWrite_{0},
        fileIndex_{0} {
    boost::filesystem::path path = iPSet.getUntrackedParameter<std::string>("pathToWriteJson");
    writePath_ /= str(boost::format("run%06d") % runNumber_);

    boost::filesystem::create_directories(writePath_);
  }

  void DQMStreamerWriteJsonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription pset;
    pset.addUntracked<unsigned int>("eventsPerLumi");
    pset.addUntracked<unsigned int>("runNumber");
    pset.addUntracked<std::string>("streamName");
    pset.addUntracked<std::string>("pathToWriteJson");
    pset.addUntracked<std::vector<std::string>>("dataFileForEachLumi");

    iDesc.addDefault(pset);
  }

  void DQMStreamerWriteJsonAnalyzer::analyze(edm::Event const&, edm::EventSetup const&) {
    if (++nEventsSeenSinceWrite_ >= eventsPerLumi_) {
      if (fileIndex_ == dataFileForEachLumi_.size()) {
        writeEndJob();
        return;
      }
      writeJson();
      ++fileIndex_;
      nEventsSeenSinceWrite_ = 0;
      return;
    }
  }

  void DQMStreamerWriteJsonAnalyzer::writeJson() const {
    auto currentFileBase =
        str(boost::format("run%06d_ls%04d_%s_local.jsn") % runNumber_ % (fileIndex_ + 2) % streamName_);
    auto currentJsonPath = (writePath_ / currentFileBase).string();

    using namespace boost::property_tree;
    ptree pt;
    ptree data;

    ptree child1, child2, child3;
    child1.put("", nEventsSeenSinceWrite_);  // Processed
    child2.put("", nEventsSeenSinceWrite_);  // Accepted
    child3.put("", dataFileForEachLumi_[fileIndex_]);

    data.push_back(std::make_pair("", child1));
    data.push_back(std::make_pair("", child2));
    data.push_back(std::make_pair("", child3));

    pt.add_child("data", data);
    pt.put("definition", "");
    pt.put("source", "");

    std::string json_tmp = currentJsonPath + ".open";
    write_json(json_tmp, pt);
    ::rename(json_tmp.c_str(), currentJsonPath.c_str());
  }

  void DQMStreamerWriteJsonAnalyzer::writeEndJob() const {
    auto currentFileBase = str(boost::format("run%06d_ls%04d_EoR.jsn") % runNumber_ % 0);
    auto currentJsonPath = (writePath_ / currentFileBase).string();

    using namespace boost::property_tree;
    ptree pt;
    std::string json_tmp = currentJsonPath + ".open";
    write_json(json_tmp, pt);
    ::rename(json_tmp.c_str(), currentJsonPath.c_str());
  }

}  // namespace dqmservices

typedef dqmservices::DQMStreamerWriteJsonAnalyzer DQMStreamerWriteJsonAnalyzer;
DEFINE_FWK_MODULE(DQMStreamerWriteJsonAnalyzer);
