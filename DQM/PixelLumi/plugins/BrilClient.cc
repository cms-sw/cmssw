#include "BrilClient.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <vector>

#include "TH2F.h"

BrilClient::BrilClient(const edm::ParameterSet& ps) {
  pathToken_ = consumes<std::string, edm::InLumi>(
      edm::InputTag("source", "sourceDataPath"));
  jsonToken_ = consumes<std::string, edm::InLumi>(
      edm::InputTag("source", "sourceJsonPath"));
}

BrilClient::~BrilClient() {}

using boost::property_tree::ptree;

template <typename T>
std::vector<T> as_vector(ptree const& pt, ptree::key_type const& key) {
  std::vector<T> r;
  for (auto& item : pt.get_child(key)) r.push_back(item.second.get_value<T>());
  return r;
}

void BrilClient::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker_,
                                       DQMStore::IGetter& igetter_,
                                       edm::LuminosityBlock const& lb,
                                       edm::EventSetup const&) {
  edm::Handle<std::string> filePath_;
  lb.getByToken(pathToken_, filePath_);

  // edm::Handle<std::string> jsonPath_;
  // lb.getByToken(jsonToken_, jsonPath_);
  //

  ptree json;
  if (!boost::filesystem::exists(*filePath_)) {
    edm::LogWarning("BrilClient") << "BrilClient"
                                  << " File missing: " << *filePath_
                                  << std::endl;

    return;
  } else {
    edm::LogWarning("BrilClient") << "BrilClient"
                                  << " Opening: " << *filePath_ << std::endl;

    read_json(std::string(*filePath_), json);
  }

  // Parse the json
  for (auto& mainTree : json.get_child("OccupancyPlots")) {
    std::string title = mainTree.second.get<std::string>("titles");
    std::size_t pos = title.find(",");
    if (pos <= 0) {
      edm::LogWarning("BrilClient") << "BrilClient::dqmEndLuminosityBlock"
                                    << " Invalid title" << title << std::endl;

      continue;
    }
    std::string name = title.substr(0, pos);

    auto nBins = as_vector<int>(mainTree.second, "nbins");    // x, y
    auto xrange = as_vector<int>(mainTree.second, "xrange");  // min, max
    auto yrange = as_vector<int>(mainTree.second, "yrange");  // min, max

    TH2F* th = new TH2F(name.c_str(), title.c_str(), nBins.at(0), xrange.at(0),
                        xrange.at(1), nBins.at(1), yrange.at(0), yrange.at(1));

    for (auto& dataArray : mainTree.second.get_child("data")) {
      int elements[3] = {0, 0, 0};  // binX, binY, binCont;
      auto element = std::begin(elements);

      for (auto& binContent : dataArray.second) {
        *element++ = stoi(binContent.second.get_value<std::string>());
        if (element == std::end(elements)) break;
      }

      th->SetBinContent(elements[0], elements[1], elements[2]);
    }

    // Add it to the DQM store
    ibooker_.setCurrentFolder("BRIL/OccupancyPlots");
    igetter_.setCurrentFolder("BRIL/OccupancyPlots");

    MonitorElement* m = igetter_.get(name);
    if (m == nullptr) {
      m = ibooker_.book2D(name, th);
    } else {
      m->getTH1F()->Add(th);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BrilClient);
