// -*- C++ -*-
//
// Package:    CondTools/SiPixel
// Class:      SiPixelClusterShapeLimitsWriter
//
/**\class SiPixelClusterShapeLimitsWriter SiPixelClusterShapeLimitsWriter.cc CondTools/SiPixel/plugins/SiPixelClusterShapeLimitsWriter.cc
 Description: builds a SiPixelClusterShapeLimits payload (pixel cluster shape filter cut windows) from the
              ASCII files in RecoTracker/PixelLowPtUtilities/data and writes it to the conditions DB
*/

#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelClusterShapeLimits.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

class SiPixelClusterShapeLimitsWriter : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelClusterShapeLimitsWriter(const edm::ParameterSet&);
  ~SiPixelClusterShapeLimitsWriter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // parse the numbers of each non-empty, non-comment line of a file
  static std::vector<std::vector<float>> readRows(const std::string& fileName);
  static std::vector<SiPixelClusterShapeLimits::Entry> readPixelFile(const std::string& fileName);

  const std::vector<edm::ParameterSet> tables_;
  const std::vector<edm::ParameterSet> rules_;
  const std::string record_;
  const bool printDebug_;
};

SiPixelClusterShapeLimitsWriter::SiPixelClusterShapeLimitsWriter(const edm::ParameterSet& iConfig)
    : tables_(iConfig.getParameter<std::vector<edm::ParameterSet>>("tables")),
      rules_(iConfig.getParameter<std::vector<edm::ParameterSet>>("rules")),
      record_(iConfig.getParameter<std::string>("record")),
      printDebug_(iConfig.getUntrackedParameter<bool>("printDebug")) {}

std::vector<std::vector<float>> SiPixelClusterShapeLimitsWriter::readRows(const std::string& fileName) {
  std::ifstream inFile(fileName);
  if (!inFile.is_open())
    throw cms::Exception("SiPixelClusterShapeLimitsWriter") << "cannot open file " << fileName;

  std::vector<std::vector<float>> rows;
  std::string line;
  unsigned int lineNumber = 0;
  while (std::getline(inFile, line)) {
    ++lineNumber;
    std::istringstream iss(line);
    std::vector<float> row;
    std::string token;
    while (iss >> token) {
      if (token[0] == '#')
        break;
      try {
        std::size_t pos;
        row.push_back(std::stof(token, &pos));
        if (pos != token.size())
          throw std::invalid_argument(token);
      } catch (const std::exception&) {
        throw cms::Exception("SiPixelClusterShapeLimitsWriter")
            << "cannot parse '" << token << "' in " << fileName << ":" << lineNumber;
      }
    }
    if (!row.empty())
      rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<SiPixelClusterShapeLimits::Entry> SiPixelClusterShapeLimitsWriter::readPixelFile(
    const std::string& fileName) {
  // each row: part dx dy + 8 limits, optionally followed by 4 statistics columns (density points density points)
  // that are not used by the filter and are not stored
  constexpr unsigned int nKeys = 3;
  constexpr unsigned int nLimits = SiPixelClusterShapeLimits::kNLimits;

  std::vector<SiPixelClusterShapeLimits::Entry> entries;
  for (const auto& row : readRows(fileName)) {
    if (row.size() != nKeys + nLimits && row.size() != nKeys + nLimits + 4)
      throw cms::Exception("SiPixelClusterShapeLimitsWriter")
          << "unexpected number of columns (" << row.size() << ") in pixel file " << fileName;
    entries.push_back({static_cast<int>(row[0]),
                       static_cast<int>(row[1]),
                       static_cast<int>(row[2]),
                       std::vector<float>(row.begin() + nKeys, row.begin() + nKeys + nLimits)});
  }
  return entries;
}

void SiPixelClusterShapeLimitsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  SiPixelClusterShapeLimits limits;

  std::map<std::string, unsigned int> tableIndex;
  for (const auto& table : tables_) {
    const auto name = table.getParameter<std::string>("name");
    const auto file = table.getParameter<edm::FileInPath>("file").fullPath();
    if (tableIndex.count(name))
      throw cms::Exception("SiPixelClusterShapeLimitsWriter") << "table '" << name << "' defined twice";
    tableIndex[name] = limits.addTable(name, readPixelFile(file));
    edm::LogInfo("SiPixelClusterShapeLimitsWriter")
        << "table " << tableIndex[name] << " '" << name << "': " << limits.tables().back().entries.size()
        << " entries from " << file;
  }

  for (const auto& rule : rules_) {
    const auto subdet = rule.getParameter<std::string>("subdet");
    const auto table = rule.getParameter<std::string>("table");
    if (subdet != "BPix" && subdet != "FPix")
      throw cms::Exception("SiPixelClusterShapeLimitsWriter") << "unknown subdet '" << subdet << "', use BPix or FPix";
    if (!tableIndex.count(table))
      throw cms::Exception("SiPixelClusterShapeLimitsWriter") << "rule uses the undefined table '" << table << "'";
    limits.addRule({subdet == "BPix" ? SiPixelClusterShapeLimits::kBPix : SiPixelClusterShapeLimits::kFPix,
                    rule.getParameter<int>("layerOrDisk"),
                    tableIndex[table]});
  }
  limits.checkComplete();

  if (printDebug_) {
    std::ostringstream out;
    limits.printAll(out);
    edm::LogPrint("SiPixelClusterShapeLimitsWriter") << out.str();
  }

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("SiPixelClusterShapeLimitsWriter") << "PoolDBOutputService is not available";

  poolDbService->writeOneIOV(limits, poolDbService->currentTime(), record_);
}

void SiPixelClusterShapeLimitsWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Writes a SiPixelClusterShapeLimits payload from the pixel cluster shape filter ASCII files");
  edm::ParameterSetDescription table;
  table.add<std::string>("name")->setComment("name used by the rules, stored in the payload for bookkeeping");
  table.add<edm::FileInPath>("file")->setComment("ASCII file with one row per cluster size: part dx dy + 8 limits");
  desc.addVPSet("tables", table, {});

  edm::ParameterSetDescription rule;
  rule.add<std::string>("subdet")->setComment("BPix or FPix");
  rule.add<int>("layerOrDisk", SiPixelClusterShapeLimits::kAny)->setComment("BPix layer or FPix disk, 0 = any");
  rule.add<std::string>("table")->setComment("name of the table used by the modules matching this rule");
  desc.addVPSet("rules", rule, {})
      ->setComment("a module uses the table of the first matching rule; BPix and FPix need a catch-all rule");
  desc.add<std::string>("record", "SiPixelClusterShapeLimitsRcd");
  desc.addUntracked<bool>("printDebug", false);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiPixelClusterShapeLimitsWriter);
