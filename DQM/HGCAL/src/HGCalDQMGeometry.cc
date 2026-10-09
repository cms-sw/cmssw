#include "DQM/HGCAL/interface/HGCalDQMGeometry.h"

#include <bitset>
#include <cmath>
#include <memory>
#include <regex>
#include <sstream>
#include <utility>

#include <Eigen/Core>

#include <TDirectory.h>
#include <TFile.h>
#include <TGraph.h>
#include <TKey.h>
#include <TTree.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

namespace hgcal {
  namespace dqm {

    HGCalDQMGeometry::HGCalDQMGeometry(std::string templateDir,
                                       std::string geometryTemplate,
                                       bool skipTriggerDQM,
                                       std::string tileboardTemplateSuffix)
        : templateDir_(std::move(templateDir)),
          geometryTemplate_(std::move(geometryTemplate)),
          tileboardTemplateSuffix_(std::move(tileboardTemplateSuffix)),
          skipTriggerDQM_(skipTriggerDQM) {}

    HGCalDQMGeometry::~HGCalDQMGeometry() {
      for (auto& [path, f] : templateFiles_) {
        if (f) {
          f->Close();
          delete f;
        }
      }
      templateFiles_.clear();
    }

    // ------------------------------------------------------------------------
    //  Cache
    // ------------------------------------------------------------------------
    TFile* HGCalDQMGeometry::templateFile(std::string const& path) const {
      auto it = templateFiles_.find(path);
      if (it != templateFiles_.end())
        return it->second;
      TFile* f = TFile::Open(path.c_str(), "READ");
      if (f)
        templateFiles_[path] = f;  // only cache real handles; callers keep null checks
      return f;
    }

    std::string HGCalDQMGeometry::sipmLookupKeyFromTypecode(std::string const& typecode) {
      static std::regex rx(R"(^T(.*)_L([0-9]+)S([0-9]+)(?:_(.*))?$)");
      std::smatch m;

      std::string density, layer, sector, serial;
      if (std::regex_match(typecode, m, rx)) {
        density = m[1].str();
        layer = m[2].str();
        sector = m[3].str();
        serial = m[4].str();
      }

      bool forTestBeam2025 = serial.find("TB2025") != std::string::npos;
      return forTestBeam2025 ? "T" + density + "_L" + layer + "_" + serial : "T" + density + "_L" + layer;
    }

    TFile* HGCalDQMGeometry::moduleTemplateFile(std::string const& typecode, bool isSiPM) const {
      std::string key = isSiPM ? sipmLookupKeyFromTypecode(typecode) : typecode.substr(0, 4);
      std::string const suffix = isSiPM ? tileboardTemplateSuffix_ : "_wafer.root";
      edm::FileInPath fip(templateDir_ + "/geometry_" + key + suffix);
      return templateFile(fip.fullPath());
    }

    TFile* HGCalDQMGeometry::trigTemplateFile(std::string const& typecode, bool isSiPM) {
      std::string key = isSiPM ? sipmLookupKeyFromTypecode(typecode) : typecode.substr(0, 4);
      edm::FileInPath fip(templateDir_ + "/geometry_" + key + "_TC_wafer.root");
      return templateFile(fip.fullPath());
    }

    TGraph* HGCalDQMGeometry::moduleBin(uint32_t dqmIndex) const {
      return dqmIndex < binstates_.size() ? binstates_[dqmIndex] : nullptr;
    }

    TGraph* HGCalDQMGeometry::triggerModuleBin(uint32_t dqmIndex) const {
      return dqmIndex < trigBinstates_.size() ? trigBinstates_[dqmIndex] : nullptr;
    }

    BoundingBox const& HGCalDQMGeometry::moduleChannelScope(MonitoredElement_t const& element) const {
      auto& scope = module_channel_scopes_.at(element.dqmIndex);
      if (!scope.isValid()) {
        scope = element.isSiPM
                    ? readModuleChannelScope(moduleTemplateFile(element.typecode, true), element.irot, element.i2)
                    : BoundingBox(-14.f, 14.f, -14.f, 14.f);
      }
      return scope;
    }

    // ------------------------------------------------------------------------
    //  Template-file readers
    //  Ownership rule: file owns everything returned; callers do not delete.
    // ------------------------------------------------------------------------
    TGraph* HGCalDQMGeometry::readModuleBin(TFile* file, bool isSiPM, int plane, int u, int v) {
      std::ostringstream oss;
      oss << "/isSiPM_" << int(isSiPM) << "/plane_" << plane << "/u_" << u << "/v_" << v;
      std::string moduleDir = oss.str();

      if (!file->cd(moduleDir.c_str())) {
        edm::LogError("HGCalDQMGeometry") << "Could not cd to " << moduleDir;
        return nullptr;
      }
      TGraph* gr = dynamic_cast<TGraph*>(gDirectory->Get("module_bin"));
      if (!gr) {
        edm::LogError("HGCalDQMGeometry") << "Bin not found in " << moduleDir;
        return nullptr;
      }
      file->cd("/");
      return gr;
    }

    std::vector<double> HGCalDQMGeometry::readModuleCenter(TFile* file, bool isSiPM, int plane, int u, int v) {
      std::vector<double> x0y0(2, 0);
      std::ostringstream oss;
      oss << "/isSiPM_" << int(isSiPM) << "/plane_" << plane << "/u_" << u << "/v_" << v;
      std::string moduleDir = oss.str();

      if (!file->cd(moduleDir.c_str())) {
        edm::LogError("HGCalDQMGeometry") << "Could not cd to " << moduleDir;
        return x0y0;
      }
      TTree* tree = dynamic_cast<TTree*>(gDirectory->Get("module_properties"));
      if (!tree) {
        edm::LogError("HGCalDQMGeometry") << "TTree not found in " << moduleDir;
        return x0y0;
      }
      double x0, y0;
      tree->SetBranchAddress("x0", &x0);
      tree->SetBranchAddress("y0", &y0);
      if (!(tree->GetEntries() > 0))
        edm::LogError("HGCalDQMGeometry") << "TTree empty in " << moduleDir;
      tree->GetEntry(0);
      x0y0[0] = x0;
      x0y0[1] = y0;
      gDirectory->cd("/");
      return x0y0;
    }

    BoundingBox HGCalDQMGeometry::readModuleScope(TFile* file, bool isSiPM, int plane, int u, int v) {
      BoundingBox box(0, 0, 0, 0);
      std::ostringstream oss;
      oss << "/isSiPM_" << int(isSiPM) << "/plane_" << plane << "/u_" << u << "/v_" << v;
      std::string moduleDir = oss.str();

      if (!file->cd(moduleDir.c_str())) {
        edm::LogError("HGCalDQMGeometry") << "Could not cd to " << moduleDir;
        return box;
      }
      TTree* tree = dynamic_cast<TTree*>(gDirectory->Get("module_properties"));
      if (!tree) {
        edm::LogError("HGCalDQMGeometry") << "TTree not found in " << moduleDir;
        return box;
      }
      double xmin, xmax, ymin, ymax;
      tree->SetBranchAddress("xmin", &xmin);
      tree->SetBranchAddress("xmax", &xmax);
      tree->SetBranchAddress("ymin", &ymin);
      tree->SetBranchAddress("ymax", &ymax);
      if (!(tree->GetEntries() > 0))
        edm::LogError("HGCalDQMGeometry") << "TTree empty in " << moduleDir;
      tree->GetEntry(0);
      box = BoundingBox(
          static_cast<float>(xmin), static_cast<float>(xmax), static_cast<float>(ymin), static_cast<float>(ymax));
      gDirectory->cd("/");
      return box;
    }

    BoundingBox HGCalDQMGeometry::readModuleChannelScope(TFile* file, char irot, int v) const {
      BoundingBox box;
      if (!file) {
        edm::LogError("HGCalDQMGeometry") << "Could not read channel scope from a null template file";
        return box;
      }

      TIter nextkey(file->GetListOfKeys());
      double const angleOffset = M_PI / 36. - M_PI / 2. + double(v) * M_PI / 18.;
      while (auto* key = static_cast<TKey*>(nextkey())) {
        std::unique_ptr<TObject> object(key->ReadObj());
        if (!object || !object->InheritsFrom("TGraph"))
          continue;

        auto* graph = static_cast<TGraph*>(object.get());
        if (graph->GetN() <= 1)
          continue;
        rotateShape(graph, irot, angleOffset);
        for (int point = 0; point < graph->GetN(); ++point) {
          box.xmin = std::min(box.xmin, float(graph->GetX()[point]));
          box.xmax = std::max(box.xmax, float(graph->GetX()[point]));
          box.ymin = std::min(box.ymin, float(graph->GetY()[point]));
          box.ymax = std::max(box.ymax, float(graph->GetY()[point]));
        }
      }
      return box;
    }

    // ------------------------------------------------------------------------
    //  Static geometry utilities
    // ------------------------------------------------------------------------
    void HGCalDQMGeometry::rotateShape(TGraph* gr, char irot, double offset) {
      // -90 deg shift puts the template in the standard position; see
      //   https://gitlab.cern.ch/hgcal-integration/hgcal_modmap/-/tree/main
      float angle(irot * M_PI / 3. + offset);
      Eigen::Matrix2d R;
      R << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
      for (int i = 0; i < gr->GetN(); i++) {
        Eigen::Vector2d v(gr->GetX()[i], gr->GetY()[i]);
        Eigen::Vector2d rv = R * v;
        gr->SetPoint(i, rv.x(), rv.y());
      }
    }

    void HGCalDQMGeometry::translateBin(TGraph* gr, float x0, float y0) {
      for (int i = 0; i < gr->GetN(); i++)
        gr->SetPoint(i, gr->GetX()[i] + x0, gr->GetY()[i] + y0);
    }

    // ------------------------------------------------------------------------
    //  build(): module positions from the mapping and templates, then plot ranges.
    // ------------------------------------------------------------------------
    void HGCalDQMGeometry::build(
        edm::EventSetup const& iSetup,
        edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> const& moduleIdxTkn,
        edm::ESGetToken<HGCalMappingModuleIndexerTrigger, HGCalElectronicsMappingRcd> const& moduleIdxTriggerTkn,
        edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> const& moduleInfoTkn,
        edm::ESGetToken<hgcal::HGCalMappingModuleTriggerParamHost, HGCalElectronicsMappingRcd> const&
            moduleInfoTriggerTkn,
        edm::ESGetToken<HGCalConfiguration, HGCalModuleConfigurationRcd> const& moduleConfigTkn) {
      auto const& moduleIndexer = iSetup.getData(moduleIdxTkn);

      size_t ntypecodes = moduleIndexer.typecodeMap().size();
      typecodes_.resize(ntypecodes, "");
      directionallayerstates_.resize(typecodes_.size(), 0);
      cassettestates_.resize(ntypecodes, 0);
      module_scopes_.resize(ntypecodes);
      module_channel_scopes_.resize(ntypecodes);
      binstates_.resize(ntypecodes, nullptr);

      auto const& moduleInfo = iSetup.getData(moduleInfoTkn);
      auto const& moduleConfig = iSetup.getData(moduleConfigTkn);

      edm::FileInPath fiptemp(templateDir_ + geometryTemplate_);
      TFile* file = templateFile(fiptemp.fullPath());

      flag_tileboard_exists_ = false;

      for (const auto& it : moduleIndexer.typecodeMap()) {
        uint32_t fedid = it.second.first;
        uint32_t imod = it.second.second;
        uint32_t dqmIndex = moduleIndexer.getIndexForModule(fedid, imod);

        std::string typecode = it.first;
        std::replace(typecode.begin(), typecode.end(), '-', '_');
        typecodes_[dqmIndex] = typecode;

        auto modInfo = moduleInfo.view()[dqmIndex];
        std::bitset<16> enabledErx((uint16_t)moduleConfig.feds[fedid].econds[imod].enabledErx);

        module_scopes_[dqmIndex] = readModuleScope(file, modInfo.isSiPM(), modInfo.plane(), modInfo.i1(), modInfo.i2());
        binstates_[dqmIndex] = readModuleBin(file, modInfo.isSiPM(), modInfo.plane(), modInfo.i1(), modInfo.i2());
        cassettestates_[dqmIndex] = modInfo.cassette();

        directionallayerstates_[dqmIndex] = modInfo.zside() ? modInfo.plane() : -modInfo.plane();
        cassettesPerLayer_[directionallayerstates_[dqmIndex]].insert(cassettestates_[dqmIndex]);
        unique_directionallayers_.insert(directionallayerstates_[dqmIndex]);

        MonitoredElement_t ele;
        ele.typecode = typecode;
        ele.nErx = enabledErx.count();
        ele.zside = modInfo.zside();
        ele.endcap = ele.zside ? 1 : -1;
        ele.isSiPM = modInfo.isSiPM();
        ele.irot = modInfo.irot();
        ele.plane = modInfo.plane();
        ele.layer = modInfo.plane() * ele.endcap;
        ele.i1 = modInfo.i1();
        ele.i2 = modInfo.i2();
        ele.x0y0 = readModuleCenter(file, ele.isSiPM, ele.plane, ele.i1, ele.i2);
        ele.fedid = fedid;
        ele.modid = imod;
        ele.econdidx = modInfo.econdidx();
        ele.cassette = modInfo.cassette();
        ele.dqmIndex = dqmIndex;

        if (ele.isSiPM)
          flag_tileboard_exists_ = true;

        auto& cassetteMap = HGCALMap_[ele.endcap][ele.layer][ele.cassette];
        cassetteMap[typecode] = ele;
        cassetteMap[typecode].moduleIndex = cassetteMap.size() - 1;
      }

      if (!skipTriggerDQM_) {
        HGCalMappingModuleIndexerTrigger const& moduleIndexerTrigger = iSetup.getData(moduleIdxTriggerTkn);
        auto const& moduleInfoTrigger = iSetup.getData(moduleInfoTriggerTkn);

        trigBinstates_.resize(moduleIndexerTrigger.typecodeMap().size(), nullptr);

        for (const auto& it : moduleIndexerTrigger.typecodeMap()) {
          std::string typecode = it.first;
          std::replace(typecode.begin(), typecode.end(), '-', '_');

          uint32_t fedid = it.second.first;
          uint32_t imod = it.second.second;
          uint32_t dqmIndex = moduleIndexerTrigger.getIndexForModule(fedid, imod);

          auto modInfo = moduleInfoTrigger.view()[dqmIndex];

          TriggerMonitoredElement_t trigEle;
          trigEle.typecode = typecode;
          trigEle.zside = modInfo.zside();
          trigEle.isSiPM = modInfo.isSiPM();
          trigEle.irot = modInfo.irot();
          trigEle.plane = modInfo.plane();
          trigEle.i1 = modInfo.i1();
          trigEle.i2 = modInfo.i2();
          trigEle.nTrLinks = moduleIndexerTrigger.getNumTrLinks(fedid, imod);
          trigEle.nTrCells = moduleIndexerTrigger.getNumChannels(fedid, imod);
          trigEle.fedid = fedid;
          trigEle.modid = imod;
          trigEle.dqmIndex = dqmIndex;
          trigEle.econtidx = modInfo.econtidx();
          trigEle.cassette = modInfo.cassette();
          trigEle.endcap = trigEle.zside ? 1 : -1;
          trigEle.layer = trigEle.plane * trigEle.endcap;
          trigEle.x0y0 = readModuleCenter(file, trigEle.isSiPM, trigEle.plane, trigEle.i1, trigEle.i2);
          trigBinstates_[dqmIndex] = readModuleBin(file, trigEle.isSiPM, trigEle.plane, trigEle.i1, trigEle.i2);

          MonitoredElementKey_t key(fedid, imod);
          TrigHGCALMap_[trigEle.endcap][trigEle.plane][trigEle.cassette].insert(key);
          TriggerModuleMap_[key] = trigEle;
        }
      }

      // Do not close file — TGraphs in binstates_ / trigBinstates_ point into
      // it and must stay valid for the job. Cache closes it in the dtor.

      nLayers_ = unique_directionallayers_.size();

      calculatePlotsCorners();
    }

    void HGCalDQMGeometry::calculatePlotsCorners() {
      constexpr float INIT_MIN = 1000.0f;
      constexpr float INIT_MAX = -1000.0f;
      constexpr float MARGIN = 20.0f;

      for (int layer : unique_directionallayers_) {
        corners_layer_[layer] = {{INIT_MIN, INIT_MAX, INIT_MIN, INIT_MAX}};
        for (int cassette : cassettesPerLayer_[layer])
          corners_cassette_[layer][cassette] = {{INIT_MIN, INIT_MAX, INIT_MIN, INIT_MAX}};
      }

      for (size_t i = 0; i < typecodes_.size(); i++) {
        int layer = directionallayerstates_[i];
        int cassette = cassettestates_[i];
        BoundingBox const& bbox = module_scopes_[i];

        auto& cornersC = corners_cassette_[layer][cassette];
        auto& cornersL = corners_layer_[layer];

        cornersC[0] = std::min(cornersC[0], bbox.xmin);
        cornersC[1] = std::max(cornersC[1], bbox.xmax);
        cornersC[2] = std::min(cornersC[2], bbox.ymin);
        cornersC[3] = std::max(cornersC[3], bbox.ymax);

        cornersL[0] = std::min(cornersL[0], bbox.xmin);
        cornersL[1] = std::max(cornersL[1], bbox.xmax);
        cornersL[2] = std::min(cornersL[2], bbox.ymin);
        cornersL[3] = std::max(cornersL[3], bbox.ymax);
      }

      for (int layer : unique_directionallayers_) {
        auto& cornersL = corners_layer_[layer];
        cornersL[0] -= MARGIN;
        cornersL[1] += MARGIN;
        cornersL[2] -= MARGIN;
        cornersL[3] += MARGIN;

        for (int cassette : cassettesPerLayer_[layer]) {
          auto& cornersC = corners_cassette_[layer][cassette];
          cornersC[0] -= MARGIN;
          cornersC[1] += MARGIN;
          cornersC[2] -= MARGIN;
          cornersC[3] += MARGIN;
        }
      }
    }

  }  // namespace dqm
}  // namespace hgcal
