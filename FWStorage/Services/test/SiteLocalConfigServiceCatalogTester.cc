#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

namespace edmtest {
  class SiteLocalConfigServiceCatalogTester : public edm::global::EDAnalyzer<> {
  public:
    SiteLocalConfigServiceCatalogTester(const edm::ParameterSet& iPSet);

    void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override {}
  };

  SiteLocalConfigServiceCatalogTester::SiteLocalConfigServiceCatalogTester(const edm::ParameterSet& iPSet) {
    std::string const overrideCatalog;

    auto const& files = iPSet.getUntrackedParameter<std::vector<edm::ParameterSet>>("files");
    for (auto const& filePSet : files) {
      auto const& fileName = filePSet.getUntrackedParameter<std::string>("file");
      unsigned int catalogIndex = filePSet.getUntrackedParameter<unsigned int>("catalogIndex");
      auto const& expectResult = filePSet.getUntrackedParameter<std::string>("expectResult");

      edm::InputFileCatalog catalog{std::vector{fileName}, overrideCatalog};
      edm::FileCatalogItem const& item = catalog.fileCatalogItems()[0];
      if (catalogIndex >= item.fileNames().size()) {
        throw cms::Exception("Assert") << "Asked catalog " << catalogIndex << " from InputFileCatalog that had only "
                                       << item.fileNames().size() << " entries";
      }
      auto const& result = item.fileName(catalogIndex);

      if (result != expectResult) {
        throw cms::Exception("Assert") << "InputFileCatalog gave '" << result << "' for catalog " << catalogIndex
                                       << ", expected '" << expectResult << "'";
      }
    }
  }
}  // namespace edmtest

using SiteLocalConfigServiceCatalogTester = edmtest::SiteLocalConfigServiceCatalogTester;
DEFINE_FWK_MODULE(SiteLocalConfigServiceCatalogTester);
