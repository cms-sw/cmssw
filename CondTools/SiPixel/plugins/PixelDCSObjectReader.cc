/** \class PixelDCSObjectReader
 *
 *  Class to dump Pixel DCS object from database to ROOT file.
 *
 *  $Date: 2009/05/09 22:21:34 $
 *  $Revision: 1.1 $
 *  \author Chung Khim Lae
 */

#include "TTree.h"

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

template <class Record>
class PixelDCSObjectReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
  typedef typename Record::Object Object;

public:
  PixelDCSObjectReader(const edm::ParameterSet&) : esToken(esConsumes()) {
    usesResource(TFileService::kSharedResource);
  }

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::ESGetToken<Object, Record> esToken;
};

template <class Record>
void PixelDCSObjectReader<Record>::analyze(const edm::Event&, const edm::EventSetup& setup) {
  const Object* object = &setup.getData(esToken);

  edm::Service<TFileService> fs;

  TTree* tree = fs->make<TTree>("tree", "");

  tree->Branch("object", &object);
  tree->Fill();
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/DataRecord/interface/PixelDCSRcds.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"

DEFINE_FWK_MODULE(PixelDCSObjectReader<PixelCaenChannelRcd>);
