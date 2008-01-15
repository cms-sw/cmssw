#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripClustersDSVBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
using namespace sistrip;

SiStripClustersDSVBuilder::SiStripClustersDSVBuilder( const edm::ParameterSet& conf ) :

  siStripLazyGetter_(conf.getParameter<edm::InputTag>("SiStripLazyGetter")),
  siStripRefGetter_(conf.getParameter<edm::InputTag>("SiStripRefGetter"))
{
  produces< DSV >();
}

SiStripClustersDSVBuilder::~SiStripClustersDSVBuilder() {}

void SiStripClustersDSVBuilder::beginJob( const edm::EventSetup& setup) {}

void SiStripClustersDSVBuilder::endJob() {}

void SiStripClustersDSVBuilder::produce( edm::Event& event,const edm::EventSetup& setup ) {  
 
  /// Retrieve RefGetter with demand from event

  edm::Handle< LazyGetter > lazygetter;
  edm::Handle< RefGetter > refgetter;
  event.getByLabel(siStripLazyGetter_, lazygetter);
  event.getByLabel(siStripRefGetter_,refgetter);
 
  /// Convert

  auto_ptr<DSV> dsv(new DSV);
  RefGetter::const_iterator iregion = refgetter->begin();
  for(;iregion!=refgetter->end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = lazygetter->begin_record()+iregion->start();
    for (;icluster!=lazygetter->begin_record()+iregion->finish();icluster++) {
      DetSet& detset = dsv->find_or_insert(icluster->geographicalId());
      detset.push_back(*icluster);
    }
  }

  /// add to event
  event.put(dsv);
}
