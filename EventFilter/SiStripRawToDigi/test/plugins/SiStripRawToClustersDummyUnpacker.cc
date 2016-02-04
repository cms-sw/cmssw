#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersDummyUnpacker.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
using namespace sistrip;

SiStripRawToClustersDummyUnpacker::SiStripRawToClustersDummyUnpacker( const edm::ParameterSet& conf ) :

  siStripLazyGetter_(conf.getUntrackedParameter<edm::InputTag>("SiStripLazyGetter")), 
  siStripRefGetter_(conf.getUntrackedParameter<edm::InputTag>("SiStripRefGetter")) 
{}

SiStripRawToClustersDummyUnpacker::~SiStripRawToClustersDummyUnpacker() {}

void SiStripRawToClustersDummyUnpacker::beginJob() {}

void SiStripRawToClustersDummyUnpacker::endJob() {}

void SiStripRawToClustersDummyUnpacker::analyze( const edm::Event& event, const edm::EventSetup& setup ) {  
 
  /// Retrieve clusters from event
  edm::Handle< LazyGetter > lazygetter;
  edm::Handle< RefGetter > refgetter;
  event.getByLabel(siStripLazyGetter_,lazygetter);
  event.getByLabel(siStripRefGetter_,refgetter);
  
  /// Unpack
  RefGetter::const_iterator iregion = refgetter->begin();
  for(;iregion!=refgetter->end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = lazygetter->begin_record()+iregion->start();
    for (;icluster!=lazygetter->begin_record()+iregion->finish();icluster++) {
      icluster->geographicalId();
    }
  }
}
