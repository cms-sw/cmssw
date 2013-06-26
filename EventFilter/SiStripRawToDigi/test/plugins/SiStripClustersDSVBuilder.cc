#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripClustersDSVBuilder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

using namespace std;
using namespace sistrip;

SiStripClustersDSVBuilder::SiStripClustersDSVBuilder( const edm::ParameterSet& conf ) :

  siStripLazyGetter_(conf.getParameter<edm::InputTag>("SiStripLazyGetter")),
  siStripRefGetter_(conf.getParameter<edm::InputTag>("SiStripRefGetter")),
  dsvnew_(conf.getUntrackedParameter<bool>("DetSetVectorNew",true))
{
  if (dsvnew_) produces< DSVnew >(); else produces< DSV >();
}

SiStripClustersDSVBuilder::~SiStripClustersDSVBuilder() 
{
}

void SiStripClustersDSVBuilder::beginJob() 
{
}

void SiStripClustersDSVBuilder::endJob() 
{
}

void SiStripClustersDSVBuilder::produce( edm::Event& event,const edm::EventSetup& setup ) {  

  /// Retrieve LazyGetter from event

  edm::Handle< LazyGetter > lazygetter;
  event.getByLabel(siStripLazyGetter_, lazygetter);

  /// Retrieve RefGetter with demand from event

  edm::Handle< RefGetter > refgetter;
  event.getByLabel(siStripRefGetter_,refgetter);

  /// create DSV product
 
  auto_ptr<DSV> dsv(new DSV());

  /// create new DSV product
 
  auto_ptr<DSVnew> dsvnew(new DSVnew());

  /// clusterize

  if (dsvnew_) clusterize(*lazygetter,*refgetter,*dsvnew); else clusterize(*lazygetter,*refgetter,*dsv);

  /// add to event

  if (dsvnew_) event.put(dsvnew); else event.put(dsv);
}

void SiStripClustersDSVBuilder::clusterize(const LazyGetter& lazygetter, const RefGetter& refgetter, DSV& dsv)
{
  RefGetter::const_iterator iregion = refgetter.begin();
  for(;iregion!=refgetter.end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = lazygetter.begin_record()+iregion->start();
    for (;icluster!=lazygetter.begin_record()+iregion->finish();icluster++) {
      DetSet& detset = dsv.find_or_insert(icluster->geographicalId());
      detset.push_back(*icluster);
    }
  }
}

void SiStripClustersDSVBuilder::clusterize(const LazyGetter& lazygetter, const RefGetter& refgetter, DSVnew& dsv)
{
  /// create filler cache

  DSVnew::FastFiller* filler = 0;

  /// fill DSVnew

  uint32_t idcache = 0;
  for(RefGetter::const_iterator iregion = refgetter.begin();iregion!=refgetter.end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = lazygetter.begin_record()+iregion->start();
    for (;icluster!=lazygetter.begin_record()+iregion->finish();icluster++) 
      {
	if (idcache!=icluster->geographicalId()) 
	  {
	    if (filler) delete filler; 
	    filler = new DSVnew::FastFiller(dsv,icluster->geographicalId());
	    idcache = icluster->geographicalId();
	  }
	
	filler->push_back(*icluster);
      }
  }
}
