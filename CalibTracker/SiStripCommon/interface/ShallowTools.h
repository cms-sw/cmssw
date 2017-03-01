#ifndef USERCODE_SHALLOWTOOLS_SHALLOWTOOLS
#define USERCODE_SHALLOWTOOLS_SHALLOWTOOLS

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class StripGeomDetUnit;
class MagneticField;
class SiStripLorentzAngle;
class Event;
namespace edm {class InputTag;}

namespace shallow {
  
typedef std::map<std::pair<uint32_t, uint16_t>, unsigned int> CLUSTERMAP;  

CLUSTERMAP make_cluster_map( const edm::Event& , const edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > & );
LocalVector drift( const StripGeomDetUnit*, const MagneticField&, const SiStripLorentzAngle&);
int findTrackIndex(const edm::Handle<edm::View<reco::Track> >& h,   const reco::Track* t);

}

#endif
