///////////////////////////////////////////////////////
//
// HLTTrackerHaloFilter
//
// See header file for infos on input parameters
// Comments on the code flow are in the cc file
//
// S.Viret: 01/03/2011 (viret@in2p3.fr)
//
///////////////////////////////////////////////////////

#include "HLTrigger/special/interface/HLTTrackerHaloFilter.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

typedef edmNew::DetSetVector<SiStripCluster>::Getter Getter;
//
// constructors and destructor
//

HLTTrackerHaloFilter::HLTTrackerHaloFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_     (config.getParameter<edm::InputTag>("inputTag")),
  max_clusTp_   (config.getParameter<int>("MaxClustersTECp")),
  max_clusTm_   (config.getParameter<int>("MaxClustersTECm")),
  sign_accu_    (config.getParameter<int>("SignalAccumulation")),
  max_clusT_    (config.getParameter<int>("MaxClustersTEC")),
  max_back_     (config.getParameter<int>("MaxAccus")),
  fastproc_     (config.getParameter<int>("FastProcessing"))
{
  clusterInputToken_ = consumes<edmNew::DetSetVector<SiStripCluster> >(inputTag_);
}

HLTTrackerHaloFilter::~HLTTrackerHaloFilter()
{
}

void
HLTTrackerHaloFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltSiStripClusters"));
  desc.add<int>("MaxClustersTECp",50);
  desc.add<int>("MaxClustersTECm",50);
  desc.add<int>("SignalAccumulation",5);
  desc.add<int>("MaxClustersTEC",60);
  desc.add<int>("MaxAccus",4);
  desc.add<int>("FastProcessing",1);
  descriptions.add("hltTrackerHaloFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTTrackerHaloFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
/*
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
  event.getByToken(clusterInputToken_, stripClusters);


  /// First initialize some variables
  int SST_clus_MAP_m[5][8][9]; memset(SST_clus_MAP_m,  0x00, sizeof(SST_clus_MAP_m));
  int SST_clus_MAP_p[5][8][9]; memset(SST_clus_MAP_p,  0x00, sizeof(SST_clus_MAP_p));
  int SST_clus_PROJ_m[5][8];   memset(SST_clus_PROJ_m, 0x00, sizeof(SST_clus_PROJ_m));
  int SST_clus_PROJ_p[5][8];   memset(SST_clus_PROJ_p, 0x00, sizeof(SST_clus_PROJ_p));

  int n_total_clus  = 0;
  int n_total_clusp = 0;
  int n_total_clusm = 0;

  int maxm          = 0;
  int maxp          = 0;

  int npeakm        = 0;
  int npeakp        = 0;

  for (auto di = stripClusters->begin(false), de=stripClusters->end(false); di!=de; ++di) {

    // Don't go further if one of the TEC cluster cut is not passed
    if (n_total_clus>max_clusT_)   return false;
    if (n_total_clusp>max_clusTp_) return false;
    if (n_total_clusm>max_clusTm_) return false;


    // Some cuts applied if fast processing requested
    if (fastproc_ && maxm<sign_accu_) return false;
    
    //    auto ds = *di;
    DetId id = di->id();
    uint32_t subdet = id.subdetId();

    // Look at the DetId, as we perform the quest only in TEC
    if ( subdet != SiStripDetId::TEC ) continue;
    if ( id%2 == 1 ) continue;
    if ( tTopo->tecRing(id)<3 || tTopo->tecIsStereo(id) ) continue;

    if ( !di->isValid() ) {
      auto dst = stripClusters->find(id,true);
      //      std::cout << "isValid: " << dst->isValid() << std::endl;
      if ( !dst->isValid() ) continue; // not umpacked
      if ( dst->empty()   ) continue;
      //    auto s1 = dst->size();
      //    std::cout << "s1: " << s1 << std::endl;
    }
    ++n_total_clus;

    int r_id = tTopo->tecRing(id)-3;
    int p_id = tTopo->tecPetalNumber(id)-1;
    int w_id = tTopo->tecWheel(id)-1;
    
    // Then we do accumulations and cuts 'on the fly'
    if ( tTopo->tecSide(id)==1 ) // Minus side (BEAM2)
      {
	++n_total_clusm;
	++SST_clus_MAP_m[r_id][p_id][w_id];	
	++SST_clus_PROJ_m[r_id][p_id]; // Accumulation
	
	if (SST_clus_PROJ_m[r_id][p_id]>maxm) maxm = SST_clus_PROJ_m[r_id][p_id];
	if (SST_clus_PROJ_m[r_id][p_id]==sign_accu_) ++npeakm;
	
	if (npeakm>=max_back_) return false; // Too many accumulations (PKAM)
      }
    else // Plus side (BEAM1)
      {
	++n_total_clusp;
	if (!SST_clus_MAP_p[r_id][p_id][w_id])
	{
	  ++SST_clus_MAP_p[r_id][p_id][w_id];	
	  ++SST_clus_PROJ_p[r_id][p_id];

	  if (SST_clus_PROJ_p[r_id][p_id]>maxp) maxp = SST_clus_PROJ_p[r_id][p_id];
	  if (SST_clus_PROJ_p[r_id][p_id]==sign_accu_) ++npeakp;

	  if (npeakp>=max_back_) return false;
	}
      }

  }

  // The final selection is applied here
  // Most of the cuts have already been applied tough

  if (n_total_clus>max_clusT_)                return false;
  if (n_total_clusp>max_clusTp_)              return false;
  if (n_total_clusm>max_clusTm_)              return false;
  if (n_total_clusp<sign_accu_)               return false;
  if (n_total_clusm<sign_accu_)               return false;
  if (maxm<sign_accu_ || maxp<sign_accu_)     return false;
  if (npeakm>=max_back_ || npeakp>=max_back_) return false;
*/
  return true;
}
