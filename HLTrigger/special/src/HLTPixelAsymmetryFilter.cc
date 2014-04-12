///////////////////////////////////////////////////////
//
// HLTPixelAsymmetryFilter
//
// See header file for infos
//
// S.Viret: 12/01/2011 (viret@in2p3.fr)
//
///////////////////////////////////////////////////////

#include "HLTrigger/special/interface/HLTPixelAsymmetryFilter.h"

//
// constructors and destructor
//

HLTPixelAsymmetryFilter::HLTPixelAsymmetryFilter(const edm::ParameterSet& config) : HLTFilter(config),
  inputTag_ (config.getParameter<edm::InputTag>("inputTag")),
  min_asym_ (config.getParameter<double>("MinAsym")),
  max_asym_ (config.getParameter<double>("MaxAsym")),
  clus_thresh_ (config.getParameter<double>("MinCharge")),
  bmincharge_ (config.getParameter<double>("MinBarrel"))
{
  inputToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(inputTag_);
  LogDebug("") << "Using the " << inputTag_ << " input collection";
  LogDebug("") << "Requesting events with a charge repartition asymmetry between " << min_asym_ << " and " << max_asym_;
  LogDebug("") << "Mean cluster charge in the barrel should be higher than" << bmincharge_ << " electrons ";
  LogDebug("") << "Only clusters with a charge larger than " << clus_thresh_ << " electrons will be used ";
}

HLTPixelAsymmetryFilter::~HLTPixelAsymmetryFilter()
{
}

void
HLTPixelAsymmetryFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltSiPixelClusters"));
  desc.add<double>("MinAsym",0.);        // minimum asymmetry
  desc.add<double>("MaxAsym",1.);        // maximum asymmetry
  desc.add<double>("MinCharge",4000.);   // minimum charge for a cluster to be selected (in e-)
  desc.add<double>("MinBarrel",10000.);  // minimum average charge in the barrel (bpix, in e-)
  descriptions.add("hltPixelAsymmetryFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTPixelAsymmetryFilter::hltFilter(edm::Event& event, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputTag_);

  // get hold of products from Event
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > clusterColl;
  event.getByToken(inputToken_, clusterColl);

  unsigned int clusterSize = clusterColl->dataSize();
  LogDebug("") << "Number of clusters accepted: " << clusterSize;

  bool accept = (clusterSize >= 2); // Not necessary to go further in this case

  if (!accept) return accept;

  double asym_pix_1  = -1;
  double asym_pix_2  = -1;

  int n_clus[3]   = {0,0,0};
  double e_clus[3] = {0.,0.,0.};

  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=clusterColl->begin(); DSViter!=clusterColl->end();DSViter++ )
  {
    edmNew::DetSet<SiPixelCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiPixelCluster>::const_iterator end  =DSViter->end();
    uint32_t detid = DSViter->id();

    bool barrel = DetId(detid).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
    bool endcap = DetId(detid).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);

    int detpos = -1;

    // First we look if we are in the endcap or in the barrel PIXELS
    // Asymmetry is computed with endcap pixels only

    if (endcap)
    {
      PixelEndcapName::HalfCylinder position = PixelEndcapName(DetId(detid)).halfCylinder();

      if (position == PixelEndcapName::mI || position == PixelEndcapName::mO) // FPIX-
	detpos = 0;

      if (position == PixelEndcapName::pI || position == PixelEndcapName::pO) // FPIX+
	detpos = 2;
    }

    if (barrel)
      detpos = 1;

    if (detpos<0) continue;

    for(edmNew::DetSet<SiPixelCluster>::const_iterator iter=begin;iter!=end;++iter)
    {
      if (iter->charge()>clus_thresh_ )
      {
	++n_clus[detpos];
	e_clus[detpos]+=iter->charge();
      }
    }
  } // End of cluster loop

  for (int i=0;i<3;++i)
  {
    if (n_clus[i])
      e_clus[i] /= n_clus[i];
  }

  if (e_clus[1] < bmincharge_) return false; // Reject event if the Barrel mean charge is too low


  if (e_clus[0]+e_clus[2] != 0) // Compute asyms, if applicable
  {
    asym_pix_1 = e_clus[0]/(e_clus[0]+e_clus[2]);
    asym_pix_2 = e_clus[2]/(e_clus[0]+e_clus[2]);
  }
  else  // Otherwise reject event
  {
    return false;
  }

  bool pkam_1 = (asym_pix_1 <= max_asym_ && asym_pix_1 >= min_asym_);
  bool pkam_2 = (asym_pix_2 <= max_asym_ && asym_pix_2 >= min_asym_);

  if (pkam_1 || pkam_2) return accept; // Final selection

  return false;
}
