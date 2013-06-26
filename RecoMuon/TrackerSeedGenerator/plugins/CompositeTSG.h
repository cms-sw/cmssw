#ifndef RecoMuon_TrackerSeedGenerator_CompositeTSG_H
#define RecoMuon_TrackerSeedGenerator_CompositeTSG_H

/** \class CompositeTSG
 * Description:
 * TrackerSeedGenerator generic class to allow more than one TSG to be used.
 * used as a SeparatingTSG of CombinedTSG
 *
 * \author Jean-Roch Vlimant, Adam Everett
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

class TrackingRegion;
class MuonServiceProxy;
class TrackerTopology;

class CompositeTSG : public TrackerSeedGenerator {

public:
  typedef std::vector<TrajectorySeed> BTSeedCollection;  
  typedef std::pair<const Trajectory*, reco::TrackRef> TrackCand;

  CompositeTSG(const edm::ParameterSet &pset);
  virtual ~CompositeTSG();

  /// initialized the TSGs
  void init(const MuonServiceProxy *service);
  /// set the event to the TSGs
  void setEvent(const edm::Event &event);

  /// provides the seeds from the TSGs: must be overloaded
  virtual void trackerSeeds(const TrackCand&, const TrackingRegion&, const TrackerTopology *, BTSeedCollection &) =0;

 protected:

  unsigned int nTSGs() { return theTSGs.size();}
  std::vector<TrackerSeedGenerator*> theTSGs;
  std::vector<std::string> theNames;
  std::string theCategory;

  const MuonServiceProxy * theProxyService;
};


#endif 
