
#ifndef RecoMuon_TrackerSeedGenerator_DualByL2TSG_H
#define RecoMuon_TrackerSeedGenerator_DualByL2TSG_H

/** \class DualByL2TSG
 * Description:
 * SeparatingTSG (TrackerSeedGenerator) which makes a check to see if a previous seed lead to a L3 track
 *
 * \author Jean-Roch vlimant, Adam Everett
 */

#include "RecoMuon/TrackerSeedGenerator/plugins/SeparatingTSG.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class DualByL2TSG : public SeparatingTSG{
 public:
  DualByL2TSG(const edm::ParameterSet &pset);

  /// decide the TSG depending on the existence of a L3 track seeded from the L2. Return value is 0 or 1.
  unsigned int selectTSG(const TrackCand&, const TrackingRegion&);

 private:
  std::string theCategory;
  edm::InputTag theL3CollectionLabelA;
  edm::Handle<reco::TrackCollection> l3muonH;
};

#endif
