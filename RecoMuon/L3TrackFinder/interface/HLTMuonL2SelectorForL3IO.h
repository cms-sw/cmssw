#ifndef RecoMuon_HLTMuonL2SelectorForL3IO_HLTMuonL2SelectorForL3IO_H
#define RecoMuon_HLTMuonL2SelectorForL3IO_HLTMuonL2SelectorForL3IO_H

/**  \class HLTMuonL2SelectorForL3IO
 * 
 *   L2 muon selector for L3 IO:
 *   finds L2 muons not previous converted into L3 muons
 *
 *   \author  Benjamin Radburn-Smith - Purdue University
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class HLTMuonL2SelectorForL3IO : public edm::stream::EDProducer<> {
  public:
  /// constructor with config
  HLTMuonL2SelectorForL3IO(const edm::ParameterSet&);
  
  /// destructor
  virtual ~HLTMuonL2SelectorForL3IO(); 

  /// default values
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// select muons
  virtual void produce(edm::Event&, const edm::EventSetup&);
    
 private:
	const edm::EDGetTokenT<reco::TrackCollection> l2Src_;
	const edm::EDGetTokenT<reco::TrackCollection> l3OISrc_;
	const bool useOuterHitPosition_;
	const double xDiffMax_,yDiffMax_,zDiffMax_,dRDiffMax_;
};

#endif
