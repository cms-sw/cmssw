#ifndef RecoTauTag_RecoTau_PFRecoTauAlgorithmBase
#define RecoTauTag_RecoTau_PFRecoTauAlgorithmBase

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauTagInfoFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

class PFRecoTauAlgorithmBase
{
 public:
  PFRecoTauAlgorithmBase();
  PFRecoTauAlgorithmBase(const edm::ParameterSet&);

  virtual ~PFRecoTauAlgorithmBase();

  void setTransientTrackBuilder(const TransientTrackBuilder*);
  //Add other common methods

  //BASIC method
  virtual reco::PFTau buildPFTau(const reco::PFTauTagInfoRef&,const reco::Vertex&) = 0;

 protected:
  const TransientTrackBuilder *TransientTrackBuilder_;
  //Add other common members

};
#endif
