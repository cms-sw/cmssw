#ifndef jhugonLCTStudies_RunSRLUTs_h
#define jhugonLCTStudies_RunSRLUTs_h

#include <string>

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

namespace csctf_analysis
{
typedef std::pair<csc::L1Track, std::vector<csctf::TrackStub> > TrackAndAssociatedStubs;
typedef std::vector< TrackAndAssociatedStubs > TrackAndAssociatedStubsCollection;

class RunSRLUTs
{
 public:

  RunSRLUTs();

  virtual ~RunSRLUTs();

  void run(std::vector<csctf::TrackStub> *stub_list);

  void makeTrackStubs(const CSCCorrelatedLCTDigiCollection * inClcts,std::vector<csctf::TrackStub> *outStubVec);

  void makeAssociatedTrackStubs(const L1CSCTrackCollection * inTrackColl,TrackAndAssociatedStubsCollection *outTrkStubCol);

 private:

  CSCSectorReceiverLUT *srLUTs_[5];
};
}
#endif
