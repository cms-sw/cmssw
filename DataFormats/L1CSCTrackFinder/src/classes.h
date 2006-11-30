#include <vector>
#include <DataFormats/L1CSCTrackFinder/interface/L1TrackId.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/Common/interface/Wrapper.h>

namespace
{
  namespace
    {
      csc::L1Track cL1TRK;
      csc::L1TrackId cL1TRKID;
      CSCTrackStub cTrkStb;
      
      std::pair<csc::L1Track,MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> > vL1Trk;

      std::vector<csc::L1Track> vL1TRK;
      std::vector<csc::L1TrackId> vL1TRKID;
      std::vector<std::pair<csc::L1Track,MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> > > vL1TrkColl;
      std::vector<CSCTrackStub> vTrkStb;
      CSCTriggerContainer<CSCTrackStub> tcTrkStb;

      edm::Wrapper<std::vector<csc::L1Track> > wL1TRK;
      edm::Wrapper<std::vector<std::pair<csc::L1Track,MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> > > > wL1TrkColl;
      edm::Wrapper<CSCTriggerContainer<CSCTrackStub> > wTrkStb;
    }
}
