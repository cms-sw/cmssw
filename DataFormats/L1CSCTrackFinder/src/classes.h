#include <vector>
#include <DataFormats/L1CSCTrackFinder/interface/L1CSCSPStatusDigi.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1TrackId.h>
#include <DataFormats/L1CSCTrackFinder/interface/L1Track.h>
#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h>
#include <DataFormats/Common/interface/Wrapper.h>

namespace {
  struct dictionary {
      csc::L1Track cL1TRK;
      csc::L1TrackId cL1TRKID;
      csctf::TrackStub cTrkStb;
      L1CSCSPStatusDigi cL1CSCstatus; // This is not a template and we don't need to instantiate it for generating dictionary, but we do

      std::pair<csc::L1Track,MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> > vL1Trk;

      std::vector<L1CSCSPStatusDigi> vL1CSCstatus;
      std::pair<int,std::vector<L1CSCSPStatusDigi> > pvL1CSCstatus;

      std::vector<csc::L1Track> vL1TRK;
      std::vector<csc::L1TrackId> vL1TRKID;
      std::vector<std::pair<csc::L1Track,MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> > > vL1TrkColl;
      std::vector<csctf::TrackStub> vTrkStb;
      CSCTriggerContainer<csctf::TrackStub> tcTrkStb;

      edm::Wrapper<std::vector<csc::L1Track> > wL1TRK;
      edm::Wrapper<std::vector<std::pair<csc::L1Track,MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi> > > > wL1TrkColl;
      edm::Wrapper<CSCTriggerContainer<csctf::TrackStub> > wTrkStb;

      edm::Wrapper<std::pair<int,std::vector<L1CSCSPStatusDigi> > > wL1CSCstatus;
  };
}
