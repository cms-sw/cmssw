#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTTrackCand.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h>
#include <DataFormats/Common/interface/Wrapper.h>

namespace{ 
  struct dictionary {
    L1MuDTChambPhDigi ph_S;
    L1MuDTChambThDigi th_S;
    L1MuDTTrackCand   tr_S;

    std::vector<L1MuDTChambPhDigi> ph_V;
    std::vector<L1MuDTChambThDigi> th_V;
    std::vector<L1MuDTTrackCand>   tr_V;

    L1MuDTChambPhContainer ph_K;
    L1MuDTChambThContainer th_K;
    L1MuDTTrackContainer   tr_K;

    edm::Wrapper<L1MuDTChambPhContainer> ph_W;
    edm::Wrapper<L1MuDTChambThContainer> th_W;
    edm::Wrapper<L1MuDTTrackContainer>   tr_W;
  };
}
