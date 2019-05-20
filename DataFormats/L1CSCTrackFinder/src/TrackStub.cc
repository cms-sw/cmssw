#include <DataFormats/L1CSCTrackFinder/interface/TrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>

namespace csctf {

  const double TrackStub::thePhiBinning = CSCTFConstants::SECTOR_RAD / (1 << CSCBitWidths::kGlobalPhiDataBitWidth);
  const double TrackStub::theEtaBinning = (CSCTFConstants::maxEta - CSCTFConstants::minEta) / (CSCTFConstants::etaBins);

  TrackStub::TrackStub(const CSCCorrelatedLCTDigi& aDigi, const DetId& aDetId)
      : CSCCorrelatedLCTDigi(aDigi), theDetId_(aDetId.rawId()), thePhi_(0), theEta_(0), link_(0) {}

  TrackStub::TrackStub(const CSCCorrelatedLCTDigi& aDigi, const DetId& aDetId, const unsigned& phi, const unsigned& eta)
      : CSCCorrelatedLCTDigi(aDigi), theDetId_(aDetId.rawId()), thePhi_(phi), theEta_(eta), link_(0) {}

  TrackStub::TrackStub(const TrackStub& aTrackStub)
      : CSCCorrelatedLCTDigi(aTrackStub),
        theDetId_(aTrackStub.theDetId_),
        thePhi_(aTrackStub.thePhi_),
        theEta_(aTrackStub.theEta_),
        link_(aTrackStub.link_) {}

  unsigned TrackStub::endcap() const {
    int e = 0;

    switch (DetId(theDetId_).subdetId()) {
      case (MuonSubdetId::DT):
        e = (DTChamberId(theDetId_).wheel() > 0) ? 1 : 2;
        break;
      case (MuonSubdetId::CSC):
        e = CSCDetId(theDetId_).endcap();
        break;
      default:
        break;
    }

    return e;
  }

  unsigned TrackStub::station() const {
    int s = 0;

    switch (DetId(theDetId_).subdetId()) {
      case (MuonSubdetId::DT):
        s = DTChamberId(theDetId_).station() + 4;
        break;
      case (MuonSubdetId::CSC):
        s = CSCDetId(theDetId_).station();
        break;
      default:
        break;
    }

    return s;
  }

  unsigned TrackStub::sector() const {
    int se = 0, temps = 0;

    switch (DetId(theDetId_).subdetId()) {
      case (MuonSubdetId::DT): {
        temps = DTChamberId(theDetId_).sector();
        const unsigned int dt2csc[12] = {6, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
        se = dt2csc[temps - 1];
      } break;
      case (MuonSubdetId::CSC):
        se = CSCTriggerNumbering::triggerSectorFromLabels(CSCDetId(theDetId_));
        break;
      default:
        break;
    }

    return se;
  }

  unsigned TrackStub::subsector() const {
    int ss = 0;

    switch (DetId(theDetId_).subdetId()) {
      case (MuonSubdetId::DT): {
        ss = DTChamberId(theDetId_).sector();
        const unsigned int dt2csc_[12] = {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
        ss = dt2csc_[ss - 1];
      } break;
      case (MuonSubdetId::CSC):
        ss = CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(theDetId_));
        break;
      default:
        break;
    }

    return ss;
  }

  unsigned TrackStub::cscid() const {
    if (DetId(theDetId_).subdetId() == MuonSubdetId::CSC)
      return CSCTriggerNumbering::triggerCscIdFromLabels(CSCDetId(theDetId_));

    return 0;  // DT chambers obviously don't have a csc id :-D
  }

  unsigned TrackStub::cscidSeparateME1a() const {
    if (DetId(theDetId_).subdetId() != MuonSubdetId::CSC)
      return 0;
    CSCDetId id(theDetId_);
    unsigned normal_cscid = CSCTriggerNumbering::triggerCscIdFromLabels(id);
    if (id.station() == 1 && id.ring() == 4)
      return normal_cscid + 9;  // 10,11,12 for ME1a
    return normal_cscid;
  }

  bool TrackStub::operator<(const TrackStub& rhs) const {
    return (rhs.isValid() && ((!(isValid())) || (getQuality() < rhs.getQuality()) ||
                              (getQuality() == rhs.getQuality() && cscid() < rhs.cscid()) ||
                              (getQuality() == rhs.getQuality() && cscid() == rhs.cscid() && (getTrknmb() == 2))));
  }

  bool TrackStub::operator>(const TrackStub& rhs) const {
    return (isValid() && ((!(rhs.isValid())) || (getQuality() > rhs.getQuality()) ||
                          (getQuality() == rhs.getQuality() && cscid() > rhs.cscid()) ||
                          (getQuality() == rhs.getQuality() && cscid() == rhs.cscid() && (getTrknmb() == 1))));
  }

}  // namespace csctf
