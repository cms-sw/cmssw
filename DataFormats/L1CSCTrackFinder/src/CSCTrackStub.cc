#include <DataFormats/L1CSCTrackFinder/interface/CSCTrackStub.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

const double CSCTrackStub::thePhiBinning = CSCConstants::SECTOR_RAD/(1<<CSCBitWidths::kGlobalPhiDataBitWidth);
const double CSCTrackStub::theEtaBinning = (CSCConstants::maxEta - CSCConstants::minEta)/(CSCConstants::etaBins);

CSCTrackStub::CSCTrackStub(const CSCCorrelatedLCTDigi& aDigi,
			   const CSCDetId& aDetId): CSCCorrelatedLCTDigi(aDigi),
						    theDetId_(aDetId.rawId()),
						    thePhi_(0),
						    theEta_(0),
						    link_(0)
{}

CSCTrackStub::CSCTrackStub(const CSCCorrelatedLCTDigi& aDigi,
			   const CSCDetId& aDetId, 
			   const unsigned& phi, const unsigned& eta): CSCCorrelatedLCTDigi(aDigi),
								      theDetId_(aDetId.rawId()),
								      thePhi_(phi),
								      theEta_(eta),
								      link_(0)
{}

CSCTrackStub::CSCTrackStub(const CSCTrackStub& aTrackStub): CSCCorrelatedLCTDigi(aTrackStub),
							    theDetId_(aTrackStub.theDetId_),
							    thePhi_(aTrackStub.thePhi_),
							    theEta_(aTrackStub.theEta_),
							    link_(aTrackStub.link_)
{}

unsigned CSCTrackStub::endcap() const
{
  return CSCDetId(theDetId_).endcap();
}

unsigned CSCTrackStub::station() const
{
  return CSCDetId(theDetId_).station();
}

unsigned CSCTrackStub::sector() const
{
  return CSCTriggerNumbering::triggerSectorFromLabels(CSCDetId(theDetId_));
}

unsigned CSCTrackStub::subsector() const
{
  return CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(theDetId_));
}

unsigned CSCTrackStub::cscid() const
{
  return CSCTriggerNumbering::triggerCscIdFromLabels(CSCDetId(theDetId_));
}

bool CSCTrackStub::operator<(const CSCTrackStub& rhs) const
{
  return ( rhs.isValid() && ( (!(isValid())) || (getQuality() < rhs.getQuality()) ||
			      (getQuality() == rhs.getQuality() && cscid() < rhs.cscid()) ||
			      (getQuality() == rhs.getQuality() && cscid() == rhs.cscid() && 
			       (getTrknmb() == 2)) ) );
}

bool CSCTrackStub::operator>(const CSCTrackStub& rhs) const
{
  return ( isValid() && ( (!(rhs.isValid())) || (getQuality() > rhs.getQuality()) ||
			  (getQuality() == rhs.getQuality() && cscid() > rhs.cscid()) ||
			  (getQuality() == rhs.getQuality() && cscid() == rhs.cscid() &&
			   (getTrknmb() == 1)) ) );
}

