#include <L1Trigger/CSCCommonTrigger/interface/CSCTrackStub.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCBitWidths.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

double CSCTrackStub::thePhiBinning = CSCConstants::SECTOR_RAD/(1<<CSCBitWidths::kGlobalPhiDataBitWidth);
double CSCTrackStub::theEtaBinning = (CSCConstants::maxEta - CSCConstants::minEta)/CSCConstants::etaBins;

CSCTrackStub::CSCTrackStub(const CSCCorrelatedLCTDigi& aDigi,
			   const CSCDetId& aDetId): theDetId_(aDetId),
						    theDigi_(aDigi),
						    thePhi_(0),
						    theEta_(0)
{}

CSCTrackStub::CSCTrackStub(const CSCCorrelatedLCTDigi& aDigi,
			   const CSCDetId& aDetId, 
			   const unsigned& phi, const unsigned& eta): theDetId_(aDetId),
								      theDigi_(aDigi),
								      thePhi_(phi),
								      theEta_(eta)
{}

CSCTrackStub::CSCTrackStub(const CSCTrackStub& aTrackStub): theDetId_(aTrackStub.theDetId_),
							    theDigi_(aTrackStub.theDigi_),
							    thePhi_(aTrackStub.thePhi_),
							    theEta_(aTrackStub.theEta_)
{}

unsigned CSCTrackStub::endcap() const
{
  return theDetId_.endcap();
}

unsigned CSCTrackStub::station() const
{
  return theDetId_.station();
}

unsigned CSCTrackStub::sector() const
{
  return CSCTriggerNumbering::triggerSectorFromLabels(theDetId_);
}

unsigned CSCTrackStub::subsector() const
{
  return CSCTriggerNumbering::triggerSubSectorFromLabels(theDetId_);
}

unsigned CSCTrackStub::cscid() const
{
  return CSCTriggerNumbering::triggerCscIdFromLabels(theDetId_);
}

bool CSCTrackStub::operator<(const CSCTrackStub& rhs) const
{
  return ( rhs.isValid() && ( (!(isValid())) || (getQuality() < rhs.getQuality()) ||
			      (getQuality() == rhs.getQuality() && cscid() < rhs.cscid()) ||
			      (getQuality() == rhs.getQuality() && cscid() == rhs.cscid() && 
			       (theDigi_.getTrknmb() == 2)) ) );
}

bool CSCTrackStub::operator>(const CSCTrackStub& rhs) const
{
  return ( isValid() && ( (!(rhs.isValid())) || (getQuality() > rhs.getQuality()) ||
			  (getQuality() == rhs.getQuality() && cscid() > rhs.cscid()) ||
			  (getQuality() == rhs.getQuality() && cscid() == rhs.cscid() &&
			   (theDigi_.getTrknmb() == 1)) ) );
}

