# ifndef SimMuL1_CLCT_h
#define SimMuL1_CLCT_h

#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
//#include <L1Trigger/CSCTriggerPrimitives/test/CSCCathodeLCTAnalyzer.h>

class CLCT
{
typedef std::vector<const CLCT*> CLCTCollection;

 public:
  /// constructor
  CLCT();
  /// copy constructor
  CLCT(const CLCT&);
  /// destructor
  ~CLCT();
  
  /// get the underlying trigger digi
  const CSCCLCTDigi* getTriggerDigi() const {return triggerDigi_;}
  /// detid
  const int getDetId() const {return detId_;}
  /// bx
  const int getBX() const {return bx_;}
  /// in readout?
  const bool inReadout() const {return inReadout_;}
  /// matched?
  const bool isDeltaOk() const {return deltaOk_;}
  ///
  const int getNumberHitsShared() const {return nHitsShared_;}
  /// strip number
  const int getStrip() const {return strip_;}
  /// delta to simtrack closest strip
  const int getDeltaStrip() const {return deltaStrip_;}
  const double getDeltaPhi() const {return deltaPhi_;}
  const double getDeltaY() const {return deltaY_;}
  /// center of strip phi
  const double getPhi() const {return phi_;}
  /// layer info
  //  const std::vector<CSCAnodeLayerInfo>& getLayerInfo() const {return layerInfo_;}
  /// simhits
  const std::vector<PSimHit>& getSimHits() const {return simHits_;}
  /// digis
  const std::vector<CSCComparatorDigi>& getDigis() const {return digis_;}
  
 private:
  /// underlying trigger digi
  const CSCCLCTDigi* triggerDigi_;
  /// layer info
  //  std::vector<CSCAnodeLayerInfo> layerInfo_;
  /// matching simhits
  std::vector<PSimHit> simHits_;
  /// matching digis
  std::vector<CSCComparatorDigi> digis_;
  
  /// detector ID
  int detId_;
  /// bunch crossing 
  int bx_;
  /// is it in the readout collection?
  bool inReadout_;
  /// was properly matched
  bool deltaOk_;
  /// number of SimHits shared with SimTrack
  int nHitsShared_;     
  /// SimHit's WG number 
  int strip_;
  /// delta to SimTrack closest wire
  int deltaStrip_;
  /// in (Z,R) -> (x,y) plane
  double deltaPhi_;
  /// deltas to SimTrack's 2D stub
  double deltaY_;
  /// center of wire group eta
  double phi_;  
};

#endif
