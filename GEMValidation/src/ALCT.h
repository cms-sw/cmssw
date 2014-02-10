#ifndef SimMuL1_ALCT_h
#define SimMuL1_ALCT_h

#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
//#include <L1Trigger/CSCTriggerPrimitives/test/CSCAnodeLCTAnalyzer.h>

class ALCT
{
 public:
  /// constructor
  ALCT();
  /// copy constructor
  ALCT(const ALCT&);
  /// destructor
  ~ALCT();
  
  /// get the underlying trigger digi
  const CSCALCTDigi* getTriggerDigi() const {return triggerDigi_;}
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
  /// WG number
  const int getWG() const {return wg_;}
  /// delta WG
  const int getDeltaWG() const {return deltaWG_;}
  /// delta phi
  const double getDeltaPhi() const {return deltaPhi_;}
  /// eta of WG center
  const double getEta() const {return eta_;}
  /// layerinfo
  //  const std::vector<CSCAnodeLayerInfo>& getLayerInfo() const {return layerInfo_;}
  /// corresponding simhits
  const std::vector<PSimHit>& getSimHits() const {return simHits_;}  
  /// corresponding digis
  const std::vector<CSCWireDigi>& getDigis() const {return digis_;} 

 private:
  /// underlying trigger digi
  const CSCALCTDigi* triggerDigi_;
  /// layer info
  //  std::vector<CSCAnodeLayerInfo> layerInfo_;
  /// matching simhits
  std::vector<PSimHit> simHits_;
  /// matching digis
  std::vector<CSCWireDigi> digis_;
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
  int wg_;
  /// delta to SimTrack closest wire
  int deltaWG_;
  /// in (Z,R) -> (x,y) plane
  double deltaPhi_;
  /// deltas to SimTrack's 2D stub
  double deltaY_;
  /// center of wire group eta
  double eta_;  
};

#endif
