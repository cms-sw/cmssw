#ifndef __SEGMENTLCTMATCHBOX_H 
#define __SEGMENTLCTMATCHBOX_H (1)

/** Author: Gian Piero Di Giovanni 
    taken from UserCode/digiovan/MuonAnalysis/TrigEff/ ,
    removed histogams */

#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "FWCore/Framework/interface/Event.h"

class SegmentLCTMatchBox {

 public:

  SegmentLCTMatchBox( int nHitsSegment = 4, 
		      int nHitsALCT    = 4, 
		      int nHitsCLCT    = 4,
		      int printLevel   = -1 );

  ~SegmentLCTMatchBox( );

  bool isLCTAble ( const CSCSegment &segment, int *match_analysis = 0 );

  bool isMatched ( const CSCSegment &segment, edm::Handle<CSCCorrelatedLCTDigiCollection>, int *match_analysis = 0 );

  bool belongsToTrigger ( const CSCSegment &segment, 
                          edm::Handle<L1CSCTrackCollection> tracks,
                          edm::Handle<CSCCorrelatedLCTDigiCollection> CSCTFlcts);

  int  whichMode ( const CSCSegment &segment, 
                   edm::Handle<L1CSCTrackCollection> tracks,
                   edm::Handle<CSCCorrelatedLCTDigiCollection> CSCTFlcts);

  int whichMode ( const CSCSegment &segment, edm::Handle<CSCCorrelatedLCTDigiCollection>, int *match_analysis = 0 );
  
  int _printLevel;

  //  compute distances in half-strips

  static int wireGroup ( const CSCRecHit2D& hit ); 
  static int halfStrip ( const CSCRecHit2D& hit );

  static int me11aNormalize ( int halfStrip );

  void setPrintLevel( int printLevel ){ _printLevel = printLevel; }

  static const int MATCH_CHAMBER  = 0x01 << 0;
  static const int MATCH_HASKEY   = 0x01 << 1;
  static const int MATCH_WIREG    = 0x01 << 2;
  static const int MATCH_STRIP    = 0x01 << 3;
  static const int MATCH_BOTH     = 0x01 << 4;
  static const int WEIRD_WIREG_SZ = 0x01 << 5;
  static const int WEIRD_STRIP_SZ = 0x01 << 6;

 private: 

  // driving variables

  int _nHitsSegment;
  int _nHitsALCT;
  int _nHitsCLCT;
  bool _monitorHist;


  // envelope arrays 

  static const int _alctEnvelopes[6];
  static const int _clctEnvelopes[6];

};

#endif
