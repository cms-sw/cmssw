#include <algorithm>
#include "L1Trigger/L1TMuonEndCap/interface/DTBunchCrossingCleaner.h"
#include "L1Trigger/L1TMuon/interface/deprecate/MuonTriggerPrimitive.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

using namespace L1TMuon;

namespace {
  typedef edm::ParameterSet PSet;
}

DTBunchCrossingCleaner::DTBunchCrossingCleaner(const PSet& ps):
  bx_window_size(ps.getParameter<int>("bxWindowSize")) {
}

TriggerPrimitiveCollection DTBunchCrossingCleaner::
clean( const TriggerPrimitiveCollection& inlist ) const {
  TriggerPrimitiveCollection leftovers = inlist;
  TriggerPrimitiveCollection outlist;
    
  auto tpin  = inlist.cbegin();
  auto inend = inlist.cend();
  for( ; tpin != inend; ++tpin ) {
    const TriggerPrimitive::DTData data = tpin->getDTData();

    // automatically add well matched tracks
    if( data.qualityCode != -1 && data.theta_quality != -1) {      
      outlist.push_back(*tpin);
      auto toerase = std::find(leftovers.begin(), leftovers.end(), *tpin);
      if( toerase != leftovers.end() ) {
	leftovers.erase(toerase);      
      }
    }

    // clean up phi/theta digis split across a BX
    // key off of the phi digis since they are of higher quality    
    if( data.qualityCode != -1 && data.theta_quality == -1) {      
      auto tp_bx = leftovers.cbegin();
      auto tp_bx_end = leftovers.cend();
      for( ; tp_bx != tp_bx_end; ++tp_bx ) {
	if( *tp_bx == *tpin ) continue;
	const TriggerPrimitive::DTData bx_data = tp_bx->getDTData();
	// look in-window and match to the segment number
	// requiring that we find a theta-segment with no phi info
	if( std::abs(bx_data.bx - data.bx) <= bx_window_size &&
	    bx_data.qualityCode == -1 && bx_data.theta_quality != -1 &&
	    data.segment_number == bx_data.segment_number ) {	  
	  // we need spoof the Digis used to create the individual objects
	  L1MuDTChambPhDigi phi_digi(data.bx,
				     data.wheel,
				     data.sector,
				     data.station,
				     data.radialAngle,
				     data.bendingAngle,
				     data.qualityCode,
				     data.Ts2TagCode,
				     data.BxCntCode);
	  int qual[7], position[7];
	  for( int i = 0; i < 7; ++i ) {
	    qual[i] = 0;
	    position[i] = 0;
	    if( bx_data.theta_bti_group == i ) {
	      qual[i] = bx_data.theta_quality;
	      position[i]  = bx_data.segment_number;	      
	    } 
	  }
	  L1MuDTChambThDigi the_digi(data.bx,
				     data.wheel,
				     data.sector,
				     data.station,
				     position,
				     qual);

	  DTChamberId the_id = tpin->detId<DTChamberId>();
	  TriggerPrimitive newtp(the_id,
				 phi_digi,
				 the_digi,
				 bx_data.theta_bti_group);
	  
	  outlist.push_back(newtp);
	  // remove these primitives from the leftovers list
	  auto phierase = std::find(leftovers.begin(), leftovers.end(), *tpin);
	  auto theerase = std::find(leftovers.begin(),leftovers.end(),*tp_bx);
	  if( phierase != leftovers.end() ) {
	    leftovers.erase(phierase);	  
	  }
	  if( theerase != leftovers.end() ) {
	    leftovers.erase(theerase);
	  }
	  break; // do not look for further matches!
	}
      }
    }
  }
  // re-insert any un-used trigger primitives
  auto lo_tp = leftovers.cbegin();
  auto lo_end = leftovers.cend();
  for( ; lo_tp != lo_end; ++lo_tp ) {    
    outlist.push_back(*lo_tp);
  }
  return outlist;
}
