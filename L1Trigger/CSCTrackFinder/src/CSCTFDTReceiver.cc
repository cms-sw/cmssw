#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>
#include <L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h>
#include <L1Trigger/DTUtilities/interface/DTConfig.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>

CSCTriggerContainer<csctf::TrackStub> CSCTFDTReceiver::process(const L1MuDTChambPhContainer* dttrig)
{
  dtstubs.clear();

  const int dt_minBX = L1MuDTTFConfig::getBxMin();
  const int dt_maxBX = L1MuDTTFConfig::getBxMax();
  const int dt_centralBX = (dt_minBX + dt_maxBX)/2;

  // consider all BX
  for(int bx = dt_minBX; bx <= dt_maxBX; ++bx)
    for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  int wheel = (e == 1) ? 2 : -2;
	  int sector = 2*s - 1;
	  int csc_bx = bx - dt_centralBX + CSCConstants::TIME_OFFSET +4; // the + 4 is from observation
	  
	  // combine two 30 degree DT sectors into a 60 degree CSC
	  // sector.
	  for(int is = sector; is <= sector+1; ++is)
	    {
	      int iss = (is == 12) ? 0 : is;
	      L1MuDTChambPhDigi* dtts[2];
	      
	      for(int stub = 0; stub < 2; ++stub)
		{
		  dtts[stub] = (stub == 0) ? dttrig->chPhiSegm1(wheel,1,iss,bx) :
		                             dttrig->chPhiSegm2(wheel,1,iss,bx);
		  if(dtts[stub])
		    {
		      // Convert stubs to CSC format (signed -> unsigned)
		      int phi = dtts[stub]->phi();
		      phi += 614; // move DTphi lower bound to zero. Determined empirically.
		      if(is > sector) phi += 2048; //make [-30,30] -> [0,60]
		      phi = ((double)phi) * 31./90. * M_PI; // scale DT binning to CSC binning.
		                                         // the scale factor is (csc binning)/(dt binning) * pi
		      phi += 491; // match up DT sector boundary inside of CSC sector. Determined empirically.

		      // DT chambers may lie outside CSC sector boundary
		      // Eventually we need to extend CSC phi definition
		      phi = (phi>0) ? phi : 0;
		      phi = (phi<(1<<(CSCBitWidths::kGlobalPhiDataBitWidth))) ? phi : 
			(1<<(CSCBitWidths::kGlobalPhiDataBitWidth))-1;

		      // change phib from 10 bits to 6
		      int phib = (dtts[stub]->phiB() + DTConfig::RESOLPSI) / 16;
		      int qual = dtts[stub]->code();
		      // barrel allows quality=0!
		      /// shift all by one and take mod 8, since DT quality of 7 is a null stub
		      qual = (qual + 1)%8;
		        
		      CSCCorrelatedLCTDigi dtinfo(stub+1,1, qual, 0, 0, 0, phib, csc_bx, (stub+1) + 2*stub);
		      DTChamberId dtid(wheel,1,is);
		      csctf::TrackStub tsCSC(dtinfo,dtid, phi, 0);

		      dtstubs.push_back(tsCSC);
		    }
		}
	    }
	}

  return dtstubs;
}

