#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>
#include <L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h>
#include <L1TriggerConfig/DTTPGConfig/interface/DTConfigTraco.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <DataFormats/MuonDetId/interface/DTChamberId.h>

#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiverLUT.h>

CSCTriggerContainer<csctf::TrackStub> CSCTFDTReceiver::process(const L1MuDTChambPhContainer* dttrig)
{
  dtstubs.clear();
  if( !dttrig ) return dtstubs;

  const int dt_minBX = L1MuDTTFConfig::getBxMin();
  const int dt_maxBX = L1MuDTTFConfig::getBxMax();

  const int dt_toffs = 0;// changed since DT tpg now centers around zero //abs(dt_maxBX - dt_minBX);

  // consider all BX
  for(int bx = dt_minBX + dt_toffs; bx <= dt_maxBX + dt_toffs; ++bx)
    for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
      for(int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
	{
	  int wheel = (e == 1) ? 2 : -2;
	  int sector = 2*s - 1;
	  int csc_bx = bx + 6;//Delay DT stubs by 6 bx.

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

		      // --------------------------------------------------------------
		      // IKF: this code has been reformulated ...
		      // --------------------------------------------------------------
		      // // Convert stubs to CSC format (signed -> unsigned)
		      // // phi was 12 bits (signed) for pi radians = 57.3 deg
		      // // relative to center of 30 degree DT sector
		      // double tmp = static_cast<const double> (dtts[stub]->phi()) /
		      // 	DTConfigTraco::RESOLPSIR * 180./M_PI + 15.;
		      // int phi = static_cast<int> (tmp/62. * (1<<(CSCBitWidths::kGlobalPhiDataBitWidth)));
		      // --------------------------------------------------------------
		      // IKF  ...and is now this line, actually works a tiny bit better.
		      // --------------------------------------------------------------
		      //		      float tmp = dtts[stub] -> phi() * 1.0;
		      //
		      //		      tmp *= 90.0;
		      //		      tmp /= 31.0;
		      //		      //		      tmp /= M_PI;
		      //		      tmp /= 3.1416;
		      //		      tmp += 1057.0;
		      //
		      //		      int phi = static_cast<int> (tmp);

		      // --------------------------------------------------------------
		      // IKF  ...and is now this line, actually works a tiny bit better.
		      // --------------------------------------------------------------
		      int phi = dtts[stub] -> phi();

		      if (phi < 0) phi += 4096;

		      if (phi > 4096)
			{ std::cout << "AAAAAAAAAAGH TOO BIG PHI:" << phi << std::endl;
			  continue; 
			}
		      if (phi < 0){
			std::cout << "AAAAAAAAH NEG PHI" << phi << std::endl;
			continue;
		      }

		      phi = CSCTFDTReceiverLUT::lut[phi];

		      // --------------------------------------------------------------

		      // DT chambers may lie outside CSC sector boundary
		      // Eventually we need to extend CSC phi definition
		      // --------------------------------------------------------------
		      // IKF: this is a protection, physically can't happen in data (bus too narrow) - 
		      // - what really happens in data?
		      // --------------------------------------------------------------

		      phi = (phi>0) ? phi : 0;
		      phi = (phi<(1<<(CSCBitWidths::kGlobalPhiDataBitWidth))) ? phi :
			(1<<(CSCBitWidths::kGlobalPhiDataBitWidth))-1;


		      // change phib from 10 bits to 5
		      int phib = ((dtts[stub]->phiB() & 0x3FF) >> 5) & 0x1F;// 0x3FF=1023, 0x1F=31
		      int qual = dtts[stub]->code();
		      // barrel allows quality=0!
		      /// shift all by one and take mod 8, since DT quality of 7 is a null stub
		      qual = (qual + 1)%8;

		      CSCCorrelatedLCTDigi dtinfo(stub+1,1, qual, 0, stub, 0, phib, csc_bx+stub, 1+(is+1)%2);
		      DTChamberId dtid(wheel,1,iss+1);
		      csctf::TrackStub tsCSC(dtinfo,dtid, phi, 0);

		      dtstubs.push_back(tsCSC);
		    }
		}
	    }
	}

  return dtstubs;
}

