#include <L1Trigger/CSCTrackFinder/interface/CSCTFSPCoreLogic.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <iostream>

//vpp_generated CSCTFSPCoreLogic::sp_;
vpp_generated_2010_01_22 CSCTFSPCoreLogic::sp_2010_01_22_;
vpp_generated_2010_07_28 CSCTFSPCoreLogic::sp_2010_07_28_;
vpp_generated_2010_09_01 CSCTFSPCoreLogic::sp_2010_09_01_;
vpp_generated_2010_10_11 CSCTFSPCoreLogic::sp_2010_10_11_;
vpp_generated_2010_12_10 CSCTFSPCoreLogic::sp_2010_12_10_;
vpp_generated_2011_01_18 CSCTFSPCoreLogic::sp_2011_01_18_;
vpp_generated_2012_01_31 CSCTFSPCoreLogic::sp_2012_01_31_;
vpp_generated_2012_03_13 CSCTFSPCoreLogic::sp_2012_03_13_;
vpp_generated_2012_07_30 CSCTFSPCoreLogic::sp_2012_07_30_;


// takes a trigger container and loads the first n bx of data into io_
void CSCTFSPCoreLogic::loadData(const CSCTriggerContainer<csctf::TrackStub>& theStubs,
				const unsigned& endcap, const unsigned& sector,
				const int& minBX, const int& maxBX)
{
  io_.clear();
  runme = 0;
  io_.resize(maxBX - minBX + 2);
  unsigned relative_bx = 0;

  for(int bx = minBX; bx <= maxBX; ++bx)
  {
  	for(int st = CSCDetId::minStationId(); st <= CSCDetId::maxStationId() + 1; ++st) // 1 - 5 for DT stubs
		{
	  	std::vector<csctf::TrackStub> stub_list;
	  	std::vector<csctf::TrackStub>::const_iterator stubi;
	  	if(st == 1)
	    {
	      stub_list = theStubs.get(endcap, st, sector, 1, bx);
	      std::vector<csctf::TrackStub> stub_list2 = theStubs.get(endcap, st, sector, 2, bx);
	      stub_list.insert(stub_list.end(), stub_list2.begin(), stub_list2.end());
	    }
	  	else stub_list = theStubs.get(endcap, st, sector, 0, bx);

	  	for(stubi = stub_list.begin(); stubi != stub_list.end(); stubi++)
	    {
	      runme |= stubi->isValid();
	      switch(st)
				{
					case 1:
		  		switch(stubi->getMPCLink())
		    	{
		    	case 1:
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 1)
						{
			  			io_[relative_bx+1].me1aVp   = stubi->isValid();
			  			io_[relative_bx+1].me1aQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1aEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1aPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1aCSCIdp  = stubi->cscid();
                                                io_[relative_bx+1].me1aCLCTp  = stubi->getCLCTPattern();
						}
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
						{
			  			io_[relative_bx+1].me1dVp   = stubi->isValid();
			  			io_[relative_bx+1].me1dQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1dEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1dPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1dAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1dCSCIdp  = stubi->cscid();
                                                io_[relative_bx+1].me1dCLCTp  = stubi->getCLCTPattern();
						}
		      break;
		    	case 2:
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 1)
						{
			  			io_[relative_bx+1].me1bVp   = stubi->isValid();
			  			io_[relative_bx+1].me1bQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1bEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1bPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1bCSCIdp  = stubi->cscid();
                                                io_[relative_bx+1].me1bCLCTp  = stubi->getCLCTPattern();
						}
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
						{
			  			io_[relative_bx+1].me1eVp   = stubi->isValid();
			  			io_[relative_bx+1].me1eQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1eEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1ePhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1eAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1eCSCIdp  = stubi->cscid();
                                                io_[relative_bx+1].me1eCLCTp  = stubi->getCLCTPattern();
						}
		      break;
		    	case 3:
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 1)
						{
			  			io_[relative_bx+1].me1cVp   = stubi->isValid();
			  			io_[relative_bx+1].me1cQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1cEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1cPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1cCSCIdp  = stubi->cscid();
                                                io_[relative_bx+1].me1cCLCTp  = stubi->getCLCTPattern();
						}
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
						{
			  			io_[relative_bx+1].me1fVp   = stubi->isValid();
			  			io_[relative_bx+1].me1fQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1fEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1fPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1fAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1fCSCIdp  = stubi->cscid();
                                                io_[relative_bx+1].me1fCLCTp  = stubi->getCLCTPattern();
						}
		      break;
		    	default:
		      	edm::LogWarning("CSCTFSPCoreLogic::loadData()") << "SERIOUS ERROR: MPC LINK " << stubi->getMPCLink()
							<< " NOT IN RANGE [1,3]\n";
		    	};
		  	break;
				case 2:
		  		switch(stubi->getMPCLink())
		    	{
		    		case 1:
		      		io_[relative_bx+1].me2aVp   = stubi->isValid();
		      		io_[relative_bx+1].me2aQp   = stubi->getQuality();
		      		io_[relative_bx+1].me2aEtap = stubi->etaPacked();
		      		io_[relative_bx+1].me2aPhip = stubi->phiPacked();
		      		io_[relative_bx+1].me2aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      	break;
		    		case 2:
		      		io_[relative_bx+1].me2bVp   = stubi->isValid();
		      		io_[relative_bx+1].me2bQp   = stubi->getQuality();
		      		io_[relative_bx+1].me2bEtap = stubi->etaPacked();
		      		io_[relative_bx+1].me2bPhip = stubi->phiPacked();
		      		io_[relative_bx+1].me2bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      	break;
		    		case 3:
		      		io_[relative_bx+1].me2cVp   = stubi->isValid();
		      		io_[relative_bx+1].me2cQp   = stubi->getQuality();
		      		io_[relative_bx+1].me2cEtap = stubi->etaPacked();
		      		io_[relative_bx+1].me2cPhip = stubi->phiPacked();
		      		io_[relative_bx+1].me2cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      	break;
		    		default:
		      		edm::LogWarning("CSCTFSPCoreLogic::loadData()") << "SERIOUS ERROR: MPC LINK " << stubi->getMPCLink()
								<< " NOT IN RANGE [1,3]\n";
		    	};
		  	break;
				case 3:
		  		switch(stubi->getMPCLink())
		    	{
		    		case 1:
		      		io_[relative_bx+1].me3aVp   = stubi->isValid();
		      		io_[relative_bx+1].me3aQp   = stubi->getQuality();
		      		io_[relative_bx+1].me3aEtap = stubi->etaPacked();
		      		io_[relative_bx+1].me3aPhip = stubi->phiPacked();
		      		io_[relative_bx+1].me3aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      	break;
		    		case 2:
		      		io_[relative_bx+1].me3bVp   = stubi->isValid();
		      		io_[relative_bx+1].me3bQp   = stubi->getQuality();
		      		io_[relative_bx+1].me3bEtap = stubi->etaPacked();
		      		io_[relative_bx+1].me3bPhip = stubi->phiPacked();
		      		io_[relative_bx+1].me3bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      	break;
		    		case 3:
		      		io_[relative_bx+1].me3cVp   = stubi->isValid();
		      		io_[relative_bx+1].me3cQp   = stubi->getQuality();
		      		io_[relative_bx+1].me3cEtap = stubi->etaPacked();
		      		io_[relative_bx+1].me3cPhip = stubi->phiPacked();
		      		io_[relative_bx+1].me3cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      	break;
		    		default:
		      		edm::LogWarning("CSCTFSPCoreLogic::loadData()") << "SERIOUS ERROR: MPC LINK " << stubi->getMPCLink()
								<< " NOT IN RANGE [1,3]\n";
		    	};
		  	break;
			case 4:
		  	switch(stubi->getMPCLink())
		    {
		    	case 1:
		      	io_[relative_bx+1].me4aVp   = stubi->isValid();
		      	io_[relative_bx+1].me4aQp   = stubi->getQuality();
		      	io_[relative_bx+1].me4aEtap = stubi->etaPacked();
		      	io_[relative_bx+1].me4aPhip = stubi->phiPacked();
		      	io_[relative_bx+1].me4aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      break;
		    	case 2:
		      	io_[relative_bx+1].me4bVp   = stubi->isValid();
		      	io_[relative_bx+1].me4bQp   = stubi->getQuality();
		      	io_[relative_bx+1].me4bEtap = stubi->etaPacked();
		      	io_[relative_bx+1].me4bPhip = stubi->phiPacked();
		      	io_[relative_bx+1].me4bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      break;
		    	case 3:
		      	io_[relative_bx+1].me4cVp   = stubi->isValid();
		      	io_[relative_bx+1].me4cQp   = stubi->getQuality();
		      	io_[relative_bx+1].me4cEtap = stubi->etaPacked();
		      	io_[relative_bx+1].me4cPhip = stubi->phiPacked();
		      	io_[relative_bx+1].me4cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
		      break;
		    	default:
		      	edm::LogWarning("CSCTFSPCoreLogic::loadData()") << "SERIOUS ERROR: MPC LINK " << stubi->getMPCLink()
							<< " NOT IN RANGE [1,3]\n";
		    };
		  break;
		case 5:
			// We need to put the DT stubs 1 BX ahead of the CSC ones for the TF firmware
			//std::cout << "DT Stub at bx: " << relative_bx << std::endl;
		  	switch(stubi->getMPCLink())
		  	{
		  	case 1:
			  if (this->GetSPFirmwareVersion() < 20100629) {
			    // introducing the bug which was causing only even DT qualities
			    // to get accepted
			    if(stubi->getQuality()%2==1)
			      {
				//io_[relative_bx].mb1aVp   = stubi->isValid();
				io_[relative_bx].mb1aVp	  = stubi->getStrip();
				io_[relative_bx].mb1aQp   = stubi->getQuality();
				io_[relative_bx].mb1aPhip = stubi->phiPacked();
                                io_[relative_bx].mb1aBendp= stubi->getBend();
			      }
			  } else {
			    io_[relative_bx].mb1aVp	  = stubi->getStrip();
			    io_[relative_bx].mb1aQp   = stubi->getQuality();
			    io_[relative_bx].mb1aPhip = stubi->phiPacked();
                            io_[relative_bx].mb1aBendp= stubi->getBend();
			  }
			break;
		    	case 2:
			  if (this->GetSPFirmwareVersion() < 20100629) {
			    // introducing the bug which was causing only even DT qualities
			    // to get accepted
			    if(stubi->getQuality()%2==1)
			      {
				//io_[relative_bx].mb1aVp   = stubi->isValid();
				io_[relative_bx].mb1bVp	  = stubi->getStrip();
				io_[relative_bx].mb1bQp   = stubi->getQuality();
				io_[relative_bx].mb1bPhip = stubi->phiPacked();
                                io_[relative_bx].mb1bBendp= stubi->getBend();
			      }
			  } else {
			    io_[relative_bx].mb1bVp	  = stubi->getStrip();
			    io_[relative_bx].mb1bQp   = stubi->getQuality();
			    io_[relative_bx].mb1bPhip = stubi->phiPacked();
                            io_[relative_bx].mb1bBendp= stubi->getBend();
			  }
			break;
		    /*case 3:
		      io_[relative_bx].mb1cVp   = stubi->isValid();
                      io_[relative_bx].mb1cQp   = stubi->getQuality();
                      io_[relative_bx].mb1cPhip = stubi->phiPacked();
                      break;
		    case 4:
		      io_[relative_bx].mb1dVp   = stubi->isValid();
                      io_[relative_bx].mb1dQp   = stubi->getQuality();
                      io_[relative_bx].mb1dPhip = stubi->phiPacked();
                      break;*/
		    default:
		      edm::LogWarning("CSCTFSPCoreLogic::loadData()") <<  "SERIOUS ERROR: DT LINK " << stubi->getMPCLink()
								      << " NOT IN RANGE [1,4]\n";
		    }
		  break;
		default:
		  edm::LogWarning("CSCTFSPCoreLogic::loadData()") << "SERIOUS ERROR: STATION  " << st << " NOT IN RANGE [1,5]\n";
		};
	    }
	}
      ++relative_bx;
    }
}

// Here we should assume the loadData() has been called...
// But in reality you don't need to.
bool CSCTFSPCoreLogic::run(const unsigned& endcap, const unsigned& sector, const unsigned& latency,
			   const unsigned& etamin1, const unsigned& etamin2, const unsigned& etamin3, const unsigned& etamin4,
		  	   const unsigned& etamin5, const unsigned& etamin6, const unsigned& etamin7, const unsigned& etamin8,
			   const unsigned& etamax1, const unsigned& etamax2, const unsigned& etamax3, const unsigned& etamax4,
	  	       const unsigned& etamax5, const unsigned& etamax6, const unsigned& etamax7, const unsigned& etamax8,
			   const unsigned& etawin1, const unsigned& etawin2, const unsigned& etawin3,
			   const unsigned& etawin4, const unsigned& etawin5, const unsigned& etawin6, const unsigned& etawin7,
			   const unsigned& mindphip, const unsigned& mindetap,
			   const unsigned& mindeta12_accp,
			   const unsigned& maxdeta12_accp, const unsigned& maxdphi12_accp,
			   const unsigned& mindeta13_accp,
			   const unsigned& maxdeta13_accp, const unsigned& maxdphi13_accp,
			   const unsigned& mindeta112_accp,
			   const unsigned& maxdeta112_accp, const unsigned& maxdphi112_accp,
			   const unsigned& mindeta113_accp,
			   const unsigned& maxdeta113_accp, const unsigned& maxdphi113_accp,
			   const unsigned& mindphip_halo, const unsigned& mindetap_halo,
			   const unsigned& straightp, const unsigned& curvedp,
			   const unsigned& mbaPhiOff, const unsigned& mbbPhiOff,
			   const unsigned& m_extend_length,
			   const unsigned& m_allowALCTonly, const unsigned& m_allowCLCTonly,
			   const unsigned& m_preTrigger, const unsigned& m_widePhi,
			   const int& minBX, const int& maxBX)
{
  mytracks.clear();

  int train_length = io_.size();
  int bx = 0;
  io_.resize(train_length + latency);
  std::vector<SPio>::iterator io;

  // run over enough clock cycles to get tracks from input stubs.
  for( io = io_.begin(); io != io_.end() && runme; io++)
  {
	

    switch(this->GetCoreFirmwareVersion()) {
    case 20100122:
      sp_2010_01_22_.wrap
	(
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,
	 
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1,etamin2,etamin3,etamin4,etamin5,etamin6,etamin7,etamin8,
	 etamax1,etamax2,etamax3,etamax4,etamax5,etamax6,etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, etawin6, etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );

      break;
    case 20100728:
      sp_2010_07_28_.wrap
	(
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,
	 
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1,etamin2,etamin3,etamin4,etamin5,etamin6,etamin7,etamin8,
	 etamax1,etamax2,etamax3,etamax4,etamax5,etamax6,etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, etawin6, etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;
    case 20100901:
      sp_2010_09_01_.wrap
	(
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,
	 
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1,etamin2,etamin3,etamin4,etamin5,etamin6,etamin7,etamin8,
	 etamax1,etamax2,etamax3,etamax4,etamax5,etamax6,etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, etawin6, etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;

    case 20101011:
      sp_2010_10_11_.wrap
	(
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,
	 
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1,etamin2,etamin3,etamin4,etamin5,/*etamin6,*/etamin7,etamin8,
	 etamax1,etamax2,etamax3,etamax4,etamax5,/*etamax6,*/etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;


    case 20101210:
      sp_2010_12_10_.wrap
	(
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,
	 
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1,etamin2,etamin3,etamin4,etamin5,/*etamin6,*/etamin7,etamin8,
	 etamax1,etamax2,etamax3,etamax4,etamax5,/*etamax6,*/etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;

    case 20110118:
      sp_2011_01_18_.wrap
	(
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,  io->me1aCLCTp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,  io->me1bCLCTp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,  io->me1cCLCTp,
	                                                                              
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,  io->me1dCLCTp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,  io->me1eCLCTp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,  io->me1fCLCTp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip, io->mb1aBendp,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip, io->mb1bBendp,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip, io->mb1cBendp,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip, io->mb1dBendp,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1,etamin2,etamin3,etamin4,etamin5,/*etamin6,*/etamin7,etamin8,
	 etamax1,etamax2,etamax3,etamax4,etamax5,/*etamax6,*/etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;
      
      case 20120131:
      setNLBTables();
      
      sp_2012_01_31_.wrap
      (
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,  io->me1aCLCTp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,  io->me1bCLCTp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,  io->me1cCLCTp,
	                                                                              
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,  io->me1dCLCTp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,  io->me1eCLCTp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,  io->me1fCLCTp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip, io->mb1aBendp,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip, io->mb1bBendp,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip, io->mb1cBendp,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip, io->mb1dBendp,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp, io->phdiff_aHp, io->phdiff_bHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp, io->phdiff_aMp, io->phdiff_bMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp, io->phdiff_aLp, io->phdiff_bLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1, etamin2, etamin3, etamin4, etamin5, /*etamin6,*/ etamin7,etamin8,
	 etamax1, etamax2, etamax3, etamax4, etamax5, /*etamax6,*/ etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;

        case 20120313:
          
      sp_2012_03_13_.wrap
      (
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,  io->me1aCLCTp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,  io->me1bCLCTp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,  io->me1cCLCTp,
	                                                                              
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,  io->me1dCLCTp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,  io->me1eCLCTp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,  io->me1fCLCTp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip, io->mb1aBendp,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip, io->mb1bBendp,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip, io->mb1cBendp,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip, io->mb1dBendp,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp, io->phdiff_aHp, io->phdiff_bHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp, io->phdiff_aMp, io->phdiff_bMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp, io->phdiff_aLp, io->phdiff_bLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1, etamin2, etamin3, etamin4, etamin5, /*etamin6,*/ etamin7,etamin8,
	 etamax1, etamax2, etamax3, etamax4, etamax5, /*etamax6,*/ etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
      break;

         case 20120730:

      sp_2012_07_30_.wrap
      (
         io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,  io->me1aCLCTp,
         io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,  io->me1bCLCTp,
         io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,  io->me1cCLCTp,

         io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,  io->me1dCLCTp,
         io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,  io->me1eCLCTp,
         io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,  io->me1fCLCTp,

         io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
         io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
         io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,

         io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
         io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
         io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,

         io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
         io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
         io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,

         io->mb1aVp, io->mb1aQp, io->mb1aPhip, io->mb1aBendp,
         io->mb1bVp, io->mb1bQp, io->mb1bPhip, io->mb1bBendp,
         io->mb1cVp, io->mb1cQp, io->mb1cPhip, io->mb1cBendp,
         io->mb1dVp, io->mb1dQp, io->mb1dPhip, io->mb1dBendp,

         io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp, io->phdiff_aHp, io->phdiff_bHp,
         io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp, io->phdiff_aMp, io->phdiff_bMp,
         io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp, io->phdiff_aLp, io->phdiff_bLp,

         io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
         io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
         io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,

         etamin1, etamin2, etamin3, etamin4, etamin5, /*etamin6,*/ etamin7,etamin8,
         etamax1, etamax2, etamax3, etamax4, etamax5, /*etamax6,*/ etamax7,etamax8,
         etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7,
         mindphip, mindetap,

         mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
         mindeta13_accp, maxdeta13_accp, maxdphi13_accp,

         mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
         mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
         mindphip_halo, mindetap_halo,

         straightp, curvedp,
         mbaPhiOff, mbbPhiOff,
         (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
         );
      break;
     
  default:
    edm::LogInfo("CSCSTFSPCoreLogic") << "Warning: using the default core is what you want?"
                                      << " Core version is " << this->GetCoreFirmwareVersion();
    setNLBTables();
    sp_2012_01_31_.wrap
      (
	 io->me1aVp, io->me1aQp, io->me1aEtap, io->me1aPhip, io->me1aCSCIdp,  io->me1aCLCTp,
	 io->me1bVp, io->me1bQp, io->me1bEtap, io->me1bPhip, io->me1bCSCIdp,  io->me1bCLCTp,
	 io->me1cVp, io->me1cQp, io->me1cEtap, io->me1cPhip, io->me1cCSCIdp,  io->me1cCLCTp,
	                                                                              
	 io->me1dVp, io->me1dQp, io->me1dEtap, io->me1dPhip, io->me1dCSCIdp,  io->me1dCLCTp,
	 io->me1eVp, io->me1eQp, io->me1eEtap, io->me1ePhip, io->me1eCSCIdp,  io->me1eCLCTp,
	 io->me1fVp, io->me1fQp, io->me1fEtap, io->me1fPhip, io->me1fCSCIdp,  io->me1fCLCTp,
	 
	 io->me2aVp, io->me2aQp, io->me2aEtap, io->me2aPhip,
	 io->me2bVp, io->me2bQp, io->me2bEtap, io->me2bPhip,
	 io->me2cVp, io->me2cQp, io->me2cEtap, io->me2cPhip,
	 
	 io->me3aVp, io->me3aQp, io->me3aEtap, io->me3aPhip,
	 io->me3bVp, io->me3bQp, io->me3bEtap, io->me3bPhip,
	 io->me3cVp, io->me3cQp, io->me3cEtap, io->me3cPhip,
	 
	 io->me4aVp, io->me4aQp, io->me4aEtap, io->me4aPhip,
	 io->me4bVp, io->me4bQp, io->me4bEtap, io->me4bPhip,
	 io->me4cVp, io->me4cQp, io->me4cEtap, io->me4cPhip,
	 
	 io->mb1aVp, io->mb1aQp, io->mb1aPhip, io->mb1aBendp,
	 io->mb1bVp, io->mb1bQp, io->mb1bPhip, io->mb1bBendp,
	 io->mb1cVp, io->mb1cQp, io->mb1cPhip, io->mb1cBendp,
	 io->mb1dVp, io->mb1dQp, io->mb1dPhip, io->mb1dBendp,
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp, io->phdiff_aHp, io->phdiff_bHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp, io->phdiff_aMp, io->phdiff_bMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp, io->phdiff_aLp, io->phdiff_bLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 etamin1, etamin2, etamin3, etamin4, etamin5, /*etamin6,*/ etamin7,etamin8,
	 etamax1, etamax2, etamax3, etamax4, etamax5, /*etamax6,*/ etamax7,etamax8,
	 etawin1, etawin2, etawin3, etawin4, etawin5, /*etawin6,*/ etawin7, 
	 mindphip, mindetap,
	 
	 mindeta12_accp, maxdeta12_accp, maxdphi12_accp,
	 mindeta13_accp, maxdeta13_accp, maxdphi13_accp,
	 
	 mindeta112_accp, maxdeta112_accp, maxdphi112_accp,
	 mindeta113_accp, maxdeta113_accp, maxdphi113_accp,
	 mindphip_halo, mindetap_halo, 
	 
	 straightp, curvedp,
	 mbaPhiOff, mbbPhiOff,
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi)
	 );
    break;
    
    }



    if ( IsVerbose() ) {
      std::cout << "Core Verbose Output For Debugging\n";
      std::cout << io->me1aVp << " " << io->me1aQp << " " << io->me1aEtap << " " << io->me1aPhip << " " << io->me1aCSCIdp << " " << io->me1aCLCTp << std::endl;
      std::cout << io->me1bVp << " " << io->me1bQp << " " << io->me1bEtap << " " << io->me1bPhip << " " << io->me1bCSCIdp << " " << io->me1bCLCTp << std::endl;
      std::cout << io->me1cVp << " " << io->me1cQp << " " << io->me1cEtap << " " << io->me1cPhip << " " << io->me1cCSCIdp << " " << io->me1cCLCTp << std::endl;
    
      std::cout << io->me1dVp << " " << io->me1dQp << " " << io->me1dEtap << " " << io->me1dPhip << " " << io->me1dCSCIdp << " " << io->me1dCLCTp << std::endl;
      std::cout << io->me1eVp << " " << io->me1eQp << " " << io->me1eEtap << " " << io->me1ePhip << " " << io->me1eCSCIdp << " " << io->me1eCLCTp << std::endl;
      std::cout << io->me1fVp << " " << io->me1fQp << " " << io->me1fEtap << " " << io->me1fPhip << " " << io->me1fCSCIdp << " " << io->me1fCLCTp << std::endl;
      
      std::cout << io->me2aVp << " " << io->me2aQp << " " << io->me2aEtap << " " << io->me2aPhip << " " << 0 << " " << 0 << std::endl;
      std::cout << io->me2bVp << " " << io->me2bQp << " " << io->me2bEtap << " " << io->me2bPhip << " " << 0 << " " << 0 << std::endl;
      std::cout << io->me2cVp << " " << io->me2cQp << " " << io->me2cEtap << " " << io->me2cPhip << " " << 0 << " " << 0 << std::endl;
      
      std::cout << io->me3aVp << " " << io->me3aQp << " " << io->me3aEtap << " " << io->me3aPhip << " " << 0 << " " << 0 << std::endl;
      std::cout << io->me3bVp << " " << io->me3bQp << " " << io->me3bEtap << " " << io->me3bPhip << " " << 0 << " " << 0 << std::endl;
      std::cout << io->me3cVp << " " << io->me3cQp << " " << io->me3cEtap << " " << io->me3cPhip << " " << 0 << " " << 0 << std::endl;
      
      std::cout << io->me4aVp << " " << io->me4aQp << " " << io->me4aEtap << " " << io->me4aPhip << " " << 0 << " " << 0 << std::endl;
      std::cout << io->me4bVp << " " << io->me4bQp << " " << io->me4bEtap << " " << io->me4bPhip << " " << 0 << " " << 0 << std::endl;
      std::cout << io->me4cVp << " " << io->me4cQp << " " << io->me4cEtap << " " << io->me4cPhip << " " << 0 << " " << 0 << std::endl;
      
      std::cout << io->mb1aVp << " " << io->mb1aQp << " " << 0 << " " << io->mb1aPhip <<" " << 0 << " " << io->mb1aBendp << std::endl;
      std::cout << io->mb1bVp << " " << io->mb1bQp << " " << 0 << " " << io->mb1bPhip <<" " << 0 << " " << io->mb1bBendp << std::endl;
      std::cout << io->mb1cVp << " " << io->mb1cQp << " " << 0 << " " << 0/*io->mb1cPhip*/ <<" " << 0 << " " << 0/*io->mb1aBendp*/ << std::endl;
      std::cout << io->mb1dVp << " " << io->mb1dQp << " " << 0 << " " << 0/*io->mb1dPhip*/ <<" " << 0 << " " << 0/*io->mb1aBendp*/ << std::endl;
      
      std::cout << io->ptHp  << " " << io->signHp  << " " << io->modeMemHp << " " << io->etaPTHp << " " << io->FRHp << " " << io->phiHp << std::endl;
      std::cout << io->ptMp  << " " << io->signMp  << " " << io->modeMemMp << " " << io->etaPTMp << " " << io->FRMp << " " << io->phiMp << std::endl;
      std::cout << io->ptLp  << " " << io->signLp  << " " << io->modeMemLp << " " << io->etaPTLp << " " << io->FRLp << " " << io->phiLp << std::endl;
      
      std::cout << io->me1idH << " " << io->me2idH << " " << io->me3idH << " " << io->me4idH << " " << io->mb1idH << " " << io->mb2idH << std::endl;
      std::cout << io->me1idM << " " << io->me2idM << " " << io->me3idM << " " << io->me4idM << " " << io->mb1idM << " " << io->mb2idM << std::endl;
      std::cout << io->me1idL << " " << io->me2idL << " " << io->me3idL << " " << io->me4idL << " " << io->mb1idL << " " << io->mb2idL << std::endl << std::endl;
    }
    ++bx;
  }
  
  
  bx = 0;

  //int nmuons = 0;
  // start from where tracks could first possibly appear
  // read out tracks from io_
  // We add first +1 to the starting position because the CSC data started 1 BX after DT,
  // and the other +1 because of the number of calls to the core (i.e. latency+1):
  for(io = io_.begin() + latency + 1 + 1; io != io_.end(); io++)
    {
      csc::L1TrackId trkHid(endcap, sector), trkMid(endcap, sector), trkLid(endcap, sector);
      trkHid.setMode(io->modeMemHp);
      trkMid.setMode(io->modeMemMp);
      trkLid.setMode(io->modeMemLp);

      csc::L1Track trkH(trkHid), trkM(trkMid), trkL(trkLid);

      ptadd LUTAddressH, LUTAddressM, LUTAddressL;

      // construct PT LUT address for all possible muons
      LUTAddressH.delta_phi_12   = io->ptHp & 0xff;
      LUTAddressH.delta_phi_23   = (io->ptHp >> 8) & 0xf;
      LUTAddressH.track_eta      = (io->etaPTHp>>1) & 0xf;
      LUTAddressH.track_mode     = io->modeMemHp & 0xf;
//	Line Replaced due to removal of spbits.h, note that
//	BWPT and MODE_ACC are now hard coded (13 and 15 respectively)
//      LUTAddressH.delta_phi_sign = (io->ptHp >> (BWPT-1)) & 0x1;
      LUTAddressH.delta_phi_sign = (io->ptHp >> (13-1)) & 0x1;
      LUTAddressH.track_fr       = io->FRHp & 0x1;

      LUTAddressM.delta_phi_12   = io->ptMp & 0xff;
      LUTAddressM.delta_phi_23   = (io->ptMp >> 8) & 0xf;
      LUTAddressM.track_eta      = (io->etaPTMp>>1) & 0xf;
      LUTAddressM.track_mode     = io->modeMemMp & 0xf;
//      LUTAddressM.delta_phi_sign = (io->ptMp >> (BWPT-1)) & 0x1;
      LUTAddressM.delta_phi_sign = (io->ptMp >> (13-1)) & 0x1;
      LUTAddressM.track_fr       = io->FRMp & 0x1;

      LUTAddressL.delta_phi_12   = io->ptLp & 0xff;
      LUTAddressL.delta_phi_23   = (io->ptLp >> 8) & 0xf;
      LUTAddressL.track_eta      = (io->etaPTLp>>1) & 0xf;
      LUTAddressL.track_mode     = io->modeMemLp & 0xf;
//      LUTAddressL.delta_phi_sign = (io->ptLp >> (BWPT-1)) & 0x1;
      LUTAddressL.delta_phi_sign = (io->ptLp >> (13-1)) & 0x1;
      LUTAddressL.track_fr       = io->FRLp & 0x1;

     // Core's input was loaded in a relative time window starting from BX=1(CSC)/0(DT)
     // If we account for latency related shift in the core's output (as we do in this loop)
     //  then output tracks appear in the same BX as input stubs.
     // To create new time window with perfectly timed-in tracks placed at BX=0 we introduce a shift:
     int shift = (maxBX - minBX)/2;

      if(LUTAddressH.track_mode)
	{
	  trkH.setPtLUTAddress(LUTAddressH.toint());
	  trkH.setChargePacked(~(io->signHp)&0x1);
	  trkH.setLocalPhi(io->phiHp);
	  trkH.setEtaPacked(io->etaPTHp);
	  trkH.setBx((int)(bx)-shift);
	  trkH.setStationIds(io->me1idH&0x7, io->me2idH&0x3, io->me3idH&0x3, io->me4idH&0x3, io->mb1idH&0x3 );
	  trkH.setTbins     (io->me1idH>>3,  io->me2idH>>2,  io->me3idH>>2,  io->me4idH>>2,  io->mb1idH>>2 );
	  trkH.setOutputLink(1);
	  if( LUTAddressH.track_mode==15 ) trkH.setFineHaloPacked(1);
	  mytracks.push_back(trkH);
	}
      if(LUTAddressM.track_mode)
	{
	  trkM.setPtLUTAddress(LUTAddressM.toint());
	  trkM.setChargePacked(~(io->signMp)&0x1);
	  trkM.setLocalPhi(io->phiMp);
	  trkM.setEtaPacked(io->etaPTMp);
	  trkM.setBx((int)(bx)-shift);
	  trkM.setStationIds(io->me1idM&0x7, io->me2idM&0x3, io->me3idM&0x3, io->me4idM&0x3, io->mb1idM&0x3 );
	  trkM.setTbins     (io->me1idM>>3,  io->me2idM>>2,  io->me3idM>>2,  io->me4idM>>2,  io->mb1idM>>2 );
	  trkM.setOutputLink(2);
	  if( LUTAddressM.track_mode==15 ) trkM.setFineHaloPacked(1);
	  mytracks.push_back(trkM);
	}
      if(LUTAddressL.track_mode)
	{
	  trkL.setPtLUTAddress(LUTAddressL.toint());
	  trkL.setChargePacked(~(io->signLp)&0x1);
	  trkL.setLocalPhi(io->phiLp);
	  trkL.setEtaPacked(io->etaPTLp);
	  trkL.setBx((int)(bx)-shift);
	  trkL.setStationIds(io->me1idL&0x7, io->me2idL&0x3, io->me3idL&0x3, io->me4idL&0x3, io->mb1idL&0x3 );
	  trkL.setTbins     (io->me1idL>>3,  io->me2idL>>2,  io->me3idL>>2,  io->me4idL>>2,  io->mb1idL>>2 );
	  trkL.setOutputLink(3);
	  if( LUTAddressL.track_mode==15 ) trkL.setFineHaloPacked(1);
	  mytracks.push_back(trkL);
	}
      ++bx;
    }
  return runme;
}

void CSCTFSPCoreLogic::setNLBTables()
{
  /*
    These arrays define the non-linear dPhi bins used by the SP core logic.
    dPhi is mapped to an integer value value. The integer value is remapped
    to phi-units in CSCTFPtLUT.
  */
  
  // initialize the dphi arrays to maximum possible value
  for (int i = 0; i < 1024; i++)
    {
      // 5-bit words
      sp_2012_01_31_.spvpp_ptu2a_comp_dphi_5[i] = (1<<5)-1;
      sp_2012_01_31_.spvpp_ptu2b_comp_dphi_5[i] = (1<<5)-1;
      sp_2012_01_31_.spvpp_ptu2c_comp_dphi_5[i] = (1<<5)-1;

      sp_2012_01_31_.spvpp_ptu3a_comp_dphi_5[i] = (1<<5)-1;
      sp_2012_01_31_.spvpp_ptu3b_comp_dphi_5[i] = (1<<5)-1;
      sp_2012_01_31_.spvpp_ptu3c_comp_dphi_5[i] = (1<<5)-1;
      
      
      // 7-bit words
      sp_2012_01_31_.spvpp_ptu2a_comp_dphi_7[i] = (1<<7)-1;
      sp_2012_01_31_.spvpp_ptu2b_comp_dphi_7[i] = (1<<7)-1;
      sp_2012_01_31_.spvpp_ptu2c_comp_dphi_7[i] = (1<<7)-1;

      sp_2012_01_31_.spvpp_ptu3a_comp_dphi_7[i] = (1<<7)-1;
      sp_2012_01_31_.spvpp_ptu3b_comp_dphi_7[i] = (1<<7)-1;
      sp_2012_01_31_.spvpp_ptu3c_comp_dphi_7[i] = (1<<7)-1;

      // 8-bit words
      sp_2012_01_31_.spvpp_ptu2a_comp_dphi_8[i] = (1<<8)-1;
      sp_2012_01_31_.spvpp_ptu2b_comp_dphi_8[i] = (1<<8)-1;
      sp_2012_01_31_.spvpp_ptu2c_comp_dphi_8[i] = (1<<8)-1;
      
      sp_2012_01_31_.spvpp_ptu3a_comp_dphi_8[i] = (1<<8)-1;
      sp_2012_01_31_.spvpp_ptu3b_comp_dphi_8[i] = (1<<8)-1;
      sp_2012_01_31_.spvpp_ptu3c_comp_dphi_8[i] = (1<<8)-1;

      sp_2012_01_31_.spvpp_ptu4a_comp_dphi_8[i] = (1<<8)-1;
      sp_2012_01_31_.spvpp_ptu4b_comp_dphi_8[i] = (1<<8)-1;
      sp_2012_01_31_.spvpp_ptu4c_comp_dphi_8[i] = (1<<8)-1;
    }
  
  // define the non-linear bin map. This takes dphi (phi-units) --> integer value
  
  // 5-bit table
  int dPhiTable_5b[256] =
    {	0	,	1	,	2	,	2	,	3	,	4	,	4	,	5	,	5	,	6	,	6	,	7	,	7	,	8	,	8	,	9	,	9	,	9	,	10	,	10	,	10	,	11	,	11	,	11	,	12	,	12	,	12	,	12	,	13	,	13	,	13	,	13	,	14	,	14	,	14	,	14	,	14	,	15	,	15	,	15	,	15	,	16	,	16	,	16	,	16	,	16	,	16	,	17	,	17	,	17	,	17	,	17	,	17	,	18	,	18	,	18	,	18	,	18	,	18	,	18	,	19	,	19	,	19	,	19	,	19	,	19	,	19	,	20	,	20	,	20	,	20	,	20	,	20	,	20	,	20	,	21	,	21	,	21	,	21	,	21	,	21	,	21	,	21	,	21	,	22	,	22	,	22	,	22	,	22	,	22	,	22	,	22	,	22	,	22	,	23	,	23	,	23	,	23	,	23	,	23	,	23	,	23	,	23	,	23	,	23	,	24	,	24	,	24	,	24	,	24	,	24	,	24	,	24	,	24	,	24	,	24	,	24	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	25	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	26	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	27	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	28	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	29	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	30	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	,	31	};

  // 7-bit table
  int dPhiTable_7b[512] =
    {	0	,	1	,	2	,	3	,	4	,	5	,	6	,	6	,	7	,	8	,	9	,	10	,	11	,	11	,	12	,	13	,	14	,	15	,	15	,	16	,	17	,	18	,	18	,	19	,	20	,	20	,	21	,	22	,	22	,	23	,	24	,	24	,	25	,	26	,	26	,	27	,	27	,	28	,	29	,	29	,	30	,	30	,	31	,	31	,	32	,	33	,	33	,	34	,	34	,	35	,	35	,	36	,	36	,	37	,	37	,	38	,	38	,	39	,	39	,	40	,	40	,	41	,	41	,	42	,	42	,	43	,	43	,	44	,	44	,	44	,	45	,	45	,	46	,	46	,	47	,	47	,	47	,	48	,	48	,	49	,	49	,	50	,	50	,	50	,	51	,	51	,	52	,	52	,	52	,	53	,	53	,	53	,	54	,	54	,	55	,	55	,	55	,	56	,	56	,	56	,	57	,	57	,	57	,	58	,	58	,	59	,	59	,	59	,	60	,	60	,	60	,	61	,	61	,	61	,	62	,	62	,	62	,	63	,	63	,	63	,	63	,	64	,	64	,	64	,	65	,	65	,	65	,	66	,	66	,	66	,	67	,	67	,	67	,	67	,	68	,	68	,	68	,	69	,	69	,	69	,	69	,	70	,	70	,	70	,	71	,	71	,	71	,	71	,	72	,	72	,	72	,	73	,	73	,	73	,	73	,	74	,	74	,	74	,	74	,	75	,	75	,	75	,	75	,	76	,	76	,	76	,	76	,	77	,	77	,	77	,	77	,	78	,	78	,	78	,	78	,	79	,	79	,	79	,	79	,	80	,	80	,	80	,	80	,	81	,	81	,	81	,	81	,	81	,	82	,	82	,	82	,	82	,	83	,	83	,	83	,	83	,	83	,	84	,	84	,	84	,	84	,	85	,	85	,	85	,	85	,	85	,	86	,	86	,	86	,	86	,	87	,	87	,	87	,	87	,	87	,	88	,	88	,	88	,	88	,	88	,	89	,	89	,	89	,	89	,	89	,	90	,	90	,	90	,	90	,	90	,	91	,	91	,	91	,	91	,	91	,	92	,	92	,	92	,	92	,	92	,	92	,	93	,	93	,	93	,	93	,	93	,	94	,	94	,	94	,	94	,	94	,	95	,	95	,	95	,	95	,	95	,	95	,	96	,	96	,	96	,	96	,	96	,	96	,	97	,	97	,	97	,	97	,	97	,	98	,	98	,	98	,	98	,	98	,	98	,	99	,	99	,	99	,	99	,	99	,	99	,	100	,	100	,	100	,	100	,	100	,	100	,	101	,	101	,	101	,	101	,	101	,	101	,	102	,	102	,	102	,	102	,	102	,	102	,	102	,	103	,	103	,	103	,	103	,	103	,	103	,	104	,	104	,	104	,	104	,	104	,	104	,	104	,	105	,	105	,	105	,	105	,	105	,	105	,	106	,	106	,	106	,	106	,	106	,	106	,	106	,	107	,	107	,	107	,	107	,	107	,	107	,	107	,	108	,	108	,	108	,	108	,	108	,	108	,	108	,	109	,	109	,	109	,	109	,	109	,	109	,	109	,	110	,	110	,	110	,	110	,	110	,	110	,	110	,	111	,	111	,	111	,	111	,	111	,	111	,	111	,	111	,	112	,	112	,	112	,	112	,	112	,	112	,	112	,	113	,	113	,	113	,	113	,	113	,	113	,	113	,	113	,	114	,	114	,	114	,	114	,	114	,	114	,	114	,	115	,	115	,	115	,	115	,	115	,	115	,	115	,	115	,	116	,	116	,	116	,	116	,	116	,	116	,	116	,	116	,	117	,	117	,	117	,	117	,	117	,	117	,	117	,	117	,	117	,	118	,	118	,	118	,	118	,	118	,	118	,	118	,	118	,	119	,	119	,	119	,	119	,	119	,	119	,	119	,	119	,	119	,	120	,	120	,	120	,	120	,	120	,	120	,	120	,	120	,	121	,	121	,	121	,	121	,	121	,	121	,	121	,	121	,	121	,	122	,	122	,	122	,	122	,	122	,	122	,	122	,	122	,	122	,	123	,	123	,	123	,	123	,	123	,	123	,	123	,	123	,	123	,	124	,	124	,	124	,	124	,	124	,	124	,	124	,	124	,	124	,	125	,	125	,	125	,	125	,	125	,	125	,	125	,	125	,	125	,	125	,	126	,	126	,	126	,	126	,	126	,	126	,	126	,	126	,	126	,	126	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	,	127	};

  // 8-bit table
  int dPhiTable_8b[512] =
    {	0	,	1	,	2	,	3	,	4	,	5	,	6	,	7	,	8	,	9	,	10	,	11	,	12	,	13	,	14	,	14	,	15	,	16	,	17	,	18	,	19	,	20	,	21	,	22	,	23	,	24	,	24	,	25	,	26	,	27	,	28	,	29	,	30	,	31	,	31	,	32	,	33	,	34	,	35	,	36	,	37	,	37	,	38	,	39	,	40	,	41	,	42	,	42	,	43	,	44	,	45	,	46	,	46	,	47	,	48	,	49	,	50	,	50	,	51	,	52	,	53	,	54	,	54	,	55	,	56	,	57	,	57	,	58	,	59	,	60	,	61	,	61	,	62	,	63	,	64	,	64	,	65	,	66	,	66	,	67	,	68	,	69	,	69	,	70	,	71	,	72	,	72	,	73	,	74	,	74	,	75	,	76	,	77	,	77	,	78	,	79	,	79	,	80	,	81	,	81	,	82	,	83	,	83	,	84	,	85	,	86	,	86	,	87	,	88	,	88	,	89	,	90	,	90	,	91	,	91	,	92	,	93	,	93	,	94	,	95	,	95	,	96	,	97	,	97	,	98	,	99	,	99	,	100	,	100	,	101	,	102	,	102	,	103	,	104	,	104	,	105	,	105	,	106	,	107	,	107	,	108	,	109	,	109	,	110	,	110	,	111	,	112	,	112	,	113	,	113	,	114	,	115	,	115	,	116	,	116	,	117	,	117	,	118	,	119	,	119	,	120	,	120	,	121	,	122	,	122	,	123	,	123	,	124	,	124	,	125	,	125	,	126	,	127	,	127	,	128	,	128	,	129	,	129	,	130	,	130	,	131	,	132	,	132	,	133	,	133	,	134	,	134	,	135	,	135	,	136	,	136	,	137	,	138	,	138	,	139	,	139	,	140	,	140	,	141	,	141	,	142	,	142	,	143	,	143	,	144	,	144	,	145	,	145	,	146	,	146	,	147	,	147	,	148	,	148	,	149	,	149	,	150	,	150	,	151	,	151	,	152	,	152	,	153	,	153	,	154	,	154	,	155	,	155	,	156	,	156	,	157	,	157	,	158	,	158	,	159	,	159	,	160	,	160	,	161	,	161	,	162	,	162	,	163	,	163	,	164	,	164	,	165	,	165	,	165	,	166	,	166	,	167	,	167	,	168	,	168	,	169	,	169	,	170	,	170	,	171	,	171	,	172	,	172	,	172	,	173	,	173	,	174	,	174	,	175	,	175	,	176	,	176	,	176	,	177	,	177	,	178	,	178	,	179	,	179	,	180	,	180	,	180	,	181	,	181	,	182	,	182	,	183	,	183	,	183	,	184	,	184	,	185	,	185	,	186	,	186	,	186	,	187	,	187	,	188	,	188	,	189	,	189	,	189	,	190	,	190	,	191	,	191	,	192	,	192	,	192	,	193	,	193	,	194	,	194	,	194	,	195	,	195	,	196	,	196	,	196	,	197	,	197	,	198	,	198	,	199	,	199	,	199	,	200	,	200	,	201	,	201	,	201	,	202	,	202	,	203	,	203	,	203	,	204	,	204	,	204	,	205	,	205	,	206	,	206	,	206	,	207	,	207	,	208	,	208	,	208	,	209	,	209	,	210	,	210	,	210	,	211	,	211	,	211	,	212	,	212	,	213	,	213	,	213	,	214	,	214	,	214	,	215	,	215	,	216	,	216	,	216	,	217	,	217	,	217	,	218	,	218	,	219	,	219	,	219	,	220	,	220	,	220	,	221	,	221	,	221	,	222	,	222	,	223	,	223	,	223	,	224	,	224	,	224	,	225	,	225	,	225	,	226	,	226	,	227	,	227	,	227	,	228	,	228	,	228	,	229	,	229	,	229	,	230	,	230	,	230	,	231	,	231	,	231	,	232	,	232	,	232	,	233	,	233	,	233	,	234	,	234	,	235	,	235	,	235	,	236	,	236	,	236	,	237	,	237	,	237	,	238	,	238	,	238	,	239	,	239	,	239	,	240	,	240	,	240	,	241	,	241	,	241	,	242	,	242	,	242	,	243	,	243	,	243	,	244	,	244	,	244	,	245	,	245	,	245	,	246	,	246	,	246	,	247	,	247	,	247	,	247	,	248	,	248	,	248	,	249	,	249	,	249	,	250	,	250	,	250	,	251	,	251	,	251	,	252	,	252	,	252	,	253	,	253	,	253	,	254	,	254	,	254	,	254	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	,	255	};
  
  // Now set the arrays
  
 // 5-bit words
  for (int i = 0; i < 256; i++)
    {
      sp_2012_01_31_.spvpp_ptu2a_comp_dphi_5[i] = dPhiTable_5b[i];
      sp_2012_01_31_.spvpp_ptu2b_comp_dphi_5[i] = dPhiTable_5b[i];
      sp_2012_01_31_.spvpp_ptu2c_comp_dphi_5[i] = dPhiTable_5b[i];

      sp_2012_01_31_.spvpp_ptu3a_comp_dphi_5[i] = dPhiTable_5b[i];
      sp_2012_01_31_.spvpp_ptu3b_comp_dphi_5[i] = dPhiTable_5b[i];
      sp_2012_01_31_.spvpp_ptu3c_comp_dphi_5[i] = dPhiTable_5b[i];
    }
  
  // 7-bit words
  for (int i = 0; i < 512; i++)
    {
      sp_2012_01_31_.spvpp_ptu2a_comp_dphi_7[i] = dPhiTable_7b[i];
      sp_2012_01_31_.spvpp_ptu2b_comp_dphi_7[i] = dPhiTable_7b[i];
      sp_2012_01_31_.spvpp_ptu2c_comp_dphi_7[i] = dPhiTable_7b[i];

      sp_2012_01_31_.spvpp_ptu3a_comp_dphi_7[i] = dPhiTable_7b[i];
      sp_2012_01_31_.spvpp_ptu3b_comp_dphi_7[i] = dPhiTable_7b[i];
      sp_2012_01_31_.spvpp_ptu3c_comp_dphi_7[i] = dPhiTable_7b[i];
    }
  
   // 8-bit words
  for (int i = 0; i < 512; i++)
    {
      sp_2012_01_31_.spvpp_ptu2a_comp_dphi_8[i] = dPhiTable_8b[i];
      sp_2012_01_31_.spvpp_ptu2b_comp_dphi_8[i] = dPhiTable_8b[i];
      sp_2012_01_31_.spvpp_ptu2c_comp_dphi_8[i] = dPhiTable_8b[i];

      sp_2012_01_31_.spvpp_ptu3a_comp_dphi_8[i] = dPhiTable_8b[i];
      sp_2012_01_31_.spvpp_ptu3b_comp_dphi_8[i] = dPhiTable_8b[i];
      sp_2012_01_31_.spvpp_ptu3c_comp_dphi_8[i] = dPhiTable_8b[i];

      sp_2012_01_31_.spvpp_ptu4a_comp_dphi_8[i] = dPhiTable_8b[i];
      sp_2012_01_31_.spvpp_ptu4b_comp_dphi_8[i] = dPhiTable_8b[i];
      sp_2012_01_31_.spvpp_ptu4c_comp_dphi_8[i] = dPhiTable_8b[i];
    }

}
  

CSCTriggerContainer<csc::L1Track> CSCTFSPCoreLogic::tracks()
{
  return mytracks;
}

//  LocalWords:  isValid
