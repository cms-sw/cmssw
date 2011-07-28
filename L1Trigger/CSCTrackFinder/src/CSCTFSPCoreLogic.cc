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
						}
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
						{
			  			io_[relative_bx+1].me1dVp   = stubi->isValid();
			  			io_[relative_bx+1].me1dQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1dEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1dPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1dAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1dCSCIdp  = stubi->cscid();
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
						}
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
						{
			  			io_[relative_bx+1].me1eVp   = stubi->isValid();
			  			io_[relative_bx+1].me1eQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1eEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1ePhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1eAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1eCSCIdp  = stubi->cscid();
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
						}
		      	if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
						{
			  			io_[relative_bx+1].me1fVp   = stubi->isValid();
			  			io_[relative_bx+1].me1fQp   = stubi->getQuality();
			  			io_[relative_bx+1].me1fEtap = stubi->etaPacked();
			  			io_[relative_bx+1].me1fPhip = stubi->phiPacked();
			  			io_[relative_bx+1].me1fAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);
			  			io_[relative_bx+1].me1fCSCIdp  = stubi->cscid();
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
			      }
			  } else {
			    io_[relative_bx].mb1aVp	  = stubi->getStrip();
			    io_[relative_bx].mb1aQp   = stubi->getQuality();
			    io_[relative_bx].mb1aPhip = stubi->phiPacked();
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
			      }
			  } else {
			    io_[relative_bx].mb1bVp	  = stubi->getStrip();
			    io_[relative_bx].mb1bQp   = stubi->getQuality();
			    io_[relative_bx].mb1bPhip = stubi->phiPacked();
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

    default:
      edm::LogInfo("CSCSTFSPCoreLogic") << "Warning: using the default core is what you want?"
					<< " Core version is " << this->GetCoreFirmwareVersion();
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
	 (m_preTrigger<<7)|(m_allowCLCTonly<<5)|(m_allowALCTonly<<4)|(m_extend_length<<1)|(m_widePhi));
      break;
      
    }

    if ( IsVerbose() ) {
      std::cout << "Core Verbose Output For Debugging\n";
      std::cout << io->me1aVp << " " << io->me1aQp << " " << io->me1aEtap << " " << io->me1aPhip << " " << io->me1aCSCIdp << std::endl;
      std::cout << io->me1bVp << " " << io->me1bQp << " " << io->me1bEtap << " " << io->me1bPhip << " " << io->me1bCSCIdp << std::endl;
      std::cout << io->me1cVp << " " << io->me1cQp << " " << io->me1cEtap << " " << io->me1cPhip << " " << io->me1cCSCIdp << std::endl;
    
      std::cout << io->me1dVp << " " << io->me1dQp << " " << io->me1dEtap << " " << io->me1dPhip << " " << io->me1dCSCIdp << std::endl;
      std::cout << io->me1eVp << " " << io->me1eQp << " " << io->me1eEtap << " " << io->me1ePhip << " " << io->me1eCSCIdp << std::endl;
      std::cout << io->me1fVp << " " << io->me1fQp << " " << io->me1fEtap << " " << io->me1fPhip << " " << io->me1fCSCIdp << std::endl;
      
      std::cout << io->me2aVp << " " << io->me2aQp << " " << io->me2aEtap << " " << io->me2aPhip << " " << 0 << std::endl;
      std::cout << io->me2bVp << " " << io->me2bQp << " " << io->me2bEtap << " " << io->me2bPhip << " " << 0 << std::endl;
      std::cout << io->me2cVp << " " << io->me2cQp << " " << io->me2cEtap << " " << io->me2cPhip << " " << 0 << std::endl;
      
      std::cout << io->me3aVp << " " << io->me3aQp << " " << io->me3aEtap << " " << io->me3aPhip << " " << 0 << std::endl;
      std::cout << io->me3bVp << " " << io->me3bQp << " " << io->me3bEtap << " " << io->me3bPhip << " " << 0 << std::endl;
      std::cout << io->me3cVp << " " << io->me3cQp << " " << io->me3cEtap << " " << io->me3cPhip << " " << 0 << std::endl;
      
      std::cout << io->me4aVp << " " << io->me4aQp << " " << io->me4aEtap << " " << io->me4aPhip << " " << 0 << std::endl;
      std::cout << io->me4bVp << " " << io->me4bQp << " " << io->me4bEtap << " " << io->me4bPhip << " " << 0 << std::endl;
      std::cout << io->me4cVp << " " << io->me4cQp << " " << io->me4cEtap << " " << io->me4cPhip << " " << 0 << std::endl;
      
      std::cout << io->mb1aVp << " " << io->mb1aQp << " " << 0 << " " << io->mb1aPhip <<" " << 0 << std::endl;
      std::cout << io->mb1bVp << " " << io->mb1bQp << " " << 0 << " " << io->mb1aPhip <<" " << 0 << std::endl;
      std::cout << io->mb1cVp << " " << io->mb1cQp << " " << 0 << " " << io->mb1aPhip <<" " << 0 << std::endl;
      std::cout << io->mb1dVp << " " << io->mb1dQp << " " << 0 << " " << io->mb1aPhip <<" " << 0 << std::endl;
      
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

CSCTriggerContainer<csc::L1Track> CSCTFSPCoreLogic::tracks()
{
  return mytracks;
}

//  LocalWords:  isValid
