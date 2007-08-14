#include <L1Trigger/CSCTrackFinder/interface/CSCTFSPCoreLogic.h>
#include <L1Trigger/CSCTrackFinder/src/SPvpp.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <L1Trigger/CSCTrackFinder/src/spbits.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

SPvpp CSCTFSPCoreLogic::sp_;


// takes a trigger container and loads the first n bx of data into io_ 
void CSCTFSPCoreLogic::loadData(const CSCTriggerContainer<csctf::TrackStub>& theStubs, 
				const unsigned& endcap, const unsigned& sector, 
				const int& minBX, const int& maxBX)
{
  io_.clear();
  runme = 0;
  io_.resize(maxBX - minBX + 1);
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
			  io_[relative_bx].me1aVp   = stubi->isValid();
			  io_[relative_bx].me1aQp   = stubi->getQuality(); 
			  io_[relative_bx].me1aEtap = stubi->etaPacked(); 
			  io_[relative_bx].me1aPhip = stubi->phiPacked();  
			  io_[relative_bx].me1aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
			  io_[relative_bx].me1aCSCIdp  = stubi->cscid();
			}
		      if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
			{
			  io_[relative_bx].me1dVp   = stubi->isValid();
			  io_[relative_bx].me1dQp   = stubi->getQuality(); 
			  io_[relative_bx].me1dEtap = stubi->etaPacked(); 
			  io_[relative_bx].me1dPhip = stubi->phiPacked();  
			  io_[relative_bx].me1dAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
			  io_[relative_bx].me1dCSCIdp  = stubi->cscid();
			}
		      break;		   
		    case 2:
		      if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 1)
			{
			  io_[relative_bx].me1bVp   = stubi->isValid();
			  io_[relative_bx].me1bQp   = stubi->getQuality(); 
			  io_[relative_bx].me1bEtap = stubi->etaPacked(); 
			  io_[relative_bx].me1bPhip = stubi->phiPacked();  
			  io_[relative_bx].me1bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
			io_[relative_bx].me1bCSCIdp  = stubi->cscid();
			}
		      if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
			{
			  io_[relative_bx].me1eVp   = stubi->isValid();
			  io_[relative_bx].me1eQp   = stubi->getQuality(); 
			  io_[relative_bx].me1eEtap = stubi->etaPacked(); 
			  io_[relative_bx].me1ePhip = stubi->phiPacked();  
			  io_[relative_bx].me1eAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
			  io_[relative_bx].me1eCSCIdp  = stubi->cscid();
			}
		      break;
		    case 3:
		      if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 1)
			{
			  io_[relative_bx].me1cVp   = stubi->isValid();
			  io_[relative_bx].me1cQp   = stubi->getQuality(); 
			  io_[relative_bx].me1cEtap = stubi->etaPacked(); 
			  io_[relative_bx].me1cPhip = stubi->phiPacked();  
			  io_[relative_bx].me1cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
			  io_[relative_bx].me1cCSCIdp  = stubi->cscid();
			}
		      if(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(stubi->getDetId().rawId())) == 2)
			{
			  io_[relative_bx].me1fVp   = stubi->isValid();
			  io_[relative_bx].me1fQp   = stubi->getQuality(); 
			  io_[relative_bx].me1fEtap = stubi->etaPacked(); 
			  io_[relative_bx].me1fPhip = stubi->phiPacked();  
			  io_[relative_bx].me1fAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
			  io_[relative_bx].me1fCSCIdp  = stubi->cscid();
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
		      io_[relative_bx].me2aVp   = stubi->isValid();
		      io_[relative_bx].me2aQp   = stubi->getQuality(); 
		      io_[relative_bx].me2aEtap = stubi->etaPacked(); 
		      io_[relative_bx].me2aPhip = stubi->phiPacked();  
		      io_[relative_bx].me2aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  		    
		      break;
		    case 2:
		      io_[relative_bx].me2bVp   = stubi->isValid();
		      io_[relative_bx].me2bQp   = stubi->getQuality(); 
		      io_[relative_bx].me2bEtap = stubi->etaPacked(); 
		      io_[relative_bx].me2bPhip = stubi->phiPacked();  
		      io_[relative_bx].me2bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
		      break;
		    case 3:
		      io_[relative_bx].me2cVp   = stubi->isValid();
		      io_[relative_bx].me2cQp   = stubi->getQuality(); 
		      io_[relative_bx].me2cEtap = stubi->etaPacked(); 
		      io_[relative_bx].me2cPhip = stubi->phiPacked();  
		      io_[relative_bx].me2cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
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
		      io_[relative_bx].me3aVp   = stubi->isValid();
		      io_[relative_bx].me3aQp   = stubi->getQuality(); 
		      io_[relative_bx].me3aEtap = stubi->etaPacked(); 
		      io_[relative_bx].me3aPhip = stubi->phiPacked();  
		      io_[relative_bx].me3aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
		      break;
		    case 2:
		      io_[relative_bx].me3bVp   = stubi->isValid();
		      io_[relative_bx].me3bQp   = stubi->getQuality(); 
		      io_[relative_bx].me3bEtap = stubi->etaPacked(); 
		      io_[relative_bx].me3bPhip = stubi->phiPacked();  
		      io_[relative_bx].me3bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
		      break;
		    case 3:
		      io_[relative_bx].me3cVp   = stubi->isValid();
		      io_[relative_bx].me3cQp   = stubi->getQuality(); 
		      io_[relative_bx].me3cEtap = stubi->etaPacked(); 
		      io_[relative_bx].me3cPhip = stubi->phiPacked();  
		      io_[relative_bx].me3cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
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
		      io_[relative_bx].me4aVp   = stubi->isValid();
		      io_[relative_bx].me4aQp   = stubi->getQuality(); 
		      io_[relative_bx].me4aEtap = stubi->etaPacked(); 
		      io_[relative_bx].me4aPhip = stubi->phiPacked();  
		      io_[relative_bx].me4aAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
		      break;
		    case 2:
		      io_[relative_bx].me4bVp   = stubi->isValid();
		      io_[relative_bx].me4bQp   = stubi->getQuality(); 
		      io_[relative_bx].me4bEtap = stubi->etaPacked(); 
		      io_[relative_bx].me4bPhip = stubi->phiPacked();  
		      io_[relative_bx].me4bAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
		      break;
		    case 3:
		      io_[relative_bx].me4cVp   = stubi->isValid();
		      io_[relative_bx].me4cQp   = stubi->getQuality(); 
		      io_[relative_bx].me4cEtap = stubi->etaPacked(); 
		      io_[relative_bx].me4cPhip = stubi->phiPacked();  
		      io_[relative_bx].me4cAmp  = (stubi->getQuality() == 1 || stubi->getQuality() == 2);  
		      break;
		    default:
		      edm::LogWarning("CSCTFSPCoreLogic::loadData()") << "SERIOUS ERROR: MPC LINK " << stubi->getMPCLink() 
								     << " NOT IN RANGE [1,3]\n";
		    };
		  break;
		case 5:
		  switch(stubi->getMPCLink())
		    {
		    case 1:
		      io_[relative_bx].mb1aVp   = stubi->isValid();
		      io_[relative_bx].mb1aQp   = stubi->getQuality();
		      io_[relative_bx].mb1aPhip = stubi->phiPacked();
		      break;
		    case 2:
		      io_[relative_bx].mb1bVp   = stubi->isValid();
                      io_[relative_bx].mb1bQp   = stubi->getQuality();
                      io_[relative_bx].mb1bPhip = stubi->phiPacked();
                      break;
		    case 3:
		      io_[relative_bx].mb1cVp   = stubi->isValid();
                      io_[relative_bx].mb1cQp   = stubi->getQuality();
                      io_[relative_bx].mb1cPhip = stubi->phiPacked();
                      break;
		    case 4:
		      io_[relative_bx].mb1dVp   = stubi->isValid();
                      io_[relative_bx].mb1dQp   = stubi->getQuality();
                      io_[relative_bx].mb1dPhip = stubi->phiPacked();
                      break;
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
			   const unsigned& etawin4, const unsigned& etawin5, const unsigned& etawin6,
			   const unsigned& bxa_on, const unsigned& extend, const int& minBX, 
			   const int& maxBX)
{
  mytracks.clear();

  int train_length = io_.size();
  int bx = 0;
  io_.resize(train_length + latency);
  std::vector<SPio>::iterator io;
  
  // run over enough clock cycles to get tracks from input stubs.
  for( io = io_.begin(); io != io_.end() && runme; io++) 
    {		
      sp_.SP
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
	 
	 io->mb2aVp, io->mb2aQp, io->mb2aPhip,                                   
	 io->mb2bVp, io->mb2bQp, io->mb2bPhip,                                   
	 io->mb2cVp, io->mb2cQp, io->mb2cPhip,                                   
	 io->mb2dVp, io->mb2dQp, io->mb2dPhip,                                   
	 
	 io->ptHp, io->signHp, io->modeMemHp, io->etaPTHp, io->FRHp, io->phiHp,
	 io->ptMp, io->signMp, io->modeMemMp, io->etaPTMp, io->FRMp, io->phiMp,
	 io->ptLp, io->signLp, io->modeMemLp, io->etaPTLp, io->FRLp, io->phiLp,
	 
	 io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH, io->mb2idH,
	 io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM, io->mb2idM,
	 io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL, io->mb2idL,
	 
	 // Adjustable registers in SP core
	 etamin1, etamin2, etamin3, etamin4, etamin5, etamin6, etamin7, etamin8, // etamin (was: 11*2, 11*2, 7*2, 7*2, 7*2, 5*2,  5*2,  5*2)
	 etamax1, etamax2, etamax3, etamax4, etamax5, etamax6, etamax7, etamax8, // etamax (was: 127,  127,  127, 127, 127, 12*2, 12*2, 12*2)
	 //DEA: beam test settings:
	 //10,  10,  10, 10, 10, 10,              // etawindow
	 //10, 0, 0, 0,                           // eta offsets - NOTE bug in first offset for June04 beam test
	 // ORCA settings:
	 etawin1, etawin2, etawin3, etawin4, etawin5, etawin6,// eta windows
	 0, 0, 0, 0, // eta offsets
	 ((extend << 1) & 0xe)|bxa_on // {reserved[11:0], extend[2:0],BXA_enable}
	 );
      /* // Extremely verbose debug
      LogDebug("CSCTFSPCoreLogic:run()") << std::hex 
					 << "Input:  F1/M1{bx, v, q, e, p, csc} " << std::dec << (int)(bx)<< std::hex << " " << io->me1aVp << " "
					 << io->me1aQp << " "<< io->me1aEtap << " " << io->me1aPhip << " " << io->me1aCSCIdp << std::endl
					 << "Input:  F1/M2{bx, v, q, e, p, csc} " << std::dec << (int)(bx)<< std::hex << " " << io->me1bVp << " "
					 << io->me1bQp << " "<< io->me1bEtap << " " << io->me1bPhip << " " << io->me1bCSCIdp << std::endl
					 << "Input:  F1/M3{bx, v, q, e, p, csc} " << std::dec << (int)(bx)<< std::hex << " " << io->me1cVp << " "
					 << io->me1cQp << " "<< io->me1cEtap << " " << io->me1cPhip << " " << io->me1cCSCIdp << std::endl
					 << "Input:  F2/M1{bx, v, q, e, p, csc} " << std::dec << (int)(bx)<< std::hex << " " << io->me1dVp << " "
					 << io->me1dQp << " "<< io->me1dEtap << " " << io->me1dPhip << " " << io->me1dCSCIdp << std::endl
					 << "Input:  F2/M2{bx, v, q, e, p, csc} " << std::dec << (int)(bx)<< std::hex << " " << io->me1eVp << " "
					 << io->me1eQp << " "<< io->me1eEtap << " " << io->me1ePhip << " " << io->me1eCSCIdp << std::endl
					 << "Input:  F2/M3{bx, v, q, e, p, csc} " << std::dec << (int)(bx)<< std::hex << " " << io->me1fVp << " "
					 << io->me1fQp << " "<< io->me1fEtap << " " << io->me1fPhip << " " << io->me1fCSCIdp << std::endl
					 << "Input:  F3/M1{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me2aVp << " "
					 << io->me2aQp << " "<< io->me2aEtap << " " << io->me2aPhip << " " << std::endl
					 << "Input:  F3/M2{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me2bVp << " "
					 << io->me2bQp << " "<< io->me2bEtap << " " << io->me2bPhip << " " << std::endl
					 << "Input:  F3/M3{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me2cVp << " "
					 << io->me2cQp << " "<< io->me2cEtap << " " << io->me2cPhip << " " << std::endl
					 << "Input:  F4/M1{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me3aVp << " "
					 << io->me3aQp << " "<< io->me3aEtap << " " << io->me3aPhip << " " << std::endl
					 << "Input:  F4/M2{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me3bVp << " "
					 << io->me3bQp << " "<< io->me3bEtap << " " << io->me3bPhip << " " << std::endl
					 << "Input:  F4/M3{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me3cVp << " "
					 << io->me3cQp << " "<< io->me3cEtap << " " << io->me3cPhip << " " << std::endl
					 << "Input:  F5/M1{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me4aVp << " "
					 << io->me4aQp << " "<< io->me4aEtap << " " << io->me4aPhip << " " << std::endl
					 << "Input:  F5/M2{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me4bVp << " "
					 << io->me4bQp << " "<< io->me4bEtap << " " << io->me4bPhip << " " << std::endl
					 << "Input:  F5/M3{bx, v, q, e, p}      " << std::dec << (int)(bx)<< std::hex << " " << io->me4cVp << " "
					 << io->me4cQp << " "<< io->me4cEtap << " " << io->me4cPhip << " " << std::endl
					 << "Input:  MB 1A{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb1aVp << " "
					 << io->mb1aQp << " "<< io->mb1aPhip << " " << std::endl
					 << "Input:  MB 1B{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb1bVp << " "
					 << io->mb1bQp << " "<< io->mb1bPhip << " " << std::endl
					 << "Input:  MB 1C{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb1cVp << " "
					 << io->mb1cQp << " "<< io->mb1cPhip << " " << std::endl
					 << "Input:  MB 1D{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb1dVp << " "
					 << io->mb1dQp << " "<< io->mb1dPhip << " " << std::endl
					 << "Input:  MB 2A{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb2aVp << " "
					 << io->mb2aQp << " "<< io->mb2aPhip << " " << std::endl
					 << "Input:  MB 2B{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb2bVp << " "
					 << io->mb2bQp << " "<< io->mb2bPhip << " " << std::endl
					 << "Input:  MB 2C{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb2cVp << " "
					 << io->mb2cQp << " "<< io->mb2cPhip << " " << std::endl
					 << "Input:  MB 2D{bx, v, q, p}         " << std::dec << (int)(bx) << std::hex << " " << io->mb2dVp << " "
					 << io->mb2dQp << " "<< io->mb2dPhip << " " << std::endl
					 << std::dec << std::endl ;

      if(io->ptHp !=0 || io->ptMp !=0 || io->ptLp !=0)
	LogDebug("CSCTFSPCoreLogic:run()")<<"ENDCAP/SECTOR "<< endcap << "/" << sector << std::endl;
      if(io->ptHp !=0)
	{
	  LogDebug("CSCTFSPCoreLogic:run()") << std::hex << "Output M1: " << std::dec << (int)(bx-latency)<< std::hex << " " << io->ptHp << "/" << io->signHp << "/"
					     <<  io->modeMemHp << "/" <<  io->etaPTHp << "/" <<  io->FRHp << "/" <<  io->phiHp<<std::endl
					     << "Stubs Used ME1/ME2/ME3/ME4/MB1/MB2 : " 
					     << io->me1idH <<'/'<< io->me2idH <<'/'<< io->me3idH <<'/'<< io->me4idH <<'/'
					     << io->mb1idH <<'/'<< io->mb2idH << std::dec << std::endl;
	}
      if(io->ptMp !=0)
	{
	  LogDebug("CSCTFSPCoreLogic:run()") << std::hex << "Output M2: " << std::dec << (int)(bx-latency)<< std::hex << " " << io->ptMp << "/" << io->signMp << "/"
					     <<  io->modeMemMp << "/" <<  io->etaPTMp << "/" <<  io->FRMp << "/" <<  io->phiMp << std::endl
					     << "Stubs Used ME1/ME2/ME3/ME4/MB1/MB2 : " 
					     << io->me1idM <<'/'<< io->me2idM <<'/'<< io->me3idM <<'/'<< io->me4idM <<'/'
					     << io->mb1idM <<'/'<< io->mb2idM << std::dec << std::endl;
	}
      if(io->ptLp !=0)
	{
	  LogDebug("CSCTFSPCoreLogic:run()") << std::hex << "Output M3: " << std::dec << (int)(bx-latency)<< std::hex << " " << io->ptLp << "/" << io->signLp << "/"
					     <<  io->modeMemLp << "/" <<  io->etaPTLp << "/" <<  io->FRLp << "/" <<  io->phiLp << std::endl
					     << "Stubs Used ME1/ME2/ME3/ME4/MB1/MB2 : " 
					     << io->me1idL <<'/'<< io->me2idL <<'/'<< io->me3idL <<'/'<< io->me4idL <<'/'
					     << io->mb1idL <<'/'<< io->mb2idL << std::dec << std::endl;
	}
      */
      ++bx;
    }
  
  bx = 0;

  // start from where tracks could first possibly appear
  // read out tracks from io_
  for(io = io_.begin() + latency; io != io_.end(); io++)
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
      LUTAddressH.delta_phi_sign = (io->ptHp >> (BWPT-1)) & 0x1;
      LUTAddressH.track_fr       = io->FRHp & 0x1;

      LUTAddressM.delta_phi_12   = io->ptMp & 0xff;
      LUTAddressM.delta_phi_23   = (io->ptMp >> 8) & 0xf;
      LUTAddressM.track_eta      = (io->etaPTMp>>1) & 0xf;
      LUTAddressM.track_mode     = io->modeMemMp & 0xf;
      LUTAddressM.delta_phi_sign = (io->ptMp >> (BWPT-1)) & 0x1;
      LUTAddressM.track_fr       = io->FRMp & 0x1;
      
      LUTAddressL.delta_phi_12   = io->ptLp & 0xff;
      LUTAddressL.delta_phi_23   = (io->ptLp >> 8) & 0xf;
      LUTAddressL.track_eta      = (io->etaPTLp>>1) & 0xf;
      LUTAddressL.track_mode     = io->modeMemLp & 0xf;
      LUTAddressL.delta_phi_sign = (io->ptLp >> (BWPT-1)) & 0x1;
      LUTAddressL.track_fr       = io->FRLp & 0x1;
    
      
      if(LUTAddressH.toint()) 
	{	  	  
	  trkH.setPtLUTAddress(LUTAddressH.toint());
	  trkH.setChargePacked(~(io->signHp)&0x1);
	  trkH.setLocalPhi(io->phiHp);
	  trkH.setEtaPacked(io->etaPTHp);
	  trkH.setBx((int)(bx)+minBX - latency);
	  trkH.setStationIds(io->me1idH, io->me2idH, io->me3idH, io->me4idH, io->mb1idH);
	  trkH.m_output_link = 1;
	  mytracks.push_back(trkH);	  
	}
      if(LUTAddressM.toint())
	{
	  trkM.setPtLUTAddress(LUTAddressM.toint());
	  trkM.setChargePacked(~(io->signMp)&0x1);
	  trkM.setLocalPhi(io->phiMp);
	  trkM.setEtaPacked(io->etaPTMp);
	  trkM.setBx((int)(bx)+minBX - latency);
	  trkM.setStationIds(io->me1idM, io->me2idM, io->me3idM, io->me4idM, io->mb1idM);
	  trkM.m_output_link = 2;
	  mytracks.push_back(trkM);
	}
      if(LUTAddressL.toint())
	{
	  trkL.setPtLUTAddress(LUTAddressL.toint());
	  trkL.setChargePacked(~(io->signLp)&0x1);
	  trkL.setLocalPhi(io->phiLp);
	  trkL.setEtaPacked(io->etaPTLp);
	  trkL.setBx((int)(bx)+minBX - latency);
	  trkL.setStationIds(io->me1idL, io->me2idL, io->me3idL, io->me4idL, io->mb1idL);
	  trkL.m_output_link = 3;
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
