#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

const std::string CSCTFSectorProcessor::FPGAs[5] = {"F1","F2","F3","F4","F5"};

CSCTFSectorProcessor::CSCTFSectorProcessor(const unsigned& endcap, 
					   const unsigned& sector,
					   const edm::ParameterSet& pset)
{
  m_endcap = endcap;
  m_sector = sector;
  m_bxa_on = pset.getUntrackedParameter<bool>("UseBXA",true);
  m_extend_length = pset.getUntrackedParameter<unsigned>("BXAExtendLength",1);
  m_latency = pset.getUntrackedParameter<unsigned>("CoreLatency",6);
  m_minBX = pset.getUntrackedParameter<int>("MinBX",-3);
  m_maxBX = pset.getUntrackedParameter<int>("MaxBX",3);

  edm::ParameterSet srLUTset = pset.getParameter<edm::ParameterSet>("SRLUT");
  edm::ParameterSet ptLUTset = pset.getParameter<edm::ParameterSet>("PTLUT");

  int i = 0;
  for(; i < 6; ++i)
    m_etawin[i] = 2;

  std::vector<unsigned> etawins = pset.getUntrackedParameter<std::vector<unsigned> >("EtaWindows");
  std::vector<unsigned>::const_iterator iter = etawins.begin();
  
  i = 0;
  for(; iter != etawins.end(); iter++)
    {
      m_etawin[i] = *iter;
      ++i;
    }
 
  for(i = 1; i <= 4; ++i)
    {
      if(i == 1)
	for(int j = 0; j < 2; j++)
	  {
	    srLUTs_[FPGAs[j]] = new CSCSectorReceiverLUT(endcap, sector, j+1, i, srLUTset);
	  }
      else
	srLUTs_[FPGAs[i]] = new CSCSectorReceiverLUT(endcap, sector, 0, i, srLUTset);
    }

  core_ = new CSCTFSPCoreLogic();

  ptLUT_ = new CSCTFPtLUT(ptLUTset);
}

CSCTFSectorProcessor::~CSCTFSectorProcessor()
{
  for(int i = 0; i < 5; ++i)
    {      
      delete srLUTs_[FPGAs[i]]; // delete the pointer
      srLUTs_[FPGAs[i]] = NULL; // point it at a safe place
    }

  delete core_;
  core_ = NULL;

  delete ptLUT_;
  ptLUT_ = NULL;  
}

bool CSCTFSectorProcessor::run(const CSCTriggerContainer<CSCTrackStub>& stubs)
{
  l1_tracks.clear();
  dt_stubs.clear();
  
  /** STEP ONE
   *  We take stubs from the MPC and assign their eta and phi
   *  coordinates using the SR Lookup tables.
   *  This is independent of what BX we are on so we can
   *  process one large vector of stubs.
   *  After this we append the stubs gained from the DT system.
   */
    
  std::vector<CSCTrackStub> stub_vec = stubs.get();
  std::vector<CSCTrackStub>::iterator itr = stub_vec.begin();
  std::vector<CSCTrackStub>::const_iterator end = stub_vec.end();

  for(; itr != end; itr++)
    {
      CSCDetId id = itr->getDetId();
      unsigned fpga = (id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station();
      
      lclphidat lclPhi = srLUTs_[FPGAs[fpga]]->localPhi(itr->getStrip(), itr->getCLCTPattern(), itr->getQuality(), itr->getBend());
      gblphidat gblPhi = srLUTs_[FPGAs[fpga]]->globalPhiME(lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
      gbletadat gblEta = srLUTs_[FPGAs[fpga]]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
      itr->setEtaPacked(gblEta.global_eta);
      itr->setPhiPacked(gblPhi.global_phi);

      LogDebug("CSCTFSectorProcessor:run()") << "LCT found, processed by FPGA: " << FPGAs[fpga] << std::endl
					     << " LCT now has (eta, phi) of: (" << itr->etaValue() << "," << itr->phiValue() <<")\n";
    }

  CSCTriggerContainer<CSCTrackStub> processedStubs(stub_vec);

  //Add stubs to be sent to DTTF.
  for(int e = CSCDetId::minEndcapId(); e <= CSCDetId::maxEndcapId(); ++e)
    for(int s = CSCTriggerNumbering::minTriggerSectorId(); s <= CSCTriggerNumbering::maxTriggerSectorId(); ++s)
      for(int ss = CSCTriggerNumbering::minTriggerSubSectorId(); 
	  ss <= CSCTriggerNumbering::maxTriggerSubSectorId(); ++ss)
	for(int bx = m_minBX; bx <= m_maxBX; ++bx)
	  dt_stubs.push_many(processedStubs.get(e,1,s,ss,bx));    


  /** STEP TWO
   *  We take the stubs filled by the SR LUTs and load them
   *  for processing into the SP core logic.
   *  After loading we run and then retrieve any tracks generated.
   */

  std::vector<csc::L1Track> tftks;

  core_->loadData(processedStubs, m_endcap, m_sector, m_minBX, m_maxBX);

  if( core_->run(m_endcap, m_sector, m_latency, m_etawin[0],
		 m_etawin[1], m_etawin[2], m_etawin[3],
		 m_etawin[4], m_etawin[5], m_bxa_on,
		 m_extend_length, m_minBX, m_maxBX) )
    {
      l1_tracks = core_->tracks();
    }
  
  tftks = l1_tracks.get();

  /** STEP THREE
   *  Now that we have the found tracks from the core,
   *  we must assign their Pt. 
   */

  std::vector<csc::L1Track>::iterator titr = tftks.begin();

  for(; titr != tftks.end(); titr++)
    {
      ptadd thePtAddress(titr->ptLUTAddress());
      ptdat thePtData = ptLUT_->Pt(thePtAddress);

      if(thePtAddress.track_fr)
	{
	  titr->setRank(thePtData.front_rank);
	  titr->setChargeValidPacked(thePtData.charge_valid_front);
	}
      else
	{
	  titr->setRank(thePtData.rear_rank);
	  titr->setChargeValidPacked(thePtData.charge_valid_rear);
	}
    }
  
  l1_tracks = tftks;

  return (l1_tracks.get().size() > 0);
}

