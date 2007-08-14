#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

const std::string CSCTFSectorProcessor::FPGAs[5] = {"F1","F2","F3","F4","F5"};

///KK
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

//KK
  m_etamin[0] = 11*2;
  m_etamin[1] = 11*2;
  m_etamin[2] = 7*2;
  m_etamin[3] = 7*2;
  m_etamin[4] = 7*2;
  m_etamin[5] = 5*2;
  m_etamin[6] = 5*2;
  m_etamin[7] = 5*2;
  std::vector<unsigned> etamins = pset.getUntrackedParameter<std::vector<unsigned> >("EtaMin",std::vector<unsigned>(0));
  for(iter=etamins.begin(),i=0; iter!=etamins.end(); iter++,i++) m_etamin[i] = *iter;

  m_etamax[0] = 127;
  m_etamax[1] = 127;
  m_etamax[2] = 127;
  m_etamax[3] = 127;
  m_etamax[4] = 127;
  m_etamax[5] = 12*2;
  m_etamax[6] = 12*2;
  m_etamax[7] = 12*2;
  std::vector<unsigned> etamaxs = pset.getUntrackedParameter<std::vector<unsigned> >("EtaMax",std::vector<unsigned>(0));
  for(iter=etamaxs.begin(),i=0; iter!=etamaxs.end(); iter++,i++) m_etamax[i] = *iter;
//KK end

  try {
    edm::ParameterSet srLUTset = pset.getParameter<edm::ParameterSet>("SRLUT");
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
      LogDebug("CSCTFSectorProcessor") << "Using stand-alone SR LUT for endcap="<<m_endcap<<", sector="<<m_sector;
  } catch(...) {
    LogDebug("CSCTFSectorProcessor") << "Using SR LUT from EventSetup for endcap="<<m_endcap<<", sector="<<m_sector;
  }

  core_ = new CSCTFSPCoreLogic();

  try {
    edm::ParameterSet ptLUTset = pset.getParameter<edm::ParameterSet>("PTLUT");
	ptLUT_ = new CSCTFPtLUT(ptLUTset);
    LogDebug("CSCTFSectorProcessor") << "Using stand-alone PT LUT for endcap="<<m_endcap<<", sector="<<m_sector;
  } catch(...){
    ptLUT_=0;
    LogDebug("CSCTFSectorProcessor") << "Using PT LUT from EventSetup for endcap="<<m_endcap<<", sector="<<m_sector;
  }
}
///
///KK
void CSCTFSectorProcessor::initialize(const edm::EventSetup& c){
  for(int i = 1; i <= 4; ++i)
    {
      if(i == 1)
	for(int j = 0; j < 2; j++)
	  {
		  if(!srLUTs_[FPGAs[j]]){
			  LogDebug("CSCTFSectorProcessor") << "Initializing SR LUT for endcap="<<m_endcap<<", station=1, sector="<<m_sector<<", sub_sector="<<j<<" from EventSetup";
			  srLUTs_[FPGAs[j]] = new CSCSectorReceiverLUT(m_endcap, m_sector, j+1, i, c);
		  }
	  }
      else
		  if(!srLUTs_[FPGAs[i]]){
			  LogDebug("CSCTFSectorProcessor") << "Initializing SR LUT for endcap="<<m_endcap<<", station="<<i<<", sector="<<m_sector<<" from EventSetup";
			  srLUTs_[FPGAs[i]] = new CSCSectorReceiverLUT(m_endcap, m_sector, 0, i, c);
		  }
    }

  if(!ptLUT_){
	  LogDebug("CSCTFSectorProcessor") << "Initializing PT LUT from EventSetup";
	  ptLUT_ = new CSCTFPtLUT(c);
  }
}
///
///KK
CSCTFSectorProcessor::~CSCTFSectorProcessor()
{
  for(int i = 0; i < 5; ++i)
    {
      if(srLUTs_[FPGAs[i]]) delete srLUTs_[FPGAs[i]]; // delete the pointer
      srLUTs_[FPGAs[i]] = NULL; // point it at a safe place
    }

  delete core_;
  core_ = NULL;

  if(ptLUT_) delete ptLUT_;
  ptLUT_ = NULL;
}
///

bool CSCTFSectorProcessor::run(const CSCTriggerContainer<csctf::TrackStub>& stubs)
{
///KK
  if( !ptLUT_ )
    throw cms::Exception("Initialize CSC TF LUTs first (missed call to CSCTFTrackProducer::beginJob?)")<<"CSCTFSectorProcessor::run";
  for(int i = 0; i < 5; ++i)
    if(!srLUTs_[FPGAs[i]])
		throw cms::Exception("Initialize CSC TF LUTs first (missed call to CSCTFTrackProducer::beginJob?)")<<"CSCTFSectorProcessor::run";
///

  l1_tracks.clear();
  dt_stubs.clear();

  /** STEP ONE
   *  We take stubs from the MPC and assign their eta and phi
   *  coordinates using the SR Lookup tables.
   *  This is independent of what BX we are on so we can
   *  process one large vector of stubs.
   *  After this we append the stubs gained from the DT system.
   */

  std::vector<csctf::TrackStub> stub_vec = stubs.get();
  std::vector<csctf::TrackStub>::iterator itr = stub_vec.begin();
  std::vector<csctf::TrackStub>::const_iterator end = stub_vec.end();

  for(; itr != end; itr++)
    {
      if(itr->station() != 5)
	{
	  CSCDetId id(itr->getDetId().rawId());
	  unsigned fpga = (id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station();

	  lclphidat lclPhi = srLUTs_[FPGAs[fpga]]->localPhi(itr->getStrip(), itr->getPattern(), itr->getQuality(), itr->getBend());
	  gblphidat gblPhi = srLUTs_[FPGAs[fpga]]->globalPhiME(lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
	  gbletadat gblEta = srLUTs_[FPGAs[fpga]]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
	  itr->setEtaPacked(gblEta.global_eta);
	  itr->setPhiPacked(gblPhi.global_phi);

	  if(itr->station() == 1) dt_stubs.push_back(*itr); // send stubs to DT

	  LogDebug("CSCTFSectorProcessor:run()") << "LCT found, processed by FPGA: " << FPGAs[fpga] << std::endl
						 << " LCT now has (eta, phi) of: (" << itr->etaValue() << "," << itr->phiValue() <<")\n";
	}
    }

  CSCTriggerContainer<csctf::TrackStub> processedStubs(stub_vec);

  /** STEP TWO
   *  We take the stubs filled by the SR LUTs and load them
   *  for processing into the SP core logic.
   *  After loading we run and then retrieve any tracks generated.
   */

  std::vector<csc::L1Track> tftks;

  core_->loadData(processedStubs, m_endcap, m_sector, m_minBX, m_maxBX);

  if( core_->run(m_endcap, m_sector, m_latency, m_etawin[0],
		 m_etamin[0], m_etamin[1], m_etamin[2], m_etamin[3],
		 m_etamin[4], m_etamin[5], m_etamin[6], m_etamin[7],
		 m_etamax[0], m_etamax[1], m_etamax[2], m_etamax[3],
		 m_etamax[4], m_etamax[5], m_etamax[6], m_etamax[7],
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

