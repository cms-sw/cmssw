#include <L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "CondFormats/DataRecord/interface/L1MuCSCTFConfigurationRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <stdlib.h>
#include <sstream>
#include <strings.h>

const std::string CSCTFSectorProcessor::FPGAs[5] = {"F1","F2","F3","F4","F5"};

CSCTFSectorProcessor::CSCTFSectorProcessor(const unsigned& endcap,
					   const unsigned& sector,
					   const edm::ParameterSet& pset,
					   bool tmb07,
					   const L1MuTriggerScales* scales,
					   const L1MuTriggerPtScale* ptScale)
{
  m_endcap = endcap;
  m_sector = sector;
  TMB07    = tmb07;
  
  // allows a configurable option to handle unganged ME1a
  m_gangedME1a = pset.getUntrackedParameter<bool>("gangedME1a", true);

  // Parameter below should always present in ParameterSet:
  m_latency = pset.getParameter<unsigned>("CoreLatency");
  m_minBX = pset.getParameter<int>("MinBX");
  m_maxBX = pset.getParameter<int>("MaxBX");
  initializeFromPSet = pset.getParameter<bool>("initializeFromPSet");
  if( m_maxBX-m_minBX >= 7 ) edm::LogWarning("CSCTFTrackBuilder::ctor")<<" BX window width >= 7BX. Resetting m_maxBX="<<(m_maxBX=m_minBX+6);

  // All following parameters may appear in either ParameterSet of in EventSetup; uninitialize:
  m_bxa_depth = -1;
  m_allowALCTonly = -1;
  m_allowCLCTonly = -1;
  m_preTrigger = -1;

  for(int index=0; index<7; index++) m_etawin[index] = -1;
  for(int index=0; index<8; index++) m_etamin[index] = -1;
  for(int index=0; index<8; index++) m_etamax[index] = -1;

  m_mindphip=-1;
  m_mindetap=-1;

  m_mindeta12_accp=-1;
  m_maxdeta12_accp=-1;
  m_maxdphi12_accp=-1;

  m_mindeta13_accp=-1;
  m_maxdeta13_accp=-1;
  m_maxdphi13_accp=-1;

  m_mindeta112_accp=-1;
  m_maxdeta112_accp=-1;
  m_maxdphi112_accp=-1;

  m_mindeta113_accp=-1;
  m_maxdeta113_accp=-1;
  m_maxdphi113_accp=-1;
  m_mindphip_halo=-1;
  m_mindetap_halo=-1;

  m_widePhi=-1;

  m_straightp=-1;
  m_curvedp=-1;

  m_mbaPhiOff=-1;
  m_mbbPhiOff=-1;

  kill_fiber = -1;
  QualityEnableME1a = -1;
  QualityEnableME1b = -1;
  QualityEnableME1c = -1;
  QualityEnableME1d = -1;
  QualityEnableME1e = -1;
  QualityEnableME1f = -1;
  QualityEnableME2a = -1;
  QualityEnableME2b = -1;
  QualityEnableME2c = -1;
  QualityEnableME3a = -1;
  QualityEnableME3b = -1;
  QualityEnableME3c = -1;
  QualityEnableME4a = -1;
  QualityEnableME4b = -1;
  QualityEnableME4c = -1;

  run_core = -1;
  trigger_on_ME1a = -1;
  trigger_on_ME1b = -1;
  trigger_on_ME2  = -1;
  trigger_on_ME3  = -1;
  trigger_on_ME4  = -1;
  trigger_on_MB1a = -1;
  trigger_on_MB1d = -1;

  singlesTrackOutput = 999;
  rescaleSinglesPhi  = -1;

  m_firmSP = -1;
  m_firmFA = -1;
  m_firmDD = -1;
  m_firmVM = -1;

  initFail_ = false;

  isCoreVerbose = pset.getParameter<bool>("isCoreVerbose");

  if(initializeFromPSet) readParameters(pset);


  // Sector Receiver LUTs initialization
  edm::ParameterSet srLUTset = pset.getParameter<edm::ParameterSet>("SRLUT");
  for(int i = 1; i <= 4; ++i)
    {
      if(i == 1)
	for(int j = 0; j < 2; j++)
          {
            srLUTs_[FPGAs[j]] = new CSCSectorReceiverLUT(endcap, sector, j+1, i, srLUTset, TMB07);
	  }
      else
	srLUTs_[FPGAs[i]] = new CSCSectorReceiverLUT(endcap, sector, 0, i, srLUTset, TMB07);
    }

  core_ = new CSCTFSPCoreLogic();

  // Pt LUTs initialization
  if(initializeFromPSet){
    edm::ParameterSet ptLUTset = pset.getParameter<edm::ParameterSet>("PTLUT");
    ptLUT_ = new CSCTFPtLUT(ptLUTset, scales, ptScale);
    LogDebug("CSCTFSectorProcessor") << "Using stand-alone PT LUT for endcap="<<m_endcap<<", sector="<<m_sector;
  } else {
    ptLUT_=0;
    LogDebug("CSCTFSectorProcessor") << "Looking for PT LUT in EventSetup for endcap="<<m_endcap<<", sector="<<m_sector;
  }

  // firmware map initialization
  // all the information are based on the firmware releases 
  // documented at http://www.phys.ufl.edu/~uvarov/SP05/SP05.htm

  // map is <m_firmSP, core_version>
  // it may happen that the same core is used for different firmware
  // versions, e.g. change in the wrapper only

  // this mapping accounts for runs starting from 132440
  // schema is year+month+day
  firmSP_Map.insert(std::pair<int,int>(20100210,20100122));
  firmSP_Map.insert(std::pair<int,int>(20100617,20100122));
  firmSP_Map.insert(std::pair<int,int>(20100629,20100122));

  firmSP_Map.insert(std::pair<int,int>(20100728,20100728));

  firmSP_Map.insert(std::pair<int,int>(20100901,20100901));

  //testing firmwares
  firmSP_Map.insert(std::pair<int,int>(20101011,20101011));
  firmSP_Map.insert(std::pair<int,int>(20101210,20101210));
  firmSP_Map.insert(std::pair<int,int>(20110204,20110118));
  firmSP_Map.insert(std::pair<int,int>(20110322,20110118));
  // 2012 core with non linear dphi
  firmSP_Map.insert(std::pair<int,int>(20120131,20120131));
  firmSP_Map.insert(std::pair<int,int>(20120227,20120131));
  //2012 core: 4 station track at |eta|>2.1 -> ME2-ME3-ME4
  firmSP_Map.insert(std::pair<int,int>(20120313,20120313));
  firmSP_Map.insert(std::pair<int,int>(20120319,20120313));
  //2012 core: 4 station track at |eta|>2.1 -> ME1-ME3-ME4 test
  firmSP_Map.insert(std::pair<int,int>(20120730,20120730));
}


void CSCTFSectorProcessor::initialize(const edm::EventSetup& c){
  initFail_ = false;
  if(!initializeFromPSet){
    // Only pT lut can be initialized from EventSetup, all front LUTs are initialized locally from their parametrizations
    LogDebug("CSCTFSectorProcessor") <<"Initializing endcap: "<<m_endcap<<" sector:"<<m_sector << "SP:" << (m_endcap-1)*6+(m_sector-1);
    LogDebug("CSCTFSectorProcessor") << "Initializing pT LUT from EventSetup";

    ptLUT_ = new CSCTFPtLUT(c);

    // Extract from EventSetup alternative (to the one, used in constructor) ParameterSet
    edm::ESHandle<L1MuCSCTFConfiguration> config;
    c.get<L1MuCSCTFConfigurationRcd>().get(config);
    // And initialize only those parameters, which left uninitialized during construction
    readParameters(config.product()->parameters((m_endcap-1)*6+(m_sector-1)));
  }

  // ---------------------------------------------------------------------------
  // This part is added per Vasile's request.
  // It will help people understanding the emulator configuration
  LogDebug("CSCTFSectorProcessor") << "\n !!! CSCTF EMULATOR CONFIGURATION !!!"
				   << "\n\nCORE CONFIGURATION"
                                   << "\n Coincidence Trigger? " << run_core
                                   << "\n Singles in ME1a? "     << trigger_on_ME1a
                                   << "\n Singles in ME1b? "     << trigger_on_ME1b
                                   << "\n Singles in ME2? "      << trigger_on_ME2
                                   << "\n Singles in ME3? "      << trigger_on_ME3
                                   << "\n Singles in ME4? "      << trigger_on_ME4
                                   << "\n Singles in MB1a? "     << trigger_on_MB1a
                                   << "\n Singles in MB1d? "     << trigger_on_MB1d

                                   << "\n BX Analyzer depth: assemble coinc. track with stubs in +/-" << m_bxa_depth << " Bxs"
                                   << "\n Is Wide Phi Extrapolation (DeltaPhi valid up to ~15 degrees, otherwise ~7.67 degrees)? " << m_widePhi
                                   << "\n PreTrigger=" << m_preTrigger

                                   << "\n CoreLatency=" << m_latency
                                   << "\n Is Phi for singles rescaled? " << rescaleSinglesPhi

                                   << "\n\nVARIOUS CONFIGURATION PARAMETERS"
                                   << "\n Allow ALCT only? " <<  m_allowALCTonly
                                   << "\n Allow CLCT only? " <<  m_allowCLCTonly

                                   << "\nQualityEnableME1a (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME1a
                                   << "\nQualityEnableME1b (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME1b
                                   << "\nQualityEnableME1c (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME1c
                                   << "\nQualityEnableME1d (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME1d
                                   << "\nQualityEnableME1e (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME1e
                                   << "\nQualityEnableME1f (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME1f
                                   << "\nQualityEnableME2a (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME2a
                                   << "\nQualityEnableME2b (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME2b
                                   << "\nQualityEnableME2c (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME2c
                                   << "\nQualityEnableME3a (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME3a
                                   << "\nQualityEnableME3b (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME3b
                                   << "\nQualityEnableME3c (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME3c
                                   << "\nQualityEnableME4a (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME4a
                                   << "\nQualityEnableME4b (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME4b
                                   << "\nQualityEnableME4c (in general accept all LCT qualities, i.e. 0xFFFF is expected)=" << QualityEnableME4c

                                   << "\nkill_fiber="          << kill_fiber
                                   << "\nSingles Output Link=" << singlesTrackOutput

    //the DAT_ETA registers meaning are explained at Table 2 of
    //http://www.phys.ufl.edu/~uvarov/SP05/LU-SP_ReferenceGuide_090915_Update.pdf

                                   << "\n\nDAT_ETA REGISTERS"
                                   << "\nMinimum eta difference for track cancellation logic=" << m_mindetap
                                   << "\nMinimum eta difference for halo track cancellation logic=" << m_mindetap_halo

                                   << "\nMinimum eta for ME1-ME2 collision tracks=" << m_etamin[0]
                                   << "\nMinimum eta for ME1-ME3 collision tracks=" << m_etamin[1]
                                   << "\nMinimum eta for ME2-ME3 collision tracks=" << m_etamin[2]
                                   << "\nMinimum eta for ME2-ME4 collision tracks=" << m_etamin[3]
                                   << "\nMinimum eta for ME3-ME4 collision tracks=" << m_etamin[4]
                                   << "\nMinimum eta for ME1-ME2 collision tracks in overlap region=" << m_etamin[5]
                                   << "\nMinimum eta for ME2-MB1 collision tracks=" << m_etamin[6]
                                   << "\nMinimum eta for ME1-ME4 collision tracks=" << m_etamin[7]

                                   << "\nMinimum eta difference for ME1-ME2 (except ME1/1) halo tracks=" << m_mindeta12_accp
                                   << "\nMinimum eta difference for ME1-ME3 (except ME1/1) halo tracks=" << m_mindeta13_accp
                                   << "\nMinimum eta difference for ME1/1-ME2 halo tracks=" << m_mindeta112_accp
                                   << "\nMinimum eta difference for ME1/1-ME3 halo tracks=" << m_mindeta113_accp

                                   << "\nMaximum eta for ME1-ME2 collision tracks=" << m_etamax[0]
                                   << "\nMaximum eta for ME1-ME3 collision tracks=" << m_etamax[1]
                                   << "\nMaximum eta for ME2-ME3 collision tracks=" << m_etamax[2]
                                   << "\nMaximum eta for ME2-ME4 collision tracks=" << m_etamax[3]
                                   << "\nMaximum eta for ME3-ME4 collision tracks=" << m_etamax[4]
                                   << "\nMaximum eta for ME1-ME2 collision tracks in overlap region=" << m_etamax[5]
                                   << "\nMaximum eta for ME2-MB1 collision tracks=" << m_etamax[6]
                                   << "\nMaximum eta for ME1-ME4 collision tracks=" << m_etamax[7]

                                   << "\nMaximum eta difference for ME1-ME2 (except ME1/1) halo tracks=" << m_maxdeta12_accp
                                   << "\nMaximum eta difference for ME1-ME3 (except ME1/1) halo tracks=" << m_maxdeta13_accp
                                   << "\nMaximum eta difference for ME1/1-ME2 halo tracks=" << m_maxdeta112_accp
                                   << "\nMaximum eta difference for ME1/1-ME3 halo tracks=" << m_maxdeta113_accp

                                   << "\nEta window for ME1-ME2 collision tracks=" << m_etawin[0]
                                   << "\nEta window for ME1-ME3 collision tracks=" << m_etawin[1]
                                   << "\nEta window for ME2-ME3 collision tracks=" << m_etawin[2]
                                   << "\nEta window for ME2-ME4 collision tracks=" << m_etawin[3]
                                   << "\nEta window for ME3-ME4 collision tracks=" << m_etawin[4]
                                   << "\nEta window for ME1-ME2 collision tracks in overlap region=" << m_etawin[5]
                                   << "\nEta window for ME1-ME4 collision tracks=" << m_etawin[6]

                                   << "\nMaximum phi difference for ME1-ME2 (except ME1/1) halo tracks=" << m_maxdphi12_accp
                                   << "\nMaximum phi difference for ME1-ME3 (except ME1/1) halo tracks=" << m_maxdphi13_accp
                                   << "\nMaximum phi difference for ME1/1-ME2 halo tracks=" << m_maxdphi112_accp
                                   << "\nMaximum phi difference for ME1/1-ME3 halo tracks=" << m_maxdphi113_accp

                                   << "\nMinimum phi difference for track cancellation logic=" << m_mindphip
                                   << "\nMinimum phi difference for halo track cancellation logic=" << m_mindphip_halo

                                   << "\nParameter for the correction of misaligned 1-2-3-4 straight tracks =" << m_straightp
                                   << "\nParameter for the correction of misaligned 1-2-3-4 curved tracks=" << m_curvedp
                                   << "\nPhi Offset for MB1A=" << m_mbaPhiOff
                                   << "\nPhi Offset for MB1D=" << m_mbbPhiOff 

				   << "\nFirmware SP year+month+day:" << m_firmSP
				   << "\nFirmware FA year+month+day:" << m_firmFA
				   << "\nFirmware DD year+month+day:" << m_firmDD
				   << "\nFirmware VM year+month+day:" << m_firmVM;


  printDisclaimer(m_firmSP,m_firmFA);

  // set core verbosity: for debugging only purpouses
  // in general the output is handled to Alex Madorsky
  core_ -> SetVerbose(isCoreVerbose);

  // Set the SP firmware 
  core_ -> SetSPFirmwareVersion (m_firmSP);
 
  // Set the firmware for the CORE
  int firmVersCore = firmSP_Map.find(m_firmSP)->second;
  core_ -> SetCoreFirmwareVersion (firmVersCore);
  edm::LogInfo( "CSCTFSectorProcessor" ) << "\nCore Firmware is set to " << core_ -> GetCoreFirmwareVersion();
  // ---------------------------------------------------------------------------

  // Check if parameters were not initialized in both: constuctor (from .cf? file) and initialize method (from EventSetup)
  if(m_bxa_depth<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"BXAdepth parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_allowALCTonly<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"AllowALCTonly parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_allowCLCTonly<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"AllowCLCTonly parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_preTrigger<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"PreTrigger parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindphip<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindphip parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindetap<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindeta parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_straightp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"straightp parameter left unitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_curvedp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"curvedp parameter left unitialized for endcap="<<m_endcap<<",sector="<<m_sector;
  }
  if(m_mbaPhiOff<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mbaPhiOff parameter left unitialized for endcap="<<m_endcap<<",sector="<<m_sector;
  }
  if(m_mbbPhiOff<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mbbPhiOff parameter left unitialized for endcap="<<m_endcap<<",sector="<<m_sector;
  }
  if(m_mindeta12_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindeta_accp12 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdeta12_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdeta_accp12 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdphi12_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdphi_accp12 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindeta13_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindeta_accp13 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdeta13_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdeta_accp13 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdphi13_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdphi_accp13 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindeta112_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindeta_accp112 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdeta112_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdeta_accp112 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdphi112_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdphi_accp112 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindeta113_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindeta_accp113 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdeta113_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdeta_accp113 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_maxdphi113_accp<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"maxdphi_accp113 parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindphip_halo<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindphip_halo parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }
  if(m_mindetap_halo<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"mindetep_halo parameter left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }

  if(m_widePhi<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<<"widePhi parameter left unitialized for endcap="<<m_endcap<<", sector="<<m_sector;
  }

  for(int index=0; index<8; index++)
    if(m_etamax[index]<0) 
    {
      initFail_ = true;
      edm::LogError("CSCTFSectorProcessor")<<"Some ("<<(8-index)<<") of EtaMax parameters left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
    }
  for(int index=0; index<8; index++)
    if(m_etamin[index]<0) 
    {
      initFail_ = true;
      edm::LogError("CSCTFSectorProcessor")<<"Some ("<<(8-index)<<") of EtaMin parameters left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
    }
  for(int index=0; index<7; index++)
    if(m_etawin[index]<0) 
    {
      initFail_ = true;
      edm::LogError("CSCTFSectorProcessor")<<"Some ("<<(6-index)<<") of EtaWindows parameters left uninitialized for endcap="<<m_endcap<<", sector="<<m_sector;
    }
  if(kill_fiber<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"kill_fiber parameter left uninitialized";
  }
  if(run_core<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"run_core parameter left uninitialized";
  }
  if(trigger_on_ME1a<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_ME1a parameter left uninitialized";
  }
  if(trigger_on_ME1b<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_ME1b parameter left uninitialized";
  }
  if(trigger_on_ME2 <0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_ME2 parameter left uninitialized";
  }
  if(trigger_on_ME3 <0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_ME3 parameter left uninitialized";
  }
  if(trigger_on_ME4 <0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_ME4 parameter left uninitialized";
  }
  if(trigger_on_MB1a<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_MB1a parameter left uninitialized";
  }
  if(trigger_on_MB1d<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"trigger_on_MB1d parameter left uninitialized";
  }
  if( trigger_on_ME1a>0 || trigger_on_ME1b>0 ||trigger_on_ME2>0  ||
      trigger_on_ME3>0  || trigger_on_ME4>0  ||trigger_on_MB1a>0 ||trigger_on_MB1d>0 )
  {
    if(singlesTrackOutput==999)
    {
       initFail_ = true;
       edm::LogError("CSCTFTrackBuilder")<<"singlesTrackOutput parameter left uninitialized";
    }
    if(rescaleSinglesPhi<0)
    {
       initFail_ = true;
       edm::LogError("CSCTFTrackBuilder")<<"rescaleSinglesPhi parameter left uninitialized";
    }
  }
  if(QualityEnableME1a<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME1a parameter left uninitialized";
  }
  if(QualityEnableME1b<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME1b parameter left uninitialized";
  }
  if(QualityEnableME1c<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME1c parameter left uninitialized";
  }
  if(QualityEnableME1d<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME1d parameter left uninitialized";
  }
  if(QualityEnableME1e<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME1e parameter left uninitialized";
  }
  if(QualityEnableME1f<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME1f parameter left uninitialized";
  }
  if(QualityEnableME2a<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME2a parameter left uninitialized";
  }
  if(QualityEnableME2b<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME2b parameter left uninitialized";
  }
  if(QualityEnableME2c<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME2c parameter left uninitialized";
  }
  if(QualityEnableME3a<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME3a parameter left uninitialized";
  }
  if(QualityEnableME3b<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME3b parameter left uninitialized";
  }
  if(QualityEnableME3c<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME3c parameter left uninitialized";
  }
  if(QualityEnableME4a<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME4a parameter left uninitialized";
  }
  if(QualityEnableME4b<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME4b parameter left uninitialized";
  }
  if(QualityEnableME4c<0)
  {
    initFail_ = true;
    edm::LogError("CSCTFTrackBuilder")<<"QualityEnableME4c parameter left uninitialized";
  }

  if (m_firmSP<1)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<< " firmwareSP parameter left uninitialized!!!\n";
  }
  if (m_firmFA<1)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<< " firmwareFA parameter left uninitialized!!!\n";
  }
  if (m_firmDD<1)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<< " firmwareDD parameter left uninitialized!!!\n";
  }
  if (m_firmVM<1)
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor")<< " firmwareVM parameter left uninitialized!!!\n";
  }

  if ( (m_firmFA != m_firmDD) ||
       (m_firmFA != m_firmVM) ||
       (m_firmDD != m_firmVM)  )
  {
    initFail_ = true;
    edm::LogError("CSCTFSectorProcessor::initialize")<< " firmwareFA (=" << m_firmFA << "), " 
   						<< " firmwareDD (=" << m_firmDD << "), " 
						<< " firmwareVM (=" << m_firmVM << ") are NOT identical: it shoultd NOT happen!\n";
  }

}

void CSCTFSectorProcessor::readParameters(const edm::ParameterSet& pset){
  m_bxa_depth = pset.getParameter<unsigned>("BXAdepth");
  m_allowALCTonly = ( pset.getParameter<bool>("AllowALCTonly") ? 1 : 0 );
  m_allowCLCTonly = ( pset.getParameter<bool>("AllowCLCTonly") ? 1 : 0 );
  m_preTrigger = pset.getParameter<unsigned>("PreTrigger");

  std::vector<unsigned>::const_iterator iter;
  int index=0;
  std::vector<unsigned> etawins = pset.getParameter<std::vector<unsigned> >("EtaWindows");
  for(iter=etawins.begin(),index=0; iter!=etawins.end()&&index<7; iter++,index++) m_etawin[index] = *iter;
  std::vector<unsigned> etamins = pset.getParameter<std::vector<unsigned> >("EtaMin");
  for(iter=etamins.begin(),index=0; iter!=etamins.end()&&index<8; iter++,index++) m_etamin[index] = *iter;
  std::vector<unsigned> etamaxs = pset.getParameter<std::vector<unsigned> >("EtaMax");
  for(iter=etamaxs.begin(),index=0; iter!=etamaxs.end()&&index<8; iter++,index++) m_etamax[index] = *iter;

  m_mindphip = pset.getParameter<unsigned>("mindphip");
  m_mindetap = pset.getParameter<unsigned>("mindetap");
  m_straightp = pset.getParameter<unsigned>("straightp");
  m_curvedp = pset.getParameter<unsigned>("curvedp");
  m_mbaPhiOff = pset.getParameter<unsigned>("mbaPhiOff");
  m_mbbPhiOff = pset.getParameter<unsigned>("mbbPhiOff");
  m_widePhi = pset.getParameter<unsigned>("widePhi");
  m_mindeta12_accp = pset.getParameter<unsigned>("mindeta12_accp");
  m_maxdeta12_accp = pset.getParameter<unsigned>("maxdeta12_accp");
  m_maxdphi12_accp = pset.getParameter<unsigned>("maxdphi12_accp");
  m_mindeta13_accp = pset.getParameter<unsigned>("mindeta13_accp");
  m_maxdeta13_accp = pset.getParameter<unsigned>("maxdeta13_accp");
  m_maxdphi13_accp = pset.getParameter<unsigned>("maxdphi13_accp");
  m_mindeta112_accp = pset.getParameter<unsigned>("mindeta112_accp");
  m_maxdeta112_accp = pset.getParameter<unsigned>("maxdeta112_accp");
  m_maxdphi112_accp = pset.getParameter<unsigned>("maxdphi112_accp");
  m_mindeta113_accp = pset.getParameter<unsigned>("mindeta113_accp");
  m_maxdeta113_accp = pset.getParameter<unsigned>("maxdeta113_accp");
  m_maxdphi113_accp = pset.getParameter<unsigned>("maxdphi113_accp");
  m_mindphip_halo = pset.getParameter<unsigned>("mindphip_halo");
  m_mindetap_halo = pset.getParameter<unsigned>("mindetap_halo");
  kill_fiber = pset.getParameter<unsigned>("kill_fiber");
  run_core = pset.getParameter<bool>("run_core");
  trigger_on_ME1a = pset.getParameter<bool>("trigger_on_ME1a");
  trigger_on_ME1b = pset.getParameter<bool>("trigger_on_ME1b");
  trigger_on_ME2 = pset.getParameter<bool>("trigger_on_ME2");
  trigger_on_ME3 = pset.getParameter<bool>("trigger_on_ME3");
  trigger_on_ME4 = pset.getParameter<bool>("trigger_on_ME4");
  trigger_on_MB1a = pset.getParameter<bool>("trigger_on_MB1a");
  trigger_on_MB1d = pset.getParameter<bool>("trigger_on_MB1d");

  singlesTrackOutput = pset.getParameter<unsigned int>("singlesTrackOutput");
  rescaleSinglesPhi  = pset.getParameter<bool>("rescaleSinglesPhi");
  QualityEnableME1a = pset.getParameter<unsigned int>("QualityEnableME1a");
  QualityEnableME1b = pset.getParameter<unsigned int>("QualityEnableME1b");
  QualityEnableME1c = pset.getParameter<unsigned int>("QualityEnableME1c");
  QualityEnableME1d = pset.getParameter<unsigned int>("QualityEnableME1d");
  QualityEnableME1e = pset.getParameter<unsigned int>("QualityEnableME1e");
  QualityEnableME1f = pset.getParameter<unsigned int>("QualityEnableME1f");
  QualityEnableME2a = pset.getParameter<unsigned int>("QualityEnableME2a");
  QualityEnableME2b = pset.getParameter<unsigned int>("QualityEnableME2b");
  QualityEnableME2c = pset.getParameter<unsigned int>("QualityEnableME2c");
  QualityEnableME3a = pset.getParameter<unsigned int>("QualityEnableME3a");
  QualityEnableME3b = pset.getParameter<unsigned int>("QualityEnableME3b");
  QualityEnableME3c = pset.getParameter<unsigned int>("QualityEnableME3c");
  QualityEnableME4a = pset.getParameter<unsigned int>("QualityEnableME4a");
  QualityEnableME4b = pset.getParameter<unsigned int>("QualityEnableME4b");
  QualityEnableME4c = pset.getParameter<unsigned int>("QualityEnableME4c");

  m_firmSP = pset.getParameter<unsigned int>("firmwareSP");
  m_firmFA = pset.getParameter<unsigned int>("firmwareFA");
  m_firmDD = pset.getParameter<unsigned int>("firmwareDD");
  m_firmVM = pset.getParameter<unsigned int>("firmwareVM");

}

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

//returns 0 for no tracks, 1 tracks found, and -1 for "exception" (what used to throw an exception)
// on -1, Producer should produce empty collections for event
int CSCTFSectorProcessor::run(const CSCTriggerContainer<csctf::TrackStub>& stubs)
{
  if(initFail_)
    return -1;

  if( !ptLUT_ )
  {
    edm::LogError("CSCTFSectorProcessor::run()") << "No CSCTF PTLUTs: Initialize CSC TF LUTs first (missed call to CSCTFTrackProducer::beginJob?\n";
    return -1;
  }


  l1_tracks.clear();
  dt_stubs.clear();
  stub_vec_filtered.clear();

  std::vector<csctf::TrackStub> stub_vec = stubs.get();

  /** STEP ZERO
   *  Remove stubs, which were masked out by kill_fiber or QualityEnable parameters
   */
  for(std::vector<csctf::TrackStub>::const_iterator itr=stub_vec.begin(); itr!=stub_vec.end(); itr++)
    switch( itr->station() ){
    case 5: stub_vec_filtered.push_back(*itr); break; // DT stubs get filtered by the core controll register
    case 4:
      switch( itr->getMPCLink() ){
      case 3: if( (kill_fiber&0x4000)==0 && QualityEnableME4c&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 2: if( (kill_fiber&0x2000)==0 && QualityEnableME4b&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 1: if( (kill_fiber&0x1000)==0 && QualityEnableME4a&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      default: edm::LogWarning("CSCTFSectorProcessor::run()") << "No MPC sorting for LCT: link="<<itr->getMPCLink()<<"\n";
      }
      break;
    case 3:
      switch( itr->getMPCLink() ){
      case 3: if( (kill_fiber&0x0800)==0 && QualityEnableME3c&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 2: if( (kill_fiber&0x0400)==0 && QualityEnableME3b&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 1: if( (kill_fiber&0x0200)==0 && QualityEnableME3a&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      default: edm::LogWarning("CSCTFSectorProcessor::run()") << "No MPC sorting for LCT: link="<<itr->getMPCLink()<<"\n";
      }
      break;
    case 2:
      switch( itr->getMPCLink() ){
      case 3: if( (kill_fiber&0x0100)==0 && QualityEnableME2c&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 2: if( (kill_fiber&0x0080)==0 && QualityEnableME2b&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 1: if( (kill_fiber&0x0040)==0 && QualityEnableME2a&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      default: edm::LogWarning("CSCTFSectorProcessor::run()") << "No MPC sorting for LCT: link="<<itr->getMPCLink()<<"\n";
      }
      break;
    case 1:
      switch( itr->getMPCLink() + (3*(CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(itr->getDetId().rawId())) - 1)) ){
      case 6: if( (kill_fiber&0x0020)==0 && QualityEnableME1f&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 5: if( (kill_fiber&0x0010)==0 && QualityEnableME1e&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 4: if( (kill_fiber&0x0008)==0 && QualityEnableME1d&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 3: if( (kill_fiber&0x0004)==0 && QualityEnableME1c&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 2: if( (kill_fiber&0x0002)==0 && QualityEnableME1b&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      case 1: if( (kill_fiber&0x0001)==0 && QualityEnableME1a&(1<<itr->getQuality()) ) stub_vec_filtered.push_back(*itr); break;
      default: edm::LogWarning("CSCTFSectorProcessor::run()") << "No MPC sorting for LCT: link="<<itr->getMPCLink()<<"\n";
      }
      break;
    default: edm::LogWarning("CSCTFSectorProcessor::run()") << "Invalid station # encountered: "<<itr->station()<<"\n";
    }

  /** STEP ONE
   *  We take stubs from the MPC and assign their eta and phi
   *  coordinates using the SR Lookup tables.
   *  This is independent of what BX we are on so we can
   *  process one large vector of stubs.
   *  After this we append the stubs gained from the DT system.
   */

  for(std::vector<csctf::TrackStub>::iterator itr=stub_vec_filtered.begin(); itr!=stub_vec_filtered.end(); itr++)
    {
      if(itr->station() != 5)
        {
          CSCDetId id(itr->getDetId().rawId());
          unsigned fpga = (id.station() == 1) ? CSCTriggerNumbering::triggerSubSectorFromLabels(id) - 1 : id.station();

          lclphidat lclPhi;
          try {
            lclPhi = srLUTs_[FPGAs[fpga]]->localPhi(itr->getStrip(), itr->getPattern(), itr->getQuality(), itr->getBend());
          } catch( cms::Exception &e ) {
            bzero(&lclPhi,sizeof(lclPhi));
            edm::LogWarning("CSCTFSectorProcessor:run()") << "Exception from LocalPhi LUT in " << FPGAs[fpga]
                                                          << "(strip="<<itr->getStrip()<<",pattern="<<itr->getPattern()<<",quality="<<itr->getQuality()<<",bend="<<itr->getBend()<<")" <<std::endl;
          }

          gblphidat gblPhi;
          try {
            unsigned csc_id = itr->cscid();
            if (!m_gangedME1a) csc_id = itr->cscidSeparateME1a();
            gblPhi = srLUTs_[FPGAs[fpga]]->globalPhiME(lclPhi.phi_local, itr->getKeyWG(), csc_id);
        
          } catch( cms::Exception &e ) {
            bzero(&gblPhi,sizeof(gblPhi));
            edm::LogWarning("CSCTFSectorProcessor:run()") << "Exception from GlobalPhi LUT in " << FPGAs[fpga]
                                                          << "(phi_local="<<lclPhi.phi_local<<",KeyWG="<<itr->getKeyWG()<<",csc="<<itr->cscid()<<")"<<std::endl;
          }

          gbletadat gblEta;
          try {
            gblEta = srLUTs_[FPGAs[fpga]]->globalEtaME(lclPhi.phi_bend_local, lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
          } catch( cms::Exception &e ) {
            bzero(&gblEta,sizeof(gblEta));
            edm::LogWarning("CSCTFSectorProcessor:run()") << "Exception from GlobalEta LUT in " << FPGAs[fpga]
                                                          << "(phi_bend_local="<<lclPhi.phi_bend_local<<",phi_local="<<lclPhi.phi_local<<",KeyWG="<<itr->getKeyWG()<<",csc="<<itr->cscid()<<")"<<std::endl;
          }

          gblphidat gblPhiDT;
          try {
            gblPhiDT = srLUTs_[FPGAs[fpga]]->globalPhiMB(lclPhi.phi_local, itr->getKeyWG(), itr->cscid());
          } catch( cms::Exception &e ) {
            bzero(&gblPhiDT,sizeof(gblPhiDT));
            edm::LogWarning("CSCTFSectorProcessor:run()") << "Exception from GlobalPhi DT LUT in " << FPGAs[fpga]
                                                          << "(phi_local="<<lclPhi.phi_local<<",KeyWG="<<itr->getKeyWG()<<",csc="<<itr->cscid()<<")"<<std::endl;
          }

          itr->setEtaPacked(gblEta.global_eta);

          if(itr->station() == 1 ) {
            //&& itr->cscId() > 6) { //only ring 3 
            itr->setPhiPacked(gblPhiDT.global_phi);// convert the DT to convert
            dt_stubs.push_back(*itr); // send stubs to DT
          }

          //reconvert the ME1 LCT to the CSCTF units.
          //the same iterator is used to fill two containers, 
          //the CSCTF one (stub_vec_filtered) and LCTs sent to DTTF (dt_stubs)
          itr->setPhiPacked(gblPhi.global_phi);

          LogDebug("CSCTFSectorProcessor:run()") << "LCT found, processed by FPGA: " << FPGAs[fpga] << std::endl
                                                 << " LCT now has (eta, phi) of: (" << itr->etaValue() << "," << itr->phiValue() <<")\n";
        }
    }
  
  CSCTriggerContainer<csctf::TrackStub> processedStubs(stub_vec_filtered);

  /** STEP TWO
   *  We take the stubs filled by the SR LUTs and load them
   *  for processing into the SP core logic.
   *  After loading we run and then retrieve any tracks generated.
   */

  std::vector<csc::L1Track> tftks;

  if(run_core){
    core_->loadData(processedStubs, m_endcap, m_sector, m_minBX, m_maxBX);
    if( core_->run(m_endcap, m_sector, m_latency,
		   m_etamin[0], m_etamin[1], m_etamin[2], m_etamin[3],
		   m_etamin[4], m_etamin[5], m_etamin[6], m_etamin[7],
		   m_etamax[0], m_etamax[1], m_etamax[2], m_etamax[3],
		   m_etamax[4], m_etamax[5], m_etamax[6], m_etamax[7],
		   m_etawin[0], m_etawin[1], m_etawin[2],
		   m_etawin[3], m_etawin[4], m_etawin[5], m_etawin[6],
		   m_mindphip, m_mindetap,
		   m_mindeta12_accp,  m_maxdeta12_accp, m_maxdphi12_accp,
		   m_mindeta13_accp,  m_maxdeta13_accp, m_maxdphi13_accp,
		   m_mindeta112_accp,  m_maxdeta112_accp, m_maxdphi112_accp,
		   m_mindeta113_accp,  m_maxdeta113_accp, m_maxdphi113_accp,
		   m_mindphip_halo, m_mindetap_halo,
		   m_straightp, m_curvedp,
		   m_mbaPhiOff, m_mbbPhiOff,
		   m_bxa_depth, m_allowALCTonly, m_allowCLCTonly, m_preTrigger, m_widePhi,
		   m_minBX, m_maxBX) )
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

	if( ((titr->ptLUTAddress()>>16)&0xf)==15 )
	  {
	    int unmodBx = titr->bx();
	    titr->setBx(unmodBx+2);
	  }
      }
  } //end of if(run_core)

  l1_tracks = tftks;


  // Add-on for singles:
  CSCTriggerContainer<csctf::TrackStub> myStubContainer[7]; //[BX]
  // Loop over CSC LCTs if triggering on them:
  if( trigger_on_ME1a || trigger_on_ME1b || trigger_on_ME2 || trigger_on_ME3 || trigger_on_ME4 || trigger_on_MB1a || trigger_on_MB1d )
    for(std::vector<csctf::TrackStub>::iterator itr=stub_vec_filtered.begin(); itr!=stub_vec_filtered.end(); itr++){
      int station = itr->station()-1;
      if(station != 4){
	int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(itr->getDetId().rawId()));
	int mpc = ( subSector ? subSector-1 : station+1 );
	if( (mpc==0&&trigger_on_ME1a) || (mpc==1&&trigger_on_ME1b) ||
	    (mpc==2&&trigger_on_ME2)  || (mpc==3&&trigger_on_ME3)  ||
	    (mpc==4&&trigger_on_ME4)  ||
	    (mpc==5&& ( (trigger_on_MB1a&&subSector%2==1) || (trigger_on_MB1d&&subSector%2==0) ) ) ){
	  int bx = itr->getBX() - m_minBX;
	  if( bx<0 || bx>=7 ) edm::LogWarning("CSCTFTrackBuilder::buildTracks()") << " LCT BX is out of ["<<m_minBX<<","<<m_maxBX<<") range: "<<itr->getBX();
	  else
	    if( itr->isValid() ) myStubContainer[bx].push_back(*itr);
	}
      }
    }

  // Core's input was loaded in a relative time window BX=[0-7)
  // To relate it to time window of tracks (centred at BX=0) we introduce a shift:
  int shift = (m_maxBX + m_minBX)/2 - m_minBX;

  // Now we put tracks from singles in a certain bx
  //   if there were no tracks from the core in this endcap/sector/bx
  CSCTriggerContainer<csc::L1Track> tracksFromSingles;
  for(int bx=0; bx<7; bx++)
    if( myStubContainer[bx].get().size() ){ // VP in this bx
      bool coreTrackExists = false;
      // tracks are not ordered to be accessible by bx => loop them all
      std::vector<csc::L1Track> tracks = l1_tracks.get();
      for(std::vector<csc::L1Track>::iterator trk=tracks.begin(); trk<tracks.end(); trk++)
	if( (trk->BX() == bx-shift && trk->outputLink() == singlesTrackOutput)
	    || (((trk->ptLUTAddress()>>16)&0xf)==15 && trk->BX()-2 == bx-shift) ){
	  coreTrackExists = true;
	  break;
	}
      if( coreTrackExists == false ){
	csc::L1TrackId trackId(m_endcap,m_sector);
	csc::L1Track   track(trackId);
	track.setBx(bx-shift);
	track.setOutputLink(singlesTrackOutput);
	//CSCCorrelatedLCTDigiCollection singles;
	std::vector<csctf::TrackStub> stubs = myStubContainer[bx].get();
	// Select best quality stub, and assign its eta/phi coordinates to the track
	int qualityME=0, qualityMB=0, ME=100, MB=100, linkME=7;
	std::vector<csctf::TrackStub>::const_iterator bestStub=stubs.end();
	for(std::vector<csctf::TrackStub>::const_iterator st_iter=stubs.begin(); st_iter!=stubs.end(); st_iter++){
	  int station = st_iter->station()-1;
	  int subSector = CSCTriggerNumbering::triggerSubSectorFromLabels(CSCDetId(st_iter->getDetId().rawId()));
	  int mpc = ( subSector ? subSector-1 : station+1 );
	  // Sort MB stubs first (priority: quality OR MB1a > MB1b for the same quality)
	  if( mpc==5 &&  (st_iter->getQuality()>qualityMB || (st_iter->getQuality()==qualityMB&&subSector<MB)) ){
	    qualityMB = st_iter->getQuality();
	    MB        = subSector;
	    if(ME>4) bestStub = st_iter; // do not select this stub if ME already had any candidate
	  }
	  // Sort ME stubs (priority: quality OR ME1a > ME1b > ME2 > ME3 > ME4 for the same quality)
	  if( mpc<5  && (st_iter->getQuality()> qualityME
			 || (st_iter->getQuality()==qualityME && mpc< ME)
			 || (st_iter->getQuality()==qualityME && mpc==ME && st_iter->getMPCLink()<linkME))) {
	    qualityME = st_iter->getQuality();
	    ME        = mpc;
	    linkME    = st_iter->getMPCLink();
	    bestStub  = st_iter;
	  }
	}
	unsigned rescaled_phi = 999;
	if (m_firmSP <= 20100210) {
	  // buggy implementation of the phi for singles in the wrapper... 
	  // at the end data/emulator have to agree: e.g. wrong in the same way
	  // BUG: getting the lowest 7 bits instead the 7 most significant ones.
	  rescaled_phi = unsigned(24*(bestStub->phiPacked()&0x7f)/128.);
	}
	else {
	  // correct implementation :-)
	  rescaled_phi = unsigned(24*(bestStub->phiPacked()>>5)/128.);
	}

	unsigned unscaled_phi =              bestStub->phiPacked()>>7       ;
	track.setLocalPhi(rescaleSinglesPhi?rescaled_phi:unscaled_phi);
	track.setEtaPacked((bestStub->etaPacked()>>2)&0x1f);
	switch( bestStub->station() ){
	case 1: track.setStationIds(bestStub->getMPCLink(),0,0,0,0); break;
	case 2: track.setStationIds(0,bestStub->getMPCLink(),0,0,0); break;
	case 3: track.setStationIds(0,0,bestStub->getMPCLink(),0,0); break;
	case 4: track.setStationIds(0,0,0,bestStub->getMPCLink(),0); break;
	case 5: track.setStationIds(0,0,0,0,bestStub->getMPCLink()); break;
	default: edm::LogError("CSCTFSectorProcessor::run()") << "Illegal LCT link="<<bestStub->station()<<"\n"; break;
	}
	//   singles.insertDigi(CSCDetId(st_iter->getDetId().rawId()),*st_iter);
	//tracksFromSingles.push_back(L1CSCTrack(track,singles));
	track.setPtLUTAddress( (1<<16) | ((bestStub->etaPacked()<<9)&0xf000) );
	ptadd thePtAddress( track.ptLUTAddress() );
	ptdat thePtData = ptLUT_->Pt(thePtAddress);
	if( thePtAddress.track_fr ){
	  track.setRank(thePtData.front_rank);
	  track.setChargeValidPacked(thePtData.charge_valid_front);
	} else {
	  track.setRank(thePtData.rear_rank);
	  track.setChargeValidPacked(thePtData.charge_valid_rear);
	}
	tracksFromSingles.push_back(track);
      }
    }
  std::vector<csc::L1Track> single_tracks = tracksFromSingles.get();
  if( single_tracks.size() ) l1_tracks.push_many(single_tracks);
  // End of add-on for singles

  return (l1_tracks.get().size() > 0);
}

// according to the firmware versions print some more information
void CSCTFSectorProcessor::printDisclaimer(int firmSP, int firmFA){
  
  edm::LogInfo( "CSCTFSectorProcessor" ) << "\n\n"
					 << "******************************* \n"
					 << "***       DISCLAIMER        *** \n"
					 << "******************************* \n"
					 << "\n Firmware SP version (year+month+day)=" << firmSP
					 << "\n Firmware FA/VM/DD version (year+month+day)=" << firmFA;
  if (firmSP==20100210)
    edm::LogInfo( "CSCTFSectorProcessor" ) << " -> KNOWN BUGS IN THE FIRMWARE:\n"
					   << "\t * Wrong phi assignment for singles\n"
					   << "\t * Wrapper passes to the core only even quality DT stubs\n"
					   << "\n -> BUGS ARE GOING TO BE EMULATED BY THE SOFTWARE\n\n";

  else 
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t * Correct phi assignment for singles\n";
  
  if (firmSP==20100629){
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t * Correct MB quality masking in the wrapper\n"
                                           << "\t * Core is 20100122\n";
  }

  if (firmSP==20100728)
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t * Inverted MB clocks\n";

  if (firmSP==20100901)
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t * Inverted charge bit\n";

  if (firmSP==20101011)
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t **** WARNING THIS FIRMWARE IS UNDER TEST ****\n"
                                           << "\t * Added CSC-DT assembling tracks ME1-MB2/1   \n";
  if (firmSP==20101210)
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t **** WARNING THIS FIRMWARE IS UNDER TEST ****\n"
                                           << "\t * New Ghost Busting Algorithm Removing Tracks\n"
                                           << "\t   Sharing at Least One LCT\n";

  if (firmSP==20110118)
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t **** WARNING THIS FIRMWARE IS UNDER TEST ****\n"
                                           << "\t * New Ghost Busting Algorithm Removing Tracks\n"
                                           << "\t   Sharing at Least One LCT\n"
                                           << "\t * Passing CLCT and PhiBend for PT LUTs\n";
  if (firmSP==20120131)
    edm::LogInfo( "CSCTFSectorProcessor" ) << "\t **** WARNING THIS FIRMWARE IS UNDER TEST ****\n"
                                           << "\t * non-linear dphi12 dphi23, use deta for PTLUTs \n";
}
