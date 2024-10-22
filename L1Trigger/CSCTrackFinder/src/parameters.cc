// -*- C++ -*-
//
// Package:     L1Trigger/CSCTrackFinder
// Class  :     parameters
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 27 May 2021 20:02:26 GMT
//

// system include files

// user include files
#include "parameters.h"
#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

edm::ParameterSet parameters(L1MuCSCTFConfiguration const& iConfig, int sp) {
  LogDebug("L1MuCSCTFConfiguration") << "SP:" << int(sp) << std::endl;

  edm::ParameterSet pset;
  if (sp >= 12)
    return pset;

  // ------------------------------------------------------
  // core configuration
  // by default everything is disabled: we need to set them
  // coincidence and singles
  bool run_core = false;
  bool trigger_on_ME1a = false;
  bool trigger_on_ME1b = false;
  bool trigger_on_ME2 = false;
  bool trigger_on_ME3 = false;
  bool trigger_on_ME4 = false;
  bool trigger_on_MB1a = false;
  bool trigger_on_MB1d = false;

  unsigned int BXAdepth = 0;
  unsigned int useDT = 0;
  unsigned int widePhi = 0;
  unsigned int PreTrigger = 0;
  // ------------------------------------------------------

  // ------------------------------------------------------
  // these are very important parameters.
  // Double check with Alex
  unsigned int CoreLatency = 7;
  bool rescaleSinglesPhi = true;

  // ask Alex if use or remove them or what
  bool AllowALCTonly = false;
  bool AllowCLCTonly = false;

  // other useful parameters in general not set in the OMDS
  unsigned int QualityEnableME1a = 0xFFFF;
  unsigned int QualityEnableME1b = 0xFFFF;
  unsigned int QualityEnableME1c = 0xFFFF;
  unsigned int QualityEnableME1d = 0xFFFF;
  unsigned int QualityEnableME1e = 0xFFFF;
  unsigned int QualityEnableME1f = 0xFFFF;
  unsigned int QualityEnableME2a = 0xFFFF;
  unsigned int QualityEnableME2b = 0xFFFF;
  unsigned int QualityEnableME2c = 0xFFFF;
  unsigned int QualityEnableME3a = 0xFFFF;
  unsigned int QualityEnableME3b = 0xFFFF;
  unsigned int QualityEnableME3c = 0xFFFF;
  unsigned int QualityEnableME4a = 0xFFFF;
  unsigned int QualityEnableME4b = 0xFFFF;
  unsigned int QualityEnableME4c = 0xFFFF;

  unsigned int kill_fiber = 0;
  unsigned int singlesTrackOutput = 1;
  // ------------------------------------------------------

  //initialization of the DAT_ETA registers with default values
  //the DAT_ETA registers meaning are explained at Table 2 of
  //http://www.phys.ufl.edu/~uvarov/SP05/LU-SP_ReferenceGuide_090915_Update.pdf
  std::vector<unsigned int> etamin(8), etamax(8), etawin(7);

  unsigned int mindetap = 8;
  unsigned int mindetap_halo = 8;

  etamin[0] = 22;
  etamin[1] = 22;
  etamin[2] = 14;
  etamin[3] = 14;
  etamin[4] = 14;
  etamin[5] = 14;
  etamin[6] = 10;
  etamin[7] = 22;

  unsigned int mindeta12_accp = 8;
  unsigned int mindeta13_accp = 19;
  unsigned int mindeta112_accp = 19;
  unsigned int mindeta113_accp = 30;

  etamax[0] = 127;
  etamax[1] = 127;
  etamax[2] = 127;
  etamax[3] = 127;
  etamax[4] = 127;
  etamax[5] = 24;
  etamax[6] = 24;
  etamax[7] = 127;

  unsigned int maxdeta12_accp = 14;
  unsigned int maxdeta13_accp = 25;
  unsigned int maxdeta112_accp = 25;
  unsigned int maxdeta113_accp = 36;

  etawin[0] = 4;
  etawin[1] = 4;
  etawin[2] = 4;
  etawin[3] = 4;
  etawin[4] = 4;
  etawin[5] = 4;
  etawin[6] = 4;

  unsigned int maxdphi12_accp = 64;
  unsigned int maxdphi13_accp = 64;
  unsigned int maxdphi112_accp = 64;
  unsigned int maxdphi113_accp = 64;

  unsigned int mindphip = 128;
  unsigned int mindphip_halo = 128;

  unsigned int straightp = 60;
  unsigned int curvedp = 200;

  unsigned int mbaPhiOff = 0;
  // this differ from the default value in the documentation because during
  // craft 09 it mbbPhiOff, as well as mbaPhiOff were not existing, thus set to 0 (they are offsets)
  // and for backward compatibility it needs to be set to 0. Anyway mbbPhiOff since its introduction in the
  // core will have to be ALWAYS part of the configuration, so it won't be never initialized to the
  // default value 2048.
  unsigned int mbbPhiOff = 0;

  int eta_cnt = 0;

  // default firmware versions (the ones used from run 132440)
  unsigned int firmwareSP = 20100210;
  unsigned int firmwareFA = 20090521;
  unsigned int firmwareDD = 20090521;
  unsigned int firmwareVM = 20090521;

  // default printout
  LogDebug("L1MuCSCTFConfiguration")
      << "\nCORE CONFIGURATION  DEFAULT VALUES"
      << "\nrun_core=" << run_core << "\ntrigger_on_ME1a=" << trigger_on_ME1a << "\ntrigger_on_ME1b=" << trigger_on_ME1b
      << "\ntrigger_on_ME2=" << trigger_on_ME2 << "\ntrigger_on_ME3=" << trigger_on_ME3
      << "\ntrigger_on_ME4=" << trigger_on_ME4 << "\ntrigger_on_MB1a=" << trigger_on_MB1a
      << "\ntrigger_on_MB1d=" << trigger_on_MB1d

      << "\nBXAdepth=" << BXAdepth << "\nuseDT=" << useDT << "\nwidePhi=" << widePhi << "\nPreTrigger=" << PreTrigger

      << "\nCoreLatency=" << CoreLatency << "\nrescaleSinglesPhi=" << rescaleSinglesPhi

      << "\n\nVARIOUS CONFIGURATION PARAMETERS DEFAULT VALUES"
      << "\nAllowALCTonly=" << AllowALCTonly << "\nAllowCLCTonly=" << AllowCLCTonly

      << "\nQualityEnableME1a=" << QualityEnableME1a << "\nQualityEnableME1b=" << QualityEnableME1b
      << "\nQualityEnableME1c=" << QualityEnableME1c << "\nQualityEnableME1d=" << QualityEnableME1d
      << "\nQualityEnableME1e=" << QualityEnableME1e << "\nQualityEnableME1f=" << QualityEnableME1f
      << "\nQualityEnableME2a=" << QualityEnableME2a << "\nQualityEnableME2b=" << QualityEnableME2b
      << "\nQualityEnableME2c=" << QualityEnableME2c << "\nQualityEnableME3a=" << QualityEnableME3a
      << "\nQualityEnableME3b=" << QualityEnableME3b << "\nQualityEnableME3c=" << QualityEnableME3c
      << "\nQualityEnableME4a=" << QualityEnableME4a << "\nQualityEnableME4b=" << QualityEnableME4b
      << "\nQualityEnableME4c=" << QualityEnableME4c

      << "\nkill_fiber=" << kill_fiber << "\nsinglesTrackOutput=" << singlesTrackOutput

      << "\n\nDEFAULT VALUES FOR DAT_ETA"
      << "\nmindetap     =" << mindetap << "\nmindetap_halo=" << mindetap_halo

      << "\netamin[0]=" << etamin[0] << "\netamin[1]=" << etamin[1] << "\netamin[2]=" << etamin[2]
      << "\netamin[3]=" << etamin[3] << "\netamin[4]=" << etamin[4] << "\netamin[5]=" << etamin[5]
      << "\netamin[6]=" << etamin[6] << "\netamin[7]=" << etamin[7]

      << "\nmindeta12_accp =" << mindeta12_accp << "\nmindeta13_accp =" << mindeta13_accp
      << "\nmindeta112_accp=" << mindeta112_accp << "\nmindeta113_accp=" << mindeta113_accp

      << "\netamax[0]=" << etamax[0] << "\netamax[1]=" << etamax[1] << "\netamax[2]=" << etamax[2]
      << "\netamax[3]=" << etamax[3] << "\netamax[4]=" << etamax[4] << "\netamax[5]=" << etamax[5]
      << "\netamax[6]=" << etamax[6] << "\netamax[7]=" << etamax[7]

      << "\nmaxdeta12_accp =" << maxdeta12_accp << "\nmaxdeta13_accp =" << maxdeta13_accp
      << "\nmaxdeta112_accp=" << maxdeta112_accp << "\nmaxdeta113_accp=" << maxdeta113_accp

      << "\netawin[0]=" << etawin[0] << "\netawin[1]=" << etawin[1] << "\netawin[2]=" << etawin[2]
      << "\netawin[3]=" << etawin[3] << "\netawin[4]=" << etawin[4] << "\netawin[5]=" << etawin[5]
      << "\netawin[6]=" << etawin[6]

      << "\nmaxdphi12_accp =" << maxdphi12_accp << "\nmaxdphi13_accp =" << maxdphi13_accp
      << "\nmaxdphi112_accp=" << maxdphi112_accp << "\nmaxdphi113_accp=" << maxdphi113_accp

      << "\nmindphip     =" << mindphip << "\nmindphip_halo=" << mindphip_halo

      << "\nstraightp=" << straightp << "\ncurvedp  =" << curvedp << "\nmbaPhiOff=" << mbaPhiOff
      << "\nmbbPhiOff=" << mbbPhiOff

      << "\n\nFIRMWARE VERSIONS"
      << "\nSP: " << firmwareSP << "\nFA: " << firmwareFA << "\nDD: " << firmwareDD << "\nVM: " << firmwareVM;

  // start filling the registers with the values in the DBS
  std::stringstream conf(iConfig[sp]);
  while (!conf.eof()) {
    char buff[1024];
    conf.getline(buff, 1024);
    std::stringstream line(buff);
    //std::cout<<"buff:"<<buff<<std::endl;
    std::string register_;
    line >> register_;
    std::string chip_;
    line >> chip_;
    std::string muon_;
    line >> muon_;
    std::string writeValue_;
    line >> writeValue_;
    std::string comments_;
    std::getline(line, comments_);

    if (register_ == "CSR_REQ" && chip_ == "SP") {
      unsigned int value = ::strtol(writeValue_.c_str(), nullptr, 16);
      run_core = (value & 0x8000);
      trigger_on_ME1a = (value & 0x0001);
      trigger_on_ME1b = (value & 0x0002);
      trigger_on_ME2 = (value & 0x0004);
      trigger_on_ME3 = (value & 0x0008);
      trigger_on_ME4 = (value & 0x0010);
      trigger_on_MB1a = (value & 0x0100);
      trigger_on_MB1d = (value & 0x0200);
    }

    if (register_ == "CSR_SCC" && chip_ == "SP") {
      unsigned int value = ::strtol(writeValue_.c_str(), nullptr, 16);

      BXAdepth = (value & 0x3);
      useDT = ((value & 0x80) >> 7);
      widePhi = ((value & 0x40) >> 6);
      PreTrigger = ((value & 0x300) >> 8);
    }

    if (register_ == "CSR_LQE" && chip_ == "F1" && muon_ == "M1")
      QualityEnableME1a = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F1" && muon_ == "M2")
      QualityEnableME1b = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F1" && muon_ == "M3")
      QualityEnableME1c = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F2" && muon_ == "M1")
      QualityEnableME1d = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F2" && muon_ == "M2")
      QualityEnableME1e = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F2" && muon_ == "M3")
      QualityEnableME1f = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F3" && muon_ == "M1")
      QualityEnableME2a = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F3" && muon_ == "M2")
      QualityEnableME2b = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F3" && muon_ == "M3")
      QualityEnableME2c = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F4" && muon_ == "M1")
      QualityEnableME3a = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F4" && muon_ == "M2")
      QualityEnableME3b = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F4" && muon_ == "M3")
      QualityEnableME3c = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F5" && muon_ == "M1")
      QualityEnableME4a = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F5" && muon_ == "M2")
      QualityEnableME4b = ::strtol(writeValue_.c_str(), nullptr, 16);
    if (register_ == "CSR_LQE" && chip_ == "F5" && muon_ == "M3")
      QualityEnableME4c = ::strtol(writeValue_.c_str(), nullptr, 16);

    if (register_ == "CSR_KFL")
      kill_fiber = ::strtol(writeValue_.c_str(), nullptr, 16);

    if (register_ == "CSR_SFC" && chip_ == "SP") {
      unsigned int value = ::strtol(writeValue_.c_str(), nullptr, 16);
      singlesTrackOutput = ((value & 0x3000) >> 12);
    }

    if (register_ == "CNT_ETA" && chip_ == "SP") {
      unsigned int value = ::strtol(writeValue_.c_str(), nullptr, 16);
      eta_cnt = value;
    }

    // LATEST VERSION FROM CORE 2010-01-22 at http://www.phys.ufl.edu/~madorsky/sp/2010-01-22
    if (register_ == "DAT_ETA" && chip_ == "SP") {
      unsigned int value = ::strtol(writeValue_.c_str(), nullptr, 16);

      //std::cout<<"DAT_ETA SP value:"<<value<<std::endl;

      if (eta_cnt == 0)
        mindetap = value;
      if (eta_cnt == 1)
        mindetap_halo = value;

      if (eta_cnt >= 2 && eta_cnt < 10)
        etamin[eta_cnt - 2] = value;

      if (eta_cnt == 10)
        mindeta12_accp = value;
      if (eta_cnt == 11)
        mindeta13_accp = value;
      if (eta_cnt == 12)
        mindeta112_accp = value;
      if (eta_cnt == 13)
        mindeta113_accp = value;

      if (eta_cnt >= 14 && eta_cnt < 22)
        etamax[eta_cnt - 14] = value;

      if (eta_cnt == 22)
        maxdeta12_accp = value;
      if (eta_cnt == 23)
        maxdeta13_accp = value;
      if (eta_cnt == 24)
        maxdeta112_accp = value;
      if (eta_cnt == 25)
        maxdeta113_accp = value;

      if (eta_cnt >= 26 && eta_cnt < 33)
        etawin[eta_cnt - 26] = value;

      if (eta_cnt == 33)
        maxdphi12_accp = value;
      if (eta_cnt == 34)
        maxdphi13_accp = value;
      if (eta_cnt == 35)
        maxdphi112_accp = value;
      if (eta_cnt == 36)
        maxdphi113_accp = value;

      if (eta_cnt == 37)
        mindphip = value;
      if (eta_cnt == 38)
        mindphip_halo = value;

      if (eta_cnt == 39)
        straightp = value;
      if (eta_cnt == 40)
        curvedp = value;
      if (eta_cnt == 41)
        mbaPhiOff = value;
      if (eta_cnt == 42)
        mbbPhiOff = value;

      eta_cnt++;
    }

    // filling the firmware variables: SP MEZZANINE
    if (register_ == "FIRMWARE" && muon_ == "SP") {
      unsigned int value = atoi(writeValue_.c_str());
      firmwareSP = value;
    }

    // filling the firmware variables: Front FPGAs
    if (register_ == "FIRMWARE" && muon_ == "FA") {
      unsigned int value = atoi(writeValue_.c_str());
      firmwareFA = value;
    }

    // filling the firmware variables: DDU
    if (register_ == "FIRMWARE" && muon_ == "DD") {
      unsigned int value = atoi(writeValue_.c_str());
      firmwareDD = value;
    }

    // filling the firmware variables: VM
    if (register_ == "FIRMWARE" && muon_ == "VM") {
      unsigned int value = atoi(writeValue_.c_str());
      firmwareVM = value;
    }
  }

  pset.addParameter<bool>("run_core", run_core);
  pset.addParameter<bool>("trigger_on_ME1a", trigger_on_ME1a);
  pset.addParameter<bool>("trigger_on_ME1b", trigger_on_ME1b);
  pset.addParameter<bool>("trigger_on_ME2", trigger_on_ME2);
  pset.addParameter<bool>("trigger_on_ME3", trigger_on_ME3);
  pset.addParameter<bool>("trigger_on_ME4", trigger_on_ME4);
  pset.addParameter<bool>("trigger_on_MB1a", trigger_on_MB1a);
  pset.addParameter<bool>("trigger_on_MB1d", trigger_on_MB1d);

  pset.addParameter<unsigned int>("BXAdepth", BXAdepth);
  pset.addParameter<unsigned int>("useDT", useDT);
  pset.addParameter<unsigned int>("widePhi", widePhi);
  pset.addParameter<unsigned int>("PreTrigger", PreTrigger);

  // this were two old settings, not used anymore. Set them to zero
  // ask Alex if he can remove them altogether
  pset.addParameter<bool>("AllowALCTonly", AllowALCTonly);
  pset.addParameter<bool>("AllowCLCTonly", AllowCLCTonly);

  pset.addParameter<int>("CoreLatency", CoreLatency);
  pset.addParameter<bool>("rescaleSinglesPhi", rescaleSinglesPhi);

  pset.addParameter<unsigned int>("QualityEnableME1a", QualityEnableME1a);
  pset.addParameter<unsigned int>("QualityEnableME1b", QualityEnableME1b);
  pset.addParameter<unsigned int>("QualityEnableME1c", QualityEnableME1c);
  pset.addParameter<unsigned int>("QualityEnableME1d", QualityEnableME1d);
  pset.addParameter<unsigned int>("QualityEnableME1e", QualityEnableME1e);
  pset.addParameter<unsigned int>("QualityEnableME1f", QualityEnableME1f);
  pset.addParameter<unsigned int>("QualityEnableME2a", QualityEnableME2a);
  pset.addParameter<unsigned int>("QualityEnableME2b", QualityEnableME2b);
  pset.addParameter<unsigned int>("QualityEnableME2c", QualityEnableME2c);
  pset.addParameter<unsigned int>("QualityEnableME3a", QualityEnableME3a);
  pset.addParameter<unsigned int>("QualityEnableME3b", QualityEnableME3b);
  pset.addParameter<unsigned int>("QualityEnableME3c", QualityEnableME3c);
  pset.addParameter<unsigned int>("QualityEnableME4a", QualityEnableME4a);
  pset.addParameter<unsigned int>("QualityEnableME4b", QualityEnableME4b);
  pset.addParameter<unsigned int>("QualityEnableME4c", QualityEnableME4c);

  pset.addParameter<unsigned int>("kill_fiber", kill_fiber);
  pset.addParameter<unsigned int>("singlesTrackOutput", singlesTrackOutput);

  // add the DAT_ETA registers to the pset
  pset.addParameter<unsigned int>("mindetap", mindetap);
  pset.addParameter<unsigned int>("mindetap_halo", mindetap_halo);

  pset.addParameter<std::vector<unsigned int> >("EtaMin", etamin);

  pset.addParameter<unsigned int>("mindeta12_accp", mindeta12_accp);
  pset.addParameter<unsigned int>("mindeta13_accp", mindeta13_accp);
  pset.addParameter<unsigned int>("mindeta112_accp", mindeta112_accp);
  pset.addParameter<unsigned int>("mindeta113_accp", mindeta113_accp);

  pset.addParameter<std::vector<unsigned int> >("EtaMax", etamax);

  pset.addParameter<unsigned int>("maxdeta12_accp", maxdeta12_accp);
  pset.addParameter<unsigned int>("maxdeta13_accp", maxdeta13_accp);
  pset.addParameter<unsigned int>("maxdeta112_accp", maxdeta112_accp);
  pset.addParameter<unsigned int>("maxdeta113_accp", maxdeta113_accp);

  pset.addParameter<std::vector<unsigned int> >("EtaWindows", etawin);

  pset.addParameter<unsigned int>("maxdphi12_accp", maxdphi12_accp);
  pset.addParameter<unsigned int>("maxdphi13_accp", maxdphi13_accp);
  pset.addParameter<unsigned int>("maxdphi112_accp", maxdphi112_accp);
  pset.addParameter<unsigned int>("maxdphi113_accp", maxdphi113_accp);

  pset.addParameter<unsigned int>("mindphip", mindphip);
  pset.addParameter<unsigned int>("mindphip_halo", mindphip_halo);

  pset.addParameter<unsigned int>("straightp", straightp);
  pset.addParameter<unsigned int>("curvedp", curvedp);
  pset.addParameter<unsigned int>("mbaPhiOff", mbaPhiOff);
  pset.addParameter<unsigned int>("mbbPhiOff", mbbPhiOff);

  pset.addParameter<unsigned int>("firmwareSP", firmwareSP);
  pset.addParameter<unsigned int>("firmwareFA", firmwareFA);
  pset.addParameter<unsigned int>("firmwareDD", firmwareDD);
  pset.addParameter<unsigned int>("firmwareVM", firmwareVM);

  // printout
  LogDebug("L1MuCSCTFConfiguration")
      << "\nCORE CONFIGURATION AFTER READING THE DBS VALUES"
      << "\nrun_core=" << run_core << "\ntrigger_on_ME1a=" << trigger_on_ME1a << "\ntrigger_on_ME1b=" << trigger_on_ME1b
      << "\ntrigger_on_ME2=" << trigger_on_ME2 << "\ntrigger_on_ME3=" << trigger_on_ME3
      << "\ntrigger_on_ME4=" << trigger_on_ME4 << "\ntrigger_on_MB1a=" << trigger_on_MB1a
      << "\ntrigger_on_MB1d=" << trigger_on_MB1d

      << "\nBXAdepth=" << BXAdepth << "\nuseDT=" << useDT << "\nwidePhi=" << widePhi << "\nPreTrigger=" << PreTrigger

      << "\nCoreLatency=" << CoreLatency << "\nrescaleSinglesPhi=" << rescaleSinglesPhi

      << "\n\nVARIOUS CONFIGURATION PARAMETERS AFTER READING THE DBS VALUES"
      << "\nAllowALCTonly=" << AllowALCTonly << "\nAllowCLCTonly=" << AllowCLCTonly

      << "\nQualityEnableME1a=" << QualityEnableME1a << "\nQualityEnableME1b=" << QualityEnableME1b
      << "\nQualityEnableME1c=" << QualityEnableME1c << "\nQualityEnableME1d=" << QualityEnableME1d
      << "\nQualityEnableME1e=" << QualityEnableME1e << "\nQualityEnableME1f=" << QualityEnableME1f
      << "\nQualityEnableME2a=" << QualityEnableME2a << "\nQualityEnableME2b=" << QualityEnableME2b
      << "\nQualityEnableME2c=" << QualityEnableME2c << "\nQualityEnableME3a=" << QualityEnableME3a
      << "\nQualityEnableME3b=" << QualityEnableME3b << "\nQualityEnableME3c=" << QualityEnableME3c
      << "\nQualityEnableME4a=" << QualityEnableME4a << "\nQualityEnableME4b=" << QualityEnableME4b
      << "\nQualityEnableME4c=" << QualityEnableME4c

      << "\nkill_fiber=" << kill_fiber << "\nsinglesTrackOutput=" << singlesTrackOutput

      << "\n\nDAT_ETA AFTER READING THE DBS VALUES"
      << "\nmindetap     =" << mindetap << "\nmindetap_halo=" << mindetap_halo

      << "\netamin[0]=" << etamin[0] << "\netamin[1]=" << etamin[1] << "\netamin[2]=" << etamin[2]
      << "\netamin[3]=" << etamin[3] << "\netamin[4]=" << etamin[4] << "\netamin[5]=" << etamin[5]
      << "\netamin[6]=" << etamin[6] << "\netamin[7]=" << etamin[7]

      << "\nmindeta12_accp =" << mindeta12_accp << "\nmindeta13_accp =" << mindeta13_accp
      << "\nmindeta112_accp=" << mindeta112_accp << "\nmindeta113_accp=" << mindeta113_accp

      << "\netamax[0]=" << etamax[0] << "\netamax[1]=" << etamax[1] << "\netamax[2]=" << etamax[2]
      << "\netamax[3]=" << etamax[3] << "\netamax[4]=" << etamax[4] << "\netamax[5]=" << etamax[5]
      << "\netamax[6]=" << etamax[6] << "\netamax[7]=" << etamax[7]

      << "\nmaxdeta12_accp =" << maxdeta12_accp << "\nmaxdeta13_accp =" << maxdeta13_accp
      << "\nmaxdeta112_accp=" << maxdeta112_accp << "\nmaxdeta113_accp=" << maxdeta113_accp

      << "\netawin[0]=" << etawin[0] << "\netawin[1]=" << etawin[1] << "\netawin[2]=" << etawin[2]
      << "\netawin[3]=" << etawin[3] << "\netawin[4]=" << etawin[4] << "\netawin[5]=" << etawin[5]
      << "\netawin[6]=" << etawin[6]

      << "\nmaxdphi12_accp =" << maxdphi12_accp << "\nmaxdphi13_accp =" << maxdphi13_accp
      << "\nmaxdphi112_accp=" << maxdphi112_accp << "\nmaxdphi113_accp=" << maxdphi113_accp

      << "\nmindphip     =" << mindphip << "\nmindphip_halo=" << mindphip_halo

      << "\nstraightp=" << straightp << "\ncurvedp  =" << curvedp << "\nmbaPhiOff=" << mbaPhiOff
      << "\nmbbPhiOff=" << mbbPhiOff

      << "\n\nFIRMWARE VERSIONS AFTER READING THE DBS VALUES"
      << "\nSP: " << firmwareSP << "\nFA: " << firmwareFA << "\nDD: " << firmwareDD << "\nVM: " << firmwareVM;

  // ---------------------------------------------------------

  return pset;
}
