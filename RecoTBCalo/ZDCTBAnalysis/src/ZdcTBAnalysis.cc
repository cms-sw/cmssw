#include "RecoTBCalo/ZDCTBAnalysis/interface/ZdcTBAnalysis.h"
#include <sstream>
#include <iostream>
#include <vector>

ZdcTBAnalysis::ZdcTBAnalysis() { ; }

void ZdcTBAnalysis::setup(const std::string& outFileName) {
  TString outName = outFileName;
  outFile = new TFile(outName, "RECREATE");
  ZdcAnalize = new TTree("ZdcAnaTree", "ZdcAnaTree");
  ZdcAnalize->Branch("Trigger",
                     0,
                     "run/I:event/I:beamTrigger/I:fakeTrigger/I:"
                     "pedestalTrigger/I:outSpillPedestalTrigger/I:inSpillPedestalTrigger/I:"
                     "laserTrigger/I:laserTrigger/I:ledTrigger/I:spillTrigger/I");
  ZdcAnalize->Branch("TDC",
                     0,
                     "trigger/D:ttcL1/D:beamCoincidence/D:laserFlash/D:qiePhase/D:"
                     "TTOF1/D:TTOF2/D:m1[5]/D:m2[5]/D:m3[5]/D:"
                     "s1[5]/D:s2[5]/D:s3[5]/D:s4[5]/D:"
                     "bh1[5]/D:bh2[5]/D:bh3[5]/D:bh4[5]/D");
  ZdcAnalize->Branch("ADC",
                     0,
                     "VM/D:V3/D:V6/D:VH1/D:VH2/D:VH3/D:VH4/D:Ecal7x7/D:"
                     "Sci521/D:Sci528/D:CK1/D:CK2/D:CK3/D:SciVLE/D:S1/D:S2/D:S3/D:S4/D:"
                     "VMF/D:VMB/D:VM1/D:VM2/D:VM3/D:VM4/D:VM5/D:VM6/D:VM7/D:VM8/D:"
                     "TOF1/D:TOF2/D:BH1/D:BH2/D:BH3/BH4/D");
  ZdcAnalize->Branch("Chamb",
                     0,
                     "WCAx[5]/D:WCAy[5]/D:WCBx[5]/D:WCBy[5]/D:"
                     "WCCx[5]/D:WCCy[5]/D:WCDx[5]/D:WCDy[5]/D:WCEx[5]/D:WCEy[5]/D:"
                     "WCFx[5]/D:WCFy[5]/D:WCGx[5]/D:WCGy[5]/D:WCHx[5]/D:WCHy[5]/D");
  ZdcAnalize->Branch("ZDCP",
                     0,
                     "zdcHAD1/D:zdcHAD2/D:zdcHAD3/D:zdcHAD4/D:"
                     "zdcEM1/D:zdcEM2/D:zdcEM3/D:zdcEM4/D:zdcEM5/D:"
                     "zdcScint1/D:zdcScint2/D:"
                     "zdcExtras[7]/D");
  ZdcAnalize->Branch("ZDCN",
                     0,
                     "zdcHAD1/D:zdcHAD2/D:zdcHAD3/D:zdcHAD4/D:"
                     "zdcEM1/D:zdcEM2/D:zdcEM3/D:zdcEM4/D:zdcEM5/D:"
                     "zdcScint1/D:zdcScint2/D:"
                     "zdcExtras[7]/D");
  ZdcAnalize->GetBranch("Trigger")->SetAddress(&trigger);
  ZdcAnalize->GetBranch("TDC")->SetAddress(&tdc);
  ZdcAnalize->GetBranch("ADC")->SetAddress(&adc);
  ZdcAnalize->GetBranch("Chamb")->SetAddress(&chamb);
  ZdcAnalize->GetBranch("ZDCP")->SetAddress(&zdcp);
  ZdcAnalize->GetBranch("ZDCN")->SetAddress(&zdcn);
  ZdcAnalize->SetAutoSave();
}

void ZdcTBAnalysis::analyze(const HcalTBTriggerData& trg) {
  // trigger
  trigger.runNum = runNumber = trg.runNumber();
  trigger.eventNum = eventNumber = trg.eventNumber();
  isBeamTrigger = trg.wasBeamTrigger();
  isFakeTrigger = trg.wasFakeTrigger();
  isCalibTrigger = trg.wasSpillIgnorantPedestalTrigger();
  isOutSpillPedestalTrigger = trg.wasOutSpillPedestalTrigger();
  isInSpillPedestalTrigger = trg.wasInSpillPedestalTrigger();
  isLaserTrigger = trg.wasLaserTrigger();
  isLedTrigger = trg.wasLEDTrigger();
  isSpillTrigger = trg.wasInSpill();

  trigger.beamTrigger = trigger.fakeTrigger = trigger.calibTrigger = trigger.outSpillPedestalTrigger =
      trigger.inSpillPedestalTrigger = trigger.laserTrigger = trigger.ledTrigger = trigger.spillTrigger = 0;

  if (isBeamTrigger)
    trigger.beamTrigger = 1;
  if (isFakeTrigger)
    trigger.fakeTrigger = 1;
  if (isCalibTrigger)
    trigger.calibTrigger = 1;
  if (isOutSpillPedestalTrigger)
    trigger.outSpillPedestalTrigger = 1;
  if (isInSpillPedestalTrigger)
    trigger.inSpillPedestalTrigger = 1;
  if (isLaserTrigger)
    trigger.laserTrigger = 1;
  if (isLedTrigger)
    trigger.ledTrigger = 1;
  if (isSpillTrigger)
    trigger.spillTrigger = 1;
}

void ZdcTBAnalysis::analyze(const HcalTBTiming& times) {
  //times
  tdc.trigger = trigger_time = times.triggerTime();
  tdc.ttcL1 = ttc_L1a_time = times.ttcL1Atime();
  tdc.laserFlash = laser_flash = times.laserFlash();
  tdc.qiePhase = qie_phase = times.qiePhase();
  tdc.TOF1 = TOF1_time = times.TOF1Stime();
  tdc.TOF2 = TOF2_time = times.TOF2Stime();

  // just take 5 first hits of multihit tdc (5 tick cycles)
  int indx = 0;
  int indTop = 5;
  for (indx = 0; indx < times.BeamCoincidenceCount(); indx++)
    if (indx < indTop)
      tdc.beamCoincidence[indx] = beam_coincidence[indx] = times.BeamCoincidenceHits(indx);
  for (indx = 0; indx < times.M1Count(); indx++)
    if (indx < indTop)
      tdc.m1[indx] = m1hits[indx] = times.M1Hits(indx);
  for (indx = 0; indx < times.M2Count(); indx++)
    if (indx < indTop)
      tdc.m2[indx] = m2hits[indx] = times.M2Hits(indx);
  for (indx = 0; indx < times.M3Count(); indx++)
    if (indx < indTop)
      tdc.m3[indx] = m3hits[indx] = times.M3Hits(indx);
  for (indx = 0; indx < times.S1Count(); indx++)
    if (indx < indTop)
      tdc.s1[indx] = s1hits[indx] = times.S1Hits(indx);
  for (indx = 0; indx < times.S2Count(); indx++)
    if (indx < indTop)
      tdc.s2[indx] = s2hits[indx] = times.S2Hits(indx);
  for (indx = 0; indx < times.S3Count(); indx++)
    if (indx < indTop)
      tdc.s3[indx] = s3hits[indx] = times.S3Hits(indx);
  for (indx = 0; indx < times.S4Count(); indx++)
    if (indx < indTop)
      tdc.s4[indx] = s4hits[indx] = times.S4Hits(indx);
  for (indx = 0; indx < times.BH1Count(); indx++)
    if (indx < indTop)
      tdc.bh1[indx] = bh1hits[indx] = times.BH1Hits(indx);
  for (indx = 0; indx < times.BH2Count(); indx++)
    if (indx < indTop)
      tdc.bh2[indx] = bh2hits[indx] = times.BH2Hits(indx);
  for (indx = 0; indx < times.BH3Count(); indx++)
    if (indx < indTop)
      tdc.bh3[indx] = bh3hits[indx] = times.BH3Hits(indx);
  for (indx = 0; indx < times.BH4Count(); indx++)
    if (indx < indTop)
      tdc.bh4[indx] = bh4hits[indx] = times.BH4Hits(indx);
}

void ZdcTBAnalysis::analyze(const HcalTBBeamCounters& bc) {
  //beam counters
  adc.VM = VMadc = bc.VMadc();
  adc.V3 = V3adc = bc.V3adc();
  adc.V6 = V6adc = bc.V6adc();
  adc.VH1 = VH1adc = bc.VH1adc();
  adc.VH2 = VH2adc = bc.VH2adc();
  adc.VH3 = VH3adc = bc.VH3adc();
  adc.VH4 = VH4adc = bc.VH4adc();
  adc.Ecal7x7 = Ecal7x7adc = bc.Ecal7x7();
  adc.Sci521 = Sci521adc = bc.Sci521adc();
  adc.Sci528 = Sci528adc = bc.Sci528adc();
  adc.CK1 = CK1adc = bc.CK1adc();
  adc.CK2 = CK2adc = bc.CK2adc();
  adc.CK3 = CK3adc = bc.CK3adc();
  adc.SciVLE = SciVLEadc = bc.SciVLEadc();
  adc.S1 = S1adc = bc.S1adc();
  adc.S2 = S2adc = bc.S2adc();
  adc.S3 = S3adc = bc.S3adc();
  adc.S4 = S4adc = bc.S4adc();
  adc.VMF = VMFadc = bc.VMFadc();
  adc.VMB = VMBadc = bc.VMBadc();
  adc.VM1 = VM1adc = bc.VM1adc();
  adc.VM2 = VM2adc = bc.VM2adc();
  adc.VM3 = VM3adc = bc.VM3adc();
  adc.VM4 = VM4adc = bc.VM4adc();
  adc.VM5 = VM5adc = bc.VM5adc();
  adc.VM6 = VM6adc = bc.VM6adc();
  adc.VM7 = VM7adc = bc.VM7adc();
  adc.VM8 = VM8adc = bc.VM8adc();
  adc.TOF1 = TOF1adc = bc.TOF1Sadc();
  adc.TOF2 = TOF2adc = bc.TOF2Sadc();
  adc.BH1 = BH1adc = bc.BH1adc();
  adc.BH2 = BH2adc = bc.BH2adc();
  adc.BH3 = BH3adc = bc.BH3adc();
  adc.BH4 = BH4adc = bc.BH4adc();
}

void ZdcTBAnalysis::analyze(const HcalTBEventPosition& chpos) {
  //chambers position
  chpos.getChamberHits('A', wcax, wcay);
  chpos.getChamberHits('B', wcbx, wcby);
  chpos.getChamberHits('C', wccx, wccy);
  chpos.getChamberHits('D', wcdx, wcdy);
  chpos.getChamberHits('E', wcex, wcey);
  chpos.getChamberHits('F', wcfx, wcfy);
  chpos.getChamberHits('G', wcgx, wcgy);
  chpos.getChamberHits('H', wchx, wchy);

  // just take 5 first hits of chambers (5 tick cycles)
  unsigned int indTop = 5;
  unsigned int indx = 0;
  for (indx = 0; indx < wcax.size(); indx++)
    if (indx < indTop)
      chamb.WCAx[indx] = wcax[indx];
  for (indx = 0; indx < wcay.size(); indx++)
    if (indx < indTop)
      chamb.WCAy[indx] = wcay[indx];
  for (indx = 0; indx < wcbx.size(); indx++)
    if (indx < indTop)
      chamb.WCBx[indx] = wcbx[indx];
  for (indx = 0; indx < wcby.size(); indx++)
    if (indx < indTop)
      chamb.WCBy[indx] = wcby[indx];
  for (indx = 0; indx < wccx.size(); indx++)
    if (indx < indTop)
      chamb.WCCx[indx] = wccx[indx];
  for (indx = 0; indx < wccy.size(); indx++)
    if (indx < indTop)
      chamb.WCCy[indx] = wccy[indx];
  for (indx = 0; indx < wcdx.size(); indx++)
    if (indx < indTop)
      chamb.WCDx[indx] = wcdx[indx];
  for (indx = 0; indx < wcdy.size(); indx++)
    if (indx < indTop)
      chamb.WCDy[indx] = wcdy[indx];
  for (indx = 0; indx < wcdx.size(); indx++)
    if (indx < indTop)
      chamb.WCEx[indx] = wcex[indx];
  for (indx = 0; indx < wcey.size(); indx++)
    if (indx < indTop)
      chamb.WCEy[indx] = wcey[indx];
  for (indx = 0; indx < wcfx.size(); indx++)
    if (indx < indTop)
      chamb.WCFx[indx] = wcfx[indx];
  for (indx = 0; indx < wcfy.size(); indx++)
    if (indx < indTop)
      chamb.WCFy[indx] = wcfy[indx];
  for (indx = 0; indx < wcgx.size(); indx++)
    if (indx < indTop)
      chamb.WCGx[indx] = wcgx[indx];
  for (indx = 0; indx < wcgy.size(); indx++)
    if (indx < indTop)
      chamb.WCGy[indx] = wcgy[indx];
  for (indx = 0; indx < wchx.size(); indx++)
    if (indx < indTop)
      chamb.WCHx[indx] = wchx[indx];
  for (indx = 0; indx < wchy.size(); indx++)
    if (indx < indTop)
      chamb.WCHy[indx] = wchy[indx];
}

void ZdcTBAnalysis::analyze(const ZDCRecHitCollection& zdcHits) {
  // zdc hits
  std::cout << "****************************************************" << std::endl;
  ZDCRecHitCollection::const_iterator i;
  for (i = zdcHits.begin(); i != zdcHits.end(); i++) {
    energy = i->energy();
    detID = i->id();
    iside = detID.zside();
    isection = detID.section();
    ichannel = detID.channel();
    idepth = detID.depth();
    std::cout << "energy: " << energy << " detID: " << detID << " side: " << iside << " section: " << isection
              << " channel: " << ichannel << " depth: " << idepth << std::endl;

    if (iside > 0) {
      if (ichannel == 1 && isection == 1)
        zdcp.zdcEMMod1 = energy;
      if (ichannel == 2 && isection == 1)
        zdcp.zdcEMMod2 = energy;
      if (ichannel == 3 && isection == 1)
        zdcp.zdcEMMod3 = energy;
      if (ichannel == 4 && isection == 1)
        zdcp.zdcEMMod4 = energy;
      if (ichannel == 5 && isection == 1)
        zdcp.zdcEMMod5 = energy;
      if (ichannel == 1 && isection == 2)
        zdcp.zdcHADMod1 = energy;
      if (ichannel == 2 && isection == 2)
        zdcp.zdcHADMod2 = energy;
      if (ichannel == 3 && isection == 2)
        zdcp.zdcHADMod3 = energy;
      if (ichannel == 4 && isection == 2)
        zdcp.zdcHADMod4 = energy;
      if (ichannel == 1 && isection == 3)
        zdcp.zdcScint1 = energy;
    }
    if (iside < 0) {
      if (ichannel == 1 && isection == 1)
        zdcn.zdcEMMod1 = energy;
      if (ichannel == 2 && isection == 1)
        zdcn.zdcEMMod2 = energy;
      if (ichannel == 3 && isection == 1)
        zdcn.zdcEMMod3 = energy;
      if (ichannel == 4 && isection == 1)
        zdcn.zdcEMMod4 = energy;
      if (ichannel == 5 && isection == 1)
        zdcn.zdcEMMod5 = energy;
      if (ichannel == 1 && isection == 2)
        zdcn.zdcHADMod1 = energy;
      if (ichannel == 2 && isection == 2)
        zdcn.zdcHADMod2 = energy;
      if (ichannel == 3 && isection == 2)
        zdcn.zdcHADMod3 = energy;
      if (ichannel == 4 && isection == 2)
        zdcn.zdcHADMod4 = energy;
      if (ichannel == 1 && isection == 3)
        zdcn.zdcScint1 = energy;
    }
  }
}

void ZdcTBAnalysis::fillTree() { ZdcAnalize->Fill(); }

void ZdcTBAnalysis::done() {
  ZdcAnalize->Print();
  outFile->cd();
  ZdcAnalize->Write();
  outFile->Close();
}
