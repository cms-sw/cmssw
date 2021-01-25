from .adapt_to_new_backend import *
dqmitems={}

def l1temulayout(i, p, *rows): i["L1TEMU/Layouts/" + p] = rows

# The quick collection is defined in ../workspaces-online.py
# for Online DQM, but we want to also include descriptions
# So we reference the 'QuickCollection' layout created here
def l1t_quickCollection(i, name, *rows):
  i["L1TEMU/Layouts/Stage2-QuickCollection/%s" % name] = rows

# If you add a plot here, remember to add the reference to ../workspaces-online.py
l1t_quickCollection(dqmitems, "00 - CaloTower Data-Emulator Status",
  [{
    'path': "L1TEMU/L1TdeStage2CaloLayer1/dataEmulSummary",
    'description': "This is a fraction of events with data-emulator mismatches, should be at 1.",
    'draw': { 'withref': "no" }
  }])

l1t_quickCollection(dqmitems,"01 - uGMT Data-Emulator misMatch ratio",
  [{
    'path': "L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison/mismatchRatio",
    'description': "uGMT - data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }])

l1t_quickCollection(dqmitems,"02 - BMTF Data-Emulator misMatch ratio",
  [{
    'path': "L1TEMU/L1TdeStage2BMTF/mismatchRatio",
    'description': "BMTF - data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }])

l1t_quickCollection(dqmitems,"03 - OMTF Data-Emulator misMatch ratio",
  [{
    'path': "L1TEMU/L1TdeStage2OMTF/mismatchRatio",
    'description': "OMTF - data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }])

l1t_quickCollection(dqmitems,"04 - EMTF Data-Emulator misMatch ratio",
  [{
    'path': "L1TEMU/L1TdeStage2EMTF/mismatchRatio",
    'description': "EMTF - data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"05 - Calo Layer2 High Level Data-Emulator Agreement Summary",
  [{
    'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/High level summary",
    'description': "Event by event comparison Data-Emulator Agreement Summary",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"06 - Calo Layer2 Jet Data-Emulator Agreement Summary",
  [{
    'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Jet Agreement Summary",
    'description': "Jet Data-Emulator Agreement Summary",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"07 - Calo Layer2 EG Data-Emulator Agreement Summary",
  [{
    'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/EG Agreement Summary",
    'description': "EG Data-Emulator Agreement Summary",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"08 - Calo Layer2 Tau Data-Emulator Agreement Summary",
  [{
    'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Tau Agreement Summary",
    'description': "Tau Data-Emulator Agreement Summary",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"09 - Calo Layer2 Energy Sum Data-Emulator Agreement Summary",
  [{
    'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Energy Sum Agreement Summary",
    'description': "Energy Sum Data-Emulator Agreement Summary",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"10 - Calo Layer2 Problem Summary",
  [{
    'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Problem Summary",
    'description': "Problematic Event Summary",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"11 - uGT Data-Emulator misMatch ratio",
  [{
    'path': "L1TEMU/L1TdeStage2uGT/dataEmulMismatchRatio_CentralBX",
    'description': "uGT Data-Emulator misMatch ratio -- Central BX",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems,"12 - uGMT Intermediate Muon Data-Emulator misMatch ratio",
  [{
    'path': "L1TEMU/L1TdeStage2uGMT/intermediate_muons/BMTF/data_vs_emulator_comparison/mismatchRatio",
    'description': "uGMT - intermediate muon BMTF data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1TEMU/L1TdeStage2uGMT/intermediate_muons/OMTF_neg/data_vs_emulator_comparison/mismatchRatio",
    'description': "uGMT - intermediate muon OMTF neg data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1TEMU/L1TdeStage2uGMT/intermediate_muons/OMTF_pos/data_vs_emulator_comparison/mismatchRatio",
    'description': "uGMT - intermediate muon OMTF pos data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1TEMU/L1TdeStage2uGMT/intermediate_muons/EMTF_neg/data_vs_emulator_comparison/mismatchRatio",
    'description': "uGMT - intermediate muon EMTF neg data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1TEMU/L1TdeStage2uGMT/intermediate_muons/EMTF_pos/data_vs_emulator_comparison/mismatchRatio",
    'description': "uGMT - intermediate muon EMTF pos data vs emulator misMatch ratio. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>.",
    'draw': { 'withref': "no" }
  }])


###############################################
### From here down is legacy/stage1 trigger ###
###           All in Legacy folder          ###
###############################################

# def l1temucommon(i, dir, name):i["L1TEMU/Layouts/00-Global-Summary/%s" % name] = \
#     DQMItem(layout=[["L1TEMU/%s/%s" % (dir, name)]])

# l1temucommon(dqmitems, "common", "sysrates")
# l1temucommon(dqmitems, "common", "errorflag")
# l1temucommon(dqmitems, "common", "sysncandData")
# l1temucommon(dqmitems, "common", "sysncandEmul")

# def l1t_rct_expert(i, p, *rows): i["L1TEMU/Layouts/03-L1TdeRCT-Summary/" + p] = rows
# l1t_rct_expert(dqmitems, "rctInputTPGEcalOcc",
#   [{ 'path': "L1TEMU/L1TdeRCT/rctInputTPGEcalOcc", 'description': "Input ECAL TPs occupancy, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctInputTPGHcalOcc",
#   [{ 'path': "L1TEMU/L1TdeRCT/rctInputTPGHcalOcc", 'description': "Input HCAL TPs occupancy, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctIsoEmEff1",
#   [{ 'path': "L1TEMU/L1TdeRCT/IsoEm/rctIsoEmEff1", 'description': "Isolated electrons efficiency, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctNisoEmEff1",
#   [{ 'path': "L1TEMU/L1TdeRCT/NisoEm/rctNisoEmEff1", 'description': "Non-Isolated electrons efficiency, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctRegEff2D",
#   [{ 'path': "L1TEMU/L1TdeRCT/RegionData/rctRegEff2D", 'description': "Regional efficiency, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctIsoEmOvereff",
#   [{ 'path': "L1TEMU/L1TdeRCT/IsoEm/rctIsoEmOvereff", 'description': "Isolated electrons overefficiency, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctNisoEmOvereff",
#   [{ 'path': "L1TEMU/L1TdeRCT/NisoEm/rctNisoEmOvereff", 'description': "Non-Isolated electrons overefficiency, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "rctRegOvereff2D",
#   [{ 'path': "L1TEMU/L1TdeRCT/RegionData/rctRegOvereff2D", 'description': "Regional overefficiency, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "RctEmIsoEmOccEtaPhi",
#   [{ 'path': "L1TEMU/L1TdeRCT/IsoEm/ServiceData/rctIsoEmDataOcc", 'description': "EmIsoOcc, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "RctEmNonIsoEmOccEtaPhi",
#   [{ 'path': "L1TEMU/L1TdeRCT/NisoEm/ServiceData/rctNisoEmDataOcc", 'description': "RctEmNonIsoEmOccEtaPhi, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "RctRegionsOccEtaPhi",
#   [{ 'path': "L1TEMU/L1TdeRCT/RegionData/ServiceData/rctRegDataOcc2D", 'description': "RctRegionsOccEtaPhi, For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
