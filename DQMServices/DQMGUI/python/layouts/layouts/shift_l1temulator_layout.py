from .adapt_to_new_backend import *
dqmitems={}

moreInfoStr = "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTriggerEmulator\">here</a>."

def l1temulayout(i, p, *rows): i["00 Shift/L1TEMU/" + p] = rows

l1temulayout(dqmitems,"00 - uGMT - data vs emulator misMatch ratio",
             [{'path': "L1TEMU/L1TdeStage2uGMT/data_vs_emulator_comparison/mismatchRatio", 'description': "uGMT - data vs emulator misMatch ratio. "+moreInfoStr}])

l1temulayout(dqmitems,"01 - BMTF - data vs emulator misMatch ratio",
             [{'path': "L1TEMU/L1TdeStage2BMTF/mismatchRatio", 'description': "BMTF - data vs emulator misMatch ratio. "+moreInfoStr}])

l1temulayout(dqmitems,"02 - OMTF - data vs emulator misMatch ratio",
             [{'path': "L1TEMU/L1TdeStage2OMTF/mismatchRatio", 'description': "OMTF - data vs emulator misMatch ratio. "+moreInfoStr}])

l1temulayout(dqmitems,"03 - EMTF - data vs emulator misMatch ratio",
             [{'path': "L1TEMU/L1TdeStage2EMTF/mismatchRatio", 'description': "EMTF - data vs emulator misMatch ratio. "+moreInfoStr}])

l1temulayout(dqmitems,"04 - CaloLayer1 - data vs emulator misMatch ratio",
             [{'path': "L1TEMU/L1TdeStage2CaloLayer1/dataEmulSummary", 'description': "CaloLayer1 - data vs emulator misMatch ratio. "+moreInfoStr}])

l1temulayout(dqmitems,"05 - CaloLayer2 - High Level Data-Emulator Agreement Summary",
             [{'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/High level summary",'description': "Event by event comparison Data-Emulator Agreement Summary. "+moreInfoStr}])

l1temulayout(dqmitems,"06 - CaloLayer2 - Jet Data-Emulator Agreement Summary",
             [{'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Jet Agreement Summary",'description': "Jet Data-Emulator Agreement Summary. "+moreInfoStr}])

l1temulayout(dqmitems,"07 - CaloLayer2 - EG Data-Emulator Agreement Summary",
             [{'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/EG Agreement Summary",'description': "EG Data-Emulator Agreement Summary. "+moreInfoStr}])

l1temulayout(dqmitems,"08 - CaloLayer2 - Tau Data-Emulator Agreement Summary",
             [{'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Tau Agreement Summary",'description': "Tau Data-Emulator Agreement Summary. "+moreInfoStr}])

l1temulayout(dqmitems,"09 - CaloLayer2 - Energy Sum Data-Emulator Agreement Summary",
             [{'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Energy Sum Agreement Summary",'description': "Energy Sum Data-Emulator Agreement Summary. "+moreInfoStr}])

l1temulayout(dqmitems,"10 - CaloLayer2 - Problem Summary",
             [{'path': "L1TEMU/L1TStage2CaloLayer2/L1TdeStage2CaloLayer2/Problem Summary",'description': "Problematic Event Summary. "+moreInfoStr}])

l1temulayout(dqmitems,"11 - uGT - Data-Emulator misMatch ratio",
             [{ 'path': "L1TEMU/L1TdeStage2uGT/dataEmulMismatchRatio_CentralBX",'description': "uGT Data-Emulator misMatch ratio -- Central BX. "+moreInfoStr}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
