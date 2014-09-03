def l1tlayout(i, p, *rows): i["00 Shift/L1T/" + p] = DQMItem(layout=rows)

def l1tlayout_coll(i, p, *rows): i["00 Shift/L1T/Collisions/" + p] = DQMItem(layout=rows)

l1tlayout(dqmitems,"00 Global Trigger Algorithm Bits",
  	[{'path': "L1T/L1TGT/algo_bits", 'description': "Global Trigger bits. x-axis: GT algorithm number; y-axis: number of events with given bit on.  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"01 Global Trigger Technical Bits",
	[{'path': "L1T/L1TGT/tt_bits", 'description': "Global Trigger Technical bits. x-axis: technical trigger algorithm number; y-axis: number of events with given bit on.  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"02 Global Muon Trigger Occupancy",
	[{'path': "L1T/L1TGMT/GMT_etaphi", 'description': "GMT Phi vs Eta. x-axis: phi in degrees; y-axis: eta; z-axis: number of GMT candidates in given phi/eta bin.  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"03 Global Muon Trigger Contributions",
  	[{'path': "L1T/L1TGMT/Regional_trigger", 'description': "Regional Muon Trigger Contribution. x-axis: muon regional trigger; y-axis: number of triggers from given subsystem.  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"04 Global Muon Trigger DT-CSC BX Correlation",
  	[{'path': "L1T/L1TGMT/bx_DT_vs_CSC", 'description': "DT-CSC BX correlation. The red maximum should be in the middle of the 3x3 histograms. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"05 Global Muon Trigger DT-RPC BX Correlation",
  	[{'path': "L1T/L1TGMT/bx_DT_vs_RPC", 'description': "DT-RPC BX correlation. The red maximum should be in the middle of the 3x3 histograms. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"06 Global Muon Trigger CSC-RPC BX Correlation",
  	[{'path': "L1T/L1TGMT/bx_CSC_vs_RPC", 'description': "CSC-RPC BX correlation. The red maximum should be in the middle of the 3x3 histograms. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"07 Global Calorimeter Trigger Iso EM Occupancy",
  	[{'path': "L1T/L1TGCT/IsoEmRankEtaPhi", 'description': "(Eta, Phi) map of Isolated Electron Occupancy. x-axis: phi (0-17) y-axis: eta (0-21) z-axis: number of isolated electron candidates. Electrons are not found in HF so eta bins 0-3 and 18-21 should be empty. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"08 Global Calorimeter Trigger NonIso EM Occupancy",
	[{'path': "L1T/L1TGCT/NonIsoEmRankEtaPhi", 'description': "(Eta,Phi) map of Non Isolated Electron Occupancy. x-axis: phi (0-17) y-axis: eta (0-21) z-axis: number of non-isolated electron candidates. Electrons are not found in HF so eta bins 0-3 and 18-21 should be empty.  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"09 Global Calorimeter Trigger Jets Occupancy",
  	[{'path': "L1T/L1TGCT/AllJetsEtEtaPhi", 'description': "(Eta,Phi) map of Central Jet Occupancy. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"10 Global Calorimeter Trigger Tau Jets Occupancy",
	[{'path': "L1T/L1TGCT/TauJetsEtEtaPhi", 'description': "(Eta,Phi) map of Tau Jet Occupancy. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"11 Physics Trigger Rate",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/Physics_Trigger_Rate", 'description': "Physics Trigger Rate. x-axis: Time(lumisection); y-axis: Rate (Hz).  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

l1tlayout(dqmitems,"12 Random Trigger Rate",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/Random_Trigger_Rate", 'description': "Random Trigger Rate. x-axis: Time(lumisection); y-axis: Rate (Hz).  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])


l1tlayout_coll(dqmitems,"00 Rate BSC MinBias (Tech Bit 41)",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_041", 'description': "Rate of Minbias BSC trigger"}])

l1tlayout_coll(dqmitems,"01 Rate BPTX AND  (Tech Bit 4)",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_004", 'description': "Rate of BPTX trigger"}])
