def l1toverviewlayout(i, p, *rows): i["Collisions/L1TFeedBack/" + p] = DQMItem(layout=rows)

l1toverviewlayout(dqmitems,"00 ECAL TP Spectra",
                   [{ 'path': "EcalEndcap/EETriggerTowerTask/EETTT Et spectrum Real Digis EE -", 'description': "Average transverse energy (4 ADC count = 1 GeV) of real L1 trigger primitives. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
                  [{ 'path': "EcalBarrel/EBTriggerTowerTask/EBTTT Et spectrum Real Digis", 'description': "Average transverse energy (4 ADC count = 1 GeV) of real L1 trigger primitives. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
                   [{ 'path': "EcalEndcap/EETriggerTowerTask/EETTT Et spectrum Real Digis EE +", 'description': "Average transverse energy (4 ADC count = 1 GeV) of real L1 trigger primitives. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }]
                  )

l1toverviewlayout(dqmitems,"01 ECAL TP Occupancy",
                  [None,
                   { 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE -", 'description': "Map of the occupancy of ECAL trigger primitives with an E_T > 1 GeV (4 ADC counts). Darker regions mean noisy towers. Physics events only. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
                   None],
                  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy", 'description': "Map of the occupancy of ECAL trigger primitives with an E_T > 1 GeV (4 ADC counts). Darker regions mean noisy towers. Physics events only. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
                  [None,
                   { 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE +", 'description': "Map of the occupancy of ECAL trigger primitives with an E_T > 1 GeV (4 ADC counts). Darker regions mean noisy towers. Physics events only. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
                   None]
                  )

l1toverviewlayout(dqmitems,"02 ECAL TP Emulator Comparison",
                  [None,
                   { 'path': "EcalEndcap/EESummaryClient/EETTT EE - Trigger Primitives Timing summary", 'description': "Sample of the emulated TP that more often matches the real TP. Matched sample appear in non-red colors. Match with on-time primitives appear yellow (expected). No match at all appears red. No events appear white. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
                   None],
                  [{ 'path': "EcalBarrel/EBSummaryClient/EBTTT Trigger Primitives Timing summary", 'description': "Sample of the emulated TP that more often matches the real TP. Matched sample appear in non-red colors. Match with on-time primitives appear yellow (expected). No match at all appears red. No events appear white. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
                  [None,
                   { 'path': "EcalEndcap/EESummaryClient/EETTT EE + Trigger Primitives Timing summary", 'description': "Sample of the emulated TP that more often matches the real TP. Matched sample appear in non-red colors. Match with on-time primitives appear yellow (expected). No match at all appears red. No events appear white. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
                   None]
                  )

l1toverviewlayout(dqmitems,"03 Rate BSCL.BSCR",
                  [{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_040", 'description': "Rate BSCL.BSCR. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"04 Rate BSC splash right",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_042", 'description': "Rate BSC splash right. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"05 Rate BSC splash left",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_043", 'description': "Rate BSC splash left. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"06 Rate BSCOR and BPTX",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_124", 'description': "Rate BSCOR and BPTX. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"07 Rate Ratio 33 over 32",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_Ratio_33_over_32", 'description': "Ratio of Tech Bit 33 rate to Tech Bit 32 rate. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"08 Rate Ratio 41 over 40",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Rate_TechBit_Ratio_41_over_40", 'description': "Ratio of Tech Bit 41 rate to Tech Bit 40 rate. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"09 Integ BSCL*BSCR Triggers vs LS",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Integral_TechBit_040", 'description': "Integrated BSCL*BSCR Triggers vs LS. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"10 Integ BSCL or BSCR Triggers vs LS",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Integral_TechBit_042_OR_043", 'description': "Integrated BSCL or BSCR Triggers vs LS. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"11 Integ HF Triggers vs LS",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/TechnicalRates/Integral_TechBit_009", 'description': "Integrated HF Triggers vs LS. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

l1toverviewlayout(dqmitems,"12 Integ BSCOR and BPTX",
  	[{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Integral_AlgoBit_124", 'description': "Integrated BSCOR and BPTX. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

