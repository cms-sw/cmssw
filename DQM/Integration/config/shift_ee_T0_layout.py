def shifteelayout(i, p, *rows): i["00 Shift/EcalEndcap/" + p] = DQMItem(layout=rows)

shifteelayout(dqmitems, "00 Report Summary",
  [{ 'path': "EcalEndcap/EventInfo/reportSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }])

shifteelayout(dqmitems, "01 Integrity Summary",
  [{ 'path': "EcalEndcap/EESummaryClient/EEIT EE - integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EEIT EE + integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }])

shifteelayout(dqmitems, "02 StatusFlags Summary",
  [{ 'path': "EcalEndcap/EESummaryClient/EESFT EE - front-end status summary", 'description': "DCC front-end status quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EESFT EE + front-end status summary", 'description': "DCC front-end status quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "03 PedestalOnline RMS",
  [{ 'path': "EcalEndcap/EESummaryClient/EEPOT EE - pedestal G12 RMS map", 'description': "RMS of the pedestals in ADC counts. Pedestal is evaluated using the first 3/10 samples of the pulse shape for all the events (calibration and physics). Expected RMS for ECAL endcap is 1.9 ADC counts (120 MeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EEPOT EE + pedestal G12 RMS map", 'description': "RMS of the pedestals in ADC counts. Pedestal is evaluated using the first 3/10 samples of the pulse shape for all the events (calibration and physics). Expected RMS for ECAL endcap is 1.9 ADC counts (120 MeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }])

shifteelayout(dqmitems, "04 Occupancy Rechits EE -",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE -", 'description': "Map of the occupancy of ECAL calibrated reconstructed hits. Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection eta", 'description': "Eta projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection phi", 'description': "Phi projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "04 Occupancy Rechits EE +",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE +", 'description': "Map of the occupancy of ECAL calibrated reconstructed hits. Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection eta", 'description': "Eta projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection phi", 'description': "Phi projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "05 Occupancy Trigger Primitives EE -",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE -", 'description': "Map of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection eta", 'description': "Eta projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection phi", 'description': "Phi projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "05 Occupancy Trigger Primitives EE +",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE +", 'description': "Map of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection eta", 'description': "Eta projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection phi", 'description': "Phi projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "06 Clusters Energy EE -",
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy map EE -", 'description': "Average energy (in GeV) of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection eta EE -", 'description': "Eta projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection phi EE -", 'description': "phi projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "06 Clusters Energy EE +",
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy map EE +", 'description': "Average energy (in GeV) of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection eta EE +", 'description': "eta projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection phi EE +", 'description': "phi projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

