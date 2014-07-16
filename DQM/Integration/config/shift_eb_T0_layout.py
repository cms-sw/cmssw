def shifteblayout(i, p, *rows): i["00 Shift/EcalBarrel/" + p] = DQMItem(layout=rows)

shifteblayout(dqmitems, "00 Report Summary",
  [{ 'path': "EcalBarrel/EventInfo/reportSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }])

shifteblayout(dqmitems, "01 Integrity Summary",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBIT integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }])

shifteblayout(dqmitems, "02 StatusFlags Summary",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBSFT front-end status summary", 'description': "DCC front-end status quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "03 PedestalOnline RMS",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBPOT pedestal G12 RMS map", 'description': "RMS of the pedestals in ADC counts. Pedestal is evaluated using the first 3/10 samples of the pulse shape for all the events (calibration and physics). Expected RMS for ECAL barrel is 1.1 ADC counts (43 MeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }])

shifteblayout(dqmitems, "04 Occupancy Rechits",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy", 'description': "Map of the occupancy of ECAL calibrated reconstructed hits. Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy projection eta", 'description': "Eta projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy projection phi", 'description': "Phi projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteblayout(dqmitems, "05 Occupancy Trigger Primitives",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy", 'description': "Map of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy projection eta", 'description': "Eta projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy projection phi", 'description': "Phi projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

shifteblayout(dqmitems, "06 Clusters Energy",
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy map", 'description': "Average energy (in GeV) of hybrid basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>" }],
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy projection eta", 'description': "Eta projection of hybrid basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy projection phi", 'description': "Phi projection of hybrid basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineEcal>DQMShiftOfflineEcal</a>", 'draw': { 'withref': "yes" } }])

