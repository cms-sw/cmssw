def shifteblayout(i, p, *rows): i["00 Shift/EcalBarrel/" + p] = DQMItem(layout=rows)

shifteblayout(dqmitems, "00 Report Summary",
  [{ 'path': "EcalBarrel/EventInfo/reportSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "01 Event Type",
  [{ 'path': "EcalBarrel/EcalInfo/EVTTYPE", 'description': "Frequency of the event types found in the DQM event-stream. If the calibration sequence is ON, histograms should show entries in COSMICS_GLOBAL, LASER_GAP, PEDESTAL_GAP, TESTPULSE_GAP. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "02 Integrity Summary",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBIT integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBSummaryClient/EBIT PN integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "03 StatusFlags Summary",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBSFT front-end status summary", 'description': "DCC front-end status quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "04 Pedestal Online Quality",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBPOT pedestal quality summary G12", 'description': "pedestal quality summary. Pedestal is evaluated using the first 3/10 samples of the pulse shape for all the events (on physics events only). Expected all green color. Legend: green = good;  red = bad;  yellow = no entries. Quality criteria: 175 < mean < 225 ADCs, RMS < 2 ADCs. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

shifteblayout(dqmitems, "05 Timing Summary",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBTMT timing quality summary", 'description': "Quality summary of the good crystals reconstructed hits with energy > 1 GeV. Hardware timing is adjusted with readout tower granularity, but finer setting can be reached. Expected all green color. Legend: green = good;  red = bad;  yellow = no sufficient statistics. Quality criteria: Average timing in each TT for calibrated rechits with energy > 1 GeV, good DB status, rechit flag = kGood OR KOutOfTime, and |time| < 7 ns is evaluated if more than 36 hits pass the above cuts. The following criteria are applied: |mean| < 2 ns and RMS < 6 ns. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "06 Occupancy Rechits",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy", 'description': "Map of the occupancy of ECAL calibrated reconstructed hits. Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy projection eta", 'description': "Eta projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit occupancy projection phi", 'description': "Phi projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteblayout(dqmitems, "07 Occupancy Trigger Primitives",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy", 'description': "Map of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy projection eta", 'description': "Eta projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBOccupancyTask/EBOT TP digi thr occupancy projection phi", 'description': "Phi projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteblayout(dqmitems, "08 Clusters Energy",
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy map", 'description': "Average energy (in GeV) of hybrid basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy projection eta", 'description': "Eta projection of hybrid basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy projection phi", 'description': "Phi projection of hybrid basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteblayout(dqmitems, "09 Blue Laser (L1) Quality",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBLT laser quality summary L1", 'description': "Quality summary of blue laser events. Expect green where the laser sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBSummaryClient/EBLT PN laser quality summary L1", 'description': "Quality summary of blue laser events for PN diodes. Expect green where the laser sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalCalibration/Laser/EcalLaser L1 (blue) quality summary EB", 'description': "Quality summary of the blue laser light source for the last sequence."}])

shifteblayout(dqmitems, "11 Pedestal High Gain Quality",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBPT pedestal quality G12 summary", 'description': "Quality summary of pedestal events for Gain 12. Expect green where the pedestal sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBSummaryClient/EBPT PN pedestal quality G16 summary", 'description': "Quality summary of pedestal events for PN Gain 16. Expect green where the pedestal sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteblayout(dqmitems, "12 TestPulse High Gain Quality",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBTPT test pulse quality G12 summary", 'description': "Quality summary of test pulse events for Gain 12. Expect green where the test pulse sequence fired, yellow elsewhere. Red spots are failed channels. Supermodules are filled as the calibration sequence reach them: expected all yellow at beginning of run, then becoming green. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalBarrel/EBSummaryClient/EBTPT PN test pulse quality G16 summary", 'description': "Quality summary of test pulse events for PN Gain 16. Expect green where the test pulse sequence fired, yellow elsewhere. Red spots are failed channels. Supermodules are filled as the calibration sequence reach them: expected all yellow at beginning of run, then becoming green. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

