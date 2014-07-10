def shifteelayout(i, p, *rows): i["00 Shift/EcalEndcap/" + p] = DQMItem(layout=rows)

shifteelayout(dqmitems, "00 Report Summary",
  [{ 'path': "EcalEndcap/EventInfo/reportSummaryMap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "01 Event Type",
  [{ 'path': "EcalEndcap/EcalInfo/EVTTYPE", 'description': "Frequency of the event types found in the DQM event-stream. If the calibration sequence is ON, histograms should show entries in COSMICS_GLOBAL, LASER_GAP, PEDESTAL_GAP, TESTPULSE_GAP. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "02 Integrity Summary",
  [{ 'path': "EcalEndcap/EESummaryClient/EEIT EE - integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EEIT EE + integrity quality summary", 'description': "Integrity quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "03 StatusFlags Summary",
  [{ 'path': "EcalEndcap/EESummaryClient/EESFT EE - front-end status summary", 'description': "DCC front-end status quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EESFT EE + front-end status summary", 'description': "DCC front-end status quality summary. Expected all green color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "04 Pedestal Online Quality",
  [{ 'path': "EcalEndcap/EESummaryClient/EEPOT EE - pedestal quality summary G12", 'description': "Pedestal quality summary. Pedestal is evaluated using the first 3/10 samples of the pulse shape for all the events (on physics events only). Expected all green color. Legend: green = good;  red = bad;  yellow = no entries. Quality criteria: 175 < mean < 225 ADCs, RMS < 4 ADCs <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EEPOT EE + pedestal quality summary G12", 'description': "pedestal quality summary. Pedestal is evaluated using the first 3/10 samples of the pulse shape for all the events (on physics events only). Expected all green color. Legend: green = good;  red = bad;  yellow = no entries. Quality criteria: 175 < mean < 225 ADCs, RMS < 4 ADCs. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

shifteelayout(dqmitems, "05 Timing Summary",
  [{ 'path': "EcalEndcap/EESummaryClient/EETMT EE - timing quality summary", 'description': "Quality summary of the crystal reconstructed hits in EE - with energy > 3(6) GeV (|eta| <(>) 2.4). Hardware timing is adjusted with readout tower granularity, but finer setting can be reached. Expected all green color. Legend: green = good;  red = bad;  yellow = no sufficient statistics. Quality criteria: Average timing in each supercrystal for calibrated rechits with energy > 3(6) GeV (|eta| <(>) 2.4), good DB status, rechit flag = kGood OR KOutOfTime, and |time| < 7 ns is evaluated if more than 60 hits pass the above cuts. The following criteria are applied: |mean| < 3 ns and RMS < 6 ns.  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EETMT EE + timing quality summary", 'description': "Quality summary of the crystal reconstructed hits in EE - with energy > 3(6) GeV (|eta| <(>) 2.4). Hardware timing is adjusted with readout tower granularity, but finer setting can be reached. Expected all green color. Legend: green = good;  red = bad;  yellow = no sufficient statistics. Quality criteria: Average timing in each supercrystal for calibrated rechits with energy > 3(6) GeV (|eta| <(>) 2.4), good DB status, rechit flag = kGood OR KOutOfTime, and |time| < 7 ns is evaluated if more than 60 hits pass the above cuts. The following criteria are applied: |mean| < 3 ns and RMS < 6 ns.  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "06 Occupancy Rechits EE -",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE -", 'description': "Map of the occupancy of ECAL calibrated reconstructed hits. Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection eta", 'description': "Eta projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE - projection phi", 'description': "Phi projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "06 Occupancy Rechits EE +",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE +", 'description': "Map of the occupancy of ECAL calibrated reconstructed hits. Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection eta", 'description': "Eta projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit occupancy EE + projection phi", 'description': "Phi projection of the occupancy of ECAL calibrated reconstructed hits. Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "07 Occupancy Trigger Primitives EE -",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE -", 'description': "Map of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection eta", 'description': "Eta projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE - projection phi", 'description': "Phi projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "07 Occupancy Trigger Primitives EE +",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE +", 'description': "Map of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform color. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection eta", 'description': "Eta projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT TP digi thr occupancy EE + projection phi", 'description': "Phi projection of the occupancy of ECAL trigger primitives with energy > 4 ADC counts (~2 GeV). Expect uniform distribution. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "08 Clusters Energy EE -",
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy map EE -", 'description': "Average energy (in GeV) of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection eta EE -", 'description': "Eta projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection phi EE -", 'description': "phi projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "08 Clusters Energy EE +",
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy map EE +", 'description': "Average energy (in GeV) of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection eta EE +", 'description': "Eta projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEClusterTask/EECLT BC energy projection phi EE +", 'description': "phi projection of 5x5 basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>", 'draw': { 'withref': "yes" } }])

shifteelayout(dqmitems, "09 Blue Laser (L1) Quality",
  [{ 'path': "EcalEndcap/EESummaryClient/EELT EE - laser quality summary L1", 'description': "Quality summary of blue laser events. Expect green where the laser sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EELT EE + laser quality summary L1", 'description': "Quality summary of blue laser events. Expect green where the laser sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EESummaryClient/EELT PN laser quality summary L1", 'description': "Quality summary of blue laser events on PN diodes. Expect green where the laser sequence fired, yellow or white elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   {'path': "EcalCalibration/Laser/EcalLaser L1 (blue) quality summary EE", 'description': "Quality summary of the blue laser light source for the last sequence."}])

shifteelayout(dqmitems, "10 Lambda 1 Led Quality", 
  [{ 'path': "EcalEndcap/EESummaryClient/EELDT EE - led quality summary L1", 'description': "Quality summary of lambda_1 led events. Expect green where the laser sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }, 
   { 'path': "EcalEndcap/EESummaryClient/EELDT EE + led quality summary L1", 'description': "Quality summary of lambda_1 led events. Expect green where the laser sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EESummaryClient/EELDT PN led quality summary L1", 'description': "Quality summary of lambda_1 led events on PN diodes. Expect green where the laser sequence fired, yellow or white elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "11 Pedestal Quality",
  [{ 'path': "EcalEndcap/EESummaryClient/EEPT EE - pedestal quality G12 summary", 'description': "Quality summary of pedestal events for Gain 12. Expect green where the pedestal sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EEPT EE + pedestal quality G12 summary", 'description': "Quality summary of pedestal events for Gain 12. Expect green where the pedestal sequence fired, yellow elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EESummaryClient/EEPT PN pedestal quality G16 summary", 'description': "Quality summary of pedestal events for PN Gain 16. Expect green where the pedestal sequence fired, yellow or white elsewhere. Red spots are failed channels. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

shifteelayout(dqmitems, "12 TestPulse Quality",
  [{ 'path': "EcalEndcap/EESummaryClient/EETPT EE - test pulse quality G12 summary", 'description': "Quality summary of test pulse events for Gain 12. Expect green where the calibration sequence fired, yellow elsewhere. Red spots are failed channels. Sectors are filled as the calibration sequence reach them: expected all yellow at beginning of run, then becoming green sector by sector. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" },
   { 'path': "EcalEndcap/EESummaryClient/EETPT EE + test pulse quality G12 summary", 'description': "Quality summary of test pulse events for Gain 12. Expect green where the calibration sequence fired, yellow elsewhere. Red spots are failed channels. Sectors are filled as the calibration sequence reach them: expected all yellow at beginning of run, then becoming green sector by sector. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }],
  [{ 'path': "EcalEndcap/EESummaryClient/EETPT PN test pulse quality G16 summary", 'description': "Quality summary of test pulse events for PN Gain 16. Expect green where the calibration sequence fired, yellow or white elsewhere. Red spots are failed channels. Sectors are filled as the calibration sequence reach them: expected all yellow at beginning of run, then becoming green sector by sector. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcal>DQMShiftEcal</a>" }])

