def ecaloverviewlayout(i, p, *rows): i["Collisions/EcalFeedBack/" + p] = DQMItem(layout=rows)

ecaloverviewlayout(dqmitems, "00 Single Event Timing EE",
  [{ 'path': "EcalEndcap/EETimingTask/EETMT timing 1D summary EE -", 'description': "Single event timing (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EETimingTask/EETMT timing 1D summary EE +", 'description': "Single event timing (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
  [{ 'path': "EcalEndcap/EETimingTask/EETMT timing EE+ - EE-", 'description': "Event by event difference between the average timing in EE+ and EE- (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected 0. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EETimingTask/EETMT timing EE+ vs EE-", 'description': "Average timing in EE- vs average timing in EE+. Only rechits with ET>600 MeV and kGood or kOutOfTime considered here. Expect one spot centered in (0,0) for collisions, two spots in (0,-20), (-20,0) for beam-halos. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "01 Timing Mean EE",
  [{ 'path': "EcalEndcap/EESummaryClient/EETMT EE - timing mean 1D summary", 'description': "Mean timing of all the channels in EE -. Timing is expected within 5.5 - 6.5 clocks. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EESummaryClient/EETMT EE + timing mean 1D summary", 'description': "Mean timing of all the channels in EE +. Timing is expected within 5.5 - 6.5 clocks. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
  [{ 'path': "EcalEndcap/EESummaryClient/EETMT timing mean", 'description': "Mean timing of all the channels in each DCC of EE. Timing is expected within 5.5 - 6.5 clocks. The error bar represents the spreads among the crystal of each DCC. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "02 Timing Map EE -",
  [{ 'path': "EcalEndcap/EETimingTask/EETMT timing map EE -", 'description': "Average timing (in clock units) of the seeds of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
  [{ 'path': "EcalEndcap/EETimingTask/EETMT timing projection eta EE -", 'description': "Average timing (in clock units) of the seeds of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EETimingTask/EETMT timing projection phi EE -", 'description': "Average timing (in clock units) of the seeds of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "02 Timing Map EE +",
  [{ 'path': "EcalEndcap/EETimingTask/EETMT timing map EE +", 'description': "Average timing (in clock units) of the seeds of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
  [{ 'path': "EcalEndcap/EETimingTask/EETMT timing projection eta EE +", 'description': "Average timing (in clock units) of the seeds of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EETimingTask/EETMT timing projection phi EE +", 'description': "Average timing (in clock units) of the seeds of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "03 Occupancy EE -",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE -", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection eta", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection phi", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "03 Occupancy EE +",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE +", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection eta", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection phi", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "04 Single Event Timing EB",
  [{ 'path': "EcalBarrel/EBTimingTask/EBTMT timing 1D summary", 'description': "Single event timing (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "05 Timing Mean EB",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBTMT timing mean 1D summary", 'description': "Mean timing of all the channels in EB along the run. Timing is expected within 5.5 - 6.5 clocks. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
  [{ 'path': "EcalBarrel/EBSummaryClient/EBTMT timing mean", 'description': "Mean timing of all the channels in each DCC of EB along the run. Timing is expected within 5.5 - 6.5 clocks. The error bar represents the spreads among the crystal of each DCC. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "06 Timing Map EB",
  [{ 'path': "EcalBarrel/EBTimingTask/EBTMT timing map", 'description': "Average timing (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }],
  [{ 'path': "EcalBarrel/EBTimingTask/EBTMT timing projection eta", 'description': "Average timing (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBTimingTask/EBTMT timing projection phi", 'description': "Average timing (in clock units) of the good rechits (good shape and amplitude > 500 MeV). Expected about 5.5 clocks. Readout tower binning (5x5 crystals) is used. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "07 Occupancy EB",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy projection eta", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } },
   { 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy projection phi", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>", 'draw': { 'withref': "yes" } }])

ecaloverviewlayout(dqmitems, "08 ES Occupancy",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z 1 P 1", 'description': "ES Occupancy with selected hits Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z -1 P 1", 'description': "ES Occupancy with selected hits Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z 1 P 2", 'description': "ES Occupancy with selected hits Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z -1 P 2", 'description': "ES Occupancy with selected hits Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecaloverviewlayout(dqmitems, "09 ES Energy Map",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z 1 P 1", 'description': "ES Energy Density with selected hits Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> "},
   { 'path': "EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z -1 P 1", 'description': "ES Energy Density with selected hits Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a>" }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z 1 P 2", 'description': "ES Energy Density with selected hits Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> "},
   { 'path': "EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z -1 P 2", 'description': "ES Energy Density with selected hits Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a>" }])

ecaloverviewlayout(dqmitems, "10 ES Timing Plot",
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 1", 'description': "ES Timing Z 1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 1", 'description': "ES Timing Z -1 P 1 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }],
  [{ 'path': "EcalPreshower/ESTimingTask/ES Timing Z 1 P 2", 'description': "ES Timing Z 1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " },
   { 'path': "EcalPreshower/ESTimingTask/ES Timing Z -1 P 2", 'description': "ES Timing Z -1 P 2 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftPreshower>DQMShiftPreshower</a> " }])

ecaloverviewlayout(dqmitems, "11 Ecal Z Mass",
  [{ 'path': "EcalCalibration/Zmass/Gaussian mean WP80 EB-EB", 'description': "Z mass formed by EB-EB electron combination" }],
  [{ 'path': "EcalCalibration/Zmass/Gaussian mean WP80 EB-EE", 'description': "Z mass formed by EB-EE electron combination" }],
  [{ 'path': "EcalCalibration/Zmass/Gaussian mean WP80 EE-EE", 'description': "Z mass formed by EB-EE electron combination" }])
