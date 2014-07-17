def ecallayout(i, p, *rows): i["Ecal/Layouts/" + p] = DQMItem(layout=rows)

ecallayout(dqmitems, "00-Global-Summary",
  [None,
   { 'path': "EcalEndcap/EESummaryClient/EE global summary EE +", 'description': "EcalEndcap (z>0) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" },
   None],
  [{ 'path': "EcalBarrel/EBSummaryClient/EB global summary", 'description': "EcalBarrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" }],
  [None,
   { 'path': "EcalEndcap/EESummaryClient/EE global summary EE -", 'description': "EcalEndcap (z<0) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" },
   None])

ecallayout(dqmitems, "01-Occupancy-Summary",
  [None,
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE +", 'description': "EcalEndcap (z>0) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" },
   None],
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy", 'description': "EcalBarrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" }],
  [None,
   { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE -", 'description': "EcalEndcap (z<0) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" },
   None])

ecallayout(dqmitems, "02-Cluster-Summary",
  [None,
   { 'path': "EcalEndcap/EEClusterTask/EECLT BC energy map EE +", 'description': "EcalEndcap (z>0) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" },
   None],
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT BC energy map", 'description': "EcalBarrel - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" }],
  [None,
   { 'path': "EcalEndcap/EEClusterTask/EECLT BC energy map EE -", 'description': "EcalEndcap (z<0) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/EcalDQM>EcalDQM</a>" },
   None])

