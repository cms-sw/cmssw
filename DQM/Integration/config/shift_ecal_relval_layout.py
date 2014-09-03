def shiftecalvalidation(i, p, *rows): i["00 Shift/Ecal/" + p] = DQMItem(layout=rows)

shiftecalvalidation(dqmitems,'01 Rec Hit Spectra',
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit spectrum", 'description': "Energy of rec hits (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE +", 'description': "Energy of rec hits (EE+) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
  { 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE -", 'description': "Energy of rec hits (EE-) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

shiftecalvalidation(dqmitems,'02 Ecal Rech hit size ',
  [{ 'path': "EcalBarrel/EcalInfo/EBMM hit number", 'description': "Number of rec hits (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EcalInfo/EEMM hit number", 'description': "Number of rec hits (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }]
)

shiftecalvalidation(dqmitems,'03 Ecal timing',
  [{ 'path': 'EcalBarrel/EBSummaryClient/EBTMT timing mean 1D summary','description':""}],
  [{ 'path': 'EcalEndcap/EESummaryClient/EETMT EE - timing mean 1D summary', 'description': ""},
   { 'path': 'EcalEndcap/EESummaryClient/EETMT EE + timing mean 1D summary', 'description': ""}]
)
