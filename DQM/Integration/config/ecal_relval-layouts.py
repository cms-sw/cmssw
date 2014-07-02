def ecalvalidationlayout(i, p, *rows): i["EcalBarrel/Layouts/" + p] = DQMItem(layout=rows)

ecalvalidationlayout(dqmitems, "00 Ecal RecHit size",
  [{ 'path': "EcalBarrel/EcalInfo/EBMM hit number", 'description': "Number of rec hits (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EcalInfo/EEMM hit number", 'description': "Number of rec hits (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "01 ES RecHit size (Z -1)",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Num of RecHits Z -1 P 1", 'description': "Number of rec hits (ES -1 P1) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Num of RecHits Z -1 P 2", 'description': "Number of rec hits (ES -1 P2) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "02 ES RecHit size (Z +1)",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Num of RecHits Z 1 P 1", 'description': "Number of rec hits (ES +1 P1) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES Num of RecHits Z 1 P 2", 'description': "Number of rec hits (ES +1 P2) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "03 EB RecHit spectra",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit spectrum", 'description': "Energy of rec hits (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "04 EE RecHit spectra",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE +", 'description': "Energy of rec hits (EE+) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit spectrum EE -", 'description': "Energy of rec hits (EE-) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "05 ES RecHit spectra (Z -1)",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1", 'description': "Energy of rec hits (ES -1 P1) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2", 'description': "Energy of rec hits (ES -1 P2) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "06 ES RecHit spectra (Z +1)",
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1", 'description': "Energy of rec hits (ES +1 P1) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2", 'description': "Energy of rec hits (ES +1 P2) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "07 EB RecHit max energy",
  [{ 'path': "EcalBarrel/EBRecoSummary/recHits_EB_energyMax", 'description': "Reconstructed hits max energy in the barrel. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "08 EE RecHit max energy",
  [{ 'path': "EcalEndcap/EERecoSummary/recHits_EEP_energyMax", 'description': "Reconstructed hits max energy in the endcaps. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/recHits_EEM_energyMax", 'description': "Reconstructed hits max energy in the endcaps. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "09 Preshower max energy",
  [{ 'path': "EcalPreshower/ESRecoSummary/recHits_ES_energyMax", 'description': "Preshower rechits max energy. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "10 EB RecHit eta",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy projection eta", 'description': "Rec hits eta(barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "11 EE RecHit eta",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection eta", 'description': "Rec hits eta(EE-) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection eta", 'description': "Rec hits eta(EE+) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "12 EB RecHit phi",
  [{ 'path': "EcalBarrel/EBOccupancyTask/EBOT rec hit thr occupancy projection phi", 'description': "Rec hits phi(barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "13 EE RecHit phi",
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE - projection phi", 'description': "Rec hits phi(EE-) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEOccupancyTask/EEOT rec hit thr occupancy EE + projection phi", 'description': "Rec hits phi(EE+) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

#Timing : missing in the offline
ecalvalidationlayout(dqmitems, "14 EB RecHit time",
  [{ 'path': "EcalBarrel/EBSummaryClient/EBTMT timing mean 1D summary", 'description': "Rec hits time(barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEc#alExpert</a>" }])

ecalvalidationlayout(dqmitems, "15 EE RecHit time",
  [{ 'path': "EcalEndcap/EESummaryClient/EETMT EE - timing mean 1D summary", 'description': "Rec hits time(EE-) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEc#alExpert</a>" }],
  [{ 'path': "EcalEndcap/EESummaryClient/EETMT EE + timing mean 1D summary", 'description': "Rec hits time(EE+) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEc#alExpert</a>" }])

ecalvalidationlayout(dqmitems, "16 Preshower timing",
  [{ 'path': "EcalPreshower/ESRecoSummary/recHits_ES_time", 'description': "Preshower timing. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "17 EB RecHit chi2",
  [{ 'path': "EcalBarrel/EBRecoSummary/recHits_EB_Chi2", 'description': "Reconstructed hits shape chi2. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "18 EE RecHit chi2",
  [{ 'path': "EcalEndcap/EERecoSummary/recHits_EEP_Chi2", 'description': "Reconstructed hits shape chi2. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/recHits_EEM_Chi2", 'description': "Reconstructed hits shape chi2. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "19 EB RecHit E1/E4",
  [{ 'path': "EcalBarrel/EBRecoSummary/recHits_EB_E1oE4", 'description': "Reconstructed hits E1/E4. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "20 RecHitsFlags",
  [{ 'path': "EcalBarrel/EBRecoSummary/recHits_EB_recoFlag", 'description': "Reconstructed hits flags. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/recHits_EE_recoFlag", 'description': "Reconstructed hits flags. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "21 reduced RecHitsFlags",
  [{ 'path': "EcalBarrel/EBRecoSummary/redRecHits_EB_recoFlag", 'description': "Reconstructed hits flags (reduced collection). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/redRecHits_EE_recoFlag", 'description': "Reconstructed hits flags (reduced collection). <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "22 Basic Clusters Hits Flags",
  [{ 'path': "EcalBarrel/EBRecoSummary/basicClusters_recHits_EB_recoFlag", 'description': "Flags of hits associated to basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/basicClusters_recHits_EE_recoFlag", 'description': "Flags of hits associated to basic clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])


ecalvalidationlayout(dqmitems, "23 Number of basic clusters",
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT BC number", 'description': "Number of Basic Clusters (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT BC number", 'description': "Number of Basic Clusters (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "24 Number of super clusters",
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT SC number", 'description': "Number of Super Clusters (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT SC number", 'description': "Number of Super Clusters (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "25 Super Cluster energy",
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT SC energy", 'description': "Energy of Super Clusters (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT SC energy", 'description': "Energy of Super Clusters (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "26 Super Clusters Eta",
  [{ 'path': "EcalBarrel/EBRecoSummary/superClusters_eta", 'description': "Super Clusters eta (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/superClusters_eta", 'description': "Super Clusters eta (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "27 Super Clusters Phi",
  [{ 'path': "EcalBarrel/EBRecoSummary/superClusters_EB_phi", 'description': "Super Clusters phi (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/superClusters_EE_phi", 'description': "Super Clusters phi (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "28 Number of crystals per super cluster",
  [{ 'path': "EcalBarrel/EBClusterTask/EBCLT SC size (crystal)", 'description': "Number of crystals per SC (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EEClusterTask/EECLT SC size (crystal)", 'description': "Number of crystals per SC (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "29 EB Super Clusters nBC",
  [{ 'path': "EcalBarrel/EBRecoSummary/superClusters_EB_nBC", 'description': "Number of basic clusters in Super Clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "30 EE Super Clusters nBC",
  [{ 'path': "EcalEndcap/EERecoSummary/superClusters_EEP_nBC", 'description': "Number of basic clusters in Super Clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/superClusters_EEM_nBC", 'description': "Number of basic clusters in Super Clusters. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "31 Super cluster seed swiss cross",
  [{ 'path': "EcalBarrel/EBRecoSummary/superClusters_EB_E1oE4", 'description': "SC seed swiss cross (barrel) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
  [{ 'path': "EcalEndcap/EERecoSummary/superClusters_EE_E1oE4", 'description': "SC seed swiss cross (endcaps) <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])

ecalvalidationlayout(dqmitems, "32 Preshower planes energy",
  [{ 'path': "EcalPreshower/ESRecoSummary/esClusters_energy_plane1", 'description': "Preshower rechits energy on plane 1. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" },
   { 'path': "EcalPreshower/ESRecoSummary/esClusters_energy_plane2", 'description': "Preshower rechits energy on plane 2. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }],
   [{ 'path': "EcalPreshower/ESRecoSummary/esClusters_energy_ratio", 'description': "Preshower rechits energy on plane1/plane2. <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftEcalExpert>DQMShiftEcalExpert</a>" }])
