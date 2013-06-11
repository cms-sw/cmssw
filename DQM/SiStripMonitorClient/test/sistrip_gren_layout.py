def sistriplayout(i, p, *rows): i["Layouts/SiStrip Layouts/" + p] = DQMItem(layout=rows)

sistriplayout(dqmitems, "SiStrip_NumberOfDigis_Summary",
  ["SiStrip/MechanicalView/TIB/Summary_NumberOfDigis_in_TIB"],
  ["SiStrip/MechanicalView/TOB/Summary_NumberOfDigis_in_TOB"],
  ["SiStrip/MechanicalView/TID/MINUS/Summary_NumberOfDigis_in_MINUS",
   "SiStrip/MechanicalView/TID/PLUS/Summary_NumberOfDigis_in_PLUS"],
  ["SiStrip/MechanicalView/TEC/MINUS/Summary_NumberOfDigis_in_MINUS",
   "SiStrip/MechanicalView/TEC/PLUS/Summary_NumberOfDigis_in_PLUS"])
sistriplayout(dqmitems, "SiStrip_NumberOfClusters_Summary",
  ["SiStrip/MechanicalView/TIB/Summary_NumberOfClusters_in_TIB"],
  ["SiStrip/MechanicalView/TOB/Summary_NumberOfClusters_in_TOB"],
  ["SiStrip/MechanicalView/TID/MINUS/Summary_NumberOfClusters_in_MINUS",
   "SiStrip/MechanicalView/TID/PLUS/Summary_NumberOfClusters_in_PLUS"],
  ["SiStrip/MechanicalView/TEC/MINUS/Summary_NumberOfClusters_in_MINUS",
   "SiStrip/MechanicalView/TEC/PLUS/Summary_NumberOfClusters_in_PLUS"])
sistriplayout(dqmitems, "SiStrip_ClusterWidth_Summary",
  ["SiStrip/MechanicalView/TIB/Summary_ClusterWidth_in_TIB"],
  ["SiStrip/MechanicalView/TOB/Summary_ClusterWidth_in_TOB"],
  ["SiStrip/MechanicalView/TID/MINUS/Summary_ClusterWidth_in_MINUS",
   "SiStrip/MechanicalView/TID/PLUS/Summary_ClusterWidth_in_PLUS"],
  ["SiStrip/MechanicalView/TEC/MINUS/Summary_ClusterWidth_in_MINUS",
   "SiStrip/MechanicalView/TEC/PLUS/Summary_ClusterWidth_in_PLUS"])
sistriplayout(dqmitems, "SiStrip_NumberOfDigis_Summary_TIB_Layer",
  ["SiStrip/MechanicalView/TIB/layer_1/Summary_NumberOfDigis_in_layer_1"],
  ["SiStrip/MechanicalView/TIB/layer_2/Summary_NumberOfDigis_in_layer_2"],
  ["SiStrip/MechanicalView/TIB/layer_3/Summary_NumberOfDigis_in_layer_3"],
  ["SiStrip/MechanicalView/TIB/layer_4/Summary_NumberOfDigis_in_layer_4"])
