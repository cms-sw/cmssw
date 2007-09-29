def dtlayout(i, p, *rows): i["Layouts/SiStrip Layouts/" + p] = DQMItem(layout=rows)

dtlayout(dqmitems, "SiStrip_Digi_Summary",
  ["SiStrip/MechanicalView/TIB/Summary_NumberOfDigis_in_TIB"],
  ["SiStrip/MechanicalView/TOB/Summary_NumberOfDigis_in_TOB"],
  ["SiStrip/MechanicalView/TID/side_0/Summary_NumberOfDigis_in_side_0",
   "SiStrip/MechanicalView/TID/side_1/Summary_NumberOfDigis_in_side_1"],
  ["SiStrip/MechanicalView/TEC/side_0/Summary_NumberOfDigis_in_side_0",
   "SiStrip/MechanicalView/TEC/side_1/Summary_NumberOfDigis_in_side_1"])
