def dtlayout(i, p, *rows): i["Layouts/SiStrip Layouts/" + p] = DQMItem(layout=rows)

dtlayout(dqmitems, "SiStrip_Digi_Summary",
  ["SiStrip/MechanicalView/TIB/Summary_NumberOfDigis_in_TIB"],
  ["SiStrip/MechanicalView/TOB/Summary_NumberOfDigis_in_TOB"],
  ["SiStrip/MechanicalView/TID/MINUS/Summary_NumberOfDigis_in_MINUS",
   "SiStrip/MechanicalView/TID/PLUS/Summary_NumberOfDigis_in_PLUS"],
  ["SiStrip/MechanicalView/TEC/MINUS/Summary_NumberOfDigis_in_MINUS",
   "SiStrip/MechanicalView/TEC/PLUS/Summary_NumberOfDigis_in_PLUS"])
