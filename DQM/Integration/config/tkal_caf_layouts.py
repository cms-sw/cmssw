def tkallayout(i, p, *rows): i["Alignment/Tracker/Layouts/" + p] = DQMItem(layout=rows)


tkallayout(dqmitems,"00 - PIXEL absolute Residuals",
  ["Alignment/Tracker/Pixel/h_Xprime_TPBBarrel_1",
   "Alignment/Tracker/Pixel/h_Yprime_TPBBarrel_1"],
  ["Alignment/Tracker/Pixel/h_Xprime_TPEEndcap_2",
   "Alignment/Tracker/Pixel/h_Yprime_TPEEndcap_2"],
  ["Alignment/Tracker/Pixel/h_Xprime_TPEEndcap_3",
   "Alignment/Tracker/Pixel/h_Yprime_TPEEndcap_3"]
)

tkallayout(dqmitems,"01 - STRIP absolute Residuals",
  ["Alignment/Tracker/Strip/h_Xprime_TIBBarrel_1",
   "Alignment/Tracker/Strip/h_Xprime_TOBBarrel_4"],
  ["Alignment/Tracker/Strip/h_Xprime_TIDEndcap_2",
   "Alignment/Tracker/Strip/h_Xprime_TECEndcap_5"],
  ["Alignment/Tracker/Strip/h_Xprime_TIDEndcap_3",
   "Alignment/Tracker/Strip/h_Xprime_TECEndcap_6"]
)

tkallayout(dqmitems,"02 - PIXEL normalized Residuals",
  ["Alignment/Tracker/Pixel/h_NormXprime_TPBBarrel_1",
   "Alignment/Tracker/Pixel/h_NormYprime_TPBBarrel_1"],
  ["Alignment/Tracker/Pixel/h_NormXprime_TPEEndcap_2",
   "Alignment/Tracker/Pixel/h_NormYprime_TPEEndcap_2"],
  ["Alignment/Tracker/Pixel/h_NormXprime_TPEEndcap_3",
   "Alignment/Tracker/Pixel/h_NormYprime_TPEEndcap_3"]
)

tkallayout(dqmitems,"03 - STRIP normalized Residuals",
  ["Alignment/Tracker/Strip/h_NormXprime_TIBBarrel_1",
   "Alignment/Tracker/Strip/h_NormXprime_TOBBarrel_4"],
  ["Alignment/Tracker/Strip/h_NormXprime_TIDEndcap_2",
   "Alignment/Tracker/Strip/h_NormXprime_TECEndcap_5"],
  ["Alignment/Tracker/Strip/h_NormXprime_TIDEndcap_3",
   "Alignment/Tracker/Strip/h_NormXprime_TECEndcap_6"]
)
tkallayout(dqmitems,"10 - PIXEL DMRs",
  ["Alignment/Tracker/Pixel/h_DmrXprime_TPBBarrel_1",
   "Alignment/Tracker/Pixel/h_DmrYprime_TPBBarrel_1"],
  ["Alignment/Tracker/Pixel/h_DmrXprime_TPEEndcap_2",
   "Alignment/Tracker/Pixel/h_DmrYprime_TPEEndcap_2"],
  ["Alignment/Tracker/Pixel/h_DmrXprime_TPEEndcap_3",
   "Alignment/Tracker/Pixel/h_DmrYprime_TPEEndcap_3"]
)

tkallayout(dqmitems,"11 - STRIP DMRs",
  ["Alignment/Tracker/Strip/h_DmrXprime_TIBBarrel_1",
   "Alignment/Tracker/Strip/h_DmrXprime_TOBBarrel_4"],
  ["Alignment/Tracker/Strip/h_DmrXprime_TIDEndcap_2",
   "Alignment/Tracker/Strip/h_DmrXprime_TECEndcap_5"],
  ["Alignment/Tracker/Strip/h_DmrXprime_TIDEndcap_3",
   "Alignment/Tracker/Strip/h_DmrXprime_TECEndcap_6"]
)
