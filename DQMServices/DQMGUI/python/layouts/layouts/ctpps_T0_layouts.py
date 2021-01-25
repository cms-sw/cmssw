from .adapt_to_new_backend import *
dqmitems={}

####################################################################################################
# Strip layouts
####################################################################################################

def CTPPSTrackingStripLayoutRP(i, p, *rows): i["CTPPS/TrackingStrip/Layouts/RP summary/" + p] = rows
def CTPPSTrackingStripLayoutDiag(i, p, *rows): i["CTPPS/TrackingStrip/Layouts/diagonal summary/" + p] = rows
def CTPPSTrackingStripLayoutAntiDiag(i, p, *rows): i["CTPPS/TrackingStrip/Layouts/antidiagonal summary/" + p] = rows
strip_rows = [ "tp", "bt" ]
strip_cols = [ "sector 45/station 220/fr", "sector 45/station 210/fr", "sector 56/station 210/fr", "sector 56/station 220/fr" ]

# layouts with no overlays
for plot in [ "active planes", "activity in planes (2D)", "vfats with any problem", "track XY profile" ]:
  rows = list()
  for rp in strip_rows:
    row = list()
    for unit in strip_cols:
      row.append("CTPPS/TrackingStrip/"+unit+"_"+rp+"/"+plot)
    rows.append(row)
  CTPPSTrackingStripLayoutRP(dqmitems, plot, *rows)

# layouts with overlays
for plot in [ "active planes", "recognized patterns", "planes contributing to fit" ]:
  rows = list()
  for rp in strip_rows:
    row = list()
    for unit in strip_cols:
      hist_u = "CTPPS/TrackingStrip/"+unit+"_"+rp+"/"+plot + " U"
      hist_v = "CTPPS/TrackingStrip/"+unit+"_"+rp+"/"+plot + " V"
      row.append( { "path" : hist_u, "overlays" : [ hist_v ] } )
    rows.append(row)
  CTPPSTrackingStripLayoutRP(dqmitems, plot + " UV", *rows)

# layouts with overlays
for suffix in [ "" ]:
  rows = list()
  for rp in strip_rows:
    row = list()
    for unit in strip_cols:
      hist_rp = "CTPPS/TrackingStrip/"+unit+"_"+rp+"/activity per BX" + suffix
      hist_base = "CTPPS/common/events per BX" + suffix
      row.append( { "path" : hist_rp, "overlays" : [ hist_base ] } )
    rows.append(row)
  CTPPSTrackingStripLayoutRP(dqmitems, "activity per BX" + suffix, *rows)

strip_diagonals = [ "diagonal 45bot - 56top", "diagonal 45top - 56bot" ]
strip_antidiagonals = [ "antidiagonal 45bot - 56bot", "antidiagonal 45top - 56top" ]
strip_diagonal_cols = [ "45_220_fr", "45_210_fr", "56_210_fr", "56_220_fr" ]
strip_diagonal_rows = [ "tp", "bt" ]

# rate plots
for plot in [ "rate - 2 RPs (220-fr)", "rate - 4 RPs" ]:
  rows = list()
  for diag in strip_diagonals:
    row = list()
    row.append("CTPPS/TrackingStrip/"+diag+"/"+plot)
    rows.append(row)
  CTPPSTrackingStripLayoutDiag(dqmitems, plot, *rows)

  rows = list()
  for adiag in strip_antidiagonals:
    row = list()
    row.append("CTPPS/TrackingStrip/"+adiag+"/"+plot)
    rows.append(row)
  CTPPSTrackingStripLayoutAntiDiag(dqmitems, plot, *rows)

# xy plots with conditions
for cond in [ "2 RPs cond", "4 RPs cond" ]:
  rows = list()
  for rp in strip_diagonal_rows:
    row = list()
    for unit in strip_diagonal_cols:
      arm=unit[:2]
      if ((arm == "45") and (rp == "bt")): diag = "diagonal 45bot - 56top"
      if ((arm == "56") and (rp == "tp")): diag = "diagonal 45bot - 56top"
      if ((arm == "45") and (rp == "tp")): diag = "diagonal 45top - 56bot"
      if ((arm == "56") and (rp == "bt")): diag = "diagonal 45top - 56bot"
      row.append("CTPPS/TrackingStrip/"+diag+"/xy hists/xy hist - "+unit+"_"+rp+" - "+cond)
    rows.append(row)
  CTPPSTrackingStripLayoutDiag(dqmitems, "XY profile - "+cond, *rows)

  rows = list()
  for rp in strip_diagonal_rows:
    row = list()
    for unit in strip_diagonal_cols:
      arm=unit[:2]
      if ((arm == "45") and (rp == "bt")): diag = "antidiagonal 45bot - 56bot"
      if ((arm == "45") and (rp == "tp")): diag = "antidiagonal 45top - 56top"
      if ((arm == "56") and (rp == "tp")): diag = "antidiagonal 45top - 56top"
      if ((arm == "56") and (rp == "bt")): diag = "antidiagonal 45bot - 56bot"
      row.append("CTPPS/TrackingStrip/"+diag+"/xy hists/xy hist - "+unit+"_"+rp+" - "+cond)
    rows.append(row)
  CTPPSTrackingStripLayoutAntiDiag(dqmitems, "XY profile - "+cond, *rows)



####################################################################################################
# Diamond layouts
####################################################################################################

diamond_stations = [ "sector 45/station 220cyl/cyl_hr", "sector 56/station 220cyl/cyl_hr" ]

def CTPPSTimingDiamondLayout(i, p, *rows): i["CTPPS/TimingDiamond/Layouts/" + p] = rows

# layouts with no overlays
TimingPlots = [ "activity per BX 0 25", "active planes", "event category", "leading edge (le and te)", "time over threshold", "hits in planes", "hits in planes lumisection", "tracks", "HPTDC Errors", "MH in channels" ]
TimingDrawOpt = [{'ytype':"log"}, {'xmax':"10"}, {'drawopts':"colztext"}, {'xmax':"25"}, {'xmin':"0", 'xmax':"25"}, {'withref':"no"}, {'withref':"no"}, {'withref':"no"}, {'drawopts':"colztext"}, {'drawopts':"colztext"}]
TimingDescription = [ "It should be similar to activity of CMS", "It should be with peaks at 0 and 4", 'Most of the event should be in "both"',
"It should be peaked around 5 ns", "It should be a broad distribution peaked around 12 ns",
"It should be full", 'It should be similar to "hits in planes"',
'It should be "uniform"', "It should be empty", "It should be as low as possible (<1%)" ]

for i in range(len(TimingPlots)):
  rows = list()
  for station in diamond_stations:
    row = list()
    path_str = "CTPPS/TimingDiamond/"+station+"/"+TimingPlots[i]
    row.append( { "path" : path_str, 'draw':TimingDrawOpt[i], 'description':TimingDescription[i] } )
    rows.append(row)

  CTPPSTimingDiamondLayout(dqmitems, TimingPlots[i], *rows)

# Efficiency display for all planes
rows = list()
for station in diamond_stations:
  row = list()
  for plane in range(4):
    hist_lead = "CTPPS/TimingDiamond/"+station+"/plane {plane_id}/Efficiency wrt pixels".format(plane_id=plane)
    row.append( { "path": hist_lead, 'draw':{'drawopts':"colz"}, 'description':"It should have most of the bins to 1 or 0 outside the sensor" } )
  rows.append(row)

  CTPPSTimingDiamondLayout(dqmitems, "efficiency", *rows)

# # clocks display for both sectors
# rows = list()
# for station in diamond_stations:
#   row = list()
#   for clk in [ 1, 3 ]:
#     hist_lead = "CTPPS/TimingDiamond/"+station+"/clock/clock{clock_id} leading edge".format(clock_id=clk)
#     row.append( { "path": hist_lead, 'draw':{'xmax':"25"} } )
#   rows.append(row)
#
#   CTPPSTimingDiamondLayout(dqmitems, "clocks edges", *rows)



####################################################################################################
# Totem Timing layouts
####################################################################################################

def TotemTimingLayout(i, p, *rows): i["CTPPS/TimingFastSilicon/Layouts/" + p] = rows
totemTiming_stations = [ "sector 45/station 220/nr_tp", "sector 45/station 220/nr_bt", "sector 56/station 220/nr_tp", "sector 56/station 220/nr_bt" ]

# layouts with no overlays
TotTimingPlots = [   "activity per BX CMS",
                     "active planes with time", \
                     "amplitude", \
                     "cell of max", \
                     "hits in planes with time", \
                     "hits in planes lumisection", \
                     "mean amplitude", \
                     "noise RMS", \
                     "recHit time", \
                     "tomography 220", \
                     "tomography 210" ]
TotTimingDrawOpt = [ {'ytype':"log"}, \
                     {'ytype':"log"}, \
                     {'withref':"no"}, \
                     {'drawopts':"colztext"}, \
                     {'drawopts':"colz"}, \
                     {'drawopts':"colz"}, \
                     {'drawopts':"colztext"}, \
                     {'drawopts':"colztext"}, \
                     {'withref':"no"}, \
                     {'drawopts':"colz"}, \
                     {'drawopts':"colz"} ]
TotTimingDescription = [ \
    "BX with activity in the detector: BX structure should be as CMS", \
    "Planes (per event) with time information", \
    "Amplitude of signals", \
    "Index of the sample with maximum amplitude: should be around 20-22", \
    "Hit map of channels with time information", \
    "Hit map in the last lumisection: should be very similar to the cumulative map", \
    "Mean amplitude of collected samples", \
    "Noise RMS in the first samples (without signal)", \
    "Time of arrival", \
    "Coincidence with strip detectors", \
    "Coincidence with strip detectors" ]

for i in range(len(TotTimingPlots)):
    rows = list()
    for station in totemTiming_stations:
      row = list()
      path_str = "CTPPS/TimingFastSilicon/"+station+"/"+TotTimingPlots[i]
      row.append( { "path" : path_str, "description" : TotTimingDescription[i], "draw":TotTimingDrawOpt[i] } )
      rows.append(row)

    TotemTimingLayout(dqmitems, TotTimingPlots[i], *rows)

TotemTimingLayout(dqmitems, "sent digis percentage",
          [{ "path" : "CTPPS/TimingFastSilicon/sent digis percentage", \
         "description" : "Percentage of hits sent by DAQ", \
         "draw": {"drawopts" : "colztext"} }])







# ####################################################################################################
# # Pixel layouts
# ####################################################################################################
sectors = [ "sector 45", "sector 56" ]
pixelstations = [ "station 210", "station 220" ]
pixstationsf=["sector 45/station 210/","sector 45/station 220/","sector 56/station 210/","sector 56/station 220/"]
pix_planes  = [ "0","1","2" ]
pix_planes2 = [ "3","4","5" ]

def CTPPSTrackingPixelLayout(i, p, *rows): i["CTPPS/TrackingPixel/Layouts/" + p] = rows

for plot in ["hit multiplicity in planes"]:
  rows = list()
  row = list()
  for station in pixelstations:
    row.append("CTPPS/TrackingPixel/sector 45/"+station+"/fr_hr/"+plot)
  rows.append(row)

  row = list()
  for station in pixelstations:
    row.append("CTPPS/TrackingPixel/sector 56/"+station+"/fr_hr/"+plot)
  rows.append(row)

  CTPPSTrackingPixelLayout(dqmitems, plot, *rows)

for plot in ["number of fired planes per event","ROCs hits multiplicity per event","track intercept point","number of tracks per event"]:
  rows = list()
  row = list()
  for station in pixelstations:
    row.append("CTPPS/TrackingPixel/sector 45/"+station+ "/fr_hr/"+plot)
  rows.append(row)

  row = list()
  for station in pixelstations:
    row.append("CTPPS/TrackingPixel/sector 56/"+station+"/fr_hr/"+plot)
  rows.append(row)

  CTPPSTrackingPixelLayout(dqmitems, plot, *rows)

for plot in ["4 fired ROCs per BX", "5 fired planes per BX"]:
  rows = list()
  for station in pixstationsf:
    row = list()
    row.append("CTPPS/TrackingPixel/"+station+"fr_hr/latency/"+plot)
    rows.append(row)

  CTPPSTrackingPixelLayout(dqmitems, plot, *rows)

for sector in sectors:
  rows = list()
  for station in pixelstations:
    row = list()
    row.append("CTPPS/TrackingPixel/"+sector+"/"+station+"/fr_hr/"+
        "ROCs_hits_multiplicity_per_event vs LS")
    rows.append(row)

  CTPPSTrackingPixelLayout(dqmitems, "ROC hits per event vs LS "+sector, *rows)

for plot in ["Pixel planes activity"]:
  rows = list()
  row = list()
  row.append("CTPPS/TrackingPixel/"+plot)
  rows.append(row)

  CTPPSTrackingPixelLayout(dqmitems, plot, *rows)

for plot in ["hits position"]:
  for sector in sectors:
    for station in pixelstations:
      rows = list()
      row = list()
      for plane in pix_planes:
        hit_pos = "CTPPS/TrackingPixel/"+sector+"/"+station+"/fr_hr/plane_"+plane+"/"+plot
        row.append( { "path": hit_pos, 'draw':{'ztype':"log"} } )
      rows.append(row)

      row = list()
      for plane in pix_planes2:
        hit_pos = "CTPPS/TrackingPixel/"+sector+"/"+station+"/fr_hr/plane_"+plane+"/"+plot
        row.append( { "path": hit_pos, 'draw':{'ztype':"log"} } )
      rows.append(row)

      CTPPSTrackingPixelLayout(dqmitems, plot+":" +sector+" "+station+" fr_hr", *rows)

apply_dqm_items_to_new_back_end(dqmitems, __file__)
