from .adapt_to_new_backend import *
dqmitems={}

def CTPPSTrackingStripLayout(i, p, *rows): i["00 Shift/CTPPS/TrackingStrip/" + p] = rows
def CTPPSTrackingPixelLayout(i, p, *rows): i["00 Shift/CTPPS/TrackingPixel/" + p] = rows

stations = [ "sector 45/station 210", "sector 56/station 210" ]
units = [ "nr", "fr" ]

sectors = [ "sector 45", "sector 56" ]
pixelstations = [ "station 210", "station 220" ]
pix_planes  = [ "0","1","2" ]
pix_planes2 = [ "3","4","5" ]


# layouts with no overlays
for plot in [ "active planes", "vfats with any problem", "track XY profile" ]:
  rows = list()
  for station in stations:
    row = list()
    for unit in units:
      row.append("CTPPS/TrackingStrip/"+station+"/"+unit+"_hr/"+plot)
    rows.append(row)

  CTPPSTrackingStripLayout(dqmitems, plot, *rows)


# layouts with overlays
for plot in [ "planes contributing to fit" ]:
  rows = list()
  for station in stations:
    row = list()
    for unit in units:
      hist_u = "CTPPS/TrackingStrip/"+station+"/"+unit+"_hr/"+plot + " U"
      hist_v = "CTPPS/TrackingStrip/"+station+"/"+unit+"_hr/"+plot + " V"
      row.append( { "path" : hist_u, "overlays" : [ hist_v ] } )
    rows.append(row)

  CTPPSTrackingStripLayout(dqmitems, plot + " UV", *rows)


# per-BX plots
for suffix in [ " (short)" ]:
  plot_list = list()
  for station in stations:
    for unit in units:
      plot_list.append("CTPPS/TrackingStrip/"+station+"/"+unit+"_hr/activity per BX" + suffix)

  base_plot = "CTPPS/events per BX" + suffix
  CTPPSTrackingStripLayout(dqmitems, "activity per BX" + suffix, [ { "path" : base_plot, "overlays" : plot_list } ])

###
# 	CTPPS Pixel
###

for sector in sectors:
  rows = list()
  for station in pixelstations:
    row = list()
    row.append("CTPPS/TrackingPixel/"+sector+"/"+station+"/fr_hr/"+
	"ROCs_hits_multiplicity_per_event vs LS")
    rows.append(row)

  CTPPSTrackingPixelLayout(dqmitems, "ROC hits per event vs LS "+sector, *rows)


for plot in ["number of fired planes per event","track intercept point","number of tracks per event"]:
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

####################################################################################################
# Diamond layouts
####################################################################################################

diamond_stations = [ "sector 45/station 220cyl/cyl_hr", "sector 56/station 220cyl/cyl_hr" ]

def CTPPSTimingDiamondLayout(i, p, *rows): i["00 Shift/CTPPS/TimingDiamond/" + p] = rows

# layouts with no overlays
TimingPlots = [ "active planes", "event category", "leading edge (le and te)", "time over threshold", "hits in planes", "hits in planes lumisection", "HPTDC Errors" ]
TimingDrawOpt = [ {'xmax':"10"}, {'drawopts':"colztext"}, {'xmax':"25"}, {'xmin':"0", 'xmax':"25"}, {'withref':"no"}, {'withref':"no"}, {'drawopts':"colztext"} ]
TimingDescription = [ "It should be with peaks at 0 and 4", 'Most of the event should be in "both"',
    "It should be peaked around 5 ns", "It should be a broad distribution peaked around 12 ns",
    "It should be full", 'It should be similar to "hits in planes"', "It should be empty" ]

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

apply_dqm_items_to_new_back_end(dqmitems, __file__)
