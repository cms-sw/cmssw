from .adapt_to_new_backend import *
dqmitems={}

def GEMLayout(i, p, *rows): i["GEM/Layouts/" + p] = rows
#def GEMLayout(i, p, *rows): return 1 # this line and the following one line is for printing out indices


nIdx = 0

GEMLayout(dqmitems, "%02i Summary"%nIdx, 
    [{"path" : "GEM/EventInfo/reportSummaryMap", 
    "description": 'For more information (... under construction)'}])
nIdx += 1

GEMLayout(dqmitems, "%02i AMC status"%nIdx, 
    [{"path" : "GEM/StatusDigi/amc_statusflag", 
    "description": 'For more information (... under construction)'}])
nIdx += 1

# StatusDigi
GEMLayout(dqmitems, "%02i GEB input status"%nIdx, 
    [
      {"path": "GEM/StatusDigi/geb_input_status_st_m1_la_1", "description": "GEB input status in GE-1/1"}, 
      {"path": "GEM/StatusDigi/geb_input_status_st_m1_la_2", "description": "GEB input status in GE-1/2"}, 
    ], 
    [
      {"path": "GEM/StatusDigi/geb_input_status_st_p1_la_1", "description": "GEB input status in GE+1/1"}, 
      {"path": "GEM/StatusDigi/geb_input_status_st_p1_la_2", "description": "GEB input status in GE+1/2"}, 
    ])
nIdx += 1


GeminisId = [ i + 1 for i in range(36) ]
GeminisIdWithTitle = [ {"chid": gid, "title": "GEMINI%02i"%gid} for gid in GeminisId ]
listLayers = ["p1_1", "p1_2", "m1_1", "m1_2"]
listLayersWithTitle = [ [ s, "GE%s%s%s"%("+" if s[ 0 ] == "p" else "-", s[ 1 ], s[ 3 ]) ] for s in listLayers ]
bIsLayerWise = True
bIsGlobalPos = True

strTitleFmt = "%(idx)02i %(title)s_%(layer)s"


listGEMChambers = []

if bIsLayerWise: 
  for layer in listLayersWithTitle:
    for gemini in GeminisIdWithTitle:
      listGEMChambers.append([gemini, layer])
else: 
  for gemini in GeminisIdWithTitle:
    for layer in listLayersWithTitle:
      listGEMChambers.append([gemini, layer])


if bIsGlobalPos: 
  for layer in listLayersWithTitle: 
    GEMLayout(dqmitems, "%02i Global position %s"%(nIdx, layer[ 1 ]), 
        [{"path": "GEM/recHit/recHit_globalPos_Gemini_GE" + layer[ 0 ], 
          "description": "Global position"}])
    nIdx += 1

for itCh in listGEMChambers: 
  gemini = itCh[ 0 ]
  layer  = itCh[ 1 ]
  
  strID = "Gemini_%i_GE%s"%(gemini[ "chid" ], layer[ 0 ])
  
  listU1 = ["GEM/StatusDigi/vfatStatus_QualityFlag_" + strID, "VFAT quality"]
  listU2 = ["GEM/StatusDigi/vfatStatus_BC_"          + strID, "Bunch crossing"]
  listU3 = ["GEM/StatusDigi/vfatStatus_EC_"          + strID, "Event counter"]
  listL1 = ["GEM/digi/Digi_Strips_"                  + strID, "Number of Digi Strips"]
  listL2 = ["GEM/recHit/VFAT_vs_ClusterSize_"        + strID, "VFAT vs ClusterSize"]
  #strPathRHHitX     = "GEM/recHit/recHit_x_" + strID
  
  gemini[ "idx" ] = nIdx
  gemini[ "layer" ] = layer[ 1 ]
  strTitle = strTitleFmt%gemini
  
  GEMLayout(dqmitems, strTitle, 
    [
      {'path': listU1[ 0 ], 'description': listU1[ 1 ]},
      {'path': listU2[ 0 ], 'description': listU2[ 1 ]},
      {'path': listU3[ 0 ], 'description': listU3[ 1 ]},
    ],
    [
      {'path': listL1[ 0 ], 'description': listL1[ 1 ]},
      {'path': listL2[ 0 ], 'description': listL2[ 1 ]},
    ]
  )
  
  nIdx += 1



apply_dqm_items_to_new_back_end(dqmitems, __file__)
