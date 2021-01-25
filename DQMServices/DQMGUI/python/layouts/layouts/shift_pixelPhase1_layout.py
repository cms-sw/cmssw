from .adapt_to_new_backend import *
dqmitems={}

def shiftpixelP1layout(i, p, *rows): i["00 Shift/PixelPhase1/" + p] = rows
shiftpixelP1layout(dqmitems, "00 - PixelPhase1 ReportSummary: Layer or Disk vs subdet",
   [{ 'path': "PixelPhase1/EventInfo/reportSummaryMap",
      'description': "Summary results of qulity tests: Layer/Disk (y-axis) vs. Subdetectors (x-axis). See the PixelPhase1/Summary/ directory for more details.",
      'draw': { 'withref': "no", 'drawopts': "COLZTEXT" }}]
   )
shiftpixelP1layout(dqmitems, "01 - PixelPhase1_Error_Summary",
   [{ 'path': "PixelPhase1/FED/nerrors_per_type_per_FED",
      'description': "Number of Errors of each type per FED. Channel 0 is assigned for errors where the channel number is not known.",
      'draw': { 'withref': "no" }}]
   )

shiftpixelP1layout(dqmitems, "02 - PixelPhase1 Digis: Ladder vs Module barrel summary",
   [{ 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/num_digis_per_SignedModule_per_SignedLadder_PXLayer_1",
      'description': "Profile of digis per event and DetID by signed ladder (y-axis) vs signed module (x-axis) in layer 1 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    { 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/num_digis_per_SignedModule_per_SignedLadder_PXLayer_2",
      'description': "Profile of digis per event and DetID by signed ladder (y-axis) vs signed module (x-axis) in layer 2 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   [{ 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/num_digis_per_SignedModule_per_SignedLadder_PXLayer_3",
      'description': "Profile of digis per event and DetID by signed ladder (y-axis) vs signed module (x-axis) in layer 3 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    { 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/num_digis_per_SignedModule_per_SignedLadder_PXLayer_4",
      'description': "Profile of digis per event and DetID by signed ladder (y-axis) vs signed module (x-axis) in layer 4 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   )

shiftpixelP1layout(dqmitems,"03 - PixelPhase1 Digis: BladePannel vs Disk endcap summary",
   [{ 'path': "PixelPhase1/Phase1_MechanicalView/PXForward/num_digis_per_PXDisk_per_SignedBladePanel_PXRing_1",
      'description': "Profile of number of digis per event and detId by signed blade pannel (y-axis) vs signed disk (x-axis) in ring 1 of endcap",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    { 'path': "PixelPhase1/Phase1_MechanicalView/PXForward/num_digis_per_PXDisk_per_SignedBladePanel_PXRing_2",
      'description': "Profile of number of digis per event and detId by signed blade pannel (y-axis) vs signed disk (x-axis) in ring 2 of endcap",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   )

shiftpixelP1layout(dqmitems, "04 - PixelPhase1_Cluster_Charge",
   [{ 'path': "PixelPhase1/Tracks/charge_PXBarrel",
      'description': "Corrected Cluster charge (On Track) in the BPix modules",
      'draw': { 'withref': "no" }},
    { 'path': "PixelPhase1/Tracks/charge_PXForward",
      'description': "Corrected Cluster charge (On Track) in FPix modules",
      'draw': { 'withref': "no" }}]
   )

shiftpixelP1layout(dqmitems, "05 - PixelPhase1 Dead ROCs Barrel",
   [{ 'path': "PixelPhase1/deadRocTrendLayer_1",
      'description': "Number of Dead ROCs in BPix Layer1",
      'draw': { 'withref': "no" }},
    { 'path': "PixelPhase1/deadRocTrendLayer_2",
      'description': "Number of Dead ROCs in BPix Layer2",
      'draw': { 'withref': "no" }}],
    [{ 'path': "PixelPhase1/deadRocTrendLayer_3",
      'description': "Number of Dead ROCs in BPix Layer3",
      'draw': { 'withref': "no" }},
    { 'path': "PixelPhase1/deadRocTrendLayer_4",
      'description': "Number of Dead ROCs in BPix Layer4",
      'draw': { 'withref': "no" }}]
)

shiftpixelP1layout(dqmitems, "05bis - PixelPhase1 Dead ROCs Endcaps",
   [{ 'path': "PixelPhase1/deadRocTrendRing_1",
      'description': "Number of dead ROCs in FPix Ring 1",
      'draw': { 'withref': "no" }},
    { 'path': "PixelPhase1/deadRocTrendRing_2",
      'description': "Number of dead ROCs in FPix Ring 2",
      'draw': { 'withref': "no" }}]
)

shiftpixelP1layout(dqmitems, "06 - PixelPhase1 Dead Channels per ROC: Ladder vs Module barrel summary",
   [{ 'path': "PixelPhase1/FED/Dead Channels per ROC_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1",
      'description': "Profile of dead channels per ROC by signed ladder (y-axis) vs signed module (x-axis) in layer 1 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ"}},
    { 'path': "PixelPhase1/FED/Dead Channels per ROC_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2",
      'description': "Profile of dead channels per ROC by signed ladder (y-axis) vs signed module (x-axis) in layer 2 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   [{ 'path': "PixelPhase1/FED/Dead Channels per ROC_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3",
      'description': "Profile of dead channels per ROC by signed ladder (y-axis) vs signed module (x-axis) in layer 3 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    { 'path': "PixelPhase1/FED/Dead Channels per ROC_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4",
      'description': "Profile of dead channels per ROC by signed ladder (y-axis) vs signed module (x-axis) in layer 4 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   )

shiftpixelP1layout(dqmitems,"07 - PixelPhase1 Dead Channels per ROC: BladePannel vs Disk endcap summary",
   [{ 'path': "PixelPhase1/FED/Dead Channels per ROC_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1",
      'description': "Profile of number of dead Channels per ROC by signed blade pannel (y-axis) vs signed disk (x-axis) in ring 1 of endcap",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    { 'path': "PixelPhase1/FED/Dead Channels per ROC_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2",
      'description': "Profile of number of dead Channels per ROC by signed blade pannel (y-axis) vs signed disk (x-axis) in ring 2 of endcap",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   )

apply_dqm_items_to_new_back_end(dqmitems, __file__)
