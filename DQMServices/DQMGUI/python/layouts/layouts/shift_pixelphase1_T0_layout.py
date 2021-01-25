from .adapt_to_new_backend import *
dqmitems={}

def shiftpixellayout(i, p, *rows): i["00 Shift/PixelPhase1/" + p] = rows
shiftpixellayout(dqmitems, "00 - PixelPhase1 ReportSummary: Layer or Disk vs subdet",
   [{ 'path':  "PixelPhase1/EventInfo/reportSummaryMap",
      'description': "Summary results of qulity tests: Layer/Disk (y-axis) vs. Subdetectors (x-axis). See the PixelPhase1/Summary/ directory for more details.",
      'draw': { 'withref': "no", 'drawopts': "COLZTEXT" }}]
   )
shiftpixellayout(dqmitems, "01 - PixelPhase1_Error_Summary",
   [{ 'path': "PixelPhase1/FED/nerrors_per_type_per_FED",
      'description': "Number of Errors of each type per FED. Channel 0 is assigned for errors where the channel number is not known.",
      'draw': { 'withref': "no" }}]
   )

shiftpixellayout(dqmitems, "02 - PixelPhase1 FED Occupancy vs Lumi Block",
            [{ 'path': "PixelPhase1/num_feddigistrend_per_LumiBlock_per_FED",
               'description': "Number of digis per FED and Lumi block",
               'draw': { 'withref': "no" }}]
              )

shiftpixellayout(dqmitems, "03 - PixelPhase1_Cluster_Number",
   [{ 'path': "PixelPhase1/Phase1_MechanicalView/num_clusters_PXBarrel",
      'description': "Number of clusters per event in Barrel",
      'draw': { 'withref': "no" }},
    { 'path': "PixelPhase1/Phase1_MechanicalView/num_clusters_PXForward",
      'description': "Number of clusters per event in Forward",
      'draw': { 'withref': "no" }}],
   [{ 'path': "PixelPhase1/Phase1_MechanicalView/num_clusters_per_LumiBlock_PXBarrel",
      'description': "Mean cluster value per lumisection in barrel",
      'draw': { 'withref': "no" }},
    { 'path': "PixelPhase1/Phase1_MechanicalView/num_clusters_per_LumiBlock_PXForward",
      'description': "Mean cluster value per lumisection in endcap",
      'draw': { 'withref': "no" }}]
   )

shiftpixellayout(dqmitems, "04 - Charge and size on track",
  [{ 'path': "PixelPhase1/Tracks/charge_PXBarrel",
     'description': "charge PXBarrel",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/charge_PXForward",
     'description': "charge PXForward",
     'draw': {'withref' : "no"}}],
  [{ 'path': "PixelPhase1/Tracks/size_PXBarrel",
     'description': "size of PXBarrel",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/size_PXForward",
     'description': "size of PXForward",
     'draw': {'withref' : "no"}}]
  )

shiftpixellayout(dqmitems, "04a - Cluster on track charge per Inner Ladders",
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_1",
     'description': "corrected cluster charge (on track) in inner ladders in PXLayer 1",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_2",
     'description': "corrected cluster charge (on track) in inner ladders in PXLayer 2",
     'draw': {'withref' : "no"}}],
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_3",
     'description': "corrected cluster charge (on track) in inner ladders in PXLayer 3",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/PXBarrel/chargeInner_PXLayer_4",
     'description': "corrected cluster charge (on track) in inner ladders in PXLayer 4",
     'draw': {'withref' : "no"}}]
  )

shiftpixellayout(dqmitems, "04b - Cluster on track charge per Outer Ladders",
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/chargeOuter_PXLayer_1",
     'description': "corrected cluster charge (on track) in outer ladders in PXLayer 1",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/PXBarrel/chargeOuter_PXLayer_2",
     'description': "corrected cluster charge (on track) in outer ladders in PXLayer 2",
     'draw': {'withref' : "no"}}],
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/chargeOuter_PXLayer_3",
     'description': "corrected cluster charge (on track) in outer ladders in PXLayer 3",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/PXBarrel/chargeOuter_PXLayer_4",
     'description': "corrected cluster charge (on track) in outer ladders in PXLayer 4",
     'draw': {'withref' : "no"}}]
  ) 

shiftpixellayout(dqmitems, "04c - Cluster charge (on-track) per Disk",
  [{'path': "PixelPhase1/Tracks/PXForward/charge_PXDisk_+1",
  'description': "Cluster on track charge in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk +1 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/charge_PXDisk_+2",
  'description': "Cluster on track charge in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk +2 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/charge_PXDisk_+3",
  'description': "Cluster on track charge in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk +3 of pixel endcap",
  'draw': { 'withref': "no"}}],
  [{'path': "PixelPhase1/Tracks/PXForward/charge_PXDisk_-1",
  'description': "Clusteron on track charge in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk -1 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/charge_PXDisk_-2",
  'description': "Cluster on track charge in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk -2 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/charge_PXDisk_-3",
  'description': "Cluster on track charge in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk -3 of pixel endcap",
  'draw': { 'withref': "no"}}],
  )

shiftpixellayout(dqmitems, "05a - PixelPhase1 DeadROC Summary",
  [{ 'path': "PixelPhase1/deadRocTotal",
     'description': "Number of total dead ROCs summary",
     'draw': { 'withref': "no" }}]
  )

shiftpixellayout(dqmitems, "05b - PixelPhase1 Digi Occupancy ROC level: Ladder vs Module barrel",
  [{ 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1",
      'description': "Digi Occupancy by signed ladder (y-axis) vs signed module (x-axis) in layer 1 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
   { 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2",
      'description': "Digi Occupancy by signed ladder (y-axis) vs signed module (x-axis) in layer 2 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
  [{ 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3",
      'description': "Digi Occupancy by signed ladder (y-axis) vs signed module (x-axis) in layer 1 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
   { 'path': "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4",
      'description': "Digi Occupancy by signed ladder (y-axis) vs signed module (x-axis) in layer 2 of barrel",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
  )

shiftpixellayout(dqmitems, "05c - PixelPhase1 Digi Occupancy ROC level: BladePanel vs Disk endcap summary",
   [{ 'path': "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1",
      'description': "Number of digis per ROC per event by signed blade panel (y-axis) vs signed disk (x-axis) in ring 1 of endcap",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    { 'path': "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2",
      'description': "Number of digis per ROC per event by signed blade panel (y-axis) vs signed disk (x-axis) in ring 2 of endcap",
      'draw': { 'withref': "no", 'drawopts': "COLZ" }}],
   )

shiftpixellayout(dqmitems, "06 - ntracks",
  [{ 'path': "PixelPhase1/Tracks/ntracks",
     'description': "Number of Tracks in all pixel det",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/ntracksinpixvolume",
     'description': "Number of Tracks in pix volume",
     'draw': {'withref' : "no"}}]
  )

shiftpixellayout(dqmitems, "07 - PixelPhase1 Residuals",
    [{ 'path': "PixelPhase1/Tracks/residual_x_PXBarrel",
       'description': "Track residuals x in PXBarrel",
       'draw': { 'withref': "no" }},
     { 'path': "PixelPhase1/Tracks/residual_x_PXForward",
       'description': "Track residuals x in PXForward",
       'draw': { 'withref': "no" }}],
    [{ 'path': "PixelPhase1/Tracks/residual_y_PXBarrel",
       'description': "Track residuals y in PXBarrel",
       'draw': { 'withref': "no" }},
     { 'path': "PixelPhase1/Tracks/residual_y_PXForward",
       'description': "Track residuals y in PXForward",
       'draw': { 'withref': "no" }}]
    )

shiftpixellayout(dqmitems, "08a - Hit Efficiency Barrel",
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_1",
     'description': "hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_1",
     'draw': {'withref' : "no", 'drawopts': "COLZ"}},
     { 'path': "PixelPhase1/Tracks/PXBarrel/hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_2",
        'description': "hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_2",
        'draw': {'withref' : "no", 'drawopts': "COLZ"}}],
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_3",
     'description': "hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_3",
     'draw': {'withref' : "no", 'drawopts': "COLZ"}},
     { 'path': "PixelPhase1/Tracks/PXBarrel/hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_4",
        'description': "hitefficiency_per_SignedModule_per_SignedLadder_PXLayer_4",
        'draw': {'withref' : "no", 'drawopts': "COLZ"}}]
  )

shiftpixellayout(dqmitems, "08b - Hit Efficiency Forward",
   [{'path': "PixelPhase1/Tracks/PXForward/hitefficiency_per_PXDisk_per_SignedBladePanel_PXRing_1",
     'description': "hitefficiency_per_PXDisk_per_SignedBl",
     'draw': { 'withref': "no", 'drawopts': "COLZ" }},
    {'path': "PixelPhase1/Tracks/PXForward/hitefficiency_per_PXDisk_per_SignedBladePanel_PXRing_2",
     'description': "hitefficiency_per_PXDisk_per_SignedB2",
     'draw': { 'withref': "no", 'drawopts': "COLZ" }}
    ]
   )

shiftpixellayout(dqmitems, "09a - Cluster size (on-track) per Ladders",
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/size_PXLayer_1",
     'description': "Cluster size (on track) in ladders in PXLayer 1",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/PXBarrel/size_PXLayer_2",
     'description': "Cluster size (on track) in ladders in PXLayer 2",
     'draw': {'withref' : "no"}}],
  [{ 'path': "PixelPhase1/Tracks/PXBarrel/size_PXLayer_3",
     'description': "Cluster size (on track) in ladders in PXLayer 3",
     'draw': {'withref' : "no"}},
   { 'path': "PixelPhase1/Tracks/PXBarrel/size_PXLayer_4",
     'description': "Cluster size (on track) in ladders in PXLayer 4",
     'draw': {'withref' : "no"}}]
  )

shiftpixellayout(dqmitems, "09b - Cluster size (on-track) per Disk",
  [{'path': "PixelPhase1/Tracks/PXForward/size_PXDisk_+1",
  'description': "Cluster on track size in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk +1 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/size_PXDisk_+2",
  'description': "Cluster on track size in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk +2 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/size_PXDisk_+3",
  'description': "Cluster on track size in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk +3 of pixel endcap",
  'draw': { 'withref': "no"}}],
  [{'path': "PixelPhase1/Tracks/PXForward/size_PXDisk_-1",
  'description': "Clusteron on track size in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk -1 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/size_PXDisk_-2",
  'description': "Cluster on track size in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk -2 of pixel endcap",
  'draw': { 'withref': "no"}},
  {'path': "PixelPhase1/Tracks/PXForward/size_PXDisk_-3",
  'description': "Cluster on track size in global coordinates by Global Y (y-axis) vs Global X (x-axis) in disk -3 of pixel endcap",
  'draw': { 'withref': "no"}}],
  )

apply_dqm_items_to_new_back_end(dqmitems, __file__)
