from .adapt_to_new_backend import *
dqmitems={}

def trackervalidationlayout(i, p, *rows): i["DataLayouts/Tk/" + p] = rows

trackervalidationlayout(dqmitems, "01 - TEC-",
                        [{ 'path':"SiStrip/MechanicalView/TEC/MINUS/Summary_ClusterStoNCorr_OnTrack__TEC__MINUS",
                           'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters"},
                         {'path':"SiStrip/MechanicalView/TEC/MINUS/Summary_TotalNumberOfClusters_OnTrack__TEC__MINUS",
                          'description': "Number of clusters"}],
                        [{'path':"SiStrip/MechanicalView/TEC/MINUS/Summary_ClusterCharge_OffTrack__TEC__MINUS",
                          'description': "OffTrack cluster charge"},
                         {'path':"SiStrip/MechanicalView/TEC/MINUS/Summary_TotalNumberOfClusters_OffTrack__TEC__MINUS",
                          'description': "Number of clusters"}])

trackervalidationlayout(dqmitems, "02 - TEC+",
                        [{ 'path':"SiStrip/MechanicalView/TEC/PLUS/Summary_ClusterStoNCorr_OnTrack__TEC__PLUS",
                           'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters"},
                         {'path':"SiStrip/MechanicalView/TEC/PLUS/Summary_TotalNumberOfClusters_OnTrack__TEC__PLUS",
                          'description': "Number of clusters"}],
                        [{'path':"SiStrip/MechanicalView/TEC/PLUS/Summary_ClusterCharge_OffTrack__TEC__PLUS",
                          'description': "OffTrack cluster charge"},
                         {'path':"SiStrip/MechanicalView/TEC/PLUS/Summary_TotalNumberOfClusters_OffTrack__TEC__PLUS",
                          'description': "Number of clusters"}])

trackervalidationlayout(dqmitems, "03 - TIB",
                        [{ 'path':"SiStrip/MechanicalView/TIB/Summary_ClusterStoNCorr_OnTrack__TIB",
                           'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters"},
                         {'path':"SiStrip/MechanicalView/TIB/Summary_TotalNumberOfClusters_OnTrack__TIB",
                          'description': "Number of clusters"}],
                        [{'path':"SiStrip/MechanicalView/TIB/Summary_ClusterCharge_OffTrack__TIB",
                          'description': "OffTrack cluster charge"},
                         {'path':"SiStrip/MechanicalView/TIB/Summary_TotalNumberOfClusters_OffTrack__TIB",
                          'description': "Number of clusters"}])

trackervalidationlayout(dqmitems, "04 - TID-",
                        [{ 'path':"SiStrip/MechanicalView/TID/MINUS/Summary_ClusterStoNCorr_OnTrack__TID__MINUS",
                           'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters"},
                         {'path':"SiStrip/MechanicalView/TID/MINUS/Summary_TotalNumberOfClusters_OnTrack__TID__MINUS",
                          'description': "Number of clusters"}],
                        [{'path':"SiStrip/MechanicalView/TID/MINUS/Summary_ClusterCharge_OffTrack__TID__MINUS",
                          'description': "OffTrack cluster charge"},
                         {'path':"SiStrip/MechanicalView/TID/MINUS/Summary_TotalNumberOfClusters_OffTrack__TID__MINUS",
                          'description': "Number of clusters"}])

trackervalidationlayout(dqmitems, "05 - TID+",
                        [{ 'path':"SiStrip/MechanicalView/TID/PLUS/Summary_ClusterStoNCorr_OnTrack__TID__PLUS",
                           'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters"},
                         {'path':"SiStrip/MechanicalView/TID/PLUS/Summary_TotalNumberOfClusters_OnTrack__TID__PLUS",
                          'description': "Number of clusters"}],
                        [{'path':"SiStrip/MechanicalView/TID/PLUS/Summary_ClusterCharge_OffTrack__TID__PLUS",
                          'description': "OffTrack cluster charge"},
                         {'path':"SiStrip/MechanicalView/TID/PLUS/Summary_TotalNumberOfClusters_OffTrack__TID__PLUS",
                          'description': "Number of clusters"}])

trackervalidationlayout(dqmitems, "06 - TOB",
                        [{ 'path':"SiStrip/MechanicalView/TOB/Summary_ClusterStoNCorr_OnTrack__TOB",
                           'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters"},
                         {'path':"SiStrip/MechanicalView/TOB/Summary_TotalNumberOfClusters_OnTrack__TOB",
                          'description': "Number of clusters"}],
                        [{'path':"SiStrip/MechanicalView/TOB/Summary_ClusterCharge_OffTrack__TOB",
                          'description': "OffTrack cluster charge"},
                         {'path':"SiStrip/MechanicalView/TOB/Summary_TotalNumberOfClusters_OffTrack__TOB",
                          'description': "Number of clusters"}])

trackervalidationlayout(dqmitems, "07 - Pixel Digis ADC Barrel",
                        [{ 'path': "PixelPhase1/Phase1_MechanicalView/num_digis_PXBarrel",
                           'description': "Number of digis per event in PXBarrel",
                           'draw': { 'withref': "no" }},
                         { 'path': "PixelPhase1/Phase1_MechanicalView/adc_PXBarrel",
                           'description': "Adc distribution of digis per event per barrel module",
                           'draw': { 'withref': "no" }}],
                        [{ 'path': "PixelPhase1/Phase1_MechanicalView/num_digis_per_LumiBlock_PXBarrel",
                           'description': "Mean adc value per lumisection",
                           'draw': { 'withref': "no" }},
                         { 'path': "PixelPhase1/Phase1_MechanicalView/adc_per_LumiBlock_PXBarrel",
                           'description': "Mean adc value per lumisection",
                           'draw': { 'withref': "no" }}])

trackervalidationlayout(dqmitems, "08 - Pixel Digis ADC Endcap",
                        [{ 'path': "PixelPhase1/Phase1_MechanicalView/num_digis_PXForward",
                           'description': "Number of digis per event in Forward",
                           'draw': { 'withref': "no" }},
                         { 'path': "PixelPhase1/Phase1_MechanicalView/adc_PXForward",
                           'description': "Adc distribution of digis per event per forward module",
                           'draw': { 'withref': "no" }}],
                        [{ 'path': "PixelPhase1/Phase1_MechanicalView/num_digis_per_LumiBlock_PXForward",
                           'description': "Mean adc value per lumisection",
                           'draw': { 'withref': "no" }},
                         { 'path': "PixelPhase1/Phase1_MechanicalView/adc_per_LumiBlock_PXForward",
                           'description': "Mean adc value per lumisection",
                           'draw': { 'withref': "no" }}])

trackervalidationlayout(dqmitems, "09 - Pixel Occupancy",
                        [{ 'path':"PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1",
                           'description': ""},
                         {'path':"PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2",
                          'description': ""}],
                        [{ 'path':"PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3",
                           'description': ""},
                         {'path':"PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4",
                          'description': ""}],
                        [{ 'path':"PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1",
                           'description': ""},
                        { 'path':"PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2",
                           'description': ""}])

trackervalidationlayout(dqmitems, "10 - PXBarrel OnTk Clusters Charge",
                        [{ 'path':"PixelPhase1/Tracks/PXBarrel/charge_PXLayer_1",
                           'description': ""},
                         {'path':"PixelPhase1/Tracks/PXBarrel/charge_PXLayer_2",
                          'description': ""}],
                        [{'path':"PixelPhase1/Tracks/PXBarrel/charge_PXLayer_3",
                          'description': ""},
                         {'path':"PixelPhase1/Tracks/PXBarrel/charge_PXLayer_4",
                          'description': ""}])

trackervalidationlayout(dqmitems, "11 - PXEndcap OnTk Clusters Charge",
                        [{ 'path':"PixelPhase1/Tracks/PXForward/charge_PXDisk_-3",
                           'description': ""},
                         {'path':"PixelPhase1/Tracks/PXForward/charge_PXDisk_-2",
                          'description': ""}],
                        [{'path':"PixelPhase1/Tracks/PXForward/charge_PXDisk_-1",
                          'description': ""},
                         {'path':"PixelPhase1/Tracks/PXForward/charge_PXDisk_+1",
                          'description': ""}],
                         [{'path':"PixelPhase1/Tracks/PXForward/charge_PXDisk_+2",
                          'description': ""},
                         {'path':"PixelPhase1/Tracks/PXForward/charge_PXDisk_+3",
                          'description': ""}])

trackervalidationlayout(dqmitems, "12 - PXBarrel OnTk Clusters Size",
                        [{ 'path':"PixelPhase1/Tracks/PXBarrel/size_PXLayer_1",
                           'description': ""},
                         {'path':"PixelPhase1/Tracks/PXBarrel/size_PXLayer_2",
                          'description': ""}],
                        [{'path':"PixelPhase1/Tracks/PXBarrel/size_PXLayer_3",
                          'description': ""},
                         {'path':"PixelPhase1/Tracks/PXBarrel/size_PXLayer_4",
                          'description': ""}])

trackervalidationlayout(dqmitems, "13 - PXEndcap OnTk Clusters Size",
                        [{ 'path':"PixelPhase1/Tracks/PXForward/size_PXDisk_-3",
                           'description': ""},
                         {'path':"PixelPhase1/Tracks/PXForward/size_PXDisk_-2",
                          'description': ""}],
                        [{'path':"PixelPhase1/Tracks/PXForward/size_PXDisk_-1",
                          'description': ""},
                         {'path':"PixelPhase1/Tracks/PXForward/size_PXDisk_+1",
                          'description': ""}],
                         [{'path':"PixelPhase1/Tracks/PXForward/size_PXDisk_+2",
                          'description': ""},
                         {'path':"PixelPhase1/Tracks/PXForward/size_PXDisk_+3",
                          'description': ""}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
