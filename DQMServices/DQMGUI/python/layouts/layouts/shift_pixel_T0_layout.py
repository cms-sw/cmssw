from .adapt_to_new_backend import *
dqmitems={}

def shiftpixellayout(i, p, *rows): i["00 Shift/Pixel/" + p] = rows
shiftpixellayout(dqmitems, "01 - Barrel OnTrack cluster positions",
        [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_Layer_1", 'description': "Global position of OnTrack clusters in Barrel/Layer_1 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"}],
        [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_Layer_2", 'description': "Global position of OnTrack clusters in Barrel/Layer_2 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"}],
        [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_Layer_3", 'description': "Global position of OnTrack clusters in Barrel/Layer_3 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"}]
)
shiftpixellayout(dqmitems, "02 - Endcap OnTrack cluster positions",
        [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_mz_Disk_1", 'description': "Global position of OnTrack clusters in Endcap -z Disk_1 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"},
            { 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_mz_Disk_2", 'description': "Global position of OnTrack clusters in Endcap -z Disk_2 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"}],
        [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_pz_Disk_1", 'description': "Global position of OnTrack clusters in Endcap +z Disk_1 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"},
            { 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_pz_Disk_2", 'description': "Global position of OnTrack clusters in Endcap +z Disk_2 <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>"}]
)
shiftpixellayout(dqmitems, "03 - Mean digi charge Barrel",
  [{ 'path': "Pixel/Barrel/SUMOFF_adc_Barrel", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "04 - Mean digi occupancy Barrel",
  [{ 'path': "Pixel/Barrel/SUMOFF_ndigis_Barrel", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "05 - Mean digi charge Endcap",
  [{ 'path': "Pixel/Endcap/SUMOFF_adc_Endcap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "06 - Mean digi occupancy Endcap",
  [{ 'path': "Pixel/Endcap/SUMOFF_ndigis_Endcap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "07 - Charge of clusters on tracks Barrel",
  [{ 'path': "Pixel/Clusters/OnTrack/charge_siPixelClusters_Barrel", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "08 - Charge of clusters on tracks Endcap",
  [{ 'path': "Pixel/Clusters/OnTrack/charge_siPixelClusters_Endcap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "09 - Charge of clusters off tracks Barrel",
  [{ 'path': "Pixel/Clusters/OffTrack/charge_siPixelClusters_Barrel", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "10 - Charge of clusters off tracks Endcap",
  [{ 'path': "Pixel/Clusters/OffTrack/charge_siPixelClusters_Endcap", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
shiftpixellayout(dqmitems, "11 - Pixel track counters",
  [{ 'path': "Pixel/Tracks/ntracks_generalTracks", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)

apply_dqm_items_to_new_back_end(dqmitems, __file__)
