def pixellayout(i, p, *rows): i["Pixel/Layouts/" + p] = DQMItem(layout=rows)

pixellayout(dqmitems, "00a - Pixel_Error_Summary",
  [{ 'path': "Pixel/AdditionalPixelErrors/FedChLErrArray",
     'description': "Error type of last error in a map of FED channels (y-axis) vs. FED (x-axis). Channel 0 is assigned for errors where the channel number is not known.", 
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "00b - Pixel_Error_Summary",
  [{ 'path': "Pixel/AdditionalPixelErrors/FedChNErrArray",
     'description': "Total number of errors in a map of FED channels (y-axis) vs. FED (x-axis). Channel 0 is assigned for errors where the channel number is not known.",
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "00c - Pixel_Error_Summary",
  [{ 'path': "Pixel/AdditionalPixelErrors/FedETypeNErrArray",
     'description': "Total number of errors per error type in a map of error type (y-axis) vs. FED (x-axis)",
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "01 - Pixel_FEDOccupancy_Summary",
  [{ 'path': "Pixel/averageDigiOccupancy",
     'description': "Average digi occupancy based on FED number (0-31 barrel, 32-39 Endcaps)",
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "02 - Pixel_Cluster_Summary",
  [{ 'path': "Pixel/Barrel/SUMOFF_charge_OnTrack_Barrel",
     'description': "Mean cluster charge (OnTrack) in kilo electrons per barrel Ladder",
     'draw': { 'withref': "no" }},
   { 'path': "Pixel/Barrel/SUMOFF_nclusters_OnTrack_Barrel",
     'description': "Mean number of clusters (OnTrack) per event per barrel Ladder",
     'draw': { 'withref': "no" }},
   { 'path': "Pixel/Barrel/SUMOFF_size_OnTrack_Barrel",
     'description': "Mean cluster size (OnTrack) in number of pixels per barrel Ladder",
     'draw': { 'withref': "no" }}],
  [{ 'path': "Pixel/Endcap/SUMOFF_charge_OnTrack_Endcap",
     'description': "Mean cluster charge (OnTrack) in kilo electrons per endcap Blade",
     'draw': { 'withref': "no" }},
   { 'path': "Pixel/Endcap/SUMOFF_nclusters_OnTrack_Endcap",
     'description': "Mean number of clusters (OnTrack) per event per endcap Blade",
     'draw': { 'withref': "no" }},
   { 'path': "Pixel/Endcap/SUMOFF_size_OnTrack_Endcap",
     'description': "Mean cluster size (OnTrack) in number of pixels per barrel Blade",
     'draw': { 'withref': "no" }}])
pixellayout(dqmitems, "03 - Pixel_Track_Summary",
  [{ 'path': "Pixel/Clusters/OnTrack/charge_siPixelClusters", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"},
   { 'path': "Pixel/Clusters/OnTrack/size_siPixelClusters", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}],
  [{ 'path': "Pixel/Clusters/OffTrack/charge_siPixelClusters", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"},
   { 'path': "Pixel/Clusters/OffTrack/size_siPixelClusters", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}],
  [{ 'path': "Pixel/Tracks/ntracks_generalTracks", 'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineSiPixel>Description for the Central DQM Shifter</a>"}]
)
pixellayout(dqmitems, "04a - HitEfficiency_perBarrelLayer",
  [{ 'path': "Pixel/Barrel/HitEfficiency_L1",             'description': "Barrel Layer 1 Hit Efficiency"},
   { 'path': "Pixel/Barrel/HitEfficiency_L2",             'description': "Barrel Layer 2 Hit Efficiency"}],
  [{ 'path': "Pixel/Barrel/HitEfficiency_L3",             'description': "Barrel Layer 3 Hit Efficiency"}]
)
pixellayout(dqmitems, "04b - HitEfficiency_perEndcapDisk",
  [{ 'path': "Pixel/Endcap/HitEfficiency_Dm1",             'description': "Endcap 1m Disk Hit Efficiency"},
   { 'path': "Pixel/Endcap/HitEfficiency_Dm2",             'description': "Endcap 2m Disk Hit Efficiency"}],
  [{ 'path': "Pixel/Endcap/HitEfficiency_Dp1",             'description': "Endcap 1p Disk Hit Efficiency"},
   { 'path': "Pixel/Endcap/HitEfficiency_Dp2",             'description': "Endcap 2p Disk Hit Efficiency"}]
)     
pixellayout(dqmitems, "05 - Barrel OnTrack cluster positions",
  [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_Layer_1", 'description': "Global position of OnTrack clusters in Barrel/Layer_1"}],
  [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_Layer_2", 'description': "Global position of OnTrack clusters in Barrel/Layer_2"}],
  [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_Layer_3", 'description': "Global position of OnTrack clusters in Barrel/Layer_3"}]
)
pixellayout(dqmitems, "06 - Endcap OnTrack cluster positions",
  [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_mz_Disk_1", 'description': "Global position of OnTrack clusters in Endcap -z Disk_1"},
   { 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_mz_Disk_2", 'description': "Global position of OnTrack clusters in Endcap -z Disk_2"}],
  [{ 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_pz_Disk_1", 'description': "Global position of OnTrack clusters in Endcap +z Disk_1"},
   { 'path': "Pixel/Clusters/OnTrack/position_siPixelClusters_pz_Disk_2", 'description': "Global position of OnTrack clusters in Endcap +z Disk_2"}]
)
pixellayout(dqmitems, "07 - Pixel_Digi_Summary",
  [{ 'path': "Pixel/Barrel/SUMOFF_adc_Barrel",
     'description': "Mean digi charge in ADC counts per barrel Ladder",
     'draw': { 'withref': "yes" }},
   { 'path': "Pixel/Barrel/SUMOFF_ndigis_Barrel",
     'description': "Mean number of digis per event per barrel Ladder",
     'draw': { 'withref': "yes" }}],
  [{ 'path': "Pixel/Endcap/SUMOFF_adc_Endcap",
     'description': "Mean digi charge in ADC counts per endcap Blade",
     'draw': { 'withref': "yes" }},
   { 'path': "Pixel/Endcap/SUMOFF_ndigis_Endcap",
     'description': "Mean number of digis per event per endcap Blade",
     'draw': { 'withref': "yes" }}]
  )
