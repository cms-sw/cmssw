def shiftpixellayout(i, p, *rows): i["00 Shift/Pixel/" + p] = DQMItem(layout=rows)
shiftpixellayout(dqmitems, "01 - Total number of errors",
  [{ 'path': "Pixel/AdditionalPixelErrors/FedChNErrArray",
     'description': "Total number of errors in a map of FED channels (y-axis) vs. FED (x-axis). Look for channels with thousands of errors!",
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "02 - Error types",
  [{ 'path': "Pixel/AdditionalPixelErrors/FedETypeNErrArray",
     'description': "Total number of errors per error type (y-axis) vs. FED (x-axis). Large amounts (hundreds)of errors other than Timeout, EventNumber, and Dcol,Pixel values would be unusual!",
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "03 - Barrel digi occupancy",
  [{ 'path': "Pixel/Barrel/SUMDIG_ndigisFREQ_Barrel",
     'description': "Total number of events with at least one digi per event per Barrel module. Spikes show new noisy modules or pixels! New holes show new missing modules!",
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "04 - Endcap digi occupancy",
  [{ 'path': "Pixel/Endcap/SUMDIG_ndigisFREQ_Endcap",
     'description': "Total number of events with at least one digi per event per Endcap module. Spikes show new noisy modules or pixels! New holes show new missing modules!",
     'draw': { 'withref': "no" }}]
  )    
shiftpixellayout(dqmitems, "05 - Barrel cluster charge",
  [{ 'path': "Pixel/Barrel/SUMCLU_charge_Barrel",
     'description': "Mean cluster charge in kilo electrons per Barrel module",
     'draw': { 'withref': "no" }}]
  )     
shiftpixellayout(dqmitems, "06 - Barrel cluster size",
  [{ 'path': "Pixel/Barrel/SUMCLU_size_Barrel",
     'description': "Mean cluster size in number of pixels per Barrel module",
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "07 - Endcap cluster charge",
  [{ 'path': "Pixel/Endcap/SUMCLU_charge_Endcap",
     'description': "Mean cluster charge in kilo electrons per Endcap module",
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "08 - Endcap cluster size",
  [{ 'path': "Pixel/Endcap/SUMCLU_size_Endcap",
     'description': "Mean cluster size in number of pixels per Endcap module",
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "09 - Cluster occupancy Barrel Layer 1",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_1",
     'description': "Cluster occupancy of Barrel Layer 1. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "10 - Cluster occupancy Barrel Layer 2",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_2",
     'description': "Cluster occupancy of Barrel Layer 2. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "11 - Cluster occupancy Barrel Layer 3",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_3",
     'description': "Cluster occupancy of Barrel Layer 3. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "12 - Cluster occupancy Endcap -z Disk 1",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_1",
     'description': "Cluster occupancy of Endcap -z Disk 1. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "13 - Cluster occupancy Endcap -z Disk 2",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_2",
     'description': "Cluster occupancy of Endcap -z Disk 2. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "14 - Cluster occupancy Endcap +z Disk 1",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_1",
     'description': "Cluster occupancy of Endcap +z Disk 1. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "15 - Cluster occupancy Endcap +z Disk 2",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_2",
     'description': "Cluster occupancy of Endcap +z Disk 2. Look for new holes compared to the example plot.",
     'draw': { 'withref': "no" }}]
  ) 
shiftpixellayout(dqmitems, "16 - Pixel event rate",
  [{ 'path': "Pixel/pixEventRate",
     'description': "Rate of events with Pixel activity above noise level (at least 4 modules with digis). This rate should be pretty consistently around 4-6Hz in every lumi section. It falls slightly with the beam intenstities going down throughout a fill.", 
     'draw': { 'withref': "no" }}]
  )
shiftpixellayout(dqmitems, "17 - Pixel event BX distribution",
  [{ 'path': "Pixel/pixEvtsPerBX",
     'description': "Distribution of Pixel events (at least 4 modules with digis) versus bucket number. The main contributions of Pixel events should come from the colliding bunches. Filled, but non-colliding bunches should be at least 2 orders of magnitudelower. Empty bunches should be close to zero.", 
     'draw': { 'withref': "no" }}]
  ) 
