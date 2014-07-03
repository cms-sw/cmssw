def pixellayout(i, p, *rows): i["Pixel/Layouts/" + p] = DQMItem(layout=rows)
pixellayout(dqmitems, "000 - Pixel FED Occupancy vs Lumi Sections",
            [{ 'path': "Pixel/avgfedDigiOccvsLumi",
               'description': "Relative digi occupancy per FED compared to average FED occupancy, plotted versus lumi section and updated every 15 lumi sections. Expected value for healthy FEDs is around 1.",
               'draw': { 'withref': "no" }}]
              )
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
pixellayout(dqmitems, "01a - Pixel_Noise_Summary",
  [{ 'path': "Pixel/Barrel/SUMDIG_ndigisFREQ_Barrel",
     'description': "Total number of events with at least one digi per event per barrel module - spikes show noisy modules or pixels!",
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "01b - Pixel_Noise_Summary",
  [{ 'path': "Pixel/Endcap/SUMDIG_ndigisFREQ_Endcap",
     'description': "Total number of events with at least one digi per event per endcap module - spikes show noisy modules or pixels!",
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "02 - Pixel_Charge_Summary",
  [{ 'path': "Pixel/Barrel/ALLMODS_adcCOMB_Barrel",
     'description': "Distribution of raw charge for all digis recorded in the Barrel modules - dominant peak should be around 90-100 ADC",
     'draw': { 'withref': "yes" }},
   { 'path': "Pixel/Endcap/ALLMODS_adcCOMB_Endcap",
     'description': "Distribution of raw charge for all digis recorded in the Endcap modules - dominant peak should be around 90-100 ADC",
     'draw': { 'withref': "yes" }}],
  [{ 'path': "Pixel/Barrel/ALLMODS_chargeCOMB_Barrel",
     'description': "Distribution of gain corrected cluster charge for all clusters in the Barrel - dominant peak should be around 21 ke-",
     'draw': { 'withref': "yes" }},
   { 'path': "Pixel/Endcap/ALLMODS_chargeCOMB_Endcap",
     'description': "Distribution of gain corrected cluster charge for all clusters in the Endcaps - dominant peak should be around 21 ke-",
     'draw': { 'withref': "yes" }}]
  )     
pixellayout(dqmitems, "03 - Pixel_Digi_Barrel_Summary",
  [{ 'path': "Pixel/Barrel/SUMDIG_adc_Barrel",
     'description': "Mean digi charge in ADC counts per barrel module",
     'draw': { 'withref': "no" }}],
  [{ 'path': "Pixel/Barrel/SUMDIG_ndigis_Barrel",
     'description': "Mean number of digis per event per barrel module",
     'draw': { 'withref': "no" }}]
  )     
pixellayout(dqmitems, "04 - Pixel_Digi_Endcap_Summary",
  [{ 'path': "Pixel/Endcap/SUMDIG_adc_Endcap",
     'description': "Mean digi charge in ADC counts per endcap module",
     'draw': { 'withref': "no" }}],
  [{ 'path': "Pixel/Endcap/SUMDIG_ndigis_Endcap",
     'description': "Mean number of digis per event per endcap module",
     'draw': { 'withref': "no" }}]
  )     
pixellayout(dqmitems, "05 - Pixel_Cluster_Barrel_Summary",
  [{ 'path': "Pixel/Barrel/SUMCLU_charge_Barrel",
     'description': "Mean cluster charge in kilo electrons per barrel module",
     'draw': { 'withref': "no" }}],
  [{ 'path': "Pixel/Barrel/SUMCLU_nclusters_Barrel",
     'description': "Mean number of clusters per event per barrel module",
     'draw': { 'withref': "no" }}],
  [{ 'path': "Pixel/Barrel/SUMCLU_size_Barrel",
     'description': "Mean cluster size in number of pixels per barrel module",
     'draw': { 'withref': "no" }}]
  )     
pixellayout(dqmitems, "06 - Pixel_Cluster_Endcap_Summary",
  [{ 'path': "Pixel/Endcap/SUMCLU_charge_Endcap",
     'description': "Mean cluster charge in kilo electrons per endcap module",
'draw': { 'withref': "no" }}],
[{ 'path': "Pixel/Endcap/SUMCLU_nclusters_Endcap",
     'description': "Mean number of clusters per event per endcap module",
     'draw': { 'withref': "no" }}],
  [{ 'path': "Pixel/Endcap/SUMCLU_size_Endcap",
     'description': "Mean cluster size in number of pixels per barrel module",
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "20a - Cluster occupancy Barrel Layer 1",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_1",
     'description': "Cluster occupancy of Barrel Layer 1",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "20b - Cluster occupancy Barrel Layer 2",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_2",
     'description': "Cluster occupancy of Barrel Layer 2",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "20c - Cluster occupancy Barrel Layer 3",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_Layer_3",
     'description': "Cluster occupancy of Barrel Layer 3",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "20d - Cluster occupancy Endcap -z Disk 1",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_1",
     'description': "Cluster occupancy of Endcap -z Disk 1",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "20e - Cluster occupancy Endcap -z Disk 2",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_mz_Disk_2",
     'description': "Cluster occupancy of Endcap -z Disk 2",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "20f - Cluster occupancy Endcap +z Disk 1",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_1",
     'description': "Cluster occupancy of Endcap +z Disk 1",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "20g - Cluster occupancy Endcap +z Disk 2",
  [{ 'path': "Pixel/Clusters/OffTrack/position_siPixelClusters_pz_Disk_2",
     'description': "Cluster occupancy of Endcap +z Disk 2",
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "30a - Pixel event rates",
  [{ 'path': "Pixel/pixEventRate",
     'description': "Rate of events with Pixel activity above noise level (at least 4 modules with digis)", 
     'draw': { 'withref': "no" }}]
  )
pixellayout(dqmitems, "30b - Pixel event BX distribution",
  [{ 'path': "Pixel/pixEvtsPerBX",
     'description': "Distribution of Pixel events (at least 4 modules with digis) versus bucket number", 
     'draw': { 'withref': "no" }}]
  ) 
pixellayout(dqmitems, "31 - Cluster_y_width_vs_cluster_eta",
  [{ 'path': "Pixel/Barrel/sizeYvsEta_siPixelClusters_Barrel",
     'description': "Cluster y width as function of cluster eta", 
     'draw': { 'withref': "no" }}]
  )
