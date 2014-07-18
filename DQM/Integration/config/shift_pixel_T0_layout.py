def shiftpixellayout(i, p, *rows): i["00 Shift/Pixel/" + p] = DQMItem(layout=rows)
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
