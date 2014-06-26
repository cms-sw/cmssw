def tklayout(i, p, *rows): i["Collisions/TrackingFeedBack/" + p] = DQMItem(layout=rows)

tklayout(dqmitems, "00 - Number Of Tracks",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfTracks_HeavyIonTk",
    'description': "Number of Reconstructed Tracks  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> "}])
tklayout(dqmitems, "01 - Track Pt",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/TrackPt_ImpactPoint_HeavyIonTk",
    'description': "Pt of Reconstructed Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> "}])
tklayout(dqmitems, "02 - Track Phi",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/TrackPhi_ImpactPoint_HeavyIonTk",
    'description': "Phi distribution of Reconstructed Tracks  -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> "}])
tklayout(dqmitems, "03 - Track Eta",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/TrackEta_ImpactPoint_HeavyIonTk",
    'description': " Eta distribution of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> "}])
tklayout(dqmitems, "04 - X-Position Of Closest Approach",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/xPointOfClosestApproach_HeavyIonTk",
    'description': "x coordinate of closest point wrt the beam "}])
tklayout(dqmitems, "05 - Y-Position Of Closest Approach",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/yPointOfClosestApproach_HeavyIonTk",
    'description': "y coordinate of closest point wrt the beam "}])
tklayout(dqmitems, "06 - Z-Position Of Closest Approach",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/zPointOfClosestApproach_HeavyIonTk",
    'description': "z coordinate of closest point wrt the beam "}])    
tklayout(dqmitems, "07 - Cluster y width vs. cluster eta",
 [{ 'path': "Pixel/Barrel/sizeYvsEta_siPixelClusters_Barrel",
    'description': "Cluster y width as function of cluster eta",
    'draw': { 'withref': "no" }}])
tklayout(dqmitems, "08 - Pixel event BX distribution",
  [{ 'path': "Pixel/pixEvtsPerBX",
     'description': "Distribution of Pixel events (at least 4 modules with digis) versus bucket number. The main contributions of Pixel events should come from the colliding bunches. Filled, but non-colliding bunches should be at least 2 orders of magnitudelower. Empty bunches should be close to zero.", 
     'draw': { 'withref': "no" }}]) 

