from .adapt_to_new_backend import *
dqmitems={}

def tklayout(i, p, *rows): i["Collisions/TrackingFeedBack/" + p] = rows

tklayout(dqmitems, "00 - Number Of Tracks",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfTracks_GenTk",
     'description': "Number of Reconstructed Tracks  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
tklayout(dqmitems, "01 - Track Pt",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/TrackPt_ImpactPoint_GenTk",
     'description': "Pt of Reconstructed Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
tklayout(dqmitems, "02 - Track Phi",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/TrackPhi_ImpactPoint_GenTk",
     'description': "Phi distribution of Reconstructed Tracks  -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
tklayout(dqmitems, "03 - Track Eta",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/TrackEta_ImpactPoint_GenTk",
     'description': " Eta distribution of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOnlineDQMInstructions>SiStripOnlineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
tklayout(dqmitems, "04 - Cluster y width vs. cluster eta",
 [{ 'path': "Pixel/Barrel/sizeYvsEta_siPixelClusters_Barrel",
    'description': "Cluster y width as function of cluster eta",
    'draw': { 'withref': "no" }}])
tklayout(dqmitems, "05 - Pixel event BX distribution",
  [{ 'path': "Pixel/pixEvtsPerBX",
     'description': "Distribution of Pixel events (at least 4 modules with digis) versus bucket number. The main contributions of Pixel events should come from the colliding bunches. Filled, but non-colliding bunches should be at least 2 orders of magnitudelower. Empty bunches should be close to zero.",
     'draw': { 'withref': "no" }}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
