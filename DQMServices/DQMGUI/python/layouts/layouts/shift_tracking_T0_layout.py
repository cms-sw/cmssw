from .adapt_to_new_backend import *
dqmitems={}

def shifttrackinglayout(i, p, *rows): i["00 Shift/Tracking/" + p] = rows

shifttrackinglayout(dqmitems, "01 - Tracking ReportSummary",
 [{ 'path': "Tracking/EventInfo/reportSummaryMap",
    'description': " Quality Test results plotted for Tracking parameters : Chi2, TrackRate, #of Hits in Track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "no" }}])
shifttrackinglayout(dqmitems, "02a - Tracks (pp collisions)",
 [{ 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/GeneralProperties/NumberOfTracks_GenTk",
    'description': "Number of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/HitProperties/NumberOfRecHitsPerTrack_GenTk",
    'description': "Number of RecHits per Track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/GeneralProperties/TrackPt_ImpactPoint_GenTk",
    'description': "Pt of Reconstructed Track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/GeneralProperties/Chi2oNDF_GenTk",
    'description': "Chi Square per DoF -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/GeneralProperties/TrackPhi_ImpactPoint_GenTk",
    'description': "Phi distribution of Reconstructed Tracks -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/GeneralProperties/TrackEta_ImpactPoint_GenTk",
    'description': " Eta distribution of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }}])
shifttrackinglayout(dqmitems, "02b - Total Hits Strip and Pixel (pp collisions)",
 [{ 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/HitProperties/Strip/NumberOfRecHitsPerTrack_Strip_GenTk",
    'description': "Number of Strip valid RecHits in each Track", 'draw': { 'withref': "yes" }},
{ 'path': "Tracking/TrackParameters/highPurityTracks/pt_1/HitProperties/Pixel/NumberOfRecHitsPerTrack_Pixel_GenTk",
    'description': "Number of Pixel valid RecHits in each Track", 'draw': { 'withref': "yes" }}])
shifttrackinglayout(dqmitems, "03 - Tracks (Cosmic Tracking)",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfTracks_CKFTk",
    'description': "Number of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/HitProperties/NumberOfRecHitsPerTrack_CKFTk",
    'description': "Number of RecHits per Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPt_CKFTk",
    'description': "Pt of Reconstructed Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/Chi2oNDF_CKFTk",
    'description': "Chi Sqare per DoF  -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPhi_CKFTk",
    'description': "Phi distribution of Reconstructed Tracks -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackEta_CKFTk",
    'description': " Eta distribution of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }}])
shifttrackinglayout(dqmitems, "04 - Tracks (HI run)",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfTracks_HeavyIonTk",
    'description': "Number of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/HitProperties/NumberOfRecHitsPerTrack_HeavyIonTk",
    'description': "Number of RecHits per Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPt_ImpactPoint_HeavyIonTk",
    'description': "Pt of Reconstructed Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/Chi2oNDF_HeavyIonTk",
    'description': "Chi Sqare per DoF  -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPhi_ImpactPoint_HeavyIonTk",
    'description': "Phi distribution of Reconstructed Tracks -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackEta_ImpactPoint_HeavyIonTk",
    'description': " Eta distribution of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
shifttrackinglayout(dqmitems, "05 - Number of Seeds (pp collisions)",
 [{ 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_initialStepSeeds_initialStep",
    'description': "Number of Seed in tracking iteration 0 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_lowPtTripletStepSeeds_lowPtTripletStep",
    'description': "Number of Seed in tracking iteration 1 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_pixelPairStepSeeds_pixelPairStep",
    'description': "Number of Seed in tracking iteration 2 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_detachedTripletStepSeeds_detachedTripletStep",
    'description': "Number of Seed in tracking iteration 3 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_mixedTripletStepSeeds_mixedTripletStep",
    'description': "Number of Seed in tracking iteration 4 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
    { 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_pixelLessStepSeeds_pixelLessStep",
    'description': "Number of Seed in tracking iteration 5 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
    { 'path': "Tracking/TrackParameters/generalTracks/TrackBuilding/NumberOfSeeds_tobTecStepSeeds_tobTecStep",
    'description': "Number of Seed in tracking iteration 6 (no entry: ERROR) - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
shifttrackinglayout(dqmitems, "06 - Primary Vertices",
 [{ 'path': "OfflinePV/offlinePrimaryVertices/vtxNbr",
    'description': "Number reconstructed primary vertices - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftOfflineTracking>DQMShiftOfflineTracking</a> ", 'draw': { 'withref': "yes" }}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
