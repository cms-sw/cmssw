def trackinglayout(i, p, *rows): i["Tracking/Layouts/" + p] = DQMItem(layout=rows)

trackinglayout(dqmitems, "01 - Tracking ReportSummary",
 [{ 'path': "Tracking/EventInfo/reportSummaryMap",
    'description': " Quality Test results plotted for Tracking parameters : Chi2, TrackRate, #of Hits in Track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "no" }}])
trackinglayout(dqmitems, "02 - Tracks (pp collisions)",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfGoodTracks_GenTk",
    'description': "Number of Reconstructed Tracks with high purity selection and pt > 1 GeV - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/HitProperties/GoodTrackNumberOfRecHitsPerTrack_GenTk",
    'description': "Number of RecHits per Track with high purity selection and pt > 1 GeV  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/GoodTrackPt_ImpactPoint_GenTk",
    'description': "Pt of Reconstructed Track with high purity selection and pt > 1 GeV  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/GoodTrackChi2oNDF_GenTk",
    'description': "Chi Square per DoF with high purity selection and pt > 1 GeV  -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/GoodTrackPhi_ImpactPoint_GenTk",
    'description': "Phi distribution of Reconstructed Tracks with high purity selection and pt > 1 GeV -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/GoodTrackEta_ImpactPoint_GenTk",
    'description': " Eta distribution of Reconstructed Tracks with high purity selection and pt > 1 GeV - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])
trackinglayout(dqmitems, "03 - Tracks (HI collisions)",
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

trackinglayout(dqmitems, "04 - Tracks (Cosmic Tracking)",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfTracks_CKFTk",
    'description': "Number of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/HitProperties/NumberOfRecHitsPerTrack_CKFTk",
    'description': "Number of RecHits per Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPt_CKFTk",
    'description': "Pt of Reconstructed Track  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/Chi2oNDF_CKFTk",
    'description': "Chi Sqare per DoF  -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPhi_CKFTk",
    'description': "Phi distribution of Reconstructed Tracks -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackEta_CKFTk",
    'description': " Eta distribution of Reconstructed Tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/SiStripOfflineDQMInstructions>SiStripOfflineDQMInstructions</a> ", 'draw': { 'withref': "yes" }}])

#trackinglayout(dqmitems, "04 - Fraction of Tracks vs LS",
#  [ { 'path': "Tracking/TrackParameters/GeneralProperties/TracksFractionVsLS_HeavyIonTk",
#      'description': "Fraction of tracks versus Lumi Section",
#      'draw': { 'withref': "yes" }}])

#trackinglayout(dqmitems, "05 - Number of rec hits per track vs LS",
#  [ { 'path': "Tracking/TrackParameters/GeneralProperties/TracksNumberOfRecHitsPerTrackVsLS_HeavyIonTk",
#      'description': "Number of rec hits per track vs LS",
#      'draw': { 'withref': "yes" }}])

#trackinglayout(dqmitems, "06 - Offline PV",
#  [{ 'path': "OfflinePV/offlinePrimaryVertices/tagDiffX",
#     'description': "Difference between PV and beamspot in x-direction"},
#   { 'path': "OfflinePV/offlinePrimaryVertices/tagDiffY",
#     'description': "Difference between PV and beamspot in y-direction"}
#    ])

trackinglayout(dqmitems, "07 - Beam Monitor",
  [{ 'path': "AlcaBeamMonitor/Validation/hxLumibased PrimaryVertex-DataBase",
     'description': ""},
   { 'path': "AlcaBeamMonitor/Validation/hyLumibased PrimaryVertex-DataBase",
     'description': ""},
   { 'path': "AlcaBeamMonitor/Validation/hzLumibased PrimaryVertex-DataBase",
     'description': ""}
   ],
   [
    { 'path': "AlcaBeamMonitor/Debug/hsigmaXLumibased PrimaryVertex-DataBase fit",
     'description': ""},
    { 'path': "AlcaBeamMonitor/Debug/hsigmaYLumibased PrimaryVertex-DataBase fit",
     'description': ""},
    { 'path': "AlcaBeamMonitor/Debug/hsigmaZLumibased PrimaryVertex-DataBase fit",
     'description': ""},
   ])
