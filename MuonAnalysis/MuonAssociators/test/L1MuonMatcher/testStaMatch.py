from UserCode.GPetrucc.muons.skimWithDecays import cms,process

##    ___                   _     ____        _        
##   |_ _|_ __  _ __  _   _| |_  |  _ \  __ _| |_ __ _ 
##    | || '_ \| '_ \| | | | __| | | | |/ _` | __/ _` |
##    | || | | | |_) | |_| | |_  | |_| | (_| | || (_| |
##   |___|_| |_| .__/ \__,_|\__| |____/ \__,_|\__\__,_|
##             |_|                                     
##
process.source.fileNames = cms.untracked.vstring(
       #'/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0003/94D3C1FC-F8C4-DD11-8A0D-0019DB29C5FC.root',
       '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0004/585A91F3-41C5-DD11-AF3F-001617C3B5D8.root',
       '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0004/88E0A226-42C5-DD11-BF2B-000423D98800.root',
       '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0004/BAEC2746-E6C5-DD11-9602-000423D944FC.root',
       '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0004/CA75BC21-42C5-DD11-AB16-000423D98634.root',
       '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0004/DC13E4EF-41C5-DD11-9C4C-001617E30D0A.root',
       '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-RECO/IDEAL_V9_v1/0004/F20B67F5-41C5-DD11-B733-001617E30D06.root' 
)
## process.source.secondaryFileNames = cms.untracked.vstring(
##         #'/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0003/6E1DA183-F6C4-DD11-BC9D-001617DBD5AC.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/0E154CEA-42C5-DD11-827D-001617E30CE8.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/12EE35A1-42C5-DD11-87DD-001617E30D12.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/288D5185-42C5-DD11-B53F-000423D6CA42.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/380462E8-42C5-DD11-BCE2-001617E30CA4.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/5C3701B6-42C5-DD11-8AE3-000423D992A4.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/5C3BA77F-42C5-DD11-BFFA-001617DBD288.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/7268217C-42C5-DD11-924D-001617E30F4C.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/86DD8C82-42C5-DD11-90A9-000423D985E4.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/8AF5B07E-42C5-DD11-A794-001617E30D40.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/A0E2C07F-42C5-DD11-A2C0-000423D6C8EE.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/A4C6E590-42C5-DD11-B514-000423D9853C.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/A81400E3-42C5-DD11-ADA4-001617C3B778.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/A8A2D07A-42C5-DD11-8185-00161757BF42.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/AE33E87B-42C5-DD11-93B3-001617C3B5F4.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/B289DFA2-42C5-DD11-9AC7-001617C3B6DC.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/B6D56BB7-42C5-DD11-947E-000423D6CAF2.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/B6E1EEB4-42C5-DD11-8018-001617C3B70E.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/BA2309A5-42C5-DD11-A63A-001617DBD224.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/CA4035EC-42C5-DD11-BC6A-001617C3B65A.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/CCA60DB1-42C5-DD11-8EB5-001617C3B6CE.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/D2E39685-42C5-DD11-83AE-001617C3B6CC.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/DAEFE955-42C5-DD11-9B11-001617E30F50.root',
##         '/store/relval/CMSSW_2_2_1/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0004/F8B4DE55-42C5-DD11-82E6-001617E30F58.root'
## )
process.maxEvents.input = -1
process.maxEvents.output = cms.untracked.int32(20)

##    ____            _   _____     _                          __              __  __                       
##   |  _ \ ___  __ _| | |_   _| __(_) __ _  __ _  ___ _ __   / _| ___  _ __  |  \/  |_   _  ___  _ __  ___ 
##   | |_) / _ \/ _` | |   | || '__| |/ _` |/ _` |/ _ \ '__| | |_ / _ \| '__| | |\/| | | | |/ _ \| '_ \/ __|
##   |  _ <  __/ (_| | |   | || |  | | (_| | (_| |  __/ |    |  _| (_) | |    | |  | | |_| | (_) | | | \__ \
##   |_| \_\___|\__,_|_|   |_||_|  |_|\__, |\__, |\___|_|    |_|  \___/|_|    |_|  |_|\__,_|\___/|_| |_|___/
##                                    |___/ |___/                                                           
##
process.muonTrigMatchHLTMu3.maxDeltaR = 0.7
process.muonTrigMatchHLTMu3.maxDPtRel = 10.0
process.muonTrigMatchHLTMu3.resolveAmbiguities = True
process.muonTrigMatchHLTMu3.resolveByMatchQuality = True

process.muonL1MatchSt = process.muonL1Match.clone(
    useTrack = cms.string('muon'),
    useState = cms.string('innermost'),
    setL1Label = cms.string('l1st'),
    setPropLabel = cms.string('propst')
)
process.patDefaultSequence.replace(process.muonL1Match, process.muonL1Match + process.muonL1MatchSt)
process.allLayer1Muons.trigPrimMatch  += [ cms.InputTag("muonL1MatchSt"),  cms.InputTag("muonL1MatchSt","propagatedReco") ]

process.allLayer1Muons.userData.userInts.src = cms.VInputTag(
    cms.InputTag("muonL1Match", "bx"),
    cms.InputTag("muonL1Match", "quality"),
    cms.InputTag("muonL1Match", "isolated"),
    cms.InputTag("muonL1MatchSt", "bx"),
    cms.InputTag("muonL1MatchSt", "quality"),
    cms.InputTag("muonL1MatchSt", "isolated"),
)
process.allLayer1Muons.userData.userFloats.src = cms.VInputTag(
    cms.InputTag("muonL1Match", "deltaR"),
    cms.InputTag("muonL1MatchSt", "deltaR"),
)


process.fishyMuons = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("selectedLayer1Muons"),
    cut = cms.string("standAloneMuon.isNonnull && !triggerMatchesByFilter('hltSingleMuPrescale3L3PreFiltered').empty() && triggerMatchesByFilter('l1st').empty()"),
    filter = cms.bool(True),
)

process.p *= process.fishyMuons

##     ___        _               _   
##    / _ \ _   _| |_ _ __  _   _| |_ 
##   | | | | | | | __| '_ \| | | | __|
##   | |_| | |_| | |_| |_) | |_| | |_ 
##    \___/ \__,_|\__| .__/ \__,_|\__|
##                   |_|              
##
process.allLayer1Muons.embedTrack          = True
process.allLayer1Muons.embedCombinedMuon   = True
process.allLayer1Muons.embedStandAloneMuon = True
process.out.outputCommands = cms.untracked.vstring(
    "drop *",
    'keep *_standAloneMuons_*_*', 'keep *_globalMuons_*_*',
    "keep *_selectedLayer1Muons_*_*",
    'keep l1extraL1MuonParticles_hltL1extraParticles_*_*',
    'keep *_patHLTMu3_*_*'
)
process.out.fileName = cms.untracked.string("/tmp/gpetrucc/testL1Sta.root")

