#
# Last update: new version for python
#
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("cluTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelClusters'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/ONE_BEAM/BSCskim_123065.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/ONE_BEAM/BSCskim_123151_Express.root')
#    fileNames =  cms.untracked.vstring('file:/afs/cern.ch/user/d/dkotlins/scratch0/SCRATCH/BSCskim_123151_Express.root')
# 123977 - minbias
#    fileNames = cms.untracked.vstring(                          
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/34CFFC67-0DE6-DE11-B867-003048D37456.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/4A50153D-10E6-DE11-AC44-0019B9F709A4.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/7A13C259-12E6-DE11-BAC6-0019B9F72CE5.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/84ED794A-17E6-DE11-9F0A-000423D9890C.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/BADD6AEC-26E6-DE11-8751-0030487A3232.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/BEE60480-0FE6-DE11-8519-001D09F2426D.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/D0F8D32C-1AE6-DE11-AD0E-0016177CA778.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/E606D9B4-18E6-DE11-8615-000423D60FF6.root')
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/LHC09/RECO/MinBias/123977/FE4AC471-14E6-DE11-90C2-001D09F241F0.root')
#     fileNames = cms.untracked.vstring(                          
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/977/34CFFC67-0DE6-DE11-B867-003048D37456.root"
#     )
# 123977 - ZeroBias
#    fileNames = cms.untracked.vstring(                          
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/1EBCF353-12E6-DE11-9A18-000423D9997E.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/3CA682CA-0EE6-DE11-8BC2-001D09F24DDF.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/569FAA2E-15E6-DE11-84C0-000423D98834.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/92B8CAFF-19E6-DE11-885E-000423D98B6C.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/96050F4A-17E6-DE11-A936-000423D9989E.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/A8473FF4-0BE6-DE11-A237-0016177CA7A0.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/BC3028F0-10E6-DE11-A6AB-001617C3B76E.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/977/DAA5FDE9-26E6-DE11-BD76-003048D37456.root"
#    )
# 123615 - zerobias
#    fileNames = cms.untracked.vstring(                          
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/443EF125-BCE2-DE11-BE94-000423D98EC8.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/460C2FED-B9E2-DE11-A0E8-0019B9F709A4.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/52309765-BDE2-DE11-AF9E-003048D2C1C4.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/52DBB9B1-BCE2-DE11-963C-003048D37560.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/663A3AD1-D1E2-DE11-87DB-003048D373AE.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/78C267D2-D1E2-DE11-9DF6-003048D375AA.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/7CB3747E-F1E2-DE11-837B-0030487A18F2.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/A259B0A3-BAE2-DE11-8682-0019B9F72CE5.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/A2D7364A-CEE2-DE11-8CD2-001617E30D12.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/CCD5D44B-BBE2-DE11-BA3D-001D09F290CE.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/D41A595F-BDE2-DE11-9823-00304879FA4A.root",
#    "/store/data/BeamCommissioning09/ZeroBias/RECO/v2/000/123/615/ECF6ADEA-C0E2-DE11-9D77-001D09F2532F.root"
#    )
# 123615 - minbias
#    fileNames = cms.untracked.vstring(                          
##    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/04206717-D1E2-DE11-B872-003048D37514.root" EMPTY
#    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/1E23FCDE-C0E2-DE11-BE56-000423D99BF2.root",
#    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/2207C752-C2E2-DE11-9FEB-003048D3756A.root",
#    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/3EBBFB26-BCE2-DE11-81F2-001D09F2516D.root",
##    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/54518C16-D1E2-DE11-8146-001617C3B79A.root" EMPTY
#    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/86D2D694-CDE2-DE11-B5D1-003048D37514.root",
#    "/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/615/BA9F53DE-BFE2-DE11-9D01-001D09F24353.root"
#    )
# 130269 - cosmics
    fileNames = cms.untracked.vstring(                          
    "/store/data/Commissioning10/Cosmics/RECO/v3/000/130/269/0A70AAC2-A929-DF11-AD28-0019DB29C620.root" 
# 0EC0015F-A829-DF11-A0C5-0019DB29C620.root
# 14860A2F-A429-DF11-8E15-001D09F282F5.root
# 1600E75B-CE29-DF11-B9CF-001D09F24E39.root
# 1AB4EBA0-B329-DF11-99BD-000423D985B0.root
# 1C9B79B4-B529-DF11-A568-000423D99264.root
# 267D9261-9A29-DF11-B128-003048673374.root
# 2A317CA8-A729-DF11-B903-001617E30CC2.root
# 2C4C70CB-C829-DF11-A96B-000423D99394.root
# 3C0440B5-B529-DF11-9A5D-000423D98868.root
# 3E5DC5CA-B029-DF11-9162-000423D99F3E.root
# 42838815-B029-DF11-A867-000423D99E46.root
# 4C09228A-9E29-DF11-A7C1-0019B9F730D2.root
# 4C8F6EB4-B529-DF11-8066-000423D98E30.root
# 4E27F842-9F29-DF11-AFD7-003048D3750A.root
# 500E62DA-BE29-DF11-930C-000423D6006E.root
# 503C2867-B629-DF11-B6A2-0030487CD716.root
# 50CF4203-AE29-DF11-BDB4-001D09F2AF1E.root
# 527FE9D2-BC29-DF11-AFA8-001D09F29597.root
# 588BF948-AC29-DF11-9BBA-000423D99F1E.root
# 5A062DCB-B029-DF11-95DC-000423D94E1C.root
# 5A786776-BD29-DF11-9CAC-001D09F248F8.root
# 5C105F30-A429-DF11-ADA5-001D09F231C9.root
# 5C2B83C5-A229-DF11-A40D-001D09F28755.root
# 7CE80B5C-CE29-DF11-873C-001D09F252DA.root
# 7E91FE70-A329-DF11-AF6C-000423D98950.root
# 7EA95F42-9F29-DF11-8C64-003048D37580.root
# 8014C545-A629-DF11-A5DF-000423D6B444.root
# 8A87EA47-C029-DF11-97E1-000423D98B5C.root
# 94877F88-B829-DF11-B766-000423D98A44.root
# 9E488769-AF29-DF11-8768-001D09F25109.root
# 9E725948-C029-DF11-AD79-000423D98868.root
# A208D52D-9D29-DF11-BC44-001D09F2B2CF.root
# A632DF4A-AC29-DF11-862A-0030487CD6DA.root
#AEEF7DD2-CA29-DF11-98A4-0030487D0D3A.root
#B42AFACA-B029-DF11-BD72-000423D8FA38.root
#B477CBC2-A929-DF11-AF53-001617C3B6CC.root
#BAE72AA9-A729-DF11-9713-001617C3B6DE.root
#BC3CC88A-B829-DF11-A6F3-000423D987E0.root
#BE863D91-A529-DF11-85D2-001D09F248F8.root
#C038AC99-AC29-DF11-AA79-001617C3B6DE.root
#C0D07EEF-B929-DF11-B482-0030487A18D8.root
#D0DE692B-9D29-DF11-B3A9-0019B9F4A1D7.root
#DA55CB66-B629-DF11-8912-0030487C6090.root
#DC97EB1B-CA29-DF11-A0D0-0019B9F70607.root
#E48D956A-9C29-DF11-9CA1-000423D9939C.root
#EA0AB190-A529-DF11-BBCA-001D09F291D7.root
#EE558AC3-A929-DF11-B2B3-001617C3B6DE.root
#F0A85977-BD29-DF11-A245-001D09F2983F.root
#F297B2B3-A529-DF11-9A27-001D09F25109.root
#F2EA6233-BE29-DF11-9C44-001D09F251CC.root
#F63602A2-BA29-DF11-A559-0030487CD716.root
#FA939598-AC29-DF11-995E-0019DB29C620.root

    )

)

# 124023
#-rw-r--r--   1 cmsprod  zh                 3296039642 Dec 12 20:40 0A849C4E-55E7-DE11-9683-000423D98DD4.root
#-rw-r--r--   1 cmsprod  zh                 2283707725 Dec 12 18:44 1E35A03B-45E7-DE11-9D15-001D09F290BF.root
#-rw-r--r--   1 cmsprod  zh                 3653451518 Dec 12 17:00 4257A89D-35E7-DE11-8057-000423D990CC.root
#-rw-r--r--   1 cmsprod  zh                 3649110380 Dec 12 17:10 4C816E01-37E7-DE11-A5C0-000423D98E6C.root
#-rw-r--r--   1 cmsprod  zh                 3554817142 Dec 12 18:19 6A36499F-40E7-DE11-9F8B-000423D991F0.root
#-rw-r--r--   1 cmsprod  zh                 2713367490 Dec 12 18:20 700C5E08-41E7-DE11-8033-001D09F24763.root
#-rw-r--r--   1 cmsprod  zh                 3595132716 Dec 12 17:31 7275C4D9-39E7-DE11-9937-000423D99996.root
#-rw-r--r--   1 cmsprod  zh                 2859171283 Dec 12 17:03 7A4E4D4B-36E7-DE11-BA4A-000423D9517C.root
#-rw-r--r--   1 cmsprod  zh                 3603098819 Dec 12 18:11 86980638-3FE7-DE11-988B-001D09F24EE3.root
#-rw-r--r--   1 cmsprod  zh                 3594478970 Dec 12 17:46 A227E50B-3CE7-DE11-90A5-000423D98920.root
#-rw-r--r--   1 cmsprod  zh                 1977080462 Dec 12 18:58 B429A464-47E7-DE11-9243-003048D373AE.root
#-rw-r--r--   1 cmsprod  zh                 2698634488 Dec 12 17:17 C0142781-38E7-DE11-95E7-000423D9A212.root
#-rw-r--r--   1 cmsprod  zh                 2909882917 Dec 12 19:13 C6B56288-49E7-DE11-A8D4-0019B9F581C9.root
#-rw-r--r--   1 cmsprod  zh                 2662056475 Dec 12 18:14 C8709623-40E7-DE11-999C-001D09F291D7.root
#-rw-r--r--   1 cmsprod  zh                 2736046732 Dec 12 16:52 D224DBEA-34E7-DE11-8C24-003048D3750A.root
#-rw-r--r--   1 cmsprod  zh                 3575876422 Dec 12 17:30 DC719AD9-39E7-DE11-B4ED-000423D94E70.root
#-rw-r--r--   1 cmsprod  zh                 2748178050 Dec 12 16:47 E24DE737-34E7-DE11-8FDE-000423D98DC4.root
#-rw-r--r--   1 cmsprod  zh                 2676049829 Dec 12 18:24 E426C2AB-41E7-DE11-AB31-001D09F24DDF.root
#-rw-r--r--   1 cmsprod  zh                 3382932019 Dec 12 18:17 E65669D8-40E7-DE11-BEC9-000423D987E0.root
#-rw-r--r--   1 cmsprod  zh                 3579589925 Dec 12 17:50 E6E66BA9-3CE7-DE11-AD6B-000423D6B444.root
#-rw-r--r--   1 cmsprod  zh                 2773090758 Dec 12 17:41 F8A7BE48-3BE7-DE11-B56F-001D09F24F1F.root

# 124230
#-rw-r--r--   1 cmsprod  zh                 2517373661 Dec 15 04:52 041A79BD-2CE9-DE11-8FF4-001D09F2983F.root
#-rw-r--r--   1 cmsprod  zh                 2871964160 Dec 15 07:05 12256032-3FE9-DE11-AC38-000423D987E0.root
#-rw-r--r--   1 cmsprod  zh                 2018717498 Dec 15 04:29 18E42F28-28E9-DE11-8195-001D09F28755.root
#-rw-r--r--   1 cmsprod  zh                 3293659087 Dec 15 08:21 24198CD2-49E9-DE11-B1EF-001D09F29849.root
#-rw-r--r--   1 cmsprod  zh                 2845092284 Dec 15 05:14 247144EC-2EE9-DE11-BB7F-001617E30F48.root
#-rw-r--r--   1 cmsprod  zh                 1896996605 Dec 15 12:36 24E60FC7-6DE9-DE11-BDB3-003048D3750A.root
#-rw-r--r--   1 cmsprod  zh                 3397420656 Dec 15 12:39 3228B1CB-6DE9-DE11-B3FC-003048D37456.root
#-rw-r--r--   1 cmsprod  zh                 3587303679 Dec 15 04:17 36168891-25E9-DE11-B20C-001D09F248F8.root
#-rw-r--r--   1 cmsprod  zh                 2174586975 Dec 15 04:01 3C099FD7-24E9-DE11-B71C-001D09F29169.root
#-rw-r--r--   1 cmsprod  zh                 2654232285 Dec 15 07:29 3C523ABC-42E9-DE11-B88E-003048D2C108.root
#-rw-r--r--   1 cmsprod  zh                 2357337784 Dec 15 04:45 40D67DF4-2AE9-DE11-AA35-0030487A322E.root
#-rw-r--r--   1 cmsprod  zh                 3375958984 Dec 15 07:27 44EA3B1B-42E9-DE11-A16B-001617C3B6CE.root
#-rw-r--r--   1 cmsprod  zh                 3211891056 Dec 15 06:04 46B7ADAE-36E9-DE11-AFB3-000423D9890C.root
#-rw-r--r--   1 cmsprod  zh                 2644980341 Dec 15 07:04 4A909E32-3FE9-DE11-B5E2-001617C3B76A.root
#-rw-r--r--   1 cmsprod  zh                 3250808332 Dec 15 06:10 50CA9C60-37E9-DE11-AED8-000423D944F0.root
#-rw-r--r--   1 cmsprod  zh                 2365563905 Dec 15 04:38 5EB0EBD6-28E9-DE11-871B-001D09F2305C.root
#-rw-r--r--   1 cmsprod  zh                 2262058377 Dec 15 05:19 62C0D850-30E9-DE11-9B47-001D09F2A690.root
#-rw-r--r--   1 cmsprod  zh                 3058745545 Dec 15 08:16 6E3BE828-49E9-DE11-94E1-003048D37560.root
#-rw-r--r--   1 cmsprod  zh                 2991857584 Dec 15 07:15 7272DA9A-40E9-DE11-BD4E-0030487A3C9A.root
#-rw-r--r--   1 cmsprod  zh                 3340194475 Dec 15 08:56 7CE302BC-4EE9-DE11-8FBF-001D09F253FC.root
#-rw-r--r--   1 cmsprod  zh                 3059391151 Dec 15 07:45 7E2258D5-44E9-DE11-884A-000423D99E46.root
#-rw-r--r--   1 cmsprod  zh                 2425913034 Dec 15 07:19 8C6A5E4F-41E9-DE11-BBEB-000423D6006E.root
#-rw-r--r--   1 cmsprod  zh                 3084894665 Dec 15 08:10 9E163E69-48E9-DE11-81FB-001617E30D12.root
#-rw-r--r--   1 cmsprod  zh                 3541182575 Dec 15 05:09 A204987E-2DE9-DE11-894B-001D09F29533.root
#-rw-r--r--   1 cmsprod  zh                 3149048503 Dec 15 05:59 A69A74F5-35E9-DE11-9233-000423D9890C.root
#-rw-r--r--   1 cmsprod  zh                 3405259630 Dec 15 08:36 A81DEAEE-4BE9-DE11-961B-001D09F28F1B.root
#-rw-r--r--   1 cmsprod  zh                 3538381271 Dec 15 04:24 AE2480F3-26E9-DE11-A4A4-0019B9F707D8.root
#-rw-r--r--   1 cmsprod  zh                 3032878189 Dec 15 08:16 B2376327-49E9-DE11-B6A1-000423D98EC4.root
#-rw-r--r--   1 cmsprod  zh                 3013527210 Dec 15 12:38 BA5AD2C7-6DE9-DE11-BCA6-003048D373AE.root
#-rw-r--r--   1 cmsprod  zh                 2372302929 Dec 15 04:54 BA5E3F16-2CE9-DE11-BA01-001D09F242EA.root
#-rw-r--r--   1 cmsprod  zh                 3042011262 Dec 15 08:16 BC45CF2A-49E9-DE11-8D00-000423D99F1E.root
#-rw-r--r--   1 cmsprod  zh                 2548861162 Dec 15 07:36 C427A06B-43E9-DE11-B7FE-001617C3B65A.root
#-rw-r--r--   1 cmsprod  zh                 3028915477 Dec 15 05:18 C6A7E057-30E9-DE11-92F2-001D09F244DE.root
#-rw-r--r--   1 cmsprod  zh                 3221672218 Dec 15 07:25 CAF48103-42E9-DE11-BA13-003048678098.root
#-rw-r--r--   1 cmsprod  zh                 2976324673 Dec 15 05:49 E8A5B59A-34E9-DE11-A3DC-001D09F23944.root
#-rw-r--r--   1 cmsprod  zh                 3080506547 Dec 15 07:19 F80FDE54-41E9-DE11-934F-003048678098.root
#-rw-r--r--   1 cmsprod  zh                 3272859729 Dec 15 07:25 F8D9EA04-42E9-DE11-BFD2-003048D2C108.root
#-rw-r--r--   1 cmsprod  zh                 2383250765 Dec 15 04:02 FA7D91DA-24E9-DE11-BA6E-001D09F2545B.root





process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
process.GlobalTag.globaltag = 'GR10_P_V3::All'
# OK for 2009 LHC data
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

process.analysis = cms.EDAnalyzer("ReadPixClusters",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("siPixelClusters"),
)

process.p = cms.Path(process.analysis)



