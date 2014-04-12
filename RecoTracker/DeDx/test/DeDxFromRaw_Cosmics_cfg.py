import FWCore.ParameterSet.Config as cms

process = cms.Process("DEDX")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(25000),
    fileNames  = cms.untracked.vstring(
### Run 60302
#        '/store/data/Commissioning08/Cosmics/RECO/EW35_3T_v1/000/060/302/2E653B32-1277-DD11-A8F2-0030487A3232.root',          #Run60302
#        '/store/data/Commissioning08/Cosmics/RECO/EW35_3T_v1/000/060/302/749B859D-2C76-DD11-9837-001D09F24498.root',          #Run60302
#        '/store/data/Commissioning08/Cosmics/RECO/EW35_3T_v1/000/060/302/8E5B9DA9-2E76-DD11-90BB-0019B9F70607.root'           #Run60302
### Run 62966
        '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/02859E73-7684-DD11-B423-000423D6C8E6.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/0288ECB3-7684-DD11-AC06-001617C3B5F4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/02940ECF-7884-DD11-9BDE-001617C3B6E8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/04289E48-6684-DD11-BE00-000423D986C4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/04D43F2D-6B84-DD11-9837-000423D944F0.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/0663FAC9-7384-DD11-952C-000423D94700.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/0820A8E6-6C84-DD11-84D0-001617C3B76A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/08C0E279-7284-DD11-AA7C-000423D98E30.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/0AA89A75-6684-DD11-A105-000423D944FC.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/0ABD64CB-6884-DD11-ABC3-000423D94494.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/0CF5ADF2-6384-DD11-B577-0019DB29C5FC.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/10F5D3F3-7484-DD11-AF5B-001617DBD556.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/10FE209C-7184-DD11-8C0A-001617E30CA4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/1AD58EFC-7B84-DD11-8031-0019DB29C614.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/1CAA6E09-6884-DD11-88B6-000423D8F63C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/1CD824A6-7384-DD11-847E-000423D98DB4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/1EF05A16-6D84-DD11-BF2B-000423D9A212.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/22984393-7984-DD11-8EB5-001617E30F48.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/22CF4EA0-7584-DD11-A511-000423D6B42C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/287F6730-6C84-DD11-8982-001617DBD288.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/28E6C767-6384-DD11-BABA-001617C3B79A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/2A24FA10-6B84-DD11-B643-000423D98844.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/2E873AB4-7084-DD11-9EA9-001617E30CE8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/2EB44549-7184-DD11-AB85-000423D99896.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/3627FA38-7084-DD11-A9D7-001617C3B6E2.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/36BD3A57-6784-DD11-97DA-000423D6B358.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/36DF4B2D-7284-DD11-AC06-000423D98F98.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/384AEF4B-6684-DD11-9508-001617E30CC8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/3AFE9B0D-6F84-DD11-AEAC-000423D98DC4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/403A347A-7D84-DD11-ADD2-0016177CA778.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/40F103F8-6484-DD11-8B5C-000423D6A6F4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/4201A618-6384-DD11-BCE9-001617E30E28.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/423C700A-6C84-DD11-A8D8-001617C3B70E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/4292E820-6E84-DD11-B597-001617E30D40.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/462E1067-6F84-DD11-A09C-000423D94A04.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/4CE876CB-6B84-DD11-A81D-000423D94A20.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/4E06B006-6E84-DD11-A316-000423D98AF0.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/505B4F5C-7784-DD11-AEC4-000423D6CAF2.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/522104D1-6284-DD11-8234-001617E30D4A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/52DCC530-7284-DD11-B976-001617DC1F70.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/56FCC408-6884-DD11-B4C9-000423D99AAA.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/58F3AE50-7984-DD11-9BE9-001617DBD332.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/5A4E353B-7484-DD11-B710-000423D9970C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/5C36036C-7C84-DD11-844B-001617E30F58.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/603F65C6-6284-DD11-BAFE-001617E30D38.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/607C1852-7384-DD11-8C8B-000423D99394.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/60CF9E27-7484-DD11-9925-001617C3B710.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/624761B5-6884-DD11-B098-001617E30CA4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/62C9526C-7084-DD11-B0E6-000423D33970.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/64ADDAE1-6E84-DD11-ACD4-000423D9517C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/660C8760-7E84-DD11-8103-000423D6B5C4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/662E2DA7-6584-DD11-9FE3-000423D99660.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/666EF690-7C84-DD11-B989-000423D98B28.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/66FE013E-7F84-DD11-B031-001617C3B706.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/6A08BE67-6884-DD11-822F-001617DBCF6A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/6E8650B8-6784-DD11-8150-000423D992DC.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/72AF5CFD-7084-DD11-953E-000423D987FC.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/766F94F0-7784-DD11-908C-001617C3B6FE.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/76FD0515-6384-DD11-8CF8-000423D98F98.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/7A19155C-7E84-DD11-8BD9-001617C3B73A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/7A422BEE-6B84-DD11-99B7-000423D985B0.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/7AD3DCB4-6784-DD11-AB00-000423D996C8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/7AEC3D06-6F84-DD11-AFD5-000423D94AA8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/7E1850C7-6184-DD11-B1C5-001617DBD224.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/7E87EC73-7C84-DD11-BD26-000423D99160.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/807F4531-7B84-DD11-955C-001617DBD230.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/80BF05A3-7784-DD11-A449-001617E30F4C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/8A1BD799-6784-DD11-ACEB-000423D99E46.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/8E90405D-6C84-DD11-91AF-000423D98B6C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/92D2C0A5-6284-DD11-B7A3-001617E30D38.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/94C98133-6B84-DD11-A822-000423D99AA2.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/9648156E-7184-DD11-B65C-000423D98C20.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/968FF9AD-6A84-DD11-B9B3-001617C3B6CC.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/98183930-6484-DD11-A45B-000423D98800.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/9873FAA3-6584-DD11-A84A-000423D94524.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/9A5609A3-7784-DD11-B640-0019DB29C620.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/9ACFB350-7984-DD11-9F38-001617C3B6C6.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/9C4EDAD4-7084-DD11-96F2-000423D9989E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/9EB8414E-7484-DD11-BD55-001617C3B6DE.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/A286EC4D-7A84-DD11-897C-001617C3B778.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/A6DA38D1-7084-DD11-8DF5-000423D99F3E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/A8D11CFC-6D84-DD11-8F48-000423D98E54.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/AAA3D426-6784-DD11-86DF-001617C3B65A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/AEA9935C-6884-DD11-B724-000423D9870C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/B00CD9EF-6184-DD11-BE6D-001617E30E2C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/B00D7F3C-7884-DD11-8CB8-001617DBD5AC.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/B26E6DAC-7584-DD11-91F9-001617C3B78C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/B48AA648-7384-DD11-A280-001617C3B76E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/B4EC2E1D-6984-DD11-9C9D-000423D174FE.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/BC8E7772-6C84-DD11-9423-00161757BF42.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/BEA27E68-6D84-DD11-B8EA-000423D94908.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/C090E88C-6784-DD11-ACE2-001617E30D00.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/C099710B-7584-DD11-BD5F-000423D992A4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/C0D12197-7484-DD11-A39B-001617C3B69C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/C4DA3B49-7384-DD11-BAF5-000423D94AA8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/C6FE498B-7584-DD11-BE9A-001617DBCF90.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/CA421835-7484-DD11-850E-000423D9863C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/CABC564A-7984-DD11-AF7E-001617E30F56.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/CC869C73-6684-DD11-AE92-000423D98BC4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/CE5EA015-7884-DD11-AADA-001617C3B64C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/D0EA5EFC-7B84-DD11-A9F3-001617E30D12.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/D2998715-7A84-DD11-8E6B-000423D6CA6E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/D2F5DA95-6D84-DD11-ADCF-000423D985E4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/DAEF6241-6A84-DD11-BFA2-000423D98A44.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/DC0954EC-6B84-DD11-A06D-000423D99B3E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/DC2E9F92-7384-DD11-B26C-000423D99F3E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/DEF25FD6-7684-DD11-8A37-0019DB2F3F9B.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/E03C73FE-7B84-DD11-835A-001617E30CD4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/E2ED7F85-6284-DD11-8302-001617C3B6CE.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/E458657A-8484-DD11-A767-001617C3B654.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/E48A7833-7284-DD11-9D81-000423D99F1E.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/E67C4A89-6584-DD11-9E87-000423DD2F34.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/E8A554C9-7784-DD11-93C0-001617DF785A.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/EAABFC82-7284-DD11-A1FC-000423D98634.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/EE7CEAC6-6B84-DD11-B551-000423D98E6C.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/F236442D-7A84-DD11-A0A3-000423D98750.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/F416F436-7084-DD11-A961-001617DBD316.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/F4273102-7A84-DD11-999E-001617C3B5E4.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/F60A7FDB-6D84-DD11-A5C8-001617E30F50.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/F8D547BE-6884-DD11-822B-000423D95030.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/FA34CEA6-6184-DD11-B69B-001617E30D06.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/FCC6D647-6A84-DD11-95D8-000423D98EC8.root',
                '/store/data/BeamCommissioning08/Cosmics/RECO/v1/000/062/966/FCD1FD91-6684-DD11-8FA8-000423D9A2AE.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(300)
)


# Conditions (Global Tag is used here):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V4P::All"
process.prefer("GlobalTag")

# Magnetic field
process.load("Configuration.StandardSequences.MagneticField_38T_cff")


#process.load("TrackingTools.TrackRefitter.TracksToTrajectories_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")

process.filterOnTracks = cms.EDFilter("TrackCountFilter",
#                                     src = cms.InputTag('ctfWithMaterialTracksP5'),
                                     src = cms.InputTag('globalCosmicMuons'),
                                     minNumber = cms.uint32(1) 
)

#import RecoTracker.TrackProducer.TrackRefitter_cff
#process.CTFRefit = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
#process.CTFRefit.src = 'ctfWithMaterialTracksP5'

#process.load("RecoTracker.TrackProducer.TrackRefitter_cff")
#process.TrackRefitter.src = 'ctfWithMaterialTracksP5'
#process.TrackRefitter.TrajectoryInEvent = True


#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")

process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.globalMuons = cms.EDProducer("TracksToTrajectories",
    Tracks = cms.InputTag("globalCosmicMuons"),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRK')
    )
)

#process.load("Analysis.DiscriminationPower.dedxDiscriminationPower_cff")

process.load("RecoTracker.DeDx.dedxEstimators_Cosmics_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")
#process.load("RecoTracker.DeDx.dedxEstimatorsFromRefitter_Cosmics_cff")

# OUT
process.TFileService = cms.Service("TFileService", fileName = cms.string('histo.root') )

process.OUT = cms.OutputModule("PoolOutputModule",
    fileName       = cms.untracked.string('/tmp/gbruno/out.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
#         'drop *',
#         'keep *_dedxHarmonic2_*_*' ,'keep *_dedxMedian_*_*', 'keep *_dedxTruncated40_*_*',
#         'keep recoTracks_*_*_*'
    )
)

#process.p = cms.Path(process.filterOnTracks * process.TrackRefitter + process.dedxDiscrimPower)

#process.p = cms.Path(process.filterOnTracks * process.TrackRefitter * process.doAlldEdXEstimators)
process.p = cms.Path(process.filterOnTracks * process.reconstructionCosmics * process.doAlldEdXEstimators)
process.outpath  = cms.EndPath(process.OUT)
process.schedule = cms.Schedule(process.p, process.outpath)
