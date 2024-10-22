import FWCore.ParameterSet.Config as cms

process = cms.Process("eleIso")
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.EventContent.EventContent_cff")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_1_0_pre1/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/16EE7689-5EF4-DD11-97EF-001D09F2514F.root',
        '/store/relval/CMSSW_3_1_0_pre1/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/66B68FBF-56F4-DD11-86DD-001617C3B710.root',
        '/store/relval/CMSSW_3_1_0_pre1/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/8A6092F3-51F4-DD11-A213-001617DBD5AC.root',
        '/store/relval/CMSSW_3_1_0_pre1/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/AE032B6E-54F4-DD11-ACA5-000423D9A212.root',
        '/store/relval/CMSSW_3_1_0_pre1/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/C6DDF78F-55F4-DD11-8FF4-001617C3B66C.root',
        '/store/relval/CMSSW_3_1_0_pre1/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/E83F680E-56F4-DD11-9976-001617C3B6CE.root'

    )
)

process.out = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('file:eleIso.root')
)

process.out.outputCommands.append('drop *_*_*_*')
process.out.outputCommands.append('keep *_gsfElectrons_*_*')
process.out.outputCommands.append('keep *_photons_*_*')
process.out.outputCommands.append('keep *_*_*_eleIso')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff")
process.load("RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequencePAT_cff")

#Couple extra modules to test that the new abs(energy) was working:
#Get IsoDeposits with no min energy cut, all RecHits stored
process.eleIsoDepositEcalFromHitsNoCut = process.eleIsoDepositEcalFromHits.clone()
process.eleIsoDepositEcalFromHitsNoCut.ExtractorPSet.energyMin = cms.double(0.0)

#Default Vetos on new IsoDeposit
process.eleIsoFromDepsEcalFromHitsAbs = process.eleIsoFromDepsEcalFromHits.clone()
process.eleIsoFromDepsEcalFromHitsAbs.deposits[0].src = 'eleIsoDepositEcalFromHitsNoCut'

#Vetos without the abs(energy) cut, using old single sided cut
process.eleIsoFromDepsEcalFromHitsNoAbs = process.eleIsoFromDepsEcalFromHits.clone()
process.eleIsoFromDepsEcalFromHitsNoAbs.deposits[0].src = 'eleIsoDepositEcalFromHitsNoCut'
process.eleIsoFromDepsEcalFromHitsNoAbs.deposits[0].vetos = cms.vstring(
    'EcalBarrel:0.045',
    'EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)',
    'EcalBarrel:ThresholdFromTransverse(0.08)',
    'EcalEndcaps:ThresholdFromTransverse(0.3)',
    'EcalEndcaps:0.070',
    'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'
)

#Rediculous inner cone to make sure the cuts were working correctly:
process.eleIsoFromDepsEcalFromHitsNoAbs2 = process.eleIsoFromDepsEcalFromHits.clone()
process.eleIsoFromDepsEcalFromHitsNoAbs2.deposits[0].src = 'eleIsoDepositEcalFromHitsNoCut'
process.eleIsoFromDepsEcalFromHitsNoAbs2.deposits[0].vetos = cms.vstring(
    'EcalBarrel:0.3',
    'EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)',
    'EcalBarrel:ThresholdFromTransverse(0.08)',
    'EcalEndcaps:ThresholdFromTransverse(0.3)',
    'EcalEndcaps:0.3',
    'EcalEndcaps:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)'
)
    

process.p1 = cms.Path(
    process.egammaIsolationSequence + 
    process.egammaIsolationSequencePAT + 
    ( 
        process.eleIsoDepositEcalFromHitsNoCut *
        process.eleIsoFromDepsEcalFromHitsAbs *
        process.eleIsoFromDepsEcalFromHitsNoAbs *
        process.eleIsoFromDepsEcalFromHitsNoAbs2 
    )
)

process.outpath = cms.EndPath(process.out)
