import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from L1Trigger.Phase2L1ParticleFlow.reprocess_L1Phase2_MC_cff import *
from CalibCalorimetry.CaloTPG.CaloTPGTranscoder_cfi import *
from Configuration.StandardSequences.SimL1Emulator_cff import *
from L1Trigger.TrackFindingTracklet.L1TrackletTracks_cff import *
from L1Trigger.L1TTrackMatch.L1TkObjectProducers_cff import *
from L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff import *

l1pfProducerTightTK = l1pfProducer.clone(trkMinStubs = 6)

pf_step = cms.Sequence(
    L1TrackletTracks + SimL1Emulator +
    L1TkElectrons + L1TkPhotons + L1TkJets + L1TkPrimaryVertex + L1TkEtMiss + L1TkHTMissVtx +  L1TkIsoElectrons + L1TkTauFromCalo + L1TkMuons +
    l1ParticleFlow + l1pfProducerTightTK
)




caloStage2 = cms.EDProducer("CandProducerFromStage2",
    srcCluster = cms.InputTag("simCaloStage2Digis","MP"),
    srcTower = cms.InputTag("simCaloStage2Digis","MP"),
    srcJet = cms.InputTag("simCaloStage2Digis","MP"),
    MP = cms.bool(True),
)


from RecoMET.METProducers.PFMET_cfi import pfMet
pfMet.calculateSignificance = False
l1MetCalo    = pfMet.clone(src = "l1pfProducer:Calo")
l1MetTK      = pfMet.clone(src = "l1pfProducer:TK")
l1MetTKV     = pfMet.clone(src = "l1pfProducer:TKVtx")
l1MetTightTK      = pfMet.clone(src = "l1pfProducerTightTK:TK")
l1MetTightTKV     = pfMet.clone(src = "l1pfProducerTightTK:TKVtx")
l1MetPF      = pfMet.clone(src = "l1pfProducer:PF")
l1MetPuppi   = pfMet.clone(src = "l1pfProducer:Puppi")

mets = cms.Sequence( l1MetCalo + l1MetTK + l1MetTKV + l1MetPF + l1MetPuppi + l1MetTightTK + l1MetTightTKV)

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
ak4L1Calo    = ak4PFJets.clone(src = 'l1pfProducer:Calo')
ak4L1TK      = ak4PFJets.clone(src = 'l1pfProducer:TK')
ak4L1TKV     = ak4PFJets.clone(src = 'l1pfProducer:TKVtx')
ak4L1TightTK      = ak4PFJets.clone(src = 'l1pfProducerTightTK:TK')
ak4L1TightTKV     = ak4PFJets.clone(src = 'l1pfProducerTightTK:TKVtx')
ak4L1PF      = ak4PFJets.clone(src = 'l1pfProducer:PF')
ak4L1Puppi   = ak4PFJets.clone(src = 'l1pfProducer:Puppi')

jets = cms.Sequence(     
    ak4L1Calo + ak4L1TK + ak4L1TKV + ak4L1PF + ak4L1Puppi  + ak4L1TightTK + ak4L1TightTKV
)

JEC_PU200 = {
     'Stage2Calo' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble( 34.414,  36.972,  39.228,  54.603,  0.000,  52.360,  47.114,  50.621,  42.136,  34.445),
                        scale   = cms.vdouble( 0.829,  0.848,  0.873,  0.082,  1.000,  0.171,  0.796,  0.994,  1.201,  1.220),
            ),
    'L1Calo' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble( 5.619,  7.247,  12.377,  34.726,  54.554,  62.323,  58.493,  70.205,  60.618,  59.272),
                        scale   = cms.vdouble( 1.016,  1.040,  1.006,  1.008,  0.988,  0.997,  0.816,  0.909,  1.224,  1.550),
            ),
    'L1TK' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble( 2.711,  2.544,  1.738,  0.874, -0.379,  3.927,  0.000,  0.000,  0.000,  0.000),
                        scale   = cms.vdouble( 0.604,  0.589,  0.568,  0.586,  0.560,  0.007,  1.000,  1.000,  1.000,  1.000),
            ),
    'L1TightTK' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble(-0.322, -0.043,  0.115, -1.060, -0.039,  3.186,  0.000,  0.000,  0.000,  0.000),
                        scale   = cms.vdouble( 0.336,  0.291,  0.304,  0.421,  0.293,  0.001,  1.000,  1.000,  1.000,  1.000),
            ),
    'L1TKV' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble(-2.628, -2.755, -2.338, -2.951, -3.166,  3.619,  0.000,  0.000,  0.000,  0.000),
                        scale   = cms.vdouble( 0.548,  0.542,  0.521,  0.542,  0.520,  0.006,  1.000,  1.000,  1.000,  1.000),
            ),
    'L1TightTKV' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble(-1.206, -0.837, -0.884, -2.664, -0.560,  3.436,  0.000,  0.000,  0.000,  0.000),
                        scale   = cms.vdouble( 0.326,  0.290,  0.300,  0.412,  0.282, -0.002,  1.000,  1.000,  1.000,  1.000),
            ),
    'L1PF' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble( 8.576,  9.932,  13.988,  41.925,  74.032,  67.278,  58.493,  70.205,  60.618,  59.272),
                        scale   = cms.vdouble( 1.130,  1.144,  1.129,  1.187,  1.161,  1.039,  0.816,  0.909,  1.224,  1.550),
            ),
    'L1Puppi' : cms.PSet(
                        etaBins = cms.vdouble( 0.500,  1.000,  1.500,  2.000,  2.500,  3.000,  3.500,  4.000,  4.500,  5.000),
                        offset  = cms.vdouble(-12.058, -12.399, -11.728, -10.557, -5.391,  4.586,  3.542,  1.825, -6.946, -17.857),
                        scale   = cms.vdouble( 1.127,  1.155,  1.124,  1.192,  1.289,  0.912,  1.008,  1.298,  1.650,  1.402),
            ),
}

JEC = JEC_PU200;

ntuple = cms.EDAnalyzer("L1TPFJetMetTreeProducer",
    jets = cms.PSet(
        AK4GenJets = cms.InputTag("ak4GenJetsNoNu"),
        AK4Stage2Calo = cms.InputTag("caloStage2:Jet"),
        AK4CaloJets = cms.InputTag("ak4L1Calo"),
        AK4TKJets = cms.InputTag("ak4L1TK"),
        AK4TKVJets = cms.InputTag("ak4L1TKV"),
        AK4TightTKJets = cms.InputTag("ak4L1TightTK"),
        AK4TightTKVJets = cms.InputTag("ak4L1TightTKV"),
        AK4PFJets = cms.InputTag("ak4L1PF"),
        AK4PuppiJets = cms.InputTag("ak4L1Puppi"),
    ),
    jecs = cms.PSet(
        AK4Stage2CaloJets = JEC['Stage2Calo'],
        AK4CaloJets = JEC['L1Calo'],
        AK4TKJets = JEC['L1TK'],
        AK4TKVJets = JEC['L1TKV'],
        AK4TightTKJets = JEC['L1TightTK'],
        AK4TightTKVJets = JEC['L1TightTKV'],
        AK4PFJets = JEC['L1PF'],
        AK4PuppiJets = JEC['L1Puppi'],
    ),
    sels = cms.PSet(
        E13Pt30 = cms.string("pt > 30 && abs(eta) < 1.3"),
        E13Pt40 = cms.string("pt > 40 && abs(eta) < 1.3"),
        #E24Pt15 = cms.string("pt > 15 && abs(eta) < 2.4"),
        #E24Pt20 = cms.string("pt > 20 && abs(eta) < 2.4"),
        E24Pt30 = cms.string("pt > 30 && abs(eta) < 2.4"),
        E24Pt40 = cms.string("pt > 40 && abs(eta) < 2.4"),
        #E24Pt50 = cms.string("pt > 50 && abs(eta) < 2.4"),
        #E24Pt60 = cms.string("pt > 60 && abs(eta) < 2.4"),
        #E24Pt80 = cms.string("pt > 80 && abs(eta) < 2.4"),
        #E30Pt15 = cms.string("pt > 15 && abs(eta) < 3.0"),
        #E30Pt20 = cms.string("pt > 20 && abs(eta) < 3.0"),
        E30Pt30 = cms.string("pt > 30 && abs(eta) < 3.0"),
        E30Pt40 = cms.string("pt > 40 && abs(eta) < 3.0"),
        #E30Pt50 = cms.string("pt > 50 && abs(eta) < 3.0"),
        #E30Pt60 = cms.string("pt > 60 && abs(eta) < 3.0"),
        #E30Pt80 = cms.string("pt > 80 && abs(eta) < 3.0"),
        #E47Pt15 = cms.string("pt > 15 && abs(eta) < 4.7"),
        #E47Pt20 = cms.string("pt > 20 && abs(eta) < 4.7"),
        E47Pt30 = cms.string("pt > 30 && abs(eta) < 4.7"),
        E47Pt40 = cms.string("pt > 40 && abs(eta) < 4.7"),
        #E47Pt50 = cms.string("pt > 50 && abs(eta) < 4.7"),
        #E47Pt60 = cms.string("pt > 50 && abs(eta) < 4.7"),
        #E47Pt80 = cms.string("pt > 80 && abs(eta) < 4.7"),
    ),
    mets = cms.PSet(
        METGen = cms.InputTag("genMetTrue"),
        METCalo = cms.InputTag("l1MetCalo"),
        METTK = cms.InputTag("l1MetTK"),
        METTKV = cms.InputTag("l1MetTKV"),
        METTightTK = cms.InputTag("l1MetTightTK"),
        METTightTKV = cms.InputTag("l1MetTightTKV"),
        METPF = cms.InputTag("l1MetPF"),
        METPuppi = cms.InputTag("l1MetPuppi"),
    ),
    specials = cms.PSet(
        TP_TkEtMiss = cms.PSet( 
            src = cms.InputTag("L1TkEtMiss","MET"),
            cut = cms.string(""),
            expr = cms.string("pt")),
        TP_TkHTMissVtx = cms.PSet( 
            src = cms.InputTag("L1TkHTMissVtx"),
            cut = cms.string(""),
            expr = cms.string("et")),
        TP_TkHTVtx = cms.PSet( 
            src = cms.InputTag("L1TkHTMissVtx"),
            cut = cms.string(""),
            expr = cms.string("EtTotal")),
    )
)


l1pfJetMetTreeProducer = cms.Sequence(
        caloStage2 +
        pf_step +
        mets + jets +
        ntuple)
