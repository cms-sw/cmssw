import FWCore.ParameterSet.Config as cms

tauPreSelection = cms.EDFilter(
    # "PFTauSelector",
    "PATTauSelector",
    src = cms.InputTag("slimmedTaus"),
    # The tau disriminators are defined here http://cmslxr.fnal.gov/lxr/source/PhysicsTools/PatAlgos/python/producersLayer1/tauProducer_cfi.py
    cut = cms.string('pt > 15. && abs(eta) < 2.5 && tauID("decayModeFinding") > 0.5') 
    # againstMuonLooseMVA, againstElectronLooseMVA5: could be in pre-selection 
    # as well
    )


cutsElectronMVA3Medium = [0.933,0.921,0.944,0.945,0.918,0.941,0.981,0.943,0.956,0.947,0.951,0.95,0.897,0.958,0.955,0.942]

electronMVA3MediumString = ''
for iCat, catCut in enumerate(cutsElectronMVA3Medium):
    mvaCut = '({{leg}}().tauID("againstElectronMVA3category") == {cat} && {{leg}}().tauID("againstElectronMVA3raw") > {catCut})'.format(cat=iCat, catCut=catCut)
    if iCat == 0:
        electronMVA3MediumString = mvaCut
    else:
        electronMVA3MediumString = '||'.join([electronMVA3MediumString, mvaCut])
electronMVA3MediumString += '|| {leg}().tauID("againstElectronMVA3category") > 15'

def getTauCuts(leg, channel='tauMu'):

    ptCut = 15.
    etaCut = 2.3

    kinematics = cms.PSet(
        pt = cms.string('{leg}().pt()>{ptCut}'.format(leg=leg, ptCut=ptCut)),
        eta = cms.string('abs({leg}().eta())<{etaCut}'.format(leg=leg, etaCut=etaCut))
        )
    iso = cms.string('{leg}().tauID("byCombinedIsolationDeltaBetaCorrRaw3Hits") < 10.0'.format(leg=leg))
    if channel == 'tauMu':
        id = cms.PSet(
            decay = cms.string('{leg}().tauID("decayModeFinding")'.format(leg=leg)),
            muRejection = cms.string('{leg}().tauID("againstMuonTight") > 0.5'.format(leg=leg))
            )
    elif channel == 'tauEle':    
        id = cms.PSet(
            decay = cms.string('{leg}().tauID("decayModeFinding")'.format(leg=leg)),
            eleRejection = cms.string(electronMVA3MediumString.format(leg=leg)),
            muRejection2 = cms.string('{leg}().tauID("againstMuonLoose") > 0.5'.format(leg=leg))

            # As long as tau ID version is not finalised, cannot apply any of the following cuts at preselection level
            # eleRejection1 = cms.string('{leg}().tauID("againstElectronMVA") > 0.5'.format(leg=leg)),
            # eleRejection2 = cms.string('{leg}().tauID("againstElectronMedium") > 0.5'.format(leg=leg)),
            # muRejection2 = cms.string('{leg}().tauID("againstMuonLoose") > 0.5'.format(leg=leg))
            )
    else :
        id = cms.PSet(
            decay = cms.string('{leg}().tauID("decayModeFinding")'.format(leg=leg)),
            )

    tauCuts = cms.PSet(
        kinematics = kinematics.clone(),
        id = id.clone(),
        iso = iso
        )
    
    return tauCuts
