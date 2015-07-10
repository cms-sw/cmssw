import FWCore.ParameterSet.Config as cms

electronPreSelection = cms.EDFilter(
    "PATElectronSelector",
    src = cms.InputTag("slimmedElectrons"),
    cut = cms.string('pt > 20. && abs(eta) < 2.5') 
    # JAN: Should add MVA cut here when studied
    )

def getEleCuts(leg, channel='tauEle'):

    ptCut = None
    etaCut = None
#    lmvaID = -99999
    lmvaID1 = -99999
    lmvaID2 = -99999
    lmvaID3 = -99999
#    isoCut = 100
    if channel == 'tauEle':
        ptCut = 24.
        etaCut = 2.1
#        lmvaID = 0.9
        lmvaID1 = 0.925
        lmvaID2 = 0.975
        lmvaID3 = 0.985
#        isoCut = 0.3
    elif channel == 'muEle':
        ptCut = 20.
        etaCut = 2.3
    else:
        raise ValueError('bad channel specification:'+channel)

    eleCuts = cms.PSet(
        kinematics = cms.PSet(
          pt = cms.string('{leg}().pt()>{ptCut}'.format(leg=leg, ptCut=ptCut)),
          eta = cms.string('abs({leg}().eta())<{etaCut}'.format(leg=leg, etaCut=etaCut))
        ),
        ID = cms.PSet(
            hitsnum = cms.string('{leg}().numberOfHits==0'.format(leg=leg)),
            convVeto = cms.string('{leg}().passConversionVeto()!=0'.format(leg=leg)),
            mvaID = cms.string('(abs({leg}().sourcePtr().superCluster().eta())<0.8 && {leg}().mvaNonTrigV0() > {lmvaID1}) || (abs({leg}().sourcePtr().superCluster().eta())>0.8 && abs({leg}().sourcePtr().superCluster().eta())<1.479 && {leg}().mvaNonTrigV0() > {lmvaID2}) || (abs({leg}().sourcePtr().superCluster().eta())>1.479 && {leg}().mvaNonTrigV0() > {lmvaID3})'.format(leg=leg, lmvaID1=lmvaID1, lmvaID2=lmvaID2, lmvaID3=lmvaID3))
        ),
    )

    return eleCuts

