import os
import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.eventContent.tauMu_cff import tauMu as tauMuEventContent
from CMGTools.H2TauTau.eventContent.tauMu_cff import tauMuDebug as tauMuDebugEventContent
from CMGTools.H2TauTau.eventContent.tauEle_cff import tauEle as tauEleEventContent
from CMGTools.H2TauTau.eventContent.tauEle_cff import tauEleDebug as tauEleDebugEventContent
from CMGTools.H2TauTau.eventContent.muEle_cff import muEle as muEleEventContent
from CMGTools.H2TauTau.eventContent.muEle_cff import muEleDebug as muEleDebugEventContent
from CMGTools.H2TauTau.eventContent.diTau_cff import diTau as diTauEventContent
from CMGTools.H2TauTau.eventContent.diTau_cff import diTauDebug as diTauDebugEventContent
from CMGTools.H2TauTau.eventContent.diMu_cff import diMu as diMuEventContent
from CMGTools.H2TauTau.eventContent.diMu_cff import diMuDebug as diMuDebugEventContent


def addOutput(process, type12, addDebugEventContent=False, addPreSel=True, oneFile=False):

    allowedTypes = ['tauMu', 'tauEle', 'muEle', 'diTau', 'diMu']
    if type12 not in allowedTypes:
        raise ValueError(type12 + ' not in allowed types: ', allowedTypes)

    # skim (basic selection)     ------
    outFileNameExt = 'CMG'
    if oneFile:
        mytype = 'htt'
    else:
        mytype = type12
    basicName = '{type}_presel_tree_{ext}.root'.format(
        type=mytype,
        ext=outFileNameExt
    )

    eventContent = None
    debugEventContent = None
    if type12 == 'tauMu':
        eventContent = tauMuEventContent
        debugEventContent = tauMuDebugEventContent
    elif type12 == 'tauEle':
        eventContent = tauEleEventContent
        debugEventContent = tauEleDebugEventContent
    elif type12 == 'muEle':
        eventContent = muEleEventContent
        debugEventContent = muEleDebugEventContent
    elif type12 == 'diMu':
        eventContent = diMuEventContent
        debugEventContent = diMuDebugEventContent
    elif type12 == 'diTau':
        eventContent = diTauEventContent
        debugEventContent = diTauDebugEventContent
    elif oneFile:
        eventContent = set(tauMuEventContent +
                           tauEleEventContent +
                           diTauEventContent +
                           muEleEventContent)
        debugEventContent = set(tauMuDebugEventContent +
                                tauEleDebugEventContent +
                                muEleDebugEventContent +
                                diTauDebugEventContent)

    prePathVString = ['{type12}PreSelPath'.format(type12=type12)]
    if oneFile:
        prePathVString = ['{type12}PreSelPath'.format(type12=ctype) for ctype in allowedTypes]
    out = cms.OutputModule(
        "PoolOutputModule",
        fileName=cms.untracked.string(basicName),
        # save only events passing the full path
        SelectEvents=cms.untracked.PSet(
            SelectEvents=cms.vstring(prePathVString)
        ),
        # save PAT Layer 1 output; you need a '*' to
        # unpack the list of commands 'patEventContent'
        outputCommands=cms.untracked.vstring('drop *')
    )
    if addDebugEventContent:
        out.outputCommands.extend(debugEventContent)
    else:
        out.outputCommands.extend(eventContent)

    # full baseline selection    ------

    pathVString = ['{type12}Path'.format(type12=type12)]
    if oneFile:
        pathVString = ['{type12}Path'.format(type12=ctype) for ctype in allowedTypes]
    outBaseline = out.clone()
    outBaseline.SelectEvents = cms.untracked.PSet(
        SelectEvents=cms.vstring(pathVString)
    )
    baselineName = '{type12}_fullsel_tree_{ext}.root'.format(
        type12=mytype,
        ext=outFileNameExt
    )
    outBaseline.fileName = baselineName

    setattr(process, os.path.splitext(baselineName)[0], outBaseline)
    process.outpath += outBaseline
    print 'adding output:', outBaseline.fileName
    if addPreSel:
        setattr(process, os.path.splitext(basicName)[0], out)
        process.outpath += out
        print 'adding output:', out.fileName


def addTauMuOutput(process, debugEventContent=False, addPreSel=True, oneFile=False):
    addOutput(process, 'tauMu', debugEventContent, addPreSel, oneFile)


def addTauEleOutput(process, debugEventContent=False, addPreSel=True, oneFile=False):
    addOutput(process, 'tauEle', debugEventContent, addPreSel, oneFile)


def addMuEleOutput(process, debugEventContent=False, addPreSel=True, oneFile=False):
    addOutput(process, 'muEle', debugEventContent, addPreSel, oneFile)


def addDiMuOutput(process, debugEventContent=False, addPreSel=True, oneFile=False):
    addOutput(process, 'diMu', debugEventContent, addPreSel, oneFile)


def addDiTauOutput(process, debugEventContent=False, addPreSel=True, oneFile=False):
    addOutput(process, 'diTau', debugEventContent, addPreSel, oneFile)
