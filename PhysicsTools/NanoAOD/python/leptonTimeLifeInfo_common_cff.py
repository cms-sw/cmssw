#
# Common definition of time-life variables for pat-leptons produced
# with {Electron,Muon,Tau}TimeLifeInfoTableProducer
#
import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.PatAlgos.patRefitVertexProducer_cfi import patRefitVertexProducer
from PhysicsTools.NanoAOD.simpleVertexFlatTableProducer_cfi import simpleVertexFlatTableProducer
from PhysicsTools.PatAlgos.patElectronTimeLifeInfoProducer_cfi import patElectronTimeLifeInfoProducer
from PhysicsTools.PatAlgos.patMuonTimeLifeInfoProducer_cfi import patMuonTimeLifeInfoProducer
from PhysicsTools.PatAlgos.patTauTimeLifeInfoProducer_cfi import patTauTimeLifeInfoProducer
from PhysicsTools.NanoAOD.simplePATElectron2TrackTimeLifeInfoFlatTableProducer_cfi import simplePATElectron2TrackTimeLifeInfoFlatTableProducer
from PhysicsTools.NanoAOD.simplePATMuon2TrackTimeLifeInfoFlatTableProducer_cfi import simplePATMuon2TrackTimeLifeInfoFlatTableProducer
from PhysicsTools.NanoAOD.simplePATTau2TrackTimeLifeInfoFlatTableProducer_cfi import simplePATTau2TrackTimeLifeInfoFlatTableProducer
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *

# common settings of lepton life-time info producer
prod_common = cms.PSet(
    pvSource = cms.InputTag("offlineSlimmedPrimaryVerticesWithBS"),
    pvChoice = cms.int32(0) #0: PV[0], 1: smallest dz
)

# impact parameter
ipVars = cms.PSet(
    #ipLength = Var("ipLength().value()", float, doc="lenght of impact parameter (3d)", precision=10),
    ipLengthSig = Var("ipLength().significance()", float, doc="significance of impact parameter", precision=10),
    IPx = Var("ipVector().x()", float, doc="x coordinate of impact parameter vector", precision=10),
    IPy = Var("ipVector().y()", float, doc="y coordinate of impact parameter vector", precision=10),
    IPz = Var("ipVector().z()", float, doc="z coordinate of impact parameter vector", precision=10)
)

# track parameters and covariance at ref. point
trackVars = cms.PSet(
    track_qoverp = Var("?hasTrack()?track().parameter(0):0", float, doc="track q/p", precision=10),
    track_lambda = Var("?hasTrack()?track().parameter(1):0", float, doc="track lambda", precision=10),
    track_phi = Var("?hasTrack()?track().parameter(2):0", float, doc="track phi", precision=10),
    #track_deltaPhi = Var("?hasTrack()?deltaPhi(track().parameter(2), phi):0", float, doc="track phi minus lepton phi", precision=10),
    track_dxy = Var("?hasTrack()?track().parameter(3):0", float, doc="track dxy", precision=10),
    track_dsz = Var("?hasTrack()?track().parameter(4):0", float, doc="track dsz", precision=10),
    bField_z = Var("?hasTrack()?bField_z:0", float, doc="z coordinate of magnetic field at track ref. point", precision=10),
)
# track covariance elements (adding to trackVars)
for i in range(0,5):
    for j in range(i,5):
        jistr = str(j)+str(i)
        setattr(trackVars, 'track_cov'+jistr, Var("?hasTrack()?track().covariance("+str(j)+","+str(i)+"):0", float, doc="track covariance element ("+str(j)+","+str(i)+")", precision=10))

# secondary vertex
svVars = cms.PSet(
    # SV
    hasRefitSV = Var("hasSV()", bool, doc="has SV refit using miniAOD quantities"),
    refitSVx = Var("?hasSV()?sv().x():0", float, doc="x coordinate of SV", precision=10),
    refitSVy = Var("?hasSV()?sv().y():0", float, doc="y coordinate of SV", precision=10),
    refitSVz = Var("?hasSV()?sv().z():0", float, doc="z coordinate of SV", precision=10),
    refitSVchi2 = Var("?hasSV()?sv().normalizedChi2():0", float, doc="reduced chi2, i.e. chi2/ndof, of SV fit", precision=8),
    #refitSVndof = Var("?hasSV()?sv().ndof():0", float, doc="ndof of SV fit", precision=8),
    # flight-length
    #refitFlightLength = Var("?hasSV()?flightLength().value():0", float, doc="flight-length,i.e. the PV to SV distance", precision=10),
    #refitFlightLengthSig = Var("?hasSV()?flightLength().significance():0", float, doc="Significance of flight-length", precision=10)
)
# secondary vertex covariance elements (adding to svVars)
for i in range(0,3):
    for j in range(i,3):
        jistr = str(j)+str(i)
        setattr(svVars, 'refitSVcov'+jistr, Var("?hasSV()?sv().covariance("+str(j)+","+str(i)+"):0", float, doc="Covariance of SV ("+str(j)+","+str(i)+")", precision=10))

# primary vertex covariance elements
pvCovVars = cms.PSet()
for i in range(0,3):
    for j in range(i,3):
        jistr = str(j)+str(i)
        setattr(pvCovVars, 'cov'+jistr, Var("covariance("+str(j)+","+str(i)+")", float, doc="vertex covariance ("+str(j)+","+str(i)+")", precision=10))

# Module to refit PV with beam-spot constraint that is not present in Run-2 samples
refittedPV = patRefitVertexProducer.clone(
    srcVertices = "offlineSlimmedPrimaryVertices",
)
run2_nanoAOD_ANY.toModify(
    prod_common, pvSource = "refittedPV")

# Definition of DQM plots
ipVarsPlots = cms.VPSet(
    #Plot1D('ipLength', 'ipLength', 25, -0.25, 0.25, 'signed lenght of impact parameter (3d)'),
    Plot1D('ipLengthSig', 'ipLengthSig', 60, -5, 10, 'signed significance of impact parameter'),
    Plot1D('IPx', 'IPx', 40, -0.02, 0.02, 'x coordinate of impact parameter vector'),
    Plot1D('IPy', 'IPy', 40, -0.02, 0.02, 'y coordinate of impact parameter vector'),
    Plot1D('IPz', 'IPz', 40, -0.02, 0.02, 'z coordinate of impact parameter vector')
)
trackVarsPlots = cms.VPSet(
    Plot1D('track_qoverp', 'track_qoverp', 40, -0.2, 0.2, 'track q/p'),
    Plot1D('track_lambda', 'track_lambda', 30, -1.5, 1.5, 'track lambda'),
    Plot1D('track_phi', 'track_phi', 20, -3.14159, 3.14159, 'track phi'),
    Plot1D('track_dxy', 'track_dxy', 20, -0.1, 0.1, 'track dxy'),
    Plot1D('track_dsz', 'track_dsz', 20, -10, 10, 'track dsz'),
    NoPlot('bField_z')
)
#no plots for track covariance elements, but store placeholders
for i in range(0,5):
    for j in range(i,5):
        trackVarsPlots.append(NoPlot('track_cov'+str(j)+str(i)))
svVarsPlots = cms.VPSet(
    Plot1D('hasRefitSV', 'hasRefitSV', 2, 0, 2, 'has SV refit using miniAOD quantities'),
    Plot1D('refitSVx', 'refitSVx', 20, -0.1, 0.1, 'x coordinate of refitted SV'),
    Plot1D('refitSVy', 'refitSVy', 20, -0.1, 0.1, 'y coordinate of refitted SV'),
    Plot1D('refitSVz', 'refitSVz', 20, -20, 20, 'z coordinate of refitted SV'),
    Plot1D('refitSVchi2', 'refitSVchi2', 20, 0, 40, 'reduced chi2 of SV fit'),
    #Plot1D('refitSVndof', 'refitSVndof', 10, 0, 10, 'ndof of SV fit')
)
#no plots for SV covariance elements, but store placeholders
for i in range(0,3):
    for j in range(i,3):
        svVarsPlots.append(NoPlot('refitSVcov'+str(j)+str(i)))

#
# Customization sequences and functions
#
# electrons
def addElectronTimeLifeInfoTask(process):
    process.electronTimeLifeInfos = patElectronTimeLifeInfoProducer.clone(
        src = process.electronTable.src,
        selection = 'pt > 15',
        pvSource = prod_common.pvSource,
        pvChoice = prod_common.pvChoice
    )
    process.electronTimeLifeInfoTable = simplePATElectron2TrackTimeLifeInfoFlatTableProducer.clone(
        name = process.electronTable.name,
        src = process.electronTable.src,
        doc = cms.string("Additional time-life info for non-prompt electrons"),
        extension = True,
        externalTypedVariables = cms.PSet()
    )
    process.electronTimeLifeInfoTask = cms.Task(
        process.electronTimeLifeInfos,
        process.electronTimeLifeInfoTable
    )
    # refit PV with beam-spot constraint that is not present in Run-2 samples
    if not hasattr(process,'refittedPV'):
        setattr(process,'refittedPV',refittedPV)
    _electronTimeLifeInfoTaskRun2 = process.electronTimeLifeInfoTask.copy()
    _electronTimeLifeInfoTaskRun2.add(process.refittedPV)
    run2_nanoAOD_ANY.toReplaceWith(process.electronTimeLifeInfoTask,
                                   _electronTimeLifeInfoTaskRun2)
    process.electronTablesTask.add(process.electronTimeLifeInfoTask)
    return process
#base vars
electronVars = cms.PSet(
    ipVars
)
for var in electronVars.parameters_():
    setattr(getattr(electronVars, var), "src", cms.InputTag("electronTimeLifeInfos"))
def addTimeLifeInfoToElectrons(process):
    if not hasattr(process,'electronTimeLifeInfoTask'):
        process = addElectronTimeLifeInfoTask(process)
    electronExtVars = cms.PSet(
        process.electronTimeLifeInfoTable.externalTypedVariables,
        electronVars
    )
    process.electronTimeLifeInfoTable.externalTypedVariables = electronExtVars
    # add DQM plots if needed
    if hasattr(process,'nanoDQM'):
        process.nanoDQM.vplots.Electron.plots.extend(ipVarsPlots)
    return process
#track vars
electronTrackVars = cms.PSet(
    trackVars
)
for var in electronTrackVars.parameters_():
    setattr(getattr(electronTrackVars, var), "src", cms.InputTag("electronTimeLifeInfos"))
def addElectronTrackVarsToTimeLifeInfo(process):
    if not hasattr(process,'electronTimeLifeInfoTask'):
        process = addElectronTimeLifeInfoTask(process)
    electronExtVars = cms.PSet(
        process.electronTimeLifeInfoTable.externalTypedVariables,
        electronTrackVars
    )
    process.electronTimeLifeInfoTable.externalTypedVariables = electronExtVars
    # add DQM plots if needed
    if hasattr(process,'nanoDQM'):
        process.nanoDQM.vplots.Electron.plots.extend(trackVarsPlots)
    return process

# muons
def addMuonTimeLifeInfoTask(process):
    process.muonTimeLifeInfos = patMuonTimeLifeInfoProducer.clone(
        src = process.muonTable.src,
        selection = 'pt > 15',
        pvSource = prod_common.pvSource,
        pvChoice = prod_common.pvChoice
    )
    process.muonTimeLifeInfoTable = simplePATMuon2TrackTimeLifeInfoFlatTableProducer.clone(
        name = process.muonTable.name,
        src = process.muonTable.src,
        doc = cms.string("Additional time-life info for non-prompt muon"),
        extension = True,
        externalTypedVariables = cms.PSet()
    )
    process.muonTimeLifeInfoTask = cms.Task(
        process.muonTimeLifeInfos,
        process.muonTimeLifeInfoTable
    )
    # refit PV with beam-spot constraint that is not present in Run-2 samples
    if not hasattr(process,'refittedPV'):
        setattr(process,'refittedPV',refittedPV)
    _muonTimeLifeInfoTaskRun2 = process.muonTimeLifeInfoTask.copy()
    _muonTimeLifeInfoTaskRun2.add(process.refittedPV)
    run2_nanoAOD_ANY.toReplaceWith(process.muonTimeLifeInfoTask,
                                   _muonTimeLifeInfoTaskRun2)
    process.muonTablesTask.add(process.muonTimeLifeInfoTask)
    return process
#base vars
muonVars = cms.PSet(
    ipVars
)
for var in muonVars.parameters_():
    setattr(getattr(muonVars, var), "src", cms.InputTag("muonTimeLifeInfos"))
def addTimeLifeInfoToMuons(process):
    if not hasattr(process,'muonTimeLifeInfoTask'):
        process = addMuonTimeLifeInfoTask(process)
    muonExtVars = cms.PSet(
        process.muonTimeLifeInfoTable.externalTypedVariables,
        muonVars
    )
    process.muonTimeLifeInfoTable.externalTypedVariables = muonExtVars
    # add DQM plots if needed
    if hasattr(process,'nanoDQM'):
        process.nanoDQM.vplots.Muon.plots.extend(ipVarsPlots)
    return process
#track vars
muonTrackVars = cms.PSet(
    trackVars
)
for var in muonTrackVars.parameters_():
    setattr(getattr(muonTrackVars, var), "src", cms.InputTag("muonTimeLifeInfos"))
def addMuonTrackVarsToTimeLifeInfo(process):
    if not hasattr(process,'muonTimeLifeInfoTask'):
        process = addMuonTimeLifeInfoTask(process)
    muonExtVars = cms.PSet(
        process.muonTimeLifeInfoTable.externalTypedVariables,
        muonTrackVars
    )
    process.muonTimeLifeInfoTable.externalTypedVariables = muonExtVars
    # add DQM plots if needed
    if hasattr(process,'nanoDQM'):
        process.nanoDQM.vplots.Muon.plots.extend(trackVarsPlots)
    return process

# taus
def addTauTimeLifeInfoTask(process):
    process.tauTimeLifeInfos = patTauTimeLifeInfoProducer.clone(
        src = process.tauTable.src,
        pvSource = prod_common.pvSource,
        pvChoice = prod_common.pvChoice
    )
    process.tauTimeLifeInfoTable = simplePATTau2TrackTimeLifeInfoFlatTableProducer.clone(
        name = process.tauTable.name,
        src = process.tauTable.src,
        doc = cms.string("Additional tau time-life info"),
        extension = True,
        externalTypedVariables = cms.PSet()
    )
    process.tauTimeLifeInfoTask = cms.Task(
        process.tauTimeLifeInfos,
        process.tauTimeLifeInfoTable
    )
    # refit PV with beam-spot constraint that is not present in Run-2 samples
    if not hasattr(process,'refittedPV'):
        setattr(process,'refittedPV',refittedPV)
    _tauTimeLifeInfoTaskRun2 = process.tauTimeLifeInfoTask.copy()
    _tauTimeLifeInfoTaskRun2.add(process.refittedPV)
    run2_nanoAOD_ANY.toReplaceWith(process.tauTimeLifeInfoTask,
                                   _tauTimeLifeInfoTaskRun2)
    process.tauTablesTask.add(process.tauTimeLifeInfoTask)
    return process
# base vars
tauVars = cms.PSet(
    svVars,
    ipVars
)
for var in tauVars.parameters_():
    setattr(getattr(tauVars, var), "src", cms.InputTag("tauTimeLifeInfos"))
def addTimeLifeInfoToTaus(process):
    if not hasattr(process,'tauTimeLifeInfoTask'):
        process = addTauTimeLifeInfoTask(process)
    tauExtVars = cms.PSet(
        process.tauTimeLifeInfoTable.externalTypedVariables,
        tauVars
    )
    process.tauTimeLifeInfoTable.externalTypedVariables = tauExtVars
    # add DQM plots if needed
    if hasattr(process,'nanoDQM'):
        process.nanoDQM.vplots.Tau.plots.extend(ipVarsPlots)
        process.nanoDQM.vplots.Tau.plots.extend(svVarsPlots)
    return process
#track vars
tauTrackVars = cms.PSet(
    trackVars
)
for var in tauTrackVars.parameters_():
    setattr(getattr(tauTrackVars, var), "src", cms.InputTag("tauTimeLifeInfos"))
def addTauTrackVarsToTimeLifeInfo(process):
    if not hasattr(process,'tauTimeLifeInfoTask'):
        process = addTauTimeLifeInfoTask(process)
    tauExtVars = cms.PSet(
        process.tauTimeLifeInfoTable.externalTypedVariables,
        tauTrackVars
    )
    process.tauTimeLifeInfoTable.externalTypedVariables = tauExtVars
    # add DQM plots if needed
    if hasattr(process,'nanoDQM'):
        process.nanoDQM.vplots.Tau.plots.extend(trackVarsPlots)
    return process

# Vertices
def addExtendVertexInfo(process):
    process.pvbsTable = simpleVertexFlatTableProducer.clone(
        src = prod_common.pvSource,
        name = "PVBS",
        doc = "main primary vertex with beam-spot",
        maxLen = 1,
        variables = cms.PSet(
            pvCovVars,
            x = Var("position().x()", float, doc = "position x coordinate, in cm", precision = 10),
            y = Var("position().y()", float, doc = "position y coordinate, in cm", precision = 10),
            z = Var("position().z()", float, doc = "position z coordinate, in cm", precision = 16),
            #ndof = Var("ndof()", float, doc = "number of degrees of freedom", precision = 8),#MB: not important
            chi2 = Var("normalizedChi2()", float, doc = "reduced chi2, i.e. chi2/ndof", precision = 8),
        ),
    )
    process.pvbsTableTask = cms.Task(process.pvbsTable)
    # refit PV with beam-spot constraint that is not present in Run-2 samples
    if not hasattr(process,'refittedPV'):
        setattr(process,'refittedPV',refittedPV)
    _pvbsTableTaskRun2 = process.pvbsTableTask.copy()
    _pvbsTableTaskRun2.add(process.refittedPV)
    run2_nanoAOD_ANY.toReplaceWith(process.pvbsTableTask,
                                   _pvbsTableTaskRun2)
    process.vertexTablesTask.add(process.pvbsTableTask)
    return process

# Time-life info without track parameters
def addTimeLifeInfoBase(process):
    addTimeLifeInfoToElectrons(process)
    addTimeLifeInfoToMuons(process)
    addTimeLifeInfoToTaus(process)
    addExtendVertexInfo(process)
    return process
# Add track parameters to time-life info
def addTrackVarsToTimeLifeInfo(process):
    addElectronTrackVarsToTimeLifeInfo(process)
    addMuonTrackVarsToTimeLifeInfo(process)
    addTauTrackVarsToTimeLifeInfo(process)
    return process
# Full
def addTimeLifeInfo(process):
    addTimeLifeInfoBase(process)
    addTrackVarsToTimeLifeInfo(process)
    return process
