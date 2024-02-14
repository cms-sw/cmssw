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
from PhysicsTools.NanoAOD.simpleCandidate2TrackTimeLifeInfoFlatTableProducer_cfi import simpleCandidate2TrackTimeLifeInfoFlatTableProducer
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

# common settings of lepton life-time info producer
prod_common = cms.PSet(
    pvSource = cms.InputTag("offlineSlimmedPrimaryVerticesWithBS"),
    pvChoice = cms.int32(0) #0: PV[0], 1: smallest dz
)

# impact parameter
ipVars = cms.PSet(
    ipLength = Var("ipLength().value()", float, doc="lenght of impact parameter (3d)", precision=10),
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
    refitSVchi2 = Var("?hasSV()?sv().chi2():0", float, doc="chi2 of SV fit", precision=8),
    refitSVndof = Var("?hasSV()?sv().ndof():0", float, doc="ndof of SV fit", precision=8),
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

#
# Customization sequences and functions
#
# electrons
electronTimeLifeInfos = patElectronTimeLifeInfoProducer.clone(
    selection = 'pt > 15',
    pvSource = prod_common.pvSource,
    pvChoice = prod_common.pvChoice
)
vars = cms.PSet(
    ipVars,
    trackVars
)
for var in vars.parameters_():
    setattr(getattr(vars, var), "src", cms.InputTag("electronTimeLifeInfos"))
electronTimeLifeInfoTable = simpleCandidate2TrackTimeLifeInfoFlatTableProducer.clone(
    doc = cms.string("Additional time-life info for non-prompt electrons"),
    extension = True,
    externalTypedVariables = vars
)
electronTimeLifeInfoTask = cms.Task(
    electronTimeLifeInfos,
    electronTimeLifeInfoTable
)
# refit PV with beam-spot constraint that is not present in Run-2 samples
_electronTimeLifeInfoTaskRun2 = electronTimeLifeInfoTask.copy()
_electronTimeLifeInfoTaskRun2.add(refittedPV)
run2_nanoAOD_ANY.toReplaceWith(electronTimeLifeInfoTask, _electronTimeLifeInfoTaskRun2)
def addTimeLifeInfoToElectrons(process):
    process.electronTimeLifeInfoTable.name = process.electronTable.name
    process.electronTimeLifeInfoTable.src = process.electronTable.src
    process.electronTimeLifeInfos.src = process.electronTable.src
    process.electronTablesTask.add(process.electronTimeLifeInfoTask)
    return process

# muons
muonTimeLifeInfos = patMuonTimeLifeInfoProducer.clone(
    selection = 'pt > 15',
    pvSource = prod_common.pvSource,
    pvChoice = prod_common.pvChoice
)
vars = cms.PSet(
    ipVars,
    trackVars
)
for var in vars.parameters_():
    setattr(getattr(vars, var), "src", cms.InputTag("muonTimeLifeInfos"))
muonTimeLifeInfoTable = simpleCandidate2TrackTimeLifeInfoFlatTableProducer.clone(
    doc = cms.string("Additional time-life info for non-prompt muon"),
    extension = True,
    externalTypedVariables = vars
)
muonTimeLifeInfoTask = cms.Task(
    muonTimeLifeInfos,
    muonTimeLifeInfoTable
)
# refit PV with beam-spot constraint that is not present in Run-2 samples
_muonTimeLifeInfoTaskRun2 = muonTimeLifeInfoTask.copy()
_muonTimeLifeInfoTaskRun2.add(refittedPV)
run2_nanoAOD_ANY.toReplaceWith(muonTimeLifeInfoTask, _muonTimeLifeInfoTaskRun2)
def addTimeLifeInfoToMuons(process):
    process.muonTimeLifeInfoTable.name = process.muonTable.name
    process.muonTimeLifeInfoTable.src = process.muonTable.src
    process.muonTimeLifeInfos.src = process.muonTable.src
    process.muonTablesTask.add(process.muonTimeLifeInfoTask)
    return process

# taus
tauTimeLifeInfos = patTauTimeLifeInfoProducer.clone(
    pvSource = prod_common.pvSource,
    pvChoice = prod_common.pvChoice
)
vars = cms.PSet(
    svVars,
    ipVars,
    trackVars
)
for var in vars.parameters_():
    setattr(getattr(vars, var), "src", cms.InputTag("tauTimeLifeInfos"))
tauTimeLifeInfoTable = simpleCandidate2TrackTimeLifeInfoFlatTableProducer.clone(
    doc = cms.string("Additional tau time-life info"),
    extension = True,
    externalTypedVariables = vars
)
tauTimeLifeInfoTask = cms.Task(
    tauTimeLifeInfos,
    tauTimeLifeInfoTable
)
# refit PV with beam-spot constraint that is not present in Run-2 samples
_tauTimeLifeInfoTaskRun2 = tauTimeLifeInfoTask.copy()
_tauTimeLifeInfoTaskRun2.add(refittedPV)
run2_nanoAOD_ANY.toReplaceWith(tauTimeLifeInfoTask,_tauTimeLifeInfoTaskRun2)
def addTimeLifeInfoToTaus(process):
    process.tauTimeLifeInfoTable.name = process.tauTable.name
    process.tauTimeLifeInfoTable.src = process.tauTable.src
    process.tauTimeLifeInfos.src = process.tauTable.src
    process.tauTablesTask.add(process.tauTimeLifeInfoTask)
    return process

# Vertices
pvbsTable = simpleVertexFlatTableProducer.clone(
    src = prod_common.pvSource,
    name = "PVBS",
    doc = "main primary vertex with beam-spot",
    maxLen = 1,
    variables = cms.PSet(
        pvCovVars,
        x = Var("position().x()", float, doc = "position x coordinate, in cm", precision = 10),
        y = Var("position().y()", float, doc = "position y coordinate, in cm", precision = 10),
        z = Var("position().z()", float, doc = "position z coordinate, in cm", precision = 16),
        ndof = Var("ndof()", float, doc = "number of degrees of freedom", precision = 8),
        chi2 = Var("normalizedChi2()", float, doc = "reduced chi2, i.e. chi2/ndof", precision = 8),
    ),
)
pvbsTableTask = cms.Task(pvbsTable)
# refit PV with beam-spot constraint that is not present in Run-2 samples
_pvbsTableTaskRun2 = pvbsTableTask.copy()
_pvbsTableTaskRun2.add( refittedPV )
run2_nanoAOD_ANY.toReplaceWith(
    pvbsTableTask, _pvbsTableTaskRun2
)
def addExtendVertexInfo(process):
    process.vertexTablesTask.add(process.pvbsTableTask)
    return process

# DQM
#FIXME!

# Full
def addTimeLifeInfo(process):
    addTimeLifeInfoToElectrons(process)
    addTimeLifeInfoToMuons(process)
    addTimeLifeInfoToTaus(process)
    addExtendVertexInfo(process)
    return process
