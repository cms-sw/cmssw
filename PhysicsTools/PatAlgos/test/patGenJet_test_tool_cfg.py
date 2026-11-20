# addPatGenJetCollection on the generated sample. Configuration asserts at load time, then the tool-made collections run next to
# the ones of patGenJets_cff; patGenJet_check_tool_output.py compares them.
#
#   cmsRun patGenJet_test_tool_cfg.py input=ttbar20_gen.root output=tool.root
#   cmsRun patGenJet_test_tool_cfg.py input=ttbar20_gen.root output=embed.root embed=True   (patGenJetsAK4 with embedGenCandidates)
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing("analysis")
options.register("input", "ttbar20_gen.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "generated sample")
options.register("output", "tool.root", VarParsing.multiplicity.singleton, VarParsing.varType.string, "EDM output")
options.register("embed", False, VarParsing.multiplicity.singleton, VarParsing.varType.bool,
                 "patGenJetsAK4: embedGenCandidates=True with the forward maps prunedGenParticlesWithStatusOne, prunedGenParticles")
options.parseArguments()

process = cms.Process("TOOL")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring("file:" + options.input))
process.load("PhysicsTools.PatAlgos.producersLayer1.patGenJets_cff")
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask
task = getPatAlgosToolsTask(process)
task.add(process.patGenJetsTask)
if options.embed:
    process.patGenJetsAK4.embedGenCandidates = True
    process.patGenJetsAK4.genCandidatesForwardMaps = cms.VInputTag("prunedGenParticlesWithStatusOne", "prunedGenParticles")

before = {l: m.dumpPython() for d in (process.producers_(), process.filters_()) for l, m in d.items()}

from PhysicsTools.PatAlgos.tools.patGenJetTools import addPatGenJetCollection
# postfix Tool: the modules of patGenJets_cff already use the labels without postfix; the user-data labels are the same after stripping
addPatGenJetCollection(process, labelName='AK8Puppi', genJetSource='ak8GenJetsNoNu', rParam=0.8,
                       groomedSource='ak8GenJetsNoNuSoftDrop', postfix='Tool')
process.selectedPatGenJetsAK8PuppiTool.cut = 'pt > 80'
process.selectedPatGenJetsAK8PuppiTool.cutLoose = 'pt > 10.'
process.selectedPatGenJetsAK8PuppiTool.nLoose = 3
addPatGenJetCollection(process, 'AK4Tool', 'ak4GenJetsNoNu', 0.4, substructure=False)
addPatGenJetCollection(process, labelName='AK8Puppi', genJetSource='ak8GenJetsNoNu', rParam=0.8,
                       groomedSource='ak8GenJetsNoNuSoftDrop', postfix='Test', substructure=False)

# ---- configuration asserts
after = {l: m.dumpPython() for d in (process.producers_(), process.filters_()) for l, m in d.items()}
assert all(after[l] == before[l] for l in before), "the tool modified an existing module"
new = sorted(set(after) - set(before))
expected = sorted(n + 'Tool' for n in [
    'NjettinessAK8PuppiForPatGenJets', 'groomedMassAK8PuppiForPatGenJets', 'nb1AK8PuppiSoftDropForPatGenJets',
    'nb2AK8PuppiSoftDropForPatGenJets', 'nb1AK8PuppiSoftDropSubjetsForPatGenJets', 'nb2AK8PuppiSoftDropSubjetsForPatGenJets',
    'NjettinessAK8PuppiSubjetsForPatGenJets', 'patGenJetsAK8Puppi',
    'patGenJetsAK8PuppiSubjets', 'selectedPatGenJetsAK8Puppi', 'packedPatGenJetsAK8Puppi']) + [
    'patGenJetsAK4Tool',
    'patGenJetsAK8PuppiTest', 'patGenJetsAK8PuppiSubjetsTest', 'selectedPatGenJetsAK8PuppiTest', 'packedPatGenJetsAK8PuppiTest']
expected = sorted(expected)
assert new == expected, "modules added by the tool: %s, expected %s" % (new, expected)
tags = lambda v: [t.value() for t in v]
p = process.patGenJetsAK8PuppiTool
assert tags(p.userData.userFloats.src) == ['NjettinessAK8PuppiForPatGenJetsTool:tau1', 'NjettinessAK8PuppiForPatGenJetsTool:tau2',
    'NjettinessAK8PuppiForPatGenJetsTool:tau3', 'NjettinessAK8PuppiForPatGenJetsTool:tau4', 'groomedMassAK8PuppiForPatGenJetsTool'], tags(p.userData.userFloats.src)
assert list(p.userData.userFloats.labelPostfixesToStrip) == ['ForPatGenJetsTool']
assert process.patGenJetsAK8PuppiSubjetsTool.jetSource.value() == 'ak8GenJetsNoNuSoftDrop:SubJets'
k = process.packedPatGenJetsAK8PuppiTool
assert k.jetSrc.value() == 'selectedPatGenJetsAK8PuppiTool' and k.algos[0].label.value() == 'SoftDropPuppi'
assert tags(k.algos[0].groomedUserFloats) == ['nb1AK8PuppiSoftDropForPatGenJetsTool:ecfN2', 'nb1AK8PuppiSoftDropForPatGenJetsTool:ecfN3',
    'nb2AK8PuppiSoftDropForPatGenJetsTool:ecfN2', 'nb2AK8PuppiSoftDropForPatGenJetsTool:ecfN3']
assert len(process.patGenJetsAK4Tool.userData.userFloats.src) == 0 and not hasattr(process, 'NjettinessAK4ToolForPatGenJets')
assert list(process.packedPatGenJetsAK8PuppiTest.labelPostfixesToStrip) == ['ForPatGenJetsTest']
for l in new:
    assert hasattr(process, l)
# all new modules are in the PAT tools task
inTask = set()
class V:
    def enter(self, m):
        if hasattr(m, 'label_'):
            try: inTask.add(m.label_())
            except Exception: pass
    def leave(self, m): pass
task.visit(V())
assert set(new) <= inTask, set(new) - inTask
# errors
for kw, msg in ((dict(labelName='AK8Puppi', genJetSource='ak8GenJetsNoNu', groomedSource='ak8GenJetsNoNuSoftDrop', postfix='Tool'), 'clash'),
                (dict(labelName='X1', genJetSource='ak8GenJetsNoNu', flavour=False), 'removed parameter flavour'),
                (dict(labelName='X2'), 'source')):
    try:
        addPatGenJetCollection(process, **kw)
    except (ValueError, KeyError, Exception) as e:
        pass
    else:
        raise AssertionError("no error for " + msg)

process.p = cms.Path(task)
keep = ["drop *", "keep *_patGenJets*_*_*", "keep *_packedPatGenJets*_*_*", "keep *_selectedPatGenJets*_*_*", "keep *_packedGenParticles_*_*", "keep *_ak4GenJetsNoNu_*_*"]
if options.embed:  # the products the forward Ptrs point into
    keep += ["keep *_prunedGenParticles_*_*", "keep *_prunedGenParticlesWithStatusOne_*_*"]
process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string(options.output), outputCommands=cms.untracked.vstring(*keep))
process.e = cms.EndPath(process.out)
