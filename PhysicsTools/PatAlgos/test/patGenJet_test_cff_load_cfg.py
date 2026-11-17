# patGenJets_cff loads; the task expands to the expected modules; every module is built by the
# framework (cmsRun validates each module's parameters against its fillDescriptions when it constructs it: maxEvents 0).
# The tables (PhysicsTools/NanoAOD/python/patGenJetTables_cff.py) are tested in PhysicsTools/NanoAOD/test.
#
#   cmsRun patGenJet_test_cff_load_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process("LOAD")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(0))
process.source = cms.Source("EmptySource")
process.load("PhysicsTools.PatAlgos.producersLayer1.patGenJets_cff")

class Labels:
    def __init__(self): self.l = set()
    def enter(self, m):
        if hasattr(m, 'label_'):
            try: self.l.add(m.label_())
            except Exception: pass
    def leave(self, m): pass

v = Labels(); process.patGenJetsTask.visit(v)
expected = {'NjettinessAK8PuppiForPatGenJets', 'nb1AK8PuppiSoftDropForPatGenJets', 'nb2AK8PuppiSoftDropForPatGenJets',
            'NjettinessAK8SubjetsForPatGenJets', 'nb1AK8PuppiSoftDropSubjetsForPatGenJets', 'nb2AK8PuppiSoftDropSubjetsForPatGenJets',
            'ak8PFJetsPuppiSoftDropMassForPatGenJets', 'patGenJetsAK4', 'patGenJetsAK8',
            'selectedPatGenJetsAK8', 'patGenJetsAK8SoftDropSubjets', 'packedPatGenJetsAK8'}
assert v.l == expected, (v.l ^ expected)
# nothing else was added to the process by loading (prototypes are imported under underscore names)
assert set(list(process.producers_()) + list(process.filters_())) == expected, set(list(process.producers_()) + list(process.filters_())) ^ expected
# labels of the user floats equal the reco ones after stripping
from PhysicsTools.PatAlgos.tools.patGenJetHelpers import STRIP
strip = lambda tag: tag.moduleLabel[:-len(STRIP)] + (":" + tag.productInstanceLabel if tag.productInstanceLabel else "") if tag.moduleLabel.endswith(STRIP) else tag.moduleLabel
fat = sorted(strip(x) for x in process.patGenJetsAK8.userData.userFloats.src)
assert fat == ['NjettinessAK8Puppi:tau1', 'NjettinessAK8Puppi:tau2', 'NjettinessAK8Puppi:tau3', 'NjettinessAK8Puppi:tau4', 'ak8PFJetsPuppiSoftDropMass'], fat
sub = sorted(strip(x) for x in process.patGenJetsAK8SoftDropSubjets.userData.userFloats.src)
assert sub == sorted(['NjettinessAK8Subjets:tau%d' % i for i in (1, 2, 3, 4)] + ['nb%d%s:ecf%s' % (b, 'AK8PuppiSoftDropSubjets', n) for b in (1, 2) for n in ('N2', 'N3')]), sub
grp = sorted(strip(x) for x in process.packedPatGenJetsAK8.algos[0].groomedUserFloats)
assert grp == sorted(['nb%d%s:ecf%s' % (b, 'AK8PuppiSoftDrop', n) for b in (1, 2) for n in ('N2', 'N3')]), grp
assert process.packedPatGenJetsAK8.algos[0].label.value() == 'SoftDropPuppi'
# selector = slimmedGenJetsAK8 cuts
s = process.selectedPatGenJetsAK8
assert (s.cut.value(), s.cutLoose.value(), s.nLoose.value()) == ('pt > 80', 'pt > 10.', 3)
# ECF and N-subjettiness on unweighted gen jets
for m in (process.NjettinessAK8PuppiForPatGenJets, process.nb1AK8PuppiSoftDropForPatGenJets, process.nb2AK8PuppiSoftDropSubjetsForPatGenJets):
    assert m.srcWeights.moduleLabel == ''

process.p = cms.Path(process.patGenJetsTask)
