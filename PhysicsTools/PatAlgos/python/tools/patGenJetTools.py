"""Tool function to add a pat::GenJet collection, in the style of PhysicsTools/PatAlgos/python/tools/jetTools.py
(class AddJetCollection(ConfigToolBase) with a module-level instance).

    from PhysicsTools.PatAlgos.tools.patGenJetTools import addPatGenJetCollection
    addPatGenJetCollection(process, labelName='AK8Puppi', genJetSource='ak8GenJetsNoNu', rParam=0.8,
                           groomedSource='ak8GenJetsNoNuSoftDrop')

Importing this file changes nothing in any process: the instance only configures a process when it is called.
The tool only ADDS modules (named below, all new); it never modifies an existing module. Later, jetToolbox calls it
once per reco collection.

Modules created (postfix P, labelName L; "ForPatGenJets" + P ends every helper label, the producers strip it, so the user-data
labels are the ones in the right-hand column and a table that reads the reco labels finds them):

  patGenJets<L><P>                       PATGenJetProducer on genJetSource          Njettiness<L>:tau1..4, groomedMass<L>
  Njettiness<L>ForPatGenJets<P>          NjettinessAdder (substructure)
  groomedMass<L>ForPatGenJets<P>         RecoJetDeltaRValueMapProducer (groomedSource given)
  and, when groomedSource is given:
  nb1<L><G>ForPatGenJets<P>, nb2<L><G>…  ECFAdder on the groomed jets              nb1<L><G>:ecfN2/N3 (copied to the fat jet)
  Njettiness<L>SubjetsForPatGenJets<P>   NjettinessAdder on the subjets            Njettiness<L>Subjets:tau1..4
  nb1<L><G>SubjetsForPatGenJets<P>, nb2… ECFAdder on the subjets                   nb1<L><G>Subjets:ecfN2/N3
  patGenJets<L>Subjets<P>                PATGenJetProducer on the subjets
  selectedPatGenJets<L><P>               PATGenJetSelector (cut '': everything; set .cut, .cutLoose, .nLoose afterwards)
  packedPatGenJets<L><P>                 PATGenJetSubstructurePacker: fat jets with subjets(subjetLabel); subjets in :SubJets
  (G = groomedName)

All modules are added to the PAT tools task.
"""
import copy
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.ConfigToolBase import ConfigToolBase
import FWCore.ParameterSet.DictTypes as dicttypes
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask


class AddPatGenJetCollection(ConfigToolBase):
    """Tool to add a pat::GenJet collection (and, for fat jets, its subjets and substructure) to a process."""
    _label = 'addPatGenJetCollection'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__(self):
        ConfigToolBase.__init__(self)
        self.addParameter(self._defaultParameters, 'labelName', 'UNDEFINED', "Label name of the new collection (patGenJets<labelName>).", str)
        self.addParameter(self._defaultParameters, 'genJetSource', cms.InputTag(''), "Input View<reco::GenJet> (e.g. ak8GenJetsNoNu).", cms.InputTag)
        self.addParameter(self._defaultParameters, 'rParam', 0.4, "Distance parameter of the input jets (N-subjettiness R0).")
        self.addParameter(self._defaultParameters, 'groomedSource', None, "Groomed jets (e.g. ak8GenJetsNoNuSoftDrop), written with their subjets; enables selector and packer.", cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'subjetSource', None, "Subjets of the groomed jets; default groomedSource:SubJets.", cms.InputTag, acceptNoneValue=True)
        self.addParameter(self._defaultParameters, 'substructure', True, "Add N-subjettiness (jets, subjets), energy correlation functions and groomed mass.")
        self.addParameter(self._defaultParameters, 'packedGenParticles', cms.InputTag('packedGenParticles'), "Association to the packed gen particles for the constituents; empty = no re-keying.", cms.InputTag)
        self.addParameter(self._defaultParameters, 'postfix', '', "Postfix of every module label.", str)
        self.addParameter(self._defaultParameters, 'groomedName', 'SoftDrop', "Name of the grooming in the labels (nb1<labelName><groomedName>).", str)
        self.addParameter(self._defaultParameters, 'subjetLabel', 'SoftDropPuppi', "Label of the subjet collection on the fat jets, subjets(label).", str)
        self._parameters = copy.deepcopy(self._defaultParameters)
        self._comment = "Tool to add a pat::GenJet collection: producer, optional soft-drop subjets and substructure."

    def getDefaultParameters(self):
        return self._defaultParameters

    def __call__(self, process, labelName=None, genJetSource=None, rParam=None, groomedSource=None, subjetSource=None,
                 substructure=None, packedGenParticles=None, postfix=None, groomedName=None, subjetLabel=None):
        values = dict(labelName=labelName, genJetSource=genJetSource, rParam=rParam, groomedSource=groomedSource,
                      subjetSource=subjetSource, substructure=substructure,
                      packedGenParticles=packedGenParticles, postfix=postfix, groomedName=groomedName, subjetLabel=subjetLabel)
        for name, value in values.items():
            if value is None:
                value = self._defaultParameters[name].value
            if isinstance(value, str) and name in ('genJetSource', 'groomedSource', 'subjetSource', 'packedGenParticles'):
                value = cms.InputTag(value)
            self.setParameter(name, value)
        self.apply(process)

    def toolCode(self, process):
        p = {k: v.value for k, v in self._parameters.items()}
        L, P = p['labelName'], p['postfix']
        if L == 'UNDEFINED':
            raise ValueError("addPatGenJetCollection: labelName is required")
        if not p['genJetSource'].moduleLabel:
            raise ValueError("addPatGenJetCollection: genJetSource is required")
        task = getPatAlgosToolsTask(process)
        G = p['groomedName']
        STRIP = 'ForPatGenJets' + P
        fat = p['groomedSource'] is not None
        genJets = p['genJetSource']
        packed = p['packedGenParticles']
        groomed = p['groomedSource']
        subjets = (p['subjetSource'] if p['subjetSource'] is not None else
                   (cms.InputTag(groomed.moduleLabel, 'SubJets') if fat else None))
        # names, all new
        names = dict(jets='patGenJets%s%s' % (L, P), tau='Njettiness%s%s%s' % (L, STRIP, ''))
        for n in list(names.values()) + (['selectedPatGenJets%s%s' % (L, P)] if fat else []):
            if hasattr(process, n):
                raise ValueError("addPatGenJetCollection: the process already has a module %s (labelName/postfix clash)" % n)

        from PhysicsTools.PatAlgos.patGenJetProducer_cfi import patGenJetProducer
        from PhysicsTools.PatAlgos.tools.patGenJetHelpers import userDataPSet

        floats, subFloats, groomedFloats = [], [], []
        if p['substructure']:
            from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness
            from RecoJets.JetProducers.ECF_cff import ecfNbeta1, ecfNbeta2
            tauLabel = 'Njettiness%s%s' % (L, STRIP)
            addToProcessAndTask(tauLabel, Njettiness.clone(src=genJets, R0=p['rParam'], srcWeights=''), process, task)
            floats += [cms.InputTag(tauLabel, 'tau%d' % i) for i in (1, 2, 3, 4)]
            if fat:
                from RecoJets.JetProducers.ECF_cff import ecfNbeta1, ecfNbeta2
                massLabel = 'groomedMass%s%s' % (L, STRIP)
                addToProcessAndTask(massLabel, cms.EDProducer("RecoJetDeltaRValueMapProducer", src=genJets, matched=groomed,
                                                              distMax=cms.double(0.8), value=cms.string('mass')), process, task)
                floats.append(cms.InputTag(massLabel))
                for b, proto in (('nb1', ecfNbeta1), ('nb2', ecfNbeta2)):
                    lab = '%s%s%s%s' % (b, L, G, STRIP)
                    addToProcessAndTask(lab, proto.clone(src=groomed, srcWeights='', cuts=cms.vstring('', '', 'pt > 250')), process, task)
                    groomedFloats += [cms.InputTag(lab, 'ecfN2'), cms.InputTag(lab, 'ecfN3')]
                    slab = '%s%s%sSubjets%s' % (b, L, G, STRIP)
                    addToProcessAndTask(slab, proto.clone(src=subjets, srcWeights=''), process, task)
                    subFloats += [cms.InputTag(slab, 'ecfN2'), cms.InputTag(slab, 'ecfN3')]
                stab = 'Njettiness%sSubjets%s' % (L, STRIP)
                addToProcessAndTask(stab, Njettiness.clone(src=subjets, R0=p['rParam'], srcWeights=''), process, task)
                subFloats += [cms.InputTag(stab, 'tau%d' % i) for i in (1, 2, 3, 4)]

        common = dict(packedGenParticles=packed)
        addToProcessAndTask(names['jets'], patGenJetProducer.clone(
            jetSource=genJets, userData=userDataPSet(floats=floats, strip=(STRIP,)), **common),
            process, task)
        if not fat:
            return
        subLabel = 'patGenJets%sSubjets%s' % (L, P)
        addToProcessAndTask(subLabel, patGenJetProducer.clone(
            jetSource=subjets, userData=userDataPSet(floats=subFloats, strip=(STRIP,)), **common),
            process, task)
        from PhysicsTools.PatAlgos.patGenJetSelector_cfi import patGenJetSelector
        from PhysicsTools.PatAlgos.patGenJetSubstructurePacker_cfi import patGenJetSubstructurePacker
        selLabel = 'selectedPatGenJets%s%s' % (L, P)
        addToProcessAndTask(selLabel, patGenJetSelector.clone(src=names['jets'], cut='', cutLoose='', nLoose=0), process, task)
        addToProcessAndTask('packedPatGenJets%s%s' % (L, P), patGenJetSubstructurePacker.clone(
            jetSrc=selLabel, distMax=0.8, fixDaughters=False, labelPostfixesToStrip=[STRIP],
            algos=cms.VPSet(cms.PSet(label=cms.string(p['subjetLabel']), groomedJets=groomed, subjets=cms.InputTag(subLabel),
                                      groomedUserFloats=cms.VInputTag(*groomedFloats), subjetInstance=cms.string('SubJets')))),
            process, task)


addPatGenJetCollection = AddPatGenJetCollection()
