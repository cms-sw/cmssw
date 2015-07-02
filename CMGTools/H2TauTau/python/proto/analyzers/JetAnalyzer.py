import random

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet, GenJet

from PhysicsTools.HeppyCore.utils.deltar import cleanObjectCollection, matchObjectCollection
from PhysicsTools.HeppyCore.utils.deltar import deltaR2

from PhysicsTools.Heppy.physicsutils.BTagSF import BTagSF
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import GenParticle
from PhysicsTools.Heppy.utils.cmsswRelease import isNewerThan

# JAN: Kept this version of the jet analyzer in the tau-tau sequence
# for now since it has all the agreed-upon features used in the tau-tau group,
# in particular the SF seeding for b-tagging.
# In the long run, it might be a good idea to switch to the generic jet analyzer
# in heppy and possibly add b-tagging in another step or add it to the generic
# jet analyzer


class JetAnalyzer(Analyzer):

    """Analyze jets.

    Copied from heppy examples and edit to not rely on heppy examples.

    This analyzer filters the jets that do not correspond to the leptons
    stored in event.selectedLeptons, and puts in the event:
    - jets: all jets passing the pt and eta cuts
    - cleanJets: the collection of jets away from the leptons
    - cleanBJets: the jets passing testBJet, and away from the leptons

    Example configuration:

    jetAna = cfg.Analyzer(
      'JetAnalyzer',
      jetCol = 'slimmedJets'
      # cmg jet input collection
      # pt threshold
      jetPt = 30,
      # eta range definition
      jetEta = 5.0,
      # seed for the btag scale factor
      btagSFseed = 0xdeadbeef,
      # if True, the PF and PU jet ID are not applied, and the jets get flagged
      relaxJetId = False,
    )
    """

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(JetAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.btagSF = BTagSF(0)
        self.is2012 = isNewerThan('CMSSW_5_2_0')

    def declareHandles(self):
        super(JetAnalyzer, self).declareHandles()

        self.handles['jets'] = AutoHandle(self.cfg_ana.jetCol,
                                          'std::vector<pat::Jet>')
        if self.cfg_comp.isMC:
            self.mchandles['genParticles'] = AutoHandle('packedGenParticles',
                                                        'std::vector<pat::PackedGenParticle>')
            self.mchandles['genJets'] = AutoHandle('slimmedGenJets',
                                                   'std::vector<reco::GenJet>')

    def beginLoop(self, setup):
        super(JetAnalyzer, self).beginLoop(setup)
        self.counters.addCounter('jets')
        count = self.counters.counter('jets')
        count.register('all events')
        count.register('at least 2 good jets')
        count.register('at least 2 clean jets')
        count.register('at least 1 b jet')
        count.register('at least 2 b jets')

    def process(self, event):

        self.readCollections(event.input)
        miniaodjets = self.handles['jets'].product()

        allJets = []
        event.jets = []
        event.bJets = []
        event.cleanJets = []
        event.cleanBJets = []

        leptons = []
        if hasattr(event, 'selectedLeptons'):
            leptons = event.selectedLeptons

        genJets = None
        if self.cfg_comp.isMC:
            genJets = map(GenJet, self.mchandles['genJets'].product())

        for maodjet in miniaodjets:
            jet = Jet(maodjet)
            allJets.append(jet)
            if self.cfg_comp.isMC and hasattr(self.cfg_comp, 'jetScale'):
                scale = random.gauss(self.cfg_comp.jetScale,
                                     self.cfg_comp.jetSmear)
                jet.scaleEnergy(scale)
            if genJets:
                # Use DeltaR = 0.25 matching like JetMET
                pairs = matchObjectCollection([jet], genJets, 0.25 * 0.25)
                if pairs[jet] is None:
                    pass
                else:
                    jet.matchedGenJet = pairs[jet]
            # Add JER correction for MC jets. Requires gen-jet matching.
            if self.cfg_comp.isMC and hasattr(self.cfg_ana, 'jerCorr') and self.cfg_ana.jerCorr:
                self.jerCorrection(jet)
            # Add JES correction for MC jets.
            if self.cfg_comp.isMC and hasattr(self.cfg_ana, 'jesCorr'):
                self.jesCorrection(jet, self.cfg_ana.jesCorr)
            if self.testJet(jet):
                event.jets.append(jet)
            if self.testBJet(jet):
                event.bJets.append(jet)

        self.counters.counter('jets').inc('all events')

        event.cleanJets, dummy = cleanObjectCollection(event.jets,
                                                       masks=leptons,
                                                       deltaRMin=0.5)
        event.cleanBJets, dummy = cleanObjectCollection(event.bJets,
                                                        masks=leptons,
                                                        deltaRMin=0.5)

        pairs = matchObjectCollection(leptons, allJets, 0.5 * 0.5)
        # associating a jet to each lepton
        for lepton in leptons:
            jet = pairs[lepton]
            if jet is None:
                lepton.jet = lepton
            else:
                lepton.jet = jet

        # associating a leg to each clean jet
        invpairs = matchObjectCollection(event.cleanJets, leptons, 99999.)
        for jet in event.cleanJets:
            leg = invpairs[jet]
            jet.leg = leg

        for jet in event.cleanJets:
            jet.matchGenParton = 999.0

        if self.cfg_comp.isMC and "BB" in self.cfg_comp.name:
            genParticles = self.mchandles['genParticles'].product()
            event.genParticles = map(GenParticle, genParticles)
            for gen in genParticles:
                if abs(gen.pdgId()) == 5 and gen.mother() and abs(gen.mother().pdgId()) == 21:
                    for jet in event.cleanJets:
                        dR = deltaR2(jet.eta(), jet.phi(), gen.eta(), gen.phi())
                        if dR < jet.matchGenParton:
                            jet.matchGenParton = dR

        event.jets30 = [jet for jet in event.jets if jet.pt() > 30]
        event.cleanJets30 = [jet for jet in event.cleanJets if jet.pt() > 30]
        if len(event.jets30) >= 2:
            self.counters.counter('jets').inc('at least 2 good jets')
        if len(event.cleanJets30) >= 2:
            self.counters.counter('jets').inc('at least 2 clean jets')
        if len(event.cleanBJets) > 0:
            self.counters.counter('jets').inc('at least 1 b jet')
            if len(event.cleanBJets) > 1:
                self.counters.counter('jets').inc('at least 2 b jets')
        return True

    def jerCorrection(self, jet):
        ''' Adds JER correction according to first method at
        https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution

        Requires some attention when genJet matching fails.
        '''
        if not hasattr(jet, 'matchedGenJet'):
            return
        #import pdb; pdb.set_trace()
        corrections = [0.052, 0.057, 0.096, 0.134, 0.288]
        maxEtas = [0.5, 1.1, 1.7, 2.3, 5.0]
        eta = abs(jet.eta())
        for i, maxEta in enumerate(maxEtas):
            if eta < maxEta:
                pt = jet.pt()
                deltaPt = (pt - jet.matchedGenJet.pt()) * corrections[i]
                totalScale = (pt + deltaPt) / pt

                if totalScale < 0.:
                    totalScale = 0.
                jet.scaleEnergy(totalScale)
                break

    def jesCorrection(self, jet, scale=0.):
        ''' Adds JES correction in number of sigmas (scale)
        '''
        # Do nothing if nothing to change
        if scale == 0.:
            return
        unc = jet.uncOnFourVectorScale()
        totalScale = 1. + scale * unc
        if totalScale < 0.:
            totalScale = 0.
        jet.scaleEnergy(totalScale)

    def testJetID(self, jet):
        jet.puJetIdPassed = jet.puJetId()
        jet.pfJetIdPassed = jet.jetID("POG_PFID_Loose")
        if self.cfg_ana.relaxJetId:
            return True
        else:
            return jet.puJetIdPassed and jet.pfJetIdPassed

    def testJet(self, jet):
        return jet.pt() > self.cfg_ana.jetPt and \
            abs( jet.eta() ) < self.cfg_ana.jetEta and \
            self.testJetID(jet)

    def testBJet(self, jet):
        # medium csv working point
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagPerformanceOP#B_tagging_Operating_Points_for_3
        jet.btagMVA = jet.btag('combinedInclusiveSecondaryVertexV2BJetTags')
        jet.btagFlag = jet.btagWP('CSVv2IVFM')
        # jet.btagFlag = self.btagSF.isbtagged(
        #     jet.pt(),
        #     jet.eta(),
        #     jet.btag("combinedInclusiveSecondaryVertexV2BJetTags"),
        #     abs(jet.partonFlavour()),
        #     not self.cfg_comp.isMC,
        #     0, 0,
        #     self.is2012
        # )
        return jet.pt() > 20 and \
            abs(jet.eta()) < 2.4 and \
            jet.btagFlag and \
            self.testJetID(jet)
