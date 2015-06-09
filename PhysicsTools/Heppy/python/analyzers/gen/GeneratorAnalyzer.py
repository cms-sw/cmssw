from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsutils.genutils import isNotFromHadronicShower, realGenMothers, realGenDaughters

def interestingPdgId(id,includeLeptons=False):        
    id = abs(id)
    return id in [6,7,8,17,18] or (includeLeptons and 11 <= id and id < 16) or (22 <= id and id < 40) or id > 1000000

class GeneratorAnalyzer( Analyzer ):
    """Save the hard-scattering final state of the event: top quarks, gauge & higgs bosons and BSM
       particles, plus their immediate decay products, and their siblings (in order to get the jets
       from matched X+jets generation.
       Incoming partons are not included, by design.

       In the default configuration, leptons, light quarks and gluons are saved before FSR (a la status 3).
       Everything else is saved after any radiation, i.e. immediately before the decay.

       Particles are saved in a list event.generatorSummary, with the index of their mothers ('motherIndex') 
       if the mother is also in the list, and with the pdgId of the mother ('motherId') and grand-mother
       ('grandmotherId'). Particles also carry their index in the miniAOD genparticles collection ('rawIndex')
       In addition, a 'sourceId' is set to the pdgId of the heaviest ancestor (or of the particle itself)
       i.e.  in  top -> W -> lepton: the lepton sourceId will be 6
             in  tt+W with W -> lepton, the sourceId of the lepton will be 24.
       sourceId will be 99 for paricles from hard scattering whose mother is light 

       If requested, the full list of genParticles is also produced in event.genParticles (with indices
       aligned to the miniAOD one). For particles that are in the generatorSummary, the same object is used.
       An extra index 'genSummaryIndex' will be added to all particles, with the index in the generatorSummary
       or -1 if the particle is not in the generatorSummary.

       Also, if requested it creates the splitted collections:
            event.genHiggsBosons = []
            event.genVBosons = []
            event.gennus     = []  # prompt neutrinos
            event.gennusFromTop = []  # Neutrinos from t->W decay
            event.genleps    = []  # leptons from direct decays
            event.gentauleps = []  # leptons from prompt taus
            event.gentaus    = []  # hadronically-decaying taus (if allGenTaus is False) or all taus (if allGenTaus is True)
            event.gentopquarks  = [] 
            event.genbquarks    = [] # b quarks from hard event (e.g. from top decays)
            event.genwzquarks   = [] # quarks from W,Z decays
            event.genbquarksFromTop = []
            event.genbquarksFromH   = []
            event.genlepsFromTop = [] #mu/ele that have a t->W chain as ancestor, also contained in event.genleps
       event.genwzquarks and event.genbquarks, might have overlaps 
       event.genbquarksFromTop and event.genbquarksFromH are all contained in event.genbquarks
       
       In addition to genParticles, if makeLHEweights is set to True, the list WeightsInfo objects of the LHE branch
       is stored in event.LHE_weights
       
       """

    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(GeneratorAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.stableBSMParticleIds  = set(cfg_ana.stableBSMParticleIds) # neutralinos and such
        self.savePreFSRParticleIds = set(cfg_ana.savePreFSRParticleIds)
        self.makeAllGenParticles   = cfg_ana.makeAllGenParticles
        self.makeSplittedGenLists  = cfg_ana.makeSplittedGenLists
        self.allGenTaus            = cfg_ana.allGenTaus if self.makeSplittedGenLists else False
	self.makeLHEweights  = cfg_ana.makeLHEweights
 
    def declareHandles(self):
        super(GeneratorAnalyzer, self).declareHandles()
        self.mchandles['genParticles'] = AutoHandle( 'prunedGenParticles', 'std::vector<reco::GenParticle>' )
	if self.makeLHEweights:
		self.mchandles['LHEweights'] = AutoHandle( 'source', 'LHEEventProduct', mayFail = True, lazy = False )

    def beginLoop(self,setup):
        super(GeneratorAnalyzer,self).beginLoop(setup)

    def makeMCInfo(self, event):
        verbose = getattr(self.cfg_ana, 'verbose', False)
        rawGenParticles = self.mchandles['genParticles'].product() 
        good = []; keymap = {};
        allGenParticles = []
        for rawIndex,p in enumerate(rawGenParticles):
            if self.makeAllGenParticles: allGenParticles.append(p)
            id     = abs(p.pdgId())
            status = p.status()
            # particles must be status > 2, except for prompt leptons, photons, neutralinos
            if status <= 2:
                if ((id not in self.stableBSMParticleIds) and
                    (id not in [11,12,13,14,15,16,22] or not isNotFromHadronicShower(p))):
                        continue
            # a particle must not be decaying into itself
            #print "  test %6d  : %+8d  %3d :  %8.2f   %+5.2f   %+5.2f : %d %d : %+8d {%6d}: %s" % ( rawIndex,
            #        p.pdgId(), p.status(), p.pt(), p.eta(), p.phi(), p.numberOfMothers(), p.numberOfDaughters(), 
            #        p.motherRef(0).pdgId() if p.numberOfMothers() > 0 else -999, p.motherRef(0).key()   if p.numberOfMothers() > 0 else -999, 
            #        "  ".join("%d[%d]" % (p.daughter(i).pdgId(), p.daughter(i).status()) for i in xrange(p.numberOfDaughters())))
            if id in self.savePreFSRParticleIds:
                # for light objects, we want them pre-radiation
                if any((p.mother(j).pdgId() == p.pdgId()) for j in xrange(p.numberOfMothers())):
                    #print "    fail auto-decay"
                    continue
            else:
                # everything else, we want it after radiation, i.e. just before decay
                if any((p.daughter(j).pdgId() == p.pdgId() and p.daughter(j).status() > 2) for j in xrange(p.numberOfDaughters())):
                    #print "    fail auto-decay"
                    continue
            # FIXME find a better criterion to discard there
            if status == 71: 
                #drop QCD radiation with unclear parentage
                continue 
            # is it an interesting particle?
            ok = False
            if interestingPdgId(id):
                #print "    pass pdgId"
                ok = True
            ### no: we don't select by decay, so that we keep the particle summary free of incoming partons and such
            # if not ok and any(interestingPdgId(d.pdgId()) for d in realGenDaughters(p)):
            #    #print "    pass dau"
            #     ok = True
            if not ok:
              for mom in realGenMothers(p):
                if interestingPdgId(mom.pdgId()) or (getattr(mom,'rawIndex',-1) in keymap):
                    #print "    interesting mom"
                    # exclude extra x from p -> p + x
                    if not any(mom.daughter(j2).pdgId() == mom.pdgId() for j2 in xrange(mom.numberOfDaughters())):
                        #print "         pass no-self-decay"
                        ok = True
                    # Account for generator feature with Higgs decaying to itself with same four-vector but no daughters
                    elif mom.pdgId() == 25 and any(mom.daughter(j2).pdgId() == 25 and mom.daughter(j2).numberOfDaughters()==0 for j2 in range(mom.numberOfDaughters())):
                        ok = True
                if abs(mom.pdgId()) == 15:
                    # if we're a tau daughter we're status 2
                    # if we passed all the previous steps, then we're a prompt lepton
                    # so we'd like to be included
                    ok = True
                if not ok and p.pt() > 10 and id in [1,2,3,4,5,21,22] and any(interestingPdgId(d.pdgId()) for d in realGenDaughters(mom)):
                    # interesting for being a parton brother of an interesting particle (to get the extra jets in ME+PS) 
                    ok = True 
            if ok:
                gp = p
                gp.rawIndex = rawIndex # remember its index, so that we can set the mother index later
                keymap[rawIndex] = len(good)
                good.append(gp)
        # connect mother links
        for igp,gp in enumerate(good):
            gp.motherIndex = -1
            gp.sourceId    = 99
            gp.genSummaryIndex = igp
            ancestor = None if gp.numberOfMothers() == 0 else gp.motherRef(0)
            while ancestor != None and ancestor.isNonnull():
                if ancestor.key() in keymap:
                    gp.motherIndex = keymap[ancestor.key()]
                    if ancestor.pdgId() != good[gp.motherIndex].pdgId():
                        print "Error keying %d: motherIndex %d, ancestor.pdgId %d, good[gp.motherIndex].pdgId() %d " % (igp, gp.motherIndex, ancestor.pdgId(),  good[gp.motherIndex].pdgId())
                    break
                ancestor = None if ancestor.numberOfMothers() == 0 else ancestor.motherRef(0)
            if abs(gp.pdgId()) not in [1,2,3,4,5,11,12,13,14,15,16,21]:
                gp.sourceId = gp.pdgId()
            if gp.motherIndex != -1:
                ancestor = good[gp.motherIndex]
                if ancestor.sourceId != 99 and (ancestor.mass() > gp.mass() or gp.sourceId == 99):
                    gp.sourceId = ancestor.sourceId
        event.generatorSummary = good
        # add the ID of the mother to be able to recreate last decay chains
        for ip,p in enumerate(good):
            moms = realGenMothers(p)
            if len(moms)==0:
                p.motherId = 0
                p.grandmotherId = 0
            elif len(moms)==1:
                p.motherId = moms[0].pdgId()
                gmoms = realGenMothers(moms[0])
                p.grandmotherId = (gmoms[0].pdgId() if len(gmoms)==1 else (0 if len(gmoms)==0 else -9999))
            else:
                #print "    unclear what mothers to give to this particle, among ","  ".join("%d[%d]" % (m.pdgId(),m.status()) for m in moms)
                p.motherId = -9999
                p.grandmotherId = -9999
            if verbose:
                print "%3d  {%6d}: %+8d  %3d :  %8.2f   %+5.2f   %+5.2f : %d %2d : %+8d {%3d}: %s" % ( ip,p.rawIndex,
                        p.pdgId(), p.status(), p.pt(), p.eta(), p.phi(), len(moms), p.numberOfDaughters(), 
                        p.motherId, p.motherIndex,
                        "  ".join("%d[%d]" % (p.daughter(i).pdgId(), p.daughter(i).status()) for i in xrange(p.numberOfDaughters())))
        if verbose:
            print "\n\n"

        if self.makeAllGenParticles:
            event.genParticles = allGenParticles

        if self.makeSplittedGenLists:
            event.genHiggsBosons = []
            event.genVBosons     = []
            event.gennus         = []
            event.gennusFromTop  = []
            event.genleps        = []
            event.gentauleps     = []
            event.gentaus        = []
            event.gentopquarks   = []
            event.genbquarks     = []
            event.genwzquarks    = []
            event.genbquarksFromTop = []
            event.genbquarksFromH   = []
            event.genlepsFromTop = []
            for p in event.generatorSummary:
                id = abs(p.pdgId())
                if id == 25: 
                    event.genHiggsBosons.append(p)
                elif id in {23,24}:
                    event.genVBosons.append(p)
                elif id in {12,14,16}:
                    event.gennus.append(p)

                    momids = [(m, abs(m.pdgId())) for m in realGenMothers(p)]

                    #have a look at the lepton mothers
                    for mom, momid in momids:
                        #lepton from W
                        if momid == 24:
                            wmomids = [abs(m.pdgId()) for m in realGenMothers(mom)]
                            #W from t
                            if 6 in wmomids:
                                #save mu,e from t->W->mu/e
                                event.gennusFromTop.append(p)

                elif id in {11,13}:
                    #taus to separate vector
                    if abs(p.motherId) == 15:
                        event.gentauleps.append(p)
                    #all muons and electrons
                    else:
                        event.genleps.append(p)
                        momids = [(m, abs(m.pdgId())) for m in realGenMothers(p)]

                        #have a look at the lepton mothers
                        for mom, momid in momids:
                            #lepton from W
                            if momid == 24:
                                wmomids = [abs(m.pdgId()) for m in realGenMothers(mom)]
                                #W from t
                                if 6 in wmomids:
                                    #save mu,e from t->W->mu/e
                                    event.genlepsFromTop.append(p)
                elif id == 15:
                    if self.allGenTaus or not any([abs(d.pdgId()) in {11,13} for d in realGenDaughters(p)]):
                        event.gentaus.append(p)
                elif id == 6:
                    event.gentopquarks.append(p)
                elif id == 5:
                    event.genbquarks.append(p)
                    momids = [abs(m.pdgId()) for m in realGenMothers(p)]
                    if  6 in momids: event.genbquarksFromTop.append(p)
                    if 25 in momids: event.genbquarksFromH.append(p)
                if id <= 5 and any([abs(m.pdgId()) in {23,24} for m in realGenMothers(p)]):
                    event.genwzquarks.append(p)

        #Add LHE weight info
	event.LHE_weights = []
	if self.makeLHEweights:
	    if self.mchandles['LHEweights'].isValid():
	    	for w in self.mchandles['LHEweights'].product().weights():
	        	event.LHE_weights.append(w)

    def process(self, event):
        self.readCollections( event.input )

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True
        # do MC level analysis
        self.makeMCInfo(event)
        return True

import PhysicsTools.HeppyCore.framework.config as cfg
setattr(GeneratorAnalyzer,"defaultConfig",
    cfg.Analyzer(GeneratorAnalyzer,
        # BSM particles that can appear with status <= 2 and should be kept
        stableBSMParticleIds = [ 1000022 ], 
        # Particles of which we want to save the pre-FSR momentum (a la status 3).
        # Note that for quarks and gluons the post-FSR doesn't make sense,
        # so those should always be in the list
        savePreFSRParticleIds = [ 1,2,3,4,5, 11,12,13,14,15,16, 21 ],
        # Make also the list of all genParticles, for other analyzers to handle
        makeAllGenParticles = True,
        # Make also the splitted lists
        makeSplittedGenLists = True,
        allGenTaus = False, 
        # Save LHE weights in LHEEventProduct
        makeLHEweights = True,
        # Print out debug information
        verbose = False,
    )
)
