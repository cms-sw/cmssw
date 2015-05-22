from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from CMGTools.TTHAnalysis.analyzers.ttHSVAnalyzer import matchToGenHadron

class ttHHeavyFlavourHadronAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHHeavyFlavourHadronAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)

    def declareHandles(self):
        super(ttHHeavyFlavourHadronAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(ttHHeavyFlavourHadronAnalyzer,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        if not self.cfg_comp.isMC: return True

        def ref2id(ref):
            return (ref.id().processIndex(), ref.id().productIndex(), ref.key()) if ref else (0,0,0)
        def flav(gp):
            id = abs(gp.pdgId())
            return max((id/1000) % 10, (id/100) % 10)
        def same(gp1,gp2):
            return gp1.pdgId() == gp2.pdgId() and gp1.status() == gp2.status() and abs(gp1.pt()-gp2.pt()) < 1e-4  and abs(gp1.eta()-gp2.eta()) < 1e-4  and abs(gp1.phi()-gp2.phi()) < 1e-4
        def descendent(child, bhadron):
            mom = child.mother(0) if child.numberOfMothers() > 0 else None
            while mom != None:
                if mom.status() != 2 or abs(mom.pdgId()) < 100 and abs(mom.pdgId()) != 15: break
                if same(bhadron,mom):
                    return True
                mom = mom.motherRef() if mom.numberOfMothers() > 0 else None
                if mom == None or mom.isNull() or not mom.isAvailable(): break
            return False

        heavyHadrons = []
        for g in event.genParticles:
            if g.status() != 2 or abs(g.pdgId()) < 100: continue
            myflav = flav(g)
            if myflav not in [4,5]: continue
            lastInChain = True
            for idau in xrange(g.numberOfDaughters()):
                if flav(g.daughter(idau)) == myflav:
                    lastInChain = False
                    break
            if not lastInChain: continue
            if myflav == 4:
                heaviestInChain = True
                mom = g.motherRef() if g.numberOfMothers() > 0 else None
                while mom != None and mom.isNonnull() and mom.isAvailable():
                    if mom.status() != 2 or abs(mom.pdgId()) < 100: break
                    if flav(mom) == 5:
                        heaviestInChain = False
                        break
                    mom = mom.motherRef() if mom.numberOfMothers() > 0 else None
                    if not heaviestInChain: continue
            # OK, here we are
            g.flav = myflav
            heavyHadrons.append(g)

        # if none is found, give up here without going through the rest, so we avoid e.g. mc matching for jets
        if len(heavyHadrons) == 0:
            event.genHeavyHadrons = heavyHadrons
            event.genBHadrons = [ h for h in heavyHadrons if h.flav == 5 ]
            event.genDHadrons = [ h for h in heavyHadrons if h.flav == 4 ]
            return True
 
        # match with IVF 
        had_ivf_pairs = []
        #print "\nNew event"
        for ihad, had in enumerate(heavyHadrons):
            #print "HAD %2d with flav %d,  %d daughters, mass %5.2f, pt %5.2f, eta %+4.2f, phi %+4.2f: " % (ihad, had.flav, had.numberOfDaughters(), had.mass(), had.pt(), had.eta(), had.phi())
            had.sv = None
            for isv,s in enumerate(event.ivf):
                #print "   SV %2d with %d mc tracks, mass %5.2f, pt %5.2f, eta %+4.2f, phi %+4.2f: %d mc-matched tracks" % (isv, s.numberOfDaughters(), s.mass(), s.pt(), s.eta(), s.phi(),len(s.mctracks))
                shared_n, shared_pt = 0, 0 
                for mct in s.mctracks:
                    if descendent(mct,had):
                        shared_n += 1; shared_pt += mct.pt()
                if shared_n:
                    #print "       matched %d tracks (total pt: %.2f) " % (shared_n, shared_pt)
                    had_ivf_pairs.append( (ihad, isv, shared_n, shared_pt) )
        had_ivf_pairs.sort(key = lambda (i1,i2,n,pt) : n + 0.0001*pt, reverse=True)
        for ihad,isv,n,pt in had_ivf_pairs:
            had = heavyHadrons[ihad]
            #print "( had %d, sv %d ): shared %d tracks, %.2f pt ==> %s" % (ihad, isv, n, pt, had.sv)
            if had.sv == None:
                had.sv = event.ivf[isv]
                #print " had %d --> sv %d " % (ihad, isv)
            #else:
            #    print " had %d is already matched " % (ihad,) 
        # match with jets:
        had_jet_pairs = []
        # first loop on jets, get and match daughters
        jetsWithMatchedDaughters = [] 
        for j in event.jetsIdOnly:
            dausWithMatch = []
            for idau in xrange(j.numberOfDaughters()):
                dau = j.daughter(idau)
                if dau.charge() == 0 or abs(dau.eta()) > 2.5: continue
                mct, dr, dpt =  matchToGenHadron(dau, event, minDR=0.05, minDpt=0.1)
                if mct == None: continue
                dausWithMatch.append((dau,mct))
            jetsWithMatchedDaughters.append((j,dausWithMatch))
        for ihad, had in enumerate(heavyHadrons):
            had.jet = None
            for ij,(j,dausWithMatch) in enumerate(jetsWithMatchedDaughters):
                shared_n, shared_pt = 0, 0 
                for dau,mct in dausWithMatch:
                   if descendent(mct,had):
                        shared_n += 1; shared_pt += mct.pt()
                if shared_n:
                    had_jet_pairs.append( (ihad, ij, shared_n, shared_pt) )
        had_jet_pairs.sort(key = lambda (i1,i2,n,pt) : n + 0.0001*pt, reverse=True)
        for ihad,ij,n,pt in had_jet_pairs:
            had = heavyHadrons[ihad]
            if had.jet == None:
                had.jet = event.jetsIdOnly[ij]
        # match with hard scattering
        for had in heavyHadrons:
            had.sourceId = 0
            srcmass = 0
            mom = had.motherRef() if had.numberOfMothers() > 0 else None
            while mom != None and mom.isNonnull() and mom.isAvailable():
                if mom.status() > 2: 
                    if mom.mass() > srcmass:
                        srcmass = mom.mass()
                        had.sourceId = mom.pdgId() 
                    if srcmass > 175:
                        break
                mom = mom.motherRef() if mom.numberOfMothers() > 0 else None
        # sort and save
        heavyHadrons.sort(key = lambda h : h.pt(), reverse=True)
        event.genHeavyHadrons = heavyHadrons
        event.genBHadrons = [ h for h in heavyHadrons if h.flav == 5 ]
        event.genDHadrons = [ h for h in heavyHadrons if h.flav == 4 ]
        #print "Summary: "
        #for had in event.genBHadrons:
        #    print "    HAD with %d daughters, mass %5.2f, pt %5.2f, eta %+4.2f, phi %+4.2f: sv %s, jet %s" % (had.numberOfDaughters(), had.mass(), had.pt(), had.eta(), had.phi(), had.sv != None, had.jet != None)
        return True
