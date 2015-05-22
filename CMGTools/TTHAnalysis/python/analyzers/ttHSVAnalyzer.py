import os
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from CMGTools.TTHAnalysis.signedSip import SignedImpactParameterComputer
from CMGTools.TTHAnalysis.tools.SVMVA import SVMVA
from PhysicsTools.HeppyCore.utils.deltar import deltaR

def matchToGenHadron(particle, event, minDR=0.05, minDpt=0.1):
        match = ( None, minDR, 2 )
        myeta, myphi = particle.eta(), particle.phi()
        # now, we don't loop over all the packed gen candidates, but rather do fast bisection to find the rightmost with eta > particle.eta() - 0.05
        etacut = myeta - match[1]
        ileft, iright = 0, len(event.packedGenForHadMatch)
        while iright - ileft > 1:
            imid = (iright + ileft)/2
            if event.packedGenForHadMatch[imid][0] > etacut:
                iright = imid
            else:
                ileft = imid
        # now scan from imid to the end (but stop much earlier)
        etacut = myeta +  match[1]
        for i in xrange(ileft,len(event.packedGenForHadMatch)):
            (eta,phi,pg) = event.packedGenForHadMatch[i]
            if eta > etacut: break
            dr  = deltaR(myeta,myphi,eta,phi)
            if dr > match[1]: continue
            if pg.charge() != particle.charge(): continue
            dpt = abs(particle.pt() - pg.pt())/(particle.pt()+pg.pt())
            if pg.pt() > 10: dpt /= 2; # scale down 
            if dpt < minDpt:
                match = ( pg, dr, dpt )
        return match 

class ttHSVAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHSVAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
	self.SVMVA = SVMVA("%s/src/CMGTools/TTHAnalysis/data/btag/ivf/%%s_BDTG.weights.xml" % os.environ['CMSSW_BASE'])

    def declareHandles(self):
        super(ttHSVAnalyzer, self).declareHandles()
        self.handles['ivf'] = AutoHandle( ('slimmedSecondaryVertices',''),'std::vector<reco::VertexCompositePtrCandidate>')
        self.mchandles['packedGen'] = AutoHandle( 'packedGenParticles', 'std::vector<pat::PackedGenParticle>' )

    def beginLoop(self, setup):
        super(ttHSVAnalyzer,self).beginLoop(setup)

       
    def process(self, event):
        self.readCollections( event.input )

        #get all vertices from IVF
        allivf = [ v for v in self.handles['ivf'].product() ]
       
        # attach distances to PV
        pv = event.goodVertices[0] if len(event.goodVertices)>0 else event.vertices[0]
        for sv in allivf:
             #sv.dz  = SignedImpactParameterComputer.vertexDz(sv, pv)
             sv.dxy = SignedImpactParameterComputer.vertexDxy(sv, pv)
             sv.d3d = SignedImpactParameterComputer.vertexD3d(sv, pv)
             sv.cosTheta = SignedImpactParameterComputer.vertexDdotP(sv, pv)
	     sv.mva = self.SVMVA(sv)
	     svtracks = []
	     for id in xrange(sv.numberOfDaughters()):
                  dau = sv.daughter(id)
                  dau.sip3d = SignedImpactParameterComputer.signedIP3D(dau.pseudoTrack(), pv, sv.momentum()).significance()
		  svtracks.append(dau)	     		     
	     svtracks.sort(key = lambda t : abs(t.dxy()), reverse = True)	     
	     sv.maxDxyTracks = svtracks[0].dxy() if len(svtracks) > 0 else -99
	     sv.secDxyTracks = svtracks[1].dxy() if len(svtracks) > 1 else -99
	     svtracks.sort(key = lambda t : t.sip3d, reverse = True)	     
	     sv.maxD3dTracks = svtracks[0].sip3d if len(svtracks) > 0 else -99
	     sv.secD3dTracks = svtracks[1].sip3d if len(svtracks) > 1 else -99
	     

        event.ivf = allivf

	if self.cfg_comp.isMC and self.cfg_ana.do_mc_match:
            event.packedGenForHadMatch = [ (p.eta(),p.phi(),p) for p in self.mchandles['packedGen'].product() if p.charge() != 0 and abs(p.eta()) < 2.7 ]
            event.packedGenForHadMatch.sort(key = lambda (e,p,x) : e)
            for s in event.ivf:
                #print "SV with %d tracks, mass %5.2f, pt %5.2f, eta %+4.2f, phi %+4.2f: " % (s.numberOfDaughters(), s.mass(), s.pt(), s.eta(), s.phi())
                mctracks, matchable, matched = 0, 0, 0
                s.mctracks = []
                ancestors = {}
                for id in xrange(s.numberOfDaughters()):
                    dau = s.daughter(id)
                    #print "  daughter track with pt %6.3f, eta %+5.3f, phi %+5.3f, dxy %+6.4f, dz %+6.4f" % (dau.pt(), dau.eta(), dau.phi(), dau.dxy(), dau.dz())
                    dau.match = matchToGenHadron(dau, event, minDR=0.05, minDpt=0.1)
                    if dau.match[0]: 
                        s.mctracks.append(dau.match[0])
                        mctracks += 1
                        # print "     \--> gen cand pdgId %+6d with pt %6.3f, eta %+5.3f, phi %+5.3f: dr = %.3f, dptrel = %.4f" % (dau.match[0].pdgId(), dau.match[0].pt(), dau.match[0].eta(), dau.match[0].phi(), dau.match[1], dau.match[2])
                        # ancestry
                        mom = dau.match[0].mother(0)
                        depth = 1; found = False
                        while mom:
                            id = abs(mom.pdgId())
                            flav = max((id/1000) % 10, (id/100) % 10)
                            key = (mom.pt(), mom.eta(), mom.phi())
                            # print "     \---> mother pdgId %+6d, flav %d (pt %6.3f, eta %+5.3f, phi %+5.3f) at %s" % (mom.pdgId(), mom.pt(), mom.eta(), mom.phi(), flav, hash(key))
                            if mom.status() != 2 or (abs(mom.pdgId()) < 100 and (abs(mom.pdgId()) != 15)):
                                break
                            if flav in [4,5]:
                                found = True
                                if key in ancestors:
                                    ancestors[key][1] += 1;
                                    ancestors[key][2] = min(ancestors[key][2], depth)
                                else:
                                    ancestors[key] = [mom, 1, depth, flav]
                            mom = mom.mother(0) if mom.numberOfMothers() > 0 else None
                            depth += 1
                        if found: matchable += 1
                s.mcMatchNTracks   = mctracks
                s.mcMatchNTracksHF = matchable
                s.mcMatchFraction  = -1.0
                s.mcFlavFirst      = 0
                s.mcFlavHeaviest   = 0
                s.mcHadron         = None
                if matchable:
                    maxhits  = max([h for (p,h,d,f) in ancestors.itervalues() ])
                    mindepth = min([d for (p,h,d,f) in ancestors.itervalues() if h == maxhits])
                    if matchable > 1:
                        s.mcMatchFraction = maxhits / float(matchable)
                    for (mom,hits,depth,flav) in ancestors.itervalues():
                        if hits != maxhits: continue
                        if depth == mindepth:
                            s.mcHadron    = mom
                            s.mcFlavFirst = flav
                        s.mcFlavHeaviest = max(flav, s.mcFlavHeaviest)
                        #print " \==> ancestor  pdgId %+6d with %d/%d hits at depth %d at %s" % (mom.pdgId(), hits, matchable, depth, hash(mom))
                        #if hits == maxhits and depth == mindepth: print "           ^^^^^--- this is our best match"

        # get the full id from a ref
        def ref2id(ref):
            return (ref.id().processIndex(), ref.id().productIndex(), ref.key())

        # Attach SVs to Jets 
        daumap = {}
        for s in event.ivf:
            s.jet = None
            for i in xrange(s.numberOfDaughters()):
                daumap[ref2id(s.daughterPtr(i))] = s
        for j in event.jetsIdOnly:
            #print "jet with pt %5.2f, eta %+4.2f, phi %+4.2f: " % (j.pt(), j.eta(), j.phi())
            jdaus = [ref2id(j.daughterPtr(i)) for i in xrange(j.numberOfDaughters())]
            j.svs = []
            for jdau in jdaus:
                if jdau in daumap:
                    #print " --> matched by ref with SV with pt %5.2f, eta %+4.2f, phi %+4.2f: " % (daumap[jdau].pt(), daumap[jdau].eta(), daumap[jdau].phi())
                    j.svs.append(daumap[jdau])
                    daumap[jdau].jet = j	    
        for s in event.ivf:
            if s.jet != None: continue
            #print "Unassociated SV with %d tracks, mass %5.2f, pt %5.2f, eta %+4.2f, phi %+4.2f: " % (s.numberOfDaughters(), s.mass(), s.pt(), s.eta(), s.phi())
            bestDr = 0.4
            for j in event.jetsIdOnly:
                dr = deltaR(s.eta(),s.phi(),j.eta(),j.phi())
                if dr < bestDr:
                   bestDr = dr
                   s.jet = j
                   #print "   close to jet with pt %5.2f, eta %+4.2f, phi %+4.2f: dr = %.3f" % (j.pt(), j.eta(), j.phi(), dr)



        #Attach SVs to leptons
        #print "\n\nNew event: "
        for l in event.selectedLeptons:
            #print "Lepton pdgId %+2d pt %5.2f, eta %+4.2f, phi %+4.2f, sip3d %5.2f, mcMatchAny %d, mcMatchId %d: " % (l.pdgId(), l.pt(), l.eta(), l.phi(), l.sip3D(), getattr(l,'mcMatchAny',-37), getattr(l,'mcMatchId',-37))
            track = l.gsfTrack() if abs(l.pdgId()) == 11  else l.track()
            l.ivfAssoc = None
            l.ivf      = None
            l.ivfSip3d = 0 # sip wrt SV
            l.ivfRedPt = 0 # pt of SV (without lepton)
            l.ivfRedM  = 0 # mass of SV (without lepton)
            for s in event.ivf:
                #dr = deltaR(l.eta(), l.phi(), s.eta(), s.phi())
                #mindr = min([deltaR(s.daughter(i).eta(),s.daughter(i).phi(),l.eta(),l.phi()) for i in xrange(s.numberOfDaughters())])
                sip3d = SignedImpactParameterComputer.signedIP3D(track.get(), s, s.momentum()).significance()
                byref = False
                daus = [ref2id(s.daughterPtr(i)) for i in xrange(s.numberOfDaughters())]
                for i in xrange(l.numberOfSourceCandidatePtrs()):
                    src = l.sourceCandidatePtr(i)
                    if src.isNonnull() and src.isAvailable():
                        if ref2id(src) in daus:
                            byref = True
                invmass  = (l.p4() + s.p4()).mass() if not byref else s.mass() 
                #go = False
                if byref:
                    l.ivfAssoc = "byref"
                    l.ivf      = s
                    l.ivfSip3d = sip3d
                    l.ivfRedM  = (s.p4() - l.p4()).mass()
                    l.ivfRedPt = (s.p4() - l.p4()).Pt()
                    #go = True
                    break
                elif l.ivfAssoc != "byref" and invmass < 6:
                    if l.ivfAssoc == None or (l.ivfAssoc[0] == "bymass" and l.ivfAssoc[1] > invmass):
                        l.ivfAssoc = ("bymass",invmass)
                        l.ivf      = s
                        l.ivfSip3d = sip3d
                        l.ivfRedM  = s.mass()
                        l.ivfRedPt = s.pt()
                        #go = True
                #bymc = False
                #if self.cfg_comp.isMC and l.mcMatchAny == 2 and s.mcHadron != None and l.mcMatchAny_gp != None:
                #    hadmothers = []
                #    mom = s.mcHadron.motherRef()
                #    while mom != None and mom.isNonnull() and mom.isAvailable():
                #        if mom.status() != 2 or abs(mom.pdgId()) < 100: break
                #        hadmothers.append(mom.key())
                #        mom = mom.motherRef() if mom.numberOfMothers() > 0 else None
                #    # now I need also to invent a ref to mom itself (won't be needed in new MiniAODs for which the packedGenParticles will have a motherRef and not just a mother)
                #    mom = s.mcHadron.motherRef()
                #    if mom.isNonnull() and mom.isAvailable():
                #        for i in xrange(mom.numberOfDaughters()):
                #            dau = mom.daughterRef(i)
                #            if dau.pdgId() == s.mcHadron.pdgId() and dau.status() == s.mcHadron.status() and abs(dau.pt()-s.mcHadron.pt()) < 1e-5 and abs(dau.eta()-s.mcHadron.eta()) < 1e-5:
                #                hadmothers.append(dau.key())
                #                break
                #    mom = l.mcMatchAny_gp.motherRef()
                #    while mom != None and mom.isNonnull() and mom.isAvailable(): # no idea why the isAvailable is needed
                #        if mom.status() != 2 or (abs(mom.pdgId()) < 100 and abs(mom.pdgId()) != 15): break
                #        if mom.key() in hadmothers:
                #            bymc = True
                #            break
                #        mom = mom.motherRef() if mom.numberOfMothers() > 0 else None 
                #if go or bymc:
                #    print "   SV with %d tracks, mass %5.2f, pt %5.2f, eta %+4.2f, phi %+4.2f, mc %s (ntkHF %d, frac %.2f): " % (s.numberOfDaughters(), s.mass(), s.pt(), s.eta(), s.phi(), s.mcHadron != None, s.mcMatchNTracksHF, s.mcMatchFraction)
                #    print "          dr = %.4f, mindr = %.4f, mass = %6.2f, sip3d %+5.2f, byref = %s, bymc = %s " % (dr, mindr, invmass, sip3d, byref, bymc)
        return True
