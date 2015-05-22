from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.Jet import Jet
from CMGTools.TTHAnalysis.analyzers.ntupleTypes import ptRelv1
from PhysicsTools.HeppyCore.utils.deltar import deltaR
from copy import copy
import ROOT

class ttHDeclusterJetsAnalyzer( Analyzer ):
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHDeclusterJetsAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName) 
        # this we copy, since we modify
        self.verbose = self.cfg_ana.verbose

    def declareHandles(self):
        super(ttHDeclusterJetsAnalyzer, self).declareHandles()

    def beginLoop(self, setup):
        super(ttHDeclusterJetsAnalyzer,self).beginLoop(setup)

    def partonCount(self, event):
        if not self.cfg_comp.isMC: return True
        partons = [ p for p in event.generatorSummary if abs(p.pdgId()) in [1,2,3,4,5,21,22] and p.pt() > self.cfg_ana.mcPartonPtCut ]
        leptons = [ l for l in event.genleps + event.gentauleps if l.pt() > self.cfg_ana.mcLeptonPtCut ]
        taus    = [ t for t in event.gentaus if t.pt() > self.cfg_ana.mcTauPtCut ]
        for i,j in enumerate(event.jets):
            j.mcNumPartons = sum([(deltaR(p,j) < 0.4) for p in partons ])
            j.mcNumLeptons = sum([(deltaR(p,j) < 0.4) for p in leptons ])
            j.mcNumTaus    = sum([(deltaR(p,j) < 0.4) for p in taus    ])
            p4any = None 
            for p in partons+leptons+taus:
                if deltaR(p,j) < 0.4:
                    p4any = p.p4() + p4any if p4any != None else p.p4()
            j.mcAnyPartonMass = p4any.M() if p4any != None else 0.0
            #print "Jet %3d of pt %8.2f, eta %+5.2f, mass %6.2f has %2d/%1d/%1d p/l/t for a total inv mass of %8.2f" % (
            #            i, j.pt(), j.eta(), j.mass(), j.mcNumPartons, j.mcNumLeptons, j.mcNumTaus, j.mcAnyPartonMass)

    def processJets(self, event):
        for j in event.jets:
            j.prunedMass = j.mass()
            j.nSubJets   = 1
            j.nSubJets25 = 1 if j.pt() > 25 else 0
            j.nSubJets30 = 1 if j.pt() > 30 else 0
            j.nSubJets40 = 1 if j.pt() > 40 else 0
            j.nSubJetsZ01 = 1
            if not self.cfg_ana.jetCut(j): 
                continue
            objects  = ROOT.std.vector(ROOT.reco.Particle.LorentzVector)()
            for idau in xrange(j.numberOfDaughters()):
               dau = j.daughter(idau)
               objects.push_back(dau.p4())
            if objects.size() <= 1: continue           
            if self.verbose: print "Jet of pt %8.2f, eta %+5.2f, mass %6.2f " % (j.pt(), j.eta(), j.mass())
            if self.cfg_comp.isMC: 
                if self.verbose: print "\t partons %2d, leptons %1d, taus %2d (total mass: %8.2f)" % ( j.mcNumPartons, j.mcNumLeptons, j.mcNumTaus, j.mcAnyPartonMass)
            if self.cfg_ana.prune:
                # kt exclusive
                reclusterJets = ROOT.heppy.ReclusterJets(objects, 1.,10)
                pruned = reclusterJets.getPruned(self.cfg_ana.pruneZCut,self.cfg_ana.pruneRCutFactor)
                j.prunedP4 = pruned
                if self.verbose: print "\t pruned mass %8.2f, ptLoss %.3f" % ( pruned.M(), pruned.Pt()/j.pt() )
            # kt inclusive R=0.2
            reclusterJets02 = ROOT.heppy.ReclusterJets(objects, 1.,0.2)          
            inclusiveJets02 = reclusterJets02.getGrouping(self.cfg_ana.ptMinSubjets)
            j.nSubJets25 = sum([(js.pt() > 25) for js in inclusiveJets02])
            j.nSubJets30 = sum([(js.pt() > 30) for js in inclusiveJets02])
            j.nSubJets40 = sum([(js.pt() > 40) for js in inclusiveJets02])
            j.nSubJetsZ01 = sum([(js.pt() > 0.1*j.pt()) for js in inclusiveJets02])
            if self.verbose: print "\t subjets: \n\t\t%s" % ("\n\t\t".join(["pt %8.2f, mass %6.2f" % (js.pt(),js.M()) for js in inclusiveJets02]))

    def processLeptons(self, event):
        event.recoveredJets = []
        event.recoveredSplitJets = []
        for l in event.selectedLeptons:
            # lepton-jet variable, jet is obtained from reclustering of associated akt04 jet daughters, with kt exclusive
            l.jetDecDR      = deltaR(l.eta(),l.phi(),l.jet.eta(),l.jet.phi())
            l.jetDecPtRatio = l.pt()/l.jet.pt()
            l.jetDecPtRel   = ptRelv1(l.p4(),l.jet.p4())
            l.jetDecPrunedPtRatio = l.pt()/l.jet.pt()
            l.jetDecPrunedMass    = l.jet.mass()
            
            # lepton-jet variable, jet is obtained from reclustering of associated akt04 jet daughters, with kt inclusive R=0.2
            l.jetDec02DR      = deltaR(l.eta(),l.phi(),l.jet.eta(),l.jet.phi())
            l.jetDec02PtRatio = l.pt()/l.jet.pt()
            l.jetDec02PtRel   = ptRelv1(l.p4(),l.jet.p4())
            l.jetDec02PrunedPtRatio = l.pt()/l.jet.pt()
            l.jetDec02PrunedMass    = l.jet.mass()
            
            # # lepton-jet variable, jet is obtained from reclustering of associated ak08 jet daughters, with kt inclusive R=0.2
            # l.fjetDec02DR      = deltaR(l.eta(),l.phi(),l.fatjet.eta(),l.fatjet.phi())
            # l.fjetDec02PtRatio = l.pt()/l.fatjet.pt()
            # l.fjetDec02PtRel   = ptRelv1(l.p4(),l.fatjet.p4())
            # l.fjetDec02PrunedPtRatio = l.pt()/l.fatjet.pt()
            # l.fjetDec02PrunedMass    = l.fatjet.mass()
            
            if not self.cfg_ana.lepCut(l,l.jetDecPtRel): continue
            if self.verbose and type(self.verbose) == int: self.verbose -= 1
            j = l.jet  
            fj = l.fatjet
            if self.verbose: print "lepton pt %6.1f [mc %d], jet pt %6.1f, mass %6.2f,  relIso %5.2f, dr(lep) = %.3f, ptRel v1 = %5.1f, ptF = %5.2f" % (l.pt(), getattr(l,'mcMatchId',0), j.pt(), j.mass(), l.relIso03, l.jetDecDR, l.jetDecPtRel, l.jetDecPtRatio)
            objects  = ROOT.std.vector(ROOT.reco.Particle.LorentzVector)()
            fobjects  = ROOT.std.vector(ROOT.reco.Particle.LorentzVector)()
            for idau in xrange(j.numberOfDaughters()):
               dau = j.daughter(idau)
               objects.push_back(dau.p4())
            for fidau in xrange(fj.numberOfDaughters()):
                fdau = fj.daughter(fidau)
                fobjects.push_back(fdau.p4()) 
               
            
            ##### considering only candidates from the ak04 jet   
            if objects.size() <= 1: continue           
            # kt exclusive
            reclusterJets = ROOT.heppy.ReclusterJets(objects, 1.,10)
            # kt inclusive R=0.2
            reclusterJets02 = ROOT.heppy.ReclusterJets(objects, 1.,0.2)          
            if self.cfg_ana.prune:
                pruned = reclusterJets.getPruned(self.cfg_ana.pruneZCut,self.cfg_ana.pruneRCutFactor)
                l.jetDecPrunedPtRatio = l.pt()/pruned.pt()
                if self.verbose:
                    print "    ... pruned jet mass: %6.2f, ptRelv1 %5.2f, ptFraction %5.2f " % (pruned.mass(), ptRelv1(l.p4(),pruned), l.pt()/pruned.pt())
                    
            # compute lepton-jet variable, jet is obtained from reclustering of associated akt04 jet daughters, with kt exclusive        
            for nsub in xrange(2,self.cfg_ana.maxSubjets+1):
               if nsub > objects.size(): break
               exclusiveJets = reclusterJets.getGroupingExclusive(nsub)
               drbest = 1; ibest = -1
               for isub in xrange(len(exclusiveJets)):
                   sj = exclusiveJets[isub]
                   dr  = deltaR(l.eta(),l.phi(),sj.eta(),sj.phi())
                   ptR = ptRelv1(l.p4(),sj)
                   ptF = l.pt()/sj.pt()
                   if dr < drbest or ibest == -1:
                       drbest = dr; ibest = isub
                   if self.cfg_ana.prune:
                       pp4 = reclusterJets.getPrunedSubjetExclusive(isub,self.cfg_ana.pruneZCut,self.cfg_ana.pruneRCutFactor)
                       ppF = l.pt()/pp4.pt()
                       ppR = ptRelv1(l.p4(),pp4)
                       pM  = pp4.mass()
                   if self.verbose:
                      if self.cfg_ana.prune:
                         print "    exclusive subjet %d/%d: pt %6.1f, mass %6.2f (pruned %6.2f), dr(lep) = %.3f, ptRel v1 = %5.1f (pruned %5.1f), ptF = %5.2f (pruned %5.2f)" % (isub,nsub, sj.pt(), sj.mass(), pM, dr, ptR, ppR, ptF, ppF)
                      else:
                         print "    exlcusive subjet %d/%d: pt %6.1f, mass %6.2f, dr(lep) = %.3f, ptRel v1 = %5.1f, ptF = %5.2f" % (isub,nsub, sj.pt(), sj.mass(), dr, ptR, ptF)
               if ibest == -1: continue
               sj = exclusiveJets[ibest]
               dr  = deltaR(l.eta(),l.phi(),sj.eta(),sj.phi())
               ptR = ptRelv1(l.p4(),sj)
               ptF = l.pt()/sj.pt()
               l.jetDecDR      = dr 
               l.jetDecPtRatio = ptF
               l.jetDecPtRel   = ptR
               if self.cfg_ana.prune:
                  pp4 = reclusterJets.getPrunedSubjetExclusive(ibest,self.cfg_ana.pruneZCut,self.cfg_ana.pruneRCutFactor)
                  ppF = l.pt()/pp4.pt()
                  ppR = ptRelv1(l.p4(),pp4)
                  pM  = pp4.mass()
                  l.jetDecPrunedPtRatio = ppF #l.pt()/pp4.pt()
                  l.jetDecPrunedMass    = pM  #l.jet.mass()
               if self.verbose:
                  if self.cfg_ana.prune:
                     print "    best exclusive subject %d/%d: pt %6.1f, mass %6.2f (pruned %6.2f), dr(lep) = %.3f, ptRel v1 = %5.1f (pruned %5.1f), ptF = %5.2f (pruned %5.2f)" % (ibest,nsub, sj.pt(), sj.mass(), pM, dr, ptR, ppR, ptF, ppF)
                  else:
                     print "    best exclusive subject %d/%d: pt %6.1f, mass %6.2f, dr(lep) = %.3f, ptRel v1 = %5.1f, ptF = %5.2f" % (ibest,nsub, sj.pt(), sj.mass(), dr, ptR, ptF)
               if dr < self.cfg_ana.drMin and ptF < self.cfg_ana.ptRatioMax and (abs(ptF-1) < self.cfg_ana.ptRatioDiff or dr < self.cfg_ana.drMatch or ptR < self.cfg_ana.ptRelMin):
                    if self.verbose: print "       ---> take this as best subject, stop reclustering, consider it successful"
                    restp4 = None
                    for (i2,s2) in enumerate(exclusiveJets):
                        if i2 == ibest: continue
                        restp4 = s2 if restp4 == None else (restp4 + s2)
                        prunedJet = copy(l.jet)
                        prunedJet.physObj = ROOT.pat.Jet(l.jet.physObj)
                        prunedJet.setP4(s2)
                        event.recoveredSplitJets.append(prunedJet)
                    prunedJet = copy(l.jet)
                    prunedJet.physObj = ROOT.pat.Jet(l.jet.physObj)
                    prunedJet.setP4(restp4)
                    event.recoveredJets.append(prunedJet)
                    break
               if self.verbose: print ""
            if self.verbose: print ""
            event.recoveredJets.sort(key = lambda j : j.pt(), reverse=True)
            event.recoveredSplitJets.sort(key = lambda j : j.pt(), reverse=True)

            # lepton-jet variable, jet is obtained from reclustering of associated akt04 jet daughters, with kt inclusive R=0.2
            inclusiveJets02 = reclusterJets02.getGrouping(self.cfg_ana.ptMinSubjets)
            drbest = 1; ibest = -1
            for isub in xrange(len(inclusiveJets02)):
                ij = inclusiveJets02[isub]
                dr  = deltaR(l.eta(),l.phi(),ij.eta(),ij.phi())
                ptR = ptRelv1(l.p4(),ij)
                ptF = l.pt()/ij.pt()
                if dr < drbest or ibest == -1:
                    drbest = dr; ibest = isub
                if self.cfg_ana.prune:
                    pp4 = reclusterJets02.getPrunedSubjetInclusive(isub,self.cfg_ana.pruneZCut,self.cfg_ana.pruneRCutFactor)
                    ppF = l.pt()/pp4.pt()
                    ppR = ptRelv1(l.p4(),pp4)
                    pM  = pp4.mass()
                if self.verbose:
                   if self.cfg_ana.prune:
                      print "    inclusive subjet %d/%d: pt %6.1f, mass %6.2f (pruned %6.2f), dr(lep) = %.3f, ptRel v1 = %5.1f (pruned %5.1f), ptF = %5.2f (pruned %5.2f)" % (isub,nsub, ij.pt(), ij.mass(), pM, dr, ptR, ppR, ptF, ppF)
                   else:
                      print "    inclusive subjet %d/%d: pt %6.1f, mass %6.2f, dr(lep) = %.3f, ptRel v1 = %5.1f, ptF = %5.2f" % (isub,nsub, ij.pt(), ij.mass(), dr, ptR, ptF)
            if ibest != -1:
                ij = inclusiveJets02[ibest]
                dr  = deltaR(l.eta(),l.phi(),ij.eta(),ij.phi())
                ptR = ptRelv1(l.p4(),ij)
                ptF = l.pt()/ij.pt()
                l.jetDec02DR      = dr 
                l.jetDec02PtRatio = ptF
                l.jetDec02PtRel   = ptR
                if self.cfg_ana.prune:
                   pp4 = reclusterJets02.getPrunedSubjetInclusive(ibest,self.cfg_ana.pruneZCut,self.cfg_ana.pruneRCutFactor)
                   ppF = l.pt()/pp4.pt()
                   ppR = ptRelv1(l.p4(),pp4)
                   pM  = pp4.mass()
                   l.jetDec02PrunedPtRatio = ppF #l.pt()/pp4.pt()
                   l.jetDec02PrunedMass    = pM  #l.jet.mass()
                if self.verbose:
                   if self.cfg_ana.prune:
                      print "    best inclusive subject %d/%d: pt %6.1f, mass %6.2f (pruned %6.2f), dr(lep) = %.3f, ptRel v1 = %5.1f (pruned %5.1f), ptF = %5.2f (pruned %5.2f)" % (ibest,nsub, ij.pt(), ij.mass(), pM, dr, ptR, ppR, ptF, ppF)
                   else:
                      print "    best inclusive subject %d/%d: pt %6.1f, mass %6.2f, dr(lep) = %.3f, ptRel v1 = %5.1f, ptF = %5.2f" % (ibest,nsub, ij.pt(), ij.mass(), dr, ptR, ptF)

                    
                     
    def process(self, event):
        self.partonCount(event)
        self.processJets(event)
        self.processLeptons(event)
        return True
