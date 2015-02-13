import math
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.HeppyCore.utils.deltar import deltaR2, deltaPhi, matchObjectCollection, matchObjectCollection2, bestMatch
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import JetReCalibrator
import PhysicsTools.HeppyCore.framework.config as cfg

from PhysicsTools.Heppy.physicsutils.QGLikelihoodCalculator import QGLikelihoodCalculator

def cleanNearestJetOnly(jets,leptons,deltaR):
    dr2 = deltaR**2
    good = [ True for j in jets ]
    for l in leptons:
        ibest, d2m = -1, dr2
        for i,j in enumerate(jets):
            d2i = deltaR2(l.eta(),l.phi(), j.eta(),j.phi())
            if d2i < d2m:
                ibest, d2m = i, d2i
        if ibest != -1: good[ibest] = False
    return [ j for (i,j) in enumerate(jets) if good[i] == True ] 

def cleanJetsAndLeptons(jets,leptons,deltaR,arbitration):
    dr2 = deltaR**2
    goodjet = [ True for j in jets ]
    goodlep = [ True for l in leptons ]
    for il, l in enumerate(leptons):
        ibest, d2m = -1, dr2
        for i,j in enumerate(jets):
            d2i = deltaR2(l.eta(),l.phi(), j.eta(),j.phi())
            if d2i < dr2:
                if arbitration(j,l) == j:
                   # if the two match, and we prefer the jet, then drop the lepton and be done
                   goodlep[il] = False
                   break 
            if d2i < d2m:
                ibest, d2m = i, d2i
        # this lepton has been killed by a jet, then we clean the jet that best matches it
        if not goodlep[il]: continue 
        if ibest != -1: goodjet[ibest] = False
    return ( [ j for (i ,j) in enumerate(jets)    if goodjet[i ] == True ], 
             [ l for (il,l) in enumerate(leptons) if goodlep[il] == True ] )


class JetAnalyzer( Analyzer ):
    """Taken from RootTools.JetAnalyzer, simplified, modified, added corrections    """
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(JetAnalyzer,self).__init__(cfg_ana, cfg_comp, looperName)
        mcGT   = cfg_ana.mcGT   if hasattr(cfg_ana,'mcGT')   else "PHYS14_25_V2"
        dataGT = cfg_ana.dataGT if hasattr(cfg_ana,'dataGT') else "GR_70_V2_AN1"
        self.shiftJEC = self.cfg_ana.shiftJEC if hasattr(self.cfg_ana, 'shiftJEC') else 0
        self.recalibrateJets = self.cfg_ana.recalibrateJets
        if   self.recalibrateJets == "MC"  : self.recalibrateJets =     self.cfg_comp.isMC
        elif self.recalibrateJets == "Data": self.recalibrateJets = not self.cfg_comp.isMC
        elif self.recalibrateJets not in [True,False]: raise RuntimeError, "recalibrateJets must be any of { True, False, 'MC', 'Data' }, while it is %r " % self.recalibrateJets
        self.doJEC = self.recalibrateJets or (self.shiftJEC != 0)
        if self.doJEC:
          if self.cfg_comp.isMC:
            self.jetReCalibrator = JetReCalibrator(mcGT,"AK4PFchs", False,cfg_ana.jecPath)
          else:
            self.jetReCalibrator = JetReCalibrator(dataGT,"AK4PFchs", True,cfg_ana.jecPath)
        self.doPuId = getattr(self.cfg_ana, 'doPuId', True)
        self.jetLepDR = getattr(self.cfg_ana, 'jetLepDR', 0.4)
        self.jetLepArbitration = getattr(self.cfg_ana, 'jetLepArbitration', lambda jet,lepton: lepton) 
        self.lepPtMin = getattr(self.cfg_ana, 'minLepPt', -1)
        self.jetGammaDR =  getattr(self.cfg_ana, 'jetGammaDR', 0.4)
        if(self.cfg_ana.doQG):
            self.qglcalc = QGLikelihoodCalculator("/afs/cern.ch/user/t/tomc/public/qgTagger/QGLikelihoodDBFiles/QGL_v1a/pdfQG_AK4chs_antib_13TeV_v1.root")

    def declareHandles(self):
        super(JetAnalyzer, self).declareHandles()
        self.handles['jets']   = AutoHandle( self.cfg_ana.jetCol, 'std::vector<pat::Jet>' )
        self.handles['genJet'] = AutoHandle( 'slimmedGenJets', 'vector<reco::GenJet>' )
        self.shiftJER = self.cfg_ana.shiftJER if hasattr(self.cfg_ana, 'shiftJER') else 0
        self.handles['rho'] = AutoHandle( ('fixedGridRhoFastjetAll','',''), 'double' )
    
    def beginLoop(self, setup):
        super(JetAnalyzer,self).beginLoop(setup)
        
    def process(self, event):
        self.readCollections( event.input )
        rho  = float(self.handles['rho'].product()[0])
        event.rho = rho

        ## Read jets, if necessary recalibrate and shift MET
        allJets = map(Jet, self.handles['jets'].product()) 

        event.deltaMetFromJEC = [0.,0.]
        if self.doJEC:
            #print "\nCalibrating jets %s for lumi %d, event %d" % (self.cfg_ana.jetCol, event.lumi, event.eventId)
            self.jetReCalibrator.correctAll(allJets, rho, delta=self.shiftJEC, metShift=event.deltaMetFromJEC)
        event.allJetsUsedForMET = allJets

        if self.cfg_comp.isMC:
            event.genJets = [ x for x in self.handles['genJet'].product() ]
            self.matchJets(event, allJets)
            if getattr(self.cfg_ana, 'smearJets', False):
                self.smearJets(event, allJets)
        
	##Sort Jets by pT 
        allJets.sort(key = lambda j : j.pt(), reverse = True)
        
	## Apply jet selection
        event.jets = []
        event.jetsFailId = []
        event.jetsAllNoID = []
        event.jetsIdOnly = []
        for jet in allJets:
            if self.testJetNoID( jet ): 
                event.jetsAllNoID.append(jet) 
                if self.testJetID (jet ):
                    
                    if(self.cfg_ana.doQG):
                        self.computeQGvars(jet)
                        jet.qgl = self.qglcalc.computeQGLikelihood(jet, rho)


                    event.jets.append(jet)
                    event.jetsIdOnly.append(jet)
                else:
                    event.jetsFailId.append(jet)
            elif self.testJetID (jet ):
                event.jetsIdOnly.append(jet)

        ## Clean Jets from leptons
        leptons = []
        if hasattr(event, 'selectedLeptons'):
            leptons = [ l for l in event.selectedLeptons if l.pt() > self.lepPtMin ]
        if self.cfg_ana.cleanJetsFromTaus and hasattr(event, 'selectedTaus'):
            leptons = leptons[:] + event.selectedTaus
        if self.cfg_ana.cleanJetsFromIsoTracks and hasattr(event, 'selectedIsoCleanTrack'):
            leptons = leptons[:] + event.selectedIsoCleanTrack
        event.cleanJetsAll, cleanLeptons = cleanJetsAndLeptons(event.jets, leptons, self.jetLepDR, self.jetLepArbitration)
        event.cleanJets    = [j for j in event.cleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        event.cleanJetsFwd = [j for j in event.cleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]
        event.discardedJets = [j for j in event.jets if j not in event.cleanJetsAll]
        if hasattr(event, 'selectedLeptons'):
            event.discardedLeptons = [ l for l in leptons if l not in cleanLeptons ]
            event.selectedLeptons  = [ l for l in event.selectedLeptons if l not in event.discardedLeptons ]

        ## Clean Jets from photons
        photons = []
        if hasattr(event, 'selectedPhotons'):
            photons = [ g for g in event.selectedPhotons ]
        event.gamma_cleanJetsAll = cleanNearestJetOnly(event.cleanJetsAll, photons, self.jetGammaDR)
        event.gamma_cleanJets    = [j for j in event.gamma_cleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        event.gamma_cleanJetsFwd = [j for j in event.gamma_cleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]
        ###


        ## Associate jets to leptons
        leptons = event.inclusiveLeptons if hasattr(event, 'inclusiveLeptons') else event.selectedLeptons
        jlpairs = matchObjectCollection( leptons, allJets, self.jetLepDR**2)

        for jet in allJets:
            jet.leptons = [l for l in jlpairs if jlpairs[l] == jet ]

        for lep in leptons:
            jet = jlpairs[lep]
            if jet is None:
                lep.jet = lep
            else:
                lep.jet = jet

        if self.cfg_comp.isMC:
            event.deltaMetFromJetSmearing = [0, 0]
            for j in event.cleanJetsAll:
                if hasattr(j, 'deltaMetFromJetSmearing'):
                    event.deltaMetFromJetSmearing[0] += j.deltaMetFromJetSmearing[0]
                    event.deltaMetFromJetSmearing[1] += j.deltaMetFromJetSmearing[1]
            event.cleanGenJets = cleanNearestJetOnly(event.genJets, leptons, self.jetLepDR)
            
            #event.nGenJets25 = 0
            #event.nGenJets25Cen = 0
            #event.nGenJets25Fwd = 0
            #for j in event.cleanGenJets:
            #    event.nGenJets25 += 1
            #    if abs(j.eta()) <= 2.4: event.nGenJets25Cen += 1
            #    else:                   event.nGenJets25Fwd += 1
                    
            self.jetFlavour(event)
        
    
        return True

        

    def testJetID(self, jet):
        jet.puJetIdPassed = jet.puJetId() 
        jet.pfJetIdPassed = jet.jetID('POG_PFID_Loose') 
        if self.cfg_ana.relaxJetId:
            return True
        else:
            return jet.pfJetIdPassed and (jet.puJetIdPassed or not(self.doPuId)) 
        
    def testJetNoID( self, jet ):
        # 2 is loose pile-up jet id
        return jet.pt() > self.cfg_ana.jetPt and \
               abs( jet.eta() ) < self.cfg_ana.jetEta;

    def computeQGvars(self, jet):

       jet.mult = 0
       sum_weight = 0.
       sum_pt = 0.
       sum_deta = 0.
       sum_dphi = 0.
       sum_deta2 = 0.
       sum_detadphi = 0.
       sum_dphi2 = 0.



       for ii in range(0, jet.numberOfDaughters()) :

         part = jet.daughter(ii)

         if part.charge() == 0 : # neutral particles 

           if part.pt() < 1.: continue

         else : # charged particles

           if part.trackHighPurity()==False: continue
           if part.fromPV()<=1: continue


         jet.mult += 1

         deta = part.eta() - jet.eta()
         dphi = deltaPhi(part.phi(), jet.phi())
         partPt = part.pt()
         weight = partPt*partPt
         sum_weight += weight
         sum_pt += partPt
         sum_deta += deta*weight
         sum_dphi += dphi*weight
         sum_deta2 += deta*deta*weight
         sum_detadphi += deta*dphi*weight
         sum_dphi2 += dphi*dphi*weight




       a = 0.
       b = 0.
       c = 0.

       if sum_weight > 0 :
         jet.ptd = math.sqrt(sum_weight)/sum_pt
         ave_deta = sum_deta/sum_weight
         ave_dphi = sum_dphi/sum_weight
         ave_deta2 = sum_deta2/sum_weight
         ave_dphi2 = sum_dphi2/sum_weight
         a = ave_deta2 - ave_deta*ave_deta
         b = ave_dphi2 - ave_dphi*ave_dphi
         c = -(sum_detadphi/sum_weight - ave_deta*ave_dphi)
       else: jet.ptd = 0.

       delta = math.sqrt(math.fabs((a-b)*(a-b)+4.*c*c))

       if a+b-delta > 0: jet.axis2 = -math.log(math.sqrt(0.5*(a+b-delta)))
       else: jet.axis2 = -1.


    def jetFlavour(self,event):
        def isFlavour(x,f):
            id = abs(x.pdgId())
            if id > 999: return (id/1000)%10 == f
            if id >  99: return  (id/100)%10 == f
            return id % 100 == f



        event.bqObjects = [ p for p in event.genParticles if (p.status() == 2 and isFlavour(p,5)) ]
        event.cqObjects = [ p for p in event.genParticles if (p.status() == 2 and isFlavour(p,4)) ]

        event.partons   = [ p for p in event.genParticles if ((p.status() == 23 or p.status() == 3) and abs(p.pdgId())>0 and (abs(p.pdgId()) in [1,2,3,4,5,21]) ) ]
        match = matchObjectCollection2(event.cleanJetsAll,
                                       event.partons,
                                       deltaRMax = 0.3)

        for jet in event.cleanJetsAll:
            parton = match[jet]
            jet.partonId = (parton.pdgId() if parton != None else 0)
            jet.partonMotherId = (parton.mother(0).pdgId() if parton != None and parton.numberOfMothers()>0 else 0)
        
        for jet in event.jets:
           (bmatch, dr) = bestMatch(jet, event.bqObjects)
           if dr < 0.4:
               jet.mcFlavour = 5
           else:
               (cmatch, dr) = bestMatch(jet, event.cqObjects) 
               if dr < 0.4:
                   jet.mcFlavour = 4
               else:
                   jet.mcFlavour = 0

        event.heaviestQCDFlavour = 5 if len(event.bqObjects) else (4 if len(event.cqObjects) else 1);
 

    def matchJets(self, event, jets):
        match = matchObjectCollection2(jets,
                                       event.genbquarks + event.genwzquarks,
                                       deltaRMax = 0.3)
        for jet in jets:
            gen = match[jet]
            jet.mcParton    = gen
            jet.mcMatchId   = (gen.sourceId     if gen != None else 0)
            jet.mcMatchFlav = (abs(gen.pdgId()) if gen != None else 0)

        match = matchObjectCollection2(jets,
                                       event.genJets,
                                       deltaRMax = 0.3)
        for jet in jets:
            jet.mcJet = match[jet]


 
    def smearJets(self, event, jets):
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TWikiTopRefSyst#Jet_energy_resolution
       for jet in jets:
            gen = jet.mcJet 
            if gen != None:
               genpt, jetpt, aeta = gen.pt(), jet.pt(), abs(jet.eta())
               # from https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution
               factor = 1.052 + self.shiftJER*math.hypot(0.012,0.062);
               if   aeta > 2.3: factor = 1.288 + self.shiftJER*math.hypot(0.127,0.154)
               elif aeta > 1.7: factor = 1.134 + self.shiftJER*math.hypot(0.035,0.066)
               elif aeta > 1.1: factor = 1.096 + self.shiftJER*math.hypot(0.017,0.063)
               elif aeta > 0.5: factor = 1.057 + self.shiftJER*math.hypot(0.012,0.056)
               ptscale = max(0.0, (jetpt + (factor-1)*(jetpt-genpt))/jetpt)
               #print "get with pt %.1f (gen pt %.1f, ptscale = %.3f)" % (jetpt,genpt,ptscale)
               jet.deltaMetFromJetSmearing = [ -(ptscale-1)*jet.rawFactor()*jet.px(), -(ptscale-1)*jet.rawFactor()*jet.py() ]
               if ptscale != 0:
                  jet.setP4(jet.p4()*ptscale)
               # leave the uncorrected unchanged for sync
               jet._rawFactorMultiplier *= (1.0/ptscale) if ptscale != 0 else 1
            #else: print "jet with pt %.1d, eta %.2f is unmatched" % (jet.pt(), jet.eta())



setattr(JetAnalyzer,"defaultConfig", cfg.Analyzer(
    class_object = JetAnalyzer,
    jetCol = 'slimmedJets',
    jetPt = 25.,
    jetEta = 4.7,
    jetEtaCentral = 2.4,
    jetLepDR = 0.4,
    jetLepArbitration = (lambda jet,lepton : lepton), # you can decide which to keep in case of overlaps; e.g. if the jet is b-tagged you might want to keep the jet
    minLepPt = 10,
    relaxJetId = False,  
    doPuId = False, # Not commissioned in 7.0.X
    doQG = False, 
    recalibrateJets = False,
    shiftJEC = 0, # set to +1 or -1 to get +/-1 sigma shifts
    smearJets = True,
    shiftJER = 0, # set to +1 or -1 to get +/-1 sigma shifts    
    cleanJetsFromTaus = False,
    cleanJetsFromIsoTracks = False,
    jecPath = ""
    )
)
