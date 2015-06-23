import math, os
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.HeppyCore.utils.deltar import deltaR2, deltaPhi, matchObjectCollection, matchObjectCollection2, bestMatch
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import JetReCalibrator
import PhysicsTools.HeppyCore.framework.config as cfg

from PhysicsTools.Heppy.physicsutils.QGLikelihoodCalculator import QGLikelihoodCalculator

import copy
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
                choice = arbitration(j,l)
                if choice == j:
                   # if the two match, and we prefer the jet, then drop the lepton and be done
                   goodlep[il] = False
                   break 
                elif choice == (j,l) or choice == (l,j):
                   # asked to keep both, so we don't consider this match
                   continue
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
            self.jetReCalibrator = JetReCalibrator(mcGT,self.cfg_ana.recalibrationType, False,cfg_ana.jecPath)
          else:
            self.jetReCalibrator = JetReCalibrator(dataGT,self.cfg_ana.recalibrationType, True,cfg_ana.jecPath)
        self.doPuId = getattr(self.cfg_ana, 'doPuId', True)
        self.jetLepDR = getattr(self.cfg_ana, 'jetLepDR', 0.4)
        self.jetLepArbitration = getattr(self.cfg_ana, 'jetLepArbitration', lambda jet,lepton: lepton) 
        self.lepPtMin = getattr(self.cfg_ana, 'minLepPt', -1)
        self.lepSelCut = getattr(self.cfg_ana, 'lepSelCut', lambda lep : True)
        self.jetGammaDR =  getattr(self.cfg_ana, 'jetGammaDR', 0.4)
        if(self.cfg_ana.doQG):
            qgdefname="{CMSSW_BASE}/src/PhysicsTools/Heppy/data/pdfQG_AK4chs_antib_13TeV_v1.root"
            self.qglcalc = QGLikelihoodCalculator(getattr(self.cfg_ana,"QGpath",qgdefname).format(CMSSW_BASE= os.environ['CMSSW_BASE']))
        if not hasattr(self.cfg_ana ,"collectionPostFix"):self.cfg_ana.collectionPostFix=""

    def declareHandles(self):
        super(JetAnalyzer, self).declareHandles()
        self.handles['jets']   = AutoHandle( self.cfg_ana.jetCol, 'std::vector<pat::Jet>' )
        self.handles['genJet'] = AutoHandle( self.cfg_ana.genJetCol, 'vector<reco::GenJet>' )
        self.shiftJER = self.cfg_ana.shiftJER if hasattr(self.cfg_ana, 'shiftJER') else 0
        self.handles['rho'] = AutoHandle( self.cfg_ana.rho, 'double' )
    
    def beginLoop(self, setup):
        super(JetAnalyzer,self).beginLoop(setup)

    def process(self, event):
        self.readCollections( event.input )
        rho  = float(self.handles['rho'].product()[0])
        self.rho = rho

        ## Read jets, if necessary recalibrate and shift MET
        if self.cfg_ana.copyJetsByValue: 
          import ROOT
          allJets = map(lambda j:Jet(ROOT.pat.Jet(j)), self.handles['jets'].product()) #copy-by-value is safe if JetAnalyzer is ran more than once
        else: 
          allJets = map(Jet, self.handles['jets'].product()) 

        self.deltaMetFromJEC = [0.,0.]
#        print "before. rho",self.rho,self.cfg_ana.collectionPostFix,'allJets len ',len(allJets),'pt', [j.pt() for j in allJets]
        if self.doJEC:
#            print "\nCalibrating jets %s for lumi %d, event %d" % (self.cfg_ana.jetCol, event.lumi, event.eventId)
            self.jetReCalibrator.correctAll(allJets, rho, delta=self.shiftJEC, metShift=self.deltaMetFromJEC)

        for delta, shift in [(1.0, "JECUp"), (0.0, ""), (-1.0, "JECDown")]:
            for j1 in allJets:
                corr = self.jetReCalibrator.getCorrection(j1, rho, delta, self.deltaMetFromJEC)
                setattr(j1, "corr"+shift, corr)

        self.allJetsUsedForMET = allJets
#        print "after. rho",self.rho,self.cfg_ana.collectionPostFix,'allJets len ',len(allJets),'pt', [j.pt() for j in allJets]

        if self.cfg_comp.isMC:
            self.genJets = [ x for x in self.handles['genJet'].product() ]
            self.matchJets(event, allJets)
            if getattr(self.cfg_ana, 'smearJets', False):
                self.smearJets(event, allJets)
        
	##Sort Jets by pT 
        allJets.sort(key = lambda j : j.pt(), reverse = True)
	## Apply jet selection
        self.jets = []
        self.jetsFailId = []
        self.jetsAllNoID = []
        self.jetsIdOnly = []
        for jet in allJets:
            if self.testJetNoID( jet ): 
                self.jetsAllNoID.append(jet) 
                if self.testJetID (jet ):
                    
                    if(self.cfg_ana.doQG):
                        jet.qgl_calc =  self.qglcalc.computeQGLikelihood
                        jet.qgl_rho =  rho


                    self.jets.append(jet)
                    self.jetsIdOnly.append(jet)
                else:
                    self.jetsFailId.append(jet)
            elif self.testJetID (jet ):
                self.jetsIdOnly.append(jet)

        ## Clean Jets from leptons
        leptons = []
        if hasattr(event, 'selectedLeptons'):
            leptons = [ l for l in event.selectedLeptons if l.pt() > self.lepPtMin and self.lepSelCut(l) ]
        if self.cfg_ana.cleanJetsFromTaus and hasattr(event, 'selectedTaus'):
            leptons = leptons[:] + event.selectedTaus
        if self.cfg_ana.cleanJetsFromIsoTracks and hasattr(event, 'selectedIsoCleanTrack'):
            leptons = leptons[:] + event.selectedIsoCleanTrack
        self.cleanJetsAll, cleanLeptons = cleanJetsAndLeptons(self.jets, leptons, self.jetLepDR, self.jetLepArbitration)
        self.cleanJets    = [j for j in self.cleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        self.cleanJetsFwd = [j for j in self.cleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]
        self.discardedJets = [j for j in self.jets if j not in self.cleanJetsAll]
        if hasattr(event, 'selectedLeptons') and self.cfg_ana.cleanSelectedLeptons:
            event.discardedLeptons = [ l for l in leptons if l not in cleanLeptons ]
            event.selectedLeptons  = [ l for l in event.selectedLeptons if l not in event.discardedLeptons ]

        ## Clean Jets from photons
        photons = []
        if hasattr(event, 'selectedPhotons'):
            if self.cfg_ana.cleanJetsFromFirstPhoton:
                photons = event.selectedPhotons[:1]
            else:
                photons = [ g for g in event.selectedPhotons ] 

        self.gamma_cleanJetsAll = cleanNearestJetOnly(self.cleanJetsAll, photons, self.jetGammaDR)
        self.gamma_cleanJets    = [j for j in self.gamma_cleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        self.gamma_cleanJetsFwd = [j for j in self.gamma_cleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]
        ###

        if self.cfg_ana.alwaysCleanPhotons:
            self.cleanJets = self.gamma_cleanJets
            self.cleanJetsAll = self.gamma_cleanJetsAll
            self.cleanJetsFwd = self.gamma_cleanJetsFwd

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
        ## Associate jets to taus 
        taus = getattr(event,'selectedTaus',[])
        jtaupairs = matchObjectCollection( taus, allJets, self.jetLepDR**2)

        for jet in allJets:
            jet.taus = [l for l in jtaupairs if jtaupairs[l] == jet ]
        for tau in taus:
            tau.jet = jtaupairs[tau]

        #MC stuff
        if self.cfg_comp.isMC:
            self.deltaMetFromJetSmearing = [0, 0]
            for j in self.cleanJetsAll:
                if hasattr(j, 'deltaMetFromJetSmearing'):
                    self.deltaMetFromJetSmearing[0] += j.deltaMetFromJetSmearing[0]
                    self.deltaMetFromJetSmearing[1] += j.deltaMetFromJetSmearing[1]
            self.cleanGenJets = cleanNearestJetOnly(self.genJets, leptons, self.jetLepDR)
            
            if self.cfg_ana.cleanGenJetsFromPhoton:
                self.cleanGenJets = cleanNearestJetOnly(self.cleanGenJets, photons, self.jetLepDR)

            
            #event.nGenJets25 = 0
            #event.nGenJets25Cen = 0
            #event.nGenJets25Fwd = 0
            #for j in event.cleanGenJets:
            #    event.nGenJets25 += 1
            #    if abs(j.eta()) <= 2.4: event.nGenJets25Cen += 1
            #    else:                   event.nGenJets25Fwd += 1
                    
            self.jetFlavour(event)

        setattr(event,"rho"                    +self.cfg_ana.collectionPostFix, self.rho                    ) 
        setattr(event,"deltaMetFromJEC"        +self.cfg_ana.collectionPostFix, self.deltaMetFromJEC        ) 
        setattr(event,"allJetsUsedForMET"      +self.cfg_ana.collectionPostFix, self.allJetsUsedForMET      ) 
        setattr(event,"jets"                   +self.cfg_ana.collectionPostFix, self.jets                   ) 
        setattr(event,"jetsFailId"             +self.cfg_ana.collectionPostFix, self.jetsFailId             ) 
        setattr(event,"jetsAllNoID"            +self.cfg_ana.collectionPostFix, self.jetsAllNoID            ) 
        setattr(event,"jetsIdOnly"             +self.cfg_ana.collectionPostFix, self.jetsIdOnly             ) 
        setattr(event,"cleanJetsAll"           +self.cfg_ana.collectionPostFix, self.cleanJetsAll           ) 
        setattr(event,"cleanJets"              +self.cfg_ana.collectionPostFix, self.cleanJets              ) 
        setattr(event,"cleanJetsFwd"           +self.cfg_ana.collectionPostFix, self.cleanJetsFwd           ) 
        setattr(event,"discardedJets"          +self.cfg_ana.collectionPostFix, self.discardedJets          ) 
        setattr(event,"gamma_cleanJetsAll"     +self.cfg_ana.collectionPostFix, self.gamma_cleanJetsAll     ) 
        setattr(event,"gamma_cleanJets"        +self.cfg_ana.collectionPostFix, self.gamma_cleanJets        ) 
        setattr(event,"gamma_cleanJetsFwd"     +self.cfg_ana.collectionPostFix, self.gamma_cleanJetsFwd     ) 


	if self.cfg_comp.isMC:
            setattr(event,"cleanGenJets"           +self.cfg_ana.collectionPostFix, self.cleanGenJets           ) 
            setattr(event,"bqObjects"              +self.cfg_ana.collectionPostFix, self.bqObjects              ) 
            setattr(event,"cqObjects"              +self.cfg_ana.collectionPostFix, self.cqObjects              ) 
            setattr(event,"partons"                +self.cfg_ana.collectionPostFix, self.partons                ) 
            setattr(event,"heaviestQCDFlavour"     +self.cfg_ana.collectionPostFix, self.heaviestQCDFlavour     ) 
            setattr(event,"deltaMetFromJetSmearing"+self.cfg_ana.collectionPostFix, self.deltaMetFromJetSmearing) 
            setattr(event,"genJets"                +self.cfg_ana.collectionPostFix, self.genJets                ) 
 
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

    def jetFlavour(self,event):
        def isFlavour(x,f):
            id = abs(x.pdgId())
            if id > 999: return (id/1000)%10 == f
            if id >  99: return  (id/100)%10 == f
            return id % 100 == f



        self.bqObjects = [ p for p in event.genParticles if (p.status() == 2 and isFlavour(p,5)) ]
        self.cqObjects = [ p for p in event.genParticles if (p.status() == 2 and isFlavour(p,4)) ]

        self.partons   = [ p for p in event.genParticles if ((p.status() == 23 or p.status() == 3) and abs(p.pdgId())>0 and (abs(p.pdgId()) in [1,2,3,4,5,21]) ) ]
        match = matchObjectCollection2(self.cleanJetsAll,
                                       self.partons,
                                       deltaRMax = 0.3)

        for jet in self.cleanJetsAll:
            parton = match[jet]
            jet.partonId = (parton.pdgId() if parton != None else 0)
            jet.partonMotherId = (parton.mother(0).pdgId() if parton != None and parton.numberOfMothers()>0 else 0)
        
        for jet in self.jets:
           (bmatch, dr) = bestMatch(jet, self.bqObjects)
           if dr < 0.4:
               jet.mcFlavour = 5
           else:
               (cmatch, dr) = bestMatch(jet, self.cqObjects) 
               if dr < 0.4:
                   jet.mcFlavour = 4
               else:
                   jet.mcFlavour = 0

        self.heaviestQCDFlavour = 5 if len(self.bqObjects) else (4 if len(self.cqObjects) else 1);
 
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
                                       self.genJets,
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
    copyJetsByValue = False,      #Whether or not to copy the input jets or to work with references (should be 'True' if JetAnalyzer is run more than once)
    genJetCol = 'slimmedGenJets',
    rho = ('fixedGridRhoFastjetAll','',''),
    jetPt = 25.,
    jetEta = 4.7,
    jetEtaCentral = 2.4,
    jetLepDR = 0.4,
    jetLepArbitration = (lambda jet,lepton : lepton), # you can decide which to keep in case of overlaps; e.g. if the jet is b-tagged you might want to keep the jet
    cleanSelectedLeptons = True, #Whether to clean 'selectedLeptons' after disambiguation. Treat with care (= 'False') if running Jetanalyzer more than once
    minLepPt = 10,
    lepSelCut = lambda lep : True,
    relaxJetId = False,  
    doPuId = False, # Not commissioned in 7.0.X
    doQG = False, 
    recalibrateJets = False,
    recalibrationType = "AK4PFchs",
    shiftJEC = 0, # set to +1 or -1 to get +/-1 sigma shifts
    smearJets = True,
    shiftJER = 0, # set to +1 or -1 to get +/-1 sigma shifts    
    cleanJetsFromFirstPhoton = False,
    cleanJetsFromTaus = False,
    cleanJetsFromIsoTracks = False,
    alwaysCleanPhotons = False,
    jecPath = "",
    cleanGenJetsFromPhoton = False,
    collectionPostFix = ""
    )
)
