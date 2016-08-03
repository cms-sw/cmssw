import math, os
from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.HeppyCore.utils.deltar import deltaR2, deltaPhi, matchObjectCollection, matchObjectCollection2, bestMatch,matchObjectCollection3
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



def shiftJERfactor(JERShift, aeta):
        factor = 1.079 + JERShift*0.026
        if   aeta > 3.2: factor = 1.056 + JERShift * 0.191
        elif aeta > 2.8: factor = 1.395 + JERShift * 0.063
        elif aeta > 2.3: factor = 1.254 + JERShift * 0.062
        elif aeta > 1.7: factor = 1.208 + JERShift * 0.046
        elif aeta > 1.1: factor = 1.121 + JERShift * 0.029
        elif aeta > 0.5: factor = 1.099 + JERShift * 0.028
        return factor 





class JetAnalyzer( Analyzer ):
    """Taken from RootTools.JetAnalyzer, simplified, modified, added corrections    """
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(JetAnalyzer,self).__init__(cfg_ana, cfg_comp, looperName)
        mcGT   = cfg_ana.mcGT   if hasattr(cfg_ana,'mcGT')   else "PHYS14_25_V2"
        dataGT = cfg_ana.dataGT if hasattr(cfg_ana,'dataGT') else "GR_70_V2_AN1"
        self.shiftJEC = self.cfg_ana.shiftJEC if hasattr(self.cfg_ana, 'shiftJEC') else 0
        self.recalibrateJets = self.cfg_ana.recalibrateJets
        self.addJECShifts = self.cfg_ana.addJECShifts if hasattr(self.cfg_ana, 'addJECShifts') else 0
        if   self.recalibrateJets == "MC"  : self.recalibrateJets =     self.cfg_comp.isMC
        elif self.recalibrateJets == "Data": self.recalibrateJets = not self.cfg_comp.isMC
        elif self.recalibrateJets not in [True,False]: raise RuntimeError, "recalibrateJets must be any of { True, False, 'MC', 'Data' }, while it is %r " % self.recalibrateJets
       
        calculateSeparateCorrections = getattr(cfg_ana,"calculateSeparateCorrections", False);
        calculateType1METCorrection  = getattr(cfg_ana,"calculateType1METCorrection",  False);
        self.doJEC = self.recalibrateJets or (self.shiftJEC != 0) or self.addJECShifts or calculateSeparateCorrections or calculateType1METCorrection
        if self.doJEC:
          doResidual = getattr(cfg_ana, 'applyL2L3Residual', 'Data')
          if   doResidual == "MC":   doResidual = self.cfg_comp.isMC
          elif doResidual == "Data": doResidual = not self.cfg_comp.isMC
          elif doResidual not in [True,False]: raise RuntimeError, "If specified, applyL2L3Residual must be any of { True, False, 'MC', 'Data'(default)}"
          GT = getattr(cfg_comp, 'jecGT', mcGT if self.cfg_comp.isMC else dataGT)
          # Now take care of the optional arguments
          kwargs = { 'calculateSeparateCorrections':calculateSeparateCorrections,
                     'calculateType1METCorrection' :calculateType1METCorrection, }
          if kwargs['calculateType1METCorrection']: kwargs['type1METParams'] = cfg_ana.type1METParams
          # instantiate the jet re-calibrator
          self.jetReCalibrator = JetReCalibrator(GT, cfg_ana.recalibrationType, doResidual, cfg_ana.jecPath, **kwargs)
        self.doPuId = getattr(self.cfg_ana, 'doPuId', True)
        self.jetLepDR = getattr(self.cfg_ana, 'jetLepDR', 0.4)
        self.jetLepArbitration = getattr(self.cfg_ana, 'jetLepArbitration', lambda jet,lepton: lepton) 
        self.lepPtMin = getattr(self.cfg_ana, 'minLepPt', -1)
        self.lepSelCut = getattr(self.cfg_ana, 'lepSelCut', lambda lep : True)
        self.jetGammaDR =  getattr(self.cfg_ana, 'jetGammaDR', 0.4)
        self.jetGammaLepDR =  getattr(self.cfg_ana, 'jetGammaLepDR', 0.4)
        self.cleanFromLepAndGammaSimultaneously = getattr(self.cfg_ana, 'cleanFromLepAndGammaSimultaneously', False)
        if self.cleanFromLepAndGammaSimultaneously:
            if hasattr(self.cfg_ana, 'jetGammaLepDR'):
                self.jetGammaLepDR =  self.jetGammaLepDR 
            elif (self.jetGammaDR == self.jetLepDR):
                self.jetGammaLepDR = self.jetGammaDR
            else:
                raise RuntimeError, "DR for simultaneous cleaning of jets from leptons and photons is not defined, and dR(gamma, jet)!=dR(lep, jet)"
        if(self.cfg_ana.doQG):
            qgdefname="{CMSSW_BASE}/src/PhysicsTools/Heppy/data/pdfQG_AK4chs_13TeV_v2b.root"
            self.qglcalc = QGLikelihoodCalculator(getattr(self.cfg_ana,"QGpath",qgdefname).format(CMSSW_BASE= os.environ['CMSSW_BASE']))
        if not hasattr(self.cfg_ana ,"collectionPostFix"):self.cfg_ana.collectionPostFix=""

    def declareHandles(self):
        super(JetAnalyzer, self).declareHandles()
        self.handles['jets']   = AutoHandle( self.cfg_ana.jetCol, 'std::vector<pat::Jet>' )
        self.handles['genJet'] = AutoHandle( self.cfg_ana.genJetCol, 'vector<reco::GenJet>' )
        self.shiftJER = self.cfg_ana.shiftJER if hasattr(self.cfg_ana, 'shiftJER') else 0
        self.addJERShifts = self.cfg_ana.addJERShifts if hasattr(self.cfg_ana, 'addJERShifts') else 0
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
          #from ROOT.heppy import JetUtils
          allJets = map(lambda j:Jet(ROOT.heppy.JetUtils.copyJet(j)), self.handles['jets'].product())  #copy-by-value is safe if JetAnalyzer is ran more than once
        else: 
          allJets = map(Jet, self.handles['jets'].product()) 
    
        #set dummy MC flavour for all jets in case we want to ntuplize discarded jets later
        for jet in allJets:
            jet.mcFlavour = 0

        self.deltaMetFromJEC = [0.,0.]
        self.type1METCorr    = [0.,0.,0.]
#        print "before. rho",self.rho,self.cfg_ana.collectionPostFix,'allJets len ',len(allJets),'pt', [j.pt() for j in allJets]
        if self.doJEC:
            if not self.recalibrateJets:  # check point that things won't change
                jetsBefore = [ (j.pt(),j.eta(),j.phi(),j.rawFactor()) for j in allJets ]
            self.jetReCalibrator.correctAll(allJets, rho, delta=self.shiftJEC, 
                                                addCorr=True, addShifts=self.addJECShifts,
                                                metShift=self.deltaMetFromJEC, type1METCorr=self.type1METCorr )           
            if not self.recalibrateJets: 
                jetsAfter = [ (j.pt(),j.eta(),j.phi(),j.rawFactor()) for j in allJets ]
                if len(jetsBefore) != len(jetsAfter): 
                    print "ERROR: I had to recompute jet corrections, and they rejected some of the jets:\nold = %s\n new = %s\n" % (jetsBefore,jetsAfter)
                else:
                    for (told, tnew) in zip(jetsBefore,jetsAfter):
                        if (deltaR2(told[1],told[2],tnew[1],tnew[2])) > 0.0001:
                            print "ERROR: I had to recompute jet corrections, and one jet direction moved: old = %s, new = %s\n" % (told, tnew)
                        elif abs(told[0]-tnew[0])/(told[0]+tnew[0]) > 0.5e-3 or abs(told[3]-tnew[3]) > 0.5e-3:
                            print "ERROR: I had to recompute jet corrections, and one jet pt or corr changed: old = %s, new = %s\n" % (told, tnew)
        self.allJetsUsedForMET = allJets
#        print "after. rho",self.rho,self.cfg_ana.collectionPostFix,'allJets len ',len(allJets),'pt', [j.pt() for j in allJets]

        if self.cfg_comp.isMC:
            self.genJets = [ x for x in self.handles['genJet'].product() ]
            if self.cfg_ana.do_mc_match:
                for igj, gj in enumerate(self.genJets):
                    gj.index = igj
#                self.matchJets(event, allJets)
                self.matchJets(event, [ j for j in allJets if j.pt()>self.cfg_ana.jetPt ]) # To match only jets above chosen threshold
            if getattr(self.cfg_ana, 'smearJets', False):
                self.smearJets(event, allJets)


                
        
	##Sort Jets by pT 
        allJets.sort(key = lambda j : j.pt(), reverse = True)
        
        leptons = []
        if hasattr(event, 'selectedLeptons'):
            leptons = [ l for l in event.selectedLeptons if l.pt() > self.lepPtMin and self.lepSelCut(l) ]
        if self.cfg_ana.cleanJetsFromTaus and hasattr(event, 'selectedTaus'):
            leptons = leptons[:] + event.selectedTaus
        if self.cfg_ana.cleanJetsFromIsoTracks and hasattr(event, 'selectedIsoCleanTrack'):
            leptons = leptons[:] + event.selectedIsoCleanTrack

	## Apply jet selection
        self.jets = []
        self.jetsFailId = []
        self.jetsAllNoID = []
        self.jetsIdOnly = []
        for jet in allJets:
            #Check if lepton and jet have overlapping PF candidates 
            leps_with_overlaps = []
            if getattr(self.cfg_ana, 'checkLeptonPFOverlap', True):
              for i in range(jet.numberOfSourceCandidatePtrs()):
                p1 = jet.sourceCandidatePtr(i) #Ptr<Candidate> p1
                for lep in leptons:
                    for j in range(lep.numberOfSourceCandidatePtrs()):
                        p2 = lep.sourceCandidatePtr(j)
                        has_overlaps = p1.key() == p2.key() and p1.refCore().id().productIndex() == p2.refCore().id().productIndex() and p1.refCore().id().processIndex() == p2.refCore().id().processIndex()
                        if has_overlaps:
                            leps_with_overlaps += [lep]
            if len(leps_with_overlaps)>0:
                for lep in leps_with_overlaps:
                    lep.jetOverlap = jet
            if self.testJetNoID( jet ): 
                self.jetsAllNoID.append(jet) 
                if(self.cfg_ana.doQG):
                    jet.qgl_calc =  self.qglcalc.computeQGLikelihood
                    jet.qgl_rho =  rho
                if self.testJetID( jet ):
                    self.jets.append(jet)
                    self.jetsIdOnly.append(jet)
                else:
                    self.jetsFailId.append(jet)
            elif self.testJetID (jet ):
                self.jetsIdOnly.append(jet)

        jetsEtaCut = [j for j in self.jets if abs(j.eta()) <  self.cfg_ana.jetEta ]
        self.cleanJetsAll, cleanLeptons = cleanJetsAndLeptons(jetsEtaCut, leptons, self.jetLepDR, self.jetLepArbitration)

        self.cleanJets    = [j for j in self.cleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        self.cleanJetsFwd = [j for j in self.cleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]
        self.discardedJets = [j for j in self.jets if j not in self.cleanJetsAll]
        if hasattr(event, 'selectedLeptons') and self.cfg_ana.cleanSelectedLeptons:
            event.discardedLeptons = [ l for l in leptons if l not in cleanLeptons ]
            event.selectedLeptons  = [ l for l in event.selectedLeptons if l not in event.discardedLeptons ]
        for lep in leptons:
            if hasattr(lep, "jetOverlap"):
                if lep.jetOverlap in self.cleanJetsAll:
                    #print "overlap reco", lep.p4().pt(), lep.p4().eta(), lep.p4().phi(), lep.jetOverlap.p4().pt(), lep.jetOverlap.p4().eta(), lep.jetOverlap.p4().phi()
                    lep.jetOverlapIdx = self.cleanJetsAll.index(lep.jetOverlap)
                elif lep.jetOverlap in self.discardedJets:
                    #print "overlap discarded", lep.p4().pt(), lep.p4().eta(), lep.p4().phi(), lep.jetOverlap.p4().pt(), lep.jetOverlap.p4().eta(), lep.jetOverlap.p4().phi()
                    lep.jetOverlapIdx = 1000 + self.discardedJets.index(lep.jetOverlap)

        ## First cleaning, then Jet Id
        self.noIdCleanJetsAll, cleanLeptons = cleanJetsAndLeptons(self.jetsAllNoID, leptons, self.jetLepDR, self.jetLepArbitration)
        self.noIdCleanJets = [j for j in self.noIdCleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        self.noIdCleanJetsFwd = [j for j in self.noIdCleanJetsAll if abs(j.eta()) >=  self.cfg_ana.jetEtaCentral ]
        self.noIdDiscardedJets = [j for j in self.jetsAllNoID if j not in self.noIdCleanJetsAll]

        ## Clean Jets from photons (first cleaning, then Jet Id)
        photons = []
        if hasattr(event, 'selectedPhotons'):
            if self.cfg_ana.cleanJetsFromFirstPhoton:
                photons = event.selectedPhotons[:1]
            else:
                photons = [ g for g in event.selectedPhotons ] 

        self.gamma_cleanJetaAll = []
        self.gamma_noIdCleanJetsAll = []

        if self.cleanFromLepAndGammaSimultaneously:
            self.gamma_cleanJetsAll = cleanNearestJetOnly(jetsEtaCut, photons+leptons, self.jetGammaLepDR)
            self.gamma_noIdCleanJetsAll = cleanNearestJetOnly(self.jetsAllNoID, photons+leptons, self.jetGammaLepDR)
        else:
            self.gamma_cleanJetsAll = cleanNearestJetOnly(self.cleanJetsAll, photons, self.jetGammaDR)
            self.gamma_noIdCleanJetsAll = cleanNearestJetOnly(self.noIdCleanJetsAll, photons, self.jetGammaDR)

        self.gamma_cleanJets    = [j for j in self.gamma_cleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        self.gamma_cleanJetsFwd = [j for j in self.gamma_cleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]

        self.gamma_noIdCleanJets    = [j for j in self.gamma_noIdCleanJetsAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        self.gamma_noIdCleanJetsFwd = [j for j in self.gamma_noIdCleanJetsAll if abs(j.eta()) >= self.cfg_ana.jetEtaCentral ]
        ###

        if self.cfg_ana.alwaysCleanPhotons:
            self.cleanJets = self.gamma_cleanJets
            self.cleanJetsAll = self.gamma_cleanJetsAll
            self.cleanJetsFwd = self.gamma_cleanJetsFwd
            #
            self.noIdCleanJets = self.gamma_noIdCleanJets
            self.noIdCleanJetsAll = self.gamma_noIdCleanJetsAll
            self.noIdCleanJetsFwd = self.gamma_noIdCleanJetsFwd

        ## Jet Id, after jet/lepton cleaning
        self.cleanJetsFailIdAll = []
        for jet in self.noIdCleanJetsAll:
            if not self.testJetID( jet ):
                self.cleanJetsFailIdAll.append(jet)
        
        self.cleanJetsFailId = [j for j in self.cleanJetsFailIdAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        
        ## Jet Id, after jet/photon cleaning
        self.gamma_cleanJetsFailIdAll = []
        for jet in self.gamma_noIdCleanJetsAll:
            if not self.testJetID( jet ):
                self.gamma_cleanJetsFailIdAll.append(jet)

        self.gamma_cleanJetsFailId = [j for j in self.gamma_cleanJetsFailIdAll if abs(j.eta()) <  self.cfg_ana.jetEtaCentral ]
        
        ## Associate jets to leptons
        incleptons = event.inclusiveLeptons if hasattr(event, 'inclusiveLeptons') else event.selectedLeptons
        jlpairs = matchObjectCollection(incleptons, allJets, self.jetLepDR**2)

        for jet in allJets:
            jet.leptons = [l for l in jlpairs if jlpairs[l] == jet ]
        for lep in incleptons:
            jet = jlpairs[lep]
            if jet is None:
                setattr(lep,"jet"+self.cfg_ana.collectionPostFix,lep)
            else:
                setattr(lep,"jet"+self.cfg_ana.collectionPostFix,jet)
        ## Associate jets to taus 
        taus = getattr(event,'selectedTaus',[])
        jtaupairs = matchObjectCollection( taus, allJets, self.jetLepDR**2)

        for jet in allJets:
            jet.taus = [l for l in jtaupairs if jtaupairs[l] == jet ]
        for tau in taus:
            setattr(tau,"jet"+self.cfg_ana.collectionPostFix,jtaupairs[tau])

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

	    if getattr(self.cfg_ana, 'attachNeutrinos', True) and hasattr(self.cfg_ana,"genNuSelection") :
		jetNus=[x for x in event.genParticles if abs(x.pdgId()) in [12,14,16] and self.cfg_ana.genNuSelection(x) ]
	 	pairs= matchObjectCollection (jetNus, self.genJets, 0.4**2)
                
		for (nu,genJet) in pairs.iteritems() :
		     if genJet is not None :
			if not hasattr(genJet,"nu") :
				genJet.nu=nu.p4()
			else :
				genJet.nu+=nu.p4()
			
            
            if self.cfg_ana.do_mc_match:
                self.jetFlavour(event)

        if hasattr(event,"jets"+self.cfg_ana.collectionPostFix): raise RuntimeError("Event already contains a jet collection with the following postfix: "+self.cfg_ana.collectionPostFix)
        setattr(event,"rho"                    +self.cfg_ana.collectionPostFix, self.rho                    ) 
        setattr(event,"deltaMetFromJEC"        +self.cfg_ana.collectionPostFix, self.deltaMetFromJEC        ) 
        setattr(event,"type1METCorr"           +self.cfg_ana.collectionPostFix, self.type1METCorr           ) 
        setattr(event,"allJetsUsedForMET"      +self.cfg_ana.collectionPostFix, self.allJetsUsedForMET      ) 
        setattr(event,"jets"                   +self.cfg_ana.collectionPostFix, self.jets                   ) 
        setattr(event,"jetsFailId"             +self.cfg_ana.collectionPostFix, self.jetsFailId             ) 
        setattr(event,"jetsAllNoID"            +self.cfg_ana.collectionPostFix, self.jetsAllNoID            ) 
        setattr(event,"jetsIdOnly"             +self.cfg_ana.collectionPostFix, self.jetsIdOnly             ) 
        setattr(event,"cleanJetsAll"           +self.cfg_ana.collectionPostFix, self.cleanJetsAll           ) 
        setattr(event,"cleanJets"              +self.cfg_ana.collectionPostFix, self.cleanJets              ) 
        setattr(event,"cleanJetsFwd"           +self.cfg_ana.collectionPostFix, self.cleanJetsFwd           ) 
        setattr(event,"cleanJetsFailIdAll"           +self.cfg_ana.collectionPostFix, self.cleanJetsFailIdAll           ) 
        setattr(event,"cleanJetsFailId"              +self.cfg_ana.collectionPostFix, self.cleanJetsFailId              ) 
        setattr(event,"discardedJets"          +self.cfg_ana.collectionPostFix, self.discardedJets          ) 
        setattr(event,"gamma_cleanJetsAll"     +self.cfg_ana.collectionPostFix, self.gamma_cleanJetsAll     ) 
        setattr(event,"gamma_cleanJets"        +self.cfg_ana.collectionPostFix, self.gamma_cleanJets        ) 
        setattr(event,"gamma_cleanJetsFwd"     +self.cfg_ana.collectionPostFix, self.gamma_cleanJetsFwd     ) 
        setattr(event,"gamma_cleanJetsFailIdAll"     +self.cfg_ana.collectionPostFix, self.gamma_cleanJetsFailIdAll     ) 
        setattr(event,"gamma_cleanJetsFailId"        +self.cfg_ana.collectionPostFix, self.gamma_cleanJetsFailId        ) 


        if self.cfg_comp.isMC:
            setattr(event,"deltaMetFromJetSmearing"+self.cfg_ana.collectionPostFix, self.deltaMetFromJetSmearing) 
            setattr(event,"cleanGenJets"           +self.cfg_ana.collectionPostFix, self.cleanGenJets           )
            setattr(event,"genJets"                +self.cfg_ana.collectionPostFix, self.genJets                )
            if self.cfg_ana.do_mc_match:
                setattr(event,"bqObjects"              +self.cfg_ana.collectionPostFix, self.bqObjects              )
                setattr(event,"cqObjects"              +self.cfg_ana.collectionPostFix, self.cqObjects              )
                setattr(event,"partons"                +self.cfg_ana.collectionPostFix, self.partons                )
                setattr(event,"heaviestQCDFlavour"     +self.cfg_ana.collectionPostFix, self.heaviestQCDFlavour     )

 
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
               #8 TeV tables
               factor = shiftJERfactor(self.shiftJER, aeta)
               ptscale = max(0.0, (jetpt + (factor-1)*(jetpt-genpt))/jetpt)             
               #print "get with pt %.1f (gen pt %.1f, ptscale = %.3f)" % (jetpt,genpt,ptscale)
               jet.deltaMetFromJetSmearing = [ -(ptscale-1)*jet.rawFactor()*jet.px(), -(ptscale-1)*jet.rawFactor()*jet.py() ]
               if ptscale != 0:
                  jet.setP4(jet.p4()*ptscale)
                  # leave the uncorrected unchanged for sync
                  jet.setRawFactor(jet.rawFactor()/ptscale)
            #else: print "jet with pt %.1d, eta %.2f is unmatched" % (jet.pt(), jet.eta())
               if (self.shiftJER==0) and (self.addJERShifts):
                   setattr(jet, "corrJER", ptscale )
                   factorJERUp= shiftJERfactor(1, aeta)
                   ptscaleJERUp = max(0.0, (jetpt + (factorJERUp-1)*(jetpt-genpt))/jetpt)
                   setattr(jet, "corrJERUp", ptscaleJERUp)
                   factorJERDown= shiftJERfactor(-1, aeta)
                   ptscaleJERDown = max(0.0, (jetpt + (factorJERDown-1)*(jetpt-genpt))/jetpt)
                   setattr(jet, "corrJERDown", ptscaleJERDown)





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
    checkLeptonPFOverlap = True,
    recalibrateJets = False,
    applyL2L3Residual = 'Data', # if recalibrateJets, apply L2L3Residual to Data only
    recalibrationType = "AK4PFchs",
    shiftJEC = 0, # set to +1 or -1 to apply +/-1 sigma shift to the nominal jet energies
    addJECShifts = False, # if true, add  "corr", "corrJECUp", and "corrJECDown" for each jet (requires uncertainties to be available!)
    smearJets = True,
    shiftJER = 0, # set to +1 or -1 to get +/-1 sigma shifts    
    jecPath = "",
    calculateSeparateCorrections = False,
    calculateType1METCorrection  = False,
    type1METParams = { 'jetPtThreshold':15., 'skipEMfractionThreshold':0.9, 'skipMuons':True },
    addJERShifts = 0, # add +/-1 sigma shifts to jets, intended to be used with shiftJER=0
    cleanJetsFromFirstPhoton = False,
    cleanJetsFromTaus = False,
    cleanJetsFromIsoTracks = False,
    alwaysCleanPhotons = False,
    do_mc_match=True,
    cleanGenJetsFromPhoton = False,
    jetGammaDR=0.4,
    cleanFromLepAndGammaSimultaneously = False,
    jetGammaLepDR=0.4,
    attachNeutrinos = True,
    genNuSelection = lambda nu : True, #FIXME: add here check for ispromptfinalstate
    collectionPostFix = ""
    )
)
