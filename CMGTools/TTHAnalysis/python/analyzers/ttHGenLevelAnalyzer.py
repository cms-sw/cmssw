import operator 
import itertools
import copy

from ROOT import TLorentzVector

from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.counter import Counter, Counters

from CMGTools.RootTools.physicsobjects.PhysicsObjects import GenParticle

from PhysicsTools.HeppyCore.utils.deltar import *

from CMGTools.RootTools.physicsobjects.genutils import *
        
class ttHGenLevelAnalyzer( Analyzer ):
    """Do generator-level analysis of a ttH->leptons decay:

       Creates in the event:
         event.genParticles   = the gen particles (pruned, as default)
         event.genHiggsDecayMode =   0  for non-Higgs
                                 15  for H -> tau tau
                                 23  for H -> Z Z
                                 24  for H -> W W
                                 xx  for H -> xx yy zzz 

          event.gentauleps = [ gen electrons and muons from hard scattering not from tau decays ]
          event.gentaus    = [ gen taus from from hard scattering ]
          event.genleps    = [ gen electrons and muons from hard scattering not from tau decays ]
          event.genbquarks  = [ gen b quarks from top quark decays ]
          event.genwzquarks = [ gen quarks from hadronic W,Z decays ]
          event.gennus    = [ gen nus from from Z,W ]

       If filterHiggsDecays is set to a list of Higgs decay modes,
       it will filter events that have those decay modes.
       e.g. [0, 15, 23, 24] will keep data, non-Higgs MC and Higgs decays to (tau, Z, W) 
       but will drop Higgs decays to other particles (e.g. bb).
      
    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(ttHGenLevelAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.doPDFWeights = hasattr(self.cfg_ana, "PDFWeights") and len(self.cfg_ana.PDFWeights) > 0
        if self.doPDFWeights:
            self.pdfWeightInit = False
    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------
        

    def declareHandles(self):
        super(ttHGenLevelAnalyzer, self).declareHandles()

        #mc information
        self.mchandles['genParticles'] = AutoHandle( 'prunedGenParticles',
                                                     'std::vector<reco::GenParticle>' )
        if self.doPDFWeights:
            self.mchandles['pdfstuff'] = AutoHandle( 'generator', 'GenEventInfoProduct' )

    def beginLoop(self, setup):
        super(ttHGenLevelAnalyzer,self).beginLoop( setup )

    def fillGenLeptons(self, event, particle, isTau=False, sourceId=25):
        """Get the gen level light leptons (prompt and/or from tau decays)"""

        for i in xrange( particle.numberOfDaughters() ):
            dau = GenParticle(particle.daughter(i))
            dau.sourceId = sourceId
            dau.isTau = isTau
            id = abs(dau.pdgId())
            moid = 0;
            if dau.numberOfMothers() > 0:
                moid = abs(dau.mother().pdgId())
            if id in [11,13]:
                if isTau: event.gentauleps.append(dau)
                else:     event.genleps.append(dau)
            elif id == 15:
                if moid in [22,23,24]:
                    event.gentaus.append(dau)
                self.fillGenLeptons(event, dau, True, sourceId)
            elif id in [22,23,24]:
                self.fillGenLeptons(event, dau, False, sourceId)
            elif id in [12,14,16]:
                event.gennus.append(dau)

    def fillWZQuarks(self, event, particle, isWZ=False, sourceId=25):
        """Descend daughters of 'particle', and add quarks from W,Z to event.genwzquarks
           isWZ is set to True if already processing daughters of W,Z's, to False before it"""

        for i in xrange( particle.numberOfDaughters() ):
            dau = GenParticle(particle.daughter(i))
            dau.sourceId = sourceId
            id = abs(dau.pdgId())
            if id <= 5 and isWZ:
                event.genwzquarks.append(dau)
            elif id in [22,23,24]:
                self.fillWZQuarks(event, dau, True, sourceId)

    def fillTopQuarks(self, event):
        """Get the b quarks from top decays into event.genbquarks"""

        event.gentopquarks = [ p for p in event.genParticles if abs(p.pdgId()) == 6 and p.numberOfDaughters() > 0 and abs(p.daughter(0).pdgId()) != 6 ]
        #if len(event.gentopquarks) != 2:
        #    print "Not two top quarks? \n%s\n" % event.gentopquarks

        for tq in event.gentopquarks:
            for i in xrange( tq.numberOfDaughters() ):
                dau = GenParticle(tq.daughter(i))
                if abs(dau.pdgId()) == 5:
                    dau.sourceId = 6
                    event.genbquarks.append( dau )
                elif abs(dau.pdgId()) == 24:
                    self.fillGenLeptons( event, dau, sourceId=6 )
                    self.fillWZQuarks(   event, dau, True, sourceId=6 )

    def makeMCInfo(self, event):
        event.genParticles = map( GenParticle, self.mchandles['genParticles'].product() )

        if False:
            for i,p in enumerate(event.genParticles):
                print " %5d: pdgId %+5d status %3d  pt %6.1f  " % (i, p.pdgId(),p.status(),p.pt()),
                if p.numberOfMothers() > 0:
                    imom, mom = p.motherRef().key(), p.mother()
                    print " | mother %5d pdgId %+5d status %3d  pt %6.1f  " % (imom, mom.pdgId(),mom.status(),mom.pt()),
                else:
                    print " | no mother particle                              ",
                    
                for j in xrange(min(3, p.numberOfDaughters())):
                    idau, dau = p.daughterRef(j).key(), p.daughter(j)
                    print " | dau[%d] %5d pdgId %+5d status %3d  pt %6.1f  " % (j,idau,dau.pdgId(),dau.status(),dau.pt()),
                print ""

        event.genHiggsBoson = None
        event.genVBosons = []
        event.gennus    = []
        event.genleps    = []
        event.gentauleps = []
        event.gentaus    = []
        event.genbquarks  = []
        event.genwzquarks = []
        event.gentopquarks  = []

        higgsBosons = [ p for p in event.genParticles if (p.pdgId() == 25) and p.numberOfDaughters() > 0 and abs(p.daughter(0).pdgId()) != 25 ]

        if len(higgsBosons) == 0:
            event.genHiggsDecayMode = 0

            ## Matching that can be done also on non-Higgs events
            ## First, top quarks
            self.fillTopQuarks( event )
            self.countBPartons( event )

            ## Then W,Z,gamma from hard scattering and that don't come from a top and don't rescatter
            def hasAncestor(particle, filter):
                for i in xrange(particle.numberOfMothers()):
                    mom = particle.mother(i)
                    if filter(mom) or hasAncestor(mom, filter): 
                        return True
                return False
            def hasDescendent(particle, filter):
                for i in xrange(particle.numberOfDaughters()):
                    dau = particle.daughter(i)
                    if filter(dau) or hasDescendent(dau, filter):
                        return True
                return False

            bosons = [ gp for gp in event.genParticles if gp.status() > 2 and  abs(gp.pdgId()) in [22,23,24]  ]

            if self.cfg_ana.verbose:
                print "\n =============="
                for i,p in enumerate(bosons):
                    print " %5d: pdgId %+5d status %3d  pt %6.1f  " % (i, p.pdgId(),p.status(),p.pt()),
                    if p.numberOfMothers() > 0:
                        imom, mom = p.motherRef().key(), p.mother()
                        print " | mother %5d pdgId %+5d status %3d  pt %6.1f  " % (imom, mom.pdgId(),mom.status(),mom.pt()),
                    else:
                        print " | no mother particle                              ",
                    
                    for j in xrange(min(3, p.numberOfDaughters())):
                        idau, dau = p.daughterRef(j).key(), p.daughter(j)
                        print " | dau[%d] %5d pdgId %+5d status %3d  pt %6.1f  " % (j,idau,dau.pdgId(),dau.status(),dau.pt()),
                        print ""

            for b in bosons:
                b.sourceId = -1
                if hasAncestor(b,   lambda gp : abs(gp.pdgId()) == 6): continue
                if hasDescendent(b, lambda gp : abs(gp.pdgId()) in [22,23,24] and gp.status() > 2): continue
                self.fillGenLeptons(event, b, sourceId=abs(b.pdgId()))
                self.fillWZQuarks(event, b, isWZ=True, sourceId=abs(b.pdgId()))
                #print " ===>  %5d: pdgId %+5d status %3d  pt %6.1f  " % (i, b.pdgId(),b.status(),b.pt()),
                #event.genVBosons.append(b)

        else:
            if len(higgsBosons) > 1: 
                print "More than one higgs? \n%s\n" % higgsBosons

            event.genHiggsBoson = GenParticle(higgsBosons[-1])
            event.genHiggsDecayMode = abs( event.genHiggsBoson.daughter(0).pdgId() )
            self.fillTopQuarks( event )
            self.countBPartons( event )
            self.fillWZQuarks(   event, event.genHiggsBoson )
            self.fillGenLeptons( event, event.genHiggsBoson, sourceId=25 )
            if self.cfg_ana.verbose:
                print "Higgs boson decay mode: ", event.genHiggsDecayMode
                print "Generator level prompt nus:\n", "\n".join(["\t%s" % p for p in event.gennus])
                print "Generator level prompt light leptons:\n", "\n".join(["\t%s" % p for p in event.genleps])
                print "Generator level light leptons from taus:\n", "\n".join(["\t%s" % p for p in event.gentauleps])
                print "Generator level prompt tau leptons:\n", "\n".join(["\t%s" % p for p in event.gentaus])
                print "Generator level b quarks from top:\n", "\n".join(["\t%s" % p for p in event.genbquarks])
                print "Generator level quarks from W, Z decays:\n", "\n".join(["\t%s" % p for p in event.genwzquarks])
        # make sure prompt leptons have a non-zero sourceId
        for p in event.genParticles:
            if isPromptLepton(p, True, includeTauDecays=True, includeMotherless=False):
                if getattr(p, 'sourceId', 0) == 0:
                    p.sourceId = 99

    def countBPartons(self,event):
        event.allBPartons = [ q for q in event.genParticles if abs(q.pdgId()) == 5 and abs(q.status()) == 2 and abs(q.pt()) > 15 ]
        event.allBPartons.sort(key = lambda q : q.pt(), reverse = True)
        event.bPartons = []
        for q in event.allBPartons:
            duplicate = False
            for q2 in event.bPartons:
                if deltaR(q.eta(),q.phi(),q2.eta(),q2.phi()) < 0.5:
                    duplicate = True
                    continue
            if not duplicate: event.bPartons.append(q)

    def initPDFWeights(self):
        from ROOT import PdfWeightProducerTool
        self.pdfWeightInit = True
        self.pdfWeightTool = PdfWeightProducerTool()
        for pdf in self.cfg_ana.PDFWeights:
            self.pdfWeightTool.addPdfSet(pdf+".LHgrid")
        self.pdfWeightTool.beginJob()

    def makePDFWeights(self, event):
        if not self.pdfWeightInit: self.initPDFWeights()
        self.pdfWeightTool.processEvent(self.mchandles['pdfstuff'].product())
        event.pdfWeights = {}
        for pdf in self.cfg_ana.PDFWeights:
            ws = self.pdfWeightTool.getWeights(pdf+".LHgrid")
            event.pdfWeights[pdf] = [w for w in ws]
            #print "Produced %d weights for %s: %s" % (len(ws),pdf,event.pdfWeights[pdf])

    def process(self, event):
        self.readCollections( event.input )

        ## creating a "sub-event" for this analyzer
        #myEvent = Event(event.iEv)
        #setattr(event, self.name, myEvent)
        #event = myEvent

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        # do MC level analysis
        self.makeMCInfo(event)

        # if MC and filtering on the Higgs decay mode, 
        # them do filter events
        if self.cfg_ana.filterHiggsDecays:
            if event.genHiggsDecayMode not in self.cfg_ana.filterHiggsDecays:
                return False

        # do PDF weights, if requested
        if self.doPDFWeights:
            self.makePDFWeights(event)
        return True
