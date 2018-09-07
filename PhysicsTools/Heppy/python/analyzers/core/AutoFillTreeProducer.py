from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
#from ROOT import TriggerBitChecker
from PhysicsTools.Heppy.analyzers.core.autovars import *
from PhysicsTools.Heppy.analyzers.objects.autophobj  import *

import six

class AutoFillTreeProducer( TreeAnalyzerNumpy ):

    #-----------------------------------
    # BASIC TREE PRODUCER 
    #-----------------------------------
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(AutoFillTreeProducer,self).__init__(cfg_ana, cfg_comp, looperName)

        ## Read whether we want vectors or flat trees
        self.scalar = not self.cfg_ana.vectorTree

        ## Read whether we want 4-vectors 
        if not getattr(self.cfg_ana, 'saveTLorentzVectors', False):
            fourVectorType.removeVariable("p4")


        self.collections = {}      
        self.globalObjects = {}
        self.globalVariables = []
        if hasattr(cfg_ana,"collections"):
            self.collections.update(cfg_ana.collections)
        if hasattr(cfg_ana,"globalObjects"):
            self.globalObjects.update(cfg_ana.globalObjects)
        if hasattr(cfg_ana,"globalVariables"):
            self.globalVariables=cfg_ana.globalVariables[:]

    def beginLoop(self, setup) :
        super(AutoFillTreeProducer, self).beginLoop(setup)

    def declareHandles(self):
        super(AutoFillTreeProducer, self).declareHandles()
#        self.handles['TriggerResults'] = AutoHandle( ('TriggerResults','','HLT'), 'edm::TriggerResults' )
        self.mchandles['GenInfo'] = AutoHandle( ('generator','',''), 'GenEventInfoProduct' )
        for k,v in six.iteritems(self.collections):
            if isinstance(v, tuple) and isinstance(v[0], AutoHandle):
                self.handles[k] = v[0]

    def declareCoreVariables(self, tr, isMC):
        """Here we declare the variables that we always want and that are hard-coded"""
        tr.var('run', int, storageType="i")
        tr.var('lumi', int, storageType="i")
        tr.var('evt', int, storageType="l")
        tr.var('isData', int)

 #       self.triggerBitCheckers = []
 #       if hasattr(self.cfg_ana, 'triggerBits'):
 #           for T, TL in six.iteritems(self.cfg_ana.triggerBits):
 #               trigVec = ROOT.vector(ROOT.string)()
 #               for TP in TL:
 #                   trigVec.push_back(TP)
 #               tr.var( 'HLT_'+T, int )
#                self.triggerBitCheckers.append( (T, TriggerBitChecker(trigVec)) )

        if not isMC:
            tr.var('intLumi', int, storageType="i")

        if isMC:
            ## cross section
            tr.var('xsec', float)
            ## PU weights
            tr.var("puWeight")
            ## number of true interactions
            tr.var("nTrueInt")
            ## generator weight
            tr.var("genWeight")
            ## PDF weights
            self.pdfWeights = []
            if hasattr(self.cfg_ana, "PDFWeights") and len(self.cfg_ana.PDFWeights) > 0:
                self.pdfWeights = self.cfg_ana.PDFWeights
                for (pdf,nvals) in self.pdfWeights:
                    if self.scalar:
                        for i in range(nvals): tr.var('pdfWeight_%s_%d' % (pdf,i))
                    else:
                        tr.vector('pdfWeight_%s' % pdf, nvals)

    def declareVariables(self,setup):
        isMC = self.cfg_comp.isMC 
        tree = self.tree
        self.declareCoreVariables(tree, isMC)

        if not hasattr(self.cfg_ana,"ignoreAnalyzerBookings") or not self.cfg_ana.ignoreAnalyzerBookings :
            #import variables declared by the analyzers
            if hasattr(setup,"globalVariables"):
                self.globalVariables+=setup.globalVariables
            if hasattr(setup,"globalObjects"):
                self.globalObjects.update(setup.globalObjects)
            if hasattr(setup,"collections"):
                self.collections.update(setup.collections)

        for v in self.globalVariables:
            v.makeBranch(tree, isMC)
        for o in six.itervalues(self.globalObjects): 
            o.makeBranches(tree, isMC)
        for c in six.itervalues(self.collections):
            if isinstance(c, tuple): c = c[-1]
            if self.scalar:
                c.makeBranchesScalar(tree, isMC)
            else:
                c.makeBranchesVector(tree, isMC)

    def fillCoreVariables(self, tr, event, isMC):
        """Here we fill the variables that we always want and that are hard-coded"""
        tr.fill('run', event.input.eventAuxiliary().id().run())
        tr.fill('lumi',event.input.eventAuxiliary().id().luminosityBlock())
        tr.fill('evt', event.input.eventAuxiliary().id().event())    
        tr.fill('isData', 0 if isMC else 1)

#       triggerResults = self.handles['TriggerResults'].product()
#       for T,TC in self.triggerBitCheckers:
#           tr.fill("HLT_"+T, TC.check(event.object(), triggerResults))

        if not isMC:
            tr.fill('intLumi', getattr(self.cfg_comp,'intLumi',1.0))

        if isMC:
            ## xsection, if available
            tr.fill('xsec', getattr(self.cfg_comp,'xSection',1.0))
            ## PU weights, check if a PU analyzer actually filled it
            if hasattr(event,"nPU"):
                tr.fill("nTrueInt", event.nPU)
                tr.fill("puWeight", event.puWeight)
            else :
                tr.fill("nTrueInt", -1)
                tr.fill("puWeight", 1.0)

            tr.fill("genWeight", self.mchandles['GenInfo'].product().weight())
            ## PDF weights
            if hasattr(event,"pdfWeights") :
                for (pdf,nvals) in self.pdfWeights:
                    if len(event.pdfWeights[pdf]) != nvals:
                        raise RuntimeError("PDF lenght mismatch for %s, declared %d but the event has %d" % (pdf,nvals,event.pdfWeights[pdf]))
                    if self.scalar:
                        for i,w in enumerate(event.pdfWeights[pdf]):
                            tr.fill('pdfWeight_%s_%d' % (pdf,i), w)
                    else:
                        tr.vfill('pdfWeight_%s' % pdf, event.pdfWeights[pdf])

    def process(self, event):
        if hasattr(self.cfg_ana,"filter") :	
            if not self.cfg_ana.filter(event) :
                return True #do not stop processing, just filter myself
        self.readCollections( event.input)
        self.fillTree(event)

    def fillTree(self, event, resetFirst=True):
        isMC = self.cfg_comp.isMC 
        if resetFirst: self.tree.reset()

        self.fillCoreVariables(self.tree, event, isMC)

        for v in self.globalVariables:
            if not isMC and v.mcOnly: continue
            v.fillBranch(self.tree, event, isMC)

        for on, o in six.iteritems(self.globalObjects): 
            if not isMC and o.mcOnly: continue
            o.fillBranches(self.tree, getattr(event, on), isMC)

        for cn, c in six.iteritems(self.collections):
            if isinstance(c, tuple) and isinstance(c[0], AutoHandle):
                if not isMC and c[-1].mcOnly: continue
                objects = self.handles[cn].product()
                setattr(event, cn, [objects[i] for i in xrange(objects.size())])
                c = c[-1]
            if not isMC and c.mcOnly: continue
            if self.scalar:
                c.fillBranchesScalar(self.tree, getattr(event, cn), isMC)
            else:
                c.fillBranchesVector(self.tree, getattr(event, cn), isMC)

        self.tree.tree.Fill()      

    def getPythonWrapper(self):
        """
        This function produces a string that contains a Python wrapper for the event.
        The wrapper is automatically generated based on the collections and allows the full
        event contents to be accessed from subsequent Analyzers using e.g.

        leps = event.selLeptons #is of type selLeptons
        pt0 = leps[0].pt

        One just needs to add the EventAnalyzer to the sequence.
        """

        isMC = self.cfg_comp.isMC 

        classes = ""
        anclass = ""
        anclass += "from PhysicsTools.HeppyCore.framework.analyzer import Analyzer\n"
        anclass += "class EventAnalyzer(Analyzer):\n"
        anclass += "    def __init__(self, cfg_ana, cfg_comp, looperName):\n"
        anclass += "        super(EventAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)\n"

        anclass += "    def process(self, event):\n"

        for cname, coll in self.collections.items():
            classes += coll.get_py_wrapper_class(isMC)
            anclass += "        event.{0} = {0}.make_array(event)\n".format(coll.name)

        return classes + "\n" + anclass

