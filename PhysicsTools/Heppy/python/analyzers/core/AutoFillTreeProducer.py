from PhysicsTools.Heppy.analyzers.core.TreeAnalyzerNumpy import TreeAnalyzerNumpy
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
#from ROOT import TriggerBitChecker
from PhysicsTools.Heppy.analyzers.core.autofillobjects import *
from PhysicsTools.Heppy.analyzers.objects.autofilltypes  import *


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

        ## Declare how we store floats by default
        self.tree.setDefaultFloatType("F"); # otherwise it's "D"
 
	self.collections = {}      
        self.globalObjects = {}
	self.globalVariables = {}
	if hasattr(cfg_ana,"collections"):
		self.collections=cfg_ana.collections
	if hasattr(cfg_ana,"globalObjects"):
		self.globalObjects=cfg_ana.globalObjects
	if hasattr(cfg_ana,"globalVariables"):
		self.globalVariables=cfg_ana.globalVariables

        if self.__class__.__name__ == "AutoFillTreeProducer":
            self.initDone = True
            self.declareVariables() 

    def declareHandles(self):
        super(AutoFillTreeProducer, self).declareHandles()
#        self.handles['TriggerResults'] = AutoHandle( ('TriggerResults','','HLT'), 'edm::TriggerResults' )
        self.mchandles['GenInfo'] = AutoHandle( ('generator','',''), 'GenEventInfoProduct' )
        for k,v in self.collections.iteritems():
            if type(v) == tuple and isinstance(v[0], AutoHandle):
                self.handles[k] = v[0]

    def declareCoreVariables(self, tr, isMC):
        """Here we declare the variables that we always want and that are hard-coded"""
        tr.var('run', int, storageType="i")
        tr.var('lumi', int, storageType="i")
        tr.var('evt', int, storageType="i")
        tr.var('isData', int)

 #       self.triggerBitCheckers = []
 #       if hasattr(self.cfg_ana, 'triggerBits'):
 #           for T, TL in self.cfg_ana.triggerBits.iteritems():
 #               trigVec = ROOT.vector(ROOT.string)()
 #               for TP in TL:
 #                   trigVec.push_back(TP)
 #               tr.var( 'HLT_'+T, int )
#                self.triggerBitCheckers.append( (T, TriggerBitChecker(trigVec)) )
 
        if isMC:
            ## PU weights
            tr.var("puWeight")
            ## number of true interactions
            tr.var("nTrueInt",int)
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

    def declareVariables(self):
        if not hasattr(self,'initDone'): return
        isMC = self.cfg_comp.isMC 
        tree = self.tree
        self.declareCoreVariables(tree, isMC)

        for v in self.globalVariables:
            v.makeBranch(tree, isMC)
        for o in self.globalObjects.itervalues(): 
            o.makeBranches(tree, isMC)
        for c in self.collections.itervalues():
            if type(c) == tuple: c = c[-1]
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

        if isMC:
            ## PU weights, check if a PU analyzer actually filled it
	    if event.hasattr("nPU"):
	            tr.fill("nTrueInt", event.nPU)
	            tr.fill("puWeight", event.eventWeight)
	    else :
                    tr.fill("nTrueInt", -1)
	            tr.fill("puWeight", 1.0)
		
            tr.fill("genWeight", self.mchandles['GenInfo'].product().weight())
            ## PDF weights
            if event.hasattr("pdfWeights") :
              for (pdf,nvals) in self.pdfWeights:
		if len(event.pdfWeights[pdf]) != nvals:
                    raise RuntimeError, "PDF lenght mismatch for %s, declared %d but the event has %d" % (pdf,nvals,event.pdfWeights[pdf])
                if self.scalar:
                    for i,w in enumerate(event.pdfWeights[pdf]):
                        tr.fill('pdfWeight_%s_%d' % (pdf,i), w)
                else:
                    tr.vfill('pdfWeight_%s' % pdf, event.pdfWeights[pdf])

    def process(self, event):
        self.readCollections( event.input)
        self.fillTree(event)
         
    def fillTree(self, event, resetFirst=True):
        isMC = self.cfg_comp.isMC 
        if resetFirst: self.tree.reset()

        self.fillCoreVariables(self.tree, event, isMC)

        for v in self.globalVariables:
            if not isMC and v.mcOnly: continue
            v.fillBranch(self.tree, event, isMC)

        for on, o in self.globalObjects.iteritems(): 
            if not isMC and o.mcOnly: continue
            o.fillBranches(self.tree, getattr(event, on), isMC)

        for cn, c in self.collections.iteritems():
            if type(c) == tuple and isinstance(c[0], AutoHandle):
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

