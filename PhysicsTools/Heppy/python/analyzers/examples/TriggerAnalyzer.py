from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.statistics.counter import Counter
from PhysicsTools.HeppyCore.utils.TriggerList import TriggerList
from PhysicsTools.HeppyCore.utils.TriggerMatching import selTriggerObjects
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import TriggerObject



class TriggerAnalyzer( Analyzer ):
    '''Access to trigger information, and trigger selection'''

    def declareHandles(self):
        super(TriggerAnalyzer, self).declareHandles()

        self.handles['cmgTriggerObjectSel'] =  AutoHandle(
            'cmgTriggerObjectSel',
            'std::vector<cmg::TriggerObject>'
            )
 
        self.handles['cmgTriggerObjectListSel'] =  AutoHandle(
            'cmgTriggerObjectListSel',
            'std::vector<cmg::TriggerObject>'
            )
 
    def beginLoop(self, setup):
        super(TriggerAnalyzer,self).beginLoop(setup)
        self.triggerList = TriggerList( self.cfg_comp.triggers )
        if hasattr(self.cfg_comp,'vetoTriggers'):
            self.vetoTriggerList = TriggerList( self.cfg_comp.vetoTriggers )
        else:
            self.vetoTriggerList = None
            
        self.counters.addCounter('Trigger')
        self.counters.counter('Trigger').register('All events')
        self.counters.counter('Trigger').register('HLT')
        

    def process(self, iEvent, event):
        self.readCollections( iEvent )
        
        event.triggerObject = self.handles['cmgTriggerObjectSel'].product()[0]
        run = iEvent.eventAuxiliary().id().run()
        lumi = iEvent.eventAuxiliary().id().luminosityBlock()
        eventId = iEvent.eventAuxiliary().id().event()

        event.run = run
        event.lumi = lumi
        event.eventId = eventId

##        if self.cfg_ana.verbose:
##            self.printTriggerObject( event.triggerObject )
        
        self.counters.counter('Trigger').inc('All events')
        # import pdb; pdb.set_trace()
        usePrescaled = False
        if hasattr( self.cfg_ana, 'usePrescaled'):
            usePrescaled = self.cfg_ana.usePrescaled

        # import pdb; pdb.set_trace()
        passed, hltPath = self.triggerList.triggerPassed(event.triggerObject,
                                                         run, lumi, self.cfg_comp.isData,
                                                         usePrescaled = usePrescaled)



        #Check the veto!
        veto=False
        if self.vetoTriggerList is not None:
            veto,hltVetoPath = self.vetoTriggerList.triggerPassed(event.triggerObject,
                                                         run,lumi,self.cfg_comp.isData,
                                                         usePrescaled = usePrescaled)

        # Check if events needs to be skipped if no trigger is found (useful for generator level studies)
        keepFailingEvents = False
        if hasattr( self.cfg_ana, 'keepFailingEvents'):
            keepFailingEvents = self.cfg_ana.keepFailingEvents
        if not passed or (passed and veto):
            event.passedTriggerAnalyzer = False
            if not keepFailingEvents:
                return False
        else:
            event.passedTriggerAnalyzer = True

        event.hltPath = hltPath 

        if hltPath is not None:
            trigObjs = map( TriggerObject,
                            self.handles['cmgTriggerObjectListSel'].product())
            # selecting the trigger objects used in this path
            event.triggerObjects = selTriggerObjects( trigObjs, hltPath )
            
        self.counters.counter('Trigger').inc('HLT')
        event.TriggerFired = 1
        return True

    def write(self, setup):
        print 'writing TriggerAnalyzer'
        super(TriggerAnalyzer, self).write(setup)
        self.triggerList.write( self.dirName )

    def __str__(self):
        tmp = super(TriggerAnalyzer,self).__str__()
        triglist = str( self.triggerList )
        return '\n'.join( [tmp, triglist ] )


##     def printTriggerObject(self, object):
##         '''FIXME : we need a trigger object class in physicsobjects.'''
##         print 'trig obj', object.pdgId(), object.pt(), object.charge(), object.eta(), object.phi()
##         for name in object.getSelectionNames():
##             hasSel = object.getSelection( name )
##             if self.cfg_ana.verbose==1 and hasSel:
##                 print name, hasSel
##             elif self.cfg_ana.verbose==2:
##                 print name, hasSel
