from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.statistics.Counter import Counter
from CMGTools.RootTools.utils.TriggerList import TriggerList
from CMGTools.RootTools.utils.TriggerMatching import selTriggerObjects
from CMGTools.RootTools.physicsobjects.PhysicsObjects import TriggerObject

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
 
    def beginLoop(self):
        super(TriggerAnalyzer,self).beginLoop()
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

        #if iEvent.eventAuxiliary().id().event() in notPassed :
        #  print 'before anything'
        #  import pdb ; pdb.set_trace()
        
        
        event.triggerObject = self.handles['cmgTriggerObjectSel'].product()[0]
        run = iEvent.eventAuxiliary().id().run()
        lumi = iEvent.eventAuxiliary().id().luminosityBlock()
        eventId = iEvent.eventAuxiliary().id().event()

        event.run = run
        event.lumi = lumi
        event.eventId = eventId

        ## if component is embed return (has no trigger obj) RHEMB will have trigger!
        if self.cfg_comp.isEmbed and len(self.cfg_comp.triggers)==0 :
          return True

##        if self.cfg_ana.verbose:
##            self.printTriggerObject( event.triggerObject )
        
        self.counters.counter('Trigger').inc('All events')
        # import pdb; pdb.set_trace()
        usePrescaled = False
        if hasattr( self.cfg_ana, 'usePrescaled'):
            usePrescaled = self.cfg_ana.usePrescaled

        # import pdb; pdb.set_trace()
                
        ### want to check whether more than one unprescaled trigger has been fired
        hltPathVec = []
        
        self.triggerList = TriggerList( self.cfg_comp.triggers )
        
        passed, hltPath = self.triggerList.triggerPassed(event.triggerObject,
                                                         run, lumi, self.cfg_comp.isData,
                                                         self.cfg_comp.isEmbed,
                                                         usePrescaled = usePrescaled)

        if passed and not hltPath == None:
          hltPathVec.append(hltPath)
        
        
        if passed and not hltPath == None:
          for tr in self.cfg_comp.triggers :       
            if tr in hltPathVec : 
              for triggerToRemove in self.triggerList.triggerList :
                if triggerToRemove.name == tr :
                  self.triggerList.triggerList.remove(triggerToRemove)

              passed2, hltPath2 = self.triggerList.triggerPassed(event.triggerObject,
                                                               run, lumi, self.cfg_comp.isData,
                                                               self.cfg_comp.isEmbed,
                                                               usePrescaled = usePrescaled)
              if passed2 and not hltPath2 == None:
                hltPathVec.append(hltPath2)
        
        event.hltPaths = set(hltPathVec)
        
        #Check the veto!
        veto=False
        if self.vetoTriggerList is not None:
            veto,hltVetoPath = self.vetoTriggerList.triggerPassed(event.triggerObject,
                                                         run,lumi,self.cfg_comp.isData,
                                                         self.cfg_comp.isEmbed,
                                                         usePrescaled = usePrescaled)

        # Check if events needs to be skipped if no trigger is found (useful for generator level studies)
        keepFailingEvents = False
        #keepFailingEvents = True
        if hasattr( self.cfg_ana, 'keepFailingEvents'):
            keepFailingEvents = self.cfg_ana.keepFailingEvents
        if not passed or (passed and veto):
            event.passedTriggerAnalyzer = False
            if not keepFailingEvents:
                #if iEvent.eventAuxiliary().id().event() in notPassed :
                #  print 'before anything'
                #  import pdb ; pdb.set_trace()
                return False
        else:
            event.passedTriggerAnalyzer = True

        #import pdb ; pdb.set_trace()
        event.hltPath = hltPath 


        ### Riccardo: I want the trigger objects corresponding to the trigger I want to fire even if it has not been fired
        if hltPath is not None :
          trigObjs = map( TriggerObject, self.handles['cmgTriggerObjectListSel'].product())
          # selecting the trigger objects used in this path
          event.triggerObjects = selTriggerObjects( trigObjs, hltPath )
        elif keepFailingEvents :
          event.triggerObjects = []
          for hltPath in self.cfg_comp.triggers :
            trigObjs = map( TriggerObject, self.handles['cmgTriggerObjectListSel'].product())
            event.triggerObjects.extend( selTriggerObjects( trigObjs, hltPath, skipPath=True ) )
          hltPath = None
          #import pdb ; pdb.set_trace()

#         if hltPath is not None:
#             trigObjs = map( TriggerObject,
#                             self.handles['cmgTriggerObjectListSel'].product())
#             # selecting the trigger objects used in this path
#             event.triggerObjects = selTriggerObjects( trigObjs, hltPath )
            
        self.counters.counter('Trigger').inc('HLT')
        event.TriggerFired = 1
        return True

    def write(self):
        print 'writing TriggerAnalyzer'
        super(TriggerAnalyzer, self).write()
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
