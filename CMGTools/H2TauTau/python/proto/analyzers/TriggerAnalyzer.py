from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle

import PhysicsTools.HeppyCore.framework.config as cfg

class TriggerInfo(object):
    def __init__(self, name, index, fired=True, prescale=1.):
        self.name = name
        self.index = index
        self.fired = fired
        self.prescale = prescale
        self.objects = []
        self.objIds = set()

    def __str__(self):
        return 'TriggerInfo: name={name}, fired={fired}, n_objects={n_o}'.format(
            name=self.name, fired=self.fired, n_o=len(self.objects))

class TriggerAnalyzer(Analyzer):
    '''Access to trigger information, and trigger selection. The required
    trigger names need to be attached to the components.'''

    def declareHandles(self):
        super(TriggerAnalyzer, self).declareHandles()

        self.handles['triggerResultsHLT'] = AutoHandle(
            ('TriggerResults', '', 'HLT'),
            'edm::TriggerResults'
            )

        self.handles['triggerObjects'] =  AutoHandle(
            'selectedPatTrigger',
            'std::vector<pat::TriggerObjectStandAlone>'
            )
 
        self.handles['triggerPrescales'] =  AutoHandle(
            'patTrigger',
            'pat::PackedTriggerPrescales'
            )
 
    def beginLoop(self, setup):
        super(TriggerAnalyzer,self).beginLoop(setup)

        self.triggerList = self.cfg_comp.triggers
        self.vetoTriggerList = None

        if hasattr(self.cfg_comp, 'vetoTriggers'):
            self.vetoTriggerList = self.cfg_comp.vetoTriggers
           
            
        self.counters.addCounter('Trigger')
        self.counters.counter('Trigger').register('All events')
        self.counters.counter('Trigger').register('HLT')
        

    def process(self, event):
        self.readCollections(event.input)
        
        event.run = event.input.eventAuxiliary().id().run()
        event.lumi = event.input.eventAuxiliary().id().luminosityBlock()
        event.eventId = event.input.eventAuxiliary().id().event()

        triggerBits = self.handles['triggerResultsHLT'].product()
        names = event.input.object().triggerNames(triggerBits)

        preScales = self.handles['triggerPrescales'].product()

        self.counters.counter('Trigger').inc('All events')

        trigger_passed = False

        trigger_infos = []
        for trigger_name in self.triggerList:
            index = names.triggerIndex(trigger_name)
            if index == len(triggerBits):
                continue
            prescale = preScales.getPrescaleForIndex(index)
            fired = triggerBits.accept(index)

            trigger_infos.append(TriggerInfo(trigger_name, index, fired, prescale))

            if fired and (prescale == 1 or self.cfg_ana.usePrescaled):
                trigger_passed = True

        if self.cfg_ana.addTriggerObjects:
            triggerObjects = self.handles['triggerObjects'].product()
            for to in triggerObjects:
                to.unpackPathNames(names)
                for info in trigger_infos:
                    if to.hasPathName(info.name, True):
                        info.objects.append(to)
                        info.objIds.add(abs(to.pdgId()))

        event.trigger_infos = trigger_infos

        if self.cfg_ana.requireTrigger:
            if not trigger_passed:
                return False
            
        self.counters.counter('Trigger').inc('HLT')
        return True

    def __str__(self):
        tmp = super(TriggerAnalyzer,self).__str__()
        triglist = str(self.triggerList)
        return '\n'.join([tmp, triglist])

setattr(TriggerAnalyzer, 'defaultConfig', 
    cfg.Analyzer(
        class_object=TriggerAnalyzer,
        requireTrigger=True,
        usePrescaled=False,
        addTriggerObjects=True,
        # vetoTriggers=[],
    )
)
