from PhysicsTools.HeppyCore.framework.analyzer import Analyzer


class EventSelector( Analyzer ):
    """Skips events that are not in the toSelect list.

    Example:

    eventSelector = cfg.Analyzer(
        'EventSelector',
        toSelect = [
            1239742, 
            38001,
            159832
        ]
    )

    it can also be used with (run,lumi,event) tuples:

    eventSelector = cfg.Analyzer(
        'EventSelector',
        toSelect = [
            (1,40,1239742),
            (1,38,38001),
        ]
    )


    The process function of this analyzer returns False if the event number
    is not in the toSelect list.
    In this list, put actual CMS event numbers obtained by doing:
       event.input.eventAuxiliary().id().event()

    not event processing number
    in this python framework.

    This analyzer is typically inserted at the beginning of the analyzer
    sequence to skip events you don't want.
    We use it in conjonction with an
      import pdb; pdb.set_trace()
    statement in a subsequent analyzer, to debug a given event in the
    toSelect list.

    This kind of procedure if you want to synchronize your selection
    with an other person at the event level. 
    """

    def process(self, event):
        run = event.input.eventAuxiliary().id().run()
        lumi = event.input.eventAuxiliary().id().luminosityBlock()
        eId = event.input.eventAuxiliary().id().event()
        if eId in self.cfg_ana.toSelect or (run, lumi, eId) in self.cfg_ana.toSelect:
            # raise ValueError('found')
            print 'Selecting', run, lumi, eId
            return True 
        else:
            return False
