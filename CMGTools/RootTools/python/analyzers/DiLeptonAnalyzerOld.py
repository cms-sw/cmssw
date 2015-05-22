import operator 
from CMGTools.RootTools.fwlite.Analyzer import Analyzer
from CMGTools.RootTools.statistics.Counter import Counter, Counters
from CMGTools.RootTools.fwlite.AutoHandle import AutoHandle
from CMGTools.RootTools.physicsobjects.DiObject import DiObject
from CMGTools.RootTools.physicsobjects.PhysicsObjects import Lepton
from CMGTools.RootTools.utils.TriggerMatching import triggerMatched

class DiLeptonAnalyzer( Analyzer ):

    # The DiObject class will be used as the di-object class
    # and the Lepton class as the lepton class
    # Child classes override this choice, and can e.g. decide to use
    # the TauMuon class as a di-object class
    # ... not sure other people can understand this comment ;-)
    DiObjectClass = DiObject
    LeptonClass = Lepton 

    def beginLoop(self):
        super(DiLeptonAnalyzer,self).beginLoop()
        self.counters.addCounter('DiLepton')
        count = self.counters.counter('DiLepton')
        count.register('all events')
        count.register('> 0 di-lepton')
        # count.register('di-lepton cut string ok')
        count.register('lepton accept')
        count.register('leg1 offline cuts passed')
        count.register('leg1 trig matched')
        count.register('leg2 offline cuts passed')
        count.register('leg2 trig matched')
        count.register('exactly 1 di-lepton')
        count.register('{min:3.1f} < m < {max:3.1f}'.format( min = self.cfg_ana.m_min,
                                                             max = self.cfg_ana.m_max ))
        

    def buildDiLeptons(self, cmgDiLeptons, event):
        '''Creates python DiLeptons from the di-leptons read from the disk.
        to be overloaded if needed.'''
        return map( self.__class__.DiObjectClass, cmgDiLeptons )


    def buildLeptons(self, cmgLeptons, event):
        '''Creates python Leptons from the leptons read from the disk.
        to be overloaded if needed.'''
        return map( self.__class__.LeptonClass, cmgLeptons )


        
    def process(self, iEvent, event):
        # access di-object collection
        # test first leg
        # test second leg
        # test di-lepton
        # apply lepton veto
        # choose best di-lepton
        # put in the event
        self.readCollections( iEvent )
        # trigger stuff could be put in a separate analyzer
        # event.triggerObject = self.handles['cmgTriggerObjectSel'].product()[0]
        event.diLeptons = self.buildDiLeptons( self.handles['diLeptons'].product(), event )
        event.leptons = self.buildLeptons( self.handles['leptons'].product(), event )

        self.counters.counter('DiLepton').inc('all events')
        # if not self.triggerList.triggerPassed(event.triggerObject):
        #    return False
        # self.counters.counter('DiLepton').inc('trigger passed ')

        # if event.eventId == 155035:
        #    import pdb; pdb.set_trace()
            
        if len(event.diLeptons) == 0:
            return False
        self.counters.counter('DiLepton').inc('> 0 di-lepton')

        # testing di-lepton itself
        selDiLeptons = event.diLeptons
        # selDiLeptons = self.selectDiLeptons( selDiLeptons ) 
        
        if not self.leptonAccept( event.leptons ):
            return False
        self.counters.counter('DiLepton').inc('lepton accept')

        # testing leg1
        selDiLeptons = [ diL for diL in selDiLeptons if \
                         self.testLeg1( diL.leg1() ) ]
        if len(selDiLeptons) == 0:
            return False
        else:
            self.counters.counter('DiLepton').inc('leg1 offline cuts passed')

        if len(self.cfg_comp.triggers)>0:
            # trigger matching leg1
            selDiLeptons = [diL for diL in selDiLeptons if \
                            self.trigMatched(event, diL.leg1(), 'leg1')]
            if len(selDiLeptons) == 0:
                return False
            else:
                self.counters.counter('DiLepton').inc('leg1 trig matched')

        # testing leg2 
        selDiLeptons = [ diL for diL in selDiLeptons if \
                         self.testLeg2( diL.leg2() ) ]
        if len(selDiLeptons) == 0:
            return False
        else:
            self.counters.counter('DiLepton').inc('leg2 offline cuts passed')

        if len(self.cfg_comp.triggers)>0:
            # trigger matching leg2
            selDiLeptons = [diL for diL in selDiLeptons if \
                            self.trigMatched(event, diL.leg2(), 'leg2')]
            if len(selDiLeptons) == 0:
                return False
            else:
                self.counters.counter('DiLepton').inc('leg2 trig matched')

        # mass cut 
        selDiLeptons = [ diL for diL in selDiLeptons if \
                         self.testMass(diL) ]
        if len(selDiLeptons)==0:
            return False
        else:
            self.counters.counter('DiLepton').inc(
                '{min:3.1f} < m < {max:3.1f}'.format( min = self.cfg_ana.m_min,
                                                      max = self.cfg_ana.m_max )
                )

        # exactly one? 
        if len(selDiLeptons)==0:
            return False
        elif len(selDiLeptons)==1:
            self.counters.counter('DiLepton').inc('exactly 1 di-lepton')
        
        event.diLepton = self.bestDiLepton( selDiLeptons )
        event.leg1 = event.diLepton.leg1()
        event.leg2 = event.diLepton.leg2()

        return True


    def declareHandles(self):        
        super(DiLeptonAnalyzer, self).declareHandles()
        self.handles['cmgTriggerObjectSel'] =  AutoHandle(
            'cmgTriggerObjectSel',
            'std::vector<cmg::TriggerObject>'
            )
    
    def leptonAccept(self, leptons):
        '''Should implement a default version running on event.leptons.'''
        return True


##     def selectDiLeptons(self, diLeptons, cutString=None):
##         '''Returns the list of input di-leptons which verify the cutstring'''
##         if cutString is None:
##             if not hasattr( self.cfg_ana, 'diLeptonCutString' ):
##                 return diLeptons
##             else:
##                 cutString = self.cfg_ana.diLeptonCutString 
##         selDiLeptons = [ diL for diL in diLeptons if \
##                          diL.getSelection(cutString) ]
##         if len(selDiLeptons) > 0:
##             self.counters.counter('DiLepton').inc( 'di-lepton cut string ok')
##         return selDiLeptons


    def testLeg(self, leg, pt, eta, iso, sel=None):
        if sel is not None and \
           not leg.getSelection(sel):
            # a cut string has to be tested, and the leg does not pass
            return False
        if leg.pt() > pt and \
           abs(leg.eta()) < eta and \
           leg.relIso( 0.5 ) < iso:
            return True
        else:
            return False

    
    def testLeg1(self, leg):
        '''Overload according to type, see e.g. TauMuAnalyzer.'''
        sel = None
        if hasattr(self.cfg_ana, 'cutString1'):
            sel = self.cfg_ana.cutString1
        return self.testLeg( leg,
                             pt = self.cfg_ana.pt1,
                             eta = self.cfg_ana.eta1,
                             iso = self.cfg_ana.iso1,
                             sel = sel )


    def testLeg2(self, leg):
        '''Overload according to type, see e.g. TauMuAnalyzer.'''
        sel = None
        if hasattr(self.cfg_ana, 'cutString2'):
            sel = self.cfg_ana.cutString2
        return self.testLeg( leg,
                             pt = self.cfg_ana.pt2,
                             eta = self.cfg_ana.eta2,
                             iso = self.cfg_ana.iso2,
                             sel = sel )
        
    def testMuon(self, muon):
        '''Returns True if a muon passes a set of cuts.
        Can be used in testLeg1 and testLeg2, in child classes.'''
        # return True
        return muon.looseId()

    
    def testElectron(self, electron):
        '''Returns True if an electron passes a set of cuts.
        Can be used in testLeg1 and testLeg2, in child classes.'''
        return True


    def testTau(self, tau):
        '''Returns True if a tau passes a set of cuts.
        Can be used in testLeg1 and testLeg2, in child classes.

        WARNING: the muon filter should be used only in the muon channel.'''
        return True


    def testMass(self, diLepton):
        mass = diLepton.mass()
        return self.cfg_ana.m_min < mass and mass < self.cfg_ana.m_max

    
    def bestDiLepton(self, diLeptons):
        '''Returns the best diLepton (the one with highest pt1 + pt2).'''
        return max( diLeptons, key=operator.methodcaller( 'sumPt' ) )
    

    def trigMatched(self, event, leg, legName):
        if not hasattr( self.cfg_ana, 'triggerMap'):
            return True
        path = event.hltPath
        triggerObjects = event.triggerObjects
        filters = self.cfg_ana.triggerMap[ path ]
        filter = None
        if legName == 'leg1':
            filter = filters[0]
        elif legName == 'leg2':
            filter = filters[1]
        else:
            raise ValueError( 'legName should be leg1 or leg2, not {leg}'.format(
                leg=legName )  )
        # the dR2Max value is 0.3^2
        pdgIds = None
        if len(filter) == 2:
            filter, pdgIds = filter[0], filter[1]
        return triggerMatched(leg, triggerObjects, path, filter,
                              dR2Max=0.089999, pdgIds=pdgIds )
