import os
import re

from CMGTools.RootTools.statistics.Counter import Counter 
from CMGTools.RootTools.utils.triggerevo import Menus
from CMGTools.RootTools.utils.TriggerJSON import TriggerJSON
from CMGTools.RootTools.utils.RLTInfo import RLTInfo



class TriggerList( object ):
    '''Holds a list of HLT trigger paths. Can be asked if, for a given triggerObject in a given event, one of the triggers in the list is passed.'''
    def __init__(self, triggerList):
        '''triggerlist is a list of HLT trigger paths.

        Internally, each trigger in the list will be kept as a Counter, which allows to count how many
        events have been tested against, and have passed each trigger.'''

        # self.triggerList = map( Counter, triggerList )
        self.triggerList = []
        for trigName in triggerList:
            # trigName = trigName.replace('*','STAR')
            trig = Counter( trigName )
            trig.register('events tested')
            trig.register('events passed')
            self.triggerList.append( trig )
        fileName = '/'.join( [os.environ['CMSSW_BASE'],
                              'src/CMGTools/RootTools/python/utils/triggerEvolution_all.txt'])
        #datasets = ['TauPlusX']
        # FIXME: This is tau-specific.
        datasets = ['Tau','TauParked','DoubleMu','DoubleMuParked','TauPlusX','SingleMu']
        self.menus = Menus( fileName, datasets )
        self.run = -1
        self.triggerJSON = TriggerJSON()
        self.rltInfo = RLTInfo()

    def restrictList(self, run, triggerList, isData, isEmbed=False):
        '''Restrict the trigger list to the list of unprescaled triggers in this run.

        Seriously speeds up the code.'''
        # import pdb; pdb.set_trace()
        # if run == 1:
        #    return triggerList
        if run != self.run:
            try:
                #import pdb ; pdb.set_trace()
                if isData :
                  selMenus  = self.menus.findUnprescaledPaths(run, 'Tau')
                  try :
                    selMenus  = self.menus.findUnprescaledPaths(run, 'TauPlusX')
                    selMenus2 = self.menus.findUnprescaledPaths(run, 'TauParked')
                    selMenus += selMenus2
                  except :
                    pass
                if isEmbed :
                  selMenus  = self.menus.findUnprescaledPaths(run, 'DoubleMu')
                  try :
                    selMenus2  = self.menus.findUnprescaledPaths(run, 'DoubleMuParked')
                    selMenus += selMenus2
                  except :
                    pass 
                #import pdb ; pdb.set_trace()
                self.unprescaledPaths = set( path.name for path in selMenus )
                self.restrictedTriggerList = [trigger \
                                              for trigger in triggerList \
                                              if trigger.name in self.unprescaledPaths ]
                # print 'restricting list: ', run, [trigger.name for trigger in self.restrictedTriggerList]
            except ValueError:
                print 'no menu with run', run, 'using full trigger list.'
                self.restrictedTriggerList = self.triggerList
            self.run = run
        if len(self.restrictedTriggerList) == 0:
            if len( self.triggerList ) != 0:
                print 'run', run, ': no path from the user list found in the list of unprescaled paths from the trigger DB. The latter could be wrong, using the user trigger list.'
            self.restrictedTriggerList = self.triggerList
        #import pdb ; pdb.set_trace()
        return self.restrictedTriggerList
        
    def triggerPassed(self, triggerObject, run, lumi, 
                      isData, isEmbed=False, usePrescaled=False):
        '''returns true if at least one of the triggers in the triggerlist passes.

        run is provided to call restrictList.
        if usePrescaled is False (DEFAULT), only the unprescaled triggers are considered.
        if triggerList is None (DEFAULT), oneself triggerlist is used. '''
        
        #import pdb ; pdb.set_trace()
        
        triggerList = self.triggerList
        if isData or isEmbed:
            triggerList = self.restrictList( run, self.triggerList, isData, isEmbed ) 
        if len(triggerList)==0:
            # no trigger specified, accepting all events
            return True, None
        passed = False
        firstTrigger = None
        for trigger in triggerList:
            trigger.inc('events tested')
            # if triggerObject.getSelectionRegExp( trigger.name ):
            #import pdb ; pdb.set_trace()
            passedName, prescaleFactor =  self.getSelectionRegExp( triggerObject, trigger.name )
            if passedName is not None:
                # prescaleFactor = triggerObject.getPrescale( passedName )
                if usePrescaled or prescaleFactor == 1 or not isData:
                    # prescales are set to 0 in MC
                    trigger.inc('events passed')
                    passed = True
                    if firstTrigger is None:
                        firstTrigger = trigger.name
                        self.triggerJSON.setdefault(trigger.name, set()).add( run )
                        self.rltInfo.add( trigger.name, run, lumi )

                # don't break, need to test all triggers in the list
                # break
        return passed, firstTrigger


    def getSelectionRegExp( self, object, triggerName ):
        '''returns trigName, prescale where:
        trigName is the name of the trigger with the lowest prescale that was passed.
        if several unprescaled triggers are found, the first one is returned.'''
        #FIXME could cache that
        pattern = re.compile( triggerName )
        maxPrescale = 9999999
        trigWithLowestPrescale = None
        for name in object.getSelectionNames():
            if pattern.match( name ):
                if object.getSelection( name ) is False:
                    return None, -1
                prescale = object.getPrescale( name )
                if prescale == 1:
                    return name, prescale
                elif prescale < maxPrescale:
                    maxPrescale = prescale
                    trigWithLowestPrescale = name
        return trigWithLowestPrescale, maxPrescale

                    
    def write(self, dirName ):
        self.triggerJSON.write( dirName )
        self.rltInfo.write( dirName )
        map( lambda x: x.write(dirName), self.triggerList)


    def computeLumi(self, json):
        self.triggerJSON.computeLumi( json )


    def __str__(self):
        head = 'TriggerList'
        triggers = '\n'.join( map(str, self.triggerList) )
        triggerJSON = str( self.triggerJSON)
        return ':\n'.join( [head, triggers, triggerJSON] )

        
if __name__ == '__main__':
    list = ['HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v[5,6]','HLT_IsoMu15_LooseIsoPFTau15_v9']
    trigList = TriggerList( list )
    print trigList
