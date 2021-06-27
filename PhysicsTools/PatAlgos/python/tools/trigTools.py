from __future__ import print_function
from builtins import range
from PhysicsTools.PatAlgos.tools.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerL1RefsEventContent

from PhysicsTools.PatAlgos.triggerLayer1.triggerMatcherExamples_cfi import _exampleTriggerMatchers
_defaultTriggerProducer      = 'patTrigger'
_defaultTriggerEventProducer = 'patTriggerEvent'
_defaultPath                 = ''
_defaultHltProcess           = 'HLT'
_defaultOutputModule         = 'out'
_defaultPostfix              = ''

_defaultTriggerMatchersComment      = "Trigger matcher modules' labels, default: ..."
_defaultTriggerProducerComment      = "PATTriggerProducer module label, default: %s"%( _defaultTriggerProducer )
_defaultTriggerEventProducerComment = "PATTriggerEventProducer module label, default: %s"%( _defaultTriggerEventProducer )
_defaultPathComment                 = "Name of path to use, default: %s"%( _defaultPath )
_defaultHltProcessComment           = "HLT process name, default: %s"%( _defaultHltProcess )
_defaultOutputModuleComment         = "Output module label, empty label indicates no output, default: %s"%( _defaultOutputModule )
_defaultPostfixComment              = "Postfix to apply to PAT module labels, default: %s"%( _defaultPostfix )

_longLine = '---------------------------------------------------------------------'


def _modulesInPath( process, pathLabel ):
    return [ m.label() for m in listModules( getattr( process, pathLabel ) ) ]


def _addEventContent( outputCommands, eventContent ):
    # add new entry to event content
    for content in eventContent:
        if content not in outputCommands:
            outputCommands += [ content ]
    # check for obsolete entries
    listToRemove = []
    for i in range( len( outputCommands ) ):
        if i in listToRemove:
            continue
        command = outputCommands[ i ]
        if command[ : 4 ] == 'keep':
            dropCommand = command.replace( 'keep ', 'drop ' )
            for j in range( i + 1, len( outputCommands ) ):
                testCommand = outputCommands[ j ]
                if testCommand == command:
                    listToRemove += [ j ]
                elif testCommand == dropCommand:
                    listToRemove += [ i, j ]
                    break
    # copy entries excl. obsolete ones
    newOutputCommands = cms.untracked.vstring()
    for i in range( len( outputCommands ) ):
        if i not in listToRemove:
            newOutputCommands += [ outputCommands[ i ] ]
    # return result
    return newOutputCommands


class SwitchOnTrigger( ConfigToolBase ):
    """  Enables trigger information in PAT
    SwitchOnTrigger( [cms.Process], triggerProducer = 'patTrigger', triggerEventProducer = 'patTriggerEvent', path = '', hltProcess = 'HLT', outputModule = 'out' )
    - [cms.Process]       : the 'cms.Process'
    - triggerProducer     : PATTriggerProducer module label;
                            optional, default: 'patTrigger'
    - triggerEventProducer: PATTriggerEventProducer module label;
                            optional, default: 'patTriggerEvent'
    - path                : name of path to use;
                            optional, default: ''
    - hltProcess          : HLT process name;
                            optional, default: 'HLT'
    - outputModule        : output module label;
                            empty label indicates no output;
                            optional, default: 'out'
    Using None as any argument restores its default value.
    """
    _label             = 'switchOnTrigger'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'triggerProducer'     , _defaultTriggerProducer     , _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'triggerEventProducer', _defaultTriggerEventProducer, _defaultTriggerEventProducerComment )
        self.addParameter( self._defaultParameters, 'path'                , _defaultPath                , _defaultPathComment )
        self.addParameter( self._defaultParameters, 'hltProcess'          , _defaultHltProcess          , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'        , _defaultOutputModule        , _defaultOutputModuleComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerProducer      = None
                , triggerEventProducer = None
                , path                 = None
                , hltProcess           = None
                , outputModule         = None
                ):
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if triggerEventProducer is None:
            triggerEventProducer = self._defaultParameters[ 'triggerEventProducer' ].value
        if path is None:
            path = self._defaultParameters[ 'path' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        self.setParameter( 'triggerProducer'     , triggerProducer )
        self.setParameter( 'triggerEventProducer', triggerEventProducer )
        self.setParameter( 'path'                , path )
        self.setParameter( 'hltProcess'          , hltProcess )
        self.setParameter( 'outputModule'        , outputModule )
        self.apply( process )

    def toolCode( self, process ):
        triggerProducer      = self._parameters[ 'triggerProducer' ].value
        triggerEventProducer = self._parameters[ 'triggerEventProducer' ].value
        path                 = self._parameters[ 'path' ].value
        hltProcess           = self._parameters[ 'hltProcess' ].value
        outputModule         = self._parameters[ 'outputModule' ].value

        task = getPatAlgosToolsTask(process)

        # Load default producers from existing config files, if needed
        if not hasattr( process, triggerProducer ):
            from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger
            addToProcessAndTask(triggerProducer, patTrigger.clone(), process, task)
        else:
            print('%s():'%( self._label ))
            print('    PATTriggerProducer module \'%s\' exists already in process'%( triggerProducer ))
            print('    ==> entry re-used')
            print(_longLine)
        if not hasattr( process, triggerEventProducer ):
            from PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi import patTriggerEvent
            addToProcessAndTask(triggerEventProducer, patTriggerEvent.clone(), process, task)
        else:
            print('%s():'%( self._label ))
            print('    PATTriggerEventProducer module \'%s\' exists already in process'%( triggerEventProducer ))
            print('    ==> entry re-used')
            print(_longLine)

        # Maintain configurations
        trigProdMod             = getattr( process, triggerProducer )
        trigProdMod.processName = hltProcess
        if trigProdMod.onlyStandAlone.value() is True:
            trigProdMod.onlyStandAlone = False
            print('    configuration parameter automatically changed')
            print('    PATTriggerProducer %s.onlyStandAlone --> %s'%( triggerProducer, trigProdMod.onlyStandAlone ))
            print(_longLine)
        trigEvtProdMod                    = getattr( process, triggerEventProducer )
        trigEvtProdMod.processName        = hltProcess
        trigEvtProdMod.patTriggerProducer = cms.InputTag( triggerProducer )
        if path != '':
            if not hasattr( process, path ):
                prodPath = cms.Path( trigProdMod + trigEvtProdMod )
                setattr( process, path, prodPath )
                print('%s():'%( self._label ))
                print('    Path \'%s\' does not exist in process'%( path ))
                print('    ==> created')
                print(_longLine)
            # Try to get the order right, but cannot deal with all possible cases.
            # Simply rely on the exclusive usage of these tools without manual intervention.
            else:
                if not triggerProducer in _modulesInPath( process, path ):
                    prodPath = getattr( process, path )
                    prodPath += trigProdMod
                if not triggerEventProducer in _modulesInPath( process, path ):
                    prodPath = getattr( process, path )
                    prodPath += trigEvtProdMod

        # Add event content
        if outputModule != '':
            patTriggerEventContent = [ 'keep patTriggerObjects_%s_*_%s'%( triggerProducer, process.name_() )
                                     , 'keep patTriggerFilters_%s_*_%s'%( triggerProducer, process.name_() )
                                     , 'keep patTriggerPaths_%s_*_%s'%( triggerProducer, process.name_() )
                                     , 'keep patTriggerEvent_%s_*_%s'%( triggerEventProducer, process.name_() )
                                     ]
            if ( hasattr( trigProdMod, 'addL1Algos' ) and trigProdMod.addL1Algos.value() is True ):
                patTriggerEventContent += [ 'keep patTriggerConditions_%s_*_%s'%( triggerProducer, process.name_() )
                                          , 'keep patTriggerAlgorithms_%s_*_%s'%( triggerProducer, process.name_() )
                                          ]
            if ( hasattr( trigProdMod, 'saveL1Refs' ) and trigProdMod.saveL1Refs.value() is True ):
                patTriggerEventContent += patTriggerL1RefsEventContent
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTrigger = SwitchOnTrigger()


class SwitchOnTriggerStandAlone( ConfigToolBase ):
    """  Enables trigger information in PAT, limited to stand-alone trigger objects
    SwitchOnTriggerStandAlone( [cms.Process], triggerProducer = 'patTrigger', path = '', hltProcess = 'HLT', outputModule = 'out' )
    - [cms.Process]       : the 'cms.Process'
    - triggerProducer     : PATTriggerProducer module label;
                            optional, default: 'patTrigger'
    - path                : name of path to use;
                            optional, default: ''
    - hltProcess          : HLT process name;
                            optional, default: 'HLT'
    - outputModule        : output module label;
                            empty label indicates no output;
                            optional, default: 'out'
    Using None as any argument restores its default value.
    """
    _label             = 'switchOnTriggerStandAlone'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'triggerProducer', _defaultTriggerProducer, _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'path'           , _defaultPath           , _defaultPathComment )
        self.addParameter( self._defaultParameters, 'hltProcess'     , _defaultHltProcess     , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'   , _defaultOutputModule   , _defaultOutputModuleComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerProducer      = None
                , path                 = None
                , hltProcess           = None
                , outputModule         = None
                ):
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if path is None:
            path = self._defaultParameters[ 'path' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        self.setParameter( 'triggerProducer', triggerProducer )
        self.setParameter( 'path'           , path )
        self.setParameter( 'hltProcess'     , hltProcess )
        self.setParameter( 'outputModule'   , outputModule )
        self.apply( process )

    def toolCode( self, process ):

        task = getPatAlgosToolsTask(process)

        triggerProducer = self._parameters[ 'triggerProducer' ].value
        path            = self._parameters[ 'path' ].value
        hltProcess      = self._parameters[ 'hltProcess' ].value
        outputModule    = self._parameters[ 'outputModule' ].value

        # Load default producer from existing config file, if needed
        if not hasattr( process, triggerProducer ):
            from PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi import patTrigger
            addToProcessAndTask(triggerProducer, patTrigger.clone( onlyStandAlone = True ), process, task)
        else:
            print('%s():'%( self._label ))
            print('    PATTriggerProducer module \'%s\' exists already in process'%( triggerProducer ))
            print('    ==> entry re-used')
            print(_longLine)

        # Maintain configuration
        trigProdMod             = getattr( process, triggerProducer )
        trigProdMod.processName = hltProcess
        if path != '':
            if not hasattr( process, path ):
                prodPath = cms.Path( trigProdMod )
                setattr( process, path, prodPath )
                print('%s():'%( self._label ))
                print('    Path \'%s\' does not exist in process'%( path ))
                print('    ==> created')
                print(_longLine)
            elif not triggerProducer in _modulesInPath( process, path ):
                prodPath = getattr( process, path )
                prodPath += trigProdMod

        # Add event content
        if outputModule != '':
            patTriggerEventContent = [ 'keep patTriggerObjectStandAlones_%s_*_%s'%( triggerProducer, process.name_() )
                                     ]
            if ( hasattr( trigProdMod, 'saveL1Refs' ) and trigProdMod.saveL1Refs.value() is True ):
                patTriggerEventContent += patTriggerL1RefsEventContent
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerStandAlone = SwitchOnTriggerStandAlone()


class SwitchOnTriggerMatching( ConfigToolBase ):
    """  Enables trigger matching in PAT
    SwitchOnTriggerMatching( [cms.Process], triggerMatchers = [default list], triggerProducer = 'patTrigger', triggerEventProducer = 'patTriggerEvent', path = '', hltProcess = 'HLT', outputModule = 'out', postfix = '' )
    - [cms.Process]       : the 'cms.Process'
    - triggerMatchers     : PAT trigger matcher module labels (list)
                            optional; default: defined in '_exampleTriggerMatchers'
                            (s. PhysicsTools/PatAlgos/python/triggerLayer1/triggerMatcherExamples_cfi.py)
    - triggerProducer     : PATTriggerProducer module label;
                            optional, default: 'patTrigger'
    - triggerEventProducer: PATTriggerEventProducer module label;
                            optional, default: 'patTriggerEvent'
    - path                : name of path to use;
                            optional, default: 'patDefaultSequence'
    - hltProcess          : HLT process name;
                            optional, default: 'HLT'
    - outputModule        : output module label;
                            empty label indicates no output;
                            optional, default: 'out'
    - postfix             : postfix to apply to PAT module labels;
                            optional, default: ''
    Using None as any argument restores its default value.
    """
    _label             = 'switchOnTriggerMatching'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'triggerMatchers'     , _exampleTriggerMatchers     , _defaultTriggerMatchersComment )
        self.addParameter( self._defaultParameters, 'exampleMatchers'     , False                       , '' )
        self.addParameter( self._defaultParameters, 'triggerProducer'     , _defaultTriggerProducer     , _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'triggerEventProducer', _defaultTriggerEventProducer, _defaultTriggerEventProducerComment )
        self.addParameter( self._defaultParameters, 'path'                , _defaultPath                , _defaultPathComment )
        self.addParameter( self._defaultParameters, 'hltProcess'          , _defaultHltProcess          , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'        , _defaultOutputModule        , _defaultOutputModuleComment )
        self.addParameter( self._defaultParameters, 'postfix'             , _defaultPostfix             , _defaultPostfixComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerMatchers      = None
                , triggerProducer      = None
                , triggerEventProducer = None
                , path                 = None
                , hltProcess           = None
                , outputModule         = None
                , postfix              = None
                ):
        if triggerMatchers is None:
            triggerMatchers = self._defaultParameters[ 'triggerMatchers' ].value
            self.setParameter( 'exampleMatchers', True )
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if triggerEventProducer is None:
            triggerEventProducer = self._defaultParameters[ 'triggerEventProducer' ].value
        if path is None:
            path = self._defaultParameters[ 'path' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        if postfix is None:
            postfix = self._defaultParameters[ 'postfix' ].value
        self.setParameter( 'triggerMatchers'     , triggerMatchers )
        self.setParameter( 'triggerProducer'     , triggerProducer )
        self.setParameter( 'triggerEventProducer', triggerEventProducer )
        self.setParameter( 'path'                , path )
        self.setParameter( 'hltProcess'          , hltProcess )
        self.setParameter( 'outputModule'        , outputModule )
        self.setParameter( 'postfix'             , postfix )
        self.apply( process )

    def toolCode( self, process ):
        triggerMatchers      = self._parameters[ 'triggerMatchers' ].value
        exampleMatchers      = self._parameters[ 'exampleMatchers' ].value
        triggerProducer      = self._parameters[ 'triggerProducer' ].value
        triggerEventProducer = self._parameters[ 'triggerEventProducer' ].value
        path                 = self._parameters[ 'path' ].value
        hltProcess           = self._parameters[ 'hltProcess' ].value
        outputModule         = self._parameters[ 'outputModule' ].value
        postfix              = self._parameters[ 'postfix' ].value

        # Load default producers from existing config file, if needed
        task = getPatAlgosToolsTask(process)
        if exampleMatchers:
            process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcherExamples_cfi" )
            task.add(process.triggerMatcherExamplesTask)
        # Switch on PAT trigger information if needed
        if not hasattr( process, triggerEventProducer ):
            print('%s():'%( self._label ))
            print('    PAT trigger production switched on automatically using')
            print('    switchOnTrigger( process, \'%s\', \'%s\', \'%s\', \'%s\', \'%s\' )'%( hltProcess, triggerProducer, triggerEventProducer, path, outputModule ))
            print(_longLine)
            switchOnTrigger( process, triggerProducer, triggerEventProducer, path, hltProcess, outputModule )

        # Maintain configurations
        trigEvtProdMod = getattr( process, triggerEventProducer )
        triggerMatchersKnown = []
        for matcher in triggerMatchers:
            if not hasattr( process, matcher ):
                print('%s():'%( self._label ))
                print('    Matcher \'%s\' not known to process'%( matcher ))
                print('    ==> skipped')
                print(_longLine)
                continue
            triggerMatchersKnown.append( matcher )
            trigMchMod         = getattr( process, matcher )
            trigMchMod.src     = cms.InputTag( trigMchMod.src.getModuleLabel() + postfix )
            trigMchMod.matched = triggerProducer
        matchers = getattr( trigEvtProdMod, 'patTriggerMatches' )
        if len( matchers ) > 0:
            print('%s():'%( self._label ))
            print('    PAT trigger matchers already attached to existing PATTriggerEventProducer \'%s\''%( triggerEventProducer ))
            print('    configuration parameters automatically changed')
            for matcher in matchers:
                trigMchMod = getattr( process, matcher )
                if trigMchMod.matched.value() is not triggerProducer:
                    trigMchMod.matched = triggerProducer
                    print('    PAT trigger matcher %s.matched --> %s'%( matcher, trigMchMod.matched ))
            print(_longLine)
        else:
            trigEvtProdMod.patTriggerMatches = cms.VInputTag()
        for matcher in triggerMatchersKnown:
            trigEvtProdMod.patTriggerMatches.append( cms.InputTag( matcher ) )

        # Add event content
        if outputModule != '':
            patTriggerEventContent = []
            for matcher in triggerMatchersKnown:
                patTriggerEventContent += [ 'keep patTriggerObjectsedmAssociation_%s_%s_%s'%( triggerEventProducer, matcher, process.name_() )
                                          , 'keep *_%s_*_*'%( getattr( process, matcher ).src.value() )
                                          ]
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerMatching = SwitchOnTriggerMatching()


class SwitchOnTriggerMatchingStandAlone( ConfigToolBase ):
    """  Enables trigger matching in PAT
    SwitchOnTriggerMatchingStandAlone( [cms.Process], triggerMatchers = [default list], triggerProducer = 'patTrigger', path = '', hltProcess = 'HLT', outputModule = 'out', postfix = '' )
    - [cms.Process]  : the 'cms.Process'
    - triggerMatchers: PAT trigger matcher module labels (list)
                       optional; default: defined in 'triggerMatchingDefaultSequence'
                       (s. PhysicsTools/PatAlgos/python/triggerLayer1/triggerMatcherExamples_cfi.py)
    - triggerProducer: PATTriggerProducer module label;
                       optional, default: 'patTrigger'
    - path           : name of path to use;
                       optional, default: ''
    - hltProcess     : HLT process name;
                       optional, default: 'HLT'
    - outputModule   : output module label;
                       empty label indicates no output;
                       optional, default: 'out'
    - postfix        : postfix to apply to PAT module labels;
                       optional, default: ''
    Using None as any argument restores its default value.
    """
    _label             = 'switchOnTriggerMatchingStandAlone'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'triggerMatchers', _exampleTriggerMatchers, _defaultTriggerMatchersComment )
        self.addParameter( self._defaultParameters, 'exampleMatchers', False                  , '' )
        self.addParameter( self._defaultParameters, 'triggerProducer', _defaultTriggerProducer, _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'path'           , _defaultPath           , _defaultPathComment )
        self.addParameter( self._defaultParameters, 'hltProcess'     , _defaultHltProcess     , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'   , _defaultOutputModule   , _defaultOutputModuleComment )
        self.addParameter( self._defaultParameters, 'postfix'        , _defaultPostfix        , _defaultPostfixComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerMatchers = None
                , triggerProducer = None
                , path            = None
                , hltProcess      = None
                , outputModule    = None
                , postfix         = None
                ):
        if triggerMatchers is None:
            triggerMatchers = self._defaultParameters[ 'triggerMatchers' ].value
            self.setParameter( 'exampleMatchers', True )
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if path is None:
            path = self._defaultParameters[ 'path' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        if postfix is None:
            postfix = self._defaultParameters[ 'postfix' ].value
        self.setParameter( 'triggerMatchers', triggerMatchers )
        self.setParameter( 'triggerProducer', triggerProducer )
        self.setParameter( 'path'           , path )
        self.setParameter( 'hltProcess'     , hltProcess )
        self.setParameter( 'outputModule'   , outputModule )
        self.setParameter( 'postfix'        , postfix )
        self.apply( process )

    def toolCode( self, process ):
        triggerMatchers = self._parameters[ 'triggerMatchers' ].value
        exampleMatchers = self._parameters[ 'exampleMatchers' ].value
        triggerProducer = self._parameters[ 'triggerProducer' ].value
        path            = self._parameters[ 'path' ].value
        hltProcess      = self._parameters[ 'hltProcess' ].value
        outputModule    = self._parameters[ 'outputModule' ].value
        postfix         = self._parameters[ 'postfix' ].value

        # Load default producers from existing config file, if needed
        task = getPatAlgosToolsTask(process)
        if exampleMatchers:
            process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcherExamples_cfi" )
            task.add(process.triggerMatcherExamplesTask)

        # Switch on PAT trigger information if needed
        if not hasattr( process, triggerProducer ):
            print('%s():'%( self._label ))
            print('    PAT trigger production switched on automatically using')
            print('    switchOnTriggerStandAlone( process, \'%s\', \'%s\', \'%s\', \'%s\' )'%( hltProcess, triggerProducer, path, outputModule ))
            print(_longLine)
            switchOnTriggerStandAlone( process, triggerProducer, path, hltProcess, outputModule )

        # Maintain configurations
        triggerMatchersKnown = []
        for matcher in triggerMatchers:
            if not hasattr( process, matcher ):
                print('%s():'%( self._label ))
                print('    Matcher \'%s\' not known to process'%( matcher ))
                print('    ==> skipped')
                print(_longLine)
                continue
            triggerMatchersKnown.append( matcher )
            trigMchMod         = getattr( process, matcher )
            trigMchMod.src     = cms.InputTag( trigMchMod.src.getModuleLabel() + postfix )
            trigMchMod.matched = triggerProducer

        # Add event content
        if outputModule != '':
            patTriggerEventContent = []
            for matcher in triggerMatchersKnown:
                patTriggerEventContent += [ 'keep patTriggerObjectStandAlonesedmAssociation_%s_*_%s'%( matcher, process.name_() )
                                          , 'keep *_%s_*_*'%( getattr( process, matcher ).src.value() )
                                          ]
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerMatchingStandAlone = SwitchOnTriggerMatchingStandAlone()


class SwitchOnTriggerMatchEmbedding( ConfigToolBase ):
    """  Enables embedding of trigger matches into PAT objects
    SwitchOnTriggerMatchEmbedding( [cms.Process], triggerMatchers = [default list], triggerProducer = 'patTrigger', path = '', hltProcess = 'HLT', outputModule = 'out', postfix = '' )
    - [cms.Process]  : the 'cms.Process'
    - triggerMatchers: PAT trigger matcher module labels (list)
                       optional; default: defined in 'triggerMatchingDefaultSequence'
                       (s. PhysicsTools/PatAlgos/python/triggerLayer1/triggerMatcherExamples_cfi.py)
    - triggerProducer: PATTriggerProducer module label;
                       optional, default: 'patTrigger'
    - path           : name of path to use;
                       optional, default: ''
    - hltProcess     : HLT process name;
                       optional, default: 'HLT'
    - outputModule   : output module label;
                       empty label indicates no output;
                       optional, default: 'out'
    - postfix        : postfix to apply to PAT module labels;
                       optional, default: ''
    Using None as any argument restores its default value.
    """
    _label             = 'switchOnTriggerMatchEmbedding'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'triggerMatchers', _exampleTriggerMatchers, _defaultTriggerMatchersComment )
        self.addParameter( self._defaultParameters, 'exampleMatchers', False                  , '' )
        self.addParameter( self._defaultParameters, 'triggerProducer', _defaultTriggerProducer, _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'path'           , _defaultPath           , _defaultPathComment )
        self.addParameter( self._defaultParameters, 'hltProcess'     , _defaultHltProcess     , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'   , _defaultOutputModule   , _defaultOutputModuleComment )
        self.addParameter( self._defaultParameters, 'postfix'        , _defaultPostfix        , _defaultPostfixComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerMatchers = None
                , triggerProducer = None
                , path            = None
                , hltProcess      = None
                , outputModule    = None
                , postfix         = None
                ):
        if triggerMatchers is None:
            triggerMatchers = self._defaultParameters[ 'triggerMatchers' ].value
            self.setParameter( 'exampleMatchers', True )
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if path is None:
            path = self._defaultParameters[ 'path' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        if postfix is None:
            postfix = self._defaultParameters[ 'postfix' ].value
        self.setParameter( 'triggerMatchers', triggerMatchers )
        self.setParameter( 'triggerProducer', triggerProducer )
        self.setParameter( 'path'           , path )
        self.setParameter( 'hltProcess'     , hltProcess )
        self.setParameter( 'outputModule'   , outputModule )
        self.setParameter( 'postfix'        , postfix )
        self.apply( process )

    def toolCode( self, process ):
        triggerMatchers = self._parameters[ 'triggerMatchers' ].value
        exampleMatchers = self._parameters[ 'exampleMatchers' ].value
        triggerProducer = self._parameters[ 'triggerProducer' ].value
        path            = self._parameters[ 'path' ].value
        hltProcess      = self._parameters[ 'hltProcess' ].value
        outputModule    = self._parameters[ 'outputModule' ].value
        postfix         = self._parameters[ 'postfix' ].value

        # Load default producers from existing config file, if needed
        task = getPatAlgosToolsTask(process)
        if exampleMatchers:
            process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcherExamples_cfi" )
            task.add(process.triggerMatcherExamplesTask)

        # Build dictionary of allowed input producers
        dictPatProducers = { 'PATPhotonCleaner'  : 'PATTriggerMatchPhotonEmbedder'
                           , 'PATElectronCleaner': 'PATTriggerMatchElectronEmbedder'
                           , 'PATMuonCleaner'    : 'PATTriggerMatchMuonEmbedder'
                           , 'PATTauCleaner'     : 'PATTriggerMatchTauEmbedder'
                           , 'PATJetCleaner'     : 'PATTriggerMatchJetEmbedder'
                           , 'PATMETCleaner'     : 'PATTriggerMatchMETEmbedder'
#                            , 'PATGenericParticleCleaner'     : ''
#                            , 'PATPFParticleCleaner'     : ''
                           , 'PATPhotonSelector'  : 'PATTriggerMatchPhotonEmbedder'
                           , 'PATElectronSelector': 'PATTriggerMatchElectronEmbedder'
                           , 'PATMuonSelector'    : 'PATTriggerMatchMuonEmbedder'
                           , 'PATTauSelector'     : 'PATTriggerMatchTauEmbedder'
                           , 'PATJetSelector'     : 'PATTriggerMatchJetEmbedder'
                           , 'PATMETSelector'     : 'PATTriggerMatchMETEmbedder'
#                            , 'PATGenericParticleSelector'     : ''
#                            , 'PATPFParticleSelector'     : ''
#                            , 'PATCompositeCandidateSelector'     : ''
                           , 'PATPhotonRefSelector'  : 'PATTriggerMatchPhotonEmbedder'
                           , 'PATElectronRefSelector': 'PATTriggerMatchElectronEmbedder'
                           , 'PATMuonRefSelector'    : 'PATTriggerMatchMuonEmbedder'
                           , 'PATTauRefSelector'     : 'PATTriggerMatchTauEmbedder'
                           , 'PATJetRefSelector'     : 'PATTriggerMatchJetEmbedder'
                           , 'PATMETRefSelector'     : 'PATTriggerMatchMETEmbedder'
#                            , 'PATGenericParticleRefSelector'     : ''
#                            , 'PATPFParticleRefSelector'     : ''
#                            , 'PATCompositeCandidateRefSelector'     : ''
                           , 'PATPhotonProducer'  : 'PATTriggerMatchPhotonEmbedder'
                           , 'PATElectronProducer': 'PATTriggerMatchElectronEmbedder'
                           , 'PATMuonProducer'    : 'PATTriggerMatchMuonEmbedder'
                           , 'PATTauProducer'     : 'PATTriggerMatchTauEmbedder'
                           , 'PATJetProducer'     : 'PATTriggerMatchJetEmbedder'
                           , 'PATMETProducer'     : 'PATTriggerMatchMETEmbedder'
#                            , 'PATGenericParticleProducer'     : ''
#                            , 'PATPFParticleProducer'     : ''
#                            , 'PATCompositeCandidateProducer'     : ''
                           , 'MuonSelectorVertex': 'PATTriggerMatchMuonEmbedder'
                           }

        # Switch on PAT trigger matching if needed
        dictConfig = {}
        if not hasattr( process, triggerProducer ):
            if exampleMatchers:
                print('%s():'%( self._label ))
                print('    PAT trigger matching switched on automatically using')
                print('    switchOnTriggerMatchingStandAlone( process, \'%s\', None, \'%s\', \'%s\', \'%s\', \'%s\' )'%( hltProcess, triggerProducer, path, outputModule, postfix ))
                print(_longLine)
                switchOnTriggerMatchingStandAlone( process, None, triggerProducer, path, hltProcess, '', postfix ) # Do not store intermediate output collections.
            else:
                print('%s():'%( self._label ))
                print('    PAT trigger matching switched on automatically using')
                print('    switchOnTriggerMatchingStandAlone( process, \'%s\', %s, \'%s\', \'%s\', \'%s\', \'%s\' )'%( hltProcess, triggerMatchers, triggerProducer, path, outputModule, postfix ))
                print(_longLine)
                switchOnTriggerMatchingStandAlone( process, triggerMatchers, triggerProducer, path, hltProcess, '', postfix ) # Do not store intermediate output collections.
        elif exampleMatchers:
            process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcherExamples_cfi" )
            task.add(process.triggerMatcherExamplesTask)

        # Build dictionary of matchers
        for matcher in triggerMatchers:
            if not hasattr( process, matcher ):
                print('%s():'%( self._label ))
                print('    PAT trigger matcher \'%s\' not known to process'%( matcher ))
                print('    ==> skipped')
                print(_longLine)
                continue
            trigMchMod = getattr( process, matcher )
            patObjProd = getattr( process, trigMchMod.src.value() + postfix )
            if trigMchMod.src.value() in dictConfig:
                dictConfig[ patObjProd.type_() ] += [ matcher ]
            else:
                dictConfig[ patObjProd.type_() ] = [ matcher ]

        # Maintain configurations
        patTriggerEventContent = []
        for patObjProdType in dictConfig.keys():
            if patObjProdType in dictPatProducers:
                for matcher in dictConfig[ patObjProdType ]:
                    trigMchMod = getattr( process, matcher )
                    patObjProd = getattr( process, trigMchMod.src.value() + postfix )
                    # Configure embedder module
                    label = patObjProd.label_() + 'TriggerMatch' # hardcoded default
                    if hasattr( process, label ):
                        print('%s():'%( self._label ))
                        print('    PAT trigger match embedder \'%s\' exists already in process'%( label ))
                        print('    ==> entry re-used')
                        print(_longLine)
                        module = getattr( process, label )
                        if not module.type_() is dictPatProducers[ patObjProdType ]:
                            print('%s():'%( self._label ))
                            print('    Configuration conflict for PAT trigger match embedder \'%s\''%( label ))
                            print('    - exists as %s'%( module.type_() ))
                            print('    - requested as %s by \'%s\''%( dictPatProducers[ patObjProdType ], matcher ))
                            print('    ==> skipped')
                            print(_longLine)
                            continue
                        if not module.src.value() is trigMchMod.src.value() + postfix:
                            print('%s():'%( self._label ))
                            print('    Configuration conflict for PAT trigger match embedder \'%s\''%( label ))
                            print('    - exists for input %s'%( module.src.value() ))
                            print('    - requested for input %s by \'%s\''%( trigMchMod.src.value() + postfix, matcher ))
                            print('    ==> skipped')
                            print(_longLine)
                            continue
                        module.matches.append( cms.InputTag( matcher ) )
                    else:
                        module         = cms.EDProducer( dictPatProducers[ patObjProdType ] )
                        module.src     = cms.InputTag( patObjProd.label_() )
                        module.matches = cms.VInputTag( matcher )
                        addToProcessAndTask(label, module, process, task)
                    # Add event content
                    patTriggerEventContent += [ 'drop *_%s_*_*'%( patObjProd.label_() )
                                              , 'keep *_%s_*_%s'%( label, process.name_() )
                                              ]
            else:
                print('%s():'%( self._label ))
                print('    Invalid input source for trigger match embedding')
                print('    ==> %s with matchers \'%s\' is skipped'%( patObjProdType, dictConfig[ patObjProdType ] ))
                print(_longLine)
        if outputModule != '':
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerMatchEmbedding = SwitchOnTriggerMatchEmbedding()
