from FWCore.GuiBrowsers.ConfigToolBase import *

from PhysicsTools.PatAlgos.tools.helpers import *
from PhysicsTools.PatAlgos.patEventContent_cff import patTriggerL1RefsEventContent

_defaultTriggerMatchers      = [ 'cleanMuonTriggerMatchHLTMu20'
                               , 'cleanMuonTriggerMatchHLTDoubleMu6'
                               , 'cleanPhotonTriggerMatchHLTPhoton26IsoVLPhoton18'
                               , 'cleanElectronTriggerMatchHLTEle27CaloIdVTCaloIsoTTrkIdTTrkIsoT'
                               , 'cleanTauTriggerMatchHLTDoubleIsoPFTau20Trk5'
                               , 'cleanJetTriggerMatchHLTJet240'
                               , 'metTriggerMatchHLTMET100'
                               , 'cleanMuonTriggerMatchHLTMu17CentralJet30'
                               , 'cleanJetTriggerMatchHLTMu17CentralJet30'
                               ]
_defaultTriggerProducer      = 'patTrigger'
_defaultTriggerEventProducer = 'patTriggerEvent'
_defaultSequence             = 'patDefaultSequence'
_defaultHltProcess           = 'HLT'
_defaultOutputModule         = 'out'
_defaultPostfix              = ''

_defaultTriggerMatchersComment      = "Trigger matcher modules' labels, default: ..."
_defaultTriggerProducerComment      = "PATTriggerProducer module label, default: %s"%( _defaultTriggerProducer )
_defaultTriggerEventProducerComment = "PATTriggerEventProducer module label, default: %s"%( _defaultTriggerEventProducer )
_defaultSequenceComment             = "Name of sequence to use, default: %s"%( _defaultSequence )
_defaultHltProcessComment           = "HLT process name, default: %s"%( _defaultHltProcess )
_defaultOutputModuleComment         = "Output module label, empty label indicates no output, default: %s"%( _defaultOutputModule )
_defaultPostfixComment              = "Postfix to apply to PAT module labels, default: %s"%( _defaultPostfix )

_longLine = '---------------------------------------------------------------------'


def _modulesInSequence( process, sequenceLabel ):
    return [ m.label() for m in listModules( getattr( process, sequenceLabel ) ) ]


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
    SwitchOnTrigger( [cms.Process], triggerProducer = 'patTrigger', triggerEventProducer = 'patTriggerEvent', sequence = 'patDefaultSequence', hltProcess = 'HLT', outputModule = 'out' )
    - [cms.Process]       : the 'cms.Process'
    - triggerProducer     : PATTriggerProducer module label;
                            optional, default: 'patTrigger'
    - triggerEventProducer: PATTriggerEventProducer module label;
                            optional, default: 'patTriggerEvent'
    - sequence            : name of sequence to use;
                            optional, default: 'patDefaultSequence'
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
        self.addParameter( self._defaultParameters, 'sequence'            , _defaultSequence            , _defaultSequenceComment )
        self.addParameter( self._defaultParameters, 'hltProcess'          , _defaultHltProcess          , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'        , _defaultOutputModule        , _defaultOutputModuleComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerProducer      = None
                , triggerEventProducer = None
                , sequence             = None
                , hltProcess           = None
                , outputModule         = None
                ):
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if triggerEventProducer is None:
            triggerEventProducer = self._defaultParameters[ 'triggerEventProducer' ].value
        if sequence is None:
            sequence = self._defaultParameters[ 'sequence' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        self.setParameter( 'triggerProducer'     , triggerProducer )
        self.setParameter( 'triggerEventProducer', triggerEventProducer )
        self.setParameter( 'sequence'            , sequence )
        self.setParameter( 'hltProcess'          , hltProcess )
        self.setParameter( 'outputModule'        , outputModule )
        self.apply( process )

    def toolCode( self, process ):
        triggerProducer      = self._parameters[ 'triggerProducer' ].value
        triggerEventProducer = self._parameters[ 'triggerEventProducer' ].value
        sequence             = self._parameters[ 'sequence' ].value
        hltProcess           = self._parameters[ 'hltProcess' ].value
        outputModule         = self._parameters[ 'outputModule' ].value

        # Load default producers from existing config files, if needed
        if not hasattr( process, triggerProducer ):
            if triggerProducer is self.getDefaultParameters()[ 'triggerProducer' ].value:
                process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi" )
        if not hasattr( process, triggerEventProducer ):
            if triggerEventProducer is self.getDefaultParameters()[ 'triggerEventProducer' ].value:
                process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerEventProducer_cfi" )

        # Maintain configurations
        prodSequence            = getattr( process, sequence )
        trigProdMod             = getattr( process, triggerProducer )
        trigProdMod.processName = hltProcess
        if triggerProducer in _modulesInSequence( process, sequence ):
            print '%s():'%( self._label )
            print '    PATTriggerProducer module %s exists already in sequence %s'%( triggerProducer, sequence )
            print '    ==> entry re-used'
            if trigProdMod.onlyStandAlone.value() is True:
                trigProdMod.onlyStandAlone = False
                print '    configuration parameter automatically changed'
                print '    PATTriggerProducer %s.onlyStandAlone --> %s'%( triggerProducer, trigProdMod.onlyStandAlone )
            print _longLine
        else:
            # Sequence arithmetics for PATTriggerProducer module
            if hasattr( process, sequence + 'Trigger' ):
                index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                getattr( process, sequence + 'Trigger' ).insert( index, trigProdMod )
            else:
                patTriggerSequence = cms.Sequence( trigProdMod )
                setattr( process, sequence + 'Trigger', patTriggerSequence )
                prodSequence *= getattr( process, sequence + 'Trigger' )
        trigEvtProdMod             = getattr( process, triggerEventProducer )
        trigEvtProdMod.processName = hltProcess
        if hasattr( trigEvtProdMod, 'patTriggerProducer' ):
            trigEvtProdMod.patTriggerProducer = triggerProducer
        else:
            trigEvtProdMod.patTriggerProducer = cms.InputTag( triggerProducer )
        if triggerEventProducer in _modulesInSequence( process, sequence ):
            print '%s():'%( self._label )
            print '    PATTriggerEventProducer module %s exists already in sequence %s'%( triggerEventProducer, sequence )
            print '    ==> entry re-used'
            print _longLine
        else:
            # Sequence arithmetics for PATTriggerEventProducer module
            if hasattr( process, sequence + 'TriggerEvent' ):
                index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                getattr( process, sequence + 'TriggerEvent' ).insert( index, trigEvtProdMod )
            else:
                patTriggerEventSequence = cms.Sequence( trigEvtProdMod )
                setattr( process, sequence + 'TriggerEvent', patTriggerEventSequence )
                prodSequence *= getattr( process, sequence + 'TriggerEvent' )

        # Add event content
        if outputModule is not '':
            patTriggerEventContent = [ 'keep patTriggerObjects_%s_*_%s'%( triggerProducer, process.name_() )
                                     , 'keep patTriggerFilters_%s_*_%s'%( triggerProducer, process.name_() )
                                     , 'keep patTriggerPaths_%s_*_%s'%( triggerProducer, process.name_() )
                                     , 'keep patTriggerEvent_%s_*_%s'%( triggerEventProducer, process.name_() )
                                     ]
            if hasattr( trigProdMod, 'addL1Algos' ) and trigProdMod.addL1Algos.value() is True:
                patTriggerEventContent += [ 'keep patTriggerConditions_%s_*_%s'%( triggerProducer, process.name_() )
                                          , 'keep patTriggerAlgorithms_%s_*_%s'%( triggerProducer, process.name_() )
                                          ]
            if hasattr( trigProdMod, 'saveL1Refs' ) and trigProdMod.saveL1Refs.value() is True:
                patTriggerEventContent += patTriggerL1RefsEventContent
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTrigger = SwitchOnTrigger()


class SwitchOnTriggerStandAlone( ConfigToolBase ):
    """  Enables trigger information in PAT, limited to stand-alone trigger objects
    SwitchOnTriggerStandAlone( [cms.Process], triggerProducer = 'patTrigger', sequence = 'patDefaultSequence', hltProcess = 'HLT', outputModule = 'out' )
    - [cms.Process]       : the 'cms.Process'
    - triggerProducer     : PATTriggerProducer module label;
                            optional, default: 'patTrigger'
    - sequence            : name of sequence to use;
                            optional, default: 'patDefaultSequence'
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
        self.addParameter( self._defaultParameters, 'sequence'       , _defaultSequence       , _defaultSequenceComment )
        self.addParameter( self._defaultParameters, 'hltProcess'     , _defaultHltProcess     , _defaultHltProcessComment )
        self.addParameter( self._defaultParameters, 'outputModule'   , _defaultOutputModule   , _defaultOutputModuleComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , triggerProducer      = None
                , sequence             = None
                , hltProcess           = None
                , outputModule         = None
                ):
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if sequence is None:
            sequence = self._defaultParameters[ 'sequence' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        self.setParameter( 'triggerProducer', triggerProducer )
        self.setParameter( 'sequence'       , sequence )
        self.setParameter( 'hltProcess'     , hltProcess )
        self.setParameter( 'outputModule'   , outputModule )
        self.apply( process )

    def toolCode( self, process ):
        triggerProducer = self._parameters[ 'triggerProducer' ].value
        sequence        = self._parameters[ 'sequence' ].value
        hltProcess      = self._parameters[ 'hltProcess' ].value
        outputModule    = self._parameters[ 'outputModule' ].value

        # Load default producer from existing config file, if needed
        if not hasattr( process, triggerProducer ):
            if triggerProducer is self.getDefaultParameters()[ 'triggerProducer' ].value:
                process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi" )

        # Maintain configuration
        prodSequence            = getattr( process, sequence )
        trigProdMod             = getattr( process, triggerProducer )
        trigProdMod.processName = hltProcess
        if triggerProducer in _modulesInSequence( process, sequence ):
            print '%s():'%( self._label )
            print '    PATTriggerProducer module %s exists already in sequence %s'%( triggerProducer, sequence )
            print '    ==> entry re-used'
            print _longLine
        else:
            # Sequence arithmetics for PATTriggerProducer module
            if trigProdMod.onlyStandAlone.value() is False:
                trigProdMod.onlyStandAlone = True
                print '%s():'%( self._label )
                print '    configuration parameter automatically changed'
                print '    PATTriggerProducer %s.onlyStandAlone --> %s'%( triggerProducer, trigProdMod.onlyStandAlone )
                print _longLine
            if hasattr( process, sequence + 'Trigger' ):
                index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                getattr( process, sequence + 'Trigger' ).insert( index, trigProdMod )
            else:
                patTriggerSequence = cms.Sequence( trigProdMod )
                setattr( process, sequence + 'Trigger', patTriggerSequence )
                prodSequence *= getattr( process, sequence + 'Trigger' )

        # Add event content
        if outputModule is not '':
            patTriggerEventContent = [ 'keep patTriggerObjectStandAlones_%s_*_%s'%( triggerProducer, process.name_() )
                                     ]
            if hasattr( trigProdMod, 'saveL1Refs' ) and trigProdMod.saveL1Refs.value() is True:
                patTriggerEventContent += patTriggerL1RefsEventContent
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerStandAlone = SwitchOnTriggerStandAlone()


class SwitchOnTriggerMatching( ConfigToolBase ):
    """  Enables trigger matching in PAT
    SwitchOnTriggerMatching( [cms.Process], triggerMatchers = [default list], triggerProducer = 'patTrigger', triggerEventProducer = 'patTriggerEvent', sequence = 'patDefaultSequence', hltProcess = 'HLT', outputModule = 'out', postfix = '' )
    - [cms.Process]       : the 'cms.Process'
    - triggerMatchers     : PAT trigger matcher module labels (list)
                            optional; default: defined in 'triggerMatchingDefaultSequence'
                            (s. PhysicsTools/PatAlgos/python/triggerLayer1/triggerMatcher_cfi.py)
    - triggerProducer     : PATTriggerProducer module label;
                            optional, default: 'patTrigger'
    - triggerEventProducer: PATTriggerEventProducer module label;
                            optional, default: 'patTriggerEvent'
    - sequence            : name of sequence to use;
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
        self.addParameter( self._defaultParameters, 'triggerMatchers'     , _defaultTriggerMatchers     , _defaultTriggerMatchersComment )
        self.addParameter( self._defaultParameters, 'triggerProducer'     , _defaultTriggerProducer     , _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'triggerEventProducer', _defaultTriggerEventProducer, _defaultTriggerEventProducerComment )
        self.addParameter( self._defaultParameters, 'sequence'            , _defaultSequence            , _defaultSequenceComment )
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
                , sequence             = None
                , hltProcess           = None
                , outputModule         = None
                , postfix              = None
                ):
        if triggerMatchers is None:
            triggerMatchers = self._defaultParameters[ 'triggerMatchers' ].value
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if triggerEventProducer is None:
            triggerEventProducer = self._defaultParameters[ 'triggerEventProducer' ].value
        if sequence is None:
            sequence = self._defaultParameters[ 'sequence' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        if postfix is None:
            postfix = self._defaultParameters[ 'postfix' ].value
        self.setParameter( 'triggerMatchers'     , triggerMatchers )
        self.setParameter( 'triggerProducer'     , triggerProducer )
        self.setParameter( 'triggerEventProducer', triggerEventProducer )
        self.setParameter( 'sequence'            , sequence )
        self.setParameter( 'hltProcess'          , hltProcess )
        self.setParameter( 'outputModule'        , outputModule )
        self.setParameter( 'postfix'             , postfix )
        self.apply( process )

    def toolCode( self, process ):
        triggerMatchers      = self._parameters[ 'triggerMatchers' ].value
        triggerProducer      = self._parameters[ 'triggerProducer' ].value
        triggerEventProducer = self._parameters[ 'triggerEventProducer' ].value
        sequence             = self._parameters[ 'sequence' ].value
        hltProcess           = self._parameters[ 'hltProcess' ].value
        outputModule         = self._parameters[ 'outputModule' ].value
        postfix              = self._parameters[ 'postfix' ].value

        # Load default producers from existing config file, if needed
        if not hasattr( process, 'triggerMatchingDefaultSequence' ):
            for matcher in triggerMatchers:
                if matcher in self.getDefaultParameters()[ 'triggerMatchers' ].value:
                    process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi" )
                    break

        # Switch on PAT trigger information if needed
        if ( triggerProducer not in _modulesInSequence( process, sequence ) or triggerEventProducer not in _modulesInSequence( process, sequence ) ):
            print '%s():'%( self._label )
            print '    PAT trigger production switched on automatically using'
            print '    switchOnTrigger( process, %s, %s, %s, %s, %s )'%( hltProcess, triggerProducer, triggerEventProducer, sequence, outputModule )
            print _longLine
            switchOnTrigger( process, triggerProducer, triggerEventProducer, sequence, hltProcess, outputModule )

        # Maintain configurations
        prodSequence   = getattr( process, sequence )
        trigEvtProdMod = getattr( process, triggerEventProducer )
        if trigEvtProdMod.patTriggerProducer.value() is not triggerProducer:
            print '%s():'%( self._label )
            print '    Configuration conflict found'
            print '    triggerProducer = %s'%( triggerProducer )
            print '    differs from'
            print '    %s.patTriggerProducer = %s'%( triggerEventProducer, trigEvtProdMod.patTriggerProducer )
            print '    parameter automatically changed'
            print '    ==> triggerProducer --> %s'%( trigEvtProdMod.patTriggerProducer )
            triggerProducer = trigEvtProdMod.patTriggerProducer
        for matcher in triggerMatchers:
            trigMchMod         = getattr( process, matcher )
            trigMchMod.src     = cms.InputTag( trigMchMod.src.getModuleLabel() + postfix )
            trigMchMod.matched = triggerProducer
            if matcher in _modulesInSequence( process, sequence ):
                print '%s():'%( self._label )
                print '    PAT trigger matcher %s exists already in sequence %s'%( matcher, sequence )
                print '    ==> entry re-used'
                print _longLine
            else:
                # Sequence arithmetics for PAT trigger matcher modules
                index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                getattr( process, sequence + 'Trigger' ).insert( index, trigMchMod )
        matchers = getattr( trigEvtProdMod, 'patTriggerMatches' )
        if len( matchers ) > 0:
            print '%s():'%( self._label )
            print '    PAT trigger matchers already attached to existing PATTriggerEventProducer %s'%( triggerEventProducer )
            print '    configuration parameters automatically changed'
            for matcher in matchers:
                trigMchMod = getattr( process, matcher )
                if trigMchMod.matched.value() is not triggerProducer:
                    removeIfInSequence( process, matcher, sequence + 'Trigger' )
                    trigMchMod.matched = triggerProducer
                    index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                    getattr( process, sequence + 'Trigger' ).insert( index, trigMchMod )
                    print '    PAT trigger matcher %s.matched --> %s'%( matcher, trigMchMod.matched )
            print _longLine
        else:
            trigEvtProdMod.patTriggerMatches = cms.VInputTag()
        trigEvtProdMod.patTriggerMatches += triggerMatchers

        # Add event content
        if outputModule is not '':
            patTriggerEventContent = []
            for matcher in triggerMatchers:
                patTriggerEventContent += [ 'keep patTriggerObjectsedmAssociation_%s_%s_%s'%( triggerEventProducer, matcher, process.name_() )
                                          , 'keep *_%s_*_*'%( getattr( process, matcher ).src.value() )
                                          ]
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerMatching = SwitchOnTriggerMatching()


class SwitchOnTriggerMatchingStandAlone( ConfigToolBase ):
    """  Enables trigger matching in PAT
    SwitchOnTriggerMatchingStandAlone( [cms.Process], triggerMatchers = [default list], triggerProducer = 'patTrigger', sequence = 'patDefaultSequence', hltProcess = 'HLT', outputModule = 'out', postfix = '' )
    - [cms.Process]  : the 'cms.Process'
    - triggerMatchers: PAT trigger matcher module labels (list)
                       optional; default: defined in 'triggerMatchingDefaultSequence'
                       (s. PhysicsTools/PatAlgos/python/triggerLayer1/triggerMatcher_cfi.py)
    - triggerProducer: PATTriggerProducer module label;
                       optional, default: 'patTrigger'
    - sequence       : name of sequence to use;
                       optional, default: 'patDefaultSequence'
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
        self.addParameter( self._defaultParameters, 'triggerMatchers', _defaultTriggerMatchers, _defaultTriggerMatchersComment )
        self.addParameter( self._defaultParameters, 'triggerProducer', _defaultTriggerProducer, _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'sequence'       , _defaultSequence       , _defaultSequenceComment )
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
                , sequence        = None
                , hltProcess      = None
                , outputModule    = None
                , postfix         = None
                ):
        if triggerMatchers is None:
            triggerMatchers = self._defaultParameters[ 'triggerMatchers' ].value
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if sequence is None:
            sequence = self._defaultParameters[ 'sequence' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        if postfix is None:
            postfix = self._defaultParameters[ 'postfix' ].value
        self.setParameter( 'triggerMatchers', triggerMatchers )
        self.setParameter( 'triggerProducer', triggerProducer )
        self.setParameter( 'sequence'       , sequence )
        self.setParameter( 'hltProcess'     , hltProcess )
        self.setParameter( 'outputModule'   , outputModule )
        self.setParameter( 'postfix'        , postfix )
        self.apply( process )

    def toolCode( self, process ):
        triggerMatchers = self._parameters[ 'triggerMatchers' ].value
        triggerProducer = self._parameters[ 'triggerProducer' ].value
        sequence        = self._parameters[ 'sequence' ].value
        hltProcess      = self._parameters[ 'hltProcess' ].value
        outputModule    = self._parameters[ 'outputModule' ].value
        postfix         = self._parameters[ 'postfix' ].value

        # Load default producers from existing config file, if needed
        if not hasattr( process, 'triggerMatchingDefaultSequence' ):
            for matcher in triggerMatchers:
                if matcher in self.getDefaultParameters()[ 'triggerMatchers' ].value:
                    process.load( "PhysicsTools.PatAlgos.triggerLayer1.triggerMatcher_cfi" )
                    break

        # Switch on PAT trigger information if needed
        if triggerProducer not in _modulesInSequence( process, sequence ):
            print '%s():'%( self._label )
            print '    PAT trigger production switched on automatically using'
            print '    switchOnTriggerStandAlone( process, %s, %s, %s, %s )'%( hltProcess, triggerProducer, sequence, outputModule )
            print _longLine
            switchOnTriggerStandAlone( process, triggerProducer, sequence, hltProcess, outputModule )

        # Maintain configurations
        for matcher in triggerMatchers:
            trigMchMod         = getattr( process, matcher )
            trigMchMod.src     = cms.InputTag( trigMchMod.src.getModuleLabel() + postfix )
            trigMchMod.matched = triggerProducer
            if matcher in _modulesInSequence( process, sequence ):
                print '%s():'%( self._label )
                print '    PAT trigger matcher %s exists already in sequence %s'%( matcher, sequence )
                print '    ==> entry re-used'
                print _longLine
            else:
                # Sequence arithmetics for PAT trigger matcher modules
                index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                getattr( process, sequence + 'Trigger' ).insert( index, trigMchMod )

        # Add event content
        if outputModule is not '':
            patTriggerEventContent = []
            for matcher in triggerMatchers:
                patTriggerEventContent += [ 'keep patTriggerObjectStandAlonesedmAssociation_%s_*_%s'%( matcher, process.name_() )
                                          , 'keep *_%s_*_*'%( getattr( process, matcher ).src.value() )
                                          ]
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerMatchingStandAlone = SwitchOnTriggerMatchingStandAlone()


class SwitchOnTriggerMatchEmbedding( ConfigToolBase ):
    """  Enables embedding of trigger matches into PAT objects
    SwitchOnTriggerMatchEmbedding( [cms.Process], triggerMatchers = [default list], triggerProducer = 'patTrigger', sequence = 'patDefaultSequence', hltProcess = 'HLT', outputModule = 'out', postfix = '' )
    - [cms.Process]  : the 'cms.Process'
    - triggerMatchers: PAT trigger matcher module labels (list)
                       optional; default: defined in 'triggerMatchingDefaultSequence'
                       (s. PhysicsTools/PatAlgos/python/triggerLayer1/triggerMatcher_cfi.py)
    - triggerProducer: PATTriggerProducer module label;
                       optional, default: 'patTrigger'
    - sequence       : name of sequence to use;
                       optional, default: 'patDefaultSequence'
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
        self.addParameter( self._defaultParameters, 'triggerMatchers', _defaultTriggerMatchers, _defaultTriggerMatchersComment )
        self.addParameter( self._defaultParameters, 'triggerProducer', _defaultTriggerProducer, _defaultTriggerProducerComment )
        self.addParameter( self._defaultParameters, 'sequence'       , _defaultSequence       , _defaultSequenceComment )
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
                , sequence        = None
                , hltProcess      = None
                , outputModule    = None
                , postfix         = None
                ):
        if triggerMatchers is None:
            triggerMatchers = self._defaultParameters[ 'triggerMatchers' ].value
        if triggerProducer is None:
            triggerProducer = self._defaultParameters[ 'triggerProducer' ].value
        if sequence is None:
            sequence = self._defaultParameters[ 'sequence' ].value
        if hltProcess is None:
            hltProcess = self._defaultParameters[ 'hltProcess' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        if postfix is None:
            postfix = self._defaultParameters[ 'postfix' ].value
        self.setParameter( 'triggerMatchers', triggerMatchers )
        self.setParameter( 'triggerProducer', triggerProducer )
        self.setParameter( 'sequence'       , sequence )
        self.setParameter( 'hltProcess'     , hltProcess )
        self.setParameter( 'outputModule'   , outputModule )
        self.setParameter( 'postfix'        , postfix )
        self.apply( process )

    def toolCode( self, process ):
        triggerMatchers = self._parameters[ 'triggerMatchers' ].value
        triggerProducer = self._parameters[ 'triggerProducer' ].value
        sequence        = self._parameters[ 'sequence' ].value
        hltProcess      = self._parameters[ 'hltProcess' ].value
        outputModule    = self._parameters[ 'outputModule' ].value
        postfix         = self._parameters[ 'postfix' ].value

        # Build dictionary of known input collections
        dictPatObjects = { 'Photons'  : 'PATTriggerMatchPhotonEmbedder'
                         , 'Electrons': 'PATTriggerMatchElectronEmbedder'
                         , 'Muons'    : 'PATTriggerMatchMuonEmbedder'
                         , 'Taus'     : 'PATTriggerMatchTauEmbedder'
                         , 'Jets'     : 'PATTriggerMatchJetEmbedder'
                         , 'METs'     : 'PATTriggerMatchMETEmbedder'
                         }
        listPatSteps   = [ 'pat', 'selectedPat', 'cleanPat' ]
        listJetAlgos   = [ 'IC5', 'SC5', 'KT4', 'KT6', 'AK5' ]
        listJetTypes   = [ 'Calo', 'PF', 'JPT' ]
        dictEmbedders  = {}
        for objects in dictPatObjects.keys():
            steps = len( listPatSteps )
            if objects is 'METs':
                steps = 1
            for step in range( steps ):
                coll = listPatSteps[ step ] + objects
                dictEmbedders[ coll ]           = dictPatObjects[ objects ]
                dictEmbedders[ coll + postfix ] = dictPatObjects[ objects ]
                if objects is 'Jets':
                    for jetAlgo in listJetAlgos:
                        for jetType in listJetTypes:
                            jetColl = coll + jetAlgo + jetType
                            dictEmbedders[ jetColl ]           = dictPatObjects[ objects ]
                            dictEmbedders[ jetColl + postfix ] = dictPatObjects[ objects ]

        # Build dictionary of matchers and switch on PAT trigger matching if needed
        dictConfig = {}
        matchingOn = False
        for matcher in triggerMatchers:
            trigMchMod = getattr( process, matcher )
            if trigMchMod.src.value() in dictConfig:
                dictConfig[ trigMchMod.src.value() ] += [ matcher ]
            else:
                dictConfig[ trigMchMod.src.value() ] = [ matcher ]
            if matcher not in _modulesInSequence( process, sequence ) and not matchingOn:
                print '%s():'%( self._label )
                print '    PAT trigger matching switched on automatically using'
                print '    switchOnTriggerMatchingStandAlone( process, %s, %s, %s, %s, %s )'%( hltProcess, triggerMatchers, triggerProducer, sequence, outputModule )
                print _longLine
                switchOnTriggerMatchingStandAlone( process, triggerMatchers, triggerProducer, sequence, hltProcess, '', postfix )
                matchingOn = True

        # Maintain configurations
        patTriggerEventContent = []
        for srcInput in dictConfig.keys():
            if dictEmbedders.has_key( srcInput ):
                # Configure embedder module
                dictIndex = srcInput
                srcInput += postfix
                if dictEmbedders.has_key( srcInput ):
                    label = srcInput + 'TriggerMatch'
                    if label in _modulesInSequence( process, sequence ):
                        print '%s():'%( self._label )
                        print '    PAT trigger match embedder %s exists already in sequence %s'%( label, sequence )
                        print '    ==> entry moved to proper place'
                        print _longLine
                        removeIfInSequence( process, label, sequence + 'Trigger' )
                    module         = cms.EDProducer( dictEmbedders[ dictIndex ] )
                    module.src     = cms.InputTag( srcInput )
                    module.matches = cms.VInputTag( dictConfig[ dictIndex ] )
                    setattr( process, label, module )
                    trigEmbMod = getattr( process, label )
                    index = len( getattr( process, sequence + 'Trigger' ).moduleNames() )
                    getattr( process, sequence + 'Trigger' ).insert( index, trigEmbMod )
                    # Add event content
                    patTriggerEventContent += [ 'drop *_%s_*_*'%( srcInput )
                                              , 'keep *_%s_*_%s'%( label, process.name_() )
                                              ]
                else:
                    print '%s():'%( self._label )
                    print '    Invalid new input source for trigger match embedding'
                    print '    ==> %s with matchers %s is skipped'%( srcInput, dictConfig[ dictIndex ] )
                    print _longLine
            else:
                print '%s():'%( self._label )
                print '    Invalid input source for trigger match embedding'
                print '    ==> %s with matchers %s is skipped'%( srcInput, dictConfig[ srcInput ] )
                print _longLine
        if outputModule is not '':
            getattr( process, outputModule ).outputCommands = _addEventContent( getattr( process, outputModule ).outputCommands, patTriggerEventContent )

switchOnTriggerMatchEmbedding = SwitchOnTriggerMatchEmbedding()


class RemoveCleaningFromTriggerMatching( ConfigToolBase ):
    """  Removes cleaning from already existing PAT trigger matching/embedding configuration
    RemoveCleaningFromTriggerMatching( [cms.Process], outputModule = 'out' )
    - [cms.Process]  : the 'cms.Process'
    - sequence       : name of sequence to use;
                       optional, default: 'patDefaultSequence'
    - outputModule   : output module label;
                       empty label indicates no output;
                       optional, default: 'out'
    Using None as any argument restores its default value.
    """
    _label             = 'removeCleaningFromTriggerMatching'
    _defaultParameters = dicttypes.SortedKeysDict()

    def __init__( self ):
        ConfigToolBase.__init__( self )
        self.addParameter( self._defaultParameters, 'sequence'    , _defaultSequence    , _defaultSequenceComment )
        self.addParameter( self._defaultParameters, 'outputModule', _defaultOutputModule, _defaultOutputModuleComment )
        self._parameters = copy.deepcopy( self._defaultParameters )
        self._comment = ""

    def getDefaultParameters( self ):
        return self._defaultParameters

    def __call__( self, process
                , sequence     = None
                , outputModule = None
                ):
        if sequence is None:
            sequence = self._defaultParameters[ 'sequence' ].value
        if outputModule is None:
            outputModule = self._defaultParameters[ 'outputModule' ].value
        self.setParameter( 'sequence'    , sequence )
        self.setParameter( 'outputModule', outputModule )
        self.apply( process )

    def toolCode( self, process ):
        sequence     = self._parameters[ 'sequence' ].value
        outputModule = self._parameters[ 'outputModule' ].value

        # Maintain configurations
        listMatchers = [ 'PATTriggerMatcherDRLessByR'
                       , 'PATTriggerMatcherDRDPtLessByR'
                       , 'PATTriggerMatcherDRLessByPt'
                       , 'PATTriggerMatcherDRDPtLessByPt'
                       , 'PATTriggerMatcherDEtaLessByDR'
                       , 'PATTriggerMatcherDEtaLessByDEta'
                       ]
        listEmbedders = [ 'PATTriggerMatchPhotonEmbedder'
                        , 'PATTriggerMatchElectronEmbedder'
                        , 'PATTriggerMatchMuonEmbedder'
                        , 'PATTriggerMatchTauEmbedder'
                        , 'PATTriggerMatchJetEmbedder'
                        , 'PATTriggerMatchMETEmbedder'
                        ]
        modules = _modulesInSequence( process, sequence )
        oldModules = []
        oldSources = []
        # input source labels
        for module in modules:
            if hasattr( process, module ):
                trigMod = getattr( process, module )
                if trigMod.type_() in listMatchers:
                    if trigMod.src.value()[ : 8 ] == 'cleanPat':
                        trigMod.src = trigMod.src.value().replace( 'cleanPat', 'selectedPat' )
                        if trigMod.label()[ : 5 ] == 'clean':
                            oldModules += [ trigMod.label() ]
                            setattr( process, trigMod.label().replace( 'clean', 'selected' ), trigMod )
                if trigMod.type_() in listEmbedders:
                    if trigMod.src.value()[ : 8 ] == 'cleanPat':
                        oldSources += [ trigMod.src.getModuleLabel() ]
                        trigMod.src = trigMod.src.value().replace( 'cleanPat', 'selectedPat' )
                        if trigMod.label()[ : 5 ] == 'clean':
                            oldModules += [ trigMod.label() ]
                            setattr( process, trigMod.label().replace( 'clean', 'selected' ), trigMod )
        # matcher labels
        for module in modules:
            if hasattr( process, module ):
                trigMod = getattr( process, module )
                if trigMod.type_() == 'PATTriggerEventProducer':
                    matchers = getattr( trigMod, 'patTriggerMatches' )
                    matchers = self._renameMatchers( matchers, oldModules )
                elif trigMod.type_() in listEmbedders:
                    matchers = getattr( trigMod, 'matches' )
                    matchers = self._renameMatchers( matchers, oldModules )

        # Maintain event content
        if outputModule is not '':
            patTriggerEventContent = getattr( process, outputModule ).outputCommands
            for statement in range( len( patTriggerEventContent ) ):
                for module in oldModules:
                    if module in patTriggerEventContent[ statement ]:
                        patTriggerEventContent[ statement ] = patTriggerEventContent[ statement ].replace( 'clean', 'selected' )
                for source in oldSources:
                    if source in patTriggerEventContent[ statement ] and 'drop' in patTriggerEventContent[ statement ]:
                        patTriggerEventContent[ statement ] = patTriggerEventContent[ statement ].replace( 'clean', 'selected' )
        print '%s():'%( self._label )
        print '    Input from cleaning has been switched to input from selection;'
        print '    matcher and embedder modules have been renamed accordingly.'
        print _longLine

    def _renameMatchers( self, matchers, oldModules ):
        for matcher in range( len( matchers ) ):
            if matchers[ matcher ] in oldModules:
                if matchers[ matcher ][ : 5 ] == 'clean':
                     matchers[ matcher ] = matchers[ matcher ].replace( 'clean', 'selected' )
        return matchers

removeCleaningFromTriggerMatching = RemoveCleaningFromTriggerMatching()
