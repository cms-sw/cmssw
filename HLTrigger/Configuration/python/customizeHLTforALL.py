import FWCore.ParameterSet.Config as cms

def customizeHLTforAll(process, _customInfo = None):

    if (_customInfo is not None):

        _maxEvents = _customInfo['maxEvents']
        _globalTag = _customInfo['globalTag']
        _inputFile = _customInfo['inputFile']
        _realData  = _customInfo['realData']
        
        import FWCore.ParameterSet.VarParsing as VarParsing
        cmsRunOptions = VarParsing.VarParsing('python')

        cmsRunOptions.maxEvents  = _maxEvents
        cmsRunOptions.register('globalTag',_globalTag,cmsRunOptions.multiplicity.singleton,cmsRunOptions.varType.string,"GlobalTag")
        cmsRunOptions.inputFiles = _inputFile
        cmsRunOptions.register('realData',_realData,cmsRunOptions.multiplicity.singleton,cmsRunOptions.varType.bool,"Real Data?")

        cmsRunOptions.parseArguments()

# report in log file
#       print cmsRunOptions

        _maxEvents = cmsRunOptions.maxEvents
        _globalTag = cmsRunOptions.globalTag
        _inputFile = cmsRunOptions.inputFiles
        _realData  = cmsRunOptions.realData

# maxEvents
        if _maxEvents != -2:
            _maxEvents = cms.untracked.int32( _maxEvents )
            if hasattr(process,'maxEvents'):
                process.maxEvents.input = _maxEvents
            else:
                process.maxEvents = cms.untracked.PSet( input = _maxEvents )

# GlobalTag
        if _globalTag == "@":
            _globalTag = _customInfo['globalTags'][_realData]
        if _globalTag != "":
            if hasattr(process,'GlobalTag'):
                from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
                process.GlobalTag = GlobalTag(process.GlobalTag, _globalTag, '')

# inputFile
        if _inputFile[0] == "@":
            _inputFile[0] = _customInfo['inputFiles'][_realData]
        if _inputFile != "":
            if hasattr(process,'source'):
                process.source.fileNames = cms.untracked.vstring( _inputFile )
                    
# MC customisation
        if not _realData:
            from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
            process = customizeHLTforMC(process)
            if _customInfo['menuType'] == "HIon":
                from HLTrigger.Configuration.CustomConfigs import MassReplaceInputTag
                process = MassReplaceInputTag(process,"rawDataRepacker","rawDataCollector")
    else:
        pass

# CMSSW version customisation
    from HLTrigger.Configuration.customizeHLTforCMSSW import customiseHLTforCMSSW
    process = customiseHLTforCMSSW(process)

    return process
