import FWCore.ParameterSet.Config as cms

def customizeHLTforAll(process, menuType = "GRun", _customInfo = None):

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
                from Configuration.AlCa.GlobalTag import GlobalTag
                process.GlobalTag = GlobalTag(process.GlobalTag, _globalTag, '')
#               process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")

# inputFile
        if _inputFile[0] == "@":
            _inputFile[0] = _customInfo['inputFiles'][_realData]
        if _inputFile != "":
            if hasattr(process,'source'):
                process.source.fileNames = cms.untracked.vstring( _inputFile )

        if not _realData:
            from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforMC
            process = customizeHLTforMC(process)

    return process
