# customization fragments to be used with cmsDriver and hltGetConfiguration
#
# V.M. Ghete 2010-06-09 initial version

import FWCore.ParameterSet.Config as cms

def customiseUnprescaleAlgoTriggers(process):

    # temporary solution FIXME

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPrescaleFactorsAlgoTrigConfig_cff")
    process.es_prefer_l1GtPrescaleFactorsAlgoTrig = cms.ESPrefer(
        "L1GtPrescaleFactorsAlgoTrigTrivialProducer", "l1GtPrescaleFactorsAlgoTrig")


    return (process)

##############################################################################

def customiseUnprescaleTechTriggers(process):

    # temporary solution FIXME

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtPrescaleFactorsTechTrigConfig_cff")
    process.es_prefer_l1GtPrescaleFactorsTechTrig = cms.ESPrefer(
        "L1GtPrescaleFactorsTechTrigTrivialProducer", "l1GtPrescaleFactorsTechTrig")

    return (process)

##############################################################################

def customiseResetMasksAlgoTriggers(process):

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff")
    process.es_prefer_l1GtTriggerMaskAlgoTrig = cms.ESPrefer(
        "L1GtTriggerMaskAlgoTrigTrivialProducer", "l1GtTriggerMaskAlgoTrig")

    return (process)

##############################################################################

def customiseResetMasksTechTriggers(process):

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff")
    process.es_prefer_l1GtTriggerMaskTechTrig = cms.ESPrefer(
        "L1GtTriggerMaskTechTrigTrivialProducer", "l1GtTriggerMaskTechTrig")

    return (process)

##############################################################################

def customiseResetVetoMasksAlgoTriggers(process):

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskVetoAlgoTrigConfig_cff")
    process.es_prefer_l1GtTriggerMaskVetoAlgoTrig = cms.ESPrefer(
        "L1GtTriggerMaskVetoAlgoTrigTrivialProducer", "l1GtTriggerMaskVetoAlgoTrig")

    return (process)

##############################################################################

def customiseResetVetoMasksTechTriggers(process):

    process.load("L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskVetoTechTrigConfig_cff")
    process.es_prefer_l1GtTriggerMaskVetoTechTrig = cms.ESPrefer(
        "L1GtTriggerMaskVetoTechTrigTrivialProducer", "l1GtTriggerMaskVetoTechTrig")

    return (process)

##############################################################################

def customiseResetPrescalesAndMasks(process):
    process = customiseUnprescaleAlgoTriggers( process )
    process = customiseUnprescaleTechTriggers( process )
    process = customiseResetMasksAlgoTriggers( process )
    process = customiseResetMasksTechTriggers( process )
    process = customiseResetVetoMasksAlgoTriggers( process )
    process = customiseResetVetoMasksTechTriggers( process )

    return (process)

##############################################################################

def customiseL1Menu(process):

    # replace the L1 menu from the global tag with one of the following alternatives

    ####### user choices

    #l1MenuSource='globalTag'
    #l1MenuSource='sqlFile'
    l1MenuSource='xmlFile'


    if l1MenuSource == 'sqlFile' :
        # the menu will be read from the SQL file instead of the global tag
        useSqlFile = '/afs/cern.ch/user/g/ghete/public/L1Menu/L1Menu_Collisions2015_25ns_v1/sqlFile/L1Menu_Collisions2015_25ns_v1_mc.db'
        menuDbTag = 'L1GtTriggerMenu_L1Menu_Collisions2015_25ns_v1_mc'
    elif l1MenuSource == 'xmlFile' :
        # the menu will be read from an XML file instead of the global tag - must copy the file in luminosityDirectory
        luminosityDirectory = "startup"
        useXmlFile = 'L1Menu_Collisions2015_25ns_v1_L1T_Scales_20101224_Imp0_0x102f.xml'

    else :
        print '   Using default L1 trigger menu from Global Tag '

    ####### end of user choices - do not change the following

    if l1MenuSource == 'xmlFile' :
        print '   Retrieve L1 trigger menu only from XML file '
        print '       ', useXmlFile
        print '       '

        process.load('L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi')
        process.l1GtTriggerMenuXml.TriggerMenuLuminosity = luminosityDirectory
        process.l1GtTriggerMenuXml.DefXmlFile = useXmlFile

        process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
        process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')



    elif l1MenuSource == 'sqlFile' :
        if useSqlFile != '' :
            print '   Retrieve L1 trigger menu only from SQLlite file '
            print '       ', useSqlFile
            print '       '

            from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
            process.l1conddb = cms.ESSource("PoolDBESSource",
                                    CondDBSetup,
                                    connect = cms.string('sqlite_file:' + useSqlFile),
                                    toGet = cms.VPSet(cms.PSet(
                                                record = cms.string('L1GtTriggerMenuRcd'),
                                                tag = cms.string(menuDbTag))),
                                                BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
                                            )
            process.es_prefer_l1conddb = cms.ESPrefer("PoolDBESSource", "l1conddb")

        else :
            print '   Error: no SQL file is given; please provide a valid SQL file for option sqlFile'

    else :
        print ''


    return process

##############################################################################

def customiseL1Menu_HI(process):

    # replace the L1 menu from the global tag with one of the following alternatives

    luminosityDirectory = "startup"
    useXmlFile = 'L1Menu_CollisionsHeavyIons2011_v0_nobsc_notau_centrality_q2_singletrack.v1.xml'

    print '   Retrieve L1 trigger menu only from XML file '
    print '       ', useXmlFile
    print '       '

    process.load('L1TriggerConfig.L1GtConfigProducers.l1GtTriggerMenuXml_cfi')
    process.l1GtTriggerMenuXml.TriggerMenuLuminosity = luminosityDirectory
    process.l1GtTriggerMenuXml.DefXmlFile = useXmlFile

    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')

    return process

##############################################################################

def customiseOutputCommands(process):

    # customization of output commands, on top of the output commands selected
    # in cmsDriver command

    # examples

    # drop all products, keep only the products from L1EmulRaw process and the FEDRawDataCollection_
    #process.output.outputCommands.append('drop *_*_*_*')
    #process.output.outputCommands.append('keep *_*_*_L1EmulRaw')
    #process.output.outputCommands.append('keep FEDRawDataCollection_*_*_*')


    return process


##############################################################################

def customiseL1EmulatorFromRaw(process):
    # customization fragment to run L1 emulator starting from a RAW file

    # run trigger primitive generation on unpacked digis
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')

    process.CaloTPG_SimL1Emulator = cms.Sequence(
        process.CaloTriggerPrimitives +
        process.SimL1Emulator )

    for path in process._Process__paths.itervalues():
        path.replace(process.SimL1Emulator, process.CaloTPG_SimL1Emulator)

    # set the new input tags after RawToDigi
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )

    process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'muonCSCDigis', 'MuonCSCComparatorDigi' )
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer       = cms.InputTag( 'muonCSCDigis', 'MuonCSCWireDigi' )
    process.simRpcTriggerDigis.label         = 'muonRPCDigis'
    process.simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis'

    return process

##############################################################################

def customiseL1GtEmulatorFromRaw(process):
    # customization fragment to run L1 GT emulator starting from a RAW file, with input from unpacked GCT and GMT products
    # assuming that "RawToDigi_cff" (or "RawToDigi_data_cff") and "SimL1Emulator_cff" have already been loaded

    # producers for technical triggers:
    # they must be re-run as their output is not available from RAW2DIGI

    # BSC Technical Trigger
    # Note: will normally not work, it requires SimHits (not available from RAW2DIGI)
    # works only on some MC samples where the SimHits are saved together with the FEDRaw
    import L1TriggerOffline.L1Analyzer.bscTrigger_cfi
    process.simBscDigis = L1TriggerOffline.L1Analyzer.bscTrigger_cfi.bscTrigger.clone()

    # RPC Technical Trigger
    import L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi
    process.simRpcTechTrigDigis = L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi.rpcTechnicalTrigger.clone()

    process.simRpcTriggerDigis.label = 'muonRPCDigis'
    process.simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis'

    # HCAL Technical Trigger
    import SimCalorimetry.HcalTrigPrimProducers.hcalTTPRecord_cfi
    process.simHcalTechTrigDigis = SimCalorimetry.HcalTrigPrimProducers.hcalTTPRecord_cfi.simHcalTTPRecord.clone()


    # Global Trigger emulator

    # do not run calo emulators - instead, use unpacked GCT digis for GT input
    process.simGtDigis.GctInputTag = 'gctDigis'

    # do not run muon emulators - instead, use unpacked GMT digis for GT input
    # (GMT digis produced by same module as the GT digis, as GT and GMT have common unpacker)
    process.simGtDigis.GmtInputTag = 'gtDigis'

    # technical triggers
    process.simGtDigis.TechnicalTriggersInputTags = cms.VInputTag(
        cms.InputTag( 'simBscDigis' ),
        cms.InputTag( 'simRpcTechTrigDigis' ),
        cms.InputTag( 'simHcalTechTrigDigis' )
        )

    process.SimL1TechnicalTriggers = cms.Sequence(
        process.simBscDigis +
        process.simRpcTechTrigDigis +
        process.simHcalTechTrigDigis
        )

    # run producers for technical triggers, L1 GT emulator only
    SimL1Emulator = cms.Sequence(
        process.SimL1TechnicalTriggers +
        process.simGtDigis )

    # replace the SimL1Emulator in all paths and sequences
    for iterable in process.sequences.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    for iterable in process.paths.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    for iterable in process.endpaths.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    process.SimL1Emulator = SimL1Emulator

    return process

##############################################################################

def customiseL1CaloAndGtEmulatorsFromRaw(process):
    # customization fragment to run calorimeter emulators (TPGs and L1 calorimeter emulators)
    # and GT emulator starting from a RAW file assuming that "RawToDigi_cff" and "SimL1Emulator_cff"
    # have already been loaded

    # run Calo TPGs on unpacked digis
    process.load('L1Trigger.Configuration.CaloTriggerPrimitives_cff')
    process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
    process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(
        cms.InputTag('hcalDigis'),
        cms.InputTag('hcalDigis')
    )

    # do not run muon emulators - instead, use unpacked GMT digis for GT input
    # (GMT digis produced by same module as the GT digis, as GT and GMT have common unpacker)
    process.simRpcTechTrigDigis.RPCDigiLabel = 'muonRPCDigis'
    process.simGtDigis.GmtInputTag = 'gtDigis'

    # run Calo TPGs, L1 GCT, technical triggers, L1 GT
    SimL1Emulator = cms.Sequence(
        process.CaloTriggerPrimitives +
        process.simRctDigis +
        process.simGctDigis +
        process.SimL1TechnicalTriggers +
        process.simGtDigis )

    # replace the SimL1Emulator in all paths and sequences
    for iterable in process.sequences.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    for iterable in process.paths.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    for iterable in process.endpaths.itervalues():
        iterable.replace( process.SimL1Emulator, SimL1Emulator)
    process.SimL1Emulator = SimL1Emulator

    return process

##############################################################################

def customiseL1TriggerReport(process):

    process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

    # boolean flag to select the input record
    # if true, it will use L1GlobalTriggerRecord
    #process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

    # input tag for GT record:
    #   GT emulator:    gtDigis (DAQ record)
    #   GT unpacker:    gtDigis (DAQ record)
    #   GT lite record: l1GtRecord
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"

    process.l1GtTrigReport.PrintVerbosity = 10
    process.l1GtTrigReport.PrintOutput = 0



    #
    return (process)

##############################################################################
