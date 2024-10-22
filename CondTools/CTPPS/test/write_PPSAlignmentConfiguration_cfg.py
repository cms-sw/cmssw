##### configuration #####
output_conditions = 'sqlite_file:alignment_config.db'  # output database
run_number = 1  # beginning of the IOV
db_tag = 'PPSAlignmentConfiguration_v1_express'  # database tag
produce_logs = True  # if set to True, a file with logs will be produced.
product_instance_label = 'db_test'  # ES product label
# ESSource parameters can be configured below
#########################

import FWCore.ParameterSet.Config as cms

process = cms.Process("writePPSAlignmentConfiguration")

# Message Logger
if produce_logs:
    process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring('write_PPSAlignmentConfiguration',
                                             'cout'
                                            ),
        write_PPSAlignmentConfiguration = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        ),
        cout = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING')
        )
    )
else:
    process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring('cout'),
        cout = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING')
        )
    )

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database
process.CondDB.connect = output_conditions

# A data source must always be defined. We don't need it, so here's a dummy one.
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(run_number),
    lastValue = cms.uint64(run_number),
    interval = cms.uint64(1)
)

# output service
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('PPSAlignmentConfigurationRcd'),
        tag = cms.string(db_tag)
    ))
)

# ESSource
process.ppsAlignmentConfigurationESSource = cms.ESSource("PPSAlignmentConfigurationESSource",
    # PPSAlignmentConfigurationESSource parameters, defaults will be taken from fillDescriptions
    label = cms.string(product_instance_label),
    sector_45 = cms.PSet(
        rp_N = cms.PSet(
            name = cms.string('db_test_RP'),
            id = cms.int32(44),
            y_max_fit_mode = cms.double(66.6)
        )
    ),
    y_alignment = cms.PSet(
        rp_L_F = cms.PSet(
            x_min = cms.double(102),
            x_max = cms.double(210.0)
        )
    ),
    extra_params = cms.vdouble( 1.15, 102., 777.7 )
)

# DB object maker
process.config_writer = cms.EDAnalyzer("WritePPSAlignmentConfiguration",
    record = cms.string('PPSAlignmentConfigurationRcd'),
    loggingOn = cms.untracked.bool(True),
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
        IOVRun = cms.untracked.uint32(1)
    ),
    label = cms.string(product_instance_label)
)

process.path = cms.Path(process.config_writer)
