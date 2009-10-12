import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *

PoolDBESSource = cms.ESSource("PoolDBESSource",
                                      CondDBCommon,
                                      toGet = cms.VPSet(
    #
    # working points
    #
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('CSVL_WP'),
    label = cms.untracked.string('CSVL_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('CSVM_WP'),
    label = cms.untracked.string('CSVM_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('CSVT_WP'),
    label = cms.untracked.string('CSVT_WP')
    ),


    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('SSVM_WP'),
    label = cms.untracked.string('SSVM_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('SSVT_WP'),
    label = cms.untracked.string('SSVT_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('JPL_WP'),
    label = cms.untracked.string('JPL_WP')
    ),
    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('JPM_WP'),
    label = cms.untracked.string('JPM_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('JPT_WP'),
    label = cms.untracked.string('JPT_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('TCHEL_WP'),
    label = cms.untracked.string('TCHEL_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('TCHEM_WP'),
    label = cms.untracked.string('TCHEM_WP')
    ),

    cms.PSet(
    record = cms.string('PerformanceWPRecord'),
    tag = cms.string('TCHPT_WP'),
    label = cms.untracked.string('TCHPT_WP')
    ),
    #
    # tables
    #
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('CSVL_T'),
    label = cms.untracked.string('CSVL_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('CSVM_T'),
    label = cms.untracked.string('CSVM_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('CSVT_T'),
    label = cms.untracked.string('CSVT_T')
    ),


    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('SSVM_T'),
    label = cms.untracked.string('SSVM_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('SSVT_T'),
    label = cms.untracked.string('SSVT_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('JPL_T'),
    label = cms.untracked.string('JPL_T')
    ),
    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('JPM_T'),
    label = cms.untracked.string('JPM_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('JPT_T'),
    label = cms.untracked.string('JPT_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('TCHEL_T'),
    label = cms.untracked.string('TCHEL_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('TCHEM_T'),
    label = cms.untracked.string('TCHEM_T')
    ),

    cms.PSet(
    record = cms.string('PerformancePayloadRecord'),
    tag = cms.string('TCHPT_T'),
    label = cms.untracked.string('TCHPT_T')
    )
    ))
 

