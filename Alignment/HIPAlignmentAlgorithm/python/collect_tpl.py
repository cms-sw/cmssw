import FWCore.ParameterSet.Config as cms
process = cms.Process("Alignment")

# "including" common configuration
<COMMON>

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.AlignmentProducer.algoConfig.collectorActive = True
process.AlignmentProducer.algoConfig.collectorNJobs  = <JOBS>
process.AlignmentProducer.algoConfig.collectorPath   = '<PATH>'
process.AlignmentProducer.algoConfig.minimumNumberOfHits = 0
process.AlignmentProducer.algoConfig.maxRelParameterError = '1e99'
process.AlignmentProducer.algoConfig.outpath = '<PATH>/main/'

"""
# this part only needed for survey constraint
process.survey = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerSurveyRcd'),
        tag = cms.string('valueTag')
    ), 
        cms.PSet(
            record = cms.string('TrackerSurveyErrorRcd'),
            tag = cms.string('errorTag')
        )),
    connect = cms.string('sqlite_file:/afs/cern.ch/cms/CAF/CMSALCA/ALCA_TRACKERALIGN/HIP/surveyObjects/measurementSurvey_207.db')
)

process.AlignmentProducer.algoConfig.surveyResiduals = ['Det']
process.AlignmentProducer.useSurvey = True
# end of survey constraint part
"""
