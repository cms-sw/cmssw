process Alignment =
{
  include "Alignment/CommonAlignmentProducer/data/AlignmentProducerCSA07.cff"

  service = MessageLogger
  {
    untracked vstring destinations = {'cout'}

    untracked PSet cout = { untracked string threshold = 'WARNING' }
  }

  source = PoolSource { untracked vstring fileNames = {'<FILE>'} }

  untracked PSet maxEvents = { untracked int32 input = <EVTS> }

  replace HIPAlignmentAlgorithm.outpath = '<PATH>/' # must put backslash
#  replace HIPAlignmentAlgorithm.apeParam = 'none'
  replace HIPAlignmentAlgorithm.minimumNumberOfHits = 0
#  replace HIPAlignmentAlgorithm.surveyResiduals = {'Panel', 'PixelEndcap'}

  replace AlignmentProducer.maxLoops = 1
  replace AlignmentProducer.algoConfig = { using HIPAlignmentAlgorithm }
  replace AlignmentProducer.doMisalignmentScenario = false

  replace AlignmentProducer.monitorConfig.AlignmentMonitorGeneric =
  {
    string outpath = '<PATH>/'
    string outfile = 'histograms.root'
    bool collectorActive = false
    int32 collectorNJobs = 0
    string collectorPath = './'
  }

  path p = { TrackRefitter }
}
