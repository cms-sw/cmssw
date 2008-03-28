import FWCore.ParameterSet.Config as cms

# raw-to-clusters facility
from EventFilter.SiStripRawToDigi.SiStripRawToClusters_cfi import *
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerSiStripRefGetterProducer_cfi import *
# module to produce refgetter for on demand tracking
siStripClusters = copy.deepcopy(measurementTrackerSiStripRefGetterProducer)
# modify measurementTracker to use refGetter
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
SiStripRawToClusters = cms.Sequence(SiStripRawToClustersFacility*siStripClusters)
siStripClusters.measurementTrackerName = ''
MeasurementTracker.Regional = True
MeasurementTracker.OnDemand = True
MeasurementTracker.stripLazyGetterProducer = 'SiStripRawToClustersFacility'

