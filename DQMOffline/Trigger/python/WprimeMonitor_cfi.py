import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topMonitoring_cfi import topMonitoring

hltWprimemonitoring = topMonitoring.clone()
hltWprimemonitoring.FolderName = cms.string('HLT/TopHLTOffline/TopMonitor/default/')

hltWprimemonitoring.histoPSet.ptPSet = cms.PSet(
  nbins = cms.uint32(  45   ),
  xmin  = cms.double(   100   ),
  xmax  = cms.double(  1000  ),
)


hltWprimemonitoring.histoPSet.elePtBinning = cms.vdouble(100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.)
hltWprimemonitoring.histoPSet.elePtBinning2D = cms.vdouble(100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.)


