import FWCore.ParameterSet.Config as cms

from DQM.SiPixelPhase1Common.SpecificationBuilder_cfi import Specification

# Default histogram configuration. This is _not_ used automatically, but you 
# can import and pass this (or clones of it) in the plugin config.
DefaultHisto = cms.PSet(
  # Setting this to False hides all plots of this HistogramManager. It does not even record any data.
  enabled = cms.bool(True),
  # If False, no histograsm are booked for DetIds where any column is undefined.
  # This is important to avoid booking lots of unused histograms for SiStrip IDs.
  bookUndefined = cms.bool(False),
  # where the plots should go.
  topFolderName = cms.string("PixelPhase1"),
  # Ths grouping is used if the plugin uses histo[...].defaultGrouping(). It should be era-dependent.
  defaultGrouping = cms.string("P1PXBBarrel|P1PXECEndcap/P1PXBLayer|P1PXECHalfDisk/P1PXBLadder|P1PXECBlade"),
  # You can add specs here that you would like to see in addition to the ones declared in the source. 
  # Doing this in the default config is a very bad idea, just here for documentation.
  # This structure is output by the SpecficationBuilder.
  specs = cms.VPSet()
  #  cms.PSet(spec = 
  #    cms.VPset(
  #      cms.PSet(
  #        type = GROUPBY, 
  #        stage = FIRST,
  #        columns = cms.vstring("P1PXBBarrel|P1PXECEndcap", "DetId"),
  #        arg = cms.string("")
  #      ),
  #     cms.PSet(
  #       type = SAVE,
  #       stage = STAGE1,
  #       columns = cms.vstring(),
  #       arg = cms.string("")
  #	)
  #   )
  # )
  #)
)

