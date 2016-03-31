import FWCore.ParameterSet.Config as cms

# Default histogram configuration. This is _not_ used automatically, but you 
# can import and pass this (or clones of it) in the plugin config.
DefaultHisto = cms.PSet(
  # Setting this to False hides all plots of this HistogramManager. It does not even record any data.
  enabled = cms.bool(True),
  # where the plots should go.
  topFolderName = cms.string("PixelPhase1"),
  # Ths grouping is used if the plugin uses histo[...].defaultGrouping(). It should be era-dependent.
  defaultGrouping = cms.string("P1PXBBarrel|P1PXECEndcap/P1PXBHalfBarrel|P1PXECHalfCylinder/P1PXBLayer|P1PXECHalfDisk/P1PXBLadder|P1PXECBlade"),
  # You can add secs here tat you would like to see in addition to the ones declared in the source. 
  # Doing this in the default config is a very bad idea, just here for documentation.
  additionalSpecs = cms.VPSet(
    cms.PSet(spec = 
      cms.vstring("groupBy P1PXBBarrel|P1PXECEndcap/Module",
		  "save")
    )
  )
)

