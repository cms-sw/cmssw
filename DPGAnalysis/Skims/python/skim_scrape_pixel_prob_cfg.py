import FWCore.ParameterSet.Config as cms

# NOTE: This filter counts the number of pixel hits incompatible
# with the track direction. If this fraction is high (>0.4) the
# event is declared to be a PKAM.
# Starting in CMSSW_3_6_0_pre3 pixel hits with low probability
# are removed from tracks in the outlier rejection. To make
# sure that this filter still works, one has to turn this feature
# OFF here:
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/TrackingTools/TrackFitters/python/RungeKuttaFitters_cff.py
# by changing:
# LogPixelProbabilityCut = cms.double(-14.0), 
# to:
# LogPixelProbabilityCut = cms.double(-16.0),


process = cms.Process("SKIM2")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.skimming = cms.EDFilter("FilterScrapingPixelProbability",
                                apply_filter                 = cms.untracked.bool( True  ),
                                select_collision             = cms.untracked.bool( False ),
                                select_pkam                  = cms.untracked.bool( True ),
                                select_other                 = cms.untracked.bool( False ),
                                low_probability              = cms.untracked.double( 0.0 ),
                                low_probability_fraction_cut = cms.untracked.double( 0.4 )
                                )


#process.configurationMetadata = cms.untracked.PSet(
#    version = cms.untracked.string('$Revision: 1.1 $'),
#    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/skim_scrape_pixel_prob_cfg.py,v $'),
#    annotation = cms.untracked.string('NoScrape skim')
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
"/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/BSCNOBEAMHALO-Dec14thSkim_v1/0102/6081FF95-72EA-DE11-94C4-0024E8768446.root"
    )
)


process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/ggiurgiu/mytest_select_pkam_only.root'),
    SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') )
)

process.p = cms.Path(process.skimming)
process.e = cms.EndPath(process.out)


