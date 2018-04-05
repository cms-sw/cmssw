import FWCore.ParameterSet.Config as cms

process = cms.Process("MagneticFieldXMLWriter")

process.load('Configuration.Geometry.MagneticFieldGeometry_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.BigXMLWriter = cms.EDAnalyzer("OutputMagneticFieldDDToDDL",
                              rotNumSeed = cms.int32(0),
                              fileName = cms.untracked.string("./mfSingleBigFile.xml")
                              )


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.BigXMLWriter)

