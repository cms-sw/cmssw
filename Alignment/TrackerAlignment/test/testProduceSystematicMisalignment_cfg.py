#=================================
#inputs
globaltag = '74X_dataRun2_Prompt_v4'    #APEs are copied from this GT (and IdealGeometry is used)
inputsqlitefile = None                  #if None, uses the GT alignment
alignmenttag = 'Alignments'             #tag name for TrackerAlignmentRcd in the input file, also used for the output file
runnumberalignmentIOV = 1               #any run number in the iov that you want to start from

outputfilename = 'outputfile.db'


#misalignment amplitudes, -999 means no misalignment
#the commented numbers are the default magnitudes, which produce a maximum movement of around 600 microns
#see Alignment/TrackerAlignment/plugins/TrackerSystematicMisalignments.cc for definitions
#see also https://twiki.cern.ch/twiki/bin/viewauth/CMS/SystematicMisalignmentsofTracker
radialEpsilon     = -999. # 5e-4
telescopeEpsilon  = -999. # 5e-4
layerRotEpsilon   = -999. # 9.43e-6               #cm^-1
bowingEpsilon     = -999. # 6.77e-9               #cm^-2
zExpEpsilon       = -999. # 2.02e-4
twistEpsilon      = -999. # 2.04e-6               #cm^-1
ellipticalEpsilon = -999. # 5e-4
skewEpsilon       = -999. # 5.5e-2                #cm
sagittaEpsilon    = -999. # 5.0e-4

#phases for phi dependent misalignments
ellipticalDelta   = 0.
skewDelta         = 0.
sagittaDelta      = 0.
#=================================




import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackerSystematicMisalignments")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Geometry.GeometryRecoDB_cff")

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.source = cms.Source("EmptySource",
                            firstRun=cms.untracked.uint32(runnumberalignmentIOV),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# initial geom
# configure the database file - use survey one for default
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.GlobalTag.globaltag=globaltag
if inputsqlitefile is not None:
    process.GlobalTag.toGet = cms.VPSet(
                                        cms.PSet(
                                                 record = cms.string('TrackerAlignmentRcd'),
                                                 tag = cms.string(alignmenttag),
                                                 connect = cms.untracked.string('sqlite_file:'+inputsqlitefile),
                                        ),
    )


# input
process.load("Alignment.TrackerAlignment.TrackerSystematicMisalignments_cfi")
process.TrackerSystematicMisalignments.fromDBGeom = True

#uncomment one or more of these to apply the misalignment(s)

process.TrackerSystematicMisalignments.radialEpsilon     = radialEpsilon
process.TrackerSystematicMisalignments.telescopeEpsilon  = telescopeEpsilon
process.TrackerSystematicMisalignments.layerRotEpsilon   = layerRotEpsilon
process.TrackerSystematicMisalignments.bowingEpsilon     = bowingEpsilon
process.TrackerSystematicMisalignments.zExpEpsilon       = zExpEpsilon
process.TrackerSystematicMisalignments.twistEpsilon      = twistEpsilon
process.TrackerSystematicMisalignments.ellipticalEpsilon = ellipticalEpsilon
process.TrackerSystematicMisalignments.skewEpsilon       = skewEpsilon
process.TrackerSystematicMisalignments.sagittaEpsilon    = sagittaEpsilon

#misalignment phases
process.TrackerSystematicMisalignments.ellipticalDelta   = ellipticalDelta
process.TrackerSystematicMisalignments.skewDelta         = skewDelta
process.TrackerSystematicMisalignments.sagittaDelta      = sagittaDelta

# output
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          toPut = cms.VPSet(
                                                            cms.PSet(
                                                                     record = cms.string('TrackerAlignmentRcd'),
                                                                     tag = cms.string(alignmenttag),
                                                            ),
                                                            cms.PSet(
                                                                     record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                                                     tag = cms.string('AlignmentErrorsExtended'),
                                                            ),
                                          ),
                                          connect = cms.string('sqlite_file:'+outputfilename),
)

process.p = cms.Path( process.TrackerSystematicMisalignments )
