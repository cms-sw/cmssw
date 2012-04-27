import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD2")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    
# For valgrind studies
# process.ProfilerService = cms.Service("ProfilerService",
#    lastEvent = cms.untracked.int32(13),
#    firstEvent = cms.untracked.int32(3),
#    paths = cms.untracked.vstring('p1')
#)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring(
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_1.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_10.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_11.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_12.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_13.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_14.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_15.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_16.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_17.root',
#    'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/ZS/reco_Neutrino_Full_18.root'),
#
#                            noEventSort = cms.untracked.bool(True),
#                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
#                            )
#

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'file:MyFirstFamosFile_2.root'),    
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)    



# With ZS
#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring(
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_1.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_10.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_11.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_12.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_13.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_14.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_15.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_16.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_17.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_18.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_19.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_2.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_20.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_21.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_22.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_23.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_24.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_25.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_26.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_27.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_28.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_29.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_3.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_30.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_31.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_32.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_33.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_34.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_35.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_36.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_37.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_38.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_39.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_4.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_40.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_41.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_42.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_43.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_44.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_45.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_46.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_47.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_48.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_49.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_5.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_50.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_51.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_52.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_53.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_54.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_55.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_56.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_57.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_58.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_59.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_6.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_60.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_61.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_62.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_63.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_64.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_65.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_66.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_67.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_68.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_69.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_7.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_70.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_71.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_72.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_73.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_74.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_75.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_76.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_77.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_78.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_79.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_8.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS/reco_Neutrino_Full_9.root'),
#                            noEventSort = cms.untracked.bool(True),
#                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
#                            )

# ZS online ? 
#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring(
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_1.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_10.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_11.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_12.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_13.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_14.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_15.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_16.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_17.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_18.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_19.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_2.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_20.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_21.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_22.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_23.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_24.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_25.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_26.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_27.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_28.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_29.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_3.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_30.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_31.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_32.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_33.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_34.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_35.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_36.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_37.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_38.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_39.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_4.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_40.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_41.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_42.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_43.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_44.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_45.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_46.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_47.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_48.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_49.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_5.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_50.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_51.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_52.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_53.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_54.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_55.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_56.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_57.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_58.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_59.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_6.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_60.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_61.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_62.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_63.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_64.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_65.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_66.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_67.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_68.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_69.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_7.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_70.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_71.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_72.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_73.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_74.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_75.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_76.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_77.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_78.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_79.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_8.root',
#'rfio:/castor/cern.ch/user/b/beaudett/CMSSW_4_2_0_pre7/Neutrinos/ZS-online/reco_Neutrino_Full_9.root'),
#                            noEventSort = cms.untracked.bool(True),
#                            duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
#                            )



# To make histograms
process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']


# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

process.noiseCheck = cms.EDAnalyzer("NoiseCheck",
                                  OutputFile=cms.string('Noisecheck-Neutrino-fast.root'),
                                  Threshold=cms.double(0.318))

# Produce Tracks and Clusters
#process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithCaloHits*process.noiseCheck)
process.p1 = cms.Path(process.noiseCheck)


# Keep the logging output to a nice level #

#process.Timing =  cms.Service("Timing")
#process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#process.MessageLogger.categories.append("FamosManager")
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
