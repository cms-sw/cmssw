import FWCore.ParameterSet.Config as cms

ecalLaserCondTools = cms.EDAnalyzer("EcalLaserCondTools",
                                    #File with a list of events for whose IOVs must be filled in the database
                                    #Use empty string to include all the IOVs found in the input files 
                                    eventListFile = cms.string(""),

                                    #Module verbosity level
                                    verbosity = cms.int32(1),

                                    #Running mode:
                                    # ascii_file_to_db: fill the database with corrections read from an ascii file
                                    # hdf_file_to_db: fill the database with corrections read from a HDF5 file
                                    # db_to_ascii_file: export IOVS from the database to an ascii file, corr_dump.txt
                                    mode = cms.string(""),

                                    #List of input files for file-to-database modes
                                    inputFiles = cms.vstring(),

                                    #Numbre of first IOVs to skip when importing IOVs from a file
                                    skipIov = cms.int32(0),
                                    
                                    #Number of IOVs to insert in the database, use -1 to insert
                                    #all the IOVs found in the input files
                                    nIovs = cms.int32(-1),

                                    #Limit database IOV insertion for IOV after the
                                    #provided time, expressed in unix time
                                    fromTime = cms.int32(0),
                                    
                                    #If positive,  IOV insertion for IOV before the
                                    #provided time, expressed in unix time
                                    toTime = cms.int32(-1),
                                    
                                    #Force p1, p2, or p3 to 1 if its value
                                    #is lower than the provided bound value
                                    transparencyMin = cms.double(-1.),

                                    #Force p1, p2, or p3 to 1 if its value
                                    #is lower than the provided bound value
                                    transparencyMax = cms.double(9999.)
                                )


