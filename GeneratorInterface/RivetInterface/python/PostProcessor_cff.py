import FWCore.ParameterSet.Config as cms

###################
#CMS_2011_S8968497
###################
postCMS_2011_S8968497 = cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S8968497"),
    efficiency = cms.vstring(""),
    resolution = cms.vstring(""),
    normalizationToIntegral = cms.untracked.vstring("d01-x01-y01",
                                          "d02-x01-y01",
                                          "d03-x01-y01",
                                          "d04-x01-y01",
                                          "d05-x01-y01",
                                          "d06-x01-y01",
                                          "d07-x01-y01",
                                          "d08-x01-y01",
                                          "d09-x01-y01")
)    

###################
#CMS_2010_S8547297
###################
postCMS_2010_S8547297= cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2010_S8547297"),
    scaleBy = cms.untracked.vstring(
                                          "d01-x01-y01 2.5",
                                          "d01-x01-y02 2.5",
                                          "d01-x01-y03 2.5",
                                          "d01-x01-y04 2.5",

                                          "d02-x01-y01 2.5",
                                          "d02-x01-y02 2.5",
                                          "d02-x01-y03 2.5",
                                          "d02-x01-y04 2.5",

                                          "d03-x01-y01 2.5",
                                          "d03-x01-y02 2.5",
                                          "d03-x01-y03 2.5",
                                          "d03-x01-y04 2.5",

                                          "d04-x01-y01 2.5",
                                          "d04-x01-y02 2.5",
                                          "d04-x01-y03 2.5",
                                          "d04-x01-y04 2.5",

                                          "d05-x01-y01 2.5",
                                          "d05-x01-y02 2.5",
                                          "d05-x01-y03 2.5",
                                          "d05-x01-y04 2.5",

                                          "d06-x01-y01 2.5",
                                          "d06-x01-y02 2.5",
                                          "d06-x01-y03 2.5",
                                          "d06-x01-y04 2.5",
                                          
                                          "d07-x01-y01 0.0331572798065",
                                          "d07-x01-y02 0.0331572798065"

                                           
    ),
    normalizationToIntegral = cms.untracked.vstring(
                                          "d01-x01-y01 nEvt",
                                          "d01-x01-y02 nEvt",
                                          "d01-x01-y03 nEvt",
                                          "d01-x01-y04 nEvt",

                                          "d02-x01-y01 nEvt",
                                          "d02-x01-y02 nEvt",
                                          "d02-x01-y03 nEvt",
                                          "d02-x01-y04 nEvt",

                                          "d03-x01-y01 nEvt",
                                          "d03-x01-y02 nEvt",
                                          "d03-x01-y03 nEvt",
                                          "d03-x01-y04 nEvt",

                                          "d04-x01-y01 nEvt",
                                          "d04-x01-y02 nEvt",
                                          "d04-x01-y03 nEvt",
                                          "d04-x01-y04 nEvt",

                                          "d05-x01-y01 nEvt",
                                          "d05-x01-y02 nEvt",
                                          "d05-x01-y03 nEvt",
                                          "d05-x01-y04 nEvt",

                                          "d06-x01-y01 nEvt",
                                          "d06-x01-y02 nEvt",
                                          "d06-x01-y03 nEvt",
                                          "d06-x01-y04 nEvt",
                                          
                                          "d07-x01-y01 nEvt",
                                          "d07-x01-y02 nEvt",

                                          "d08-x01-y01 nEvt",
                                          "d08-x01-y02 nEvt"
                                          )
)

###################
#CMS_2010_S8656010
###################
postCMS_2010_S8656010= cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2010_S8656010"),
    scaleBy = cms.untracked.vstring(
                                          "d01-x01-y01 2.5",
                                          "d01-x01-y02 2.5",
                                          "d01-x01-y03 2.5",
                                          "d01-x01-y04 2.5",

                                          "d02-x01-y01 2.5",
                                          "d02-x01-y02 2.5",
                                          "d02-x01-y03 2.5",
                                          "d02-x01-y04 2.5",

                                          "d03-x01-y01 2.5",
                                          "d03-x01-y02 2.5",
                                          "d03-x01-y03 2.5",
                                          "d03-x01-y04 2.5",

                                          "d04-x01-y01 2.5",
                                          "d04-x01-y02 2.5",
                                          "d04-x01-y03 2.5",
                                          "d04-x01-y04 2.5",

                                          "d05-x01-y01 2.5",
                                          "d05-x01-y02 2.5",
                                          "d05-x01-y03 2.5",
                                          "d05-x01-y04 2.5",

                                          "d06-x01-y01 2.5",
                                          "d06-x01-y02 2.5",
                                          "d06-x01-y03 2.5",
                                          "d06-x01-y04 2.5",

                                          "d07-x01-y01 0.0331572798065",
                                          "d07-x01-y02 0.0331572798065"


    ),
    normalizationToIntegral = cms.untracked.vstring(
                                          "d01-x01-y01 nEvt",
                                          "d01-x01-y02 nEvt",
                                          "d01-x01-y03 nEvt",
                                          "d01-x01-y04 nEvt",

                                          "d02-x01-y01 nEvt",
                                          "d02-x01-y02 nEvt",
                                          "d02-x01-y03 nEvt",
                                          "d02-x01-y04 nEvt",

                                          "d03-x01-y01 nEvt",
                                          "d03-x01-y02 nEvt",
                                          "d03-x01-y03 nEvt",
                                          "d03-x01-y04 nEvt",

                                          "d04-x01-y01 nEvt",
                                          "d04-x01-y02 nEvt",
                                          "d04-x01-y03 nEvt",
                                          "d04-x01-y04 nEvt",

                                          "d05-x01-y01 nEvt",
                                          "d05-x01-y02 nEvt",
                                          "d05-x01-y03 nEvt",
                                          "d05-x01-y04 nEvt",

                                          "d06-x01-y01 nEvt",
                                          "d06-x01-y02 nEvt",
                                          "d06-x01-y03 nEvt",
                                          "d06-x01-y04 nEvt",

                                          "d07-x01-y01 nEvt",
                                          "d07-x01-y02 nEvt",

                                          "d08-x01-y01 nEvt",
                                          "d08-x01-y02 nEvt"
                                          )
)

###################
#CMS_2011_S8884919
###################
postCMS_2011_S8884919 = cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S8884919"),
    normalizeToIntegral = cms.untracked.vstring(
                                          "d01-x01-y01",
                                          "d02-x01-y01",
                                          "d03-x01-y01",
                                          "d04-x01-y01",
                                          "d05-x01-y01",
                                          "d06-x01-y01",
                                          "d07-x01-y01",
                                          "d08-x01-y01",
                                          "d09-x01-y01",
                                          "d10-x01-y01",
                                          "d11-x01-y01",
                                          "d12-x01-y01",
                                          "d13-x01-y01",
                                          "d14-x01-y01",
                                          "d15-x01-y01",
                                          "d16-x01-y01",
                                          "d17-x01-y01",
                                          "d17-x01-y02",
                                          "d18-x01-y01",
                                          "d18-x01-y02",
                                          "d19-x01-y01",
                                          "d19-x01-y02",
                                          "d20-x01-y01",
                                          "d21-x01-y01",
                                          "d22-x01-y01",
                                          "d23-x01-y01",
                                          "d24-x01-y01",
                                          "d25-x01-y01"
    )
)    

###################
#CMS_2011_S8941262
###################
postCMS_2011_S8941262 = cms.EDAnalyzer("DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S8941262"),
    xsection = cms.untracked.double(1000),
    normalizeToLumi = cms.untracked.vstring(
                                         "d01-x01-y01",
                                         "d02-x01-y01",
                                         "d03-x01-y01"
    ),
    scaleBy = cms.untracked.vstring(
                                         "d01-x01-y01 0.000001", #norm from picobarn to microbarn
                                         "d02-x01-y01 0.001",    #norm from picobarn to nanobarn
                                         "d03-x01-y01 0.001"     #norm from picobarn to nanobarn  
    )
)    

###################
#CMS_2011_S8950903
###################
postCMS_2011_S8950903 = cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S8950903"),
    normalizeToIntegral = cms.untracked.vstring(
                                         "d01-x01-y01"
    )
) 

###################
#CMS_2011_S8957746
###################
postCMS_2011_S8957746 = cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S8957746"),
    normalizeToIntegral = cms.untracked.vstring(
                                         "d01-x01-y01",
                                         "d01-x02-y01",
                                         "d01-x03-y01",
                                         "d02-x01-y01",
                                         "d02-x02-y01",
                                         "d02-x03-y01"
    )
)

###################
#CMS_2011_S8968497
###################
postCMS_2011_S8968497 = cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S8968497"),
    normalizeToIntegral = cms.untracked.vstring(
                                         "d01-x01-y01"
    )
)    

###################
#CMS_2011_S9086218 
###################
postCMS_2011_S9086218 = cms.EDAnalyzer(
    "DQMRivetClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S9086218"),
    xsection = cms.untracked.double(1000),
    normalizeToLumi = cms.untracked.vstring(
                                         "d01-x01-y01 nEvt" 
    ),
    scaleBy = cms.untracked.vstring(
                                         "d01-x01-y01 0.5"  
    )
)    

###################
#CMS_2011_S9088458
###################
postCMS_2011_S9088458 = cms.EDAnalyzer(
    "DQMGenericClient",
    subDirs = cms.untracked.vstring("Rivet/CMS_2011_S9088458"),
    efficiencyProfile = cms.untracked.vstring("d01-x01-y01 d01-x01-y01 trijet dijet"),
    resolution = cms.vstring(""),
    efficiency = cms.vstring("")
)    

####SEQUENCES
RivetDQMPostProcessor = cms.Sequence( postCMS_2011_S8968497 + 
                                      postCMS_2010_S8547297 + 
                                      postCMS_2010_S8656010 +
                                      postCMS_2011_S8884919 +
                                      postCMS_2011_S8941262 + 
                                      postCMS_2011_S8950903 + 
                                      postCMS_2011_S8957746 + 
                                      postCMS_2011_S8968497 +
                                      postCMS_2011_S9086218 +
                                      postCMS_2011_S9088458 )

