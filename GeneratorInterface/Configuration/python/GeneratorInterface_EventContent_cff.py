import FWCore.ParameterSet.Config as cms

#RAW content 
GeneratorInterfaceRAW = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',	# arbitrary labels, not just "source"?
        'keep LHEEventProduct_*_*_*',
        'keep GenRunInfoProduct_generator_*_*', 
        'keep GenEventInfoProduct_generator_*_*', 
        'keep edmHepMCProduct_generator_*_*', 
        'keep *_genParticles_*_*', 

# These products will go away!
        'keep edmGenInfoProduct_source_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*',
# and these too, because they will be run as producers/filters as "generator"
        'keep edmHepMCProduct_source_*_*', 
        'keep GenRunInfoProduct_source_*_*')
)

#RECO content
GeneratorInterfaceRECO = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',	# arbitrary labels, not just "source"?
        'keep LHEEventProduct_*_*_*',
        'keep GenRunInfoProduct_generator_*_*', 
        'keep GenEventInfoProduct_generator_*_*', 
        'keep edmHepMCProduct_generator_*_*', 
        'keep *_genParticles_*_*', 

# These products will go away!
        'keep edmGenInfoProduct_source_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*',
# and these too, because they will be run as producers/filters as "generator"
        'keep edmHepMCProduct_source_*_*', 
        'keep GenRunInfoProduct_source_*_*')
)

#AOD content
GeneratorInterfaceAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep LHERunInfoProduct_*_*_*',	# arbitrary labels, not just "source"?
        'keep LHEEventProduct_*_*_*',
        'keep GenRunInfoProduct_generator_*_*', 
        'keep GenEventInfoProduct_generator_*_*', 
        'keep *_genParticles_*_*', 

# These products will go away!
        'keep edmGenInfoProduct_source_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*',
# and these too, because they will be run as producers/filters as "generator"
        'keep GenRunInfoProduct_source_*_*')
)
