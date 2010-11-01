from DPGAnalysis.Skims.CSCSkim_cfi import *
#set to minimum activity
cscSkim.minimumSegments = 1
cscSkim.minimumHitChambers = 1

# this is for filtering on HLT path
hltBeamHalo = cms.EDFilter("HLTHighLevel",
     TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
     HLTPaths = cms.vstring('HLT_CSCBeamHalo','HLT_CSCBeamHaloOverlapRing1','HLT_CSCBeamHaloOverlapRing','HLT_CSCBeamHaloRing2or3'), # provide list of HLT paths (or patterns) you want
     eventSetupPathsKey = cms.string(''), # not empty => use read paths from AlCaRecoTriggerBitsRcd via this key
     andOr = cms.bool(True),             # how to deal with multiple triggers: True (OR) accept if ANY is true, False (AND) accept if ALL are true
     throw = cms.bool(False)    # throw exception on unknown path names
 )

cscSkimAloneSeq = cms.Sequence(cscSkim)
cscHLTSkimSeq = cms.Sequence(hltBeamHalo)
cscSkimseq = cms.Sequence(hltBeamHalo+cscSkim)



