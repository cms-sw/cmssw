import FWCore.ParameterSet.Config as cms 
process = cms.Process('jecdb') 
process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:JEC_Summer09_7TeV_ReReco332.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet( 
         record = cms.string('L2Relative_IC5Calo'), 
         tag    = cms.string('L2Relative_IC5Calo'), 
         label  = cms.string('L2Relative_IC5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_IC5PF'), 
         tag    = cms.string('L2Relative_IC5PF'), 
         label  = cms.string('L2Relative_IC5PF') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_AK5JPT'), 
         tag    = cms.string('L2Relative_AK5JPT'), 
         label  = cms.string('L2Relative_AK5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_AK5Calo'), 
         tag    = cms.string('L2Relative_AK5Calo'), 
         label  = cms.string('L2Relative_AK5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_AK5PF'), 
         tag    = cms.string('L2Relative_AK5PF'), 
         label  = cms.string('L2Relative_AK5PF') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_AK5TRK'), 
         tag    = cms.string('L2Relative_AK5TRK'), 
         label  = cms.string('L2Relative_AK5TRK') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_AK7Calo'), 
         tag    = cms.string('L2Relative_AK7Calo'), 
         label  = cms.string('L2Relative_AK7Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_AK7PF'), 
         tag    = cms.string('L2Relative_AK7PF'), 
         label  = cms.string('L2Relative_AK7PF') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_KT4Calo'), 
         tag    = cms.string('L2Relative_KT4Calo'), 
         label  = cms.string('L2Relative_KT4Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_KT4PF'), 
         tag    = cms.string('L2Relative_KT4PF'), 
         label  = cms.string('L2Relative_KT4PF') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_KT6Calo'), 
         tag    = cms.string('L2Relative_KT6Calo'), 
         label  = cms.string('L2Relative_KT6Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L2Relative_KT6PF'), 
         tag    = cms.string('L2Relative_KT6PF'), 
         label  = cms.string('L2Relative_KT6PF') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_IC5Calo'), 
         tag    = cms.string('L3Absolute_IC5Calo'), 
         label  = cms.string('L3Absolute_IC5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_IC5PF'), 
         tag    = cms.string('L3Absolute_IC5PF'), 
         label  = cms.string('L3Absolute_IC5PF') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_AK5JPT'), 
         tag    = cms.string('L3Absolute_AK5JPT'), 
         label  = cms.string('L3Absolute_AK5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_AK5Calo'), 
         tag    = cms.string('L3Absolute_AK5Calo'), 
         label  = cms.string('L3Absolute_AK5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_AK5PF'), 
         tag    = cms.string('L3Absolute_AK5PF'), 
         label  = cms.string('L3Absolute_AK5PF') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_AK5TRK'), 
         tag    = cms.string('L3Absolute_AK5TRK'), 
         label  = cms.string('L3Absolute_AK5TRK') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_AK7Calo'), 
         tag    = cms.string('L3Absolute_AK7Calo'), 
         label  = cms.string('L3Absolute_AK7Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_AK7PF'), 
         tag    = cms.string('L3Absolute_AK7PF'), 
         label  = cms.string('L3Absolute_AK7PF') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_KT4Calo'), 
         tag    = cms.string('L3Absolute_KT4Calo'), 
         label  = cms.string('L3Absolute_KT4Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_KT4PF'), 
         tag    = cms.string('L3Absolute_KT4PF'), 
         label  = cms.string('L3Absolute_KT4PF') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_KT6Calo'), 
         tag    = cms.string('L3Absolute_KT6Calo'), 
         label  = cms.string('L3Absolute_KT6Calo') 
      ), 
      cms.PSet( 
         record = cms.string('L3Absolute_KT6PF'), 
         tag    = cms.string('L3Absolute_KT6PF'), 
         label  = cms.string('L3Absolute_KT6PF') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_bJ'), 
         tag    = cms.string('L5Flavor_IC5Calo_bJ'), 
         label  = cms.string('L5Flavor_IC5Calo_bJ') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_cJ'), 
         tag    = cms.string('L5Flavor_IC5Calo_cJ'), 
         label  = cms.string('L5Flavor_IC5Calo_cJ') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_qJ'), 
         tag    = cms.string('L5Flavor_IC5Calo_qJ'), 
         label  = cms.string('L5Flavor_IC5Calo_qJ') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_gJ'), 
         tag    = cms.string('L5Flavor_IC5Calo_gJ'), 
         label  = cms.string('L5Flavor_IC5Calo_gJ') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_bT'), 
         tag    = cms.string('L5Flavor_IC5Calo_bT'), 
         label  = cms.string('L5Flavor_IC5Calo_bT') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_cT'), 
         tag    = cms.string('L5Flavor_IC5Calo_cT'), 
         label  = cms.string('L5Flavor_IC5Calo_cT') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_qT'), 
         tag    = cms.string('L5Flavor_IC5Calo_qT'), 
         label  = cms.string('L5Flavor_IC5Calo_qT') 
      ), 
      cms.PSet( 
         record = cms.string('L5Flavor_IC5Calo_gT'), 
         tag    = cms.string('L5Flavor_IC5Calo_gT'), 
         label  = cms.string('L5Flavor_IC5Calo_gT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_bJ'), 
         tag    = cms.string('L7Parton_IC5_bJ'), 
         label  = cms.string('L7Parton_IC5_bJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_bJ'), 
         tag    = cms.string('L7Parton_AK5_bJ'), 
         label  = cms.string('L7Parton_AK5_bJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_bJ'), 
         tag    = cms.string('L7Parton_AK7_bJ'), 
         label  = cms.string('L7Parton_AK7_bJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_bJ'), 
         tag    = cms.string('L7Parton_KT4_bJ'), 
         label  = cms.string('L7Parton_KT4_bJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_bJ'), 
         tag    = cms.string('L7Parton_KT6_bJ'), 
         label  = cms.string('L7Parton_KT6_bJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_cJ'), 
         tag    = cms.string('L7Parton_IC5_cJ'), 
         label  = cms.string('L7Parton_IC5_cJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_cJ'), 
         tag    = cms.string('L7Parton_AK5_cJ'), 
         label  = cms.string('L7Parton_AK5_cJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_cJ'), 
         tag    = cms.string('L7Parton_AK7_cJ'), 
         label  = cms.string('L7Parton_AK7_cJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_cJ'), 
         tag    = cms.string('L7Parton_KT4_cJ'), 
         label  = cms.string('L7Parton_KT4_cJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_cJ'), 
         tag    = cms.string('L7Parton_KT6_cJ'), 
         label  = cms.string('L7Parton_KT6_cJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_qJ'), 
         tag    = cms.string('L7Parton_IC5_qJ'), 
         label  = cms.string('L7Parton_IC5_qJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_qJ'), 
         tag    = cms.string('L7Parton_AK5_qJ'), 
         label  = cms.string('L7Parton_AK5_qJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_qJ'), 
         tag    = cms.string('L7Parton_AK7_qJ'), 
         label  = cms.string('L7Parton_AK7_qJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_qJ'), 
         tag    = cms.string('L7Parton_KT4_qJ'), 
         label  = cms.string('L7Parton_KT4_qJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_qJ'), 
         tag    = cms.string('L7Parton_KT6_qJ'), 
         label  = cms.string('L7Parton_KT6_qJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_gJ'), 
         tag    = cms.string('L7Parton_IC5_gJ'), 
         label  = cms.string('L7Parton_IC5_gJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_gJ'), 
         tag    = cms.string('L7Parton_AK5_gJ'), 
         label  = cms.string('L7Parton_AK5_gJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_gJ'), 
         tag    = cms.string('L7Parton_AK7_gJ'), 
         label  = cms.string('L7Parton_AK7_gJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_gJ'), 
         tag    = cms.string('L7Parton_KT4_gJ'), 
         label  = cms.string('L7Parton_KT4_gJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_gJ'), 
         tag    = cms.string('L7Parton_KT6_gJ'), 
         label  = cms.string('L7Parton_KT6_gJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_jJ'), 
         tag    = cms.string('L7Parton_IC5_jJ'), 
         label  = cms.string('L7Parton_IC5_jJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_jJ'), 
         tag    = cms.string('L7Parton_AK5_jJ'), 
         label  = cms.string('L7Parton_AK5_jJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_jJ'), 
         tag    = cms.string('L7Parton_AK7_jJ'), 
         label  = cms.string('L7Parton_AK7_jJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_jJ'), 
         tag    = cms.string('L7Parton_KT4_jJ'), 
         label  = cms.string('L7Parton_KT4_jJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_jJ'), 
         tag    = cms.string('L7Parton_KT6_jJ'), 
         label  = cms.string('L7Parton_KT6_jJ') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_bT'), 
         tag    = cms.string('L7Parton_IC5_bT'), 
         label  = cms.string('L7Parton_IC5_bT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_bT'), 
         tag    = cms.string('L7Parton_AK5_bT'), 
         label  = cms.string('L7Parton_AK5_bT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_bT'), 
         tag    = cms.string('L7Parton_AK7_bT'), 
         label  = cms.string('L7Parton_AK7_bT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_bT'), 
         tag    = cms.string('L7Parton_KT4_bT'), 
         label  = cms.string('L7Parton_KT4_bT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_bT'), 
         tag    = cms.string('L7Parton_KT6_bT'), 
         label  = cms.string('L7Parton_KT6_bT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_cT'), 
         tag    = cms.string('L7Parton_IC5_cT'), 
         label  = cms.string('L7Parton_IC5_cT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_cT'), 
         tag    = cms.string('L7Parton_AK5_cT'), 
         label  = cms.string('L7Parton_AK5_cT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_cT'), 
         tag    = cms.string('L7Parton_AK7_cT'), 
         label  = cms.string('L7Parton_AK7_cT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_cT'), 
         tag    = cms.string('L7Parton_KT4_cT'), 
         label  = cms.string('L7Parton_KT4_cT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_cT'), 
         tag    = cms.string('L7Parton_KT6_cT'), 
         label  = cms.string('L7Parton_KT6_cT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_qT'), 
         tag    = cms.string('L7Parton_IC5_qT'), 
         label  = cms.string('L7Parton_IC5_qT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_qT'), 
         tag    = cms.string('L7Parton_AK5_qT'), 
         label  = cms.string('L7Parton_AK5_qT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_qT'), 
         tag    = cms.string('L7Parton_AK7_qT'), 
         label  = cms.string('L7Parton_AK7_qT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_qT'), 
         tag    = cms.string('L7Parton_KT4_qT'), 
         label  = cms.string('L7Parton_KT4_qT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_qT'), 
         tag    = cms.string('L7Parton_KT6_qT'), 
         label  = cms.string('L7Parton_KT6_qT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_IC5_tT'), 
         tag    = cms.string('L7Parton_IC5_tT'), 
         label  = cms.string('L7Parton_IC5_tT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK5_tT'), 
         tag    = cms.string('L7Parton_AK5_tT'), 
         label  = cms.string('L7Parton_AK5_tT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_AK7_tT'), 
         tag    = cms.string('L7Parton_AK7_tT'), 
         label  = cms.string('L7Parton_AK7_tT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT4_tT'), 
         tag    = cms.string('L7Parton_KT4_tT'), 
         label  = cms.string('L7Parton_KT4_tT') 
      ), 
      cms.PSet( 
         record = cms.string('L7Parton_KT6_tT'), 
         tag    = cms.string('L7Parton_KT6_tT'), 
         label  = cms.string('L7Parton_KT6_tT') 
      ), 
      cms.PSet( 
         record = cms.string('L4EMF_AK5Calo'), 
         tag    = cms.string('L4EMF_AK5Calo'), 
         label  = cms.string('L4EMF_AK5Calo') 
      ) 
   ) 
) 
process.dbWriterL2RelativeIC5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_IC5Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_IC5Calo') 
) 
process.dbWriterL2RelativeIC5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_IC5PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_IC5PF') 
) 
process.dbWriterL2RelativeAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_AK5JPT.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_AK5JPT') 
) 
process.dbWriterL2RelativeAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_AK5Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_AK5Calo') 
) 
process.dbWriterL2RelativeAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_AK5PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_AK5PF') 
) 
process.dbWriterL2RelativeAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_AK5TRK.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_AK5TRK') 
) 
process.dbWriterL2RelativeAK7Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_AK7Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_AK7Calo') 
) 
process.dbWriterL2RelativeAK7PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_AK7PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_AK7PF') 
) 
process.dbWriterL2RelativeKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_KT4Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_KT4Calo') 
) 
process.dbWriterL2RelativeKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_KT4PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_KT4PF') 
) 
process.dbWriterL2RelativeKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_KT6Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_KT6Calo') 
) 
process.dbWriterL2RelativeKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L2Relative_KT6PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L2Relative_KT6PF') 
) 
process.dbWriterL3AbsoluteIC5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_IC5Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_IC5Calo') 
) 
process.dbWriterL3AbsoluteIC5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_IC5PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_IC5PF') 
) 
process.dbWriterL3AbsoluteAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_AK5JPT.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_AK5JPT') 
) 
process.dbWriterL3AbsoluteAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_AK5Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_AK5Calo') 
) 
process.dbWriterL3AbsoluteAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_AK5PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_AK5PF') 
) 
process.dbWriterL3AbsoluteAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_AK5TRK.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_AK5TRK') 
) 
process.dbWriterL3AbsoluteAK7Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_AK7Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_AK7Calo') 
) 
process.dbWriterL3AbsoluteAK7PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_AK7PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_AK7PF') 
) 
process.dbWriterL3AbsoluteKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_KT4Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_KT4Calo') 
) 
process.dbWriterL3AbsoluteKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_KT4PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_KT4PF') 
) 
process.dbWriterL3AbsoluteKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_KT6Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_KT6Calo') 
) 
process.dbWriterL3AbsoluteKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Summer09_7TeV_ReReco332_L3Absolute_KT6PF.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L3Absolute_KT6PF') 
) 
process.dbWriterL5bJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('bJ'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5cJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('cJ'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5qJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('qJ'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5gJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('gJ'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5bT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('bT'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5cT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('cT'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5qT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('qT'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL5gT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L5Flavor_IC5Calo.txt'), 
   option       = cms.untracked.string('gT'), 
   label        = cms.untracked.string('L5Flavor_IC5Calo') 
) 
process.dbWriterL7IC5bJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('bJ'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5bJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('bJ'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7bJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('bJ'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4bJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('bJ'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6bJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('bJ'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5cJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('cJ'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5cJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('cJ'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7cJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('cJ'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4cJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('cJ'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6cJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('cJ'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5qJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('qJ'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5qJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('qJ'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7qJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('qJ'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4qJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('qJ'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6qJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('qJ'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5gJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('gJ'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5gJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('gJ'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7gJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('gJ'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4gJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('gJ'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6gJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('gJ'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5jJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('jJ'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5jJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('jJ'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7jJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('jJ'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4jJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('jJ'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6jJ = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('jJ'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5bT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('bT'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5bT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('bT'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7bT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('bT'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4bT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('bT'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6bT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('bT'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5cT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('cT'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5cT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('cT'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7cT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('cT'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4cT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('cT'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6cT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('cT'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5qT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('qT'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5qT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('qT'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7qT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('qT'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4qT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('qT'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6qT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('qT'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL7IC5tT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_IC5.txt'), 
   option       = cms.untracked.string('tT'), 
   label        = cms.untracked.string('L7Parton_IC5') 
) 
process.dbWriterL7AK5tT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK5.txt'), 
   option       = cms.untracked.string('tT'), 
   label        = cms.untracked.string('L7Parton_AK5') 
) 
process.dbWriterL7AK7tT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_AK7.txt'), 
   option       = cms.untracked.string('tT'), 
   label        = cms.untracked.string('L7Parton_AK7') 
) 
process.dbWriterL7KT4tT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT4.txt'), 
   option       = cms.untracked.string('tT'), 
   label        = cms.untracked.string('L7Parton_KT4') 
) 
process.dbWriterL7KT6tT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L7Parton_KT6.txt'), 
   option       = cms.untracked.string('tT'), 
   label        = cms.untracked.string('L7Parton_KT6') 
) 
process.dbWriterL4 = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('L4EMF_AK5Calo.txt'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('L4EMF_AK5Calo') 
) 
process.p = cms.Path( 
process.dbWriterL2RelativeIC5Calo * 
process.dbWriterL2RelativeIC5PF * 
process.dbWriterL2RelativeAK5JPT * 
process.dbWriterL2RelativeAK5Calo * 
process.dbWriterL2RelativeAK5PF * 
process.dbWriterL2RelativeAK5TRK * 
process.dbWriterL2RelativeAK7Calo * 
process.dbWriterL2RelativeAK7PF * 
process.dbWriterL2RelativeKT4Calo * 
process.dbWriterL2RelativeKT4PF * 
process.dbWriterL2RelativeKT6Calo * 
process.dbWriterL2RelativeKT6PF * 
process.dbWriterL3AbsoluteIC5Calo * 
process.dbWriterL3AbsoluteIC5PF * 
process.dbWriterL3AbsoluteAK5JPT * 
process.dbWriterL3AbsoluteAK5Calo * 
process.dbWriterL3AbsoluteAK5PF * 
process.dbWriterL3AbsoluteAK5TRK * 
process.dbWriterL3AbsoluteAK7Calo * 
process.dbWriterL3AbsoluteAK7PF * 
process.dbWriterL3AbsoluteKT4Calo * 
process.dbWriterL3AbsoluteKT4PF * 
process.dbWriterL3AbsoluteKT6Calo * 
process.dbWriterL3AbsoluteKT6PF * 
process.dbWriterL5bJ * 
process.dbWriterL5cJ * 
process.dbWriterL5qJ * 
process.dbWriterL5gJ * 
process.dbWriterL5bT * 
process.dbWriterL5cT * 
process.dbWriterL5qT * 
process.dbWriterL5gT * 
process.dbWriterL7IC5bJ * 
process.dbWriterL7AK5bJ * 
process.dbWriterL7AK7bJ * 
process.dbWriterL7KT4bJ * 
process.dbWriterL7KT6bJ * 
process.dbWriterL7IC5cJ * 
process.dbWriterL7AK5cJ * 
process.dbWriterL7AK7cJ * 
process.dbWriterL7KT4cJ * 
process.dbWriterL7KT6cJ * 
process.dbWriterL7IC5qJ * 
process.dbWriterL7AK5qJ * 
process.dbWriterL7AK7qJ * 
process.dbWriterL7KT4qJ * 
process.dbWriterL7KT6qJ * 
process.dbWriterL7IC5gJ * 
process.dbWriterL7AK5gJ * 
process.dbWriterL7AK7gJ * 
process.dbWriterL7KT4gJ * 
process.dbWriterL7KT6gJ * 
process.dbWriterL7IC5jJ * 
process.dbWriterL7AK5jJ * 
process.dbWriterL7AK7jJ * 
process.dbWriterL7KT4jJ * 
process.dbWriterL7KT6jJ * 
process.dbWriterL7IC5bT * 
process.dbWriterL7AK5bT * 
process.dbWriterL7AK7bT * 
process.dbWriterL7KT4bT * 
process.dbWriterL7KT6bT * 
process.dbWriterL7IC5cT * 
process.dbWriterL7AK5cT * 
process.dbWriterL7AK7cT * 
process.dbWriterL7KT4cT * 
process.dbWriterL7KT6cT * 
process.dbWriterL7IC5qT * 
process.dbWriterL7AK5qT * 
process.dbWriterL7AK7qT * 
process.dbWriterL7KT4qT * 
process.dbWriterL7KT6qT * 
process.dbWriterL7IC5tT * 
process.dbWriterL7AK5tT * 
process.dbWriterL7AK7tT * 
process.dbWriterL7KT4tT * 
process.dbWriterL7KT6tT * 
process.dbWriterL4
) 
