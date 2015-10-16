import FWCore.ParameterSet.Config as cms

process = cms.Process("L1MicroGMTEmulator")

process.load("FWCore.MessageService.MessageLogger_cfi")


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.dumper = cms.EDAnalyzer("l1t::MicroGMTLUTDumper",
    out_directory = cms.string("lut_dump"),
    AbsIsoCheckMemLUTSettings = cms.PSet (
        areaSum_in_width = cms.int32(5), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    IdxSelMemPhiLUTSettings = cms.PSet (
        phi_in_width = cms.int32(10), 
        out_width = cms.int32(6),
        filename = cms.string(""),
     ) ,
      
    FwdPosSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    BONegMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    OvlNegSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    IdxSelMemEtaLUTSettings = cms.PSet (
        eta_in_width = cms.int32(9), 
        out_width = cms.int32(5),
        filename = cms.string(""),
     ) ,
      
    FOPosMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    FwdNegSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    BPhiExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6), 
        pTred_in_width = cms.int32(6), 
        out_width = cms.int32(4),
        filename = cms.string(""),
     ) ,
      
    BrlSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    RelIsoCheckMemLUTSettings = cms.PSet (
        areaSum_in_width = cms.int32(5), 
        pT_in_width = cms.int32(9), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    OPhiExtrapolationLUTSettings = cms.PSet ( 
        etaAbsRed_in_width = cms.int32(6), 
        pTred_in_width = cms.int32(6), 
        out_width = cms.int32(4),
        filename = cms.string(""),
     ) ,
      
    OvlPosSingleMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    FEtaExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6), 
        pTred_in_width = cms.int32(6), 
        out_width = cms.int32(4),
        filename = cms.string(""),
     ) ,
      
    BOPosMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    OEtaExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6), 
        pTred_in_width = cms.int32(6), 
        out_width = cms.int32(4),
        filename = cms.string(""),
     ) ,
      
    BEtaExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6), 
        pTred_in_width = cms.int32(6), 
        out_width = cms.int32(4),
        filename = cms.string(""),
     ) ,
      
    FPhiExtrapolationLUTSettings = cms.PSet (
        etaAbsRed_in_width = cms.int32(6), 
        pTred_in_width = cms.int32(6), 
        out_width = cms.int32(4),
        filename = cms.string(""),
     ) ,
      
    FONegMatchQualLUTSettings = cms.PSet (
        deltaEtaRed_in_width = cms.int32(4), 
        deltaPhiRed_in_width = cms.int32(3), 
        out_width = cms.int32(1),
        filename = cms.string(""),
     ) ,
      
    SortRankLUTSettings = cms.PSet (
        pT_in_width = cms.int32(9), 
        qual_in_width = cms.int32(4), 
        out_width = cms.int32(10),
        filename = cms.string(""),
     )
)

process.dumpPath = cms.Path( process.dumper )
process.schedule = cms.Schedule(process.dumpPath)
