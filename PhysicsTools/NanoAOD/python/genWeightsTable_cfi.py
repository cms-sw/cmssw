import FWCore.ParameterSet.Config as cms

genWeightsTable = cms.EDProducer("GenWeightsTableProducer",
    genEvent = cms.InputTag("generator"),
    genLumiInfoHeader = cms.InputTag("generator"),
    lheInfo = cms.VInputTag(cms.InputTag("externalLHEProducer"), cms.InputTag("source")),
    preferredPDFs = cms.VPSet( # see https://lhapdf.hepforge.org/pdfsets.html
        cms.PSet( name = cms.string("NNPDF31_nnlo_hessian_pdfas"), lhaid = cms.uint32(306000) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_as_0118_hessian"), lhaid = cms.uint32(304400) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_as_0118_mc_hessian_pdfas"), lhaid = cms.uint32(325300) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_as_0118_mc"), lhaid = cms.uint32(316200) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_as_0118_nf_4_mc_hessian"), lhaid = cms.uint32(325500) ),
        cms.PSet( name = cms.string("NNPDF31_nnlo_as_0118_nf_4"), lhaid = cms.uint32(320900) ),
        cms.PSet( name = cms.string("NNPDF30_nlo_as_0118"), lhaid = cms.uint32(260000) ), # for some 92X samples. Note that the nominal weight, 260000, is not included in the LHE ...
        cms.PSet( name = cms.string("NNPDF30_lo_as_0130"), lhaid = cms.uint32(262000) ), # some MLM 80X samples have only this (e.g. /store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v2/120000/02A210D6-F5C3-E611-B570-008CFA197BD4.root )
        cms.PSet( name = cms.string("NNPDF30_nlo_nf_4_pdfas"), lhaid = cms.uint32(292000) ), # some FXFX 80X samples have only this (e.g. WWTo1L1Nu2Q, WWTo4Q)
        cms.PSet( name = cms.string("NNPDF30_nlo_nf_5_pdfas"), lhaid = cms.uint32(292200) ), # some FXFX 80X samples have only this (e.g. DYJetsToLL_Pt, WJetsToLNu_Pt, DYJetsToNuNu_Pt)
        cms.PSet( name = cms.string("PDF4LHC15_nnlo_30_pdfas"), lhaid = cms.uint32(91400) ),
        cms.PSet( name = cms.string("PDF4LHC15_nlo_30_pdfas"), lhaid = cms.uint32(90400) ),
        cms.PSet( name = cms.string("PDF4LHC15_nlo_30"), lhaid = cms.uint32(90900) ),
    ),
    namedWeightIDs = cms.vstring(),
    namedWeightLabels = cms.vstring(),
    lheWeightPrecision = cms.int32(14),
    maxPdfWeights = cms.uint32(150),
    keepAllPSWeights = cms.bool(False),
    debug = cms.untracked.bool(False),
)
