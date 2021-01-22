### NOTE: This is prepared to run on the newest PDFs with LHAPDF >=3.8.4
### so it requires local installation of LHAPDF libraries in order to run 
### out of the box. Otherwise, substitute the PDF sets by older sets

import FWCore.ParameterSet.Config as cms

# Process name
process = cms.Process("PDFANA")

# Max events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    #input = cms.untracked.int32(10)
)

# Printouts
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(100)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

# Input files (on disk)
process.source = cms.Source("PoolSource",
      fileNames = cms.untracked.vstring("file:/ciet3b/data3/Fall10_All_MinimalAOD/WmunuPLUS/WmunuPLUS_1.root")
)

# Produce PDF weights (maximum is 3)
process.pdfWeights = cms.EDProducer("PdfWeightProducer",
      # Fix POWHEG if buggy (this PDF set will also appear on output, 
      # so only two more PDF sets can be added in PdfSetNames if not "")
      FixPOWHEG = cms.untracked.string("CT10.LHgrid"),
      GenTag = cms.untracked.InputTag("prunedGenParticles"),
      PdfInfoTag = cms.untracked.InputTag("generator"),
      PdfSetNames = cms.untracked.vstring(
              "MSTW2008nlo68cl.LHgrid"
            , "NNPDF20_100.LHgrid"
      )
)

# Selector and parameters
process.goodMuonsForW = cms.EDFilter("MuonViewRefSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('isGlobalMuon=1 && isTrackerMuon=1 && abs(eta)<2.1 && abs(globalTrack().dxy)<0.2 && pt>20. && globalTrack().normalizedChi2<10 && globalTrack().hitPattern().numberOfValidTrackerHits>10 && globalTrack().hitPattern().numberOfValidMuonHits>0 && globalTrack().hitPattern().numberOfValidPixelHits>0 && numberOfMatches>1 && (isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)<0.15*pt'),
  filter = cms.bool(True)
)

# Collect uncertainties for rate and acceptance
process.pdfSystematics = cms.EDFilter("PdfSystematicsAnalyzer",
      SelectorPath = cms.untracked.string('pdfana'),
      PdfWeightTags = cms.untracked.VInputTag(
              "pdfWeights:CT10"
            , "pdfWeights:MSTW2008nlo68cl"
            , "pdfWeights:NNPDF20"
      )
)

# Main path
process.pdfana = cms.Path(
       process.pdfWeights
      *process.goodMuonsForW
)

process.end = cms.EndPath(process.pdfSystematics)
