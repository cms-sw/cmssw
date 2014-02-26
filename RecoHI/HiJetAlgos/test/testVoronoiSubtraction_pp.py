import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
#        "/store/relval/CMSSW_6_2_0_pre8/RelValTTbar/GEN-SIM-RECO/PU_PRE_ST62_V8-v1/00000/0C80F82F-DCE4-E211-976F-002481E94B9C.root"
        "/store/relval/CMSSW_5_3_11/RelValTTbar/GEN-SIM-RECO/START53_LV3_Alca7TeV_14Jun2013-v1/00000/CA8F250D-F1D4-E211-895D-001E67398B2E.root"
    )
)

process.TFileService = cms.Service("TFileService",
                                     fileName = cms.string("tree1000.root")
                                   )

process.subtract = cms.EDProducer('VoronoiBackgroundProducer',
                                  src = cms.InputTag('particleFlow'),
                                  equalizeR = cms.double(0.3),
				  doEqualize = cms.bool(True),
				  equalizeThreshold0 = cms.double(5.0),
				  equalizeThreshold1 = cms.double(35.0)
                                  )

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("RecoHI.HiJetAlgos.HiRecoPFJets_cff")
process.load("CmsHi.JetAnalysis.inclusiveJetAnalyzer_cff")
process.load("CmsHi.JetAnalysis.PatAna_MC_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.GlobalTag.toGet.extend([ cms.PSet(record = cms.string("JetCorrectionsRecord"),
                                                           tag = cms.string("JetCorrectorParametersCollection_Fall12_V5_MC_AK5PF"),
                                                           connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
                                                           label = cms.untracked.string("AK5PF")
                                                           ),
                                 ])

process.akVs5PFJets = process.ak5PFJets.clone(
    doPVCorrection = False,
    doPUOffsetCorr = True,    
    subtractorName = cms.string("VoronoiSubtractor"),
#    subtractorName = cms.string("MultipleAlgoIterator"),

    bkg = cms.InputTag("subtract"),
    src = cms.InputTag('particleFlow'),
    dropZeros = cms.untracked.bool(True),
    doAreaFastjet = False
)

process.akPu5PFmatch = process.patJetGenJetMatch.clone(
  src = cms.InputTag("akVs5PFJets"),
  matched = cms.InputTag("ak5GenJets")
  )

#process.genPartons.src = cms.InputTag("genParticles")
process.akPu5PFparton = process.patJetPartonMatch.clone(src = cms.InputTag("akVs5PFJets"),
                                                        matched = cms.InputTag("genParticles")
                                                        )


process.akVs5PFcorr = process.akPu5PFcorr.clone(
    src = cms.InputTag("akVs5PFJets"),
    payload = "AK5PF"
    )

process.ak5PFcorr = process.akPu5PFcorr.clone(
        src = cms.InputTag("ak5PFJets"),
        payload = "AK5PF"
        )

process.akVs5PFpatJets = process.akPu5PFpatJets.clone(jetSource = cms.InputTag("akVs5PFJets"),
                                                      jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5PFcorr")),
                                                      genJetMatch = cms.InputTag("akPu5PFmatch"),
                                                      genPartonMatch = cms.InputTag("akPu5PFparton"),
                                                      jetIDMap = cms.InputTag("akPu1PFJetID"),
                                                      )



process.ak5PFpatJets = process.akPu5PFpatJets.clone(jetSource = cms.InputTag("ak5PFJets"),
                                                    jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak5PFcorr")),
                                                    addGenJetMatch = cms.bool(False),
                                                    addGenPartonMatch = cms.bool(False),                                                    
                                                    )


process.akVs5PFJetAnalyzer = process.inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5PFpatJets"),
                                                                genjetTag = 'ak5GenJets',
                                                                rParam = 0.5,
                                                                matchJets = cms.untracked.bool(True),
                                                                matchTag = 'ak5PFpatJets',
                                                                pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                                trackTag = cms.InputTag("generalTracks"),
                                                                fillGenJets = True,
                                                                isMC = True,
                                                                genParticles = cms.untracked.InputTag("genParticles")
                                                                )


process.p = cms.Path(process.subtract*
                     process.akVs5PFJets*
                     process.akPu5PFmatch*
                     process.akPu5PFparton*
                     process.akVs5PFcorr*
                     process.akVs5PFpatJets*
                     process.ak5PFcorr*
                     process.ak5PFpatJets*                     
                     process.akVs5PFJetAnalyzer
                     )
