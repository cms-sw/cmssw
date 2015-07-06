import FWCore.ParameterSet.Config as cms

# Kinematic reweighting for the embedded samples from here https://twiki.cern.ch/twiki/bin/viewauth/CMS/MuonTauReplacementRecHit
# Can also put this into a separate file under tools

def setupEmbedding(process, channel):

    isEmbedded = process.source.fileNames[0].find('embedded') != -1
    isRHEmbedded = process.source.fileNames[0].find('RHembedded') != -1

    if isEmbedded and isRHEmbedded:
        process.load('TauAnalysis.MCEmbeddingTools.embeddingKineReweight_cff')

        if channel == 'all':
            print 'ERROR: not possible to run all the channels for the embedded samples right now'

        # for "standard" e+tau channel
        if channel == 'tau-ele':
            process.embeddingKineReweightRECembedding.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_ePtGt20tauPtGt18_recEmbedded.root")
            process.tauElePath.insert(-1, process.embeddingKineReweightSequenceRECembedding)

        # for e+tau channel of "soft lepton" analysis
        #embeddingKineReweightRECembedding.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_ePt9to30tauPtGt18_recEmbedded.root")

        # for "standard" mu+tau channel
        if channel == 'tau-mu':
            process.embeddingKineReweightRECembedding.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_muPtGt16tauPtGt18_recEmbedded.root")
            process.tauMuPath.insert(-1, process.embeddingKineReweightSequenceRECembedding)

        # for mu+tau channel of "soft lepton" analysis
        #embeddingKineReweightRECembedding.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_muPt7to25tauPtGt18_recEmbedded.root")

        # for tautau channel
        if channel == 'di-tau':
            process.embeddingKineReweightRECembedding.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_tautau_recEmbedded.root")
            process.diTauPath.insert(-1, process.embeddingKineReweightSequenceRECembedding)

        print "Embedded samples; using kinematic reweighting file:", process.embeddingKineReweightRECembedding.inputFileName

        # for emu, mumu and ee channels
        #embeddingKineReweightRECembedding.inputFileName = cms.FileInPath("TauAnalysis/MCEmbeddingTools/data/embeddingKineReweight_recEmbedding_emu.root")
        return isEmbedded
