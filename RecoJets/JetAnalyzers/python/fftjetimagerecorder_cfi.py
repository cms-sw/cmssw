#
# FFTJet image recorder configuration
#
# I. Volobouev, June 2, 2011
#
import FWCore.ParameterSet.Config as cms

fftjet_image_recorder = cms.EDAnalyzer(
    "FFTJetImageRecorder",
    #
    # Label for the histograms produced by FFTJetEFlowSmoother
    histoLabel = cms.InputTag("fftjetsmooth", "FFTJetEFlowSmoother")
)
