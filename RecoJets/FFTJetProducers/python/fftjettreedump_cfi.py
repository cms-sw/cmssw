# FFTJet clustering tree dumper configuration

import os, errno, sys
import FWCore.ParameterSet.Config as cms

from RecoJets.FFTJetProducers.fftjetcommon_cfi import *

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise

# Output directory for the trees
clustering_trees_outdir = "./ClusteringTrees"
if (sys.argv[0] == "cmsRun"):
    mkdir_p(clustering_trees_outdir)

# Base name for the output files
trees_basename = "clustree"

# Configure the FFTJet tree dumper module
fftjetTreeDumper = cms.EDAnalyzer(
    "FFTJetTreeDump",
    #
    # Label for the input clustering tree (either sparse or dense)
    treeLabel = cms.InputTag("fftjetpatreco", "FFTJetPatternRecognition"),
    #
    # Prefix for the output trees
    outputPrefix = cms.string(clustering_trees_outdir + '/' + trees_basename),
    #
    # Eta limit for the plot
    etaMax = cms.double(fftjet_standard_eta_range),
    #
    # Do we have the complete event at the lowest tree scale?
    insertCompleteEvent = cms.bool(fftjet_insert_complete_event),
    completeEventScale = cms.double(fftjet_complete_event_scale),
    #
    # The initial set of scales used by the pattern recognition stage.
    # This is also the final set unless clustering tree construction
    # is adaptive. Needed here for reading back non-adaptive trees.
    InitialScales = fftjet_patreco_scales_50,
    #
    # Clustering tree distance functor
    TreeDistanceCalculator = fftjet_fixed_bandwidth_distance,
    #
    # Which quantity will be mapped into OpenDX glyph size?
    GlyphSize = cms.PSet(
        Class = cms.string("ScaledMagnitude2")
    ),
    #
    # Which quantity will be mapped into OpenDX glyph color?
    GlyphColor = cms.PSet(
        Class = cms.string("ScaledHessianDet")
    )
)
