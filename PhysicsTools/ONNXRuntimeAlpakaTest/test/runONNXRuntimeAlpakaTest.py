import FWCore.ParameterSet.Config as cms
from PhysicsTools.ONNXRuntimeAlpakaTest.options_cff import parse_args
from PhysicsTools.ONNXRuntimeAlpakaTest.modules import (
    onnxtest_DataSource_alpaka,
    onnxtest_InspectionSink
)

args = parse_args()
process = cms.Process("ONNXRuntimeAlpakaTest")

# enable multithreading
process.options.numberOfThreads = max(args.numberOfThreads, 1)
process.options.numberOfStreams = max(args.numberOfStreams, 1)

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# process a limited number of events
process.maxEvents.input = max(args.numberOfEvents, 1)

# empty source
process.source = cms.Source("EmptySource")

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options.wantSummary = args.wantSummary

alpaka = cms.untracked.PSet(
    backend = cms.untracked.string(args.backend)
)
data = "PhysicsTools/ONNXRuntimeAlpakaTest/data/"

# setup chain configs
process.path = cms.Path()
# data provider
process.DataSource = onnxtest_DataSource_alpaka(
    totalSize = max(args.totalSize, 0),
    alpaka = alpaka
)
process.path += process.DataSource

# --only SimpleNet
if "SimpleNet" in args.only:
    from PhysicsTools.ONNXRuntimeAlpakaTest.modules import onnxtest_SimpleNet_alpaka, onnxtest_SimpleNetMiniBatch_alpaka
    # SoA columns packed into a [batch, 3] tensor
    process.SimpleNet = onnxtest_SimpleNet_alpaka(
        model = data + "SimpleNet.onnx",
        particles = "DataSource",
        alpaka = alpaka
    )
    process.path += process.SimpleNet

    # SoA memory used directly as a [3, padded size] tensor
    process.SimpleNetFeatureMajor = onnxtest_SimpleNet_alpaka(
        model = data + "SimpleNetFeatureMajor.onnx",
        featureMajor = True,
        particles = "DataSource",
        alpaka = alpaka
    )
    process.path += process.SimpleNetFeatureMajor

    # mini-batches, in a module whose queue changes from event to event
    process.SimpleNetMiniBatch = onnxtest_SimpleNetMiniBatch_alpaka(
        model = data + "SimpleNet.onnx",
        batchSize = args.batchSize,
        particles = "DataSource",
        alpaka = alpaka
    )
    process.path += process.SimpleNetMiniBatch

# --only MultiHeadNet
if "MultiHeadNet" in args.only:
    from PhysicsTools.ONNXRuntimeAlpakaTest.modules import onnxtest_MultiHeadNet_alpaka
    process.MultiHeadNet = onnxtest_MultiHeadNet_alpaka(
        model = data + "MultiHeadNet.onnx",
        particles = "DataSource",
        alpaka = alpaka
    )
    process.path += process.MultiHeadNet

# --only MaskedNet
if "MaskedNet" in args.only:
    from PhysicsTools.ONNXRuntimeAlpakaTest.modules import onnxtest_MaskedNet_alpaka
    process.MaskedNet = onnxtest_MaskedNet_alpaka(
        model = data + "MaskedNet.onnx",
        particles = "DataSource",
        alpaka = alpaka
    )
    process.path += process.MaskedNet

# --only TinyResNet
if "TinyResNet" in args.only:
    from PhysicsTools.ONNXRuntimeAlpakaTest.modules import onnxtest_TinyResNet_alpaka
    process.TinyResNet = onnxtest_TinyResNet_alpaka(
        model = data + "TinyResNet.onnx",
        images = "DataSource",
        alpaka = alpaka
    )
    process.path += process.TinyResNet

    process.TinyResNetMiniBatch = onnxtest_TinyResNet_alpaka(
        model = data + "TinyResNet.onnx",
        batchSize = args.batchSize,
        images = "DataSource",
        alpaka = alpaka
    )
    process.path += process.TinyResNetMiniBatch

# check the results of the models that have been run
process.InspectionSink = onnxtest_InspectionSink(
    particles = "DataSource",
    simple_net = "SimpleNet",
    simple_net_feature_major = "SimpleNetFeatureMajor",
    simple_net_minibatch = "SimpleNetMiniBatch",
    masked_net = "MaskedNet",
    multi_head_net = "MultiHeadNet",
    resnet = "TinyResNet",
    resnet_minibatch = "TinyResNetMiniBatch"
)
process.path += process.InspectionSink
