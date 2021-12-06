from __future__ import absolute_import
import FWCore.ParameterSet.Config as cms
from .pfDeepDoubleBvLJetTags_cfi import pfDeepDoubleBvLJetTags
from .pfDeepDoubleCvLJetTags_cfi import pfDeepDoubleCvLJetTags
from .pfDeepDoubleCvBJetTags_cfi import pfDeepDoubleCvBJetTags

pfMassIndependentDeepDoubleBvLV2JetTags = pfDeepDoubleBvLJetTags.clone(
    model_path="RecoBTag/Combined/data/DeepDoubleX/102X/V02/BvL.onnx",
    input_names=["input_1", "input_2", "input_3", "input_4"],
    version="V2",
)

pfMassIndependentDeepDoubleCvLV2JetTags = pfDeepDoubleCvLJetTags.clone(
    model_path="RecoBTag/Combined/data/DeepDoubleX/102X/V02/CvL.onnx",
    input_names=["input_1", "input_2", "input_3", "input_4"],
    version="V2",
)

pfMassIndependentDeepDoubleCvBV2JetTags = pfDeepDoubleCvBJetTags.clone(
    model_path="RecoBTag/Combined/data/DeepDoubleX/102X/V02/CvB.onnx",
    input_names=["input_1", "input_2", "input_3", "input_4"],
    version="V2",
)
