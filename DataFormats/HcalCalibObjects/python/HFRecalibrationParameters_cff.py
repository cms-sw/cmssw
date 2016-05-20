import FWCore.ParameterSet.Config as cms

# These parameters are for an HF recalibration function of the form:
#
#   1 + (A * Sqrt(integratedLuminosity)) + (B * integratedLuminosity)
#
# Each pair of parameters is associated with a:
#
#   depth (1 and 2)
#
# and
#
#   ieta 30-41
# For a total of 48 parameters.

HFRecalParameterBlock = cms.PSet(                               
    HFdepthOneParameterA = cms.vdouble(0.004123, 0.006020, 0.008201, 0.010489, 0.013379, 0.016997, 0.021464, 0.027371, 0.034195, 0.044807, 0.058939, 0.125497),
    HFdepthOneParameterB = cms.vdouble(-0.000004, -0.000002, 0.000000, 0.000004, 0.000015, 0.000026, 0.000063, 0.000084, 0.000160, 0.000107, 0.000425, 0.000209),

    HFdepthTwoParameterA = cms.vdouble(0.002861, 0.004168, 0.006400, 0.008388, 0.011601, 0.014425, 0.018633, 0.023232, 0.028274, 0.035447, 0.051579, 0.086593),
    HFdepthTwoParameterB = cms.vdouble(-0.000002, -0.000000, -0.000007, -0.000006, -0.000002, 0.000001, 0.000019, 0.000031, 0.000067, 0.000012, 0.000157, -0.000003)
)