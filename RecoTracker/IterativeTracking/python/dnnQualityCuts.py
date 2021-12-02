# https://indico.cern.ch/event/947686/contributions/3981867/attachments/2090510/3512658/TrackingDNN_24_8_2020.pdf
import FWCore.ParameterSet.Config as cms
qualityCutDictionary = cms.PSet(
   InitialStep         =        cms.vdouble(-0.56, -0.08, 0.17),
   LowPtQuadStep       =        cms.vdouble(-0.35, 0.13, 0.36),
   HighPtTripletStep   =        cms.vdouble(0.41, 0.49, 0.57),
   LowPtTripletStep    =        cms.vdouble(-0.29, 0.09, 0.36),
   DetachedQuadStep    =        cms.vdouble(-0.63, -0.14, 0.49),
   DetachedTripletStep =        cms.vdouble(-0.32, 0.24, 0.81),
   PixelPairStep       =        cms.vdouble(-0.38, -0.23, 0.04),
   MixedTripletStep    =        cms.vdouble(-0.83, -0.63, -0.38),
   PixelLessStep       =        cms.vdouble(-0.60, -0.40, 0.02),
   TobTecStep          =        cms.vdouble(-0.71, -0.58, -0.46), 
   JetCoreRegionalStep =        cms.vdouble(-0.53, -0.33, 0.18)
)

from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF

trackdnn_CKF.toModify(qualityCutDictionary, 
   InitialStep         =        [-0.49, 0.08, 0.34],
   LowPtQuadStep       =        [-0.29, 0.17, 0.39],
   HighPtTripletStep   =        [0.5, 0.58, 0.65],
   LowPtTripletStep    =        [-0.30, 0.06, 0.32],
   DetachedQuadStep    =        [-0.61, -0.09, 0.51],
   DetachedTripletStep =        [-0.38, 0.31, 0.83],
   PixelPairStep       =        [-0.25, -0.07, 0.19],
   MixedTripletStep    =        [-0.86, -0.57, -0.12],
   PixelLessStep       =        [-0.81, -0.61, -0.17],
   TobTecStep          =        [-0.67, -0.54, -0.40],
   JetCoreRegionalStep =        [0.00, 0.03, 0.68]
)
