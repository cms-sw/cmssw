# https://indico.cern.ch/event/947686/contributions/3981867/attachments/2090510/3512658/TrackingDNN_24_8_2020.pdf
import FWCore.ParameterSet.Config as cms
qualityCutDictionary = cms.PSet(
   InitialStep         =        cms.vdouble(-0.48, 0.03, 0.25),
   LowPtQuadStep       =        cms.vdouble(-0.33, 0.18, 0.41),
   HighPtTripletStep   =        cms.vdouble(0.48, 0.55, 0.62),
   LowPtTripletStep    =        cms.vdouble(-0.21, 0.17, 0.41),
   DetachedQuadStep    =        cms.vdouble(-0.62, -0.09 ,0.50),
   DetachedTripletStep =        cms.vdouble(-0.52, 0.04, 0.76),
   PixelPairStep       =        cms.vdouble(-0.47, -0.33, -0.05),
   MixedTripletStep    =        cms.vdouble(-0.87, -0.61 ,-0.13),
   PixelLessStep       =        cms.vdouble(-0.20, -0.10, 0.40),
   TobTecStep          =        cms.vdouble(-0.44, -0.26, -0.14), 
   JetCoreRegionalStep =        cms.vdouble(-0.14, 0.13, 0.61)

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
