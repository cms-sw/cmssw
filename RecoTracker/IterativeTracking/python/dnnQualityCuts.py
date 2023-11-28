# https://indico.cern.ch/event/947686/contributions/3981867/attachments/2090510/3512658/TrackingDNN_24_8_2020.pdf
import FWCore.ParameterSet.Config as cms
qualityCutDictionary = cms.PSet(
   InitialStep         =        cms.vdouble(-0.35,  0.1,   0.28),
   LowPtQuadStep       =        cms.vdouble(-0.37,  0.08,  0.28),
   HighPtTripletStep   =        cms.vdouble(0.47,   0.55,  0.62),
   LowPtTripletStep    =        cms.vdouble(-0.26,  0.09,  0.33),
   DetachedQuadStep    =        cms.vdouble(-0.66, -0.15,  0.46),
   DetachedTripletStep =        cms.vdouble(-0.42,  0.16,  0.78),
   PixelPairStep       =        cms.vdouble(-0.31, -0.13,  0.13),
   MixedTripletStep    =        cms.vdouble(-0.86, -0.68, -0.43),
   PixelLessStep       =        cms.vdouble(-0.80, -0.69, -0.40),
   TobTecStep          =        cms.vdouble(-0.76, -0.65, -0.55),
   DisplacedRegionalStep =        cms.vdouble(-0.76, -0.65, -0.55),
   JetCoreRegionalStep =        cms.vdouble(-0.62, -0.49,  0.02)
)

from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF

trackdnn_CKF.toModify(qualityCutDictionary, 
   InitialStep         =        [-0.57,  0.02,  0.3 ],
   LowPtQuadStep       =        [-0.33,  0.13,  0.35],
   HighPtTripletStep   =        [0.52,   0.6,   0.67],
   LowPtTripletStep    =        [-0.23,  0.15,  0.41],
   DetachedQuadStep    =        [-0.6,  -0.06,  0.54],
   DetachedTripletStep =        [-0.35,  0.33,  0.84],
   PixelPairStep       =        [-0.29, -0.1,   0.18],
   MixedTripletStep    =        [-0.87, -0.61, -0.17],
   PixelLessStep       =        [-0.82, -0.61, -0.16],
   TobTecStep          =        [-0.75, -0.65, -0.53],
   DisplacedRegionalStep =        [-0.75, -0.65, -0.53],
   JetCoreRegionalStep =        [-0.14, -0.12,  0.63],
)
