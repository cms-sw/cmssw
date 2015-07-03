import FWCore.ParameterSet.Config as cms
from RecoBTag.CTagging.trainingvars import get_var_pset
#use import as to mask it to process.load() (hopefully)
import RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexSoftLeptonComputer_cfi as sl_cfg 
#import candidateCombinedSecondaryVertexSoftLeptonComputer

charmTagsComputer = cms.ESProducer(
   'CharmTaggerESProducer',
   #clone the cfg only
   slComputerCfg = cms.PSet(
      **sl_cfg.candidateCombinedSecondaryVertexSoftLeptonComputer.parameters_()
      ),
   weightFile = cms.FileInPath('RecoBTag/CTagging/data/c_vs_udsg.weight.xml'),
   variables = cms.VPSet(
      #would be cool if we could directly read the TMVA wieight file and get the information from there
      #the main problem is unpack cms.FileInPath from python, to make it grid-safe
      get_var_pset("trackSip2dSig_0"),
      get_var_pset("trackSip2dSig_1"),
      get_var_pset("trackSip2dSig_2"),
      get_var_pset("trackSip3dSig_0"),
      get_var_pset("trackSip3dSig_1"),
      get_var_pset("trackSip3dSig_2"),
      get_var_pset("trackSip2dVal_0"),
      get_var_pset("trackSip2dVal_1"),
      get_var_pset("trackSip2dVal_2"),
      get_var_pset("trackSip3dVal_0"),
      get_var_pset("trackSip3dVal_1"),
      get_var_pset("trackSip3dVal_2"),
      get_var_pset("trackJetDist_0"),
      get_var_pset("trackJetDist_1"),
      get_var_pset("trackJetDist_2"),
      get_var_pset("trackDecayLenVal_0"),
      get_var_pset("trackDecayLenVal_1"),
      get_var_pset("trackDecayLenVal_2"),
      get_var_pset("vertexMass_0"),
      get_var_pset("vertexEnergyRatio_0"),
      get_var_pset("trackSip2dSigAboveCharm_0"),
      get_var_pset("trackSip3dSigAboveCharm_0"),
      get_var_pset("flightDistance2dSig_0"),
      get_var_pset("flightDistance3dSig_0"),
      get_var_pset("vertexJetDeltaR_0"),
      get_var_pset("trackSip2dValAboveCharm_0"),
      get_var_pset("trackSip3dValAboveCharm_0"),
      get_var_pset("chargedHadronEnergyFraction"),
      get_var_pset("massVertexEnergyFraction_0"),
      get_var_pset("vertexBoostOverSqrtJetPt_0"),
      get_var_pset("leptonPtRel_0"),
      get_var_pset("leptonSip3d_0"),
      get_var_pset("leptonSip3d_1"),
      get_var_pset("vertexNTracks_0"),
      get_var_pset("jetNSecondaryVertices")
      ),
   computer = cms.ESInputTag('combinedSecondaryVertexSoftLeptonComputer'),
   tagInfos = cms.VInputTag(
      cms.InputTag('pfImpactParameterTagInfos'),
      cms.InputTag('pfInclusiveSecondaryVertexFinderCtagLTagInfos'),
      cms.InputTag('softPFMuonsTagInfos'),
      cms.InputTag('softPFElectronsTagInfos'),
      )
   )
