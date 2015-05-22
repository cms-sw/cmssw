import FWCore.ParameterSet.Config as cms

from CMGTools.RootTools.utils.vertexWeight.vertexWeightsSummer11_cfi import *
from CMGTools.RootTools.utils.vertexWeight.vertexWeightsFall11_cfi import *
from CMGTools.RootTools.utils.vertexWeight.vertexWeights3DSummer11_cfi import *
from CMGTools.RootTools.utils.vertexWeight.vertexWeights3DFall11_cfi import *
from CMGTools.RootTools.utils.vertexWeight.vertexWeights2012_cfi import *

vertexWeightSequence = cms.Sequence(

    #2011 weights using observed pileup
    vertexWeightEPSJul8
    +vertexWeightLeptonPhoton
    +vertexWeightMay10ReReco
    +vertexWeightPromptRecov4
    +vertexWeight05AugReReco
    +vertexWeightPromptRecov6
    +vertexWeight2invfb
    +vertexWeight2011B
    +vertexWeight2011AB
    
    +vertexWeightFall11EPSJul8
    +vertexWeightFall11LeptonPhoton
    +vertexWeightFall11May10ReReco
    +vertexWeightFall11PromptRecov4
    +vertexWeightFall1105AugReReco
    +vertexWeightFall11PromptRecov6
    +vertexWeightFall112invfb
    +vertexWeightFall112011B
    +vertexWeightFall112011AB

    #3D reweighting technique in 2011
    +vertexWeight3DMay10ReReco
    +vertexWeight3DPromptRecov4
    +vertexWeight3D05AugReReco
    +vertexWeight3DPromptRecov6
    +vertexWeight3D2invfb
    +vertexWeight3D2011B
    +vertexWeight3D2011AB

    +vertexWeight3DFall11May10ReReco
    +vertexWeight3DFall11PromptRecov4
    +vertexWeight3DFall1105AugReReco
    +vertexWeight3DFall11PromptRecov6
    +vertexWeight3DFall112invfb
    +vertexWeight3DFall112011B
    +vertexWeight3DFall112011AB

    #2012 weights use true pileup
    +vertexWeightSummer12MCICHEPData
    +vertexWeightSummer12MC53XICHEPData
    +vertexWeightSummer12MC53XHCPData
    +vertexWeightSummer12MC53X2012D6fbData
    +vertexWeightSummer12MC53X2012ABCDData
    +vertexWeightSummer12MC53X2012BCDData
    
    )
