import os
import imp
import math
from CMGTools.RootTools.Style import formatPad,Style
from CMGTools.H2TauTau.proto.HistogramSet import histogramSet
from CMGTools.H2TauTau.proto.plotter.H2TauTauDataMC_diTau import *
from CMGTools.H2TauTau.proto.plotter.QCDEstimation_diTau  import *
from CMGTools.H2TauTau.proto.plotter.SaveHistograms_diTau import *

from plot_H2TauTauDataMC_diTau import run2012, just125

def lists(just125,susy,run2012) :

  if susy : signal = ['HiggsSUSYBB','HiggsSUSYGluGlu']
  else    : signal = ['HiggsGGH','HiggsVBF','HiggsVH']
  
  bkg = ['DYJets'         ,
         'WJets'          ,
         'Tbar_tW'        ,
         'T_tW'           ,
         'TTJetsFullLept' ,
         'TTJetsSemiLept' ,
         'TTJetsHadronic' ,
         'WWJetsTo2L2Nu'  ,
         'WZJetsTo2L2Q'   ,
         'WZJetsTo3LNu'   ,
         'ZZJetsTo4L'     ,
         'ZZJetsTo2L2Nu'  ,
         'ZZJetsTo2L2Q'   ,
         #'HiggsGGHtoWW125',
         #'HiggsVBFtoWW125',
         #'HiggsVHtoWW125' ,
        ]   #,'TTJets','WW','WZ','ZZ'
  
  if susy : bkg.extend(['HiggsGGH125','HiggsVBF125','HiggsVH125','TTJets_emb']) 
  else    : pass                                      
  #data2012  = ['data_Run2012A_PromptReco_v1',                                      'data_Run2012B_PromptReco_v1','data_Run2012C_PromptReco_v1','data_Run2012C_PromptReco_v2' ,'data_Run2012D_PromptReco_v1']
  #embed2012 = ['embed_Run2012A_13Jul2012_v1','embed_Run2012A_recover_06Aug2012_v1','embed_Run2012B_13Jul2012_v4','embed_Run2012C_24Aug2012_v1','embed_Run2012C_PromptReco_v2','embed_2012D_PromptReco_v1'  ]
  data2012  = ['data_Run2012A_22Jan2013_v1' , 'data_Run2012B_22Jan2013_v1' , 'data_Run2012C_22Jan2013_v1' , 'data_Run2012D_22Jan2013_v1' ]
  embed2012 = ['embed_Run2012A_22Jan2013_v1', 'embed_Run2012B_22Jan2013_v1', 'embed_Run2012C_22Jan2013_v1', 'embed_Run2012D_22Jan2013_v1']

  if just125 : masses = [125]
  else       : masses = [90,95,100,105,110,115,120,125,130,135,140,145,150,155,160]
  if susy    : masses = [80,90,100,110,120,130,140,160,180,200,250,300,350,400,450,500,600,700,800,900,1000]

  return signal, bkg, data2012, embed2012, masses

def componentsWithData (selComps, weights, susy) :

  signal, bkg, data2012, embed2012, masses = lists(just125,susy,run2012)
  
  print signal, bkg, data2012, embed2012, masses
  
  allcomps = []
  for c in [bkg,data2012,embed2012] :
    allcomps.extend(c)
  
  selCompsDataMass = {}
  weightsDataMass  = {}

  selCompsDataMassNoSignal = {} 
  weightsDataMassNoSignal  = {}

  for comp in allcomps :
    if comp in selComps.keys() :
      selCompsDataMassNoSignal[comp]  = copy.deepcopy(selComps[comp])
      weightsDataMassNoSignal[comp]   = copy.deepcopy(weights[comp])
 
  for mPoint in masses :
    
    selCompsDataMass[mPoint] = {}
    weightsDataMass[mPoint]  = {}

    for comp in signal : 
      selCompsDataMass[mPoint][comp+str(mPoint)] = copy.deepcopy(selComps[comp +str(mPoint)])          
      weightsDataMass[mPoint][comp+str(mPoint)]  = copy.deepcopy(weights[comp+str(mPoint)])
    
    selCompsDataMass[mPoint].update(selCompsDataMassNoSignal)
    weightsDataMass[mPoint].update(weightsDataMassNoSignal)
  
  return selCompsDataMass, weightsDataMass

def componentsWithOutData (selComps, weights, susy) :
  
  signal, bkg, data2012, embed2012, masses = lists(just125,susy,run2012)

  allcomps = []
  for c in [bkg,embed2012] :
    allcomps.extend(c)
  
  selCompsDataMass = {}
  weightsDataMass  = {}

  selCompsDataMassNoSignal = {} 
  weightsDataMassNoSignal  = {}

  for comp in allcomps :
    if comp in selComps.keys() :
      selCompsDataMassNoSignal[comp]  = copy.deepcopy(selComps[comp])
      weightsDataMassNoSignal[comp]   = copy.deepcopy(weights[comp])
 
  for mPoint in masses :
    
    selCompsDataMass[mPoint] = {}
    weightsDataMass[mPoint]  = {}
    
    for comp in signal : 
      selCompsDataMass[mPoint][comp+str(mPoint)] = copy.deepcopy(selComps[comp +str(mPoint)])          
      weightsDataMass[mPoint][comp+str(mPoint)]  = copy.deepcopy(weights[comp+str(mPoint)])
    
    selCompsDataMass[mPoint].update(selCompsDataMassNoSignal)
    weightsDataMass[mPoint].update(weightsDataMassNoSignal)
    
  return selCompsDataMass, weightsDataMass

def componentsWithOutSignal (selComps, weights, susy) :

  signal, bkg, data2012, embed2012, masses = lists(just125,susy,run2012)

  allcomps = []
  for c in [bkg,data2012,embed2012] :
    allcomps.extend(c)
 
  selCompsDataMassNoSignal = {} 
  weightsDataMassNoSignal  = {}
 
  for comp in allcomps :
    if comp in selComps.keys() :
      selCompsDataMassNoSignal[comp]  = copy.deepcopy(selComps[comp])
      weightsDataMassNoSignal[comp]   = copy.deepcopy(weights[comp])
  return selCompsDataMassNoSignal, weightsDataMassNoSignal
 
