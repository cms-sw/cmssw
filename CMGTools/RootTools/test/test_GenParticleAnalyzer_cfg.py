import copy
import os
import glob
import CMGTools.RootTools.fwlite.Config as cfg
from CMGTools.RootTools.fwlite.Config import printComps
from CMGTools.RootTools.RootTools import * 
from CMGTools.Production.getFiles import getFiles

genAna = cfg.Analyzer(
    'GenParticleAnalyzer',
    src = 'genParticlesPruned'
    )

###############################################################################



DY = cfg.MCComponent(
    name = 'DY',
    files = getFiles('/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM/V5_B/PAT_CMG_V5_16_0', 'cmgtools', 'cmgTuple.*root')[:5],
    xSection = 1, 
    nGenEvents = 1,
    triggers = [],
    effCorrFactor = 1 )
DY.splitFactor = 1


selectedComponents = [DY]  

sequence = cfg.Sequence( [
    genAna
   ] )

# creation of the processing configuration.
# we define here on which components to run, and
# what is the sequence of analyzers to run on each event. 
config = cfg.Config( components = selectedComponents,
                     sequence = sequence )

printComps(config.components, True)
