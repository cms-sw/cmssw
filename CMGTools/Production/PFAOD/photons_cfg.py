import sys
import os

sys.path.append( os.getcwd() )
from base import *

import __main__

tag = 'photons'

output = '_'.join( [sampleStr, tag] )
output += '.root'

print 'output file', output 

process.out.fileName = output 



process.out.outputCommands.extend(
    [
    'keep recoPhotonCores_photonCore__*',
    'keep recoPhotons_pfPhotonTranslator_pfphot_*',
    'keep recoPhotons_photons__*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLoose_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLooseEM_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDTight_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_pfphot_*',
    'keep recoSuperClusters_pfPhotonTranslator_pfphot_*',
    ])

