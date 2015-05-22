import sys
import os

sys.path.append( os.getcwd() )
from base import *

import __main__

tag = 'caloClusters'

output = '_'.join( [sampleStr, tag] )
output += '.root'

print 'output file', output 

process.out.fileName = output 



process.out.outputCommands.extend(
    [
    'keep recoCaloClusters_hybridSuperClusters_hybridBarrelBasicClusters_*',
    'keep recoCaloClusters_multi5x5BasicClusters_multi5x5EndcapBasicClusters_*',
    'keep recoCaloClusters_hybridSuperClusters_uncleanOnlyHybridBarrelBasicClusters_*',
    'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
    'keep recoCaloClusters_pfPhotonTranslator_pfphot_*',
    ])

