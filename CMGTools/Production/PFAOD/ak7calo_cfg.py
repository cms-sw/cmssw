import sys
import os

sys.path.append( os.getcwd() )
from base import *

import __main__

tag = 'ak7CaloJets'

output = '_'.join( [sampleStr, tag] )
output += '.root'

print 'output file', output 

process.out.fileName = output 

process.out.outputCommands.extend(
    [
    'keep *_ak7CaloJets_*_*',
    ])
