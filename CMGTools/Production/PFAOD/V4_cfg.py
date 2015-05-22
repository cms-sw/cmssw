import sys
import os

sys.path.append( os.getcwd() )
from base import *

import __main__

tag = 'V4'

output = '_'.join( [sampleStr, tag] )
output += '.root'

print 'output file', output 

process.out.fileName = output 

process.out.outputCommands.extend( V4 )
