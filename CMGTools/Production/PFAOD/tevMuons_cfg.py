import sys
import os

sys.path.append( os.getcwd() )
from base import *

import __main__

tag = 'tevMuons'

output = '_'.join( [sampleStr, tag] )
output += '.root'

print 'output file', output 

process.out.fileName = output 



process.out.outputCommands.extend(
    [
    'keep recoTracks_tevMuons_default_*',
    'keep recoTracks_tevMuons_dyt_*',
    'keep recoTracks_tevMuons_firstHit_*',
    'keep recoTracks_tevMuons_picky_*',
    'keep recoTrackExtras_tevMuons_default_*',
    'keep recoTrackExtras_tevMuons_dyt_*',
    'keep recoTrackExtras_tevMuons_firstHit_*',
    'keep recoTrackExtras_tevMuons_picky_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_default_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_dyt_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_firstHit_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_picky_*',
    ])

