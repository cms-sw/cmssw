#! /usr/bin/env python
# Author: Jacob Herman
# Date: 7/24/2011
# Purpose: This script allows the user to veiw the APV noise module for passed modules and noise dictionaries by default or optionally for given assignment from an assignment dictionary

from HVMapToolsClasses import HVMapNoise
from optparse import OptionParser
import pickle

#Initialize Command line interface
parser = OptionParser()
parser.add_option('-a',action = 'store', nargs = 2, dest = 'assign', help = "Veiw APV noise details by assignment in a given assignment dictionary. Takes two ordered arguments: 1) the directory of the assignment 2) the assignment. Only one word assignments supported, however if 'HV1' is passed it will print all assignments containing the sub string 'HV1'")

(Commands, args) = parser.parse_args()

if len(args) == 0:
    print "Waring: no pedestal noise information passed"

Noise = HVMapNoise('ViewNoise')    
#fill Noise and remove so detIDs can be parsed

for path in args[:]:
    
    
    if '.pkl' in str(path):
        
        Noise.AddFromPickle(path)
        #print args
        args.remove(path)
        #print args
        
#Veiw by detID
if Commands.assign is None:

    for detID in args:
        try:
            Noise.APVDetails(detID)
        except KeyError:
            print detID, "Not Found"
            
#View by assignment
else:
    file = open(Commands.assign[0])
    assigndict = pickle.load(file)

    for detID in assigndict.keys():
        if Commands.assign[1] in assigndict[detID]:
            try:
                Noise.APVDetails(detID)
                print detID, "Was assigned: ", assigndict[detID]
            except KeyError:
                print assigndict[detID], detID,'Not Found'
            
    
