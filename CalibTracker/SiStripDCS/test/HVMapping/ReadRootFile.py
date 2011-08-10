#! /usr/bin/env python
# Author:        Jake Herman
# Date:          22 June 2011
# Purpose:       Program reads in root files containing ADC noise means for pedestal runs and makes histograms for each each detector in each run type
#TO DO: Find a way to make creating histos quiet


import os,time,sys

sys.path.append('Classes/')

from optparse import OptionParser

#intialize command line interface
parser = OptionParser(usage='''Usage: ./ReadRootFile.py [options] [PickledNoise]
 Examples:
 ./ReadRootFile.py YourPedestalRunsDirectory/* -b
    Wildcard is accepted to read all files in a absolute path directory
 ./ReadRootFile.py YourFirstRootFile YourSecondRootFile YourThirdRootfile
    Individual rootfiles can be used as arguments''')

parser.add_option('-b',action = 'store_true', default = False, dest = 'batch', help = 'Runs pyroot in batch mode so that no graphics will be desplayed on screen')
parser.add_option('-c', action = 'store_true', default = False ,dest = 'comp', help = 'Tells script to compensate for differences in noise due to DAQ mode by multiplying raw noise by 1.7 if the data was taken in PEAK mode')

(Commands, args) = parser.parse_args()

#Intialize pyroot in batch mode if selected
if Commands.batch:
    sys.argv.append('-b')

#import things concerning pyroot
from HVMapToolsClasses import HVMapNoise
from ROOT import TCanvas

if 'args' not in dir():
    print "\n\nError: User did not specify any root files for reading"
    sys.exit()

#Parse Arguments
for RootFile in args:

   

    #Initialize Noise Container
    RootFileNoise = HVMapNoise(RootFile.split('/')[-1])

    #Acquire data from root file
    RootFileNoise.ReadRootFile(RootFile, Commands.comp)

    #Make directory structure if necessary
    if 'Output' not in os.listdir('.'):
        os.system('mkdir %s' %('Output'))

    if Commands.comp:
        if (RootFileNoise.name.split('-')[0]+'_DAQ_COMP') not in os.listdir('Output'):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP'))

        if 'PickledNoise' not in os.listdir('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP'):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP' + '/PickledNoise'))

        if 'Histos'  not in os.listdir('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP'):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP' +'/Histos' ))

        if 'RawNoise' not in os.listdir('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP' +'/Histos'):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0]+'_DAQ_COMP' +'/Histos/RawNoise' ))

    else:
        if (RootFileNoise.name.split('-')[0]) not in os.listdir('Output'):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0]))
            
        if 'PickledNoise' not in os.listdir('Output/' + RootFileNoise.name.split('-')[0]):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0] + '/PickledNoise'))

        if 'Histos'  not in os.listdir('Output/' + RootFileNoise.name.split('-')[0]):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0] +'/Histos' ))

        if 'RawNoise' not in os.listdir('Output/' + RootFileNoise.name.split('-')[0] +'/Histos'):
            os.system('mkdir %s' %('Output/' + RootFileNoise.name.split('-')[0] +'/Histos/RawNoise' ))
            
    #Pickle
    if Commands.comp:
        RootFileNoise.PickleNoise('Output/'+RootFileNoise.name.split('-')[0]+'_DAQ_Comp' + '/PickledNoise/'+RootFileNoise.name.replace('.root','.pkl'))
    else:
        RootFileNoise.PickleNoise('Output/'+RootFileNoise.name.split('-')[0]+'/PickledNoise/'+RootFileNoise.name.replace('.root','.pkl'))
        
    #Add data to RootFileNoise.Histo
    histo =  RootFileNoise.IntializeHisto(0, 15,500)
    RootFileNoise.AddToHisto(histo,RootFileNoise.name.split('-')[1],RootFileNoise.name.split('-')[2])
                              
    #Draw Histo to canvas and save
    canvas = TCanvas(RootFileNoise.name.replace('.root',''),RootFileNoise.name.replace('.root',''),200,10,700,500)
    histo.Draw()



    #Save Image
    canvas.Print('Output/' + RootFileNoise.name.split('-')[0] +'/Histos/RawNoise/' + RootFileNoise.name.replace('.root','.png'))

    
                           

    

