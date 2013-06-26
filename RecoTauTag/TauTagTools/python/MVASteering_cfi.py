"""
        MVASteering.py
        Author: Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)

        Define the MVA configurations (ie TaNC) to be used in training/testing
                - which neural net 
                - which algorithms (shrinkingConePFTauDecayModeProducer, etc)
        Define locations of train/test ROOT files
"""

import sys
import os
# Get CMSSW base
try:
   Project_Area = os.environ["CMSSW_BASE"]
except KeyError:
   print "$CMSSW_BASE enviroment variable not set!  Please run eval `scramv1 ru -[c]sh`"
   sys.exit(1)

import FWCore.ParameterSet.Config as cms
import glob

# Get defintions (w/ decay mode mapping) from the python defintion file
from RecoTauTag.TauTagTools.TauMVAConfigurations_cfi import *

#######  USER PARAMETERS  #######################################
#################################################################

# Define lists of neural nets corresponding to a total configuration
# The Neural net objects used here (SingleNet, OneProngNoPiZero, etc) must be defined in
# the TauMVAConfigurations_cfi.py found in ../python
MVACollections = {}


# Use the Tau neural classifier configuration
MVACollections['TaNC'] = TaNC.value()  # <--- defined in TauMVAConfigurations_cfi.py

# non isolated, single net only
# MVACollections['SingleNet'] = SingleNetBasedTauID.value()       

# isolation applied, neural net for each decay mode
#MVACollections['MultiNetIso'] = MultiNetIso.value()

# isolation applied, single neural net 
#MVACollections['SingleNetIso'] = [SingleNetIso]

# For training/evaluating on an isolated sample, define the isolated criteria here
IsolationCutForTraining = "Alt$(ChargedOutlierPt[0], 0) < 1.0 && Alt$(NeutralOutlierPt[0], 0) < 1.5" #no tracks above 1 GeV, no gammas above 1.5  GeV

#Define the PFRecoTauDecayMode source to use (in the case of more than one separate directories will be created for each training sample)
myTauAlgorithms = ["shrinkingConePFTauDecayModeProducer"]

# If true, output will be weighted such that the signal and background distributions have the same
#  Pt-Eta distribution.  To generate the weights file, run python BuildWeights.py
#UseWeights = True
UseWeights = True

# If this is true, the weighted pt-eta distributions for each decay mode will be the same.
#  otherwise, the pt-eta distribution of the entire training sample (post DM preselection)
#  will be used.
WeightByIndividualDecayMode = False

# If true, single isolatated charged pions and three prongs
#  with charge  +-3 will not be included in the training sample
ExcludePrepassAndPrefail = False

# If greater than zero, require that either the lead track or lead pion have pt greater 
#  than the value supplied
RequireLeadPionPt = 5.0
LeadPionRequirementString = "( (MainTrackPt > %f && MainTrackAngle < 0.1) || (Alt$(TrackPt[0],0) > %f && Alt$(TrackAngle[0], 20) < 0.1) || (Alt$(PiZeroPt[0], 0) > %f && Alt$(PiZeroAngle[0], 20) < 0.1) )" % (RequireLeadPionPt,RequireLeadPionPt,RequireLeadPionPt)

"""
Example of multiple algorithms
myTauAlgorithms = ["pfTauDecayModeHighEfficiency",
                   "pfTauDecayModeInsideOut"]
"""

# define locations of signal/background root files
TauTagToolsWorkingDirectory = os.path.join(Project_Area, "src/RecoTauTag/TauTagTools")
SignalRootDir               = os.path.join(TauTagToolsWorkingDirectory, "test", "ztt")
BackgroundRootDir           = os.path.join(TauTagToolsWorkingDirectory, "test", "qcd")

#Globs to get files for training and evaluation.  If you want to ensure different sets, you can do something like
# add a requirement such as *[0123].root for training and *[4].root.  (files not ending in four used for trianing, ending in four used for testing)
SignalFileTrainingGlob     = "%s/*[012356789].root" % SignalRootDir
BackgroundFileTrainingGlob = "%s/*[012356789].root" % BackgroundRootDir
#SignalFileTrainingGlob     = "%s/*[0].root" % SignalRootDir
#BackgroundFileTrainingGlob = "%s/*[0].root" % BackgroundRootDir

SignalFileTestingGlob     = "%s/*4.root" % SignalRootDir
BackgroundFileTestingGlob = "%s/*4.root" % BackgroundRootDir

#################################################################
#####  DO NOT MODIFY BELOW THIS LINE (experts only) #############
#################################################################

def GetTrainingFile(computerName, anAlgo):
   return os.path.join(TauTagToolsWorkingDirectory, "test", "TrainDir_%s_%s" % (computerName, anAlgo), "%s.mva" % computerName)

#Find the unique mva types to train
listOfMVANames = {}
for name, mvaCollection in MVACollections.iteritems():
   for _mva in mvaCollection:
      name = _mva.computerName.value()
      if not name in listOfMVANames:
         listOfMVANames[name] = _mva

myModules = []
for name, _mva in listOfMVANames.iteritems():
   myModules.append(_mva)

SignalTrainFiles         = glob.glob(SignalFileTrainingGlob)
BackgroundTrainFiles     = glob.glob(BackgroundFileTrainingGlob)

SignalTestingFiles         = glob.glob(SignalFileTestingGlob)
BackgroundTestingFiles     = glob.glob(BackgroundFileTestingGlob)

# Catch dumb errors before we begin
def EverythingInItsRightPlace():
   if not len(SignalTrainFiles) or not len(BackgroundTrainFiles) or not len(SignalTestingFiles) or not len(BackgroundTestingFiles):
      raise IOError, "The signal/background root file training/testing file list is empty! Check the SignalFileTrainingGlob etc. in MVASteering.py"

   # Ensure that we have all the necessary XML files 
   for aModule in myModules:
      computerName = aModule.computerName.value() #conver to python string
      xmlFileLoc   = os.path.join(TauTagToolsWorkingDirectory, "xml", "%s.xml" % computerName)
      if not os.path.exists(xmlFileLoc):
         raise IOError, "Can't find xml configuration file for %s - please check that %s exists!" % (computerName, xmlFileLoc)

   if not os.path.exists(SignalRootDir):
      raise IOError, "Signal root file directory (%s) does not exist! Have you created the MVA raw training data?" % SignalRootDir
   if not os.path.exists(BackgroundRootDir):
      raise IOError, "Background root file directory (%s) does not exist! Have you created the MVA raw training data?" % BackgroundRootDir
