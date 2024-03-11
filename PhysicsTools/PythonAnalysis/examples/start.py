# startup commands for interactive use

import ROOT
from ROOT import gSystem
from PhysicsTools.PythonAnalysis import *

gSystem.Load("libFWCoreFWLite.so")
ROOT.FWLiteEnabler.enable()

# foo bar baz
