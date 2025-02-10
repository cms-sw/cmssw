from PhysicsTools.PythonAnalysis import *
from ROOT import gSystem

# prepare the FWLite autoloading mechanism
gSystem.Load("libFWCoreFWLite.so")
ROOT.FWLiteEnabler.enable()

# access the event tree
events = EventTree("generatorOutput.root")

# event loop
for event in events:
    genEvent = event.VtxSmeared.GetEvent()
    print(genEvent)
