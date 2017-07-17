from PhysicsTools.PythonAnalysis import *
from ROOT import *
# prepare the FWLite autoloading mechanism
gSystem.Load("libFWCoreFWLite.so")
FWLiteEnabler::enable()

# load the file with the generator output
theFile = TFile("generatorOutput.root")

events = theFile.Get("Events")

# Needed for SetAddress to work right
events.GetEntry()

# set the buffers for the branches you want to access
# 1) create a buffer
# 2) open the root branch
# 3) connect buffer and branch
# example: generator particles
source = edm.HepMCProduct()
sourceBranch = events.GetBranch(events.GetAlias("source"))
sourceBranch.SetAddress(source)

# now loop over the events
for index in all(events):
    
    # update all branches - the buffers are filled automatically
    # Hint: put all you branches in a list and loop over it
    sourceBranch.GetEntry(index)
    events.GetEntry(index,0)

    # do something with the data
    genEvent = source.GetEvent();
    print genEvent.event_number()
