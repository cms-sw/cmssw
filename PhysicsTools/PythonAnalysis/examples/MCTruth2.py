from cmstools import *
from ROOT import *

# prepare the FWLite autoloading mechanism
gSystem.Load("libFWCoreFWLite.so")
AutoLibraryLoader.enable()

# load the file with the generator output
theFile = TFile("generatorOutput.root")

# access the event tree
events = EventTree(theFile.Get("Events"))

# access the products inside the tree
# aliases can be used directly
sourceBranch = events.branch("source")

# loop over the events
for event in events:
    genEvent = sourceBranch().GetEvent()
    print genEvent
