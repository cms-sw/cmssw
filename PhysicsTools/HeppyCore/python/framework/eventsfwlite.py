from DataFormats.FWLite import Events as FWLiteEvents

from ROOT import gROOT, gSystem, AutoLibraryLoader

print "Loading FW Lite"
gSystem.Load("libFWCoreFWLite");
gROOT.ProcessLine('FWLiteEnabler::enable();')

gSystem.Load("libFWCoreFWLite");
gSystem.Load("libDataFormatsPatCandidates");

from ROOT import gInterpreter
gInterpreter.ProcessLine("using namespace reco;")
gInterpreter.ProcessLine("using edm::refhelper::FindUsingAdvance;")

class Events(object):
    def __init__(self, files, tree_name,  options=None):
	if options is not None :
		if not hasattr(options,"inputFiles"):
		 	options.inputFiles=files
		if not hasattr(options,"maxEvents"):
			options.maxEvents = 0	
		if not hasattr(options,"secondaryInputFiles"):
			options.secondaryInputFiles = []
	        self.events = FWLiteEvents(options=options)
	else :
	        self.events = FWLiteEvents(files)

    def __len__(self):
        return self.events.size()

    def __getattr__(self, key):
        return getattr(self.events, key)

    def __getitem__(self, iEv):
        self.events.to(iEv)
        return self

