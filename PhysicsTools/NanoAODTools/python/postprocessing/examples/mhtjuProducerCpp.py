# Example of calling a C++ helper from a Module.
#
# Run with:
# nano_postproc.py outDir /eos/cms/store/user/andrey/f.root -I PhysicsTools.NanoAODTools.postprocessing.examples.mhtjuProducerCpp mhtju

from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
import ROOT
import os
ROOT.PyConfig.IgnoreCommandLineOptions = True


# MHT producer, unclean jets only (no lepton overlap cleaning, no jet selection)
class mhtjuProducerCpp(Module):
    def __init__(self):
        base = os.getenv("NANOAODTOOLS_BASE")
        if base:
            # Running in standalone mode: compile the C++ helper
            if "/MhtjuProducerCppWorker_cc.so" not in ROOT.gSystem.GetLibraries():
                print("Load C++ MhtjuProducerCppWorker worker module")
                ROOT.gROOT.ProcessLine(
                    ".L %s/test/examples/MhtjuProducerCppWorker.cc+O" % base)
        else:
            # Load the helper from the CMSSW compiled. This is not required if
            # dictionaries for the helper are generated with classes_def.xml and
            # classes.h
            base = "%s/src/PhysicsTools/NanoAODTools" % os.getenv("CMSSW_BASE")
            ROOT.gSystem.Load("libPhysicsToolsNanoAODToolsTest.so")
            ROOT.gROOT.ProcessLine(".L %s/test/examples/MhtjuProducerCppWorker.h" % base)
        self.worker = ROOT.MhtjuProducerCppWorker()
        pass

    def beginJob(self):
        pass

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.initReaders(inputTree)  # initReaders must be called in beginFile
        self.out = wrappedOutputTree
        self.out.branch("MHTju_pt", "F")
        self.out.branch("MHTju_phi", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    # this function gets the pointers to Value and ArrayReaders and sets
    # them in the C++ worker class
    def initReaders(self, tree):
        self.nJet = tree.valueReader("nJet")
        self.Jet_pt = tree.arrayReader("Jet_pt")
        self.Jet_phi = tree.arrayReader("Jet_phi")
        self.worker.setJets(self.nJet, self.Jet_pt, self.Jet_phi)
        # self._ttreereaderversion must be set AFTER all calls to
        # tree.valueReader or tree.arrayReader
        self._ttreereaderversion = tree._ttreereaderversion

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail,
        go to next event)"""

        # do this check at every event, as other modules might have read
        # further branches
        if event._tree._ttreereaderversion > self._ttreereaderversion:
            self.initReaders(event._tree)
        # do NOT access other branches in python between the check/call to
        # initReaders and the call to C++ worker code
        output = self.worker.getHT()
        
        self.out.fillBranch("MHTju_pt", output[0])
        self.out.fillBranch("MHTju_phi", -output[1])  # note the minus
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid
# having them loaded when not needed

mhtju = lambda: mhtjuProducerCpp()
