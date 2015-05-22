
def load():
    #load the libaries needed
    from ROOT import gROOT,gSystem
    gSystem.Load("libCintex")
    gROOT.ProcessLine('ROOT::Cintex::Cintex::Enable();')
        
    #now the RootTools stuff
    gSystem.Load("libCMGToolsExternal")

load()
