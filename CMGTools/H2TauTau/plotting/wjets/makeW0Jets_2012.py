import copy
from CMGTools.RootTools.PyRoot import *
from CMGTools.RootTools.statistics.TreeNumpy import *

files = []

class Component(object):
    def __init__(self, name, numberForNaming = 99):
        self.name = name.rstrip('/')
        self.tree = None
        self.numberForNaming = numberForNaming
        self.attachTree()

    def attachTree(self):
        fileName = '{name}/H2TauTauTreeProducerTauEle/H2TauTauTreeProducerTauEle_tree.root'.format(name=self.name)
        treeName = 'H2TauTauTreeProducerTauEle'
        self.file = TFile(fileName)
        self.tree = self.file.Get(treeName)
        self.tree.SetName('H2TauTauTreeProducerTauEle_{0:d}'.format(self.numberForNaming))


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    
class H2TauTauSoup(TreeNumpy):

    def __init__(self, name, title):
        super(H2TauTauSoup, self).__init__(name,title)



# .... .... .... .... .... .... .... .... .... .... .... .... ....

    
    def importEntries(self, comp, nEntries=-1):
        print 'importing', comp.name
        tree = comp.tree
        for index, ie in enumerate(tree):
            if index%1000==0: print 'entry:', index
            skip = True
            for varName in self.vars:
                if not hasattr(ie, varName):
                    continue
                val = getattr(ie, varName)
                if varName == 'NUP':
                    nJets = int(val-5)
                    if nJets == 0 :
                        skip = False
            if skip == False :
                for varName in self.vars:
                    if not hasattr(ie, varName):
                        continue
                    val = getattr(ie, varName)
                    self.fill(varName, val)
                self.tree.Fill()
                skip = True # this should be useless and harmless
            if nEntries>0 and index+1>=nEntries:
                return 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    
if __name__ == '__main__':

    # import sys
    import imp

    args = sys.argv[1:]

##     anaDir = args[0]
##     cfgFileName = args[1]
##     file = open( cfgFileName, 'r' )
##     cfg = imp.load_source( 'cfg', cfgFileName, file)

# call the function from the folder that contains the components
# arguments are: the inclusive component, the exclusive components in order

    numberForNaming = 0
    incComp = Component( args[0], numberForNaming )
    numberForNaming = numberForNaming + 1  
    print 'Inclusive WJets sample: ', incComp.name

    W0jetsFile = TFile('W0jets.root','recreate')
    W0jets = H2TauTauSoup('H2TauTauTreeProducerTauEle', 'H2TauTauTreeProducerTauEle')
    W0jets.copyStructure( incComp.tree )
    W0jets.importEntries (incComp)

    W0jetsFile.Write()
    W0jetsFile.Close()
