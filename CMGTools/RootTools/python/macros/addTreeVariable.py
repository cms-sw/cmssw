import numpy
import math
from CMGTools.RootTools.PyRoot import *

def addShiftVariables(fileName, treeName=None):
    file = TFile(fileName)
    tree = None
    if treeName is not None:
        tree = file.Get(treeName)
    else:
        for key in file.GetListOfKeys():
            obj = file.Get(key.GetName())
            if type(obj) is TTree:
                tree = obj
                print 'found tree', key.GetName()
                break
    if tree == None:
        return False

    outfile = TFile(fileName.replace('.root', '_new.root'), 'recreate')
    outtree = tree.CloneTree(0)

    visMassUp = numpy.zeros(1,float)
    visMassUpBr = outtree.Branch('visMassUp', visMassUp, 'visMassUp/D')
    svfitMassUp = numpy.zeros(1,float)
    svfitMassUpBr = outtree.Branch('svfitMassUp', svfitMassUp, 'svfitMassUp/D')

    visMassDown = numpy.zeros(1,float)
    visMassDownBr = outtree.Branch('visMassDown', visMassDown, 'visMassDown/D')
    svfitMassDown = numpy.zeros(1,float)
    svfitMassDownBr = outtree.Branch('svfitMassDown', svfitMassDown, 'svfitMassDown/D')

    upFactor = math.sqrt(1.03)
    downFactor = math.sqrt(0.97)

    for index in range(0, tree.GetEntries()):
        tree.GetEntry(index)
        # import pdb; pdb.set_trace()
        visMassUp[0] = tree.visMass * upFactor
        svfitMassUp[0] = tree.svfitMass * upFactor
        visMassDown[0] = tree.visMass * downFactor
        svfitMassDown[0] = tree.svfitMass * downFactor
        outtree.Fill()
    outtree.AutoSave()
    outfile.Close()


if __name__ == '__main__':

    import sys

    files = sys.argv[1:]

    for file in files:
        print 'processing', file
        addShiftVariables( file )

