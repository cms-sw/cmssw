from ROOT import TFile
from PhysicsTools.HeppyCore.statistics.tree import Tree
import os

FNAME="test_tree.root"

def create_tree(filename=FNAME, nentries=None):
    if not nentries: 
        if os.path.isfile(filename):
            #default number of entries, file exists
            return filename
        else: 
            nentries = 200
    nentries = int(nentries)
    outfile = TFile(filename, 'recreate')
    tree = Tree('test_tree', 'A test tree')
    tree.var('var1')
    for i in range(nentries):
        tree.fill('var1', i)
        tree.tree.Fill()
    outfile.Write()
    outfile.Close()
    return outfile.GetName()

def remove_tree(filename=FNAME):
    os.remove(filename)
    
if __name__ == '__main__':
    create_tree()

