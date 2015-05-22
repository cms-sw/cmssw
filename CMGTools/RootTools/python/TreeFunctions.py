from ROOT import TTree, gDirectory

def setEventList( tree, cut=None ):
    '''to undo, call tree.SetEventList(0)'''
    print 'now browsing the full tree... might take a while, but drawing will then be faster!'
    tree.Draw('>>pyplus', cut)
    pyplus = gDirectory.Get('pyplus')
    pyplus.SetReapplyCut(True)
    tree.SetEventList(pyplus)
    
def scan( tree, cut=None ):
    '''this scan can be used in input to the scanToVEventRange.py script'''
    out = tree.Scan('EventAuxiliary.id().run():EventAuxiliary.id().luminosityBlock():EventAuxiliary.id().event():mht.obj.pt():met.obj.pt():ht.obj.sumEt()', cut, 'colsize=14')

