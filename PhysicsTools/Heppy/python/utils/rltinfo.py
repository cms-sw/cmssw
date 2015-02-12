from ROOT import TFile
from PhysicsTools.HeppyCore.statistics.tree import Tree as Tree

class MyInteger(object):
    def __init__(self, integer ):
        self.integer = integer
    def __add__(self, other):
        if hasattr(other, 'integer'):
            self.integer += other.integer
        else:
            self.integer += other 
        return self
    def __str__(self):
        return str(self.integer)
    

class RLTInfo( object ):
    def __init__(self):
        self.dict = {}

    def add(self, trigger, run, lumi):
        nEv = self.dict.setdefault( (trigger, run, lumi), MyInteger(0) )
        nEv += 1

    def __str__(self):
        lines = []
        for rlt, count in self.dict.iteritems():
            lines.append( ': '.join( [str(rlt), str(count)] ))
        return '\n'.join(lines)

    def write(self, dirName, fileName='RLTInfo.root'):
        f = TFile('/'.join( [dirName, fileName]), 'RECREATE')
        t = Tree('RLTInfo','HLT/Run/Lumi information')
        t.var('run', int )
        t.var('lumi', int )
        t.var('counts', int )
        t.var('trigger', int )
        for rlt, count in self.dict.iteritems():
            t.fill('run', rlt[1])
            t.fill('lumi', rlt[2])
            t.fill( 'counts', count.integer)
            t.tree.Fill()
        f.Write()
        f.Close()
        
if __name__ == '__main__':

    rltinfo = RLTInfo()
    rltinfo.add('HLT1', 128, 1)
    rltinfo.add('HLT1', 128, 1)
    rltinfo.add('HLT1', 128, 2)
    rltinfo.add('HLT1', 129, 2)
    rltinfo.add('HLT2', 129, 2)

    for rlt, count in rltinfo.dict.iteritems():
        print rlt, count

    rltinfo.write('.')
