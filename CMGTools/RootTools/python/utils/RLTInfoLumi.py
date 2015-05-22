import os
import sys
import json
import pprint
import re
from ROOT import TFile, TTree

class RLTInfoLumi(object):
    def __init__(self, inputRootFileName, treename):
        self.file = TFile(inputRootFileName)
        self.tree = self.file.Get(treename)
        self.rawDict = {}
        for ie in self.tree:
            # print ie.run
            self.rawDict.setdefault( str(ie.run), [] ).append(ie.lumi)
        map(list.sort, self.rawDict.values() )
        # pprint.pprint( self.rawDict )
        # now the list of lumi is sorted for each run.
        # converting this list to the std lumi range definition
        self.compactDict = {}
        for key, lumis in self.rawDict.iteritems():
            min = -1
            max = -1
            last = -1
            # print lumis
            # import pdb; pdb.set_trace()
            ranges = []
            for lumi in lumis:
                if min==-1:
                    min = lumi
                    max = lumi
                elif lumi-last>1:
                    max = last
                    ranges.append([min, max])
                    min = lumi
                else:
                    max = lumi
                last = lumi
            max = last
            ranges.append([min,max])
            self.compactDict[key] = ranges
            
    def writeJson(self, outputJsonFileName=None ):
        if outputJsonFileName is None:
            outputJsonFileName = self.file.GetName().replace('.root','.json')
        jstr = json.dumps( self.compactDict )
        # print jstr
        ofile = open(outputJsonFileName, 'w')
        ofile.write( jstr )
        ofile.close()        

    def computeLumi(self, lumiCalc, inputJsonFileName=None):
        if inputJsonFileName is None:
            inputJsonFileName = self.file.GetName().replace('.root','.json')
        outputLumiFileName = inputJsonFileName.replace('.json','.lumi')
##         if lumiCalc is None:
##             lumiCalc = 'pixelLumiCalc.py'
        cmd = [lumiCalc, 'overview -i',
               inputJsonFileName, '>',
               outputLumiFileName]
        cmds = ' '.join( cmd )
        print cmds
        os.system( cmds )
        lumiFile = open(outputLumiFileName)
        self.sumdlum = 0
        self.sumrlum = 0
        pattern = re.compile( '\w\((\S+)\)' )
        dunit = None
        runit = None
        for line in lumiFile:
            spl = line.split('|')
            # print spl
            if len(spl)==6 and spl[0]=='' and spl[5]=='\n':
                # print line
                try:
                    self.sumdlum = float(spl[2]) 
                    self.sumrlum = float(spl[4]) 
                except ValueError:
                    dunit = pattern.search( spl[2] ).group(1)
                    runit = pattern.search( spl[4] ).group(1)
        if dunit == '/nb':
            self.sumdlum /= 1000.
        elif dunit == '/ub':
            self.sumdlum /= 1e6
        elif dunit == '/fb':
            self.sumdlum *= 1000.
        elif dunit == '/pb':
            pass
        else:
            raise ValueError('Unrecognized unit! '+dunit)
        if runit == '/nb':
            self.sumrlum /= 1000.
        elif runit == '/ub':
            self.sumrlum /= 1e6
        elif runit == '/fb':
            self.sumrlum *= 1000.
        elif runit == '/pb':
            pass
        else:
            raise ValueError('Unrecognized unit! '+dunit)
        
        lumiFile.close()
        print 'luminosity:',inputJsonFileName, self.sumdlum, '(delivered /pb)', self.sumrlum, '(recorded /pb)' 
        
if __name__ == '__main__':

    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog [options] <RLT root file>
    """

    parser.add_option("-l", "--lumicalc", dest="lumicalc",
                      default='pixelLumiCalc.py',
                      help='Lumi calc command (e.g. lumiCalc2.py, pixelLumiCalc.py)')
    parser.add_option("-t", "--treename", dest="treename",
                      default='RLTInfo',
                      help='name of the RLT tree')

    (options,args) = parser.parse_args()

    rltlum = RLTInfoLumi(sys.argv[1], options.treename)
    # rltlum.tree.Print()
    # import pprint
    # pprint.pprint( rltlum.rawDict )
    # pprint.pprint( rltlum.compactDict )
    
    rltlum.writeJson()
    rltlum.computeLumi(options.lumicalc)
