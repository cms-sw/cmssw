import json
import os 

class YRParser(object):
    def __init__(self,jsonFile):
        f=open(jsonFile)
        self.dict=json.load(f)

    def get(self,mass):
        return  (item for item in self.dict if item["mH"] == mass).next()


yrparser7TeV = YRParser( '/'.join( [os.environ['CMSSW_BASE'],
                                    'src/CMGTools/RootTools/python/yellowreport/YR_7TeV.json']))
yrparser8TeV = YRParser( '/'.join( [os.environ['CMSSW_BASE'],
                                    'src/CMGTools/RootTools/python/yellowreport/YR_8TeV.json']))
yrparser13TeV = YRParser( '/'.join( [os.environ['CMSSW_BASE'],
                                    'src/CMGTools/RootTools/python/yellowreport/YR_13TeV.json']))

if __name__ == '__main__':

    import sys
    mass = float(sys.argv[1])

    process = ['GGH', 'VBF', 'WH', 'ZH', 'TTH']
    
    print 'mass', mass 

    def printSigma(parser):
        tot = 0
        for p in process:
            sigma = parser.get(mass)[p]['sigma']
            print '\t', p, sigma
            tot += sigma
        print '\tTOTAL', tot
            
    print '7 TeV'
    printSigma(yrparser7TeV)
    print '8 TeV'
    printSigma(yrparser8TeV)
    print '13 TeV'
    printSigma(yrparser13TeV)
  
