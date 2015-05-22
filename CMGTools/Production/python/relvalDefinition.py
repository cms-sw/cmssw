
import re
pattern = re.compile('/(\S+)/(CMSSW_\d+_\d+_\d+(_pre\d+)?)-((MC|START)\d*(_\d+)?_V\S+).*-v\d+/.*')

def datasetToRelval( dataset ):
    # print dataset
    m = pattern.match(dataset)
    if m!= None:
        # print m.group(1), m.group(2), m.group(4)
        return (m.group(1), m.group(2), m.group(4) )
    else:
        print 'does not match!'


class relvalDefinition:
    def __init__( self, dataset):
        (relval, cmssw, tag) = datasetToRelval( dataset )
        self.cmssw = cmssw
        self.relval = relval
        self.tag = tag
        self.dataset = dataset

    def __str__( self ):
        outstr =  str(self.dataset) + ':\t' + self.id()
        return outstr

    def id(self):
        return str(self.relval) + '/' + str( self.cmssw ) + '/' + str( self.tag ) 

    
class relvalList:
    def __init__( self ):
        self.list = []
    
    def add(self, dataset):
        self.list.append( relvalDefinition(dataset) )


    def __str__(self):
        
        str = 'relvals: \n'
        for relval in self.list:
            str += relval.__str__() + '\n'
        return str

