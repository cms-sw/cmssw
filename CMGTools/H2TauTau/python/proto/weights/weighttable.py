import os
import glob


class Range(object):

    def __init__(self, rangeStr):
        elems = rangeStr.split('<')
        self.varName = elems[1].lstrip().rstrip()
        self.min = float(elems[0])
        self.max = float(elems[2])

    def __str__(self):
        theStr = '{min:6.3f} < {varName} < {max:6.3f}'.format(
            min = self.min,
            varName = self.varName,
            max = self.max
            )
        return theStr

    def includes(self, value):
        return value >= self.min and value < self.max


class Value(object):

    def __init__(self, valueStr):
        # print valueStr
        self.value, self.err = map( float, valueStr.split('pm') )
        

    def __str__(self):
        theStr = '{value:6.3f} pm {err:6.3f}'.format(value=self.value, err=self.err)
        return theStr


class Weight(object):

    def __init__(self, line):
        self.line = line.rstrip('\n')
        info = self.line.split('*')
        values = []
        self.rangeEta = None
        self.rangePt = None
        for elem in info:
            try:
                # import pdb; pdb.set_trace()
                range = Range(elem)
                if range.varName.lower() == 'pt':
                    self.rangePt = range
                elif range.varName.lower() == 'eta':
                    self.rangeEta = range
                else:
                    raise ValueError( elem + ' is not a valid Range, variable unknown:' + range.varName )
            except:
                values.append( Value( elem) )
                # pass
                # is maybe a value? 
        self.effData = None
        self.effMC = None
        self.weight = None
        if len(values)==1:
            self.weight = values[0]
        elif len(values)==3:
            self.effMC = values[0]
            self.effData = values[1]
            self.weight = values[2]

    def includes(self, pt, eta):
        etaOk = True
        ptOk = True
        if self.rangeEta:
            etaOk = self.rangeEta.includes(eta)
        if self.rangePt:
            ptOk = self.rangePt.includes(pt)
        return etaOk and ptOk


    def __str__(self):
        theStrs = [ str(self.rangePt),
                    str(self.rangeEta) ]
        if self.effMC:
            theStrs.append( str(self.effMC) )
            theStrs.append( str(self.effData) )
        theStrs.append( str(self.weight) )
        return '\t*\t'.join(theStrs)


class WeightTable(list):
    
    def __init__(self, fileName):
        self.name = os.path.splitext( os.path.basename(fileName) )[0] 
        file = open(fileName)
        for line in file:
            if not line.strip():
                continue
            self.append( Weight(line) )

    def __str__(self):
        theStrs = [self.name]
        for weight in self:
            theStrs.append( str(weight) )
        return '\n'.join(theStrs)

    def weight(self, pt, eta):
        selectedWeights = []
        for weight in self:
            if weight.includes( pt, eta ):
                selectedWeights.append( weight )
        if len(selectedWeights) == 0:
            errmsg = []
            errmsg.append( 'no weight found in table ' + self.name )
            errmsg.append( str(self) )
            errmsg.append('for pt={pt}, eta={eta}'.format(pt=pt,
                                                          eta=eta))
            raise ValueError( '\n'.join(errmsg) )
        elif len(selectedWeights) == 1:
            return selectedWeights[0]
        else:
            errmsg = []
            errmsg.append( 'two possible weigt lines found in table ' + self.name )
            errmsg.extend( map(str, selectedWeights) )
            raise ValueError( '\n'.join(errmsg) )


# converting all text files matching this pattern into a python WeightTable object,
# and binding this object here with a name equal to the file name
# e.g. mu_iso_taumu_2012 => mu_iso_taumu_2012.txt
weightPat = os.environ['CMSSW_BASE'] + '/src/CMGTools/H2TauTau/python/proto/weights/*.txt'
txtFiles = glob.glob( weightPat )
for fname in sorted(txtFiles):
    wt = WeightTable( fname )
    locals()[wt.name] = wt
    print 'weighttable: loading', wt.name
    
    
if __name__ == '__main__':

    import sys

    # fnam = sys.argv[1]
    # wt = WeightTable( fnam )
    # print wt 
