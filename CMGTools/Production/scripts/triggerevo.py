import pprint
import copy
import re

class Menu(object):
    def __init__(self, header):
        self.header = header
        self.datasets = {}
        self.runs = []

    def __str__(self):
        tmp = [ str(self.header) ]
        tmp.append( ','.join( map(str, self.runs) ) )
        data = ['\t{data}'.format(data=data) for data in self.datasets.values() ]
        tmp.extend( data )
        return '\n'.join( tmp )

        
class Dataset(object):
    def __init__(self, header):
        header  = header.rstrip('\n')
        self.header = header
        self.name = header.split()[1]
        self.paths = {}

    def __str__(self):
        tmp = [ str(self.header) ]
        data = ['\t\t{data}'.format(data=data) for data in self.paths.values() ]
        tmp.extend( data )
        return '\n'.join( tmp )


class HLTPath(object):
    def __init__(self, line):
        self.line = line.lstrip().rstrip('\n')
        data = self.line.split()
        self.name = data.pop(0)
        seedinfo = []
        self.prescales = []
        for field in data:
            try:
                prescale = int(field)
                self.prescales.append( prescale )
            except ValueError:
                seedinfo.append( field )
        self.l1seed = ' '.join( seedinfo )

    def isPrescaled(self):
        if self.prescales == [1]*len(self.prescales):
            return False
        else:
            return True
        
    def __str__(self):
        return '{presc}, {hlt}, {l1}, {scales}'.format(
            # line=self.line,
            presc=self.isPrescaled(),
            hlt=self.name,
            l1=self.l1seed,
            scales=str(self.prescales))
        
class MenuHeader(object):
    def __init__(self, headerLine):
        data = headerLine.lstrip('/').split('/')
        self.headerline = headerLine
        self.data = data
        self.period = data[2]
        self.lumi = float(data[3])
        self.version = data[4]
        self.hltversion = data[6]

    def __str__(self):
        return self.headerline

def parseInputFile( fileName, datasets , nMenus=999999):
    dataFile = open( fileName )
    # [ (line.split()[0], line) for line in dataFile]
    # pprint.pprint( data[:100] )
    currentMenu = None
    currentDataset = None
    menus = []
    runList = False
    for line in dataFile:
        # print line
        line = line.rstrip('\n')
        if runList:
            currentMenu.runs = map(int, line.split(','))
            # print currentMenu.runs
            # import pdb
            # pdb.set_trace()
            runList = False
        elif line.startswith('/cdaq'):
            if len(menus) == nMenus:
                break
            header = MenuHeader(line)
            currentMenu = Menu( header )
            menus.append( currentMenu )
            runList = True
        elif line.lstrip().startswith('dataset'):
            dataset = Dataset(line)
            if dataset.name in datasets:
                currentDataset = Dataset(line) 
                currentMenu.datasets[ currentDataset.name ] = currentDataset
            else:
                currentDataset = None
        elif currentDataset is not None and line.lstrip().startswith('HLT'):
            path = HLTPath( line )
            currentDataset.paths[ path.name ] = path        
    return menus


def findUnprescaledRange( pathName, datasetName, menus ):
    runs = []
    unprescaledMenus = []
    for menu in menus:
        dataset = menu.datasets[ datasetName ]
        path = dataset.paths.get( pathName, None ) 
        if path is not None and not path.isPrescaled():
            runs.extend( menu.runs )
            unprescaledMenus.append( menu )
    return (runs, unprescaledMenus)


if __name__ == '__main__':

    import sys

    dataFile = 'triggerEvolution_all.txt'
    datasets = ['TauPlusX']
    nMenus = 10000
    menus = parseInputFile( dataFile, datasets, nMenus)
    #    for menu in menus:
    #        print menu

    (runs, unprescaledMenus) = findUnprescaledRange( sys.argv[1], 'TauPlusX', menus) 
