#!/bin/env python

import pprint 
import dbsApi, dbsApiException
from xml.dom.minidom import parse, parseString


db = dbsApi.DbsApi()

def processDataset(dataset):
    # import pdb; pdb.set_trace()
    dataset = dataset.rstrip('\n')
    try:
        xmldict = db.listDatasetSummary(dataset)
        return xmldict
    except dbsApiException.DbsBadRequest:
        return None


class Table(object):
    def __init__(self, filename=None):
        self.lines = []
        if filename:
            ifile = open(filename)
            for line in ifile:
                info = self.parseLine(line)
                if info:
                    self.lines.append( info )

    def parseLine(self, line):
        line = line.lstrip('|')
        line = line.rstrip('|\n')
        if line=='':
            return None
        spl = line.split('|')
        if len(spl)!= 6:
            raise ValueError('bad line, need 6 fields: "{line}"'.format(line=line))
        info = dict(
            path = spl[0].strip(),
            nfiles = spl[1].strip(),
            nevts = spl[2].strip(),
            location = spl[3].strip(),
            person = spl[4].strip(),
            priority = spl[5].strip()
            )
        return info

    def addLine(self, path, nfiles, nevts, location, person='', priority=''):        
        info = dict(
            path = path.strip(),
            nfiles = nfiles.strip(),
            nevts = nevts.strip(),
            location = location.strip(),
            person = person.strip(),
            priority = priority.strip()
            )      
        self.lines.append( info )

    def absorb(self, other):
        for line in self.lines:
            # print line
            path = line['path']
            # matching lines in other
            molines = [l for l in other.lines if l['path']==path]
##             if path=='/VBF_HToTauTau_M-115_8TeV-powheg-pythia6/Summer12_DR53X-PU_S10_START53_V7A-v1/AODSIM':
##                 import pdb; pdb.set_trace()
            if len(molines)==0:
                continue
            elif len(molines)>1:
                raise ValueError('duplicate entry in other: ' + path)
            else:
                moline = molines[0]
                identical = ['path']
                for i in identical:
                    if line[i]!=moline[i]:
                        raise ValueError( str(line) + ' and ' + str(moline) + ' differ for at least one of the following fields: ' + str(identical))
                line['location'] = moline['location']
                line['person'] = moline['person']
                line['priority'] = moline['priority']
        
    def __str__(self):
        lines = [] 
        for l in self.lines:
            lines.append( '|{path} | {nfiles}| {nevts}| {location} | {person} | {priority} |'.format( path=l['path'], nfiles=l['nfiles'], nevts=l['nevts'], location=l['location'], person=l['person'], priority=l['priority']) )
        lines.sort()
        return '\n'.join(lines)
    

def printDataset(info):
    print '|', ' | '.join(info), '|  |  |'
    
if __name__ == '__main__':

    import sys
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = "%prog [options] <txt_file_list>\n"
    parser.add_option("-i","--input", dest="input",
                      default=None,
                      help="input table from the twiki")
    
    (options,args) = parser.parse_args()

    if len(args)!=1:
        print parser.usage()
        sys.exit(1)

    txtlist = open( args[0] )
    extable = Table( options.input )
    newtable = Table()
    
    cern = []
    grid = []
    skipped = []

    datasetsInTxt = []
    datasetsInInputTable = [line['path'] for line in extable.lines]
    for dataset in txtlist:
        dataset = dataset.rstrip()
        if dataset == '':
            continue
        datasetsInTxt.append(dataset)
        report = processDataset(dataset)
        if report is None:
            report = dict(
                path = dataset, 
                number_of_files = '-1',
                number_of_events = '-1'
                )
            skipped.append(dataset)
        location = 'GRID'
        if int(report['number_of_files']) < 0:
            location = '?'
        elif int(report['number_of_files'])>450:
            location = 'CERN'
        newtable.addLine(path = report['path'],
                         nfiles = report['number_of_files'],
                         nevts = report['number_of_events'],
                         location = location)

    cern.sort()
    grid.sort()

    set_datasetsInInputTable = set(datasetsInInputTable)
    set_datasetsInTxt = set(datasetsInTxt)

    print '-'*70
    print extable
    print '-'*70
    print newtable
    print '#'*70
    newtable.absorb( extable )
    print newtable
    print '#'*70
    paths = []
    for line in newtable.lines:
        paths.append(line['path'])
    for p in sorted(paths):
        print p
    print
    print
    print 'number of lines in text file    = ', len(datasetsInTxt)
    print 'number of lines in input table  = ', len(extable.lines) 
    print 'number of lines in output table = ', len(newtable.lines) 
    print 'datasets added in text file since last time:'
    pprint.pprint( set_datasetsInTxt - set_datasetsInInputTable )
    print 'datasets removed from text file since last time or added to the twiki(!):'
    pprint.pprint( set_datasetsInInputTable - set_datasetsInTxt)
    
