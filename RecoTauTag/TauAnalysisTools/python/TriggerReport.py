#!/usr/bin/env python
'''
TriggerReport.py

Author: Evan K. Friis, UC Davis (evan.friis@cern.ch)

Utility to merge and manage multiple trigger reports from CMSSW log files.
See ./TriggerReport.py for usage

'''

import re
import itertools
import copy
import glob
import sys

def processBlock(input, startRegex, headerRegex, meatRegex, endRegex, meatHandler, debug = False):
    ''' Process a block of text and return the meat 

    Block is defined by a start, header, meat, and end regexes.
    Each line of meat is yielded as meatHandler(meat)

    '''

    inBlock = False
    for line in [rawline.strip() for rawline in input]:
        # check if we are currently processing a block
        if inBlock:               
            # skip column header if desired
            if headerRegex and headerRegex.match(line): 
                if debug: print "HEADER-----%s" % line
                continue
            # check if we have reached the end of the block
            if endRegex.match(line): 
                if debug: print "END--------%s" % line
                raise StopIteration
            # if we reach this point, we are parsing the interesting data
            meat = meatRegex.match(line)
            if not meat:
                raise ValueError, \
                        "Error parsing data!  Trying to match '%s' to regex patter '%s'" \
                        % (line, meatRegex.pattern)
            if debug: print "MEAT-------%s" % line
            yield meatHandler(meat)

        # currently not in block, see if this line starts it
        if startRegex.match(line):
            inBlock = True
            if debug: print "START------%s" % line

class Efficiency(object):
    " Base efficiency object for a path/module"
    def __init__(self, name = "", visited=0, run=0, passed=0, failed=0, trig=1, bit=0, error=0, **catchall):
        self.name         = name
        self.trig         = int(trig)
        self.bit          = int(bit)
        self.error        = int(error)
        self.visited      = int(visited)
        self.run          = int(run)
        self.passed       = int(passed)

    def efficiency(self):
        " Get efficiency (visited/passed)"
        return (self.visited > 0 and self.passed/float(self.visited) or 0)
   
    def absorb(self,other):
        ''' Add another efficiency into this one '''
        if other.name != self.name:
            raise AttributeError, \
                    "Attempint to add non-compatbile efficiency objects!  %s + %s" \
                    % (self.name, other.name)
        self.trig    += other.trig # ???
        self.bit     += other.bit #???
        self.error   += other.error
        self.visited += other.visited
        self.run     += other.run
        self.passed  += other.passed

class LogFileTriggerReport(object):
    " Get trigger report from an EDM logfile"
    def __init__(self,fileName):
        self.file = open(fileName, 'r')
        # load into memory the trig report lines
        self.trigReport = []
        # Define log file regexes
        self.grepper           = re.compile('^TrigReport')
        self.pathListFinder    = re.compile(
            'TrigReport ---------- Path   Summary ------------')
        self.endPathListFinder = re.compile(
            'TrigReport -------End-Path   Summary ------------')
        self.moduleSummaryFinder = re.compile(
            'TrigReport ---------- Module Summary ------------')
        self.headerFinder      = re.compile(
            'TrigReport  Trig Bit#        Run     Passed     Failed      Error Name')
        self.moduleSummaryHeader = re.compile(
            r'TrigReport    Visited        Run     Passed     Failed      Error Name')
        self.pathModuleSummaryHeader = re.compile(
            r'TrigReport  Trig Bit#    Visited     Passed     Failed      Error Name')
        self.eventSummaryFinder  = re.compile(
            r'TrigReport ---------- Event  Summary ------------')
        self.eventSummaryParser  = re.compile(
            r'TrigReport Events total = (?P<total>\d+) passed = (?P<passed>\d+) failed = (?P<failed>\d+)')
        self.pathModulesfinder = lambda path : re.compile(
            r'TrigReport ---------- Modules in Path:\s+%s\s+[-]+' % path)
        self.anyModuleSummaryFinder = re.compile('TrigReport [-]+ Module')
        self.nullRegex = re.compile(r'asdfasdfasdfasdfasdfasdfasdf') # hack to match nothing
        self.statsFinder = re.compile(r'''
                                      TrigReport\s+
                                      (?P<trig>\d+)\s+
                                      (?P<bit>\d+)\s+
                                      (?P<visited>\d+)\s+
                                      (?P<passed>\d+)\s+
                                      (?P<failed>\d+)\s+
                                      (?P<errror>\d+)\s+
                                      (?P<name>\w+)\s*$
                                      ''', re.VERBOSE)
        self.moduleSummaryStatsFinder = re.compile(r'''
                                      TrigReport\s+
                                      (?P<visited>\d+)\s+
                                      (?P<run>\d+)\s+
                                      (?P<passed>\d+)\s+
                                      (?P<failed>\d+)\s+
                                      (?P<errror>\d+)\s+
                                      (?P<name>\w+)\s*$
                                      ''', re.VERBOSE)
        
        # store in memory all lines of the file that begin w/ TrigReport
        for line in self.file:
            if self.grepper.match(line):
                self.trigReport.append(line)
        self.file.close()

    def paths(self):
        ''' Generator to return summary info for each path in the report '''
        handler = lambda match : PathReport(**match.groupdict())
        for x in processBlock(self.trigReport, 
                              self.pathListFinder,
                              self.headerFinder,
                              self.statsFinder,
                              self.endPathListFinder,
                              handler):
            yield x

    def endpaths(self):
        ''' Yields end path summaries in report '''
        handler = lambda match : PathReport(**match.groupdict())
        for x in processBlock(self.trigReport,
                              self.endPathListFinder,
                              self.headerFinder,
                              self.statsFinder,
                              self.anyModuleSummaryFinder,
                              handler):
            yield x

    def modulesFromSummary(self):
        ''' Yields all modules from the summary '''
        handler = lambda match : PathReport(**match.groupdict())
        for x in processBlock(self.trigReport,
                              self.moduleSummaryFinder,
                              self.moduleSummaryHeader,
                              self.moduleSummaryStatsFinder,
                              self.nullRegex,
                              handler):
            yield x

    def eventSummary(self):
        " Build an event summary from a match"
        handler = lambda match : (
            Efficiency(visited = match.group('total'), 
                       passed = match.group('passed'), 
                       failed = match.group('failed')))

    def modulesInPath(self, path):
        ''' Generator to return the summary information from the list of modules in a path '''
        pathModuleFinder = self.pathModulesfinder(path)
        handler = lambda match : Efficiency(**match.groupdict())
        for x in processBlock(self.trigReport,
                              pathModuleFinder,
                              self.pathModuleSummaryHeader,
                              self.statsFinder,
                              self.anyModuleSummaryFinder,
                              handler):
            yield x

class PathReport(Efficiency):
    " Hold informations about modules in a path "
    def __init__(self, **kwargs):
        super(PathReport, self).__init__(**kwargs)
        self.modules     = [] # need to preserve order

    def moduleEfficiencies(self):
        " Yield an Efficiency object for each module in the path"
        if not len(self.modules):
            return
        cumulativeEfficiency = 1.0
        for module in self.modules:
            output = copy.copy(module) # make a shallow copy
            cumulativeEfficiency *= output.efficiency() # scale the cumulative efficiency
            output.cumulative = cumulativeEfficiency
            yield output

    def efficiencyReport(self, printAll = True, csv=False, showCumulativeEfficiencies=True):
        " Print a helpful report on moudles in this path"
        print " * Efficiency for modules in path %s - total efficiency: (%i/%i)"\
                % (self.name, self.passed, self.visited)
        maxModuleNameLength = \
                lambda max, module : max > len(module.name) and max or len(module.name)
        maxLength = reduce(maxModuleNameLength, self.modules, 0)
        formatString = " * * %-"+str(maxLength+2)+"s %15s  %15s %10s %10s"
        #check if want comma seperated
        if csv: 
            formatString = "%s,%s,%s,%s,%s,%s"

        print formatString % ("Name", "Rel Eff", "Total Eff", "Passed", "Total")
        for module in self.moduleEfficiencies():
            if printAll or module.efficiency() < 1.:
                percent = lambda eff : "%0.2f%%" % (eff*100.)
                cumEfficiency = -1
                if showCumulativeEfficiencies:
                    cumEfficiency = percent(module.cumulative)
                print formatString % (
                    module.name, 
                    percent(module.efficiency()), 
                    cumEfficiency, 
                    module.passed, module.visited) 

    def absorb(self, other):
        ''' Add another path reports results to this one '''
        Efficiency.absorb(self, other)  # add the overal summary numbers
        for localModule, otherModule in itertools.izip(self.modules, other.modules):
            localModule.absorb(otherModule)

class TriggerReport(PathReport):
    " Object representing a trigger report "
    def __init__(self, **kwargs):
        super(TriggerReport, self).__init__(**kwargs)
        self.paths = {}

    def addLogFile(self,files, quiet=True):
        " Parse log files and add to report "
        files = glob.glob(files)
        for file in files:
            if not quiet:
                print "Examining file: %s" % file
            logfile = LogFileTriggerReport(file)

            # Absorb the module summary into this trigger report
            class DummyTrigReport(object):
                def __init__(self, name):
                    self.name = name
                    self.trig = -1
                    self.bit = -1
                    self.visited = -1
                    self.passed = -1
                    self.run = -1
                    self.error = -1

            dummy = DummyTrigReport(self.name)
            dummy.modules = [module for module in logfile.modulesFromSummary()]
            if not self.modules:
                self.modules = dummy.modules
            else:
                self.absorb(dummy)

            # Absorb all the sub paths
            paths = [path for path in logfile.paths()]
            #populate the sub modules of each path, and absorb it into 
            # a previous copy if it exists
            for path in paths:
                path.modules = [
                    module for module in logfile.modulesInPath(path.name)]
                #check if we haven't added this path
                if path.name not in self.paths.keys():
                    if not quiet:
                        print "Adding path %s with %i sub-modules" % (path.name, len(path.modules))
                    self.paths[path.name] = path
                # if it already exists, merge it w/ the existing one
                else:
                    self.paths[path.name].absorb(path)

if __name__ == "__main__":
    from optparse import OptionParser                                                                                                                                                                       
    usage = \
'''%prog [options] logfile1 [logfile2, ... logfileN]

Example: ./%prog -p "my_path" crab_output/res/CMSSW_*.stderr
merges all the trigger reports in the log files listed, then prints out
the step by step module efficiencies for path "my_path"
'''
    parser = OptionParser(usage)
    parser.add_option("-p", "--path", help="Print efficiencies for path")
    parser.add_option("-s", "--summary", help="Print out the module summary", action="store_true")
    parser.add_option("-v", "--verbose", help="Print modules for which efficiency = 100%", action="store_true")
    parser.add_option("-l", "--list", help="List available paths", action="store_true")
    parser.add_option("-d", "--debug", help="Debug output when parsing files", action="store_true")

    (options, args) = parser.parse_args()

    # Require at least one option passed
    if len(args) == 0:
        parser.print_help()
        sys.exit()

    # If no path is specified, set list to true
    if options.path is None:
        options.list = True

    # Load files
    report = TriggerReport(name="report")
    for file in args:
        quiet = True
        if options.debug: quiet = False
        report.addLogFile(file, quiet)

    if options.list is not None:
        print "Available paths:"
        for i, (path, pathReport) in enumerate(report.paths.iteritems()):
            print " %3i: %-30s  Passed: %-10i Visited: %-10i Efficiency: %-10.3f" % \
                    (i, path, pathReport.passed, pathReport.visited, pathReport.efficiency())

    if options.verbose is None:
        options.verbose = False

    if options.summary is not None:
        report.efficiencyReport(printAll = options.verbose, showCumulativeEfficiencies=False)

    if options.path is not None:
        try:
            report.paths[options.path].efficiencyReport(printAll=options.verbose)
        except KeyError:
            print "ERROR: path %s not found in list of available paths!" % options.path
            sys.exit()
