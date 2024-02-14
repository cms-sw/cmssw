#! /usr/bin/env python3

from builtins import zip
from builtins import object
from past.utils import old_div
from builtins import range
import sys
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
import array
import ROOT
import math

sepRE      = re.compile (r'[\s,;:]+')
nonSpaceRE = re.compile (r'\S')

##########################
## #################### ##
## ## LumiInfo Class ## ##
## #################### ##
##########################

class LumiInfo (object):

    lastSingleXingRun = 136175
    lumiSectionLength = 23.310779

    def __init__ (self, line):
        self.totInstLum  = 0.
        self.aveInstLum  = 0.
        self.numXings    = 0
        self.instLums    = []
        self.events      = []
        self.xingInfo    = False
        self.badXingInfo = False
        pieces = sepRE.split (line.strip())
        size = len (pieces)
        if size % 2:
            raise RuntimeError("Odd number of pieces")
        if size < 4:
            raise RuntimeError("Not enough pieces")
        try:
            self.run       = int   (pieces[0])
            self.lumi      = int   (pieces[1])
            self.delivered = float (pieces[2])
            self.recorded  = float (pieces[3])
        except:
            raise RuntimeError("Pieces not right format")
        if size > 4:
            try:
                for xing, lum in zip (pieces[4::2],pieces[5::2]):
                    xing = int   (xing)
                    lum  = float (lum)
                    self.instLums.append( (xing, lum) )
                    self.totInstLum += lum
                    self.numXings += 1
            except:
                raise RuntimeError("Inst Lumi Info malformed")
            self.aveInstLum = old_div(self.totInstLum, (self.numXings))
            self.xingInfo   = True
        self.key       = (self.run, self.lumi)
        self.keyString = self.key.__str__()


    def fixXingInfo (self):
        if self.numXings:
            # You shouldn't try and fix an event if it already has
            # xing information.
            raise RuntimeError("This event %s already has Xing information" \
                  % self.keyString)
        if self.run > LumiInfo.lastSingleXingRun:
            # this run may have more than one crossing.  I don't know
            # how to fix this.
            self.badXingInfo = True
            return False
        self.numXings = 1
        xing = 1
        self.aveInstLum = self.totInstLum = lum = \
                          old_div(self.delivered, LumiInfo.lumiSectionLength)
        self.instLums.append( (xing, lum) )
        self.xingInfo = True
        return True
        

    def deadtime (self):
        if not self.delivered:
            return 1
        return 1 - (old_div(self.recorded, self.delivered))


    def __str__ (self):
        return "%6d, %4d: %6.1f (%4.1f%%) %4.2f (%3d)" % \
               (self.run,
                self.lumi,
                self.delivered,
                self.deadtime(),
                self.totInstLum,
                self.numXings)


##############################
## ######################## ##
## ## LumiInfoCont Class ## ##
## ######################## ##
##############################

class LumiInfoCont (dict):

    def __init__ (self, filename, **kwargs):
        print("loading luminosity information from '%s'." % filename)
        source = open (filename, 'r')
        self.minMaxKeys = ['totInstLum', 'aveInstLum', 'numXings',
                     'delivered', 'recorded']
        self._min = {}
        self._max = {}
        self.totalRecLum = 0.
        self.xingInfo    = False
        self.allowNoXing = kwargs.get ('ignore')
        self.noWarnings  = kwargs.get ('noWarnings')
        self.minRun      = 0
        self.maxRun      = 0
        self.minIntLum   = 0
        self.maxIntLum   = 0
        
        for key in self.minMaxKeys:
            self._min[key] = -1
            self._max[key] =  0
        for line in source:
            try:
                lumi = LumiInfo (line)
            except:
                continue
            self[lumi.key] = lumi
            self.totalRecLum += lumi.recorded
            if not self.xingInfo and lumi.xingInfo:
                self.xingInfo = True
            if lumi.xingInfo:
                #print "yes", lumi.keyString
                if not self.xingInfo:
                    print("huh?")
            for key in self.minMaxKeys:
                val = getattr (lumi, key)
                if val < self._min[key] or self._min[key] < 0:
                    self._min[key] = val
                if val > self._max[key] or not self._max[key]:
                    self._max[key] = val
        source.close()
        ######################################################
        ## Now that everything is setup, switch integrated  ##
        ## luminosity to more reasonable units.             ##
        ######################################################
        # the default is '1/mb', but that's just silly.
        self.invunits = 'nb'
        lumFactor = 1e3        
        if   self.totalRecLum > 1e9:
            lumFactor = 1e9
            self.invunits = 'fb'
        elif self.totalRecLum > 1e6:
            lumFactor = 1e6
            self.invunits = 'pb'
        # use lumFactor to make everything consistent
        #print "units", self.invunits, "factor", lumFactor
        self.totalRecLum /= lumFactor
        for lumis in self.values():
            lumis.delivered /= lumFactor
            lumis.recorded  /= lumFactor
        # Probably want to rename this next subroutine, but I'll leave
        # it alone for now...
        self._integrateContainer()



    def __str__ (self):
        retval = 'run,     lum     del ( dt  ) inst (#xng)\n'
        for key, value in sorted (self.items()):
            retval += "%s\n" % value
        return retval


    def min (self, key):
        return self._min[key]


    def max (self, key):
        return self._max[key]


    def keys (self):
        return sorted (dict.keys (self))


    def iteritems (self):
        return sorted (dict.iteritems (self))


    def _integrateContainer (self):
        # calculate numbers for recorded integrated luminosity
        total = 0.
        for key, lumi in self.items():
            total += lumi.recorded
            lumi.totalRecorded = total
            lumi.fracRecorded  = old_div(total, self.totalRecLum)
        # calculate numbers for average xing instantaneous luminosity
        if not self.xingInfo:
            # nothing to do here
            return
        xingKeyList = []
        maxAveInstLum = 0.
        for key, lumi in self.items():
            if not lumi.xingInfo and not lumi.fixXingInfo():
                if not self.noWarnings:
                    print("Do not have lumi xing info for %s" % lumi.keyString)
                if not self.allowNoXing:
                    print("Setting no Xing info flag")
                    self.xingInfo = False
                    return
                continue
            xingKeyList.append( (lumi.aveInstLum, key) )
            if lumi.aveInstLum > maxAveInstLum:
                maxAveInstLum = lumi.aveInstLum
        xingKeyList.sort()
        total = 0.
        for tup in xingKeyList:
            lumi = self[tup[1]]
            total += lumi.recorded
            lumi.totalAXILrecorded = total
            lumi.fracAXILrecorded  = old_div(total, self.totalRecLum)
            lumi.fracAXIL = old_div(lumi.aveInstLum, maxAveInstLum)


#############################
## ####################### ##
## ## General Functions ## ##
## ####################### ##
#############################

def loadEvents (filename, cont, options):
    eventsDict = {}
    print("loading events from '%s'" % filename)
    events = open (filename, 'r')
    runIndex, lumiIndex, eventIndex, weightIndex = 0, 1, 2, 3
    if options.relOrder:
        lumiIndex, eventIndex = 2,1
    minPieces = 3
    totalWeight = 0.
    if options.weights:
        minPieces = 4
    for line in events:
        pieces = sepRE.split (line.strip())
        if len (pieces) < minPieces:
            if nonSpaceRE.search (line):
                print("skipping", line)
            continue
        try:
            run, lumi, event = int( pieces[runIndex]   ), \
                               int( pieces[lumiIndex]  ), \
                               int( pieces[eventIndex] )
        except:
            continue
        key = (run, lumi)
        if key not in cont:
            if options.ignore:
                print("Warning, %s is not found in the lumi information" \
                      % key.__str__())
                continue
            else:
                raise RuntimeError("%s is not found in lumi information.  Use '--ignoreNoLumiEvents' option to ignore these events and continue." \
                      % key.__str__())
        if options.edfMode != 'time' and not cont[key].xingInfo:
            if options.ignore:
                print("Warning, %s does not have Xing information" \
                      % key.__str__())
                continue
            else:
                raise RuntimeError("%s  does not have Xing information.  Use '--ignoreNoLumiEvents' option to ignore these events and continue." \
                      % key.__str__())            
        if options.weights:
            weight = float (pieces[weightIndex])
        else:
            weight = 1
        eventsDict.setdefault( key, []).append( (event, weight) )
        totalWeight += weight
    events.close()
    return eventsDict, totalWeight


def makeEDFplot (lumiCont, eventsDict, totalWeight, outputFile, options):
    # make TGraph
    xVals      = [0]
    yVals      = [0]
    expectedVals = [0]
    predVals   = [0]
    weight = 0
    expectedChunks = []
    ########################
    ## Time Ordering Mode ##
    ########################
    if 'time' == options.edfMode:
        # if we have a minimum run number, clear the lists
        if lumiCont.minRun or lumiCont.minIntLum:
            xVals      = []
            yVals      = []
            expectedVals = []
            predVals   = []
        # loop over events
        for key, eventList in sorted( eventsDict.items() ):
            usePoints = True
            # should we add this point?
            if lumiCont.minRun and lumiCont.minRun > key[0] or \
               lumiCont.maxRun and lumiCont.maxRun < key[0]:
                usePoints = False
            for event in eventList:
                weight += event[1]
                if not usePoints:
                    continue
                factor = old_div(weight, totalWeight)
                try:
                    intLum = lumiCont[key].totalRecorded
                except:
                    raise RuntimeError("key %s not found in lumi information" \
                          % key.__str__())
                if lumiCont.minIntLum and lumiCont.minIntLum > intLum or \
                   lumiCont.maxIntLum and lumiCont.maxIntLum < intLum:
                    continue
                lumFrac = old_div(intLum, lumiCont.totalRecLum)
                xVals.append( lumiCont[key].totalRecorded)
                yVals.append (factor)
                expectedVals.append (lumFrac)
                predVals.append   (lumFrac * options.pred)
        # put on the last point if we aren't giving a maximum run
        if not lumiCont.maxRun and not lumiCont.maxIntLum:
            xVals.append (lumiCont.totalRecLum)
            yVals.append (1)
            expectedVals.append (1)
            predVals.append (options.pred)
        ####################
        ## Reset Expected ##
        ####################
        if options.resetExpected:
            slope = old_div((yVals[-1] - yVals[0]), (xVals[-1] - xVals[0]))
            print("slope", slope)
            for index, old in enumerate (expectedVals):
                expectedVals[index] = yVals[0] + \
                                    slope * (xVals[index] - xVals[0])
        #############################################
        ## Break Expected by Integrated Luminosity ##
        #############################################
        if options.breakExpectedIntLum:
            breakExpectedIntLum = []
            for chunk in options.breakExpectedIntLum:
                pieces = sepRE.split (chunk)
                try:
                    for piece in pieces:
                        breakExpectedIntLum.append( float(piece) )
                except:
                    raise RuntimeError("'%s' from '%s' is not a valid float" \
                          % (piece, chunk))
            breakExpectedIntLum.sort()
            boundaries = []
            breakIndex = 0
            done = False
            for index, xPos in enumerate (xVals):
                if xPos > breakExpectedIntLum[breakIndex]:
                    boundaries.append (index)
                    while breakIndex < len (breakExpectedIntLum):
                        breakIndex += 1
                        if breakIndex >= len (breakExpectedIntLum):
                            done = True
                            break
                        # If this next position is different, than
                        # we're golden.  Otherwise, let it go through
                        # the loop again.
                        if xPos <= breakExpectedIntLum[breakIndex]:
                            break
                    if done:
                        break
            # do we have any boundaries?
            if not boundaries:
                raise RuntimeError("No values of 'breakExpectedIntLum' are in current range.")
            # is the first boundary at 0?  If not, add 0
            if boundaries[0]:
                boundaries.insert (0, 0)
            # is the last boundary at the end?  If not, make the end a
            # boundary
            if boundaries[-1] != len (xVals) - 1:
                boundaries.append( len (xVals) - 1 )
            rangeList = list(zip (boundaries, boundaries[1:]))
            for thisRange in rangeList:
                upper = thisRange[1]
                lower = thisRange[0]
                slope = old_div((yVals[upper] - yVals[lower]), \
                        (xVals[upper] - xVals[lower]))
                print("slope", slope)
                # now go over the range inclusively
                pairList = []
                for index in range (lower, upper + 1):
                    newExpected = yVals[lower] + \
                                slope * (xVals[index] - xVals[lower])
                    pairList.append( (xVals[index], newExpected) )
                    expectedVals[index] = newExpected
                expectedChunks.append (pairList)
    ###########################################
    ## Instantanous Luminosity Ordering Mode ##
    ###########################################
    elif 'instLum' == options.edfMode or 'instIntLum' == options.edfMode:
        eventTupList = []
        if not lumiCont.xingInfo:
            raise RuntimeError("Luminosity Xing information missing.")
        for key, eventList in sorted( eventsDict.items() ):
            try:
                lumi =  lumiCont[key]
                instLum   = lumi.aveInstLum
                fracAXIL  = lumi.fracAXILrecorded
                totalAXIL = lumi.totalAXILrecorded
            except:
                raise RuntimeError("key %s not found in lumi information" \
                      % key.__str__())
            for event in eventList:
                eventTupList.append( (instLum, fracAXIL, totalAXIL, key,
                                      event[0], event[1], ) )
        eventTupList.sort()
        for eventTup in eventTupList:
            weight += eventTup[5]
            factor = old_div(weight, totalWeight)
            if 'instLum' == options.edfMode:
                xVals.append (eventTup[0])
            else:
                xVals.append (eventTup[2])
            yVals.append (factor)
            expectedVals.append (eventTup[1])
            predVals.append   (eventTup[1] * options.pred)
    else:
        raise RuntimeError("It looks like Charles screwed up if you are seeing this.")

    size = len (xVals)
    step = int (old_div(math.sqrt(size), 2) + 1)
    if options.printValues:
        for index in range (size):
            print("%8f %8f %8f" % (xVals[index], yVals[index], expectedVals[index]), end=' ')
            if index > step:
                denom = xVals[index] - xVals[index - step]
                numer = yVals[index] - yVals[index - step]
                if denom:
                    print(" %8f" % (old_div(numer, denom)), end=' ')
            if 0 == index % step:
                print(" **", end=' ') ## indicates statistically independent
                             ## slope measurement
            print()
        print()

    xArray = array.array ('d', xVals)
    yArray = array.array ('d', yVals)
    expected = array.array ('d', expectedVals)
    graph = ROOT.TGraph( size, xArray, yArray)
    graph.SetTitle (options.title)
    graph.SetMarkerStyle (20)
    expectedGraph = ROOT.TGraph( size, xArray, expected)
    expectedGraph.SetLineColor (ROOT.kRed)
    expectedGraph.SetLineWidth (3)
    if options.noDataPoints:
        expectedGraph.SetLineStyle (2)

    # run statistical tests
    if options.weights:
        print("average weight per event:", old_div(weight, ( size - 1)))
    maxDistance = ROOT.TMath.KolmogorovTest (size, yArray,
                                             size, expected,
                                             "M")
    prob = ROOT.TMath.KolmogorovProb( maxDistance * math.sqrt( size ) )

    # display everything
    ROOT.gROOT.SetStyle('Plain')
    ROOT.gROOT.SetBatch()
    c1 = ROOT.TCanvas()
    graph.GetXaxis().SetRangeUser (min (xVals), max (xVals))
    minValue = min (min(yVals), min(expected))
    if options.pred:
        minValue = min (minValue, min (predVals))
    graph.GetYaxis().SetRangeUser (minValue,
                                   max (max(yVals), max(expected), max(predVals)))
    graph.SetLineWidth (3)
    if options.noDataPoints:
        graph.Draw ("AL")
    else:
        graph.Draw ("ALP")
    if 'instLum' == options.edfMode:
        graph.GetXaxis().SetTitle ("Average Xing Inst. Luminosity (1/ub/s)")
        graph.GetXaxis().SetRangeUser (0., lumiCont.max('aveInstLum'))
    else:
        if 'instIntLum' == options.edfMode:
            graph.GetXaxis().SetTitle ("Integrated Luminosity - Inst. Lum. Ordered (1/%s)" \
                                       % lumiCont.invunits)
        else:
            graph.GetXaxis().SetTitle ("Integrated Luminosity (1/%s)" \
                                       % lumiCont.invunits)
    graph.GetYaxis().SetTitle ("Fraction of Events Seen")
    expectedGraphs = []
    if expectedChunks:
        for index, chunk in enumerate (expectedChunks):
            expectedXarray = array.array ('d', [item[0] for item in chunk])
            expectedYarray = array.array ('d', [item[1] for item in chunk])
            expectedGraph = ROOT.TGraph( len(chunk),
                                         expectedXarray,
                                         expectedYarray )
            expectedGraph.SetLineWidth (3)
            if options.noDataPoints:
                expectedGraph.SetLineStyle (2)
            if index % 2:
                expectedGraph.SetLineColor (ROOT.kBlue)
            else:
                expectedGraph.SetLineColor (ROOT.kRed)
            expectedGraph.Draw("L")
            expectedGraphs.append (expectedGraph)
        exptectedGraph = expectedGraphs[0]
    else:
        expectedGraph.Draw ("L")
    green = 0
    if options.pred:
        predArray = array.array ('d', predVals)
        green = ROOT.TGraph (size, xArray, predArray)
        green.SetLineWidth (3)
        green.SetLineColor (8)
        green.Draw ('l')
    legend = ROOT.TLegend(0.15, 0.65, 0.50, 0.85)
    legend.SetFillStyle (0)
    legend.SetLineColor(ROOT.kWhite)
    observed = 'Observed'
    if options.weights:
        observed += ' (weighted)'
    legend.AddEntry(graph, observed,"PL")
    if options.resetExpected:
        legend.AddEntry(expectedGraph,  "Expected from partial yield","L")
    else:
        legend.AddEntry(expectedGraph,  "Expected from total yield","L")
    if options.pred:
        legend.AddEntry(green, options.predLabel,"L")
    legend.AddEntry("","D_{stat}=%.3f, N=%d" % (maxDistance, size),"")
    legend.AddEntry("","P_{KS}=%.3f" % prob,"")
    legend.Draw()

    # save file
    c1.Print (outputFile)


######################
## ################ ##
## ## ########## ## ##
## ## ## Main ## ## ##
## ## ########## ## ##
## ################ ##
######################

if __name__ == '__main__':
    ##########################
    ## command line options ##
    ##########################
    allowedEDF = ['time', 'instLum', 'instIntLum']
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, usage='%(prog)s [options] lumi.csv events.txt output.png', description='Script for generating EDF curves. See https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideGenerateEDF for more details.')
    plotGroup  = parser.add_argument_group("Plot Options")
    rangeGroup = parser.add_argument_group("Range Options")
    inputGroup = parser.add_argument_group("Input Options")
    modeGroup  = parser.add_argument_group("Mode Options")
    plotGroup.add_argument('--title', dest='title', type=str,
                           default = 'Empirical Distribution Function',
                           help = 'title of plot')
    plotGroup.add_argument('--predicted', dest='pred', type=float,
                           default = 0,
                           help = 'factor by which predicted curve is greater than observed')
    plotGroup.add_argument('--predLabel', dest='predLabel', type=str,
                           default = 'Predicted',
                           help = 'label of predicted in legend')
    plotGroup.add_argument('--noDataPoints', dest='noDataPoints',
                           default = False, action='store_true',
                           help="Draw lines but no points for data")
    rangeGroup.add_argument('--minRun', dest='minRun', type=int, default=0,
                            help='Minimum run number to consider')
    rangeGroup.add_argument('--maxRun', dest='maxRun', type=int, default=0,
                            help='Maximum run number to consider')
    rangeGroup.add_argument('--minIntLum', dest='minIntLum', type=float, default=0,
                            help='Minimum integrated luminosity to consider')
    rangeGroup.add_argument('--maxIntLum', dest='maxIntLum', type=float, default=0,
                            help='Maximum integrated luminosity to consider')
    rangeGroup.add_argument('--resetExpected', dest='resetExpected',
                            default = False, action='store_true',
                            help='Reset expected from total yield to highest point considered')
    rangeGroup.add_argument('--breakExpectedIntLum', dest='breakExpectedIntLum',
                            type=str, action='append', default=[],
                            help='Break expected curve into pieces at integrated luminosity boundaries')
    inputGroup.add_argument('--ignoreNoLumiEvents', dest='ignore',
                            default = False, action='store_true',
                            help = 'Ignore (with a warning) events that do not have a lumi section')
    inputGroup.add_argument('--noWarnings', dest='noWarnings',
                            default = False,action='store_true',
                            help = 'Do not print warnings about missing luminosity information')
    inputGroup.add_argument('--runEventLumi', dest='relOrder',
                            default = False, action='store_true',
                            help = 'Parse event list assuming Run, Event #, Lumi# order')
    inputGroup.add_argument('--weights', dest='weights', default = False, action='store_true',
                            help = 'Read fourth column as a weight')
    modeGroup.add_argument('--print', dest='printValues', default = False, action='store_true',
                           help = 'Print X and Y values of EDF plot')
    modeGroup.add_argument('--runsWithLumis', dest='runsWithLumis',
                           type=str,action='append', default=[],
                           help='Print out run and lumi sections corresponding to integrated luminosities provided and then exits')
    modeGroup.add_argument('--edfMode', dest='edfMode', type=str,
                           default='time',
                           help="EDF Mode", choices=allowedEDF)
    parser.add_argument("lumi_csv", metavar="lumi.csv", type=str)
    parser.add_argument("events_txt", metavar="events.txt", type=str, nargs='?')
    parser.add_argument("output_png", metavar="output.png", type=str, nargs='?')
    options = parser.parse_args()

    if not options.runsWithLumis and (options.events_txt is None or options.output_png is None):
        parser.error("Must provide lumi.csv, events.txt, and output.png")

    ##########################
    ## load Luminosity info ##
    ##########################
    cont = LumiInfoCont (options.lumi_csv, **options.__dict__)
    cont.minRun    = options.minRun
    cont.maxRun    = options.maxRun
    cont.minIntLum = options.minIntLum
    cont.maxIntLum = options.maxIntLum

    ##################################################
    ## look for which runs correspond to what total ##
    ## recorded integrated luminosity               ##
    ##################################################
    if options.runsWithLumis:
        recLumis = []
        for line in options.runsWithLumis:
            pieces = sepRE.split (line)
            for piece in pieces:
                try:
                    recLumValue = float (piece)
                except:
                    raise RuntimeError("'%s' in '%s' is not a float" % \
                          (piece, line))
                if recLumValue <= 0:
                    raise RuntimeError("You must provide positive values for -runsWithLumis ('%f' given)" % recLumValue)
                recLumis.append (recLumValue)
        if not recLumis:
            raise RuntimeError("What did Charles do now?")
        recLumis.sort()
        recLumIndex = 0
        recLumValue = recLumis [recLumIndex]
        prevRecLumi = 0.
        done = False
        for key, lumi in cont.items():
            if prevRecLumi >= recLumValue and recLumValue < lumi.totalRecorded:
                # found it
                print("%s contains total recorded lumi %f" % \
                      (key.__str__(), recLumValue))
                while True:
                    recLumIndex += 1
                    if recLumIndex == len (recLumis):
                        done = True
                        break
                    recLumValue = recLumis [recLumIndex]
                    if prevRecLumi >= recLumValue and recLumValue < lumi.totalRecorded:
                        # found it
                        print("%s contains total recorded lumi %f" % \
                              (key.__str__(), recLumValue))
                    else:
                        break
                if done:
                    break
            prevRecLumi = lumi.totalRecorded
        if recLumIndex < len (recLumis):
            print("Theses lumis not found: %s" % recLumis[recLumIndex:])
        sys.exit()

    ####################
    ## make EDF plots ##
    ####################
    if options.edfMode != 'time' and not cont.xingInfo:
        raise RuntimeError("'%s' does not have Xing info" % options.lumi_csv)
    eventsDict, totalWeight = loadEvents (options.events_txt, cont, options)
    makeEDFplot (cont, eventsDict, totalWeight, options.output_png, options)
