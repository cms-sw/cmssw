#! /usr/bin/env python

import sys
import re
import optparse
from pprint import pprint
import array
import ROOT
import math

sepRE = re.compile (r'[\s,;:]+')

##########################
## #################### ##
## ## LumiInfo Class ## ##
## #################### ##
##########################

class LumiInfo (object):

    lastSingleXingRun = 136175
    lumiSectionLength = 23.310779
    minRun            = 0
    maxRun            = 0

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
            raise RuntimeError, "Odd number of pieces"
        if size < 4:
            raise RuntimeError, "Not enough pieces"
        try:
            self.run       = int   (pieces[0])
            self.lumi      = int   (pieces[1])
            self.delivered = float (pieces[2])
            self.recorded  = float (pieces[3])
        except:
            raise RuntimeError, "Pieces not right format"
        if LumiInfo.minRun and self.run < LumiInfo.minRun or \
           LumiInfo.maxRun and self.run > LumiInfo.maxRun:
            raise RuntimeError, "Run %d out of requested range" % self.run
        if size > 4:
            try:
                for xing, lum in zip (pieces[4::2],pieces[5::2]):
                    xing = int   (xing)
                    lum  = float (lum)
                    self.instLums.append( (xing, lum) )
                    self.totInstLum += lum
                    self.numXings += 1
            except:
                raise RuntimeError, "Inst Lumi Info malformed"
            self.aveInstLum = self.totInstLum / (self.numXings)
            self.xingInfo   = True
        self.key       = (self.run, self.lumi)
        self.keyString = self.key.__str__()


    def fixXingInfo (self):
        if self.numXings:
            # You shouldn't try and fix an event if it already has
            # xing information.
            raise RuntimeError, "This event %s already has Xing information" \
                  % self.keyString
        if self.run > LumiInfo.lastSingleXingRun:
            # this run may have more than one crossing.  I don't know
            # how to fix this.
            self.badXingInfo = True
            return False
        self.numXings = 1
        xing = 1
        self.aveInstLum = self.totInstLum = lum = \
                          self.delivered / LumiInfo.lumiSectionLength
        self.instLums.append( (xing, lum) )
        self.xingInfo = True
        return True
        

    def deadtime (self):
        if not self.delivered:
            return 1
        return 1 - (self.recorded / self.delivered)


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
        print "loading luminosity information from '%s'." % filename
        source = open (filename, 'r')
        self.minMaxKeys = ['totInstLum', 'aveInstLum', 'numXings',
                     'delivered', 'recorded']
        self._min = {}
        self._max = {}
        self.totalRecLum = 0.
        self.xingInfo = False
        self.allowNoXing = kwargs.get ('ignore')
        self.noWarnings  = kwargs.get ('noWarnings')
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
                    print "huh?"
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
        for key, value in sorted (self.iteritems()):
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
        for key, lumi in self.iteritems():
            total += lumi.recorded
            lumi.totalRecorded = total
            lumi.fracRecorded  = total / self.totalRecLum
        # calculate numbers for average xing instantaneous luminosity
        if not self.xingInfo:
            # nothing to do here
            return
        xingKeyList = []
        maxAveInstLum = 0.
        for key, lumi in self.iteritems():
            if not lumi.xingInfo and not lumi.fixXingInfo():
                if not self.noWarnings:
                    print "Do not have lumi xing info for %s" % lumi.keyString
                if not self.allowNoXing:
                    print "Setting no Xing info flag"
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
            lumi.fracAXILrecorded  = total / self.totalRecLum
            lumi.fracAXIL = lumi.aveInstLum / maxAveInstLum


#############################
## ####################### ##
## ## General Functions ## ##
## ####################### ##
#############################

def loadEvents (filename, cont, options):
    eventsDict = {}
    print "loading events from '%s'" % filename
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
            print "skipping", line
            continue
        try:
            run, lumi, event = int( pieces[runIndex]   ), \
                               int( pieces[lumiIndex]  ), \
                               int( pieces[eventIndex] )
        except:
            continue
        if LumiInfo.minRun and run < LumiInfo.minRun or \
           LumiInfo.maxRun and run > LumiInfo.maxRun:
            continue
        key = (run, lumi)
        if not cont.has_key (key):
            if options.ignore:
                print "Warning, %s is not found in the lumi information" \
                      % key.__str__()
                continue
            else:
                raise RuntimeError, "%s is not found in lumi information" \
                      % key.__str__()
        if options.edfMode != 'time' and not cont[key].xingInfo:
            if options.ignore:
                print "Warning, %s does not have Xing information" \
                      % key.__str__()
                continue
            else:
                raise RuntimeError, "%s  does not have Xing information" \
                      % key.__str__()            
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
    theoryVals = [0]
    predVals   = [0]
    weight = 0
    if 'time' == options.edfMode:
        for key, eventList in sorted( eventsDict.iteritems() ):
            for event in eventList:
                weight += event[1]
                factor = weight / totalWeight
                try:
                    intLum = lumiCont[key].totalRecorded
                except:
                    raise RuntimeError, "key %s not found in lumi information" \
                          % key.__str__()
                lumFrac = intLum / lumiCont.totalRecLum
                xVals.append (intLum)
                yVals.append (factor)
                theoryVals.append (lumFrac)
                predVals.append   (lumFrac * options.pred)
    elif 'instLum' == options.edfMode or 'instIntLum' == options.edfMode:
        eventTupList = []
        if not lumiCont.xingInfo:
            raise RuntimeError, "Luminosity Xing information missing."
        for key, eventList in sorted( eventsDict.iteritems() ):
            try:
                lumi =  lumiCont[key]
                instLum   = lumi.aveInstLum
                fracAXIL  = lumi.fracAXILrecorded
                totalAXIL = lumi.totalAXILrecorded
            except:
                raise RuntimeError, "key %s not found in lumi information" \
                      % key.__str__()
            for event in eventList:
                eventTupList.append( (instLum, fracAXIL, totalAXIL, key,
                                      event[0], event[1], ) )
        eventTupList.sort()
        for eventTup in eventTupList:
            weight += eventTup[5]
            factor = weight / totalWeight
            if 'instLum' == options.edfMode:
                xVals.append (eventTup[0])
            else:
                xVals.append (eventTup[2])
            yVals.append (factor)
            theoryVals.append (eventTup[1])
            predVals.append   (eventTup[1] * options.pred)
    else:
        raise RuntimeError, "It looks like Charles screwed up if you are seeing this."

    size = len (xVals)
    step = int (math.sqrt(size) / 2 + 1)
    if options.printValues:
        for index in range (size):
            print "%8f %8f %8f" % (xVals[index], yVals[index], theoryVals[index]),
            if index > step:
                denom = xVals[index] - xVals[index - step]
                numer = yVals[index] - yVals[index - step]
                if denom:
                    print " %8f" % (numer / denom),
            if 0 == index % step:
                print " **", ## indicates statistically independent
                             ## slope measurement
            print
        print

    xArray = array.array ('d', xVals)
    yArray = array.array ('d', yVals)
    theory = array.array ('d', theoryVals)
    graph = ROOT.TGraph( size, xArray, yArray)
    graph.SetTitle (options.title)
    graph.SetMarkerStyle (20)
    theoryGraph = ROOT.TGraph( size, xArray, theory)
    theoryGraph.SetLineColor (ROOT.kRed)
    theoryGraph.SetLineWidth (2)

    # run statistical tests
    if options.weights:
        print "average weight per event:", weight / ( size - 1)
    maxDistance = ROOT.TMath.KolmogorovTest (size, yArray,
                                             size, theory,
                                             "M")
    prob = ROOT.TMath.KolmogorovProb( maxDistance * math.sqrt( size ) )

    # display everything
    ROOT.gROOT.SetStyle('Plain')
    ROOT.gROOT.SetBatch()
    c1 = ROOT.TCanvas()
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
        graph.GetXaxis().SetRangeUser (0., lumiCont.totalRecLum)
    graph.GetYaxis().SetTitle ("Fraction of Events Seen")
    if options.pred > 1:
        graph.GetYaxis().SetRangeUser (0., options.pred)
    else:
        graph.GetYaxis().SetRangeUser (0., 1.)
    theoryGraph.Draw ("L")
    green = 0
    if options.pred:
        predArray = array.array ('d', predVals)
        green = ROOT.TGraph (size, xArray, predArray)
        green.SetLineWidth (2)
        green.SetLineColor (8)
        green.Draw ('l')
    legend = ROOT.TLegend(0.15, 0.65, 0.50, 0.85)
    legend.SetFillStyle (0)
    legend.SetLineColor(ROOT.kWhite)
    observed = 'Observed'
    if options.weights:
        observed += ' (weighted)'
    legend.AddEntry(graph, observed,"PL")
    legend.AddEntry(theoryGraph,  "Expected from total yield","L")
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
    parser = optparse.OptionParser ("Usage: %prog [options] lumi.csv events.txt output.png")
    parser.add_option ('--title', dest='title', type='string',
                       default = 'Empirical Distribution Function',
                       help = 'title of plot (default %default)')
    parser.add_option ('--predicted', dest='pred', type='float',
                       default = 0,
                       help = 'factor by which predicted curve is greater than observed')
    parser.add_option ('--predLabel', dest='predLabel', type='string',
                       default = 'Predicted',
                       help = 'label of predicted in legend')
    parser.add_option ('--minRun', dest='minRun', type='int', default=0,
                       help='Minimum run number to consider')
    parser.add_option ('--maxRun', dest='maxRun', type='int', default=0,
                       help='Maximum run number to consider')
    parser.add_option ('--weights', dest='weights', action='store_true',
                       help = 'Read fourth column as a weight')
    parser.add_option ('--print', dest='printValues', action='store_true',
                       help = 'Print X and Y values')
    parser.add_option ('--ignoreNoLumiEvents', dest='ignore', action='store_true',
                       help = 'Ignore (with a warning) events that do not have a lumi section')
    parser.add_option ('--noWarnings', dest='noWarnings', action='store_true',
                       help = 'No warnings')
    parser.add_option ('--runEventLumi', dest='relOrder', action='store_true',
                       help = 'Parse event list assuming Run, Event #, Lumi# order')
    parser.add_option ('--runsWithLumis', dest='runsWithLumis', type='string',
                       action='append', default=[],
                       help='Print out run and lumi sections corresponding to integrated luminosities provided and then exits')
    parser.add_option ('--edfMode', dest='edfMode', type='string',
                       default='time',
                       help="EDF Mode %s (default '%%default')" % allowedEDF)
    (options, args) = parser.parse_args()

    if options.edfMode not in allowedEDF:
        raise RuntimeError, "edfMode (currently '%s') must be one of %s" \
              % (options.edfMode, allowedEDF)

    if len (args) != 3 and not (options.runsWithLumis and len(args) >= 1):
        raise RuntimeError, "Must provide lumi.csv, events.txt, and output.png"

    LumiInfo.minRun = options.minRun
    LumiInfo.maxRun = options.maxRun

    ##########################
    ## load Luminosity info ##
    ##########################
    cont = LumiInfoCont (args[0], **options.__dict__)

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
                    raise RuntimeError, "'%s' in '%s' is not a float" % \
                          (piece, line)
                if recLumValue <= 0:
                    raise RuntimeError, "You must provide positive values for -runsWithLumis ('%f' given)" % recLumValue
                recLumis.append (recLumValue)
        if not recLumis:
            raise RuntimeError, "What did Charles do now?"
        recLumis.sort()
        recLumIndex = 0
        recLumValue = recLumis [recLumIndex]
        prevRecLumi = 0.
        done = False
        for key, lumi in cont.iteritems():
            if prevRecLumi >= recLumValue and recLumValue < lumi.totalRecorded:
                # found it
                print "%s contains total recorded lumi %f" % \
                      (key.__str__(), recLumValue)
                while True:
                    recLumIndex += 1
                    if recLumIndex == len (recLumis):
                        done = True
                        break
                    recLumValue = recLumis [recLumIndex]
                    if prevRecLumi >= recLumValue and recLumValue < lumi.totalRecorded:
                        # found it
                        print "%s contains total recorded lumi %f" % \
                              (key.__str__(), recLumValue)
                    else:
                        break
                if done:
                    break
            prevRecLumi = lumi.totalRecorded
        if recLumIndex < len (recLumis):
            print "Theses lumis not found: %s" % recLumis[recLumIndex:]
        sys.exit()

    ####################
    ## make EDF plots ##
    ####################
    if options.edfMode != 'time' and not cont.xingInfo:
        raise RuntimeError, "'%s' does not have Xing info" % args[0]
    eventsDict, totalWeight = loadEvents (args[1], cont, options)
    makeEDFplot (cont, eventsDict, totalWeight, args[2], options)
