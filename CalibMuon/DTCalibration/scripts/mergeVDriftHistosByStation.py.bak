#! /usr/bin/env python

import ROOT
import sys

def getHistoName(wheel,station,sector):
    wheelStr = 'W' + str(wheel)
    stationStr = 'St' + str(station)
    sectorStr = 'Sec' + str(sector)
    name = "hRPhiVDriftCorr_" + wheelStr + "_" + stationStr + "_" + sectorStr

    return name

def mergeHistosWheelSector(file, wheel, sector):

    histWheelSector = None 
    for station in range(1,5):
        if sector in (13,14) and station != 4: continue
        name = getHistoName(wheel,station,sector)
        hist = file.Get(name)
        if hist:
            print "Adding",hist.GetName()  
            if not histWheelSector: histWheelSector = hist.Clone( "h_W%d_Sec%d" % (wheel,sector) )
            else: histWheelSector.Add(hist)

    return histWheelSector

def mergeHistosWheelStation(file, wheel, station):

    sectors = range(1,13)
    if station == 4: sectors.extend([13,14])
    histWheelStation = None
    for sector in sectors:
        name = getHistoName(wheel,station,sector)
        hist = file.Get(name)
        if hist:
            print "Adding",hist.GetName()
            if not histWheelStation: histWheelStation = hist.Clone( "h_W%d_St%d" % (wheel,station) )
            else: histWheelStation.Add(hist)

    return histWheelStation

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser ("Usage: %prog [--options]")
    # Options
    parser.add_option("-f","--file", dest="file", help="Input file name")
    parser.add_option("-o","--out", dest="out", default="merged.root", help="Output file name")
    (options, args) = parser.parse_args()

    if not options.file:
        parser.error('must set an input file')
    
    file = ROOT.TFile(options.file,"READ")
    ROOT.gROOT.cd()
 
    wheels = range(-2,3)
    stations = range(1,5)
    sectors = range(1,15)
    histos = {}
    for wheel in wheels:
        for station in stations:
            print "Merging histos from Wheel %d, Station %d" % (wheel,station)
            histos[(wheel,station)] = mergeHistosWheelStation(file,wheel,station) 
            
    file.Close()
  
    outputFile = ROOT.TFile(options.out,"RECREATE")
    outputFile.cd()
    for wheel in wheels:
        wheelStr = 'W' + str(wheel)
        for station in stations:
            stationStr = 'St' + str(station)
            for sector in sectors:
                if sector in (13,14) and station != 4: continue
                sectorStr = 'Sec' + str(sector)
                name = "hRPhiVDriftCorr_" + wheelStr + "_" + stationStr + "_" + sectorStr
                print "Writing",name 
                histos[(wheel,station)].Clone(name).Write()
 
    outputFile.Close()

    sys.exit(0)
