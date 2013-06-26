import FWCore.ParameterSet.Config as cms

MuonGeometrySanityCheck = cms.EDAnalyzer(
    "MuonGeometrySanityCheck",
    printout = cms.string("all"),
    tolerance = cms.double(1e-6),
    prefix = cms.string("CHECK"),
    frames = cms.VPSet(),
    points = cms.VPSet(),
    )

def detectors(dt=True, csc=True, me42=False, chambers=True, superlayers=False, layers=False):
    output = []
    if dt:
        for wheelName in "-2", "-1", "0", "+1", "+2":
            for stationName in "1", "2", "3", "4":
                numSectors = 12
                if stationName == "4": numSectors = 14
                for sectorName in map(str, range(1, numSectors+1)):
                    name = "MB" + wheelName + "/" + stationName + "/" + sectorName
                    if chambers: output.append(name)

                    superlayerNames = "1", "2", "3"
                    if stationName == "4": superlayerNames = "1", "3"
                    for superlayerName in superlayerNames:
                        name = "MB" + wheelName + "/" + stationName + "/" + sectorName + "/" + superlayerName
                        if superlayers: output.append(name)

                        for layerName in "1", "2", "3", "4":
                            name = "MB" + wheelName + "/" + stationName + "/" + sectorName + "/" + superlayerName + "/" + layerName
                            if layers: output.append(name)

    if csc:
        for stationName in "-4", "-3", "-2", "-1", "+1", "+2", "+3", "+4":
            ringNames = "1", "2"
            if stationName in ("-1", "+1"): ringNames = "1", "2", "3", "4"
            for ringName in ringNames:
                numChambers = 36
                if stationName + "/" + ringName in ("-4/1", "-3/1", "-2/1", "+2/1", "+3/1", "+4/1"): numChambers = 18
                for chamberName in map(str, range(1, numChambers+1)):
                    name = "ME" + stationName + "/" + ringName + "/" + chamberName
                    if chambers:
                        if me42 or stationName + "/" + ringName not in ("-4/2", "+4/2"):
                            output.append(name)

                    for layerName in "1", "2", "3", "4", "5", "6":
                        name = "ME" + stationName + "/" + ringName + "/" + chamberName + "/" + layerName
                        if layers:
                            if me42 or stationName + "/" + ringName not in ("-4/2", "+4/2"):
                                output.append(name)

    return output
