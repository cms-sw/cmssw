import FWCore.ParameterSet.Config as cms


def make_vpset_fromfile(filename):
    psets = []
    with open(filename, 'r') as f:
        fedid = 0
        channel = 0
        for line in f:
            line = line.split()
            detid = int(line[0])
            layer = int(line[1])
            if(layer > 4):
                type = (layer > 7) and "2S" or "PS"
                psets.append(cms.PSet(
                    moduleType=cms.string(type),
                    detid=cms.uint32(detid),
                    gbtid=cms.uint32(0),
                    fedid=cms.uint32(fedid),
                    fedch=cms.uint32(channel),
                    powerGroup=cms.uint32(0),
                    coolingLoop=cms.uint32(0))
                    )
                channel += 1
                if channel == 72:
                    channel = 0
                    fedid += 1
        while channel != 72:
            detid = 0 
            psets.append(cms.PSet(
                moduleType=cms.string(type),
                detid=cms.uint32(detid),
                gbtid=cms.uint32(0),
                fedid=cms.uint32(fedid),
                fedch=cms.uint32(channel),
                powerGroup=cms.uint32(0),
                coolingLoop=cms.uint32(0))
                )
            channel += 1

    return psets

my_psets = make_vpset_fromfile('/afs/cern.ch/user/f/favereau/detids_tracker.txt')

Phase2TrackerCabling = cms.ESSource("Phase2TrackerCablingCfgESSource", modules = cms.VPSet( *my_psets ))
