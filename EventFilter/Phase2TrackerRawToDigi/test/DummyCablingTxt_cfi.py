import FWCore.ParameterSet.Config as cms


def make_vpset_fromfile(filename):
    psets = []
    mtype = "Dummy"
    with open(filename, 'r') as f:
        channel = 0
        for line in f:
            line = line.split()
            detid   = int(line[0])
            fedid   = int(line[1])
            channel = int(line[2])
            psets.append(cms.PSet(
                moduleType=cms.string(mtype),
                detid=cms.uint32(detid),
                gbtid=cms.uint32(0),
                fedid=cms.uint32(fedid),
                fedch=cms.uint32(channel),
                powerGroup=cms.uint32(0),
                coolingLoop=cms.uint32(0))
                )
        while channel != 72:
            detid = 0 
            psets.append(cms.PSet(
                moduleType=cms.string(mtype),
                detid=cms.uint32(detid),
                gbtid=cms.uint32(0),
                fedid=cms.uint32(fedid),
                fedch=cms.uint32(channel),
                powerGroup=cms.uint32(0),
                coolingLoop=cms.uint32(0))
                )
            channel += 1

    return psets

my_psets = make_vpset_fromfile('detids_phase2.txt')

Phase2TrackerCabling = cms.ESSource("Phase2TrackerCablingCfgESSource", modules = cms.VPSet( *my_psets ))
