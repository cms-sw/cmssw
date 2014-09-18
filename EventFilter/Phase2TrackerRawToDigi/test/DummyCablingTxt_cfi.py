import FWCore.ParameterSet.Config as cms


def make_vpset_fromfile(filename):
    psets = []
    with open(filename, 'r') as f:
        fedid = 0
        channel = 0
        for line in f:
            psets.append(cms.PSet(
                moduleType=cms.string("2S"),
                detid=cms.uint32(int(line)),
                gbtid=cms.uint32(0),
                fedid=cms.uint32(fedid),
                fedch=cms.uint32(channel),
                powerGroup=cms.uint32(0),
                coolingLoop=cms.uint32(0))
                )
            channel += 1
            if channel >= 72:
                channel = 0
                fedid += 1
    return psets

# my_psets = make_vpset_fromfile('/afs/cern.ch/user/f/favereau/detids_tracker.txt')
my_psets = make_vpset_fromfile('/afs/cern.ch/user/f/favereau/detids.txt')

Phase2TrackerCabling = cms.ESSource("Phase2TrackerCablingCfgESSource", modules = cms.VPSet( *my_psets ))
