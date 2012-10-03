import FWCore.ParameterSet.Config as cms

ecalClusterTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        TrendNBC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of basic clusters'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of basic clusters per event in EB/EE.')
        ),
        BCOccupancyProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection phi%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            description = cms.untracked.string('Projection of the basic cluster occupancy.')
        ),
        BCSizeMapProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection eta%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.')
        ),
        BCSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the basic cluster size (number of crystals).')
        ),
        SCR9 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC R9'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.2),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of E_seed / E_3x3 of the super clusters.')
        ),
        SCNum = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC number'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the number of super clusters per event in EB/EE.')
        ),
        TrendNSC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of super clusters'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of super clusters per event in EB/EE.')
        ),
        TrendSCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of super clusters'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the mean size (number of crystals) of the super clusters.')
        ),
        SCSeedEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed crystal energy'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Energy distribution of the crystals that seeded super clusters.')
        ),
        SCE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Super cluster energy distribution.')
        ),
        BCE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Basic cluster energy distribution.')
        ),
        BCSizeMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size map%(suffix)s'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean size (number of crystals) of the basic clusters.')
        ),
        SCNcrystals = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size (crystal)'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the super cluster size (number of crystals).')
        ),
        BCSizeMapProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection phi%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            description = cms.untracked.string('Projection of the mean size (number of crystals) of the basic clusters.')
        ),
        BCEMapProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection phi%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            description = cms.untracked.string('Projection of the mean energy of the basic clusters.')
        ),
        SingleCrystalCluster = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC single crystal cluster seed occupancy map%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occupancy map of the occurrence of super clusters with only one constituent')
        ),
        BCNum = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the number of basic clusters per event.')
        ),
        BCOccupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number map%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Basic cluster occupancy.')
        ),
        BCOccupancyProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC number projection eta%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of the basic cluster occupancy.')
        ),
        SCNBCs = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC size'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.0),
                nbins = cms.untracked.int32(15),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of the super cluster size (number of basic clusters)')
        ),
        SCSeedOccupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC seed occupancy map%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.')
        ),
        BCEMapProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy projection eta%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of the mean energy of the basic clusters.')
        ),
        SCELow = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy (low scale)'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Energy distribution of the super clusters (low scale).')
        ),
        TrendBCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of basic clusters'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the mean size of the basic clusters.')
        ),
        SCClusterVsSeed = cms.untracked.PSet(
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(150.0),
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC energy vs seed crystal energy'),
            description = cms.untracked.string('Relation between super cluster energy and its seed crystal energy.')
        ),
        BCEMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy map%(suffix)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean energy of the basic clusters.')
        )
    )
)

