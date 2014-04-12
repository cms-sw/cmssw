import FWCore.ParameterSet.Config as cms

energyThreshold = 2.

triggerTypes = cms.untracked.vstring('ECAL', 'HCAL', 'CSC', 'DT', 'RPC')

ecalClusterTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        doExtra = cms.untracked.bool(True),
        energyThreshold = cms.untracked.double(energyThreshold),
        egTriggerAlgos = cms.untracked.vstring(
            "L1_SingleEG2",
            "L1_SingleEG5",
            "L1_SingleEG8",
            "L1_SingleEG10",
            "L1_SingleEG12",
            "L1_SingleEG15",
            "L1_SingleEG20",
            "L1_SingleEG25",
            "L1_DoubleNoIsoEG_BTB_tight",
            "L1_DoubleNoIsoEG_BTB_loose",
            "L1_DoubleNoIsoEGTopBottom",
            "L1_DoubleNoIsoEGTopBottomCen",
            "L1_DoubleNoIsoEGTopBottomCen2",
            "L1_DoubleNoIsoEGTopBottomCenVert"
        ),
        L1GlobalTriggerReadoutRecordTag = cms.untracked.InputTag("gtDigis"),
        L1MuGMTReadoutCollectionTag = cms.untracked.InputTag("gtDigis"),
        swissCrossMaxThreshold = cms.untracked.double(3.)
    ),    
    MEs = cms.untracked.PSet(
        TrendNBC = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s number of basic clusters'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the number of basic clusters per event in EB/EE.')
        ),
        TrendBCSize = cms.untracked.PSet(
            path = cms.untracked.string('Ecal/Trends/ClusterTask %(prefix)s size of basic clusters'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('Trend'),
            description = cms.untracked.string('Trend of the mean size of the basic clusters.')
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
        BCSizeMapProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC size projection phi%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
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
        BCEMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC energy map%(suffix)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean energy of the basic clusters.')
        ),
        BCEtMapProjEta = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection eta%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('transverse energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Projection of the mean Et of the basic clusters.')
        ),
        BCEtMapProjPhi = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT BC ET projection phi%(suffix)s'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('transverse energy (GeV)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('ProjPhi'),
            description = cms.untracked.string('Projection of the mean Et of the basic clusters.')
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
        SingleCrystalCluster = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTask/%(prefix)sCLT SC single crystal cluster seed occupancy map%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occupancy map of the occurrence of super clusters with only one constituent')
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
        SCSizeVsEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC size (crystal) vs energy (GeV)'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.),
                low = cms.untracked.double(0.),
                nbins = cms.untracked.int32(100),
                title = cms.untracked.string('energy (GeV)')
            ),
            description = cms.untracked.string('Mean SC size in crystals as a function of the SC energy.')
        ),
        SCSeedOccupancyHighE = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (high energy clusters) %(supercrystal)s binned'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters with energy > ' + str(energyThreshold) + ' GeV.')
        ),
        SCSeedOccupancyTrig = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed occupancy map%(suffix)s (%(trig)s triggered) %(supercrystal)s binned'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occupancy map of the crystals that seeded super clusters.'),
            multi = cms.untracked.PSet(
                trig = triggerTypes
            )
        ),
        SCSeedTimeTrigEx = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing (%(trig)s exclusive triggered)'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.),
                low = cms.untracked.double(-25.),
                nbins = cms.untracked.int32(100),
                title = cms.untracked.string('time (ns)')
            ),
            description = cms.untracked.string('Timing distribution of the crystals that seeded super clusters.'),
            multi = cms.untracked.PSet(
                trig = triggerTypes
            )        
        ),
        SCSeedTimeMapTrigEx = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sClusterTaskExtras/%(prefix)sCLTE SC seed crystal timing map%(suffix)s (%(trig)s exclusive triggered) %(supercrystal)s binned'),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            zaxis = cms.untracked.PSet(
                high = cms.untracked.double(25.),
                low = cms.untracked.double(-25.),
                title = cms.untracked.string('time (ns)')
            ),
            description = cms.untracked.string('Mean timing of the crystals that seeded super clusters.'),
            multi = cms.untracked.PSet(
                trig = triggerTypes
            )
        ),
        SCOccupancyProjEta = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_eta"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('ProjEta'),
            description = cms.untracked.string('Supercluster eta.')
        ),
        SCOccupancyProjPhi = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/superClusters_%(subdetshortsig)s_phi"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('ProjPhi'),
            description = cms.untracked.string('Supercluster phi.')
        ),
        SCSwissCross = cms.untracked.PSet(
            path = cms.untracked.string("EcalBarrel/EBRecoSummary/superClusters_EB_E1oE4"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.5),
                low = cms.untracked.double(0.),
                nbins = cms.untracked.int32(100)
            ),
            description = cms.untracked.string('Swiss cross for SC maximum-energy crystal.')
        ),
        Triggers = cms.untracked.PSet( # not exactly cluster related, but is useful to know how many times each category fired
            path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE triggers'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('None'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(5),
                high = cms.untracked.double(5.),
                low = cms.untracked.double(0.),
                labels = triggerTypes,
                title = cms.untracked.string('triggers')
            ),
            description = cms.untracked.string('Counter for the trigger categories')
        ),
        ExclusiveTriggers = cms.untracked.PSet( # not exactly cluster related, but is useful to know how many times each category fired
            path = cms.untracked.string('EcalBarrel/EBClusterTaskExtras/EBCLTE exclusive triggers'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('None'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(5),
                high = cms.untracked.double(5.),
                low = cms.untracked.double(0.),
                labels = triggerTypes,
                title = cms.untracked.string('triggers')
            ),
            description = cms.untracked.string('Counter for the trigger categories')
        )
    )
)

