import FWCore.ParameterSet.Config as cms

nHitsBins = 100
nHitsMax = 5000

energyBins = 100
energyMax = 2.0

timeBins = 100
timeMax = 12.5

flagsBins = 40
flagsMax = 1500

deltaBins = 101
delta = 0.2

ecalGpuTask = cms.untracked.PSet(
    MEs = cms.untracked.PSet(
        # CPU Plots
        RecHitCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT number of rec hits cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(nHitsMax),
                nbins = cms.untracked.int32(nHitsBins),
                low = cms.untracked.double(0),
                title = cms.untracked.string('Rec Hits per Event')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Number of total CPU Rec Hits per Event')
        ),
        RecHitCpuEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit energy cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(energyMax),
                nbins = cms.untracked.int32(energyBins),
                low = cms.untracked.double(0),
                title = cms.untracked.string('Energy (Gev)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('CPU Rec Hit Energy (GeV)')
        ),
        RecHitCpuTime = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit time cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(timeMax),
                nbins = cms.untracked.int32(timeBins),
                low = cms.untracked.double(-timeMax),
                title = cms.untracked.string('Time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('CPU Rec Hit Time')
        ),
        RecHitCpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit flags cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(flagsMax),
                nbins = cms.untracked.int32(flagsBins),
                low = cms.untracked.double(0),
                title = cms.untracked.string('Flags')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('CPU Rec Hit Flags')
        ),
        # GPU-CPU Difference Plots
        RecHitGpuCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT number of rec hits diff gpu-cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(delta*nHitsMax),
                nbins = cms.untracked.int32(deltaBins),
                low = cms.untracked.double(-delta*nHitsMax),
                title = cms.untracked.string('GPU-CPU Rec Hits per Event')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('GPU-CPU difference of number of total Rec Hits per Event')
        ),
        RecHitGpuCpuEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit energy diff gpu-cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(delta*energyMax),
                nbins = cms.untracked.int32(deltaBins),
                low = cms.untracked.double(-delta*energyMax),
                title = cms.untracked.string('GPU-CPU Energy (GeV)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('GPU-CPU difference of Rec Hit Energy (GeV)')
        ),
        RecHitGpuCpuTime = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit time diff gpu-cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(delta*timeMax),
                nbins = cms.untracked.int32(deltaBins),
                low = cms.untracked.double(-delta*timeMax),
                title = cms.untracked.string('GPU-CPU Time (ns)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('GPU-CPU difference of Rec Hit Time')
        ),
        RecHitGpuCpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit flags diff gpu-cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(delta*flagsMax),
                nbins = cms.untracked.int32(deltaBins),
                low = cms.untracked.double(-delta*flagsMax),
                title = cms.untracked.string('GPU-CPU Flags')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('GPU-CPU differnece of Rec Hit Flags')
        )
    )
)
