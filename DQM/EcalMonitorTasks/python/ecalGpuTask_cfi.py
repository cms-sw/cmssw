import FWCore.ParameterSet.Config as cms

digiSamples_ = [1,2,3,4,5,6,7,8,9,10]
uncalibOOTAmps_ = [4,6]

ecalGpuTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        runGpuTask = cms.untracked.bool(False),
        gpuOnlyPlots = cms.untracked.bool(True),
        uncalibOOTAmps = cms.untracked.vint32(uncalibOOTAmps_)
    ),
    MEs = cms.untracked.PSet(
        # CPU Digi
        DigiCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT digi nDigis cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Digis per Event')
            ),
            description = cms.untracked.string('Number of CPU Digis per Event')
        ),
        DigiCpuAmplitude = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT digi amplitude sample %(sample)s cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                sample = cms.untracked.vint32(digiSamples_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(4096),
                title = cms.untracked.string('ADC Counts')
            ),
            description = cms.untracked.string('CPU digi amplitudes for individual digi samples (1-10)')
        ),
        # GPU Digi (optional)
        DigiGpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT digi nDigis gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Digis per Event')
            ),
            description = cms.untracked.string('Number of GPU Digis per Event')
        ),
        DigiGpuAmplitude = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT digi amplitude sample %(sample)s gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                sample = cms.untracked.vint32(digiSamples_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(4096),
                title = cms.untracked.string('ADC Counts')
            ),
            description = cms.untracked.string('GPU digi amplitudes for individual digi samples (1-10)')
        ),
        # Digi GPU-CPU Difference
        DigiGpuCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT digi nDigis gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-500),
                high = cms.untracked.double(500),
                title = cms.untracked.string('GPU-CPU Digis per Event')
            ),
            description = cms.untracked.string('GPU-CPU difference of number of Digis per Event')
        ),
        DigiGpuCpuAmplitude = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT digi amplitude sample %(sample)s gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                sample = cms.untracked.vint32(digiSamples_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-100),
                high = cms.untracked.double(100),
                title = cms.untracked.string('ADC Counts')
            ),
            description = cms.untracked.string('GPU-CPU difference of digi amplitude for individual digi samples (1-10)')
        ),
        # CPU UncalibRecHit
        UncalibCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit nHits cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Uncalibrated Rec Hits per Event')
            ),
            description = cms.untracked.string('Number of CPU Uncalibrated Rec Hits per Event')
        ),
        UncalibCpuAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit amplitude cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Amplitude')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit reconstructed amplitude')
        ),
        UncalibCpuAmpError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit amplitudeError cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('Amplitude Error')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit reconstructed amplitude uncertainty')
        ),
        UncalibCpuPedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit pedestal cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1000),
                title = cms.untracked.string('Pedestal')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit reconstructed pedestal')
        ),
        UncalibCpuJitter = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit jitter cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-5),
                high = cms.untracked.double(5),
                title = cms.untracked.string('Jitter')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit reconstructed time jitter')
        ),
        UncalibCpuJitterError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit jitterError cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0),
                high = cms.untracked.double(0.25), # If you edit this, also change 10k bin in GpuTask.cc
                title = cms.untracked.string('Jitter Error')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit reconstructed time jitter uncertainty. 10000 is special value, shown in last bin')
        ),
        UncalibCpuChi2 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit chi2 cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('Chi2')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit chi2 of the pulse')
        ),
        UncalibCpuOOTAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                OOTAmp = cms.untracked.vint32(uncalibOOTAmps_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(500),
                title = cms.untracked.string('OOT Amplitude')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit out-of-time reconstructed amplitude. Indicies go from 0 to 9, with event BX at index 5. Index 4 == BX-1, index 6 == BX+1, etc.')
        ),
        UncalibCpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit flags cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(64),
                low = cms.untracked.double(0),
                high = cms.untracked.double(64),
                title = cms.untracked.string('Flags')
            ),
            description = cms.untracked.string('CPU Uncalibrated Rec Hit flag to be propagated to RecHit')
        ),
        # GPU UncalibRecHit (optional)
        UncalibGpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit nHits gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Uncalibrated Rec Hits per Event')
            ),
            description = cms.untracked.string('Number of GPU Uncalibrated Rec Hits per Event')
        ),
        UncalibGpuAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit amplitude gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Amplitude')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit reconstructed amplitude')
        ),
        UncalibGpuAmpError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit amplitudeError gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('Amplitude Error')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit reconstructed amplitude uncertainty')
        ),
        UncalibGpuPedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit pedestal gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1000),
                title = cms.untracked.string('Pedestal')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit reconstructed pedestal')
        ),
        UncalibGpuJitter = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit jitter gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-5),
                high = cms.untracked.double(5),
                title = cms.untracked.string('Jitter')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit reconstructed time jitter')
        ),
        UncalibGpuJitterError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit jitterError gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0),
                high = cms.untracked.double(0.25), # If you edit this, also change 10k bin in GpuTask.cc
                title = cms.untracked.string('Jitter Error')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit reconstructed time jitter uncertainty. 10000 is special value, shown in last bin')
        ),
        UncalibGpuChi2 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit chi2 gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('Chi2')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit chi2 of the pulse')
        ),
        UncalibGpuOOTAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                OOTAmp = cms.untracked.vint32(uncalibOOTAmps_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(500),
                title = cms.untracked.string('OOT Amplitude')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit out-of-time reconstructed amplitude. Indicies go from 0 to 9, with event BX at index 5. Index 4 == BX-1, index 6 == BX+1, etc.')
        ),
        UncalibGpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit flags gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(64),
                low = cms.untracked.double(0),
                high = cms.untracked.double(64),
                title = cms.untracked.string('Flags')
            ),
            description = cms.untracked.string('GPU Uncalibrated Rec Hit flag to be propagated to RecHit')
        ),
        # UncalibRecHit GPU-CPU Difference
        UncalibGpuCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit nHits gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-500),
                high = cms.untracked.double(500),
                title = cms.untracked.string('GPU-CPU Uncalibrated Rec Hits per Event')
            ),
            description = cms.untracked.string('GPU-CPU difference of number of Uncalibrated Rec Hits per Event')
        ),
        UncalibGpuCpuAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit amplitude gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-100),
                high = cms.untracked.double(100),
                title = cms.untracked.string('GPU-CPU Amplitude')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit reconstructed amplitude')
        ),
        UncalibGpuCpuAmpError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit amplitudeError gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-50),
                high = cms.untracked.double(50),
                title = cms.untracked.string('GPU-CPU Amplitude Error')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit reconstructed amplitude uncertainty')
        ),
        UncalibGpuCpuPedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit pedestal gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-50),
                high = cms.untracked.double(50),
                title = cms.untracked.string('GPU-CPU Pedestal')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit reconstructed pedestal')
        ),
        UncalibGpuCpuJitter = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit jitter gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-1),
                high = cms.untracked.double(1),
                title = cms.untracked.string('GPU-CPU Jitter')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit reconstructed time jitter')
        ),
        UncalibGpuCpuJitterError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit jitterError gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-0.03),
                high = cms.untracked.double(0.03),
                title = cms.untracked.string('GPU-CPU Jitter Error')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit reconstructed time jitter uncertainty. 10000 is special value, shown in last bin')
        ),
        UncalibGpuCpuChi2 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit chi2 gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-20),
                high = cms.untracked.double(20),
                title = cms.untracked.string('GPU-CPU Chi2')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit chi2 of the pulse')
        ),
        UncalibGpuCpuOOTAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                OOTAmp = cms.untracked.vint32(uncalibOOTAmps_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-50),
                high = cms.untracked.double(50),
                title = cms.untracked.string('GPU-CPU OOT Amplitude')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit out-of-time reconstructed amplitude. Indicies go from 0 to 9, with event BX at index 5. Index 4 == BX-1, index 6 == BX+1, etc.')
        ),
        UncalibGpuCpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT uncalib rec hit flags gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(128),
                low = cms.untracked.double(-64),
                high = cms.untracked.double(64),
                title = cms.untracked.string('GPU-CPU Flags')
            ),
            description = cms.untracked.string('GPU-CPU difference of Uncalibrated Rec Hit flag to be propagated to RecHit')
        ),
        # CPU RecHit
        RecHitCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit nHits cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Rec Hits per Event')
            ),
            description = cms.untracked.string('Number of CPU Rec Hits per Event')
        ),
        RecHitCpuEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit energy cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5.0),
                title = cms.untracked.string('Energy (Gev)')
            ),
            description = cms.untracked.string('CPU Rec Hit Energy (GeV)')
        ),
        RecHitCpuTime = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit time cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                high = cms.untracked.double(25.0),
                title = cms.untracked.string('Time (ns)')
            ),
            description = cms.untracked.string('CPU Rec Hit Time')
        ),
        RecHitCpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit flags cpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1024),
                title = cms.untracked.string('Flags')
            ),
            description = cms.untracked.string('CPU Rec Hit Flags')
        ),
        # GPU RecHit (optional)
        RecHitGpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit nHits gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('Rec Hits per Event')
            ),
            description = cms.untracked.string('Number of GPU Rec Hits per Event')
        ),
        RecHitGpuEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit energy gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5.0),
                title = cms.untracked.string('Energy (Gev)')
            ),
            description = cms.untracked.string('GPU Rec Hit Energy (GeV)')
        ),
        RecHitGpuTime = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit time gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                high = cms.untracked.double(25.0),
                title = cms.untracked.string('Time (ns)')
            ),
            description = cms.untracked.string('GPU Rec Hit Time')
        ),
        RecHitGpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit flags gpu'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1024),
                title = cms.untracked.string('Flags')
            ),
            description = cms.untracked.string('GPU Rec Hit Flags')
        ),
        # RecHit GPU-CPU Difference
        RecHitGpuCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit nHits gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-500),
                high = cms.untracked.double(500),
                title = cms.untracked.string('GPU-CPU Rec Hits per Event')
            ),
            description = cms.untracked.string('GPU-CPU difference of number of total Rec Hits per Event')
        ),
        RecHitGpuCpuEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit energy gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-1.0),
                high = cms.untracked.double(1.0),
                title = cms.untracked.string('GPU-CPU Energy (GeV)')
            ),
            description = cms.untracked.string('GPU-CPU difference of Rec Hit Energy (GeV)')
        ),
        RecHitGpuCpuTime = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit time gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-2.5),
                high = cms.untracked.double(2.5),
                title = cms.untracked.string('GPU-CPU Time (ns)')
            ),
            description = cms.untracked.string('GPU-CPU difference of Rec Hit Time')
        ),
        RecHitGpuCpuFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/%(prefix)sGT rec hit flags gpu-cpu diff'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-1024),
                high = cms.untracked.double(1024),
                title = cms.untracked.string('GPU-CPU Flags')
            ),
            description = cms.untracked.string('GPU-CPU differnece of Rec Hit Flags')
        )
    )
)
