import FWCore.ParameterSet.Config as cms

digiSamples_ = [1,2,3,4,5,6,7,8,9,10]
uncalibOOTAmps_ = [4,6]

ecalGpuTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        runGpuTask = cms.untracked.bool(False),

        # Default plots for each object are 1D CPU distributions and 1D GPU-CPU diff
        enableDigi = cms.untracked.bool(True),
        enableUncalib = cms.untracked.bool(True),
        enableRecHit = cms.untracked.bool(True),

        # 1D flags enable distributions of GPU values
        # 2D flags enable 2D comparison maps
        digi1D = cms.untracked.bool(True),
        digi2D = cms.untracked.bool(True),
        uncalib1D = cms.untracked.bool(True),
        uncalib2D = cms.untracked.bool(True),
        rechit1D = cms.untracked.bool(True),
        rechit2D = cms.untracked.bool(True),

        uncalibOOTAmps = cms.untracked.vint32(uncalibOOTAmps_)
    ),
    MEs = cms.untracked.PSet(
        # CPU Digi
        DigiCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi nDigis cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi amplitude sample %(sample)s cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi nDigis gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi amplitude sample %(sample)s gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi nDigis gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi amplitude sample %(sample)s gpu-cpu diff'),
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
        # Digi 2D plots
        Digi2D = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi nDigis gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('CPU Digis per Event')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('GPU Digis per Event')
            ),
            description = cms.untracked.string('Number of Digis per Event. GPU vs CPU comparison')
        ),
        Digi2DAmplitude = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/Digis/%(prefix)sGT digi amplitude sample %(sample)s gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                sample = cms.untracked.vint32(digiSamples_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(4096),
                title = cms.untracked.string('CPU ADC Counts')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(4096),
                title = cms.untracked.string('GPU ADC Counts')
            ),
            description = cms.untracked.string('Digi amplitudes for individual digi samples (1-10). GPU vs CPU comparison')
        ),
        # CPU UncalibRecHit
        UncalibCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit flags cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit flags gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit flags gpu-cpu diff'),
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
        # UncalibRecHit 2D plots
        Uncalib2D = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit nHits gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('CPU Uncalibrated Rec Hits per Event')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('GPU Uncalibrated Rec Hits per Event')
            ),
            description = cms.untracked.string('Number of Uncalibrated Rec Hits per Event. GPU vs CPU comparison')
        ),
        Uncalib2DAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitude gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('CPU Amplitude')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('GPU Amplitude')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit reconstructed amplitude. GPU vs CPU comparison')
        ),
        Uncalib2DAmpError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit amplitudeError gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('CPU Amplitude Error')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('GPU Amplitude Error')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit reconstructed amplitude uncertainty. GPU vs CPU comparison')
        ),
        Uncalib2DPedestal = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit pedestal gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1000),
                title = cms.untracked.string('CPU Pedestal')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1000),
                title = cms.untracked.string('GPU Pedestal')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit reconstructed pedestal. GPU vs CPU comparison')
        ),
        Uncalib2DJitter = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitter gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-5),
                high = cms.untracked.double(5),
                title = cms.untracked.string('CPU Jitter')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-5),
                high = cms.untracked.double(5),
                title = cms.untracked.string('GPU Jitter')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit reconstructed time jitter. GPU vs CPU comparison')
        ),
        Uncalib2DJitterError = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit jitterError gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0),
                high = cms.untracked.double(0.25), # If you edit this, also change 10k bin in GpuTask.cc
                title = cms.untracked.string('CPU Jitter Error')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(25),
                low = cms.untracked.double(0),
                high = cms.untracked.double(0.25), # If you edit this, also change 10k bin in GpuTask.cc
                title = cms.untracked.string('GPU Jitter Error')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit reconstructed time jitter uncertainty. GPU vs CPU comparison. 10000 is special value, shown in last bin')
        ),
        Uncalib2DChi2 = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit chi2 gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('CPU Chi2')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(200),
                title = cms.untracked.string('GPU Chi2')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit chi2 of the pulse. GPU vs CPU comparison')
        ),
        Uncalib2DOOTAmp = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit OOT amplitude %(OOTAmp)s gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            multi = cms.untracked.PSet(
                OOTAmp = cms.untracked.vint32(uncalibOOTAmps_)
            ),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(500),
                title = cms.untracked.string('CPU OOT Amplitude')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(500),
                title = cms.untracked.string('GPU OOT Amplitude')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit out-of-time reconstructed amplitude. GPU vs CPU comparison. Indicies go from 0 to 9, with event BX at index 5. Index 4 == BX-1, index 6 == BX+1, etc.')
        ),
        Uncalib2DFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/UncalibRecHits/%(prefix)sGT uncalib rec hit flags gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(64),
                low = cms.untracked.double(0),
                high = cms.untracked.double(64),
                title = cms.untracked.string('CPU Flags')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(64),
                low = cms.untracked.double(0),
                high = cms.untracked.double(64),
                title = cms.untracked.string('GPU Flags')
            ),
            description = cms.untracked.string('Uncalibrated Rec Hit flag to be propagated to RecHit. GPU vs CPU comparison')
        ),
        # CPU RecHit
        RecHitCpu = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit nHits cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit energy cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit time cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit flags cpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit nHits gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit energy gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit time gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit flags gpu'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit nHits gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit energy gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit time gpu-cpu diff'),
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
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit flags gpu-cpu diff'),
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
        ),
        # RecHit 2D plots
        RecHit2D = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit nHits gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('CPU Rec Hits per Event')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5000),
                title = cms.untracked.string('GPU Rec Hits per Event')
            ),
            description = cms.untracked.string('Number of Rec Hits per Event. GPU vs CPU comparison')
        ),
        RecHit2DEnergy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit energy gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5.0),
                title = cms.untracked.string('CPU Energy (Gev)')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(5.0),
                title = cms.untracked.string('GPU Energy (Gev)')
            ),
            description = cms.untracked.string('Rec Hit Energy (GeV). GPU vs CPU comparison')
        ),
        RecHit2DTime = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit time gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                high = cms.untracked.double(25.0),
                title = cms.untracked.string('CPU Time (ns)')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-25.0),
                high = cms.untracked.double(25.0),
                title = cms.untracked.string('GPU Time (ns)')
            ),
            description = cms.untracked.string('Rec Hit Time. GPU vs CPU comparison')
        ),
        RecHit2DFlags = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sGpuTask/RecHits/%(prefix)sGT rec hit flags gpu-cpu map2D'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1024),
                title = cms.untracked.string('CPU Flags')
            ),
            yaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0),
                high = cms.untracked.double(1024),
                title = cms.untracked.string('GPU Flags')
            ),
            description = cms.untracked.string('Rec Hit Flags. GPU vs CPU comparison')
        )
    )
)
