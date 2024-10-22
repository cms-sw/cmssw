import FWCore.ParameterSet.Config as cms

ecalPiZeroTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        # Parameters needed for pi0 finding
        seleXtalMinEnergy = cms.double(0.0),
        clusSeedThr = cms.double(0.5),
        clusEtaSize = cms.int32(3),
        clusPhiSize = cms.int32(3),
        selePtGammaOne = cms.double(0.9),
        selePtGammaTwo = cms.double(0.9),
        seleS4S9GammaOne = cms.double(0.85),
        seleS4S9GammaTwo = cms.double(0.85),
        selePtPi0 = cms.double(2.5),
        selePi0Iso = cms.double(0.5),
        selePi0BeltDR = cms.double(0.2),
        selePi0BeltDeta = cms.double(0.05),
        seleMinvMaxPi0 = cms.double(0.5),
        seleMinvMinPi0 = cms.double(0.0),
        posCalcParameters = cms.PSet(T0_barl      = cms.double(5.7),
                                     T0_endc      = cms.double(3.1),
                                     T0_endcPresh = cms.double(1.2),
                                     LogWeighted  = cms.bool(True),
                                     W0           = cms.double(4.2),
                                     X0           = cms.double(0.89)
                                     ),
    ),
    MEs = cms.untracked.PSet(
        Pi0MinvEB = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPiZeroTask/%(prefix)sPZT%(suffix)s Pi0 Invariant Mass'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.),
                high = cms.untracked.double(0.5),
                title = cms.untracked.string('Inv Mass [GeV]')
            ),
            description = cms.untracked.string('Pi0 Invariant Mass in EB')
        ),
        Pi0Pt1EB = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPiZeroTask/%(prefix)sPZT%(suffix)s Pi0 Pt 1st most energetic photon'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.),
                high = cms.untracked.double(20.),
                title = cms.untracked.string('1st photon Pt [GeV]')
            ),
            description = cms.untracked.string('Pt 1st most energetic Pi0 photon in EB')
        ),
        Pi0Pt2EB = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPiZeroTask/%(prefix)sPZT%(suffix)s Pi0 Pt 2nd most energetic photon'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.),
                high = cms.untracked.double(20.),
                title = cms.untracked.string('2nd photon Pt [GeV]')
            ),
            description = cms.untracked.string('Pt 2nd most energetic Pi0 photon in EB')
        ),
        Pi0PtEB = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPiZeroTask/%(prefix)sPZT%(suffix)s Pi0 Pt'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.),
                high = cms.untracked.double(20.),
                title = cms.untracked.string('Pi0 Pt [GeV]')
            ),
            description = cms.untracked.string('Pi0 Pt in EB')
        ),
        Pi0IsoEB = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPiZeroTask/%(prefix)sPZT%(suffix)s Pi0 Iso'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                nbins = cms.untracked.int32(50),
                low = cms.untracked.double(0.),
                high = cms.untracked.double(1.),
                title = cms.untracked.string('Pi0 Iso')
            ),
            description = cms.untracked.string('Pi0 Iso in EB')
        )
    )
)
