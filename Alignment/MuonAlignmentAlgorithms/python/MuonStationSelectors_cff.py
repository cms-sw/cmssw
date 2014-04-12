import FWCore.ParameterSet.Config as cms

mb1 = cms.PSet(
  rRanges = cms.vdouble(370., 470.),
  zRanges = cms.vdouble(-560., 560.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

mb2 = cms.PSet(
  rRanges = cms.vdouble(470., 570.),
  zRanges = cms.vdouble(-560., 560.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

mb3 = cms.PSet(
  rRanges = cms.vdouble(570., 670.),
  zRanges = cms.vdouble(-560., 560.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

mb4 = cms.PSet(
  rRanges = cms.vdouble(670., 870.),
  zRanges = cms.vdouble(-560., 560.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus1 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(500., 750.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus11 = cms.PSet(
  rRanges = cms.vdouble(50., 275.),
  zRanges = cms.vdouble(500., 700.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus12 = cms.PSet(
  rRanges = cms.vdouble(275., 480.),
  zRanges = cms.vdouble(650., 750.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus13 = cms.PSet(
  rRanges = cms.vdouble(480., 800.),
  zRanges = cms.vdouble(650., 750.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus2 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(750., 875.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus21 = cms.PSet(
  rRanges = cms.vdouble(50., 350.),
  zRanges = cms.vdouble(750., 875.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus22 = cms.PSet(
  rRanges = cms.vdouble(350., 800.),
  zRanges = cms.vdouble(750., 875.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus3 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(875., 980.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus31 = cms.PSet(
  rRanges = cms.vdouble(50., 350.),
  zRanges = cms.vdouble(875., 980.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus32 = cms.PSet(
  rRanges = cms.vdouble(350., 800.),
  zRanges = cms.vdouble(875., 980.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus4 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(980., 1100.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meplus41 = cms.PSet(
  rRanges = cms.vdouble(50., 350.),
  zRanges = cms.vdouble(980., 1100.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus1 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(-750., -500.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus11 = cms.PSet(
  rRanges = cms.vdouble(50., 275.),
  zRanges = cms.vdouble(-700., -500.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus12 = cms.PSet(
  rRanges = cms.vdouble(275., 480.),
  zRanges = cms.vdouble(-750., -650.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus13 = cms.PSet(
  rRanges = cms.vdouble(480., 800.),
  zRanges = cms.vdouble(-750., -650.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus2 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(-875., -750.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus21 = cms.PSet(
  rRanges = cms.vdouble(50., 350.),
  zRanges = cms.vdouble(-875., -750.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus22 = cms.PSet(
  rRanges = cms.vdouble(350., 800.),
  zRanges = cms.vdouble(-875., -750.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus3 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(-980., -875.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus31 = cms.PSet(
  rRanges = cms.vdouble(50., 350.),
  zRanges = cms.vdouble(-980., -875.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus32 = cms.PSet(
  rRanges = cms.vdouble(350., 800.),
  zRanges = cms.vdouble(-980., -875.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus4 = cms.PSet(
  rRanges = cms.vdouble(),
  zRanges = cms.vdouble(-1100., -980.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

meminus41 = cms.PSet(
  rRanges = cms.vdouble(50., 350.),
  zRanges = cms.vdouble(-1100., -980.),
  etaRanges = cms.vdouble(), phiRanges = cms.vdouble(), xRanges = cms.vdouble(), yRanges = cms.vdouble()
  )

MuonStationSelectors = {"mb1": mb1,
                        "mb2": mb2,
                        "mb3": mb3,
                        "mb4": mb4,
                        "meplus1": meplus1,
                        "meplus11": meplus11,
                        "meplus12": meplus12,
                        "meplus13": meplus13,
                        "meplus2": meplus2,
                        "meplus21": meplus21,
                        "meplus22": meplus22,
                        "meplus3": meplus3,
                        "meplus31": meplus31,
                        "meplus32": meplus32,
                        "meplus4": meplus4,
                        "meplus41": meplus41,
                        "meminus1": meminus1,
                        "meminus11": meminus11,
                        "meminus12": meminus12,
                        "meminus13": meminus13,
                        "meminus2": meminus2,
                        "meminus21": meminus21,
                        "meminus22": meminus22,
                        "meminus3": meminus3,
                        "meminus31": meminus31,
                        "meminus32": meminus32,
                        "meminus4": meminus4,
                        "meminus41": meminus41,
                        }
