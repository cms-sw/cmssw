import FWCore.ParameterSet.Config as cms

BeamHaloGenFilter = cms.EDFilter("BHFilter",
    scintillators_type = cms.int32(-1),
    #	How to setup the "trigger" conditions
    # 
    #     Trig Type =     -1      only minus MSC
    #                     0       both + and - side
    #                     1       only plus BSC
    #
    #     Scintillators Type      -1      only PADs
    #                             0       both PADs and DISK
    #                             +1      only DISKs
    #
    trig_type = cms.int32(0),
    #           bool filter = true
    # Fast magnetic-field propagator **** not yet used ****
    # Beam Halo muons are straight propagated 
    #           untracked bool InTK=true
    # The radii of the cylinders for propagation
    radii = cms.vdouble(700.0, 638.0, 597.0, 536.0, 491.0, 
        462.0, 403.0, 385.0, 340.0, 329.0, 
        315.0, 200.0, 96.5, 61.0, 41.8, 
        34.2),
    # The corresponding values of the magnetic field
    bfiel = cms.vdouble(-0.05, -1.78, -0.01, -1.75, -0.02, 
        -1.86, -0.02, -1.99, -0.05, 1.0, 
        3.1, 4.16, 4.1, 4.07, 4.07, 
        4.07),
    # The overall reduction factor for the magnetic field
    factor = cms.double(1.0),
    # The zeds of the cylinders for propagation :
    zeds = cms.vdouble(600.0, 600.0, 600.0, 600.0, 600.0, 
        600.0, 600.0, 600.0, 600.0, 600.0, 
        600.0, 600.0, 600.0, 600.0, 600.0, 
        600.0)
)


