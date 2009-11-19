import FWCore.ParameterSet.Config as cms

# -----------------------------------------------------------------------
# This contains TIB-TID-TEC-(TOB) global-level movements 
SurveyInfoScenario = cms.PSet(
    # TIB+ 
    # FIXME Assuming very small rotation angles we rotate of -asin(n_y) ~ -n_y
    #       around x axis and of asin(n_x) ~ n_x around y axis  
    TIBHalfBarrel2 = cms.PSet(
        TIBLayer1 = cms.PSet(
            phiYlocal = cms.double(0.001004),
            phiZlocal = cms.double(0.001188),
            phiXlocal = cms.double(-0.001863),
            dX = cms.double(0.08),
            dY = cms.double(-0.19)
        ),
        TIBLayer3 = cms.PSet(
            phiYlocal = cms.double(-0.000459),
            phiZlocal = cms.double(0.000274),
            phiXlocal = cms.double(-0.00043),
            dX = cms.double(0.07),
            dY = cms.double(-0.19)
        ),
        TIBLayer2 = cms.PSet(
            phiYlocal = cms.double(0.002385),
            phiZlocal = cms.double(0.001353),
            phiXlocal = cms.double(-0.002074),
            dX = cms.double(0.04),
            dY = cms.double(-0.2)
        ),
        dZ = cms.double(-0.13),
        TIBLayer4 = cms.PSet(
            phiYlocal = cms.double(0.001262),
            phiZlocal = cms.double(0.0),
            phiXlocal = cms.double(1e-06),
            dX = cms.double(0.07),
            dY = cms.double(-0.2)
        )
    ),
    saveToDbase = cms.untracked.bool(False),
    TOBHalfBarrels = cms.PSet(
        dZ = cms.double(-0.14)
    ),
    TIDEndcap2 = cms.PSet(
        TIDDisk1 = cms.PSet(
            dZ = cms.double(-0.57),
            dX = cms.double(-0.1),
            dY = cms.double(-0.15)
        ),
        TIDDisk2 = cms.PSet(
            dZ = cms.double(-0.6),
            dX = cms.double(-0.09),
            dY = cms.double(-0.19)
        ),
        TIDDisk3 = cms.PSet(
            dZ = cms.double(-0.58),
            dX = cms.double(-0.09),
            dY = cms.double(-0.22)
        ),
        phiZlocal = cms.double(0.000548)
    ),
    TECEndcap1 = cms.PSet(
        TECDisk1 = cms.PSet(
            phiXlocal = cms.double(-5e-06),
#            dY = cms.double(0.0208),
#            dZ = cms.double(0.0019),
#            dX = cms.double(0.0069),
#            phiZlocal = cms.double(1e-05),
            dY = cms.double(0.0168),
            dZ = cms.double(0.0019),
            dX = cms.double(0.0116),
            phiZlocal = cms.double(1e-05),

            phiYlocal = cms.double(6.7e-05)
        ),	
        TECDisk2 = cms.PSet(
            phiXlocal = cms.double(4.9e-05),
#            dY = cms.double(0.0261),
#            dZ = cms.double(0.0002),
#            dX = cms.double(0.0214),
#            phiZlocal = cms.double(4e-05),
            dY = cms.double(0.0118),
            dZ = cms.double(0.0002),
            dX = cms.double(0.0056),
            phiZlocal = cms.double(3.8e-05),
            phiYlocal = cms.double(-1.7e-05)
        ),
        TECDisk3 = cms.PSet(
            phiXlocal = cms.double(-6.7e-05),
#            dY = cms.double(0.0014),
#            dZ = cms.double(0.0083),
#            dX = cms.double(0.0073),
#            phiZlocal = cms.double(-2.2e-05),
            dY = cms.double(0.0090),
            dZ = cms.double(0.0085),
            dX = cms.double(0.0115),
            phiZlocal = cms.double(-2.2e-05),
            phiYlocal = cms.double(6.8e-05)
        ),	
        TECDisk4 = cms.PSet(
            phiXlocal = cms.double(-4e-05),
#            dY = cms.double(-0.0043),
#            dZ = cms.double(-0.0002),
#            dX = cms.double(-0.0101),
#            phiZlocal = cms.double(5.3e-05),
            dY = cms.double(0.0030),
            dZ = cms.double(-0.0002),
            dX = cms.double(0.0058),
            phiZlocal = cms.double(4.9e-05),
            phiYlocal = cms.double(0.000141)
        ),				
        TECDisk5 = cms.PSet(
            phiXlocal = cms.double(-0.000125),
#            dY = cms.double(-0.0339),
#            dZ = cms.double(-0.0053),
#            dX = cms.double(0.0035),
#            phiZlocal = cms.double(0.00016),
            dY = cms.double(-0.0094),
            dZ = cms.double(-0.0052),
            dX = cms.double(0.0078),
            phiZlocal = cms.double(15.5e-05),
            phiYlocal = cms.double(3.2e-05)
        ),
        TECDisk6 = cms.PSet(
            phiXlocal = cms.double(-0.000298),
#            dY = cms.double(-0.0754),
#            dZ = cms.double(0.0012),
#            dX = cms.double(-0.0304),
#            phiZlocal = cms.double(6.9e-05),
            dY = cms.double(-0.0140),
            dZ = cms.double(0.0012),
            dX = cms.double(0.0070),
            phiZlocal = cms.double(6.5e-05),
            phiYlocal = cms.double(0.000181)
        ),
        TECDisk7 = cms.PSet(
            phiXlocal = cms.double(-0.000198),
#            dY = cms.double(-0.0451),
#            dZ = cms.double(0.0061),
#            dX = cms.double(-0.0274),
#            phiZlocal = cms.double(-0.000128),
            dY = cms.double(-0.0005),
            dZ = cms.double(0.0058),
            dX = cms.double(0.0086),
            phiZlocal = cms.double(-12.2e-05),
            phiYlocal = cms.double(0.000158)
        ),		
        TECDisk8 = cms.PSet(
            phiXlocal = cms.double(-4.5e-05),
#            dY = cms.double(-0.0147),
#            dZ = cms.double(-0.0071),
#            dX = cms.double(-0.0536),
#            phiZlocal = cms.double(-5.3e-05),
            dY = cms.double(0.0011),
            dZ = cms.double(-0.0066),
            dX = cms.double(-0.0103),
            phiZlocal = cms.double(-5.0e-05),
            phiYlocal = cms.double(0.000189)
        ),
        TECDisk9 = cms.PSet(
            phiXlocal = cms.double(9.8e-05),
#            dY = cms.double(0.0168),
#            dZ = cms.double(-0.0144),
#            dX = cms.double(-0.0129),
#            phiZlocal = cms.double(-5.2e-05),
            dY = cms.double(-0.0027),
            dZ = cms.double(-0.0148),
            dX = cms.double(-0.0305),
            phiZlocal = cms.double(-5.3e-05),
            phiYlocal = cms.double(-5e-05)
        ),				
        dZ = cms.double(-0.1412),
        dX = cms.double(-0.0908),
        phiZlocal = cms.double(0.000179),
        dY = cms.double(-0.0412),
        phiYlocal = cms.double(0.000204),
		phiXlocal = cms.double(0.000107)
    ),
    TECEndcap2 = cms.PSet(
		TECDisk1 = cms.PSet(
            phiXlocal = cms.double(1.9e-05),
#            dY = cms.double(-0.0098),
#            dZ = cms.double(-0.0212),
#            dX = cms.double(-0.0222),
#            phiZlocal = cms.double(0.000361),
            dY = cms.double(-0.0027),
            dZ = cms.double(-0.0256),
            dX = cms.double(0.0227),
            phiZlocal = cms.double(12.5e-05),
            phiYlocal = cms.double(-0.000214)
        ),
		TECDisk2 = cms.PSet(
            phiXlocal = cms.double(2.8e-05),
#            dY = cms.double(-0.0053),
#            dZ = cms.double(0.0159),
#            dX = cms.double(-0.0089),
#            phiZlocal = cms.double(0.000444),
            dY = cms.double(-0.0024),
            dZ = cms.double(0.0067),
            dX = cms.double(0.0204),
            phiZlocal = cms.double(24.4e-05),
            phiYlocal = cms.double(-9.2e-05)
        ),
        TECDisk3 = cms.PSet(
            phiXlocal = cms.double(-4.2e-05),
#            dY = cms.double(0.0143),
#            dZ = cms.double(-0.0112),
#            dX = cms.double(-0.0133),
#            phiZlocal = cms.double(0.000362),
            dY = cms.double(0.0017),
            dZ = cms.double(-0.0167),
            dX = cms.double(0.0121),
            phiZlocal = cms.double(19.7e-05),
            phiYlocal = cms.double(-0.000123)
        ),
        TECDisk4 = cms.PSet(
            phiXlocal = cms.double(4.6e-05),
#            dY = cms.double(-0.013),
#            dZ = cms.double(-0.0014),
#            dX = cms.double(-0.0077),
#            phiZlocal = cms.double(0.000164),
            dY = cms.double(-0.0079),
            dZ = cms.double(-0.0067),
            dX = cms.double(0.0191),
            phiZlocal = cms.double(5.7e-05),
            phiYlocal = cms.double(-0.00011)
        ),
        TECDisk5 = cms.PSet(
            phiXlocal = cms.double(7e-05),
#            dY = cms.double(-0.0299),
#            dZ = cms.double(-0.0155),
#            dX = cms.double(-0.0176),
#            phiZlocal = cms.double(8.2e-05),
            dY = cms.double(-0.0204),
            dZ = cms.double(-0.0117),
            dX = cms.double(-0.0038),
            phiZlocal = cms.double(1.5e-05),
            phiYlocal = cms.double(-5.7e-05)
        ),
        TECDisk6 = cms.PSet(
            phiXlocal = cms.double(-6.8e-05),
#            dY = cms.double(0.0087),
#            dZ = cms.double(-0.0123),
#            dX = cms.double(-0.0264),
#            phiZlocal = cms.double(8e-05),
            dY = cms.double(-0.0059),
            dZ = cms.double(-0.0062),
            dX = cms.double(-0.0168),
            phiZlocal = cms.double(8.1e-05),
            phiYlocal = cms.double(-9.7e-05)
        ),
		TECDisk7 = cms.PSet(
            phiXlocal = cms.double(0.000176),
#            dY = cms.double(-0.0419),
#            dZ = cms.double(0.0142),
#            dX = cms.double(0.0248),
#            phiZlocal = cms.double(-0.000197),
            dY = cms.double(-0.0021),
            dZ = cms.double(0.0168),
            dX = cms.double(-0.0095),
            phiZlocal = cms.double(-10.6e-05),
            phiYlocal = cms.double(0.000153)
        ),
        TECDisk8 = cms.PSet(
            phiXlocal = cms.double(-2.9e-05),
#            dY = cms.double(0.014),
#            dZ = cms.double(0.0139),
#            dX = cms.double(-0.0318),
#            phiZlocal = cms.double(-0.000356),
            dY = cms.double(0.0186),
            dZ = cms.double(0.0152),
            dX = cms.double(-0.0167),
            phiZlocal = cms.double(-16.1e-05),
            phiYlocal = cms.double(-0.000115)
        ),				
        TECDisk9 = cms.PSet(
            phiXlocal = cms.double(-0.000204),
#            dY = cms.double(0.0488),
#            dZ = cms.double(0.0324),
#            dX = cms.double(0.0852),
#            phiZlocal = cms.double(-0.000512),
            dY = cms.double(0.0385),
            dZ = cms.double(0.0090),
            dX = cms.double(-0.0158),
            phiZlocal = cms.double(-20.7e-05),
            phiYlocal = cms.double(0.000322)
        ),
        dZ = cms.double(-0.1617),
        dX = cms.double(0.0802),
        phiZlocal = cms.double(0.001152),
        dY = cms.double(-0.0117),
        phiYlocal = cms.double(0.000385),
		phiXlocal = cms.double(-0.000105)	
    ),
    seed = cms.int32(1234567),
    # TIB-
    TIBHalfBarrel1 = cms.PSet(
        TIBLayer1 = cms.PSet(
            phiYlocal = cms.double(0.000325),
            phiZlocal = cms.double(-9.1e-05),
            phiXlocal = cms.double(0.000141),
            dX = cms.double(0.03),
            dY = cms.double(-0.25)
        ),
        TIBLayer3 = cms.PSet(
            phiYlocal = cms.double(-9.8e-05),
            phiZlocal = cms.double(-0.000823),
            phiXlocal = cms.double(0.001273),
            dX = cms.double(0.03),
            dY = cms.double(-0.18)
        ),
        TIBLayer2 = cms.PSet(
            phiYlocal = cms.double(0.000976),
            phiZlocal = cms.double(-0.001371),
            phiXlocal = cms.double(0.002088),
            dX = cms.double(0.05),
            dY = cms.double(-0.21)
        ),
        dZ = cms.double(-0.46),
        TIBLayer4 = cms.PSet(
            phiYlocal = cms.double(-0.000277),
            phiZlocal = cms.double(-0.000165),
            phiXlocal = cms.double(0.00025),
            dX = cms.double(0.02),
            dY = cms.double(-0.19)
        )
    ),
    TIDEndcap1 = cms.PSet(
        TIDDisk1 = cms.PSet(
            dZ = cms.double(0.03),
            dX = cms.double(-0.1),
            dY = cms.double(-0.15)
        ),
        TIDDisk2 = cms.PSet(
            dZ = cms.double(0.0),
            dX = cms.double(-0.09),
            dY = cms.double(-0.2)
        ),
        TIDDisk3 = cms.PSet(
            dZ = cms.double(0.06),
            dX = cms.double(-0.09),
            dY = cms.double(-0.15)
        ),
        phiZlocal = cms.double(-0.000475)
    ),
    distribution = cms.string('fixed'),
    setError = cms.bool(False),
)
####################################
SurveyInfoScenario_noTIBlay = cms.PSet(
    saveToDbase = cms.untracked.bool(False),
    TOBHalfBarrels = cms.PSet(
        dZ = cms.double(-0.14)
    ),
    TIDEndcap2 = cms.PSet(
        TIDDisk1 = cms.PSet(
            dZ = cms.double(-0.57),
            dX = cms.double(-0.1),
            dY = cms.double(-0.15)
        ),
        TIDDisk2 = cms.PSet(
            dZ = cms.double(-0.6),
            dX = cms.double(-0.09),
            dY = cms.double(-0.19)
        ),
        TIDDisk3 = cms.PSet(
            dZ = cms.double(-0.58),
            dX = cms.double(-0.09),
            dY = cms.double(-0.22)
        ),
        phiZlocal = cms.double(0.000548)
    ),
    TECEndcap1 = cms.PSet(
        TECDisk9 = cms.PSet(
            phiXlocal = cms.double(9.8e-05),
            dY = cms.double(0.0168),
            dZ = cms.double(-0.0144),
            dX = cms.double(-0.0129),
            phiZlocal = cms.double(-5.2e-05),
            phiYlocal = cms.double(-5e-05)
        ),
        phiXlocal = cms.double(0.000107),
        TECDisk5 = cms.PSet(
            phiXlocal = cms.double(-0.000125),
            dY = cms.double(-0.0339),
            dZ = cms.double(-0.0053),
            dX = cms.double(0.0035),
            phiZlocal = cms.double(0.00016),
            phiYlocal = cms.double(3.2e-05)
        ),
        TECDisk4 = cms.PSet(
            phiXlocal = cms.double(-4e-05),
            dY = cms.double(-0.0043),
            dZ = cms.double(-0.0002),
            dX = cms.double(-0.0101),
            phiZlocal = cms.double(5.3e-05),
            phiYlocal = cms.double(0.000141)
        ),
        TECDisk7 = cms.PSet(
            phiXlocal = cms.double(-0.000198),
            dY = cms.double(-0.0451),
            dZ = cms.double(0.0061),
            dX = cms.double(-0.0274),
            phiZlocal = cms.double(-0.000128),
            phiYlocal = cms.double(0.000158)
        ),
        TECDisk6 = cms.PSet(
            phiXlocal = cms.double(-0.000298),
            dY = cms.double(-0.0754),
            dZ = cms.double(0.0012),
            dX = cms.double(-0.0304),
            phiZlocal = cms.double(6.9e-05),
            phiYlocal = cms.double(0.000181)
        ),
        TECDisk1 = cms.PSet(
            phiXlocal = cms.double(-5e-06),
            dY = cms.double(0.0208),
            dZ = cms.double(0.0019),
            dX = cms.double(0.0069),
            phiZlocal = cms.double(1e-05),
            phiYlocal = cms.double(6.7e-05)
        ),
        TECDisk3 = cms.PSet(
            phiXlocal = cms.double(-6.7e-05),
            dY = cms.double(0.0014),
            dZ = cms.double(0.0083),
            dX = cms.double(0.0073),
            phiZlocal = cms.double(-2.2e-05),
            phiYlocal = cms.double(6.8e-05)
        ),
        TECDisk2 = cms.PSet(
            phiXlocal = cms.double(4.9e-05),
            dY = cms.double(0.0261),
            dZ = cms.double(0.0002),
            dX = cms.double(0.0214),
            phiZlocal = cms.double(4e-05),
            phiYlocal = cms.double(-1.7e-05)
        ),
        dZ = cms.double(-0.1412),
        dX = cms.double(-0.0908),
        phiZlocal = cms.double(0.000179),
        dY = cms.double(-0.0412),
        phiYlocal = cms.double(0.000204),
        TECDisk8 = cms.PSet(
            phiXlocal = cms.double(-4.5e-05),
            dY = cms.double(-0.0147),
            dZ = cms.double(-0.0071),
            dX = cms.double(-0.0536),
            phiZlocal = cms.double(-5.3e-05),
            phiYlocal = cms.double(0.000189)
        )
    ),
    TECEndcap2 = cms.PSet(
        TECDisk9 = cms.PSet(
            phiXlocal = cms.double(-0.000204),
            dY = cms.double(0.0488),
            dZ = cms.double(0.0324),
            dX = cms.double(0.0852),
            phiZlocal = cms.double(-0.000512),
            phiYlocal = cms.double(0.000322)
        ),
        phiXlocal = cms.double(-0.000105),
        TECDisk5 = cms.PSet(
            phiXlocal = cms.double(7e-05),
            dY = cms.double(-0.0299),
            dZ = cms.double(-0.0155),
            dX = cms.double(-0.0176),
            phiZlocal = cms.double(8.2e-05),
            phiYlocal = cms.double(-5.7e-05)
        ),
        TECDisk4 = cms.PSet(
            phiXlocal = cms.double(4.6e-05),
            dY = cms.double(-0.013),
            dZ = cms.double(-0.0014),
            dX = cms.double(-0.0077),
            phiZlocal = cms.double(0.000164),
            phiYlocal = cms.double(-0.00011)
        ),
        TECDisk7 = cms.PSet(
            phiXlocal = cms.double(0.000176),
            dY = cms.double(-0.0419),
            dZ = cms.double(0.0142),
            dX = cms.double(0.0248),
            phiZlocal = cms.double(-0.000197),
            phiYlocal = cms.double(0.000153)
        ),
        TECDisk6 = cms.PSet(
            phiXlocal = cms.double(-6.8e-05),
            dY = cms.double(0.0087),
            dZ = cms.double(-0.0123),
            dX = cms.double(-0.0264),
            phiZlocal = cms.double(8e-05),
            phiYlocal = cms.double(-9.7e-05)
        ),
        TECDisk1 = cms.PSet(
            phiXlocal = cms.double(1.9e-05),
            dY = cms.double(-0.0098),
            dZ = cms.double(-0.0212),
            dX = cms.double(-0.0222),
            phiZlocal = cms.double(0.000361),
            phiYlocal = cms.double(-0.000214)
        ),
        TECDisk3 = cms.PSet(
            phiXlocal = cms.double(-4.2e-05),
            dY = cms.double(0.0143),
            dZ = cms.double(-0.0112),
            dX = cms.double(-0.0133),
            phiZlocal = cms.double(0.000362),
            phiYlocal = cms.double(-0.000123)
        ),
        TECDisk2 = cms.PSet(
            phiXlocal = cms.double(2.8e-05),
            dY = cms.double(-0.0053),
            dZ = cms.double(0.0159),
            dX = cms.double(-0.0089),
            phiZlocal = cms.double(0.000444),
            phiYlocal = cms.double(-9.2e-05)
        ),
        dZ = cms.double(-0.1617),
        dX = cms.double(0.0802),
        phiZlocal = cms.double(0.001152),
        dY = cms.double(-0.0117),
        phiYlocal = cms.double(0.000385),
        TECDisk8 = cms.PSet(
            phiXlocal = cms.double(-2.9e-05),
            dY = cms.double(0.014),
            dZ = cms.double(0.0139),
            dX = cms.double(-0.0318),
            phiZlocal = cms.double(-0.000356),
            phiYlocal = cms.double(-0.000115)
        )
    ),
    seed = cms.int32(1234567),
    TIDEndcap1 = cms.PSet(
        TIDDisk1 = cms.PSet(
            dZ = cms.double(0.03),
            dX = cms.double(-0.1),
            dY = cms.double(-0.15)
        ),
        TIDDisk2 = cms.PSet(
            dZ = cms.double(0.0),
            dX = cms.double(-0.09),
            dY = cms.double(-0.2)
        ),
        TIDDisk3 = cms.PSet(
            dZ = cms.double(0.06),
            dX = cms.double(-0.09),
            dY = cms.double(-0.15)
        ),
        phiZlocal = cms.double(-0.000475)
    ),
    distribution = cms.string('fixed'),
    setError = cms.bool(False),
)

