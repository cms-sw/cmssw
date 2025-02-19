import FWCore.ParameterSet.Config as cms

TSPhiParametersBlock = cms.PSet(
    TSPhiParameters = cms.PSet(
        #Enable/Disable Inner SL checking for 1st/2nd tracks & carry in TSM
        TSMNOE1 = cms.bool(True),
        TSMNOE2 = cms.bool(False),
        TSTREN12 = cms.bool(True),
        TSTREN9 = cms.bool(True),
        TSTREN8 = cms.bool(True),
        TSTREN11 = cms.bool(True),
        TSTREN3 = cms.bool(True),
        TSTREN2 = cms.bool(True),
        TSTREN1 = cms.bool(True),
        # Used Traco mask
        # 1 means enabled
        TSTREN0 = cms.bool(True),
        TSTREN7 = cms.bool(True),
        TSTREN6 = cms.bool(True),
        TSTREN5 = cms.bool(True),
        TSTREN4 = cms.bool(True),
        TSSNOE2 = cms.bool(False),
        # Enable/Disable Inner SL checking for 1st/2nd tracks & carry in TSS
        TSSNOE1 = cms.bool(True),
        TSMCCE2 = cms.bool(False),
        TSTREN19 = cms.bool(True),
        #Enable/Disable correlation checking for 1st/2nd tracks & carry in TSM
        TSMCCE1 = cms.bool(True),
        TSTREN17 = cms.bool(True),
        TSTREN16 = cms.bool(True),
        TSTREN15 = cms.bool(True),
        TSTREN14 = cms.bool(True),
        TSTREN13 = cms.bool(True),
        # Priority in TSS for 1st/2nd tracks
        # 1 is H/L
        # 2 is In/Out
        # 3 is Corr/NotCorr
        # valid parameters are 1,2,3 combinations
        TSSMSK1 = cms.int32(312),
        TSSMSK2 = cms.int32(312),
        TSTREN10 = cms.bool(True),
        TSMMSK2 = cms.int32(312),
        # Priority in TSM for 1st/2nd tracks
        # 1 is H/L
        # 2 is In/Out
        # 3 is Corr/NotCorr
        # valid parameters are 1,2,3 combinations
        TSMMSK1 = cms.int32(312),
        # Handling of second track (carry) in case of pile-up in TSM
        # 1 Get best 2nd previous BX
        # 2 Get best 2nd previous BX if 1st is Low
        # 0 Reject 2nd track
        TSMHSP = cms.int32(1),
        # Enable/Disable correlation checking for 1st/2nd tracks & carry in TSS
        TSSCCE1 = cms.bool(True),
        TSSCCE2 = cms.bool(False),
        # Correlated ghost 2 suppression option in TSS
        # 0 Reject also if Correlated
        # 1 Accept if correlated
        TSSCGS2 = cms.bool(True),
        TSSCCEC = cms.bool(False),
        # Enable/Disable Htrig checking for 1st/2nd tracks & carry in TSM
        TSMHTE1 = cms.bool(True),
        TSMHTE2 = cms.bool(False),
        # Debug flag
        Debug = cms.untracked.bool(False),
        TSSHTE2 = cms.bool(False),
        # Correlated ghost 1 suppression option in TSM
        # 0 Reject also if Correlated
        # 1 Accept if correlated
        TSMCGS1 = cms.bool(True),
        # Correlated ghost 2 suppression option in TSM
        # 0 Reject also if Correlated
        # 1 Accept if correlated
        TSMCGS2 = cms.bool(True),
        # Enable/Disable Htrig checking for 1st/2nd tracks & carry in TSS
        TSSHTE1 = cms.bool(True),
        TSTREN22 = cms.bool(True),
        TSSNOEC = cms.bool(False),
        TSTREN20 = cms.bool(True),
        TSTREN21 = cms.bool(True),
        # Ghost 1 suppression options in TSM
        # 1 If Outer adj to 1st tr
        # 2 Always
        # 0 Never
        TSMGS1 = cms.int32(1),
        # Ghost 2 suppression options in TSM
        # 1 If Outer same TRACO of uncorr 1st tr
        # 2 If Outer same TRACO of 1st tr
        # 3 Always
        # 4 If Outer same TRACO of inner 1st tr
        # 0 Never
        TSMGS2 = cms.int32(1),
        TSSHTEC = cms.bool(False),
        # TsmWord used to mask TSMS or TSS
        #  bit numbering 7 6 5 4 3 2 1 0
        #
        #  bit 0 = 1  --> TSMS OK     => normal mode (default)
        #  bit 0 = 0  --> TSMS broken => back-up mode (see example a)
        #  bits 1-6 = 0 --> broken TSS (see example b)
        #  bits 1-6 = 1 --> working TSS (default)
        TSMWORD = cms.int32(255),
        TSMHTEC = cms.bool(False),
        # Correlated ghost 1 suppression option in TSS
        # 0 Reject also if Correlated
        # 1 Accept if correlated
        TSSCGS1 = cms.bool(True),
        TSTREN23 = cms.bool(True),
        # Ghost 2 suppression options in TSS
        # 1 If Outer same TRACO of uncorr 1st tr
        # 2 If Outer same TRACO of 1st tr
        # 3 Always
        # 4 If Outer same TRACO of inner 1st tr
        # 0 Never
        TSSGS2 = cms.int32(1),
        TSMNOEC = cms.bool(False),
        # Ghost 1 suppression options in TSS
        # 1 If Outer adj to 1st tr
        # 2 Always
        # 0 Never
        TSSGS1 = cms.int32(1),
        TSTREN18 = cms.bool(True),
        TSMCCEC = cms.bool(False)
    )
)


