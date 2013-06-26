import FWCore.ParameterSet.Config as cms

TracoParametersBlock = cms.PSet(
    TracoParameters = cms.PSet(
        SPRGCOMP = cms.int32(2),

        FHTMSK = cms.int32(0), ## single HTRIG enabling on first/second tracks

        DD = cms.int32(18), ## DD traco parameter: this is fixed

        SSLMSK = cms.int32(0),
        LVALIDIFH = cms.int32(0), ## flag for Low validation parameter

        FHTPRF = cms.int32(1), ## preference to HTRIG on first/second tracks

        FSLMSK = cms.int32(0), ## preference to inner on first/second tracks

        SHTPRF = cms.int32(1),
        Debug = cms.untracked.int32(0), ## Debug flag

        SHTMSK = cms.int32(0),
        TRGENB3 = cms.int32(1),
        SHISM = cms.int32(0),
        IBTIOFF = cms.int32(0), ## IBTIOFF traco parameter

        KPRGCOM = cms.int32(255), ## bending angle cut for all stations and triggers

        KRAD = cms.int32(0), ## KRAD traco parameter: fixed due to hardware bug

        FLTMSK = cms.int32(1), ## single LTRIG enabling on first/second tracks

        LTS = cms.int32(0), ## suppr. of LTRIG in 4 BX before HTRIG

        SLTMSK = cms.int32(1),
        FPRGCOMP = cms.int32(2), ## K tollerance for correlation in TRACO

        TRGENB9 = cms.int32(1),
        TRGENB8 = cms.int32(1),
        LTF = cms.int32(0), ## single LTRIG accept enabling on first/second tracks

        TRGENB1 = cms.int32(1),
        TRGENB0 = cms.int32(1), ## bti masks

        FHISM = cms.int32(0), ## ascend. order for K sorting first/second tracks

        TRGENB2 = cms.int32(1),
        TRGENB5 = cms.int32(1),
        TRGENB4 = cms.int32(1),
        TRGENB7 = cms.int32(1),
        TRGENB6 = cms.int32(1),
        TRGENB15 = cms.int32(1),
        TRGENB14 = cms.int32(1),
        TRGENB11 = cms.int32(1),
        TRGENB10 = cms.int32(1),
        TRGENB13 = cms.int32(1),
        TRGENB12 = cms.int32(1),
        REUSEO = cms.int32(1),
        REUSEI = cms.int32(1), ## recycling of TRACO cand. in inner/outer SL

        BTIC = cms.int32(32) ## BTIC traco parameter

    )
)


