import FWCore.ParameterSet.Config as cms

Psi1_pset    = cms.PSet( name = cms.string( 'Psi1' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 2.00 ),
                      massMax = cms.double( 3.40 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( 3.096916 ),
                  constrSigma = cms.double( 0.000040 )
)
PhiMuMu_pset = cms.PSet( name = cms.string( 'PhiMuMu' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 0.50 ),
                      massMax = cms.double( 1.50 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( 1.019461 ),
                  constrSigma = cms.double( 0.004266 )
)
Psi2_pset    = cms.PSet( name = cms.string( 'Psi2' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 3.40 ),
                      massMax = cms.double( 6.00 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( 3.686109 ),
                  constrSigma = cms.double( 0.000129 )
)
Ups_pset     = cms.PSet( name = cms.string( 'Ups' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double(  6.00 ),
                      massMax = cms.double( 12.00 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( -1.0 ),
                  constrSigma = cms.double( -1.0 )
)
Ups1_pset    = cms.PSet( name = cms.string( 'Ups1' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 6.00 ),
                      massMax = cms.double( 9.75 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( 9.46030 ),
                  constrSigma = cms.double( 0.00026 )
)
Ups2_pset    = cms.PSet( name = cms.string( 'Ups2' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double(  9.75 ),
                      massMax = cms.double( 10.20 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( 10.02326 ),
                  constrSigma = cms.double(  0.00031 )
)
Ups3_pset    = cms.PSet( name = cms.string( 'Ups3' ),
                        ptMin = cms.double( 2.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 10.20 ),
                      massMax = cms.double( 12.00 ),
                      probMin = cms.double( -1.0 ),
                  constrMass  = cms.double( 10.3552 ),
                  constrSigma = cms.double(  0.0005 )
)
Kx0_pset     = cms.PSet( name = cms.string( 'Kx0' ),
                        ptMin = cms.double( 0.7 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 0.75 ),
                      massMax = cms.double( 1.05 ),
                      probMin = cms.double( 0.0 ),
                  constrMass  = cms.double( -1.0 ),
                  constrSigma = cms.double( -1.0 )
)
PhiKK_pset   = cms.PSet( name = cms.string( 'PhiKK' ),
                        ptMin = cms.double( 0.7 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 1.00 ),
                      massMax = cms.double( 1.04 ),
                      probMin = cms.double( 0.0 ),
                  constrMass  = cms.double( -1.0 ),
                  constrSigma = cms.double( -1.0 )
)
Bu_pset      = cms.PSet( name = cms.string( 'Bu' ),
                        ptMin = cms.double( 0.7 ),
                       etaMax = cms.double( 10.0 ),
                      mJPsiMin = cms.double( 2.80 ),
                      mJPsiMax = cms.double( 3.40 ),
                       massMin = cms.double( 3.50 ),
                       massMax = cms.double( 8.00 ),
                       probMin = cms.double( 0.02 ),
                    massFitMin = cms.double( 5.00 ),
                    massFitMax = cms.double( 6.00 ),
                   constrMJPsi = cms.bool( True )
)
Bp_pset      = cms.PSet( name = cms.string( 'Bp' ),
                        ptMin = cms.double( 0.7 ),
                       etaMax = cms.double( 10.0 ),
                      mJPsiMin = cms.double( 2.80 ),
                      mJPsiMax = cms.double( 3.40 ),
                       massMin = cms.double( 3.50 ),
                       massMax = cms.double( 8.00 ),
                       probMin = cms.double( 0.02 ),
                    massFitMin = cms.double( 5.00 ),
                    massFitMax = cms.double( 6.00 ),
                   constrMJPsi = cms.bool( False ),
                   constrMPsi2 = cms.bool( True )
)
Bd_pset      = cms.PSet( name = cms.string( 'Bd' ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                      mKx0Min = cms.double( 0.77 ),
                      mKx0Max = cms.double( 1.02 ),
                      massMin = cms.double( 3.50 ),
                      massMax = cms.double( 8.00 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 5.00 ),
                   massFitMax = cms.double( 6.00 ),
                  constrMJPsi = cms.bool( True )
)
Bs_pset      = cms.PSet( name = cms.string( 'Bs' ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                      mPhiMin = cms.double( 1.005 ),
                      mPhiMax = cms.double( 1.035 ),
                      massMin = cms.double( 3.50 ),
                      massMax = cms.double( 8.00 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 5.00 ),
                   massFitMax = cms.double( 6.00 ),
                  constrMJPsi = cms.bool( True )
)
K0s_pset     = cms.PSet( name = cms.string( 'K0s' ),
                        ptMin = cms.double( 0.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 0.0 ),
                      massMax = cms.double( 20.0 ),
                      probMin = cms.double( -1.0 )
)
Lambda0_pset = cms.PSet( name = cms.string( 'Lambda0' ),
                        ptMin = cms.double( 0.0 ),
                       etaMax = cms.double( 10.0 ),
                      massMin = cms.double( 0.0 ),
                      massMax = cms.double( 20.0 ),
                      probMin = cms.double( -1.0 )
)
B0_pset      = cms.PSet( name = cms.string( 'B0' ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                      mK0sMin = cms.double( 0.00 ),
                      mK0sMax = cms.double( 2.00 ),
                      massMin = cms.double( 3.50 ),
                      massMax = cms.double( 8.00 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 5.00 ),
                   massFitMax = cms.double( 6.00 ),
                  constrMJPsi = cms.bool( True )
)
Lambdab_pset = cms.PSet( name = cms.string( 'Lambdab' ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                  mLambda0Min = cms.double( 0.00 ),
                  mLambda0Max = cms.double( 3.00 ),
                      massMin = cms.double( 3.50 ),
                      massMax = cms.double( 8.00 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 5.00 ),
                   massFitMax = cms.double( 6.00 ),
                  constrMJPsi = cms.bool( True )
)
Bc_pset      = cms.PSet( name = cms.string( 'Bc' ),
                        ptMin = cms.double( 3.0 ),
                       etaMax = cms.double( 10.0 ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                      massMin = cms.double( 4.00 ),
                      massMax = cms.double( 9.00 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 6.00 ),
                   massFitMax = cms.double( 7.00 ),
                  constrMJPsi = cms.bool( True )
)
Psi2S_pset   = cms.PSet( name = cms.string( 'Psi2S' ),
                        ptMin = cms.double( 1.0 ),
                       etaMax = cms.double( 10.0 ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                      massMin = cms.double( 3.00 ),
                      massMax = cms.double( 4.50 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 3.60 ),
                   massFitMax = cms.double( 3.80 ),
                  constrMJPsi = cms.bool( True )
)
X3872_pset   = cms.PSet( name = cms.string( 'X3872' ),
                        ptMin = cms.double( 1.0 ),
                       etaMax = cms.double( 10.0 ),
                     mJPsiMin = cms.double( 2.80 ),
                     mJPsiMax = cms.double( 3.40 ),
                      massMin = cms.double( 3.00 ),
                      massMax = cms.double( 4.50 ),
                      probMin = cms.double( 0.02 ),
                   massFitMin = cms.double( 3.80 ),
                   massFitMax = cms.double( 4.00 ),
                  constrMJPsi = cms.bool( True )
)


recoSelect = cms.VPSet(
     Psi1_pset,
  PhiMuMu_pset,
     Psi2_pset,
      Ups_pset,
     Ups1_pset,
     Ups2_pset,
     Ups3_pset,
      Kx0_pset,
    PhiKK_pset,
       Bu_pset,
       Bp_pset,
       Bd_pset,
       Bs_pset,
      K0s_pset,
  Lambda0_pset,
       B0_pset,
  Lambdab_pset,
       Bc_pset,
    Psi2S_pset,
    X3872_pset
)
