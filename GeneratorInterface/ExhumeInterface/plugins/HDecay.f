C         Last modification on February 27th 1997 by M.S.
C ==================================================================
C ================= PROGRAM HDECAY: COMMENTS =======================
C ==================================================================
C
C                       ***************
C                       * VERSION 2.0 *
C                       ***************
C
C
C  This program calculates the total decay widths and the branching 
C  ratios of the C Standard Model Higgs boson (HSM) as well as those 
C  of the neutral (HL= the light CP-even, HH= the heavy CP-even, HA= 
C  the pseudoscalar) and the charged (HC) Higgs bosons of the Minimal
C  Supersymmetric extension of the Standard Model (MSSM). It includes:
C
C - All the decay channels which are kinematically allowed and which
C   have branching ratios larger than 10**(-4). 
C
C - All QCD corrections to the fermionic and gluonic decay modes.
C   Most of these corrections are mapped into running masses in a
C   consistent way with some freedom for including high order terms. 
C
C - Below--threshold three--body decays with off--shell top quarks
C   or ONE off-shell gauge boson, as well as some decays with one
C   off-shell Higgs boson in the MSSM. 
C
C - Double off-shell decays: HSM,HL,HH --> W*W*,Z*Z* -->4 fermions,
C   which could be important for Higgs masses close to MW or MZ.
C
C - In the MSSM, the radiative corrections with full squark mixing and 
C   uses the RG improved values of Higgs masses and couplings with the 
C   main NLO corrections implemented (based on M.Carena, M. Quiros and
C   C.E.M. Wagner, hep-ph/9508343). 
C
C - In the MSSM, all the decays into CHARGINOS, NEUTRALINOS, SLEPTONS 
C   and SQUARKS (with mixing in the stop and sbottom sectors). 
C
C - Chargino, slepton and squark loops in the 2 photon decays and squark
C   loops in the gluonic decays (including QCD corrections). 
C
C  ===================================================================
C  This peogram has been written by A.Djouadi, J.Kalinowski and M.Spira.
C  For details on how to use the program see: hep-ph/96XXX. For any
C  question, comment, suggestion or complaint, please contact us at:


C ================ IT USES AS INPUT PARAMETERS:
C
C   IHIGGS: =0: CALCULATE BRANCHING RATIOS OF SM HIGGS BOSON
C           =1: CALCULATE BRANCHING RATIOS OF MSSM h BOSON
C           =2: CALCULATE BRANCHING RATIOS OF MSSM H BOSON
C           =3: CALCULATE BRANCHING RATIOS OF MSSM A BOSON
C           =4: CALCULATE BRANCHING RATIOS OF MSSM H+ BOSON
C           =5: CALCULATE BRANCHING RATIOS OF ALL MSSM HIGGS BOSONS
C
C TGBET:    TAN(BETA) FOR MSSM
C MABEG:    START VALUE OF M_A FOR MSSM AND M_H FOR SM
C MAEND:    END VALUE OF M_A FOR MSSM AND M_H FOR SM
C NMA:      NUMBER OF ITERATIONS FOR M_A
C ALS(MZ):  VALUE FOR ALPHA_S(M_Z)
C MSBAR(1): MSBAR MASS OF STRANGE QUARK AT SCALE Q=1 GEV
C MC:       CHARM POLE MASS
C MB:       BOTTOM POLE MASS
C MT:       TOP POLE MASS
C MTAU:     TAU MASS
C MMUON:    MUON MASS
C ALPH:     INVERSE QED COUPLING
C GF:       FERMI CONSTANT
C GAMW:     W WIDTH
C GAMZ:     Z WIDTH
C MZ:       Z MASS
C MW:       W MASS
C VUS:      CKM PARAMETER V_US
C VCB:      CKM PARAMETER V_CB
C VUB/VCB:  RATIO V_UB/V_CB
C NNLO (M): =1: USE O(ALPHA_S) FORMULA FOR POLE MASS --> MSBAR MASS
C           =2: USE O(ALPHA_S**2) FORMULA FOR POLE MASS --> MSBAR MASS
C
C ON-SHELL: =0: INCLUDE OFF_SHELL DECAYS H,A --> T*T*, A --> Z*H,
C               H --> W*H+,Z*A, H+ --> W*A, W*H, T*B
C           =1: EXCLUDE THE OFF-SHELL DECAYS ABOVE
C
C ON-SH-WZ: =0: INCLUDE DOUBLE OFF-SHELL PAIR DECAYS PHI --> W*W*,Z*Z*
C           =1: INCLUDE ONLY SINGLE OFF-SHELL DECAYS PHI --> W*W,Z*Z
C
C IPOLE:    =0 COMPUTES RUNNING HIGGS MASSES (FASTER) 
C           =1 COMPUTES POLE HIGGS MASSES 
C
C OFF-SUSY: =0: INCLUDE DECAYS (AND LOOPS) INTO SUPERSYMMETRIC PARTICLES
C           =1: EXCLUDE DECAYS (AND LOOPS) INTO SUPERSYMMETRIC PARTICLES
C
C INIDEC:   =0: PRINT OUT SUMS OF CHARGINO/NEUTRALINO/SFERMION DECAYS
C           =1: PRINT OUT INDIVIDUAL CHARGINO/NEUTRALINO/SFERMION DECAYS
C
C NF-GG:    NUMBER OF LIGHT FLAVORS INCLUDED IN THE GLUONIC DECAYS 
C            PHI --> GG* --> GQQ (3,4 OR 5)

C =====================================================================
C =========== BEGINNING OF THE SUBROUTINE FOR THE DECAYS ==============
C !!!!!!!!!!!!!! Any change below this line is at your own risk!!!!!!!!
C =====================================================================

C fix unused TGBET warning  
C     SUBROUTINE HDEC(TGBET)
      SUBROUTINE HDEC()
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION LAMB
      DIMENSION XX(4),YY(4)
      COMPLEX*16 CF,CG,CI1,CI2,CA,CB,CTT,CTB,CTC,CTW,CLT,CLB,CLW,
     .           CAT,CAB,CAC,CAW,CTL,CAL
      COMMON/HMASS/AMSM,AMA,AML,AMH,AMCH
      COMMON/MASSES/AMS,AMC,AMB,AMT
      COMMON/ALS/XLAMBDA,AMC0,AMB0,AMT0,N0
      COMMON/PARAM/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      COMMON/CKMPAR/VUS,VCB,VUB
      COMMON/WZWDTH/GAMC0,GAMT0,GAMT1,GAMW,GAMZ
      COMMON/ONSHELL/IONSH,IONWZ,IOFSUSY
      COMMON/OLDFASH/NFGG
      COMMON/FLAG/IHIGGS,NNLO,IPOLE
      COMMON/WIDTHSM/SMBRB,SMBRL,SMBRM,SMBRS,SMBRC,SMBRT,SMBRG,SMBRGA,
     .               SMBRZGA,SMBRW,SMBRZ,SMWDTH
      COMMON/COUP/GAT,GAB,GLT,GLB,GHT,GHB,GZAH,GZAL,
     .            GHHH,GLLL,GHLL,GLHH,GHAA,GLAA,GLVV,GHVV,
     .            GLPM,GHPM,B,A
      BETA(X)=DSQRT(1.D0-4.D0*X)
      LAMB(X,Y)=DSQRT((1.D0-X-Y)**2-4.D0*X*Y)
      HVV(X,Y)= GF/(4.D0*PI*DSQRT(2.D0))*X**3/2.D0*BETA(Y)
     .            *(1.D0-4.D0*Y+12.D0*Y**2)
      HFF(X,Y)= GF/(4*PI*DSQRT(2.D0))*X**3*Y*(BETA(Y))**3
      HV(V)=3.D0*(1.D0-8.D0*V+20.D0*V**2)/DSQRT((4.D0*V-1.D0))
     .      *DACOS((3.D0*V-1.D0)/2.D0/DSQRT(V**3))
     .      -(1.D0-V)*(47.D0/2.D0*V-13.D0/2.D0+1.D0/V)
     .      -3.D0/2.D0*(1.D0-6.D0*V+4.D0*V**2)*DLOG(V)
      QCD0(X) = (1+X**2)*(4*SP((1-X)/(1+X)) + 2*SP((X-1)/(X+1))
     .        - 3*DLOG((1+X)/(1-X))*DLOG(2/(1+X))
     .        - 2*DLOG((1+X)/(1-X))*DLOG(X))
     .        - 3*X*DLOG(4/(1-X**2)) - 4*X*DLOG(X)
      HQCDM(X)=QCD0(X)/X+(3+34*X**2-13*X**4)/16/X**3*DLOG((1+X)/(1-X))
     .        + 3.D0/8/X**2*(7*X**2-1)
      HQCD(X)=(4.D0/3*HQCDM(BETA(X))
     .        +2*(4.D0/3-DLOG(X))*(1-10*X)/(1-4*X))*ASH/PI
     .       + (29.14671D0 + RATCOUP*(1.570D0 - 2*DLOG(HIGTOP)/3
     .                                     + DLOG(X)**2/9))*(ASH/PI)**2
     .       + (164.14D0 - 25.77D0*5 + 0.259D0*5**2)*(ASH/PI)**3
      QCDH(X)=1.D0+HQCD(X)
      TQCDH(X)=1.D0+4.D0/3*HQCDM(BETA(X))*ASH/PI
C
      XI(X,Y) = 2*X/(1-X-Y+LAMB(X,Y))
      BIJ(X,Y) = (1-X-Y)/LAMB(X,Y)*(
     .          4*SP(XI(X,Y)*XI(Y,X))
     .        - 2*SP(-XI(X,Y)) - 2*SP(-XI(Y,X))
     .        + 2*DLOG(XI(X,Y)*XI(Y,X))*DLOG(1-XI(X,Y)*XI(Y,X))
     .        - DLOG(XI(X,Y))*DLOG(1+XI(X,Y))
     .        - DLOG(XI(Y,X))*DLOG(1+XI(Y,X))
     .          )
     .        -4*(DLOG(1-XI(X,Y)*XI(Y,X))
     .      +XI(X,Y)*XI(Y,X)/(1-XI(X,Y)*XI(Y,X))*DLOG(XI(X,Y)*XI(Y,X)))
     .        + (LAMB(X,Y)+X-Y)/LAMB(X,Y)*(DLOG(1+XI(X,Y))
     .                 - XI(X,Y)/(1+XI(X,Y))*DLOG(XI(X,Y)))
     .        + (LAMB(X,Y)-X+Y)/LAMB(X,Y)*(DLOG(1+XI(Y,X))
     .                 - XI(Y,X)/(1+XI(Y,X))*DLOG(XI(Y,X)))
      ELW(AMH,AMF,QF,ACF)=ALPH/PI*3.D0/2*QF**2
     .                              *(3.D0/2-DLOG(AMH**2/AMF**2))
     .      +GF/8/DSQRT(2.D0)/PI**2*(ACF*AMT**2
     .        +AMW**2*(3*DLOG(CS)/SS-5)+AMZ**2*(0.5D0
     .          -3*(1-4*SS*DABS(QF))**2))
      CF(CA) = -CDLOG(-(1+CDSQRT(1-CA))/(1-CDSQRT(1-CA)))**2/4
      CG(CA) = CDSQRT(1-CA)/2*CDLOG(-(1+CDSQRT(1-CA))/(1-CDSQRT(1-CA)))
      CI1(CA,CB) = CA*CB/2/(CA-CB)
     .           + CA**2*CB**2/2/(CA-CB)**2*(CF(CA)-CF(CB))
     .           + CA**2*CB/(CA-CB)**2*(CG(CA)-CG(CB))
      CI2(CA,CB) = -CA*CB/2/(CA-CB)*(CF(CA)-CF(CB))
      HGGQCD(ASG,NF)=1.D0+ASG/PI*(95.D0/4.D0-NF*7.D0/6.D0)
      HFFSELF(AMH)=1.D0+GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.117203D0
     .            -(GF*AMH**2/16.D0/PI**2/DSQRT(2.D0))**2*32.6567D0
      HVVSELF(AMH)=1.D0+GF*AMH**2/16.D0/PI**2/DSQRT(2.D0)*2.800952D0
     .            +(GF*AMH**2/16.D0/PI**2/DSQRT(2.D0))**2*62.0308D0

      PI=4D0*DATAN(1D0)
      SS=1.D0-(AMW/AMZ)**2

      CS=1.D0-SS

C--DECOUPLING THE TOP QUARK FROM ALPHAS
      AMT0=3.D8

C        =========================================================
C                              SM HIGGS DECAYS
C        =========================================================
      AMXX=AMH
      AMH=AMSM
C     =============  RUNNING MASSES 
      RMS = RUNM(AMH,3)
      RMC = RUNM(AMH,4)
      RMB = RUNM(AMH,5)
      RMT = RUNM(AMH,6)
      RATCOUP = 1
      HIGTOP = AMH**2/AMT**2

      ASH=ALPHAS(AMH,2)
      AMC0=1.D8
      AMB0=2.D8
      AS3=ALPHAS(AMH,2)
      AMC0=AMC
      AS4=ALPHAS(AMH,2)
      AMB0=AMB
C     AMT0=AMT
C     =============== PARTIAL WIDTHS 
C  H ---> G G
C
       EPS=1.D-8
       NFEXT = 3
       ASG = AS3
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CAT = 2*CTT*(1+(1-CTT)*CF(CTT))
       CAB = 2*CTB*(1+(1-CTB)*CF(CTB))
       FQCD=HGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB)**2*FQCD
       HGG=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8
C  H ---> G G* ---> G CC   TO BE ADDED TO H ---> CC
       NFEXT = 4
       ASG = AS4
       FQCD=HGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB)**2*FQCD
       DCC=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8 - HGG

C  H ---> G G* ---> G BB   TO BE ADDED TO H ---> BB
       NFEXT = 5
       ASG = ASH
       FQCD=HGGQCD(ASG,NFEXT)
       XFAC = CDABS(CAT+CAB)**2*FQCD
       DBB=HVV(AMH,0.D0)*(ASG/PI)**2*XFAC/8 - HGG - DCC

      IF(NFGG.EQ.5)THEN
       HGG = HGG + DBB + DCC
       DBB = 0
       DCC = 0
      ELSEIF(NFGG.EQ.4)THEN
       HGG = HGG + DCC
       DCC = 0
      ENDIF

C  H ---> MU MU
      IF(AMH.LE.2*AMMUON) THEN
       HMM = 0
      ELSE
      HMM=HFF(AMH,(AMMUON/AMH)**2)
     .    *(1+ELW(AMH,AMMUON,-1.D0,7.D0))
     .    *HFFSELF(AMH)
      ENDIF
C  H ---> TAU TAU
      IF(AMH.LE.2*AMTAU) THEN
       HLL = 0
      ELSE
      HLL=HFF(AMH,(AMTAU/AMH)**2)
     .    *(1+ELW(AMH,AMTAU,-1.D0,7.D0))
     .    *HFFSELF(AMH)
      ENDIF
C  H --> SS

      IF(AMH.LE.2*AMS) THEN
       HSS = 0
      ELSE
       HS2=3.D0*HFF(AMH,(RMS/AMH)**2)
     .    *QCDH(RMS**2/AMH**2)
     .    *(1+ELW(AMH,RMS,-1.D0/3.D0,7.D0))
     .    *HFFSELF(AMH)
       IF(HS2.LT.0.D0) HS2 = 0
       HS1=3.D0*HFF(AMH,(AMS/AMH)**2)
     .    *TQCDH(AMS**2/AMH**2)
     .    *HFFSELF(AMH)
       RAT = 2*AMS/AMH
       HSS = QQINT(RAT,HS1,HS2)
      ENDIF
C  H --> CC
      IF(AMH.LE.2*AMC) THEN
       HCC = 0
      ELSE
       HC2=3.D0*HFF(AMH,(RMC/AMH)**2)
     .    *QCDH(RMC**2/AMH**2)
     .    *(1+ELW(AMH,RMC,2.D0/3.D0,7.D0))
     .    *HFFSELF(AMH)
     .   + DCC
       IF(HC2.LT.0.D0) HC2 = 0
       HC1=3.D0*HFF(AMH,(AMC/AMH)**2)
     .    *TQCDH(AMC**2/AMH**2)
     .    *HFFSELF(AMH)
       RAT = 2*AMC/AMH
       HCC = QQINT(RAT,HC1,HC2)
      ENDIF
C  H --> BB :
      IF(AMH.LE.2*AMB) THEN
       HBB = 0
      ELSE
       HB2=3.D0*HFF(AMH,(RMB/AMH)**2)
     .    *QCDH(RMB**2/AMH**2)
     .    *(1+ELW(AMH,RMB,-1.D0/3.D0,1.D0))
     .    *HFFSELF(AMH)
     .   + DBB
       IF(HB2.LT.0.D0) HB2 = 0
       HB1=3.D0*HFF(AMH,(AMB/AMH)**2)
     .    *TQCDH(AMB**2/AMH**2)
     .    *HFFSELF(AMH)
       RAT = 2*AMB/AMH
       HBB = QQINT(RAT,HB1,HB2)
      ENDIF
C  H ---> TT
      RATCOUP = 0
      IF(IONSH.EQ.0)THEN
       DLD=3D0
       DLU=5D0
       XM1 = 2D0*AMT-DLD
       XM2 = 2D0*AMT+DLU
       IF (AMH.LE.AMT+AMW+AMB) THEN
       HTT=0.D0
       ELSEIF (AMH.LE.XM1) THEN
        FACTT=6.D0*GF**2*AMH**3*AMT**2/2.D0/128.D0/PI**3
        CALL HTOTTS(AMH,AMT,AMB,AMW,HTTS)
        HTT=FACTT*HTTS
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        FACTT=6.D0*GF**2*XX(1)**3*AMT**2/2.D0/128.D0/PI**3
        CALL HTOTTS(XX(1),AMT,AMB,AMW,HTTS)
        YY(1)=FACTT*HTTS
        FACTT=6.D0*GF**2*XX(2)**3*AMT**2/2.D0/128.D0/PI**3
        CALL HTOTTS(XX(2),AMT,AMB,AMW,HTTS)
        YY(2)=FACTT*HTTS
        XMT = RUNM(XX(3),6)
        XY2=3.D0*HFF(XX(3),(XMT/XX(3))**2)
     .    *QCDH(XMT**2/XX(3)**2)
     .    *HFFSELF(XX(3))
        IF(XY2.LT.0.D0) XY2 = 0
        XY1=3.D0*HFF(XX(3),(AMT/XX(3))**2)
     .    *TQCDH(AMT**2/XX(3)**2)
     .    *HFFSELF(XX(3))
        RAT = 2*AMT/XX(3)
        YY(3) = QQINT(RAT,XY1,XY2)
        XMT = RUNM(XX(4),6)
        XY2=3.D0*HFF(XX(4),(XMT/XX(4))**2)
     .    *QCDH(XMT**2/XX(4)**2)
     .    *HFFSELF(XX(4))
        IF(XY2.LT.0.D0) XY2 = 0
        XY1=3.D0*HFF(XX(4),(AMT/XX(4))**2)
     .    *TQCDH(AMT**2/XX(4)**2)
     .    *HFFSELF(XX(4))
        RAT = 2*AMT/XX(4)
        YY(4) = QQINT(RAT,XY1,XY2)
        HTT = FINT_(AMH,XX,YY)
       ELSE
        HT2=3.D0*HFF(AMH,(RMT/AMH)**2)
     .    *QCDH(RMT**2/AMH**2)
     .    *HFFSELF(AMH)
        IF(HT2.LT.0.D0) HT2 = 0
        HT1=3.D0*HFF(AMH,(AMT/AMH)**2)
     .    *TQCDH(AMT**2/AMH**2)
     .    *HFFSELF(AMH)
        RAT = 2*AMT/AMH
        HTT = QQINT(RAT,HT1,HT2)
       ENDIF
      ELSE
       IF (AMH.LE.2.D0*AMT) THEN
        HTT=0.D0
       ELSE
        HT2=3.D0*HFF(AMH,(RMT/AMH)**2)
     .    *QCDH(RMT**2/AMH**2)
     .    *HFFSELF(AMH)
        IF(HT2.LT.0.D0) HT2 = 0
        HT1=3.D0*HFF(AMH,(AMT/AMH)**2)
     .    *TQCDH(AMT**2/AMH**2)
     .    *HFFSELF(AMH)
        RAT = 2*AMT/AMH
        HTT = QQINT(RAT,HT1,HT2)
       ENDIF
      ENDIF
C  H ---> GAMMA GAMMA
       EPS=1.D-8
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTC = 4*AMC**2/AMH**2*DCMPLX(1D0,-EPS)
       CTL = 4*AMTAU**2/AMH**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AMH**2*DCMPLX(1D0,-EPS)
       CAW = -(2+3*CTW+3*CTW*(2-CTW)*CF(CTW))
       CAT = 4/3D0 * 2*CTT*(1+(1-CTT)*CF(CTT))
       CAB = 1/3D0 * 2*CTB*(1+(1-CTB)*CF(CTB))
       CAC = 4/3D0 * 2*CTC*(1+(1-CTC)*CF(CTC))
       CAL =         2*CTL*(1+(1-CTL)*CF(CTL))
       XFAC = CDABS(CAT+CAB+CAC+CAL+CAW)**2
       HGA=HVV(AMH,0.D0)*(ALPH/PI)**2/16.D0*XFAC

C  H ---> Z GAMMA
      IF(AMH.LE.AMZ)THEN
       HZGA=0
      ELSE
       EPS=1.D-8
       TS = SS/CS
       FT = -3*2D0/3*(1-4*2D0/3*SS)/DSQRT(SS*CS)
       FB = 3*1D0/3*(-1+4*1D0/3*SS)/DSQRT(SS*CS)
       CTT = 4*AMT**2/AMH**2*DCMPLX(1D0,-EPS)
       CTB = 4*AMB**2/AMH**2*DCMPLX(1D0,-EPS)
       CTW = 4*AMW**2/AMH**2*DCMPLX(1D0,-EPS)
       CLT = 4*AMT**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLB = 4*AMB**2/AMZ**2*DCMPLX(1D0,-EPS)
       CLW = 4*AMW**2/AMZ**2*DCMPLX(1D0,-EPS)
       CAT = FT*(CI1(CTT,CLT) - CI2(CTT,CLT))
       CAB = FB*(CI1(CTB,CLB) - CI2(CTB,CLB))
       CAW = -1/DSQRT(TS)*(4*(3-TS)*CI2(CTW,CLW)
     .     + ((1+2/CTW)*TS - (5+2/CTW))*CI1(CTW,CLW))
       XFAC = CDABS(CAT+CAB+CAW)**2
       ACOUP = DSQRT(2D0)*GF*AMZ**2*SS*CS/PI**2
       HZGA = GF/(4.D0*PI*DSQRT(2.D0))*AMH**3*(ALPH/PI)*ACOUP/16.D0
     .        *XFAC*(1-AMZ**2/AMH**2)**3
      ENDIF

C  H ---> W W
      IF(IONWZ.EQ.0)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMW-DLD
       XM2 = 2D0*AMW+DLU
       IF (AMH.LE.XM1) THEN
        CALL HTOVV(AMH,AMW,GAMW,HTWW)
        HWW = 3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/AMH**3*HTWW
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV(XX(1),AMW,GAMW,HTWW)
        YY(1)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(1)**3*HTWW
        CALL HTOVV(XX(2),AMW,GAMW,HTWW)
        YY(2)=3D0/2D0*GF*AMW**4/DSQRT(2D0)/PI/XX(2)**3*HTWW
        YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
     .       *HVVSELF(XX(3))
        YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
     .       *HVVSELF(XX(4))
        HWW = FINT_(AMH,XX,YY)
       ELSE
        HWW=HVV(AMH,AMW**2/AMH**2)
     .     *HVVSELF(AMH)
       ENDIF
      ELSE
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMW-DLD
       XM2 = 2D0*AMW+DLU
      IF (AMH.LE.AMW) THEN
       HWW=0
      ELSE IF (AMH.LE.XM1) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       HWW=HV(AMW**2/AMH**2)*CWW*AMH
      ELSE IF (AMH.LT.XM2) THEN
       CWW=3.D0*GF**2*AMW**4/16.D0/PI**3
       XX(1) = XM1-1D0
       XX(2) = XM1
       XX(3) = XM2
       XX(4) = XM2+1D0
       YY(1)=HV(AMW**2/XX(1)**2)*CWW*XX(1)
       YY(2)=HV(AMW**2/XX(2)**2)*CWW*XX(2)
       YY(3)=HVV(XX(3),AMW**2/XX(3)**2)
     .      *HVVSELF(XX(3))
       YY(4)=HVV(XX(4),AMW**2/XX(4)**2)
     .      *HVVSELF(XX(4))
       HWW = FINT_(AMH,XX,YY)
      ELSE
       HWW=HVV(AMH,AMW**2/AMH**2)
     .     *HVVSELF(AMH)
      ENDIF
      ENDIF

C  H ---> Z Z
      IF(IONWZ.EQ.0)THEN
       DLD=2D0
       DLU=2D0
       XM1 = 2D0*AMZ-DLD
       XM2 = 2D0*AMZ+DLU
       IF (AMH.LE.XM1) THEN
        CALL HTOVV(AMH,AMZ,GAMZ,HTZZ)
        HZZ = 3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/AMH**3*HTZZ
       ELSEIF (AMH.LE.XM2) THEN
        XX(1) = XM1-1D0
        XX(2) = XM1
        XX(3) = XM2
        XX(4) = XM2+1D0
        CALL HTOVV(XX(1),AMZ,GAMZ,HTZZ)
        YY(1)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(1)**3*HTZZ
        CALL HTOVV(XX(2),AMZ,GAMZ,HTZZ)
        YY(2)=3D0/4D0*GF*AMZ**4/DSQRT(2D0)/PI/XX(2)**3*HTZZ
        YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
     .       *HVVSELF(XX(3))
        YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
     .       *HVVSELF(XX(4))
        HZZ = FINT_(AMH,XX,YY)
       ELSE
        HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0
     .      *HVVSELF(AMH)
       ENDIF
      ELSE
      DLD=2D0
      DLU=2D0
      XM1 = 2D0*AMZ-DLD
      XM2 = 2D0*AMZ+DLU
      IF (AMH.LE.AMZ) THEN
      HZZ=0
      ELSE IF (AMH.LE.XM1) THEN
      CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
      HZZ=HV(AMZ**2/AMH**2)*CZZ*AMH
      ELSE IF (AMH.LT.XM2) THEN
      CZZ=3.D0*GF**2*AMZ**4/192.D0/PI**3*(7-40/3.D0*SS+160/9.D0*SS**2)
      XX(1) = XM1-1D0
      XX(2) = XM1
      XX(3) = XM2
      XX(4) = XM2+1D0
      YY(1)=HV(AMZ**2/XX(1)**2)*CZZ*XX(1)
      YY(2)=HV(AMZ**2/XX(2)**2)*CZZ*XX(2)
      YY(3)=HVV(XX(3),AMZ**2/XX(3)**2)/2
     .      *HVVSELF(XX(3))
      YY(4)=HVV(XX(4),AMZ**2/XX(4)**2)/2
     .      *HVVSELF(XX(4))
      HZZ = FINT_(AMH,XX,YY)
      ELSE
      HZZ=HVV(AMH,AMZ**2/AMH**2)/2.D0
     .   *HVVSELF(AMH)
      ENDIF
      ENDIF

C
C    ==========  TOTAL WIDTH AND BRANCHING RATIOS 
C
      WTOT=HLL+HMM+HSS+HCC+HBB+HTT+HGG+HGA+HZGA+HWW+HZZ

      SMBRT=HTT/WTOT
      SMBRB=HBB/WTOT
      SMBRL=HLL/WTOT
      SMBRM=HMM/WTOT
      SMBRC=HCC/WTOT
      SMBRS=HSS/WTOT
      SMBRG=HGG/WTOT
      SMBRGA=HGA/WTOT
      SMBRZGA=HZGA/WTOT
      SMBRW=HWW/WTOT
      SMBRZ=HZZ/WTOT
      SMWDTH=WTOT

      AMH=AMXX

      RETURN
      END
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     ************************************************************
C     SUBROUTINE FOR HSM ---> V*V* ---> 4F
C     ************************************************************
      SUBROUTINE HTOVV(AMH,AMV,GAMV,HTVV)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/VVOFF/AMH1,AMV1,GAMV1
      COMMON/PREC/IP
      EXTERNAL FTOVV1
      IP=20
      AMH1=AMH
      AMV1=AMV
      GAMV1=GAMV
      DLT=1D0/IP
      SUM=0D0
      DO 1 I=1,IP
       UU=DLT*I
       DD=UU-DLT
       CALL QGAUS1(FTOVV1,DD,UU,RES)
       SUM=SUM+RES
1     CONTINUE
      HTVV=SUM
      RETURN
      END

      DOUBLE PRECISION FUNCTION FTOVV1(XX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/FIRST/X1
      COMMON/PREC/IP
      EXTERNAL FTOVV2
      X1=XX
      DLT=1D0/IP
      SUM=0D0
      DO 1 I=1,IP
       UU=DLT*I
       DD=UU-DLT
       CALL QGAUS2(FTOVV2,DD,UU,RES)
       SUM=SUM+RES
1     CONTINUE
      FTOVV1=SUM
      RETURN
      END

      DOUBLE PRECISION FUNCTION FTOVV2(XX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION YY(2)
      COMMON/FIRST/X1
      YY(1)=X1
      YY(2)=XX
      FTOVV2=FTOVV(YY)
      RETURN
      END

      DOUBLE PRECISION FUNCTION FTOVV(XX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION LAMB
      DIMENSION XX(2)
      COMMON/VVOFF/AMH,AMW,GAMW
      LAMB(X,Y)=DSQRT((1.D0-X-Y)**2-4.D0*X*Y)
      PI=4D0*DATAN(1D0)
      ICASE = 1
      IF(ICASE.EQ.0)THEN
       YY = AMH**2
       Y1 = DATAN((YY-AMW**2)/AMW/GAMW)
       Y2 = -DATAN((AMW**2)/AMW/GAMW)
       DJAC = Y1-Y2
       T1 = TAN(Y1*XX(1)+Y2*(1.D0-XX(1)))
       SP = AMW**2 + AMW*GAMW*T1
       YY = (AMH-DSQRT(SP))**2
       Y1 = DATAN((YY-AMW**2)/AMW/GAMW)
       Y2 = -DATAN((AMW**2)/AMW/GAMW)
       DJAC = DJAC*(Y1-Y2)
       T2 = TAN(Y1*XX(2)+Y2*(1.D0-XX(2)))
       SM = AMW**2 + AMW*GAMW*T2
       AM2=AMH**2
       GAM = AM2*LAMB(SP/AM2,SM/AM2)*(1+LAMB(SP/AM2,SM/AM2)**2*AMH**4
     .                               /SP/SM/12)
       PRO1 = SP/AMW**2
       PRO2 = SM/AMW**2
       FTOVV = PRO1*PRO2*GAM*DJAC/PI**2
      ELSE
       SP = AMH**2*XX(1)
       SM = (AMH-DSQRT(SP))**2*XX(2)
       DJAC = AMH**2*(AMH-DSQRT(SP))**2/PI**2
       AM2=AMH**2
       GAM = AM2*LAMB(SP/AM2,SM/AM2)*(1+LAMB(SP/AM2,SM/AM2)**2*AMH**4
     .                               /SP/SM/12)
       PRO1 = SP*GAMW/AMW/((SP-AMW**2)**2+AMW**2*GAMW**2)
       PRO2 = SM*GAMW/AMW/((SM-AMW**2)**2+AMW**2*GAMW**2)
       FTOVV = PRO1*PRO2*GAM*DJAC
      ENDIF
      RETURN
      END

C     ************************************************************
C     SUBROUTINE FOR HSM ---> TT* ---> TBW
C     ************************************************************
      SUBROUTINE HTOTTS(AMH,AMT,AMB,AMW,HTTS)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,K
      COMMON/PREC1/IP
      EXTERNAL FUNSTT1
      COMMON/IKSY0/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP0/AMH0,AMT0,AMB0,AMW0
      AMH0=AMH
      AMT0=AMT
      AMB0=AMB
      AMW0=AMW
      IP=5
      M1=AMB
      M2=AMT
      M3=AMW
C     FIRST INTEGRATE OVER X2, i.e. (1+3) SYSTEM
C        CHECK WHETHER ENOUGH PHASE SPACE
      MASTOT=M1+M2+M3
      IF(MASTOT.GE.AMH) GOTO 12
      ECM=AMH
      S=ECM**2
      U1=(ECM-M2)**2
      D1=(M1+M3)**2
      U=(S-D1+M2**2)/s
      D=(S-U1+M2**2)/s
      DEL=(U-D)/IP
      U=D+DEL
      XSEC=0.D0
      DO K=1,IP
      CALL QGAUS1(FUNSTT1,D,U,SS)
      D=U
      U=D+DEL
      XSEC=XSEC+SS
      ENDDO
      HTTS=XSEC
12    CONTINUE
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNSTT1(XL)
      IMPLICIT REAL*8(A-Z)
      INTEGER IP,I
      COMMON/IKSY0/X1,X2,M1,M2,M3,ECM,S
      COMMON/PREC1/IP
      EXTERNAL FUNSTT2
      X2=XL
      S13=S-S*X2+M2**2
      TEM=2.D0*DSQRT(S13)
      E2S=(S-S13-M2**2)/TEM
      E3S=(S13+M3**2-M1**2)/TEM
C     SECOND INTEGRAL OVER X1, i.e. (2+3) SYSTEM
      U1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)-DSQRT(E3S**2-M3**2))**2
      D1=(E2S+E3S)**2-(DSQRT(E2S**2-M2**2)+DSQRT(E3S**2-M3**2))**2
      U=(S-D1+M1**2)/s
      D=(S-U1+M1**2)/s
      DEL=(U-D)/IP
      FUNSTT1=0.d0
      U=D+DEL
      DO I=1,IP
      CALL QGAUS2(FUNSTT2,D,U,SS)
      FUNSTT1=FUNSTT1+SS
      D=U
      U=D+DEL
      ENDDO
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNSTT2(XK)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY0/X1,X2,M1,M2,M3,ECM,S
      X1=XK
      CALL ELEMSTT(SS)
      FUNSTT2=SS
      RETURN
      END

      SUBROUTINE ELEMSTT(RES)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY0/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP0/AMH,AMT,AMB,AMW
      COMMON/WZWDTH/GAMC0,GAMT0,GAMT1,GAMW0,GAMZ0
      GAMT=GAMT0**2*AMT**2/AMH**4
      GAMW=GAMW0**2*AMW**2/AMH**4
      W=AMW**2/AMH**2
      T=AMT**2/AMH**2
      Y1=1-X2
      Y2=1-X1
      X0=2.D0-X1-X2
      W1=(1.D0-X2)
      W3=(1.-X1-X2)
      W11=1.D0/((1.D0-X2)**2+GAMT)
      W33=1.D0/(W3**2+GAMW**2)
      W13=W1*W3*W11*W33

      R11=4*T*W-16.*T*W*Y1-4.*T*Y2*Y1+8.*T*Y1+32.*T*W**2-20
     . .*T*Y1**2+8.*W*Y2*Y1+4.*W*Y1**2-4.*Y2*Y1**2-16.*T**2*W-
     .  32.*T**2*Y1+4.*T**2-16.*T**3-8.*W**2+4.*Y1**2-4.*Y1**3
      R33=-4.*T*W+4.*T*W*Y2-2.*T*W*Y2*Y1+4.*T*W*Y1+T*W*Y2**2-
     .  3.*T*W*Y1**2+2.*T*Y2*Y1-3.*T*Y2*Y1**2+4.*T*W**2-4.*T*W**3
     .  +T*Y2**2-3.*T*Y2**2*Y1-T*Y2**3+T*Y1**2-T*Y1**3+4.*T**2
     .  *W-4.*T**2*W*Y2-4.*T**2*W*Y1-2.*T**2*Y2*Y1-4.*T**2*W**2-
     .  T**2*Y2**2-T**2*Y1**2+4.*W**2*Y2*Y1-8.*W**3*Y2-8.*W**3*Y1
     .  +4.*W**3+8.*W**4
      R13=8.*W-24.*T*W+16.*T*W*Y1 -4.*T*Y2+16.*T*Y2*Y1-4.*T*
     .  Y1+16.*T*W**2+4.*T*Y2**2+12.*T*Y1**2-8.*W*Y2-12.*W*Y2*Y1
     .  -8.*W*Y1+4.*W*Y1**2-4.*Y2*Y1+8.*Y2*Y1**2+16.*T**2*W+8.
     .  *T**2*Y2+8.*T**2*Y1+16.*W**2*Y2+24.*W**2*Y1+4.*Y2**2*Y1-
     .  32.*W**3-4.*Y1**2+4.*Y1**3
      RES=R11*W11+4.D0*R33*W33/T-2.D0*R13*W13
      RETURN
      END

      DOUBLE PRECISION FUNCTION FUNHTT2(XK)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY2/X1,X2,M1,M2,M3,ECM,S
      X1=XK
      CALL ELEMHTT(SS)
      FUNHTT2=SS
      RETURN
      END

      SUBROUTINE ELEMHTT(RES)
      IMPLICIT REAL*8(A-Z)
      COMMON/IKSY2/X1,X2,M1,M2,M3,ECM,S
      COMMON/TOP2/AMH,AMT,AMB,AMW,AMCH,TB,GHT,GAT,GHVV
      COMMON/WZWDTH/GAMC0,GAMT0,GAMT1,GAMW0,GAMZ0
      GAMT=GAMT1**2*AMT**2/AMH**4
      GAMC=GAMC0**2*AMCH**2/AMH**4
      GAMW=GAMW0**2*AMW**2/AMH**4
      CH=AMCH**2/AMH**2
      W=AMW**2/AMH**2
      T=AMT**2/AMH**2
      Y1=1-X2
      Y2=1-X1
      X0=2.D0-X1-X2
      W1=(1.D0-X2)
      W2=(1.D0-X0+W-CH)
      W3=-(1.-X1-X2)
      W22=1.D0/ ((1.D0-X0+W-CH)**2+GAMC)
      W11=1.D0/((1.D0-X2)**2+GAMT)
      W33=1.D0/(W3**2+GAMW**2)
      W12=W1*W2*W11*W22
      W13=W1*W3*W11*W33
      W23=W2*W3*W22*W33

      R11=4*T*W-16.*T*W*Y1-4.*T*Y2*Y1+8.*T*Y1+32.*T*W**2-20
     . .*T*Y1**2+8.*W*Y2*Y1+4.*W*Y1**2-4.*Y2*Y1**2-16.*T**2*W-
     .  32.*T**2*Y1+4.*T**2-16.*T**3-8.*W**2+4.*Y1**2-4.*Y1**3
      R22=-16.*W+16.*T*W-8.*T*Y2*Y1-4.*T*Y2**2-4.*T*Y1**2+16
     .  .*W*Y2 + 8.*W*Y2*Y1 + 16.*W*Y1 + 4.*W*Y2**2 + 4.*W*Y1**2+8.*Y2*
     .  Y1-12.*Y2*Y1**2-12.*Y2**2*Y1-16.*W**2+4.*Y2**2-4.*Y2**3
     .  +4.*Y1**2-4.*Y1**3
      R33=-4.*T*W+4.*T*W*Y2-2.*T*W*Y2*Y1+4.*T*W*Y1+T*W*Y2**2-
     .  3.*T*W*Y1**2+2.*T*Y2*Y1-3.*T*Y2*Y1**2+4.*T*W**2-4.*T*W**3
     .  +T*Y2**2-3.*T*Y2**2*Y1-T*Y2**3+T*Y1**2-T*Y1**3+4.*T**2
     .  *W-4.*T**2*W*Y2-4.*T**2*W*Y1-2.*T**2*Y2*Y1-4.*T**2*W**2-
     .  T**2*Y2**2-T**2*Y1**2+4.*W**2*Y2*Y1-8.*W**3*Y2-8.*W**3*Y1
     .  +4.*W**3+8.*W**4
      R12=-16.*W+48.*T*W-16.*T*W*Y2+16.*T*W*Y1+8.*T*Y2-32.*T
     .  *Y2*Y1+8.*T*Y1-8.*T*Y2**2 - 24.*T*Y1**2+16.*W*Y2+8.*W*Y2*
     .  Y1+16.*W*Y1+8.*W*Y1**2+8.*Y2*Y1-16.*Y2*Y1**2-16.*T**2*Y2
     .  -16.*T**2*Y1-8.*Y2**2*Y1-16.*W**2+8.*Y1**2-8.*Y1**3
      R13=8.*W-24.*T*W+16.*T*W*Y1 -4.*T*Y2+16.*T*Y2*Y1-4.*T*
     .  Y1+16.*T*W**2+4.*T*Y2**2+12.*T*Y1**2-8.*W*Y2-12.*W*Y2*Y1
     .  -8.*W*Y1+4.*W*Y1**2-4.*Y2*Y1+8.*Y2*Y1**2+16.*T**2*W+8.
     .  *T**2*Y2+8.*T**2*Y1+16.*W**2*Y2+24.*W**2*Y1+4.*Y2**2*Y1-
     .  32.*W**3-4.*Y1**2+4.*Y1**3
      R23=16.*W-16.*T*W+8.*T*W*Y2+8.*T*W*Y1+8.*T*Y2*Y1+4.*T*
     .  Y2**2+4.*T*Y1**2-16.*W*Y2-16.*W*Y1-4.*W*Y2**2+4.*W*Y1**2
     .  -8.*Y2*Y1+12.*Y2*Y1**2+8.*W**2*Y2-8.*W**2*Y1+12.*Y2**2*
     .  Y1-4.*Y2**2+4.*Y2**3-4.*Y1**2+4.*Y1**3
      GLVV=DSQRT(1.D0-GHVV**2)
      RES=GHT**2*R11*W11+GLVV**2*GAT**2*R22*W22+
     .    4.D0*GHVV**2*R33*W33/T+2.D0*GHT*GLVV*GAT*R12*W12+
     .    2.D0*GHT*GHVV*R13*W13+2.D0*GHVV*GLVV*GAT*R23*W23
      RETURN
      END


C   *****************  INTEGRATION ROUTINE ***********************
C    Returns SS as integral of FUNC from A to B, by 10-point Gauss-
C    Legendre integration
      SUBROUTINE QGAUS1(FUNC,A,B,SS)
      IMPLICIT REAL*8(A-Z)
      INTEGER J
      DIMENSION X(5),W(5)
      EXTERNAL FUNC
      DATA X/.1488743389D0,.4333953941D0,.6794095682D0
     .  ,.8650633666D0,.9739065285D0/
      DATA W/.2955242247D0,.2692667193D0,.2190863625D0
     .  ,.1494513491D0,.0666713443D0/
      XM=0.5D0*(B+A)
      XR=0.5D0*(B-A)
      SS=0.D0
      DO 11 J=1,5
        DX=XR*X(J)
        SS=SS+W(J)*(FUNC(XM+DX)+FUNC(XM-DX))
11    CONTINUE
      SS=XR*SS
      RETURN
      END

C     Returns SS as integral of FUNC from A to B, by 10-point Gauss-
C      Legendre integration
      SUBROUTINE QGAUS2(FUNC,A,B,SS)
      IMPLICIT REAL*8(A-Z)
      INTEGER J
      DIMENSION X(5),W(5)
      EXTERNAL FUNC
      DATA X/.1488743389D0,.4333953941D0,.6794095682D0
     .  ,.8650633666D0,.9739065285D0/
      DATA W/.2955242247D0,.2692667193D0,.2190863625D0
     .  ,.1494513491D0,.0666713443D0/
      XM=0.5D0*(B+A)
      XR=0.5D0*(B-A)
      SS=0.D0
      DO 11 J=1,5
        DX=XR*X(J)
        SS=SS+W(J)*(FUNC(XM+DX)+FUNC(XM-DX))
11    CONTINUE
      SS=XR*SS
      RETURN
      END

C ******************************************************************

      DOUBLE PRECISION FUNCTION RUNM(Q,NF)
      PARAMETER (NN=6)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (ZETA3 = 1.202056903159594D0)
      DIMENSION AM(NN),YMSB(NN)
      COMMON/ALS/XLAMBDA,AMCA,AMBA,AMTA,N0A
      COMMON/MASSES/AMS,AMC,AMB,AMT
      COMMON/STRANGE/AMSB
      COMMON/RUN/XMSB,XMHAT,XKFAC
      COMMON/FLAG/IHIGGS,NNLO,IPOLE
      SAVE ISTRANGE
      B0(NF)=(33.D0-2.D0*NF)/12D0
      B1(NF) = (102D0-38D0/3D0*NF)/16D0
      B2(NF) = (2857D0/2D0-5033D0/18D0*NF+325D0/54D0*NF**2)/64D0
      G0(NF) = 1D0
      G1(NF) = (202D0/3D0-20D0/9D0*NF)/16D0
      G2(NF) = (1249D0-(2216D0/27D0+160D0/3D0*ZETA3)*NF
     .       - 140D0/81D0*NF**2)/64D0
      C1(NF) = G1(NF)/B0(NF) - B1(NF)*G0(NF)/B0(NF)**2
      C2(NF) = ((G1(NF)/B0(NF) - B1(NF)*G0(NF)/B0(NF)**2)**2
     .       + G2(NF)/B0(NF) + B1(NF)**2*G0(NF)/B0(NF)**3
     .       - B1(NF)*G1(NF)/B0(NF)**2 - B2(NF)*G0(NF)/B0(NF)**2)/2D0
      TRAN(X,XK)=1D0+4D0/3D0*ALPHAS(X,2)/PI+XK*(ALPHAS(X,2)/PI)**2
      CQ(X,NF)=(2D0*B0(NF)*X)**(G0(NF)/B0(NF))
     .            *(1D0+C1(NF)*X+C2(NF)*X**2)
      DATA ISTRANGE/0/
      PI=4D0*DATAN(1D0)
      ACC = 1.D-8
      AM(1) = 0
      AM(2) = 0
C--------------------------------------------
      IMSBAR = 0
      IF(IMSBAR.EQ.1)THEN
       IF(ISTRANGE.EQ.0)THEN
C--STRANGE POLE MASS FROM MSBAR-MASS AT 1 GEV
        AMSD = XLAMBDA
        AMSU = 1.D8
123     AMS  = (AMSU+AMSD)/2
        AM(3) = AMS
        XMSB = AMS/CQ(ALPHAS(AMS,2)/PI,3)
     .            *CQ(ALPHAS(1.D0,2)/PI,3)/TRAN(AMS,0D0)
        DD = (XMSB-AMSB)/AMSB
        IF(DABS(DD).GE.ACC)THEN
         IF(DD.LE.0.D0)THEN
          AMSD = AM(3)
         ELSE
          AMSU = AM(3)
         ENDIF
         GOTO 123
        ENDIF
        ISTRANGE=1
       ENDIF
       AM(3) = AMSB
      ELSE
       AMS=AMSB
       AM(3) = AMS
      ENDIF
C--------------------------------------------
      AM(3) = AMSB
      AM(4) = AMC
      AM(5) = AMB
      AM(6) = AMT
      XK = 16.11D0
      DO 1 I=1,NF-1
       XK = XK - 1.04D0*(1.D0-AM(I)/AM(NF))
1     CONTINUE
      IF(NF.GE.4)THEN
       XMSB = AM(NF)/TRAN(AM(NF),0D0)
       XMHAT = XMSB/CQ(ALPHAS(AM(NF),2)/PI,NF)
      ELSE
       XMSB = 0
       XMHAT = 0
      ENDIF
      YMSB(3) = AMSB

      IF(NF.EQ.3)THEN
       YMSB(4) = YMSB(3)*CQ(ALPHAS(AM(4),2)/PI,3)/
     .                   CQ(ALPHAS(1.D0,2)/PI,3)
       YMSB(5) = YMSB(4)*CQ(ALPHAS(AM(5),2)/PI,4)/
     .                   CQ(ALPHAS(AM(4),2)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS(AM(6),2)/PI,5)/
     .                   CQ(ALPHAS(AM(5),2)/PI,5)
      ELSEIF(NF.EQ.4)THEN
       YMSB(4) = XMSB
       YMSB(5) = YMSB(4)*CQ(ALPHAS(AM(5),2)/PI,4)/
     .                   CQ(ALPHAS(AM(4),2)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS(AM(6),2)/PI,5)/
     .                   CQ(ALPHAS(AM(5),2)/PI,5)
      ELSEIF(NF.EQ.5)THEN
       YMSB(5) = XMSB
       YMSB(4) = YMSB(5)*CQ(ALPHAS(AM(4),2)/PI,4)/
     .                   CQ(ALPHAS(AM(5),2)/PI,4)
       YMSB(6) = YMSB(5)*CQ(ALPHAS(AM(6),2)/PI,5)/
     .                   CQ(ALPHAS(AM(5),2)/PI,5)
      ELSEIF(NF.EQ.6)THEN
       YMSB(6) = XMSB
       YMSB(5) = YMSB(6)*CQ(ALPHAS(AM(5),2)/PI,5)/
     .                   CQ(ALPHAS(AM(6),2)/PI,5)
       YMSB(4) = YMSB(5)*CQ(ALPHAS(AM(4),2)/PI,4)/
     .                   CQ(ALPHAS(AM(5),2)/PI,4)
      ENDIF
      IF(Q.LT.AMC)THEN
       N0=3
       Q0 = 1.D0
      ELSEIF(Q.LE.AMB)THEN
       N0=4
       Q0 = AMC
      ELSEIF(Q.LE.AMT)THEN
       N0=5
       Q0 = AMB
      ELSE
       N0=6
       Q0 = AMT
      ENDIF
      IF(NNLO.EQ.1.AND.NF.GT.3)THEN
       XKFAC = TRAN(AM(NF),0D0)/TRAN(AM(NF),XK)
      ELSE
       XKFAC = 1D0
      ENDIF

      RUNM = YMSB(N0)*CQ(ALPHAS(Q,2)/PI,N0)/
     .               CQ(ALPHAS(Q0,2)/PI,N0)
     .       * XKFAC
      RETURN
      END

      DOUBLE PRECISION FUNCTION ALPHAS(Q,N)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XLB(6)
      COMMON/ALSLAM/XLB1(6),XLB2(6)
      COMMON/ALS/XLAMBDA,AMC,AMB,AMT,N0
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      ALS1(NF,X)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB(NF)**2))
      ALS2(NF,X)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB(NF)**2))
     .          *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB(NF)**2))
     .           /DLOG(X**2/XLB(NF)**2))
      PI=4.D0*DATAN(1.D0)
      IF(N.EQ.1)THEN
       DO 1 I=1,6
        XLB(I)=XLB1(I)
1      CONTINUE
      ELSE
       DO 2 I=1,6
        XLB(I)=XLB2(I)
2      CONTINUE
      ENDIF
      IF(Q.LT.AMC)THEN
       NF=3
      ELSEIF(Q.LE.AMB)THEN
       NF=4
      ELSEIF(Q.LE.AMT)THEN
       NF=5
      ELSE
       NF=6
      ENDIF
      IF(N.EQ.1)THEN
        ALPHAS=ALS1(NF,Q)
      ELSE
        ALPHAS=ALS2(NF,Q)

      ENDIF
      RETURN
      END

      SUBROUTINE ALSINI(ACC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XLB(6)
      COMMON/ALSLAM/XLB1(6),XLB2(6)
      COMMON/ALS/XLAMBDA,AMC,AMB,AMT,N0
      PI=4.D0*DATAN(1.D0)
      XLB1(1)=0D0
      XLB1(2)=0D0
      XLB2(1)=0D0
      XLB2(2)=0D0
C...  Initialize with zero
      DO I=1,6
         XLB(I) = 0.0
      ENDDO
      IF(N0.EQ.3)THEN
       XLB(3)=XLAMBDA
       XLB(4)=XLB(3)*(XLB(3)/AMC)**(2.D0/25.D0)
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
      ELSEIF(N0.EQ.4)THEN
       XLB(4)=XLAMBDA
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
      ELSEIF(N0.EQ.5)THEN
       XLB(5)=XLAMBDA
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
      ELSEIF(N0.EQ.6)THEN
       XLB(6)=XLAMBDA
       XLB(5)=XLB(6)*(XLB(6)/AMT)**(-2.D0/23.D0)
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
      ENDIF
      DO 1 I=1,6
       XLB1(I)=XLB(I)
1     CONTINUE
      IF(N0.EQ.3)THEN
       XLB(3)=XLAMBDA
       XLB(4)=XLB(3)*(XLB(3)/AMC)**(2.D0/25.D0)
     .             *(2.D0*DLOG(AMC/XLB(3)))**(-107.D0/1875.D0)
       XLB(4)=XITER(AMC,XLB(3),3,XLB(4),4,ACC)
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
     .             *(2.D0*DLOG(AMB/XLB(4)))**(-963.D0/13225.D0)
       XLB(5)=XITER(AMB,XLB(4),4,XLB(5),5,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.4)THEN
       XLB(4)=XLAMBDA
       XLB(5)=XLB(4)*(XLB(4)/AMB)**(2.D0/23.D0)
     .             *(2.D0*DLOG(AMB/XLB(4)))**(-963.D0/13225.D0)
       XLB(5)=XITER(AMB,XLB(4),4,XLB(5),5,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER(AMC,XLB(4),4,XLB(3),3,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.5)THEN
       XLB(5)=XLAMBDA
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
     .             *(2.D0*DLOG(AMB/XLB(5)))**(963.D0/14375.D0)
       XLB(4)=XITER(AMB,XLB(5),5,XLB(4),4,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER(AMC,XLB(4),4,XLB(3),3,ACC)
       XLB(6)=XLB(5)*(XLB(5)/AMT)**(2.D0/21.D0)
     .            *(2.D0*DLOG(AMT/XLB(5)))**(-321.D0/3381.D0)
       XLB(6)=XITER(AMT,XLB(5),5,XLB(6),6,ACC)
      ELSEIF(N0.EQ.6)THEN
       XLB(6)=XLAMBDA
       XLB(5)=XLB(6)*(XLB(6)/AMT)**(-2.D0/23.D0)
     .            *(2.D0*DLOG(AMT/XLB(6)))**(321.D0/3703.D0)
       XLB(5)=XITER(AMT,XLB(6),6,XLB(5),5,ACC)
       XLB(4)=XLB(5)*(XLB(5)/AMB)**(-2.D0/25.D0)
     .             *(2.D0*DLOG(AMB/XLB(5)))**(963.D0/14375.D0)
       XLB(4)=XITER(AMB,XLB(5),5,XLB(4),4,ACC)
       XLB(3)=XLB(4)*(XLB(4)/AMC)**(-2.D0/27.D0)
     .             *(2.D0*DLOG(AMC/XLB(4)))**(107.D0/2025.D0)
       XLB(3)=XITER(AMC,XLB(4),4,XLB(3),3,ACC)
      ENDIF
      DO 2 I=1,6
       XLB2(I)=XLB(I)
2     CONTINUE
      RETURN
      END

      DOUBLE PRECISION FUNCTION XITER(Q,XLB1,NF1,XLB,NF2,ACC)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      ALS2(NF,X,XLB)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB**2))
     .              *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB**2))
     .              /DLOG(X**2/XLB**2))
      AA(NF)=12D0*PI/B0(NF)
      BB(NF)=B1(NF)/AA(NF)
      XIT(A,B,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)))
      PI=4.D0*DATAN(1.D0)
      XLB2=XLB
      II=0
1     II=II+1
      X=DLOG(Q**2/XLB2**2)
      ALP=ALS2(NF1,Q,XLB1)
      A=AA(NF2)/ALP
      B=BB(NF2)*ALP
      XX=XIT(A,B,X)
      XLB2=Q*DEXP(-XX/2.D0)
      Y1=ALS2(NF1,Q,XLB1)
      Y2=ALS2(NF2,Q,XLB2)
      DY=DABS(Y2-Y1)/Y1
      IF(DY.GE.ACC) GOTO 1
      XITER=XLB2
      RETURN
      END

      DOUBLE PRECISION FUNCTION FINT_(Z,XX,YY)
C--ONE-DIMENSIONAL CUBIC INTERPOLATION
C--Z  = WANTED POINT
C--XX = ARRAY OF 4 DISCRETE X-VALUES AROUND Z
C--YY = ARRAY OF 4 DISCRETE FUNCTION-VALUES AROUND Z
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XX(4),YY(4)
      X = DLOG(Z)
      X0=DLOG(XX(1))
      X1=DLOG(XX(2))
      X2=DLOG(XX(3))
      X3=DLOG(XX(4))
      Y0=DLOG(YY(1))
      Y1=DLOG(YY(2))
      Y2=DLOG(YY(3))
      Y3=DLOG(YY(4))
      A0=(X-X1)*(X-X2)*(X-X3)/(X0-X1)/(X0-X2)/(X0-X3)
      A1=(X-X0)*(X-X2)*(X-X3)/(X1-X0)/(X1-X2)/(X1-X3)
      A2=(X-X0)*(X-X1)*(X-X3)/(X2-X0)/(X2-X1)/(X2-X3)
      A3=(X-X0)*(X-X1)*(X-X2)/(X3-X0)/(X3-X1)/(X3-X2)
      FINT_=DEXP(A0*Y0+A1*Y1+A2*Y2+A3*Y3)
      RETURN
      END

      DOUBLE PRECISION FUNCTION SP(X)
C--REAL DILOGARITHM (SPENCE-FUNCTION)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 CX,LI2
      CX = DCMPLX(X,0.D0)
      SP = DREAL(LI2(CX))
      RETURN
      END
 
      COMPLEX*16 FUNCTION LI2(X)
C--COMPLEX DILOGARITHM (SPENCE-FUNCTION)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 X,Y,CLI2
      COMMON/CONST/ZETA2,ZETA3
      ZERO=1.D-16
      XR=DREAL(X)
      XI=DIMAG(X)
      R2=XR*XR+XI*XI
      LI2=0
      IF(R2.LE.ZERO)THEN
        LI2=X
        RETURN
      ENDIF
      RR=XR/R2
      IF(R2.EQ.1.D0.AND.XI.EQ.0.D0)THEN
        IF(XR.EQ.1.D0)THEN
          LI2=DCMPLX(ZETA2)
        ELSE
          LI2=-DCMPLX(ZETA2/2.D0)
        ENDIF
        RETURN
      ELSEIF(R2.GT.1.D0.AND.RR.GT.0.5D0)THEN
        Y=(X-1.D0)/X
        LI2=CLI2(Y)+ZETA2-CDLOG(X)*CDLOG(1.D0-X)+0.5D0*CDLOG(X)**2
        RETURN
      ELSEIF(R2.GT.1.D0.AND.RR.LE.0.5D0)THEN
        Y=1.D0/X
        LI2=-CLI2(Y)-ZETA2-0.5D0*CDLOG(-X)**2
        RETURN
      ELSEIF(R2.LE.1.D0.AND.XR.GT.0.5D0)THEN
        Y=1.D0-X
        LI2=-CLI2(Y)+ZETA2-CDLOG(X)*CDLOG(1.D0-X)
       RETURN
      ELSEIF(R2.LE.1.D0.AND.XR.LE.0.5D0)THEN
        Y=X
        LI2=CLI2(Y)
        RETURN
      ENDIF
      END
 
      COMPLEX*16 FUNCTION CLI2(X)
C--TAYLOR-EXPANSION FOR COMPLEX DILOGARITHM (SPENCE-FUNCTION)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 X,Z
      COMMON/BERNOULLI/B2(18),B12(18),B3(18)
      COMMON/POLY/NBER
      N=NBER-1
      Z=-CDLOG(1.D0-X)
      CLI2=B2(NBER)
      DO 111 I=N,1,-1
        CLI2=Z*CLI2+B2(I)
111   CONTINUE
      CLI2=Z**2*CLI2+Z
      RETURN
      END
 
      DOUBLE PRECISION FUNCTION FACULT(N)
C--DOUBLE PRECISION VERSION OF FACULTY
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      FACULT=1.D0
      IF(N.EQ.0)RETURN
      DO 999 I=1,N
        FACULT=FACULT*DFLOAT(I)
999   CONTINUE
      RETURN
      END
 
      SUBROUTINE BERNINI(N)
C--INITIALIZATION OF COEFFICIENTS FOR POLYLOGARITHMS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION B(18),PB(19)
      COMMON/BERNOULLI/B2(18),B12(18),B3(18)
      COMMON/CONST/ZETA2,ZETA3
      COMMON/POLY/NBER
 
      NBER=N
      PI=4.D0*DATAN(1.D0)
 
      B(1)=-1.D0/2.D0
      B(2)=1.D0/6.D0
      B(3)=0.D0
      B(4)=-1.D0/30.D0
      B(5)=0.D0
      B(6)=1.D0/42.D0
      B(7)=0.D0
      B(8)=-1.D0/30.D0
      B(9)=0.D0
      B(10)=5.D0/66.D0
      B(11)=0.D0
      B(12)=-691.D0/2730.D0
      B(13)=0.D0
      B(14)=7.D0/6.D0
      B(15)=0.D0
      B(16)=-3617.D0/510.D0
      B(17)=0.D0
      B(18)=43867.D0/798.D0
      ZETA2=PI**2/6.D0
      ZETA3=1.202056903159594D0
 
      DO 995 I=1,18
        B2(I)=B(I)/FACULT(I+1)
        B12(I)=DFLOAT(I+1)/FACULT(I+2)*B(I)/2.D0
        PB(I+1)=B(I)
        B3(I)=0.D0
995   CONTINUE
      PB(1)=1.D0
      DO 996 I=1,18
      DO 996 J=0,I
        B3(I)=B3(I)+PB(J+1)*PB(I-J+1)/FACULT(I-J)/FACULT(J+1)
     .                                            /DFLOAT(I+1)
996   CONTINUE
 
      RETURN
      END

      DOUBLE PRECISION FUNCTION QQINT(RAT,H1,H2)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      N = 2
      QQINT = RAT**N * H1 + (1-RAT**N) * H2
      RETURN
      END

      DOUBLE PRECISION FUNCTION XITLA(NO,ALP,ACC)
C--ITERATION ROUTINE TO DETERMINE IMPROVED LAMBDA'S
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON/PARAM/GF,ALPH,AMTAU,AMMUON,AMZ,AMW
      B0(NF)=33.D0-2.D0*NF
      B1(NF)=6.D0*(153.D0-19.D0*NF)/B0(NF)**2
      ALS2(NF,X,XLB)=12.D0*PI/(B0(NF)*DLOG(X**2/XLB**2))
     .              *(1.D0-B1(NF)*DLOG(DLOG(X**2/XLB**2))
     .              /DLOG(X**2/XLB**2))
      AA(NF)=12D0*PI/B0(NF)
      BB(NF)=B1(NF)/AA(NF)
      XIT(A,B,X)=A/2.D0*(1D0+DSQRT(1D0-4D0*B*DLOG(X)))
      PI=4.D0*DATAN(1.D0)
      NF=5
      Q=AMZ
      XLB=Q*DEXP(-AA(NF)/ALP/2.D0)
      IF(NO.EQ.1)GOTO 111
      II=0
1     II=II+1
      X=DLOG(Q**2/XLB**2)
      A=AA(NF)/ALP
      B=BB(NF)*ALP
      XX=XIT(A,B,X)
      XLB=Q*DEXP(-XX/2.D0)
      Y1=ALP
      Y2=ALS2(NF,Q,XLB)
      DY=DABS(Y2-Y1)/Y1
      IF(DY.GE.ACC) GOTO 1
111   XITLA=XLB
      RETURN
      END
