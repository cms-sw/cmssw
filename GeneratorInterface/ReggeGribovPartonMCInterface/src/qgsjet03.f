
C======================================================================C
C                                                                      C
C      QQQ         GGG       SSSS     JJJJJJJ    EEEEEEE    TTTTTTT    C
C     Q   Q       G   G     S    S          J    E             T       C
C    Q     Q     G          S               J    E             T       C
C    Q     Q     G   GGG     SSSS           J    EEEEE         T       C
C    Q   Q Q     G     G         S          J    E             T       C
C     Q   Q       G   G     S    S     J   J     E             T       C
C      QQQ QQ      GGG       SSSS       JJJ      EEEEEEE       T       C
C                                                                      C
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
C                    QUARK - GLUON - STRING - MODEL                    C
C                                                                      C
C                HIGH ENERGY HADRON INTERACTION PROGRAM                C
C                                                                      C
C                                  BY                                  C
C                                                                      C
C                 N. N. KALMYKOV AND S. S. OSTAPCHENKO                 C
C                                                                      C
C               MOSCOW STATE UNIVERSITY,  MOSCOW, RUSSIA               C
C                      e-mail: serg@eas.npi.msu.su                     C
C----------------------------------------------------------------------C
C                 SUBROUTINE VERSION TO BE LINKED WITH                 C
C                       C O N E X  or N E X U S                        C
C                                  BY                                  C
C                               T. PIEROG                              C
C                                 FROM                                 C
C                 SUBROUTINE VERSION TO BE LINKED WITH                 C
C                             C O R S I K A                            C
C               KARLSRUHE  AIR SHOWER SIMULATION PROGRAM               C
C                          WITH MODIFICATIONS                          C
C                                  BY                                  C
C                      D. HECK  IK3 FZK KARLSRUHE                      C
C----------------------------------------------------------------------C
C                 last modification:  Sep. 27, 2003                    C
C               Version qgsjet03cx.f from qgsjet03.f                   C
C----------------------------------------------------------------------C
C  modifications for Corsika are marked by cdh and for Conex by ctp    C
c  (common subroutine name with nexus : add QGS in front of the name   C
C=======================================================================

ctp      SUBROUTINE PSAINI
      SUBROUTINE QGSPSAINI
c Common initialization procedure
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      CHARACTER *7 TY
      LOGICAL LCALC!,LSECT
********************************************
      DIMENSION EQ(17),MIJ(17,17,4),NIJ(17,17,4),CSJET(17,17,68),
     *CS1(17,17,68),GZ0(2),GZ1(3)
      COMMON /Q_XSECT/  GSECT(10,5,4)
      COMMON /Q_AREA1/  IA(2),ICZ,ICP
      COMMON /Q_AREA5/  RD(2),CR1(2),CR2(2),CR3(2)
********************************************
      COMMON /Q_AREA6/  PI,BM,AM
      COMMON /Q_AREA7/  RP1
      COMMON /Q_AREA10/ STMASS,AM0,AMN,AMK,AMC,AMLAMC,AMLAM,AMETA
      COMMON /Q_AREA15/ FP(5),RQ(5),CD(5)
      COMMON /Q_AREA16/ CC(5)
      COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
      COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
      COMMON /Q_AREA19/ AHL(5)
********************************************
      COMMON /Q_AREA22/ SJV0,FJS0(5,3)
********************************************
      COMMON /Q_AREA23/ RJV(50)
      COMMON /Q_AREA24/ RJS(50,5,10)
      COMMON /Q_AREA27/ FP0(5)
      COMMON /Q_AREA28/ ARR(4)
      COMMON /Q_AREA29/ CSTOT(17,17,68)
      COMMON /Q_AREA30/ CS0(17,17,68)
      COMMON /Q_AREA31/ CSBORN(17,68)
      COMMON /Q_AREA32/ CSQ(17,2,2),CSBQ(17,2,2)
      COMMON /Q_AREA33/ FSUD(10,2)
      COMMON /Q_AREA34/ QRT(10,101,2)
      COMMON /Q_AREA35/ SJV(10,5),FJS(10,5,15)
      COMMON /Q_AREA39/ JCALC
      COMMON /Q_AREA40/ JDIFR
      COMMON /Q_AREA41/ TY(5)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
********************************************
      COMMON /Q_AREA44/ GZ(10,5,4),GZP(10,5,4)
c Auxiliary common blocks to calculate hadron-nucleus cross-sections
      COMMON /Q_AR1/    ANORM
      COMMON /Q_AR2/    RRR,RRRM
********************************************
cdh 8/10/98
      COMMON /Q_AREA48/ QGSASECT(10,6,4)
cdh 8/12/98
      COMMON /Q_VERSION/VERSION
      SAVE
********************************************
ctp 11/03/03
      character*500 fndat,fnncs
      common/qgsfname/  fndat, fnncs, ifdat, ifncs
      common/qgsnfname/ nfndat, nfnncs
c-------------------------------------------------
        WRITE(MONIOU,100)
 100    FORMAT(' ',
     *           '====================================================',
     *     /,' ','|                                                  |',
     *     /,' ','|           QUARK GLUON STRING JET MODEL           |',
     *     /,' ','|                                                  |',
     *     /,' ','|         HADRONIC INTERACTION MONTE CARLO         |',
     *     /,' ','|                        BY                        |',
     *     /,' ','|        N.N. KALMYKOV AND S.S. OSTAPCHENKO        |',
     *     /,' ','|                                                  |',
     *     /,' ','|            e-mail: serg@eas.npi.msu.su           |',
     *     /,' ','|                                                  |',
     *     /,' ','| Publication to be cited when using this program: |',
     *     /,' ','| N.N. Kalmykov & S.S. Ostapchenko, A.I. Pavlov    |',
     *     /,' ','| Nucl. Phys. B (Proc. Suppl.) 52B (1997) 17       |',
     *     /,' ','|                                                  |',
     *     /,' ','====================================================',
     *     /)
        IF(DEBUG.GE.1)WRITE (MONIOU,210)
210     FORMAT(2X,'QGSPSAINI - MAIN INITIALIZATION PROCEDURE')
cdh
        VERSION = 2003.

c AHL(i) - parameter for the energy sharing procedure (govern leading hadronic state
c inelasticity for primary pion, nucleon, kaon, D-meson, Lambda_C correspondingly)
      AHL(1)=1.D0-2.D0*ARR(1)
      AHL(2)=1.D0-ARR(1)-ARR(2)
      AHL(3)=1.D0-ARR(1)-ARR(3)
      AHL(4)=1.D0-ARR(1)-ARR(4)
      AHL(5)=AHL(2)+ARR(1)-ARR(4)

c-------------------------------------------------
c 1/CC(i) = C_i - shower enhancement coefficients for one vertex
c (C_ab=C_a*C_b) (i - ICZ)
      CC(2)=1.D0/DSQRT(CD(2))
      CC(1)=1.D0/CC(2)/CD(1)
      CC(3)=1.D0/CC(2)/CD(3)
      CC(4)=1.D0/CC(2)/CD(4)
      CC(5)=1.D0/CC(2)/CD(5)

c FP0(i) - vertex constant (FP_ij=FP0_i*FP0_j) for pomeron-hadron interaction (i - ICZ)
      FP0(2)=DSQRT(FP(2))
      FP0(1)=FP(1)/FP0(2)
      FP0(3)=FP(3)/FP0(2)
      FP0(4)=FP(4)/FP0(2)
      FP0(5)=FP(5)/FP0(2)

c SH - hard interaction effective squared (SH=pi*R_h^2, R_h^2=4/Q0^2)
      SH=4.D0/QT0*PI
c Auxiliary constants for the hard interaction
      AQT0=DLOG(4.D0*QT0)
      QLOG=DLOG(QT0/ALM)
      QLL=DLOG(QLOG)

********************************************
      IF(IFDAT.NE.1)THEN
        INQUIRE(FILE='QGSDAT01',EXIST=LCALC)
      ELSE                                                  !ctp
        INQUIRE(FILE=FNDAT(1:NFNDAT),EXIST=LCALC)
      ENDIF
      IF(LCALC)then
        IF(DEBUG.GE.1)WRITE (MONIOU,211)
211     FORMAT(2X,'PSAINI: HARD CROSS SECTION RATIOS READOUT FROM THE'
     *    ,' FILE QGSDAT01')
        IF(IFDAT.NE.1)THEN
          OPEN(1,FILE='QGSDAT01',STATUS='OLD')
        ELSE                                                  !ctp
          OPEN(IFDAT,FILE=FNDAT(1:NFNDAT),STATUS='OLD')
        ENDIF
        READ (1,*)CSBORN,CS0,CSTOT,CSQ,CSBQ,
     *  FSUD,QRT,SJV,FJS,RJV,RJS,GZ,GZP,GSECT
        CLOSE(1)
      ELSE
********************************************

ctp        IF(DEBUG.GE.1)WRITE (MONIOU,201)
        WRITE (MONIOU,201)
201     FORMAT(2X,'QGSPSAINI: HARD CROSS SECTIONS CALCULATION')
c--------------------------------------------------
c Hard pomeron inclusive cross sections calculation
c--------------------------------------------------
c EQ(I) - energy squared tabulation (Q0^2, 4*Q0^2, ...)
      DO 1 I=1,17
1     EQ(I)=QT0*4.D0**FLOAT(I-1)

      DO 2 I=1,17
c QI - effective momentum (Qt**2/(1-z)**2) cutoff for the Born process
      QI=EQ(I)
c M, L define parton types (1-g, 2-q)
      DO 2 M=1,2
      DO 2 L=1,2
c K defines c.m. energy squared for the process (for current energy tabulation)
      DO 2 K=1,17
      K1=K+17*(M-1)+34*(L-1)
      IF(K.LE.I.OR.K.EQ.2)THEN
        CSBORN(I,K1)=0.D0
      ELSE
c SK - c.m. energy squared for the hard interaction
        SK=EQ(K)
c CSBORN(I,K1) - Born cross-section (2->2 process) - procedure QGSPSBORN
        CSBORN(I,K1)=QGSPSBORN(QI,SK,M-1,L-1)
      ENDIF
2     CONTINUE

c Cross-sections initialization
      DO 3 I=1,17
      DO 3 J=1,17
      N=MAX(I,J)
      DO 3 M=1,2
      DO 3 L=1,2
      ML=M+2*L-2
      DO 3 K=1,17
      K1=K+17*(M-1)+34*(L-1)
      CSJET(I,J,K1)=0.D0
      IF(K.LE.N.OR.K.EQ.2)THEN
        CSTOT(I,J,K1)=-80.D0
        CS0(I,J,K1)=-80.D0
        MIJ(I,J,ML)=K+1
        NIJ(I,J,ML)=K+1
      ELSE
        CSTOT(I,J,K1)=DLOG(CSBORN(N,K1))
        CS0(I,J,K1)=CSTOT(I,J,K1)
      ENDIF
3     CONTINUE

c N-maximal number of ladder runs taken into account
      N=2
4     CONTINUE
        IF(DEBUG.GE.2)WRITE (MONIOU,202)N,EQ(MIJ(1,1,1)),EQ(NIJ(1,1,1))
202     FORMAT(2X,'PSAINI: NUMBER OF LADDER RUNS TO BE CONSIDERED:',I2/
     *  4X,'MINIMAL MASSES SQUARED FOR THE UNORDERED AND STRICTLY',
     *  ' ORDERED LADDERS:'/4X,E10.3,3X,E10.3)
      DO 6 I=1,17
c QI - effective momentum cutoff for upper end of the ladder
      QI=EQ(I)
      DO 6 J=1,17
c QJ - effective momentum cutoff for lower end of the ladder
      QJ=EQ(J)
c QQ - maximal effective momentum cutoff
      QQ=MAX(QI,QJ)
c S2MIN - minimal energy squared for 2->2 subprocess
      S2MIN=MAX(QQ,4.D0*QT0)
      SM=DSQRT(QT0/S2MIN)
c SMIN - minimal energy squared for 2->3 subprocess
      SMIN=S2MIN*(1.D0+SM)/(1.D0-SM)

c M, L define parton types (1-g, 2-q)
      DO 6 M=1,2
      DO 6 L=1,2
      ML=M+2*L-2
c KMIN corresponds to minimal energy at which more runs are to be considered -
c stored in array NIJ(I,J,ML) - for strictly ordered ladder
      KMIN=NIJ(I,J,ML)
      IF(KMIN.LE.17)THEN
        DO 5 K=KMIN,17
        SK=EQ(K)
        IF(SK.LE.SMIN)THEN
          NIJ(I,J,ML)=NIJ(I,J,ML)+1
        ELSE
          K1=K+17*(M-1)+34*(L-1)
c CS1(I,J,K1) - cross-section for strictly ordered ladder (highest virtuality run
c is the lowest one) - procedure QGSPSJET1
          CS1(I,J,K1)=QGSPSJET1(QI,QJ,SK,S2MIN,M-1,L)
        ENDIF
5       CONTINUE
      ENDIF
6     CONTINUE

      DO 8 I=1,17
      DO 8 J=1,17
      DO 8 M=1,2
      DO 8 L=1,2
      ML=M+2*L-2
      KMIN=NIJ(I,J,ML)
      IF(KMIN.LE.17)THEN
        DO 7 K=KMIN,17
        K1=K+17*(M-1)+34*(L-1)
c CSJ - cross-section for strictly ordered ladder (highest virtuality run is the
c lowest one) - Born contribution is added
        CSJ=CS1(I,J,K1)+CSBORN(MAX(I,J),K1)
        IF(DEBUG.GE.2)WRITE (MONIOU,204)CSJ,EXP(CS0(I,J,K1))
204     FORMAT(2X,'PSAINI: NEW AND OLD VALUES OF THE CONTRIBUTION',
     *  ' OF THE STRICTLY ORDERED LADDER:'/4X,E10.3,3X,E10.3)
        IF(CSJ.EQ.0.D0.OR.ABS(1.D0-EXP(CS0(I,J,K1))/CSJ).LT.1.D-2)THEN
               NIJ(I,J,ML)=NIJ(I,J,ML)+1
        ELSE
c CS0(I,J,K1) - cross-section logarithm for strictly ordered ladder
          CS0(I,J,K1)=DLOG(CSJ)
        ENDIF
7       CONTINUE
      ENDIF
8     CONTINUE

      DO 10 I=1,17
      QI=EQ(I)
      DO 10 J=1,17
      QJ=EQ(J)
      QQ=MAX(QI,QJ)
      S2MIN=MAX(QQ,4.D0*QT0)
      SM=DSQRT(QT0/S2MIN)
c SMIN - minimal energy squared for 2->3 subprocess
      SMIN=S2MIN*(1.D0+SM)/(1.D0-SM)

      DO 10 M=1,2
      DO 10 L=1,2
      ML=M+2*L-2
c KMIN corresponds to minimal energy at which more runs are to be considered
c stored in array MIJ(I,J,ML) - for any ordering in the ladder
      KMIN=MIJ(I,J,ML)
      IF(KMIN.LE.17)THEN
        DO 9 K=KMIN,17
        SK=EQ(K)
        IF(SK.LE.SMIN)THEN
          MIJ(I,J,ML)=MIJ(I,J,ML)+1
        ELSE
          K1=K+17*(M-1)+34*(L-1)
c CS1(I,J,K1) - cross-section for any ordering in the ladder (highest virtuality
c run is somewhere in the middle; runs above and below it are strictly ordered
c towards highest effective momentum run) - procedure QGSPSJET
          CS1(I,J,K1)=QGSPSJET(QI,QJ,SK,S2MIN,M-1,L)
        ENDIF
9       CONTINUE
      ENDIF
10    CONTINUE

      DO 12 I=1,17
      DO 12 J=1,17
      DO 12 M=1,2
      DO 12 L=1,2
      ML=M+2*L-2
c KMIN corresponds to minimal energy at which more runs are to be considered
      KMIN=MIJ(I,J,ML)
      IF(KMIN.LE.17)THEN
        DO 11 K=KMIN,17
        K1=K+17*(M-1)+34*(L-1)
        K2=K+17*(L-1)+34*(M-1)
        CSJ=CS1(I,J,K1)+EXP(CS0(J,I,K2))
        IF(CSJ.EQ.0.D0.OR.ABS(1.D0-EXP(CSTOT(I,J,K1))/CSJ).LT.1.D-2)
     *  MIJ(I,J,ML)=MIJ(I,J,ML)+1
        IF(DEBUG.GE.2)WRITE (MONIOU,203)CSJ,EXP(CSTOT(I,J,K1))
203     FORMAT(2X,'PSAINI: NEW AND OLD VALUES OF THE UNORDERED LADDER',
     *  ' CROSS SECTION:'/4X,E10.3,3X,E10.3)
11      CSTOT(I,J,K1)=DLOG(CSJ)
      ENDIF
12    CONTINUE

c One more run
      N=N+1
      DO 13 L=1,4
13    IF(MIJ(1,1,L).LE.17.OR.NIJ(1,1,L).LE.17)GOTO 4

c Logarithms of the Born cross-section are calculated - to be interpolated in the
c QGSPSBINT procedure
      DO 14 I=1,17
      DO 14 K=1,17
      DO 14 M=1,2
      DO 14 L=1,2
      K1=K+17*(M-1)+34*(L-1)
      IF(K.LE.I.OR.K.EQ.2)THEN
        CSBORN(I,K1)=-80.D0
      ELSE
        CSBORN(I,K1)=DLOG(CSBORN(I,K1))
      ENDIF
14    CONTINUE

c Total and Born hard cross-sections logarithms for minimal cutoff (QT0) - to be
c interpolated in the PSJINT0 procedure
      DO 15 M=1,2
      DO 15 L=1,2
      DO 15 K=1,17
      IF(K.LE.2)THEN
        CSQ(K,M,L)=-80.D0
        CSBQ(K,M,L)=-80.D0
      ELSE
        K1=K+17*(M-1)+34*(L-1)
        CSBQ(K,M,L)=CSBORN(1,K1)
        CSQ(K,M,L)=CSTOT(1,1,K1)
      ENDIF
15    CONTINUE

c-------------------------------------------------
c FSUD(K,M)=-ln(SUD) - timelike Sudakov formfactor logarithm - procedure
c PSUDT(QMAX,M-1), M=1 - g, M=2 - q
      DO 17 M=1,2
      FSUD(1,M)=0.D0
      DO 17 K=2,10
c QMAX is the maximal effective momentum ( Qt**2/z**2/(1-z)**2 in case of the timelike
c evolution )
      QMAX=QTF*4.D0**(1.D0+K)
17    FSUD(K,M)=PSUDT(QMAX,M-1)

c QRT(K,L,M) - effective momentum logarithm for timelike branching ( ln QQ/16/QTF )
c for given QMAX (defined by K, QLMAX = ln QMAX/16/QTF ) and a number
c of random number values (defined by L) - to be interpolated by the PSQINT
c procedure; M=1 - g, M=2 - q
      DO 18 M=1,2
      DO 18 K=1,10
      QLMAX=1.38629D0*(K-1)
      QRT(K,1,M)=0.D0
      QRT(K,101,M)=QLMAX
      DO 18 I=1,99
      IF(K.EQ.1)THEN
        QRT(K,I+1,M)=0.D0
      ELSE
        QRT(K,I+1,M)=PSROOT(QLMAX,.01D0*I,M)
      ENDIF
18    CONTINUE
c-------------------------------------------------

        IF(DEBUG.GE.2)WRITE (MONIOU,205)
205    FORMAT(2X,'QGSPSAINI: PRETABULATION OF THE INTERACTION EIKONALS')
c-------------------------------------------------
************************************************************************
c-------------------------------------------------
c Interaction cross sections
c Factors for interaction eikonals calculation
c (convolution of the hard cross-sections with partons structure functions)
c - to be used in the PSPSFAZ procedure
c-------------------------------------------------
      IA(1)=1
c-------------------------------------------------
      DO 21 IE=1,10
c Energy of the interaction (per nucleon)
      E0N=10.D0**IE
c-------------------------------------------------
c Energy dependent factors:
c WP0, WM0 - initial light cone momenta for the interaction (E+-p)
      S=2.D0*E0N*AMN
c Y0 - total rapidity range for the interaction
      Y0=DLOG(S)

c Type of the incident hadron (icz = 1: pion, 2: nucleon, 3: kaon, etc
      DO 21 ICZ=1,5
c RS - soft pomeron elastic scattering slope (lambda_ab)
      RS=RQ(ICZ)+ALFP*Y0
c RS0 - initial slope (sum of the pomeron-hadron vertices slopes squared - R_ab)
      RS0=RQ(ICZ)
c FS - factor for pomeron eikonal calculation
c                            (gamma_ab * s**del /lambda_ab * C_ab
      FS=FP(ICZ)*EXP(Y0*DEL)/RS*CD(ICZ)
c RP1 - factor for the impact parameter dependence of the eikonal ( in fm^2 )
      RP1=RS*4.D0*.0391D0/AM**2
c Factor for cross-sections calculation ( in mb )
      G0=PI*RP1/CD(ICZ)*AM**2*10.D0
c SJV - valence-valence cross-section (divided by 8*pi*lambda_ab)
      SJV(IE,ICZ)=QGSPSHARD(S,ICZ)
      SJV0=SJV(IE,ICZ)

      DO 19 I=1,5
      DO 19 M=1,3
      Z=.2D0*I
c Eikonals for gluon-gluon and valence-gluon semihard interactions
c (m=1 - gg, 2 - qg, 3 - gq);
c Z - impact parameter factor ( exp(-b**2/R_p) )
      M1=M+3*(ICZ-1)
      FJS(IE,I,M1)=DLOG(QGSPSFSH(S,Z,ICZ,M-1)/Z)
      FJS0(I,M)=FJS(IE,I,M1)
19    CONTINUE

      DO 20 IIA=1,4
c Target mass number IA(2)
      IA(2)=4**(IIA-1)
      IF(DEBUG.GE.1)WRITE (MONIOU,206)E0N,TY(ICZ),IA(2)
206   FORMAT(2X,'QGSPSAINI: INITIAL PARTICLE ENERGY:',E10.3,2X,
     *'ITS TYPE:',A7,2X,'TARGET MASS NUMBER:',I2)
c-------------------------------------------------
c Nuclear radii
      IF(IA(2).GT.10)THEN
c RD - Wood-Saxon density radius (fit to the data of Murthy et al.)
        RD(2)=0.7D0*FLOAT(IA(2))**.446/AM
      ELSE
c RD - gaussian density radius (for light nucleus)
        RD(2)=.9D0*FLOAT(IA(2))**.3333/AM
      ENDIF

      IF(IA(2).EQ.1)THEN
c Hadron-proton interaction
c BM - impact parameter cutoff value
        BM=2.D0*DSQRT(RP1)
c XXFZ - impact parameter integration for the hadron-nucleon interaction eikonal;
c GZ0 - total and absorptive cross-sections (up to a factor); first parameter is
c used only in case of hadron-nucleus interaction (to make convolution with target
c nucleus profile function)
        CALL XXFZ(0.D0,GZ0)
        if (debug .ge.1) write (moniou,*)gz0
c GTOT - total cross-section
        GTOT=G0*GZ0(1)
c GABS - cut pomerons cross-section
        GABS=G0*GZ0(2)*.5D0
c GD0 - cross-section for the cut between pomerons
        GD0=GTOT-GABS
c GDP - projectile diffraction cross section
        GDP=(1.D0-CC(ICZ))*CC(2)*GD0
c GDT - target diffraction cross section
        GDT=(1.D0-CC(2))*CC(ICZ)*GD0
c  GDD - double diffractive cross section
        GDD=(1.D0-CC(ICZ))*(1.D0-CC(2))*GD0
c GIN - inelastic cross section
        GIN=GABS+GDP+GDT+GDD
        GEL=GD0*CC(ICZ)*CC(2)
c
        IF(DEBUG.GE.1)WRITE (MONIOU,225)GTOT,GIN,GEL,GDP,GDT,GDD
c
225     FORMAT(2X,'QGSPSAINI: HADRON-PROTON CROSS SECTIONS:'/
     *  4X,'GTOT=',E10.3,2X,'GIN=',E10.3,2X,'GEL=',E10.3/4X,
     *  'GDIFR_PROJ=',E10.3,2X,'GDIFR_TARG=',E10.3,2X,
     *  'G_DOUBLE_DIFR',E10.3)
c GZ - probability to have target diffraction
        GZ(IE,ICZ,IIA)=GDT/GIN
        GZP(IE,ICZ,IIA)=(GDP+GDD)/GIN   ! so00
C??????
        GSECT(IE,ICZ,IIA)=LOG(GIN)
C??????
      ELSE

c Hadron-nucleus interaction
c BM - impact parameter cutoff value
        BM=RD(2)+DLOG(29.D0)
c RRR - Wood-Saxon radius for the target nucleus
        RRR=RD(2)
c RRRM - auxiliary parameter for numerical integration
        RRRM=RRR+DLOG(9.D0)
c ANORM - nuclear density normalization factor multiplied by RP1
        ANORM=1.5D0/PI/RRR**3/(1.D0+(PI/RRR)**2)*RP1

c GAU(GZ) - cross sections calculation ( integration over impact parameters less than
c BM )
        CALL XXGAU(GZ1)
c GAU1(GZ) - cross sections calculation ( integration over impact
c parameters greater than BM )
        CALL XXGAU1(GZ1)
c GIN - total inelastic cross section
        GIN=GZ1(1)+GZ1(2)+GZ1(3)
c
        IF(DEBUG.GE.1)WRITE (MONIOU,224)
     *  GIN*10.D0,GZ1(1)*10.D0,GZ1(2)*10.D0
c
224     FORMAT(2X,'QGSPSAINI: HADRON-NUCLEUS CROSS SECTIONS:'/
     *  4X,'GIN=',E10.3,2X,'GDIFR_TARG=',E10.3,2X,
     *  'GDIFR_PROJ=',E10.3)
c GZ - probability to have target diffraction
        GZ(IE,ICZ,IIA)=GZ1(1)/GIN
        GZP(IE,ICZ,IIA)=GZ1(2)/GIN   ! so00
C??????
        GIN=GIN*10.
        GSECT(IE,ICZ,IIA)=LOG(GIN)
C??????
      ENDIF
20    CONTINUE
21    CONTINUE

c Rejection functions calculation - to be interpolated in the RJINT procedure
      DO 23 I=1,50
c Rapidity range tabulation for the hard interaction
      YJ=AQT0+.5D0*I
c Rejection function for valence quark energy distribution
      RJV(I)=PSREJV(EXP(YJ))

      DO 22 J=1,5
      DO 22 M=1,2
      Z=.2D0*J
      DO 22 ICZ=1,5
c RS0 - initial slope (sum of the pomeron-hadron vertices slopes squared - R_ab)
      RS0=RQ(ICZ)
      M1=M+2*(ICZ-1)
c Rejection function for semihard block energy distribution  (m=1 - gg,
c 2 - qg)
      RJS(I,J,M1)=PSREJS(EXP(YJ),Z,M-1)
22    CONTINUE
23    CONTINUE

ctp        IF(DEBUG.GE.1)WRITE (MONIOU,212)
        WRITE (MONIOU,212)
212     FORMAT(2X,'PSAINI: HARD CROSS SECTIONS ARE WRITTEN TO THE FILE'
     *  ,' QGSDAT01')
        IF(IFDAT.NE.1)THEN
          OPEN(1,FILE='QGSDAT01',STATUS='unknown')
        ELSE                                                  !ctp
          OPEN(IFDAT,FILE=FNDAT(1:NFNDAT),STATUS='unknown')
        ENDIF
        WRITE (1,*)CSBORN,CS0,CSTOT,CSQ,CSBQ,
     *  FSUD,QRT,SJV,FJS,RJV,RJS,GZ,GZP,GSECT
        CLOSE(1)
      ENDIF
************************************************************************

cdh 8/10/98
c Nuclear cross sections
      IF(IFNCS.NE.2)THEN
        INQUIRE(FILE='SECTNU',EXIST=LCALC)
      ELSE                                                  !ctp
        INQUIRE(FILE=FNNCS(1:NFNNCS),EXIST=LCALC)
      ENDIF
      IF(LCALC)then
        IF(DEBUG.GE.1)WRITE (MONIOU,208)
208     FORMAT(2X,'PSAINI: NUCLEAR CROSS SECTIONS READOUT FROM THE FILE'
     *  ,' SECTNU')
        IF(IFNCS.NE.2)THEN
          OPEN(2,FILE='SECTNU',STATUS='OLD')
        ELSE                                                  !ctp
          OPEN(IFNCS,FILE=FNNCS(1:NFNNCS),STATUS='OLD')
        ENDIF
        READ (2,*)QGSASECT
        CLOSE(2)
      ELSE
cdh     NITER=1000          !NUMBER OF ITERATIONS
        NITER=5000          !NUMBER OF ITERATIONS
        DO IE=1,10
          E0N=10.D0**IE
        DO IIA1=1,6
          IAP=2**IIA1
        DO IIA2=1,4
          IAT=4**(IIA2-1)
ctp          IF(DEBUG.GE.1)WRITE (MONIOU,207)E0N,IAP,IAT
          WRITE (MONIOU,207)E0N,IAP,IAT
207       FORMAT(2X,'QGSPSAINI: INITIAL NUCLEUS ENERGY:',E10.3,2X,
     *    'PROJECTILE MASS:',I2,2X,'TARGET MASS:',I2)
          CALL XXAINI(E0N,2,IAP,IAT)
          CALL CROSSC(NITER,GTOT,GPROD,GABS,GDD,GQEL,GCOH)
          IF(DEBUG.GE.1)WRITE (MONIOU,209)
     *    GTOT,GPROD,GABS,GDD,GQEL,GCOH
c         WRITE (*,*)GTOT,GPROD
209       FORMAT(2X,'GTOT',D10.3,'  GPROD',D10.3,' GABS',D10.3/2X,
     *    'GDD',D10.3,'  GQEL',D10.3,' GCOH',D10.3)
          QGSASECT(IE,IIA1,IIA2)=LOG(GPROD)
        ENDDO
        ENDDO
        ENDDO
        IF(IFNCS.NE.2)THEN
          OPEN(2,FILE='SECTNU',STATUS='UNKNOWN')
        ELSE                                                  !ctp
          OPEN(IFNCS,FILE=FNNCS(1:NFNNCS),STATUS='UNKNOWN')
        ENDIF
        WRITE (2,*)QGSASECT
        CLOSE(2)
      ENDIF
cdh  end
      IF(DEBUG.GE.3)WRITE (MONIOU,218)
218   FORMAT(2X,'QGSPSAINI - END')
      RETURN
      END
C=======================================================================

        FUNCTION PSAPINT(X,J,L)
c PSAPINT - integrated Altarelli-Parisi function
c X - light cone momentum share value,
c J - type of initial parton (0 - g, 1 - q)
c L - type of final parton (0 - g, 1 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)X,J,L
201     FORMAT(2X,'PSAPINT: X=',E10.3,2X,'J= ',I1,2X,'L= ',I1)
        IF(J.EQ.0)THEN
          IF(L.EQ.0)THEN
            PSAPINT=6.D0*(DLOG(X/(1.D0-X))-X**3/3.D0+X**2/2.D0-2.D0*X)
          ELSE
            PSAPINT=3.D0*(X+X**3/1.5D0-X*X)
          ENDIF
        ELSE
          IF(L.EQ.0)THEN
            PSAPINT=(DLOG(X)-X+.25D0*X*X)/.375D0
          ELSE
            Z=1.D0-X
            PSAPINT=-(DLOG(Z)-Z+.25D0*Z*Z)/.375D0
          ENDIF
        ENDIF
        IF(DEBUG.GE.2)WRITE (MONIOU,202)PSAPINT
202     FORMAT(2X,'PSAPINT=',E10.3)
        RETURN
        END
C=======================================================================

      SUBROUTINE PSASET
c Common model parameters setting
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      CHARACTER*7 TY
      COMMON /Q_AREA15/ FP(5),RQ(5),CD(5)
      COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
      COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
      COMMON /Q_AREA25/ AHV(5)
      COMMON /Q_AREA26/ FACTORK
      COMMON /Q_AREA41/ TY(5)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      COMMON /Q_QGSNEX1/ XA,XB,BQGS,BMAXQGS,BMAXNEX,BMINNEX     !ctp
      DIMENSION XA(210,3),XB(210,3)
      SAVE

        IF(DEBUG.GE.1)WRITE (MONIOU,210)
210     FORMAT(2X,'PSASET - COMMON MODEL PARAMETERS SETTING')

      BQGS=0.d0            !ctp used to link with nexus
      BMAXQGS=0.d0         !ctp used to link with nexus
      BMAXNEX=-1.d0         !ctp used to link with nexus
      BMINNEX=0.d0         !ctp used to link with nexus

c Soft pomeron parameters:
c DEL - overcriticity,
c ALFP - trajectory slope;
c FP(i) - vertices for pomeron-hadrons interaction (gamma(i)*gamma(proton)),
c RQ(i) - vertices slopes (R(i)**2+R(proton)**2),
c CD(i) - shower enhancement coefficients
c (i=1,...5 - pion,proton,kaon,D-meson,Lambda_C ),
c (Kaidalov et al., Sov.J.Nucl.Phys.,1984 - proton and pion parameters)
      DEL=.07D0
      ALFP=.21D0

      FP(1)=2.43D0
      RQ(1)=2.4D0
      CD(1)=1.6D0

      FP(2)=3.64D0
      RQ(2)=3.56D0
      CD(2)=1.5D0

      FP(3)=1.75D0
      RQ(3)=2.D0
      CD(3)=1.7D0

      FP(4)=1.21D0
      RQ(4)=1.78D0
      CD(4)=2.0D0

      FP(5)=2.43D0
      RQ(5)=2.4D0
      CD(5)=2.0D0

c-------------------------------------------------
c Hard interaction parameters:
c ALM  - Lambda_QCD squared,
c QT0  - Q**2 cutoff,
c RR   - vertex constant square for soft pomeron interaction with the hard block (r**2),;
c BET  - gluon structure function parameter for the soft pomeron ((1-x)**BET),
c AMJ0 - jet mass,
c QTF  - Q**2 cutoff for the timelike evolution,
c FACTORK - K-factor value;
c DELH is not a parameter of the model; it is used only for energy sharing
c procedure - initially energy is shared according to s**DELH dependence
c for the hard interaction cross-section and then rejection is used according
c to real Sigma_hard(s) dependence.
      ALM=.04D0
      RR=.35D0     !  produces 76 mbarn for p-pbar at Tevatron energies
cdh   RR=.53D0     !  produces 80 mbarn for p-pbar at Tevatron energies
      QT0=4.D0
      BET=1.D0
      DELH=0.25D0
      AMJ0=0.D0
      QTF=.5D0
      FACTORK=2.D0

c-------------------------------------------------
c Valence quark structure functions for the hard scattering
c (~1/sqrt(x)*(1-x)**AHV(i), i=1,...5 corresponds to pion, nucleon etc.)
      AHV(1)=1.5D0
      AHV(2)=2.5D0
      AHV(3)=2.D0
      AHV(4)=4.D0
      AHV(5)=5.D0
c Initial particle types
      TY(1)='pion   '
      TY(2)='nucleon'
      TY(3)='kaon   '
      TY(4)='D-meson'
      TY(5)='LambdaC'
      RETURN
      END
C=======================================================================

        FUNCTION QGSPSBINT(QQ,S,M,L)
C QGSPSBINT - Born cross-section interpolation
c QQ - effective momentum cutoff for the scattering,
c S - total c.m. energy squared for the scattering,
c M - parton type at current end of the ladder (1 - g, 2 - q)
c L - parton type at opposite end of the ladder (1 - g, 2 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION WI(3),WK(3)
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA31/ CSJ(17,68)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)QQ,S,M,L
201     FORMAT(2X,'QGSPSBINT: QQ=',E10.3,2X,'S= ',E10.3,2X,'M= ',I1,2X,
     *  'L= ',I1)
        QGSPSBINT=0.D0
        IF(S.LE.MAX(4.D0*QT0,QQ))THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSBINT
202     FORMAT(2X,'QGSPSBINT=',E10.3)
          RETURN
        ENDIF

        ML=17*(M-1)+34*(L-1)
        QLI=DLOG(QQ/QT0)/1.38629d0
        SL=DLOG(S/QT0)/1.38629d0
        SQL=SL-QLI
        I=INT(QLI)
        K=INT(SL)
        IF(I.GT.13)I=13

        IF(SQL.GT.10.D0)THEN
          IF(K.GT.14)K=14
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WK(2)=SL-K
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 1 I1=1,3
          DO 1 K1=1,3
1         QGSPSBINT=QGSPSBINT+CSJ(I+I1,K+K1+ML)*WI(I1)*WK(K1)
          QGSPSBINT=EXP(QGSPSBINT)
        ELSEIF(SQL.LT.1.D0.AND.I.NE.0)THEN
          SQ=(S/QQ-1.D0)/3.D0
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)

          DO 2 I1=1,3
          I2=I+I1
          K2=I2+1+ML
2         QGSPSBINT=QGSPSBINT+CSJ(I2,K2)*WI(I1)
          QGSPSBINT=EXP(QGSPSBINT)*SQ
        ELSEIF(K.EQ.1)THEN
          SQ=(S/QT0/4.D0-1.D0)/3.D0
          WI(2)=QLI
          WI(1)=1.D0-QLI

          DO 3 I1=1,2
3         QGSPSBINT=QGSPSBINT+CSJ(I1,3+ML)*WI(I1)
          QGSPSBINT=EXP(QGSPSBINT)*SQ
        ELSEIF(K.LT.15)THEN
          KL=INT(SQL)
          IF(I+KL.GT.12)I=12-KL
          IF(I+KL.EQ.1)KL=2
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WK(2)=SQL-KL
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 4 I1=1,3
          I2=I+I1
          DO 4 K1=1,3
          K2=I2+KL+K1-1+ML
4         QGSPSBINT=QGSPSBINT+CSJ(I2,K2)*WI(I1)*WK(K1)
          QGSPSBINT=EXP(QGSPSBINT)

        ELSE
          K=15
          IF(I.GT.K-3)I=K-3
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WK(2)=SL-K
          WK(1)=1.D0-WK(2)

          DO 5 I1=1,3
          DO 5 K1=1,2
5         QGSPSBINT=QGSPSBINT+CSJ(I+I1,K+K1+ML)*WI(I1)*WK(K1)
          QGSPSBINT=EXP(QGSPSBINT)
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSBINT
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSBORN(QQ,S,IQ1,IQ2)
c PSFBORN -hard 2->2 parton scattering Born cross-section
c S is the c.m. energy square for the scattering process,
c IQ1 - parton type at current end of the ladder (0 - g, 1,2 - q)
c IQ2 - parton type at opposite end of the ladder (0 - g, 1,2 - q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA6/  PI,BM,AM
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA26/ FACTORK
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        COMMON /Q_AR13/  X1(7),A1(7)
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)QQ,S,IQ1,IQ2
201     FORMAT(2X,'QGSPSBORN: QQ=',E10.3,2X,'S= ',E10.3,2X,'IQ1= ',I1
     *  ,2X,'IQ2= ',I1)
        TMIN=S*(.5D0-DSQRT(.25D0-QT0/S))
        TMIN=MAX(TMIN,S*QQ/(S+QQ))

        IF(IQ1*IQ2.EQ.0)THEN
          IQ=IQ2
        ELSE
          IQ=2
        ENDIF

        QGSPSBORN=0.D0
        DO 1 I=1,7
        DO 1 M=1,2
        T=2.D0*TMIN/(1.D0+2.D0*TMIN/S-X1(I)*(2*M-3)*(1.D0-2.D0*TMIN/S))
        QT=T*(1.D0-T/S)
        FB=PSFBORN(S,T,IQ1,IQ)+PSFBORN(S,S-T,IQ1,IQ)
1       QGSPSBORN=QGSPSBORN+A1(I)*FB/DLOG(QT/ALM)**2*T**2
        QGSPSBORN=QGSPSBORN*(.5D0/TMIN-1.D0/S)*FACTORK*PI**3/2.25D0**2
     &            /S**2
        IF(IQ1.EQ.0.AND.IQ2.EQ.0)QGSPSBORN=QGSPSBORN*.5D0
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSBORN
202     FORMAT(2X,'QGSPSBORN=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE PSCAJET(QQ,IQ1,QV,ZV,QM,IQV,LDAU,LPAR,JQ)
c Final state emission process (all branchings as well as parton masses
c are determined)
C-----------------------------------------------------------------------
c QQ - maximal effective momentum transfer for the first branching
c IQ1, IQ2 - initial jet flavours in forward and backward direction
c (0 - for gluon)
c QV(i,j) - effective momentum for the branching of the parton in i-th row
c on j-th level (0 - in case of no branching)  - to be determined
c ZV(i,j) - Z-value for the branching of the parton in i-th row
c on j-th level - to be determined
c QM(i,j) - mass squared for the parton in i-th row
c on j-th level - to be determined
c IQV(i,j) - flavour for the parton in i-th row on j-th level
c - to be determined
c LDAU(i,j) - first daughter row for the branching of the parton in i-th row
c on j-th level - to be determined
c LPAR(i,j) - the parent row for the parton in i-th row
c on j-th level - to be determined
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION QMAX(30,50),IQM(2),LNV(50),
     *  QV(30,50),ZV(30,50),QM(30,50),IQV(30,50),
     *  LDAU(30,49),LPAR(30,50)

        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG

        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)QQ,IQ1,JQ
201     FORMAT(2X,'PSCAJET: QQ=',E10.3,2X,'IQ1= ',I1,2X,'JQ=',I1)

        DO 1 I=2,20
1        LNV(I)=0
        LNV(1)=1
        QMAX(1,1)=QQ
        IQV(1,1)=IQ1
        NLEV=1
        NROW=1

2        QLMAX=DLOG(QMAX(NROW,NLEV)/QTF/16.D0)
         IQ=MIN(1,IABS(IQV(NROW,NLEV)))+1

        IF(PSRAN(B10).GT.PSUDINT(QLMAX,IQ))THEN
          Q=PSQINT(QLMAX,PSRAN(B10),IQ)
          Z=PSZSIM(Q,IQ)

          LL=LNV(NLEV+1)+1
          LDAU(NROW,NLEV)=LL
          LPAR(LL,NLEV+1)=NROW
          LPAR(LL+1,NLEV+1)=NROW
          LNV(NLEV+1)=LL+1

          IF(IQ.NE.1)THEN
            IF((3-2*JQ)*IQV(NROW,NLEV).GT.0)THEN
              IQM(1)=0
              IQM(2)=IQV(NROW,NLEV)
            ELSE
              IQM(2)=0
              IQM(1)=IQV(NROW,NLEV)
              Z=1.D0-Z
            ENDIF
          ELSE
*********************************************************
            WG=QGSPSFAP(Z,0,0)
*********************************************************
            WG=WG/(WG+QGSPSFAP(Z,0,1))
            IF(PSRAN(B10).LT.WG)THEN
              IQM(1)=0
              IQM(2)=0
            ELSE
              IQM(1)=INT(3.D0*PSRAN(B10)+1.D0)*(3-2*JQ)
              IQM(2)=-IQM(1)
            ENDIF
            IF(PSRAN(B10).LT..5D0)Z=1.D0-Z
          ENDIF

          QV(NROW,NLEV)=Q
          ZV(NROW,NLEV)=Z

          NROW=LL
          NLEV=NLEV+1
          QMAX(NROW,NLEV)=Q*Z**2
          QMAX(NROW+1,NLEV)=Q*(1.D0-Z)**2
          IQV(NROW,NLEV)=IQM(1)
          IQV(NROW+1,NLEV)=IQM(2)
        IF(DEBUG.GE.3)WRITE (MONIOU,203)NLEV,NROW,Q,Z
203     FORMAT(2X,'PSCAJET: NEW BRANCHING AT LEVEL NLEV=',I2,
     *  ' NROW=',I2/4X,' EFFECTIVE MOMENTUM Q=',E10.3,2X,' Z=',E10.3)
          GOTO 2
        ELSE

          QV(NROW,NLEV)=0.D0
          ZV(NROW,NLEV)=0.D0
          QM(NROW,NLEV)=AMJ0
        IF(DEBUG.GE.3)WRITE (MONIOU,204)NLEV,NROW
204     FORMAT(2X,'PSCAJET: NEW FINAL JET AT LEVEL NLEV=',I2,
     *  ' NROW=',I2)
        ENDIF

4       CONTINUE
      IF(NLEV.EQ.1)THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'PSCAJET - END')
        RETURN
      ENDIF
        LPROW=LPAR(NROW,NLEV)

        IF(LDAU(LPROW,NLEV-1).EQ.NROW)THEN
          NROW=NROW+1
          GOTO 2
        ELSE
          Z=ZV(LPROW,NLEV-1)
          QM(LPROW,NLEV-1)=Z*(1.D0-Z)*QV(LPROW,NLEV-1)
     *          +QM(NROW-1,NLEV)/Z+QM(NROW,NLEV)/(1.D0-Z)
          NROW=LPROW
          NLEV=NLEV-1
        IF(DEBUG.GE.3)WRITE (MONIOU,205)NLEV,NROW,QM(LPROW,NLEV)
205     FORMAT(2X,'PSCAJET: JET MASS AT LEVEL NLEV=',I2,
     *  ' NROW=',I2,' - QM=',E10.3)
          GOTO 4
        ENDIF
        END
C=======================================================================

      SUBROUTINE PSCONF
c Simulation of the interaction configuration: impact parameter, nucleons positions,
c numbers of cut soft pomerons and semihard blocks, their connections.
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
c XA(56,3),XB(56,3) - arrays for projectile and target nucleons positions recording,
c FHARD(i) give the factors to the scattering amplitude due to
c valence quark-gluon (i=1),  gluon-valence quark (i=2) and
c valence quark-valence quark (i=3) interactions
cdh   DIMENSION XA(56,3),XB(56,3),FHARD(3)
      DIMENSION XA(210,3),XB(210,3),FHARD(3)
      COMMON /Q_QGSNEX1/ XA,XB,BQGS,BMAXQGS,BMAXNEX,BMINNEX      !ctp
      COMMON /Q_AREA1/  IA(2),ICZ,ICP
      COMMON /Q_AREA2/  S,Y0,WP0,WM0
      COMMON /Q_AREA6/  PI,BM,AM
c Arrays for interaction configuration recording:
c LQA(i) (LQB(j)) - numbers of cut soft pomerons, connected to i-th projectile
c (j-th target) nucleon (hadron);
c LHA(i) (LHB(j)) - the same for hard pomerons numbers;
c IAS(k) (IBS(k)) - number (position in array) of the projectile (target) nucleon,
c connected to k-th block of soft pomerons;
c NQS(k) - number of soft pomerons in k-th block;
c IAH(k) (IBH(k)) - number (position in array) of the projectile (target) nucleon,
c connected to k-th hard pomeron;
c ZH(k) - impact parameter between the two nucleons connected to k-th hard pomeron
c (more exactly exp(-b**2/RP1));
c LVA(i)=1 if valence quark from i-th nucleon (i=1 for hadron) is involved into
c the hard interaction and LVA(i)=0 otherwise, LVB(j) - similar.
      COMMON /Q_AREA9/  LQA(56),LQB(56),NQS(1000),IAS(1000),IBS(1000),
     *                LHA(56),LHB(56),ZH(4000),IAH(4000),IBH(4000),
     *                IQH(4000),LVA(56),LVB(56)
      COMMON /Q_AREA10/ STMASS,AM0,AMN,AMK,AMC,AMLAMC,AMLAM,AMETA
      COMMON /Q_AREA11/ B10
c NSP - number of secondary particles
      COMMON /Q_AREA12/ NSP
      COMMON /Q_AREA16/ CC(5)
      COMMON /Q_AREA40/ JDIFR
      COMMON /Q_AREA43/ MONIOU
**************************************************
      COMMON /Q_AREA45/ GDT,GDP    !so00
**************************************************
      COMMON /Q_AREA99/ NWT
      COMMON /Q_DEBUG/  DEBUG

ctp from epos
      integer ng1evt,ng2evt,ikoevt
      real    rglevt,sglevt,eglevt,fglevt,typevt
      common/c2evt/ng1evt,ng2evt,rglevt,sglevt,eglevt,fglevt,ikoevt
     *,typevt            !in epos.inc


      DIMENSION IWT(56)
      SAVE
      EXTERNAL PSRAN

        IF(DEBUG.GE.1)WRITE (MONIOU,201)
201     FORMAT(2X,'PSCONF - CONFIGURATION OF THE INTERACTION')

100     NSP=0
        typevt=1
        IF(IA(1).EQ.1)THEN
**************************************************
          IF(JDIFR.EQ.1.AND.PSRAN(B10).LT.GDT)THEN
c Target diffraction
            IF(IA(2).NE.1)THEN
c ICT - partner target nucleon type (proton - 2 or neutron - 3)
              ICT=INT(2.5+PSRAN(B10))
            ELSE
c Target proton
              ICT=2
            ENDIF
            WPI=WP0
            WMI=WM0
c              write (*,*)'difr'
            CALL XXDTG(WPI,WMI,ICP,ICT,0)
            typevt=-4
            goto 21   !so00
          ELSEIF(ABS(JDIFR).EQ.1.AND.PSRAN(B10).LT.GDP)THEN  !so00
            IF(IA(2).NE.1)THEN  !so00
c ICT - partner target nucleon type (proton - 2 or neutron - 3)
              ICT=INT(2.5+PSRAN(B10))  !so00
            ELSE  !so00
c Target proton
              ICT=2  !so00
            ENDIF  !so00
            IF(DEBUG.GE.2)WRITE (MONIOU,206)  !so00
206         FORMAT(2X,'PROJECTILE HADRON DIFFRACTION')  !so00
            ICP0=ICP  !so00
            WPI=WP0  !so00
            WMI=WM0  !so00
            LQ=0  !so00
            CALL XXDPR(WPI,WMI,ICP0,ICT,LQ)  !so00
            typevt=4
            goto 21   !so00
          ENDIF
**************************************************
c For hadron projectile we have given position in transverse plane;
c initially primary hadron is positioned at (X,Y)=(0,0)
          DO 1 I=1,3
1          XA(1,I)=0.D0
      ENDIF

c-------------------------------------------------
c Inelastic interaction at B<BM (usual case)
c-------------------------------------------------
c NW - number of wounded nucleons in the primary (NW=1 for hadron);
c NT - number of target nucleons being in their active diffractive state;
c LS - number of cut soft pomeron blocks (froissarons);
c NHP - number of cut pomerons having hard block (referred below as hard blocks);
c NQS(k) - number of cut soft pomerons in k-th block;
c IAS(k) (IBS(k)) - number (position in array) of the projectile (target) nucleon,
c connected to k-th block of soft pomerons;
c IAH(k) (IBH(k)) - number 3(position in array) of the projectile (target) nucleon,
c connected to k-th hard pomeron;
c ZH(k) - impact parameter between the two nucleons connected to k-th hard pomeron
c (more exactly exp(-b**2/RP1));
c LQA(i) (LQB(j)) - total number of cut soft pomerons, connected to i-th projectile
c (j-th target) nucleon (hadron);
c LHA(i) (LHB(j)) - total number of cut hard blocks, connected to i-th projectile
c (j-th target) nucleon (hadron);
c LVA(i)=1 if valence quark from i-th nucleon (i=1 for hadron) is involved into
c the hard interaction and LVA(i)=0 otherwise, LVB(j) - similar.
c-------------------------------------------------
c Initialization
      DO 3 I=1,IA(1)
        LHA(I)=0
        LVA(I)=0
3       LQA(I)=0
        DO 4 I=1,IA(2)
        LHB(I)=0
        LVB(I)=0
4       LQB(I)=0

c-------------------------------------------------
c The beginning
5       CONTINUE
**************************************************
        IF(IA(2).NE.1)THEN  !changed!!!!!!!!! dh 8/10/98
c For target nucleus number of target nucleons being in their active
c diffractive state is simulated (for each nucleon probability equals
c 1./C_n,  - shower enhancenment coefficient)
          NT=0
          DO 6 I=1,IA(2)
6         NT=NT+INT(CC(2)+PSRAN(B10))
c In case of no active target nucleon the event is rejected
          IF(NT.EQ.0)GOTO 5
        IF(DEBUG.GE.3)WRITE (MONIOU,203)NT
203     FORMAT(2X,'PSCONF: NUMBER OF ACTIVE TARGET NUCLEONS NT=',
     *  I2)
c PSGEA(NT,XB,2) - target nucleons positions simulation:
cdh       CALL PSGEA(NT,XB,2)  !changed!!!!!!!!!
          CALL PSGEA(IA(2),XB,2)  !changed!!!!!!!!! 25.03.99
c NT - number of target nucleons being in their active diffractive state;
c XB(i,n) - n-th nucleon coordinates (i=1,2,3 corresponds to x,y,z);
c parameter 2 means target
        ELSE                   !changed!!!!!!!!! dh 8/10/98
          NT=1                 !changed!!!!!!!!! dh 8/10/98
          XB(1,1)=0.D0         !changed!!!!!!!!! dh 8/10/98
          XB(1,2)=0.D0         !changed!!!!!!!!! dh 8/10/98
        ENDIF                  !changed!!!!!!!!! dh 8/10/98
**************************************************

c-------------------------------------------------
c Impact parameter  square is simulated uniformly (B**2<BM**2)
        B=BM*DSQRT(PSRAN(B10))
        IF(BMAXNEX.GE.0.D0)THEN
         B1=BMINNEX/AM
         B2=MIN(BM*AM,BMAXNEX)/AM
         if(B1.gt.B2)stop'bmin > bmax in QGSJet'
          B=DSQRT(B1*B1+(B2*B2-B1*B1)*PSRAN(B10))
          BQGS=B*AM                      !ctp
        ENDIF
        IF(DEBUG.GE.2)WRITE (MONIOU,204)B*AM
204     FORMAT(2X,'PSCONF: IMPACT PARAMETER FOR THE INTERACTION:',
     *  E10.3,' FM')
c PSGEA(IA(1),XA,1) - projectile nucleons positions simulation:
c IA(1) - projectile nucleus mass number;
c XA(i,n) - n-th nucleon coordinates (i=1,2,3 corresponds to x,y,z);
c parameter 1 means projectile
        IF(IA(1).GT.1)CALL PSGEA(IA(1),XA,1)

        NW=0
        LS=0
        NS=0
        NHP=0
        DO 101 IT = 1,NT
          IWT(IT) = 0
 101    CONTINUE

c-------------------------------------------------
c Cycle over all projectile nucleons ( for projectile hadron we have only IN=1 )
        DO 14 IN=1,IA(1)
        IF(DEBUG.GE.2.AND.ICZ.EQ.2)WRITE (MONIOU,205)IN
205     FORMAT(2X,'PSCONF: ',I2,'-TH PROJECTILE NUCLEON')
c Only nucleons in their active diffractive state are considered (for each nucleon
c probability equals 1./C_n, C_n = 1./CC(2) - shower enhancenment coefficient)
        IF(IA(1).NE.1.AND.PSRAN(B10).GT.CC(2))GOTO 12
c Projectile nucleons positions are shifted according the to impact parameter B
        X=XA(IN,1)+B
        Y=XA(IN,2)

        IQS=0
        NW=NW+1
c-------------------------------------------------
c Cycle over all target nucleons in active state
        DO 11 M=1,NT
c Z - b-factor for pomeron eikonal calculation (exp(-R_ij/R_p))
        Z=PSDR(X-XB(M,1),Y-XB(M,2))
c VV - eikonal for nucleon-nucleon (hadron-nucleon) interaction
c (sum of the soft and semihard eikonals)
        VV=2.D0*PSFAZ(Z,FSOFT,FHARD,FSHARD)
        EV=EXP(-VV)
c EH - eikonal contribution of valence quarks hard interactions
        EH=FHARD(1)+FHARD(2)+FHARD(3)
c        eh=0.d0
        AKS=PSRAN(B10)
c 1.-EXP(-VV)*(1.D0-2.D0*EH) is the probability for inelastic nucleon-nucleon
c (hadron-nucleon) interaction (for given nucleons positions)
        IF(AKS.GT.1.D0-EV*(1.D0-2.D0*EH))GOTO 11
        IF(DEBUG.GE.2)WRITE (MONIOU,208)M
208     FORMAT(2X,'PSCONF: INTERACTION WITH',I2,'-TH TARGET NUCLEON')
C  INCREMENT THE NUMBER IWT OF WOUNDED TARGET NUCLEONS
        IWT(M) = 1

c-------------------------------------------------
c IQV - type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
        IQV=0

c 2*EH*EV = 2*EH*EXP(-VV) - probability for only valence quarks hard interactions
c (with no one soft or semihard)
        SUM=2.D0*EH*EV

c-------------------------------------------------
        IF(AKS.LT.SUM)THEN
          AKS1=EH*PSRAN(B10)
          IF(AKS1.LT.FHARD(1))THEN
c Rejection in case of valence quark already involved into the interaction
            IF(LVA(NW).NE.0)GOTO 11
c LVA(NW)=1 - valence quark-gluon interaction
            LVA(NW)=1
            IQV=1
          ELSEIF(AKS1.LT.FHARD(1)+FHARD(2))THEN
c Rejection in case of valence quark already involved into the interaction
            IF(LVB(M).NE.0)GOTO 11
c LVB(M)=1 - gluon-valence quark interaction
            LVB(M)=1
            IQV=2
          ELSE
c Rejection in case of valence quarks already involved into the interaction
            IF(LVA(NW)+LVB(M).NE.0)GOTO 11
c LVA(NW)=LVB(M)=1 - valence quark-valence quark interaction
            LVA(NW)=1
            LVB(M)=1
            IQV=3
          ENDIF
          N=1
c LNH - number of new hard blocks (resulted from current nucleon-nucleon interaction)
          LNH=1
          GOTO 22
        ENDIF
c-------------------------------------------------

c LNH - number of new hard blocks - initialization
        LNH=0
c WH - probability to have semihard interaction
        WH=2.D0*FSHARD/VV
c N - number of cut pomerons (both soft ones and having hard blocks) for the
c nucleon-nucleon (hadron-nucleon) interaction - is determined according to Poisson
c with average value VV (twice the eikonal)
        DO 7 N=1,45
        EV=EV*VV/N
        SUM=SUM+EV
7       IF(AKS.LT.SUM)GOTO 8

c LNH - number of hard blocks for nucleon-nucleon (hadron-nucleon)
c interaction (according to WH probability)
8       DO 9 I=1,N
9       LNH=LNH+INT(WH+PSRAN(B10))

c-------------------------------------------------
        AKS1=.5D0*PSRAN(B10)
c EH is the probability to have valence quarks interactions in addition to the
c soft and semihard
        IF(AKS1.LT.EH)THEN
          IF(AKS1.LT.FHARD(1))THEN
            IF(LVA(NW).NE.0)GOTO 22
c Valence quark-gluon interaction
            LVA(NW)=1
            IQV=1
          ELSEIF(AKS1.LT.FHARD(1)+FHARD(2))THEN
            IF(LVB(M).NE.0)GOTO 22
c Gluon-valence quark interaction
            LVB(M)=1
            IQV=2
          ELSE
            IF(LVA(NW)+LVB(M).NE.0)GOTO 22
c Valence quark-valence quark interaction
            LVA(NW)=1
            LVB(M)=1
            IQV=3
          ENDIF
          N=N+1
          LNH=LNH+1
        ENDIF

22      IQS=1
        IF(LNH.NE.0)THEN
c-------------------------------------------------
c New hard blocks recording:
c LNH - number of new hard blocks,
c LHA(i) (LHB(j)) - total number of cut hard blocks, connected to i-th projectile
c (j-th target) nucleon (hadron);
c IAH(k) (IBH(k)) - number (position in array) of the projectile (target) nucleon,
c connected to k-th hard block;
c ZH(k) - factor exp(-R_ij/R_p) for k-th hard block;
c IQH(k) - type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
c-------------------------------------------------
c N - number of cut soft pomerons
          N=N-LNH
          LHA(NW)=LHA(NW)+LNH
          LHB(M)=LHB(M)+LNH
          DO 10 I=1,LNH
          I1=NHP+I
          If (I1 .ge. 4000) then
            write(moniou,*)'psconf: I1 > 4000, index out of bounds'
            stop
          endif
          IF(I.EQ.1.AND.IQV.NE.0)THEN
            IQH(I1)=IQV
          ELSE
            IQH(I1)=0
          ENDIF
        IF(DEBUG.GE.2)WRITE (MONIOU,209)I1,NW,M,IQH(I1)
209     FORMAT(2X,'PSCONF: ',I4,'-TH HARD BLOCK IS CONNECTED TO',1X,
     *  I2,'-TH PROJECTILE NUCLEON (HADRON) AND'/4X,I2,
     *  '-TH TARGET NUCLEON; TYPE OF THE SEMIHARD INTERACTION:',I1)
          ZH(I1)=Z
          IAH(I1)=NW
10        IBH(I1)=M
c-------------------------------------------------
c NHP - total number of hard blocks
          NHP=NHP+LNH
        ENDIF

c-------------------------------------------------
        IF(N.GT.0)THEN
c One more block of soft pomerons; soft block characteristics recording
          LS=LS+1
          IAS(LS)=NW
          IBS(LS)=M
          LQA(NW)=LQA(NW)+N
          LQB(M)=LQB(M)+N
          NQS(LS)=N
        IF(DEBUG.GE.2)WRITE (MONIOU,210)LS,NW,M,N
210     FORMAT(2X,'PSCONF: ',I4,'-TH SOFT BLOCK IS CONNECTED TO',1X,
     *  I2,'-TH PROJECTILE NUCLEON (HADRON) AND'/4X,I2,
     *  '-TH TARGET NUCLEON; NUMBER OF POMERONS IN THE BLOCK NP=',
     *  I2)
        ENDIF
11      CONTINUE
c-------------------------------------------------

        IF(IQS.NE.0)GOTO 14
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c Projectile diffraction
c For each projectile nucleon (hadron) diffractive dissociation probability is
c (1.D0-CC(ICZ))*PSV(X,Y,XB,NT);
c XXV(X,Y,XB,NT) - nucleon-nucleus scattering eikonal factor
c ( (1-eikonal)**2 ) for given nucleons positions
c (For projectile hadron only in case of JPERI=0, otherwise it was considered
c before at any impact parameter )
        IF(JDIFR.EQ.1 .AND. IA(1).NE. 1
     *  .AND.PSRAN(B10).LT.(1.D0-CC(ICZ))*PSV(X,Y,XB,NT))THEN
**************************************************
          IF(IA(2).NE.1)THEN
c ICT - partner target nucleon type (proton - 2 or neutron - 3)
            ICT=INT(2.5+PSRAN(B10))
          ELSE
c Target proton
            ICT=2
          ENDIF
c Projectile nucleon
          IF(DEBUG.GE.2)WRITE(MONIOU,207)IN
207       FORMAT(2X,I2,'-TH PROJECTILE NUCLEON DIFFRACTION')
          ICP0=INT(2.5+PSRAN(B10))
          WPI=WP0
          WMI=WM0
          IF(IA(2).EQ.1)THEN
            LQ=0
          ELSE
            LQ=1
          ENDIF
          CALL XXDPR(WPI,WMI,ICP0,ICT,LQ)
          GOTO 14
        ENDIF
**************************************************
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c No interaction for projectile nucleon considered
        NW=NW-1
12      CONTINUE

c One more projectile spectator (noninteracting) nucleon (spectator positions
c are recorded to simulate nuclear fragmentation)
        NS=NS+1
        IF(NS.NE.IN)THEN
          DO 13 L=1,3
13          XA(NS,L)=XA(IN,L)
        ENDIF
14      CONTINUE

c In case of no one interacting (or D-diffracted) nucleon the event is
c rejected, new impact parameter is generated and all the procedure is
c repeated
      IF(NS.EQ.IA(1))THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,211)
211     FORMAT(2X,'PSCONF: NO ONE NUCLEON (HADRON) INTERACTS - ',
     *  'REJECTION')
         GOTO 5
      ENDIF
c-------------------------------------------------
cdh   if(nhp.gt.150)then            ! changed 18. Feb. 04
      if(nhp.gt.1500)then
        WRITE (MONIOU,213)NHP
213     FORMAT(2X,'PSCONF: TOO GREAT NUMBER OF HARD POMERONS: NHP=',
     *  I5,' - REJECTION')
         GOTO 100
      endif

      NWT = 0
C  number of interacting target nucleons
      DO 102 IT = 1,NT
        NWT = NWT + IWT(IT)
 102  CONTINUE

cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c Fragmentation of the spectator part of the nucleus
      CALL XXFRAGM(NS,XA)
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

c Inelastic interaction - energy sharing procedure
        IF(NW.NE.0)CALL PSSHAR(LS,NHP,NW,NT)
21      continue                     !so00
        IF(DEBUG.GE.3)WRITE (MONIOU,212)
212     FORMAT(2X,'PSCONF - END')
        RETURN
        END
C=======================================================================

       SUBROUTINE QGSPSCS(C,S)
c C,S - COS and SIN generation for uniformly distributed angle 0<fi<2*pi
c-----------------------------------------------------------------------
       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
       COMMON /Q_AREA11/ B10
       COMMON /Q_AREA43/ MONIOU
       COMMON /Q_DEBUG/  DEBUG
       SAVE
       EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'QGSPSCS - COS(FI) AND SIN(FI) ARE GENERATED',
     *  ' (0<FI<2*PI)')
1      S1=2.D0*PSRAN(B10)-1.D0
       S2=2.D0*PSRAN(B10)-1.D0
       S3=S1*S1+S2*S2
       IF(S3.GT.1.D0)GOTO 1
       S3=DSQRT(S3)
       C=S1/S3
       S=S2/S3
        IF(DEBUG.GE.3)WRITE (MONIOU,202)C,S
202     FORMAT(2X,'QGSPSCS: C=',E10.3,2X,'S=',E10.3)
       RETURN
       END
C=======================================================================

        SUBROUTINE QGSPSDEFTR(S,EP,EY)
c Determination of the parameters for the Lorentz transform to the rest frame
c system for 4-vector EP
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EY(3),EP(4)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)EP,S
201     FORMAT(2X,'QGSPSDEFTR - LORENTZ BOOST PARAMETERS:'/
     *  4X,'4-VECTOR EP=',4E10.3/4X,'4-VECTOR SQUARED S=',E10.3)
        DO 2 I=1,3
        IF(EP(I+1).EQ.0.D0)THEN
          EY(I)=1.D0
        ELSE
            WP=EP(1)+EP(I+1)
          WM=EP(1)-EP(I+1)
          IF(WM/WP.LT.1.D-8)THEN
            WW=S
            DO 1 L=1,3
1            IF(L.NE.I)WW=WW+EP(L+1)**2
            WM=WW/WP
          ENDIF
          EY(I)=DSQRT(WM/WP)
          EP(1)=WP*EY(I)
          EP(I+1)=0.D0
        ENDIF
2       CONTINUE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)EY
202     FORMAT(2X,'QGSPSDEFTR: LORENTZ BOOST PARAMETERS EY(I)=',2X
     *  ,3E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE QGSPSDEFROT(EP,S0X,C0X,S0,C0)
c Determination of the parameters the spacial rotation to the lab. system
c for 4-vector EP
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)EP
201     FORMAT(2X,'QGSPSDEFROT - SPACIAL ROTATION PARAMETERS'/4X,
     *  '4-VECTOR EP=',2X,4(E10.3,1X))
c Transverse momentum square for the current parton (EP)
        PT2=EP(3)**2+EP(4)**2
        IF(PT2.NE.0.D0)THEN
          PT=DSQRT(PT2)
c System rotation to get Pt=0 - Euler angles are determined (C0X = cos theta,
c S0X = sin theta, C0 = cos phi, S0 = sin phi)
          C0X=EP(3)/PT
          S0X=EP(4)/PT
c Total momentum for the gluon
          PL=DSQRT(PT2+EP(2)**2)
          S0=PT/PL
          C0=EP(2)/PL
        ELSE
          C0X=1.D0
          S0X=0.D0
          PL=ABS(EP(2))
          S0=0.D0
          C0=EP(2)/PL
        ENDIF

        EP(2)=PL
        EP(3)=0.D0
        EP(4)=0.D0
        IF(DEBUG.GE.3)WRITE (MONIOU,202)S0X,C0X,S0,C0,EP
202     FORMAT(2X,'QGSPSDEFROT: SPACIAL ROTATION PARAMETERS'/
     *  4X,'S0X=',E10.3,2X,'C0X=',E10.3,2X,'S0=',E10.3,2X,'C0=',E10.3/
     *  4X,'ROTATED 4-VECTOR EP=',4(E10.3,1X))
        RETURN
        END
C=======================================================================

        FUNCTION PSDR(X,Y)
c PSDR - impact parameter factor for eikonals calculation (exp(-Rij/Rp)=Z)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA7/  RP
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)X,Y
201     FORMAT(2X,'PSDR: NUCLEON COORDINATES - X=',E10.3,2X,'Y=',E10.3)
        PSDR=EXP(-(X*X+Y*Y)/RP)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSDR
202     FORMAT(2X,'PSDR=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSFAP(X,J,L)
C QGSPSFAP - Altarelli-Parisi function (multiplied by X)
c X - light cone momentum share value,
c J - type of the parent parton (0-g,1-q)
c L - type of the daughter parton (0-g,1-q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)X,J,L
201     FORMAT(2X,'QGSPSFAP - ALTARELLI-PARISI FUNCTION:',2X,
     *  'X=',E10.3,2X,'J=',I1,2X,'L=',I1)
        IF(J.EQ.0)THEN
          IF(L.EQ.0)THEN
            QGSPSFAP=((1.D0-X)/X+X/(1.D0-X)+X*(1.D0-X))*6.d0
          ELSE
            QGSPSFAP=(X**2+(1.D0-X)**2)*3.d0
          ENDIF
        ELSE
          IF(l.EQ.0)THEN
            QGSPSFAP=(1.D0+(1.D0-X)**2)/X/.75D0
          ELSE
            QGSPSFAP=(X**2+1.D0)/(1.D0-X)/.75D0
          ENDIF
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSFAP
202     FORMAT(2X,'QGSPSFAP=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSFAZ(Z,FSOFT,FHARD,FSHARD)
c Interaction eikonal for hadron-nucleon (nucleon-nucleon) scattering
c Z - impact parameter factor, Z=exp(-b**2/Rp),
c FSOFT - soft pomeron eikonal - to be determined,
c FSHARD - semihard interaction eikonal (gg) - to be determined,
c FHARD(k) - hard interaction eikonal (k=1 - qg, 2 - gq, 3 - qq) -
c to be determined,
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION FHARD(3)
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA22/ SJV,FJS(5,3)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)Z
201     FORMAT(2X,'PSFAZ - HADRON-NUCLEON (NUCLEON-NUCLEON)',
     *  ' INTERACTION EIKONAL; Z=',E10.3)
        FSOFT=FS*Z
        FHARD(3)=SJV*Z**(RS/RS0)

        JZ=INT(5.D0*Z)
        IF(JZ.GT.3)JZ=3
        WZ=5.D0*Z-JZ

        DO 1 I=1,3
        IF(JZ.EQ.0)THEN
          FSR=(EXP(FJS(1,I))*WZ+(EXP(FJS(2,I))-2.D0*
     *    EXP(FJS(1,I)))*WZ*(WZ-1.D0)*.5D0)*Z
        ELSE
          FSR=EXP(FJS(JZ,I)+(FJS(JZ+1,I)-FJS(JZ,I))*WZ
     *    +(FJS(JZ+2,I)+FJS(JZ,I)-2.D0*FJS(JZ+1,I))
     *    *WZ*(WZ-1.D0)*.5D0)*Z
        ENDIF
        IF(I.NE.1)THEN
          FHARD(I-1)=FSR
        ELSE
          FSHARD=FSR
        ENDIF
1       CONTINUE

        PSFAZ=FSOFT+FSHARD
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSFAZ,FSOFT,FSHARD,FHARD
202     FORMAT(2X,'PSFAZ=',E10.3,2X,'FSOFT=',E10.3,2X,'FSHARD=',E10.3/4X
     *    ,'FHARD=',3E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSFBORN(S,T,IQ1,IQ2)
c PSFBORN - integrand for the Born cross-section (matrix element squared)
c S - total c.m. energy squared for the scattering,
c T - invariant variable for the scattering abs[(p1-p3)**2],
c IQ1 - parton type at current end of the ladder (0 - g, 1,2 - q)
c IQ2 - parton type at opposite end of the ladder (0 - g, 1,2 - q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,T,IQ1,IQ2
201     FORMAT(2X,'PSFBORN - HARD SCATTERING MATRIX ELEMENT SQUARED:'/
     *  4X,'S=',E10.3,2X,'|T|=',E10.3,2X,'IQ1=',I2,2X,'IQ2=',I2)
        U=S-T
        IF(IQ1.EQ.0.AND.IQ2.EQ.0)THEN
c Gluon-gluon
          PSFBORN=(3.D0-T*U/S**2+S*U/T**2+S*T/U**2)*4.5D0
        ELSEIF(IQ1*IQ2.EQ.0)THEN
c Gluon-quark
          PSFBORN=(S**2+U**2)/T**2+(S/U+U/S)/2.25D0
        ELSEIF(IQ1.EQ.IQ2)THEN
c Quark-quark (of the same flavor)
          PSFBORN=((S**2+U**2)/T**2+(S**2+T**2)/U**2)/2.25D0
     *          -S**2/T/U/3.375D0
        ELSEIF(IQ1+IQ2.EQ.0)THEN
c Quark-antiquark (of the same flavor)
          PSFBORN=((S**2+U**2)/T**2+(U**2+T**2)/S**2)/2.25D0
     *          -U**2/T/S/3.375D0
        ELSE
c Quark-quark (different flavors)
          PSFBORN=(S**2+U**2)/T**2/2.25D0
        ENDIF
        IF(DEBUG.GE.2)WRITE (MONIOU,202)PSFBORN
202     FORMAT(2X,'PSFBORN=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSFSH(S,Z,ICZ,IQQ)
c QGSPSFSH - semihard interaction eikonal
c S - energy squared for the interaction (hadron-hadron),
c ICZ - type of the primaty hadron (nucleon)
c Z - impact parameter factor, Z=exp(-b**2/Rp),
c IQQ - type of the hard interaction (0 - gg, 1 - qg, 2 - gq)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA6/  PI,BM,AM
        COMMON /Q_AREA15/ FP(5),RQ(5),CD(5)
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA19/ AHL(5)
        COMMON /Q_AREA25/ AHV(5)
        COMMON /Q_AREA27/ FP0(5)
        COMMON /Q_AR13/    X1(7),A1(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,Z,IQQ,ICZ
201     FORMAT(2X,'QGSPSFSH - SEMIHARD INTERACTION EIKONAL:'/
     *  4X,'S=',E10.3,2X,'Z=',E10.3,2X,'IQQ=',I1,2X,'ICZ=',I1)
        XMIN=4.D0*QT0/S
        XMIN=XMIN**(DELH-DEL)
        QGSPSFSH=0.D0
        IF(IQQ.EQ.1)THEN
          ICV=ICZ
          ICQ=2
        ELSEIF(IQQ.EQ.2)THEN
          ICV=2
          ICQ=ICZ
        ENDIF
        IQ=(IQQ+1)/2

c Numerical integration over Z1
        DO 3 I=1,7
        DO 3 M=1,2
        Z1=(.5D0*(1.D0+XMIN-(2*M-3)*X1(I)*(1.D0-XMIN)))**(1.D0/
     *  (DELH-DEL))
c SJ is the DLA inclusive hard partonic (gluon-gluon) interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer squre QT0 and c.m. energy square s_hard = exp YJ;
c SJB - Born cross-section
        CALL PSJINT0(Z1*S,SJ,SJB,IQ,0)
c GY= Sigma_hard_tot(YJ,QT0) - total hard partonic (gluon-gluon)
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s_hard = exp YJ; SH=pi*R_hard**2 (R_hard**2=4/QT0)
        GY=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB
        IF(DEBUG.GE.3)WRITE (MONIOU,203)Z1*S,GY
203     FORMAT(2X,'QGSPSFSH:',2X,'S_HARD=',E10.3,2X,'SIGMA_HARD=',E10.3)

        IF(IQQ.EQ.0)THEN
          ST2=0.D0
          DO 1 J=1,7
          DO 1 K=1,2
          XX=.5D0*(1.D0+X1(J)*(2*K-3))
1         ST2=ST2+A1(J)*QGSPSFTILD(Z1**XX,ICZ)*
     *    QGSPSFTILD(Z1**(1.D0-XX),2)

          RH=RS0-ALF*DLOG(Z1)
          QGSPSFSH=QGSPSFSH-A1(I)*DLOG(Z1)*GY/Z1**DELH*Z**(RS/RH)/RH*ST2
        ELSE

          ST2=0.D0
          DO 2 J=1,7
          DO 2 K=1,2
          XX=.5D0*(1.D0+X1(J)*(2*K-3))
          XAM=Z1**(DEL+.5D0)
          XA=(XAM+(1.D0-XAM)*XX)**(1.D0/(DEL+.5D0))
          RH=RS0+ALF*DLOG(XA/Z1)
2         ST2=ST2+A1(J)*(1.D0-XA)**AHV(ICV)*Z**(RS/RH)/RH*
     *    QGSPSFTILD(Z1/XA,ICQ)
          ST2=ST2*(1.D0-XAM)

          QGSPSFSH=QGSPSFSH+A1(I)*GY/Z1**DELH*ST2
        ENDIF
3       CONTINUE

        IF(IQQ.EQ.0)THEN
          QGSPSFSH=QGSPSFSH*.125D0*RR*(1.D0-XMIN)/(DELH-DEL)*FP0(ICZ)
     *                     *FP0(2)
     *    *CD(ICZ)
        ELSE
          QGSPSFSH=QGSPSFSH*DSQRT(RR)/16.D0*FP0(ICQ)*(1.D0-XMIN)
     *    /(DELH-DEL)/(DEL+.5D0)*GAMFUN(AHV(ICV)+1.5D0)
     *    /GAMFUN(AHV(ICV)+1.D0)/PI*CD(ICZ)
          IF(ICZ.EQ.2.OR.IQQ.EQ.2)THEN
            QGSPSFSH=QGSPSFSH*3.D0
          ELSEIF((ICZ-1)*(ICZ-3)*(ICZ-5).EQ.0)THEN
            QGSPSFSH=QGSPSFSH*2.D0
          ENDIF
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSFSH
202     FORMAT(2X,'QGSPSFSH=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSFTILD(Z,ICZ)
c QGSPSFTILD - auxilliary function for semihard eikonals calculation -
c integration over semihard block light cone momentum share x
c Z - x-cutoff from below,
c ICZ - type of the hadron to which the semihard block is connected
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA19/ AHL(5)
        COMMON /Q_AR13/  X1(7),A1(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)Z,ICZ
201     FORMAT(2X,'QGSPSFTILD:',2X,'Z=',E10.3,2X,'ICZ=',I1)
        QGSPSFTILD=0.
        DO 1 I=1,7
        DO 1 M=1,2
        XB=1.D0-(1.D0-Z)*(.5D0*(1.D0+(2*M-3)*X1(I)))**(1.D0/
     *  (AHL(ICZ)+1.D0))
1       QGSPSFTILD=QGSPSFTILD+A1(I)*XB**DEL*(1.D0-Z/XB)**BET
        QGSPSFTILD=QGSPSFTILD*.5D0*(1.D0-Z)**(AHL(ICZ)+1.D0)
     *             /(AHL(ICZ)+1.D0)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSFTILD
202     FORMAT(2X,'QGSPSFTILD=',E10.3)
        RETURN
        END
C=======================================================================

      SUBROUTINE PSGEA(IA,XA,JJ)
c PSGEA - nuclear configuration simulation (nucleons positions)
c IA - number of nucleons to be considered
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
cdh   DIMENSION XA(56,3)
      DIMENSION XA(210,3)
      COMMON /Q_AREA5/  RD(2),CA1(2),CA2(2),CA3(2)
      COMMON /Q_AREA11/ B10
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE
      EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)JJ,IA
201     FORMAT(2X,'PSGEA - CONFIGURATION OF THE NUCLEUS ',I1,';',2X,
     *  'COORDINATES FOR ',I2,' NUCLEONS')
cdh     IF(JJ.EQ.2.OR.IA.GE.10)THEN
        IF(IA.GE.10)THEN !this line had been changed!!!!!!! dh 8/10/98
cdh
          DO 7 I=1,IA
1         ZUK=PSRAN(B10)*CA1(JJ)-1.D0
          IF(ZUK)2,2,3
2         TT=RD(JJ)*(PSRAN(B10)**.3333D0-1.D0)
          GOTO 6
3         IF(ZUK.GT.CA2(JJ))GOTO 4
          TT=-DLOG(PSRAN(B10))
          GOTO 6
4         IF(ZUK.GT.CA3(JJ))GOTO 5
          TT=-DLOG(PSRAN(B10))-DLOG(PSRAN(B10))
          GOTO 6
5         TT=-DLOG(PSRAN(B10))-DLOG(PSRAN(B10))-DLOG(PSRAN(B10))
6         IF(PSRAN(B10).GT.1.D0/(1.D0+EXP(-ABS(TT))))GOTO 1
          RIM=TT+RD(JJ)
          Z=RIM*(2.D0*PSRAN(B10)-1.D0)
          RIM=DSQRT(RIM*RIM-Z*Z)
          XA(I,3)=Z
          CALL QGSPSCS(C,S)
          XA(I,1)=RIM*C
7         XA(I,2)=RIM*S
        ELSE

          DO 9 L=1,3
          SUMM=0.D0
          DO 8 I=1,IA-1
          J=IA-I
          AKS=RD(JJ)*(PSRAN(B10)+PSRAN(B10)+PSRAN(B10)-1.5D0)
          K=J+1
          XA(K,L)=SUMM-AKS*SQRT(FLOAT(J)/K)
8         SUMM=SUMM+AKS/SQRT(FLOAT(J*K))
9         XA(1,L)=SUMM
        ENDIF
        IF(DEBUG.GE.3)THEN
          WRITE (MONIOU,203)
          DO 206 I=1,IA
206       WRITE (MONIOU,204)I,(XA(I,L),L=1,3)
          WRITE (MONIOU,202)
        ENDIF
202     FORMAT(2X,'PSGEA - END')
203     FORMAT(2X,'PSGEA:  POSITIONS OF THE NUCLEONS')
204     FORMAT(2X,'PSGEA: ',I2,' - ',3(E10.3,1X))
        RETURN
        END
C=======================================================================

        FUNCTION PSGINT(Z)
c Auxiliary function for eikonal cross-sections calculation
c GINT = int(dt) [0<t<Z] (1-exp(-t))/t
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      COMMON /Q_AR13/  X1(7),A1(7)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE

        F(Z,X)=(1.-EXP(-.5*Z*(1.+X)))/(1.+X)

        IF(DEBUG.GE.2)WRITE (MONIOU,201)Z
201     FORMAT(2X,'PSGINT:',2X,'Z=',E10.3)
        PSGINT=0.
        DO 5 I=1,7
5       PSGINT=PSGINT+A1(I)*(F(Z,X1(I))+F(Z,-X1(I)))
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSGINT
202     FORMAT(2X,'PSGINT=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSHARD(S,ICZ)
c QGSPSHARD - hard quark-quark interaction cross-section
c S - energy squared for the interaction (hadron-hadron),
c ICZ - type of the primaty hadron (nucleon)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AR13/    X1(7),A1(7)
        COMMON /Q_AREA6/  PI,BM,AM
        COMMON /Q_AREA15/ FP(5),RQ(5),CD(5)
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA19/ AHL(5)
        COMMON /Q_AREA25/ AHV(5)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,ICZ
201     FORMAT(2X,'QGSPSHARD - HARD QUARK-QUARK INTERACTION CROSS',
     *  ' SECTION:',
     *  2X,'S=',E10.3,2X,'ICZ=',I1)
        XMIN=4.D0*QT0/S
        XMIN=XMIN**(DELH+.5D0)
        QGSPSHARD=0.D0

c Numerical integration over Z1
        DO 2 I=1,7
        DO 2 M=1,2
        Z1=(.5D0*(1.D0+XMIN-(2*M-3)*X1(I)*(1.D0-XMIN)))**(1.D0/
     *  (DELH+.5D0))

        ST2=0.D0
        DO 1 J=1,7
        DO 1 K=1,2
        XX=.5D0*(1.D0+X1(J)*(2*K-3))
        ST2=ST2+A1(J)*(1.D0-Z1**XX)**AHV(ICZ)*
     *  (1.D0-Z1**(1.D0-XX))**AHV(2)
1       CONTINUE

c SJ is the DLA inclusive hard partonic (gluon-gluon) interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer squre QT0 and c.m. energy square s_hard = exp YJ;
c SJB - Born cross-section
        CALL PSJINT0(Z1*S,SJ,SJB,1,1)
c GY= Sigma_hard_tot(YJ,QT0) - total hard partonic (quark-quark)
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s_hard = exp YJ; SH=pi*R_hard**2 (R_hard**2=4/QT0)
        GY=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB

        IF(DEBUG.GE.3)WRITE (MONIOU,203)Z1*S,GY
203    FORMAT(2X,'QGSPSHARD:',2X,'S_HARD=',E10.3,2X,'SIGMA_HARD=',E10.3)
        QGSPSHARD=QGSPSHARD-A1(I)*DLOG(Z1)*GY/Z1**DELH*ST2
2       CONTINUE

        QGSPSHARD=QGSPSHARD*(1.D0-XMIN)/(.5D0+DELH)*.25D0
        QGSPSHARD=QGSPSHARD/(GAMFUN(AHV(ICZ)+1.D0)*GAMFUN(AHV(2)+1.D0)
     *  *PI)*GAMFUN(AHV(ICZ)+1.5D0)*GAMFUN(AHV(2)+1.5D0)

        IF(ICZ.EQ.2)THEN
          QGSPSHARD=QGSPSHARD*9.D0
        ELSEIF((ICZ-1)*(ICZ-3)*(ICZ-5).EQ.0)THEN
          QGSPSHARD=QGSPSHARD*6.D0
        ELSE
          QGSPSHARD=QGSPSHARD*3.D0
        ENDIF

c Hard cross-section is divided by Regge radius RS0 and multiplied by
c shower enhancement coefficient CD(ICZ) - to be used for the eikonal
c calculation
        QGSPSHARD=QGSPSHARD/(8.D0*PI*RS0)*CD(ICZ)
        IF(DEBUG.GE.2)WRITE (MONIOU,202)QGSPSHARD
202     FORMAT(2X,'QGSPSHARD=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE PSHOT(WP0,WM0,Z,IPC,EPC,IZP,IZT,ICZ,IQQ)
c Semihard jets production simulation (resulted from parton-parton
c interaction);
c WP0,WM0 - light cone momenta shares (E+-P_l) for the initial partons
c IZP, IZT - types for target and projectile nucleons (hadron)
c WPQ - light cone momenta for the soft preevolution - to be determined below
c IQQ - type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        CHARACTER*2 TYQ
        DIMENSION EPT(4),EP3(4),EPJ(4),EPJ1(4),EY(3),
     *  QMIN(2),WP(2),IQC(2),IQP(2),
     *  IPC(2,2),EPC(8,2),IQJ(2),EQJ(4,2),IPQ(2,2),EPQ(8,2),
     *  ebal(4),
     *  QV1(30,50),ZV1(30,50),QM1(30,50),IQV1(30,50),
     *  LDAU1(30,49),LPAR1(30,50),
     *  QV2(30,50),ZV2(30,50),QM2(30,50),IQV2(30,50),
     *  LDAU2(30,49),LPAR2(30,50)!,EP(4,2),EPT0(4)
        COMMON /Q_AREA6/  PI,BM,AMMM
        COMMON /Q_AREA8/  WWM,BE(4),DC(5),DETA,ALMPT
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA42/ TYQ(15)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_AREA46/ EPJET(4,2,15000),IPJET(2,15000)
        COMMON /Q_AREA47/ NJTOT
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.1)WRITE (MONIOU,201)IQQ,WP0,WM0
201     FORMAT(2X,'PSHOT - SEMIHARD INTERACTION SIMULATION:'/
     *  4X,'TYPE OF THE INTERACTION:',I2/
     *  4X,'INITIAL LIGHT CONE MOMENTA:',2E10.3)
c S - total energy squared for the semihard interaction (including preevolution)
        NJTOT0=NJTOT
        IZP0=IZP
        IZT0=IZT

301     S=WP0*WM0
        NJTOT=NJTOT0
        IZP=IZP0
        IZT=IZT0

        IF(IQQ.EQ.3)THEN
c WPI,WMI - light cone momenta for the hard interaction
          WPI=WP0
          WMI=WM0
c PSJINT0(S,SJ,SJB,1,1) - cross-sections interpolation:
c SJ - inclusive hard quark-quark interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer square QT0 and c.m. energy square s_hard = S;
c SJB - Born cross-section
          CALL PSJINT0(S,SJ,SJB,1,1)
c GY= Sigma_hard_tot(YJ,QT0) - total hard quark-quark
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s_hard = S
          GY=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB

        ELSE
c-------------------------------------------------
c Rejection function normalization
c-------------------------------------------------
c XMIN corresponds to minimal energy squared for the hard interaction - 4.D0*(QT0+AMJ0)
c AMJ0 - jet mass squared (could be put equal zero)
          XMIN=4.D0*(QT0+AMJ0)/S
          XMIN1=XMIN**(DELH-DEL)
c S - maximal available energy for the rejection function normalization
c Auxilliary type of parton (1 - gluon, 2 - (anti-)quark)
          IQ=(IQQ+1)/2
c Rejection function initialization (corresponding to maximal preevolution - minimal x):
c Ysoft = - ln x, (1-x)**bet is due to gluon structure function in the soft pomeron
          IF(IQQ.EQ.0)THEN
             GB0=-DLOG(XMIN)*(1.D0-DSQRT(XMIN))**(2.D0*BET)
          ELSE
             GB0=(1.D0-XMIN)**BET
          ENDIF

c SJ0 is the inclusive hard (parton IQ - gluon) interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer square QT0 and c.m. energy square s_hard = SI;
c SJB0 - Born cross-section
          CALL PSJINT0(S,SJ,SJB,IQ,0)
c GY= Sigma_hard_tot(YJ,QT0) - total hard  (parton IQ - gluon)
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s_hard = SI
          GY0=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB
          GB0=GB0*GY0/S**DELH/RS0*Z
c-------------------------------------------------
c End of rejection function normalization
c-------------------------------------------------

c-------------------------------------------------
c The sharing of the light cone momenta between soft preevolution and
c hard interaction:
c ( first energy-momentum is shared according to
c f_hard(YJ)~ZPM**(DELH-DEL-1) and then rejected as
c W_rej ~Sigma_hard_tot(YJ) / exp(DELH*YJ)
c ZPM = s_hard / S
c YJ = ln s_hard - rapidity range for the hard parton-parton interaction;
c-------------------------------------------------
1         ZPM=(XMIN1+PSRAN(B10)*(1.D0-XMIN1))**(1.D0/(DELH-DEL))
c SJ is the DLA inclusive hard partonic (gluon-gluon) interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer square QT0 and c.m. energy square s_hard = exp YJ;
c SJB - Born cross-section
          CALL PSJINT0(ZPM*S,SJ,SJB,IQ,0)
          YJ=DLOG(ZPM*S)
c RH - interaction radius due to soft preevolution
          RH=RS0-ALF*DLOG(ZPM)

          IF(IQQ.EQ.0)THEN
c XP, XM - light cone momunta shares for the hard interaction
            XP=ZPM**PSRAN(B10)
            XM=ZPM/XP
c Ysoft = - ln ZPM - part of rejection function,
c (1-XP)**bet*(1-XM)**bet is due to gluon structure function in the soft pomeron
            GBYJ=-DLOG(ZPM)*((1.-XP)*(1.-XM))**BET
c WPI,WMI - light cone momenta for the hard interaction
            WPI=WP0*XP
            WMI=WM0*XM
          ELSE
            IF(IQQ.EQ.1)THEN
              WPI=WP0
              WMI=WM0*ZPM
            ELSE
              WPI=WP0*ZPM
              WMI=WM0
            ENDIF
            GBYJ=(1.D0-ZPM)**BET
          ENDIF

c GY= Sigma_hard_tot(YJ,QT0) - total hard partonic
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s_hard = exp YJ
          GY=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB

c-------------------------------------------------
c GBYJ - rejection function for the YJ (ZPM) simulation:
c GBYJ ~  Sigma_hard_tot(YJ,QT0) / exp(DELH*YJ) * exp(-b**2/RH) / RH,
          GBYJ=GBYJ*GY*EXP(-DELH*YJ)/GB0*Z**(RS/RH)/RH
          IF(PSRAN(B10).GT.GBYJ)GOTO 1
        ENDIF
c-------------------------------------------------
        S=WPI*WMI

        IF(DEBUG.GE.2)WRITE (MONIOU,203)S
203     FORMAT(2X,'PSHOT: MASS SQUARED FOR THE HARD PARTON-PARTON',
     *  ' INTERACTION:',E10.3)

c In case of valence quark hard interaction the type of quark is determined by the
c procedure PSVDEF - flavor combinatorics (not good here); IQC(1) - flavor
c for the upper quark (0 in case of gluon),
c IQC(2) - the same for the lower one
        DO 302 I=1,8
        DO 302 M=1,2
302     EPC(I,M)=0.D0

        IF((IQQ-1)*(IQQ-3).EQ.0)THEN
          CALL PSVDEF(IZP,IC1,ICZ)
          IQC(1)=IC1
          IPC(1,1)=0
          IPC(2,1)=0
        ELSE
          IQC(1)=0
          IPC(1,1)=-INT(2.D0*PSRAN(B10)+1.D0)
          IPC(2,1)=-IPC(1,1)
          WP1=WP0-WPI
          WP2=WP1*PSRAN(B10)
          WP1=WP1-WP2
          EPC(1,1)=.5D0*WP1
          EPC(2,1)=EPC(1,1)
          EPC(5,1)=.5D0*WP2
          EPC(6,1)=EPC(5,1)
               ENDIF

        IF((IQQ-2)*(IQQ-3).EQ.0)THEN
          CALL PSVDEF(IZT,IC1,2)
          IQC(2)=IC1
          IPC(1,2)=0
          IPC(2,2)=0
        ELSE
          IQC(2)=0
          IPC(1,2)=-INT(2.D0*PSRAN(B10)+1.D0)
          IPC(2,2)=-IPC(1,2)
          WM1=WM0-WMI
          WM2=WM1*PSRAN(B10)
          WM1=WM1-WM2
          EPC(1,2)=.5D0*WM1
          EPC(2,2)=-EPC(1,2)
          EPC(5,2)=.5D0*WM2
          EPC(6,2)=-EPC(5,2)
        ENDIF

        EPT(1)=.5D0*(WPI+WMI)
        EPT(2)=.5D0*(WPI-WMI)
        EPT(3)=0.D0
        EPT(4)=0.D0
c Minimal 4-momentum transfer squares above and below current ladder run
        QMIN(1)=QT0
        QMIN(2)=QT0
        DO 303 L=1,2
        DO 303 M=1,2
              IPQ(L,M)=IPC(L,M)
        DO 303 I=1,4
303     EPQ(I+4*(L-1),M)=EPC(I+4*(L-1),M)
c Minimal 4-momentum transfer square for gluon-gluon (virtual) interaction
          QMINN=MAX(QMIN(1),QMIN(2))
          SI=QGSPSNORM(EPT)

5         CONTINUE
c 4-momentum squared (c.m. energy square for gluon-gluon (virtual)
c interaction)
        IF(DEBUG.GE.2)WRITE (MONIOU,208)ILAD, SI,IQC,EPT
208     FORMAT(2X,'PSHOT: ',I2,'-TH HARD LADDER;',
     *  ' MASS SQUARED FOR THE LADDDER:',E10.3/
     *  4X,'LADDER END FLAVORS:',2I3/4X,
     *  'LADDER 4-MOMENTUM: ',4E10.3)

        ebal(1)=.5*(wp0+wm0)-ept(1)
        ebal(2)=.5*(wp0-wm0)-ept(2)
        ebal(3)=0.d0-ept(3)
        ebal(4)=0.d0-ept(4)
        do 503 l=1,4
        do 501 m=1,2
        ebal(l)=ebal(l)-epq(l,m)
501     if(iqc(m).eq.0)   ebal(l)=ebal(l)-epq(l+4,m)
        if(njtot.ne.0)then
           do 502 i=1,njtot
           do 502 m=1,2
502        ebal(l)=ebal(l)-epjet(l,m,i)
        endif
503        continue
c            write (*,*)'ebal',ebal,si,njtot

          PT2=EPT(3)**2+EPT(4)**2
          PT=DSQRT(PT2)
          WW=SI+PT2
          SWW=DSQRT(WW)

          IQP(1)=MIN(1,IABS(IQC(1)))
          IQP(2)=MIN(1,IABS(IQC(2)))

c Longitudinal momenta for the interaction
          WP(1)=EPT(1)+EPT(2)
          WP(2)=EPT(1)-EPT(2)

          S2MIN=MAX(QMINN,4.D0*(QT0+AMJ0))
c WWMIN is the minimal energy square needed for triple s-channel gluons
c production with transverse momentum squares q_t**2 above QMIN(JJ),QMINN
          WWMIN=(S2MIN+(PT-DSQRT(QT0))**2+(QT0+AMJ0)*(DSQRT(S2MIN/QT0)-
     *          1.D0))/(1.D0-DSQRT(QT0/S2MIN))
c SJB/SJ is the probability for the last pair of gluons production
c (SJB is the Born cross-section and SJ is the inclusive interaction
c (cut ladder) cross-section)
          SJ=PSJINT(QMIN(1),QMIN(2),SI,IQP(1)+1,IQP(2)+1)
          SJB=QGSPSBINT(QMINN,SI,IQP(1)+1,IQP(2)+1)

        IF(DEBUG.GE.2)WRITE (MONIOU,251)S2MIN,WWMIN,SJ,SJB
251     FORMAT(2X,'PSHOT: KINEMATICAL BOUNDS S2MIN=',E10.3,
     *   2X,'WWMIN=',E10.3/4X,'JET CROSS SETION SJ=',E10.3,
     *   2X,'BORN CROSS SECTION SJB=',E10.3)

          IF(PSRAN(B10).LT.SJB/SJ.
     *          OR.WW.LT.1.2D0*WWMIN)GOTO 12

          IF((SJ-SJB)/SJ.GT..1D0)THEN
            SJ1=PSJINT1(QMIN(1),QMIN(2),SI,IQP(1)+1,IQP(2)+1)
            SJ2=PSJINT1(QMIN(2),QMIN(1),SI,IQP(2)+1,IQP(1)+1)
            DSJ=(SJ2-SJ1)/(SJ-SJB)*.5D0
          ELSE
            DSJ=0.D0
          ENDIF
c Current s-channel gluon is simulated either above the run (JJ=1) or
c below it (JJ=2)
          JJ=INT(1.5D0+DSJ+PSRAN(B10))

          AQ=-(SI+AMJ0+2.D0*PT*DSQRT(QT0))/WW
          BQ=(QT0+AMJ0)/WW
          CQ=QT0/WW
          PQ=-AQ**2/3.D0+BQ
          QQ=AQ**3/13.5D0-AQ*BQ/3.D0+CQ
          PQ=DSQRT(-PQ/3.D0)
          COSQ=-.5D0*QQ/PQ**3
          FQ=ATAN(1.D0/COSQ**2-1.D0)
          IF(COSQ.LT.0.D0)FQ=PI-FQ
          FQ=FQ/3.D0

c XMIN is the minimal longitudinal momentum transfer share in current
c ladder run (corresponding to minimal 4-momentum transfer square QMIN(JJ))
          XMIN=1.D0+AQ/3.D0-2.D0*PQ*COS(FQ)
          XMAX=1.D0+AQ/3.D0-PQ*(DSQRT(3.D0)*SIN(FQ)-COS(FQ))
c QQMAX is the maximal 4-momentum transfer square in the current run
c (corresponding to X=XMIN and 4-momentum transfer at next simulation
c step to be equal QMAX)
          QQMAX=QT0/(1.D0-XMAX)**2
          QQMIN=QT0/(1.D0-XMIN)**2

          IF(QQMIN.LT.S2MIN)THEN
            XMM=(SI-S2MIN+AMJ0+2.D0*PT*DSQRT(QT0))/WW*.5D0
            XMIN=1.D0-XMM-DSQRT(XMM*XMM-(QT0+AMJ0)/WW)
            QQMIN=QT0/(1.D0-XMIN)**2

            IF(QQMIN.LT.QMIN(JJ))THEN
              QQMIN=QMIN(JJ)
              XMM1=WW-2.D0*PT*DSQRT(QQMIN)+QQMIN
              XMM=(SI-S2MIN+AMJ0)/XMM1*.5D0
              XMIN=1.D0-XMM-DSQRT(XMM*XMM-AMJ0/XMM1)
            ENDIF
          ENDIF

*********************************************************
          XM0=MAX(.5D0,1.D0-DSQRT(QT0/QMIN(JJ)))
          IF(XM0.GT..95D0*XMAX.OR.XM0.LT.1.05D0*XMIN)
     *    XM0=.5D0*(XMAX+XMIN)
          QM0=QT0/(1.D0-XM0)**2
          S2MAX=XM0*WW

          SJ0=PSJINT(QM0,QMIN(3-JJ),S2MAX,1,IQP(3-JJ)+1)*
     *    QGSPSFAP(XM0,IQP(JJ),0)+
     *    PSJINT(QM0,QMIN(3-JJ),S2MAX,2,IQP(3-JJ)+1)
     *    *QGSPSFAP(XM0,IQP(JJ),1)

          GB0=SJ0*QM0/QLOG*QGSPSUDS(QM0,IQP(JJ))*1.5D0
          IF(XM0.LE..5D0)THEN
            GB0=GB0*XM0**(1.D0-DELH)
          ELSE
            GB0=GB0*(1.D0-XM0)*2.D0**DELH
          ENDIF
c XMIN, XMAX are put into power DELH to simulate X value below
          XMIN2=MAX(.5D0,XMIN)
          XMIN1=XMIN**DELH
          XMAX1=MIN(XMAX,.5D0)**DELH
          IF(XMIN.GE..5D0)THEN
            DJL=1.D0
          ELSEIF(XMAX.LT..5D0)THEN
            DJL=0.D0
          ELSE
            DJL=1.D0/(1.D0+((2.D0*XMIN)**DELH-1.D0)/DELH/
     *      DLOG(2.D0*(1.D0-XMAX)))
          ENDIF

7         CONTINUE
c Simulation of the longitudinal momentum transfer share in current
c ladder run - from XMIN to XMAX according to dX * X**(DELH-1)
          IF(PSRAN(B10).GT.DJL)THEN
            X=(XMIN1+PSRAN(B10)*(XMAX1-XMIN1))**(1.D0/DELH)
          ELSE
            X=1.D0-(1.D0-XMIN2)*((1.D0-XMAX)/(1.D0-XMIN2))**PSRAN(B10)
          ENDIF
*********************************************************

c Effective momentum squared QQ in the ladder run is simulated
c first as dq**2/q**4 from QMIN(J) to QMAX
          QQ=QQMIN/(1.D0+PSRAN(B10)*(QQMIN/QQMAX-1.D0))

        IF(DEBUG.GE.2)WRITE (MONIOU,253)QQ,QQMIN,QQMAX
253     FORMAT(2X,'PSHOT: QQ=',E10.3,2X,'QQMIN=',E10.3,2X,
     *  'QQMAX=',E10.3)

          QT2=QQ*(1.D0-X)**2
          IF(QT2.LT.QT0)GOTO 7

          IF(QQ.GT.QMINN)THEN
            QMIN2=QQ
          ELSE
            QMIN2=QMINN
          ENDIF

          QT=DSQRT(QT2)
          CALL QGSPSCS(CCOS,SSIN)
c EP3 is now 4-vector for s-channel gluon produced in current ladder run
          EP3(3)=QT*CCOS
          EP3(4)=QT*SSIN
          PT2=(EPT(3)-EP3(3))**2+(EPT(4)-EP3(4))**2
          S2MIN2=MAX(S2MIN,QMIN2)

          ZMIN=(QT2+AMJ0)/WW/(1.D0-X)
c S2 is the maximal c.m. energy square for the parton-parton interaction
c in the next ladder run
          S2=X*(1.D0-ZMIN)*WW-PT2
c Rejection in case of too low WW2 (insufficient for elastic gluon-gluon
c scattering with transverse momentum square q_t**2 above QMIN2)
          IF(S2.LT.S2MIN2)GOTO 7

          SJ1=PSJINT(QQ,QMIN(3-JJ),S2,1,IQP(3-jj)+1)
     *    *QGSPSFAP(X,IQP(JJ),0)
          SJ2=PSJINT(QQ,QMIN(3-JJ),S2,2,IQP(3-jj)+1)
     *    *QGSPSFAP(X,IQP(JJ),1)

c GB7 is the rejection function for X and Q**2 simulation. It consists
c from factor
c Q**2/Qmin**2 * ln(Qmin**2/Lambda_qcd**2)/ln(Q**2/Lambda_qcd**2)
c from Q**2 simulation and factor SJ/(X*WW)**DELH * const from X simulation
          GB7=(SJ1+SJ2)/DLOG(QT2/ALM)*QQ*QGSPSUDS(QQ,IQP(JJ))/GB0

*********************************************************
          IF(X.LE..5D0)THEN
            GB7=GB7*X**(1.D0-DELH)
          ELSE
            GB7=GB7*(1.D0-X)*2.D0**DELH
          ENDIF
*********************************************************
          IF(PSRAN(B10).GT.GB7)GOTO 7

           IF(PSRAN(B10).LT.SJ1/(SJ1+SJ2))THEN
             IF(IQC(JJ).EQ.0)THEN
               JT=1
               JQ=INT(1.5D0+PSRAN(B10))
               IQJ(1)=IPQ(JQ,JJ)
               IQJ(2)=0
               DO 31 I=1,4
               EQJ(I,1)=EPQ(I+4*(JQ-1),JJ)
31            EQJ(I,2)=0.D0
            ELSE
              JT=2
              IF(IQC(JJ).GT.0)THEN
                JQ=1
              ELSE
                JQ=2
              ENDIF
              IQJ(1)=0
              DO 32 I=1,4
32            EQJ(I,1)=0.D0

              IPQ(JQ,JJ)=IPQ(1,JJ)
              DO 135 I=1,4
135           EPQ(I+4*(JQ-1),JJ)=EPQ(I,JJ)
            ENDIF
            IQ1=IQC(JJ)
            IQC(JJ)=0

          ELSE
            IF(IQP(JJ).NE.0)THEN
              IQ1=0
              JT=3
              IF(IQC(JJ).GT.0)THEN
                JQ=1
              ELSE
                JQ=2
              ENDIF
              IQJ(1)=IPQ(1,JJ)
              IQJ(2)=0
              DO 33 I=1,4
              EQJ(I,1)=EPQ(I,JJ)
33            EQJ(I,2)=0.D0

            ELSE
              IQ1=INT(3.D0*PSRAN(B10)+1.D0)*(2*INT(.5D0+PSRAN(B10))-1)
              IQC(JJ)=-IQ1
              JT=4
              IF(IQ1.GT.0)THEN
                JQ=1
              ELSE
                JQ=2
              ENDIF
              IQJ(1)=IPQ(JQ,JJ)
              DO 34 I=1,4
34            EQJ(I,1)=EPQ(I+4*(JQ-1),JJ)
            ENDIF
          ENDIF
          IF(DEBUG.GE.3)WRITE (MONIOU,240)JT

          CALL PSCAJET(QT2,IQ1,QV1,ZV1,QM1,IQV1,
     *          LDAU1,LPAR1,JQ)
          Z=(QT2+QM1(1,1))/WW/(1.D0-X)
          SI=X*(1.D0-Z)*WW-PT2

          IF(SI.GT.S2MIN2)THEN
            IQ=MIN(1,IABS(IQC(JJ)))+1
            GB=PSJINT(QQ,QMIN(3-JJ),SI,IQ,IQP(3-JJ)+1)/
     *      PSJINT(QQ,QMIN(3-JJ),S2,IQ,IQP(3-JJ)+1)
            IF(PSRAN(B10).GT.GB)GOTO 301
          ELSE
            GOTO 301
          ENDIF

          WP3=WP(JJ)*(1.D0-X)
          WM3=(QT2+QM1(1,1))/WP3
          EP3(1)=.5D0*(WP3+WM3)
          EP3(2)=.5D0*(WP3-WM3)*(3-2*JJ)

          PT3=DSQRT(EP3(3)**2+EP3(4)**2)

          CALL PSREC(EP3,QV1,ZV1,QM1,IQV1,LDAU1,LPAR1,IQJ,EQJ,JFL,JQ)
          IF(JFL.EQ.0)GOTO 301

          IF(JT.EQ.1)THEN
            IPQ(JQ,JJ)=IQJ(2)
            DO 35 I=1,4
35          EPQ(I+4*(JQ-1),JJ)=EQJ(I,2)

            IF(IPC(JQ,JJ).EQ.0)THEN
              IPC(JQ,JJ)=IQJ(1)
              DO 36 I=1,4
36            EPC(I+4*(JQ-1),JJ)=EQJ(I,1)
            ENDIF

          ELSEIF(JT.EQ.2)THEN
            IPQ(3-JQ,JJ)=IQJ(1)
            DO 37 I=1,4
37          EPQ(I+4*(2-JQ),JJ)=EQJ(I,1)

          ELSEIF(JT.EQ.3)THEN
            IPQ(1,JJ)=IQJ(2)
            DO 38 I=1,4
38          EPQ(I,JJ)=EQJ(I,2)

            IF(IPC(JQ,JJ).EQ.0)THEN
              IPC(JQ,JJ)=IQJ(1)
              DO 39 I=1,4
39            EPC(I+4*(JQ-1),JJ)=EQJ(I,1)
            ENDIF

          ELSEIF(JT.EQ.4)THEN
            IF(IPC(JQ,JJ).EQ.0)THEN
               IPC(JQ,JJ)=IQJ(1)
               DO 40 I=1,4
40            EPC(I+4*(JQ-1),JJ)=EQJ(I,1)
            ENDIF
            IF(JQ.EQ.1)THEN
              IPQ(1,JJ)=IPQ(2,JJ)
              DO 30 I=1,4
30            EPQ(I,JJ)=EPQ(I+4,JJ)
            ENDIF
          ENDIF

          IF(IABS(IQ1).EQ.3)THEN
            IQQQ=8+IQ1/3*4
          ELSE
            IQQQ=8+IQ1
          ENDIF
        IF(DEBUG.GE.2)WRITE (MONIOU,209)TYQ(IQQQ),QT2,EP3
209     FORMAT(2X,'PSHOT: NEW JET FLAVOR:',A2,
     *  ' PT SQUARED FOR THE JET:',E10.3/
     *  4X,'JET 4-MOMENTUM:',4E10.3)
          DO 8 I=1,4
8         EPT(I)=EPT(I)-EP3(I)
c C.m. energy square, minimal  4-momentum transfer square and gluon 4-vector
c for the next ladder run
          QMIN(JJ)=QQ
          QMINN=QMIN2

c Next simulation step will be considered for current ladder
          GOTO 5
C------------------------------------------------

C------------------------------------------------
c The last gluon pair production (elastic scattering) in the ladder
c is simulated
12        CONTINUE
          IF(DEBUG.GE.2)WRITE (MONIOU,211)SI
211     FORMAT(2X,'PSHOT: HIGHEST VIRTUALITY SUBPROCESS IN THE LADDER'/
     *  4X,'MASS SQUARED FOR THE PROCESS:',E10.3)

          XMIN=QMINN/(QMINN+SI)
          XMIN1=.5D0-DSQRT(.25D0-(QT0+AMJ0)/SI)
          XMIN=MAX(XMIN,XMIN1)
          TMIN=SI*XMIN

          IF(IQC(1).NE.0.OR.IQC(2).NE.0)THEN
            GB0=TMIN**2/DLOG(TMIN*(1.D0-XMIN)/ALM)**2*
     *      PSFBORN(SI,TMIN,IQC(1),IQC(2))
          ELSE
            GB0=.25D0*SI**2/DLOG(TMIN*(1.D0-XMIN)/ALM)**2*
     *      PSFBORN(SI,.5D0*SI,IQC(1),IQC(2))
          ENDIF

C------------------------------------------------
c 4-momentum transfer squared is simulated first as dq_t**2/q_t**4 from
c tmin to s/2
13        Q2=TMIN/(1.D0-PSRAN(B10)*(1.D0-2.D0*TMIN/SI))
          Z=Q2/SI
          QT2=Q2*(1.D0-Z)
          IF(PSRAN(B10).LT..5D0)THEN
            JM=2
            TQ=SI-Q2
          ELSE
            JM=1
            TQ=Q2
          ENDIF

          GB=Q2**2/DLOG(QT2/ALM)**2/GB0*
     *    PSFBORN(SI,TQ,IQC(1),IQC(2))
          IF(DEBUG.GE.3)WRITE (MONIOU,241)Q2,GB
241     FORMAT(2X,'PSHOT: Q2=',E10.3,' GB=',E10.3)

          IF(PSRAN(B10).GT.GB)GOTO 13

          IF(IQC(1).EQ.0.AND.IQC(2).EQ.0)THEN
            JQ=INT(1.5D0+PSRAN(B10))
            IQJ(1)=IPQ(JQ,JM)
            DO 51 I=1,4
51          EQJ(I,1)=EPQ(I+4*(JQ-1),JM)

            IF(PSRAN(B10).LT..5D0)THEN
              JT=1
              IF(IPQ(3-JQ,JM)*IPQ(JQ,3-JM).NE.0)THEN
                IPJ=IPQ(3-JQ,JM)
                IPJ1=IPQ(JQ,3-JM)
                IF(IABS(IPJ).EQ.3)IPJ=IPJ*4/3
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                DO 52 I=1,4
                EPJ(I)=EPQ(I+4*(2-JQ),JM)
52              EPJ1(I)=EPQ(I+4*(JQ-1),3-JM)
                CALL PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
                IF(JFL.EQ.0)GOTO 301
              ELSEIF(IPQ(3-JQ,JM).NE.0)THEN
                IPC(JQ,3-JM)=IPQ(3-JQ,JM)
                DO 53 I=1,4
53                   EPC(I+4*(JQ-1),3-JM)=EPQ(I+4*(2-JQ),JM)
              ELSEIF(IPQ(JQ,3-JM).NE.0)THEN
                IPC(3-JQ,JM)=IPQ(JQ,3-JM)
                DO 54 I=1,4
54              EPC(I+4*(2-JQ),JM)=EPQ(I+4*(JQ-1),3-JM)
              ENDIF

              IQJ(2)=0
                     DO 55 I=1,4
55            EQJ(I,2)=0.D0

            ELSE
              JT=2
              IQJ(2)=IPQ(3-JQ,3-JM)
              DO 56 I=1,4
56            EQJ(I,2)=EPQ(I+4*(2-JQ),3-JM)
            ENDIF

          ELSEIF(IQC(1)*IQC(2).EQ.0)THEN
            IF(IQC(1)+IQC(2).GT.0)THEN
              JQ=1
            ELSE
              JQ=2
            ENDIF

            IF(PSRAN(B10).LT..5D0)THEN
              IF(IQC(JM).EQ.0)THEN
                JT=3
                IQJ(1)=IPQ(JQ,JM)
                IQJ(2)=0
                DO 57 I=1,4
                EQJ(I,1)=EPQ(I+4*(JQ-1),JM)
57              EQJ(I,2)=0.D0

                IF(IPQ(3-JQ,JM)*IPQ(1,3-JM).NE.0)THEN
                  IPJ=IPQ(3-JQ,JM)
                  IPJ1=IPQ(1,3-JM)
                  IF(IABS(IPJ).EQ.3)IPJ=IPJ*4/3
                  IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                  DO 58 I=1,4
                  EPJ(I)=EPQ(I+4*(2-JQ),JM)
58                EPJ1(I)=EPQ(I,3-JM)
                  CALL PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
                  IF(JFL.EQ.0)GOTO 301
                ELSEIF(IPQ(3-JQ,JM).NE.0)THEN
                  IPC(JQ,3-JM)=IPQ(3-JQ,JM)
                  DO 59 I=1,4
59                EPC(I+4*(JQ-1),3-JM)=EPQ(I+4*(2-JQ),JM)
                ELSEIF(IPQ(1,3-JM).NE.0)THEN
                  IPC(3-JQ,JM)=IPQ(1,3-JM)
                  DO 60 I=1,4
60                EPC(I+4*(2-JQ),JM)=EPQ(I,3-JM)
                ENDIF

              ELSE
                JT=4
                IQJ(1)=0
                DO 61 I=1,4
61              EQJ(I,1)=0.D0

                IF(IPQ(1,JM)*IPQ(3-JQ,3-JM).NE.0)THEN
                  IPJ=IPQ(1,JM)
                  IPJ1=IPQ(3-JQ,3-JM)
                  IF(IABS(IPJ).EQ.3)IPJ=IPJ*4/3
                  IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                  DO 62 I=1,4
                  EPJ(I)=EPQ(I,JM)
62                EPJ1(I)=EPQ(I+4*(2-JQ),3-JM)
                  CALL PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
                  IF(JFL.EQ.0)GOTO 301
                ELSEIF(IPQ(3-JQ,3-JM).NE.0)THEN
                  IPC(JQ,JM)=IPQ(3-JQ,3-JM)
                  DO 63 I=1,4
63                EPC(I+4*(JQ-1),JM)=EPQ(I+4*(2-JQ),3-JM)
                ELSEIF(IPQ(1,JM).NE.0)THEN
                  IPC(3-JQ,3-JM)=IPQ(1,JM)
                  DO 64 I=1,4
64                EPC(I+4*(2-JQ),3-JM)=EPQ(I,JM)
                ENDIF
              ENDIF

            ELSE
              IF(IQC(JM).EQ.0)THEN
                JT=5
                IQJ(2)=IPQ(3-JQ,JM)
                IQJ(1)=IPQ(1,3-JM)
                DO 65 I=1,4
                EQJ(I,2)=EPQ(I+4*(2-JQ),JM)
65              EQJ(I,1)=EPQ(I,3-JM)
              ELSE
                JT=6
                IQJ(1)=IPQ(JQ,3-JM)
                DO 66 I=1,4
66              EQJ(I,1)=EPQ(I+4*(JQ-1),3-JM)
              ENDIF
            ENDIF

          ELSEIF(IQC(1)*IQC(2).GT.0)THEN
            JT=7
            IF(IQC(1).GT.0)THEN
              JQ=1
            ELSE
              JQ=2
            ENDIF
            IQJ(1)=IPQ(1,3-JM)
            DO 67 I=1,4
67          EQJ(I,1)=EPQ(I,3-JM)

          ELSE
            JT=8
            IF(IQC(JM).GT.0)THEN
              JQ=1
            ELSE
              JQ=2
            ENDIF
            IQJ(1)=0
            DO 68 I=1,4
68          EQJ(I,1)=0.D0

            IF(IPQ(1,JM)*IPQ(1,3-JM).NE.0)THEN
              IPJ=IPQ(1,JM)
              IPJ1=IPQ(1,3-JM)
              IF(IABS(IPJ).EQ.3)IPJ=IPJ*4/3
              IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
              DO 69 I=1,4
              EPJ(I)=EPQ(I,JM)
69            EPJ1(I)=EPQ(I,3-JM)
              CALL PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
              IF(JFL.EQ.0)GOTO 301
            ELSEIF(IPQ(1,3-JM).NE.0)THEN
              IPC(JQ,JM)=IPQ(1,3-JM)
              DO 70 I=1,4
70            EPC(I+4*(JQ-1),JM)=EPQ(I,3-JM)
            ELSEIF(IPQ(1,JM).NE.0)THEN
              IPC(3-JQ,3-JM)=IPQ(1,JM)
              DO 71 I=1,4
71            EPC(I+4*(2-JQ),3-JM)=EPQ(I,JM)
            ENDIF
          ENDIF
          IF(JT.NE.8)THEN
            JQ2=JQ
          ELSE
            JQ2=3-JQ
          ENDIF
          IF(DEBUG.GE.3)WRITE (MONIOU,240)JT
240       FORMAT(2X,'PSHOT: COLOUR CONNECTION JT=:',I1)

          CALL PSCAJET(QT2,IQC(JM),QV1,ZV1,QM1,IQV1,
     *    LDAU1,LPAR1,JQ)
          CALL PSCAJET(QT2,IQC(3-JM),QV2,ZV2,QM2,IQV2,
     *    LDAU2,LPAR2,JQ2)

          AMT1=QT2+QM1(1,1)
          AMT2=QT2+QM2(1,1)

          IF(DSQRT(SI).GT.DSQRT(AMT1)+DSQRT(AMT2))THEN
            Z=XXTWDEC(SI,AMT1,AMT2)
          ELSE
            GOTO 301
          ENDIF

          CALL QGSPSDEFTR(SI,EPT,EY)

          WP3=Z*DSQRT(SI)
          WM3=(QT2+QM1(1,1))/WP3
          EP3(1)=.5D0*(WP3+WM3)
          EP3(2)=.5D0*(WP3-WM3)
          QT=DSQRT(QT2)
          CALL QGSPSCS(CCOS,SSIN)
c ep3 is now 4-vector for first s-channel gluon produced in the ladder run
          EP3(3)=QT*CCOS
          EP3(4)=QT*SSIN

          CALL QGSPSTRANS(EP3,EY)
          PT3=DSQRT(EP3(3)**2+EP3(4)**2)

          CALL PSREC(EP3,QV1,ZV1,QM1,IQV1,LDAU1,LPAR1,IQJ,EQJ,JFL,JQ)
          IF(JFL.EQ.0)GOTO 301

          if(iabs(IQC(JM)).eq.3)then
            iqqq=8+IQC(JM)/3*4
          else
            iqqq=8+IQC(JM)
          endif
          IF(DEBUG.GE.2)WRITE (MONIOU,209)TYQ(IQQQ),QT2

          WP3=(1.D0-Z)*DSQRT(SI)
          WM3=(QT2+QM2(1,1))/WP3
          EP3(1)=.5D0*(WP3+WM3)
          EP3(2)=.5D0*(WP3-WM3)
          EP3(3)=-QT*CCOS
          EP3(4)=-QT*SSIN
          CALL QGSPSTRANS(EP3,EY)
          PT3=DSQRT(EP3(3)**2+EP3(4)**2)

          IF(JT.EQ.1)THEN
            IF(IPC(JQ,JM).EQ.0)THEN
              IPC(JQ,JM)=IQJ(1)
              DO 72 I=1,4
72            EPC(I+4*(JQ-1),JM)=EQJ(I,1)
            ENDIF

            IQJ(1)=IQJ(2)
            IQJ(2)=IPQ(3-JQ,3-JM)
            DO 73 I=1,4
            EQJ(I,1)=EQJ(I,2)
73          EQJ(I,2)=EPQ(I+4*(2-JQ),3-JM)

          ELSEIF(JT.EQ.2)THEN
            IF(IPC(JQ,JM).EQ.0)THEN
              IPC(JQ,JM)=IQJ(1)
              DO 74 I=1,4
74            EPC(I+4*(JQ-1),JM)=EQJ(I,1)
            ENDIF
            IF(IPC(3-JQ,3-JM).EQ.0)THEN
              IPC(3-JQ,3-JM)=IQJ(2)
              DO 75 I=1,4
75            EPC(I+4*(2-JQ),3-JM)=EQJ(I,2)
            ENDIF

            IQJ(2)=IPQ(3-JQ,JM)
            IQJ(1)=IPQ(JQ,3-JM)
            DO 76 I=1,4
            EQJ(I,2)=EPQ(I+4*(2-JQ),JM)
76          EQJ(I,1)=EPQ(I+4*(JQ-1),3-JM)

          ELSEIF(JT.EQ.3)THEN
            IF(IPC(JQ,JM).EQ.0)THEN
              IPC(JQ,JM)=IQJ(1)
              DO 77 I=1,4
77            EPC(I+4*(JQ-1),JM)=EQJ(I,1)
            ENDIF
            IQJ(1)=IQJ(2)
            DO 78 I=1,4
78          EQJ(I,1)= EQJ(I,2)

          ELSEIF(JT.EQ.4)THEN
            IQJ(2)=IQJ(1)
            IQJ(1)=IPQ(JQ,3-JM)
            DO 79 I=1,4
            EQJ(I,2)=EQJ(I,1)
79          EQJ(I,1)=EPQ(I+4*(JQ-1),3-JM)

          ELSEIF(JT.EQ.5)THEN
            IF(IPC(3-JQ,JM).EQ.0)THEN
              IPC(3-JQ,JM)=IQJ(2)
              DO 80 I=1,4
80            EPC(I+4*(2-JQ),JM)=EQJ(I,2)
            ENDIF
            IF(IPC(JQ,3-JM).EQ.0)THEN
              IPC(JQ,3-JM)=IQJ(1)
              DO 81 I=1,4
81            EPC(I+4*(JQ-1),3-JM)=EQJ(I,1)
            ENDIF

            IQJ(1)=IPQ(JQ,JM)
            DO 82 I=1,4
82          EQJ(I,1)=EPQ(I+4*(JQ-1),JM)

          ELSEIF(JT.EQ.6)THEN
            IF(IPC(JQ,3-JM).EQ.0)THEN
              IPC(JQ,3-JM)=IQJ(1)
              DO 83 I=1,4
83            EPC(I+4*(JQ-1),3-JM)=EQJ(I,1)
            ENDIF

            IQJ(2)=IPQ(3-JQ,3-JM)
            IQJ(1)=IPQ(1,JM)
            DO 84 I=1,4
            EQJ(I,2)=EPQ(I+4*(2-JQ),3-JM)
84          EQJ(I,1)=EPQ(I,JM)

          ELSEIF(JT.EQ.7)THEN
            IF(IPC(JQ,3-JM).EQ.0)THEN
              IPC(JQ,3-JM)=IQJ(1)
              DO 85 I=1,4
85            EPC(I+4*(JQ-1),3-JM)=EQJ(I,1)
            ENDIF
            IQJ(1)=IPQ(1,JM)
            DO 86 I=1,4
86          EQJ(I,1)= EPQ(I,JM)
          ENDIF

          CALL PSREC(EP3,QV2,ZV2,QM2,IQV2,LDAU2,LPAR2,IQJ,EQJ,JFL,JQ2)
          IF(JFL.EQ.0)GOTO 301

          if(iabs(IQC(3-JM)).eq.3)then
            iqqq=8+IQC(3-JM)/3*4
          else
            iqqq=8+IQC(3-JM)
          endif
          IF(DEBUG.GE.2)WRITE (MONIOU,209)TYQ(IQQQ),QT2
          IF(DEBUG.GE.2)WRITE (MONIOU,212)NJTOT
212       FORMAT(2X,'PSHOT: TOTAL NUMBER OF JETS:',I2)

          IF(JT.EQ.1)THEN
            IF(IPC(3-JQ,3-JM).EQ.0)THEN
              IPC(3-JQ,3-JM)=IQJ(2)
              DO 87 I=1,4
87            EPC(I+4*(2-JQ),3-JM)=EQJ(I,2)
            ENDIF

          ELSEIF(JT.EQ.2)THEN
            IF(IPC(3-JQ,JM).EQ.0)THEN
              IPC(3-JQ,JM)=IQJ(2)
              DO 88 I=1,4
88            EPC(I+4*(2-JQ),JM)=EQJ(I,2)
            ENDIF
            IF(IPC(JQ,3-JM).EQ.0)THEN
              IPC(JQ,3-JM)=IQJ(1)
              DO 89 I=1,4
89            EPC(I+4*(JQ-1),3-JM)=EQJ(I,1)
            ENDIF

          ELSEIF(JT.EQ.4)THEN
            IF(IPC(JQ,3-JM).EQ.0)THEN
              IPC(JQ,3-JM)=IQJ(1)
              DO 90 I=1,4
90            EPC(I+4*(JQ-1),3-JM)=EQJ(I,1)
            ENDIF

          ELSEIF(JT.EQ.5)THEN
            IF(IPC(JQ,JM).EQ.0)THEN
              IPC(JQ,JM)=IQJ(1)
              DO 91 I=1,4
91            EPC(I+4*(JQ-1),JM)=EQJ(I,1)
            ENDIF

          ELSEIF(JT.EQ.6)THEN
            IF(IPC(3-JQ,3-JM).EQ.0)THEN
              IPC(3-JQ,3-JM)=IQJ(2)
              DO 92 I=1,4
92            EPC(I+4*(2-JQ),3-JM)=EQJ(I,2)
            ENDIF
            IF(IPC(JQ,JM).EQ.0)THEN
              IPC(JQ,JM)=IQJ(1)
              DO 93 I=1,4
93            EPC(I+4*(JQ-1),JM)=EQJ(I,1)
            ENDIF

          ELSEIF(JT.EQ.7)THEN
            IF(IPC(JQ,JM).EQ.0)THEN
              IPC(JQ,JM)=IQJ(1)
              DO 94 I=1,4
94            EPC(I+4*(JQ-1),JM)=EQJ(I,1)
            ENDIF
          ENDIF
C------------------------------------------------

        IF(DEBUG.GE.3)WRITE (MONIOU,217)
217     FORMAT(2X,'PSHOT - END')
        ebal(1)=.5*(wp0+wm0)
        ebal(2)=.5*(wp0-wm0)
        ebal(3)=0.d0
        ebal(4)=0.d0
        do 500 i=1,njtot
        do 500 m=1,2
        do 500 l=1,4
500        ebal(l)=ebal(l)-epjet(l,m,i)
c            write (*,*)'ebal',ebal
        RETURN
        END
C=======================================================================

        SUBROUTINE PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
c Procedure for jet hadronization - each gluon is
c considered to be splitted into quark-antiquark pair and usual soft
c strings are assumed to be formed between quark and antiquark
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EPJ(4),EPJ1(4),EPT(4)
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        COMMON /Q_AREA46/ EPJET(4,2,15000),IPJET(2,15000)
        COMMON /Q_AREA47/ NJTOT
        SAVE

c        if(ipj*ipj1.gt.0.and.iabs(ipj).ne.3.and.iabs(ipj).le.4.
c     *  and.iabs(ipj1).ne.3.and.iabs(ipj1).le.4.or.
c     *  ipj*ipj1.lt.0.and.(iabs(ipj).eq.3.or.iabs(ipj).gt.4.
c     *  or.iabs(ipj1).eq.3.or.iabs(ipj1).eq.4))then
c      write (*,*)'ipj,ipj1',ipj,ipj1
c           read (*,*)
c        endif

        IF(DEBUG.GE.2)WRITE (MONIOU,201)IPJ,IPJ1,EPJ,EPJ1
201     FORMAT(2X,'PSJDEF: PARTON FLAVORS',
     *  ': IPJ=',I2,2X,'IPJ1=',I2/
     *  4X,'PARTON 4-MOMENTA:',2X,4(E10.3,1X))
        DO 1 I=1,4
1       EPT(I)=EPJ(I)+EPJ1(I)

c Invariant mass squared for the jet
        WW=QGSPSNORM(EPt)
c Minimal mass squared for the jet
        IF(IABS(IPJ).LE.2)THEN
          AM1=AM(1)
        ELSEIF(IABS(IPJ).EQ.4)THEN
          AM1=AM(3)
        ELSE
          AM1=AM(2)
        ENDIF
        IF(IABS(IPJ1).LE.2)THEN
          AM2=AM(1)
        ELSEIF(IABS(IPJ1).EQ.4)THEN
          AM2=AM(3)
        ELSE
          AM2=AM(2)
        ENDIF
        AMJ=(AM1+AM2)**2

        IF(AMJ.GT.WW)THEN
          JFL=0
          RETURN
        ELSE
          JFL=1
        ENDIF

        NJTOT=NJTOT+1
        IF( NJTOT . GT. 15000 ) THEN
          WRITE(MONIOU,*)'PSJDEF: TOO MANY JETS'
          WRITE(MONIOU,*)'PSJDEF: NJTOT = ',NJTOT
          STOP
        ENDIF
        IPJET(1,NJTOT)=IPJ
        IPJET(2,NJTOT)=IPJ1
        DO 2 I=1,4
        EPJET(I,1,NJTOT)=EPJ(I)
2       EPJET(I,2,NJTOT)=EPJ1(I)

        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'PSJDEF - END')
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSJET(Q1,Q2,S,S2MIN,J,L)
C QGSPSJET - inclusive hard cross-section calculation (one more run is added
c to the ladder) - for any ordering
c Q1 - effective momentum cutoff for current end of the ladder,
c Q2 - effective momentum cutoff for opposide end of the ladder,
c S - total c.m. energy squared for the ladder,
c S2MIN - minimal c.m. energy squared for BORN process (above Q1 and Q2)
c J - parton type at current end of the ladder (0 - g, 1 - q)
c L - parton type at opposite end of the ladder (1 - g, 2 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA6/  PI,BM,AM
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON/AR13/X1(7),A1(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,Q1,Q2,S2MIN,J,L
201     FORMAT(2X,'QGSPSJET - UNORDERED LADDER CROSS SECTION:'/
     *  4X,'S=',E10.3,2X,'Q1=',E10.3,2X,'Q2=',E10.3,2X,'S2MIN=',
     *  E10.3,2X,'J=',I1,2X,'L=',I1)
        QGSPSJET=0.D0

        P=DSQRT(1.D0-3.D0*QT0/S)
        COSF=(1.D0-18.D0*QT0/S)/P**3
        FI=ATAN(1.D0/COSF**2-1.D0)
        IF(COSF.LT.0.D0)FI=PI-FI
        FI=FI/3.D0
        ZMAX=(2.D0-P*(DSQRT(3.D0)*SIN(FI)-COS(FI)))/3.D0
        ZMIN=(1.D0-P*COS(FI))/1.5D0

        IF(QT0/(1.D0-ZMIN)**2.LT.S2MIN)
     *  ZMIN=.5D0*(1.D0+S2MIN/S-DSQRT((1.D0-S2MIN/S)**2-4.D0*QT0/S))

***********************************************************
        IF(1.D0-ZMIN.LT.DSQRT(QT0/Q1))THEN
          QMIN=QT0/(1.D0-ZMIN)**2
        ELSE
          QMIN=Q1
        ENDIF

        QMAX=QT0/(1.D0-ZMAX)**2
        SUD0=QGSPSUDS(QMIN,J)
***********************************************************

        IF(DEBUG.GE.3)WRITE (MONIOU,203)QMIN,QMAX
203     FORMAT(2X,'QGSPSJET:',2X,'QMIN=',E10.3,2X,'QMAX=',E10.3)
        IF(QMAX.GT.QMIN)THEN

c Numerical integration over transverse momentum square;
c Gaussian integration is used
          DO 3 I=1,7
          DO 3 M=1,2
          QI=2.D0*QMIN/(1.D0+QMIN/QMAX+(2*M-3)*X1(I)*(1.D0-QMIN/QMAX))

          ZMAX=(1.D0-DSQRT(QT0/QI))**DELH
          ZMIN=((QI+MAX(QI,S2MIN))/(QI+S))**DELH

          FSJ=0.D0

        IF(DEBUG.GE.3)WRITE (MONIOU,204)QI,ZMIN,ZMAX
204     FORMAT(2X,'QGSPSJET:',2X,'QI=',E10.3,2X,'ZMIN=',E10.3,2X,
     *  'ZMAX=',E10.3)
          IF(ZMAX.GT.ZMIN)THEN
            DO 2 I1=1,7
            DO 2 M1=1,2
            Z=(.5D0*(ZMAX+ZMIN+(2*M1-3)*X1(I1)*(ZMAX-ZMIN)))**
     *      (1.D0/DELH)
            QT=QI*(1.D0-Z)**2
            S2=Z*S-QI*(1.D0-Z)

            SJ=0.D0
            DO 1 K=1,2
1           SJ=SJ+PSJINT(QI,Q2,S2,K,L)*QGSPSFAP(Z,J,K-1)*Z
2           FSJ=FSJ+A1(I1)*SJ/DLOG(QT/ALM)/Z**DELH
            FSJ=FSJ*(ZMAX-ZMIN)
          ENDIF

3         QGSPSJET=QGSPSJET+A1(I)*FSJ*QI*QGSPSUDS(QI,J)
          QGSPSJET=QGSPSJET*(1.D0/QMIN-1.D0/QMAX)/SUD0/DELH/18.D0
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSJET
202     FORMAT(2X,'QGSPSJET=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSJET1(Q1,Q2,S,S2MIN,J,L)
C QGSPSJET1 - inclusive hard cross-section calculation (one more run is added
c to the ladder) - for strict ordering
c Q1 - effective momentum cutoff for current end of the ladder,
c Q2 - effective momentum cutoff for opposide end of the ladder,
c S - total c.m. energy squared for the ladder,
c S2MIN - minimal c.m. energy squared for BORN process (above Q1 and Q2)
c J - parton type at current end of the ladder (0 - g, 1 - q)
c L - parton type at opposite end of the ladder (1 - g, 2 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA6/  PI,BM,AM
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON/AR13/X1(7),A1(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,Q1,Q2,S2MIN,J,L
201     FORMAT(2X,'QGSPSJET1 - STRICTLY ORDERED LADDER CROSS SECTION:'/
     *  4X,'S=',E10.3,2X,'Q1=',E10.3,2X,'Q2=',E10.3,2X,'S2MIN=',
     *  E10.3,2X,'J=',I1,2X,'L=',I1)
        QGSPSJET1=0.D0

        P=DSQRT(1.D0-3.D0*QT0/S)
        COSF=(1.D0-18.D0*QT0/S)/P**3
        FI=ATAN(1.D0/COSF**2-1.D0)
        IF(COSF.LT.0.D0)FI=PI-FI
        FI=FI/3.D0
        ZMAX=(2.D0-P*(DSQRT(3.D0)*SIN(FI)-COS(FI)))/3.D0
        ZMIN=(1.D0-P*COS(FI))/1.5D0

        IF(QT0/(1.D0-ZMIN)**2.LT.S2MIN)
     *  ZMIN=.5D0*(1.D0+S2MIN/S-DSQRT((1.D0-S2MIN/S)**2-4.D0*QT0/S))

***********************************************************
        IF(1.D0-ZMIN.LT.DSQRT(QT0/Q1))THEN
          QMIN=QT0/(1.D0-ZMIN)**2
        ELSE
          QMIN=Q1
        ENDIF

        QMAX=QT0/(1.D0-ZMAX)**2
        SUD0=QGSPSUDS(QMIN,J)
***********************************************************

        IF(DEBUG.GE.3)WRITE (MONIOU,203)QMIN,QMAX
203     FORMAT(2X,'QGSPSJET1:',2X,'QMIN=',E10.3,2X,'QMAX=',E10.3)
        IF(QMAX.GT.QMIN)THEN

c Numerical integration over transverse momentum square;
c Gaussian integration is used
          DO 3 I=1,7
          DO 3 M=1,2
          QI=2.D0*QMIN/(1.D0+QMIN/QMAX+(2*M-3)*X1(I)*(1.D0-QMIN/QMAX))

          ZMAX=(1.D0-DSQRT(QT0/QI))**DELH
          ZMIN=((QI+MAX(QI,S2MIN))/(QI+S))**DELH

          FSJ=0.D0

        IF(DEBUG.GE.3)WRITE (MONIOU,204)QI,ZMIN,ZMAX
204     FORMAT(2X,'QGSPSJET1:',2X,'QI=',E10.3,2X,'ZMIN=',E10.3,2X,
     *  'ZMAX=',E10.3)
          IF(ZMAX.GT.ZMIN)THEN
            DO 2 I1=1,7
            DO 2 M1=1,2
            Z=(.5D0*(ZMAX+ZMIN+(2*M1-3)*X1(I1)*(ZMAX-ZMIN)))**
     *      (1.D0/DELH)
            QT=QI*(1.D0-Z)**2
            S2=Z*S-QI*(1.D0-Z)

            SJ=0.D0
            DO 1 K=1,2
1           SJ=SJ+PSJINT1(QI,Q2,S2,K,L)*QGSPSFAP(Z,J,K-1)*Z
2           FSJ=FSJ+A1(I1)*SJ/DLOG(QT/ALM)/Z**DELH
            FSJ=FSJ*(ZMAX-ZMIN)
          ENDIF

3         QGSPSJET1=QGSPSJET1+A1(I)*FSJ*QI*QGSPSUDS(QI,J)
          QGSPSJET1=QGSPSJET1*(1.D0/QMIN-1.D0/QMAX)/SUD0/DELH/18.D0
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSJET1
202     FORMAT(2X,'QGSPSJET1=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSJINT(Q1,Q2,S,M,L)
C PSJINT - inclusive hard cross-section interpolation - for any ordering
c in the ladder
c Q1 - effective momentum cutoff for current end of the ladder,
c Q2 - effective momentum cutoff for opposide end of the ladder,
c S - total c.m. energy squared for the ladder,
c M - parton type at current end of the ladder (1 - g, 2 - q)
c L - parton type at opposite end of the ladder (1 - g, 2 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION WI(3),WJ(3),WK(3)
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA29/ CSJ(17,17,68)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,Q1,Q2,M,L
201     FORMAT(2X,'PSJINT - UNORDERED LADDER CROSS SECTION INTERPOL.:'/
     *  4X,'S=',E10.3,2X,'Q1=',E10.3,2X,'Q2=',E10.3,2X,
     *  2X,'M=',I1,2X,'L=',I1)
        PSJINT=0.D0
        QQ=MAX(Q1,Q2)
      IF(S.LE.MAX(4.D0*QT0,QQ))THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSJINT
202     FORMAT(2X,'PSJINT=',E10.3)
        RETURN
      ENDIF

        ML=17*(M-1)+34*(L-1)
        QLI=DLOG(Q1/QT0)/1.38629D0
        QLJ=DLOG(Q2/QT0)/1.38629D0
        SL=DLOG(S/QT0)/1.38629D0
        SQL=SL-MAX(QLI,QLJ)
        I=INT(QLI)
        J=INT(QLJ)
        K=INT(SL)
        IF(I.GT.13)I=13
        IF(J.GT.13)J=13

        IF(SQL.GT.10.D0)THEN
          IF(K.GT.14)K=14
          IF(I.GT.K-3)I=K-3
          IF(J.GT.K-3)J=K-3
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)
          WK(2)=SL-K
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 1 I1=1,3
          DO 1 J1=1,3
          DO 1 K1=1,3
1         PSJINT=PSJINT+CSJ(I+I1,J+J1,K+K1+ML)*WI(I1)*WJ(J1)*WK(K1)
          PSJINT=EXP(PSJINT)
        ELSEIF(SQL.LT.1.D0.AND.I+J.NE.0)THEN
          SQ=(S/MAX(Q1,Q2)-1.D0)/3.D0
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)

          DO 2 I1=1,3
          I2=I+I1
          DO 2 J1=1,3
          J2=J+J1
          K2=MAX(I2,J2)+1+ML
2         PSJINT=PSJINT+CSJ(I2,J2,K2)*WI(I1)*WJ(J1)
          PSJINT=EXP(PSJINT)*SQ
        ELSEIF(K.EQ.1)THEN
          SQ=(S/QT0/4.D0-1.D0)/3.D0
          WI(2)=QLI
          WI(1)=1.D0-QLI
          WJ(2)=QLJ
          WJ(1)=1.D0-QLJ

          DO 3 I1=1,2
          DO 3 J1=1,2
3         PSJINT=PSJINT+CSJ(I1,J1,3+ML)*WI(I1)*WJ(J1)
          PSJINT=EXP(PSJINT)*SQ
        ELSEIF(K.LT.15)THEN
          KL=INT(SQL)
          IF(I+KL.GT.12)I=12-KL
          IF(J+KL.GT.12)J=12-KL
          IF(I+J+KL.EQ.1)KL=2
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)
          WK(2)=SQL-KL
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 4 I1=1,3
          I2=I+I1
          DO 4 J1=1,3
          J2=J+J1
          DO 4 K1=1,3
          K2=MAX(I2,J2)+KL+K1-1+ML
4         PSJINT=PSJINT+CSJ(I2,J2,K2)*WI(I1)*WJ(J1)*WK(K1)
          PSJINT=EXP(PSJINT)
        ELSE
          K=15
          IF(I.GT.K-3)I=K-3
          IF(J.GT.K-3)J=K-3
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)
          WK(2)=SL-K
          WK(1)=1.D0-WK(2)

          DO 5 I1=1,3
          DO 5 J1=1,3
          DO 5 K1=1,2
5         PSJINT=PSJINT+CSJ(I+I1,J+J1,K+K1+ML)*WI(I1)*WJ(J1)*WK(K1)
          PSJINT=EXP(PSJINT)
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSJINT
        RETURN
        END
C=======================================================================

        SUBROUTINE PSJINT0(S,SJ,SJB,M,L)
C PSJINT0 - inclusive hard cross-section interpolation - for minimal
c effective momentum cutoff in the ladder
c S - total c.m. energy squared for the ladder,
c SJ - inclusive jet cross-section,
c SJB - Born cross-section,
c M - parton type at current end of the ladder (0 - g, 1 - q)
c L - parton type at opposite end of the ladder (0 - g, 1 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION WK(3)
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA32/ CSJ(17,2,2),CSB(17,2,2)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,M,L
201     FORMAT(2X,'PSJINT0 - HARD CROSS SECTION INTERPOLATION:'/
     *  4X,'S=',E10.3,2X,'M=',I1,2X,'L=',I1)
        SJ=0.D0
        SJB=0.D0
      IF(S.LE.4.D0*QT0)THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,202)SJ,SJB
202     FORMAT(2X,'PSJINT0: SJ=',E10.3,2X,'SJB=',E10.3)
        RETURN
      ENDIF

        SL=DLOG(S/QT0)/1.38629d0
        K=INT(SL)
        IF(K.EQ.1)THEN
          SQ=(S/QT0/4.D0-1.D0)/3.D0
          SJB=EXP(CSB(3,M+1,L+1))*SQ
          SJ=EXP(CSJ(3,M+1,L+1))*SQ
        ELSE
          IF(K.GT.14)K=14
          WK(2)=SL-K
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 1 K1=1,3
          SJ=SJ+CSJ(K+K1,M+1,L+1)*WK(K1)
1         SJB=SJB+CSB(K+K1,M+1,L+1)*WK(K1)
          SJB=EXP(SJB)
          SJ=EXP(SJ)
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)SJ,SJB
        RETURN
        END
C=======================================================================

        FUNCTION PSJINT1(Q1,Q2,S,M,L)
C PSJINT1 - inclusive hard cross-section interpolation - for strict ordering
c in the ladder
c Q1 - effective momentum cutoff for current end of the ladder,
c Q2 - effective momentum cutoff for opposide end of the ladder,
c S - total c.m. energy squared for the ladder,
c M - parton type at current end of the ladder (1 - g, 2 - q)
c L - parton type at opposite end of the ladder (1 - g, 2 - q)
C-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION WI(3),WJ(3),WK(3)
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA30/ CSJ(17,17,68)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,Q1,Q2,M,L
201     FORMAT(2X,'PSJINT1 - STRICTLY ORDERED LADDER CROSS SECTION',
     *  ' INTERPOLATION:'/
     *  4X,'S=',E10.3,2X,'Q1=',E10.3,2X,'Q2=',E10.3,2X,
     *  4X,'M=',I1,2X,'L=',I1)
        PSJINT1=0.D0
        QQ=MAX(Q1,Q2)
      IF(S.LE.MAX(4.D0*QT0,QQ))THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSJINT1
202     FORMAT(2X,'PSJINT1=',E10.3)
        RETURN
      ENDIF

        ML=17*(M-1)+34*(L-1)
        QLI=DLOG(Q1/QT0)/1.38629d0
        QLJ=DLOG(Q2/QT0)/1.38629d0
        SL=DLOG(S/QT0)/1.38629d0
        SQL=SL-MAX(QLI,QLJ)
        I=INT(QLI)
        J=INT(QLJ)
        K=INT(SL)
        IF(I.GT.13)I=13
        IF(J.GT.13)J=13

        IF(SQL.GT.10.D0)THEN
          IF(K.GT.14)K=14
          IF(I.GT.K-3)I=K-3
          IF(J.GT.K-3)J=K-3
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)
          WK(2)=SL-K
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 1 I1=1,3
          DO 1 J1=1,3
          DO 1 K1=1,3
1         PSJINT1=PSJINT1+CSJ(I+I1,J+J1,K+K1+ML)*WI(I1)*WJ(J1)*WK(K1)
          PSJINT1=EXP(PSJINT1)
        ELSEIF(SQL.LT.1.D0.AND.I+J.NE.0)THEN
          SQ=(S/MAX(Q1,Q2)-1.D0)/3.D0
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)

          DO 2 I1=1,3
          I2=I+I1
          DO 2 J1=1,3
          J2=J+J1
          K2=MAX(I2,J2)+1+ML
2         PSJINT1=PSJINT1+CSJ(I2,J2,K2)*WI(I1)*WJ(J1)
          PSJINT1=EXP(PSJINT1)*SQ
        ELSEIF(K.EQ.1)THEN
          SQ=(S/QT0/4.D0-1.D0)/3.D0
          WI(2)=QLI
          WI(1)=1.D0-QLI
          WJ(2)=QLJ
          WJ(1)=1.D0-QLJ

          DO 3 I1=1,2
          DO 3 J1=1,2
3         PSJINT1=PSJINT1+CSJ(I1,J1,3+ML)*WI(I1)*WJ(J1)
          PSJINT1=EXP(PSJINT1)*SQ
        ELSEIF(K.LT.15)THEN
          KL=INT(SQL)
          IF(I+KL.GT.12)I=12-KL
          IF(J+KL.GT.12)J=12-KL
          IF(I+J+KL.EQ.1)KL=2

          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)
          WK(2)=SQL-KL
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          DO 4 I1=1,3
          I2=I+I1
          DO 4 J1=1,3
          J2=J+J1
          DO 4 K1=1,3
          K2=MAX(I2,J2)+KL+K1-1+ML
4         PSJINT1=PSJINT1+CSJ(I2,J2,K2)*WI(I1)*WJ(J1)*WK(K1)
          PSJINT1=EXP(PSJINT1)
        ELSE
          K=15
          IF(I.GT.K-3)I=K-3
          IF(J.GT.K-3)J=K-3
          WI(2)=QLI-I
          WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
          WI(1)=1.D0-WI(2)+WI(3)
          WI(2)=WI(2)-2.D0*WI(3)
          WJ(2)=QLJ-J
          WJ(3)=WJ(2)*(WJ(2)-1.D0)*.5D0
          WJ(1)=1.D0-WJ(2)+WJ(3)
          WJ(2)=WJ(2)-2.D0*WJ(3)
          WK(2)=SL-K
          WK(1)=1.D0-WK(2)

          DO 5 I1=1,3
          DO 5 J1=1,3
          DO 5 K1=1,2
5         PSJINT1=PSJINT1+CSJ(I+I1,J+J1,K+K1+ML)*WI(I1)*WJ(J1)*WK(K1)
          PSJINT1=EXP(PSJINT1)
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSJINT1
        RETURN
        END
C=======================================================================

       FUNCTION QGSPSLAM(S,A,B)
c Kinematical function for two particle decay - maximal Pt-value
c A - first particle mass squared,
C B - second particle mass squared,
C S - two particle invariant mass
c-----------------------------------------------------------------------
       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
       COMMON /Q_AREA43/ MONIOU
       COMMON /Q_DEBUG/  DEBUG
       SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,A,B
201     FORMAT(2X,'QGSPSLAM - KINEMATICAL FUNCTION S=',E10.3,2X,'A=',
     *  E10.3,2X,'B=',E10.3)
       QGSPSLAM=.25D0/S*(S+A-B)**2-A
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSLAM
202     FORMAT(2X,'QGSPSLAM=',E10.3)
       RETURN
       END
C=======================================================================

        FUNCTION QGSPSNORM(EP)
c 4-vector squared calculation
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)EP
201     FORMAT(2X,'QGSPSNORM - 4-VECTOR SQUARED FOR ',
     *  'EP=',4(E10.3,1X))
        QGSPSNORM=EP(1)**2
        DO 1 I=1,3
1       QGSPSNORM=QGSPSNORM-EP(I+1)**2
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSNORM
202     FORMAT(2X,'QGSPSNORM=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE PSREC(EP,QV,ZV,QM,IQV,LDAU,LPAR,IQJ,EQJ,JFL,JQ)
c Jet reconstructuring procedure - 4-momenta for all final jets are determined
c EP(i) - jet 4-momentum
C-----------------------------------------------------------------------
c QV(i,j) - effective momentum for the branching of the parton in i-th row
c on j-th level (0 - in case of no branching)
c ZV(i,j) - Z-value for the branching of the parton in i-th row
c on j-th level
c QM(i,j) - mass squared for the parton in i-th row
c on j-th level
c IQV(i,j) - flavours for the parton in i-th row on j-th level
c LDAU(i,j) - first daughter row for the branching of the parton in i-th row
c on j-th level
c LPAR(i,j) - the parent row for the parton in i-th row on j-th level
C-----------------------------------------------------------------------
               IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4),EP3(4),EPV(4,30,50),
     *  QV(30,50),ZV(30,50),QM(30,50),IQV(30,50),
     *  LDAU(30,49),LPAR(30,50),
     *  IQJ(2),EQJ(4,2),IPQ(2,30,50),EPQ(8,30,50),
     *  EPJ(4),EPJ1(4)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)JQ,EP,IQJ
201     FORMAT(2X,'PSREC - JET RECONSTRUCTURING: JQ=',I1/
     *  4X,'JET 4-MOMENTUM EP=',4(E10.3,1X)/4X,'IQJ=',2I2)
        JFL = 1
        DO 1 I=1,4
        EPV(I,1,1)=EP(I)
1       EPQ(I,1,1)=EQJ(I,1)
        IPQ(1,1,1)=IQJ(1)

        IF(IQV(1,1).EQ.0)THEN
          DO 2 I=1,4
2         EPQ(I+4,1,1)=EQJ(I,2)
          IPQ(2,1,1)=IQJ(2)
        ENDIF

        NLEV=1
        NROW=1

3       CONTINUE

        IF(QV(NROW,NLEV).EQ.0.D0)THEN
           IPJ=IQV(NROW,NLEV)
           IF(IPJ.NE.0)THEN
             IF(EPQ(1,NROW,NLEV).NE.0.D0)THEN
               IF(IABS(IPJ).EQ.3)IPJ=IPJ*4/3
              DO 4 I=1,4
              EPJ(I)=EPV(I,NROW,NLEV)
4             EPJ1(I)=EPQ(I,NROW,NLEV)
              IPJ1=IPQ(1,NROW,NLEV)
              IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
              CALL PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
        IF(DEBUG.GE.3)WRITE (MONIOU,211)IPJ,IPJ1,JFL
211     FORMAT(2X,'PSREC - NEW STRING FLAVOURS: ',2I3,' JFL=',I1)
              IF(JFL.EQ.0)RETURN
            ELSE
              IPQ(1,NROW,NLEV)=IPJ
              DO 5 I=1,4
5             EPQ(I,NROW,NLEV)=EPV(I,NROW,NLEV)
        IF(DEBUG.GE.3)WRITE (MONIOU,212)IPJ,
     *  (EPV(I,NROW,NLEV),I=1,4),JFL
212     FORMAT(2X,'PSREC: NEW FINAL JET FLAVOR: ',I3,2X,
     *         'JET 4-MOMENTUM:', 4(E10.3,1X),' JFL=',I1)
            ENDIF

           ELSE
             IPJ=INT(2.D0*PSRAN(B10)+1.D0)*(3-2*JQ)
            DO 6 I=1,4
6           EPJ(I)=.5D0*EPV(I,NROW,NLEV)

            DO 9 M=1,2
            IF(EPQ(1+4*(M-1),NROW,NLEV).NE.0.D0)THEN
              DO 7 I=1,4
7             EPJ1(I)=EPQ(4*(M-1)+I,NROW,NLEV)
              IPJ1=IPQ(M,NROW,NLEV)
              IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
              CALL PSJDEF(IPJ,IPJ1,EPJ,EPJ1,JFL)
              IF(JFL.EQ.0)RETURN
            ELSE
              IPQ(M,NROW,NLEV)=IPJ
              DO 8 I=1,4
8             EPQ(4*(M-1)+I,NROW,NLEV)=EPJ(I)
            ENDIF
9           IPJ=-IPJ
          ENDIF

        IF(DEBUG.GE.3)WRITE (MONIOU,204)NLEV,NROW,IQV(NROW,NLEV),
     *  (EPV(I,NROW,NLEV),I=1,4)
204     FORMAT(2X,'PSREC: FINAL JET AT LEVEL NLEV=',I2,
     *  ' NROW=',I2/4X,'JET FLAVOR: ',I3,2X,'JET 4-MOMENTUM:',
     *  4(E10.3,1X))
         ELSE

          DO 10 I=1,4
10        EP3(I)=EPV(I,NROW,NLEV)
          CALL QGSPSDEFROT(EP3,S0X,C0X,S0,C0)
          Z=ZV(NROW,NLEV)
          QT2=(Z*(1.D0-Z))**2*QV(NROW,NLEV)
          LDROW=LDAU(NROW,NLEV)

          WP0=EP3(1)+EP3(2)
          WPI=Z*WP0
          WMI=(QT2+QM(LDROW,NLEV+1))/WPI
          EP3(1)=.5D0*(WPI+WMI)
          EP3(2)=.5D0*(WPI-WMI)
          QT=DSQRT(QT2)
          CALL QGSPSCS(C,S)
          EP3(3)=QT*C
          EP3(4)=QT*S
          CALL QGSPSROTAT(EP3,S0X,C0X,S0,C0)

          DO 11 I=1,4
11        EPV(I,LDROW,NLEV+1)=EP3(I)
        IF(DEBUG.GE.3)WRITE (MONIOU,206)NLEV+1,LDROW,EP3
206     FORMAT(2X,'PSREC: JET AT LEVEL NLEV=',I2,
     *  ' NROW=',I2/4X,'JET 4-MOMENTUM:',4(E10.3,1X))

          WPI=(1.D0-Z)*WP0
          WMI=(QT2+QM(LDROW+1,NLEV+1))/WPI
          EP3(1)=.5D0*(WPI+WMI)
          EP3(2)=.5D0*(WPI-WMI)
          EP3(3)=-QT*C
          EP3(4)=-QT*S
          CALL QGSPSROTAT(EP3,S0X,C0X,S0,C0)
        IF(DEBUG.GE.3)WRITE (MONIOU,206)NLEV+1,LDROW+1,EP3

          DO 12 I=1,4
12        EPV(I,LDROW+1,NLEV+1)=EP3(I)

          IF(IQV(NROW,NLEV).EQ.0)THEN
            IF(IQV(LDROW,NLEV+1).NE.0)THEN
              IPQ(1,LDROW,NLEV+1)=IPQ(1,NROW,NLEV)
              IPQ(1,LDROW+1,NLEV+1)=IPQ(2,NROW,NLEV)
              DO 13 I=1,4
              EPQ(I,LDROW,NLEV+1)=EPQ(I,NROW,NLEV)
13            EPQ(I,LDROW+1,NLEV+1)=EPQ(I+4,NROW,NLEV)
            ELSE
              IPQ(1,LDROW,NLEV+1)=IPQ(1,NROW,NLEV)
              IPQ(2,LDROW,NLEV+1)=0
              IPQ(1,LDROW+1,NLEV+1)=0
              IPQ(2,LDROW+1,NLEV+1)=IPQ(2,NROW,NLEV)
              DO 14 I=1,4
              EPQ(I,LDROW,NLEV+1)=EPQ(I,NROW,NLEV)
              EPQ(I+4,LDROW,NLEV+1)=0.D0
              EPQ(I,LDROW+1,NLEV+1)=0.D0
14            EPQ(I+4,LDROW+1,NLEV+1)=EPQ(I+4,NROW,NLEV)
            ENDIF
          ELSE
            IF(IQV(LDROW,NLEV+1).EQ.0)THEN
              IPQ(1,LDROW,NLEV+1)=IPQ(1,NROW,NLEV)
              IPQ(2,LDROW,NLEV+1)=0
              IPQ(1,LDROW+1,NLEV+1)=0
              DO 15 I=1,4
              EPQ(I,LDROW,NLEV+1)=EPQ(I,NROW,NLEV)
              EPQ(I+4,LDROW,NLEV+1)=0.D0
15            EPQ(I,LDROW+1,NLEV+1)=0.D0
            ELSE
              IPQ(1,LDROW,NLEV+1)=0
              IPQ(1,LDROW+1,NLEV+1)=0
              IPQ(2,LDROW+1,NLEV+1)=IPQ(1,NROW,NLEV)
              DO 16 I=1,4
              EPQ(I,LDROW,NLEV+1)=0.D0
              EPQ(I,LDROW+1,NLEV+1)=0.D0
16            EPQ(I+4,LDROW+1,NLEV+1)=EPQ(I,NROW,NLEV)
            ENDIF
          ENDIF

          NROW=LDROW
          NLEV=NLEV+1
          GOTO 3
        ENDIF

17      CONTINUE
        IF(NLEV.EQ.1)THEN
          IQJ(1)=IPQ(1,1,1)
          DO 18 I=1,4
18        EQJ(I,1)=EPQ(I,1,1)
          IF(IQV(1,1).EQ.0)THEN
            IQJ(2)=IPQ(2,1,1)
            DO 19 I=1,4
19          EQJ(I,2)=EPQ(I+4,1,1)
          ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)iqj
202     FORMAT(2X,'PSREC - END',2x,'iqj=',2i2)
        RETURN
      ENDIF

        LPROW=LPAR(NROW,NLEV)

        IF(LDAU(LPROW,NLEV-1).EQ.NROW)THEN
           IF(IQV(NROW,NLEV).EQ.0)THEN
             IF(EPQ(1,LPROW,NLEV-1).EQ.0.D0)THEN
              IPQ(1,LPROW,NLEV-1)=IPQ(1,NROW,NLEV)
              DO 20 I=1,4
20            EPQ(I,LPROW,NLEV-1)=EPQ(I,NROW,NLEV)
            ENDIF
            IPQ(1,NROW+1,NLEV)=IPQ(2,NROW,NLEV)
            DO 21 I=1,4
21          EPQ(I,NROW+1,NLEV)=EPQ(I+4,NROW,NLEV)
          ELSE
            IF(IQV(LPROW,NLEV-1).EQ.0)THEN
              IF(EPQ(1,LPROW,NLEV-1).EQ.0.D0)THEN
                IPQ(1,LPROW,NLEV-1)=IPQ(1,NROW,NLEV)
                DO 22 I=1,4
22              EPQ(I,LPROW,NLEV-1)=EPQ(I,NROW,NLEV)
              ENDIF
            ELSE
              IPQ(1,NROW+1,NLEV)=IPQ(1,NROW,NLEV)
              DO 23 I=1,4
23            EPQ(I,NROW+1,NLEV)=EPQ(I,NROW,NLEV)
            ENDIF
          ENDIF
          NROW=NROW+1
          GOTO 3

        ELSE
          IF(IQV(NROW,NLEV).EQ.0)THEN
            IF(IQV(LPROW,NLEV-1).EQ.0)THEN
              IF(EPQ(5,LPROW,NLEV-1).EQ.0.D0)THEN
                IPQ(2,LPROW,NLEV-1)=IPQ(2,NROW,NLEV)
                DO 24 I=1,4
24              EPQ(I+4,LPROW,NLEV-1)=EPQ(I+4,NROW,NLEV)
              ENDIF
            ELSE
              IF(EPQ(1,LPROW,NLEV-1).EQ.0.D0)THEN
                IPQ(1,LPROW,NLEV-1)=IPQ(2,NROW,NLEV)
                DO 25 I=1,4
25              EPQ(I,LPROW,NLEV-1)=EPQ(I+4,NROW,NLEV)
              ENDIF
            ENDIF
          ELSE
            IF(IQV(LPROW,NLEV-1).EQ.0.AND.
     *      EPQ(5,LPROW,NLEV-1).EQ.0.D0)THEN
                IPQ(2,LPROW,NLEV-1)=IPQ(1,NROW,NLEV)
                DO 26 I=1,4
26              EPQ(I+4,LPROW,NLEV-1)=EPQ(I,NROW,NLEV)
            ENDIF
          ENDIF

          NROW=LPROW
          NLEV=NLEV-1
          GOTO 17
        ENDIF
        END
C=======================================================================

      FUNCTION PSREJS(S,Z,IQQ)
c PSREJS - rejection function for the energy sharing for semihard
c interaction (Hi_semihard(S)/S**delh)
c S - energy squared for the semihard interaction,
c Z - impact parameter factor, Z=exp(-b**2/Rp),
c IQQ - type of the hard interaction (0 - gg, 1 - qg, 2 - gq)
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
      COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
      COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
      COMMON /Q_AR13/    X1(7),A1(7)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,Z,IQQ
201     FORMAT(2X,'PSREJS - REJECTION FUNCTION TABULATION: '/
     *  4X,'S=',E10.3,2X,'Z=',E10.3,2X,'IQQ=',I1)
      XMIN=4.D0*(QT0+AMJ0)/S
      XMIN=XMIN**(DELH-DEL)
      PSREJS=0.D0

c Numerical integration over Z1
      DO 2 I=1,7
      DO 2 M=1,2
      Z1=(.5D0*(1.D0+XMIN-(2*M-3)*X1(I)*(1.D0-XMIN)))**(1.D0/
     *(DELH-DEL))

c SJ is the inclusive hard partonic interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer squre QT0 and c.m. energy square s_hard = exp YJ;
c SJB - Born cross-section
      YJ=DLOG(Z1*S)
      CALL PSJINT0(Z1*S,SJ,SJB,IQQ,0)
c GY= Sigma_hard_tot(YJ,QT0) - total hard partonic
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s_hard = exp YJ; SH=pi*R_hard**2 (R_hard**2=4/QT0)
      GY=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB
      RH=RS0-ALF*DLOG(Z1)

      IF(IQQ.NE.0)THEN
        PSREJS=PSREJS+A1(I)*GY/(Z1*S)**DELH*Z**(RS0/RH)/RH*
     *  (1.D0-Z1)*BET
      ELSE
        ST2=0.D0
        DO 1 J=1,7
1       ST2=ST2+A1(J)*((1.D0-Z1**(.5D0*(1.D0+X1(J))))*
     *  (1.D0-Z1**(.5D0*(1.D0-X1(J)))))**BET

        PSREJS=PSREJS-A1(I)*DLOG(Z1)*GY/(Z1*S)**DELH*Z**(RS0/RH)/RH*ST2
      ENDIF
2     CONTINUE
      PSREJS=DLOG(PSREJS*(1.D0-XMIN)/Z)
        IF(DEBUG.GE.2)WRITE (MONIOU,202)PSREJS
202     FORMAT(2X,'PSREJS=',E10.3)
      RETURN
      END
C=======================================================================

        FUNCTION PSREJV(S)
c PSREJV - rejection function for the energy sharing for quark-quark hard
c interaction (sigma_hard(S)/S**delh)
c S - energy squared for the hard interaction
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S
201     FORMAT(2X,'PSREJV - REJECTION FUNCTION TABULATION: ',
     *  'S=',E10.3)
c SJ is the inclusive hard QUARK-QUARK interaction
c cross-section (inclusive cut ladder cross section) for minimal
c 4-momentum transfer squre QT0 and c.m. energy square s;
c SJB - Born cross-section
        CALL PSJINT0(S,SJ,SJB,1,1)

c GY= Sigma_hard_tot(YJ,QT0) - total hard partonic (quark-quark)
c interaction cross-section for minimal 4-momentum transfer square QT0 and
c c.m. energy square s; SH=pi*R_hard**2 (R_hard**2=4/QT0)
        GY=2.D0*SH*PSGINT((SJ-SJB)/SH*.5D0)+SJB
        PSREJV=DLOG(GY/S**DELH)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSREJV
202     FORMAT(2X,'PSREJV=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSRJINT(YJ,Z0,IQQ)
c PSRJINT - Rejection function for the energy sharing (Hi_semih(S)/S**delh)
c YJ=ln S,
c Z0 - impact parameter factor, Z0=exp(-b**2/Rp),
c IQQ - type of hard interaction (0 - gg; 1 - qg, 2 - gq; 3 - qq)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION A(3)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALF,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA23/ RJV(50)
        COMMON /Q_AREA24/ RJS(50,5,10)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)YJ,Z0,IQQ
201     FORMAT(2X,'PSRJINT - REJECTION FUNCTION INTERPOLATION:'/
     *  4X,'YJ=',E10.3,2X,'Z0=',E10.3,2X,'IQQ=',I1)
        YY=(YJ-AQT0)*2.D0
*       JY=INT(YY)
        JY=MIN(48,INT(YY))       !  modified 15.oct.03 D.H.

        IF(IQQ.EQ.3)THEN
          IF(JY.EQ.0)THEN
            PSRJINT=EXP(RJV(1))*YY+(EXP(RJV(2))-2.D0*
     *      EXP(RJV(1)))*YY*(YY-1.D0)*.5D0
          ELSE
            PSRJINT=EXP(RJV(JY)+(RJV(JY+1)-RJV(JY))*(YY-JY)
     *      +(RJV(JY+2)+RJV(JY)-2.D0*RJV(JY+1))*(YY-JY)*
     *      (YY-JY-1.D0)*.5D0)
          ENDIF
        ELSE
          Z=Z0**(RS/RS0)
          IQ=(IQQ+1)/2+1+2*(ICZ-1)
          JZ=INT(5.D0*Z)
          IF(JZ.GT.3)JZ=3
          WZ=5.D0*Z-JZ

          IF(JZ.EQ.0)THEN
            I1=2
          ELSE
            I1=1
          ENDIF

          DO 1 I=I1,3
          J1=JZ+I-1
          IF(JY.EQ.0)THEN
            A(I)=EXP(RJS(1,J1,IQ))*YY+(EXP(RJS(2,J1,IQ))-2.D0*
     *      EXP(RJS(1,J1,IQ)))*YY*(YY-1.D0)*.5D0
            IF(A(I).GT.0.D0)THEN
              A(I)=DLOG(A(I))
            ELSE
              A(I)=-80.D0
            ENDIF
          ELSE
            A(I)=RJS(JY,J1,IQ)+(RJS(JY+1,J1,IQ)-
     *      RJS(JY,J1,IQ))*(YY-JY)
     *      +(RJS(JY+2,J1,IQ)+RJS(JY,J1,IQ)-2.D0*
     *      RJS(JY+1,J1,IQ))*(YY-JY)*(YY-JY-1.D0)*.5D0
          ENDIF
1         CONTINUE

          IF(JZ.NE.0)THEN
            PSRJINT=EXP(A(1)+(A(2)-A(1))*WZ+(A(3)+A(1)-2.D0*A(2))*WZ*
     *      (WZ-1.D0)*.5D0)*Z
          ELSE
            PSRJINT=(EXP(A(2))*WZ+(EXP(A(3))-2.D0*EXP(A(2)))*WZ*
     *      (WZ-1.D0)*.5D0)*Z
            IF(PSRJINT.LE.0.D0)PSRJINT=1.D-10
          ENDIF
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSRJINT
202     FORMAT(2X,'PSRJINT=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSROOT(QLMAX,G,J)
c PSROOT - effective momentum tabulation for given set of random number
c values and maximal effective momentum QMAX values - according to the
c probability of branching: (1 - timelike Sudakov formfactor)
c QLMAX - ln QMAX/16/QTF,
c G - dzeta number (some function of ksi)
c J - type of the parton (1-g,2-q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)QLMAX,G,J
201     FORMAT(2X,'PSQINT - BRANCHING MOMENTUM TABULATION:'/
     *  4X,'QLMAX=',E10.3,2X,'G=',E10.3,2X,'J=',I1)
        QL0=0.D0
        QL1=QLMAX
        F0=-G
        F1=1.D0-G
        SUD0=-DLOG(PSUDINT(QLMAX,J))

1       QL2=QL1-(QL1-QL0)*F1/(F1-F0)
        IF(QL2.LT.0.D0)THEN
          QL2=0.D0
          F2=-G
        ELSEIF(QL2.GT.QLMAX)THEN
          QL2=QLMAX
          F2=1.D0-G
        ELSE
          F2=-DLOG(PSUDINT(QL2,J))/SUD0-G
        ENDIF

        IF(ABS(F2).GT.1.D-3)THEN
          QL0=QL1
          QL1=QL2
          F0=F1
          F1=F2
          GOTO 1
        ELSE
          PSROOT=QL2
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSROOT
202     FORMAT(2X,'PSROOT=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE QGSPSROTAT(EP,S0X,C0X,S0,C0)
c Spacial rotation to the lab. system for 4-vector EP
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4),EP1(3)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)EP,S0X,C0X,S0,C0
201     FORMAT(2X,'QGSPSROTAT - SPACIAL ROTATION:'/4X,
     *  '4-VECTOR EP=',4(E10.3,1X)/4X,'S0X=',E10.3,'C0X=',E10.3,
     *  2X,'S0=',E10.3,'C0=',E10.3)
        EP1(3)=EP(4)
        EP1(2)=EP(2)*S0+EP(3)*C0
        EP1(1)=EP(2)*C0-EP(3)*S0

        EP(2)=EP1(1)
        EP(4)=EP1(2)*S0X+EP1(3)*C0X
        EP(3)=EP1(2)*C0X-EP1(3)*S0X
        IF(DEBUG.GE.3)WRITE (MONIOU,202)EP
202     FORMAT(2X,'QGSPSROTAT: ROTATED 4-VECTOR EP=',
     *  2X,4E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSQINT(QLMAX,G,J)
c PSQINT - effective momentum interpolation for given random number G
c and maximal effective momentum QMAX
c QLMAX - ln QMAX/16/QTF,
c G - random number (0<G<1)
c J - type of the parton (1-g,2-q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION WI(3),WK(3)
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA34/ QRT(10,101,2)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)QLMAX,G,J
201     FORMAT(2X,'PSQINT - BRANCHING MOMENTUM INTERPOLATION:'/
     *  4X,'QLMAX=',E10.3,2X,'G=',E10.3,2X,'J=',I1)
        QLI=QLMAX/1.38629d0
        SUD0=1.D0/PSUDINT(QLMAX,J)
        SL=100.D0*DLOG(1.D0-G*(1.D0-SUD0))/DLOG(SUD0)
        I=INT(QLI)
        K=INT(SL)
        IF(K.GT.98)K=98
        WK(2)=SL-K
        WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
        WK(1)=1.D0-WK(2)+WK(3)
        WK(2)=WK(2)-2.D0*WK(3)
        PSQINT=0.D0

        IF(I.GT.7)I=7
        WI(2)=QLI-I
        WI(3)=WI(2)*(WI(2)-1.D0)*.5D0
        WI(1)=1.D0-WI(2)+WI(3)
        WI(2)=WI(2)-2.D0*WI(3)

        DO 1 K1=1,3
        DO 1 I1=1,3
1       PSQINT=PSQINT+QRT(I+I1,K+K1,J)*WI(I1)*WK(K1)
        IF(PSQINT.LE.0.D0)PSQINT=0.D0
        PSQINT=16.D0*QTF*EXP(PSQINT)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSQINT
202     FORMAT(2X,'PSQINT=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE PSSHAR(LS,NHP,NW,NT)
c Inelastic interaction - energy sharing procedure:
c LS - total number of  cut soft pomeron blocks (froissarons),
c NHP - total number of hard pomerons,
c NW - number of interacting projectile nucleons (excluding diffracted),
c NT - number of target nucleons in active state
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
C D.H.  REAL*16 GBH,GBH0
        DIMENSION WP(56),WM(56),WHA(4000),WHB(4000),LHA0(56),
     *  LHB0(56),IZP(56),IZT(56),WP0H(56),WM0H(56),
     *  WPP(2),WMM(2),EP3(4),LQA0(56),LQB0(56),IPC(2,2),EPC(8,2),
     *  ILA(56),ILB(56),ELA(4,56),ELB(4,56),EP(4),EP1(4)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP
        COMMON /Q_AREA2/  S,Y0,WP0,WM0
        COMMON /Q_AREA9/  LQA(56),LQB(56),NQS(1000),IAS(1000),
     *  IBS(1000),LHA(56),LHB(56),ZH(4000),IAH(4000),IBH(4000),
     *  IQH(4000),LVA(56),LVB(56)
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA12/ NSH
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA19/ AHL(5)
        COMMON /Q_AREA20/ WPPP
        COMMON /Q_AREA25/ AHV(5)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        COMMON /Q_AREA47/ NJTOT

ctp from epos
      integer ng1evt,ng2evt,ikoevt
      real    rglevt,sglevt,eglevt,fglevt,typevt
      common/c2evt/ng1evt,ng2evt,rglevt,sglevt,eglevt,fglevt,ikoevt
     *,typevt            !in epos.inc

        SAVE
        EXTERNAL PSRAN
        DATA xdiv /100.D0/

        IF(DEBUG.GE.1)WRITE (MONIOU,201)NW,NT,NHP,LS
201     FORMAT(2X,'PSSHARE - ENERGY SHARING PROCEDURE'/
     *  4X,'NUMBER OF WOUNDED PROJECTILE NUCLEONS(HADRONS) NW=',I2/
     *  4X,'NUMBER OF TARGET NUCLEONS IN THE ACTIVE STATE NT=',I2/
     *  4X,'NUMBER OF SEMIHARD BLOCKS NHP=',I3/
     *  4X,'NUMBER OF SOFT POMERON BLOCKS LS=',I3)
        NSH1=NSH
        DO 101 I=1,NW
101     LQA0(I)=LQA(I)
        DO 102 I=1,NT
102     LQB0(I)=LQB(I)

100     NSH=NSH1
        NJTOT=0
        DO 103 I=1,NW
103     LQA(I)=LQA0(I)
        DO 104 I=1,NT
104     LQB(I)=LQB0(I)
c-------------------------------------------------
c Initial nucleons (hadrons) types recording
        IF(IA(1).NE.1)THEN
c IZP(i) - i-th projectile nucleons type (proton - 2, neutron - 3)
          DO 1 I=1,NW
1         IZP(I)=INT(2.5+PSRAN(B10))
        ELSE
c IZP(1)=ICP - projectile hadron type
          IZP(1)=ICP
        ENDIF
        IF(IA(2).NE.1)THEN
c IZT(j) - j-th target nucleon type (proton - 2 or neutron - 3)
          DO 2 I=1,NT
2         IZT(I)=INT(2.5+PSRAN(B10))
        ELSE
c Target proton
         IZT(1)=2
        ENDIF
c-------------------------------------------------

c WREJ - parameter for energy sharing (to minimise rejection)
        WREJ=.0001D0

3       CONTINUE

        IF(NHP.NE.0)THEN
        IF(DEBUG.GE.3)WRITE (MONIOU,211)NHP
211     FORMAT(2X,'PSSHARE: NUMBER OF HARD POMERONS NHP=',I3)
c-------------------------------------------------
c-------------------------------------------------
c Rejection function initialization:
c-------------------------------------------------
c energy-momentum will be shared between pomerons
c according to s**DEL dependence for soft pomeron and
c according to s**DELH dependence for pomeron with hard block,
c then rejection is used according to real Sigma_hard(s) dependence.
c Rejection is expected to be minimal for the uniform energy
c distribution between pomerons ( s_hard = s / LHA(I) / LHB(J) )
          GBH0=.6D0
c NREJ - total number of rejections
          NREJ=0
          NHP1=NHP

          DO 5 IH=1,NHP1
        IF(DEBUG.GE.3)WRITE (MONIOU,212)IH
212     FORMAT(2X,'PSSHARE: GBH-INI; CONTRIBUTION FROM ',I3,
     *   '-TH HARD POMERON')
c-------------------------------------------------
c LHA(i) (LHB(j)) - total number of cut hard blocks, connected to i-th projectile
c (j-th target) nucleon (hadron);
c IAH(ih) (IBH(ih)) - number (position in array) of the projectile (target) nucleon,
c connected to ih-th hard block;
c ZH(ih) - factor exp(-R_ij/R_p) for ih-th hard block;
c IQH(ih) - type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
          IQQ=IQH(IH)
          Z=ZH(IH)
          I=IAH(IH)
          J=IBH(IH)

c Uniform energy distribution between hard pomerons
          ZA=1.D0/LHA(I)
          ZB=1.D0/LHB(J)
c SI - c.m. energy squared for one hard block
          SI=ZA*ZB*S

          IF(SI.LT.4.D0*(QT0+AMJ0))THEN
c-------------------------------------------------
c One hard pomeron is removed (the energy is insufficient to simulate
c great number of pomerons)
c-------------------------------------------------
            NHP=NHP-1
            LHA(I)=LHA(I)-1
            LHB(J)=LHB(J)-1

            IF(IQQ.EQ.1)THEN
              LVA(I)=0
            ELSEIF(IQQ.EQ.2)THEN
              LVB(J)=0
            ELSEIF(IQQ.EQ.3)THEN
              LVA(I)=0
              LVB(J)=0
            ENDIF
c Rewriting of other hard pomerons characteristics
            IF(NHP.GE.IH)THEN
              DO 4 IH1=IH,NHP
              IQH(IH1)=IQH(IH1+1)
              ZH(IH1)=ZH(IH1+1)
              IAH(IH1)=IAH(IH1+1)
4             IBH(IH1)=IBH(IH1+1)
            ENDIF
c End of removing - event will be simulated from the very beginning
c-------------------------------------------------
            GOTO 3
          ENDIF

c Total rapidity for the interaction (for one hard block)
          YI=DLOG(SI)
          IF(YI.GT.17.D0)YI=17.D0
c Rejection function normalization (on maximal available energy)
          GBH0=GBH0/PSRJINT(YI,Z,IQQ)
          GBH0 = GBH0/xdiv
5         CONTINUE
        IF(DEBUG.GE.3)WRITE (MONIOU,213)
213     FORMAT(2X,'PSSHARE: GBH-INI - END')
c-------------------------------------------------
c End of rejection function normalization
c-------------------------------------------------

c-------------------------------------------------
c LHA0(i), LHB0(j) arrays are used for energy sharing procedure
c (they define number of remained cut hard blocks connected to given nucleon from
c projectile or target respectively);
c WP, WM - arrays for the rest of light cone momenta (E+-P_l) for those
c nucleons (hadrons)
c Hard pomerons connected to valence quarks are excluded from LHA0(i), LHB0(j)
c (to be considered separetely)
6         DO 7 I=1,NW
          LHA0(I)=LHA(I)-LVA(I)
7         WP(I)=WP0

          DO 8 I=1,NT
          LHB0(I)=LHB(I)-LVB(I)
8         WM(I)=WM0

c-------------------------------------------------
c Projectile valence quarks light cone momenta are choosen according to
c 1/sqrt(x) * x**delh * (1-x)**AHV(ICZ), ICZ is the type of the projectile
          DO 10 I=1,NW
          IF(LVA(I).NE.0)THEN
9           XW=PSRAN(B10)**(1.D0/(.5D0+DELH))
            IF(PSRAN(B10).GT.(1.D0-XW)**AHV(ICZ))GOTO 9
        IF(DEBUG.GE.3)WRITE (MONIOU,214)I,XW
214     FORMAT(2X,'PSSHARE: ',I2,'-TH PROJ. NUCLEON (HADRON); LIGHT',
     *  ' CONE MOMENTUM SHARE XW=',E10.3)
c WP0H(i) -  valence quark light cone momentum for i-th projectile nucleon
            WP0H(I)=XW*WP(I)
c WP(i) - the remainder of the light cone momentum for i-th projectile nucleon
            WP(I)=WP(I)*(1.D0-XW)
          ENDIF
10        CONTINUE

c Target valence quarks light cone momenta are choosen according to
c 1/sqrt(x) * x**delh * (1-x)**AHV(2) (target nucleon)
          DO 12 I=1,NT
          IF(LVB(I).NE.0)THEN
11          XW=PSRAN(B10)**(1.D0/(.5D0+DELH))
            IF(PSRAN(B10).GT.(1.D0-XW)**AHV(2))GOTO 11
        IF(DEBUG.GE.3)WRITE (MONIOU,215)I,XW
215     FORMAT(2X,'PSSHARE: ',I2,'-TH TARGET NUCLEON (HADRON); LIGHT',
     *  ' CONE MOMENTUM SHARE XW=',E10.3)
c WM0H(i) -  valence quark light cone momentum for i-th target nucleon
            WM0H(I)=XW*WM(I)
c WM(i) - the remainder of the light cone momentum for i-th target nucleon
            WM(I)=WM(I)*(1.D0-XW)
          ENDIF
12        CONTINUE
c-------------------------------------------------

          GBH=GBH0
c-------------------------------------------------
c Cycle over all cut hard blocks
c-------------------------------------------------
          DO 18 IH=1,NHP1
c-------------------------------------------------
c IAH(ih) (IBH(ih)) - number (position in array) of the projectile (target) nucleon,
c connected to ih-th hard block;
c ZH(ih) - factor exp(-R_ij/R_p) for ih-th hard block;
c IQH(ih) - type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
          IQQ=IQH(IH)
          Z=ZH(IH)
          I=IAH(IH)
          J=IBH(IH)

          IF((IQQ-3)*(IQQ-1).EQ.0)THEN
c WHA(ih) - light cone momentum (E+P_l) for ih-th hard block
c Read out of the valence quark light cone momentum
            WHA(IH)=WP0H(I)
          ELSE
c LHA0(i) - number of remained cut hard blocks connected to i-th projectile nucleon
            LHA0(I)=LHA0(I)-1
c Energy is shared between pomerons according to s**DEL dependence for soft
c pomeron and according to s**DELH dependence for the hard block;
c AHL(ICZ) determines energetic spectrum of the leading hadronic state of
c type ICZ
            BPI=1.D0/(1.D0+AHL(ICZ)+
     *      (1.D0+DELH)*LHA0(I))
c            BPI=1.D0/(1.D0+AHL(ICZ)+(1.D0+DEL)*LQA(I)+
c     *      (1.D0+DELH)*LHA0(I))
15          XW=1.-PSRAN(B10)**BPI
c Rejection according to XW**DELH
            IF(PSRAN(B10).GT.XW**DELH)GOTO 15
c WHA(ih) - light cone momentum (E+P_l) for ih-th hard block
            WHA(IH)=WP(I)*XW
c WP(i) - the remainder of the light cone momentum for i-th projectile nucleon
            WP(I)=WP(I)*(1.D0-XW)
          ENDIF

          IF((IQQ-3)*(IQQ-2).EQ.0)THEN
c WHB(ih) - light cone momentum (E-P_l) for ih-th hard block
c Read out of the valence quark light cone momentum
            WHB(IH)=WM0H(J)
          ELSE
c Energy is shared between pomerons - in the same way as above
            LHB0(J)=LHB0(J)-1
            BPI=1.D0/(1.D0+AHL(2)+(1.D0+DELH)
     *      *LHB0(J))
c            BPI=1.D0/(1.D0+AHL(2)+(1.D0+DEL)*LQB(J)+(1.D0+DELH)
c     *      *LHB0(J))
16          XW=1.-PSRAN(B10)**BPI
            IF(PSRAN(B10).GT.XW**DELH)GOTO 16
c WHB(ih) - light cone momentum (E-P_l) for ih-th hard block
            WHB(IH)=WM(J)*XW
c WM(j) - the remainder of the light cone momentum for j-th target nucleon
            WM(J)=WM(J)*(1.D0-XW)
          ENDIF

c Invariant mass for ih-th hard block
          SW=WHA(IH)*WHB(IH)
          IF(SW.LT.4.D0*(QT0+AMJ0))THEN
c Rejection in case of insufficient mass
            NREJ=NREJ+1

            IF(NREJ.GT.30)THEN
c-------------------------------------------------
c In case of great number of rejections number of hard blocks is put down
c-------------------------------------------------
c Number of remained hard blocks
              NHP=NHP-1
              LHA(I)=LHA(I)-1
              LHB(J)=LHB(J)-1

              IF(IQQ.EQ.1)THEN
                LVA(I)=0
              ELSEIF(IQQ.EQ.2)THEN
                LVB(J)=0
              ELSEIF(IQQ.EQ.3)THEN
                LVA(I)=0
                LVB(J)=0
              ENDIF

              IF(NHP.GE.IH)THEN
                DO 17 IH1=IH,NHP
                IQH(IH1)=IQH(IH1+1)
                ZH(IH1)=ZH(IH1+1)
                IAH(IH1)=IAH(IH1+1)
17              IBH(IH1)=IBH(IH1+1)
              ENDIF
              GOTO 3
c-------------------------------------------------
c End of removing - event will be simulated from the very beginning
c-------------------------------------------------

            ELSE
              GOTO 6
            ENDIF
          ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,216)IH,WHA(IH),WHB(IH),WP(I),WM(J)
216     FORMAT(2X,'PSSHARE: ',I3,'-TH SEMIHARD BLOCK; LIGHT',
     *  ' CONE MOMENTA SHARES:',2E10.3/
     *  4X,'REMAINED LIGHT CONE MOMENTA:',2E10.3)

          YH=DLOG(SW)
c PSRINT(YH,Z,IQQ) - phi_hard(s_hard) / s_hard ** DELH;
c YH = ln s_hard;
c Z - factor exp(-R_ij/R_p) for the hard block;
c IQQ - type of the hard interaction: 0 - gg, 1 - qg, 2 - gq, 3 - qq
c Rejection function is multiplied by PSRINT(YH,Z,IQQ) for the ih-th block
          GBH=GBH*PSRJINT(YH,Z,IQQ)
          GBH = GBH * xdiv
18        CONTINUE
c End of the loop for rejection function determination
c-------------------------------------------------

c-------------------------------------------------
c Rejection procedure (due to the deviation of the  phi_hard(s_hard)
c dependence from pure powerlike  s_hard ** DELH law)
        IF(DEBUG.GE.2)WRITE (MONIOU,217)1.D0-GBH,NHP
217     FORMAT(2X,'PSSHARE: REJECTION PROBABILITY:',E10.3,
     *  2X,'NUMBER OF SEMIHARD BLOCKS:',I3)
          IF(PSRAN(B10).GT.GBH)THEN
            NREJ=NREJ+1

            IF(NREJ.GT.30)THEN
        IF(DEBUG.GE.2)WRITE (MONIOU,218)
218     FORMAT(2X,'PSSHARE: MORE THAN 30 REJECTIONS - HARD POMERON',
     *  ' NUMBER IS PUT DOWN')
c-------------------------------------------------
c In case of great number of rejections number of hard blocks is put down
c LNH - number of hard blocks to be removed
c-------------------------------------------------
              LNH=1+NHP/20
              DO 19 IHP=NHP-LNH+1,NHP
              IIH=IAH(IHP)
              JIH=IBH(IHP)
              IQQ=IQH(IHP)

              IF(IQQ.EQ.1)THEN
                LVA(IIH)=0
              ELSEIF(IQQ.EQ.2)THEN
                LVB(JIH)=0
              ELSEIF(IQQ.EQ.3)THEN
                LVA(IIH)=0
                LVB(JIH)=0
              ENDIF

              LHA(IIH)=LHA(IIH)-1
19            LHB(JIH)=LHB(JIH)-1

              NHP=NHP-LNH
              GOTO 3
c-------------------------------------------------
c End of removing - event will be simulated from the very beginning
c-------------------------------------------------
            ELSE
              GOTO 6
            ENDIF
          ENDIF

***********************************************************************
          DO 31 I=1,NW
31        LHA0(I)=LHA(I)
          DO 32 I=1,NT
32        LHB0(I)=LHB(I)
***********************************************************************

c-------------------------------------------------
c Particle production for all cut pomerons with hard blocks
c-------------------------------------------------
          DO 20 IH=1,NHP
          IQQ=IQH(IH)
          Z=ZH(IH)
          I=IAH(IH)
          J=IBH(IH)
***********************************************************************
          LHA0(I)=LHA0(I)-1
          LHB0(J)=LHB0(J)-1
***********************************************************************
c WPI, WMI - light cone momenta for current (ih-th) hard pomeron
          WPI=WHA(IH)
          WMI=WHB(IH)
        IF(DEBUG.GE.2)WRITE (MONIOU,219)IH,IQQ,WPI,WMI,WP(I),WM(J)
219     FORMAT(2X,'PSSHARE: ',I3,
     *  '-TH HARD BLOCK; TYPE OF THE INTERACTION:',I1/
     *  4X,'INITIAL LIGHT CONE MOMENTA:',2E10.3/
     *  4X,'REMAINED LIGHT CONE MOMENTA:',2E10.3)
c-------------------------------------------------
c PSHOT procedure is used for hard partonic interaction -
c initial jets simulation
          CALL PSHOT(WPI,WMI,Z,IPC,EPC,IZP(I),IZT(J),ICZ,IQQ)
          IF(IQQ.EQ.1.OR.IQQ.EQ.3)THEN
            IF((IABS(IZP(I)).GT.5.OR.IABS(IZP(I)).EQ.3).AND.
     *      IZP(I).GT.0.OR.IABS(IZP(I)).NE.3.AND.
     *      IABS(IZP(I)).LE.5.AND.IZP(I).LT.0)THEN
              JQ=1
            ELSE
              JQ=2
            ENDIF
            ILA(I)=IPC(JQ,1)
            DO 330 L=1,4
330         ELA(L,I)=EPC(L+4*(JQ-1),1)
          ENDIF
          IF(IQQ.EQ.2.OR.IQQ.EQ.3)THEN
            IF((IABS(IZT(J)).GT.5.OR.IABS(IZT(J)).EQ.3).AND.
     *      IZT(J).GT.0.OR.IABS(IZT(J)).NE.3.AND.
     *      IABS(IZT(J)).LE.5.AND.IZT(J).LT.0)THEN
              JQ=1
            ELSE
              JQ=2
            ENDIF
            ILB(J)=IPC(JQ,2)
            DO 331 L=1,4
331         ELB(L,J)=EPC(L+4*(JQ-1),2)
          ENDIF
          IF(IQQ.EQ.3.AND.ILA(I)+ILB(J).EQ.0)NIAS=J
c-------------------------------------------------
c          SW=WP(I)*WM(J)
c          IF(WP(I).LT.0.D0.OR.WM(J).LT.0.D0.OR.
c     *    SW.LT.(AM(ICZ)+AM(2))**2)THEN
c            NREJ=NREJ+1
c          write (*,*)'i,j,WP(I),WM(J),sw',i,j,WP(I),WM(J),sw
c            GOTO 100
c          ENDIF

c Leading hadronic state fragmentation is treated in the same way as low mass
c diffraction (exhitation mass is determined by secodary reggeon intercept
c dM**2~M**(-3))
          IF(LQA(I)+LHA0(I).EQ.0.AND.LQB(J)+LHB0(J).EQ.0)THEN
            IF(LVA(I).EQ.0.AND.LVB(J).EQ.0)THEN
              CALL XXDDFR(WP(I),WM(J),IZP(I),IZT(J))
            ELSEIF(LVA(I).EQ.0)THEN
              CALL XXDPR(WP(I),WM(J),IZP(I),IZT(J),1)
              IF(ILB(J).NE.0)THEN
                DO 341 L=1,4
341             EP1(L)=ELB(L,J)
                EP(1)=.5D0*WM(J)
                EP(2)=-EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILB(J)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZT(J),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ELSEIF(LVB(J).EQ.0)THEN
              CALL XXDTG(WP(I),WM(J),IZP(I),IZT(J),1)
              IF(ILA(I).NE.0)THEN
                DO 342 L=1,4
342             EP1(L)=ELA(L,I)
                EP(1)=.5D0*WP(I)
                EP(2)=EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILA(I)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZP(I),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ELSE
              IF(ILA(I).NE.0)THEN
                DO 343 L=1,4
343             EP1(L)=ELA(L,I)
                EP(1)=.5D0*WP(I)
                EP(2)=EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILA(I)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZP(I),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
              IF(ILB(J).NE.0)THEN
                DO 351 L=1,4
351             EP1(L)=ELB(L,J)
                EP(1)=.5D0*WM(J)
                EP(2)=-EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILB(J)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZT(J),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ENDIF
          ELSEIF(LQA(I)+LHA0(I).EQ.0)THEN
            IF(LVA(I).EQ.0)THEN
              CALL XXDPR(WP(I),WM(J),IZP(I),IZT(J),LQB(J)+LHB0(J))
            ELSE
              IF(ILA(I).NE.0)THEN
                DO 344 L=1,4
344             EP1(L)=ELA(L,I)
                EP(1)=.5D0*WP(I)
                EP(2)=EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILA(I)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZP(I),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ENDIF
          ELSEIF(LQB(J)+LHB0(J).EQ.0)THEN
            IF(LVB(J).EQ.0)THEN
              CALL XXDTG(WP(I),WM(J),IZP(I),IZT(J),LQA(I)+LHA0(I))
            ELSE
              IF(ILB(J).NE.0)THEN
                DO 345 L=1,4
345             EP1(L)=ELB(L,J)
                EP(1)=.5D0*WM(J)
                EP(2)=-EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILB(J)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZT(J),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ENDIF
          ENDIF
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
20        CONTINUE
c-------------------------------------------------
c End of the hard blocks loop
c-------------------------------------------------

        ELSE
c-------------------------------------------------
c Initial light cone momenta initialization in case of no one cut hard block
          DO 21 I=1,NW
21        WP(I)=WP0
          DO 22 I=1,NT
22        WM(I)=WM0
        ENDIF

        IF(LS.NE.0)THEN
c-------------------------------------------------
c The loop for all cut froissarons (blocks of soft pomerons)
c-------------------------------------------------
          DO 28 IS=1,LS
c NP=NQS(is) - number of cut pomerons in is-th block;
c IAS(is) (IBS(is)) - number (position in array) of the projectile (target) nucleon,
c connected to is-th block of soft pomerons;
c LQA(i) (LQB(j)) - total number of cut soft pomerons, connected to i-th projectile
c (j-th target) nucleon (hadron);
c WP(i) (WM(j)) - the remainder of the light cone momentum for i-th projectile
c (j-th target) nucleon (hadron);
c NP=NQS(is) - number of cut pomerons in is-th block;
c LQ1, LQ2 define the numbers of the remained cut pomerons  connected
c to given nucleons (hadrons)
          I=IAS(IS)
          J=IBS(IS)
          LQ1=LQA(I)
          LQ2=LQB(J)
          WPN=WP(I)
          WMN=WM(J)
          NP=NQS(IS)
      IF(DEBUG.GE.3)WRITE (MONIOU,222)IS,I,J,NP
222   FORMAT(2X,'PSSHARE: ',I3,'-TH SOFT POMERON BLOCK IS',
     *      ' CONNECTED TO ',I2,
     *      '-TH PROJECTILE NUCLEON'/4x,'(HADRON) AND ',I2,
     *      '-TH TARGET NUCLEON'/
     *      4X,'NUMBER OF CUT SOFT POMERONS IN THE BLOCK:',I2)
c-------------------------------------------------
c The loop for all cut pomerons in the block
          DO 27 IP=1,NP

cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c High mass diffraction - probability WPPP
14        JPP=0
cdh       IF(LQ1.EQ.1.AND.WPN.EQ.WP0.AND.PSRAN(B10).LT.WPPP)THEN
          IF(LQ1.EQ.1.AND.WPN.EQ.WP0.AND.PSRAN(B10).LT.WPPP
     *    .AND.LVB(J).EQ.0)THEN    !!!!!!!!!!!!!!!!!!so-07.03.99
c In case of only one cut soft pomeron high mass diffraction is simulated with the
c probability WPPP/2 or triple pomeron contribution - also WPPP/2 to have AGK cancell.
c - only for projectile hadron (nucleons) (for target - neglected)
c YW is the branching point position (in rapidity)
            YW=1.D0+PSRAN(B10)*(Y0-2.D0)
      IF(DEBUG.GE.3)WRITE (MONIOU,223)YW
223   FORMAT(2X,'PSSHARE: TRIPLE POMERON CONTRIBUTION YW=',E10.3)
c Light cone momentum (E+P_l) for the diffractive state (which is just usual cut
c pomeron)
            XPW=EXP(-YW)
            JPP=1
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

          ELSE
            LQ1=LQ1-1
c Energy-momentum is shared between pomerons according to s**DEL dependence for soft
c pomeron; AHL(ICZ) determines energy spectrum of leading hadronic
c state of type ICZ
            BPI=1.D0/(1.D0+AHL(ICZ)+(1.D0+DEL)*LQ1)
23          XPW=1.-PSRAN(B10)**BPI
c Rejection according to XW**DEL
            IF(PSRAN(B10).GT.XPW**DEL)GOTO 23
          ENDIF

          LQ2=LQ2-1
c Energy-momentum is shared between pomerons according to s**DEL dependence for soft
c pomeron - similar to projectile case
          BPI=1.D0/(1.D0+AHL(2)+(1.D0+DEL)*LQ2)
24        XMW=1.-PSRAN(B10)**BPI
c Rejection according to XW**DEL
          IF(PSRAN(B10).GT.XMW**DEL)GOTO 24
c-------------------------------------------------

cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c High mass diffraction is rejected in case of insufficient energy
         IF(JPP.EQ.1.AND.XPW*XMW*WPN*WMN.LT.2.72D0)THEN
            LQ2=LQ2+1
            GOTO 14
          ENDIF
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

c WPI is the light cone momentum (E+P_l) for the pomeron;
c WPN is the remainder of the light cone momentum for given nucleon (hadron)
          WPI=WPN*XPW
          WPN=WPN-WPI
          WMI=WMN*XMW
          WMN=WMN-WMI

************************************************************************
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        IF(LQ1.EQ.0.AND.LVA(I).EQ.0)THEN
          CALL IXXDEF(IZP(I),IC11,IC12,ICZ)
        ELSE
          IC11=0
          IC12=0
        ENDIF
        IF(LQ2.EQ.0.AND.LVB(J).EQ.0)THEN
          CALL IXXDEF(IZT(J),IC21,IC22,2)
        ELSE
          IC21=0
          IC22=0
        ENDIF

cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c Fragmentation process for the pomeron ( quarks and antiquarks types at the
c ends of the two strings are determined, energy-momentum is shared
c between them and strings fragmentation is simulated )
      IF(DEBUG.GE.3)WRITE (MONIOU,224)IP,WPI,WMI
224   FORMAT(2X,'PSSHARE: ',I2,'-TH SOFT POMERON IN THE BLOCK'/
     *      4X,'LIGHT CONE MOMENTA FOR THE POMERON:',2E10.3)
          CALL XXSTR(WPI,WMI,WPN,WMN,IC11,IC12,IC22,IC21)
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c Triple pomeron contribution simulation
          IF(JPP.EQ.1)THEN
            IF(PSRAN(B10).LT..5D0)THEN
              SW=WPN*WMN
              IF(WPN.LT.0.D0.OR.WMN.LT.0.D0.OR.
     *        SW.LT.(AM(ICZ)+AM(2))**2)THEN
cdh
                if (debug.ge.1)
     *            write (*,*)'difr,i,j,WPn,WMn,sw,lq1,lq2',
     *                             i,j,WPn,WMn,sw,lq1,lq2
                NREJ=NREJ+1
                GOTO 100
              ENDIF
              typevt=3       !high mass diffraction

              IF(LQ2.EQ.0)THEN
                CALL XXDTG(WPN,WMN,IZP(I),IZT(J),0)
              ELSE
                WP1=WPN
                WM1=AM(ICZ)**2/WP1
                EP3(1)=.5D0*(WP1+WM1)
                EP3(2)=.5D0*(WP1-WM1)
                EP3(3)=0.D0
                EP3(4)=0.D0
                CALL XXREG(EP3,IZP(I))
                WMN=WMN-WM1
                WPN=0.D0
              ENDIF
              GOTO 30
            ELSE

c Triple pomeron contribution simulation (both pomerons are cut)
      IF(DEBUG.GE.3)WRITE (MONIOU,225)
225   FORMAT(2X,'PSSHARE: TRIPLE POMERON CONRITRIBUTION WITH 3 CUT',
     *' POMERONS')
              WMM(1)=1.D0/WPI
              WMN=WMN-WMM(1)
c Light cone momentum (E-P_l) sharing for the two pomerons
              WMM(2)=WMM(1)*PSRAN(B10)
              WMM(1)=WMM(1)-WMM(2)
              LQ1=2
              DO 26 L=1,2
              LQ1=LQ1-1
c Light cone momentum (E+P_l) sharing for the two pomerons
              BPI=(1.D0+DEL)*LQ1+1.D0+AHL(ICZ)
              BPI=1.D0/BPI
25            XPW=1.-PSRAN(B10)**BPI
              IF(PSRAN(B10).GT.XPW**DEL)GOTO 25
              WPP(L)=WPN*XPW
              WPN=WPN*(1.D0-XPW)
c Fragmentation process for the pomerons
26            CALL XXSTR(WPP(L),WMM(L),WPN,WMN,0,0,0,0)
              SW=WPN*WMN
              IF(WPN.LT.0.D0.OR.WMN.LT.0.D0.OR.
     *        SW.LT.(AM(ICZ)+AM(2))**2)THEN
                NREJ=NREJ+1
                GOTO 100
              ENDIF
            ENDIF
          ENDIF
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
27        CONTINUE
c End of the pomeron loop
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c          SW=WPN*WMN
c          IF(WPN.LT.0.D0.OR.WMN.LT.0.D0.OR.
c     *    SW.LT.(AM(ICZ)+AM(2))**2)THEN
c            NREJ=NREJ+1
c            GOTO 100
c          ENDIF

c Leading hadronic state fragmentation is treated in the same way as low mass
c diffraction (exhitation mass is determined by secodary reggeon intercept
c dM**2~M**(-3))
          IF(LQ1.EQ.0.AND.LQ2.EQ.0)THEN
            IF(LVA(I).EQ.0.AND.LVB(J).EQ.0)THEN
              CALL XXDDFR(WPN,WMN,IZP(I),IZT(J))
            ELSEIF(LVA(I).EQ.0)THEN
              CALL XXDPR(WPN,WMN,IZP(I),IZT(J),1)
              IF(ILB(J).NE.0)THEN
                DO 346 L=1,4
346             EP1(L)=ELB(L,J)
                EP(1)=.5D0*WMN
                EP(2)=-EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILB(J)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZT(J),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ELSEIF(LVB(J).EQ.0)THEN
              CALL XXDTG(WPN,WMN,IZP(I),IZT(J),1)
              IF(ILA(I).NE.0)THEN
                DO 347 L=1,4
347             EP1(L)=ELA(L,I)
                EP(1)=.5D0*WPN
                EP(2)=EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILA(I)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZP(I),IPJ1,EP,EP1,JFL)
               IF(JFL.EQ.0)GOTO 100
              ENDIF
            ELSE
              IF(ILA(I).NE.0)THEN
                DO 348 L=1,4
348             EP1(L)=ELA(L,I)
                EP(1)=.5D0*WPN
                EP(2)=EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILA(I)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZP(I),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
              IF(ILB(J).NE.0)THEN
                DO 349 L=1,4
349             EP1(L)=ELB(L,J)
                EP(1)=.5D0*WMN
                EP(2)=-EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILB(J)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZT(J),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ENDIF

          ELSEIF(LQ1.EQ.0)THEN
            IF(LVA(I).EQ.0)THEN
              CALL XXDPR(WPN,WMN,IZP(I),IZT(J),LQ2)
            ELSE
              IF(ILA(I).NE.0)THEN
                DO 350 L=1,4
350             EP1(L)=ELA(L,I)
                EP(1)=.5D0*WPN
                EP(2)=EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILA(I)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZP(I),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ENDIF

          ELSEIF(LQ2.EQ.0)THEN
            IF(LVB(J).EQ.0)THEN
              CALL XXDTG(WPN,WMN,IZP(I),IZT(J),LQ1)
            ELSE
              IF(ILB(J).NE.0)THEN
                DO 352 L=1,4
352             EP1(L)=ELB(L,J)
                EP(1)=.5D0*WMN
                EP(2)=-EP(1)
                EP(3)=0.D0
                EP(4)=0.D0
                IPJ1=ILB(J)
                IF(IABS(IPJ1).EQ.3)IPJ1=IPJ1*4/3
                CALL PSJDEF(IZT(J),IPJ1,EP,EP1,JFL)
                IF(JFL.EQ.0)GOTO 100
              ENDIF
            ENDIF
          ENDIF
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
c-------------------------------------------------
c The numbers of the remained cut pomerons connected to given nucleons (hadrons)
c as well as the rest of the longitudinal momenta for these nucleons are
c recorded
30        LQA(I)=LQ1
          LQB(J)=LQ2
          WP(I)=WPN
28        WM(J)=WMN
        ENDIF
c-------------------------------------------------
c End of the soft blocks loop
c-------------------------------------------------
        IF(IA(1).EQ.1.AND.LVA(1).NE.0.AND.ILA(1).EQ.0)THEN
          EP(1)=.5D0*WP(1)
          EP(2)=EP(1)
          EP(3)=0.D0
          EP(4)=0.D0
          EP1(1)=.5D0*WM(NIAS)
          EP1(2)=-EP1(1)
          EP1(3)=0.D0
          EP1(4)=0.D0
          CALL PSJDEF(IZP(1),IZT(NIAS),EP,EP1,JFL)
          IF(JFL.EQ.0)GOTO 100
        ENDIF
cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        CALL XXJETSIM
************************************************************************
      IF(DEBUG.GE.3)WRITE (MONIOU,227)
227   FORMAT(2X,'PSSHARE - END')
        RETURN
        END
C=======================================================================

      SUBROUTINE QGSPSTRANS(EP,EY)
c Lorentz transform according to parameters EY ( determining Lorentz shift
c along the Z,X,Y-axis respectively (EY(1),EY(2),EY(3)))
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EY(3),EP(4)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)EP,EY
201     FORMAT(2X,'QGSPSTRANS - LORENTZ BOOST FOR 4-VECTOR'/4X,'EP=',
     *  2X,4(E10.3,1X)/4X,'BOOST PARAMETERS EY=',3E10.3)
c Lorentz transform to lab. system according to 1/EY(i) parameters
        DO 1 I=1,3
        IF(EY(4-I).NE.1.D0)THEN
          WP=(EP(1)+EP(5-I))/EY(4-I)
          WM=(EP(1)-EP(5-I))*EY(4-I)
          EP(1)=.5D0*(WP+WM)
          EP(5-I)=.5D0*(WP-WM)
        ENDIF
1       CONTINUE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)EP
202     FORMAT(2X,'QGSPSTRANS: TRANSFORMED 4-VECTOR EP=',
     *  2X,4(E10.3,1X))
        RETURN
        END
C=======================================================================

      SUBROUTINE QGSPSTRANS1(EP,EY)
c Lorentz transform according to parameters EY ( determining Lorentz shift
c along the Z,X,Y-axis respectively (EY(1),EY(2),EY(3)))
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EY(3),EP(4)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)EP,EY
201     FORMAT(2X,'QGSPSTRANS1 - LORENTZ BOOST FOR 4-VECTOR'/4X,'EP=',
     *  2X,4(E10.3,1X)/4X,'BOOST PARAMETERS EY=',3E10.3)
c Lorentz transform to lab. system according to 1/EY(i) parameters
          DO 2 I=1,3
          IF(EY(I).NE.1.D0)THEN
            WP=(EP(1)+EP(I+1))*EY(I)
            WM=(EP(1)-EP(I+1))/EY(I)
            EP(1)=.5D0*(WP+WM)
            EP(I+1)=.5D0*(WP-WM)
          ENDIF
2         CONTINUE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)EP
202     FORMAT(2X,'QGSPSTRANS1: TRANSFORMED 4-VECTOR EP=',
     *  2X,4(E10.3,1X))
        RETURN
        END
C=======================================================================

        FUNCTION PSUDINT(QLMAX,J)
c PSUDINT - timelike Sudakov formfactor interpolation
c QLMAX - ln QMAX/16/QTF,
c J - type of the parton (0-g,1-q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION WK(3)
        COMMON /Q_AREA33/ FSUD(10,2)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)J,QLMAX
201     FORMAT(2X,'PSUDINT - SPACELIKE FORM FACTOR INTERPOLATION:'/
     *  4X,'PARTON TYPE J=',
     *  I1,2X,'MOMENTUM LOGARITHM QLMAX=',E10.3)
        QL=QLMAX/1.38629d0

        IF(QL.LE.0.D0)THEN
          PSUDINT=1.D0
        ELSE
          K=INT(QL)
          IF(K.GT.7)K=7
          WK(2)=QL-K
          WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
          WK(1)=1.D0-WK(2)+WK(3)
          WK(2)=WK(2)-2.D0*WK(3)

          PSUDINT=0.D0
          DO 1 K1=1,3
1         PSUDINT=PSUDINT+FSUD(K+K1,J)*WK(K1)
          IF(PSUDINT.LE.0.D0)PSUDINT=0.D0
          PSUDINT=EXP(-PSUDINT)
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSUDINT
202     FORMAT(2X,'PSUDINT=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION QGSPSUDS(Q,J)
c QGSPSUDS - spacelike Sudakov formfactor
c Q - maximal value of the effective momentum,
c J - type of parton (0 - g, 1 - q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA6/  PI,BM,AM
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)J,Q
201     FORMAT(2X,'QGSPSUDS - SPACELIKE FORM FACTOR: PARTON TYPE J=',
     *  I1,2X,'MOMENTUM Q=',E10.3)
        IF(Q.GT.QT0)THEN
          QLM=DLOG(Q/ALM)
          QGSPSUDS=(QLM*DLOG(QLM/QLOG)-DLOG(Q/QT0))/9.D0

          IF(J.EQ.0)THEN
            QGSPSUDS=QGSPSUDS*6.D0
          ELSE
            QGSPSUDS=QGSPSUDS/.375D0
          ENDIF
          QGSPSUDS=EXP(-QGSPSUDS)

        ELSE
          QGSPSUDS=1.D0
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)QGSPSUDS
202     FORMAT(2X,'QGSPSUDS=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSUDT(QMAX,J)
c PSUDT - timelike Sudakov formfactor
c QMAX - maximal value of the effective momentum,
c J - type of parton (0 - g, 1 - q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON/AR13/X1(7),A1(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)J,QMAX
201     FORMAT(2X,'PSUDT - TIMELIKE FORM FACTOR: PARTON TYPE J=',
     *  I1,2X,'MOMENTUM QMAX=',E10.3)
        PSUDT=0.D0
        QLMAX=DLOG(DLOG(QMAX/16.D0/ALM))
        QFL=DLOG(DLOG(QTF/ALM))

c Numerical integration over transverse momentum square;
c Gaussian integration is used
          DO 1 I=1,7
          DO 1 M=1,2
          QTL=.5D0*(QLMAX+QFL+(2*M-3)*X1(I)*(QLMAX-QFL))
          QT=ALM*EXP(EXP(QTL))
          IF(QT.GE.QMAX/16.D0)QT=QMAX/16.0001D0
          ZMIN=.5D0-DSQRT((.25D0-DSQRT(QT/QMAX)))
          ZMAX=1.D0-ZMIN
          IF(J.EQ.0)THEN
******************************************************
            AP=(PSAPINT(ZMAX,0,0)-PSAPINT(ZMIN,0,0)+
     *      PSAPINT(ZMAX,0,1)-PSAPINT(ZMIN,0,1))*.5D0
******************************************************
          ELSE
            AP=PSAPINT(ZMAX,1,0)-PSAPINT(ZMIN,1,0)
          ENDIF
1         PSUDT=PSUDT+A1(I)*AP
          PSUDT=PSUDT*(QLMAX-QFL)/9.D0
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSUDT
202     FORMAT(2X,'PSUDT=',E10.3)
        RETURN
        END
C=======================================================================

        FUNCTION PSV(X,Y,XB,IB)
c XXV - eikonal dependent factor for hadron-nucleus interaction
c (used for total and diffractive hadron-nucleus cross-sections calculation)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
cdh     DIMENSION XB(56,3),FHARD(3)
        DIMENSION XB(210,3),FHARD(3)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)X,Y,IB
201     FORMAT(2X,'PSV - EIKONAL FACTOR: NUCLEON COORDINATES X=',
     *  E10.3,2X,'Y=',E10.3/4X,'NUMBER OF ACTIVE TARGET NUCLEONS IB='
     *  ,I2)
        DV=0.D0
c????????????????????????????????????????????
        DO 1 M=1,IB
        Z=PSDR(X-XB(M,1),Y-XB(M,2))
        DV=DV+PSFAZ(Z,FSOFT,FHARD,FSHARD)+FHARD(1)+FHARD(2)+FHARD(3)
1       CONTINUE
        PSV=(1.D0-EXP(-DV))**2

C       DH=1.D0
C       DO 1 M=1,IB
C       Z=PSDR(X-XB(M,1),Y-XB(M,2))
C       DV=DV+PSFAZ(Z,FSOFT,FHARD,FSHARD)
C 1     DH=DH*(1.D0-FHARD(1)-FHARD(2)-FHARD(3))
c????????????????????????????????????????????????
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSV
202     FORMAT(2X,'PSV=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE PSVDEF(ICH,IC1,ICZ)
c Determination of valence quark flavour -
c for valence quark hard scattering
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)ICH,ICZ
201     FORMAT(2X,'PSVDEF: HADRON TYPE ICH=',I2,' AUXILLIARY TYPE ICZ='
     *  ,I1)

        IS=IABS(ICH)/ICH
        IF(ICZ.EQ.1)THEN
          IC1=ICH*(1-3*INT(.5+PSRAN(B10)))
          ICH=-IC1-ICH
        ELSEIF(ICZ.EQ.2)THEN
          IF(PSRAN(B10).GT..33333D0.OR.ICH.LT.0)THEN
            IC1=ICH-IS
            ICH=3*IS
          ELSE
            IC1=4*IS-ICH
            ICH=ICH+4*IS
          ENDIF
        ELSEIF(ICZ.EQ.3)THEN
          IC1=ICH-3*IS
          ICH=-4*IS
        ELSEIF(ICZ.EQ.4)THEN
          IC1=ICH-9*IS
          ICH=5*IS
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)IC1,ICH
202     FORMAT(2X,'PSVDEF-END: QUARK FLAVOR IC1=',I2,
     *  'DIQUARK TYPE ICH=',I2)
        RETURN
        END
C=======================================================================

        FUNCTION PSZSIM(QQ,J)
c PSZSIM - light cone momentum share simulation (for the timelike
c branching)
c QQ - effective momentum value,
c J - type of the parent parton (0-g,1-q)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA18/ ALM,QT0,QLOG,QLL,AQT0,QTF,BET,AMJ0
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN


        IF(DEBUG.GE.2)WRITE (MONIOU,201)QQ,J
201     FORMAT(2X,'PSZSIM - Z-SHARE SIMULATION: QQ=',E10.3,2X,'J=',I1)
        ZMIN=.5D0-DSQRT(.25D0-DSQRT(QTF/QQ))
        QLF=DLOG(QTF/ALM)

1       CONTINUE
        IF(J.EQ.1)THEN
          PSZSIM=.5D0*(2.D0*ZMIN)**PSRAN(B10)
******************************************************
          GB=PSZSIM*(QGSPSFAP(PSZSIM,0,0)+QGSPSFAP(PSZSIM,0,1))/7.5D0
******************************************************
        ELSE
          PSZSIM=ZMIN*((1.D0-ZMIN)/ZMIN)**PSRAN(B10)
          GB=PSZSIM*QGSPSFAP(PSZSIM,1,0)*.375D0
        ENDIF
        QT=QQ*(PSZSIM*(1.D0-PSZSIM))**2
        GB=GB/DLOG(QT/ALM)*QLF
        IF(DEBUG.GE.3)WRITE (MONIOU,203)QT,GB
203     FORMAT(2X,'PSZSIM: QT=',E10.3,2X,'GB=',E10.3)
        IF(PSRAN(B10).GT.GB)GOTO 1
        IF(DEBUG.GE.3)WRITE (MONIOU,202)PSZSIM
202     FORMAT(2X,'PSZSIM=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE IXXDEF(ICH,IC1,IC2,ICZ)
c Determination of parton flavours in forward and backward direction -
c for valence quark hard scattering
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)ICH,ICZ
201     FORMAT(2X,'IXXDEF: HADRON TYPE ICH=',I2,' AUXILLIARY TYPE ICZ='
     *  ,I1)
        IS=IABS(ICH)/ICH
        IF(ICZ.EQ.1)THEN
          IC1=ICH*(1-3*INT(.5+PSRAN(B10)))
          ICH1=ICH*INT(.5D0+PSRAN(B10))
          IC2=-IC1*IABS(ICH1)-(ICH+IC1)*IABS(ICH-ICH1)

        ELSEIF(ICZ.EQ.2)THEN
c Valence quark type simulation ( for the proton )
          IC1=INT(1.3333+PSRAN(B10))
c Leading nucleon type simulation ( flavors combinatorics )
          ICH1=(2-IC1)*INT(PSRAN(B10)+.5)+2
c The type of the parton at the end of the rest string ( after the
c leading nucleon ejection )
          IC2=(3-ICH1)*(2-IC1)-2

          IF(IABS(ICH).EQ.3)THEN
            IC1=3-IC1
            IC2=-3-IC2
            ICH1=5-ICH1
          ENDIF
          IF(ICH.LT.0)THEN
            IC1=-IC1
            IC2=-IC2
            ICH1=-ICH1
          ENDIF

        ELSEIF(ICZ.EQ.3)THEN
          IC1=ICH-3*IS
          IC2=-IS*INT(1.5+PSRAN(B10))
          ICH1=3*IS-IC2
        ELSEIF(ICZ.EQ.4)THEN
          IC1=ICH-9*IS
          IC2=IS*INT(1.5+PSRAN(B10))
          ICH1=9*IS-IC2
        ELSEIF(ICZ.EQ.5)THEN
          IC1=IS*INT(1.5+PSRAN(B10))
          IC2=-IC1
          ICH1=ICH
        ENDIF

        ICH=ICH1
        IF(DEBUG.GE.3)WRITE (MONIOU,202)IC1,IC2,ICH
202     FORMAT(2X,'IXXDEF-END: PARTON FLAVORS IC1=',I2,' IC2=',I2,
     *  'NEW HADRON TYPE ICH=',I2)
        RETURN
        END
C=======================================================================

      FUNCTION IXXSON(NS,AW,G)
c Poisson distribution:
c AW - average value,
c NS-1 - maximal allowed value,
c G - random number
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)AW,NS-1,G
201     FORMAT(2X,'IXXSON - POISSON DITR.: AVERAGE AW=',E10.3,
     *  ' MAXIMAL VALUE NS=',I2,' RANDOM NUMBER G=',E10.3)
      W=EXP(-AW)
        SUMM=W
        DO 1 I=1,NS
        J = I
        IF(G.LT.SUMM)GOTO 2
        W=W*AW/I
1       SUMM=SUMM+W
2       IXXSON=J-1
        IF(DEBUG.GE.3)WRITE (MONIOU,202)IXXSON
202     FORMAT(2X,'IXXSON=',I2)
        RETURN
        END
C=======================================================================

      SUBROUTINE XXAINI(E0N,ICP0,IAP,IAT)
c Additional initialization procedure
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
******************************************************
      DIMENSION WK(3),WA(3)
******************************************************
      COMMON /Q_AREA1/  IA(2),ICZ,ICP
      COMMON /Q_AREA2/  S,Y0,WP0,WM0
      COMMON /Q_AREA4/  EY0(3)
      COMMON /Q_AREA5/  RD(2),CR1(2),CR2(2),CR3(2)
      COMMON /Q_AREA6/  PI,BM,AM
      COMMON /Q_AREA7/  RP1
      COMMON /Q_AREA10/ STMASS,AM0,AMN,AMK,AMC,AMLAMC,AMLAM,AMETA
      COMMON /Q_AREA15/ FP(5),RQ(5),CD(5)
      COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
      COMMON /Q_AREA22/ SJV,FJS(5,3)
      COMMON /Q_AREA35/  SJV0(10,5),FJS0(10,5,15)
      COMMON /Q_AREA43/ MONIOU
******************************************************
      COMMON /Q_AREA44/ GZ(10,5,4),GZP(10,5,4)
      COMMON /Q_AREA45/ GDT,GDP   !so00
******************************************************
      COMMON /Q_DEBUG/  DEBUG
      COMMON /Q_QGSNEX1/ XA,XB,BQGS,BMAXQGS,BMAXNEX,BMINNEX   !ctp
      DIMENSION XA(210,3),XB(210,3)
      SAVE

        IF(DEBUG.GE.1)WRITE (MONIOU,201)ICP0,IAP,IAT,E0N
201     FORMAT(2X,'XXAINI - MINIINITIALIZATION: PARTICLE TYPE ICP0=',
     *  I1,2X,'PROJECTILE MASS NUMBER IAP=',I2/4X,
     *  'TARGET MASS NUMBER IAT=',I2,' INTERACTION ENERGY E0N=',E10.3)
      ICP=ICP0
      IA(1)=IAP
      IA(2)=IAT
c ICZ - auxiliary type for the primary particle (1- pion, 2 - nucleon, 3 - kaon,
c 4 - D-meson, 5 - Lambda_C)
      IF(IABS(ICP).LT.6)THEN
        ICZ=IABS(ICP)/2+1
      ELSE
        ICZ=(IABS(ICP)+1)/2
      ENDIF

c Energy dependent factors:
c WP0, WM0 - initial light cone momenta for the interaction (E+-p)
      S=2.D0*E0N*AMN
      WP0=DSQRT(S)
      WM0=WP0
c Y0 - total rapidity range for the interaction
      Y0=DLOG(S)
c RS - soft pomeron elastic scattering slope (lambda_ab)
      RS=RQ(ICZ)+ALFP*Y0
c RS0 - initial slope (sum of the pomeron-hadron vertices slopes squared - R_ab)
      RS0=RQ(ICZ)
c FS - factor for pomeron eikonal calculation (gamma_ab * s**del /lambda_ab * C_ab
      FS=FP(ICZ)*EXP(Y0*DEL)/RS*CD(ICZ)
c RP1 - factor for the impact parameter dependence of the eikonal ( in fm^2 )
      RP1=RS*4.D0*.0391D0/AM**2

      EY0(2)=1.D0
      EY0(3)=1.D0
      EY0(1)=DSQRT(AMN/E0N/2.D0)

c-------------------------------------------------
c Nuclear radii and weights for nuclear configurations simulation - procedure GEA
      DO 1 I=1,2
c RD(I) - Wood-Saxon density radius (fit to the data of Murthy et al.)
      RD(I)=0.7D0*FLOAT(IA(I))**.446/AM
      CR1(I)=1.D0+3.D0/RD(I)+6.D0/RD(I)**2+6.D0/RD(I)**3
      CR2(I)=3.D0/RD(I)
      CR3(I)=3.D0/RD(I)+6.D0/RD(I)**2
      IF(IA(I).LT.10.AND.IA(I).NE.1)THEN
c RD(I) - gaussian density radius (for light nucleus)
        RD(I)=.9D0*FLOAT(IA(I))**.3333/AM
        IF(IA(I).EQ.2)RD(I)=3.16D0
c RD -> RD * A / (A-1) - to use Van Hove simulation method - procedure GEA
        RD(I)=RD(I)*DSQRT(2.D0*IA(I)/(IA(I)-1.))
      ENDIF
1     CONTINUE

      GDT=0.D0
c-------------------------------------------------
c Impact parameter cutoff setting
c-------------------------------------------------
      IF(IA(1).NE.1)THEN
c Primary nucleus:
c Impact parameter cutoff value ( only impact parameters less than BM are
c simulated; probability to have larger impact parameter is less than 1% )
        BM=RD(1)+RD(2)+5.D0
      ELSE
c Hadron-nucleus interaction
c BM - impact parameter cutoff value
        BM=RD(2)+5.D0
      ENDIF

      BMAXQGS=BM*AM                 !ctp

      YE=DLOG10(E0N)
      IF(YE.LT.1.D0)YE=1.D0
      JE=INT(YE)
      IF(JE.GT.8)JE=8

******************************************************
      WK(2)=YE-JE
      WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
      WK(1)=1.D0-WK(2)+WK(3)
      WK(2)=WK(2)-2.D0*WK(3)

      SJV=SJV0(JE,ICZ)*WK(1)+SJV0(JE+1,ICZ)*WK(2)+SJV0(JE+2,ICZ)*WK(3)

      DO 2 I=1,5
      DO 2 M=1,3
      M1=M+3*(ICZ-1)
2     FJS(I,M)=FJS0(JE,I,M1)*WK(1)+FJS0(JE+1,I,M1)*WK(2)+
     *FJS0(JE+2,I,M1)*WK(3)

      GDT=0.D0
      GDP=0.D0  !so00
      IF(IA(1).EQ.1)THEN
        YA=IA(2)
        YA=DLOG(YA)/1.38629D0+1.D0
        JA=MIN(INT(YA),2)
        WA(2)=YA-JA
        WA(3)=WA(2)*(WA(2)-1.D0)*.5D0
        WA(1)=1.D0-WA(2)+WA(3)
        WA(2)=WA(2)-2.D0*WA(3)
        DO 3 I=1,3
        DO 3 M=1,3
        GDP=GDP+GZP(JE+I-1,ICZ,JA+M-1)*WK(I)*WA(M)  !so00
3       GDT=GDT+GZ(JE+I-1,ICZ,JA+M-1)*WK(I)*WA(M)
      ENDIF
c        write (*,*)'gdt=',gdt
******************************************************

        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXAINI - END')
      RETURN
      END
C=======================================================================

      SUBROUTINE XXASET
c Particular model parameters setting
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      CHARACTER *2 TYQ
      COMMON /Q_AREA3/  RMIN,EMAX,EEV
      COMMON /Q_AREA6/  PI,BM,AM
      COMMON /Q_AREA8/  WWM,BE(4),DC(5),DETA,ALMPT
      COMMON /Q_AREA10/ STMASS,AM0,AMN,AMK,AMC,AMLAMC,AMLAM,AMETA
      COMMON /Q_AREA11/ B10
      COMMON /Q_AREA20/ WPPP
      COMMON /Q_AREA21/ DMMIN(5)
      COMMON /Q_AREA28/ ARR(4)
      COMMON /Q_AREA40/ JDIFR
      COMMON /Q_AREA42/ TYQ(15)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE

        IF(DEBUG.GE.1)WRITE (MONIOU,201)
201     FORMAT(2X,'XXASET - HADRONIZATION PARAMETERS SETTING')
c Regge intercepts for the uu~, qqq~q~, us~, uc~ trajectories
      ARR(1)=0.5D0
      ARR(2)=-.5D0
      ARR(3)=0.D0
      ARR(4)=-2.D0
c WPPP - Triple pomeron interaction probability (for two cut pomerons and cut
c between them)
      WPPP=0.4d0
c JDIFR - flag for the low mass diffraction (for JDIFR=0 not considered)
      JDIFR=1

c-------------------------------------------------
c Parameters for the soft fragmentation:
c DC(i) - relative probabilities for udu~d~(i=1), ss~(i=2), cc~(i=3)-pairs creation
c from the vacuum for the quark (u,d,u~,d~) fragmentation;
c ss~(i=4), cc~(i=5) - for the diquark (ud, u~d~) fragmentation
      DC(1)=.06D0
      DC(2)=.10D0
*     DC(3)=.0003D0     ! To switch off charmed particles set to 0.000
      DC(3)=.000D0
      DC(4)=.36D0
*     DC(5)=.01D0     ! To switch off charmed particles set to 0.000
      DC(5)=.0D0
cc  DETA - ratio of etas production density to all pions production density (1/9)
      DETA=.11111D0
c WWM defines mass threshold for string to decay into three or more hadrons
c ( ajustable parameter for string fragmentation )
      WWM=.53D0
c BE(i) - parameter for Pt distribution (exponential) for uu~(dd~), ss~, qqq~q~,
c cc~ pairs respectively (for the soft fragmentation)
      BE(1)=.22D0
      BE(2)=.35D0
      BE(3)=.29D0
      BE(4)=.40D0
c ALMPT - parameter for the fragmentation functions (soft ones):
c ALMPT = 1 + 2 * alfa_R * <pt**2> (Kaidalov proposed 0.5 value for ALMPT-1,
c Sov.J.Nucl.Phys.,1987))
      ALMPT=1.7D0

c-------------------------------------------------
c Parameters for nuclear spectator part fragmentation:
c RMIN - coupling radius squared (fm^2),
c EMAX - relative critical energy ( divided per mean excitation energy (~12.5 Mev)),
c EEV - relative evaporation energy ( divided per mean excitation energy (~12.5 Mev))
      RMIN=3.35D0
      EMAX=.11D0
      EEV=.25D0

c-------------------------------------------------
c DMMIN(i) - minimal diffractive mass for low-mass diffraction for pion, nucleon,
c kaon, D-meson, Lambda_C corresp.
      DMMIN(1)=.76D0
      DMMIN(2)=1.24D0
      DMMIN(3)=.89D0
      DMMIN(4)=2.01D0
      DMMIN(5)=2.45D0
c Proton, kaon, pion, D-meson, Lambda, Lambda_C, eta masses
      AMN=.939D0
      AMK=.496D0
      AM0=.14D0
      AMC=1.868D0
      AMLAM=1.116D0
      AMLAMC=2.27D0
      AMETA=.548D0

c-------------------------------------------------
c B10 - initial value of the pseudorandom number,
c PI  - pi-number
c AM  - diffusive radius for the Saxon-Wood nuclear density parametrization
      B10=.43876194D0
      PI=3.1416D0
      AM=.523D0

C STMASS - minimal string mass to produce secondary particles
      STMASS=4.D0*AM0**2
c Here and below all radii, distances and so on are divided by AM.
      RMIN=RMIN/AM**2

      TYQ(1)='DD'
      TYQ(2)='UU'
      TYQ(3)='C '
      TYQ(4)='S '
      TYQ(5)='UD'
      TYQ(6)='D '
      TYQ(7)='U '
      TYQ(8)='G '
      TYQ(9)='u '
      TYQ(10)='d '
      TYQ(11)='ud'
      TYQ(12)='s '
      TYQ(13)='c '
      TYQ(14)='uu'
      TYQ(15)='dd'
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXASET - END')
      RETURN
      END
C=======================================================================

        SUBROUTINE XXDDFR(WP0,WM0,ICP,ICT)
c Double diffractive dissociation
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP3(4),EY(3)!,EP1(4),EP2(4)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP0
        COMMON /Q_AREA2/  S,Y0,WP00,WM00
        COMMON /Q_AREA8/  WWM,BE(4),DC(5),DETA,ALMPT
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA21/ DMMIN(5)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)ICP,ICT,WP0,WM0
201     FORMAT(2X,'XXDDFR - LEADING CLUSTERS HADRONIZATION:'
     *  /4X,'CLUSTER TYPES ICP=',I2,2X,
     *  'ICT=',I2/4X,'AVAILABLE LIGHT CONE MOMENTA: WP0=',E10.3,
     *  ' WM0=',E10.3)
        DO 100 I=1,3
100     EY(I)=1.D0

        SD0=WP0*WM0
        IF(SD0.LT.0.D0)SD0=0.D0
        DDMIN1=DMMIN(ICZ)
        DDMIN2=DMMIN(2)
        DDMAX1=MIN(5.D0,DSQRT(SD0)-DDMIN2)

        IF(DDMAX1.LT.DDMIN1)THEN
c Registration of too slow "leading" hadron if its energy is insufficient for
c diffractive exhitation
          IF(DSQRT(SD0).LT.AM(ICZ)+AM(2))THEN
            IF(WP0.GT.0.D0.AND.(AM(ICZ)+AM(2))**2/WP0.LT..5D0*WM00)THEN
              SD0=(AM(ICZ)+AM(2))**2
              WM0=SD0/WP0
            ELSE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
              RETURN
            ENDIF
          ENDIF

          EP3(3)=0.D0
          EP3(4)=0.D0
          XW=XXTWDEC(SD0,AM(ICZ)**2,AM(2)**2)
          WP1=XW*WP0
          WM1=AM(ICZ)**2/WP1
          EP3(1)=.5D0*(WP1+WM1)
          EP3(2)=.5D0*(WP1-WM1)
          CALL XXREG(EP3,ICP)
          WM2=WM0-WM1
          WP2=AM(2)**2/WM2
          EP3(1)=.5D0*(WP2+WM2)
          EP3(2)=.5D0*(WP2-WM2)
          CALL XXREG(EP3,ICT)
          WP0=0.D0
          WM0=0.D0
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
          RETURN
        ENDIF

        DMASS1=(DDMIN1/(1.D0-PSRAN(B10)*(1.D0-DDMIN1/DDMAX1)))**2
        DDMAX2=MIN(5.D0,DSQRT(SD0)-DSQRT(DMASS1))
        DMASS2=(DDMIN2/(1.D0-PSRAN(B10)*(1.D0-DDMIN2/DDMAX2)))**2

        WPD1=WP0*XXTWDEC(SD0,DMASS1,DMASS2)
        WMD1=DMASS1/WPD1
        WMD2=WM0-WMD1
        WPD2=DMASS2/WMD2

        IF(ICP.NE.0)IS=IABS(ICP)/ICP
        IF(ICZ.EQ.5)THEN
          ICH1=ICP
          ICH2=0
          AMH1=AM(5)**2
          AMH2=AM(1)**2

          PTMAX=QGSPSLAM(DMASS1,AMH1,AMH2)
          IF(PTMAX.LT.0.)PTMAX=0.
          IF(PTMAX.LT.BE(4)**2)THEN
1           PTI=PTMAX*PSRAN(B10)
            IF(PSRAN(B10).GT.EXP(-DSQRT(PTI)/BE(4)))GOTO 1
          ELSE
2           PTI=(BE(4)*DLOG(PSRAN(B10)*PSRAN(B10)))**2
            IF(PTI.GT.PTMAX)GOTO 2
          ENDIF
          AMT1=AMH1+PTI
          AMT2=AMH2+PTI
          Z=XXTWDEC(DMASS1,AMT1,AMT2)
          WP1=WPD1*Z
          WM1=AMT1/WP1
          EP3(1)=.5D0*(WP1+WM1)
          EP3(2)=.5D0*(WP1-WM1)
          PT=DSQRT(PTI)
          CALL QGSPSCS(C,S)
          EP3(3)=PT*C
          EP3(4)=PT*S
          CALL XXREG(EP3,ICH1)

          WP1=WPD1*(1.D0-Z)
          WM1=AMT2/WP1
          EP3(1)=.5D0*(WP1+WM1)
          EP3(2)=.5D0*(WP1-WM1)
          EP3(3)=-PT*C
          EP3(4)=-PT*S
          CALL XXREG(EP3,ICH2)
          GOTO 3
        ENDIF

        IF(ICZ.EQ.1)THEN
          IF(ICP.NE.0)THEN
            IC1=ICP*(1-3*INT(.5D0+PSRAN(B10)))
            IC2=-ICP-IC1
          ELSE
            IC1=INT(1.5D0+PSRAN(B10))*(2*INT(.5D0+PSRAN(B10))-1)
            IC2=-IC1
          ENDIF
        ELSEIF(ICZ.EQ.2)THEN
          IF(PSRAN(B10).GT..33333D0)THEN
            IC1=3*IS
            IC2=ICP-IS
          ELSE
            IC1=ICP+4*IS
            IC2=4*IS-ICP
          ENDIF
        ELSEIF(ICZ.EQ.3)THEN
          IC1=-4*IS
          IC2=ICP-3*IS
        ELSEIF(ICZ.EQ.4)THEN
          IC1=5*IS
          IC2=ICP-9*IS
        ENDIF
        CALL XXGENER(WPD1,WMD1,EY,0.D0,1.D0,0.D0,1.D0,IC1,IC2)

3       CONTINUE
        IS=IABS(ICT)/ICT
        IF(PSRAN(B10).GT..33333D0)THEN
          IC1=3*IS
          IC2=ICT-IS
        ELSE
          IC1=ICT+4*IS
          IC2=4*IS-ICT
        ENDIF
        CALL XXGENER(WPD2,WMD2,EY,0.D0,1.D0,0.D0,1.D0,IC2,IC1)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXDDFR - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXDEC2(EP,EP1,EP2,WW,A,B)
c Two particle decay
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        dimension ep(4),ep1(4),ep2(4),EY(3)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        COMMON /Q_AREA11/ B10
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'XXDEC2 - TWO PARTICLE DECAY')

        PL=QGSPSLAM(WW,A,B)
        EP1(1)=DSQRT(PL+A)
        EP2(1)=DSQRT(PL+B)
        PL=DSQRT(PL)
        COSZ=2.D0*PSRAN(B10)-1.D0
        PT=PL*DSQRT(1.D0-COSZ**2)
        EP1(2)=PL*COSZ
        CALL QGSPSCS(C,S)
        EP1(3)=PT*C
        EP1(4)=PT*S
        do 1 I=2,4
1       EP2(I)=-EP1(I)
        CALL QGSPSDEFTR(WW,EP,EY)
        CALL QGSPSTRANS(EP1,EY)
        CALL QGSPSTRANS(EP2,EY)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXDEC2 - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXDEC3(EP,EP1,EP2,EP3,SWW,AM1,AM2,AM3)

c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4),EP1(4),EP2(4),EP3(4),EPT(4),EY(3)
        COMMON/AREA11/B10
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'XXDEC3 - THREE PARTICLE DECAY')
        AM12=AM1**2
        AM23=(AM2+AM3)**2
        AM32=(AM2-AM3)**2
        S23MAX=(SWW-AM1)**2
        EMAX=.25D0*(SWW+(AM12-AM23)/SWW)**2
        GB0=DSQRT((EMAX-AM12)/EMAX*(1.D0-AM23/S23MAX)
     *  *(1.D0-AM32/S23MAX))
1       P1=PSRAN(B10)*(EMAX-AM12)
        E1=DSQRT(P1+AM12)
        S23=SWW**2+AM12-2.D0*E1*SWW
        GB=DSQRT(P1*(1.D0-AM23/S23)*(1.D0-AM32/S23))/E1/GB0
        IF(PSRAN(B10).GT.GB)GOTO 1

        P1=DSQRT(P1)
        EP1(1)=E1
        COSZ=2.D0*PSRAN(B10)-1.D0
        PT=P1*DSQRT(1.D0-COSZ**2)
        EP1(2)=P1*COSZ
        CALL QGSPSCS(C,S)
        EP1(3)=PT*C
        EP1(4)=PT*S
        do 2 I=2,4
2       EPT(I)=-EP1(I)
        EPT(1)=SWW-EP1(1)
        CALL QGSPSDEFTR(SWW**2,EP,EY)
        CALL QGSPSTRANS(EP1,EY)
        CALL QGSPSTRANS(EPT,EY)

        CALL XXDEC2(EPT,EP2,EP3,S23,AM2**2,AM3**2)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXDEC3 - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXDPR(WP0,WM0,ICP,ICT,LQ2)
c Projectile hadron dissociation
c Leading hadronic state hadronization
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP3(4),EY(3)!,EP1(4),EP2(4)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP0
        COMMON /Q_AREA2/  S,Y0,WP00,WM00
        COMMON /Q_AREA8/  WWM,BE(4),DC(5),DETA,ALMPT
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
        COMMON /Q_AREA21/ DMMIN(5)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN


        IF(DEBUG.GE.2)WRITE (MONIOU,201)ICP,ICT,WP0,WM0
201     FORMAT(2X,'XXDPR - LEADING (PROJECTILE) CLUSTER HADRONIZATION:'
     *  /4X,'CLUSTER TYPE ICP=',I2,2X,'TARGET TYPE ',
     *  'ICT=',I2/4X,'AVAILABLE LIGHT CONE MOMENTA: WP0=',E10.3,
     *  ' WM0=',E10.3)
        DO 100 I=1,3
100     EY(I)=1.D0

        SD0=WP0*WM0
        IF(SD0.LT.0.D0)SD0=0.D0
        DDMAX=MIN(5.D0,DSQRT(SD0)-AM(2))
        DDMIN=DMMIN(ICZ)

        IF(DDMAX.LT.DDMIN)THEN
c Registration of too slow "leading" hadron if its energy is insufficient for
c diffractive exhitation
          EP3(3)=0.D0
          EP3(4)=0.D0

          IF(LQ2.NE.0)THEN
            WPI=WP0
            IF(AM(ICZ)**2.GT.WPI*WM0)THEN
              IF(WPI.GT.0.D0.AND.AM(ICZ)**2/WPI.LT..5D0*WM00)THEN
                WMI=AM(ICZ)**2/WPI
                WM0=WMI
              ELSE
                RETURN
              ENDIF
cdh 2 lines added  in accordance with s. ostapchenko 17.9.99
            ELSE
              WMI=AM(ICZ)**2/WPI
cdh
            ENDIF
            WM0=WM0-WMI
            WP0=0.D0
            EP3(1)=.5D0*(WPI+WMI)
            EP3(2)=.5D0*(WPI-WMI)
            CALL XXREG(EP3,ICP)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
            RETURN
          ELSE

            IF(DSQRT(SD0).LT.AM(ICZ)+AM(2))THEN
              IF(WP0.GT.0.D0.AND.(AM(ICZ)+AM(2))**2/WP0.LT..5D0*WM00)
     *        THEN
                SD0=(AM(ICZ)+AM(2))**2
                WM0=SD0/WP0
              ELSE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
                RETURN
              ENDIF
            ENDIF
            XW=XXTWDEC(SD0,AM(ICZ)**2,AM(2)**2)
            WP1=XW*WP0
            WM1=AM(ICZ)**2/WP1
            EP3(1)=.5D0*(WP1+WM1)
            EP3(2)=.5D0*(WP1-WM1)
            CALL XXREG(EP3,ICP)
            WM2=WM0-WM1
            WP2=AM(2)**2/WM2
            EP3(1)=.5D0*(WP2+WM2)
            EP3(2)=.5D0*(WP2-WM2)
            CALL XXREG(EP3,ICT)
            WP0=0.D0
            WM0=0.D0
          ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
          RETURN
        ENDIF

        IF(ICP.NE.0)IS=IABS(ICP)/ICP

        DMASS=DDMIN**2/(1.D0-PSRAN(B10)*(1.D0-(DDMIN/DDMAX)))**2

        IF(LQ2.NE.0)THEN
          WPD=WP0
          WMD=DMASS/WPD
          WM0=WM0-WMD
          WP0=0.D0
        ELSE
        IF(ICZ.EQ.5)THEN
          WPD=WP0*XXTWDEC(SD0,DMASS,AM(2)**2)
          WMD=DMASS/WPD
          WM2=WM0-WMD
          WP2=AM(2)**2/WM2
          EP3(1)=.5D0*(WP2+WM2)
          EP3(2)=.5D0*(WP2-WM2)
          EP3(3)=0.D0
          EP3(4)=0.D0
          CALL XXREG(EP3,ICT)
        ELSE
          PTMAX=QGSPSLAM(SD0,DMASS,AM(2)**2)
          IF(PTMAX.LT.0.)PTMAX=0.
          PTI=-1.D0/RS*DLOG(1.D0-PSRAN(B10)*(1.D0-EXP(-RS*PTMAX)))

          AMT1=DMASS+PTI
          AMT2=AM(2)**2+PTI
          WPD=WP0*XXTWDEC(SD0,AMT1,AMT2)
          WMD=AMT1/WPD
          WM2=WM0-WMD
          WP2=AMT2/WM2
          PT=DSQRT(PTI)
          CALL QGSPSCS(CCOS,SSIN)
          EP3(3)=PT*CCOS
          EP3(4)=PT*SSIN
          EP3(1)=.5D0*(WP2+WM2)
          EP3(2)=.5D0*(WP2-WM2)
          CALL XXREG(EP3,ICT)
          EP3(3)=-EP3(3)
          EP3(4)=-EP3(4)
          EP3(1)=.5D0*(WPD+WMD)
          EP3(2)=.5D0*(WPD-WMD)
          CALL QGSPSDEFTR(DMASS,EP3,EY)
          WPD=DSQRT(DMASS)
          WMD=WPD
        ENDIF
          WP0=0.D0
          WM0=0.D0
        ENDIF

        IF(ICZ.EQ.5)THEN
          ICH1=ICP
          ICH2=0
          AMH1=AM(5)**2
          AMH2=AM(1)**2

          PTMAX=QGSPSLAM(DMASS,AMH1,AMH2)
          IF(PTMAX.LT.0.)PTMAX=0.
          IF(PTMAX.LT.BE(4)**2)THEN
1           PTI=PTMAX*PSRAN(B10)
            IF(PSRAN(B10).GT.EXP(-DSQRT(PTI)/BE(4)))GOTO 1
          ELSE
2           PTI=(BE(4)*DLOG(PSRAN(B10)*PSRAN(B10)))**2
            IF(PTI.GT.PTMAX)GOTO 2
          ENDIF
          AMT1=AMH1+PTI
          AMT2=AMH2+PTI
          Z=XXTWDEC(DMASS,AMT1,AMT2)
          WP1=WPD*Z
          WM1=AMT1/WP1
          EP3(1)=.5D0*(WP1+WM1)
          EP3(2)=.5D0*(WP1-WM1)
          PT=DSQRT(PTI)
          CALL QGSPSCS(C,S)
          EP3(3)=PT*C
          EP3(4)=PT*S
          CALL XXREG(EP3,ICH1)

          WP1=WPD*(1.D0-Z)
          WM1=AMT2/WP1
          EP3(1)=.5D0*(WP1+WM1)
          EP3(2)=.5D0*(WP1-WM1)
          EP3(3)=-PT*C
          EP3(4)=-PT*S
          CALL XXREG(EP3,ICH2)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
          RETURN
        ENDIF

        IF(ICZ.EQ.1)THEN
          IF(ICP.NE.0)THEN
            IC1=ICP*(1-3*INT(.5D0+PSRAN(B10)))
            IC2=-ICP-IC1
          ELSE
            IC1=INT(1.5D0+PSRAN(B10))*(2*INT(.5D0+PSRAN(B10))-1)
            IC2=-IC1
          ENDIF
        ELSEIF(ICZ.EQ.2)THEN
          IF(PSRAN(B10).GT..33333D0)THEN
            IC1=3*IS
            IC2=ICP-IS
          ELSE
            IC1=ICP+4*IS
            IC2=4*IS-ICP
          ENDIF
        ELSEIF(ICZ.EQ.3)THEN
          IC1=-4*IS
          IC2=ICP-3*IS
        ELSEIF(ICZ.EQ.4)THEN
          IC1=5*IS
          IC2=ICP-9*IS
        ENDIF
        CALL XXGENER(WPD,WMD,EY,0.D0,1.D0,0.D0,1.D0,
     *  IC1,IC2)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXDPR - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXDTG(WP0,WM0,ICP,ICT,LQ1)
c Target nucleon dissociation
c Leading hadronic state hadronization
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP3(4),EY(3)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP0
        COMMON /Q_AREA2/  S,Y0,WP00,WM00
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA17/ DEL,RS,RS0,FS,ALFP,RR,SH,DELH
        COMMON /Q_AREA21/ DMMIN(5)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN


        IF(DEBUG.GE.2)WRITE (MONIOU,201)ICT,ICP,WP0,WM0
201     FORMAT(2X,'XXDTG - LEADING (TARGET) CLUSTER HADRONIZATION:'
     *  /4X,'CLUSTER TYPE ICT=',I2,2X,'PROJECTILE TYPE ',
     *  'ICP=',I2/4X,'AVAILABLE LIGHT CONE MOMENTA: WP0=',E10.3,
     *  ' WM0=',E10.3)
        DO 100 I=1,3
100     EY(I)=1.D0

        SD0=WP0*WM0
        IF(SD0.LT.0.D0)SD0=0.D0
        DDMIN=DMMIN(2)
        DDMAX=MIN(5.D0,DSQRT(SD0)-AM(ICZ))

        IF(DDMAX.LT.DDMIN)THEN
c Registration of too slow "leading" hadron if its energy is insufficient for
c diffractive exhitation
          EP3(3)=0.D0
          EP3(4)=0.D0

          IF(LQ1.NE.0)THEN
            WMI=WM0
            IF( WP0.LE.0.D0.OR.AM(2)**2.GT.WMI*WP0)RETURN
            WPI=AM(2)**2/WMI
            WP0=WP0-WPI
            WM0=0.D0
            EP3(1)=.5D0*(WPI+WMI)
            EP3(2)=.5D0*(WPI-WMI)
            CALL XXREG(EP3,ICT)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
            RETURN
          ELSE

            IF(DSQRT(SD0).LT.AM(ICZ)+AM(2))THEN
              IF(WP0.GT.0.D0.AND.(AM(ICZ)+AM(2))**2/WP0.LT..5D0*WM00)
     *        THEN
                SD0=(AM(ICZ)+AM(2))**2
                WM0=SD0/WP0
              ELSE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
                RETURN
              ENDIF
            ENDIF
            XW=XXTWDEC(SD0,AM(ICZ)**2,AM(2)**2)
            WP1=XW*WP0
            WM1=AM(ICZ)**2/WP1
            EP3(1)=.5D0*(WP1+WM1)
            EP3(2)=.5D0*(WP1-WM1)
            CALL XXREG(EP3,ICP)
            WM2=WM0-WM1
            WP2=AM(2)**2/WM2
            EP3(1)=.5D0*(WP2+WM2)
            EP3(2)=.5D0*(WP2-WM2)
            CALL XXREG(EP3,ICT)
            WP0=0.D0
            WM0=0.D0
          ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
          RETURN
        ENDIF

        DMASS=(DDMIN/(1.D0-PSRAN(B10)*(1.D0-DDMIN/DDMAX)))**2
        IF(LQ1.NE.0)THEN
          WMD=WM0
          WPD=DMASS/WMD
          WP0=WP0-WPD
          WM0=0.D0
        ELSE
          PTMAX=QGSPSLAM(SD0,DMASS,AM(ICZ)**2)
          IF(PTMAX.LT.0.)PTMAX=0.
          PTI=-1.D0/RS*DLOG(1.D0-PSRAN(B10)*(1.D0-EXP(-RS*PTMAX)))

          AMT1=DMASS+PTI
          AMT2=AM(ICZ)**2+PTI
          WMD=WM0*XXTWDEC(SD0,AMT1,AMT2)
          WPD=AMT1/WMD
          WP2=WP0-WPD
          WM2=AMT2/WP2
          PT=DSQRT(PTI)
          CALL QGSPSCS(CCOS,SSIN)
          EP3(3)=PT*CCOS
          EP3(4)=PT*SSIN
          EP3(1)=.5D0*(WP2+WM2)
          EP3(2)=.5D0*(WP2-WM2)
          CALL XXREG(EP3,ICP)
          EP3(3)=-EP3(3)
          EP3(4)=-EP3(4)
          EP3(1)=.5D0*(WPD+WMD)
          EP3(2)=.5D0*(WPD-WMD)
          CALL QGSPSDEFTR(DMASS,EP3,EY)
          WPD=DSQRT(DMASS)
          WMD=WPD
          WP0=0.D0
          WM0=0.D0
        ENDIF

        IS=IABS(ICT)/ICT
        IF(PSRAN(B10).GT..33333D0)THEN
          IC1=3*IS
          IC2=ICT-IS
        ELSE
          IC1=ICT+4*IS
          IC2=4*IS-ICT
        ENDIF
        CALL XXGENER(WPD,WMD,EY,
     *  0.D0,1.D0,0.D0,1.D0,IC2,IC1)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXDTG - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXFAU(B,GZ)
c Integrands for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION GZ(3),GZ0(2)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP
        COMMON /Q_AREA16/ CC(5)
        COMMON /Q_AR1/    ANORM
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'XXFAU - INTEGRANDS FOR HADRON-HADRON AND '
     *    ,'HADRON-NUCLEUS CROSS-SECTIONS CALCULATION')

        CALL XXFZ(B,GZ0)
        DO 1 L=1,2
1       GZ0(L)=GZ0(L)*CC(2)*ANORM*.5D0

        AB=FLOAT(IA(2))

        GZ1=(1.D0-GZ0(1))**AB
        GZ2=(1.D0-GZ0(2))**AB
        GZ3=(1.D0-CC(2)*GZ0(2)-2.D0*(1.D0-CC(2))*GZ0(1))**AB


        GZ(1)=CC(ICZ)**2*(GZ2-GZ3)
        GZ(2)=CC(ICZ)*(1.D0-CC(ICZ))*(1.D0+GZ2-2.D0*GZ1)
        GZ(3)=CC(ICZ)*(1.D0-GZ2)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXFAU - END')
        RETURN
        END
C=======================================================================

         SUBROUTINE XXFRAG(SA,NA,RC)
c Connected nucleon clasters extraction - used for the nuclear spectator part
c multifragmentation:
c-----------------------------------------------------------------------
         IMPLICIT DOUBLE PRECISION (A-H,O-Z)
         INTEGER DEBUG
cdh      DIMENSION SA(56,3)
         DIMENSION SA(210,3)
         COMMON /Q_AREA13/ NSF,IAF(56)
         COMMON /Q_AREA43/ MONIOU
         COMMON /Q_DEBUG/  DEBUG
         SAVE

         IF(DEBUG.GE.2)WRITE (MONIOU,201)NA
201      FORMAT(2X,'XXFRAG-MULTIFRAGMENTATION: NUCLEUS MASS NUMBER: NA='
     *   ,I2)
         IF(DEBUG.GE.3)THEN
           WRITE (MONIOU,203)
203        FORMAT(2X,'NUCLEONS COORDINATES:')
204        FORMAT(2X,3E10.3)
           DO 205 I=1,NA
205        WRITE (MONIOU,204)(SA(I,L),L=1,3)
         ENDIF

         NI=1
         NG=1
         J=0
1        J=J+1
         J1=NI+1
         DO 4 I=J1,NA
         RI=0.D0
         DO 2 M=1,3
2        RI=RI+(SA(J,M)-SA(I,M))**2
         IF(RI.GT.RC)GOTO 4
         NI=NI+1
         NG=NG+1
         IF(I.EQ.NI)GOTO 4
         DO 3 M=1,3
         S0=SA(NI,M)
         SA(NI,M)=SA(I,M)
3        SA(I,M)=S0
4        CONTINUE
         IF(J.LT.NI.AND.NA-NI.GT.0)GOTO 1
         NSF=NSF+1
         IAF(NSF)=NG
         IF(DEBUG.GE.3)WRITE (MONIOU,206)NSF,IAF(NSF)
206      FORMAT(2X,'XXFRAG: FRAGMENT N',I2,2X,'FRAGMENT MASS - ',I2)
         NG=1
         J=NI
         NI=NI+1
         IF(NA-NI)6,5,1
5        NSF=NSF+1
         IAF(NSF)=1
         IF(DEBUG.GE.3)WRITE (MONIOU,206)NSF,IAF(NSF)
6        CONTINUE
         IF(DEBUG.GE.3)WRITE (MONIOU,202)
202      FORMAT(2X,'XXFRAG - END')
         RETURN
         END
C=======================================================================

      SUBROUTINE XXFRAGM(NS,XA)
c Fragmentation of the spectator part of the nucleus
c XA(56,3) - arrays for spectator nucleons positions
c NS - total number of spectators
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
cdh   DIMENSION XA(56,3)
      DIMENSION XA(210,3)
      INTEGER DEBUG
      COMMON /Q_AREA1/  IA(2),ICZ,ICP
      COMMON /Q_AREA3/  RMIN,EMAX,EEV
      COMMON /Q_AREA11/ B10
c NSF - number of secondary fragments;
c IAF(i) - mass of the i-th fragment
      COMMON /Q_AREA13/ NSF,IAF(56)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE
      EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)NS
201     FORMAT(2X,'XXFRAGM: NUMBER OF SPECTATORS: NS=',I2)

        NSF=0

        IF(NS-1)6,1,2
c Single spectator nucleon is recorded
1     NSF=NSF+1
      IAF(NSF)=1
        IF(DEBUG.GE.3)WRITE (MONIOU,205)
205     FORMAT(2X,'XXFRAGM - SINGLE SPECTATOR')
        GOTO 6
2       EEX=0.D0
c EEX - spectator part excitation energy; calculated as the sum of excitations
c from all wounded nucleons ( including diffractively excited )
        DO 3 I=1,IA(1)-NS
c Partial excitation is simulated according to distribution f(E) ~ 1/sqrt(E)
c * exp(-E/(2*<E>)), for sqrt(E) we have then normal distribution
3     EEX=EEX+(PSRAN(B10)+PSRAN(B10)+PSRAN(B10)+
     *      PSRAN(B10)+PSRAN(B10)-2.5D0)**2*2.4D0
          IF(DEBUG.GE.3)WRITE (MONIOU,203)EEX
203     FORMAT(2X,'XXFRAGM: EXCITATION ENERGY: EEX=',E10.3)

c If the excitation energy per spectator is larger than EMAX
c multifragmentation takes place ( percolation algorithm is used for it )
        IF(EEX/NS.GT.EMAX)THEN
c Multifragmentation
          CALL XXFRAG(XA,NS,RMIN)
        ELSE

c Otherwise average number of eveporated nucleons equals EEX/EEV, where
c EEV - mean excitation energy carried out by one nucleon
          NF=IXXSON(NS,EEX/EEV,PSRAN(B10))
          NSF=NSF+1
c Recording of the fragment produced
          IAF(NSF)=NS-NF
          IF(DEBUG.GE.3)WRITE (MONIOU,206)IAF(NSF)
206     FORMAT(2X,'XXFRAGM - EVAPORATION: MASS NUMBER OF THE FRAGMENT:'
     *  ,I2)

c Some part of excitation energy is carried out by alphas; we determine the
c number of alphas simply as NF/4
          NAL=NF/4
          IF(NAL.NE.0)THEN
c Recording of the evaporated alphas
            DO 4 I=1,NAL
            NSF=NSF+1
4           IAF(NSF)=4
          ENDIF

          NF=NF-4*NAL
          IF(NF.NE.0)THEN
c Recording of the evaporated nucleons
            DO 5 I=1,NF
            NSF=NSF+1
5           IAF(NSF)=1
          ENDIF
          IF(DEBUG.GE.3)WRITE (MONIOU,204)NF,NAL
204     FORMAT(2X,'XXFRAGM - EVAPORATION: NUMBER OF NUCLEONS NF=',I2,
     *  'NUMBER OF ALPHAS NAL=',I2)
        ENDIF
6       CONTINUE
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXFRAGM - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXFZ(B,GZ)
c Hadron-hadron and hadron-nucleus cross sections calculation
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION GZ(2),FHARD(3)
        COMMON /Q_AREA1/  IA(2),ICZ,ICP
        COMMON /Q_AREA2/  S,Y0,WP0,WM0
        COMMON /Q_AREA7/  RP1
        COMMON /Q_AR13/    X1(7),A1(7)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'XXFZ - HADRONIC CROSS-SECTIONS CALCULATION')

        DO 1 L=1,2
1       GZ(L)=0.D0
        E1=EXP(-1.D0)

        DO 2 I1=1,7
        DO 2 M=1,2
        Z=.5D0+X1(I1)*(M-1.5D0)
        S1=DSQRT(RP1*Z)
        ZV1=EXP(-Z)
        S2=DSQRT(RP1*(1.D0-DLOG(Z)))
        ZV2=E1*Z
C??????????
C       VV1=EXP(-PSFAZ(ZV1,FSOFT,FHARD,FSHARD))*(1.D0-FHARD(1)
C    *  -FHARD(2)-FHARD(3))
C       VV2=EXP(-PSFAZ(ZV2,FSOFT,FHARD,FSHARD))*(1.D0-FHARD(1)
C    *  -FHARD(2)-FHARD(3))

        VV1=EXP(-PSFAZ(ZV1,FSOFT,FHARD,FSHARD)-FHARD(1)
     *  -FHARD(2)-FHARD(3))
        VV2=EXP(-PSFAZ(ZV2,FSOFT,FHARD,FSHARD)-FHARD(1)
     *  -FHARD(2)-FHARD(3))
c???????????

        IF(IA(2).EQ.1)THEN
          CG1=1.D0
          CG2=1.D0
        ELSE
          CG1=XXROT(B,S1)
          CG2=XXROT(B,S2)
        ENDIF

        DO 2 L=1,2
2       GZ(L)=GZ(L)+ A1(I1)*(CG1*(1.D0-VV1**L)+CG2*(1.D0-VV2**L)/Z)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXFZ - END')
        RETURN
        END
C=======================================================================

      SUBROUTINE XXGAU(GZ)
c Impact parameter integration for impact parameters <BM -
c for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      DIMENSION GZ(3),GZ0(3)
      COMMON /Q_AREA6/ PI,BM,AM
      COMMON /Q_AR13/   X1(7),A1(7)
      COMMON /Q_AR2/   R,RM
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'XXGAU - NUCLEAR CROSS-SECTIONS CALCULATION')

      DO 1 I=1,3
1     GZ(I)=0.D0

      DO 2 I=1,7
      DO 2 M=1,2
      B=BM*DSQRT(.5D0+X1(I)*(M-1.5D0))
      CALL XXFAU(B,GZ0)
      DO 2 L=1,3
2     GZ(L)=GZ(L)+GZ0(L)*A1(I)
      DO 3 L=1,3
3     GZ(L)=GZ(L)*(BM*AM)**2*PI*.5D0
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXGAU - END')
      RETURN
      END
C=======================================================================

      SUBROUTINE XXGAU1(GZ)
c Impact parameter integration for impact parameters >BM -
c for hadron-hadron and hadron-nucleus cross-sections calculation
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      DIMENSION GZ(3),GZ0(3)
      COMMON /Q_AREA6/ PI,BM,AM
      COMMON /Q_AR15/   X5(2),A5(2)
      COMMON /Q_AR2/   R,RM
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)
201     FORMAT(2X,'XXGAU1 - NUCLEAR CROSS-SECTIONS CALCULATION')

      DO 1 I=1,2
      B=BM+X5(I)
      CALL XXFAU(B,GZ0)
      DO 1 L=1,3
1     GZ(L)=GZ(L)+GZ0(L)*A5(I)*EXP(X5(I))*B*2.D0*PI*AM*AM
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXGAU1 - END')
      RETURN
      END
C=======================================================================

        SUBROUTINE XXGENER(WP0,WM0,EY0,S0X,C0X,S0,C0,IC1,IC2)
c To simulate the fragmentation of the string into secondary hadrons
c The algorithm conserves energy-momentum;
c WP0, WM0 are initial longitudinal momenta ( E+p, E-p ) of the quarks
c at the ends of the string; IC1, IC2 - their types
c The following partons types are used: 1 - u, -1 - U, 2 - d, -2 - D,
c 3 - ud, -3 - UD, 4 - s, -4 - S, 5 - c, -5 - C,
c  6 - uu, 7 - dd, -6 - UU, -7 - DD
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        CHARACTER *2 TYQ
        DIMENSION WP(2),IC(2),EPT(4),EP(4),EY(3),EY0(3)
c WP(1), WP(2) - current longitudinal momenta of the partons at the string
c ends, IC(1), IC(2) - their types
        COMMON /Q_AREA8/  WWM,BEP,BEN,BEK,BEC,DC(5),DETA,ALMPT
        COMMON /Q_AREA10/ STMASS,AM0,AMN,AMK,AMC,AMLAMC,AMLAM,AMETA
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA19/ AHL(5)
********************************************************
        COMMON /Q_AREA21/ DMMIN(5)
********************************************************
        COMMON /Q_AREA28/ ARR(4)
        COMMON /Q_AREA42/ TYQ(15)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)TYQ(8+IC1),TYQ(8+IC2),
     *  WP0,WM0,EY0,S0X,C0X,S0,C0
201     FORMAT(2X,'XXGENER: PARTON FLAVORS AT THE ENDS OF THE STRING:',
     *  2X,A2,2X,A2/4X,'LIGHT CONE MOMENTA OF THE STRING: ',E10.3,
     *  2X,E10.3/4X,'EY0=',3E10.3/4X,
     *  'S0X=',E10.3,2X,'C0X=',E10.3,2X,'S0=',E10.3,2X,'C0=',E10.3)

        WW=WP0*WM0
        EPT(1)=.5D0*(WP0+WM0)
        EPT(2)=.5D0*(WP0-WM0)
        EPT(3)=0.D0
        EPT(4)=0.D0
        IC(1)=IC1
        IC(2)=IC2

1     SWW=DSQRT(WW)
      CALL QGSPSDEFTR(WW,EPT,EY)
      J=INT(2.D0*PSRAN(B10))+1
      IF(DEBUG.GE.3)THEN
        IQT=8+IC(J)
        WRITE (MONIOU,203)J,TYQ(IQT),WW
203     FORMAT(2X,'XXGENER: CURRENT PARTON FLAVOR AT THE END ',I1,
     *  ' OF THE STRING: ',A2/4X,' STRING MASS: ',E10.3)
      ENDIF

      IAB=IABS(IC(J))
      IS=IC(J)/IAB
      IF(IAB.GT.5)IAB=3
      IAJ=IABS(IC(3-J))
      IF(IAJ.GT.5)IAJ=3
      IF(IAJ.EQ.3)THEN
        RESTM=AMN
      ELSEIF(IAJ.EQ.4)THEN
          RESTM=AMK
      ELSEIF(IAJ.EQ.5)THEN
        RESTM=AMC
      ELSE
        RESTM=AM0
      ENDIF

      IF(IAB.LE.2.AND.SWW.GT.RESTM+2.D0*AM0+WWM.OR.
     *IAB.EQ.3.AND.SWW.GT.RESTM+AM0+AMN+WWM.OR.
     *IAB.EQ.4.AND.SWW.GT.RESTM+AM0+AMK+WWM.OR.
     *IAB.EQ.5.AND.SWW.GT.RESTM+AM0+AMC+WWM)THEN

        IF(IAB.LE.2)THEN
          IF(SWW.GT.RESTM+2.D0*AMC.AND.PSRAN(B10).LT.DC(3))THEN
c D-meson generation
            RESTM=(RESTM+AMC)**2
            BET=BEC
            AMI=AMC**2
            ALF=ALMPT-ARR(4)
            BLF=AHL(4)
            IC0=IC(J)-9*IS
            IC(J)=5*IS
          ELSEIF(SWW.GT.RESTM+2.D0*AMN.AND.PSRAN(B10).LT.DC(1))THEN
c Nucleon generation
            RESTM=(RESTM+AMN)**2
            BET=BEN
            AMI=AMN**2
            ALF=ALMPT-ARR(2)
            BLF=AHL(2)
            IC0=IC(J)+IS
            IC(J)=-3*IS
          ELSEIF(SWW.GT.RESTM+2.D0*AMK.AND.PSRAN(B10).LT.DC(2))THEN
c Kaon generation
            RESTM=(RESTM+AMK)**2
            BET=BEK
            AMI=AMK**2
            ALF=ALMPT-ARR(3)
            BLF=AHL(3)
            IC0=IC(J)+3*IS
            IC(J)=4*IS
          ELSEIF(SWW.GT.RESTM+AMETA+AM0.AND.PSRAN(B10).LT.DETA)THEN
c Eta generation
            RESTM=(RESTM+AM0)**2
            BET=BEK
            AMI=AMETA**2
            ALF=ALMPT-ARR(1)
            BLF=AHL(1)
            IC0=10
          ELSE
c Pion generation
            RESTM=(RESTM+AM0)**2
            BET=BEP
            AMI=AM0**2
            ALF=ALMPT-ARR(1)
            BLF=AHL(1)

            IF(PSRAN(B10).LT..3333D0)THEN
              IC0=0
            ELSE
              IC0=3*IS-2*IC(J)
              IC(J)=3*IS-IC(J)
            ENDIF
          ENDIF

        ELSEIF(IAB.EQ.3)THEN
          IF(SWW.GT.RESTM+AMC+AMLAMC.AND.PSRAN(B10).LT.DC(5).AND.
     *    IABS(IC(J)).EQ.3)THEN
c Lambda_C generation
            RESTM=(RESTM+AMC)**2
            BET=BEC
            AMI=AMLAMC**2
            ALF=ALMPT-ARR(4)
            BLF=AHL(5)
            IC0=9*IS
            IC(J)=-5*IS
          ELSEIF(SWW.GT.RESTM+AMK+AMLAM.AND.PSRAN(B10).LT.DC(4).AND.
     *    IABS(IC(J)).EQ.3)THEN
c Lambda generation
            RESTM=(RESTM+AMK)**2
            BET=BEK
            AMI=AMLAM**2
            ALF=ALMPT-ARR(3)
            BLF=AHL(2)+ARR(1)-ARR(3)
            IC0=6*IS
            IC(J)=-4*IS
          ELSE
c Nucleon generation
            RESTM=(RESTM+AM0)**2
            BET=BEN
            AMI=AMN**2
            ALF=ALMPT-ARR(1)
            BLF=AHL(2)
            IF(IABS(IC(J)).EQ.3)THEN
              IC0=IS*INT(2.5D0+PSRAN(B10))
              IC(J)=IS-IC0
            ELSE
              IC0=IC(J)-4*IS
              IC(J)=IC0-4*IS
            ENDIF
          ENDIF

        ELSEIF(IAB.EQ.4)THEN
          IF(SWW.GT.RESTM+AMN+AMLAM.AND.PSRAN(B10).LT.DC(1))THEN
c Lambda generation
            RESTM=(RESTM+AMN)**2
            BET=BEN
            AMI=AMLAM**2
            ALF=ALMPT-ARR(2)
            BLF=AHL(2)+ARR(1)-ARR(3)
            IC0=6*IS
            IC(J)=-3*IS
          ELSE
c Kaon generation
            RESTM=(RESTM+AM0)**2
            BET=BEP
            AMI=AMK**2
            ALF=ALMPT-ARR(1)
            BLF=AHL(3)
            IC(J)=IS*INT(1.5D0+PSRAN(B10))
            IC0=-3*IS-IC(J)
          ENDIF

        ELSEIF(IAB.EQ.5)THEN
          IF(SWW.GT.RESTM+AMN+AMLAMC.AND.PSRAN(B10).LT.DC(1))THEN
c Lambda_C generation
            RESTM=(RESTM+AMN)**2
            BET=BEN
            AMI=AMLAMC**2
            ALF=ALMPT-ARR(2)
            BLF=AHL(5)
            IC0=9*IS
            IC(J)=-3*IS
          ELSE
c D-meson generation
            RESTM=(RESTM+AM0)**2
            BET=BEP
            AMI=AMC**2
            ALF=ALMPT-ARR(1)
            BLF=AHL(4)
            IC(J)=IS*INT(1.5D0+PSRAN(B10))
            IC0=9*IS-IC(J)
          ENDIF
        ENDIF

********************************************************
        PTMAX=QGSPSLAM(WW,RESTM,AMI)
        IF(PTMAX.LT.0.)PTMAX=0.

        IF(PTMAX.LT.BET**2)THEN
2         PTI=PTMAX*PSRAN(B10)
          IF(PSRAN(B10).GT.EXP(-DSQRT(PTI)/BET))GOTO 2
        ELSE
3         PTI=(BET*DLOG(PSRAN(B10)*PSRAN(B10)))**2
          IF(PTI.GT.PTMAX)GOTO 3
        ENDIF

        AMT=AMI+PTI
        RESTM1=RESTM+PTI
********************************************************
c        ALF=ALF+2.*PTI

        ZMIN=DSQRT(AMT/WW)
        ZMAX=XXTWDEC(WW,AMT,RESTM1)
        Z1=(1.-ZMAX)**ALF
        Z2=(1.-ZMIN)**ALF
4       Z=1.-(Z1+(Z2-Z1)*PSRAN(B10))**(1./ALF)
        IF(PSRAN(B10).GT.(Z/ZMAX)**BLF)GOTO 4
        WP(J)=Z*SWW
        WP(3-J)=AMT/WP(J)
        EP(1)=.5D0*(WP(1)+WP(2))
        EP(2)=.5D0*(WP(1)-WP(2))
        PTI=DSQRT(PTI)
        CALL QGSPSCS(C,S)
        EP(3)=PTI*C
        EP(4)=PTI*S

        EPT(1)=SWW-EP(1)
        DO 5 I=2,4
5       EPT(I)=-EP(I)
        WW=QGSPSNORM(EPT)
        IF(WW.LT.RESTM)GOTO 4

        CALL QGSPSTRANS(EP,EY)
        CALL QGSPSTRANS(EPT,EY)

        IF(S0X.NE.0.D0.OR.S0.NE.0.D0)THEN
          CALL QGSPSROTAT(EP,S0X,C0X,S0,C0)
        ENDIF

        IF(EY0(1)*EY0(2)*EY0(3).NE.1.D0)THEN
          CALL QGSPSTRANS(EP,EY0)
        ENDIF
        CALL XXREG(EP,IC0)
      ELSE


        AMI2=RESTM**2
        BET=BEP
        IF(IAB.LE.2.AND.IAJ.LE.2)THEN
          AMI=AM0**2
          IC0=-IC(1)-IC(2)
          IF(IC0.NE.0)THEN
            IC(J)=IC0*INT(.5D0+PSRAN(B10))
            IC(3-J)=IC0-IC(J)
          ELSE
            IF(PSRAN(B10).LT..2D0)THEN
              IC(J)=0
              IC(3-J)=0
            ELSE
              IC(J)=3*IS-2*IC(J)
              IC(3-J)=-IC(J)
            ENDIF
          ENDIF

        ELSEIF(IAB.EQ.3.OR.IAJ.EQ.3)THEN
          IF(IAB.EQ.3)THEN
            AMI=AMN**2
            IF(IABS(IC(J)).EQ.3)THEN
              IF(IAJ.EQ.3)THEN
                IF(IABS(IC(3-J)).EQ.3)THEN
                  IC(J)=IS*INT(2.5D0+PSRAN(B10))
                  IC(3-J)=-IC(J)
                ELSE
                  IC(3-J)=IC(3-J)+4*IS
                  IC(J)=5*IS+IC(3-J)
                ENDIF
              ELSEIF(IAJ.LT.3)THEN
                IF(PSRAN(B10).LT..3333D0)THEN
                  IC(J)=IC(3-J)+IS
                  IC(3-J)=0
                ELSE
                  IC(J)=IS*(4-IAJ)
                  IC(3-J)=IS*(3-2*IAJ)
                ENDIF
              ELSEIF(IAJ.EQ.4)THEN
                IC(J)=IS*INT(2.5D0+PSRAN(B10))
                IC(3-J)=-IC(J)-2*IS
              ELSEIF(IAJ.EQ.5)THEN
                IC(J)=IS*INT(2.5D0+PSRAN(B10))
                IC(3-J)=-IC(J)+10*IS
              ENDIF
            ELSE
              IC(J)=IC(J)-4*IS
              IC0=IC(J)-4*IS
              IF(IAJ.EQ.3)THEN
                IC(3-J)=IC0-IS
              ELSEIF(IAJ.LT.3)THEN
                IC(3-J)=-IC(3-J)-IC0
              ELSEIF(IAJ.EQ.4)THEN
                IC(3-J)=IC0-3*IS
              ELSEIF(IAJ.EQ.5)THEN
                IC(3-J)=IC0+9*IS
              ENDIF
            ENDIF
          ELSE
            IF(IABS(IC(3-J)).EQ.3)THEN
              IF(IAB.LT.3)THEN
                AMI=AM0**2
                IF(PSRAN(B10).LT..3333D0)THEN
                  IC(3-J)=IC(J)+IS
                  IC(J)=0
                ELSE
                  IC(3-J)=IS*(4-IAB)
                  IC(J)=IS*(3-2*IAB)
                ENDIF
              ELSEIF(IAB.EQ.4)THEN
                AMI=AMK**2
                IC(3-J)=IS*INT(2.5D0+PSRAN(B10))
                IC(J)=-IC(3-J)-2*IS
              ELSEIF(IAB.EQ.5)THEN
                AMI=AMC**2
                IC(3-J)=IS*INT(2.5D0+PSRAN(B10))
                IC(J)=-IC(3-J)+10*IS
              ENDIF
            ELSE
              IC(3-J)=IC(3-J)-4*IS
              IC0=IC(3-J)-4*IS
              IF(IAB.LT.3)THEN
                AMI=AM0**2
                IC(J)=-IC0-IC(J)
              ELSEIF(IAB.EQ.4)THEN
                AMI=AMK**2
                IC(J)=IC0-3*IS
              ELSEIF(IAB.EQ.5)THEN
                AMI=AMC**2
                IC(J)=IC0+9*IS
              ENDIF
            ENDIF
          ENDIF

        ELSEIF(IAB.EQ.4.OR.IAJ.EQ.4)THEN

          IF(IAB.EQ.4)THEN
            AMI=AMK**2

            IF(IAJ.EQ.4)THEN
              IC(J)=-IS*INT(4.5D0+PSRAN(B10))
              IC(3-J)=-IC(J)
            ELSEIF(IAJ.EQ.5)THEN
              IC(J)=-IS*INT(4.5D0+PSRAN(B10))
              IC(3-J)=-IC(J)-12*IS
            ELSE
              IC0=IC(3-J)+INT(.6667D0+PSRAN(B10))*(-3*IS-2*IC(3-J))
              IC(J)=IC0-3*IS
              IC(3-J)=IC0-IC(3-J)
            ENDIF
          ELSE
            IF(IAB.LE.2)THEN
              AMI=AM0**2
              IC0=IC(J)+INT(.6667D0+PSRAN(B10))*(3*IS-2*IC(J))
              IC(J)=IC0-IC(J)
              IC(3-J)=IC0+3*IS
            ELSEIF(IAB.EQ.5)THEN
              AMI=AMC**2
              IC(3-J)=IS*INT(4.5D0+PSRAN(B10))
              IC(J)=-IC(3-J)+12*IS
            ENDIF
          ENDIF

        ELSEIF(IAB.EQ.5.OR.IAJ.EQ.5)THEN

          IF(IAB.EQ.5)THEN
            AMI=AMC**2

            IF(IAJ.EQ.5)THEN
              IC(J)=IS*INT(7.5D0+PSRAN(B10))
              IC(3-J)=-IC(J)
            ELSE
              IC0=IC(3-J)+INT(.6667D0+PSRAN(B10))*(-3*IS-2*IC(3-J))
              IC(J)=IC0+9*IS
              IC(3-J)=IC0-IC(3-J)
            ENDIF
          ELSE
            AMI=AM0**2
            IC0=IC(J)+INT(.6667D0+PSRAN(B10))*(3*IS-2*IC(J))
            IC(J)=IC0-IC(J)
            IC(3-J)=IC0-9*IS
          ENDIF
        ENDIF

        PTMAX=QGSPSLAM(WW,AMI2,AMI)
        IF(PTMAX.LT.0.)PTMAX=0.
        IF(PTMAX.LT.BET**2)THEN
6         PTI=PTMAX*PSRAN(B10)
          IF(PSRAN(B10).GT.EXP(-DSQRT(PTI)/BET))GOTO 6
        ELSE
7         PTI=(BET*DLOG(PSRAN(B10)*PSRAN(B10)))**2
          IF(PTI.GT.PTMAX)GOTO 7
        ENDIF

        AMT1=AMI+PTI
        AMT2=AMI2+PTI

        Z=XXTWDEC(WW,AMT1,AMT2)
        WP(J)=Z*SWW
        WP(3-J)=AMT1/WP(J)
        EP(1)=.5D0*(WP(1)+WP(2))
        EP(2)=.5D0*(WP(1)-WP(2))
        PTI=DSQRT(PTI)
        CALL QGSPSCS(C,S)
        EP(3)=PTI*C
        EP(4)=PTI*S

        EPT(1)=SWW-EP(1)
        DO 8 I=2,4
8       EPT(I)=-EP(I)

        CALL QGSPSTRANS(EP,EY)
        CALL QGSPSTRANS(EPT,EY)

        IF(S0X.NE.0.D0.OR.S0.NE.0.D0)THEN
          CALL QGSPSROTAT(EP,S0X,C0X,S0,C0)
          CALL QGSPSROTAT(EPT,S0X,C0X,S0,C0)
        ENDIF
        IF(EY0(1)*EY0(2)*EY0(3).NE.1.D0)THEN
          CALL QGSPSTRANS(EP,EY0)
          CALL QGSPSTRANS(EPT,EY0)
        ENDIF

        CALL XXREG(EP,IC(J))
        CALL XXREG(EPT,IC(3-J))
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXGENER - END')
        RETURN
      ENDIF
      GOTO 1
      END
C=======================================================================

        SUBROUTINE XXJETSIM
c Procedure for jet hadronization - each gluon is
c considered to be splitted into quark-antiquark pair and usual soft
c strings are assumed to be formed between quark and antiquark
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4),EP1(4),ey(3)
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        COMMON /Q_AREA46/ EPJET(4,2,15000),IPJET(2,15000)
        COMMON /Q_AREA47/ NJTOT
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)NJTOT
201     FORMAT(2X,'XXJETSIM: TOTAL NUMBER OF JETS NJTOT=',I4)
        IF(NJTOT.EQ.0)RETURN
        DO 2 NJ=1,NJTOT
        DO 1 I=1,4
        EP1(I)=EPJET(I,1,NJ)
1       EP(I)=EP1(I)+EPJET(I,2,NJ)
        PT3=DSQRT(EP1(3)**2+EP1(4)**2)
        PT4=DSQRT(EPJET(3,2,NJ)**2+EPJET(4,2,NJ)**2)

c Invariant mass square for the jet
        WW=QGSPSNORM(EP)
        SWW=DSQRT(WW)

        CALL QGSPSDEFTR(WW,EP,EY)
        CALL QGSPSTRANS1(EP1,EY)
        CALL QGSPSDEFROT(EP1,S0X,C0X,S0,C0)

2       CALL XXGENER(SWW,SWW,EY,S0X,C0X,S0,C0,IPJET(1,NJ),IPJET(2,NJ))
        IF(DEBUG.GE.3)WRITE (MONIOU,202)
202     FORMAT(2X,'XXJETSIM - END')
        RETURN
        END
C=======================================================================

        SUBROUTINE XXREG(EP0,IC)
c Registration of the produced hadron;
c EP - 4-momentum,
c IC - hadron type
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EP(4),EP0(4)
        COMMON /Q_AREA4/  EY0(3)
        COMMON /Q_AREA10/ STMASS,AM0,AMN,AMK,AMC,AMLAMC,AMLAM,AMETA
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA12/ NSH
        COMMON /Q_AREA14/ ESP(4,95000),ICH(95000)
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)IC,EP0
201     FORMAT(2X,'XXREG: IC=',I2,2X,'C.M. 4-MOMENTUM:',2X,4(E10.3,1X))
         pt=dsqrt(ep0(3)**2+ep0(4)**2)
c         if(pt.gt.11.d0)write (MONIOU,*)'pt,ic,ep',pt,ic,ep0
c         if(pt.gt.11.d0)write (*,*)'pt,ic,ep',pt,ic,ep0

        NSH=NSH+1
        IF (NSH .GT. 95000) THEN
          WRITE(MONIOU,*)'XXREG: TOO MANY SECONDARY PARTICLES'
          WRITE(MONIOU,*)'XXREG: NSH = ',NSH
          STOP
        ENDIF
        DO 4 I=1,4
4       EP(I)=EP0(I)
ctp        CALL QGSPSTRANS(EP,EY0)
        IF(DEBUG.GE.3)WRITE (MONIOU,202)EP
202     FORMAT(2X,'XXREG: LAB. 4-MOMENTUM:',2X,4(E10.3,1X))

        ICH(NSH)=IC
        DO 3 I=1,4
3       ESP(I,NSH)=EP(I)

        IF(DEBUG.GE.3)WRITE (MONIOU,203)
203     FORMAT(2X,'XXREG - END')
        RETURN
        END
C=======================================================================

        FUNCTION XXROT(S,B)
c Convolution of nuclear profile functions (axial angle integration)
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AR18/  X2(4),A2
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)B
201     FORMAT(2X,'XXROT - AXIAL ANGLE INTEGRATION OF THE ',
     *  'NUCLEAR PROFILE FUNCTION'/4X,
     *  'IMPACT PARAMETER B=',E10.3,2X,'NUCLEON COORDINATE S=',E10.3)

        XXROT=0.
        DO 1 I=1,4
        SB1=B**2+S**2-2.*B*S*(2.*X2(I)-1.)
        SB2=B**2+S**2-2.*B*S*(1.-2.*X2(I))
1       XXROT=XXROT+(XXT(SB1)+XXT(SB2))
        XXROT=XXROT*A2
        IF(DEBUG.GE.3)WRITE (MONIOU,202)XXROT
202     FORMAT(2X,'XXROT=',E10.3)
        RETURN
        END
C=======================================================================

        SUBROUTINE XXSTR(WPI0,WMI0,WP0,WM0,IC10,IC120,IC210,IC20)
**************************************************
c Fragmentation process for the pomeron ( quarks and antiquarks types at the
c ends of the two strings are determined, energy-momentum is shared
c between them and strings fragmentation is simulated )
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        DIMENSION EY(3)
        COMMON /Q_AREA6/  PI,BM,AMMM
        COMMON /Q_AREA10/ STMASS,AM(7)
        COMMON /Q_AREA11/ B10
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE
        EXTERNAL PSRAN

        IF(DEBUG.GE.2)WRITE (MONIOU,201)WPI0,WMI0,WP0,WM0
201     FORMAT(2X,'XXSTR: WPI0=',E10.3,2X,'WMI0=',E10.3,2X,
     *  'WP0=',E10.3,2X,'WM0=',E10.3)
         DO 1 I=1,3
1        EY(I)=1.D0

         WPI=WPI0
         WMI=WMI0
c Quark-antiquark types (1 - u, 2 - d, -1 - u~, -2 - d~); s- and d- quarks are
c taken into consideration at the fragmentation step
**************************************************
        IF(IC10.EQ.0)THEN
          IC1=INT(1.5+PSRAN(B10))
          IC12=-IC1
        ELSEIF(IC10.GT.0)THEN
          IC1=IC10
          IC12=IC120
        ELSE
          IC1=IC120
          IC12=IC10
        ENDIF
        IF(IC20.EQ.0)THEN
          IC2=INT(1.5+PSRAN(B10))
          IC21=-IC2
        ELSEIF(IC20.gt.0)THEN
          IC2=IC20
          IC21=IC210
        ELSE
          IC2=IC210
          IC21=IC20
        ENDIF
**************************************************

c Longitudinal momenta for the strings
        WP1=WPI*COS(PI*PSRAN(B10))**2
        WM1=WMI*COS(PI*PSRAN(B10))**2
        WPI=WPI-WP1
        WMI=WMI-WM1
c String masses
        SM1=WP1*WM1
        SM2=WPI*WMI
c Too short strings are neglected (energy is given to partner string or to the hadron
c (nucleon) to which the pomeron is connected)
        IF(SM1.GT.STMASS.AND.SM2.GT.STMASS)THEN
c Strings fragmentation is simulated - GENER
          CALL XXGENER(WP1,WM1,EY,0.D0,1.D0,0.D0,1.D0,IC1,IC21)
          CALL XXGENER(WPI,WMI,EY,0.D0,1.D0,0.D0,1.D0,IC12,IC2)
        ELSEIF(SM1.GT.STMASS)THEN
          CALL XXGENER(WP1+WPI,WM1+WMI,EY,0.D0,1.D0,0.D0,1.D0,IC1,IC21)
        ELSEIF(SM2.GT.STMASS)THEN
          CALL XXGENER(WPI+WP1,WMI+WM1,EY,0.D0,1.D0,0.D0,1.D0,IC12,IC2)
        ELSE
          WP0=WP0+WP1+WPI
          WM0=WM0+WM1+WMI
        ENDIF
        IF(DEBUG.GE.3)WRITE (MONIOU,202)WP0,WM0
202     FORMAT(2X,'XXSTR - RETURNED LIGHT CONE MOMENTA:',
     *  2X,'WP0=',E10.3,2X,'WM0=',E10.3)
        RETURN
        END
C=======================================================================

      FUNCTION XXT(B)
c Nuclear profile function value at impact parameter squared B
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER DEBUG
      COMMON /Q_AREA6/ PI,BM,AM
      COMMON /Q_AR2/   R,RM
      COMMON /Q_AR15/   X5(2),A5(2)
      COMMON /Q_AR19/   X9(3),A9(3)
      COMMON /Q_AREA43/ MONIOU
      COMMON /Q_DEBUG/  DEBUG
      SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)B
201     FORMAT(2X,'XXT - NUCLEAR PROFILE FUNCTION VALUE AT IMPACT',
     *  ' PARAMETER SQUARED B=',E10.3)
      XXT=0.
      ZM=RM**2-B
      IF(ZM.GT.4.*B)THEN
        ZM=DSQRT(ZM)
      ELSE
        ZM=2.*DSQRT(B)
      ENDIF

      DO 1 I=1,3
      Z1=ZM*(1.+X9(I))*0.5
      Z2=ZM*(1.-X9(I))*0.5
      QUQ=DSQRT(B+Z1**2)-R
      IF (QUQ.LT.85.)XXT=XXT+A9(I)/(1.+EXP(QUQ))
      QUQ=DSQRT(B+Z2**2)-R
      IF (QUQ.LT.85.)XXT=XXT+A9(I)/(1.+EXP(QUQ))
1     CONTINUE
      XXT=XXT*ZM*0.5
      DT=0.
      DO 2 I=1,2
      Z1=X5(I)+ZM
      QUQ=DSQRT(B+Z1**2)-R-X5(I)
      IF (QUQ.LT.85.)DT=DT+A5(I)/(EXP(-X5(I))+EXP(QUQ))
2     CONTINUE
      XXT=XXT+DT
      IF(DEBUG.GE.3)WRITE (MONIOU,202)XXT
202   FORMAT(2X,'XXT=',E10.3)
      RETURN
      END
C=======================================================================

        FUNCTION XXTWDEC(S,A,B)
c Kinematical function for two particle decay -
C light cone momentum share for
c the particle of mass squared A,
C B - partner's mass squared,
C S - two particle invariant mass
c-----------------------------------------------------------------------
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        INTEGER DEBUG
        COMMON /Q_AREA43/ MONIOU
        COMMON /Q_DEBUG/  DEBUG
        SAVE

        IF(DEBUG.GE.2)WRITE (MONIOU,201)S,A,B
201     FORMAT(2X,'XXTWDEC: S=',E10.3,2X,'A=',E10.3,2X,'B=',E10.3)

        X=.5D0*(1.D0+(A-B)/S)
        DX=(X*X-A/S)
        IF(DX.GT.0.D0)THEN
          X=X+DSQRT(DX)
        ELSE
          X=DSQRT(A/S)
        ENDIF
        XXTWDEC=X
        IF(DEBUG.GE.3)WRITE (MONIOU,202)XXTWDEC
202     FORMAT(2X,'XXTWDEC=',E10.3)
        RETURN
        END
C=======================================================================

      DOUBLE PRECISION FUNCTION GAMFUN(Y)
C Gamma function : See Abramowitz, page 257, form. 6.4.40
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      DOUBLE PRECISION
     +     Y,R,S,T,AFSPL,X,
     +     COEF(10),PI,ZEROD,HALFD,ONED,TWOD,TEND
      SAVE
C
      DATA COEF/8.3333333333333334D-02,-2.7777777777777778D-03,
     .          7.9365079365079365D-04,-5.9523809523809524D-04,
     .          8.4175084175084175D-04,-1.9175269175269175D-03,
     .          6.4102564102564103D-03,-2.9550653594771242D-02,
     .          0.1796443723688306    ,-0.6962161084529506    /
      DATA PI/  3.141592653589793D0/
      DATA ZEROD/0.D0/,HALFD/0.5D0/,ONED/1.D0/,TWOD/2.D0/,TEND/10.D0/
C
      X=Y
      AFSPL=ONED
      N=INT(TEND-Y)
      DO 10 I=0,N
        AFSPL=AFSPL*X
        X=X+ONED
10    CONTINUE
      R=(X-HALFD)* LOG(X)-X+HALFD* LOG(TWOD*PI)
      S=X
      T=ZEROD
      DO 20 I=1,10
        T=T+COEF(I)/S
        S=S*X**2
20    CONTINUE
      GAMFUN = EXP(R+T)/AFSPL
      END
C=======================================================================

       BLOCK DATA PSDATA
c Constants for numerical integration (Gaussian weights)
c-----------------------------------------------------------------------
       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       COMMON /Q_AR13/ X1(7),A1(7)
       COMMON /Q_AR15/ X5(2),A5(2)
       COMMON /Q_AR18/ X2(4),A2
       COMMON /Q_AR19/ X9(3),A9(3)

       DATA X1/.9862838D0,.9284349D0,.8272013D0,.6872929D0,.5152486D0,
     * .3191124D0,.1080549D0/
       DATA A1/.03511946D0,.08015809D0,.1215186D0,.1572032D0,
     * .1855384D0,.2051985D0,.2152639D0/
       DATA X2/.00960736D0,.0842652D0,.222215D0,.402455D0/
       DATA A2/.392699D0/
       DATA X5/.585786D0,3.41421D0/
       DATA A5/.853553D0,.146447D0/
       DATA X9/.93247D0,.661209D0,.238619D0/
       DATA A9/.171324D0,.360762D0,.467914D0/
       END

c following subroutine/function added 8/10/98 dh
C=======================================================================

      SUBROUTINE CROSSC(NITER,GTOT,GPROD,GABS,GDD,GQEL,GCOH)
c Nucleus-nucleus (nucleus-hydrogen) interaction cross sections
c GTOT  - total cross section
c GPROD - production cross section (projectile diffraction included)
c GABS  - cut Pomerons cross section
c GDD   - projectile diffraction cross section
c GQEL  - quasielastic (projectile nucleon knock-out) cross section
c GCOH  - coherent (elastic with respect to the projectile) cross section
c (target diffraction is not treated explicitely and contributes to
c GDD, GQEL, GCOH).
c-------------------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
cdh   DIMENSION WABS(8),WDD(8),WQEL(8),WCOH(8),WTOT(8),
cdh  *WPROD(8),B0(8),XA(64,3),XB(64,3),AI(8)
      DIMENSION WABS(20),WDD(20),WQEL(20),WCOH(20),WTOT(20),
     *WPROD(20),B0(20),AI(20),XA(210,3),XB(210,3)
      COMMON /Q_AREA1/  IA(2),ICZ,ICP
      COMMON /Q_AREA6/  PI,BM,AM
      COMMON /Q_AREA16/ CC(5)
      COMMON /Q_AR13/    X1(7),A1(7)
      COMMON /Q_AR15/    X5(2),A5(2)
      COMMON /Q_AR19/    X9(3),A9(3)
      SAVE
      EXTERNAL PSRAN

      E1=EXP(-1.D0)

cdh   DO I=1,3
cdh     B0(7-I)=BM*SQRT((1.+X9(I))/2.)
cdh     B0(I)=BM*SQRT((1.-X9(I))/2.)
cdh     AI(I)=A9(I)*(BM*AM)**2*5.*PI
cdh     AI(7-I)=AI(I)
      DO I=1,7
        B0(15-I)=BM*SQRT((1.+X1(I))/2.)
        B0(I)=BM*SQRT((1.-X1(I))/2.)
        AI(I)=A1(I)*(BM*AM)**2*5.*PI
        AI(15-I)=AI(I)
      ENDDO

cdh   DO I=1,2
cdh     B0(6+I)=BM+X5(I)
cdh     AI(6+I)=A5(I)*B0(I)*EXP(X5(I))*20.*AM**2*PI
      DO I=1,3
        TP=(1.+X9(I))/2.
        TM=(1.-X9(I))/2.
        B0(14+I)=BM-LOG(TP)
        B0(21-I)=BM-LOG(TM)
        AI(14+I)=A9(I)*B0(14+I)/TP*10.*AM**2*PI
        AI(21-I)=A9(I)*B0(21-I)/TM*10.*AM**2*PI
      ENDDO

cdh   DO I=1,8
      DO I=1,20
        WABS(I)=0.
        WDD(I)=0.
        WQEL(I)=0.
        WCOH(I)=0.
      ENDDO

      DO 1 NC=1,NITER
        NT=0
        DO I=1,IA(2)
          NT=NT+INT(PSRAN(B10)+CC(2))
        ENDDO
        IF(NT.EQ.0)GOTO 1
        IF(IA(1).EQ.1)THEN
          XA(1,1)=0.D0
          XA(1,2)=0.D0
          XA(1,3)=0.D0
        ELSE
          CALL PSGEA(IA(1),XA,1)
        ENDIF
        IF(IA(2).EQ.1)THEN
          XB(1,1)=0.D0
          XB(1,2)=0.D0
          XB(1,3)=0.D0
        ELSE
          CALL PSGEA(IA(2),XB,2)
        ENDIF

cdh     DO I=1,8
        DO I=1,20
          CALL GAUCR(B0(I),GABS,GDD,GQEL,GCOH,XA,XB,IA(1),NT)
          WABS(I)=WABS(I)+GABS
          WDD(I)=WDD(I)+GDD
          WQEL(I)=WQEL(I)+GQEL
          WCOH(I)=WCOH(I)+GCOH
        ENDDO
1     CONTINUE

      GABS=0.
      GDD=0.
      GQEL=0.
      GCOH=0.
cdh   DO I=1,8
      DO I=1,20
        WABS(I)=WABS(I)/NITER
        WDD(I)=WDD(I)/NITER
        WQEL(I)=WQEL(I)/NITER
        WCOH(I)=WCOH(I)/NITER
        WPROD(I)=WABS(I)+WDD(I)
        WTOT(I)=WPROD(I)+WQEL(I)+WCOH(I)
        GABS=GABS+AI(I)*WABS(I)
        GDD=GDD+AI(I)*WDD(I)
        GQEL=GQEL+AI(I)*WQEL(I)
        GCOH=GCOH+AI(I)*WCOH(I)
      ENDDO
      GPROD=GABS+GDD
      GTOT=GPROD+GQEL+GCOH
      RETURN
      END

c following subroutine/function added 8/10/98 dh
C=======================================================================

      SUBROUTINE GAUCR(B,GABS,GDD,GQEL,GCOH,XA,XB,IA,NT)
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION XA(210,3),XB(210,3)
      COMMON /Q_AREA15/ FP(5),RQ(5),CD(5)
      COMMON /Q_AREA16/ CC(5)
      SAVE

      GABS=1.
      GDD=1.
      GQEL=1.
      GCOH=1.
      DO N=1,IA
        VV=1.D0-DSQRT(PSV(XA(N,1)+B,XA(N,2),XB,NT))
        GABS=GABS*(1.-CC(2)*(1.-VV*VV))
        GDD=GDD*(1.-CC(2)*(1.-VV))**2
        GQEL=GQEL*(1.-2.D0*CC(2)*(1.-VV))
        GCOH=GCOH*(1.-CC(2)*(1.-VV))
      ENDDO
      GCOH=1.-2.*GCOH+GQEL
      GQEL=GDD-GQEL
      GDD=GABS-GDD
      GABS=1.-GABS
      RETURN
      END

c following subroutine/function added 8/10/98 dh
C=======================================================================

      DOUBLE PRECISION FUNCTION SECTNU(E0N,IAP,IAT)
c Nucleus-nucleus (nucleus-hydrogen) particle production cross section
c E0N - lab. energy per projectile nucleon,
c IAP - projectile mass number (2<IAP<210)
c IAT - target mass number     (1<IAT<210)
c-----------------------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION WK(3),WA(3),WB(3)
      COMMON /Q_AREA48/ QGSASECT(10,6,4)
      SAVE

      SECTNU=0.D0
      YE=DLOG10(E0N)
      IF(YE.LT.1.D0)YE=1.D0
      JE=INT(YE)
      IF(JE.GT.8)JE=8

      WK(2)=YE-JE
      WK(3)=WK(2)*(WK(2)-1.D0)*.5D0
      WK(1)=1.D0-WK(2)+WK(3)
      WK(2)=WK(2)-2.D0*WK(3)

      YA=IAP
      YA=DLOG(YA/2.D0)/.69315D0+1.D0
      JA=MIN(INT(YA),4)
      WA(2)=YA-JA
      WA(3)=WA(2)*(WA(2)-1.D0)*.5D0
      WA(1)=1.D0-WA(2)+WA(3)
      WA(2)=WA(2)-2.D0*WA(3)

      YB=IAT
      YB=DLOG(YB)/1.38629D0+1.D0
      JB=MIN(INT(YB),2)
      WB(2)=YB-JB
      WB(3)=WB(2)*(WB(2)-1.D0)*.5D0
      WB(1)=1.D0-WB(2)+WB(3)
      WB(2)=WB(2)-2.D0*WB(3)

      DO I=1,3
      DO M=1,3
      DO L=1,3
        SECTNU=SECTNU+QGSASECT(JE+I-1,JA+M-1,JB+L-1)*WK(I)*WA(M)*WB(L)
      ENDDO
      ENDDO
      ENDDO
      SECTNU=EXP(SECTNU)
      RETURN
      END

