C-+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C- Version EDDE 2.1.2 /R.Ryutin,A.Sobol ++++++++++++++++++++++
C-+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C-+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      PROGRAM  MAIN 
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-...global EDDE parameters
      INTEGER MXGLPAR      
      REAL EDDEPAR
      PARAMETER   (MXGLPAR=200)
      COMMON /EDDEGLPAR/ EDDEPAR(MXGLPAR)

c-- initialization 
      CALL EDDEINI
c
      NTOT=EDDEPAR(2)       
      DO NEV=1,NTOT
       CALL EDDEEVE
       IF(EDDEPAR(3).EQ.1) CALL EDDE_PYUPEV
c       IF(EDDEPAR(3).EQ.1) CALL PYUPEV
       IF(NEV.LE.3)        CALL PYLIST(1)
      ENDDO 

C...Final statistics.
      CALL PYSTAT(1)

C...Produce final Les Houches Event File.
      IF(EDDEPAR(3).EQ.1) CALL PYLHEF
    
      STOP
      END            
c--------------------------------------------------------------------
 
      SUBROUTINE EDDESW
C...global EDDE parameters
      INTEGER     MXGLPAR      
      REAL EDDEPAR
      PARAMETER   (MXGLPAR=200)
      COMMON /EDDEGLPAR/ EDDEPAR(MXGLPAR)
c
c      CALL EDDEDEF

      RETURN
      END
C
c----------------------------------------------------------------------              
c-*... EDDE initialization ! A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE EDDEINI
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
C-... model parameters
      EXTERNAL EDDEDATA
c-...global EDDE parameters
      INTEGER     MXGLPAR      
      REAL EDDEPAR
      PARAMETER   (MXGLPAR=200)
      COMMON /EDDEGLPAR/ EDDEPAR(MXGLPAR)
c-...some EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)     
c----- parameters to calculate total cross-sections -------      
c----- RS1 parameters -------------------------------------
      INTEGER NRS0 
      DOUBLE PRECISION RSXI0,RSGAM0,RSMH0,RSMR0 
      COMMON/EDDERS1/ RSXI0,RSGAM0,RSMH0,RSMR0,NRS0
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c-...some EDDE functions
      DOUBLE PRECISION EDDECS
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
      EXTERNAL PYDATA
      INTEGER  PYCOMP
      COMMON /PYJETS/ N, NPAD, K(4000,5), P(4000,5), V(4000,5)
      COMMON /PYDAT1/ MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYDAT3/ MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON /PYSUBS/ MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYDATR/ MRPY(6),RRPY(100)  
C-...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/
C      
      CALL EDDEDEF                ! Read default parameters for generation

      CALL EDDEPUTDAT ! to include the data
      CALL EDDETITLE ! title page for the generator

      AMP   =PMAS(PYCOMP(2212),1)                    !proton mass
C--
c      CALL PYINIT('USER','p','p',SQS)            !PYTHIA < 6.403
C-...incoming protons                           !PYTHIA >= 6.404
       K(1,1)=11
       K(1,2)=2212
       P(1,1)=0.D0
       P(1,2)=0.D0
       P(1,4)=SQS/2       
       P(1,3)=DSQRT(P(1,4)**2 - AMP**2)
       P(1,5)=AMP 
       K(2,1)=11
       K(2,2)=2212
       P(2,1)=0.D0
       P(2,2)=0.D0
       P(2,4)=SQS/2  
       P(2,3)=-DSQRT(P(2,4)**2 - AMP**2) 
       P(2,5)=AMP 
       MSTP(127)=1
      CALL PYINIT('3MOM','p','p',SQS)            !PYTHIA >= 6.404
c
C
      MSUB(IPROC)=1
      PARI(1)   = EDDECS(IPROC)*1.D-12      ! cross section in mb
      PARI(7)   = 1.D0 
      IDBMUP(1) = 2212 
      IDBMUP(2) = 2212 
      EBMUP(1)  = SQS/2
      EBMUP(2)  = SQS/2
      IDWTUP    = 1
      IF(IPROC.EQ.440)   IDWTUP    = 3
      IF(IPROC.EQ.446)   IDWTUP    = 3  
      NPRUP     = 1
      LPRUP(1)  = IPROC
      XSECUP(1) = 1.D0 
      XMAXUP(1) = 1.D0 
      XSEC(IPROC,3) = PARI(1)
      NGEN(IPROC,3) = 0
      XSECUP(IPROC) = PARI(1)*1.D+09
      WRITE(*,*)' CS(',IPROC,')=',PARI(1)*1.D+12,' fb'

cccccccccccccccccccccccccccccccccccccccccc
C...Temporary files for initialization/event output.
      IF(EDDEPAR(3).EQ.1) THEN
       MSTP(161)=77
       OPEN(77,FILE='edde.init',STATUS='unknown')
       MSTP(162)=78
       OPEN(78,FILE='edde.evnt',STATUS='unknown')

C...Final Les Houches Event File, obtained by combining above two.
       MSTP(163)=79
       OPEN(79,FILE='edde.lhe',STATUS='unknown')
c      MSTP(164) = 1
       CALL EDDE_PYUPIN
c      CALL PYUPIN
      ENDIF
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      RETURN
      END
c----------------------------------------------------------------------              
c-*... read initual parameters from edde.ffr card file /A.Sobol
c----------------------------------------------------------------------              
      SUBROUTINE EDDEDEF
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-...global EDDE parameters
      INTEGER     MXGLPAR      
      REAL EDDEPAR
      PARAMETER   (MXGLPAR=200)
      COMMON /EDDEGLPAR/ EDDEPAR(MXGLPAR)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
      EXTERNAL PYDATA
      COMMON /PYDAT1/ MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYDAT3/ MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON /PYSUBS/ MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON /PYDATR/ MRPY(6),RRPY(100) 
C
      EDDEPAR(1)=440  ! code of the process
      EDDEPAR(2)= 10000           ! number of events to generate
      EDDEPAR(3)= 1             ! key for Les Houches data(1-save,0-no) 
      EDDEPAR(4)=14000   ! pp centre mass energy in GeV 
      EDDEPAR(5)=25            ! code of the central particle M
      EDDEPAR(6)=120  ! mass in GeV of the central particle M
      EDDEPAR(7)=1   ! key for J^PC of the central particle
      EDDEPAR(8)=20  ! transverse mass cut
      EDDEPAR(9)=0   ! type of the central particle in RS1 model',
      EDDEPAR(10)=0.16 ! mixing parameter
      EDDEPAR(11)=0.246   ! scale parameter 
      EDDEPAR(12)=150   ! "bare" mass of Higgs
      EDDEPAR(13)=110   ! "bare" mass of Radion
      EDDEPAR(14)=10.1     ! pseudirapidity interval for "soft" radiation
      EDDEPAR(15)=4.8     ! mass of the final "hard" quark
        MRPY(1)=77123456          ! State of random number generator   
        MSEL =0                 !full user control
C...some PYTHIA definitions...
        MSTP (61) =1             ! Initial-state QCD and QED radiation
        MSTP (71) =1             ! Final-state QCD and QED radiation
        MSTP (81) =1             ! multiple interaction
        MSTP (111)=1             ! fragmentation and decay
        MSTP (122)=0             ! switch off X section print out
C...Higgs decay definition...
        MDME (210,1) =0           ! h0 -> d dbar
        MDME (211,1) =0           ! h0 -> u ubar
        MDME (212,1) =0           ! h0 -> s sbar
        MDME (213,1) =0           ! h0 -> c cbar
        MDME (214,1) =1           ! h0 -> b bbar 
        MDME (215,1) =0           ! h0 -> t tbar  
        MDME (216,1) =-1          ! h0 -> b' b'bar
        MDME (217,1) =-1          ! h0 -> t' t'bar
        MDME (218,1) =0           ! h0 -> e+e-
        MDME (219,1) =0           ! h0 -> mu+mu- 
        MDME (220,1) =0           ! h0 -> tau+tau-
        MDME (221,1) =-1          ! h0 -> tau'+ tau'-
        MDME (222,1) =0           ! h0 ->  gg 
        MDME (223,1) =0           ! h0-> gamma gamma
        MDME (224,1) =0           ! h0 -> gamma Z0  
        MDME (225,1) =0           ! h0 -> Z0 Z0  
        MDME (226,1) =0           ! h0 -> W+W-        
C     
      RETURN
      END
C
c-!!+++++++++++++ EDDE2.1 SUBROUTINES ++++++++++++++++++++++++
c------------------------------------------------------------- 
C-++++++++++++++++++++++++++++++++++++++++++++++
C- EDDE2.1: PYTHIA interface - event generation
C-++++++++++++++++++++++++++++++++++++++++++++++
c-----------------------------------------------------------------              
c-*... event generation/A.Sobol,R.Ryutin
c-----------------------------------------------------------------              
      SUBROUTINE EDDEEVE
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)

c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c-...global EDDE parameters from FFR file
      INTEGER MXGLPAR
      REAL EDDEPAR
      PARAMETER   (MXGLPAR=200)
      COMMON /EDDEGLPAR/ EDDEPAR(MXGLPAR)

c--- call the data one time from EDDEINI!!!
c- CALL EDDEPUTDAT

C-...incoming protons
       KK(1,1)=11
       KK(1,2)=2212
       PP(1,1)=0.D0
       PP(1,2)=0.D0
       PP(1,4)=SQS/2       
       PP(1,3)=DSQRT(PP(1,4)**2 - AMP**2)
       PP(1,5)=AMP 
       KK(2,1)=11
       KK(2,2)=2212
       PP(2,1)=0.D0
       PP(2,2)=0.D0
       PP(2,4)=SQS/2  
       PP(2,3)=-DSQRT(PP(2,4)**2 - AMP**2) 
       PP(2,5)=AMP 
     
c      print*,'EDDEEVE',IPROC
c       KDUMMY=IPROC

      IF(IPROC.EQ.440) CALL EDDERES              ! pp -> pHp (pH*p,pR*p)
      IF(IPROC.EQ.441) CALL EDDEQQ               ! pp -> p QQbar p, Q=b default
      IF(IPROC.EQ.442) CALL EDDEGG               ! pp -> p gg p
      IF(IPROC.EQ.443) CALL EDDE2GAM          ! pp -> p gammagamma p
      IF(IPROC.EQ.444) CALL EDDEQQG            ! pp -> p QQbarg g p
      IF(IPROC.EQ.445) CALL EDDE3G               ! pp -> p gg g p
C- SEMI-INCLUSIVE ------ 
      IF(IPROC.EQ.446) CALL SIDDERES              ! pp -> p{X H Y}p (p{X H*,R* Y}p)
      IF(IPROC.EQ.447) CALL SIDDEQQ                ! pp -> p {X QQbar Y} p
      IF(IPROC.EQ.448) CALL SIDDEGG                ! pp -> p {X gg Y} p
      IF(IPROC.EQ.449) CALL SIDDE2GAM            ! pp -> p {X gammagamma Y} p
C-- reserved for the future version
C-      IF(IPROC.EQ.410) CALL SIDDEQQG          ! pp -> p {X QQbarg g Y} p
C-      IF(IPROC.EQ.411) CALL SIDDE3G               ! pp -> p {X gg g Y} p
C-      
      IF(IPROC.EQ.500) CALL EDDEFLAT           ! jet with flat E vs eta distr. 

      IF(EDDEPAR(3).NE.1) CALL PYEXEC

      RETURN
      END
      
c----------------------------------------------------------------------              
c-*...exclusive resonance generation /A.Sobol,R.Ryutin 
c----------------------------------------------------------------------              
      SUBROUTINE EDDERES
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
      
 1    CONTINUE
      X=EDDEX(AM0)
      CALL EDDETTPHI(NFI,AM0,T1,T2,FI0) 
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)
      XF1 = 1.D0-X
      XF2 = 1.D0-(AM0*AM0+DT12)/S/X           
      IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
      IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
      ENDIF
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...central particle
      KK(5,1) = 1
      KK(5,2) = KCP
      KK(5,3) = 0 
      DO I=1,3
       PP(5,I)=-(PP(3,I)+PP(4,I))
      ENDDO
      PP(5,4) = DSQRT(PP(5,1)**2+PP(5,2)**2+PP(5,3)**2+AM0**2)
      PP(5,5) = AM0
      NN = 5 ! number of particles and systems 
      MINT(1)=440

      RETURN
      END        
c----------------------------------------------------------------------              
c-*... exclusive Q Qbar generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE EDDEQQ
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTQ2,MXX,PQ1(5),PQ2(5),PMX(5),PVS(5)
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
        PRINT*,'EDDEQQ:Attention: generator works in the ETcut'
        PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
       ENDIF
      ENDIF     

  1   CONTINUE
      CALL GENEREXQQ(MQ,MXX,ETAJ)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c---
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0) 
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)      
      PKAP = 1.D0/DCOSH(ETAJ)**2
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MXX*MXX/(PMX(5)*PMX(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF
       MXX=PMX(5)
       MTQ2 = MXX*MXX*PKAP/4.D0
C-... QQbar system in the X-rest frame
C-... parton1 - momentum
      PQ1(5) = PMAS(5,1)
      PT=DSQRT(MTQ2-PQ1(5)**2)
      FI  = 2.*PI*PYR(0)
      PQ1(1)=PT*DCOS(FI)
      PQ1(2)=PT*DSIN(FI)
      PQ1(3)=SIGNETA*MXX*DSQRT(1.D0-PKAP)/2.D0
      PQ1(4)=DSQRT(PQ1(1)**2+PQ1(2)**2+PQ1(3)**2+PQ1(5)**2) 
C-... parton2 - momentum
      PQ2(5) = PMAS(5,1) 
      DO I=1,3
       PQ2(I)=-PQ1(I)
      ENDDO
      PQ2(4)=DSQRT(PQ2(1)**2+PQ2(2)**2+PQ2(3)**2+PQ2(5)**2)   
C-... QQbar system in the lab. frame
C-... lorentz transformation
      CALL XCMTOLAB(PMX,PQ1,PVS)
      DO I=1,5
       PQ1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMX,PQ2,PVS)
      DO I=1,5
       PQ2(I)=PVS(I)
      ENDDO
C-... equal probability for Q and Qbar
      IF(PYR(0).LT.0.5) THEN
       DO I=1,5
        PVS(I)=PQ1(I)
        PQ1(I)=PQ2(I)
        PQ2(I)=PVS(I)
       ENDDO
      ENDIF       
C-...Q - quark and Qbar quark in the lab. frame
      DO I=1,5
       PP(6,I)=PQ1(I)
       PP(7,I)=PQ2(I)
      ENDDO 
C-...Q Qbar system and its decay products definition.
C-...B Bbar by default (change to other numbers,
C-... insert option to ffr-file!!!)
      DO I=1,4
       PP(5,I) = PP(6,I) + PP(7,I) 
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)
      DO I=1,5
       PQ1(I) = PP(6,I)  
       PQ2(I) = PP(7,I)  
      ENDDO
       CALL PY2ENT(-6,5,-5,PP(5,5)) 
      DO I=1,5
       PP(6,I) = PQ1(I)
       PP(7,I) = PQ2(I) 
      ENDDO
      NN=7
      KK(5,1)=11
      KK(5,2)=90 
      KK(5,3)=0
      KK(5,4)=6
      KK(5,5)=7
      KK(6,3)=5
      KK(7,3)=5
      MINT(1)=441
c      CALL PYSHOW(6,7,PP(5,5)) 

      RETURN
      END
c----------------------------------------------------------------------              
c- *... exclusive g g generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE EDDEGG
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c- *...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)

c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTG2,MXX,PG1(5),PG2(5),PMX(5),PVS(5)
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
        PRINT*,'EDDEGG:Attention: generator works in the ETcut'
        PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
       ENDIF
      ENDIF   

 1    CONTINUE
      CALL GENEREXGG(MXX,ETAJ)
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c---
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)      
      PKAP = 1.D0/DCOSH(ETAJ)**2
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MXX*MXX/(PMX(5)*PMX(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF
       MXX=PMX(5)
       MTG2 = MXX*MXX*PKAP/4.D0
C-... gg system in the X-rest frame
C-... parton1 - momentum
      PG1(5) = PMAS(21,1)
      PT=DSQRT(MTG2-PG1(5)**2)
      FI  = 2.*PI*PYR(0)
      PG1(1)=PT*DCOS(FI)
      PG1(2)=PT*DSIN(FI)
C-...PG1(3)=MTG*DSINH(ETAJ) 
      PG1(3)=SIGNETA*MXX*DSQRT(1.D0-PKAP)/2.D0
      PG1(4)=DSQRT(PG1(1)**2+PG1(2)**2+PG1(3)**2+PG1(5)**2) 
C-... parton2 - momentum
      PG2(5) = PMAS(21,1) 
      DO I=1,3
       PG2(I)=-PG1(I)
      ENDDO
      PG2(4)=DSQRT(PG2(1)**2+PG2(2)**2+PG2(3)**2+PG2(5)**2)
C-... gg system in the lab. frame
C-... lorentz transformation
      CALL XCMTOLAB(PMX,PG1,PVS)
      DO I=1,5
       PG1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMX,PG2,PVS)
      DO I=1,5
       PG2(I)=PVS(I)
      ENDDO    

c- system in the lab.      
      DO I=1,4
       PP(5,I) = PG1(I) + PG2(I) 
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)
C-...gg in the lab. frame
      DO I=1,5
       PP(6,I)=PG1(I)
       PP(7,I)=PG2(I)
      ENDDO 
C-...gg-system and its decay products definition 
       CALL PY2ENT(-6,21,21,PP(5,5)) 
      DO I=1,5
       PP(6,I) = PG1(I)
       PP(7,I) = PG2(I)
      ENDDO  
      NN=7
      KK(5,1)=11
      KK(5,2)=90 
      KK(5,3)=0
      KK(5,4)=6
      KK(5,5)=7
      KK(6,3)=5
      KK(7,3)=5
      MINT(1)=442
c      CALL PYSHOW(6,7,PP(5,5)) 
        
      RETURN
      END

c----------------------------------------------------------------------              
c-*... exclusive 2gamma generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE EDDE2GAM
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)

c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTG2,MXX,PG1(5),PG2(5),PMX(5),PVS(5)
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
        PRINT*,'EDDE2GAM:Attention: generator works in the ETcut'
        PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
       ENDIF
      ENDIF   
C
 1    CONTINUE
      CALL GENEREX2GAM(MXX,ETAJ)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c---      
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0) 
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)      
      PKAP = 1.D0/DCOSH(ETAJ)**2
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MXX*MXX/(PMX(5)*PMX(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF
       MXX=PMX(5)
       MTG2 = MXX*MXX*PKAP/4.D0
C-... gamma gamma system in the X-rest frame
C-... parton1 - momentum
      PG1(5) = PMAS(22,1)
      PT=DSQRT(MTG2-PG1(5)**2)
      FI  = 2.*PI*PYR(0)
      PG1(1)=PT*DCOS(FI)
      PG1(2)=PT*DSIN(FI)
      PG1(3)=SIGNETA*MXX*DSQRT(1.D0-PKAP)/2.D0
      PG1(4)=DSQRT(PG1(1)**2+PG1(2)**2+PG1(3)**2+PG1(5)**2) 
C-... parton2 - momentum
      PG2(5) = PMAS(22,1) 
      DO I=1,3
       PG2(I)=-PG1(I)
      ENDDO
      PG2(4)=DSQRT(PG2(1)**2+PG2(2)**2+PG2(3)**2+PG2(5)**2)
C-... gamma gamma system in the lab. frame
C-... lorentz transformation
      CALL XCMTOLAB(PMX,PG1,PVS)
      DO I=1,5
       PG1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMX,PG2,PVS)
      DO I=1,5
       PG2(I)=PVS(I)
      ENDDO    
C-...gamma gamma in the lab. frame
      DO I=1,5
       PP(6,I) = PG1(I) 
       PP(7,I) = PG2(I) 
      ENDDO
C-...gamma gamma-system definition 
      DO I=1,4
       PP(5,I) = PG1(I) + PG2(I) 
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)
       CALL PY2ENT(6,22,22,PP(5,5)) 
      DO I=1,5
       PP(6,I) = PG1(I)
       PP(7,I) = PG2(I)
      ENDDO  
c      CALL PYSHOW(6,7,PP(5,5)) 
       NN=7
       KK(5,1)=11
       KK(5,2)=90 
       KK(6,1)=1
       KK(6,2)=22 
       KK(6,3)=5        
       KK(7,1)=1
       KK(7,2)=22 
       KK(7,3)=5
       MINT(1)=443
        
      RETURN
      END

c----------------------------------------------------------------------              
c-*... exclusive Q Qbar g generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE EDDEQQG
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)

c-...some of global EDDE variables
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
       INTEGER NAPR,NFI
       DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
       COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
       DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
       COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTQ2, MXX,MJJ
      DOUBLE PRECISION PQ1(5),PQ2(5),PG3(5),PMX(5),PVS(5)
       LOGICAL FIRST
       DATA FIRST /.TRUE./
       SAVE FIRST

       IF(FIRST) THEN
        IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
         PRINT*,'EDDEQQG:Attention: generator works in the ETcut'
         PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
        ENDIF
       ENDIF   

c-------------------------------------------------------------------

 1    CONTINUE
      CALL GENEREXQQG(MJJ,ETAJ,XG3,FIS,THETAS)
      MXX=MJJ/DSQRT(1.D0-XG3)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c---            
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0)
      PKAP = 1.D0/DCOSH(ETAJ)**2
      DT12 = DABS(T1)+DABS(T2)
      DT12 = DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2 
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)
       MJJ=PMX(5)*DSQRT(1-XG3)
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MXX*MXX/(PMX(5)*PMX(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF
       MXX=PMX(5)
C------ momenta of partons in the rest X-system frame ----------
C------ max-E quark
      PQ1(5) = PMAS(5,1)
      AUX1=(2.D0-XG3)**2-(XG3*DCOS(THETAS))**2
      AUX2=DSQRT(1.D0-AUX1*(PQ1(5)/(MXX*(1-XG3)))**2)
      XQ1=2.D0*(1-XG3)*(2.D0-XG3+XG3*DABS(DCOS(THETAS))*AUX2)/AUX1
      MTQ2 = (XQ1*MXX*0.5D0)**2*PKAP
      PT=DSQRT(MTQ2-PQ1(5)**2)
c------------------------------------    
      PFULL=DSQRT((XQ1*MXX*0.5D0)**2-PQ1(5)**2)
C---- new value of THETA_max-E parton in the rest X-system frame
      THETA2=DACOS(SIGNETA*DSQRT(PFULL**2-PT**2)/PFULL)
C------ parton 1 momentum -------------------------------------      
      FI  = 2.*PI*PYR(0)
      PQ1(1)= PT*DCOS(FI)
      PQ1(2)= PT*DSIN(FI)
      PQ1(3)= SIGNETA*DSQRT(PFULL**2-PT**2)
      PQ1(4)=DSQRT(PQ1(1)**2+PQ1(2)**2+PQ1(3)**2+PQ1(5)**2)       
C---  gluon (parton 3) momentum ---------------------
      PG3(5) = PMAS(21,1)
      PG3(4) = XG3*MXX/2.D0
      PFULL  = DSQRT(PG3(4)**2-PG3(5)**2)      
      AUX1 = DSIN(THETAS)*DSIN(FI)*DSIN(FIS)
      AUX1 = AUX1-DSIN(THETAS)*DCOS(THETA2)*DCOS(FI)*DCOS(FIS)
      AUX1 = AUX1-DCOS(THETAS)*DSIN(THETA2)*DCOS(FI)
      AUX2 = -DCOS(THETAS)*DSIN(THETA2)*DSIN(FI)
      AUX2 = AUX2-DSIN(THETAS)*DCOS(THETA2)*DSIN(FI)*DCOS(FIS)
      AUX2 = AUX2-DSIN(THETAS)*DCOS(FI)*DSIN(FIS)
      AUX3 = -DCOS(THETAS)*DCOS(THETA2)
      AUX3 = AUX3+DSIN(THETAS)*DSIN(THETA2)*DCOS(FIS)
      PG3(1) = PFULL*AUX1     
      PG3(2) = PFULL*AUX2
      PG3(3) = PFULL*AUX3
C-- parton 2 momentum ------------------------------------------
      DO I=1,3
       PQ2(I)=-(PQ1(I)+PG3(I))
      ENDDO
      PQ2(5)= PMAS(5,1) 
      PQ2(4)= DSQRT(PQ2(1)**2+PQ2(2)**2+PQ2(3)**2+PQ2(5)**2)
C-- variables for PY3ENT
      XQ2=2.D0-XG3-XQ1
C-- lorentz transformation of momenta to the lab.      
      CALL XCMTOLAB(PMX,PQ1,PVS)
      DO I=1,5
      PQ1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMX,PG3,PVS)
      DO I=1,5
      PG3(I)=PVS(I)
      ENDDO
      CALL XCMTOLAB(PMX,PQ2,PVS)
      DO I=1,5
      PQ2(I)=PVS(I)
      ENDDO      
C-...interchange between Q and Qbar (equal probability)
      IF (PYR(0).LT.0.5) THEN
       DO I=1,5
        PVS(I)=PQ1(I)
        PQ1(I)=PQ2(I)
        PQ2(I)=PVS(I)
        XQSAVE=XQ1
        XQ1=XQ2
        XQ2=XQSAVE
       ENDDO
      ENDIF
C-... PYTHIA: definition of momenta      
       DO I=1,5
        PP(6,I) = PQ1(I)
        PP(7,I) = PG3(I)
        PP(8,I) = PQ2(I)
       ENDDO      
c-...Q Qbar g system and its decay products definition.
      DO I=1,4
       PP(5,I) = PQ1(I) + PQ2(I) +PG3(I)
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)

      CALL PY3ENT(-6,5,21,-5,PP(5,5),XQ1,XQ2) 

       DO I=1,5
        PP(6,I) = PQ1(I)
        PP(7,I) = PG3(I)
        PP(8,I) = PQ2(I)
       ENDDO

      NN=8
      KK(5,1)=11
      KK(5,2)=90 
      KK(5,3)=0
      KK(5,4)=6
      KK(5,5)=8
      KK(6,3)=5
      KK(7,3)=5
      KK(8,3)=5
      MINT(1)=444

c      CALL PYSHOW(6,-3,PP(5,5)) 
        
      RETURN
      END

c----------------------------------------------------------------------              
c-... exclusive 3 g generation /R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE EDDE3G
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)

c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTG2,MXX,MJJ
      DOUBLE PRECISION PG1(5),PG2(5),PG3(5),PMX(5),PVS(5)
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

       IF(FIRST) THEN
        IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
         PRINT*,'EDDE3G:Attention: generator works in the ETcut'
         PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
        ENDIF
       ENDIF   

c--------------------------------------------------------------------

 1    CONTINUE
      CALL GENEREX3G(MJJ,ETAJ,XG3,PTG3,FIG3)
      MXX=MJJ/DSQRT(1.D0-XG3)     
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c---            
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0)
      PKAP = 1.D0/DCOSH(ETAJ)**2
      DT12 = DABS(T1)+DABS(T2)
      DT12 = DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2 
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)
       MJJ=PMX(5)*DSQRT(1-XG3)       
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MXX*MXX/(PMX(5)*PMX(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF
       MXX=PMX(5)
C---- new value of THETA_max-E parton in the rest X-system frame
      THETA2=DACOS(SIGNETA*DSQRT(1-PKAP))
      FI  = 2.*PI*PYR(0)
C------ momenta of partons in the rest X-system frame ----------      
C---  gluon (parton 3) momentum ---------------------
      PG3(5) = PMAS(21,1)
      PG3(4) = XG3*MXX/2.D0
      PFULL  = DSQRT(PG3(4)**2-PG3(5)**2)
C--- abs. value of the longitudinal momentum      
      PLG=DSQRT(PFULL**2-PTG3**2)
 2    CONTINUE
      FIG3=2.D0*PI*PYR(0)      
      CONSA=PTG3*DSIN(THETA2)*DCOS(FI-FIG3)
      CONSA=CONSA+PLG*DCOS(THETA2)
      CONSB=PTG3*DSIN(THETA2)*DCOS(FI-FIG3)
      CONSB=CONSB-PLG*DCOS(THETA2)      
C------- CONSTRAINT --------
       IF (CONSA.GE.0.D0.AND.CONSB.GE.0.D0) GOTO 2
C-------------------------------------------------------------
      PG3(1) = PTG3*DCOS(FIG3)     
      PG3(2) = PTG3*DSIN(FIG3)
      PG3(3) = PLG
C-------------------------------------------------------------     
C-       IF (CONSA.LT.0.D0.AND.CONSB.GE.0.D0) THEN
C-        PG3(3)=PLG
C-       ENDIF
       IF (CONSA.GE.0.D0.AND.CONSB.LT.0.D0) THEN
        PG3(3)=-PLG
       ENDIF       
       IF (CONSA.LT.0.D0.AND.CONSB.LT.0.D0) THEN
        IF (PYR(0).LE.0.5) THEN       
         PG3(3)=PLG
        ELSE
         PG3(3)=-PLG
        ENDIF
       ENDIF       
C------ max-E gluon
      PG1(5) = PMAS(21,1)
      AUX1=PG3(3)*DCOS(THETA2)+PTG3*DSIN(THETA2)*DCOS(FI-FIG3)
      AUX1=AUX1*2.D0/MXX
      XG1=2.D0*(1-XG3)/(2.D0-XG3+AUX1)
      MTG2 = (XG1*MXX*0.5D0)**2*PKAP
      PT=DSQRT(MTG2-PG1(5)**2)
c------------------------------------    
      PFULL=DSQRT((XG1*MXX/2.D0)**2-PG1(5)**2)
C------ parton 1 momentum -------------------------------------      
      PG1(1)= PT*DCOS(FI)
      PG1(2)= PT*DSIN(FI)
      PG1(3)= SIGNETA*DSQRT(PFULL**2-PT**2)
      PG1(4)=DSQRT(PG1(1)**2+PG1(2)**2+PG1(3)**2+PG1(5)**2) 
C-- parton 2 momentum ------------------------------------------
      DO I=1,3
       PG2(I)=-(PG1(I)+PG3(I))
      ENDDO
      PG2(5)= PMAS(21,1) 
      PG2(4)= DSQRT(PG2(1)**2+PG2(2)**2+PG2(3)**2+PG2(5)**2)
C-- variables for PY3ENT
      XG2=2.D0-XG3-XG1     
C-- lorentz transformation of momenta to the lab.      
      CALL XCMTOLAB(PMX,PG1,PVS)
      DO I=1,5
      PG1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMX,PG3,PVS)
      DO I=1,5
      PG3(I)=PVS(I)
      ENDDO
      CALL XCMTOLAB(PMX,PG2,PVS)
      DO I=1,5
      PG2(I)=PVS(I)
      ENDDO                     
C-... PYTHIA: definition of momenta      
       DO I=1,5
        PP(6,I) = PG1(I)
        PP(7,I) = PG3(I)
        PP(8,I) = PG2(I)
       ENDDO      
c-...3 g system and its decay products definition.
      DO I=1,4
       PP(5,I) = PG1(I) + PG2(I) +PG3(I)
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)

      CALL PY3ENT(-6,21,21,21,PP(5,5),XG1,XG2) 

       DO I=1,5
        PP(6,I) = PG1(I)
        PP(7,I) = PG3(I)
        PP(8,I) = PG2(I)
       ENDDO

      NN=8
      KK(5,1)=11
      KK(5,2)=90 
      KK(5,3)=0
      KK(5,4)=6
      KK(5,5)=8
      KK(6,3)=5
      KK(7,3)=5
      KK(8,3)=5
      MINT(1)=445

c      CALL PYSHOW(6,-3,PP(5,5)) 
        
      RETURN
      END

c----------------------------------------------------------------------              
c-...semi-inclusive resonance generation /A.Sobol,R.Ryutin 
c----------------------------------------------------------------------              
      SUBROUTINE SIDDERES
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- cascads parameters ----------------
      INTEGER NG1,NG2,NGSUM,NGS,CEVENT(2,500),NA2,NA3
      DOUBLE PRECISION PNG1(5,500),PNG2(5,500),PGC1(5),PGC2(5),
     & PGSUM1(5),PGSUM2(5),PVS2(500,5) 
c----- for this subroutine -----------------------------------      
      INTEGER NFAIL
      DOUBLE PRECISION MXX,PMX(5),PVS(5),PAUX(5),AUXM,PCS(5)
      
 1    CONTINUE
c--- 2 cascads generation
      CALL SICASCAD2(AM0,NG1,PNG1,PGC1,NG2,PNG2,PGC2,MXX,NFAIL)
      IF (NFAIL.EQ.1) GOTO 1
      
c--- definition
c---
      X=EDDEX(MXX)
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0) 
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)
      XF1 = 1.D0-X
      XF2 = 1.D0-(MXX*MXX+DT12)/S/X           
      IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
      IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
      ENDIF
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...central system ---------------
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      AUXM=MXX
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
       MXX=PMX(5)
c- mass correction for all momenta -------
c- and sum of final momenta -
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO
      IF (NG1.GT.0) THEN 
       DO J=1,NG1       
        DO I=1,4
         PAUX(I)=PNG1(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(1,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG1(I,J)=PVS(I)
         PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
        ENDDO
        PNG1(4,J)=DSQRT(PNG1(1,J)**2+PNG1(2,J)**2+PNG1(3,J)**2)
        PGSUM1(4)=PGSUM1(4)+PNG1(4,J)
       ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
       DO J=1,NG2       
        DO I=1,4
         PAUX(I)=PNG2(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(2,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG2(I,J)=PVS(I)
         PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
        ENDDO
        PNG2(4,J)=DSQRT(PNG2(1,J)**2+PNG2(2,J)**2+PNG2(3,J)**2)
        PGSUM2(4)=PGSUM2(4)+PNG2(4,J)
       ENDDO
      ENDIF
c- momentum of central system -
      PGC1(4)=0.5D0*MXX-PGSUM1(4)
      PGC1(3)=0.5D0*MXX-PGSUM1(3)
      PGC2(4)=0.5D0*MXX-PGSUM2(4)
      PGC2(3)=-0.5D0*MXX-PGSUM2(3)      
      DO I=1,4
         PCS(I)=PGC1(I)+PGC2(I)
      ENDDO
       AM0=DSQRT(PCS(4)**2-PCS(1)**2-PCS(2)**2-PCS(3)**2)
       PCS(5)=AM0
c----------------------------------------------------       
c-- 2 cascads momenta transformation ----------------
c-- from CMX to lab. --------------------------------
c- transformation

      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO

      IF (NG1.GT.0) THEN 
      DO J=1,NG1       
       DO I=1,4
       PAUX(I)=PNG1(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG1(I,J)=PVS(I)
       PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
       ENDDO
      ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
      DO J=1,NG2       
       DO I=1,4
       PAUX(I)=PNG2(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG2(I,J)=PVS(I)
       PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
       ENDDO     
      ENDDO
      ENDIF      
      
       DO I=1,4
       PAUX(I)=PCS(I)
       ENDDO
       PAUX(5)=PCS(5)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PCS(I)=PVS(I)
       ENDDO       
      
      DO I=1,4
      PAUX(I)=PGSUM1(I)+PGSUM2(I)+PCS(I)
      ENDDO
      PAUX(5)=DSQRT(PAUX(4)**2-PAUX(3)**2-PAUX(2)**2-PAUX(1)**2)
c---- momentum of resonance particle 
c--- number of particles (2+2)(protons)+1(systemX)+NGS+NG1+NG2
c--- NGS - number of gg or ggg singlet systems -------
c-   NN = 5+NGS+NG1+NG2
      NGS=0 ! number of singlet systems
      NGSUM=NG1+NG2 !<80
      NN=5+NGS+NGSUM 
c---      
      KK(5,1) = 1
      KK(5,2) = KCP
      KK(5,3) = 0 
      DO I=1,3
       PP(5,I)=PP(1,I)+PP(2,I)
       PP(5,I)=PP(5,I)-(PP(3,I)+PP(4,I)+PGSUM1(I)+PGSUM2(I))
      ENDDO
      PP(5,4) = DSQRT(PP(5,1)**2+PP(5,2)**2+PP(5,3)**2+AM0**2)
      PP(5,5) = AM0
c--- combinatorics of cascad gluons N singlet systems -----
c--- Formulaes for all cases:
c- number of terms in n2 couples and n3 triples combined
c- from N gluons: N!/(n2! (2!)**n2 n3! (3!)**n3)
c- total number of terms for N-even:
c- Ntot=Sum[{k=0,INT[N/6]},
c- {N!/((N/2-3*k)! (2!)**(N/2-3*k) (2*k)! (3!)**(2*k))}]
c- total number of terms for N-add:
c- Ntot=Sum[{k=0,INT[(N-3)/6]},
c- {N!/(((N-3)/2-3*k)! (2!)**((N-3)/2-3*k) (2*k+1)! (3!)**(2*k+1))}] 
c-- simple case realization --------------
c-- in the complicated case we have to define
c-- NA2+NA3 singlet systems of gluons with specified
c-- 4-momentum, i.e. call COMBSING from here!
c-      KK(6,1)=11
c-      KK(6,2)=90
c-      KK(6,3)=0
c-      KK(6,4)=0
c-      KK(6,5)=0
c-      DO I=1,4
c-       PP(6,I) = PGSUM1(I)+PGSUM2(I)
c-      ENDDO
c-       PP(6,5) = DSQRT(PP(6,4)**2-PP(6,1)**2-PP(6,2)**2-PP(6,3)**2)
c- check out
c-      WRITE(*,*)'check sum all!!!' 
c-      DO I=1,4
c-       WRITE(*,*)' psum(',I,')=',PP(3,I)+PP(4,I)+PP(5,I)+PP(6,I)
c-      ENDDO

c--- NG1+NG2 gluons
c-  KK(I,1)=2 or 1,KK(I,2)=21,KK(I,3)=0,KK(I,4)=0,KK(I,5)=0      
c- gluons permutations
       CALL COMBSING(NGSUM,NA2,NA3,CEVENT)
      DO I=6,NGSUM+5
       KK(I,1)=CEVENT(2,I-5)
       KK(I,2)=21 
       KK(I,3)=0
       KK(I,4)=0
       KK(I,5)=0
      ENDDO       
c-- all the momenta to PYTHIA              
      IF (NG1.GT.0) THEN
      DO I=6,NG1+5
       DO J=1,5 
        PP(I,J)=PNG1(J,I-5)
       ENDDO
      ENDDO
      ENDIF
      IF (NG2.GT.0) THEN      
      DO I=NG1+6,NGSUM+5
       DO J=1,5 
        PP(I,J)=PNG2(J,I-5-NG1)
       ENDDO
      ENDDO
      ENDIF
c- permutations of momenta according to CEVENT(1,I)
      DO I=1,NGSUM
       DO J=1,5
        PVS2(I,J)=PP(5+I,J)
       ENDDO       
      ENDDO
      DO I=1,NGSUM
       DO J=1,5
        PP(5+I,J)=PVS2(CEVENT(1,I),J)
       ENDDO       
      ENDDO
c--- all the gluons to PYLIST
c-      CALL PYSHOW(7,-NGSUM,PP(6,5))  
      MINT(1)=446
    
      RETURN
      END
  
c----------------------------------------------------------------------              
c-*... semi-inclusive Q Qbar generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE SIDDEQQ
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- cascads parameters ----------------
      INTEGER NG1,NG2,NGSUM,NGS,CEVENT(2,500),NA2,NA3
      DOUBLE PRECISION PNG1(5,500),PNG2(5,500),PGC1(5),PGC2(5),
     & PGSUM1(5),PGSUM2(5),PVS2(500,5)       
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTQ2,MXX,MJJ,PQ1(5),PQ2(5),
     & PMX(5),PMJJ(5),PVS(5),PAUX(5),AUXM,PCS(5)
      INTEGER NFAIL
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
        PRINT*,'EDDEQQ:Attention: generator works in the ETcut'
        PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
       ENDIF
      ENDIF     

  1   CONTINUE
      CALL GENERSIQQ(MQ,MJJ,ETAJ)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c--- 2 cascads generation
c-      WRITE(*,*)'MJJ_gen=',MJJ 
c-      WRITE(*,*)'point 1'
      CALL SICASCAD2(MJJ,NG1,PNG1,PGC1,NG2,PNG2,PGC2,MXX,NFAIL)
      IF (NFAIL.EQ.1) GOTO 1
c-      WRITE(*,*)'point 2' 
c---
c-      WRITE(*,*)'SIDDEQQ==========================='
c-      WRITE(*,*)'MJJ_OLD=',MJJ,' MX_OLD=',MXX
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0) 
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)      
      PKAP = 1.D0/DCOSH(ETAJ)**2
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      AUXM=MXX 
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
       MXX=PMX(5)
c-      WRITE(*,*)'PMX: ',PMX(4),' ',PMX(1),' ',PMX(2),' ',PMX(3),
c-     & ' ',PMX(5) 
c-      WRITE(*,*)'soft gluons before transform'
c-      IF (NG1.GT.0) THEN 
c-      DO J=1,NG1       
c-      WRITE(*,*)'PG',J,':  ',PNG1(4,J),' ',PNG1(1,J),' ',
c-     & PNG1(2,J),' ',PNG1(3,J),' ',PNG1(5,J)       
c-      ENDDO 
c-      ENDIF 
c----------------------------------------------------
c-      IF (NG2.GT.0) THEN
c-      DO J=1,NG2       
c-      WRITE(*,*)'PG',J+NG1,':  ',PNG2(4,J),' ',PNG2(1,J),' ',
c-     & PNG2(2,J),' ',PNG2(3,J),' ',PNG2(5,J)       
c-      ENDDO
c-      ENDIF
c----------------------------------------------------
c- mass correction for all momenta -------
c- and sum of final momenta -
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO
      IF (NG1.GT.0) THEN 
       DO J=1,NG1       
        DO I=1,4
         PAUX(I)=PNG1(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(1,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG1(I,J)=PVS(I)
         PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
        ENDDO
        PNG1(4,J)=DSQRT(PNG1(1,J)**2+PNG1(2,J)**2+PNG1(3,J)**2)
        PGSUM1(4)=PGSUM1(4)+PNG1(4,J)
       ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
       DO J=1,NG2       
        DO I=1,4
         PAUX(I)=PNG2(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(2,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG2(I,J)=PVS(I)
         PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
        ENDDO
        PNG2(4,J)=DSQRT(PNG2(1,J)**2+PNG2(2,J)**2+PNG2(3,J)**2)
        PGSUM2(4)=PGSUM2(4)+PNG2(4,J)
       ENDDO
      ENDIF
c- momentum of central system -
      PGC1(4)=0.5D0*MXX-PGSUM1(4)
      PGC1(3)=0.5D0*MXX-PGSUM1(3)
      PGC2(4)=0.5D0*MXX-PGSUM2(4)
      PGC2(3)=-0.5D0*MXX-PGSUM2(3)      
      DO I=1,4
         PCS(I)=PGC1(I)+PGC2(I)
      ENDDO
       PCS(5)=DSQRT(PCS(4)**2-PCS(1)**2-PCS(2)**2-PCS(3)**2)
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MJJ*MJJ/(PCS(5)*PCS(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF       
       MJJ=PCS(5)
      
c-      WRITE(*,*)'MX_NEW=',MXX,' MJJ_NEW=',MJJ
c-      WRITE(*,*)'particles before transform corrected'
c-      IF (NG1.GT.0) THEN 
c-      DO J=1,NG1       
c-      WRITE(*,*)'PG',J,':  ',PNG1(4,J),' ',PNG1(1,J),' ',
c-     & PNG1(2,J),' ',PNG1(3,J),' ',PNG1(5,J)       
c-      ENDDO 
c-      ENDIF 
c----------------------------------------------------
c-      IF (NG2.GT.0) THEN
c-      DO J=1,NG2       
c-      WRITE(*,*)'PG',J+NG1,':  ',PNG2(4,J),' ',PNG2(1,J),' ',
c-     & PNG2(2,J),' ',PNG2(3,J),' ',PNG2(5,J)       
c-      ENDDO
c-      ENDIF
c-      WRITE(*,*)'---------------------------------------------'
c-      WRITE(*,*)'PCS:    ',PCS(4),' ',PCS(1),' ',
c-     & PCS(2),' ',PCS(3),' ',PCS(5)    
c-      WRITE(*,*)'Pf1:    ',PGC1(4),' ',PGC1(1),' ',
c-     & PGC1(2),' ',PGC1(3),' ',PGC1(5)
c-      WRITE(*,*)'Pf2:    ',PGC2(4),' ',PGC2(1),' ',
c-     & PGC2(2),' ',PGC2(3),' ',PGC2(5)
c-      WRITE(*,*)'--------------------------------------------'
c-      WRITE(*,*)'SUM:    ',PCS(4)+PGSUM1(4)+PGSUM2(4),' ',
c-     & PCS(1)+PGSUM1(1)+PGSUM2(1),' ',
c-     & PCS(2)+PGSUM1(2)+PGSUM2(2),' ',
c-     & PCS(3)+PGSUM1(3)+PGSUM2(3),' ',
c-     & DSQRT((PCS(4)+PGSUM1(4)+PGSUM2(4))**2-
c-     & (PCS(3)+PGSUM1(3)+PGSUM2(3))**2-
c-     & (PCS(2)+PGSUM1(2)+PGSUM2(2))**2-
c-     & (PCS(1)+PGSUM1(1)+PGSUM2(1))**2)
c----------------------------------------------------       
C--- momentum of JJ-system in the gg C.M. -----------------
C--- and new value of MJJ ---------------------------------
      DO I=1,5   
       PMJJ(I)=PCS(I)
      ENDDO
c-- 2 cascads momenta transformation ----------------
c-- from CMX to lab. --------------------------------
c- and sum of final momenta -
      
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO

c- transformation      
      IF (NG1.GT.0) THEN 
      DO J=1,NG1       
       DO I=1,4
       PAUX(I)=PNG1(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG1(I,J)=PVS(I)
       PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
       ENDDO 
      ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
      DO J=1,NG2       
       DO I=1,4
       PAUX(I)=PNG2(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG2(I,J)=PVS(I)
       PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
       ENDDO 
      ENDDO
      ENDIF 
C-...        
c-------
       MTQ2=MJJ*MJJ*PKAP/4.D0
C-... QQbar system in the JJ-rest frame
C-... parton1 - momentum
      PQ1(5) = PMAS(5,1)
      PT=DSQRT(MTQ2-PQ1(5)**2)
      FI  = 2.*PI*PYR(0)
      PQ1(1)=PT*DCOS(FI)
      PQ1(2)=PT*DSIN(FI)
      PQ1(3)=SIGNETA*MJJ*DSQRT(1.D0-PKAP)/2.D0
      PQ1(4)=DSQRT(PQ1(1)**2+PQ1(2)**2+PQ1(3)**2+PQ1(5)**2) 
C-... parton2 - momentum
      PQ2(5) = PMAS(5,1) 
      DO I=1,3
       PQ2(I)=-PQ1(I)
      ENDDO
      PQ2(4)=DSQRT(PQ2(1)**2+PQ2(2)**2+PQ2(3)**2+PQ2(5)**2)   
C-... QQbar system in the lab. frame
C-... lorentz transformation
      CALL XCMTOLAB(PMJJ,PQ1,PAUX)
      CALL XCMTOLAB(PMX,PAUX,PVS)
      DO I=1,5
       PQ1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMJJ,PQ2,PAUX)
      CALL XCMTOLAB(PMX,PAUX,PVS)
      DO I=1,5
       PQ2(I)=PVS(I)
      ENDDO
C-... equal probability for Q and Qbar
      IF(PYR(0).LT.0.5) THEN
       DO I=1,5
        PVS(I)=PQ1(I)
        PQ1(I)=PQ2(I)
        PQ2(I)=PVS(I)
       ENDDO
      ENDIF
c-      WRITE(*,*)'particles after transform'
c-      IF (NG1.GT.0) THEN 
c-      DO J=1,NG1       
c-      WRITE(*,*)'PG',J,':  ',PNG1(4,J),' ',PNG1(1,J),' ',
c-     & PNG1(2,J),' ',PNG1(3,J),' ',PNG1(5,J)       
c-      ENDDO 
c-      ENDIF 
c----------------------------------------------------
c-      IF (NG2.GT.0) THEN
c-      DO J=1,NG2       
c-      WRITE(*,*)'PG',J+NG1,':  ',PNG2(4,J),' ',PNG2(1,J),' ',
c-     & PNG2(2,J),' ',PNG2(3,J),' ',PNG2(5,J)       
c-      ENDDO
c-      ENDIF
c-      WRITE(*,*)'---------------------------------------------'
c-      WRITE(*,*)'PQ1:   ',PQ1(4),' ',PQ1(1),' ',
c-     & PQ1(2),' ',PQ1(3),' ',PQ1(5)    
c-      WRITE(*,*)'PQ2:   ',PQ2(4),' ',PQ2(1),' ',
c-     & PQ2(2),' ',PQ2(3),' ',PQ2(5) 
c-      WRITE(*,*)'PSUM1: ',PGSUM1(4),' ',PGSUM1(1),' ',
c-     & PGSUM1(2),' ',PGSUM1(3),' ',PGSUM1(5) 
c-      WRITE(*,*)'PSUM2: ',PGSUM2(4),' ',PGSUM2(1),' ',
c-     & PGSUM2(2),' ',PGSUM2(3),' ',PGSUM2(5)     
c-      WRITE(*,*)'--------------------------------------------'
c-      WRITE(*,*)'SUM:    ',PQ1(4)+PQ2(4)+PGSUM1(4)+PGSUM2(4),' ',
c-     & PQ1(1)+PQ2(1)+PGSUM1(1)+PGSUM2(1),' ',
c-     & PQ1(2)+PQ2(2)+PGSUM1(2)+PGSUM2(2),' ',
c-     & PQ1(3)+PQ2(3)+PGSUM1(3)+PGSUM2(3),' ',
c-     & DSQRT((PQ1(4)+PQ2(4)+PGSUM1(4)+PGSUM2(4))**2-
c-     & (PQ1(3)+PQ2(3)+PGSUM1(3)+PGSUM2(3))**2-
c-     & (PQ1(2)+PQ2(2)+PGSUM1(2)+PGSUM2(2))**2-
c-     & (PQ1(1)+PQ2(1)+PGSUM1(1)+PGSUM2(1))**2)      
C-...Q - quark and Qbar quark in the lab. frame
      DO I=1,5
       PP(6,I)=PQ1(I)
       PP(7,I)=PQ2(I)
      ENDDO 
C-...Q Qbar system and its decay products definition.
C-...B Bbar by default (change to other numbers,
C-... insert option to ffr-file!!!)
       KK(5,1)=11
       KK(5,2)=90 
      DO I=1,4
       PP(5,I) = PP(6,I) + PP(7,I) 
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)
      DO I=1,5
       PQ1(I) = PP(6,I)  
       PQ2(I) = PP(7,I)  
      ENDDO
       CALL PY2ENT(-6,5,-5,PP(5,5)) 
      DO I=1,5
       PP(6,I) = PQ1(I)
       PP(7,I) = PQ2(I) 
      ENDDO
       NGS=0 ! number of singlet systems
       NGSUM=NG1+NG2 !<80
       NN=7+NGS+NGSUM
c-- simple case realization --------------
c-      KK(8,1)=11
c-      KK(8,2)=90
c-      DO I=1,4
c-       PP(8,I) = PGSUM1(I)+PGSUM2(I)
c-      ENDDO
c-       PP(8,5) = DSQRT(PP(8,4)**2-PP(8,1)**2-PP(8,2)**2-PP(8,3)**2)
c--- NG1+NG2 gluons
c-  KK(I,1)=2 or 1,KK(I,2)=21,KK(I,3)=0,KK(I,4)=0,KK(I,5)=0      
c- gluons permutations
       CALL COMBSING(NGSUM,NA2,NA3,CEVENT)
      DO I=8,NGSUM+7
       KK(I,1)=CEVENT(2,I-7)
       KK(I,2)=21 
       KK(I,3)=0
       KK(I,4)=0
       KK(I,5)=0
      ENDDO       
c-- all the momenta to PYTHIA              
      IF (NG1.GT.0) THEN
      DO I=8,NG1+7
       DO J=1,5 
        PP(I,J)=PNG1(J,I-7)
       ENDDO
      ENDDO
      ENDIF
      IF (NG2.GT.0) THEN      
      DO I=NG1+8,NGSUM+7
       DO J=1,5 
        PP(I,J)=PNG2(J,I-7-NG1)
       ENDDO
      ENDDO
      ENDIF
c- permutations of momenta according to CEVENT(1,I)
      DO I=1,NGSUM
       DO J=1,5
        PVS2(I,J)=PP(7+I,J)
       ENDDO       
      ENDDO
      DO I=1,NGSUM
       DO J=1,5
        PP(7+I,J)=PVS2(CEVENT(1,I),J)
       ENDDO       
      ENDDO 
c--- QQbar to PYLIST	
      CALL PYSHOW(6,7,PP(5,5))
c--- all the gluons to PYLIST
c-      CALL PYSHOW(9,-NGSUM,PP(8,5))   
      MINT(1)=447

      RETURN
      END  
      
c----------------------------------------------------------------------              
c- *... semi-inclusive g g generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE SIDDEGG
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c- *...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- cascads parameters ----------------
      INTEGER NG1,NG2,NGSUM,NGS,CEVENT(2,500),NA2,NA3
      DOUBLE PRECISION PNG1(5,500),PNG2(5,500),PGC1(5),PGC2(5),
     & PGSUM1(5),PGSUM2(5),PVS2(500,5)         
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTG2,MXX,MJJ,PG1(5),PG2(5),
     & PMX(5),PMJJ(5),PVS(5),PAUX(5),AUXM,PCS(5)
      INTEGER NFAIL
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
        PRINT*,'EDDEGG:Attention: generator works in the ETcut'
        PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
       ENDIF
      ENDIF   

 1    CONTINUE
      CALL GENERSIGG(MJJ,ETAJ)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c--- 2 cascads generation
      CALL SICASCAD2(MJJ,NG1,PNG1,PGC1,NG2,PNG2,PGC2,MXX,NFAIL)
      IF (NFAIL.EQ.1) GOTO 1
c---      
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0)
c---
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)      
      PKAP = 1.D0/DCOSH(ETAJ)**2
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      AUXM=MXX 
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
       MXX=PMX(5)
c----------------------------------------------------
c- mass correction for all momenta -------
c- and sum of final momenta -
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO
      IF (NG1.GT.0) THEN 
       DO J=1,NG1       
        DO I=1,4
         PAUX(I)=PNG1(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(1,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG1(I,J)=PVS(I)
         PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
        ENDDO
        PNG1(4,J)=DSQRT(PNG1(1,J)**2+PNG1(2,J)**2+PNG1(3,J)**2)
        PGSUM1(4)=PGSUM1(4)+PNG1(4,J)
       ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
       DO J=1,NG2       
        DO I=1,4
         PAUX(I)=PNG2(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(2,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG2(I,J)=PVS(I)
         PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
        ENDDO
        PNG2(4,J)=DSQRT(PNG2(1,J)**2+PNG2(2,J)**2+PNG2(3,J)**2)
        PGSUM2(4)=PGSUM2(4)+PNG2(4,J)
       ENDDO
      ENDIF
c- momentum of central system -
      PGC1(4)=0.5D0*MXX-PGSUM1(4)
      PGC1(3)=0.5D0*MXX-PGSUM1(3)
      PGC2(4)=0.5D0*MXX-PGSUM2(4)
      PGC2(3)=-0.5D0*MXX-PGSUM2(3)      
      DO I=1,4
         PCS(I)=PGC1(I)+PGC2(I)
      ENDDO
       PCS(5)=DSQRT(PCS(4)**2-PCS(1)**2-PCS(2)**2-PCS(3)**2)
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MJJ*MJJ/(PCS(5)*PCS(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF 
       MJJ=PCS(5)
C--- momentum of JJ-system in the gg C.M. -----------------
C--- and new value of MJJ ---------------------------------
      DO I=1,5   
       PMJJ(I)=PCS(I)
      ENDDO
c-- 2 cascads momenta transformation ----------------
c-- from CMX to lab. --------------------------------
c- and sum of final momenta -
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO
      
      IF (NG1.GT.0) THEN 
      DO J=1,NG1       
       DO I=1,4
       PAUX(I)=PNG1(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG1(I,J)=PVS(I)
       PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
       ENDDO 
      ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
      DO J=1,NG2       
       DO I=1,4
       PAUX(I)=PNG2(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG2(I,J)=PVS(I)
       PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
       ENDDO 
      ENDDO
      ENDIF 
C-...
c-------
       MTG2 = MJJ*MJJ*PKAP/4.D0
C-... gg system in the X-rest frame
C-... parton1 - momentum
      PG1(5) = PMAS(21,1)
      PT=DSQRT(MTG2-PG1(5)**2)
      FI  = 2.*PI*PYR(0)
      PG1(1)=PT*DCOS(FI)
      PG1(2)=PT*DSIN(FI)
C-...PG1(3)=MTG*DSINH(ETAJ) 
      PG1(3)=SIGNETA*MJJ*DSQRT(1.D0-PKAP)/2.D0
      PG1(4)=DSQRT(PG1(1)**2+PG1(2)**2+PG1(3)**2+PG1(5)**2) 
C-... parton2 - momentum
      PG2(5) = PMAS(21,1) 
      DO I=1,3
       PG2(I)=-PG1(I)
      ENDDO
      PG2(4)=DSQRT(PG2(1)**2+PG2(2)**2+PG2(3)**2+PG2(5)**2)
C-... gg system in the lab. frame
C-... lorentz transformation
      CALL XCMTOLAB(PMJJ,PG1,PAUX)
      CALL XCMTOLAB(PMX,PAUX,PVS)      
      DO I=1,5
       PG1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMJJ,PG2,PAUX)
      CALL XCMTOLAB(PMX,PAUX,PVS)      
      DO I=1,5
       PG2(I)=PVS(I)
      ENDDO    
C-...gg in the lab. frame
      DO I=1,5
       PP(6,I)=PG1(I)
       PP(7,I)=PG2(I)
      ENDDO 
C-...gg-system and its decay products definition 
       KK(5,1)=11
       KK(5,2)=90 
      DO I=1,4
       PP(5,I) = PG1(I) + PG2(I) 
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)
       CALL PY2ENT(-6,21,21,PP(5,5)) 
      DO I=1,5
       PP(6,I) = PG1(I)
       PP(7,I) = PG2(I)
      ENDDO  
       NGS=0 ! number of singlet systems
       NGSUM=NG1+NG2 !<80
       NN=7+NGS+NGSUM
c-- simple case realization --------------
c-      KK(8,1)=11
c-      KK(8,2)=90
c-      DO I=1,4
c-       PP(8,I) = PGSUM1(I)+PGSUM2(I)
c-      ENDDO
c-       PP(8,5) = DSQRT(PP(8,4)**2-PP(8,1)**2-PP(8,2)**2-PP(8,3)**2)
c--- NG1+NG2 gluons
c-  KK(I,1)=2 or 1,KK(I,2)=21,KK(I,3)=0,KK(I,4)=0,KK(I,5)=0      
c- gluons permutations
       CALL COMBSING(NGSUM,NA2,NA3,CEVENT)
      DO I=8,NGSUM+7
       KK(I,1)=CEVENT(2,I-7)
       KK(I,2)=21 
       KK(I,3)=0
       KK(I,4)=0
       KK(I,5)=0
      ENDDO       
c-- all the momenta to PYTHIA              
      IF (NG1.GT.0) THEN
      DO I=8,NG1+7
       DO J=1,5 
        PP(I,J)=PNG1(J,I-7)
       ENDDO
      ENDDO
      ENDIF
      IF (NG2.GT.0) THEN      
      DO I=NG1+8,NGSUM+7
       DO J=1,5 
        PP(I,J)=PNG2(J,I-7-NG1)
       ENDDO
      ENDDO
      ENDIF
c- permutations of momenta according to CEVENT(1,I)
      DO I=1,NGSUM
       DO J=1,5
        PVS2(I,J)=PP(7+I,J)
       ENDDO       
      ENDDO
      DO I=1,NGSUM
       DO J=1,5
        PP(7+I,J)=PVS2(CEVENT(1,I),J)
       ENDDO       
      ENDDO
c--- {gg}_singlet to PYLIST	
      CALL PYSHOW(6,7,PP(5,5))
c--- all the gluons to PYLIST
c-      CALL PYSHOW(9,-NGSUM,PP(8,5))
      MINT(1)=448
        
      RETURN
      END

c----------------------------------------------------------------------              
c-*... semi-inclusive 2gamma generation /A.Sobol,R.Ryutin
c----------------------------------------------------------------------              
      SUBROUTINE SIDDE2GAM
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT1/ MINT(400),VINT(400)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some of global EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c----- cascads parameters ----------------
      INTEGER NG1,NG2,NGSUM,NGS,CEVENT(2,500),NA2,NA3
      DOUBLE PRECISION PNG1(5,500),PNG2(5,500),PGC1(5),PGC2(5),
     & PGSUM1(5),PGSUM2(5),PVS2(500,5) 
c----- for this subroutine -----------------------------------      
      DOUBLE PRECISION MTG2,MXX,MJJ,PG1(5),PG2(5),
     & PMX(5),PMJJ(5),PVS(5),PAUX(5),AUXM,PCS(5)
      INTEGER NFAIL
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       IF((2*ETJCUT).LT.14.D0.OR.(2*ETJCUT).GT.300.D0) THEN
        PRINT*,'EDDE2GAM:Attention: generator works in the ETcut'
        PRINT*,'range 7-150 GeV; you use the cut = ',ETJCUT
       ENDIF
      ENDIF   

 1    CONTINUE
      CALL GENERSI2GAM(MJJ,ETAJ)
c--- sign of etaj
      IF (ETAJ.EQ.0.D0) THEN
       SIGNETA=0.D0
      ELSE
       SIGNETA=ETAJ/DABS(ETAJ)
      ENDIF 
c--- 2 cascads generation
      CALL SICASCAD2(MJJ,NG1,PNG1,PGC1,NG2,PNG2,PGC2,MXX,NFAIL)
      IF (NFAIL.EQ.1) GOTO 1
c--- 
      CALL EDDETTPHI(NFI,MXX,T1,T2,FI0) 
      DT12= DABS(T1)+DABS(T2)
      DT12= DT12+2.D0*DSQRT(DABS(T1)*DABS(T2))*DCOS(FI0)      
      PKAP = 1.D0/DCOSH(ETAJ)**2
      X1  = EDDEX(MXX)
      XF1 = 1.D0-X1
      X2  = (MXX*MXX+DT12)/(S*X1)
      XF2 = 1.D0-X2   
       IF(XF2.LE.0.OR.XF2.GT.1) GOTO 1
       IF(PYR(0).LE.0.5) THEN
        XSAVE=XF1
        XF1=XF2
        XF2=XSAVE
       ENDIF
      X1=1.D0-XF1
      X2=1.D0-XF2
C-...scattered proton 1
      PZ  = PP(1,3)*XF1
      PT  = DSQRT(DABS(T1))
      FI  = 2.*PI*PYR(0)
      PX  = PT*DCOS(FI)
      PY  = PT*DSIN(FI)
      PP(3,1) = PX
      PP(3,2) = PY
      PP(3,3) = PZ
      PP(3,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(3,5) = AMP 
      KK(3,1) = 1
      KK(3,2) = 2212
      KK(3,3) = 1
C-...scattered proton 2
      PZ  = PP(2,3)*XF2
      PT  = DSQRT(DABS(T2))
      PX  = PT*DCOS(FI+FI0)
      PY  = PT*DSIN(FI+FI0)
      PP(4,1) = PX
      PP(4,2) = PY
      PP(4,3) = PZ
      PP(4,4) = DSQRT(PX**2+PY**2+PZ**2+AMP**2)
      PP(4,5) = AMP 
      KK(4,1) = 1
      KK(4,2) = 2212
      KK(4,3) = 2 
C-...
C--- momentum of X-system in the lab ---------------------
C--- and new value of MX ---------------------------------
      AUXM=MXX
      DO I=1,4   
       PMX(I)=PP(1,I)+PP(2,I)-PP(3,I)-PP(4,I)
      ENDDO
       PMX(5)=DSQRT(PMX(4)**2-PMX(1)**2-PMX(2)**2-PMX(3)**2)        
       MXX=PMX(5)
c----------------------------------------------------       
c- mass correction for all momenta -------
c- and sum of final momenta -
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO
      IF (NG1.GT.0) THEN 
       DO J=1,NG1       
        DO I=1,4
         PAUX(I)=PNG1(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(1,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG1(I,J)=PVS(I)
         PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
        ENDDO
        PNG1(4,J)=DSQRT(PNG1(1,J)**2+PNG1(2,J)**2+PNG1(3,J)**2)
        PGSUM1(4)=PGSUM1(4)+PNG1(4,J)
       ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
       DO J=1,NG2       
        DO I=1,4
         PAUX(I)=PNG2(I,J)
        ENDDO
        PAUX(5)=PMAS(21,1)
        CALL SIMXCOR(2,AUXM,MXX,PAUX,PVS)
        DO I=1,3
         PNG2(I,J)=PVS(I)
         PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
        ENDDO
        PNG2(4,J)=DSQRT(PNG2(1,J)**2+PNG2(2,J)**2+PNG2(3,J)**2)
        PGSUM2(4)=PGSUM2(4)+PNG2(4,J)
       ENDDO
      ENDIF
c- momentum of central system -
      PGC1(4)=0.5D0*MXX-PGSUM1(4)
      PGC1(3)=0.5D0*MXX-PGSUM1(3)
      PGC2(4)=0.5D0*MXX-PGSUM2(4)
      PGC2(3)=-0.5D0*MXX-PGSUM2(3)      
      DO I=1,4
         PCS(I)=PGC1(I)+PGC2(I)
      ENDDO
       PCS(5)=DSQRT(PCS(4)**2-PCS(1)**2-PCS(2)**2-PCS(3)**2)
c-was a BUG! new PKAP sometimes was >1,changed to if
       AUX1=PKAP
       PKAP=PKAP*MJJ*MJJ/(PCS(5)*PCS(5))
       IF (PKAP.GT.1.D0) THEN
        PKAP=AUX1
       ENDIF 
       MJJ=PCS(5)
C--- momentum of JJ-system in the gg C.M. -----------------
C--- and new value of MJJ ---------------------------------
      DO I=1,5   
       PMJJ(I)=PCS(I)
      ENDDO

c-- 2 cascads momenta transformation ----------------
c-- from CMX to lab. --------------------------------
c- and sum of final momenta -
      DO I=1,5 
       PGSUM1(I)=0.D0
       PGSUM2(I)=0.D0
      ENDDO
      
      IF (NG1.GT.0) THEN 
      DO J=1,NG1       
       DO I=1,4
       PAUX(I)=PNG1(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG1(I,J)=PVS(I)
       PGSUM1(I)=PGSUM1(I)+PNG1(I,J)
       ENDDO 
      ENDDO 
      ENDIF 
c----------------------------------------------------
      IF (NG2.GT.0) THEN
      DO J=1,NG2       
       DO I=1,4
       PAUX(I)=PNG2(I,J)
       ENDDO
       PAUX(5)=PMAS(21,1)
       CALL XCMTOLAB(PMX,PAUX,PVS)
       DO I=1,5
       PNG2(I,J)=PVS(I)
       PGSUM2(I)=PGSUM2(I)+PNG2(I,J)
       ENDDO 
      ENDDO
      ENDIF  
C-...
       MTG2 = MJJ*MJJ*PKAP/4.D0
C-... gamma gamma system in the X-rest frame
C-... parton1 - momentum
      PG1(5) = PMAS(22,1)
      PT=DSQRT(MTG2-PG1(5)**2)
      FI  = 2.*PI*PYR(0)
      PG1(1)=PT*DCOS(FI)
      PG1(2)=PT*DSIN(FI)
      PG1(3)=SIGNETA*MJJ*DSQRT(1.D0-PKAP)/2.D0
      PG1(4)=DSQRT(PG1(1)**2+PG1(2)**2+PG1(3)**2+PG1(5)**2) 
C-... parton2 - momentum
      PG2(5) = PMAS(22,1) 
      DO I=1,3
       PG2(I)=-PG1(I)
      ENDDO
      PG2(4)=DSQRT(PG2(1)**2+PG2(2)**2+PG2(3)**2+PG2(5)**2)
C-... gamma gamma system in the lab. frame
C-... lorentz transformation
      CALL XCMTOLAB(PMJJ,PG1,PAUX)
      CALL XCMTOLAB(PMX,PAUX,PVS)
      DO I=1,5
       PG1(I)=PVS(I)
      ENDDO      
      CALL XCMTOLAB(PMJJ,PG2,PAUX)
      CALL XCMTOLAB(PMX,PAUX,PVS)
      DO I=1,5
       PG2(I)=PVS(I)
      ENDDO    
C-...gamma gamma in the lab. frame
      DO I=1,5
       PP(6,I) = PG1(I)
       PP(7,I) = PG2(I)
      ENDDO  
       KK(6,1)=1
       KK(6,2)=22 
       KK(6,3)=0        
       KK(7,1)=1
       KK(7,2)=22 
       KK(7,3)=0
      DO I=1,5
       PP(6,I) = PG1(I) 
       PP(7,I) = PG2(I) 
      ENDDO
C-...gamma gamma-system definition 
       KK(5,1)=11
       KK(5,2)=90 
      DO I=1,4
       PP(5,I) = PG1(I) + PG2(I) 
      ENDDO
       PP(5,5) = DSQRT(PP(5,4)**2-PP(5,1)**2-PP(5,2)**2-PP(5,3)**2)
       CALL PY2ENT(6,22,22,PP(5,5)) 
      DO I=1,5
       PP(6,I) = PG1(I)
       PP(7,I) = PG2(I)
      ENDDO  

       NGS=0 ! number of singlet systems
       NGSUM=NG1+NG2 !<80
       NN=7+NGS+NGSUM
c-- simple case realization --------------
c-      KK(8,1)=11
c-      KK(8,2)=90
c-      DO I=1,4
c-       PP(8,I) = PGSUM1(I)+PGSUM2(I)
c-      ENDDO
c-       PP(8,5) = DSQRT(PP(8,4)**2-PP(8,1)**2-PP(8,2)**2-PP(8,3)**2)
c--- NG1+NG2 gluons
c-  KK(I,1)=2 or 1,KK(I,2)=21,KK(I,3)=0,KK(I,4)=0,KK(I,5)=0      
c- gluons permutations
       CALL COMBSING(NGSUM,NA2,NA3,CEVENT)
      DO I=8,NGSUM+7
       KK(I,1)=CEVENT(2,I-7)
       KK(I,2)=21 
       KK(I,3)=0
       KK(I,4)=0
       KK(I,5)=0
      ENDDO       
c-- all the momenta to PYTHIA              
      IF (NG1.GT.0) THEN
      DO I=8,NG1+7
       DO J=1,5 
        PP(I,J)=PNG1(J,I-7)
       ENDDO
      ENDDO
      ENDIF
      IF (NG2.GT.0) THEN      
      DO I=NG1+8,NGSUM+7
       DO J=1,5 
        PP(I,J)=PNG2(J,I-7-NG1)
       ENDDO
      ENDDO
      ENDIF
c- permutations of momenta according to CEVENT(1,I)
      DO I=1,NGSUM
       DO J=1,5
        PVS2(I,J)=PP(7+I,J)
       ENDDO       
      ENDDO
      DO I=1,NGSUM
       DO J=1,5
        PP(7+I,J)=PVS2(CEVENT(1,I),J)
       ENDDO       
      ENDDO
c--- 2gamma to PYLIST	
      CALL PYSHOW(6,7,PP(5,5))
c--- all the gluons to PYLIST
c-      CALL PYSHOW(9,-NGSUM,PP(8,5))
      MINT(1)=449
        
      RETURN
      END      

C-++++++++++++++++++++++++++++++++++++++++++
C- EDDE2.1: basic subroutines and functions
C-++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE EDDETITLE
      
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      IMPLICIT INTEGER (I-N)
       
      WRITE(*,*)'*************************************
     & *****************************************'
      WRITE(*,*)'                                    '
      WRITE(*,*)' OOOOO    OOOO     OOOO     OOOOO   '
      WRITE(*,*)' O        O   O    O   O    O       '
      WRITE(*,*)' OOOOO    O   O    O   O    OOOOO   '
      WRITE(*,*)' O        O   O    O   O    O       '
      WRITE(*,*)' OOOOO    OOOO     OOOO     OOOOO   '
      WRITE(*,*)'    X        O        I       V     '
      WRITE(*,*)'                     F       E      '
      WRITE(*,*)'      C      U      F       N       '
      WRITE(*,*)'                   R       T        '
      WRITE(*,*)'        L    B    A       S         '
      WRITE(*,*)'         U       C                  '
      WRITE(*,*)'          S  L  T                   '
      WRITE(*,*)'           I   I                    '
      WRITE(*,*)'            VEV                     '
      WRITE(*,*)'                                    '
      WRITE(*,*)'     Version 2.1.2 (2008)           '
      WRITE(*,*)'                                    '
      WRITE(*,*)'  By Diffractive groupe:            '
      WRITE(*,*)'  ------------------                '      
      WRITE(*,*)'  V.Petrov,R.Ryutin,A.Sobol         '
      WRITE(*,*)' (IHEP,Protvino)                    '
      WRITE(*,*)'  J.-P.Guillaud                     '
      WRITE(*,*)' (LAPP,Annecy)                      '
      WRITE(*,*)'                                    '      
      WRITE(*,*)'                                    '
      WRITE(*,*)'                                    '
      WRITE(*,*)'                                    '
      WRITE(*,*)'*************************************
     & *****************************************'      

      RETURN
      END
C
C--- 1-dim integration ----------------
       FUNCTION DISIMP(TOTFUN,P1,P2,ERR)
       
       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       IMPLICIT INTEGER (I-N)
       
       M=1000
 1     N=2*M
       H=(P2-P1)/N
       X1=P1+H
       XD1=P1+0.5D0*H
       SN=TOTFUN(P1)+TOTFUN(P2)+2.D0*TOTFUN(X1)
       S2N=TOTFUN(P1)+TOTFUN(P2)+4.D0*TOTFUN(XD1)
       VY=0.D0       
       DO I=1,M-1
       VAR1=P1+I*H
       VAR2=P1+(2*I+1)*H
       VAR3=P1+(2*I+1)*0.5D0*H
       VY=VY+2.D0*TOTFUN(VAR1)
       SN=SN+2.D0*TOTFUN(VAR2)
       S2N=S2N+4.D0*TOTFUN(VAR3)        
       ENDDO
       DO I=M,N-1 
       VAR1=P1+I*H
       VAR3=P1+(2*I+1)*0.5D0*H
       VY=VY+2.D0*TOTFUN(VAR1)
       S2N=S2N+4.D0*TOTFUN(VAR3)         
       ENDDO
       SN=SN+VY
       SN=SN*H/3.D0
       S2N=S2N+VY
       S2N=S2N*H/6.D0
       IF (DABS(S2N).GT.0.D0) THEN
       RN=DABS(S2N-SN)/(15.D0*DABS(S2N))
       ELSE
       RN=1.D0
       ENDIF
       M=2*M
       IF (RN.GT.ERR) GOTO 1             
       DISIMP=S2N
       
       RETURN
       END
       
c-------------------------------------------------------------              
c- ... model functions !R.Ryutin
c-------------------------------------------------------------
c--------ADDITIONAL FUNCTIONS and SUBROUTINES-----------------
C-============================================================
C-==== linear interpolation of any f(x,y) from data table ====
C-============================================================
C--- check+
       SUBROUTINE LINTERPOL2(FUNDAT,N1,N2,X0,Y0,DX,DY,XV,YV,FUN)
       
       IMPLICIT NONE
       INTEGER I,J
       INTEGER N1,N2
       DOUBLE PRECISION FUNDAT(N1,N2),X0,Y0,DX,DY
       DOUBLE PRECISION XV,YV,FUN
       DOUBLE PRECISION X(2),Y(2),C(2),D(2),F(2,2)
       
       I=DINT((XV-X0)/DX)+1
       J=DINT((YV-Y0)/DY)+1
       
       F(1,1)=FUNDAT(I,J)
       F(2,1)=FUNDAT(I+1,J)
       F(1,2)=FUNDAT(I,J+1)
       F(2,2)=FUNDAT(I+1,J+1)
       IF (I.GE.N1) THEN
           F(1,2)=F(1,1)
           F(2,2)=F(2,1)
       ENDIF
       IF (J.GE.N2) THEN
           F(2,1)=F(1,1)
           F(2,2)=F(1,2)
       ENDIF
       IF (I.GE.N1.AND.J.GE.N2) THEN
            F(1,2)=F(1,1)
            F(2,1)=F(1,1)
            F(2,2)=F(1,1)
       ENDIF
       X(1)=X0+(I-1)*DX
       X(2)=X0+I*DX
       Y(1)=Y0+(J-1)*DY
       Y(2)=Y0+J*DY
       
       C(1)=(F(1,1)+F(2,2)-F(1, 2)-F(2, 1))/(DX*DY)
       C(2)=(F(2,1)*X(1)-F(2,2)*X(1)-F(1,1)*X(2)+F(1,2)*X(2))/(DX*DY)
       D(1)=(F(1,2)*Y(1)-F(2,2)*Y(1)-F(1,1)*Y(2)+F(2,1)*Y(2))/(DX*DY)
       D(2)=(F(2,2)*X(1)*Y(1)+F(1,1)*X(2)*Y(2))/(DX*DY)
       D(2)=D(2)-(F(1,2)*X(2)*Y(1)+F(2,1)*X(1)*Y(2))/(DX*DY)
             
       FUN=XV*(C(1)*YV+D(1))+C(2)*YV+D(2)
       IF (I.GT.N1.OR.J.GT.N2) THEN
c-          WRITE(*,*)'OUT OF THE INTERPOLATION RANGE'
          FUN=0.D0
       ENDIF       
                    
       RETURN
       END  


C-============================================================
C-=========== factorial integer ==============================
C-============================================================
       FUNCTION FACTN(NI)

       IMPLICIT NONE
       INTEGER FACTN 
       INTEGER FCT,NI,NI0,J
       
       FCT=1
       NI0=NI
       IF (NI.LT.1) THEN
        NI0=1
       ENDIF
       DO J=1,NI0
        FCT=FCT*J
       ENDDO
       
       FACTN=FCT
       
       RETURN
       END
C-============================================================

C-============================================================
C-=========== factorial ======================================
C-============================================================
       FUNCTION FACT(NI)

       IMPLICIT NONE
       DOUBLE PRECISION FACT 
       INTEGER FCT,NI,NI0,J
       
       FCT=1
       NI0=NI
       IF (NI.LT.1) THEN
        NI0=1
       ENDIF
       DO J=1,NI0
        FCT=FCT*J
       ENDDO
       
       FACT=DBLE(FCT)
       
       RETURN
       END
C-============================================================

c--------"SOFT" FUNCTIONS and SUBROUTINES---------------------
C-============================================================
C-= bare t-hard slope function (for total cs. calc.) =========
C-= AMP_gp-gp~EXP(-TSLOPEBH(sqrt(s)/M)*|T|) ==================
C-============================================================
C--- check+
      FUNCTION TSLOPEBH(VAR)
      IMPLICIT NONE
      DOUBLE PRECISION TSLOPEBH,VAR,RAUX
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
     
       RAUX=0.25D0*RG(3)+0.125D0*RP(3)
       TSLOPEBH=AP(3)*DLOG(VAR)+RAUX

      RETURN
      END
C-============================================================
C-============================================================
C-= t-slope argument correction ============================== 
C-= fqt2Isud=0.756*0.6<qt**2>/(1+0.6<qt**2>) =================
C-= from qt-loop integral with general semi-incl. sudakov ====
C-============================================================
C--- check+
       FUNCTION FQT2ISUD(MU,DETA)

       IMPLICIT NONE
       DOUBLE PRECISION  FQT2ISUD,MU,DETA,QT2SR

       QT2SR=1.38D0*MU**0.93D0*(2.9D0/MU)**(0.055D0*DETA)
       FQT2ISUD=0.756D0*0.6D0*QT2SR/(1.D0+0.3D0*QT2SR)
       
       RETURN
       END
C-============================================================
C-!!!!!!!!!!!!!!!!!!!!!!!!

C-============================================================
C- t1,t2,phi_(1,2) generator for the EDDE and SI DDE =========
C-============================================================
C--- check+
      SUBROUTINE EDDETTPHI(NX,MX,GT1,GT2,GFI0)
      
      IMPLICIT NONE
      
      DOUBLE PRECISION TSLOPEBH,FQT2ISUD,FACT
c-... Mass of the central system and random value
      DOUBLE PRECISION MX,MU,PYR
c-... parameters of 3 Pomeron trajectories and "hard" slope (2*B_hard(J/Psi)) 
      DOUBLE COMPLEX OP(3),BP(3)
      DOUBLE PRECISION REBP(3),IMBP(3)
c-... variables to generate      
      DOUBLE PRECISION GT1,GT2,GFI0
c-... aux. parameters
      DOUBLE PRECISION AUX0,BETA
      DOUBLE PRECISION AUX3,AUX4,AUX5,AUX6
      DOUBLE PRECISION AUX7,AUX8,AUX9,AUX10,AUX11
c-... "hard" slope, function for the "soft" survival factor
      DOUBLE PRECISION REBH,REBH2,IMBH,IMBH2
      DOUBLE COMPLEX BH,BH2
      DOUBLE COMPLEX SFU1,SFU2,SFU3,SFU4
      DOUBLE COMPLEX SFU10,SFU20,SFU30,SFU40
c      DOUBLE COMPLEX BIND,FLL2,FLL2M
      DOUBLE COMPLEX BIND
      DOUBLE PRECISION REBIND,IMBIND
c      DOUBLE COMPLEX FU1,FU2,FU3,FU4
      DOUBLE COMPLEX FU1,FU2,FU3
      DOUBLE PRECISION FIM,FIMM,SURV2,SURV2M
      DOUBLE PRECISION VPB,VBVSP,VBVSP2
c-... gp->gp amplitude unitar.corr. generation parameters .....
      DOUBLE COMPLEX AUX1,AUX2,SSFU1,SSFU2,SSFU0
      DOUBLE PRECISION AAGP1,AAGP2,AAGP0
      DOUBLE PRECISION AAGP10,AAGP20,AAGPX,AAGPXM
c-... first generation      
      DOUBLE PRECISION RT1,RT2,RFI0,RPT,RATFG
c-... NAPR=7 optimal value for the eikonal approximation.
      INTEGER N1,N2,N3,NX
c-... FLAGS=1 0++ Higgs,Chi; FLAGS=2 ... other states.      
      INTEGER FLAGS 
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2     
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI    
c-    
c-      LOGICAL START
c-      DATA START / .TRUE. /
c-      SAVE START, BP, OP
c-      
c----------------------------------------------------------------------------
c-      Switch on (off) for different phi0 distributions                     
c-      FLAGS=1(default) -> general heavy 0++ (Higgs, H*, R*,...)                           *
c-      FLAGS=2 -> (Jz=0 and +-2)
c-      FLAGS=3 -> heavy conserved 0-+
c-      FLAGS=4 -> conserved "glueball"
c-      FLAGS=5 -> (Jz=+-2) ^
c----------------------------------------------------------------------------
c-----------------------------------
      FLAGS=NX 
c-      IF (START) THEN
c-... Pomerons
      REBP(1)=2.D0*AP(1)*DLOG(SQS)+RP(1)/4.D0
      IMBP(1)=-AP(1)*PI/2.D0
      REBP(2)=2.D0*AP(2)*DLOG(SQS)+RP(2)/4.D0
      IMBP(2)=-AP(2)*PI/2.D0
      REBP(3)=2.D0*AP(3)*DLOG(SQS)+RP(3)/4.D0
      IMBP(3)=-AP(3)*PI/2.D0      
      BP(1)=REI*REBP(1)+MNI*IMBP(1)
      BP(2)=REI*REBP(2)+MNI*IMBP(2)
      BP(3)=REI*REBP(3)+MNI*IMBP(3)
      OP(1)=CP(1)*SQS**(2.D0*DP(1))/BP(1)      
      OP(1)=OP(1)*(REI*DCOS(0.5D0*PI*DP(1))-MNI*DSIN(0.5D0*PI*DP(1)))
      OP(2)=CP(2)*SQS**(2.D0*DP(2))/BP(2)
      OP(2)=OP(2)*(REI*DCOS(0.5D0*PI*DP(2))-MNI*DSIN(0.5D0*PI*DP(2))) 
      OP(3)=CP(3)*SQS**(2.D0*DP(3))/BP(3)
      OP(3)=OP(3)*(REI*DCOS(0.5D0*PI*DP(3))-MNI*DSIN(0.5D0*PI*DP(3)))
c-      START = .FALSE.
c-      END IF
c---     
      MU=MX*0.5D0
      AUX0=SQS*FQT2ISUD(MU,0.D0)/MX
      REBH=TSLOPEBH(AUX0)
      IMBH=-AP(3)*PI/2.D0
      BH=REI*REBH+MNI*IMBH
      BH2=BH+BH
      REBH2=REBH+REBH
      IMBH2=IMBH+IMBH
      BETA=0.4D0*DLOG(SQS/MX**1.7D0)
c---
 1     CALL GENERTTUP(MX,RT1,RT2)
        RFI0=PI*PYR(0)
        RPT=RT1+RT2-2*DSQRT(RT1*RT2)*DCOS(RFI0)
        IF (RPT.GT.2.5D0) GOTO 1
c-... T1,T2 dep. of gp->gp amplitudes (unitarized) ------------
       AUX1=REI*CGP*AUX0**DP(3)
       AUX1=AUX1/(8.D0*PI*BH)
       AUX1=AUX1*(REI*DCOS(0.5D0*PI*DP(3))-MNI*DSIN(0.5D0*PI*DP(3)))
       SSFU1=REI
       SSFU2=REI
       SSFU0=REI
        DO N1=2,NAPR+2
         AUX2=(-AUX1)**(N1-1)/(N1*N1*FACT(N1-1))
         AUX3=REBH*(N1-1)*RT1/N1
         AUX4=REBH*(N1-1)*RT2/N1
         AUX5=IMBH*(N1-1)*RT1/N1
         AUX6=IMBH*(N1-1)*RT2/N1
        SSFU1=SSFU1+AUX2*DEXP(AUX3)*(REI*DCOS(AUX5)-MNI*DSIN(AUX5))
        SSFU2=SSFU2+AUX2*DEXP(AUX4)*(REI*DCOS(AUX6)-MNI*DSIN(AUX6))
        SSFU0=SSFU0+AUX2
        ENDDO
        AAGP1=SSFU1*DCONJG(SSFU1)
        AAGP2=SSFU2*DCONJG(SSFU2)
        AAGP0=SSFU0*DCONJG(SSFU0)
        AAGP10=AAGP0
        AAGP20=AAGP0
        IF (RT1.GT.3.7D0) THEN
         AAGP10=AAGP10*DEXP(0.4D0*DLOG(SQS/MX**1.7D0)*(RT1-3.7D0))
        ENDIF
        IF (RT2.GT.3.7D0) THEN
         AAGP20=AAGP20*DEXP(0.4D0*DLOG(SQS/MX**1.7D0)*(RT1-3.7D0))
        ENDIF
        AAGPX=AAGP1*AAGP2
        AAGPXM=AAGP10*AAGP20
c-... general survival distribution
         SFU3=0.D0*REI
         SFU30=0.D0*REI
        DO N1=1,NAPR
           SFU2=0.D0*REI
           SFU20=0.D0*REI
         DO N2=0,N1
           SFU1=0.D0*REI
           SFU10=0.D0*REI
          DO N3=0,N1-N2
           REBIND=N2*REBP(1)/(REBP(1)**2+IMBP(1)**2)
           REBIND=REBIND+N3*REBP(2)/(REBP(2)**2+IMBP(2)**2)
           REBIND=REBIND+(N1-N2-N3)*REBP(3)/(REBP(3)**2+IMBP(3)**2)
           IMBIND=-N2*IMBP(1)/(REBP(1)**2+IMBP(1)**2)
           IMBIND=IMBIND-N3*IMBP(2)/(REBP(2)**2+IMBP(2)**2)
           IMBIND=IMBIND-(N1-N2-N3)*IMBP(3)/(REBP(3)**2+IMBP(3)**2)           
           BIND=REI*REBIND+MNI*IMBIND
           AUX7=REBH2*REBIND-IMBH2*IMBIND+1.D0
           AUX8=IMBH2*REBIND+REBH2*IMBIND
           AUX9=RPT/(AUX7**2+AUX8**2)
           AUX10=(REBH**2-IMBH**2)*(REBIND*AUX7+IMBIND*AUX8)
           AUX10=AUX10-2.D0*IMBH*REBH*(IMBIND*AUX7-REBIND*AUX8)
           AUX10=AUX10*AUX9
           AUX11=(REBH**2-IMBH**2)*(IMBIND*AUX7-REBIND*AUX8)
           AUX11=AUX11+2.D0*IMBH*REBH*(REBIND*AUX7+IMBIND*AUX8)
           AUX11=AUX11*AUX9
           FU1=DEXP(AUX10)*(REI*DCOS(AUX11)+MNI*DSIN(AUX11))
           FU2=FACT(N3)*FACT(N1-N2-N3)*(BH2*BIND+1)
           FU3=(OP(2)/OP(3))**N3
           SFU1=SFU1+FU1*FU3/FU2                    
           SFU10=SFU10+FU3/FU2
          ENDDO
          SFU2=SFU2+SFU1*(OP(1)/OP(3))**N2/FACT(N2)
          SFU20=SFU20+SFU10*(OP(1)/OP(3))**N2/FACT(N2)
         ENDDO 
         SFU3=SFU3+SFU2*(-OP(3)/(16.D0*PI))**N1
         SFU30=SFU30+SFU20*(-OP(3)/(16.D0*PI))**N1
        ENDDO 
          SFU4=(1+SFU3)*(1+SFU3)+FKK
          SFU40=(1+SFU30)*(1+SFU30)+FKK
          SURV2=SFU4*DCONJG(SFU4)*AAGPX
          SURV2M=SFU40*DCONJG(SFU40)*AAGPXM
c-... additional phi0-distribution from mass dependence
c-... significant contribution only for masses <3 GeV
          FIM=((MX*MX+RT1+RT1+RT2+RT2-RPT)/(MX*MX))**(DP(3)+DP(3))
          SURV2=SURV2*FIM
          FIMM=(MX*MX+T1MAX+T2MAX+2.D0*DSQRT(T1MAX*T2MAX))/(MX*MX)
          FIMM=FIMM**(DP(3)+DP(3))
          SURV2M=SURV2M*FIMM
c-... additional phi0-distribution for different states 0++,...
       IF (FLAGS.EQ.2) THEN
       VPB=0.3D0*MX**0.65D0
       VBVSP=1.D0+VPB*DSQRT(RT1*RT2)*DCOS(RFI0)
       VBVSP2=1.D0+VPB*DSQRT(T1MAX*T2MAX)
       SURV2=SURV2*VBVSP*VBVSP
       SURV2M=SURV2M*VBVSP2*VBVSP2
       ENDIF
c-
       IF (FLAGS.EQ.3) THEN
       SURV2=SURV2*RT1*RT2*DSIN(RFI0)**2
       SURV2M=SURV2M*T1MAX*T2MAX
       ENDIF
c-       
       IF (FLAGS.EQ.4) THEN
       SURV2=SURV2*RT1*RT2*DCOS(RFI0)**2
       SURV2M=SURV2M*T1MAX*T2MAX
       ENDIF
c-
       IF (FLAGS.EQ.5) THEN
       VPB=0.3D0*MX**0.65D0
       VBVSP=VPB*DSQRT(RT1*RT2)*DCOS(RFI0)
       VBVSP2=VPB*DSQRT(T1MAX*T2MAX)
       SURV2=SURV2*VBVSP*VBVSP
       SURV2M=SURV2M*VBVSP2*VBVSP2
       ENDIF
c-... Monte-carlo generation of the distribution SURV2          
       RATFG=SURV2/SURV2M
       IF (RATFG.LE.PYR(0)) GOTO 1
c-
       GT1=RT1
       GT2=RT2
       GFI0=RFI0
c-                    
      RETURN
      END
c---uper t generation ----------------------------------------
      SUBROUTINE GENERTTUP(MX,RT1,RT2)

      IMPLICIT NONE
      
      DOUBLE PRECISION TSLOPEBH,FQT2ISUD,PYR
      DOUBLE PRECISION RF1,RF2,RT1,RT2,MX
      DOUBLE PRECISION AUX0,AUX1,AUX2,AUX3,AUX4
      DOUBLE PRECISION AUX5,AUX6,AUX7
      DOUBLE PRECISION AUX8,AUX9
      DOUBLE PRECISION REBH,REBH2,MU,BETA
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2     
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI  
c---

      MU=MX*0.5D0
      AUX0=SQS*FQT2ISUD(MU,0.D0)/MX
      REBH=TSLOPEBH(AUX0)
      REBH2=REBH+REBH
      BETA=0.4D0*DLOG(SQS/MX**1.7D0)
      AUX1=DEXP(-REBH2*T1MIN)/REBH2
      AUX2=DEXP(-REBH2*T2MIN)/REBH2
      AUX3=DEXP(-REBH2*3.7D0)
      AUX4=DEXP(-(REBH2-BETA)*T1MAX-BETA*3.7D0)/(REBH2-BETA)
      AUX5=DEXP(-(REBH2-BETA)*T2MAX-BETA*3.7D0)/(REBH2-BETA)
      AUX6=AUX1-AUX3/REBH2
      AUX7=AUX2-AUX3/REBH2
      AUX8=AUX6+AUX3/(REBH2-BETA)-AUX4
      AUX9=AUX7+AUX3/(REBH2-BETA)-AUX5
      RF1=PYR(0)*AUX8
      RF2=PYR(0)*AUX9
      IF (RF1.LE.AUX6) THEN
      RT1=-DLOG((AUX1-RF1)*REBH2)/REBH2
      ELSE
      RT1=DLOG((AUX6+AUX3/(REBH2-BETA)-RF1)*(REBH2-BETA))
      RT1=(RT1+3.7D0*BETA)/(BETA-REBH2)
      ENDIF
      IF (RF2.LE.AUX7) THEN
      RT2=-DLOG((AUX2-RF2)*REBH2)/REBH2      
      ELSE
      RT2=DLOG((AUX7+AUX3/(REBH2-BETA)-RF2)*(REBH2-BETA))
      RT2=(RT2+3.7D0*BETA)/(BETA-REBH2)      
      ENDIF      

      RETURN
      END
C-============================================================


C-============================================================
C- x-distribution for a particle production in EDDE  /R.Ryutin
C-============================================================
C--- check+
      FUNCTION EDDEX(MX)
c-... xmax=1,xmin=MX**2/s - kinematical limit ..............................
c-... interval xmin/0.1<x<0.1 plays the main role in the theoretical 
c-... calculations
c-... x is the usual kinematical variable, x=1-x_Feinm.
c----------------------------------------------------------------------------c 
c----------------------------------------------------------------------------c
c-      maximal xi=1-xFeinm value (default = 0.1)                 
c-      Region of the model applicability at LHC                           
c----------------------------------------------------------------------------c
      IMPLICIT NONE

c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
      DOUBLE PRECISION EDDEX
      DOUBLE PRECISION MX,BTM,PYR
c      DOUBLE PRECISION Y0,Y1,YSR,YSM,YGEN,FYGEN
      DOUBLE PRECISION Y0,Y1,YSR,YGEN,FYGEN
C-     
       BTM=MX*MX/(SQS*SQS)
       Y1=DLOG(SQS/MX)
       Y0=Y1+DLOG(0.1D0)
       IF (SQS.LE.8.D+03) THEN
         Y0=Y1+DLOG(0.2D0)
       ENDIF       
       IF (SQS.LE.3.D+03) THEN
         Y0=Y1+DLOG(0.3D0)
       ENDIF
       YSR=1.2*Y0
  1    YGEN=Y1*PYR(0)  
       FYGEN=DEXP(-(YGEN/YSR)**10.D0)
      IF (PYR(0).GE.FYGEN) GOTO 1
       EDDEX = DSQRT(BTM)*DEXP(YGEN)
      RETURN
      END
C-============================================================
      
C-============================================================            
C-Transformation from the X-rest frame to the lab. frame/R.Ryutin
C-============================================================                    
C--- check+
      SUBROUTINE XCMTOLAB(PMXP,PB3P,PVSP)
C-
      IMPLICIT NONE
C-       
      DOUBLE PRECISION PMXP(5),PB3P(5),PVSP(5),BETA(3),GAMMA,SCAL
      INTEGER I 
C-
      DO I=1,3
      BETA(I)=-PMXP(I)/PMXP(4)
      ENDDO
C-      
      GAMMA=1.D0/DSQRT(1-BETA(1)**2-BETA(2)**2-BETA(3)**2)
      SCAL=BETA(1)*PB3P(1)+BETA(2)*PB3P(2)+BETA(3)*PB3P(3)
C-
      DO I=1,3
      PVSP(I)=PB3P(I)+GAMMA*BETA(I)*(GAMMA*SCAL/(GAMMA+1.D0)-PB3P(4))
      ENDDO
C-      
      PVSP(4)=GAMMA*(PB3P(4)-SCAL)
      PVSP(5)=DSQRT(DABS(PVSP(4)**2-PVSP(1)**2-PVSP(2)**2-PVSP(3)**2))
C-
      RETURN      
      END     
C-============================================================

c--------"HARD" FUNCTIONS and SUBROUTINES --------------------
C-============================================================
C-============ QCD coupling ==================================
C-============================================================
C--- check+
       FUNCTION ALPHAS(MU)

       IMPLICIT NONE
       DOUBLE PRECISION  ALPHAS,MU,AUXLOG
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           

      AUXLOG=2.D0*DLOG(MU/LAMQCD)
      ALPHAS=(1.D0-NLOSW*BF1*DLOG(AUXLOG)/(BF0*AUXLOG))
      ALPHAS=ALPHAS/(BF0*AUXLOG)
      
       RETURN
       END
C-============================================================
C-============================================================
C-==== 1st integral of QCD coupling in t=Ln(MU^2/LAMQCD^2) ===
C-============================================================
C--- check+
       FUNCTION IALPHAS(MU1,MU2)

       IMPLICIT NONE
       DOUBLE PRECISION IALPHAS,MU1,MU2
       DOUBLE PRECISION LN1,LN2,AUX1,AUX2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           

      LN1=2.D0*DLOG(MU1/LAMQCD)
      LN2=2.D0*DLOG(MU2/LAMQCD)
      AUX1=(1+DLOG(LN1))/LN1
      AUX2=(1+DLOG(LN2))/LN2
      IALPHAS=BF1*NLOSW*(AUX2-AUX1)/(BF0*BF0)
      IALPHAS=IALPHAS+(DLOG(LN2)-DLOG(LN1))/BF0      
      
       RETURN
       END
C-============================================================
C-============================================================
C-= 1st integral of (QCD coupling)^2 in t=Ln(MU^2/LAMQCD^2) ==
C-============================================================
C--- check+
       FUNCTION IALPHAS2(MU1,MU2)

       IMPLICIT NONE
       DOUBLE PRECISION IALPHAS2,MU1,MU2
       DOUBLE PRECISION LN1,LN2,LLN1,LLN2
       DOUBLE PRECISION AUX1,AUX2,AXX1,AXX2
       DOUBLE PRECISION AYY1,AYY2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           

      LN1=2.D0*DLOG(MU1/LAMQCD)
      LN2=2.D0*DLOG(MU2/LAMQCD)
      LLN1=DLOG(LN1)
      LLN2=DLOG(LN2)
      AUX1=BF1*NLOSW*(2.D0+3.D0*LLN1)-9.D0*BF0*LN1
      AUX2=BF1*NLOSW*(2.D0+3.D0*LLN2)-9.D0*BF0*LN2
      AXX1=54.D0*BF0**2*LN1**2
      AXX1=AXX1+BF1*NLOSW*(4.D0*BF1-27.D0*BF0*LN1)
      AXX2=54.D0*BF0**2*LN2**2
      AXX2=AXX2+BF1*NLOSW*(4.D0*BF1-27.D0*BF0*LN2)
      AYY1=(AXX1+6.D0*BF1*NLOSW*LLN1*AUX1)
      AYY1=-AYY1/(54.D0*BF0**4*LN1**3)
      AYY2=(AXX2+6.D0*BF1*NLOSW*LLN2*AUX2)
      AYY2=-AYY2/(54.D0*BF0**4*LN2**3)
      IALPHAS2=AYY2-AYY1    
      
       RETURN
       END
C-============================================================
C-============================================================
C-= Integral(x**N*Pgg(x,MU)/(ALPHAS(MU)/(2*PI)) dx:A->B ======
C-============================================================
C--- check+
       FUNCTION IPGG(N,A,B)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION  IPGG,A,B
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW             
       
       IF (N.EQ.0) THEN
         IPGG=DLOG((1-A)*B/((1-B)*A))
         IPGG=IPGG-(B-A)*((B+A)**2/NC-(B+A)/2+2)
         IPGG=IPGG*2.D0*NC
       ENDIF
       
       IF (N.EQ.1) THEN
         IPGG=DLOG((1-A)/(1-B))
         IPGG=IPGG+0.5D0*(B-A)*(B**2+A**2+A*B)/NC
         IPGG=IPGG-0.5D0*(B-A)*((B+A)+(B**3+A**3+A*B*(B+A))/4.D0)
         IPGG=2.D0*NC*IPGG
       ENDIF

C--- INTEGRAL OF NF*Pqg(X) dX X:0->1 -----       
       IF (N.GT.1) THEN
         IPGG=NF/3.D0
       ENDIF  
       
       RETURN
       END
C-============================================================
c-- gg-> a b and gg-> a b c cross-sections on partons level --
c-- dsigma(Mgg,etaj*)/d(etaj*), ------------------------------ 
c-- etaj*=(eta1-eta2)/2: -etamax->+etamax, -------------------
c-- etamax=ArcCosh(Mgg/Mjcut) -------------------------------- 
c-- functions for generation etaj*,Mgg in gg c.m.f. ----------
C--- check+
C--- in exclusive PLUM->PLUM-PSIDD2, ISUDGEN2->ISUD02 --------
C-============================================================
C-========= exclusive g g -> g g Jz=0 ========================
C-============================================================

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCSGG(N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCSGG,ALPHAS
       DOUBLE PRECISION M,ETA,AUX1
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
      
      DCSGG=18.D0*PI*(ALPHAS(M/2.D0)/M)**2 
      IF (N.EQ.0) THEN
         DCSGG=DCSGG*DCOSH(ETA)*DCOSH(ETA)
      ELSE
         AUX1=(DCOSH(ETA)**3*(DCOSH(ETA)-1.D0))**(1/2.09D0-1.D0) 
         DCSGG=DCSGG*0.5D0*2.62108D0*DSINH(ETA)*DCOSH(ETA)
         DCSGG=DCSGG*DCOSH(ETA)*(4.D0*DCOSH(ETA)-3.D0)
         DCSGG=DCSGG*AUX1/2.09D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CSGG(M)

       IMPLICIT NONE
       DOUBLE PRECISION  CSGG,ALPHAS,M,U
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        CSGG=18.D0*PI*(ALPHAS(M/2.D0)/M)**2*(U+0.5D0*DSINH(U+U))
       
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
       SUBROUTINE FINVUGG(RF,MG,GETA)

       IMPLICIT NONE
       DOUBLE PRECISION GETA,RF,R,MG
       DOUBLE PRECISION AUX1,AUX2,AUX3,AUX4

        R=RF**2.09D0*MG**3*(MG-1)
        AUX1=(DSQRT(768.D0*R+81.D0)-9.D0)**(1.D0/3.D0)
        AUX2=1-16.D0*(2.D0/3.D0)**(1.D0/3.D0)*R**(2.D0/3.D0)/AUX1
        AUX2=AUX2+2.D0*AUX1*R**(1.D0/3.D0)*(2.D0/3.D0)**(2.D0/3.D0)
        AUX3=1+8.D0*(2.D0/3.D0)**(1.D0/3.D0)*R**(2.D0/3.D0)/AUX1    
        AUX3=AUX3-AUX1*R**(1.D0/3.D0)*(2.D0/3.D0)**(2.D0/3.D0) 
        AUX4=0.25D0+0.25D0*DSQRT(AUX2)
        AUX4=AUX4+0.5D0*DSQRT(AUX3+1.D0/DSQRT(AUX2))/DSQRT(2.D0)    
        GETA=DLOG(AUX4+DSQRT(AUX4**2-1))
        
       RETURN
       END  

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMUGG(N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMUGG,MG
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUD02
       DOUBLE PRECISION CSGG,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

        IF (N.EQ.0) THEN
          FMUGG=CSGG(MG*MGGCUT)
          FMUGG=FMUGG*SOFTSURV(1,MG*MGGCUT)
          FMUGG=FMUGG*ISUD02(1,0.5D0*MG*MGGCUT)
          FMUGG=FMUGG/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1/MGGCUT
          AUX2=18.D0*PI*AUX1*AUX1*2.62108D0
          POWCS=2.D0*0.15D0+2.D0
          STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
          STE2=1.D0/2.09D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMUGG=AUX2*AUX3
          FMUGG=FMUGG*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMUGG=FMUGG*ISUD02(2,0.5D0*MG*MGGCUT)*MG**PSUD
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENEREXGG(GMGG,GETAJ)
       
       IMPLICIT NONE
      
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION FMUGG,DCSGG
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION POWCS,STE1,STE2
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

      POWCS=2.D0*0.15D0+2.D0 
      STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
      STE2=1.D0/2.09D0
 51   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMUGG(0,RMG)/FMUGG(1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 51
      GMG=RMG
      GMGG=MGGCUT*GMG

 52   CALL FINVUGG(PYR(0),GMG,RETAJ)         
      RAT=DCSGG(0,GMGG,RETAJ)/DCSGG(1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 52
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 
      
       RETURN
       END
 
C-============================================================
C-============================================================
C-========= exclusive g g -> Q Qbar Jz=0 =====================
C-============================================================

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCSQQ(MQ,N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCSQQ,ALPHAS
       DOUBLE PRECISION MQ,M,ETA,AUX1
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
      
      DCSQQ=4.D0*PI*(ALPHAS(M/2.D0)/M)**2/3.D0
      DCSQQ=DCSQQ*(MQ/M)**2*(1.D0-4.D0*(MQ/M)**2)
      IF (N.EQ.0) THEN
         DCSQQ=DCSQQ*DCOSH(ETA)*DCOSH(ETA)
      ELSE
         AUX1=(DCOSH(ETA)**3*(DCOSH(ETA)-1.D0))**(1/2.09D0-1.D0) 
         DCSQQ=DCSQQ*0.5D0*2.62108D0*DSINH(ETA)*DCOSH(ETA)
         DCSQQ=DCSQQ*DCOSH(ETA)*(4.D0*DCOSH(ETA)-3.D0)
         DCSQQ=DCSQQ*AUX1/2.09D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CSQQ(MQ,M)

       IMPLICIT NONE
       DOUBLE PRECISION  CSQQ,ALPHAS,MQ,M,U
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        CSQQ=4.D0*PI*(ALPHAS(M/2.D0)/M)**2/3.D0
        CSQQ=CSQQ*(MQ/M)**2*(1.D0-4.D0*(MQ/M)**2)
        CSQQ=CSQQ*(U+0.5D0*DSINH(U+U))
       
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
c---SUBROUTINE FINVUQQ(RF,MG,GETA)=FINVUGG(RF,MG,GETA)---

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMUQQ(MQ,N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMUQQ,MG,MQ
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUD02
       DOUBLE PRECISION CSQQ,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

        IF (N.EQ.0) THEN
          FMUQQ=CSQQ(MQ,MG*MGGCUT)
          FMUQQ=FMUQQ*SOFTSURV(1,MG*MGGCUT)
          FMUQQ=FMUQQ*ISUD02(1,0.5D0*MG*MGGCUT)
          FMUQQ=FMUQQ/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1*MQ/MGGCUT**2
          AUX2=4.D0*PI*AUX1*AUX1*2.62108D0/3.D0
          POWCS=2.D0*0.15D0+2.D0+2.D0
          STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
          STE2=1.D0/2.09D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMUQQ=AUX2*AUX3
          FMUQQ=FMUQQ*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMUQQ=FMUQQ*ISUD02(2,0.5D0*MG*MGGCUT)*MG**PSUD
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENEREXQQ(MQ,GMGG,GETAJ)
       
       IMPLICIT NONE
     
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION FMUQQ,DCSQQ,MQ
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION POWCS,STE1,STE2
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2           

      POWCS=2.D0*0.15D0+2.D0+2.D0 
      STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
      STE2=1.D0/2.09D0
 61   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMUQQ(MQ,0,RMG)/FMUQQ(MQ,1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 61
      GMG=RMG
      GMGG=MGGCUT*GMG

 62   CALL FINVUGG(PYR(0),GMG,RETAJ)         
      RAT=DCSQQ(MQ,0,GMGG,RETAJ)/DCSQQ(MQ,1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 62
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 
      
       RETURN
       END
 
C-============================================================
C-============================================================
C-============================================================
C-========= exclusive g g -> gamma gamma Jz=0 ================
C-============================================================

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCS2GAM(N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCS2GAM,ALPHAS
       DOUBLE PRECISION M,ETA,ALPHAE
       DOUBLE PRECISION AUX0,AUX1,AUX2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         

      ALPHAE=1.D0/128.D0      
      DCS2GAM=(11.D0*ALPHAS(M/2.D0)*ALPHAE/(18.D0*M))**2/PI
      AUX0=1.D0-2.D0*ETA*DTANH(ETA)
      AUX0=AUX0+(0.25D0*PI**2+ETA**2)*(1.D0+DTANH(ETA)**2)
      AUX1=(1+AUX0*AUX0)/DCOSH(ETA)**2
      IF (N.EQ.0) THEN
         DCS2GAM=DCS2GAM*AUX1
      ELSE
         AUX2=((DCOSH(ETA)-1.D0)**3/DCOSH(ETA)**2)**(1/6.D0-1.D0) 
         DCS2GAM=DCS2GAM*0.5D0*41.3772D0*DSINH(ETA)
         DCS2GAM=DCS2GAM*(DCOSH(ETA)+2.D0)/DCOSH(ETA)**3
         DCS2GAM=DCS2GAM*AUX2*(DCOSH(ETA)-1.D0)**2/6.D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CS2GAM(M)

       IMPLICIT NONE
       DOUBLE PRECISION  CS2GAM,ALPHAS,M,U
       DOUBLE PRECISION  ALPHAE,AUX0,AUX1,AUX2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
        ALPHAE=1.D0/128.D0 
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        CS2GAM=(11.D0*ALPHAS(M/2.D0)*ALPHAE/(18.D0*M))**2/PI
C--- using parametrization on U ---------------------------
C--- U>2, <1%; 0.05<U<2, <3%; -----------------------------
C--- 0.01<U<0.05, <8%; U<0.01,<0.001% ---------------------
        IF (U.GT.0.01D0) THEN
         AUX0=U-0.01D0
        ELSE
         AUX0=0.D0
        ENDIF 
        AUX1=1.D0+3.3D0*DEXP(-4.83D0*AUX0**0.53D0)
        AUX2=95.583D0*AUX1*U**1.37D0/(2.9D0+U**1.5614D0)
        IF (U.LE.0.01D0.AND.U.GT.0.D0) THEN
         AUX2=AUX2/(5.4408D0*U**0.37D0)
        ENDIF 
        IF (U.EQ.0.D0) THEN
         AUX2=0.D0
        ENDIF
        CS2GAM=CS2GAM*AUX2
         
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
       SUBROUTINE FINVU2GAM(RF,MG,GETA)

       IMPLICIT NONE
       DOUBLE PRECISION GETA,RF,R,MG
       DOUBLE PRECISION AUX1,AUX2

        R=RF**6.D0*(MG-1)**3/MG**2
        AUX1=DSQRT(4.D0*R+27.D0)*R*3.D0*DSQRT(3.D0)
        AUX1=AUX1+27.D0*R+18.D0*R**2+2.D0*R**3
        AUX1=(0.5D0*AUX1)**(1.D0/3.D0)
        AUX2=((3.D0+R)+R*(6.D0+R)/AUX1+AUX1)/3.D0
        GETA=DLOG(AUX2+DSQRT(AUX2**2-1))
        
       RETURN
       END  

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMU2GAM(N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMU2GAM,MG,ALPHAE
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUD02
       DOUBLE PRECISION CS2GAM,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

        ALPHAE=1/128.D0
        IF (N.EQ.0) THEN
          FMU2GAM=CS2GAM(MG*MGGCUT)
          FMU2GAM=FMU2GAM*SOFTSURV(1,MG*MGGCUT)
          FMU2GAM=FMU2GAM*ISUD02(1,0.5D0*MG*MGGCUT)
          FMU2GAM=FMU2GAM/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1/MGGCUT
          AUX2=(11.D0*ALPHAE/18.D0)**2*AUX1*AUX1*41.3772D0/PI
          POWCS=2.D0*0.15D0+2.D0
          STE1=(PLUM+PSIDD2)+POWCS+1.D0/3.D0
          STE2=0.5D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMU2GAM=AUX2*AUX3
          FMU2GAM=FMU2GAM*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMU2GAM=FMU2GAM*ISUD02(2,0.5D0*MG*MGGCUT)*MG**PSUD
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENEREX2GAM(GMGG,GETAJ)
       
       IMPLICIT NONE
     
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION FMU2GAM,DCS2GAM
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
        DOUBLE PRECISION RMG,RETAJ,RAT
      DOUBLE PRECISION POWCS,STE1,STE2
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2           

      POWCS=2.D0*0.15D0+2.D0 
      STE1=(PLUM+PSIDD2)+POWCS+1.D0/3.D0
      STE2=0.5D0
 71   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMU2GAM(0,RMG)/FMU2GAM(1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 71
      GMG=RMG
      GMGG=MGGCUT*GMG

 72   CALL FINVU2GAM(PYR(0),GMG,RETAJ)         
      RAT=DCS2GAM(0,GMGG,RETAJ)/DCS2GAM(1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 72
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 
      
       RETURN
       END
 
C-============================================================
C-============================================================
C-============================================================
C-========= exclusive g g -> g g g* Jz=0 =====================
C-============================================================
c--- aux. functions for differential cross-section -----------
c--- phase space for the 3rd gluon dx3 dY3, integrated in fi3 --
c--- in the C.M. of the initial g g system -------------------
c--- pt3=LAMQCD*EXP(Y3/2), fi3 - isotropic -------------------

c--- part of the amplitude for the 3rd gluon -----------------
       FUNCTION DXY3G(X3,Y3,M,ETA) 

       IMPLICIT NONE
       DOUBLE PRECISION DXY3G,ALPHAS
       DOUBLE PRECISION M,MU,ETA,X3,Y3,PT3,RDI
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           
c---- table for 3g ratio dIsud3g(x,mu)**2/Isud0(mu)**2 -------
      DOUBLE PRECISION RDI3G,FRDI3G,DX2,DY2,X02,Y02
      COMMON/EDDETAB2/ RDI3G(630),FRDI3G(30,21),
     & DX2,DY2,X02,Y02 

       MU=M/2.D0
       PT3=LAMQCD*DEXP(Y3/2.D0)
       IF (MU.LE.150.D0) THEN 
         CALL LINTERPOL2(FRDI3G,30,21,X02,Y02,DX2,DY2,MU,X3,RDI)
       ELSE 
         CALL LINTERPOL2(FRDI3G,30,21,X02,Y02,DX2,DY2,150.D0,X3,RDI)
       ENDIF       
       DXY3G=RDI*3.D0*ALPHAS(PT3)/(8.D0*PI*X3*(1-X3))
       DXY3G=DXY3G*(1.D0+(1-X3)**4+X3**4*(1.D0-0.5D0/DCOSH(ETA)**2))

       RETURN
       END

c--- integrated in Y part of the 3rd gluon -------------------
c--- N=0 exact, N>0 upper estimation for X3 generator --------
       FUNCTION DX3G(N,X3,M,ETA) 

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DX3G,IALPHAS
c       DOUBLE PRECISION M,MU,ETA,X3,PT3,RDI
       DOUBLE PRECISION M,MU,ETA,X3,RDI
       DOUBLE PRECISION PT3MIN,PT3MAX
c       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION AUX1,AUX2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)        
c---- table for 3g ratio dIsud3g(x,mu)**2/Isud0(mu)**2 -------
      DOUBLE PRECISION RDI3G,FRDI3G,DX2,DY2,X02,Y02
      COMMON/EDDETAB2/ RDI3G(630),FRDI3G(30,21),
     & DX2,DY2,X02,Y02 

       MU=M/2.D0
       PT3MAX=MU*X3/(1-X3)
       PT3MIN=MU*X3/DCOSH(ETASIMAX/2.D0)
       IF (X3.GT.0.5D0) THEN
        PT3MAX=MU*(1-X3)/X3
       ENDIF 
       IF (PT3MIN.LT.0.5D0) THEN
        PT3MIN=0.5D0
       ENDIF 

       IF (N.EQ.0) THEN
         IF (MU.LE.150.D0) THEN 
           CALL LINTERPOL2(FRDI3G,30,21,X02,Y02,DX2,DY2,MU,X3,RDI)
         ELSE 
           CALL LINTERPOL2(FRDI3G,30,21,X02,Y02,DX2,DY2,150.D0,X3,RDI)
         ENDIF
        DX3G=RDI*3.D0/(8.D0*PI*X3*(1-X3))
        DX3G=DX3G*(1.D0+(1-X3)**4+X3**4*(1.D0-0.5D0/DCOSH(ETA)**2))
        DX3G=DX3G*IALPHAS(PT3MIN,PT3MAX)
       ELSE
        AUX1=(MU/5.D0)**PAR3G(2)
        AUX1=AUX1*PAR3G(1)
        AUX2=PAR3G(3)
        DX3G=AUX1*AUX2/((1-X3)*DSQRT(X3))
       ENDIF

       RETURN
       END

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCS3G(N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCS3G,ALPHAS
       DOUBLE PRECISION M,MU,ETA
       DOUBLE PRECISION AUX1,RIAX,RIBX
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2     
c--- table for 3g ratio Isud3ga(etasimax,mu)**2/Isud0(mu)**2 -
c--- table for 3g ratio Isud3gb(etasimax,mu)**2/Isud0(mu)**2 -
      DOUBLE PRECISION RI3GA,RI3GB,FRI3GA,FRI3GB,
     & DX3,DY3,X03,Y03
      COMMON/EDDETAB3/ RI3GA(480),RI3GB(480),
     & FRI3GA(30,16),FRI3GB(30,16),DX3,DY3,X03,Y03     

      MU=0.5D0*M
      IF (ETASIMAX.LE.15.D0.AND.MU.LE.150.D0) THEN
       CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,MU,ETASIMAX,RIAX)
       CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,MU,ETASIMAX,RIBX)
      ELSE
       IF (ETASIMAX.GT.15.D0.AND.MU.LE.150.D0) THEN
        CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,MU,15.D0,RIAX)
        CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,MU,15.D0,RIBX)      
       ENDIF
       IF (ETASIMAX.GT.15.D0.AND.MU.GT.150.D0) THEN
        CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,150.D0,15.D0,RIAX)
        CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,150.D0,15.D0,RIBX)
        RIAX=RIAX/MU
        RIBX=RIBX/MU
       ENDIF
       IF (ETASIMAX.LE.15.D0.AND.MU.GT.150.D0) THEN
        CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,150.D0,ETASIMAX,
     &   RIAX)
        CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,150.D0,ETASIMAX,
     &   RIBX)
        RIAX=RIAX/MU
        RIBX=RIBX/MU
       ENDIF
      ENDIF 

      DCS3G=18.D0*PI*(ALPHAS(M/2.D0)/M)**2 
      IF (N.EQ.0) THEN
         AUX1=(1.D0-0.5D0/DCOSH(ETA)**2)*RIAX+RIBX
         DCS3G=DCS3G*DCOSH(ETA)**2*AUX1
      ELSE
         AUX1=(DCOSH(ETA)**3*(DCOSH(ETA)-1.D0))**(1/2.09D0-1.D0) 
         DCS3G=DCS3G*0.5D0*2.62108D0*DSINH(ETA)*DCOSH(ETA)
         DCS3G=DCS3G*DCOSH(ETA)*(4.D0*DCOSH(ETA)-3.D0)
         DCS3G=DCS3G*(RIAX+RIBX)*AUX1/2.09D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CS3G(M)

       IMPLICIT NONE
       DOUBLE PRECISION CS3G,ALPHAS,M,U
       DOUBLE PRECISION MU,RIAX,RIBX,AUX1 
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- table for 3g ratio Isud3ga(etasimax,mu)**2/Isud0(mu)**2 -
c--- table for 3g ratio Isud3gb(etasimax,mu)**2/Isud0(mu)**2 -
      DOUBLE PRECISION RI3GA,RI3GB,FRI3GA,FRI3GB,
     & DX3,DY3,X03,Y03
      COMMON/EDDETAB3/ RI3GA(480),RI3GB(480),
     & FRI3GA(30,16),FRI3GB(30,16),DX3,DY3,X03,Y03     

      MU=0.5D0*M
      IF (ETASIMAX.LE.15.D0.AND.MU.LE.150.D0) THEN
       CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,MU,ETASIMAX,RIAX)
       CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,MU,ETASIMAX,RIBX)
      ELSE
       IF (ETASIMAX.GT.15.D0.AND.MU.LE.150.D0) THEN
        CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,MU,15.D0,RIAX)
        CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,MU,15.D0,RIBX)      
       ENDIF
       IF (ETASIMAX.GT.15.D0.AND.MU.GT.150.D0) THEN
        CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,150.D0,15.D0,RIAX)
        CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,150.D0,15.D0,RIBX)
        RIAX=RIAX/MU
        RIBX=RIBX/MU
       ENDIF
       IF (ETASIMAX.LE.15.D0.AND.MU.GT.150.D0) THEN
        CALL LINTERPOL2(FRI3GA,30,16,X03,Y03,DX3,DY3,150.D0,ETASIMAX,
     &   RIAX)
        CALL LINTERPOL2(FRI3GB,30,16,X03,Y03,DX3,DY3,150.D0,ETASIMAX,
     &   RIBX)
        RIAX=RIAX/MU
        RIBX=RIBX/MU
       ENDIF
      ENDIF
      U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
      AUX1=0.5D0*DSINH(U+U)*RIAX+(U+0.5D0*DSINH(U+U))*RIBX
      CS3G=18.D0*PI*(ALPHAS(M/2.D0)/M)**2*AUX1
       
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
c--- SUBROUTINE FINVU3G(RF,MG,GETA)=FINVUGG(RF,MG,GETA)

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMU3G(N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMU3G,MG
c       DOUBLE PRECISION AUX1,AUX2,AUX3,AUX4
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUD02
       DOUBLE PRECISION CS3G,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)    
      
        IF (N.EQ.0) THEN
          FMU3G=CS3G(MG*MGGCUT)
          FMU3G=FMU3G*SOFTSURV(1,MG*MGGCUT)
          FMU3G=FMU3G*ISUD02(1,0.5D0*MG*MGGCUT)
          FMU3G=FMU3G/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1/MGGCUT
          AUX2=18.D0*PI*AUX1*AUX1*2.62108D0
          AUX2=AUX2*PAR3G(4)*(0.5D0*MGGCUT)**PAR3G(5)
          POWCS=2.D0*0.15D0+2.D0-PAR3G(5)
          STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
          STE2=1.D0/2.09D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMU3G=AUX2*AUX3
          FMU3G=FMU3G*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMU3G=FMU3G*ISUD02(2,0.5D0*MG*MGGCUT)*MG**PSUD
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENEREX3G(GMGG,GETAJ,GX3,GPT3,GFI3)
       
       IMPLICIT NONE
   
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION FMU3G,DCS3G,DX3G
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION GX3,RX3,GPT3,RPT3,GFI3
       DOUBLE PRECISION X3GMIN,X3GMAX,X3GMAX0
       DOUBLE PRECISION PT3GMIN,PT3GMAX
       DOUBLE PRECISION POWCS,STE1,STE2
c       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION AUX1,AUX2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2           
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)  

      POWCS=2.D0*0.15D0+2.D0-PAR3G(5) 
      STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
      STE2=1.D0/2.09D0
 91   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMU3G(0,RMG)/FMU3G(1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 91
      GMG=RMG
      GMGG=MGGCUT*GMG

 92   CALL FINVUGG(PYR(0),GMG,RETAJ)         
      RAT=DCS3G(0,GMGG,RETAJ)/DCS3G(1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 92
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 

       X3GMIN=0.5D0/(0.5D0*GMGG+0.5D0)
C-       X3GMAX0=1.D0-X3GMIN
       X3GMAX0=2.D0/3.D0
       X3GMAX=0.5D0*DCOSH(0.5D0*ETASIMAX)
       X3GMAX=X3GMAX*(DSQRT(1.D0+4.D0/DCOSH(0.5D0*ETASIMAX))-1.D0)
       IF (X3GMAX.GT.X3GMAX0) THEN
        X3GMAX=X3GMAX0
       ENDIF 
c---- generation of 3rd jet variables
       GFI3=2.D0*PI*PYR(0)
       AUX1=(1.D0-DSQRT(X3GMIN))
       AUX2=(1.D0-DSQRT(X3GMAX))
 95   RX3=(1.D0-AUX1*(AUX2/AUX1)**PYR(0))**2
      RAT=DX3G(0,RX3,GMGG,GETAJ)/DX3G(1,RX3,GMGG,GETAJ)
      IF (RAT.LE.PYR(0)) GOTO 95 
      GX3=RX3
c---
c---was a BUG! 1-GX3 was changed to DSQRT(1-GX3)
       PT3GMAX=0.5D0*GMGG*GX3/DSQRT(1-GX3)
       PT3GMIN=0.5D0*GMGG*GX3/DCOSH(ETASIMAX/2.D0)
       IF (GX3.GT.0.5D0) THEN
        PT3GMAX=0.5D0*GMGG*(1-GX3)/GX3
       ENDIF 
       IF (PT3GMIN.LT.0.5D0) THEN
        PT3GMIN=0.5D0
       ENDIF 
       AUX1=DLOG(PT3GMAX/LAMQCD)/DLOG(PT3GMIN/LAMQCD)
       RPT3=LAMQCD*(PT3GMIN/LAMQCD)**(AUX1**PYR(0))
       IF (NLOSW.EQ.1) THEN
 96   RPT3=LAMQCD*(PT3GMIN/LAMQCD)**(AUX1**PYR(0))
        AUX2=BF1*DLOG(2.D0*DLOG(RPT3/LAMQCD))
        RAT=1.D0-AUX2/(BF0*2.D0*DLOG(RPT3/LAMQCD))
        IF (RAT.LE.PYR(0)) GOTO 96
       ENDIF
       GPT3=RPT3
      
       RETURN
       END
 
C-============================================================
C-============================================================
C-============================================================
C-========= exclusive g g -> Q Qbar g Jz=0; MQ=0 =============
C-============================================================

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCSQQG(N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCSQQG,ALPHAS
       DOUBLE PRECISION M,ETA,AUX1
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)
       
      DCSQQG=4.D0*PI*(ALPHAS(M/2.D0)/M)**2/3.D0
c--- mult. by additional amplitude factor -------------------- 
c--- of integrated phase space -------------------------------
c--- of g-soft jet. ------------------------------------------
      DCSQQG=DCSQQG*4.D0*ALPHAS(M/2.D0)/(3.D0*PI)
      DCSQQG=DCSQQG*XMAX3J**4*DSIN(0.5D0*DER3J)**2
      IF (N.EQ.0) THEN
         DCSQQG=DCSQQG*DCOSH(ETA)*DCOSH(ETA)
      ELSE
         AUX1=(DCOSH(ETA)**3*(DCOSH(ETA)-1.D0))**(1/2.09D0-1.D0) 
         DCSQQG=DCSQQG*0.5D0*2.62108D0*DSINH(ETA)*DCOSH(ETA)
         DCSQQG=DCSQQG*DCOSH(ETA)*(4.D0*DCOSH(ETA)-3.D0)
         DCSQQG=DCSQQG*AUX1/2.09D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CSQQG(M)

       IMPLICIT NONE
       DOUBLE PRECISION  CSQQG,ALPHAS,M,U
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)
       
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        CSQQG=4.D0*PI*(ALPHAS(M/2.D0)/M)**2/3.D0
c--- mult. by additional amplitude factor -------------------- 
c--- of integrated phase space -------------------------------
c--- of g-soft jet. ------------------------------------------
        CSQQG=CSQQG*4.D0*ALPHAS(M/2.D0)/(3.D0*PI)
        CSQQG=CSQQG*XMAX3J**4*DSIN(0.5D0*DER3J)**2
        CSQQG=CSQQG*(U+0.5D0*DSINH(U+U))
       
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
c---SUBROUTINE FINVUQQG(RF,MG,GETA)=FINVUGG(RF,MG,GETA)---

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMUQQG(N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMUQQG,MG
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUD02
       DOUBLE PRECISION CSQQG,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)

        IF (N.EQ.0) THEN
          FMUQQG=CSQQG(MG*MGGCUT)
          FMUQQG=FMUQQG*SOFTSURV(1,MG*MGGCUT)
          FMUQQG=FMUQQG*ISUD02(1,0.5D0*MG*MGGCUT)
          FMUQQG=FMUQQG/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1**1.5D0/MGGCUT
          AUX2=4.D0*PI*AUX1*AUX1*2.62108D0/3.D0
          AUX2=AUX2*XMAX3J**4*DSIN(0.5D0*DER3J)**2
          AUX2=AUX2*4.D0/(3.D0*PI)
          POWCS=3.D0*0.15D0+2.D0
          STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
          STE2=1.D0/2.09D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMUQQG=AUX2*AUX3
          FMUQQG=FMUQQG*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMUQQG=FMUQQG*ISUD02(2,0.5D0*MG*MGGCUT)*MG**PSUD
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENEREXQQG(GMGG,GETAJ,GX3,GFIS,GTHETAS)
       
       IMPLICIT NONE
  
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION GX3,GFIS,GTHETAS
       DOUBLE PRECISION FMUQQG,DCSQQG
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION POWCS,STE1,STE2,AUX1
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW              
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5) 

      POWCS=3.D0*0.15D0+2.D0 
      STE1=(PLUM+PSIDD2)+POWCS-3.D0/2.09D0
      STE2=1.D0/2.09D0
 81   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMUQQG(0,RMG)/FMUQQG(1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 81
      GMG=RMG
      GMGG=MGGCUT*GMG

 82   CALL FINVUGG(PYR(0),GMG,RETAJ)         
      RAT=DCSQQG(0,GMGG,RETAJ)/DCSQQG(1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 82
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 

c--- GX3,GFIS,GTHETAS -------
        GFIS=PYR(0)*2.D0*PI
        GX3=PYR(0)**0.25D0*XMAX3J
        AUX1=PYR(0)*DSIN(DER3J/2.D0)**2
        GTHETAS=2.D0*DATAN(1.D0/DSQRT(1.D0/AUX1-1.D0))
      
       RETURN
       END
 
C-============================================================
C-============================================================
C-=============================================================
C-========== semi-inclusive g g -> g g Jz!=0 ==================
C-=============================================================

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCSGGSI(N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCSGGSI,ALPHAS
       DOUBLE PRECISION M,ETA,AUX1
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
      
      DCSGGSI=36.D0*PI*(ALPHAS(M/2.D0)/M)**2
      AUX1=DCOSH(ETA)*(1.D0-0.25D0/DCOSH(ETA)**2)
      DCSGGSI=DCSGGSI*AUX1*AUX1      
c-      IF (N.EQ.0) THEN
c-         AUX1=DCOSH(ETA)*(1.D0-0.25D0/DCOSH(ETA)**2)
c-         DCSGGSI=DCSGGSI*AUX1*AUX1
c-      ELSE
c--- here we have exact inverse function of the integral -----
c--- i.e. we need no upper estimation ------------------------
c-         AUX1=DCOSH(ETA)*(1.D0-0.25D0/DCOSH(ETA)**2)
c-         DCSGGSI=DCSGGSI*AUX1*AUX1
c-      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CSGGSI(M)

       IMPLICIT NONE
       DOUBLE PRECISION  CSGGSI,ALPHAS,M,U
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        CSGGSI=36.D0*PI*(ALPHAS(M/2.D0)/M)**2
        CSGGSI=CSGGSI*(0.125D0*DTANH(U)+0.5D0*DSINH(U+U))
       
       RETURN
       END              

c--- subroutine to generate etaj* --------------- 
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
       SUBROUTINE FINVUGGSI(RF,MG,GETA)

       IMPLICIT NONE
       DOUBLE PRECISION GETA,RF,R,MG
       DOUBLE PRECISION AUX1,AUX2,AUX3,AUX4
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           

       AUX1=DLOG(MG+DSQRT(MG**2-1))
       R=RF*(0.125D0*DTANH(AUX1)+0.5D0*DSINH(AUX1+AUX1))
       AUX2=3.D0*DSQRT(3.D0)/DSQRT(27.D0+64.D0*R*R)
c--- AUX3=DASIN(AUX2)/3.D0-PI/6.D0 ---       
       AUX3=2.D0*DATAN((1.D0-DSQRT(1.D0-AUX2**2))/AUX2)/3.D0-PI/6.D0
       AUX4=2.D0*DSIN(AUX3)*DSQRT(27.D0+64.D0*R*R)
       AUX4=AUX4+8.D0*R
c-       GETA=DATANH(AUX4/3.D0)
       GETA=0.5D0*DLOG((1.D0+AUX4/3.D0)/(1.D0-AUX4/3.D0))    
        
       RETURN
       END  

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMUGGSI(N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMUGGSI,MG
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUDGEN2
       DOUBLE PRECISION CSGGSI,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

        IF (N.EQ.0) THEN
          FMUGGSI=CSGGSI(MG*MGGCUT)
          FMUGGSI=FMUGGSI*SOFTSURV(1,MG*MGGCUT)
          FMUGGSI=FMUGGSI*ISUDGEN2(1,0.5D0*MG*MGGCUT)
          FMUGGSI=FMUGGSI/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1/MGGCUT
          AUX2=36.D0*PI*AUX1*AUX1*1.49332D0
          POWCS=2.D0*0.15D0+2.D0
          STE1=PLUM+POWCS-1.48D0
          STE2=0.48D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMUGGSI=AUX2*AUX3
          FMUGGSI=FMUGGSI*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMUGGSI=FMUGGSI*ISUDGEN2(2,0.5D0*MG*MGGCUT)
          FMUGGSI=FMUGGSI*MG**(PSUD-PSIDD2)
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENERSIGG(GMGG,GETAJ)
       
       IMPLICIT NONE
    
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
c       DOUBLE PRECISION FMUGGSI,DCSGGSI
       DOUBLE PRECISION FMUGGSI
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION POWCS,STE1,STE2
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2           

      POWCS=2.D0*0.15D0+2.D0 
      STE1=PLUM+POWCS-1.48D0
      STE2=0.48D0
 53   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMUGGSI(0,RMG)/FMUGGSI(1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 53
      GMG=RMG
      GMGG=MGGCUT*GMG

      CALL FINVUGGSI(PYR(0),GMG,RETAJ)         
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 
      
       RETURN
       END
 
C-=============================================================
C-=============================================================
C-=============================================================
C-========== semi-inclusive g g -> Q Qbar Jz!=0 ===============
C-=============================================================
c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCSQQSI(MQ,N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCSQQSI,ALPHAS
       DOUBLE PRECISION MQ,M,ETA
       DOUBLE PRECISION AUX1,AUX2,AUX3
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
      
      DCSQQSI=PI*(ALPHAS(M/2.D0)/M)**2/3.D0
      IF (N.EQ.0) THEN
         AUX1=8.D0*(MQ*DCOSH(ETA)/M)**2
         AUX2=1.D0-1.D0/(2.D0*DCOSH(ETA)**2)
         AUX3=AUX2*(1.D0-AUX1)+AUX1*(1.D0-2.D0*(MQ/M)**2)
         DCSQQSI=DCSQQSI*AUX3
          IF (DCSQQSI.LE.0.D0) THEN
	   DCSQQSI=0.D0
          ENDIF
      ELSE
         AUX1=((DCOSH(ETA)-1.D0)**4/DCOSH(ETA))**(1/8.05D0-1.D0)
         AUX2=(DCOSH(ETA)-1.D0)**3*DSINH(ETA)
         AUX2=AUX2*(3.D0*DCOSH(ETA)+1.D0)/DCOSH(ETA)**2
         DCSQQSI=DCSQQSI*0.5D0*2.7053D0*AUX1*AUX2/8.05D0
         DCSQQSI=DCSQQSI*1.005D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CSQQSI(MQ,M)

       IMPLICIT NONE
       DOUBLE PRECISION CSQQSI,ALPHAS
       DOUBLE PRECISION MQ,M,U,AUX1
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        AUX1=2.D0*U*(1.D0+(2.D0*MQ/M)**2-8.D0*(MQ/M)**4)
        AUX1=AUX1-8.D0*(MQ/M)**4*DSINH(U+U)
        AUX1=AUX1-DTANH(U)
        CSQQSI=PI*(ALPHAS(M/2.D0)/M)**2/3.D0
        CSQQSI=CSQQSI*AUX1
        IF (CSQQSI.LE.0.D0) THEN
         CSQQSI=0.D0
        ENDIF 
       
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
      SUBROUTINE FINVUQQSI(RF,MG,GETA)

       IMPLICIT NONE
       DOUBLE PRECISION GETA,RF,R,MG
       DOUBLE PRECISION AUX1,AUX2,AUX3,AUX4
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           

       R=RF**8.05D0*(MG-1.D0)**4/MG
       AUX1=9.D0*R*R+DSQRT(3.D0*R**3*(256.D0+27.D0*R))
       AUX2=(2.D0*AUX1)**(1.D0/3.D0)
       AUX2=AUX2-8.D0*R*(3.D0/AUX1)**(1.D0/3.D0)
       AUX3=DSQRT(AUX2)
       AUX4=(DSQRT(12.D0*R/AUX3-AUX2)+AUX3)/(2.D0*6.D0**(1.D0/3.D0))
       AUX4=AUX4+1.D0
       GETA=DLOG(AUX4+DSQRT(AUX4**2-1))
        
       RETURN
       END 
       
c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMUQQSI(MQ,N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMUQQSI,MG,MQ
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUDGEN2
       DOUBLE PRECISION CSQQSI,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

        IF (N.EQ.0) THEN
          FMUQQSI=CSQQSI(MQ,MG*MGGCUT)
          FMUQQSI=FMUQQSI*SOFTSURV(1,MG*MGGCUT)
          FMUQQSI=FMUQQSI*ISUDGEN2(1,0.5D0*MG*MGGCUT)
          FMUQQSI=FMUQQSI/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1/MGGCUT
          AUX2=PI*AUX1*AUX1*2.7053D0/3.D0
          POWCS=2.D0*0.15D0+2.D0
          STE1=PLUM+POWCS+1.D0/8.05D0
          STE2=4.D0/8.05D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMUQQSI=AUX2*AUX3
          FMUQQSI=FMUQQSI*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMUQQSI=FMUQQSI*ISUDGEN2(2,0.5D0*MG*MGGCUT)
          FMUQQSI=FMUQQSI*MG**(PSUD-PSIDD2)
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENERSIQQ(MQ,GMGG,GETAJ)
       
       IMPLICIT NONE
      
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION FMUQQSI,DCSQQSI,MQ
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION POWCS,STE1,STE2
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2           

      POWCS=2.D0*0.15D0+2.D0 
      STE1=PLUM+POWCS+1.D0/8.05D0
      STE2=4.D0/8.05D0
 63   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMUQQSI(MQ,0,RMG)/FMUQQSI(MQ,1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 63
      GMG=RMG
      GMGG=MGGCUT*GMG

 64   CALL FINVUQQSI(PYR(0),GMG,RETAJ)         
      RAT=DCSQQSI(MQ,0,GMGG,RETAJ)/DCSQQSI(MQ,1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 64
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 
      
       RETURN
       END
 
C-=============================================================
C-=============================================================
C-=============================================================
C-========== semi-inclusive g g -> gamma gamma Jz!=0 ==========
C-=============================================================
c--- aux function -----
      FUNCTION AU2GAMSI(ETA)

       IMPLICIT NONE
       DOUBLE PRECISION AU2GAMSI,ETA,TETA
       DOUBLE PRECISION AUX0,AUX1,AUX2,AUX3
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         

       TETA=DTANH(ETA)
       AUX0=5.D0+(TETA+2.D0)*TETA
       AUX1=DLOG(0.5D0*(1+TETA))
       AUX2=1.D0+(3.D0+TETA)*AUX1/(1.D0-TETA)
       AUX2=AUX2+0.5D0*AUX0*(AUX1/(1.D0-TETA))**2
       AUX3=PI*((1.D0-TETA)*(3.D0+TETA)+AUX0*AUX1)/(1.D0-TETA)**2
       AU2GAMSI=AUX2*AUX2+AUX3*AUX3
      
       RETURN
       END

c--- differential cross-section -----------------
c--- and its upper limit for etaj* generation ---
c--- N=0 exact, N>0 upper limit -----------------
       FUNCTION DCS2GAMSI(N,M,ETA)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION DCS2GAMSI,ALPHAS
       DOUBLE PRECISION AU2GAMSI
       DOUBLE PRECISION M,ETA,ALPHAE
       DOUBLE PRECISION AUX0,AUX1,AUX2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         

      ALPHAE=1.D0/128.D0      
      DCS2GAMSI=(11.D0*ALPHAS(M/2.D0)*ALPHAE/(18.D0*M))**2/PI
      AUX0=1.D0-2.D0*ETA*DTANH(ETA)
      AUX0=AUX0+(0.25D0*PI**2+ETA**2)*(1.D0+DTANH(ETA)**2)
      AUX1=(1+AUX0*AUX0+AU2GAMSI(ETA)+AU2GAMSI(-ETA))/DCOSH(ETA)**2
      IF (N.EQ.0) THEN
         DCS2GAMSI=DCS2GAMSI*AUX1
      ELSE
         AUX2=((DCOSH(ETA)-1.D0)**3/DCOSH(ETA)**2)**(1/6.D0-1.D0) 
         DCS2GAMSI=DCS2GAMSI*0.5D0*72.0894D0*DSINH(ETA)
         DCS2GAMSI=DCS2GAMSI*(DCOSH(ETA)+2.D0)/DCOSH(ETA)**3
         DCS2GAMSI=DCS2GAMSI*AUX2*(DCOSH(ETA)-1.D0)**2/6.D0
         DCS2GAMSI=DCS2GAMSI*1.265D0
      ENDIF 
      
       RETURN
       END
       
c--- integrated cross-section -----------
       FUNCTION CS2GAMSI(M)

       IMPLICIT NONE
       DOUBLE PRECISION  CS2GAMSI,ALPHAS,M,U
       DOUBLE PRECISION  ALPHAE,AUX0,AUX1
       DOUBLE PRECISION  AUX2,AUX3
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
     
        ALPHAE=1.D0/128.D0 
        U=DLOG(M/MGGCUT+DSQRT((M/MGGCUT)**2-1))
        CS2GAMSI=(11.D0*ALPHAS(M/2.D0)*ALPHAE/(18.D0*M))**2/PI
C--- using parametrization on U ---------------------------
C--- U>2, <1%; 0.05<U<2, <3%; -----------------------------
C--- 0.01<U<0.05, <8%; U<0.01,<0.001% ---------------------
        IF (U.GT.0.01D0) THEN
         AUX0=U-0.01D0
        ELSE
         AUX0=0.D0
        ENDIF 
        AUX1=1.D0+3.3D0*DEXP(-4.83D0*AUX0**0.53D0)
        AUX2=95.583D0*AUX1*U**1.37D0/(2.9D0+U**1.5614D0)
        IF (U.LE.0.01D0.AND.U.GT.0.D0) THEN
         AUX2=AUX2/(5.4408D0*U**0.37D0)
        ENDIF
        IF (U.EQ.0.D0) THEN
         AUX2=0.D0
        ENDIF 
c------------------        
        IF (U.LE.0.05D0) THEN
         AUX3=0.8765D0*9.85D0*U
        ENDIF
        IF (U.GT.0.05D0.AND.U.LE.0.33D0) THEN
         AUX3=9.85D0*U**1.05D0
        ENDIF
        IF (U.GT.0.33D0.AND.U.LE.1.8D0) THEN
	 AUX3=0.97D0*14.15D0*U**1.36D0
        ENDIF
        IF (U.GT.1.8D0) THEN
	 AUX3=33.D0+13.D0*(U-1.8D0)**0.45D0
        ENDIF
        IF (U.GT.1.7D0.AND.U.LE.2.8D0) THEN
	 AUX3=AUX3*0.87
        ENDIF
        
        CS2GAMSI=CS2GAMSI*(AUX2+AUX3)
         
       RETURN
       END              

c--- subroutine to generate upper curve --------- 
c--- on etaj* dependence ------------------------
c--- RF=PYR(0), MG=MGG/MGGCUT --------------------
c---SUBROUTINE FINVU2GAMSI(RF,MG,GETA)=FINVU2GAM(RF,MG,GETA) - 

c--- upper function for general MG=MGG/MGGCUTdependence ------
c--- N=0 - exact function, N>0 - upper curve ----
       FUNCTION FMU2GAMSI(N,MG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION FMU2GAMSI,MG,ALPHAE
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION SOFTSURV,ISUDGEN2
       DOUBLE PRECISION CS2GAMSI,POWCS
       DOUBLE PRECISION GFMGEN,STE1,STE2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW                
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2

        ALPHAE=1/128.D0
        IF (N.EQ.0) THEN
          FMU2GAMSI=CS2GAMSI(MG*MGGCUT)
          FMU2GAMSI=FMU2GAMSI*SOFTSURV(1,MG*MGGCUT)
          FMU2GAMSI=FMU2GAMSI*ISUDGEN2(1,0.5D0*MG*MGGCUT)
          FMU2GAMSI=FMU2GAMSI/MG**(PLUM-PSURV-PSUD+PSIDD2)
        ELSE
          AUX1=0.236469D0*(1-NLOSW*0.14D0)/(0.5D0*MGGCUT)**0.15D0
          AUX1=AUX1/MGGCUT
          AUX2=(11.D0*ALPHAE/18.D0)**2*AUX1*AUX1*72.0894D0/PI
          POWCS=2.D0*0.15D0+2.D0
          STE1=PLUM+POWCS+1.D0/3.D0
          STE2=0.5D0
          AUX3=GFMGEN(STE1,STE2,MG)  
          FMU2GAMSI=AUX2*AUX3
          FMU2GAMSI=FMU2GAMSI*SOFTSURV(2,MG*MGGCUT)*MG**PSURV
          FMU2GAMSI=FMU2GAMSI*ISUDGEN2(2,0.5D0*MG*MGGCUT)
          FMU2GAMSI=FMU2GAMSI*MG**(PSUD-PSIDD2)
        ENDIF 
           
       RETURN
       END  

c--- generator of (Mgg,etaj*) in c.m. of initial gg ----------
       SUBROUTINE GENERSI2GAM(GMGG,GETAJ)
       
       IMPLICIT NONE
    
       DOUBLE PRECISION PYR,GMG,GMGG,GETAJ
       DOUBLE PRECISION FMU2GAMSI,DCS2GAMSI
c       DOUBLE PRECISION RMG,RMGG,RETAJ,RAT
       DOUBLE PRECISION RMG,RETAJ,RAT
       DOUBLE PRECISION POWCS,STE1,STE2
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2           

      POWCS=2.D0*0.15D0+2.D0 
      STE1=PLUM+POWCS+1.D0/3.D0
      STE2=0.5D0
 73   CALL GFMGENINV(STE1,STE2,PYR(0),RMG)  
      RAT=FMU2GAMSI(0,RMG)/FMU2GAMSI(1,RMG)
      IF (RAT.LE.PYR(0)) GOTO 73
      GMG=RMG
      GMGG=MGGCUT*GMG

 74   CALL FINVU2GAM(PYR(0),GMG,RETAJ)         
      RAT=DCS2GAMSI(0,GMGG,RETAJ)/DCS2GAMSI(1,GMGG,RETAJ)
      IF (RAT.LE.PYR(0)) GOTO 74
      IF (PYR(0).LE.0.5D0) THEN
       GETAJ=RETAJ
      ELSE
       GETAJ=-RETAJ
      ENDIF 
      
       RETURN
       END
 
C-=============================================================
C-=============================================================

C-============================================================
C-== functions for most processes ============================
c--- general aux. upper function for MG dependence -----------
       FUNCTION GFMGEN(ST1,ST2,MG)

       IMPLICIT NONE
       DOUBLE PRECISION GFMGEN,MG,ST1,ST2
       DOUBLE PRECISION V1,V2,VM,FVM
     
          VM=ST1/(ST1-ST2)
          FVM=(VM-1.D0)**ST2/VM**ST1
          V1=1.D0+FVM**(1.D0/ST2)
          V2=1.D0/FVM**(1.D0/(ST1-ST2))
          IF (MG.LT.V1) THEN                   
            GFMGEN=(MG-1.D0)**ST2
          ELSE 
            IF (MG.LE.V2) THEN 
             GFMGEN=FVM
            ELSE
             GFMGEN=1.D0/MG**(ST1-ST2)
            ENDIF
          ENDIF  
                     
       RETURN
       END  

c--- general aux. subroutine for MG generation ---------------
c--- RF=PYR(0) in PYTHIA -------------------------------------
       SUBROUTINE GFMGENINV(ST1,ST2,RF,GMG)

       IMPLICIT NONE
       DOUBLE PRECISION GMG,ST1,ST2,RF,R
       DOUBLE PRECISION V1,V2,VM,V3
       DOUBLE PRECISION FVM,FV1,FV2,FV3
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2       
     
          VM=ST1/(ST1-ST2)
          FVM=(VM-1.D0)**ST2/VM**ST1
          V1=1.D0+FVM**(1.D0/ST2)
          V2=1.D0/FVM**(1.D0/(ST1-ST2))
          V3=MXMAX/MGGCUT
          FV1=(V1-1.D0)**(ST2+1.D0)/(ST2+1.D0)
          FV2=FV1+(V2-V1)*FVM
          FV3=FV2+(V2**(ST2+1.D0-ST1)-V3**(ST2+1.D0-ST1))/(ST1-ST2-1.D0)
          R=RF*FV3
          
          IF (R.LT.FV1) THEN                   
            GMG=1.D0+((ST2+1.D0)*R)**(1.D0/(ST2+1.D0))
          ELSE 
            IF (R.LE.FV2) THEN 
             GMG=(R+V1*FVM+FV1)/FVM
            ELSE
             GMG=1.D0/((FV2-R)*(ST1-ST2-1.D0)+V2**(ST2+1.D0-ST1))
             GMG=GMG**(1.D0/(ST1-ST2-1.D0))
            ENDIF
          ENDIF  
                     
       RETURN
       END   
C-============================================================
C-============================================================

C-++================== B E G I N  C A S C A D ================
c- mass correction subroutine
      SUBROUTINE SIMXCOR(N,MO,MN,PO,PN)
      
      IMPLICIT NONE
      INTEGER N
      DOUBLE PRECISION MO,MN,PO(5),PN(5),PPL,PMI,PT,X

      PPL=(PO(4)+PO(3))/DSQRT(2.D0)
      PMI=(PO(4)-PO(3))/DSQRT(2.D0)
      PT=DSQRT(PO(1)**2+PO(2)**2)
      PN(1)=PO(1)
      PN(2)=PO(2)
      PN(5)=PO(5)
      IF (N.EQ.1) THEN
       X=DSQRT(2.D0)*PPL/MO
       PN(4)=0.5D0*MN*X*(1.D0+(PT/(MN*X))**2)
       PN(3)=0.5D0*MN*X*(1.D0-(PT/(MN*X))**2)
      ELSE
       X=DSQRT(2.D0)*PMI/MO
       PN(4)=0.5D0*MN*X*(1.D0+(PT/(MN*X))**2)
       PN(3)=-0.5D0*MN*X*(1.D0-(PT/(MN*X))**2)
      ENDIF
     
      RETURN
      END

C-============================================================
C-= combinatorics of singlets N -> 2*N2+3*N3 all events ======
C-============================================================
c-check+
       SUBROUTINE COMBSING(N,N2,N3,CEVENT)
       
       IMPLICIT NONE
       
       INTEGER FACTN
c- numbers of gluons, parity, 
c- max. number of 2*N2+3*N3 (N2,N3-INTEGER) combinations minus 1, 
c- and number of possible combinations
       INTEGER N,INDC,KMAX,NSUMEV
c- generated event,
c- corresponding K to the generated event,
c- corresponding N2 and N3 to the event.
       INTEGER NRK,K,N2,N3
c- vars, for distribution of N numbers between 2*N2 and 3*N3 cells:
c- generators for probability for 2*N2 cell and 3*N3 cell, their sum,
c- index of massives for both cells
       INTEGER I2,I3,I23,K2,K3
c- massiv of cell 2*N2, massive of cell 3*N3,
c- indexes of indexes for C2 and C3.
c- use only 2*N2 elements from C2 and IC2, 3*N3 from C3 and IC3
c-       INTEGER C2(N),C3(N),IC2(N),IC3(N)
c- order of numbers of gluons=event for PYTHIA
       INTEGER CEVENT(2,500),CAUX(500),CAUX2(500)
c- number of events at each K=0->KMAX => KEVE(K+1),
c- check points for K generation KEVES(K+1)=SUM[0->K][KEVE(K+1)]
c- for even:N!/{([N/2]-3K)!2^([N/2]-3K)(2K)! 6^(2K)} -------------
c- for add: N!/{([(N-3)/2]-3K)!2^([(N-3)/2]-3K)(2K+1)!6^(2K+1)} ---
c- NEVE23 number of events in 2*N2 (or 3*N3) from N: N!/(2*N2)!/(3*N3)! --
c- number of events inside couples and triples --------------------
c- NEVE2=(2*N2)!/N2!/2**N2, NEVE3=(3*N3)!/N3!/6**N3 ---------------
       INTEGER KEVE(500),KEVES(500)
c- random 0->1
       DOUBLE PRECISION PYR 
c- indexes,probability, aux. vars.
c       INTEGER I,J,L,II,JJ,LL,IC,III 
       INTEGER I,J,L,II,JJ,LL,IC 
c       DOUBLE PRECISION PROB,AUX1,AUX2
       DOUBLE PRECISION PROB,AUX1

       K=0
c- definition N - even or add  
c- and to fill CEVENT(2,N) 1rst iteration
        DO I=1,N
         CEVENT(1,I)=I
         CAUX(I)=I
        ENDDO
        AUX1=0.5D0*N-DBLE(INT(N/2))
c-        WRITE(*,*)' N=',N,' AUX1=',AUX1
        IF (AUX1.EQ.0.D0) THEN
         INDC=0
         KMAX=INT(N/6)
         J=INT(N/2)         
         N2=INT(N/2)
         N3=0
         DO I=1,J
          CEVENT(2,2*I-1)=2
          CEVENT(2,2*I)=1
         ENDDO         
        ELSE
         INDC=1
         KMAX=INT((N-3)/6)
         J=INT((N-3)/2)
         N3=1
         N2=N-3*N3
         CEVENT(2,N)=1
         CEVENT(2,N-1)=2
         CEVENT(2,N-2)=2
         IF (J.GE.1) THEN
          DO I=1,J
           CEVENT(2,2*I-1)=2
           CEVENT(2,2*I)=1
          ENDDO
         ENDIF
        ENDIF        
c- definition of total number of indep. events.
       NSUMEV=0
       IF (INDC.EQ.0) THEN
        DO I=0,KMAX
         KEVE(I+1)=FACTN(N)/(FACTN(INT(N/2)-3*I)*
     &    2**(INT(N/2)-3*I)*FACTN(2*I)*6**(2*I))
         NSUMEV=NSUMEV+KEVE(I+1)
         KEVES(I+1)=NSUMEV
        ENDDO
       ELSE
        DO I=0,KMAX
         KEVE(I+1)=FACTN(N)/(FACTN(INT((N-3)/2)-3*I)*
     &   2**(INT((N-3)/2)-3*I)*FACTN(2*I+1)*6**(2*I+1))
         NSUMEV=NSUMEV+KEVE(I+1)
         KEVES(I+1)=NSUMEV
        ENDDO 
       ENDIF       
c- generation of random number 1->NRK
       IF (N.EQ.6.OR.N.GT.7) THEN
 1       NRK=DINT(PYR(0)*NSUMEV)+1
        IF (NRK.EQ.(NSUMEV+1)) GOTO 1
        DO I=1,KMAX
         IF (NRK.LE.KEVES(I+1).AND.NRK.GT.KEVES(I)) THEN
          K=I
         ELSE 
          K=0 
         ENDIF
        ENDDO
       ELSE
        K=0
       ENDIF
c- number of couples and triples       
       IF (INDC.EQ.0) THEN
        N2=INT(N/2)-3*K
        N3=2*K
       ELSE
        N2=INT((N-3)/2)-3*K
        N3=2*K+1
       ENDIF
c- massive CEVENT(2,I)
       IF (N2.GE.1) THEN
        DO I=1,N2
        CEVENT(2,2*I-1)=2
        CEVENT(2,2*I)=1
        ENDDO
       ENDIF
       IF (N3.GE.1) THEN
        DO I=1,N3
        CEVENT(2,2*N2+3*I-2)=2
        CEVENT(2,2*N2+3*I-1)=2
        CEVENT(2,2*N2+3*I)=1
        ENDDO
       ENDIF 
c- fragmentation into 2 cells of 2*N2 and 3*N3 elements
       IF (N2.GT.0.AND.N3.GT.0) THEN 
c-
        I2=2*N2
        I3=3*N3
        K2=1
        K3=1
c-        WRITE(*,*)'I2=',I2,'I3=',I3,'K2=',K2,'K3=',K3
c- J - number to put into one of cells
        J=1
 2      I23=I2+I3
        PROB=DBLE(I2)/DBLE(I23)
        IF (PYR(0).LE.PROB) THEN
         CEVENT(1,K2)=J
         K2=K2+1  
         I2=I2-1
        ELSE
         CEVENT(1,2*N2+K3)=J
         K3=K3+1 
         I3=I3-1
        ENDIF
c-        WRITE(*,*)'---------------'        
c-        WRITE(*,*)'PROB=',PROB,'J=',J
c-        WRITE(*,*)'I2=',I2,'I3=',I3,'K2=',K2,'K3=',K3        
        J=J+1
        IF (I2.GT.0.AND.I3.GT.0) GOTO 2
        IF (I2.EQ.0.AND.I3.GT.0) THEN
         DO I=K3,3*N3
          CEVENT(1,2*N2+I)=J
          J=J+1
         ENDDO
        ENDIF
        IF (I2.GT.0.AND.I3.EQ.0) THEN
         DO I=K2,2*N2
          CEVENT(1,I)=J
          J=J+1
         ENDDO
        ENDIF        
c- numbers are distributed in CELL2(2*N2 elements)
c- and CELL3(3*N3) elements        
       ENDIF
  
c- distribution of numbers between couples in CELL2
       IF (N2.GT.1)THEN
        DO I=1,2*N2
         CAUX(I)=CEVENT(1,I)
        ENDDO    
        DO I=0,N2-2
c- generate 2 numbers 1 <= II < JJ <= 2*(N2-I)        
          J=2*(N2-I)         
 3        II=INT(PYR(0)*(J-1))+1
          IF (II.EQ.J) GOTO 3 
          J=J+1   
 4        JJ=INT(PYR(0)*(J-II-1))+II+1
          IF (JJ.EQ.J) GOTO 4
c- take these 2 numbers from CAUX and put to CEVENT
         CEVENT(1,2*(I+1)-1)=CAUX(II)
         CEVENT(1,2*(I+1))=CAUX(JJ)
c- set to 0 elements, that we have taken
         CAUX(II)=0
         CAUX(JJ)=0
c- reformat CAUX
         IC=1
         DO L=1,2*(N2-I)
          IF (CAUX(L).GT.0) THEN
           CAUX2(IC)=CAUX(L)
           IC=IC+1
          ENDIF
         ENDDO
         DO L=1,2*(N2-I-1)
          CAUX(L)=CAUX2(L)
         ENDDO         
        ENDDO
c- last 2 elements        
        CEVENT(1,2*N2-1)=CAUX(1)
        CEVENT(1,2*N2)=CAUX(2)
       ENDIF

c- distribution of numbers between triples in CELL3
       IF (N3.GT.1) THEN
        DO I=1,3*N3
         CAUX(I)=CEVENT(1,2*N2+I)
        ENDDO    

        DO I=0,N3-2
c- generate 3 numbers 1 <= II < JJ < LL<= 3*(N3-I)        
          J=3*(N3-I)-1         
 5        II=INT(PYR(0)*(J-1))+1
          IF (II.EQ.J) GOTO 5 
          J=J+1   
 6        JJ=INT(PYR(0)*(J-II-1))+II+1
          IF (JJ.EQ.J) GOTO 6
          J=J+1 
 7        LL=INT(PYR(0)*(J-JJ-1))+JJ+1         
c- take these 2 numbers from CAUX and put to CEVENT
         CEVENT(1,2*N2+3*(I+1)-2)=CAUX(II)
         CEVENT(1,2*N2+3*(I+1)-1)=CAUX(JJ)
         CEVENT(1,2*N2+3*(I+1))=CAUX(LL)
c- set to 0 elements, that we have taken
         CAUX(II)=0
         CAUX(JJ)=0
         CAUX(LL)=0
        
c- reformat CAUX
         IC=1
         DO L=1,3*(N3-I)
          IF (CAUX(L).GT.0) THEN
           CAUX2(IC)=CAUX(L)
           IC=IC+1
          ENDIF
         ENDDO
         DO L=1,3*(N3-I-1)
          CAUX(L)=CAUX2(L)
         ENDDO                
        ENDDO
c- last 3 elements        
        CEVENT(1,N-2)=CAUX(1)
        CEVENT(1,N-1)=CAUX(2)
        CEVENT(1,N)=CAUX(3)
       ENDIF    
       
       RETURN
       END
         
c------------cycles for combinatorics--------
c--- general cycle from NA to NB elements of massive
c--- of N elements, 1<=NA<NB<=N
       SUBROUTINE CYCLGEN(N,CEVENT1,NA,NB,CEVENT2)

       IMPLICIT NONE
       
       INTEGER N,CEVENT1(2,N),CEVENT2(2,N),NA,NB
       INTEGER I,CSAVE
       
       DO I=1,N
       CEVENT2(1,I)=CEVENT1(1,I)
       CEVENT2(2,I)=CEVENT1(2,I)
       ENDDO       
       
       CSAVE=CEVENT2(1,NA)
       DO I=NA,NB-1
        CEVENT2(1,I)=CEVENT2(1,I+1)        
       ENDDO
       CEVENT2(1,NB)=CSAVE
       
       RETURN
       END

C-============================================================
C-= 2 gluon cascades from colliding gluons in C.M.X. =========
C-============================================================
       SUBROUTINE SICASCAD2(MC,N1,PG1,P1,N2,PG2,P2,MX,NFAIL)
       
       IMPLICIT NONE
c- number of gluons in both cascads
c       INTEGER I,J,N1,N2,IAU,NFAIL
       INTEGER I,N1,N2,IAU,NFAIL
c- jj mass, {X+jj+Y} mass,momenta of cascad gluons,
c- momenta of final colliding hard gluons
       DOUBLE PRECISION MC,MX,PG1(5,500),PG2(5,500),
     & P1(5),P2(5)
c- P1(5)=DSQRT(ABS(P1(4)**2-P1(3)**2-P1(2)**2-P1(1)**2))
c- P2(5)=DSQRT(ABS(P2(4)**2-P2(3)**2-P2(2)**2-P2(1)**2))
c- because virtuality is negative!
c- variables for any cascad
       DOUBLE PRECISION MU,XG1(500),PTG1(500),FIG1(500),FX1,
     & FPT1,FFI1,XG2(500),PTG2(500),FIG2(500),FX2,FPT2,FFI2
c- parameter of Rj**2=(MC/MX)**2>RJMIN**2, 
c- RJMIN2=1/RJMIN**2 restriction
       DOUBLE PRECISION RJMIN2
c- aux.
       DOUBLE PRECISION A1,A2,AQ,ASRP,ASRM,AC,AB,KORP,KORM,
     & AUX1,AUX2
       
       MU=0.5D0*MC
       RJMIN2=100.D0
c-       WRITE(*,*)'point cascad2 1'
       IAU=0
       NFAIL=0
 1     CONTINUE       
       CALL SICASCAD(MU,N1,XG1,PTG1,FIG1,FX1,FPT1,FFI1)
c-       WRITE(*,*)'point cascad2 1.5'
 2     CALL SICASCAD(MU,N2,XG2,PTG2,FIG2,FX2,FPT2,FFI2)
c-       WRITE(*,*)'MU=',MU,' => attempt N ',IAU,': N1=',N1,' N2=',N2 
c- singlet restriction
c-       IF (N1.EQ.0.AND.N2.EQ.0.OR.N1.EQ.0.AND.N2.EQ.1.OR.
c-     & N1.EQ.1.AND.N2.EQ.0) THEN
c-        WRITE(*,*)'FAIL'
c-       ENDIF 
       IAU=IAU+1
       IF (IAU.GT.3000) THEN
        NFAIL=1
       ELSE
        NFAIL=0
       ENDIF 
       IF (IAU.GT.3000) GOTO 3
       IF (N1.EQ.0.AND.N2.EQ.0) GOTO 1
       IF (N1.EQ.0.AND.N2.EQ.1.OR.N1.EQ.1.AND.N2.EQ.0) GOTO 2
       
c-         WRITE(*,*)'point cascad2 2'
c-       WRITE(*,*)IAU,' STEPS to generate 2 cascads '
c-       WRITE(*,*)'X1=',FX1,' X2=',FX2
c-       WRITE(*,*)'PT1=',FPT1,' PT2=',FPT2
c- full procedure of MX extraction
       A1=0.D0
       A2=0.D0
       AQ=0.D0
      IF (N1.GT.0) THEN  
       DO I=1,N1
       A1=A1+PTG1(I)**2/XG1(I) 
       ENDDO
      ENDIF
      IF (N2.GT.0) THEN
       DO I=1,N2
       A2=A2+PTG2(I)**2/XG2(I) 
       ENDDO
      ENDIF 
      IF (N1.EQ.0) THEN
       FPT1=0.D0
       FFI1=0.D0
       FX1=1.D0
      ENDIF
      IF (N2.EQ.0) THEN
       FPT2=0.D0
       FFI2=0.D0
       FX2=1.D0
      ENDIF
       A1=A1/MU**2
       A2=A2/MU**2
       AQ=(FPT1**2+FPT2**2+2*FPT1*FPT2*DCOS(FFI1-FFI2))/MU**2
       AB=(1.D0+0.25D0*((FX2*A2+FX1*A1)+AQ))
c- discriminant of the equation
       AC=AB*AB-A1*A2*FX1*FX2/4.D0
c- roots (MX/MC)**2=KORP OR KORM?
       KORM=0.5D0*(AB-DSQRT(AC))/(FX1*FX2)
       KORP=0.5D0*(AB+DSQRT(AC))/(FX1*FX2)
c- if KORM is possible? Comparison. P1(4)+P2(4)>=MC! gives
       ASRM=KORM-((1.D0+DSQRT(1.D0+0.25D0*(A1+A2)*(FX1+FX2)))
     &  /(FX1+FX2))**2
       ASRP=KORP-((1.D0+DSQRT(1.D0+0.25D0*(A1+A2)*(FX1+FX2)))
     &  /(FX1+FX2))**2
c- (MX/MC)**2>AUX1 AND >AUX2: from P1(4)>0,P2(4)>0.
       AUX1=0.25D0*A1/FX1
       AUX2=0.25D0*A2/FX2
c-       WRITE(*,*)'AUX OUTPUT from CASCAD2============'
c-       WRITE(*,*)'A1=',A1,'  A2=',A2
c-       WRITE(*,*)'AQ=',AQ,'  AB=',AB,' AC=',AC
c-       WRITE(*,*)'KORM=',KORM,'  KORP=',KORP
c-       WRITE(*,*)'ASRM=',ASRM,' ASRP=',ASRP
c-       WRITE(*,*)'AUX1=',AUX1,' AUX2=',AUX2
c- verification discriminant>=0, (MX/MC)**2<RJMIN2 (~10-100)
c- and P1(4)+P2(4)>=MC for KORP solution
c-       WRITE(*,*)'point 1'
       IF (AC.LT.0.D0.OR.KORM.GT.RJMIN2.OR.ASRP.LT.0.D0) GOTO 1
c-       WRITE(*,*)'point 2'
       IF (KORP.LE.AUX1.OR.KORP.LE.AUX2) GOTO 1 
c- verification of P1(4)+P2(4)>=MC,
c- P1(4)>0,P2(4)>0 for KORM solution
       IF (ASRM.LT.0.D0.OR.KORM.LE.AUX1.OR.KORM.LE.AUX2) THEN 
        KORM=0.D0
       ENDIF
c- determination of MX
c-        WRITE(*,*)'point 3' 
        IF (KORP.GT.RJMIN2.AND.KORM.EQ.0.D0) GOTO 1
c-        WRITE(*,*)'point 4'
c- choose KORP anyway if KORP.LE.RJMIN2
c-        IF (KORP.LE.RJMIN2.AND.KORM.EQ.0.D0) THEN
c-         MX=MC*DSQRT(KORP)
c-        ENDIF
c- in this case both roots satisfy relations!
c-        IF (KORP.LE.RJMIN2.AND.KORM.GT.0.D0) THEN
c-        MX=MC*DSQRT(KORP)
c-        ENDIF
c-         WRITE(*,*)'point cascad2 3'
        IF (KORP.LE.RJMIN2) THEN
         MX=MC*DSQRT(KORP)
        ENDIF
c- only in this case KORM!
        IF (KORP.GT.RJMIN2.AND.KORM.GT.0.D0) THEN
         MX=MC*DSQRT(KORM)
        ENDIF    
c-       WRITE(*,*)'MX=',MX,' MC=',MC
c-       WRITE(*,*)'AUX OUTPUT from CASCAD2============'
c- determination of all momenta in X C.M. frame
c- hard gluons
      IF (N1.GT.0) THEN 
       P1(1)=FPT1*DCOS(FFI1) 
       P1(2)=FPT1*DSIN(FFI1)
       P1(3)=0.5D0*MX*(FX1+A1*0.25D0*(MC/MX)**2)
       P1(4)=0.5D0*MX*(FX1-A1*0.25D0*(MC/MX)**2)
       P1(5)=DSQRT(DABS(P1(4)**2-P1(3)**2-P1(2)**2-P1(1)**2))
      ENDIF
      IF (N2.GT.0) THEN
       P2(1)=FPT2*DCOS(FFI2) 
       P2(2)=FPT2*DSIN(FFI2)
       P2(3)=-0.5D0*MX*(FX2+A2*0.25D0*(MC/MX)**2)
       P2(4)=0.5D0*MX*(FX2-A2*0.25D0*(MC/MX)**2)
       P2(5)=DSQRT(DABS(P2(4)**2-P2(3)**2-P2(2)**2-P2(1)**2))       
      ENDIF 
      IF (N1.EQ.0) THEN
       P1(1)=0.D0 
       P1(2)=0.D0
       P1(3)=0.5D0*MX
       P1(4)=0.5D0*MX
       P1(5)=DSQRT(DABS(P1(4)**2-P1(3)**2-P1(2)**2-P1(1)**2))      
      ENDIF
      IF (N2.EQ.0) THEN
       P2(1)=0.D0 
       P2(2)=0.D0
       P2(3)=-0.5D0*MX
       P2(4)=0.5D0*MX
       P2(5)=DSQRT(DABS(P2(4)**2-P2(3)**2-P2(2)**2-P2(1)**2))       
      ENDIF
c- cascade gluons
c-       WRITE(*,*)' direct output from cascad 2---------'       
c-       WRITE(*,*)'N1=',N1,' N2=',N2,' MX=',MX
      IF (N1.GT.0) THEN 
       DO I=1,N1
        PG1(1,I)=PTG1(I)*DCOS(FIG1(I))
        PG1(2,I)=PTG1(I)*DSIN(FIG1(I))
        PG1(3,I)=0.5D0*MX*XG1(I)*(1.D0-(PTG1(I)/MX/XG1(I))**2)
        PG1(4,I)=0.5D0*MX*XG1(I)*(1.D0+(PTG1(I)/MX/XG1(I))**2)
        PG1(5,I)=0.D0
c-        WRITE(*,*)'XG1(',I,')=',XG1(I),' PTG1(',I,')=',PTG1(I)
c-        WRITE(*,*)I,'. mg**2=',(PG1(4,I)**2-PG1(3,I)**2-
c-     &  PG1(2,I)**2-PG1(1,I)**2)
c-        WRITE(*,*)'PG1(',I,'): ',PG1(4,I),' ',PG1(1,I),' ',
c-     &  PG1(2,I),' ',PG1(3,I)   
       ENDDO       
      ENDIF
      IF (N2.GT.0) THEN
       DO I=1,N2
        PG2(1,I)=PTG2(I)*DCOS(FIG2(I))
        PG2(2,I)=PTG2(I)*DSIN(FIG2(I))
        PG2(3,I)=-0.5D0*MX*XG2(I)*(1.D0-(PTG2(I)/MX/XG2(I))**2)
        PG2(4,I)=0.5D0*MX*XG2(I)*(1.D0+(PTG2(I)/MX/XG2(I))**2)
        PG2(5,I)=0.D0
c-        WRITE(*,*)'XG2(',I,')=',XG2(I),' PTG2(',I,')=',PTG2(I)
c-        WRITE(*,*)I+N1,'. mg**2=',(PG2(4,I)**2-PG2(3,I)**2-
c-     &  PG2(2,I)**2-PG2(1,I)**2)
c-        WRITE(*,*)'PG2(',I,'): ',PG2(4,I),' ',PG2(1,I),' ',
c-     &  PG2(2,I),' ',PG2(3,I)     
       ENDDO       
      ENDIF
c-        WRITE(*,*)'final hard gluons'
c-        WRITE(*,*)'Pf1: ',P1(4),' ',P1(1),' ',
c-     &  P1(2),' ',P1(3),' ',P1(5)
c-        WRITE(*,*)'Pf2: ',P2(4),' ',P2(1),' ',
c-     &  P2(2),' ',P2(3),' ',P2(5)
c-        AUX1=((P1(4)+P2(4))**2-
c-     & (P1(3)+P2(3))**2-(P1(2)+P2(2))**2-(P1(1)+P2(1))**2)  
c-        WRITE(*,*)'central mass**2 => ',AUX1  
c-       WRITE(*,*)' direct output from cascad 2---------'      
      
 3     RETURN
       END

C-============================================================
C-= 1 gluon cascade from colliding gluon in C.M.X. ===========
C-============================================================
       SUBROUTINE SICASCAD(MU,NG,XG,PTG,FIG,FX,FPT,FFI)
       
       IMPLICIT NONE
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2     
C-number of generated gluons and turn on/off parameter       
       INTEGER NG,NKON
C-generated kinematical variables for hard gluons
       DOUBLE PRECISION XGH(500),PTGH(500),FIGH(500)
C-for soft gluons
       DOUBLE PRECISION XG(500),PTG(500),FIG(500)
C-generated final gluon variables
       DOUBLE PRECISION MU,FX,FPT,FFI
C- aux vars. -----------------       
       INTEGER I,ISAVE
       DOUBLE PRECISION PYR,AUX1,AUX2,AUX3,AUX4 

       DO I=1,500
       XG(I)=0.D0
       XGH(I)=0.D0
       PTG(I)=0.D0
       PTGH(I)=0.D0
       FIG(I)=0.D0
       FIGH(I)=0.D0
       ENDDO

 22    NKON=0
       NG=0
       I=1

c- begin iterations
c-       WRITE(*,*)'begin iterations'
c-      WRITE(*,*)'point cascad 1'
      CALL GENERFINAL(MU,FX,FPT,FFI,NKON)
c-      WRITE(*,*)'point cascad 1.5'
c-      
c-      WRITE(*,*)'X_f=',FX,' PT_f=',FPT,' NKON=',NKON
c-
       IF (NKON.EQ.0) THEN 
        PTGH(1)=FPT
        XGH(1)=FX
        FIGH(1)=FFI
       ENDIF
 
c- here XG,PTG -> for hard virtual gluons
 1     CONTINUE
c-       WRITE(*,*)'label 1'
       ISAVE=I 
       I=I+1 
c-       WRITE(*,*)'ISAVE=',ISAVE,' I=',I
c- PYTHIA restriction -----       
c- in this case compensate momentum to conserve 4-mom.
       IF (I.GT.80) THEN
       NKON=1
       ENDIF
 2     CONTINUE
c-       WRITE(*,*)'label 2'
       IF (NKON.EQ.0) THEN        
c- generation of next gluon
c-       WRITE(*,*)'NKON=',NKON
c-       WRITE(*,*)'call genercptx -----------'
       CALL GENERCPTX(MU,XGH(ISAVE),PTGH(ISAVE),PTGH(I),XGH(I),NKON)
c-       WRITE(*,*)'XGH(',ISAVE,')=',XGH(ISAVE)
c-       WRITE(*,*)'PTGH(',ISAVE,')=',PTGH(ISAVE)
c-       WRITE(*,*)'PTGH(',I,')=',PTGH(I)
c-       WRITE(*,*)'XGH(',I,')=',XGH(I)
c-       WRITE(*,*)'NKON=',NKON
       ENDIF
       IF (NKON.EQ.0) THEN
       FIGH(I)=2*PI*PYR(0) 
       ENDIF
c- check point for current gluon
c- rap. gap. constr.
c-       WRITE(*,*)'NKON=',NKON  
       IF (NKON.EQ.0) THEN
        AUX1=DSQRT(PTGH(I)**2+PTGH(ISAVE)**2-
     &  2.D0*PTGH(I)*PTGH(ISAVE)*DCOS(FIGH(I)-FIGH(ISAVE)))
        AUX1=AUX1*DCOSH(0.5D0*ETASIMAX)/MU
        AUX1=AUX1/(XGH(I)-XGH(ISAVE))
       ELSE
        AUX1=PTGH(ISAVE)*DCOSH(0.5D0*ETASIMAX)/MU
        AUX1=AUX1/(1.D0-XGH(ISAVE))
       ENDIF
c-       WRITE(*,*)'rap.gap. check=>',AUX1 
       IF (NKON.EQ.0.AND.AUX1.LT.1.D0) GOTO 2
       IF (NKON.EQ.0.AND.AUX1.GE.1.D0) GOTO 1
c-       WRITE(*,*)'point cascad 2' 
c-       
c- number of generated gluons
       NG=ISAVE-1
c- end of iterations
c- initial hard gluons and conservation lows  
       PTGH(NG+1)=0.D0
       XGH(NG+1)=1.D0   
       FIGH(NG+1)=0.D0 
c- check point for sum of all gluons
c- and
c- transform hard virtual to soft momenta
       IF (NG.GT.0) THEN
        AUX3=0.D0
        DO I=1,NG
         XG(I)=XGH(I+1)-XGH(I)
         PTG(I)=DSQRT(PTGH(I+1)**2+PTGH(I)**2-2*PTGH(I+1)*PTGH(I)*
     &   DCOS(FIGH(I+1)-FIGH(I)))
         AUX1=PTGH(I+1)*DCOS(FIGH(I+1))-PTGH(I)*DCOS(FIGH(I))
         AUX2=PTGH(I+1)*DSIN(FIGH(I+1))-PTGH(I)*DSIN(FIGH(I))
          IF (AUX2.GE.0.D0) THEN
           IF (AUX1.EQ.0.D0) THEN
            FIG(I)=0.5D0*PI
           ELSE
            FIG(I)=DACOS(AUX1/PTG(I))
           ENDIF
          ELSE
           IF (AUX1.EQ.0.D0) THEN
            FIG(I)=1.5D0*PI
           ELSE
            FIG(I)=-DACOS(AUX1/PTG(I))
           ENDIF       
          ENDIF
          AUX3=AUX3+PTG(I)*PTG(I)/XG(I)
        ENDDO
        AUX3=AUX3*FX+FPT*FPT
        AUX4=MU*MU
       ELSE
        AUX3=0.D0
        AUX4=1.D0
       ENDIF 
c- check, that virtuality of final hard gluon > or =(-MU**2)
       IF (AUX3.GT.AUX4) GOTO 22
       
       RETURN
       END

C-============================================================
C-= generation of final X,PT,FI for hard colliding gluon  ====
C-============================================================
       SUBROUTINE GENERFINAL(MU,FX,FPT,FFI,NKON)
       
       IMPLICIT NONE
c- functions and variables
c      INTEGER N,NKON,IAU,IBU
      INTEGER NKON,IAU,IBU
      DOUBLE PRECISION CFUNX,CFUNPT,MU,FX,RX,FPT,RPT,FFI,RAT 
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW  
c- rnd. generator     
      DOUBLE PRECISION PYR
c- commons for cascad generation      
      DOUBLE PRECISION FGGQ0,FGGQS
      COMMON/EDDEFGG/FGGQ0,FGGQS
c- aux. vars.
      DOUBLE PRECISION FGGQ0P,QMIN 
 
       NKON=0 
       IAU=0
       IBU=0
 1     CALL CFUNXUINV(MU,PYR(0),RX) 
       RAT=CFUNX(0,MU,RX)/CFUNX(2,MU,RX)
       IAU=IAU+1
       IF (RAT.EQ.0.D0) THEN
       IBU=IBU+1
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) THEN
        NKON=1
        FX=1.D0
        FPT=0.D0
        FFI=0.D0
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) GOTO 3
       IF (RAT.LE.PYR(0)) GOTO 1
       FX=RX
       IF (FX.LT.0.25D0) THEN
        FGGQ0P=(0.5D0+DSQRT(0.25D0-FX))/(0.5D0-DSQRT(0.25D0-FX))
       ELSE
        FGGQ0P=DSQRT(FX)/(1.D0-DSQRT(FX))
       ENDIF
       IF (FGGQ0P.GT.FGGQS) THEN
        QMIN=FGGQ0P
       ELSE
        QMIN=FGGQS
       ENDIF
       
       IAU=0
       IBU=0
 2     CALL CFUNPTUINV(FX,MU,PYR(0),RPT)
       RAT=CFUNPT(0,FX,RPT)/CFUNPT(2,FX,RPT)
       IAU=IAU+1
       IF (RAT.EQ.0.D0) THEN
       IBU=IBU+1
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) THEN
        NKON=1
        FX=1.D0
        FPT=0.D0
        FFI=0.D0
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) GOTO 3
       IF (RAT.LE.PYR(0)) GOTO 2
       FPT=RPT
       IF (FPT.LT.QMIN) THEN
       NKON=1
       ENDIF
      
       FFI=2.D0*PI*PYR(0)
       
 3      RETURN
       END

C-============================================================
C-= generation step backward X,PT of previous hard gluon =====
C-============================================================
       SUBROUTINE GENERCPTX(MU,XFIX,PTFIX,GPT,GX,NKON)
       
       IMPLICIT NONE
       INTEGER IAU,IBU
c- functions and variables
c      INTEGER N,NKON
      INTEGER NKON
c      DOUBLE PRECISION CFUNPT,CFUNZ,MU,XFIX,PTFIX,GPT,GX,
c     & RPT,RX,RZ,RAT 
      DOUBLE PRECISION CFUNPT,CFUNZ,MU,XFIX,PTFIX,GPT,GX,
     & RPT,RZ,RAT 
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW  
c- rnd. generator     
      DOUBLE PRECISION PYR
c- commons for cascad generation      
      DOUBLE PRECISION FGGQ0,FGGQS
      COMMON/EDDEFGG/FGGQ0,FGGQS
c- aux. vars.
      DOUBLE PRECISION FGGQ0P,QMIN,FZMAX,FZMIN,AUX1 

       NKON=0 
c- check pt-lower limit
       IF (XFIX.LT.0.25D0) THEN
        FGGQ0P=(0.5D0+DSQRT(0.25D0-XFIX))/(0.5D0-DSQRT(0.25D0-XFIX))
        FGGQ0P=FGGQ0P*FGGQ0
       ELSE
        FGGQ0P=DSQRT(XFIX)/(1.D0-DSQRT(XFIX))
        FGGQ0P=FGGQ0P*FGGQ0
       ENDIF
       IF (FGGQ0P.GT.FGGQS) THEN
        QMIN=FGGQ0P
       ELSE
        QMIN=FGGQS
       ENDIF
       IF (QMIN.GE.PTFIX) THEN
        NKON=1
        GPT=0.D0
        GX=1.D0
       ENDIF
       IF (NKON.EQ.1) GOTO 3
c--- pt generation
       IAU=0
       IBU=0
 1     CALL CFUNPTUINV(XFIX,PTFIX,PYR(0),RPT)
       RAT=CFUNPT(0,XFIX,RPT)/CFUNPT(2,XFIX,RPT)
       IAU=IAU+1
       IF (RAT.EQ.0.D0) THEN
       IBU=IBU+1
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) THEN
c-        WRITE(*,*)'!!!!!!! exit from pt generation !!!!!!!'
c-        WRITE(*,*)' XFIX=',XFIX,' PTFIX=',PTFIX
c-        WRITE(*,*)'++++++++++++++++++++++++++++'
        NKON=1
        GPT=0.D0
        GX=1.D0
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) GOTO 3
       IF (RAT.LE.PYR(0)) GOTO 1
c- 
       IF (RPT.LT.QMIN) THEN
        NKON=1
       ENDIF
       GPT=RPT 
c--- z-generation 
       IAU=0
       IBU=0
       IF (NKON.EQ.0) THEN
 2      CALL CFUNZUINV(MU,XFIX,GPT,PYR(0),RZ)
        RAT=CFUNZ(0,MU,XFIX,GPT,RZ)/CFUNZ(2,MU,XFIX,GPT,RZ) 
        IAU=IAU+1 
       IF (RAT.EQ.0.D0) THEN
       IBU=IBU+1
       ENDIF
       IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) THEN
c-         WRITE(*,*)'!!!!!!! exit from pt generation !!!!!!!'
c-         WRITE(*,*)' XFIX=',XFIX,' PTFIX=',PTFIX
c-         WRITE(*,*)'++++++++++++++++++++++++++++'
         NKON=1
         GPT=0.D0
         GX=1.D0
        ENDIF
        IF (IBU.GT.10.AND.IAU.GT.10.OR.IAU.GT.1000) GOTO 4
        IF (RAT.LE.PYR(0)) GOTO 2
c- check point for exit
        GX=XFIX/RZ
        AUX1=FGGQ0/(GPT+FGGQ0)
        FZMIN=XFIX*(GPT+FGGQ0)/GPT
        IF (AUX1.GT.FZMIN) THEN
         FZMIN=AUX1
        ENDIF
        AUX1=GPT/(GPT+FGGQ0)
        FZMAX=XFIX*(GPT+FGGQ0)/FGGQ0
        IF (AUX1.LT.FZMAX) THEN
         FZMAX=AUX1
        ENDIF 
c-       
        IF (RZ.LT.FZMIN.OR.RZ.GT.FZMAX) THEN
         NKON=1
        ENDIF             
 4     CONTINUE       
       ENDIF
       
 3     RETURN
       END

C- generation of upper functions -----------------------------
C-============================================================
C-= inverse of integrated upper function for x**(2*DP(3))* ===
C-= *(fM(x,MU)-fM(x,ptmin(x)))                  ==============
C-= limits for x are inserted ================================
C-============================================================
       SUBROUTINE CFUNXUINV(MU,RF,RX)
       
       IMPLICIT NONE
       
c- RF=PYR(0) ----------------------
       DOUBLE PRECISION MU,RF,RX,R
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
       INTEGER NAPR,NFI
       DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
       COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI      
c- commons for cascad generation      
       DOUBLE PRECISION FGGQ0,FGGQS
       COMMON/EDDEFGG/FGGQ0,FGGQS
c- aux. vars.
c       DOUBLE PRECISION LIM1,LIM2,YMU,FMIN,FMAX
       DOUBLE PRECISION LIM1,LIM2,FMIN,FMAX

c- upper and lower limits of generation
       LIM1=FGGQ0/(MU+FGGQ0)
       LIM2=(1.D0-LIM1)*(1.D0-LIM1)
c-       YMU=2.D0*DLOG(MU/LAMQCD)
c- R=RF*(Ifun(LIM2)-Ifun(LIM1))+Ifun(LIM1)
c- Ifun - direct integral,
c- inverse function of the Ifun
       FMIN=DLOG((1.D0+DSQRT(1.D0-LIM1))/(1.D0-DSQRT(1.D0-LIM1)))
       FMAX=DLOG((1.D0+DSQRT(1.D0-LIM2))/(1.D0-DSQRT(1.D0-LIM2)))
       R=RF*(FMAX-FMIN)+FMIN              
c- temporary flat distribution       
       RX=1.D0-DTANH(0.5D0*R)**2 
       
       RETURN
       END

C-============================================================
C-= inverse of integrated upper function for =================
C-= dfM(x,pt)/dLog(pt**2/LAMQCD**2) ==========================
C-============================================================       
       SUBROUTINE CFUNPTUINV(X,PT,RF,RPT)
       
       IMPLICIT NONE
c- RF=PYR(0) ----------------------
       DOUBLE PRECISION RF,X,PT,RPT,RY,R
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW         
c- commons for cascad generation      
       DOUBLE PRECISION FGGQ0,FGGQS
       COMMON/EDDEFGG/FGGQ0,FGGQS
c- aux. vars.
       DOUBLE PRECISION LIM1,LIM2,LIM1Y,LIM2Y,FMIN,FMAX

c- upper and lower limits of generation
       LIM1=FGGQ0
       LIM2=PT
       LIM1Y=2.D0*DLOG(LIM1/LAMQCD)
       LIM2Y=2.D0*DLOG(LIM2/LAMQCD)
       FMIN=LIM1Y**4.D0
       FMAX=LIM2Y**4.D0
c- R=RF*(Ifun(LIM2)-Ifun(LIM1))
c- Ifun - direct integral,
c- inverse function of the Ifun
       R=RF*(FMAX-FMIN)+FMIN
c- temporary flat distribution        
       RY=R**0.25D0 
       RPT=LAMQCD*DEXP(0.5D0*RY)
       
       RETURN
       END
       
C-============================================================
C-= inverse of integrated upper function for =================
C-= fM(z,pt)*Pgg(x/z)/z ======================================
C-============================================================         
       SUBROUTINE CFUNZUINV(MU,X,PT,RF,RZ)
       
       IMPLICIT NONE
c- RF=PYR(0) ----------------------
       DOUBLE PRECISION RF,MU,X,PT,RZ,R  
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW 
c- commons for cascad generation      
       DOUBLE PRECISION FGGQ0,FGGQS
       COMMON/EDDEFGG/FGGQ0,FGGQS
c- aux. vars.
c       DOUBLE PRECISION LIM1,LIM2,FMIN,FMAX,YPT,DFMAX,AUX1,AUX2
       DOUBLE PRECISION LIM1,LIM2,DFMAX,AUX1,AUX2

c- upper and lower limits of generation
       LIM1=X*(MU+FGGQ0)/MU
       LIM2=MU/(MU+FGGQ0)
c- R=RF*(Ifun(LIM2)-Ifun(LIM1))
c- Ifun - direct integral,
c- inverse function of the Ifun
       IF (LIM1.LT.0.67D0.AND.X.LT.0.67D0) THEN
         AUX1=DLOG(0.67D0-X)-DLOG(LIM1-X)
         DFMAX=DLOG(DSQRT(1.D0-X)-DSQRT(1.D0-LIM2))
         DFMAX=DFMAX-DLOG(DSQRT(1.D0-X)+DSQRT(1.D0-LIM2))
         DFMAX=DFMAX-DLOG(DSQRT(1.D0-X)-DSQRT(1.D0-0.67D0))
         DFMAX=DFMAX+DLOG(DSQRT(1.D0-X)+DSQRT(1.D0-0.67D0))
         DFMAX=DFMAX/DSQRT(1.D0-X)
         DFMAX=DFMAX+AUX1
         R=RF*DFMAX+DLOG(LIM1-X)
         IF (R.LT.AUX1) THEN
          RZ=X+DEXP(R)*(LIM1-X)           
         ELSE
          AUX2=(R-AUX1)*DSQRT(1.D0-X)
          AUX2=AUX2+DLOG(DSQRT(1.D0-X)-DSQRT(1.D0-0.67D0))
          AUX2=AUX2-DLOG(DSQRT(1.D0-X)+DSQRT(1.D0-0.67D0))
          RZ=1.D0-(1.D0-X)*DTANH(-0.5D0*AUX2)**2.D0
         ENDIF         
       ELSE
         AUX1=DLOG(DSQRT(1.D0-X)-DSQRT(1.D0-LIM1))
         AUX1=AUX1-DLOG(DSQRT(1.D0-X)+DSQRT(1.D0-LIM1))
         AUX1=AUX1/DSQRT(1.D0-X)
         DFMAX=DLOG(DSQRT(1.D0-X)-DSQRT(1.D0-LIM2))
         DFMAX=DFMAX-DLOG(DSQRT(1.D0-X)+DSQRT(1.D0-LIM2))
         DFMAX=DFMAX/DSQRT(1.D0-X)-AUX1
         R=RF*DFMAX+AUX1
         RZ=1.D0-(1.D0-X)*DTANH(-0.5D0*R*DSQRT(1.D0-X))**2.D0
       ENDIF
       
       RETURN
       END

c- functions for the above cascad generators -----------------
C-============================================================
C-= interpol.and upper function for x**(2*DP(3))* ============
C-= *(fM(x,YMU)-fM(x,Yptmin(x))), Y=DLOG(pt**2/LAMQCD**2)  ===
C-============================================================       
       FUNCTION CFUNX(N,MU,FX)

       IMPLICIT NONE
       
       INTEGER N
       DOUBLE PRECISION CFUNX,MU,FX
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW  
c-- tables for fM used in x**(2*DP(3))*(fM(x,MU)-fM(x,minpt(x))) 
c- tables for fM(x,pt)
       DOUBLE PRECISION FT4,FMT4,DX4,DY4,X04,Y04
       COMMON/EDDETAB4/ FT4(10201),FMT4(101,101),
     & DX4,DY4,X04,Y04
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
       INTEGER NAPR,NFI
       DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
       COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI      
c- commons for cascad generation      
       DOUBLE PRECISION FGGQ0,FGGQS
       COMMON/EDDEFGG/FGGQ0,FGGQS
c- aux. vars.
       DOUBLE PRECISION FGGQ0P,QMIN,AUX1,AUX2,YMU,YQMIN
      
c- check pt-lower limit
       IF (FX.LT.0.25D0) THEN
        FGGQ0P=(0.5D0+DSQRT(0.25D0-FX))/(0.5D0-DSQRT(0.25D0-FX))
       ELSE
        FGGQ0P=DSQRT(FX)/(1.D0-DSQRT(FX))
       ENDIF
       IF (FGGQ0P.GT.FGGQS) THEN
        QMIN=FGGQ0P
       ELSE
        QMIN=FGGQS
       ENDIF
       YMU=2.D0*DLOG(MU/LAMQCD)
       YQMIN=2.D0*DLOG(QMIN/LAMQCD)
       
       IF (N.LE.1) THEN        
        IF (YMU.GT.YQMIN) THEN
         CALL LINTERPOL2(FMT4,101,101,X04,Y04,DX4,DY4,FX,YMU,AUX1)
         CALL LINTERPOL2(FMT4,101,101,X04,Y04,DX4,DY4,FX,YQMIN,AUX2)
         IF (AUX1.LE.0.D0) THEN
          AUX1=0.D0
c-          WRITE(*,*)'AUX1<=0',' X=',FX,' YMU=',YMU,' YMIN=',YQMIN
         ENDIF
         IF (AUX2.LE.0.D0) THEN
          AUX2=0.D0
c-          WRITE(*,*)'AUX2<=0',' X=',FX,' YMU=',YMU,' YMIN=',YQMIN
         ENDIF         
        CFUNX=FX**(2.D0*DP(3))*(AUX1-AUX2)
         IF (AUX2.GT.AUX1) THEN
          CFUNX=0.D0
c-          WRITE(*,*)'AUX2>AUX1',' X=',FX,' YMU=',YMU,' YMIN=',YQMIN
         ENDIF
        ELSE
         CFUNX=0.D0
        ENDIF
       ELSE
c--- change to upper function!       
        CFUNX=0.285052D-02*YMU**2.05D0/(FX*DSQRT(1.D0-FX))
        CFUNX=CFUNX*1.02505D0
       ENDIF
       
       RETURN
       END
       
C-============================================================
C-= interp. and upper function for ===========================
C-= dfM(x,Y)/dY, Y=Log(pt**2/LAMQCD**2) =====================
C-============================================================  
       FUNCTION CFUNPT(N,X,PT)

       IMPLICIT NONE

       INTEGER N
       DOUBLE PRECISION CFUNPT,X,PT
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW  
c-- tables for dfM_g/g(x,Y)/dY, Y=DLOG(PT**2/LAMQCD**2) ---
       DOUBLE PRECISION DFT5,DFMT5,DX5,DY5,X05,Y05
       COMMON/EDDETAB5/ DFT5(10201),DFMT5(101,101),
     & DX5,DY5,X05,Y05    
c- aux. vars.
       DOUBLE PRECISION YPT 
       YPT=2.D0*DLOG(PT/LAMQCD)
       
       IF (N.LE.1) THEN        
        CALL LINTERPOL2(DFMT5,101,101,X05,Y05,DX5,DY5,X,YPT,CFUNPT)
        IF (CFUNPT.LT.0.D0) THEN
         CFUNPT=0.D0
        ENDIF
       ELSE
c--- change to upper function!       
        CFUNPT=0.50544D-04*YPT**3.D0/(X**1.5D0*(1.D0-X)**0.47D0)
       ENDIF
       
       RETURN
       END 

C-============================================================
C-= interp. and upper function for ===========================
C-= fM(z,pt)*Pgg(x/z)/z ======================================
C-============================================================         
       FUNCTION CFUNZ(N,MU,X,PT,Z)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION CFUNZ,X,PT,Z
c------ fundamental constants -----------------------------
       INTEGER NF,NC,NLOSW
       DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
       DOUBLE COMPLEX MNI,REI
       COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW  
c-- tables for fM used in Pgg(x/z)*fM(z,pt)/z 
c- tables for fM(x,pt)
       DOUBLE PRECISION FT4,FMT4,DX4,DY4,X04,Y04
       COMMON/EDDETAB4/ FT4(10201),FMT4(101,101),
     & DX4,DY4,X04,Y04
c- commons for cascad generation      
       DOUBLE PRECISION FGGQ0,FGGQS
       COMMON/EDDEFGG/FGGQ0,FGGQS     
c- aux. vars.
       DOUBLE PRECISION AUX1,AUX2,AUX3,YPT,MU,ZMAX
       DOUBLE PRECISION AUX4,AUX5
       
       AUX4=X04+100.D0*DX4
       AUX5=Y04+100.D0*DY4
       ZMAX=MU/(MU+FGGQ0)
       YPT=2.D0*DLOG(PT/LAMQCD)
       IF (N.LE.1) THEN
        IF (Z.GE.X04.AND.Z.LE.AUX4.AND.YPT.GE.Y04.AND.YPT.LE.AUX5)
     &  THEN
         CALL LINTERPOL2(FMT4,101,101,X04,Y04,DX4,DY4,Z,YPT,AUX1)
        ELSE
         AUX1=0.D0
        ENDIF
        IF (AUX1.LE.0.D0) THEN
         AUX1=0.D0
        ENDIF
        AUX2=X/Z
        AUX3=(1.D0-AUX2*(1-AUX2))**2/(AUX2*(1-AUX2))
        CFUNZ=AUX1*AUX3/Z
       ELSE
         CFUNZ=0.012726D0*YPT**1.65D0/X**1.3D0/(Z-X)
        IF (Z.GE.0.67D0) THEN        
         CFUNZ=CFUNZ*X**0.3D0*ZMAX/DSQRT(1.D0-Z)        
        ENDIF        
       ENDIF
       
       RETURN
       END
C-============================================================

C-++================== E N D  C A S C A D ====================

c--------TOTAL CROSS-SECTIONS FUNCTIONS AND SUBS. ------------
C-===========================================================
C-==== qt-loop integral with exclusive sudakov Isud0(MU)**2 ==
C-==== N=1 - parametrization, N>=2 - upper function ==========
C-==== N=0 - exact value (reserved, now the same as N=1) =====
C-============================================================
       FUNCTION ISUD02(N,MU)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION  ISUD02,MU
      
       IF (N.LT.2) THEN
         ISUD02=12.7244D0*DLOG(MU)**3.55D0
         ISUD02=ISUD02/(MU**1.118D0*(1.2D0+MU**1.23D0))
       ELSE
         ISUD02=5.0746D0/MU**0.6795D0
       ENDIF
       
       RETURN
       END
C-============================================================
C-============================================================
C- qt-loop integral with semi-incl. correction due to rad. ===
C- Isudeta2(MU,etamax)=Integral[Deltafgg*x**(2*DP(3)))]**2* ==
C- *Isud0(MU)**2 ============================================= 
C-==== N=1 - parametrization, N>=2 - upper function ==========
C-==== N=0 - exact value (reserved, now the same as N=1) =====
C-============================================================
C--- check+
       FUNCTION ISUDGEN2(N,MU)

       IMPLICIT NONE
       INTEGER N
c       DOUBLE PRECISION ISUD02,A1,AUX2,AUX3
       DOUBLE PRECISION ISUD02,A1
       DOUBLE PRECISION ISUDGEN2,MU
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c---- table for Intfgg**2 for Lum_sidde=Lum_excl*Intfgg**2 ---
      DOUBLE PRECISION LUM1,FLUM1,DX1,DY1,X01,Y01
      COMMON/EDDETAB1/ LUM1(480),FLUM1(30,16),
     & DX1,DY1,X01,Y01         

c---    CALL EDDEPUTDAT  ! call one time from the main program
       IF (N.LT.2) THEN
c-         IF (MU.LE.150.D0.AND.ETASIMAX.LE.30.D0) THEN  
c-      CALL LINTERPOL2(FLUM1,30,16,X01,Y01,DX1,DY1,MU,ETASIMAX,A1)
c-         ELSE
c-          IF (MU.GT.150.D0.AND.ETASIMAX.LE.30.D0) THEN
c-      CALL LINTERPOL2(FLUM1,30,16,X01,Y01,DX1,DY1,150.D0,ETASIMAX,A1)
c-           A1=A1*150.D0/MU
c-          ENDIF
c-          IF (MU.GT.150.D0.AND.ETASIMAX.GT.30.D0) THEN
c-      CALL LINTERPOL2(FLUM1,30,16,X01,Y01,DX1,DY1,150.D0,30.D0,A1)
c-           A1=A1*150.D0/MU
c-          ENDIF
c-          IF (MU.LE.150.D0.AND.ETASIMAX.GT.30.D0) THEN
c-      CALL LINTERPOL2(FLUM1,30,16,X01,Y01,DX1,DY1,MU,30.D0,A1)
c-          ENDIF
c-         ENDIF
c- 3-4% parametrization for new semi-inclusive correction (see math)
        IF (MU.LT.30.D0) THEN
         A1=0.015D0*(MU-1.D0)**1.42D0
        ELSE
         A1=0.221739D-01*MU**1.3D0
        ENDIF 
        ISUDGEN2=A1*ISUD02(1,MU)
       ELSE
        ISUDGEN2=PSIDD1*MU**PSIDD2*ISUD02(2,MU)
c-        IF (MU.GT.150.D0) THEN
c-         ISUDGEN2=PSIDD1*150.D0**PSIDD2*ISUD02(2,150.D0)*150.D0/MU
c-        ENDIF
       ENDIF
       
       RETURN
       END
C-============================================================

C-============================================================
C-====== soft surv. prob. (soft rescattering effects) ========
C-== N=0 exact(reserved, now like N=1), N=1 parametrization ==
C-== N>1 upper function ======================================
C-============================================================
C--- check+
       FUNCTION SOFTSURV(N,MGG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION SOFTSURV,MGG
       DOUBLE PRECISION TSLOPEBH,VAR  
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2      

       VAR=SQS/MGG 
       IF (N.LE.1) THEN
c-        SOFTSURV=2.43617D0/SQS**0.46232D0/MGG**0.066D0
c-        SOFTSURV=SOFTSURV-0.53D-03
c-        SOFTSURV=SOFTSURV-0.0538D0*(MGG/SQS)**4.262D0
c-        SOFTSURV=SOFTSURV-1.0538D-13*MGG**0.25935D0*SQS**2.35394D0
c-        SOFTSURV=SOFTSURV*1.03D0
        SOFTSURV=0.388D0/TSLOPEBH(VAR)**2
        SOFTSURV=SOFTSURV/MGG**0.23D0
        SOFTSURV=SOFTSURV*DEXP(-0.001055D0*(SQS**2-10000.D0)**0.4D0)
        SOFTSURV=SOFTSURV*(1.D0+3.04D0*(SQS/20000.D0)**2)
       ELSE
c-        SOFTSURV=1.03D0*2.43617D0/SQS**0.46232D0/MGG**0.066D0
          SOFTSURV=6.6678D0/(SQS**0.57D0*MGG**0.066D0)
       ENDIF
       
       RETURN
       END
C-============================================================

C-============================================================
C-====== integral in rapidity of X system ========
C-== N=0 for resonance production ============================
C-== N=1 for other processes =================================
C-============================================================
C--- check+
       FUNCTION IRAPX(N,MGG)

       IMPLICIT NONE
       INTEGER N
       DOUBLE PRECISION IRAPX,MGG
       DOUBLE PRECISION AUX1,AUX2,AUX3,AUX4
       DOUBLE PRECISION AUX5,AUX6,AUX7,AUX8
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2      
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX
 
       IF (N.EQ.0) THEN        
         AUX1=1.21D0*MGG**2/SQS**2/XI2MIN
         IF (AUX1.LT.XI1MIN) THEN
          AUX1=XI1MIN
         ENDIF
         AUX2=1.21D0*MGG**2/SQS**2/XI1MIN
         IF (AUX2.LT.XI2MIN) THEN
          AUX2=XI2MIN
         ENDIF
         AUX3=0.81D0*MGG**2/SQS**2/XI1MAX
         IF (AUX3.GT.XI2MAX) THEN
          AUX3=XI2MAX
         ENDIF
         AUX4=0.81D0*MGG**2/SQS**2/XI2MAX
         IF (AUX4.GT.XI1MAX) THEN
          AUX4=XI1MAX
         ENDIF
        
         IF (AUX1.GT.XI1MAX) THEN
          AUX1=XI1MAX
         ENDIF
         IF (AUX2.GT.XI2MAX) THEN
          AUX2=XI2MAX
         ENDIF       
         IF (AUX3.LT.XI2MIN) THEN
          AUX3=XI2MIN
         ENDIF       
         IF (AUX4.LT.XI1MIN) THEN
          AUX4=XI1MIN
         ENDIF
       ENDIF  

       IF (N.EQ.1) THEN
         AUX1=MGG**2/SQS**2/XI2MIN
         IF (AUX1.LT.XI1MIN) THEN
          AUX1=XI1MIN
         ENDIF
         AUX2=MGG**2/SQS**2/XI1MIN
         IF (AUX2.LT.XI2MIN) THEN
          AUX2=XI2MIN
         ENDIF
         AUX3=MGG**2/SQS**2/XI1MAX
         IF (AUX3.GT.XI2MAX) THEN
          AUX3=XI2MAX
         ENDIF
         AUX4=MGG**2/SQS**2/XI2MAX
         IF (AUX4.GT.XI1MAX) THEN
          AUX4=XI1MAX
         ENDIF

         AUX5=0.8D0*MGG**2*XI1MAX/MGGCUT**2
         IF (AUX5.LT.XI1MIN) THEN
          AUX5=XI1MIN
         ENDIF
         AUX6=0.8D0*MGG**2*XI2MAX/MGGCUT**2
         IF (AUX6.LT.XI2MIN) THEN
          AUX6=XI2MIN
         ENDIF
         AUX7=MGGCUT**2/(0.8D0*XI1MAX*SQS**2)
         IF (AUX7.GT.XI2MAX) THEN
          AUX7=XI2MAX
         ENDIF
         AUX8=MGGCUT**2/(0.8D0*XI2MAX*SQS**2)
         IF (AUX8.GT.XI1MAX) THEN
          AUX8=XI1MAX
         ENDIF
 
        IF (AUX5.LT.AUX1.AND.AUX5.LT.XI1MAX) THEN
         AUX1=AUX5
        ENDIF 
        IF (XI1MAX.LT.AUX1.AND.XI1MAX.LT.AUX5) THEN
         AUX1=XI1MAX
        ENDIF
        IF (AUX6.LT.AUX2.AND.AUX6.LT.XI2MAX) THEN
         AUX2=AUX6
        ENDIF 
        IF (XI2MAX.LT.AUX2.AND.XI2MAX.LT.AUX6) THEN
         AUX2=XI2MAX
        ENDIF
        IF (AUX7.GT.AUX3.AND.AUX7.LT.XI2MIN) THEN
         AUX3=AUX7
        ENDIF 
        IF (XI2MIN.GT.AUX3.AND.XI2MIN.GT.AUX7) THEN
         AUX3=XI2MIN
        ENDIF
        IF (AUX8.GT.AUX4.AND.AUX8.LT.XI1MIN) THEN
         AUX4=AUX8
        ENDIF 
        IF (XI1MIN.GT.AUX4.AND.XI1MIN.GT.AUX8) THEN
         AUX4=XI1MIN
        ENDIF              
       ENDIF

       IRAPX=0.5D0*(DLOG(AUX1*AUX2/(AUX3*AUX4)))       

       RETURN
       END
C-============================================================

C-============================================================
C-====== integrated gg luminosity ============================
C- N1=0 exclusive, N1=1 semiinclusive ========================
C- N2=0 resonance, N2=1 other processes ======================
C- MGG=mass of the central system  ===========================
C-============================================================
       FUNCTION M2DLUMDM2(N1,N2,MGG)

       IMPLICIT NONE 

       INTEGER N1,N2
       DOUBLE PRECISION M2DLUMDM2,MGG,MU
       DOUBLE PRECISION SOFTSURV,TSLOPEBH
       DOUBLE PRECISION ISUDGEN2,ISUD02,IRAPX
       DOUBLE COMPLEX SGNTR
       DOUBLE PRECISION AUX1,AUX2,AUX3
       DOUBLE PRECISION AUX4,AUX5
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2 
      
       MU=MGG*0.5D0
       SGNTR=MNI+REI*DTAN(PI*DP(3)/2.D0)

       AUX1=(SGNTR*DCONJG(SGNTR))**2
       AUX1=AUX1*CGP**4/(32.D0*PI**6)
       AUX1=AUX1*(SQS/MGG)**(4.D0*DP(3))
       
       AUX2=DEXP(-2.D0*TSLOPEBH(SQS/MGG)*T1MIN)
       AUX2=AUX2-DEXP(-2.D0*TSLOPEBH(SQS/MGG)*T1MAX)
       AUX3=DEXP(-2.D0*TSLOPEBH(SQS/MGG)*T2MIN)
       AUX3=AUX2-DEXP(-2.D0*TSLOPEBH(SQS/MGG)*T2MAX)       
       AUX4=AUX2*AUX3/(2.D0*TSLOPEBH(SQS/MGG))**2
       AUX4=AUX4*AUX1*SOFTSURV(1,MGG)
       
       IF (N1.EQ.0) THEN
        AUX5=ISUD02(1,MU)
       ENDIF
       IF (N1.GT.0) THEN
        AUX5=ISUDGEN2(1,MU)
       ENDIF
       
       M2DLUMDM2=AUX4*AUX5*IRAPX(N2,MGG)
       
       RETURN
       END
C-============================================================

C--- parton level -- c.-s. for resonances --------------------
C-============================================================
C-========SM Higgs functions =================================
C-============================================================
c------------------ VAR=MH**2/(4*Mt**2) ---------
       FUNCTION FSMH(VAR)

       IMPLICIT NONE
       DOUBLE PRECISION VAR,AUX0
       DOUBLE COMPLEX FSMH,AUX1,AUX2,AUX3
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    

       IF(VAR.GE.1.D0) THEN
        AUX0=DSQRT(1.D0-1.D0/VAR)
        AUX1=REI*DSQRT(1.D0-1.D0/VAR)
        AUX2=REI*DLOG((1.D0+AUX0)/(1.D0-AUX0))-MNI*PI
        AUX3=AUX2*AUX2
       ELSE
        AUX0=DSQRT(1.D0/VAR-1.D0)
        AUX1=REI*DSQRT(1.D0/VAR-1.D0)
        AUX2=REI*DATAN(1.D0/AUX0)
        AUX3=-4.D0*AUX2*AUX2
       ENDIF
C-   
       FSMH=(1.D0-0.25D0*(1.D0-1.D0/VAR)*AUX3)/VAR  
              
       RETURN
       END

c---- Higgs gg-width ----------------------------
       FUNCTION WIDTHSMH(MH)

       IMPLICIT NONE
       DOUBLE PRECISION MH,GF,MTOP,WIDTHSMH
       DOUBLE PRECISION ALPHAS,MU,VAR
       DOUBLE COMPLEX FSMH
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    

       MU=MH*0.5D0
       MTOP=174.D0
       GF=1.17D-05 
       VAR=0.25D0*(MH/MTOP)**2
       WIDTHSMH=FSMH(VAR)*DCONJG(FSMH(VAR))
       WIDTHSMH=WIDTHSMH*(ALPHAS(MU)/(2.D0*PI))**2
       WIDTHSMH=WIDTHSMH*MH**3*GF/(4.D0*PI*DSQRT(2.D0))
       WIDTHSMH=WIDTHSMH*(1.2D0+(PI*PI+5.5D0)*ALPHAS(MU)/PI)
              
       RETURN
       END
C-============================================================
C-============================================================
C-======== RS1 functions  and subs. ==========================
C-============================================================
c-subroutine to calc. obs. mass and width --------------------
      SUBROUTINE EDDERS1C(NRS,RSXI,RSGAM,RSMH,RSMR,RSMOBS,RSWD,BR)

       IMPLICIT NONE
       INTEGER NRS       
       DOUBLE PRECISION RSXI,RSGAM,RSMH,RSMR,RSMOBS,RSWD,BR
       DOUBLE PRECISION ALPHAS,GF,MTOP,MU
       DOUBLE COMPLEX FSMH 
c--- additional variables -----------------------
c       DOUBLE PRECISION AUX1,AUX2,AUX3,AUX4,AUX5,AUX6
       DOUBLE PRECISION AUX1,AUX2
       DOUBLE PRECISION RSXIMIN,RSXIMAX
       DOUBLE PRECISION RSZ,RSTHETA,RSAF,RSBF,RSCF,RSDF
       DOUBLE PRECISION RSA34F,RSA12F,RSMEDGE2,RSMEDGE3
       DOUBLE COMPLEX RSFMIXH,RSFMIXR
       DOUBLE PRECISION VARH,VARR 
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW           
c----- RS1 parameters -------------------------------------
      INTEGER NRS0 
      DOUBLE PRECISION RSXI0,RSGAM0,RSMH0,RSMR0 
      COMMON/EDDERS1/ RSXI0,RSGAM0,RSMH0,RSMR0,NRS0

       GF=1.17D-05
       MTOP=174.D0
       RSXIMIN=-(1.D0+DSQRT(1.D0+4.D0/RSGAM**2))/12.D0
       RSXIMAX=(-1.D0+DSQRT(1.D0+4.D0/RSGAM**2))/12.D0
       IF (RSMH.GT.RSMR) THEN
       RSXIMIN=-(1.D0+DSQRT(1.D0+8.D0*(1.D0-(RSMR/RSMH)**2)/RSGAM**2))
       RSXIMIN=RSXIMIN/24.D0
       RSXIMAX=(-1.D0+DSQRT(1.D0+8.D0*(1.D0-(RSMR/RSMH)**2)/RSGAM**2))
       RSXIMAX=RSXIMAX/24.D0
       ENDIF
       IF (RSXI.LT.RSXIMIN.OR.RSXI.GT.RSXIMAX) THEN
          PRINT*,'Attention: RS1 parameters are out of the range'
       ENDIF
       
        RSZ=DSQRT(1-6.D0*RSXI*RSGAM**2*(1.D0+6.D0*RSXI))
        AUX1=12.D0*RSXI*RSGAM*RSZ*RSMH**2
        AUX2=(RSMH**2*(RSZ**2-36.D0*(RSXI*RSGAM)**2)-RSMR**2)
        IF (AUX2.EQ.0.D0) THEN
         RSTHETA=PI/4.D0
         PRINT*,'ATTENTION: MIXING ANGLE IS INDEFINITE +-PI/4'
        ELSE
         RSTHETA=0.5D0*DATAN(AUX1/AUX2)       
        ENDIF
       
       RSAF=DCOS(RSTHETA)/RSZ
       RSBF=-DSIN(RSTHETA)/RSZ
       RSCF=DSIN(RSTHETA)-6.D0*RSXI*RSGAM*DCOS(RSTHETA)/RSZ
       RSDF=DCOS(RSTHETA)+6.D0*RSXI*RSGAM*DSIN(RSTHETA)/RSZ
       RSA34F=RSDF+RSBF*RSGAM
       RSA12F=RSAF+RSCF/RSGAM
       
       RSMEDGE2=RSMR/DSQRT(RSZ**2-36.D0*(RSXI*RSGAM)**2)
       RSMEDGE3=RSMH*DSQRT(RSZ**2-36.D0*(RSXI*RSGAM)**2)

       VARH=0.25D0*(RSMH/MTOP)**2
       VARR=0.25D0*(RSMR/MTOP)**2
       RSFMIXH=FSMH(VARH)*RSA34F+REI*7.D0*RSGAM*RSBF
       RSFMIXR=(FSMH(VARR)*RSA12F+7.D0*RSAF)*RSGAM 
c--- mass of the observable particle NRS=1 H*,NRS=2 R* -------
c--- BR is the coefficient to multiply branching of SM Higgs -
c- to obtain a branching for RS1 particle --------------------
c- (or simply mult. the total c.-s.) ---------------------------
c- use the generator for SM Higgs with new c.-s. or branchings -
       IF (NRS.EQ.1) THEN
        RSMOBS=DSQRT((RSDF*RSMH)**2+(RSBF*RSMR)**2)
        RSWD=RSFMIXH*DCONJG(RSFMIXH)
        BR=RSA34F**2
       ENDIF 
       IF (NRS.EQ.2) THEN
        RSMOBS=DSQRT((RSCF*RSMH)**2+(RSAF*RSMR)**2)
        RSWD=RSFMIXR*DCONJG(RSFMIXR)
        BR=(RSGAM*RSA12F)**2
       ENDIF
       IF (NRS.EQ.0.OR.NRS.GT.2) THEN
        PRINT*,'RS ID PARAMETER IS OUT OF THE RANGE'
        PRINT*,'SET AUTOMATICALLY TO RADION ID->2'
        NRS=2
        RSMOBS=DSQRT((RSCF*RSMH)**2+(RSAF*RSMR)**2)
        RSWD=RSFMIXR*DCONJG(RSFMIXR)
        BR=(RSGAM*RSA12F)**2
       ENDIF

        MU=RSMOBS*0.5D0
        RSWD=RSWD*RSMOBS**3*GF/(4.D0*PI*DSQRT(2.D0))
        RSWD=RSWD*(ALPHAS(MU)/(2.D0*PI))**2
        RSWD=RSWD*(1.2D0+(PI*PI+5.5D0)*ALPHAS(MU)/PI)       
       
       RETURN
       END
C-============================================================

C--- dSigma/dMc functions for different systems --------------
C-============================================================
C-============ exclusive pp->p+gg+p (fb/GeV)==================
C-============================================================
       FUNCTION TOTEXGG(M)

       IMPLICIT NONE
       DOUBLE PRECISION CSGG,TOTEXGG,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ     

       KFAC=1.3D0
       TOTEXGG=M2DLUMDM2(0,1,M)*KFAC*CSGG(M)
       TOTEXGG=TOTEXGG*CSMB*1.0D+12*2.D0/M
      
       
       RETURN
       END
C-============================================================
C-============================================================
C-============ exclusive pp->p+QQbar+p (fb/GeV) ==============
C-============================================================
       FUNCTION TOTEXQQ(M)

       IMPLICIT NONE
       DOUBLE PRECISION CSQQ,TOTEXQQ,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ

       KFAC=1.3D0
       TOTEXQQ=M2DLUMDM2(0,1,M)*KFAC*CSQQ(MQ,M)
       TOTEXQQ=TOTEXQQ*CSMB*1.0D+12*2.D0/M
       
       RETURN
       END
C-============================================================
C-============================================================
C-============ exclusive pp->p+gamma gamma+p (fb/GeV)=========
C-============================================================
       FUNCTION TOTEX2GAM(M)

       IMPLICIT NONE
       DOUBLE PRECISION CS2GAM,TOTEX2GAM,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ     

       KFAC=1.3D0
       TOTEX2GAM=M2DLUMDM2(0,1,M)*KFAC*CS2GAM(M)
       TOTEX2GAM=TOTEX2GAM*CSMB*1.0D+12*2.D0/M
      
       
       RETURN
       END
C-============================================================
C-============================================================
C-============ exclusive pp->p+gg g*+p (fb/GeV)===============
C-============================================================
       FUNCTION TOTEX3G(M)

       IMPLICIT NONE
       DOUBLE PRECISION CS3G,TOTEX3G,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ     

       KFAC=1.3D0
       TOTEX3G=M2DLUMDM2(0,1,M)*KFAC*CS3G(M)
       TOTEX3G=TOTEX3G*CSMB*1.0D+12*2.D0/M
      
       
       RETURN
       END
C-============================================================
C-============================================================
C-============ exclusive pp->p+QQbar g*+p (fb/GeV)== (MQ=0) ==
C-============================================================
       FUNCTION TOTEXQQG(M)

       IMPLICIT NONE
       DOUBLE PRECISION CSQQG,TOTEXQQG,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ     

       KFAC=1.3D0
       TOTEXQQG=M2DLUMDM2(0,1,M)*KFAC*CSQQG(M)
       TOTEXQQG=TOTEXQQG*CSMB*1.0D+12*2.D0/M
      
       
       RETURN
       END
C-============================================================
C-++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C-============================================================
C-============ semi-inclusive pp->p+{X gg Y}+p (fb/GeV)=======
C-============================================================
       FUNCTION TOTSIGG(M)

       IMPLICIT NONE
       DOUBLE PRECISION CSGGSI,TOTSIGG,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ     

       KFAC=1.3D0
       TOTSIGG=M2DLUMDM2(1,1,M)*KFAC*CSGGSI(M)
       TOTSIGG=TOTSIGG*CSMB*1.0D+12*2.D0/M
            
       RETURN
       END
C-============================================================
C-============================================================
C-============ semi-inclusive pp->p+{X QQbar Y}+p (fb/GeV) ===
C-============================================================
       FUNCTION TOTSIQQ(M)

       IMPLICIT NONE
       DOUBLE PRECISION CSQQSI,TOTSIQQ,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ

       KFAC=1.3D0
       TOTSIQQ=M2DLUMDM2(1,1,M)*KFAC*CSQQSI(MQ,M)
       TOTSIQQ=TOTSIQQ*CSMB*1.0D+12*2.D0/M
       
       RETURN
       END
C-============================================================
C-============================================================
C-======= semi-inclusive pp->p+{X gamma gamma Y}+p (fb/GeV)===
C-============================================================
       FUNCTION TOTSI2GAM(M)

       IMPLICIT NONE
       DOUBLE PRECISION CS2GAMSI,TOTSI2GAM,M,KFAC
       DOUBLE PRECISION M2DLUMDM2
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ     

       KFAC=1.3D0
       TOTSI2GAM=M2DLUMDM2(1,1,M)*KFAC*CS2GAMSI(M)
       TOTSI2GAM=TOTSI2GAM*CSMB*1.0D+12*2.D0/M
             
       RETURN
       END
C-============================================================

C-============================================================
c-    ------------------------- 
c-.. integrated cross section of the SM Higgs production in 
c-.. EDDE and SI DDE(in fb), 
c-.. M_H=30-300 GeV (and RS1 also)
c-    ------------------------- 
c-.. integrated cross sections (in fb) 
c-.. in EDDE and SI DDE at MX=30-300 GeV, X=gg,QQbar,2gamma,3g,QQbarg
c-...........................................................................
C-============================================================
      FUNCTION EDDECS(IP)

      IMPLICIT NONE

c------ dSigma/dMc functions for different systems -----------      
      EXTERNAL TOTEXGG,TOTEXQQ,TOTEX2GAM,TOTEX3G,TOTEXQQG
      EXTERNAL TOTSIGG,TOTSIQQ,TOTSI2GAM
      
      DOUBLE PRECISION TOTEXGG,TOTEXQQ,TOTEX2GAM,TOTEX3G,TOTEXQQG
      DOUBLE PRECISION TOTSIGG,TOTSIQQ,TOTSI2GAM
c-----------------
      INTEGER IP
      
      DOUBLE PRECISION DISIMP,EDDECS
c-----------------
c------ integrated gg luminocity -----------------------------      
      DOUBLE PRECISION M2DLUMDM2
      
c      DOUBLE PRECISION LIM1,LIM2,AUX1,AUX2,AUX3
      DOUBLE PRECISION LIM1,LIM2,AUX1,AUX2
c----- for resonances -------------------------------------
      DOUBLE PRECISION WIDTHSMH
      DOUBLE PRECISION RSMOBS,RSWD,RSBR
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW          
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX          
c----- RS1 parameters -------------------------------------
      INTEGER NRS0
      DOUBLE PRECISION RSXI0,RSGAM0,RSMH0,RSMR0 
      COMMON/EDDERS1/ RSXI0,RSGAM0,RSMH0,RSMR0,NRS0      
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ    
     
       IF (IP.EQ.400) THEN                    
        IF(AM0.LT.90.D0.OR.AM0.GT.300.D0) THEN
         PRINT*,'EDDECS SM Higgs--------->Attention:'
       PRINT*,'generator works good in the Higgs mass region',
     & 90.D0,'---',300.D0
         PRINT*,'you define mass of the Higgs = ',AM0
        ENDIF
        IF (NRS0.EQ.0) THEN
         WRITE(*,*)'XXXXXXXXXXXXXXXXXXXXX'
         EDDECS=M2DLUMDM2(0,0,AM0)*WIDTHSMH(AM0)*2.D0*PI**2/AM0**3
         WRITE(*,*)'M2DLUMDM2=',M2DLUMDM2(0,0,AM0)
         WRITE(*,*)'WIDTH=',WIDTHSMH(AM0)
         WRITE(*,*)'CS=',EDDECS
         WRITE(*,*)'XXXXXXXXXXXXXXXXXXXXX'
c-!!!
c-        AUX3=WIDTHSMH(AM0)   
c-        PRINT*,'MH=',AM0,'   WIDTH=',AUX3
        ELSE          
       CALL EDDERS1C(NRS0,RSXI0,RSGAM0,RSMH0,RSMR0,RSMOBS,RSWD,RSBR)
         EDDECS=M2DLUMDM2(0,0,RSMOBS)*RSWD*2.D0*PI**2*RSBR/RSMOBS**3 
c-!!!
c-        PRINT*,'MRS1=',RSMOBS,'   WIDTH=',RSWD,'  BR=',RSBR
C---- TAKE OUT RSBR IF BRANCHING IS RENORMALIZED ----------
          IF(RSMOBS.LT.30.D0.OR.RSMOBS.GT.300.D0) THEN
          PRINT*,'EDDECS RS1 --------->Attention:'
          PRINT*,'generator works good in the mass region',30.D0,'---',300.D0
          PRINT*,'calculation gives mass of the central particle = '
     &    ,RSMOBS
          ENDIF
        ENDIF
        EDDECS=EDDECS*CSMB*1.0D+12
       ENDIF
       
       IF (IP.EQ.406) THEN                    
        IF(AM0.LT.90.D0.OR.AM0.GT.300.D0) THEN
         PRINT*,'EDDECS SM Higgs--------->Attention:'
         PRINT*,'generator works good in the Higgs mass region',
     &   90.D0,'---',300.D0
         PRINT*,'you define mass of the Higgs = ',AM0
        ENDIF
        IF (NRS0.EQ.0) THEN
         EDDECS=M2DLUMDM2(1,0,AM0)*WIDTHSMH(AM0)
         EDDECS=EDDECS*2.D0*PI**2/AM0**3/2.D0
c-!!!
c-        AUX3=WIDTHSMH(AM0)   
c-        PRINT*,'MH=',AM0,'   WIDTH=',AUX3
        ELSE
       CALL EDDERS1C(NRS0,RSXI0,RSGAM0,RSMH0,RSMR0,RSMOBS,RSWD,RSBR)
         EDDECS=M2DLUMDM2(1,0,RSMOBS)*RSWD 
         EDDECS=EDDECS*2.D0*PI**2*RSBR/RSMOBS**3/2.D0   
c-!!!
c-        PRINT*,'MRS1=',RSMOBS,'   WIDTH=',RSWD,'  BR=',RSBR
C---- TAKE OUT RSBR IF BRANCHING IS RENORMALIZED ----------
          IF(RSMOBS.LT.30.D0.OR.RSMOBS.GT.300.D0) THEN
          PRINT*,'EDDECS RS1 --------->Attention:'
          PRINT*,'generator works good in the mass region',30.D0,'---',300.D0
          PRINT*,'calculation gives mass of the central particle = '
     &    ,RSMOBS
          ENDIF
        ENDIF
        EDDECS=EDDECS*CSMB*1.0D+12
       ENDIF

C----limits for the integration in MGG -----------------------
        LIM1=DSQRT(XI1MIN*XI2MIN)*SQS
        LIM2=DSQRT(XI1MAX*XI2MAX)*SQS
        AUX1=LIM2
        IF (AUX1.GT.MGGCUT) THEN
         AUX1=MGGCUT
        ENDIF
        IF (LIM1.LE.AUX1) THEN
         LIM1=MGGCUT
        ENDIF               

       AUX2=0.001D0 
       IF (IP.EQ.401) THEN
        EDDECS=DISIMP(TOTEXQQ,LIM1,LIM2,AUX2)              
       ENDIF
       
       IF (IP.EQ.402) THEN
        EDDECS=DISIMP(TOTEXGG,LIM1,LIM2,AUX2) 
       ENDIF       

       IF (IP.EQ.403) THEN
        EDDECS=DISIMP(TOTEX2GAM,LIM1,LIM2,AUX2)              
       ENDIF

       IF (IP.EQ.404) THEN
        EDDECS=DISIMP(TOTEXQQG,LIM1,LIM2,AUX2)              
       ENDIF

       IF (IP.EQ.405) THEN
        EDDECS=DISIMP(TOTEX3G,LIM1,LIM2,AUX2)              
       ENDIF

       IF (IP.EQ.407) THEN
        EDDECS=DISIMP(TOTSIQQ,LIM1,LIM2,AUX2)             
       ENDIF

       IF (IP.EQ.408) THEN
        EDDECS=DISIMP(TOTSIGG,LIM1,LIM2,AUX2)              
       ENDIF

       IF (IP.EQ.409) THEN
        EDDECS=DISIMP(TOTSI2GAM,LIM1,LIM2,AUX2)              
       ENDIF

c-        PRINT*,'c.-s. is calculated with accuracy',AUX2*100.D0,'%'
c-        PRINT*,'LIM1=',LIM1,' LIM2=',LIM2,' IP=',IP
        
      RETURN
      END

C-============================================================ 
      SUBROUTINE EDDEPUTDAT
      
      IMPLICIT NONE
    
C-... model parameters
      EXTERNAL EDDEDATA,EDDETABLES
c-...global EDDE parameters from FFR file
      INTEGER MXGLPAR
      REAL EDDEPAR
      PARAMETER   (MXGLPAR=200)
      COMMON /EDDEGLPAR/ EDDEPAR(MXGLPAR)
      
      INTEGER I,J,IJ
      DOUBLE PRECISION AUX1,AUX2,AUX3
c---- tables ------------------------------------      
c---- table for Intfgg**2 for Lum_sidde=Lum_excl*Intfgg**2 ---
      DOUBLE PRECISION LUM1,FLUM1,DX1,DY1,X01,Y01
      COMMON/EDDETAB1/ LUM1(480),FLUM1(30,16),
     & DX1,DY1,X01,Y01
c---- table for 3g ratio dIsud3g(x,mu)**2/Isud0(mu)**2 -------
      DOUBLE PRECISION RDI3G,FRDI3G,DX2,DY2,X02,Y02
      COMMON/EDDETAB2/ RDI3G(630),FRDI3G(30,21),
     & DX2,DY2,X02,Y02 
c--- table for 3g ratio Isud3ga(etasimax,mu)**2/Isud0(mu)**2 -
c--- table for 3g ratio Isud3gb(etasimax,mu)**2/Isud0(mu)**2 -
      DOUBLE PRECISION RI3GA,RI3GB,FRI3GA,FRI3GB,
     & DX3,DY3,X03,Y03
      COMMON/EDDETAB3/ RI3GA(480),RI3GB(480),
     & FRI3GA(30,16),FRI3GB(30,16),DX3,DY3,X03,Y03
c- tables for fM(x,pt)
       DOUBLE PRECISION FT4,FMT4,DX4,DY4,X04,Y04
       COMMON/EDDETAB4/ FT4(10201),FMT4(101,101),
     & DX4,DY4,X04,Y04 
c-- tables for dfM_g/g(x,Y)/dY, Y=DLOG(PT**2/LAMQCD**2) ---
       DOUBLE PRECISION DFT5,DFMT5,DX5,DY5,X05,Y05
       COMMON/EDDETAB5/ DFT5(10201),DFMT5(101,101),
     & DX5,DY5,X05,Y05     
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)     
c----- parameters to calculate total cross-sections -------      
c----- RS1 parameters -------------------------------------
      INTEGER NRS0 
      DOUBLE PRECISION RSXI0,RSGAM0,RSMH0,RSMR0 
      COMMON/EDDERS1/ RSXI0,RSGAM0,RSMH0,RSMR0,NRS0
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c- commons for cascad generation      
      DOUBLE PRECISION FGGQ0,FGGQS
      COMMON/EDDEFGG/FGGQ0,FGGQS
c----- data that needs calculation
c----- f_g/g lower limit and minimal generated PT_soft_g
       FGGQ0=0.3D0
       FGGQS=1.D0
c--- RS1 data from FFR ----------------
       NRS0=EDDEPAR(9) ! FFRDAT EDDEPAR(6)
       RSXI0=EDDEPAR(10) ! FFRDAT EDDEPAR(5)
       RSGAM0=EDDEPAR(11) ! FFRDAT EDDEPAR(6)
       RSMH0=EDDEPAR(12) ! FFRDAT EDDEPAR(7)
       RSMR0=EDDEPAR(13) ! FFRDAT EDDEPAR(8)
c--- other parameters
c--- code of central particle
       KCP=EDDEPAR(5) ! FFRDAT EDDEPAR(4)
c--- code of the process
       IPROC=EDDEPAR(1) ! FFRDAT EDDEPAR(1)
c--- mass of the central particle
       AM0=EDDEPAR(6) ! FFRDAT EDDEPAR(3)
      IF (NRS0.GT.0) THEN
      CALL EDDERS1C(NRS0,RSXI0,RSGAM0,RSMH0,RSMR0,AUX1,AUX2,AUX3)
      AM0=AUX1
      ENDIF
c--- phi dependence code
       NFI=EDDEPAR(7) ! FFRDAT EDDEPAR(5)
c------ constants -----------------------------------------
       CF=(NC*NC-1)*TF/NC
       BF0=(11.D0*NC-2.D0*NF)/(12.D0*PI)
       BF1=34.D0*NC*NC/3.D0-20.D0*NC*NF*TF/3.D0
       BF1=(BF1-4.D0*CF*TF*NF)/(BF0*16.D0*PI*PI)      
c------ cuts for hard c.-s. variables
c- external input SQS ---
       SQS=EDDEPAR(4)  ! FFRDAT EDDEPAR(1)
       S=SQS*SQS
       MQ=4.8D0 ! FFRDAT EDDEPAR(10)
       MXMAX=500.D0
c------ additional par. for c.-s. calculations ------------
       FKK=0.0196721D0+0.737705D-06*SQS 
c- external input MGGCUT -------
c- cut on the central mass ----- 
       ETJCUT=EDDEPAR(8) ! FFRDAT EDDEPAR(2)
       MGGCUT=2.D0*ETJCUT
       ETAJMAX=DLOG(MXMAX/MGGCUT+DSQRT((MXMAX/MGGCUT)**2-1))
c- external input ETASIMAX -----------------
c- cut on semi-incl. |eta_smth|<ETASIMAX/2 -
c- rap-gap constraint ----------------------
c- parameter is relevant only for exclusive 3j case,
c- SIDDE calculations for all rapidities by def.
       ETASIMAX=EDDEPAR(14) ! FFRDAT EDDEPAR(9)     
       PSURV=0.66D-01
       PSUD=0.6795D0 ! for exclusive case
c--- calculation of estimation pars for ISUDGEN2 -------------
c-       IF (ETASIMAX.LE.0.D0) THEN
c-        PSIDD1=0.D0
c-        PSIDD2=0.D0
c-       ENDIF 
c-       IF (ETASIMAX.LE.2.D0.AND.ETASIMAX.GT.0.D0) THEN
c-        PSIDD1=(0.9D-05+0.5D0*ETASIMAX*(0.02023D0-0.9D-05))*5.D0**0.7D0
c-        PSIDD2=-0.7D0
c-       ENDIF
c-       IF (ETASIMAX.LE.4.D0.AND.ETASIMAX.GT.2.D0) THEN
c-        PSIDD1=(0.02023D0+0.123D0*(ETASIMAX-2.D0))*5.D0**0.7D0
c-        PSIDD2=-0.7D0
c-       ENDIF
c-       IF (ETASIMAX.LE.6.D0.AND.ETASIMAX.GT.4.D0) THEN
c-        PSIDD1=(0.215D0+0.3D0*(ETASIMAX-4.D0))*5.D0**0.5D0
c-        PSIDD2=-0.5D0
c-       ENDIF       
c-       IF (ETASIMAX.LE.8.D0.AND.ETASIMAX.GT.6.D0) THEN
c-        PSIDD1=(0.5D0+0.27D0*(ETASIMAX-6.D0))*5.D0**0.1D0
c-        PSIDD2=-0.1D0
c-       ENDIF
c-       IF (ETASIMAX.LE.12.D0.AND.ETASIMAX.GT.8.D0) THEN
c-        PSIDD1=(0.673D0+0.17D0*(ETASIMAX-8.D0))/5.D0**0.35D0
c-        PSIDD2=0.35D0
c-       ENDIF
c-       IF (ETASIMAX.LE.16.D0.AND.ETASIMAX.GT.12.D0) THEN
c-        PSIDD1=(1.032D0+0.05D0*(ETASIMAX-12.D0))/5.D0**0.6D0
c-        PSIDD2=0.6D0
c-       ENDIF
c-       IF (ETASIMAX.GT.16.D0) THEN
c-        PSIDD1=0.9173/5.D0**0.8D0
c-        PSIDD2=0.8D0
c-       ENDIF
c- all the above PSIDDN for old semi-inclusive estimations.
c- for new sidde etamax->Infinity,is not required
       PSIDD1=0.112732D-01
       PSIDD2=1.5D0
c- overall gg-lum. negative power in MJJ for the dsig/dMJJ --- 
       PLUM=4.D0*DP(3)+1.D0+PSURV+PSUD-PSIDD2
c--- data for 3 jets
c--- model restrictions on the phase space of g-jet -------
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
       DER3J=0.75D0
       XMAX3J=0.72D0
       PAR3G(1)=0.22D0
       PAR3G(2)=(0.13D0*DLOG(DCOSH(ETASIMAX/2.D0))-0.2D0)
c---
       AUX1=0.61D0+2.42D0*(ETASIMAX*0.5D0-0.5D0)**0.7D0
        IF (ETASIMAX.GE.4.4D0.AND.ETASIMAX.LT.7.D0) THEN
         AUX1=4.16D0 
        ENDIF 
        IF (ETASIMAX.GE.7.D0.AND.ETASIMAX.LT.15.D0) THEN
         AUX1=1.D0+5.65D0/(0.5D0*ETASIMAX-2.D0)**1.6D0
        ENDIF
        IF (ETASIMAX.GE.15.D0) THEN
         AUX1=1.D0+5.65D0/5.5D0**1.6D0
        ENDIF
        IF (ETASIMAX.GT.10.D0) THEN
         AUX1=AUX1-0.2D0
        ENDIF
        PAR3G(3)=AUX1
c---
        AUX2=0.128D0*ETASIMAX/2.D0
        IF (ETASIMAX.GT.4.D0) THEN
         AUX2=AUX2+0.01D0*ETASIMAX/2.D0-0.09D0
        ENDIF
        IF (ETASIMAX.GT.8.D0) THEN
         AUX2=AUX2-0.05D0*ETASIMAX/2.D0-0.08D0
        ENDIF
        IF (ETASIMAX.GT.12.D0) THEN
         AUX2=AUX2-0.1D0*(ETASIMAX/2.D0-6.D0)-0.21D0
        ENDIF
        IF (ETASIMAX.GT.15.D0) THEN
         AUX2=AUX2-0.1D0*(15.D0/2.D0-6.D0)-0.21D0
        ENDIF        
        PAR3G(4)=AUX2
c---
        AUX3=0.2D0
        IF (ETASIMAX.GT.4.D0) THEN
         AUX3=AUX3+0.12D0
        ENDIF
        IF (ETASIMAX.GT.8.D0) THEN
         AUX3=AUX3+0.3D0
        ENDIF
        IF (ETASIMAX.GT.12.D0) THEN
         AUX3=AUX3+0.2D0
        ENDIF
        PAR3G(5)=AUX3

c--- manipulation with tables
       I=1
       J=1
       IJ=1
       DO I=1,30
         DO J=1,16
           FLUM1(I,J)=LUM1(IJ) 
           FRI3GA(I,J)=RI3GA(IJ)
           FRI3GB(I,J)=RI3GB(IJ)
           IJ=IJ+1
         ENDDO
       ENDDO
       
       I=1
       J=1
       IJ=1
       DO I=1,30
         DO J=1,21 
           FRDI3G(I,J)=RDI3G(IJ)
           IJ=IJ+1
         ENDDO
       ENDDO       

       I=1
       J=1
       IJ=1
       DO I=1,101
         DO J=1,101
           FMT4(I,J)=FT4(IJ) 
           DFMT5(I,J)=DFT5(IJ)
           IJ=IJ+1
         ENDDO
       ENDDO

      RETURN
      END

c-------------------------------------------------------------              
c- ... model parameters !R.Ryutin
c-------------------------------------------------------------              
      BLOCK DATA EDDEDATA
      IMPLICIT NONE        
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX   
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5) 
c----- RS1 parameters -------------------------------------
      INTEGER NRS0 
      DOUBLE PRECISION RSXI0,RSGAM0,RSMH0,RSMR0 
      COMMON/EDDERS1/ RSXI0,RSGAM0,RSMH0,RSMR0,NRS0      
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ

c----- parameters to calculate total cross-sections -------
      DATA XI1MIN/0.1D-04/,XI2MIN/0.1D-04/
      DATA XI1MAX/0.1D0/,XI2MAX/0.1D0/
c----- data------------------------------------------------     
c------ constants -----------------------------------------
      DATA PI/3.141592654D0/, CSMB/0.38D+00/
      DATA REI/(1.D0,0.D0)/,MNI/(0.D0,1.D0)/
      DATA NC/3/,NF/5/,NLOSW/1/,LAMQCD/0.5D-01/
      DATA TF/0.5D0/ 
c------ IPomeron linear trajectories: constants, ----------
c------ intercepts-1, pp-radii, gp-radii, alpha_primes ----  
      DATA CP/0.5300D+02,0.9700D+01,0.1670D+01/
      DATA DP/0.5800D-01,0.1670D+00,0.2030D+00/
      DATA RP/0.6300D+01,0.3100D+01,0.2480D+01/
      DATA RG/0.6300D+01,0.3100D+01,0.2540D+01/
      DATA AP/0.5600D+00,0.2730D+00,0.9400D-01/
c------ default values: |t_i| from 0.01 to 5 GeV**2 -------                  
      DATA T1MIN/0.1D-02/, T1MAX/0.7D+01/
      DATA T2MIN/0.1D-02/, T2MAX/0.7D+01/
c------ approximation: number of soft terms ---------------
      DATA NAPR/9/
c------ main constant gp-gp amplitude ---------------------
      DATA CGP/0.316D+01/ ! 3.16D0(2.7->3.8) for new data from CDF

       END    
      
c-------------------------------------------------------------              
c-... tables for interpolation !R.Ryutin
c-------------------------------------------------------------              
      BLOCK DATA EDDETABLES

      IMPLICIT NONE            
c---- table for Intfgg**2 for Lum_sidde=Lum_excl*Intfgg**2 ---
      DOUBLE PRECISION LUM1,FLUM1,DX1,DY1,X01,Y01
      COMMON/EDDETAB1/ LUM1(480),FLUM1(30,16),
     & DX1,DY1,X01,Y01
c---- table for 3g ratio dIsud3g(x,mu)**2/Isud0(mu)**2 -------
      DOUBLE PRECISION RDI3G,FRDI3G,DX2,DY2,X02,Y02
      COMMON/EDDETAB2/ RDI3G(630),FRDI3G(30,21),
     & DX2,DY2,X02,Y02 
c--- table for 3g ratio Isud3ga(etasimax,mu)**2/Isud0(mu)**2 -
c--- table for 3g ratio Isud3gb(etasimax,mu)**2/Isud0(mu)**2 -
      DOUBLE PRECISION RI3GA,RI3GB,FRI3GA,FRI3GB,
     & DX3,DY3,X03,Y03
      COMMON/EDDETAB3/ RI3GA(480),RI3GB(480),
     & FRI3GA(30,16),FRI3GB(30,16),DX3,DY3,X03,Y03     
c- tables for fM_g/g(x,pt)=g(x,pt)*Tsud(pt,mu)
       DOUBLE PRECISION FT4,FMT4,DX4,DY4,X04,Y04
       COMMON/EDDETAB4/ FT4(10201),FMT4(101,101),
     & DX4,DY4,X04,Y04 
c-- tables for dfM_g/g(x,Y)/dY, Y=DLOG(PT**2/LAMQCD**2) ---
       DOUBLE PRECISION DFT5,DFMT5,DX5,DY5,X05,Y05
       COMMON/EDDETAB5/ DFT5(10201),DFMT5(101,101),
     & DX5,DY5,X05,Y05    
      
      DATA DX1/5.D0/,DY1/2.D0/,X01/5.D0/,Y01/0.D0/
      DATA  LUM1/0.D0,0.01514D0,0.13131D0,0.33366D0,
     &  0.55813D0,0.6785D0,0.68486D0,0.68486D0,
     &  0.68486D0,0.68486D0,0.68486D0,0.68486D0,
     &  0.68486D0,0.68486D0,0.68486D0,0.68486D0,
     &  0.D0,0.0124D0,0.14447D0,0.44316D0,
     &  0.8491D0,1.26662D0,1.51587D0,1.56017D0,
     &  1.56017D0,1.56017D0,1.56017D0,1.56017D0,
     &  1.56017D0,1.56017D0,1.56017D0,1.56017D0,
     &  0.D0,0.00853D0,0.12174D0,0.4405D0,0.91081D0,
     &  1.48394D0,1.95785D0,2.18438D0,2.20792D0,
     &  2.20792D0,2.20792D0,2.20792D0,2.20792D0,
     &  2.20792D0,2.20792D0,2.20792D0,0.D0,0.00574D0,
     &  0.09705D0,0.40424D0,0.90146D0,1.55077D0,
     &  2.18262D0,2.58063D0,2.70382D0,2.70426D0,
     &  2.70426D0,2.70426D0,2.70426D0,2.70426D0,
     &  2.70426D0,2.70426D0,0.D0,0.0043D0,0.07622D0,
     &  0.35995D0,0.86601D0,1.55117D0,2.29222D0,
     &  2.83432D0,3.07586D0,3.1015D0,3.1015D0,3.1015D0,
     &  3.1015D0,3.1015D0,3.1015D0,3.1015D0,0.D0,0.00357D0,
     &  0.05963D0,0.31645D0,0.81955D0,1.52056D0,2.33685D0,
     &  2.99805D0,3.35059D0,3.43041D0,3.43041D0,
     &  3.43041D0,3.43041D0,3.43041D0,3.43041D0,3.43041D0,
     &  0.D0,0.0031D0,0.04663D0,0.27665D0,0.76919D0,
     &  1.47543D0,2.34316D0,3.10253D0,3.55679D0,
     &  3.7069D0,3.70999D0,3.70999D0,3.70999D0,
     &  3.70999D0,3.70999D0,3.70999D0,0.D0,0.00277D0,
     &  0.03857D0,0.24132D0,0.7185D0,1.424D0,2.32617D0,
     &  3.1666D0,3.71324D0,3.93374D0,3.95253D0,
     &  3.95253D0,3.95253D0,3.95253D0,3.95253D0,
     &  3.95253D0,0.D0,0.00252D0,0.03338D0,0.21035D0,
     &  0.66931D0,1.37055D0,2.29489D0,3.20237D0,
     &  3.83258D0,4.1213D0,4.16637D0,4.16637D0,
     &  4.16637D0,4.16637D0,4.16637D0,4.16637D0,
     &  0.D0,0.00232D0,0.02961D0,0.18337D0,
     &  0.62253D0,1.31736D0,2.25482D0,3.21787D0,
     &  3.92364D0,4.27788D0,4.35741D0,4.35741D0,
     &  4.35741D0,4.35741D0,4.35741D0,4.35741D0,
     &  0.D0,0.00217D0,0.0267D0,0.15991D0,0.57856D0,
     &  1.26553D0,2.20949D0,3.21864D0,3.99275D0,
     &  4.40956D0,4.52926D0,4.52994D0,4.52994D0,4.52994D0,
     &  4.52994D0,4.52994D0,0.D0,0.00204D0,0.02438D0,
     &  0.1395D0,0.5375D0,1.21542D0,2.16123D0,
     &  3.20859D0,4.04459D0,4.52096D0,4.68147D0, 
     &  4.6872D0,4.6872D0,4.6872D0,4.6872D0,4.6872D0,
     &  0.D0,0.00193D0,0.02248D0,0.12174D0,0.49933D0,
     &  1.1672D0,2.11151D0,3.19053D0,4.08264D0,
     &  4.61561D0,4.81656D0,4.83165D0,4.83165D0,4.83165D0,
     &  4.83165D0,4.83165D0,0.D0,0.00183D0,0.02089D0,
     &  0.10634D0,0.46393D0,1.12092D0,2.06138D0,
     &  3.16652D0,4.10956D0,4.69627D0,4.93714D0,4.96521D0,
     &  4.96521D0,4.96521D0,4.96521D0,4.96521D0,0.D0,
     &  0.00175D0,0.01954D0,0.09505D0,0.43115D0,1.07659D0,
     &  2.01152D0,3.13808D0,4.12744D0,4.76515D0,5.0452D0,
     &  5.08942D0,5.08942D0,5.08942D0,5.08942D0,5.08942D0,
     &  0.D0,0.00168D0,0.01838D0,0.08629D0,0.4008D0,
     &  1.03418D0,1.96238D0,3.10637D0,4.13789D0,4.82402D0,
     &  5.1425D0,5.20553D0,5.20553D0,5.20553D0,5.20553D0,
     &  5.20553D0,0.D0,0.00162D0,0.01737D0,0.07911D0,
     &  0.37273D0,0.99364D0,1.91425D0,3.07228D0,
     &  4.14223D0,4.8743D0,5.23035D0,5.31453D0,5.31453D0,
     &  5.31453D0,5.31453D0,5.31453D0,0.D0,0.00156D0,
     &  0.01648D0,0.07307D0,0.34674D0,0.95491D0,1.86733D0,
     &  3.03648D0,4.14148D0,4.91719D0,5.30992D0,5.41685D0,
     &  5.41728D0,5.41728D0,5.41728D0,5.41728D0,0.D0,0.00151D0,
     &  0.01569D0,0.0679D0,0.32267D0,0.9179D0,1.82173D0,
     &  2.99949D0,4.1365D0,4.95366D0,5.38219D0,5.51199D0,
     &  5.51446D0,5.51446D0,5.51446D0,5.51446D0,0.D0,
     &  0.00146D0,0.01499D0,0.06342D0,0.30038D0,0.88254D0,
     &  1.77752D0,2.96172D0,4.12799D0,4.98455D0,5.44796D0,
     &  5.60063D0,5.60668D0,5.60668D0,5.60668D0,5.60668D0,
     &  0.D0,0.00142D0,0.01435D0,0.0595D0,0.27971D0,
     &  0.84876D0,1.73473D0,2.92348D0,4.1165D0,5.01054D0,
     &  5.50793D0,5.68342D0,5.69443D0,5.69443D0,5.69443D0,
     &  5.69443D0,0.D0,0.00139D0,0.01378D0,0.05603D0,
     &  0.26054D0,0.81649D0,1.69337D0,2.88504D0,4.10253D0,
     &  5.03222D0,5.56272D0,5.76083D0,5.77814D0,5.77814D0,
     &  5.77814D0,5.77814D0,0.D0,0.00136D0,0.01326D0,0.05294D0,
     &  0.24274D0,0.78564D0,1.6534D0,2.84659D0,4.08644D0,
     &  5.0501D0,5.61283D0,5.83357D0,5.85819D0,5.85819D0,
     &  5.85819D0,5.85819D0,0.D0,0.00133D0,0.01278D0,0.05017D0,
     &  0.22621D0,0.75614D0,1.61481D0,2.80827D0,4.0686D0,
     &  5.06461D0,5.65872D0,5.90185D0,5.93489D0,5.93489D0,
     &  5.93489D0,5.93489D0,0.D0,0.00131D0,0.01234D0,0.04766D0,
     &  0.21084D0,0.72793D0,1.57757D0,2.77021D0,4.04928D0,
     &  5.07612D0,5.70077D0,5.96611D0,6.00854D0,6.00854D0,
     &  6.00854D0,6.00854D0,0.D0,0.00128D0,0.01194D0,0.04539D0,
     &  0.19654D0,0.70094D0,1.54162D0,2.73251D0,4.02871D0,
     &  5.08496D0,5.73936D0,6.02668D0,6.07938D0,6.07938D0,
     &  6.07938D0,6.07938D0,0.D0,0.00126D0,0.01156D0,
     &  0.04333D0,0.18323D0,0.67511D0,1.50692D0,2.69524D0,
     &  4.0071D0,5.0912D0,5.77476D0,6.08385D0,6.14764D0,
     &  6.14764D0,6.14764D0,6.14764D0,0.D0,0.D0,0.01122D0,
     &  0.04144D0,0.17083D0,0.65038D0,1.47343D0,2.65846D0,
     &  3.98462D0,5.09573D0,5.80727D0,6.13788D0,6.2135D0,
     &  6.2135D0,6.2135D0,6.2135D0,0.D0,0.D0,0.01089D0,
     &  0.0397D0,0.15928D0,0.62669D0,1.44106D0,2.6222D0,
     &  3.96141D0,5.09812D0,5.8371D0,6.18902D0,6.27707D0,
     &  6.27713D0,6.27713D0,6.27713D0,0.D0,0.D0,0.0106D0,
     &  0.0381D0,0.14863D0,0.604D0,1.40977D0,2.58656D0,
     &  3.93762D0,5.09879D0,5.8645D0,6.23745D0,6.33809D0,
     &  6.33871D0,6.33871D0,6.33871D0/
      DATA DX2/5.D0/,DY2/0.0497D0/,X02/5.D0/,Y02/0.003D0/      
      DATA RDI3G/1.D0,1.D0,1.D0,1.00001D0,1.00007D0,1.00027D0,
     &  1.00068D0,1.00139D0,1.00247D0,1.00405D0,
     &  1.00631D0,1.00913D0,1.01218D0,1.01535D0,1.01853D0,
     &  1.02154D0,1.02409D0,1.02582D0,1.02641D0,1.02642D0,
     &  1.02642D0,1.D0,1.D0,1.00003D0,1.00029D0,1.00101D0,
     &  1.00229D0,1.00421D0,1.00684D0,1.01029D0,1.01472D0,
     &  1.02041D0,1.02718D0,1.03459D0,1.04266D0,1.05147D0,
     &  1.06096D0,1.07093D0,1.08066D0,1.0884D0,1.0913D0,
     &  1.09131D0,1.D0,1.D0,1.00021D0,1.00113D0,1.00299D0,
     &  1.00583D0,1.00964D0,1.01448D0,1.02046D0,1.02778D0,
     &  1.03683D0,1.04739D0,1.05901D0,1.0719D0,1.08632D0,
     &  1.10259D0,1.12095D0,1.14113D0,1.16115D0,1.1743D0,
     &  1.17527D0,1.D0,1.00002D0,1.0006D0,1.00247D0,1.00574D0,
     &  1.01032D0,1.01613D0,1.0232D0,1.03166D0,1.04178D0,
     &  1.05402D0,1.06818D0,1.0838D0,1.10128D0,1.12114D0,
     &  1.14412D0,1.17103D0,1.20253D0,1.23763D0,1.26768D0,
     &  1.27289D0,1.D0,1.00006D0,1.00118D0,1.0042D0,1.00902D0,
     &  1.01541D0,1.02323D0,1.0325D0,1.04337D0,1.05617D0,
     &  1.07145D0,1.08903D0,1.10848D0,1.13037D0,1.15552D0,
     &  1.18507D0,1.22056D0,1.26381D0,1.31568D0,1.36814D0,
     &  1.38253D0,1.D0,1.00014D0,1.00195D0,1.00624D0,1.01268D0,
     &  1.0209D0,1.03071D0,1.04212D0,1.05532D0,1.07069D0,
     &  1.08889D0,1.10976D0,1.13289D0,1.15906D0,1.18936D0,
     &  1.22539D0,1.26944D0,1.3247D0,1.39454D0,1.474D0,
     &  1.50387D0,1.D0,1.00025D0,1.00287D0,1.00853D0,1.01662D0,
     &  1.02665D0,1.03842D0,1.05191D0,1.06737D0,1.08524D0,
     &  1.10625D0,1.13029D0,1.15699D0,1.18732D0,1.22268D0,
     &  1.26511D0,1.31772D0,1.3852D0,1.47394D0,1.58436D0,1.63714D0,
     &  1.D0,1.0004D0,1.00393D0,1.01101D0,1.02076D0,1.03259D0,
     &  1.04627D0,1.0618D0,1.07945D0,1.09973D0,1.12347D0,
     &  1.15059D0,1.18078D0,1.21519D0,1.25551D0,1.30429D0,
     &  1.36546D0,1.44535D0,1.55381D0,1.6987D0,1.78287D0,1.D0,
     &  1.0006D0,1.00511D0,1.01364D0,1.02504D0,1.03866D0,1.0542D0,
     &  1.07172D0,1.09151D0,1.11414D0,1.14054D0,1.17066D0,1.20426D0,
     &  1.24267D0,1.28791D0,1.34299D0,1.41274D0,1.50522D0,1.63414D0,
     &  1.81671D0,1.94177D0,1.D0,1.00082D0,1.00639D0,1.01639D0,
     &  1.02944D0,1.04481D0,1.06219D0,1.08165D0,1.10353D0,1.12845D0,
     &  1.15745D0,1.19051D0,1.22745D0,1.26982D0,1.31991D0,1.38127D0,
     &  1.45963D0,1.56488D0,1.71494D0,1.93821D0,2.11472D0,1.D0,
     &  1.00109D0,1.00775D0,1.01923D0,1.03393D0,1.05101D0,1.0702D0,
     &  1.09156D0,1.11548D0,1.14265D0,1.17419D0,1.21014D0,1.25038D0,
     &  1.29665D0,1.35157D0,1.41919D0,1.5062D0,1.62439D0,1.79625D0,
     &  2.06309D0,2.30268D0,1.D0,1.00139D0,1.00919D0,1.02216D0,
     &  1.03848D0,1.05726D0,1.07821D0,1.10144D0,1.12736D0,1.15674D0,
     &  1.19078D0,1.22957D0,1.27307D0,1.32321D0,1.38291D0,1.45678D0,
     &  1.55248D0,1.68382D0,1.87811D0,2.19127D0,2.50671D0,1.D0,
     &  1.00172D0,1.01069D0,1.02515D0,1.04307D0,1.06352D0,1.08622D0,
     &  1.11128D0,1.13917D0,1.17072D0,1.20722D0,1.24881D0,1.29553D0,
     &  1.3495D0,1.41398D0,1.4941D0,1.59854D0,1.74319D0,1.96055D0,
     &  2.32273D0,2.72799D0,1.D0,1.00208D0,1.01224D0,1.02819D0,
     &  1.0477D0,
     &  1.0698D0,1.09421D0,1.12107D0,1.1509D0,1.18459D0,1.22351D0,
     &  1.26788D0,1.31778D0,1.37556D0,1.44479D0,1.53117D0,1.6444D0,
     &  1.80257D0,2.0436D0,2.45745D0,2.96775D0,1.D0,1.00247D0,
     &  1.01384D0,
     &  1.03127D0,1.05236D0,1.07608D0,1.10217D0,1.13081D0,1.16255D0,
     &  1.19834D0,1.23966D0,1.28677D0,1.33983D0,1.4014D0,1.47538D0,
     &  1.56803D0,1.69012D0,1.86198D0,2.12729D0,2.59541D0,3.22734D0,
     &  1.D0,1.00288D0,1.01549D0,1.03439D0,1.05703D0,1.08236D0,
     &  1.11011D0,1.1405D0,1.17413D0,1.212D0,1.25569D0,1.30551D0,
     &  1.36171D0,1.42705D0,1.50577D0,1.60471D0,1.73571D0,1.92147D0,
     &  2.21166D0,2.73663D0,3.50817D0,1.D0,1.00332D0,1.01717D0,
     &  1.03754D0,1.06171D0,1.08862D0,1.11802D0,1.15014D0,1.18563D0,
     &  1.22556D0,1.27159D0,1.32411D0,1.38343D0,1.45253D0,1.53598D0,
     &  1.64122D0,1.78121D0,1.98105D0,2.29672D0,2.88111D0,3.81177D0,
     &  1.D0,1.00379D0,1.01888D0,1.04071D0,1.0664D0,1.09488D0,
     &  1.12589D0,1.15972D0,1.19705D0,1.23902D0,1.28738D0,1.34257D0,
     &  1.40499D0,1.47784D0,1.56603D0,1.67759D0,1.82663D0,2.04075D0,
     &  2.3825D0,3.02887D0,4.13975D0,1.D0,1.00427D0,1.02062D0,
     &  1.0439D0,1.07109D0,1.10111D0,1.13373D0,1.16925D0,1.2084D0,
     &  1.25239D0,1.30305D0,1.3609D0,1.42642D0,1.50301D0,1.59594D0,
     &  1.71384D0,1.87201D0,2.10059D0,2.46902D0,3.17993D0,4.49381D0,
     &  1.D0,1.00478D0,1.02239D0,1.0471D0,1.07579D0,1.10733D0,
     &  1.14154D0,1.17873D0,1.21969D0,1.26567D0,1.31862D0,1.37911D0,
     &  1.44771D0,1.52803D0,1.62571D0,1.74999D0,1.91736D0,2.1606D0,
     &  2.55629D0,3.33429D0,4.87576D0,1.D0,1.0053D0,1.02418D0,
     &  1.05031D0,1.08047D0,1.11354D0,1.14931D0,1.18815D0,1.2309D0,
     &  1.27887D0,1.3341D0,1.39721D0,1.46888D0,1.55294D0,1.65536D0,
     &  1.78604D0,1.96269D0,2.2208D0,2.64433D0,3.49198D0,5.28752D0,
     &  1.D0,1.00584D0,1.02598D0,1.05353D0,1.08516D0,1.11972D0,
     &  1.15704D0,1.19753D0,1.24205D0,1.29199D0,1.34948D0,1.41521D0,
     &  1.48994D0,1.57773D0,1.68491D0,1.82202D0,2.00802D0,2.28119D0,
     &  2.73317D0,3.65301D0,5.73111D0,1.D0,1.00639D0,1.02781D0,
     &  1.05676D0,1.08983D0,1.12587D0,1.16474D0,1.20685D0,1.25313D0,
     &  1.30504D0,1.36477D0,1.43311D0,1.51089D0,1.60241D0,1.71437D0,
     &  1.85794D0,2.05337D0,2.34181D0,2.8228D0,3.81742D0,6.20865D0,
     &  1.D0,1.00696D0,1.02964D0,1.05999D0,1.0945D0,1.13201D0,
     &  1.1724D0,1.21613D0,1.26416D0,1.31801D0,1.37998D0,1.45091D0,
     &  1.53175D0,1.627D0,1.74374D0,1.89381D0,2.09874D0,2.40264D0,
     &  2.91325D0,3.98522D0,6.72239D0,1.D0,1.00755D0,1.03149D0,
     &  1.06323D0,1.09916D0,1.13812D0,1.18002D0,1.22536D0,1.27512D0,
     &  1.33091D0,1.39511D0,1.46863D0,1.55252D0,1.6515D0,1.77303D0,
     &  1.92963D0,2.14415D0,2.46371D0,3.00452D0,4.15642D0,7.2747D0,
     &  1.D0,1.00815D0,1.03335D0,1.06647D0,1.1038D0,1.14421D0,
     &  1.18762D0,1.23454D0,1.28603D0,1.34375D0,1.41016D0,1.48627D0,
     &  1.5732D0,1.67591D0,1.80226D0,1.96542D0,2.18962D0,2.52503D0,
     &  3.09663D0,4.33105D0,7.86807D0,1.D0,1.00876D0,1.03523D0,
     &  1.0697D0,1.10844D0,1.15028D0,1.19517D0,1.24368D0,1.29689D0,
     &  1.35652D0,1.42515D0,1.50383D0,1.5938D0,1.70026D0,1.83142D0,
     &  2.00119D0,2.23514D0,2.58661D0,3.18958D0,4.50914D0,8.50513D0,
     &  1.D0,1.00938D0,1.03711D0,1.07294D0,1.11306D0,1.15633D0,
     &  1.2027D0,1.25277D0,1.30769D0,1.36923D0,1.44006D0,1.52132D0,
     &  1.61433D0,1.72453D0,1.86054D0,2.03694D0,2.28072D0,2.64846D0,
     &  3.28339D0,4.6907D0,9.18863D0,1.D0,1.01001D0,1.03899D0,
     &  1.07617D0,1.11767D0,1.16235D0,1.21019D0,1.26182D0,1.31844D0,
     &  1.38189D0,1.45492D0,1.53874D0,1.63479D0,1.74874D0,1.8896D0,
     &  2.07268D0,2.32638D0,2.71058D0,3.37806D0,4.87575D0,9.92145D0,
     &  1.D0,1.01065D0,1.04088D0,1.0794D0,1.12227D0,1.16835D0,
     &  1.21765D0,1.27084D0,1.32915D0,1.39449D0,1.46971D0,1.55609D0,
     &  1.65518D0,1.77289D0,1.91863D0,2.10842D0,2.37213D0,2.77299D0,
     &  3.4736D0,5.06432D0,10.7066D0/
      DATA DX3/5.D0/,DY3/1.D0/,X03/5.D0/,Y03/0.D0/            
      DATA RI3GA/0.D0,0.00186D0,0.0037D0,0.00802D0,0.01695D0,
     &  0.03333D0,0.04732D0,0.04732D0,0.04732D0,0.04732D0,
     &  0.04732D0,0.04732D0,0.04732D0,0.04732D0,
     &  0.04732D0,0.04732D0,0.D0,0.00161D0,0.00319D0,
     &  0.00688D0,0.01448D0,0.02894D0,0.05445D0,0.09271D0,
     &  0.09977D0,0.09977D0,0.09977D0,0.09977D0,0.09977D0,
     &  0.09977D0,0.09977D0,0.09977D0,0.D0,0.00151D0,
     &  0.00299D0,0.00643D0,0.01351D0,0.02696D0,0.0508D0,
     &  0.09034D0,0.1443D0,0.14729D0,0.14729D0,
     &  0.14729D0,0.14729D0,0.14729D0,0.14729D0,0.14729D0,
     &  0.D0,0.00145D0,0.00287D0,0.00619D0,0.01302D0,
     &  0.026D0,0.04901D0,0.08744D0,0.1474D0,
     &  0.1929D0,0.1929D0,0.1929D0,0.1929D0,0.1929D0,
     &  0.1929D0,0.1929D0,0.D0,0.00142D0,
     &  0.00281D0,0.00605D0,0.01275D0,0.0255D0,0.04815D0,
     &  0.08606D0,0.14619D0,0.22959D0,0.2382D0,
     &  0.2382D0,0.2382D0,0.2382D0,0.2382D0,0.2382D0,
     &  0.D0,0.00139D0,0.00277D0,0.00597D0,0.0126D0,
     &  0.02526D0,0.0478D0,0.08563D0,0.14592D0,
     &  0.23598D0,0.28414D0,0.28414D0,0.28414D0,0.28414D0,
     &  0.28414D0,0.28414D0,0.D0,0.00138D0,0.00274D0,
     &  0.00593D0,0.01253D0,0.02517D0,0.04776D0,
     &  0.08579D0,0.14659D0,0.23905D0,0.33133D0,
     &  0.33133D0,0.33133D0,0.33133D0,0.33133D0,0.33133D0,
     &  0.D0,0.00137D0,0.00273D0,0.0059D0,0.0125D0,
     &  0.02518D0,0.04792D0,0.08634D0,0.14795D0,
     &  0.24232D0,0.37082D0,0.38022D0,0.38022D0,0.38022D0,
     &  0.38022D0,0.38022D0,0.D0,0.00137D0,0.00272D0,
     &  0.00589D0,0.01251D0,0.02526D0,0.04822D0,
     &  0.08717D0,0.14983D0,0.24621D0,0.38536D0,
     &  0.4312D0,0.4312D0,0.4312D0,0.4312D0,0.4312D0,
     &  0.D0,0.00136D0,0.00271D0,0.00589D0,0.01254D0,
     &  0.02539D0,0.04862D0,0.08819D0,0.1521D0,
     &  0.25072D0,0.39602D0,0.48462D0,0.48462D0,0.48462D0,
     &  0.48462D0,0.48462D0,0.D0,0.00136D0,0.00271D0,
     &  0.00591D0,0.01259D0,0.02556D0,0.0491D0,
     &  0.08937D0,0.15466D0,0.25577D0,0.40621D0,
     &  0.54079D0,0.54079D0,0.54079D0,0.54079D0,0.54079D0,
     &  0.D0,0.00136D0,0.00272D0,0.00592D0,0.01266D0,
     &  0.02576D0,0.04963D0,0.09066D0,0.15747D0,
     &  0.2613D0,0.41671D0,0.60004D0,0.60004D0,0.60004D0,
     &  0.60004D0,0.60004D0,0.D0,0.00137D0,0.00273D0,
     &  0.00595D0,0.01273D0,0.02598D0,0.05021D0,
     &  0.09204D0,0.16047D0,0.26723D0,0.42777D0,
     &  0.64814D0,0.66268D0,0.66268D0,0.66268D0,0.66268D0,
     &  0.D0,0.00137D0,0.00273D0,0.00597D0,0.01282D0,
     &  0.02622D0,0.05082D0,0.09351D0,0.16363D0,
     &  0.27352D0,0.43946D0,0.67619D0,0.72906D0,0.72906D0,
     &  0.72906D0,0.72906D0,0.D0,0.00137D0,0.00274D0,
     &  0.00601D0,0.01291D0,0.02647D0,0.05147D0,
     &  0.09504D0,0.16694D0,0.28014D0,0.45178D0,
     &  0.70102D0,0.7995D0,0.7995D0,0.7995D0,0.7995D0,
     &  0.D0,0.00138D0,0.00276D0,0.00604D0,0.01301D0,
     &  0.02674D0,0.05214D0,0.09662D0,0.17038D0,
     &  0.28707D0,0.46475D0,0.72544D0,0.87438D0,0.87438D0,
     &  0.87438D0,0.87438D0,0.D0,0.00138D0,0.00277D0,
     &  0.00608D0,0.01311D0,0.02702D0,0.05283D0,
     &  0.09825D0,0.17392D0,0.29429D0,0.47835D0,0.75034D0,
     &  0.95408D0,0.95408D0,0.95408D0,0.95408D0,0.D0,
     &  0.00139D0,0.00278D0,0.00611D0,0.01322D0,
     &  0.0273D0,0.05355D0,0.09992D0,0.17758D0,
     &  0.30178D0,0.49258D0,0.77612D0,1.03901D0,1.03901D0,
     &  1.03901D0,1.03901D0,0.D0,0.00139D0,0.0028D0,
     &  0.00615D0,0.01333D0,0.02759D0,0.05427D0,
     &  0.10163D0,0.18132D0,0.30954D0,0.50745D0,
     &  0.80297D0,1.12959D0,1.12959D0,1.12959D0,1.12959D0,
     &  0.D0,0.0014D0,0.00281D0,0.0062D0,0.01344D0,
     &  0.02789D0,0.05502D0,0.10338D0,0.18516D0,
     &  0.31755D0,0.52297D0,0.83102D0,1.2263D0,1.2263D0,
     &  1.2263D0,1.2263D0,0.D0,0.00141D0,
     &  0.00283D0,0.00624D0,0.01356D0,0.0282D0,0.05577D0,
     &  0.10515D0,0.18908D0,0.32583D0,0.53915D0,
     &  0.86038D0,1.30964D0,1.32961D0,1.32961D0,1.32961D0,
     &  0.D0,0.00141D0,0.00284D0,0.00628D0,0.01368D0,
     &  0.02851D0,0.05654D0,0.10696D0,0.19309D0,
     &  0.33437D0,0.55599D0,0.89111D0,1.37219D0,1.44004D0,
     &  1.44004D0,1.44004D0,0.D0,0.00142D0,0.00286D0,
     &  0.00633D0,0.0138D0,0.02882D0,0.05731D0,
     &  0.10879D0,0.19717D0,0.34316D0,0.57351D0,
     &  0.92328D0,1.43238D0,1.55816D0,1.55816D0,1.55816D0,
     &  0.D0,0.00143D0,0.00288D0,0.00637D0,0.01392D0,
     &  0.02914D0,0.0581D0,0.11064D0,0.20134D0,
     &  0.35222D0,0.59174D0,0.95696D0,1.49335D0,1.68456D0,
     &  1.68456D0,1.68456D0,0.D0,0.00144D0,0.0029D0,0.00642D0,
     &  0.01404D0,0.02946D0,0.05889D0,0.11252D0,
     &  0.20558D0,0.36155D0,0.6107D0,0.99223D0,1.55624D0,
     &  1.81983D0,1.81983D0,1.81983D0,0.D0,
     &  0.00144D0,0.00291D0,0.00646D0,0.01417D0,0.02978D0,
     &  0.05969D0,0.11443D0,0.20989D0,0.37115D0,0.63041D0,
     &  1.02915D0,1.62165D0,1.96468D0,1.96468D0,
     &  1.96468D0,0.D0,0.00145D0,0.00293D0,0.00651D0,
     &  0.0143D0,0.03011D0,0.0605D0,0.11635D0,0.21428D0,
     &  0.38103D0,0.6509D0,1.0678D0,1.68997D0,2.11979D0,
     &  2.11979D0,2.11979D0,0.D0,0.00146D0,0.00295D0,
     &  0.00656D0,0.01442D0,0.03044D0,0.06132D0,0.1183D0,
     &  0.21874D0,0.39119D0,0.67221D0,1.10826D0,1.76151D0,
     &  2.28593D0,2.28593D0,2.28593D0,0.D0,0.00147D0,
     &  0.00297D0,0.00661D0,0.01455D0,0.03077D0,0.06214D0,
     &  0.12026D0,0.22328D0,0.40165D0,0.69435D0,1.1506D0,
     &  1.83651D0,2.46389D0,2.46389D0,2.46389D0,0.D0,
     &  0.00148D0,0.00299D0,0.00666D0,0.01468D0,0.03111D0,
     &  0.06297D0,0.12225D0,0.22789D0,0.41242D0,0.71737D0,
     &  1.19493D0,1.91521D0,2.6545D0,2.6545D0,2.6545D0/
      DATA RI3GB/0.D0,0.08401D0,0.14638D0,0.22297D0,0.29921D0,
     &  0.3667D0,0.39779D0,0.39779D0,0.39779D0,0.39779D0,
     &  0.39779D0,0.39779D0,0.39779D0,0.39779D0,0.39779D0,
     &  0.39779D0,0.D0,0.08822D0,0.16918D0,0.27107D0,
     &  0.37626D0,0.47827D0,0.57519D0,0.65659D0,0.66657D0,
     &  0.66657D0,0.66657D0,0.66657D0,0.66657D0,0.66657D0,
     &  0.66657D0,0.66657D0,0.D0,0.0897D0,0.18101D0,0.29714D0,
     &  0.41849D0,0.53772D0,0.65403D0,0.76747D0,0.86138D0,
     &  0.86503D0,0.86503D0,0.86503D0,0.86503D0,0.86503D0,
     &  0.86503D0,0.86503D0,0.D0,0.09062D0,0.18907D0,
     &  0.31513D0,0.4479D0,0.57937D0,0.70871D0,0.83782D0,
     &  0.96514D0,1.03075D0,1.03075D0,1.03075D0,1.03075D0,
     &  1.03075D0,1.03075D0,1.03075D0,0.D0,0.09135D0,
     &  0.19524D0,0.32891D0,0.47056D0,0.61167D0,0.75128D0,
     &  0.8919D0,1.0358D0,1.16789D0,1.17802D0,1.17802D0,
     &  1.17802D0,1.17802D0,1.17802D0,1.17802D0,0.D0,
     &  0.09199D0,0.20027D0,0.34013D0,0.48906D0,0.63818D0,
     &  0.78643D0,0.9366D0,1.09257D0,1.25149D0,1.31392D0,
     &  1.31392D0,1.31392D0,1.31392D0,1.31392D0,1.31392D0,
     &  0.D0,0.09257D0,0.20454D0,0.34961D0,0.50472D0,
     &  0.66073D0,0.81651D0,0.97506D0,1.14113D0,1.31665D0,
     &  1.44255D0,1.44255D0,1.44255D0,1.44255D0,
     &  1.44255D0,1.44255D0,0.D0,0.09312D0,0.20827D0,
     &  0.35785D0,0.51835D0,0.68042D0,0.84293D0,1.00903D0,
     &  1.1841D0,1.37257D0,1.55605D0,1.56653D0,
     &  1.56653D0,1.56653D0,1.56653D0,1.56653D0,0.D0,
     &  0.09365D0,0.21159D0,0.36516D0,0.53044D0,
     &  0.69794D0,0.86655D0,1.0396D0,1.22295D0,1.42262D0,
     &  1.63327D0,1.68771D0,1.68771D0,1.68771D0,
     &  1.68771D0,1.68771D0,0.D0,0.09416D0,0.21459D0,
     &  0.37173D0,0.54132D0,0.71376D0,0.88797D0,1.06751D0,
     &  1.25863D0,1.46853D0,1.69818D0,1.80743D0,1.80743D0,
     &  1.80743D0,1.80743D0,1.80743D0,0.D0,0.09465D0,
     &  0.21734D0,0.37772D0,0.55124D0,0.72821D0,0.90763D0,
     &  1.09326D0,1.29177D0,1.5113D0,1.75673D0,1.92679D0,
     &  1.92679D0,1.92679D0,1.92679D0,1.92679D0,0.D0,
     &  0.09513D0,0.21988D0,0.38324D0,0.56038D0,0.74154D0,
     &  0.92584D0,1.11725D0,1.32286D0,1.55164D0,1.8112D0,
     &  2.04668D0,2.04668D0,2.04668D0,2.04668D0,
     &  2.04668D0,0.D0,0.0956D0,0.22226D0,0.38835D0,
     &  0.56885D0,0.75394D0,0.94282D0,1.13974D0,
     &  1.35221D0,1.58999D0,1.8628D0,2.15217D0,2.16786D0,
     &  2.16786D0,2.16786D0,2.16786D0,0.D0,0.09606D0,
     &  0.22448D0,0.39313D0,0.57676D0,0.76554D0,
     &  0.95877D0,1.16098D0,1.38011D0,1.62672D0,1.91229D0,
     &  2.2318D0,2.29105D0,2.29105D0,2.29105D0,
     &  2.29105D0,0.D0,0.09651D0,0.22658D0,0.39762D0,
     &  0.5842D0,0.77646D0,0.97383D0,1.18113D0,
     &  1.40676D0,1.66209D0,1.96017D0,2.30384D0,
     &  2.41683D0,2.41683D0,2.41683D0,2.41683D0,0.D0,
     &  0.09695D0,0.22856D0,0.40185D0,0.59121D0,0.78678D0,
     &  0.98811D0,1.20034D0,1.43234D0,1.69633D0,2.00682D0,
     &  2.37216D0,2.54588D0,2.54588D0,2.54588D0,
     &  2.54588D0,0.D0,0.09739D0,0.23046D0,0.40587D0,
     &  0.59787D0,0.79659D0,1.00173D0,1.21872D0,
     &  1.45697D0,1.72961D0,2.05252D0,2.43832D0,2.67876D0,
     &  2.67876D0,2.67876D0,2.67876D0,0.D0,0.09781D0,
     &  0.23227D0,0.40969D0,0.60421D0,0.80595D0,
     &  1.01474D0,1.23638D0,1.48079D0,1.76206D0,2.0975D0,
     &  2.50321D0,2.81605D0,2.81605D0,2.81605D0,
     &  2.81605D0,0.D0,0.09823D0,0.234D0,0.41334D0,
     &  0.61026D0,0.81489D0,1.02723D0,1.2534D0,
     &  1.50388D0,1.7938D0,2.14192D0,2.56741D0,2.95834D0,
     &  2.95834D0,2.95834D0,2.95834D0,0.D0,
     &  0.09865D0,0.23566D0,0.41684D0,0.61605D0,0.82347D0,
     &  1.03924D0,1.26983D0,1.52633D0,1.82496D0,2.18597D0,
     &  2.63133D0,3.10624D0,3.10624D0,3.10624D0,
     &  3.10624D0,0.D0,0.09906D0,0.23727D0,0.4202D0,
     &  0.62162D0,0.83172D0,1.05082D0,1.28575D0,
     &  1.5482D0,1.8556D0,2.22976D0,2.69528D0,3.23944D0,
     &  3.26037D0,3.26037D0,3.26037D0,0.D0,
     &  0.09946D0,0.23882D0,0.42343D0,0.62697D0,
     &  0.83968D0,1.06202D0,1.30119D0,1.56956D0,
     &  1.88581D0,2.27341D0,2.75952D0,3.34852D0,3.42133D0,
     &  3.42133D0,3.42133D0,0.D0,0.09986D0,
     &  0.24031D0,0.42655D0,0.63214D0,0.84737D0,1.07286D0,
     &  1.31621D0,1.59046D0,1.91567D0,2.31705D0,
     &  2.82428D0,3.45289D0,3.58982D0,3.58982D0,3.58982D0,
     &  0.D0,0.10025D0,0.24177D0,0.42957D0,0.63714D0,
     &  0.85481D0,1.08338D0,1.33085D0,1.61094D0,
     &  1.94523D0,2.36075D0,2.88975D0,3.55625D0,3.76652D0,
     &  3.76652D0,3.76652D0,0.D0,0.10064D0,
     &  0.24318D0,0.43248D0,0.64198D0,0.86202D0,1.09361D0,
     &  1.34513D0,1.63106D0,1.97455D0,2.40461D0,
     &  2.95609D0,3.6601D0,3.95212D0,3.95212D0,3.95212D0,
     &  0.D0,0.10103D0,0.24455D0,0.43531D0,0.64667D0,
     &  0.86903D0,1.10358D0,1.35909D0,1.65083D0,
     &  2.00368D0,2.44872D0,3.02349D0,3.76532D0,4.14741D0,
     &  4.14741D0,4.14741D0,0.D0,0.10141D0,0.24588D0,
     &  0.43807D0,0.65123D0,0.87585D0,1.11329D0,
     &  1.37275D0,1.67031D0,2.03267D0,2.49317D0,
     &  3.0921D0,3.87253D0,4.35317D0,4.35317D0,4.35317D0,
     &  0.D0,0.10179D0,0.24718D0,0.44074D0,0.65567D0,
     &  0.88245D0,1.12277D0,1.38614D0,1.68951D0,2.06156D0,
     &  2.53801D0,3.16207D0,3.98217D0,4.57024D0,
     &  4.57024D0,4.57024D0,0.D0,0.10216D0,0.24845D0,
     &  0.44334D0,0.65998D0,0.88896D0,1.13204D0,
     &  1.39928D0,1.70846D0,2.09039D0,2.58334D0,
     &  3.23353D0,4.09464D0,4.79949D0,4.79949D0,4.79949D0,
     &  0.D0,0.10253D0,0.24969D0,0.44588D0,0.66419D0,
     &  0.89528D0,1.14111D0,1.41218D0,1.7272D0,2.11921D0,
     &  2.62922D0,3.30664D0,4.21031D0,5.04184D0,
     &  5.04184D0,5.04184D0/     
      DATA DX4/0.996D-02/,DY4/0.124292D0/,X04/0.2D-02/,Y04/3.58352D0/      
      DATA FT4/0.0D+00,0.0D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.67391D+02,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.39393D+02,0.42077D+02,
     & 0.45088D+02,0.48520D+02,0.52316D+02,0.56383D+02,0.60690D+02,
     & 0.65223D+02,0.69972D+02,0.74933D+02,0.80103D+02,0.85480D+02,
     & 0.91061D+02,0.96847D+02,0.10284D+03,0.10903D+03,0.11542D+03,
     & 0.12202D+03,0.12882D+03,0.13583D+03,0.14304D+03,0.15045D+03,
     & 0.15807D+03,0.16591D+03,0.17646D+03,0.18440D+03,0.18687D+03,
     & 0.18103D+03,0.15688D+03,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.16969D+02,
     & 0.18150D+02,0.19477D+02,0.20902D+02,0.22416D+02,0.24014D+02,
     & 0.25693D+02,0.27453D+02,0.29294D+02,0.31215D+02,0.33216D+02,
     & 0.35296D+02,0.37453D+02,0.39679D+02,0.41983D+02,0.44370D+02,
     & 0.46846D+02,0.49412D+02,0.52071D+02,0.54825D+02,0.57677D+02,
     & 0.60630D+02,0.63684D+02,0.66844D+02,0.70111D+02,0.73488D+02,
     & 0.76977D+02,0.80580D+02,0.84302D+02,0.88143D+02,0.92107D+02,
     & 0.96193D+02,0.10041D+03,0.10478D+03,0.11240D+03,0.11664D+03,
     & 0.11393D+03,0.10074D+03,0.69647D+02,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.90052D+01,0.94235D+01,
     & 0.99804D+01,0.10859D+02,0.12038D+02,0.13256D+02,0.14192D+02,
     & 0.15080D+02,0.16008D+02,0.17009D+02,0.18056D+02,0.19151D+02,
     & 0.20296D+02,0.21492D+02,0.22741D+02,0.24046D+02,0.25407D+02,
     & 0.26826D+02,0.28305D+02,0.29847D+02,0.31451D+02,0.33121D+02,
     & 0.34859D+02,0.36665D+02,0.38542D+02,0.40492D+02,0.42517D+02,
     & 0.44619D+02,0.46800D+02,0.49062D+02,0.51410D+02,0.53839D+02,
     & 0.56358D+02,0.58968D+02,0.61670D+02,0.64467D+02,0.67362D+02,
     & 0.70357D+02,0.73451D+02,0.76651D+02,0.79970D+02,0.86246D+02,
     & 0.89434D+02,0.86312D+02,0.73658D+02,0.45611D+02,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.68600D+01,0.72406D+01,0.76356D+01,0.80552D+01,0.84924D+01,
     & 0.89159D+01,0.93890D+01,0.99931D+01,0.10721D+02,0.11477D+02,
     & 0.12135D+02,0.12788D+02,0.13470D+02,0.14194D+02,0.14949D+02,
     & 0.15737D+02,0.16559D+02,0.17415D+02,0.18307D+02,0.19236D+02,
     & 0.20204D+02,0.21211D+02,0.22258D+02,0.23347D+02,0.24480D+02,
     & 0.25656D+02,0.26878D+02,0.28149D+02,0.29465D+02,0.30832D+02,
     & 0.32250D+02,0.33721D+02,0.35245D+02,0.36826D+02,0.38463D+02,
     & 0.40158D+02,0.41914D+02,0.43732D+02,0.45613D+02,0.47560D+02,
     & 0.49573D+02,0.51655D+02,0.53805D+02,0.56027D+02,0.58332D+02,
     & 0.62752D+02,0.64956D+02,0.62628D+02,0.53451D+02,0.33115D+02,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.49956D+01,0.53037D+01,0.56835D+01,
     & 0.60757D+01,0.64064D+01,0.67298D+01,0.70686D+01,0.74294D+01,
     & 0.78059D+01,0.82032D+01,0.86113D+01,0.90204D+01,0.94327D+01,
     & 0.98629D+01,0.10328D+02,0.10815D+02,0.11321D+02,0.11844D+02,
     & 0.12387D+02,0.12949D+02,0.13532D+02,0.14137D+02,0.14762D+02,
     & 0.15410D+02,0.16081D+02,0.16775D+02,0.17494D+02,0.18236D+02,
     & 0.19004D+02,0.19798D+02,0.20619D+02,0.21467D+02,0.22343D+02,
     & 0.23248D+02,0.24183D+02,0.25148D+02,0.26145D+02,0.27173D+02,
     & 0.28234D+02,0.29329D+02,0.30458D+02,0.31623D+02,0.32824D+02,
     & 0.34062D+02,0.35338D+02,0.36654D+02,0.38008D+02,0.39404D+02,
     & 0.40846D+02,0.43549D+02,0.44924D+02,0.43589D+02,0.38158D+02,
     & 0.25652D+02,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.41564D+01,0.43425D+01,0.45280D+01,0.47293D+01,0.49696D+01,
     & 0.52493D+01,0.55407D+01,0.58079D+01,0.60767D+01,0.63572D+01,
     & 0.66531D+01,0.69615D+01,0.72888D+01,0.76226D+01,0.79473D+01,
     & 0.82648D+01,0.85951D+01,0.89632D+01,0.93517D+01,0.97541D+01,
     & 0.10169D+02,0.10598D+02,0.11042D+02,0.11501D+02,0.11976D+02,
     & 0.12467D+02,0.12975D+02,0.13499D+02,0.14041D+02,0.14601D+02,
     & 0.15179D+02,0.15777D+02,0.16393D+02,0.17029D+02,0.17686D+02,
     & 0.18363D+02,0.19062D+02,0.19783D+02,0.20527D+02,0.21294D+02,
     & 0.22084D+02,0.22899D+02,0.23739D+02,0.24604D+02,0.25496D+02,
     & 0.26415D+02,0.27361D+02,0.28336D+02,0.29340D+02,0.30373D+02,
     & 0.31436D+02,0.32534D+02,0.34596D+02,0.35641D+02,0.34610D+02,
     & 0.30443D+02,0.20756D+02,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.35918D+01,
     & 0.37597D+01,0.39205D+01,0.40839D+01,0.42582D+01,0.44393D+01,
     & 0.46262D+01,0.48196D+01,0.50213D+01,0.52329D+01,0.54533D+01,
     & 0.56823D+01,0.59201D+01,0.61669D+01,0.64232D+01,0.66890D+01,
     & 0.69649D+01,0.72516D+01,0.75476D+01,0.78549D+01,0.81727D+01,
     & 0.85021D+01,0.88434D+01,0.91955D+01,0.95603D+01,0.99375D+01,
     & 0.10327D+02,0.10731D+02,0.11147D+02,0.11578D+02,0.12022D+02,
     & 0.12481D+02,0.12956D+02,0.13445D+02,0.13951D+02,0.14472D+02,
     & 0.15011D+02,0.15566D+02,0.16140D+02,0.16730D+02,0.17340D+02,
     & 0.17969D+02,0.18617D+02,0.19285D+02,0.19974D+02,0.20684D+02,
     & 0.21416D+02,0.22170D+02,0.22947D+02,0.23747D+02,0.24572D+02,
     & 0.25420D+02,0.26293D+02,0.27195D+02,0.28899D+02,0.29756D+02,
     & 0.28889D+02,0.25416D+02,0.17324D+02,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.29802D+01,0.30922D+01,0.32229D+01,
     & 0.33613D+01,0.35004D+01,0.36386D+01,0.37811D+01,0.39327D+01,
     & 0.40892D+01,0.42472D+01,0.44072D+01,0.45737D+01,0.47521D+01,
     & 0.49388D+01,0.51326D+01,0.53331D+01,0.55410D+01,0.57565D+01,
     & 0.59799D+01,0.62115D+01,0.64510D+01,0.66992D+01,0.69566D+01,
     & 0.72222D+01,0.74974D+01,0.77821D+01,0.80766D+01,0.83811D+01,
     & 0.86959D+01,0.90213D+01,0.93576D+01,0.97051D+01,0.10064D+02,
     & 0.10435D+02,0.10818D+02,0.11214D+02,0.11622D+02,0.12044D+02,
     & 0.12479D+02,0.12928D+02,0.13392D+02,0.13870D+02,0.14364D+02,
     & 0.14873D+02,0.15398D+02,0.15939D+02,0.16498D+02,0.17073D+02,
     & 0.17667D+02,0.18278D+02,0.18909D+02,0.19559D+02,0.20228D+02,
     & 0.20918D+02,0.21627D+02,0.22358D+02,0.23114D+02,0.24541D+02,
     & 0.25260D+02,0.24532D+02,0.21622D+02,0.14800D+02,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.26312D+01,0.27281D+01,0.28242D+01,0.29231D+01,
     & 0.30267D+01,0.31349D+01,0.32478D+01,0.33655D+01,0.34883D+01,
     & 0.36163D+01,0.37496D+01,0.38877D+01,0.40303D+01,0.41779D+01,
     & 0.43319D+01,0.44920D+01,0.46579D+01,0.48299D+01,0.50081D+01,
     & 0.51926D+01,0.53836D+01,0.55816D+01,0.57863D+01,0.59982D+01,
     & 0.62174D+01,0.64443D+01,0.66788D+01,0.69214D+01,0.71721D+01,
     & 0.74312D+01,0.76990D+01,0.79756D+01,0.82615D+01,0.85568D+01,
     & 0.88617D+01,0.91767D+01,0.95017D+01,0.98374D+01,0.10184D+02,
     & 0.10542D+02,0.10910D+02,0.11291D+02,0.11684D+02,0.12090D+02,
     & 0.12508D+02,0.12939D+02,0.13384D+02,0.13842D+02,0.14315D+02,
     & 0.14803D+02,0.15305D+02,0.15823D+02,0.16357D+02,0.16907D+02,
     & 0.17474D+02,0.18057D+02,0.18658D+02,0.19278D+02,0.19917D+02,
     & 0.21123D+02,0.21732D+02,0.21121D+02,0.18668D+02,0.12874D+02,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.24035D+01,0.24850D+01,0.25685D+01,0.26535D+01,
     & 0.27411D+01,0.28311D+01,0.29246D+01,0.30227D+01,0.31256D+01,
     & 0.32328D+01,0.33440D+01,0.34595D+01,0.35795D+01,0.37040D+01,
     & 0.38332D+01,0.39672D+01,0.41061D+01,0.42501D+01,0.43993D+01,
     & 0.45538D+01,0.47137D+01,0.48793D+01,0.50506D+01,0.52280D+01,
     & 0.54114D+01,0.56012D+01,0.57974D+01,0.60003D+01,0.62100D+01,
     & 0.64268D+01,0.66508D+01,0.68823D+01,0.71214D+01,0.73685D+01,
     & 0.76236D+01,0.78871D+01,0.81592D+01,0.84402D+01,0.87302D+01,
     & 0.90296D+01,0.93386D+01,0.96575D+01,0.99866D+01,0.10326D+02,
     & 0.10676D+02,0.11038D+02,0.11410D+02,0.11795D+02,0.12191D+02,
     & 0.12600D+02,0.13022D+02,0.13456D+02,0.13904D+02,0.14366D+02,
     & 0.14842D+02,0.15332D+02,0.15837D+02,0.16358D+02,0.16894D+02,
     & 0.17448D+02,0.18490D+02,0.19017D+02,0.18493D+02,0.16381D+02,
     & 0.11361D+02,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.21434D+01,0.22103D+01,0.22761D+01,0.23439D+01,0.24147D+01,
     & 0.24890D+01,0.25664D+01,0.26471D+01,0.27310D+01,0.28182D+01,
     & 0.29090D+01,0.30034D+01,0.31014D+01,0.32031D+01,0.33087D+01,
     & 0.34182D+01,0.35319D+01,0.36495D+01,0.37714D+01,0.38977D+01,
     & 0.40286D+01,0.41640D+01,0.43042D+01,0.44493D+01,0.45993D+01,
     & 0.47545D+01,0.49151D+01,0.50810D+01,0.52526D+01,0.54300D+01,
     & 0.56132D+01,0.58026D+01,0.59983D+01,0.62004D+01,0.64092D+01,
     & 0.66248D+01,0.68475D+01,0.70774D+01,0.73148D+01,0.75599D+01,
     & 0.78128D+01,0.80739D+01,0.83434D+01,0.86214D+01,0.89083D+01,
     & 0.92043D+01,0.95097D+01,0.98247D+01,0.10150D+02,0.10485D+02,
     & 0.10830D+02,0.11187D+02,0.11554D+02,0.11933D+02,0.12323D+02,
     & 0.12726D+02,0.13141D+02,0.13568D+02,0.14009D+02,0.14462D+02,
     & 0.14929D+02,0.15412D+02,0.16318D+02,0.16778D+02,0.16330D+02,
     & 0.14509D+02,0.10143D+02,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.19277D+01,
     & 0.19804D+01,0.20359D+01,0.20936D+01,0.21518D+01,0.22119D+01,
     & 0.22749D+01,0.23410D+01,0.24100D+01,0.24817D+01,0.25563D+01,
     & 0.26339D+01,0.27145D+01,0.27983D+01,0.28853D+01,0.29755D+01,
     & 0.30693D+01,0.31662D+01,0.32669D+01,0.33712D+01,0.34792D+01,
     & 0.35911D+01,0.37069D+01,0.38268D+01,0.39508D+01,0.40792D+01,
     & 0.42119D+01,0.43492D+01,0.44912D+01,0.46379D+01,0.47896D+01,
     & 0.49464D+01,0.51084D+01,0.52758D+01,0.54487D+01,0.56273D+01,
     & 0.58118D+01,0.60023D+01,0.61989D+01,0.64020D+01,0.66116D+01,
     & 0.68280D+01,0.70514D+01,0.72819D+01,0.75199D+01,0.77654D+01,
     & 0.80187D+01,0.82802D+01,0.85498D+01,0.88280D+01,0.91151D+01,
     & 0.94110D+01,0.97162D+01,0.10031D+02,0.10356D+02,0.10690D+02,
     & 0.11035D+02,0.11391D+02,0.11758D+02,0.12135D+02,0.12525D+02,
     & 0.12926D+02,0.13339D+02,0.13766D+02,0.14564D+02,0.14971D+02,
     & 0.14583D+02,0.12992D+02,0.91437D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.17935D+01,
     & 0.18391D+01,0.18858D+01,0.19333D+01,0.19829D+01,0.20350D+01,
     & 0.20894D+01,0.21462D+01,0.22055D+01,0.22671D+01,0.23314D+01,
     & 0.23981D+01,0.24676D+01,0.25398D+01,0.26146D+01,0.26924D+01,
     & 0.27730D+01,0.28567D+01,0.29434D+01,0.30333D+01,0.31263D+01,
     & 0.32227D+01,0.33225D+01,0.34258D+01,0.35327D+01,0.36433D+01,
     & 0.37577D+01,0.38760D+01,0.39983D+01,0.41247D+01,0.42554D+01,
     & 0.43905D+01,0.45301D+01,0.46743D+01,0.48233D+01,0.49772D+01,
     & 0.51361D+01,0.53003D+01,0.54699D+01,0.56449D+01,0.58257D+01,
     & 0.60123D+01,0.62049D+01,0.64038D+01,0.66091D+01,0.68209D+01,
     & 0.70395D+01,0.72652D+01,0.74979D+01,0.77380D+01,0.79858D+01,
     & 0.82416D+01,0.85051D+01,0.87770D+01,0.90575D+01,0.93467D+01,
     & 0.96450D+01,0.99525D+01,0.10269D+02,0.10596D+02,0.10933D+02,
     & 0.11281D+02,0.11638D+02,0.12007D+02,0.12388D+02,0.13096D+02,
     & 0.13460D+02,0.13123D+02,0.11724D+02,0.83103D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.16748D+01,
     & 0.17127D+01,0.17526D+01,0.17939D+01,0.18366D+01,0.18812D+01,
     & 0.19281D+01,0.19771D+01,0.20283D+01,0.20815D+01,0.21371D+01,
     & 0.21948D+01,0.22550D+01,0.23174D+01,0.23823D+01,0.24496D+01,
     & 0.25196D+01,0.25921D+01,0.26673D+01,0.27452D+01,0.28260D+01,
     & 0.29096D+01,0.29962D+01,0.30859D+01,0.31786D+01,0.32747D+01,
     & 0.33740D+01,0.34766D+01,0.35828D+01,0.36926D+01,0.38061D+01,
     & 0.39234D+01,0.40446D+01,0.41698D+01,0.42992D+01,0.44329D+01,
     & 0.45710D+01,0.47136D+01,0.48608D+01,0.50129D+01,0.51700D+01,
     & 0.53321D+01,0.54995D+01,0.56724D+01,0.58508D+01,0.60349D+01,
     & 0.62249D+01,0.64211D+01,0.66237D+01,0.68324D+01,0.70479D+01,
     & 0.72703D+01,0.74997D+01,0.77364D+01,0.79805D+01,0.82324D+01,
     & 0.84921D+01,0.87599D+01,0.90361D+01,0.93209D+01,0.96146D+01,
     & 0.99174D+01,0.10229D+02,0.10551D+02,0.10882D+02,0.11225D+02,
     & 0.11858D+02,0.12186D+02,0.11891D+02,0.10654D+02,0.76053D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.15394D+01,0.15700D+01,
     & 0.16024D+01,0.16358D+01,0.16709D+01,0.17077D+01,0.17465D+01,
     & 0.17870D+01,0.18295D+01,0.18737D+01,0.19200D+01,0.19682D+01,
     & 0.20185D+01,0.20706D+01,0.21249D+01,0.21814D+01,0.22401D+01,
     & 0.23010D+01,0.23642D+01,0.24297D+01,0.24977D+01,0.25681D+01,
     & 0.26410D+01,0.27166D+01,0.27948D+01,0.28758D+01,0.29596D+01,
     & 0.30463D+01,0.31360D+01,0.32287D+01,0.33246D+01,0.34237D+01,
     & 0.35262D+01,0.36321D+01,0.37416D+01,0.38547D+01,0.39715D+01,
     & 0.40922D+01,0.42169D+01,0.43457D+01,0.44787D+01,0.46161D+01,
     & 0.47581D+01,0.49044D+01,0.50557D+01,0.52118D+01,0.53731D+01,
     & 0.55394D+01,0.57112D+01,0.58884D+01,0.60714D+01,0.62602D+01,
     & 0.64551D+01,0.66562D+01,0.68637D+01,0.70778D+01,0.72986D+01,
     & 0.75264D+01,0.77615D+01,0.80039D+01,0.82539D+01,0.85118D+01,
     & 0.87777D+01,0.90518D+01,0.93345D+01,0.96258D+01,0.99261D+01,
     & 0.10237D+02,0.10807D+02,0.11105D+02,0.10845D+02,0.97435D+01,
     & 0.70016D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.14514D+01,0.14769D+01,
     & 0.15040D+01,0.15327D+01,0.15629D+01,0.15947D+01,0.16280D+01,
     & 0.16631D+01,0.16999D+01,0.17384D+01,0.17786D+01,0.18205D+01,
     & 0.18642D+01,0.19098D+01,0.19572D+01,0.20066D+01,0.20578D+01,
     & 0.21111D+01,0.21664D+01,0.22238D+01,0.22833D+01,0.23450D+01,
     & 0.24090D+01,0.24752D+01,0.25438D+01,0.26149D+01,0.26884D+01,
     & 0.27645D+01,0.28432D+01,0.29246D+01,0.30088D+01,0.30959D+01,
     & 0.31859D+01,0.32789D+01,0.33751D+01,0.34744D+01,0.35771D+01,
     & 0.36831D+01,0.37927D+01,0.39059D+01,0.40229D+01,0.41436D+01,
     & 0.42684D+01,0.43973D+01,0.45302D+01,0.46675D+01,0.48093D+01,
     & 0.49557D+01,0.51068D+01,0.52628D+01,0.54239D+01,0.55901D+01,
     & 0.57617D+01,0.59388D+01,0.61216D+01,0.63102D+01,0.65048D+01,
     & 0.67056D+01,0.69128D+01,0.71265D+01,0.73470D+01,0.75745D+01,
     & 0.78091D+01,0.80511D+01,0.83007D+01,0.85581D+01,0.88232D+01,
     & 0.90967D+01,0.93795D+01,0.98961D+01,0.10168D+02,0.99394D+01,
     & 0.89547D+01,0.64789D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.13724D+01,0.13941D+01,
     & 0.14171D+01,0.14415D+01,0.14673D+01,0.14946D+01,0.15235D+01,
     & 0.15538D+01,0.15857D+01,0.16192D+01,0.16541D+01,0.16906D+01,
     & 0.17288D+01,0.17687D+01,0.18102D+01,0.18534D+01,0.18984D+01,
     & 0.19451D+01,0.19937D+01,0.20441D+01,0.20965D+01,0.21507D+01,
     & 0.22070D+01,0.22653D+01,0.23258D+01,0.23883D+01,0.24531D+01,
     & 0.25202D+01,0.25896D+01,0.26614D+01,0.27357D+01,0.28125D+01,
     & 0.28919D+01,0.29740D+01,0.30589D+01,0.31466D+01,0.32373D+01,
     & 0.33311D+01,0.34278D+01,0.35278D+01,0.36311D+01,0.37379D+01,
     & 0.38481D+01,0.39620D+01,0.40796D+01,0.42011D+01,0.43265D+01,
     & 0.44560D+01,0.45897D+01,0.47278D+01,0.48703D+01,0.50175D+01,
     & 0.51694D+01,0.53262D+01,0.54881D+01,0.56551D+01,0.58276D+01,
     & 0.60055D+01,0.61892D+01,0.63787D+01,0.65742D+01,0.67760D+01,
     & 0.69842D+01,0.71989D+01,0.74205D+01,0.76490D+01,0.78847D+01,
     & 0.81276D+01,0.83782D+01,0.86374D+01,0.91078D+01,0.93573D+01,
     & 0.91552D+01,0.82704D+01,0.60221D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.13018D+01,0.13199D+01,
     & 0.13394D+01,0.13602D+01,0.13823D+01,0.14058D+01,0.14307D+01,
     & 0.14570D+01,0.14846D+01,0.15136D+01,0.15441D+01,0.15760D+01,
     & 0.16094D+01,0.16443D+01,0.16806D+01,0.17186D+01,0.17581D+01,
     & 0.17992D+01,0.18420D+01,0.18864D+01,0.19326D+01,0.19805D+01,
     & 0.20301D+01,0.20816D+01,0.21350D+01,0.21904D+01,0.22477D+01,
     & 0.23070D+01,0.23684D+01,0.24320D+01,0.24977D+01,0.25658D+01,
     & 0.26361D+01,0.27089D+01,0.27841D+01,0.28619D+01,0.29423D+01,
     & 0.30253D+01,0.31112D+01,0.31999D+01,0.32915D+01,0.33862D+01,
     & 0.34841D+01,0.35851D+01,0.36895D+01,0.37973D+01,0.39087D+01,
     & 0.40237D+01,0.41425D+01,0.42652D+01,0.43918D+01,0.45226D+01,
     & 0.46576D+01,0.47971D+01,0.49410D+01,0.50897D+01,0.52431D+01,
     & 0.54014D+01,0.55649D+01,0.57336D+01,0.59078D+01,0.60875D+01,
     & 0.62730D+01,0.64644D+01,0.66618D+01,0.68656D+01,0.70758D+01,
     & 0.72927D+01,0.75163D+01,0.77469D+01,0.79855D+01,0.84157D+01,
     & 0.86460D+01,0.84669D+01,0.76691D+01,0.56194D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.12382D+01,0.12534D+01,
     & 0.12698D+01,0.12874D+01,0.13063D+01,0.13264D+01,0.13479D+01,
     & 0.13706D+01,0.13945D+01,0.14198D+01,0.14464D+01,0.14743D+01,
     & 0.15035D+01,0.15341D+01,0.15660D+01,0.15994D+01,0.16342D+01,
     & 0.16705D+01,0.17082D+01,0.17474D+01,0.17882D+01,0.18306D+01,
     & 0.18745D+01,0.19202D+01,0.19675D+01,0.20166D+01,0.20674D+01,
     & 0.21200D+01,0.21746D+01,0.22310D+01,0.22894D+01,0.23499D+01,
     & 0.24125D+01,0.24771D+01,0.25440D+01,0.26132D+01,0.26847D+01,
     & 0.27587D+01,0.28351D+01,0.29141D+01,0.29957D+01,0.30800D+01,
     & 0.31672D+01,0.32572D+01,0.33502D+01,0.34463D+01,0.35456D+01,
     & 0.36482D+01,0.37541D+01,0.38635D+01,0.39765D+01,0.40932D+01,
     & 0.42137D+01,0.43381D+01,0.44666D+01,0.45993D+01,0.47363D+01,
     & 0.48778D+01,0.50239D+01,0.51746D+01,0.53303D+01,0.54910D+01,
     & 0.56569D+01,0.58281D+01,0.60048D+01,0.61871D+01,0.63753D+01,
     & 0.65695D+01,0.67699D+01,0.69765D+01,0.71897D+01,0.74103D+01,
     & 0.78054D+01,0.80187D+01,0.78597D+01,0.71377D+01,0.52614D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.11807D+01,0.11932D+01,
     & 0.12069D+01,0.12218D+01,0.12378D+01,0.12550D+01,0.12734D+01,
     & 0.12930D+01,0.13138D+01,0.13358D+01,0.13589D+01,0.13833D+01,
     & 0.14090D+01,0.14358D+01,0.14640D+01,0.14934D+01,0.15241D+01,
     & 0.15561D+01,0.15895D+01,0.16242D+01,0.16604D+01,0.16980D+01,
     & 0.17370D+01,0.17775D+01,0.18196D+01,0.18632D+01,0.19085D+01,
     & 0.19553D+01,0.20039D+01,0.20542D+01,0.21062D+01,0.21601D+01,
     & 0.22159D+01,0.22736D+01,0.23333D+01,0.23950D+01,0.24589D+01,
     & 0.25249D+01,0.25931D+01,0.26637D+01,0.27366D+01,0.28119D+01,
     & 0.28898D+01,0.29703D+01,0.30535D+01,0.31394D+01,0.32282D+01,
     & 0.33200D+01,0.34148D+01,0.35127D+01,0.36138D+01,0.37183D+01,
     & 0.38262D+01,0.39376D+01,0.40528D+01,0.41717D+01,0.42945D+01,
     & 0.44213D+01,0.45522D+01,0.46875D+01,0.48271D+01,0.49713D+01,
     & 0.51202D+01,0.52739D+01,0.54325D+01,0.55964D+01,0.57654D+01,
     & 0.59400D+01,0.61201D+01,0.63060D+01,0.64977D+01,0.66956D+01,
     & 0.69004D+01,0.72648D+01,0.74632D+01,0.73216D+01,0.66657D+01,
     & 0.49412D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.11285D+01,0.11386D+01,
     & 0.11499D+01,0.11623D+01,0.11758D+01,0.11904D+01,0.12061D+01,
     & 0.12229D+01,0.12409D+01,0.12599D+01,0.12801D+01,0.13014D+01,
     & 0.13239D+01,0.13475D+01,0.13722D+01,0.13982D+01,0.14253D+01,
     & 0.14537D+01,0.14833D+01,0.15141D+01,0.15462D+01,0.15796D+01,
     & 0.16144D+01,0.16505D+01,0.16880D+01,0.17269D+01,0.17672D+01,
     & 0.18091D+01,0.18525D+01,0.18974D+01,0.19439D+01,0.19921D+01,
     & 0.20420D+01,0.20937D+01,0.21471D+01,0.22024D+01,0.22595D+01,
     & 0.23187D+01,0.23798D+01,0.24430D+01,0.25084D+01,0.25759D+01,
     & 0.26458D+01,0.27180D+01,0.27926D+01,0.28696D+01,0.29493D+01,
     & 0.30316D+01,0.31167D+01,0.32046D+01,0.32954D+01,0.33892D+01,
     & 0.34861D+01,0.35863D+01,0.36897D+01,0.37965D+01,0.39069D+01,
     & 0.40209D+01,0.41387D+01,0.42603D+01,0.43860D+01,0.45157D+01,
     & 0.46497D+01,0.47880D+01,0.49309D+01,0.50784D+01,0.52308D+01,
     & 0.53880D+01,0.55504D+01,0.57180D+01,0.58910D+01,0.60695D+01,
     & 0.62538D+01,0.64445D+01,0.67817D+01,0.69668D+01,0.68405D+01,
     & 0.62431D+01,0.46533D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.10806D+01,0.10886D+01,
     & 0.10977D+01,0.11078D+01,0.11191D+01,0.11314D+01,0.11447D+01,
     & 0.11590D+01,0.11745D+01,0.11909D+01,0.12084D+01,0.12270D+01,
     & 0.12466D+01,0.12673D+01,0.12891D+01,0.13120D+01,0.13360D+01,
     & 0.13611D+01,0.13873D+01,0.14147D+01,0.14433D+01,0.14730D+01,
     & 0.15040D+01,0.15362D+01,0.15697D+01,0.16044D+01,0.16405D+01,
     & 0.16780D+01,0.17168D+01,0.17571D+01,0.17988D+01,0.18420D+01,
     & 0.18867D+01,0.19331D+01,0.19810D+01,0.20306D+01,0.20820D+01,
     & 0.21351D+01,0.21900D+01,0.22469D+01,0.23056D+01,0.23663D+01,
     & 0.24292D+01,0.24941D+01,0.25612D+01,0.26306D+01,0.27023D+01,
     & 0.27764D+01,0.28530D+01,0.29321D+01,0.30139D+01,0.30984D+01,
     & 0.31857D+01,0.32760D+01,0.33692D+01,0.34655D+01,0.35651D+01,
     & 0.36679D+01,0.37741D+01,0.38838D+01,0.39972D+01,0.41143D+01,
     & 0.42353D+01,0.43602D+01,0.44892D+01,0.46225D+01,0.47602D+01,
     & 0.49023D+01,0.50491D+01,0.52007D+01,0.53572D+01,0.55188D+01,
     & 0.56856D+01,0.58578D+01,0.60360D+01,0.63490D+01,0.65224D+01,
     & 0.64095D+01,0.58634D+01,0.43924D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.10359D+01,0.10420D+01,
     & 0.10491D+01,0.10573D+01,0.10665D+01,0.10767D+01,0.10879D+01,
     & 0.11001D+01,0.11133D+01,0.11274D+01,0.11425D+01,0.11587D+01,
     & 0.11758D+01,0.11939D+01,0.12130D+01,0.12331D+01,0.12543D+01,
     & 0.12765D+01,0.12997D+01,0.13240D+01,0.13494D+01,0.13759D+01,
     & 0.14036D+01,0.14323D+01,0.14622D+01,0.14933D+01,0.15256D+01,
     & 0.15592D+01,0.15940D+01,0.16301D+01,0.16676D+01,0.17064D+01,
     & 0.17466D+01,0.17882D+01,0.18314D+01,0.18760D+01,0.19222D+01,
     & 0.19700D+01,0.20195D+01,0.20707D+01,0.21236D+01,0.21783D+01,
     & 0.22349D+01,0.22935D+01,0.23540D+01,0.24165D+01,0.24812D+01,
     & 0.25481D+01,0.26172D+01,0.26886D+01,0.27624D+01,0.28387D+01,
     & 0.29176D+01,0.29991D+01,0.30833D+01,0.31704D+01,0.32603D+01,
     & 0.33533D+01,0.34493D+01,0.35485D+01,0.36511D+01,0.37570D+01,
     & 0.38665D+01,0.39796D+01,0.40964D+01,0.42171D+01,0.43418D+01,
     & 0.44706D+01,0.46037D+01,0.47411D+01,0.48830D+01,0.50296D+01,
     & 0.51809D+01,0.53371D+01,0.54985D+01,0.56655D+01,0.59570D+01,
     & 0.61198D+01,0.60188D+01,0.55184D+01,0.41533D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.99295D+00,0.99731D+00,
     & 0.10027D+01,0.10091D+01,0.10164D+01,0.10247D+01,0.10341D+01,
     & 0.10442D+01,0.10554D+01,0.10675D+01,0.10805D+01,0.10945D+01,
     & 0.11093D+01,0.11251D+01,0.11420D+01,0.11596D+01,0.11782D+01,
     & 0.11978D+01,0.12184D+01,0.12399D+01,0.12624D+01,0.12861D+01,
     & 0.13108D+01,0.13363D+01,0.13630D+01,0.13908D+01,0.14198D+01,
     & 0.14499D+01,0.14811D+01,0.15135D+01,0.15470D+01,0.15819D+01,
     & 0.16180D+01,0.16556D+01,0.16945D+01,0.17347D+01,0.17763D+01,
     & 0.18194D+01,0.18639D+01,0.19100D+01,0.19577D+01,0.20071D+01,
     & 0.20582D+01,0.21111D+01,0.21658D+01,0.22224D+01,0.22809D+01,
     & 0.23413D+01,0.24038D+01,0.24684D+01,0.25352D+01,0.26042D+01,
     & 0.26755D+01,0.27493D+01,0.28255D+01,0.29042D+01,0.29857D+01,
     & 0.30698D+01,0.31568D+01,0.32467D+01,0.33396D+01,0.34357D+01,
     & 0.35349D+01,0.36374D+01,0.37434D+01,0.38529D+01,0.39661D+01,
     & 0.40830D+01,0.42038D+01,0.43286D+01,0.44575D+01,0.45907D+01,
     & 0.47283D+01,0.48705D+01,0.50172D+01,0.51688D+01,0.53257D+01,
     & 0.55979D+01,0.57512D+01,0.56605D+01,0.52004D+01,0.39294D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.94744D+00,0.95009D+00,
     & 0.95402D+00,0.95843D+00,0.96446D+00,0.97068D+00,0.97857D+00,
     & 0.98667D+00,0.99620D+00,0.10065D+01,0.10173D+01,0.10298D+01,
     & 0.10421D+01,0.10560D+01,0.10710D+01,0.10859D+01,0.11024D+01,
     & 0.11202D+01,0.11379D+01,0.11568D+01,0.11772D+01,0.11985D+01,
     & 0.12200D+01,0.12426D+01,0.12667D+01,0.12919D+01,0.13180D+01,
     & 0.13445D+01,0.13722D+01,0.14014D+01,0.14318D+01,0.14635D+01,
     & 0.14962D+01,0.15298D+01,0.15644D+01,0.16005D+01,0.16379D+01,
     & 0.16769D+01,0.17173D+01,0.17592D+01,0.18025D+01,0.18472D+01,
     & 0.18935D+01,0.19412D+01,0.19905D+01,0.20413D+01,0.20941D+01,
     & 0.21487D+01,0.22052D+01,0.22637D+01,0.23242D+01,0.23869D+01,
     & 0.24516D+01,0.25186D+01,0.25878D+01,0.26594D+01,0.27334D+01,
     & 0.28099D+01,0.28890D+01,0.29707D+01,0.30552D+01,0.31425D+01,
     & 0.32328D+01,0.33261D+01,0.34225D+01,0.35221D+01,0.36251D+01,
     & 0.37315D+01,0.38415D+01,0.39551D+01,0.40725D+01,0.41939D+01,
     & 0.43192D+01,0.44487D+01,0.45826D+01,0.47207D+01,0.48635D+01,
     & 0.50113D+01,0.52664D+01,0.54110D+01,0.53292D+01,0.49043D+01,
     & 0.37162D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.89934D+00,0.90054D+00,
     & 0.90276D+00,0.90596D+00,0.91010D+00,0.91517D+00,0.92114D+00,
     & 0.92800D+00,0.93572D+00,0.94431D+00,0.95372D+00,0.96399D+00,
     & 0.97510D+00,0.98703D+00,0.99985D+00,0.10134D+01,0.10279D+01,
     & 0.10432D+01,0.10593D+01,0.10763D+01,0.10942D+01,0.11129D+01,
     & 0.11325D+01,0.11530D+01,0.11745D+01,0.11969D+01,0.12202D+01,
     & 0.12445D+01,0.12697D+01,0.12961D+01,0.13234D+01,0.13518D+01,
     & 0.13812D+01,0.14117D+01,0.14434D+01,0.14762D+01,0.15103D+01,
     & 0.15456D+01,0.15822D+01,0.16200D+01,0.16592D+01,0.16997D+01,
     & 0.17417D+01,0.17851D+01,0.18300D+01,0.18765D+01,0.19247D+01,
     & 0.19744D+01,0.20259D+01,0.20792D+01,0.21343D+01,0.21913D+01,
     & 0.22502D+01,0.23112D+01,0.23741D+01,0.24393D+01,0.25066D+01,
     & 0.25763D+01,0.26483D+01,0.27228D+01,0.27998D+01,0.28794D+01,
     & 0.29617D+01,0.30469D+01,0.31349D+01,0.32258D+01,0.33199D+01,
     & 0.34171D+01,0.35176D+01,0.36216D+01,0.37290D+01,0.38400D+01,
     & 0.39547D+01,0.40733D+01,0.41958D+01,0.43225D+01,0.44533D+01,
     & 0.45885D+01,0.47285D+01,0.49686D+01,0.51059D+01,0.50319D+01,
     & 0.46380D+01,0.35229D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.85773D+00,0.85748D+00,
     & 0.85826D+00,0.86004D+00,0.86276D+00,0.86640D+00,0.87094D+00,
     & 0.87634D+00,0.88258D+00,0.88967D+00,0.89758D+00,0.90629D+00,
     & 0.91582D+00,0.92614D+00,0.93726D+00,0.94918D+00,0.96189D+00,
     & 0.97540D+00,0.98971D+00,0.10048D+01,0.10208D+01,0.10375D+01,
     & 0.10551D+01,0.10735D+01,0.10928D+01,0.11130D+01,0.11340D+01,
     & 0.11560D+01,0.11788D+01,0.12026D+01,0.12273D+01,0.12531D+01,
     & 0.12798D+01,0.13075D+01,0.13363D+01,0.13662D+01,0.13971D+01,
     & 0.14292D+01,0.14625D+01,0.14970D+01,0.15326D+01,0.15696D+01,
     & 0.16079D+01,0.16475D+01,0.16885D+01,0.17309D+01,0.17749D+01,
     & 0.18203D+01,0.18673D+01,0.19160D+01,0.19663D+01,0.20184D+01,
     & 0.20722D+01,0.21279D+01,0.21855D+01,0.22451D+01,0.23068D+01,
     & 0.23705D+01,0.24364D+01,0.25046D+01,0.25751D+01,0.26480D+01,
     & 0.27234D+01,0.28014D+01,0.28821D+01,0.29654D+01,0.30517D+01,
     & 0.31408D+01,0.32330D+01,0.33283D+01,0.34268D+01,0.35287D+01,
     & 0.36340D+01,0.37429D+01,0.38555D+01,0.39718D+01,0.40921D+01,
     & 0.42163D+01,0.43447D+01,0.44778D+01,0.47047D+01,0.48353D+01,
     & 0.47682D+01,0.44015D+01,0.33511D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.82026D+00,
     & 0.81966D+00,0.82007D+00,0.82144D+00,0.82374D+00,0.82692D+00,
     & 0.83097D+00,0.83586D+00,0.84157D+00,0.84808D+00,0.85539D+00,
     & 0.86347D+00,0.87233D+00,0.88195D+00,0.89234D+00,0.90349D+00,
     & 0.91540D+00,0.92807D+00,0.94151D+00,0.95572D+00,0.97070D+00,
     & 0.98647D+00,0.10030D+01,0.10204D+01,0.10386D+01,0.10576D+01,
     & 0.10774D+01,0.10981D+01,0.11197D+01,0.11422D+01,0.11655D+01,
     & 0.11898D+01,0.12150D+01,0.12413D+01,0.12685D+01,0.12967D+01,
     & 0.13260D+01,0.13563D+01,0.13878D+01,0.14204D+01,0.14541D+01,
     & 0.14891D+01,0.15253D+01,0.15628D+01,0.16016D+01,0.16418D+01,
     & 0.16834D+01,0.17265D+01,0.17710D+01,0.18171D+01,0.18648D+01,
     & 0.19141D+01,0.19651D+01,0.20180D+01,0.20726D+01,0.21291D+01,
     & 0.21876D+01,0.22480D+01,0.23106D+01,0.23753D+01,0.24422D+01,
     & 0.25114D+01,0.25830D+01,0.26571D+01,0.27337D+01,0.28129D+01,
     & 0.28948D+01,0.29796D+01,0.30672D+01,0.31578D+01,0.32515D+01,
     & 0.33485D+01,0.34487D+01,0.35523D+01,0.36594D+01,0.37702D+01,
     & 0.38847D+01,0.40030D+01,0.41254D+01,0.42521D+01,0.44672D+01,
     & 0.45919D+01,0.45309D+01,0.41886D+01,0.31964D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.78828D+00,
     & 0.78632D+00,0.78540D+00,0.78547D+00,0.78647D+00,0.78838D+00,
     & 0.79116D+00,0.79477D+00,0.79920D+00,0.80442D+00,0.81042D+00,
     & 0.81718D+00,0.82469D+00,0.83295D+00,0.84195D+00,0.85167D+00,
     & 0.86213D+00,0.87332D+00,0.88524D+00,0.89790D+00,0.91129D+00,
     & 0.92542D+00,0.94030D+00,0.95594D+00,0.97234D+00,0.98952D+00,
     & 0.10075D+01,0.10263D+01,0.10459D+01,0.10663D+01,0.10875D+01,
     & 0.11096D+01,0.11327D+01,0.11566D+01,0.11814D+01,0.12072D+01,
     & 0.12339D+01,0.12617D+01,0.12905D+01,0.13203D+01,0.13512D+01,
     & 0.13833D+01,0.14165D+01,0.14508D+01,0.14864D+01,0.15233D+01,
     & 0.15614D+01,0.16009D+01,0.16418D+01,0.16841D+01,0.17279D+01,
     & 0.17732D+01,0.18201D+01,0.18686D+01,0.19188D+01,0.19708D+01,
     & 0.20245D+01,0.20801D+01,0.21376D+01,0.21971D+01,0.22587D+01,
     & 0.23224D+01,0.23883D+01,0.24565D+01,0.25270D+01,0.26000D+01,
     & 0.26754D+01,0.27535D+01,0.28343D+01,0.29178D+01,0.30042D+01,
     & 0.30936D+01,0.31860D+01,0.32816D+01,0.33805D+01,0.34827D+01,
     & 0.35884D+01,0.36977D+01,0.38107D+01,0.39276D+01,0.40487D+01,
     & 0.42530D+01,0.43723D+01,0.43168D+01,0.39965D+01,0.30568D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.76109D+00,
     & 0.75776D+00,0.75552D+00,0.75430D+00,0.75405D+00,0.75473D+00,
     & 0.75628D+00,0.75868D+00,0.76190D+00,0.76591D+00,0.77069D+00,
     & 0.77622D+00,0.78249D+00,0.78949D+00,0.79720D+00,0.80563D+00,
     & 0.81476D+00,0.82460D+00,0.83514D+00,0.84638D+00,0.85832D+00,
     & 0.87098D+00,0.88434D+00,0.89842D+00,0.91322D+00,0.92876D+00,
     & 0.94504D+00,0.96207D+00,0.97987D+00,0.99845D+00,0.10178D+01,
     & 0.10380D+01,0.10590D+01,0.10809D+01,0.11036D+01,0.11272D+01,
     & 0.11517D+01,0.11771D+01,0.12035D+01,0.12308D+01,0.12592D+01,
     & 0.12886D+01,0.13191D+01,0.13507D+01,0.13834D+01,0.14172D+01,
     & 0.14523D+01,0.14886D+01,0.15262D+01,0.15652D+01,0.16055D+01,
     & 0.16472D+01,0.16903D+01,0.17350D+01,0.17812D+01,0.18291D+01,
     & 0.18786D+01,0.19298D+01,0.19828D+01,0.20377D+01,0.20945D+01,
     & 0.21532D+01,0.22140D+01,0.22769D+01,0.23420D+01,0.24093D+01,
     & 0.24790D+01,0.25511D+01,0.26256D+01,0.27028D+01,0.27826D+01,
     & 0.28652D+01,0.29506D+01,0.30390D+01,0.31305D+01,0.32250D+01,
     & 0.33228D+01,0.34240D+01,0.35286D+01,0.36368D+01,0.37487D+01,
     & 0.38647D+01,0.40593D+01,0.41737D+01,0.41232D+01,0.38227D+01,
     & 0.29304D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.73837D+00,
     & 0.73362D+00,0.73003D+00,0.72752D+00,0.72603D+00,0.72549D+00,
     & 0.72585D+00,0.72708D+00,0.72913D+00,0.73199D+00,0.73561D+00,
     & 0.73999D+00,0.74510D+00,0.75092D+00,0.75745D+00,0.76468D+00,
     & 0.77259D+00,0.78118D+00,0.79045D+00,0.80040D+00,0.81103D+00,
     & 0.82233D+00,0.83431D+00,0.84698D+00,0.86033D+00,0.87438D+00,
     & 0.88913D+00,0.90460D+00,0.92079D+00,0.93771D+00,0.95537D+00,
     & 0.97380D+00,0.99300D+00,0.10130D+01,0.10338D+01,0.10554D+01,
     & 0.10779D+01,0.11012D+01,0.11255D+01,0.11506D+01,0.11767D+01,
     & 0.12037D+01,0.12318D+01,0.12608D+01,0.12909D+01,0.13221D+01,
     & 0.13544D+01,0.13879D+01,0.14226D+01,0.14584D+01,0.14956D+01,
     & 0.15341D+01,0.15739D+01,0.16151D+01,0.16578D+01,0.17019D+01,
     & 0.17476D+01,0.17949D+01,0.18439D+01,0.18946D+01,0.19470D+01,
     & 0.20013D+01,0.20575D+01,0.21156D+01,0.21758D+01,0.22380D+01,
     & 0.23025D+01,0.23691D+01,0.24381D+01,0.25096D+01,0.25834D+01,
     & 0.26599D+01,0.27390D+01,0.28209D+01,0.29056D+01,0.29932D+01,
     & 0.30838D+01,0.31776D+01,0.32746D+01,0.33750D+01,0.34788D+01,
     & 0.35861D+01,0.36975D+01,0.38833D+01,0.39933D+01,0.39472D+01,
     & 0.36648D+01,0.28156D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.71364D+00,0.70864D+00,0.70481D+00,0.70204D+00,0.70028D+00,
     & 0.69946D+00,0.69954D+00,0.70046D+00,0.70220D+00,0.70472D+00,
     & 0.70799D+00,0.71200D+00,0.71672D+00,0.72214D+00,0.72824D+00,
     & 0.73501D+00,0.74246D+00,0.75056D+00,0.75932D+00,0.76873D+00,
     & 0.77880D+00,0.78951D+00,0.80089D+00,0.81292D+00,0.82561D+00,
     & 0.83898D+00,0.85302D+00,0.86774D+00,0.88316D+00,0.89929D+00,
     & 0.91612D+00,0.93369D+00,0.95201D+00,0.97108D+00,0.99092D+00,
     & 0.10116D+01,0.10330D+01,0.10553D+01,0.10784D+01,0.11024D+01,
     & 0.11274D+01,0.11532D+01,0.11800D+01,0.12077D+01,0.12365D+01,
     & 0.12664D+01,0.12973D+01,0.13292D+01,0.13624D+01,0.13967D+01,
     & 0.14323D+01,0.14691D+01,0.15072D+01,0.15466D+01,0.15875D+01,
     & 0.16297D+01,0.16735D+01,0.17188D+01,0.17657D+01,0.18143D+01,
     & 0.18645D+01,0.19165D+01,0.19704D+01,0.20261D+01,0.20838D+01,
     & 0.21435D+01,0.22053D+01,0.22692D+01,0.23354D+01,0.24039D+01,
     & 0.24748D+01,0.25482D+01,0.26242D+01,0.27028D+01,0.27841D+01,
     & 0.28683D+01,0.29554D+01,0.30455D+01,0.31388D+01,0.32353D+01,
     & 0.33350D+01,0.34383D+01,0.35454D+01,0.37232D+01,0.38291D+01,
     & 0.37872D+01,0.35210D+01,0.27110D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.69766D+00,0.69116D+00,0.68593D+00,0.68185D+00,0.67883D+00,
     & 0.67682D+00,0.67574D+00,0.67554D+00,0.67618D+00,0.67762D+00,
     & 0.67983D+00,0.68278D+00,0.68644D+00,0.69080D+00,0.69585D+00,
     & 0.70156D+00,0.70792D+00,0.71494D+00,0.72259D+00,0.73088D+00,
     & 0.73980D+00,0.74936D+00,0.75954D+00,0.77036D+00,0.78181D+00,
     & 0.79391D+00,0.80665D+00,0.82004D+00,0.83409D+00,0.84881D+00,
     & 0.86420D+00,0.88029D+00,0.89708D+00,0.91458D+00,0.93281D+00,
     & 0.95179D+00,0.97153D+00,0.99205D+00,0.10134D+01,0.10355D+01,
     & 0.10585D+01,0.10824D+01,0.11071D+01,0.11327D+01,0.11593D+01,
     & 0.11869D+01,0.12155D+01,0.12451D+01,0.12757D+01,0.13075D+01,
     & 0.13404D+01,0.13745D+01,0.14098D+01,0.14463D+01,0.14841D+01,
     & 0.15233D+01,0.15639D+01,0.16059D+01,0.16493D+01,0.16944D+01,
     & 0.17409D+01,0.17892D+01,0.18391D+01,0.18908D+01,0.19444D+01,
     & 0.19998D+01,0.20571D+01,0.21165D+01,0.21780D+01,0.22416D+01,
     & 0.23075D+01,0.23757D+01,0.24463D+01,0.25193D+01,0.25950D+01,
     & 0.26732D+01,0.27543D+01,0.28381D+01,0.29249D+01,0.30147D+01,
     & 0.31076D+01,0.32037D+01,0.33032D+01,0.34064D+01,0.35769D+01,
     & 0.36792D+01,0.36409D+01,0.33897D+01,0.26156D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.68562D+00,0.67749D+00,0.67075D+00,0.66526D+00,0.66094D+00,
     & 0.65768D+00,0.65542D+00,0.65409D+00,0.65363D+00,0.65400D+00,
     & 0.65516D+00,0.65708D+00,0.65972D+00,0.66307D+00,0.66710D+00,
     & 0.67180D+00,0.67715D+00,0.68314D+00,0.68976D+00,0.69700D+00,
     & 0.70486D+00,0.71333D+00,0.72242D+00,0.73212D+00,0.74243D+00,
     & 0.75336D+00,0.76490D+00,0.77707D+00,0.78987D+00,0.80330D+00,
     & 0.81738D+00,0.83211D+00,0.84751D+00,0.86358D+00,0.88035D+00,
     & 0.89781D+00,0.91600D+00,0.93492D+00,0.95459D+00,0.97503D+00,
     & 0.99626D+00,0.10183D+01,0.10412D+01,0.10649D+01,0.10895D+01,
     & 0.11150D+01,0.11415D+01,0.11689D+01,0.11974D+01,0.12268D+01,
     & 0.12573D+01,0.12889D+01,0.13216D+01,0.13555D+01,0.13907D+01,
     & 0.14270D+01,0.14647D+01,0.15036D+01,0.15440D+01,0.15858D+01,
     & 0.16291D+01,0.16739D+01,0.17203D+01,0.17684D+01,0.18181D+01,
     & 0.18696D+01,0.19230D+01,0.19782D+01,0.20354D+01,0.20946D+01,
     & 0.21559D+01,0.22193D+01,0.22850D+01,0.23530D+01,0.24234D+01,
     & 0.24964D+01,0.25718D+01,0.26499D+01,0.27308D+01,0.28145D+01,
     & 0.29011D+01,0.29908D+01,0.30836D+01,0.31796D+01,0.32792D+01,
     & 0.34430D+01,0.35419D+01,0.35071D+01,0.32695D+01,0.25283D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.67757D+00,0.66763D+00,0.65922D+00,0.65222D+00,0.64648D+00,
     & 0.64191D+00,0.63841D+00,0.63590D+00,0.63432D+00,0.63362D+00,
     & 0.63373D+00,0.63463D+00,0.63627D+00,0.63863D+00,0.64169D+00,
     & 0.64541D+00,0.64979D+00,0.65480D+00,0.66044D+00,0.66669D+00,
     & 0.67356D+00,0.68102D+00,0.68908D+00,0.69774D+00,0.70699D+00,
     & 0.71684D+00,0.72728D+00,0.73832D+00,0.74996D+00,0.76222D+00,
     & 0.77509D+00,0.78858D+00,0.80270D+00,0.81747D+00,0.83289D+00,
     & 0.84898D+00,0.86574D+00,0.88320D+00,0.90137D+00,0.92026D+00,
     & 0.93990D+00,0.96030D+00,0.98148D+00,0.10035D+01,0.10263D+01,
     & 0.10500D+01,0.10745D+01,0.10999D+01,0.11263D+01,0.11536D+01,
     & 0.11820D+01,0.12113D+01,0.12417D+01,0.12732D+01,0.13059D+01,
     & 0.13397D+01,0.13747D+01,0.14109D+01,0.14484D+01,0.14873D+01,
     & 0.15276D+01,0.15693D+01,0.16125D+01,0.16572D+01,0.17035D+01,
     & 0.17515D+01,0.18011D+01,0.18526D+01,0.19058D+01,0.19610D+01,
     & 0.20181D+01,0.20772D+01,0.21385D+01,0.22019D+01,0.22675D+01,
     & 0.23355D+01,0.24059D+01,0.24788D+01,0.25543D+01,0.26324D+01,
     & 0.27133D+01,0.27970D+01,0.28837D+01,0.29734D+01,0.30663D+01,
     & 0.31626D+01,0.33202D+01,0.34160D+01,0.33843D+01,0.31593D+01,
     & 0.24483D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.66165D+00,0.65140D+00,0.64271D+00,0.63542D+00,
     & 0.62942D+00,0.62460D+00,0.62085D+00,0.61810D+00,0.61629D+00,
     & 0.61533D+00,0.61520D+00,0.61585D+00,0.61723D+00,0.61932D+00,
     & 0.62210D+00,0.62553D+00,0.62961D+00,0.63432D+00,0.63963D+00,
     & 0.64555D+00,0.65206D+00,0.65916D+00,0.66684D+00,0.67510D+00,
     & 0.68394D+00,0.69336D+00,0.70335D+00,0.71393D+00,0.72509D+00,
     & 0.73685D+00,0.74920D+00,0.76215D+00,0.77572D+00,0.78991D+00,
     & 0.80473D+00,0.82020D+00,0.83632D+00,0.85311D+00,0.87059D+00,
     & 0.88876D+00,0.90766D+00,0.92730D+00,0.94769D+00,0.96886D+00,
     & 0.99083D+00,0.10136D+01,0.10373D+01,0.10618D+01,0.10872D+01,
     & 0.11135D+01,0.11408D+01,0.11691D+01,0.11984D+01,0.12288D+01,
     & 0.12602D+01,0.12928D+01,0.13266D+01,0.13616D+01,0.13978D+01,
     & 0.14353D+01,0.14742D+01,0.15144D+01,0.15561D+01,0.15993D+01,
     & 0.16440D+01,0.16903D+01,0.17382D+01,0.17879D+01,0.18394D+01,
     & 0.18927D+01,0.19478D+01,0.20050D+01,0.20642D+01,0.21255D+01,
     & 0.21890D+01,0.22547D+01,0.23228D+01,0.23933D+01,0.24664D+01,
     & 0.25420D+01,0.26202D+01,0.27013D+01,0.27852D+01,0.28721D+01,
     & 0.29620D+01,0.30554D+01,0.32073D+01,0.33002D+01,0.32714D+01,
     & 0.30581D+01,0.23748D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.65973D+00,0.64740D+00,0.63681D+00,0.62781D+00,
     & 0.62023D+00,0.61396D+00,0.60887D+00,0.60488D+00,0.60189D+00,
     & 0.59983D+00,0.59864D+00,0.59827D+00,0.59867D+00,0.59980D+00,
     & 0.60164D+00,0.60416D+00,0.60732D+00,0.61112D+00,0.61553D+00,
     & 0.62054D+00,0.62615D+00,0.63233D+00,0.63909D+00,0.64642D+00,
     & 0.65432D+00,0.66277D+00,0.67180D+00,0.68138D+00,0.69153D+00,
     & 0.70225D+00,0.71355D+00,0.72542D+00,0.73788D+00,0.75093D+00,
     & 0.76459D+00,0.77886D+00,0.79375D+00,0.80928D+00,0.82546D+00,
     & 0.84230D+00,0.85982D+00,0.87804D+00,0.89697D+00,0.91664D+00,
     & 0.93706D+00,0.95825D+00,0.98023D+00,0.10030D+01,0.10267D+01,
     & 0.10512D+01,0.10767D+01,0.11030D+01,0.11303D+01,0.11586D+01,
     & 0.11879D+01,0.12183D+01,0.12498D+01,0.12824D+01,0.13162D+01,
     & 0.13512D+01,0.13875D+01,0.14250D+01,0.14639D+01,0.15042D+01,
     & 0.15460D+01,0.15892D+01,0.16340D+01,0.16804D+01,0.17285D+01,
     & 0.17782D+01,0.18298D+01,0.18832D+01,0.19386D+01,0.19959D+01,
     & 0.20552D+01,0.21167D+01,0.21804D+01,0.22464D+01,0.23147D+01,
     & 0.23854D+01,0.24587D+01,0.25346D+01,0.26132D+01,0.26946D+01,
     & 0.27789D+01,0.28661D+01,0.29567D+01,0.31033D+01,0.31936D+01,
     & 0.31675D+01,0.29649D+01,0.23073D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.66215D+00,0.64744D+00,0.63471D+00,0.62376D+00,
     & 0.61443D+00,0.60654D+00,0.59998D+00,0.59463D+00,0.59038D+00,
     & 0.58714D+00,0.58484D+00,0.58341D+00,0.58281D+00,0.58297D+00,
     & 0.58386D+00,0.58546D+00,0.58772D+00,0.59063D+00,0.59416D+00,
     & 0.59829D+00,0.60303D+00,0.60834D+00,0.61422D+00,0.62066D+00,
     & 0.62767D+00,0.63522D+00,0.64333D+00,0.65198D+00,0.66119D+00,
     & 0.67094D+00,0.68125D+00,0.69212D+00,0.70356D+00,0.71556D+00,
     & 0.72814D+00,0.74130D+00,0.75506D+00,0.76943D+00,0.78441D+00,
     & 0.80003D+00,0.81629D+00,0.83320D+00,0.85079D+00,0.86908D+00,
     & 0.88807D+00,0.90780D+00,0.92827D+00,0.94952D+00,0.97156D+00,
     & 0.99443D+00,0.10181D+01,0.10427D+01,0.10682D+01,0.10946D+01,
     & 0.11220D+01,0.11503D+01,0.11797D+01,0.12102D+01,0.12418D+01,
     & 0.12744D+01,0.13083D+01,0.13434D+01,0.13798D+01,0.14174D+01,
     & 0.14565D+01,0.14969D+01,0.15388D+01,0.15821D+01,0.16271D+01,
     & 0.16737D+01,0.17219D+01,0.17719D+01,0.18237D+01,0.18773D+01,
     & 0.19329D+01,0.19904D+01,0.20501D+01,0.21119D+01,0.21758D+01,
     & 0.22421D+01,0.23108D+01,0.23819D+01,0.24556D+01,0.25319D+01,
     & 0.26110D+01,0.26928D+01,0.27775D+01,0.28655D+01,0.30073D+01,
     & 0.30951D+01,0.30715D+01,0.28789D+01,0.22451D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.63665D+00,0.62349D+00,
     & 0.61216D+00,0.60247D+00,0.59426D+00,0.58739D+00,0.58176D+00,
     & 0.57724D+00,0.57375D+00,0.57121D+00,0.56954D+00,0.56870D+00,
     & 0.56863D+00,0.56929D+00,0.57065D+00,0.57267D+00,0.57533D+00,
     & 0.57861D+00,0.58249D+00,0.58695D+00,0.59199D+00,0.59759D+00,
     & 0.60374D+00,0.61043D+00,0.61767D+00,0.62545D+00,0.63377D+00,
     & 0.64262D+00,0.65201D+00,0.66194D+00,0.67242D+00,0.68344D+00,
     & 0.69502D+00,0.70717D+00,0.71988D+00,0.73317D+00,0.74705D+00,
     & 0.76152D+00,0.77662D+00,0.79234D+00,0.80869D+00,0.82571D+00,
     & 0.84339D+00,0.86177D+00,0.88085D+00,0.90067D+00,0.92123D+00,
     & 0.94257D+00,0.96470D+00,0.98765D+00,0.10115D+01,0.10361D+01,
     & 0.10617D+01,0.10882D+01,0.11157D+01,0.11442D+01,0.11737D+01,
     & 0.12043D+01,0.12359D+01,0.12688D+01,0.13028D+01,0.13381D+01,
     & 0.13746D+01,0.14124D+01,0.14516D+01,0.14922D+01,0.15343D+01,
     & 0.15779D+01,0.16231D+01,0.16699D+01,0.17184D+01,0.17687D+01,
     & 0.18208D+01,0.18747D+01,0.19306D+01,0.19886D+01,0.20486D+01,
     & 0.21107D+01,0.21752D+01,0.22419D+01,0.23110D+01,0.23827D+01,
     & 0.24569D+01,0.25337D+01,0.26133D+01,0.26957D+01,0.27813D+01,
     & 0.29185D+01,0.30041D+01,0.29829D+01,0.27996D+01,0.21877D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.62731D+00,
     & 0.61369D+00,0.60193D+00,0.59186D+00,0.58330D+00,0.57611D+00,
     & 0.57018D+00,0.56538D+00,0.56163D+00,0.55884D+00,0.55693D+00,
     & 0.55585D+00,0.55555D+00,0.55598D+00,0.55711D+00,0.55890D+00,
     & 0.56133D+00,0.56437D+00,0.56800D+00,0.57221D+00,0.57699D+00,
     & 0.58232D+00,0.58819D+00,0.59460D+00,0.60154D+00,0.60902D+00,
     & 0.61702D+00,0.62554D+00,0.63459D+00,0.64417D+00,0.65428D+00,
     & 0.66493D+00,0.67612D+00,0.68786D+00,0.70015D+00,0.71300D+00,
     & 0.72643D+00,0.74045D+00,0.75506D+00,0.77027D+00,0.78611D+00,
     & 0.80259D+00,0.81973D+00,0.83753D+00,0.85602D+00,0.87523D+00,
     & 0.89516D+00,0.91584D+00,0.93730D+00,0.95955D+00,0.98263D+00,
     & 0.10066D+01,0.10314D+01,0.10571D+01,0.10837D+01,0.11114D+01,
     & 0.11400D+01,0.11697D+01,0.12004D+01,0.12323D+01,0.12653D+01,
     & 0.12995D+01,0.13350D+01,0.13717D+01,0.14098D+01,0.14493D+01,
     & 0.14901D+01,0.15325D+01,0.15764D+01,0.16219D+01,0.16691D+01,
     & 0.17179D+01,0.17685D+01,0.18210D+01,0.18754D+01,0.19317D+01,
     & 0.19901D+01,0.20506D+01,0.21132D+01,0.21782D+01,0.22455D+01,
     & 0.23152D+01,0.23874D+01,0.24622D+01,0.25397D+01,0.26200D+01,
     & 0.27034D+01,0.28364D+01,0.29199D+01,0.29009D+01,0.27263D+01,
     & 0.21348D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.60523D+00,0.59302D+00,0.58253D+00,0.57359D+00,
     & 0.56607D+00,0.55981D+00,0.55473D+00,0.55070D+00,0.54765D+00,
     & 0.54550D+00,0.54419D+00,0.54366D+00,0.54386D+00,0.54476D+00,
     & 0.54632D+00,0.54852D+00,0.55133D+00,0.55473D+00,0.55870D+00,
     & 0.56323D+00,0.56830D+00,0.57392D+00,0.58006D+00,0.58673D+00,
     & 0.59391D+00,0.60162D+00,0.60984D+00,0.61857D+00,0.62782D+00,
     & 0.63759D+00,0.64789D+00,0.65872D+00,0.67008D+00,0.68198D+00,
     & 0.69443D+00,0.70744D+00,0.72102D+00,0.73519D+00,0.74994D+00,
     & 0.76530D+00,0.78129D+00,0.79791D+00,0.81518D+00,0.83313D+00,
     & 0.85176D+00,0.87111D+00,0.89119D+00,0.91202D+00,0.93362D+00,
     & 0.95604D+00,0.97928D+00,0.10034D+01,0.10283D+01,0.10543D+01,
     & 0.10811D+01,0.11089D+01,0.11377D+01,0.11676D+01,0.11986D+01,
     & 0.12307D+01,0.12640D+01,0.12984D+01,0.13342D+01,0.13712D+01,
     & 0.14096D+01,0.14493D+01,0.14905D+01,0.15332D+01,0.15775D+01,
     & 0.16234D+01,0.16709D+01,0.17202D+01,0.17713D+01,0.18242D+01,
     & 0.18791D+01,0.19359D+01,0.19948D+01,0.20559D+01,0.21192D+01,
     & 0.21847D+01,0.22527D+01,0.23231D+01,0.23960D+01,0.24716D+01,
     & 0.25499D+01,0.26312D+01,0.27603D+01,0.28418D+01,0.28248D+01,
     & 0.26584D+01,0.20859D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.59807D+00,0.58536D+00,0.57443D+00,
     & 0.56510D+00,0.55720D+00,0.55061D+00,0.54522D+00,0.54090D+00,
     & 0.53759D+00,0.53518D+00,0.53363D+00,0.53286D+00,0.53284D+00,
     & 0.53352D+00,0.53486D+00,0.53683D+00,0.53941D+00,0.54259D+00,
     & 0.54633D+00,0.55062D+00,0.55546D+00,0.56082D+00,0.56671D+00,
     & 0.57312D+00,0.58004D+00,0.58746D+00,0.59540D+00,0.60384D+00,
     & 0.61278D+00,0.62224D+00,0.63221D+00,0.64270D+00,0.65371D+00,
     & 0.66525D+00,0.67733D+00,0.68995D+00,0.70313D+00,0.71688D+00,
     & 0.73120D+00,0.74612D+00,0.76165D+00,0.77779D+00,0.79457D+00,
     & 0.81201D+00,0.83012D+00,0.84892D+00,0.86844D+00,0.88869D+00,
     & 0.90970D+00,0.93149D+00,0.95409D+00,0.97752D+00,0.10018D+01,
     & 0.10270D+01,0.10531D+01,0.10802D+01,0.11082D+01,0.11373D+01,
     & 0.11675D+01,0.11987D+01,0.12311D+01,0.12646D+01,0.12994D+01,
     & 0.13355D+01,0.13728D+01,0.14116D+01,0.14517D+01,0.14933D+01,
     & 0.15364D+01,0.15811D+01,0.16275D+01,0.16755D+01,0.17253D+01,
     & 0.17769D+01,0.18303D+01,0.18858D+01,0.19432D+01,0.20028D+01,
     & 0.20645D+01,0.21285D+01,0.21948D+01,0.22634D+01,0.23347D+01,
     & 0.24084D+01,0.24849D+01,0.25642D+01,0.26897D+01,0.27693D+01,
     & 0.27543D+01,0.25956D+01,0.20406D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.57894D+00,
     & 0.56752D+00,0.55775D+00,0.54947D+00,0.54252D+00,0.53679D+00,
     & 0.53218D+00,0.52858D+00,0.52592D+00,0.52411D+00,0.52311D+00,
     & 0.52286D+00,0.52331D+00,0.52443D+00,0.52619D+00,0.52855D+00,
     & 0.53150D+00,0.53502D+00,0.53909D+00,0.54370D+00,0.54883D+00,
     & 0.55448D+00,0.56064D+00,0.56730D+00,0.57447D+00,0.58213D+00,
     & 0.59030D+00,0.59896D+00,0.60812D+00,0.61778D+00,0.62796D+00,
     & 0.63864D+00,0.64984D+00,0.66157D+00,0.67384D+00,0.68664D+00,
     & 0.70001D+00,0.71393D+00,0.72844D+00,0.74354D+00,0.75924D+00,
     & 0.77557D+00,0.79253D+00,0.81015D+00,0.82845D+00,0.84745D+00,
     & 0.86716D+00,0.88761D+00,0.90882D+00,0.93083D+00,0.95365D+00,
     & 0.97731D+00,0.10018D+01,0.10273D+01,0.10536D+01,0.10810D+01,
     & 0.11093D+01,0.11387D+01,0.11691D+01,0.12007D+01,0.12334D+01,
     & 0.12673D+01,0.13024D+01,0.13388D+01,0.13766D+01,0.14157D+01,
     & 0.14563D+01,0.14984D+01,0.15420D+01,0.15872D+01,0.16340D+01,
     & 0.16826D+01,0.17330D+01,0.17852D+01,0.18393D+01,0.18954D+01,
     & 0.19535D+01,0.20138D+01,0.20762D+01,0.21410D+01,0.22081D+01,
     & 0.22777D+01,0.23497D+01,0.24245D+01,0.25020D+01,0.26240D+01,
     & 0.27020D+01,0.26889D+01,0.25373D+01,0.19988D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.57371D+00,0.56177D+00,0.55153D+00,0.54282D+00,0.53549D+00,
     & 0.52942D+00,0.52448D+00,0.52059D+00,0.51765D+00,0.51559D+00,
     & 0.51434D+00,0.51386D+00,0.51408D+00,0.51499D+00,0.51652D+00,
     & 0.51867D+00,0.52141D+00,0.52471D+00,0.52856D+00,0.53295D+00,
     & 0.53786D+00,0.54328D+00,0.54921D+00,0.55563D+00,0.56255D+00,
     & 0.56997D+00,0.57787D+00,0.58626D+00,0.59515D+00,0.60453D+00,
     & 0.61440D+00,0.62478D+00,0.63567D+00,0.64708D+00,0.65901D+00,
     & 0.67147D+00,0.68447D+00,0.69803D+00,0.71215D+00,0.72685D+00,
     & 0.74214D+00,0.75805D+00,0.77457D+00,0.79174D+00,0.80957D+00,
     & 0.82808D+00,0.84729D+00,0.86722D+00,0.88790D+00,0.90935D+00,
     & 0.93160D+00,0.95466D+00,0.97858D+00,0.10034D+01,0.10291D+01,
     & 0.10558D+01,0.10834D+01,0.11120D+01,0.11417D+01,0.11725D+01,
     & 0.12044D+01,0.12375D+01,0.12718D+01,0.13074D+01,0.13442D+01,
     & 0.13824D+01,0.14220D+01,0.14631D+01,0.15057D+01,0.15498D+01,
     & 0.15956D+01,0.16431D+01,0.16923D+01,0.17433D+01,0.17961D+01,
     & 0.18510D+01,0.19078D+01,0.19667D+01,0.20278D+01,0.20911D+01,
     & 0.21567D+01,0.22248D+01,0.22952D+01,0.23684D+01,0.24443D+01,
     & 0.25630D+01,0.26394D+01,0.26280D+01,0.24833D+01,0.19602D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.55716D+00,0.54641D+00,0.53724D+00,
     & 0.52950D+00,0.52305D+00,0.51777D+00,0.51357D+00,0.51034D+00,
     & 0.50801D+00,0.50652D+00,0.50579D+00,0.50579D+00,0.50647D+00,
     & 0.50779D+00,0.50973D+00,0.51225D+00,0.51534D+00,0.51899D+00,
     & 0.52316D+00,0.52785D+00,0.53306D+00,0.53876D+00,0.54496D+00,
     & 0.55164D+00,0.55882D+00,0.56647D+00,0.57462D+00,0.58324D+00,
     & 0.59236D+00,0.60196D+00,0.61205D+00,0.62265D+00,0.63375D+00,
     & 0.64536D+00,0.65750D+00,0.67017D+00,0.68338D+00,0.69715D+00,
     & 0.71148D+00,0.72639D+00,0.74190D+00,0.75802D+00,0.77476D+00,
     & 0.79216D+00,0.81021D+00,0.82896D+00,0.84841D+00,0.86859D+00,
     & 0.88952D+00,0.91123D+00,0.93375D+00,0.95710D+00,0.98132D+00,
     & 0.10064D+01,0.10324D+01,0.10594D+01,0.10874D+01,0.11164D+01,
     & 0.11465D+01,0.11777D+01,0.12100D+01,0.12435D+01,0.12782D+01,
     & 0.13142D+01,0.13516D+01,0.13903D+01,0.14304D+01,0.14721D+01,
     & 0.15152D+01,0.15600D+01,0.16064D+01,0.16545D+01,0.17044D+01,
     & 0.17561D+01,0.18097D+01,0.18653D+01,0.19230D+01,0.19828D+01,
     & 0.20447D+01,0.21090D+01,0.21756D+01,0.22446D+01,0.23162D+01,
     & 0.23905D+01,0.25062D+01,0.25812D+01,0.25715D+01,0.24332D+01,
     & 0.19244D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.54236D+00,
     & 0.53271D+00,0.52452D+00,0.51767D+00,0.51203D+00,0.50749D+00,
     & 0.50395D+00,0.50135D+00,0.49959D+00,0.49862D+00,0.49838D+00,
     & 0.49883D+00,0.49994D+00,0.50166D+00,0.50398D+00,0.50687D+00,
     & 0.51030D+00,0.51427D+00,0.51875D+00,0.52374D+00,0.52924D+00,
     & 0.53522D+00,0.54168D+00,0.54863D+00,0.55606D+00,0.56396D+00,
     & 0.57234D+00,0.58120D+00,0.59054D+00,0.60037D+00,0.61070D+00,
     & 0.62151D+00,0.63284D+00,0.64467D+00,0.65703D+00,0.66992D+00,
     & 0.68335D+00,0.69734D+00,0.71189D+00,0.72703D+00,0.74277D+00,
     & 0.75912D+00,0.77611D+00,0.79375D+00,0.81206D+00,0.83106D+00,
     & 0.85077D+00,0.87123D+00,0.89244D+00,0.91445D+00,0.93727D+00,
     & 0.96093D+00,0.98547D+00,0.10109D+01,0.10373D+01,0.10647D+01,
     & 0.10930D+01,0.11224D+01,0.11529D+01,0.11845D+01,0.12173D+01,
     & 0.12513D+01,0.12865D+01,0.13230D+01,0.13609D+01,0.14002D+01,
     & 0.14409D+01,0.14831D+01,0.15269D+01,0.15724D+01,0.16194D+01,
     & 0.16683D+01,0.17189D+01,0.17714D+01,0.18259D+01,0.18824D+01,
     & 0.19409D+01,0.20016D+01,0.20646D+01,0.21298D+01,0.21975D+01,
     & 0.22677D+01,0.23406D+01,0.24534D+01,0.25269D+01,0.25189D+01,
     & 0.23867D+01,0.18913D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.53939D+00,0.52920D+00,0.52053D+00,0.51325D+00,0.50722D+00,
     & 0.50232D+00,0.49847D+00,0.49556D+00,0.49353D+00,0.49230D+00,
     & 0.49182D+00,0.49205D+00,0.49293D+00,0.49444D+00,0.49655D+00,
     & 0.49923D+00,0.50246D+00,0.50623D+00,0.51051D+00,0.51530D+00,
     & 0.52058D+00,0.52636D+00,0.53261D+00,0.53934D+00,0.54655D+00,
     & 0.55423D+00,0.56238D+00,0.57100D+00,0.58010D+00,0.58968D+00,
     & 0.59975D+00,0.61030D+00,0.62135D+00,0.63290D+00,0.64496D+00,
     & 0.65755D+00,0.67067D+00,0.68434D+00,0.69857D+00,0.71336D+00,
     & 0.72874D+00,0.74473D+00,0.76134D+00,0.77859D+00,0.79649D+00,
     & 0.81507D+00,0.83436D+00,0.85436D+00,0.87512D+00,0.89664D+00,
     & 0.91897D+00,0.94212D+00,0.96613D+00,0.99103D+00,0.10168D+01,
     & 0.10436D+01,0.10714D+01,0.11001D+01,0.11300D+01,0.11609D+01,
     & 0.11930D+01,0.12263D+01,0.12608D+01,0.12966D+01,0.13336D+01,
     & 0.13721D+01,0.14120D+01,0.14534D+01,0.14963D+01,0.15408D+01,
     & 0.15869D+01,0.16348D+01,0.16844D+01,0.17359D+01,0.17893D+01,
     & 0.18446D+01,0.19021D+01,0.19616D+01,0.20233D+01,0.20873D+01,
     & 0.21537D+01,0.22225D+01,0.22940D+01,0.24042D+01,0.24764D+01,
     & 0.24699D+01,0.23436D+01,0.18606D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.52672D+00,0.51753D+00,0.50979D+00,
     & 0.50333D+00,0.49806D+00,0.49385D+00,0.49063D+00,0.48831D+00,
     & 0.48681D+00,0.48608D+00,0.48607D+00,0.48673D+00,0.48803D+00,
     & 0.48992D+00,0.49240D+00,0.49543D+00,0.49900D+00,0.50308D+00,
     & 0.50767D+00,0.51276D+00,0.51833D+00,0.52438D+00,0.53091D+00,
     & 0.53791D+00,0.54537D+00,0.55330D+00,0.56170D+00,0.57058D+00,
     & 0.57992D+00,0.58974D+00,0.60005D+00,0.61084D+00,0.62213D+00,
     & 0.63392D+00,0.64622D+00,0.65905D+00,0.67242D+00,0.68634D+00,
     & 0.70081D+00,0.71587D+00,0.73151D+00,0.74777D+00,0.76465D+00,
     & 0.78217D+00,0.80037D+00,0.81924D+00,0.83884D+00,0.85916D+00,
     & 0.88024D+00,0.90210D+00,0.92478D+00,0.94829D+00,0.97268D+00,
     & 0.99797D+00,0.10242D+01,0.10514D+01,0.10796D+01,0.11088D+01,
     & 0.11391D+01,0.11706D+01,0.12032D+01,0.12370D+01,0.12721D+01,
     & 0.13084D+01,0.13461D+01,0.13852D+01,0.14258D+01,0.14679D+01,
     & 0.15115D+01,0.15568D+01,0.16037D+01,0.16524D+01,0.17029D+01,
     & 0.17553D+01,0.18096D+01,0.18659D+01,0.19244D+01,0.19849D+01,
     & 0.20478D+01,0.21130D+01,0.21806D+01,0.22508D+01,0.23584D+01,
     & 0.24294D+01,0.24244D+01,0.23036D+01,0.18323D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.52527D+00,0.51552D+00,
     & 0.50727D+00,0.50036D+00,0.49468D+00,0.49011D+00,0.48654D+00,
     & 0.48391D+00,0.48214D+00,0.48115D+00,0.48089D+00,0.48132D+00,
     & 0.48239D+00,0.48408D+00,0.48634D+00,0.48917D+00,0.49254D+00,
     & 0.49643D+00,0.50083D+00,0.50572D+00,0.51110D+00,0.51696D+00,
     & 0.52329D+00,0.53008D+00,0.53734D+00,0.54507D+00,0.55326D+00,
     & 0.56191D+00,0.57104D+00,0.58063D+00,0.59070D+00,0.60125D+00,
     & 0.61229D+00,0.62383D+00,0.63587D+00,0.64843D+00,0.66152D+00,
     & 0.67514D+00,0.68932D+00,0.70406D+00,0.71939D+00,0.73532D+00,
     & 0.75186D+00,0.76903D+00,0.78686D+00,0.80536D+00,0.82456D+00,
     & 0.84448D+00,0.86514D+00,0.88658D+00,0.90881D+00,0.93186D+00,
     & 0.95577D+00,0.98057D+00,0.10063D+01,0.10329D+01,0.10606D+01,
     & 0.10893D+01,0.11190D+01,0.11499D+01,0.11818D+01,0.12150D+01,
     & 0.12494D+01,0.12851D+01,0.13221D+01,0.13605D+01,0.14003D+01,
     & 0.14416D+01,0.14844D+01,0.15288D+01,0.15749D+01,0.16227D+01,
     & 0.16723D+01,0.17237D+01,0.17771D+01,0.18324D+01,0.18898D+01,
     & 0.19493D+01,0.20110D+01,0.20751D+01,0.21415D+01,0.22105D+01,
     & 0.23157D+01,0.23856D+01,0.23820D+01,0.22666D+01,0.18061D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.51450D+00,0.50570D+00,0.49831D+00,0.49218D+00,0.48721D+00,
     & 0.48329D+00,0.48033D+00,0.47825D+00,0.47699D+00,0.47647D+00,
     & 0.47666D+00,0.47751D+00,0.47897D+00,0.48104D+00,0.48366D+00,
     & 0.48683D+00,0.49053D+00,0.49474D+00,0.49944D+00,0.50463D+00,
     & 0.51030D+00,0.51644D+00,0.52304D+00,0.53011D+00,0.53764D+00,
     & 0.54562D+00,0.55407D+00,0.56299D+00,0.57236D+00,0.58221D+00,
     & 0.59254D+00,0.60334D+00,0.61464D+00,0.62644D+00,0.63874D+00,
     & 0.65157D+00,0.66492D+00,0.67882D+00,0.69328D+00,0.70831D+00,
     & 0.72392D+00,0.74015D+00,0.75699D+00,0.77448D+00,0.79263D+00,
     & 0.81147D+00,0.83101D+00,0.85128D+00,0.87231D+00,0.89413D+00,
     & 0.91675D+00,0.94022D+00,0.96455D+00,0.98979D+00,0.10160D+01,
     & 0.10431D+01,0.10712D+01,0.11004D+01,0.11307D+01,0.11621D+01,
     & 0.11947D+01,0.12285D+01,0.12635D+01,0.12998D+01,0.13375D+01,
     & 0.13766D+01,0.14172D+01,0.14593D+01,0.15029D+01,0.15482D+01,
     & 0.15951D+01,0.16439D+01,0.16944D+01,0.17468D+01,0.18012D+01,
     & 0.18576D+01,0.19162D+01,0.19769D+01,0.20398D+01,0.21051D+01,
     & 0.21730D+01,0.22759D+01,0.23448D+01,0.23425D+01,0.22322D+01,
     & 0.17820D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.51450D+00,0.50510D+00,0.49718D+00,0.49058D+00,
     & 0.48518D+00,0.48087D+00,0.47755D+00,0.47515D+00,0.47360D+00,
     & 0.47281D+00,0.47275D+00,0.47336D+00,0.47460D+00,0.47645D+00,
     & 0.47888D+00,0.48185D+00,0.48535D+00,0.48937D+00,0.49389D+00,
     & 0.49889D+00,0.50437D+00,0.51033D+00,0.51675D+00,0.52363D+00,
     & 0.53096D+00,0.53876D+00,0.54702D+00,0.55573D+00,0.56490D+00,
     & 0.57454D+00,0.58466D+00,0.59524D+00,0.60632D+00,0.61788D+00,
     & 0.62995D+00,0.64253D+00,0.65563D+00,0.66927D+00,0.68345D+00,
     & 0.69820D+00,0.71353D+00,0.72946D+00,0.74600D+00,0.76317D+00,
     & 0.78099D+00,0.79949D+00,0.81868D+00,0.83859D+00,0.85925D+00,
     & 0.88067D+00,0.90289D+00,0.92594D+00,0.94985D+00,0.97464D+00,
     & 0.10003D+01,0.10270D+01,0.10547D+01,0.10833D+01,0.11131D+01,
     & 0.11440D+01,0.11760D+01,0.12091D+01,0.12436D+01,0.12793D+01,
     & 0.13163D+01,0.13548D+01,0.13947D+01,0.14360D+01,0.14789D+01,
     & 0.15235D+01,0.15696D+01,0.16176D+01,0.16673D+01,0.17188D+01,
     & 0.17723D+01,0.18278D+01,0.18854D+01,0.19451D+01,0.20071D+01,
     & 0.20713D+01,0.21381D+01,0.22389D+01,0.23068D+01,0.23059D+01,
     & 0.22005D+01,0.17598D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.50548D+00,0.49698D+00,
     & 0.48986D+00,0.48400D+00,0.47927D+00,0.47558D+00,0.47284D+00,
     & 0.47097D+00,0.46990D+00,0.46957D+00,0.46994D+00,0.47095D+00,
     & 0.47258D+00,0.47480D+00,0.47757D+00,0.48088D+00,0.48471D+00,
     & 0.48904D+00,0.49386D+00,0.49916D+00,0.50493D+00,0.51117D+00,
     & 0.51787D+00,0.52503D+00,0.53264D+00,0.54071D+00,0.54923D+00,
     & 0.55821D+00,0.56766D+00,0.57757D+00,0.58795D+00,0.59881D+00,
     & 0.61016D+00,0.62200D+00,0.63435D+00,0.64722D+00,0.66061D+00,
     & 0.67454D+00,0.68903D+00,0.70409D+00,0.71974D+00,0.73599D+00,
     & 0.75287D+00,0.77038D+00,0.78856D+00,0.80743D+00,0.82700D+00,
     & 0.84730D+00,0.86837D+00,0.89021D+00,0.91287D+00,0.93638D+00,
     & 0.96075D+00,0.98603D+00,0.10122D+01,0.10395D+01,0.10676D+01,
     & 0.10969D+01,0.11272D+01,0.11587D+01,0.11914D+01,0.12252D+01,
     & 0.12604D+01,0.12968D+01,0.13346D+01,0.13739D+01,0.14146D+01,
     & 0.14568D+01,0.15006D+01,0.15460D+01,0.15932D+01,0.16421D+01,
     & 0.16929D+01,0.17455D+01,0.18002D+01,0.18569D+01,0.19157D+01,
     & 0.19767D+01,0.20399D+01,0.21057D+01,0.22045D+01,0.22714D+01,
     & 0.22718D+01,0.21712D+01,0.17393D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.50687D+00,
     & 0.49774D+00,0.49007D+00,0.48370D+00,0.47852D+00,0.47442D+00,
     & 0.47131D+00,0.46910D+00,0.46773D+00,0.46712D+00,0.46723D+00,
     & 0.46800D+00,0.46941D+00,0.47140D+00,0.47397D+00,0.47709D+00,
     & 0.48072D+00,0.48487D+00,0.48951D+00,0.49463D+00,0.50023D+00,
     & 0.50629D+00,0.51282D+00,0.51980D+00,0.52723D+00,0.53512D+00,
     & 0.54347D+00,0.55226D+00,0.56152D+00,0.57124D+00,0.58143D+00,
     & 0.59209D+00,0.60323D+00,0.61487D+00,0.62700D+00,0.63964D+00,
     & 0.65281D+00,0.66650D+00,0.68075D+00,0.69556D+00,0.71095D+00,
     & 0.72693D+00,0.74353D+00,0.76076D+00,0.77865D+00,0.79720D+00,
     & 0.81646D+00,0.83644D+00,0.85716D+00,0.87866D+00,0.90095D+00,
     & 0.92408D+00,0.94807D+00,0.97294D+00,0.99875D+00,0.10255D+01,
     & 0.10533D+01,0.10821D+01,0.11119D+01,0.11429D+01,0.11750D+01,
     & 0.12084D+01,0.12430D+01,0.12788D+01,0.13161D+01,0.13547D+01,
     & 0.13948D+01,0.14363D+01,0.14795D+01,0.15242D+01,0.15707D+01,
     & 0.16189D+01,0.16689D+01,0.17207D+01,0.17746D+01,0.18304D+01,
     & 0.18884D+01,0.19485D+01,0.20108D+01,0.20757D+01,0.21725D+01,
     & 0.22385D+01,0.22402D+01,0.21441D+01,0.17205D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.49949D+00,0.49120D+00,0.48429D+00,0.47862D+00,
     & 0.47408D+00,0.47057D+00,0.46801D+00,0.46630D+00,0.46540D+00,
     & 0.46523D+00,0.46576D+00,0.46692D+00,0.46870D+00,0.47106D+00,
     & 0.47397D+00,0.47741D+00,0.48137D+00,0.48583D+00,0.49078D+00,
     & 0.49620D+00,0.50209D+00,0.50844D+00,0.51525D+00,0.52252D+00,
     & 0.53023D+00,0.53841D+00,0.54703D+00,0.55611D+00,0.56565D+00,
     & 0.57565D+00,0.58612D+00,0.59707D+00,0.60851D+00,0.62044D+00,
     & 0.63287D+00,0.64583D+00,0.65930D+00,0.67332D+00,0.68790D+00,
     & 0.70304D+00,0.71878D+00,0.73512D+00,0.75208D+00,0.76969D+00,
     & 0.78796D+00,0.80692D+00,0.82659D+00,0.84700D+00,0.86817D+00,
     & 0.89013D+00,0.91290D+00,0.93653D+00,0.96103D+00,0.98644D+00,
     & 0.10128D+01,0.10401D+01,0.10685D+01,0.10979D+01,0.11284D+01,
     & 0.11601D+01,0.11929D+01,0.12270D+01,0.12624D+01,0.12990D+01,
     & 0.13371D+01,0.13766D+01,0.14175D+01,0.14600D+01,0.15042D+01,
     & 0.15499D+01,0.15974D+01,0.16467D+01,0.16979D+01,0.17509D+01,
     & 0.18060D+01,0.18631D+01,0.19224D+01,0.19839D+01,0.20479D+01,
     & 0.21428D+01,0.22080D+01,0.22109D+01,0.21192D+01,0.17033D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.50227D+00,0.49331D+00,0.48580D+00,
     & 0.47960D+00,0.47458D+00,0.47064D+00,0.46769D+00,0.46564D+00,
     & 0.46441D+00,0.46396D+00,0.46421D+00,0.46513D+00,0.46667D+00,
     & 0.46881D+00,0.47152D+00,0.47477D+00,0.47854D+00,0.48281D+00,
     & 0.48758D+00,0.49283D+00,0.49855D+00,0.50473D+00,0.51138D+00,
     & 0.51848D+00,0.52603D+00,0.53403D+00,0.54248D+00,0.55139D+00,
     & 0.56076D+00,0.57059D+00,0.58089D+00,0.59166D+00,0.60290D+00,
     & 0.61465D+00,0.62688D+00,0.63963D+00,0.65291D+00,0.66671D+00,
     & 0.68107D+00,0.69599D+00,0.71149D+00,0.72759D+00,0.74430D+00,
     & 0.76165D+00,0.77966D+00,0.79835D+00,0.81773D+00,0.83784D+00,
     & 0.85871D+00,0.88035D+00,0.90279D+00,0.92608D+00,0.95023D+00,
     & 0.97528D+00,0.10013D+01,0.10282D+01,0.10562D+01,0.10852D+01,
     & 0.11152D+01,0.11465D+01,0.11788D+01,0.12124D+01,0.12473D+01,
     & 0.12834D+01,0.13210D+01,0.13599D+01,0.14003D+01,0.14422D+01,
     & 0.14857D+01,0.15309D+01,0.15777D+01,0.16263D+01,0.16768D+01,
     & 0.17291D+01,0.17834D+01,0.18398D+01,0.18983D+01,0.19590D+01,
     & 0.20222D+01,0.21153D+01,0.21797D+01,0.21839D+01,0.20964D+01,
     & 0.16876D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.49644D+00,
     & 0.48828D+00,0.48149D+00,0.47595D+00,0.47155D+00,0.46817D+00,
     & 0.46574D+00,0.46417D+00,0.46340D+00,0.46337D+00,0.46403D+00,
     & 0.46533D+00,0.46724D+00,0.46973D+00,0.47278D+00,0.47635D+00,
     & 0.48044D+00,0.48503D+00,0.49011D+00,0.49566D+00,0.50168D+00,
     & 0.50816D+00,0.51509D+00,0.52248D+00,0.53032D+00,0.53862D+00,
     & 0.54737D+00,0.55657D+00,0.56623D+00,0.57636D+00,0.58696D+00,
     & 0.59803D+00,0.60959D+00,0.62165D+00,0.63421D+00,0.64729D+00,
     & 0.66089D+00,0.67505D+00,0.68975D+00,0.70504D+00,0.72091D+00,
     & 0.73739D+00,0.75450D+00,0.77226D+00,0.79069D+00,0.80981D+00,
     & 0.82965D+00,0.85022D+00,0.87157D+00,0.89371D+00,0.91668D+00,
     & 0.94050D+00,0.96521D+00,0.99084D+00,0.10174D+01,0.10450D+01,
     & 0.10736D+01,0.11033D+01,0.11341D+01,0.11660D+01,0.11992D+01,
     & 0.12335D+01,0.12692D+01,0.13062D+01,0.13447D+01,0.13845D+01,
     & 0.14259D+01,0.14688D+01,0.15134D+01,0.15596D+01,0.16076D+01,
     & 0.16574D+01,0.17091D+01,0.17627D+01,0.18183D+01,0.18761D+01,
     & 0.19361D+01,0.19984D+01,0.20899D+01,0.21535D+01,0.21589D+01,
     & 0.20756D+01,0.16734D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.49175D+00,0.48433D+00,0.47822D+00,0.47331D+00,
     & 0.46948D+00,0.46663D+00,0.46469D+00,0.46359D+00,0.46325D+00,
     & 0.46363D+00,0.46467D+00,0.46635D+00,0.46861D+00,0.47145D+00,
     & 0.47482D+00,0.47872D+00,0.48313D+00,0.48803D+00,0.49341D+00,
     & 0.49927D+00,0.50558D+00,0.51236D+00,0.51959D+00,0.52728D+00,
     & 0.53541D+00,0.54400D+00,0.55305D+00,0.56255D+00,0.57252D+00,
     & 0.58296D+00,0.59386D+00,0.60526D+00,0.61714D+00,0.62952D+00,
     & 0.64242D+00,0.65584D+00,0.66980D+00,0.68432D+00,0.69940D+00,
     & 0.71506D+00,0.73132D+00,0.74821D+00,0.76574D+00,0.78393D+00,
     & 0.80280D+00,0.82238D+00,0.84269D+00,0.86376D+00,0.88562D+00,
     & 0.90829D+00,0.93181D+00,0.95620D+00,0.98150D+00,0.10077D+01,
     & 0.10350D+01,0.10632D+01,0.10925D+01,0.11229D+01,0.11544D+01,
     & 0.11872D+01,0.12211D+01,0.12563D+01,0.12929D+01,0.13308D+01,
     & 0.13702D+01,0.14110D+01,0.14534D+01,0.14974D+01,0.15431D+01,
     & 0.15905D+01,0.16396D+01,0.16907D+01,0.17436D+01,0.17986D+01,
     & 0.18557D+01,0.19149D+01,0.19765D+01,0.20664D+01,0.21293D+01,
     & 0.21359D+01,0.20566D+01,0.16605D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.49629D+00,0.48817D+00,0.48144D+00,
     & 0.47597D+00,0.47164D+00,0.46834D+00,0.46600D+00,0.46453D+00,
     & 0.46387D+00,0.46395D+00,0.46472D+00,0.46613D+00,0.46817D+00,
     & 0.47078D+00,0.47395D+00,0.47766D+00,0.48188D+00,0.48660D+00,
     & 0.49181D+00,0.49750D+00,0.50365D+00,0.51027D+00,0.51735D+00,
     & 0.52488D+00,0.53286D+00,0.54130D+00,0.55020D+00,0.55955D+00,
     & 0.56936D+00,0.57964D+00,0.59039D+00,0.60163D+00,0.61335D+00,
     & 0.62557D+00,0.63829D+00,0.65154D+00,0.66532D+00,0.67965D+00,
     & 0.69454D+00,0.71001D+00,0.72607D+00,0.74275D+00,0.76006D+00,
     & 0.77803D+00,0.79667D+00,0.81601D+00,0.83608D+00,0.85689D+00,
     & 0.87849D+00,0.90088D+00,0.92411D+00,0.94821D+00,0.97320D+00,
     & 0.99913D+00,0.10260D+01,0.10539D+01,0.10829D+01,0.11129D+01,
     & 0.11440D+01,0.11764D+01,0.12099D+01,0.12447D+01,0.12808D+01,
     & 0.13183D+01,0.13572D+01,0.13976D+01,0.14395D+01,0.14829D+01,
     & 0.15280D+01,0.15749D+01,0.16235D+01,0.16739D+01,0.17263D+01,
     & 0.17806D+01,0.18370D+01,0.18956D+01,0.19565D+01,0.20448D+01,
     & 0.21071D+01,0.21148D+01,0.20394D+01,0.16489D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.49307D+00,
     & 0.48565D+00,0.47957D+00,0.47469D+00,0.47091D+00,0.46813D+00,
     & 0.46626D+00,0.46524D+00,0.46500D+00,0.46547D+00,0.46662D+00,
     & 0.46840D+00,0.47079D+00,0.47374D+00,0.47724D+00,0.48127D+00,
     & 0.48581D+00,0.49085D+00,0.49637D+00,0.50236D+00,0.50882D+00,
     & 0.51574D+00,0.52312D+00,0.53096D+00,0.53925D+00,0.54800D+00,
     & 0.55721D+00,0.56687D+00,0.57701D+00,0.58761D+00,0.59869D+00,
     & 0.61026D+00,0.62232D+00,0.63489D+00,0.64797D+00,0.66159D+00,
     & 0.67574D+00,0.69046D+00,0.70574D+00,0.72162D+00,0.73810D+00,
     & 0.75522D+00,0.77298D+00,0.79140D+00,0.81052D+00,0.83036D+00,
     & 0.85094D+00,0.87228D+00,0.89442D+00,0.91739D+00,0.94121D+00,
     & 0.96592D+00,0.99155D+00,0.10181D+01,0.10457D+01,0.10743D+01,
     & 0.11040D+01,0.11348D+01,0.11668D+01,0.11999D+01,0.12343D+01,
     & 0.12700D+01,0.13071D+01,0.13455D+01,0.13854D+01,0.14269D+01,
     & 0.14698D+01,0.15145D+01,0.15608D+01,0.16088D+01,0.16587D+01,
     & 0.17105D+01,0.17642D+01,0.18200D+01,0.18779D+01,0.19382D+01,
     & 0.20250D+01,0.20867D+01,0.20956D+01,0.20240D+01,0.16386D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.49909D+00,0.49093D+00,0.48417D+00,0.47870D+00,0.47438D+00,
     & 0.47111D+00,0.46882D+00,0.46741D+00,0.46681D+00,0.46697D+00,
     & 0.46783D+00,0.46934D+00,0.47148D+00,0.47421D+00,0.47750D+00,
     & 0.48133D+00,0.48568D+00,0.49053D+00,0.49588D+00,0.50171D+00,
     & 0.50801D+00,0.51478D+00,0.52202D+00,0.52971D+00,0.53786D+00,
     & 0.54646D+00,0.55552D+00,0.56505D+00,0.57504D+00,0.58550D+00,
     & 0.59644D+00,0.60786D+00,0.61978D+00,0.63220D+00,0.64513D+00,
     & 0.65859D+00,0.67258D+00,0.68713D+00,0.70225D+00,0.71795D+00,
     & 0.73426D+00,0.75118D+00,0.76875D+00,0.78698D+00,0.80589D+00,
     & 0.82552D+00,0.84587D+00,0.86699D+00,0.88889D+00,0.91161D+00,
     & 0.93518D+00,0.95962D+00,0.98498D+00,0.10113D+01,0.10386D+01,
     & 0.10669D+01,0.10962D+01,0.11267D+01,0.11583D+01,0.11911D+01,
     & 0.12251D+01,0.12605D+01,0.12971D+01,0.13352D+01,0.13746D+01,
     & 0.14156D+01,0.14581D+01,0.15023D+01,0.15481D+01,0.15956D+01,
     & 0.16450D+01,0.16962D+01,0.17494D+01,0.18046D+01,0.18619D+01,
     & 0.19215D+01,0.20070D+01,0.20681D+01,0.20782D+01,0.20103D+01,
     & 0.16295D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.49733D+00,0.48984D+00,0.48370D+00,
     & 0.47879D+00,0.47501D+00,0.47223D+00,0.47040D+00,0.46942D+00,
     & 0.46923D+00,0.46978D+00,0.47101D+00,0.47289D+00,0.47537D+00,
     & 0.47844D+00,0.48206D+00,0.48621D+00,0.49088D+00,0.49605D+00,
     & 0.50172D+00,0.50786D+00,0.51447D+00,0.52156D+00,0.52910D+00,
     & 0.53711D+00,0.54557D+00,0.55450D+00,0.56389D+00,0.57374D+00,
     & 0.58407D+00,0.59487D+00,0.60616D+00,0.61793D+00,0.63021D+00,
     & 0.64300D+00,0.65631D+00,0.67016D+00,0.68455D+00,0.69951D+00,
     & 0.71505D+00,0.73119D+00,0.74795D+00,0.76534D+00,0.78338D+00,
     & 0.80211D+00,0.82153D+00,0.84168D+00,0.86259D+00,0.88427D+00,
     & 0.90676D+00,0.93009D+00,0.95429D+00,0.97939D+00,0.10054D+01,
     & 0.10324D+01,0.10605D+01,0.10895D+01,0.11197D+01,0.11510D+01,
     & 0.11834D+01,0.12171D+01,0.12521D+01,0.12884D+01,0.13260D+01,
     & 0.13651D+01,0.14057D+01,0.14478D+01,0.14914D+01,0.15368D+01,
     & 0.15839D+01,0.16327D+01,0.16834D+01,0.17361D+01,0.17908D+01,
     & 0.18475D+01,0.19066D+01,0.19906D+01,0.20512D+01,0.20624D+01,
     & 0.19983D+01,0.16215D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.49665D+00,
     & 0.48979D+00,0.48423D+00,0.47986D+00,0.47657D+00,0.47427D+00,
     & 0.47287D+00,0.47230D+00,0.47251D+00,0.47343D+00,0.47503D+00,
     & 0.47725D+00,0.48008D+00,0.48348D+00,0.48742D+00,0.49190D+00,
     & 0.49689D+00,0.50238D+00,0.50836D+00,0.51482D+00,0.52175D+00,
     & 0.52915D+00,0.53702D+00,0.54535D+00,0.55414D+00,0.56339D+00,
     & 0.57312D+00,0.58331D+00,0.59399D+00,0.60514D+00,0.61678D+00,
     & 0.62893D+00,0.64158D+00,0.65476D+00,0.66847D+00,0.68272D+00,
     & 0.69753D+00,0.71293D+00,0.72891D+00,0.74550D+00,0.76273D+00,
     & 0.78061D+00,0.79915D+00,0.81840D+00,0.83836D+00,0.85907D+00,
     & 0.88055D+00,0.90283D+00,0.92594D+00,0.94991D+00,0.97478D+00,
     & 0.10006D+01,0.10273D+01,0.10551D+01,0.10839D+01,0.11137D+01,
     & 0.11447D+01,0.11769D+01,0.12103D+01,0.12449D+01,0.12808D+01,
     & 0.13181D+01,0.13568D+01,0.13970D+01,0.14387D+01,0.14820D+01,
     & 0.15269D+01,0.15735D+01,0.16219D+01,0.16722D+01,0.17243D+01,
     & 0.17785D+01,0.18347D+01,0.18932D+01,0.19759D+01,0.20360D+01,
     & 0.20484D+01,0.19879D+01,0.16146D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.50471D+00,0.49704D+00,0.49076D+00,0.48575D+00,0.48188D+00,
     & 0.47906D+00,0.47721D+00,0.47623D+00,0.47606D+00,0.47665D+00,
     & 0.47794D+00,0.47989D+00,0.48246D+00,0.48562D+00,0.48935D+00,
     & 0.49362D+00,0.49842D+00,0.50373D+00,0.50954D+00,0.51584D+00,
     & 0.52262D+00,0.52987D+00,0.53760D+00,0.54579D+00,0.55445D+00,
     & 0.56357D+00,0.57317D+00,0.58324D+00,0.59379D+00,0.60482D+00,
     & 0.61634D+00,0.62836D+00,0.64088D+00,0.65393D+00,0.66751D+00,
     & 0.68163D+00,0.69631D+00,0.71156D+00,0.72740D+00,0.74385D+00,
     & 0.76092D+00,0.77864D+00,0.79703D+00,0.81611D+00,0.83590D+00,
     & 0.85643D+00,0.87772D+00,0.89981D+00,0.92272D+00,0.94648D+00,
     & 0.97113D+00,0.99669D+00,0.10232D+01,0.10507D+01,0.10792D+01,
     & 0.11089D+01,0.11396D+01,0.11715D+01,0.12045D+01,0.12388D+01,
     & 0.12745D+01,0.13114D+01,0.13498D+01,0.13896D+01,0.14309D+01,
     & 0.14738D+01,0.15183D+01,0.15645D+01,0.16125D+01,0.16623D+01,
     & 0.17140D+01,0.17676D+01,0.18234D+01,0.18813D+01,0.19628D+01,
     & 0.20225D+01,0.20360D+01,0.19790D+01,0.16089D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.50555D+00,0.49847D+00,0.49275D+00,
     & 0.48825D+00,0.48486D+00,0.48249D+00,0.48106D+00,0.48049D+00,
     & 0.48071D+00,0.48166D+00,0.48331D+00,0.48560D+00,0.48851D+00,
     & 0.49200D+00,0.49606D+00,0.50066D+00,0.50579D+00,0.51142D+00,
     & 0.51755D+00,0.52417D+00,0.53128D+00,0.53886D+00,0.54692D+00,
     & 0.55544D+00,0.56444D+00,0.57391D+00,0.58386D+00,0.59429D+00,
     & 0.60519D+00,0.61660D+00,0.62850D+00,0.64091D+00,0.65383D+00,
     & 0.66729D+00,0.68129D+00,0.69584D+00,0.71096D+00,0.72667D+00,
     & 0.74299D+00,0.75992D+00,0.77750D+00,0.79574D+00,0.81466D+00,
     & 0.83429D+00,0.85466D+00,0.87578D+00,0.89769D+00,0.92042D+00,
     & 0.94398D+00,0.96843D+00,0.99379D+00,0.10201D+01,0.10474D+01,
     & 0.10757D+01,0.11050D+01,0.11355D+01,0.11671D+01,0.11999D+01,
     & 0.12339D+01,0.12693D+01,0.13059D+01,0.13439D+01,0.13834D+01,
     & 0.14244D+01,0.14669D+01,0.15110D+01,0.15569D+01,0.16044D+01,
     & 0.16538D+01,0.17051D+01,0.17583D+01,0.18136D+01,0.18710D+01,
     & 0.19513D+01,0.20106D+01,0.20252D+01,0.19717D+01,0.16041D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.51544D+00,0.50748D+00,
     & 0.50096D+00,0.49575D+00,0.49174D+00,0.48881D+00,0.48687D+00,
     & 0.48584D+00,0.48566D+00,0.48625D+00,0.48756D+00,0.48956D+00,
     & 0.49219D+00,0.49544D+00,0.49926D+00,0.50365D+00,0.50857D+00,
     & 0.51402D+00,0.51998D+00,0.52644D+00,0.53339D+00,0.54083D+00,
     & 0.54875D+00,0.55714D+00,0.56601D+00,0.57536D+00,0.58518D+00,
     & 0.59549D+00,0.60629D+00,0.61758D+00,0.62937D+00,0.64166D+00,
     & 0.65448D+00,0.66782D+00,0.68170D+00,0.69614D+00,0.71114D+00,
     & 0.72673D+00,0.74292D+00,0.75973D+00,0.77718D+00,0.79528D+00,
     & 0.81406D+00,0.83355D+00,0.85377D+00,0.87473D+00,0.89648D+00,
     & 0.91904D+00,0.94243D+00,0.96669D+00,0.99186D+00,0.10180D+01,
     & 0.10450D+01,0.10731D+01,0.11023D+01,0.11325D+01,0.11638D+01,
     & 0.11964D+01,0.12302D+01,0.12652D+01,0.13016D+01,0.13393D+01,
     & 0.13785D+01,0.14191D+01,0.14613D+01,0.15051D+01,0.15506D+01,
     & 0.15977D+01,0.16467D+01,0.16976D+01,0.17504D+01,0.18052D+01,
     & 0.18622D+01,0.19414D+01,0.20002D+01,0.20160D+01,0.19660D+01,
     & 0.16005D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.51790D+00,0.51050D+00,0.50451D+00,0.49979D+00,0.49623D+00,
     & 0.49373D+00,0.49220D+00,0.49157D+00,0.49176D+00,0.49271D+00,
     & 0.49438D+00,0.49671D+00,0.49968D+00,0.50326D+00,0.50742D+00,
     & 0.51213D+00,0.51739D+00,0.52316D+00,0.52945D+00,0.53624D+00,
     & 0.54353D+00,0.55130D+00,0.55956D+00,0.56831D+00,0.57753D+00,
     & 0.58724D+00,0.59743D+00,0.60812D+00,0.61930D+00,0.63098D+00,
     & 0.64316D+00,0.65587D+00,0.66911D+00,0.68288D+00,0.69722D+00,
     & 0.71211D+00,0.72759D+00,0.74366D+00,0.76036D+00,0.77768D+00,
     & 0.79567D+00,0.81432D+00,0.83368D+00,0.85376D+00,0.87458D+00,
     & 0.89618D+00,0.91859D+00,0.94182D+00,0.96592D+00,0.99091D+00,
     & 0.10168D+01,0.10437D+01,0.10716D+01,0.11005D+01,0.11306D+01,
     & 0.11617D+01,0.11940D+01,0.12275D+01,0.12623D+01,0.12984D+01,
     & 0.13359D+01,0.13748D+01,0.14151D+01,0.14570D+01,0.15005D+01,
     & 0.15456D+01,0.15924D+01,0.16410D+01,0.16915D+01,0.17439D+01,
     & 0.17984D+01,0.18550D+01,0.19330D+01,0.19915D+01,0.20084D+01,
     & 0.19618D+01,0.15978D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.52151D+00,0.51463D+00,0.50913D+00,
     & 0.50487D+00,0.50174D+00,0.49966D+00,0.49852D+00,0.49826D+00,
     & 0.49882D+00,0.50012D+00,0.50213D+00,0.50480D+00,0.50811D+00,
     & 0.51202D+00,0.51650D+00,0.52155D+00,0.52713D+00,0.53324D+00,
     & 0.53987D+00,0.54699D+00,0.55462D+00,0.56274D+00,0.57136D+00,
     & 0.58046D+00,0.59005D+00,0.60013D+00,0.61070D+00,0.62177D+00,
     & 0.63335D+00,0.64544D+00,0.65804D+00,0.67118D+00,0.68486D+00,
     & 0.69909D+00,0.71388D+00,0.72926D+00,0.74523D+00,0.76182D+00,
     & 0.77904D+00,0.79691D+00,0.81545D+00,0.83469D+00,0.85464D+00,
     & 0.87534D+00,0.89681D+00,0.91908D+00,0.94217D+00,0.96611D+00,
     & 0.99095D+00,0.10167D+01,0.10434D+01,0.10711D+01,0.10999D+01,
     & 0.11297D+01,0.11606D+01,0.11927D+01,0.12260D+01,0.12606D+01,
     & 0.12965D+01,0.13337D+01,0.13723D+01,0.14124D+01,0.14540D+01,
     & 0.14971D+01,0.15419D+01,0.15884D+01,0.16367D+01,0.16868D+01,
     & 0.17389D+01,0.17930D+01,0.18492D+01,0.19262D+01,0.19843D+01,
     & 0.20024D+01,0.19591D+01,0.15961D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.53414D+00,0.52628D+00,
     & 0.51989D+00,0.51485D+00,0.51102D+00,0.50831D+00,0.50661D+00,
     & 0.50586D+00,0.50596D+00,0.50686D+00,0.50851D+00,0.51086D+00,
     & 0.51387D+00,0.51750D+00,0.52174D+00,0.52656D+00,0.53194D+00,
     & 0.53785D+00,0.54430D+00,0.55127D+00,0.55874D+00,0.56672D+00,
     & 0.57520D+00,0.58417D+00,0.59364D+00,0.60361D+00,0.61407D+00,
     & 0.62504D+00,0.63652D+00,0.64851D+00,0.66102D+00,0.67406D+00,
     & 0.68765D+00,0.70178D+00,0.71649D+00,0.73177D+00,0.74765D+00,
     & 0.76414D+00,0.78126D+00,0.79903D+00,0.81747D+00,0.83660D+00,
     & 0.85645D+00,0.87703D+00,0.89838D+00,0.92053D+00,0.94349D+00,
     & 0.96730D+00,0.99200D+00,0.10176D+01,0.10442D+01,0.10717D+01,
     & 0.11003D+01,0.11300D+01,0.11607D+01,0.11926D+01,0.12257D+01,
     & 0.12601D+01,0.12957D+01,0.13327D+01,0.13711D+01,0.14109D+01,
     & 0.14522D+01,0.14951D+01,0.15396D+01,0.15858D+01,0.16338D+01,
     & 0.16836D+01,0.17353D+01,0.17891D+01,0.18449D+01,0.19209D+01,
     & 0.19787D+01,0.19979D+01,0.19581D+01,0.15954D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.53961D+00,0.53224D+00,0.52631D+00,0.52169D+00,0.51828D+00,
     & 0.51595D+00,0.51463D+00,0.51424D+00,0.51469D+00,0.51594D+00,
     & 0.51792D+00,0.52061D+00,0.52395D+00,0.52791D+00,0.53249D+00,
     & 0.53764D+00,0.54335D+00,0.54960D+00,0.55639D+00,0.56371D+00,
     & 0.57153D+00,0.57987D+00,0.58872D+00,0.59806D+00,0.60791D+00,
     & 0.61827D+00,0.62914D+00,0.64051D+00,0.65241D+00,0.66483D+00,
     & 0.67778D+00,0.69128D+00,0.70533D+00,0.71995D+00,0.73515D+00,
     & 0.75094D+00,0.76735D+00,0.78438D+00,0.80206D+00,0.82041D+00,
     & 0.83945D+00,0.85920D+00,0.87968D+00,0.90093D+00,0.92296D+00,
     & 0.94581D+00,0.96951D+00,0.99408D+00,0.10196D+01,0.10460D+01,
     & 0.10734D+01,0.11018D+01,0.11313D+01,0.11619D+01,0.11936D+01,
     & 0.12266D+01,0.12607D+01,0.12962D+01,0.13329D+01,0.13711D+01,
     & 0.14107D+01,0.14518D+01,0.14944D+01,0.15387D+01,0.15846D+01,
     & 0.16323D+01,0.16818D+01,0.17333D+01,0.17867D+01,0.18422D+01,
     & 0.19172D+01,0.19747D+01,0.19951D+01,0.19586D+01,0.15957D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.54635D+00,0.53943D+00,0.53392D+00,
     & 0.52971D+00,0.52668D+00,0.52473D+00,0.52377D+00,0.52372D+00,
     & 0.52451D+00,0.52609D+00,0.52841D+00,0.53142D+00,0.53509D+00,
     & 0.53939D+00,0.54430D+00,0.54978D+00,0.55583D+00,0.56243D+00,
     & 0.56957D+00,0.57724D+00,0.58543D+00,0.59414D+00,0.60336D+00,
     & 0.61309D+00,0.62334D+00,0.63410D+00,0.64538D+00,0.65718D+00,
     & 0.66951D+00,0.68238D+00,0.69580D+00,0.70977D+00,0.72431D+00,
     & 0.73943D+00,0.75515D+00,0.77147D+00,0.78843D+00,0.80603D+00,
     & 0.82430D+00,0.84326D+00,0.86292D+00,0.88332D+00,0.90447D+00,
     & 0.92642D+00,0.94917D+00,0.97276D+00,0.99722D+00,0.10226D+01,
     & 0.10489D+01,0.10762D+01,0.11045D+01,0.11338D+01,0.11643D+01,
     & 0.11959D+01,0.12286D+01,0.12626D+01,0.12979D+01,0.13345D+01,
     & 0.13724D+01,0.14118D+01,0.14527D+01,0.14951D+01,0.15391D+01,
     & 0.15848D+01,0.16323D+01,0.16815D+01,0.17327D+01,0.17858D+01,
     & 0.18410D+01,0.19150D+01,0.19723D+01,0.19939D+01,0.19607D+01,
     & 0.15969D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.56247D+00,0.55441D+00,
     & 0.54790D+00,0.54278D+00,0.53895D+00,0.53628D+00,0.53468D+00,
     & 0.53407D+00,0.53435D+00,0.53548D+00,0.53739D+00,0.54003D+00,
     & 0.54337D+00,0.54737D+00,0.55200D+00,0.55724D+00,0.56306D+00,
     & 0.56946D+00,0.57641D+00,0.58391D+00,0.59194D+00,0.60050D+00,
     & 0.60959D+00,0.61920D+00,0.62933D+00,0.63999D+00,0.65117D+00,
     & 0.66288D+00,0.67512D+00,0.68791D+00,0.70125D+00,0.71514D+00,
     & 0.72961D+00,0.74466D+00,0.76031D+00,0.77657D+00,0.79346D+00,
     & 0.81099D+00,0.82919D+00,0.84807D+00,0.86767D+00,0.88799D+00,
     & 0.90907D+00,0.93093D+00,0.95359D+00,0.97710D+00,0.10015D+01,
     & 0.10267D+01,0.10530D+01,0.10801D+01,0.11083D+01,0.11375D+01,
     & 0.11679D+01,0.11993D+01,0.12319D+01,0.12658D+01,0.13009D+01,
     & 0.13373D+01,0.13751D+01,0.14143D+01,0.14550D+01,0.14972D+01,
     & 0.15410D+01,0.15865D+01,0.16337D+01,0.16827D+01,0.17336D+01,
     & 0.17865D+01,0.18414D+01,0.19145D+01,0.19715D+01,0.19943D+01,
     & 0.19645D+01,0.15991D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.57151D+00,0.56384D+00,0.55771D+00,0.55296D+00,0.54948D+00,
     & 0.54715D+00,0.54589D+00,0.54560D+00,0.54621D+00,0.54766D+00,
     & 0.54989D+00,0.55285D+00,0.55651D+00,0.56084D+00,0.56581D+00,
     & 0.57138D+00,0.57755D+00,0.58430D+00,0.59161D+00,0.59947D+00,
     & 0.60788D+00,0.61683D+00,0.62631D+00,0.63632D+00,0.64687D+00,
     & 0.65794D+00,0.66956D+00,0.68172D+00,0.69442D+00,0.70768D+00,
     & 0.72151D+00,0.73591D+00,0.75090D+00,0.76648D+00,0.78268D+00,
     & 0.79951D+00,0.81698D+00,0.83512D+00,0.85395D+00,0.87348D+00,
     & 0.89374D+00,0.91475D+00,0.93655D+00,0.95915D+00,0.98258D+00,
     & 0.10069D+01,0.10321D+01,0.10582D+01,0.10853D+01,0.11134D+01,
     & 0.11425D+01,0.11727D+01,0.12040D+01,0.12365D+01,0.12703D+01,
     & 0.13052D+01,0.13415D+01,0.13792D+01,0.14182D+01,0.14587D+01,
     & 0.15008D+01,0.15444D+01,0.15897D+01,0.16367D+01,0.16854D+01,
     & 0.17361D+01,0.17888D+01,0.18435D+01,0.19157D+01,0.19725D+01,
     & 0.19965D+01,0.19700D+01,0.16023D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.58204D+00,0.57473D+00,0.56894D+00,
     & 0.56452D+00,0.56137D+00,0.55936D+00,0.55841D+00,0.55844D+00,
     & 0.55936D+00,0.56113D+00,0.56367D+00,0.56696D+00,0.57094D+00,
     & 0.57560D+00,0.58090D+00,0.58682D+00,0.59334D+00,0.60044D+00,
     & 0.60812D+00,0.61636D+00,0.62515D+00,0.63449D+00,0.64438D+00,
     & 0.65481D+00,0.66578D+00,0.67730D+00,0.68937D+00,0.70200D+00,
     & 0.71518D+00,0.72894D+00,0.74327D+00,0.75820D+00,0.77373D+00,
     & 0.78987D+00,0.80665D+00,0.82407D+00,0.84217D+00,0.86094D+00,
     & 0.88043D+00,0.90064D+00,0.92160D+00,0.94334D+00,0.96588D+00,
     & 0.98926D+00,0.10135D+01,0.10386D+01,0.10647D+01,0.10917D+01,
     & 0.11197D+01,0.11487D+01,0.11789D+01,0.12101D+01,0.12425D+01,
     & 0.12761D+01,0.13110D+01,0.13471D+01,0.13847D+01,0.14236D+01,
     & 0.14640D+01,0.15058D+01,0.15493D+01,0.15944D+01,0.16412D+01,
     & 0.16898D+01,0.17403D+01,0.17927D+01,0.18472D+01,0.19186D+01,
     & 0.19753D+01,0.20005D+01,0.19773D+01,0.16064D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.59415D+00,
     & 0.58716D+00,0.58168D+00,0.57757D+00,0.57472D+00,0.57301D+00,
     & 0.57237D+00,0.57269D+00,0.57392D+00,0.57598D+00,0.57884D+00,
     & 0.58244D+00,0.58675D+00,0.59174D+00,0.59737D+00,0.60364D+00,
     & 0.61051D+00,0.61798D+00,0.62603D+00,0.63466D+00,0.64385D+00,
     & 0.65360D+00,0.66390D+00,0.67477D+00,0.68619D+00,0.69817D+00,
     & 0.71071D+00,0.72382D+00,0.73751D+00,0.75178D+00,0.76666D+00,
     & 0.78213D+00,0.79823D+00,0.81496D+00,0.83234D+00,0.85039D+00,
     & 0.86913D+00,0.88858D+00,0.90875D+00,0.92967D+00,0.95137D+00,
     & 0.97388D+00,0.99721D+00,0.10214D+01,0.10465D+01,0.10725D+01,
     & 0.10994D+01,0.11274D+01,0.11564D+01,0.11864D+01,0.12176D+01,
     & 0.12499D+01,0.12835D+01,0.13182D+01,0.13543D+01,0.13917D+01,
     & 0.14305D+01,0.14708D+01,0.15125D+01,0.15559D+01,0.16008D+01,
     & 0.16475D+01,0.16959D+01,0.17462D+01,0.17985D+01,0.18528D+01,
     & 0.19233D+01,0.19798D+01,0.20063D+01,0.19865D+01,0.16115D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.61635D+00,0.60798D+00,0.60126D+00,0.59605D+00,0.59222D+00,
     & 0.58964D+00,0.58822D+00,0.58785D+00,0.58846D+00,0.58998D+00,
     & 0.59234D+00,0.59550D+00,0.59941D+00,0.60404D+00,0.60936D+00,
     & 0.61534D+00,0.62196D+00,0.62919D+00,0.63704D+00,0.64547D+00,
     & 0.65449D+00,0.66409D+00,0.67427D+00,0.68501D+00,0.69632D+00,
     & 0.70820D+00,0.72066D+00,0.73370D+00,0.74732D+00,0.76153D+00,
     & 0.77635D+00,0.79178D+00,0.80783D+00,0.82453D+00,0.84188D+00,
     & 0.85989D+00,0.87860D+00,0.89802D+00,0.91817D+00,0.93906D+00,
     & 0.96074D+00,0.98321D+00,0.10065D+01,0.10307D+01,0.10557D+01,
     & 0.10817D+01,0.11086D+01,0.11365D+01,0.11655D+01,0.11955D+01,
     & 0.12266D+01,0.12589D+01,0.12923D+01,0.13270D+01,0.13630D+01,
     & 0.14003D+01,0.14391D+01,0.14792D+01,0.15209D+01,0.15641D+01,
     & 0.16090D+01,0.16555D+01,0.17038D+01,0.17539D+01,0.18060D+01,
     & 0.18602D+01,0.19299D+01,0.19864D+01,0.20141D+01,0.19976D+01,
     & 0.16177D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.63179D+00,0.62365D+00,0.61716D+00,
     & 0.61219D+00,0.60860D+00,0.60627D+00,0.60510D+00,0.60500D+00,
     & 0.60588D+00,0.60767D+00,0.61033D+00,0.61379D+00,0.61801D+00,
     & 0.62296D+00,0.62861D+00,0.63493D+00,0.64191D+00,0.64951D+00,
     & 0.65774D+00,0.66657D+00,0.67600D+00,0.68602D+00,0.69663D+00,
     & 0.70783D+00,0.71961D+00,0.73198D+00,0.74493D+00,0.75848D+00,
     & 0.77264D+00,0.78740D+00,0.80278D+00,0.81880D+00,0.83546D+00,
     & 0.85278D+00,0.87078D+00,0.88946D+00,0.90886D+00,0.92899D+00,
     & 0.94987D+00,0.97154D+00,0.99400D+00,0.10173D+01,0.10414D+01,
     & 0.10665D+01,0.10924D+01,0.11193D+01,0.11472D+01,0.11761D+01,
     & 0.12061D+01,0.12372D+01,0.12694D+01,0.13029D+01,0.13375D+01,
     & 0.13735D+01,0.14107D+01,0.14494D+01,0.14895D+01,0.15310D+01,
     & 0.15742D+01,0.16189D+01,0.16653D+01,0.17135D+01,0.17636D+01,
     & 0.18156D+01,0.18696D+01,0.19386D+01,0.19950D+01,0.20240D+01,
     & 0.20109D+01,0.16248D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.64931D+00,
     & 0.64133D+00,0.63503D+00,0.63026D+00,0.62688D+00,0.62477D+00,
     & 0.62383D+00,0.62397D+00,0.62510D+00,0.62717D+00,0.63010D+00,
     & 0.63385D+00,0.63838D+00,0.64365D+00,0.64964D+00,0.65631D+00,
     & 0.66364D+00,0.67163D+00,0.68024D+00,0.68948D+00,0.69934D+00,
     & 0.70980D+00,0.72086D+00,0.73252D+00,0.74479D+00,0.75766D+00,
     & 0.77114D+00,0.78523D+00,0.79994D+00,0.81528D+00,0.83126D+00,
     & 0.84789D+00,0.86519D+00,0.88317D+00,0.90184D+00,0.92123D+00,
     & 0.94135D+00,0.96223D+00,0.98390D+00,0.10064D+01,0.10296D+01,
     & 0.10538D+01,0.10788D+01,0.11048D+01,0.11317D+01,0.11596D+01,
     & 0.11885D+01,0.12185D+01,0.12495D+01,0.12818D+01,0.13152D+01,
     & 0.13498D+01,0.13857D+01,0.14230D+01,0.14616D+01,0.15016D+01,
     & 0.15431D+01,0.15862D+01,0.16309D+01,0.16773D+01,0.17254D+01,
     & 0.17753D+01,0.18273D+01,0.18812D+01,0.19494D+01,0.20057D+01,
     & 0.20361D+01,0.20264D+01,0.16329D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.66910D+00,0.66124D+00,0.65508D+00,0.65046D+00,
     & 0.64724D+00,0.64532D+00,0.64459D+00,0.64495D+00,0.64632D+00,
     & 0.64864D+00,0.65184D+00,0.65588D+00,0.66071D+00,0.66630D+00,
     & 0.67262D+00,0.67964D+00,0.68735D+00,0.69572D+00,0.70474D+00,
     & 0.71439D+00,0.72468D+00,0.73560D+00,0.74713D+00,0.75928D+00,
     & 0.77206D+00,0.78545D+00,0.79948D+00,0.81413D+00,0.82942D+00,
     & 0.84537D+00,0.86197D+00,0.87925D+00,0.89722D+00,0.91589D+00,
     & 0.93528D+00,0.95540D+00,0.97629D+00,0.99797D+00,0.10204D+01,
     & 0.10438D+01,0.10679D+01,0.10930D+01,0.11190D+01,0.11459D+01,
     & 0.11738D+01,0.12027D+01,0.12327D+01,0.12638D+01,0.12960D+01,
     & 0.13294D+01,0.13641D+01,0.14000D+01,0.14372D+01,0.14758D+01,
     & 0.15158D+01,0.15573D+01,0.16004D+01,0.16450D+01,0.16913D+01,
     & 0.17394D+01,0.17893D+01,0.18412D+01,0.18951D+01,0.19626D+01,
     & 0.20189D+01,0.20507D+01,0.20444D+01,0.16421D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.69144D+00,0.68363D+00,
     & 0.67754D+00,0.67303D+00,0.66994D+00,0.66817D+00,0.66761D+00,
     & 0.66817D+00,0.66976D+00,0.67231D+00,0.67577D+00,0.68009D+00,
     & 0.68522D+00,0.69112D+00,0.69778D+00,0.70516D+00,0.71324D+00,
     & 0.72200D+00,0.73143D+00,0.74152D+00,0.75226D+00,0.76364D+00,
     & 0.77566D+00,0.78833D+00,0.80163D+00,0.81557D+00,0.83017D+00,
     & 0.84541D+00,0.86132D+00,0.87790D+00,0.89516D+00,0.91312D+00,
     & 0.93178D+00,0.95118D+00,0.97132D+00,0.99223D+00,0.10139D+01,
     & 0.10364D+01,0.10598D+01,0.10840D+01,0.11091D+01,0.11351D+01,
     & 0.11620D+01,0.11900D+01,0.12189D+01,0.12490D+01,0.12801D+01,
     & 0.13124D+01,0.13458D+01,0.13805D+01,0.14164D+01,0.14536D+01,
     & 0.14923D+01,0.15323D+01,0.15738D+01,0.16169D+01,0.16615D+01,
     & 0.17078D+01,0.17558D+01,0.18058D+01,0.18576D+01,0.19115D+01,
     & 0.19784D+01,0.20347D+01,0.20678D+01,0.20650D+01,0.16524D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.71663D+00,0.70880D+00,0.70273D+00,0.69826D+00,0.69525D+00,
     & 0.69359D+00,0.69317D+00,0.69390D+00,0.69568D+00,0.69845D+00,
     & 0.70215D+00,0.70674D+00,0.71216D+00,0.71838D+00,0.72537D+00,
     & 0.73310D+00,0.74156D+00,0.75073D+00,0.76058D+00,0.77111D+00,
     & 0.78232D+00,0.79419D+00,0.80672D+00,0.81991D+00,0.83377D+00,
     & 0.84829D+00,0.86347D+00,0.87934D+00,0.89589D+00,0.91313D+00,
     & 0.93109D+00,0.94976D+00,0.96917D+00,0.98933D+00,0.10103D+01,
     & 0.10320D+01,0.10546D+01,0.10779D+01,0.11022D+01,0.11273D+01,
     & 0.11534D+01,0.11804D+01,0.12084D+01,0.12374D+01,0.12675D+01,
     & 0.12987D+01,0.13310D+01,0.13645D+01,0.13993D+01,0.14352D+01,
     & 0.14725D+01,0.15112D+01,0.15513D+01,0.15928D+01,0.16359D+01,
     & 0.16806D+01,0.17269D+01,0.17749D+01,0.18249D+01,0.18768D+01,
     & 0.19306D+01,0.19969D+01,0.20533D+01,0.20879D+01,0.20886D+01,
     & 0.16638D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.74508D+00,0.73714D+00,0.73100D+00,
     & 0.72651D+00,0.72353D+00,0.72193D+00,0.72161D+00,0.72247D+00,
     & 0.72441D+00,0.72738D+00,0.73131D+00,0.73615D+00,0.74185D+00,
     & 0.74837D+00,0.75570D+00,0.76379D+00,0.77263D+00,0.78221D+00,
     & 0.79249D+00,0.80348D+00,0.81517D+00,0.82754D+00,0.84061D+00,
     & 0.85435D+00,0.86878D+00,0.88390D+00,0.89972D+00,0.91623D+00,
     & 0.93345D+00,0.95140D+00,0.97008D+00,0.98951D+00,0.10097D+01,
     & 0.10307D+01,0.10524D+01,0.10750D+01,0.10985D+01,0.11228D+01,
     & 0.11480D+01,0.11741D+01,0.12012D+01,0.12293D+01,0.12584D+01,
     & 0.12886D+01,0.13199D+01,0.13523D+01,0.13859D+01,0.14207D+01,
     & 0.14568D+01,0.14941D+01,0.15329D+01,0.15730D+01,0.16147D+01,
     & 0.16578D+01,0.17026D+01,0.17489D+01,0.17970D+01,0.18470D+01,
     & 0.18989D+01,0.19529D+01,0.20186D+01,0.20751D+01,0.21111D+01,
     & 0.21154D+01,0.16763D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.77727D+00,
     & 0.76911D+00,0.76282D+00,0.75824D+00,0.75521D+00,0.75362D+00,
     & 0.75334D+00,0.75429D+00,0.75636D+00,0.75950D+00,0.76363D+00,
     & 0.76870D+00,0.77466D+00,0.78149D+00,0.78915D+00,0.79760D+00,
     & 0.80683D+00,0.81681D+00,0.82754D+00,0.83900D+00,0.85119D+00,
     & 0.86408D+00,0.87769D+00,0.89201D+00,0.90705D+00,0.92280D+00,
     & 0.93927D+00,0.95646D+00,0.97440D+00,0.99309D+00,0.10125D+01,
     & 0.10328D+01,0.10538D+01,0.10756D+01,0.10983D+01,0.11218D+01,
     & 0.11462D+01,0.11715D+01,0.11977D+01,0.12249D+01,0.12530D+01,
     & 0.12823D+01,0.13126D+01,0.13440D+01,0.13765D+01,0.14102D+01,
     & 0.14451D+01,0.14813D+01,0.15188D+01,0.15576D+01,0.15979D+01,
     & 0.16396D+01,0.16829D+01,0.17278D+01,0.17742D+01,0.18223D+01,
     & 0.18725D+01,0.19246D+01,0.19787D+01,0.20439D+01,0.21005D+01,
     & 0.21381D+01,0.21459D+01,0.16901D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.81381D+00,0.80531D+00,0.79876D+00,0.79398D+00,
     & 0.79083D+00,0.78916D+00,0.78888D+00,0.78986D+00,0.79202D+00,
     & 0.79528D+00,0.79958D+00,0.80487D+00,0.81108D+00,0.81820D+00,
     & 0.82617D+00,0.83498D+00,0.84460D+00,0.85500D+00,0.86618D+00,
     & 0.87812D+00,0.89081D+00,0.90425D+00,0.91843D+00,0.93335D+00,
     & 0.94902D+00,0.96543D+00,0.98258D+00,0.10005D+01,0.10192D+01,
     & 0.10387D+01,0.10589D+01,0.10800D+01,0.11019D+01,0.11246D+01,
     & 0.11482D+01,0.11727D+01,0.11981D+01,0.12244D+01,0.12517D+01,
     & 0.12800D+01,0.13094D+01,0.13398D+01,0.13714D+01,0.14040D+01,
     & 0.14379D+01,0.14730D+01,0.15093D+01,0.15469D+01,0.15859D+01,
     & 0.16264D+01,0.16682D+01,0.17116D+01,0.17566D+01,0.18031D+01,
     & 0.18514D+01,0.19020D+01,0.19547D+01,0.20093D+01,0.20741D+01,
     & 0.21309D+01,0.21698D+01,0.21809D+01,0.17051D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.85548D+00,0.84649D+00,
     & 0.83954D+00,0.83445D+00,0.83107D+00,0.82924D+00,0.82886D+00,
     & 0.82982D+00,0.83201D+00,0.83535D+00,0.83978D+00,0.84525D+00,
     & 0.85169D+00,0.85907D+00,0.86735D+00,0.87650D+00,0.88650D+00,
     & 0.89733D+00,0.90896D+00,0.92140D+00,0.93462D+00,0.94861D+00,
     & 0.96339D+00,0.97893D+00,0.99525D+00,0.10123D+01,0.10302D+01,
     & 0.10489D+01,0.10684D+01,0.10887D+01,0.11098D+01,0.11317D+01,
     & 0.11545D+01,0.11782D+01,0.12028D+01,0.12284D+01,0.12548D+01,
     & 0.12823D+01,0.13107D+01,0.13402D+01,0.13708D+01,0.14025D+01,
     & 0.14354D+01,0.14694D+01,0.15047D+01,0.15412D+01,0.15790D+01,
     & 0.16183D+01,0.16589D+01,0.17009D+01,0.17445D+01,0.17897D+01,
     & 0.18363D+01,0.18848D+01,0.19358D+01,0.19671D+01,0.20444D+01,
     & 0.21089D+01,0.21659D+01,0.22062D+01,0.22206D+01,0.17214D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.90330D+00,0.89361D+00,0.88610D+00,0.88055D+00,0.87680D+00,
     & 0.87470D+00,0.87413D+00,0.87497D+00,0.87711D+00,0.88048D+00,
     & 0.88499D+00,0.89059D+00,0.89722D+00,0.90483D+00,0.91340D+00,
     & 0.92288D+00,0.93326D+00,0.94450D+00,0.95660D+00,0.96953D+00,
     & 0.98329D+00,0.99786D+00,0.10132D+01,0.10294D+01,0.10465D+01,
     & 0.10643D+01,0.10829D+01,0.11024D+01,0.11227D+01,0.11438D+01,
     & 0.11659D+01,0.11888D+01,0.12126D+01,0.12373D+01,0.12629D+01,
     & 0.12895D+01,0.13171D+01,0.13457D+01,0.13754D+01,0.14062D+01,
     & 0.14374D+01,0.14712D+01,0.15055D+01,0.15409D+01,0.15777D+01,
     & 0.16157D+01,0.16552D+01,0.16960D+01,0.17383D+01,0.17821D+01,
     & 0.18275D+01,0.18744D+01,0.19231D+01,0.19743D+01,0.21973D+01,
     & 0.20836D+01,0.21477D+01,0.22051D+01,0.22471D+01,0.22653D+01,
     & 0.17391D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.95858D+00,0.94796D+00,0.93966D+00,
     & 0.93345D+00,0.92917D+00,0.92665D+00,0.92575D+00,0.92636D+00,
     & 0.92836D+00,0.93166D+00,0.93617D+00,0.94184D+00,0.94861D+00,
     & 0.95643D+00,0.96525D+00,0.97504D+00,0.98577D+00,0.99743D+00,
     & 0.10100D+01,0.10234D+01,0.10377D+01,0.10529D+01,0.10689D+01,
     & 0.10858D+01,0.11035D+01,0.11221D+01,0.11415D+01,0.11618D+01,
     & 0.11830D+01,0.12051D+01,0.12280D+01,0.12519D+01,0.12768D+01,
     & 0.13026D+01,0.13293D+01,0.13571D+01,0.13859D+01,0.14158D+01,
     & 0.14469D+01,0.14849D+01,0.15123D+01,0.15468D+01,0.15825D+01,
     & 0.16195D+01,0.16578D+01,0.16975D+01,0.17386D+01,0.17812D+01,
     & 0.18253D+01,0.18710D+01,0.19182D+01,0.19672D+01,0.20185D+01,
     & 0.23807D+01,0.21272D+01,0.21909D+01,0.22486D+01,0.22927D+01,
     & 0.23154D+01,0.17582D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.10112D+01,0.10018D+01,0.99472D+00,0.98968D+00,0.98653D+00,
     & 0.98514D+00,0.98536D+00,0.98708D+00,0.99020D+00,0.99462D+00,
     & 0.10003D+01,0.10071D+01,0.10151D+01,0.10241D+01,0.10342D+01,
     & 0.10452D+01,0.10573D+01,0.10703D+01,0.10842D+01,0.10990D+01,
     & 0.11148D+01,0.11315D+01,0.11490D+01,0.11675D+01,0.11869D+01,
     & 0.12071D+01,0.12283D+01,0.12504D+01,0.12735D+01,0.12975D+01,
     & 0.13224D+01,0.13484D+01,0.13753D+01,0.14033D+01,0.14323D+01,
     & 0.14624D+01,0.14937D+01,0.16382D+01,0.15597D+01,0.15944D+01,
     & 0.16305D+01,0.16678D+01,0.17064D+01,0.17465D+01,0.17880D+01,
     & 0.18309D+01,0.18753D+01,0.19213D+01,0.19690D+01,0.20184D+01,
     & 0.20696D+01,0.20957D+01,0.21779D+01,0.22410D+01,0.22993D+01,
     & 0.23456D+01,0.23729D+01,0.17788D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.10857D+01,0.10749D+01,0.10665D+01,
     & 0.10604D+01,0.10563D+01,0.10542D+01,0.10538D+01,0.10551D+01,
     & 0.10579D+01,0.10620D+01,0.10676D+01,0.10744D+01,0.10824D+01,
     & 0.10915D+01,0.11018D+01,0.11132D+01,0.11255D+01,0.11389D+01,
     & 0.11534D+01,0.11688D+01,0.11851D+01,0.12025D+01,0.12207D+01,
     & 0.12400D+01,0.12602D+01,0.12813D+01,0.13034D+01,0.13265D+01,
     & 0.13506D+01,0.13757D+01,0.14017D+01,0.14289D+01,0.14570D+01,
     & 0.14863D+01,0.15167D+01,0.15482D+01,0.15784D+01,0.16147D+01,
     & 0.16498D+01,0.16862D+01,0.17240D+01,0.17631D+01,0.18036D+01,
     & 0.18455D+01,0.18888D+01,0.19337D+01,0.19801D+01,0.20282D+01,
     & 0.20780D+01,0.21296D+01,0.21830D+01,0.22384D+01,0.23013D+01,
     & 0.23601D+01,0.24084D+01,0.24400D+01,0.18010D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.11619D+01,0.11517D+01,0.11441D+01,0.11388D+01,0.11356D+01,
     & 0.11343D+01,0.11349D+01,0.11371D+01,0.11408D+01,0.11460D+01,
     & 0.11527D+01,0.11606D+01,0.11698D+01,0.11801D+01,0.11917D+01,
     & 0.12044D+01,0.12182D+01,0.12330D+01,0.12489D+01,0.12659D+01,
     & 0.12839D+01,0.13029D+01,0.13229D+01,0.13439D+01,0.13660D+01,
     & 0.13891D+01,0.14132D+01,0.14384D+01,0.14646D+01,0.14919D+01,
     & 0.15202D+01,0.15497D+01,0.15804D+01,0.16121D+01,0.16451D+01,
     & 0.16793D+01,0.17148D+01,0.17516D+01,0.17898D+01,0.18219D+01,
     & 0.18705D+01,0.19129D+01,0.19568D+01,0.20022D+01,0.20491D+01,
     & 0.20977D+01,0.21481D+01,0.22002D+01,0.22542D+01,0.23102D+01,
     & 0.23730D+01,0.24324D+01,0.24828D+01,0.25185D+01,0.18248D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.12673D+01,0.12548D+01,0.12450D+01,
     & 0.12379D+01,0.12331D+01,0.12305D+01,0.12299D+01,0.12312D+01,
     & 0.12343D+01,0.12389D+01,0.12451D+01,0.12528D+01,0.12618D+01,
     & 0.12722D+01,0.12838D+01,0.12967D+01,0.13107D+01,0.13260D+01,
     & 0.13423D+01,0.13599D+01,0.13785D+01,0.13982D+01,0.14190D+01,
     & 0.14409D+01,0.14639D+01,0.14880D+01,0.15132D+01,0.15395D+01,
     & 0.15670D+01,0.15955D+01,0.16253D+01,0.16562D+01,0.16883D+01,
     & 0.17216D+01,0.17562D+01,0.17920D+01,0.18293D+01,0.18679D+01,
     & 0.20212D+01,0.19496D+01,0.19926D+01,0.20370D+01,0.20830D+01,
     & 0.21306D+01,0.21798D+01,0.22308D+01,0.22837D+01,0.23385D+01,
     & 0.23952D+01,0.24581D+01,0.25183D+01,0.25709D+01,0.26110D+01,
     & 0.18501D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.13820D+01,0.13692D+01,0.13593D+01,0.13523D+01,0.13478D+01,
     & 0.13455D+01,0.13454D+01,0.13472D+01,0.13510D+01,0.13564D+01,
     & 0.13635D+01,0.13721D+01,0.13822D+01,0.13937D+01,0.14066D+01,
     & 0.14208D+01,0.14363D+01,0.14531D+01,0.14711D+01,0.14903D+01,
     & 0.15107D+01,0.15323D+01,0.15550D+01,0.15790D+01,0.16041D+01,
     & 0.16305D+01,0.16580D+01,0.16867D+01,0.17166D+01,0.17478D+01,
     & 0.17802D+01,0.18139D+01,0.18489D+01,0.18852D+01,0.19229D+01,
     & 0.19620D+01,0.20384D+01,0.20447D+01,0.20882D+01,0.21333D+01,
     & 0.21800D+01,0.22283D+01,0.22783D+01,0.23301D+01,0.23838D+01,
     & 0.24394D+01,0.24970D+01,0.25601D+01,0.26212D+01,0.26763D+01,
     & 0.27210D+01,0.18768D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.15263D+01,0.15125D+01,
     & 0.15019D+01,0.14944D+01,0.14897D+01,0.14873D+01,0.14873D+01,
     & 0.14894D+01,0.14935D+01,0.14995D+01,0.15073D+01,0.15168D+01,
     & 0.15279D+01,0.15405D+01,0.15546D+01,0.15702D+01,0.15871D+01,
     & 0.16055D+01,0.16251D+01,0.16461D+01,0.16684D+01,0.16920D+01,
     & 0.17169D+01,0.17431D+01,0.17706D+01,0.17993D+01,0.18294D+01,
     & 0.18608D+01,0.18935D+01,0.19275D+01,0.19629D+01,0.19997D+01,
     & 0.20379D+01,0.20776D+01,0.21137D+01,0.21614D+01,0.22056D+01,
     & 0.22515D+01,0.22989D+01,0.23481D+01,0.23990D+01,0.24517D+01,
     & 0.25064D+01,0.25630D+01,0.26216D+01,0.26852D+01,0.27475D+01,
     & 0.28051D+01,0.28547D+01,0.19047D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.17122D+01,0.16965D+01,0.16842D+01,0.16752D+01,
     & 0.16693D+01,0.16664D+01,0.16658D+01,0.16677D+01,0.16719D+01,
     & 0.16781D+01,0.16863D+01,0.16964D+01,0.17083D+01,0.17219D+01,
     & 0.17371D+01,0.17539D+01,0.17723D+01,0.17922D+01,0.18136D+01,
     & 0.18365D+01,0.18608D+01,0.18866D+01,0.19138D+01,0.19424D+01,
     & 0.19724D+01,0.20038D+01,0.20367D+01,0.20710D+01,0.21067D+01,
     & 0.21440D+01,0.21827D+01,0.22230D+01,0.22649D+01,0.23083D+01,
     & 0.23533D+01,0.24000D+01,0.24484D+01,0.24986D+01,0.25505D+01,
     & 0.26043D+01,0.26601D+01,0.27085D+01,0.27779D+01,0.28421D+01,
     & 0.29057D+01,0.29661D+01,0.30207D+01,0.19330D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.19411D+01,0.19255D+01,0.19136D+01,0.19051D+01,0.19001D+01,
     & 0.18983D+01,0.18992D+01,0.19024D+01,0.19082D+01,0.19164D+01,
     & 0.19266D+01,0.19389D+01,0.19531D+01,0.19692D+01,0.19871D+01,
     & 0.20068D+01,0.20282D+01,0.20513D+01,0.20761D+01,0.21025D+01,
     & 0.21305D+01,0.21601D+01,0.21913D+01,0.22241D+01,0.22584D+01,
     & 0.22944D+01,0.23319D+01,0.23712D+01,0.24120D+01,0.24546D+01,
     & 0.24988D+01,0.25447D+01,0.25924D+01,0.26419D+01,0.26932D+01,
     & 0.27464D+01,0.28015D+01,0.28587D+01,0.29921D+01,0.29794D+01,
     & 0.30446D+01,0.31099D+01,0.31651D+01,0.32331D+01,0.19603D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.22615D+01,
     & 0.22443D+01,0.22310D+01,0.22214D+01,0.22156D+01,0.22135D+01,
     & 0.22148D+01,0.22194D+01,0.22260D+01,0.22352D+01,0.22470D+01,
     & 0.22612D+01,0.22776D+01,0.22962D+01,0.23168D+01,0.23394D+01,
     & 0.23640D+01,0.23905D+01,0.24189D+01,0.24492D+01,0.24813D+01,
     & 0.25151D+01,0.25508D+01,0.25883D+01,0.26278D+01,0.26690D+01,
     & 0.27121D+01,0.27571D+01,0.28039D+01,0.28526D+01,0.29032D+01,
     & 0.29558D+01,0.30104D+01,0.30670D+01,0.31258D+01,0.33129D+01,
     & 0.32501D+01,0.33167D+01,0.33841D+01,0.35093D+01,0.35161D+01,
     & 0.19834D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.27374D+01,0.27155D+01,
     & 0.26978D+01,0.26841D+01,0.26747D+01,0.26693D+01,0.26681D+01,
     & 0.26707D+01,0.26771D+01,0.26870D+01,0.27002D+01,0.27153D+01,
     & 0.27330D+01,0.27534D+01,0.27763D+01,0.28016D+01,0.28293D+01,
     & 0.28592D+01,0.28914D+01,0.29256D+01,0.29620D+01,0.30009D+01,
     & 0.30418D+01,0.30849D+01,0.31302D+01,0.31776D+01,0.32271D+01,
     & 0.32788D+01,0.33327D+01,0.33887D+01,0.34470D+01,0.35076D+01,
     & 0.35592D+01,0.36359D+01,0.37043D+01,0.37742D+01,0.40196D+01,
     & 0.39158D+01,0.19951D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.34565D+01,
     & 0.34336D+01,0.34153D+01,0.34014D+01,0.33922D+01,0.33876D+01,
     & 0.33875D+01,0.33918D+01,0.34004D+01,0.34132D+01,0.34299D+01,
     & 0.34504D+01,0.34746D+01,0.35021D+01,0.35328D+01,0.35665D+01,
     & 0.36021D+01,0.36405D+01,0.36817D+01,0.37257D+01,0.37724D+01,
     & 0.38217D+01,0.38736D+01,0.39282D+01,0.39853D+01,0.40451D+01,
     & 0.41074D+01,0.41724D+01,0.42400D+01,0.43106D+01,0.43836D+01,
     & 0.44469D+01,0.45353D+01,0.19775D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.48019D+01,0.47742D+01,0.47517D+01,0.47344D+01,
     & 0.47225D+01,0.47158D+01,0.47145D+01,0.47184D+01,0.47276D+01,
     & 0.47420D+01,0.47614D+01,0.47858D+01,0.48150D+01,0.48490D+01,
     & 0.48875D+01,0.49305D+01,0.49779D+01,0.50294D+01,0.50849D+01,
     & 0.51444D+01,0.52076D+01,0.52745D+01,0.53450D+01,0.54190D+01,
     & 0.54965D+01,0.55772D+01,0.56610D+01,0.18794D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.81673D+01,0.81335D+01,0.81063D+01,
     & 0.80858D+01,0.80720D+01,0.80651D+01,0.80650D+01,0.80719D+01,
     & 0.80857D+01,0.81064D+01,0.81341D+01,0.81687D+01,0.82103D+01,
     & 0.82588D+01,0.83143D+01,0.83769D+01,0.84466D+01,0.13663D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00/
      DATA DX5/0.996D-02/,DY5/0.124292D0/,X05/0.2D-02/,Y05/3.58352D0/      
      DATA DFT5/0.0D+00,0.0D+00,0.0000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.16724D+03,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.22469D+02,0.23712D+02,
     & 0.24801D+02,0.25991D+02,0.27275D+02,0.28637D+02,0.30073D+02,
     & 0.31580D+02,0.33163D+02,0.34807D+02,0.36528D+02,0.38712D+02,
     & 0.40581D+02,0.42528D+02,0.44537D+02,0.46626D+02,0.48789D+02,
     & 0.51028D+02,0.53344D+02,0.55738D+02,0.58211D+02,0.60762D+02,
     & 0.63418D+02,0.66114D+02,0.70202D+02,0.72923D+02,0.72842D+02,
     & 0.68501D+02,0.54074D+02,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.84672D+01,
     & 0.88282D+01,0.92229D+01,0.96429D+01,0.10088D+02,0.10650D+02,
     & 0.11142D+02,0.11660D+02,0.12203D+02,0.12772D+02,0.13098D+02,
     & 0.13828D+02,0.14456D+02,0.15108D+02,0.15785D+02,0.16489D+02,
     & 0.17223D+02,0.17987D+02,0.18785D+02,0.19609D+02,0.20469D+02,
     & 0.21575D+02,0.22507D+02,0.23476D+02,0.24476D+02,0.25517D+02,
     & 0.26596D+02,0.27714D+02,0.28873D+02,0.30074D+02,0.31317D+02,
     & 0.32604D+02,0.33948D+02,0.35319D+02,0.37758D+02,0.39110D+02,
     & 0.38222D+02,0.33922D+02,0.23028D+02,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.43350D+01,0.45180D+01,
     & 0.46817D+01,0.48919D+01,0.51926D+01,0.54616D+01,0.57007D+01,
     & 0.58756D+01,0.61181D+01,0.63770D+01,0.66490D+01,0.69349D+01,
     & 0.72984D+01,0.76123D+01,0.79418D+01,0.82875D+01,0.86499D+01,
     & 0.90296D+01,0.95122D+01,0.99261D+01,0.10359D+02,0.10812D+02,
     & 0.11285D+02,0.11778D+02,0.12294D+02,0.12833D+02,0.13392D+02,
     & 0.13976D+02,0.14729D+02,0.15364D+02,0.16027D+02,0.16712D+02,
     & 0.17426D+02,0.18168D+02,0.18939D+02,0.19740D+02,0.20571D+02,
     & 0.21434D+02,0.22328D+02,0.23265D+02,0.24221D+02,0.25981D+02,
     & 0.26924D+02,0.26178D+02,0.22846D+02,0.14698D+02,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.29118D+01,0.30112D+01,0.31432D+01,0.32544D+01,0.33160D+01,
     & 0.34638D+01,0.35885D+01,0.37252D+01,0.39082D+01,0.40655D+01,
     & 0.42238D+01,0.44274D+01,0.45990D+01,0.47800D+01,0.49699D+01,
     & 0.51690D+01,0.54253D+01,0.56427D+01,0.58703D+01,0.61086D+01,
     & 0.63577D+01,0.66181D+01,0.69545D+01,0.72369D+01,0.75318D+01,
     & 0.78394D+01,0.81602D+01,0.84949D+01,0.88435D+01,0.92081D+01,
     & 0.95850D+01,0.99787D+01,0.10497D+02,0.10925D+02,0.11370D+02,
     & 0.11830D+02,0.12309D+02,0.12807D+02,0.13323D+02,0.13859D+02,
     & 0.14415D+02,0.14992D+02,0.15590D+02,0.16217D+02,0.16854D+02,
     & 0.18024D+02,0.18654D+02,0.18166D+02,0.15961D+02,0.10453D+02,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.20132D+01,0.20791D+01,0.21725D+01,
     & 0.22519D+01,0.23140D+01,0.23906D+01,0.24933D+01,0.25789D+01,
     & 0.26693D+01,0.27893D+01,0.28890D+01,0.29900D+01,0.31198D+01,
     & 0.32273D+01,0.33441D+01,0.34991D+01,0.36275D+01,0.37615D+01,
     & 0.39015D+01,0.40476D+01,0.42382D+01,0.43964D+01,0.45613D+01,
     & 0.47331D+01,0.49121D+01,0.50985D+01,0.53439D+01,0.55441D+01,
     & 0.57524D+01,0.59688D+01,0.61936D+01,0.64271D+01,0.66696D+01,
     & 0.69224D+01,0.71824D+01,0.74533D+01,0.78213D+01,0.81133D+01,
     & 0.84170D+01,0.87288D+01,0.90528D+01,0.93881D+01,0.97351D+01,
     & 0.10094D+02,0.10465D+02,0.10849D+02,0.11246D+02,0.11661D+02,
     & 0.12081D+02,0.12838D+02,0.13252D+02,0.12960D+02,0.11584D+02,
     & 0.79715D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.15157D+01,0.15590D+01,0.16186D+01,0.16632D+01,0.17130D+01,
     & 0.17817D+01,0.18382D+01,0.19137D+01,0.19753D+01,0.20584D+01,
     & 0.21267D+01,0.21985D+01,0.22942D+01,0.23732D+01,0.24548D+01,
     & 0.25615D+01,0.26491D+01,0.27424D+01,0.28666D+01,0.29686D+01,
     & 0.30749D+01,0.31858D+01,0.33014D+01,0.34538D+01,0.35787D+01,
     & 0.37088D+01,0.38442D+01,0.39852D+01,0.41318D+01,0.43270D+01,
     & 0.44843D+01,0.46477D+01,0.48173D+01,0.49933D+01,0.51761D+01,
     & 0.53656D+01,0.55632D+01,0.57660D+01,0.59773D+01,0.62687D+01,
     & 0.64962D+01,0.67326D+01,0.69751D+01,0.72269D+01,0.74874D+01,
     & 0.77568D+01,0.80352D+01,0.83231D+01,0.86206D+01,0.89278D+01,
     & 0.92496D+01,0.95741D+01,0.10158D+02,0.10479D+02,0.10253D+02,
     & 0.91919D+01,0.63667D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.11969D+01,
     & 0.12408D+01,0.12771D+01,0.13125D+01,0.13617D+01,0.14137D+01,
     & 0.14550D+01,0.15112D+01,0.15565D+01,0.16190D+01,0.16698D+01,
     & 0.17387D+01,0.17945D+01,0.18532D+01,0.19316D+01,0.19958D+01,
     & 0.20631D+01,0.21528D+01,0.22263D+01,0.23031D+01,0.24056D+01,
     & 0.24888D+01,0.25757D+01,0.26662D+01,0.27607D+01,0.28863D+01,
     & 0.29882D+01,0.30944D+01,0.32049D+01,0.33198D+01,0.34394D+01,
     & 0.36002D+01,0.37283D+01,0.38614D+01,0.39996D+01,0.41431D+01,
     & 0.42920D+01,0.44464D+01,0.46074D+01,0.47727D+01,0.49448D+01,
     & 0.51852D+01,0.53706D+01,0.55634D+01,0.57610D+01,0.59664D+01,
     & 0.61788D+01,0.63984D+01,0.66256D+01,0.68604D+01,0.71032D+01,
     & 0.73539D+01,0.76168D+01,0.78815D+01,0.83581D+01,0.86197D+01,
     & 0.84375D+01,0.75740D+01,0.52546D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.94139D+00,0.97286D+00,0.10063D+01,
     & 0.10282D+01,0.10641D+01,0.11022D+01,0.11323D+01,0.11742D+01,
     & 0.12186D+01,0.12538D+01,0.13022D+01,0.13410D+01,0.13943D+01,
     & 0.14372D+01,0.14958D+01,0.15430D+01,0.15924D+01,0.16590D+01,
     & 0.17129D+01,0.17695D+01,0.18452D+01,0.19066D+01,0.19709D+01,
     & 0.20574D+01,0.21270D+01,0.21995D+01,0.22752D+01,0.23540D+01,
     & 0.24598D+01,0.25448D+01,0.26332D+01,0.27253D+01,0.28210D+01,
     & 0.29205D+01,0.30558D+01,0.31624D+01,0.32731D+01,0.33880D+01,
     & 0.35073D+01,0.36311D+01,0.37595D+01,0.38933D+01,0.40307D+01,
     & 0.41738D+01,0.43761D+01,0.45303D+01,0.46907D+01,0.48550D+01,
     & 0.50257D+01,0.52024D+01,0.53851D+01,0.55740D+01,0.57694D+01,
     & 0.59714D+01,0.61800D+01,0.63990D+01,0.66192D+01,0.70142D+01,
     & 0.72323D+01,0.70847D+01,0.63753D+01,0.44443D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.77271D+00,0.79721D+00,0.82247D+00,0.84949D+00,
     & 0.87796D+00,0.89993D+00,0.93124D+00,0.96435D+00,0.99042D+00,
     & 0.10268D+01,0.10652D+01,0.10956D+01,0.11377D+01,0.11712D+01,
     & 0.12174D+01,0.12542D+01,0.13048D+01,0.13452D+01,0.13876D+01,
     & 0.14450D+01,0.14912D+01,0.15395D+01,0.16047D+01,0.16571D+01,
     & 0.17119D+01,0.17864D+01,0.18456D+01,0.19074D+01,0.19719D+01,
     & 0.20389D+01,0.21298D+01,0.22020D+01,0.22771D+01,0.23553D+01,
     & 0.24366D+01,0.25211D+01,0.26371D+01,0.27275D+01,0.28214D+01,
     & 0.29188D+01,0.30200D+01,0.31249D+01,0.32337D+01,0.33471D+01,
     & 0.34634D+01,0.35847D+01,0.37581D+01,0.38886D+01,0.40245D+01,
     & 0.41636D+01,0.43081D+01,0.44577D+01,0.46123D+01,0.47723D+01,
     & 0.49376D+01,0.51086D+01,0.52851D+01,0.54707D+01,0.56569D+01,
     & 0.59895D+01,0.61743D+01,0.60534D+01,0.54626D+01,0.38308D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.66311D+00,0.68341D+00,0.70486D+00,0.72754D+00,
     & 0.75147D+00,0.77673D+00,0.79614D+00,0.82384D+00,0.85307D+00,
     & 0.87591D+00,0.90792D+00,0.94171D+00,0.96826D+00,0.10052D+01,
     & 0.10345D+01,0.10750D+01,0.11070D+01,0.11514D+01,0.11865D+01,
     & 0.12233D+01,0.12735D+01,0.13135D+01,0.13554D+01,0.14123D+01,
     & 0.14577D+01,0.15051D+01,0.15701D+01,0.16213D+01,0.16748D+01,
     & 0.17304D+01,0.17884D+01,0.18675D+01,0.19298D+01,0.19946D+01,
     & 0.20621D+01,0.21322D+01,0.22051D+01,0.23061D+01,0.23840D+01,
     & 0.24650D+01,0.25489D+01,0.26361D+01,0.27265D+01,0.28202D+01,
     & 0.29180D+01,0.30182D+01,0.31227D+01,0.32737D+01,0.33863D+01,
     & 0.35033D+01,0.36232D+01,0.37478D+01,0.38767D+01,0.40100D+01,
     & 0.41479D+01,0.42905D+01,0.44378D+01,0.45901D+01,0.47502D+01,
     & 0.49108D+01,0.51961D+01,0.53557D+01,0.52553D+01,0.47546D+01,
     & 0.33522D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.55785D+00,0.57428D+00,0.59162D+00,0.60990D+00,0.62917D+00,
     & 0.64949D+00,0.67092D+00,0.69349D+00,0.71074D+00,0.73545D+00,
     & 0.76152D+00,0.78174D+00,0.81022D+00,0.84027D+00,0.86372D+00,
     & 0.89654D+00,0.92233D+00,0.95822D+00,0.98647D+00,0.10258D+01,
     & 0.10566D+01,0.10889D+01,0.11333D+01,0.11685D+01,0.12052D+01,
     & 0.12555D+01,0.12953D+01,0.13368D+01,0.13942D+01,0.14390D+01,
     & 0.14858D+01,0.15344D+01,0.15851D+01,0.16548D+01,0.17093D+01,
     & 0.17659D+01,0.18248D+01,0.18861D+01,0.19497D+01,0.20386D+01,
     & 0.21066D+01,0.21772D+01,0.22504D+01,0.23264D+01,0.24052D+01,
     & 0.24870D+01,0.25722D+01,0.26596D+01,0.27506D+01,0.28837D+01,
     & 0.29818D+01,0.30839D+01,0.31883D+01,0.32969D+01,0.34093D+01,
     & 0.35255D+01,0.36456D+01,0.37699D+01,0.38983D+01,0.40311D+01,
     & 0.41706D+01,0.43105D+01,0.45577D+01,0.46971D+01,0.46133D+01,
     & 0.41859D+01,0.29696D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.47360D+00,
     & 0.48704D+00,0.50120D+00,0.51612D+00,0.53184D+00,0.54840D+00,
     & 0.56585D+00,0.58421D+00,0.60354D+00,0.62390D+00,0.63936D+00,
     & 0.66160D+00,0.68504D+00,0.70311D+00,0.72868D+00,0.75563D+00,
     & 0.77652D+00,0.80595D+00,0.82889D+00,0.86101D+00,0.88612D+00,
     & 0.92125D+00,0.94863D+00,0.97730D+00,0.10170D+01,0.10481D+01,
     & 0.10806D+01,0.11255D+01,0.11607D+01,0.11975D+01,0.12486D+01,
     & 0.12882D+01,0.13296D+01,0.13726D+01,0.14173D+01,0.14794D+01,
     & 0.15275D+01,0.15776D+01,0.16296D+01,0.16836D+01,0.17398D+01,
     & 0.18190D+01,0.18790D+01,0.19412D+01,0.20058D+01,0.20729D+01,
     & 0.21424D+01,0.22145D+01,0.22897D+01,0.23668D+01,0.24471D+01,
     & 0.25657D+01,0.26522D+01,0.27423D+01,0.28345D+01,0.29303D+01,
     & 0.30294D+01,0.31319D+01,0.32380D+01,0.33476D+01,0.34610D+01,
     & 0.35782D+01,0.37015D+01,0.38249D+01,0.40418D+01,0.41651D+01,
     & 0.40945D+01,0.37251D+01,0.26576D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.41617D+00,
     & 0.42786D+00,0.44017D+00,0.45315D+00,0.46681D+00,0.48117D+00,
     & 0.49628D+00,0.51217D+00,0.52889D+00,0.54646D+00,0.56495D+00,
     & 0.57891D+00,0.59908D+00,0.62031D+00,0.63658D+00,0.65970D+00,
     & 0.68408D+00,0.70285D+00,0.72942D+00,0.75000D+00,0.77898D+00,
     & 0.80147D+00,0.83314D+00,0.85765D+00,0.88330D+00,0.91899D+00,
     & 0.94682D+00,0.97589D+00,0.10163D+01,0.10477D+01,0.10805D+01,
     & 0.11264D+01,0.11618D+01,0.11986D+01,0.12370D+01,0.12769D+01,
     & 0.13326D+01,0.13755D+01,0.14200D+01,0.14664D+01,0.15145D+01,
     & 0.15645D+01,0.16356D+01,0.16890D+01,0.17444D+01,0.18019D+01,
     & 0.18616D+01,0.19234D+01,0.19876D+01,0.20545D+01,0.21231D+01,
     & 0.21946D+01,0.23012D+01,0.23782D+01,0.24584D+01,0.25404D+01,
     & 0.26257D+01,0.27140D+01,0.28053D+01,0.28997D+01,0.29974D+01,
     & 0.30984D+01,0.32027D+01,0.33126D+01,0.34225D+01,0.36145D+01,
     & 0.37245D+01,0.36648D+01,0.33432D+01,0.23988D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.36751D+00,
     & 0.37777D+00,0.38862D+00,0.39998D+00,0.41194D+00,0.42452D+00,
     & 0.43771D+00,0.45158D+00,0.46615D+00,0.48146D+00,0.49754D+00,
     & 0.51444D+00,0.52712D+00,0.54553D+00,0.56489D+00,0.57964D+00,
     & 0.60070D+00,0.62290D+00,0.63986D+00,0.66402D+00,0.68261D+00,
     & 0.70894D+00,0.72923D+00,0.75798D+00,0.78007D+00,0.80317D+00,
     & 0.83554D+00,0.86060D+00,0.88676D+00,0.92333D+00,0.95161D+00,
     & 0.98109D+00,0.10227D+01,0.10545D+01,0.10876D+01,0.11220D+01,
     & 0.11579D+01,0.12083D+01,0.12468D+01,0.12868D+01,0.13284D+01,
     & 0.13716D+01,0.14164D+01,0.14808D+01,0.15287D+01,0.15784D+01,
     & 0.16300D+01,0.16835D+01,0.17390D+01,0.17966D+01,0.18566D+01,
     & 0.19181D+01,0.19823D+01,0.20788D+01,0.21480D+01,0.22200D+01,
     & 0.22936D+01,0.23701D+01,0.24494D+01,0.25313D+01,0.26161D+01,
     & 0.27037D+01,0.27944D+01,0.28881D+01,0.29869D+01,0.30856D+01,
     & 0.32568D+01,0.33558D+01,0.33050D+01,0.30231D+01,0.21811D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.31380D+00,0.32248D+00,
     & 0.33494D+00,0.34453D+00,0.35459D+00,0.36514D+00,0.37623D+00,
     & 0.38786D+00,0.40005D+00,0.41285D+00,0.42629D+00,0.44039D+00,
     & 0.45518D+00,0.47072D+00,0.48232D+00,0.49922D+00,0.51697D+00,
     & 0.53042D+00,0.54971D+00,0.57004D+00,0.58547D+00,0.60758D+00,
     & 0.62447D+00,0.64853D+00,0.66695D+00,0.69321D+00,0.71324D+00,
     & 0.73419D+00,0.76373D+00,0.78643D+00,0.81012D+00,0.84346D+00,
     & 0.86906D+00,0.89574D+00,0.93366D+00,0.96241D+00,0.99234D+00,
     & 0.10235D+01,0.10559D+01,0.11019D+01,0.11366D+01,0.11728D+01,
     & 0.12103D+01,0.12494D+01,0.12899D+01,0.13485D+01,0.13918D+01,
     & 0.14367D+01,0.14833D+01,0.15316D+01,0.15818D+01,0.16338D+01,
     & 0.16881D+01,0.17436D+01,0.18016D+01,0.18896D+01,0.19521D+01,
     & 0.20172D+01,0.20837D+01,0.21529D+01,0.22245D+01,0.22986D+01,
     & 0.23753D+01,0.24546D+01,0.25366D+01,0.26214D+01,0.27108D+01,
     & 0.28000D+01,0.29539D+01,0.30436D+01,0.30003D+01,0.27515D+01,
     & 0.19956D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.27879D+00,0.28658D+00,
     & 0.29474D+00,0.30643D+00,0.31541D+00,0.32481D+00,0.33466D+00,
     & 0.34497D+00,0.35580D+00,0.36713D+00,0.37900D+00,0.39145D+00,
     & 0.40450D+00,0.41818D+00,0.43254D+00,0.44319D+00,0.45879D+00,
     & 0.47516D+00,0.48748D+00,0.50525D+00,0.52396D+00,0.53808D+00,
     & 0.55841D+00,0.57384D+00,0.59596D+00,0.61277D+00,0.63688D+00,
     & 0.65515D+00,0.67425D+00,0.70135D+00,0.72203D+00,0.74362D+00,
     & 0.77419D+00,0.79749D+00,0.82178D+00,0.85654D+00,0.88269D+00,
     & 0.90992D+00,0.93825D+00,0.96771D+00,0.10098D+01,0.10414D+01,
     & 0.10743D+01,0.11084D+01,0.11439D+01,0.11808D+01,0.12344D+01,
     & 0.12738D+01,0.13146D+01,0.13569D+01,0.14008D+01,0.14464D+01,
     & 0.14937D+01,0.15430D+01,0.15935D+01,0.16461D+01,0.17269D+01,
     & 0.17837D+01,0.18429D+01,0.19033D+01,0.19663D+01,0.20314D+01,
     & 0.20988D+01,0.21685D+01,0.22406D+01,0.23152D+01,0.23924D+01,
     & 0.24738D+01,0.25550D+01,0.26940D+01,0.27759D+01,0.27389D+01,
     & 0.25181D+01,0.18358D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.24526D+00,0.25517D+00,
     & 0.26254D+00,0.27025D+00,0.28126D+00,0.28970D+00,0.29852D+00,
     & 0.30775D+00,0.31741D+00,0.32751D+00,0.33808D+00,0.34915D+00,
     & 0.36074D+00,0.37288D+00,0.38559D+00,0.39893D+00,0.40876D+00,
     & 0.42322D+00,0.43839D+00,0.44973D+00,0.46617D+00,0.48348D+00,
     & 0.49645D+00,0.51525D+00,0.52941D+00,0.54983D+00,0.56525D+00,
     & 0.58751D+00,0.60425D+00,0.62175D+00,0.64674D+00,0.66568D+00,
     & 0.68543D+00,0.71361D+00,0.73493D+00,0.75715D+00,0.78917D+00,
     & 0.81308D+00,0.83797D+00,0.86387D+00,0.89081D+00,0.92959D+00,
     & 0.95846D+00,0.98848D+00,0.10197D+01,0.10521D+01,0.10858D+01,
     & 0.11352D+01,0.11712D+01,0.12084D+01,0.12471D+01,0.12872D+01,
     & 0.13289D+01,0.13721D+01,0.14172D+01,0.14633D+01,0.15114D+01,
     & 0.15858D+01,0.16378D+01,0.16919D+01,0.17472D+01,0.18047D+01,
     & 0.18643D+01,0.19259D+01,0.19897D+01,0.20557D+01,0.21240D+01,
     & 0.21946D+01,0.22691D+01,0.23434D+01,0.24698D+01,0.25448D+01,
     & 0.25132D+01,0.23163D+01,0.16969D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.21827D+00,0.22474D+00,
     & 0.23416D+00,0.24115D+00,0.24845D+00,0.25886D+00,0.26682D+00,
     & 0.27513D+00,0.28381D+00,0.29288D+00,0.30236D+00,0.31227D+00,
     & 0.32262D+00,0.33346D+00,0.34480D+00,0.35666D+00,0.36910D+00,
     & 0.37822D+00,0.39168D+00,0.40580D+00,0.41629D+00,0.43157D+00,
     & 0.44765D+00,0.45963D+00,0.47707D+00,0.49013D+00,0.50907D+00,
     & 0.52328D+00,0.54391D+00,0.55933D+00,0.57544D+00,0.59858D+00,
     & 0.61600D+00,0.63417D+00,0.66025D+00,0.67984D+00,0.70026D+00,
     & 0.72989D+00,0.75186D+00,0.77473D+00,0.79851D+00,0.82325D+00,
     & 0.85912D+00,0.88562D+00,0.91319D+00,0.94184D+00,0.97160D+00,
     & 0.10025D+01,0.10483D+01,0.10812D+01,0.11155D+01,0.11509D+01,
     & 0.11878D+01,0.12260D+01,0.12656D+01,0.13070D+01,0.13494D+01,
     & 0.13936D+01,0.14625D+01,0.15102D+01,0.15600D+01,0.16107D+01,
     & 0.16636D+01,0.17184D+01,0.17750D+01,0.18336D+01,0.18943D+01,
     & 0.19571D+01,0.20220D+01,0.20906D+01,0.21588D+01,0.22743D+01,
     & 0.23435D+01,0.23163D+01,0.21401D+01,0.15751D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.19168D+00,0.20018D+00,
     & 0.20636D+00,0.21533D+00,0.22198D+00,0.22892D+00,0.23878D+00,
     & 0.24632D+00,0.25417D+00,0.26236D+00,0.27091D+00,0.27983D+00,
     & 0.28914D+00,0.29887D+00,0.30905D+00,0.31967D+00,0.33079D+00,
     & 0.34243D+00,0.35091D+00,0.36350D+00,0.37670D+00,0.38644D+00,
     & 0.40070D+00,0.41570D+00,0.42681D+00,0.44306D+00,0.45516D+00,
     & 0.47280D+00,0.48595D+00,0.50515D+00,0.51940D+00,0.53429D+00,
     & 0.55582D+00,0.57191D+00,0.58869D+00,0.61293D+00,0.63102D+00,
     & 0.64986D+00,0.67740D+00,0.69766D+00,0.71876D+00,0.74070D+00,
     & 0.76351D+00,0.79683D+00,0.82127D+00,0.84668D+00,0.87310D+00,
     & 0.90055D+00,0.92905D+00,0.97156D+00,0.10020D+01,0.10335D+01,
     & 0.10662D+01,0.11002D+01,0.11354D+01,0.11720D+01,0.12102D+01,
     & 0.12492D+01,0.12900D+01,0.13541D+01,0.13981D+01,0.14440D+01,
     & 0.14908D+01,0.15397D+01,0.15902D+01,0.16425D+01,0.16966D+01,
     & 0.17526D+01,0.18105D+01,0.18705D+01,0.19339D+01,0.19970D+01,
     & 0.21029D+01,0.21669D+01,0.21436D+01,0.19851D+01,0.14674D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.17025D+00,0.17579D+00,
     & 0.18391D+00,0.18982D+00,0.19838D+00,0.20471D+00,0.21131D+00,
     & 0.22068D+00,0.22783D+00,0.23527D+00,0.24302D+00,0.25109D+00,
     & 0.25952D+00,0.26830D+00,0.27746D+00,0.28703D+00,0.29704D+00,
     & 0.30748D+00,0.31842D+00,0.32634D+00,0.33815D+00,0.35052D+00,
     & 0.35960D+00,0.37296D+00,0.38700D+00,0.39733D+00,0.41254D+00,
     & 0.42381D+00,0.44028D+00,0.45249D+00,0.47044D+00,0.48366D+00,
     & 0.49748D+00,0.51758D+00,0.53250D+00,0.54806D+00,0.57068D+00,
     & 0.58745D+00,0.60491D+00,0.63060D+00,0.64936D+00,0.66890D+00,
     & 0.68922D+00,0.71034D+00,0.74142D+00,0.76404D+00,0.78756D+00,
     & 0.81202D+00,0.83742D+00,0.86381D+00,0.90346D+00,0.93159D+00,
     & 0.96078D+00,0.99107D+00,0.10225D+01,0.10551D+01,0.10890D+01,
     & 0.11243D+01,0.11605D+01,0.11982D+01,0.12581D+01,0.12989D+01,
     & 0.13414D+01,0.13848D+01,0.14300D+01,0.14768D+01,0.15253D+01,
     & 0.15755D+01,0.16274D+01,0.16811D+01,0.17367D+01,0.17956D+01,
     & 0.18540D+01,0.19517D+01,0.20111D+01,0.19912D+01,0.18481D+01,
     & 0.13716D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.14899D+00,0.15642D+00,
     & 0.16166D+00,0.16939D+00,0.17498D+00,0.18311D+00,0.18913D+00,
     & 0.19539D+00,0.20430D+00,0.21108D+00,0.21814D+00,0.22547D+00,
     & 0.23312D+00,0.24110D+00,0.24939D+00,0.25805D+00,0.26708D+00,
     & 0.27651D+00,0.28636D+00,0.29665D+00,0.30407D+00,0.31518D+00,
     & 0.32681D+00,0.33530D+00,0.34785D+00,0.36105D+00,0.37069D+00,
     & 0.38496D+00,0.39545D+00,0.41094D+00,0.42230D+00,0.43912D+00,
     & 0.45144D+00,0.46430D+00,0.48313D+00,0.49702D+00,0.51150D+00,
     & 0.53268D+00,0.54827D+00,0.56451D+00,0.58856D+00,0.60601D+00,
     & 0.62416D+00,0.64305D+00,0.66269D+00,0.69177D+00,0.71279D+00,
     & 0.73465D+00,0.75737D+00,0.78097D+00,0.80549D+00,0.84260D+00,
     & 0.86873D+00,0.89585D+00,0.92399D+00,0.95320D+00,0.98350D+00,
     & 0.10149D+01,0.10478D+01,0.10814D+01,0.11165D+01,0.11726D+01,
     & 0.12105D+01,0.12501D+01,0.12904D+01,0.13325D+01,0.13760D+01,
     & 0.14211D+01,0.14678D+01,0.15161D+01,0.15661D+01,0.16178D+01,
     & 0.16726D+01,0.17270D+01,0.18173D+01,0.18728D+01,0.18557D+01,
     & 0.17260D+01,0.12859D+01,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.13010D+00,0.13714D+00,
     & 0.14424D+00,0.14921D+00,0.15655D+00,0.16183D+00,0.16955D+00,
     & 0.17522D+00,0.18112D+00,0.18958D+00,0.19598D+00,0.20264D+00,
     & 0.20957D+00,0.21681D+00,0.22435D+00,0.23218D+00,0.24036D+00,
     & 0.24891D+00,0.25781D+00,0.26711D+00,0.27682D+00,0.28378D+00,
     & 0.29426D+00,0.30523D+00,0.31317D+00,0.32499D+00,0.33742D+00,
     & 0.34644D+00,0.35987D+00,0.36968D+00,0.38423D+00,0.39486D+00,
     & 0.41068D+00,0.42218D+00,0.43420D+00,0.45189D+00,0.46485D+00,
     & 0.47836D+00,0.49826D+00,0.51281D+00,0.52795D+00,0.55054D+00,
     & 0.56681D+00,0.58374D+00,0.60135D+00,0.61966D+00,0.64696D+00,
     & 0.66655D+00,0.68693D+00,0.70811D+00,0.73012D+00,0.75298D+00,
     & 0.78783D+00,0.81219D+00,0.83746D+00,0.86370D+00,0.89092D+00,
     & 0.91918D+00,0.94849D+00,0.97914D+00,0.10105D+01,0.10432D+01,
     & 0.10960D+01,0.11313D+01,0.11682D+01,0.12059D+01,0.12451D+01,
     & 0.12858D+01,0.13278D+01,0.13714D+01,0.14165D+01,0.14632D+01,
     & 0.15116D+01,0.15628D+01,0.16135D+01,0.16974D+01,0.17493D+01,
     & 0.17347D+01,0.16167D+01,0.12086D+01,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.11534D+00,0.11986D+00,
     & 0.12663D+00,0.13344D+00,0.13818D+00,0.14519D+00,0.15021D+00,
     & 0.15755D+00,0.16292D+00,0.16849D+00,0.17651D+00,0.18254D+00,
     & 0.18881D+00,0.19535D+00,0.20215D+00,0.20925D+00,0.21664D+00,
     & 0.22436D+00,0.23241D+00,0.24081D+00,0.24958D+00,0.25875D+00,
     & 0.26527D+00,0.27516D+00,0.28551D+00,0.29294D+00,0.30409D+00,
     & 0.31581D+00,0.32426D+00,0.33694D+00,0.34613D+00,0.35984D+00,
     & 0.36979D+00,0.38469D+00,0.39546D+00,0.40670D+00,0.42337D+00,
     & 0.43550D+00,0.44814D+00,0.46688D+00,0.48048D+00,0.49464D+00,
     & 0.51592D+00,0.53113D+00,0.54696D+00,0.56342D+00,0.58054D+00,
     & 0.60625D+00,0.62457D+00,0.64361D+00,0.66342D+00,0.68399D+00,
     & 0.70536D+00,0.73819D+00,0.76095D+00,0.78458D+00,0.80911D+00,
     & 0.83456D+00,0.86098D+00,0.88839D+00,0.91707D+00,0.94634D+00,
     & 0.97694D+00,0.10267D+01,0.10598D+01,0.10944D+01,0.11296D+01,
     & 0.11663D+01,0.12044D+01,0.12438D+01,0.12846D+01,0.13268D+01,
     & 0.13706D+01,0.14159D+01,0.14639D+01,0.15114D+01,0.15895D+01,
     & 0.16382D+01,0.16257D+01,0.15181D+01,0.11382D+01,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.99732D-01,0.10634D+00,
     & 0.11071D+00,0.11725D+00,0.12382D+00,0.12834D+00,0.13508D+00,
     & 0.13985D+00,0.14687D+00,0.15197D+00,0.15725D+00,0.16489D+00,
     & 0.17059D+00,0.17653D+00,0.18270D+00,0.18913D+00,0.19582D+00,
     & 0.20280D+00,0.21007D+00,0.21766D+00,0.22557D+00,0.23384D+00,
     & 0.24249D+00,0.24858D+00,0.25789D+00,0.26765D+00,0.27461D+00,
     & 0.28513D+00,0.29619D+00,0.30409D+00,0.31605D+00,0.32465D+00,
     & 0.33759D+00,0.34690D+00,0.36096D+00,0.37105D+00,0.38157D+00,
     & 0.39731D+00,0.40867D+00,0.42051D+00,0.43820D+00,0.45094D+00,
     & 0.46420D+00,0.48428D+00,0.49851D+00,0.51333D+00,0.52874D+00,
     & 0.54478D+00,0.56904D+00,0.58620D+00,0.60405D+00,0.62260D+00,
     & 0.64188D+00,0.66191D+00,0.69292D+00,0.71426D+00,0.73641D+00,
     & 0.75940D+00,0.78326D+00,0.80803D+00,0.83373D+00,0.86063D+00,
     & 0.88808D+00,0.91679D+00,0.96386D+00,0.99493D+00,0.10273D+01,
     & 0.10604D+01,0.10949D+01,0.11306D+01,0.11676D+01,0.12059D+01,
     & 0.12456D+01,0.12867D+01,0.13292D+01,0.13743D+01,0.14190D+01,
     & 0.14920D+01,0.15378D+01,0.15272D+01,0.14285D+01,0.10736D+01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.85168D-01,0.91792D-01,
     & 0.98248D-01,0.10248D+00,0.10883D+00,0.11518D+00,0.11953D+00,
     & 0.12602D+00,0.13058D+00,0.13732D+00,0.14217D+00,0.14720D+00,
     & 0.15450D+00,0.15992D+00,0.16555D+00,0.17141D+00,0.17750D+00,
     & 0.18384D+00,0.19045D+00,0.19733D+00,0.20450D+00,0.21199D+00,
     & 0.21981D+00,0.22799D+00,0.23369D+00,0.24249D+00,0.25170D+00,
     & 0.25823D+00,0.26816D+00,0.27861D+00,0.28601D+00,0.29730D+00,
     & 0.30535D+00,0.31758D+00,0.32631D+00,0.33960D+00,0.34905D+00,
     & 0.35892D+00,0.37378D+00,0.38443D+00,0.39553D+00,0.41224D+00,
     & 0.42419D+00,0.43663D+00,0.45562D+00,0.46898D+00,0.48289D+00,
     & 0.49737D+00,0.51242D+00,0.53537D+00,0.55147D+00,0.56823D+00,
     & 0.58565D+00,0.60375D+00,0.62256D+00,0.65191D+00,0.67195D+00,
     & 0.69275D+00,0.71435D+00,0.73677D+00,0.76005D+00,0.78421D+00,
     & 0.80950D+00,0.83531D+00,0.86231D+00,0.90694D+00,0.93617D+00,
     & 0.96669D+00,0.99779D+00,0.10302D+01,0.10639D+01,0.10987D+01,
     & 0.11348D+01,0.11722D+01,0.12109D+01,0.12510D+01,0.12935D+01,
     & 0.13356D+01,0.14040D+01,0.14473D+01,0.14382D+01,0.13474D+01,
     & 0.10145D+01,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.73691D-01,0.78065D-01,
     & 0.84590D-01,0.90923D-01,0.95048D-01,0.10124D+00,0.10740D+00,
     & 0.11160D+00,0.11786D+00,0.12224D+00,0.12873D+00,0.13338D+00,
     & 0.13818D+00,0.14518D+00,0.15034D+00,0.15571D+00,0.16128D+00,
     & 0.16707D+00,0.17310D+00,0.17937D+00,0.18590D+00,0.19272D+00,
     & 0.19982D+00,0.20724D+00,0.21499D+00,0.22035D+00,0.22869D+00,
     & 0.23743D+00,0.24356D+00,0.25297D+00,0.26287D+00,0.26983D+00,
     & 0.28052D+00,0.28809D+00,0.29968D+00,0.30789D+00,0.32049D+00,
     & 0.32937D+00,0.33865D+00,0.35273D+00,0.36274D+00,0.37318D+00,
     & 0.38902D+00,0.40025D+00,0.41196D+00,0.42997D+00,0.44254D+00,
     & 0.45563D+00,0.46926D+00,0.48342D+00,0.50520D+00,0.52036D+00,
     & 0.53613D+00,0.55253D+00,0.56959D+00,0.58730D+00,0.61516D+00,
     & 0.63404D+00,0.65364D+00,0.67399D+00,0.69513D+00,0.71707D+00,
     & 0.73985D+00,0.76371D+00,0.78805D+00,0.81352D+00,0.85596D+00,
     & 0.88356D+00,0.91237D+00,0.94173D+00,0.97240D+00,0.10042D+01,
     & 0.10371D+01,0.10712D+01,0.11065D+01,0.11431D+01,0.11810D+01,
     & 0.12213D+01,0.12611D+01,0.13255D+01,0.13665D+01,0.13587D+01,
     & 0.12748D+01,0.96151D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.60195D-01,0.67215D-01,
     & 0.71562D-01,0.78019D-01,0.84252D-01,0.88289D-01,0.94340D-01,
     & 0.10035D+00,0.10441D+00,0.11049D+00,0.11471D+00,0.12098D+00,
     & 0.12543D+00,0.13004D+00,0.13677D+00,0.14171D+00,0.14683D+00,
     & 0.15215D+00,0.15768D+00,0.16342D+00,0.16940D+00,0.17562D+00,
     & 0.18211D+00,0.18887D+00,0.19592D+00,0.20330D+00,0.20835D+00,
     & 0.21628D+00,0.22459D+00,0.23037D+00,0.23932D+00,0.24873D+00,
     & 0.25528D+00,0.26545D+00,0.27258D+00,0.28360D+00,0.29133D+00,
     & 0.30331D+00,0.31168D+00,0.32043D+00,0.33382D+00,0.34326D+00,
     & 0.35311D+00,0.36817D+00,0.37877D+00,0.38982D+00,0.40694D+00,
     & 0.41881D+00,0.43117D+00,0.44403D+00,0.45741D+00,0.47812D+00,
     & 0.49243D+00,0.50733D+00,0.52283D+00,0.53894D+00,0.55569D+00,
     & 0.58222D+00,0.60006D+00,0.61859D+00,0.63783D+00,0.65782D+00,
     & 0.67857D+00,0.70013D+00,0.72271D+00,0.74573D+00,0.76985D+00,
     & 0.81033D+00,0.83647D+00,0.86376D+00,0.89157D+00,0.92063D+00,
     & 0.95073D+00,0.98193D+00,0.10143D+01,0.10477D+01,0.10825D+01,
     & 0.11184D+01,0.11566D+01,0.11944D+01,0.12552D+01,0.12942D+01,
     & 0.12875D+01,0.12098D+01,0.91402D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.54187D-01,
     & 0.61228D-01,0.65565D-01,0.71979D-01,0.78143D-01,0.82111D-01,
     & 0.88051D-01,0.93930D-01,0.97879D-01,0.10379D+00,0.10787D+00,
     & 0.11394D+00,0.11824D+00,0.12266D+00,0.12916D+00,0.13390D+00,
     & 0.13880D+00,0.14390D+00,0.14918D+00,0.15467D+00,0.16038D+00,
     & 0.16633D+00,0.17252D+00,0.17897D+00,0.18571D+00,0.19275D+00,
     & 0.19751D+00,0.20508D+00,0.21300D+00,0.21846D+00,0.22699D+00,
     & 0.23596D+00,0.24216D+00,0.25185D+00,0.25859D+00,0.26909D+00,
     & 0.27641D+00,0.28782D+00,0.29574D+00,0.30401D+00,0.31677D+00,
     & 0.32570D+00,0.33502D+00,0.34937D+00,0.35940D+00,0.36986D+00,
     & 0.38618D+00,0.39742D+00,0.40912D+00,0.42129D+00,0.43396D+00,
     & 0.45372D+00,0.46727D+00,0.48139D+00,0.49607D+00,0.51134D+00,
     & 0.52720D+00,0.55253D+00,0.56944D+00,0.58700D+00,0.60525D+00,
     & 0.62419D+00,0.64389D+00,0.66433D+00,0.68576D+00,0.70761D+00,
     & 0.73049D+00,0.76921D+00,0.79402D+00,0.81995D+00,0.84636D+00,
     & 0.87397D+00,0.90258D+00,0.93222D+00,0.96295D+00,0.99480D+00,
     & 0.10278D+01,0.10620D+01,0.10984D+01,0.11343D+01,0.11919D+01,
     & 0.12290D+01,0.12234D+01,0.11512D+01,0.87117D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.40846D-01,
     & 0.48556D-01,0.55649D-01,0.59996D-01,0.66397D-01,0.72517D-01,
     & 0.76432D-01,0.82284D-01,0.88055D-01,0.91907D-01,0.97681D-01,
     & 0.10164D+00,0.10755D+00,0.11169D+00,0.11596D+00,0.12224D+00,
     & 0.12680D+00,0.13151D+00,0.13640D+00,0.14147D+00,0.14674D+00,
     & 0.15221D+00,0.15790D+00,0.16383D+00,0.17001D+00,0.17645D+00,
     & 0.18318D+00,0.18770D+00,0.19493D+00,0.20251D+00,0.20768D+00,
     & 0.21583D+00,0.22441D+00,0.23028D+00,0.23954D+00,0.24593D+00,
     & 0.25596D+00,0.26289D+00,0.27380D+00,0.28131D+00,0.28916D+00,
     & 0.30135D+00,0.30982D+00,0.31865D+00,0.33237D+00,0.34189D+00,
     & 0.35181D+00,0.36741D+00,0.37807D+00,0.38917D+00,0.40073D+00,
     & 0.41276D+00,0.43165D+00,0.44452D+00,0.45793D+00,0.47187D+00,
     & 0.48638D+00,0.50145D+00,0.52569D+00,0.54176D+00,0.55845D+00,
     & 0.57579D+00,0.59381D+00,0.61253D+00,0.63198D+00,0.65236D+00,
     & 0.67315D+00,0.69492D+00,0.73203D+00,0.75565D+00,0.78035D+00,
     & 0.80550D+00,0.83179D+00,0.85905D+00,0.88730D+00,0.91658D+00,
     & 0.94693D+00,0.97840D+00,0.10110D+01,0.10457D+01,0.10800D+01,
     & 0.11346D+01,0.11701D+01,0.11654D+01,0.10982D+01,0.83239D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.26627D-01,
     & 0.35372D-01,0.43230D-01,0.50412D-01,0.54790D-01,0.61201D-01,
     & 0.67302D-01,0.71180D-01,0.76968D-01,0.82654D-01,0.86424D-01,
     & 0.92080D-01,0.95931D-01,0.10169D+00,0.10571D+00,0.10983D+00,
     & 0.11594D+00,0.12033D+00,0.12487D+00,0.12958D+00,0.13445D+00,
     & 0.13952D+00,0.14478D+00,0.15024D+00,0.15593D+00,0.16186D+00,
     & 0.16804D+00,0.17449D+00,0.17878D+00,0.18572D+00,0.19297D+00,
     & 0.19789D+00,0.20570D+00,0.21391D+00,0.21949D+00,0.22836D+00,
     & 0.23443D+00,0.24404D+00,0.25063D+00,0.26108D+00,0.26821D+00,
     & 0.27567D+00,0.28735D+00,0.29540D+00,0.30380D+00,0.31694D+00,
     & 0.32599D+00,0.33543D+00,0.35038D+00,0.36052D+00,0.37108D+00,
     & 0.38208D+00,0.39353D+00,0.41163D+00,0.42389D+00,0.43665D+00,
     & 0.44992D+00,0.46373D+00,0.47810D+00,0.50134D+00,0.51665D+00,
     & 0.53255D+00,0.54908D+00,0.56625D+00,0.58409D+00,0.60263D+00,
     & 0.62207D+00,0.64189D+00,0.66266D+00,0.69831D+00,0.72085D+00,
     & 0.74442D+00,0.76843D+00,0.79354D+00,0.81956D+00,0.84654D+00,
     & 0.87452D+00,0.90351D+00,0.93358D+00,0.96476D+00,0.99797D+00,
     & 0.10307D+01,0.10827D+01,0.11166D+01,0.11128D+01,0.10501D+01,
     & 0.79716D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.10744D-01,
     & 0.21073D-01,0.30100D-01,0.38148D-01,0.45453D-01,0.49881D-01,
     & 0.56333D-01,0.62439D-01,0.66296D-01,0.72041D-01,0.77663D-01,
     & 0.81367D-01,0.86924D-01,0.90684D-01,0.96316D-01,0.10022D+00,
     & 0.10422D+00,0.11016D+00,0.11441D+00,0.11880D+00,0.12334D+00,
     & 0.12804D+00,0.13292D+00,0.13799D+00,0.14325D+00,0.14872D+00,
     & 0.15442D+00,0.16037D+00,0.16657D+00,0.17065D+00,0.17731D+00,
     & 0.18429D+00,0.18897D+00,0.19647D+00,0.20436D+00,0.20967D+00,
     & 0.21818D+00,0.22397D+00,0.23319D+00,0.23947D+00,0.24950D+00,
     & 0.25629D+00,0.26340D+00,0.27461D+00,0.28228D+00,0.29029D+00,
     & 0.30290D+00,0.31153D+00,0.32052D+00,0.33488D+00,0.34455D+00,
     & 0.35462D+00,0.36511D+00,0.37603D+00,0.39342D+00,0.40511D+00,
     & 0.41728D+00,0.42996D+00,0.44314D+00,0.45685D+00,0.47919D+00,
     & 0.49380D+00,0.50899D+00,0.52477D+00,0.54117D+00,0.55822D+00,
     & 0.57593D+00,0.59452D+00,0.61345D+00,0.63331D+00,0.66763D+00,
     & 0.68919D+00,0.71174D+00,0.73470D+00,0.75872D+00,0.78363D+00,
     & 0.80946D+00,0.83623D+00,0.86400D+00,0.89280D+00,0.92266D+00,
     & 0.95449D+00,0.98582D+00,0.10354D+01,0.10680D+01,0.10650D+01,
     & 0.10063D+01,0.76507D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.47398D-02,0.15599D-01,0.24969D-01,0.33255D-01,0.40718D-01,
     & 0.45219D-01,0.51740D-01,0.57875D-01,0.61726D-01,0.67452D-01,
     & 0.73030D-01,0.76677D-01,0.82157D-01,0.85839D-01,0.91364D-01,
     & 0.95166D-01,0.99062D-01,0.10486D+00,0.10898D+00,0.11323D+00,
     & 0.11762D+00,0.12217D+00,0.12688D+00,0.13177D+00,0.13685D+00,
     & 0.14213D+00,0.14762D+00,0.15335D+00,0.15933D+00,0.16322D+00,
     & 0.16964D+00,0.17635D+00,0.18081D+00,0.18804D+00,0.19563D+00,
     & 0.20069D+00,0.20889D+00,0.21441D+00,0.22328D+00,0.22928D+00,
     & 0.23893D+00,0.24541D+00,0.25220D+00,0.26299D+00,0.27031D+00,
     & 0.27796D+00,0.29009D+00,0.29833D+00,0.30692D+00,0.32074D+00,
     & 0.32998D+00,0.33961D+00,0.34963D+00,0.36007D+00,0.37681D+00,
     & 0.38798D+00,0.39962D+00,0.41174D+00,0.42435D+00,0.43747D+00,
     & 0.45899D+00,0.47297D+00,0.48750D+00,0.50260D+00,0.51830D+00,
     & 0.53461D+00,0.55158D+00,0.56938D+00,0.58751D+00,0.60654D+00,
     & 0.63963D+00,0.66030D+00,0.68192D+00,0.70393D+00,0.72696D+00,
     & 0.75085D+00,0.77561D+00,0.80130D+00,0.82795D+00,0.85558D+00,
     & 0.88425D+00,0.91481D+00,0.94487D+00,0.99225D+00,0.10236D+01,
     & 0.10213D+01,0.96627D+00,0.73577D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.10144D-01,0.19922D-01,0.28496D-01,
     & 0.36159D-01,0.40755D-01,0.47375D-01,0.53564D-01,0.57425D-01,
     & 0.63152D-01,0.68704D-01,0.72310D-01,0.77732D-01,0.81347D-01,
     & 0.86785D-01,0.90500D-01,0.94301D-01,0.99972D-01,0.10398D+00,
     & 0.10810D+00,0.11237D+00,0.11677D+00,0.12134D+00,0.12607D+00,
     & 0.13098D+00,0.13608D+00,0.14139D+00,0.14692D+00,0.15269D+00,
     & 0.15642D+00,0.16261D+00,0.16909D+00,0.17335D+00,0.18032D+00,
     & 0.18764D+00,0.19248D+00,0.20039D+00,0.20567D+00,0.21422D+00,
     & 0.21995D+00,0.22926D+00,0.23546D+00,0.24195D+00,0.25235D+00,
     & 0.25936D+00,0.26668D+00,0.27838D+00,0.28626D+00,0.29449D+00,
     & 0.30781D+00,0.31666D+00,0.32588D+00,0.33548D+00,0.34547D+00,
     & 0.36162D+00,0.37232D+00,0.38347D+00,0.39509D+00,0.40717D+00,
     & 0.41974D+00,0.44051D+00,0.45391D+00,0.46784D+00,0.48232D+00,
     & 0.49738D+00,0.51303D+00,0.52930D+00,0.54638D+00,0.56378D+00,
     & 0.58204D+00,0.61403D+00,0.63387D+00,0.65464D+00,0.67578D+00,
     & 0.69790D+00,0.72085D+00,0.74465D+00,0.76934D+00,0.79495D+00,
     & 0.82152D+00,0.84908D+00,0.87850D+00,0.90740D+00,0.95276D+00,
     & 0.98292D+00,0.98124D+00,0.92965D+00,0.70893D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.46055D-02,0.14903D-01,
     & 0.23821D-01,0.31729D-01,0.36444D-01,0.43191D-01,0.49463D-01,
     & 0.53349D-01,0.59097D-01,0.64646D-01,0.68225D-01,0.73603D-01,
     & 0.77167D-01,0.82532D-01,0.86174D-01,0.89890D-01,0.95454D-01,
     & 0.99357D-01,0.10338D+00,0.10752D+00,0.11180D+00,0.11623D+00,
     & 0.12082D+00,0.12558D+00,0.13052D+00,0.13566D+00,0.14101D+00,
     & 0.14660D+00,0.15017D+00,0.15615D+00,0.16241D+00,0.16650D+00,
     & 0.17323D+00,0.18031D+00,0.18495D+00,0.19259D+00,0.19764D+00,
     & 0.20591D+00,0.21140D+00,0.22039D+00,0.22634D+00,0.23256D+00,
     & 0.24260D+00,0.24932D+00,0.25634D+00,0.26764D+00,0.27520D+00,
     & 0.28309D+00,0.29596D+00,0.30445D+00,0.31329D+00,0.32250D+00,
     & 0.33209D+00,0.34770D+00,0.35797D+00,0.36867D+00,0.37982D+00,
     & 0.39142D+00,0.40349D+00,0.42357D+00,0.43644D+00,0.44982D+00,
     & 0.46373D+00,0.47820D+00,0.49324D+00,0.50888D+00,0.52530D+00,
     & 0.54203D+00,0.55958D+00,0.59055D+00,0.60964D+00,0.62963D+00,
     & 0.64996D+00,0.67126D+00,0.69334D+00,0.71626D+00,0.74003D+00,
     & 0.76469D+00,0.79028D+00,0.81683D+00,0.84519D+00,0.87302D+00,
     & 0.91653D+00,0.94563D+00,0.94453D+00,0.89605D+00,0.68430D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.98577D-02,0.19183D-01,0.27384D-01,0.32244D-01,0.39152D-01,
     & 0.45534D-01,0.49462D-01,0.55253D-01,0.60817D-01,0.64380D-01,
     & 0.69734D-01,0.73257D-01,0.78566D-01,0.82146D-01,0.85791D-01,
     & 0.91263D-01,0.95076D-01,0.98997D-01,0.10304D+00,0.10721D+00,
     & 0.11151D+00,0.11597D+00,0.12060D+00,0.12539D+00,0.13038D+00,
     & 0.13557D+00,0.14099D+00,0.14441D+00,0.15021D+00,0.15628D+00,
     & 0.16020D+00,0.16672D+00,0.17357D+00,0.17802D+00,0.18542D+00,
     & 0.19027D+00,0.19827D+00,0.20354D+00,0.21224D+00,0.21795D+00,
     & 0.22393D+00,0.23364D+00,0.24010D+00,0.24684D+00,0.25778D+00,
     & 0.26504D+00,0.27262D+00,0.28508D+00,0.29323D+00,0.30173D+00,
     & 0.31058D+00,0.31981D+00,0.33491D+00,0.34478D+00,0.35508D+00,
     & 0.36580D+00,0.37696D+00,0.38857D+00,0.40802D+00,0.42040D+00,
     & 0.43327D+00,0.44666D+00,0.46058D+00,0.47506D+00,0.49011D+00,
     & 0.50593D+00,0.52204D+00,0.53895D+00,0.56898D+00,0.58737D+00,
     & 0.60664D+00,0.62624D+00,0.64676D+00,0.66806D+00,0.69015D+00,
     & 0.71308D+00,0.73687D+00,0.76156D+00,0.78718D+00,0.81456D+00,
     & 0.84141D+00,0.88321D+00,0.91133D+00,0.91076D+00,0.86515D+00,
     & 0.66164D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.47067D-02,0.14533D-01,0.23080D-01,0.28114D-01,
     & 0.35222D-01,0.41741D-01,0.45729D-01,0.51586D-01,0.57184D-01,
     & 0.60744D-01,0.66092D-01,0.69584D-01,0.74854D-01,0.78381D-01,
     & 0.81968D-01,0.87362D-01,0.91097D-01,0.94932D-01,0.98878D-01,
     & 0.10295D+00,0.10715D+00,0.11149D+00,0.11599D+00,0.12066D+00,
     & 0.12551D+00,0.13055D+00,0.13581D+00,0.13910D+00,0.14473D+00,
     & 0.15062D+00,0.15439D+00,0.16072D+00,0.16736D+00,0.17165D+00,
     & 0.17881D+00,0.18348D+00,0.19124D+00,0.19631D+00,0.20474D+00,
     & 0.21024D+00,0.21599D+00,0.22540D+00,0.23161D+00,0.23810D+00,
     & 0.24870D+00,0.25569D+00,0.26299D+00,0.27507D+00,0.28291D+00,
     & 0.29109D+00,0.29962D+00,0.30850D+00,0.32314D+00,0.33265D+00,
     & 0.34257D+00,0.35289D+00,0.36364D+00,0.37483D+00,0.39370D+00,
     & 0.40563D+00,0.41804D+00,0.43094D+00,0.44436D+00,0.45832D+00,
     & 0.47284D+00,0.48810D+00,0.50363D+00,0.51994D+00,0.54912D+00,
     & 0.56687D+00,0.58547D+00,0.60439D+00,0.62420D+00,0.64477D+00,
     & 0.66611D+00,0.68826D+00,0.71124D+00,0.73510D+00,0.75985D+00,
     & 0.78633D+00,0.81227D+00,0.85249D+00,0.87971D+00,0.87964D+00,
     & 0.83668D+00,0.64076D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.70402D-04,0.98217D-02,0.18777D-01,
     & 0.24017D-01,0.31358D-01,0.38052D-01,0.42118D-01,0.48064D-01,
     & 0.53717D-01,0.57287D-01,0.62645D-01,0.66119D-01,0.71364D-01,
     & 0.74850D-01,0.78386D-01,0.83719D-01,0.87387D-01,0.91147D-01,
     & 0.95010D-01,0.98988D-01,0.10309D+00,0.10733D+00,0.11172D+00,
     & 0.11627D+00,0.12099D+00,0.12591D+00,0.13102D+00,0.13420D+00,
     & 0.13967D+00,0.14540D+00,0.14903D+00,0.15517D+00,0.16163D+00,
     & 0.16576D+00,0.17272D+00,0.17722D+00,0.18475D+00,0.18964D+00,
     & 0.19783D+00,0.20312D+00,0.20866D+00,0.21780D+00,0.22379D+00,
     & 0.23004D+00,0.24033D+00,0.24707D+00,0.25410D+00,0.26584D+00,
     & 0.27340D+00,0.28129D+00,0.28951D+00,0.29807D+00,0.31230D+00,
     & 0.32147D+00,0.33103D+00,0.34100D+00,0.35137D+00,0.36217D+00,
     & 0.38051D+00,0.39202D+00,0.40400D+00,0.41645D+00,0.42941D+00,
     & 0.44289D+00,0.45691D+00,0.47165D+00,0.48665D+00,0.50241D+00,
     & 0.53080D+00,0.54796D+00,0.56594D+00,0.58423D+00,0.60339D+00,
     & 0.62328D+00,0.64392D+00,0.66535D+00,0.68758D+00,0.71067D+00,
     & 0.73463D+00,0.76028D+00,0.78537D+00,0.82413D+00,0.85051D+00,
     & 0.85090D+00,0.81040D+00,0.62148D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.49900D-02,
     & 0.14429D-01,0.19911D-01,0.27539D-01,0.34434D-01,0.38596D-01,
     & 0.44658D-01,0.50387D-01,0.53981D-01,0.59365D-01,0.62832D-01,
     & 0.68069D-01,0.71525D-01,0.75020D-01,0.80307D-01,0.83917D-01,
     & 0.87611D-01,0.91402D-01,0.95301D-01,0.99318D-01,0.10346D+00,
     & 0.10775D+00,0.11220D+00,0.11681D+00,0.12160D+00,0.12659D+00,
     & 0.12965D+00,0.13498D+00,0.14056D+00,0.14407D+00,0.15004D+00,
     & 0.15634D+00,0.16032D+00,0.16709D+00,0.17143D+00,0.17876D+00,
     & 0.18347D+00,0.19144D+00,0.19655D+00,0.20189D+00,0.21079D+00,
     & 0.21656D+00,0.22260D+00,0.23261D+00,0.23911D+00,0.24590D+00,
     & 0.25732D+00,0.26462D+00,0.27224D+00,0.28018D+00,0.28845D+00,
     & 0.30229D+00,0.31115D+00,0.32039D+00,0.33001D+00,0.34004D+00,
     & 0.35047D+00,0.36832D+00,0.37945D+00,0.39103D+00,0.40307D+00,
     & 0.41559D+00,0.42863D+00,0.44219D+00,0.45645D+00,0.47096D+00,
     & 0.48621D+00,0.51387D+00,0.53048D+00,0.54790D+00,0.56560D+00,
     & 0.58415D+00,0.60341D+00,0.62341D+00,0.64416D+00,0.66570D+00,
     & 0.68807D+00,0.71130D+00,0.73617D+00,0.76049D+00,0.79789D+00,
     & 0.82349D+00,0.82431D+00,0.78610D+00,0.60364D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.28952D-03,0.99912D-02,0.15757D-01,0.23716D-01,0.30855D-01,
     & 0.35136D-01,0.41339D-01,0.47168D-01,0.50798D-01,0.56229D-01,
     & 0.59699D-01,0.64944D-01,0.68378D-01,0.71844D-01,0.77097D-01,
     & 0.80658D-01,0.84298D-01,0.88026D-01,0.91857D-01,0.95796D-01,
     & 0.99861D-01,0.10406D+00,0.10841D+00,0.11292D+00,0.11760D+00,
     & 0.12248D+00,0.12543D+00,0.13064D+00,0.13608D+00,0.13947D+00,
     & 0.14530D+00,0.15143D+00,0.15528D+00,0.16188D+00,0.16607D+00,
     & 0.17322D+00,0.17777D+00,0.18554D+00,0.19047D+00,0.19564D+00,
     & 0.20430D+00,0.20988D+00,0.21572D+00,0.22547D+00,0.23176D+00,
     & 0.23832D+00,0.24944D+00,0.25650D+00,0.26387D+00,0.27155D+00,
     & 0.27955D+00,0.29303D+00,0.30160D+00,0.31054D+00,0.31985D+00,
     & 0.32956D+00,0.33966D+00,0.35706D+00,0.36783D+00,0.37903D+00,
     & 0.39069D+00,0.40282D+00,0.41544D+00,0.42857D+00,0.44239D+00,
     & 0.45644D+00,0.47122D+00,0.49821D+00,0.51431D+00,0.53119D+00,
     & 0.54835D+00,0.56634D+00,0.58502D+00,0.60441D+00,0.62454D+00,
     & 0.64544D+00,0.66715D+00,0.68968D+00,0.71384D+00,0.73743D+00,
     & 0.77356D+00,0.79845D+00,0.79967D+00,0.76360D+00,0.58712D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.54151D-02,0.11512D-01,0.19860D-01,
     & 0.27285D-01,0.31707D-01,0.38082D-01,0.44035D-01,0.47715D-01,
     & 0.53211D-01,0.56697D-01,0.61965D-01,0.65388D-01,0.68834D-01,
     & 0.74067D-01,0.77590D-01,0.81182D-01,0.84857D-01,0.88627D-01,
     & 0.92502D-01,0.96495D-01,0.10062D+00,0.10488D+00,0.10930D+00,
     & 0.11388D+00,0.11865D+00,0.12151D+00,0.12660D+00,0.13192D+00,
     & 0.13520D+00,0.14090D+00,0.14688D+00,0.15060D+00,0.15706D+00,
     & 0.16111D+00,0.16808D+00,0.17249D+00,0.18007D+00,0.18484D+00,
     & 0.18984D+00,0.19830D+00,0.20370D+00,0.20935D+00,0.21886D+00,
     & 0.22495D+00,0.23130D+00,0.24216D+00,0.24899D+00,0.25612D+00,
     & 0.26356D+00,0.27131D+00,0.28447D+00,0.29277D+00,0.30143D+00,
     & 0.31045D+00,0.31985D+00,0.32964D+00,0.34663D+00,0.35706D+00,
     & 0.36792D+00,0.37922D+00,0.39098D+00,0.40322D+00,0.41595D+00,
     & 0.42936D+00,0.44299D+00,0.45732D+00,0.48370D+00,0.49932D+00,
     & 0.51571D+00,0.53237D+00,0.54983D+00,0.56797D+00,0.58680D+00,
     & 0.60635D+00,0.62665D+00,0.64774D+00,0.66963D+00,0.69313D+00,
     & 0.71604D+00,0.75099D+00,0.77520D+00,0.77681D+00,0.74273D+00,
     & 0.57178D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.66511D-03,0.71296D-02,
     & 0.15931D-01,0.23692D-01,0.28282D-01,0.34861D-01,0.40963D-01,
     & 0.44710D-01,0.50290D-01,0.53803D-01,0.59110D-01,0.62533D-01,
     & 0.65970D-01,0.71196D-01,0.74688D-01,0.78243D-01,0.81875D-01,
     & 0.85593D-01,0.89411D-01,0.93340D-01,0.97391D-01,0.10158D+00,
     & 0.10591D+00,0.11041D+00,0.11509D+00,0.11786D+00,0.12285D+00,
     & 0.12806D+00,0.13123D+00,0.13680D+00,0.14266D+00,0.14627D+00,
     & 0.15258D+00,0.15650D+00,0.16332D+00,0.16759D+00,0.17500D+00,
     & 0.17962D+00,0.18447D+00,0.19273D+00,0.19797D+00,0.20344D+00,
     & 0.21274D+00,0.21864D+00,0.22480D+00,0.23540D+00,0.24203D+00,
     & 0.24894D+00,0.25616D+00,0.26367D+00,0.27653D+00,0.28458D+00,
     & 0.29298D+00,0.30173D+00,0.31085D+00,0.32035D+00,0.33696D+00,
     & 0.34708D+00,0.35762D+00,0.36859D+00,0.38000D+00,0.39188D+00,
     & 0.40425D+00,0.41727D+00,0.43050D+00,0.44443D+00,0.47023D+00,
     & 0.48542D+00,0.50135D+00,0.51753D+00,0.53451D+00,0.55214D+00,
     & 0.57044D+00,0.58946D+00,0.60920D+00,0.62970D+00,0.65100D+00,
     & 0.67388D+00,0.69615D+00,0.73000D+00,0.75358D+00,0.75555D+00,
     & 0.72335D+00,0.55753D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.24240D-02,0.11891D-01,0.20042D-01,0.24829D-01,0.31648D-01,
     & 0.37930D-01,0.41758D-01,0.47445D-01,0.50998D-01,0.56361D-01,
     & 0.59795D-01,0.63230D-01,0.68464D-01,0.71936D-01,0.75463D-01,
     & 0.79058D-01,0.82734D-01,0.86504D-01,0.90377D-01,0.94368D-01,
     & 0.98488D-01,0.10275D+00,0.10717D+00,0.11176D+00,0.11445D+00,
     & 0.11934D+00,0.12445D+00,0.12753D+00,0.13300D+00,0.13874D+00,
     & 0.14223D+00,0.14841D+00,0.15222D+00,0.15890D+00,0.16304D+00,
     & 0.17029D+00,0.17478D+00,0.17948D+00,0.18757D+00,0.19265D+00,
     & 0.19796D+00,0.20706D+00,0.21278D+00,0.21876D+00,0.22914D+00,
     & 0.23557D+00,0.24229D+00,0.24929D+00,0.25659D+00,0.26917D+00,
     & 0.27699D+00,0.28514D+00,0.29364D+00,0.30250D+00,0.31173D+00,
     & 0.32799D+00,0.33783D+00,0.34806D+00,0.35872D+00,0.36981D+00,
     & 0.38136D+00,0.39338D+00,0.40604D+00,0.41890D+00,0.43244D+00,
     & 0.45773D+00,0.47250D+00,0.48800D+00,0.50374D+00,0.52027D+00,
     & 0.53742D+00,0.55524D+00,0.57375D+00,0.59297D+00,0.61293D+00,
     & 0.63367D+00,0.65597D+00,0.67765D+00,0.71047D+00,0.73346D+00,
     & 0.73577D+00,0.70533D+00,0.54428D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.76940D-02,0.16301D-01,0.21317D-01,
     & 0.28416D-01,0.34909D-01,0.38838D-01,0.44654D-01,0.48260D-01,
     & 0.53697D-01,0.57152D-01,0.60597D-01,0.65853D-01,0.69313D-01,
     & 0.72820D-01,0.76389D-01,0.80031D-01,0.83760D-01,0.87588D-01,
     & 0.91526D-01,0.95587D-01,0.99787D-01,0.10414D+00,0.10866D+00,
     & 0.11127D+00,0.11608D+00,0.12109D+00,0.12408D+00,0.12945D+00,
     & 0.13508D+00,0.13848D+00,0.14454D+00,0.14824D+00,0.15479D+00,
     & 0.15880D+00,0.16592D+00,0.17028D+00,0.17484D+00,0.18277D+00,
     & 0.18770D+00,0.19287D+00,0.20178D+00,0.20735D+00,0.21316D+00,
     & 0.22333D+00,0.22958D+00,0.23610D+00,0.24291D+00,0.25001D+00,
     & 0.26234D+00,0.26994D+00,0.27786D+00,0.28613D+00,0.29475D+00,
     & 0.30373D+00,0.31966D+00,0.32923D+00,0.33918D+00,0.34955D+00,
     & 0.36034D+00,0.37158D+00,0.38327D+00,0.39560D+00,0.40811D+00,
     & 0.42130D+00,0.44611D+00,0.46049D+00,0.47559D+00,0.49092D+00,
     & 0.50702D+00,0.52373D+00,0.54109D+00,0.55913D+00,0.57785D+00,
     & 0.59732D+00,0.61753D+00,0.63929D+00,0.66041D+00,0.69226D+00,
     & 0.71471D+00,0.71734D+00,0.68856D+00,0.53193D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.32064D-02,0.12429D-01,
     & 0.17712D-01,0.25139D-01,0.31879D-01,0.35927D-01,0.41897D-01,
     & 0.45571D-01,0.51101D-01,0.54588D-01,0.58054D-01,0.63346D-01,
     & 0.66803D-01,0.70299D-01,0.73849D-01,0.77467D-01,0.81164D-01,
     & 0.84951D-01,0.88847D-01,0.92859D-01,0.97003D-01,0.10129D+00,
     & 0.10575D+00,0.10828D+00,0.11302D+00,0.11796D+00,0.12086D+00,
     & 0.12614D+00,0.13168D+00,0.13498D+00,0.14093D+00,0.14453D+00,
     & 0.15096D+00,0.15487D+00,0.16185D+00,0.16609D+00,0.17053D+00,
     & 0.17831D+00,0.18311D+00,0.18813D+00,0.19688D+00,0.20230D+00,
     & 0.20795D+00,0.21793D+00,0.22401D+00,0.23036D+00,0.23698D+00,
     & 0.24389D+00,0.25599D+00,0.26338D+00,0.27110D+00,0.27915D+00,
     & 0.28754D+00,0.29628D+00,0.31192D+00,0.32123D+00,0.33093D+00,
     & 0.34102D+00,0.35153D+00,0.36248D+00,0.37387D+00,0.38589D+00,
     & 0.39808D+00,0.41092D+00,0.43529D+00,0.44932D+00,0.46404D+00,
     & 0.47899D+00,0.49468D+00,0.51098D+00,0.52791D+00,0.54550D+00,
     & 0.56378D+00,0.58276D+00,0.60249D+00,0.62374D+00,0.64434D+00,
     & 0.67527D+00,0.69721D+00,0.70015D+00,0.67295D+00,0.52042D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.83839D-02,0.13979D-01,0.21785D-01,0.28812D-01,0.33002D-01,
     & 0.39154D-01,0.42912D-01,0.48555D-01,0.52086D-01,0.55583D-01,
     & 0.60926D-01,0.64390D-01,0.67884D-01,0.71424D-01,0.75025D-01,
     & 0.78698D-01,0.82457D-01,0.86316D-01,0.90288D-01,0.94384D-01,
     & 0.98621D-01,0.10302D+00,0.10549D+00,0.11016D+00,0.11502D+00,
     & 0.11785D+00,0.12305D+00,0.12850D+00,0.13171D+00,0.13757D+00,
     & 0.14107D+00,0.14740D+00,0.15120D+00,0.15807D+00,0.16219D+00,
     & 0.16652D+00,0.17417D+00,0.17884D+00,0.18373D+00,0.19233D+00,
     & 0.19760D+00,0.20311D+00,0.21291D+00,0.21884D+00,0.22502D+00,
     & 0.23147D+00,0.23820D+00,0.25009D+00,0.25730D+00,0.26482D+00,
     & 0.27266D+00,0.28083D+00,0.28935D+00,0.30472D+00,0.31380D+00,
     & 0.32325D+00,0.33309D+00,0.34334D+00,0.35401D+00,0.36512D+00,
     & 0.37684D+00,0.38873D+00,0.40126D+00,0.42523D+00,0.43891D+00,
     & 0.45329D+00,0.46787D+00,0.48318D+00,0.49910D+00,0.51563D+00,
     & 0.53280D+00,0.55064D+00,0.56919D+00,0.58846D+00,0.60923D+00,
     & 0.62933D+00,0.65941D+00,0.68086D+00,0.68410D+00,0.65839D+00,
     & 0.50968D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.40855D-02,0.10076D-01,0.18323D-01,0.25682D-01,
     & 0.30039D-01,0.36403D-01,0.40261D-01,0.46041D-01,0.49629D-01,
     & 0.53168D-01,0.58579D-01,0.62059D-01,0.65560D-01,0.69099D-01,
     & 0.72690D-01,0.76348D-01,0.80086D-01,0.83915D-01,0.87853D-01,
     & 0.91911D-01,0.96105D-01,0.10045D+00,0.10286D+00,0.10747D+00,
     & 0.11227D+00,0.11503D+00,0.12015D+00,0.12553D+00,0.12866D+00,
     & 0.13443D+00,0.13784D+00,0.14407D+00,0.14778D+00,0.15454D+00,
     & 0.15856D+00,0.16278D+00,0.17031D+00,0.17486D+00,0.17963D+00,
     & 0.18809D+00,0.19323D+00,0.19860D+00,0.20825D+00,0.21402D+00,
     & 0.22005D+00,0.22635D+00,0.23291D+00,0.24461D+00,0.25163D+00,
     & 0.25897D+00,0.26662D+00,0.27459D+00,0.28291D+00,0.29803D+00,
     & 0.30689D+00,0.31611D+00,0.32571D+00,0.33571D+00,0.34612D+00,
     & 0.35697D+00,0.36841D+00,0.38001D+00,0.39224D+00,0.41585D+00,
     & 0.42922D+00,0.44327D+00,0.45750D+00,0.47247D+00,0.48801D+00,
     & 0.50417D+00,0.52095D+00,0.53839D+00,0.55651D+00,0.57535D+00,
     & 0.59568D+00,0.61532D+00,0.64459D+00,0.66558D+00,0.66911D+00,
     & 0.64482D+00,0.49964D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.10106D-03,0.59565D-02,0.14716D-01,
     & 0.22461D-01,0.27011D-01,0.33623D-01,0.37600D-01,0.43538D-01,
     & 0.47199D-01,0.50794D-01,0.56289D-01,0.59795D-01,0.63312D-01,
     & 0.66859D-01,0.70451D-01,0.74101D-01,0.77824D-01,0.81635D-01,
     & 0.85546D-01,0.89572D-01,0.93730D-01,0.98033D-01,0.10038D+00,
     & 0.10494D+00,0.10969D+00,0.11238D+00,0.11744D+00,0.12275D+00,
     & 0.12580D+00,0.13150D+00,0.13483D+00,0.14098D+00,0.14459D+00,
     & 0.15126D+00,0.15518D+00,0.15929D+00,0.16671D+00,0.17116D+00,
     & 0.17581D+00,0.18415D+00,0.18916D+00,0.19440D+00,0.20391D+00,
     & 0.20955D+00,0.21543D+00,0.22158D+00,0.22798D+00,0.23950D+00,
     & 0.24636D+00,0.25353D+00,0.26100D+00,0.26878D+00,0.27690D+00,
     & 0.29180D+00,0.30045D+00,0.30946D+00,0.31883D+00,0.32860D+00,
     & 0.33877D+00,0.34937D+00,0.36056D+00,0.37188D+00,0.38384D+00,
     & 0.40712D+00,0.42018D+00,0.43392D+00,0.44784D+00,0.46247D+00,
     & 0.47767D+00,0.49347D+00,0.50989D+00,0.52694D+00,0.54468D+00,
     & 0.56312D+00,0.58302D+00,0.60222D+00,0.63072D+00,0.65129D+00,
     & 0.65509D+00,0.63215D+00,0.49026D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.14128D-02,
     & 0.10923D-01,0.19116D-01,0.23891D-01,0.30790D-01,0.34908D-01,
     & 0.41035D-01,0.44780D-01,0.48443D-01,0.54041D-01,0.57583D-01,
     & 0.61128D-01,0.64691D-01,0.68291D-01,0.71942D-01,0.75660D-01,
     & 0.79458D-01,0.83352D-01,0.87354D-01,0.91483D-01,0.95753D-01,
     & 0.98049D-01,0.10257D+00,0.10727D+00,0.10989D+00,0.11490D+00,
     & 0.12015D+00,0.12313D+00,0.12876D+00,0.13201D+00,0.13809D+00,
     & 0.14162D+00,0.14820D+00,0.15203D+00,0.15605D+00,0.16337D+00,
     & 0.16771D+00,0.17225D+00,0.18048D+00,0.18538D+00,0.19050D+00,
     & 0.19988D+00,0.20538D+00,0.21113D+00,0.21714D+00,0.22340D+00,
     & 0.23476D+00,0.24146D+00,0.24846D+00,0.25576D+00,0.26338D+00,
     & 0.27131D+00,0.28601D+00,0.29446D+00,0.30326D+00,0.31243D+00,
     & 0.32198D+00,0.33192D+00,0.34228D+00,0.35323D+00,0.36430D+00,
     & 0.37600D+00,0.39897D+00,0.41176D+00,0.42521D+00,0.43882D+00,
     & 0.45314D+00,0.46802D+00,0.48349D+00,0.49956D+00,0.51625D+00,
     & 0.53361D+00,0.55166D+00,0.57118D+00,0.58997D+00,0.61774D+00,
     & 0.63791D+00,0.64198D+00,0.62033D+00,0.48148D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.68994D-02,0.15612D-01,0.20649D-01,0.27879D-01,
     & 0.32162D-01,0.38507D-01,0.42353D-01,0.46100D-01,0.51820D-01,
     & 0.55411D-01,0.58992D-01,0.62582D-01,0.66200D-01,0.69860D-01,
     & 0.73580D-01,0.77374D-01,0.81256D-01,0.85244D-01,0.89351D-01,
     & 0.93595D-01,0.95841D-01,0.10033D+00,0.10499D+00,0.10756D+00,
     & 0.11252D+00,0.11772D+00,0.12063D+00,0.12620D+00,0.12938D+00,
     & 0.13538D+00,0.13883D+00,0.14535D+00,0.14909D+00,0.15302D+00,
     & 0.16025D+00,0.16450D+00,0.16894D+00,0.17706D+00,0.18185D+00,
     & 0.18686D+00,0.19613D+00,0.20151D+00,0.20713D+00,0.21300D+00,
     & 0.21913D+00,0.23035D+00,0.23690D+00,0.24375D+00,0.25089D+00,
     & 0.25834D+00,0.26610D+00,0.28062D+00,0.28889D+00,0.29750D+00,
     & 0.30646D+00,0.31581D+00,0.32554D+00,0.33568D+00,0.34640D+00,
     & 0.35723D+00,0.36868D+00,0.39139D+00,0.40391D+00,0.41708D+00,
     & 0.43041D+00,0.44444D+00,0.45901D+00,0.47416D+00,0.48990D+00,
     & 0.50626D+00,0.52328D+00,0.54096D+00,0.56011D+00,0.57850D+00,
     & 0.60559D+00,0.62538D+00,0.62971D+00,0.60930D+00,0.47326D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.24503D-02,0.11907D-01,0.17250D-01,
     & 0.24863D-01,0.29336D-01,0.35933D-01,0.39901D-01,0.43747D-01,
     & 0.49611D-01,0.53262D-01,0.56891D-01,0.60518D-01,0.64162D-01,
     & 0.67842D-01,0.71573D-01,0.75371D-01,0.79251D-01,0.83230D-01,
     & 0.87324D-01,0.91549D-01,0.93749D-01,0.98208D-01,0.10284D+00,
     & 0.10535D+00,0.11028D+00,0.11543D+00,0.11828D+00,0.12381D+00,
     & 0.12691D+00,0.13286D+00,0.13624D+00,0.14268D+00,0.14635D+00,
     & 0.15019D+00,0.15735D+00,0.16150D+00,0.16585D+00,0.17389D+00,
     & 0.17857D+00,0.18347D+00,0.19264D+00,0.19791D+00,0.20341D+00,
     & 0.20915D+00,0.21515D+00,0.22625D+00,0.23266D+00,0.23936D+00,
     & 0.24635D+00,0.25365D+00,0.26125D+00,0.27561D+00,0.28370D+00,
     & 0.29213D+00,0.30091D+00,0.31006D+00,0.31959D+00,0.32952D+00,
     & 0.34003D+00,0.35063D+00,0.36185D+00,0.38432D+00,0.39659D+00,
     & 0.40951D+00,0.42256D+00,0.43631D+00,0.45060D+00,0.46545D+00,
     & 0.48088D+00,0.49693D+00,0.51361D+00,0.53095D+00,0.54975D+00,
     & 0.56777D+00,0.59421D+00,0.61364D+00,0.61822D+00,0.59899D+00,
     & 0.46556D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.79564D-02,
     & 0.13653D-01,0.21710D-01,0.26406D-01,0.33293D-01,0.37402D-01,
     & 0.41367D-01,0.47398D-01,0.51123D-01,0.54812D-01,0.58486D-01,
     & 0.62168D-01,0.65875D-01,0.69626D-01,0.73436D-01,0.77323D-01,
     & 0.81302D-01,0.85390D-01,0.89604D-01,0.91760D-01,0.96200D-01,
     & 0.10081D+00,0.10327D+00,0.10816D+00,0.11329D+00,0.11607D+00,
     & 0.12156D+00,0.12460D+00,0.13050D+00,0.13381D+00,0.14020D+00,
     & 0.14378D+00,0.14755D+00,0.15464D+00,0.15871D+00,0.16297D+00,
     & 0.17093D+00,0.17552D+00,0.18032D+00,0.18939D+00,0.19455D+00,
     & 0.19995D+00,0.20557D+00,0.21145D+00,0.22243D+00,0.22872D+00,
     & 0.23528D+00,0.24213D+00,0.24928D+00,0.25673D+00,0.27095D+00,
     & 0.27888D+00,0.28714D+00,0.29574D+00,0.30470D+00,0.31404D+00,
     & 0.32378D+00,0.33408D+00,0.34448D+00,0.35548D+00,0.37773D+00,
     & 0.38976D+00,0.40244D+00,0.41524D+00,0.42873D+00,0.44275D+00,
     & 0.45732D+00,0.47246D+00,0.48820D+00,0.50457D+00,0.52159D+00,
     & 0.54006D+00,0.55772D+00,0.58354D+00,0.60264D+00,0.60747D+00,
     & 0.58937D+00,0.45835D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.36312D-02,0.98146D-02,0.18386D-01,0.23340D-01,0.30562D-01,
     & 0.34837D-01,0.38939D-01,0.45165D-01,0.48978D-01,0.52739D-01,
     & 0.56474D-01,0.60203D-01,0.63949D-01,0.67729D-01,0.71561D-01,
     & 0.75461D-01,0.79448D-01,0.83538D-01,0.87749D-01,0.89865D-01,
     & 0.94294D-01,0.98889D-01,0.10130D+00,0.10617D+00,0.11127D+00,
     & 0.11400D+00,0.11945D+00,0.12243D+00,0.12829D+00,0.13153D+00,
     & 0.13788D+00,0.14139D+00,0.14508D+00,0.15212D+00,0.15611D+00,
     & 0.16028D+00,0.16818D+00,0.17267D+00,0.17738D+00,0.18638D+00,
     & 0.19144D+00,0.19672D+00,0.20224D+00,0.20800D+00,0.21889D+00,
     & 0.22505D+00,0.23149D+00,0.23821D+00,0.24522D+00,0.25253D+00,
     & 0.26662D+00,0.27439D+00,0.28249D+00,0.29093D+00,0.29972D+00,
     & 0.30888D+00,0.31843D+00,0.32855D+00,0.33874D+00,0.34953D+00,
     & 0.37160D+00,0.38341D+00,0.39586D+00,0.40842D+00,0.42166D+00,
     & 0.43542D+00,0.44972D+00,0.46459D+00,0.48004D+00,0.49612D+00,
     & 0.51283D+00,0.53100D+00,0.54832D+00,0.57355D+00,0.59233D+00,
     & 0.59740D+00,0.58040D+00,0.45158D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.56803D-02,0.14850D-01,0.20104D-01,
     & 0.27713D-01,0.32180D-01,0.36442D-01,0.42893D-01,0.46810D-01,
     & 0.50659D-01,0.54466D-01,0.58256D-01,0.62050D-01,0.65870D-01,
     & 0.69731D-01,0.73655D-01,0.77658D-01,0.81758D-01,0.85974D-01,
     & 0.88053D-01,0.92478D-01,0.97066D-01,0.99431D-01,0.10429D+00,
     & 0.10937D+00,0.11205D+00,0.11747D+00,0.12040D+00,0.12623D+00,
     & 0.12941D+00,0.13571D+00,0.13916D+00,0.14278D+00,0.14978D+00,
     & 0.15368D+00,0.15778D+00,0.16562D+00,0.17003D+00,0.17465D+00,
     & 0.18358D+00,0.18855D+00,0.19373D+00,0.19915D+00,0.20480D+00,
     & 0.21561D+00,0.22165D+00,0.22797D+00,0.23456D+00,0.24144D+00,
     & 0.24862D+00,0.26260D+00,0.27022D+00,0.27817D+00,0.28646D+00,
     & 0.29509D+00,0.30408D+00,0.31345D+00,0.32339D+00,0.33339D+00,
     & 0.34398D+00,0.36589D+00,0.37750D+00,0.38973D+00,0.40206D+00,
     & 0.41507D+00,0.42858D+00,0.44263D+00,0.45724D+00,0.47243D+00,
     & 0.48822D+00,0.50464D+00,0.52253D+00,0.53953D+00,0.56419D+00,
     & 0.58266D+00,0.58797D+00,0.57202D+00,0.44524D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.10789D-02,0.11057D-01,
     & 0.16660D-01,0.24715D-01,0.29405D-01,0.33855D-01,0.40563D-01,
     & 0.44604D-01,0.48556D-01,0.52449D-01,0.56311D-01,0.60166D-01,
     & 0.64034D-01,0.67938D-01,0.71894D-01,0.75922D-01,0.80040D-01,
     & 0.84270D-01,0.86314D-01,0.90745D-01,0.95331D-01,0.97656D-01,
     & 0.10251D+00,0.10757D+00,0.11021D+00,0.11562D+00,0.11848D+00,
     & 0.12429D+00,0.12741D+00,0.13369D+00,0.13708D+00,0.14063D+00,
     & 0.14759D+00,0.15142D+00,0.15544D+00,0.16324D+00,0.16757D+00,
     & 0.17211D+00,0.18099D+00,0.18586D+00,0.19095D+00,0.19627D+00,
     & 0.20183D+00,0.21256D+00,0.21850D+00,0.22470D+00,0.23118D+00,
     & 0.23793D+00,0.24498D+00,0.25887D+00,0.26636D+00,0.27417D+00,
     & 0.28230D+00,0.29078D+00,0.29961D+00,0.30882D+00,0.31858D+00,
     & 0.32840D+00,0.33881D+00,0.36059D+00,0.37200D+00,0.38403D+00,
     & 0.39614D+00,0.40893D+00,0.42221D+00,0.43602D+00,0.45038D+00,
     & 0.46531D+00,0.48084D+00,0.49699D+00,0.51461D+00,0.53130D+00,
     & 0.55543D+00,0.57361D+00,0.57915D+00,0.56422D+00,0.43928D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.69474D-02,0.12960D-01,0.21534D-01,0.26483D-01,0.31150D-01,
     & 0.38156D-01,0.42339D-01,0.46411D-01,0.50408D-01,0.54356D-01,
     & 0.58284D-01,0.62201D-01,0.66168D-01,0.70167D-01,0.74229D-01,
     & 0.78375D-01,0.82626D-01,0.84639D-01,0.89083D-01,0.93678D-01,
     & 0.95963D-01,0.10081D+00,0.10588D+00,0.10847D+00,0.11387D+00,
     & 0.11668D+00,0.12248D+00,0.12554D+00,0.13181D+00,0.13513D+00,
     & 0.13862D+00,0.14555D+00,0.14932D+00,0.15327D+00,0.16103D+00,
     & 0.16529D+00,0.16975D+00,0.17859D+00,0.18337D+00,0.18838D+00,
     & 0.19361D+00,0.19906D+00,0.20974D+00,0.21558D+00,0.22167D+00,
     & 0.22804D+00,0.23468D+00,0.24161D+00,0.25542D+00,0.26278D+00,
     & 0.27045D+00,0.27845D+00,0.28678D+00,0.29546D+00,0.30451D+00,
     & 0.31412D+00,0.32376D+00,0.33399D+00,0.35567D+00,0.36689D+00,
     & 0.37872D+00,0.39064D+00,0.40321D+00,0.41628D+00,0.42987D+00,
     & 0.44399D+00,0.45868D+00,0.47395D+00,0.48985D+00,0.50721D+00,
     & 0.52361D+00,0.54722D+00,0.56513D+00,0.57089D+00,0.55695D+00,
     & 0.43369D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.22874D-02,0.89514D-02,0.18128D-01,0.23378D-01,
     & 0.28298D-01,0.35645D-01,0.39995D-01,0.44208D-01,0.48324D-01,
     & 0.52375D-01,0.56390D-01,0.60394D-01,0.64410D-01,0.68462D-01,
     & 0.72569D-01,0.76753D-01,0.81033D-01,0.83019D-01,0.87483D-01,
     & 0.92095D-01,0.94345D-01,0.99206D-01,0.10428D+00,0.10682D+00,
     & 0.11222D+00,0.11499D+00,0.12078D+00,0.12379D+00,0.13004D+00,
     & 0.13331D+00,0.13674D+00,0.14366D+00,0.14736D+00,0.15125D+00,
     & 0.15899D+00,0.16317D+00,0.16755D+00,0.17637D+00,0.18107D+00,
     & 0.18599D+00,0.19113D+00,0.19650D+00,0.20714D+00,0.21287D+00,
     & 0.21887D+00,0.22513D+00,0.23166D+00,0.23848D+00,0.25224D+00,
     & 0.25948D+00,0.26702D+00,0.27488D+00,0.28307D+00,0.29161D+00,
     & 0.30051D+00,0.30997D+00,0.31945D+00,0.32952D+00,0.35111D+00,
     & 0.36215D+00,0.37381D+00,0.38553D+00,0.39790D+00,0.41076D+00,
     & 0.42413D+00,0.43804D+00,0.45249D+00,0.46753D+00,0.48318D+00,
     & 0.50030D+00,0.51642D+00,0.53954D+00,0.55719D+00,0.56318D+00,
     & 0.55019D+00,0.42845D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.45216D-02,0.14449D-01,
     & 0.20051D-01,0.25264D-01,0.33004D-01,0.37547D-01,0.41925D-01,
     & 0.46181D-01,0.50352D-01,0.54469D-01,0.58561D-01,0.62654D-01,
     & 0.66770D-01,0.70932D-01,0.75162D-01,0.79482D-01,0.81443D-01,
     & 0.85938D-01,0.90575D-01,0.92791D-01,0.97670D-01,0.10276D+00,
     & 0.10526D+00,0.11067D+00,0.11339D+00,0.11919D+00,0.12215D+00,
     & 0.12840D+00,0.13161D+00,0.13499D+00,0.14189D+00,0.14554D+00,
     & 0.14937D+00,0.15709D+00,0.16121D+00,0.16552D+00,0.17431D+00,
     & 0.17894D+00,0.18379D+00,0.18885D+00,0.19413D+00,0.20474D+00,
     & 0.21038D+00,0.21628D+00,0.22244D+00,0.22887D+00,0.23558D+00,
     & 0.24930D+00,0.25642D+00,0.26384D+00,0.27158D+00,0.27964D+00,
     & 0.28804D+00,0.29681D+00,0.30612D+00,0.31545D+00,0.32536D+00,
     & 0.34690D+00,0.35777D+00,0.36925D+00,0.38079D+00,0.39297D+00,
     & 0.40564D+00,0.41881D+00,0.43250D+00,0.44674D+00,0.46155D+00,
     & 0.47696D+00,0.49386D+00,0.50971D+00,0.53236D+00,0.54976D+00,
     & 0.55598D+00,0.54392D+00,0.42352D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.20707D-03,
     & 0.10441D-01,0.16453D-01,0.22009D-01,0.30203D-01,0.34970D-01,
     & 0.39539D-01,0.43958D-01,0.48268D-01,0.52505D-01,0.56700D-01,
     & 0.60882D-01,0.65076D-01,0.69305D-01,0.73592D-01,0.77963D-01,
     & 0.79902D-01,0.84437D-01,0.89108D-01,0.91293D-01,0.96199D-01,
     & 0.10131D+00,0.10377D+00,0.10920D+00,0.11188D+00,0.11769D+00,
     & 0.12060D+00,0.12686D+00,0.13002D+00,0.13335D+00,0.14026D+00,
     & 0.14385D+00,0.14762D+00,0.15534D+00,0.15939D+00,0.16364D+00,
     & 0.17242D+00,0.17698D+00,0.18175D+00,0.18673D+00,0.19194D+00,
     & 0.20253D+00,0.20809D+00,0.21390D+00,0.21996D+00,0.22630D+00,
     & 0.23291D+00,0.24660D+00,0.25361D+00,0.26092D+00,0.26854D+00,
     & 0.27648D+00,0.28475D+00,0.29338D+00,0.30256D+00,0.31174D+00,
     & 0.32150D+00,0.34301D+00,0.35372D+00,0.36505D+00,0.37640D+00,
     & 0.38841D+00,0.40089D+00,0.41387D+00,0.42736D+00,0.44139D+00,
     & 0.45599D+00,0.47117D+00,0.48787D+00,0.50345D+00,0.52565D+00,
     & 0.54282D+00,0.54926D+00,0.53810D+00,0.41889D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.60327D-02,0.12530D-01,0.18486D-01,0.27206D-01,
     & 0.32234D-01,0.37023D-01,0.41631D-01,0.46103D-01,0.50480D-01,
     & 0.54797D-01,0.59083D-01,0.63368D-01,0.67676D-01,0.72034D-01,
     & 0.76466D-01,0.78386D-01,0.82969D-01,0.87686D-01,0.89843D-01,
     & 0.94785D-01,0.99925D-01,0.10235D+00,0.10781D+00,0.11045D+00,
     & 0.11628D+00,0.11915D+00,0.12543D+00,0.12854D+00,0.13182D+00,
     & 0.13874D+00,0.14228D+00,0.14599D+00,0.15372D+00,0.15771D+00,
     & 0.16190D+00,0.17069D+00,0.17518D+00,0.17988D+00,0.18479D+00,
     & 0.18992D+00,0.20051D+00,0.20598D+00,0.21170D+00,0.21768D+00,
     & 0.22393D+00,0.23044D+00,0.24413D+00,0.25104D+00,0.25824D+00,
     & 0.26574D+00,0.27356D+00,0.28172D+00,0.29022D+00,0.29928D+00,
     & 0.30831D+00,0.31793D+00,0.33944D+00,0.35000D+00,0.36117D+00,
     & 0.37236D+00,0.38420D+00,0.39650D+00,0.40930D+00,0.42260D+00,
     & 0.43643D+00,0.45082D+00,0.46580D+00,0.48230D+00,0.49763D+00,
     & 0.51939D+00,0.53634D+00,0.54300D+00,0.53272D+00,0.41454D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.10626D-02,0.82110D-02,0.14639D-01,
     & 0.23970D-01,0.29301D-01,0.34348D-01,0.39175D-01,0.43835D-01,
     & 0.48374D-01,0.52831D-01,0.57240D-01,0.61631D-01,0.66035D-01,
     & 0.70475D-01,0.74980D-01,0.76884D-01,0.81529D-01,0.86299D-01,
     & 0.88432D-01,0.93419D-01,0.98601D-01,0.10100D+00,0.10648D+00,
     & 0.10909D+00,0.11495D+00,0.11778D+00,0.12409D+00,0.12716D+00,
     & 0.13039D+00,0.13733D+00,0.14082D+00,0.14448D+00,0.15222D+00,
     & 0.15616D+00,0.16029D+00,0.16909D+00,0.17352D+00,0.17815D+00,
     & 0.18300D+00,0.18805D+00,0.19866D+00,0.20406D+00,0.20970D+00,
     & 0.21560D+00,0.22175D+00,0.22817D+00,0.24188D+00,0.24869D+00,
     & 0.25578D+00,0.26318D+00,0.27089D+00,0.27893D+00,0.28731D+00,
     & 0.29625D+00,0.30515D+00,0.31464D+00,0.33617D+00,0.34659D+00,
     & 0.35761D+00,0.36864D+00,0.38032D+00,0.39246D+00,0.40508D+00,
     & 0.41820D+00,0.43184D+00,0.44604D+00,0.46082D+00,0.47714D+00,
     & 0.49222D+00,0.51357D+00,0.53031D+00,0.53718D+00,0.52776D+00,
     & 0.41046D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.32495D-02,
     & 0.10399D-01,0.20447D-01,0.26131D-01,0.31477D-01,0.36558D-01,
     & 0.41437D-01,0.46164D-01,0.50785D-01,0.55336D-01,0.59852D-01,
     & 0.64364D-01,0.68902D-01,0.73493D-01,0.75385D-01,0.80103D-01,
     & 0.84939D-01,0.87049D-01,0.92093D-01,0.97326D-01,0.99692D-01,
     & 0.10523D+00,0.10780D+00,0.11370D+00,0.11649D+00,0.12283D+00,
     & 0.12586D+00,0.12905D+00,0.13602D+00,0.13946D+00,0.14308D+00,
     & 0.15085D+00,0.15473D+00,0.15881D+00,0.16764D+00,0.17201D+00,
     & 0.17658D+00,0.18136D+00,0.18635D+00,0.19698D+00,0.20230D+00,
     & 0.20787D+00,0.21369D+00,0.21976D+00,0.22610D+00,0.23983D+00,
     & 0.24655D+00,0.25354D+00,0.26084D+00,0.26844D+00,0.27638D+00,
     & 0.28465D+00,0.29348D+00,0.30225D+00,0.31161D+00,0.33319D+00,
     & 0.34347D+00,0.35436D+00,0.36524D+00,0.37677D+00,0.38875D+00,
     & 0.40120D+00,0.41415D+00,0.42761D+00,0.44163D+00,0.45621D+00,
     & 0.47236D+00,0.48721D+00,0.50816D+00,0.52470D+00,0.53179D+00,
     & 0.52321D+00,0.40662D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.56547D-02,0.16574D-01,0.22673D-01,0.28368D-01,
     & 0.33746D-01,0.38879D-01,0.43824D-01,0.48633D-01,0.53350D-01,
     & 0.58009D-01,0.62650D-01,0.67300D-01,0.71993D-01,0.73876D-01,
     & 0.78680D-01,0.83594D-01,0.85685D-01,0.90797D-01,0.96093D-01,
     & 0.98431D-01,0.10402D+00,0.10656D+00,0.11251D+00,0.11527D+00,
     & 0.12166D+00,0.12465D+00,0.12780D+00,0.13481D+00,0.13821D+00,
     & 0.14177D+00,0.14959D+00,0.15342D+00,0.15744D+00,0.16632D+00,
     & 0.17063D+00,0.17514D+00,0.17986D+00,0.18479D+00,0.19546D+00,
     & 0.20071D+00,0.20621D+00,0.21195D+00,0.21795D+00,0.22421D+00,
     & 0.23799D+00,0.24461D+00,0.25152D+00,0.25872D+00,0.26623D+00,
     & 0.27405D+00,0.28222D+00,0.29094D+00,0.29959D+00,0.30883D+00,
     & 0.33049D+00,0.34064D+00,0.35141D+00,0.36214D+00,0.37352D+00,
     & 0.38535D+00,0.39765D+00,0.41043D+00,0.42373D+00,0.43756D+00,
     & 0.45197D+00,0.46796D+00,0.48258D+00,0.50314D+00,0.51949D+00,
     & 0.52680D+00,0.51904D+00,0.40301D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.58134D-03,0.12280D-01,0.18868D-01,
     & 0.24972D-01,0.30698D-01,0.36125D-01,0.41323D-01,0.46351D-01,
     & 0.51259D-01,0.56086D-01,0.60874D-01,0.65656D-01,0.70466D-01,
     & 0.72345D-01,0.77250D-01,0.82255D-01,0.84330D-01,0.89522D-01,
     & 0.94892D-01,0.97206D-01,0.10286D+00,0.10538D+00,0.11139D+00,
     & 0.11411D+00,0.12056D+00,0.12352D+00,0.12662D+00,0.13369D+00,
     & 0.13705D+00,0.14057D+00,0.14844D+00,0.15223D+00,0.15620D+00,
     & 0.16513D+00,0.16938D+00,0.17384D+00,0.17850D+00,0.18337D+00,
     & 0.19410D+00,0.19928D+00,0.20471D+00,0.21039D+00,0.21631D+00,
     & 0.22250D+00,0.23634D+00,0.24288D+00,0.24970D+00,0.25681D+00,
     & 0.26422D+00,0.27195D+00,0.28001D+00,0.28864D+00,0.29717D+00,
     & 0.30630D+00,0.32806D+00,0.33809D+00,0.34874D+00,0.35933D+00,
     & 0.37058D+00,0.38226D+00,0.39441D+00,0.40704D+00,0.42018D+00,
     & 0.43384D+00,0.44807D+00,0.46393D+00,0.47832D+00,0.49851D+00,
     & 0.51468D+00,0.52221D+00,0.51525D+00,0.39962D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.74723D-02,
     & 0.14640D-01,0.21228D-01,0.27359D-01,0.33131D-01,0.38625D-01,
     & 0.43908D-01,0.49037D-01,0.54059D-01,0.59017D-01,0.63951D-01,
     & 0.68896D-01,0.70776D-01,0.75798D-01,0.80909D-01,0.82971D-01,
     & 0.88257D-01,0.93717D-01,0.96006D-01,0.10174D+00,0.10423D+00,
     & 0.11032D+00,0.11301D+00,0.11953D+00,0.12245D+00,0.12552D+00,
     & 0.13266D+00,0.13597D+00,0.13946D+00,0.14739D+00,0.15113D+00,
     & 0.15506D+00,0.16405D+00,0.16826D+00,0.17266D+00,0.17727D+00,
     & 0.18209D+00,0.19288D+00,0.19801D+00,0.20337D+00,0.20898D+00,
     & 0.21484D+00,0.22095D+00,0.23488D+00,0.24134D+00,0.24807D+00,
     & 0.25510D+00,0.26242D+00,0.27006D+00,0.27802D+00,0.28656D+00,
     & 0.29498D+00,0.30400D+00,0.32589D+00,0.33581D+00,0.34635D+00,
     & 0.35681D+00,0.36792D+00,0.37947D+00,0.39148D+00,0.40396D+00,
     & 0.41695D+00,0.43045D+00,0.44452D+00,0.46024D+00,0.47441D+00,
     & 0.49425D+00,0.51025D+00,0.51799D+00,0.51183D+00,0.39643D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.18253D-02,0.98999D-02,0.17061D-01,0.23671D-01,0.29848D-01,
     & 0.35688D-01,0.41268D-01,0.46653D-01,0.51900D-01,0.57056D-01,
     & 0.62165D-01,0.67267D-01,0.69153D-01,0.74309D-01,0.79544D-01,
     & 0.81597D-01,0.86991D-01,0.92552D-01,0.94823D-01,0.10065D+00,
     & 0.10312D+00,0.10929D+00,0.11196D+00,0.11856D+00,0.12144D+00,
     & 0.12448D+00,0.13170D+00,0.13498D+00,0.13842D+00,0.14643D+00,
     & 0.15013D+00,0.15401D+00,0.16309D+00,0.16725D+00,0.17161D+00,
     & 0.17617D+00,0.18093D+00,0.19181D+00,0.19688D+00,0.20218D+00,
     & 0.20773D+00,0.21352D+00,0.21957D+00,0.23360D+00,0.23999D+00,
     & 0.24664D+00,0.25358D+00,0.26082D+00,0.26837D+00,0.27625D+00,
     & 0.28470D+00,0.29301D+00,0.30193D+00,0.32399D+00,0.33380D+00,
     & 0.34423D+00,0.35456D+00,0.36555D+00,0.37698D+00,0.38885D+00,
     & 0.40119D+00,0.41403D+00,0.42739D+00,0.44129D+00,0.45690D+00,
     & 0.47086D+00,0.49035D+00,0.50618D+00,0.51414D+00,0.50875D+00,
     & 0.39343D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.43712D-02,0.12380D-01,0.19560D-01,
     & 0.26215D-01,0.32459D-01,0.38386D-01,0.44070D-01,0.49575D-01,
     & 0.54962D-01,0.60274D-01,0.65558D-01,0.67456D-01,0.72767D-01,
     & 0.78144D-01,0.80192D-01,0.85712D-01,0.91391D-01,0.93645D-01,
     & 0.99584D-01,0.10202D+00,0.10830D+00,0.11094D+00,0.11764D+00,
     & 0.12049D+00,0.12350D+00,0.13081D+00,0.13406D+00,0.13746D+00,
     & 0.14557D+00,0.14923D+00,0.15307D+00,0.16225D+00,0.16636D+00,
     & 0.17067D+00,0.17518D+00,0.17989D+00,0.19088D+00,0.19589D+00,
     & 0.20114D+00,0.20662D+00,0.21236D+00,0.21834D+00,0.23250D+00,
     & 0.23882D+00,0.24540D+00,0.25226D+00,0.25942D+00,0.26689D+00,
     & 0.27468D+00,0.28305D+00,0.29126D+00,0.30008D+00,0.32233D+00,
     & 0.33204D+00,0.34238D+00,0.35259D+00,0.36347D+00,0.37477D+00,
     & 0.38651D+00,0.39873D+00,0.41142D+00,0.42463D+00,0.43839D+00,
     & 0.45390D+00,0.46764D+00,0.48680D+00,0.50247D+00,0.51066D+00,
     & 0.50603D+00,0.39061D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.70485D-02,
     & 0.14931D-01,0.22154D-01,0.28877D-01,0.35211D-01,0.41245D-01,
     & 0.47055D-01,0.52706D-01,0.58252D-01,0.63746D-01,0.65662D-01,
     & 0.71153D-01,0.76694D-01,0.78740D-01,0.84406D-01,0.90220D-01,
     & 0.92461D-01,0.98523D-01,0.10095D+00,0.10734D+00,0.10995D+00,
     & 0.11676D+00,0.11959D+00,0.12257D+00,0.12999D+00,0.13320D+00,
     & 0.13657D+00,0.14479D+00,0.14841D+00,0.15222D+00,0.16150D+00,
     & 0.16557D+00,0.16984D+00,0.17431D+00,0.17897D+00,0.19008D+00,
     & 0.19504D+00,0.20023D+00,0.20567D+00,0.21135D+00,0.21727D+00,
     & 0.23158D+00,0.23782D+00,0.24434D+00,0.25113D+00,0.25821D+00,
     & 0.26560D+00,0.27331D+00,0.28161D+00,0.28972D+00,0.29845D+00,
     & 0.32093D+00,0.33055D+00,0.34079D+00,0.35089D+00,0.36165D+00,
     & 0.37284D+00,0.38446D+00,0.39655D+00,0.40911D+00,0.42219D+00,
     & 0.43581D+00,0.45122D+00,0.46475D+00,0.48359D+00,0.49912D+00,
     & 0.50752D+00,0.50364D+00,0.38795D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.10153D-02,0.96664D-02,0.17572D-01,0.24864D-01,0.31679D-01,
     & 0.38124D-01,0.44289D-01,0.50248D-01,0.56068D-01,0.61803D-01,
     & 0.63746D-01,0.69444D-01,0.75173D-01,0.77224D-01,0.83056D-01,
     & 0.89026D-01,0.91259D-01,0.97462D-01,0.99867D-01,0.10639D+00,
     & 0.10899D+00,0.11593D+00,0.11873D+00,0.12168D+00,0.12923D+00,
     & 0.13240D+00,0.13575D+00,0.14409D+00,0.14767D+00,0.15144D+00,
     & 0.16086D+00,0.16489D+00,0.16912D+00,0.17354D+00,0.17817D+00,
     & 0.18942D+00,0.19433D+00,0.19947D+00,0.20485D+00,0.21048D+00,
     & 0.21635D+00,0.23083D+00,0.23701D+00,0.24345D+00,0.25018D+00,
     & 0.25719D+00,0.26450D+00,0.27213D+00,0.28037D+00,0.28838D+00,
     & 0.29702D+00,0.31977D+00,0.32930D+00,0.33946D+00,0.34945D+00,
     & 0.36011D+00,0.37118D+00,0.38269D+00,0.39466D+00,0.40710D+00,
     & 0.42005D+00,0.43353D+00,0.44887D+00,0.46219D+00,0.48072D+00,
     & 0.49610D+00,0.50473D+00,0.50159D+00,0.38544D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.33428D-02,0.12347D-01,0.20323D-01,
     & 0.27711D-01,0.34642D-01,0.41224D-01,0.47546D-01,0.53681D-01,
     & 0.59696D-01,0.61676D-01,0.67613D-01,0.73559D-01,0.75621D-01,
     & 0.81644D-01,0.87792D-01,0.90016D-01,0.96385D-01,0.98777D-01,
     & 0.10545D+00,0.10803D+00,0.11512D+00,0.11790D+00,0.12083D+00,
     & 0.12851D+00,0.13166D+00,0.13498D+00,0.14346D+00,0.14701D+00,
     & 0.15075D+00,0.16032D+00,0.16431D+00,0.16850D+00,0.17288D+00,
     & 0.17747D+00,0.18888D+00,0.19375D+00,0.19884D+00,0.20417D+00,
     & 0.20975D+00,0.21557D+00,0.23024D+00,0.23637D+00,0.24275D+00,
     & 0.24941D+00,0.25635D+00,0.26360D+00,0.27115D+00,0.27933D+00,
     & 0.28725D+00,0.29581D+00,0.31886D+00,0.32831D+00,0.33839D+00,
     & 0.34827D+00,0.35883D+00,0.36980D+00,0.38121D+00,0.39306D+00,
     & 0.40539D+00,0.41821D+00,0.43156D+00,0.44684D+00,0.45995D+00,
     & 0.47818D+00,0.49343D+00,0.50228D+00,0.49986D+00,0.38307D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.62023D-02,
     & 0.15131D-01,0.23208D-01,0.30719D-01,0.37794D-01,0.44540D-01,
     & 0.51046D-01,0.57388D-01,0.59416D-01,0.65630D-01,0.71825D-01,
     & 0.73905D-01,0.80147D-01,0.86500D-01,0.88722D-01,0.95279D-01,
     & 0.97661D-01,0.10451D+00,0.10707D+00,0.11433D+00,0.11709D+00,
     & 0.11999D+00,0.12784D+00,0.13097D+00,0.13426D+00,0.14290D+00,
     & 0.14642D+00,0.15013D+00,0.15987D+00,0.16383D+00,0.16798D+00,
     & 0.17233D+00,0.17687D+00,0.18847D+00,0.19329D+00,0.19834D+00,
     & 0.20363D+00,0.20916D+00,0.21493D+00,0.22983D+00,0.23590D+00,
     & 0.24222D+00,0.24881D+00,0.25569D+00,0.26287D+00,0.27036D+00,
     & 0.27849D+00,0.28631D+00,0.29480D+00,0.31820D+00,0.32756D+00,
     & 0.33758D+00,0.34736D+00,0.35783D+00,0.36870D+00,0.38000D+00,
     & 0.39175D+00,0.40396D+00,0.41667D+00,0.42990D+00,0.44514D+00,
     & 0.45803D+00,0.47597D+00,0.49110D+00,0.50016D+00,0.49847D+00,
     & 0.38083D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.22364D-03,0.91255D-02,0.18044D-01,0.26252D-01,0.33916D-01,
     & 0.41165D-01,0.48106D-01,0.54829D-01,0.56920D-01,0.63455D-01,
     & 0.69940D-01,0.72045D-01,0.78541D-01,0.85128D-01,0.87354D-01,
     & 0.94124D-01,0.96501D-01,0.10355D+00,0.10610D+00,0.11355D+00,
     & 0.11628D+00,0.11917D+00,0.12720D+00,0.13031D+00,0.13357D+00,
     & 0.14240D+00,0.14590D+00,0.14957D+00,0.15950D+00,0.16343D+00,
     & 0.16755D+00,0.17186D+00,0.17638D+00,0.18818D+00,0.19296D+00,
     & 0.19797D+00,0.20322D+00,0.20870D+00,0.21443D+00,0.22959D+00,
     & 0.23560D+00,0.24186D+00,0.24840D+00,0.25522D+00,0.26234D+00,
     & 0.26976D+00,0.27784D+00,0.28558D+00,0.29399D+00,0.31778D+00,
     & 0.32707D+00,0.33703D+00,0.34671D+00,0.35709D+00,0.36788D+00,
     & 0.37908D+00,0.39073D+00,0.40283D+00,0.41543D+00,0.42854D+00,
     & 0.44376D+00,0.45643D+00,0.47409D+00,0.48910D+00,0.49839D+00,
     & 0.49741D+00,0.37870D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.19086D-02,0.12056D-01,0.21111D-01,
     & 0.29484D-01,0.37333D-01,0.44790D-01,0.51962D-01,0.54133D-01,
     & 0.61044D-01,0.67865D-01,0.70006D-01,0.76794D-01,0.83652D-01,
     & 0.85886D-01,0.92901D-01,0.95277D-01,0.10255D+00,0.10509D+00,
     & 0.11276D+00,0.11548D+00,0.11835D+00,0.12659D+00,0.12967D+00,
     & 0.13292D+00,0.14195D+00,0.14542D+00,0.14907D+00,0.15923D+00,
     & 0.16312D+00,0.16720D+00,0.17149D+00,0.17597D+00,0.18801D+00,
     & 0.19275D+00,0.19772D+00,0.20293D+00,0.20838D+00,0.21406D+00,
     & 0.22951D+00,0.23547D+00,0.24168D+00,0.24816D+00,0.25493D+00,
     & 0.26198D+00,0.26935D+00,0.27739D+00,0.28503D+00,0.29338D+00,
     & 0.31762D+00,0.32684D+00,0.33674D+00,0.34632D+00,0.35662D+00,
     & 0.36732D+00,0.37844D+00,0.38999D+00,0.40200D+00,0.41449D+00,
     & 0.42750D+00,0.44271D+00,0.45515D+00,0.47254D+00,0.48743D+00,
     & 0.49696D+00,0.49667D+00,0.37668D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.47183D-02,
     & 0.15129D-01,0.24364D-01,0.32938D-01,0.41010D-01,0.48714D-01,
     & 0.50986D-01,0.58338D-01,0.65551D-01,0.67741D-01,0.74869D-01,
     & 0.82038D-01,0.84289D-01,0.91585D-01,0.93965D-01,0.10150D+00,
     & 0.10404D+00,0.11195D+00,0.11466D+00,0.11752D+00,0.12599D+00,
     & 0.12905D+00,0.13228D+00,0.14155D+00,0.14500D+00,0.14862D+00,
     & 0.15902D+00,0.16288D+00,0.16694D+00,0.17120D+00,0.17565D+00,
     & 0.18796D+00,0.19267D+00,0.19760D+00,0.20277D+00,0.20818D+00,
     & 0.21383D+00,0.22960D+00,0.23551D+00,0.24167D+00,0.24810D+00,
     & 0.25481D+00,0.26181D+00,0.26912D+00,0.27713D+00,0.28469D+00,
     & 0.29297D+00,0.31770D+00,0.32686D+00,0.33671D+00,0.34621D+00,
     & 0.35644D+00,0.36705D+00,0.37808D+00,0.38955D+00,0.40146D+00,
     & 0.41385D+00,0.42676D+00,0.44199D+00,0.45419D+00,0.47132D+00,
     & 0.48611D+00,0.49586D+00,0.49628D+00,0.37474D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.79489D-02,0.18382D-01,0.27838D-01,0.36654D-01,
     & 0.44993D-01,0.47392D-01,0.55266D-01,0.62941D-01,0.65194D-01,
     & 0.72718D-01,0.80248D-01,0.82525D-01,0.90145D-01,0.92536D-01,
     & 0.10037D+00,0.10291D+00,0.11110D+00,0.11380D+00,0.11665D+00,
     & 0.12539D+00,0.12844D+00,0.13165D+00,0.14118D+00,0.14461D+00,
     & 0.14821D+00,0.15889D+00,0.16273D+00,0.16676D+00,0.17099D+00,
     & 0.17541D+00,0.18802D+00,0.19270D+00,0.19760D+00,0.20273D+00,
     & 0.20811D+00,0.21372D+00,0.22985D+00,0.23572D+00,0.24183D+00,
     & 0.24822D+00,0.25488D+00,0.26183D+00,0.26908D+00,0.27707D+00,
     & 0.28454D+00,0.29276D+00,0.31805D+00,0.32715D+00,0.33696D+00,
     & 0.34637D+00,0.35653D+00,0.36707D+00,0.37802D+00,0.38940D+00,
     & 0.40122D+00,0.41352D+00,0.42634D+00,0.44162D+00,0.45357D+00,
     & 0.47044D+00,0.48513D+00,0.49512D+00,0.49621D+00,0.37288D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.49151D-03,0.11297D-01,0.21851D-01,
     & 0.31574D-01,0.40681D-01,0.43241D-01,0.51739D-01,0.59961D-01,
     & 0.62296D-01,0.70285D-01,0.78236D-01,0.80550D-01,0.88544D-01,
     & 0.90954D-01,0.99132D-01,0.10167D+00,0.11018D+00,0.11288D+00,
     & 0.11573D+00,0.12477D+00,0.12781D+00,0.13101D+00,0.14084D+00,
     & 0.14425D+00,0.14783D+00,0.15882D+00,0.16264D+00,0.16664D+00,
     & 0.17085D+00,0.17525D+00,0.18820D+00,0.19284D+00,0.19771D+00,
     & 0.20281D+00,0.20816D+00,0.21373D+00,0.23028D+00,0.23610D+00,
     & 0.24217D+00,0.24851D+00,0.25512D+00,0.26202D+00,0.26923D+00,
     & 0.27721D+00,0.28458D+00,0.29275D+00,0.31866D+00,0.32771D+00,
     & 0.33749D+00,0.34681D+00,0.35690D+00,0.36738D+00,0.37826D+00,
     & 0.38956D+00,0.40130D+00,0.41351D+00,0.42623D+00,0.44161D+00,
     & 0.45328D+00,0.46990D+00,0.48450D+00,0.49473D+00,0.49650D+00,
     & 0.37109D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.25188D-02,
     & 0.14733D-01,0.25581D-01,0.35626D-01,0.38391D-01,0.47642D-01,
     & 0.56516D-01,0.58958D-01,0.67496D-01,0.75942D-01,0.78306D-01,
     & 0.86736D-01,0.89175D-01,0.97748D-01,0.10030D+00,0.10918D+00,
     & 0.11188D+00,0.11473D+00,0.12412D+00,0.12715D+00,0.13033D+00,
     & 0.14051D+00,0.14390D+00,0.14747D+00,0.15882D+00,0.16261D+00,
     & 0.16659D+00,0.17077D+00,0.17515D+00,0.18848D+00,0.19309D+00,
     & 0.19794D+00,0.20301D+00,0.20832D+00,0.21388D+00,0.23088D+00,
     & 0.23666D+00,0.24269D+00,0.24898D+00,0.25555D+00,0.26241D+00,
     & 0.26957D+00,0.27754D+00,0.28483D+00,0.29294D+00,0.31956D+00,
     & 0.32856D+00,0.33832D+00,0.34755D+00,0.35758D+00,0.36799D+00,
     & 0.37880D+00,0.39003D+00,0.40169D+00,0.41383D+00,0.42647D+00,
     & 0.44197D+00,0.45333D+00,0.46972D+00,0.48423D+00,0.49470D+00,
     & 0.49715D+00,0.36934D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.57377D-02,0.18417D-01,0.29627D-01,0.32651D-01,
     & 0.42823D-01,0.52486D-01,0.55064D-01,0.64259D-01,0.73290D-01,
     & 0.75721D-01,0.84663D-01,0.87142D-01,0.96175D-01,0.98751D-01,
     & 0.10806D+00,0.11076D+00,0.11361D+00,0.12341D+00,0.12643D+00,
     & 0.12961D+00,0.14017D+00,0.14355D+00,0.14711D+00,0.15886D+00,
     & 0.16262D+00,0.16659D+00,0.17075D+00,0.17511D+00,0.18888D+00,
     & 0.19346D+00,0.19827D+00,0.20332D+00,0.20861D+00,0.21414D+00,
     & 0.23166D+00,0.23740D+00,0.24339D+00,0.24964D+00,0.25617D+00,
     & 0.26299D+00,0.27010D+00,0.27809D+00,0.28527D+00,0.29334D+00,
     & 0.32073D+00,0.32969D+00,0.33944D+00,0.34859D+00,0.35857D+00,
     & 0.36892D+00,0.37967D+00,0.39083D+00,0.40242D+00,0.41448D+00,
     & 0.42704D+00,0.44273D+00,0.45375D+00,0.46991D+00,0.48433D+00,
     & 0.49505D+00,0.49816D+00,0.36762D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.95085D-02,0.22410D-01,
     & 0.25769D-01,0.37083D-01,0.47711D-01,0.50465D-01,0.60454D-01,
     & 0.70183D-01,0.72704D-01,0.82250D-01,0.84785D-01,0.94358D-01,
     & 0.96964D-01,0.10677D+00,0.10949D+00,0.11235D+00,0.12260D+00,
     & 0.12562D+00,0.12880D+00,0.13981D+00,0.14318D+00,0.14673D+00,
     & 0.15893D+00,0.16268D+00,0.16663D+00,0.17077D+00,0.17512D+00,
     & 0.18937D+00,0.19392D+00,0.19871D+00,0.20374D+00,0.20901D+00,
     & 0.21452D+00,0.23261D+00,0.23832D+00,0.24427D+00,0.25049D+00,
     & 0.25697D+00,0.26375D+00,0.27083D+00,0.27885D+00,0.28592D+00,
     & 0.29395D+00,0.32221D+00,0.33114D+00,0.34089D+00,0.34995D+00,
     & 0.35988D+00,0.37018D+00,0.38087D+00,0.39197D+00,0.40350D+00,
     & 0.41549D+00,0.42798D+00,0.44392D+00,0.45454D+00,0.47048D+00,
     & 0.48483D+00,0.49580D+00,0.49957D+00,0.36591D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.41980D-03,
     & 0.13562D-01,0.17392D-01,0.30152D-01,0.41981D-01,0.44962D-01,
     & 0.55921D-01,0.66495D-01,0.69133D-01,0.79401D-01,0.82012D-01,
     & 0.92220D-01,0.94873D-01,0.10525D+00,0.10800D+00,0.11088D+00,
     & 0.12166D+00,0.12469D+00,0.12788D+00,0.13940D+00,0.14276D+00,
     & 0.14630D+00,0.15902D+00,0.16276D+00,0.16669D+00,0.17082D+00,
     & 0.17515D+00,0.18996D+00,0.19449D+00,0.19926D+00,0.20426D+00,
     & 0.20951D+00,0.21500D+00,0.23375D+00,0.23942D+00,0.24534D+00,
     & 0.25152D+00,0.25798D+00,0.26472D+00,0.27176D+00,0.27982D+00,
     & 0.28677D+00,0.29476D+00,0.32402D+00,0.33291D+00,0.34267D+00,
     & 0.35164D+00,0.36153D+00,0.37178D+00,0.38243D+00,0.39347D+00,
     & 0.40494D+00,0.41687D+00,0.42930D+00,0.44560D+00,0.45572D+00,
     & 0.47145D+00,0.48573D+00,0.49696D+00,0.50138D+00,0.36419D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.25041D-02,0.64656D-02,0.21654D-01,0.35004D-01,
     & 0.38282D-01,0.50447D-01,0.62057D-01,0.64850D-01,0.75988D-01,
     & 0.78702D-01,0.89667D-01,0.92385D-01,0.10345D+00,0.10622D+00,
     & 0.10914D+00,0.12054D+00,0.12358D+00,0.12678D+00,0.13890D+00,
     & 0.14226D+00,0.14580D+00,0.15911D+00,0.16284D+00,0.16676D+00,
     & 0.17088D+00,0.17520D+00,0.19063D+00,0.19514D+00,0.19989D+00,
     & 0.20488D+00,0.21011D+00,0.21558D+00,0.23508D+00,0.24072D+00,
     & 0.24660D+00,0.25275D+00,0.25917D+00,0.26588D+00,0.27289D+00,
     & 0.28103D+00,0.28784D+00,0.29580D+00,0.32616D+00,0.33502D+00,
     & 0.34481D+00,0.35370D+00,0.36355D+00,0.37376D+00,0.38436D+00,
     & 0.39536D+00,0.40677D+00,0.41864D+00,0.43102D+00,0.44783D+00,
     & 0.45732D+00,0.47285D+00,0.48707D+00,0.49856D+00,0.50362D+00,
     & 0.36243D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.10670D-01,
     & 0.26377D-01,0.30047D-01,0.43741D-01,0.56639D-01,0.59637D-01,
     & 0.71843D-01,0.74693D-01,0.86572D-01,0.89378D-01,0.10125D+00,
     & 0.10408D+00,0.10705D+00,0.11918D+00,0.12224D+00,0.12547D+00,
     & 0.13826D+00,0.14163D+00,0.14517D+00,0.15917D+00,0.16289D+00,
     & 0.16681D+00,0.17092D+00,0.17524D+00,0.19138D+00,0.19588D+00,
     & 0.20061D+00,0.20559D+00,0.21080D+00,0.21626D+00,0.23660D+00,
     & 0.24221D+00,0.24806D+00,0.25418D+00,0.26057D+00,0.26726D+00,
     & 0.27424D+00,0.28249D+00,0.28913D+00,0.29706D+00,0.32866D+00,
     & 0.33751D+00,0.34735D+00,0.35614D+00,0.36595D+00,0.37613D+00,
     & 0.38669D+00,0.39765D+00,0.40901D+00,0.42084D+00,0.43317D+00,
     & 0.45075D+00,0.45938D+00,0.47472D+00,0.48888D+00,0.50064D+00,
     & 0.50632D+00,0.36061D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.14007D-03,0.15368D-01,0.19709D-01,0.35391D-01,0.49927D-01,
     & 0.53198D-01,0.66733D-01,0.69766D-01,0.82765D-01,0.85690D-01,
     & 0.98541D-01,0.10145D+00,0.10448D+00,0.11748D+00,0.12058D+00,
     & 0.12383D+00,0.13744D+00,0.14082D+00,0.14438D+00,0.15917D+00,
     & 0.16289D+00,0.16680D+00,0.17092D+00,0.17523D+00,0.19220D+00,
     & 0.19668D+00,0.20140D+00,0.20636D+00,0.21157D+00,0.21702D+00,
     & 0.23832D+00,0.24390D+00,0.24972D+00,0.25581D+00,0.26218D+00,
     & 0.26884D+00,0.27579D+00,0.28423D+00,0.29063D+00,0.29854D+00,
     & 0.33156D+00,0.34040D+00,0.35031D+00,0.35899D+00,0.36878D+00,
     & 0.37893D+00,0.38946D+00,0.40038D+00,0.41171D+00,0.42349D+00,
     & 0.43579D+00,0.45457D+00,0.46195D+00,0.47712D+00,0.49123D+00,
     & 0.50326D+00,0.50954D+00,0.35869D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.19845D-02,0.57088D-02,0.24801D-01,
     & 0.41470D-01,0.45111D-01,0.60335D-01,0.63615D-01,0.78007D-01,
     & 0.81097D-01,0.95144D-01,0.98159D-01,0.10129D+00,0.11532D+00,
     & 0.11848D+00,0.12178D+00,0.13634D+00,0.13975D+00,0.14333D+00,
     & 0.15906D+00,0.16277D+00,0.16669D+00,0.17081D+00,0.17514D+00,
     & 0.19306D+00,0.19752D+00,0.20223D+00,0.20719D+00,0.21239D+00,
     & 0.21784D+00,0.24024D+00,0.24579D+00,0.25159D+00,0.25766D+00,
     & 0.26400D+00,0.27063D+00,0.27757D+00,0.28629D+00,0.29237D+00,
     & 0.30025D+00,0.33489D+00,0.34372D+00,0.35374D+00,0.36229D+00,
     & 0.37207D+00,0.38220D+00,0.39270D+00,0.40359D+00,0.41488D+00,
     & 0.42663D+00,0.43890D+00,0.45930D+00,0.46503D+00,0.48004D+00,
     & 0.49411D+00,0.50642D+00,0.51330D+00,0.35664D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.10300D-01,0.30612D-01,0.34757D-01,0.52184D-01,0.55801D-01,
     & 0.71966D-01,0.75281D-01,0.90817D-01,0.93980D-01,0.97236D-01,
     & 0.11254D+00,0.11578D+00,0.11916D+00,0.13487D+00,0.13831D+00,
     & 0.14193D+00,0.15876D+00,0.16249D+00,0.16642D+00,0.17055D+00,
     & 0.17489D+00,0.19392D+00,0.19838D+00,0.20309D+00,0.20804D+00,
     & 0.21325D+00,0.21870D+00,0.24236D+00,0.24789D+00,0.25367D+00,
     & 0.25971D+00,0.26603D+00,0.27265D+00,0.27956D+00,0.28880D+00,
     & 0.29433D+00,0.30219D+00,0.33869D+00,0.34752D+00,0.35769D+00,
     & 0.36608D+00,0.37585D+00,0.38596D+00,0.39645D+00,0.40732D+00,
     & 0.41859D+00,0.43031D+00,0.44255D+00,0.47162D+00,0.46861D+00,
     & 0.48345D+00,0.49750D+00,0.51012D+00,0.51761D+00,0.35439D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.15797D-01,0.21016D-01,0.41595D-01,
     & 0.45679D-01,0.64158D-01,0.67785D-01,0.85218D-01,0.88586D-01,
     & 0.92024D-01,0.10891D+00,0.11227D+00,0.11575D+00,0.13285D+00,
     & 0.13636D+00,0.14004D+00,0.15818D+00,0.16194D+00,0.16589D+00,
     & 0.17005D+00,0.17441D+00,0.19476D+00,0.19921D+00,0.20392D+00,
     & 0.20888D+00,0.21408D+00,0.21955D+00,0.24469D+00,0.25020D+00,
     & 0.25596D+00,0.26198D+00,0.26828D+00,0.27488D+00,0.28178D+00,
     & 0.29231D+00,0.29652D+00,0.30437D+00,0.34301D+00,0.35185D+00,
     & 0.36224D+00,0.37042D+00,0.38018D+00,0.39030D+00,0.40077D+00,
     & 0.41163D+00,0.42290D+00,0.43459D+00,0.44679D+00,0.48303D+00,
     & 0.47269D+00,0.48737D+00,0.50141D+00,0.51439D+00,0.52250D+00,
     & 0.35191D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.44230D-03,0.31822D-02,
     & 0.27490D-01,0.32261D-01,0.53861D-01,0.57928D-01,0.77837D-01,
     & 0.81497D-01,0.85192D-01,0.10408D+00,0.10761D+00,0.11124D+00,
     & 0.13007D+00,0.13367D+00,0.13743D+00,0.15719D+00,0.16098D+00,
     & 0.16498D+00,0.16917D+00,0.17357D+00,0.19549D+00,0.19995D+00,
     & 0.20466D+00,0.20963D+00,0.21485D+00,0.22033D+00,0.24722D+00,
     & 0.25271D+00,0.25845D+00,0.26445D+00,0.27074D+00,0.27733D+00,
     & 0.28421D+00,0.29628D+00,0.29893D+00,0.30678D+00,0.34791D+00,
     & 0.35677D+00,0.36748D+00,0.37537D+00,0.38515D+00,0.39527D+00,
     & 0.40574D+00,0.41660D+00,0.42786D+00,0.43955D+00,0.45170D+00,
     & 0.47301D+00,0.47746D+00,0.49198D+00,0.50600D+00,0.51935D+00,
     & 0.52810D+00,0.34910D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.71912D-02,0.12718D-01,0.39948D-01,0.44650D-01,
     & 0.67908D-01,0.71987D-01,0.76048D-01,0.97556D-01,0.10132D+00,
     & 0.10517D+00,0.12620D+00,0.12994D+00,0.13382D+00,0.15556D+00,
     & 0.15943D+00,0.16349D+00,0.16774D+00,0.17219D+00,0.19601D+00,
     & 0.20049D+00,0.20522D+00,0.21021D+00,0.21546D+00,0.22096D+00,
     & 0.24994D+00,0.25541D+00,0.26113D+00,0.26712D+00,0.27340D+00,
     & 0.27997D+00,0.28685D+00,0.29447D+00,0.30156D+00,0.30941D+00,
     & 0.35347D+00,0.36237D+00,0.37359D+00,0.38103D+00,0.39082D+00,
     & 0.40096D+00,0.41144D+00,0.42231D+00,0.43358D+00,0.44527D+00,
     & 0.45741D+00,0.46919D+00,0.48311D+00,0.49749D+00,0.51151D+00,
     & 0.52524D+00,0.53455D+00,0.34588D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.19484D-01,
     & 0.25605D-01,0.54208D-01,0.58909D-01,0.63505D-01,0.88553D-01,
     & 0.92675D-01,0.96834D-01,0.12074D+00,0.12467D+00,0.12874D+00,
     & 0.15300D+00,0.15697D+00,0.16112D+00,0.16547D+00,0.17001D+00,
     & 0.19618D+00,0.20068D+00,0.20545D+00,0.21048D+00,0.21576D+00,
     & 0.22131D+00,0.25280D+00,0.25826D+00,0.26397D+00,0.26995D+00,
     & 0.27622D+00,0.28279D+00,0.28967D+00,0.29676D+00,0.30439D+00,
     & 0.31224D+00,0.35979D+00,0.36873D+00,0.38100D+00,0.38747D+00,
     & 0.39730D+00,0.40746D+00,0.41797D+00,0.42886D+00,0.44015D+00,
     & 0.45186D+00,0.46401D+00,0.47793D+00,0.48973D+00,0.50400D+00,
     & 0.51803D+00,0.53217D+00,0.54200D+00,0.34212D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.21246D-02,0.34384D-01,0.40304D-01,0.45757D-01,
     & 0.75816D-01,0.80469D-01,0.85099D-01,0.11288D+00,0.11712D+00,
     & 0.12146D+00,0.14900D+00,0.15314D+00,0.15744D+00,0.16193D+00,
     & 0.16659D+00,0.19573D+00,0.20029D+00,0.20511D+00,0.21020D+00,
     & 0.21554D+00,0.22115D+00,0.25574D+00,0.26119D+00,0.26689D+00,
     & 0.27287D+00,0.27914D+00,0.28572D+00,0.29260D+00,0.29981D+00,
     & 0.30735D+00,0.31523D+00,0.36697D+00,0.37597D+00,0.39167D+00,
     & 0.39481D+00,0.40468D+00,0.41488D+00,0.42544D+00,0.43637D+00,
     & 0.44769D+00,0.45943D+00,0.47162D+00,0.48570D+00,0.49740D+00,
     & 0.51159D+00,0.52565D+00,0.54025D+00,0.55054D+00,0.33767D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.57755D-02,0.11087D-01,
     & 0.17457D-01,0.57204D-01,0.62691D-01,0.68045D-01,0.10132D+00,
     & 0.10603D+00,0.11079D+00,0.14277D+00,0.14717D+00,0.15171D+00,
     & 0.15640D+00,0.16126D+00,0.19425D+00,0.19891D+00,0.20383D+00,
     & 0.20901D+00,0.21445D+00,0.22014D+00,0.25862D+00,0.26408D+00,
     & 0.26979D+00,0.27578D+00,0.28206D+00,0.28864D+00,0.29555D+00,
     & 0.30278D+00,0.31034D+00,0.31825D+00,0.37517D+00,0.38424D+00,
     & 0.39660D+00,0.40320D+00,0.41313D+00,0.42339D+00,0.43400D+00,
     & 0.44499D+00,0.45636D+00,0.46815D+00,0.48039D+00,0.49485D+00,
     & 0.50627D+00,0.52041D+00,0.53452D+00,0.54965D+00,0.56034D+00,
     & 0.33230D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.26535D-01,0.33966D-01,0.41107D-01,
     & 0.83761D-01,0.89235D-01,0.94659D-01,0.13294D+00,0.13775D+00,
     & 0.14267D+00,0.14771D+00,0.15288D+00,0.19106D+00,0.19588D+00,
     & 0.20095D+00,0.20628D+00,0.21186D+00,0.21770D+00,0.26122D+00,
     & 0.26670D+00,0.27243D+00,0.27844D+00,0.28475D+00,0.29136D+00,
     & 0.29830D+00,0.30557D+00,0.31318D+00,0.32113D+00,0.38453D+00,
     & 0.39371D+00,0.40312D+00,0.41285D+00,0.42287D+00,0.43321D+00,
     & 0.44389D+00,0.45495D+00,0.46639D+00,0.47826D+00,0.49057D+00,
     & 0.50559D+00,0.51660D+00,0.53070D+00,0.54489D+00,0.56068D+00,
     & 0.57165D+00,0.32572D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.83460D-03,
     & 0.39626D-02,0.55085D-01,0.62352D-01,0.69161D-01,0.11700D+00,
     & 0.12254D+00,0.12809D+00,0.13369D+00,0.13936D+00,0.18488D+00,
     & 0.18998D+00,0.19532D+00,0.20090D+00,0.20672D+00,0.21278D+00,
     & 0.26305D+00,0.26860D+00,0.27438D+00,0.28044D+00,0.28680D+00,
     & 0.29347D+00,0.30046D+00,0.30780D+00,0.31547D+00,0.32351D+00,
     & 0.39527D+00,0.40459D+00,0.41405D+00,0.42399D+00,0.43412D+00,
     & 0.44457D+00,0.45536D+00,0.46653D+00,0.47807D+00,0.49004D+00,
     & 0.50245D+00,0.51826D+00,0.52867D+00,0.54278D+00,0.55708D+00,
     & 0.57373D+00,0.58480D+00,0.31752D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.88612D-02,0.14943D-01,0.22038D-01,
     & 0.90092D-01,0.96920D-01,0.10359D+00,0.11018D+00,0.11673D+00,
     & 0.17332D+00,0.17894D+00,0.18474D+00,0.19075D+00,0.19697D+00,
     & 0.20341D+00,0.26319D+00,0.26887D+00,0.27477D+00,0.28094D+00,
     & 0.28740D+00,0.29417D+00,0.30127D+00,0.30871D+00,0.31650D+00,
     & 0.32464D+00,0.40758D+00,0.41711D+00,0.42687D+00,0.43687D+00,
     & 0.44717D+00,0.45777D+00,0.46871D+00,0.48002D+00,0.49171D+00,
     & 0.50381D+00,0.51637D+00,0.53597D+00,0.54287D+00,0.55704D+00,
     & 0.57149D+00,0.58920D+00,0.60018D+00,0.30707D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.35219D-01,0.44993D-01,0.54588D-01,0.63944D-01,
     & 0.73043D-01,0.15141D+00,0.15800D+00,0.16469D+00,0.17150D+00,
     & 0.17845D+00,0.18556D+00,0.25967D+00,0.26563D+00,0.27178D+00,
     & 0.27817D+00,0.28484D+00,0.29181D+00,0.29910D+00,0.30673D+00,
     & 0.31471D+00,0.32304D+00,0.42163D+00,0.43146D+00,0.44148D+00,
     & 0.45174D+00,0.46227D+00,0.47310D+00,0.48426D+00,0.49577D+00,
     & 0.50767D+00,0.51998D+00,0.53272D+00,0.55490D+00,0.55962D+00,
     & 0.57392D+00,0.58858D+00,0.61007D+00,0.61829D+00,0.29342D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.17941D-02,0.54727D-02,0.10368D+00,0.11357D+00,0.12309D+00,
     & 0.13229D+00,0.14124D+00,0.14999D+00,0.24791D+00,0.25450D+00,
     & 0.26121D+00,0.26809D+00,0.27521D+00,0.28259D+00,0.29028D+00,
     & 0.29828D+00,0.30662D+00,0.31530D+00,0.43732D+00,0.44762D+00,
     & 0.45807D+00,0.46871D+00,0.47960D+00,0.49076D+00,0.50224D+00,
     & 0.51407D+00,0.52627D+00,0.53887D+00,0.55191D+00,0.56809D+00,
     & 0.57937D+00,0.59391D+00,0.60887D+00,0.63612D+00,0.63971D+00,
     & 0.27507D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.12764D-01,0.19343D-01,
     & 0.26989D-01,0.35571D-01,0.44970D-01,0.55081D-01,0.21632D+00,
     & 0.22460D+00,0.23270D+00,0.24077D+00,0.24893D+00,0.25724D+00,
     & 0.26577D+00,0.27455D+00,0.28363D+00,0.29302D+00,0.45355D+00,
     & 0.46466D+00,0.47583D+00,0.48713D+00,0.49862D+00,0.51035D+00,
     & 0.52236D+00,0.53470D+00,0.54739D+00,0.56047D+00,0.57397D+00,
     & 0.58722D+00,0.60234D+00,0.61729D+00,0.63272D+00,0.65407D+00,
     & 0.66488D+00,0.24948D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.94740D-01,0.10861D+00,0.12224D+00,0.13557D+00,0.14869D+00,
     & 0.16166D+00,0.17456D+00,0.18743D+00,0.20033D+00,0.21330D+00,
     & 0.46656D+00,0.47926D+00,0.49180D+00,0.50431D+00,0.51688D+00,
     & 0.52959D+00,0.54252D+00,0.55571D+00,0.56924D+00,0.58313D+00,
     & 0.59743D+00,0.61220D+00,0.62738D+00,0.64310D+00,0.65934D+00,
     & 0.67506D+00,0.69339D+00,0.21209D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.30674D-02,
     & 0.73205D-02,0.41327D+00,0.43658D+00,0.45860D+00,0.47958D+00,
     & 0.49973D+00,0.51924D+00,0.53827D+00,0.55698D+00,0.57551D+00,
     & 0.59396D+00,0.61246D+00,0.63108D+00,0.64991D+00,0.66904D+00,
     & 0.68852D+00,0.70846D+00,0.72881D+00,0.15052D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.26769D-01,0.59100D-01,0.89129D-01,
     & 0.11729D+00,0.14392D+00,0.16932D+00,0.19374D+00,0.21738D+00,
     & 0.24042D+00,0.26303D+00,0.28534D+00,0.30748D+00,0.32956D+00,
     & 0.35168D+00,0.37393D+00,0.39636D+00,0.41917D+00,0.58148D-01,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,0.00000D+00,
     & 0.00000D+00/

       END
c----------------------------------------------------------------------              
c-*... jet with flat E vs eta distr. /A.Sobol
c----------------------------------------------------------------------              
      SUBROUTINE EDDEFLAT
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
c-*...standard PYTHIA ( v. >= 6.2) commons for initialization
c      INTEGER PYCOMP
      COMMON /PYJETS/ NN, NPAD, KK(4000,5), PP(4000,5), VV(4000,5)
      COMMON /PYDAT2/ KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON /PYINT5/ NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON /PYPARS/ MSTP(200),PARP(200),MSTI(200),PARI(200)
c-...some EDDE variables
c------ fundamental constants -----------------------------
      INTEGER NF,NC,NLOSW
      DOUBLE PRECISION PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1
      DOUBLE COMPLEX MNI,REI
      COMMON/EDDEFUND/ MNI,REI,PI,CSMB,LAMQCD,
     & TF,CF,BF0,BF1,NF,NC,NLOSW    
c------ parameters for soft rescattering (trajectories)----
c------ (t1,t2,fi0 dependence) ----------------------------
      INTEGER NAPR,NFI
      DOUBLE PRECISION CP,DP,RP,RG,AP,
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP
      COMMON/EDDESOFT/ CP(3),DP(3),RP(3),RG(3),AP(3),
     &   T1MIN,T1MAX,T2MIN,T2MAX,FKK,CGP,NAPR,NFI 
c----- parameters to calculate overall s-dependence -------
      DOUBLE PRECISION XI1MIN,XI2MIN,XI1MAX,XI2MAX
      COMMON/EDDETOT/ XI1MIN,XI2MIN,XI1MAX,XI2MAX
c----- parameters for hard cross-sections -----------------
      DOUBLE PRECISION MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
      COMMON/EDDEHARD/ MGGCUT,ETJCUT,MXMAX,
     & ETAJMAX,PLUM,PSURV,PSUD,ETASIMAX,SQS,
     & PSIDD1,PSIDD2
c--- restrictions on the phase space of g-jet ---
c--- DER3J - max. angle between g-jet and parallel jet ----
c--- XMAX3J - max ratio 2*Eg/MJJ --------------------------
c--- parameters for 3g functions -----------------------------
      DOUBLE PRECISION DER3J,XMAX3J,PAR3G
      COMMON/EDDE3JP/ DER3J,XMAX3J,PAR3G(5)     
c----- parameters to calculate total cross-sections -------      
c----- RS1 parameters -------------------------------------
      INTEGER NRS0 
      DOUBLE PRECISION RSXI0,RSGAM0,RSMH0,RSMR0 
      COMMON/EDDERS1/ RSXI0,RSGAM0,RSMH0,RSMR0,NRS0
c----- additional global parameters -----
      INTEGER KCP,IPROC
      DOUBLE PRECISION AM0,AMP,S,MQ
      COMMON/EDDEOTHER/ KCP,IPROC,AM0,AMP,S,MQ
c-...global EDDE parameters from FFR file
      INTEGER MXGLPAR
      INTEGER IPARS
      REAL RPARS
      PARAMETER   (MXGLPAR=200)
      COMMON /KI_TXPAR/ IPARS(MXGLPAR), RPARS(MXGLPAR)

      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST

      IF(FIRST) THEN
       KOD   =IPARS(30)
       EMIN  =RPARS(30)
       EMAX  =RPARS(31)
       ETAMIN=RPARS(32)
       ETAMAX=RPARS(33)
      ENDIF     
      
      E=PYR(0)*(EMAX-EMIN)
      ETA=ETAMIN+PYR(0)*(ETAMAX-ETAMIN)
      TET=2*DATAN(EXP(-ETA))
      FI  = 2.*PI*PYR(0)
      
      CALL PY1ENT(3,KOD,E,TET,FI)

      RETURN
      END
c**************************************************************************

C...EDDE_PYUPEV
C...write event in the format needed for the
C...Les Houches event record.
 
      SUBROUTINE EDDE_PYUPEV
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
 
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYCTAG/NCT,MCT(4000,2)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      COMMON/PYINT4/MWID(500),WIDS(500,5)
      SAVE /PYJETS/,/PYCTAG/,/PYDAT1/,/PYDAT2/,/PYDAT3/,/PYPARS/,
     &/PYINT1/,/PYINT2/,/PYINT4/
 
C...HEPEUP for output.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/
      
c IDWTUP = 3 requires XWGTUP = 1
      XWGTUP_wlhe = 1
      
C...Trace colour tags; convert to LHA style labels.
      NCT=100
      DO 150 I=MINT(84)+1,N
        MCT(I,1)=0
        MCT(I,2)=0
  150 CONTINUE
      DO 160 I=MINT(84)+1,N
        KQ=KCHG(PYCOMP(K(I,2)),2)*ISIGN(1,K(I,2))
        IF(K(I,1).EQ.3.OR.K(I,1).EQ.13.OR.K(I,1).EQ.14) THEN
          IF(K(I,4).NE.0.AND.(KQ.EQ.1.OR.KQ.EQ.2).AND.MCT(I,1).EQ.0)
     &    THEN
            IMO=MOD(K(I,4)/MSTU(5),MSTU(5))
            IDA=MOD(K(I,4),MSTU(5))
            IF(IMO.NE.0.AND.MOD(K(IMO,5)/MSTU(5),MSTU(5)).EQ.I.AND.
     &      MCT(IMO,2).NE.0) THEN
              MCT(I,1)=MCT(IMO,2)
            ELSEIF(IMO.NE.0.AND.MOD(K(IMO,4),MSTU(5)).EQ.I.AND.
     &      MCT(IMO,1).NE.0) THEN
              MCT(I,1)=MCT(IMO,1)
            ELSEIF(IDA.NE.0.AND.MOD(K(IDA,5),MSTU(5)).EQ.I.AND.
     &      MCT(IDA,2).NE.0) THEN
              MCT(I,1)=MCT(IDA,2)
            ELSE
              NCT=NCT+1
              MCT(I,1)=NCT
            ENDIF
          ENDIF
          IF(K(I,5).NE.0.AND.(KQ.EQ.-1.OR.KQ.EQ.2).AND.MCT(I,2).EQ.0)
     &    THEN
            IMO=MOD(K(I,5)/MSTU(5),MSTU(5))
            IDA=MOD(K(I,5),MSTU(5))
            IF(IMO.NE.0.AND.MOD(K(IMO,4)/MSTU(5),MSTU(5)).EQ.I.AND.
     &      MCT(IMO,1).NE.0) THEN
              MCT(I,2)=MCT(IMO,1)
            ELSEIF(IMO.NE.0.AND.MOD(K(IMO,5),MSTU(5)).EQ.I.AND.
     &      MCT(IMO,2).NE.0) THEN
              MCT(I,2)=MCT(IMO,2)
            ELSEIF(IDA.NE.0.AND.MOD(K(IDA,4),MSTU(5)).EQ.I.AND.
     &      MCT(IDA,1).NE.0) THEN
              MCT(I,2)=MCT(IDA,1)
            ELSE
              NCT=NCT+1
              MCT(I,2)=NCT
            ENDIF
          ENDIF
        ENDIF
  160 CONTINUE
  
C...Put event in HEPEUP commonblock.
      NUP=N-MINT(84)
      IDPRUP=MINT(1)
c      IDPRUP=1
      XWGTUP=1D0
c      SCALUP=VINT(53)
c      AQEDUP=VINT(57)
c      AQCDUP=VINT(58)
      SCALUP=-1
      AQEDUP=-1
      AQCDUP=-1
      DO 180 I=1,NUP
        IDUP(I)=K(I+MINT(84),2)
        IF(I.LE.2) THEN
          ISTUP(I)=-1
          MOTHUP(1,I)=0
          MOTHUP(2,I)=0
        ELSEIF(K(I+4,3).EQ.0) THEN
          ISTUP(I)=1
          MOTHUP(1,I)=1
          MOTHUP(2,I)=2
        ELSE
          ISTUP(I)=1
          MOTHUP(1,I)=K(I+MINT(84),3)-MINT(84)
          MOTHUP(2,I)=0
        ENDIF
c        IF(I.GE.3.AND.K(I+MINT(84),3).GT.0)
        IF(I.GE.5.AND.K(I+MINT(84),3).GT.0)
     &  ISTUP(K(I+MINT(84),3)-MINT(84))=2
        ICOLUP(1,I)=MCT(I+MINT(84),1)
        ICOLUP(2,I)=MCT(I+MINT(84),2)
        DO 170 J=1,5
          PUP(J,I)=P(I+MINT(84),J)
  170   CONTINUE
        VTIMUP(I)=V(I,5)
        SPINUP(I)=9D0
  180 CONTINUE
  
C...Optionally write out event to disk. Minimal size for time/spin fields.
      IF(MSTP(162).GT.0) THEN
        WRITE(MSTP(162),5200) NUP,IDPRUP,XWGTUP_wlhe,SCALUP,AQEDUP
     +   ,AQCDUP
        DO 190 I=1,NUP
          IF(VTIMUP(I).EQ.0D0) THEN
            WRITE(MSTP(162),5300) IDUP(I),ISTUP(I),MOTHUP(1,I),
     &      MOTHUP(2,I),ICOLUP(1,I),ICOLUP(2,I),(PUP(J,I),J=1,5),
     &      ' 0. 9.'
          ELSE
            WRITE(MSTP(162),5400) IDUP(I),ISTUP(I),MOTHUP(1,I),
     &      MOTHUP(2,I),ICOLUP(1,I),ICOLUP(2,I),(PUP(J,I),J=1,5),
     &      VTIMUP(I),' 9.'
          ENDIF
  190   CONTINUE

C...Optional extra line with parton-density information.
        IF(MSTP(165).GE.1) WRITE(MSTP(162),5500) MSTI(15),MSTI(16),
     &  PARI(33),PARI(34),PARI(23),PARI(29),PARI(30) 
      ENDIF
 
C...Print formats.

 5200 FORMAT(1P,2I6,4E14.6)
 5300 FORMAT(1P,I8,5I5,5E18.10,A6)
 5400 FORMAT(1P,I8,5I5,5E18.10,E12.4,A3)
 5500 FORMAT(1P,'#pdf ',2I5,5E18.10)
 
      RETURN
      END
      
C*********************************************************************
 
C...EDDE_PYUPIN
C...Fills the HEPRUP commonblock with info on incoming beams and allowed
C...processes, and optionally stores that information on file.
 
      SUBROUTINE EDDE_PYUPIN
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      SAVE /PYJETS/,/PYSUBS/,/PYPARS/,/PYINT5/
 
C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/
 
C...Store info on incoming beams.
      IDBMUP(1)=K(1,2)
      IDBMUP(2)=K(2,2)
      EBMUP(1)=P(1,4)
      EBMUP(2)=P(2,4)
c      PDFGUP(1)=0
c      PDFGUP(2)=0
c      PDFSUP(1)=MSTP(51)
c      PDFSUP(2)=MSTP(51)
      PDFGUP(1)=-9
      PDFGUP(2)=-9
      PDFSUP(1)=-9
      PDFSUP(2)=-9
 
C...Event weighting strategy.
      IDWTUP=3
 
C...Info on individual processes.
      NPRUP=0
      DO 100 ISUB=1,500
        IF(MSUB(ISUB).EQ.1) THEN
          NPRUP=NPRUP+1
          XSECUP(NPRUP)=1D9*XSEC(ISUB,3)
          XERRUP(NPRUP)=XSECUP(NPRUP)/SQRT(MAX(1D0,DBLE(NGEN(ISUB,3))))
          XMAXUP(NPRUP)=1D0
          LPRUP(NPRUP)=ISUB
        ENDIF
  100 CONTINUE
 
C...Write info to file.
      IF(MSTP(161).GT.0) THEN
        WRITE(MSTP(161),5100) IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     &  PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP
        DO 110 IPR=1,NPRUP
          WRITE(MSTP(161),5200) XSECUP(IPR),XERRUP(IPR),XMAXUP(IPR),
     &    LPRUP(IPR)
  110   CONTINUE
      ENDIF
 
C...Formats for printout.
 5100 FORMAT(1P,2I8,2E14.6,6I6)
 5200 FORMAT(1P,3E14.6,I6)
 
      RETURN
      END      
