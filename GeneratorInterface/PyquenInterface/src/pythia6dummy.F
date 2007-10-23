 
C*********************************************************************
 
C...FHHIGGSCORR
C...Dummy function, to be removed when FEYNHIGGS is to be linked.
 
      SUBROUTINE FHHIGGSCORR(IERR, RMHIGG, SAEFF, UHIGGS)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...FeynHiggs variables
      DOUBLE PRECISION RMHIGG(4)
      DOUBLE COMPLEX SAEFF, UHIGGS(3,3)
      DOUBLE COMPLEX DMU,
     &     AE33, AU33, AD33, AE22, AU22, AD22, AE11, AU11, AD11,
     &     DM1, DM2, DM3

C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link FEYNHIGGS correctly.'/
     &1X,'Dummy routine FHSETPARA in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
      RETURN
      END
 
C*********************************************************************
 
C...FHSETFLAGS
C...Dummy function, to be removed when FEYNHIGGS is to be linked.
 
      SUBROUTINE FHSETFLAGS(IERR,IMSP,IFR,ITBR,IHMX,IP2A,ILP,ITR,IBR)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
Cmssmpart = 4     # full MSSM [recommended]
Cfieldren = 0     # MSbar field ren. [strongly recommended]
Ctanbren =  0     # MSbar TB-ren. [strongly recommended]
Chiggsmix = 2     # 2x2 (h0-HH) mixing in the neutral Higgs sector
Cp2approx = 0     # no approximation [recommended]
Clooplevel= 2     # include 2-loop corrections
Ctl_running_mt= 1 # running top mass in 2-loop corrections [recommended]
Ctl_bot_resum = 1 # resummed MB in 2-loop corrections [recommended]
 
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link FEYNHIGGS correctly.'/
     &1X,'Dummy routine FHSETFLAGS in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
      RETURN
      END
 
C*********************************************************************
 
C...FHSETPARA
C...Dummy function, to be removed when FEYNHIGGS is to be linked.
 
      SUBROUTINE FHSETPARA(IER,SCF,DMT,DMB,DMW,DMZ,DTANB,DMA,DMH,DM3L,
     &     DM3E,DM3Q,DM3U,DM3D,DM2L,DM2E,DM2Q,DM2U, DM2D,DM1L,DM1E,DM1Q,
     &     DM1U,DM1D,DMU,AE33,AU33,AD33,AE22,AU22,AD22,AE11,AU11,AD11,
     &     DM1,DM2,DM3,RLT,RLB,QTAU,QT,QB)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
      DOUBLE COMPLEX SAEFF, UHIGGS(3,3)
      DOUBLE COMPLEX DMU,
     &     AE33, AU33, AD33, AE22, AU22, AD22, AE11, AU11, AD11,
     &     DM1, DM2, DM3

C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link FEYNHIGGS correctly.'/
     &1X,'Dummy routine FHSETPARA in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
      RETURN
      END
 
C*********************************************************************
 
C...PYEVWT
C...Dummy routine, which the user can replace in order to multiply the
C...standard PYTHIA differential cross-section by a process- and
C...kinematics-dependent factor WTXS. For MSTP(142)=1 this corresponds
C...to generation of weighted events, with weight 1/WTXS, while for
C...MSTP(142)=2 it corresponds to a modification of the underlying
C...physics.
 
      SUBROUTINE PYEVWT(WTXS)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      SAVE /PYDAT1/,/PYINT1/,/PYINT2/
 
C...Set default weight for WTXS.
      WTXS=1D0
 
C...Read out subprocess number.
      ISUB=MINT(1)
      ISTSB=ISET(ISUB)
 
C...Read out tau, y*, cos(theta), tau' (where defined, else =0).
      TAU=VINT(21)
      YST=VINT(22)
      CTH=0D0
      IF(ISTSB.EQ.2.OR.ISTSB.EQ.4) CTH=VINT(23)
      TAUP=0D0
      IF(ISTSB.GE.3.AND.ISTSB.LE.5) TAUP=VINT(26)
 
C...Read out x_1, x_2, x_F, shat, that, uhat, p_T^2.
      X1=VINT(41)
      X2=VINT(42)
      XF=X1-X2
      SHAT=VINT(44)
      THAT=VINT(45)
      UHAT=VINT(46)
      PT2=VINT(48)
 
C...Modifications by user to be put here.
 
C...Stop program if this routine is ever called.
C...You should not copy these lines to your own routine.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link your PYEVWT routine ',
     &'correctly.'/1X,'Dummy routine in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END
 
C*********************************************************************
 
C...PYKCUT
C...Dummy routine, which the user can replace in order to make cuts on
C...the kinematics on the parton level before the matrix elements are
C...evaluated and the event is generated. The cross-section estimates
C...will automatically take these cuts into account, so the given
C...values are for the allowed phase space region only. MCUT=0 means
C...that the event has passed the cuts, MCUT=1 that it has failed.
 
      SUBROUTINE PYKCUT(MCUT)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      SAVE /PYDAT1/,/PYINT1/,/PYINT2/
 
C...Set default value (accepting event) for MCUT.
      MCUT=0
 
C...Read out subprocess number.
      ISUB=MINT(1)
      ISTSB=ISET(ISUB)
 
C...Read out tau, y*, cos(theta), tau' (where defined, else =0).
      TAU=VINT(21)
      YST=VINT(22)
      CTH=0D0
      IF(ISTSB.EQ.2.OR.ISTSB.EQ.4) CTH=VINT(23)
      TAUP=0D0
      IF(ISTSB.GE.3.AND.ISTSB.LE.5) TAUP=VINT(26)
 
C...Calculate x_1, x_2, x_F.
      IF(ISTSB.LE.2.OR.ISTSB.GE.5) THEN
        X1=SQRT(TAU)*EXP(YST)
        X2=SQRT(TAU)*EXP(-YST)
      ELSE
        X1=SQRT(TAUP)*EXP(YST)
        X2=SQRT(TAUP)*EXP(-YST)
      ENDIF
      XF=X1-X2
 
C...Calculate shat, that, uhat, p_T^2.
      SHAT=TAU*VINT(2)
      SQM3=VINT(63)
      SQM4=VINT(64)
      RM3=SQM3/SHAT
      RM4=SQM4/SHAT
      BE34=SQRT(MAX(0D0,(1D0-RM3-RM4)**2-4D0*RM3*RM4))
      RPTS=4D0*VINT(71)**2/SHAT
      BE34L=SQRT(MAX(0D0,(1D0-RM3-RM4)**2-4D0*RM3*RM4-RPTS))
      RM34=2D0*RM3*RM4
      RSQM=1D0+RM34
      RTHM=(4D0*RM3*RM4+RPTS)/(1D0-RM3-RM4+BE34L)
      THAT=-0.5D0*SHAT*MAX(RTHM,1D0-RM3-RM4-BE34*CTH)
      UHAT=-0.5D0*SHAT*MAX(RTHM,1D0-RM3-RM4+BE34*CTH)
      PT2=MAX(VINT(71)**2,0.25D0*SHAT*BE34**2*(1D0-CTH**2))
 
C...Decisions by user to be put here.
 
C...Stop program if this routine is ever called.
C...You should not copy these lines to your own routine.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link your PYKCUT routine ',
     &'correctly.'/1X,'Dummy routine in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END
  
C*********************************************************************
 
C...PYTAUD
C...Dummy routine, to be replaced by user, to handle the decay of a
C...polarized tau lepton.
C...Input:
C...ITAU is the position where the decaying tau is stored in /PYJETS/.
C...IORIG is the position where the mother of the tau is stored;
C...     is 0 when the mother is not stored.
C...KFORIG is the flavour of the mother of the tau;
C...     is 0 when the mother is not known.
C...Note that IORIG=0 does not necessarily imply KFORIG=0;
C...     e.g. in B hadron semileptonic decays the W  propagator
C...     is not explicitly stored but the W code is still unambiguous.
C...Output:
C...NDECAY is the number of decay products in the current tau decay.
C...These decay products should be added to the /PYJETS/ common block,
C...in positions N+1 through N+NDECAY. For each product I you must
C...give the flavour codes K(I,2) and the five-momenta P(I,1), P(I,2),
C...P(I,3), P(I,4) and P(I,5). The rest will be stored automatically.
 
      SUBROUTINE PYTAUD(ITAU,IORIG,KFORIG,NDECAY)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYJETS/,/PYDAT1/
 
C...Stop program if this routine is ever called.
C...You should not copy these lines to your own routine.
      NDECAY=ITAU+IORIG+KFORIG
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link your PYTAUD routine ',
     &'correctly.'/1X,'Dummy routine in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END
 
C*********************************************************************
 
C...PYTIME
C...Finds current date and time.
C...Since this task is not standardized in Fortran 77, the routine
C...is dummy, to be replaced by the user. Examples are given for
C...the Fortran 90 routine and DEC Fortran 77, and what to do if
C...you do not have access to suitable routines.
 
      SUBROUTINE PYTIME(IDATI)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
      CHARACTER*8 ATIME
C...Local array.
      INTEGER IDATI(6),IDTEMP(3),IVAL(8)
 
C...Example 0: if you do not have suitable routines.
      DO 100 J=1,6
      IDATI(J)=0
  100 CONTINUE
 
C...Example 1: Fortran 90 routine.
C      CALL DATE_AND_TIME(VALUES=IVAL)
C      IDATI(1)=IVAL(1)
C      IDATI(2)=IVAL(2)
C      IDATI(3)=IVAL(3)
C      IDATI(4)=IVAL(5)
C      IDATI(5)=IVAL(6)
C      IDATI(6)=IVAL(7)
 
C...Example 2: DEC Fortran 77. AIX.
C      CALL IDATE(IMON,IDAY,IYEAR)
C      IDATI(1)=IYEAR
C      IDATI(2)=IMON
C      IDATI(3)=IDAY
C      CALL ITIME(IHOUR,IMIN,ISEC)
C      IDATI(4)=IHOUR
C      IDATI(5)=IMIN
C      IDATI(6)=ISEC
 
C...Example 3: DEC Fortran, IRIX, IRIX64.
C      CALL IDATE(IMON,IDAY,IYEAR)
C      IDATI(1)=IYEAR
C      IDATI(2)=IMON
C      IDATI(3)=IDAY
C      CALL TIME(ATIME)
C      IHOUR=0
C      IMIN=0
C      ISEC=0
C      READ(ATIME(1:2),'(I2)') IHOUR
C      READ(ATIME(4:5),'(I2)') IMIN
C      READ(ATIME(7:8),'(I2)') ISEC
C      IDATI(4)=IHOUR
C      IDATI(5)=IMIN
C      IDATI(6)=ISEC
 
C...Example 4: GNU LINUX libU77, SunOS.
C      CALL IDATE(IDTEMP)
C      IDATI(1)=IDTEMP(3)
C      IDATI(2)=IDTEMP(2)
C      IDATI(3)=IDTEMP(1)
C      CALL ITIME(IDTEMP)
C      IDATI(4)=IDTEMP(1)
C      IDATI(5)=IDTEMP(2)
C      IDATI(6)=IDTEMP(3)
 
C...Common code to ensure right century.
      IDATI(1)=2000+MOD(IDATI(1),100)
 
      RETURN
      END
 
C*********************************************************************
 
C...SSMSSM
C...Dummy function, to be removed when ISAJET (ISASUSY) is to be linked.
 
      SUBROUTINE SSMSSM(RDUM1,RDUM2,RDUM3,RDUM4,RDUM5,RDUM6,RDUM7,
     &RDUM8,RDUM9,RDUM10,RDUM11,RDUM12,RDUM13,RDUM14,RDUM15,RDUM16,
     &RDUM17,RDUM18,RDUM19,RDUM20,RDUM21,RDUM22,RDUM23,RDUM24,RDUM25,
     &IDUM1,IDUM2)
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      REAL RDUM1,RDUM2,RDUM3,RDUM4,RDUM5,RDUM6,RDUM7,RDUM8,RDUM9,
     &RDUM10,RDUM11,RDUM12,RDUM13,RDUM14,RDUM15,RDUM16,RDUM17,RDUM18,
     &RDUM19,RDUM20,RDUM21,RDUM22,RDUM23,RDUM24,RDUM25
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link ISAJET correctly.'/
     &1X,'Dummy routine SSMSSM in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
      RETURN
      END
 
C*********************************************************************
 
C...SUGRA
C...Dummy routine, to be removed when ISAJET (ISASUSY) is to be linked.
 
      SUBROUTINE SUGRA(MZERO,MHLF,AZERO,TANB,SGNMU,MTOP,IMODL)
       IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      REAL MZERO,MHLF,AZERO,TANB,SGNMU,MTOP
      INTEGER IMODL
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link ISAJET correctly.'/
     &1X,'Dummy routine SUGRA in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END

C...Old example: handles a simple Pythia 6.4 initialization file.
 
c      SUBROUTINE UPINIT
 
C...Double precision and integer declarations.
c      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
c      IMPLICIT INTEGER(I-N)
 
C...Commonblocks.
c      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
c      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
c      SAVE /PYDAT1/,/PYPARS/
 
C...User process initialization commonblock.
c      INTEGER MAXPUP
c      PARAMETER (MAXPUP=100)
c      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
c      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
c      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
c     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
c     &LPRUP(MAXPUP)
c      SAVE /HEPRUP/
 
C...Read info from file.
c      IF(MSTP(161).GT.0) THEN
c        READ(MSTP(161),*,END=110,ERR=110) IDBMUP(1),IDBMUP(2),EBMUP(1),
c     &  EBMUP(2),PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP
c        DO 100 IPR=1,NPRUP
c          READ(MSTP(161),*,END=110,ERR=110) XSECUP(IPR),XERRUP(IPR),
c     &    XMAXUP(IPR),LPRUP(IPR)
c  100   CONTINUE
c        RETURN
C...Error or prematurely reached end of file.
c  110   WRITE(MSTU(11),5000)
c        STOP
 
C...Else not implemented.
c      ELSE
c        WRITE(MSTU(11),5100)
c        STOP
c      ENDIF
 
C...Format for error printout.
c 5000 FORMAT(1X,'Error: UPINIT routine failed to read information'/
c     &1X,'Execution stopped!')
c 5100 FORMAT(1X,'Error: You have not implemented UPINIT routine'/
c     &1X,'Dummy routine in PYTHIA file called instead.'/
c     &1X,'Execution stopped!')
 
c      RETURN
c      END
 
C*********************************************************************
 
C...UPEVNT
C...Dummy routine, to be replaced by a user implementing external
C...processes. Depending on cross section model chosen, it either has
C...to generate a process of the type IDPRUP requested, or pick a type
C...itself and generate this event. The event is to be stored in the
C...HEPEUP commonblock, including (often) an event weight.

C...New example: handles a standard Les Houches Events File.

      SUBROUTINE UPEVNT
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...PYTHIA commonblock: only used to provide read unit MSTP(162).
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      SAVE /PYPARS/
 
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/

C...Lines to read in assumed never longer than 200 characters. 
      PARAMETER (MAXLEN=200)
      CHARACTER*(MAXLEN) STRING

C...Format for reading lines.
      CHARACTER*6 STRFMT
      STRFMT='(A000)'
      WRITE(STRFMT(3:5),'(I3)') MAXLEN

C...Loop until finds line beginning with "<event>" or "<event ". 
  100 READ(MSTP(162),STRFMT,END=130,ERR=130) STRING
      IBEG=0
  110 IBEG=IBEG+1
C...Allow indentation.
      IF(STRING(IBEG:IBEG).EQ.' '.AND.IBEG.LT.MAXLEN-6) GOTO 110 
      IF(STRING(IBEG:IBEG+6).NE.'<event>'.AND.
     &STRING(IBEG:IBEG+6).NE.'<event ') GOTO 100

C...Read first line of event info.
      READ(MSTP(162),*,END=130,ERR=130) NUP,IDPRUP,XWGTUP,SCALUP,
     &AQEDUP,AQCDUP

C...Read NUP subsequent lines with information on each particle.
      DO 120 I=1,NUP
        READ(MSTP(162),*,END=130,ERR=130) IDUP(I),ISTUP(I),
     &  MOTHUP(1,I),MOTHUP(2,I),ICOLUP(1,I),ICOLUP(2,I),
     &  (PUP(J,I),J=1,5),VTIMUP(I),SPINUP(I)
  120 CONTINUE
      RETURN

C...Error exit, typically when no more events.
  130 WRITE(*,*) ' Failed to read LHEF event information.'
      WRITE(*,*) ' Will assume end of file has been reached.'
      NUP=0
      MSTI(51)=1
 
      RETURN
      END
 
C*********************************************************************
 
C...UPINIT
C...Dummy routine, to be replaced by a user implementing external
C...processes. Is supposed to fill the HEPRUP commonblock with info
C...on incoming beams and allowed processes.

C...New example: handles a standard Les Houches Events File.

      SUBROUTINE UPINIT
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
 
C...PYTHIA commonblock: only used to provide read unit MSTP(161).
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      SAVE /PYPARS/
 
C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
      SAVE /HEPRUP/

C...Lines to read in assumed never longer than 200 characters. 
      PARAMETER (MAXLEN=200)
      CHARACTER*(MAXLEN) STRING

C...Format for reading lines.
      CHARACTER*6 STRFMT
      STRFMT='(A000)'
      WRITE(STRFMT(3:5),'(I3)') MAXLEN

C...Loop until finds line beginning with "<init>" or "<init ". 
  100 READ(MSTP(161),STRFMT,END=130,ERR=130) STRING
      IBEG=0
  110 IBEG=IBEG+1
C...Allow indentation.
      IF(STRING(IBEG:IBEG).EQ.' '.AND.IBEG.LT.MAXLEN-5) GOTO 110 
      IF(STRING(IBEG:IBEG+5).NE.'<init>'.AND.
     &STRING(IBEG:IBEG+5).NE.'<init ') GOTO 100

C...Read first line of initialization info.
      READ(MSTP(161),*,END=130,ERR=130) IDBMUP(1),IDBMUP(2),EBMUP(1),
     &EBMUP(2),PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP

C...Read NPRUP subsequent lines with information on each process.
      DO 120 IPR=1,NPRUP
        READ(MSTP(161),*,END=130,ERR=130) XSECUP(IPR),XERRUP(IPR),
     &  XMAXUP(IPR),LPRUP(IPR)
  120 CONTINUE
      RETURN

C...Error exit: give up if initalization does not work.
  130 WRITE(*,*) ' Failed to read LHEF initialization information.'
      WRITE(*,*) ' Event generation will be stopped.'
      STOP  
 
      RETURN
      END
 
C*********************************************************************
 
C...VISAJE
C...Dummy function, to be removed when ISAJET (ISASUSY) is to be linked.
 
      FUNCTION VISAJE()
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      CHARACTER*40 VISAJE
 
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
 
C...Assign default value.
      VISAJE='Undefined'
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link ISAJET correctly.'/
     &1X,'Dummy function VISAJE in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END

C...Old example: handles a simple Pythia 6.4 event file.
 
c      SUBROUTINE UPEVNT
 
C...Double precision and integer declarations.
c      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
c      IMPLICIT INTEGER(I-N)
 
C...Commonblocks.
c      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
c      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
c      SAVE /PYDAT1/,/PYPARS/
 
C...User process event common block.
c      INTEGER MAXNUP
c      PARAMETER (MAXNUP=500)
c      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
c      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
c      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
c     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
c     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
c      SAVE /HEPEUP/
 
C...Read info from file.
c      IF(MSTP(162).GT.0) THEN
c        READ(MSTP(162),*,END=110,ERR=110) NUP,IDPRUP,XWGTUP,SCALUP,
c     &  AQEDUP,AQCDUP
c        DO 100 I=1,NUP
c          READ(MSTP(162),*,END=110,ERR=110) IDUP(I),ISTUP(I),
c     &    MOTHUP(1,I),MOTHUP(2,I),ICOLUP(1,I),ICOLUP(2,I),
c     &    (PUP(J,I),J=1,5),VTIMUP(I),SPINUP(I)
c  100   CONTINUE
c        RETURN
C...Special when reached end of file or other error.
c  110   NUP=0
 
C...Else not implemented.
c      ELSE
c        WRITE(MSTU(11),5000)
c        STOP
c      ENDIF
 
C...Format for error printout.
c 5000 FORMAT(1X,'Error: You have not implemented UPEVNT routine'/
c     &1X,'Dummy routine in PYTHIA file called instead.'/
c     &1X,'Execution stopped!')
 
c      RETURN
c      END
 
C*********************************************************************
 
C...UPVETO
C...Dummy routine, to be replaced by user, to veto event generation
C...on the parton level, after parton showers but before multiple
C...interactions, beam remnants and hadronization is added.
C...If resonances like W, Z, top, Higgs and SUSY particles are handed
C...undecayed from UPEVNT, or are generated by PYTHIA, they will also
C...be undecayed at this stage; if decayed their decay products will
C...have been allowed to shower.
 
C...All partons at the end of the shower phase are stored in the
C...HEPEVT commonblock. The interesting information is
C...NHEP = the number of such partons, in entries 1 <= i <= NHEP,
C...IDHEP(I) = the particle ID code according to PDG conventions,
C...PHEP(J,I) = the (p_x, p_y, p_z, E, m) of the particle.
C...All ISTHEP entries are 1, while the rest is zeroed.
 
C...The user decision is to be conveyed by the IVETO value.
C...IVETO = 0 : retain current event and generate in full;
C...      = 1 : abort generation of current event and move to next.
 
      SUBROUTINE UPVETO(IVETO)
 
C...HEPEVT commonblock.
#include "hepevt.inc"
 
C...Next few lines allow you to see what info PYVETO extracted from
C...the full event record for the first two events.
C...Delete if you don't want it.
      DATA NLIST/0/
      SAVE NLIST
      IF(NLIST.LE.2) THEN
        WRITE(*,*) ' Full event record at time of UPVETO call:'
        CALL PYLIST(1)
        WRITE(*,*) ' Part of event record made available to UPVETO:'
        CALL PYLIST(5)
        NLIST=NLIST+1
      ENDIF
 
C...Make decision here.
      IVETO = 0
 
      RETURN
      END
