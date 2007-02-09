 
*                 Millepede - Linear Least Squares
*                 ================================
*     A Least Squares Method for Detector Alignment - Fortran code
*
*     TESTMP      short test program for detector aligment
*                 with GENER (generator) + ZRAND, ZNORM (random gen.)
*
*     The execution of the test program needs a MAIN program:
*                 CALL TESTMP(0)
*                 CALL TESTMP(1)
*                 END
*
*     INITGL      initialization
*     PARGLO       optional: initialize parameters with nonzero values
*     PARSIG       optional: define sigma for single parameter
*     INITUN       optional: unit for iterations
*     CONSTF       optional: constraints
*     EQULOC      equations for local fit
*     ZERLOC
*     FITLOC      local parameter fit (+entry KILLOC)
*     FITGLO      final global parameter fit
*     ERRPAR      parameter errors
*     CORPAR      parameter correlations
*     PRTGLO      print result on file
*
*     Special matrix subprograms (all in Double Precision):
*        SPMINV   matrix inversion + solution
*        SPAVAT   special double matrix product
*        SPAX     product matrix times vector
*
*     PXHIST      histogram printing
*     CHFMT       formatting of real numbers
*     CHINDL      limit of chi**2/nd
*     FITMUT/FITMIN vector input/ouput for parallel processing
 
      SUBROUTINE TESTMP(IARG)       ! test program IARG = 0 or 1
*     test of millepede
*     IARG = 0   test with constraint
*     IARG = 1   test without constraint
      REAL DERGB(10),DERLC(2),PAR(10)
      PARAMETER (NPLAN=10)
      REAL ARRAY(1000)
      REAL X(NPLAN),Y(NPLAN),SIGMA(NPLAN),BIAS(NPLAN),HEFF(NPLAN)
      REAL FROT(NPLAN)
      DATA X/5.0,9.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0/
      DATA SIGMA/2*0.0020,8*0.0300/
      DATA BIAS
     + /0.00,-0.04,0.15,0.03,-0.075,0.045,0.035,-0.08,0.09,-0.05/
      DATA HEFF/4*0.9,0.50,5*0.9/
*     ...
      NCASES=1000
      CALL INITGL(10,2,3,1)        ! define dimension parameters
      CALL PARSIG(1,0.0)           ! parameter 1 is fixed
      IF(IARG.EQ.0) THEN
         FROT(1)=0.0
         DO I=2,10
          FROT(I)=1.0/(X(I)-X(1))
         END DO
         CALL CONSTF(FROT,0.0)     ! constraint: total rotation zero
      ELSE
         DO I=3,10
          CALL PARSIG(I,0.03)      ! parameters 3...8
         END DO
      END IF
      CALL INITUN(11,10.0)         ! option iterations
      CALL ZERLOC(DERGB,DERLC)     ! initialization of arrays to zero
*     -------------- loop
      DO NC=1,NCASES
*      generate straight line parameters
       CALL GENER(Y,A,B,X,SIGMA,BIAS,HEFF)
       DO I=1,NPLAN
*       calibrate  Y(I) = A + B * X(I)
        IF(Y(I).NE.0.0) THEN
           DERLC(1)= 1.0
           DERLC(2)= X(I)
           DERGB(I)= 1.0
           CALL EQULOC(DERGB,DERLC,Y(I),SIGMA(I)) ! single measurement
        END IF
       END DO
       CALL FITLOC
      END DO ! end loop cases
*     -------------- loop end
 10   CONTINUE
      CALL FITGLO(PAR)             ! final solution
      END
      SUBROUTINE GENER(Y,A,B,X,SIGMA,BIAS,HEFF)  !  generate Y values
      REAL Y(*)
      PARAMETER (NPLAN=10)
      REAL X(NPLAN),SIGMA(NPLAN),BIAS(NPLAN),HEFF(NPLAN)
*     ...
*     generate straight line parameters
      A=20.0*ZRAND()-10.0        ! A =   -10 ...  +10   uniform
      B= 0.3*ZNORM()             ! B =  -0.3 ...  +0.3  gaussian
      DO I=1,NPLAN  ! planes
        IF(ZRAND().LT.HEFF(I)) THEN
           Y(I)=A+B*X(I) + BIAS(I)+SIGMA(I)*ZNORM()
        ELSE
           Y(I)=0.0  ! unmeasured
        END IF
      END DO
      END
      FUNCTION ZRAND()  ! return random number U(0,1)
*     (simple generator, showing principle)
      PARAMETER (IA=205,IC=29573,IM=139968)
      DATA LAST/4711/
      LAST=MOD(IA*LAST+IC,IM)
      IF(LAST.EQ.0) LAST=MOD(IA*LAST+IC,IM)
      ZRAND=FLOAT(LAST)/FLOAT(IM)
      END
      FUNCTION ZNORM()  ! return random number N(0,1)
*     (simple generator, showing principle)
      PARAMETER (IA=205,IC=29573,IM=139968)
      DATA LAST/4711/
      ZNORM=-6.0
      DO I=1,12
       LAST=MOD(IA*LAST+IC,IM)
       ZNORM=ZNORM+FLOAT(LAST)/FLOAT(IM)
      END DO
      END
 
 
      SUBROUTINE INITGL(NAGBAR,NALCAR,NSTD,IPRLIM)
*     initialization of package
*     NAGB = number of global parameters
*     DERGB(1) ... DERGB(NAGB) = derivatives w.r.t. global parameters
*     NALC = number of local parameters (maximum)
*     DERLC(1) ... DERLC(NALC) = derivatives w.r.t. local parameters
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      INTEGER NDR(7)
      DATA NDR/1,2,5,10,20,50,100/
*     ...
*     write(6,*)"In INITGL"
*     write(6,*)NAGBAR,NALCAR,NSTD,IPRLIM
      ICNLIM=IPRLIM
      IF(ICNLIM.GE.0) WRITE(*,199)
 199  FORMAT(
     +'                                                       '/
     +'              *   o   o                        o       '/
     +'                  o   o                        o       '/
     +'   o ooooo    o   o   o    oo   ooo    oo    ooo   oo  '/
     +'    o  o  o   o   o   o   o  o  o  o  o  o  o  o  o  o '/
     +'    o  o  o   o   o   o   oooo  o  o  oooo  o  o  oooo '/
     +'    o  o  o   o   o   o   o     ooo   o     o  o  o    '/
     +'    o  o  o   o   oo  oo   oo   o      oo    ooo   oo  starting'/
     +'                                o                      ')
      ITERT=0  ! reset iteration counter/flag
      LUNIT=0  ! unit for scratch storage
      CFACT=1.0! factor for cut during iterations
      NCS  =0  ! number of constraints
      ICNPR=0  ! number of printouts
      LOCTOT=0 ! total number of local fits
      LOCREJ=0 ! number of rejected fits
      NSTDEV=MAX(0,MIN(NSTD,3))  ! 0 ... 4
      CFACTR=1.0
      NAGB=NAGBAR
      NALC=NALCAR
      IF(ICNLIM.GE.0) THEN
         WRITE(*,*) '                               '
         WRITE(*,*) 'Number of global parameters       ',NAGB
         WRITE(*,*) 'Number of local parameters        ',NALC
         WRITE(*,*) '   Number of standard deviations  ',NSTDEV
         WRITE(*,*) '   Number of test printouts       ',ICNLIM
      END IF
      IF(NAGB.GT.MGLOBL.OR.NALC.GT.MLOCAL) THEN
         WRITE(*,*) 'Too many parameter - STOP'
         STOP
      END IF
      IF(NSTDEV.NE.0.AND.ICNLIM.GE.0) THEN
         WRITE(*,*) 'Final cut corresponds to',NSTDEV,
     +              ' standard deviations.'
         WRITE(*,*) 'The actual cuts are made in Chisquare/Ndf at:'
         DO I=1,7
          WRITE(*,101) NDR(I),CHINDL(NSTDEV,NDR(I))
 101      FORMAT(20X,'Ndf =',I4,'   Limit =',F8.2)
         END DO
      END IF
*     reset matrices for global variables
      DO I=1,NAGB
       BGVEC(I)=0.0
       PPARM(I)=0.0   ! previous values of parameters set to zero
       DPARM(I)=0.0   ! corrections of parameters zero
       PSIGM(I)=-1.0  ! no sigma defined for parameter I
       NLNPA(I)=0     ! linear parameter by default
      END DO
      DO I=1,(NAGB*NAGB+NAGB)/2
       CGMAT(I)=0.0
      END DO
*     reset histogram
      NHIST=0
      DO I=1,51
       MHIST(I)=0
       KHIST(I)=0
       LHIST(I)=0
      END DO
*     reset matrices for local variables
      SUMM=0.0D0
      NSUM=0
      DO I=1,NALC
       BLVEC(I)=0.0
      END DO
      DO I=1,(NALC*NALC+NALC)/2
       CLMAT(I)=0.0
      END DO
      NST=0        ! reset counter for derivatives
      NFL=0        ! reset flag for derivative storage
      END
 
      SUBROUTINE PARGLO(PAR)
*     optional: initialize parameters with nonzero values
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      REAL PAR(*)
      DO I=1,NAGB
       PPARM(I)=PAR(I)
      END DO
      END
 
      SUBROUTINE PARSIG(INDEX,SIGMA)
*     optional: define sigma for single parameter
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      IF(INDEX.LT.1.OR.INDEX.GT.NAGB) RETURN
      IF(SIGMA.LT.0.0) RETURN
*     write(6,*)"IN PARSIG"
*     write(6,*)INDEX,SIGMA
      PSIGM(INDEX)=SIGMA
      END
 
      SUBROUTINE NONLIN(INDEX)
*     optional: set nonlinear flag for single parameter
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      IF(INDEX.LT.1.OR.INDEX.GT.NAGB) RETURN
      NLNPA(INDEX)=1
      END
 
      SUBROUTINE INITUN(LUN,CUTFAC)
*     optional: unit for iterations
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      LOGICAL EXS,OPN
*     test file for existence
*     write(6,*)"IN INITUN"
*     write(6,*)LUN,CUTFAC
      INQUIRE(UNIT=LUN,OPENED=OPN,IOSTAT=IOS)
      IF(IOS.NE.0) THEN
         STOP '<INITUN: Inquire error>'
      END IF
      IOSTAT=0
      IF(.NOT.OPN) THEN
         OPEN(UNIT=LUN,FORM='UNFORMATTED',STATUS='SCRATCH',IOSTAT=IOS)
         IF(IOS.NE.0) THEN
            STOP '<INITUN: Open error>'
         END IF
         IF(ICNLIM.GE.0) WRITE(*,*) 'Scratch file opened'
      END IF
      LUNIT=LUN
      CFACTR=MAX(1.0,CUTFAC)     ! > 1.0
      IF(ICNLIM.GE.0) WRITE(*,*) 'Initial cut factor is',CFACTR
      ITERT=1  ! iteration 1 is first iteration
      END
 
      SUBROUTINE CONSTF(DERCS,RHS)
*     optional: constraints
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      REAL DERCS(*)
*     write(6,*)"IN CONSTF"
*     write(6,*)(DERCS(ik),ik=1,10),RHS
      IF(NCS.GE.MCS) STOP '<INITCS> too many constraints'
      DO I=1,NAGB
       ADERCS(NAGB*NCS+I)=DERCS(I)
      END DO
      NCS=NCS+1
      ARHS(NCS)=RHS
      WRITE(*,*) 'Number of constraints increased to ',NCS
      END
 
      SUBROUTINE EQULOC(DERGB,DERLC,RRMEAS,SIGMA)
*     a single equation with its derivatives
*     DERGB(1) ... DERGB(NAGB) = derivatives w.r.t. global parameters
*     DERLC(1) ... DERLC(NALC) = derivatives w.r.t. local parameters
*     RMEAS       = measured value
*     SIGMA       = standard deviation
*     (WGHT       = weight = 1/SIGMA**2)
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      REAL DERGB(*),DERLC(*)
*     ...
*     write(6,*)"IN EQULOC"
*     write(6,*)(DERGB(io), io=1,10)
*     write(6,*)(DERLC(io), io=1,2)
*     write(6,*)RRMEAS
*     write(6,*)SIGMA
      RMEAS=RRMEAS
      IF(SIGMA.LE.0.0) THEN
         DO I=1,NALC           ! local parameters
          DERLC(I)=0.0         !  reset
         END DO
         DO I=1,NAGB           ! global parameters
          DERGB(I)=0.0         !  reset
         END DO
         RETURN
      END IF
      WGHT=1.0/SIGMA**2
      NONZER=0
      IALC=0
      IBLC=-1
      DO I=1,NALC              ! count number of local parameters
       IF(DERLC(I).NE.0.0) THEN
          NONZER=NONZER+1
          IF(IALC.EQ.0) IALC=I ! first and last index
          IBLC=I
       END IF
      END DO
      IAGB=0
      IBGB=-1
      DO I=1,NAGB              ! ... plus global parameters
       IF(DERGB(I).NE.0.0) THEN
          NONZER=NONZER+1
          IF(IAGB.EQ.0) IAGB=I ! first and last index
          IBGB=I
       END IF
      END DO
      IF(NST+NONZER+2.GE.NSTORE) THEN
         NFL=1   ! set overflow flag
         RETURN  ! ignore data
      END IF
      NST=NST+1
      INDST(NST)=0
      AREST(NST)=RMEAS
      DO I=IALC,IBLC           ! local parameters
       IF(DERLC(I).NE.0.0) THEN
          NST=NST+1
          INDST(NST)=I         ! store index ...
          AREST(NST)=DERLC(I)  ! ... and value of nonzero derivative
          DERLC(I)=0.0         ! reset
       END IF
      END DO
      NST=NST+1
      INDST(NST)=0
      AREST(NST)=WGHT
      DO I=IAGB,IBGB           ! global parameters
       IF(DERGB(I).NE.0.0) THEN
          NST=NST+1
          INDST(NST)=I         ! store index ...
          AREST(NST)=DERGB(I)  ! ... and value of nonzero derivative
          DERGB(I)=0.0         ! reset
       END IF
      END DO
      END
 
      SUBROUTINE ZERLOC(DERGB,DERLC)
*     reset derivatives
*     DERGB(1) ... DERGB(NAGB) = derivatives w.r.t. global parameters
*     DERLC(1) ... DERLC(NALC) = derivatives w.r.t. local parameters
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      REAL DERGB(*),DERLC(*)
*     write(6,*)"IN ZERLOC Anfang"
*     write(6,*)(DERLC(in),in=1,NALC)
*     write(6,*)(DERGB(il),il=1,NAGB)
      DO I=1,NALC           ! local parameters
       DERLC(I)=0.0         !  reset
      END DO
      DO I=1,NAGB           ! global parameters
       DERGB(I)=0.0         !  reset
      END DO
*     write(6,*)"IN ZERLOC Ende"
*     write(6,*)(DERLC(in),in=1,NALC)
*     write(6,*)(DERGB(il),il=1,NAGB)
      END
 
      SUBROUTINE FITLOC
*     fit after end of local block - faster(?) version
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
*      WRITE(*,*) 'FITLOC for Positive TEC'
      ICNPR=ICNPR+1
      INCRP=0
      IF(ITERT.EQ.1) THEN
         IF(NST.GT.0) THEN ! write to scratch file
            WRITE(LUNIT) NST,(INDST(I),I=1,NST),(AREST(I),I=1,NST)
         END IF
      END IF
*     reset matrices for local variables ooooooooooooooooooooooooooooooo
      SUMM=0.0D0
      NSUM=0
      DO I=1,NALC
       BLVEC(I)=0.0
      END DO
      DO I=1,(NALC*NALC+NALC)/2
       CLMAT(I)=0.0
      END DO
*     reset pointer matrix for mixed variables
      DO I=1,NAGB
       INDNZ(I)=0
      END DO
      NAGBN=0          ! actual number of global parameters
*     normal equations - symmetric matrix for local parameters
      IF(NST.LE.1) GOTO 100
      IST=0
   10 JA =0 ! new equation
      JB =0
   20 IST=IST+1
      IF(IST.GT.NST.OR.INDST(IST).EQ.0) THEN
         IF(JA.EQ.0) THEN
            JA=IST             ! first zero: measured value
         ELSE IF(JB.EQ.0) THEN
            JB=IST             ! second zero: weight
         ELSE
            IST=IST-1          ! end of equation
            RMEAS=AREST(JA)          ! use the data
*           subtract global ... from measured value
            DO J=1,IST-JB
             IJ=INDST(JB+J)           ! subtract from measured value ...
             IF(NLNPA(IJ).EQ.0) THEN  ! ... for linear parameter
             RMEAS=RMEAS-AREST(JB+J)*(PPARM(IJ)+DPARM(IJ))
             ELSE                     ! ... for nonlinear parameter
             RMEAS=RMEAS-AREST(JB+J)*DPARM(IJ)
             END IF
            END DO
*           end-subtract
            WGHT =AREST(JB)          ! ... and the weight
            DO J=1,JB-JA-1           ! number of derivatives
             IJ=INDST(JA+J)
             BLVEC(IJ)=BLVEC(IJ)+WGHT*RMEAS*AREST(JA+J)
             DO K=1,J
              IK=INDST(JA+K)
              JK=(IJ*IJ-IJ)/2+IK
              CLMAT(JK)=CLMAT(JK)+WGHT*AREST(JA+J)*AREST(JA+K)
             END DO
            END DO
 
            IF(IST.EQ.NST) GOTO 21
            GOTO 10 ! next equation
         END IF
      END IF
      IF(IST.LE.NST) GOTO 20
*     end of data - determine local fit parameters
 21   CALL SPMINV(CLMAT,BLVEC,NALC,NRANK,SCDIAG,SCFLAG)
                                                ! inversion and solution
      IF(ICNPR.LE.ICNLIM) THEN
         WRITE(*,*) '                                                  '
         WRITE(*,*) '__________________________________________________'
         WRITE(*,*) 'Printout of local fit (FITLOC) with rank=',NRANK
         WRITE(*,*) '   Result of local fit: (Index/Parameter/error)'
         WRITE(*,103) (J,BLVEC(J),SQRT(CLMAT((J*J+J/2))),J=1,JB-JA-1)
 103     FORMAT(2(I6,2G12.4))
      END IF
*     calculate residuals
      SUMM=0.0D0
      NSUM=0
*     second loop for residual calculation -----------------------------
      IST=0
   30 JA =0 ! new equation
      JB =0
   40 IST=IST+1
      IF(IST.GT.NST.OR.INDST(IST).EQ.0) THEN
         IF(JA.EQ.0) THEN
            JA=IST             ! first zero: measured value
         ELSE IF(JB.EQ.0) THEN
            JB=IST             ! second zero: weight
         ELSE
            IST=IST-1          ! end of equation
            IF(ICNPR.LE.ICNLIM.AND.INCRP.LE.11) THEN
               NDERLC=JB-JA-1
               NDERGL=IST-JB
               INCRP=INCRP+1
               WRITE(*,*) '            '
               WRITE(*,*) INCRP,'. equation: ',
     +        'measured value ',AREST(JA),' +-',1.0/SQRT(AREST(JB))
               IF(INCRP.LE.11) THEN
               WRITE(*,*)
     +        ' number of derivates (global, local):',NDERGL,',',NDERLC
               WRITE(*,*)
     +         '  Global derivatives are: (index/derivative/parvalue)'
               WRITE(*,101) (INDST(JB+J),AREST(JB+J),
     +                       PPARM(INDST(JB+J)),J=1,IST-JB)
 101           FORMAT(2(I6,2G12.4))
               WRITE(*,*)
     +         '  Local derivatives are: (index/derivative)'
               WRITE(*,102) (INDST(JA+J),AREST(JA+J),J=1,JB-JA-1)
 102           FORMAT(3(I6,G14.6))
               END IF
               IF(INCRP.EQ.11) THEN
               WRITE(*,*)  '...   (+ further equations)'
               END IF
            END IF
            RMEAS=AREST(JA)          ! use the data
*           subtract global ... from measured value
            DO J=1,IST-JB
             IJ=INDST(JB+J)           ! subtract from measured value ...
             IF(NLNPA(IJ).EQ.0) THEN  ! ... for linear parameter
             RMEAS=RMEAS-AREST(JB+J)*(PPARM(IJ)+DPARM(IJ))
             ELSE                     ! ... for nonlinear parameter
             RMEAS=RMEAS-AREST(JB+J)*DPARM(IJ)
             END IF
            END DO
*           end-subtract
            WGHT =AREST(JB)          ! ... and the weight
            DO J=1,JB-JA-1           ! number of derivatives
             IJ=INDST(JA+J)
             RMEAS=RMEAS-AREST(JA+J)*BLVEC(IJ)
            END DO
            IF(ICNPR.LE.ICNLIM) THEN
               WRITE(*,104) WGHT*RMEAS**2,RMEAS
 104           FORMAT(' Chi square contribution=',F8.2,
     +             5X,G12.4,'= residuum')
            END IF
*           residual histogram
            IHBIN=1.0+5.0*(RMEAS*SQRT(WGHT)+5.0)
            IF(IHBIN.LT. 1) IHBIN=51
            IF(IHBIN.GT.50) IHBIN=51
            LHIST(IHBIN)=LHIST(IHBIN)+1  ! residial histogram
            SUMM=SUMM+WGHT*RMEAS**2
            NSUM=NSUM+1              ! ... and count equation
            IF(IST.EQ.NST) GOTO 41
            GOTO 30 ! next equation
         END IF
      END IF
      IF(IST.LE.NST) GOTO 40
 41   NDF=NSUM-NRANK
      IF(ICNPR.LE.ICNLIM) THEN
         WRITE(*,*) 'Entry number',ICNPR
         WRITE(*,*) 'Final chi square, degrees of freedom',SUMM,NDF
      END IF
      RMS=0.0
      IF(NDF.GT.0) THEN             ! histogram entry
         RMS=SUMM/FLOAT(NDF)
         NHIST=NHIST+1
         IHBIN=1+5.0*RMS                 ! bins width 0.2
         IF(IHBIN.LT. 1) IHBIN= 1
         IF(IHBIN.GT.50) IHBIN=51
         MHIST(IHBIN)=MHIST(IHBIN)+1
      END IF
      LOCTOT=LOCTOT+1
*     make eventual cut
      IF(NSTDEV.NE.0.AND.NDF.GT.0) THEN
         CUTVAL=CHINDL(NSTDEV,NDF)*CFACTR
         IF(ICNPR.LE.ICNLIM) THEN
            WRITE(*,*) 'Reject if Chisq/Ndf=',RMS,' > ',CUTVAL
         END IF
         IF(RMS.GT.CUTVAL) THEN
            LOCREJ=LOCREJ+1
            GOTO 100
         END IF
      END IF
*     third loop for global parameters ---------------------------------
      IST=0
   50 JA =0 ! new equation
      JB =0
   60 IST=IST+1
      IF(IST.GT.NST.OR.INDST(IST).EQ.0) THEN
         IF(JA.EQ.0) THEN
            JA=IST             ! first zero: measured value
         ELSE IF(JB.EQ.0) THEN
            JB=IST             ! second zero: weight
         ELSE
            IST=IST-1          ! end of equation
            RMEAS=AREST(JA)          ! use the data
*           subtract global ... from measured value
            DO J=1,IST-JB
             IJ=INDST(JB+J)           ! subtract from measured value ...
             IF(NLNPA(IJ).EQ.0) THEN  ! ... for linear parameter
             RMEAS=RMEAS-AREST(JB+J)*(PPARM(IJ)+DPARM(IJ))
             ELSE                     ! ... for nonlinear parameter
             RMEAS=RMEAS-AREST(JB+J)*DPARM(IJ)
             END IF
            END DO
*           end-subtract
            WGHT =AREST(JB)          ! ... and the weight
*           normal equations - symmetric matrix for global parameters
            DO J=1,IST-JB
             IJ=INDST(JB+J)
             BGVEC(IJ)=BGVEC(IJ)+WGHT*RMEAS*AREST(JB+J)
             DO K=1,J
              IK=INDST(JB+K)
              JK=(IJ*IJ-IJ)/2+IK
              CGMAT(JK)=CGMAT(JK)+WGHT*AREST(JB+J)*AREST(JB+K)
             END DO
            END DO
*           normal equations - rectangular matrix for global/local pars
            DO J=1,IST-JB
             IJ=INDST(JB+J)   ! index of global variable
*            new code start
             IJN=INDNZ(IJ)    ! get index of index
             IF(IJN.EQ.0) THEN
*               new global variable - initialize matrix row
                DO K=1,NALC
                 CLCMAT(NAGBN*NALC+K)=0.0
                END DO
                NAGBN=NAGBN+1
                INDNZ(IJ)=NAGBN     ! insert pointer
                INDBK(NAGBN)=IJ     ! ... and pointer back
                IJN=NAGBN
             END IF
*            new code end
             DO K=1,JB-JA-1
              IK=INDST(JA+K)
C             JK=IK+(IJ-1)*NALC    ! old code
              JK=IK+(IJN-1)*NALC   ! new code
              CLCMAT(JK)=CLCMAT(JK)+WGHT*AREST(JB+J)*AREST(JA+K) !<=
             END DO
            END DO
            IF(IST.EQ.NST) GOTO 70
            GOTO 50 ! next equation
         END IF
      END IF
      IF(IST.LE.NST) GOTO 60
*     -------------------------------------------------------------
*     update global matrices
 70   CALL SPAVAT(CLMAT,CLCMAT,CORRM,NALC,NAGBN) ! correction matrix
      CALL SPAX(CLCMAT,BLVEC,CORRV,NAGBN,NALC)   ! correction vector
      IJN=0
      DO IN=1,NAGBN
       I=INDBK(IN)      ! get pointer back to global index
       BGVEC(I)=BGVEC(I)-CORRV(IN)
       DO JN=1,IN
        J=INDBK(JN)
        IJN=IJN+1
        IF(I.GE.J) THEN
           IJ=J+(I*I-I)/2
        ELSE
           IJ=I+(J*J-J)/2
        END IF
        CGMAT(IJ)=CGMAT(IJ)-CORRM(IJN)
       END DO
      END DO
      ENTRY KILLOC
 100  IF(ITERT.LE.1) THEN
*        histogram of used store space
         IF(NFL.EQ.0) THEN
            IBIN=1.0+50.0*FLOAT(NST)/FLOAT(NSTORE)
            IBIN=MIN(IBIN,50)
            KHIST(IBIN)=KHIST(IBIN)+1
         ELSE
            KHIST(51)=KHIST(51)+1
         END IF
      END IF
      NST=0      ! reset counter
      NFL=0      ! reset overflow flag
      END
 
      SUBROUTINE FITGLO(PAR)
*     final global fit
      REAL PAR(*)
      CHARACTER*2 PATEXT
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      IF(ITERT.LE.1) ITELIM=10 ! maximum number of iterations
      IF(ITERT.NE.0) THEN
         LTHIST=0
         DO I=1,51
          LTHIST=LTHIST+LHIST(I)
         END DO
         IF(ICNLIM.GE.0) THEN
            WRITE(*,*) '  Initial residual histogram:',
     +              '  Total',LTHIST,' entries with',LHIST(51),
     +              ' overflows'
            CALL PXHIST(LHIST,50,-5.0,+5.0) ! histogram printout
         END IF
      END IF
      KHTOT=0
      DO I=1,51
       KHTOT=KHTOT+KHIST(I)
      END DO
      IF(ICNLIM.GE.0) THEN
         WRITE(*,*) ' '
         WRITE(*,*) 'Histogram of used local fit storage:',
     +    ' total',KHTOT,' local fits with',KHIST(51),
     +    ' overflows'
      END IF
      IF(KHIST(51).NE.0.AND.ICNLIM.GE.0) THEN
         WRITE(*,*) 'Parameter NSTORE to small! Change and rerun!'
      END IF
      IF(ICNLIM.GE.0) THEN
         CALL PXHIST(KHIST,50,0.0,FLOAT(NSTORE)) ! histogram printout
      END IF
 10   IF(ICNLIM.GE.0) THEN
         WRITE(*,*) '    '
         WRITE(*,*) '... making global fit ...'
      END IF
      NN=0
      II=0          ! modify matrix acccording to PSIGM value
      DO I=1,NAGB
       LL=II
       II=II+I
       IF(PSIGM(I).EQ.0.0) THEN
          NN=NN+1        ! count number of fixed parameters
          DO J=1,NAGB
           LL=LL+1
           IF(J.GT.I) LL=LL+J-2
           CGMAT(LL)=0.0 ! reset row and column for parameter
          END DO
       ELSE IF(PSIGM(I).GT.0.0) THEN
          CGMAT(II)=CGMAT(II)+1.0/PSIGM(I)**2 ! add to diagonal
       END IF
      END DO
      II=0                ! start saving diagonal elements
      DO I=1,NAGB
       II=II+I
       DIAG(I)=CGMAT(II)  ! save original diagonal elements
      END DO
      NVAR=NAGB
      WRITE(*,*) 'Number of constraints ',NCS
      IF(NCS.NE.0) THEN   ! add constraints
         II=(NAGB*NAGB+NAGB)/2
         DO I=1,NCS  ! loop on constraints
          SUM=ARHS(I)
          DO J=1,NAGB
           CGMAT(II+J)=ADERCS(NAGB*(I-1)+J)
           SUM=SUM-ADERCS(NAGB*(I-1)+J)*(PPARM(J)+DPARM(J))
          END DO
          DO J=1,I
           CGMAT(II+NAGB+J)=0.0
          END DO
          NVAR=NVAR+1
          II  =II+NVAR
          BGVEC(NVAR)=SUM
         END DO
      END IF
*     =================================================
      CALL SPMINV(CGMAT,BGVEC,NVAR,NRANK,SCDIAG,SCFLAG)!matrix inversion
      NDEFEC=NVAR-NRANK-NN  ! rank defect
*     =====================================================
      DO I=1,NAGB
       DPARM(I)=DPARM(I)+BGVEC(I)   ! accumulate corrections
      END DO
      CALL PRTGLO(66)
      IF(ICNLIM.GE.0) THEN
      WRITE(*,*) 'The rank defect of the symmetric',NVAR,'-by-',NVAR,
     +           ' matrix is ',NDEFEC,' (should be zero).'
      END IF
      IF(ITERT.EQ.0.OR.NSTDEV.EQ.0.OR.ITERT.GE.ITELIM) GOTO 90
*     iterations
      IF(ICNLIM.GE.0) THEN
      WRITE(*,*) '                   '
      WRITE(*,*) '  Total',LOCTOT,' local fits,',LOCREJ,' rejected.'
      WRITE(*,*) '  Histogram of Chisq/Ndf:',
     +           '  Total',NHIST,' entries with',MHIST(51),
     +           ' overflows'
      CALL PXHIST(MHIST,50,0.0,10.0) ! histogram printout
      END IF
*     reset histogram
      NHIST=0
      DO I=1,51
       MHIST(I)=0
       LHIST(I)=0
      END DO
      ITERT=ITERT+1
      LOCTOT=0
      LOCREJ=0
      IF(CFACTR.NE.1.0) THEN
         CFACTR=SQRT(CFACTR)
         IF(CFACTR.LT.1.2) THEN
            CFACTR=1.0
            ITELIM=ITERT+1
         END IF
      END IF
      IF(ICNLIM.GE.0) WRITE(*,107) ITERT,CFACTR
 107  FORMAT(' Iteration',I3,' with cut factor=',F6.2)
*     reset matrices for global variables
      DO I=1,NAGB
       BGVEC(I)=0.0
      END DO
      DO I=1,(NAGB*NAGB+NAGB)/2
       CGMAT(I)=0.0
      END DO
      REWIND LUNIT
 20   READ(LUNIT,END=10) NST,(INDST(I),I=1,NST),(AREST(I),I=1,NST)
      CALL FITLOC
      GOTO 20
*     ==================================================================
 90   IF(ICNLIM.GE.0) THEN
         WRITE(*,*) '                         '
         WRITE(*,*) '         Result of fit for global parameters'
         WRITE(*,*) '         ==================================='
         WRITE(*,101)
      END IF
      II=0
      DO I=1,NAGB
       II=II+I
       ERR=SQRT(ABS(CGMAT(II)))
       IF(CGMAT((I*I+I)/2).LT.0.0) ERR=-ERR
       GCOR=0.0
       IF(CGMAT(II)*DIAG(I).GT.0.0) THEN
*         global correlation
          GCOR=SQRT(ABS(1.0-1.0/(CGMAT(II)*DIAG(I))))
       END IF
       IF(I.LE.25.OR.NAGB-I.LE.25) THEN
          PATEXT='  '
          IF(NLNPA(I).NE.0) PATEXT='nl'
          IF(ICNLIM.GE.0) WRITE(*,102) I,PATEXT,
     +        PPARM(I),PPARM(I)+DPARM(I),DPARM(I),BGVEC(I),ERR,GCOR
       END IF
       PAR(I)=PPARM(I)+DPARM(I)    ! copy of result to array in argument
      END DO
      DO I=1,NCS                   ! constraints
       IF(I.EQ.1.and.ICNLIM.GE.0) WRITE(*,*) '                      '
       SUM=0.0
       DO J=1,NAGB
        PPP=PPARM(J)+DPARM(J)
        SUM=SUM+PPP*ADERCS(NAGB*(I-1)+J)
       END DO
       IF(ICNLIM.GE.0) WRITE(*,106) I,SUM,ARHS(I),SUM-ARHS(I)
      END DO
      IF(ICNLIM.GE.0) THEN
         WRITE(*,*) '                   '
         WRITE(*,*) '  Total',LOCTOT,' local fits,',LOCREJ,' rejected.'
         WRITE(*,*) '  Histogram of RMS:',
     +              '  Total',NHIST,' entries with',MHIST(51),
     +              ' overflows'
         CALL PXHIST(MHIST,50,0.0,10.0) ! histogram printout
      END IF
      LTHIST=0
      DO I=1,51
       LTHIST=LTHIST+LHIST(I)
      END DO
      IF(ICNLIM.GE.0) THEN
         WRITE(*,*) '  Residual histogram:',
     +              '  Total',LTHIST,' entries with',LHIST(51),
     +              ' overflows'
         CALL PXHIST(LHIST,50,-5.0,+5.0) ! histogram printout
         WRITE(*,199)
      END IF
 199  FORMAT(
     +'                                                       '/
     +'              *   o   o                        o       '/
     +'                  o   o                        o       '/
     +'   o ooooo    o   o   o    oo   ooo    oo    ooo   oo  '/
     +'    o  o  o   o   o   o   o  o  o  o  o  o  o  o  o  o '/
     +'    o  o  o   o   o   o   oooo  o  o  oooo  o  o  oooo '/
     +'    o  o  o   o   o   o   o     ooo   o     o  o  o    '/
     +'    o  o  o   o   oo  oo   oo   o      oo    ooo   oo   ending.'/
     +'                                o                      ')
 101  FORMAT(1X,'   I         initial       final      differ',
     +                   '     lastcor    Error glcor'/
     +       1X,' ---     ----------- ----------- -----------',
     +                   ' ----------- -------- -----')
 102  FORMAT(1X,I4,1X,A2,1X,4F12.5,F9.5,F6.3)
 106  FORMAT(' Constraint',I2,'     Sum - RHS =',G12.5,' -',G12.5,
     +       ' = ',G12.5)
      END
 
      FUNCTION ERRPAR(I)
*     return error for parameter I
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      ERRPAR=0.0
      IF(I.LE.0.OR.I.GT.NAGB) RETURN
      II=(I*I+I)/2
      ERRPAR=SQRT(ABS(CGMAT(II)))
      IF(CGMAT(II).LT.0.0) ERRPAR=ERRPAR
      END
 
      FUNCTION CORPAR(I,J)
*     return correlation between parameters I and J
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      CORPAR=0.0
      IF(I.LE.0.OR.I.GT.NAGB) RETURN
      IF(J.LE.0.OR.J.GT.NAGB) RETURN
      IF(I.EQ.J)              RETURN
      II=(I*I+I)/2
      JJ=(J*J+J)/2
      K=MAX(I,J)
      IJ=(K*K-K)/2+MIN(I,J)
      ERR=SQRT(ABS(CGMAT(II)*CGMAT(JJ)))
      IF(ERR.NE.0.0) CORPAR=CGMAT(IJ)/ERR
      END
 
      SUBROUTINE PRTGLO(LUN)
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      CHARACTER*2 PATEXT
*     ...
      LUP=LUN
      IF(LUP.EQ.0) LUP=6
      WRITE(LUP,*) '         Result of fit for global parameters'
      WRITE(LUP,*) '         ==================================='
      WRITE(LUP,101)
      II=0
      DO I=1,NAGB
       II=II+I
       ERR=SQRT(ABS(CGMAT(II)))
       IF(CGMAT(II).LT.0.0) ERR=-ERR
       GCOR=0.0
       IF(CGMAT(II)*DIAG(I).GT.0.0) THEN
*         global correlation
          GCOR=SQRT(ABS(1.0-1.0/(CGMAT(II)*DIAG(I))))
       END IF
       PATEXT='  '
       IF(NLNPA(I).NE.0) PATEXT='nl'
       WRITE(LUP,102) I,PATEXT,
     +        PPARM(I),PPARM(I)+DPARM(I),DPARM(I),BGVEC(I),ERR,GCOR
      END DO
 101  FORMAT(1X,'   I         initial       final      differ',
     +                   '     lastcor    Error glcor'/
     +       1X,' ---     ----------- ----------- -----------',
     +                   ' ----------- -------- -----')
 102  FORMAT(1X,I4,1X,A2,1X,4F12.5,F9.5,F6.3)
      END
 
 
      SUBROUTINE SPMINV(V,B,N,NRANK,DIAG,FLAG)
*     obtain solution of a system of linear equations with symmetric
*     matrix and the inverse.
*
*                    - - -
*        CALL SPMINV(V,B,N,NRANK,...,...)      solve  V * X = B
*                    - -   -----
*
*           V = symmetric N-by-N matrix in symmetric storage mode
*               V(1) = V11, V(2) = V12, V(3) = V22, V(4) = V13, . . .
*               replaced by inverse matrix
*           B = N-vector, replaced by solution vector
*
*     DIAG(N) =  double precision scratch array
*     FLAG(N) =  logical scratch array
*
*     Method of solution is by elimination selecting the  pivot  on  the
*     diagonal each stage. The rank of the matrix is returned in  NRANK.
*     For NRANK ne N, all remaining  rows  and  cols  of  the  resulting
*     matrix V and the corresponding elements of  B  are  set  to  zero.
*
      DOUBLE PRECISION V(*),B(N),DIAG(N),VKK,VJK,EPS
      LOGICAL          FLAG(*)
      PARAMETER     (EPS=1.0D-10)
*     ...
      DO I=1,N
       FLAG(I)=.TRUE.             ! reset flags
       DIAG(I)=ABS(V((I*I+I)/2))  ! save abs of diagonal elements
      END DO
      NRANK=0
      DO I=1,N                    ! start of loop
       K  =0
       JJ =0
       KK =0
       VKK=0.0D0
       DO J=1,N                   ! search for pivot
        JJ=JJ+J
        IF(FLAG(J)) THEN          ! not used so far
           IF(ABS(V(JJ)).GT.MAX(ABS(VKK),EPS*DIAG(J))) THEN
              VKK=V(JJ)           ! pivot (candidate)
              K  =J               ! index of pivot
              KK =JJ              ! index of diagonal element
           END IF
        END IF
       END DO
       IF(K.NE.0) THEN            ! pivot found
          NRANK=NRANK+1           ! increase rank and ...
          FLAG(K)=.FALSE.         ! ... reset flag
          VKK    =1.0/VKK
          V(KK)  =-VKK
          B(K)   =B(K)*VKK
          JK     =KK-K
          JL     =0
          DO J=1,N                ! elimination
           IF(J.EQ.K) THEN
              JK=KK
              JL=JL+J
           ELSE
              IF(J.LT.K) THEN
                 JK=JK+1
              ELSE
                 JK=JK+J-1
              END IF
              VJK  =V(JK)
              V(JK)=VKK*VJK
              B(J) =B(J)-B(K)*VJK
              LK   =KK-K
              DO L=1,J
               JL=JL+1
               IF(L.EQ.K) THEN
                  LK=KK
               ELSE
                  IF(L.LT.K) THEN
                     LK=LK+1
                  ELSE
                     LK=LK+L-1
                  END IF
                  V(JL)=V(JL)-V(LK)*VJK
               END IF
              END DO
           END IF
          END DO
       ELSE
          DO K=1,N
           IF(FLAG(K)) THEN
              B(K)=0.0D0       ! clear vector element
              DO J=1,K
               IF(FLAG(J)) V((K*K-K)/2+J)=0.0D0  ! clear matrix row/col
              END DO
           END IF
          END DO
          GOTO 10
       END IF
      END DO             ! end of loop
 10   DO IJ=1,(N*N+N)/2
       V(IJ)=-V(IJ)      ! finally reverse sign of all matrix elements
      END DO
      END
 
      SUBROUTINE SPAVAT(V,A,W,N,M)
*     multiply symmetric N-by-N matrix from the left with general M-by-N
*     matrix and from the right with the transposed of the same  general
*     matrix  to  form  symmetric  M-by-M   matrix   (used   for   error
*     propagation).
*
*                    - -   - -                                   T
*        CALL SPAVAT(V,A,W,N,M)         W   =   A   *   V   *   A
*                        -             M*M     M*N     N*N     N*M
*
*        where V = symmetric N-by-N matrix
*              A = general N-by-M matrix
*              W = symmetric M-by-M matrix
*
      DOUBLE PRECISION V,A,W,CIK
      DIMENSION V(*),A(*),W(*)
*     ...
      DO I=1,(M*M+M)/2
       W(I)=0.0                ! reset output matrix
      END DO
      IL=-N
      IJS=0
      DO I=1,M                 ! do I
       IJS=IJS+I-1             !
       IL=IL+N                 !
       LKL=0                   !
       DO K=1,N                !   do K
        CIK=0.0D0              !
        LKL=LKL+K-1            !
        LK=LKL                 !
        DO L=1,K               !     do L
         LK=LK+1               !     .
         CIK=CIK+A(IL+L)*V(LK) !     .
        END DO                 !     end do L
        DO L=K+1,N             !     do L
         LK=LK+L-1             !     .
         CIK=CIK+A(IL+L)*V(LK) !     .
        END DO                 !     end do L
        JK=K                   !
        IJ=IJS                 !
        DO J=1,I               !     do J
         IJ=IJ+1               !     .
         W(IJ)=W(IJ)+CIK*A(JK) !     .
         JK=JK+N               !     .
        END DO                 !     end do J
       END DO                  !   end do K
      END DO                   ! end do I
      END
 
      SUBROUTINE SPAX(A,X,Y,M,N)
*     multiply general M-by-N matrix A and N-vector X
*
*                   - -   - -
*        CALL  SPAX(A,X,Y,M,N)          Y   :=   A   *    X
*                       -               M       M*N       N
*
*        where A = general M-by-N matrix (A11 A12 ... A1N  A21 A22 ...)
*              X = N vector
*              Y = M vector
*
      DOUBLE PRECISION A(*),X(*),Y(*)
*     ...
      IJ=0
      DO I=1,M
       Y(I)=0.0D0
       DO J=1,N
        IJ=IJ+1
        Y(I)=Y(I)+A(IJ)*X(J)
       END DO
      END DO
      END
 
 
      SUBROUTINE PXHIST(INC,N,XA,XB)
*     print X histogram
      PARAMETER (MAXPL=70)
      INTEGER INC(N),NUM(MAXPL)
      PARAMETER (NP=MAXPL/10+1)
      REAL P(NP)
      EQUIVALENCE (NVAL,FVAL)
      CHARACTER TEXT(10)*130  ! 10 rows of X's
      CHARACTER XCHAR(NP)*8,ECHAR*4
      CHARACTER CHN(0:10)*1
      DATA CHN/'0','1','2','3','4','5','6','7','8','9','0'/
*     ...
      IF(N.LT.10) RETURN
      NTYP=0
      DO I=1,N
       IF(INC(I).NE.0) THEN
          NVAL=INC(I)
          IF(ABS(FVAL).LT.1.0E-30) THEN
             NTYP=NTYP+1  ! integer
          ELSE
             NTYP=NTYP-1  ! floating point
          END IF
       END IF
      END DO
      M=1        ! M bins combined to 1
  10  NRED=N/M
      IF(NRED.GT.MAXPL) THEN
         M=M+M   ! M = power of 2
         GOTO 10
      END IF
      NRED=N/M   ! reduced number of bins
      NK=0
      IF(NTYP.GE.0) THEN               ! integer
         DO I=1,N,M ! add M bins together
          MUM=0
          DO K=I,MIN(I+M-1,N)
           MUM=MUM+INC(K)
          END DO
          NK=NK+1
          NUM(NK)=MUM  ! copy to array NUM(.)
         END DO
      ELSE                            ! floating point
         DO I=1,N,M ! add M bins together
          SUM=0
          DO K=I,MIN(I+M-1,N)
           NVAL=INC(K)
           SUM=SUM+FVAL
          END DO
          NK=NK+1
          NUM(NK)=SUM+0.5  ! copy to array NUM(.)
         END DO
      END IF
      INMAX=1        ! find maximum bin
      DO I=1,NK
       IF(NUM(I).GT.NUM(INMAX)) INMAX=I
      END DO
      INH=NUM(INMAX) ! maximum bin content
      IDIV=1+(INH-1)/10     ! X equivalent
      NR=INH/IDIV    ! number of X lines
      DO L=1,NR
       TEXT(L)=' '   ! blank text line
      END DO
      DO K=1,NRED
       LR=NUM(K)/IDIV
       IF(LR.NE.0) THEN
          DO L=1,LR
           TEXT(L)(K:K)='X'
          END DO
       ELSE
          IF(NUM(K).NE.0) TEXT(1)(I:I)='.'
       END IF
      END DO
      DO L=NR,1,-1  ! print X's
       WRITE(*,103) TEXT(L)(1:NRED)
      END DO
      N10=1+(NRED-1)/10
      TEXT(1)=' '
      DO I=1,N10
       IUP=MIN(10*I,NRED)
       TEXT(1)(10*I-9:IUP)='----+----+'
       IF(IUP.EQ.10*I) TEXT(1)(IUP:IUP)=CHN(I)
      END DO
      WRITE(*,103) TEXT(1)(1:NRED)
      NAP=NRED/10+1
      IF(NAP*10-10.GT.NRED) NAP=NAP-1
      DO L=1,NAP
       P(L)=XA+FLOAT(10*L-10)*(XB-XA)/FLOAT(NRED)
      END DO
      CALL CHFMT(P,NAP,XCHAR,ECHAR)
      WRITE(*,104) (XCHAR(L),L=1,NAP)
      IF(ECHAR.NE.' ') WRITE(*,105) (ECHAR,L=1,NAP)
      WRITE(*,103) ' '
      NT=0
      DO K=1,10
       TEXT(K)=' '
       DO I=1,NRED
        IF(NUM(I).NE.0) THEN
           NT=K
           L=MOD(NUM(I),10)
           NUM(I)=NUM(I)/10
           TEXT(K)(I:I)=CHN(L)
        END IF
       END DO
      END DO
      DO L=NT,1,-1
       WRITE(*,103) TEXT(L)(1:NRED)
      END DO
      WRITE(*,103) ' '
  103 FORMAT(5X,A)
  104 FORMAT(12(A8,2X)/)
  105 FORMAT(3X,12(A4,6X))
      END
 
      SUBROUTINE CHFMT(X,N,XCHAR,ECHAR)
*     prepare printout of array of real numbers as character strings
*
*                  - -
*        CALL CHFMT(X,N,XCHAR,ECHAR)
*                      ----- -----
*     where X( )     = array of n real values
*           XCHAR( ) = array of n character*8 variables
*           ECHAR    = character*4 variable
*
*     CHFMT converts an array of  n  real  values  into n character*  8
*     variables (containing the values as text) and  a  common  exponent
*     for printing. unneccessary zeros are suppressed.
*
*
*     example: x(1)=1200.0, x(2)=1700.0 with n=2 are converted to
*               xchar(1)='  1.2   ', xchar(2)='  1.7   ', echar='e 03'
*
*
      REAL X(*)
      CHARACTER*1 CH(10)
      CHARACTER*4 ECHAR
      CHARACTER*8 XCHAR(N),FXF,SXF,NULL
      DATA NULL/'00000000'/,CH/'0','1','2','3','4','5','6','7','8','9'/
*     ...
*     determine factor fc, so that fc*xmax is 5 digit number
      IP=0
      XM=0.0
      DO I=1,N
       XM=AMAX1(XM,ABS(X(I)))
      END DO
      IF(XM.NE.0.0) THEN
         JP=104-IFIX(ALOG10(ABS(XM))+100.04)
         FC=10.0**JP
      ELSE
         JP=5
         FC=1.0
      END IF
*     store digits as characters and find jm = first nonzero digit
      IM=6
      DO I=1,N
       FXF=NULL
       IJ=FC*ABS(X(I))+0.5
       JM=6
       IF(IJ.NE.0) THEN
          DO J=1,5
           JN=MOD(IJ,10)
           IJ=IJ/10
           IF(JN.NE.0.AND.JM.EQ.6) JM=J
           FXF(J:J)=CH(JN+1)
          END DO
          IM=MIN(IM,JM)
       END IF
       XCHAR(I)=FXF
      END DO
      JM=IM
*     determine exponent as a multiple of 3
   32 IF(JP.LT.1) THEN
         JP=JP+3
         IP=IP+3
         GOTO 32
      END IF
   34 IF(JP.GT.JM+4.OR.JP.GE.8) THEN
         JP=JP-3
         IP=IP-3
         GOTO 34
      END IF
*     loop to convert to print format
      JA=MIN(JM,JP)
      JB=MAX(6,JP+1)
      DO 90 I=1,N
      FXF=XCHAR(I)
      SXF=' '
      IB=7+(JB-JA)/2
      DO 80 J=JA,JB
      IF(FXF(J:J).NE.CH(1)) GOTO 70
      IF(J.GT.JP+1) GOTO 50
      IF(FXF.NE.NULL.OR.J.GE.JP) GOTO 70
      IB=IB-1
      GOTO 80
   50 DO K=J,JB
       IF(FXF(K:K).NE.CH(1)) GOTO 70
      END DO
      GOTO 80
*     insert digit
   70 IB=IB-1
      SXF(IB:IB)=FXF(J:J)
      IF(J.EQ.JP) THEN
*     insert decimal dot
         IB=IB-1
         SXF(IB:IB)='.'
      END IF
   80 CONTINUE
*     insert - sign
      IF(X(I).LT.0.0) SXF(IB-1:IB-1)='-'
   90 XCHAR(I)=SXF
*     prepare print format for exponent
      ECHAR=' '
      IF(IP.NE.0) THEN
         ECHAR='E 0 '
         IF(IP.LE.0) THEN
            ECHAR(2:2)='-'
            IP=IABS(IP)
         END IF
         J=MOD(IP,10)
         ECHAR(4:4)=CH(J+1)
         IP=(IP-J)/10
         IF(IP.NE.0) THEN
            J=MOD(IP,10)
            ECHAR(3:3)=CH(J+1)
         END IF
      END IF
      END
 
      FUNCTION CHINDL(N,ND)
*     return limit in chi^2/ND for N sigmas (N=1, 2 or 3)
      REAL PN(3),SN(3),TABLE(30,3)
      DATA PN/0.31731,0.0455002785,2.69985E-3/         ! probabilities
      DATA SN/0.47523,1.690140,2.782170/
      DATA TABLE/
     +  1.0000, 1.1479, 1.1753, 1.1798, 1.1775, 1.1730, 1.1680, 1.1630,
     +  1.1581, 1.1536, 1.1493, 1.1454, 1.1417, 1.1383, 1.1351, 1.1321,
     +  1.1293, 1.1266, 1.1242, 1.1218, 1.1196, 1.1175, 1.1155, 1.1136,
     +  1.1119, 1.1101, 1.1085, 1.1070, 1.1055, 1.1040,
     +  4.0000, 3.0900, 2.6750, 2.4290, 2.2628, 2.1415, 2.0481, 1.9736,
     +  1.9124, 1.8610, 1.8171, 1.7791, 1.7457, 1.7161, 1.6897, 1.6658,
     +  1.6442, 1.6246, 1.6065, 1.5899, 1.5745, 1.5603, 1.5470, 1.5346,
     +  1.5230, 1.5120, 1.5017, 1.4920, 1.4829, 1.4742,
     +  9.0000, 5.9146, 4.7184, 4.0628, 3.6410, 3.3436, 3.1209, 2.9468,
     +  2.8063, 2.6902, 2.5922, 2.5082, 2.4352, 2.3711, 2.3143, 2.2635,
     +  2.2178, 2.1764, 2.1386, 2.1040, 2.0722, 2.0428, 2.0155, 1.9901,
     +  1.9665, 1.9443, 1.9235, 1.9040, 1.8855, 1.8681/
*     ...
      IF(ND.LT.1) THEN
         CHINDL=0.0
      ELSE
         M=MAX(1,MIN(N,3))         ! 1, 2 or 3 sigmas
         IF(ND.LE.30) THEN
            CHINDL=TABLE(ND,M)     ! from table
         ELSE                      ! approximation for ND > 30
            CHINDL=(SN(M)+SQRT(FLOAT(ND+ND-1)))**2/FLOAT(ND+ND)
         END IF
      END IF
      END
 
      SUBROUTINE FITMUT(NVEC,VEC)
*     get matrix information out
*     ------------------------------------------------------------------
*     Basic dimension parameters
      PARAMETER (MGLOBL=1400,MLOCAL=100,NSTORE=10000,MCS=10) ! dimensions
      PARAMETER (MGL=MGLOBL+MCS)                     ! derived parameter
*     derived parameters
      PARAMETER (MSYMGB  =(MGLOBL*MGLOBL+MGLOBL)/2,
     +           MSYM    =(MGL*MGL+MGL)/2,
     +           MSYMLC=(MLOCAL*MLOCAL+MLOCAL)/2,
     +           MRECTA= MGLOBL*MLOCAL,
     +           MGLOCS= MGLOBL*MCS,
     +           MSYMCS= (MCS*MCS+MCS)/2 )
      DOUBLE PRECISION CGMAT,CLMAT,CLCMAT,BGVEC,BLVEC,
     +                 CORRM,CORRV,SUMM,DIAG,SCDIAG,PPARM,DPARM,ADERCS,
     +                 ARHS
      LOGICAL SCFLAG
      COMMON/LSQRED/CGMAT(MSYM),CLMAT(MSYMLC),CLCMAT(MRECTA),
     +              DIAG(MGL),BGVEC(MGL),BLVEC(MLOCAL),
     +              CORRM(MSYMGB),CORRV(MGLOBL),PSIGM(MGLOBL),
     +              PPARM(MGLOBL),ADERCS(MGLOCS),ARHS(MCS),
     +              DPARM(MGLOBL),
     +              SCDIAG(MGLOBL),SUMM,SCFLAG(MGLOBL),
     +              INDGB(MGLOBL),INDLC(MLOCAL),LOCTOT,LOCREJ,
     +              NAGB,NALC,NSUM,NHIST,MHIST(51),KHIST(51),LHIST(51),
     +              NST,NFL,INDST(NSTORE),AREST(NSTORE),ITERT,LUNIT,NCS,
     +              NLNPA(MGLOBL),NSTDEV,CFACTR,ICNPR,ICNLIM,
     +              INDNZ(MGLOBL),INDBK(MGLOBL)
*     ------------------------------------------------------------------
      REAL VEC(*)
*     ...
      NVEC=(NAGB*NAGB+NAGB)/2+NAGB
      DO I=1,NVEC-NAGB
       VEC(I)=CGMAT(I)
      END DO
      DO I=1,NAGB
       VEC(NVEC-NAGB+I)=BGVEC(I)
      END DO
      WRITE(*,*) 'FITMUT ',NVEC,(VEC(I),I=1,NVEC)
      RETURN
      ENTRY FITMIN(NVEC,VEC)
*     insert information
      IF(NVEC.NE.(NAGB*NAGB+NAGB)/2+NAGB) THEN
         WRITE(*,*) ' Wrong dimensions in FITMUT/FITMIN'
         WRITE(*,*) ' Argument NVEC =',NVEC
         WRITE(*,*) ' Expected NVEC =',(NAGB*NAGB+NAGB)/2+NAGB
         STOP 'FITMIN'
      END IF
      DO I=1,NVEC-NAGB
       CGMAT(I)=VEC(I)
      END DO
      DO I=1,NAGB
       BGVEC(I)=VEC(NVEC-NAGB+I)
      END DO
      END
 
 
 
 
 
 
 
 
 
 
 
