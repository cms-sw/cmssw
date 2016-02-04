C=====================================================================
C  This routine was modified from the PYTHIA 6.420 code, which is
C              (C) Torbjorn Sjostrand, Lund 2008.
C
C  The modifications are part of the HARDCOL package for 
C  hard color singlet exchange, and refer to PYTHIA version 6.420.
C  Modifications implemented by Rikard Enberg, 2001-2003 and 2008.  
C  See http://www.isv.uu.se/thep/hardcol/      
C
C The modification for PYTHIA v6.420 was implemented by Sheila Amaral
C Modified: 16 Oct 2009
C=====================================================================

C***********************************************************************
 
C...PYSIGH
C...Differential matrix elements for all included subprocesses
C...Note that what is coded is (disregarding the COMFAC factor)
C...1) for 2 -> 1 processes: s-hat/pi*d(sigma-hat), where,
C...when d(sigma-hat) is given in the zero-width limit, the delta
C...function in tau is replaced by a (modified) Breit-Wigner:
C...1/pi*s*H_res/((s*tau-m_res^2)^2+H_res^2),
C...where H_res = s-hat/m_res*Gamma_res(s-hat);
C...2) for 2 -> 2 processes: (s-hat)**2/pi*d(sigma-hat)/d(t-hat);
C...i.e., dimensionless quantities
C...3) for 2 -> 3 processes: abs(M)^2, where the total cross-section is
C...Integral abs(M)^2/(2shat') * (prod_(i=1)^3 d^3p_i/((2pi)^3*2E_i)) *
C...(2pi)^4 delta^4(P - sum p_i)
C...COMFAC contains the factor pi/s (or equivalent) and
C...the conversion factor from GeV^-2 to mb
 
      SUBROUTINE PYSIGH(NCHN,SIGS)
 
C...Double precision and integer declarations
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Parameter statement to help give large particle numbers.
      PARAMETER (KSUSY1=1000000,KSUSY2=2000000,KTECHN=3000000,
     &KEXCIT=4000000,KDIMEN=5000000)
C...Commonblocks
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)
      COMMON/PYINT4/MWID(500),WIDS(500,5)
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON/PYINT7/SIGT(0:6,0:6,0:5)
      COMMON/PYMSSM/IMSS(0:99),RMSS(0:99)
      COMMON/PYSSMT/ZMIX(4,4),UMIX(2,2),VMIX(2,2),SMZ(4),SMW(2),
     &SFMIX(16,4),ZMIXI(4,4),UMIXI(2,2),VMIXI(2,2)
      COMMON/PYTCSM/ITCM(0:99),RTCM(0:99)
      COMMON/PYPUED/IUED(0:99),RUED(0:99)
      COMMON/PYSGCM/ISUB,ISUBSV,MMIN1,MMAX1,MMIN2,MMAX2,MMINA,MMAXA,
     &KFAC(2,-40:40),COMFAC,FACK,FACA,SH,TH,UH,SH2,TH2,UH2,SQM3,SQM4,
     &SHR,SQPTH,TAUP,BE34,CTH,X(2),SQMZ,SQMW,GMMZ,GMMW,
     &AEM,AS,XW,XW1,XWC,XWV,POLL,POLR,POLLL,POLRR
      COMMON/PYTCCO/COEFX(194:380,2)
      SAVE /PYJETS/,/PYDAT1/,/PYDAT2/,/PYDAT3/,/PYSUBS/,/PYPARS/,
     &/PYINT1/,/PYINT2/,/PYINT3/,/PYINT4/,/PYINT5/,/PYINT7/,
     &/PYMSSM/,/PYSSMT/,/PYTCSM/,/PYPUED/,/PYSGCM/,/PYTCCO/
C...Local arrays and complex variables
      DIMENSION XPQ(-25:25)
 
C...Map of processes onto which routine to call
C...in order to evaluate cross section:
C...0 = not implemented;
C...1 = standard QCD (including photons);
C...2 = heavy flavours;
C...3 = W/Z;
C...4 = Higgs (2 doublets; including longitudinal W/Z scattering);
C...5 = SUSY;
C...6 = Technicolor;
C...7 = exotics (Z'/W'/LQ/R/f*/H++/Z_R/W_R/G*).
C...8 = Universal Extra Dimensions
      DIMENSION MAPPR(500)
      DATA (MAPPR(I),I=1,180)/
     &    3,  3,  4,  0,  4,  0,  0,  4,  0,  1,
     1    1,  1,  1,  1,  3,  3,  0,  1,  3,  3,
     2    0,  3,  3,  4,  3,  4,  0,  1,  1,  3,
     3    3,  4,  1,  1,  3,  3,  0,  0,  0,  0,
     4    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     5    0,  0,  1,  1,  0,  0,  0,  1,  0,  0,
     6    0,  0,  0,  0,  0,  0,  0,  1,  3,  3,
     7    4,  4,  4,  0,  0,  4,  4,  0,  0,  1,
     8    2,  2,  2,  2,  2,  2,  2,  2,  2,  0,
     9    1,  1,  1,  1,  1,  1,  0,  0,  1,  0,
     &    0,  4,  4,  2,  2,  2,  2,  2,  0,  4,
     1    4,  4,  4,  1,  1,  0,  0,  0,  0,  0,
     2    4,  4,  4,  4,  0,  0,  0,  0,  0,  0,
     3    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     4    7,  7,  4,  7,  7,  7,  7,  7,  6,  0,
     5    4,  4,  4,  0,  0,  4,  4,  4,  0,  0,
     6    4,  7,  7,  7,  6,  6,  7,  7,  7,  0,
     7    4,  4,  4,  4,  0,  4,  4,  4,  4,  0/
      DATA (MAPPR(I),I=181,500)/
     8    4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
     9    6,  6,  6,  6,  6,  0,  0,  0,  0,  0,
     &    100*5,
     &    5,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     &    8,  8,  8,  8,  8,  8,  8,  8,  8,  0,
     1    20*0,
     4    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
     5    7,  7,  7,  7,  0,  0,  0,  0,  0,  0,
     6    6,  6,  6,  6,  6,  6,  6,  6,  0,  6,
     7    6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
     8    6,  6,  6,  6,  6,  6,  6,  6,  0,  0,
     9    7,  7,  7,  7,  7,  0,  0,  0,  0,  0,
     &    4,  4,  18*0,
     2    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     3    2,  2,  2,  2,  2,  2,  2,  2,  2,  0,
     4     20*0,
     6    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
     7    2,  2,  2,  2,  2,  2,  2,  2,  2,  0,
     8     20*0/
 
C=====================================================================
C BEGIN HARDCOL MODIFICATION
C=====================================================================
      EXTERNAL HCS_XS
C=====================================================================
C END HARDCOL MODIFICATION
C=====================================================================


C...Reset number of channels and cross-section
      NCHN=0
      SIGS=0D0
 
C...Read process to consider.
      ISUB=MINT(1)
      ISUBSV=ISUB
      MAP=MAPPR(ISUB)
 
C...Read kinematical variables and limits
      ISTSB=ISET(ISUBSV)
      TAUMIN=VINT(11)
      YSTMIN=VINT(12)
      CTNMIN=VINT(13)
      CTPMIN=VINT(14)
      TAUPMN=VINT(16)
      TAU=VINT(21)
      YST=VINT(22)
      CTH=VINT(23)
      XT2=VINT(25)
      TAUP=VINT(26)
      TAUMAX=VINT(31)
      YSTMAX=VINT(32)
      CTNMAX=VINT(33)
      CTPMAX=VINT(34)
      TAUPMX=VINT(36)
 
C...Derive kinematical quantities
      TAUE=TAU
      IF(ISTSB.GE.3.AND.ISTSB.LE.5) TAUE=TAUP
      X(1)=SQRT(TAUE)*EXP(YST)
      X(2)=SQRT(TAUE)*EXP(-YST)
      IF(MINT(45).EQ.2.AND.ISTSB.GE.1) THEN
        IF(X(1).GT.1D0-1D-7) RETURN
      ELSEIF(MINT(45).EQ.3) THEN
        X(1)=MIN(1D0-1.1D-10,X(1))
      ENDIF
      IF(MINT(46).EQ.2.AND.ISTSB.GE.1) THEN
        IF(X(2).GT.1D0-1D-7) RETURN
      ELSEIF(MINT(46).EQ.3) THEN
        X(2)=MIN(1D0-1.1D-10,X(2))
      ENDIF
      SH=MAX(1D0,TAU*VINT(2))
      SQM3=VINT(63)
      SQM4=VINT(64)
      RM3=SQM3/SH
      RM4=SQM4/SH
      BE34=SQRT(MAX(0D0,(1D0-RM3-RM4)**2-4D0*RM3*RM4))
      RPTS=4D0*VINT(71)**2/SH
      BE34L=SQRT(MAX(0D0,(1D0-RM3-RM4)**2-4D0*RM3*RM4-RPTS))
      RM34=MAX(1D-20,2D0*RM3*RM4)
      RSQM=1D0+RM34
      IF(2D0*VINT(71)**2/MAX(1D0,VINT(21)*VINT(2)).LT.0.0001D0)
     &RM34=MAX(RM34,2D0*VINT(71)**2/MAX(1D0,VINT(21)*VINT(2)))
      RTHM=(4D0*RM3*RM4+RPTS)/(1D0-RM3-RM4+BE34L)
      IF(ISTSB.EQ.0) THEN
        TH=VINT(45)
        UH=-0.5D0*SH*MAX(RTHM,1D0-RM3-RM4+BE34*CTH)
        SQPTH=MAX(VINT(71)**2,0.25D0*SH*BE34**2*VINT(59)**2)
      ELSE
C...Kinematics with incoming masses tricky: now depends on how
C...subprocess has been set up w.r.t. order of incoming partons.
        RM1=0D0
        IF(MINT(15).EQ.22.AND.VINT(3).LT.0D0) RM1=-VINT(3)**2/SH
        RM2=0D0
        IF(MINT(16).EQ.22.AND.VINT(4).LT.0D0) RM2=-VINT(4)**2/SH
        IF(ISUB.EQ.35) THEN
          RM2=MIN(RM1,RM2)
          RM1=0D0
        ENDIF
        BE12=SQRT(MAX(0D0,(1D0-RM1-RM2)**2-4D0*RM1*RM2))
        TUCOM=(1D0-RM1-RM2)*(1D0-RM3-RM4)
        TH=-0.5D0*SH*MAX(RTHM,TUCOM-2D0*RM1*RM4-2D0*RM2*RM3-
     &  BE12*BE34*CTH)
        UH=-0.5D0*SH*MAX(RTHM,TUCOM-2D0*RM1*RM3-2D0*RM2*RM4+
     &  BE12*BE34*CTH)
        SQPTH=MAX(VINT(71)**2,0.25D0*SH*BE34**2*(1D0-CTH**2))
      ENDIF
      SHR=SQRT(SH)
      SH2=SH**2
      TH2=TH**2
      UH2=UH**2
 
C...Choice of Q2 scale for hard process (e.g. alpha_s).
      IF(ISTSB.EQ.1.OR.ISTSB.EQ.3.OR.ISTSB.EQ.5) THEN
        Q2=SH
      ELSEIF(ISTSB.EQ.8) THEN
        IF(MINT(107).EQ.4) Q2=VINT(307)
        IF(MINT(108).EQ.4) Q2=VINT(308)
      ELSEIF(MOD(ISTSB,2).EQ.0.OR.ISTSB.EQ.9) THEN
        Q2IN1=0D0
        IF(MINT(11).EQ.22.AND.VINT(3).LT.0D0) Q2IN1=VINT(3)**2
        Q2IN2=0D0
        IF(MINT(12).EQ.22.AND.VINT(4).LT.0D0) Q2IN2=VINT(4)**2
        IF(MSTP(32).EQ.1) THEN
          Q2=2D0*SH*TH*UH/(SH**2+TH**2+UH**2)
        ELSEIF(MSTP(32).EQ.2) THEN
          Q2=SQPTH+0.5D0*(SQM3+SQM4)
        ELSEIF(MSTP(32).EQ.3) THEN
          Q2=MIN(-TH,-UH)
        ELSEIF(MSTP(32).EQ.4) THEN
          Q2=SH
        ELSEIF(MSTP(32).EQ.5) THEN
          Q2=-TH
        ELSEIF(MSTP(32).EQ.6) THEN
          XSF1=X(1)
          IF(ISTSB.EQ.9) XSF1=X(1)/VINT(143)
          XSF2=X(2)
          IF(ISTSB.EQ.9) XSF2=X(2)/VINT(144)
          Q2=(1D0+XSF1*Q2IN1/SH+XSF2*Q2IN2/SH)*
     &    (SQPTH+0.5D0*(SQM3+SQM4))
        ELSEIF(MSTP(32).EQ.7) THEN
          Q2=(1D0+Q2IN1/SH+Q2IN2/SH)*(SQPTH+0.5D0*(SQM3+SQM4))
        ELSEIF(MSTP(32).EQ.8) THEN
          Q2=SQPTH+0.5D0*(Q2IN1+Q2IN2+SQM3+SQM4)
        ELSEIF(MSTP(32).EQ.9) THEN
          Q2=SQPTH+Q2IN1+Q2IN2+SQM3+SQM4
        ELSEIF(MSTP(32).EQ.10) THEN
          Q2=VINT(2)
C..Begin JA 040914
        ELSEIF(MSTP(32).EQ.11) THEN
          Q2=0.25*(SQM3+SQM4+2*SQRT(SQM3*SQM4))
        ELSEIF(MSTP(32).EQ.12) THEN
          Q2=PARP(193)
C..End JA
        ELSEIF(MSTP(32).EQ.13) THEN
          Q2=SQPTH
        ENDIF
        IF(MINT(35).LE.2.AND.ISTSB.EQ.9) Q2=SQPTH
        IF(ISTSB.EQ.9.AND.MSTP(82).GE.2) Q2=Q2+
     &  (PARP(82)*(VINT(1)/PARP(89))**PARP(90))**2
      ENDIF
 
C...Choice of Q2 scale for parton densities.
      Q2SF=Q2
C..Begin JA 040914
      IF(MSTP(32).EQ.12.AND.(MOD(ISTSB,2).EQ.0.OR.ISTSB.EQ.9)
     &     .OR.MSTP(39).EQ.8.AND.(ISTSB.GE.3.AND.ISTSB.LE.5))
     &     Q2=PARP(194)
C..End JA
      IF(ISTSB.GE.3.AND.ISTSB.LE.5) THEN
        Q2SF=PMAS(23,1)**2
        IF(ISUB.EQ.8.OR.ISUB.EQ.76.OR.ISUB.EQ.77.OR.ISUB.EQ.124.OR.
     &  ISUB.EQ.174.OR.ISUB.EQ.179.OR.ISUB.EQ.351) Q2SF=PMAS(24,1)**2 
        IF(ISUB.EQ.352) Q2SF=PMAS(PYCOMP(9900024),1)**2
        IF(ISUB.EQ.121.OR.ISUB.EQ.122.OR.ISUB.EQ.181.OR.ISUB.EQ.182.OR.
     &  ISUB.EQ.186.OR.ISUB.EQ.187.OR.ISUB.EQ.401.OR.ISUB.EQ.402) THEN
          Q2SF=PMAS(PYCOMP(KFPR(ISUBSV,2)),1)**2
          IF(MSTP(39).EQ.2) Q2SF=
     &         MAX(VINT(201)**2+VINT(202),VINT(206)**2+VINT(207))
          IF(MSTP(39).EQ.3) Q2SF=SH
          IF(MSTP(39).EQ.4) Q2SF=VINT(26)*VINT(2)
          IF(MSTP(39).EQ.5) Q2SF=PMAS(PYCOMP(KFPR(ISUBSV,1)),1)**2
C..Begin JA 040914
          IF(MSTP(39).EQ.6) Q2SF=0.25*(VINT(201)+SQRT(SH))**2
          IF(MSTP(39).EQ.7) Q2SF=
     &         (VINT(201)**2+VINT(202)+VINT(206)**2+VINT(207))/2d0
          IF(MSTP(39).EQ.8) Q2SF=PARP(193)
C..End JA
        ENDIF
      ENDIF
      IF(MINT(35).GE.3.AND.ISTSB.EQ.9) Q2SF=SQPTH
 
      Q2PS=Q2SF
      Q2SF=Q2SF*PARP(34)
      IF(MSTP(69).GE.1.AND.MINT(47).EQ.5) Q2SF=VINT(2)
      IF(MSTP(69).GE.2) Q2SF=VINT(2)
 
C...Identify to which class(es) subprocess belongs
      ISMECR=0
      ISQCD=0
      ISJETS=0
      IF (ISUBSV.EQ.1.OR.ISUBSV.EQ.2.OR.ISUBSV.EQ.3.OR.
     &     ISUBSV.EQ.102.OR.ISUBSV.EQ.141.OR.ISUBSV.EQ.142.OR.
     &     ISUBSV.EQ.144.OR.ISUBSV.EQ.151.OR.ISUBSV.EQ.152.OR.
     &     ISUBSV.EQ.156.OR.ISUBSV.EQ.157) ISMECR=1
      IF (ISUBSV.EQ.11.OR.ISUBSV.EQ.12.OR.ISUBSV.EQ.13.OR.
     &     ISUBSV.EQ.28.OR.ISUBSV.EQ.53.OR.ISUBSV.EQ.68) ISQCD=1
      IF ((ISUBSV.EQ.81.OR.ISUBSV.EQ.82).AND.MINT(55).LE.5) ISQCD=1
      IF (ISUBSV.GE.381.AND.ISUBSV.LE.386) ISQCD=1
      IF ((ISUBSV.EQ.387.OR.ISUBSV.EQ.388).AND.MINT(55).LE.5) ISQCD=1
      IF (ISTSB.EQ.9) ISQCD=1
      IF ((ISUBSV.GE.86.AND.ISUBSV.LE.89).OR.ISUBSV.EQ.107.OR.
     &     (ISUBSV.GE.14.AND.ISUBSV.LE.16).OR.(ISUBSV.GE.29.AND.
     &     ISUBSV.LE.32).OR.(ISUBSV.GE.111.AND.ISUBSV.LE.113).OR.
     &     ISUBSV.EQ.115.OR.(ISUBSV.GE.183.AND.ISUBSV.LE.185).OR.
     &     (ISUBSV.GE.188.AND.ISUBSV.LE.190).OR.ISUBSV.EQ.161.OR.
     &     ISUBSV.EQ.167.OR.ISUBSV.EQ.168.OR.(ISUBSV.GE.393.AND.
     &     ISUBSV.LE.395).OR.(ISUBSV.GE.421.AND.ISUBSV.LE.439).OR.
     &     (ISUBSV.GE.461.AND.ISUBSV.LE.479)) ISJETS=1
C...WBF is special case of ISJETS
      IF (ISUBSV.EQ.5.OR.ISUBSV.EQ.8.OR.
     &    (ISUBSV.GE.71.AND.ISUBSV.LE.73).OR.
     &    ISUBSV.EQ.76.OR.ISUBSV.EQ.77.OR.
     &    (ISUBSV.GE.121.AND.ISUBSV.LE.124).OR.
     &    ISUBSV.EQ.173.OR.ISUBSV.EQ.174.OR.
     &    ISUBSV.EQ.178.OR.ISUBSV.EQ.179.OR.
     &    ISUBSV.EQ.181.OR.ISUBSV.EQ.182.OR.
     &    ISUBSV.EQ.186.OR.ISUBSV.EQ.187.OR.
     &    ISUBSV.EQ.351.OR.ISUBSV.EQ.352) ISJETS=2
C...Some processes with photons also belong here.
      IF (ISUBSV.EQ.10.OR.(ISUBSV.GE.18.AND.ISUBSV.LE.20).OR.
     &     (ISUBSV.GE.33.AND.ISUBSV.LE.36).OR.ISUBSV.EQ.54.OR.
     &     ISUBSV.EQ.58.OR.ISUBSV.EQ.69.OR.ISUBSV.EQ.70.OR.
     &     ISUBSV.EQ.80.OR.(ISUBSV.GE.83.AND.ISUBSV.LE.85).OR.
     &     (ISUBSV.GE.106.AND.ISUBSV.LE.110).OR.ISUBSV.EQ.114.OR.
     &     (ISUBSV.GE.131.AND.ISUBSV.LE.140)) ISJETS=3

C...Choice of Q2 scale for parton-shower activity.
      IF(MSTP(22).GE.1.AND.(ISUB.EQ.10.OR.ISUB.EQ.83).AND.
     &(MINT(43).EQ.2.OR.MINT(43).EQ.3)) THEN
        XBJ=X(2)
        IF(MINT(43).EQ.3) XBJ=X(1)
        IF(MSTP(22).EQ.1) THEN
          Q2PS=-TH
        ELSEIF(MSTP(22).EQ.2) THEN
          Q2PS=((1D0-XBJ)/XBJ)*(-TH)
        ELSEIF(MSTP(22).EQ.3) THEN
          Q2PS=SQRT((1D0-XBJ)/XBJ)*(-TH)
        ELSE
          Q2PS=(1D0-XBJ)*MAX(1D0,-LOG(XBJ))*(-TH)
        ENDIF
      ENDIF
C...For multiple interactions, start from scale defined above
C...For all other QCD or "+jets"-type events, start shower from pThard.
      IF (ISJETS.EQ.1.OR.ISQCD.EQ.1.AND.ISTSB.NE.9) Q2PS=SQPTH
      IF((MSTP(68).EQ.1.OR.MSTP(68).EQ.3).AND.ISMECR.EQ.1) THEN
C...Max shower scale = s for ME corrected processes.
C...(pT-ordering: max pT2 is s/4)
        Q2PS=VINT(2)
        IF (MINT(35).GE.3) Q2PS=Q2PS*0.25D0
      ELSEIF(MSTP(68).GE.2.AND.ISQCD.EQ.0.AND.ISJETS.EQ.0) THEN
C...Max shower scale = s for all non-QCD, non-"+ jet" type processes.
C...(pT-ordering: max pT2 is s/4)
        Q2PS=VINT(2)
        IF (MINT(35).GE.3) Q2PS=Q2PS*0.25D0
      ENDIF
      IF(MINT(35).EQ.2.AND.ISTSB.EQ.9) Q2PS=SQPTH

C...Elastic and diffractive events not associated with scales so set 0.
      IF(ISUBSV.GE.91.AND.ISUBSV.LE.94) THEN
        Q2SF=0D0
        Q2PS=0D0
      ENDIF
 
C...Store derived kinematical quantities
      VINT(41)=X(1)
      VINT(42)=X(2)
      VINT(44)=SH
      VINT(43)=SQRT(SH)
      VINT(45)=TH
      VINT(46)=UH
      IF(ISTSB.NE.8) VINT(48)=SQPTH
      IF(ISTSB.NE.8) VINT(47)=SQRT(SQPTH)
      VINT(50)=TAUP*VINT(2)
      VINT(49)=SQRT(MAX(0D0,VINT(50)))
      VINT(52)=Q2
      VINT(51)=SQRT(Q2)
      VINT(54)=Q2SF
      VINT(53)=SQRT(Q2SF)
      VINT(56)=Q2PS
      VINT(55)=SQRT(Q2PS)
 
C...Set starting scale for multiple interactions
      IF (ISUBSV.EQ.95) THEN
        XT2GMX=0D0
      ELSEIF(MSTP(86).EQ.3.OR.(MSTP(86).EQ.2.AND.ISUBSV.NE.11.AND.
     &      ISUBSV.NE.12.AND.ISUBSV.NE.13.AND.ISUBSV.NE.28.AND.
     &      ISUBSV.NE.53.AND.ISUBSV.NE.68.AND.ISUBSV.NE.95.AND.
     &      ISUBSV.NE.96)) THEN
C...All accessible phase space allowed.
        XT2GMX=(1D0-VINT(41))*(1D0-VINT(42))
      ELSE
C...Scale of hard process sets limit.
C...2 -> 1. Limit is tau = x1*x2.
C...2 -> 2. Limit is XT2 for hard process + FS masses.
C...2 -> n > 2. Limit is tau' = tau of outer process.
        XT2GMX=VINT(25)
        IF(ISTSB.EQ.1) XT2GMX=VINT(21)
        IF(ISTSB.EQ.2)
     &      XT2GMX=(4D0*VINT(48)+2D0*VINT(63)+2D0*VINT(64))/VINT(2)
        IF(ISTSB.GE.3.AND.ISTSB.LE.5) XT2GMX=VINT(26)
      ENDIF
      VINT(62)=0.25D0*XT2GMX*VINT(2)
      VINT(61)=SQRT(MAX(0D0,VINT(62)))
 
C...Calculate parton distributions
      IF(ISTSB.LE.0) GOTO 160
      IF(MINT(47).GE.2) THEN
        DO 110 I=3-MIN(2,MINT(45)),MIN(2,MINT(46))
          XSF=X(I)
          IF(ISTSB.EQ.9) XSF=X(I)/VINT(142+I)
          IF(ISUB.EQ.99) THEN
            IF(MINT(140+I).EQ.0) THEN
              XSF=VINT(309-I)/(VINT(2)+VINT(309-I)-VINT(I+2)**2)
            ELSE
              XSF=VINT(309-I)/(VINT(2)+VINT(307)+VINT(308))
            ENDIF
            VINT(40+I)=XSF
            Q2SF=VINT(309-I)
          ENDIF
          MINT(105)=MINT(102+I)
          MINT(109)=MINT(106+I)
          VINT(120)=VINT(2+I)
          IF(MSTP(57).LE.1) THEN
            CALL PYPDFU(MINT(10+I),XSF,Q2SF,XPQ)
          ELSE
            CALL PYPDFL(MINT(10+I),XSF,Q2SF,XPQ)
          ENDIF
C...Safety margin against heavy flavour very close to threshold,
C...e.g. caused by mismatch in c and b masses.
          IF(Q2SF.LT.1.1*PMAS(4,1)**2) THEN
            XPQ(4)=0D0
            XPQ(-4)=0D0
          ENDIF
          IF(Q2SF.LT.1.1*PMAS(5,1)**2) THEN
            XPQ(5)=0D0
            XPQ(-5)=0D0
          ENDIF
          DO 100 KFL=-25,25
            XSFX(I,KFL)=XPQ(KFL)
  100     CONTINUE
  110   CONTINUE
      ENDIF
 
C...Calculate alpha_em, alpha_strong and K-factor
      XW=PARU(102)
      XWV=XW
      IF(MSTP(8).GE.2.OR.(ISUB.GE.71.AND.ISUB.LE.77)) XW=
     &1D0-(PMAS(24,1)/PMAS(23,1))**2
      XW1=1D0-XW
      XWC=1D0/(16D0*XW*XW1)
      AEM=PYALEM(Q2)
      IF(MSTP(8).GE.1) AEM=SQRT(2D0)*PARU(105)*PMAS(24,1)**2*XW/PARU(1)
      IF(MSTP(33).NE.3) AS=PYALPS(PARP(34)*Q2)
      FACK=1D0
      FACA=1D0
      IF(MSTP(33).EQ.1) THEN
        FACK=PARP(31)
      ELSEIF(MSTP(33).EQ.2) THEN
        FACK=PARP(31)
        FACA=PARP(32)/PARP(31)
      ELSEIF(MSTP(33).EQ.3) THEN
        Q2AS=PARP(33)*Q2
        IF(ISTSB.EQ.9.AND.MSTP(82).GE.2) Q2AS=Q2AS+
     &  PARU(112)*PARP(82)*(VINT(1)/PARP(89))**PARP(90)
        AS=PYALPS(Q2AS)
      ENDIF
      VINT(138)=1D0
      VINT(57)=AEM
      VINT(58)=AS
 
C...Set flags for allowed reacting partons/leptons
      DO 140 I=1,2
        DO 120 J=-25,25
          KFAC(I,J)=0
  120   CONTINUE
        IF(MINT(44+I).EQ.1) THEN
          KFAC(I,MINT(10+I))=1
        ELSEIF(MINT(40+I).EQ.1.AND.MSTP(12).EQ.0) THEN
          KFAC(I,MINT(10+I))=1
          KFAC(I,22)=1
          KFAC(I,24)=1
          KFAC(I,-24)=1
        ELSE
          DO 130 J=-25,25
            KFAC(I,J)=KFIN(I,J)
            IF(IABS(J).GT.MSTP(58).AND.IABS(J).LE.10) KFAC(I,J)=0
            IF(XSFX(I,J).LT.1D-10) KFAC(I,J)=0
  130     CONTINUE
        ENDIF
  140 CONTINUE
 
C...Lower and upper limit for fermion flavour loops
      MMIN1=0
      MMAX1=0
      MMIN2=0
      MMAX2=0
      DO 150 J=-20,20
        IF(KFAC(1,-J).EQ.1) MMIN1=-J
        IF(KFAC(1,J).EQ.1) MMAX1=J
        IF(KFAC(2,-J).EQ.1) MMIN2=-J
        IF(KFAC(2,J).EQ.1) MMAX2=J
  150 CONTINUE
      MMINA=MIN(MMIN1,MMIN2)
      MMAXA=MAX(MMAX1,MMAX2)
 
C...Common resonance mass and width combinations
      SQMZ=PMAS(23,1)**2
      SQMW=PMAS(24,1)**2
      GMMZ=PMAS(23,1)*PMAS(23,2)
      GMMW=PMAS(24,1)*PMAS(24,2)
 
C...Polarization factors...implemented so far for W+W-(25)
      POLR=(1D0+PARJ(132))*(1D0-PARJ(131))
      POLL=(1D0-PARJ(132))*(1D0+PARJ(131))
      POLRR=(1D0+PARJ(132))*(1D0+PARJ(131))
      POLLL=(1D0-PARJ(132))*(1D0-PARJ(131))
 
C...Phase space integral in tau
      COMFAC=PARU(1)*PARU(5)/VINT(2)
      IF(MINT(41).EQ.2.AND.MINT(42).EQ.2) COMFAC=COMFAC*FACK
      IF((MINT(47).GE.2.OR.(ISTSB.GE.3.AND.ISTSB.LE.5)).AND.
     &ISTSB.NE.8.AND.ISTSB.NE.9) THEN
        ATAU1=LOG(TAUMAX/TAUMIN)
        ATAU2=(TAUMAX-TAUMIN)/(TAUMAX*TAUMIN)
        H1=COEF(ISUBSV,1)+(ATAU1/ATAU2)*COEF(ISUBSV,2)/TAU
        IF(MINT(72).GE.1) THEN
          TAUR1=VINT(73)
          GAMR1=VINT(74)
          ATAUD=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR1)/(TAUMAX+TAUR1))
          ATAU3=ATAUD/TAUR1
          IF(ATAUD.GT.1D-10) H1=H1+
     &    (ATAU1/ATAU3)*COEF(ISUBSV,3)/(TAU+TAUR1)
          ATAUD=ATAN((TAUMAX-TAUR1)/GAMR1)-ATAN((TAUMIN-TAUR1)/GAMR1)
          ATAU4=ATAUD/GAMR1
          IF(ATAUD.GT.1D-10) H1=H1+
     &    (ATAU1/ATAU4)*COEF(ISUBSV,4)*TAU/((TAU-TAUR1)**2+GAMR1**2)
        ENDIF
        IF(MINT(72).GE.2) THEN
          TAUR2=VINT(75)
          GAMR2=VINT(76)
          ATAUD=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR2)/(TAUMAX+TAUR2))
          ATAU5=ATAUD/TAUR2
          IF(ATAUD.GT.1D-10) H1=H1+
     &    (ATAU1/ATAU5)*COEF(ISUBSV,5)/(TAU+TAUR2)
          ATAUD=ATAN((TAUMAX-TAUR2)/GAMR2)-ATAN((TAUMIN-TAUR2)/GAMR2)
          ATAU6=ATAUD/GAMR2
          IF(ATAUD.GT.1D-10) H1=H1+
     &    (ATAU1/ATAU6)*COEF(ISUBSV,6)*TAU/((TAU-TAUR2)**2+GAMR2**2)
        ENDIF
        IF(MINT(72).EQ.3) THEN
          TAUR3=VINT(77)
          GAMR3=VINT(78)
          ATAUD=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR3)/(TAUMAX+TAUR3))
          ATAU50=ATAUD/TAUR3
          IF(ATAUD.GT.1D-10) H1=H1+
     &    (ATAU1/ATAU50)*COEFX(ISUBSV,1)/(TAU+TAUR3)
          ATAUD=ATAN((TAUMAX-TAUR3)/GAMR3)-ATAN((TAUMIN-TAUR3)/GAMR3)
          ATAU60=ATAUD/GAMR3
          IF(ATAUD.GT.1D-10) H1=H1+
     &    (ATAU1/ATAU60)*COEFX(ISUBSV,2)*TAU/((TAU-TAUR3)**2+GAMR3**2)
        ENDIF
        IF(MINT(47).EQ.5.AND.(ISTSB.LE.2.OR.ISTSB.GE.5)) THEN
          ATAU7=LOG(MAX(2D-10,1D0-TAUMIN)/MAX(2D-10,1D0-TAUMAX))
          IF(ATAU7.GT.1D-10) H1=H1+(ATAU1/ATAU7)*COEF(ISUBSV,7)*TAU/
     &    MAX(2D-10,1D0-TAU)
        ELSEIF(MINT(47).GE.6.AND.(ISTSB.LE.2.OR.ISTSB.GE.5)) THEN
          ATAU7=LOG(MAX(1D-10,1D0-TAUMIN)/MAX(1D-10,1D0-TAUMAX))
          IF(ATAU7.GT.1D-10) H1=H1+(ATAU1/ATAU7)*COEF(ISUBSV,7)*TAU/
     &    MAX(1D-10,1D0-TAU)
        ENDIF
        COMFAC=COMFAC*ATAU1/(TAU*H1)
      ENDIF
 
C...Phase space integral in y*
      IF((MINT(47).EQ.4.OR.MINT(47).EQ.5).AND.ISTSB.NE.8.AND.ISTSB.NE.9)
     &THEN
        AYST0=YSTMAX-YSTMIN
        IF(AYST0.LT.1D-10) THEN
          COMFAC=0D0
        ELSE
          AYST1=0.5D0*(YSTMAX-YSTMIN)**2
          AYST2=AYST1
          AYST3=2D0*(ATAN(EXP(YSTMAX))-ATAN(EXP(YSTMIN)))
          H2=(AYST0/AYST1)*COEF(ISUBSV,8)*(YST-YSTMIN)+
     &    (AYST0/AYST2)*COEF(ISUBSV,9)*(YSTMAX-YST)+
     &    (AYST0/AYST3)*COEF(ISUBSV,10)/COSH(YST)
          IF(MINT(45).EQ.3) THEN
            YST0=-0.5D0*LOG(TAUE)
            AYST4=LOG(MAX(1D-10,EXP(YST0-YSTMIN)-1D0)/
     &      MAX(1D-10,EXP(YST0-YSTMAX)-1D0))
            IF(AYST4.GT.1D-10) H2=H2+(AYST0/AYST4)*COEF(ISUBSV,11)/
     &      MAX(1D-10,1D0-EXP(YST-YST0))
          ENDIF
          IF(MINT(46).EQ.3) THEN
            YST0=-0.5D0*LOG(TAUE)
            AYST5=LOG(MAX(1D-10,EXP(YST0+YSTMAX)-1D0)/
     &      MAX(1D-10,EXP(YST0+YSTMIN)-1D0))
            IF(AYST5.GT.1D-10) H2=H2+(AYST0/AYST5)*COEF(ISUBSV,12)/
     &      MAX(1D-10,1D0-EXP(-YST-YST0))
          ENDIF
          COMFAC=COMFAC*AYST0/H2
        ENDIF
      ENDIF
 
C...2 -> 1 processes: reduction in angular part of phase space integral
C...for case of decaying resonance
      ACTH0=CTNMAX-CTNMIN+CTPMAX-CTPMIN
      IF((ISTSB.EQ.1.OR.ISTSB.EQ.3.OR.ISTSB.EQ.5)) THEN
        IF(MDCY(PYCOMP(KFPR(ISUBSV,1)),1).EQ.1) THEN
          IF(KFPR(ISUB,1).EQ.25.OR.KFPR(ISUB,1).EQ.37.OR.
     &    KFPR(ISUB,1).EQ.39) THEN
            COMFAC=COMFAC*0.5D0*ACTH0
          ELSE
            COMFAC=COMFAC*0.125D0*(3D0*ACTH0+CTNMAX**3-CTNMIN**3+
     &      CTPMAX**3-CTPMIN**3)
          ENDIF
        ENDIF
 
C...2 -> 2 processes: angular part of phase space integral
      ELSEIF(ISTSB.EQ.2.OR.ISTSB.EQ.4) THEN
        ACTH1=LOG((MAX(RM34,RSQM-CTNMIN)*MAX(RM34,RSQM-CTPMIN))/
     &  (MAX(RM34,RSQM-CTNMAX)*MAX(RM34,RSQM-CTPMAX)))
        ACTH2=LOG((MAX(RM34,RSQM+CTNMAX)*MAX(RM34,RSQM+CTPMAX))/
     &  (MAX(RM34,RSQM+CTNMIN)*MAX(RM34,RSQM+CTPMIN)))
        ACTH3=1D0/MAX(RM34,RSQM-CTNMAX)-1D0/MAX(RM34,RSQM-CTNMIN)+
     &  1D0/MAX(RM34,RSQM-CTPMAX)-1D0/MAX(RM34,RSQM-CTPMIN)
        ACTH4=1D0/MAX(RM34,RSQM+CTNMIN)-1D0/MAX(RM34,RSQM+CTNMAX)+
     &  1D0/MAX(RM34,RSQM+CTPMIN)-1D0/MAX(RM34,RSQM+CTPMAX)
        H3=COEF(ISUBSV,13)+
     &  (ACTH0/ACTH1)*COEF(ISUBSV,14)/MAX(RM34,RSQM-CTH)+
     &  (ACTH0/ACTH2)*COEF(ISUBSV,15)/MAX(RM34,RSQM+CTH)+
     &  (ACTH0/ACTH3)*COEF(ISUBSV,16)/MAX(RM34,RSQM-CTH)**2+
     &  (ACTH0/ACTH4)*COEF(ISUBSV,17)/MAX(RM34,RSQM+CTH)**2
        COMFAC=COMFAC*ACTH0*0.5D0*BE34/H3
 
C...2 -> 2 processes: take into account final state Breit-Wigners
        COMFAC=COMFAC*VINT(80)
      ENDIF
 
C...2 -> 3, 4 processes: phace space integral in tau'
      IF(MINT(47).GE.2.AND.ISTSB.GE.3.AND.ISTSB.LE.5) THEN
        ATAUP1=LOG(TAUPMX/TAUPMN)
        ATAUP2=((1D0-TAU/TAUPMX)**4-(1D0-TAU/TAUPMN)**4)/(4D0*TAU)
        H4=COEF(ISUBSV,18)+
     &  (ATAUP1/ATAUP2)*COEF(ISUBSV,19)*(1D0-TAU/TAUP)**3/TAUP
        IF(MINT(47).EQ.5) THEN
          ATAUP3=LOG(MAX(2D-10,1D0-TAUPMN)/MAX(2D-10,1D0-TAUPMX))
          H4=H4+(ATAUP1/ATAUP3)*COEF(ISUBSV,20)*TAUP/MAX(2D-10,1D0-TAUP)
        ELSEIF(MINT(47).GE.6) THEN
          ATAUP3=LOG(MAX(1D-10,1D0-TAUPMN)/MAX(1D-10,1D0-TAUPMX))
          H4=H4+(ATAUP1/ATAUP3)*COEF(ISUBSV,20)*TAUP/MAX(1D-10,1D0-TAUP)
        ENDIF
        COMFAC=COMFAC*ATAUP1/H4
      ENDIF
 
C...2 -> 3, 4 processes: effective W/Z parton distributions
      IF(ISTSB.EQ.3.OR.ISTSB.EQ.4) THEN
        IF(1D0-TAU/TAUP.GT.1D-4) THEN
          FZW=(1D0+TAU/TAUP)*LOG(TAUP/TAU)-2D0*(1D0-TAU/TAUP)
        ELSE
          FZW=1D0/6D0*(1D0-TAU/TAUP)**3*TAU/TAUP
        ENDIF
        COMFAC=COMFAC*FZW
      ENDIF
 
C...2 -> 3 processes: phase space integrals for pT1, pT2, y3, mirror
      IF(ISTSB.EQ.5) THEN
        COMFAC=COMFAC*VINT(205)*VINT(210)*VINT(212)*VINT(214)/
     &  (128D0*PARU(1)**4*VINT(220))*(TAU**2/TAUP)
      ENDIF
 
C...Phase space integral for low-pT and multiple interactions
      IF(ISTSB.EQ.9) THEN
        COMFAC=PARU(1)*PARU(5)*FACK*0.5D0*VINT(2)/SH2
        ATAU1=LOG(2D0*(1D0+SQRT(1D0-XT2))/XT2-1D0)
        ATAU2=2D0*ATAN(1D0/XT2-1D0)/SQRT(XT2)
        H1=COEF(ISUBSV,1)+(ATAU1/ATAU2)*COEF(ISUBSV,2)/SQRT(TAU)
        COMFAC=COMFAC*ATAU1/H1
        AYST0=YSTMAX-YSTMIN
        AYST1=0.5D0*(YSTMAX-YSTMIN)**2
        AYST3=2D0*(ATAN(EXP(YSTMAX))-ATAN(EXP(YSTMIN)))
        H2=(AYST0/AYST1)*COEF(ISUBSV,8)*(YST-YSTMIN)+
     &  (AYST0/AYST1)*COEF(ISUBSV,9)*(YSTMAX-YST)+
     &  (AYST0/AYST3)*COEF(ISUBSV,10)/COSH(YST)
        COMFAC=COMFAC*AYST0/H2
        IF(MSTP(82).LE.1) COMFAC=COMFAC*XT2**2*(1D0/VINT(149)-1D0)
C...For MSTP(82)>=2 an additional factor (xT2/(xT2+VINT(149))**2 is
C...introduced to make cross-section finite for xT2 -> 0
        IF(MSTP(82).GE.2) COMFAC=COMFAC*XT2**2/(VINT(149)*
     &  (1D0+VINT(149)))
      ENDIF
 
C...Real gamma + gamma: include factor 2 when different nature
  160 IF(MINT(11).EQ.22.AND.MINT(12).EQ.22.AND.MINT(123).GE.4.AND.
     &MSTP(14).LE.10) COMFAC=2D0*COMFAC
 
C...Extra factors to include the effects of
C...longitudinal resolved photons (but not direct or DIS ones).
      DO 170 ISDE=1,2
        IF(MINT(10+ISDE).EQ.22.AND.MINT(106+ISDE).GE.1.AND.
     &  MINT(106+ISDE).LE.3) THEN
          VINT(314+ISDE)=1D0
          XY=PARP(166+ISDE)
          IF(MSTP(16).EQ.0) THEN
            IF(VINT(304+ISDE).GT.0D0.AND.VINT(304+ISDE).LT.1D0)
     &      XY=VINT(304+ISDE)
          ELSE
            IF(VINT(308+ISDE).GT.0D0.AND.VINT(308+ISDE).LT.1D0)
     &      XY=VINT(308+ISDE)
          ENDIF
          Q2GA=VINT(306+ISDE)
          IF(MSTP(17).GT.0.AND.XY.GT.0D0.AND.XY.LT.1D0.AND.
     &    Q2GA.GT.0D0) THEN
            REDUCE=0D0
            IF(MSTP(17).EQ.1) THEN
              REDUCE=4D0*Q2*Q2GA/(Q2+Q2GA)**2
            ELSEIF(MSTP(17).EQ.2) THEN
              REDUCE=4D0*Q2GA/(Q2+Q2GA)
            ELSEIF(MSTP(17).EQ.3) THEN
              PMVIRT=PMAS(PYCOMP(113),1)
              REDUCE=4D0*Q2GA/(PMVIRT**2+Q2GA)
            ELSEIF(MSTP(17).EQ.4.AND.MINT(106+ISDE).EQ.1) THEN
              PMVIRT=PMAS(PYCOMP(113),1)
              REDUCE=4D0*PMVIRT**2*Q2GA/(PMVIRT**2+Q2GA)**2
            ELSEIF(MSTP(17).EQ.4.AND.MINT(106+ISDE).EQ.2) THEN
              PMVIRT=PMAS(PYCOMP(113),1)
              REDUCE=4D0*PMVIRT**2*Q2GA/(PMVIRT**2+Q2GA)**2
            ELSEIF(MSTP(17).EQ.4.AND.MINT(106+ISDE).EQ.3) THEN
              PMVSMN=4D0*PARP(15)**2
              PMVSMX=4D0*VINT(154)**2
              REDTRA=1D0/(PMVSMN+Q2GA)-1D0/(PMVSMX+Q2GA)
              REDLON=(3D0*PMVSMN+Q2GA)/(PMVSMN+Q2GA)**3-
     &        (3D0*PMVSMX+Q2GA)/(PMVSMX+Q2GA)**3
              REDUCE=4D0*(Q2GA/6D0)*REDLON/REDTRA
            ELSEIF(MSTP(17).EQ.5.AND.MINT(106+ISDE).EQ.1) THEN
              PMVIRT=PMAS(PYCOMP(113),1)
              REDUCE=4D0*Q2GA/(PMVIRT**2+Q2GA)
            ELSEIF(MSTP(17).EQ.5.AND.MINT(106+ISDE).EQ.2) THEN
              PMVIRT=PMAS(PYCOMP(113),1)
              REDUCE=4D0*Q2GA/(PMVIRT**2+Q2GA)
            ELSEIF(MSTP(17).EQ.5.AND.MINT(106+ISDE).EQ.3) THEN
              PMVSMN=4D0*PARP(15)**2
              PMVSMX=4D0*VINT(154)**2
              REDTRA=1D0/(PMVSMN+Q2GA)-1D0/(PMVSMX+Q2GA)
              REDLON=1D0/(PMVSMN+Q2GA)**2-1D0/(PMVSMX+Q2GA)**2
              REDUCE=4D0*(Q2GA/2D0)*REDLON/REDTRA
            ENDIF
            BEAMAS=PYMASS(11)
            IF(VINT(302+ISDE).GT.0D0) BEAMAS=VINT(302+ISDE)
            FRACLT=1D0/(1D0+XY**2/2D0/(1D0-XY)*
     &      (1D0-2D0*BEAMAS**2/Q2GA))
            VINT(314+ISDE)=1D0+PARP(165)*REDUCE*FRACLT
          ENDIF
        ELSE
          VINT(314+ISDE)=1D0
        ENDIF
        COMFAC=COMFAC*VINT(314+ISDE)
  170 CONTINUE
 
C...Evaluate cross sections - done in separate routines by kind
C...of physics, to keep PYSIGH of sensible size.
      IF(MAP.EQ.1) THEN
C...Standard QCD (including photons).
        CALL PYSGQC(NCHN,SIGS)
      ELSEIF(MAP.EQ.2) THEN
C...Heavy flavours.
        CALL PYSGHF(NCHN,SIGS)
      ELSEIF(MAP.EQ.3) THEN
C...W/Z.
        CALL PYSGWZ(NCHN,SIGS)
      ELSEIF(MAP.EQ.4) THEN
C...Higgs (2 doublets; including longitudinal W/Z scattering).
        CALL PYSGHG(NCHN,SIGS)
      ELSEIF(MAP.EQ.5) THEN
C...SUSY.
        CALL PYSGSU(NCHN,SIGS)
      ELSEIF(MAP.EQ.6) THEN
C...Technicolor.
        CALL PYSGTC(NCHN,SIGS)
      ELSEIF(MAP.EQ.7) THEN
C...Exotics (Z'/W'/LQ/R/f*/H++/Z_R/W_R/G*).
        CALL PYSGEX(NCHN,SIGS)
      ELSEIF(MAP.EQ.8) THEN
C... Universal Extra Dimensions
         CALL PYXUED(NCHN,SIGS)


C=====================================================================
C BEGIN HARDCOL MODIFICATION
C=====================================================================

C...This sets up processes ISUB = 401,...,406 to be elastic scattering
C...by hard color singlet (BFKL pomeron) exchange, calculated in two
C...different ways.
C
C...In these processes q (q') means any quark or antiquark.
C...Observe that the parton colors are the same in the initial and 
C...final states.

      ELSEIF(ISUB.GE.403.OR.ISUB.LE.408) THEN
       
        IF((MSUB(403).NE.0.OR.MSUB(404).NE.0.OR.MSUB(405).NE.0)
     &  .AND.(MSUB(406).NE.0.OR.MSUB(407).NE.0.OR.MSUB(408).NE.0))
     &  THEN
          WRITE(*,*) 
          WRITE(*,*) 'ERROR: BOTH MUELLER-TANG AND BFKL SWITCHED ON!' 
          WRITE(*,*) '       STOPPING EXECUTION...' 
          WRITE(*,*) 
          STOP
        ENDIF

        IF(ISUB.EQ.403) THEN
C...q + q' -> q + q' color singlet exchange (Mueller-Tang)
          RAPID=DLOG(ABS(SH/TH))          
          FACFF=COMFAC*(0.25D0/PARU(1))*((AS*4D0/3D0)**4)/TH2
          FACFF=FACFF*(0.86643D0*RAPID**2-5.71049D0*RAPID+15.85634D0)
C...We have to multiply by shat**2/pi as it's not in our dsigma/dthat
          FACFF=FACFF*SH2/PARU(1)          
          DO 401 I=MMIN1,MMAX1
            IA=IABS(I)
            IF(I.EQ.0.OR.IA.GT.MSTP(58).OR.KFAC(1,I).EQ.0) GOTO 401
            DO 411 J=MMIN2,MMAX2
              JA=IABS(J)
              IF(J.EQ.0.OR.JA.GT.MSTP(58).OR.KFAC(2,J).EQ.0) GOTO 411
              NCHN=NCHN+1
              ISIG(NCHN,1)=I
              ISIG(NCHN,2)=J
              ISIG(NCHN,3)=1
              SIGH(NCHN)=FACFF
              IF(I.EQ.J) SIGH(NCHN)=SIGH(NCHN)*0.5D0
  411       CONTINUE
  401     CONTINUE     
        

        ELSEIF(ISUB.EQ.404) THEN
C...f + g -> f + g color singlet exchange (Mueller-Tang)
          RAPID=DLOG(ABS(SH/TH))          
          FACFG=COMFAC*(0.25D0/PARU(1))*((AS*4D0/3D0)**4)/TH2
          FACFG=FACFG*(0.86643D0*RAPID**2-5.71049D0*RAPID+15.85634D0)
          FACFG=FACFG*81D0/16D0
          FACFG=FACFG*SH2/PARU(1)
          DO 421 I=MMINA,MMAXA
            IF(I.EQ.0.OR.IABS(I).GT.10) GOTO 421
            DO 431 ISDE=1,2
              IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 431
              IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 431
              NCHN=NCHN+1
              ISIG(NCHN,ISDE)=I
              ISIG(NCHN,3-ISDE)=21
              ISIG(NCHN,3)=1
              SIGH(NCHN)=FACFG
  431       CONTINUE
  421     CONTINUE

        ELSEIF(ISUB.EQ.405) THEN
C...g + g -> g + g (Mueller-Tang)
          RAPID=DLOG(ABS(SH/TH))          
          FACGG=COMFAC*(0.25D0/PARU(1))*((AS*4D0/3D0)**4)/TH2
          FACGG=FACGG*(0.86643D0*RAPID**2-5.71049D0*RAPID+15.85634D0)
          FACGG=FACGG*(81D0/16D0)**2
          FACGG=FACGG*SH2/PARU(1)
          IF(KFAC(1,21)*KFAC(2,21).NE.0) THEN  
            NCHN=NCHN+1
            ISIG(NCHN,1)=21
            ISIG(NCHN,2)=21
            ISIG(NCHN,3)=1
            SIGH(NCHN)=0.5D0*FACGG
          ENDIF

        ELSEIF(ISUB.EQ.406) THEN
C...q + q' -> q + q'  (BFKL)
          RAPID=DLOG(ABS(SH/TH))          
          FACFF=COMFAC*HCS_XS(RAPID,SQRT(-TH))
          FACFF=FACFF*SH2/PARU(1)          
          DO 501 I=MMIN1,MMAX1
            IA=IABS(I)
            IF(I.EQ.0.OR.IA.GT.MSTP(58).OR.KFAC(1,I).EQ.0) GOTO 501
            DO 511 J=MMIN2,MMAX2
              JA=IABS(J)
              IF(J.EQ.0.OR.JA.GT.MSTP(58).OR.KFAC(2,J).EQ.0) GOTO 511
              NCHN=NCHN+1
              ISIG(NCHN,1)=I
              ISIG(NCHN,2)=J
              ISIG(NCHN,3)=1
              SIGH(NCHN)=FACFF
              IF(I.EQ.J) SIGH(NCHN)=SIGH(NCHN)*0.5D0
  511       CONTINUE
  501     CONTINUE     
        

        ELSEIF(ISUB.EQ.407) THEN
C...f + g -> f + g (BFKL)
          RAPID=DLOG(ABS(SH/TH))          
          FACFG=COMFAC*HCS_XS(RAPID,SQRT(-TH))
          FACFG=FACFG*81D0/16D0
          FACFG=FACFG*SH2/PARU(1)
          DO 521 I=MMINA,MMAXA
            IF(I.EQ.0.OR.IABS(I).GT.10) GOTO 521
            DO 531 ISDE=1,2
              IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 531
              IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 531
              NCHN=NCHN+1
              ISIG(NCHN,ISDE)=I
              ISIG(NCHN,3-ISDE)=21
              ISIG(NCHN,3)=1
              SIGH(NCHN)=FACFG
  531       CONTINUE
  521     CONTINUE

        ELSEIF(ISUB.EQ.408) THEN
C...g + g -> g + g (BFKL)
          RAPID=DLOG(ABS(SH/TH))          
          FACGG=COMFAC*HCS_XS(RAPID,SQRT(-TH))
          FACGG=FACGG*(81D0/16D0)**2
          FACGG=FACGG*SH2/PARU(1)
          IF(KFAC(1,21)*KFAC(2,21).NE.0) THEN  
            NCHN=NCHN+1
            ISIG(NCHN,1)=21
            ISIG(NCHN,2)=21
            ISIG(NCHN,3)=1
            SIGH(NCHN)=0.5D0*FACGG
          ENDIF
        ENDIF    

C=====================================================================
C END HARDCOL MODIFICATION
C=====================================================================


      ENDIF
 
C...Multiply with parton distributions
      IF(ISUB.LE.90.OR.ISUB.GE.96) THEN
        DO 180 ICHN=1,NCHN
          IF(MINT(45).GE.2) THEN
            KFL1=ISIG(ICHN,1)
            SIGH(ICHN)=SIGH(ICHN)*XSFX(1,KFL1)
          ENDIF
          IF(MINT(46).GE.2) THEN
            KFL2=ISIG(ICHN,2)
            SIGH(ICHN)=SIGH(ICHN)*XSFX(2,KFL2)
          ENDIF
          SIGS=SIGS+SIGH(ICHN)
  180   CONTINUE
      ENDIF
 
      RETURN
      END
