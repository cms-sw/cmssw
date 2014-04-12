 
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
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      SAVE /PYDAT1/,/PYINT1/,/PYINT2/,/PYSUBS/
C... CSA specific 
      integer CSAMODE
      integer pad
      double precision  MUONRW, GAMMAJRW, ZJRW, ZPRW, HLTRW,
     &  SUSYRW, WWRW, PTPOWER
      common /EXPAR/ pad, CSAMODE,MUONRW, GAMMAJRW, ZJRW, ZPRW, 
     &  HLTRW, SUSYRW, WWRW, PTPOWER
 
C...Set default weight for WTXS.

C      write(*,*) 'Reweighting event ...'
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

      PTHAT = SQRT(PT2)  
 
 
C     CSAMODE  :   selection of reweighting algorithm for CSA06 production
C                  1 for QCD dijet
C                  2 for EWK soup 
C                  3 for HLT soup
C                  4 for soft muon soup
C                  5 for exotics soup ?
C                  6 for cross-section reweighted quarkonia production

 
      IF (CSAMODE.LE.0.OR.CSAMODE.GT.7) THEN
         write (*,*) ' CSAMODE not properly set !! No reweighting!! '
         write (*,*) ' CSAMODE = ', CSAMODE
      ENDIF      

 
 
C...Optional printout
 
C      write (*,*) ' CSAMODE = ', CSAMODE
C      write (*,*) ' MUONRW = ', MUONRW
C      write (*,*) ' GAMMAJRW = ', GAMMAJRW
C      write (*,*) ' ZJRW = ', ZJRW
 
C...Weights for QCD dijet sample
      
      IF (CSAMODE.EQ.1) THEN

      IF (ISUB.EQ.11.OR.ISUB.EQ.68.OR.ISUB.EQ.28.OR.ISUB.EQ.53
     & .OR.ISUB.EQ.12.OR.ISUB.EQ.13) THEN 
       IF(PTHAT.GE.0.AND.PTHAT.LT.15) WTXS = 0.025 
       IF(PTHAT.GE.15.AND.PTHAT.LT.20) WTXS = 1.8405931
       IF(PTHAT.GE.20.AND.PTHAT.LT.30) WTXS = 8.60482502 
       IF(PTHAT.GE.30.AND.PTHAT.LT.50) WTXS = 35.4135551 
       IF(PTHAT.GE.50.AND.PTHAT.LT.80) WTXS = 263.720733 
       IF(PTHAT.GE.80.AND.PTHAT.LT.120) WTXS = 936.023193
       IF(PTHAT.GE.120.AND.PTHAT.LT.170) WTXS = 5525.80176
       IF(PTHAT.GE.170.AND.PTHAT.LT.230) WTXS = 27337.9121
       IF(PTHAT.GE.230.AND.PTHAT.LT.300) WTXS = 115738.633
       IF(PTHAT.GE.300.AND.PTHAT.LT.380) WTXS = 432008.344
       IF(PTHAT.GE.380.AND.PTHAT.LT.470) WTXS = 1461105.62
       IF(PTHAT.GE.470.AND.PTHAT.LT.600) WTXS = 3999869.75
       IF(PTHAT.GE.600.AND.PTHAT.LT.800) WTXS = 8180579.5
       IF(PTHAT.GE.800.AND.PTHAT.LT.1000) WTXS = 46357008.
       IF(PTHAT.GE.1000.AND.PTHAT.LT.1400) WTXS = 152645456.
       IF(PTHAT.GE.1400.AND.PTHAT.LT.1800) WTXS = 1.56872026D9
       IF(PTHAT.GE.1800.AND.PTHAT.LT.2200) WTXS = 1.14387118D10
       IF(PTHAT.GE.2200.AND.PTHAT.LT.2600) WTXS = 6.9543682D10
       IF(PTHAT.GE.2600.AND.PTHAT.LT.3000) WTXS = 3.86604466D11
       IF(PTHAT.GE.3000.AND.PTHAT.LT.3500) WTXS = 1.96279625D12
       IF(PTHAT.GE.3500.AND.PTHAT.LT.4000) WTXS = 1.70783513D13

      ENDIF

      IF (ISUB.EQ.14.OR.ISUB.EQ.29) THEN
       IF(PTHAT.GE.9.AND.PTHAT.LT.44) WTXS = 36
       IF(PTHAT.GE.44.AND.PTHAT.LT.220) WTXS = 7500
       IF (GAMMAJRW.GT.(1.0D-14)) WTXS = WTXS * GAMMAJRW
      ENDIF

      IF (ISUB.EQ.15.OR.ISUB.EQ.30) THEN
       IF(PTHAT.GE.9.AND.PTHAT.LT.44) WTXS = 10.6
       IF(PTHAT.GE.44.AND.PTHAT.LT.220) WTXS = 90
       IF (ZJRW.GT.(1.0D-14)) WTXS = WTXS * ZJRW
      ENDIF

     

C... Fit function form
C      WTXS = (150.564d0*(PT2/25.0d0)**(6.28335d0)*
C     & exp(-6.28335d0*(PT2/25.0d0))
C     & + 0.035313d0*PT2-0.00628d0*log(PT2+1)*log(PT2+1))/
C     & (1.04992d0*exp(-0.245*PT2)) 

      ENDIF
 
 
C...Weights for EWK sample
    
      IF (CSAMODE.EQ.2) THEN
 
      IF (ISUB.EQ.2) WTXS=0.2
      IF (ISUB.EQ.102) WTXS=400.    
      IF (ISUB.EQ.123) WTXS=400.    
      IF (ISUB.EQ.124) WTXS=400.  
      
      IF (ISUB.EQ.25) WTXS = WWRW 
            
      ENDIF
      
C... Weights for HLT sample

      IF (CSAMODE.EQ.3) THEN
      
c      IF (ISUB.EQ.2) WTXS=0.2 
      IF (ISUB.EQ.11.OR.ISUB.EQ.68.OR.ISUB.EQ.28.OR.ISUB.EQ.53
     & .OR.ISUB.EQ.12.OR.ISUB.EQ.13) THEN 
        IF(PTHAT.LT.350) THEN 
	 WTXS=1.0D-8 
	 IF (HLTRW.GT.(1.0D-14)) WTXS = WTXS * HLTRW
	ENDIF
	IF(PTHAT.GE.350) WTXS=1.0	
      ENDIF
c      IF (ISUB.EQ.81.OR.ISUB.EQ.82) WTXS=100. 
       
      ENDIF      

C...Weights for Soft Muon sample

      IF (CSAMODE.EQ.4) THEN
      
       IF (ISUB.EQ.86) THEN
         WTXS = 1.25D7       
         IF (MUONRW.GT.(1.0D-14)) WTXS = WTXS * MUONRW
       ENDIF
      
      ENDIF

C...Optional weights for zprime and susy (exotics soup?)

      IF (CSAMODE.GE.1.OR.CSAMODE.LE.3.OR.CSAMODE.EQ.5) THEN
      
      
        IF (ISUB.EQ.141) THEN
          IF (ZPRW.GT.(1.0D-14)) WTXS = ZPRW 
        ENDIF
       
        IF (ISUB.GE.201.AND.ISUB.LE.296) THEN
          IF (SUSYRW.GT.(1.0D-14)) WTXS = SUSYRW 
        ENDIF

      ENDIF 


C...Weights for cross-section reweighted quarkonia      

      IF (CSAMODE.EQ.6) THEN

C...Copy form for pT0 as used in multiple interactions.
      PT0=PARP(82)*(VINT(1)/PARP(89))**PARP(90)
      PT20=PT0**2

C...Introduce dampening factor. 
      WTXS=(PT2/(PT20+PT2))**2

C...Also dampen alpha_strong by using larger Q2 scale.
      Q2=VINT(52)
      WTXS=WTXS*(PYALPS(PT20+Q2)/PYALPS(Q2))**3

      ENDIF 

      IF (CSAMODE.EQ.7) THEN
        WTXS=(PTHAT/CKIN(3))**PTPOWER
      ENDIF
       

      RETURN
      END
