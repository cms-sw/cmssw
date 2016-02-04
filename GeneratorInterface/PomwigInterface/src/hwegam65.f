CDECK  ID>, HWEGAM.
*CMZ :-        -26/04/91  11.11.55  by  Bryan Webber
*-- Author :    Bryan Webber & Luca Stanco
* Modified to substitue pomeron (reggeon) flux for photon flux
* B. Cox and J. Forshaw 11/07/00
* Bug fix for double pomeron 12/06/01 B. Cox
* Modified for herwig65 21/11/02 B. Cox
* Changed outgoing particles to P and PBAR (depending on whether incoming 
* 'lepton' is an electron or positron) 16/06/04
* Added H1 2006 Diffractive PDFs 11/02/07 B. Cox
C-----------------------------------------------------------------------
      SUBROUTINE HWEGAM(IHEP,ZMI,ZMA,WWA)
C-----------------------------------------------------------------------
C     GENERATES A PHOTON IN WEIZSACKER-WILLIAMS (WWA=.TRUE.) OR
C     ELSE EQUIVALENT PHOTON APPROX FROM INCOMING E+, E-, MU+ OR MU-
C-----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWRGEN,HWRUNI,EGMIN,ZMIN,ZMAX,ZGAM,SS,ZMI,ZMA,
     & PPL,PMI,QT2,Q2,QQMIN,QQMAX,S0,A,RPM(2)
      INTEGER IHEP,IHADIS,HQ,I
      LOGICAL WWA
      DOUBLE PRECISION F,C,FN
      DOUBLE PRECISION B,alphap,zh1
      PARAMETER (zh1=0.003)
      EXTERNAL HWRGEN,HWRUNI
      DATA EGMIN/5.D0/
      IF (IERROR.NE.0)  RETURN
      IF (IHEP.LT.1.OR.IHEP.GT.2) CALL HWWARN('HWEGAM',500,*999)
      SS=PHEP(5,3)
      IF (IHEP.EQ.1) THEN
        IHADIS=2
      ELSE
        IHADIS=1
        IF (JDAHEP(1,IHADIS).NE.0) IHADIS=JDAHEP(1,IHADIS)
      ENDIF
C---DEFINE LIMITS FOR GAMMA MOMENTUM FRACTION
      IF (ZMI.LE.0D0 .OR. ZMA.GT.1D0) THEN
        IF (IPRO.EQ.13.OR.IPRO.EQ.14) THEN
          S0 = EMMIN**2
        ELSEIF(IPRO.EQ.15.OR.IPRO.EQ.18.OR.IPRO.EQ.22.OR.IPRO.EQ.24.OR.
     &         IPRO.EQ.50.OR.IPRO.EQ.53.OR.IPRO.EQ.55)THEN
          S0 = 4.D0*PTMIN**2
        ELSEIF (IPRO.EQ.17.OR.IPRO.EQ.51) THEN
          HQ = MOD(IPROC,100)
          S0 = 4.D0*(PTMIN**2+RMASS(HQ)**2)
        ELSEIF (IPRO.EQ.16.OR.IPRO.EQ.19.OR.IPRO.EQ.95) THEN
          S0 = MAX(2*RMASS(1),RMASS(201)-GAMMAX*GAMH)**2
        ELSEIF (IPRO.EQ.23) THEN
          S0 = MAX(2*RMASS(1),RMASS(201)-GAMMAX*GAMH)**2
          S0 = (PTMIN+SQRT(PTMIN**2+S0))**2
        ELSEIF (IPRO.EQ.20) THEN
          S0 = RMASS(201)**2
        ELSEIF (IPRO.EQ.21) THEN
          S0 = (PTMIN+SQRT(PTMIN**2+RMASS(198)**2))**2
C--PR MOD 7/7/99
        ELSEIF(IPRO.EQ.30) THEN
          S0 = 4.0D0*(PTMIN**2+RMMNSS**2)
        ELSEIF(IPRO.EQ.40.OR.IPRO.EQ.41) THEN
          HQ = IPROC-100*IPRO
          RPM(1) = RMMNSS
          RPM(2) = ZERO
          IF(HQ.GE.10.AND.HQ.LT.20) THEN
            RPM(1) = ABS(RMASS(450))
            IF(HQ.GT.10) RPM(1) = ABS(RMASS(449+MOD(HQ,10)))
          ELSEIF(HQ.GE.20.AND.HQ.LT.30) THEN
            RPM(1) = ABS(RMASS(454))
            IF(HQ.GT.20) RPM(1) = ABS(RMASS(453+MOD(HQ,20)))
          ELSEIF(HQ.EQ.30) THEN
            RPM(1) = RMASS(449)
          ELSEIF(HQ.EQ.40) THEN
            IF(IPRO.EQ.40) THEN
              RPM(1) = RMASS(425)
              DO I=1,5
                RPM(1) = MIN(RPM(1),RMASS(425+I))
              ENDDO
            ELSE
              RPM(1) = MIN(RMASS(405),RMASS(406))
            ENDIF
            RPM(2) = RMASS(198)
          ELSEIF(HQ.EQ.50) THEN
            IF(IPRO.EQ.40) THEN
              RPM(1) = RMASS(425)
              DO I=1,5
                RPM(1) = MIN(RPM(1),RMASS(425+I))
              ENDDO
              DO I=1,3
                RPM(2) = MIN(RPM(1),RMASS(433+2*I))
              ENDDO
             RPM(1) = MIN(RPM(1),RPM(2))
              RPM(2) = RMASS(203)
              DO I=1,2
                RPM(2) = MIN(RPM(2),RMASS(204+I))
              ENDDO
            ELSE
              RPM(1) = RMASS(401)
              RPM(2) = RMASS(413)
              DO I=1,5
                RPM(1) = MIN(RPM(1),RMASS(401+I))
                RPM(2) = MIN(RPM(2),RMASS(413+I))
              ENDDO
              RPM(1) = MIN(RPM(1),RPM(2))
              RPM(2) = RMASS(203)
              DO I=1,2
                RPM(2) = MIN(RPM(2),RMASS(204+I))
              ENDDO
            ENDIF
            RPM(2) = RMASS(203)
            DO I=1,2
              RPM(2) = MIN(RPM(2),RMASS(204+I))
            ENDDO
          ELSEIF(HQ.GE.60) THEN
            RPM(1) = ZERO
          ENDIF
          RPM(1) = RPM(1)**2
          RPM(2) = RPM(2)**2
          S0 = RPM(1)+RPM(2)+TWO*(PTMIN**2+
     &         SQRT(RPM(1)*RPM(2)+PTMIN**2*(RPM(1)+RPM(2)+PTMIN**2)))
C--end of mod
        ELSEIF (IPRO.EQ.52) THEN
          HQ = MOD(IPROC,100)
          S0 = (PTMIN+SQRT(PTMIN**2+RMASS(HQ)**2))**2
        ELSEIF (IPRO.EQ.80) THEN
          S0 = WHMIN**2
        ELSEIF (IPRO.EQ.90) THEN
          S0 = Q2MIN
        ELSEIF (IPRO.EQ.91.OR.IPRO.EQ.92) THEN
          S0 = Q2MIN+4.D0*PTMIN**2
          HQ = MOD(IPROC,100)
          IF (HQ.GT.0) S0 = S0+4.D0*RMASS(HQ)**2
          IF (IPRO.EQ.91) S0 = MAX(S0,EMMIN**2)
        ELSE
          S0 = 0
        ENDIF
        IF (S0.GT.0) THEN
          S0 = (SQRT(S0)+ABS(PHEP(5,IHADIS)))**2-PHEP(5,IHADIS)**2
          S0 = MAX(S0,WHMIN**2)
          ZMIN = S0 / (SS**2 - PHEP(5,IHEP)**2 - PHEP(5,IHADIS)**2)
          ZMAX = ONE
        ELSE
C---UNKNOWN PROCESS: USE ENERGY CUTOFF, AND WARN USER
          IF (FSTWGT) CALL HWWARN('HWEGAM',1,*999)
          ZMIN = EGMIN / PHEP(4,IHEP)
          ZMAX = ONE
        ENDIF
      ELSE
        ZMIN=ZMI
        ZMAX=ZMA
      ENDIF
C---APPLY USER DEFINED CUTS YWWMIN,YWWMAX AND INDIRECT LIMITS ON Z
      IF (.NOT.WWA) THEN
        ZMIN=MAX(ZMIN,YWWMIN,SQRT(Q2WWMN)/ABS(PHEP(3,IHEP)))
        ZMAX=MIN(ZMAX,YWWMAX)
        IF (ZMIN.GT.ZMAX) THEN
          GAMWT=ZERO
          RETURN
        ENDIF
      ENDIF
C---Q2WWMN AND Q2WWMX ARE USER-DEFINED LIMITS IN THE Q**2 INTEGRATION
        QQMAX=Q2WWMX
        QQMIN=Q2WWMN
C        IF (QQMIN.GT.QQMAX) CALL HWWARN('HWEGAM',50,*10)
C---GENERATE GAMMA MOMENTUM FRACTION
        ZGAM=(ZMIN/ZMAX)**HWRGEN(1)*ZMAX
C---POMERON (REGGEON) FLUX
C     . B.C. 11/02/07
        IF (NSTRU.GE.9.AND.NSTRU.LE.11) THEN
           CALL FLUX(F,ZGAM,QQMIN,QQMAX,NSTRU)
           CALL FLUX(FN,ZH1,QQMIN,QQMAX,NSTRU)
        ELSEIF (NSTRU.EQ.12) THEN ! POMERON 2006 FIT A
           CALL h12006flux(ZGAM,-QQMAX,1,1,1,F)
           CALL h12006flux(ZH1,-QQMAX,1,1,1,FN)
        ELSEIF (NSTRU.EQ.13) THEN ! REGGEON 2006 FIT A
           CALL h12006flux(ZGAM,-QQMAX,1,1,2,F)
           CALL h12006flux(ZH1,-QQMAX,1,1,2,FN)
        ELSEIF (NSTRU.EQ.14) THEN ! POMERON 2006 FIT B
           CALL h12006flux(ZGAM,-QQMAX,1,2,1,F)
           CALL h12006flux(ZH1,-QQMAX,1,2,1,FN)
        ELSEIF (NSTRU.EQ.15) THEN ! REGGEON 2006 FIT B
           CALL h12006flux(ZGAM,-QQMAX,1,2,2,F)
           CALL h12006flux(ZH1,-QQMAX,1,2,2,FN)
        ELSE
           WRITE(*,*) 'HWEGAM : POMWIG : NSTRU OUT OF RANGE'
           STOP
        ENDIF
           

C---CALCULATE GAMWT
        C=1.D0/DLOG(ZMAX/ZMIN)
C B.C. 11/02/07
C        IF (NSTRU.EQ.9) THEN
        IF (NSTRU.EQ.9.OR.NSTRU.EQ.12.OR.NSTRU.EQ.14) THEN
           GAMWT = GAMWT*F*ZGAM/(C*FN*zh1)   
        ELSEIF (NSTRU.EQ.10.OR.NSTRU.EQ.11.OR.NSTRU.EQ.13
     +          .OR.NSTRU.EQ.15) THEN 
           GAMWT = GAMWT*F*ZGAM/C             
        ELSE
           WRITE(*,*) 'POMWIG : NSTRU OUT OF RANGE'
           STOP
        ENDIF

C B. C. 11/02/07 FOr H1 2006 pdfs use alphap and B from the H1 flux as in h12006flux.f   
C. Also fix bug in original pomwig to use alphap and B for reggeon (previous versions 
C  used pomeron alphap and B for reggeon flux)
        
cc JVY: "local" fix - set defaults to avoid compiler's complains of uninit variables
        b = -1.0d0
	alphap = 1.0d0

C---  PICK T (=Q2)        
        IF (NSTRU.EQ.9) THEN
           alphap=0.26D0
           B=4.6D0
        ELSEIF (NSTRU.EQ.10) THEN
           alphap = 0.9D0
           B = 2.0D0
        ELSEIF (NSTRU.EQ.11) THEN
           alphap=0.26D0
           B=4.6D0
        ELSEIF (NSTRU.EQ.12.OR.NSTRU.EQ.14) THEN
           alphap=0.06D0
           B=5.5D0
        ELSEIF (NSTRU.EQ.13.OR.NSTRU.EQ.15) THEN
           alphap=0.3D0
           B=1.6D0
        ENDIF
        C=B+alphap*DLOG(1.D0/ZGAM)
        Q2=(1.D0/C)*DLOG(1.D0/(DEXP(-C*QQMAX)+HWRGEN(1)*
     +       (DEXP(-C*QQMIN)-DEXP(-C*QQMAX))))
        IF (GAMWT.LT.ZERO) GAMWT=ZERO
C---  FILL PHOTON
      NHEP=NHEP+1
      IDHW(NHEP)=59
      ISTHEP(NHEP)=3
      IDHEP(NHEP)=22
      JMOHEP(1,NHEP)=IHEP
      JMOHEP(2,NHEP)=0
      JDAHEP(1,NHEP)=0
      JDAHEP(2,NHEP)=0
      JDAHEP(1,IHEP)=NHEP
      IF (WWA) THEN
C---FOR COLLINEAR KINEMATICS, ZGAM IS THE ENERGY FRACTION
        PHEP(4,NHEP)=PHEP(4,IHEP)*ZGAM
        PHEP(3,NHEP)=PHEP(3,IHEP)-SIGN(SQRT(
     &     (PHEP(4,IHEP)-PHEP(4,NHEP))**2-PHEP(5,IHEP)**2),PHEP(3,IHEP))
        PHEP(2,NHEP)=0
        PHEP(1,NHEP)=0
        CALL HWUMAS(PHEP(1,NHEP))
      ELSE
C---FOR EXACT KINEMATICS, ZGAM IS TAKEN TO BE FRACTION OF (E+PZ)
        PPL=ZGAM*(ABS(PHEP(3,IHEP))+PHEP(4,IHEP))
        QT2=(ONE-ZGAM)*Q2-(ZGAM*PHEP(5,IHEP))**2
        PMI=(QT2-Q2)/PPL
        PHEP(5,NHEP)=-SQRT(Q2)
        PHEP(4,NHEP)=(PPL+PMI)/TWO
        PHEP(3,NHEP)=SIGN((PPL-PMI)/TWO,PHEP(3,IHEP))
        CALL HWRAZM(SQRT(QT2),PHEP(1,NHEP),PHEP(2,NHEP))
      ENDIF
C---UPDATE OVERALL CM FRAME
      JMOHEP(IHEP,3)=NHEP
      CALL HWVDIF(4,PHEP(1,3),PHEP(1,IHEP),PHEP(1,3))
      CALL HWVSUM(4,PHEP(1,NHEP),PHEP(1,3),PHEP(1,3))
      CALL HWUMAS(PHEP(1,3))
C---FILL OUTGOING LEPTON
      NHEP=NHEP+1
C--- If outgoing particle is positron, change to anti-proton
C--- If outgoing particle is electron, change to proton
C--- B.C. 16/06/04
C--      IDHW(NHEP)=IDHW(IHEP)
      if (IDHW(IHEP).eq.121) then
         IDHW(NHEP)=73
         IDHEP(NHEP)=2212
      else
         IDHW(NHEP)=91
         IDHEP(NHEP)=-2212
      endif
      ISTHEP(NHEP)=1
c--      IDHEP(NHEP)=IDHEP(IHEP)
      JMOHEP(1,NHEP)=IHEP
      JMOHEP(2,NHEP)=0
      JDAHEP(1,NHEP)=0
      JDAHEP(2,NHEP)=0
      JDAHEP(2,IHEP)=NHEP
      CALL HWVDIF(4,PHEP(1,IHEP),PHEP(1,NHEP-1),PHEP(1,NHEP))
      PHEP(5,NHEP)=PHEP(5,IHEP)
 999  END
