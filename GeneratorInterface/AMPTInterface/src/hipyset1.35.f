c.................... hipyset1.35.f
C
C
C
C     Modified for HIJING program
c
c    modification July 22, 1997  In pyremnn put an upper limit
c     on the total pt kick the parton can accumulate via multiple
C     scattering. Set the upper limit to be the sqrt(s)/2,
c     this is fix cronin bug for Pb+Pb events at SPS energy.
c
C
C Last modification Oct. 1993 to comply with non-vax
C machines' compiler 
C
C*********************************************************************  
    
      SUBROUTINE LU2ENT(IP,KF1,KF2,PECM)    
    
C...Purpose: to store two partons/particles in their CM frame,  
C...with the first along the +z axis.   
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
    
C...Standard checks.    
      MSTU(28)=0    
      IF(MSTU(12).GE.1) CALL LULIST(0)  
      IPA=MAX(1,IABS(IP))   
      IF(IPA.GT.MSTU(4)-1) CALL LUERRM(21,  
     &'(LU2ENT:) writing outside LUJETS memory')    
      KC1=LUCOMP(KF1)   
      KC2=LUCOMP(KF2)   
      IF(KC1.EQ.0.OR.KC2.EQ.0) CALL LUERRM(12,  
     &'(LU2ENT:) unknown flavour code') 
    
C...Find masses. Reset K, P and V vectors.  
      PM1=0.    
      IF(MSTU(10).EQ.1) PM1=P(IPA,5)    
      IF(MSTU(10).GE.2) PM1=ULMASS(KF1) 
      PM2=0.    
      IF(MSTU(10).EQ.1) PM2=P(IPA+1,5)  
      IF(MSTU(10).GE.2) PM2=ULMASS(KF2) 
      DO 100 I=IPA,IPA+1    
      DO 100 J=1,5  
      K(I,J)=0  
      P(I,J)=0. 
  100 V(I,J)=0. 
    
C...Check flavours. 
      KQ1=KCHG(KC1,2)*ISIGN(1,KF1)  
      KQ2=KCHG(KC2,2)*ISIGN(1,KF2)  
      IF(KQ1+KQ2.NE.0.AND.KQ1+KQ2.NE.4) CALL LUERRM(2,  
     &'(LU2ENT:) unphysical flavour combination')   
      K(IPA,2)=KF1  
      K(IPA+1,2)=KF2    
    
C...Store partons/particles in K vectors for normal case.   
      IF(IP.GE.0) THEN  
        K(IPA,1)=1  
        IF(KQ1.NE.0.AND.KQ2.NE.0) K(IPA,1)=2    
        K(IPA+1,1)=1    
    
C...Store partons in K vectors for parton shower evolution. 
      ELSE  
        IF(KQ1.EQ.0.OR.KQ2.EQ.0) CALL LUERRM(2, 
     &  '(LU2ENT:) requested flavours can not develop parton shower')   
        K(IPA,1)=3  
        K(IPA+1,1)=3    
        K(IPA,4)=MSTU(5)*(IPA+1)    
        K(IPA,5)=K(IPA,4)   
        K(IPA+1,4)=MSTU(5)*IPA  
        K(IPA+1,5)=K(IPA+1,4)   
      ENDIF 
    
C...Check kinematics and store partons/particles in P vectors.  
      IF(PECM.LE.PM1+PM2) CALL LUERRM(13,   
     &'(LU2ENT:) energy smaller than sum of masses')    
      PA=SQRT(MAX(0.,(PECM**2-PM1**2-PM2**2)**2-(2.*PM1*PM2)**2))/  
     &(2.*PECM) 
      P(IPA,3)=PA   
      P(IPA,4)=SQRT(PM1**2+PA**2)   
      P(IPA,5)=PM1  
      P(IPA+1,3)=-PA    
      P(IPA+1,4)=SQRT(PM2**2+PA**2) 
      P(IPA+1,5)=PM2    
    
C...Set N. Optionally fragment/decay.   
      N=IPA+1   
      IF(IP.EQ.0) CALL LUEXEC   
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUGIVE(CHIN)   
    
C...Purpose: to set values of commonblock variables.    
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/LUDAT4/CHAF(500)   
      CHARACTER CHAF*8  
      SAVE /LUDAT4/ 
      CHARACTER CHIN*(*),CHFIX*104,CHBIT*104,CHOLD*8,CHNEW*8,   
     &CHNAM*4,CHVAR(17)*4,CHALP(2)*26,CHIND*8,CHINI*10,CHINR*16 
      DATA CHVAR/'N','K','P','V','MSTU','PARU','MSTJ','PARJ','KCHG',    
     &'PMAS','PARF','VCKM','MDCY','MDME','BRAT','KFDP','CHAF'/  
      DATA CHALP/'abcdefghijklmnopqrstuvwxyz',  
     &'ABCDEFGHIJKLMNOPQRSTUVWXYZ'/ 
    
C...Length of character variable. Subdivide it into instructions.   
      IF(MSTU(12).GE.1) CALL LULIST(0)  
      CHBIT=CHIN//' '   
      LBIT=101  
  100 LBIT=LBIT-1   
      IF(CHBIT(LBIT:LBIT).EQ.' ') GOTO 100  
      LTOT=0    
      DO 110 LCOM=1,LBIT    
      IF(CHBIT(LCOM:LCOM).EQ.' ') GOTO 110  
      LTOT=LTOT+1   
      CHFIX(LTOT:LTOT)=CHBIT(LCOM:LCOM) 
  110 CONTINUE  
      LLOW=0    
  120 LHIG=LLOW+1   
  130 LHIG=LHIG+1   
      IF(LHIG.LE.LTOT.AND.CHFIX(LHIG:LHIG).NE.';') GOTO 130 
      LBIT=LHIG-LLOW-1  
      CHBIT(1:LBIT)=CHFIX(LLOW+1:LHIG-1)    
    
C...Identify commonblock variable.  
      LNAM=1    
  140 LNAM=LNAM+1   
      IF(CHBIT(LNAM:LNAM).NE.'('.AND.CHBIT(LNAM:LNAM).NE.'='.AND.   
     &LNAM.LE.4) GOTO 140   
      CHNAM=CHBIT(1:LNAM-1)//' '    
      DO 150 LCOM=1,LNAM-1  
      DO 150 LALP=1,26  
  150 IF(CHNAM(LCOM:LCOM).EQ.CHALP(1)(LALP:LALP)) CHNAM(LCOM:LCOM)= 
     &CHALP(2)(LALP:LALP)   
      IVAR=0    
      DO 160 IV=1,17    
  160 IF(CHNAM.EQ.CHVAR(IV)) IVAR=IV    
      IF(IVAR.EQ.0) THEN    
        CALL LUERRM(18,'(LUGIVE:) do not recognize variable '//CHNAM)   
        LLOW=LHIG   
        IF(LLOW.LT.LTOT) GOTO 120   
        RETURN  
      ENDIF 
    
C...Identify any indices.   
      I=0   
      J=0   
      IF(CHBIT(LNAM:LNAM).EQ.'(') THEN  
        LIND=LNAM   
  170   LIND=LIND+1 
        IF(CHBIT(LIND:LIND).NE.')'.AND.CHBIT(LIND:LIND).NE.',') GOTO 170    
        CHIND=' '   
        IF((CHBIT(LNAM+1:LNAM+1).EQ.'C'.OR.CHBIT(LNAM+1:LNAM+1).EQ.'c').    
     &  AND.(IVAR.EQ.9.OR.IVAR.EQ.10.OR.IVAR.EQ.13.OR.IVAR.EQ.17)) THEN 
          CHIND(LNAM-LIND+11:8)=CHBIT(LNAM+2:LIND-1)    
          READ(CHIND,'(I8)') I1 
          I=LUCOMP(I1)  
        ELSE    
          CHIND(LNAM-LIND+10:8)=CHBIT(LNAM+1:LIND-1)    
          READ(CHIND,'(I8)') I  
        ENDIF   
        LNAM=LIND   
        IF(CHBIT(LNAM:LNAM).EQ.')') LNAM=LNAM+1 
      ENDIF 
      IF(CHBIT(LNAM:LNAM).EQ.',') THEN  
        LIND=LNAM   
  180   LIND=LIND+1 
        IF(CHBIT(LIND:LIND).NE.')'.AND.CHBIT(LIND:LIND).NE.',') GOTO 180    
        CHIND=' '   
        CHIND(LNAM-LIND+10:8)=CHBIT(LNAM+1:LIND-1)  
        READ(CHIND,'(I8)') J    
        LNAM=LIND+1 
      ENDIF 
C...cms initialize variable
      CHOLD=' '
C...Check that indices allowed and save old value.  
      IERR=1    
      IF(CHBIT(LNAM:LNAM).NE.'=') GOTO 190  
      IF(IVAR.EQ.1) THEN    
        IF(I.NE.0.OR.J.NE.0) GOTO 190   
        IOLD=N  
      ELSEIF(IVAR.EQ.2) THEN    
        IF(I.LT.1.OR.I.GT.MSTU(4).OR.J.LT.1.OR.J.GT.5) GOTO 190 
        IOLD=K(I,J) 
      ELSEIF(IVAR.EQ.3) THEN    
        IF(I.LT.1.OR.I.GT.MSTU(4).OR.J.LT.1.OR.J.GT.5) GOTO 190 
        ROLD=P(I,J) 
      ELSEIF(IVAR.EQ.4) THEN    
        IF(I.LT.1.OR.I.GT.MSTU(4).OR.J.LT.1.OR.J.GT.5) GOTO 190 
        ROLD=V(I,J) 
      ELSEIF(IVAR.EQ.5) THEN    
        IF(I.LT.1.OR.I.GT.200.OR.J.NE.0) GOTO 190   
        IOLD=MSTU(I)    
      ELSEIF(IVAR.EQ.6) THEN    
        IF(I.LT.1.OR.I.GT.200.OR.J.NE.0) GOTO 190   
        ROLD=PARU(I)    
      ELSEIF(IVAR.EQ.7) THEN    
        IF(I.LT.1.OR.I.GT.200.OR.J.NE.0) GOTO 190   
        IOLD=MSTJ(I)    
      ELSEIF(IVAR.EQ.8) THEN    
        IF(I.LT.1.OR.I.GT.200.OR.J.NE.0) GOTO 190   
        ROLD=PARJ(I)    
      ELSEIF(IVAR.EQ.9) THEN    
        IF(I.LT.1.OR.I.GT.MSTU(6).OR.J.LT.1.OR.J.GT.3) GOTO 190 
        IOLD=KCHG(I,J)  
      ELSEIF(IVAR.EQ.10) THEN   
        IF(I.LT.1.OR.I.GT.MSTU(6).OR.J.LT.1.OR.J.GT.4) GOTO 190 
        ROLD=PMAS(I,J)  
      ELSEIF(IVAR.EQ.11) THEN   
        IF(I.LT.1.OR.I.GT.2000.OR.J.NE.0) GOTO 190  
        ROLD=PARF(I)    
      ELSEIF(IVAR.EQ.12) THEN   
        IF(I.LT.1.OR.I.GT.4.OR.J.LT.1.OR.J.GT.4) GOTO 190   
        ROLD=VCKM(I,J)  
      ELSEIF(IVAR.EQ.13) THEN   
        IF(I.LT.1.OR.I.GT.MSTU(6).OR.J.LT.1.OR.J.GT.3) GOTO 190 
        IOLD=MDCY(I,J)  
      ELSEIF(IVAR.EQ.14) THEN   
        IF(I.LT.1.OR.I.GT.MSTU(7).OR.J.LT.1.OR.J.GT.2) GOTO 190 
        IOLD=MDME(I,J)  
      ELSEIF(IVAR.EQ.15) THEN   
        IF(I.LT.1.OR.I.GT.MSTU(7).OR.J.NE.0) GOTO 190   
        ROLD=BRAT(I)    
      ELSEIF(IVAR.EQ.16) THEN   
        IF(I.LT.1.OR.I.GT.MSTU(7).OR.J.LT.1.OR.J.GT.5) GOTO 190 
        IOLD=KFDP(I,J)  
      ELSEIF(IVAR.EQ.17) THEN   
        IF(I.LT.1.OR.I.GT.MSTU(6).OR.J.NE.0) GOTO 190   
        CHOLD=CHAF(I)   
      ENDIF 
      IERR=0    
  190 IF(IERR.EQ.1) THEN    
        CALL LUERRM(18,'(LUGIVE:) unallowed indices for '// 
     &  CHBIT(1:LNAM-1))    
        LLOW=LHIG   
        IF(LLOW.LT.LTOT) GOTO 120   
        RETURN  
      ENDIF 
    
C...Print current value of variable. Loop back. 
      IF(LNAM.GE.LBIT) THEN 
        CHBIT(LNAM:14)=' '  
        CHBIT(15:60)=' has the value                                '   
        IF(IVAR.EQ.1.OR.IVAR.EQ.2.OR.IVAR.EQ.5.OR.IVAR.EQ.7.OR. 
     &  IVAR.EQ.9.OR.IVAR.EQ.13.OR.IVAR.EQ.14.OR.IVAR.EQ.16) THEN   
          WRITE(CHBIT(51:60),'(I10)') IOLD  
        ELSEIF(IVAR.NE.17) THEN 
          WRITE(CHBIT(47:60),'(F14.5)') ROLD    
        ELSE    
          CHBIT(53:60)=CHOLD    
        ENDIF   
        IF(MSTU(13).GE.1) WRITE(MSTU(11),1000) CHBIT(1:60)  
        LLOW=LHIG   
        IF(LLOW.LT.LTOT) GOTO 120   
        RETURN  
      ENDIF 
    
C...Read in new variable value. 
      IF(IVAR.EQ.1.OR.IVAR.EQ.2.OR.IVAR.EQ.5.OR.IVAR.EQ.7.OR.   
     &IVAR.EQ.9.OR.IVAR.EQ.13.OR.IVAR.EQ.14.OR.IVAR.EQ.16) THEN 
        CHINI=' '   
        CHINI(LNAM-LBIT+11:10)=CHBIT(LNAM+1:LBIT)   
        READ(CHINI,'(I10)') INEW    
      ELSEIF(IVAR.NE.17) THEN   
        CHINR=' '   
        CHINR(LNAM-LBIT+17:16)=CHBIT(LNAM+1:LBIT)   
        READ(CHINR,'(F16.2)') RNEW  
      ELSE  
        CHNEW=CHBIT(LNAM+1:LBIT)//' '   
      ENDIF 
    
C...Store new variable value.   
      IF(IVAR.EQ.1) THEN    
        N=INEW  
      ELSEIF(IVAR.EQ.2) THEN    
        K(I,J)=INEW 
      ELSEIF(IVAR.EQ.3) THEN    
        P(I,J)=RNEW 
      ELSEIF(IVAR.EQ.4) THEN    
        V(I,J)=RNEW 
      ELSEIF(IVAR.EQ.5) THEN    
        MSTU(I)=INEW    
      ELSEIF(IVAR.EQ.6) THEN    
        PARU(I)=RNEW    
      ELSEIF(IVAR.EQ.7) THEN    
        MSTJ(I)=INEW    
      ELSEIF(IVAR.EQ.8) THEN    
        PARJ(I)=RNEW    
      ELSEIF(IVAR.EQ.9) THEN    
        KCHG(I,J)=INEW  
      ELSEIF(IVAR.EQ.10) THEN   
        PMAS(I,J)=RNEW  
      ELSEIF(IVAR.EQ.11) THEN   
        PARF(I)=RNEW    
      ELSEIF(IVAR.EQ.12) THEN   
        VCKM(I,J)=RNEW  
      ELSEIF(IVAR.EQ.13) THEN   
        MDCY(I,J)=INEW  
      ELSEIF(IVAR.EQ.14) THEN   
        MDME(I,J)=INEW  
      ELSEIF(IVAR.EQ.15) THEN   
        BRAT(I)=RNEW    
      ELSEIF(IVAR.EQ.16) THEN   
        KFDP(I,J)=INEW  
      ELSEIF(IVAR.EQ.17) THEN   
        CHAF(I)=CHNEW   
      ENDIF 
    
C...Write old and new value. Loop back. 
      CHBIT(LNAM:14)=' '    
      CHBIT(15:60)=' changed from                to               ' 
      IF(IVAR.EQ.1.OR.IVAR.EQ.2.OR.IVAR.EQ.5.OR.IVAR.EQ.7.OR.   
     &IVAR.EQ.9.OR.IVAR.EQ.13.OR.IVAR.EQ.14.OR.IVAR.EQ.16) THEN 
        WRITE(CHBIT(33:42),'(I10)') IOLD    
        WRITE(CHBIT(51:60),'(I10)') INEW    
      ELSEIF(IVAR.NE.17) THEN   
        WRITE(CHBIT(29:42),'(F14.5)') ROLD  
        WRITE(CHBIT(47:60),'(F14.5)') RNEW  
      ELSE  
        CHBIT(35:42)=CHOLD  
        CHBIT(53:60)=CHNEW  
      ENDIF 
      IF(MSTU(13).GE.1) WRITE(MSTU(11),1000) CHBIT(1:60)    
      LLOW=LHIG 
      IF(LLOW.LT.LTOT) GOTO 120 
    
C...Format statement for output on unit MSTU(11) (by default 6).    
 1000 FORMAT(5X,A60)    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUEXEC 
    
C...Purpose: to administrate the fragmentation and decay chain. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      DIMENSION PS(2,6) 
    
C...Initialize and reset.   
      MSTU(24)=0    
      IF(MSTU(12).GE.1) CALL LULIST(0)  
      MSTU(31)=MSTU(31)+1   
      MSTU(1)=0 
      MSTU(2)=0 
      MSTU(3)=0 
      MCONS=1   
    
C...Sum up momentum, energy and charge for starting entries.    
      NSAV=N    
      DO 100 I=1,2  
      DO 100 J=1,6  
  100 PS(I,J)=0.    
      DO 120 I=1,N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 120  
      DO 110 J=1,4  
  110 PS(1,J)=PS(1,J)+P(I,J)    
      PS(1,6)=PS(1,6)+LUCHGE(K(I,2))    
  120 CONTINUE  
      PARU(21)=PS(1,4)  
    
C...Prepare system for subsequent fragmentation/decay.  
      CALL LUPREP(0)    
    
C...Loop through jet fragmentation and particle decays. 
      MBE=0 
  130 MBE=MBE+1 
      IP=0  
  140 IP=IP+1   
      KC=0  
      IF(K(IP,1).GT.0.AND.K(IP,1).LE.10) KC=LUCOMP(K(IP,2)) 
      IF(KC.EQ.0) THEN  
    
C...Particle decay if unstable and allowed. Save long-lived particle    
C...decays until second pass after Bose-Einstein effects.   
      ELSEIF(KCHG(KC,2).EQ.0) THEN  
clin-4/2008 break up compound IF statements:
c        IF(MSTJ(21).GE.1.AND.MDCY(KC,1).GE.1.AND.(MSTJ(51).LE.0.OR.MBE. 
c     &  EQ.2.OR.PMAS(KC,2).GE.PARJ(91).OR.IABS(K(IP,2)).EQ.311))    
c     &  CALL LUDECY(IP) 
         if(MSTJ(21).GE.1.AND.MDCY(KC,1).GE.1) then
            if(MSTJ(51).LE.0.OR.MBE.EQ.2.OR.PMAS(KC,2).GE.PARJ(91)
     &           .OR.IABS(K(IP,2)).EQ.311)
     &           CALL LUDECY(IP) 
         endif
c    
C...Decay products may develop a shower.    
        IF(MSTJ(92).GT.0) THEN  
          IP1=MSTJ(92)  
          QMAX=SQRT(MAX(0.,(P(IP1,4)+P(IP1+1,4))**2-(P(IP1,1)+P(IP1+1,  
     &    1))**2-(P(IP1,2)+P(IP1+1,2))**2-(P(IP1,3)+P(IP1+1,3))**2))    
          CALL LUSHOW(IP1,IP1+1,QMAX)   
          CALL LUPREP(IP1)  
          MSTJ(92)=0    
        ELSEIF(MSTJ(92).LT.0) THEN  
          IP1=-MSTJ(92) 
clin-8/19/02 avoid actual argument in common blocks of LUSHOW:
c          CALL LUSHOW(IP1,-3,P(IP,5))   
          pip5=P(IP,5)
          CALL LUSHOW(IP1,-3,pip5)   
          CALL LUPREP(IP1)  
          MSTJ(92)=0    
        ENDIF   
    
C...Jet fragmentation: string or independent fragmentation. 
      ELSEIF(K(IP,1).EQ.1.OR.K(IP,1).EQ.2) THEN 
        MFRAG=MSTJ(1)   
        IF(MFRAG.GE.1.AND.K(IP,1).EQ.1) MFRAG=2 
        IF(MSTJ(21).GE.2.AND.K(IP,1).EQ.2.AND.N.GT.IP) THEN 
          IF(K(IP+1,1).EQ.1.AND.K(IP+1,3).EQ.K(IP,3).AND.   
     &    K(IP,3).GT.0.AND.K(IP,3).LT.IP) THEN  
            IF(KCHG(LUCOMP(K(K(IP,3),2)),2).EQ.0) MFRAG=MIN(1,MFRAG)    
          ENDIF 
        ENDIF   
        IF(MFRAG.EQ.1) then
           CALL LUSTRF(IP)  
        endif
        IF(MFRAG.EQ.2) CALL LUINDF(IP)  
        IF(MFRAG.EQ.2.AND.K(IP,1).EQ.1) MCONS=0 
        IF(MFRAG.EQ.2.AND.(MSTJ(3).LE.0.OR.MOD(MSTJ(3),5).EQ.0)) MCONS=0    
      ENDIF 
    
C...Loop back if enough space left in LUJETS and no error abort.    
      IF(MSTU(24).NE.0.AND.MSTU(21).GE.2) THEN  
      ELSEIF(IP.LT.N.AND.N.LT.MSTU(4)-20-MSTU(32)) THEN 
        GOTO 140    
      ELSEIF(IP.LT.N) THEN  
        CALL LUERRM(11,'(LUEXEC:) no more memory left in LUJETS')   
      ENDIF 
    
C...Include simple Bose-Einstein effect parametrization if desired. 
      IF(MBE.EQ.1.AND.MSTJ(51).GE.1) THEN   
        CALL LUBOEI(NSAV)   
        GOTO 130    
      ENDIF 
    
C...Check that momentum, energy and charge were conserved.  
      DO 160 I=1,N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 160  
      DO 150 J=1,4  
  150 PS(2,J)=PS(2,J)+P(I,J)    
      PS(2,6)=PS(2,6)+LUCHGE(K(I,2))    
  160 CONTINUE  
      PDEV=(ABS(PS(2,1)-PS(1,1))+ABS(PS(2,2)-PS(1,2))+ABS(PS(2,3)-  
     &PS(1,3))+ABS(PS(2,4)-PS(1,4)))/(1.+ABS(PS(2,4))+ABS(PS(1,4))) 
      IF(MCONS.EQ.1.AND.PDEV.GT.PARU(11)) CALL LUERRM(15,   
     &'(LUEXEC:) four-momentum was not conserved')  
c      IF(MCONS.EQ.1.AND.PDEV.GT.PARU(11)) then
c         CALL LUERRM(15,   
c     &'(LUEXEC:) four-momentum was not conserved')  
c         write(6,*) 'PS1,2=',PS(1,1),PS(1,2),PS(1,3),PS(1,4),
c     1        '*',PS(2,1),PS(2,2),PS(2,3),PS(2,4)
c      endif

      IF(MCONS.EQ.1.AND.ABS(PS(2,6)-PS(1,6)).GT.0.1) CALL LUERRM(15,    
     &'(LUEXEC:) charge was not conserved') 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUPREP(IP) 
    
C...Purpose: to rearrange partons along strings, to allow small systems 
C...to collapse into one or two particles and to check flavours.    
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      DIMENSION DPS(5),DPC(5),UE(3) 
    
C...Rearrange parton shower product listing along strings: begin loop.  
      I1=N  
      DO 130 MQGST=1,2  
      DO 120 I=MAX(1,IP),N  
      IF(K(I,1).NE.3) GOTO 120  
      KC=LUCOMP(K(I,2)) 
      IF(KC.EQ.0) GOTO 120  
      KQ=KCHG(KC,2) 
      IF(KQ.EQ.0.OR.(MQGST.EQ.1.AND.KQ.EQ.2)) GOTO 120  
    
C...Pick up loose string end.   
      KCS=4 
      IF(KQ*ISIGN(1,K(I,2)).LT.0) KCS=5 
      IA=I  
      NSTP=0    
  100 NSTP=NSTP+1   
      IF(NSTP.GT.4*N) THEN  
        CALL LUERRM(14,'(LUPREP:) caught in infinite loop') 
        RETURN  
      ENDIF 
    
C...Copy undecayed parton.  
      IF(K(IA,1).EQ.3) THEN 
        IF(I1.GE.MSTU(4)-MSTU(32)-5) THEN   
          CALL LUERRM(11,'(LUPREP:) no more memory left in LUJETS') 
          RETURN    
        ENDIF   
        I1=I1+1 
        K(I1,1)=2   
        IF(NSTP.GE.2.AND.IABS(K(IA,2)).NE.21) K(I1,1)=1 
        K(I1,2)=K(IA,2) 
        K(I1,3)=IA  
        K(I1,4)=0   
        K(I1,5)=0   
        DO 110 J=1,5    
        P(I1,J)=P(IA,J) 
  110   V(I1,J)=V(IA,J) 
        K(IA,1)=K(IA,1)+10  
        IF(K(I1,1).EQ.1) GOTO 120   
      ENDIF 
    
C...Go to next parton in colour space.  
      IB=IA 
      IF(MOD(K(IB,KCS)/MSTU(5)**2,2).EQ.0.AND.MOD(K(IB,KCS),MSTU(5)).   
     &NE.0) THEN    
        IA=MOD(K(IB,KCS),MSTU(5))   
        K(IB,KCS)=K(IB,KCS)+MSTU(5)**2  
        MREV=0  
      ELSE  
        IF(K(IB,KCS).GE.2*MSTU(5)**2.OR.MOD(K(IB,KCS)/MSTU(5),MSTU(5)). 
     &  EQ.0) KCS=9-KCS 
        IA=MOD(K(IB,KCS)/MSTU(5),MSTU(5))   
        K(IB,KCS)=K(IB,KCS)+2*MSTU(5)**2    
        MREV=1  
      ENDIF 
      IF(IA.LE.0.OR.IA.GT.N) THEN   
        CALL LUERRM(12,'(LUPREP:) colour rearrangement failed') 
        RETURN  
      ENDIF 
      IF(MOD(K(IA,4)/MSTU(5),MSTU(5)).EQ.IB.OR.MOD(K(IA,5)/MSTU(5), 
     &MSTU(5)).EQ.IB) THEN  
        IF(MREV.EQ.1) KCS=9-KCS 
        IF(MOD(K(IA,KCS)/MSTU(5),MSTU(5)).NE.IB) KCS=9-KCS  
        K(IA,KCS)=K(IA,KCS)+2*MSTU(5)**2    
      ELSE  
        IF(MREV.EQ.0) KCS=9-KCS 
        IF(MOD(K(IA,KCS),MSTU(5)).NE.IB) KCS=9-KCS  
        K(IA,KCS)=K(IA,KCS)+MSTU(5)**2  
      ENDIF 
      IF(IA.NE.I) GOTO 100  
      K(I1,1)=1 
  120 CONTINUE  
  130 CONTINUE  
      N=I1  
    
C...Find lowest-mass colour singlet jet system, OK if above thresh.  
      IF(MSTJ(14).LE.0) GOTO 320    
      NS=N  
  140 NSIN=N-NS 
      PDM=1.+PARJ(32)   
      IC=0
      IC1=0
      IC2=0
      DO 190 I=MAX(1,IP),NS 
      IF(K(I,1).NE.1.AND.K(I,1).NE.2) THEN  
      ELSEIF(K(I,1).EQ.2.AND.IC.EQ.0) THEN  
        NSIN=NSIN+1 
        IC=I    
        DO 150 J=1,4    
  150   DPS(J)=dble(P(I,J))
        MSTJ(93)=1  
        DPS(5)=dble(ULMASS(K(I,2)))
      ELSEIF(K(I,1).EQ.2) THEN  
        DO 160 J=1,4    
  160   DPS(J)=DPS(J)+dble(P(I,J))
      ELSEIF(IC.NE.0.AND.KCHG(LUCOMP(K(I,2)),2).NE.0) THEN  
        DO 170 J=1,4    
  170   DPS(J)=DPS(J)+dble(P(I,J))
        MSTJ(93)=1  
        DPS(5)=DPS(5)+dble(ULMASS(K(I,2)))
        PD=sngl(SQRT(MAX(0D0,DPS(4)**2
     1       -DPS(1)**2-DPS(2)**2-DPS(3)**2))-DPS(5))    
        IF(PD.LT.PDM) THEN  
          PDM=PD    
          DO 180 J=1,5  
  180     DPC(J)=DPS(J) 
          IC1=IC    
          IC2=I 
        ENDIF   
        IC=0    
      ELSE  
        NSIN=NSIN+1 
      ENDIF 
  190 CONTINUE  
      IF(PDM.GE.PARJ(32)) GOTO 320  
    
C...Fill small-mass system as cluster.  
      NSAV=N    
      PECM=sngl(SQRT(MAX(0D0,DPC(4)**2-DPC(1)**2-DPC(2)**2-DPC(3)**2)))
      K(N+1,1)=11   
      K(N+1,2)=91   
      K(N+1,3)=IC1  
      K(N+1,4)=N+2  
      K(N+1,5)=N+3  
      P(N+1,1)=sngl(DPC(1))
      P(N+1,2)=sngl(DPC(2))  
      P(N+1,3)=sngl(DPC(3))  
      P(N+1,4)=sngl(DPC(4))
      P(N+1,5)=PECM 
    
C...Form two particles from flavours of lowest-mass system, if feasible.    
      K(N+2,1)=1    
      K(N+3,1)=1    
      IF(MSTU(16).NE.2) THEN    
        K(N+2,3)=N+1    
        K(N+3,3)=N+1    
      ELSE  
        K(N+2,3)=IC1    
        K(N+3,3)=IC2    
      ENDIF 
      K(N+2,4)=0    
      K(N+3,4)=0    
      K(N+2,5)=0    
      K(N+3,5)=0    
      IF(IABS(K(IC1,2)).NE.21) THEN 
        KC1=LUCOMP(K(IC1,2))    
        KC2=LUCOMP(K(IC2,2))    
        IF(KC1.EQ.0.OR.KC2.EQ.0) GOTO 320   
        KQ1=KCHG(KC1,2)*ISIGN(1,K(IC1,2))   
        KQ2=KCHG(KC2,2)*ISIGN(1,K(IC2,2))   
        IF(KQ1+KQ2.NE.0) GOTO 320   
  200   CALL LUKFDI(K(IC1,2),0,KFLN,K(N+2,2))   
        CALL LUKFDI(K(IC2,2),-KFLN,KFLDMP,K(N+3,2)) 
        IF(K(N+2,2).EQ.0.OR.K(N+3,2).EQ.0) GOTO 200 
      ELSE  
        IF(IABS(K(IC2,2)).NE.21) GOTO 320   
  210   CALL LUKFDI(1+INT((2.+PARJ(2))*RLU(0)),0,KFLN,KFDMP)    
        CALL LUKFDI(KFLN,0,KFLM,K(N+2,2))   
        CALL LUKFDI(-KFLN,-KFLM,KFLDMP,K(N+3,2))    
        IF(K(N+2,2).EQ.0.OR.K(N+3,2).EQ.0) GOTO 210 
      ENDIF 
      P(N+2,5)=ULMASS(K(N+2,2)) 
      P(N+3,5)=ULMASS(K(N+3,2)) 
      IF(P(N+2,5)+P(N+3,5)+PARJ(64).GE.PECM.AND.NSIN.EQ.1) GOTO 320 
      IF(P(N+2,5)+P(N+3,5)+PARJ(64).GE.PECM) GOTO 260   
    
C...Perform two-particle decay of jet system, if possible.  
clin-5/2012:
c      IF(PECM.GE.0.02d0*DPC(4)) THEN  
      IF(dble(PECM).GE.0.02d0*DPC(4)) THEN  
        PA=SQRT((PECM**2-(P(N+2,5)+P(N+3,5))**2)*(PECM**2-  
     &  (P(N+2,5)-P(N+3,5))**2))/(2.*PECM)  
        UE(3)=2.*RLU(0)-1.  
        PHI=PARU(2)*RLU(0)  
        UE(1)=SQRT(1.-UE(3)**2)*COS(PHI)    
        UE(2)=SQRT(1.-UE(3)**2)*SIN(PHI)    
        DO 220 J=1,3    
        P(N+2,J)=PA*UE(J)   
  220   P(N+3,J)=-PA*UE(J)  
        P(N+2,4)=SQRT(PA**2+P(N+2,5)**2)    
        P(N+3,4)=SQRT(PA**2+P(N+3,5)**2)    
        CALL LUDBRB(N+2,N+3,0.,0.,DPC(1)/DPC(4),DPC(2)/DPC(4),  
     &  DPC(3)/DPC(4))  
      ELSE  
        NP=0    
        DO 230 I=IC1,IC2    
  230   IF(K(I,1).EQ.1.OR.K(I,1).EQ.2) NP=NP+1  
        HA=P(IC1,4)*P(IC2,4)-P(IC1,1)*P(IC2,1)-P(IC1,2)*P(IC2,2)-   
     &  P(IC1,3)*P(IC2,3)   
        IF(NP.GE.3.OR.HA.LE.1.25*P(IC1,5)*P(IC2,5)) GOTO 260    
        HD1=0.5*(P(N+2,5)**2-P(IC1,5)**2)   
        HD2=0.5*(P(N+3,5)**2-P(IC2,5)**2)   
        HR=SQRT(MAX(0.,((HA-HD1-HD2)**2-(P(N+2,5)*P(N+3,5))**2)/    
     &  (HA**2-(P(IC1,5)*P(IC2,5))**2)))-1. 
        HC=P(IC1,5)**2+2.*HA+P(IC2,5)**2    
        HK1=((P(IC2,5)**2+HA)*HR+HD1-HD2)/HC    
        HK2=((P(IC1,5)**2+HA)*HR+HD2-HD1)/HC    
        DO 240 J=1,4    
        P(N+2,J)=(1.+HK1)*P(IC1,J)-HK2*P(IC2,J) 
  240   P(N+3,J)=(1.+HK2)*P(IC2,J)-HK1*P(IC1,J) 
      ENDIF 
      DO 250 J=1,4  
      V(N+1,J)=V(IC1,J) 
      V(N+2,J)=V(IC1,J) 
  250 V(N+3,J)=V(IC2,J) 
      V(N+1,5)=0.   
      V(N+2,5)=0.   
      V(N+3,5)=0.   
      N=N+3 
      GOTO 300  
    
C...Else form one particle from the flavours available, if possible.    
  260 K(N+1,5)=N+2  
      IF(IABS(K(IC1,2)).GT.100.AND.IABS(K(IC2,2)).GT.100) THEN  
        GOTO 320    
      ELSEIF(IABS(K(IC1,2)).NE.21) THEN 
        CALL LUKFDI(K(IC1,2),K(IC2,2),KFLDMP,K(N+2,2))  
      ELSE  
        KFLN=1+INT((2.+PARJ(2))*RLU(0)) 
        CALL LUKFDI(KFLN,-KFLN,KFLDMP,K(N+2,2)) 
      ENDIF 
      IF(K(N+2,2).EQ.0) GOTO 260    
      P(N+2,5)=ULMASS(K(N+2,2)) 
    
C...Find parton/particle which combines to largest extra mass.  
      IR=0  
      HA=0. 
      DO 280 MCOMB=1,3  
      IF(IR.NE.0) GOTO 280  
      DO 270 I=MAX(1,IP),N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10.OR.(I.GE.IC1.AND.I.LE.IC2. 
     &AND.K(I,1).GE.1.AND.K(I,1).LE.2)) GOTO 270    
      IF(MCOMB.EQ.1) KCI=LUCOMP(K(I,2)) 
      IF(MCOMB.EQ.1.AND.KCI.EQ.0) GOTO 270  
      IF(MCOMB.EQ.1.AND.KCHG(KCI,2).EQ.0.AND.I.LE.NS) GOTO 270  
      IF(MCOMB.EQ.2.AND.IABS(K(I,2)).GT.10.AND.IABS(K(I,2)).LE.100) 
     &GOTO 270  
      HCR=sngl(DPC(4))*P(I,4)-sngl(DPC(1))*P(I,1)
     1     -sngl(DPC(2))*P(I,2)-sngl(DPC(3))*P(I,3)   
      IF(HCR.GT.HA) THEN    
        IR=I    
        HA=HCR  
      ENDIF 
  270 CONTINUE  
  280 CONTINUE  
    
C...Shuffle energy and momentum to put new particle on mass shell.  
      HB=PECM**2+HA 
      HC=P(N+2,5)**2+HA 
      HD=P(IR,5)**2+HA
C******************CHANGES BY HIJING************  
      HK2=0.0
      IF(HA**2-(PECM*P(IR,5))**2.EQ.0.0.OR.HB+HD.EQ.0.0) GO TO 285
C******************
      HK2=0.5*(HB*SQRT(((HB+HC)**2-4.*(HB+HD)*P(N+2,5)**2)/ 
     &(HA**2-(PECM*P(IR,5))**2))-(HB+HC))/(HB+HD)   
  285 HK1=(0.5*(P(N+2,5)**2-PECM**2)+HD*HK2)/HB 
      DO 290 J=1,4  
      P(N+2,J)=(1.+HK1)*sngl(DPC(J))-HK2*P(IR,J)  
      P(IR,J)=(1.+HK2)*P(IR,J)-HK1*sngl(DPC(J))
      V(N+1,J)=V(IC1,J) 
  290 V(N+2,J)=V(IC1,J) 
      V(N+1,5)=0.   
      V(N+2,5)=0.   
      N=N+2 
    
C...Mark collapsed system and store daughter pointers. Iterate. 
  300 DO 310 I=IC1,IC2  
      IF((K(I,1).EQ.1.OR.K(I,1).EQ.2).AND.KCHG(LUCOMP(K(I,2)),2).NE.0)  
     &THEN  
        K(I,1)=K(I,1)+10    
        IF(MSTU(16).NE.2) THEN  
          K(I,4)=NSAV+1 
          K(I,5)=NSAV+1 
        ELSE    
          K(I,4)=NSAV+2 
          K(I,5)=N  
        ENDIF   
      ENDIF 
  310 CONTINUE  
      IF(N.LT.MSTU(4)-MSTU(32)-5) GOTO 140  
    
C...Check flavours and invariant masses in parton systems.  
  320 NP=0  
      KFN=0 
      KQS=0 
      DO 330 J=1,5  
  330 DPS(J)=0d0
      DO 360 I=MAX(1,IP),N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 360  
      KC=LUCOMP(K(I,2)) 
      IF(KC.EQ.0) GOTO 360  
      KQ=KCHG(KC,2)*ISIGN(1,K(I,2)) 
      IF(KQ.EQ.0) GOTO 360  
      NP=NP+1   
      IF(KQ.NE.2) THEN  
        KFN=KFN+1   
        KQS=KQS+KQ  
        MSTJ(93)=1  
        DPS(5)=DPS(5)+dble(ULMASS(K(I,2)))
      ENDIF 
      DO 340 J=1,4  
  340 DPS(J)=DPS(J)+dble(P(I,J))

clin-4/12/01:
c     np: # of partons, KFN: number of quarks and diquarks, 
c     KC=0 for color singlet system, -1 for quarks and anti-diquarks, 
c     1 for quarks and anti-diquarks, and 2 for gluons:
      IF(K(I,1).EQ.1) THEN  
clin-4/12/01     end of color singlet system.
        IF(NP.NE.1.AND.(KFN.EQ.1.OR.KFN.GE.3.OR.KQS.NE.0)) CALL 
     &  LUERRM(2,'(LUPREP:) unphysical flavour combination')    

clin-4/16/01: 'jet system' should be defined as np.ne.2:
c        IF(NP.NE.1.AND.DPS(4)**2-DPS(1)**2-DPS(2)**2-DPS(3)**2.LT.  
c     &  (0.9*PARJ(32)+DPS(5))**2) CALL LUERRM(3,    
c     &  '(LUPREP:) too small mass in jet system')   
        IF(NP.NE.2.AND.DPS(4)**2-DPS(1)**2-DPS(2)**2-DPS(3)**2.LT.  
     &  (0.9d0*dble(PARJ(32))+DPS(5))**2) then 
           CALL LUERRM(3,    
     &  '(LUPREP:) too small mass in jet system')   
           write (6,*) 'DPS(1-5),KI1-5=',DPS(1),DPS(2),DPS(3),DPS(4),
     1 DPS(5),'*',K(I,1),K(I,2),K(I,3),K(I,4),K(I,5)
        endif

        NP=0    
        KFN=0   
        KQS=0   
        DO 350 J=1,5    
  350   DPS(J)=0d0
      ENDIF 
  360 CONTINUE  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUSTRF(IP) 
C...Purpose: to handle the fragmentation of an arbitrary colour singlet 
C...jet system according to the Lund string fragmentation model.    
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      DIMENSION DPS(5),KFL(3),PMQ(3),PX(3),PY(3),GAM(3),IE(2),PR(2),    
     &IN(9),DHM(4),DHG(4),DP(5,5),IRANK(2),MJU(4),IJU(3),PJU(5,5),  
     &TJU(5),KFJH(2),NJS(2),KFJS(2),PJS(4,5)    
    
C...Function: four-product of two vectors.  
      FOUR(I,J)=P(I,4)*P(J,4)-P(I,1)*P(J,1)-P(I,2)*P(J,2)-P(I,3)*P(J,3) 
      DFOUR(I,J)=DP(I,4)*DP(J,4)-DP(I,1)*DP(J,1)-DP(I,2)*DP(J,2)-   
     &DP(I,3)*DP(J,3)   
    
C...Reset counters. Identify parton system. 
      MSTJ(91)=0    
      NSAV=N    
      NP=0  
      KQSUM=0   
      DO 100 J=1,5  
  100 DPS(J)=0d0 
      MJU(1)=0  
      MJU(2)=0  
      I=IP-1    
  110 I=I+1 
      IF(I.GT.MIN(N,MSTU(4)-MSTU(32))) THEN 
        CALL LUERRM(12,'(LUSTRF:) failed to reconstruct jet system')    
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      IF(K(I,1).NE.1.AND.K(I,1).NE.2.AND.K(I,1).NE.41) GOTO 110 
      KC=LUCOMP(K(I,2)) 
      IF(KC.EQ.0) GOTO 110  
      KQ=KCHG(KC,2)*ISIGN(1,K(I,2)) 
      IF(KQ.EQ.0) GOTO 110  
      IF(N+5*NP+11.GT.MSTU(4)-MSTU(32)-5) THEN  
        CALL LUERRM(11,'(LUSTRF:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 

cms.. pre-initialize to avoid compiler warning
      JR=0

C...Take copy of partons to be considered. Check flavour sum.   
      NP=NP+1   
      DO 120 J=1,5  
      K(N+NP,J)=K(I,J)  
      P(N+NP,J)=P(I,J)  
  120 DPS(J)=DPS(J)+dble(P(I,J))
      K(N+NP,3)=I   
      IF(P(N+NP,4)**2.LT.P(N+NP,1)**2+P(N+NP,2)**2+P(N+NP,3)**2) THEN   
        P(N+NP,4)=SQRT(P(N+NP,1)**2+P(N+NP,2)**2+P(N+NP,3)**2+  
     &  P(N+NP,5)**2)   
        DPS(4)=DPS(4)+dble(MAX(0.,P(N+NP,4)-P(I,4)))
      ENDIF 
      IF(KQ.NE.2) KQSUM=KQSUM+KQ    
      IF(K(I,1).EQ.41) THEN 
        KQSUM=KQSUM+2*KQ    
        IF(KQSUM.EQ.KQ) MJU(1)=N+NP 
        IF(KQSUM.NE.KQ) MJU(2)=N+NP 
      ENDIF 
      IF(K(I,1).EQ.2.OR.K(I,1).EQ.41) GOTO 110  
      IF(KQSUM.NE.0) THEN   
        CALL LUERRM(12,'(LUSTRF:) unphysical flavour combination')  
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 

C...Boost copied system to CM frame (for better numerical precision).   
      CALL LUDBRB(N+1,N+NP,0.,0.,-DPS(1)/DPS(4),-DPS(2)/DPS(4), 
     &-DPS(3)/DPS(4))   

C...Search for very nearby partons that may be recombined.  
      NTRYR=0   
      PARU12=PARU(12)   
      PARU13=PARU(13)   
      MJU(3)=MJU(1) 
      MJU(4)=MJU(2) 
      NR=NP 
  130 IF(NR.GE.3) THEN  
        PDRMIN=2.*PARU12    
        IR=0
        DO 140 I=N+1,N+NR   
        IF(I.EQ.N+NR.AND.IABS(K(N+1,2)).NE.21) GOTO 140 
        I1=I+1  
        IF(I.EQ.N+NR) I1=N+1    
        IF(K(I,1).EQ.41.OR.K(I1,1).EQ.41) GOTO 140  
        IF(MJU(1).NE.0.AND.I1.LT.MJU(1).AND.IABS(K(I1,2)).NE.21)    
     &  GOTO 140    
        IF(MJU(2).NE.0.AND.I.GT.MJU(2).AND.IABS(K(I,2)).NE.21) GOTO 140 
        PAP=SQRT((P(I,1)**2+P(I,2)**2+P(I,3)**2)*(P(I1,1)**2+   
     &  P(I1,2)**2+P(I1,3)**2)) 
        PVP=P(I,1)*P(I1,1)+P(I,2)*P(I1,2)+P(I,3)*P(I1,3)    
        PDR=4.*(PAP-PVP)**2/(PARU13**2*PAP+2.*(PAP-PVP))    
        IF(PDR.LT.PDRMIN) THEN  
          IR=I  
          PDRMIN=PDR    
        ENDIF   
  140   CONTINUE    
    
C...Recombine very nearby partons to avoid machine precision problems.  
        IF(PDRMIN.LT.PARU12.AND.IR.EQ.N+NR) THEN    
          DO 150 J=1,4  
  150     P(N+1,J)=P(N+1,J)+P(N+NR,J)   
          P(N+1,5)=SQRT(MAX(0.,P(N+1,4)**2-P(N+1,1)**2-P(N+1,2)**2- 
     &    P(N+1,3)**2)) 
          NR=NR-1   
          GOTO 130  
        ELSEIF(PDRMIN.LT.PARU12) THEN   
          DO 160 J=1,4  
  160     P(IR,J)=P(IR,J)+P(IR+1,J) 
          P(IR,5)=SQRT(MAX(0.,P(IR,4)**2-P(IR,1)**2-P(IR,2)**2- 
     &    P(IR,3)**2))  
          DO 170 I=IR+1,N+NR-1  
          K(I,2)=K(I+1,2)   
          DO 170 J=1,5  
  170     P(I,J)=P(I+1,J)   
          IF(IR.EQ.N+NR-1) K(IR,2)=K(N+NR,2)    
          NR=NR-1   
          IF(MJU(1).GT.IR) MJU(1)=MJU(1)-1  
          IF(MJU(2).GT.IR) MJU(2)=MJU(2)-1  
          GOTO 130  
        ENDIF   
      ENDIF 
      NTRYR=NTRYR+1 
    
C...Reset particle counter. Skip ahead if no junctions are present; 
C...this is usually the case!   
      NRS=MAX(5*NR+11,NP)   
      NTRY=0    
  180 NTRY=NTRY+1   
      IF(NTRY.GT.100.AND.NTRYR.LE.4) THEN   
        PARU12=4.*PARU12    
        PARU13=2.*PARU13    
        GOTO 130    
      ELSEIF(NTRY.GT.100) THEN  
        CALL LUERRM(14,'(LUSTRF:) caught in infinite loop') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      I=N+NRS   
      IF(MJU(1).EQ.0.AND.MJU(2).EQ.0) GOTO 500  
      DO 490 JT=1,2 
      NJS(JT)=0 
      IF(MJU(JT).EQ.0) GOTO 490 
      JS=3-2*JT 
    
C...Find and sum up momentum on three sides of junction. Check flavours.    
      DO 190 IU=1,3 
      IJU(IU)=0 
      DO 190 J=1,5  
  190 PJU(IU,J)=0.  
      IU=0  
      DO 200 I1=N+1+(JT-1)*(NR-1),N+NR+(JT-1)*(1-NR),JS 
      IF(K(I1,2).NE.21.AND.IU.LE.2) THEN    
        IU=IU+1 
        IJU(IU)=I1  
      ENDIF 
      DO 200 J=1,4  
  200 PJU(IU,J)=PJU(IU,J)+P(I1,J)   
      DO 210 IU=1,3 
  210 PJU(IU,5)=SQRT(PJU(IU,1)**2+PJU(IU,2)**2+PJU(IU,3)**2)    
      IF(K(IJU(3),2)/100.NE.10*K(IJU(1),2)+K(IJU(2),2).AND. 
     &K(IJU(3),2)/100.NE.10*K(IJU(2),2)+K(IJU(1),2)) THEN   
        CALL LUERRM(12,'(LUSTRF:) unphysical flavour combination')  
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
    
C...Calculate (approximate) boost to rest frame of junction.    
      T12=(PJU(1,1)*PJU(2,1)+PJU(1,2)*PJU(2,2)+PJU(1,3)*PJU(2,3))/  
     &(PJU(1,5)*PJU(2,5))   
      T13=(PJU(1,1)*PJU(3,1)+PJU(1,2)*PJU(3,2)+PJU(1,3)*PJU(3,3))/  
     &(PJU(1,5)*PJU(3,5))   
      T23=(PJU(2,1)*PJU(3,1)+PJU(2,2)*PJU(3,2)+PJU(2,3)*PJU(3,3))/  
     &(PJU(2,5)*PJU(3,5))   
      T11=SQRT((2./3.)*(1.-T12)*(1.-T13)/(1.-T23))  
      T22=SQRT((2./3.)*(1.-T12)*(1.-T23)/(1.-T13))  
      TSQ=SQRT((2.*T11*T22+T12-1.)*(1.+T12))    
      T1F=(TSQ-T22*(1.+T12))/(1.-T12**2)    
      T2F=(TSQ-T11*(1.+T12))/(1.-T12**2)    
      DO 220 J=1,3  
  220 TJU(J)=-(T1F*PJU(1,J)/PJU(1,5)+T2F*PJU(2,J)/PJU(2,5)) 
      TJU(4)=SQRT(1.+TJU(1)**2+TJU(2)**2+TJU(3)**2) 
      DO 230 IU=1,3 
  230 PJU(IU,5)=TJU(4)*PJU(IU,4)-TJU(1)*PJU(IU,1)-TJU(2)*PJU(IU,2)- 
     &TJU(3)*PJU(IU,3)  
    
C...Put junction at rest if motion could give inconsistencies.  
      IF(PJU(1,5)+PJU(2,5).GT.PJU(1,4)+PJU(2,4)) THEN   
        DO 240 J=1,3    
  240   TJU(J)=0.   
        TJU(4)=1.   
        PJU(1,5)=PJU(1,4)   
        PJU(2,5)=PJU(2,4)   
        PJU(3,5)=PJU(3,4)   
      ENDIF 
    
C...Start preparing for fragmentation of two strings from junction. 
      ISTA=I    
      DO 470 IU=1,2 
      NS=IJU(IU+1)-IJU(IU)  
    
C...Junction strings: find longitudinal string directions.  
      DO 260 IS=1,NS    
      IS1=IJU(IU)+IS-1  
      IS2=IJU(IU)+IS    
      DO 250 J=1,5  
      DP(1,J)=dble(0.5*P(IS1,J))
      IF(IS.EQ.1) DP(1,J)=dble(P(IS1,J))
      DP(2,J)=dble(0.5*P(IS2,J))
  250 IF(IS.EQ.NS) DP(2,J)=-dble(PJU(IU,J))
      IF(IS.EQ.NS) DP(2,4)=dble(
     1     SQRT(PJU(IU,1)**2+PJU(IU,2)**2+PJU(IU,3)**2))
      IF(IS.EQ.NS) DP(2,5)=0d0   
      DP(3,5)=DFOUR(1,1)    
      DP(4,5)=DFOUR(2,2)    
      DHKC=DFOUR(1,2)   
      IF(DP(3,5)+2d0*DHKC+DP(4,5).LE.0d0) THEN    
        DP(1,4)=SQRT(DP(1,1)**2+DP(1,2)**2+DP(1,3)**2)  
        DP(2,4)=SQRT(DP(2,1)**2+DP(2,2)**2+DP(2,3)**2)  
        DP(3,5)=0D0 
        DP(4,5)=0D0 
        DHKC=DFOUR(1,2) 
      ENDIF 
      DHKS=SQRT(DHKC**2-DP(3,5)*DP(4,5))    
      DHK1=0.5d0*((DP(4,5)+DHKC)/DHKS-1d0) 
      DHK2=0.5d0*((DP(3,5)+DHKC)/DHKS-1d0) 
      IN1=N+NR+4*IS-3   
      P(IN1,5)=sngl(SQRT(DP(3,5)+2d0*DHKC+DP(4,5)))
      DO 260 J=1,4  
      P(IN1,J)=sngl((1d0+DHK1)*DP(1,J)-DHK2*DP(2,J))
  260 P(IN1+1,J)=sngl((1d0+DHK2)*DP(2,J)-DHK1*DP(1,J))
    
C...Junction strings: initialize flavour, momentum and starting pos.    
      ISAV=I    
  270 NTRY=NTRY+1   
      IF(NTRY.GT.100.AND.NTRYR.LE.4) THEN   
        PARU12=4.*PARU12    
        PARU13=2.*PARU13    
        GOTO 130    
      ELSEIF(NTRY.GT.100) THEN  
        CALL LUERRM(14,'(LUSTRF:) caught in infinite loop') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      I=ISAV    
      IRANKJ=0  
      IE(1)=K(N+1+(JT/2)*(NP-1),3)  
      IN(4)=N+NR+1  
      IN(5)=IN(4)+1 
      IN(6)=N+NR+4*NS+1 
      DO 280 JQ=1,2 
      DO 280 IN1=N+NR+2+JQ,N+NR+4*NS-2+JQ,4 
      P(IN1,1)=2-JQ 
      P(IN1,2)=JQ-1 
  280 P(IN1,3)=1.   
      KFL(1)=K(IJU(IU),2)   
      PX(1)=0.  
      PY(1)=0.  
      GAM(1)=0. 
      DO 290 J=1,5  
  290 PJU(IU+3,J)=0.    
    
C...Junction strings: find initial transverse directions.   
      DO 300 J=1,4  
      DP(1,J)=dble(P(IN(4),J))
      DP(2,J)=dble(P(IN(4)+1,J))
      DP(3,J)=0d0    
  300 DP(4,J)=0d0    
      DP(1,4)=SQRT(DP(1,1)**2+DP(1,2)**2+DP(1,3)**2)    
      DP(2,4)=SQRT(DP(2,1)**2+DP(2,2)**2+DP(2,3)**2)    
      DP(5,1)=DP(1,1)/DP(1,4)-DP(2,1)/DP(2,4)   
      DP(5,2)=DP(1,2)/DP(1,4)-DP(2,2)/DP(2,4)   
      DP(5,3)=DP(1,3)/DP(1,4)-DP(2,3)/DP(2,4)   
      IF(DP(5,1)**2.LE.DP(5,2)**2+DP(5,3)**2) DP(3,1)=1d0    
      IF(DP(5,1)**2.GT.DP(5,2)**2+DP(5,3)**2) DP(3,3)=1d0    
      IF(DP(5,2)**2.LE.DP(5,1)**2+DP(5,3)**2) DP(4,2)=1d0    
      IF(DP(5,2)**2.GT.DP(5,1)**2+DP(5,3)**2) DP(4,3)=1d0    
      DHC12=DFOUR(1,2)  
      DHCX1=DFOUR(3,1)/DHC12    
      DHCX2=DFOUR(3,2)/DHC12    
      DHCXX=1D0/SQRT(1D0+2D0*DHCX1*DHCX2*DHC12) 
      DHCY1=DFOUR(4,1)/DHC12    
      DHCY2=DFOUR(4,2)/DHC12    
      DHCYX=DHCXX*(DHCX1*DHCY2+DHCX2*DHCY1)*DHC12   
      DHCYY=1D0/SQRT(1D0+2D0*DHCY1*DHCY2*DHC12-DHCYX**2)    
      DO 310 J=1,4  
      DP(3,J)=DHCXX*(DP(3,J)-DHCX2*DP(1,J)-DHCX1*DP(2,J))   
      P(IN(6),J)=sngl(DP(3,J))
  310 P(IN(6)+1,J)=sngl(DHCYY*(DP(4,J)-DHCY2*DP(1,J)-DHCY1*DP(2,J)-  
     &DHCYX*DP(3,J)))    
    
C...Junction strings: produce new particle, origin. 
  320 I=I+1 
      IF(2*I-NSAV.GE.MSTU(4)-MSTU(32)-5) THEN   
        CALL LUERRM(11,'(LUSTRF:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      IRANKJ=IRANKJ+1   
      K(I,1)=1  
      K(I,3)=IE(1)  
      K(I,4)=0  
      K(I,5)=0  
    
C...Junction strings: generate flavour, hadron, pT, z and Gamma.    
  330 CALL LUKFDI(KFL(1),0,KFL(3),K(I,2))   
      IF(K(I,2).EQ.0) GOTO 270  
      IF(MSTJ(12).GE.3.AND.IRANKJ.EQ.1.AND.IABS(KFL(1)).LE.10.AND.  
     &IABS(KFL(3)).GT.10) THEN  
        IF(RLU(0).GT.PARJ(19)) GOTO 330 
      ENDIF 
      P(I,5)=ULMASS(K(I,2)) 
      CALL LUPTDI(KFL(1),PX(3),PY(3))   
      PR(1)=P(I,5)**2+(PX(1)+PX(3))**2+(PY(1)+PY(3))**2 
      CALL LUZDIS(KFL(1),KFL(3),PR(1),Z)    
      GAM(3)=(1.-Z)*(GAM(1)+PR(1)/Z)    
      DO 340 J=1,3  
  340 IN(J)=IN(3+J) 

C...Junction strings: stepping within or from 'low' string region easy. 
      IF(IN(1)+1.EQ.IN(2).AND.Z*P(IN(1)+2,3)*P(IN(2)+2,3)*  
     &P(IN(1),5)**2.GE.PR(1)) THEN  
        P(IN(1)+2,4)=Z*P(IN(1)+2,3) 
        P(IN(2)+2,4)=PR(1)/(P(IN(1)+2,4)*P(IN(1),5)**2) 
        DO 350 J=1,4    
  350   P(I,J)=(PX(1)+PX(3))*P(IN(3),J)+(PY(1)+PY(3))*P(IN(3)+1,J)  
        GOTO 420    
      ELSEIF(IN(1)+1.EQ.IN(2)) THEN 
        P(IN(2)+2,4)=P(IN(2)+2,3)   
        P(IN(2)+2,1)=1. 
        IN(2)=IN(2)+4   
        IF(IN(2).GT.N+NR+4*NS) GOTO 270 
        IF(FOUR(IN(1),IN(2)).LE.1E-2) THEN  
          P(IN(1)+2,4)=P(IN(1)+2,3) 
          P(IN(1)+2,1)=0.   
          IN(1)=IN(1)+4 
        ENDIF   
      ENDIF 
    
C...Junction strings: find new transverse directions.   
  360 IF(IN(1).GT.N+NR+4*NS.OR.IN(2).GT.N+NR+4*NS.OR.   
     &IN(1).GT.IN(2)) GOTO 270  
      IF(IN(1).NE.IN(4).OR.IN(2).NE.IN(5)) THEN 
        DO 370 J=1,4    
        DP(1,J)=dble(P(IN(1),J))
        DP(2,J)=dble(P(IN(2),J))
        DP(3,J)=0d0  
  370   DP(4,J)=0d0  
        DP(1,4)=SQRT(DP(1,1)**2+DP(1,2)**2+DP(1,3)**2)  
        DP(2,4)=SQRT(DP(2,1)**2+DP(2,2)**2+DP(2,3)**2)  
        DHC12=DFOUR(1,2)    
clin-5/2012:
c        IF(DHC12.LE.1E-2) THEN  
        IF(DHC12.LE.1D-2) THEN  
          P(IN(1)+2,4)=P(IN(1)+2,3) 
          P(IN(1)+2,1)=0.   
          IN(1)=IN(1)+4 
          GOTO 360  
        ENDIF   
        IN(3)=N+NR+4*NS+5   
        DP(5,1)=DP(1,1)/DP(1,4)-DP(2,1)/DP(2,4) 
        DP(5,2)=DP(1,2)/DP(1,4)-DP(2,2)/DP(2,4) 
        DP(5,3)=DP(1,3)/DP(1,4)-DP(2,3)/DP(2,4) 
        IF(DP(5,1)**2.LE.DP(5,2)**2+DP(5,3)**2) DP(3,1)=1d0  
        IF(DP(5,1)**2.GT.DP(5,2)**2+DP(5,3)**2) DP(3,3)=1d0  
        IF(DP(5,2)**2.LE.DP(5,1)**2+DP(5,3)**2) DP(4,2)=1d0  
        IF(DP(5,2)**2.GT.DP(5,1)**2+DP(5,3)**2) DP(4,3)=1d0  
        DHCX1=DFOUR(3,1)/DHC12  
        DHCX2=DFOUR(3,2)/DHC12  
        DHCXX=1D0/SQRT(1D0+2D0*DHCX1*DHCX2*DHC12)   
        DHCY1=DFOUR(4,1)/DHC12  
        DHCY2=DFOUR(4,2)/DHC12  
        DHCYX=DHCXX*(DHCX1*DHCY2+DHCX2*DHCY1)*DHC12 
        DHCYY=1D0/SQRT(1D0+2D0*DHCY1*DHCY2*DHC12-DHCYX**2)  
        DO 380 J=1,4    
        DP(3,J)=DHCXX*(DP(3,J)-DHCX2*DP(1,J)-DHCX1*DP(2,J)) 
        P(IN(3),J)=sngl(DP(3,J))
  380   P(IN(3)+1,J)=sngl(DHCYY*(DP(4,J)-DHCY2*DP(1,J)-DHCY1*DP(2,J)-    
     &  DHCYX*DP(3,J)))  
C...Express pT with respect to new axes, if sensible.   
        PXP=-(PX(3)*FOUR(IN(6),IN(3))+PY(3)*FOUR(IN(6)+1,IN(3)))    
        PYP=-(PX(3)*FOUR(IN(6),IN(3)+1)+PY(3)*FOUR(IN(6)+1,IN(3)+1))    
        IF(ABS(PXP**2+PYP**2-PX(3)**2-PY(3)**2).LT.0.01) THEN   
          PX(3)=PXP 
          PY(3)=PYP 
        ENDIF   
      ENDIF 
    
C...Junction strings: sum up known four-momentum, coefficients for m2.  
      DO 400 J=1,4  
      DHG(J)=0d0 
      P(I,J)=PX(1)*P(IN(6),J)+PY(1)*P(IN(6)+1,J)+PX(3)*P(IN(3),J)+  
     &PY(3)*P(IN(3)+1,J)    
      DO 390 IN1=IN(4),IN(1)-4,4    
  390 P(I,J)=P(I,J)+P(IN1+2,3)*P(IN1,J) 
      DO 400 IN2=IN(5),IN(2)-4,4    
  400 P(I,J)=P(I,J)+P(IN2+2,3)*P(IN2,J) 
      DHM(1)=dble(FOUR(I,I))
      DHM(2)=dble(2.*FOUR(I,IN(1)))   
      DHM(3)=dble(2.*FOUR(I,IN(2)))  
      DHM(4)=dble(2.*FOUR(IN(1),IN(2))) 
    
C...Junction strings: find coefficients for Gamma expression.   
      DO 410 IN2=IN(1)+1,IN(2),4    
      DO 410 IN1=IN(1),IN2-1,4  
      DHC=dble(2.*FOUR(IN1,IN2))
      DHG(1)=DHG(1)+dble(P(IN1+2,1)*P(IN2+2,1))*DHC   
      IF(IN1.EQ.IN(1)) DHG(2)=DHG(2)-dble(P(IN2+2,1))*DHC 
      IF(IN2.EQ.IN(2)) DHG(3)=DHG(3)+dble(P(IN1+2,1))*DHC 
  410 IF(IN1.EQ.IN(1).AND.IN2.EQ.IN(2)) DHG(4)=DHG(4)-DHC   
    
C...Junction strings: solve (m2, Gamma) equation system for energies.   
      DHS1=DHM(3)*DHG(4)-DHM(4)*DHG(3)  
clin-5/2012:
c      IF(ABS(DHS1).LT.1E-4) GOTO 270    
      IF(DABS(DHS1).LT.1D-4) GOTO 270    
      DHS2=DHM(4)*(dble(GAM(3))-DHG(1))-DHM(2)*DHG(3)-DHG(4)* 
     &(dble(P(I,5))**2-DHM(1))+DHG(2)*DHM(3)  
      DHS3=DHM(2)*(dble(GAM(3))-DHG(1))
     1     -DHG(2)*(dble(P(I,5))**2-DHM(1)) 
      P(IN(2)+2,4)=0.5*sngl(SQRT(MAX(0D0,DHS2**2-4d0*DHS1*DHS3))
     &     /ABS(DHS1)-DHS2/DHS1)
      IF(DHM(2)+DHM(4)*dble(P(IN(2)+2,4)).LE.0d0) GOTO 270 
      P(IN(1)+2,4)=(P(I,5)**2-sngl(DHM(1))-sngl(DHM(3))*P(IN(2)+2,4))/  
     &(sngl(DHM(2))+sngl(DHM(4))*P(IN(2)+2,4))  

C...Junction strings: step to new region if necessary.  
      IF(P(IN(2)+2,4).GT.P(IN(2)+2,3)) THEN 
        P(IN(2)+2,4)=P(IN(2)+2,3)   
        P(IN(2)+2,1)=1. 
        IN(2)=IN(2)+4   
        IF(IN(2).GT.N+NR+4*NS) GOTO 270 
        IF(FOUR(IN(1),IN(2)).LE.1E-2) THEN  
          P(IN(1)+2,4)=P(IN(1)+2,3) 
          P(IN(1)+2,1)=0.   
          IN(1)=IN(1)+4 
        ENDIF   
        GOTO 360    
      ELSEIF(P(IN(1)+2,4).GT.P(IN(1)+2,3)) THEN 
        P(IN(1)+2,4)=P(IN(1)+2,3)   
        P(IN(1)+2,1)=0. 
        IN(1)=IN(1)+JS  
        GOTO 710    
      ENDIF 
    
C...Junction strings: particle four-momentum, remainder, loop back. 
  420 DO 430 J=1,4  
      P(I,J)=P(I,J)+P(IN(1)+2,4)*P(IN(1),J)+P(IN(2)+2,4)*P(IN(2),J) 
  430 PJU(IU+3,J)=PJU(IU+3,J)+P(I,J)    
      IF(P(I,4).LE.0.) GOTO 270 
      PJU(IU+3,5)=TJU(4)*PJU(IU+3,4)-TJU(1)*PJU(IU+3,1)-    
     &TJU(2)*PJU(IU+3,2)-TJU(3)*PJU(IU+3,3) 
      IF(PJU(IU+3,5).LT.PJU(IU,5)) THEN 
        KFL(1)=-KFL(3)  
        PX(1)=-PX(3)    
        PY(1)=-PY(3)    
        GAM(1)=GAM(3)   
        IF(IN(3).NE.IN(6)) THEN 
          DO 440 J=1,4  
          P(IN(6),J)=P(IN(3),J) 
  440     P(IN(6)+1,J)=P(IN(3)+1,J) 
        ENDIF   
        DO 450 JQ=1,2   
        IN(3+JQ)=IN(JQ) 
        P(IN(JQ)+2,3)=P(IN(JQ)+2,3)-P(IN(JQ)+2,4)   
  450   P(IN(JQ)+2,1)=P(IN(JQ)+2,1)-(3-2*JQ)*P(IN(JQ)+2,4)  
        GOTO 320    
      ENDIF 
    
C...Junction strings: save quantities left after each string.   
      IF(IABS(KFL(1)).GT.10) GOTO 270   
      I=I-1 
      KFJH(IU)=KFL(1)   
      DO 460 J=1,4  
  460 PJU(IU+3,J)=PJU(IU+3,J)-P(I+1,J)  
  470 CONTINUE  
    
C...Junction strings: put together to new effective string endpoint.    
      NJS(JT)=I-ISTA    
      KFJS(JT)=K(K(MJU(JT+2),3),2)  
      KFLS=2*INT(RLU(0)+3.*PARJ(4)/(1.+3.*PARJ(4)))+1   
      IF(KFJH(1).EQ.KFJH(2)) KFLS=3 
      IF(ISTA.NE.I) KFJS(JT)=ISIGN(1000*MAX(IABS(KFJH(1)),  
     &IABS(KFJH(2)))+100*MIN(IABS(KFJH(1)),IABS(KFJH(2)))+  
     &KFLS,KFJH(1)) 
      DO 480 J=1,4  
      PJS(JT,J)=PJU(1,J)+PJU(2,J)+P(MJU(JT),J)  
  480 PJS(JT+2,J)=PJU(4,J)+PJU(5,J) 
      PJS(JT,5)=SQRT(MAX(0.,PJS(JT,4)**2-PJS(JT,1)**2-PJS(JT,2)**2- 
     &PJS(JT,3)**2))    
  490 CONTINUE  
    
C...Open versus closed strings. Choose breakup region for latter.   
  500 IF(MJU(1).NE.0.AND.MJU(2).NE.0) THEN  
        NS=MJU(2)-MJU(1)    
        NB=MJU(1)-N 
      ELSEIF(MJU(1).NE.0) THEN  
        NS=N+NR-MJU(1)  
        NB=MJU(1)-N 
      ELSEIF(MJU(2).NE.0) THEN  
        NS=MJU(2)-N 
        NB=1    
      ELSEIF(IABS(K(N+1,2)).NE.21) THEN 
        NS=NR-1 
        NB=1    
      ELSE  
        NS=NR+1 
        W2SUM=0.    
        DO 510 IS=1,NR  
        P(N+NR+IS,1)=0.5*FOUR(N+IS,N+IS+1-NR*(IS/NR))   
  510   W2SUM=W2SUM+P(N+NR+IS,1)    
        W2RAN=RLU(0)*W2SUM  
        NB=0    
  520   NB=NB+1 
        W2SUM=W2SUM-P(N+NR+NB,1)    
        IF(W2SUM.GT.W2RAN.AND.NB.LT.NR) GOTO 520    
      ENDIF 
    
C...Find longitudinal string directions (i.e. lightlike four-vectors).  
      DO 540 IS=1,NS    
      IS1=N+IS+NB-1-NR*((IS+NB-2)/NR)   
      IS2=N+IS+NB-NR*((IS+NB-1)/NR) 
      DO 530 J=1,5  
      DP(1,J)=dble(P(IS1,J))
      IF(IABS(K(IS1,2)).EQ.21) DP(1,J)=0.5d0*DP(1,J)  
      IF(IS1.EQ.MJU(1)) DP(1,J)=dble(PJS(1,J)-PJS(3,J))
      DP(2,J)=dble(P(IS2,J))
      IF(IABS(K(IS2,2)).EQ.21) DP(2,J)=0.5d0*DP(2,J)  
  530 IF(IS2.EQ.MJU(2)) DP(2,J)=dble(PJS(2,J)-PJS(4,J))
      DP(3,5)=DFOUR(1,1)    
      DP(4,5)=DFOUR(2,2)    
      DHKC=DFOUR(1,2)   
      IF(DP(3,5)+2.d0*DHKC+DP(4,5).LE.0.d0) THEN    
        DP(3,5)=DP(1,5)**2  
        DP(4,5)=DP(2,5)**2  
        DP(1,4)=SQRT(DP(1,1)**2+DP(1,2)**2+DP(1,3)**2+DP(1,5)**2)   
        DP(2,4)=SQRT(DP(2,1)**2+DP(2,2)**2+DP(2,3)**2+DP(2,5)**2)   
        DHKC=DFOUR(1,2) 
      ENDIF 
      DHKS=SQRT(DHKC**2-DP(3,5)*DP(4,5))    
      DHK1=0.5d0*((DP(4,5)+DHKC)/DHKS-1.d0) 
      DHK2=0.5d0*((DP(3,5)+DHKC)/DHKS-1.d0) 
      IN1=N+NR+4*IS-3   
      P(IN1,5)=SQRT(sngl(DP(3,5)+2.d0*DHKC+DP(4,5)))
      DO 540 J=1,4  
      P(IN1,J)=sngl((1.d0+DHK1)*DP(1,J)-DHK2*DP(2,J))
  540 P(IN1+1,J)=sngl((1.d0+DHK2)*DP(2,J)-DHK1*DP(1,J))
    
C...Begin initialization: sum up energy, set starting position. 
      ISAV=I    
  550 NTRY=NTRY+1   
      IF(NTRY.GT.100.AND.NTRYR.LE.4) THEN   
        PARU12=4.*PARU12    
        PARU13=2.*PARU13    
        GOTO 130    
      ELSEIF(NTRY.GT.100) THEN  
        CALL LUERRM(14,'(LUSTRF:) caught in infinite loop') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      I=ISAV    
      DO 560 J=1,4  
      P(N+NRS,J)=0. 
      DO 560 IS=1,NR    
  560 P(N+NRS,J)=P(N+NRS,J)+P(N+IS,J)   
      DO 570 JT=1,2 
      IRANK(JT)=0   
      IF(MJU(JT).NE.0) IRANK(JT)=NJS(JT)    
      IF(NS.GT.NR) IRANK(JT)=1  
      IE(JT)=K(N+1+(JT/2)*(NP-1),3) 
      IN(3*JT+1)=N+NR+1+4*(JT/2)*(NS-1) 
      IN(3*JT+2)=IN(3*JT+1)+1   
      IN(3*JT+3)=N+NR+4*NS+2*JT-1   
      DO 570 IN1=N+NR+2+JT,N+NR+4*NS-2+JT,4 
      P(IN1,1)=2-JT 
      P(IN1,2)=JT-1 
  570 P(IN1,3)=1.   
    
C...Initialize flavour and pT variables for open string.    
      IF(NS.LT.NR) THEN 
        PX(1)=0.    
        PY(1)=0.    
        IF(NS.EQ.1.AND.MJU(1)+MJU(2).EQ.0) CALL LUPTDI(0,PX(1),PY(1))   
        PX(2)=-PX(1)    
        PY(2)=-PY(1)    
        DO 580 JT=1,2   
        KFL(JT)=K(IE(JT),2) 
        IF(MJU(JT).NE.0) KFL(JT)=KFJS(JT)   
        MSTJ(93)=1  
        PMQ(JT)=ULMASS(KFL(JT)) 
  580   GAM(JT)=0.  
    
C...Closed string: random initial breakup flavour, pT and vertex.   
      ELSE  
        KFL(3)=INT(1.+(2.+PARJ(2))*RLU(0))*(-1)**INT(RLU(0)+0.5)    
        CALL LUKFDI(KFL(3),0,KFL(1),KDUMP)  
        KFL(2)=-KFL(1)  
        IF(IABS(KFL(1)).GT.10.AND.RLU(0).GT.0.5) THEN   
          KFL(2)=-(KFL(1)+ISIGN(10000,KFL(1)))  
        ELSEIF(IABS(KFL(1)).GT.10) THEN 
          KFL(1)=-(KFL(2)+ISIGN(10000,KFL(2)))  
        ENDIF   
        CALL LUPTDI(KFL(1),PX(1),PY(1)) 
        PX(2)=-PX(1)    
        PY(2)=-PY(1)    
        PR3=MIN(25.,0.1*P(N+NR+1,5)**2) 
  590   CALL LUZDIS(KFL(1),KFL(2),PR3,Z)    
        ZR=PR3/(Z*P(N+NR+1,5)**2)   
        IF(ZR.GE.1.) GOTO 590   

        DO 600 JT=1,2   
        MSTJ(93)=1  
        PMQ(JT)=ULMASS(KFL(JT)) 
        GAM(JT)=PR3*(1.-Z)/Z    
        IN1=N+NR+3+4*(JT/2)*(NS-1)  
        P(IN1,JT)=1.-Z  
        P(IN1,3-JT)=JT-1    
        P(IN1,3)=(2-JT)*(1.-Z)+(JT-1)*Z 
        P(IN1+1,JT)=ZR  
        P(IN1+1,3-JT)=2-JT  
  600   P(IN1+1,3)=(2-JT)*(1.-ZR)+(JT-1)*ZR 
      ENDIF 
    
C...Find initial transverse directions (i.e. spacelike four-vectors).   
      DO 640 JT=1,2 
      IF(JT.EQ.1.OR.NS.EQ.NR-1) THEN    
        IN1=IN(3*JT+1)  
        IN3=IN(3*JT+3)  
        DO 610 J=1,4    
        DP(1,J)=dble(P(IN1,J))
        DP(2,J)=dble(P(IN1+1,J))
        DP(3,J)=0.d0
  610   DP(4,J)=0.d0
        DP(1,4)=DSQRT(DP(1,1)**2+DP(1,2)**2+DP(1,3)**2)  
        DP(2,4)=DSQRT(DP(2,1)**2+DP(2,2)**2+DP(2,3)**2)  
        DP(5,1)=DP(1,1)/DP(1,4)-DP(2,1)/DP(2,4) 
        DP(5,2)=DP(1,2)/DP(1,4)-DP(2,2)/DP(2,4) 
        DP(5,3)=DP(1,3)/DP(1,4)-DP(2,3)/DP(2,4) 
        IF(DP(5,1)**2.LE.DP(5,2)**2+DP(5,3)**2) DP(3,1)=1.d0
        IF(DP(5,1)**2.GT.DP(5,2)**2+DP(5,3)**2) DP(3,3)=1.d0
        IF(DP(5,2)**2.LE.DP(5,1)**2+DP(5,3)**2) DP(4,2)=1.d0
        IF(DP(5,2)**2.GT.DP(5,1)**2+DP(5,3)**2) DP(4,3)=1.d0
        DHC12=DFOUR(1,2)    
        DHCX1=DFOUR(3,1)/DHC12  
        DHCX2=DFOUR(3,2)/DHC12  
        DHCXX=1D0/SQRT(1D0+2D0*DHCX1*DHCX2*DHC12)   
        DHCY1=DFOUR(4,1)/DHC12  
        DHCY2=DFOUR(4,2)/DHC12  
        DHCYX=DHCXX*(DHCX1*DHCY2+DHCX2*DHCY1)*DHC12 
        DHCYY=1D0/SQRT(1D0+2D0*DHCY1*DHCY2*DHC12-DHCYX**2)  
        DO 620 J=1,4    
        DP(3,J)=DHCXX*(DP(3,J)-DHCX2*DP(1,J)-DHCX1*DP(2,J)) 
        P(IN3,J)=sngl(DP(3,J))
  620   P(IN3+1,J)=sngl(DHCYY*(DP(4,J)-DHCY2*DP(1,J)-DHCY1*DP(2,J)-  
     &  DHCYX*DP(3,J)))
      ELSE  
        DO 630 J=1,4    
        P(IN3+2,J)=P(IN3,J) 
  630   P(IN3+3,J)=P(IN3+1,J)   
      ENDIF 
  640 CONTINUE  
    
C...Remove energy used up in junction string fragmentation. 
      IF(MJU(1)+MJU(2).GT.0) THEN   
        DO 660 JT=1,2   
        IF(NJS(JT).EQ.0) GOTO 660   
        DO 650 J=1,4    
  650   P(N+NRS,J)=P(N+NRS,J)-PJS(JT+2,J)   
  660   CONTINUE    
      ENDIF 
    
C...Produce new particle: side, origin. 
  670 I=I+1 
      IF(2*I-NSAV.GE.MSTU(4)-MSTU(32)-5) THEN   
        CALL LUERRM(11,'(LUSTRF:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      JT=int(1.5+RLU(0))
      IF(IABS(KFL(3-JT)).GT.10) JT=3-JT 
      JR=3-JT   
      JS=3-2*JT 
      IRANK(JT)=IRANK(JT)+1 
      K(I,1)=1  
      K(I,3)=IE(JT) 
      K(I,4)=0  
      K(I,5)=0  
    
C...Generate flavour, hadron and pT.    
  680 CALL LUKFDI(KFL(JT),0,KFL(3),K(I,2))  
      IF(K(I,2).EQ.0) GOTO 550  
      IF(MSTJ(12).GE.3.AND.IRANK(JT).EQ.1.AND.IABS(KFL(JT)).LE.10.AND.  
     &IABS(KFL(3)).GT.10) THEN  
        IF(RLU(0).GT.PARJ(19)) GOTO 680 
      ENDIF 
      P(I,5)=ULMASS(K(I,2)) 
      CALL LUPTDI(KFL(JT),PX(3),PY(3))  
      PR(JT)=P(I,5)**2+(PX(JT)+PX(3))**2+(PY(JT)+PY(3))**2  
    
C...Final hadrons for small invariant mass. 
      MSTJ(93)=1    
      PMQ(3)=ULMASS(KFL(3)) 
      WMIN=PARJ(32+MSTJ(11))+PMQ(1)+PMQ(2)+PARJ(36)*PMQ(3)  
      IF(IABS(KFL(JT)).GT.10.AND.IABS(KFL(3)).GT.10) WMIN=  
     &WMIN-0.5*PARJ(36)*PMQ(3)  
      WREM2=FOUR(N+NRS,N+NRS)   
      IF(WREM2.LT.0.10) GOTO 550    
      IF(WREM2.LT.MAX(WMIN*(1.+(2.*RLU(0)-1.)*PARJ(37)),    
     &PARJ(32)+PMQ(1)+PMQ(2))**2) GOTO 810  
    
C...Choose z, which gives Gamma. Shift z for heavy flavours.    
      CALL LUZDIS(KFL(JT),KFL(3),PR(JT),Z)  

      KFL1A=IABS(KFL(1))    
      KFL2A=IABS(KFL(2))    
      IF(MAX(MOD(KFL1A,10),MOD(KFL1A/1000,10),MOD(KFL2A,10),    
     &MOD(KFL2A/1000,10)).GE.4) THEN    
        PR(JR)=(PMQ(JR)+PMQ(3))**2+(PX(JR)-PX(3))**2+(PY(JR)-PY(3))**2  
        PW12=SQRT(MAX(0.,(WREM2-PR(1)-PR(2))**2-4.*PR(1)*PR(2)))    
        Z=(WREM2+PR(JT)-PR(JR)+PW12*(2.*Z-1.))/(2.*WREM2)   
        PR(JR)=(PMQ(JR)+PARJ(32+MSTJ(11)))**2+(PX(JR)-PX(3))**2+    
     &  (PY(JR)-PY(3))**2   
        IF((1.-Z)*(WREM2-PR(JT)/Z).LT.PR(JR)) GOTO 810  
      ENDIF 
      GAM(3)=(1.-Z)*(GAM(JT)+PR(JT)/Z)  
      DO 690 J=1,3  
  690 IN(J)=IN(3*JT+J)  
    
C...Stepping within or from 'low' string region easy.   
      IF(IN(1)+1.EQ.IN(2).AND.Z*P(IN(1)+2,3)*P(IN(2)+2,3)*  
     &P(IN(1),5)**2.GE.PR(JT)) THEN 
        P(IN(JT)+2,4)=Z*P(IN(JT)+2,3)   
        P(IN(JR)+2,4)=PR(JT)/(P(IN(JT)+2,4)*P(IN(1),5)**2)  
        DO 700 J=1,4    
  700   P(I,J)=(PX(JT)+PX(3))*P(IN(3),J)+(PY(JT)+PY(3))*P(IN(3)+1,J)    
        GOTO 770    
      ELSEIF(IN(1)+1.EQ.IN(2)) THEN 
        P(IN(JR)+2,4)=P(IN(JR)+2,3) 
        P(IN(JR)+2,JT)=1.   
        IN(JR)=IN(JR)+4*JS  
        IF(JS*IN(JR).GT.JS*IN(4*JR)) GOTO 550   
        IF(FOUR(IN(1),IN(2)).LE.1E-2) THEN  
          P(IN(JT)+2,4)=P(IN(JT)+2,3)   
          P(IN(JT)+2,JT)=0. 
          IN(JT)=IN(JT)+4*JS    
        ENDIF   
      ENDIF 
    
C...Find new transverse directions (i.e. spacelike string vectors). 
  710 IF(JS*IN(1).GT.JS*IN(3*JR+1).OR.JS*IN(2).GT.JS*IN(3*JR+2).OR. 
     &IN(1).GT.IN(2)) GOTO 550  
      IF(IN(1).NE.IN(3*JT+1).OR.IN(2).NE.IN(3*JT+2)) THEN   
        DO 720 J=1,4    
        DP(1,J)=dble(P(IN(1),J))
        DP(2,J)=dble(P(IN(2),J))
        DP(3,J)=0.d0
  720   DP(4,J)=0.d0
        DP(1,4)=DSQRT(DP(1,1)**2+DP(1,2)**2+DP(1,3)**2)  
        DP(2,4)=DSQRT(DP(2,1)**2+DP(2,2)**2+DP(2,3)**2)  
        DHC12=DFOUR(1,2)    
clin-5/2012:
c        IF(DHC12.LE.1E-2) THEN  
        IF(DHC12.LE.1D-2) THEN  
          P(IN(JT)+2,4)=P(IN(JT)+2,3)   
          P(IN(JT)+2,JT)=0. 
          IN(JT)=IN(JT)+4*JS    
          GOTO 710  
        ENDIF   
        IN(3)=N+NR+4*NS+5   
        DP(5,1)=DP(1,1)/DP(1,4)-DP(2,1)/DP(2,4) 
        DP(5,2)=DP(1,2)/DP(1,4)-DP(2,2)/DP(2,4) 
        DP(5,3)=DP(1,3)/DP(1,4)-DP(2,3)/DP(2,4) 
        IF(DP(5,1)**2.LE.DP(5,2)**2+DP(5,3)**2) DP(3,1)=1.d0
        IF(DP(5,1)**2.GT.DP(5,2)**2+DP(5,3)**2) DP(3,3)=1.d0
        IF(DP(5,2)**2.LE.DP(5,1)**2+DP(5,3)**2) DP(4,2)=1.d0
        IF(DP(5,2)**2.GT.DP(5,1)**2+DP(5,3)**2) DP(4,3)=1.d0
        DHCX1=DFOUR(3,1)/DHC12  
        DHCX2=DFOUR(3,2)/DHC12  
        DHCXX=1D0/SQRT(1D0+2D0*DHCX1*DHCX2*DHC12)   
        DHCY1=DFOUR(4,1)/DHC12  
        DHCY2=DFOUR(4,2)/DHC12  
        DHCYX=DHCXX*(DHCX1*DHCY2+DHCX2*DHCY1)*DHC12 
        DHCYY=1D0/SQRT(1D0+2D0*DHCY1*DHCY2*DHC12-DHCYX**2)  
        DO 730 J=1,4    
        DP(3,J)=DHCXX*(DP(3,J)-DHCX2*DP(1,J)-DHCX1*DP(2,J)) 
        P(IN(3),J)=sngl(DP(3,J))
  730   P(IN(3)+1,J)=sngl(DHCYY*(DP(4,J)-DHCY2*DP(1,J)-DHCY1*DP(2,J)-    
     &  DHCYX*DP(3,J))) 
C...Express pT with respect to new axes, if sensible.   
        PXP=-(PX(3)*FOUR(IN(3*JT+3),IN(3))+PY(3)*   
     &  FOUR(IN(3*JT+3)+1,IN(3)))   
        PYP=-(PX(3)*FOUR(IN(3*JT+3),IN(3)+1)+PY(3)* 
     &  FOUR(IN(3*JT+3)+1,IN(3)+1)) 
        IF(ABS(PXP**2+PYP**2-PX(3)**2-PY(3)**2).LT.0.01) THEN   
          PX(3)=PXP 
          PY(3)=PYP 
        ENDIF   
      ENDIF 
    
C...Sum up known four-momentum. Gives coefficients for m2 expression.   
      DO 750 J=1,4  
      DHG(J)=0.d0
      P(I,J)=PX(JT)*P(IN(3*JT+3),J)+PY(JT)*P(IN(3*JT+3)+1,J)+   
     &PX(3)*P(IN(3),J)+PY(3)*P(IN(3)+1,J)   
      DO 740 IN1=IN(3*JT+1),IN(1)-4*JS,4*JS 
  740 P(I,J)=P(I,J)+P(IN1+2,3)*P(IN1,J) 
      DO 750 IN2=IN(3*JT+2),IN(2)-4*JS,4*JS 
  750 P(I,J)=P(I,J)+P(IN2+2,3)*P(IN2,J) 
      DHM(1)=dble(FOUR(I,I))
      DHM(2)=dble(2.*FOUR(I,IN(1)))  
      DHM(3)=dble(2.*FOUR(I,IN(2)))
      DHM(4)=dble(2.*FOUR(IN(1),IN(2)))
    
C...Find coefficients for Gamma expression. 
      DO 760 IN2=IN(1)+1,IN(2),4    
      DO 760 IN1=IN(1),IN2-1,4  
      DHC=dble(2.*FOUR(IN1,IN2))
      DHG(1)=DHG(1)+dble(P(IN1+2,JT)*P(IN2+2,JT))*DHC 
      IF(IN1.EQ.IN(1)) DHG(2)=DHG(2)-dble(float(JS)*P(IN2+2,JT))*DHC 
      IF(IN2.EQ.IN(2)) DHG(3)=DHG(3)+dble(float(JS)*P(IN1+2,JT))*DHC 
  760 IF(IN1.EQ.IN(1).AND.IN2.EQ.IN(2)) DHG(4)=DHG(4)-DHC   
    
C...Solve (m2, Gamma) equation system for energies taken.   
      DHS1=DHM(JR+1)*DHG(4)-DHM(4)*DHG(JR+1)    
clin-5/2012:
c      IF(ABS(DHS1).LT.1E-4) GOTO 550    
      IF(DABS(DHS1).LT.1D-4) GOTO 550    
      DHS2=DHM(4)*(dble(GAM(3))-DHG(1))-DHM(JT+1)*DHG(JR+1)-DHG(4)*   
     &(dble(P(I,5))**2-DHM(1))+DHG(JT+1)*DHM(JR+1)    
      DHS3=DHM(JT+1)*(dble(GAM(3))-DHG(1))-DHG(JT+1)
     &     *(dble(P(I,5))**2-DHM(1))   
      P(IN(JR)+2,4)=0.5*sngl((SQRT(MAX(0D0,DHS2**2-4.d0*DHS1*DHS3)))
     &/ABS(DHS1)-DHS2/DHS1)
      IF(DHM(JT+1)+DHM(4)*dble(P(IN(JR)+2,4)).LE.0.d0) GOTO 550 
      P(IN(JT)+2,4)=(P(I,5)**2-sngl(DHM(1))-sngl(DHM(JR+1))
     &     *P(IN(JR)+2,4))/(sngl(DHM(JT+1))+sngl(DHM(4))*P(IN(JR)+2,4))
    
C...Step to new region if necessary.    
      IF(P(IN(JR)+2,4).GT.P(IN(JR)+2,3)) THEN   
        P(IN(JR)+2,4)=P(IN(JR)+2,3) 
        P(IN(JR)+2,JT)=1.   
        IN(JR)=IN(JR)+4*JS  
        IF(JS*IN(JR).GT.JS*IN(4*JR)) GOTO 550   
        IF(FOUR(IN(1),IN(2)).LE.1E-2) THEN  
          P(IN(JT)+2,4)=P(IN(JT)+2,3)   
          P(IN(JT)+2,JT)=0. 
          IN(JT)=IN(JT)+4*JS    
        ENDIF   
        GOTO 710    
      ELSEIF(P(IN(JT)+2,4).GT.P(IN(JT)+2,3)) THEN   
        P(IN(JT)+2,4)=P(IN(JT)+2,3) 
        P(IN(JT)+2,JT)=0.   
        IN(JT)=IN(JT)+4*JS  
        GOTO 710    
      ENDIF 
    
C...Four-momentum of particle. Remaining quantities. Loop back. 
  770 DO 780 J=1,4  
      P(I,J)=P(I,J)+P(IN(1)+2,4)*P(IN(1),J)+P(IN(2)+2,4)*P(IN(2),J) 
  780 P(N+NRS,J)=P(N+NRS,J)-P(I,J)  
      IF(P(I,4).LE.0.) GOTO 550 
      KFL(JT)=-KFL(3)   
      PMQ(JT)=PMQ(3)    
      PX(JT)=-PX(3) 
      PY(JT)=-PY(3) 
      GAM(JT)=GAM(3)    
      IF(IN(3).NE.IN(3*JT+3)) THEN  
        DO 790 J=1,4    
        P(IN(3*JT+3),J)=P(IN(3),J)  
  790   P(IN(3*JT+3)+1,J)=P(IN(3)+1,J)  
      ENDIF 
      DO 800 JQ=1,2 
      IN(3*JT+JQ)=IN(JQ)    
      P(IN(JQ)+2,3)=P(IN(JQ)+2,3)-P(IN(JQ)+2,4) 
  800 P(IN(JQ)+2,JT)=P(IN(JQ)+2,JT)-JS*(3-2*JQ)*P(IN(JQ)+2,4)   
      GOTO 670  
    
C...Final hadron: side, flavour, hadron, mass.  
  810 I=I+1 
      K(I,1)=1  
      K(I,3)=IE(JR) 
      K(I,4)=0  
      K(I,5)=0  
      CALL LUKFDI(KFL(JR),-KFL(3),KFLDMP,K(I,2))    
      IF(K(I,2).EQ.0) GOTO 550  
      P(I,5)=ULMASS(K(I,2)) 
      PR(JR)=P(I,5)**2+(PX(JR)-PX(3))**2+(PY(JR)-PY(3))**2  

C...Final two hadrons: find common setup of four-vectors.   
      JQ=1  
      IF(P(IN(4)+2,3)*P(IN(5)+2,3)*FOUR(IN(4),IN(5)).LT.P(IN(7),3)* 
     &P(IN(8),3)*FOUR(IN(7),IN(8))) JQ=2    
      DHC12=dble(FOUR(IN(3*JQ+1),IN(3*JQ+2)))
      DHR1=dble(FOUR(N+NRS,IN(3*JQ+2)))/DHC12
      DHR2=dble(FOUR(N+NRS,IN(3*JQ+1)))/DHC12
      IF(IN(4).NE.IN(7).OR.IN(5).NE.IN(8)) THEN 
        PX(3-JQ)=-FOUR(N+NRS,IN(3*JQ+3))-PX(JQ) 
        PY(3-JQ)=-FOUR(N+NRS,IN(3*JQ+3)+1)-PY(JQ)   
        PR(3-JQ)=P(I+(JT+JQ-3)**2-1,5)**2+(PX(3-JQ)+(2*JQ-3)*JS*    
     &  PX(3))**2+(PY(3-JQ)+(2*JQ-3)*JS*PY(3))**2   
      ENDIF 
    
C...Solve kinematics for final two hadrons, if possible.    
      WREM2=WREM2+(PX(1)+PX(2))**2+(PY(1)+PY(2))**2 
      FD=(SQRT(PR(1))+SQRT(PR(2)))/SQRT(WREM2)  
      IF(MJU(1)+MJU(2).NE.0.AND.I.EQ.ISAV+2.AND.FD.GE.1.) GOTO 180  
      IF(FD.GE.1.) GOTO 550 
      FA=WREM2+PR(JT)-PR(JR)    
      IF(MSTJ(11).EQ.2) PREV=0.5*FD**PARJ(37+MSTJ(11))  
      IF(MSTJ(11).NE.2) PREV=0.5*EXP(MAX(-100.,LOG(FD)* 
     &PARJ(37+MSTJ(11))*(PR(1)+PR(2))**2))  
      FB=SIGN(SQRT(MAX(0.,FA**2-4.*WREM2*PR(JT))),JS*(RLU(0)-PREV)) 
      KFL1A=IABS(KFL(1))    
      KFL2A=IABS(KFL(2))    
      IF(MAX(MOD(KFL1A,10),MOD(KFL1A/1000,10),MOD(KFL2A,10),    
     &MOD(KFL2A/1000,10)).GE.6) FB=SIGN(SQRT(MAX(0.,FA**2-  
     &4.*WREM2*PR(JT))),FLOAT(JS))  
      DO 820 J=1,4  
      P(I-1,J)=(PX(JT)+PX(3))*P(IN(3*JQ+3),J)+(PY(JT)+PY(3))*   
     &P(IN(3*JQ+3)+1,J)+0.5*(sngl(DHR1)*(FA+FB)*P(IN(3*JQ+1),J)+  
     &sngl(DHR2)*(FA-FB)*P(IN(3*JQ+2),J))/WREM2   
  820 P(I,J)=P(N+NRS,J)-P(I-1,J)    

C...Mark jets as fragmented and give daughter pointers. 
      N=I-NRS+1 
      DO 830 I=NSAV+1,NSAV+NP   
      IM=K(I,3) 
      K(IM,1)=K(IM,1)+10    
      IF(MSTU(16).NE.2) THEN    
        K(IM,4)=NSAV+1  
        K(IM,5)=NSAV+1  
      ELSE  
        K(IM,4)=NSAV+2  
        K(IM,5)=N   
      ENDIF 
  830 CONTINUE  
    
C...Document string system. Move up particles.  
      NSAV=NSAV+1   
      K(NSAV,1)=11  
      K(NSAV,2)=92  
      K(NSAV,3)=IP  
      K(NSAV,4)=NSAV+1  
      K(NSAV,5)=N   
      DO 840 J=1,4  
      P(NSAV,J)=sngl(DPS(J))
  840 V(NSAV,J)=V(IP,J) 
      P(NSAV,5)=SQRT(sngl(MAX(0D0,DPS(4)**2-DPS(1)**2-DPS(2)**2
     &     -DPS(3)**2)))
      V(NSAV,5)=0.
      DO 850 I=NSAV+1,N 

      DO 850 J=1,5  
      K(I,J)=K(I+NRS-1,J)   
      P(I,J)=P(I+NRS-1,J)   
  850 V(I,J)=0. 
    
C...Order particles in rank along the chain. Update mother pointer. 
      DO 860 I=NSAV+1,N 
      DO 860 J=1,5  
      K(I-NSAV+N,J)=K(I,J)  
  860 P(I-NSAV+N,J)=P(I,J)  
      I1=NSAV   
      DO 880 I=N+1,2*N-NSAV 
      IF(K(I,3).NE.IE(1)) GOTO 880  
      I1=I1+1   
      DO 870 J=1,5  
      K(I1,J)=K(I,J)    
  870 P(I1,J)=P(I,J)    
      IF(MSTU(16).NE.2) K(I1,3)=NSAV    
  880 CONTINUE  
      DO 900 I=2*N-NSAV,N+1,-1  
      IF(K(I,3).EQ.IE(1)) GOTO 900  
      I1=I1+1   
      DO 890 J=1,5  
      K(I1,J)=K(I,J)    
  890 P(I1,J)=P(I,J)    
      IF(MSTU(16).NE.2) K(I1,3)=NSAV    
  900 CONTINUE  
    
C...Boost back particle system. Set production vertices.    
      CALL LUDBRB(NSAV+1,N,0.,0.,DPS(1)/DPS(4),DPS(2)/DPS(4),   
     &DPS(3)/DPS(4))    
      DO 910 I=NSAV+1,N 

      DO 910 J=1,4  
  910 V(I,J)=V(IP,J)    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUINDF(IP) 
    
C...Purpose: to handle the fragmentation of a jet system (or a single   
C...jet) according to independent fragmentation models. 
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      DIMENSION DPS(5),PSI(4),NFI(3),NFL(3),IFET(3),KFLF(3),    
     &KFLO(2),PXO(2),PYO(2),WO(2)   

C...Reset counters. Identify parton system and take copy. Check flavour.    
      NSAV=N    
      NJET=0    
      KQSUM=0   
      DO 100 J=1,5  
  100 DPS(J)=0.d0
      I=IP-1    
  110 I=I+1 
      IF(I.GT.MIN(N,MSTU(4)-MSTU(32))) THEN 
        CALL LUERRM(12,'(LUINDF:) failed to reconstruct jet system')    
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      IF(K(I,1).NE.1.AND.K(I,1).NE.2) GOTO 110  
      KC=LUCOMP(K(I,2)) 
      IF(KC.EQ.0) GOTO 110  
      KQ=KCHG(KC,2)*ISIGN(1,K(I,2)) 
      IF(KQ.EQ.0) GOTO 110  
      NJET=NJET+1   
      IF(KQ.NE.2) KQSUM=KQSUM+KQ    
      DO 120 J=1,5  
      K(NSAV+NJET,J)=K(I,J) 
      P(NSAV+NJET,J)=P(I,J) 
  120 DPS(J)=DPS(J)+dble(P(I,J))
      K(NSAV+NJET,3)=I  
      IF(K(I,1).EQ.2.OR.(MSTJ(3).LE.5.AND.N.GT.I.AND.   
     &K(I+1,1).EQ.2)) GOTO 110  
      IF(NJET.NE.1.AND.KQSUM.NE.0) THEN 
        CALL LUERRM(12,'(LUINDF:) unphysical flavour combination')  
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
    
C...Boost copied system to CM frame. Find CM energy and sum flavours.   
      IF(NJET.NE.1) CALL LUDBRB(NSAV+1,NSAV+NJET,0.,0.,-DPS(1)/DPS(4),  
     &-DPS(2)/DPS(4),-DPS(3)/DPS(4))    
      PECM=0.   
      DO 130 J=1,3  
  130 NFI(J)=0  
      DO 140 I=NSAV+1,NSAV+NJET 
      PECM=PECM+P(I,4)  
      KFA=IABS(K(I,2))  
      IF(KFA.LE.3) THEN 
        NFI(KFA)=NFI(KFA)+ISIGN(1,K(I,2))   
      ELSEIF(KFA.GT.1000) THEN  
        KFLA=MOD(KFA/1000,10)   
        KFLB=MOD(KFA/100,10)    
        IF(KFLA.LE.3) NFI(KFLA)=NFI(KFLA)+ISIGN(1,K(I,2))   
        IF(KFLB.LE.3) NFI(KFLB)=NFI(KFLB)+ISIGN(1,K(I,2))   
      ENDIF 
  140 CONTINUE  
    
C...Loop over attempts made. Reset counters.    
      NTRY=0    
  150 NTRY=NTRY+1   
      N=NSAV+NJET   
      IF(NTRY.GT.200) THEN  
        CALL LUERRM(14,'(LUINDF:) caught in infinite loop') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      DO 160 J=1,3  
      NFL(J)=NFI(J) 
      IFET(J)=0 
  160 KFLF(J)=0 
    
C...Loop over jets to be fragmented.    
      DO 230 IP1=NSAV+1,NSAV+NJET   
      MSTJ(91)=0    
      NSAV1=N   
    
C...Initial flavour and momentum values. Jet along +z axis. 
      KFLH=IABS(K(IP1,2))   
      IF(KFLH.GT.10) KFLH=MOD(KFLH/1000,10) 
      KFLO(2)=0 
      WF=P(IP1,4)+SQRT(P(IP1,1)**2+P(IP1,2)**2+P(IP1,3)**2) 
    
C...Initial values for quark or diquark jet.    
  170 IF(IABS(K(IP1,2)).NE.21) THEN 
        NSTR=1  
        KFLO(1)=K(IP1,2)    
        CALL LUPTDI(0,PXO(1),PYO(1))    
        WO(1)=WF    
    
C...Initial values for gluon treated like random quark jet. 
      ELSEIF(MSTJ(2).LE.2) THEN 
        NSTR=1  
        IF(MSTJ(2).EQ.2) MSTJ(91)=1 
        KFLO(1)=INT(1.+(2.+PARJ(2))*RLU(0))*(-1)**INT(RLU(0)+0.5)   
        CALL LUPTDI(0,PXO(1),PYO(1))    
        WO(1)=WF    
    
C...Initial values for gluon treated like quark-antiquark jet pair, 
C...sharing energy according to Altarelli-Parisi splitting function.    
      ELSE  
        NSTR=2  
        IF(MSTJ(2).EQ.4) MSTJ(91)=1 
        KFLO(1)=INT(1.+(2.+PARJ(2))*RLU(0))*(-1)**INT(RLU(0)+0.5)   
        KFLO(2)=-KFLO(1)    
        CALL LUPTDI(0,PXO(1),PYO(1))    
        PXO(2)=-PXO(1)  
        PYO(2)=-PYO(1)  
        WO(1)=WF*RLU(0)**(1./3.)    
        WO(2)=WF-WO(1)  
      ENDIF 
    
C...Initial values for rank, flavour, pT and W+.    
      DO 220 ISTR=1,NSTR    
  180 I=N   
      IRANK=0   
      KFL1=KFLO(ISTR)   
      PX1=PXO(ISTR) 
      PY1=PYO(ISTR) 
      W=WO(ISTR)    
    
C...New hadron. Generate flavour and hadron species.    
  190 I=I+1 
      IF(I.GE.MSTU(4)-MSTU(32)-NJET-5) THEN 
        CALL LUERRM(11,'(LUINDF:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      IRANK=IRANK+1 
      K(I,1)=1  
      K(I,3)=IP1    
      K(I,4)=0  
      K(I,5)=0  
  200 CALL LUKFDI(KFL1,0,KFL2,K(I,2))   
      IF(K(I,2).EQ.0) GOTO 180  
      IF(MSTJ(12).GE.3.AND.IRANK.EQ.1.AND.IABS(KFL1).LE.10.AND. 
     &IABS(KFL2).GT.10) THEN    
        IF(RLU(0).GT.PARJ(19)) GOTO 200 
      ENDIF 
    
C...Find hadron mass. Generate four-momentum.   
      P(I,5)=ULMASS(K(I,2)) 
      CALL LUPTDI(KFL1,PX2,PY2) 
      P(I,1)=PX1+PX2    
      P(I,2)=PY1+PY2    
      PR=P(I,5)**2+P(I,1)**2+P(I,2)**2  
      CALL LUZDIS(KFL1,KFL2,PR,Z)   
      P(I,3)=0.5*(Z*W-PR/(Z*W)) 
      P(I,4)=0.5*(Z*W+PR/(Z*W)) 
      IF(MSTJ(3).GE.1.AND.IRANK.EQ.1.AND.KFLH.GE.4.AND. 
     &P(I,3).LE.0.001) THEN 
        IF(W.GE.P(I,5)+0.5*PARJ(32)) GOTO 180   
        P(I,3)=0.0001   
        P(I,4)=SQRT(PR) 
        Z=P(I,4)/W  
      ENDIF 
    
C...Remaining flavour and momentum. 
      KFL1=-KFL2    
      PX1=-PX2  
      PY1=-PY2  
      W=(1.-Z)*W    
      DO 210 J=1,5  
  210 V(I,J)=0. 
    
C...Check if pL acceptable. Go back for new hadron if enough energy.    
      IF(MSTJ(3).GE.0.AND.P(I,3).LT.0.) I=I-1   
      IF(W.GT.PARJ(31)) GOTO 190    
  220 N=I   
      IF(MOD(MSTJ(3),5).EQ.4.AND.N.EQ.NSAV1) WF=WF+0.1*PARJ(32) 
      IF(MOD(MSTJ(3),5).EQ.4.AND.N.EQ.NSAV1) GOTO 170   
    
C...Rotate jet to new direction.    
      THE=ULANGL(P(IP1,3),SQRT(P(IP1,1)**2+P(IP1,2)**2))    
      PHI=ULANGL(P(IP1,1),P(IP1,2)) 
      CALL LUDBRB(NSAV1+1,N,THE,PHI,0D0,0D0,0D0)    
      K(K(IP1,3),4)=NSAV1+1 
      K(K(IP1,3),5)=N   
    
C...End of jet generation loop. Skip conservation in some cases.    
  230 CONTINUE  
      IF(NJET.EQ.1.OR.MSTJ(3).LE.0) GOTO 470    
      IF(MOD(MSTJ(3),5).NE.0.AND.N-NSAV-NJET.LT.2) GOTO 150 
    
C...Subtract off produced hadron flavours, finished if zero.    
      DO 240 I=NSAV+NJET+1,N    
      KFA=IABS(K(I,2))  
      KFLA=MOD(KFA/1000,10) 
      KFLB=MOD(KFA/100,10)  
      KFLC=MOD(KFA/10,10)   
      IF(KFLA.EQ.0) THEN    
        IF(KFLB.LE.3) NFL(KFLB)=NFL(KFLB)-ISIGN(1,K(I,2))*(-1)**KFLB    
        IF(KFLC.LE.3) NFL(KFLC)=NFL(KFLC)+ISIGN(1,K(I,2))*(-1)**KFLB    
      ELSE  
        IF(KFLA.LE.3) NFL(KFLA)=NFL(KFLA)-ISIGN(1,K(I,2))   
        IF(KFLB.LE.3) NFL(KFLB)=NFL(KFLB)-ISIGN(1,K(I,2))   
        IF(KFLC.LE.3) NFL(KFLC)=NFL(KFLC)-ISIGN(1,K(I,2))   
      ENDIF 
  240 CONTINUE  
      NREQ=(IABS(NFL(1))+IABS(NFL(2))+IABS(NFL(3))-IABS(NFL(1)+ 
     &NFL(2)+NFL(3)))/2+IABS(NFL(1)+NFL(2)+NFL(3))/3    
      IF(NREQ.EQ.0) GOTO 320    
    
C...Take away flavour of low-momentum particles until enough freedom.   
      NREM=0    
  250 IREM=0    
      P2MIN=PECM**2 
      DO 260 I=NSAV+NJET+1,N    
      P2=P(I,1)**2+P(I,2)**2+P(I,3)**2  
      IF(K(I,1).EQ.1.AND.P2.LT.P2MIN) IREM=I    
  260 IF(K(I,1).EQ.1.AND.P2.LT.P2MIN) P2MIN=P2  
      IF(IREM.EQ.0) GOTO 150    
      K(IREM,1)=7   
      KFA=IABS(K(IREM,2))   
      KFLA=MOD(KFA/1000,10) 
      KFLB=MOD(KFA/100,10)  
      KFLC=MOD(KFA/10,10)   
      IF(KFLA.GE.4.OR.KFLB.GE.4) K(IREM,1)=8    
      IF(K(IREM,1).EQ.8) GOTO 250   
      IF(KFLA.EQ.0) THEN    
        ISGN=ISIGN(1,K(IREM,2))*(-1)**KFLB  
        IF(KFLB.LE.3) NFL(KFLB)=NFL(KFLB)+ISGN  
        IF(KFLC.LE.3) NFL(KFLC)=NFL(KFLC)-ISGN  
      ELSE  
        IF(KFLA.LE.3) NFL(KFLA)=NFL(KFLA)+ISIGN(1,K(IREM,2))    
        IF(KFLB.LE.3) NFL(KFLB)=NFL(KFLB)+ISIGN(1,K(IREM,2))    
        IF(KFLC.LE.3) NFL(KFLC)=NFL(KFLC)+ISIGN(1,K(IREM,2))    
      ENDIF 
      NREM=NREM+1   
      NREQ=(IABS(NFL(1))+IABS(NFL(2))+IABS(NFL(3))-IABS(NFL(1)+ 
     &NFL(2)+NFL(3)))/2+IABS(NFL(1)+NFL(2)+NFL(3))/3    
      IF(NREQ.GT.NREM) GOTO 250 
      DO 270 I=NSAV+NJET+1,N    
  270 IF(K(I,1).EQ.8) K(I,1)=1  
    
C...Find combination of existing and new flavours for hadron.   
  280 NFET=2    
      IF(NFL(1)+NFL(2)+NFL(3).NE.0) NFET=3  
      IF(NREQ.LT.NREM) NFET=1   
      IF(IABS(NFL(1))+IABS(NFL(2))+IABS(NFL(3)).EQ.0) NFET=0    
      DO 290 J=1,NFET   
      IFET(J)=1+int((IABS(NFL(1))+IABS(NFL(2))+IABS(NFL(3)))*RLU(0))
      KFLF(J)=ISIGN(1,NFL(1))   
      IF(IFET(J).GT.IABS(NFL(1))) KFLF(J)=ISIGN(2,NFL(2))   
  290 IF(IFET(J).GT.IABS(NFL(1))+IABS(NFL(2))) KFLF(J)=ISIGN(3,NFL(3))  
      IF(NFET.EQ.2.AND.(IFET(1).EQ.IFET(2).OR.KFLF(1)*KFLF(2).GT.0))    
     &GOTO 280  
      IF(NFET.EQ.3.AND.(IFET(1).EQ.IFET(2).OR.IFET(1).EQ.IFET(3).OR.    
     &IFET(2).EQ.IFET(3).OR.KFLF(1)*KFLF(2).LT.0.OR.KFLF(1)*KFLF(3).    
     &LT.0.OR.KFLF(1)*(NFL(1)+NFL(2)+NFL(3)).LT.0)) GOTO 280    
      IF(NFET.EQ.0) KFLF(1)=1+INT((2.+PARJ(2))*RLU(0))  
      IF(NFET.EQ.0) KFLF(2)=-KFLF(1)    
      IF(NFET.EQ.1) KFLF(2)=ISIGN(1+INT((2.+PARJ(2))*RLU(0)),-KFLF(1))  
      IF(NFET.LE.2) KFLF(3)=0   
      IF(KFLF(3).NE.0) THEN 
        KFLFC=ISIGN(1000*MAX(IABS(KFLF(1)),IABS(KFLF(3)))+  
     &  100*MIN(IABS(KFLF(1)),IABS(KFLF(3)))+1,KFLF(1)) 
        IF(KFLF(1).EQ.KFLF(3).OR.(1.+3.*PARJ(4))*RLU(0).GT.1.)  
     &  KFLFC=KFLFC+ISIGN(2,KFLFC)  
      ELSE  
        KFLFC=KFLF(1)   
      ENDIF 
      CALL LUKFDI(KFLFC,KFLF(2),KFLDMP,KF)  
      IF(KF.EQ.0) GOTO 280  
      DO 300 J=1,MAX(2,NFET)    
  300 NFL(IABS(KFLF(J)))=NFL(IABS(KFLF(J)))-ISIGN(1,KFLF(J))    
    
C...Store hadron at random among free positions.    
      NPOS=MIN(1+INT(RLU(0)*NREM),NREM) 
      DO 310 I=NSAV+NJET+1,N    
      IF(K(I,1).EQ.7) NPOS=NPOS-1   
      IF(K(I,1).EQ.1.OR.NPOS.NE.0) GOTO 310 
      K(I,1)=1  
      K(I,2)=KF 
      P(I,5)=ULMASS(K(I,2)) 
      P(I,4)=SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2+P(I,5)**2)  
  310 CONTINUE  
      NREM=NREM-1   
      NREQ=(IABS(NFL(1))+IABS(NFL(2))+IABS(NFL(3))-IABS(NFL(1)+ 
     &NFL(2)+NFL(3)))/2+IABS(NFL(1)+NFL(2)+NFL(3))/3    
      IF(NREM.GT.0) GOTO 280    
    
C...Compensate for missing momentum in global scheme (3 options).   
  320 IF(MOD(MSTJ(3),5).NE.0.AND.MOD(MSTJ(3),5).NE.4) THEN  
        DO 330 J=1,3    
        PSI(J)=0.   
        DO 330 I=NSAV+NJET+1,N  
  330   PSI(J)=PSI(J)+P(I,J)    
        PSI(4)=PSI(1)**2+PSI(2)**2+PSI(3)**2    
        PWS=0.  
        DO 340 I=NSAV+NJET+1,N  
        IF(MOD(MSTJ(3),5).EQ.1) PWS=PWS+P(I,4)  
        IF(MOD(MSTJ(3),5).EQ.2) PWS=PWS+SQRT(P(I,5)**2+(PSI(1)*P(I,1)+  
     &  PSI(2)*P(I,2)+PSI(3)*P(I,3))**2/PSI(4)) 
  340   IF(MOD(MSTJ(3),5).EQ.3) PWS=PWS+1.  
cms..preinitialize
        PW=0.
        DO 360 I=NSAV+NJET+1,N  
        IF(MOD(MSTJ(3),5).EQ.1) PW=P(I,4)   
        IF(MOD(MSTJ(3),5).EQ.2) PW=SQRT(P(I,5)**2+(PSI(1)*P(I,1)+   
     &  PSI(2)*P(I,2)+PSI(3)*P(I,3))**2/PSI(4)) 
        IF(MOD(MSTJ(3),5).EQ.3) PW=1.   
        DO 350 J=1,3    
  350   P(I,J)=P(I,J)-PSI(J)*PW/PWS 
  360   P(I,4)=SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2+P(I,5)**2)    
    
C...Compensate for missing momentum withing each jet separately.    
      ELSEIF(MOD(MSTJ(3),5).EQ.4) THEN  
        DO 370 I=N+1,N+NJET 
        K(I,1)=0    
        DO 370 J=1,5    
  370   P(I,J)=0.   
        DO 390 I=NSAV+NJET+1,N  
        IR1=K(I,3)  
        IR2=N+IR1-NSAV  
        K(IR2,1)=K(IR2,1)+1 
        PLS=(P(I,1)*P(IR1,1)+P(I,2)*P(IR1,2)+P(I,3)*P(IR1,3))/  
     &  (P(IR1,1)**2+P(IR1,2)**2+P(IR1,3)**2)   
        DO 380 J=1,3    
  380   P(IR2,J)=P(IR2,J)+P(I,J)-PLS*P(IR1,J)   
        P(IR2,4)=P(IR2,4)+P(I,4)    
  390   P(IR2,5)=P(IR2,5)+PLS   
        PSS=0.  
        DO 400 I=N+1,N+NJET 
  400   IF(K(I,1).NE.0) PSS=PSS+P(I,4)/(PECM*(0.8*P(I,5)+0.2))  
        DO 420 I=NSAV+NJET+1,N  
        IR1=K(I,3)  
        IR2=N+IR1-NSAV  
        PLS=(P(I,1)*P(IR1,1)+P(I,2)*P(IR1,2)+P(I,3)*P(IR1,3))/  
     &  (P(IR1,1)**2+P(IR1,2)**2+P(IR1,3)**2)   
        DO 410 J=1,3    
  410   P(I,J)=P(I,J)-P(IR2,J)/K(IR2,1)+(1./(P(IR2,5)*PSS)-1.)*PLS* 
     &  P(IR1,J)    
  420   P(I,4)=SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2+P(I,5)**2)    
      ENDIF 
    
C...Scale momenta for energy conservation.  
      IF(MOD(MSTJ(3),5).NE.0) THEN  
        PMS=0.  
        PES=0.  
        PQS=0.  
        DO 430 I=NSAV+NJET+1,N  
        PMS=PMS+P(I,5)  
        PES=PES+P(I,4)  
  430   PQS=PQS+P(I,5)**2/P(I,4)    
        IF(PMS.GE.PECM) GOTO 150    
        NECO=0  
  440   NECO=NECO+1 
        PFAC=(PECM-PQS)/(PES-PQS)   
        PES=0.  
        PQS=0.  
        DO 460 I=NSAV+NJET+1,N  
        DO 450 J=1,3    
  450   P(I,J)=PFAC*P(I,J)  
        P(I,4)=SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2+P(I,5)**2)    
        PES=PES+P(I,4)  
  460   PQS=PQS+P(I,5)**2/P(I,4)    
        IF(NECO.LT.10.AND.ABS(PECM-PES).GT.2E-6*PECM) GOTO 440  
      ENDIF 
    
C...Origin of produced particles and parton daughter pointers.  
  470 DO 480 I=NSAV+NJET+1,N    
      IF(MSTU(16).NE.2) K(I,3)=NSAV+1   
  480 IF(MSTU(16).EQ.2) K(I,3)=K(K(I,3),3)  
      DO 490 I=NSAV+1,NSAV+NJET 
      I1=K(I,3) 
      K(I1,1)=K(I1,1)+10    
      IF(MSTU(16).NE.2) THEN    
        K(I1,4)=NSAV+1  
        K(I1,5)=NSAV+1  
      ELSE  
        K(I1,4)=K(I1,4)-NJET+1  
        K(I1,5)=K(I1,5)-NJET+1  
        IF(K(I1,5).LT.K(I1,4)) THEN 
          K(I1,4)=0 
          K(I1,5)=0 
        ENDIF   
      ENDIF 
  490 CONTINUE  
    
C...Document independent fragmentation system. Remove copy of jets. 
      NSAV=NSAV+1   
      K(NSAV,1)=11  
      K(NSAV,2)=93  
      K(NSAV,3)=IP  
      K(NSAV,4)=NSAV+1  
      K(NSAV,5)=N-NJET+1    
      DO 500 J=1,4  
      P(NSAV,J)=sngl(DPS(J))
  500 V(NSAV,J)=V(IP,J) 
      P(NSAV,5)=SQRT(sngl(MAX(0D0,DPS(4)**2-DPS(1)**2-DPS(2)**2
     &     -DPS(3)**2)))
      V(NSAV,5)=0.  
      DO 510 I=NSAV+NJET,N  
      DO 510 J=1,5  
      K(I-NJET+1,J)=K(I,J)  
      P(I-NJET+1,J)=P(I,J)  
  510 V(I-NJET+1,J)=V(I,J)  
      N=N-NJET+1    
    
C...Boost back particle system. Set production vertices.    
      IF(NJET.NE.1) CALL LUDBRB(NSAV+1,N,0.,0.,DPS(1)/DPS(4),   
     &DPS(2)/DPS(4),DPS(3)/DPS(4))  
      DO 520 I=NSAV+1,N 
      DO 520 J=1,4  
  520 V(I,J)=V(IP,J)    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUDECY(IP) 
    
C...Purpose: to handle the decay of unstable particles. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      DIMENSION VDCY(4),KFLO(4),KFL1(4),PV(10,5),RORD(10),UE(3),BE(3),  
     &WTCOR(10) 
clin-2/18/03 for resonance decay in hadron cascade:
      common/resdcy/NSAV,iksdcy
      SAVE /resdcy/
      DATA WTCOR/2.,5.,15.,60.,250.,1500.,1.2E4,1.2E5,150.,16./ 
    
C...Functions: momentum in two-particle decays, four-product and    
C...matrix element times phase space in weak decays.    
      PAWT(A,B,C)=SQRT((A**2-(B+C)**2)*(A**2-(B-C)**2))/(2.*A)  
      FOUR(I,J)=P(I,4)*P(J,4)-P(I,1)*P(J,1)-P(I,2)*P(J,2)-P(I,3)*P(J,3) 
      HMEPS(HA)=((1.-HRQ-HA)**2+3.*HA*(1.+HRQ-HA))* 
     &SQRT((1.-HRQ-HA)**2-4.*HRQ*HA)    
    
C...Initial values. 
      NTRY=0    
      NSAV=N    
      KFA=IABS(K(IP,2)) 
      KFS=ISIGN(1,K(IP,2))  
      KC=LUCOMP(KFA)    
      MSTJ(92)=0    
    
C...Choose lifetime and determine decay vertex. 
      IF(K(IP,1).EQ.5) THEN 
        V(IP,5)=0.  
      ELSEIF(K(IP,1).NE.4) THEN 
        V(IP,5)=-PMAS(KC,4)*LOG(RLU(0)) 
      ENDIF 
      DO 100 J=1,4  
  100 VDCY(J)=V(IP,J)+V(IP,5)*P(IP,J)/P(IP,5)   
    
C...Determine whether decay allowed or not. 
      MOUT=0    
      IF(MSTJ(22).EQ.2) THEN    
        IF(PMAS(KC,4).GT.PARJ(71)) MOUT=1   
      ELSEIF(MSTJ(22).EQ.3) THEN    
        IF(VDCY(1)**2+VDCY(2)**2+VDCY(3)**2.GT.PARJ(72)**2) MOUT=1  
      ELSEIF(MSTJ(22).EQ.4) THEN    
        IF(VDCY(1)**2+VDCY(2)**2.GT.PARJ(73)**2) MOUT=1 
        IF(ABS(VDCY(3)).GT.PARJ(74)) MOUT=1 
      ENDIF 
      IF(MOUT.EQ.1.AND.K(IP,1).NE.5) THEN   
        K(IP,1)=4   
        RETURN  
      ENDIF 
    
C...Check existence of decay channels. Particle/antiparticle rules. 
      KCA=KC    
      IF(MDCY(KC,2).GT.0) THEN  
        MDMDCY=MDME(MDCY(KC,2),2)   
        IF(MDMDCY.GT.80.AND.MDMDCY.LE.90) KCA=MDMDCY    
      ENDIF 
      IF(MDCY(KCA,2).LE.0.OR.MDCY(KCA,3).LE.0) THEN 
        CALL LUERRM(9,'(LUDECY:) no decay channel defined') 
        RETURN  
      ENDIF 
      IF(MOD(KFA/1000,10).EQ.0.AND.(KCA.EQ.85.OR.KCA.EQ.87)) KFS=-KFS   
      IF(KCHG(KC,3).EQ.0) THEN  
        KFSP=1  
        KFSN=0  
        IF(RLU(0).GT.0.5) KFS=-KFS  
      ELSEIF(KFS.GT.0) THEN 
        KFSP=1  
        KFSN=0  
      ELSE  
        KFSP=0  
        KFSN=1  
      ENDIF 
    
C...Sum branching ratios of allowed decay channels. 
clin  110 NOPE=0    
      NOPE=0    
      BRSU=0.   
      DO 120 IDL=MDCY(KCA,2),MDCY(KCA,2)+MDCY(KCA,3)-1  
      IF(MDME(IDL,1).NE.1.AND.KFSP*MDME(IDL,1).NE.2.AND.    
     &KFSN*MDME(IDL,1).NE.3) GOTO 120   
      IF(MDME(IDL,2).GT.100) GOTO 120   
      NOPE=NOPE+1   
      BRSU=BRSU+BRAT(IDL)   
  120 CONTINUE  
      IF(NOPE.EQ.0) THEN    
        CALL LUERRM(2,'(LUDECY:) all decay channels closed by user')    
        RETURN  
      ENDIF 
    
C...Select decay channel among allowed ones.    
  130 RBR=BRSU*RLU(0)   
      IDL=MDCY(KCA,2)-1 
cms.. preinitialize..
      IDC=0.
  140 IDL=IDL+1 
      IF(MDME(IDL,1).NE.1.AND.KFSP*MDME(IDL,1).NE.2.AND.    
     &KFSN*MDME(IDL,1).NE.3) THEN   
        IF(IDL.LT.MDCY(KCA,2)+MDCY(KCA,3)-1) GOTO 140   
      ELSEIF(MDME(IDL,2).GT.100) THEN   
        IF(IDL.LT.MDCY(KCA,2)+MDCY(KCA,3)-1) GOTO 140   
      ELSE  
        IDC=IDL 
        RBR=RBR-BRAT(IDL)   
        IF(IDL.LT.MDCY(KCA,2)+MDCY(KCA,3)-1.AND.RBR.GT.0.) GOTO 140 
      ENDIF 
    
C...Start readout of decay channel: matrix element, reset counters. 
      MMAT=MDME(IDC,2)  
  150 NTRY=NTRY+1   
      IF(NTRY.GT.1000) THEN 
        CALL LUERRM(14,'(LUDECY:) caught in infinite loop') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      I=N   
      NP=0  
      NQ=0  
      MBST=0    
      IF(MMAT.GE.11.AND.MMAT.NE.46.AND.P(IP,4).GT.20.*P(IP,5)) MBST=1   
      DO 160 J=1,4  
      PV(1,J)=0.    
  160 IF(MBST.EQ.0) PV(1,J)=P(IP,J) 
      IF(MBST.EQ.1) PV(1,4)=P(IP,5) 
      PV(1,5)=P(IP,5)   
      PS=0. 
      PSQ=0.    
      MREM=0    
    
C...Read out decay products. Convert to standard flavour code.  
      JTMAX=5   
      IF(MDME(IDC+1,2).EQ.101) JTMAX=10 
      DO 170 JT=1,JTMAX 
      IF(JT.LE.5) KP=KFDP(IDC,JT)   
      IF(JT.GE.6) KP=KFDP(IDC+1,JT-5)   
      IF(KP.EQ.0) GOTO 170  
      KPA=IABS(KP)  
      KCP=LUCOMP(KPA)   
      IF(KCHG(KCP,3).EQ.0.AND.KPA.NE.81.AND.KPA.NE.82) THEN 
        KFP=KP  
      ELSEIF(KPA.NE.81.AND.KPA.NE.82) THEN  
        KFP=KFS*KP  
      ELSEIF(KPA.EQ.81.AND.MOD(KFA/1000,10).EQ.0) THEN  
        KFP=-KFS*MOD(KFA/10,10) 
      ELSEIF(KPA.EQ.81.AND.MOD(KFA/100,10).GE.MOD(KFA/10,10)) THEN  
        KFP=KFS*(100*MOD(KFA/10,100)+3) 
      ELSEIF(KPA.EQ.81) THEN    
        KFP=KFS*(1000*MOD(KFA/10,10)+100*MOD(KFA/100,10)+1) 
      ELSEIF(KP.EQ.82) THEN 
        CALL LUKFDI(-KFS*INT(1.+(2.+PARJ(2))*RLU(0)),0,KFP,KDUMP)   
        IF(KFP.EQ.0) GOTO 150   
        MSTJ(93)=1  
        IF(PV(1,5).LT.PARJ(32)+2.*ULMASS(KFP)) GOTO 150 
      ELSEIF(KP.EQ.-82) THEN    
        KFP=-KFP    
        IF(IABS(KFP).GT.10) KFP=KFP+ISIGN(10000,KFP)    
      ENDIF 
      IF(KPA.EQ.81.OR.KPA.EQ.82) KCP=LUCOMP(KFP)    
    
C...Add decay product to event record or to quark flavour list. 
      KFPA=IABS(KFP)    
      KQP=KCHG(KCP,2)   
      IF(MMAT.GE.11.AND.MMAT.LE.30.AND.KQP.NE.0) THEN   
        NQ=NQ+1 
        KFLO(NQ)=KFP    
        MSTJ(93)=2  
        PSQ=PSQ+ULMASS(KFLO(NQ))    
      ELSEIF(MMAT.GE.42.AND.MMAT.LE.43.AND.NP.EQ.3.AND.MOD(NQ,2).EQ.1)  
     &THEN  
        NQ=NQ-1 
        PS=PS-P(I,5)    
        K(I,1)=1    
        KFI=K(I,2)  
        CALL LUKFDI(KFP,KFI,KFLDMP,K(I,2))  
        IF(K(I,2).EQ.0) GOTO 150    
        MSTJ(93)=1  
        P(I,5)=ULMASS(K(I,2))   
        PS=PS+P(I,5)    
      ELSE  
        I=I+1   
        NP=NP+1 
        IF(MMAT.NE.33.AND.KQP.NE.0) NQ=NQ+1 
        IF(MMAT.EQ.33.AND.KQP.NE.0.AND.KQP.NE.2) NQ=NQ+1    
        K(I,1)=1+MOD(NQ,2)  
        IF(MMAT.EQ.4.AND.JT.LE.2.AND.KFP.EQ.21) K(I,1)=2    
        IF(MMAT.EQ.4.AND.JT.EQ.3) K(I,1)=1  
        K(I,2)=KFP  
        K(I,3)=IP   
        K(I,4)=0    
        K(I,5)=0    
        P(I,5)=ULMASS(KFP)  
        IF(MMAT.EQ.45.AND.KFPA.EQ.89) P(I,5)=PARJ(32)   
        PS=PS+P(I,5)    
      ENDIF 
  170 CONTINUE  
    
C...Choose decay multiplicity in phase space model. 
cms.. preinitialize
      PQT=0.

  180 IF(MMAT.GE.11.AND.MMAT.LE.30) THEN    
        PSP=PS  
        CNDE=PARJ(61)*LOG(MAX((PV(1,5)-PS-PSQ)/PARJ(62),1.1))   
        IF(MMAT.EQ.12) CNDE=CNDE+PARJ(63)   
  190   NTRY=NTRY+1 
        IF(NTRY.GT.1000) THEN   
          CALL LUERRM(14,'(LUDECY:) caught in infinite loop')   
          IF(MSTU(21).GE.1) RETURN  
        ENDIF   
        IF(MMAT.LE.20) THEN 
          GAUSS=SQRT(-2.*CNDE*LOG(MAX(1E-10,RLU(0))))*  
     &    SIN(PARU(2)*RLU(0))   
          ND=int(0.5+0.5*NP+0.25*NQ+CNDE+GAUSS)
          IF(ND.LT.NP+NQ/2.OR.ND.LT.2.OR.ND.GT.10) GOTO 190 
          IF(MMAT.EQ.13.AND.ND.EQ.2) GOTO 190   
          IF(MMAT.EQ.14.AND.ND.LE.3) GOTO 190   
          IF(MMAT.EQ.15.AND.ND.LE.4) GOTO 190   
        ELSE    
          ND=MMAT-20    
        ENDIF   
    
C...Form hadrons from flavour content.  
        DO 200 JT=1,4   
  200   KFL1(JT)=KFLO(JT)   
        IF(ND.EQ.NP+NQ/2) GOTO 220  
        DO 210 I=N+NP+1,N+ND-NQ/2   
        JT=1+INT((NQ-1)*RLU(0)) 
        CALL LUKFDI(KFL1(JT),0,KFL2,K(I,2)) 
        IF(K(I,2).EQ.0) GOTO 190    
  210   KFL1(JT)=-KFL2  
  220   JT=2    
        JT2=3   
        JT3=4   
        IF(NQ.EQ.4.AND.RLU(0).LT.PARJ(66)) JT=4 
        IF(JT.EQ.4.AND.ISIGN(1,KFL1(1)*(10-IABS(KFL1(1))))* 
     &  ISIGN(1,KFL1(JT)*(10-IABS(KFL1(JT)))).GT.0) JT=3    
        IF(JT.EQ.3) JT2=2   
        IF(JT.EQ.4) JT3=2   
        CALL LUKFDI(KFL1(1),KFL1(JT),KFLDMP,K(N+ND-NQ/2+1,2))   
        IF(K(N+ND-NQ/2+1,2).EQ.0) GOTO 190  
        IF(NQ.EQ.4) CALL LUKFDI(KFL1(JT2),KFL1(JT3),KFLDMP,K(N+ND,2))   
        IF(NQ.EQ.4.AND.K(N+ND,2).EQ.0) GOTO 190 
    
C...Check that sum of decay product masses not too large.   
        PS=PSP  
        DO 230 I=N+NP+1,N+ND    
        K(I,1)=1    
        K(I,3)=IP   
        K(I,4)=0    
        K(I,5)=0    
        P(I,5)=ULMASS(K(I,2))   
  230   PS=PS+P(I,5)    
        IF(PS+PARJ(64).GT.PV(1,5)) GOTO 190 
    
C...Rescale energy to subtract off spectator quark mass.    
      ELSEIF((MMAT.EQ.31.OR.MMAT.EQ.33.OR.MMAT.EQ.44.OR.MMAT.EQ.45).    
     &AND.NP.GE.3) THEN 
        PS=PS-P(N+NP,5) 
        PQT=(P(N+NP,5)+PARJ(65))/PV(1,5)    
        DO 240 J=1,5    
        P(N+NP,J)=PQT*PV(1,J)   
  240   PV(1,J)=(1.-PQT)*PV(1,J)    
        IF(PS+PARJ(64).GT.PV(1,5)) GOTO 150 
        ND=NP-1 
        MREM=1  
    
C...Phase space factors imposed in W decay. 
      ELSEIF(MMAT.EQ.46) THEN   
        MSTJ(93)=1  
        PSMC=ULMASS(K(N+1,2))   
        MSTJ(93)=1  
        PSMC=PSMC+ULMASS(K(N+2,2))  
        IF(MAX(PS,PSMC)+PARJ(32).GT.PV(1,5)) GOTO 130   
        HR1=(P(N+1,5)/PV(1,5))**2   
        HR2=(P(N+2,5)/PV(1,5))**2   
        IF((1.-HR1-HR2)*(2.+HR1+HR2)*SQRT((1.-HR1-HR2)**2-4.*HR1*HR2).  
     &  LT.2.*RLU(0)) GOTO 130  
        ND=NP   
    
C...Fully specified final state: check mass broadening effects. 
      ELSE  
        IF(NP.GE.2.AND.PS+PARJ(64).GT.PV(1,5)) GOTO 150 
        ND=NP   
      ENDIF 
    
C...Select W mass in decay Q -> W + q, without W propagator.    
      IF(MMAT.EQ.45.AND.MSTJ(25).LE.0) THEN 
        HLQ=(PARJ(32)/PV(1,5))**2   
        HUQ=(1.-(P(N+2,5)+PARJ(64))/PV(1,5))**2 
        HRQ=(P(N+2,5)/PV(1,5))**2   
  250   HW=HLQ+RLU(0)*(HUQ-HLQ) 
        IF(HMEPS(HW).LT.RLU(0)) GOTO 250    
        P(N+1,5)=PV(1,5)*SQRT(HW)   
    
C...Ditto, including W propagator. Divide mass range into three regions.    
      ELSEIF(MMAT.EQ.45) THEN   
        HQW=(PV(1,5)/PMAS(24,1))**2 
        HLW=(PARJ(32)/PMAS(24,1))**2    
        HUW=((PV(1,5)-P(N+2,5)-PARJ(64))/PMAS(24,1))**2 
        HRQ=(P(N+2,5)/PV(1,5))**2   
        HG=PMAS(24,2)/PMAS(24,1)    
        HATL=ATAN((HLW-1.)/HG)  
        HM=MIN(1.,HUW-0.001)    
        HMV1=HMEPS(HM/HQW)/((HM-1.)**2+HG**2)   
  260   HM=HM-HG    
        HMV2=HMEPS(HM/HQW)/((HM-1.)**2+HG**2)   
        HSAV1=HMEPS(HM/HQW) 
        HSAV2=1./((HM-1.)**2+HG**2) 
        IF(HMV2.GT.HMV1.AND.HM-HG.GT.HLW) THEN  
          HMV1=HMV2 
          GOTO 260  
        ENDIF   
        HMV=MIN(2.*HMV1,HMEPS(HM/HQW)/HG**2)    
        HM1=1.-SQRT(1./HMV-HG**2)   
        IF(HM1.GT.HLW.AND.HM1.LT.HM) THEN   
          HM=HM1    
        ELSEIF(HMV2.LE.HMV1) THEN   
          HM=MAX(HLW,HM-MIN(0.1,1.-HM)) 
        ENDIF   
        HATM=ATAN((HM-1.)/HG)   
        HWT1=(HATM-HATL)/HG 
        HWT2=HMV*(MIN(1.,HUW)-HM)   
        HWT3=0. 
cms.. preinitialize..
        HMP1=0.
        HATU=0.
        IF(HUW.GT.1.) THEN  
          HATU=ATAN((HUW-1.)/HG)    
          HMP1=HMEPS(1./HQW)    
          HWT3=HMP1*HATU/HG 
        ENDIF   
    
C...Select mass region and W mass there. Accept according to weight.    
  270   HREG=RLU(0)*(HWT1+HWT2+HWT3)    
        IF(HREG.LE.HWT1) THEN   
          HW=1.+HG*TAN(HATL+RLU(0)*(HATM-HATL)) 
          HACC=HMEPS(HW/HQW)    
        ELSEIF(HREG.LE.HWT1+HWT2) THEN  
          HW=HM+RLU(0)*(MIN(1.,HUW)-HM) 
          HACC=HMEPS(HW/HQW)/((HW-1.)**2+HG**2)/HMV 
        ELSE    
          HW=1.+HG*TAN(RLU(0)*HATU) 
          HACC=HMEPS(HW/HQW)/HMP1   
        ENDIF   
        IF(HACC.LT.RLU(0)) GOTO 270 
        P(N+1,5)=PMAS(24,1)*SQRT(HW)    
      ENDIF 
    
C...Determine position of grandmother, number of sisters, Q -> W sign.  
      NM=0  
      MSGN=0    
cms..preinitialize
      IM=0
      IF(MMAT.EQ.3.OR.MMAT.EQ.46) THEN  
        IM=K(IP,3)  
        IF(IM.LT.0.OR.IM.GE.IP) IM=0    
        IF(IM.NE.0) KFAM=IABS(K(IM,2))  
        IF(IM.NE.0.AND.MMAT.EQ.3) THEN  
          DO 280 IL=MAX(IP-2,IM+1),MIN(IP+2,N)  
  280     IF(K(IL,3).EQ.IM) NM=NM+1 
          IF(NM.NE.2.OR.KFAM.LE.100.OR.MOD(KFAM,10).NE.1.OR.    
     &    MOD(KFAM/1000,10).NE.0) NM=0  
        ELSEIF(IM.NE.0.AND.MMAT.EQ.46) THEN 
          MSGN=ISIGN(1,K(IM,2)*K(IP,2)) 
          IF(KFAM.GT.100.AND.MOD(KFAM/1000,10).EQ.0) MSGN=  
     &    MSGN*(-1)**MOD(KFAM/100,10)   
        ENDIF   
      ENDIF 
    
C...Kinematics of one-particle decays.  
      IF(ND.EQ.1) THEN  
        DO 290 J=1,4    
  290   P(N+1,J)=P(IP,J)    
        GOTO 510    
      ENDIF 
    
C...Calculate maximum weight ND-particle decay. 
      PV(ND,5)=P(N+ND,5)    
cms .. preinitialize...
      WTMAX=1.
      IF(ND.GE.3) THEN  
        WTMAX=1./WTCOR(ND-2)    
        PMAX=PV(1,5)-PS+P(N+ND,5)   
        PMIN=0. 
        DO 300 IL=ND-1,1,-1 
        PMAX=PMAX+P(N+IL,5) 
        PMIN=PMIN+P(N+IL+1,5)   
  300   WTMAX=WTMAX*PAWT(PMAX,PMIN,P(N+IL,5))   
      ENDIF 
    
C...Find virtual gamma mass in Dalitz decay.    
cms.. preinitialize..
      PMST=0.
      PMES=0.
  310 IF(ND.EQ.2) THEN  
      ELSEIF(MMAT.EQ.2) THEN    
        PMES=4.*PMAS(11,1)**2   
        PMRHO2=PMAS(131,1)**2   
        PGRHO2=PMAS(131,2)**2   
  320   PMST=PMES*(P(IP,5)**2/PMES)**RLU(0) 
        WT=(1+0.5*PMES/PMST)*SQRT(MAX(0.,1.-PMES/PMST))*    
     &  (1.-PMST/P(IP,5)**2)**3*(1.+PGRHO2/PMRHO2)/ 
     &  ((1.-PMST/PMRHO2)**2+PGRHO2/PMRHO2) 
        IF(WT.LT.RLU(0)) GOTO 320   
        PV(2,5)=MAX(2.00001*PMAS(11,1),SQRT(PMST))  
    
C...M-generator gives weight. If rejected, try again.   
      ELSE  
  330   RORD(1)=1.  
        DO 350 IL1=2,ND-1   
        RSAV=RLU(0) 
        DO 340 IL2=IL1-1,1,-1   
        IF(RSAV.LE.RORD(IL2)) GOTO 350  
  340   RORD(IL2+1)=RORD(IL2)   
  350   RORD(IL2+1)=RSAV    
        RORD(ND)=0. 
        WT=1.   
        DO 360 IL=ND-1,1,-1 
        PV(IL,5)=PV(IL+1,5)+P(N+IL,5)+(RORD(IL)-RORD(IL+1))*(PV(1,5)-PS)    
  360   WT=WT*PAWT(PV(IL,5),PV(IL+1,5),P(N+IL,5))   
        IF(WT.LT.RLU(0)*WTMAX) GOTO 330 
      ENDIF 
    
C...Perform two-particle decays in respective CM frame. 
  370 DO 390 IL=1,ND-1  
      PA=PAWT(PV(IL,5),PV(IL+1,5),P(N+IL,5))    
      UE(3)=2.*RLU(0)-1.    
      PHI=PARU(2)*RLU(0)    
      UE(1)=SQRT(1.-UE(3)**2)*COS(PHI)  
      UE(2)=SQRT(1.-UE(3)**2)*SIN(PHI)  
      DO 380 J=1,3  
      P(N+IL,J)=PA*UE(J)    
  380 PV(IL+1,J)=-PA*UE(J)  
      P(N+IL,4)=SQRT(PA**2+P(N+IL,5)**2)    
  390 PV(IL+1,4)=SQRT(PA**2+PV(IL+1,5)**2)  
    
C...Lorentz transform decay products to lab frame.  
      DO 400 J=1,4  
  400 P(N+ND,J)=PV(ND,J)    
      DO 430 IL=ND-1,1,-1   
      DO 410 J=1,3  
  410 BE(J)=PV(IL,J)/PV(IL,4)   
      GA=PV(IL,4)/PV(IL,5)  
      DO 430 I=N+IL,N+ND    
      BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)    
      DO 420 J=1,3  
  420 P(I,J)=P(I,J)+GA*(GA*BEP/(1.+GA)+P(I,4))*BE(J)    
  430 P(I,4)=GA*(P(I,4)+BEP)    
    
C...Matrix elements for omega and phi decays.   
      IF(MMAT.EQ.1) THEN    
        WT=(P(N+1,5)*P(N+2,5)*P(N+3,5))**2-(P(N+1,5)*FOUR(N+2,N+3))**2  
     &  -(P(N+2,5)*FOUR(N+1,N+3))**2-(P(N+3,5)*FOUR(N+1,N+2))**2    
     &  +2.*FOUR(N+1,N+2)*FOUR(N+1,N+3)*FOUR(N+2,N+3)   
        IF(MAX(WT*WTCOR(9)/P(IP,5)**6,0.001).LT.RLU(0)) GOTO 310    
    
C...Matrix elements for pi0 or eta Dalitz decay to gamma e+ e-. 
      ELSEIF(MMAT.EQ.2) THEN    
        FOUR12=FOUR(N+1,N+2)    
        FOUR13=FOUR(N+1,N+3)    
        FOUR23=0.5*PMST-0.25*PMES   
        WT=(PMST-0.5*PMES)*(FOUR12**2+FOUR13**2)+   
     &  PMES*(FOUR12*FOUR13+FOUR12**2+FOUR13**2)    
        IF(WT.LT.RLU(0)*0.25*PMST*(P(IP,5)**2-PMST)**2) GOTO 370    
    
C...Matrix element for S0 -> S1 + V1 -> S1 + S2 + S3 (S scalar, 
C...V vector), of form cos**2(theta02) in V1 rest frame.    
      ELSEIF(MMAT.EQ.3.AND.NM.EQ.2) THEN    
        IF((P(IP,5)**2*FOUR(IM,N+1)-FOUR(IP,IM)*FOUR(IP,N+1))**2.LE.    
     &  RLU(0)*(FOUR(IP,IM)**2-(P(IP,5)*P(IM,5))**2)*(FOUR(IP,N+1)**2-  
     &  (P(IP,5)*P(N+1,5))**2)) GOTO 370    
    
C...Matrix element for "onium" -> g + g + g or gamma + g + g.   
      ELSEIF(MMAT.EQ.4) THEN    
        HX1=2.*FOUR(IP,N+1)/P(IP,5)**2  
        HX2=2.*FOUR(IP,N+2)/P(IP,5)**2  
        HX3=2.*FOUR(IP,N+3)/P(IP,5)**2  
        WT=((1.-HX1)/(HX2*HX3))**2+((1.-HX2)/(HX1*HX3))**2+ 
     &  ((1.-HX3)/(HX1*HX2))**2 
        IF(WT.LT.2.*RLU(0)) GOTO 310    
        IF(K(IP+1,2).EQ.22.AND.(1.-HX1)*P(IP,5)**2.LT.4.*PARJ(32)**2)   
     &  GOTO 310    
    
C...Effective matrix element for nu spectrum in tau -> nu + hadrons.    
      ELSEIF(MMAT.EQ.41) THEN   
        HX1=2.*FOUR(IP,N+1)/P(IP,5)**2  
        IF(8.*HX1*(3.-2.*HX1)/9..LT.RLU(0)) GOTO 310    
    
C...Matrix elements for weak decays (only semileptonic for c and b) 
      ELSEIF(MMAT.GE.42.AND.MMAT.LE.44.AND.ND.EQ.3) THEN    
        IF(MBST.EQ.0) WT=FOUR(IP,N+1)*FOUR(N+2,N+3) 
        IF(MBST.EQ.1) WT=P(IP,5)*P(N+1,4)*FOUR(N+2,N+3) 
        IF(WT.LT.RLU(0)*P(IP,5)*PV(1,5)**3/WTCOR(10)) GOTO 310  
      ELSEIF(MMAT.GE.42.AND.MMAT.LE.44) THEN    
        DO 440 J=1,4    
        P(N+NP+1,J)=0.  
        DO 440 IS=N+3,N+NP  
  440   P(N+NP+1,J)=P(N+NP+1,J)+P(IS,J) 
        IF(MBST.EQ.0) WT=FOUR(IP,N+1)*FOUR(N+2,N+NP+1)  
        IF(MBST.EQ.1) WT=P(IP,5)*P(N+1,4)*FOUR(N+2,N+NP+1)  
        IF(WT.LT.RLU(0)*P(IP,5)*PV(1,5)**3/WTCOR(10)) GOTO 310  
    
C...Angular distribution in W decay.    
      ELSEIF(MMAT.EQ.46.AND.MSGN.NE.0) THEN 
        IF(MSGN.GT.0) WT=FOUR(IM,N+1)*FOUR(N+2,IP+1)    
        IF(MSGN.LT.0) WT=FOUR(IM,N+2)*FOUR(N+1,IP+1)    
        IF(WT.LT.RLU(0)*P(IM,5)**4/WTCOR(10)) GOTO 370  
      ENDIF 
    
C...Scale back energy and reattach spectator.   
      IF(MREM.EQ.1) THEN    
        DO 450 J=1,5    
  450   PV(1,J)=PV(1,J)/(1.-PQT)    
        ND=ND+1 
        MREM=0  
      ENDIF 
    
C...Low invariant mass for system with spectator quark gives particle,  
C...not two jets. Readjust momenta accordingly. 
      IF((MMAT.EQ.31.OR.MMAT.EQ.45).AND.ND.EQ.3) THEN   
        MSTJ(93)=1  
        PM2=ULMASS(K(N+2,2))    
        MSTJ(93)=1  
        PM3=ULMASS(K(N+3,2))    
        IF(P(N+2,5)**2+P(N+3,5)**2+2.*FOUR(N+2,N+3).GE. 
     &  (PARJ(32)+PM2+PM3)**2) GOTO 510 
        K(N+2,1)=1  
        KFTEMP=K(N+2,2) 
        CALL LUKFDI(KFTEMP,K(N+3,2),KFLDMP,K(N+2,2))    
        IF(K(N+2,2).EQ.0) GOTO 150  
        P(N+2,5)=ULMASS(K(N+2,2))   
        PS=P(N+1,5)+P(N+2,5)    
        PV(2,5)=P(N+2,5)    
        MMAT=0  
        ND=2    
        GOTO 370    
      ELSEIF(MMAT.EQ.44) THEN   
        MSTJ(93)=1  
        PM3=ULMASS(K(N+3,2))    
        MSTJ(93)=1  
        PM4=ULMASS(K(N+4,2))    
        IF(P(N+3,5)**2+P(N+4,5)**2+2.*FOUR(N+3,N+4).GE. 
     &  (PARJ(32)+PM3+PM4)**2) GOTO 480 
        K(N+3,1)=1  
        KFTEMP=K(N+3,2) 
        CALL LUKFDI(KFTEMP,K(N+4,2),KFLDMP,K(N+3,2))    
        IF(K(N+3,2).EQ.0) GOTO 150  
        P(N+3,5)=ULMASS(K(N+3,2))   
        DO 460 J=1,3    
  460   P(N+3,J)=P(N+3,J)+P(N+4,J)  
        P(N+3,4)=SQRT(P(N+3,1)**2+P(N+3,2)**2+P(N+3,3)**2+P(N+3,5)**2)  
        HA=P(N+1,4)**2-P(N+2,4)**2  
        HB=HA-(P(N+1,5)**2-P(N+2,5)**2) 
        HC=(P(N+1,1)-P(N+2,1))**2+(P(N+1,2)-P(N+2,2))**2+   
     &  (P(N+1,3)-P(N+2,3))**2  
        HD=(PV(1,4)-P(N+3,4))**2    
        HE=HA**2-2.*HD*(P(N+1,4)**2+P(N+2,4)**2)+HD**2  
        HF=HD*HC-HB**2  
        HG=HD*HC-HA*HB  
        HH=(SQRT(HG**2+HE*HF)-HG)/(2.*HF)   
        DO 470 J=1,3    
        PCOR=HH*(P(N+1,J)-P(N+2,J)) 
        P(N+1,J)=P(N+1,J)+PCOR  
  470   P(N+2,J)=P(N+2,J)-PCOR  
        P(N+1,4)=SQRT(P(N+1,1)**2+P(N+1,2)**2+P(N+1,3)**2+P(N+1,5)**2)  
        P(N+2,4)=SQRT(P(N+2,1)**2+P(N+2,2)**2+P(N+2,3)**2+P(N+2,5)**2)  
        ND=ND-1 
      ENDIF 
    
C...Check invariant mass of W jets. May give one particle or start over.    
  480 IF(MMAT.GE.42.AND.MMAT.LE.44.AND.IABS(K(N+1,2)).LT.10) THEN   
        PMR=SQRT(MAX(0.,P(N+1,5)**2+P(N+2,5)**2+2.*FOUR(N+1,N+2)))  
        MSTJ(93)=1  
        PM1=ULMASS(K(N+1,2))    
        MSTJ(93)=1  
        PM2=ULMASS(K(N+2,2))    
        IF(PMR.GT.PARJ(32)+PM1+PM2) GOTO 490    
        KFLDUM=INT(1.5+RLU(0))  
        CALL LUKFDI(K(N+1,2),-ISIGN(KFLDUM,K(N+1,2)),KFLDMP,KF1)    
        CALL LUKFDI(K(N+2,2),-ISIGN(KFLDUM,K(N+2,2)),KFLDMP,KF2)    
        IF(KF1.EQ.0.OR.KF2.EQ.0) GOTO 150   
        PSM=ULMASS(KF1)+ULMASS(KF2) 
        IF(MMAT.EQ.42.AND.PMR.GT.PARJ(64)+PSM) GOTO 490 
        IF(MMAT.GE.43.AND.PMR.GT.0.2*PARJ(32)+PSM) GOTO 490 
        IF(ND.EQ.4.OR.KFA.EQ.15) GOTO 150   
        K(N+1,1)=1  
        KFTEMP=K(N+1,2) 
        CALL LUKFDI(KFTEMP,K(N+2,2),KFLDMP,K(N+1,2))    
        IF(K(N+1,2).EQ.0) GOTO 150  
        P(N+1,5)=ULMASS(K(N+1,2))   
        K(N+2,2)=K(N+3,2)   
        P(N+2,5)=P(N+3,5)   
        PS=P(N+1,5)+P(N+2,5)    
        PV(2,5)=P(N+3,5)    
        MMAT=0  
        ND=2    
        GOTO 370    
      ENDIF 
    
C...Phase space decay of partons from W decay. 
cms.. preinitialize - should never get called - for compiler only
      PMR=0.
  490 IF(MMAT.EQ.42.AND.IABS(K(N+1,2)).LT.10) THEN  
        KFLO(1)=K(N+1,2)    
        KFLO(2)=K(N+2,2)    
        K(N+1,1)=K(N+3,1)   
        K(N+1,2)=K(N+3,2)   
        DO 500 J=1,5    
        PV(1,J)=P(N+1,J)+P(N+2,J)   
  500   P(N+1,J)=P(N+3,J)   
        PV(1,5)=PMR 
        N=N+1   
        NP=0    
        NQ=2    
        PS=0.   
        MSTJ(93)=2  
        PSQ=ULMASS(KFLO(1)) 
        MSTJ(93)=2  
        PSQ=PSQ+ULMASS(KFLO(2)) 
        MMAT=11 
        GOTO 180    
      ENDIF 
    
C...Boost back for rapidly moving particle. 
  510 N=N+ND    
      IF(MBST.EQ.1) THEN    
        DO 520 J=1,3    
  520   BE(J)=P(IP,J)/P(IP,4)   
        GA=P(IP,4)/P(IP,5)  
        DO 540 I=NSAV+1,N   
        BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)  
        DO 530 J=1,3    
  530   P(I,J)=P(I,J)+GA*(GA*BEP/(1.+GA)+P(I,4))*BE(J)  
  540   P(I,4)=GA*(P(I,4)+BEP)  
      ENDIF 
    
C...Fill in position of decay vertex.   
      DO 560 I=NSAV+1,N 
      DO 550 J=1,4  
  550 V(I,J)=VDCY(J)    
  560 V(I,5)=0. 
    
C...Set up for parton shower evolution from jets.   
      IF(MSTJ(23).GE.1.AND.MMAT.EQ.4.AND.K(NSAV+1,2).EQ.21) THEN    
        K(NSAV+1,1)=3   
        K(NSAV+2,1)=3   
        K(NSAV+3,1)=3   
        K(NSAV+1,4)=MSTU(5)*(NSAV+2)    
        K(NSAV+1,5)=MSTU(5)*(NSAV+3)    
        K(NSAV+2,4)=MSTU(5)*(NSAV+3)    
        K(NSAV+2,5)=MSTU(5)*(NSAV+1)    
        K(NSAV+3,4)=MSTU(5)*(NSAV+1)    
        K(NSAV+3,5)=MSTU(5)*(NSAV+2)    
        MSTJ(92)=-(NSAV+1)  
      ELSEIF(MSTJ(23).GE.1.AND.MMAT.EQ.4) THEN  
        K(NSAV+2,1)=3   
        K(NSAV+3,1)=3   
        K(NSAV+2,4)=MSTU(5)*(NSAV+3)    
        K(NSAV+2,5)=MSTU(5)*(NSAV+3)    
        K(NSAV+3,4)=MSTU(5)*(NSAV+2)    
        K(NSAV+3,5)=MSTU(5)*(NSAV+2)    
        MSTJ(92)=NSAV+2 
      ELSEIF(MSTJ(23).GE.1.AND.(MMAT.EQ.32.OR.MMAT.EQ.44.OR.MMAT.EQ.46).    
     &AND.IABS(K(NSAV+1,2)).LE.10.AND.IABS(K(NSAV+2,2)).LE.10) THEN 
        K(NSAV+1,1)=3   
        K(NSAV+2,1)=3   
        K(NSAV+1,4)=MSTU(5)*(NSAV+2)    
        K(NSAV+1,5)=MSTU(5)*(NSAV+2)    
        K(NSAV+2,4)=MSTU(5)*(NSAV+1)    
        K(NSAV+2,5)=MSTU(5)*(NSAV+1)    
        MSTJ(92)=NSAV+1 
      ELSEIF(MSTJ(23).GE.1.AND.MMAT.EQ.33.AND.IABS(K(NSAV+2,2)).EQ.21)  
     &THEN  
        K(NSAV+1,1)=3   
        K(NSAV+2,1)=3   
        K(NSAV+3,1)=3   
        KCP=LUCOMP(K(NSAV+1,2)) 
        KQP=KCHG(KCP,2)*ISIGN(1,K(NSAV+1,2))    
        JCON=4  
        IF(KQP.LT.0) JCON=5 
        K(NSAV+1,JCON)=MSTU(5)*(NSAV+2) 
        K(NSAV+2,9-JCON)=MSTU(5)*(NSAV+1)   
        K(NSAV+2,JCON)=MSTU(5)*(NSAV+3) 
        K(NSAV+3,9-JCON)=MSTU(5)*(NSAV+2)   
        MSTJ(92)=NSAV+1 
      ELSEIF(MSTJ(23).GE.1.AND.MMAT.EQ.33) THEN 
        K(NSAV+1,1)=3   
        K(NSAV+3,1)=3   
        K(NSAV+1,4)=MSTU(5)*(NSAV+3)    
        K(NSAV+1,5)=MSTU(5)*(NSAV+3)    
        K(NSAV+3,4)=MSTU(5)*(NSAV+1)    
        K(NSAV+3,5)=MSTU(5)*(NSAV+1)    
        MSTJ(92)=NSAV+1 
      ENDIF 
    
C...Mark decayed particle.  
      IF(K(IP,1).EQ.5) K(IP,1)=15   
      IF(K(IP,1).LE.10) K(IP,1)=11  
      K(IP,4)=NSAV+1    
      K(IP,5)=N 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUKFDI(KFL1,KFL2,KFL3,KF)  
    
C...Purpose: to generate a new flavour pair and combine off a hadron.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
    
C...Default flavour values. Input consistency checks.   
      KF1A=IABS(KFL1)   
      KF2A=IABS(KFL2)   
      KFL3=0    
      KF=0  
      IF(KF1A.EQ.0) RETURN  
      IF(KF2A.NE.0) THEN    
        IF(KF1A.LE.10.AND.KF2A.LE.10.AND.KFL1*KFL2.GT.0) RETURN 
        IF(KF1A.GT.10.AND.KF2A.GT.10) RETURN    
        IF((KF1A.GT.10.OR.KF2A.GT.10).AND.KFL1*KFL2.LT.0) RETURN    
      ENDIF 
    
C...Check if tabulated flavour probabilities are to be used.    
      IF(MSTJ(15).EQ.1) THEN    
        KTAB1=-1    
        IF(KF1A.GE.1.AND.KF1A.LE.6) KTAB1=KF1A  
        KFL1A=MOD(KF1A/1000,10) 
        KFL1B=MOD(KF1A/100,10)  
        KFL1S=MOD(KF1A,10)  
        IF(KFL1A.GE.1.AND.KFL1A.LE.4.AND.KFL1B.GE.1.AND.KFL1B.LE.4) 
     &  KTAB1=6+KFL1A*(KFL1A-2)+2*KFL1B+(KFL1S-1)/2 
        IF(KFL1A.GE.1.AND.KFL1A.LE.4.AND.KFL1A.EQ.KFL1B) KTAB1=KTAB1-1  
        IF(KF1A.GE.1.AND.KF1A.LE.6) KFL1A=KF1A  
        KTAB2=0 
        IF(KF2A.NE.0) THEN  
          KTAB2=-1  
          IF(KF2A.GE.1.AND.KF2A.LE.6) KTAB2=KF2A    
          KFL2A=MOD(KF2A/1000,10)   
          KFL2B=MOD(KF2A/100,10)    
          KFL2S=MOD(KF2A,10)    
          IF(KFL2A.GE.1.AND.KFL2A.LE.4.AND.KFL2B.GE.1.AND.KFL2B.LE.4)   
     &    KTAB2=6+KFL2A*(KFL2A-2)+2*KFL2B+(KFL2S-1)/2   
          IF(KFL2A.GE.1.AND.KFL2A.LE.4.AND.KFL2A.EQ.KFL2B) KTAB2=KTAB2-1    
        ENDIF   
        IF(KTAB1.GE.0.AND.KTAB2.GE.0) GOTO 140  
      ENDIF 
    
C...Parameters and breaking diquark parameter combinations. 
  100 PAR2=PARJ(2)  
      PAR3=PARJ(3)  
      PAR4=3.*PARJ(4)   
cms.. preinitialize to avoid compiler warning
      PARSM=0.
      PARS2=0.
      PARDM=0.
      PAR4M=0.
      PAR3M=0.
      PARS0=0.
      PARS1=0.
      IF(MSTJ(12).GE.2) THEN    
        PAR3M=SQRT(PARJ(3)) 
        PAR4M=1./(3.*SQRT(PARJ(4))) 
        PARDM=PARJ(7)/(PARJ(7)+PAR3M*PARJ(6))   
        PARS0=PARJ(5)*(2.+(1.+PAR2*PAR3M*PARJ(7))*(1.+PAR4M))   
        PARS1=PARJ(7)*PARS0/(2.*PAR3M)+PARJ(5)*(PARJ(6)*(1.+PAR4M)+ 
     &  PAR2*PAR3M*PARJ(6)*PARJ(7)) 
        PARS2=PARJ(5)*2.*PARJ(6)*PARJ(7)*(PAR2*PARJ(7)+(1.+PAR4M)/PAR3M)    
        PARSM=MAX(PARS0,PARS1,PARS2)    
        PAR4=PAR4*(1.+PARSM)/(1.+PARSM/(3.*PAR4M))  
      ENDIF 
    
C...Choice of whether to generate meson or baryon.  
      MBARY=0   
      KFDA=0    
      IF(KF1A.LE.10) THEN   
        IF(KF2A.EQ.0.AND.MSTJ(12).GE.1.AND.(1.+PARJ(1))*RLU(0).GT.1.)   
     &  MBARY=1 
        IF(KF2A.GT.10) MBARY=2  
        IF(KF2A.GT.10.AND.KF2A.LE.10000) KFDA=KF2A  
      ELSE  
        MBARY=2 
        IF(KF1A.LE.10000) KFDA=KF1A 
      ENDIF 
    
C...Possibility of process diquark -> meson + new diquark.  
      IF(KFDA.NE.0.AND.MSTJ(12).GE.2) THEN  
        KFLDA=MOD(KFDA/1000,10) 
        KFLDB=MOD(KFDA/100,10)  
        KFLDS=MOD(KFDA,10)  
        WTDQ=PARS0  
        IF(MAX(KFLDA,KFLDB).EQ.3) WTDQ=PARS1    
        IF(MIN(KFLDA,KFLDB).EQ.3) WTDQ=PARS2    
        IF(KFLDS.EQ.1) WTDQ=WTDQ/(3.*PAR4M) 
        IF((1.+WTDQ)*RLU(0).GT.1.) MBARY=-1 
        IF(MBARY.EQ.-1.AND.KF2A.NE.0) RETURN    
      ENDIF 
    
C...Flavour for meson, possibly with new flavour.   
      IF(MBARY.LE.0) THEN   
        KFS=ISIGN(1,KFL1)   
        IF(MBARY.EQ.0) THEN 
          IF(KF2A.EQ.0) KFL3=ISIGN(1+INT((2.+PAR2)*RLU(0)),-KFL1)   
          KFLA=MAX(KF1A,KF2A+IABS(KFL3))    
          KFLB=MIN(KF1A,KF2A+IABS(KFL3))    
          IF(KFLA.NE.KF1A) KFS=-KFS 
    
C...Splitting of diquark into meson plus new diquark.   
        ELSE    
          KFL1A=MOD(KF1A/1000,10)   
          KFL1B=MOD(KF1A/100,10)    
  110     KFL1D=KFL1A+INT(RLU(0)+0.5)*(KFL1B-KFL1A) 
          KFL1E=KFL1A+KFL1B-KFL1D   
          IF((KFL1D.EQ.3.AND.RLU(0).GT.PARDM).OR.(KFL1E.EQ.3.AND.   
     &    RLU(0).LT.PARDM)) THEN    
            KFL1D=KFL1A+KFL1B-KFL1D 
            KFL1E=KFL1A+KFL1B-KFL1E 
          ENDIF 
          KFL3A=1+INT((2.+PAR2*PAR3M*PARJ(7))*RLU(0))   
          IF((KFL1E.NE.KFL3A.AND.RLU(0).GT.(1.+PAR4M)/MAX(2.,1.+PAR4M)).    
     &    OR.(KFL1E.EQ.KFL3A.AND.RLU(0).GT.2./MAX(2.,1.+PAR4M)))    
     &    GOTO 110  
          KFLDS=3   
          IF(KFL1E.NE.KFL3A) KFLDS=2*INT(RLU(0)+1./(1.+PAR4M))+1    
          KFL3=ISIGN(10000+1000*MAX(KFL1E,KFL3A)+100*MIN(KFL1E,KFL3A)+  
     &    KFLDS,-KFL1)  
          KFLA=MAX(KFL1D,KFL3A) 
          KFLB=MIN(KFL1D,KFL3A) 
          IF(KFLA.NE.KFL1D) KFS=-KFS    
        ENDIF   
    
C...Form meson, with spin and flavour mixing for diagonal states.   
        KMUL=0
        IF(KFLA.LE.2) KMUL=INT(PARJ(11)+RLU(0)) 
        IF(KFLA.EQ.3) KMUL=INT(PARJ(12)+RLU(0)) 
        IF(KFLA.GE.4) KMUL=INT(PARJ(13)+RLU(0)) 
        IF(KMUL.EQ.0.AND.PARJ(14).GT.0.) THEN   
          IF(RLU(0).LT.PARJ(14)) KMUL=2 
        ELSEIF(KMUL.EQ.1.AND.PARJ(15)+PARJ(16)+PARJ(17).GT.0.) THEN 
          RMUL=RLU(0)   
          IF(RMUL.LT.PARJ(15)) KMUL=3   
          IF(KMUL.EQ.1.AND.RMUL.LT.PARJ(15)+PARJ(16)) KMUL=4    
          IF(KMUL.EQ.1.AND.RMUL.LT.PARJ(15)+PARJ(16)+PARJ(17)) KMUL=5   
        ENDIF   
        KFLS=3  
        IF(KMUL.EQ.0.OR.KMUL.EQ.3) KFLS=1   
        IF(KMUL.EQ.5) KFLS=5    
        IF(KFLA.NE.KFLB) THEN   
          KF=(100*KFLA+10*KFLB+KFLS)*KFS*(-1)**KFLA 
        ELSE    
          RMIX=RLU(0)   
          IMIX=2*KFLA+10*KMUL   
          IF(KFLA.LE.3) KF=110*(1+INT(RMIX+PARF(IMIX-1))+   
     &    INT(RMIX+PARF(IMIX)))+KFLS    
          IF(KFLA.GE.4) KF=110*KFLA+KFLS    
        ENDIF   
        IF(KMUL.EQ.2.OR.KMUL.EQ.3) KF=KF+ISIGN(10000,KF)    
        IF(KMUL.EQ.4) KF=KF+ISIGN(20000,KF) 
    
C...Generate diquark flavour.   
      ELSE  
  120   IF(KF1A.LE.10.AND.KF2A.EQ.0) THEN   
          KFLA=KF1A 
  130     KFLB=1+INT((2.+PAR2*PAR3)*RLU(0)) 
          KFLC=1+INT((2.+PAR2*PAR3)*RLU(0)) 
          KFLDS=1   
          IF(KFLB.GE.KFLC) KFLDS=3  
          IF(KFLDS.EQ.1.AND.PAR4*RLU(0).GT.1.) GOTO 130 
          IF(KFLDS.EQ.3.AND.PAR4.LT.RLU(0)) GOTO 130    
          KFL3=ISIGN(1000*MAX(KFLB,KFLC)+100*MIN(KFLB,KFLC)+KFLDS,KFL1) 
    
C...Take diquark flavour from input.    
        ELSEIF(KF1A.LE.10) THEN 
          KFLA=KF1A 
          KFLB=MOD(KF2A/1000,10)    
          KFLC=MOD(KF2A/100,10) 
          KFLDS=MOD(KF2A,10)    
    
C...Generate (or take from input) quark to go with diquark. 
        ELSE    
          IF(KF2A.EQ.0) KFL3=ISIGN(1+INT((2.+PAR2)*RLU(0)),KFL1)    
          KFLA=KF2A+IABS(KFL3)  
          KFLB=MOD(KF1A/1000,10)    
          KFLC=MOD(KF1A/100,10) 
          KFLDS=MOD(KF1A,10)    
        ENDIF   
    
C...SU(6) factors for formation of baryon. Try again if fails.  
        KBARY=KFLDS 
        IF(KFLDS.EQ.3.AND.KFLB.NE.KFLC) KBARY=5 
        IF(KFLA.NE.KFLB.AND.KFLA.NE.KFLC) KBARY=KBARY+1 
        WT=PARF(60+KBARY)+PARJ(18)*PARF(70+KBARY)   
        IF(MBARY.EQ.1.AND.MSTJ(12).GE.2) THEN   
          WTDQ=PARS0    
          IF(MAX(KFLB,KFLC).EQ.3) WTDQ=PARS1    
          IF(MIN(KFLB,KFLC).EQ.3) WTDQ=PARS2    
          IF(KFLDS.EQ.1) WTDQ=WTDQ/(3.*PAR4M)   
          IF(KFLDS.EQ.1) WT=WT*(1.+WTDQ)/(1.+PARSM/(3.*PAR4M))  
          IF(KFLDS.EQ.3) WT=WT*(1.+WTDQ)/(1.+PARSM) 
        ENDIF   
        IF(KF2A.EQ.0.AND.WT.LT.RLU(0)) GOTO 120 
    
C...Form baryon. Distinguish Lambda- and Sigmalike baryons. 
        KFLD=MAX(KFLA,KFLB,KFLC)    
        KFLF=MIN(KFLA,KFLB,KFLC)    
        KFLE=KFLA+KFLB+KFLC-KFLD-KFLF   
        KFLS=2  
        IF((PARF(60+KBARY)+PARJ(18)*PARF(70+KBARY))*RLU(0).GT.  
     &  PARF(60+KBARY)) KFLS=4  
        KFLL=0  
        IF(KFLS.EQ.2.AND.KFLD.GT.KFLE.AND.KFLE.GT.KFLF) THEN    
          IF(KFLDS.EQ.1.AND.KFLA.EQ.KFLD) KFLL=1    
          IF(KFLDS.EQ.1.AND.KFLA.NE.KFLD) KFLL=INT(0.25+RLU(0)) 
          IF(KFLDS.EQ.3.AND.KFLA.NE.KFLD) KFLL=INT(0.75+RLU(0)) 
        ENDIF   
        IF(KFLL.EQ.0) KF=ISIGN(1000*KFLD+100*KFLE+10*KFLF+KFLS,KFL1)    
        IF(KFLL.EQ.1) KF=ISIGN(1000*KFLD+100*KFLF+10*KFLE+KFLS,KFL1)    
      ENDIF 
      RETURN    
    
C...Use tabulated probabilities to select new flavour and hadron.   
  140 IF(KTAB2.EQ.0.AND.MSTJ(12).LE.0) THEN 
        KT3L=1  
        KT3U=6  
      ELSEIF(KTAB2.EQ.0.AND.KTAB1.GE.7.AND.MSTJ(12).LE.1) THEN  
        KT3L=1  
        KT3U=6  
      ELSEIF(KTAB2.EQ.0) THEN   
        KT3L=1  
        KT3U=22 
      ELSE  
        KT3L=KTAB2  
        KT3U=KTAB2  
      ENDIF 
      RFL=0.    
      DO 150 KTS=0,2    
      DO 150 KT3=KT3L,KT3U  
      RFL=RFL+PARF(120+80*KTAB1+25*KTS+KT3) 
  150 CONTINUE  
cms.. preinitialize to avoid compiler warning
      KTAB3=0.
      RFL=RLU(0)*RFL    
      DO 160 KTS=0,2    
      KTABS=KTS 
      DO 160 KT3=KT3L,KT3U  
      KTAB3=KT3 
      RFL=RFL-PARF(120+80*KTAB1+25*KTS+KT3) 
  160 IF(RFL.LE.0.) GOTO 170    
  170 CONTINUE  
    
C...Reconstruct flavour of produced quark/diquark.  
      IF(KTAB3.LE.6) THEN   
        KFL3A=KTAB3 
        KFL3B=0 
        KFL3=ISIGN(KFL3A,KFL1*(2*KTAB1-13)) 
      ELSE  
        KFL3A=1 
        IF(KTAB3.GE.8) KFL3A=2  
        IF(KTAB3.GE.11) KFL3A=3 
        IF(KTAB3.GE.16) KFL3A=4 
        KFL3B=(KTAB3-6-KFL3A*(KFL3A-2))/2   
        KFL3=1000*KFL3A+100*KFL3B+1 
        IF(KFL3A.EQ.KFL3B.OR.KTAB3.NE.6+KFL3A*(KFL3A-2)+2*KFL3B) KFL3=  
     &  KFL3+2  
        KFL3=ISIGN(KFL3,KFL1*(13-2*KTAB1))  
      ENDIF 
    
C...Reconstruct meson code. 
      IF(KFL3A.EQ.KFL1A.AND.KFL3B.EQ.KFL1B.AND.(KFL3A.LE.3.OR.  
     &KFL3B.NE.0)) THEN 
        RFL=RLU(0)*(PARF(143+80*KTAB1+25*KTABS)+PARF(144+80*KTAB1+  
     &  25*KTABS)+PARF(145+80*KTAB1+25*KTABS))  
        KF=110+2*KTABS+1    
        IF(RFL.GT.PARF(143+80*KTAB1+25*KTABS)) KF=220+2*KTABS+1 
        IF(RFL.GT.PARF(143+80*KTAB1+25*KTABS)+PARF(144+80*KTAB1+    
     &  25*KTABS)) KF=330+2*KTABS+1 
      ELSEIF(KTAB1.LE.6.AND.KTAB3.LE.6) THEN    
        KFLA=MAX(KTAB1,KTAB3)   
        KFLB=MIN(KTAB1,KTAB3)   
        KFS=ISIGN(1,KFL1)   
        IF(KFLA.NE.KF1A) KFS=-KFS   
        KF=(100*KFLA+10*KFLB+2*KTABS+1)*KFS*(-1)**KFLA  
      ELSEIF(KTAB1.GE.7.AND.KTAB3.GE.7) THEN    
        KFS=ISIGN(1,KFL1)   
        IF(KFL1A.EQ.KFL3A) THEN 
          KFLA=MAX(KFL1B,KFL3B) 
          KFLB=MIN(KFL1B,KFL3B) 
          IF(KFLA.NE.KFL1B) KFS=-KFS    
        ELSEIF(KFL1A.EQ.KFL3B) THEN 
          KFLA=KFL3A    
          KFLB=KFL1B    
          KFS=-KFS  
        ELSEIF(KFL1B.EQ.KFL3A) THEN 
          KFLA=KFL1A    
          KFLB=KFL3B    
        ELSEIF(KFL1B.EQ.KFL3B) THEN 
          KFLA=MAX(KFL1A,KFL3A) 
          KFLB=MIN(KFL1A,KFL3A) 
          IF(KFLA.NE.KFL1A) KFS=-KFS    
        ELSE    
          CALL LUERRM(2,'(LUKFDI:) no matching flavours for qq -> qq')  
          GOTO 100  
        ENDIF   
        KF=(100*KFLA+10*KFLB+2*KTABS+1)*KFS*(-1)**KFLA  
    
C...Reconstruct baryon code.    
      ELSE  
        IF(KTAB1.GE.7) THEN 
          KFLA=KFL3A    
          KFLB=KFL1A    
          KFLC=KFL1B    
        ELSE    
          KFLA=KFL1A    
          KFLB=KFL3A    
          KFLC=KFL3B    
        ENDIF   
        KFLD=MAX(KFLA,KFLB,KFLC)    
        KFLF=MIN(KFLA,KFLB,KFLC)    
        KFLE=KFLA+KFLB+KFLC-KFLD-KFLF   
        IF(KTABS.EQ.0) KF=ISIGN(1000*KFLD+100*KFLF+10*KFLE+2,KFL1)  
        IF(KTABS.GE.1) KF=ISIGN(1000*KFLD+100*KFLE+10*KFLF+2*KTABS,KFL1)    
      ENDIF 
    
C...Check that constructed flavour code is an allowed one.  
      IF(KFL2.NE.0) KFL3=0  
      KC=LUCOMP(KF) 
      IF(KC.EQ.0) THEN  
        CALL LUERRM(2,'(LUKFDI:) user-defined flavour probabilities '// 
     &  'failed')   
        GOTO 100    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUPTDI(KFL,PX,PY)  
    
C...Purpose: to generate transverse momentum according to a Gaussian.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
    
C...Generate p_T and azimuthal angle, gives p_x and p_y.    
      KFLA=IABS(KFL)    
      PT=PARJ(21)*SQRT(-LOG(MAX(1E-10,RLU(0)))) 
      IF(MSTJ(91).EQ.1) PT=PARJ(22)*PT  
      IF(KFLA.EQ.0.AND.MSTJ(13).LE.0) PT=0. 
      PHI=PARU(2)*RLU(0)    
      PX=PT*COS(PHI)    
      PY=PT*SIN(PHI)    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUZDIS(KFL1,KFL2,PR,Z) 
    
C...Purpose: to generate the longitudinal splitting variable z. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
    
C...Check if heavy flavour fragmentation.   
      KFLA=IABS(KFL1)   
      KFLB=IABS(KFL2)   
      KFLH=KFLA 
      IF(KFLA.GE.10) KFLH=MOD(KFLA/1000,10) 
    
C...Lund symmetric scaling function: determine parameters of shape. 
      IF(MSTJ(11).EQ.1.OR.(MSTJ(11).EQ.3.AND.KFLH.LE.3)) THEN   
        FA=PARJ(41) 
        IF(MSTJ(91).EQ.1) FA=PARJ(43)   
        IF(KFLB.GE.10) FA=FA+PARJ(45)   
        FB=PARJ(42)*PR  
        IF(MSTJ(91).EQ.1) FB=PARJ(44)*PR    
        FC=1.   
        IF(KFLA.GE.10) FC=FC-PARJ(45)   
        IF(KFLB.GE.10) FC=FC+PARJ(45)   
        MC=1    
        IF(ABS(FC-1.).GT.0.01) MC=2 
    
C...Determine position of maximum. Special cases for a = 0 or a = c.    
        IF(FA.LT.0.02) THEN 
          MA=1  
          ZMAX=1.   
          IF(FC.GT.FB) ZMAX=FB/FC   
        ELSEIF(ABS(FC-FA).LT.0.01) THEN 
          MA=2  
          ZMAX=FB/(FB+FC)   
        ELSE    
          MA=3  
          ZMAX=0.5*(FB+FC-SQRT((FB-FC)**2+4.*FA*FB))/(FC-FA)    
          IF(ZMAX.GT.0.99.AND.FB.GT.100.) ZMAX=1.-FA/FB 
        ENDIF   
    
C...Subdivide z range if distribution very peaked near endpoint.    
        MMAX=2
cms .. redefine variables to avoid compiler warning
        ZDIV=0.
        ZDIVC=0.
        FINT=0.
        IF(ZMAX.LT.0.1) THEN    
          MMAX=1    
          ZDIV=2.75*ZMAX    
          IF(MC.EQ.1) THEN  
            FINT=1.-LOG(ZDIV)   
          ELSE  
            ZDIVC=ZDIV**(1.-FC) 
            FINT=1.+(1.-1./ZDIVC)/(FC-1.)   
          ENDIF 
        ELSEIF(ZMAX.GT.0.85.AND.FB.GT.1.) THEN  
          MMAX=3    
          FSCB=SQRT(4.+(FC/FB)**2)  
          ZDIV=FSCB-1./ZMAX-(FC/FB)*LOG(ZMAX*0.5*(FSCB+FC/FB))  
          IF(MA.GE.2) ZDIV=ZDIV+(FA/FB)*LOG(1.-ZMAX)    
          ZDIV=MIN(ZMAX,MAX(0.,ZDIV))   
          FINT=1.+FB*(1.-ZDIV)  
        ENDIF   
    
C...Choice of z, preweighted for peaks at low or high z.    
  100   Z=RLU(0)    
        FPRE=1. 
        IF(MMAX.EQ.1) THEN  
          IF(FINT*RLU(0).LE.1.) THEN    
            Z=ZDIV*Z    
          ELSEIF(MC.EQ.1) THEN  
            Z=ZDIV**Z   
            FPRE=ZDIV/Z 
          ELSE  
            Z=1./(ZDIVC+Z*(1.-ZDIVC))**(1./(1.-FC)) 
            FPRE=(ZDIV/Z)**FC   
          ENDIF 
        ELSEIF(MMAX.EQ.3) THEN  
          IF(FINT*RLU(0).LE.1.) THEN    
            Z=ZDIV+LOG(Z)/FB    
            FPRE=EXP(FB*(Z-ZDIV))   
          ELSE  
            Z=ZDIV+Z*(1.-ZDIV)  
          ENDIF 
        ENDIF   
    
C...Weighting according to correct formula. 
        IF(Z.LE.FB/(50.+FB).OR.Z.GE.1.) GOTO 100    
        FVAL=(ZMAX/Z)**FC*EXP(FB*(1./ZMAX-1./Z))    
        IF(MA.GE.2) FVAL=((1.-Z)/(1.-ZMAX))**FA*FVAL    
        IF(FVAL.LT.RLU(0)*FPRE) GOTO 100    
    
C...Generate z according to Field-Feynman, SLAC, (1-z)**c OR z**c.  
      ELSE  
        FC=PARJ(50+MAX(1,KFLH)) 
        IF(MSTJ(91).EQ.1) FC=PARJ(59)   
  110   Z=RLU(0)    
        IF(FC.GE.0..AND.FC.LE.1.) THEN  
          IF(FC.GT.RLU(0)) Z=1.-Z**(1./3.)  
        ELSEIF(FC.GT.-1.) THEN  
          IF(-4.*FC*Z*(1.-Z)**2.LT.RLU(0)*((1.-Z)**2-FC*Z)**2) GOTO 110 
        ELSE    
          IF(FC.GT.0.) Z=1.-Z**(1./FC)  
          IF(FC.LT.0.) Z=Z**(-1./FC)    
        ENDIF   
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUSHOW(IP1,IP2,QMAX)   
    
C...Purpose: to generate timelike parton showers from given partons.    
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      DIMENSION PMTH(5,40),PS(5),PMA(4),PMSD(4),IEP(4),IPA(4),  
     &KFLA(4),KFLD(4),KFL(4),ITRY(4),ISI(4),ISL(4),DP(4),DPT(5,4)   
    
C...Initialization of cutoff masses etc.    
      IF(MSTJ(41).LE.0.OR.(MSTJ(41).EQ.1.AND.QMAX.LE.PARJ(82)).OR.  
     &QMAX.LE.MIN(PARJ(82),PARJ(83)).OR.MSTJ(41).GE.3) RETURN   
      PMTH(1,21)=ULMASS(21) 
      PMTH(2,21)=SQRT(PMTH(1,21)**2+0.25*PARJ(82)**2)   
      PMTH(3,21)=2.*PMTH(2,21)  
      PMTH(4,21)=PMTH(3,21) 
      PMTH(5,21)=PMTH(3,21) 
      PMTH(1,22)=ULMASS(22) 
      PMTH(2,22)=SQRT(PMTH(1,22)**2+0.25*PARJ(83)**2)   
      PMTH(3,22)=2.*PMTH(2,22)  
      PMTH(4,22)=PMTH(3,22) 
      PMTH(5,22)=PMTH(3,22) 
      PMQTH1=PARJ(82)   
      IF(MSTJ(41).EQ.2) PMQTH1=MIN(PARJ(82),PARJ(83))   
      PMQTH2=PMTH(2,21) 
      IF(MSTJ(41).EQ.2) PMQTH2=MIN(PMTH(2,21),PMTH(2,22))   
      DO 100 IF=1,8 
      PMTH(1,IF)=ULMASS(IF) 
      PMTH(2,IF)=SQRT(PMTH(1,IF)**2+0.25*PMQTH1**2) 
      PMTH(3,IF)=PMTH(2,IF)+PMQTH2  
      PMTH(4,IF)=SQRT(PMTH(1,IF)**2+0.25*PARJ(82)**2)+PMTH(2,21)    
  100 PMTH(5,IF)=SQRT(PMTH(1,IF)**2+0.25*PARJ(83)**2)+PMTH(2,22)    
      PT2MIN=MAX(0.5*PARJ(82),1.1*PARJ(81))**2  
      ALAMS=PARJ(81)**2 
      ALFM=LOG(PT2MIN/ALAMS)    
    
C...Store positions of shower initiating partons.   
      M3JC=0    
cms..pre-initialization
      NPA=0
cms..pre-initialization
      ZM=0.
      IF(IP1.GT.0.AND.IP1.LE.MIN(N,MSTU(4)-MSTU(32)).AND.IP2.EQ.0) THEN 
        NPA=1   
        IPA(1)=IP1  
      ELSEIF(MIN(IP1,IP2).GT.0.AND.MAX(IP1,IP2).LE.MIN(N,MSTU(4)-   
     &MSTU(32))) THEN   
        NPA=2   
        IPA(1)=IP1  
        IPA(2)=IP2  
      ELSEIF(IP1.GT.0.AND.IP1.LE.MIN(N,MSTU(4)-MSTU(32)).AND.IP2.LT.0.  
     &AND.IP2.GE.-3) THEN   
        NPA=IABS(IP2)   
        DO 110 I=1,NPA  
  110   IPA(I)=IP1+I-1  
      ELSE  
        CALL LUERRM(12, 
     &  '(LUSHOW:) failed to reconstruct showering system') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
    
C...Check on phase space available for emission.    
      IREJ=0    
      DO 120 J=1,5  
  120 PS(J)=0.  
      PM=0. 
      DO 130 I=1,NPA    
      KFLA(I)=IABS(K(IPA(I),2)) 
      PMA(I)=P(IPA(I),5)    
      IF(KFLA(I).NE.0.AND.(KFLA(I).LE.8.OR.KFLA(I).EQ.21))  
     &PMA(I)=PMTH(3,KFLA(I))    
      PM=PM+PMA(I)  
      IF(KFLA(I).EQ.0.OR.(KFLA(I).GT.8.AND.KFLA(I).NE.21).OR.   
     &PMA(I).GT.QMAX) IREJ=IREJ+1   
      DO 130 J=1,4  
  130 PS(J)=PS(J)+P(IPA(I),J)   
      IF(IREJ.EQ.NPA) RETURN    
      PS(5)=SQRT(MAX(0.,PS(4)**2-PS(1)**2-PS(2)**2-PS(3)**2))   
      IF(NPA.EQ.1) PS(5)=PS(4)  
      IF(PS(5).LE.PM+PMQTH1) RETURN 
      IF(NPA.EQ.2.AND.MSTJ(47).GE.1) THEN   
        IF(KFLA(1).GE.1.AND.KFLA(1).LE.8.AND.KFLA(2).GE.1.AND.  
     &  KFLA(2).LE.8) M3JC=1    
        IF(MSTJ(47).GE.2) M3JC=1    
      ENDIF 
    
C...Define imagined single initiator of shower for parton system.   
      NS=N  
      IF(N.GT.MSTU(4)-MSTU(32)-5) THEN  
        CALL LUERRM(11,'(LUSHOW:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      IF(NPA.GE.2) THEN 
        K(N+1,1)=11 
        K(N+1,2)=21 
        K(N+1,3)=0  
        K(N+1,4)=0  
        K(N+1,5)=0  
        P(N+1,1)=0. 
        P(N+1,2)=0. 
        P(N+1,3)=0. 
        P(N+1,4)=PS(5)  
        P(N+1,5)=PS(5)  
        V(N+1,5)=PS(5)**2   
        N=N+1   
      ENDIF 
    
C...Loop over partons that may branch.  
      NEP=NPA   
      IM=NS 
      IF(NPA.EQ.1) IM=NS-1  
  140 IM=IM+1   
      IF(N.GT.NS) THEN  
        IF(IM.GT.N) GOTO 380    
        KFLM=IABS(K(IM,2))  
        IF(KFLM.EQ.0.OR.(KFLM.GT.8.AND.KFLM.NE.21)) GOTO 140    
        IF(P(IM,5).LT.PMTH(2,KFLM)) GOTO 140    
        IGM=K(IM,3) 
      ELSE  
        IGM=-1  
      ENDIF 
      IF(N+NEP.GT.MSTU(4)-MSTU(32)-5) THEN  
        CALL LUERRM(11,'(LUSHOW:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
    
C...Position of aunt (sister to branching parton).  
C...Origin and flavour of daughters.    
      IAU=0 
      IF(IGM.GT.0) THEN 
        IF(K(IM-1,3).EQ.IGM) IAU=IM-1   
        IF(N.GE.IM+1.AND.K(IM+1,3).EQ.IGM) IAU=IM+1 
      ENDIF 
      IF(IGM.GE.0) THEN 
        K(IM,4)=N+1 
        DO 150 I=1,NEP  
  150   K(N+I,3)=IM 
      ELSE  
        K(N+1,3)=IPA(1) 
      ENDIF 
      IF(IGM.LE.0) THEN 
        DO 160 I=1,NEP  
  160   K(N+I,2)=K(IPA(I),2)    
      ELSEIF(KFLM.NE.21) THEN   
        K(N+1,2)=K(IM,2)    
        K(N+2,2)=K(IM,5)    
      ELSEIF(K(IM,5).EQ.21) THEN    
        K(N+1,2)=21 
        K(N+2,2)=21 
      ELSE  
        K(N+1,2)=K(IM,5)    
        K(N+2,2)=-K(IM,5)   
      ENDIF 
    
C...Reset flags on daughers and tries made. 
      DO 170 IP=1,NEP   
      K(N+IP,1)=3   
      K(N+IP,4)=0   
      K(N+IP,5)=0   
      KFLD(IP)=IABS(K(N+IP,2))  
      ITRY(IP)=0    
      ISL(IP)=0 
      ISI(IP)=0 
  170 IF(KFLD(IP).GT.0.AND.(KFLD(IP).LE.8.OR.KFLD(IP).EQ.21)) ISI(IP)=1 
      ISLM=0    
    
C...Maximum virtuality of daughters.    
cms..pre-initialization
      PEM=0.
      IF(IGM.LE.0) THEN 
        DO 180 I=1,NPA  
        IF(NPA.GE.3) P(N+I,4)=(PS(4)*P(IPA(I),4)-PS(1)*P(IPA(I),1)- 
     &  PS(2)*P(IPA(I),2)-PS(3)*P(IPA(I),3))/PS(5)  
        P(N+I,5)=MIN(QMAX,PS(5))    
        IF(NPA.GE.3) P(N+I,5)=MIN(P(N+I,5),P(N+I,4))    
  180   IF(ISI(I).EQ.0) P(N+I,5)=P(IPA(I),5)    
      ELSE  
        IF(MSTJ(43).LE.2) PEM=V(IM,2)   
        IF(MSTJ(43).GE.3) PEM=P(IM,4)   
        P(N+1,5)=MIN(P(IM,5),V(IM,1)*PEM)   
        P(N+2,5)=MIN(P(IM,5),(1.-V(IM,1))*PEM)  
        IF(K(N+2,2).EQ.22) P(N+2,5)=PMTH(1,22)  
      ENDIF 
      DO 190 I=1,NEP    
      PMSD(I)=P(N+I,5)  
      IF(ISI(I).EQ.1) THEN  
        IF(P(N+I,5).LE.PMTH(3,KFLD(I))) P(N+I,5)=PMTH(1,KFLD(I))    
      ENDIF 
  190 V(N+I,5)=P(N+I,5)**2  
    
C...Choose one of the daughters for evolution.  
  200 INUM=0    
      IF(NEP.EQ.1) INUM=1   
      DO 210 I=1,NEP    
  210 IF(INUM.EQ.0.AND.ISL(I).EQ.1) INUM=I  
      DO 220 I=1,NEP    
      IF(INUM.EQ.0.AND.ITRY(I).EQ.0.AND.ISI(I).EQ.1) THEN   
        IF(P(N+I,5).GE.PMTH(2,KFLD(I))) INUM=I  
      ENDIF 
  220 CONTINUE  
      IF(INUM.EQ.0) THEN    
        RMAX=0. 
        DO 230 I=1,NEP  
        IF(ISI(I).EQ.1.AND.PMSD(I).GE.PMQTH2) THEN  
          RPM=P(N+I,5)/PMSD(I)  
          IF(RPM.GT.RMAX.AND.P(N+I,5).GE.PMTH(2,KFLD(I))) THEN  
            RMAX=RPM    
            INUM=I  
          ENDIF 
        ENDIF   
  230   CONTINUE    
      ENDIF 
    
C...Store information on choice of evolving daughter.   
      INUM=MAX(1,INUM)  
      IEP(1)=N+INUM 
      DO 240 I=2,NEP    
      IEP(I)=IEP(I-1)+1 
  240 IF(IEP(I).GT.N+NEP) IEP(I)=N+1    
      DO 250 I=1,NEP    
  250 KFL(I)=IABS(K(IEP(I),2))  
      ITRY(INUM)=ITRY(INUM)+1   
      IF(ITRY(INUM).GT.200) THEN    
        CALL LUERRM(14,'(LUSHOW:) caught in infinite loop') 
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      Z=0.5 
      IF(KFL(1).EQ.0.OR.(KFL(1).GT.8.AND.KFL(1).NE.21)) GOTO 300    
      IF(P(IEP(1),5).LT.PMTH(2,KFL(1))) GOTO 300    
    
C...Calculate allowed z range.  
cms.. pre-initialization for compiler
      PMED=0.
      IF(NEP.EQ.1) THEN 
        PMED=PS(4)  
      ELSEIF(IGM.EQ.0.OR.MSTJ(43).LE.2) THEN    
        PMED=P(IM,5)    
      ELSE  
        IF(INUM.EQ.1) PMED=V(IM,1)*PEM  
        IF(INUM.EQ.2) PMED=(1.-V(IM,1))*PEM 
      ENDIF 
      IF(MOD(MSTJ(43),2).EQ.1) THEN 
        ZC=PMTH(2,21)/PMED  
        ZCE=PMTH(2,22)/PMED 
      ELSE  
        ZC=0.5*(1.-SQRT(MAX(0.,1.-(2.*PMTH(2,21)/PMED)**2)))    
        IF(ZC.LT.1E-4) ZC=(PMTH(2,21)/PMED)**2  
        ZCE=0.5*(1.-SQRT(MAX(0.,1.-(2.*PMTH(2,22)/PMED)**2)))   
        IF(ZCE.LT.1E-4) ZCE=(PMTH(2,22)/PMED)**2    
      ENDIF 
      ZC=MIN(ZC,0.491)  
      ZCE=MIN(ZCE,0.491)    
      IF((MSTJ(41).EQ.1.AND.ZC.GT.0.49).OR.(MSTJ(41).EQ.2.AND.  
     &MIN(ZC,ZCE).GT.0.49)) THEN    
        P(IEP(1),5)=PMTH(1,KFL(1))  
        V(IEP(1),5)=P(IEP(1),5)**2  
        GOTO 300    
      ENDIF 
    
C...Integral of Altarelli-Parisi z kernel for QCD.  
      IF(MSTJ(49).EQ.0.AND.KFL(1).EQ.21) THEN   
        FBR=6.*LOG((1.-ZC)/ZC)+MSTJ(45)*(0.5-ZC)    
      ELSEIF(MSTJ(49).EQ.0) THEN    
        FBR=(8./3.)*LOG((1.-ZC)/ZC) 
    
C...Integral of Altarelli-Parisi z kernel for scalar gluon. 
      ELSEIF(MSTJ(49).EQ.1.AND.KFL(1).EQ.21) THEN   
        FBR=(PARJ(87)+MSTJ(45)*PARJ(88))*(1.-2.*ZC) 
      ELSEIF(MSTJ(49).EQ.1) THEN    
        FBR=(1.-2.*ZC)/3.   
        IF(IGM.EQ.0.AND.M3JC.EQ.1) FBR=4.*FBR   
    
C...Integral of Altarelli-Parisi z kernel for Abelian vector gluon. 
      ELSEIF(KFL(1).EQ.21) THEN 
        FBR=6.*MSTJ(45)*(0.5-ZC)    
      ELSE  
        FBR=2.*LOG((1.-ZC)/ZC)  
      ENDIF 
    
C...Integral of Altarelli-Parisi kernel for photon emission.    
      FBRE=0.
      IF(MSTJ(41).EQ.2.AND.KFL(1).GE.1.AND.KFL(1).LE.8) 
     &FBRE=(KCHG(KFL(1),1)/3.)**2*2.*LOG((1.-ZCE)/ZCE)  
    
C...Inner veto algorithm starts. Find maximum mass for evolution.   
cms.. pre-initialization
      PM2=0.
  260 PMS=V(IEP(1),5)   
      IF(IGM.GE.0) THEN 
        PM2=0.  
        DO 270 I=2,NEP  
        PM=P(IEP(I),5)  
        IF(KFL(I).GT.0.AND.(KFL(I).LE.8.OR.KFL(I).EQ.21)) PM=   
     &  PMTH(2,KFL(I))  
  270   PM2=PM2+PM  
        PMS=MIN(PMS,(P(IM,5)-PM2)**2)   
      ENDIF 
    
C...Select mass for daughter in QCD evolution.  
      B0=27./6. 
      DO 280 IF=4,MSTJ(45)  
  280 IF(PMS.GT.4.*PMTH(2,IF)**2) B0=(33.-2.*IF)/6. 
      IF(MSTJ(44).LE.0) THEN    
        PMSQCD=PMS*EXP(MAX(-100.,LOG(RLU(0))*PARU(2)/(PARU(111)*FBR)))  
      ELSEIF(MSTJ(44).EQ.1) THEN    
        PMSQCD=4.*ALAMS*(0.25*PMS/ALAMS)**(RLU(0)**(B0/FBR))    
      ELSE  
        PMSQCD=PMS*RLU(0)**(ALFM*B0/FBR)    
      ENDIF 
      IF(ZC.GT.0.49.OR.PMSQCD.LE.PMTH(4,KFL(1))**2) PMSQCD= 
     &PMTH(2,KFL(1))**2 
      V(IEP(1),5)=PMSQCD    
      MCE=1 
    
C...Select mass for daughter in QED evolution.  
      IF(MSTJ(41).EQ.2.AND.KFL(1).GE.1.AND.KFL(1).LE.8) THEN    
        PMSQED=PMS*EXP(MAX(-100.,LOG(RLU(0))*PARU(2)/(PARU(101)*FBRE))) 
        IF(ZCE.GT.0.49.OR.PMSQED.LE.PMTH(5,KFL(1))**2) PMSQED=  
     &  PMTH(2,KFL(1))**2   
        IF(PMSQED.GT.PMSQCD) THEN   
          V(IEP(1),5)=PMSQED    
          MCE=2 
        ENDIF   
      ENDIF 
    
C...Check whether daughter mass below cutoff.   
      P(IEP(1),5)=SQRT(V(IEP(1),5)) 
      IF(P(IEP(1),5).LE.PMTH(3,KFL(1))) THEN    
        P(IEP(1),5)=PMTH(1,KFL(1))  
        V(IEP(1),5)=P(IEP(1),5)**2  
        GOTO 300    
      ENDIF 
    
C...Select z value of branching: q -> qgamma.   
      IF(MCE.EQ.2) THEN 
        Z=1.-(1.-ZCE)*(ZCE/(1.-ZCE))**RLU(0)    
        IF(1.+Z**2.LT.2.*RLU(0)) GOTO 260   
        K(IEP(1),5)=22  
    
C...Select z value of branching: q -> qg, g -> gg, g -> qqbar.  
      ELSEIF(MSTJ(49).NE.1.AND.KFL(1).NE.21) THEN   
        Z=1.-(1.-ZC)*(ZC/(1.-ZC))**RLU(0)   
        IF(1.+Z**2.LT.2.*RLU(0)) GOTO 260   
        K(IEP(1),5)=21  
      ELSEIF(MSTJ(49).EQ.0.AND.MSTJ(45)*(0.5-ZC).LT.RLU(0)*FBR) THEN    
        Z=(1.-ZC)*(ZC/(1.-ZC))**RLU(0)  
        IF(RLU(0).GT.0.5) Z=1.-Z    
        IF((1.-Z*(1.-Z))**2.LT.RLU(0)) GOTO 260 
        K(IEP(1),5)=21  
      ELSEIF(MSTJ(49).NE.1) THEN    
        Z=ZC+(1.-2.*ZC)*RLU(0)  
        IF(Z**2+(1.-Z)**2.LT.RLU(0)) GOTO 260   
        KFLB=1+INT(MSTJ(45)*RLU(0)) 
        PMQ=4.*PMTH(2,KFLB)**2/V(IEP(1),5)  
        IF(PMQ.GE.1.) GOTO 260  
        PMQ0=4.*PMTH(2,21)**2/V(IEP(1),5)   
        IF(MOD(MSTJ(43),2).EQ.0.AND.(1.+0.5*PMQ)*SQRT(1.-PMQ).LT.   
     &  RLU(0)*(1.+0.5*PMQ0)*SQRT(1.-PMQ0)) GOTO 260    
        K(IEP(1),5)=KFLB    
    
C...Ditto for scalar gluon model.   
      ELSEIF(KFL(1).NE.21) THEN 
        Z=1.-SQRT(ZC**2+RLU(0)*(1.-2.*ZC))  
        K(IEP(1),5)=21  
      ELSEIF(RLU(0)*(PARJ(87)+MSTJ(45)*PARJ(88)).LE.PARJ(87)) THEN  
        Z=ZC+(1.-2.*ZC)*RLU(0)  
        K(IEP(1),5)=21  
      ELSE  
        Z=ZC+(1.-2.*ZC)*RLU(0)  
        KFLB=1+INT(MSTJ(45)*RLU(0)) 
        PMQ=4.*PMTH(2,KFLB)**2/V(IEP(1),5)  
        IF(PMQ.GE.1.) GOTO 260  
        K(IEP(1),5)=KFLB    
      ENDIF 
      IF(MCE.EQ.1.AND.MSTJ(44).GE.2) THEN   
        IF(Z*(1.-Z)*V(IEP(1),5).LT.PT2MIN) GOTO 260 
        IF(ALFM/LOG(V(IEP(1),5)*Z*(1.-Z)/ALAMS).LT.RLU(0)) GOTO 260 
      ENDIF 
    
C...Check if z consistent with chosen m.    
      IF(KFL(1).EQ.21) THEN 
        KFLGD1=IABS(K(IEP(1),5))    
        KFLGD2=KFLGD1   
      ELSE  
        KFLGD1=KFL(1)   
        KFLGD2=IABS(K(IEP(1),5))    
      ENDIF 
      PED=0.
      IF(NEP.EQ.1) THEN 
        PED=PS(4)   
      ELSEIF(NEP.GE.3) THEN 
        PED=P(IEP(1),4) 
      ELSEIF(IGM.EQ.0.OR.MSTJ(43).LE.2) THEN    
        PED=0.5*(V(IM,5)+V(IEP(1),5)-PM2**2)/P(IM,5)    
      ELSE  
        IF(IEP(1).EQ.N+1) PED=V(IM,1)*PEM   
        IF(IEP(1).EQ.N+2) PED=(1.-V(IM,1))*PEM  
      ENDIF 
      IF(MOD(MSTJ(43),2).EQ.1) THEN 
        PMQTH3=0.5*PARJ(82) 
        IF(KFLGD2.EQ.22) PMQTH3=0.5*PARJ(83)    
        PMQ1=(PMTH(1,KFLGD1)**2+PMQTH3**2)/V(IEP(1),5)  
        PMQ2=(PMTH(1,KFLGD2)**2+PMQTH3**2)/V(IEP(1),5)  
        ZD=SQRT(MAX(0.,(1.-V(IEP(1),5)/PED**2)*((1.-PMQ1-PMQ2)**2-  
     &  4.*PMQ1*PMQ2))) 
        ZH=1.+PMQ1-PMQ2 
      ELSE  
        ZD=SQRT(MAX(0.,1.-V(IEP(1),5)/PED**2))  
        ZH=1.   
      ENDIF 
      ZL=0.5*(ZH-ZD)    
      ZU=0.5*(ZH+ZD)    
      IF(Z.LT.ZL.OR.Z.GT.ZU) GOTO 260   
      IF(KFL(1).EQ.21) V(IEP(1),3)=LOG(ZU*(1.-ZL)/MAX(1E-20,ZL* 
     &(1.-ZU))) 
      IF(KFL(1).NE.21) V(IEP(1),3)=LOG((1.-ZL)/MAX(1E-10,1.-ZU))    
    
C...Three-jet matrix element correction.    
      IF(IGM.EQ.0.AND.M3JC.EQ.1) THEN   
        X1=Z*(1.+V(IEP(1),5)/V(NS+1,5)) 
        X2=1.-V(IEP(1),5)/V(NS+1,5) 
        X3=(1.-X1)+(1.-X2)  
        IF(MCE.EQ.2) THEN   
          KI1=K(IPA(INUM),2)    
          KI2=K(IPA(3-INUM),2)  
          QF1=KCHG(IABS(KI1),1)*ISIGN(1,KI1)/3. 
          QF2=KCHG(IABS(KI2),1)*ISIGN(1,KI2)/3. 
          WSHOW=QF1**2*(1.-X1)/X3*(1.+(X1/(2.-X2))**2)+ 
     &    QF2**2*(1.-X2)/X3*(1.+(X2/(2.-X1))**2)    
          WME=(QF1*(1.-X1)/X3-QF2*(1.-X2)/X3)**2*(X1**2+X2**2)  
        ELSEIF(MSTJ(49).NE.1) THEN  
          WSHOW=1.+(1.-X1)/X3*(X1/(2.-X2))**2+  
     &    (1.-X2)/X3*(X2/(2.-X1))**2    
          WME=X1**2+X2**2   
        ELSE    
          WSHOW=4.*X3*((1.-X1)/(2.-X2)**2+(1.-X2)/(2.-X1)**2)   
          WME=X3**2 
        ENDIF   
        IF(WME.LT.RLU(0)*WSHOW) GOTO 260    
    
C...Impose angular ordering by rejection of nonordered emission.    
      ELSEIF(MCE.EQ.1.AND.IGM.GT.0.AND.MSTJ(42).GE.2) THEN  
        MAOM=1  
        ZM=V(IM,1)  
        IF(IEP(1).EQ.N+2) ZM=1.-V(IM,1) 
        THE2ID=Z*(1.-Z)*(ZM*P(IM,4))**2/V(IEP(1),5) 
        IAOM=IM 
  290   IF(K(IAOM,5).EQ.22) THEN    
          IAOM=K(IAOM,3)    
          IF(K(IAOM,3).LE.NS) MAOM=0    
          IF(MAOM.EQ.1) GOTO 290    
        ENDIF   
        IF(MAOM.EQ.1) THEN  
          THE2IM=V(IAOM,1)*(1.-V(IAOM,1))*P(IAOM,4)**2/V(IAOM,5)    
          IF(THE2ID.LT.THE2IM) GOTO 260 
        ENDIF   
      ENDIF 
    
C...Impose user-defined maximum angle at first branching.   
      IF(MSTJ(48).EQ.1) THEN    
        IF(NEP.EQ.1.AND.IM.EQ.NS) THEN  
          THE2ID=Z*(1.-Z)*PS(4)**2/V(IEP(1),5)  
          IF(THE2ID.LT.1./PARJ(85)**2) GOTO 260 
        ELSEIF(NEP.EQ.2.AND.IEP(1).EQ.NS+2) THEN    
          THE2ID=Z*(1.-Z)*(0.5*P(IM,4))**2/V(IEP(1),5)  
          IF(THE2ID.LT.1./PARJ(85)**2) GOTO 260 
        ELSEIF(NEP.EQ.2.AND.IEP(1).EQ.NS+3) THEN    
          THE2ID=Z*(1.-Z)*(0.5*P(IM,4))**2/V(IEP(1),5)  
          IF(THE2ID.LT.1./PARJ(86)**2) GOTO 260 
        ENDIF   
      ENDIF 
    
C...End of inner veto algorithm. Check if only one leg evolved so far.  
  300 V(IEP(1),1)=Z 
      ISL(1)=0  
      ISL(2)=0  
      IF(NEP.EQ.1) GOTO 330 
      IF(NEP.EQ.2.AND.P(IEP(1),5)+P(IEP(2),5).GE.P(IM,5)) GOTO 200  
      DO 310 I=1,NEP    
      IF(ITRY(I).EQ.0.AND.KFLD(I).GT.0.AND.(KFLD(I).LE.8.OR.KFLD(I).EQ. 
     &21)) THEN 
        IF(P(N+I,5).GE.PMTH(2,KFLD(I))) GOTO 200    
      ENDIF 
  310 CONTINUE  
    
C...Check if chosen multiplet m1,m2,z1,z2 is physical.  
cms.. pre-initialization
      PTS=0.
      PA1S=0.
      PA2S=0.
      PA3S=0.
      IF(NEP.EQ.3) THEN 
        PA1S=(P(N+1,4)+P(N+1,5))*(P(N+1,4)-P(N+1,5))    
        PA2S=(P(N+2,4)+P(N+2,5))*(P(N+2,4)-P(N+2,5))    
        PA3S=(P(N+3,4)+P(N+3,5))*(P(N+3,4)-P(N+3,5))    
        PTS=0.25*(2.*PA1S*PA2S+2.*PA1S*PA3S+2.*PA2S*PA3S-   
     &  PA1S**2-PA2S**2-PA3S**2)/PA1S   
        IF(PTS.LE.0.) GOTO 200  
      ELSEIF(IGM.EQ.0.OR.MSTJ(43).LE.2.OR.MOD(MSTJ(43),2).EQ.0) THEN    
        DO 320 I1=N+1,N+2   
        KFLDA=IABS(K(I1,2)) 
        IF(KFLDA.EQ.0.OR.(KFLDA.GT.8.AND.KFLDA.NE.21)) GOTO 320 
        IF(P(I1,5).LT.PMTH(2,KFLDA)) GOTO 320   
        IF(KFLDA.EQ.21) THEN    
          KFLGD1=IABS(K(I1,5))  
          KFLGD2=KFLGD1 
        ELSE    
          KFLGD1=KFLDA  
          KFLGD2=IABS(K(I1,5))  
        ENDIF   
        I2=2*N+3-I1 
        IF(IGM.EQ.0.OR.MSTJ(43).LE.2) THEN  
          PED=0.5*(V(IM,5)+V(I1,5)-V(I2,5))/P(IM,5) 
        ELSE    
cms.. modified to avoid comp. warning
cc..          IF(I1.EQ.N+1) ZM=V(IM,1)  
          ZM=V(IM,1)
          IF(I1.EQ.N+2) ZM=1.-V(IM,1)   
          PML=SQRT((V(IM,5)-V(N+1,5)-V(N+2,5))**2-  
     &    4.*V(N+1,5)*V(N+2,5)) 
          PED=PEM*(0.5*(V(IM,5)-PML+V(I1,5)-V(I2,5))+PML*ZM)/V(IM,5)    
        ENDIF   
        IF(MOD(MSTJ(43),2).EQ.1) THEN   
          PMQTH3=0.5*PARJ(82)   
          IF(KFLGD2.EQ.22) PMQTH3=0.5*PARJ(83)  
          PMQ1=(PMTH(1,KFLGD1)**2+PMQTH3**2)/V(I1,5)    
          PMQ2=(PMTH(1,KFLGD2)**2+PMQTH3**2)/V(I1,5)    
          ZD=SQRT(MAX(0.,(1.-V(I1,5)/PED**2)*((1.-PMQ1-PMQ2)**2-    
     &    4.*PMQ1*PMQ2)))   
          ZH=1.+PMQ1-PMQ2   
        ELSE    
          ZD=SQRT(MAX(0.,1.-V(I1,5)/PED**2))    
          ZH=1. 
        ENDIF   
        ZL=0.5*(ZH-ZD)  
        ZU=0.5*(ZH+ZD)  
        IF(I1.EQ.N+1.AND.(V(I1,1).LT.ZL.OR.V(I1,1).GT.ZU)) ISL(1)=1 
        IF(I1.EQ.N+2.AND.(V(I1,1).LT.ZL.OR.V(I1,1).GT.ZU)) ISL(2)=1 
        IF(KFLDA.EQ.21) V(I1,4)=LOG(ZU*(1.-ZL)/MAX(1E-20,ZL*(1.-ZU)))   
        IF(KFLDA.NE.21) V(I1,4)=LOG((1.-ZL)/MAX(1E-10,1.-ZU))   
  320   CONTINUE    
        IF(ISL(1).EQ.1.AND.ISL(2).EQ.1.AND.ISLM.NE.0) THEN  
          ISL(3-ISLM)=0 
          ISLM=3-ISLM   
        ELSEIF(ISL(1).EQ.1.AND.ISL(2).EQ.1) THEN    
          ZDR1=MAX(0.,V(N+1,3)/V(N+1,4)-1.) 
          ZDR2=MAX(0.,V(N+2,3)/V(N+2,4)-1.) 
          IF(ZDR2.GT.RLU(0)*(ZDR1+ZDR2)) ISL(1)=0   
          IF(ISL(1).EQ.1) ISL(2)=0  
          IF(ISL(1).EQ.0) ISLM=1    
          IF(ISL(2).EQ.0) ISLM=2    
        ENDIF   
        IF(ISL(1).EQ.1.OR.ISL(2).EQ.1) GOTO 200 
      ENDIF 
      IF(IGM.GT.0.AND.MOD(MSTJ(43),2).EQ.1.AND.(P(N+1,5).GE.    
     &PMTH(2,KFLD(1)).OR.P(N+2,5).GE.PMTH(2,KFLD(2)))) THEN 
        PMQ1=V(N+1,5)/V(IM,5)   
        PMQ2=V(N+2,5)/V(IM,5)   
        ZD=SQRT(MAX(0.,(1.-V(IM,5)/PEM**2)*((1.-PMQ1-PMQ2)**2-  
     &  4.*PMQ1*PMQ2))) 
        ZH=1.+PMQ1-PMQ2 
        ZL=0.5*(ZH-ZD)  
        ZU=0.5*(ZH+ZD)  
        IF(V(IM,1).LT.ZL.OR.V(IM,1).GT.ZU) GOTO 200 
      ENDIF 
    
C...Accepted branch. Construct four-momentum for initial partons.   
  330 MAZIP=0   
      MAZIC=0
cms.. pre-initialization for compiler
      PZM=0.
      PMLS=0.
      PT=0.
      IF(NEP.EQ.1) THEN 
        P(N+1,1)=0. 
        P(N+1,2)=0. 
        P(N+1,3)=SQRT(MAX(0.,(P(IPA(1),4)+P(N+1,5))*(P(IPA(1),4)-   
     &  P(N+1,5)))) 
        P(N+1,4)=P(IPA(1),4)    
        V(N+1,2)=P(N+1,4)   
      ELSEIF(IGM.EQ.0.AND.NEP.EQ.2) THEN    
        PED1=0.5*(V(IM,5)+V(N+1,5)-V(N+2,5))/P(IM,5)    
        P(N+1,1)=0. 
        P(N+1,2)=0. 
        P(N+1,3)=SQRT(MAX(0.,(PED1+P(N+1,5))*(PED1-P(N+1,5))))  
        P(N+1,4)=PED1   
        P(N+2,1)=0. 
        P(N+2,2)=0. 
        P(N+2,3)=-P(N+1,3)  
        P(N+2,4)=P(IM,5)-PED1   
        V(N+1,2)=P(N+1,4)   
        V(N+2,2)=P(N+2,4)   
      ELSEIF(NEP.EQ.3) THEN 
        P(N+1,1)=0. 
        P(N+1,2)=0. 
        P(N+1,3)=SQRT(MAX(0.,PA1S)) 
        P(N+2,1)=SQRT(PTS)  
        P(N+2,2)=0. 
        P(N+2,3)=0.5*(PA3S-PA2S-PA1S)/P(N+1,3)  
        P(N+3,1)=-P(N+2,1)  
        P(N+3,2)=0. 
        P(N+3,3)=-(P(N+1,3)+P(N+2,3))   
        V(N+1,2)=P(N+1,4)   
        V(N+2,2)=P(N+2,4)   
        V(N+3,2)=P(N+3,4)   
    
C...Construct transverse momentum for ordinary branching in shower. 
      ELSE  
        ZM=V(IM,1)  
        PZM=SQRT(MAX(0.,(PEM+P(IM,5))*(PEM-P(IM,5))))   
        PMLS=(V(IM,5)-V(N+1,5)-V(N+2,5))**2-4.*V(N+1,5)*V(N+2,5)    
        IF(PZM.LE.0.) THEN  
          PTS=0.    
        ELSEIF(MOD(MSTJ(43),2).EQ.1) THEN   
          PTS=(PEM**2*(ZM*(1.-ZM)*V(IM,5)-(1.-ZM)*V(N+1,5)- 
     &    ZM*V(N+2,5))-0.25*PMLS)/PZM**2    
        ELSE    
          PTS=PMLS*(ZM*(1.-ZM)*PEM**2/V(IM,5)-0.25)/PZM**2  
        ENDIF   
        PT=SQRT(MAX(0.,PTS))    
    
C...Find coefficient of azimuthal asymmetry due to gluon polarization.  
        HAZIP=0.    
        IF(MSTJ(49).NE.1.AND.MOD(MSTJ(46),2).EQ.1.AND.K(IM,2).EQ.21.    
     &  AND.IAU.NE.0) THEN  
          IF(K(IGM,3).NE.0) MAZIP=1 
          ZAU=V(IGM,1)  
          IF(IAU.EQ.IM+1) ZAU=1.-V(IGM,1)   
          IF(MAZIP.EQ.0) ZAU=0. 
          IF(K(IGM,2).NE.21) THEN   
            HAZIP=2.*ZAU/(1.+ZAU**2)    
          ELSE  
            HAZIP=(ZAU/(1.-ZAU*(1.-ZAU)))**2    
          ENDIF 
          IF(K(N+1,2).NE.21) THEN   
            HAZIP=HAZIP*(-2.*ZM*(1.-ZM))/(1.-2.*ZM*(1.-ZM)) 
          ELSE  
            HAZIP=HAZIP*(ZM*(1.-ZM)/(1.-ZM*(1.-ZM)))**2 
          ENDIF 
        ENDIF   
    
C...Find coefficient of azimuthal asymmetry due to soft gluon   
C...interference.   
        HAZIC=0.    
        IF(MSTJ(46).GE.2.AND.(K(N+1,2).EQ.21.OR.K(N+2,2).EQ.21).    
     &  AND.IAU.NE.0) THEN  
          IF(K(IGM,3).NE.0) MAZIC=N+1   
          IF(K(IGM,3).NE.0.AND.K(N+1,2).NE.21) MAZIC=N+2    
          IF(K(IGM,3).NE.0.AND.K(N+1,2).EQ.21.AND.K(N+2,2).EQ.21.AND.   
     &    ZM.GT.0.5) MAZIC=N+2  
          IF(K(IAU,2).EQ.22) MAZIC=0    
          ZS=ZM 
          IF(MAZIC.EQ.N+2) ZS=1.-ZM 
          ZGM=V(IGM,1)  
          IF(IAU.EQ.IM-1) ZGM=1.-V(IGM,1)   
          IF(MAZIC.EQ.0) ZGM=1. 
          HAZIC=(P(IM,5)/P(IGM,5))*SQRT((1.-ZS)*(1.-ZGM)/(ZS*ZGM))  
          HAZIC=MIN(0.95,HAZIC) 
        ENDIF   
      ENDIF 
    
C...Construct kinematics for ordinary branching in shower.  
  340 IF(NEP.EQ.2.AND.IGM.GT.0) THEN    
        IF(MOD(MSTJ(43),2).EQ.1) THEN   
          P(N+1,4)=PEM*V(IM,1)  
        ELSE    
          P(N+1,4)=PEM*(0.5*(V(IM,5)-SQRT(PMLS)+V(N+1,5)-V(N+2,5))+ 
     &    SQRT(PMLS)*ZM)/V(IM,5)    
        ENDIF   
        PHI=PARU(2)*RLU(0)  
        P(N+1,1)=PT*COS(PHI)    
        P(N+1,2)=PT*SIN(PHI)    
        IF(PZM.GT.0.) THEN  
          P(N+1,3)=0.5*(V(N+2,5)-V(N+1,5)-V(IM,5)+2.*PEM*P(N+1,4))/PZM  
        ELSE    
          P(N+1,3)=0.   
        ENDIF   
        P(N+2,1)=-P(N+1,1)  
        P(N+2,2)=-P(N+1,2)  
        P(N+2,3)=PZM-P(N+1,3)   
        P(N+2,4)=PEM-P(N+1,4)   
        IF(MSTJ(43).LE.2) THEN  
          V(N+1,2)=(PEM*P(N+1,4)-PZM*P(N+1,3))/P(IM,5)  
          V(N+2,2)=(PEM*P(N+2,4)-PZM*P(N+2,3))/P(IM,5)  
        ENDIF   
      ENDIF 
    
C...Rotate and boost daughters. 
      IF(IGM.GT.0) THEN 
        IF(MSTJ(43).LE.2) THEN  
          BEX=P(IGM,1)/P(IGM,4) 
          BEY=P(IGM,2)/P(IGM,4) 
          BEZ=P(IGM,3)/P(IGM,4) 
          GA=P(IGM,4)/P(IGM,5)  
          GABEP=GA*(GA*(BEX*P(IM,1)+BEY*P(IM,2)+BEZ*P(IM,3))/(1.+GA)-   
     &    P(IM,4))  
        ELSE    
          BEX=0.    
          BEY=0.    
          BEZ=0.    
          GA=1. 
          GABEP=0.  
        ENDIF   
        THE=ULANGL(P(IM,3)+GABEP*BEZ,SQRT((P(IM,1)+GABEP*BEX)**2+   
     &  (P(IM,2)+GABEP*BEY)**2))    
        PHI=ULANGL(P(IM,1)+GABEP*BEX,P(IM,2)+GABEP*BEY) 
        DO 350 I=N+1,N+2    
        DP(1)=dble(COS(THE)*COS(PHI)*P(I,1)-SIN(PHI)*P(I,2)+ 
     &  SIN(THE)*COS(PHI)*P(I,3))
        DP(2)=dble(COS(THE)*SIN(PHI)*P(I,1)+COS(PHI)*P(I,2)+ 
     &  SIN(THE)*SIN(PHI)*P(I,3))
        DP(3)=dble(-SIN(THE)*P(I,1)+COS(THE)*P(I,3))
        DP(4)=dble(P(I,4))
        DBP=dble(BEX)*DP(1)+dble(BEY)*DP(2)+dble(BEZ)*DP(3)   
        DGABP=dble(GA)*(dble(GA)*DBP/(1D0+dble(GA))+DP(4))    
        P(I,1)=sngl(DP(1)+DGABP*dble(BEX))
        P(I,2)=sngl(DP(2)+DGABP*dble(BEY))
        P(I,3)=sngl(DP(3)+DGABP*dble(BEZ))
  350   P(I,4)=GA*sngl(DP(4)+DBP)   
      ENDIF 
    
C...Weight with azimuthal distribution, if required.    
      IF(MAZIP.NE.0.OR.MAZIC.NE.0) THEN 
        DO 360 J=1,3    
        DPT(1,J)=dble(P(IM,J))
        DPT(2,J)=dble(P(IAU,J))  
  360   DPT(3,J)=dble(P(N+1,J))
        DPMA=DPT(1,1)*DPT(2,1)+DPT(1,2)*DPT(2,2)+DPT(1,3)*DPT(2,3)  
        DPMD=DPT(1,1)*DPT(3,1)+DPT(1,2)*DPT(3,2)+DPT(1,3)*DPT(3,3)  
        DPMM=DPT(1,1)**2+DPT(1,2)**2+DPT(1,3)**2    
        DO 370 J=1,3    
        DPT(4,J)=DPT(2,J)-DPMA*DPT(1,J)/DPMM    
  370   DPT(5,J)=DPT(3,J)-DPMD*DPT(1,J)/DPMM    
        DPT(4,4)=DSQRT(DPT(4,1)**2+DPT(4,2)**2+DPT(4,3)**2)  
        DPT(5,4)=DSQRT(DPT(5,1)**2+DPT(5,2)**2+DPT(5,3)**2)  
clin-5/2012:
c        IF(MIN(DPT(4,4),DPT(5,4)).GT.0.1*PARJ(82)) THEN 
        IF(sngl(MIN(DPT(4,4),DPT(5,4))).GT.(0.1*PARJ(82))) THEN 
           CAD=sngl((DPT(4,1)*DPT(5,1)+DPT(4,2)*DPT(5,2)+ 
     &    DPT(4,3)*DPT(5,3))/(DPT(4,4)*DPT(5,4)))
          IF(MAZIP.NE.0) THEN   
            IF(1.+HAZIP*(2.*CAD**2-1.).LT.RLU(0)*(1.+ABS(HAZIP)))   
     &      GOTO 340    
          ENDIF 
          IF(MAZIC.NE.0) THEN   
            IF(MAZIC.EQ.N+2) CAD=-CAD   
            IF((1.-HAZIC)*(1.-HAZIC*CAD)/(1.+HAZIC**2-2.*HAZIC*CAD).    
     &      LT.RLU(0)) GOTO 340 
          ENDIF 
        ENDIF   
      ENDIF 
    
C...Continue loop over partons that may branch, until none left.    
      IF(IGM.GE.0) K(IM,1)=14   
      N=N+NEP   
      NEP=2 
      IF(N.GT.MSTU(4)-MSTU(32)-5) THEN  
        CALL LUERRM(11,'(LUSHOW:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) N=NS  
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      GOTO 140  
    
C...Set information on imagined shower initiator.   
  380 IF(NPA.GE.2) THEN 
        K(NS+1,1)=11    
        K(NS+1,2)=94    
        K(NS+1,3)=IP1   
        IF(IP2.GT.0.AND.IP2.LT.IP1) K(NS+1,3)=IP2   
        K(NS+1,4)=NS+2  
        K(NS+1,5)=NS+1+NPA  
        IIM=1   
      ELSE  
        IIM=0   
      ENDIF 
    
C...Reconstruct string drawing information. 
      DO 390 I=NS+1+IIM,N   
      IF(K(I,1).LE.10.AND.K(I,2).EQ.22) THEN    
        K(I,1)=1    
      ELSEIF(K(I,1).LE.10) THEN 
        K(I,4)=MSTU(5)*(K(I,4)/MSTU(5)) 
        K(I,5)=MSTU(5)*(K(I,5)/MSTU(5)) 
      ELSEIF(K(MOD(K(I,4),MSTU(5))+1,2).NE.22) THEN 
        ID1=MOD(K(I,4),MSTU(5)) 
        IF(K(I,2).GE.1.AND.K(I,2).LE.8) ID1=MOD(K(I,4),MSTU(5))+1   
        ID2=2*MOD(K(I,4),MSTU(5))+1-ID1 
        K(I,4)=MSTU(5)*(K(I,4)/MSTU(5))+ID1 
        K(I,5)=MSTU(5)*(K(I,5)/MSTU(5))+ID2 
        K(ID1,4)=K(ID1,4)+MSTU(5)*I 
        K(ID1,5)=K(ID1,5)+MSTU(5)*ID2   
        K(ID2,4)=K(ID2,4)+MSTU(5)*ID1   
        K(ID2,5)=K(ID2,5)+MSTU(5)*I 
      ELSE  
        ID1=MOD(K(I,4),MSTU(5)) 
        ID2=ID1+1   
        K(I,4)=MSTU(5)*(K(I,4)/MSTU(5))+ID1 
        K(I,5)=MSTU(5)*(K(I,5)/MSTU(5))+ID1 
        K(ID1,4)=K(ID1,4)+MSTU(5)*I 
        K(ID1,5)=K(ID1,5)+MSTU(5)*I 
        K(ID2,4)=0  
        K(ID2,5)=0  
      ENDIF 
  390 CONTINUE  
    
C...Transformation from CM frame.   
      IF(NPA.GE.2) THEN 
        BEX=PS(1)/PS(4) 
        BEY=PS(2)/PS(4) 
        BEZ=PS(3)/PS(4) 
        GA=PS(4)/PS(5)  
        GABEP=GA*(GA*(BEX*P(IPA(1),1)+BEY*P(IPA(1),2)+BEZ*P(IPA(1),3))  
     &  /(1.+GA)-P(IPA(1),4))   
      ELSE  
        BEX=0.  
        BEY=0.  
        BEZ=0.  
        GABEP=0.    
      ENDIF 
      THE=ULANGL(P(IPA(1),3)+GABEP*BEZ,SQRT((P(IPA(1),1)    
     &+GABEP*BEX)**2+(P(IPA(1),2)+GABEP*BEY)**2))   
      PHI=ULANGL(P(IPA(1),1)+GABEP*BEX,P(IPA(1),2)+GABEP*BEY)   
      IF(NPA.EQ.3) THEN 
        CHI=ULANGL(COS(THE)*COS(PHI)*(P(IPA(2),1)+GABEP*BEX)+COS(THE)*  
     &  SIN(PHI)*(P(IPA(2),2)+GABEP*BEY)-SIN(THE)*(P(IPA(2),3)+GABEP*   
     &  BEZ),-SIN(PHI)*(P(IPA(2),1)+GABEP*BEX)+COS(PHI)*(P(IPA(2),2)+   
     &  GABEP*BEY)) 
        CALL LUDBRB(NS+1,N,0.,CHI,0D0,0D0,0D0)  
      ENDIF 
      DBEX=DBLE(BEX)    
      DBEY=DBLE(BEY)    
      DBEZ=DBLE(BEZ)    
      CALL LUDBRB(NS+1,N,THE,PHI,DBEX,DBEY,DBEZ)    
    
C...Decay vertex of shower. 
      DO 400 I=NS+1,N   
      DO 400 J=1,5  
  400 V(I,J)=V(IP1,J)   
    
C...Delete trivial shower, else connect initiators. 
      IF(N.EQ.NS+NPA+IIM) THEN  
        N=NS    
      ELSE  
        DO 410 IP=1,NPA 
        K(IPA(IP),1)=14 
        K(IPA(IP),4)=K(IPA(IP),4)+NS+IIM+IP 
        K(IPA(IP),5)=K(IPA(IP),5)+NS+IIM+IP 
        K(NS+IIM+IP,3)=IPA(IP)  
        IF(IIM.EQ.1.AND.MSTU(16).NE.2) K(NS+IIM+IP,3)=NS+1  
        K(NS+IIM+IP,4)=MSTU(5)*IPA(IP)+K(NS+IIM+IP,4)   
  410   K(NS+IIM+IP,5)=MSTU(5)*IPA(IP)+K(NS+IIM+IP,5)   
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUBOEI(NSAV)   
    
C...Purpose: to modify event so as to approximately take into account   
C...Bose-Einstein effects according to a simple phenomenological    
C...parametrization.    
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      DIMENSION DPS(4),KFBE(9),NBE(0:9),BEI(100)    
      DATA KFBE/211,-211,111,321,-321,130,310,221,331/  
    
C...Boost event to overall CM frame. Calculate CM energy.   
      IF((MSTJ(51).NE.1.AND.MSTJ(51).NE.2).OR.N-NSAV.LE.1) RETURN   
      DO 100 J=1,4  
  100 DPS(J)=0.d0
      DO 120 I=1,N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 120  
      DO 110 J=1,4  
  110 DPS(J)=DPS(J)+dble(P(I,J))
  120 CONTINUE  
      CALL LUDBRB(0,0,0.,0.,-DPS(1)/DPS(4),-DPS(2)/DPS(4),  
     &-DPS(3)/DPS(4))   
      PECM=0.   
      DO 130 I=1,N  
  130 IF(K(I,1).GE.1.AND.K(I,1).LE.10) PECM=PECM+P(I,4) 
    
C...Reserve copy of particles by species at end of record.  
      NBE(0)=N+MSTU(3)  
      DO 160 IBE=1,MIN(9,MSTJ(51))  
      NBE(IBE)=NBE(IBE-1)   
      DO 150 I=NSAV+1,N 
      IF(K(I,2).NE.KFBE(IBE)) GOTO 150  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 150  
      IF(NBE(IBE).GE.MSTU(4)-MSTU(32)-5) THEN   
        CALL LUERRM(11,'(LUBOEI:) no more memory left in LUJETS')   
        RETURN  
      ENDIF 
      NBE(IBE)=NBE(IBE)+1   
      K(NBE(IBE),1)=I   
      DO 140 J=1,3  
  140 P(NBE(IBE),J)=0.  
  150 CONTINUE  
  160 CONTINUE  
    
C...Tabulate integral for subsequent momentum shift.    
cms.. preinitialize for compiler
      NBIN=0
      BEEX=0.
      PMHQ=0.
      QDEL=0.
      DO 210 IBE=1,MIN(9,MSTJ(51))  
      IF(IBE.NE.1.AND.IBE.NE.4.AND.IBE.LE.7) GOTO 180   
      IF(IBE.EQ.1.AND.MAX(NBE(1)-NBE(0),NBE(2)-NBE(1),NBE(3)-NBE(2)).   
     &LE.1) GOTO 180    
      IF(IBE.EQ.4.AND.MAX(NBE(4)-NBE(3),NBE(5)-NBE(4),NBE(6)-NBE(5),    
     &NBE(7)-NBE(6)).LE.1) GOTO 180 
      IF(IBE.GE.8.AND.NBE(IBE)-NBE(IBE-1).LE.1) GOTO 180    
      IF(IBE.EQ.1) PMHQ=2.*ULMASS(211)  
      IF(IBE.EQ.4) PMHQ=2.*ULMASS(321)  
      IF(IBE.EQ.8) PMHQ=2.*ULMASS(221)  
      IF(IBE.EQ.9) PMHQ=2.*ULMASS(331)  
      QDEL=0.1*MIN(PMHQ,PARJ(93))   
      IF(MSTJ(51).EQ.1) THEN    
        NBIN=MIN(100,NINT(9.*PARJ(93)/QDEL))    
        BEEX=EXP(0.5*QDEL/PARJ(93)) 
        BERT=EXP(-QDEL/PARJ(93))    
      ELSE  
        NBIN=MIN(100,NINT(3.*PARJ(93)/QDEL))    
      ENDIF 
      DO 170 IBIN=1,NBIN    
      QBIN=QDEL*(IBIN-0.5)  
      BEI(IBIN)=QDEL*(QBIN**2+QDEL**2/12.)/SQRT(QBIN**2+PMHQ**2)    
      IF(MSTJ(51).EQ.1) THEN    
        BEEX=BEEX*BERT  
        BEI(IBIN)=BEI(IBIN)*BEEX    
      ELSE  
        BEI(IBIN)=BEI(IBIN)*EXP(-(QBIN/PARJ(93))**2)    
      ENDIF 
  170 IF(IBIN.GE.2) BEI(IBIN)=BEI(IBIN)+BEI(IBIN-1) 
    
C...Loop through particle pairs and find old relative momentum. 
  180 DO 200 I1M=NBE(IBE-1)+1,NBE(IBE)-1    
      I1=K(I1M,1)   
      DO 200 I2M=I1M+1,NBE(IBE) 
      I2=K(I2M,1)   
      Q2OLD=MAX(0.,(P(I1,4)+P(I2,4))**2-(P(I1,1)+P(I2,1))**2-(P(I1,2)+  
     &P(I2,2))**2-(P(I1,3)+P(I2,3))**2-(P(I1,5)+P(I2,5))**2)    
      QOLD=SQRT(Q2OLD)  
    
C...Calculate new relative momentum.    
      IF(QOLD.LT.0.5*QDEL) THEN 
        QMOV=QOLD/3.    
      ELSEIF(QOLD.LT.(NBIN-0.1)*QDEL) THEN  
        RBIN=QOLD/QDEL  
        IBIN=int(RBIN)
        RINP=(RBIN**3-IBIN**3)/(3*IBIN*(IBIN+1)+1)  
        QMOV=(BEI(IBIN)+RINP*(BEI(IBIN+1)-BEI(IBIN)))*  
     &  SQRT(Q2OLD+PMHQ**2)/Q2OLD   
      ELSE  
        QMOV=BEI(NBIN)*SQRT(Q2OLD+PMHQ**2)/Q2OLD    
      ENDIF 
      Q2NEW=Q2OLD*(QOLD/(QOLD+3.*PARJ(92)*QMOV))**(2./3.)   
    
C...Calculate and save shift to be performed on three-momenta.  
      HC1=(P(I1,4)+P(I2,4))**2-(Q2OLD-Q2NEW)    
      HC2=(Q2OLD-Q2NEW)*(P(I1,4)-P(I2,4))**2    
      HA=0.5*(1.-SQRT(HC1*Q2NEW/(HC1*Q2OLD-HC2)))   
      DO 190 J=1,3  
      PD=HA*(P(I2,J)-P(I1,J))   
      P(I1M,J)=P(I1M,J)+PD  
  190 P(I2M,J)=P(I2M,J)-PD  
  200 CONTINUE  
  210 CONTINUE  
    
C...Shift momenta and recalculate energies. 
      DO 230 IM=NBE(0)+1,NBE(MIN(9,MSTJ(51)))   
      I=K(IM,1) 
      DO 220 J=1,3  
  220 P(I,J)=P(I,J)+P(IM,J) 
  230 P(I,4)=SQRT(P(I,5)**2+P(I,1)**2+P(I,2)**2+P(I,3)**2)  
    
C...Rescale all momenta for energy conservation.    
      PES=0.    
      PQS=0.    
      DO 240 I=1,N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 240  
      PES=PES+P(I,4)    
      PQS=PQS+P(I,5)**2/P(I,4)  
  240 CONTINUE  
      FAC=(PECM-PQS)/(PES-PQS)  
      DO 260 I=1,N  
      IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 260  
      DO 250 J=1,3  
  250 P(I,J)=FAC*P(I,J) 
      P(I,4)=SQRT(P(I,5)**2+P(I,1)**2+P(I,2)**2+P(I,3)**2)  
  260 CONTINUE  
    
C...Boost back to correct reference frame.  
      CALL LUDBRB(0,0,0.,0.,DPS(1)/DPS(4),DPS(2)/DPS(4),DPS(3)/DPS(4))  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION ULMASS(KF)   
    
C...Purpose: to give the mass of a particle/parton. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
    
C...Reset variables. Compressed code.   
      ULMASS=0. 
      KFA=IABS(KF)  
      KC=LUCOMP(KF) 
      IF(KC.EQ.0) RETURN    
      PARF(106)=PMAS(6,1)   
      PARF(107)=PMAS(7,1)   
      PARF(108)=PMAS(8,1)   
    
C...Guarantee use of constituent masses for internal checks.    
      IF((MSTJ(93).EQ.1.OR.MSTJ(93).EQ.2).AND.KFA.LE.10) THEN   
        ULMASS=PARF(100+KFA)    
        IF(MSTJ(93).EQ.2) ULMASS=MAX(0.,ULMASS-PARF(121))   
    
C...Masses that can be read directly off table. 
      ELSEIF(KFA.LE.100.OR.KC.LE.80.OR.KC.GT.100) THEN  
        ULMASS=PMAS(KC,1)   
    
C...Find constituent partons and their masses.  
      ELSE  
        KFLA=MOD(KFA/1000,10)   
        KFLB=MOD(KFA/100,10)    
        KFLC=MOD(KFA/10,10) 
        KFLS=MOD(KFA,10)    
        KFLR=MOD(KFA/10000,10)  
        PMA=PARF(100+KFLA)  
        PMB=PARF(100+KFLB)  
        PMC=PARF(100+KFLC)  
    
C...Construct masses for various meson, diquark and baryon cases.   
        IF(KFLA.EQ.0.AND.KFLR.EQ.0.AND.KFLS.LE.3) THEN  
cms...... initialize to something at first to avoid compiler warning
          PMSPL=-3./(PMA*PMB)
          IF(KFLS.EQ.1) PMSPL=-3./(PMB*PMC) 
          IF(KFLS.GE.3) PMSPL=1./(PMB*PMC)  
          ULMASS=PARF(111)+PMB+PMC+PARF(113)*PARF(101)**2*PMSPL 
        ELSEIF(KFLA.EQ.0) THEN  
          KMUL=2    
          IF(KFLS.EQ.1) KMUL=3  
          IF(KFLR.EQ.2) KMUL=4  
          IF(KFLS.EQ.5) KMUL=5  
          ULMASS=PARF(113+KMUL)+PMB+PMC 
        ELSEIF(KFLC.EQ.0) THEN
cms...... initialize to something at first to avoid compiler warning
          PMSPL=-3./(PMA*PMB)
          IF(KFLS.EQ.1) PMSPL=-3./(PMA*PMB) 
          IF(KFLS.EQ.3) PMSPL=1./(PMA*PMB)  
          ULMASS=2.*PARF(112)/3.+PMA+PMB+PARF(114)*PARF(101)**2*PMSPL   
          IF(MSTJ(93).EQ.1) ULMASS=PMA+PMB  
          IF(MSTJ(93).EQ.2) ULMASS=MAX(0.,ULMASS-PARF(122)- 
     &    2.*PARF(112)/3.)  
        ELSE    
          IF(KFLS.EQ.2.AND.KFLA.EQ.KFLB) THEN   
            PMSPL=1./(PMA*PMB)-2./(PMA*PMC)-2./(PMB*PMC)    
          ELSEIF(KFLS.EQ.2.AND.KFLB.GE.KFLC) THEN   
            PMSPL=-2./(PMA*PMB)-2./(PMA*PMC)+1./(PMB*PMC)   
          ELSEIF(KFLS.EQ.2) THEN    
            PMSPL=-3./(PMB*PMC) 
          ELSE  
            PMSPL=1./(PMA*PMB)+1./(PMA*PMC)+1./(PMB*PMC)    
          ENDIF 
          ULMASS=PARF(112)+PMA+PMB+PMC+PARF(114)*PARF(101)**2*PMSPL 
        ENDIF   
      ENDIF 
    
C...Optional mass broadening according to truncated Breit-Wigner    
C...(either in m or in m^2).    
      IF(MSTJ(24).GE.1.AND.PMAS(KC,2).GT.1E-4) THEN 
        IF(MSTJ(24).EQ.1.OR.(MSTJ(24).EQ.2.AND.KFA.GT.100)) THEN    
          ULMASS=ULMASS+0.5*PMAS(KC,2)*TAN((2.*RLU(0)-1.)*  
     &    ATAN(2.*PMAS(KC,3)/PMAS(KC,2)))   
        ELSE    
          PM0=ULMASS    
          PMLOW=ATAN((MAX(0.,PM0-PMAS(KC,3))**2-PM0**2)/    
     &    (PM0*PMAS(KC,2))) 
          PMUPP=ATAN((PM0+PMAS(KC,3))**2-PM0**2)/(PM0*PMAS(KC,2))   
          ULMASS=SQRT(MAX(0.,PM0**2+PM0*PMAS(KC,2)*TAN(PMLOW+   
     &    (PMUPP-PMLOW)*RLU(0))))   
        ENDIF   
      ENDIF 
      MSTJ(93)=0    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUNAME(KF,CHAU)    
    
C...Purpose: to give the particle/parton name as a character string.    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT4/CHAF(500)   
      CHARACTER CHAF*8  
      SAVE /LUDAT4/ 
      CHARACTER CHAU*16 
    
C...Initial values. Charge. Subdivide code. 
      CHAU=' '  
      KFA=IABS(KF)  
      KC=LUCOMP(KF) 
      IF(KC.EQ.0) RETURN    
      KQ=LUCHGE(KF) 
      KFLA=MOD(KFA/1000,10) 
      KFLB=MOD(KFA/100,10)  
      KFLC=MOD(KFA/10,10)   
      KFLS=MOD(KFA,10)  
      KFLR=MOD(KFA/10000,10)    
    
C...Read out root name and spin for simple particle.    
      IF(KFA.LE.100.OR.(KFA.GT.100.AND.KC.GT.100)) THEN 
        CHAU=CHAF(KC)   
        LEN=0   
        DO 100 LEM=1,8  
  100   IF(CHAU(LEM:LEM).NE.' ') LEN=LEM    
    
C...Construct root name for diquark. Add on spin.   
      ELSEIF(KFLC.EQ.0) THEN    
        CHAU(1:2)=CHAF(KFLA)(1:1)//CHAF(KFLB)(1:1)  
        IF(KFLS.EQ.1) CHAU(3:4)='_0'    
        IF(KFLS.EQ.3) CHAU(3:4)='_1'    
        LEN=4   
    
C...Construct root name for heavy meson. Add on spin and heavy flavour. 
      ELSEIF(KFLA.EQ.0) THEN    
        IF(KFLB.EQ.5) CHAU(1:1)='B' 
        IF(KFLB.EQ.6) CHAU(1:1)='T' 
        IF(KFLB.EQ.7) CHAU(1:1)='L' 
        IF(KFLB.EQ.8) CHAU(1:1)='H' 
        LEN=1   
        IF(KFLR.EQ.0.AND.KFLS.EQ.1) THEN    
        ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.3) THEN    
          CHAU(2:2)='*' 
          LEN=2 
        ELSEIF(KFLR.EQ.1.AND.KFLS.EQ.3) THEN    
          CHAU(2:3)='_1'    
          LEN=3 
        ELSEIF(KFLR.EQ.1.AND.KFLS.EQ.1) THEN    
          CHAU(2:4)='*_0'   
          LEN=4 
        ELSEIF(KFLR.EQ.2) THEN  
          CHAU(2:4)='*_1'   
          LEN=4 
        ELSEIF(KFLS.EQ.5) THEN  
          CHAU(2:4)='*_2'   
          LEN=4 
        ENDIF   
        IF(KFLC.GE.3.AND.KFLR.EQ.0.AND.KFLS.LE.3) THEN  
          CHAU(LEN+1:LEN+2)='_'//CHAF(KFLC)(1:1)    
          LEN=LEN+2 
        ELSEIF(KFLC.GE.3) THEN  
          CHAU(LEN+1:LEN+1)=CHAF(KFLC)(1:1) 
          LEN=LEN+1 
        ENDIF   
    
C...Construct root name and spin for heavy baryon.  
      ELSE  
        IF(KFLB.LE.2.AND.KFLC.LE.2) THEN    
          CHAU='Sigma ' 
          IF(KFLC.GT.KFLB) CHAU='Lambda'    
          IF(KFLS.EQ.4) CHAU='Sigma*'   
          LEN=5 
          IF(CHAU(6:6).NE.' ') LEN=6    
        ELSEIF(KFLB.LE.2.OR.KFLC.LE.2) THEN 
          CHAU='Xi '    
          IF(KFLA.GT.KFLB.AND.KFLB.GT.KFLC) CHAU='Xi''' 
          IF(KFLS.EQ.4) CHAU='Xi*'  
          LEN=2 
          IF(CHAU(3:3).NE.' ') LEN=3    
        ELSE    
          CHAU='Omega ' 
          IF(KFLA.GT.KFLB.AND.KFLB.GT.KFLC) CHAU='Omega'''  
          IF(KFLS.EQ.4) CHAU='Omega*'   
          LEN=5 
          IF(CHAU(6:6).NE.' ') LEN=6    
        ENDIF   
    
C...Add on heavy flavour content for heavy baryon.  
        CHAU(LEN+1:LEN+2)='_'//CHAF(KFLA)(1:1)  
        LEN=LEN+2   
        IF(KFLB.GE.KFLC.AND.KFLC.GE.4) THEN 
          CHAU(LEN+1:LEN+2)=CHAF(KFLB)(1:1)//CHAF(KFLC)(1:1)    
          LEN=LEN+2 
        ELSEIF(KFLB.GE.KFLC.AND.KFLB.GE.4) THEN 
          CHAU(LEN+1:LEN+1)=CHAF(KFLB)(1:1) 
          LEN=LEN+1 
        ELSEIF(KFLC.GT.KFLB.AND.KFLB.GE.4) THEN 
          CHAU(LEN+1:LEN+2)=CHAF(KFLC)(1:1)//CHAF(KFLB)(1:1)    
          LEN=LEN+2 
        ELSEIF(KFLC.GT.KFLB.AND.KFLC.GE.4) THEN 
          CHAU(LEN+1:LEN+1)=CHAF(KFLC)(1:1) 
          LEN=LEN+1 
        ENDIF   
      ENDIF 
    
C...Add on bar sign for antiparticle (where necessary). 
      IF(KF.GT.0.OR.LEN.EQ.0) THEN  
      ELSEIF(KFA.GT.10.AND.KFA.LE.40.AND.KQ.NE.0) THEN  
      ELSEIF(KFA.EQ.89.OR.(KFA.GE.91.AND.KFA.LE.99)) THEN   
      ELSEIF(KFA.GT.100.AND.KFLA.EQ.0.AND.KQ.NE.0) THEN 
      ELSEIF(MSTU(15).LE.1) THEN    
        CHAU(LEN+1:LEN+1)='~'   
        LEN=LEN+1   
      ELSE  
        CHAU(LEN+1:LEN+3)='bar' 
        LEN=LEN+3   
      ENDIF 
    
C...Add on charge where applicable (conventional cases skipped).    
      IF(KQ.EQ.6) CHAU(LEN+1:LEN+2)='++'    
      IF(KQ.EQ.-6) CHAU(LEN+1:LEN+2)='--'   
      IF(KQ.EQ.3) CHAU(LEN+1:LEN+1)='+' 
      IF(KQ.EQ.-3) CHAU(LEN+1:LEN+1)='-'    
      IF(KQ.EQ.0.AND.(KFA.LE.22.OR.LEN.EQ.0)) THEN  
      ELSEIF(KQ.EQ.0.AND.(KFA.GE.81.AND.KFA.LE.100)) THEN   
      ELSEIF(KFA.GT.100.AND.KFLA.EQ.0.AND.KFLB.EQ.KFLC.AND. 
     &KFLB.NE.1) THEN   
      ELSEIF(KQ.EQ.0) THEN  
        CHAU(LEN+1:LEN+1)='0'   
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION LUCHGE(KF)   
    
C...Purpose: to give three times the charge for a particle/parton.  
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
    
C...Initial values. Simple case of direct readout.  
      LUCHGE=0  
      KFA=IABS(KF)  
      KC=LUCOMP(KFA)    
      IF(KC.EQ.0) THEN  
      ELSEIF(KFA.LE.100.OR.KC.LE.80.OR.KC.GT.100) THEN  
        LUCHGE=KCHG(KC,1)   
    
C...Construction from quark content for heavy meson, diquark, baryon.   
      ELSEIF(MOD(KFA/1000,10).EQ.0) THEN    
        LUCHGE=(KCHG(MOD(KFA/100,10),1)-KCHG(MOD(KFA/10,10),1))*    
     &  (-1)**MOD(KFA/100,10)   
      ELSEIF(MOD(KFA/10,10).EQ.0) THEN  
        LUCHGE=KCHG(MOD(KFA/1000,10),1)+KCHG(MOD(KFA/100,10),1) 
      ELSE  
        LUCHGE=KCHG(MOD(KFA/1000,10),1)+KCHG(MOD(KFA/100,10),1)+    
     &  KCHG(MOD(KFA/10,10),1)  
      ENDIF 
    
C...Add on correct sign.    
      LUCHGE=LUCHGE*ISIGN(1,KF) 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION LUCOMP(KF)   
    
C...Purpose: to compress the standard KF codes for use in mass and decay    
C...arrays; also to check whether a given code actually is defined. 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
    
C...Subdivide KF code into constituent pieces.  
      LUCOMP=0  
      KFA=IABS(KF)  
      KFLA=MOD(KFA/1000,10) 
      KFLB=MOD(KFA/100,10)  
      KFLC=MOD(KFA/10,10)   
      KFLS=MOD(KFA,10)  
      KFLR=MOD(KFA/10000,10)    
    
C...Simple cases: direct translation or special codes.  
      IF(KFA.EQ.0.OR.KFA.GE.100000) THEN    
      ELSEIF(KFA.LE.100) THEN   
        LUCOMP=KFA  
        IF(KF.LT.0.AND.KCHG(KFA,3).EQ.0) LUCOMP=0   
      ELSEIF(KFLS.EQ.0) THEN    
        IF(KF.EQ.130) LUCOMP=221    
        IF(KF.EQ.310) LUCOMP=222    
        IF(KFA.EQ.210) LUCOMP=281   
        IF(KFA.EQ.2110) LUCOMP=282  
        IF(KFA.EQ.2210) LUCOMP=283  
    
C...Mesons. 
      ELSEIF(KFA-10000*KFLR.LT.1000) THEN   
        IF(KFLB.EQ.0.OR.KFLB.EQ.9.OR.KFLC.EQ.0.OR.KFLC.EQ.9) THEN   
        ELSEIF(KFLB.LT.KFLC) THEN   
        ELSEIF(KF.LT.0.AND.KFLB.EQ.KFLC) THEN   
        ELSEIF(KFLB.EQ.KFLC) THEN   
          IF(KFLR.EQ.0.AND.KFLS.EQ.1) THEN  
            LUCOMP=110+KFLB 
          ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.3) THEN  
            LUCOMP=130+KFLB 
          ELSEIF(KFLR.EQ.1.AND.KFLS.EQ.3) THEN  
            LUCOMP=150+KFLB 
          ELSEIF(KFLR.EQ.1.AND.KFLS.EQ.1) THEN  
            LUCOMP=170+KFLB 
          ELSEIF(KFLR.EQ.2.AND.KFLS.EQ.3) THEN  
            LUCOMP=190+KFLB 
          ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.5) THEN  
            LUCOMP=210+KFLB 
          ENDIF 
        ELSEIF(KFLB.LE.5.AND.KFLC.LE.3) THEN    
          IF(KFLR.EQ.0.AND.KFLS.EQ.1) THEN  
            LUCOMP=100+((KFLB-1)*(KFLB-2))/2+KFLC   
          ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.3) THEN  
            LUCOMP=120+((KFLB-1)*(KFLB-2))/2+KFLC   
          ELSEIF(KFLR.EQ.1.AND.KFLS.EQ.3) THEN  
            LUCOMP=140+((KFLB-1)*(KFLB-2))/2+KFLC   
          ELSEIF(KFLR.EQ.1.AND.KFLS.EQ.1) THEN  
            LUCOMP=160+((KFLB-1)*(KFLB-2))/2+KFLC   
          ELSEIF(KFLR.EQ.2.AND.KFLS.EQ.3) THEN  
            LUCOMP=180+((KFLB-1)*(KFLB-2))/2+KFLC   
          ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.5) THEN  
            LUCOMP=200+((KFLB-1)*(KFLB-2))/2+KFLC   
          ENDIF 
        ELSEIF((KFLS.EQ.1.AND.KFLR.LE.1).OR.(KFLS.EQ.3.AND.KFLR.LE.2).  
     &  OR.(KFLS.EQ.5.AND.KFLR.EQ.0)) THEN  
          LUCOMP=80+KFLB    
        ENDIF   
    
C...Diquarks.   
      ELSEIF((KFLR.EQ.0.OR.KFLR.EQ.1).AND.KFLC.EQ.0) THEN   
        IF(KFLS.NE.1.AND.KFLS.NE.3) THEN    
        ELSEIF(KFLA.EQ.9.OR.KFLB.EQ.0.OR.KFLB.EQ.9) THEN    
        ELSEIF(KFLA.LT.KFLB) THEN   
        ELSEIF(KFLS.EQ.1.AND.KFLA.EQ.KFLB) THEN 
        ELSE    
          LUCOMP=90 
        ENDIF   
    
C...Spin 1/2 baryons.   
      ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.2) THEN  
        IF(KFLA.EQ.9.OR.KFLB.EQ.0.OR.KFLB.EQ.9.OR.KFLC.EQ.9) THEN   
        ELSEIF(KFLA.LE.KFLC.OR.KFLA.LT.KFLB) THEN   
        ELSEIF(KFLA.GE.6.OR.KFLB.GE.4.OR.KFLC.GE.4) THEN    
          LUCOMP=80+KFLA    
        ELSEIF(KFLB.LT.KFLC) THEN   
          LUCOMP=300+((KFLA+1)*KFLA*(KFLA-1))/6+(KFLC*(KFLC-1))/2+KFLB  
        ELSE    
          LUCOMP=330+((KFLA+1)*KFLA*(KFLA-1))/6+(KFLB*(KFLB-1))/2+KFLC  
        ENDIF   
    
C...Spin 3/2 baryons.   
      ELSEIF(KFLR.EQ.0.AND.KFLS.EQ.4) THEN  
        IF(KFLA.EQ.9.OR.KFLB.EQ.0.OR.KFLB.EQ.9.OR.KFLC.EQ.9) THEN   
        ELSEIF(KFLA.LT.KFLB.OR.KFLB.LT.KFLC) THEN   
        ELSEIF(KFLA.GE.6.OR.KFLB.GE.4) THEN 
          LUCOMP=80+KFLA    
        ELSE    
          LUCOMP=360+((KFLA+1)*KFLA*(KFLA-1))/6+(KFLB*(KFLB-1))/2+KFLC  
        ENDIF   
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUERRM(MERR,CHMESS)    
    
C...Purpose: to inform user of errors in program execution. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      CHARACTER CHMESS*(*)  

      write (6,*) 'merr,chmess=',merr,chmess
    
C...Write first few warnings, then be silent.   
      IF(MERR.LE.10) THEN   
        MSTU(27)=MSTU(27)+1 
        MSTU(28)=MERR   
        IF(MSTU(25).EQ.1.AND.MSTU(27).LE.MSTU(26)) WRITE(MSTU(11),1000) 
     &  MERR,MSTU(31),CHMESS    
    
C...Write first few errors, then be silent or stop program. 
      ELSEIF(MERR.LE.20) THEN   
        MSTU(23)=MSTU(23)+1 
        MSTU(24)=MERR-10    
        IF(MSTU(21).GE.1.AND.MSTU(23).LE.MSTU(22)) WRITE(MSTU(11),1100) 
     &  MERR-10,MSTU(31),CHMESS 
        IF(MSTU(21).GE.2.AND.MSTU(23).GT.MSTU(22)) THEN 
          WRITE(MSTU(11),1100) MERR-10,MSTU(31),CHMESS  
          WRITE(MSTU(11),1200)  
          IF(MERR.NE.17) CALL LULIST(2) 
          STOP  
        ENDIF   
    
C...Stop program in case of irreparable error.  
      ELSE  
        WRITE(MSTU(11),1300) MERR-20,MSTU(31),CHMESS    
        STOP    
      ENDIF 
    
C...Formats for output. 
 1000 FORMAT(/5X,'Advisory warning type',I2,' given after',I6,  
     &' LUEXEC calls:'/5X,A)    
 1100 FORMAT(/5X,'Error type',I2,' has occured after',I6,   
     &' LUEXEC calls:'/5X,A)    
 1200 FORMAT(5X,'Execution will be stopped after listing of last ', 
     &'event!') 
 1300 FORMAT(/5X,'Fatal error type',I2,' has occured after',I6, 
     &' LUEXEC calls:'/5X,A/5X,'Execution will now be stopped!')    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION ULALPS(Q2)   
    
C...Purpose: to give the value of alpha_strong. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
    
C...Constant alpha_strong trivial.  
      IF(MSTU(111).LE.0) THEN   
        ULALPS=PARU(111)    
        MSTU(118)=MSTU(112) 
        PARU(117)=0.    
        PARU(118)=PARU(111) 
        RETURN  
      ENDIF 
    
C...Find effective Q2, number of flavours and Lambda.   
      Q2EFF=Q2  
      IF(MSTU(115).GE.2) Q2EFF=MAX(Q2,PARU(114))    
      NF=MSTU(112)  
      ALAM2=PARU(112)**2    
  100 IF(NF.GT.MAX(2,MSTU(113))) THEN   
        Q2THR=PARU(113)*PMAS(NF,1)**2   
        IF(Q2EFF.LT.Q2THR) THEN 
          NF=NF-1   
          ALAM2=ALAM2*(Q2THR/ALAM2)**(2./(33.-2.*NF))   
          GOTO 100  
        ENDIF   
      ENDIF 
  110 IF(NF.LT.MIN(8,MSTU(114))) THEN   
        Q2THR=PARU(113)*PMAS(NF+1,1)**2 
        IF(Q2EFF.GT.Q2THR) THEN 
          NF=NF+1   
          ALAM2=ALAM2*(ALAM2/Q2THR)**(2./(33.-2.*NF))   
          GOTO 110  
        ENDIF   
      ENDIF 
      IF(MSTU(115).EQ.1) Q2EFF=Q2EFF+ALAM2  
      PARU(117)=SQRT(ALAM2) 
    
C...Evaluate first or second order alpha_strong.    
      B0=(33.-2.*NF)/6. 
      ALGQ=LOG(Q2EFF/ALAM2) 
      IF(MSTU(111).EQ.1) THEN   
        ULALPS=PARU(2)/(B0*ALGQ)    
      ELSE  
        B1=(153.-19.*NF)/6. 
        ULALPS=PARU(2)/(B0*ALGQ)*(1.-B1*LOG(ALGQ)/(B0**2*ALGQ)) 
      ENDIF 
      MSTU(118)=NF  
      PARU(118)=ULALPS  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION ULANGL(X,Y)  
    
C...Purpose: to reconstruct an angle from given x and y coordinates.    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
    
      ULANGL=0. 
      R=SQRT(X**2+Y**2) 
      IF(R.LT.1E-20) RETURN 
      IF(ABS(X)/R.LT.0.8) THEN  
        ULANGL=SIGN(ACOS(X/R),Y)    
      ELSE  
        ULANGL=ASIN(Y/R)    
        IF(X.LT.0..AND.ULANGL.GE.0.) THEN   
          ULANGL=PARU(1)-ULANGL 
        ELSEIF(X.LT.0.) THEN    
          ULANGL=-PARU(1)-ULANGL    
        ENDIF   
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION RLU(IDUM)    
    
C...Purpose: to generate random numbers uniformly distributed between   
C...0 and 1, excluding the endpoints.   
      COMMON/LUDATR/MRLU(6),RRLU(100)   
      SAVE /LUDATR/ 
      EQUIVALENCE (MRLU1,MRLU(1)),(MRLU2,MRLU(2)),(MRLU3,MRLU(3)),  
     &(MRLU4,MRLU(4)),(MRLU5,MRLU(5)),(MRLU6,MRLU(6)),  
     &(RRLU98,RRLU(98)),(RRLU99,RRLU(99)),(RRLU00,RRLU(100))    
    
C...Initialize generation from given seed.  
      IF(MRLU2.EQ.0) THEN   
        IJ=MOD(MRLU1/30082,31329)   
        KL=MOD(MRLU1,30082) 
        I=MOD(IJ/177,177)+2 
        J=MOD(IJ,177)+2 
        K=MOD(KL/169,178)+1 
        L=MOD(KL,169)   
        DO 110 II=1,97  
        S=0.    
        T=0.5   
        DO 100 JJ=1,24  
        M=MOD(MOD(I*J,179)*K,179)   
        I=J 
        J=K 
        K=M 
        L=MOD(53*L+1,169)   
        IF(MOD(L*M,64).GE.32) S=S+T 
  100   T=0.5*T 
  110   RRLU(II)=S  
        TWOM24=1.   
        DO 120 I24=1,24 
  120   TWOM24=0.5*TWOM24   
        RRLU98=362436.*TWOM24   
        RRLU99=7654321.*TWOM24  
        RRLU00=16777213.*TWOM24 
        MRLU2=1 
        MRLU3=0 
        MRLU4=97    
        MRLU5=33    
      ENDIF 
    
C...Generate next random number.    
  130 RUNI=RRLU(MRLU4)-RRLU(MRLU5)  
      IF(RUNI.LT.0.) RUNI=RUNI+1.   
      RRLU(MRLU4)=RUNI  
      MRLU4=MRLU4-1 
      IF(MRLU4.EQ.0) MRLU4=97   
      MRLU5=MRLU5-1 
      IF(MRLU5.EQ.0) MRLU5=97   
      RRLU98=RRLU98-RRLU99  
      IF(RRLU98.LT.0.) RRLU98=RRLU98+RRLU00 
      RUNI=RUNI-RRLU98  
      IF(RUNI.LT.0.) RUNI=RUNI+1.   
      IF(RUNI.LE.0.OR.RUNI.GE.1.) GOTO 130  
    
C...Update counters. Random number to output.   
      MRLU3=MRLU3+1 
      IF(MRLU3.EQ.1000000000) THEN  
        MRLU2=MRLU2+1   
        MRLU3=0 
      ENDIF 
      RLU=RUNI  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUROBO(THE,PHI,BEX,BEY,BEZ)    
    
C...Purpose: to perform rotations and boosts.   
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      DIMENSION ROT(3,3),PR(3),VR(3),DP(4),DV(4)    
    
C...Find range of rotation/boost. Convert boost to double precision.    
      IMIN=1    
      IF(MSTU(1).GT.0) IMIN=MSTU(1) 
      IMAX=N    
      IF(MSTU(2).GT.0) IMAX=MSTU(2) 
      DBX=dble(BEX)
      DBY=dble(BEY)
      DBZ=dble(BEZ)
      GOTO 100  
    
C...Entry for specific range and double precision boost.    
      ENTRY LUDBRB(IMI,IMA,THE,PHI,DBEX,DBEY,DBEZ)  
      IMIN=IMI  
      IF(IMIN.LE.0) IMIN=1  
      IMAX=IMA  
      IF(IMAX.LE.0) IMAX=N  
      DBX=DBEX  
      DBY=DBEY  
      DBZ=DBEZ  
    
C...Check range of rotation/boost.  
  100 IF(IMIN.GT.MSTU(4).OR.IMAX.GT.MSTU(4)) THEN   
        CALL LUERRM(11,'(LUROBO:) range outside LUJETS memory') 
        RETURN  
      ENDIF 
    
C...Rotate, typically from z axis to direction (theta,phi). 
clin-5/2012:
c      IF(THE**2+PHI**2.GT.1E-20) THEN   
      IF((THE**2+PHI**2).GT.1E-20) THEN   
        ROT(1,1)=COS(THE)*COS(PHI)  
        ROT(1,2)=-SIN(PHI)  
        ROT(1,3)=SIN(THE)*COS(PHI)  
        ROT(2,1)=COS(THE)*SIN(PHI)  
        ROT(2,2)=COS(PHI)   
        ROT(2,3)=SIN(THE)*SIN(PHI)  
        ROT(3,1)=-SIN(THE)  
        ROT(3,2)=0. 
        ROT(3,3)=COS(THE)   
        DO 130 I=IMIN,IMAX  
        IF(K(I,1).LE.0) GOTO 130    
        DO 110 J=1,3    
        PR(J)=P(I,J)    
  110   VR(J)=V(I,J)    
        DO 120 J=1,3    
        P(I,J)=ROT(J,1)*PR(1)+ROT(J,2)*PR(2)+ROT(J,3)*PR(3) 
  120   V(I,J)=ROT(J,1)*VR(1)+ROT(J,2)*VR(2)+ROT(J,3)*VR(3) 
  130   CONTINUE    
      ENDIF 
    
C...Boost, typically from rest to momentum/energy=beta. 
clin-5/2012:
c      IF(DBX**2+DBY**2+DBZ**2.GT.1E-20) THEN    
      IF((DBX**2+DBY**2+DBZ**2).GT.1D-20) THEN    
        DB=SQRT(DBX**2+DBY**2+DBZ**2)   
        IF(DB.GT.0.99999999D0) THEN 
C...Rescale boost vector if too close to unity. 
          CALL LUERRM(3,'(LUROBO:) boost vector too large') 
          DBX=DBX*(0.99999999D0/DB) 
          DBY=DBY*(0.99999999D0/DB) 
          DBZ=DBZ*(0.99999999D0/DB) 
          DB=0.99999999D0   
        ENDIF   
        DGA=1D0/SQRT(1D0-DB**2) 
        DO 150 I=IMIN,IMAX  
        IF(K(I,1).LE.0) GOTO 150    
        DO 140 J=1,4    
        DP(J)=dble(P(I,J))
  140   DV(J)=dble(V(I,J))
        DBP=DBX*DP(1)+DBY*DP(2)+DBZ*DP(3)   
        DGABP=DGA*(DGA*DBP/(1D0+DGA)+DP(4)) 
        P(I,1)=sngl(DP(1)+DGABP*DBX)
        P(I,2)=sngl(DP(2)+DGABP*DBY) 
        P(I,3)=sngl(DP(3)+DGABP*DBZ) 
        P(I,4)=sngl(DGA*(DP(4)+DBP)) 
        DBV=DBX*DV(1)+DBY*DV(2)+DBZ*DV(3)   
        DGABV=DGA*(DGA*DBV/(1D0+DGA)+DV(4)) 
        V(I,1)=sngl(DV(1)+DGABV*DBX) 
        V(I,2)=sngl(DV(2)+DGABV*DBY) 
        V(I,3)=sngl(DV(3)+DGABV*DBZ) 
        V(I,4)=sngl(DGA*(DV(4)+DBV))
  150   CONTINUE    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
C THIS SUBROUTINE IS ONLY FOR THE USE OF HIJING TO ROTATE OR BOOST
C        THE FOUR MOMENTUM ONLY
C*********************************************************************
    
      SUBROUTINE HIROBO(THE,PHI,BEX,BEY,BEZ)    
    
C...Purpose: to perform rotations and boosts.   
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      DIMENSION ROT(3,3),PR(3),DP(4)
cms      VR(3),DV(4)    
    
C...Find range of rotation/boost. Convert boost to double precision.    
      IMIN=1    
      IF(MSTU(1).GT.0) IMIN=MSTU(1) 
      IMAX=N    
      IF(MSTU(2).GT.0) IMAX=MSTU(2) 
      DBX=dble(BEX)
      DBY=dble(BEY) 
      DBZ=dble(BEZ)  
    
C...Check range of rotation/boost.  
      IF(IMIN.GT.MSTU(4).OR.IMAX.GT.MSTU(4)) THEN   
        CALL LUERRM(11,'(LUROBO:) range outside LUJETS memory') 
        RETURN  
      ENDIF 
    
C...Rotate, typically from z axis to direction (theta,phi). 
clin-5/2012:
c      IF(THE**2+PHI**2.GT.1E-20) THEN   
      IF((THE**2+PHI**2).GT.1E-20) THEN   
        ROT(1,1)=COS(THE)*COS(PHI)  
        ROT(1,2)=-SIN(PHI)  
        ROT(1,3)=SIN(THE)*COS(PHI)  
        ROT(2,1)=COS(THE)*SIN(PHI)  
        ROT(2,2)=COS(PHI)   
        ROT(2,3)=SIN(THE)*SIN(PHI)  
        ROT(3,1)=-SIN(THE)  
        ROT(3,2)=0. 
        ROT(3,3)=COS(THE)   
        DO 130 I=IMIN,IMAX  
        IF(K(I,1).LE.0) GOTO 130    
        DO 110 J=1,3    
  110   PR(J)=P(I,J)   
        DO 120 J=1,3    
  120   P(I,J)=ROT(J,1)*PR(1)+ROT(J,2)*PR(2)+ROT(J,3)*PR(3) 
  130   CONTINUE    
      ENDIF 
    
C...Boost, typically from rest to momentum/energy=beta. 
clin-5/2012:
c      IF(DBX**2+DBY**2+DBZ**2.GT.1E-20) THEN    
      IF((DBX**2+DBY**2+DBZ**2).GT.1D-20) THEN    
        DB=SQRT(DBX**2+DBY**2+DBZ**2)   
        IF(DB.GT.0.99999999D0) THEN 
C...Rescale boost vector if too close to unity. 
          CALL LUERRM(3,'(LUROBO:) boost vector too large') 
          DBX=DBX*(0.99999999D0/DB) 
          DBY=DBY*(0.99999999D0/DB) 
          DBZ=DBZ*(0.99999999D0/DB) 
          DB=0.99999999D0   
        ENDIF   
        DGA=1D0/SQRT(1D0-DB**2) 
        DO 150 I=IMIN,IMAX  
        IF(K(I,1).LE.0) GOTO 150    
        DO 140 J=1,4    
  140   DP(J)=dble(P(I,J))
        DBP=DBX*DP(1)+DBY*DP(2)+DBZ*DP(3)   
        DGABP=DGA*(DGA*DBP/(1D0+DGA)+DP(4)) 
        P(I,1)=sngl(DP(1)+DGABP*DBX)
        P(I,2)=sngl(DP(2)+DGABP*DBY) 
        P(I,3)=sngl(DP(3)+DGABP*DBZ) 
        P(I,4)=sngl(DGA*(DP(4)+DBP)) 
  150   CONTINUE    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LUEDIT(MEDIT)  
    
C...Purpose: to perform global manipulations on the event record,   
C...in particular to exclude unstable or undetectable partons/particles.    
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      DIMENSION NS(2),PTS(2),PLS(2) 
    
C...Remove unwanted partons/particles.  
      IF((MEDIT.GE.0.AND.MEDIT.LE.3).OR.MEDIT.EQ.5) THEN    
        IMAX=N  
        IF(MSTU(2).GT.0) IMAX=MSTU(2)   
        I1=MAX(1,MSTU(1))-1 
        DO 110 I=MAX(1,MSTU(1)),IMAX    
        IF(K(I,1).EQ.0.OR.K(I,1).GT.20) GOTO 110    
        IF(MEDIT.EQ.1) THEN 
          IF(K(I,1).GT.10) GOTO 110 
        ELSEIF(MEDIT.EQ.2) THEN 
          IF(K(I,1).GT.10) GOTO 110 
          KC=LUCOMP(K(I,2)) 
          IF(KC.EQ.0.OR.KC.EQ.12.OR.KC.EQ.14.OR.KC.EQ.16.OR.KC.EQ.18)   
     &    GOTO 110  
        ELSEIF(MEDIT.EQ.3) THEN 
          IF(K(I,1).GT.10) GOTO 110 
          KC=LUCOMP(K(I,2)) 
          IF(KC.EQ.0) GOTO 110  
          IF(KCHG(KC,2).EQ.0.AND.LUCHGE(K(I,2)).EQ.0) GOTO 110  
        ELSEIF(MEDIT.EQ.5) THEN 
          IF(K(I,1).EQ.13.OR.K(I,1).EQ.14) GOTO 110 
          KC=LUCOMP(K(I,2)) 
          IF(KC.EQ.0) GOTO 110  
          IF(K(I,1).GE.11.AND.KCHG(KC,2).EQ.0) GOTO 110 
        ENDIF   
    
C...Pack remaining partons/particles. Origin no longer known.   
        I1=I1+1 
        DO 100 J=1,5    
        K(I1,J)=K(I,J)  
        P(I1,J)=P(I,J)  
  100   V(I1,J)=V(I,J)  
        K(I1,3)=0   
  110   CONTINUE    
        N=I1    
    
C...Selective removal of class of entries. New position of retained.    
      ELSEIF(MEDIT.GE.11.AND.MEDIT.LE.15) THEN  
        I1=0    
        DO 120 I=1,N    
        K(I,3)=MOD(K(I,3),MSTU(5))  
        IF(MEDIT.EQ.11.AND.K(I,1).LT.0) GOTO 120    
        IF(MEDIT.EQ.12.AND.K(I,1).EQ.0) GOTO 120    
        IF(MEDIT.EQ.13.AND.(K(I,1).EQ.11.OR.K(I,1).EQ.12.OR.    
     &  K(I,1).EQ.15).AND.K(I,2).NE.94) GOTO 120    
        IF(MEDIT.EQ.14.AND.(K(I,1).EQ.13.OR.K(I,1).EQ.14.OR.    
     &  K(I,2).EQ.94)) GOTO 120 
        IF(MEDIT.EQ.15.AND.K(I,1).GE.21) GOTO 120   
        I1=I1+1 
        K(I,3)=K(I,3)+MSTU(5)*I1    
  120   CONTINUE    
    
C...Find new event history information and replace old. 
        DO 140 I=1,N    
        IF(K(I,1).LE.0.OR.K(I,1).GT.20.OR.K(I,3)/MSTU(5).EQ.0) GOTO 140 
        ID=I    
  130   IM=MOD(K(ID,3),MSTU(5)) 
        IF(MEDIT.EQ.13.AND.IM.GT.0.AND.IM.LE.N) THEN    
          IF((K(IM,1).EQ.11.OR.K(IM,1).EQ.12.OR.K(IM,1).EQ.15).AND. 
     &    K(IM,2).NE.94) THEN   
            ID=IM   
            GOTO 130    
          ENDIF 
        ELSEIF(MEDIT.EQ.14.AND.IM.GT.0.AND.IM.LE.N) THEN    
          IF(K(IM,1).EQ.13.OR.K(IM,1).EQ.14.OR.K(IM,2).EQ.94) THEN  
            ID=IM   
            GOTO 130    
          ENDIF 
        ENDIF   
        K(I,3)=MSTU(5)*(K(I,3)/MSTU(5)) 
        IF(IM.NE.0) K(I,3)=K(I,3)+K(IM,3)/MSTU(5)   
        IF(K(I,1).NE.3.AND.K(I,1).NE.13.AND.K(I,1).NE.14) THEN  
          IF(K(I,4).GT.0.AND.K(I,4).LE.MSTU(4)) K(I,4)= 
     &    K(K(I,4),3)/MSTU(5)   
          IF(K(I,5).GT.0.AND.K(I,5).LE.MSTU(4)) K(I,5)= 
     &    K(K(I,5),3)/MSTU(5)   
        ELSE    
          KCM=MOD(K(I,4)/MSTU(5),MSTU(5))   
          IF(KCM.GT.0.AND.KCM.LE.MSTU(4)) KCM=K(KCM,3)/MSTU(5)  
          KCD=MOD(K(I,4),MSTU(5))   
          IF(KCD.GT.0.AND.KCD.LE.MSTU(4)) KCD=K(KCD,3)/MSTU(5)  
          K(I,4)=MSTU(5)**2*(K(I,4)/MSTU(5)**2)+MSTU(5)*KCM+KCD 
          KCM=MOD(K(I,5)/MSTU(5),MSTU(5))   
          IF(KCM.GT.0.AND.KCM.LE.MSTU(4)) KCM=K(KCM,3)/MSTU(5)  
          KCD=MOD(K(I,5),MSTU(5))   
          IF(KCD.GT.0.AND.KCD.LE.MSTU(4)) KCD=K(KCD,3)/MSTU(5)  
          K(I,5)=MSTU(5)**2*(K(I,5)/MSTU(5)**2)+MSTU(5)*KCM+KCD 
        ENDIF   
  140   CONTINUE    
    
C...Pack remaining entries. 
        I1=0    
        DO 160 I=1,N    
        IF(K(I,3)/MSTU(5).EQ.0) GOTO 160    
        I1=I1+1 
        DO 150 J=1,5    
        K(I1,J)=K(I,J)  
        P(I1,J)=P(I,J)  
  150   V(I1,J)=V(I,J)  
        K(I1,3)=MOD(K(I1,3),MSTU(5))    
  160   CONTINUE    
        N=I1    
    
C...Save top entries at bottom of LUJETS commonblock.   
      ELSEIF(MEDIT.EQ.21) THEN  
        IF(2*N.GE.MSTU(4)) THEN 
          CALL LUERRM(11,'(LUEDIT:) no more memory left in LUJETS') 
          RETURN    
        ENDIF   
        DO 170 I=1,N    
        DO 170 J=1,5    
        K(MSTU(4)-I,J)=K(I,J)   
        P(MSTU(4)-I,J)=P(I,J)   
  170   V(MSTU(4)-I,J)=V(I,J)   
        MSTU(32)=N  
    
C...Restore bottom entries of commonblock LUJETS to top.    
      ELSEIF(MEDIT.EQ.22) THEN  
        DO 180 I=1,MSTU(32) 
        DO 180 J=1,5    
        K(I,J)=K(MSTU(4)-I,J)   
        P(I,J)=P(MSTU(4)-I,J)   
  180   V(I,J)=V(MSTU(4)-I,J)   
        N=MSTU(32)  
    
C...Mark primary entries at top of commonblock LUJETS as untreated. 
      ELSEIF(MEDIT.EQ.23) THEN  
        I1=0    
        DO 190 I=1,N    
        KH=K(I,3)   
        IF(KH.GE.1) THEN    
          IF(K(KH,1).GT.20) KH=0    
        ENDIF   
        IF(KH.NE.0) GOTO 200    
        I1=I1+1 
  190   IF(K(I,1).GT.10.AND.K(I,1).LE.20) K(I,1)=K(I,1)-10  
  200   N=I1    
    
C...Place largest axis along z axis and second largest in xy plane. 
      ELSEIF(MEDIT.EQ.31.OR.MEDIT.EQ.32) THEN   
        CALL LUDBRB(1,N+MSTU(3),0.,-ULANGL(P(MSTU(61),1),   
     &  P(MSTU(61),2)),0D0,0D0,0D0) 
        CALL LUDBRB(1,N+MSTU(3),-ULANGL(P(MSTU(61),3),  
     &  P(MSTU(61),1)),0.,0D0,0D0,0D0)  
        CALL LUDBRB(1,N+MSTU(3),0.,-ULANGL(P(MSTU(61)+1,1), 
     &  P(MSTU(61)+1,2)),0D0,0D0,0D0)   
        IF(MEDIT.EQ.31) RETURN  
    
C...Rotate to put slim jet along +z axis.   
        DO 210 IS=1,2   
        NS(IS)=0    
        PTS(IS)=0.  
  210   PLS(IS)=0.  
        DO 220 I=1,N    
        IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 220    
        IF(MSTU(41).GE.2) THEN  
          KC=LUCOMP(K(I,2)) 
          IF(KC.EQ.0.OR.KC.EQ.12.OR.KC.EQ.14.OR.KC.EQ.16.OR.    
     &    KC.EQ.18) GOTO 220    
          IF(MSTU(41).GE.3.AND.KCHG(KC,2).EQ.0.AND.LUCHGE(K(I,2)).EQ.0) 
     &    GOTO 220  
        ENDIF   
        IS=int(2.-SIGN(0.5,P(I,3)))
        NS(IS)=NS(IS)+1 
        PTS(IS)=PTS(IS)+SQRT(P(I,1)**2+P(I,2)**2)   
  220   CONTINUE    
        IF(NS(1)*PTS(2)**2.LT.NS(2)*PTS(1)**2)  
     &  CALL LUDBRB(1,N+MSTU(3),PARU(1),0.,0D0,0D0,0D0) 
    
C...Rotate to put second largest jet into -z,+x quadrant.   
        DO 230 I=1,N    
        IF(P(I,3).GE.0.) GOTO 230   
        IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 230    
        IF(MSTU(41).GE.2) THEN  
          KC=LUCOMP(K(I,2)) 
          IF(KC.EQ.0.OR.KC.EQ.12.OR.KC.EQ.14.OR.KC.EQ.16.OR.    
     &    KC.EQ.18) GOTO 230    
          IF(MSTU(41).GE.3.AND.KCHG(KC,2).EQ.0.AND.LUCHGE(K(I,2)).EQ.0) 
     &    GOTO 230  
        ENDIF   
        IS=int(2.-SIGN(0.5,P(I,1)))
        PLS(IS)=PLS(IS)-P(I,3)  
  230   CONTINUE    
        IF(PLS(2).GT.PLS(1)) CALL LUDBRB(1,N+MSTU(3),0.,PARU(1),    
     &  0D0,0D0,0D0)    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE LULIST(MLIST)  
    
C...Purpose: to give program heading, or list an event, or particle 
C...data, or current parameter values.  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      CHARACTER CHAP*16,CHAC*16,CHAN*16,CHAD(5)*16,CHMO(12)*3,CHDL(7)*4 
      DIMENSION PS(6)   
      DATA CHMO/'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep',  
     &'Oct','Nov','Dec'/,CHDL/'(())',' ','()','!!','<>','==','(==)'/    
    
C...Initialization printout: version number and date of last change.    
C      IF(MLIST.EQ.0.OR.MSTU(12).EQ.1) THEN  
C        WRITE(MSTU(11),1000) MSTU(181),MSTU(182),MSTU(185), 
C     &  CHMO(MSTU(184)),MSTU(183)   
C        MSTU(12)=0  
C        IF(MLIST.EQ.0) RETURN   
C      ENDIF 
    
C...List event data, including additional lines after N.    
      IF(MLIST.GE.1.AND.MLIST.LE.3) THEN    
        IF(MLIST.EQ.1) WRITE(MSTU(11),1100) 
        IF(MLIST.EQ.2) WRITE(MSTU(11),1200) 
        IF(MLIST.EQ.3) WRITE(MSTU(11),1300) 
        LMX=12  
        IF(MLIST.GE.2) LMX=16   
        ISTR=0  
        IMAX=N  
        IF(MSTU(2).GT.0) IMAX=MSTU(2)   
        DO 120 I=MAX(1,MSTU(1)),MAX(IMAX,N+MAX(0,MSTU(3)))  
        IF((I.GT.IMAX.AND.I.LE.N).OR.K(I,1).LT.0) GOTO 120  
    
C...Get particle name, pad it and check it is not too long. 
        CALL LUNAME(K(I,2),CHAP)    
        LEN=0   
        DO 100 LEM=1,16 
  100   IF(CHAP(LEM:LEM).NE.' ') LEN=LEM    
        MDL=(K(I,1)+19)/10  
        LDL=0   
        IF(MDL.EQ.2.OR.MDL.GE.8) THEN   
          CHAC=CHAP 
          IF(LEN.GT.LMX) CHAC(LMX:LMX)='?'  
        ELSE    
          LDL=1 
          IF(MDL.EQ.1.OR.MDL.EQ.7) LDL=2    
          IF(LEN.EQ.0) THEN 
            CHAC=CHDL(MDL)(1:2*LDL)//' '    
          ELSE  
            CHAC=CHDL(MDL)(1:LDL)//CHAP(1:MIN(LEN,LMX-2*LDL))// 
     &      CHDL(MDL)(LDL+1:2*LDL)//' ' 
            IF(LEN+2*LDL.GT.LMX) CHAC(LMX:LMX)='?'  
          ENDIF 
        ENDIF   
    
C...Add information on string connection.   
        IF(K(I,1).EQ.1.OR.K(I,1).EQ.2.OR.K(I,1).EQ.11.OR.K(I,1).EQ.12)  
     &  THEN    
          KC=LUCOMP(K(I,2)) 
          KCC=0 
          IF(KC.NE.0) KCC=KCHG(KC,2)    
          IF(KCC.NE.0.AND.ISTR.EQ.0) THEN   
            ISTR=1  
            IF(LEN+2*LDL+3.LE.LMX) CHAC(LMX-1:LMX-1)='A'    
          ELSEIF(KCC.NE.0.AND.(K(I,1).EQ.2.OR.K(I,1).EQ.12)) THEN   
            IF(LEN+2*LDL+3.LE.LMX) CHAC(LMX-1:LMX-1)='I'    
          ELSEIF(KCC.NE.0) THEN 
            ISTR=0  
            IF(LEN+2*LDL+3.LE.LMX) CHAC(LMX-1:LMX-1)='V'    
          ENDIF 
        ENDIF   
    
C...Write data for particle/jet.    
        IF(MLIST.EQ.1.AND.ABS(P(I,4)).LT.9999.) THEN    
          WRITE(MSTU(11),1400) I,CHAC(1:12),(K(I,J1),J1=1,3),   
     &    (P(I,J2),J2=1,5)  
        ELSEIF(MLIST.EQ.1.AND.ABS(P(I,4)).LT.99999.) THEN   
          WRITE(MSTU(11),1500) I,CHAC(1:12),(K(I,J1),J1=1,3),   
     &    (P(I,J2),J2=1,5)  
        ELSEIF(MLIST.EQ.1) THEN 
          WRITE(MSTU(11),1600) I,CHAC(1:12),(K(I,J1),J1=1,3),   
     &    (P(I,J2),J2=1,5)  
        ELSEIF(MSTU(5).EQ.10000.AND.(K(I,1).EQ.3.OR.K(I,1).EQ.13.OR.    
     &  K(I,1).EQ.14)) THEN 
          WRITE(MSTU(11),1700) I,CHAC,(K(I,J1),J1=1,3), 
     &    K(I,4)/100000000,MOD(K(I,4)/10000,10000),MOD(K(I,4),10000),   
     &    K(I,5)/100000000,MOD(K(I,5)/10000,10000),MOD(K(I,5),10000),   
     &    (P(I,J2),J2=1,5)  
        ELSE    
          WRITE(MSTU(11),1800) I,CHAC,(K(I,J1),J1=1,5),(P(I,J2),J2=1,5) 
        ENDIF   
        IF(MLIST.EQ.3) WRITE(MSTU(11),1900) (V(I,J),J=1,5)  
    
C...Insert extra separator lines specified by user. 
        IF(MSTU(70).GE.1) THEN  
          ISEP=0    
          DO 110 J=1,MIN(10,MSTU(70))   
  110     IF(I.EQ.MSTU(70+J)) ISEP=1    
          IF(ISEP.EQ.1.AND.MLIST.EQ.1) WRITE(MSTU(11),2000) 
          IF(ISEP.EQ.1.AND.MLIST.GE.2) WRITE(MSTU(11),2100) 
        ENDIF   
  120   CONTINUE    
    
C...Sum of charges and momenta. 
        DO 130 J=1,6    
  130   PS(J)=PLU(0,J)  
        IF(MLIST.EQ.1.AND.ABS(PS(4)).LT.9999.) THEN 
          WRITE(MSTU(11),2200) PS(6),(PS(J),J=1,5)  
        ELSEIF(MLIST.EQ.1.AND.ABS(PS(4)).LT.99999.) THEN    
          WRITE(MSTU(11),2300) PS(6),(PS(J),J=1,5)  
        ELSEIF(MLIST.EQ.1) THEN 
          WRITE(MSTU(11),2400) PS(6),(PS(J),J=1,5)  
        ELSE    
          WRITE(MSTU(11),2500) PS(6),(PS(J),J=1,5)  
        ENDIF   
    
C...Give simple list of KF codes defined in program.    
      ELSEIF(MLIST.EQ.11) THEN  
        WRITE(MSTU(11),2600)    
        DO 140 KF=1,40  
        CALL LUNAME(KF,CHAP)    
        CALL LUNAME(-KF,CHAN)   
        IF(CHAP.NE.' '.AND.CHAN.EQ.' ') WRITE(MSTU(11),2700) KF,CHAP    
  140   IF(CHAN.NE.' ') WRITE(MSTU(11),2700) KF,CHAP,-KF,CHAN   
        DO 150 KFLS=1,3,2   
        DO 150 KFLA=1,8 
        DO 150 KFLB=1,KFLA-(3-KFLS)/2   
        KF=1000*KFLA+100*KFLB+KFLS  
        CALL LUNAME(KF,CHAP)    
        CALL LUNAME(-KF,CHAN)   
  150   WRITE(MSTU(11),2700) KF,CHAP,-KF,CHAN   
        DO 170 KMUL=0,5 
        KFLS=3  
        IF(KMUL.EQ.0.OR.KMUL.EQ.3) KFLS=1   
        IF(KMUL.EQ.5) KFLS=5    
        KFLR=0  
        IF(KMUL.EQ.2.OR.KMUL.EQ.3) KFLR=1   
        IF(KMUL.EQ.4) KFLR=2    
        DO 170 KFLB=1,8 
        DO 160 KFLC=1,KFLB-1    
        KF=10000*KFLR+100*KFLB+10*KFLC+KFLS 
        CALL LUNAME(KF,CHAP)    
        CALL LUNAME(-KF,CHAN)   
  160   WRITE(MSTU(11),2700) KF,CHAP,-KF,CHAN   
        KF=10000*KFLR+110*KFLB+KFLS 
        CALL LUNAME(KF,CHAP)    
  170   WRITE(MSTU(11),2700) KF,CHAP    
        KF=130  
        CALL LUNAME(KF,CHAP)    
        WRITE(MSTU(11),2700) KF,CHAP    
        KF=310  
        CALL LUNAME(KF,CHAP)    
        WRITE(MSTU(11),2700) KF,CHAP    
        DO 190 KFLSP=1,3    
        KFLS=2+2*(KFLSP/3)  
        DO 190 KFLA=1,8 
        DO 190 KFLB=1,KFLA  
        DO 180 KFLC=1,KFLB  
        IF(KFLSP.EQ.1.AND.(KFLA.EQ.KFLB.OR.KFLB.EQ.KFLC)) GOTO 180  
        IF(KFLSP.EQ.2.AND.KFLA.EQ.KFLC) GOTO 180    
        IF(KFLSP.EQ.1) KF=1000*KFLA+100*KFLC+10*KFLB+KFLS   
        IF(KFLSP.GE.2) KF=1000*KFLA+100*KFLB+10*KFLC+KFLS   
        CALL LUNAME(KF,CHAP)    
        CALL LUNAME(-KF,CHAN)   
        WRITE(MSTU(11),2700) KF,CHAP,-KF,CHAN   
  180   CONTINUE    
  190   CONTINUE    
    
C...List parton/particle data table. Check whether to be listed.    
      ELSEIF(MLIST.EQ.12) THEN  
        WRITE(MSTU(11),2800)    
        MSTJ24=MSTJ(24) 
        MSTJ(24)=0  
        KFMAX=20883 
        IF(MSTU(2).NE.0) KFMAX=MSTU(2)  
        DO 220 KF=MAX(1,MSTU(1)),KFMAX  
        KC=LUCOMP(KF)   
        IF(KC.EQ.0) GOTO 220    
        IF(MSTU(14).EQ.0.AND.KF.GT.100.AND.KC.LE.100) GOTO 220  
        IF(MSTU(14).GT.0.AND.KF.GT.100.AND.MAX(MOD(KF/1000,10), 
     &  MOD(KF/100,10)).GT.MSTU(14)) GOTO 220   
    
C...Find particle name and mass. Print information. 
        CALL LUNAME(KF,CHAP)    
        IF(KF.LE.100.AND.CHAP.EQ.' '.AND.MDCY(KC,2).EQ.0) GOTO 220  
        CALL LUNAME(-KF,CHAN)   
        PM=ULMASS(KF)   
        WRITE(MSTU(11),2900) KF,KC,CHAP,CHAN,KCHG(KC,1),KCHG(KC,2), 
     &  KCHG(KC,3),PM,PMAS(KC,2),PMAS(KC,3),PMAS(KC,4),MDCY(KC,1)   
    
C...Particle decay: channel number, branching ration, matrix element,   
C...decay products. 
        IF(KF.GT.100.AND.KC.LE.100) GOTO 220    
        DO 210 IDC=MDCY(KC,2),MDCY(KC,2)+MDCY(KC,3)-1   
        DO 200 J=1,5    
  200   CALL LUNAME(KFDP(IDC,J),CHAD(J))    
  210   WRITE(MSTU(11),3000) IDC,MDME(IDC,1),MDME(IDC,2),BRAT(IDC), 
     &  (CHAD(J),J=1,5) 
  220   CONTINUE    
        MSTJ(24)=MSTJ24 
    
C...List parameter value table. 
      ELSEIF(MLIST.EQ.13) THEN  
        WRITE(MSTU(11),3100)    
        DO 230 I=1,200  
  230   WRITE(MSTU(11),3200) I,MSTU(I),PARU(I),MSTJ(I),PARJ(I),PARF(I)  
      ENDIF 
    
C...Format statements for output on unit MSTU(11) (by default 6).   
clin 1000 FORMAT(///20X,'The Lund Monte Carlo - JETSET version ',I1,'.',I1/ 
clin     &20X,'**  Last date of change:  ',I2,1X,A3,1X,I4,'  **'/)  
 1100 FORMAT(///28X,'Event listing (summary)'//4X,'I  particle/jet KS', 
     &5X,'KF orig    p_x      p_y      p_z       E        m'/)  
 1200 FORMAT(///28X,'Event listing (standard)'//4X,'I  particle/jet',   
     &'  K(I,1)   K(I,2) K(I,3)     K(I,4)      K(I,5)       P(I,1)',   
     &'       P(I,2)       P(I,3)       P(I,4)       P(I,5)'/)  
 1300 FORMAT(///28X,'Event listing (with vertices)'//4X,'I  particle/j',    
     &'et  K(I,1)   K(I,2) K(I,3)     K(I,4)      K(I,5)       P(I,1)', 
     &'       P(I,2)       P(I,3)       P(I,4)       P(I,5)'/73X,   
     &'V(I,1)       V(I,2)       V(I,3)       V(I,4)       V(I,5)'/)    
 1400 FORMAT(1X,I4,2X,A12,1X,I2,1X,I6,1X,I4,5F9.3)  
 1500 FORMAT(1X,I4,2X,A12,1X,I2,1X,I6,1X,I4,5F9.2)  
 1600 FORMAT(1X,I4,2X,A12,1X,I2,1X,I6,1X,I4,5F9.1)  
 1700 FORMAT(1X,I4,2X,A16,1X,I3,1X,I8,2X,I4,2(3X,I1,2I4),5F13.5)    
 1800 FORMAT(1X,I4,2X,A16,1X,I3,1X,I8,2X,I4,2(3X,I9),5F13.5)    
 1900 FORMAT(66X,5(1X,F12.3))   
 2000 FORMAT(1X,78('='))    
 2100 FORMAT(1X,130('='))   
 2200 FORMAT(19X,'sum:',F6.2,5X,5F9.3)  
 2300 FORMAT(19X,'sum:',F6.2,5X,5F9.2)  
 2400 FORMAT(19X,'sum:',F6.2,5X,5F9.1)  
 2500 FORMAT(19X,'sum charge:',F6.2,3X,'sum momentum and inv. mass:',   
     &5F13.5)   
 2600 FORMAT(///20X,'List of KF codes in program'/) 
 2700 FORMAT(4X,I6,4X,A16,6X,I6,4X,A16) 
 2800 FORMAT(///30X,'Particle/parton data table'//5X,'KF',5X,'KC',4X,   
     &'particle',8X,'antiparticle',6X,'chg  col  anti',8X,'mass',7X,    
     &'width',7X,'w-cut',5X,'lifetime',1X,'decay'/11X,'IDC',1X,'on/off',    
     &1X,'ME',3X,'Br.rat.',4X,'decay products') 
 2900 FORMAT(/1X,I6,3X,I4,4X,A16,A16,3I5,1X,F12.5,2(1X,F11.5),  
     &2X,F12.5,3X,I2)   
 3000 FORMAT(10X,I4,2X,I3,2X,I3,2X,F8.5,4X,5A16)    
 3100 FORMAT(///20X,'Parameter value table'//4X,'I',3X,'MSTU(I)',   
     &8X,'PARU(I)',3X,'MSTJ(I)',8X,'PARJ(I)',8X,'PARF(I)')  
 3200 FORMAT(1X,I4,1X,I9,1X,F14.5,1X,I9,1X,F14.5,1X,F14.5)  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION PLU(I,J) 
    
C...Purpose: to provide various real-valued event related data. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      DIMENSION PSUM(4) 
    
C...Set default value. For I = 0 sum of momenta or charges, 
C...or invariant mass of system.    
      PLU=0.    
      IF(I.LT.0.OR.I.GT.MSTU(4).OR.J.LE.0) THEN 
      ELSEIF(I.EQ.0.AND.J.LE.4) THEN    
        DO 100 I1=1,N   
  100   IF(K(I1,1).GT.0.AND.K(I1,1).LE.10) PLU=PLU+P(I1,J)  
      ELSEIF(I.EQ.0.AND.J.EQ.5) THEN    
        DO 110 J1=1,4   
        PSUM(J1)=0. 
        DO 110 I1=1,N   
  110   IF(K(I1,1).GT.0.AND.K(I1,1).LE.10) PSUM(J1)=PSUM(J1)+P(I1,J1)   
        PLU=SQRT(MAX(0.,PSUM(4)**2-PSUM(1)**2-PSUM(2)**2-PSUM(3)**2))   
      ELSEIF(I.EQ.0.AND.J.EQ.6) THEN    
        DO 120 I1=1,N   
  120   IF(K(I1,1).GT.0.AND.K(I1,1).LE.10) PLU=PLU+LUCHGE(K(I1,2))/3.   
      ELSEIF(I.EQ.0) THEN   
    
C...Direct readout of P matrix. 
      ELSEIF(J.LE.5) THEN   
        PLU=P(I,J)  
    
C...Charge, total momentum, transverse momentum, transverse mass.   
      ELSEIF(J.LE.12) THEN  
        IF(J.EQ.6) PLU=LUCHGE(K(I,2))/3.    
        IF(J.EQ.7.OR.J.EQ.8) PLU=P(I,1)**2+P(I,2)**2+P(I,3)**2  
        IF(J.EQ.9.OR.J.EQ.10) PLU=P(I,1)**2+P(I,2)**2   
        IF(J.EQ.11.OR.J.EQ.12) PLU=P(I,5)**2+P(I,1)**2+P(I,2)**2    
        IF(J.EQ.8.OR.J.EQ.10.OR.J.EQ.12) PLU=SQRT(PLU)  
    
C...Theta and phi angle in radians or degrees.  
      ELSEIF(J.LE.16) THEN  
        IF(J.LE.14) PLU=ULANGL(P(I,3),SQRT(P(I,1)**2+P(I,2)**2))    
        IF(J.GE.15) PLU=ULANGL(P(I,1),P(I,2))   
        IF(J.EQ.14.OR.J.EQ.16) PLU=PLU*180./PARU(1) 
    
C...True rapidity, rapidity with pion mass, pseudorapidity. 
      ELSEIF(J.LE.19) THEN  
        PMR=0.  
        IF(J.EQ.17) PMR=P(I,5)  
        IF(J.EQ.18) PMR=ULMASS(211) 
        PR=MAX(1E-20,PMR**2+P(I,1)**2+P(I,2)**2)    
        PLU=SIGN(LOG(MIN((SQRT(PR+P(I,3)**2)+ABS(P(I,3)))/SQRT(PR), 
     &  1E20)),P(I,3))  
    
C...Energy and momentum fractions (only to be used in CM frame).    
      ELSEIF(J.LE.25) THEN  
        IF(J.EQ.20) PLU=2.*SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2)/PARU(21) 
        IF(J.EQ.21) PLU=2.*P(I,3)/PARU(21)  
        IF(J.EQ.22) PLU=2.*SQRT(P(I,1)**2+P(I,2)**2)/PARU(21)   
        IF(J.EQ.23) PLU=2.*P(I,4)/PARU(21)  
        IF(J.EQ.24) PLU=(P(I,4)+P(I,3))/PARU(21)    
        IF(J.EQ.25) PLU=(P(I,4)-P(I,3))/PARU(21)    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      BLOCK DATA LUDATA 
    
C...Purpose: to give default values to parameters and particle and  
C...decay data. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/LUDAT4/CHAF(500)   
      CHARACTER CHAF*8  
      SAVE /LUDAT4/ 
      COMMON/LUDATR/MRLU(6),RRLU(100)   
      SAVE /LUDATR/ 
    
C...LUDAT1, containing status codes and most parameters.    
      DATA MSTU/    
     &    0,    0,    0, 9000,10000,  500, 2000,    0,    0,    2,  
     1    6,    1,    1,    0,    1,    1,    0,    0,    0,    0,  
     2    2,   10,    0,    0,    1,   10,    0,    0,    0,    0,  
     3    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     4    2,    2,    1,    4,    2,    1,    1,    0,    0,    0,  
     5   25,   24,    0,    1,    0,    0,    0,    0,    0,    0,  
     6    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     7  40*0,   
     1    1,    5,    3,    5,    0,    0,    0,    0,    0,    0,  
     2  60*0,   
     8    7,    2, 1989,   11,   25,    0,    0,    0,    0,    0,  
     9    0,    0,    0,    0,    0,    0,    0,    0,    0,    0/  
      DATA PARU/    
     & 3.1415927, 6.2831854, 0.1973, 5.068, 0.3894, 2.568,   4*0.,  
     1 0.001, 0.09, 0.01,  0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     2   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     3   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     4  2.0,  1.0, 0.25,  2.5, 0.05,   0.,   0., 0.0001, 0.,   0.,  
     5  2.5,  1.5,  7.0,  1.0,  0.5,  2.0,  3.2,   0.,   0.,   0.,  
     6  40*0.,  
     & 0.0072974, 0.230, 0., 0., 0.,   0.,   0.,   0.,   0.,   0.,  
     1 0.20, 0.25,  1.0,  4.0,   0.,   0.,   0.,   0.,   0.,   0.,  
     2  1.0,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     3  70*0./  
      DATA MSTJ/    
     &    1,    3,    0,    0,    0,    0,    0,    0,    0,    0,  
     1    1,    2,    0,    1,    0,    0,    0,    0,    0,    0,  
     2    2,    1,    1,    2,    1,    0,    0,    0,    0,    0,  
     3    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     4    1,    2,    4,    2,    5,    0,    1,    0,    0,    0,  
     5    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,  
     6  40*0,   
     &    5,    2,    7,    5,    1,    1,    0,    2,    0,    1,  
     1    0,    0,    0,    0,    1,    1,    0,    0,    0,    0,  
     2  80*0/   
      DATA PARJ/    
     & 0.10, 0.30, 0.40, 0.05, 0.50, 0.50, 0.50,   0.,   0.,   0.,  
     1 0.50, 0.60, 0.75,   0.,   0.,   0.,   0.,  1.0,  1.0,   0.,  
     2 0.35,  1.0,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     3 0.10,  1.0,  0.8,  1.5,  0.8,  2.0,  0.2,  2.5,  0.6,  2.5,  
     4  0.5,  0.9,  0.5,  0.9,  0.5,   0.,   0.,   0.,   0.,   0.,  
     5 0.77, 0.77, 0.77,   0.,   0.,   0.,   0.,   0.,  1.0,   0.,  
     6  4.5,  0.7,  0., 0.003,  0.5,  0.5,   0.,   0.,   0.,   0.,  
     7  10., 1000., 100., 1000., 0.,   0.,   0.,   0.,   0.,   0.,  
     8  0.4,  1.0,  1.0,   0.,  10.,  10.,   0.,   0.,   0.,   0.,  
     9 0.02,  1.0,  0.2,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     &   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     1   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     2  1.5,  0.5, 91.2, 2.40, 0.02,  2.0,  1.0, 0.25,0.002,   0.,  
     3   0.,   0.,   0.,   0., 0.01, 0.99,   0.,   0.,  0.2,   0.,  
     4  60*0./  
    
C...LUDAT2, with particle data and flavour treatment parameters.    
      DATA (KCHG(I,1),I=   1, 500)/-1,2,-1,2,-1,2,-1,2,2*0,-3,0,-3,0,   
     &-3,0,-3,6*0,3,9*0,3,2*0,3,46*0,2,-1,2,-1,2,3,11*0,3,0,2*3,    
     &0,3,0,3,12*0,3,0,2*3,0,3,0,3,12*0,3,0,2*3,0,3,0,3,12*0,3,0,2*3,0, 
     &3,0,3,12*0,3,0,2*3,0,3,0,3,12*0,3,0,2*3,0,3,0,3,72*0,3,0,3,28*0,  
     &3,2*0,3,8*0,-3,8*0,3,0,-3,0,3,-3,3*0,3,6,0,3,5*0,-3,0,3,-3,0,-3,  
     &4*0,-3,0,3,6,-3,0,3,-3,0,-3,0,3,6,0,3,5*0,-3,0,3,-3,0,-3,114*0/   
      DATA (KCHG(I,2),I=   1, 500)/8*1,12*0,2,68*0,-1,410*0/    
      DATA (KCHG(I,3),I=   1, 500)/8*1,2*0,8*1,5*0,1,9*0,1,2*0,1,2*0,1, 
     &41*0,1,0,7*1,10*0,9*1,11*0,9*1,11*0,9*1,11*0,9*1,11*0,9*1,    
     &11*0,9*1,71*0,3*1,22*0,1,5*0,1,0,2*1,6*0,1,0,2*1,6*0,2*1,0,5*1,   
     &0,6*1,4*0,6*1,4*0,16*1,4*0,6*1,114*0/ 
      DATA (PMAS(I,1),I=   1, 500)/.0099,.0056,.199,1.35,5.,90.,120.,   
     &200.,2*0.,.00051,0.,.1057,0.,1.7841,0.,60.,5*0.,91.2,80.,15., 
     &6*0.,300.,900.,600.,300.,900.,300.,2*0.,5000.,60*0.,.1396,.4977,  
     &.4936,1.8693,1.8645,1.9693,5.2794,5.2776,5.47972,0.,.135,.5488,   
     &.9575,2.9796,9.4,117.99,238.,397.,2*0.,.7669,.8962,.8921, 
     &2.0101,2.0071,2.1127,2*5.3354,5.5068,0.,.77,.782,1.0194,3.0969,   
     &9.4603,118.,238.,397.,2*0.,1.233,2*1.3,2*2.322,2.51,2*5.73,5.97,  
     &0.,1.233,1.17,1.41,3.46,9.875,118.42,238.42,397.42,2*0.,  
     &.983,2*1.429,2*2.272,2.46,2*5.68,5.92,0.,.983,1.,1.4,3.4151,  
     &9.8598,118.4,238.4,397.4,2*0.,1.26,2*1.401,2*2.372,   
     &2.56,2*5.78,6.02,0.,1.26,1.283,1.422,3.5106,9.8919,118.5,238.5,   
     &397.5,2*0.,1.318,2*1.426,2*2.422,2.61,2*5.83,6.07,0.,1.318,1.274, 
     &1.525,3.5563,9.9132,118.45,238.45,397.45,2*0.,2*.4977,    
     &83*0.,1.1156,5*0.,2.2849,0.,2*2.46,6*0.,5.62,0.,2*5.84,6*0.,  
     &.9396,.9383,0.,1.1974,1.1926,1.1894,1.3213,1.3149,0.,2.454,   
     &2.4529,2.4522,2*2.55,2.73,4*0.,3*5.8,2*5.96,6.12,4*0.,1.234,  
     &1.233,1.232,1.231,1.3872,1.3837,1.3828,1.535,1.5318,1.6724,3*2.5, 
     &2*2.63,2.8,4*0.,3*5.81,2*5.97,6.13,114*0./    
      DATA (PMAS(I,2),I=   1, 500)/22*0.,2.4,2.3,88*0.,.0002,.001,  
     &6*0.,.149,.0505,.0513,7*0.,.153,.0085,.0044,7*0.,.15,2*.09,2*.06, 
     &.04,3*.1,0.,.15,.335,.08,2*.01,5*0.,.057,2*.287,2*.06,.04,3*.1,   
     &0.,.057,0.,.25,.0135,6*0.,.4,2*.184,2*.06,.04,3*.1,0.,.4,.025,    
     &.055,.0135,6*0.,.11,.115,.099,2*.06,4*.1,0.,.11,.185,.076,.0026,  
     &146*0.,4*.115,.039,2*.036,.0099,.0091,131*0./ 
      DATA (PMAS(I,3),I=   1, 500)/22*0.,2*20.,88*0.,.002,.005,6*0.,.4, 
     &2*.2,7*0.,.4,.1,.015,7*0.,.25,2*.01,3*.08,2*.2,.12,0.,.25,.2, 
     &.001,2*.02,5*0.,.05,2*.4,3*.08,2*.2,.12,0.,.05,0.,.35,.05,6*0.,   
     &3*.3,2*.08,.06,2*.2,.12,0.,.3,.05,.025,.001,6*0.,.25,4*.12,4*.2,  
     &0.,.25,.17,.2,.01,146*0.,4*.14,.04,2*.035,2*.05,131*0./   
      DATA (PMAS(I,4),I=   1, 500)/12*0.,658650.,0.,.091,68*0.,.1,.43,  
     &15*0.,7803.,0.,3709.,.32,.128,.131,3*.393,84*0.,.004,26*0.,   
     &15540.,26.75,83*0.,78.88,5*0.,.054,0.,2*.13,6*0.,.393,0.,2*.393,  
     &9*0.,44.3,0.,24.,49.1,86.9,6*0.,.13,9*0.,.393,13*0.,24.6,130*0./  
      DATA PARF/    
     &  0.5, 0.25,  0.5, 0.25,   1.,  0.5,   0.,   0.,   0.,   0.,  
     1  0.5,   0.,  0.5,   0.,   1.,   1.,   0.,   0.,   0.,   0.,  
     2  0.5,   0.,  0.5,   0.,   1.,   1.,   0.,   0.,   0.,   0.,  
     3  0.5,   0.,  0.5,   0.,   1.,   1.,   0.,   0.,   0.,   0.,  
     4  0.5,   0.,  0.5,   0.,   1.,   1.,   0.,   0.,   0.,   0.,  
     5  0.5,   0.,  0.5,   0.,   1.,   1.,   0.,   0.,   0.,   0.,  
     6 0.75,  0.5,   0., 0.1667, 0.0833, 0.1667, 0., 0., 0.,   0.,  
     7   0.,   0.,   1., 0.3333, 0.6667, 0.3333, 0., 0., 0.,   0.,  
     8   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     9   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     & 0.325, 0.325, 0.5, 1.6,  5.0,   0.,   0.,   0.,   0.,   0.,  
     1   0., 0.11, 0.16, 0.048, 0.50, 0.45, 0.55, 0.60,  0.,   0.,  
     2  0.2,  0.1,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  
     3  1870*0./    
      DATA ((VCKM(I,J),J=1,4),I=1,4)/   
     1  0.95150,  0.04847,  0.00003,  0.00000,  
     2  0.04847,  0.94936,  0.00217,  0.00000,  
     3  0.00003,  0.00217,  0.99780,  0.00000,  
     4  0.00000,  0.00000,  0.00000,  1.00000/  
    
C...LUDAT3, with particle decay parameters and data.    
      DATA (MDCY(I,1),I=   1, 500)/14*0,1,0,1,5*0,3*1,6*0,1,4*0,1,2*0,  
     &1,42*0,7*1,12*0,1,0,6*1,0,8*1,2*0,9*1,0,8*1,2*0,9*1,0,8*1,2*0,    
     &9*1,0,8*1,2*0,9*1,0,8*1,2*0,9*1,0,8*1,3*0,1,83*0,1,5*0,1,0,2*1,   
     &6*0,1,0,2*1,9*0,5*1,0,6*1,4*0,6*1,4*0,16*1,4*0,6*1,114*0/ 
      DATA (MDCY(I,2),I=   1, 500)/1,9,17,25,33,41,49,57,2*0,65,69,71,  
     &76,78,118,120,125,2*0,127,136,149,166,186,6*0,203,4*0,219,2*0,    
     &227,42*0,236,237,241,250,252,254,256,11*0,276,277,279,285,406,    
     &574,606,607,608,0,609,611,617,623,624,625,626,627,2*0,628,629,    
     &632,635,638,640,641,642,643,0,644,645,650,658,661,670,685,686,    
     &2*0,687,688,693,698,700,702,703,705,707,0,709,710,713,717,718,    
     &719,721,722,2*0,723,726,728,730,734,738,740,744,748,0,752,755,    
     &759,763,765,767,769,770,2*0,771,773,775,777,779,781,784,786,788,  
     &0,791,793,806,810,812,814,816,817,2*0,818,824,835,846,854,862,    
     &867,875,883,0,888,895,903,905,907,909,911,912,2*0,913,921,83*0,   
     &923,5*0,927,0,1001,1002,6*0,1003,0,1004,1005,9*0,1006,1008,1009,  
     &1012,1013,0,1015,1016,1017,1018,1019,1020,4*0,1021,1022,1023, 
     &1024,1025,1026,4*0,1027,1028,1031,1034,1035,1038,1041,1044,1046,  
     &1048,1052,1053,1054,1055,1057,1059,4*0,1060,1061,1062,1063,1064,  
     &1065,114*0/   
      DATA (MDCY(I,3),I=   1, 500)/8*8,2*0,4,2,5,2,40,2,5,2,2*0,9,13,   
     &17,20,17,6*0,16,4*0,8,2*0,9,42*0,1,4,9,3*2,20,11*0,1,2,6,121,168, 
     &32,3*1,0,2,2*6,5*1,2*0,1,3*3,2,4*1,0,1,5,8,3,9,15,2*1,2*0,1,2*5,  
     &2*2,1,3*2,0,1,3,4,2*1,2,2*1,2*0,3,2*2,2*4,2,3*4,0,3,2*4,3*2,2*1,  
     &2*0,5*2,3,2*2,3,0,2,13,4,3*2,2*1,2*0,6,2*11,2*8,5,2*8,5,0,7,8,    
     &4*2,2*1,2*0,8,2,83*0,4,5*0,74,0,2*1,6*0,1,0,2*1,9*0,2,1,3,1,2,0,  
     &6*1,4*0,6*1,4*0,1,2*3,1,3*3,2*2,4,3*1,2*2,1,4*0,6*1,114*0/    
      DATA (MDME(I,1),I=   1,2000)/6*1,-1,7*1,-1,7*1,-1,7*1,-1,7*1,-1,  
     &7*1,-1,85*1,2*-1,7*1,2*-1,3*1,2*-1,6*1,2*-1,6*1,3*-1,3*1,-1,3*1,  
     &-1,3*1,5*-1,3*1,-1,6*1,2*-1,3*1,-1,11*1,2*-1,6*1,2*-1,3*1,-1,3*1, 
     &-1,4*1,2*-1,2*1,-1,488*1,2*0,1275*1/  
      DATA (MDME(I,2),I=   1,2000)/70*102,42,6*102,2*42,2*0,7*41,2*0,   
     &23*41,6*102,45,28*102,8*32,9*0,16*32,4*0,8*32,4*0,32,4*0,8*32,    
     &8*0,4*32,4*0,6*32,3*0,12,2*42,2*11,9*42,6*45,20*46,7*0,34*42, 
     &86*0,2*25,26,24*42,142*0,25,26,0,10*42,19*0,2*13,3*85,0,2,4*0,2,  
     &8*0,2*32,87,88,3*3,0,2*3,0,2*3,0,3,5*0,3,1,0,3,2*0,2*3,3*0,1,4*0, 
     &12,3*0,4*32,2*4,6*0,5*32,2*4,2*45,87,88,30*0,12,32,0,32,87,88,    
     &41*0,12,0,32,0,32,87,88,40*0,12,0,32,0,32,87,88,88*0,12,0,32,0,   
     &32,87,88,2*0,4*42,8*0,14*42,50*0,10*13,2*84,3*85,14*0,84,5*0,85,  
     &974*0/    
      DATA (BRAT(I)  ,I=   1, 525)/70*0.,1.,6*0.,2*.177,.108,.225,.003, 
     &.06,.02,.025,.013,2*.004,.007,.014,2*.002,2*.001,.054,.014,.016,  
     &.005,2*.012,5*.006,.002,2*.001,5*.002,6*0.,1.,28*0.,.143,.111,    
     &.143,.111,.143,.085,2*0.,.03,.058,.03,.058,.03,.058,3*0.,.25,.01, 
     &2*0.,.01,.25,4*0.,.24,5*0.,3*.08,3*0.,.01,.08,.82,5*0.,.09,6*0.,  
     &.143,.111,.143,.111,.143,.085,2*0.,.03,.058,.03,.058,.03,.058,    
     &4*0.,1.,5*0.,4*.215,2*0.,2*.07,0.,1.,2*.08,.76,.08,2*.112,.05,    
     &.476,.08,.14,.01,.015,.005,1.,0.,1.,0.,1.,0.,.25,.01,2*0.,.01,    
     &.25,4*0.,.24,5*0.,3*.08,0.,1.,2*.5,.635,.212,.056,.017,.048,.032, 
     &.035,.03,2*.015,.044,2*.022,9*.001,.035,.03,2*.015,.044,2*.022,   
     &9*.001,.028,.017,.066,.02,.008,2*.006,.003,.001,2*.002,.003,.001, 
     &2*.002,.005,.002,.005,.006,.004,.012,2*.005,.008,2*.005,.037, 
     &.004,.067,2*.01,2*.001,3*.002,.003,8*.002,.005,4*.004,.015,.005,  
     &.027,2*.005,.007,.014,.007,.01,.008,.012,.015,11*.002,3*.004, 
     &.002,.004,6*.002,2*.004,.005,.011,.005,.015,.02,2*.01,3*.004, 
     &5*.002,.015,.02,2*.01,3*.004,5*.002,.038,.048,.082,.06,.028,.021, 
     &2*.005,2*.002,.005,.018,.005,.01,.008,.005,3*.004,.001,3*.003,    
     &.001,2*.002,.003,2*.002,2*.001,.002,.001,.002,.001,.005,4*.003,   
     &.001,2*.002,.003,2*.001,.013,.03,.058,.055,3*.003,2*.01,.007, 
     &.019,4*.005,.015,3*.005,8*.002,3*.001,.002,2*.001,.003,16*.001/   
      DATA (BRAT(I)  ,I= 526, 893)/.019,2*.003,.002,.005,.004,.008, 
     &.003,.006,.003,.01,5*.002,2*.001,2*.002,11*.001,.002,14*.001, 
     &.018,.005,.01,2*.015,.017,4*.015,.017,3*.015,.025,.08,2*.025,.04, 
     &.001,2*.005,.02,.04,2*.06,.04,.01,4*.005,.25,.115,3*1.,.988,.012, 
     &.389,.319,.237,.049,.005,.001,.441,.205,.301,.03,.022,.001,6*1.,  
     &.665,.333,.002,.666,.333,.001,.49,.34,.17,.52,.48,5*1.,.893,.08,  
     &.017,2*.005,.495,.343,3*.043,.019,.013,.001,2*.069,.862,3*.027,   
     &.015,.045,.015,.045,.77,.029,6*.02,5*.05,.115,.015,.5,0.,3*1.,    
     &.28,.14,.313,.157,.11,.28,.14,.313,.157,.11,.667,.333,.667,.333,  
     &1.,.667,.333,.667,.333,2*.5,1.,.333,.334,.333,4*.25,2*1.,.3,.7,   
     &2*1.,.8,2*.1,.667,.333,.667,.333,.6,.3,.067,.033,.6,.3,.067,.033, 
     &2*.5,.6,.3,.067,.033,.6,.3,.067,.033,2*.4,2*.1,.8,2*.1,.52,.26,   
     &2*.11,.62,.31,2*.035,.007,.993,.02,.98,.3,.7,2*1.,2*.5,.667,.333, 
     &.667,.333,.667,.333,.667,.333,2*.35,.3,.667,.333,.667,.333,2*.35, 
     &.3,2*.5,3*.14,.1,.05,4*.08,.028,.027,.028,.027,4*.25,.273,.727,   
     &.35,.65,.3,.7,2*1.,2*.35,.144,.105,.048,.003,.332,.166,.168,.084, 
     &.086,.043,.059,2*.029,2*.002,.332,.166,.168,.084,.086,.043,.059,  
     &2*.029,2*.002,.3,.15,.16,.08,.13,.06,.08,.04,.3,.15,.16,.08,.13,  
     &.06,.08,.04,2*.4,.1,2*.05,.3,.15,.16,.08,.13,.06,.08,.04,.3,.15,  
     &.16,.08,.13,.06,.08,.04,2*.4,.1,2*.05,2*.35,.144,.105,2*.024/ 
      DATA (BRAT(I)  ,I= 894,2000)/.003,.573,.287,.063,.028,2*.021, 
     &.004,.003,2*.5,.15,.85,.22,.78,.3,.7,2*1.,.217,.124,2*.193,   
     &2*.135,.002,.001,.686,.314,.641,.357,2*.001,.018,2*.005,.003, 
     &.002,2*.006,.018,2*.005,.003,.002,2*.006,.005,.025,.015,.006, 
     &2*.005,.004,.005,5*.004,2*.002,2*.004,.003,.002,2*.003,3*.002,    
     &2*.001,.002,2*.001,2*.002,5*.001,4*.003,2*.005,2*.002,2*.001, 
     &2*.002,2*.001,.255,.057,2*.035,.15,2*.075,.03,2*.015,5*1.,.999,   
     &.001,1.,.516,.483,.001,1.,.995,.005,13*1.,.331,.663,.006,.663,    
     &.331,.006,1.,.88,2*.06,.88,2*.06,.88,2*.06,.667,2*.333,.667,.676, 
     &.234,.085,.005,3*1.,4*.5,7*1.,935*0./ 
      DATA (KFDP(I,1),I=   1, 499)/21,22,23,4*-24,25,21,22,23,4*24,25,  
     &21,22,23,4*-24,25,21,22,23,4*24,25,21,22,23,4*-24,25,21,22,23,    
     &4*24,25,21,22,23,4*-24,25,21,22,23,4*24,25,22,23,-24,25,23,24,    
     &-12,22,23,-24,25,23,24,-12,-14,34*16,22,23,-24,25,23,24,-89,22,   
     &23,-24,25,23,24,1,2,3,4,5,6,7,8,21,1,2,3,4,5,6,7,8,11,13,15,17,   
     &37,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,37,4*-1,4*-3,4*-5, 
     &4*-7,-11,-13,-15,-17,1,2,3,4,5,6,7,8,11,13,15,17,21,2*22,23,24,1, 
     &2,3,4,5,6,7,8,11,12,13,14,15,16,17,18,-1,-3,-5,-7,-11,-13,-15,    
     &-17,1,2,3,4,5,6,11,13,15,82,-11,-13,2*2,-12,-14,-16,2*-2,2*-4,-2, 
     &-4,2*89,2*-89,2*89,4*-1,4*-3,4*-5,4*-7,-11,-13,-15,-17,-13,130,   
     &310,-13,3*211,12,14,16*-11,16*-13,-311,-313,-311,-313,-311,-313,  
     &-311,-313,2*111,2*221,2*331,2*113,2*223,2*333,-311,-313,2*-311,   
     &-313,3*-311,-321,-323,-321,2*211,2*213,-213,113,3*213,3*211,  
     &2*213,2*-311,-313,-321,2*-311,-313,-311,-313,4*-311,-321,-323,    
     &2*-321,3*211,213,2*211,213,5*211,213,4*211,3*213,211,213,321,311, 
     &3,2*2,12*-11,12*-13,-321,-323,-321,-323,-311,-313,-311,-313,-311, 
     &-313,-311,-313,-311,-313,-311,-321,-323,-321,-323,211,213,211,    
     &213,111,221,331,113,223,333,221,331,113,223,113,223,113,223,333,  
     &223,333,321,323,321,323,311,313,-321,-323,3*-321,-323,2*-321, 
     &-323,-321,-311,-313,3*-311,-313,2*-311,-313,-321,-323,3*-321/ 
      DATA (KFDP(I,1),I= 500, 873)/-323,2*-321,-311,2*333,211,213,  
     &2*211,2*213,4*211,10*111,-321,-323,5*-321,-323,2*-321,-311,-313,  
     &4*-311,-313,4*-311,-321,-323,2*-321,-323,-321,-313,-311,-313, 
     &-311,211,213,2*211,213,4*211,111,221,113,223,113,223,2*3,-15, 
     &5*-11,5*-13,221,331,333,221,331,333,211,213,211,213,321,323,321,  
     &323,2212,221,331,333,221,2*2,3*0,3*22,111,211,2*22,2*211,111, 
     &3*22,111,3*21,2*0,211,321,3*311,2*321,421,2*411,2*421,431,511,    
     &521,531,2*211,22,211,2*111,321,130,-213,113,213,211,22,111,11,13, 
     &82,11,13,15,1,2,3,4,21,22,11,12,13,14,15,16,1,2,3,4,5,21,22,2*89, 
     &2*0,223,321,311,323,313,2*311,321,313,323,321,421,2*411,421,433,  
     &521,2*511,521,523,513,223,213,113,-213,313,-313,323,-323,82,21,   
     &663,21,2*0,221,213,113,321,2*311,321,421,411,423,413,411,421,413, 
     &423,431,433,521,511,523,513,511,521,513,523,521,511,531,533,221,  
     &213,-213,211,111,321,130,211,111,321,130,443,82,553,21,663,21,    
     &2*0,113,213,323,2*313,323,423,2*413,423,421,411,433,523,2*513,    
     &523,521,511,533,213,-213,10211,10111,-10211,2*221,213,2*113,-213, 
     &2*321,2*311,313,-313,323,-323,443,82,553,21,663,21,2*0,213,113,   
     &221,223,321,211,321,311,323,313,323,313,321,5*311,321,313,323,    
     &313,323,311,4*321,421,411,423,413,423,413,421,2*411,421,413,423,  
     &413,423,411,2*421,411,433,2*431,521,511,523,513,523,513,521/  
      DATA (KFDP(I,1),I= 874,2000)/2*511,521,513,523,513,523,511,2*521, 
     &511,533,2*531,213,-213,221,223,321,130,111,211,111,2*211,321,130, 
     &221,111,321,130,443,82,553,21,663,21,2*0,111,211,-12,12,-14,14,   
     &211,111,211,111,2212,2*2112,-12,7*-11,7*-13,2*2224,2*2212,2*2214, 
     &2*3122,2*3212,2*3214,5*3222,4*3224,2*3322,3324,2*2224,5*2212, 
     &5*2214,2*2112,2*2114,2*3122,2*3212,2*3214,2*3222,2*3224,4*2,3,    
     &2*2,1,2*2,5*0,2112,-12,3122,2212,2112,2212,3*3122,3*4122,4132,    
     &4232,0,3*5122,5132,5232,0,2112,2212,2*2112,2212,2112,2*2212,3122, 
     &3212,3112,3122,3222,3112,3122,3222,3212,3322,3312,3322,3312,3122, 
     &3322,3312,-12,3*4122,2*4132,2*4232,4332,3*5122,5132,5232,5332,    
     &935*0/    
      DATA (KFDP(I,2),I=   1, 496)/3*1,2,4,6,8,1,3*2,1,3,5,7,2,3*3,2,4, 
     &6,8,3,3*4,1,3,5,7,4,3*5,2,4,6,8,5,3*6,1,3,5,7,6,3*7,2,4,6,8,7,    
     &3*8,1,3,5,7,8,2*11,12,11,12,2*11,2*13,14,13,14,13,11,13,-211, 
     &-213,-211,-213,-211,-213,3*-211,-321,-323,-321,-323,2*-321,   
     &4*-211,-213,-211,-213,-211,-213,-211,-213,-211,-213,6*-211,2*15,  
     &16,15,16,15,18,2*17,18,17,18,17,-1,-2,-3,-4,-5,-6,-7,-8,21,-1,-2, 
     &-3,-4,-5,-6,-7,-8,-11,-13,-15,-17,-37,-1,-2,-3,-4,-5,-6,-7,-8,    
     &-11,-12,-13,-14,-15,-16,-17,-18,-37,2,4,6,8,2,4,6,8,2,4,6,8,2,4,  
     &6,8,12,14,16,18,-1,-2,-3,-4,-5,-6,-7,-8,-11,-13,-15,-17,21,22,    
     &2*23,-24,-1,-2,-3,-4,-5,-6,-7,-8,-11,-12,-13,-14,-15,-16,-17,-18, 
     &2,4,6,8,12,14,16,18,-3,-4,-5,-6,-7,-8,-13,-15,-17,-82,12,14,-1,   
     &-3,11,13,15,1,4,3,4,1,3,5,3,6,4,7,5,2,4,6,8,2,4,6,8,2,4,6,8,2,4,  
     &6,8,12,14,16,18,14,2*0,14,111,211,111,-11,-13,16*12,16*14,2*211,  
     &2*213,2*321,2*323,211,213,211,213,211,213,211,213,211,213,211,    
     &213,2*211,213,7*211,213,211,111,211,111,2*211,-213,213,2*113,223, 
     &2*113,221,321,2*311,321,313,4*211,213,113,213,-213,2*211,213,113, 
     &111,221,331,111,113,223,4*113,223,6*211,213,4*211,-321,-311,3*-1, 
     &12*12,12*14,2*211,2*213,2*111,2*221,2*331,2*113,2*223,333,2*321,  
     &2*323,2*-211,2*-213,6*111,4*221,2*331,3*113,2*223,2*-211,2*-213,  
     &113,111,2*211,213,6*211,321,2*211,213,211,2*111,113,2*223,2*321/  
      DATA (KFDP(I,2),I= 497, 863)/323,321,2*311,313,2*311,111,211, 
     &2*-211,-213,-211,-213,-211,-213,3*-211,5*111,2*113,223,113,223,   
     &2*211,213,5*211,213,3*211,213,2*211,2*111,221,113,223,3*321,323,  
     &2*321,323,311,313,311,313,3*211,2*-211,-213,3*-211,4*111,2*113,   
     &2*-1,16,5*12,5*14,3*211,3*213,2*111,2*113,2*-311,2*-313,-2112,    
     &3*321,323,2*-1,3*0,22,11,22,111,-211,211,11,2*-211,111,113,223,   
     &22,111,3*21,2*0,111,-211,111,22,211,111,22,211,111,22,111,5*22,   
     &2*-211,111,-211,2*111,-321,310,211,111,2*-211,221,22,-11,-13,-82, 
     &-11,-13,-15,-1,-2,-3,-4,2*21,-11,-12,-13,-14,-15,-16,-1,-2,-3,-4, 
     &-5,2*21,5,3,2*0,211,-213,113,-211,111,223,211,111,211,111,223,    
     &211,111,-211,2*111,-211,111,211,111,-321,-311,111,-211,111,211,   
     &-311,311,-321,321,-82,21,22,21,2*0,211,111,211,-211,111,211,111,  
     &211,111,211,111,-211,111,-211,3*111,-211,111,-211,111,211,111,    
     &211,111,-321,-311,3*111,-211,211,-211,111,-321,310,-211,111,-321, 
     &310,22,-82,22,21,22,21,2*0,211,111,-211,111,211,111,211,111,-211, 
     &111,321,311,111,-211,111,211,111,-321,-311,111,-211,211,-211,111, 
     &2*211,111,-211,211,111,211,-321,2*-311,-321,-311,311,-321,321,22, 
     &-82,22,21,22,21,2*0,111,3*211,-311,22,-211,111,-211,111,-211,211, 
     &-213,113,223,221,22,211,111,211,111,2*211,213,113,223,221,22,211, 
     &111,211,111,4*211,-211,111,-211,111,-211,211,-211,211,321,311/    
      DATA (KFDP(I,2),I= 864,2000)/2*111,211,-211,111,-211,111,-211,    
     &211,-211,2*211,111,211,111,4*211,-321,-311,2*111,211,-211,211,    
     &111,211,-321,310,22,-211,111,2*-211,-321,310,221,111,-321,310,22, 
     &-82,22,21,22,21,2*0,111,-211,11,-11,13,-13,-211,111,-211,111, 
     &-211,111,22,11,7*12,7*14,-321,-323,-311,-313,-311,-313,211,213,   
     &211,213,211,213,111,221,331,113,223,111,221,113,223,321,323,321,  
     &-211,-213,111,221,331,113,223,111,221,331,113,223,211,213,211,    
     &213,321,323,321,323,321,323,311,313,311,313,2*-1,-3,-1,2203,  
     &2*3201,2203,2101,2103,5*0,-211,11,22,111,211,22,-211,111,22,-211, 
     &111,211,2*22,0,-211,111,211,2*22,0,2*-211,111,22,111,211,22,211,  
     &2*-211,2*111,-211,2*211,111,211,-211,2*111,211,-321,-211,111,11,  
     &-211,111,211,111,22,111,2*22,-211,111,211,3*22,935*0/ 
      DATA (KFDP(I,3),I=   1, 918)/70*0,14,6*0,2*16,2*0,5*111,310,130,  
     &2*0,2*111,310,130,113,211,223,221,2*113,2*211,2*223,2*221,2*113,  
     &221,113,2*213,-213,123*0,4*3,4*4,1,4,3,2*2,6*81,25*0,-211,3*111,  
     &-311,-313,-311,2*-321,2*-311,111,221,331,113,223,211,111,211,111, 
     &-311,-313,-311,2*-321,2*-311,111,221,331,113,223,211,111,211,111, 
     &20*0,3*111,2*221,331,113,223,3*211,-211,111,-211,111,211,111,211, 
     &-211,111,113,111,223,2*111,-311,4*211,2*111,2*211,111,7*211,  
     &7*111,113,221,2*223,2*-211,-213,4*-211,-213,-211,-213,-211,2*211, 
     &2,2*0,-321,-323,-311,-321,-311,2*-321,-211,-213,2*-211,211,-321,  
     &-323,-311,-321,-311,2*-321,-211,-213,2*-211,211,46*0,3*111,113,   
     &2*221,331,2*223,-311,3*-211,-213,8*111,113,3*211,213,2*111,-211,  
     &3*111,113,111,2*113,221,331,223,111,221,331,113,223,113,2*223,    
     &2*221,3*111,221,113,223,4*211,3*-211,-213,-211,5*111,-321,3*211,  
     &3*111,2*211,2*111,2*-211,-213,3*111,221,113,223,6*111,3*0,221,    
     &331,333,321,311,221,331,333,321,311,19*0,3,5*0,-11,0,2*111,-211,  
     &-11,11,2*221,3*0,111,22*0,111,2*0,22,111,5*0,111,12*0,2*21,11*0,  
     &2*21,2*-6,111*0,-211,2*111,-211,3*111,-211,111,211,15*0,111,6*0,  
     &111,-211,9*0,111,-211,9*0,111,-211,111,-211,4*0,111,-211,111, 
     &-211,4*0,-211,4*0,111,-211,111,-211,4*0,111,-211,111,-211,4*0,    
     &-211,3*0,-211,5*0,111,211,3*0,111,10*0,2*111,211,-211,211,-211/   
      DATA (KFDP(I,3),I= 919,2000)/7*0,2212,3122,3212,3214,2112,2114,   
     &2212,2112,3122,3212,3214,2112,2114,2212,2112,50*0,3*3,1,12*0, 
     &2112,43*0,3322,949*0/ 
      DATA (KFDP(I,4),I=   1,2000)/83*0,3*111,9*0,-211,3*0,111,2*-211,  
     &0,111,0,2*111,113,221,111,-213,-211,211,123*0,13*81,37*0,111, 
     &3*211,111,5*0,-211,111,-211,111,2*0,111,3*211,111,5*0,-211,111,   
     &-211,111,50*0,2*111,2*-211,2*111,-211,211,3*111,211,14*111,221,   
     &113,223,2*111,2*113,223,2*111,-1,4*0,-211,111,-211,211,111,2*0,   
     &2*111,-211,2*0,-211,111,-211,211,111,2*0,2*111,-211,96*0,6*111,   
     &3*-211,-213,4*111,113,6*111,3*-211,3*111,2*-211,2*111,3*-211, 
     &12*111,6*0,-321,-311,3*0,-321,-311,19*0,-3,11*0,-11,280*0,111,    
     &-211,3*0,111,29*0,-211,111,5*0,-211,111,50*0,2101,2103,2*2101,    
     &1006*0/   
      DATA (KFDP(I,5),I=   1,2000)/85*0,111,15*0,111,7*0,111,0,2*111,   
     &175*0,111,-211,111,7*0,2*111,4*0,111,-211,111,7*0,2*111,93*0,111, 
     &-211,111,3*0,111,-211,4*0,111,-211,111,3*0,111,-211,1571*0/   
    
C...LUDAT4, with character strings. 
      DATA (CHAF(I)  ,I=   1, 331)/'d','u','s','c','b','t','l','h', 
     &2*' ','e','nu_e','mu','nu_mu','tau','nu_tau','chi','nu_chi',  
     &2*' ','g','gamma','Z','W','H',6*' ','Z''','Z"','W''','H''','H"',  
     &'H',2*' ','R',40*' ','specflav','rndmflav','phasespa','c-hadron', 
     &'b-hadron','t-hadron','l-hadron','h-hadron','Wvirt','diquark',    
     &'cluster','string','indep.','CMshower','SPHEaxis','THRUaxis', 
     &'CLUSjet','CELLjet','table',' ','pi',2*'K',2*'D','D_s',2*'B', 
     &'B_s',' ','pi','eta','eta''','eta_c','eta_b','eta_t','eta_l', 
     &'eta_h',2*' ','rho',2*'K*',2*'D*','D*_s',2*'B*','B*_s',' ','rho', 
     &'omega','phi','J/psi','Upsilon','Theta','Theta_l','Theta_h',  
     &2*' ','b_1',2*'K_1',2*'D_1','D_1s',2*'B_1','B_1s',' ','b_1',  
     &'h_1','h''_1','h_1c','h_1b','h_1t','h_1l','h_1h',2*' ','a_0', 
     &2*'K*_0',2*'D*_0','D*_0s',2*'B*_0','B*_0s',' ','a_0','f_0',   
     &'f''_0','chi_0c','chi_0b','chi_0t','chi_0l','chi_0h',2*' ','a_1', 
     &2*'K*_1',2*'D*_1','D*_1s',2*'B*_1','B*_1s',' ','a_1','f_1',   
     &'f''_1','chi_1c','chi_1b','chi_1t','chi_1l','chi_1h',2*' ','a_2', 
     &2*'K*_2',2*'D*_2','D*_2s',2*'B*_2','B*_2s',' ','a_2','f_2',   
     &'f''_2','chi_2c','chi_2b','chi_2t','chi_2l','chi_2h',2*' ','K_L', 
     &'K_S',58*' ','pi_diffr','n_diffr','p_diffr',22*' ','Lambda',5*' ',    
     &'Lambda_c',' ',2*'Xi_c',6*' ','Lambda_b',' ',2*'Xi_b',6*' '/  
      DATA (CHAF(I)  ,I= 332, 500)/'n','p',' ',3*'Sigma',2*'Xi',' ',    
     &3*'Sigma_c',2*'Xi''_c','Omega_c', 
     &4*' ',3*'Sigma_b',2*'Xi''_b','Omega_b',4*' ',4*'Delta',   
     &3*'Sigma*',2*'Xi*','Omega',3*'Sigma*_c',2*'Xi*_c','Omega*_c', 
     &4*' ',3*'Sigma*_b',2*'Xi*_b','Omega*_b',114*' '/  
    
C...LUDATR, with initial values for the random number generator.    
      DATA MRLU/19780503,0,0,97,33,0/   
    
      END   
      SUBROUTINE PYINIT(FRAME,BEAM,TARGET,WIN)  
    
C...Initializes the generation procedure; finds maxima of the   
C...differential cross-sections to be used for weighting.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/LUDAT4/CHAF(500)   
      CHARACTER CHAF*8  
      SAVE /LUDAT4/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      CHARACTER*(*) FRAME,BEAM,TARGET   
      CHARACTER CHFRAM*8,CHBEAM*8,CHTARG*8,CHMO(12)*3,CHLH(2)*6 
      DATA CHMO/'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep',  
     &'Oct','Nov','Dec'/, CHLH/'lepton','hadron'/   
    
clin-12/2012 correct NN differential cross section in HIJING:
      WRITE(MSTU(11),*) 'In PYINIT: BEAM,TARGET= ',BEAM,TARGET

C...Write headers.  
C      IF(MSTP(122).GE.1) WRITE(MSTU(11),1000) MSTP(181),MSTP(182),  
C     &MSTP(185),CHMO(MSTP(184)),MSTP(183)   
      CALL LULIST(0)
C      IF(MSTP(122).GE.1) WRITE(MSTU(11),1100)  
    
C...Identify beam and target particles and initialize kinematics.   
      CHFRAM=FRAME//' ' 
      CHBEAM=BEAM//' '  
      CHTARG=TARGET//' '    
      CALL PYINKI(CHFRAM,CHBEAM,CHTARG,WIN) 
    
C...Select partonic subprocesses to be included in the simulation.  
      IF(MSEL.NE.0) THEN    
        DO 100 I=1,200  
  100   MSUB(I)=0   
      ENDIF 
      IF(MINT(43).EQ.1.AND.(MSEL.EQ.1.OR.MSEL.EQ.2)) THEN   
C...Lepton+lepton -> gamma/Z0 or W. 
        IF(MINT(11)+MINT(12).EQ.0) MSUB(1)=1    
        IF(MINT(11)+MINT(12).NE.0) MSUB(2)=1    
      ELSEIF(MSEL.EQ.1) THEN    
C...High-pT QCD processes:  
        MSUB(11)=1  
        MSUB(12)=1  
        MSUB(13)=1  
        MSUB(28)=1  
        MSUB(53)=1  
        MSUB(68)=1  
        IF(MSTP(82).LE.1.AND.CKIN(3).LT.PARP(81)) MSUB(95)=1    
        IF(MSTP(82).GE.2.AND.CKIN(3).LT.PARP(82)) MSUB(95)=1    
      ELSEIF(MSEL.EQ.2) THEN    
C...All QCD processes:  
        MSUB(11)=1  
        MSUB(12)=1  
        MSUB(13)=1  
        MSUB(28)=1  
        MSUB(53)=1  
        MSUB(68)=1  
        MSUB(91)=1  
        MSUB(92)=1  
        MSUB(93)=1  
        MSUB(95)=1  
      ELSEIF(MSEL.GE.4.AND.MSEL.LE.8) THEN  
C...Heavy quark production. 
        MSUB(81)=1  
        MSUB(82)=1  
        DO 110 J=1,MIN(8,MDCY(21,3))    
  110   MDME(MDCY(21,2)+J-1,1)=0    
        MDME(MDCY(21,2)+MSEL-1,1)=1 
      ELSEIF(MSEL.EQ.10) THEN   
C...Prompt photon production:   
        MSUB(14)=1  
        MSUB(18)=1  
        MSUB(29)=1  
      ELSEIF(MSEL.EQ.11) THEN   
C...Z0/gamma* production:   
        MSUB(1)=1   
      ELSEIF(MSEL.EQ.12) THEN   
C...W+/- production:    
        MSUB(2)=1   
      ELSEIF(MSEL.EQ.13) THEN   
C...Z0 + jet:   
        MSUB(15)=1  
        MSUB(30)=1  
      ELSEIF(MSEL.EQ.14) THEN   
C...W+/- + jet: 
        MSUB(16)=1  
        MSUB(31)=1  
      ELSEIF(MSEL.EQ.15) THEN   
C...Z0 & W+/- pair production:  
        MSUB(19)=1  
        MSUB(20)=1  
        MSUB(22)=1  
        MSUB(23)=1  
        MSUB(25)=1  
      ELSEIF(MSEL.EQ.16) THEN   
C...H0 production:  
        MSUB(3)=1   
        MSUB(5)=1   
        MSUB(8)=1   
        MSUB(102)=1 
      ELSEIF(MSEL.EQ.17) THEN   
C...H0 & Z0 or W+/- pair production:    
        MSUB(24)=1  
        MSUB(26)=1  
      ELSEIF(MSEL.EQ.21) THEN   
C...Z'0 production: 
        MSUB(141)=1 
      ELSEIF(MSEL.EQ.22) THEN   
C...H+/- production:    
        MSUB(142)=1 
      ELSEIF(MSEL.EQ.23) THEN   
C...R production:   
        MSUB(143)=1 
      ENDIF 
    
C...Count number of subprocesses on.    
      MINT(44)=0    
      DO 120 ISUB=1,200 
      IF(MINT(43).LT.4.AND.ISUB.GE.91.AND.ISUB.LE.96.AND.   
     &MSUB(ISUB).EQ.1) THEN 
        WRITE(MSTU(11),1200) ISUB,CHLH(MINT(41)),CHLH(MINT(42)) 
        STOP    
      ELSEIF(MSUB(ISUB).EQ.1.AND.ISET(ISUB).EQ.-1) THEN 
        WRITE(MSTU(11),1300) ISUB   
        STOP    
      ELSEIF(MSUB(ISUB).EQ.1.AND.ISET(ISUB).LE.-2) THEN 
        WRITE(MSTU(11),1400) ISUB   
        STOP    
      ELSEIF(MSUB(ISUB).EQ.1) THEN  
        MINT(44)=MINT(44)+1 
      ENDIF 
  120 CONTINUE  
      IF(MINT(44).EQ.0) THEN    
        WRITE(MSTU(11),1500)    
        STOP    
      ENDIF 
      MINT(45)=MINT(44)-MSUB(91)-MSUB(92)-MSUB(93)-MSUB(94) 
    
C...Maximum 4 generations; set maximum number of allowed flavours.  
      MSTP(1)=MIN(4,MSTP(1))    
      MSTU(114)=MIN(MSTU(114),2*MSTP(1))    
      MSTP(54)=MIN(MSTP(54),2*MSTP(1))  
    
C...Sum up Cabibbo-Kobayashi-Maskawa factors for each quark/lepton. 
      DO 140 I=-20,20   
      VINT(180+I)=0.    
      IA=IABS(I)    
      IF(IA.GE.1.AND.IA.LE.2*MSTP(1)) THEN  
        DO 130 J=1,MSTP(1)  
        IB=2*J-1+MOD(IA,2)  
        IPM=(5-ISIGN(1,I))/2    
        IDC=J+MDCY(IA,2)+2  
  130   IF(MDME(IDC,1).EQ.1.OR.MDME(IDC,1).EQ.IPM) VINT(180+I)= 
     &  VINT(180+I)+VCKM((IA+1)/2,(IB+1)/2) 
      ELSEIF(IA.GE.11.AND.IA.LE.10+2*MSTP(1)) THEN  
        VINT(180+I)=1.  
      ENDIF 
  140 CONTINUE  
    
C...Choose Lambda value to use in alpha-strong. 
      MSTU(111)=MSTP(2) 
      IF(MSTP(3).GE.1) THEN 
        ALAM=PARP(1)    
        IF(MSTP(51).EQ.1) ALAM=0.2  
        IF(MSTP(51).EQ.2) ALAM=0.29 
        IF(MSTP(51).EQ.3) ALAM=0.2  
        IF(MSTP(51).EQ.4) ALAM=0.4  
        IF(MSTP(51).EQ.11) ALAM=0.16    
        IF(MSTP(51).EQ.12) ALAM=0.26    
        IF(MSTP(51).EQ.13) ALAM=0.36    
        PARP(1)=ALAM    
        PARP(61)=ALAM   
        PARU(112)=ALAM  
        PARJ(81)=ALAM   
      ENDIF 
    
C...Initialize widths and partial widths for resonances.    
      CALL PYINRE   
    
C...Reset variables for cross-section calculation.  
      DO 150 I=0,200    
      DO 150 J=1,3  
      NGEN(I,J)=0   
  150 XSEC(I,J)=0.  
      VINT(108)=0.  
    
C...Find parametrized total cross-sections. 
      IF(MINT(43).EQ.4) CALL PYXTOT 
    
C...Maxima of differential cross-sections.  
      IF(MSTP(121).LE.0) CALL PYMAXI    
    
C...Initialize possibility of overlayed events. 
      IF(MSTP(131).NE.0) CALL PYOVLY(1) 
    
C...Initialize multiple interactions with variable impact parameter.    
      IF(MINT(43).EQ.4.AND.(MINT(45).NE.0.OR.MSTP(131).NE.0).AND.   
     &MSTP(82).GE.2) CALL PYMULT(1) 
C      IF(MSTP(122).GE.1) WRITE(MSTU(11),1600)  
    
C...Formats for initialization information. 
clin 1000 FORMAT(///20X,'The Lund Monte Carlo - PYTHIA version ',I1,'.',I1/ 
clin     &20X,'**  Last date of change:  ',I2,1X,A3,1X,I4,'  **'/)  
clin 1100 FORMAT('1',18('*'),1X,'PYINIT: initialization of PYTHIA ',    
clin     &'routines',1X,17('*'))    
 1200 FORMAT(1X,'Error: process number ',I3,' not meaningful for ',A6,  
     &'-',A6,' interactions.'/1X,'Execution stopped!')  
 1300 FORMAT(1X,'Error: requested subprocess',I4,' not implemented.'/   
     &1X,'Execution stopped!')  
 1400 FORMAT(1X,'Error: requested subprocess',I4,' not existing.'/  
     &1X,'Execution stopped!')  
 1500 FORMAT(1X,'Error: no subprocess switched on.'/    
     &1X,'Execution stopped.')  
clin 1600 FORMAT(/1X,22('*'),1X,'PYINIT: initialization completed',1X,  
clin     &22('*'))  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYTHIA 
    
C...Administers the generation of a high-pt event via calls to a number 
C...of subroutines; also computes cross-sections.   
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
    
C...Loop over desired number of overlayed events (normally 1).  
      MINT(7)=0 
      MINT(8)=0 
      NOVL=1    
      IF(MSTP(131).NE.0) CALL PYOVLY(2) 
      IF(MSTP(131).NE.0) NOVL=MINT(81)  
      MINT(83)=0    
      MINT(84)=MSTP(126)    
      MSTU(70)=0    
      DO 190 IOVL=1,NOVL    
      IF(MINT(84)+100.GE.MSTU(4)) THEN  
        CALL LUERRM(11, 
     &  '(PYTHIA:) no more space in LUJETS for overlayed events')   
        IF(MSTU(21).GE.1) GOTO 200  
      ENDIF 
      MINT(82)=IOVL 
    
C...Generate variables of hard scattering.  
  100 CONTINUE  
      IF(IOVL.EQ.1) NGEN(0,2)=NGEN(0,2)+1   
      MINT(31)=0    
      MINT(51)=0    
      CALL PYRAND   
      ISUB=MINT(1)  
      IF(IOVL.EQ.1) THEN    
        NGEN(ISUB,2)=NGEN(ISUB,2)+1 
    
C...Store information on hard interaction.  
        DO 110 J=1,200  
        MSTI(J)=0   
  110   PARI(J)=0.  
        MSTI(1)=MINT(1) 
        MSTI(2)=MINT(2) 
        MSTI(11)=MINT(11)   
        MSTI(12)=MINT(12)   
        MSTI(15)=MINT(15)   
        MSTI(16)=MINT(16)   
        MSTI(17)=MINT(17)   
        MSTI(18)=MINT(18)   
        PARI(11)=VINT(1)    
        PARI(12)=VINT(2)    
        IF(ISUB.NE.95) THEN 
          DO 120 J=13,22    
  120     PARI(J)=VINT(30+J)    
          PARI(33)=VINT(41) 
          PARI(34)=VINT(42) 
          PARI(35)=PARI(33)-PARI(34)    
          PARI(36)=VINT(21) 
          PARI(37)=VINT(22) 
          PARI(38)=VINT(26) 
          PARI(41)=VINT(23) 
        ENDIF   
      ENDIF 
    
      IF(MSTP(111).EQ.-1) GOTO 160  
      IF(ISUB.LE.90.OR.ISUB.GE.95) THEN 
C...Hard scattering (including low-pT): 
C...reconstruct kinematics and colour flow of hard scattering.  
        CALL PYSCAT 
        IF(MINT(51).EQ.1) GOTO 100  
    
C...Showering of initial state partons (optional).  
        IPU1=MINT(84)+1 
        IPU2=MINT(84)+2 
        IF(MSTP(61).GE.1.AND.MINT(43).NE.1.AND.ISUB.NE.95)  
     &  CALL PYSSPA(IPU1,IPU2)  
        NSAV1=N 
    
C...Multiple interactions.  
        IF(MSTP(81).GE.1.AND.MINT(43).EQ.4.AND.ISUB.NE.95)  
     &  CALL PYMULT(6)  
        MINT(1)=ISUB    
        NSAV2=N 
    
C...Hadron remnants and primordial kT.  
        CALL PYREMN(IPU1,IPU2)  
        IF(MINT(51).EQ.1) GOTO 100  
        NSAV3=N 
    
C...Showering of final state partons (optional).    
        IPU3=MINT(84)+3 
        IPU4=MINT(84)+4 
        IF(MSTP(71).GE.1.AND.ISUB.NE.95.AND.K(IPU3,1).GT.0.AND. 
     &  K(IPU3,1).LE.10.AND.K(IPU4,1).GT.0.AND.K(IPU4,1).LE.10) THEN    
          QMAX=SQRT(PARP(71)*VINT(52))  
          IF(ISUB.EQ.5) QMAX=SQRT(PMAS(23,1)**2)    
          IF(ISUB.EQ.8) QMAX=SQRT(PMAS(24,1)**2)    
          CALL LUSHOW(IPU3,IPU4,QMAX)   
        ENDIF   
    
C...Sum up transverse and longitudinal momenta. 
        IF(IOVL.EQ.1) THEN  
          PARI(65)=2.*PARI(17)  
          DO 130 I=MSTP(126)+1,N    
          IF(K(I,1).LE.0.OR.K(I,1).GT.10) GOTO 130  
          PT=SQRT(P(I,1)**2+P(I,2)**2)  
          PARI(69)=PARI(69)+PT  
          IF(I.LE.NSAV1.OR.I.GT.NSAV3) PARI(66)=PARI(66)+PT 
          IF(I.GT.NSAV1.AND.I.LE.NSAV2) PARI(68)=PARI(68)+PT    
  130     CONTINUE  
          PARI(67)=PARI(68) 
          PARI(71)=VINT(151)    
          PARI(72)=VINT(152)    
          PARI(73)=VINT(151)    
          PARI(74)=VINT(152)    
        ENDIF   
    
C...Decay of final state resonances.    
        IF(MSTP(41).GE.1.AND.ISUB.NE.95) CALL PYRESD    
    
      ELSE  
C...Diffractive and elastic scattering. 
        CALL PYDIFF 
        IF(IOVL.EQ.1) THEN  
          PARI(65)=2.*PARI(17)  
          PARI(66)=PARI(65) 
          PARI(69)=PARI(65) 
        ENDIF   
      ENDIF 
    
C...Recalculate energies from momenta and masses (if desired).  
      IF(MSTP(113).GE.1) THEN   
        DO 140 I=MINT(83)+1,N   
  140   IF(K(I,1).GT.0.AND.K(I,1).LE.10) P(I,4)=SQRT(P(I,1)**2+ 
     &  P(I,2)**2+P(I,3)**2+P(I,5)**2)  
      ENDIF 
    
C...Rearrange partons along strings, check invariant mass cuts. 
      MSTU(28)=0    
      CALL LUPREP(MINT(84)+1)   
      IF(MSTP(112).EQ.1.AND.MSTU(28).EQ.3) GOTO 100 
      IF(MSTP(125).EQ.0.OR.MSTP(125).EQ.1) THEN 
        DO 150 I=MINT(84)+1,N   
        IF(K(I,2).NE.94) GOTO 150   
        K(I+1,3)=MOD(K(I+1,4)/MSTU(5),MSTU(5))  
        K(I+2,3)=MOD(K(I+2,4)/MSTU(5),MSTU(5))  
  150   CONTINUE    
        CALL LUEDIT(12) 
        CALL LUEDIT(14) 
        IF(MSTP(125).EQ.0) CALL LUEDIT(15)  
        IF(MSTP(125).EQ.0) MINT(4)=0    
      ENDIF 
    
C...Introduce separators between sections in LULIST event listing.  
      IF(IOVL.EQ.1.AND.MSTP(125).LE.0) THEN 
        MSTU(70)=1  
        MSTU(71)=N  
      ELSEIF(IOVL.EQ.1) THEN    
        MSTU(70)=3  
        MSTU(71)=2  
        MSTU(72)=MINT(4)    
        MSTU(73)=N  
      ENDIF 
    
C...Perform hadronization (if desired). 
      IF(MSTP(111).GE.1) CALL LUEXEC    
      IF(MSTP(125).EQ.0.OR.MSTP(125).EQ.1) CALL LUEDIT(14)  
    
C...Calculate Monte Carlo estimates of cross-sections.  
  160 IF(IOVL.EQ.1) THEN    
        IF(MSTP(111).NE.-1) NGEN(ISUB,3)=NGEN(ISUB,3)+1 
        NGEN(0,3)=NGEN(0,3)+1   
        XSEC(0,3)=0.    
        DO 170 I=1,200  
        IF(I.EQ.96) THEN    
          XSEC(I,3)=0.  
        ELSEIF(MSUB(95).EQ.1.AND.(I.EQ.11.OR.I.EQ.12.OR.I.EQ.13.OR. 
     &  I.EQ.28.OR.I.EQ.53.OR.I.EQ.68)) THEN    
          XSEC(I,3)=XSEC(96,2)*NGEN(I,3)/MAX(1.,FLOAT(NGEN(96,1))*  
     &    FLOAT(NGEN(96,2)))    
        ELSEIF(NGEN(I,1).EQ.0) THEN 
          XSEC(I,3)=0.  
        ELSEIF(NGEN(I,2).EQ.0) THEN 
          XSEC(I,3)=XSEC(I,2)*NGEN(0,3)/(FLOAT(NGEN(I,1))*  
     &    FLOAT(NGEN(0,2))) 
        ELSE    
          XSEC(I,3)=XSEC(I,2)*NGEN(I,3)/(FLOAT(NGEN(I,1))*  
     &    FLOAT(NGEN(I,2))) 
        ENDIF   
  170   XSEC(0,3)=XSEC(0,3)+XSEC(I,3)   
        IF(MSUB(95).EQ.1) THEN  
          NGENS=NGEN(91,3)+NGEN(92,3)+NGEN(93,3)+NGEN(94,3)+NGEN(95,3)  
          XSECS=XSEC(91,3)+XSEC(92,3)+XSEC(93,3)+XSEC(94,3)+XSEC(95,3)  
          XMAXS=XSEC(95,1)  
          IF(MSUB(91).EQ.1) XMAXS=XMAXS+XSEC(91,1)  
          IF(MSUB(92).EQ.1) XMAXS=XMAXS+XSEC(92,1)  
          IF(MSUB(93).EQ.1) XMAXS=XMAXS+XSEC(93,1)  
          IF(MSUB(94).EQ.1) XMAXS=XMAXS+XSEC(94,1)  
          FAC=1.    
          IF(NGENS.LT.NGEN(0,3)) FAC=(XMAXS-XSECS)/(XSEC(0,3)-XSECS)    
          XSEC(11,3)=FAC*XSEC(11,3) 
          XSEC(12,3)=FAC*XSEC(12,3) 
          XSEC(13,3)=FAC*XSEC(13,3) 
          XSEC(28,3)=FAC*XSEC(28,3) 
          XSEC(53,3)=FAC*XSEC(53,3) 
          XSEC(68,3)=FAC*XSEC(68,3) 
          XSEC(0,3)=XSEC(91,3)+XSEC(92,3)+XSEC(93,3)+XSEC(94,3)+    
     &    XSEC(95,1)    
        ENDIF   
    
C...Store final information.    
        MINT(5)=MINT(5)+1   
        MSTI(3)=MINT(3) 
        MSTI(4)=MINT(4) 
        MSTI(5)=MINT(5) 
        MSTI(6)=MINT(6) 
        MSTI(7)=MINT(7) 
        MSTI(8)=MINT(8) 
        MSTI(13)=MINT(13)   
        MSTI(14)=MINT(14)   
        MSTI(21)=MINT(21)   
        MSTI(22)=MINT(22)   
        MSTI(23)=MINT(23)   
        MSTI(24)=MINT(24)   
        MSTI(25)=MINT(25)   
        MSTI(26)=MINT(26)   
        MSTI(31)=MINT(31)   
        PARI(1)=XSEC(0,3)   
        PARI(2)=XSEC(0,3)/MINT(5)   
        PARI(31)=VINT(141)  
        PARI(32)=VINT(142)  
        IF(ISUB.NE.95.AND.MINT(7)*MINT(8).NE.0) THEN    
          PARI(42)=2.*VINT(47)/VINT(1)  
          DO 180 IS=7,8 
          PARI(36+IS)=P(MINT(IS),3)/VINT(1) 
          PARI(38+IS)=P(MINT(IS),4)/VINT(1) 
          I=MINT(IS)    
          PR=MAX(1E-20,P(I,5)**2+P(I,1)**2+P(I,2)**2)   
          PARI(40+IS)=SIGN(LOG(MIN((SQRT(PR+P(I,3)**2)+ABS(P(I,3)))/    
     &    SQRT(PR),1E20)),P(I,3))   
          PR=MAX(1E-20,P(I,1)**2+P(I,2)**2) 
          PARI(42+IS)=SIGN(LOG(MIN((SQRT(PR+P(I,3)**2)+ABS(P(I,3)))/    
     &    SQRT(PR),1E20)),P(I,3))   
          PARI(44+IS)=P(I,3)/SQRT(P(I,1)**2+P(I,2)**2+P(I,3)**2)    
          PARI(46+IS)=ULANGL(P(I,3),SQRT(P(I,1)**2+P(I,2)**2))  
          PARI(48+IS)=ULANGL(P(I,1),P(I,2)) 
  180     CONTINUE  
        ENDIF   
        PARI(61)=VINT(148)  
        IF(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3) THEN 
          MSTU(161)=MINT(21)    
          MSTU(162)=0   
        ELSE    
          MSTU(161)=MINT(21)    
          MSTU(162)=MINT(22)    
        ENDIF   
      ENDIF 
    
C...Prepare to go to next overlayed event.  
      MSTI(41)=IOVL 
      IF(IOVL.GE.2.AND.IOVL.LE.10) MSTI(40+IOVL)=ISUB   
      IF(MSTU(70).LT.10) THEN   
        MSTU(70)=MSTU(70)+1 
        MSTU(70+MSTU(70))=N 
      ENDIF 
      MINT(83)=N    
      MINT(84)=N+MSTP(126)  
  190 CONTINUE  
    
C...Information on overlayed events.    
      IF(MSTP(131).EQ.1.AND.MSTP(133).GE.1) THEN    
        PARI(91)=VINT(132)  
        PARI(92)=VINT(133)  
        PARI(93)=VINT(134)  
        IF(MSTP(133).EQ.2) PARI(93)=PARI(93)*XSEC(0,3)/VINT(131)    
      ENDIF 
    
C...Transform to the desired coordinate frame.  
  200 CALL PYFRAM(MSTP(124))    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYINKI(CHFRAM,CHBEAM,CHTARG,WIN)   
    
C...Identifies the two incoming particles and sets up kinematics,   
C...including rotations and boosts to/from CM frame.    
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      CHARACTER CHFRAM*8,CHBEAM*8,CHTARG*8,CHCOM(3)*8,CHALP(2)*26,  
     &CHIDNT(3)*8,CHTEMP*8,CHCDE(18)*8,CHINIT*76    
      DIMENSION LEN(3),KCDE(18) 
      DATA CHALP/'abcdefghijklmnopqrstuvwxyz',  
     &'ABCDEFGHIJKLMNOPQRSTUVWXYZ'/ 
      DATA CHCDE/'e-      ','e+      ','nue     ','nue~    ',   
     &'mu-     ','mu+     ','numu    ','numu~   ','tau-    ',   
     &'tau+    ','nutau   ','nutau~  ','pi+     ','pi-     ',   
     &'n       ','n~      ','p       ','p~      '/  
      DATA KCDE/11,-11,12,-12,13,-13,14,-14,15,-15,16,-16,  
     &211,-211,2112,-2112,2212,-2212/   
    
C...Convert character variables to lowercase and find their length. 
      CHCOM(1)=CHFRAM   
      CHCOM(2)=CHBEAM   
      CHCOM(3)=CHTARG   
      DO 120 I=1,3  
      LEN(I)=8  
      DO 100 LL=8,1,-1  
      IF(LEN(I).EQ.LL.AND.CHCOM(I)(LL:LL).EQ.' ') LEN(I)=LL-1   
      DO 100 LA=1,26    
  100 IF(CHCOM(I)(LL:LL).EQ.CHALP(2)(LA:LA)) CHCOM(I)(LL:LL)=   
     &CHALP(1)(LA:LA)   
      CHIDNT(I)=CHCOM(I)    
      DO 110 LL=1,6 
      IF(CHIDNT(I)(LL:LL+2).EQ.'bar') THEN  
        CHTEMP=CHIDNT(I)    
        CHIDNT(I)=CHTEMP(1:LL-1)//'~'//CHTEMP(LL+3:8)//'  ' 
      ENDIF 
  110 CONTINUE  
      DO 120 LL=1,8 
      IF(CHIDNT(I)(LL:LL).EQ.'_') THEN  
        CHTEMP=CHIDNT(I)    
        CHIDNT(I)=CHTEMP(1:LL-1)//CHTEMP(LL+1:8)//' '   
      ENDIF 
  120 CONTINUE  
    
C...Set initial state. Error for unknown codes. Reset variables.    
      N=2   
      DO 140 I=1,2  
      K(I,2)=0  
      DO 130 J=1,18 
  130 IF(CHIDNT(I+1).EQ.CHCDE(J)) K(I,2)=KCDE(J)    
      P(I,5)=ULMASS(K(I,2)) 
      MINT(40+I)=1  
      IF(IABS(K(I,2)).GT.100) MINT(40+I)=2  
      DO 140 J=1,5  
  140 V(I,J)=0. 
      IF(K(1,2).EQ.0) WRITE(MSTU(11),1000) CHBEAM(1:LEN(2)) 
      IF(K(2,2).EQ.0) WRITE(MSTU(11),1100) CHTARG(1:LEN(3)) 
      IF(K(1,2).EQ.0.OR.K(2,2).EQ.0) STOP   
      DO 150 J=6,10 
  150 VINT(J)=0.    
      CHINIT=' '    
    
C...Set up kinematics for events defined in CM frame.   
      IF(CHCOM(1)(1:2).EQ.'cm') THEN    
        IF(CHCOM(2)(1:1).NE.'e') THEN   
          LOFFS=(34-(LEN(2)+LEN(3)))/2  
          CHINIT(LOFFS+1:76)='PYTHIA will be initialized for a '//  
     &    CHCOM(2)(1:LEN(2))//'-'//CHCOM(3)(1:LEN(3))//' collider'//' ' 
        ELSE    
          LOFFS=(33-(LEN(2)+LEN(3)))/2  
          CHINIT(LOFFS+1:76)='PYTHIA will be initialized for an '// 
     &    CHCOM(2)(1:LEN(2))//'-'//CHCOM(3)(1:LEN(3))//' collider'//' ' 
        ENDIF   
C        WRITE(MSTU(11),1200) CHINIT 
C        WRITE(MSTU(11),1300) WIN    
        S=WIN**2    
        P(1,1)=0.   
        P(1,2)=0.   
        P(2,1)=0.   
        P(2,2)=0.   
        P(1,3)=SQRT(((S-P(1,5)**2-P(2,5)**2)**2-(2.*P(1,5)*P(2,5))**2)/ 
     &  (4.*S)) 
        P(2,3)=-P(1,3)  
        P(1,4)=SQRT(P(1,3)**2+P(1,5)**2)    
        P(2,4)=SQRT(P(2,3)**2+P(2,5)**2)    
    
C...Set up kinematics for fixed target events.  
      ELSEIF(CHCOM(1)(1:3).EQ.'fix') THEN   
        LOFFS=(29-(LEN(2)+LEN(3)))/2    
        CHINIT(LOFFS+1:76)='PYTHIA will be initialized for '//  
     &  CHCOM(2)(1:LEN(2))//' on '//CHCOM(3)(1:LEN(3))//    
     &  ' fixed target'//' '    
C        WRITE(MSTU(11),1200) CHINIT 
C        WRITE(MSTU(11),1400) WIN    
        P(1,1)=0.   
        P(1,2)=0.   
        P(2,1)=0.   
        P(2,2)=0.   
        P(1,3)=WIN  
        P(1,4)=SQRT(P(1,3)**2+P(1,5)**2)    
        P(2,3)=0.   
        P(2,4)=P(2,5)   
        S=P(1,5)**2+P(2,5)**2+2.*P(2,4)*P(1,4)  
        VINT(10)=P(1,3)/(P(1,4)+P(2,4)) 
        CALL LUROBO(0.,0.,0.,0.,-VINT(10))  
C        WRITE(MSTU(11),1500) SQRT(S)    
    
C...Set up kinematics for events in user-defined frame. 
      ELSEIF(CHCOM(1)(1:3).EQ.'use') THEN   
        LOFFS=(13-(LEN(1)+LEN(2)))/2    
        CHINIT(LOFFS+1:76)='PYTHIA will be initialized for '//  
     &  CHCOM(2)(1:LEN(2))//' on '//CHCOM(3)(1:LEN(3))//    
     &  'user-specified configuration'//' ' 
C        WRITE(MSTU(11),1200) CHINIT 
C        WRITE(MSTU(11),1600)    
C        WRITE(MSTU(11),1700) CHCOM(2),P(1,1),P(1,2),P(1,3)  
C        WRITE(MSTU(11),1700) CHCOM(3),P(2,1),P(2,2),P(2,3)  
        P(1,4)=SQRT(P(1,1)**2+P(1,2)**2+P(1,3)**2+P(1,5)**2)    
        P(2,4)=SQRT(P(2,1)**2+P(2,2)**2+P(2,3)**2+P(2,5)**2)    
        DO 160 J=1,3    
  160   VINT(7+J)=sngl((DBLE(P(1,J))+DBLE(P(2,J)))
     &          /DBLE(P(1,4)+P(2,4)))
        CALL LUROBO(0.,0.,-VINT(8),-VINT(9),-VINT(10))  
        VINT(7)=ULANGL(P(1,1),P(1,2))   
        CALL LUROBO(0.,-VINT(7),0.,0.,0.)   
        VINT(6)=ULANGL(P(1,3),P(1,1))   
        CALL LUROBO(-VINT(6),0.,0.,0.,0.)   
        S=P(1,5)**2+P(2,5)**2+2.*(P(1,4)*P(2,4)-P(1,3)*P(2,3))  
C        WRITE(MSTU(11),1500) SQRT(S)    
    
C...Unknown frame. Error for too low CM energy. 
      ELSE  
        WRITE(MSTU(11),1800) CHFRAM(1:LEN(1))   
        STOP    
      ENDIF 
      IF(S.LT.PARP(2)**2) THEN  
        WRITE(MSTU(11),1900) SQRT(S)    
        STOP    
      ENDIF 
    
C...Save information on incoming particles. 
      MINT(11)=K(1,2)   
      MINT(12)=K(2,2)   
      MINT(43)=2*MINT(41)+MINT(42)-2    
      VINT(1)=SQRT(S)   
      VINT(2)=S 
      VINT(3)=P(1,5)    
      VINT(4)=P(2,5)    
      VINT(5)=P(1,3)    
    
C...Store constants to be used in generation.   
      IF(MSTP(82).LE.1) VINT(149)=4.*PARP(81)**2/S  
      IF(MSTP(82).GE.2) VINT(149)=4.*PARP(82)**2/S  
    
C...Formats for initialization and error information.   
 1000 FORMAT(1X,'Error: unrecognized beam particle ''',A,'''.'/ 
     &1X,'Execution stopped!')  
 1100 FORMAT(1X,'Error: unrecognized target particle ''',A,'''.'/   
     &1X,'Execution stopped!')  
clin 1200 FORMAT(/1X,78('=')/1X,'I',76X,'I'/1X,'I',A76,'I') 
c 1300 FORMAT(1X,'I',18X,'at',1X,F10.3,1X,'GeV center-of-mass energy',   
c     &19X,'I'/1X,'I',76X,'I'/1X,78('='))    
c 1400 FORMAT(1X,'I',22X,'at',1X,F10.3,1X,'GeV/c lab-momentum',22X,'I')  
c 1500 FORMAT(1X,'I',76X,'I'/1X,'I',11X,'corresponding to',1X,F10.3,1X,  
c     &'GeV center-of-mass energy',12X,'I'/1X,'I',76X,'I'/1X,78('='))    
c 1600 FORMAT(1X,'I',76X,'I'/1X,'I',24X,'px (GeV/c)',3X,'py (GeV/c)',3X, 
c     &'pz (GeV/c)',16X,'I') 
clin 1700 FORMAT(1X,'I',15X,A8,3(2X,F10.3,1X),15X,'I')  
 1800 FORMAT(1X,'Error: unrecognized coordinate frame ''',A,'''.'/  
     &1X,'Execution stopped!')  
 1900 FORMAT(1X,'Error: too low CM energy,',F8.3,' GeV for event ', 
     &'generation.'/1X,'Execution stopped!')    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYINRE 
    
C...Calculates full and effective widths of guage bosons, stores masses 
C...and widths, rescales coefficients to be used for resonance  
C...production generation.  
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      COMMON/PYINT6/PROC(0:200) 
      CHARACTER PROC*28 
      SAVE /PYINT6/ 
      DIMENSION WDTP(0:40),WDTE(0:40,0:5)   
    
C...Calculate full and effective widths of gauge bosons.    
      AEM=PARU(101) 
      XW=PARU(102)  
      DO 100 I=21,40    
      DO 100 J=0,40 
      WIDP(I,J)=0.  
  100 WIDE(I,J)=0.  
    
C...W+/-:   
      WMAS=PMAS(24,1)   
      WFAC=AEM/(24.*XW)*WMAS    
      CALL PYWIDT(24,WMAS,WDTP,WDTE)    
      WIDS(24,1)=((WDTE(0,1)+WDTE(0,2))*(WDTE(0,1)+WDTE(0,3))+  
     &(WDTE(0,1)+WDTE(0,2)+WDTE(0,1)+WDTE(0,3))*(WDTE(0,4)+WDTE(0,5))+  
     &2.*WDTE(0,4)*WDTE(0,5))/WDTP(0)**2    
      WIDS(24,2)=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))/WDTP(0)    
      WIDS(24,3)=(WDTE(0,1)+WDTE(0,3)+WDTE(0,4))/WDTP(0)    
      DO 110 I=0,40 
      WIDP(24,I)=WFAC*WDTP(I)   
  110 WIDE(24,I)=WFAC*WDTE(I,0) 
    
C...H+/-:   
      HCMAS=PMAS(37,1)  
      HCFAC=AEM/(8.*XW)*(HCMAS/WMAS)**2*HCMAS   
      CALL PYWIDT(37,HCMAS,WDTP,WDTE)   
      WIDS(37,1)=((WDTE(0,1)+WDTE(0,2))*(WDTE(0,1)+WDTE(0,3))+  
     &(WDTE(0,1)+WDTE(0,2)+WDTE(0,1)+WDTE(0,3))*(WDTE(0,4)+WDTE(0,5))+  
     &2.*WDTE(0,4)*WDTE(0,5))/WDTP(0)**2    
      WIDS(37,2)=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))/WDTP(0)    
      WIDS(37,3)=(WDTE(0,1)+WDTE(0,3)+WDTE(0,4))/WDTP(0)    
      DO 120 I=0,40 
      WIDP(37,I)=HCFAC*WDTP(I)  
  120 WIDE(37,I)=HCFAC*WDTE(I,0)    
    
C...Z0: 
      ZMAS=PMAS(23,1)   
      ZFAC=AEM/(48.*XW*(1.-XW))*ZMAS    
      CALL PYWIDT(23,ZMAS,WDTP,WDTE)    
      WIDS(23,1)=((WDTE(0,1)+WDTE(0,2))**2+ 
     &2.*(WDTE(0,1)+WDTE(0,2))*(WDTE(0,4)+WDTE(0,5))+   
     &2.*WDTE(0,4)*WDTE(0,5))/WDTP(0)**2    
      WIDS(23,2)=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))/WDTP(0)    
      WIDS(23,3)=0. 
      DO 130 I=0,40 
      WIDP(23,I)=ZFAC*WDTP(I)   
  130 WIDE(23,I)=ZFAC*WDTE(I,0) 
    
C...H0: 
      HMAS=PMAS(25,1)   
      HFAC=AEM/(8.*XW)*(HMAS/WMAS)**2*HMAS  
      CALL PYWIDT(25,HMAS,WDTP,WDTE)    
      WIDS(25,1)=((WDTE(0,1)+WDTE(0,2))**2+ 
     &2.*(WDTE(0,1)+WDTE(0,2))*(WDTE(0,4)+WDTE(0,5))+   
     &2.*WDTE(0,4)*WDTE(0,5))/WDTP(0)**2    
      WIDS(25,2)=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))/WDTP(0)    
      WIDS(25,3)=0. 
      DO 140 I=0,40 
      WIDP(25,I)=HFAC*WDTP(I)   
  140 WIDE(25,I)=HFAC*WDTE(I,0) 
    
C...Z'0:    
      ZPMAS=PMAS(32,1)  
      ZPFAC=AEM/(48.*XW*(1.-XW))*ZPMAS  
      CALL PYWIDT(32,ZPMAS,WDTP,WDTE)   
      WIDS(32,1)=((WDTE(0,1)+WDTE(0,2)+WDTE(0,3))**2+   
     &2.*(WDTE(0,1)+WDTE(0,2))*(WDTE(0,4)+WDTE(0,5))+   
     &2.*WDTE(0,4)*WDTE(0,5))/WDTP(0)**2    
      WIDS(32,2)=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))/WDTP(0)    
      WIDS(32,3)=0. 
      DO 150 I=0,40 
      WIDP(32,I)=ZPFAC*WDTP(I)  
  150 WIDE(32,I)=ZPFAC*WDTE(I,0)    
    
C...R:  
      RMAS=PMAS(40,1)   
      RFAC=0.08*RMAS/((MSTP(1)-1)*(1.+6.*(1.+ULALPS(RMAS**2)/PARU(1)))) 
      CALL PYWIDT(40,RMAS,WDTP,WDTE)    
      WIDS(40,1)=((WDTE(0,1)+WDTE(0,2))*(WDTE(0,1)+WDTE(0,3))+  
     &(WDTE(0,1)+WDTE(0,2)+WDTE(0,1)+WDTE(0,3))*(WDTE(0,4)+WDTE(0,5))+  
     &2.*WDTE(0,4)*WDTE(0,5))/WDTP(0)**2    
      WIDS(40,2)=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))/WDTP(0)    
      WIDS(40,3)=(WDTE(0,1)+WDTE(0,3)+WDTE(0,4))/WDTP(0)    
      DO 160 I=0,40 
      WIDP(40,I)=WFAC*WDTP(I)   
  160 WIDE(40,I)=WFAC*WDTE(I,0) 
    
C...Q:  
      KFLQM=1   
      DO 170 I=1,MIN(8,MDCY(21,3))  
      IDC=I+MDCY(21,2)-1    
      IF(MDME(IDC,1).LE.0) GOTO 170 
      KFLQM=I   
  170 CONTINUE  
      MINT(46)=KFLQM    
      KFPR(81,1)=KFLQM  
      KFPR(81,2)=KFLQM  
      KFPR(82,1)=KFLQM  
      KFPR(82,2)=KFLQM  
    
C...Set resonance widths and branching ratios in JETSET.    
      DO 180 I=1,6  
      IF(I.LE.3) KC=I+22    
      IF(I.EQ.4) KC=32  
      IF(I.EQ.5) KC=37  
      IF(I.EQ.6) KC=40  
      PMAS(KC,2)=WIDP(KC,0) 
      PMAS(KC,3)=MIN(0.9*PMAS(KC,1),10.*PMAS(KC,2)) 
      DO 180 J=1,MDCY(KC,3) 
      IDC=J+MDCY(KC,2)-1    
      BRAT(IDC)=WIDE(KC,J)/WIDE(KC,0)   
  180 CONTINUE  
    
C...Special cases in treatment of gamma*/Z0: redefine process name. 
      IF(MSTP(43).EQ.1) THEN    
        PROC(1)='f + fb -> gamma*'  
      ELSEIF(MSTP(43).EQ.2) THEN    
        PROC(1)='f + fb -> Z0'  
      ELSEIF(MSTP(43).EQ.3) THEN    
        PROC(1)='f + fb -> gamma*/Z0'   
      ENDIF 
    
C...Special cases in treatment of gamma*/Z0/Z'0: redefine process name. 
      IF(MSTP(44).EQ.1) THEN    
        PROC(141)='f + fb -> gamma*'    
      ELSEIF(MSTP(44).EQ.2) THEN    
        PROC(141)='f + fb -> Z0'    
      ELSEIF(MSTP(44).EQ.3) THEN    
        PROC(141)='f + fb -> Z''0'  
      ELSEIF(MSTP(44).EQ.4) THEN    
        PROC(141)='f + fb -> gamma*/Z0' 
      ELSEIF(MSTP(44).EQ.5) THEN    
        PROC(141)='f + fb -> gamma*/Z''0'   
      ELSEIF(MSTP(44).EQ.6) THEN    
        PROC(141)='f + fb -> Z0/Z''0'   
      ELSEIF(MSTP(44).EQ.7) THEN    
        PROC(141)='f + fb -> gamma*/Z0/Z''0'    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYXTOT 
    
C...Parametrizes total, double diffractive, single diffractive and  
C...elastic cross-sections for different energies and beams.    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      DIMENSION BCS(5,8),BCB(2,5),BCC(3)    
    
C...The following data lines are coefficients needed in the 
C...Block, Cahn parametrization of total cross-section and nuclear  
C...slope parameter; see below. 
      DATA ((BCS(I,J),J=1,8),I=1,5)/    
     1 41.74, 0.66, 0.0000, 337.,  0.0, 0.0, -39.3, 0.48,   
     2 41.66, 0.60, 0.0000, 306.,  0.0, 0.0, -34.6, 0.51,   
     3 41.36, 0.63, 0.0000, 299.,  7.3, 0.5, -40.4, 0.47,   
     4 41.68, 0.63, 0.0083, 330.,  0.0, 0.0, -39.0, 0.48,   
     5 41.13, 0.59, 0.0074, 278., 10.5, 0.5, -41.2, 0.46/   
      DATA ((BCB(I,J),J=1,5),I=1,2)/    
     1 10.79, -0.049, 0.040, 21.5, 1.23,    
     2  9.92, -0.027, 0.013, 18.9, 1.07/    
      DATA BCC/2.0164346,-0.5590311,0.0376279/  
    
C...Total cross-section and nuclear slope parameter for pp and p-pbar   
      NFIT=MIN(5,MAX(1,MSTP(31)))   
      SIGP=BCS(NFIT,1)+BCS(NFIT,2)*(-0.25*PARU(1)**2*   
     &(1.-0.25*BCS(NFIT,3)*PARU(1)**2)+(1.+0.5*BCS(NFIT,3)*PARU(1)**2)* 
     &(LOG(VINT(2)/BCS(NFIT,4)))**2+BCS(NFIT,3)*    
     &(LOG(VINT(2)/BCS(NFIT,4)))**4)/   
     &((1.-0.25*BCS(NFIT,3)*PARU(1)**2)**2+2.*BCS(NFIT,3)*  
     &(1.+0.25*BCS(NFIT,3)*PARU(1)**2)*(LOG(VINT(2)/BCS(NFIT,4)))**2+   
     &BCS(NFIT,3)**2*(LOG(VINT(2)/BCS(NFIT,4)))**4)+BCS(NFIT,5)*    
     &VINT(2)**(BCS(NFIT,6)-1.)*SIN(0.5*PARU(1)*BCS(NFIT,6))    
      SIGM=-BCS(NFIT,7)*VINT(2)**(BCS(NFIT,8)-1.)*  
     &COS(0.5*PARU(1)*BCS(NFIT,8))  
      REFP=BCS(NFIT,2)*PARU(1)*LOG(VINT(2)/BCS(NFIT,4))/    
     &((1.-0.25*BCS(NFIT,3)*PARU(1)**2)**2+2.*BCS(NFIT,3)*  
     &(1.+0.25*BCS(NFIT,3)*PARU(1)**2)+(LOG(VINT(2)/BCS(NFIT,4)))**2+   
     &BCS(NFIT,3)**2*(LOG(VINT(2)/BCS(NFIT,4)))**4)-BCS(NFIT,5)*    
     &VINT(2)**(BCS(NFIT,6)-1.)*COS(0.5*PARU(1)*BCS(NFIT,6))    
      REFM=-BCS(NFIT,7)*VINT(2)**(BCS(NFIT,8)-1.)*  
     &SIN(0.5*PARU(1)*BCS(NFIT,8))  
      SIGMA=SIGP-ISIGN(1,MINT(11)*MINT(12))*SIGM    
      RHO=(REFP-ISIGN(1,MINT(11)*MINT(12))*REFM)/SIGMA  
    
C...Nuclear slope parameter B, curvature C: 
      NFIT=1    
      IF(MSTP(31).GE.4) NFIT=2  
      BP=BCB(NFIT,1)+BCB(NFIT,2)*LOG(VINT(2))+  
     &BCB(NFIT,3)*(LOG(VINT(2)))**2 
      BM=BCB(NFIT,4)+BCB(NFIT,5)*LOG(VINT(2))   
      B=BP-ISIGN(1,MINT(11)*MINT(12))*SIGM/SIGP*(BM-BP) 
      VINT(121)=B   
      C=-0.5*BCC(2)/BCC(3)*(1.-SQRT(MAX(0.,1.+4.*BCC(3)/BCC(2)**2*  
     &(1.E-03*VINT(1)-BCC(1)))))    
      VINT(122)=C   
    
C...Elastic scattering cross-section (fixed by sigma-tot, rho and B).   
      SIGEL=SIGMA**2*(1.+RHO**2)/(16.*PARU(1)*PARU(5)*B)    
    
C...Single diffractive scattering cross-section from Goulianos: 
      SIGSD=2.*0.68*(1.+36./VINT(2))*LOG(0.6+0.1*VINT(2))   
    
C...Double diffractive scattering cross-section (essentially fixed by   
C...sigma-sd and sigma-el). 
      SIGDD=SIGSD**2/(3.*SIGEL) 
    
C...Total non-elastic, non-diffractive cross-section.   
      SIGND=SIGMA-SIGDD-SIGSD-SIGEL 
    
C...Rescale for pions.  
      IF(IABS(MINT(11)).EQ.211.AND.IABS(MINT(12)).EQ.211) THEN  
        SIGMA=4./9.*SIGMA   
        SIGDD=4./9.*SIGDD   
        SIGSD=4./9.*SIGSD   
        SIGEL=4./9.*SIGEL   
        SIGND=4./9.*SIGND   
      ELSEIF(IABS(MINT(11)).EQ.211.OR.IABS(MINT(12)).EQ.211) THEN   
        SIGMA=2./3.*SIGMA   
        SIGDD=2./3.*SIGDD   
        SIGSD=2./3.*SIGSD   
        SIGEL=2./3.*SIGEL   
        SIGND=2./3.*SIGND   
      ENDIF 
    
C...Save cross-sections in common block PYPARA. 
      VINT(101)=SIGMA   
      VINT(102)=SIGEL   
      VINT(103)=SIGSD   
      VINT(104)=SIGDD   
      VINT(106)=SIGND   
      XSEC(95,1)=SIGND  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYMAXI 
    
C...Finds optimal set of coefficients for kinematical variable selection    
C...and the maximum of the part of the differential cross-section used  
C...in the event weighting. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      COMMON/PYINT6/PROC(0:200) 
      CHARACTER PROC*28 
      SAVE /PYINT6/ 
      CHARACTER CVAR(4)*4   
      DIMENSION NPTS(4),MVARPT(200,4),VINTPT(200,30),SIGSPT(200),   
     &NAREL(6),WTREL(6),WTMAT(6,6),COEFU(6),IACCMX(4),SIGSMX(4),    
     &SIGSSM(3) 
      DATA CVAR/'tau ','tau''','y*  ','cth '/   
      INTEGER :: IOFF=0
C...Select subprocess to study: skip cases not applicable.  
      VINT(143)=1.  
      VINT(144)=1.  
      XSEC(0,1)=0.  
      DO 350 ISUB=1,200 
      IF(ISUB.GE.91.AND.ISUB.LE.95) THEN    
        XSEC(ISUB,1)=VINT(ISUB+11)  
        IF(MSUB(ISUB).NE.1) GOTO 350    
        GOTO 340    
      ELSEIF(ISUB.EQ.96) THEN   
        IF(MINT(43).NE.4) GOTO 350  
        IF(MSUB(95).NE.1.AND.MSTP(81).LE.0.AND.MSTP(131).LE.0) GOTO 350 
      ELSEIF(ISUB.EQ.11.OR.ISUB.EQ.12.OR.ISUB.EQ.13.OR.ISUB.EQ.28.OR.   
     &ISUB.EQ.53.OR.ISUB.EQ.68) THEN    
        IF(MSUB(ISUB).NE.1.OR.MSUB(95).EQ.1) GOTO 350   
      ELSE  
        IF(MSUB(ISUB).NE.1) GOTO 350    
      ENDIF 
      MINT(1)=ISUB  
      ISTSB=ISET(ISUB)  
      IF(ISUB.EQ.96) ISTSB=2    
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1000) ISUB  
    
C...Find resonances (explicit or implicit in cross-section).    
      MINT(72)=0    
      KFR1=0    
cms.. reinitializing to avoid compiler warning
      TAUR2=PMAS(KFR2,1)**2/VINT(2)
      GAMR2=PMAS(KFR2,1)*PMAS(KFR2,2)/VINT(2)
      TAUR1=0.
      GAMR1=0.
      IF(ISTSB.EQ.1.OR.ISTSB.EQ.3) THEN 
        KFR1=KFPR(ISUB,1)   
      ELSEIF(ISUB.GE.71.AND.ISUB.LE.77) THEN    
        KFR1=25 
      ENDIF 
      IF(KFR1.NE.0) THEN    
        TAUR1=PMAS(KFR1,1)**2/VINT(2)   
        GAMR1=PMAS(KFR1,1)*PMAS(KFR1,2)/VINT(2) 
        MINT(72)=1  
        MINT(73)=KFR1   
        VINT(73)=TAUR1  
        VINT(74)=GAMR1  
      ENDIF 
      IF(ISUB.EQ.141) THEN  
        KFR2=23 
        TAUR2=PMAS(KFR2,1)**2/VINT(2)   
        GAMR2=PMAS(KFR2,1)*PMAS(KFR2,2)/VINT(2) 
        MINT(72)=2  
        MINT(74)=KFR2   
        VINT(75)=TAUR2  
        VINT(76)=GAMR2  
      ENDIF 
    
C...Find product masses and minimum pT of process.  
      SQM3=0.   
      SQM4=0.   
      MINT(71)=0    
      VINT(71)=CKIN(3)  
      IF(ISTSB.EQ.2.OR.ISTSB.EQ.4) THEN 
        IF(KFPR(ISUB,1).NE.0) SQM3=PMAS(KFPR(ISUB,1),1)**2  
        IF(KFPR(ISUB,2).NE.0) SQM4=PMAS(KFPR(ISUB,2),1)**2  
        IF(MIN(SQM3,SQM4).LT.CKIN(6)**2) MINT(71)=1 
        IF(MINT(71).EQ.1) VINT(71)=MAX(CKIN(3),CKIN(5)) 
        IF(ISUB.EQ.96.AND.MSTP(82).LE.1) VINT(71)=PARP(81)  
        IF(ISUB.EQ.96.AND.MSTP(82).GE.2) VINT(71)=0.08*PARP(82) 
      ENDIF 
      VINT(63)=SQM3 
      VINT(64)=SQM4 
    
C...Number of points for each variable: tau, tau', y*, cos(theta-hat).  
      NPTS(1)=2+2*MINT(72)  
      IF(MINT(43).EQ.1.AND.(ISTSB.EQ.1.OR.ISTSB.EQ.2)) NPTS(1)=1    
      NPTS(2)=1 
      IF(MINT(43).GE.2.AND.(ISTSB.EQ.3.OR.ISTSB.EQ.4)) NPTS(2)=2    
      NPTS(3)=1 
      IF(MINT(43).EQ.4) NPTS(3)=3   
      NPTS(4)=1 
      IF(ISTSB.EQ.2.OR.ISTSB.EQ.4) NPTS(4)=5    
      NTRY=NPTS(1)*NPTS(2)*NPTS(3)*NPTS(4)  
    
C...Reset coefficients of cross-section weighting.  
      DO 100 J=1,20 
  100 COEF(ISUB,J)=0.   
      COEF(ISUB,1)=1.   
      COEF(ISUB,7)=0.5  
      COEF(ISUB,8)=0.5  
      COEF(ISUB,10)=1.  
      COEF(ISUB,15)=1.  
      MCTH=0    
      MTAUP=0   
      CTH=0.    
      TAUP=0.   
      SIGSAM=0. 
    
C...Find limits and select tau, y*, cos(theta-hat) and tau' values, 
C...in grid of phase space points.  
      CALL PYKLIM(1)    
      NACC=0    
      DO 120 ITRY=1,NTRY    
      IF(MOD(ITRY-1,NPTS(2)*NPTS(3)*NPTS(4)).EQ.0) THEN 
        MTAU=1+(ITRY-1)/(NPTS(2)*NPTS(3)*NPTS(4))   
        CALL PYKMAP(1,MTAU,0.5) 
        IF(ISTSB.EQ.3.OR.ISTSB.EQ.4) CALL PYKLIM(4) 
      ENDIF 
      IF((ISTSB.EQ.3.OR.ISTSB.EQ.4).AND.MOD(ITRY-1,NPTS(3)*NPTS(4)).    
     &EQ.0) THEN    
        MTAUP=1+MOD((ITRY-1)/(NPTS(3)*NPTS(4)),NPTS(2)) 
        CALL PYKMAP(4,MTAUP,0.5)    
      ENDIF 
      IF(MOD(ITRY-1,NPTS(3)*NPTS(4)).EQ.0) CALL PYKLIM(2)   
      IF(MOD(ITRY-1,NPTS(4)).EQ.0) THEN 
        MYST=1+MOD((ITRY-1)/NPTS(4),NPTS(3))    
        CALL PYKMAP(2,MYST,0.5) 
        CALL PYKLIM(3)  
      ENDIF 
      IF(ISTSB.EQ.2.OR.ISTSB.EQ.4) THEN 
        MCTH=1+MOD(ITRY-1,NPTS(4))  
        CALL PYKMAP(3,MCTH,0.5) 
      ENDIF 
      IF(ISUB.EQ.96) VINT(25)=VINT(21)*(1.-VINT(23)**2) 
    
C...Calculate and store cross-section.  
      MINT(51)=0    
      CALL PYKLIM(0)    
      IF(MINT(51).EQ.1) GOTO 120    
      NACC=NACC+1   
      MVARPT(NACC,1)=MTAU   
      MVARPT(NACC,2)=MTAUP  
      MVARPT(NACC,3)=MYST   
      MVARPT(NACC,4)=MCTH   
      DO 110 J=1,30 
  110 VINTPT(NACC,J)=VINT(10+J) 
      CALL PYSIGH(NCHN,SIGS)    
      SIGSPT(NACC)=SIGS 
      IF(SIGS.GT.SIGSAM) SIGSAM=SIGS    
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1100) MTAU,MTAUP,MYST,MCTH, 
     &VINT(21),VINT(22),VINT(23),VINT(26),SIGS  
  120 CONTINUE  
      IF(SIGSAM.EQ.0.) THEN 
        WRITE(MSTU(11),1200) ISUB   
        STOP    
      ENDIF 
    
C...Calculate integrals in tau and y* over maximal phase space limits.  
      TAUMIN=VINT(11)   
      TAUMAX=VINT(31)   
      ATAU1=LOG(TAUMAX/TAUMIN)  
      ATAU2=(TAUMAX-TAUMIN)/(TAUMAX*TAUMIN)
cms.. declare ataus outside to avoid compiler warning
      ATAU3=0.
      ATAU4=0.
      ATAU5=0.
      ATAU6=0.
      IF(NPTS(1).GE.3) THEN 
        ATAU3=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR1)/(TAUMAX+TAUR1))/TAUR1    
        ATAU4=(ATAN((TAUMAX-TAUR1)/GAMR1)-ATAN((TAUMIN-TAUR1)/GAMR1))/  
     &  GAMR1   
      ENDIF 
      IF(NPTS(1).GE.5) THEN 
        ATAU5=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR2)/(TAUMAX+TAUR2))/TAUR2    
        ATAU6=(ATAN((TAUMAX-TAUR2)/GAMR2)-ATAN((TAUMIN-TAUR2)/GAMR2))/  
     &  GAMR2   
      ENDIF 
      YSTMIN=0.5*LOG(TAUMIN)    
      YSTMAX=-YSTMIN    
      AYST0=YSTMAX-YSTMIN   
      AYST1=0.5*(YSTMAX-YSTMIN)**2  
      AYST3=2.*(ATAN(EXP(YSTMAX))-ATAN(EXP(YSTMIN)))    
    
C...Reset. Sum up cross-sections in points calculated.  
      DO 230 IVAR=1,4   
      IF(NPTS(IVAR).EQ.1) GOTO 230  
      IF(ISUB.EQ.96.AND.IVAR.EQ.4) GOTO 230 
      NBIN=NPTS(IVAR)   
      DO 130 J1=1,NBIN  
      NAREL(J1)=0   
      WTREL(J1)=0.  
      COEFU(J1)=0.  
      DO 130 J2=1,NBIN  
  130 WTMAT(J1,J2)=0.   
      DO 140 IACC=1,NACC    
      IBIN=MVARPT(IACC,IVAR)    
      NAREL(IBIN)=NAREL(IBIN)+1 
      WTREL(IBIN)=WTREL(IBIN)+SIGSPT(IACC)  
    
C...Sum up tau cross-section pieces in points used. 
      IF(IVAR.EQ.1) THEN    
        TAU=VINTPT(IACC,11) 
        WTMAT(IBIN,1)=WTMAT(IBIN,1)+1.  
        WTMAT(IBIN,2)=WTMAT(IBIN,2)+(ATAU1/ATAU2)/TAU   
        IF(NBIN.GE.3) THEN  
          WTMAT(IBIN,3)=WTMAT(IBIN,3)+(ATAU1/ATAU3)/(TAU+TAUR1) 
          WTMAT(IBIN,4)=WTMAT(IBIN,4)+(ATAU1/ATAU4)*TAU/    
     &    ((TAU-TAUR1)**2+GAMR1**2) 
        ENDIF   
        IF(NBIN.GE.5) THEN  
          WTMAT(IBIN,5)=WTMAT(IBIN,5)+(ATAU1/ATAU5)/(TAU+TAUR2) 
          WTMAT(IBIN,6)=WTMAT(IBIN,6)+(ATAU1/ATAU6)*TAU/    
     &    ((TAU-TAUR2)**2+GAMR2**2) 
        ENDIF   
    
C...Sum up tau' cross-section pieces in points used.    
      ELSEIF(IVAR.EQ.2) THEN    
        TAU=VINTPT(IACC,11) 
        TAUP=VINTPT(IACC,16)    
        TAUPMN=VINTPT(IACC,6)   
        TAUPMX=VINTPT(IACC,26)  
        ATAUP1=LOG(TAUPMX/TAUPMN)   
        ATAUP2=((1.-TAU/TAUPMX)**4-(1.-TAU/TAUPMN)**4)/(4.*TAU) 
        WTMAT(IBIN,1)=WTMAT(IBIN,1)+1.  
        WTMAT(IBIN,2)=WTMAT(IBIN,2)+(ATAUP1/ATAUP2)*(1.-TAU/TAUP)**3/   
     &  TAUP    
    
C...Sum up y* and cos(theta-hat) cross-section pieces in points used.   
      ELSEIF(IVAR.EQ.3) THEN    
        YST=VINTPT(IACC,12) 
        WTMAT(IBIN,1)=WTMAT(IBIN,1)+(AYST0/AYST1)*(YST-YSTMIN)  
        WTMAT(IBIN,2)=WTMAT(IBIN,2)+(AYST0/AYST1)*(YSTMAX-YST)  
        WTMAT(IBIN,3)=WTMAT(IBIN,3)+(AYST0/AYST3)/COSH(YST) 
      ELSE  
        RM34=2.*SQM3*SQM4/(VINTPT(IACC,11)*VINT(2))**2  
        RSQM=1.+RM34    
        CTHMAX=SQRT(1.-4.*VINT(71)**2/(TAUMAX*VINT(2))) 
        CTHMIN=-CTHMAX  
        IF(CTHMAX.GT.0.9999) RM34=MAX(RM34,2.*VINT(71)**2/  
     &  (TAUMAX*VINT(2)))   
        ACTH1=CTHMAX-CTHMIN 
        ACTH2=LOG(MAX(RM34,RSQM-CTHMIN)/MAX(RM34,RSQM-CTHMAX))  
        ACTH3=LOG(MAX(RM34,RSQM+CTHMAX)/MAX(RM34,RSQM+CTHMIN))  
        ACTH4=1./MAX(RM34,RSQM-CTHMAX)-1./MAX(RM34,RSQM-CTHMIN) 
        ACTH5=1./MAX(RM34,RSQM+CTHMIN)-1./MAX(RM34,RSQM+CTHMAX) 
        CTH=VINTPT(IACC,13) 
        WTMAT(IBIN,1)=WTMAT(IBIN,1)+1.  
        WTMAT(IBIN,2)=WTMAT(IBIN,2)+(ACTH1/ACTH2)/MAX(RM34,RSQM-CTH)    
        WTMAT(IBIN,3)=WTMAT(IBIN,3)+(ACTH1/ACTH3)/MAX(RM34,RSQM+CTH)    
        WTMAT(IBIN,4)=WTMAT(IBIN,4)+(ACTH1/ACTH4)/MAX(RM34,RSQM-CTH)**2 
        WTMAT(IBIN,5)=WTMAT(IBIN,5)+(ACTH1/ACTH5)/MAX(RM34,RSQM+CTH)**2 
      ENDIF 
  140 CONTINUE  
    
C...Check that equation system solvable; else trivial way out.  
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1300) CVAR(IVAR)    
      MSOLV=1   
      DO 150 IBIN=1,NBIN    
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1400) (WTMAT(IBIN,IRED),    
     &IRED=1,NBIN),WTREL(IBIN)  
  150 IF(NAREL(IBIN).EQ.0) MSOLV=0  
      IF(MSOLV.EQ.0) THEN   
        DO 160 IBIN=1,NBIN  
  160   COEFU(IBIN)=1.  
    
C...Solve to find relative importance of cross-section pieces.  
      ELSE  
        DO 170 IRED=1,NBIN-1    
        DO 170 IBIN=IRED+1,NBIN 
        RQT=WTMAT(IBIN,IRED)/WTMAT(IRED,IRED)   
        WTREL(IBIN)=WTREL(IBIN)-RQT*WTREL(IRED) 
        DO 170 ICOE=IRED,NBIN   
  170   WTMAT(IBIN,ICOE)=WTMAT(IBIN,ICOE)-RQT*WTMAT(IRED,ICOE)  
        DO 190 IRED=NBIN,1,-1   
        DO 180 ICOE=IRED+1,NBIN 
  180   WTREL(IRED)=WTREL(IRED)-WTMAT(IRED,ICOE)*COEFU(ICOE)    
  190   COEFU(IRED)=WTREL(IRED)/WTMAT(IRED,IRED)    
      ENDIF 
    
C...Normalize coefficients, with piece shared democratically.   
      COEFSU=0. 
      DO 200 IBIN=1,NBIN    
      COEFU(IBIN)=MAX(0.,COEFU(IBIN))   
  200 COEFSU=COEFSU+COEFU(IBIN) 
      IF(IVAR.EQ.1) IOFF=0  
      IF(IVAR.EQ.2) IOFF=14 
      IF(IVAR.EQ.3) IOFF=6  
      IF(IVAR.EQ.4) IOFF=9  
      IF(COEFSU.GT.0.) THEN 
        DO 210 IBIN=1,NBIN  
  210   COEF(ISUB,IOFF+IBIN)=PARP(121)/NBIN+(1.-PARP(121))*COEFU(IBIN)/ 
     &  COEFSU  
      ELSE  
        DO 220 IBIN=1,NBIN  
  220   COEF(ISUB,IOFF+IBIN)=1./NBIN    
      ENDIF 
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1500) CVAR(IVAR),   
     &(COEF(ISUB,IOFF+IBIN),IBIN=1,NBIN)    
  230 CONTINUE  
    
C...Find two most promising maxima among points previously determined.  
      DO 240 J=1,4  
      IACCMX(J)=0   
  240 SIGSMX(J)=0.  
      NMAX=0    
      DO 290 IACC=1,NACC    
      DO 250 J=1,30 
  250 VINT(10+J)=VINTPT(IACC,J) 
      CALL PYSIGH(NCHN,SIGS)    
      IEQ=0 
      DO 260 IMV=1,NMAX 
  260 IF(ABS(SIGS-SIGSMX(IMV)).LT.1E-4*(SIGS+SIGSMX(IMV))) IEQ=IMV  
      IF(IEQ.EQ.0) THEN 
        DO 270 IMV=NMAX,1,-1    
        IIN=IMV+1   
        IF(SIGS.LE.SIGSMX(IMV)) GOTO 280    
        IACCMX(IMV+1)=IACCMX(IMV)   
  270   SIGSMX(IMV+1)=SIGSMX(IMV)   
        IIN=1   
  280   IACCMX(IIN)=IACC    
        SIGSMX(IIN)=SIGS    
        IF(NMAX.LE.1) NMAX=NMAX+1   
      ENDIF 
  290 CONTINUE  
    
C...Read out starting position for search.  
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1600)   
      SIGSAM=SIGSMX(1)  
      DO 330 IMAX=1,NMAX    
      IACC=IACCMX(IMAX) 
      MTAU=MVARPT(IACC,1)   
      MTAUP=MVARPT(IACC,2)  
      MYST=MVARPT(IACC,3)   
      MCTH=MVARPT(IACC,4)   
      VTAU=0.5  
      VYST=0.5  
      VCTH=0.5  
      VTAUP=0.5 
    
C...Starting point and step size in parameter space.    
      DO 320 IRPT=1,2   
      DO 310 IVAR=1,4   
      IF(NPTS(IVAR).EQ.1) GOTO 310  
      IF(IVAR.EQ.1) VVAR=VTAU   
      IF(IVAR.EQ.2) VVAR=VTAUP  
      IF(IVAR.EQ.3) VVAR=VYST   
      IF(IVAR.EQ.4) VVAR=VCTH   
      IF(IVAR.EQ.1) MVAR=MTAU   
      IF(IVAR.EQ.2) MVAR=MTAUP  
      IF(IVAR.EQ.3) MVAR=MYST   
      IF(IVAR.EQ.4) MVAR=MCTH   
      IF(IRPT.EQ.1) VDEL=0.1    
      IF(IRPT.EQ.2) VDEL=MAX(0.01,MIN(0.05,VVAR-0.02,0.98-VVAR))    
      IF(IRPT.EQ.1) VMAR=0.02   
      IF(IRPT.EQ.2) VMAR=0.002  
      IMOV0=1   
      IF(IRPT.EQ.1.AND.IVAR.EQ.1) IMOV0=0   
      DO 300 IMOV=IMOV0,8   
    
C...Define new point in parameter space.    
      IF(IMOV.EQ.0) THEN    
        INEW=2  
        VNEW=VVAR   
      ELSEIF(IMOV.EQ.1) THEN    
        INEW=3  
        VNEW=VVAR+VDEL  
      ELSEIF(IMOV.EQ.2) THEN    
        INEW=1  
        VNEW=VVAR-VDEL  
      ELSEIF(SIGSSM(3).GE.MAX(SIGSSM(1),SIGSSM(2)).AND. 
     &VVAR+2.*VDEL.LT.1.-VMAR) THEN 
        VVAR=VVAR+VDEL  
        SIGSSM(1)=SIGSSM(2) 
        SIGSSM(2)=SIGSSM(3) 
        INEW=3  
        VNEW=VVAR+VDEL  
      ELSEIF(SIGSSM(1).GE.MAX(SIGSSM(2),SIGSSM(3)).AND. 
     &VVAR-2.*VDEL.GT.VMAR) THEN    
        VVAR=VVAR-VDEL  
        SIGSSM(3)=SIGSSM(2) 
        SIGSSM(2)=SIGSSM(1) 
        INEW=1  
        VNEW=VVAR-VDEL  
      ELSEIF(SIGSSM(3).GE.SIGSSM(1)) THEN   
        VDEL=0.5*VDEL   
        VVAR=VVAR+VDEL  
        SIGSSM(1)=SIGSSM(2) 
        INEW=2  
        VNEW=VVAR   
      ELSE  
        VDEL=0.5*VDEL   
        VVAR=VVAR-VDEL  
        SIGSSM(3)=SIGSSM(2) 
        INEW=2  
        VNEW=VVAR   
      ENDIF 
    
C...Convert to relevant variables and find derived new limits.  
      IF(IVAR.EQ.1) THEN    
        VTAU=VNEW   
        CALL PYKMAP(1,MTAU,VTAU)    
        IF(ISTSB.EQ.3.OR.ISTSB.EQ.4) CALL PYKLIM(4) 
      ENDIF 
      IF(IVAR.LE.2.AND.(ISTSB.EQ.3.OR.ISTSB.EQ.4)) THEN 
        IF(IVAR.EQ.2) VTAUP=VNEW    
        CALL PYKMAP(4,MTAUP,VTAUP)  
      ENDIF 
      IF(IVAR.LE.2) CALL PYKLIM(2)  
      IF(IVAR.LE.3) THEN    
        IF(IVAR.EQ.3) VYST=VNEW 
        CALL PYKMAP(2,MYST,VYST)    
        CALL PYKLIM(3)  
      ENDIF 
      IF(ISTSB.EQ.2.OR.ISTSB.EQ.4) THEN 
        IF(IVAR.EQ.4) VCTH=VNEW 
        CALL PYKMAP(3,MCTH,VCTH)    
      ENDIF 
      IF(ISUB.EQ.96) VINT(25)=VINT(21)*(1.-VINT(23)**2) 
    
C...Evaluate cross-section. Save new maximum. Final maximum.    
      CALL PYSIGH(NCHN,SIGS)    
      SIGSSM(INEW)=SIGS 
      IF(SIGS.GT.SIGSAM) SIGSAM=SIGS    
      IF(MSTP(122).GE.2) WRITE(MSTU(11),1700) IMAX,IVAR,MVAR,IMOV,  
     &VNEW,VINT(21),VINT(22),VINT(23),VINT(26),SIGS 
  300 CONTINUE  
  310 CONTINUE  
  320 CONTINUE  
      IF(IMAX.EQ.1) SIGS11=SIGSAM   
  330 CONTINUE  
      XSEC(ISUB,1)=1.05*SIGSAM  
  340 IF(ISUB.NE.96) XSEC(0,1)=XSEC(0,1)+XSEC(ISUB,1)   
  350 CONTINUE  
    
C...Print summary table.    
      IF(MSTP(122).GE.1) THEN   
        WRITE(MSTU(11),1800)    
        WRITE(MSTU(11),1900)    
        DO 360 ISUB=1,200   
        IF(MSUB(ISUB).NE.1.AND.ISUB.NE.96) GOTO 360 
        IF(ISUB.EQ.96.AND.MINT(43).NE.4) GOTO 360   
        IF(ISUB.EQ.96.AND.MSUB(95).NE.1.AND.MSTP(81).LE.0) GOTO 360 
        IF(MSUB(95).EQ.1.AND.(ISUB.EQ.11.OR.ISUB.EQ.12.OR.ISUB.EQ.13.OR.    
     &  ISUB.EQ.28.OR.ISUB.EQ.53.OR.ISUB.EQ.68)) GOTO 360   
        WRITE(MSTU(11),2000) ISUB,PROC(ISUB),XSEC(ISUB,1)   
  360   CONTINUE    
        WRITE(MSTU(11),2100)    
      ENDIF 
    
C...Format statements for maximization results. 
 1000 FORMAT(/1X,'Coefficient optimization and maximum search for ',    
     &'subprocess no',I4/1X,'Coefficient modes     tau',10X,'y*',9X,    
     &'cth',9X,'tau''',7X,'sigma')  
 1100 FORMAT(1X,4I4,F12.8,F12.6,F12.7,F12.8,1P,E12.4)   
 1200 FORMAT(1X,'Error: requested subprocess ',I3,' has vanishing ',    
     &'cross-section.'/1X,'Execution stopped!')
 1300 FORMAT(1X,'Coefficients of equation system to be solved for ',A4) 
 1400 FORMAT(1X,1P,7E11.3)  
 1500 FORMAT(1X,'Result for ',A4,':',6F9.4) 
 1600 FORMAT(1X,'Maximum search for given coefficients'/2X,'MAX VAR ',  
     &'MOD MOV   VNEW',7X,'tau',7X,'y*',8X,'cth',7X,'tau''',7X,'sigma') 
 1700 FORMAT(1X,4I4,F8.4,F11.7,F9.3,F11.6,F11.7,1P,E12.4)   
 1800 FORMAT(/1X,8('*'),1X,'PYMAXI: summary of differential ',  
     &'cross-section maximum search',1X,8('*')) 
 1900 FORMAT(/11X,58('=')/11X,'I',38X,'I',17X,'I'/11X,'I  ISUB  ',  
     &'Subprocess name',15X,'I  Maximum value  I'/11X,'I',38X,'I',  
     &17X,'I'/11X,58('=')/11X,'I',38X,'I',17X,'I')  
 2000 FORMAT(11X,'I',2X,I3,3X,A28,2X,'I',2X,1P,E12.4,3X,'I')    
 2100 FORMAT(11X,'I',38X,'I',17X,'I'/11X,58('='))   
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYOVLY(MOVLY)  
    
C...Initializes multiplicity distribution and selects mutliplicity  
C...of overlayed events, i.e. several events occuring at the same   
C...beam crossing.  
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      DIMENSION WTI(0:100)  
      SAVE IMAX,WTI,WTS 
    
C...Sum of allowed cross-sections for overlayed events. 
      IF(MOVLY.EQ.1) THEN   
        VINT(131)=VINT(106) 
        IF(MSTP(132).GE.2) VINT(131)=VINT(131)+VINT(104)    
        IF(MSTP(132).GE.3) VINT(131)=VINT(131)+VINT(103)    
        IF(MSTP(132).GE.4) VINT(131)=VINT(131)+VINT(102)    
    
C...Initialize multiplicity distribution for unbiased events.   
        IF(MSTP(133).EQ.1) THEN 
          XNAVE=VINT(131)*PARP(131) 
          IF(XNAVE.GT.40.) WRITE(MSTU(11),1000) XNAVE   
          WTI(0)=EXP(-MIN(50.,XNAVE))   
          WTS=0.    
          WTN=0.    
          DO 100 I=1,100    
          WTI(I)=WTI(I-1)*XNAVE/I   
          IF(I-2.5.GT.XNAVE.AND.WTI(I).LT.1E-6) GOTO 110    
          WTS=WTS+WTI(I)    
          WTN=WTN+WTI(I)*I  
  100     IMAX=I    
  110     VINT(132)=XNAVE   
          VINT(133)=WTN/WTS 
          VINT(134)=WTS 
    
C...Initialize mutiplicity distribution for biased events.  
        ELSEIF(MSTP(133).EQ.2) THEN 
          XNAVE=VINT(131)*PARP(131) 
          IF(XNAVE.GT.40.) WRITE(MSTU(11),1000) XNAVE   
          WTI(1)=EXP(-MIN(50.,XNAVE))*XNAVE 
          WTS=WTI(1)    
          WTN=WTI(1)    
          DO 120 I=2,100    
          WTI(I)=WTI(I-1)*XNAVE/(I-1)   
          IF(I-2.5.GT.XNAVE.AND.WTI(I).LT.1E-6) GOTO 130    
          WTS=WTS+WTI(I)    
          WTN=WTN+WTI(I)*I  
  120     IMAX=I    
  130     VINT(132)=XNAVE   
          VINT(133)=WTN/WTS 
          VINT(134)=WTS 
        ENDIF   
    
C...Pick multiplicity of overlayed events.  
      ELSE  
        IF(MSTP(133).EQ.0) THEN 
          MINT(81)=MAX(1,MSTP(134)) 
        ELSE    
          WTR=WTS*RLU(0)    
          DO 140 I=1,IMAX   
          MINT(81)=I    
          WTR=WTR-WTI(I)    
          IF(WTR.LE.0.) GOTO 150    
  140     CONTINUE  
  150     CONTINUE  
        ENDIF   
      ENDIF 
    
C...Format statement for error message. 
 1000 FORMAT(1X,'Warning: requested average number of events per bunch',    
     &'crossing too large, ',1P,E12.4)  
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYRAND 
    
C...Generates quantities characterizing the high-pT scattering at the   
C...parton level according to the matrix elements. Chooses incoming,    
C...reacting partons, their momentum fractions and one of the possible  
C...subprocesses.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
    
C...Initial values, specifically for (first) semihard interaction.  
      MINT(17)=0    
      MINT(18)=0    
      VINT(143)=1.  
      VINT(144)=1.  
      IF(MSUB(95).EQ.1.OR.MINT(82).GE.2) CALL PYMULT(2) 
      ISUB=0    
  100 MINT(51)=0    
    
C...Choice of process type - first event of overlay.    
      IF(MINT(82).EQ.1.AND.(ISUB.LE.90.OR.ISUB.GT.96)) THEN 
        RSUB=XSEC(0,1)*RLU(0)   
        DO 110 I=1,200  
        IF(MSUB(I).NE.1) GOTO 110   
        ISUB=I  
        RSUB=RSUB-XSEC(I,1) 
        IF(RSUB.LE.0.) GOTO 120 
  110   CONTINUE    
  120   IF(ISUB.EQ.95) ISUB=96  
    
C...Choice of inclusive process type - overlayed events.    
      ELSEIF(MINT(82).GE.2.AND.ISUB.EQ.0) THEN  
        RSUB=VINT(131)*RLU(0)   
        ISUB=96 
        IF(RSUB.GT.VINT(106)) ISUB=93   
        IF(RSUB.GT.VINT(106)+VINT(104)) ISUB=92 
        IF(RSUB.GT.VINT(106)+VINT(104)+VINT(103)) ISUB=91   
      ENDIF 
      IF(MINT(82).EQ.1) NGEN(0,1)=NGEN(0,1)+1   
      IF(MINT(82).EQ.1) NGEN(ISUB,1)=NGEN(ISUB,1)+1 
      MINT(1)=ISUB  
    
C...Find resonances (explicit or implicit in cross-section).    
      MINT(72)=0    
      KFR1=0    
      IF(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3) THEN   
        KFR1=KFPR(ISUB,1)   
      ELSEIF(ISUB.GE.71.AND.ISUB.LE.77) THEN    
        KFR1=25 
      ENDIF 
      IF(KFR1.NE.0) THEN    
        TAUR1=PMAS(KFR1,1)**2/VINT(2)   
        GAMR1=PMAS(KFR1,1)*PMAS(KFR1,2)/VINT(2) 
        MINT(72)=1  
        MINT(73)=KFR1   
        VINT(73)=TAUR1  
        VINT(74)=GAMR1  
      ENDIF 
      IF(ISUB.EQ.141) THEN  
        KFR2=23 
        TAUR2=PMAS(KFR2,1)**2/VINT(2)   
        GAMR2=PMAS(KFR2,1)*PMAS(KFR2,2)/VINT(2) 
        MINT(72)=2  
        MINT(74)=KFR2   
        VINT(75)=TAUR2  
        VINT(76)=GAMR2  
      ENDIF 
    
C...Find product masses and minimum pT of process,  
C...optionally with broadening according to a truncated Breit-Wigner.   
      VINT(63)=0.   
      VINT(64)=0.   
      MINT(71)=0    
      VINT(71)=CKIN(3)  
      IF(MINT(82).GE.2) VINT(71)=0. 
      IF(ISET(ISUB).EQ.2.OR.ISET(ISUB).EQ.4) THEN   
        DO 130 I=1,2    
        IF(KFPR(ISUB,I).EQ.0) THEN  
        ELSEIF(MSTP(42).LE.0) THEN  
          VINT(62+I)=PMAS(KFPR(ISUB,I),1)**2    
        ELSE    
          VINT(62+I)=ULMASS(KFPR(ISUB,I))**2    
        ENDIF   
  130   CONTINUE    
        IF(MIN(VINT(63),VINT(64)).LT.CKIN(6)**2) MINT(71)=1 
        IF(MINT(71).EQ.1) VINT(71)=MAX(CKIN(3),CKIN(5)) 
      ENDIF 
    
      IF(ISET(ISUB).EQ.0) THEN  
C...Double or single diffractive, or elastic scattering:    
C...choose m^2 according to 1/m^2 (diffractive), constant (elastic) 
        IS=INT(1.5+RLU(0))  
        VINT(63)=VINT(3)**2 
        VINT(64)=VINT(4)**2 
        IF(ISUB.EQ.92.OR.ISUB.EQ.93) VINT(62+IS)=PARP(111)**2   
        IF(ISUB.EQ.93) VINT(65-IS)=PARP(111)**2 
        SH=VINT(2)  
        SQM1=VINT(3)**2 
        SQM2=VINT(4)**2 
        SQM3=VINT(63)   
        SQM4=VINT(64)   
        SQLA12=(SH-SQM1-SQM2)**2-4.*SQM1*SQM2   
        SQLA34=(SH-SQM3-SQM4)**2-4.*SQM3*SQM4   
        THTER1=SQM1+SQM2+SQM3+SQM4-(SQM1-SQM2)*(SQM3-SQM4)/SH-SH    
        THTER2=SQRT(MAX(0.,SQLA12))*SQRT(MAX(0.,SQLA34))/SH 
        THL=0.5*(THTER1-THTER2) 
        THU=0.5*(THTER1+THTER2) 
        THM=MIN(MAX(THL,PARP(101)),THU) 
        JTMAX=0 
        IF(ISUB.EQ.92.OR.ISUB.EQ.93) JTMAX=ISUB-91  
        DO 140 JT=1,JTMAX   
        MINT(13+3*JT-IS*(2*JT-3))=1 
        SQMMIN=VINT(59+3*JT-IS*(2*JT-3))    
        SQMI=VINT(8-3*JT+IS*(2*JT-3))**2    
        SQMJ=VINT(3*JT-1-IS*(2*JT-3))**2    
        SQMF=VINT(68-3*JT+IS*(2*JT-3))  
        SQUA=0.5*SH/SQMI*((1.+(SQMI-SQMJ)/SH)*THM+SQMI-SQMF-    
     &  SQMJ**2/SH+(SQMI+SQMJ)*SQMF/SH+(SQMI-SQMJ)**2/SH**2*SQMF)   
        QUAR=SH/SQMI*(THM*(THM+SH-SQMI-SQMJ-SQMF*(1.-(SQMI-SQMJ)/SH))+  
     &  SQMI*SQMJ-SQMJ*SQMF*(1.+(SQMI-SQMJ-SQMF)/SH))   
        SQMMAX=SQUA+SQRT(MAX(0.,SQUA**2-QUAR))  
        IF(ABS(QUAR/SQUA**2).LT.1.E-06) SQMMAX=0.5*QUAR/SQUA    
        SQMMAX=MIN(SQMMAX,(VINT(1)-SQRT(SQMF))**2)  
        VINT(59+3*JT-IS*(2*JT-3))=SQMMIN*(SQMMAX/SQMMIN)**RLU(0)    
  140   CONTINUE    
C...Choose t-hat according to exp(B*t-hat+C*t-hat^2).   
        SQM3=VINT(63)   
        SQM4=VINT(64)   
        SQLA34=(SH-SQM3-SQM4)**2-4.*SQM3*SQM4   
        THTER1=SQM1+SQM2+SQM3+SQM4-(SQM1-SQM2)*(SQM3-SQM4)/SH-SH    
        THTER2=SQRT(MAX(0.,SQLA12))*SQRT(MAX(0.,SQLA34))/SH 
        THL=0.5*(THTER1-THTER2) 
        THU=0.5*(THTER1+THTER2) 
        B=VINT(121) 
        C=VINT(122) 
        IF(ISUB.EQ.92.OR.ISUB.EQ.93) THEN   
          B=0.5*B   
          C=0.5*C   
        ENDIF   
        THM=MIN(MAX(THL,PARP(101)),THU) 
        EXPTH=0.    
        THARG=B*(THM-THU)   
        IF(THARG.GT.-20.) EXPTH=EXP(THARG)  
  150   TH=THU+LOG(EXPTH+(1.-EXPTH)*RLU(0))/B   
        TH=MAX(THM,MIN(THU,TH)) 
        RATLOG=MIN((B+C*(TH+THM))*(TH-THM),(B+C*(TH+THU))*(TH-THU)) 
        IF(RATLOG.LT.LOG(RLU(0))) GOTO 150  
        VINT(21)=1. 
        VINT(22)=0. 
        VINT(23)=MIN(1.,MAX(-1.,(2.*TH-THTER1)/THTER2)) 
    
C...Note: in the following, by In is meant the integral over the    
C...quantity multiplying coefficient cn.    
C...Choose tau according to h1(tau)/tau, where  
C...h1(tau) = c0 + I0/I1*c1*1/tau + I0/I2*c2*1/(tau+tau_R) +    
C...I0/I3*c3*tau/((s*tau-m^2)^2+(m*Gamma)^2) +  
C...I0/I4*c4*1/(tau+tau_R') +   
C...I0/I5*c5*tau/((s*tau-m'^2)^2+(m'*Gamma')^2), and    
C...c0 + c1 + c2 + c3 + c4 + c5 = 1 
      ELSEIF(ISET(ISUB).GE.1.AND.ISET(ISUB).LE.4) THEN  
        CALL PYKLIM(1)  
        IF(MINT(51).NE.0) GOTO 100  
        RTAU=RLU(0) 
        MTAU=1  
        IF(RTAU.GT.COEF(ISUB,1)) MTAU=2 
        IF(RTAU.GT.COEF(ISUB,1)+COEF(ISUB,2)) MTAU=3    
        IF(RTAU.GT.COEF(ISUB,1)+COEF(ISUB,2)+COEF(ISUB,3)) MTAU=4   
        IF(RTAU.GT.COEF(ISUB,1)+COEF(ISUB,2)+COEF(ISUB,3)+COEF(ISUB,4)) 
     &  MTAU=5  
        IF(RTAU.GT.COEF(ISUB,1)+COEF(ISUB,2)+COEF(ISUB,3)+COEF(ISUB,4)+ 
     &  COEF(ISUB,5)) MTAU=6    
        CALL PYKMAP(1,MTAU,RLU(0))  
    
C...2 -> 3, 4 processes:    
C...Choose tau' according to h4(tau,tau')/tau', where   
C...h4(tau,tau') = c0 + I0/I1*c1*(1 - tau/tau')^3/tau', and 
C...c0 + c1 = 1.    
        IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) THEN 
          CALL PYKLIM(4)    
          IF(MINT(51).NE.0) GOTO 100    
          RTAUP=RLU(0)  
          MTAUP=1   
          IF(RTAUP.GT.COEF(ISUB,15)) MTAUP=2    
          CALL PYKMAP(4,MTAUP,RLU(0))   
        ENDIF   
    
C...Choose y* according to h2(y*), where    
C...h2(y*) = I0/I1*c1*(y*-y*min) + I0/I2*c2*(y*max-y*) +    
C...I0/I3*c3*1/cosh(y*), I0 = y*max-y*min, and c1 + c2 + c3 = 1.    
        CALL PYKLIM(2)  
        IF(MINT(51).NE.0) GOTO 100  
        RYST=RLU(0) 
        MYST=1  
        IF(RYST.GT.COEF(ISUB,7)) MYST=2 
        IF(RYST.GT.COEF(ISUB,7)+COEF(ISUB,8)) MYST=3    
        CALL PYKMAP(2,MYST,RLU(0))  
    
C...2 -> 2 processes:   
C...Choose cos(theta-hat) (cth) according to h3(cth), where 
C...h3(cth) = c0 + I0/I1*c1*1/(A - cth) + I0/I2*c2*1/(A + cth) +    
C...I0/I3*c3*1/(A - cth)^2 + I0/I4*c4*1/(A + cth)^2,    
C...A = 1 + 2*(m3*m4/sh)^2 (= 1 for massless products), 
C...and c0 + c1 + c2 + c3 + c4 = 1. 
        CALL PYKLIM(3)  
        IF(MINT(51).NE.0) GOTO 100  
        IF(ISET(ISUB).EQ.2.OR.ISET(ISUB).EQ.4) THEN 
          RCTH=RLU(0)   
          MCTH=1    
          IF(RCTH.GT.COEF(ISUB,10)) MCTH=2  
          IF(RCTH.GT.COEF(ISUB,10)+COEF(ISUB,11)) MCTH=3    
          IF(RCTH.GT.COEF(ISUB,10)+COEF(ISUB,11)+COEF(ISUB,12)) MCTH=4  
          IF(RCTH.GT.COEF(ISUB,10)+COEF(ISUB,11)+COEF(ISUB,12)+ 
     &    COEF(ISUB,13)) MCTH=5 
          CALL PYKMAP(3,MCTH,RLU(0))    
        ENDIF   
    
C...Low-pT or multiple interactions (first semihard interaction).   
      ELSEIF(ISET(ISUB).EQ.5) THEN  
        CALL PYMULT(3)  
        ISUB=MINT(1)    
      ENDIF 
    
C...Choose azimuthal angle. 
      VINT(24)=PARU(2)*RLU(0)   
    
C...Check against user cuts on kinematics at parton level.  
      MINT(51)=0    
      IF(ISUB.LE.90.OR.ISUB.GT.100) CALL PYKLIM(0)  
      IF(MINT(51).NE.0) GOTO 100    
      IF(MINT(82).EQ.1.AND.MSTP(141).GE.1) THEN 
        MCUT=0  
        IF(MSUB(91)+MSUB(92)+MSUB(93)+MSUB(94)+MSUB(95).EQ.0)   
     &  CALL PYKCUT(MCUT)   
        IF(MCUT.NE.0) GOTO 100  
      ENDIF 
    
C...Calculate differential cross-section for different subprocesses.    
      CALL PYSIGH(NCHN,SIGS)    
    
C...Calculations for Monte Carlo estimate of all cross-sections.    
      IF(MINT(82).EQ.1.AND.ISUB.LE.90.OR.ISUB.GE.96) THEN   
        XSEC(ISUB,2)=XSEC(ISUB,2)+SIGS  
      ELSEIF(MINT(82).EQ.1) THEN    
        XSEC(ISUB,2)=XSEC(ISUB,2)+XSEC(ISUB,1)  
      ENDIF 
    
C...Multiple interactions: store results of cross-section calculation.  
      IF(MINT(43).EQ.4.AND.MSTP(82).GE.3) THEN  
        VINT(153)=SIGS  
        CALL PYMULT(4)  
      ENDIF 
    
C...Weighting using estimate of maximum of differential cross-section.  
      VIOL=SIGS/XSEC(ISUB,1)    
      IF(VIOL.LT.RLU(0)) GOTO 100   
    
C...Check for possible violation of estimated maximum of differential   
C...cross-section used in weighting.    
      IF(MSTP(123).LE.0) THEN   
        IF(VIOL.GT.1.) THEN 
          WRITE(MSTU(11),1000) VIOL,NGEN(0,3)+1 
          WRITE(MSTU(11),1100) ISUB,VINT(21),VINT(22),VINT(23),VINT(26) 
          STOP  
        ENDIF   
      ELSEIF(MSTP(123).EQ.1) THEN   
        IF(VIOL.GT.VINT(108)) THEN  
          VINT(108)=VIOL    
C          IF(VIOL.GT.1.) THEN   
C            WRITE(MSTU(11),1200) VIOL,NGEN(0,3)+1   
C            WRITE(MSTU(11),1100) ISUB,VINT(21),VINT(22),VINT(23),   
C     &      VINT(26)    
C          ENDIF 
        ENDIF   
      ELSEIF(VIOL.GT.VINT(108)) THEN    
        VINT(108)=VIOL  
        IF(VIOL.GT.1.) THEN 
          XDIF=XSEC(ISUB,1)*(VIOL-1.)   
          XSEC(ISUB,1)=XSEC(ISUB,1)+XDIF    
          IF(MSUB(ISUB).EQ.1.AND.(ISUB.LE.90.OR.ISUB.GT.96))    
     &    XSEC(0,1)=XSEC(0,1)+XDIF  
C          WRITE(MSTU(11),1200) VIOL,NGEN(0,3)+1 
C          WRITE(MSTU(11),1100) ISUB,VINT(21),VINT(22),VINT(23),VINT(26) 
C          IF(ISUB.LE.9) THEN    
C            WRITE(MSTU(11),1300) ISUB,XSEC(ISUB,1)  
C          ELSEIF(ISUB.LE.99) THEN   
C            WRITE(MSTU(11),1400) ISUB,XSEC(ISUB,1)  
C          ELSE  
C            WRITE(MSTU(11),1500) ISUB,XSEC(ISUB,1)  
C          ENDIF 
          VINT(108)=1.  
        ENDIF   
      ENDIF 
    
C...Multiple interactions: choose impact parameter. 
      VINT(148)=1.  
      IF(MINT(43).EQ.4.AND.(ISUB.LE.90.OR.ISUB.GE.96).AND.MSTP(82).GE.3)    
     &THEN  
        CALL PYMULT(5)  
        IF(VINT(150).LT.RLU(0)) GOTO 100    
      ENDIF 
      IF(MINT(82).EQ.1.AND.MSUB(95).EQ.1) THEN  
        IF(ISUB.LE.90.OR.ISUB.GE.95) NGEN(95,1)=NGEN(95,1)+1    
        IF(ISUB.LE.90.OR.ISUB.GE.96) NGEN(96,2)=NGEN(96,2)+1    
      ENDIF 
      IF(ISUB.LE.90.OR.ISUB.GE.96) MINT(31)=MINT(31)+1  
    
C...Choose flavour of reacting partons (and subprocess).    
      RSIGS=SIGS*RLU(0) 
      QT2=VINT(48)  
      RQQBAR=PARP(87)*(1.-(QT2/(QT2+(PARP(88)*PARP(82))**2))**2)    
      IF(ISUB.NE.95.AND.(ISUB.NE.96.OR.MSTP(82).LE.1.OR.    
     &RLU(0).GT.RQQBAR)) THEN   
        DO 190 ICHN=1,NCHN  
        KFL1=ISIG(ICHN,1)   
        KFL2=ISIG(ICHN,2)   
        MINT(2)=ISIG(ICHN,3)    
        RSIGS=RSIGS-SIGH(ICHN)  
        IF(RSIGS.LE.0.) GOTO 210    
  190   CONTINUE    
    
C...Multiple interactions: choose qqbar preferentially at small pT. 
      ELSEIF(ISUB.EQ.96) THEN   
        CALL PYSPLI(MINT(11),21,KFL1,KFLDUM)    
        CALL PYSPLI(MINT(12),21,KFL2,KFLDUM)    
        MINT(1)=11  
        MINT(2)=1   
        IF(KFL1.EQ.KFL2.AND.RLU(0).LT.0.5) MINT(2)=2    
    
C...Low-pT: choose string drawing configuration.    
      ELSE  
        KFL1=21 
        KFL2=21 
        RSIGS=6.*RLU(0) 
        MINT(2)=1   
        IF(RSIGS.GT.1.) MINT(2)=2   
        IF(RSIGS.GT.2.) MINT(2)=3   
      ENDIF 
    
C...Reassign QCD process. Partons before initial state radiation.   
  210 IF(MINT(2).GT.10) THEN    
        MINT(1)=MINT(2)/10  
        MINT(2)=MOD(MINT(2),10) 
      ENDIF 
      MINT(15)=KFL1 
      MINT(16)=KFL2 
      MINT(13)=MINT(15) 
      MINT(14)=MINT(16) 
      VINT(141)=VINT(41)    
      VINT(142)=VINT(42)    
    
C...Format statements for differential cross-section maximum violations.    
 1000 FORMAT(1X,'Error: maximum violated by',1P,E11.3,1X,   
     &'in event',1X,I7,'.'/1X,'Execution stopped!') 
 1100 FORMAT(1X,'ISUB = ',I3,'; Point of violation:'/1X,'tau=',1P, 
     &E11.3,', y* =',E11.3,', cthe = ',0P,F11.7,', tau'' =',1P,E11.3)   
clin 1200 FORMAT(1X,'Warning: maximum violated by',1P,E11.3,1X, 
c     &'in event',1X,I7) 
c 1300 FORMAT(1X,'XSEC(',I1,',1) increased to',1P,E11.3) 
c 1400 FORMAT(1X,'XSEC(',I2,',1) increased to',1P,E11.3) 
clin 1500 FORMAT(1X,'XSEC(',I3,',1) increased to',1P,E11.3) 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYSCAT 
    
C...Finds outgoing flavours and event type; sets up the kinematics  
C...and colour flow of the hard scattering. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      DIMENSION WDTP(0:40),WDTE(0:40,0:5),PMQ(2),Z(2),CTHE(2),PHI(2)    
    
C...Choice of subprocess, number of documentation lines.    
      ISUB=MINT(1)  
      IDOC=6+ISET(ISUB) 
      IF(ISUB.EQ.95) IDOC=8 
      MINT(3)=IDOC-6    
      IF(IDOC.GE.9) IDOC=IDOC+2 
      MINT(4)=IDOC  
      IPU1=MINT(84)+1   
      IPU2=MINT(84)+2   
      IPU3=MINT(84)+3   
      IPU4=MINT(84)+4   
      IPU5=MINT(84)+5   
      IPU6=MINT(84)+6   
    
C...Reset K, P and V vectors. Store incoming particles. 
      DO 100 JT=1,MSTP(126)+10  
      I=MINT(83)+JT 
      DO 100 J=1,5  
      K(I,J)=0  
      P(I,J)=0. 
  100 V(I,J)=0. 
      DO 110 JT=1,2 
      I=MINT(83)+JT 
      K(I,1)=21 
      K(I,2)=MINT(10+JT)    
      P(I,1)=0. 
      P(I,2)=0. 
      P(I,5)=VINT(2+JT) 
      P(I,3)=VINT(5)*(-1)**(JT+1)   
  110 P(I,4)=SQRT(P(I,3)**2+P(I,5)**2)  
      MINT(6)=2 
      KFRES=0   
    
C...Store incoming partons in their CM-frame.   
      SH=VINT(44)   
      SHR=SQRT(SH)  
      SHP=VINT(26)*VINT(2)  
      SHPR=SQRT(SHP)    
      SHUSER=SHR    
      IF(ISET(ISUB).GE.3) SHUSER=SHPR   
      DO 120 JT=1,2 
      I=MINT(84)+JT 
      K(I,1)=14 
      K(I,2)=MINT(14+JT)    
      K(I,3)=MINT(83)+2+JT  
  120 P(I,5)=ULMASS(K(I,2)) 
      IF(P(IPU1,5)+P(IPU2,5).GE.SHUSER) THEN    
        P(IPU1,5)=0.    
        P(IPU2,5)=0.    
      ENDIF 
      P(IPU1,4)=0.5*(SHUSER+(P(IPU1,5)**2-P(IPU2,5)**2)/SHUSER) 
      P(IPU1,3)=SQRT(MAX(0.,P(IPU1,4)**2-P(IPU1,5)**2)) 
      P(IPU2,4)=SHUSER-P(IPU1,4)    
      P(IPU2,3)=-P(IPU1,3)  
    
C...Copy incoming partons to documentation lines.   
      DO 130 JT=1,2 
      I1=MINT(83)+4+JT  
      I2=MINT(84)+JT    
      K(I1,1)=21    
      K(I1,2)=K(I2,2)   
      K(I1,3)=I1-2  
      DO 130 J=1,5  
  130 P(I1,J)=P(I2,J)   
    
C...Choose new quark flavour for relevant annihilation graphs.  
      KFLQ=0
      IF(ISUB.EQ.12.OR.ISUB.EQ.53) THEN 
        CALL PYWIDT(21,SHR,WDTP,WDTE)   
        RKFL=(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))*RLU(0) 
        DO 140 I=1,2*MSTP(1)    
        KFLQ=I  
        RKFL=RKFL-(WDTE(I,1)+WDTE(I,2)+WDTE(I,4))   
        IF(RKFL.LE.0.) GOTO 150 
  140   CONTINUE    
  150   CONTINUE    
      ENDIF 
    
C...Final state flavours and colour flow: default values.   
      JS=1  
      MINT(21)=MINT(15) 
      MINT(22)=MINT(16) 
      MINT(23)=0    
      MINT(24)=0    
      KCC=20    
      KCS=ISIGN(1,MINT(15)) 
    
      IF(ISUB.LE.10) THEN   
      IF(ISUB.EQ.1) THEN    
C...f + fb -> gamma*/Z0.    
        KFRES=23    
    
      ELSEIF(ISUB.EQ.2) THEN    
C...f + fb' -> W+/- .   
        KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))   
        KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))   
        KFRES=ISIGN(24,KCH1+KCH2)   
    
      ELSEIF(ISUB.EQ.3) THEN    
C...f + fb -> H0.   
        KFRES=25    
    
      ELSEIF(ISUB.EQ.4) THEN    
C...gamma + W+/- -> W+/-.   
    
      ELSEIF(ISUB.EQ.5) THEN    
C...Z0 + Z0 -> H0.  
        XH=SH/SHP   
        MINT(21)=MINT(15)   
        MINT(22)=MINT(16)   
        PMQ(1)=ULMASS(MINT(21)) 
        PMQ(2)=ULMASS(MINT(22)) 
  240   JT=INT(1.5+RLU(0))  
        ZMIN=2.*PMQ(JT)/SHPR    
        ZMAX=1.-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/(SHPR*(SHPR-PMQ(3-JT)))  
        ZMAX=MIN(1.-XH,ZMAX)    
        Z(JT)=ZMIN+(ZMAX-ZMIN)*RLU(0)   
        IF(-1.+(1.+XH)/(1.-Z(JT))-XH/(1.-Z(JT))**2.LT.  
     &  (1.-XH)**2/(4.*XH)*RLU(0)) GOTO 240 
        SQC1=1.-4.*PMQ(JT)**2/(Z(JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 240  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(23,1)**2-PMQ(JT)**2)/(Z(JT)*SHP) 
        CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1   
        CTHE(JT)=MIN(1.,MAX(-1.,CTHE(JT)))  
        Z(3-JT)=1.-XH/(1.-Z(JT))    
        SQC1=1.-4.*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 240  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(23,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP) 
        CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1 
        CTHE(3-JT)=MIN(1.,MAX(-1.,CTHE(3-JT)))  
        PHIR=PARU(2)*RLU(0) 
        CPHI=COS(PHIR)  
        ANG=CTHE(1)*CTHE(2)-SQRT(1.-CTHE(1)**2)*SQRT(1.-CTHE(2)**2)*CPHI    
        Z1=2.-Z(JT) 
        Z2=ANG*SQRT(Z(JT)**2-4.*PMQ(JT)**2/SHP) 
        Z3=1.-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP    
        Z(3-JT)=2./(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*    
     &  PMQ(3-JT)**2/SHP))  
        ZMIN=2.*PMQ(3-JT)/SHPR  
        ZMAX=1.-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))    
        ZMAX=MIN(1.-XH,ZMAX)    
        IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 240 
        KCC=22  
        KFRES=25    
    
      ELSEIF(ISUB.EQ.6) THEN    
C...Z0 + W+/- -> W+/-.  
    
      ELSEIF(ISUB.EQ.7) THEN    
C...W+ + W- -> Z0.  
    
      ELSEIF(ISUB.EQ.8) THEN    
C...W+ + W- -> H0.  
        XH=SH/SHP   
  250   DO 280 JT=1,2   
        I=MINT(14+JT)   
        IA=IABS(I)  
        IF(IA.LE.10) THEN   
          RVCKM=VINT(180+I)*RLU(0)  
          DO 270 J=1,MSTP(1)    
          IB=2*J-1+MOD(IA,2)    
          IPM=(5-ISIGN(1,I))/2  
          IDC=J+MDCY(IA,2)+2    
          IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 270  
          MINT(20+JT)=ISIGN(IB,I)   
          RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)   
          IF(RVCKM.LE.0.) GOTO 280  
  270     CONTINUE  
        ELSE    
          IB=2*((IA+1)/2)-1+MOD(IA,2)   
          MINT(20+JT)=ISIGN(IB,I)   
        ENDIF   
  280   PMQ(JT)=ULMASS(MINT(20+JT)) 
        JT=INT(1.5+RLU(0))  
        ZMIN=2.*PMQ(JT)/SHPR    
        ZMAX=1.-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/(SHPR*(SHPR-PMQ(3-JT)))  
        ZMAX=MIN(1.-XH,ZMAX)    
        Z(JT)=ZMIN+(ZMAX-ZMIN)*RLU(0)   
        IF(-1.+(1.+XH)/(1.-Z(JT))-XH/(1.-Z(JT))**2.LT.  
     &  (1.-XH)**2/(4.*XH)*RLU(0)) GOTO 250 
        SQC1=1.-4.*PMQ(JT)**2/(Z(JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 250  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(24,1)**2-PMQ(JT)**2)/(Z(JT)*SHP) 
        CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1   
        CTHE(JT)=MIN(1.,MAX(-1.,CTHE(JT)))  
        Z(3-JT)=1.-XH/(1.-Z(JT))    
        SQC1=1.-4.*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 250  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(24,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP) 
        CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1 
        CTHE(3-JT)=MIN(1.,MAX(-1.,CTHE(3-JT)))  
        PHIR=PARU(2)*RLU(0) 
        CPHI=COS(PHIR)  
        ANG=CTHE(1)*CTHE(2)-SQRT(1.-CTHE(1)**2)*SQRT(1.-CTHE(2)**2)*CPHI    
        Z1=2.-Z(JT) 
        Z2=ANG*SQRT(Z(JT)**2-4.*PMQ(JT)**2/SHP) 
        Z3=1.-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP    
        Z(3-JT)=2./(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*    
     &  PMQ(3-JT)**2/SHP))  
        ZMIN=2.*PMQ(3-JT)/SHPR  
        ZMAX=1.-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))    
        ZMAX=MIN(1.-XH,ZMAX)    
        IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 250 
        KCC=22  
        KFRES=25    
      ENDIF 
    
      ELSEIF(ISUB.LE.20) THEN   
      IF(ISUB.EQ.11) THEN   
C...f + f' -> f + f'; th = (p(f)-p(f))**2.  
        KCC=MINT(2) 
        IF(MINT(15)*MINT(16).LT.0) KCC=KCC+2    
    
      ELSEIF(ISUB.EQ.12) THEN   
C...f + fb -> f' + fb'; th = (p(f)-p(f'))**2.   
        MINT(21)=ISIGN(KFLQ,MINT(15))   
        MINT(22)=-MINT(21)  
        KCC=4   
    
      ELSEIF(ISUB.EQ.13) THEN   
C...f + fb -> g + g; th arbitrary.  
        MINT(21)=21 
        MINT(22)=21 
        KCC=MINT(2)+4   
    
      ELSEIF(ISUB.EQ.14) THEN   
C...f + fb -> g + gam; th arbitrary.    
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=21  
        MINT(23-JS)=22  
        KCC=17+JS   
    
      ELSEIF(ISUB.EQ.15) THEN   
C...f + fb -> g + Z0; th arbitrary. 
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=21  
        MINT(23-JS)=23  
        KCC=17+JS   
    
      ELSEIF(ISUB.EQ.16) THEN   
C...f + fb' -> g + W+/-; th = (p(f)-p(W-))**2 or (p(fb')-p(W+))**2. 
        KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))   
        KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))   
        IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2  
        MINT(20+JS)=21  
        MINT(23-JS)=ISIGN(24,KCH1+KCH2) 
        KCC=17+JS   
    
      ELSEIF(ISUB.EQ.17) THEN   
C...f + fb -> g + H0; th arbitrary. 
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=21  
        MINT(23-JS)=25  
        KCC=17+JS   
    
      ELSEIF(ISUB.EQ.18) THEN   
C...f + fb -> gamma + gamma; th arbitrary.  
        MINT(21)=22 
        MINT(22)=22 
    
      ELSEIF(ISUB.EQ.19) THEN   
C...f + fb -> gamma + Z0; th arbitrary. 
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=22  
        MINT(23-JS)=23  
    
      ELSEIF(ISUB.EQ.20) THEN   
C...f + fb' -> gamma + W+/-; th = (p(f)-p(W-))**2 or (p(fb')-p(W+))**2. 
        KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))   
        KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))   
        IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2  
        MINT(20+JS)=22  
        MINT(23-JS)=ISIGN(24,KCH1+KCH2) 
      ENDIF 
    
      ELSEIF(ISUB.LE.30) THEN   
      IF(ISUB.EQ.21) THEN   
C...f + fb -> gamma + H0; th arbitrary. 
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=22  
        MINT(23-JS)=25  
    
      ELSEIF(ISUB.EQ.22) THEN   
C...f + fb -> Z0 + Z0; th arbitrary.    
        MINT(21)=23 
        MINT(22)=23 
    
      ELSEIF(ISUB.EQ.23) THEN   
C...f + fb' -> Z0 + W+/-; th = (p(f)-p(W-))**2 or (p(fb')-p(W+))**2.    
        KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))   
        KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))   
        IF(MINT(15)*(KCH1+KCH2).LT.0) JS=2  
        MINT(20+JS)=23  
        MINT(23-JS)=ISIGN(24,KCH1+KCH2) 
    
      ELSEIF(ISUB.EQ.24) THEN   
C...f + fb -> Z0 + H0; th arbitrary.    
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=23  
        MINT(23-JS)=25  
    
      ELSEIF(ISUB.EQ.25) THEN   
C...f + fb -> W+ + W-; th = (p(f)-p(W-))**2.    
        MINT(21)=-ISIGN(24,MINT(15))    
        MINT(22)=-MINT(21)  
    
      ELSEIF(ISUB.EQ.26) THEN   
C...f + fb' -> W+/- + H0; th = (p(f)-p(W-))**2 or (p(fb')-p(W+))**2.    
        KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))   
        KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))   
        IF(MINT(15)*(KCH1+KCH2).GT.0) JS=2  
        MINT(20+JS)=ISIGN(24,KCH1+KCH2) 
        MINT(23-JS)=25  
    
      ELSEIF(ISUB.EQ.27) THEN   
C...f + fb -> H0 + H0.  
    
      ELSEIF(ISUB.EQ.28) THEN   
C...f + g -> f + g; th = (p(f)-p(f))**2.    
        KCC=MINT(2)+6   
        IF(MINT(15).EQ.21) KCC=KCC+2    
        IF(MINT(15).NE.21) KCS=ISIGN(1,MINT(15))    
        IF(MINT(16).NE.21) KCS=ISIGN(1,MINT(16))    
    
      ELSEIF(ISUB.EQ.29) THEN   
C...f + g -> f + gamma; th = (p(f)-p(f))**2.    
        IF(MINT(15).EQ.21) JS=2 
        MINT(23-JS)=22  
        KCC=15+JS   
        KCS=ISIGN(1,MINT(14+JS))    
    
      ELSEIF(ISUB.EQ.30) THEN   
C...f + g -> f + Z0; th = (p(f)-p(f))**2.   
        IF(MINT(15).EQ.21) JS=2 
        MINT(23-JS)=23  
        KCC=15+JS   
        KCS=ISIGN(1,MINT(14+JS))    
      ENDIF 
    
      ELSEIF(ISUB.LE.40) THEN   
      IF(ISUB.EQ.31) THEN   
C...f + g -> f' + W+/-; th = (p(f)-p(f'))**2; choose flavour f'.    
        IF(MINT(15).EQ.21) JS=2 
        I=MINT(14+JS)   
        IA=IABS(I)  
        MINT(23-JS)=ISIGN(24,KCHG(IA,1)*I)  
        RVCKM=VINT(180+I)*RLU(0)    
        DO 220 J=1,MSTP(1)  
        IB=2*J-1+MOD(IA,2)  
        IPM=(5-ISIGN(1,I))/2    
        IDC=J+MDCY(IA,2)+2  
        IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 220    
        MINT(20+JS)=ISIGN(IB,I) 
        RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2) 
        IF(RVCKM.LE.0.) GOTO 230    
  220   CONTINUE    
  230   KCC=15+JS   
        KCS=ISIGN(1,MINT(14+JS))    
    
      ELSEIF(ISUB.EQ.32) THEN   
C...f + g -> f + H0; th = (p(f)-p(f))**2.   
        IF(MINT(15).EQ.21) JS=2 
        MINT(23-JS)=25  
        KCC=15+JS   
        KCS=ISIGN(1,MINT(14+JS))    
    
      ELSEIF(ISUB.EQ.33) THEN   
C...f + gamma -> f + g. 
    
      ELSEIF(ISUB.EQ.34) THEN   
C...f + gamma -> f + gamma. 
    
      ELSEIF(ISUB.EQ.35) THEN   
C...f + gamma -> f + Z0.    
    
      ELSEIF(ISUB.EQ.36) THEN   
C...f + gamma -> f' + W+/-. 
    
      ELSEIF(ISUB.EQ.37) THEN   
C...f + gamma -> f + H0.    
    
      ELSEIF(ISUB.EQ.38) THEN   
C...f + Z0 -> f + g.    
    
      ELSEIF(ISUB.EQ.39) THEN   
C...f + Z0 -> f + gamma.    
    
      ELSEIF(ISUB.EQ.40) THEN   
C...f + Z0 -> f + Z0.   
      ENDIF 
    
      ELSEIF(ISUB.LE.50) THEN   
      IF(ISUB.EQ.41) THEN   
C...f + Z0 -> f' + W+/-.    
    
      ELSEIF(ISUB.EQ.42) THEN   
C...f + Z0 -> f + H0.   
    
      ELSEIF(ISUB.EQ.43) THEN   
C...f + W+/- -> f' + g. 
    
      ELSEIF(ISUB.EQ.44) THEN   
C...f + W+/- -> f' + gamma. 
    
      ELSEIF(ISUB.EQ.45) THEN   
C...f + W+/- -> f' + Z0.    
    
      ELSEIF(ISUB.EQ.46) THEN   
C...f + W+/- -> f' + W+/-.  
    
      ELSEIF(ISUB.EQ.47) THEN   
C...f + W+/- -> f' + H0.    
    
      ELSEIF(ISUB.EQ.48) THEN   
C...f + H0 -> f + g.    
    
      ELSEIF(ISUB.EQ.49) THEN   
C...f + H0 -> f + gamma.    
    
      ELSEIF(ISUB.EQ.50) THEN   
C...f + H0 -> f + Z0.   
      ENDIF 
    
      ELSEIF(ISUB.LE.60) THEN   
      IF(ISUB.EQ.51) THEN   
C...f + H0 -> f' + W+/-.    
    
      ELSEIF(ISUB.EQ.52) THEN   
C...f + H0 -> f + H0.   
    
      ELSEIF(ISUB.EQ.53) THEN   
C...g + g -> f + fb; th arbitrary.  
        KCS=(-1)**INT(1.5+RLU(0))   
        MINT(21)=ISIGN(KFLQ,KCS)    
        MINT(22)=-MINT(21)  
        KCC=MINT(2)+10  
    
      ELSEIF(ISUB.EQ.54) THEN   
C...g + gamma -> f + fb.    
    
      ELSEIF(ISUB.EQ.55) THEN   
C...g + Z0 -> f + fb.   
    
      ELSEIF(ISUB.EQ.56) THEN   
C...g + W+/- -> f + fb'.    
    
      ELSEIF(ISUB.EQ.57) THEN   
C...g + H0 -> f + fb.   
    
      ELSEIF(ISUB.EQ.58) THEN   
C...gamma + gamma -> f + fb.    
    
      ELSEIF(ISUB.EQ.59) THEN   
C...gamma + Z0 -> f + fb.   
    
      ELSEIF(ISUB.EQ.60) THEN   
C...gamma + W+/- -> f + fb'.    
      ENDIF 
    
      ELSEIF(ISUB.LE.70) THEN   
      IF(ISUB.EQ.61) THEN   
C...gamma + H0 -> f + fb.   
    
      ELSEIF(ISUB.EQ.62) THEN   
C...Z0 + Z0 -> f + fb.  
    
      ELSEIF(ISUB.EQ.63) THEN   
C...Z0 + W+/- -> f + fb'.   
    
      ELSEIF(ISUB.EQ.64) THEN   
C...Z0 + H0 -> f + fb.  
    
      ELSEIF(ISUB.EQ.65) THEN   
C...W+ + W- -> f + fb.  
    
      ELSEIF(ISUB.EQ.66) THEN   
C...W+/- + H0 -> f + fb'.   
    
      ELSEIF(ISUB.EQ.67) THEN   
C...H0 + H0 -> f + fb.  
    
      ELSEIF(ISUB.EQ.68) THEN   
C...g + g -> g + g; th arbitrary.   
        KCC=MINT(2)+12  
        KCS=(-1)**INT(1.5+RLU(0))   
    
      ELSEIF(ISUB.EQ.69) THEN   
C...gamma + gamma -> W+ + W-.   
    
      ELSEIF(ISUB.EQ.70) THEN   
C...gamma + W+/- -> gamma + W+/-    
      ENDIF 
    
      ELSEIF(ISUB.LE.80) THEN   
      IF(ISUB.EQ.71.OR.ISUB.EQ.72) THEN 
C...Z0 + Z0 -> Z0 + Z0; Z0 + Z0 -> W+ + W-. 
        XH=SH/SHP   
        MINT(21)=MINT(15)   
        MINT(22)=MINT(16)   
        PMQ(1)=ULMASS(MINT(21)) 
        PMQ(2)=ULMASS(MINT(22)) 
  290   JT=INT(1.5+RLU(0))  
        ZMIN=2.*PMQ(JT)/SHPR    
        ZMAX=1.-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/(SHPR*(SHPR-PMQ(3-JT)))  
        ZMAX=MIN(1.-XH,ZMAX)    
        Z(JT)=ZMIN+(ZMAX-ZMIN)*RLU(0)   
        IF(-1.+(1.+XH)/(1.-Z(JT))-XH/(1.-Z(JT))**2.LT.  
     &  (1.-XH)**2/(4.*XH)*RLU(0)) GOTO 290 
        SQC1=1.-4.*PMQ(JT)**2/(Z(JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 290  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(23,1)**2-PMQ(JT)**2)/(Z(JT)*SHP) 
        CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1   
        CTHE(JT)=MIN(1.,MAX(-1.,CTHE(JT)))  
        Z(3-JT)=1.-XH/(1.-Z(JT))    
        SQC1=1.-4.*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 290  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(23,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP) 
        CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1 
        CTHE(3-JT)=MIN(1.,MAX(-1.,CTHE(3-JT)))  
        PHIR=PARU(2)*RLU(0) 
        CPHI=COS(PHIR)  
        ANG=CTHE(1)*CTHE(2)-SQRT(1.-CTHE(1)**2)*SQRT(1.-CTHE(2)**2)*CPHI    
        Z1=2.-Z(JT) 
        Z2=ANG*SQRT(Z(JT)**2-4.*PMQ(JT)**2/SHP) 
        Z3=1.-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP    
        Z(3-JT)=2./(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*    
     &  PMQ(3-JT)**2/SHP))  
        ZMIN=2.*PMQ(3-JT)/SHPR  
        ZMAX=1.-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))    
        ZMAX=MIN(1.-XH,ZMAX)    
        IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 290 
        KCC=22  
    
      ELSEIF(ISUB.EQ.73) THEN   
C...Z0 + W+/- -> Z0 + W+/-. 
        XH=SH/SHP   
  300   JT=INT(1.5+RLU(0))  
        I=MINT(14+JT)   
        IA=IABS(I)  
        IF(IA.LE.10) THEN   
          RVCKM=VINT(180+I)*RLU(0)  
          DO 320 J=1,MSTP(1)    
          IB=2*J-1+MOD(IA,2)    
          IPM=(5-ISIGN(1,I))/2  
          IDC=J+MDCY(IA,2)+2    
          IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 320  
          MINT(20+JT)=ISIGN(IB,I)   
          RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)   
          IF(RVCKM.LE.0.) GOTO 330  
  320     CONTINUE  
        ELSE    
          IB=2*((IA+1)/2)-1+MOD(IA,2)   
          MINT(20+JT)=ISIGN(IB,I)   
        ENDIF   
  330   PMQ(JT)=ULMASS(MINT(20+JT)) 
        MINT(23-JT)=MINT(17-JT) 
        PMQ(3-JT)=ULMASS(MINT(23-JT))   
        JT=INT(1.5+RLU(0))  
        ZMIN=2.*PMQ(JT)/SHPR    
        ZMAX=1.-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/(SHPR*(SHPR-PMQ(3-JT)))  
        ZMAX=MIN(1.-XH,ZMAX)    
        Z(JT)=ZMIN+(ZMAX-ZMIN)*RLU(0)   
        IF(-1.+(1.+XH)/(1.-Z(JT))-XH/(1.-Z(JT))**2.LT.  
     &  (1.-XH)**2/(4.*XH)*RLU(0)) GOTO 300 
        SQC1=1.-4.*PMQ(JT)**2/(Z(JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 300  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(23,1)**2-PMQ(JT)**2)/(Z(JT)*SHP) 
        CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1   
        CTHE(JT)=MIN(1.,MAX(-1.,CTHE(JT)))  
        Z(3-JT)=1.-XH/(1.-Z(JT))    
        SQC1=1.-4.*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 300  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(23,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP) 
        CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1 
        CTHE(3-JT)=MIN(1.,MAX(-1.,CTHE(3-JT)))  
        PHIR=PARU(2)*RLU(0) 
        CPHI=COS(PHIR)  
        ANG=CTHE(1)*CTHE(2)-SQRT(1.-CTHE(1)**2)*SQRT(1.-CTHE(2)**2)*CPHI    
        Z1=2.-Z(JT) 
        Z2=ANG*SQRT(Z(JT)**2-4.*PMQ(JT)**2/SHP) 
        Z3=1.-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP    
        Z(3-JT)=2./(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*    
     &  PMQ(3-JT)**2/SHP))  
        ZMIN=2.*PMQ(3-JT)/SHPR  
        ZMAX=1.-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))    
        ZMAX=MIN(1.-XH,ZMAX)    
        IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 300 
        KCC=22  
    
      ELSEIF(ISUB.EQ.74) THEN   
C...Z0 + H0 -> Z0 + H0. 
    
      ELSEIF(ISUB.EQ.75) THEN   
C...W+ + W- -> gamma + gamma.   
    
      ELSEIF(ISUB.EQ.76.OR.ISUB.EQ.77) THEN 
C...W+ + W- -> Z0 + Z0; W+ + W- -> W+ + W-. 
        XH=SH/SHP   
  340   DO 370 JT=1,2   
        I=MINT(14+JT)   
        IA=IABS(I)  
        IF(IA.LE.10) THEN   
          RVCKM=VINT(180+I)*RLU(0)  
          DO 360 J=1,MSTP(1)    
          IB=2*J-1+MOD(IA,2)    
          IPM=(5-ISIGN(1,I))/2  
          IDC=J+MDCY(IA,2)+2    
          IF(MDME(IDC,1).NE.1.AND.MDME(IDC,1).NE.IPM) GOTO 360  
          MINT(20+JT)=ISIGN(IB,I)   
          RVCKM=RVCKM-VCKM((IA+1)/2,(IB+1)/2)   
          IF(RVCKM.LE.0.) GOTO 370  
  360     CONTINUE  
        ELSE    
          IB=2*((IA+1)/2)-1+MOD(IA,2)   
          MINT(20+JT)=ISIGN(IB,I)   
        ENDIF   
  370   PMQ(JT)=ULMASS(MINT(20+JT)) 
        JT=INT(1.5+RLU(0))  
        ZMIN=2.*PMQ(JT)/SHPR    
        ZMAX=1.-PMQ(3-JT)/SHPR-(SH-PMQ(JT)**2)/(SHPR*(SHPR-PMQ(3-JT)))  
        ZMAX=MIN(1.-XH,ZMAX)    
        Z(JT)=ZMIN+(ZMAX-ZMIN)*RLU(0)   
        IF(-1.+(1.+XH)/(1.-Z(JT))-XH/(1.-Z(JT))**2.LT.  
     &  (1.-XH)**2/(4.*XH)*RLU(0)) GOTO 340 
        SQC1=1.-4.*PMQ(JT)**2/(Z(JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 340  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(24,1)**2-PMQ(JT)**2)/(Z(JT)*SHP) 
        CTHE(JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1   
        CTHE(JT)=MIN(1.,MAX(-1.,CTHE(JT)))  
        Z(3-JT)=1.-XH/(1.-Z(JT))    
        SQC1=1.-4.*PMQ(3-JT)**2/(Z(3-JT)**2*SHP)    
        IF(SQC1.LT.1.E-8) GOTO 340  
        C1=SQRT(SQC1)   
        C2=1.+2.*(PMAS(24,1)**2-PMQ(3-JT)**2)/(Z(3-JT)*SHP) 
        CTHE(3-JT)=(C2-(C2**2-C1**2)/(C2+(2.*RLU(0)-1.)*C1))/C1 
        CTHE(3-JT)=MIN(1.,MAX(-1.,CTHE(3-JT)))  
        PHIR=PARU(2)*RLU(0) 
        CPHI=COS(PHIR)  
        ANG=CTHE(1)*CTHE(2)-SQRT(1.-CTHE(1)**2)*SQRT(1.-CTHE(2)**2)*CPHI    
        Z1=2.-Z(JT) 
        Z2=ANG*SQRT(Z(JT)**2-4.*PMQ(JT)**2/SHP) 
        Z3=1.-Z(JT)-XH+(PMQ(1)**2+PMQ(2)**2)/SHP    
        Z(3-JT)=2./(Z1**2-Z2**2)*(Z1*Z3+Z2*SQRT(Z3**2-(Z1**2-Z2**2)*    
     &  PMQ(3-JT)**2/SHP))  
        ZMIN=2.*PMQ(3-JT)/SHPR  
        ZMAX=1.-PMQ(JT)/SHPR-(SH-PMQ(3-JT)**2)/(SHPR*(SHPR-PMQ(JT)))    
        ZMAX=MIN(1.-XH,ZMAX)    
        IF(Z(3-JT).LT.ZMIN.OR.Z(3-JT).GT.ZMAX) GOTO 340 
        KCC=22  
    
      ELSEIF(ISUB.EQ.78) THEN   
C...W+/- + H0 -> W+/- + H0. 
    
      ELSEIF(ISUB.EQ.79) THEN   
C...H0 + H0 -> H0 + H0. 
      ENDIF 
    
      ELSEIF(ISUB.LE.90) THEN   
      IF(ISUB.EQ.81) THEN   
C...q + qb -> Q' + Qb'; th = (p(q)-p(q'))**2.   
        MINT(21)=ISIGN(MINT(46),MINT(15))   
        MINT(22)=-MINT(21)  
        KCC=4   
    
      ELSEIF(ISUB.EQ.82) THEN   
C...g + g -> Q + Qb; th arbitrary.  
        KCS=(-1)**INT(1.5+RLU(0))   
        MINT(21)=ISIGN(MINT(46),KCS)    
        MINT(22)=-MINT(21)  
        KCC=MINT(2)+10  
      ENDIF 
    
      ELSEIF(ISUB.LE.100) THEN  
      IF(ISUB.EQ.95) THEN   
C...Low-pT ( = energyless g + g -> g + g).  
        KCC=MINT(2)+12  
        KCS=(-1)**INT(1.5+RLU(0))   
    
      ELSEIF(ISUB.EQ.96) THEN   
C...Multiple interactions (should be reassigned to QCD process).    
      ENDIF 
    
      ELSEIF(ISUB.LE.110) THEN  
      IF(ISUB.EQ.101) THEN  
C...g + g -> gamma*/Z0. 
        KCC=21  
        KFRES=22    
    
      ELSEIF(ISUB.EQ.102) THEN  
C...g + g -> H0.    
        KCC=21  
        KFRES=25    
      ENDIF 
    
      ELSEIF(ISUB.LE.120) THEN  
      IF(ISUB.EQ.111) THEN  
C...f + fb -> g + H0; th arbitrary. 
        IF(RLU(0).GT.0.5) JS=2  
        MINT(20+JS)=21  
        MINT(23-JS)=25  
        KCC=17+JS   
    
      ELSEIF(ISUB.EQ.112) THEN  
C...f + g -> f + H0; th = (p(f) - p(f))**2. 
        IF(MINT(15).EQ.21) JS=2 
        MINT(23-JS)=25  
        KCC=15+JS   
        KCS=ISIGN(1,MINT(14+JS))    
    
      ELSEIF(ISUB.EQ.113) THEN  
C...g + g -> g + H0; th arbitrary.  
        IF(RLU(0).GT.0.5) JS=2  
        MINT(23-JS)=25  
        KCC=22+JS   
        KCS=(-1)**INT(1.5+RLU(0))   
    
      ELSEIF(ISUB.EQ.114) THEN  
C...g + g -> gamma + gamma; th arbitrary.   
        IF(RLU(0).GT.0.5) JS=2  
        MINT(21)=22 
        MINT(22)=22 
        KCC=21  
    
      ELSEIF(ISUB.EQ.115) THEN  
C...g + g -> gamma + Z0.    
    
      ELSEIF(ISUB.EQ.116) THEN  
C...g + g -> Z0 + Z0.   
    
      ELSEIF(ISUB.EQ.117) THEN  
C...g + g -> W+ + W-.   
      ENDIF 
    
      ELSEIF(ISUB.LE.140) THEN  
      IF(ISUB.EQ.121) THEN  
C...g + g -> f + fb + H0.   
      ENDIF 
    
      ELSEIF(ISUB.LE.160) THEN  
      IF(ISUB.EQ.141) THEN  
C...f + fb -> gamma*/Z0/Z'0.    
        KFRES=32    
    
      ELSEIF(ISUB.EQ.142) THEN  
C...f + fb' -> H+/-.    
        KCH1=KCHG(IABS(MINT(15)),1)*ISIGN(1,MINT(15))   
        KCH2=KCHG(IABS(MINT(16)),1)*ISIGN(1,MINT(16))   
        KFRES=ISIGN(37,KCH1+KCH2)   
    
      ELSEIF(ISUB.EQ.143) THEN  
C...f + fb' -> R.   
        KFRES=ISIGN(40,MINT(15)+MINT(16))   
      ENDIF 
    
      ELSE  
      IF(ISUB.EQ.161) THEN  
C...g + f -> H+/- + f'; th = (p(f)-p(f))**2.    
        IF(MINT(16).EQ.21) JS=2 
        IA=IABS(MINT(17-JS))    
        MINT(20+JS)=ISIGN(37,KCHG(IA,1)*MINT(17-JS))    
        JA=IA+MOD(IA,2)-MOD(IA+1,2) 
        MINT(23-JS)=ISIGN(JA,MINT(17-JS))   
        KCC=18-JS   
        IF(MINT(15).NE.21) KCS=ISIGN(1,MINT(15))    
        IF(MINT(16).NE.21) KCS=ISIGN(1,MINT(16))    
      ENDIF 
      ENDIF 
    
      IF(IDOC.EQ.7) THEN    
C...Resonance not decaying: store colour connection indices.    
        I=MINT(83)+7    
        K(IPU3,1)=1 
        K(IPU3,2)=KFRES 
        K(IPU3,3)=I 
        P(IPU3,4)=SHUSER    
        P(IPU3,5)=SHUSER    
        K(IPU1,4)=IPU2  
        K(IPU1,5)=IPU2  
        K(IPU2,4)=IPU1  
        K(IPU2,5)=IPU1  
        K(I,1)=21   
        K(I,2)=KFRES    
        P(I,4)=SHUSER   
        P(I,5)=SHUSER   
        N=IPU3  
        MINT(21)=KFRES  
        MINT(22)=0  
    
      ELSEIF(IDOC.EQ.8) THEN    
C...2 -> 2 processes: store outgoing partons in their CM-frame. 
        DO 390 JT=1,2   
        I=MINT(84)+2+JT 
        K(I,1)=1    
        IF(IABS(MINT(20+JT)).LE.10.OR.MINT(20+JT).EQ.21) K(I,1)=3   
        K(I,2)=MINT(20+JT)  
        K(I,3)=MINT(83)+IDOC+JT-2   
        IF(IABS(K(I,2)).LE.10.OR.K(I,2).EQ.21) THEN 
          P(I,5)=ULMASS(K(I,2)) 
        ELSE    
          P(I,5)=SQRT(VINT(63+MOD(JS+JT,2)))    
        ENDIF   
  390   CONTINUE    
        IF(P(IPU3,5)+P(IPU4,5).GE.SHR) THEN 
          KFA1=IABS(MINT(21))   
          KFA2=IABS(MINT(22))   
          IF((KFA1.GT.3.AND.KFA1.NE.21).OR.(KFA2.GT.3.AND.KFA2.NE.21))  
     &    THEN  
            MINT(51)=1  
            RETURN  
          ENDIF 
          P(IPU3,5)=0.  
          P(IPU4,5)=0.  
        ENDIF   
        P(IPU3,4)=0.5*(SHR+(P(IPU3,5)**2-P(IPU4,5)**2)/SHR) 
        P(IPU3,3)=SQRT(MAX(0.,P(IPU3,4)**2-P(IPU3,5)**2))   
        P(IPU4,4)=SHR-P(IPU3,4) 
        P(IPU4,3)=-P(IPU3,3)    
        N=IPU4  
        MINT(7)=MINT(83)+7  
        MINT(8)=MINT(83)+8  
    
C...Rotate outgoing partons using cos(theta)=(th-uh)/lam(sh,sqm3,sqm4). 
        CALL LUDBRB(IPU3,IPU4,ACOS(VINT(23)),VINT(24),0D0,0D0,0D0)  
    
      ELSEIF(IDOC.EQ.9) THEN    
C'''2 -> 3 processes:   
    
      ELSEIF(IDOC.EQ.11) THEN   
C...Z0 + Z0 -> H0, W+ + W- -> H0: store Higgs and outgoing partons. 
        PHI(1)=PARU(2)*RLU(0)   
        PHI(2)=PHI(1)-PHIR  
        DO 400 JT=1,2   
        I=MINT(84)+2+JT 
        K(I,1)=1    
        IF(IABS(MINT(20+JT)).LE.10.OR.MINT(20+JT).EQ.21) K(I,1)=3   
        K(I,2)=MINT(20+JT)  
        K(I,3)=MINT(83)+IDOC+JT-2   
        P(I,5)=ULMASS(K(I,2))   
        IF(0.5*SHPR*Z(JT).LE.P(I,5)) P(I,5)=0.  
        PABS=SQRT(MAX(0.,(0.5*SHPR*Z(JT))**2-P(I,5)**2))    
        PTABS=PABS*SQRT(MAX(0.,1.-CTHE(JT)**2)) 
        P(I,1)=PTABS*COS(PHI(JT))   
        P(I,2)=PTABS*SIN(PHI(JT))   
        P(I,3)=PABS*CTHE(JT)*(-1)**(JT+1)   
        P(I,4)=0.5*SHPR*Z(JT)   
        IZW=MINT(83)+6+JT   
        K(IZW,1)=21 
        K(IZW,2)=23 
        IF(ISUB.EQ.8) K(IZW,2)=ISIGN(24,LUCHGE(MINT(14+JT)))    
        K(IZW,3)=IZW-2  
        P(IZW,1)=-P(I,1)    
        P(IZW,2)=-P(I,2)    
        P(IZW,3)=(0.5*SHPR-PABS*CTHE(JT))*(-1)**(JT+1)  
        P(IZW,4)=0.5*SHPR*(1.-Z(JT))    
  400   P(IZW,5)=-SQRT(MAX(0.,P(IZW,3)**2+PTABS**2-P(IZW,4)**2))    
        I=MINT(83)+9    
        K(IPU5,1)=1 
        K(IPU5,2)=KFRES 
        K(IPU5,3)=I 
        P(IPU5,5)=SHR   
        P(IPU5,1)=-P(IPU3,1)-P(IPU4,1)  
        P(IPU5,2)=-P(IPU3,2)-P(IPU4,2)  
        P(IPU5,3)=-P(IPU3,3)-P(IPU4,3)  
        P(IPU5,4)=SHPR-P(IPU3,4)-P(IPU4,4)  
        K(I,1)=21   
        K(I,2)=KFRES    
        DO 410 J=1,5    
  410   P(I,J)=P(IPU5,J)    
        N=IPU5  
        MINT(23)=KFRES  
    
      ELSEIF(IDOC.EQ.12) THEN   
C...Z0 and W+/- scattering: store bosons and outgoing partons.  
        PHI(1)=PARU(2)*RLU(0)   
        PHI(2)=PHI(1)-PHIR  
        DO 420 JT=1,2   
        I=MINT(84)+2+JT 
        K(I,1)=1    
        IF(IABS(MINT(20+JT)).LE.10.OR.MINT(20+JT).EQ.21) K(I,1)=3   
        K(I,2)=MINT(20+JT)  
        K(I,3)=MINT(83)+IDOC+JT-2   
        P(I,5)=ULMASS(K(I,2))   
        IF(0.5*SHPR*Z(JT).LE.P(I,5)) P(I,5)=0.  
        PABS=SQRT(MAX(0.,(0.5*SHPR*Z(JT))**2-P(I,5)**2))    
        PTABS=PABS*SQRT(MAX(0.,1.-CTHE(JT)**2)) 
        P(I,1)=PTABS*COS(PHI(JT))   
        P(I,2)=PTABS*SIN(PHI(JT))   
        P(I,3)=PABS*CTHE(JT)*(-1)**(JT+1)   
        P(I,4)=0.5*SHPR*Z(JT)   
        IZW=MINT(83)+6+JT   
        K(IZW,1)=21 
        IF(MINT(14+JT).EQ.MINT(20+JT)) THEN 
          K(IZW,2)=23   
        ELSE    
          K(IZW,2)=ISIGN(24,LUCHGE(MINT(14+JT))-LUCHGE(MINT(20+JT)))    
        ENDIF   
        K(IZW,3)=IZW-2  
        P(IZW,1)=-P(I,1)    
        P(IZW,2)=-P(I,2)    
        P(IZW,3)=(0.5*SHPR-PABS*CTHE(JT))*(-1)**(JT+1)  
        P(IZW,4)=0.5*SHPR*(1.-Z(JT))    
        P(IZW,5)=-SQRT(MAX(0.,P(IZW,3)**2+PTABS**2-P(IZW,4)**2))    
        IPU=MINT(84)+4+JT   
        K(IPU,1)=3  
        K(IPU,2)=KFPR(ISUB,JT)  
        K(IPU,3)=MINT(83)+8+JT  
        IF(IABS(K(IPU,2)).LE.10.OR.K(IPU,2).EQ.21) THEN 
          P(IPU,5)=ULMASS(K(IPU,2)) 
        ELSE    
          P(IPU,5)=SQRT(VINT(63+MOD(JS+JT,2)))  
        ENDIF   
        MINT(22+JT)=K(IZW,2)    
  420   CONTINUE    
        IF(ISUB.EQ.72) K(MINT(84)+4+INT(1.5+RLU(0)),2)=-24  
C...Find rotation and boost for hard scattering subsystem.  
        I1=MINT(83)+7   
        I2=MINT(83)+8   
        BEXCM=(P(I1,1)+P(I2,1))/(P(I1,4)+P(I2,4))   
        BEYCM=(P(I1,2)+P(I2,2))/(P(I1,4)+P(I2,4))   
        BEZCM=(P(I1,3)+P(I2,3))/(P(I1,4)+P(I2,4))   
        GAMCM=(P(I1,4)+P(I2,4))/SHR 
        BEPCM=BEXCM*P(I1,1)+BEYCM*P(I1,2)+BEZCM*P(I1,3) 
        PX=P(I1,1)+GAMCM*(GAMCM/(1.+GAMCM)*BEPCM-P(I1,4))*BEXCM 
        PY=P(I1,2)+GAMCM*(GAMCM/(1.+GAMCM)*BEPCM-P(I1,4))*BEYCM 
        PZ=P(I1,3)+GAMCM*(GAMCM/(1.+GAMCM)*BEPCM-P(I1,4))*BEZCM 
        THECM=ULANGL(PZ,SQRT(PX**2+PY**2))  
        PHICM=ULANGL(PX,PY) 
C...Store hard scattering subsystem. Rotate and boost it.   
        SQLAM=(SH-P(IPU5,5)**2-P(IPU6,5)**2)**2-4.*P(IPU5,5)**2*    
     &  P(IPU6,5)**2    
        PABS=SQRT(MAX(0.,SQLAM/(4.*SH)))    
        CTHWZ=VINT(23)  
        STHWZ=SQRT(MAX(0.,1.-CTHWZ**2)) 
        PHIWZ=VINT(24)-PHICM    
        P(IPU5,1)=PABS*STHWZ*COS(PHIWZ) 
        P(IPU5,2)=PABS*STHWZ*SIN(PHIWZ) 
        P(IPU5,3)=PABS*CTHWZ    
        P(IPU5,4)=SQRT(PABS**2+P(IPU5,5)**2)    
        P(IPU6,1)=-P(IPU5,1)    
        P(IPU6,2)=-P(IPU5,2)    
        P(IPU6,3)=-P(IPU5,3)    
        P(IPU6,4)=SQRT(PABS**2+P(IPU6,5)**2)    
        CALL LUDBRB(IPU5,IPU6,THECM,PHICM,DBLE(BEXCM),DBLE(BEYCM),  
     &  DBLE(BEZCM))    
        DO 430 JT=1,2   
        I1=MINT(83)+8+JT    
        I2=MINT(84)+4+JT    
        K(I1,1)=21  
        K(I1,2)=K(I2,2) 
        DO 430 J=1,5    
  430   P(I1,J)=P(I2,J) 
        N=IPU6  
        MINT(7)=MINT(83)+9  
        MINT(8)=MINT(83)+10 
      ENDIF 
    
      IF(IDOC.GE.8) THEN    
C...Store colour connection indices.    
        DO 440 J=1,2    
        JC=J    
        IF(KCS.EQ.-1) JC=3-J    
        IF(ICOL(KCC,1,JC).NE.0.AND.K(IPU1,1).EQ.14) K(IPU1,J+3)=    
     &  K(IPU1,J+3)+MINT(84)+ICOL(KCC,1,JC) 
        IF(ICOL(KCC,2,JC).NE.0.AND.K(IPU2,1).EQ.14) K(IPU2,J+3)=    
     &  K(IPU2,J+3)+MINT(84)+ICOL(KCC,2,JC) 
        IF(ICOL(KCC,3,JC).NE.0.AND.K(IPU3,1).EQ.3) K(IPU3,J+3)= 
     &  MSTU(5)*(MINT(84)+ICOL(KCC,3,JC))   
  440   IF(ICOL(KCC,4,JC).NE.0.AND.K(IPU4,1).EQ.3) K(IPU4,J+3)= 
     &  MSTU(5)*(MINT(84)+ICOL(KCC,4,JC))   
    
C...Copy outgoing partons to documentation lines.   
        DO 450 I=1,2    
        I1=MINT(83)+IDOC-2+I    
        I2=MINT(84)+2+I 
        K(I1,1)=21  
        K(I1,2)=K(I2,2) 
        IF(IDOC.LE.9) K(I1,3)=0 
        IF(IDOC.GE.11) K(I1,3)=MINT(83)+2+I 
        DO 450 J=1,5    
  450   P(I1,J)=P(I2,J) 
      ENDIF 
      MINT(52)=N    
    
C...Low-pT events: remove gluons used for string drawing purposes.  
      IF(ISUB.EQ.95) THEN   
        K(IPU3,1)=K(IPU3,1)+10  
        K(IPU4,1)=K(IPU4,1)+10  
        DO 460 J=41,66  
  460   VINT(J)=0.  
        DO 470 I=MINT(83)+5,MINT(83)+8  
        DO 470 J=1,5    
  470   P(I,J)=0.   
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYSSPA(IPU1,IPU2)  
    
C...Generates spacelike parton showers. 
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      DIMENSION KFLS(4),IS(2),XS(2),ZS(2),Q2S(2),TEVS(2),ROBO(5),   
     &XFS(2,-6:6),XFA(-6:6),XFB(-6:6),XFN(-6:6),WTAP(-6:6),WTSF(-6:6),  
     &THE2(2),ALAM(2),DQ2(3),DPC(3),DPD(4),DPB(4)   
    
C...Calculate maximum virtuality and check that evolution possible. 
      IPUS1=IPU1    
      IPUS2=IPU2    
      ISUB=MINT(1)  
      Q2E=VINT(52)  
      IF(ISET(ISUB).EQ.1) THEN  
        Q2E=Q2E/PARP(67)    
      ELSEIF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) THEN   
        Q2E=PMAS(23,1)**2   
        IF(ISUB.EQ.8.OR.ISUB.EQ.76.OR.ISUB.EQ.77) Q2E=PMAS(24,1)**2 
      ENDIF 
      TMAX=LOG(PARP(67)*PARP(63)*Q2E/PARP(61)**2)   
      IF(PARP(67)*Q2E.LT.MAX(PARP(62)**2,2.*PARP(61)**2).OR.    
     &TMAX.LT.0.2) RETURN   
    
C...Common constants and initial values. Save normal Lambda value.  
      XE0=2.*PARP(65)/VINT(1)   
      ALAMS=PARU(111)   
      PARU(111)=PARP(61)    
      NS=N  
  100 N=NS  
      DO 110 JT=1,2 
      KFLS(JT)=MINT(14+JT)  
      KFLS(JT+2)=KFLS(JT)   
      XS(JT)=VINT(40+JT)    
      ZS(JT)=1. 
      Q2S(JT)=PARP(67)*Q2E  
      TEVS(JT)=TMAX 
      ALAM(JT)=PARP(61) 
      THE2(JT)=100. 
      DO 110 KFL=-6,6   
  110 XFS(JT,KFL)=XSFX(JT,KFL)  
      DSH=dble(VINT(44))
      IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) DSH=dble(VINT(26)*VINT(2))
cms.. pre-initialize for compiler
      KFLA=0
      Z=0.
      TEVB=0.
      THE2T=0.

C...Pick up leg with highest virtuality.    
  120 N=N+1 
      JT=1  
      IF(N.GT.NS+1.AND.Q2S(2).GT.Q2S(1)) JT=2   
      KFLB=KFLS(JT) 
      XB=XS(JT) 
      DO 130 KFL=-6,6   
  130 XFB(KFL)=XFS(JT,KFL)  
      DSHR=2D0*SQRT(DSH)    
      DSHZ=DSH/DBLE(ZS(JT)) 
      XE=MAX(XE0,XB*(1./(1.-PARP(66))-1.))  
      IF(XB+XE.GE.0.999) THEN   
        Q2B=0.  
        GOTO 220    
      ENDIF 
    
C...Maximum Q2 without or with Q2 ordering. Effective Lambda and n_f.   
      IF(MSTP(62).LE.1) THEN    
        Q2B=0.5*(1./ZS(JT)+1.)*Q2S(JT)+0.5*(1./ZS(JT)-1.)*(Q2S(3-JT)-   
     &  SNGL(DSH)+SQRT((SNGL(DSH)+Q2S(1)+Q2S(2))**2+8.*Q2S(1)*Q2S(2)*   
     &  ZS(JT)/(1.-ZS(JT))))    
        TEVB=LOG(PARP(63)*Q2B/ALAM(JT)**2)  
      ELSE  
        Q2B=Q2S(JT) 
        TEVB=TEVS(JT)   
      ENDIF 
      ALSDUM=ULALPS(PARP(63)*Q2B)   
      TEVB=TEVB+2.*LOG(ALAM(JT)/PARU(117))  
      TEVBSV=TEVB   
      ALAM(JT)=PARU(117)    
      B0=(33.-2.*MSTU(118))/6.  
    
C...Calculate Altarelli-Parisi and structure function weights.  
      DO 140 KFL=-6,6   
      WTAP(KFL)=0.  
  140 WTSF(KFL)=0.  
      IF(KFLB.EQ.21) THEN   
        WTAPQ=16.*(1.-SQRT(XB+XE))/(3.*SQRT(XB))    
        DO 150 KFL=-MSTP(54),MSTP(54)   
        IF(KFL.EQ.0) WTAP(KFL)=6.*LOG((1.-XB)/XE)   
  150   IF(KFL.NE.0) WTAP(KFL)=WTAPQ    
      ELSE  
        WTAP(0)=0.5*XB*(1./(XB+XE)-1.)  
        WTAP(KFLB)=8.*LOG((1.-XB)*(XB+XE)/XE)/3.    
      ENDIF 
  160 WTSUM=0.  
      IF(KFLB.NE.21) XFBO=XFB(KFLB) 
      IF(KFLB.EQ.21) XFBO=XFB(0)
C***************************************************************
C**********ERROR HAS OCCURED HERE
      IF(XFBO.EQ.0.0) THEN
                WRITE(MSTU(11),1000)
                WRITE(MSTU(11),1001) KFLB,XFB(KFLB)
                XFBO=0.00001
      ENDIF
C****************************************************************    
      DO 170 KFL=-MSTP(54),MSTP(54) 
      WTSF(KFL)=XFB(KFL)/XFBO   
  170 WTSUM=WTSUM+WTAP(KFL)*WTSF(KFL)   
      WTSUM=MAX(0.0001,WTSUM)   
    
C...Choose new t: fix alpha_s, alpha_s(Q2), alpha_s(k_T2).  
  180 IF(MSTP(64).LE.0) THEN    
        TEVB=TEVB+LOG(RLU(0))*PARU(2)/(PARU(111)*WTSUM) 
      ELSEIF(MSTP(64).EQ.1) THEN    
        TEVB=TEVB*EXP(MAX(-100.,LOG(RLU(0))*B0/WTSUM))  
      ELSE  
        TEVB=TEVB*EXP(MAX(-100.,LOG(RLU(0))*B0/(5.*WTSUM))) 
      ENDIF 
  190 Q2REF=ALAM(JT)**2*EXP(TEVB)   
      Q2B=Q2REF/PARP(63)    
    
C...Evolution ended or select flavour for branching parton. 
      IF(Q2B.LT.PARP(62)**2) THEN   
        Q2B=0.  
      ELSE  
        WTRAN=RLU(0)*WTSUM  
        KFLA=-MSTP(54)-1    
  200   KFLA=KFLA+1 
        WTRAN=WTRAN-WTAP(KFLA)*WTSF(KFLA)   
        IF(KFLA.LT.MSTP(54).AND.WTRAN.GT.0.) GOTO 200   
        IF(KFLA.EQ.0) KFLA=21   
    
C...Choose z value and corrective weight.   
        IF(KFLB.EQ.21.AND.KFLA.EQ.21) THEN  
          Z=1./(1.+((1.-XB)/XB)*(XE/(1.-XB))**RLU(0))   
          WTZ=(1.-Z*(1.-Z))**2  
        ELSEIF(KFLB.EQ.21) THEN 
          Z=XB/(1.-RLU(0)*(1.-SQRT(XB+XE)))**2  
          WTZ=0.5*(1.+(1.-Z)**2)*SQRT(Z)    
        ELSEIF(KFLA.EQ.21) THEN 
          Z=XB*(1.+RLU(0)*(1./(XB+XE)-1.))  
          WTZ=1.-2.*Z*(1.-Z)    
        ELSE    
          Z=1.-(1.-XB)*(XE/((XB+XE)*(1.-XB)))**RLU(0)   
          WTZ=0.5*(1.+Z**2) 
        ENDIF   
    
C...Option with resummation of soft gluon emission as effective z shift.    
        IF(MSTP(65).GE.1) THEN  
          RSOFT=6.  
          IF(KFLB.NE.21) RSOFT=8./3.    
          Z=Z*(TEVB/TEVS(JT))**(RSOFT*XE/((XB+XE)*B0))  
          IF(Z.LE.XB) GOTO 180  
        ENDIF   
    
C...Option with alpha_s(k_T2)Q2): demand k_T2 > cutoff, reweight.   
        IF(MSTP(64).GE.2) THEN  
          IF((1.-Z)*Q2B.LT.PARP(62)**2) GOTO 180    
          ALPRAT=TEVB/(TEVB+LOG(1.-Z))  
          IF(ALPRAT.LT.5.*RLU(0)) GOTO 180  
          IF(ALPRAT.GT.5.) WTZ=WTZ*ALPRAT/5.    
        ENDIF   
    
C...Option with angular ordering requirement.   
        IF(MSTP(62).GE.3) THEN  
          THE2T=(4.*Z**2*Q2B)/(VINT(2)*(1.-Z)*XB**2)    
          IF(THE2T.GT.THE2(JT)) GOTO 180    
        ENDIF   
    
C...Weighting with new structure functions. 
        CALL PYSTFU(MINT(10+JT),XB,Q2REF,XFN,JT)   
        IF(KFLB.NE.21) XFBN=XFN(KFLB)   
        IF(KFLB.EQ.21) XFBN=XFN(0)  
        IF(XFBN.LT.1E-20) THEN  
          IF(KFLA.EQ.KFLB) THEN 
            TEVB=TEVBSV 
            WTAP(KFLB)=0.   
            GOTO 160    
          ELSEIF(TEVBSV-TEVB.GT.0.2) THEN   
            TEVB=0.5*(TEVBSV+TEVB)  
            GOTO 190    
          ELSE  
            XFBN=1E-10  
          ENDIF 
        ENDIF   
        DO 210 KFL=-MSTP(54),MSTP(54)   
  210   XFB(KFL)=XFN(KFL)   
        XA=XB/Z 
        CALL PYSTFU(MINT(10+JT),XA,Q2REF,XFA,JT)   
        IF(KFLA.NE.21) XFAN=XFA(KFLA)   
        IF(KFLA.EQ.21) XFAN=XFA(0)  
        IF(XFAN.LT.1E-20) GOTO 160  
        IF(KFLA.NE.21) WTSFA=WTSF(KFLA) 
        IF(KFLA.EQ.21) WTSFA=WTSF(0)    
        IF(WTZ*XFAN/XFBN.LT.RLU(0)*WTSFA) GOTO 160  
      ENDIF 
    
C...Define two hard scatterers in their CM-frame.   
  220 IF(N.EQ.NS+2) THEN    
        DQ2(JT)=dble(Q2B)
        DPLCM=DSQRT((DSH+DQ2(1)+DQ2(2))**2-4D0*DQ2(1)*DQ2(2))/DSHR   
        DO 240 JR=1,2   
        I=NS+JR 
        IF(JR.EQ.1) IPO=IPUS1   
        IF(JR.EQ.2) IPO=IPUS2   
        DO 230 J=1,5    
        K(I,J)=0    
        P(I,J)=0.   
  230   V(I,J)=0.   
        K(I,1)=14   
        K(I,2)=KFLS(JR+2)   
        K(I,4)=IPO  
        K(I,5)=IPO  
        P(I,3)=sngl(DPLCM)*(-1)**(JR+1)   
        P(I,4)=sngl((DSH+DQ2(3-JR)-DQ2(JR))/DSHR)
        P(I,5)=-SQRT(SNGL(DQ2(JR))) 
        K(IPO,1)=14 
        K(IPO,3)=I  
        K(IPO,4)=MOD(K(IPO,4),MSTU(5))+MSTU(5)*I    
  240   K(IPO,5)=MOD(K(IPO,5),MSTU(5))+MSTU(5)*I    
    
C...Find maximum allowed mass of timelike parton.   
      ELSEIF(N.GT.NS+2) THEN    
        JR=3-JT 
        DQ2(3)=dble(Q2B)
        DPC(1)=dble(P(IS(1),4))
        DPC(2)=dble(P(IS(2),4))
        DPC(3)=dble(0.5*(ABS(P(IS(1),3))+ABS(P(IS(2),3))))
        DPD(1)=DSH+DQ2(JR)+DQ2(JT)  
        DPD(2)=DSHZ+DQ2(JR)+DQ2(3)  
        DPD(3)=SQRT(DPD(1)**2-4D0*DQ2(JR)*DQ2(JT))  
        DPD(4)=SQRT(DPD(2)**2-4D0*DQ2(JR)*DQ2(3))   
        IKIN=0  
        IF(Q2S(JR).GE.(0.5*PARP(62))**2.AND.DPD(1)-DPD(3).GE.   
     &  1D-10*DPD(1)) IKIN=1    
        IF(IKIN.EQ.0) DMSMA=(DQ2(JT)/DBLE(ZS(JT))-DQ2(3))*(DSH/ 
     &  (DSH+DQ2(JT))-DSH/(DSHZ+DQ2(3)))    
        IF(IKIN.EQ.1) DMSMA=(DPD(1)*DPD(2)-DPD(3)*DPD(4))/(2.d0*  
     &  DQ2(JR))-DQ2(JT)-DQ2(3) 
    
C...Generate timelike parton shower (if required).  
        IT=N    
        DO 250 J=1,5    
        K(IT,J)=0   
        P(IT,J)=0.  
  250   V(IT,J)=0.  
        K(IT,1)=3   
        K(IT,2)=21  
        IF(KFLB.EQ.21.AND.KFLS(JT+2).NE.21) K(IT,2)=-KFLS(JT+2) 
        IF(KFLB.NE.21.AND.KFLS(JT+2).EQ.21) K(IT,2)=KFLB    
        P(IT,5)=ULMASS(K(IT,2)) 
        IF(SNGL(DMSMA).LE.P(IT,5)**2) GOTO 100  
        IF(MSTP(63).GE.1) THEN  
          P(IT,4)=sngl((DSHZ-DSH-dble(P(IT,5))**2)/DSHR)
          P(IT,3)=SQRT(P(IT,4)**2-P(IT,5)**2)   
          IF(MSTP(63).EQ.1) THEN    
            Q2TIM=sngl(DMSMA)
          ELSEIF(MSTP(63).EQ.2) THEN    
            Q2TIM=MIN(SNGL(DMSMA),PARP(71)*Q2S(JT)) 
          ELSE  
C'''Here remains to introduce angular ordering in first branching.  
            Q2TIM=sngl(DMSMA)
          ENDIF 
          CALL LUSHOW(IT,0,SQRT(Q2TIM)) 
          IF(N.GE.IT+1) P(IT,5)=P(IT+1,5)   
        ENDIF   
    
C...Reconstruct kinematics of branching: timelike parton shower.    
        DMS=dble(P(IT,5)**2)
        IF(IKIN.EQ.0) DPT2=(DMSMA-DMS)*(DSHZ+DQ2(3))/(DSH+DQ2(JT))  
        IF(IKIN.EQ.1) DPT2=(DMSMA-DMS)*(0.5d0*DPD(1)*DPD(2)
     &       +0.5d0*DPD(3)*
     &  DPD(4)-DQ2(JR)*(DQ2(JT)+DQ2(3)+DMS))/(4.d0*DSH*DPC(3)**2) 
        IF(DPT2.LT.0.d0) GOTO 100 
        DPB(1)=(0.5d0*DPD(2)-DPC(JR)*(DSHZ+DQ2(JR)-DQ2(JT)-DMS)/  
     &  DSHR)/DPC(3)-DPC(3) 
        P(IT,1)=SQRT(SNGL(DPT2))    
        P(IT,3)=sngl(DPB(1))*(-1)**(JT+1) 
        P(IT,4)=sngl((DSHZ-DSH-DMS)/DSHR)
        IF(N.GE.IT+1) THEN  
          DPB(1)=SQRT(DPB(1)**2+DPT2)   
          DPB(2)=SQRT(DPB(1)**2+DMS)    
          DPB(3)=dble(P(IT+1,3))
          DPB(4)=SQRT(DPB(3)**2+DMS)    
          DBEZ=(DPB(4)*DPB(1)-DPB(3)*DPB(2))/(DPB(4)*DPB(2)-DPB(3)* 
     &    DPB(1))   
          CALL LUDBRB(IT+1,N,0.,0.,0D0,0D0,DBEZ)    
          THE=ULANGL(P(IT,3),P(IT,1))   
          CALL LUDBRB(IT+1,N,THE,0.,0D0,0D0,0D0)    
        ENDIF   
    
C...Reconstruct kinematics of branching: spacelike parton.  
        DO 260 J=1,5    
        K(N+1,J)=0  
        P(N+1,J)=0. 
  260   V(N+1,J)=0. 
        K(N+1,1)=14 
        K(N+1,2)=KFLB   
        P(N+1,1)=P(IT,1)    
        P(N+1,3)=P(IT,3)+P(IS(JT),3)    
        P(N+1,4)=P(IT,4)+P(IS(JT),4)    
        P(N+1,5)=-SQRT(SNGL(DQ2(3)))    
    
C...Define colour flow of branching.    
        K(IS(JT),3)=N+1 
        K(IT,3)=N+1 
        ID1=IT  
        IF((K(N+1,2).GT.0.AND.K(N+1,2).NE.21.AND.K(ID1,2).GT.0.AND. 
     &  K(ID1,2).NE.21).OR.(K(N+1,2).LT.0.AND.K(ID1,2).EQ.21).OR.   
     &  (K(N+1,2).EQ.21.AND.K(ID1,2).EQ.21.AND.RLU(0).GT.0.5).OR.   
     &  (K(N+1,2).EQ.21.AND.K(ID1,2).LT.0)) ID1=IS(JT)  
        ID2=IT+IS(JT)-ID1   
        K(N+1,4)=K(N+1,4)+ID1   
        K(N+1,5)=K(N+1,5)+ID2   
        K(ID1,4)=K(ID1,4)+MSTU(5)*(N+1) 
        K(ID1,5)=K(ID1,5)+MSTU(5)*ID2   
        K(ID2,4)=K(ID2,4)+MSTU(5)*ID1   
        K(ID2,5)=K(ID2,5)+MSTU(5)*(N+1) 
        N=N+1   
    
C...Boost to new CM-frame.  
        CALL LUDBRB(NS+1,N,0.,0.,-DBLE((P(N,1)+P(IS(JR),1))/(P(N,4)+    
     &  P(IS(JR),4))),0D0,-DBLE((P(N,3)+P(IS(JR),3))/(P(N,4)+   
     &  P(IS(JR),4))))  
        IR=N+(JT-1)*(IS(1)-N)   
        CALL LUDBRB(NS+1,N,-ULANGL(P(IR,3),P(IR,1)),PARU(2)*RLU(0), 
     &  0D0,0D0,0D0)    
      ENDIF 
    
C...Save quantities, loop back. 
      IS(JT)=N  
      Q2S(JT)=Q2B   
      DQ2(JT)=dble(Q2B)
      IF(MSTP(62).GE.3) THE2(JT)=THE2T  
      DSH=DSHZ  
      IF(Q2B.GE.(0.5*PARP(62))**2) THEN 
        KFLS(JT+2)=KFLS(JT) 
        KFLS(JT)=KFLA   
        XS(JT)=XA   
        ZS(JT)=Z    
        DO 270 KFL=-6,6 
  270   XFS(JT,KFL)=XFA(KFL)    
        TEVS(JT)=TEVB   
      ELSE  
        IF(JT.EQ.1) IPU1=N  
        IF(JT.EQ.2) IPU2=N  
      ENDIF 
      IF(N.GT.MSTU(4)-MSTU(32)-10) THEN 
        CALL LUERRM(11,'(PYSSPA:) no more memory left in LUJETS')   
        IF(MSTU(21).GE.1) N=NS  
        IF(MSTU(21).GE.1) RETURN    
      ENDIF 
      IF(MAX(Q2S(1),Q2S(2)).GE.(0.5*PARP(62))**2.OR.N.LE.NS+1) GOTO 120 
    
C...Boost hard scattering partons to frame of shower initiators.    
      DO 280 J=1,3  
  280 ROBO(J+2)=(P(NS+1,J)+P(NS+2,J))/(P(NS+1,4)+P(NS+2,4)) 
      DO 290 J=1,5  
  290 P(N+2,J)=P(NS+1,J)    
      ROBOT=ROBO(3)**2+ROBO(4)**2+ROBO(5)**2    
      IF(ROBOT.GE.0.999999) THEN    
        ROBOT=1.00001*SQRT(ROBOT)   
        ROBO(3)=ROBO(3)/ROBOT   
        ROBO(4)=ROBO(4)/ROBOT   
        ROBO(5)=ROBO(5)/ROBOT   
      ENDIF 
      CALL LUDBRB(N+2,N+2,0.,0.,-DBLE(ROBO(3)),-DBLE(ROBO(4)),  
     &-DBLE(ROBO(5)))   
      ROBO(2)=ULANGL(P(N+2,1),P(N+2,2)) 
      ROBO(1)=ULANGL(P(N+2,3),SQRT(P(N+2,1)**2+P(N+2,2)**2))    
      CALL LUDBRB(MINT(83)+5,NS,ROBO(1),ROBO(2),DBLE(ROBO(3)),  
     &DBLE(ROBO(4)),DBLE(ROBO(5)))  
    
C...Store user information. Reset Lambda value. 
      K(IPU1,3)=MINT(83)+3  
      K(IPU2,3)=MINT(83)+4  
      DO 300 JT=1,2 
      MINT(12+JT)=KFLS(JT)  
  300 VINT(140+JT)=XS(JT)   
      PARU(111)=ALAMS   
 1000 FORMAT(5X,'structure function has a zero point here')
 1001 FORMAT(5X,'xf(x,i=',I5,')=',F10.5)

      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYMULT(MMUL)   
    
C...Initializes treatment of multiple interactions, selects kinematics  
C...of hardest interaction if low-pT physics included in run, and   
C...generates all non-hardest interactions. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      DIMENSION NMUL(20),SIGM(20),KSTR(500,2)   
      SAVE XT2,XT2FAC,XC2,XTS,IRBIN,RBIN,NMUL,SIGM  
    
C...Initialization of multiple interaction treatment.   
      IF(MMUL.EQ.1) THEN    
        IF(MSTP(122).GE.1) WRITE(MSTU(11),1000) MSTP(82)    
        ISUB=96 
        MINT(1)=96  
        VINT(63)=0. 
        VINT(64)=0. 
        VINT(143)=1.    
        VINT(144)=1.    
    
C...Loop over phase space points: xT2 choice in 20 bins.    
  100   SIGSUM=0.   
        DO 120 IXT2=1,20    
        NMUL(IXT2)=MSTP(83) 
        SIGM(IXT2)=0.   
        DO 110 ITRY=1,MSTP(83)  
        RSCA=0.05*((21-IXT2)-RLU(0))    
        XT2=VINT(149)*(1.+VINT(149))/(VINT(149)+RSCA)-VINT(149) 
        XT2=MAX(0.01*VINT(149),XT2) 
        VINT(25)=XT2    
    
C...Choose tau and y*. Calculate cos(theta-hat).    
        IF(RLU(0).LE.COEF(ISUB,1)) THEN 
          TAUP=(2.*(1.+SQRT(1.-XT2))/XT2-1.)**RLU(0)    
          TAU=XT2*(1.+TAUP)**2/(4.*TAUP)    
        ELSE    
          TAU=XT2*(1.+TAN(RLU(0)*ATAN(SQRT(1./XT2-1.)))**2) 
        ENDIF   
        VINT(21)=TAU    
        CALL PYKLIM(2)  
        RYST=RLU(0) 
        MYST=1  
        IF(RYST.GT.COEF(ISUB,7)) MYST=2 
        IF(RYST.GT.COEF(ISUB,7)+COEF(ISUB,8)) MYST=3    
        CALL PYKMAP(2,MYST,RLU(0))  
        VINT(23)=SQRT(MAX(0.,1.-XT2/TAU))*(-1)**INT(1.5+RLU(0)) 
    
C...Calculate differential cross-section.   
        VINT(71)=0.5*VINT(1)*SQRT(XT2)  
        CALL PYSIGH(NCHN,SIGS)  
  110   SIGM(IXT2)=SIGM(IXT2)+SIGS  
  120   SIGSUM=SIGSUM+SIGM(IXT2)    
        SIGSUM=SIGSUM/(20.*MSTP(83))    
    
C...Reject result if sigma(parton-parton) is smaller than hadronic one. 
        IF(SIGSUM.LT.1.1*VINT(106)) THEN    
          IF(MSTP(122).GE.1) WRITE(MSTU(11),1100) PARP(82),SIGSUM   
          PARP(82)=0.9*PARP(82) 
          VINT(149)=4.*PARP(82)**2/VINT(2)  
          GOTO 100  
        ENDIF   
        IF(MSTP(122).GE.1) WRITE(MSTU(11),1200) PARP(82), SIGSUM    
    
C...Start iteration to find k factor.   
        YKE=SIGSUM/VINT(106)    
        SO=0.5  
        XI=0.   
        YI=0.   
        XK=0.5  
        IIT=0   
  130   IF(IIT.EQ.0) THEN   
          XK=2.*XK  
        ELSEIF(IIT.EQ.1) THEN   
          XK=0.5*XK 
        ELSE    
          XK=XI+(YKE-YI)*(XF-XI)/(YF-YI)    
        ENDIF   
    
C...Evaluate overlap integrals.
        IF(MSTP(82).EQ.2) THEN  
          SP=0.5*PARU(1)*(1.-EXP(-XK))  
          SOP=SP/PARU(1)    
        ELSE    
cms.. removing to avoid comp warning
cc .. IF(MSTP(82).EQ.3) DELTAB=0.02 
          DELTAB=0.02
          IF(MSTP(82).EQ.4) DELTAB=MIN(0.01,0.05*PARP(84))  
          SP=0. 
          SOP=0.    
          B=-0.5*DELTAB 
  140     B=B+DELTAB    
          IF(MSTP(82).EQ.3) THEN    
            OV=EXP(-B**2)/PARU(2)   
          ELSE  
            CQ2=PARP(84)**2 
            OV=((1.-PARP(83))**2*EXP(-MIN(100.,B**2))+2.*PARP(83)*  
     &      (1.-PARP(83))*2./(1.+CQ2)*EXP(-MIN(100.,B**2*2./(1.+CQ2)))+ 
     &      PARP(83)**2/CQ2*EXP(-MIN(100.,B**2/CQ2)))/PARU(2)   
          ENDIF 
          PACC=1.-EXP(-MIN(100.,PARU(1)*XK*OV)) 
          SP=SP+PARU(2)*B*DELTAB*PACC   
          SOP=SOP+PARU(2)*B*DELTAB*OV*PACC  
          IF(B.LT.1..OR.B*PACC.GT.1E-6) GOTO 140    
        ENDIF   
        YK=PARU(1)*XK*SO/SP 
    
C...Continue iteration until convergence.   
        IF(YK.LT.YKE) THEN  
          XI=XK 
          YI=YK 
          IF(IIT.EQ.1) IIT=2    
        ELSE    
          XF=XK 
          YF=YK 
          IF(IIT.EQ.0) IIT=1    
        ENDIF   
        IF(ABS(YK-YKE).GE.1E-5*YKE) GOTO 130    
    
C...Store some results for subsequent use.  
        VINT(145)=SIGSUM    
        VINT(146)=SOP/SO    
        VINT(147)=SOP/SP    
    
C...Initialize iteration in xT2 for hardest interaction.    
      ELSEIF(MMUL.EQ.2) THEN    
        IF(MSTP(82).LE.0) THEN  
        ELSEIF(MSTP(82).EQ.1) THEN  
          XT2=1.    
          XT2FAC=XSEC(96,1)/VINT(106)*VINT(149)/(1.-VINT(149))  
        ELSEIF(MSTP(82).EQ.2) THEN  
          XT2=1.    
          XT2FAC=VINT(146)*XSEC(96,1)/VINT(106)*VINT(149)*(1.+VINT(149))    
        ELSE    
          XC2=4.*CKIN(3)**2/VINT(2) 
          IF(CKIN(3).LE.CKIN(5).OR.MINT(82).GE.2) XC2=0.    
        ENDIF   
    
      ELSEIF(MMUL.EQ.3) THEN    
C...Low-pT or multiple interactions (first semihard interaction):   
C...choose xT2 according to dpT2/pT2**2*exp(-(sigma above pT2)/norm)    
C...or (MSTP(82)>=2) dpT2/(pT2+pT0**2)**2*exp(-....).   
        ISUB=MINT(1)    
        IF(MSTP(82).LE.0) THEN  
          XT2=0.    
        ELSEIF(MSTP(82).EQ.1) THEN  
          XT2=XT2FAC*XT2/(XT2FAC-XT2*LOG(RLU(0)))   
        ELSEIF(MSTP(82).EQ.2) THEN  
          IF(XT2.LT.1..AND.EXP(-XT2FAC*XT2/(VINT(149)*(XT2+ 
     &    VINT(149)))).GT.RLU(0)) XT2=1.    
          IF(XT2.GE.1.) THEN    
            XT2=(1.+VINT(149))*XT2FAC/(XT2FAC-(1.+VINT(149))*LOG(1.-    
     &      RLU(0)*(1.-EXP(-XT2FAC/(VINT(149)*(1.+VINT(149)))))))-  
     &      VINT(149)   
          ELSE  
            XT2=-XT2FAC/LOG(EXP(-XT2FAC/(XT2+VINT(149)))+RLU(0)*    
     &      (EXP(-XT2FAC/VINT(149))-EXP(-XT2FAC/(XT2+VINT(149)))))- 
     &      VINT(149)   
          ENDIF 
          XT2=MAX(0.01*VINT(149),XT2)   
        ELSE    
          XT2=(XC2+VINT(149))*(1.+VINT(149))/(1.+VINT(149)- 
     &    RLU(0)*(1.-XC2))-VINT(149)    
          XT2=MAX(0.01*VINT(149),XT2)   
        ENDIF   
        VINT(25)=XT2    
    
C...Low-pT: choose xT2, tau, y* and cos(theta-hat) fixed.   
        IF(MSTP(82).LE.1.AND.XT2.LT.VINT(149)) THEN 
          IF(MINT(82).EQ.1) NGEN(0,1)=NGEN(0,1)-1   
          IF(MINT(82).EQ.1) NGEN(ISUB,1)=NGEN(ISUB,1)-1 
          ISUB=95   
          MINT(1)=ISUB  
          VINT(21)=0.01*VINT(149)   
          VINT(22)=0.   
          VINT(23)=0.   
          VINT(25)=0.01*VINT(149)   
    
        ELSE    
C...Multiple interactions (first semihard interaction). 
C...Choose tau and y*. Calculate cos(theta-hat).    
          IF(RLU(0).LE.COEF(ISUB,1)) THEN   
            TAUP=(2.*(1.+SQRT(1.-XT2))/XT2-1.)**RLU(0)  
            TAU=XT2*(1.+TAUP)**2/(4.*TAUP)  
          ELSE  
            TAU=XT2*(1.+TAN(RLU(0)*ATAN(SQRT(1./XT2-1.)))**2)   
          ENDIF 
          VINT(21)=TAU  
          CALL PYKLIM(2)    
          RYST=RLU(0)   
          MYST=1    
          IF(RYST.GT.COEF(ISUB,7)) MYST=2   
          IF(RYST.GT.COEF(ISUB,7)+COEF(ISUB,8)) MYST=3  
          CALL PYKMAP(2,MYST,RLU(0))    
          VINT(23)=SQRT(MAX(0.,1.-XT2/TAU))*(-1)**INT(1.5+RLU(0))   
        ENDIF   
        VINT(71)=0.5*VINT(1)*SQRT(VINT(25)) 
    
C...Store results of cross-section calculation. 
      ELSEIF(MMUL.EQ.4) THEN    
        ISUB=MINT(1)    
        XTS=VINT(25)    
        IF(ISET(ISUB).EQ.1) XTS=VINT(21)    
        IF(ISET(ISUB).EQ.2) XTS=(4.*VINT(48)+2.*VINT(63)+2.*VINT(64))/  
     &  VINT(2) 
        IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) XTS=VINT(26) 
        RBIN=MAX(0.000001,MIN(0.999999,XTS*(1.+VINT(149))/  
     &  (XTS+VINT(149))))   
        IRBIN=INT(1.+20.*RBIN)  
        IF(ISUB.EQ.96) NMUL(IRBIN)=NMUL(IRBIN)+1    
        IF(ISUB.EQ.96) SIGM(IRBIN)=SIGM(IRBIN)+VINT(153)    
    
C...Choose impact parameter.    
      ELSEIF(MMUL.EQ.5) THEN    
        IF(MSTP(82).EQ.3) THEN  
          VINT(148)=RLU(0)/(PARU(2)*VINT(147))  
        ELSE    
          RTYPE=RLU(0)  
          CQ2=PARP(84)**2   
          IF(RTYPE.LT.(1.-PARP(83))**2) THEN    
            B2=-LOG(RLU(0)) 
          ELSEIF(RTYPE.LT.1.-PARP(83)**2) THEN  
            B2=-0.5*(1.+CQ2)*LOG(RLU(0))    
          ELSE  
            B2=-CQ2*LOG(RLU(0)) 
          ENDIF 
          VINT(148)=((1.-PARP(83))**2*EXP(-MIN(100.,B2))+2.*PARP(83)*   
     &    (1.-PARP(83))*2./(1.+CQ2)*EXP(-MIN(100.,B2*2./(1.+CQ2)))+ 
     &    PARP(83)**2/CQ2*EXP(-MIN(100.,B2/CQ2)))/(PARU(2)*VINT(147))   
        ENDIF   
    
C...Multiple interactions (variable impact parameter) : reject with 
C...probability exp(-overlap*cross-section above pT/normalization). 
        RNCOR=(IRBIN-20.*RBIN)*NMUL(IRBIN)  
        SIGCOR=(IRBIN-20.*RBIN)*SIGM(IRBIN) 
        DO 150 IBIN=IRBIN+1,20  
        RNCOR=RNCOR+NMUL(IBIN)  
  150   SIGCOR=SIGCOR+SIGM(IBIN)    
        SIGABV=(SIGCOR/RNCOR)*VINT(149)*(1.-XTS)/(XTS+VINT(149))    
        VINT(150)=EXP(-MIN(100.,VINT(146)*VINT(148)*SIGABV/VINT(106)))  
    
C...Generate additional multiple semihard interactions. 
      ELSEIF(MMUL.EQ.6) THEN    
    
C...Reconstruct strings in hard scattering. 
        ISUB=MINT(1)    
        NMAX=MINT(84)+4 
        IF(ISET(ISUB).EQ.1) NMAX=MINT(84)+2 
        NSTR=0  
        DO 170 I=MINT(84)+1,NMAX    
        KCS=KCHG(LUCOMP(K(I,2)),2)*ISIGN(1,K(I,2))  
        IF(KCS.EQ.0) GOTO 170   
        DO 160 J=1,4    
        IF(KCS.EQ.1.AND.(J.EQ.2.OR.J.EQ.4)) GOTO 160    
        IF(KCS.EQ.-1.AND.(J.EQ.1.OR.J.EQ.3)) GOTO 160   
        IF(J.LE.2) THEN 
          IST=MOD(K(I,J+3)/MSTU(5),MSTU(5)) 
        ELSE    
          IST=MOD(K(I,J+1),MSTU(5)) 
        ENDIF   
        IF(IST.LT.MINT(84).OR.IST.GT.I) GOTO 160    
        IF(KCHG(LUCOMP(K(IST,2)),2).EQ.0) GOTO 160  
        NSTR=NSTR+1 
        IF(J.EQ.1.OR.J.EQ.4) THEN   
          KSTR(NSTR,1)=I    
          KSTR(NSTR,2)=IST  
        ELSE    
          KSTR(NSTR,1)=IST  
          KSTR(NSTR,2)=I    
        ENDIF   
  160   CONTINUE    
  170   CONTINUE    
    
C...Set up starting values for iteration in xT2.    
        XT2=VINT(25)    
        IF(ISET(ISUB).EQ.1) XT2=VINT(21)    
        IF(ISET(ISUB).EQ.2) XT2=(4.*VINT(48)+2.*VINT(63)+2.*VINT(64))/  
     &  VINT(2) 
        IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) XT2=VINT(26) 
        ISUB=96 
        MINT(1)=96  
        IF(MSTP(82).LE.1) THEN  
          XT2FAC=XSEC(ISUB,1)*VINT(149)/((1.-VINT(149))*VINT(106))  
        ELSE    
          XT2FAC=VINT(146)*VINT(148)*XSEC(ISUB,1)/VINT(106)*    
     &    VINT(149)*(1.+VINT(149))  
        ENDIF   
        VINT(63)=0. 
        VINT(64)=0. 
        VINT(151)=0.    
        VINT(152)=0.    
        VINT(143)=1.-VINT(141)  
        VINT(144)=1.-VINT(142)  
    
C...Iterate downwards in xT2.   
  180   IF(MSTP(82).LE.1) THEN  
          XT2=XT2FAC*XT2/(XT2FAC-XT2*LOG(RLU(0)))   
          IF(XT2.LT.VINT(149)) GOTO 220 
        ELSE    
          IF(XT2.LE.0.01*VINT(149)) GOTO 220    
          XT2=XT2FAC*(XT2+VINT(149))/(XT2FAC-(XT2+VINT(149))*   
     &    LOG(RLU(0)))-VINT(149)    
          IF(XT2.LE.0.) GOTO 220    
          XT2=MAX(0.01*VINT(149),XT2)   
        ENDIF   
        VINT(25)=XT2    
    
C...Choose tau and y*. Calculate cos(theta-hat).    
        IF(RLU(0).LE.COEF(ISUB,1)) THEN 
          TAUP=(2.*(1.+SQRT(1.-XT2))/XT2-1.)**RLU(0)    
          TAU=XT2*(1.+TAUP)**2/(4.*TAUP)    
        ELSE    
          TAU=XT2*(1.+TAN(RLU(0)*ATAN(SQRT(1./XT2-1.)))**2) 
        ENDIF   
        VINT(21)=TAU    
        CALL PYKLIM(2)  
        RYST=RLU(0) 
        MYST=1  
        IF(RYST.GT.COEF(ISUB,7)) MYST=2 
        IF(RYST.GT.COEF(ISUB,7)+COEF(ISUB,8)) MYST=3    
        CALL PYKMAP(2,MYST,RLU(0))  
        VINT(23)=SQRT(MAX(0.,1.-XT2/TAU))*(-1)**INT(1.5+RLU(0)) 
    
C...Check that x not used up. Accept or reject kinematical variables.   
        X1M=SQRT(TAU)*EXP(VINT(22)) 
        X2M=SQRT(TAU)*EXP(-VINT(22))    
        IF(VINT(143)-X1M.LT.0.01.OR.VINT(144)-X2M.LT.0.01) GOTO 180 
        VINT(71)=0.5*VINT(1)*SQRT(XT2)  
        CALL PYSIGH(NCHN,SIGS)  
        IF(SIGS.LT.XSEC(ISUB,1)*RLU(0)) GOTO 180    
    
C...Reset K, P and V vectors. Select some variables.    
        DO 190 I=N+1,N+2    
        DO 190 J=1,5    
        K(I,J)=0    
        P(I,J)=0.   
  190   V(I,J)=0.   
        RFLAV=RLU(0)    
        PT=0.5*VINT(1)*SQRT(XT2)    
        PHI=PARU(2)*RLU(0)  
        CTH=VINT(23)    
    
C...Add first parton to event record.   
        K(N+1,1)=3  
        K(N+1,2)=21 
        IF(RFLAV.GE.MAX(PARP(85),PARP(86))) K(N+1,2)=   
     &  1+INT((2.+PARJ(2))*RLU(0))  
        P(N+1,1)=PT*COS(PHI)    
        P(N+1,2)=PT*SIN(PHI)    
        P(N+1,3)=0.25*VINT(1)*(VINT(41)*(1.+CTH)-VINT(42)*(1.-CTH)) 
        P(N+1,4)=0.25*VINT(1)*(VINT(41)*(1.+CTH)+VINT(42)*(1.-CTH)) 
        P(N+1,5)=0. 
    
C...Add second parton to event record.  
        K(N+2,1)=3  
        K(N+2,2)=21 
        IF(K(N+1,2).NE.21) K(N+2,2)=-K(N+1,2)   
        P(N+2,1)=-P(N+1,1)  
        P(N+2,2)=-P(N+1,2)  
        P(N+2,3)=0.25*VINT(1)*(VINT(41)*(1.-CTH)-VINT(42)*(1.+CTH)) 
        P(N+2,4)=0.25*VINT(1)*(VINT(41)*(1.-CTH)+VINT(42)*(1.+CTH)) 
        P(N+2,5)=0. 
    
        IF(RFLAV.LT.PARP(85).AND.NSTR.GE.1) THEN    
C....Choose relevant string pieces to place gluons on.  
          DO 210 I=N+1,N+2  
          DMIN=1E8  
          DO 200 ISTR=1,NSTR    
          I1=KSTR(ISTR,1)   
          I2=KSTR(ISTR,2)   
          DIST=(P(I,4)*P(I1,4)-P(I,1)*P(I1,1)-P(I,2)*P(I1,2)-   
     &    P(I,3)*P(I1,3))*(P(I,4)*P(I2,4)-P(I,1)*P(I2,1)-   
     &    P(I,2)*P(I2,2)-P(I,3)*P(I2,3))/MAX(1.,P(I1,4)*P(I2,4)-    
     &    P(I1,1)*P(I2,1)-P(I1,2)*P(I2,2)-P(I1,3)*P(I2,3))  
          IF(ISTR.EQ.1.OR.DIST.LT.DMIN) THEN    
            DMIN=DIST   
            IST1=I1 
            IST2=I2 
            ISTM=ISTR   
          ENDIF 
  200     CONTINUE  
    
C....Colour flow adjustments, new string pieces.    
          IF(K(IST1,4)/MSTU(5).EQ.IST2) K(IST1,4)=MSTU(5)*I+    
     &    MOD(K(IST1,4),MSTU(5))    
          IF(MOD(K(IST1,5),MSTU(5)).EQ.IST2) K(IST1,5)= 
     &    MSTU(5)*(K(IST1,5)/MSTU(5))+I 
          K(I,5)=MSTU(5)*IST1   
          K(I,4)=MSTU(5)*IST2   
          IF(K(IST2,5)/MSTU(5).EQ.IST1) K(IST2,5)=MSTU(5)*I+    
     &    MOD(K(IST2,5),MSTU(5))    
          IF(MOD(K(IST2,4),MSTU(5)).EQ.IST1) K(IST2,4)= 
     &    MSTU(5)*(K(IST2,4)/MSTU(5))+I 
          KSTR(ISTM,2)=I    
          KSTR(NSTR+1,1)=I  
          KSTR(NSTR+1,2)=IST2   
  210     NSTR=NSTR+1   
    
C...String drawing and colour flow for gluon loop.  
        ELSEIF(K(N+1,2).EQ.21) THEN 
          K(N+1,4)=MSTU(5)*(N+2)    
          K(N+1,5)=MSTU(5)*(N+2)    
          K(N+2,4)=MSTU(5)*(N+1)    
          K(N+2,5)=MSTU(5)*(N+1)    
          KSTR(NSTR+1,1)=N+1    
          KSTR(NSTR+1,2)=N+2    
          KSTR(NSTR+2,1)=N+2    
          KSTR(NSTR+2,2)=N+1    
          NSTR=NSTR+2   
    
C...String drawing and colour flow for q-qbar pair. 
        ELSE    
          K(N+1,4)=MSTU(5)*(N+2)    
          K(N+2,5)=MSTU(5)*(N+1)    
          KSTR(NSTR+1,1)=N+1    
          KSTR(NSTR+1,2)=N+2    
          NSTR=NSTR+1   
        ENDIF   
    
C...Update remaining energy; iterate.   
        N=N+2   
        IF(N.GT.MSTU(4)-MSTU(32)-10) THEN   
          CALL LUERRM(11,'(PYMULT:) no more memory left in LUJETS') 
          IF(MSTU(21).GE.1) RETURN  
        ENDIF   
        MINT(31)=MINT(31)+1 
        VINT(151)=VINT(151)+VINT(41)    
        VINT(152)=VINT(152)+VINT(42)    
        VINT(143)=VINT(143)-VINT(41)    
        VINT(144)=VINT(144)-VINT(42)    
        IF(MINT(31).LT.240) GOTO 180    
  220   CONTINUE    
      ENDIF 
    
C...Format statements for printout. 
 1000 FORMAT(/1X,'****** PYMULT: initialization of multiple inter', 
     &'actions for MSTP(82) =',I2,' ******')    
 1100 FORMAT(8X,'pT0 =',F5.2,' GeV gives sigma(parton-parton) =',1P,    
     &E9.2,' mb: rejected') 
 1200 FORMAT(8X,'pT0 =',F5.2,' GeV gives sigma(parton-parton) =',1P,    
     &E9.2,' mb: accepted') 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYREMN(IPU1,IPU2)  
    
C...Adds on target remnants (one or two from each side) and 
C...includes primordial kT. 
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
      SAVE /HPARNT/
      COMMON/HSTRNG/NFP(300,15),PPHI(300,15),NFT(300,15),PTHI(300,15)
      SAVE /HSTRNG/
C...COMMON BLOCK FROM HIJING
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      DIMENSION KFLCH(2),KFLSP(2),CHI(2),PMS(6),IS(2),ROBO(5)   
    
C...Special case for lepton-lepton interaction. 
      IF(MINT(43).EQ.1) THEN    
        DO 100 JT=1,2   
        I=MINT(83)+JT+2 
        K(I,1)=21   
        K(I,2)=K(I-2,2) 
        K(I,3)=I-2  
        DO 100 J=1,5    
  100   P(I,J)=P(I-2,J) 
      ENDIF 
    
C...Find event type, set pointers.  
cms.. pre-initialize
      IQ=0
      IF(IPU1.EQ.0.AND.IPU2.EQ.0) RETURN    
      ISUB=MINT(1)  
      ILEP=0    
      IF(IPU1.EQ.0) ILEP=1  
      IF(IPU2.EQ.0) ILEP=2  
      IF(ISUB.EQ.95) ILEP=-1    
      IF(ILEP.EQ.1) IQ=MINT(84)+1   
      IF(ILEP.EQ.2) IQ=MINT(84)+2   
      IP=MAX(IPU1,IPU2) 
      ILEPR=MINT(83)+5-ILEP 
      NS=N  
    
C...Define initial partons, including primordial kT.    
cms.. pre-initialize
      SHS=0.
  110 DO 130 JT=1,2 
      I=MINT(83)+JT+2   
      IF(JT.EQ.1) IPU=IPU1  
      IF(JT.EQ.2) IPU=IPU2  
      K(I,1)=21 
      K(I,3)=I-2    
      IF(ISUB.EQ.95) THEN   
        K(I,2)=21   
        SHS=0.  
      ELSEIF(MINT(40+JT).EQ.1.AND.IPU.NE.0) THEN    
        K(I,2)=K(IPU,2) 
        P(I,5)=P(IPU,5) 
        P(I,1)=0.   
        P(I,2)=0.   
        PMS(JT)=P(I,5)**2   
      ELSEIF(IPU.NE.0) THEN 
        K(I,2)=K(IPU,2) 
        P(I,5)=P(IPU,5) 
C...No primordial kT or chosen according to truncated Gaussian or   
C...exponential.
C
c     X.N. Wang (7.22.97)
c
        RPT1=0.0
        RPT2=0.0
        ssw2=(PPHI(IHNT2(11),4)+PTHI(IHNT2(12),4))**2
     &       -(PPHI(IHNT2(11),1)+PTHI(IHNT2(12),1))**2
     &       -(PPHI(IHNT2(11),2)+PTHI(IHNT2(12),2))**2
     &       -(PPHI(IHNT2(11),3)+PTHI(IHNT2(12),3))**2
C
C********this is s of the current NN collision
        IF(ssw2.LE.4.0*PARP(93)**2) GOTO 1211
c
        IF(IHPR2(5).LE.0) THEN
120             IF(MSTP(91).LE.0) THEN
               PT=0. 
             ELSEIF(MSTP(91).EQ.1) THEN
               PT=PARP(91)*SQRT(-LOG(RLU(0)))
             ELSE    
               RPT1=RLU(0)   
               RPT2=RLU(0)   
               PT=-PARP(92)*LOG(RPT1*RPT2)   
             ENDIF   
             IF(PT.GT.PARP(93)) GOTO 120 
             PHI=PARU(2)*RLU(0)  
             RPT1=PT*COS(PHI)  
             RPT2=PT*SIN(PHI)
        ELSE IF(IHPR2(5).EQ.1) THEN
             IF(JT.EQ.1) JPT=NFP(IHNT2(11),11)
             IF(JT.EQ.2) JPT=NFT(IHNT2(12),11)
1205             PTGS=PARP(91)*SQRT(-LOG(RLU(0)))
             IF(PTGS.GT.PARP(93)) GO TO 1205
             PHI=2.0*HIPR1(40)*RLU(0)
             RPT1=PTGS*COS(PHI)
             RPT2=PTGS*SIN(PHI)
             DO 1210 iint=1,JPT-1
                PKCSQ=PARP(91)*SQRT(-LOG(RLU(0)))
                PHI=2.0*HIPR1(40)*RLU(0)
                RPT1=RPT1+PKCSQ*COS(PHI)
                RPT2=RPT2+PKCSQ*SIN(PHI)
1210             CONTINUE
             IF(RPT1**2+RPT2**2.GE.ssw2/4.0) GO TO 1205
        ENDIF
C     X.N. Wang
C                     ********When initial interaction among soft partons is
C                             assumed the primordial pt comes from the sum of
C                             pt of JPT-1 number of initial interaction, JPT
C                             is the number of interaction including present
C                             one that nucleon hassuffered 
1211    P(I,1)=RPT1
        P(I,2)=RPT2  
        PMS(JT)=P(I,5)**2+P(I,1)**2+P(I,2)**2   
      ELSE  
        K(I,2)=K(IQ,2)  
        Q2=VINT(52) 
        P(I,5)=-SQRT(Q2)    
        PMS(JT)=-Q2 
        SHS=(1.-VINT(43-JT))*Q2/VINT(43-JT)+VINT(5-JT)**2   
      ENDIF 
  130 CONTINUE  
    
C...Kinematics construction for initial partons.    
      I1=MINT(83)+3 
      I2=MINT(83)+4 
      IF(ILEP.EQ.0) SHS=VINT(141)*VINT(142)*VINT(2)+    
     &(P(I1,1)+P(I2,1))**2+(P(I1,2)+P(I2,2))**2 
      SHR=SQRT(MAX(0.,SHS)) 
      IF(ILEP.EQ.0) THEN    
        IF((SHS-PMS(1)-PMS(2))**2-4.*PMS(1)*PMS(2).LE.0.) GOTO 110  
        P(I1,4)=0.5*(SHR+(PMS(1)-PMS(2))/SHR)   
        P(I1,3)=SQRT(MAX(0.,P(I1,4)**2-PMS(1))) 
        P(I2,4)=SHR-P(I1,4) 
        P(I2,3)=-P(I1,3)    
      ELSEIF(ILEP.EQ.1) THEN    
        P(I1,4)=P(IQ,4) 
        P(I1,3)=P(IQ,3) 
        P(I2,4)=P(IP,4) 
        P(I2,3)=P(IP,3) 
      ELSEIF(ILEP.EQ.2) THEN    
        P(I1,4)=P(IP,4) 
        P(I1,3)=P(IP,3) 
        P(I2,4)=P(IQ,4) 
        P(I2,3)=P(IQ,3) 
      ENDIF 
      IF(MINT(43).EQ.1) RETURN  
    
C...Transform partons to overall CM-frame (not for leptoproduction).    
      IF(ILEP.EQ.0) THEN    
        ROBO(3)=(P(I1,1)+P(I2,1))/SHR   
        ROBO(4)=(P(I1,2)+P(I2,2))/SHR   
        CALL LUDBRB(I1,I2,0.,0.,-DBLE(ROBO(3)),-DBLE(ROBO(4)),0D0)  
        ROBO(2)=ULANGL(P(I1,1),P(I1,2)) 
        CALL LUDBRB(I1,I2,0.,-ROBO(2),0D0,0D0,0D0)  
        ROBO(1)=ULANGL(P(I1,3),P(I1,1)) 
        CALL LUDBRB(I1,I2,-ROBO(1),0.,0D0,0D0,0D0)  
        NMAX=MAX(MINT(52),IPU1,IPU2)    
        CALL LUDBRB(I1,NMAX,ROBO(1),ROBO(2),DBLE(ROBO(3)),DBLE(ROBO(4)),    
     &  0D0)    
        ROBO(5)=MAX(-0.999999,MIN(0.999999,(VINT(141)-VINT(142))/   
     &  (VINT(141)+VINT(142)))) 
        CALL LUDBRB(I1,NMAX,0.,0.,0D0,0D0,DBLE(ROBO(5)))    
      ENDIF 
    
C...Check invariant mass of remnant system: 
C...hadronic events or leptoproduction. 
cms.. pre-initialize to avoid compiler warning
      PEH=0.
      PZH=0.
      PEI=0.
      PZI=0.
      IF(ILEP.LE.0) THEN    
        IF(MSTP(81).LE.0.OR.MSTP(82).LE.0.OR.ISUB.EQ.95) THEN   
          VINT(151)=0.  
          VINT(152)=0.  
        ENDIF   
        PEH=P(I1,4)+P(I2,4)+0.5*VINT(1)*(VINT(151)+VINT(152))   
        PZH=P(I1,3)+P(I2,3)+0.5*VINT(1)*(VINT(151)-VINT(152))   
        SHH=(VINT(1)-PEH)**2-(P(I1,1)+P(I2,1))**2-(P(I1,2)+P(I2,2))**2- 
     &  PZH**2  
        PMMIN=P(MINT(83)+1,5)+P(MINT(83)+2,5)+ULMASS(K(I1,2))+  
     &  ULMASS(K(I2,2)) 
        IF(SHR.GE.VINT(1).OR.SHH.LE.(PMMIN+PARP(111))**2) THEN  
          MINT(51)=1    
          RETURN    
        ENDIF   
        SHR=SQRT(SHH+(P(I1,1)+P(I2,1))**2+(P(I1,2)+P(I2,2))**2) 
      ELSE  
        PEI=P(IQ,4)+P(IP,4) 
        PZI=P(IQ,3)+P(IP,3) 
        PMS(ILEP)=MAX(0.,PEI**2-PZI**2) 
        PMMIN=P(ILEPR-2,5)+ULMASS(K(ILEPR,2))+SQRT(PMS(ILEP))   
        IF(SHR.LE.PMMIN+PARP(111)) THEN 
          MINT(51)=1    
          RETURN    
        ENDIF   
      ENDIF 
    
C...Subdivide remnant if necessary, store first parton. 
  140 I=NS  
      DO 190 JT=1,2 
      IF(JT.EQ.ILEP) GOTO 190   
      IF(JT.EQ.1) IPU=IPU1  
      IF(JT.EQ.2) IPU=IPU2  
      CALL PYSPLI(MINT(10+JT),MINT(12+JT),KFLCH(JT),KFLSP(JT))  
      I=I+1 
      IS(JT)=I  
      DO 150 J=1,5  
      K(I,J)=0  
      P(I,J)=0. 
  150 V(I,J)=0. 
      K(I,1)=3  
      K(I,2)=KFLSP(JT)  
      K(I,3)=MINT(83)+JT    
      P(I,5)=ULMASS(K(I,2)) 
    
C...First parton colour connections and transverse mass.    
      KFLS=(3-KCHG(LUCOMP(KFLSP(JT)),2)*ISIGN(1,KFLSP(JT)))/2   
      K(I,KFLS+3)=IPU   
      K(IPU,6-KFLS)=MOD(K(IPU,6-KFLS),MSTU(5))+MSTU(5)*I    
      IF(KFLCH(JT).EQ.0) THEN   
        P(I,1)=-P(MINT(83)+JT+2,1)  
        P(I,2)=-P(MINT(83)+JT+2,2)  
        PMS(JT)=P(I,5)**2+P(I,1)**2+P(I,2)**2   
    
C...When extra remnant parton or hadron: find relative pT, store.   
      ELSE  
        CALL LUPTDI(1,P(I,1),P(I,2))    
        PMS(JT+2)=P(I,5)**2+P(I,1)**2+P(I,2)**2 
        I=I+1   
        DO 160 J=1,5    
        K(I,J)=0    
        P(I,J)=0.   
  160   V(I,J)=0.   
        K(I,1)=1    
        K(I,2)=KFLCH(JT)    
        K(I,3)=MINT(83)+JT  
        P(I,5)=ULMASS(K(I,2))   
        P(I,1)=-P(MINT(83)+JT+2,1)-P(I-1,1) 
        P(I,2)=-P(MINT(83)+JT+2,2)-P(I-1,2) 
        PMS(JT+4)=P(I,5)**2+P(I,1)**2+P(I,2)**2 
C...Relative distribution of energy for particle into two jets. 
        IMB=1   
        IF(MOD(MINT(10+JT)/1000,10).NE.0) IMB=2 
        IF(IABS(KFLCH(JT)).LE.10.OR.KFLCH(JT).EQ.21) THEN   
          CHIK=PARP(92+2*IMB)   
          IF(MSTP(92).LE.1) THEN    
            IF(IMB.EQ.1) CHI(JT)=RLU(0) 
            IF(IMB.EQ.2) CHI(JT)=1.-SQRT(RLU(0))    
          ELSEIF(MSTP(92).EQ.2) THEN    
            CHI(JT)=1.-RLU(0)**(1./(1.+CHIK))   
          ELSEIF(MSTP(92).EQ.3) THEN    
            CUT=2.*0.3/VINT(1)  
  170       CHI(JT)=RLU(0)**2   
            IF((CHI(JT)**2/(CHI(JT)**2+CUT**2))**0.25*(1.-CHI(JT))**CHIK    
     &      .LT.RLU(0)) GOTO 170    
          ELSE  
            CUT=2.*0.3/VINT(1)  
            CUTR=(1.+SQRT(1.+CUT**2))/CUT   
  180       CHIR=CUT*CUTR**RLU(0)   
            CHI(JT)=(CHIR**2-CUT**2)/(2.*CHIR)  
            IF((1.-CHI(JT))**CHIK.LT.RLU(0)) GOTO 180   
          ENDIF 
C...Relative distribution of energy for particle into jet plus particle.    
        ELSE    
          IF(MSTP(92).LE.1) THEN    
            IF(IMB.EQ.1) CHI(JT)=RLU(0) 
            IF(IMB.EQ.2) CHI(JT)=1.-SQRT(RLU(0))    
          ELSE  
            CHI(JT)=1.-RLU(0)**(1./(1.+PARP(93+2*IMB))) 
          ENDIF 
          IF(MOD(KFLCH(JT)/1000,10).NE.0) CHI(JT)=1.-CHI(JT)    
        ENDIF   
        PMS(JT)=PMS(JT+4)/CHI(JT)+PMS(JT+2)/(1.-CHI(JT))    
        KFLS=KCHG(LUCOMP(KFLCH(JT)),2)*ISIGN(1,KFLCH(JT))   
        IF(KFLS.NE.0) THEN  
          K(I,1)=3  
          KFLS=(3-KFLS)/2   
          K(I,KFLS+3)=IPU   
          K(IPU,6-KFLS)=MOD(K(IPU,6-KFLS),MSTU(5))+MSTU(5)*I    
        ENDIF   
      ENDIF 
  190 CONTINUE  
      IF(SHR.LE.SQRT(PMS(1))+SQRT(PMS(2))) GOTO 140 
      N=I   
    
C...Reconstruct kinematics of remnants.
C...cms initialize variable
      PZ=0. 
      DO 200 JT=1,2 
      IF(JT.EQ.ILEP) GOTO 200   
      PE=0.5*(SHR+(PMS(JT)-PMS(3-JT))/SHR)  
      PZ=SQRT(PE**2-PMS(JT))    
      IF(KFLCH(JT).EQ.0) THEN   
        P(IS(JT),4)=PE  
        P(IS(JT),3)=PZ*(-1)**(JT-1) 
      ELSE  
        PW1=CHI(JT)*(PE+PZ) 
        P(IS(JT)+1,4)=0.5*(PW1+PMS(JT+4)/PW1)   
        P(IS(JT)+1,3)=0.5*(PW1-PMS(JT+4)/PW1)*(-1)**(JT-1)  
        P(IS(JT),4)=PE-P(IS(JT)+1,4)    
        P(IS(JT),3)=PZ*(-1)**(JT-1)-P(IS(JT)+1,3)   
      ENDIF 
  200 CONTINUE  
    
C...Hadronic events: boost remnants to correct longitudinal frame.  
      IF(ILEP.LE.0) THEN    
        CALL LUDBRB(NS+1,N,0.,0.,0D0,0D0,-DBLE(PZH/(VINT(1)-PEH)))  
C...Leptoproduction events: boost colliding subsystem.  
      ELSE  
        NMAX=MAX(IP,MINT(52))   
        PEF=SHR-PE  
        PZF=PZ*(-1)**(ILEP-1)   
        PT2=P(ILEPR,1)**2+P(ILEPR,2)**2 
        PHIPT=ULANGL(P(ILEPR,1),P(ILEPR,2)) 
        CALL LUDBRB(MINT(84)+1,NMAX,0.,-PHIPT,0D0,0D0,0D0)  
        RQP=P(IQ,3)*(PT2+PEI**2)-P(IQ,4)*PEI*PZI    
        SINTH=P(IQ,4)*SQRT(PT2*(PT2+PEI**2)/(RQP**2+PT2*    
     &  P(IQ,4)**2*PZI**2))*SIGN(1.,-RQP)   
        CALL LUDBRB(MINT(84)+1,NMAX,ASIN(SINTH),0.,0D0,0D0,0D0) 
        BETAX=(-PEI*PZI*SINTH+SQRT(PT2*(PT2+PEI**2-(PZI*SINTH)**2)))/   
     &  (PT2+PEI**2)    
        CALL LUDBRB(MINT(84)+1,NMAX,0.,0.,DBLE(BETAX),0D0,0D0)  
        CALL LUDBRB(MINT(84)+1,NMAX,0.,PHIPT,0D0,0D0,0D0)   
        PEM=P(IQ,4)+P(IP,4) 
        PZM=P(IQ,3)+P(IP,3) 
        BETAZ=(-PEM*PZM+PZF*SQRT(PZF**2+PEM**2-PZM**2))/(PZF**2+PEM**2) 
        CALL LUDBRB(MINT(84)+1,NMAX,0.,0.,0D0,0D0,DBLE(BETAZ))  
        CALL LUDBRB(I1,I2,ASIN(SINTH),0.,DBLE(BETAX),0D0,0D0)   
        CALL LUDBRB(I1,I2,0.,PHIPT,0D0,0D0,DBLE(BETAZ)) 
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYRESD 
    
C...Allows resonances to decay (including parton showers for hadronic   
C...channels).  
      IMPLICIT DOUBLE PRECISION(D)  
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      DIMENSION IREF(10,6),KDCY(2),KFL1(2),KFL2(2),NSD(2),ILIN(6),  
     &COUP(6,4),PK(6,4),PKK(6,6),CTHE(2),PHI(2),WDTP(0:40), 
     &WDTE(0:40,0:5)    
      COMPLEX FGK,HA(6,6),HC(6,6)   
    
C...The F, Xi and Xj functions of Gunion and Kunszt 
C...(Phys. Rev. D33, 665, plus errata from the authors).    
      FGK(I1,I2,I3,I4,I5,I6)=4.*HA(I1,I3)*HC(I2,I6)*(HA(I1,I5)* 
     &HC(I1,I4)+HA(I3,I5)*HC(I3,I4))    
      DIGK(DT,DU)=-4.d0*D34*D56+DT*(3.d0*DT+4.d0*DU)
     &     +DT**2*(DT*DU/(D34*D56)-  
     &2.d0*(1.d0/D34+1.d0/D56)*(DT+DU)+2.d0*(D34/D56+D56/D34))
      DJGK(DT,DU)=8.d0*(D34+D56)**2-8.d0*(D34+D56)*(DT+DU)-6.d0*DT*DU-    
     &2.d0*DT*DU*(DT*DU/(D34*D56)-2.d0*(1.d0/D34+1.d0/D56)*(DT+DU)+ 
     &2.d0*(D34/D56+D56/D34)) 
    
C...Define initial two objects, initialize loop.    
      ISUB=MINT(1)  
      SH=VINT(44)
C...Initialize variable with default value
      DO I=1,6
         IREF(1,I)=0.0
      ENDDO

      IF(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3) THEN   
        IREF(1,1)=MINT(84)+2+ISET(ISUB) 
        IREF(1,2)=0 
        IREF(1,3)=MINT(83)+6+ISET(ISUB) 
        IREF(1,4)=0 
      ELSEIF(ISET(ISUB).EQ.2.OR.ISET(ISUB).EQ.4) THEN   
        IREF(1,1)=MINT(84)+1+ISET(ISUB) 
        IREF(1,2)=MINT(84)+2+ISET(ISUB) 
        IREF(1,3)=MINT(83)+5+ISET(ISUB) 
        IREF(1,4)=MINT(83)+6+ISET(ISUB) 
      ENDIF 
      NP=1  
      IP=0  
  100 IP=IP+1   
      NINH=0    
    
C...Loop over one/two resonances; reset decay rates.    
      JTMAX=2   
cms.. pre-intialize
      I12=0
      IF(IP.EQ.1.AND.(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3)) JTMAX=1  
      DO 140 JT=1,JTMAX 
      KDCY(JT)=0    
      KFL1(JT)=0    
      KFL2(JT)=0
      NSD(JT)=IREF(IP,JT)   
      ID=IREF(IP,JT)    
      IF(ID.EQ.0) GOTO 140  
      KFA=IABS(K(ID,2)) 
      IF(KFA.LT.23.OR.KFA.GT.40) GOTO 140   
      IF(MDCY(KFA,1).NE.0) THEN 
        IF(ISUB.EQ.1.OR.ISUB.EQ.141) MINT(61)=1 
        CALL PYWIDT(KFA,P(ID,5),WDTP,WDTE)  
        IF(KCHG(KFA,3).EQ.0) THEN   
          IPM=2 
        ELSE    
          IPM=(5+ISIGN(1,K(ID,2)))/2    
        ENDIF   
        IF(JTMAX.EQ.1.OR.IABS(K(IREF(IP,1),2)).NE.IABS(K(IREF(IP,2),2)))    
     &  THEN    
          I12=4 
        ELSE    
          IF(JT.EQ.1) I12=INT(4.5+RLU(0))   
          I12=9-I12 
        ENDIF   
        RKFL=(WDTE(0,1)+WDTE(0,IPM)+WDTE(0,I12))*RLU(0) 
        DO 120 I=1,MDCY(KFA,3)  
        IDC=I+MDCY(KFA,2)-1 
        KFL1(JT)=KFDP(IDC,1)*ISIGN(1,K(ID,2))   
        KFL2(JT)=KFDP(IDC,2)*ISIGN(1,K(ID,2))   
        RKFL=RKFL-(WDTE(I,1)+WDTE(I,IPM)+WDTE(I,I12))   
        IF(RKFL.LE.0.) GOTO 130 
  120   CONTINUE    
  130   CONTINUE    
      ENDIF 
    
C...Summarize result on decay channel chosen.   
      IF((KFA.EQ.23.OR.KFA.EQ.24).AND.KFL1(JT).EQ.0) NINH=NINH+1    
      IF(KFL1(JT).EQ.0) GOTO 140    
      KDCY(JT)=2    
      IF(IABS(KFL1(JT)).LE.10.OR.KFL1(JT).EQ.21) KDCY(JT)=1 
      IF((IABS(KFL1(JT)).GE.23.AND.IABS(KFL1(JT)).LE.25).OR.    
     &(IABS(KFL1(JT)).EQ.37)) KDCY(JT)=3    
      NSD(JT)=N 
    
C...Fill decay products, prepared for parton showers for quarks.    
clin-8/19/02 avoid actual argument in common blocks of LU2ENT:
      pid5=P(ID,5)
      IF(KDCY(JT).EQ.1) THEN    
c        CALL LU2ENT(-(N+1),KFL1(JT),KFL2(JT),P(ID,5))   
        CALL LU2ENT(-(N+1),KFL1(JT),KFL2(JT),pid5)   
      ELSE  
c        CALL LU2ENT(N+1,KFL1(JT),KFL2(JT),P(ID,5))  
        CALL LU2ENT(N+1,KFL1(JT),KFL2(JT),pid5)  
      ENDIF 

      IF(JTMAX.EQ.1) THEN   
        CTHE(JT)=VINT(13)+(VINT(33)-VINT(13)+VINT(34)-VINT(14))*RLU(0)  
        IF(CTHE(JT).GT.VINT(33)) CTHE(JT)=CTHE(JT)+VINT(14)-VINT(33)    
        PHI(JT)=VINT(24)    
      ELSE  
        CTHE(JT)=2.*RLU(0)-1.   
        PHI(JT)=PARU(2)*RLU(0)  
      ENDIF 
  140 CONTINUE  
      IF(MINT(3).EQ.1.AND.IP.EQ.1) THEN 
        MINT(25)=KFL1(1)    
        MINT(26)=KFL2(1)    
      ENDIF 
      IF(JTMAX.EQ.1.AND.KDCY(1).EQ.0) GOTO 530  
      IF(JTMAX.EQ.2.AND.KDCY(1).EQ.0.AND.KDCY(2).EQ.0) GOTO 530 
      IF(MSTP(45).LE.0.OR.IREF(IP,2).EQ.0.OR.NINH.GE.1) GOTO 500    
      IF(K(IREF(1,1),2).EQ.25.AND.IP.EQ.1) GOTO 500 
      IF(K(IREF(1,1),2).EQ.25.AND.KDCY(1)*KDCY(2).EQ.0) GOTO 500    
    
C...Order incoming partons and outgoing resonances. 
      ILIN(1)=MINT(84)+1    
      IF(K(MINT(84)+1,2).GT.0) ILIN(1)=MINT(84)+2   
      IF(K(ILIN(1),2).EQ.21) ILIN(1)=2*MINT(84)+3-ILIN(1)   
      ILIN(2)=2*MINT(84)+3-ILIN(1)  
      IMIN=1    
      IF(IREF(IP,5).EQ.25) IMIN=3   
      IMAX=2    
      IORD=1    
      IF(K(IREF(IP,1),2).EQ.23) IORD=2  
      IF(K(IREF(IP,1),2).EQ.24.AND.K(IREF(IP,2),2).EQ.-24) IORD=2   
      IF(IABS(K(IREF(IP,IORD),2)).EQ.25) IORD=3-IORD    
      IF(KDCY(IORD).EQ.0) IORD=3-IORD   
    
C...Order decay products of resonances. 
      DO 390 JT=IORD,3-IORD,3-2*IORD    
      IF(KDCY(JT).EQ.0) THEN    
        ILIN(IMAX+1)=NSD(JT)    
        IMAX=IMAX+1 
      ELSEIF(K(NSD(JT)+1,2).GT.0) THEN  
        ILIN(IMAX+1)=N+2*JT-1   
        ILIN(IMAX+2)=N+2*JT 
        IMAX=IMAX+2 
        K(N+2*JT-1,2)=K(NSD(JT)+1,2)    
        K(N+2*JT,2)=K(NSD(JT)+2,2)  
      ELSE  
        ILIN(IMAX+1)=N+2*JT 
        ILIN(IMAX+2)=N+2*JT-1   
        IMAX=IMAX+2 
        K(N+2*JT-1,2)=K(NSD(JT)+1,2)    
        K(N+2*JT,2)=K(NSD(JT)+2,2)  
      ENDIF 
  390 CONTINUE  
    
C...Find charge, isospin, left- and righthanded couplings.  
      XW=PARU(102)  
      DO 410 I=IMIN,IMAX    
      DO 400 J=1,4  
  400 COUP(I,J)=0.  
      KFA=IABS(K(ILIN(I),2))    
      IF(KFA.GT.20) GOTO 410    
      COUP(I,1)=LUCHGE(KFA)/3.  
      COUP(I,2)=(-1)**MOD(KFA,2)    
      COUP(I,4)=-2.*COUP(I,1)*XW    
      COUP(I,3)=COUP(I,2)+COUP(I,4) 
  410 CONTINUE  
      SQMZ=PMAS(23,1)**2    
      GZMZ=PMAS(23,1)*PMAS(23,2)    
      SQMW=PMAS(24,1)**2    
      GZMW=PMAS(24,1)*PMAS(24,2)    
      SQMZP=PMAS(32,1)**2   
      GZMZP=PMAS(32,1)*PMAS(32,2)   
    
C...Select random angles; construct massless four-vectors.  
  420 DO 430 I=N+1,N+4  
      K(I,1)=1  
      DO 430 J=1,5  
  430 P(I,J)=0. 
      DO 440 JT=1,JTMAX 
      IF(KDCY(JT).EQ.0) GOTO 440    
      ID=IREF(IP,JT)    
      P(N+2*JT-1,3)=0.5*P(ID,5) 
      P(N+2*JT-1,4)=0.5*P(ID,5) 
      P(N+2*JT,3)=-0.5*P(ID,5)  
      P(N+2*JT,4)=0.5*P(ID,5)   
      CTHE(JT)=2.*RLU(0)-1. 
      PHI(JT)=PARU(2)*RLU(0)    
      CALL LUDBRB(N+2*JT-1,N+2*JT,ACOS(CTHE(JT)),PHI(JT),   
     &DBLE(P(ID,1)/P(ID,4)),DBLE(P(ID,2)/P(ID,4)),DBLE(P(ID,3)/P(ID,4)))    
  440 CONTINUE  
    
C...Store incoming and outgoing momenta, with random rotation to    
C...avoid accidental zeroes in HA expressions.  
      DO 450 I=1,IMAX   
      K(N+4+I,1)=1  
      P(N+4+I,4)=SQRT(P(ILIN(I),1)**2+P(ILIN(I),2)**2+P(ILIN(I),3)**2+  
     &P(ILIN(I),5)**2)  
      P(N+4+I,5)=P(ILIN(I),5)   
      DO 450 J=1,3  
  450 P(N+4+I,J)=P(ILIN(I),J)   
      THERR=ACOS(2.*RLU(0)-1.)  
      PHIRR=PARU(2)*RLU(0)  
      CALL LUDBRB(N+5,N+4+IMAX,THERR,PHIRR,0D0,0D0,0D0) 
      DO 460 I=1,IMAX   
      DO 460 J=1,4  
  460 PK(I,J)=P(N+4+I,J)    
    
C...Calculate internal products.    
      IF(ISUB.EQ.22.OR.ISUB.EQ.23.OR.ISUB.EQ.25) THEN   
        DO 470 I1=IMIN,IMAX-1   
        DO 470 I2=I1+1,IMAX 
        HA(I1,I2)=SQRT((PK(I1,4)-PK(I1,3))*(PK(I2,4)+PK(I2,3))/ 
     &  (1E-20+PK(I1,1)**2+PK(I1,2)**2))*CMPLX(PK(I1,1),PK(I1,2))-  
     &  SQRT((PK(I1,4)+PK(I1,3))*(PK(I2,4)-PK(I2,3))/   
     &  (1E-20+PK(I2,1)**2+PK(I2,2)**2))*CMPLX(PK(I2,1),PK(I2,2))   
        HC(I1,I2)=CONJG(HA(I1,I2))  
        IF(I1.LE.2) HA(I1,I2)=CMPLX(0.,1.)*HA(I1,I2)    
        IF(I1.LE.2) HC(I1,I2)=CMPLX(0.,1.)*HC(I1,I2)    
        HA(I2,I1)=-HA(I1,I2)    
  470   HC(I2,I1)=-HC(I1,I2)    
      ENDIF 
      DO 480 I=1,2  
      DO 480 J=1,4  
  480 PK(I,J)=-PK(I,J)  
      DO 490 I1=IMIN,IMAX-1 
      DO 490 I2=I1+1,IMAX   
      PKK(I1,I2)=2.*(PK(I1,4)*PK(I2,4)-PK(I1,1)*PK(I2,1)-   
     &PK(I1,2)*PK(I2,2)-PK(I1,3)*PK(I2,3))  
  490 PKK(I2,I1)=PKK(I1,I2) 
   
cms.. pre-initialize
      WT=0.
      IF(IREF(IP,5).EQ.25) THEN 
C...Angular weight for H0 -> Z0 + Z0 or W+ + W- -> 4 quarks/leptons 
        WT=16.*PKK(3,5)*PKK(4,6)    
        IF(IP.EQ.1) WTMAX=SH**2 
        IF(IP.GE.2) WTMAX=P(IREF(IP,6),5)**4    
    
      ELSEIF(ISUB.EQ.1) THEN    
        IF(KFA.NE.37) THEN  
C...Angular weight for gamma*/Z0 -> 2 quarks/leptons    
          EI=KCHG(IABS(MINT(15)),1)/3.  
          AI=SIGN(1.,EI+0.1)    
          VI=AI-4.*EI*XW    
          EF=KCHG(KFA,1)/3. 
          AF=SIGN(1.,EF+0.1)    
          VF=AF-4.*EF*XW    
          GG=1. 
          GZ=1./(8.*XW*(1.-XW))*SH*(SH-SQMZ)/((SH-SQMZ)**2+GZMZ**2) 
          ZZ=1./(16.*XW*(1.-XW))**2*SH**2/((SH-SQMZ)**2+GZMZ**2)    
          IF(MSTP(43).EQ.1) THEN    
C...Only gamma* production included 
            GZ=0.   
            ZZ=0.   
          ELSEIF(MSTP(43).EQ.2) THEN    
C...Only Z0 production included 
            GG=0.   
            GZ=0.   
          ENDIF 
          ASYM=2.*(EI*AI*GZ*EF*AF+4.*VI*AI*ZZ*VF*AF)/(EI**2*GG*EF**2+   
     &    EI*VI*GZ*EF*VF+(VI**2+AI**2)*ZZ*(VF**2+AF**2))    
          WT=1.+ASYM*CTHE(JT)+CTHE(JT)**2   
          WTMAX=2.+ABS(ASYM)    
        ELSE    
C...Angular weight for gamma*/Z0 -> H+ + H- 
          WT=1.-CTHE(JT)**2 
          WTMAX=1.  
        ENDIF   
    
      ELSEIF(ISUB.EQ.2) THEN    
C...Angular weight for W+/- -> 2 quarks/leptons 
        WT=(1.+CTHE(JT))**2 
        WTMAX=4.    
    
      ELSEIF(ISUB.EQ.15.OR.ISUB.EQ.19) THEN 
C...Angular weight for f + fb -> gluon/gamma + Z0 ->    
C...-> gluon/gamma + 2 quarks/leptons   
        WT=((COUP(1,3)*COUP(3,3))**2+(COUP(1,4)*COUP(3,4))**2)* 
     &  (PKK(1,3)**2+PKK(2,4)**2)+((COUP(1,3)*COUP(3,4))**2+    
     &  (COUP(1,4)*COUP(3,3))**2)*(PKK(1,4)**2+PKK(2,3)**2) 
        WTMAX=(COUP(1,3)**2+COUP(1,4)**2)*(COUP(3,3)**2+COUP(3,4)**2)*  
     &  ((PKK(1,3)+PKK(1,4))**2+(PKK(2,3)+PKK(2,4))**2) 
    
      ELSEIF(ISUB.EQ.16.OR.ISUB.EQ.20) THEN 
C...Angular weight for f + fb' -> gluon/gamma + W+/- -> 
C...-> gluon/gamma + 2 quarks/leptons   
        WT=PKK(1,3)**2+PKK(2,4)**2  
        WTMAX=(PKK(1,3)+PKK(1,4))**2+(PKK(2,3)+PKK(2,4))**2 
    
      ELSEIF(ISUB.EQ.22) THEN   
C...Angular weight for f + fb -> Z0 + Z0 -> 4 quarks/leptons    
        S34=P(IREF(IP,IORD),5)**2   
        S56=P(IREF(IP,3-IORD),5)**2 
        TI=PKK(1,3)+PKK(1,4)+S34    
        UI=PKK(1,5)+PKK(1,6)+S56    
        WT=COUP(1,3)**4*((COUP(3,3)*COUP(5,3)*ABS(FGK(1,2,3,4,5,6)/ 
     &  TI+FGK(1,2,5,6,3,4)/UI))**2+(COUP(3,4)*COUP(5,3)*ABS(   
     &  FGK(1,2,4,3,5,6)/TI+FGK(1,2,5,6,4,3)/UI))**2+(COUP(3,3)*    
     &  COUP(5,4)*ABS(FGK(1,2,3,4,6,5)/TI+FGK(1,2,6,5,3,4)/UI))**2+ 
     &  (COUP(3,4)*COUP(5,4)*ABS(FGK(1,2,4,3,6,5)/TI+FGK(1,2,6,5,4,3)/  
     &  UI))**2)+COUP(1,4)**4*((COUP(3,3)*COUP(5,3)*ABS(    
     &  FGK(2,1,5,6,3,4)/TI+FGK(2,1,3,4,5,6)/UI))**2+(COUP(3,4)*    
     &  COUP(5,3)*ABS(FGK(2,1,6,5,3,4)/TI+FGK(2,1,3,4,6,5)/UI))**2+ 
     &  (COUP(3,3)*COUP(5,4)*ABS(FGK(2,1,5,6,4,3)/TI+FGK(2,1,4,3,5,6)/  
     &  UI))**2+(COUP(3,4)*COUP(5,4)*ABS(FGK(2,1,6,5,4,3)/TI+   
     &  FGK(2,1,4,3,6,5)/UI))**2)   
        WTMAX=4.*S34*S56*(COUP(1,3)**4+COUP(1,4)**4)*(COUP(3,3)**2+ 
     &  COUP(3,4)**2)*(COUP(5,3)**2+COUP(5,4)**2)*4.*(TI/UI+UI/TI+  
     &  2.*SH*(S34+S56)/(TI*UI)-S34*S56*(1./TI**2+1./UI**2))    
    
      ELSEIF(ISUB.EQ.23) THEN   
C...Angular weight for f + fb' -> Z0 + W +/- -> 4 quarks/leptons    
        D34=dble(P(IREF(IP,IORD),5)**2)
        D56=dble(P(IREF(IP,3-IORD),5)**2)
        DT=dble(PKK(1,3)+PKK(1,4))+D34    
        DU=dble(PKK(1,5)+PKK(1,6))+D56    
        CAWZ=COUP(2,3)/SNGL(DT)-2.*(1.-XW)*COUP(1,2)/(SH-SQMW)  
        CBWZ=COUP(1,3)/SNGL(DU)+2.*(1.-XW)*COUP(1,2)/(SH-SQMW)  
        WT=COUP(5,3)**2*ABS(CAWZ*FGK(1,2,3,4,5,6)+CBWZ* 
     &  FGK(1,2,5,6,3,4))**2+COUP(5,4)**2*ABS(CAWZ* 
     &  FGK(1,2,3,4,6,5)+CBWZ*FGK(1,2,6,5,3,4))**2  
        WTMAX=4.*sngl(D34*D56)*(COUP(5,3)**2+COUP(5,4)**2)*(CAWZ**2*  
     &       sngl(DIGK(DT,DU))+CBWZ**2*sngl(DIGK(DU,DT))
     &       +CAWZ*CBWZ*sngl(DJGK(DT,DU)))  
    
      ELSEIF(ISUB.EQ.24) THEN   
C...Angular weight for f + fb -> Z0 + H0 -> 2 quarks/leptons + H0   
        WT=((COUP(1,3)*COUP(3,3))**2+(COUP(1,4)*COUP(3,4))**2)* 
     &  PKK(1,3)*PKK(2,4)+((COUP(1,3)*COUP(3,4))**2+(COUP(1,4)* 
     &  COUP(3,3))**2)*PKK(1,4)*PKK(2,3)    
        WTMAX=(COUP(1,3)**2+COUP(1,4)**2)*(COUP(3,3)**2+COUP(3,4)**2)*  
     &  (PKK(1,3)+PKK(1,4))*(PKK(2,3)+PKK(2,4)) 
    
      ELSEIF(ISUB.EQ.25) THEN   
C...Angular weight for f + fb -> W+ + W- -> 4 quarks/leptons    
        D34=dble(P(IREF(IP,IORD),5)**2)
        D56=dble(P(IREF(IP,3-IORD),5)**2)
        DT=dble(PKK(1,3)+PKK(1,4))+D34    
        DU=dble(PKK(1,5)+PKK(1,6))+D56    
        CDWW=(COUP(1,3)*SQMZ/(SH-SQMZ)+COUP(1,2))/SH    
        CAWW=CDWW+0.5*(COUP(1,2)+1.)/SNGL(DT)   
        CBWW=CDWW+0.5*(COUP(1,2)-1.)/SNGL(DU)   
        CCWW=COUP(1,4)*SQMZ/(SH-SQMZ)/SH    
        WT=ABS(CAWW*FGK(1,2,3,4,5,6)-CBWW*FGK(1,2,5,6,3,4))**2+ 
     &  CCWW**2*ABS(FGK(2,1,5,6,3,4)-FGK(2,1,3,4,5,6))**2   
        WTMAX=4.*sngl(D34*D56)*(CAWW**2*sngl(DIGK(DT,DU))
     &       +CBWW**2*sngl(DIGK(DU,DT))-CAWW*CBWW*sngl(DJGK(DT,DU))
     &       +CCWW**2*sngl(DIGK(DT,DU)+DIGK(DU,DT)-DJGK(DT,DU)))
    
      ELSEIF(ISUB.EQ.26) THEN   
C...Angular weight for f + fb' -> W+/- + H0 -> 2 quarks/leptons + H0    
        WT=PKK(1,3)*PKK(2,4)    
        WTMAX=(PKK(1,3)+PKK(1,4))*(PKK(2,3)+PKK(2,4))   
    
      ELSEIF(ISUB.EQ.30) THEN   
C...Angular weight for f + g -> f + Z0 -> f + 2 quarks/leptons  
        IF(K(ILIN(1),2).GT.0) WT=((COUP(1,3)*COUP(3,3))**2+ 
     &  (COUP(1,4)*COUP(3,4))**2)*(PKK(1,4)**2+PKK(3,5)**2)+    
     &  ((COUP(1,3)*COUP(3,4))**2+(COUP(1,4)*COUP(3,3))**2)*    
     &  (PKK(1,3)**2+PKK(4,5)**2)   
        IF(K(ILIN(1),2).LT.0) WT=((COUP(1,3)*COUP(3,3))**2+ 
     &  (COUP(1,4)*COUP(3,4))**2)*(PKK(1,3)**2+PKK(4,5)**2)+    
     &  ((COUP(1,3)*COUP(3,4))**2+(COUP(1,4)*COUP(3,3))**2)*    
     &  (PKK(1,4)**2+PKK(3,5)**2)   
        WTMAX=(COUP(1,3)**2+COUP(1,4)**2)*(COUP(3,3)**2+COUP(3,4)**2)*  
     &  ((PKK(1,3)+PKK(1,4))**2+(PKK(3,5)+PKK(4,5))**2) 
    
      ELSEIF(ISUB.EQ.31) THEN   
C...Angular weight for f + g -> f' + W+/- -> f' + 2 quarks/leptons  
        IF(K(ILIN(1),2).GT.0) WT=PKK(1,4)**2+PKK(3,5)**2    
        IF(K(ILIN(1),2).LT.0) WT=PKK(1,3)**2+PKK(4,5)**2    
        WTMAX=(PKK(1,3)+PKK(1,4))**2+(PKK(3,5)+PKK(4,5))**2 
    
      ELSEIF(ISUB.EQ.141) THEN  
C...Angular weight for gamma*/Z0/Z'0 -> 2 quarks/leptons    
        EI=KCHG(IABS(MINT(15)),1)/3.    
        AI=SIGN(1.,EI+0.1)  
        VI=AI-4.*EI*XW  
        API=SIGN(1.,EI+0.1) 
        VPI=API-4.*EI*XW    
        EF=KCHG(KFA,1)/3.   
        AF=SIGN(1.,EF+0.1)  
        VF=AF-4.*EF*XW  
        APF=SIGN(1.,EF+0.1) 
        VPF=APF-4.*EF*XW    
        GG=1.   
        GZ=1./(8.*XW*(1.-XW))*SH*(SH-SQMZ)/((SH-SQMZ)**2+GZMZ**2)   
        GZP=1./(8.*XW*(1.-XW))*SH*(SH-SQMZP)/((SH-SQMZP)**2+GZMZP**2)   
        ZZ=1./(16.*XW*(1.-XW))**2*SH**2/((SH-SQMZ)**2+GZMZ**2)  
        ZZP=2./(16.*XW*(1.-XW))**2* 
     &  SH**2*((SH-SQMZ)*(SH-SQMZP)+GZMZ*GZMZP)/    
     &  (((SH-SQMZ)**2+GZMZ**2)*((SH-SQMZP)**2+GZMZP**2))   
        ZPZP=1./(16.*XW*(1.-XW))**2*SH**2/((SH-SQMZP)**2+GZMZP**2)  
        IF(MSTP(44).EQ.1) THEN  
C...Only gamma* production included 
          GZ=0. 
          GZP=0.    
          ZZ=0. 
          ZZP=0.    
          ZPZP=0.   
        ELSEIF(MSTP(44).EQ.2) THEN  
C...Only Z0 production included 
          GG=0. 
          GZ=0. 
          GZP=0.    
          ZZP=0.    
          ZPZP=0.   
        ELSEIF(MSTP(44).EQ.3) THEN  
C...Only Z'0 production included    
          GG=0. 
          GZ=0. 
          GZP=0.    
          ZZ=0. 
          ZZP=0.    
        ELSEIF(MSTP(44).EQ.4) THEN  
C...Only gamma*/Z0 production included  
          GZP=0.    
          ZZP=0.    
          ZPZP=0.   
        ELSEIF(MSTP(44).EQ.5) THEN  
C...Only gamma*/Z'0 production included 
          GZ=0. 
          ZZ=0. 
          ZZP=0.    
        ELSEIF(MSTP(44).EQ.6) THEN  
C...Only Z0/Z'0 production included 
          GG=0. 
          GZ=0. 
          GZP=0.    
        ENDIF   
        ASYM=2.*(EI*AI*GZ*EF*AF+EI*API*GZP*EF*APF+4.*VI*AI*ZZ*VF*AF+    
     &  (VI*API+VPI*AI)*ZZP*(VF*APF+VPF*AF)+4.*VPI*API*ZPZP*VPF*APF)/   
     &  (EI**2*GG*EF**2+EI*VI*GZ*EF*VF+EI*VPI*GZP*EF*VPF+   
     &  (VI**2+AI**2)*ZZ*(VF**2+AF**2)+(VI*VPI+AI*API)*ZZP* 
     &  (VF*VPF+AF*APF)+(VPI**2+API**2)*ZPZP*(VPF**2+APF**2))   
        WT=1.+ASYM*CTHE(JT)+CTHE(JT)**2 
        WTMAX=2.+ABS(ASYM)  
    
      ELSE  
        WT=1.   
        WTMAX=1.    
      ENDIF 
C...Obtain correct angular distribution by rejection techniques.    
      IF(WT.LT.RLU(0)*WTMAX) GOTO 420   
    
C...Construct massive four-vectors using angles chosen. Mark decayed    
C...resonances, add documentation lines. Shower evolution.  
  500 DO 520 JT=1,JTMAX 
      IF(KDCY(JT).EQ.0) GOTO 520    
      ID=IREF(IP,JT)    
      CALL LUDBRB(NSD(JT)+1,NSD(JT)+2,ACOS(CTHE(JT)),PHI(JT),   
     &DBLE(P(ID,1)/P(ID,4)),DBLE(P(ID,2)/P(ID,4)),DBLE(P(ID,3)/P(ID,4)))    
      K(ID,1)=K(ID,1)+10    
      K(ID,4)=NSD(JT)+1 
      K(ID,5)=NSD(JT)+2 
      IDOC=MINT(83)+MINT(4) 
      DO 510 I=NSD(JT)+1,NSD(JT)+2  
      MINT(4)=MINT(4)+1 
      I1=MINT(83)+MINT(4)   
      K(I,3)=I1 
      K(I1,1)=21    
      K(I1,2)=K(I,2)    
      K(I1,3)=IREF(IP,JT+2) 
      DO 510 J=1,5  
  510 P(I1,J)=P(I,J)    
      IF(JTMAX.EQ.1) THEN   
        MINT(7)=MINT(83)+6+2*ISET(ISUB) 
        MINT(8)=MINT(83)+7+2*ISET(ISUB) 
      ENDIF 
clin-8/19/02 avoid actual argument in common blocks of LUSHOW:
c      IF(MSTP(71).GE.1.AND.KDCY(JT).EQ.1) CALL LUSHOW(NSD(JT)+1,    
c     &NSD(JT)+2,P(ID,5))    
      pid5=P(ID,5)
      IF(MSTP(71).GE.1.AND.KDCY(JT).EQ.1) CALL LUSHOW(NSD(JT)+1,    
     &NSD(JT)+2,pid5)    
    
C...Check if new resonances were produced, loop back if needed. 
      IF(KDCY(JT).NE.3) GOTO 520    
      NP=NP+1   
      IREF(NP,1)=NSD(JT)+1  
      IREF(NP,2)=NSD(JT)+2  
      IREF(NP,3)=IDOC+1 
      IREF(NP,4)=IDOC+2 
      IREF(NP,5)=K(IREF(IP,JT),2)   
      IREF(NP,6)=IREF(IP,JT)    
  520 CONTINUE  
  530 IF(IP.LT.NP) GOTO 100 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYDIFF 
    
C...Handles diffractive and elastic scattering. 
      COMMON/LUJETS/N,K(9000,5),P(9000,5),V(9000,5)
      SAVE /LUJETS/ 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
    
C...Reset K, P and V vectors. Store incoming particles. 
      DO 100 JT=1,MSTP(126)+10  
      I=MINT(83)+JT 
      DO 100 J=1,5  
      K(I,J)=0  
      P(I,J)=0. 
  100 V(I,J)=0. 
      N=MINT(84)    
      MINT(3)=0 
      MINT(21)=0    
      MINT(22)=0    
      MINT(23)=0    
      MINT(24)=0    
      MINT(4)=4 
      DO 110 JT=1,2 
      I=MINT(83)+JT 
      K(I,1)=21 
      K(I,2)=MINT(10+JT)    
      P(I,5)=VINT(2+JT) 
      P(I,3)=VINT(5)*(-1)**(JT+1)   
  110 P(I,4)=SQRT(P(I,3)**2+P(I,5)**2)  
      MINT(6)=2 
    
C...Subprocess; kinematics. 
      ISUB=MINT(1)  
      SQLAM=(VINT(2)-VINT(63)-VINT(64))**2-4.*VINT(63)*VINT(64) 
      PZ=SQRT(SQLAM)/(2.*VINT(1))   
      DO 150 JT=1,2 
      I=MINT(83)+JT 
      PE=(VINT(2)+VINT(62+JT)-VINT(65-JT))/(2.*VINT(1)) 
    
C...Elastically scattered particle. 
      IF(MINT(16+JT).LE.0) THEN 
        N=N+1   
        K(N,1)=1    
        K(N,2)=K(I,2)   
        K(N,3)=I+2  
        P(N,3)=PZ*(-1)**(JT+1)  
        P(N,4)=PE   
        P(N,5)=P(I,5)   
    
C...Diffracted particle: valence quark kicked out.  
      ELSEIF(MSTP(101).EQ.1) THEN   
        N=N+2   
        K(N-1,1)=2  
        K(N,1)=1    
        K(N-1,3)=I+2    
        K(N,3)=I+2  
        CALL PYSPLI(K(I,2),21,K(N,2),K(N-1,2))  
        P(N-1,5)=ULMASS(K(N-1,2))   
        P(N,5)=ULMASS(K(N,2))   
        SQLAM=(VINT(62+JT)-P(N-1,5)**2-P(N,5)**2)**2-   
     &  4.*P(N-1,5)**2*P(N,5)**2    
        P(N-1,3)=(PE*SQRT(SQLAM)+PZ*(VINT(62+JT)+P(N-1,5)**2-   
     &  P(N,5)**2))/(2.*VINT(62+JT))*(-1)**(JT+1)   
        P(N-1,4)=SQRT(P(N-1,3)**2+P(N-1,5)**2)  
        P(N,3)=PZ*(-1)**(JT+1)-P(N-1,3) 
        P(N,4)=SQRT(P(N,3)**2+P(N,5)**2)    
    
C...Diffracted particle: gluon kicked out.  
      ELSE  
        N=N+3   
        K(N-2,1)=2  
        K(N-1,1)=2  
        K(N,1)=1    
        K(N-2,3)=I+2    
        K(N-1,3)=I+2    
        K(N,3)=I+2  
        CALL PYSPLI(K(I,2),21,K(N,2),K(N-2,2))  
        K(N-1,2)=21 
        P(N-2,5)=ULMASS(K(N-2,2))   
        P(N-1,5)=0. 
        P(N,5)=ULMASS(K(N,2))   
C...Energy distribution for particle into two jets. 
  120   IMB=1   
        IF(MOD(K(I,2)/1000,10).NE.0) IMB=2  
        CHIK=PARP(92+2*IMB) 
        IF(MSTP(92).LE.1) THEN  
          IF(IMB.EQ.1) CHI=RLU(0)   
          IF(IMB.EQ.2) CHI=1.-SQRT(RLU(0))  
        ELSEIF(MSTP(92).EQ.2) THEN  
          CHI=1.-RLU(0)**(1./(1.+CHIK)) 
        ELSEIF(MSTP(92).EQ.3) THEN  
          CUT=2.*0.3/VINT(1)    
  130     CHI=RLU(0)**2 
          IF((CHI**2/(CHI**2+CUT**2))**0.25*(1.-CHI)**CHIK.LT.  
     &    RLU(0)) GOTO 130  
        ELSE    
          CUT=2.*0.3/VINT(1)    
          CUTR=(1.+SQRT(1.+CUT**2))/CUT 
  140     CHIR=CUT*CUTR**RLU(0) 
          CHI=(CHIR**2-CUT**2)/(2.*CHIR)    
          IF((1.-CHI)**CHIK.LT.RLU(0)) GOTO 140 
        ENDIF   
        IF(CHI.LT.P(N,5)**2/VINT(62+JT).OR.CHI.GT.1.-P(N-2,5)**2/   
     &  VINT(62+JT)) GOTO 120   
        SQM=P(N-2,5)**2/(1.-CHI)+P(N,5)**2/CHI  
        IF((SQRT(SQM)+PARJ(32))**2.GE.VINT(62+JT)) GOTO 120 
        PZI=(PE*(VINT(62+JT)-SQM)+PZ*(VINT(62+JT)+SQM))/    
     &  (2.*VINT(62+JT))    
        PEI=SQRT(PZI**2+SQM)    
        PQQP=(1.-CHI)*(PEI+PZI) 
        P(N-2,3)=0.5*(PQQP-P(N-2,5)**2/PQQP)*(-1)**(JT+1)   
        P(N-2,4)=SQRT(P(N-2,3)**2+P(N-2,5)**2)  
        P(N-1,3)=(PZ-PZI)*(-1)**(JT+1)  
        P(N-1,4)=ABS(P(N-1,3))  
        P(N,3)=PZI*(-1)**(JT+1)-P(N-2,3)    
        P(N,4)=SQRT(P(N,3)**2+P(N,5)**2)    
      ENDIF 
    
C...Documentation lines.    
      K(I+2,1)=21   
      IF(MINT(16+JT).EQ.0) K(I+2,2)=MINT(10+JT) 
      IF(MINT(16+JT).NE.0) K(I+2,2)=10*(MINT(10+JT)/10) 
      K(I+2,3)=I    
      P(I+2,3)=PZ*(-1)**(JT+1)  
      P(I+2,4)=PE   
      P(I+2,5)=SQRT(VINT(62+JT))    
  150 CONTINUE  
    
C...Rotate outgoing partons/particles using cos(theta). 
      CALL LUDBRB(MINT(83)+3,N,ACOS(VINT(23)),VINT(24),0D0,0D0,0D0) 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYFRAM(IFRAME) 
    
C...Performs transformations between different coordinate frames.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
    
      IF(IFRAME.LT.1.OR.IFRAME.GT.2) THEN   
        WRITE(MSTU(11),1000) IFRAME,MINT(6) 
        RETURN  
      ENDIF 
      IF(IFRAME.EQ.MINT(6)) RETURN  
    
      IF(MINT(6).EQ.1) THEN 
C...Transform from fixed target or user specified frame to  
C...CM-frame of incoming particles. 
        CALL LUROBO(0.,0.,-VINT(8),-VINT(9),-VINT(10))  
        CALL LUROBO(0.,-VINT(7),0.,0.,0.)   
        CALL LUROBO(-VINT(6),0.,0.,0.,0.)   
        MINT(6)=2   
    
      ELSE  
C...Transform from particle CM-frame to fixed target or user specified  
C...frame.  
        CALL LUROBO(VINT(6),VINT(7),VINT(8),VINT(9),VINT(10))   
        MINT(6)=1   
      ENDIF 
      MSTI(6)=MINT(6)   
    
 1000 FORMAT(1X,'Error: illegal values in subroutine PYFRAM.',1X,   
     &'No transformation performed.'/1X,'IFRAME =',1X,I5,'; MINT(6) =', 
     &1X,I5)    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYWIDT(KFLR,RMAS,WDTP,WDTE)    
    
C...Calculates full and partial widths of resonances.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      DIMENSION WDTP(0:40),WDTE(0:40,0:5)   
    
C...Some common constants.  
      KFLA=IABS(KFLR)   
      SQM=RMAS**2   
      AS=ULALPS(SQM)    
      AEM=PARU(101) 
      XW=PARU(102)  
      RADC=1.+AS/PARU(1)    
    
C...Reset width information.    
      DO 100 I=0,40 
      WDTP(I)=0.    
      DO 100 J=0,5  
  100 WDTE(I,J)=0.  
   
cms... Do a whole bunch of intialization...
      GGF=0.
      GZF=0.
      GZPF=0.
      ZZF=0.
      ZZPF=0.
      ZPZPF=0.

      IF(KFLA.EQ.21) THEN   
C...QCD:    
        DO 110 I=1,MDCY(21,3)   
        IDC=I+MDCY(21,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 110  
        IF(I.LE.8) THEN 
C...QCD -> q + qb   
          WDTP(I)=(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))   
          WID2=1.   
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
        ENDIF   
  110   CONTINUE    
    
      ELSEIF(KFLA.EQ.23) THEN   
C...Z0: 
        EI=KCHG(IABS(MINT(15)),1)/3.  
        AI=SIGN(1.,EI)    
        VI=AI-4.*EI*XW    
        SQMZ=PMAS(23,1)**2    
        GZMZ=PMAS(23,2)*PMAS(23,1)    
        GGI=EI**2 
        GZI=EI*VI/(8.*XW*(1.-XW))*SQM*(SQM-SQMZ)/ 
     &  ((SQM-SQMZ)**2+GZMZ**2)   
        ZZI=(VI**2+AI**2)/(16.*XW*(1.-XW))**2*SQM**2/ 
     &  ((SQM-SQMZ)**2+GZMZ**2)   
        IF(MINT(61).EQ.1) THEN
          IF(MSTP(43).EQ.1) THEN    
C...Only gamma* production included 
            GZI=0.  
            ZZI=0.  
          ELSEIF(MSTP(43).EQ.2) THEN    
C...Only Z0 production included 
            GGI=0.  
            GZI=0.  
          ENDIF 
        ELSEIF(MINT(61).EQ.2) THEN  
          VINT(111)=0.  
          VINT(112)=0.  
          VINT(114)=0.  
        ENDIF   
        DO 120 I=1,MDCY(23,3)   
        IDC=I+MDCY(23,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 120  
        IF(I.LE.8) THEN 
C...Z0 -> q + qb    
          EF=KCHG(I,1)/3.   
          AF=SIGN(1.,EF+0.1)    
          VF=AF-4.*EF*XW    
          IF(MINT(61).EQ.0) THEN    
            WDTP(I)=3.*(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*   
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
          ELSEIF(MINT(61).EQ.1) THEN    
            WDTP(I)=3.*((GGI*EF**2+GZI*EF*VF+ZZI*VF**2)*    
     &      (1.+2.*RM1)+ZZI*AF**2*(1.-4.*RM1))* 
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
          ELSEIF(MINT(61).EQ.2) THEN    
            GGF=3.*EF**2*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))*RADC   
            GZF=3.*EF*VF*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))*RADC   
            ZZF=3.*(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*   
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
          ENDIF 
          WID2=1.   
        ELSEIF(I.LE.16) THEN    
C...Z0 -> l+ + l-, nu + nub 
          EF=KCHG(I+2,1)/3. 
          AF=SIGN(1.,EF+0.1)    
          VF=AF-4.*EF*XW    
          WDTP(I)=(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*    
     &    SQRT(MAX(0.,1.-4.*RM1))   
          IF(MINT(61).EQ.0) THEN    
            WDTP(I)=(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*  
     &      SQRT(MAX(0.,1.-4.*RM1)) 
          ELSEIF(MINT(61).EQ.1) THEN    
            WDTP(I)=((GGI*EF**2+GZI*EF*VF+ZZI*VF**2)*   
     &      (1.+2.*RM1)+ZZI*AF**2*(1.-4.*RM1))* 
     &      SQRT(MAX(0.,1.-4.*RM1)) 
          ELSEIF(MINT(61).EQ.2) THEN    
            GGF=EF**2*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))   
            GZF=EF*VF*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))   
            ZZF=(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*  
     &      SQRT(MAX(0.,1.-4.*RM1)) 
          ENDIF 
          WID2=1.   
        ELSE    
C...Z0 -> H+ + H-   
          CF=2.*(1.-2.*XW)  
          IF(MINT(61).EQ.0) THEN    
            WDTP(I)=0.25*CF**2*(1.-4.*RM1)*SQRT(MAX(0.,1.-4.*RM1))  
          ELSEIF(MINT(61).EQ.1) THEN    
            WDTP(I)=0.25*(GGI+GZI*CF+ZZI*CF**2)*(1.-4.*RM1)*    
     &      SQRT(MAX(0.,1.-4.*RM1)) 
          ELSEIF(MINT(61).EQ.2) THEN    
            GGF=0.25*(1.-4.*RM1)*SQRT(MAX(0.,1.-4.*RM1))    
            GZF=0.25*CF*(1.-4.*RM1)*SQRT(MAX(0.,1.-4.*RM1)) 
            ZZF=0.25*CF**2*(1.-4.*RM1)*SQRT(MAX(0.,1.-4.*RM1))  
          ENDIF 
          WID2=WIDS(37,1)   
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
clin-4/2008 modified a la pythia6115.f to avoid undefined values (GGF,GZF,ZZF):
c          VINT(111)=VINT(111)+GGF*WID2  
c          VINT(112)=VINT(112)+GZF*WID2  
c          VINT(114)=VINT(114)+ZZF*WID2  
          IF(MINT(61).EQ.2) THEN    
             VINT(111)=VINT(111)+GGF*WID2  
             VINT(112)=VINT(112)+GZF*WID2  
             VINT(114)=VINT(114)+ZZF*WID2  
          ENDIF
clin-4/2008-end
        ENDIF   
  120   CONTINUE    
        IF(MSTP(43).EQ.1) THEN  
C...Only gamma* production included 
          VINT(112)=0.  
          VINT(114)=0.  
        ELSEIF(MSTP(43).EQ.2) THEN  
C...Only Z0 production included 
          VINT(111)=0.  
          VINT(112)=0.  
        ENDIF   
    
      ELSEIF(KFLA.EQ.24) THEN   
C...W+/-:   
        DO 130 I=1,MDCY(24,3)   
        IDC=I+MDCY(24,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 130  
        IF(I.LE.16) THEN    
C...W+/- -> q + qb' 
          WDTP(I)=3.*(2.-RM1-RM2-(RM1-RM2)**2)* 
     &    SQRT(MAX(0.,(1.-RM1-RM2)**2-4.*RM1*RM2))* 
     &    VCKM((I-1)/4+1,MOD(I-1,4)+1)*RADC 
          WID2=1.   
        ELSE    
C...W+/- -> l+/- + nu   
          WDTP(I)=(2.-RM1-RM2-(RM1-RM2)**2)*    
     &    SQRT(MAX(0.,(1.-RM1-RM2)**2-4.*RM1*RM2))  
          WID2=1.   
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
        ENDIF   
  130   CONTINUE    
    
      ELSEIF(KFLA.EQ.25) THEN   
C...H0: 
        DO 170 I=1,MDCY(25,3)   
        IDC=I+MDCY(25,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 170  
        IF(I.LE.8) THEN 
C...H0 -> q + qb    
          WDTP(I)=3.*RM1*(1.-4.*RM1)*SQRT(MAX(0.,1.-4.*RM1))*RADC   
          WID2=1.   
        ELSEIF(I.LE.12) THEN    
C...H0 -> l+ + l-   
          WDTP(I)=RM1*(1.-4.*RM1)*SQRT(MAX(0.,1.-4.*RM1))   
          WID2=1.   
        ELSEIF(I.EQ.13) THEN    
C...H0 -> g + g; quark loop contribution only   
          ETARE=0.  
          ETAIM=0.  
          DO 140 J=1,2*MSTP(1)  
          EPS=(2.*PMAS(J,1)/RMAS)**2    
          IF(EPS.LE.1.) THEN    
            IF(EPS.GT.1.E-4) THEN   
              ROOT=SQRT(1.-EPS) 
              RLN=LOG((1.+ROOT)/(1.-ROOT))  
            ELSE    
              RLN=LOG(4./EPS-2.)    
            ENDIF   
            PHIRE=0.25*(RLN**2-PARU(1)**2)  
            PHIIM=0.5*PARU(1)*RLN   
          ELSE  
            PHIRE=-(ASIN(1./SQRT(EPS)))**2  
            PHIIM=0.    
          ENDIF 
          ETARE=ETARE+0.5*EPS*(1.+(EPS-1.)*PHIRE)   
          ETAIM=ETAIM+0.5*EPS*(EPS-1.)*PHIIM    
  140     CONTINUE  
          ETA2=ETARE**2+ETAIM**2    
          WDTP(I)=(AS/PARU(1))**2*ETA2  
          WID2=1.   
        ELSEIF(I.EQ.14) THEN    
C...H0 -> gamma + gamma; quark, charged lepton and W loop contributions 
          ETARE=0.  
          ETAIM=0.  
          DO 150 J=1,3*MSTP(1)+1    
          IF(J.LE.2*MSTP(1)) THEN   
            EJ=KCHG(J,1)/3. 
            EPS=(2.*PMAS(J,1)/RMAS)**2  
          ELSEIF(J.LE.3*MSTP(1)) THEN   
            JL=2*(J-2*MSTP(1))-1    
            EJ=KCHG(10+JL,1)/3. 
            EPS=(2.*PMAS(10+JL,1)/RMAS)**2  
          ELSE  
            EPS=(2.*PMAS(24,1)/RMAS)**2 
          ENDIF 
          IF(EPS.LE.1.) THEN    
            IF(EPS.GT.1.E-4) THEN   
              ROOT=SQRT(1.-EPS) 
              RLN=LOG((1.+ROOT)/(1.-ROOT))  
            ELSE    
              RLN=LOG(4./EPS-2.)    
            ENDIF   
            PHIRE=0.25*(RLN**2-PARU(1)**2)  
            PHIIM=0.5*PARU(1)*RLN   
          ELSE  
            PHIRE=-(ASIN(1./SQRT(EPS)))**2  
            PHIIM=0.    
          ENDIF 
          IF(J.LE.2*MSTP(1)) THEN   
            ETARE=ETARE+0.5*3.*EJ**2*EPS*(1.+(EPS-1.)*PHIRE)    
            ETAIM=ETAIM+0.5*3.*EJ**2*EPS*(EPS-1.)*PHIIM 
          ELSEIF(J.LE.3*MSTP(1)) THEN   
            ETARE=ETARE+0.5*EJ**2*EPS*(1.+(EPS-1.)*PHIRE)   
            ETAIM=ETAIM+0.5*EJ**2*EPS*(EPS-1.)*PHIIM    
          ELSE  
            ETARE=ETARE-0.5-0.75*EPS*(1.+(EPS-2.)*PHIRE)    
            ETAIM=ETAIM+0.75*EPS*(EPS-2.)*PHIIM 
          ENDIF 
  150     CONTINUE  
          ETA2=ETARE**2+ETAIM**2    
          WDTP(I)=(AEM/PARU(1))**2*0.5*ETA2 
          WID2=1.   
        ELSEIF(I.EQ.15) THEN    
C...H0 -> gamma + Z0; quark, charged lepton and W loop contributions    
          ETARE=0.  
          ETAIM=0.  
          DO 160 J=1,3*MSTP(1)+1    
          IF(J.LE.2*MSTP(1)) THEN   
            EJ=KCHG(J,1)/3. 
            AJ=SIGN(1.,EJ+0.1)  
            VJ=AJ-4.*EJ*XW  
            EPS=(2.*PMAS(J,1)/RMAS)**2  
            EPSP=(2.*PMAS(J,1)/PMAS(23,1))**2   
          ELSEIF(J.LE.3*MSTP(1)) THEN   
            JL=2*(J-2*MSTP(1))-1    
            EJ=KCHG(10+JL,1)/3. 
            AJ=SIGN(1.,EJ+0.1)  
            VJ=AJ-4.*EJ*XW  
            EPS=(2.*PMAS(10+JL,1)/RMAS)**2  
            EPSP=(2.*PMAS(10+JL,1)/PMAS(23,1))**2   
          ELSE  
            EPS=(2.*PMAS(24,1)/RMAS)**2 
            EPSP=(2.*PMAS(24,1)/PMAS(23,1))**2  
          ENDIF 
          IF(EPS.LE.1.) THEN    
            ROOT=SQRT(1.-EPS)   
            IF(EPS.GT.1.E-4) THEN   
              RLN=LOG((1.+ROOT)/(1.-ROOT))  
            ELSE    
              RLN=LOG(4./EPS-2.)    
            ENDIF   
            PHIRE=0.25*(RLN**2-PARU(1)**2)  
            PHIIM=0.5*PARU(1)*RLN   
            PSIRE=-(1.+0.5*ROOT*RLN)    
            PSIIM=0.5*PARU(1)*ROOT  
          ELSE  
            PHIRE=-(ASIN(1./SQRT(EPS)))**2  
            PHIIM=0.    
            PSIRE=-(1.+SQRT(EPS-1.)*ASIN(1./SQRT(EPS))) 
            PSIIM=0.    
          ENDIF 
          IF(EPSP.LE.1.) THEN   
            ROOT=SQRT(1.-EPSP)  
            IF(EPSP.GT.1.E-4) THEN  
              RLN=LOG((1.+ROOT)/(1.-ROOT))  
            ELSE    
              RLN=LOG(4./EPSP-2.)   
            ENDIF   
            PHIREP=0.25*(RLN**2-PARU(1)**2) 
            PHIIMP=0.5*PARU(1)*RLN  
            PSIREP=-(1.+0.5*ROOT*RLN)   
            PSIIMP=0.5*PARU(1)*ROOT 
          ELSE  
            PHIREP=-(ASIN(1./SQRT(EPSP)))**2    
            PHIIMP=0.   
            PSIREP=-(1.+SQRT(EPSP-1.)*ASIN(1./SQRT(EPSP)))  
            PSIIMP=0.   
          ENDIF 
          FXYRE=EPS*EPSP/(8.*(EPS-EPSP))*(1.-EPS*EPSP/(EPS-EPSP)*(PHIRE-    
     &    PHIREP)+2.*EPS/(EPS-EPSP)*(PSIRE-PSIREP)) 
          FXYIM=EPS*EPSP/(8.*(EPS-EPSP))*(-EPS*EPSP/(EPS-EPSP)*(PHIIM-  
     &    PHIIMP)+2.*EPS/(EPS-EPSP)*(PSIIM-PSIIMP)) 
          F1RE=EPS*EPSP/(2.*(EPS-EPSP))*(PHIRE-PHIREP)  
          F1IM=EPS*EPSP/(2.*(EPS-EPSP))*(PHIIM-PHIIMP)  
          IF(J.LE.2*MSTP(1)) THEN   
            ETARE=ETARE-3.*EJ*VJ*(FXYRE-0.25*F1RE)  
            ETAIM=ETAIM-3.*EJ*VJ*(FXYIM-0.25*F1IM)  
          ELSEIF(J.LE.3*MSTP(1)) THEN   
            ETARE=ETARE-EJ*VJ*(FXYRE-0.25*F1RE) 
            ETAIM=ETAIM-EJ*VJ*(FXYIM-0.25*F1IM) 
          ELSE  
            ETARE=ETARE-SQRT(1.-XW)*(((1.+2./EPS)*XW/SQRT(1.-XW)-   
     &      (5.+2./EPS))*FXYRE+(3.-XW/SQRT(1.-XW))*F1RE)    
            ETAIM=ETAIM-SQRT(1.-XW)*(((1.+2./EPS)*XW/SQRT(1.-XW)-   
     &      (5.+2./EPS))*FXYIM+(3.-XW/SQRT(1.-XW))*F1IM)    
          ENDIF 
  160     CONTINUE  
          ETA2=ETARE**2+ETAIM**2    
          WDTP(I)=(AEM/PARU(1))**2*(1.-(PMAS(23,1)/RMAS)**2)**3/XW*ETA2 
          WID2=WIDS(23,2)   
        ELSE    
C...H0 -> Z0 + Z0, W+ + W-  
          WDTP(I)=(1.-4.*RM1+12.*RM1**2)*SQRT(MAX(0.,1.-4.*RM1))/   
     &    (2.*(18-I))   
          WID2=WIDS(7+I,1)  
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
        ENDIF   
  170   CONTINUE    
    
      ELSEIF(KFLA.EQ.32) THEN   
C...Z'0:    
        EI=KCHG(IABS(MINT(15)),1)/3.  
        AI=SIGN(1.,EI)    
        VI=AI-4.*EI*XW    
        SQMZ=PMAS(23,1)**2    
        GZMZ=PMAS(23,2)*PMAS(23,1)    
        API=SIGN(1.,EI)   
        VPI=API-4.*EI*XW  
        SQMZP=PMAS(32,1)**2   
        GZPMZP=PMAS(32,2)*PMAS(32,1)  
        GGI=EI**2 
        GZI=EI*VI/(8.*XW*(1.-XW))*SQM*(SQM-SQMZ)/ 
     &  ((SQM-SQMZ)**2+GZMZ**2)   
        GZPI=EI*VPI/(8.*XW*(1.-XW))*SQM*(SQM-SQMZP)/  
     &  ((SQM-SQMZP)**2+GZPMZP**2)    
        ZZI=(VI**2+AI**2)/(16.*XW*(1.-XW))**2*SQM**2/ 
     &  ((SQM-SQMZ)**2+GZMZ**2)   
        ZZPI=2.*(VI*VPI+AI*API)/(16.*XW*(1.-XW))**2*  
     &  SQM**2*((SQM-SQMZ)*(SQM-SQMZP)+GZMZ*GZPMZP)/  
     &  (((SQM-SQMZ)**2+GZMZ**2)*((SQM-SQMZP)**2+GZPMZP**2))  
        ZPZPI=(VPI**2+API**2)/(16.*XW*(1.-XW))**2*SQM**2/ 
     &  ((SQM-SQMZP)**2+GZPMZP**2)    
        IF(MINT(61).EQ.1) THEN
          IF(MSTP(44).EQ.1) THEN    
C...Only gamma* production included 
            GZI=0.  
            GZPI=0. 
            ZZI=0.  
            ZZPI=0. 
            ZPZPI=0.    
          ELSEIF(MSTP(44).EQ.2) THEN    
C...Only Z0 production included 
            GGI=0.  
            GZI=0.  
            GZPI=0. 
            ZZPI=0. 
            ZPZPI=0.    
          ELSEIF(MSTP(44).EQ.3) THEN    
C...Only Z'0 production included    
            GGI=0. 
            GZI=0.  
            GZPI=0. 
            ZZI=0.  
            ZZPI=0. 
          ELSEIF(MSTP(44).EQ.4) THEN    
C...Only gamma*/Z0 production included  
            GZPI=0. 
            ZZPI=0. 
            ZPZPI=0.    
          ELSEIF(MSTP(44).EQ.5) THEN    
C...Only gamma*/Z'0 production included 
            GZI=0.  
            ZZI=0.  
            ZZPI=0. 
          ELSEIF(MSTP(44).EQ.6) THEN    
C...Only Z0/Z'0 production included 
            GGI=0.  
            GZI=0.  
            GZPI=0. 
          ENDIF 
        ELSEIF(MINT(61).EQ.2) THEN  
          VINT(111)=0.  
          VINT(112)=0.  
          VINT(113)=0.  
          VINT(114)=0.  
          VINT(115)=0.  
          VINT(116)=0.  
        ENDIF   
        DO 180 I=1,MDCY(32,3)   
        IDC=I+MDCY(32,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 180  
        IF(I.LE.8) THEN 
C...Z'0 -> q + qb   
          EF=KCHG(I,1)/3.   
          AF=SIGN(1.,EF+0.1)    
          VF=AF-4.*EF*XW    
          APF=SIGN(1.,EF+0.1)   
          VPF=APF-4.*EF*XW  
          IF(MINT(61).EQ.0) THEN    
            WDTP(I)=3.*(VPF**2*(1.+2.*RM1)+APF**2*(1.-4.*RM1))* 
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
          ELSEIF(MINT(61).EQ.1) THEN    
            WDTP(I)=3.*((GGI*EF**2+GZI*EF*VF+GZPI*EF*VPF+ZZI*VF**2+ 
     &      ZZPI*VF*VPF+ZPZPI*VPF**2)*(1.+2.*RM1)+(ZZI*AF**2+   
     &      ZZPI*AF*APF+ZPZPI*APF**2)*(1.-4.*RM1))* 
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
          ELSEIF(MINT(61).EQ.2) THEN    
            GGF=3.*EF**2*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))*RADC   
            GZF=3.*EF*VF*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))*RADC   
            GZPF=3.*EF*VPF*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))*RADC 
            ZZF=3.*(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*   
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
            ZZPF=3.*(VF*VPF*(1.+2.*RM1)+AF*APF*(1.-4.*RM1))*    
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
            ZPZPF=3.*(VPF**2*(1.+2.*RM1)+APF**2*(1.-4.*RM1))*   
     &      SQRT(MAX(0.,1.-4.*RM1))*RADC    
          ENDIF
          WID2=1.   
        ELSE    
C...Z'0 -> l+ + l-, nu + nub    
          EF=KCHG(I+2,1)/3. 
          AF=SIGN(1.,EF+0.1)    
          VF=AF-4.*EF*XW    
clin-4/2008 modified above a la pythia6115.f to avoid undefined variable API:
c          APF=SIGN(1.,EF+0.1)   
c          VPF=API-4.*EF*XW  
          IF(I.LE.10) THEN
             VPF=PARU(127-2*MOD(I,2))
             APF=PARU(128-2*MOD(I,2))
          ELSEIF(I.LE.12) THEN
             VPF=PARJ(186-2*MOD(I,2))
             APF=PARJ(187-2*MOD(I,2))
          ELSE
             VPF=PARJ(194-2*MOD(I,2))
             APF=PARJ(195-2*MOD(I,2))
          ENDIF
clin-4/2008-end
          IF(MINT(61).EQ.0) THEN    
            WDTP(I)=(VPF**2*(1.+2.*RM1)+APF**2*(1.-4.*RM1))*    
     &      SQRT(MAX(0.,1.-4.*RM1)) 
          ELSEIF(MINT(61).EQ.1) THEN    
            WDTP(I)=((GGI*EF**2+GZI*EF*VF+GZPI*EF*VPF+ZZI*VF**2+    
     &      ZZPI*VF*VPF+ZPZPI*VPF**2)*(1.+2.*RM1)+(ZZI*AF**2+   
     &      ZZPI*AF*APF+ZPZPI*APF**2)*(1.-4.*RM1))* 
     &      SQRT(MAX(0.,1.-4.*RM1)) 
          ELSEIF(MINT(61).EQ.2) THEN    
            GGF=EF**2*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))   
            GZF=EF*VF*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1))   
            GZPF=EF*VPF*(1.+2.*RM1)*SQRT(MAX(0.,1.-4.*RM1)) 
            ZZF=(VF**2*(1.+2.*RM1)+AF**2*(1.-4.*RM1))*  
     &      SQRT(MAX(0.,1.-4.*RM1)) 
            ZZPF=(VF*VPF*(1.+2.*RM1)+AF*APF*(1.-4.*RM1))*   
     &      SQRT(MAX(0.,1.-4.*RM1)) 
            ZPZPF=(VPF**2*(1.+2.*RM1)+APF**2*(1.-4.*RM1))*  
     &      SQRT(MAX(0.,1.-4.*RM1))
          ENDIF 
          WID2=1.   
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
clin-4/2008:
c          VINT(111)=VINT(111)+GGF   
c          VINT(112)=VINT(112)+GZF   
c          VINT(113)=VINT(113)+GZPF  
c          VINT(114)=VINT(114)+ZZF   
c          VINT(115)=VINT(115)+ZZPF  
c          VINT(116)=VINT(116)+ZPZPF 
          IF(MINT(61).EQ.2) THEN    
             VINT(111)=VINT(111)+GGF   
             VINT(112)=VINT(112)+GZF   
             VINT(113)=VINT(113)+GZPF  
             VINT(114)=VINT(114)+ZZF   
             VINT(115)=VINT(115)+ZZPF  
             VINT(116)=VINT(116)+ZPZPF 
          ENDIF
clin-4/2008-end
        ENDIF   
  180   CONTINUE    
        IF(MSTP(44).EQ.1) THEN  
C...Only gamma* production included 
          VINT(112)=0.  
          VINT(113)=0.  
          VINT(114)=0.  
          VINT(115)=0.  
          VINT(116)=0.  
        ELSEIF(MSTP(44).EQ.2) THEN  
C...Only Z0 production included 
          VINT(111)=0.  
          VINT(112)=0.  
          VINT(113)=0.  
          VINT(115)=0.  
          VINT(116)=0.  
        ELSEIF(MSTP(44).EQ.3) THEN  
C...Only Z'0 production included    
          VINT(111)=0.  
          VINT(112)=0.  
          VINT(113)=0.  
          VINT(114)=0.  
          VINT(115)=0.  
        ELSEIF(MSTP(44).EQ.4) THEN  
C...Only gamma*/Z0 production included  
          VINT(113)=0.  
          VINT(115)=0.  
          VINT(116)=0.  
        ELSEIF(MSTP(44).EQ.5) THEN  
C...Only gamma*/Z'0 production included 
          VINT(112)=0.  
          VINT(114)=0.  
          VINT(115)=0.  
        ELSEIF(MSTP(44).EQ.6) THEN  
C...Only Z0/Z'0 production included 
          VINT(111)=0.  
          VINT(112)=0.  
          VINT(113)=0.  
        ENDIF   
    
      ELSEIF(KFLA.EQ.37) THEN   
C...H+/-:   
        DO 190 I=1,MDCY(37,3)   
        IDC=I+MDCY(37,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 190  
        IF(I.LE.4) THEN 
C...H+/- -> q + qb' 
          WDTP(I)=3.*((RM1*PARU(121)+RM2/PARU(121))*    
     &    (1.-RM1-RM2)-4.*RM1*RM2)* 
     &    SQRT(MAX(0.,(1.-RM1-RM2)**2-4.*RM1*RM2))*RADC 
          WID2=1.   
        ELSE    
C...H+/- -> l+/- + nu   
          WDTP(I)=((RM1*PARU(121)+RM2/PARU(121))*   
     &    (1.-RM1-RM2)-4.*RM1*RM2)* 
     &    SQRT(MAX(0.,(1.-RM1-RM2)**2-4.*RM1*RM2))  
          WID2=1.   
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
        ENDIF   
  190   CONTINUE    
    
      ELSEIF(KFLA.EQ.40) THEN   
C...R:  
        DO 200 I=1,MDCY(40,3)   
        IDC=I+MDCY(40,2)-1  
        RM1=(PMAS(IABS(KFDP(IDC,1)),1)/RMAS)**2 
        RM2=(PMAS(IABS(KFDP(IDC,2)),1)/RMAS)**2 
        IF(SQRT(RM1)+SQRT(RM2).GT.1..OR.MDME(IDC,1).LT.0) GOTO 200  
        IF(I.LE.4) THEN 
C...R -> q + qb'    
          WDTP(I)=3.*RADC   
          WID2=1.   
        ELSE    
C...R -> l+ + l'-   
          WDTP(I)=1.    
          WID2=1.   
        ENDIF   
        WDTP(0)=WDTP(0)+WDTP(I) 
        IF(MDME(IDC,1).GT.0) THEN   
          WDTE(I,MDME(IDC,1))=WDTP(I)*WID2  
          WDTE(0,MDME(IDC,1))=WDTE(0,MDME(IDC,1))+WDTE(I,MDME(IDC,1))   
          WDTE(I,0)=WDTE(I,MDME(IDC,1)) 
          WDTE(0,0)=WDTE(0,0)+WDTE(I,0) 
        ENDIF   
  200   CONTINUE    
    
      ENDIF 
      MINT(61)=0    
    
      RETURN    
      END   
    
C***********************************************************************    
    
      SUBROUTINE PYKLIM(ILIM)   
    
C...Checks generated variables against pre-set kinematical limits;  
C...also calculates limits on variables used in generation. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
    
C...Common kinematical expressions. 
      ISUB=MINT(1)  
      IF(ISUB.EQ.96) GOTO 110   
      SQM3=VINT(63) 
      SQM4=VINT(64) 
      IF(ILIM.NE.1) THEN    
        TAU=VINT(21)    
        RM3=SQM3/(TAU*VINT(2))  
        RM4=SQM4/(TAU*VINT(2))  
        BE34=SQRT((1.-RM3-RM4)**2-4.*RM3*RM4)   
      ENDIF 
      PTHMIN=CKIN(3)    
      IF(MIN(SQM3,SQM4).LT.CKIN(6)**2) PTHMIN=MAX(CKIN(3),CKIN(5))  
      IF(ILIM.EQ.0) THEN    
C...Check generated values of tau, y*, cos(theta-hat), and tau' against 
C...pre-set kinematical limits. 
        YST=VINT(22)    
        CTH=VINT(23)    
        TAUP=VINT(26)   
        IF(ISET(ISUB).LE.2) THEN    
          X1=SQRT(TAU)*EXP(YST) 
          X2=SQRT(TAU)*EXP(-YST)    
        ELSE    
          X1=SQRT(TAUP)*EXP(YST)    
          X2=SQRT(TAUP)*EXP(-YST)   
        ENDIF   
        XF=X1-X2    
        IF(TAU*VINT(2).LT.CKIN(1)**2) MINT(51)=1    
        IF(CKIN(2).GE.0..AND.TAU*VINT(2).GT.CKIN(2)**2) MINT(51)=1  
        IF(X1.LT.CKIN(21).OR.X1.GT.CKIN(22)) MINT(51)=1 
        IF(X2.LT.CKIN(23).OR.X2.GT.CKIN(24)) MINT(51)=1 
        IF(XF.LT.CKIN(25).OR.XF.GT.CKIN(26)) MINT(51)=1 
        IF(YST.LT.CKIN(7).OR.YST.GT.CKIN(8)) MINT(51)=1 
        IF(ISET(ISUB).EQ.2.OR.ISET(ISUB).EQ.4) THEN 
          PTH=0.5*BE34*SQRT(TAU*VINT(2)*(1.-CTH**2))    
          Y3=YST+0.5*LOG((1.+RM3-RM4+BE34*CTH)/(1.+RM3-RM4-BE34*CTH))   
          Y4=YST+0.5*LOG((1.+RM4-RM3-BE34*CTH)/(1.+RM4-RM3+BE34*CTH))   
          YLARGE=MAX(Y3,Y4) 
          YSMALL=MIN(Y3,Y4) 
          ETALAR=10.    
          ETASMA=-10.   
          STH=SQRT(1.-CTH**2)   
          IF(STH.LT.1.E-6) GOTO 100 
          EXPET3=((1.+RM3-RM4)*SINH(YST)+BE34*COSH(YST)*CTH+    
     &    SQRT(((1.+RM3-RM4)*COSH(YST)+BE34*SINH(YST)*CTH)**2-4.*RM3))/ 
     &    (BE34*STH)    
          EXPET4=((1.-RM3+RM4)*SINH(YST)-BE34*COSH(YST)*CTH+    
     &    SQRT(((1.-RM3+RM4)*COSH(YST)-BE34*SINH(YST)*CTH)**2-4.*RM4))/ 
     &    (BE34*STH)    
          ETA3=LOG(MIN(1.E10,MAX(1.E-10,EXPET3)))   
          ETA4=LOG(MIN(1.E10,MAX(1.E-10,EXPET4)))   
          ETALAR=MAX(ETA3,ETA4) 
          ETASMA=MIN(ETA3,ETA4) 
  100     CTS3=((1.+RM3-RM4)*SINH(YST)+BE34*COSH(YST)*CTH)/ 
     &    SQRT(((1.+RM3-RM4)*COSH(YST)+BE34*SINH(YST)*CTH)**2-4.*RM3)   
          CTS4=((1.-RM3+RM4)*SINH(YST)-BE34*COSH(YST)*CTH)/ 
     &    SQRT(((1.-RM3+RM4)*COSH(YST)-BE34*SINH(YST)*CTH)**2-4.*RM4)   
          CTSLAR=MAX(CTS3,CTS4) 
          CTSSMA=MIN(CTS3,CTS4) 
          IF(PTH.LT.PTHMIN) MINT(51)=1  
          IF(CKIN(4).GE.0..AND.PTH.GT.CKIN(4)) MINT(51)=1   
          IF(YLARGE.LT.CKIN(9).OR.YLARGE.GT.CKIN(10)) MINT(51)=1    
          IF(YSMALL.LT.CKIN(11).OR.YSMALL.GT.CKIN(12)) MINT(51)=1   
          IF(ETALAR.LT.CKIN(13).OR.ETALAR.GT.CKIN(14)) MINT(51)=1   
          IF(ETASMA.LT.CKIN(15).OR.ETASMA.GT.CKIN(16)) MINT(51)=1   
          IF(CTSLAR.LT.CKIN(17).OR.CTSLAR.GT.CKIN(18)) MINT(51)=1   
          IF(CTSSMA.LT.CKIN(19).OR.CTSSMA.GT.CKIN(20)) MINT(51)=1   
          IF(CTH.LT.CKIN(27).OR.CTH.GT.CKIN(28)) MINT(51)=1 
        ENDIF   
        IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) THEN 
          IF(TAUP*VINT(2).LT.CKIN(31)**2) MINT(51)=1    
          IF(CKIN(32).GE.0..AND.TAUP*VINT(2).GT.CKIN(32)**2) MINT(51)=1 
        ENDIF   
    
      ELSEIF(ILIM.EQ.1) THEN    
C...Calculate limits on tau 
C...0) due to definition    
        TAUMN0=0.   
        TAUMX0=1.   
C...1) due to limits on subsystem mass  
        TAUMN1=CKIN(1)**2/VINT(2)   
        TAUMX1=1.   
        IF(CKIN(2).GE.0.) TAUMX1=CKIN(2)**2/VINT(2) 
C...2) due to limits on pT-hat (and non-overlapping rapidity intervals) 
        TM3=SQRT(SQM3+PTHMIN**2)    
        TM4=SQRT(SQM4+PTHMIN**2)    
        YDCOSH=1.   
        IF(CKIN(9).GT.CKIN(12)) YDCOSH=COSH(CKIN(9)-CKIN(12))   
        TAUMN2=(TM3**2+2.*TM3*TM4*YDCOSH+TM4**2)/VINT(2)    
        TAUMX2=1.   
C...3) due to limits on pT-hat and cos(theta-hat)   
        CTH2MN=MIN(CKIN(27)**2,CKIN(28)**2) 
        CTH2MX=MAX(CKIN(27)**2,CKIN(28)**2) 
        TAUMN3=0.   
        IF(CKIN(27)*CKIN(28).GT.0.) TAUMN3= 
     &  (SQRT(SQM3+PTHMIN**2/(1.-CTH2MN))+  
     &  SQRT(SQM4+PTHMIN**2/(1.-CTH2MN)))**2/VINT(2)    
        TAUMX3=1.   
        IF(CKIN(4).GE.0..AND.CTH2MX.LT.1.) TAUMX3=  
     &  (SQRT(SQM3+CKIN(4)**2/(1.-CTH2MX))+ 
     &  SQRT(SQM4+CKIN(4)**2/(1.-CTH2MX)))**2/VINT(2)   
C...4) due to limits on x1 and x2   
        TAUMN4=CKIN(21)*CKIN(23)    
        TAUMX4=CKIN(22)*CKIN(24)    
C...5) due to limits on xF  
        TAUMN5=0.   
        TAUMX5=MAX(1.-CKIN(25),1.+CKIN(26)) 
        VINT(11)=MAX(TAUMN0,TAUMN1,TAUMN2,TAUMN3,TAUMN4,TAUMN5) 
        VINT(31)=MIN(TAUMX0,TAUMX1,TAUMX2,TAUMX3,TAUMX4,TAUMX5) 
        IF(MINT(43).EQ.1.AND.(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.2)) THEN 
          VINT(11)=0.99999  
          VINT(31)=1.00001  
        ENDIF   
        IF(VINT(31).LE.VINT(11)) MINT(51)=1 
    
      ELSEIF(ILIM.EQ.2) THEN    
C...Calculate limits on y*  
        IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) TAU=VINT(26) 
        TAURT=SQRT(TAU) 
C...0) due to kinematics    
        YSTMN0=LOG(TAURT)   
        YSTMX0=-YSTMN0  
C...1) due to explicit limits   
        YSTMN1=CKIN(7)  
        YSTMX1=CKIN(8)  
C...2) due to limits on x1  
        YSTMN2=LOG(MAX(TAU,CKIN(21))/TAURT) 
        YSTMX2=LOG(MAX(TAU,CKIN(22))/TAURT) 
C...3) due to limits on x2  
        YSTMN3=-LOG(MAX(TAU,CKIN(24))/TAURT)    
        YSTMX3=-LOG(MAX(TAU,CKIN(23))/TAURT)    
C...4) due to limits on xF  
        YEPMN4=0.5*ABS(CKIN(25))/TAURT  
        YSTMN4=SIGN(LOG(SQRT(1.+YEPMN4**2)+YEPMN4),CKIN(25))    
        YEPMX4=0.5*ABS(CKIN(26))/TAURT  
        YSTMX4=SIGN(LOG(SQRT(1.+YEPMX4**2)+YEPMX4),CKIN(26))    
C...5) due to simultaneous limits on y-large and y-small    
        YEPSMN=(RM3-RM4)*SINH(CKIN(9)-CKIN(11)) 
        YEPSMX=(RM3-RM4)*SINH(CKIN(10)-CKIN(12))    
        YDIFMN=ABS(LOG(SQRT(1.+YEPSMN**2)-YEPSMN))  
        YDIFMX=ABS(LOG(SQRT(1.+YEPSMX**2)-YEPSMX))  
        YSTMN5=0.5*(CKIN(9)+CKIN(11)-YDIFMN)    
        YSTMX5=0.5*(CKIN(10)+CKIN(12)+YDIFMX)   
C...6) due to simultaneous limits on cos(theta-hat) and y-large or  
C...   y-small  
        CTHLIM=SQRT(1.-4.*PTHMIN**2/(BE34*TAU*VINT(2))) 
        RZMN=BE34*MAX(CKIN(27),-CTHLIM) 
        RZMX=BE34*MIN(CKIN(28),CTHLIM)  
        YEX3MX=(1.+RM3-RM4+RZMX)/MAX(1E-10,1.+RM3-RM4-RZMX) 
        YEX4MX=(1.+RM4-RM3-RZMN)/MAX(1E-10,1.+RM4-RM3+RZMN) 
        YEX3MN=MAX(1E-10,1.+RM3-RM4+RZMN)/(1.+RM3-RM4-RZMN) 
        YEX4MN=MAX(1E-10,1.+RM4-RM3-RZMX)/(1.+RM4-RM3+RZMX) 
        YSTMN6=CKIN(9)-0.5*LOG(MAX(YEX3MX,YEX4MX))  
        YSTMX6=CKIN(12)-0.5*LOG(MIN(YEX3MN,YEX4MN)) 
        VINT(12)=MAX(YSTMN0,YSTMN1,YSTMN2,YSTMN3,YSTMN4,YSTMN5,YSTMN6)  
        VINT(32)=MIN(YSTMX0,YSTMX1,YSTMX2,YSTMX3,YSTMX4,YSTMX5,YSTMX6)  
        IF(MINT(43).EQ.1) THEN  
          VINT(12)=-0.00001 
          VINT(32)=0.00001  
        ELSEIF(MINT(43).EQ.2) THEN  
          VINT(12)=0.99999*YSTMX0   
          VINT(32)=1.00001*YSTMX0   
        ELSEIF(MINT(43).EQ.3) THEN  
          VINT(12)=-1.00001*YSTMX0  
          VINT(32)=-0.99999*YSTMX0  
        ENDIF   
        IF(VINT(32).LE.VINT(12)) MINT(51)=1 
    
      ELSEIF(ILIM.EQ.3) THEN    
C...Calculate limits on cos(theta-hat)  
        YST=VINT(22)    
C...0) due to definition    
        CTNMN0=-1.  
        CTNMX0=0.   
        CTPMN0=0.   
        CTPMX0=1.   
C...1) due to explicit limits   
        CTNMN1=MIN(0.,CKIN(27)) 
        CTNMX1=MIN(0.,CKIN(28)) 
        CTPMN1=MAX(0.,CKIN(27)) 
        CTPMX1=MAX(0.,CKIN(28)) 
C...2) due to limits on pT-hat  
        CTNMN2=-SQRT(1.-4.*PTHMIN**2/(BE34**2*TAU*VINT(2))) 
        CTPMX2=-CTNMN2  
        CTNMX2=0.   
        CTPMN2=0.   
        IF(CKIN(4).GE.0.) THEN  
          CTNMX2=-SQRT(MAX(0.,1.-4.*CKIN(4)**2/(BE34**2*TAU*VINT(2))))  
          CTPMN2=-CTNMX2    
        ENDIF   
C...3) due to limits on y-large and y-small 
        CTNMN3=MIN(0.,MAX((1.+RM3-RM4)/BE34*TANH(CKIN(11)-YST), 
     &  -(1.-RM3+RM4)/BE34*TANH(CKIN(10)-YST))) 
        CTNMX3=MIN(0.,(1.+RM3-RM4)/BE34*TANH(CKIN(12)-YST), 
     &  -(1.-RM3+RM4)/BE34*TANH(CKIN(9)-YST))   
        CTPMN3=MAX(0.,(1.+RM3-RM4)/BE34*TANH(CKIN(9)-YST),  
     &  -(1.-RM3+RM4)/BE34*TANH(CKIN(12)-YST))  
        CTPMX3=MAX(0.,MIN((1.+RM3-RM4)/BE34*TANH(CKIN(10)-YST), 
     &  -(1.-RM3+RM4)/BE34*TANH(CKIN(11)-YST))) 
        VINT(13)=MAX(CTNMN0,CTNMN1,CTNMN2,CTNMN3)   
        VINT(33)=MIN(CTNMX0,CTNMX1,CTNMX2,CTNMX3)   
        VINT(14)=MAX(CTPMN0,CTPMN1,CTPMN2,CTPMN3)   
        VINT(34)=MIN(CTPMX0,CTPMX1,CTPMX2,CTPMX3)   
        IF(VINT(33).LE.VINT(13).AND.VINT(34).LE.VINT(14)) MINT(51)=1    
    
      ELSEIF(ILIM.EQ.4) THEN    
C...Calculate limits on tau'    
C...0) due to kinematics    
cms.. reinitializing tau due to compiler warning        
        TAU=VINT(21)
        TAPMN0=TAU  
        TAPMX0=1.   
C...1) due to explicit limits   
        TAPMN1=CKIN(31)**2/VINT(2)  
        TAPMX1=1.   
        IF(CKIN(32).GE.0.) TAPMX1=CKIN(32)**2/VINT(2)   
        VINT(16)=MAX(TAPMN0,TAPMN1) 
        VINT(36)=MIN(TAPMX0,TAPMX1) 
        IF(MINT(43).EQ.1) THEN  
          VINT(16)=0.99999  
          VINT(36)=1.00001  
        ENDIF   
        IF(VINT(36).LE.VINT(16)) MINT(51)=1 
    
      ENDIF 
      RETURN    
    
C...Special case for low-pT and multiple interactions:  
C...effective kinematical limits for tau, y*, cos(theta-hat).   
  110 IF(ILIM.EQ.0) THEN    
      ELSEIF(ILIM.EQ.1) THEN    
        IF(MSTP(82).LE.1) VINT(11)=4.*PARP(81)**2/VINT(2)   
        IF(MSTP(82).GE.2) VINT(11)=PARP(82)**2/VINT(2)  
        VINT(31)=1. 
      ELSEIF(ILIM.EQ.2) THEN    
        VINT(12)=0.5*LOG(VINT(21))  
        VINT(32)=-VINT(12)  
      ELSEIF(ILIM.EQ.3) THEN    
        IF(MSTP(82).LE.1) ST2EFF=4.*PARP(81)**2/(VINT(21)*VINT(2))  
        IF(MSTP(82).GE.2) ST2EFF=0.01*PARP(82)**2/(VINT(21)*VINT(2))    
        VINT(13)=-SQRT(MAX(0.,1.-ST2EFF))   
        VINT(33)=0. 
        VINT(14)=0. 
        VINT(34)=-VINT(13)  
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYKMAP(IVAR,MVAR,VVAR) 
    
C...Maps a uniform distribution into a distribution of a kinematical    
C...variable according to one of the possibilities allowed. It is   
C...assumed that kinematical limits have been set by a PYKLIM call. 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
    
C...Convert VVAR to tau variable.   
      ISUB=MINT(1)  
      IF(IVAR.EQ.1) THEN    
        TAUMIN=VINT(11) 
        TAUMAX=VINT(31) 
        IF(MVAR.EQ.3.OR.MVAR.EQ.4) THEN 
          TAURE=VINT(73)    
          GAMRE=VINT(74)    
        ELSEIF(MVAR.EQ.5.OR.MVAR.EQ.6) THEN 
          TAURE=VINT(75)    
          GAMRE=VINT(76)    
        ELSE
cms..   needed re-initialization to avoid compiler warning
          TAURE=VINT(75)
          GAMRE=VINT(76)
        ENDIF   
        IF(MINT(43).EQ.1.AND.(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.2)) THEN 
          TAU=1.    
        ELSEIF(MVAR.EQ.1) THEN  
          TAU=TAUMIN*(TAUMAX/TAUMIN)**VVAR  
        ELSEIF(MVAR.EQ.2) THEN  
          TAU=TAUMAX*TAUMIN/(TAUMIN+(TAUMAX-TAUMIN)*VVAR)   
        ELSEIF(MVAR.EQ.3.OR.MVAR.EQ.5) THEN 
          RATGEN=(TAURE+TAUMAX)/(TAURE+TAUMIN)*TAUMIN/TAUMAX    
          TAU=TAURE*TAUMIN/((TAURE+TAUMIN)*RATGEN**VVAR-TAUMIN) 
        ELSE    
          AUPP=ATAN((TAUMAX-TAURE)/GAMRE)   
          ALOW=ATAN((TAUMIN-TAURE)/GAMRE)   
          TAU=TAURE+GAMRE*TAN(ALOW+(AUPP-ALOW)*VVAR)    
        ENDIF   
        VINT(21)=MIN(TAUMAX,MAX(TAUMIN,TAU))    

C...Convert VVAR to y* variable.    
      ELSEIF(IVAR.EQ.2) THEN    
        YSTMIN=VINT(12) 
        YSTMAX=VINT(32) 
        IF(MINT(43).EQ.1) THEN  
          YST=0.    
        ELSEIF(MINT(43).EQ.2) THEN  
          IF(ISET(ISUB).LE.2) YST=-0.5*LOG(VINT(21))    
          IF(ISET(ISUB).GE.3) YST=-0.5*LOG(VINT(26))    
        ELSEIF(MINT(43).EQ.3) THEN  
          IF(ISET(ISUB).LE.2) YST=0.5*LOG(VINT(21)) 
          IF(ISET(ISUB).GE.3) YST=0.5*LOG(VINT(26)) 
        ELSEIF(MVAR.EQ.1) THEN  
          YST=YSTMIN+(YSTMAX-YSTMIN)*SQRT(VVAR) 
        ELSEIF(MVAR.EQ.2) THEN  
          YST=YSTMAX-(YSTMAX-YSTMIN)*SQRT(1.-VVAR)  
        ELSE    
          AUPP=ATAN(EXP(YSTMAX))    
          ALOW=ATAN(EXP(YSTMIN))    
          YST=LOG(TAN(ALOW+(AUPP-ALOW)*VVAR))   
        ENDIF   
        VINT(22)=MIN(YSTMAX,MAX(YSTMIN,YST))    
    
C...Convert VVAR to cos(theta-hat) variable.    
      ELSEIF(IVAR.EQ.3) THEN    
        RM34=2.*VINT(63)*VINT(64)/(VINT(21)*VINT(2))**2 
        RSQM=1.+RM34    
        IF(2.*VINT(71)**2/(VINT(21)*VINT(2)).LT.0.0001) RM34=MAX(RM34,  
     &  2.*VINT(71)**2/(VINT(21)*VINT(2)))  
        CTNMIN=VINT(13) 
        CTNMAX=VINT(33) 
        CTPMIN=VINT(14) 
        CTPMAX=VINT(34) 
        IF(MVAR.EQ.1) THEN  
          ANEG=CTNMAX-CTNMIN    
          APOS=CTPMAX-CTPMIN    
          IF(ANEG.GT.0..AND.VVAR*(ANEG+APOS).LE.ANEG) THEN  
            VCTN=VVAR*(ANEG+APOS)/ANEG  
            CTH=CTNMIN+(CTNMAX-CTNMIN)*VCTN 
          ELSE  
            VCTP=(VVAR*(ANEG+APOS)-ANEG)/APOS   
            CTH=CTPMIN+(CTPMAX-CTPMIN)*VCTP 
          ENDIF 
        ELSEIF(MVAR.EQ.2) THEN  
          RMNMIN=MAX(RM34,RSQM-CTNMIN)  
          RMNMAX=MAX(RM34,RSQM-CTNMAX)  
          RMPMIN=MAX(RM34,RSQM-CTPMIN)  
          RMPMAX=MAX(RM34,RSQM-CTPMAX)  
          ANEG=LOG(RMNMIN/RMNMAX)   
          APOS=LOG(RMPMIN/RMPMAX)   
          IF(ANEG.GT.0..AND.VVAR*(ANEG+APOS).LE.ANEG) THEN  
            VCTN=VVAR*(ANEG+APOS)/ANEG  
            CTH=RSQM-RMNMIN*(RMNMAX/RMNMIN)**VCTN   
          ELSE  
            VCTP=(VVAR*(ANEG+APOS)-ANEG)/APOS   
            CTH=RSQM-RMPMIN*(RMPMAX/RMPMIN)**VCTP   
          ENDIF 
        ELSEIF(MVAR.EQ.3) THEN  
          RMNMIN=MAX(RM34,RSQM+CTNMIN)  
          RMNMAX=MAX(RM34,RSQM+CTNMAX)  
          RMPMIN=MAX(RM34,RSQM+CTPMIN)  
          RMPMAX=MAX(RM34,RSQM+CTPMAX)  
          ANEG=LOG(RMNMAX/RMNMIN)   
          APOS=LOG(RMPMAX/RMPMIN)   
          IF(ANEG.GT.0..AND.VVAR*(ANEG+APOS).LE.ANEG) THEN  
            VCTN=VVAR*(ANEG+APOS)/ANEG  
            CTH=RMNMIN*(RMNMAX/RMNMIN)**VCTN-RSQM   
          ELSE  
            VCTP=(VVAR*(ANEG+APOS)-ANEG)/APOS   
            CTH=RMPMIN*(RMPMAX/RMPMIN)**VCTP-RSQM   
          ENDIF 
        ELSEIF(MVAR.EQ.4) THEN  
          RMNMIN=MAX(RM34,RSQM-CTNMIN)  
          RMNMAX=MAX(RM34,RSQM-CTNMAX)  
          RMPMIN=MAX(RM34,RSQM-CTPMIN)  
          RMPMAX=MAX(RM34,RSQM-CTPMAX)  
          ANEG=1./RMNMAX-1./RMNMIN  
          APOS=1./RMPMAX-1./RMPMIN  
          IF(ANEG.GT.0..AND.VVAR*(ANEG+APOS).LE.ANEG) THEN  
            VCTN=VVAR*(ANEG+APOS)/ANEG  
            CTH=RSQM-1./(1./RMNMIN+ANEG*VCTN)   
          ELSE  
            VCTP=(VVAR*(ANEG+APOS)-ANEG)/APOS   
            CTH=RSQM-1./(1./RMPMIN+APOS*VCTP)   
          ENDIF 
        ELSEIF(MVAR.EQ.5) THEN  
          RMNMIN=MAX(RM34,RSQM+CTNMIN)  
          RMNMAX=MAX(RM34,RSQM+CTNMAX)  
          RMPMIN=MAX(RM34,RSQM+CTPMIN)  
          RMPMAX=MAX(RM34,RSQM+CTPMAX)  
          ANEG=1./RMNMIN-1./RMNMAX  
          APOS=1./RMPMIN-1./RMPMAX  
          IF(ANEG.GT.0..AND.VVAR*(ANEG+APOS).LE.ANEG) THEN  
            VCTN=VVAR*(ANEG+APOS)/ANEG  
            CTH=1./(1./RMNMIN-ANEG*VCTN)-RSQM   
          ELSE  
            VCTP=(VVAR*(ANEG+APOS)-ANEG)/APOS   
            CTH=1./(1./RMPMIN-APOS*VCTP)-RSQM   
          ENDIF
        ELSE
cms ...  needed to avoid compiler warning - should do nothing
          CTH=CTNMIN
        ENDIF   
        IF(CTH.LT.0.) CTH=MIN(CTNMAX,MAX(CTNMIN,CTH))   
        IF(CTH.GT.0.) CTH=MIN(CTPMAX,MAX(CTPMIN,CTH))   
        VINT(23)=CTH    
    
C...Convert VVAR to tau' variable.  
      ELSEIF(IVAR.EQ.4) THEN    
        TAU=VINT(11)    
        TAUPMN=VINT(16) 
        TAUPMX=VINT(36) 
        IF(MINT(43).EQ.1) THEN  
          TAUP=1.   
        ELSEIF(MVAR.EQ.1) THEN  
          TAUP=TAUPMN*(TAUPMX/TAUPMN)**VVAR 
        ELSE    
          AUPP=(1.-TAU/TAUPMX)**4   
          ALOW=(1.-TAU/TAUPMN)**4   
          TAUP=TAU/(1.-(ALOW+(AUPP-ALOW)*VVAR)**0.25)   
        ENDIF   
        VINT(26)=MIN(TAUPMX,MAX(TAUPMN,TAUP))   
      ENDIF 
    
      RETURN    
      END   
    
C***********************************************************************    
    
      SUBROUTINE PYSIGH(NCHN,SIGS)  
    
C...Differential matrix elements for all included subprocesses. 
C...Note that what is coded is (disregarding the COMFAC factor) 
C...1) for 2 -> 1 processes: s-hat/pi*d(sigma-hat), where,  
C...when d(sigma-hat) is given in the zero-width limit, the delta   
C...function in tau is replaced by a Breit-Wigner:  
C...1/pi*(s*m_res*Gamma_res)/((s*tau-m_res^2)^2+(m_res*Gamma_res)^2);   
C...2) for 2 -> 2 processes: (s-hat)**2/pi*d(sigma-hat)/d(t-hat);   
C...i.e., dimensionless quantities. COMFAC contains the factor  
C...pi/s and the conversion factor from GeV^-2 to mb.   
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/LUDAT3/MDCY(500,3),MDME(2000,2),BRAT(2000),KFDP(2000,5)    
      SAVE /LUDAT3/ 
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      DIMENSION X(2),XPQ(-6:6),KFAC(2,-40:40),WDTP(0:40),WDTE(0:40,0:5) 

C...Reset number of channels and cross-section. 
      NCHN=0    
      SIGS=0.   
    
C...Read kinematical variables and limits.  
      ISUB=MINT(1)  
      TAUMIN=VINT(11)   
      YSTMIN=VINT(12)   
      CTNMIN=VINT(13)   
      CTPMIN=VINT(14)   
      XT2MIN=VINT(15)   
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
      XT2MAX=VINT(35)   
      TAUPMX=VINT(36)   

C...Common conversion factors (including Jacobian) for subprocesses.    
cms.. rearranged to avoid compiler warnings
      SQMZ=PMAS(23,1)**2    
      GMMZ=PMAS(23,1)*PMAS(23,2)    
      SQMW=PMAS(24,1)**2    
      GMMW=PMAS(24,1)*PMAS(24,2)    
      SQMH=PMAS(25,1)**2    
      GMMH=PMAS(25,1)*PMAS(25,2)    
      SQMZP=PMAS(32,1)**2   
      GMMZP=PMAS(32,1)*PMAS(32,2)   
      SQMHC=PMAS(37,1)**2   
      GMMHC=PMAS(37,1)*PMAS(37,2)   
      SQMR=PMAS(40,1)**2    
      GMMR=PMAS(40,1)*PMAS(40,2)    
      AEM=PARU(101) 
      XW=PARU(102)   
      MIN1=0    
      MAX1=0    
      MIN2=0    
      MAX2=0 
      MINA=MIN(MIN1,MIN2)   
      MAXA=MAX(MAX1,MAX2) 
      FACA=1.
      COMFAC=PARU(1)*PARU(5)/VINT(2)
      AS=ULALPS(Q2)

C...Derive kinematical quantities.  
      IF(ISET(ISUB).LE.2.OR.ISET(ISUB).EQ.5) THEN   
        X(1)=SQRT(TAU)*EXP(YST) 
        X(2)=SQRT(TAU)*EXP(-YST)    
      ELSE  
        X(1)=SQRT(TAUP)*EXP(YST)    
        X(2)=SQRT(TAUP)*EXP(-YST)   
      ENDIF 
      IF(MINT(43).EQ.4.AND.ISET(ISUB).GE.1.AND. 
     &(X(1).GT.0.999.OR.X(2).GT.0.999)) RETURN  
      SH=TAU*VINT(2)    
      SQM3=VINT(63) 
      SQM4=VINT(64) 
      RM3=SQM3/SH   
      RM4=SQM4/SH   
      BE34=SQRT((1.-RM3-RM4)**2-4.*RM3*RM4) 
      RPTS=4.*VINT(71)**2/SH    
      BE34L=SQRT(MAX(0.,(1.-RM3-RM4)**2-4.*RM3*RM4-RPTS))   
      RM34=2.*RM3*RM4   
      RSQM=1.+RM34  
      RTHM=(4.*RM3*RM4+RPTS)/(1.-RM3-RM4+BE34L) 
      TH=-0.5*SH*MAX(RTHM,1.-RM3-RM4-BE34*CTH)  
      UH=-0.5*SH*MAX(RTHM,1.-RM3-RM4+BE34*CTH)  
      SQPTH=0.25*SH*BE34**2*(1.-CTH**2) 
      SH2=SH**2 
      TH2=TH**2 
      UH2=UH**2 
    
C...Choice of Q2 scale. 
      IF(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3) THEN   
        Q2=SH   
      ELSEIF(MOD(ISET(ISUB),2).EQ.0.OR.ISET(ISUB).EQ.5) THEN    
        IF(MSTP(32).EQ.1) THEN  
          Q2=2.*SH*TH*UH/(SH**2+TH**2+UH**2)    
        ELSEIF(MSTP(32).EQ.2) THEN  
          Q2=SQPTH+0.5*(SQM3+SQM4)  
        ELSEIF(MSTP(32).EQ.3) THEN  
          Q2=MIN(-TH,-UH)   
        ELSEIF(MSTP(32).EQ.4) THEN  
          Q2=SH 
        ENDIF   
        IF(ISET(ISUB).EQ.5.AND.MSTP(82).GE.2) Q2=Q2+PARP(82)**2 
      ENDIF 
    
C...Store derived kinematical quantities.   
      VINT(41)=X(1) 
      VINT(42)=X(2) 
      VINT(44)=SH   
      VINT(43)=SQRT(SH) 
      VINT(45)=TH   
      VINT(46)=UH   
      VINT(48)=SQPTH    
      VINT(47)=SQRT(SQPTH)  
      VINT(50)=TAUP*VINT(2) 
      VINT(49)=SQRT(MAX(0.,VINT(50)))   
      VINT(52)=Q2   
      VINT(51)=SQRT(Q2) 
    
C...Calculate parton structure functions.   
      IF(ISET(ISUB).LE.0) GOTO 145  
      IF(MINT(43).GE.2) THEN    
        Q2SF=Q2 
        IF(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4) THEN 
          Q2SF=PMAS(23,1)**2    
          IF(ISUB.EQ.8.OR.ISUB.EQ.76.OR.ISUB.EQ.77) Q2SF=PMAS(24,1)**2  
        ENDIF   
        DO 100 I=3-MINT(41),MINT(42)    
        XSF=X(I)    
        IF(ISET(ISUB).EQ.5) XSF=X(I)/VINT(142+I)    
        CALL PYSTFU(MINT(10+I),XSF,Q2SF,XPQ,I)    
        DO 100 KFL=-6,6 
  100   XSFX(I,KFL)=XPQ(KFL)
      ENDIF 
    
C...Calculate alpha_strong and K-factor.    
      IF(MSTP(33).NE.3) AS=ULALPS(Q2)   
      FACK=1.   
      FACA=1.   
      IF(MSTP(33).EQ.1) THEN    
        FACK=PARP(31)   
      ELSEIF(MSTP(33).EQ.2) THEN    
        FACK=PARP(31)   
        FACA=PARP(32)/PARP(31)  
      ELSEIF(MSTP(33).EQ.3) THEN    
        Q2AS=PARP(33)*Q2    
        IF(ISET(ISUB).EQ.5.AND.MSTP(82).GE.2) Q2AS=Q2AS+    
     &  PARU(112)*PARP(82)  
        AS=ULALPS(Q2AS) 
      ENDIF 
      RADC=1.+AS/PARU(1)    
    
C...Set flags for allowed reacting partons/leptons. 
      DO 130 I=1,2  
      DO 110 J=-40,40   
  110 KFAC(I,J)=0   
      IF(MINT(40+I).EQ.1) THEN  
        KFAC(I,MINT(10+I))=1    
      ELSE  
        DO 120 J=-40,40 
        KFAC(I,J)=KFIN(I,J) 
        IF(ABS(J).GT.MSTP(54).AND.J.NE.21) KFAC(I,J)=0  
        IF(ABS(J).LE.6) THEN    
          IF(XSFX(I,J).LT.1.E-10) KFAC(I,J)=0   
        ELSEIF(J.EQ.21) THEN    
          IF(XSFX(I,0).LT.1.E-10) KFAC(I,21)=0  
        ENDIF   
  120   CONTINUE    
      ENDIF 
  130 CONTINUE  
    
C...Lower and upper limit for flavour loops.    
      DO 140 J=-20,20   
      IF(KFAC(1,-J).EQ.1) MIN1=-J   
      IF(KFAC(1,J).EQ.1) MAX1=J 
      IF(KFAC(2,-J).EQ.1) MIN2=-J   
      IF(KFAC(2,J).EQ.1) MAX2=J 
  140 CONTINUE  
      MINA=MIN(MIN1,MIN2)   
      MAXA=MAX(MAX1,MAX2)   
    
C...Phase space integral in tau and y*. 
      COMFAC=PARU(1)*PARU(5)/VINT(2)    
      IF(MINT(43).EQ.4) COMFAC=COMFAC*FACK  
      IF((MINT(43).GE.2.OR.ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4).AND. 
     &ISET(ISUB).NE.5) THEN 
        ATAU0=LOG(TAUMAX/TAUMIN)    
        ATAU1=(TAUMAX-TAUMIN)/(TAUMAX*TAUMIN)   
        H1=COEF(ISUB,1)+(ATAU0/ATAU1)*COEF(ISUB,2)/TAU  
        IF(MINT(72).GE.1) THEN  
          TAUR1=VINT(73)    
          GAMR1=VINT(74)    
          ATAU2=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR1)/(TAUMAX+TAUR1))/TAUR1  
          ATAU3=(ATAN((TAUMAX-TAUR1)/GAMR1)-ATAN((TAUMIN-TAUR1)/GAMR1))/    
     &    GAMR1 
          H1=H1+(ATAU0/ATAU2)*COEF(ISUB,3)/(TAU+TAUR1)+ 
     &    (ATAU0/ATAU3)*COEF(ISUB,4)*TAU/((TAU-TAUR1)**2+GAMR1**2)  
        ENDIF   
        IF(MINT(72).EQ.2) THEN  
          TAUR2=VINT(75)    
          GAMR2=VINT(76)    
          ATAU4=LOG(TAUMAX/TAUMIN*(TAUMIN+TAUR2)/(TAUMAX+TAUR2))/TAUR2  
          ATAU5=(ATAN((TAUMAX-TAUR2)/GAMR2)-ATAN((TAUMIN-TAUR2)/GAMR2))/    
     &    GAMR2 
          H1=H1+(ATAU0/ATAU4)*COEF(ISUB,5)/(TAU+TAUR2)+ 
     &    (ATAU0/ATAU5)*COEF(ISUB,6)*TAU/((TAU-TAUR2)**2+GAMR2**2)  
        ENDIF   
        COMFAC=COMFAC*ATAU0/(TAU*H1)    
      ENDIF 
      IF(MINT(43).EQ.4.AND.ISET(ISUB).NE.5) THEN    
        AYST0=YSTMAX-YSTMIN 
        AYST1=0.5*(YSTMAX-YSTMIN)**2    
        AYST2=AYST1 
        AYST3=2.*(ATAN(EXP(YSTMAX))-ATAN(EXP(YSTMIN)))  
        H2=(AYST0/AYST1)*COEF(ISUB,7)*(YST-YSTMIN)+(AYST0/AYST2)*   
     &  COEF(ISUB,8)*(YSTMAX-YST)+(AYST0/AYST3)*COEF(ISUB,9)/COSH(YST)  
        COMFAC=COMFAC*AYST0/H2  
      ENDIF 
    
C...2 -> 1 processes: reduction in angular part of phase space integral 
C...for case of decaying resonance. 
      ACTH0=CTNMAX-CTNMIN+CTPMAX-CTPMIN 
clin-4/2008 modified a la pythia6115.f to avoid invalid MDCY subcript#1,
c     also break up compound IF statements:
c      IF((ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3).AND.  
c     &MDCY(KFPR(ISUB,1),1).EQ.1) THEN   
c        IF(KFPR(ISUB,1).EQ.25.OR.KFPR(ISUB,1).EQ.37) THEN   
c          COMFAC=COMFAC*0.5*ACTH0   
c        ELSE    
c          COMFAC=COMFAC*0.125*(3.*ACTH0+CTNMAX**3-CTNMIN**3+    
c     &    CTPMAX**3-CTPMIN**3)  
c        ENDIF   
      IF(ISET(ISUB).EQ.1.OR.ISET(ISUB).EQ.3) THEN
         if(MDCY(LUCOMP(KFPR(ISUB,1)),1).EQ.1) then
            IF(KFPR(ISUB,1).EQ.25.OR.KFPR(ISUB,1).EQ.37) THEN   
               COMFAC=COMFAC*0.5*ACTH0   
            ELSE    
               COMFAC=COMFAC*0.125*(3.*ACTH0+CTNMAX**3-CTNMIN**3+    
     &              CTPMAX**3-CTPMIN**3)  
            ENDIF
         endif
c
C...2 -> 2 processes: angular part of phase space integral. 
      ELSEIF(ISET(ISUB).EQ.2.OR.ISET(ISUB).EQ.4) THEN   
        ACTH1=LOG((MAX(RM34,RSQM-CTNMIN)*MAX(RM34,RSQM-CTPMIN))/    
     &  (MAX(RM34,RSQM-CTNMAX)*MAX(RM34,RSQM-CTPMAX)))  
        ACTH2=LOG((MAX(RM34,RSQM+CTNMAX)*MAX(RM34,RSQM+CTPMAX))/    
     &  (MAX(RM34,RSQM+CTNMIN)*MAX(RM34,RSQM+CTPMIN)))  
        ACTH3=1./MAX(RM34,RSQM-CTNMAX)-1./MAX(RM34,RSQM-CTNMIN)+    
     &  1./MAX(RM34,RSQM-CTPMAX)-1./MAX(RM34,RSQM-CTPMIN)   
        ACTH4=1./MAX(RM34,RSQM+CTNMIN)-1./MAX(RM34,RSQM+CTNMAX)+    
     &  1./MAX(RM34,RSQM+CTPMIN)-1./MAX(RM34,RSQM+CTPMAX)   
        H3=COEF(ISUB,10)+   
     &  (ACTH0/ACTH1)*COEF(ISUB,11)/MAX(RM34,RSQM-CTH)+ 
     &  (ACTH0/ACTH2)*COEF(ISUB,12)/MAX(RM34,RSQM+CTH)+ 
     &  (ACTH0/ACTH3)*COEF(ISUB,13)/MAX(RM34,RSQM-CTH)**2+  
     &  (ACTH0/ACTH4)*COEF(ISUB,14)/MAX(RM34,RSQM+CTH)**2   
        COMFAC=COMFAC*ACTH0*0.5*BE34/H3 
      ENDIF 
    
C...2 -> 3, 4 processes: phace space integral in tau'.  
      IF(MINT(43).GE.2.AND.(ISET(ISUB).EQ.3.OR.ISET(ISUB).EQ.4)) THEN   
        ATAUP0=LOG(TAUPMX/TAUPMN)   
        ATAUP1=((1.-TAU/TAUPMX)**4-(1.-TAU/TAUPMN)**4)/(4.*TAU) 
        H4=COEF(ISUB,15)+   
     &  ATAUP0/ATAUP1*COEF(ISUB,16)/TAUP*(1.-TAU/TAUP)**3   
        IF(1.-TAU/TAUP.GT.1.E-4) THEN   
          FZW=(1.+TAU/TAUP)*LOG(TAUP/TAU)-2.*(1.-TAU/TAUP)  
        ELSE    
          FZW=1./6.*(1.-TAU/TAUP)**3*TAU/TAUP   
        ENDIF   
        COMFAC=COMFAC*ATAUP0*FZW/H4 
      ENDIF 
    
C...Phase space integral for low-pT and multiple interactions.  
      IF(ISET(ISUB).EQ.5) THEN  
        COMFAC=PARU(1)*PARU(5)*FACK*0.5*VINT(2)/SH2 
        ATAU0=LOG(2.*(1.+SQRT(1.-XT2))/XT2-1.)  
        ATAU1=2.*ATAN(1./XT2-1.)/SQRT(XT2)  
        H1=COEF(ISUB,1)+(ATAU0/ATAU1)*COEF(ISUB,2)/SQRT(TAU)    
        COMFAC=COMFAC*ATAU0/H1  
        AYST0=YSTMAX-YSTMIN 
        AYST1=0.5*(YSTMAX-YSTMIN)**2    
        AYST3=2.*(ATAN(EXP(YSTMAX))-ATAN(EXP(YSTMIN)))  
        H2=(AYST0/AYST1)*COEF(ISUB,7)*(YST-YSTMIN)+(AYST0/AYST1)*   
     &  COEF(ISUB,8)*(YSTMAX-YST)+(AYST0/AYST3)*COEF(ISUB,9)/COSH(YST)  
        COMFAC=COMFAC*AYST0/H2  
        IF(MSTP(82).LE.1) COMFAC=COMFAC*XT2**2*(1./VINT(149)-1.)    
C...For MSTP(82)>=2 an additional factor (xT2/(xT2+VINT(149))**2 is 
C...introduced to make cross-section finite for xT2 -> 0.   
        IF(MSTP(82).GE.2) COMFAC=COMFAC*XT2**2/(VINT(149)*  
     &  (1.+VINT(149))) 
      ENDIF 
    
C...A: 2 -> 1, tree diagrams.   
    
  145 IF(ISUB.LE.10) THEN   
      IF(ISUB.EQ.1) THEN    
C...f + fb -> gamma*/Z0.    
        MINT(61)=2  
        CALL PYWIDT(23,SQRT(SH),WDTP,WDTE)  
        FACZ=COMFAC*AEM**2*4./3.    
        DO 150 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 150    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        FACF=1. 
        IF(IABS(I).LE.10) FACF=FACA/3.  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACF*FACZ*(EI**2*VINT(111)+EI*VI/(8.*XW*(1.-XW))*    
     &  SH*(SH-SQMZ)/((SH-SQMZ)**2+GMMZ**2)*VINT(112)+(VI**2+AI**2)/    
     &  (16.*XW*(1.-XW))**2*SH2/((SH-SQMZ)**2+GMMZ**2)*VINT(114))   
  150   CONTINUE    
    
      ELSEIF(ISUB.EQ.2) THEN    
C...f + fb' -> W+/-.    
        CALL PYWIDT(24,SQRT(SH),WDTP,WDTE)  
        FACW=COMFAC*(AEM/XW)**2*1./24*SH2/((SH-SQMW)**2+GMMW**2)    
        DO 170 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 170   
        IA=IABS(I)  
        DO 160 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 160   
        JA=IABS(J)  
        IF(I*J.GT.0.OR.MOD(IA+JA,2).EQ.0) GOTO 160  
        IF((IA.LE.10.AND.JA.GT.10).OR.(IA.GT.10.AND.JA.LE.10)) GOTO 160 
        KCHW=(KCHG(IA,1)*ISIGN(1,I)+KCHG(JA,1)*ISIGN(1,J))/3    
        FACF=1. 
        IF(IA.LE.10) FACF=VCKM((IA+1)/2,(JA+1)/2)*FACA/3.   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACF*FACW*(WDTE(0,1)+WDTE(0,(5-KCHW)/2)+WDTE(0,4))   
  160   CONTINUE    
  170   CONTINUE    
    
      ELSEIF(ISUB.EQ.3) THEN    
C...f + fb -> H0.   
        CALL PYWIDT(25,SQRT(SH),WDTP,WDTE)  
        FACH=COMFAC*(AEM/XW)**2*1./48.*(SH/SQMW)**2*    
     &  SH2/((SH-SQMH)**2+GMMH**2)*(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))  
        DO 180 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 180    
        RMQ=PMAS(IABS(I),1)**2/SH   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*RMQ*SQRT(MAX(0.,1.-4.*RMQ)) 
  180   CONTINUE    
    
      ELSEIF(ISUB.EQ.4) THEN    
C...gamma + W+/- -> W+/-.   
    
      ELSEIF(ISUB.EQ.5) THEN    
C...Z0 + Z0 -> H0.  
        CALL PYWIDT(25,SQRT(SH),WDTP,WDTE)  
        FACH=COMFAC*1./(128.*PARU(1)**2*16.*(1.-XW)**3)*(AEM/XW)**4*    
     &  (SH/SQMW)**2*SH2/((SH-SQMH)**2+GMMH**2)*    
     &  (WDTE(0,1)+WDTE(0,2)+WDTE(0,4)) 
        DO 200 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 200   
        DO 190 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 190   
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        EJ=KCHG(IABS(J),1)/3.   
        AJ=SIGN(1.,EJ)  
        VJ=AJ-4.*EJ*XW  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*(VI**2+AI**2)*(VJ**2+AJ**2) 
  190   CONTINUE    
  200   CONTINUE    
    
      ELSEIF(ISUB.EQ.6) THEN    
C...Z0 + W+/- -> W+/-.  
    
      ELSEIF(ISUB.EQ.7) THEN    
C...W+ + W- -> Z0.  
    
      ELSEIF(ISUB.EQ.8) THEN    
C...W+ + W- -> H0.  
        CALL PYWIDT(25,SQRT(SH),WDTP,WDTE)  
        FACH=COMFAC*1./(128*PARU(1)**2)*(AEM/XW)**4*(SH/SQMW)**2*   
     &  SH2/((SH-SQMH)**2+GMMH**2)*(WDTE(0,1)+WDTE(0,2)+WDTE(0,4))  
        DO 220 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 220   
        EI=SIGN(1.,FLOAT(I))*KCHG(IABS(I),1)    
        DO 210 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 210   
        EJ=SIGN(1.,FLOAT(J))*KCHG(IABS(J),1)    
        IF(EI*EJ.GT.0.) GOTO 210    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*VINT(180+I)*VINT(180+J) 
  210   CONTINUE    
  220   CONTINUE    
      ENDIF 
    
C...B: 2 -> 2, tree diagrams.   
    
      ELSEIF(ISUB.LE.20) THEN   
      IF(ISUB.EQ.11) THEN   
C...f + f' -> f + f'.   
        FACQQ1=COMFAC*AS**2*4./9.*(SH2+UH2)/TH2 
        FACQQB=COMFAC*AS**2*4./9.*((SH2+UH2)/TH2*FACA-  
     &  MSTP(34)*2./3.*UH2/(SH*TH)) 
        FACQQ2=COMFAC*AS**2*4./9.*((SH2+TH2)/UH2-   
     &  MSTP(34)*2./3.*SH2/(TH*UH)) 
        DO 240 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 240   
        DO 230 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 230   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQQ1   
        IF(I.EQ.-J) SIGH(NCHN)=FACQQB   
        IF(I.EQ.J) THEN 
          SIGH(NCHN)=0.5*SIGH(NCHN) 
          NCHN=NCHN+1   
          ISIG(NCHN,1)=I    
          ISIG(NCHN,2)=J    
          ISIG(NCHN,3)=2    
          SIGH(NCHN)=0.5*FACQQ2 
        ENDIF   
  230   CONTINUE    
  240   CONTINUE    
    
      ELSEIF(ISUB.EQ.12) THEN   
C...f + fb -> f' + fb' (q + qb -> q' + qb' only).   
        CALL PYWIDT(21,SQRT(SH),WDTP,WDTE)  
        FACQQB=COMFAC*AS**2*4./9.*(TH2+UH2)/SH2*(WDTE(0,1)+WDTE(0,2)+   
     &  WDTE(0,3)+WDTE(0,4))    
        DO 250 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 250    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQQB   
  250   CONTINUE    
    
      ELSEIF(ISUB.EQ.13) THEN   
C...f + fb -> g + g (q + qb -> g + g only). 
        FACGG1=COMFAC*AS**2*32./27.*(UH/TH-(2.+MSTP(34)*1./4.)*UH2/SH2) 
        FACGG2=COMFAC*AS**2*32./27.*(TH/UH-(2.+MSTP(34)*1./4.)*TH2/SH2) 
        DO 260 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 260    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=0.5*FACGG1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=2  
        SIGH(NCHN)=0.5*FACGG2   
  260   CONTINUE    
    
      ELSEIF(ISUB.EQ.14) THEN   
C...f + fb -> g + gamma (q + qb -> g + gamma only). 
        FACGG=COMFAC*AS*AEM*8./9.*(TH2+UH2)/(TH*UH) 
        DO 270 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 270    
        EI=KCHG(IABS(I),1)/3.   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGG*EI**2  
  270   CONTINUE    
    
      ELSEIF(ISUB.EQ.15) THEN   
C...f + fb -> g + Z0 (q + qb -> g + Z0 only).   
        FACZG=COMFAC*AS*AEM/(XW*(1.-XW))*1./18.*    
     &  (TH2+UH2+2.*SQM4*SH)/(TH*UH)    
        FACZG=FACZG*WIDS(23,2)  
        DO 280 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 280    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACZG*(VI**2+AI**2)  
  280   CONTINUE    
    
      ELSEIF(ISUB.EQ.16) THEN   
C...f + fb' -> g + W+/- (q + qb' -> g + W+/- only). 
        FACWG=COMFAC*AS*AEM/XW*2./9.*(TH2+UH2+2.*SQM4*SH)/(TH*UH)   
        DO 300 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 300   
        IA=IABS(I)  
        DO 290 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 290   
        JA=IABS(J)  
        IF(I*J.GT.0.OR.MOD(IA+JA,2).EQ.0) GOTO 290  
        KCHW=(KCHG(IA,1)*ISIGN(1,I)+KCHG(JA,1)*ISIGN(1,J))/3    
        FCKM=1. 
        IF(MINT(43).EQ.4) FCKM=VCKM((IA+1)/2,(JA+1)/2)  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACWG*FCKM*WIDS(24,(5-KCHW)/2)   
  290   CONTINUE    
  300   CONTINUE    
    
      ELSEIF(ISUB.EQ.17) THEN   
C...f + fb -> g + H0 (q + qb -> g + H0 only).   
    
      ELSEIF(ISUB.EQ.18) THEN   
C...f + fb -> gamma + gamma.    
        FACGG=COMFAC*FACA*AEM**2*1./3.*(TH2+UH2)/(TH*UH)    
        DO 310 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 310    
        EI=KCHG(IABS(I),1)/3.   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGG*EI**4  
  310   CONTINUE    
    
      ELSEIF(ISUB.EQ.19) THEN   
C...f + fb -> gamma + Z0.   
        FACGZ=COMFAC*FACA*AEM**2/(XW*(1.-XW))*1./24.*   
     &  (TH2+UH2+2.*SQM4*SH)/(TH*UH)    
        FACGZ=FACGZ*WIDS(23,2)  
        DO 320 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 320    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGZ*EI**2*(VI**2+AI**2)    
  320   CONTINUE    
    
      ELSEIF(ISUB.EQ.20) THEN   
C...f + fb' -> gamma + W+/-.    
        FACGW=COMFAC*FACA*AEM**2/XW*1./6.*  
     &  ((2.*UH-TH)/(3.*(SH-SQM4)))**2*(TH2+UH2+2.*SQM4*SH)/(TH*UH) 
        DO 340 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 340   
        IA=IABS(I)  
        DO 330 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 330   
        JA=IABS(J)  
        IF(I*J.GT.0.OR.MOD(IA+JA,2).EQ.0) GOTO 330  
        KCHW=(KCHG(IA,1)*ISIGN(1,I)+KCHG(JA,1)*ISIGN(1,J))/3    
        FCKM=1. 
        IF(MINT(43).EQ.4) FCKM=VCKM((IA+1)/2,(JA+1)/2)  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGW*FCKM*WIDS(24,(5-KCHW)/2)   
  330   CONTINUE    
  340   CONTINUE    
      ENDIF 
    
      ELSEIF(ISUB.LE.30) THEN   
      IF(ISUB.EQ.21) THEN   
C...f + fb -> gamma + H0.   
    
      ELSEIF(ISUB.EQ.22) THEN   
C...f + fb -> Z0 + Z0.  
        FACZZ=COMFAC*FACA*(AEM/(XW*(1.-XW)))**2*1./768.*    
     &  (UH/TH+TH/UH+2.*(SQM3+SQM4)*SH/(TH*UH)- 
     &  SQM3*SQM4*(1./TH2+1./UH2))  
        FACZZ=FACZZ*WIDS(23,1)  
        DO 350 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 350    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACZZ*(VI**4+6.*VI**2*AI**2+AI**4)   
  350   CONTINUE    
    
      ELSEIF(ISUB.EQ.23) THEN   
C...f + fb' -> Z0 + W+/-.   
        FACZW=COMFAC*FACA*(AEM/XW)**2*1./6. 
        FACZW=FACZW*WIDS(23,2)  
        THUH=MAX(TH*UH-SQM3*SQM4,SH*CKIN(3)**2) 
        DO 370 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 370   
        IA=IABS(I)  
        DO 360 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 360   
        JA=IABS(J)  
        IF(I*J.GT.0.OR.MOD(IA+JA,2).EQ.0) GOTO 360  
        KCHW=(KCHG(IA,1)*ISIGN(1,I)+KCHG(JA,1)*ISIGN(1,J))/3    
        EI=KCHG(IA,1)/3.    
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        EJ=KCHG(JA,1)/3.    
        AJ=SIGN(1.,EJ)  
        VJ=AJ-4.*EJ*XW  
        IF(VI+AI.GT.0) THEN 
          VISAV=VI  
          AISAV=AI  
          VI=VJ 
          AI=AJ 
          VJ=VISAV  
          AJ=AISAV  
        ENDIF   
        FCKM=1. 
        IF(MINT(43).EQ.4) FCKM=VCKM((IA+1)/2,(JA+1)/2)  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACZW*FCKM*(1./(SH-SQMW)**2* 
     &  ((9.-8.*XW)/4.*THUH+(8.*XW-6.)/4.*SH*(SQM3+SQM4))+  
     &  (THUH-SH*(SQM3+SQM4))/(2.*(SH-SQMW))*((VJ+AJ)/TH-(VI+AI)/UH)+   
     &  THUH/(16.*(1.-XW))*((VJ+AJ)**2/TH2+(VI+AI)**2/UH2)+ 
     &  SH*(SQM3+SQM4)/(8.*(1.-XW))*(VI+AI)*(VJ+AJ)/(TH*UH))*   
     &  WIDS(24,(5-KCHW)/2) 
  360   CONTINUE    
  370   CONTINUE    
    
      ELSEIF(ISUB.EQ.24) THEN   
C...f + fb -> Z0 + H0.  
        THUH=MAX(TH*UH-SQM3*SQM4,SH*CKIN(3)**2) 
        FACHZ=COMFAC*FACA*(AEM/(XW*(1.-XW)))**2*1./96.* 
     &  (THUH+2.*SH*SQMZ)/(SH-SQMZ)**2  
        FACHZ=FACHZ*WIDS(23,2)*WIDS(25,2)   
        DO 380 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 380    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHZ*(VI**2+AI**2)  
  380   CONTINUE    
    
      ELSEIF(ISUB.EQ.25) THEN   
C...f + fb -> W+ + W-.  
        FACWW=COMFAC*FACA*(AEM/XW)**2*1./12.    
        FACWW=FACWW*WIDS(24,1)  
        THUH=MAX(TH*UH-SQM3*SQM4,SH*CKIN(3)**2) 
        DO 390 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 390    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        DSIGWW=THUH/SH2*(3.-(SH-3.*(SQM3+SQM4))/(SH-SQMZ)*  
     &  (VI+AI)/(2.*AI*(1.-XW))+(SH/(SH-SQMZ))**2*  
     &  (1.-2.*(SQM3+SQM4)/SH+12.*SQM3*SQM4/SH2)*(VI**2+AI**2)/ 
     &  (8.*(1.-XW)**2))-2.*SQMZ/(SH-SQMZ)*(VI+AI)/AI+  
     &  SQMZ*SH/(SH-SQMZ)**2*(1.-2.*(SQM3+SQM4)/SH)*(VI**2+AI**2)/  
     &  (2.*(1.-XW))    
        IF(KCHG(IABS(I),1).LT.0) THEN   
          DSIGWW=DSIGWW+2.*(1.+SQMZ/(SH-SQMZ)*(VI+AI)/(2.*AI))* 
     &    (THUH/(SH*TH)-(SQM3+SQM4)/TH)+THUH/TH2    
        ELSE    
          DSIGWW=DSIGWW+2.*(1.+SQMZ/(SH-SQMZ)*(VI+AI)/(2.*AI))* 
     &    (THUH/(SH*UH)-(SQM3+SQM4)/UH)+THUH/UH2    
        ENDIF   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACWW*DSIGWW 
  390   CONTINUE    
    
      ELSEIF(ISUB.EQ.26) THEN   
C...f + fb' -> W+/- + H0.   
        THUH=MAX(TH*UH-SQM3*SQM4,SH*CKIN(3)**2) 
        FACHW=COMFAC*FACA*(AEM/XW)**2*1./24.*(THUH+2.*SH*SQMW)/ 
     &  (SH-SQMW)**2    
        FACHW=FACHW*WIDS(25,2)  
        DO 410 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 410   
        IA=IABS(I)  
        DO 400 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(1,J).EQ.0) GOTO 400   
        JA=IABS(J)  
        IF(I*J.GT.0.OR.MOD(IA+JA,2).EQ.0) GOTO 400  
        KCHW=(KCHG(IA,1)*ISIGN(1,I)+KCHG(JA,1)*ISIGN(1,J))/3    
        FCKM=1. 
        IF(MINT(43).EQ.4) FCKM=VCKM((IA+1)/2,(JA+1)/2)  
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHW*FCKM*WIDS(24,(5-KCHW)/2)   
  400   CONTINUE    
  410   CONTINUE    
    
      ELSEIF(ISUB.EQ.27) THEN   
C...f + fb -> H0 + H0.  
    
      ELSEIF(ISUB.EQ.28) THEN   
C...f + g -> f + g (q + g -> q + g only).   
        FACQG1=COMFAC*AS**2*4./9.*((2.+MSTP(34)*1./4.)*UH2/TH2-UH/SH)*  
     &  FACA    
        FACQG2=COMFAC*AS**2*4./9.*((2.+MSTP(34)*1./4.)*SH2/TH2-SH/UH)   
        DO 430 I=MINA,MAXA  
        IF(I.EQ.0) GOTO 430 
        DO 420 ISDE=1,2 
        IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 420    
        IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 420    
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQG1   
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=2  
        SIGH(NCHN)=FACQG2   
  420   CONTINUE    
  430   CONTINUE    
    
      ELSEIF(ISUB.EQ.29) THEN   
C...f + g -> f + gamma (q + g -> q + gamma only).   
        FGQ=COMFAC*FACA*AS*AEM*1./3.*(SH2+UH2)/(-SH*UH) 
        DO 450 I=MINA,MAXA  
        IF(I.EQ.0) GOTO 450 
        EI=KCHG(IABS(I),1)/3.   
        FACGQ=FGQ*EI**2 
        DO 440 ISDE=1,2 
        IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 440    
        IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 440    
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGQ    
  440   CONTINUE    
  450   CONTINUE    
    
      ELSEIF(ISUB.EQ.30) THEN   
C...f + g -> f + Z0 (q + g -> q + Z0 only). 
        FZQ=COMFAC*FACA*AS*AEM/(XW*(1.-XW))*1./48.* 
     &  (SH2+UH2+2.*SQM4*TH)/(-SH*UH)   
        FZQ=FZQ*WIDS(23,2)  
        DO 470 I=MINA,MAXA  
        IF(I.EQ.0) GOTO 470 
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        FACZQ=FZQ*(VI**2+AI**2) 
        DO 460 ISDE=1,2 
        IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 460    
        IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 460    
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACZQ    
  460   CONTINUE    
  470   CONTINUE    
      ENDIF 
    
      ELSEIF(ISUB.LE.40) THEN   
      IF(ISUB.EQ.31) THEN   
C...f + g -> f' + W+/- (q + g -> q' + W+/- only).   
        FACWQ=COMFAC*FACA*AS*AEM/XW*1./12.* 
     &  (SH2+UH2+2.*SQM4*TH)/(-SH*UH)   
        DO 490 I=MINA,MAXA  
        IF(I.EQ.0) GOTO 490 
        IA=IABS(I)  
        KCHW=ISIGN(1,KCHG(IA,1)*ISIGN(1,I)) 
        DO 480 ISDE=1,2 
        IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 480    
        IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 480    
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACWQ*VINT(180+I)*WIDS(24,(5-KCHW)/2)    
  480   CONTINUE    
  490   CONTINUE    
    
      ELSEIF(ISUB.EQ.32) THEN   
C...f + g -> f + H0 (q + g -> q + H0 only). 
    
      ELSEIF(ISUB.EQ.33) THEN   
C...f + gamma -> f + g (q + gamma -> q + g only).   
    
      ELSEIF(ISUB.EQ.34) THEN   
C...f + gamma -> f + gamma. 
    
      ELSEIF(ISUB.EQ.35) THEN   
C...f + gamma -> f + Z0.    
    
      ELSEIF(ISUB.EQ.36) THEN   
C...f + gamma -> f' + W+/-. 
    
      ELSEIF(ISUB.EQ.37) THEN   
C...f + gamma -> f + H0.    
    
      ELSEIF(ISUB.EQ.38) THEN   
C...f + Z0 -> f + g (q + Z0 -> q + g only). 
    
      ELSEIF(ISUB.EQ.39) THEN   
C...f + Z0 -> f + gamma.    
    
      ELSEIF(ISUB.EQ.40) THEN   
C...f + Z0 -> f + Z0.   
      ENDIF 
    
      ELSEIF(ISUB.LE.50) THEN   
      IF(ISUB.EQ.41) THEN   
C...f + Z0 -> f' + W+/-.    
    
      ELSEIF(ISUB.EQ.42) THEN   
C...f + Z0 -> f + H0.   
    
      ELSEIF(ISUB.EQ.43) THEN   
C...f + W+/- -> f' + g (q + W+/- -> q' + g only).   
    
      ELSEIF(ISUB.EQ.44) THEN   
C...f + W+/- -> f' + gamma. 
    
      ELSEIF(ISUB.EQ.45) THEN   
C...f + W+/- -> f' + Z0.    
    
      ELSEIF(ISUB.EQ.46) THEN   
C...f + W+/- -> f' + W+/-.  
    
      ELSEIF(ISUB.EQ.47) THEN   
C...f + W+/- -> f' + H0.    
    
      ELSEIF(ISUB.EQ.48) THEN   
C...f + H0 -> f + g (q + H0 -> q + g only). 
    
      ELSEIF(ISUB.EQ.49) THEN   
C...f + H0 -> f + gamma.    
    
      ELSEIF(ISUB.EQ.50) THEN   
C...f + H0 -> f + Z0.   
      ENDIF 
    
      ELSEIF(ISUB.LE.60) THEN   
      IF(ISUB.EQ.51) THEN   
C...f + H0 -> f' + W+/-.    
    
      ELSEIF(ISUB.EQ.52) THEN   
C...f + H0 -> f + H0.   
    
      ELSEIF(ISUB.EQ.53) THEN   
C...g + g -> f + fb (g + g -> q + qb only). 
        CALL PYWIDT(21,SQRT(SH),WDTP,WDTE)  
        FACQQ1=COMFAC*AS**2*1./6.*(UH/TH-(2.+MSTP(34)*1./4.)*UH2/SH2)*  
     &  (WDTE(0,1)+WDTE(0,2)+WDTE(0,3)+WDTE(0,4))*FACA  
        FACQQ2=COMFAC*AS**2*1./6.*(TH/UH-(2.+MSTP(34)*1./4.)*TH2/SH2)*  
     &  (WDTE(0,1)+WDTE(0,2)+WDTE(0,3)+WDTE(0,4))*FACA  
        IF(KFAC(1,21)*KFAC(2,21).EQ.0) GOTO 500 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQQ1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=2  
        SIGH(NCHN)=FACQQ2   
  500   CONTINUE    
    
      ELSEIF(ISUB.EQ.54) THEN   
C...g + gamma -> f + fb (g + gamma -> q + qb only). 
    
      ELSEIF(ISUB.EQ.55) THEN   
C...g + gamma -> f + fb (g + gamma -> q + qb only). 
    
      ELSEIF(ISUB.EQ.56) THEN   
C...g + gamma -> f + fb (g + gamma -> q + qb only). 
    
      ELSEIF(ISUB.EQ.57) THEN   
C...g + gamma -> f + fb (g + gamma -> q + qb only). 
    
      ELSEIF(ISUB.EQ.58) THEN   
C...gamma + gamma -> f + fb.    
    
      ELSEIF(ISUB.EQ.59) THEN   
C...gamma + Z0 -> f + fb.   
    
      ELSEIF(ISUB.EQ.60) THEN   
C...gamma + W+/- -> f + fb'.    
      ENDIF 
    
      ELSEIF(ISUB.LE.70) THEN   
      IF(ISUB.EQ.61) THEN   
C...gamma + H0 -> f + fb.   
    
      ELSEIF(ISUB.EQ.62) THEN   
C...Z0 + Z0 -> f + fb.  
    
      ELSEIF(ISUB.EQ.63) THEN   
C...Z0 + W+/- -> f + fb'.   
    
      ELSEIF(ISUB.EQ.64) THEN   
C...Z0 + H0 -> f + fb.  
    
      ELSEIF(ISUB.EQ.65) THEN   
C...W+ + W- -> f + fb.  
    
      ELSEIF(ISUB.EQ.66) THEN   
C...W+/- + H0 -> f + fb'.   
    
      ELSEIF(ISUB.EQ.67) THEN   
C...H0 + H0 -> f + fb.  
    
      ELSEIF(ISUB.EQ.68) THEN   
C...g + g -> g + g. 
        FACGG1=COMFAC*AS**2*9./4.*(SH2/TH2+2.*SH/TH+3.+2.*TH/SH+    
     &  TH2/SH2)*FACA   
        FACGG2=COMFAC*AS**2*9./4.*(UH2/SH2+2.*UH/SH+3.+2.*SH/UH+    
     &  SH2/UH2)*FACA   
        FACGG3=COMFAC*AS**2*9./4.*(TH2/UH2+2.*TH/UH+3+2.*UH/TH+UH2/TH2) 
        IF(KFAC(1,21)*KFAC(2,21).EQ.0) GOTO 510 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=0.5*FACGG1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=2  
        SIGH(NCHN)=0.5*FACGG2   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=3  
        SIGH(NCHN)=0.5*FACGG3   
  510   CONTINUE    
    
      ELSEIF(ISUB.EQ.69) THEN   
C...gamma + gamma -> W+ + W-.   
    
      ELSEIF(ISUB.EQ.70) THEN   
C...gamma + W+/- -> gamma + W+/-.   
      ENDIF 
    
      ELSEIF(ISUB.LE.80) THEN   
      IF(ISUB.EQ.71) THEN   
C...Z0 + Z0 -> Z0 + Z0. 
        BE2=1.-4.*SQMZ/SH   
        TH=-0.5*SH*BE2*(1.-CTH) 
        UH=-0.5*SH*BE2*(1.+CTH) 
        SHANG=1./(1.-XW)*SQMW/SQMZ*(1.+BE2)**2  
        ASHRE=(SH-SQMH)/((SH-SQMH)**2+GMMH**2)*SHANG    
        ASHIM=-GMMH/((SH-SQMH)**2+GMMH**2)*SHANG    
        THANG=1./(1.-XW)*SQMW/SQMZ*(BE2-CTH)**2 
        ATHRE=(TH-SQMH)/((TH-SQMH)**2+GMMH**2)*THANG    
        ATHIM=-GMMH/((TH-SQMH)**2+GMMH**2)*THANG    
        UHANG=1./(1.-XW)*SQMW/SQMZ*(BE2+CTH)**2 
        AUHRE=(UH-SQMH)/((UH-SQMH)**2+GMMH**2)*UHANG    
        AUHIM=-GMMH/((UH-SQMH)**2+GMMH**2)*UHANG    
        FACH=0.5*COMFAC*1./(4096.*PARU(1)**2*16.*(1.-XW)**2)*   
     &  (AEM/XW)**4*(SH/SQMW)**2*((ASHRE+ATHRE+AUHRE)**2+   
     &  (ASHIM+ATHIM+AUHIM)**2)*SQMZ/SQMW   
        DO 530 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 530   
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        AVI=AI**2+VI**2 
        DO 520 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 520   
        EJ=KCHG(IABS(J),1)/3.   
        AJ=SIGN(1.,EJ)  
        VJ=AJ-4.*EJ*XW  
        AVJ=AJ**2+VJ**2 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*AVI*AVJ 
  520   CONTINUE    
  530   CONTINUE    
    
      ELSEIF(ISUB.EQ.72) THEN   
C...Z0 + Z0 -> W+ + W-. 
        BE2=SQRT((1.-4.*SQMW/SH)*(1.-4.*SQMZ/SH))   
        CTH2=CTH**2 
        TH=-0.5*SH*(1.-2.*(SQMW+SQMZ)/SH-BE2*CTH)   
        UH=-0.5*SH*(1.-2.*(SQMW+SQMZ)/SH+BE2*CTH)   
        SHANG=4.*SQRT(SQMW/(SQMZ*(1.-XW)))*(1.-2.*SQMW/SH)* 
     &  (1.-2.*SQMZ/SH) 
        ASHRE=(SH-SQMH)/((SH-SQMH)**2+GMMH**2)*SHANG    
        ASHIM=-GMMH/((SH-SQMH)**2+GMMH**2)*SHANG    
        ATWRE=(1.-XW)/SQMZ*SH/(TH-SQMW)*((CTH-BE2)**2*(3./2.+BE2/2.*CTH-    
     &  (SQMW+SQMZ)/SH+(SQMW-SQMZ)**2/(SH*SQMW))+4.*((SQMW+SQMZ)/SH*    
     &  (1.-3.*CTH2)+8.*SQMW*SQMZ/SH2*(2.*CTH2-1.)+ 
     &  4.*(SQMW**2+SQMZ**2)/SH2*CTH2+2.*(SQMW+SQMZ)/SH*BE2*CTH))   
        ATWIM=0.    
        AUWRE=(1.-XW)/SQMZ*SH/(UH-SQMW)*((CTH+BE2)**2*(3./2.-BE2/2.*CTH-    
     &  (SQMW+SQMZ)/SH+(SQMW-SQMZ)**2/(SH*SQMW))+4.*((SQMW+SQMZ)/SH*    
     &  (1.-3.*CTH2)+8.*SQMW*SQMZ/SH2*(2.*CTH2-1.)+ 
     &  4.*(SQMW**2+SQMZ**2)/SH2*CTH2-2.*(SQMW+SQMZ)/SH*BE2*CTH))   
        AUWIM=0.    
        A4RE=2.*(1.-XW)/SQMZ*(3.-CTH2-4.*(SQMW+SQMZ)/SH)    
        A4IM=0. 
        FACH=COMFAC*1./(4096.*PARU(1)**2*16.*(1.-XW)**2)*(AEM/XW)**4*   
     &  (SH/SQMW)**2*((ASHRE+ATWRE+AUWRE+A4RE)**2+  
     &  (ASHIM+ATWIM+AUWIM+A4IM)**2)*SQMZ/SQMW  
        DO 550 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 550   
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        AVI=AI**2+VI**2 
        DO 540 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 540   
        EJ=KCHG(IABS(J),1)/3.   
        AJ=SIGN(1.,EJ)  
        VJ=AJ-4.*EJ*XW  
        AVJ=AJ**2+VJ**2 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*AVI*AVJ 
  540   CONTINUE    
  550   CONTINUE    
    
      ELSEIF(ISUB.EQ.73) THEN   
C...Z0 + W+/- -> Z0 + W+/-. 
        BE2=1.-2.*(SQMZ+SQMW)/SH+((SQMZ-SQMW)/SH)**2    
        EP1=1.+(SQMZ-SQMW)/SH   
        EP2=1.-(SQMZ-SQMW)/SH   
        TH=-0.5*SH*BE2*(1.-CTH) 
        UH=(SQMZ-SQMW)**2/SH-0.5*SH*BE2*(1.+CTH)    
        THANG=SQRT(SQMW/(SQMZ*(1.-XW)))*(BE2-EP1*CTH)*(BE2-EP2*CTH) 
        ATHRE=(TH-SQMH)/((TH-SQMH)**2+GMMH**2)*THANG    
        ATHIM=-GMMH/((TH-SQMH)**2+GMMH**2)*THANG    
        ASWRE=(1.-XW)/SQMZ*SH/(SH-SQMW)*(-BE2*(EP1+EP2)**4*CTH+ 
     &  1./4.*(BE2+EP1*EP2)**2*((EP1-EP2)**2-4.*BE2*CTH)+   
     &  2.*BE2*(BE2+EP1*EP2)*(EP1+EP2)**2*CTH-  
     &  1./16.*SH/SQMW*(EP1**2-EP2**2)**2*(BE2+EP1*EP2)**2) 
        ASWIM=0.    
        AUWRE=(1.-XW)/SQMZ*SH/(UH-SQMW)*(-BE2*(EP2+EP1*CTH)*    
     &  (EP1+EP2*CTH)*(BE2+EP1*EP2)+BE2*(EP2+EP1*CTH)*  
     &  (BE2+EP1*EP2*CTH)*(2.*EP2-EP2*CTH+EP1)-BE2*(EP2+EP1*CTH)**2*    
     &  (BE2-EP2**2*CTH)-1./8.*(BE2+EP1*EP2*CTH)**2*((EP1+EP2)**2+  
     &  2.*BE2*(1.-CTH))+1./32.*SH/SQMW*(BE2+EP1*EP2*CTH)**2*   
     &  (EP1**2-EP2**2)**2-BE2*(EP1+EP2*CTH)*(EP2+EP1*CTH)* 
     &  (BE2+EP1*EP2)+BE2*(EP1+EP2*CTH)*(BE2+EP1*EP2*CTH)*  
     &  (2.*EP1-EP1*CTH+EP2)-BE2*(EP1+EP2*CTH)**2*(BE2-EP1**2*CTH)- 
     &  1./8.*(BE2+EP1*EP2*CTH)**2*((EP1+EP2)**2+2.*BE2*(1.-CTH))+  
     &  1./32.*SH/SQMW*(BE2+EP1*EP2*CTH)**2*(EP1**2-EP2**2)**2) 
        AUWIM=0.    
        A4RE=(1.-XW)/SQMZ*(EP1**2*EP2**2*(CTH**2-1.)-   
     &  2.*BE2*(EP1**2+EP2**2+EP1*EP2)*CTH-2.*BE2*EP1*EP2)  
        A4IM=0. 
        FACH=COMFAC*1./(4096.*PARU(1)**2*4.*(1.-XW))*(AEM/XW)**4*   
     &  (SH/SQMW)**2*((ATHRE+ASWRE+AUWRE+A4RE)**2+  
     &  (ATHIM+ASWIM+AUWIM+A4IM)**2)*SQRT(SQMZ/SQMW)    
        DO 570 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 570   
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        AVI=AI**2+VI**2 
        DO 560 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 560   
        EJ=KCHG(IABS(J),1)/3.   
        AJ=SIGN(1.,EJ)  
        VJ=AI-4.*EJ*XW  
        AVJ=AJ**2+VJ**2 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*(AVI*VINT(180+J)+VINT(180+I)*AVJ)   
  560   CONTINUE    
  570   CONTINUE    
    
      ELSEIF(ISUB.EQ.75) THEN   
C...W+ + W- -> gamma + gamma.   
    
      ELSEIF(ISUB.EQ.76) THEN   
C...W+ + W- -> Z0 + Z0. 
        BE2=SQRT((1.-4.*SQMW/SH)*(1.-4.*SQMZ/SH))   
        CTH2=CTH**2 
        TH=-0.5*SH*(1.-2.*(SQMW+SQMZ)/SH-BE2*CTH)   
        UH=-0.5*SH*(1.-2.*(SQMW+SQMZ)/SH+BE2*CTH)   
        SHANG=4.*SQRT(SQMW/(SQMZ*(1.-XW)))*(1.-2.*SQMW/SH)* 
     &  (1.-2.*SQMZ/SH) 
        ASHRE=(SH-SQMH)/((SH-SQMH)**2+GMMH**2)*SHANG    
        ASHIM=-GMMH/((SH-SQMH)**2+GMMH**2)*SHANG    
        ATWRE=(1.-XW)/SQMZ*SH/(TH-SQMW)*((CTH-BE2)**2*(3./2.+BE2/2.*CTH-    
     &  (SQMW+SQMZ)/SH+(SQMW-SQMZ)**2/(SH*SQMW))+4.*((SQMW+SQMZ)/SH*    
     &  (1.-3.*CTH2)+8.*SQMW*SQMZ/SH2*(2.*CTH2-1.)+ 
     &  4.*(SQMW**2+SQMZ**2)/SH2*CTH2+2.*(SQMW+SQMZ)/SH*BE2*CTH))   
        ATWIM=0.    
        AUWRE=(1.-XW)/SQMZ*SH/(UH-SQMW)*((CTH+BE2)**2*(3./2.-BE2/2.*CTH-    
     &  (SQMW+SQMZ)/SH+(SQMW-SQMZ)**2/(SH*SQMW))+4.*((SQMW+SQMZ)/SH*    
     &  (1.-3.*CTH2)+8.*SQMW*SQMZ/SH2*(2.*CTH2-1.)+ 
     &  4.*(SQMW**2+SQMZ**2)/SH2*CTH2-2.*(SQMW+SQMZ)/SH*BE2*CTH))   
        AUWIM=0.    
        A4RE=2.*(1.-XW)/SQMZ*(3.-CTH2-4.*(SQMW+SQMZ)/SH)    
        A4IM=0. 
        FACH=0.5*COMFAC*1./(4096.*PARU(1)**2)*(AEM/XW)**4*(SH/SQMW)**2* 
     &  ((ASHRE+ATWRE+AUWRE+A4RE)**2+(ASHIM+ATWIM+AUWIM+A4IM)**2)   
        DO 590 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 590   
        EI=SIGN(1.,FLOAT(I))*KCHG(IABS(I),1)    
        DO 580 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 580   
        EJ=SIGN(1.,FLOAT(J))*KCHG(IABS(J),1)    
        IF(EI*EJ.GT.0.) GOTO 580    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*VINT(180+I)*VINT(180+J) 
  580   CONTINUE    
  590   CONTINUE    
    
      ELSEIF(ISUB.EQ.77) THEN   
C...W+/- + W+/- -> W+/- + W+/-. 
        BE2=1.-4.*SQMW/SH   
        BE4=BE2**2  
        CTH2=CTH**2 
        CTH3=CTH**3 
        TH=-0.5*SH*BE2*(1.-CTH) 
        UH=-0.5*SH*BE2*(1.+CTH) 
        SHANG=(1.+BE2)**2   
        ASHRE=(SH-SQMH)/((SH-SQMH)**2+GMMH**2)*SHANG    
        ASHIM=-GMMH/((SH-SQMH)**2+GMMH**2)*SHANG    
        THANG=(BE2-CTH)**2  
        ATHRE=(TH-SQMH)/((TH-SQMH)**2+GMMH**2)*THANG    
        ATHIM=-GMMH/((TH-SQMH)**2+GMMH**2)*THANG    
        SGZANG=1./SQMW*BE2*(3.-BE2)**2*CTH  
        ASGRE=XW*SGZANG 
        ASGIM=0.    
        ASZRE=(1.-XW)*SH/(SH-SQMZ)*SGZANG   
        ASZIM=0.    
        TGZANG=1./SQMW*(BE2*(4.-2.*BE2+BE4)+BE2*(4.-10.*BE2+BE4)*CTH+   
     &  (2.-11.*BE2+10.*BE4)*CTH2+BE2*CTH3) 
        ATGRE=0.5*XW*SH/TH*TGZANG   
        ATGIM=0.    
        ATZRE=0.5*(1.-XW)*SH/(TH-SQMZ)*TGZANG   
        ATZIM=0.    
        A4RE=1./SQMW*(1.+2.*BE2-6.*BE2*CTH-CTH2)    
        A4IM=0. 
        FACH=COMFAC*1./(4096.*PARU(1)**2)*(AEM/XW)**4*(SH/SQMW)**2* 
     &  ((ASHRE+ATHRE+ASGRE+ASZRE+ATGRE+ATZRE+A4RE)**2+ 
     &  (ASHIM+ATHIM+ASGIM+ASZIM+ATGIM+ATZIM+A4IM)**2)  
        DO 610 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 610   
        EI=SIGN(1.,FLOAT(I))*KCHG(IABS(I),1)    
        DO 600 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 600   
        EJ=SIGN(1.,FLOAT(J))*KCHG(IABS(J),1)    
        IF(EI*EJ.GT.0.) GOTO 600    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH*VINT(180+I)*VINT(180+J) 
  600   CONTINUE    
  610   CONTINUE    
    
      ELSEIF(ISUB.EQ.78) THEN   
C...W+/- + H0 -> W+/- + H0. 
    
      ELSEIF(ISUB.EQ.79) THEN   
C...H0 + H0 -> H0 + H0. 
    
      ENDIF 
    
C...C: 2 -> 2, tree diagrams with masses.   
    
      ELSEIF(ISUB.LE.90) THEN   
      IF(ISUB.EQ.81) THEN   
C...q + qb -> Q + QB.   
        FACQQB=COMFAC*AS**2*4./9.*(((TH-SQM3)**2+   
     &  (UH-SQM3)**2)/SH2+2.*SQM3/SH)   
        IF(MSTP(35).GE.1) THEN  
          IF(MSTP(35).EQ.1) THEN    
            ALSSG=PARP(35)  
          ELSE  
            MST115=MSTU(115)    
            MSTU(115)=MSTP(36)  
            Q2BN=SQRT(SQM3*((SQRT(SH)-2.*SQRT(SQM3))**2+PARP(36)**2))   
            ALSSG=ULALPS(Q2BN)  
            MSTU(115)=MST115    
          ENDIF 
          XREPU=PARU(1)*ALSSG/(6.*SQRT(MAX(1E-20,1.-4.*SQM3/SH)))   
          FREPU=XREPU/(EXP(MIN(100.,XREPU))-1.) 
          PARI(81)=FREPU    
          FACQQB=FACQQB*FREPU   
        ENDIF   
        DO 620 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 620    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQQB   
  620   CONTINUE    
    
      ELSEIF(ISUB.EQ.82) THEN   
C...g + g -> Q + QB.    
        FACQQ1=COMFAC*FACA*AS**2*1./6.*((UH-SQM3)/(TH-SQM3)-    
     &  2.*(UH-SQM3)**2/SH2+4.*SQM3/SH*(TH*UH-SQM3**2)/(TH-SQM3)**2)    
        FACQQ2=COMFAC*FACA*AS**2*1./6.*((TH-SQM3)/(UH-SQM3)-    
     &  2.*(TH-SQM3)**2/SH2+4.*SQM3/SH*(TH*UH-SQM3**2)/(UH-SQM3)**2)    
        IF(MSTP(35).GE.1) THEN  
          IF(MSTP(35).EQ.1) THEN    
            ALSSG=PARP(35)  
          ELSE  
            MST115=MSTU(115)    
            MSTU(115)=MSTP(36)  
            Q2BN=SQRT(SQM3*((SQRT(SH)-2.*SQRT(SQM3))**2+PARP(36)**2))   
            ALSSG=ULALPS(Q2BN)  
            MSTU(115)=MST115    
          ENDIF 
          XATTR=4.*PARU(1)*ALSSG/(3.*SQRT(MAX(1E-20,1.-4.*SQM3/SH)))    
          FATTR=XATTR/(1.-EXP(-MIN(100.,XATTR)))    
          XREPU=PARU(1)*ALSSG/(6.*SQRT(MAX(1E-20,1.-4.*SQM3/SH)))   
          FREPU=XREPU/(EXP(MIN(100.,XREPU))-1.) 
          FATRE=(2.*FATTR+5.*FREPU)/7.  
          PARI(81)=FATRE    
          FACQQ1=FACQQ1*FATRE   
          FACQQ2=FACQQ2*FATRE   
        ENDIF   
        IF(KFAC(1,21)*KFAC(2,21).EQ.0) GOTO 630 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQQ1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=2  
        SIGH(NCHN)=FACQQ2   
  630   CONTINUE    
    
      ENDIF 
    
C...D: Mimimum bias processes.  
    
      ELSEIF(ISUB.LE.100) THEN  
      IF(ISUB.EQ.91) THEN   
C...Elastic scattering. 
        SIGS=XSEC(ISUB,1)   
    
      ELSEIF(ISUB.EQ.92) THEN   
C...Single diffractive scattering.  
        SIGS=XSEC(ISUB,1)   
    
      ELSEIF(ISUB.EQ.93) THEN   
C...Double diffractive scattering.  
        SIGS=XSEC(ISUB,1)   
    
      ELSEIF(ISUB.EQ.94) THEN   
C...Central diffractive scattering. 
        SIGS=XSEC(ISUB,1)   
    
      ELSEIF(ISUB.EQ.95) THEN   
C...Low-pT scattering.  
        SIGS=XSEC(ISUB,1)   
    
      ELSEIF(ISUB.EQ.96) THEN   
C...Multiple interactions: sum of QCD processes.    
        CALL PYWIDT(21,SQRT(SH),WDTP,WDTE)  
    
C...q + q' -> q + q'.   
        FACQQ1=COMFAC*AS**2*4./9.*(SH2+UH2)/TH2 
        FACQQB=COMFAC*AS**2*4./9.*((SH2+UH2)/TH2*FACA-  
     &  MSTP(34)*2./3.*UH2/(SH*TH)) 
        FACQQ2=COMFAC*AS**2*4./9.*((SH2+TH2)/UH2-   
     &  MSTP(34)*2./3.*SH2/(TH*UH)) 
        DO 650 I=-3,3   
        IF(I.EQ.0) GOTO 650 
        DO 640 J=-3,3   
        IF(J.EQ.0) GOTO 640 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=111    
        SIGH(NCHN)=FACQQ1   
        IF(I.EQ.-J) SIGH(NCHN)=FACQQB   
        IF(I.EQ.J) THEN 
          SIGH(NCHN)=0.5*SIGH(NCHN) 
          NCHN=NCHN+1   
          ISIG(NCHN,1)=I    
          ISIG(NCHN,2)=J    
          ISIG(NCHN,3)=112  
          SIGH(NCHN)=0.5*FACQQ2 
        ENDIF   
  640   CONTINUE    
  650   CONTINUE    
    
C...q + qb -> q' + qb' or g + g.    
        FACQQB=COMFAC*AS**2*4./9.*(TH2+UH2)/SH2*(WDTE(0,1)+WDTE(0,2)+   
     &  WDTE(0,3)+WDTE(0,4))    
        FACGG1=COMFAC*AS**2*32./27.*(UH/TH-(2.+MSTP(34)*1./4.)*UH2/SH2) 
        FACGG2=COMFAC*AS**2*32./27.*(TH/UH-(2.+MSTP(34)*1./4.)*TH2/SH2) 
        DO 660 I=-3,3   
        IF(I.EQ.0) GOTO 660 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=121    
        SIGH(NCHN)=FACQQB   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=131    
        SIGH(NCHN)=0.5*FACGG1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=132    
        SIGH(NCHN)=0.5*FACGG2   
  660   CONTINUE    
    
C...q + g -> q + g. 
        FACQG1=COMFAC*AS**2*4./9.*((2.+MSTP(34)*1./4.)*UH2/TH2-UH/SH)*  
     &  FACA    
        FACQG2=COMFAC*AS**2*4./9.*((2.+MSTP(34)*1./4.)*SH2/TH2-SH/UH)   
        DO 680 I=-3,3   
        IF(I.EQ.0) GOTO 680 
        DO 670 ISDE=1,2 
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=281    
        SIGH(NCHN)=FACQG1   
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=282    
        SIGH(NCHN)=FACQG2   
  670   CONTINUE    
  680   CONTINUE    
    
C...g + g -> q + qb or g + g.   
        FACQQ1=COMFAC*AS**2*1./6.*(UH/TH-(2.+MSTP(34)*1./4.)*UH2/SH2)*  
     &  (WDTE(0,1)+WDTE(0,2)+WDTE(0,3)+WDTE(0,4))*FACA  
        FACQQ2=COMFAC*AS**2*1./6.*(TH/UH-(2.+MSTP(34)*1./4.)*TH2/SH2)*  
     &  (WDTE(0,1)+WDTE(0,2)+WDTE(0,3)+WDTE(0,4))*FACA  
        FACGG1=COMFAC*AS**2*9./4.*(SH2/TH2+2.*SH/TH+3.+2.*TH/SH+    
     &  TH2/SH2)*FACA   
        FACGG2=COMFAC*AS**2*9./4.*(UH2/SH2+2.*UH/SH+3.+2.*SH/UH+    
     &  SH2/UH2)*FACA   
        FACGG3=COMFAC*AS**2*9./4.*(TH2/UH2+2.*TH/UH+3+2.*UH/TH+UH2/TH2) 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=531    
        SIGH(NCHN)=FACQQ1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=532    
        SIGH(NCHN)=FACQQ2   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=681    
        SIGH(NCHN)=0.5*FACGG1   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=682    
        SIGH(NCHN)=0.5*FACGG2   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=683    
        SIGH(NCHN)=0.5*FACGG3   
      ENDIF 
    
C...E: 2 -> 1, loop diagrams.   
    
      ELSEIF(ISUB.LE.110) THEN  
      IF(ISUB.EQ.101) THEN  
C...g + g -> gamma*/Z0. 
    
      ELSEIF(ISUB.EQ.102) THEN  
C...g + g -> H0.    
        CALL PYWIDT(25,SQRT(SH),WDTP,WDTE)  
        ETARE=0.    
        ETAIM=0.    
        DO 690 I=1,2*MSTP(1)    
        EPS=4.*PMAS(I,1)**2/SH  
        IF(EPS.LE.1.) THEN  
          IF(EPS.GT.1.E-4) THEN 
            ROOT=SQRT(1.-EPS)   
            RLN=LOG((1.+ROOT)/(1.-ROOT))    
          ELSE  
            RLN=LOG(4./EPS-2.)  
          ENDIF 
          PHIRE=0.25*(RLN**2-PARU(1)**2)    
          PHIIM=0.5*PARU(1)*RLN 
        ELSE    
          PHIRE=-(ASIN(1./SQRT(EPS)))**2    
          PHIIM=0.  
        ENDIF   
        ETARE=ETARE+0.5*EPS*(1.+(EPS-1.)*PHIRE) 
        ETAIM=ETAIM+0.5*EPS*(EPS-1.)*PHIIM  
  690   CONTINUE    
        ETA2=ETARE**2+ETAIM**2  
        FACH=COMFAC*FACA*(AS/PARU(1)*AEM/XW)**2*1./512.*    
     &  (SH/SQMW)**2*ETA2*SH2/((SH-SQMH)**2+GMMH**2)*   
     &  (WDTE(0,1)+WDTE(0,2)+WDTE(0,4)) 
        IF(KFAC(1,21)*KFAC(2,21).EQ.0) GOTO 700 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACH 
  700   CONTINUE    
    
      ENDIF 
    
C...F: 2 -> 2, box diagrams.    
    
      ELSEIF(ISUB.LE.120) THEN  
      IF(ISUB.EQ.111) THEN  
C...f + fb -> g + H0 (q + qb -> g + H0 only).   
        A5STUR=0.   
        A5STUI=0.   
        DO 710 I=1,2*MSTP(1)    
        SQMQ=PMAS(I,1)**2   
        EPSS=4.*SQMQ/SH 
        EPSH=4.*SQMQ/SQMH   
        A5STUR=A5STUR+SQMQ/SQMH*(4.+4.*SH/(TH+UH)*(PYW1AU(EPSS,1)-  
     &  PYW1AU(EPSH,1))+(1.-4.*SQMQ/(TH+UH))*(PYW2AU(EPSS,1)-   
     &  PYW2AU(EPSH,1)))    
        A5STUI=A5STUI+SQMQ/SQMH*(4.*SH/(TH+UH)*(PYW1AU(EPSS,2)- 
     &  PYW1AU(EPSH,2))+(1.-4.*SQMQ/(TH+UH))*(PYW2AU(EPSS,2)-   
     &  PYW2AU(EPSH,2)))    
  710   CONTINUE    
        FACGH=COMFAC*FACA/(144.*PARU(1)**2)*AEM/XW*AS**3*SQMH/SQMW* 
     &  SQMH/SH*(UH**2+TH**2)/(UH+TH)**2*(A5STUR**2+A5STUI**2)  
        FACGH=FACGH*WIDS(25,2)  
        DO 720 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 720    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGH    
  720   CONTINUE    
    
      ELSEIF(ISUB.EQ.112) THEN  
C...f + g -> f + H0 (q + g -> q + H0 only). 
        A5TSUR=0.   
        A5TSUI=0.   
        DO 730 I=1,2*MSTP(1)    
        SQMQ=PMAS(I,1)**2   
        EPST=4.*SQMQ/TH 
        EPSH=4.*SQMQ/SQMH   
        A5TSUR=A5TSUR+SQMQ/SQMH*(4.+4.*TH/(SH+UH)*(PYW1AU(EPST,1)-  
     &  PYW1AU(EPSH,1))+(1.-4.*SQMQ/(SH+UH))*(PYW2AU(EPST,1)-   
     &  PYW2AU(EPSH,1)))    
        A5TSUI=A5TSUI+SQMQ/SQMH*(4.*TH/(SH+UH)*(PYW1AU(EPST,2)- 
     &  PYW1AU(EPSH,2))+(1.-4.*SQMQ/(SH+UH))*(PYW2AU(EPST,2)-   
     &  PYW2AU(EPSH,2)))    
  730   CONTINUE    
        FACQH=COMFAC*FACA/(384.*PARU(1)**2)*AEM/XW*AS**3*SQMH/SQMW* 
     &  SQMH/(-TH)*(UH**2+SH**2)/(UH+SH)**2*(A5TSUR**2+A5TSUI**2)   
        FACQH=FACQH*WIDS(25,2)  
        DO 750 I=MINA,MAXA  
        IF(I.EQ.0) GOTO 750 
        DO 740 ISDE=1,2 
        IF(ISDE.EQ.1.AND.KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 740    
        IF(ISDE.EQ.2.AND.KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 740    
        NCHN=NCHN+1 
        ISIG(NCHN,ISDE)=I   
        ISIG(NCHN,3-ISDE)=21    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACQH    
  740   CONTINUE    
  750   CONTINUE    
    
      ELSEIF(ISUB.EQ.113) THEN  
C...g + g -> g + H0.    
        A2STUR=0.   
        A2STUI=0.   
        A2USTR=0.   
        A2USTI=0.   
        A2TUSR=0.   
        A2TUSI=0.   
        A4STUR=0.   
        A4STUI=0.   
        DO 760 I=6,2*MSTP(1)    
C'''Only t-quarks yet included  
        SQMQ=PMAS(I,1)**2   
        EPSS=4.*SQMQ/SH 
        EPST=4.*SQMQ/TH 
        EPSU=4.*SQMQ/UH 
        EPSH=4.*SQMQ/SQMH   
        IF(EPSH.LT.1.E-6) GOTO 760  
        BESTU=0.5*(1.+SQRT(1.+EPSS*TH/UH))  
        BEUST=0.5*(1.+SQRT(1.+EPSU*SH/TH))  
        BETUS=0.5*(1.+SQRT(1.+EPST*UH/SH))  
        BEUTS=BESTU 
        BETSU=BEUST 
        BESUT=BETUS 
        W3STUR=PYI3AU(BESTU,EPSH,1)-PYI3AU(BESTU,EPSS,1)-   
     &  PYI3AU(BESTU,EPSU,1)    
        W3STUI=PYI3AU(BESTU,EPSH,2)-PYI3AU(BESTU,EPSS,2)-   
     &  PYI3AU(BESTU,EPSU,2)    
        W3SUTR=PYI3AU(BESUT,EPSH,1)-PYI3AU(BESUT,EPSS,1)-   
     &  PYI3AU(BESUT,EPST,1)    
        W3SUTI=PYI3AU(BESUT,EPSH,2)-PYI3AU(BESUT,EPSS,2)-   
     &  PYI3AU(BESUT,EPST,2)    
        W3TSUR=PYI3AU(BETSU,EPSH,1)-PYI3AU(BETSU,EPST,1)-   
     &  PYI3AU(BETSU,EPSU,1)    
        W3TSUI=PYI3AU(BETSU,EPSH,2)-PYI3AU(BETSU,EPST,2)-   
     &  PYI3AU(BETSU,EPSU,2)    
        W3TUSR=PYI3AU(BETUS,EPSH,1)-PYI3AU(BETUS,EPST,1)-   
     &  PYI3AU(BETUS,EPSS,1)    
        W3TUSI=PYI3AU(BETUS,EPSH,2)-PYI3AU(BETUS,EPST,2)-   
     &  PYI3AU(BETUS,EPSS,2)    
        W3USTR=PYI3AU(BEUST,EPSH,1)-PYI3AU(BEUST,EPSU,1)-   
     &  PYI3AU(BEUST,EPST,1)    
        W3USTI=PYI3AU(BEUST,EPSH,2)-PYI3AU(BEUST,EPSU,2)-   
     &  PYI3AU(BEUST,EPST,2)    
        W3UTSR=PYI3AU(BEUTS,EPSH,1)-PYI3AU(BEUTS,EPSU,1)-   
     &  PYI3AU(BEUTS,EPSS,1)    
        W3UTSI=PYI3AU(BEUTS,EPSH,2)-PYI3AU(BEUTS,EPSU,2)-   
     &  PYI3AU(BEUTS,EPSS,2)    
        B2STUR=SQMQ/SQMH**2*(SH*(UH-SH)/(SH+UH)+2.*TH*UH*(UH+2.*SH)/    
     &  (SH+UH)**2*(PYW1AU(EPST,1)-PYW1AU(EPSH,1))+(SQMQ-SH/4.)*    
     &  (0.5*PYW2AU(EPSS,1)+0.5*PYW2AU(EPSH,1)-PYW2AU(EPST,1)+W3STUR)+  
     &  SH**2*(2.*SQMQ/(SH+UH)**2-0.5/(SH+UH))*(PYW2AU(EPST,1)- 
     &  PYW2AU(EPSH,1))+0.5*TH*UH/SH*(PYW2AU(EPSH,1)-2.*PYW2AU(EPST,1))+    
     &  0.125*(SH-12.*SQMQ-4.*TH*UH/SH)*W3TSUR) 
        B2STUI=SQMQ/SQMH**2*(2.*TH*UH*(UH+2.*SH)/(SH+UH)**2*    
     &  (PYW1AU(EPST,2)-PYW1AU(EPSH,2))+(SQMQ-SH/4.)*   
     &  (0.5*PYW2AU(EPSS,2)+0.5*PYW2AU(EPSH,2)-PYW2AU(EPST,2)+W3STUI)+  
     &  SH**2*(2.*SQMQ/(SH+UH)**2-0.5/(SH+UH))*(PYW2AU(EPST,2)- 
     &  PYW2AU(EPSH,2))+0.5*TH*UH/SH*(PYW2AU(EPSH,2)-2.*PYW2AU(EPST,2))+    
     &  0.125*(SH-12.*SQMQ-4.*TH*UH/SH)*W3TSUI) 
        B2SUTR=SQMQ/SQMH**2*(SH*(TH-SH)/(SH+TH)+2.*UH*TH*(TH+2.*SH)/    
     &  (SH+TH)**2*(PYW1AU(EPSU,1)-PYW1AU(EPSH,1))+(SQMQ-SH/4.)*    
     &  (0.5*PYW2AU(EPSS,1)+0.5*PYW2AU(EPSH,1)-PYW2AU(EPSU,1)+W3SUTR)+  
     &  SH**2*(2.*SQMQ/(SH+TH)**2-0.5/(SH+TH))*(PYW2AU(EPSU,1)- 
     &  PYW2AU(EPSH,1))+0.5*UH*TH/SH*(PYW2AU(EPSH,1)-2.*PYW2AU(EPSU,1))+    
     &  0.125*(SH-12.*SQMQ-4.*UH*TH/SH)*W3USTR) 
        B2SUTI=SQMQ/SQMH**2*(2.*UH*TH*(TH+2.*SH)/(SH+TH)**2*    
     &  (PYW1AU(EPSU,2)-PYW1AU(EPSH,2))+(SQMQ-SH/4.)*   
     &  (0.5*PYW2AU(EPSS,2)+0.5*PYW2AU(EPSH,2)-PYW2AU(EPSU,2)+W3SUTI)+  
     &  SH**2*(2.*SQMQ/(SH+TH)**2-0.5/(SH+TH))*(PYW2AU(EPSU,2)- 
     &  PYW2AU(EPSH,2))+0.5*UH*TH/SH*(PYW2AU(EPSH,2)-2.*PYW2AU(EPSU,2))+    
     &  0.125*(SH-12.*SQMQ-4.*UH*TH/SH)*W3USTI) 
        B2TSUR=SQMQ/SQMH**2*(TH*(UH-TH)/(TH+UH)+2.*SH*UH*(UH+2.*TH)/    
     &  (TH+UH)**2*(PYW1AU(EPSS,1)-PYW1AU(EPSH,1))+(SQMQ-TH/4.)*    
     &  (0.5*PYW2AU(EPST,1)+0.5*PYW2AU(EPSH,1)-PYW2AU(EPSS,1)+W3TSUR)+  
     &  TH**2*(2.*SQMQ/(TH+UH)**2-0.5/(TH+UH))*(PYW2AU(EPSS,1)- 
     &  PYW2AU(EPSH,1))+0.5*SH*UH/TH*(PYW2AU(EPSH,1)-2.*PYW2AU(EPSS,1))+    
     &  0.125*(TH-12.*SQMQ-4.*SH*UH/TH)*W3STUR) 
        B2TSUI=SQMQ/SQMH**2*(2.*SH*UH*(UH+2.*TH)/(TH+UH)**2*    
     &  (PYW1AU(EPSS,2)-PYW1AU(EPSH,2))+(SQMQ-TH/4.)*   
     &  (0.5*PYW2AU(EPST,2)+0.5*PYW2AU(EPSH,2)-PYW2AU(EPSS,2)+W3TSUI)+  
     &  TH**2*(2.*SQMQ/(TH+UH)**2-0.5/(TH+UH))*(PYW2AU(EPSS,2)- 
     &  PYW2AU(EPSH,2))+0.5*SH*UH/TH*(PYW2AU(EPSH,2)-2.*PYW2AU(EPSS,2))+    
     &  0.125*(TH-12.*SQMQ-4.*SH*UH/TH)*W3STUI) 
        B2TUSR=SQMQ/SQMH**2*(TH*(SH-TH)/(TH+SH)+2.*UH*SH*(SH+2.*TH)/    
     &  (TH+SH)**2*(PYW1AU(EPSU,1)-PYW1AU(EPSH,1))+(SQMQ-TH/4.)*    
     &  (0.5*PYW2AU(EPST,1)+0.5*PYW2AU(EPSH,1)-PYW2AU(EPSU,1)+W3TUSR)+  
     &  TH**2*(2.*SQMQ/(TH+SH)**2-0.5/(TH+SH))*(PYW2AU(EPSU,1)- 
     &  PYW2AU(EPSH,1))+0.5*UH*SH/TH*(PYW2AU(EPSH,1)-2.*PYW2AU(EPSU,1))+    
     &  0.125*(TH-12.*SQMQ-4.*UH*SH/TH)*W3UTSR) 
        B2TUSI=SQMQ/SQMH**2*(2.*UH*SH*(SH+2.*TH)/(TH+SH)**2*    
     &  (PYW1AU(EPSU,2)-PYW1AU(EPSH,2))+(SQMQ-TH/4.)*   
     &  (0.5*PYW2AU(EPST,2)+0.5*PYW2AU(EPSH,2)-PYW2AU(EPSU,2)+W3TUSI)+  
     &  TH**2*(2.*SQMQ/(TH+SH)**2-0.5/(TH+SH))*(PYW2AU(EPSU,2)- 
     &  PYW2AU(EPSH,2))+0.5*UH*SH/TH*(PYW2AU(EPSH,2)-2.*PYW2AU(EPSU,2))+    
     &  0.125*(TH-12.*SQMQ-4.*UH*SH/TH)*W3UTSI) 
        B2USTR=SQMQ/SQMH**2*(UH*(TH-UH)/(UH+TH)+2.*SH*TH*(TH+2.*UH)/    
     &  (UH+TH)**2*(PYW1AU(EPSS,1)-PYW1AU(EPSH,1))+(SQMQ-UH/4.)*    
     &  (0.5*PYW2AU(EPSU,1)+0.5*PYW2AU(EPSH,1)-PYW2AU(EPSS,1)+W3USTR)+  
     &  UH**2*(2.*SQMQ/(UH+TH)**2-0.5/(UH+TH))*(PYW2AU(EPSS,1)- 
     &  PYW2AU(EPSH,1))+0.5*SH*TH/UH*(PYW2AU(EPSH,1)-2.*PYW2AU(EPSS,1))+    
     &  0.125*(UH-12.*SQMQ-4.*SH*TH/UH)*W3SUTR) 
        B2USTI=SQMQ/SQMH**2*(2.*SH*TH*(TH+2.*UH)/(UH+TH)**2*    
     &  (PYW1AU(EPSS,2)-PYW1AU(EPSH,2))+(SQMQ-UH/4.)*   
     &  (0.5*PYW2AU(EPSU,2)+0.5*PYW2AU(EPSH,2)-PYW2AU(EPSS,2)+W3USTI)+  
     &  UH**2*(2.*SQMQ/(UH+TH)**2-0.5/(UH+TH))*(PYW2AU(EPSS,2)- 
     &  PYW2AU(EPSH,2))+0.5*SH*TH/UH*(PYW2AU(EPSH,2)-2.*PYW2AU(EPSS,2))+    
     &  0.125*(UH-12.*SQMQ-4.*SH*TH/UH)*W3SUTI) 
        B2UTSR=SQMQ/SQMH**2*(UH*(SH-UH)/(UH+SH)+2.*TH*SH*(SH+2.*UH)/    
     &  (UH+SH)**2*(PYW1AU(EPST,1)-PYW1AU(EPSH,1))+(SQMQ-UH/4.)*    
     &  (0.5*PYW2AU(EPSU,1)+0.5*PYW2AU(EPSH,1)-PYW2AU(EPST,1)+W3UTSR)+  
     &  UH**2*(2.*SQMQ/(UH+SH)**2-0.5/(UH+SH))*(PYW2AU(EPST,1)- 
     &  PYW2AU(EPSH,1))+0.5*TH*SH/UH*(PYW2AU(EPSH,1)-2.*PYW2AU(EPST,1))+    
     &  0.125*(UH-12.*SQMQ-4.*TH*SH/UH)*W3TUSR) 
        B2UTSI=SQMQ/SQMH**2*(2.*TH*SH*(SH+2.*UH)/(UH+SH)**2*    
     &  (PYW1AU(EPST,2)-PYW1AU(EPSH,2))+(SQMQ-UH/4.)*   
     &  (0.5*PYW2AU(EPSU,2)+0.5*PYW2AU(EPSH,2)-PYW2AU(EPST,2)+W3UTSI)+  
     &  UH**2*(2.*SQMQ/(UH+SH)**2-0.5/(UH+SH))*(PYW2AU(EPST,2)- 
     &  PYW2AU(EPSH,2))+0.5*TH*SH/UH*(PYW2AU(EPSH,2)-2.*PYW2AU(EPST,2))+    
     &  0.125*(UH-12.*SQMQ-4.*TH*SH/UH)*W3TUSI) 
        B4STUR=SQMQ/SQMH*(-2./3.+(SQMQ/SQMH-1./4.)*(PYW2AU(EPSS,1)- 
     &  PYW2AU(EPSH,1)+W3STUR)) 
        B4STUI=SQMQ/SQMH*(SQMQ/SQMH-1./4.)*(PYW2AU(EPSS,2)- 
     &  PYW2AU(EPSH,2)+W3STUI)  
        B4TUSR=SQMQ/SQMH*(-2./3.+(SQMQ/SQMH-1./4.)*(PYW2AU(EPST,1)- 
     &  PYW2AU(EPSH,1)+W3TUSR)) 
        B4TUSI=SQMQ/SQMH*(SQMQ/SQMH-1./4.)*(PYW2AU(EPST,2)- 
     &  PYW2AU(EPSH,2)+W3TUSI)  
        B4USTR=SQMQ/SQMH*(-2./3.+(SQMQ/SQMH-1./4.)*(PYW2AU(EPSU,1)- 
     &  PYW2AU(EPSH,1)+W3USTR)) 
        B4USTI=SQMQ/SQMH*(SQMQ/SQMH-1./4.)*(PYW2AU(EPSU,2)- 
     &  PYW2AU(EPSH,2)+W3USTI)  
        A2STUR=A2STUR+B2STUR+B2SUTR 
        A2STUI=A2STUI+B2STUI+B2SUTI 
        A2USTR=A2USTR+B2USTR+B2UTSR 
        A2USTI=A2USTI+B2USTI+B2UTSI 
        A2TUSR=A2TUSR+B2TUSR+B2TSUR 
        A2TUSI=A2TUSI+B2TUSI+B2TSUI 
        A4STUR=A4STUR+B4STUR+B4USTR+B4TUSR  
        A4STUI=A4STUI+B4STUI+B4USTI+B4TUSI  
  760   CONTINUE    
        FACGH=COMFAC*FACA*3./(128.*PARU(1)**2)*AEM/XW*AS**3*    
     &  SQMH/SQMW*SQMH**3/(SH*TH*UH)*(A2STUR**2+A2STUI**2+A2USTR**2+    
     &  A2USTI**2+A2TUSR**2+A2TUSI**2+A4STUR**2+A4STUI**2)  
        FACGH=FACGH*WIDS(25,2)  
        IF(KFAC(1,21)*KFAC(2,21).EQ.0) GOTO 770 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGH    
  770   CONTINUE    
    
      ELSEIF(ISUB.EQ.114) THEN  
C...g + g -> gamma + gamma. 
        ASRE=0. 
        ASIM=0. 
        DO 780 I=1,2*MSTP(1)    
        EI=KCHG(IABS(I),1)/3.   
        SQMQ=PMAS(I,1)**2   
        EPSS=4.*SQMQ/SH 
        EPST=4.*SQMQ/TH 
        EPSU=4.*SQMQ/UH 
        IF(EPSS+ABS(EPST)+ABS(EPSU).LT.3.E-6) THEN  
          A0STUR=1.+(TH-UH)/SH*LOG(TH/UH)+0.5*(TH2+UH2)/SH2*    
     &    (LOG(TH/UH)**2+PARU(1)**2)    
          A0STUI=0. 
          A0TSUR=1.+(SH-UH)/TH*LOG(-SH/UH)+0.5*(SH2+UH2)/TH2*   
     &    LOG(-SH/UH)**2    
          A0TSUI=-PARU(1)*((SH-UH)/TH+(SH2+UH2)/TH2*LOG(-SH/UH))    
          A0UTSR=1.+(TH-SH)/UH*LOG(-TH/SH)+0.5*(TH2+SH2)/UH2*   
     &    LOG(-TH/SH)**2    
          A0UTSI=PARU(1)*((TH-SH)/UH+(TH2+SH2)/UH2*LOG(-TH/SH)) 
          A1STUR=-1.    
          A1STUI=0. 
          A2STUR=-1.    
          A2STUI=0. 
        ELSE    
          BESTU=0.5*(1.+SQRT(1.+EPSS*TH/UH))    
          BEUST=0.5*(1.+SQRT(1.+EPSU*SH/TH))    
          BETUS=0.5*(1.+SQRT(1.+EPST*UH/SH))    
          BEUTS=BESTU   
          BETSU=BEUST   
          BESUT=BETUS   
          A0STUR=1.+(1.+2.*TH/SH)*PYW1AU(EPST,1)+(1.+2.*UH/SH)* 
     &    PYW1AU(EPSU,1)+0.5*((TH2+UH2)/SH2-EPSS)*(PYW2AU(EPST,1)+  
     &    PYW2AU(EPSU,1))-0.25*EPST*(1.-0.5*EPSS)*(PYI3AU(BESUT,EPSS,1)+    
     &    PYI3AU(BESUT,EPST,1))-0.25*EPSU*(1.-0.5*EPSS)*    
     &    (PYI3AU(BESTU,EPSS,1)+PYI3AU(BESTU,EPSU,1))+  
     &    0.25*(-2.*(TH2+UH2)/SH2+4.*EPSS+EPST+EPSU+0.5*EPST*EPSU)* 
     &    (PYI3AU(BETSU,EPST,1)+PYI3AU(BETSU,EPSU,1))   
          A0STUI=(1.+2.*TH/SH)*PYW1AU(EPST,2)+(1.+2.*UH/SH)*    
     &    PYW1AU(EPSU,2)+0.5*((TH2+UH2)/SH2-EPSS)*(PYW2AU(EPST,2)+  
     &    PYW2AU(EPSU,2))-0.25*EPST*(1.-0.5*EPSS)*(PYI3AU(BESUT,EPSS,2)+    
     &    PYI3AU(BESUT,EPST,2))-0.25*EPSU*(1.-0.5*EPSS)*    
     &    (PYI3AU(BESTU,EPSS,2)+PYI3AU(BESTU,EPSU,2))+  
     &    0.25*(-2.*(TH2+UH2)/SH2+4.*EPSS+EPST+EPSU+0.5*EPST*EPSU)* 
     &    (PYI3AU(BETSU,EPST,2)+PYI3AU(BETSU,EPSU,2))   
          A0TSUR=1.+(1.+2.*SH/TH)*PYW1AU(EPSS,1)+(1.+2.*UH/TH)* 
     &    PYW1AU(EPSU,1)+0.5*((SH2+UH2)/TH2-EPST)*(PYW2AU(EPSS,1)+  
     &    PYW2AU(EPSU,1))-0.25*EPSS*(1.-0.5*EPST)*(PYI3AU(BETUS,EPST,1)+    
     &    PYI3AU(BETUS,EPSS,1))-0.25*EPSU*(1.-0.5*EPST)*    
     &    (PYI3AU(BETSU,EPST,1)+PYI3AU(BETSU,EPSU,1))+  
     &    0.25*(-2.*(SH2+UH2)/TH2+4.*EPST+EPSS+EPSU+0.5*EPSS*EPSU)* 
     &    (PYI3AU(BESTU,EPSS,1)+PYI3AU(BESTU,EPSU,1))   
          A0TSUI=(1.+2.*SH/TH)*PYW1AU(EPSS,2)+(1.+2.*UH/TH)*    
     &    PYW1AU(EPSU,2)+0.5*((SH2+UH2)/TH2-EPST)*(PYW2AU(EPSS,2)+  
     &    PYW2AU(EPSU,2))-0.25*EPSS*(1.-0.5*EPST)*(PYI3AU(BETUS,EPST,2)+    
     &    PYI3AU(BETUS,EPSS,2))-0.25*EPSU*(1.-0.5*EPST)*    
     &    (PYI3AU(BETSU,EPST,2)+PYI3AU(BETSU,EPSU,2))+  
     &    0.25*(-2.*(SH2+UH2)/TH2+4.*EPST+EPSS+EPSU+0.5*EPSS*EPSU)* 
     &    (PYI3AU(BESTU,EPSS,2)+PYI3AU(BESTU,EPSU,2))   
          A0UTSR=1.+(1.+2.*TH/UH)*PYW1AU(EPST,1)+(1.+2.*SH/UH)* 
     &    PYW1AU(EPSS,1)+0.5*((TH2+SH2)/UH2-EPSU)*(PYW2AU(EPST,1)+  
     &    PYW2AU(EPSS,1))-0.25*EPST*(1.-0.5*EPSU)*(PYI3AU(BEUST,EPSU,1)+    
     &    PYI3AU(BEUST,EPST,1))-0.25*EPSS*(1.-0.5*EPSU)*    
     &    (PYI3AU(BEUTS,EPSU,1)+PYI3AU(BEUTS,EPSS,1))+  
     &    0.25*(-2.*(TH2+SH2)/UH2+4.*EPSU+EPST+EPSS+0.5*EPST*EPSS)* 
     &    (PYI3AU(BETUS,EPST,1)+PYI3AU(BETUS,EPSS,1))   
          A0UTSI=(1.+2.*TH/UH)*PYW1AU(EPST,2)+(1.+2.*SH/UH)*    
     &    PYW1AU(EPSS,2)+0.5*((TH2+SH2)/UH2-EPSU)*(PYW2AU(EPST,2)+  
     &    PYW2AU(EPSS,2))-0.25*EPST*(1.-0.5*EPSU)*(PYI3AU(BEUST,EPSU,2)+    
     &    PYI3AU(BEUST,EPST,2))-0.25*EPSS*(1.-0.5*EPSU)*    
     &    (PYI3AU(BEUTS,EPSU,2)+PYI3AU(BEUTS,EPSS,2))+  
     &    0.25*(-2.*(TH2+SH2)/UH2+4.*EPSU+EPST+EPSS+0.5*EPST*EPSS)* 
     &    (PYI3AU(BETUS,EPST,2)+PYI3AU(BETUS,EPSS,2))   
          A1STUR=-1.-0.25*(EPSS+EPST+EPSU)*(PYW2AU(EPSS,1)+ 
     &    PYW2AU(EPST,1)+PYW2AU(EPSU,1))+0.25*(EPSU+0.5*EPSS*EPST)* 
     &    (PYI3AU(BESUT,EPSS,1)+PYI3AU(BESUT,EPST,1))+  
     &    0.25*(EPST+0.5*EPSS*EPSU)*(PYI3AU(BESTU,EPSS,1)+  
     &    PYI3AU(BESTU,EPSU,1))+0.25*(EPSS+0.5*EPST*EPSU)*  
     &    (PYI3AU(BETSU,EPST,1)+PYI3AU(BETSU,EPSU,1))   
          A1STUI=-0.25*(EPSS+EPST+EPSU)*(PYW2AU(EPSS,2)+PYW2AU(EPST,2)+ 
     &    PYW2AU(EPSU,2))+0.25*(EPSU+0.5*EPSS*EPST)*    
     &    (PYI3AU(BESUT,EPSS,2)+PYI3AU(BESUT,EPST,2))+  
     &    0.25*(EPST+0.5*EPSS*EPSU)*(PYI3AU(BESTU,EPSS,2)+  
     &    PYI3AU(BESTU,EPSU,2))+0.25*(EPSS+0.5*EPST*EPSU)*  
     &    (PYI3AU(BETSU,EPST,2)+PYI3AU(BETSU,EPSU,2))   
          A2STUR=-1.+0.125*EPSS*EPST*(PYI3AU(BESUT,EPSS,1)+ 
     &    PYI3AU(BESUT,EPST,1))+0.125*EPSS*EPSU*(PYI3AU(BESTU,EPSS,1)+  
     &    PYI3AU(BESTU,EPSU,1))+0.125*EPST*EPSU*(PYI3AU(BETSU,EPST,1)+  
     &    PYI3AU(BETSU,EPSU,1)) 
          A2STUI=0.125*EPSS*EPST*(PYI3AU(BESUT,EPSS,2)+ 
     &    PYI3AU(BESUT,EPST,2))+0.125*EPSS*EPSU*(PYI3AU(BESTU,EPSS,2)+  
     &    PYI3AU(BESTU,EPSU,2))+0.125*EPST*EPSU*(PYI3AU(BETSU,EPST,2)+  
     &    PYI3AU(BETSU,EPSU,2)) 
        ENDIF   
        ASRE=ASRE+EI**2*(A0STUR+A0TSUR+A0UTSR+4.*A1STUR+A2STUR) 
        ASIM=ASIM+EI**2*(A0STUI+A0TSUI+A0UTSI+4.*A1STUI+A2STUI) 
  780   CONTINUE    
        FACGG=COMFAC*FACA/(8.*PARU(1)**2)*AS**2*AEM**2*(ASRE**2+ASIM**2)    
        IF(KFAC(1,21)*KFAC(2,21).EQ.0) GOTO 790 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACGG    
  790   CONTINUE    
    
      ELSEIF(ISUB.EQ.115) THEN  
C...g + g -> gamma + Z0.    
    
      ELSEIF(ISUB.EQ.116) THEN  
C...g + g -> Z0 + Z0.   
    
      ELSEIF(ISUB.EQ.117) THEN  
C...g + g -> W+ + W-.   
    
      ENDIF 
    
C...G: 2 -> 3, tree diagrams.   
    
      ELSEIF(ISUB.LE.140) THEN  
      IF(ISUB.EQ.121) THEN  
C...g + g -> f + fb + H0.   
    
      ENDIF 
    
C...H: 2 -> 1, tree diagrams, non-standard model processes. 
    
      ELSEIF(ISUB.LE.160) THEN  
      IF(ISUB.EQ.141) THEN  
C...f + fb -> gamma*/Z0/Z'0.    
        MINT(61)=2  
        CALL PYWIDT(32,SQRT(SH),WDTP,WDTE)  
        FACZP=COMFAC*AEM**2*4./9.   
        DO 800 I=MINA,MAXA  
        IF(I.EQ.0.OR.KFAC(1,I)*KFAC(2,-I).EQ.0) GOTO 800    
        EI=KCHG(IABS(I),1)/3.   
        AI=SIGN(1.,EI)  
        VI=AI-4.*EI*XW  
        API=SIGN(1.,EI) 
        VPI=API-4.*EI*XW    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACZP*(EI**2*VINT(111)+EI*VI/(8.*XW*(1.-XW))*    
     &  SH*(SH-SQMZ)/((SH-SQMZ)**2+GMMZ**2)*VINT(112)+EI*VPI/(8.*XW*    
     &  (1.-XW))*SH*(SH-SQMZP)/((SH-SQMZP)**2+GMMZP**2)*VINT(113)+  
     &  (VI**2+AI**2)/(16.*XW*(1.-XW))**2*SH2/((SH-SQMZ)**2+GMMZ**2)*   
     &  VINT(114)+2.*(VI*VPI+AI*API)/(16.*XW*(1.-XW))**2*SH2*   
     &  ((SH-SQMZ)*(SH-SQMZP)+GMMZ*GMMZP)/(((SH-SQMZ)**2+GMMZ**2)*  
     &  ((SH-SQMZP)**2+GMMZP**2))*VINT(115)+(VPI**2+API**2)/    
     &  (16.*XW*(1.-XW))**2*SH2/((SH-SQMZP)**2+GMMZP**2)*VINT(116)) 
  800   CONTINUE    
    
      ELSEIF(ISUB.EQ.142) THEN  
C...f + fb' -> H+/-.    
        CALL PYWIDT(37,SQRT(SH),WDTP,WDTE)  
        FHC=COMFAC*(AEM/XW)**2*1./48.*(SH/SQMW)**2*SH2/ 
     &  ((SH-SQMHC)**2+GMMHC**2)    
C'''No construction yet for leptons 
        DO 840 I=1,MSTP(54)/2   
        IL=2*I-1    
        IU=2*I  
        RMQL=PMAS(IL,1)**2/SH   
        RMQU=PMAS(IU,1)**2/SH   
        FACHC=FHC*((RMQL*PARU(121)+RMQU/PARU(121))*(1.-RMQL-RMQU)-  
     &  4.*RMQL*RMQU)/SQRT(MAX(0.,(1.-RMQL-RMQU)**2-4.*RMQL*RMQU))  
        IF(KFAC(1,IL)*KFAC(2,-IU).EQ.0) GOTO 810    
        KCHHC=(KCHG(IL,1)-KCHG(IU,1))/3 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=IL 
        ISIG(NCHN,2)=-IU    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHC*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4))  
  810   IF(KFAC(1,-IL)*KFAC(2,IU).EQ.0) GOTO 820    
        KCHHC=(-KCHG(IL,1)+KCHG(IU,1))/3    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=-IL    
        ISIG(NCHN,2)=IU 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHC*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4))  
  820   IF(KFAC(1,IU)*KFAC(2,-IL).EQ.0) GOTO 830    
        KCHHC=(KCHG(IU,1)-KCHG(IL,1))/3 
        NCHN=NCHN+1 
        ISIG(NCHN,1)=IU 
        ISIG(NCHN,2)=-IL    
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHC*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4))  
  830   IF(KFAC(1,-IU)*KFAC(2,IL).EQ.0) GOTO 840    
        KCHHC=(-KCHG(IU,1)+KCHG(IL,1))/3    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=-IU    
        ISIG(NCHN,2)=IL 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHC*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4))  
  840   CONTINUE    
    
      ELSEIF(ISUB.EQ.143) THEN  
C...f + fb -> R.    
        CALL PYWIDT(40,SQRT(SH),WDTP,WDTE)  
        FACR=COMFAC*(AEM/XW)**2*1./9.*SH2/((SH-SQMR)**2+GMMR**2)    
        DO 860 I=MIN1,MAX1  
        IF(I.EQ.0.OR.KFAC(1,I).EQ.0) GOTO 860   
        IA=IABS(I)  
        DO 850 J=MIN2,MAX2  
        IF(J.EQ.0.OR.KFAC(2,J).EQ.0) GOTO 850   
        JA=IABS(J)  
        IF(I*J.GT.0.OR.IABS(IA-JA).NE.2) GOTO 850   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=J  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACR*(WDTE(0,1)+WDTE(0,(10-(I+J))/4)+WDTE(0,4))  
  850   CONTINUE    
  860   CONTINUE    
    
      ENDIF 
    
C...I: 2 -> 2, tree diagrams, non-standard model processes. 
    
      ELSE  
      IF(ISUB.EQ.161) THEN  

clin-7/2018 add "CALL PYWIDT()" to get rid of compiler warning message;
c     however, expect this statement not to be reached:
        CALL PYWIDT(40,SQRT(SH),WDTP,WDTE)  
C...f + g -> f' + H+/- (q + g -> q' + H+/- only).   
c     if reached, write a message to standard output and then stop the run:
        write(6,*) 'ISUB=161 reached: check arguments of CALL PYWIDT()'
        stop
clin-7/2018-end

        FHCQ=COMFAC*FACA*AS*AEM/XW*1./24    
        DO 900 I=1,MSTP(54) 
        IU=I+MOD(I,2)   
        SQMQ=PMAS(IU,1)**2  
        FACHCQ=FHCQ/PARU(121)*SQMQ/SQMW*(SH/(SQMQ-UH)+  
     &  2.*SQMQ*(SQMHC-UH)/(SQMQ-UH)**2+(SQMQ-UH)/SH+   
     &  2.*SQMQ/(SQMQ-UH)+2.*(SQMHC-UH)/(SQMQ-UH)*(SQMHC-SQMQ-SH)/SH)   
        IF(KFAC(1,-I)*KFAC(2,21).EQ.0) GOTO 870 
        KCHHC=ISIGN(1,-KCHG(I,1))   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=-I 
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHCQ*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4)) 
  870   IF(KFAC(1,I)*KFAC(2,21).EQ.0) GOTO 880  
        KCHHC=ISIGN(1,KCHG(I,1))    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=I  
        ISIG(NCHN,2)=21 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHCQ*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4)) 
  880   IF(KFAC(1,21)*KFAC(2,-I).EQ.0) GOTO 890 
        KCHHC=ISIGN(1,-KCHG(I,1))   
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=-I 
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHCQ*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4)) 
  890   IF(KFAC(1,21)*KFAC(2,I).EQ.0) GOTO 900  
        KCHHC=ISIGN(1,KCHG(I,1))    
        NCHN=NCHN+1 
        ISIG(NCHN,1)=21 
        ISIG(NCHN,2)=I  
        ISIG(NCHN,3)=1  
        SIGH(NCHN)=FACHCQ*(WDTE(0,1)+WDTE(0,(5-KCHHC)/2)+WDTE(0,4)) 
  900   CONTINUE    
    
      ENDIF 
      ENDIF 
    
C...Multiply with structure functions.  
      IF(ISUB.LE.90.OR.ISUB.GE.96) THEN 
        DO 910 ICHN=1,NCHN  
        IF(MINT(41).EQ.2) THEN  
          KFL1=ISIG(ICHN,1) 
          IF(KFL1.EQ.21) KFL1=0 
          SIGH(ICHN)=SIGH(ICHN)*XSFX(1,KFL1)    
        ENDIF   
        IF(MINT(42).EQ.2) THEN  
          KFL2=ISIG(ICHN,2) 
          IF(KFL2.EQ.21) KFL2=0 
          SIGH(ICHN)=SIGH(ICHN)*XSFX(2,KFL2)    
        ENDIF   
  910   SIGS=SIGS+SIGH(ICHN)    
      ENDIF 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYSTFU(KF,X,Q2,XPQ,JBT)    

C                        *******JBT specifies beam or target of the particle
C...Gives proton and pi+ parton structure functions according to a few  
C...different parametrizations. Note that what is coded is x times the  
C...probability distribution, i.e. xq(x,Q2) etc.    
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
      SAVE /HPARNT/
      COMMON/hjcrdn/YP(3,300),YT(3,300)
      SAVE /hjcrdn/
C                        ********COMMON BLOCK FROM HIJING
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      DIMENSION XPQ(-6:6),XQ(6),TX(6),TT(6),TS(6),NEHLQ(8,2),   
     &CEHLQ(6,6,2,8,2),CDO(3,6,5,2),COW(3,5,4,2)    
    
C...The following data lines are coefficients needed in the 
C...Eichten, Hinchliffe, Lane, Quigg proton structure function  
C...parametrizations, see below.    
C...Powers of 1-x in different cases.   
      DATA NEHLQ/3,4,7,5,7,7,7,7,3,4,7,6,7,7,7,7/   
C...Expansion coefficients for up valence quark distribution.   
      DATA (((CEHLQ(IX,IT,NX,1,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 7.677E-01,-2.087E-01,-3.303E-01,-2.517E-02,-1.570E-02,-1.000E-04,    
     2-5.326E-01,-2.661E-01, 3.201E-01, 1.192E-01, 2.434E-02, 7.620E-03,    
     3 2.162E-01, 1.881E-01,-8.375E-02,-6.515E-02,-1.743E-02,-5.040E-03,    
     4-9.211E-02,-9.952E-02, 1.373E-02, 2.506E-02, 8.770E-03, 2.550E-03,    
     5 3.670E-02, 4.409E-02, 9.600E-04,-7.960E-03,-3.420E-03,-1.050E-03,    
     6-1.549E-02,-2.026E-02,-3.060E-03, 2.220E-03, 1.240E-03, 4.100E-04,    
     1 2.395E-01, 2.905E-01, 9.778E-02, 2.149E-02, 3.440E-03, 5.000E-04,    
     2 1.751E-02,-6.090E-03,-2.687E-02,-1.916E-02,-7.970E-03,-2.750E-03,    
     3-5.760E-03,-5.040E-03, 1.080E-03, 2.490E-03, 1.530E-03, 7.500E-04,    
     4 1.740E-03, 1.960E-03, 3.000E-04,-3.400E-04,-2.900E-04,-1.800E-04,    
     5-5.300E-04,-6.400E-04,-1.700E-04, 4.000E-05, 6.000E-05, 4.000E-05,    
     6 1.700E-04, 2.200E-04, 8.000E-05, 1.000E-05,-1.000E-05,-1.000E-05/    
      DATA (((CEHLQ(IX,IT,NX,1,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 7.237E-01,-2.189E-01,-2.995E-01,-1.909E-02,-1.477E-02, 2.500E-04,    
     2-5.314E-01,-2.425E-01, 3.283E-01, 1.119E-01, 2.223E-02, 7.070E-03,    
     3 2.289E-01, 1.890E-01,-9.859E-02,-6.900E-02,-1.747E-02,-5.080E-03,    
     4-1.041E-01,-1.084E-01, 2.108E-02, 2.975E-02, 9.830E-03, 2.830E-03,    
     5 4.394E-02, 5.116E-02,-1.410E-03,-1.055E-02,-4.230E-03,-1.270E-03,    
     6-1.991E-02,-2.539E-02,-2.780E-03, 3.430E-03, 1.720E-03, 5.500E-04,    
     1 2.410E-01, 2.884E-01, 9.369E-02, 1.900E-02, 2.530E-03, 2.400E-04,    
     2 1.765E-02,-9.220E-03,-3.037E-02,-2.085E-02,-8.440E-03,-2.810E-03,    
     3-6.450E-03,-5.260E-03, 1.720E-03, 3.110E-03, 1.830E-03, 8.700E-04,    
     4 2.120E-03, 2.320E-03, 2.600E-04,-4.900E-04,-3.900E-04,-2.300E-04,    
     5-6.900E-04,-8.200E-04,-2.000E-04, 7.000E-05, 9.000E-05, 6.000E-05,    
     6 2.400E-04, 3.100E-04, 1.100E-04, 0.000E+00,-2.000E-05,-2.000E-05/    
C...Expansion coefficients for down valence quark distribution. 
      DATA (((CEHLQ(IX,IT,NX,2,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 3.813E-01,-8.090E-02,-1.634E-01,-2.185E-02,-8.430E-03,-6.200E-04,    
     2-2.948E-01,-1.435E-01, 1.665E-01, 6.638E-02, 1.473E-02, 4.080E-03,    
     3 1.252E-01, 1.042E-01,-4.722E-02,-3.683E-02,-1.038E-02,-2.860E-03,    
     4-5.478E-02,-5.678E-02, 8.900E-03, 1.484E-02, 5.340E-03, 1.520E-03,    
     5 2.220E-02, 2.567E-02,-3.000E-05,-4.970E-03,-2.160E-03,-6.500E-04,    
     6-9.530E-03,-1.204E-02,-1.510E-03, 1.510E-03, 8.300E-04, 2.700E-04,    
     1 1.261E-01, 1.354E-01, 3.958E-02, 8.240E-03, 1.660E-03, 4.500E-04,    
     2 3.890E-03,-1.159E-02,-1.625E-02,-9.610E-03,-3.710E-03,-1.260E-03,    
     3-1.910E-03,-5.600E-04, 1.590E-03, 1.590E-03, 8.400E-04, 3.900E-04,    
     4 6.400E-04, 4.900E-04,-1.500E-04,-2.900E-04,-1.800E-04,-1.000E-04,    
     5-2.000E-04,-1.900E-04, 0.000E+00, 6.000E-05, 4.000E-05, 3.000E-05,    
     6 7.000E-05, 8.000E-05, 2.000E-05,-1.000E-05,-1.000E-05,-1.000E-05/    
      DATA (((CEHLQ(IX,IT,NX,2,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 3.578E-01,-8.622E-02,-1.480E-01,-1.840E-02,-7.820E-03,-4.500E-04,    
     2-2.925E-01,-1.304E-01, 1.696E-01, 6.243E-02, 1.353E-02, 3.750E-03,    
     3 1.318E-01, 1.041E-01,-5.486E-02,-3.872E-02,-1.038E-02,-2.850E-03,    
     4-6.162E-02,-6.143E-02, 1.303E-02, 1.740E-02, 5.940E-03, 1.670E-03,    
     5 2.643E-02, 2.957E-02,-1.490E-03,-6.450E-03,-2.630E-03,-7.700E-04,    
     6-1.218E-02,-1.497E-02,-1.260E-03, 2.240E-03, 1.120E-03, 3.500E-04,    
     1 1.263E-01, 1.334E-01, 3.732E-02, 7.070E-03, 1.260E-03, 3.400E-04,    
     2 3.660E-03,-1.357E-02,-1.795E-02,-1.031E-02,-3.880E-03,-1.280E-03,    
     3-2.100E-03,-3.600E-04, 2.050E-03, 1.920E-03, 9.800E-04, 4.400E-04,    
     4 7.700E-04, 5.400E-04,-2.400E-04,-3.900E-04,-2.400E-04,-1.300E-04,    
     5-2.600E-04,-2.300E-04, 2.000E-05, 9.000E-05, 6.000E-05, 4.000E-05,    
     6 9.000E-05, 1.000E-04, 2.000E-05,-2.000E-05,-2.000E-05,-1.000E-05/    
C...Expansion coefficients for up and down sea quark distributions. 
      DATA (((CEHLQ(IX,IT,NX,3,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 6.870E-02,-6.861E-02, 2.973E-02,-5.400E-03, 3.780E-03,-9.700E-04,    
     2-1.802E-02, 1.400E-04, 6.490E-03,-8.540E-03, 1.220E-03,-1.750E-03,    
     3-4.650E-03, 1.480E-03,-5.930E-03, 6.000E-04,-1.030E-03,-8.000E-05,    
     4 6.440E-03, 2.570E-03, 2.830E-03, 1.150E-03, 7.100E-04, 3.300E-04,    
     5-3.930E-03,-2.540E-03,-1.160E-03,-7.700E-04,-3.600E-04,-1.900E-04,    
     6 2.340E-03, 1.930E-03, 5.300E-04, 3.700E-04, 1.600E-04, 9.000E-05,    
     1 1.014E+00,-1.106E+00, 3.374E-01,-7.444E-02, 8.850E-03,-8.700E-04,    
     2 9.233E-01,-1.285E+00, 4.475E-01,-9.786E-02, 1.419E-02,-1.120E-03,    
     3 4.888E-02,-1.271E-01, 8.606E-02,-2.608E-02, 4.780E-03,-6.000E-04,    
     4-2.691E-02, 4.887E-02,-1.771E-02, 1.620E-03, 2.500E-04,-6.000E-05,    
     5 7.040E-03,-1.113E-02, 1.590E-03, 7.000E-04,-2.000E-04, 0.000E+00,    
     6-1.710E-03, 2.290E-03, 3.800E-04,-3.500E-04, 4.000E-05, 1.000E-05/    
      DATA (((CEHLQ(IX,IT,NX,3,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 1.008E-01,-7.100E-02, 1.973E-02,-5.710E-03, 2.930E-03,-9.900E-04,    
     2-5.271E-02,-1.823E-02, 1.792E-02,-6.580E-03, 1.750E-03,-1.550E-03,    
     3 1.220E-02, 1.763E-02,-8.690E-03,-8.800E-04,-1.160E-03,-2.100E-04,    
     4-1.190E-03,-7.180E-03, 2.360E-03, 1.890E-03, 7.700E-04, 4.100E-04,    
     5-9.100E-04, 2.040E-03,-3.100E-04,-1.050E-03,-4.000E-04,-2.400E-04,    
     6 1.190E-03,-1.700E-04,-2.000E-04, 4.200E-04, 1.700E-04, 1.000E-04,    
     1 1.081E+00,-1.189E+00, 3.868E-01,-8.617E-02, 1.115E-02,-1.180E-03,    
     2 9.917E-01,-1.396E+00, 4.998E-01,-1.159E-01, 1.674E-02,-1.720E-03,    
     3 5.099E-02,-1.338E-01, 9.173E-02,-2.885E-02, 5.890E-03,-6.500E-04,    
     4-3.178E-02, 5.703E-02,-2.070E-02, 2.440E-03, 1.100E-04,-9.000E-05,    
     5 8.970E-03,-1.392E-02, 2.050E-03, 6.500E-04,-2.300E-04, 2.000E-05,    
     6-2.340E-03, 3.010E-03, 5.000E-04,-3.900E-04, 6.000E-05, 1.000E-05/    
C...Expansion coefficients for gluon distribution.  
      DATA (((CEHLQ(IX,IT,NX,4,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 9.482E-01,-9.578E-01, 1.009E-01,-1.051E-01, 3.456E-02,-3.054E-02,    
     2-9.627E-01, 5.379E-01, 3.368E-01,-9.525E-02, 1.488E-02,-2.051E-02,    
     3 4.300E-01,-8.306E-02,-3.372E-01, 4.902E-02,-9.160E-03, 1.041E-02,    
     4-1.925E-01,-1.790E-02, 2.183E-01, 7.490E-03, 4.140E-03,-1.860E-03,    
     5 8.183E-02, 1.926E-02,-1.072E-01,-1.944E-02,-2.770E-03,-5.200E-04,    
     6-3.884E-02,-1.234E-02, 5.410E-02, 1.879E-02, 3.350E-03, 1.040E-03,    
     1 2.948E+01,-3.902E+01, 1.464E+01,-3.335E+00, 5.054E-01,-5.915E-02,    
     2 2.559E+01,-3.955E+01, 1.661E+01,-4.299E+00, 6.904E-01,-8.243E-02,    
     3-1.663E+00, 1.176E+00, 1.118E+00,-7.099E-01, 1.948E-01,-2.404E-02,    
     4-2.168E-01, 8.170E-01,-7.169E-01, 1.851E-01,-1.924E-02,-3.250E-03,    
     5 2.088E-01,-4.355E-01, 2.239E-01,-2.446E-02,-3.620E-03, 1.910E-03,    
     6-9.097E-02, 1.601E-01,-5.681E-02,-2.500E-03, 2.580E-03,-4.700E-04/    
      DATA (((CEHLQ(IX,IT,NX,4,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 2.367E+00, 4.453E-01, 3.660E-01, 9.467E-02, 1.341E-01, 1.661E-02,    
     2-3.170E+00,-1.795E+00, 3.313E-02,-2.874E-01,-9.827E-02,-7.119E-02,    
     3 1.823E+00, 1.457E+00,-2.465E-01, 3.739E-02, 6.090E-03, 1.814E-02,    
     4-1.033E+00,-9.827E-01, 2.136E-01, 1.169E-01, 5.001E-02, 1.684E-02,    
     5 5.133E-01, 5.259E-01,-1.173E-01,-1.139E-01,-4.988E-02,-2.021E-02,    
     6-2.881E-01,-3.145E-01, 5.667E-02, 9.161E-02, 4.568E-02, 1.951E-02,    
     1 3.036E+01,-4.062E+01, 1.578E+01,-3.699E+00, 6.020E-01,-7.031E-02,    
     2 2.700E+01,-4.167E+01, 1.770E+01,-4.804E+00, 7.862E-01,-1.060E-01,    
     3-1.909E+00, 1.357E+00, 1.127E+00,-7.181E-01, 2.232E-01,-2.481E-02,    
     4-2.488E-01, 9.781E-01,-8.127E-01, 2.094E-01,-2.997E-02,-4.710E-03,    
     5 2.506E-01,-5.427E-01, 2.672E-01,-3.103E-02,-1.800E-03, 2.870E-03,    
     6-1.128E-01, 2.087E-01,-6.972E-02,-2.480E-03, 2.630E-03,-8.400E-04/    
C...Expansion coefficients for strange sea quark distribution.  
      DATA (((CEHLQ(IX,IT,NX,5,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 4.968E-02,-4.173E-02, 2.102E-02,-3.270E-03, 3.240E-03,-6.700E-04,    
     2-6.150E-03,-1.294E-02, 6.740E-03,-6.890E-03, 9.000E-04,-1.510E-03,    
     3-8.580E-03, 5.050E-03,-4.900E-03,-1.600E-04,-9.400E-04,-1.500E-04,    
     4 7.840E-03, 1.510E-03, 2.220E-03, 1.400E-03, 7.000E-04, 3.500E-04,    
     5-4.410E-03,-2.220E-03,-8.900E-04,-8.500E-04,-3.600E-04,-2.000E-04,    
     6 2.520E-03, 1.840E-03, 4.100E-04, 3.900E-04, 1.600E-04, 9.000E-05,    
     1 9.235E-01,-1.085E+00, 3.464E-01,-7.210E-02, 9.140E-03,-9.100E-04,    
     2 9.315E-01,-1.274E+00, 4.512E-01,-9.775E-02, 1.380E-02,-1.310E-03,    
     3 4.739E-02,-1.296E-01, 8.482E-02,-2.642E-02, 4.760E-03,-5.700E-04,    
     4-2.653E-02, 4.953E-02,-1.735E-02, 1.750E-03, 2.800E-04,-6.000E-05,    
     5 6.940E-03,-1.132E-02, 1.480E-03, 6.500E-04,-2.100E-04, 0.000E+00,    
     6-1.680E-03, 2.340E-03, 4.200E-04,-3.400E-04, 5.000E-05, 1.000E-05/    
      DATA (((CEHLQ(IX,IT,NX,5,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 6.478E-02,-4.537E-02, 1.643E-02,-3.490E-03, 2.710E-03,-6.700E-04,    
     2-2.223E-02,-2.126E-02, 1.247E-02,-6.290E-03, 1.120E-03,-1.440E-03,    
     3-1.340E-03, 1.362E-02,-6.130E-03,-7.900E-04,-9.000E-04,-2.000E-04,    
     4 5.080E-03,-3.610E-03, 1.700E-03, 1.830E-03, 6.800E-04, 4.000E-04,    
     5-3.580E-03, 6.000E-05,-2.600E-04,-1.050E-03,-3.800E-04,-2.300E-04,    
     6 2.420E-03, 9.300E-04,-1.000E-04, 4.500E-04, 1.700E-04, 1.100E-04,    
     1 9.868E-01,-1.171E+00, 3.940E-01,-8.459E-02, 1.124E-02,-1.250E-03,    
     2 1.001E+00,-1.383E+00, 5.044E-01,-1.152E-01, 1.658E-02,-1.830E-03,    
     3 4.928E-02,-1.368E-01, 9.021E-02,-2.935E-02, 5.800E-03,-6.600E-04,    
     4-3.133E-02, 5.785E-02,-2.023E-02, 2.630E-03, 1.600E-04,-8.000E-05,    
     5 8.840E-03,-1.416E-02, 1.900E-03, 5.800E-04,-2.500E-04, 1.000E-05,    
     6-2.300E-03, 3.080E-03, 5.500E-04,-3.700E-04, 7.000E-05, 1.000E-05/    
C...Expansion coefficients for charm sea quark distribution.    
      DATA (((CEHLQ(IX,IT,NX,6,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 9.270E-03,-1.817E-02, 9.590E-03,-6.390E-03, 1.690E-03,-1.540E-03,    
     2 5.710E-03,-1.188E-02, 6.090E-03,-4.650E-03, 1.240E-03,-1.310E-03,    
     3-3.960E-03, 7.100E-03,-3.590E-03, 1.840E-03,-3.900E-04, 3.400E-04,    
     4 1.120E-03,-1.960E-03, 1.120E-03,-4.800E-04, 1.000E-04,-4.000E-05,    
     5 4.000E-05,-3.000E-05,-1.800E-04, 9.000E-05,-5.000E-05,-2.000E-05,    
     6-4.200E-04, 7.300E-04,-1.600E-04, 5.000E-05, 5.000E-05, 5.000E-05,    
     1 8.098E-01,-1.042E+00, 3.398E-01,-6.824E-02, 8.760E-03,-9.000E-04,    
     2 8.961E-01,-1.217E+00, 4.339E-01,-9.287E-02, 1.304E-02,-1.290E-03,    
     3 3.058E-02,-1.040E-01, 7.604E-02,-2.415E-02, 4.600E-03,-5.000E-04,    
     4-2.451E-02, 4.432E-02,-1.651E-02, 1.430E-03, 1.200E-04,-1.000E-04,    
     5 1.122E-02,-1.457E-02, 2.680E-03, 5.800E-04,-1.200E-04, 3.000E-05,    
     6-7.730E-03, 7.330E-03,-7.600E-04,-2.400E-04, 1.000E-05, 0.000E+00/    
      DATA (((CEHLQ(IX,IT,NX,6,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 9.980E-03,-1.945E-02, 1.055E-02,-6.870E-03, 1.860E-03,-1.560E-03,    
     2 5.700E-03,-1.203E-02, 6.250E-03,-4.860E-03, 1.310E-03,-1.370E-03,    
     3-4.490E-03, 7.990E-03,-4.170E-03, 2.050E-03,-4.400E-04, 3.300E-04,    
     4 1.470E-03,-2.480E-03, 1.460E-03,-5.700E-04, 1.200E-04,-1.000E-05,    
     5-9.000E-05, 1.500E-04,-3.200E-04, 1.200E-04,-6.000E-05,-4.000E-05,    
     6-4.200E-04, 7.600E-04,-1.400E-04, 4.000E-05, 7.000E-05, 5.000E-05,    
     1 8.698E-01,-1.131E+00, 3.836E-01,-8.111E-02, 1.048E-02,-1.300E-03,    
     2 9.626E-01,-1.321E+00, 4.854E-01,-1.091E-01, 1.583E-02,-1.700E-03,    
     3 3.057E-02,-1.088E-01, 8.022E-02,-2.676E-02, 5.590E-03,-5.600E-04,    
     4-2.845E-02, 5.164E-02,-1.918E-02, 2.210E-03,-4.000E-05,-1.500E-04,    
     5 1.311E-02,-1.751E-02, 3.310E-03, 5.100E-04,-1.200E-04, 5.000E-05,    
     6-8.590E-03, 8.380E-03,-9.200E-04,-2.600E-04, 1.000E-05,-1.000E-05/    
C...Expansion coefficients for bottom sea quark distribution.   
      DATA (((CEHLQ(IX,IT,NX,7,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 9.010E-03,-1.401E-02, 7.150E-03,-4.130E-03, 1.260E-03,-1.040E-03,    
     2 6.280E-03,-9.320E-03, 4.780E-03,-2.890E-03, 9.100E-04,-8.200E-04,    
     3-2.930E-03, 4.090E-03,-1.890E-03, 7.600E-04,-2.300E-04, 1.400E-04,    
     4 3.900E-04,-1.200E-03, 4.400E-04,-2.500E-04, 2.000E-05,-2.000E-05,    
     5 2.600E-04, 1.400E-04,-8.000E-05, 1.000E-04, 1.000E-05, 1.000E-05,    
     6-2.600E-04, 3.200E-04, 1.000E-05,-1.000E-05, 1.000E-05,-1.000E-05,    
     1 8.029E-01,-1.075E+00, 3.792E-01,-7.843E-02, 1.007E-02,-1.090E-03,    
     2 7.903E-01,-1.099E+00, 4.153E-01,-9.301E-02, 1.317E-02,-1.410E-03,    
     3-1.704E-02,-1.130E-02, 2.882E-02,-1.341E-02, 3.040E-03,-3.600E-04,    
     4-7.200E-04, 7.230E-03,-5.160E-03, 1.080E-03,-5.000E-05,-4.000E-05,    
     5 3.050E-03,-4.610E-03, 1.660E-03,-1.300E-04,-1.000E-05, 1.000E-05,    
     6-4.360E-03, 5.230E-03,-1.610E-03, 2.000E-04,-2.000E-05, 0.000E+00/    
      DATA (((CEHLQ(IX,IT,NX,7,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 8.980E-03,-1.459E-02, 7.510E-03,-4.410E-03, 1.310E-03,-1.070E-03,    
     2 5.970E-03,-9.440E-03, 4.800E-03,-3.020E-03, 9.100E-04,-8.500E-04,    
     3-3.050E-03, 4.440E-03,-2.100E-03, 8.500E-04,-2.400E-04, 1.400E-04,    
     4 5.300E-04,-1.300E-03, 5.600E-04,-2.700E-04, 3.000E-05,-2.000E-05,    
     5 2.000E-04, 1.400E-04,-1.100E-04, 1.000E-04, 0.000E+00, 0.000E+00,    
     6-2.600E-04, 3.200E-04, 0.000E+00,-3.000E-05, 1.000E-05,-1.000E-05,    
     1 8.672E-01,-1.174E+00, 4.265E-01,-9.252E-02, 1.244E-02,-1.460E-03,    
     2 8.500E-01,-1.194E+00, 4.630E-01,-1.083E-01, 1.614E-02,-1.830E-03,    
     3-2.241E-02,-5.630E-03, 2.815E-02,-1.425E-02, 3.520E-03,-4.300E-04,    
     4-7.300E-04, 8.030E-03,-5.780E-03, 1.380E-03,-1.300E-04,-4.000E-05,    
     5 3.460E-03,-5.380E-03, 1.960E-03,-2.100E-04, 1.000E-05, 1.000E-05,    
     6-4.850E-03, 5.950E-03,-1.890E-03, 2.600E-04,-3.000E-05, 0.000E+00/    
C...Expansion coefficients for top sea quark distribution.  
      DATA (((CEHLQ(IX,IT,NX,8,1),IX=1,6),IT=1,6),NX=1,2)/  
     1 4.410E-03,-7.480E-03, 3.770E-03,-2.580E-03, 7.300E-04,-7.100E-04,    
     2 3.840E-03,-6.050E-03, 3.030E-03,-2.030E-03, 5.800E-04,-5.900E-04,    
     3-8.800E-04, 1.660E-03,-7.500E-04, 4.700E-04,-1.000E-04, 1.000E-04,    
     4-8.000E-05,-1.500E-04, 1.200E-04,-9.000E-05, 3.000E-05, 0.000E+00,    
     5 1.300E-04,-2.200E-04,-2.000E-05,-2.000E-05,-2.000E-05,-2.000E-05,    
     6-7.000E-05, 1.900E-04,-4.000E-05, 2.000E-05, 0.000E+00, 0.000E+00,    
     1 6.623E-01,-9.248E-01, 3.519E-01,-7.930E-02, 1.110E-02,-1.180E-03,    
     2 6.380E-01,-9.062E-01, 3.582E-01,-8.479E-02, 1.265E-02,-1.390E-03,    
     3-2.581E-02, 2.125E-02, 4.190E-03,-4.980E-03, 1.490E-03,-2.100E-04,    
     4 7.100E-04, 5.300E-04,-1.270E-03, 3.900E-04,-5.000E-05,-1.000E-05,    
     5 3.850E-03,-5.060E-03, 1.860E-03,-3.500E-04, 4.000E-05, 0.000E+00,    
     6-3.530E-03, 4.460E-03,-1.500E-03, 2.700E-04,-3.000E-05, 0.000E+00/    
      DATA (((CEHLQ(IX,IT,NX,8,2),IX=1,6),IT=1,6),NX=1,2)/  
     1 4.260E-03,-7.530E-03, 3.830E-03,-2.680E-03, 7.600E-04,-7.300E-04,    
     2 3.640E-03,-6.050E-03, 3.030E-03,-2.090E-03, 5.900E-04,-6.000E-04,    
     3-9.200E-04, 1.710E-03,-8.200E-04, 5.000E-04,-1.200E-04, 1.000E-04,    
     4-5.000E-05,-1.600E-04, 1.300E-04,-9.000E-05, 3.000E-05, 0.000E+00,    
     5 1.300E-04,-2.100E-04,-1.000E-05,-2.000E-05,-2.000E-05,-1.000E-05,    
     6-8.000E-05, 1.800E-04,-5.000E-05, 2.000E-05, 0.000E+00, 0.000E+00,    
     1 7.146E-01,-1.007E+00, 3.932E-01,-9.246E-02, 1.366E-02,-1.540E-03,    
     2 6.856E-01,-9.828E-01, 3.977E-01,-9.795E-02, 1.540E-02,-1.790E-03,    
     3-3.053E-02, 2.758E-02, 2.150E-03,-4.880E-03, 1.640E-03,-2.500E-04,    
     4 9.200E-04, 4.200E-04,-1.340E-03, 4.600E-04,-8.000E-05,-1.000E-05,    
     5 4.230E-03,-5.660E-03, 2.140E-03,-4.300E-04, 6.000E-05, 0.000E+00,    
     6-3.890E-03, 5.000E-03,-1.740E-03, 3.300E-04,-4.000E-05, 0.000E+00/    
    
C...The following data lines are coefficients needed in the 
C...Duke, Owens proton structure function parametrizations, see below.  
C...Expansion coefficients for (up+down) valence quark distribution.    
      DATA ((CDO(IP,IS,1,1),IS=1,6),IP=1,3)/    
     1 4.190E-01, 3.460E+00, 4.400E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2 4.000E-03, 7.240E-01,-4.860E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     3-7.000E-03,-6.600E-02, 1.330E+00, 0.000E+00, 0.000E+00, 0.000E+00/    
      DATA ((CDO(IP,IS,1,2),IS=1,6),IP=1,3)/    
     1 3.740E-01, 3.330E+00, 6.030E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2 1.400E-02, 7.530E-01,-6.220E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     3 0.000E+00,-7.600E-02, 1.560E+00, 0.000E+00, 0.000E+00, 0.000E+00/    
C...Expansion coefficients for down valence quark distribution. 
      DATA ((CDO(IP,IS,2,1),IS=1,6),IP=1,3)/    
     1 7.630E-01, 4.000E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2-2.370E-01, 6.270E-01,-4.210E-01, 0.000E+00, 0.000E+00, 0.000E+00,    
     3 2.600E-02,-1.900E-02, 3.300E-02, 0.000E+00, 0.000E+00, 0.000E+00/    
      DATA ((CDO(IP,IS,2,2),IS=1,6),IP=1,3)/    
     1 7.610E-01, 3.830E+00, 0.000E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2-2.320E-01, 6.270E-01,-4.180E-01, 0.000E+00, 0.000E+00, 0.000E+00,    
     3 2.300E-02,-1.900E-02, 3.600E-02, 0.000E+00, 0.000E+00, 0.000E+00/    
C...Expansion coefficients for (up+down+strange) sea quark distribution.    
      DATA ((CDO(IP,IS,3,1),IS=1,6),IP=1,3)/    
     1 1.265E+00, 0.000E+00, 8.050E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2-1.132E+00,-3.720E-01, 1.590E+00, 6.310E+00,-1.050E+01, 1.470E+01,    
     3 2.930E-01,-2.900E-02,-1.530E-01,-2.730E-01,-3.170E+00, 9.800E+00/    
      DATA ((CDO(IP,IS,3,2),IS=1,6),IP=1,3)/    
     1 1.670E+00, 0.000E+00, 9.150E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2-1.920E+00,-2.730E-01, 5.300E-01, 1.570E+01,-1.010E+02, 2.230E+02,    
     3 5.820E-01,-1.640E-01,-7.630E-01,-2.830E+00, 4.470E+01,-1.170E+02/    
C...Expansion coefficients for charm sea quark distribution.    
      DATA ((CDO(IP,IS,4,1),IS=1,6),IP=1,3)/    
     1 0.000E+00,-3.600E-02, 6.350E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2 1.350E-01,-2.220E-01, 3.260E+00,-3.030E+00, 1.740E+01,-1.790E+01,    
     3-7.500E-02,-5.800E-02,-9.090E-01, 1.500E+00,-1.130E+01, 1.560E+01/    
       DATA ((CDO(IP,IS,4,2),IS=1,6),IP=1,3)/   
     1 0.000E+00,-1.200E-01, 3.510E+00, 0.000E+00, 0.000E+00, 0.000E+00,    
     2 6.700E-02,-2.330E-01, 3.660E+00,-4.740E-01, 9.500E+00,-1.660E+01,    
     3-3.100E-02,-2.300E-02,-4.530E-01, 3.580E-01,-5.430E+00, 1.550E+01/    
C...Expansion coefficients for gluon distribution.  
      DATA ((CDO(IP,IS,5,1),IS=1,6),IP=1,3)/    
     1 1.560E+00, 0.000E+00, 6.000E+00, 9.000E+00, 0.000E+00, 0.000E+00,    
     2-1.710E+00,-9.490E-01, 1.440E+00,-7.190E+00,-1.650E+01, 1.530E+01,    
     3 6.380E-01, 3.250E-01,-1.050E+00, 2.550E-01, 1.090E+01,-1.010E+01/    
      DATA ((CDO(IP,IS,5,2),IS=1,6),IP=1,3)/    
     1 8.790E-01, 0.000E+00, 4.000E+00, 9.000E+00, 0.000E+00, 0.000E+00,    
     2-9.710E-01,-1.160E+00, 1.230E+00,-5.640E+00,-7.540E+00,-5.960E-01,    
     3 4.340E-01, 4.760E-01,-2.540E-01,-8.170E-01, 5.500E+00, 1.260E-01/    
    
C...The following data lines are coefficients needed in the 
C...Owens pion structure function parametrizations, see below.  
C...Expansion coefficients for up and down valence quark distributions. 
      DATA ((COW(IP,IS,1,1),IS=1,5),IP=1,3)/    
     1  4.0000E-01,  7.0000E-01,  0.0000E+00,  0.0000E+00,  0.0000E+00, 
     2 -6.2120E-02,  6.4780E-01,  0.0000E+00,  0.0000E+00,  0.0000E+00, 
     3 -7.1090E-03,  1.3350E-02,  0.0000E+00,  0.0000E+00,  0.0000E+00/ 
      DATA ((COW(IP,IS,1,2),IS=1,5),IP=1,3)/    
     1  4.0000E-01,  6.2800E-01,  0.0000E+00,  0.0000E+00,  0.0000E+00, 
     2 -5.9090E-02,  6.4360E-01,  0.0000E+00,  0.0000E+00,  0.0000E+00, 
     3 -6.5240E-03,  1.4510E-02,  0.0000E+00,  0.0000E+00,  0.0000E+00/ 
C...Expansion coefficients for gluon distribution.  
      DATA ((COW(IP,IS,2,1),IS=1,5),IP=1,3)/    
     1  8.8800E-01,  0.0000E+00,  3.1100E+00,  6.0000E+00,  0.0000E+00, 
     2 -1.8020E+00, -1.5760E+00, -1.3170E-01,  2.8010E+00, -1.7280E+01, 
     3  1.8120E+00,  1.2000E+00,  5.0680E-01, -1.2160E+01,  2.0490E+01/ 
      DATA ((COW(IP,IS,2,2),IS=1,5),IP=1,3)/    
     1  7.9400E-01,  0.0000E+00,  2.8900E+00,  6.0000E+00,  0.0000E+00, 
     2 -9.1440E-01, -1.2370E+00,  5.9660E-01, -3.6710E+00, -8.1910E+00, 
     3  5.9660E-01,  6.5820E-01, -2.5500E-01, -2.3040E+00,  7.7580E+00/ 
C...Expansion coefficients for (up+down+strange) quark sea distribution.    
      DATA ((COW(IP,IS,3,1),IS=1,5),IP=1,3)/    
     1  9.0000E-01,  0.0000E+00,  5.0000E+00,  0.0000E+00,  0.0000E+00, 
     2 -2.4280E-01, -2.1200E-01,  8.6730E-01,  1.2660E+00,  2.3820E+00, 
     3  1.3860E-01,  3.6710E-03,  4.7470E-02, -2.2150E+00,  3.4820E-01/ 
      DATA ((COW(IP,IS,3,2),IS=1,5),IP=1,3)/    
     1  9.0000E-01,  0.0000E+00,  5.0000E+00,  0.0000E+00,  0.0000E+00, 
     2 -1.4170E-01, -1.6970E-01, -2.4740E+00, -2.5340E+00,  5.6210E-01, 
     3 -1.7400E-01, -9.6230E-02,  1.5750E+00,  1.3780E+00, -2.7010E-01/ 
C...Expansion coefficients for charm quark sea distribution.    
      DATA ((COW(IP,IS,4,1),IS=1,5),IP=1,3)/    
     1  0.0000E+00, -2.2120E-02,  2.8940E+00,  0.0000E+00,  0.0000E+00, 
     2  7.9280E-02, -3.7850E-01,  9.4330E+00,  5.2480E+00,  8.3880E+00, 
     3 -6.1340E-02, -1.0880E-01, -1.0852E+01, -7.1870E+00, -1.1610E+01/ 
      DATA ((COW(IP,IS,4,2),IS=1,5),IP=1,3)/    
     1  0.0000E+00, -8.8200E-02,  1.9240E+00,  0.0000E+00,  0.0000E+00, 
     2  6.2290E-02, -2.8920E-01,  2.4240E-01, -4.4630E+00, -8.3670E-01, 
     3 -4.0990E-02, -1.0820E-01,  2.0360E+00,  5.2090E+00, -4.8400E-02/ 

C...Euler's beta function, requires ordinary Gamma function 
clin-10/25/02 get rid of argument usage mismatch in PYGAMM():
c      EULBT(X,Y)=PYGAMM(X)*PYGAMM(Y)/PYGAMM(X+Y)
    
C...Reset structure functions, check x and hadron flavour.  
      ALAM=0.   
      DO 100 KFL=-6,6   
  100 XPQ(KFL)=0.   
      IF(X.LT.0..OR.X.GT.1.) THEN   
        WRITE(MSTU(11),1000) X  
        RETURN  
      ENDIF 
      KFA=IABS(KF)  
      IF(KFA.NE.211.AND.KFA.NE.2212.AND.KFA.NE.2112) THEN   
        WRITE(MSTU(11),1100) KF 
        RETURN  
      ENDIF 
    
C...Call user-supplied structure function. Select proton/neutron/pion.  
      IF(MSTP(51).EQ.0.OR.MSTP(52).GE.2) THEN   
        KFE=KFA 
        IF(KFA.EQ.2112) KFE=2212    
        CALL PYSTFE(KFE,X,Q2,XPQ)   
        GOTO 230    
      ENDIF 
      IF(KFA.EQ.211) GOTO 200   
    
      IF(MSTP(51).EQ.1.OR.MSTP(51).EQ.2) THEN   
C...Proton structure functions from Eichten, Hinchliffe, Lane, Quigg.   
C...Allowed variable range: 5 GeV2 < Q2 < 1E8 GeV2; 1E-4 < x < 1    
    
C...Determine set, Lamdba and x and t expansion variables.  
        NSET=MSTP(51)   
        IF(NSET.EQ.1) ALAM=0.2  
        IF(NSET.EQ.2) ALAM=0.29 
        TMIN=LOG(5./ALAM**2)    
        TMAX=LOG(1E8/ALAM**2)   
        IF(MSTP(52).EQ.0) THEN  
          T=TMIN    
        ELSE    
          T=LOG(Q2/ALAM**2) 
        ENDIF   
        VT=MAX(-1.,MIN(1.,(2.*T-TMAX-TMIN)/(TMAX-TMIN)))    
        NX=1    
        IF(X.LE.0.1) NX=2   
        IF(NX.EQ.1) VX=(2.*X-1.1)/0.9   
        IF(NX.EQ.2) VX=MAX(-1.,(2.*LOG(X)+11.51293)/6.90776)    
        CXS=1.  
        IF(X.LT.1E-4.AND.ABS(PARP(51)-1.).GT.0.01) CXS= 
     &  (1E-4/X)**(PARP(51)-1.) 
    
C...Chebyshev polynomials for x and t expansion.    
        TX(1)=1.    
        TX(2)=VX    
        TX(3)=2.*VX**2-1.   
        TX(4)=4.*VX**3-3.*VX    
        TX(5)=8.*VX**4-8.*VX**2+1.  
        TX(6)=16.*VX**5-20.*VX**3+5.*VX 
        TT(1)=1.    
        TT(2)=VT    
        TT(3)=2.*VT**2-1.   
        TT(4)=4.*VT**3-3.*VT    
        TT(5)=8.*VT**4-8.*VT**2+1.  
        TT(6)=16.*VT**5-20.*VT**3+5.*VT 
    
C...Calculate structure functions.  
        DO 120 KFL=1,6  
        XQSUM=0.    
        DO 110 IT=1,6   
        DO 110 IX=1,6   
  110   XQSUM=XQSUM+CEHLQ(IX,IT,NX,KFL,NSET)*TX(IX)*TT(IT)  
  120   XQ(KFL)=XQSUM*(1.-X)**NEHLQ(KFL,NSET)*CXS   
    
C...Put into output array.  
        XPQ(0)=XQ(4)    
        XPQ(1)=XQ(2)+XQ(3)  
        XPQ(2)=XQ(1)+XQ(3)  
        XPQ(3)=XQ(5)    
        XPQ(4)=XQ(6)    
        XPQ(-1)=XQ(3)   
        XPQ(-2)=XQ(3)   
        XPQ(-3)=XQ(5)   
        XPQ(-4)=XQ(6)   
    
C...Special expansion for bottom (thresh effects).   
        IF(MSTP(54).GE.5) THEN  
          IF(NSET.EQ.1) TMIN=8.1905 
          IF(NSET.EQ.2) TMIN=7.4474 
          IF(T.LE.TMIN) GOTO 140    
          VT=MAX(-1.,MIN(1.,(2.*T-TMAX-TMIN)/(TMAX-TMIN)))  
          TT(1)=1.  
          TT(2)=VT  
          TT(3)=2.*VT**2-1. 
          TT(4)=4.*VT**3-3.*VT  
          TT(5)=8.*VT**4-8.*VT**2+1.    
          TT(6)=16.*VT**5-20.*VT**3+5.*VT   
          XQSUM=0.  
          DO 130 IT=1,6 
          DO 130 IX=1,6 
  130     XQSUM=XQSUM+CEHLQ(IX,IT,NX,7,NSET)*TX(IX)*TT(IT)  
          XPQ(5)=XQSUM*(1.-X)**NEHLQ(7,NSET)    
          XPQ(-5)=XPQ(5)    
  140     CONTINUE  
        ENDIF   
    
C...Special expansion for top (thresh effects).  
        IF(MSTP(54).GE.6) THEN  
          IF(NSET.EQ.1) TMIN=11.5528    
          IF(NSET.EQ.2) TMIN=10.8097    
          TMIN=TMIN+2.*LOG(PMAS(6,1)/30.)   
          TMAX=TMAX+2.*LOG(PMAS(6,1)/30.)   
          IF(T.LE.TMIN) GOTO 160    
          VT=MAX(-1.,MIN(1.,(2.*T-TMAX-TMIN)/(TMAX-TMIN)))  
          TT(1)=1.  
          TT(2)=VT  
          TT(3)=2.*VT**2-1. 
          TT(4)=4.*VT**3-3.*VT  
          TT(5)=8.*VT**4-8.*VT**2+1.    
          TT(6)=16.*VT**5-20.*VT**3+5.*VT   
          XQSUM=0.  
          DO 150 IT=1,6 
          DO 150 IX=1,6 
  150     XQSUM=XQSUM+CEHLQ(IX,IT,NX,8,NSET)*TX(IX)*TT(IT)  
          XPQ(6)=XQSUM*(1.-X)**NEHLQ(8,NSET)    
          XPQ(-6)=XPQ(6)    
  160     CONTINUE  
        ENDIF   
    
      ELSEIF(MSTP(51).EQ.3.OR.MSTP(51).EQ.4) THEN   
C...Proton structure functions from Duke, Owens.    
C...Allowed variable range: 4 GeV2 < Q2 < approx 1E6 GeV2.  
    
C...Determine set, Lambda and s expansion parameter.    
        NSET=MSTP(51)-2 
        IF(NSET.EQ.1) ALAM=0.2  
        IF(NSET.EQ.2) ALAM=0.4  
        IF(MSTP(52).LE.0) THEN  
          SD=0. 
        ELSE    
          SD=LOG(LOG(MAX(Q2,4.)/ALAM**2)/LOG(4./ALAM**2))   
        ENDIF   
    
C...Calculate structure functions.  
        DO 180 KFL=1,5  
        DO 170 IS=1,6   
  170   TS(IS)=CDO(1,IS,KFL,NSET)+CDO(2,IS,KFL,NSET)*SD+    
     &  CDO(3,IS,KFL,NSET)*SD**2    
        IF(KFL.LE.2) THEN   

clin-10/25/02 evaluate EULBT(TS(1),TS(2)+1.):
c          XQ(KFL)=X**TS(1)*(1.-X)**TS(2)*(1.+TS(3)*X)/(EULBT(TS(1),    
c     &    TS(2)+1.)*(1.+TS(3)*TS(1)/(TS(1)+TS(2)+1.)))  
           eulbt1=PYGAMM(TS(1))*PYGAMM(TS(2)+1.)/PYGAMM(TS(1)+TS(2)+1.)
           XQ(KFL)=X**TS(1)*(1.-X)**TS(2)*(1.+TS(3)*X)/(EULBT1
     &          *(1.+TS(3)*TS(1)/(TS(1)+TS(2)+1.)))  
        ELSE    
           XQ(KFL)=TS(1)*X**TS(2)*(1.-X)**TS(3)*(1.+TS(4)*X+TS(5)*X**2+  
     &    TS(6)*X**3)   
        ENDIF   


  180   CONTINUE    
    
C...Put into output arrays. 
        XPQ(0)=XQ(5)    
        XPQ(1)=XQ(2)+XQ(3)/6.   
        XPQ(2)=3.*XQ(1)-XQ(2)+XQ(3)/6.  
        XPQ(3)=XQ(3)/6. 
        XPQ(4)=XQ(4)    
        XPQ(-1)=XQ(3)/6.    
        XPQ(-2)=XQ(3)/6.    
        XPQ(-3)=XQ(3)/6.    
        XPQ(-4)=XQ(4)   
    
C...Proton structure functions from Diemoz, Ferroni, Longo, Martinelli. 
C...These are accessed via PYSTFE since the files needed may not always 
C...available.  
      ELSEIF(MSTP(51).GE.11.AND.MSTP(51).LE.13) THEN    
        CALL PYSTFE(2212,X,Q2,XPQ)  
    
C...Unknown proton parametrization. 
      ELSE  
        WRITE(MSTU(11),1200) MSTP(51)   
      ENDIF 
      GOTO 230  
    
  200 IF((MSTP(51).GE.1.AND.MSTP(51).LE.4).OR.  
     &(MSTP(51).GE.11.AND.MSTP(51).LE.13)) THEN 
C...Pion structure functions from Owens.    
C...Allowed variable range: 4 GeV2 < Q2 < approx 2000 GeV2. 
    
C...Determine set, Lambda and s expansion variable. 
        NSET=1  
        IF(MSTP(51).EQ.2.OR.MSTP(51).EQ.4.OR.MSTP(51).EQ.13) NSET=2 
        IF(NSET.EQ.1) ALAM=0.2  
        IF(NSET.EQ.2) ALAM=0.4  
        IF(MSTP(52).LE.0) THEN  
          SD=0. 
        ELSE    
          SD=LOG(LOG(MAX(Q2,4.)/ALAM**2)/LOG(4./ALAM**2))   
        ENDIF   
    
C...Calculate structure functions.  
        DO 220 KFL=1,4  
        DO 210 IS=1,5   
  210   TS(IS)=COW(1,IS,KFL,NSET)+COW(2,IS,KFL,NSET)*SD+    
     &  COW(3,IS,KFL,NSET)*SD**2    
        IF(KFL.EQ.1) THEN   

clin-10/25/02 get rid of argument usage mismatch in PYGAMM():
c          XQ(KFL)=X**TS(1)*(1.-X)**TS(2)/EULBT(TS(1),TS(2)+1.) 
           eulbt2=PYGAMM(TS(1))*PYGAMM(TS(2)+1.)/PYGAMM(TS(1)+TS(2)+1.)
           XQ(KFL)=X**TS(1)*(1.-X)**TS(2)/EULBT2
        ELSE    
          XQ(KFL)=TS(1)*X**TS(2)*(1.-X)**TS(3)*(1.+TS(4)*X+TS(5)*X**2)  
        ENDIF   
  220   CONTINUE    
    
C...Put into output arrays. 
        XPQ(0)=XQ(2)    
        XPQ(1)=XQ(3)/6. 
        XPQ(2)=XQ(1)+XQ(3)/6.   
        XPQ(3)=XQ(3)/6. 
        XPQ(4)=XQ(4)    
        XPQ(-1)=XQ(1)+XQ(3)/6.  
        XPQ(-2)=XQ(3)/6.    
        XPQ(-3)=XQ(3)/6.    
        XPQ(-4)=XQ(4)   
    
C...Unknown pion parametrization.   
      ELSE  
        WRITE(MSTU(11),1200) MSTP(51)   
      ENDIF 
    
C...Isospin conjugation for neutron, charge conjugation for antipart.   
  230 IF(KFA.EQ.2112) THEN  
        XPS=XPQ(1)  
        XPQ(1)=XPQ(2)   
        XPQ(2)=XPS  
        XPS=XPQ(-1) 
        XPQ(-1)=XPQ(-2) 
        XPQ(-2)=XPS 
      ENDIF 
      IF(KF.LT.0) THEN  
        DO 240 KFL=1,4  
        XPS=XPQ(KFL)    
        XPQ(KFL)=XPQ(-KFL)  
  240   XPQ(-KFL)=XPS   
      ENDIF 
    
C...Check positivity and reset above maximum allowed flavour.   
      DO 250 KFL=-6,6   
      XPQ(KFL)=MAX(0.,XPQ(KFL)) 
  250 IF(IABS(KFL).GT.MSTP(54)) XPQ(KFL)=0. 

C...consider nuclear effect on the structure function
              IF((JBT.NE.1.AND.JBT.NE.2).OR.IHPR2(6).EQ.0
     &                  .OR.IHNT2(16).EQ.1) GO TO 400
              ATNM=IHNT2(2*JBT-1)
              IF(ATNM.LE.1.0) GO TO 400
              IF(JBT.EQ.1) THEN
               BBR2=(YP(1,IHNT2(11))**2+YP(2,IHNT2(11))**2)/1.44/
     1              ATNM**0.66666
              ELSEIF(JBT.EQ.2) THEN
               BBR2=(YT(1,IHNT2(12))**2+YT(2,IHNT2(12))**2)/1.44/
     1              ATNM**0.66666
              ENDIF
              BBR2=MIN(1.0,BBR2)
        ABX=(ATNM**0.33333333-1.0)
              APX=HIPR1(6)*4.0/3.0*ABX*SQRT(1.0-BBR2)
              AAX=1.192*ALOG(ATNM)**0.1666666
              RRX=AAX*(X**3-1.2*X**2+0.21*X)+1.0
     &           -(APX-1.079*ABX*SQRT(X)/ALOG(ATNM+1.0))
     1           *EXP(-X**2.0/0.01)
              DO 300 KFL=-6,6
                XPQ(KFL)=XPQ(KFL)*RRX
300           CONTINUE
C                        ********consider the nuclear effect on the structure
C                                function which also depends on the impact
C                                parameter of the nuclear reaction

 400          CONTINUE    
C...Formats for error printouts.    
 1000 FORMAT(' Error: x value outside physical range, x =',1P,E12.3)    
 1100 FORMAT(' Error: illegal particle code for structure function,',   
     &' KF =',I5)   
 1200 FORMAT(' Error: bad value of parameter MSTP(51) in PYSTFU,',  
     &' MSTP(51) =',I5) 
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYSPLI(KF,KFLIN,KFLCH,KFLSP)   
    
C...In case of a hadron remnant which is more complicated than just a   
C...quark or a diquark, split it into two (partons or hadron + parton). 
      DIMENSION KFL(3)  
    
C...Preliminaries. Parton composition.  
      KFA=IABS(KF)  
      KFS=ISIGN(1,KF)   
      KFL(1)=MOD(KFA/1000,10)   
      KFL(2)=MOD(KFA/100,10)    
      KFL(3)=MOD(KFA/10,10) 
      KFLR=KFLIN*KFS    
      KFLCH=0   
    
C...Subdivide meson.    
      IF(KFL(1).EQ.0) THEN  
        KFL(2)=KFL(2)*(-1)**KFL(2)  
        KFL(3)=-KFL(3)*(-1)**IABS(KFL(2))   
        IF(KFLR.EQ.KFL(2)) THEN 
          KFLSP=KFL(3)  
        ELSEIF(KFLR.EQ.KFL(3)) THEN 
          KFLSP=KFL(2)  
        ELSEIF(IABS(KFLR).EQ.21.AND.RLU(0).GT.0.5) THEN 
          KFLSP=KFL(2)  
          KFLCH=KFL(3)  
        ELSEIF(IABS(KFLR).EQ.21) THEN   
          KFLSP=KFL(3)  
          KFLCH=KFL(2)  
        ELSEIF(KFLR*KFL(2).GT.0) THEN   
          CALL LUKFDI(-KFLR,KFL(2),KFDUMP,KFLCH)    
          KFLSP=KFL(3)  
        ELSE    
          CALL LUKFDI(-KFLR,KFL(3),KFDUMP,KFLCH)    
          KFLSP=KFL(2)  
        ENDIF   
    
C...Subdivide baryon.   
      ELSE  
        NAGR=0  
        DO 100 J=1,3    
  100   IF(KFLR.EQ.KFL(J)) NAGR=NAGR+1  
        IF(NAGR.GE.1) THEN  
          RAGR=0.00001+(NAGR-0.00002)*RLU(0)    
          IAGR=0    
          DO 110 J=1,3  
          IF(KFLR.EQ.KFL(J)) RAGR=RAGR-1.   
  110     IF(IAGR.EQ.0.AND.RAGR.LE.0.) IAGR=J   
        ELSE    
          IAGR=int(1.00001+2.99998*RLU(0))
        ENDIF   
        ID1=1   
        IF(IAGR.EQ.1) ID1=2 
        IF(IAGR.EQ.1.AND.KFL(3).GT.KFL(2)) ID1=3    
        ID2=6-IAGR-ID1  
        KSP=3   
        IF(MOD(KFA,10).EQ.2.AND.KFL(1).EQ.KFL(2)) THEN  
          IF(IAGR.NE.3.AND.RLU(0).GT.0.25) KSP=1    
        ELSEIF(MOD(KFA,10).EQ.2.AND.KFL(2).GE.KFL(3)) THEN  
          IF(IAGR.NE.1.AND.RLU(0).GT.0.25) KSP=1    
        ELSEIF(MOD(KFA,10).EQ.2) THEN   
          IF(IAGR.EQ.1) KSP=1   
          IF(IAGR.NE.1.AND.RLU(0).GT.0.75) KSP=1    
        ENDIF   
        KFLSP=1000*KFL(ID1)+100*KFL(ID2)+KSP    
        IF(KFLIN.EQ.21) THEN    
          KFLCH=KFL(IAGR)   
        ELSEIF(NAGR.EQ.0.AND.KFLR.GT.0) THEN    
          CALL LUKFDI(-KFLR,KFL(IAGR),KFDUMP,KFLCH) 
        ELSEIF(NAGR.EQ.0) THEN  
          CALL LUKFDI(10000+KFLSP,-KFLR,KFDUMP,KFLCH)   
          KFLSP=KFL(IAGR)   
        ENDIF   
      ENDIF 
    
C...Add on correct sign for result. 
      KFLCH=KFLCH*KFS   
      KFLSP=KFLSP*KFS   
    
      RETURN    
      END   
    
C*********************************************************************  
    
      FUNCTION PYGAMM(X)    
    
C...Gives ordinary Gamma function Gamma(x) for positive, real arguments;    
C...see M. Abramowitz, I. A. Stegun: Handbook of Mathematical Functions 
C...(Dover, 1965) 6.1.36.   
      DIMENSION B(8)    
clin      DATA B/-0.577191652,0.988205891,-0.897056937,0.918206857, 
clin     &-0.756704078,0.482199394,-0.193527818,0.035868343/    
      DATA B/-0.57719165,0.98820589,-0.89705694,0.91820686, 
     &-0.75670408,0.48219939,-0.19352782,0.03586834/    
    
      NX=INT(X) 
      DX=X-NX   
    
      PYGAMM=1. 
      DO 100 I=1,8  
  100 PYGAMM=PYGAMM+B(I)*DX**I  
      IF(X.LT.1.) THEN  
        PYGAMM=PYGAMM/X 
      ELSE  
        DO 110 IX=1,NX-1    
  110   PYGAMM=(X-IX)*PYGAMM    
      ENDIF 
    
      RETURN    
      END   
    
C***********************************************************************    
    
      FUNCTION PYW1AU(EPS,IREIM)    
    
C...Calculates real and imaginary parts of the auxiliary function W1;   
C...see R. K. Ellis, I. Hinchliffe, M. Soldate and J. J. van der Bij,   
C...FERMILAB-Pub-87/100-T, LBL-23504, June, 1987    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
    
clin-8/2014:
c      ASINH(X)=LOG(X+SQRT(X**2+1.)) 
      ACOSH(X)=LOG(X+SQRT(X**2-1.)) 
    
      IF(EPS.LT.0.) THEN    
        W1RE=2.*SQRT(1.-EPS)*ASINH(SQRT(-1./EPS))   
        W1IM=0. 
      ELSEIF(EPS.LT.1.) THEN    
        W1RE=2.*SQRT(1.-EPS)*ACOSH(SQRT(1./EPS))    
        W1IM=-PARU(1)*SQRT(1.-EPS)  
      ELSE  
        W1RE=2.*SQRT(EPS-1.)*ASIN(SQRT(1./EPS)) 
        W1IM=0. 
      ENDIF 
    
      PYW1AU = 0.
      IF(IREIM.EQ.1) PYW1AU=W1RE    
      IF(IREIM.EQ.2) PYW1AU=W1IM    
    
      RETURN    
      END   
    
C***********************************************************************    
    
      FUNCTION PYW2AU(EPS,IREIM)    
    
C...Calculates real and imaginary parts of the auxiliary function W2;   
C...see R. K. Ellis, I. Hinchliffe, M. Soldate and J. J. van der Bij,   
C...FERMILAB-Pub-87/100-T, LBL-23504, June, 1987    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
    
clin-8/2014:
c      ASINH(X)=LOG(X+SQRT(X**2+1.)) 
      ACOSH(X)=LOG(X+SQRT(X**2-1.)) 
    
      IF(EPS.LT.0.) THEN    
        W2RE=4.*(ASINH(SQRT(-1./EPS)))**2   
        W2IM=0. 
      ELSEIF(EPS.LT.1.) THEN    
        W2RE=4.*(ACOSH(SQRT(1./EPS)))**2-PARU(1)**2 
        W2IM=-4.*PARU(1)*ACOSH(SQRT(1./EPS))    
      ELSE  
        W2RE=-4.*(ASIN(SQRT(1./EPS)))**2    
        W2IM=0. 
      ENDIF 
    
cms ... else needed to avoid compiler warning
      PYW2AU = 0.
      IF(IREIM.EQ.1) THEN
        PYW2AU=W2RE    
      ELSEIF(IREIM.EQ.2) THEN
        PYW2AU=W2IM    
      ENDIF

      RETURN    
      END   
    
C***********************************************************************    
    
      FUNCTION PYI3AU(BE,EPS,IREIM) 
    
C...Calculates real and imaginary parts of the auxiliary function I3;   
C...see R. K. Ellis, I. Hinchliffe, M. Soldate and J. J. van der Bij,   
C...FERMILAB-Pub-87/100-T, LBL-23504, June, 1987    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 

cms ... needed to avoid compiler warning
      GA=0.5
      IF(EPS.LT.1.) GA=0.5*(1.+SQRT(1.-EPS))    

      IF(EPS.LT.0.) THEN    
        F3RE=PYSPEN((GA-1.)/(GA+BE-1.),0.,1)-PYSPEN(GA/(GA+BE-1.),0.,1)+    
     &  PYSPEN((BE-GA)/BE,0.,1)-PYSPEN((BE-GA)/(BE-1.),0.,1)+   
     &  (LOG(BE)**2-LOG(BE-1.)**2)/2.+LOG(GA)*LOG((GA+BE-1.)/BE)+   
     &  LOG(GA-1.)*LOG((BE-1.)/(GA+BE-1.))  
        F3IM=0. 
      ELSEIF(EPS.LT.1.) THEN    
        F3RE=PYSPEN((GA-1.)/(GA+BE-1.),0.,1)-PYSPEN(GA/(GA+BE-1.),0.,1)+    
     &  PYSPEN(GA/(GA-BE),0.,1)-PYSPEN((GA-1.)/(GA-BE),0.,1)+   
     &  LOG(GA/(1.-GA))*LOG((GA+BE-1.)/(BE-GA)) 
        F3IM=-PARU(1)*LOG((GA+BE-1.)/(BE-GA))   
      ELSE  
        RSQ=EPS/(EPS-1.+(2.*BE-1.)**2)  
        RCTHE=RSQ*(1.-2.*BE/EPS)    
        RSTHE=SQRT(RSQ-RCTHE**2)    
        RCPHI=RSQ*(1.+2.*(BE-1.)/EPS)   
        RSPHI=SQRT(RSQ-RCPHI**2)    
        R=SQRT(RSQ) 
        THE=ACOS(RCTHE/R)   
        PHI=ACOS(RCPHI/R)   
        F3RE=PYSPEN(RCTHE,RSTHE,1)+PYSPEN(RCTHE,-RSTHE,1)-  
     &  PYSPEN(RCPHI,RSPHI,1)-PYSPEN(RCPHI,-RSPHI,1)+   
     &  (PHI-THE)*(PHI+THE-PARU(1)) 
        F3IM=PYSPEN(RCTHE,RSTHE,2)+PYSPEN(RCTHE,-RSTHE,2)-  
     &  PYSPEN(RCPHI,RSPHI,2)-PYSPEN(RCPHI,-RSPHI,2)    
      ENDIF 

cms ... needed to avoid compiler warning
      PYI3AU = 0.
      IF(IREIM.EQ.1) THEN
        PYI3AU=2./(2.*BE-1.)*F3RE  
      ELSEIF(IREIM.EQ.2) THEN
        PYI3AU=2./(2.*BE-1.)*F3IM  
      ENDIF

      RETURN    
      END   
    
C***********************************************************************    
    
      FUNCTION PYSPEN(XREIN,XIMIN,IREIM)    
    
C...Calculates real and imaginary part of Spence function; see  
C...G. 't Hooft and M. Veltman, Nucl. Phys. B153 (1979) 365.    
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      DIMENSION B(0:14) 
    
      DATA B/   
     & 1.000000E+00,        -5.000000E-01,         1.666667E-01,    
     & 0.000000E+00,        -3.333333E-02,         0.000000E+00,    
     & 2.380952E-02,         0.000000E+00,        -3.333333E-02,    
     & 0.000000E+00,         7.575757E-02,         0.000000E+00,    
     &-2.531135E-01,         0.000000E+00,         1.166667E+00/    
    
      XRE=XREIN 
      XIM=XIMIN
      PYSPEN=0.
      IF(ABS(1.-XRE).LT.1.E-6.AND.ABS(XIM).LT.1.E-6) THEN   
        IF(IREIM.EQ.1) PYSPEN=PARU(1)**2/6. 
        IF(IREIM.EQ.2) PYSPEN=0.    
        RETURN  
      ENDIF 
    
      XMOD=SQRT(XRE**2+XIM**2)  
      IF(XMOD.LT.1.E-6) THEN    
        IF(IREIM.EQ.1) PYSPEN=0.    
        IF(IREIM.EQ.2) PYSPEN=0.    
        RETURN  
      ENDIF 
    
      XARG=SIGN(ACOS(XRE/XMOD),XIM) 
      SP0RE=0.  
      SP0IM=0.  
      SGN=1.    
      IF(XMOD.GT.1.) THEN   
        ALGXRE=LOG(XMOD)    
        ALGXIM=XARG-SIGN(PARU(1),XARG)  
        SP0RE=-PARU(1)**2/6.-(ALGXRE**2-ALGXIM**2)/2.   
        SP0IM=-ALGXRE*ALGXIM    
        SGN=-1. 
        XMOD=1./XMOD    
        XARG=-XARG  
        XRE=XMOD*COS(XARG)  
        XIM=XMOD*SIN(XARG)  
      ENDIF 
      IF(XRE.GT.0.5) THEN   
        ALGXRE=LOG(XMOD)    
        ALGXIM=XARG 
        XRE=1.-XRE  
        XIM=-XIM    
        XMOD=SQRT(XRE**2+XIM**2)    
        XARG=SIGN(ACOS(XRE/XMOD),XIM)   
        ALGYRE=LOG(XMOD)    
        ALGYIM=XARG 
        SP0RE=SP0RE+SGN*(PARU(1)**2/6.-(ALGXRE*ALGYRE-ALGXIM*ALGYIM))   
        SP0IM=SP0IM-SGN*(ALGXRE*ALGYIM+ALGXIM*ALGYRE)   
        SGN=-SGN    
      ENDIF 
    
      XRE=1.-XRE    
      XIM=-XIM  
      XMOD=SQRT(XRE**2+XIM**2)  
      XARG=SIGN(ACOS(XRE/XMOD),XIM) 
      ZRE=-LOG(XMOD)    
      ZIM=-XARG 
    
      SPRE=0.   
      SPIM=0.   
      SAVERE=1. 
      SAVEIM=0. 
      DO 100 I=0,14 
      TERMRE=(SAVERE*ZRE-SAVEIM*ZIM)/FLOAT(I+1) 
      TERMIM=(SAVERE*ZIM+SAVEIM*ZRE)/FLOAT(I+1) 
      SAVERE=TERMRE 
      SAVEIM=TERMIM 
      SPRE=SPRE+B(I)*TERMRE 
  100 SPIM=SPIM+B(I)*TERMIM 
    
cms ... needed to avoid compiler warning
      IF(IREIM.EQ.1) THEN
        PYSPEN=SP0RE+SGN*SPRE  
      ELSEIF(IREIM.EQ.2) THEN
        PYSPEN=SP0IM+SGN*SPIM  
      ENDIF

      RETURN    
      END   
    
C*********************************************************************  
    
      BLOCK DATA PYDATA 
    
C...Give sensible default values to all status codes and parameters.    
      COMMON/PYSUBS/MSEL,MSUB(200),KFIN(2,-40:40),CKIN(200) 
      SAVE /PYSUBS/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      COMMON/PYINT1/MINT(400),VINT(400) 
      SAVE /PYINT1/ 
      COMMON/PYINT2/ISET(200),KFPR(200,2),COEF(200,20),ICOL(40,4,2) 
      SAVE /PYINT2/ 
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)  
      SAVE /PYINT3/ 
      COMMON/AMPTPYINT4/WIDP(21:40,0:40),WIDE(21:40,0:40),WIDS(21:40,3) 
      SAVE /AMPTPYINT4/ 
      COMMON/PYINT5/NGEN(0:200,3),XSEC(0:200,3) 
      SAVE /PYINT5/ 
      COMMON/PYINT6/PROC(0:200) 
      CHARACTER PROC*28 
      SAVE /PYINT6/ 
    
C...Default values for allowed processes and kinematics constraints.    
      DATA MSEL/1/  
      DATA MSUB/200*0/  
      DATA ((KFIN(I,J),J=-40,40),I=1,2)/40*1,0,80*1,0,40*1/ 
      DATA CKIN/    
     &   2.0, -1.0,  0.0, -1.0,  1.0,  1.0, -10.,  10., -10.,  10., 
     1  -10.,  10., -10.,  10., -10.,  10., -1.0,  1.0, -1.0,  1.0, 
     2   0.0,  1.0,  0.0,  1.0, -1.0,  1.0, -1.0,  1.0,   0.,   0., 
     3   2.0, -1.0,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     4   160*0./    
    
C...Default values for main switches and parameters. Reset information. 
      DATA (MSTP(I),I=1,100)/   
     &     3,    1,    2,    0,    0,    0,    0,    0,    0,    0, 
     1     0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     2     0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     3     1,    2,    0,    0,    0,    2,    0,    0,    0,    0, 
     4     1,    0,    3,    7,    1,    0,    0,    0,    0,    0, 
     5     1,    1,   20,    6,    0,    0,    0,    0,    0,    0, 
     6     1,    2,    2,    2,    1,    0,    0,    0,    0,    0, 
     7     1,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     8     1,    1,  100,    0,    0,    0,    0,    0,    0,    0, 
     9     1,    4,    0,    0,    0,    0,    0,    0,    0,    0/ 
      DATA (MSTP(I),I=101,200)/ 
     &     1,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     1     1,    1,    1,    0,    0,    0,    0,    0,    0,    0, 
     2     0,    1,    2,    1,    1,   20,    0,    0,    0,    0, 
     3     0,    4,    0,    1,    0,    0,    0,    0,    0,    0, 
     4     0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     5     0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     6     0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     7     0,    0,    0,    0,    0,    0,    0,    0,    0,    0, 
     8     5,    3, 1989,   11,   24,    0,    0,    0,    0,    0, 
     9     0,    0,    0,    0,    0,    0,    0,    0,    0,    0/ 
      DATA (PARP(I),I=1,100)/   
     &  0.25,  10.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     1    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     2    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     3   1.5,  2.0, 0.075,  0.,  0.2,   0.,   0.,   0.,   0.,   0., 
     4    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     5   1.0, 2.26, 1.E4, 1.E-4,  0.,   0.,   0.,   0.,   0.,   0., 
     6  0.25,  1.0, 0.25,  1.0,  2.0, 1.E-3, 4.0,   0.,   0.,   0., 
     7   4.0,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     8   1.6, 1.85,  0.5,  0.2, 0.33, 0.66,  0.7,  0.5,   0.,   0., 
     9  0.44, 0.44,  2.0,  1.0,   0.,  3.0,  1.0, 0.75,   0.,   0./ 
      DATA (PARP(I),I=101,200)/ 
     & -0.02,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     1   2.0,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     2   0.4,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     3  0.01,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     4    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     5    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     6    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     7    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     8    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 
     9    0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0./ 
      DATA MSTI/200*0/  
      DATA PARI/200*0./ 
      DATA MINT/400*0/  
      DATA VINT/400*0./ 
    
C...Constants for the generation of the various processes.  
      DATA (ISET(I),I=1,100)/   
     &    1,    1,    1,   -1,    3,   -1,   -1,    3,   -2,   -2,  
     1    2,    2,    2,    2,    2,    2,   -1,    2,    2,    2,  
     2   -1,    2,    2,    2,    2,    2,   -1,    2,    2,    2,  
     3    2,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  
     4   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  
     5   -1,   -1,    2,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  
     6   -1,   -1,   -1,   -1,   -1,   -1,   -1,    2,   -1,   -1,  
     7    4,    4,    4,   -1,   -1,    4,    4,   -1,   -1,   -2,  
     8    2,    2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     9    0,    0,    0,   -1,    0,    5,   -2,   -2,   -2,   -2/  
      DATA (ISET(I),I=101,200)/ 
     &   -1,    1,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     1    2,    2,    2,    2,   -1,   -1,   -1,   -2,   -2,   -2,  
     2   -1,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     3   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     4    1,    1,    1,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     5   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     6    2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     7   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     8   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,  
     9   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2,   -2/  
      DATA ((KFPR(I,J),J=1,2),I=1,50)/  
     &   23,    0,   24,    0,   25,    0,   24,    0,   25,    0,  
     &   24,    0,   23,    0,   25,    0,    0,    0,    0,    0,  
     1    0,    0,    0,    0,   21,   21,   21,   22,   21,   23,  
     1   21,   24,   21,   25,   22,   22,   22,   23,   22,   24,  
     2   22,   25,   23,   23,   23,   24,   23,   25,   24,   24,  
     2   24,   25,   25,   25,    0,   21,    0,   22,    0,   23,  
     3    0,   24,    0,   25,    0,   21,    0,   22,    0,   23,  
     3    0,   24,    0,   25,    0,   21,    0,   22,    0,   23,  
     4    0,   24,    0,   25,    0,   21,    0,   22,    0,   23,  
     4    0,   24,    0,   25,    0,   21,    0,   22,    0,   23/  
      DATA ((KFPR(I,J),J=1,2),I=51,100)/    
     5    0,   24,    0,   25,    0,    0,    0,    0,    0,    0,  
     5    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     6    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     6    0,    0,    0,    0,   21,   21,   24,   24,   22,   24,  
     7   23,   23,   24,   24,   23,   24,   23,   25,   22,   22,  
     7   23,   23,   24,   24,   24,   25,   25,   25,    0,    0,  
     8    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     8    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     9    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     9    0,    0,    0,    0,    0,    0,    0,    0,    0,    0/  
      DATA ((KFPR(I,J),J=1,2),I=101,150)/   
     &   23,    0,   25,    0,    0,    0,    0,    0,    0,    0,  
     &    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     1   21,   25,    0,   25,   21,   25,   22,   22,   22,   23,  
     1   23,   23,   24,   24,    0,    0,    0,    0,    0,    0,  
     2    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     2    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     3    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     3    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     4   32,    0,   37,    0,   40,    0,    0,    0,    0,    0,  
     4    0,    0,    0,    0,    0,    0,    0,    0,    0,    0/  
      DATA ((KFPR(I,J),J=1,2),I=151,200)/   
     5    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     5    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     6    0,   37,    0,    0,    0,    0,    0,    0,    0,    0,  
     6    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     7    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     7    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     8    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     8    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     9    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  
     9    0,    0,    0,    0,    0,    0,    0,    0,    0,    0/  
      DATA COEF/4000*0./    
      DATA (((ICOL(I,J,K),K=1,2),J=1,4),I=1,40)/    
     1 4,0,3,0,2,0,1,0,3,0,4,0,1,0,2,0,2,0,0,1,4,0,0,3,3,0,0,4,1,0,0,2, 
     2 3,0,0,4,1,4,3,2,4,0,0,3,4,2,1,3,2,0,4,1,4,0,2,3,4,0,3,4,2,0,1,2, 
     3 3,2,1,0,1,4,3,0,4,3,3,0,2,1,1,0,3,2,1,4,1,0,0,2,2,4,3,1,2,0,0,1, 
     4 3,2,1,4,1,4,3,2,4,2,1,3,4,2,1,3,3,4,4,3,1,2,2,1,2,0,3,1,2,0,0,0, 
     5 4,2,1,0,0,0,1,0,3,0,0,3,1,2,0,0,4,0,0,4,0,0,1,2,2,0,0,1,4,4,3,3, 
     6 2,2,1,1,4,4,3,3,3,3,4,4,1,1,2,2,3,2,1,3,1,2,0,0,4,2,1,4,0,0,1,2, 
     7 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
     8 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
     9 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
     & 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0/ 
    
C...Character constants: name of processes. 
      DATA PROC(0)/                    'All included subprocesses   '/  
      DATA (PROC(I),I=1,20)/    
     1'f + fb -> gamma*/Z0         ',  'f + fb'' -> W+/-             ', 
     2'f + fb -> H0                ',  'gamma + W+/- -> W+/-        ',  
     3'Z0 + Z0 -> H0               ',  'Z0 + W+/- -> W+/-           ',  
     4'                            ',  'W+ + W- -> H0               ',  
     5'                            ',  '                            ',  
     6'f + f'' -> f + f''            ','f + fb -> f'' + fb''          ',    
     7'f + fb -> g + g             ',  'f + fb -> g + gamma         ',  
     8'f + fb -> g + Z0            ',  'f + fb'' -> g + W+/-         ', 
     9'f + fb -> g + H0            ',  'f + fb -> gamma + gamma     ',  
     &'f + fb -> gamma + Z0        ',  'f + fb'' -> gamma + W+/-     '/ 
      DATA (PROC(I),I=21,40)/   
     1'f + fb -> gamma + H0        ',  'f + fb -> Z0 + Z0           ',  
     2'f + fb'' -> Z0 + W+/-        ', 'f + fb -> Z0 + H0           ',  
     3'f + fb -> W+ + W-           ',  'f + fb'' -> W+/- + H0        ', 
     4'f + fb -> H0 + H0           ',  'f + g -> f + g              ',  
     5'f + g -> f + gamma          ',  'f + g -> f + Z0             ',  
     6'f + g -> f'' + W+/-          ', 'f + g -> f + H0             ',  
     7'f + gamma -> f + g          ',  'f + gamma -> f + gamma      ',  
     8'f + gamma -> f + Z0         ',  'f + gamma -> f'' + W+/-      ', 
     9'f + gamma -> f + H0         ',  'f + Z0 -> f + g             ',  
     &'f + Z0 -> f + gamma         ',  'f + Z0 -> f + Z0            '/  
      DATA (PROC(I),I=41,60)/   
     1'f + Z0 -> f'' + W+/-         ', 'f + Z0 -> f + H0            ',  
     2'f + W+/- -> f'' + g          ', 'f + W+/- -> f'' + gamma      ', 
     3'f + W+/- -> f'' + Z0         ', 'f + W+/- -> f'' + W+/-       ', 
     4'f + W+/- -> f'' + H0         ', 'f + H0 -> f + g             ',  
     5'f + H0 -> f + gamma         ',  'f + H0 -> f + Z0            ',  
     6'f + H0 -> f'' + W+/-         ', 'f + H0 -> f + H0            ',  
     7'g + g -> f + fb             ',  'g + gamma -> f + fb         ',  
     8'g + Z0 -> f + fb            ',  'g + W+/- -> f + fb''         ', 
     9'g + H0 -> f + fb            ',  'gamma + gamma -> f + fb     ',  
     &'gamma + Z0 -> f + fb        ',  'gamma + W+/- -> f + fb''     '/ 
      DATA (PROC(I),I=61,80)/   
     1'gamma + H0 -> f + fb        ',  'Z0 + Z0 -> f + fb           ',  
     2'Z0 + W+/- -> f + fb''        ', 'Z0 + H0 -> f + fb           ',  
     3'W+ + W- -> f + fb           ',  'W+/- + H0 -> f + fb''        ', 
     4'H0 + H0 -> f + fb           ',  'g + g -> g + g              ',  
     5'gamma + gamma -> W+ + W-    ',  'gamma + W+/- -> gamma + W+/-',  
     6'Z0 + Z0 -> Z0 + Z0          ',  'Z0 + Z0 -> W+ + W-          ',  
     7'Z0 + W+/- -> Z0 + W+/-      ',  'Z0 + Z0 -> Z0 + H0          ',  
     8'W+ + W- -> gamma + gamma    ',  'W+ + W- -> Z0 + Z0          ',  
     9'W+/- + W+/- -> W+/- + W+/-  ',  'W+/- + H0 -> W+/- + H0      ',  
     &'H0 + H0 -> H0 + H0          ',  '                            '/  
      DATA (PROC(I),I=81,100)/  
     1'q + qb -> Q + QB, massive   ',  'g + g -> Q + QB, massive    ',  
     2'                            ',  '                            ',  
     3'                            ',  '                            ',  
     4'                            ',  '                            ',  
     5'                            ',  '                            ',  
     6'Elastic scattering          ',  'Single diffractive          ',  
     7'Double diffractive          ',  'Central diffractive         ',  
     8'Low-pT scattering           ',  'Semihard QCD 2 -> 2         ',  
     9'                            ',  '                            ',  
     &'                            ',  '                            '/  
      DATA (PROC(I),I=101,120)/ 
     1'g + g -> gamma*/Z0          ',  'g + g -> H0                 ',  
     2'                            ',  '                            ',  
     3'                            ',  '                            ',  
     4'                            ',  '                            ',  
     5'                            ',  '                            ',  
     6'f + fb -> g + H0            ',  'q + g -> q + H0             ',  
     7'g + g -> g + H0             ',  'g + g -> gamma + gamma      ',  
     8'g + g -> gamma + Z0         ',  'g + g -> Z0 + Z0            ',  
     9'g + g -> W+ + W-            ',  '                            ',  
     &'                            ',  '                            '/  
      DATA (PROC(I),I=121,140)/ 
     1'g + g -> f + fb + H0        ',  '                            ',  
     2'                            ',  '                            ',  
     3'                            ',  '                            ',  
     4'                            ',  '                            ',  
     5'                            ',  '                            ',  
     6'                            ',  '                            ',  
     7'                            ',  '                            ',  
     8'                            ',  '                            ',  
     9'                            ',  '                            ',  
     &'                            ',  '                            '/  
      DATA (PROC(I),I=141,160)/ 
     1'f + fb -> gamma*/Z0/Z''0     ', 'f + fb'' -> H+/-             ', 
     2'f + fb -> R                 ',  '                            ',  
     3'                            ',  '                            ',  
     4'                            ',  '                            ',  
     5'                            ',  '                            ',  
     6'                            ',  '                            ',  
     7'                            ',  '                            ',  
     8'                            ',  '                            ',  
     9'                            ',  '                            ',  
     &'                            ',  '                            '/  
      DATA (PROC(I),I=161,180)/ 
     1'f + g -> f'' + H+/-          ', '                            ',  
     2'                            ',  '                            ',  
     3'                            ',  '                            ',  
     4'                            ',  '                            ',  
     5'                            ',  '                            ',  
     6'                            ',  '                            ',  
     7'                            ',  '                            ',  
     8'                            ',  '                            ',  
     9'                            ',  '                            ',  
     &'                            ',  '                            '/  
      DATA (PROC(I),I=181,200)/     20*'                            '/  
    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYKCUT(MCUT)   
    
C...Dummy routine, which the user can replace in order to make cuts on  
C...the kinematics on the parton level before the matrix elements are   
C...evaluated and the event is generated. The cross-section estimates   
C...will automatically take these cuts into account, so the given   
C...values are for the allowed phase space region only. MCUT=0 means    
C...that the event has passed the cuts, MCUT=1 that it has failed.  
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
    
      MCUT=0    
    
      RETURN    
      END   
    
C*********************************************************************  
    
      SUBROUTINE PYSTFE(KF,X,Q2,XPQ)    
    
C...This is a dummy routine, where the user can introduce an interface  
C...to his own external structure function parametrization. 
C...Arguments in:   
C...KF : 2212 for p, 211 for pi+; isospin conjugation for n and charge  
C...    conjugation for pbar, nbar or pi- is performed by PYSTFU.   
C...X : x value.    
C...Q2 : Q^2 value. 
C...Arguments out:  
C...XPQ(-6:6) : x * f(x,Q2), with index according to KF code,   
C...    except that gluon is placed in 0. Thus XPQ(0) = xg, 
C...    XPQ(1) = xd, XPQ(-1) = xdbar, XPQ(2) = xu, XPQ(-2) = xubar, 
C...    XPQ(3) = xs, XPQ(-3) = xsbar, XPQ(4) = xc, XPQ(-4) = xcbar, 
C...    XPQ(5) = xb, XPQ(-5) = xbbar, XPQ(6) = xt, XPQ(-6) = xtbar. 
C...    
C...One such interface, to the Diemos, Ferroni, Longo, Martinelli   
C...proton structure functions, already comes with the package. What    
C...the user needs here is external files with the three routines   
C...FXG160, FXG260 and FXG360 of the authors above, plus the    
C...interpolation routine FINT, which is part of the CERN library   
C...KERNLIB package. To avoid problems with unresolved external 
C...references, the external calls are commented in the current 
C...version. To enable this option, remove the C* at the beginning  
C...of the relevant lines.  
C...    
C...Alternatively, the routine can be used as an interface to the   
C...structure function evolution program of Tung. This can be achieved  
C...by removing C* at the beginning of some of the lines below. 
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200) 
      SAVE /LUDAT1/ 
      COMMON/LUDAT2/KCHG(500,3),PMAS(500,4),PARF(2000),VCKM(4,4)    
      SAVE /LUDAT2/ 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200) 
      SAVE /PYPARS/ 
      DIMENSION XPQ(-6:6),XFDFLM(9) 
      CHARACTER CHDFLM(9)*5,HEADER*40   
      DATA CHDFLM/'UPVAL','DOVAL','GLUON','QBAR ','UBAR ','SBAR ',  
     &'CBAR ','BBAR ','TBAR '/  
      DATA HEADER/'Tung evolution package has been invoked'/    
      DATA INIT/0/  
    
C...Proton structure functions from Diemoz, Ferroni, Longo, Martinelli. 
C...Allowed variable range 10 GeV2 < Q2 < 1E8 GeV2, 5E-5 < x < .95. 
      IF(MSTP(51).GE.11.AND.MSTP(51).LE.13.AND.MSTP(52).LE.1) THEN  
        XDFLM=MAX(0.51E-4,X)    
        Q2DFLM=MAX(10.,MIN(1E8,Q2)) 
        IF(MSTP(52).EQ.0) Q2DFLM=10.    
        DO 100 J=1,9    
        IF(MSTP(52).EQ.1.AND.J.EQ.9) THEN   
          Q2DFLM=Q2DFLM*(40./PMAS(6,1))**2  
          Q2DFLM=MAX(10.,MIN(1E8,Q2))   
        ENDIF   
        XFDFLM(J)=0.    
C...Remove C* on following three lines to enable the DFLM options.  
C*      IF(MSTP(51).EQ.11) CALL FXG160(XDFLM,Q2DFLM,CHDFLM(J),XFDFLM(J))    
C*      IF(MSTP(51).EQ.12) CALL FXG260(XDFLM,Q2DFLM,CHDFLM(J),XFDFLM(J))    
C*      IF(MSTP(51).EQ.13) CALL FXG360(XDFLM,Q2DFLM,CHDFLM(J),XFDFLM(J))    
  100   CONTINUE    
        IF(X.LT.0.51E-4.AND.ABS(PARP(51)-1.).GT.0.01) THEN  
          CXS=(0.51E-4/X)**(PARP(51)-1.)    
          DO 110 J=1,7  
  110     XFDFLM(J)=XFDFLM(J)*CXS   
        ENDIF   
        XPQ(0)=XFDFLM(3)    
        XPQ(1)=XFDFLM(2)+XFDFLM(5)  
        XPQ(2)=XFDFLM(1)+XFDFLM(5)  
        XPQ(3)=XFDFLM(6)    
        XPQ(4)=XFDFLM(7)    
        XPQ(5)=XFDFLM(8)    
        XPQ(6)=XFDFLM(9)    
        XPQ(-1)=XFDFLM(5)   
        XPQ(-2)=XFDFLM(5)   
        XPQ(-3)=XFDFLM(6)   
        XPQ(-4)=XFDFLM(7)   
        XPQ(-5)=XFDFLM(8)   
        XPQ(-6)=XFDFLM(9)   
    
C...Proton structure function evolution from Wu-Ki Tung: parton 
C...distribution functions incorporating heavy quark mass effects.  
C...Allowed variable range: PARP(52) < Q < PARP(53); PARP(54) < x < 1.  
      ELSE  
        IF(INIT.EQ.0) THEN  
          I1=0  
          IF(MSTP(52).EQ.4) I1=1    
          IHDRN=1   
          NU=MSTP(53)   
          I2=MSTP(51)   
          IF(MSTP(51).GE.11) I2=MSTP(51)-3  
          I3=0  
          IF(MSTP(52).EQ.3) I3=1    
    
C...Convert to Lambda in CWZ scheme (approximately linear relation).    
          ALAM=0.75*PARP(1) 
          TPMS=PMAS(6,1)    
          QINI=PARP(52) 
          QMAX=PARP(53) 
          XMIN=PARP(54) 
    
C...Initialize evolution (perform calculation or read results from  
C...file).  
C...Remove C* on following two lines to enable Tung initialization. 
C*        CALL PDFSET(I1,IHDRN,ALAM,TPMS,QINI,QMAX,XMIN,NU,HEADER,  
C*   &    I2,I3,IRET,IRR)   
          INIT=1    
        ENDIF   
    
C...Put into output array.  
        Q=SQRT(Q2)  
        DO 200 I=-6,6   
        FIXQ=0. 
C...Remove C* on following line to enable structure function call.  
C*      FIXQ=MAX(0.,PDF(10,1,I,X,Q,IR)) 
  200   XPQ(I)=X*FIXQ   
    
C...Change order of u and d quarks from Tung to PYTHIA convention.  
        XPS=XPQ(1)  
        XPQ(1)=XPQ(2)   
        XPQ(2)=XPS  
        XPS=XPQ(-1) 
        XPQ(-1)=XPQ(-2) 
        XPQ(-2)=XPS 
      ENDIF 
    
      RETURN    
      END   

