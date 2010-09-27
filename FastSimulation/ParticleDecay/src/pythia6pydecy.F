c
c --- This is just a copy of pydecy from Pythia, modified to use
c     a different common block (so as to not interfere with the
c     users event) and ignoring any decays turned off by the
c     user. Furthemore, the decay vertices are determined within 
c     a uniform axial (along z) magnetic field, the value of which 
c     has to be passed in argument (in Tesla).
c     Note that if PYTHIA is used to generate the event and MSTJ(22)
c     was set to a nonzero value, this flag has no effect here.
c

C*********************************************************************
 
C...PYDECY
C...Handles the decay of unstable particles.
 
      SUBROUTINE PYTHIA6PYDECY(IP)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYCOMP
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      SAVE /PYJETS/,/PYDAT1/,/PYDAT2/,/PYDAT3/
c
c --- Hack
c
      EXTERNAL PYDATA
c
c --- End Hack
c

C...Local arrays.
      DIMENSION VDCY(4),KFLO(4),KFL1(4),PV(10,5),RORD(10),UE(3),BE(3),
     &WTCOR(10),PTAU(4),PCMTAU(4),DBETAU(3)
      CHARACTER CIDC*4
      DATA WTCOR/2D0,5D0,15D0,60D0,250D0,1500D0,1.2D4,1.2D5,150D0,16D0/

C...Functions: momentum in two-particle decays and four-product.
      PAWT(A,B,C)=SQRT((A**2-(B+C)**2)*(A**2-(B-C)**2))/(2D0*A)
      FOUR(I,J)=P(I,4)*P(J,4)-P(I,1)*P(J,1)-P(I,2)*P(J,2)-P(I,3)*P(J,3)
 
C...Initial values.
      NTRY=0
      NSAV=N
      KFA=IABS(K(IP,2))
      KFS=ISIGN(1,K(IP,2))
      KC=PYCOMP(KFA)
      MSTJ(92)=0

C-ap  to silence a couple of compiler warnings.
      wtmax = 0 
      pqt   = 0
      pmst  = 0
      pmr   = 0
      pmes  = 0
      isis  = 0
      im    = 0
      mmix  = 0
 
C...Choose lifetime and determine decay vertex.
C      IF(K(IP,1).EQ.5) THEN
C        V(IP,5)=0D0
C      ELSEIF(K(IP,1).NE.4) THEN
C        V(IP,5)=-PMAS(KC,4)*LOG(PYR(0))
C      ENDIF
C PJ : Decay the particle immediately
      V(IP,5) = 0D0
      DO 100 J=1,4
        VDCY(J)=V(IP,J)+V(IP,5)*P(IP,J)/P(IP,5)
 100  CONTINUE
 
C...Determine whether decay allowed or not.
      MOUT=0

C PJ Well the decay is always allowed in this case
C      IF(MSTJ(22).EQ.2) THEN
C        IF(PMAS(KC,4).GT.PARJ(71)) MOUT=1
C      ELSEIF(MSTJ(22).EQ.3) THEN
C        IF(VDCY(1)**2+VDCY(2)**2+VDCY(3)**2.GT.PARJ(72)**2) MOUT=1
C      ELSEIF(MSTJ(22).EQ.4) THEN
C        IF(VDCY(1)**2+VDCY(2)**2.GT.PARJ(73)**2) MOUT=1
C        IF(ABS(VDCY(3)).GT.PARJ(74)) MOUT=1
C      ENDIF

      IF(MOUT.EQ.1.AND.K(IP,1).NE.5) THEN
        K(IP,1)=4
        RETURN
      ENDIF
 
C...Interface to external tau decay library (for tau polarization).
      IF(KFA.EQ.15.AND.MSTJ(28).GE.1) THEN
 
C...Starting values for pointers and momenta.
        ITAU=IP
        DO 110 J=1,4
          PTAU(J)=P(ITAU,J)
          PCMTAU(J)=P(ITAU,J)
  110   CONTINUE
 
C...Iterate to find position and code of mother of tau.
        IMTAU=ITAU
  120   IMTAU=K(IMTAU,3)
 
        IF(IMTAU.EQ.0) THEN
C...If no known origin then impossible to do anything further.
          KFORIG=0
          IORIG=0
 
        ELSEIF(K(IMTAU,2).EQ.K(ITAU,2)) THEN
C...If tau -> tau + gamma then add gamma energy and loop.
          IF(K(K(IMTAU,4),2).EQ.22) THEN
            DO 130 J=1,4
              PCMTAU(J)=PCMTAU(J)+P(K(IMTAU,4),J)
  130       CONTINUE
          ELSEIF(K(K(IMTAU,5),2).EQ.22) THEN
            DO 140 J=1,4
              PCMTAU(J)=PCMTAU(J)+P(K(IMTAU,5),J)
  140       CONTINUE
          ENDIF
          GOTO 120
 
        ELSEIF(IABS(K(IMTAU,2)).GT.100) THEN
C...If coming from weak decay of hadron then W is not stored in record,
C...but can be reconstructed by adding neutrino momentum.
          KFORIG=-ISIGN(24,K(ITAU,2))
          IORIG=0
          DO 160 II=K(IMTAU,4),K(IMTAU,5)
            IF(K(II,2)*ISIGN(1,K(ITAU,2)).EQ.-16) THEN
              DO 150 J=1,4
                PCMTAU(J)=PCMTAU(J)+P(II,J)
  150         CONTINUE
            ENDIF
  160     CONTINUE
 
        ELSE
C...If coming from resonance decay then find latest copy of this
C...resonance (may not completely agree).
          KFORIG=K(IMTAU,2)
          IORIG=IMTAU
          DO 170 II=IMTAU+1,IP-1
            IF(K(II,2).EQ.KFORIG.AND.K(II,3).EQ.IORIG.AND.
     &      ABS(P(II,5)-P(IORIG,5)).LT.1D-5*P(IORIG,5)) IORIG=II
  170     CONTINUE
          DO 180 J=1,4
            PCMTAU(J)=P(IORIG,J)
  180     CONTINUE
        ENDIF
 
C...Boost tau to rest frame of production process (where known)
C...and rotate it to sit along +z axis.
        DO 190 J=1,3
          DBETAU(J)=PCMTAU(J)/PCMTAU(4)
  190   CONTINUE
        IF(KFORIG.NE.0) CALL PYROBO(ITAU,ITAU,0D0,0D0,-DBETAU(1),
     &  -DBETAU(2),-DBETAU(3))
        PHITAU=PYANGL(P(ITAU,1),P(ITAU,2))
        CALL PYROBO(ITAU,ITAU,0D0,-PHITAU,0D0,0D0,0D0)
        THETAU=PYANGL(P(ITAU,3),P(ITAU,1))
        CALL PYROBO(ITAU,ITAU,-THETAU,0D0,0D0,0D0,0D0)
 
C...Call tau decay routine (if meaningful) and fill extra info.
        IF(KFORIG.NE.0.OR.MSTJ(28).EQ.2) THEN
          CALL PYTAUD(ITAU,IORIG,KFORIG,NDECAY)
          DO 200 II=NSAV+1,NSAV+NDECAY
            K(II,1)=1
            K(II,3)=IP
            K(II,4)=0
            K(II,5)=0
  200     CONTINUE
          N=NSAV+NDECAY
        ENDIF
 
C...Boost back decay tau and decay products.
        DO 210 J=1,4
          P(ITAU,J)=PTAU(J)
  210   CONTINUE
        IF(KFORIG.NE.0.OR.MSTJ(28).EQ.2) THEN
          CALL PYROBO(NSAV+1,N,THETAU,PHITAU,0D0,0D0,0D0)
          IF(KFORIG.NE.0) CALL PYROBO(NSAV+1,N,0D0,0D0,DBETAU(1),
     &    DBETAU(2),DBETAU(3))
 
C...Skip past ordinary tau decay treatment.
          MMAT=0
          MBST=0
          ND=0
          GOTO 630
        ENDIF
      ENDIF
 
C...B-Bbar mixing: flip sign of meson appropriately.
      MMIX=0
      IF((KFA.EQ.511.OR.KFA.EQ.531).AND.MSTJ(26).GE.1) THEN
        XBBMIX=PARJ(76)
        IF(KFA.EQ.531) XBBMIX=PARJ(77)
        IF(SIN(0.5D0*XBBMIX*V(IP,5)/PMAS(KC,4))**2.GT.PYR(0)) MMIX=1
        IF(MMIX.EQ.1) KFS=-KFS
      ENDIF
 
C...Check existence of decay channels. Particle/antiparticle rules.
      KCA=KC
      IF(MDCY(KC,2).GT.0) THEN
        MDMDCY=MDME(MDCY(KC,2),2)
        IF(MDMDCY.GT.80.AND.MDMDCY.LE.90) KCA=MDMDCY
      ENDIF
      IF(MDCY(KCA,2).LE.0.OR.MDCY(KCA,3).LE.0) THEN
        CALL PYERRM(9,'(PYDECY:) no decay channel defined')
        RETURN
      ENDIF
      IF(MOD(KFA/1000,10).EQ.0.AND.KCA.EQ.85) KFS=-KFS
      IF(KCHG(KC,3).EQ.0) THEN
        KFSP=1
        KFSN=0
        IF(PYR(0).GT.0.5D0) KFS=-KFS
      ELSEIF(KFS.GT.0) THEN
        KFSP=1
        KFSN=0
      ELSE
        KFSP=0
        KFSN=1
      ENDIF
 
C...Sum branching ratios of allowed decay channels.
      NOPE=0
      BRSU=0D0
      DO 230 IDL=MDCY(KCA,2),MDCY(KCA,2)+MDCY(KCA,3)-1
c
c --- Hack
c
c         IF(MDME(IDL,1).NE.1.AND.KFSP*MDME(IDL,1).NE.2.AND.
c     1        KFSN*MDME(IDL,1).NE.3) GOTO 230
c
c --- End Hack
c
        IF(MDME(IDL,2).GT.100) GOTO 230
        NOPE=NOPE+1
        BRSU=BRSU+BRAT(IDL)
  230 CONTINUE
      IF(NOPE.EQ.0) THEN
        CALL PYERRM(2,'(PYDECY:) all decay channels closed by user')
        RETURN
      ENDIF
 
C...Select decay channel among allowed ones.
  240 RBR=BRSU*PYR(0)
      IDL=MDCY(KCA,2)-1
  250 IDL=IDL+1
c
c --- Hack
c
c      IF(MDME(IDL,1).NE.1.AND.KFSP*MDME(IDL,1).NE.2.AND.
c     1     KFSN*MDME(IDL,1).NE.3) THEN
c        IF(IDL.LT.MDCY(KCA,2)+MDCY(KCA,3)-1) GOTO 250
c      ELSEIF(MDME(IDL,2).GT.100) THEN
c        IF(IDL.LT.MDCY(KCA,2)+MDCY(KCA,3)-1) GOTO 250
c      ELSE
c
c --- End Hack
c
        IDC=IDL
        RBR=RBR-BRAT(IDL)
        IF(IDL.LT.MDCY(KCA,2)+MDCY(KCA,3)-1.AND.RBR.GT.0D0) GOTO 250
c
c --- Hack
c
c      ENDIF
c
c --- End Hack
c
 
C...Start readout of decay channel: matrix element, reset counters.
      MMAT=MDME(IDC,2)
  260 NTRY=NTRY+1
      IF(MOD(NTRY,200).EQ.0) THEN
        WRITE(CIDC,'(I4)') IDC
C...Do not print warning for some well-known special cases.
        IF(KFA.NE.113.AND.KFA.NE.115.AND.KFA.NE.215)
     &  CALL PYERRM(4,'(PYDECY:) caught in loop for decay channel'//
     &  CIDC)
        GOTO 240
      ENDIF
      IF(NTRY.GT.1000) THEN
        CALL PYERRM(14,'(PYDECY:) caught in infinite loop')
        IF(MSTU(21).GE.1) RETURN
      ENDIF
      I=N
      NP=0
      NQ=0
      MBST=0
      IF(MMAT.GE.11.AND.P(IP,4).GT.20D0*P(IP,5)) MBST=1
      DO 270 J=1,4
        PV(1,J)=0D0
        IF(MBST.EQ.0) PV(1,J)=P(IP,J)
  270 CONTINUE
      IF(MBST.EQ.1) PV(1,4)=P(IP,5)
      PV(1,5)=P(IP,5)
      PS=0D0
      PSQ=0D0
      MREM=0
      MHADDY=0
      IF(KFA.GT.80) MHADDY=1
C.. Random flavour and popcorn system memory.
      IRNDMO=0
      JTMO=0
      MSTU(121)=0
      MSTU(125)=10
 
C...Read out decay products. Convert to standard flavour code.
      JTMAX=5
      IF(MDME(IDC+1,2).EQ.101) JTMAX=10
      DO 280 JT=1,JTMAX
        IF(JT.LE.5) KP=KFDP(IDC,JT)
        IF(JT.GE.6) KP=KFDP(IDC+1,JT-5)
        IF(KP.EQ.0) GOTO 280
        KPA=IABS(KP)
        KCP=PYCOMP(KPA)
        IF(KPA.GT.80) MHADDY=1
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
          CALL PYDCYK(-KFS*INT(1D0+(2D0+PARJ(2))*PYR(0)),0,KFP,KDUMP)
          IF(KFP.EQ.0) GOTO 260
          KFP=-KFP
          IRNDMO=1
          MSTJ(93)=1
          IF(PV(1,5).LT.PARJ(32)+2D0*PYMASS(KFP)) GOTO 260
        ELSEIF(KP.EQ.-82) THEN
          KFP=MSTU(124)
        ENDIF
        IF(KPA.EQ.81.OR.KPA.EQ.82) KCP=PYCOMP(KFP)
 
C...Add decay product to event record or to quark flavour list.
        KFPA=IABS(KFP)
        KQP=KCHG(KCP,2)
        IF(MMAT.GE.11.AND.MMAT.LE.30.AND.KQP.NE.0) THEN
          NQ=NQ+1
          KFLO(NQ)=KFP
C...set rndmflav popcorn system pointer
          IF(KP.EQ.82.AND.MSTU(121).GT.0) JTMO=NQ
          MSTJ(93)=2
          PSQ=PSQ+PYMASS(KFLO(NQ))
        ELSEIF((MMAT.EQ.42.OR.MMAT.EQ.43.OR.MMAT.EQ.48).AND.NP.EQ.3.AND.
     &    MOD(NQ,2).EQ.1) THEN
          NQ=NQ-1
          PS=PS-P(I,5)
          K(I,1)=1
          KFI=K(I,2)
          CALL PYKFDI(KFP,KFI,KFLDMP,K(I,2))
          IF(K(I,2).EQ.0) GOTO 260
          MSTJ(93)=1
          P(I,5)=PYMASS(K(I,2))
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
          P(I,5)=PYMASS(KFP)
          PS=PS+P(I,5)
        ENDIF
  280 CONTINUE
 
C...Check masses for resonance decays.
      IF(MHADDY.EQ.0) THEN
        IF(PS+PARJ(64).GT.PV(1,5)) GOTO 240
      ENDIF
 
C...Choose decay multiplicity in phase space model.
  290 IF(MMAT.GE.11.AND.MMAT.LE.30) THEN
        PSP=PS
        CNDE=PARJ(61)*LOG(MAX((PV(1,5)-PS-PSQ)/PARJ(62),1.1D0))
        IF(MMAT.EQ.12) CNDE=CNDE+PARJ(63)
  300   NTRY=NTRY+1
C...Reset popcorn flags if new attempt. Re-select rndmflav if failed.
        IF(IRNDMO.EQ.0) THEN
           MSTU(121)=0
           JTMO=0
        ELSEIF(IRNDMO.EQ.1) THEN
           IRNDMO=2
        ELSE
           GOTO 260
        ENDIF
        IF(NTRY.GT.1000) THEN
          CALL PYERRM(14,'(PYDECY:) caught in infinite loop')
          IF(MSTU(21).GE.1) RETURN
        ENDIF
        IF(MMAT.LE.20) THEN
          GAUSS=SQRT(-2D0*CNDE*LOG(MAX(1D-10,PYR(0))))*
     &    SIN(PARU(2)*PYR(0))
          ND=0.5D0+0.5D0*NP+0.25D0*NQ+CNDE+GAUSS
          IF(ND.LT.NP+NQ/2.OR.ND.LT.2.OR.ND.GT.10) GOTO 300
          IF(MMAT.EQ.13.AND.ND.EQ.2) GOTO 300
          IF(MMAT.EQ.14.AND.ND.LE.3) GOTO 300
          IF(MMAT.EQ.15.AND.ND.LE.4) GOTO 300
        ELSE
          ND=MMAT-20
        ENDIF
C.. Set maximum popcorn meson number. Test rndmflav popcorn size.
        MSTU(125)=ND-NQ/2
        IF(MSTU(121).GT.MSTU(125)) GOTO 300
 
C...Form hadrons from flavour content.
        DO 310 JT=1,NQ
          KFL1(JT)=KFLO(JT)
  310   CONTINUE
        IF(ND.EQ.NP+NQ/2) GOTO 330
        DO 320 I=N+NP+1,N+ND-NQ/2
C.. Stick to started popcorn system, else pick side at random
          JT=JTMO
          IF(JT.EQ.0) JT=1+INT((NQ-1)*PYR(0))
          CALL PYDCYK(KFL1(JT),0,KFL2,K(I,2))
          IF(K(I,2).EQ.0) GOTO 300
          MSTU(125)=MSTU(125)-1
          JTMO=0
          IF(MSTU(121).GT.0) JTMO=JT
          KFL1(JT)=-KFL2
  320   CONTINUE
  330   JT=2
        JT2=3
        JT3=4
        IF(NQ.EQ.4.AND.PYR(0).LT.PARJ(66)) JT=4
        IF(JT.EQ.4.AND.ISIGN(1,KFL1(1)*(10-IABS(KFL1(1))))*
     &  ISIGN(1,KFL1(JT)*(10-IABS(KFL1(JT)))).GT.0) JT=3
        IF(JT.EQ.3) JT2=2
        IF(JT.EQ.4) JT3=2
        CALL PYDCYK(KFL1(1),KFL1(JT),KFLDMP,K(N+ND-NQ/2+1,2))
        IF(K(N+ND-NQ/2+1,2).EQ.0) GOTO 300
        IF(NQ.EQ.4) CALL PYDCYK(KFL1(JT2),KFL1(JT3),KFLDMP,K(N+ND,2))
        IF(NQ.EQ.4.AND.K(N+ND,2).EQ.0) GOTO 300
 
C...Check that sum of decay product masses not too large.
        PS=PSP
        DO 340 I=N+NP+1,N+ND
          K(I,1)=1
          K(I,3)=IP
          K(I,4)=0
          K(I,5)=0
          P(I,5)=PYMASS(K(I,2))
          PS=PS+P(I,5)
  340   CONTINUE
        IF(PS+PARJ(64).GT.PV(1,5)) GOTO 300
 
C...Rescale energy to subtract off spectator quark mass.
      ELSEIF((MMAT.EQ.31.OR.MMAT.EQ.33.OR.MMAT.EQ.44)
     &  .AND.NP.GE.3) THEN
        PS=PS-P(N+NP,5)
        PQT=(P(N+NP,5)+PARJ(65))/PV(1,5)
        DO 350 J=1,5
          P(N+NP,J)=PQT*PV(1,J)
          PV(1,J)=(1D0-PQT)*PV(1,J)
  350   CONTINUE
        IF(PS+PARJ(64).GT.PV(1,5)) GOTO 260
        ND=NP-1
        MREM=1
 
C...Fully specified final state: check mass broadening effects.
      ELSE
        IF(NP.GE.2.AND.PS+PARJ(64).GT.PV(1,5)) GOTO 260
        ND=NP
      ENDIF
 
C...Determine position of grandmother, number of sisters.
      NM=0
      KFAS=0
      MSGN=0
      IF(MMAT.EQ.3) THEN
        IM=K(IP,3)
        IF(IM.LT.0.OR.IM.GE.IP) IM=0
        IF(IM.NE.0) THEN
          KFAM=IABS(K(IM,2))
          DO 360 IL=MAX(IP-2,IM+1),MIN(IP+2,N)
            IF(K(IL,3).EQ.IM) THEN
               NM=NM+1
               IF(IL.NE.IP) ISIS=IL
            END IF
  360     CONTINUE
          IF(NM.NE.2.OR.KFAM.LE.100.OR.MOD(KFAM,10).NE.1.OR.
     &    MOD(KFAM/1000,10).NE.0) NM=0
          IF(NM.EQ.2) THEN
            KFAS=IABS(K(ISIS,2))
            IF((KFAS.LE.100.OR.MOD(KFAS,10).NE.1.OR.
     &      MOD(KFAS/1000,10).NE.0).AND.KFAS.NE.22) NM=0
          ENDIF
        ENDIF
      ENDIF
 
C...Kinematics of one-particle decays.
      IF(ND.EQ.1) THEN
        DO 370 J=1,4
          P(N+1,J)=P(IP,J)
  370   CONTINUE
        GOTO 630
      ENDIF
 
C...Calculate maximum weight ND-particle decay.
      PV(ND,5)=P(N+ND,5)
      IF(ND.GE.3) THEN
        WTMAX=1D0/WTCOR(ND-2)
        PMAX=PV(1,5)-PS+P(N+ND,5)
        PMIN=0D0
        DO 380 IL=ND-1,1,-1
          PMAX=PMAX+P(N+IL,5)
          PMIN=PMIN+P(N+IL+1,5)
          WTMAX=WTMAX*PAWT(PMAX,PMIN,P(N+IL,5))
  380   CONTINUE
      ENDIF
 
C...Find virtual gamma mass in Dalitz decay.
  390 IF(ND.EQ.2) THEN
      ELSEIF(MMAT.EQ.2) THEN
        PMES=4D0*PMAS(11,1)**2
        PMRHO2=PMAS(131,1)**2
        PGRHO2=PMAS(131,2)**2
  400   PMST=PMES*(P(IP,5)**2/PMES)**PYR(0)
        WT=(1+0.5D0*PMES/PMST)*SQRT(MAX(0D0,1D0-PMES/PMST))*
     &  (1D0-PMST/P(IP,5)**2)**3*(1D0+PGRHO2/PMRHO2)/
     &  ((1D0-PMST/PMRHO2)**2+PGRHO2/PMRHO2)
        IF(WT.LT.PYR(0)) GOTO 400
        PV(2,5)=MAX(2.00001D0*PMAS(11,1),SQRT(PMST))
 
C...M-generator gives weight. If rejected, try again.
      ELSE
  410   RORD(1)=1D0
        DO 440 IL1=2,ND-1
          RSAV=PYR(0)
          DO 420 IL2=IL1-1,1,-1
            IF(RSAV.LE.RORD(IL2)) GOTO 430
            RORD(IL2+1)=RORD(IL2)
  420     CONTINUE
  430     RORD(IL2+1)=RSAV
  440   CONTINUE
        RORD(ND)=0D0
        WT=1D0
        DO 450 IL=ND-1,1,-1
          PV(IL,5)=PV(IL+1,5)+P(N+IL,5)+(RORD(IL)-RORD(IL+1))*
     &    (PV(1,5)-PS)
          WT=WT*PAWT(PV(IL,5),PV(IL+1,5),P(N+IL,5))
  450   CONTINUE
        IF(WT.LT.PYR(0)*WTMAX) GOTO 410
      ENDIF
 
C...Perform two-particle decays in respective CM frame.
  460 DO 480 IL=1,ND-1
        PA=PAWT(PV(IL,5),PV(IL+1,5),P(N+IL,5))
        UE(3)=2D0*PYR(0)-1D0
        PHI=PARU(2)*PYR(0)
        UE(1)=SQRT(1D0-UE(3)**2)*COS(PHI)
        UE(2)=SQRT(1D0-UE(3)**2)*SIN(PHI)
        DO 470 J=1,3
          P(N+IL,J)=PA*UE(J)
          PV(IL+1,J)=-PA*UE(J)
  470   CONTINUE
        P(N+IL,4)=SQRT(PA**2+P(N+IL,5)**2)
        PV(IL+1,4)=SQRT(PA**2+PV(IL+1,5)**2)
  480 CONTINUE
 
C...Lorentz transform decay products to lab frame.
      DO 490 J=1,4
        P(N+ND,J)=PV(ND,J)
  490 CONTINUE
      DO 530 IL=ND-1,1,-1
        DO 500 J=1,3
          BE(J)=PV(IL,J)/PV(IL,4)
  500   CONTINUE
        GA=PV(IL,4)/PV(IL,5)
        DO 520 I=N+IL,N+ND
          BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)
          DO 510 J=1,3
            P(I,J)=P(I,J)+GA*(GA*BEP/(1D0+GA)+P(I,4))*BE(J)
  510     CONTINUE
          P(I,4)=GA*(P(I,4)+BEP)
  520   CONTINUE
  530 CONTINUE
 
C...Check that no infinite loop in matrix element weight.
      NTRY=NTRY+1
      IF(NTRY.GT.800) GOTO 560
 
C...Matrix elements for omega and phi decays.
      IF(MMAT.EQ.1) THEN
        WT=(P(N+1,5)*P(N+2,5)*P(N+3,5))**2-(P(N+1,5)*FOUR(N+2,N+3))**2
     &  -(P(N+2,5)*FOUR(N+1,N+3))**2-(P(N+3,5)*FOUR(N+1,N+2))**2
     &  +2D0*FOUR(N+1,N+2)*FOUR(N+1,N+3)*FOUR(N+2,N+3)
        IF(MAX(WT*WTCOR(9)/P(IP,5)**6,0.001D0).LT.PYR(0)) GOTO 390
 
C...Matrix elements for pi0 or eta Dalitz decay to gamma e+ e-.
      ELSEIF(MMAT.EQ.2) THEN
        FOUR12=FOUR(N+1,N+2)
        FOUR13=FOUR(N+1,N+3)
        WT=(PMST-0.5D0*PMES)*(FOUR12**2+FOUR13**2)+
     &  PMES*(FOUR12*FOUR13+FOUR12**2+FOUR13**2)
        IF(WT.LT.PYR(0)*0.25D0*PMST*(P(IP,5)**2-PMST)**2) GOTO 460
 
C...Matrix element for S0 -> S1 + V1 -> S1 + S2 + S3 (S scalar,
C...V vector), of form cos**2(theta02) in V1 rest frame, and for
C...S0 -> gamma + V1 -> gamma + S2 + S3, of form sin**2(theta02).
      ELSEIF(MMAT.EQ.3.AND.NM.EQ.2) THEN
        FOUR10=FOUR(IP,IM)
        FOUR12=FOUR(IP,N+1)
        FOUR02=FOUR(IM,N+1)
        PMS1=P(IP,5)**2
        PMS0=P(IM,5)**2
        PMS2=P(N+1,5)**2
        IF(KFAS.NE.22) HNUM=(FOUR10*FOUR12-PMS1*FOUR02)**2
        IF(KFAS.EQ.22) HNUM=PMS1*(2D0*FOUR10*FOUR12*FOUR02-
     &  PMS1*FOUR02**2-PMS0*FOUR12**2-PMS2*FOUR10**2+PMS1*PMS0*PMS2)
        HNUM=MAX(1D-6*PMS1**2*PMS0*PMS2,HNUM)
        HDEN=(FOUR10**2-PMS1*PMS0)*(FOUR12**2-PMS1*PMS2)
        IF(HNUM.LT.PYR(0)*HDEN) GOTO 460
 
C...Matrix element for "onium" -> g + g + g or gamma + g + g.
      ELSEIF(MMAT.EQ.4) THEN
        HX1=2D0*FOUR(IP,N+1)/P(IP,5)**2
        HX2=2D0*FOUR(IP,N+2)/P(IP,5)**2
        HX3=2D0*FOUR(IP,N+3)/P(IP,5)**2
        WT=((1D0-HX1)/(HX2*HX3))**2+((1D0-HX2)/(HX1*HX3))**2+
     &  ((1D0-HX3)/(HX1*HX2))**2
        IF(WT.LT.2D0*PYR(0)) GOTO 390
        IF(K(IP+1,2).EQ.22.AND.(1D0-HX1)*P(IP,5)**2.LT.4D0*PARJ(32)**2)
     &  GOTO 390
 
C...Effective matrix element for nu spectrum in tau -> nu + hadrons.
      ELSEIF(MMAT.EQ.41) THEN
        HX1=2D0*FOUR(IP,N+1)/P(IP,5)**2
        HXM=MIN(0.75D0,2D0*(1D0-PS/P(IP,5)))
        IF(HX1*(3D0-2D0*HX1).LT.PYR(0)*HXM*(3D0-2D0*HXM)) GOTO 390
 
C...Matrix elements for weak decays (only semileptonic for c and b)
      ELSEIF((MMAT.EQ.42.OR.MMAT.EQ.43.OR.MMAT.EQ.44.OR.MMAT.EQ.48)
     &  .AND.ND.EQ.3) THEN
        IF(MBST.EQ.0) WT=FOUR(IP,N+1)*FOUR(N+2,N+3)
        IF(MBST.EQ.1) WT=P(IP,5)*P(N+1,4)*FOUR(N+2,N+3)
        IF(WT.LT.PYR(0)*P(IP,5)*PV(1,5)**3/WTCOR(10)) GOTO 390
      ELSEIF(MMAT.EQ.42.OR.MMAT.EQ.43.OR.MMAT.EQ.44.OR.MMAT.EQ.48) THEN
        DO 550 J=1,4
          P(N+NP+1,J)=0D0
          DO 540 IS=N+3,N+NP
            P(N+NP+1,J)=P(N+NP+1,J)+P(IS,J)
  540     CONTINUE
  550   CONTINUE
        IF(MBST.EQ.0) WT=FOUR(IP,N+1)*FOUR(N+2,N+NP+1)
        IF(MBST.EQ.1) WT=P(IP,5)*P(N+1,4)*FOUR(N+2,N+NP+1)
        IF(WT.LT.PYR(0)*P(IP,5)*PV(1,5)**3/WTCOR(10)) GOTO 390
      ENDIF
 
C...Scale back energy and reattach spectator.
  560 IF(MREM.EQ.1) THEN
        DO 570 J=1,5
          PV(1,J)=PV(1,J)/(1D0-PQT)
  570   CONTINUE
        ND=ND+1
        MREM=0
      ENDIF
 
C...Low invariant mass for system with spectator quark gives particle,
C...not two jets. Readjust momenta accordingly.
      IF(MMAT.EQ.31.AND.ND.EQ.3) THEN
        MSTJ(93)=1
        PM2=PYMASS(K(N+2,2))
        MSTJ(93)=1
        PM3=PYMASS(K(N+3,2))
        IF(P(N+2,5)**2+P(N+3,5)**2+2D0*FOUR(N+2,N+3).GE.
     &  (PARJ(32)+PM2+PM3)**2) GOTO 630
        K(N+2,1)=1
        KFTEMP=K(N+2,2)
        CALL PYKFDI(KFTEMP,K(N+3,2),KFLDMP,K(N+2,2))
        IF(K(N+2,2).EQ.0) GOTO 260
        P(N+2,5)=PYMASS(K(N+2,2))
        PS=P(N+1,5)+P(N+2,5)
        PV(2,5)=P(N+2,5)
        MMAT=0
        ND=2
        GOTO 460
      ELSEIF(MMAT.EQ.44) THEN
        MSTJ(93)=1
        PM3=PYMASS(K(N+3,2))
        MSTJ(93)=1
        PM4=PYMASS(K(N+4,2))
        IF(P(N+3,5)**2+P(N+4,5)**2+2D0*FOUR(N+3,N+4).GE.
     &  (PARJ(32)+PM3+PM4)**2) GOTO 600
        K(N+3,1)=1
        KFTEMP=K(N+3,2)
        CALL PYKFDI(KFTEMP,K(N+4,2),KFLDMP,K(N+3,2))
        IF(K(N+3,2).EQ.0) GOTO 260
        P(N+3,5)=PYMASS(K(N+3,2))
        DO 580 J=1,3
          P(N+3,J)=P(N+3,J)+P(N+4,J)
  580   CONTINUE
        P(N+3,4)=SQRT(P(N+3,1)**2+P(N+3,2)**2+P(N+3,3)**2+P(N+3,5)**2)
        HA=P(N+1,4)**2-P(N+2,4)**2
        HB=HA-(P(N+1,5)**2-P(N+2,5)**2)
        HC=(P(N+1,1)-P(N+2,1))**2+(P(N+1,2)-P(N+2,2))**2+
     &  (P(N+1,3)-P(N+2,3))**2
        HD=(PV(1,4)-P(N+3,4))**2
        HE=HA**2-2D0*HD*(P(N+1,4)**2+P(N+2,4)**2)+HD**2
        HF=HD*HC-HB**2
        HG=HD*HC-HA*HB
        HH=(SQRT(HG**2+HE*HF)-HG)/(2D0*HF)
        DO 590 J=1,3
          PCOR=HH*(P(N+1,J)-P(N+2,J))
          P(N+1,J)=P(N+1,J)+PCOR
          P(N+2,J)=P(N+2,J)-PCOR
  590   CONTINUE
        P(N+1,4)=SQRT(P(N+1,1)**2+P(N+1,2)**2+P(N+1,3)**2+P(N+1,5)**2)
        P(N+2,4)=SQRT(P(N+2,1)**2+P(N+2,2)**2+P(N+2,3)**2+P(N+2,5)**2)
        ND=ND-1
      ENDIF
 
C...Check invariant mass of W jets. May give one particle or start over.
  600 IF((MMAT.EQ.42.OR.MMAT.EQ.43.OR.MMAT.EQ.44.OR.MMAT.EQ.48)
     &.AND.IABS(K(N+1,2)).LT.10) THEN
        PMR=SQRT(MAX(0D0,P(N+1,5)**2+P(N+2,5)**2+2D0*FOUR(N+1,N+2)))
        MSTJ(93)=1
        PM1=PYMASS(K(N+1,2))
        MSTJ(93)=1
        PM2=PYMASS(K(N+2,2))
        IF(PMR.GT.PARJ(32)+PM1+PM2) GOTO 610
        KFLDUM=INT(1.5D0+PYR(0))
        CALL PYKFDI(K(N+1,2),-ISIGN(KFLDUM,K(N+1,2)),KFLDMP,KF1)
        CALL PYKFDI(K(N+2,2),-ISIGN(KFLDUM,K(N+2,2)),KFLDMP,KF2)
        IF(KF1.EQ.0.OR.KF2.EQ.0) GOTO 260
        PSM=PYMASS(KF1)+PYMASS(KF2)
        IF((MMAT.EQ.42.OR.MMAT.EQ.48).AND.PMR.GT.PARJ(64)+PSM) GOTO 610
        IF(MMAT.GE.43.AND.PMR.GT.0.2D0*PARJ(32)+PSM) GOTO 610
        IF(MMAT.EQ.48) GOTO 390
        IF(ND.EQ.4.OR.KFA.EQ.15) GOTO 260
        K(N+1,1)=1
        KFTEMP=K(N+1,2)
        CALL PYKFDI(KFTEMP,K(N+2,2),KFLDMP,K(N+1,2))
        IF(K(N+1,2).EQ.0) GOTO 260
        P(N+1,5)=PYMASS(K(N+1,2))
        K(N+2,2)=K(N+3,2)
        P(N+2,5)=P(N+3,5)
        PS=P(N+1,5)+P(N+2,5)
        IF(PS+PARJ(64).GT.PV(1,5)) GOTO 260
        PV(2,5)=P(N+3,5)
        MMAT=0
        ND=2
        GOTO 460
      ENDIF
 
C...Phase space decay of partons from W decay.
  610 IF((MMAT.EQ.42.OR.MMAT.EQ.48).AND.IABS(K(N+1,2)).LT.10) THEN
        KFLO(1)=K(N+1,2)
        KFLO(2)=K(N+2,2)
        K(N+1,1)=K(N+3,1)
        K(N+1,2)=K(N+3,2)
        DO 620 J=1,5
          PV(1,J)=P(N+1,J)+P(N+2,J)
          P(N+1,J)=P(N+3,J)
  620   CONTINUE
        PV(1,5)=PMR
        N=N+1
        NP=0
        NQ=2
        PS=0D0
        MSTJ(93)=2
        PSQ=PYMASS(KFLO(1))
        MSTJ(93)=2
        PSQ=PSQ+PYMASS(KFLO(2))
        MMAT=11
        GOTO 290
      ENDIF
 
C...Boost back for rapidly moving particle.
  630 N=N+ND
      IF(MBST.EQ.1) THEN
        DO 640 J=1,3
          BE(J)=P(IP,J)/P(IP,4)
  640   CONTINUE
        GA=P(IP,4)/P(IP,5)
        DO 660 I=NSAV+1,N
          BEP=BE(1)*P(I,1)+BE(2)*P(I,2)+BE(3)*P(I,3)
          DO 650 J=1,3
            P(I,J)=P(I,J)+GA*(GA*BEP/(1D0+GA)+P(I,4))*BE(J)
  650     CONTINUE
          P(I,4)=GA*(P(I,4)+BEP)
  660   CONTINUE
      ENDIF
 
C...Fill in position of decay vertex.
      DO 680 I=NSAV+1,N
        DO 670 J=1,4
          V(I,J)=VDCY(J)
  670   CONTINUE
        V(I,5)=0D0
  680 CONTINUE
 
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
      ELSEIF(MSTJ(23).GE.1.AND.(MMAT.EQ.32.OR.MMAT.EQ.44).AND.
     &  IABS(K(NSAV+1,2)).LE.10.AND.IABS(K(NSAV+2,2)).LE.10) THEN
        K(NSAV+1,1)=3
        K(NSAV+2,1)=3
        K(NSAV+1,4)=MSTU(5)*(NSAV+2)
        K(NSAV+1,5)=MSTU(5)*(NSAV+2)
        K(NSAV+2,4)=MSTU(5)*(NSAV+1)
        K(NSAV+2,5)=MSTU(5)*(NSAV+1)
        MSTJ(92)=NSAV+1
      ELSEIF(MSTJ(23).GE.1.AND.(MMAT.EQ.32.OR.MMAT.EQ.44).AND.
     &  IABS(K(NSAV+1,2)).LE.20.AND.IABS(K(NSAV+2,2)).LE.20) THEN
        MSTJ(92)=NSAV+1
      ELSEIF(MSTJ(23).GE.1.AND.MMAT.EQ.33.AND.IABS(K(NSAV+2,2)).EQ.21)
     &  THEN
        K(NSAV+1,1)=3
        K(NSAV+2,1)=3
        K(NSAV+3,1)=3
        KCP=PYCOMP(K(NSAV+1,2))
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
 
C...Mark decayed particle; special option for B-Bbar mixing.
      IF(K(IP,1).EQ.5) K(IP,1)=15
      IF(K(IP,1).LE.10) K(IP,1)=11
      IF(MMIX.EQ.1.AND.MSTJ(26).EQ.2.AND.K(IP,1).EQ.11) K(IP,1)=12
      K(IP,4)=NSAV+1
      K(IP,5)=N
 
      RETURN
      END
