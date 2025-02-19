C*********************************************************************
 
C...PYRHAD
C...Initialize R-hadron data, like names (for event listing) and 
C...charges. So far nothing but the minimum required.
C...Feel free to choose between hadron-like names or 
C...flavour-content-based ones.

      SUBROUTINE PYGLRHAD
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
C...Parameter statement to help give large particle numbers
C...(left- and righthanded SUSY, excited fermions).
      PARAMETER (KSUSY1=1000000,KSUSY2=2000000,KEXCIT=4000000)
C...Commonblocks.
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT4/CHAF(500,2)
      CHARACTER CHAF*16
      SAVE /PYDAT2/,/PYDAT4/

C...Local R-hadron data arrays, to fill into the global ones.
      DIMENSION KFRH(20),KCHGRH(20),KANTRH(20)
      CHARACTER*16 CHRHA(20),CHRHB(20)
C...Codes.
      DATA KFRH/1000993,1009213,1009313,1009323,1009113,1009223,
     &1009333,1091114,1092114,1092214,1092224,1093114,1093214,
     &1093224,1093314,1093324,1093334,3*0/
C...Three times charge.
      DATA KCHGRH/0,3,0,3,0,0,0,-3,0,3,6,-3,0,3,-3,0,-3,3*0/
C...Existence of a distinct antiparticle.
      DATA KANTRH/0,3*1,3*0,10*1,3*0/
C...One possibility: hadron-like names.
      DATA CHRHA/'~g glueball','~g rho+','~g K*0','~g K*+',
     &'~g rho0','~g omega','~g phi','~g Delta-','~g Delta0',
     &'~g Delta+','~g Delta++','~g Sigma*-','~g Sigma*0',
     &'~g Sigma*+','~g Xi*-','~g Xi*0 ','~g Omega-',3*' '/
      DATA CHRHB/' ','~g rho-','~g K*bar0','~g K*-',3*' ',
     &'~g Deltabar+','~g Deltabar0','~g Deltabar-','~g Deltabar--',
     &'~g Sigma*bar+','~g Sigma*bar0','~g Sigma*bar-','~g Xi*bar+',
     &'~g Xi*bar0','~g Omegabar+',3*' '/
C...Another possibility: flavour-contents-based names.
c      DATA CHRHA/'~g g','~g u dbar','~g d sbar','~g u sbar',
c     &'~g d dbar','~g u ubar','~g s sbar','~g ddd','~g udd',
c     &'~g uud','~g uuu','~g sdd','~g sud','~g suu','~g ssd',
c     &'~g ssu','~g sss',3*' '/
c      DATA CHRHB/' ','~g d ubar','~g s dbar','~g s ubar',3*' ',
c     &'~g ddd bar','~g udd bar','~g uud bar','~g uuu bar',
c     &'~g sdd bar','~g sud bar','~g suu bar','~g ssd bar',
c     &'~g ssu bar','~g sss bar',3*' '/

C...Fill in data.
      DO 100 I=1,17
        KC=400+I
        KCHG(KC,1)=KCHGRH(I)
        KCHG(KC,2)=0
        KCHG(KC,3)=KANTRH(I)
        KCHG(KC,4)=KFRH(I)
        CHAF(KC,1)=CHRHA(I)
        CHAF(KC,2)=CHRHB(I)
  100 CONTINUE    
  
      RETURN
      END
 

C*********************************************************************
 
C...PYGLFR
C...Fragments the string near to a gluino, to form a gluino-hadron, 
C...either by producing a new g-g pair or two new q-qbar ones.
 
      SUBROUTINE PYGLFR
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
C...Parameter statement to help give large particle numbers
C...(left- and righthanded SUSY, excited fermions).
      PARAMETER (KSUSY1=1000000,KSUSY2=2000000,KEXCIT=4000000)
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
C...Note that dimensions below grew from 4000 to 8000 in Pythia 6.2!
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      SAVE /PYJETS/,/PYDAT1/,/PYDAT2/,/PYDAT3/,/PYPARS/,/PYINT1/,
     &/PYINT2/
C...Local array.
      DIMENSION PSUM(5),KFSAV(2),PMSAV(2),PSAV(2,5) 

C...Initialise some variables that may be used before being set
      PMNEW=0.0D0
      IRECSV=0
      IS=0

C...Free parameter: relative probability for gluino-gluon-ball.
C...(But occasional low-mass string will never become it anyway.)
      PROBGG=0.1D0
 
C...Free parameter: gluon constituent mass.
      PMGLU=0.7D0
 
C...Free parameter: max kinetic energy in gluino-hadron.
      PMKIN=0.5D0

C...Switch off popcorn baryon production. (Not imperative, but more
C...failed events when popcorn is allowed.)
      MSTJ12=MSTJ(12)
      MSTJ(12)=1

C...Convenient shorthand.
      KFGL=KSUSY1+21

C...Loopback point for serious problems, with new try.
      LOOP=0
      CALL PYEDIT(21)
      CHGSAV=PYP(0,6)
   90 LOOP=LOOP+1
      IF(LOOP.GT.1) CALL PYEDIT(22)

C...Take copy of string system(s), leaving extra free slot after gluino.
C...(Eventually to be overwritten by one q and one qbar string break.)
      NOLD=N
      NGLUI=0
      DO 120 I=1,NOLD
        ICOPY=0
        IF(K(I,1).EQ.2) ICOPY=1
        IF(K(I,1).EQ.1.AND.I.GE.2) THEN
          IF(K(I-1,1).EQ.12) ICOPY=1
        ENDIF
        IF(ICOPY.EQ.1) THEN  
          N=N+1
          DO 100 J=1,5
            K(N,J)=K(I,J)
            P(N,J)=P(I,J)
            V(N,J)=V(I,J)
  100     CONTINUE
          K(I,1)=K(I,1)+10
          K(I,4)=N
          K(I,5)=N
          K(N,3)=I
          IF(K(I,2).EQ.KFGL) THEN
            NGLUI=NGLUI+1  
            N=N+1
            DO 110 J=1,5
              K(N,J)=K(N-1,J)
              P(N,J)=0D0
              V(N,J)=V(I,J)
  110       CONTINUE
            K(I,5)=N
            K(N,2)=21
          ENDIF
        ENDIF
  120 CONTINUE

C...Loop over (up to) two gluinos per event.
      DO 300 IGLUI=1,NGLUI

C...Identify position of gluino (randomize order of treatment).
        IGL=0
        NGL=0
        DO 130 I=1,N
          IF(K(I,1).EQ.2.AND.K(I,2).EQ.KFGL) THEN
            NGL=NGL+1
            IF(IGLUI.EQ.NGLUI) THEN
              IGL=I
            ELSEIF(NGL.EQ.1) THEN
              IF(PYR(0).LT.0.5D0) IGL=I
            ELSEIF(IGL.EQ.0) THEN
              IGL=I
            ENDIF
          ENDIF
  130   CONTINUE

C...Identify range of partons on string the gluino belongs to. 
        IMIN=IGL
  140   IMIN=IMIN-1
        IF(K(IMIN-1,1).EQ.2) GOTO 140
        IMAX=IGL
  150   IMAX=IMAX+1
        IF(K(IMAX,1).EQ.2) GOTO 150
 
C...Find mass of this gluino-string. 
        DO 170 J=1,5
          PSUM(J)=0D0
          DO 160 I=IMIN,IMAX
            PSUM(J)=PSUM(J)+P(I,J)
  160     CONTINUE
  170   CONTINUE
        PSUM(5)=SQRT(MAX(0D0,PSUM(4)**2-PSUM(1)**2-PSUM(2)**2-
     &  PSUM(3)**2))
 
C...If low-mass, then consider gluino-hadron already formed.
        IF(PSUM(5).LE.P(IGL,5)+P(IMIN,5)+P(IMAX,5)+PMKIN) THEN
          DO 180 I=IMIN,IMAX
            K(I,1)=15+IGLUI
  180     CONTINUE
          GOTO 300
        ENDIF    

C...Else break string by production of new gg or two new qqbar pairs.
C...(Also diquarks allowed, but not popcorn, and not two adjacent.)
        IF(PYR(0).LT.PROBGG) THEN
C...Let a gluon occupy two slots, to make administration work the same
C...way as for the qqbar case.
          KFSAV(1)=21
          KFSAV(2)=21
          PMSAV(1)=0.5D0*PMGLU  
          PMSAV(2)=0.5D0*PMGLU  
        ELSE
  185     CALL PYDCYK(K(IMIN,2),0,KFSAV(1),KFDUM)
          CALL PYDCYK(K(IMAX,2),0,KFSAV(2),KFDUM)
          IF(IABS(KFSAV(1)).GT.10.AND.IABS(KFSAV(2)).GT.10) GOTO 185
          IF(IABS(KFSAV(1)).GT.10.AND.IABS(K(IGL-1,2)).GT.10) GOTO 185
          IF(IABS(KFSAV(2)).GT.10.AND.IABS(K(IGL+2,2)).GT.10) GOTO 185
          KFSAV(1)=ISIGN(MOD(IABS(KFSAV(1)),10000),KFSAV(1))
          KFSAV(2)=ISIGN(MOD(IABS(KFSAV(2)),10000),KFSAV(2))
          MSTJ(93)=1 
          PMSAV(1)=PYMASS(KFSAV(1))         
          MSTJ(93)=1 
          PMSAV(2)=PYMASS(KFSAV(2))
        ENDIF         

C...Mass of gluino-hadron.
        PMGSAV=P(IGL,5)
        PMGB=P(IGL,5)+PMSAV(1)+PMSAV(2)

C...Pick at random order in which both sides of gluino string break.
        ISIDE=INT(1.5D0+PYR(0))
        DO 220 ISDE=1,2
          IF(ISDE.EQ.1) IS=ISIDE
          IF(ISDE.EQ.2) IS=3-ISIDE

C...Pick momentum sharing according to fragmentation function as if bottom.
          PMBSAV=PARF(105)
          PARF(105)=PMGSAV
          CALL PYZDIS(5,0,PMGB**2,ZGL)
          PARF(105)=PMBSAV 
          ZGL=MAX(0.9D0,MIN(0.9999D0,ZGL)) 
          DO 190 J=1,5
            PSAV(IS,J)=(1D0-ZGL)*P(IGL,J)
            P(IGL,J)=ZGL*P(IGL,J)
  190    CONTINUE

C...Target gluino-hadron mass for this stage of momentum reshuffling.
          PMOLD=P(IGL,5)
          IF(ISDE.EQ.1) PMNEW=PMGSAV+PMSAV(IS)
          IF(ISDE.EQ.2) PMNEW=PMGB 

C...Recoiling parton from which to shuffle momentum. System momentum.
          IF(IS.EQ.1) IREC=IGL-1
          IF(IS.EQ.2) IREC=IGL+2
  200     DO 210 J=1,4
            PSUM(J)=P(IGL,J)+P(IREC,J)
  210     CONTINUE           

C...Boost to rest frame of system, and align gluino along +z axis.
          CALL PYROBO(IGL,IGL,0D0,0D0,-PSUM(1)/PSUM(4),
     &    -PSUM(2)/PSUM(4),-PSUM(3)/PSUM(4))
          CALL PYROBO(IREC,IREC,0D0,0D0,-PSUM(1)/PSUM(4),
     &    -PSUM(2)/PSUM(4),-PSUM(3)/PSUM(4))
          PHI=PYANGL(P(IGL,1),P(IGL,2))
          CALL PYROBO(IGL,IGL,0D0,-PHI,0D0,0D0,0D0)
          CALL PYROBO(IREC,IREC,0D0,-PHI,0D0,0D0,0D0)
          THETA=PYANGL(P(IGL,3),P(IGL,1)) 
          CALL PYROBO(IGL,IGL,-THETA,0D0,0D0,0D0,0D0)
          CALL PYROBO(IREC,IREC,-THETA,0D0,0D0,0D0,0D0)

C...Calculate new kinematics in this frame, for desired gluino mass.
          ETOT=P(IGL,4)+P(IREC,4)
          IF(ETOT.GT.PMNEW+P(IREC,5)) THEN
            IFAIL=0
            PZNEW=0.5D0*SQRT(MAX(0D0,(ETOT**2-PMNEW**2-P(IREC,5)**2)**2-
     &      4D0*PMNEW**2*P(IREC,5)**2))/ETOT
            P(IGL,3)=PZNEW
            P(IGL,4)=SQRT(PZNEW**2+PMNEW**2)
            P(IGL,5)=PMNEW
            P(IREC,3)=-PZNEW
            P(IREC,4)=SQRT(PZNEW**2+P(IREC,5)**2)

C...If not enough momentum, take what can be taken.
          ELSE
            IFAIL=1
            PMOLD=ETOT-P(IREC,5)
            P(IGL,3)=0D0
            P(IGL,4)=PMOLD
            P(IGL,5)=PMOLD
            P(IREC,3)=0D0
            P(IREC,4)=P(IREC,5)
          ENDIF

C...Bost back to lab frame.
          CALL PYROBO(IGL,IGL,THETA,PHI,PSUM(1)/PSUM(4),
     &    PSUM(2)/PSUM(4),PSUM(3)/PSUM(4))
          CALL PYROBO(IREC,IREC,THETA,PHI,PSUM(1)/PSUM(4),
     &    PSUM(2)/PSUM(4),PSUM(3)/PSUM(4))

C...Loop back when not enough momentum could be shuffled.
C...(As long as there is something left on either side.)
          IF(IFAIL.EQ.1) THEN
  215       IF(IS.EQ.1.AND.IREC.GT.IMIN) THEN
              IREC=IREC-1
              GOTO 200
            ELSEIF(IS.EQ.2.AND.IREC.LT.IMAX) THEN
              IREC=IREC+1
              GOTO 200
            ELSEIF(ISDE.EQ.2.AND.IS.EQ.3-ISIDE) THEN
              IS=ISIDE
              IREC=IRECSV
              GOTO 215
            ENDIF
          ENDIF

C...End loop over fragmentation of two sides around gluino.
         IRECSV=IREC
  220   CONTINUE

C...New slot at end of record for gluino R-hadron.
        DO 230 J=1,5
          K(N+1,J)=0
          P(N+1,J)=P(IGL,J)
          V(N+1,J)=V(IGL,J)
  230   CONTINUE
 
C...Status and code of this slot.
        K(N+1,1)=15+IGLUI
        KFSVMX=MAX(IABS(KFSAV(1)),IABS(KFSAV(2)))
        KFSVMN=MIN(IABS(KFSAV(1)),IABS(KFSAV(2)))
C...Gluino-ball.
        IF(KFSVMX.EQ.21) THEN
          K(N+1,2)=KSUSY1+993
C...Gluino-meson.
        ELSEIF(KFSVMX.LT.10) THEN
          K(N+1,2)=KSUSY1+9000+100*KFSVMX+10*KFSVMN+3
          IF(KFSVMX.EQ.KFSVMN) THEN
          ELSEIF(MOD(KFSVMX,2).EQ.0) THEN
            IF(KFSVMX.EQ.KFSAV(1).OR.KFSVMX.EQ.KFSAV(2))
     &      K(N+1,2)=-K(N+1,2) 
          ELSE
            IF(KFSVMX.EQ.-KFSAV(1).OR.KFSVMX.EQ.-KFSAV(2))
     &      K(N+1,2)=-K(N+1,2) 
          ENDIF             
C...Gluino-baryon.
        ELSE
          KFSVX1=KFSVMX/1000
          KFSVX2=MOD(KFSVMX/100,10)
          KFA=MAX(KFSVX1,KFSVX2,KFSVMN)
          KFC=MIN(KFSVX1,KFSVX2,KFSVMN)
          KFB=KFSVX1+KFSVX2+KFSVMN-KFA-KFC
          K(N+1,2)=SIGN(KSUSY1+90000+1000*KFA+100*KFB+10*KFC+4,
     &    -KFSAV(1))
        ENDIF
        K(N+1,3)=K(IGL,3)
        N=N+1
        
C...Code and momentum of two new string endpoints.
        K(IGL,2)=KFSAV(1)
        K(IGL+1,2)=KFSAV(2)
        IF(KFSAV(1).NE.21) K(IGL,1)=1
        DO 240 J=1,5
          P(IGL,J)=PSAV(1,J)
          P(IGL+1,J)=PSAV(2,J)
  240   CONTINUE
 
C...End of loop over two gluinos.
  300 CONTINUE

C...Cleanup: remove zero-energy gluons.
      NNOW=N
      N=NOLD
      DO 330 I=NOLD+1,NNOW
        IF(K(I,2).EQ.21.AND.P(I,4).LT.1D-10) THEN
        ELSEIF(I.EQ.N+1) THEN
          N=N+1
        ELSE
          N=N+1
          DO 320 J=1,5
            K(N,J)=K(I,J)
            P(N,J)=P(I,J)
            V(N,J)=V(I,J)
  320     CONTINUE
        ENDIF
  330 CONTINUE

C...Finish off with standard hadronization.
C...Note that the R-hadrons are not affected here, since they
C...(deliberately erroneously) have been bookkept as decayed.
      CALL PYEXEC

C...Restore code of gluino-hadrons to non-fragmented numbers.
      N6=0
      N7=0
      DO 340 I=1,N
        IF(K(I,1).EQ.16.OR.K(I,1).EQ.17) K(I,1)=K(I,1)-10
        IF(K(I,1).EQ.6) N6=N6+1
        IF(K(I,1).EQ.7) N7=N7+1
  340 CONTINUE
      IF(N6.GT.1.OR.N7.GT.1) MSTU(24)=1

C...Extracheck charge.
      CHGNEW=PYP(0,6)
      IF(ABS(CHGNEW-CHGSAV).GT.0.1D0) MSTU(24)=1

C...In case of trouble, make up to five attempts.
      IF(MSTU(24).NE.0.AND.LOOP.LT.5) THEN
        WRITE(*,*) '     ...give it new try...'
        MSTU(23)=MSTU(23)-1
        GOTO 90
      ELSEIF(MSTU(24).NE.0) THEN
        WRITE(*,*) '     ...but still fail after repeated attempts!'
      ELSEIF(MSTU(24).EQ.0.AND.LOOP.GT.1) THEN
        WRITE(*,*) '     ...and now it worked!'
      ENDIF

C...Finished. Restore baryon production model.
      MSTJ(12)=MSTJ12
      
      RETURN
      END
 
