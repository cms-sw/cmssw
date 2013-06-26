C*********************************************************************
 
C...PYRHAD
C...Initialize stop_1 R-hadron data, like names (for event listing) 
C...and charges. So far nothing but the minimum required.
C...Feel free to choose between hadron-like names or 
C...flavour-content-based ones.

      SUBROUTINE PYSTRHAD
 
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
      DATA KFRH/1000612,1000622,1000632,1000642,1000652,1006113,
     &1006211,1006213,1006223,1006311,1006313,1006321,1006323,
     &1006333,6*0/
C...Three times charge.
      DATA KCHGRH/3,0,3,0,3,0,3,3,6,0,0,3,3,0,6*0/
C...Existence of a distinct antiparticle.
      DATA KANTRH/14*1,6*0/
C...One possibility: hadron-like names.
      DATA CHRHA/'~T+','~T0','~T_s+','~T_c0','~T_b+','~T_dd10',
     &'~T_ud0+','~T_ud1+','~T_uu1++','~T_sd00','~T_sd10',
     &'~T_su0+','~T_su1+','~T_ss10',6*' '/
      DATA CHRHB/'~Tbar-','~Tbar0','~Tbar_s-','~Tbar_c0','~Tbar_b-',
     &'~Tbar_dd10','~Tbar_ud0-','~Tbar_ud1-','~Tbar_uu1--',
     &'~Tbar_sd00','~Tbar_sd10','~Tbar_su0-','~Tbar_su1-',
     &'~Tbar_ss10',6*' '/
C...Another possibility: flavour-contents-based names.
C      DATA CHRHA/'~t dbar','~t ubar','~t sbar','~t cbar','~t bbar',
C     &'~t dd1','~t ud0','~t ud1','~t uu1','~t sd0','~t sd1','~t su0',
C     &'~t su1','~t ss1',6*' '/
C      DATA CHRHB/'~tbar d','~tbar u','~tbar s','~tbar c','~tbar b',
C     &'~tbar dd1bar','~tbar ud0bar','~tbar ud1bar','~tbar uu1bar',
C     &'~tbar sd0bar','~tbar sd1bar','~tbar su0bar','~tbar su1bar',
C     &'~tbar ss1bar',6*' '/

C...Fill in data.
      DO 100 I=1,14
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
 
C...PYSTFR
C...Fragments the string near to a stop, to form a stop-hadron, 
C...by producing a new q-qbar pair.
 
      SUBROUTINE PYSTFR(IERR)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      INTEGER PYCOMP
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
      DIMENSION PSUM(5),PSAV(5),IPOSST(10) 

C...Default is no error.
      IERR=0
 
C...Free parameter: max kinetic energy in gluino-hadron.
      PMKIN=0.5D0
 
C...Free parameter: part of stop mass that does not participate
C...in weak decay.
      PMINAC=0.5D0

C...Switch off popcorn baryon production. (Not imperative, but more
C...failed events when popcorn is allowed.)
      MSTJ12=MSTJ(12)
      MSTJ(12)=1

C...Convenient shorthand.
      KFST=KSUSY1+6
      KCST=PYCOMP(KFST)
      KFGL=KSUSY1+21
      
C...Loopback point for serious problems, with new try.
      LOOP=0
      CALL PYEDIT(21)
      CHGSAV=PYP(0,6)
   90 LOOP=LOOP+1
      IF(LOOP.GT.1) CALL PYEDIT(22)

C...Give up when too much problems.
      IF(LOOP.GT.5) THEN
        WRITE(*,*) ' Problematical event skipped'
        IERR=1
        RETURN
      ENDIF

C...Take copy of string system(s).
      NOLD=N
      NSTOP=0
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
          IF(IABS(K(I,2)).EQ.KFST) THEN
            NSTOP=NSTOP+1
            IPOSST(NSTOP)=N
          ENDIF   
        ENDIF
  120 CONTINUE
      NTMP=N

C...Loop over (up to) two stops per event.
C...Identify position of stop (randomize order of treatment).
      IRNST=INT(1.5D0+PYR(0))
      DO 300 ISTOP=1,NSTOP
        IST=IPOSST(1)
        IF(NSTOP.EQ.2.AND.ISTOP.NE.IRNST) IST=IPOSST(2)

C...Identify range of partons on string the stop belongs to. 
        IMIN=IST+1
  140   IMIN=IMIN-1
        IF(K(IMIN-1,1).EQ.2) GOTO 140
        IMAX=IST-1
  150   IMAX=IMAX+1
        IF(K(IMAX,1).EQ.2) GOTO 150
        IOTHER=IMAX
        IF(IST.EQ.IMAX) IOTHER=IMIN  
 
C...Find mass of this stop-string. 
        DO 170 J=1,5
          PSUM(J)=0D0
          DO 160 I=IMIN,IMAX
            PSUM(J)=PSUM(J)+P(I,J)
  160     CONTINUE
  170   CONTINUE
        PSUM(5)=SQRT(MAX(0D0,PSUM(4)**2-PSUM(1)**2-PSUM(2)**2-
     &  PSUM(3)**2))
 
C...If low-mass, then consider stop-hadron already formed.
        IF(PSUM(5).LE.P(IST,5)+P(IOTHER,5)+PMKIN) THEN
          DO 180 I=IMIN,IMAX
            K(I,1)=2
            IF(I.EQ.IMAX) K(I,1)=1
            IF(I.NE.IST) THEN
              DO 175 J=1,5
                P(IST,J)=P(IST,J)+P(I,J)
                P(I,J)=0D0
  175         CONTINUE
            ENDIF
  180     CONTINUE
          P(IST,5)=SQRT(MAX(0D0,P(IST,4)**2-P(IST,1)**2-P(IST,2)**2-
     &    P(IST,3)**2))
          GOTO 300
        ENDIF    

C...Else break string by production of new qqbar pair.
C...(Also diquarks allowed, but not popcorn.)
        INFLAV=ISIGN(4,K(IST,2))
        CALL PYDCYK(INFLAV,0,KFSAV,KFDUM)
        KFSAV=ISIGN(MOD(IABS(KFSAV),10000),KFSAV)
        MSTJ(93)=1 
        PMSAV=PYMASS(KFSAV)         

C...Mass of stop-hadron.
        PMSSAV=P(IST,5)
        PMSHAD=P(IST,5)+PMSAV

C...Pick momentum sharing according to fragmentation function as if bottom.
        PMBSAV=PARF(105)
        PARF(105)=PMSSAV
        CALL PYZDIS(5,0,PMSHAD**2,ZST)
        PARF(105)=PMBSAV 
        ZST=MAX(0.9D0,MIN(0.9999D0,ZST)) 
        DO 190 J=1,5
          PSAV(J)=(1D0-ZST)*P(IST,J)
          P(IST,J)=ZST*P(IST,J)
  190  CONTINUE

C...Recoiling parton from which to shuffle momentum. System momentum.
        IF(IST.EQ.IMIN) IREC=IST+1
        IF(IST.EQ.IMAX) IREC=IST-1
  200   DO 210 J=1,4
          PSUM(J)=P(IST,J)+P(IREC,J)
  210   CONTINUE           

C...Boost to rest frame of system, and align stop along +z axis.
        CALL PYROBO(IST,IST,0D0,0D0,-PSUM(1)/PSUM(4),
     &  -PSUM(2)/PSUM(4),-PSUM(3)/PSUM(4))
        CALL PYROBO(IREC,IREC,0D0,0D0,-PSUM(1)/PSUM(4),
     &  -PSUM(2)/PSUM(4),-PSUM(3)/PSUM(4))
        PHI=PYANGL(P(IST,1),P(IST,2))
        CALL PYROBO(IST,IST,0D0,-PHI,0D0,0D0,0D0)
        CALL PYROBO(IREC,IREC,0D0,-PHI,0D0,0D0,0D0)
        THETA=PYANGL(P(IST,3),P(IST,1)) 
        CALL PYROBO(IST,IST,-THETA,0D0,0D0,0D0,0D0)
        CALL PYROBO(IREC,IREC,-THETA,0D0,0D0,0D0,0D0)

C...Calculate new kinematics in this frame, for desired stop hadron mass.
        ETOT=P(IST,4)+P(IREC,4)
        PMREC=P(IREC,5)
        IF(K(IREC,2).NE.21.AND.IABS(K(IREC,2)).NE.KFST) THEN
          MSTJ(93)=1 
          PMREC=PYMASS(K(IREC,2))         
        ENDIF 
        IF(ETOT.GT.PMSHAD+PMREC) THEN
          IFAIL=0
          PZNEW=0.5D0*SQRT(MAX(0D0,(ETOT**2-PMSHAD**2-PMREC**2)**2-
     &    4D0*PMSHAD**2*PMREC**2))/ETOT
          P(IST,3)=PZNEW
          P(IST,4)=SQRT(PZNEW**2+PMSHAD**2)
          P(IST,5)=PMSHAD
          P(IREC,3)=-PZNEW
          P(IREC,4)=SQRT(PZNEW**2+PMREC**2)
          P(IREC,5)=PMREC

C...If not enough momentum, take what can be taken.
        ELSE
          IFAIL=1
          P(IST,3)=0D0
          P(IST,4)=ETOT-PMREC
          P(IST,5)=P(IST,4)
          P(IREC,3)=0D0
          P(IREC,4)=PMREC
          P(IREC,5)=PMREC
        ENDIF

C...Bost back to lab frame.
        CALL PYROBO(IST,IST,THETA,PHI,PSUM(1)/PSUM(4),
     &  PSUM(2)/PSUM(4),PSUM(3)/PSUM(4))
        CALL PYROBO(IREC,IREC,THETA,PHI,PSUM(1)/PSUM(4),
     &  PSUM(2)/PSUM(4),PSUM(3)/PSUM(4))

C...Loop back when not enough momentum could be shuffled.
C...(As long as there is something left.)
        IF(IFAIL.EQ.1) THEN
          IF(IST.EQ.IMIN.AND.IREC.LT.IMAX) THEN
            IREC=IREC+1
            GOTO 200
          ELSEIF(IST.EQ.IMAX.AND.IREC.GT.IMIN) THEN
            IREC=IREC-1
            GOTO 200
          ENDIF
        ENDIF

C...Particle code for stop-hadron.
        KFSTHD=0 
        IF(K(IST,2).GT.0) THEN
          IF(KFSAV.LE.-1.AND.KFSAV.GE.-5) KFSTHD=KSUSY1+600-10*KFSAV+2
          IF(KFSAV.GE.1103.AND.KFSAV.LE.3303) KFSTHD=KSUSY1+6000+
     &    (KFSAV/10)+MOD(KFSAV,10)
        ELSE
          IF(KFSAV.GE.1.AND.KFSAV.LE.5) KFSTHD=KSUSY1+600+10*KFSAV+2
          IF(KFSAV.LE.-1103.AND.KFSAV.GE.-3303) KFSTHD=KSUSY1+6000+
     &    (IABS(KFSAV)/10)+MOD(IABS(KFSAV),10)
          KFSTHD=-KFSTHD
        ENDIF
        IF(KFSTHD.EQ.0) THEN
          WRITE(*,*) ' Failed to find R-hadron code from ',
     &    K(IST,2),KFSAV 
          IERR=1 
          RETURN
        ENDIF

C...New slot at end of record for stop-hadron
        DO 230 J=1,5
          K(N+1,J)=0
          P(N+1,J)=P(IST,J)
          V(N+1,J)=V(IST,J)
  230   CONTINUE
        K(N+1,1)=5+ISTOP
        K(N+1,2)=KFSTHD
        K(N+1,3)=K(IST,3)
        N=N+1
        
C...Code and momentum of new string endpoint.
        K(IST,2)=-KFSAV
        DO 240 J=1,5
          P(IST,J)=PSAV(J)
  240   CONTINUE
 
C...End of loop over two stops.
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
      NNOW=N

C...Check that no low-mass system of diquark-antidiquark kind,
C...or very low-mass of any kind.
      KFBEG=0
      DO 332 J=1,5
        PSUM(J)=0D0
  332 CONTINUE
      DO 338 I=NOLD+1,NNOW
        DO 334 J=1,4
          PSUM(J)=PSUM(J)+P(I,J)
  334   CONTINUE
        IF(KFBEG.EQ.0) THEN
          KFBEG=IABS(K(I,2))
          MSTJ(93)=1 
          PSUM(5)=PSUM(5)+PYMASS(K(I,2))         
        ELSEIF(K(I,1).EQ.1) THEN
          KFEND=IABS(K(I,2))
          MSTJ(93)=1 
          PSUM(5)=PSUM(5)+PYMASS(K(I,2))         
          DELTA=SQRT(MAX(0D0,PSUM(4)**2-PSUM(1)**2-PSUM(2)**2-
     &    PSUM(3)**2))-PSUM(5)
          IF(KFBEG.GT.10.AND.KFBEG.LT.10000.AND.KFEND.GT.10.AND.
     &    KFEND.LT.10000.AND.DELTA.LT.PARJ(32).AND.(KFBEG.NE.21
     &    .AND.KFEND.NE.21)) GOTO 90
          IF(DELTA.LT.0D0) GOTO 90
          KFBEG=0
          DO 336 J=1,5
            PSUM(J)=0D0
  336     CONTINUE
        ENDIF
  338 CONTINUE

C...Finished with stop hadronization. Restore baryon production model.
      MSTJ(12)=MSTJ12

C...Now hadronize everything else. Some cheating to allow sensible
C...momentum shuffling.
      MSTJ16=MSTJ(16)
      MSTJ(16)=0
      CALL PYEXEC
      MSTJ(16)=MSTJ16
      IF(MSTU(24).NE.0) THEN
        WRITE(*,*) ' Event to be skipped'
        IERR=1 
      ENDIF

      RETURN
      END
