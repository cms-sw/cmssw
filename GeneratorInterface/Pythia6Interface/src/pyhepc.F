 
C*********************************************************************
 
C...PYHEPC
C...Converts PYTHIA event record contents to or from
C...the standard event record commonblock.
 
      SUBROUTINE PYHEPC(MCONV)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYCOMP
C...Commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      SAVE /PYJETS/,/PYDAT1/,/PYDAT2/
C...HEPEVT commonblock.
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
      
C...Store HEPEVT commonblock size (for interfacing issues).
      MSTU(8)=NMXHEP
      
c init variable(s)
      inew = 1
 
C...Conversion from PYTHIA to standard, the easy part.
      IF(MCONV.EQ.1) THEN
        NEVHEP=0
        IF(N.GT.NMXHEP) CALL PYERRM(8,
     &  '(PYHEPC:) no more space in /HEPEVT/')
        NHEP=MIN(N,NMXHEP)
        DO 150 I=1,NHEP
          ISTHEP(I)=0
          IF(K(I,1).GE.1.AND.K(I,1).LE.10) ISTHEP(I)=1
          IF(K(I,1).GE.11.AND.K(I,1).LE.20) ISTHEP(I)=2
          IF(K(I,1).GE.21.AND.K(I,1).LE.30) ISTHEP(I)=3
          IF(K(I,1).GE.31.AND.K(I,1).LE.100) ISTHEP(I)=K(I,1)
          IDHEP(I)=K(I,2)
          JMOHEP(1,I)=K(I,3)
          JMOHEP(2,I)=0
          IF(K(I,1).NE.3.AND.K(I,1).NE.13.AND.K(I,1).NE.14) THEN
            JDAHEP(1,I)=K(I,4)
            JDAHEP(2,I)=K(I,5)
          ELSE
            JDAHEP(1,I)=0
            JDAHEP(2,I)=0
          ENDIF
          DO 100 J=1,5
            PHEP(J,I)=P(I,J)
  100     CONTINUE
          DO 110 J=1,4
            VHEP(J,I)=V(I,J)
  110     CONTINUE
 
C...Check if new event (from pileup).
          IF(I.EQ.1) THEN
            INEW=1
          ELSE
            IF(K(I,1).EQ.21.AND.K(I-1,1).NE.21) INEW=I
          ENDIF
 
C...Fill in missing mother information.
          IF(I.GE.INEW+2.AND.K(I,1).EQ.21.AND.K(I,3).EQ.0) THEN
            IMO1=I-2
  120       IF(IMO1.GT.INEW.AND.K(IMO1+1,1).EQ.21.AND.K(IMO1+1,3).EQ.0)
     &      THEN
              IMO1=IMO1-1
              GOTO 120
            ENDIF
            JMOHEP(1,I)=IMO1
            JMOHEP(2,I)=IMO1+1
          ELSEIF(K(I,2).GE.91.AND.K(I,2).LE.93) THEN
            I1=K(I,3)-1
  130       I1=I1+1
            IF(I1.GE.I) CALL PYERRM(8,
     &      '(PYHEPC:) translation of inconsistent event history')
            IF(I1.LT.I.AND.K(I1,1).NE.1.AND.K(I1,1).NE.11) GOTO 130
            KC=PYCOMP(K(I1,2))
            IF(I1.LT.I.AND.KC.EQ.0) GOTO 130
            IF(I1.LT.I.AND.KCHG(KC,2).EQ.0) GOTO 130
            JMOHEP(2,I)=I1
          ELSEIF(K(I,2).EQ.94) THEN
            NJET=2
            IF(NHEP.GE.I+3.AND.K(I+3,3).LE.I) NJET=3
            IF(NHEP.GE.I+4.AND.K(I+4,3).LE.I) NJET=4
            JMOHEP(2,I)=MOD(K(I+NJET,4)/MSTU(5),MSTU(5))
            IF(JMOHEP(2,I).EQ.JMOHEP(1,I)) JMOHEP(2,I)=
     &      MOD(K(I+1,4)/MSTU(5),MSTU(5))
          ENDIF
 
C...Fill in missing daughter information.
          IF(K(I,2).EQ.94.AND.MSTU(16).NE.2) THEN
            DO 140 I1=JDAHEP(1,I),JDAHEP(2,I)
              I2=MOD(K(I1,4)/MSTU(5),MSTU(5))
              JDAHEP(1,I2)=I
  140       CONTINUE
          ENDIF
          IF(K(I,2).GE.91.AND.K(I,2).LE.94) GOTO 150
          I1=JMOHEP(1,I)
          IF(I1.LE.0.OR.I1.GT.NHEP) GOTO 150
          IF(K(I1,1).NE.13.AND.K(I1,1).NE.14) GOTO 150
          IF(JDAHEP(1,I1).EQ.0) THEN
            JDAHEP(1,I1)=I
          ELSE
            JDAHEP(2,I1)=I
          ENDIF
  150   CONTINUE
        DO 160 I=1,NHEP
          IF(K(I,1).NE.13.AND.K(I,1).NE.14) GOTO 160
          IF(JDAHEP(2,I).EQ.0) JDAHEP(2,I)=JDAHEP(1,I)
  160   CONTINUE
 
C...Conversion from standard to PYTHIA, the easy part.
      ELSE
        IF(NHEP.GT.MSTU(4)) CALL PYERRM(8,
     &  '(PYHEPC:) no more space in /PYJETS/')
        N=MIN(NHEP,MSTU(4))
        NKQ=0
        KQSUM=0
        DO 190 I=1,N
          K(I,1)=0
          IF(ISTHEP(I).EQ.1) K(I,1)=1
          IF(ISTHEP(I).EQ.2) THEN
             K(I,1)=11
             IF(K(I,4).GT.0.AND.(K(I,4).EQ.K(I,5)).AND.
     $ (K(K(I,4),2).GE.91.AND.K(K(I,4),2).LE.93).AND.
     $ (I.LT.N).AND.(K(I,4).EQ.K(I+1,4))) K(I,1)=12
          ENDIF
          IF(ISTHEP(I).EQ.3) K(I,1)=21
          K(I,2)=IDHEP(I)
          K(I,3)=JMOHEP(1,I)
          K(I,4)=JDAHEP(1,I)
          K(I,5)=JDAHEP(2,I)
          DO 170 J=1,5
            P(I,J)=PHEP(J,I)
  170     CONTINUE
          DO 180 J=1,4
            V(I,J)=VHEP(J,I)
  180     CONTINUE
          V(I,5)=0D0
          IF(ISTHEP(I).EQ.2.AND.PHEP(4,I).GT.PHEP(5,I)) THEN
            I1=JDAHEP(1,I)
            IF(I1.GT.0.AND.I1.LE.NHEP) V(I,5)=(VHEP(4,I1)-VHEP(4,I))*
     &      PHEP(5,I)/PHEP(4,I)
          ENDIF
 
C...Fill in missing information on colour connection in jet systems.
          IF(ISTHEP(I).EQ.1) THEN
            KC=PYCOMP(K(I,2))
            KQ=0
            IF(KC.NE.0) KQ=KCHG(KC,2)*ISIGN(1,K(I,2))
            IF(KQ.NE.0) NKQ=NKQ+1
            IF(KQ.NE.2) KQSUM=KQSUM+KQ
            IF(KQ.NE.0.AND.KQSUM.NE.0) THEN
              K(I,1)=2
            ELSEIF(KQ.EQ.2.AND.I.LT.N) THEN
              IF(K(I+1,2).EQ.21) K(I,1)=2
            ENDIF
          ENDIF
  190   CONTINUE
        IF(NKQ.EQ.1.OR.KQSUM.NE.0) CALL PYERRM(8,
     &  '(PYHEPC:) input parton configuration not colour singlet')
      ENDIF
 
      END
