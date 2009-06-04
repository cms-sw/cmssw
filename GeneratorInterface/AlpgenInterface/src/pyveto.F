 
C*********************************************************************
 
C...PYVETO
C...Interface to UPVETO, which allows user to veto event generation
C...on the parton level, after parton showers but before multiple
C...interactions, beam remnants and hadronization is added.
 
      SUBROUTINE PYVETO(IVETO)
 
C...All real arithmetic in double precision.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
C...Three Pythia functions return integers, so need declaring.
      INTEGER PYK,PYCHGE,PYCOMP
 
C...PYTHIA commonblocks.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      SAVE /PYJETS/,/PYPARS/,/PYINT1/
C...HEPEVT commonblock.
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
C...Local array.
      DIMENSION IRESO(100)
 
C...Define longitudinal boost from initiator rest frame to cm frame.
      GAMMA=0.5D0*(VINT(141)+VINT(142))/SQRT(VINT(141)*VINT(142))
      GABEZ=0.5D0*(VINT(141)-VINT(142))/SQRT(VINT(141)*VINT(142))
 
C... Reset counters.
      NEVHEP=0
      NHEP=0
      NRESO=0
 
C...First pass: identify final locations of resonances
C...and of their daughters before showering.
      DO 150 I=MINT(84)+3,N
        ISTORE=0
        IMOTH=0
 
C...Skip shower CM frame documentation lines.
        IF(K(I,2).EQ.94) THEN
 
C...  Store a new intermediate product, when mother in documentation.
        ELSEIF(MSTP(128).EQ.0.AND.K(I,3).GT.MINT(83)+6.AND.
     &  K(I,3).LE.MINT(84)) THEN
          ISTORE=1
          NHEP=NHEP+1
          II=NHEP
          NRESO=NRESO+1
          IRESO(NRESO)=I
          IMOTH=MAX(0,K(K(I,3),3)-(MINT(83)+6))
 
C...  Store a new intermediate product, when mother in main section.
        ELSEIF(MSTP(128).EQ.1.AND.K(I-MINT(84)+MINT(83)+4,1).EQ.21.AND.
     &  K(I-MINT(84)+MINT(83)+4,2).EQ.K(I,2)) THEN
          ISTORE=1
          NHEP=NHEP+1
          II=NHEP
          NRESO=NRESO+1
          IRESO(NRESO)=I
          IMOTH=MAX(0,K(I-MINT(84)+MINT(83)+4,3)-(MINT(83)+6))
 
C...Update a product when a new copy of it has been created.
        ELSE
          IHIST=K(I,3)
          IF(K(IHIST,2).EQ.94) IHIST=K(IHIST,3)+(I-1-IHIST)
          IR=0
          DO 100 IRI=1,NRESO
            IF(IHIST.EQ.IRESO(IRI)) IR=IRI
  100     CONTINUE
C...Flavours must match, and exclude gluon and photon emission.
          IF(K(IHIST,2).NE.K(I,2)) IR=0
          IF(IR.GT.0.AND.I.LT.N) THEN
            IF(K(I+1,3).EQ.K(I,3).AND.(K(I+1,2).EQ.21.OR.
     &      K(I+1,2).EQ.22)) IR=0
          ENDIF
          IF(IR.GT.0) THEN
            ISTORE=1
            II=IR
            IRESO(IR)=I
            IMOTH=JMOHEP(1,II)
          ENDIF
        ENDIF
 
        IF(ISTORE.EQ.1) THEN
C...Copy parton info, boosting momenta along z axis to cm frame.
          ISTHEP(II)=2
          IDHEP(II)=K(I,2)
          PHEP(1,II)=P(I,1)
          PHEP(2,II)=P(I,2)
          PHEP(3,II)=GAMMA*P(I,3)+GABEZ*P(I,4)
          PHEP(4,II)=GAMMA*P(I,4)+GABEZ*P(I,3)
          PHEP(5,II)=P(I,5)
C...Store one mother. Rest of history and vertex info zeroed.
          JMOHEP(1,II)=IMOTH
          JMOHEP(2,II)=0
          JDAHEP(1,II)=0
          JDAHEP(2,II)=0
          VHEP(1,II)=0D0
          VHEP(2,II)=0D0
          VHEP(3,II)=0D0
          VHEP(4,II)=0D0
        ENDIF
  150 CONTINUE
 
C...Second pass: identify current set of "final" partons.
      DO 200 I=MINT(84)+3,N
        ISTORE=0
        IMOTH=0
 
C...Store a final parton.
        IF(K(I,1).GE.1.AND.K(I,1).LE.10) THEN
          ISTORE=1
          NHEP=NHEP+1
          II=NHEP
C..Trace it back through shower, to check if from documented particle.
          IHIST=I
          ISAVE=IHIST
  160     CONTINUE
          IF(IHIST.GT.MINT(84)) THEN
            IF(K(IHIST,2).EQ.94) IHIST=K(IHIST,3)+(ISAVE-1-IHIST)
            DO 170 IRI=1,NRESO
              IF(IHIST.EQ.IRESO(IRI)) IMOTH=IRI
  170       CONTINUE
            ISAVE=IHIST
            IHIST=K(IHIST,3)
            IF(IMOTH.EQ.0) GOTO 160
          ENDIF
        ENDIF
 
        IF(ISTORE.EQ.1) THEN
C...Copy parton info, boosting momenta along z axis to cm frame.
          ISTHEP(II)=1
          IDHEP(II)=K(I,2)
          PHEP(1,II)=P(I,1)
          PHEP(2,II)=P(I,2)
          PHEP(3,II)=GAMMA*P(I,3)+GABEZ*P(I,4)
          PHEP(4,II)=GAMMA*P(I,4)+GABEZ*P(I,3)
          PHEP(5,II)=P(I,5)
C...Store one mother. Rest of history and vertex info zeroed.
          JMOHEP(1,II)=IMOTH
          JMOHEP(2,II)=0
          JDAHEP(1,II)=0
          JDAHEP(2,II)=0
          VHEP(1,II)=0D0
          VHEP(2,II)=0D0
          VHEP(3,II)=0D0
          VHEP(4,II)=0D0
        ENDIF
  200 CONTINUE
 
C...Call user-written routine to decide whether to keep events.
      CALL UPVETO(IVETO)
 
      RETURN
      END
