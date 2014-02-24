      SUBROUTINE MYPHOTOSHELPER(IPARR, MNEVHEP, MNHEP, MISTHEP, MIDHEP,
     &MJMOHEP, MJDAHEP, MPHEP, MVHEP, NHEPOUT, PHEPOUT)
C.----------------------------------------------------------------------
C.
C.    PHOTOS_MAKE:   General search routine
C.
C.    Purpose:  Search through the /PH_HEPEVT/ standard HEP common, sta-
C.              rting from  the IPPAR-th  particle.  Whenevr  branching 
C.              point is found routine PHTYPE(IP) is called.
C.              Finally if calls on PHTYPE(IP) modified entries, common
C               /PH_HEPEVT/ is ordered.
C.
C.    Input Parameter:    IPPAR:  Pointer   to   decaying  particle  in
C.                                /PH_HEPEVT/ and the common itself,
C.
C.    Output Parameters:  Common  /PH_HEPEVT/, either with or without 
C.                                new particles added.
C.
C.    Author(s):  Z. Was, B. van Eijk             Created at:  26/11/89
C.                                                Last Update: 30/08/93
C.
C.----------------------------------------------------------------------
      IMPLICIT NONE
      REAL*8 PHOTON(5)
      INTEGER IP,IPARR,IPPAR,I,J,K,L,NLAST
      DOUBLE PRECISION DATA
      INTEGER MOTHER,POSPHO
      LOGICAL CASCAD
#include "ph_hepevt.inc"
c      INTEGER NMXHEP
c      PARAMETER (NMXHEP=4000)
c      INTEGER NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
c      INTEGER JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP)
c      REAL*8 PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      INTEGER MNEVHEP,MNHEP,NHEPOUT,MISTHEP(NMXHEP),MIDHEP(NMXHEP)
      INTEGER MJMOHEP(2,NMXHEP),MJDAHEP(2,NMXHEP)
      REAL*8 MPHEP(5,NMXHEP),PHEPOUT(5,NMXHEP),MVHEP(4,NMXHEP)
      LOGICAL QEDRAD
      COMMON/PHOQED/QEDRAD(NMXHEP)
      INTEGER NMXPHO
      PARAMETER (NMXPHO=10000)
      INTEGER ISTACK(0:NMXPHO),NUMIT,NTRY,KK,LL,II,NA,FIRST,LAST
      INTEGER FIRSTA,LASTA,IPP,IDA1,IDA2,MOTHER2,IDPHO,ISPHO
      REAL*8 PORIG(5,NMXPHO)
C--
      IPPAR=ABS(IPARR)
C--   Store pointers for cascade treatement...
      IP=IPPAR
      NLAST=NHEP
      CASCAD=.FALSE.

C--   Fill the HEPEVT common block
      NEVHEP = MNEVHEP
      NHEP = MNHEP
      DO I=1,NHEP      
        ISTHEP(I) = MISTHEP(I)
        IDHEP(I) = MIDHEP(I)
        DO J = 1,2
          JMOHEP(J,I) = MJMOHEP(J,I)
          JDAHEP(J,I) = MJDAHEP(J,I)
        ENDDO
        DO K = 1,5
          PHEP(K,I) = MPHEP(K,I)
        ENDDO
        DO KK = 1,4
          VHEP(KK,I) = MVHEP(KK,I)
        ENDDO
      ENDDO

C--   Check decay multiplicity and minimum of correctness..
      IF ((JDAHEP(1,IP).EQ.0).OR.(JMOHEP(1,JDAHEP(1,IP)).NE.IP)) RETURN
C--
C-- single branch mode 
C-- we start looking for the decay points in the cascade 
C-- IPPAR is original position where the program was called
      ISTACK(0)=IPPAR
C--   NUMIT denotes number of secondary decay branches
      NUMIT=0
C--   NTRY denotes number of secondary branches already checked for 
C--        for existence of further branches 
      NTRY=0
C-- let-s search if IPARR does not prevent searching. 
      IF (IPARR.GT.0)  THEN
 30    CONTINUE
         DO I=JDAHEP(1,IP),JDAHEP(2,IP)
          IF (JDAHEP(1,I).NE.0.AND.JMOHEP(1,JDAHEP(1,I)).EQ.I) THEN
            NUMIT=NUMIT+1
              IF (NUMIT.GT.NMXPHO) THEN
               DATA=NUMIT
               CALL PHOERR(7,'PHOTOS',DATA)
              ENDIF
            ISTACK(NUMIT)=I
          ENDIF
         ENDDO
      IF(NUMIT.GT.NTRY) THEN
       NTRY=NTRY+1
       IP=ISTACK(NTRY)
       GOTO 30
      ENDIF
      ENDIF
C-- let-s do generation
      DO 25 KK=0,NUMIT
        NA=NHEP
        FIRST=JDAHEP(1,ISTACK(KK))
        LAST=JDAHEP(2,ISTACK(KK))
        DO II=1,LAST-FIRST+1
         DO LL=1,5
          PORIG(LL,II)=PHEP(LL,FIRST+II-1) 
         ENDDO
        ENDDO
C--
        CALL PHTYPE(ISTACK(KK))
C--
C--  Correct energy/momentum of cascade daughters
        IF(NHEP.GT.NA) THEN 
        DO II=1,LAST-FIRST+1
          IPP=FIRST+II-1
          FIRSTA=JDAHEP(1,IPP)
          LASTA=JDAHEP(2,IPP)
          IF(JMOHEP(1,IPP).EQ.ISTACK(KK))
     $      CALL PHOBOS(IPP,PORIG(1,II),PHEP(1,IPP),FIRSTA,LASTA) 
        ENDDO
        ENDIF
 25   CONTINUE
C--
C--   rearrange  /PH_HEPEVT/  to get correct order..
        IF (NHEP.GT.NLAST) THEN
          DO 160 I=NLAST+1,NHEP
C--
C--   Photon mother and position...
            MOTHER=JMOHEP(1,I)
            POSPHO=JDAHEP(2,MOTHER)+1
C--   Intermediate save of photon energy/momentum and pointers
              DO 90 J=1,5
   90         PHOTON(J)=PHEP(J,I)
              ISPHO =ISTHEP(I)
              IDPHO =IDHEP(I)
              MOTHER2 =JMOHEP(2,I)
              IDA1 =JDAHEP(1,I)
              IDA2 =JDAHEP(2,I)
C--
C--   Exclude photon in sequence !
            IF (POSPHO.NE.NHEP) THEN
C--
C--
C--   Order /PH_HEPEVT/
              DO 120 K=I,POSPHO+1,-1
                ISTHEP(K)=ISTHEP(K-1)
                QEDRAD(K)=QEDRAD(K-1)
                IDHEP(K)=IDHEP(K-1)
                DO 100 L=1,2
                JMOHEP(L,K)=JMOHEP(L,K-1)
  100           JDAHEP(L,K)=JDAHEP(L,K-1)
                DO 110 L=1,5
  110           PHEP(L,K)=PHEP(L,K-1)
                DO 120 L=1,4
  120         VHEP(L,K)=VHEP(L,K-1)
C--
C--   Correct pointers assuming most dirty /PH_HEPEVT/...
              DO 130 K=1,NHEP
                DO 130 L=1,2
                  IF ((JMOHEP(L,K).NE.0).AND.(JMOHEP(L,K).GE.
     &            POSPHO)) JMOHEP(L,K)=JMOHEP(L,K)+1
                  IF ((JDAHEP(L,K).NE.0).AND.(JDAHEP(L,K).GE.
     &            POSPHO)) JDAHEP(L,K)=JDAHEP(L,K)+1
  130         CONTINUE
C--
C--   Store photon energy/momentum
              DO 140 J=1,5
  140         PHEP(J,POSPHO)=PHOTON(J)
            ENDIF
C--
C--   Store pointers for the photon...
            JDAHEP(2,MOTHER)=POSPHO
            ISTHEP(POSPHO)=ISPHO
            IDHEP(POSPHO)=IDPHO
            JMOHEP(1,POSPHO)=MOTHER
            JMOHEP(2,POSPHO)=MOTHER2
            JDAHEP(1,POSPHO)=IDA1
            JDAHEP(2,POSPHO)=IDA2
C--
C--   Get photon production vertex position
            DO 150 J=1,4
  150       VHEP(J,POSPHO)=VHEP(J,POSPHO-1)
  160     CONTINUE
        ENDIF

      NHEPOUT = NHEP
      DO I=1,NHEPOUT    
        DO K = 1,5
          PHEPOUT(K,I) = PHEP(K,I)
c         print *, PHEPOUT(K,I)
        ENDDO
      ENDDO

      RETURN
      END
