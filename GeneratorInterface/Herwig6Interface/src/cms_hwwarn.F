CDECK  ID>, HWWARN.
*CMZ :-        -26/04/91  10.18.58  by  Bryan Webber
*-- Author :    Bryan Webber
C
C
C----------------------------------------------------------------------
C   modified by Fabian Stoeckli, March 2007
C     in order to catch run stop for CMSSW
C
C-----------------------------------------------------------------------
      SUBROUTINE HWWARN(SUBRTN,ICODE)
C-----------------------------------------------------------------------
C     DEALS WITH ERRORS DURING EXECUTION
C     SUBRTN = NAME OF CALLING SUBROUTINE
C     ICODE  = ERROR CODE:    - -1 NONFATAL, KILL EVENT & PRINT NOTHING
C                            0- 49 NONFATAL, PRINT WARNING & CONTINUE
C                           50- 99 NONFATAL, PRINT WARNING & JUMP
C                          100-199 NONFATAL, DUMP & KILL EVENT
C                          200-299    FATAL, TERMINATE RUN
C                          300-399    FATAL, DUMP EVENT & TERMINATE RUN
C                          400-499    FATAL, DUMP EVENT & STOP DEAD
C                          500-       FATAL, STOP DEAD WITH NO DUMP
C-----------------------------------------------------------------------
      IMPLICIT NONE
      DOUBLE PRECISION ZERO,ONE,TWO,THREE,FOUR,HALF
      PARAMETER (ZERO =0.D0, ONE =1.D0, TWO =2.D0,
     &           THREE=3.D0, FOUR=4.D0, HALF=0.5D0)

C
      INTEGER NMXHEP,IERROR,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP,IDHW,
     & ISTAT,LWEVT,MAXER,MAXPR,NRN,NUMER,NUMERU,NWGTS

      DOUBLE PRECISION PHEP,VHEP,AVWGT,EVWGT,GAMWT,TLOUT,WBIGST,WGTMAX,
     & WGTSUM,WSQSUM

      LOGICAL NOWGT,GENSOF

      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     & JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)

      COMMON/HWEVNT/AVWGT,EVWGT,GAMWT,TLOUT,WBIGST,WGTMAX,WGTSUM,WSQSUM,
     & IDHW(NMXHEP),IERROR,ISTAT,LWEVT,MAXER,MAXPR,NOWGT,NRN(2),NUMER,
     & NUMERU,NWGTS,GENSOF

      INTEGER ICODE
      CHARACTER*6 SUBRTN

      LOGICAL EXIT
      CALL CMS_HWWARN(SUBRTN,ICODE,EXIT)
      IF (EXIT) RETURN

      IF (ICODE.GE.0) WRITE (6,10) SUBRTN,ICODE
   10 FORMAT(/' HWWARN CALLED FROM SUBPROGRAM ',A6,': CODE =',I4)
      IF (ICODE.LT.0) THEN
         IERROR=ICODE
         RETURN
      ELSEIF (ICODE.LT.100) THEN
         WRITE (6,20) NEVHEP,NRN,EVWGT
   20    FORMAT(' EVENT',I8,':   SEEDS =',I11,' &',I11,
     &'  WEIGHT =',E11.4/' EVENT SURVIVES. EXECUTION CONTINUES')
         IF (ICODE.GT.49) RETURN
      ELSEIF (ICODE.LT.200) THEN
         WRITE (6,30) NEVHEP,NRN,EVWGT
   30    FORMAT(' EVENT',I8,':   SEEDS =',I11,' &',I11,
     &'  WEIGHT =',E11.4/' EVENT KILLED.   EXECUTION CONTINUES')
         IERROR=ICODE
         RETURN
      ELSEIF (ICODE.LT.300) THEN
         WRITE (6,40)
   40    FORMAT(' EVENT SURVIVES.  RUN ENDS GRACEFULLY')
         CALL HWEFIN
         CALL HWAEND
         CALL cmsending(ICODE)
      ELSEIF (ICODE.LT.400) THEN
         WRITE (6,50)
   50    FORMAT(' EVENT KILLED: DUMP FOLLOWS.  RUN ENDS GRACEFULLY')
         IERROR=ICODE
         CALL HWUEPR
         CALL HWUBPR
         CALL HWEFIN
         CALL HWAEND
         CALL cmsending(ICODE)
      ELSEIF (ICODE.LT.500) THEN
         WRITE (6,60)
   60    FORMAT(' EVENT KILLED: DUMP FOLLOWS.  RUN STOPS DEAD')
         IERROR=ICODE
         CALL HWUEPR
         CALL HWUBPR
         CALL cmsending(ICODE)
      ELSE
         WRITE (6,70)
   70    FORMAT(' RUN CANNOT CONTINUE')
         CALL cmsending(ICODE)
      ENDIF
      END

C------------      
C      
C On 4/2/2012 - Modification by J.V.Yarba, FNAL: 
C Makeup for old-stryle fortran tauola routines that are called by Herwig6 core code
C
C------------

      SUBROUTINE INIETC(jakk1,jakk2,itd,ifpho)
      COMMON / IDFC  / IDFF
      COMMON / TAURAD / XK0DEC,ITDKRC
      DOUBLE PRECISION            XK0DEC
      COMMON / JAKI   /  JAK1,JAK2,JAKP,JAKM,KTOM
      COMMON /PHOACT/ IFPHOT
      SAVE
C KTO=1 will denote tau+, thus :: IDFF=-15
          IDFF=-15
C XK0 for tau decays.
          XK0DEC=0.01
C radiative correction switch in tau --> e (mu) decays !
          ITDKRC=itd
C switches of tau+ tau- decay modes !!
          JAK1=jakk1
          JAK2=jakk2
C photos activation switch
ccc --> Note by J.V.Yarba: this is no longer relevant with tauola++
ccc -->                    but we keep it in for historical reasons
          IFPHOT=IFPHO
      end
      
      SUBROUTINE INIPHX(XK00)
* ----------------------------------------------------------------------
*     INITIALISATION OF PARAMETERS
*     USED IN QED and/or GSW ROUTINES
* ----------------------------------------------------------------------
      COMMON / QEDPRM /ALFINV,ALFPI,XK0
      REAL*8           ALFINV,ALFPI,XK0
      REAL*8 PI8,XK00
*
      PI8    = 4.D0*DATAN(1.D0)
      ALFINV = 137.03604D0
      ALFPI  = 1D0/(ALFINV*PI8)
      XK0=XK00
      END
