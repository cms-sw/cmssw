      SUBROUTINE MYPHOTOS(ID, NEVHEP, NHEP, ISTHEP, IDHEP,
     &JMOHEP, JDAHEP, PHEP, VHEP, NHEPOUT, PHEPOUT)
C.----------------------------------------------------------------------
C.
C.    PHOTOS:  General search routine + _GET + _SET
C.
C.    Purpose:  /HEPEVT/ is not anymore a standard at least 
C.              REAL*8 REAL*4 are in use. PHOTOS_GET and PHOTOS_SET
C.              were to be introduced.
C.              
C.
C.    Input Parameters:   ID see routine PHOTOS_MAKE
C.
C.    Output Parameters:  None
C.
C.    Author(s):  Z. Was                          Created at:  21/07/98
C.                                                Last Update: 21/07/98
C.
C.----------------------------------------------------------------------
      COMMON /PHLUPY/ IPOIN,IPOINM  
      INTEGER         IPOIN,IPOINM
      COMMON /PHNUM/ IEV
      INTEGER        IEV
      INTEGER PHLUN
      COMMON/PHOLUN/PHLUN
      INTEGER NMXHEP
      PARAMETER (NMXHEP=4000)
      INTEGER NEVHEP,NHEP,NHEPOUT,ISTHEP(NMXHEP),IDHEP(NMXHEP)
      INTEGER JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP)
      REAL*8 PHEP(5,NMXHEP),PHEPOUT(5,NMXHEP),VHEP(4,NMXHEP)

      IF (1.GT.IPOINM.AND.1.LT.IPOIN ) THEN
       WRITE(PHLUN,*) 'EVENT NR=',IEV,
     $             'WE ARE TESTING /HEPEVT/ at IPOINT=1 (input)'
       CALL PHODMP
      ENDIF
c      CALL PHOTOS_GET
      CALL MYPHOTOSHELPER(ID, NEVHEP, NHEP, ISTHEP,IDHEP,
     &JMOHEP, JDAHEP, PHEP, VHEP, NHEPOUT, PHEPOUT)
c      CALL PHOTOS_SET
      IF (1.GT.IPOINM.AND.1.LT.IPOIN ) THEN
       WRITE(PHLUN,*) 'EVENT NR=',IEV,
     $             'WE ARE TESTING /HEPEVT/ at IPOINT=1 (output)'
       CALL PHODMP
      ENDIF

      END
