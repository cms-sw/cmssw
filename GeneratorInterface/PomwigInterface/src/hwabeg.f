C * 07/03/2003, Tibor Kucs
C * User's routine for initialization
C-----------------------------------------------------------------------
      SUBROUTINE HWABEG
      INCLUDE 'HERWIG65.INC'
      INTEGER IFIT
      DOUBLE PRECISION XPQ(-6:6),XP,Q2

c---Initialization of the QCD_1994 subroutine
      Q2=75D0
      IFIT=5
      XP=1D-1
      CALL QCD_1994(XP,Q2,XPQ,IFIT)


      END

