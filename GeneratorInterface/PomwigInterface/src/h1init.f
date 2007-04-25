      SUBROUTINE H1INIT(IFIT)
      INCLUDE 'HERWIG65.INC'
      INTEGER IFIT
      DOUBLE PRECISION XPQ(-6:6),XP,Q2

c---Initialization of the QCD_1994 subroutine
      Q2=75D0
C      IFIT=5
      XP=1D-1
      CALL QCD_1994(XP,Q2,XPQ,IFIT)


      END

