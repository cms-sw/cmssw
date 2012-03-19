      SUBROUTINE LUNREAD(filename)


      IMPLICIT NONE

      CHARACTER*(*) filename

      DOUBLE PRECISION VIPWID,DXRCYL,DXZMAX,DXRSPH
      LOGICAL WZRFR,FIX4JT
      INTEGER IMSSM, IHIGGS,PARITY,LRSUSY
      COMMON/HW6202/VIPWID(3),DXRCYL,DXZMAX,DXRSPH,WZRFR,FIX4JT,
     & IMSSM,IHIGGS,PARITY,LRSUSY

      OPEN(UNIT=LRSUSY,FORM='FORMATTED',STATUS='OLD',ERR=999,
     $     FILE=filename)

      CALL HWISSP !reading in susy/sugra point
      CLOSE(UNIT=LRSUSY) 

      GOTO 1111

  999 WRITE (6,*)
      WRITE (6,*) 'SUSY input file did not open correctly.'
      WRITE (6,*) 'Please check that it is in the right place.'
      WRITE (6,*) 'Examples can be obtained from the isawig web page'
      CALL CMSENDING(299)

 1111 CONTINUE

      END
 
