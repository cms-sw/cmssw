C***********************************************************************
C*=======================                                              *
       SUBROUTINE BSDATE
C*=======================                                              *
C*((Purpose))                                                          *
C*    Changethe format of the time stamp.                              *
C*    This program should be modified according to the machine.        *
C*((Author))                                                           *
C*    S.Kawabata  Nov. '91 at KEK                                      *
C*    For HP      Jul. '92 at KEK                                      *
C***********************************************************************
       COMMON /BDATE/ IDATE(3),ITIME(2)
       COMMON /SLATE/ IS(40)
*            IDATE(1) : year        ITIME(1) : hour
*            IDATE(2) : month       ITIME(2) : minute
*            IDATE(3) : day
 
       CALL UXDATE(IY,IM,ID,IHH,IMM)
CERN   call datime(id,it)
CERN   CALL UCOPY(IS(1),IDATE(1),5)
CERN   IDATE(1) = MOD(IDATE(1),1900)
*      IDATE(1) = IY
*Y2K
       IDATE(1) = MOD(IY,100)
       IDATE(2) = IM
       IDATE(3) = ID
       ITIME(1) = IHH
       ITIME(2) = IMM
       RETURN
       END


*CMZ :          24/06/94  10.51.47  by  Unknown
*-- Author :
C
C***********************************************************************
C*=================================                                    *
C* SUBROUTINE BSTIME( TIME, IFLG )                                     *
C*=================================                                    *
C*((Purpose))                                                          *
C*        Interface routine to get used CPU time from FORTRAN          *
C*        Library routine CLOCK etc.                                   *
C*((Input))                                                            *
C*        IFLG  : Flag                                                 *
C*          IFLG = 0 : Initialization of clock routine.                *
C*          IFLG = 1 : Get used CPU time.                              *
C*((Output))                                                           *
C*        TIME  : Used CPU time in second.                             *
C*                                                                     *
C*       Coded by S.Kawabata        Oct. '85                           *
C*                                                                     *
C***********************************************************************
C
      SUBROUTINE BSTIME( TIME, IFLG )
C
      save time_init
C
 
      IF( IFLG .NE. 0 ) THEN
C
C         iutime.c should be compiled.
C
          TIME = uxtime() - time_init
*         CALL TIMEX(TIME)
C
      ELSE
 
          time_init = uxtime()
*         CALL TIMEST(9999999.)
          TIME      = 0.0
 
      ENDIF
C
      RETURN
      END
