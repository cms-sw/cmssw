      SUBROUTINE RNHRAN(Y,N,XLO,XWID,XRAN)
      CHARACTER*6 NAME(2)
 
      DIMENSION Y(*)
 
      DATA IERR /0/
      DATA NAME /'RNHRAN','RNHPRE'/
 
      NTRY=1
      IF(Y(N) .EQ. 1) GOTO 4
      WRITE(6,101) NAME(1),Y(N)
      GOTO 5
 
      ENTRY RNHPRE(Y,N)
      NTRY=2
 
    5 YTOT=0
      DO 1 I = 1,N
      IF(Y(I) .LT. 0) GOTO 9
      YTOT=YTOT+Y(I)
    1 Y(I)=YTOT
      IF(YTOT .LE. 0) GOTO 9
      YINV=1/YTOT
      DO 2 I = 1,N
    2 Y(I)=Y(I)*YINV
      Y(N)=1
      IF(NTRY .EQ. 2) RETURN
 
    4 CALL RANLUX(YR,1)
      L=LOCATR(Y,N,YR)
      IF(L .LT. 0) THEN
       L=-L
       XRAN=XLO+XWID*(L+((YR-Y(L))/(Y(L+1)-Y(L))))
      ELSEIF(L .EQ. 0) THEN
       XRAN=XLO+XWID*(YR/Y(1))
      ELSE
       XRAN=XLO+L*XWID
      ENDIF
      RETURN
 
    9 IERR=IERR+1
      IF(IERR .LT. 6) WRITE(6,102) NAME(NTRY)
      WRITE(6,'(1X,10F13.7)') (Y(K),K=1,N)
      XRAN=0
      RETURN
  101 FORMAT(/7X,'+++++ CERN V149 ',A6,' : Y(N) = ',E15.6,' .NE. 1; ',
     1           'Y(I) NOT IN CUMULATIVE FORM.'/)
  102 FORMAT(/7X,'+++++ CERN V149 ',A6,' : NOT ALL VALUES Y(I) > 0'/)
      END

