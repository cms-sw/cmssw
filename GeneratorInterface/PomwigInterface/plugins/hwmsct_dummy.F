CCC   ADDED BY FABIAN STOECKLI
CCC   CAUSE C++ DOESN'T LIKE LOGICAL !!!!


      DOUBLE PRECISION FUNCTION HWMSCT_DUMMY(KERNTRR)

      DOUBLE PRECISION ABORE, KERNTRR
      LOGICAL DOWORK
      EXTERNAL HWMSCT
CCC   Avoid a dummy argument warning. Only needed on gcc 4.3.4
      KERNTRR=KERNTRR
      CALL HWMSCT(DOWORK)

      ABORE=0.0d0

      IF(DOWORK) THEN
         ABORE = 1.0d0
      ELSE 
         ABORE = 0.0d0
      ENDIF
      

C      write(*,*) ABORE, DOWORK

      HWMSCT_DUMMY = ABORE
      
      RETURN
      END
