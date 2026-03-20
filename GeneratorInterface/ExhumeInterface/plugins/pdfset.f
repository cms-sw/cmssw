      SUBROUTINE MY_PDFSET(PDF)
      
      DOUBLE PRECISION PDF
      DOUBLE PRECISION VAL(20)
      CHARACTER*20 PARM(20)

      PARM(1)  = 'DEFAULT             '
      VAL(1)   =  PDF
      CALL PDFSET(PARM,VAL)
      
      RETURN
      END


