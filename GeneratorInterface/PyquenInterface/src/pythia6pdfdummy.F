 
C*********************************************************************
 
C...PDFSET
C...Dummy routine, to be removed when PDFLIB is to be linked.
 
      SUBROUTINE PDFSET(PARM,VALUE)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
C...Local arrays and character variables.
      CHARACTER*20 PARM(20)
      DOUBLE PRECISION VALUE(20)
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
      PARM(20)=PARM(1)
      VALUE(20)=VALUE(1)
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link PDFLIB correctly.'/
     &1X,'Dummy routine PDFSET in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END
 
C*********************************************************************
 
C...STRUCTM
C...Dummy routine, to be removed when PDFLIB is to be linked.
 
      SUBROUTINE STRUCTM(XX,QQ,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
C...Local variables
      DOUBLE PRECISION XX,QQ,UPV,DNV,USEA,DSEA,STR,CHM,BOT,TOP,GLU
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
      UPV=XX+QQ
      DNV=XX+2D0*QQ
      USEA=XX+3D0*QQ
      DSEA=XX+4D0*QQ
      STR=XX+5D0*QQ
      CHM=XX+6D0*QQ
      BOT=XX+7D0*QQ
      TOP=XX+8D0*QQ
      GLU=XX+9D0*QQ
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link PDFLIB correctly.'/
     &1X,'Dummy routine STRUCTM in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END
 
C*********************************************************************
 
C...STRUCTP
C...Dummy routine, to be removed when PDFLIB is to be linked.
 
      SUBROUTINE STRUCTP(XX,QQ2,P2,IP2,UPV,DNV,USEA,DSEA,STR,CHM,
     &BOT,TOP,GLU)
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      SAVE /PYDAT1/
C...Local variables
      DOUBLE PRECISION XX,QQ2,P2,UPV,DNV,USEA,DSEA,STR,CHM,BOT,
     &TOP,GLU
 
C...Stop program if this routine is ever called.
      WRITE(MSTU(11),5000)
      IF(PYR(0).LT.10D0) STOP
      UPV=XX+QQ2
      DNV=XX+2D0*QQ2
      USEA=XX+3D0*QQ2
      DSEA=XX+4D0*QQ2
      STR=XX+5D0*QQ2
      CHM=XX+6D0*QQ2
      BOT=XX+7D0*QQ2
      TOP=XX+8D0*QQ2
      GLU=XX+9D0*QQ2
 
C...Format for error printout.
 5000 FORMAT(1X,'Error: you did not link PDFLIB correctly.'/
     &1X,'Dummy routine STRUCTP in PYTHIA file called instead.'/
     &1X,'Execution stopped!')
 
      RETURN
      END
