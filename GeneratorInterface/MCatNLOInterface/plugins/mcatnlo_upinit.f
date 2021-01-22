
C----------------------------------------------------------------------
      SUBROUTINE MCATNLOUPINIT(IPROC, QQIN)
C----------------------------------------------------------------------
C  Reads MC@NLO input headers and fills Les Houches run common HEPRUP
C----------------------------------------------------------------------

C----------------------------------------------------------------------
CCC MOFIFICATIONS FOR CMSSW, FST
C      INCLUDE 'HERWIG65.INC'
C--Les Houches Common Blocks

C to make sure we don't have uninitialized variables      
      IMPLICIT NONE
      INTEGER IPROC, IERROR
      DOUBLE PRECISION PBEAM1, PBEAM2
      CHARACTER*8 PART1,PART2
      DOUBLE PRECISION ZERO,ONE,TWO,THREE,FOUR,HALF
      PARAMETER (ZERO =0.D0, ONE =1.D0, TWO =2.D0,
     &           THREE=3.D0, FOUR=4.D0, HALF=0.5D0)

C***** TO TAKE CARE OF
      DOUBLE PRECISION RMASS(1000)
      DOUBLE PRECISION EMMIN, EMMAX, GAMMAX, GAMW, GAMZ
      INTEGER EMMINS,EMMAXS,GAMMAXS, GAMWS, GAMZS
      INTEGER RMASSS(1000)
      COMMON/MCPARS/ EMMIN, EMMAX, GAMMAX, RMASS, GAMW, GAMZ,
     &     EMMINS, EMMAXS, GAMMAXS, RMASSS, GAMWS, GAMZS
C----------------------------------------------------------------------

      INTEGER MAXPUP
      PARAMETER(MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON /HEPRUP/ IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &                IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),
     &                XMAXUP(MAXPUP),LPRUP(MAXPUP)
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &              ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &              SPINUP(MAXNUP)
      DOUBLE PRECISION XCKECM,XCKPB1,XTMP1,XTMP2,XTMP3,XTMP4,XMT,XMW,
     & XMZ,XMH,XMV,XM1,XM2,XM3,XM4,XM5,XM21,XLAM,GAH,GAT,GAW,TINY
      DOUBLE PRECISION XMV1,GAV1,GAMAX1,XMV2,GAV2,GAMAX2

      DOUBLE PRECISION GAMZ0, GAMW0

      INTEGER IVVCODE,IFAIL,MQQ,NQQ,IHW,I,NDNS1,NDNS2,JPR,JPR0,IH,IHQ,
     & IVHVEC,IVHLEP,IVLEP1,IVLEP2
      CHARACTER*60 TMPSTR,TMPSTR2
      CHARACTER*4 STRP1,STRP2
      CHARACTER*8 STRGRP1,STRGRP2
      CHARACTER*2 STRSCH
      CHARACTER*3 STRFMT
      CHARACTER*(*) QQIN
C *** FST: NOT NEEDED WITHIN CMSSW
C      LOGICAL FK88STRNOEQ,OLDFORM
      LOGICAL OLDFORM
      DATA TINY/1.D-3/
      COMMON/NQQCOM/MQQ,NQQ
      COMMON/VHLIN/IVHVEC,IVHLEP
      COMMON/VVLIN/IVLEP1,IVLEP2
C


CC SET 'TOUCHED' FLAGS TO ZERO (CMSSW)
      EMMINS=0
      EMMAXS=0
      GAMMAXS=0
      GAMWS=0
      GAMZS=0
c... initialize
      NDNS1=0
      NDNS2=0
      DO I=1,1000
         RMASSS(I)=0
      ENDDO
      

CC SET UESLESS IERROR TO ZERO (CMSSW)
      IERROR=0

      IF (IERROR.NE.0) RETURN
      OLDFORM=.FALSE.
C--SET UP INPUT FILES
      OPEN(UNIT=61,FILE=QQIN,STATUS='UNKNOWN')
C--READ HEADERS OF EVENT FILE
      READ(61,801)XCKECM,XTMP1,XTMP2,XTMP3,XTMP4,TMPSTR

C------ FST-HACK: ONLY ALLOW PBEAM1=PBEAM2
      PBEAM1=XCKECM/2D0
      PBEAM2=XCKECM/2D0

      READ(61,802)IVVCODE,TMPSTR
      IVVCODE=MOD(IVVCODE,10000)
C---CHECK PROCESS CODE
      JPR0=MOD(ABS(IPROC),10000)
      JPR=JPR0/100
      IF (JPR.NE.IVVCODE/100) CALL HWWARN('UPINIT',500)
      IF ((JPR.EQ.17.OR.JPR.EQ.28.OR.JPR.EQ.36).AND.
     & IVVCODE.NE.MOD(ABS(IPROC),10000)) CALL HWWARN('UPINIT',501)

      IF (JPR.EQ.13.OR.JPR.EQ.14) THEN

C----------------------------------------------------------------
C---- ADDED FROM mcatnlo_hwdriver.f (MC@NLO 3.4), Fabian Stoeckli

         IF(JPR0.EQ.1396)THEN
C          WRITE(*,*)'Enter M_GAMMA*(INF), M_GAMMA*(SUP)'
           READ(61,808)EMMIN,EMMAX,TMPSTR
           EMMINS=1
           EMMAXS=1

         ELSEIF(JPR0.EQ.1397)THEN
C           WRITE(*,*)'Enter Z0 mass, width, and GammaX'
           READ(61,809)RMASS(200),GAMZ0,GAMMAX, TMPSTR
           IF(GAMZ0.NE.0.D0) THEN
              GAMZ=GAMZ0
              GAMZS=1
           ENDIF
           RMASSS(200)=1
           GAMMAXS=1

         ELSEIF(JPR0.EQ.1497.OR.JPR0.EQ.1498)THEN
           WRITE(*,*)'Enter W mass, width, and GammaX'
           READ(61,809)RMASS(198),GAMW0,GAMMAX, TMPSTR
           RMASS(199)=RMASS(198)

           RMASSS(199)=1
           RMASSS(198)=1

           IF(GAMW0.NE.0.D0) THEN
              GAMW=GAMW0
              GAMWS=1
           ENDIF

         ELSEIF( (JPR0.GE.1350.AND.JPR0.LE.1356) .OR.
     #           (JPR0.GE.1361.AND.JPR0.LE.1366) )THEN
C           WRITE(*,*)'Enter Z0 mass, width'
            
            READ(61,809) XMV ,GAMZ0, GAMMAX, TMPSTR
            
            RMASS(200)=XMV
            RMASSS(200)=1
            GAMZ=GAMZ0
            GAMZS=1
            GAMMAXS=1

C           WRITE(*,*)'Enter GAMMAX, M_Z*(INF), M_Z*(SUP)'
C           READ(*,*)GAMMAX,EMMIN,EMMAX

C           EMMINS=1
C           EMMAXS=1

            IF(IPROC.GT.0) THEN
               
               WRITE(*,*) 'IMPOSSIBLE ERROR'
               STOP
               
               EMMIN=RMASS(200)-GAMZ*GAMMAX
               EMMAX=RMASS(200)+GAMZ*GAMMAX
            ENDIF
            
         ELSEIF(JPR0.GE.1371.AND.JPR0.LE.1373)THEN
C     WRITE(*,*)'Enter M_LL(INF), M_LL(SUP)'
           READ(61,808)EMMIN,EMMAX,TMPSTR
           
           EMMINS=1
           EMMAXS=1
           
        ELSEIF( (JPR0.GE.1450.AND.JPR0.LE.1453) .OR.
     #          (JPR0.GE.1461.AND.JPR0.LE.1463) .OR.
     #          (JPR0.GE.1471.AND.JPR0.LE.1473) )THEN
C     WRITE(*,*)'Enter W mass, width'
           READ(61,809) RMASS(198),GAMW, GAMMAX, TMPSTR
           RMASS(199)=RMASS(198)
           
           RMASSS(199)=1
           RMASSS(198)=1
           GAMWS=1
           GAMMAXS=1
           
C     WRITE(*,*)'Enter GAMMAX, M_W*(INF), M_W*(SUP)'
C     READ(*,*)GAMMAX,EMMIN,EMMAX
           
C     EMMINS=1
C     EMMAXS=1
           
        ENDIF
        
      ELSEIF (JPR.EQ.28) THEN
C     WRITE(*,*)'Enter W mass, width'
C     READ(*,*)RMASS(198),GAMW0
C     RMASS(199)=RMASS(198)
C     WRITE(*,*)'Enter Z mass, width'
C     READ(*,*)RMASS(200),GAMZ0
C     WRITE(*,*)'Enter VGAMMAX, V2GAMMAX'
C     READ(*,*)VGAMMAX,V2GAMMAX
C     IF(GAMW0.NE.0.D0)GAMW=GAMW0
C     IF(GAMZ0.NE.0.D0)GAMZ=GAMZ0
         
         
         READ(61,808)XMW,XMZ,TMPSTR
C--   CHECK VECTOR BOSON MASSES
         IF(ABS(XMW-RMASS(198)).GT.TINY .OR.
     #        ABS(XMZ-RMASS(200)).GT.TINY) CALL HWWARN('UPINIT',502)
         
         RMASS(198)=XMW
         RMASS(199)=XMW
         RMASS(200)=XMZ
         RMASSS(198)=1
         RMASSS(199)=1
         RMASSS(200)=1
         
         READ(61,810)IVLEP1,IVLEP2,TMPSTR
         READ(61,809)XMV1,GAV1,GAMAX1,TMPSTR
         READ(61,809)XMV2,GAV2,GAMAX2,TMPSTR         
         IF(GAV1.NE.0d0) THEN
            IF(JPR0.EQ.2860) THEN
               GAMZ=GAV1
               GAMZS=1
            ELSE
               GAMW=GAV1
               GAMWS=1
            ENDIF
         ENDIF
         IF(GAV2.NE.0d0) THEN
            IF(JPR0.EQ.2850) THEN
               GAMW=GAV2
               GAMWS=1
            ELSE
               GAMZ=GAV2
               GAMZS=1
            ENDIF
         ENDIF
         GAMMAX=MAX(GAMAX1,GAMAX2)         
         GAMMAXS=1
         
C     ELSEIF (JPR.EQ.16) THEN
C     WRITE(*,*)'Enter Higgs boson and top masses'
C     READ(*,*)RMASS(201),RMASS(6)
         
      ELSEIF (JPR.EQ.16.OR.JPR.EQ.36) THEN
         READ(61,809)XMH,GAH,XMT,TMPSTR
C--   CHECK HIGGS AND TOP MASSES
         IH=201
         IF (JPR.EQ.36) IH=IVVCODE/10-158
         IF(ABS(XMH-RMASS(IH)).GT.TINY) CALL HWWARN('UPINIT',503)
         IF(ABS(XMT-RMASS(6)) .GT.TINY) CALL HWWARN('UPINIT',504)
         
         RMASS(IH)=XMH
         RMASSS(IH)=1
         RMASS(6)=XMT
         RMASSS(6)=1
         
C      ELSEIF (JPR.EQ.17) THEN
C         IF(ABS(IPROC).EQ.1705.OR.ABS(IPROC).EQ.11705)THEN
C           WRITE(*,*)'Enter bottom mass'
C           READ(*,*)RMASS(5)
C         ELSEIF(ABS(IPROC).EQ.1706.OR.ABS(IPROC).EQ.11706)THEN
C           WRITE(*,*)'Enter top mass, W mass'
C           READ(*,'(A)') TMPSTR2
C           READ(TMPSTR2,*,ERR=616)RMASS(6),RMASS(198)
C           RMASS(199)=RMASS(198)
C           GOTO 617
C 616       OLDFORM=.TRUE.
C           READ(TMPSTR2,*) RMASS(6)
C 617       CONTINUE
C     ENDIF
         
         
      ELSEIF (JPR.EQ.17.OR.JPR.EQ.51) THEN
         IHQ=MOD(JPR0,10)
         IF (IHQ.EQ.6) THEN
            READ(61,'(A)') TMPSTR2
            IF (TMPSTR2(17:19).EQ.'M_Q') THEN
               OLDFORM=.TRUE.
               READ(TMPSTR2,803)XMT,TMPSTR
            ELSE
               READ(TMPSTR2,808)XMT,GAT,TMPSTR
            ENDIF
         ELSE
            READ(61,803)XMT,TMPSTR
         ENDIF
C--   CHECK HEAVY QUARK MASS
         IF(ABS(XMT-RMASS(IHQ)).GT.TINY) CALL HWWARN('UPINIT',505)
         
         RMASS(IHQ)=XMT
         RMASSS(IHQ)=1
         
         IF (IHQ.EQ.6) THEN
            IF(.NOT.OLDFORM)THEN
               READ(61,808)XMW,GAW,TMPSTR
               READ(61,810)IVLEP1,IVLEP2,TMPSTR
               
               RMASS(198)=XMW
               RMASS(199)=XMW
               RMASSS(198)=1
               RMASSS(199)=1
               
C--   CHECK W BOSON MASS WHEN TOPS DECAY
               IF( IVLEP1.NE.7.AND.IVLEP2.NE.7 .AND.
     #              ABS(XMW-RMASS(198)).GT.TINY ) 
     #              CALL HWWARN('UPINIT',502)
               IF( IVLEP1.NE.7.AND.IVLEP2.NE.7 ) THEN
                  RMASS(198)=XMW
                  RMASSS(198)=1
               ENDIF
            ELSE
               XMW=0.D0
               GAW=0.D0
               IVLEP1=7
               IVLEP2=7
            ENDIF
         ENDIF
         

C      ELSEIF (JPR.EQ.26) THEN
C     WRITE(*,*)'Enter W mass, width'
C         READ(*,*)RMASS(198),GAMW0
C     RMASS(199)=RMASS(198)
C     WRITE(*,*)'Enter Higgs boson mass'
C         READ(*,*)RMASS(201)
C     IF(GAMW0.NE.0.D0)GAMW=GAMW0
C      ELSEIF (JPR.EQ.27) THEN
C         WRITE(*,*)'Enter Z mass, width'
C         READ(*,*)RMASS(200),GAMZ0
C         WRITE(*,*)'Enter Higgs boson mass'
C         READ(*,*)RMASS(201)
C         IF(GAMZ0.NE.0.D0)GAMZ=GAMZ0


      ELSEIF (JPR.EQ.26.OR.JPR.EQ.27.OR.JPR.EQ.19) THEN
         READ(61,810)IVHVEC,IVHLEP,TMPSTR
         READ(61,809)XMV,GAW,GAMMAX,TMPSTR
         READ(61,809)XMH,GAH,GAMMAX,TMPSTR
         GAMMAXS=1
         IF( (JPR.EQ.26.AND.ABS(XMV-RMASS(199)).GT.TINY) .OR.
     #        (JPR.EQ.27.AND.ABS(XMV-RMASS(200)).GT.TINY) )
     #        CALL HWWARN('UPINIT',508)
         
C--------CMSSW: SET MASSES
         SELECT CASE (JPR)
         CASE (27)
            RMASS(200)=XMV
            RMASSS(200)=1
            IF(GAW.NE.0D0) THEN
               GAMZ=GAW
               GAMZS=1
            ENDIF
         CASE (26)
            RMASS(198)=XMV
            RMASSS(198)=1
            RMASS(199)=XMV
            RMASSS(199)=1
            IF(GAW.NE.0D0) THEN
               GAMW=GAW
               GAMWS=1
            ENDIF
         END SELECT
         
         IF(ABS(XMH-RMASS(201)).GT.TINY) CALL HWWARN('UPINIT',509)
         
         RMASS(201)=XMH
         RMASSS(201)=1
         

C      ELSEIF (JPR.EQ.20) THEN
C         WRITE(*,*)'Enter top mass, W mass'
C         READ(*,'(A)') TMPSTR2
C         READ(TMPSTR2,*,ERR=618)RMASS(6),RMASS(198)
C         RMASS(199)=RMASS(198)
C         GOTO 619
C 618     OLDFORM=.TRUE.
C         READ(TMPSTR2,*) RMASS(6)
C 619     CONTINUE

      ELSEIF (JPR.EQ.20) THEN
         READ(61,'(A)') TMPSTR2
         IF (TMPSTR2(28:43).EQ.'M_top, Gamma_top') THEN
            READ(TMPSTR2,808)XMT,GAT,TMPSTR
         ELSE
            OLDFORM=.TRUE.
            READ(TMPSTR2,803)XMT,TMPSTR
         ENDIF
C-- CHECK TOP QUARK MASS
         IF(ABS(XMT-RMASS(6)).GT.TINY) CALL HWWARN('UPINIT',511)
         RMASS(6)=XMT
         RMASSS(6)=1
         IF(OLDFORM)GOTO 444
         READ(61,808)XMW,GAW,TMPSTR

         RMASS(198)=XMW
         RMASSS(198)=1
         RMASS(199)=XMW
         RMASSS(199)=1

         IF(JPR0.LT.2030)THEN
            READ(61,812)IVLEP1,TMPSTR
C--   CHECK W BOSON MASS WHEN TOPS DECAY
            IF( IVLEP1.NE.7 .AND.
     #        ABS(XMW-RMASS(198)).GT.TINY ) CALL HWWARN('UPINIT',502)
         ELSE
            READ(61,810)IVLEP1,IVLEP2,TMPSTR
C--   CHECK W BOSON MASS
            IF(ABS(XMW-RMASS(198)).GT.TINY) CALL HWWARN('UPINIT',502)
         ENDIF
 444     CONTINUE
      ELSEIF (JPR.NE.1) THEN
         CALL HWWARN('UPINIT',506)
      ENDIF
      
      READ(61,804)XM1,XM2,XM3,XM4,XM5,XM21,TMPSTR
      IF (JPR.NE.1) THEN
         READ(61,805)STRP1,STRP2,TMPSTR
         READ(61,806)STRGRP1,NDNS1,TMPSTR
         IF (JPR.EQ.51) THEN
            READ(61,806)STRGRP2,NDNS2,TMPSTR
            READ(61,803)XCKPB1,TMPSTR
            IF(ABS(XCKPB1-PBEAM1).GT.TINY) CALL HWWARN('UPINIT',512)
         ELSE
            STRGRP2=STRGRP1
            NDNS2=NDNS1
         ENDIF
      ENDIF
      READ(61,807)XLAM,STRSCH,TMPSTR
C--CHECK THAT EVENT FILE HAS BEEN GENERATED CONSISTENTLY WITH 
C--HERWIG PARAMETERS ADOPTED HERE
      IFAIL=0
C-- CM ENERGY
      IF( ABS(XCKECM-2D0*SQRT(PBEAM1*PBEAM2)).GT.TINY .OR.
C--   QUARK AND GLUON MASSES
     #     ABS(XM1-RMASS(1)).GT.TINY .OR.
     #     ABS(XM2-RMASS(2)).GT.TINY .OR.
     #     ABS(XM3-RMASS(3)).GT.TINY .OR.
     #     ABS(XM4-RMASS(4)).GT.TINY .OR.
     #     ABS(XM5-RMASS(5)).GT.TINY .OR.
     #     ABS(XM21-RMASS(13)).GT.TINY) IFAIL=1

      RMASS(1)=XM1
      RMASSS(1)=1
      RMASS(2)=XM2
      RMASSS(2)=1
      RMASS(3)=XM3
      RMASSS(3)=1
      RMASS(4)=XM4
      RMASSS(4)=1
      RMASS(5)=XM5
      RMASSS(5)=1
      RMASS(13)=XM21
      RMASSS(13)=1

      DO I=1,5
         RMASS(I+6)=RMASS(I)
         RMASSS(I+6)=1
      ENDDO

C-- LAMBDA_QCD: NOW REMOVED TO ALLOW MORE FLEXIBILITY (NNLO EFFECT ANYHOW)
C     #     ABS(XLAM-QCDLAM).GT.TINY .OR.
C-- REPLACE THE FOLLOWING WITH A CONDITION ON STRSCH, IF CONSISTENT 
C-- INFORMATION ON PDF SCHEME WILL BE AVAILABLE FROM PDF LIBRARIES AND HERWIG
C-- COLLIDING PARTICLE TYPE


CC *** FST: SET PATRICLES FROM FILE

      PART1=STRP1
      PART2=STRP2

CC *** AND SET THE PFGSUPs
      PDFSUP(1)=NDNS1
      PDFSUP(2)=NDNS2

CC-------- MODIFED FOR CMSSW: WE SET PDF INFO OURSELVES (20.3.2008, FST)------------



C      IF (JPR.NE.1.AND.IFAIL.EQ.0) THEN
C         IF(
C     #        FK88STRNOEQ(STRP1,PART1) .OR.
C     #        FK88STRNOEQ(STRP2,PART2) )IFAIL=1
C--IF PDF LIBRARY IS USED, CHECK PDF CONSISTENCY
C         IF( IFAIL.EQ.0 .AND. MODPDF(1).NE.-1)THEN
C            IF( 
C     #          FK88STRNOEQ(STRGRP1,AUTPDF(1)) .OR.
C     #          FK88STRNOEQ(STRGRP2,AUTPDF(2)) .OR.
C     #          ABS(NDNS1-MODPDF(1)).GT.TINY .OR.
C     #          ABS(NDNS2-MODPDF(2)).GT.TINY )IFAIL=1
C--WHEN LHAPDF IS LINKED, AUTPDF() IS A MC@NLO-DEFINED STRING
C            IF(AUTPDF(1).EQ.'LHAPDF'.OR.AUTPDF(1).EQ.'LHAEXT')THEN
C               AUTPDF(1)='DEFAULT'
C               AUTPDF(2)='DEFAULT'
C     ENDIF
C      ENDIF
C      ENDIF

      IF(IFAIL.EQ.1) CALL HWWARN('UPINIT',507)
      CALL HWUIDT(3,IDBMUP(1),IHW,PART1)
      CALL HWUIDT(3,IDBMUP(2),IHW,PART2)
      EBMUP(1)=PBEAM1
      EBMUP(2)=PBEAM2
      DO I=1,2
         PDFGUP(I)=-1
C         PDFSUP(I)=-1
      ENDDO
      IDWTUP=-4
      NPRUP=1
      LPRUP(1)=IVVCODE
C-- TEST FOR NEW FORMAT INPUT MOMENTA: (PX,PY,PZ,M)
      READ(61,811) STRFMT,TMPSTR
      IF (STRFMT.NE.'P,M') CALL HWWARN('UPINIT',510)
      READ(61,900) MQQ
      NQQ=0
C-- LARGEST EXPECTED NUMBER OF LEGS
      NUP=10
      AQEDUP=ZERO
      AQCDUP=ZERO
      DO I=1,NUP
         VTIMUP(I)=ZERO
         SPINUP(I)=9.
      ENDDO
 801  FORMAT(5(1X,D10.4),1X,A)
 802  FORMAT(1X,I6,1X,A)
 803  FORMAT(1X,D10.4,1X,A)
 804  FORMAT(6(1X,D10.4),1X,A)
 805  FORMAT(2(1X,A4),1X,A)
 806  FORMAT(1X,A8,1X,I6,1X,A)
 807  FORMAT(1X,D10.4,1X,A2,1X,A)
 808  FORMAT(2(1X,D10.4),1X,A)
 809  FORMAT(3(1X,D10.4),1X,A)
 810  FORMAT(2(1X,I2),1X,A)
 811  FORMAT(1X,A3,1X,A)
 812  FORMAT(1X,I2,1X,A)
 900  FORMAT(I9)
      END
