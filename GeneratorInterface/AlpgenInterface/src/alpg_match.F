*-- AUTHOR :    MICHELANGELO MANGANO
C----------------------------------------------------------------------
      SUBROUTINE UPVETO(IPVETO)
C----------------------------------------------------------------------
C     SUBROUTINE TO IMPLEMENT THE MLM JET MATCHING CRITERION
C----------------------------------------------------------------------
      IMPLICIT NONE
C...GUP EVENT COMMON BLOCK
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &              ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &              SPINUP(MAXNUP)
C...HEPEVT COMMONBLOCK.
      INTEGER NMXHEP,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
c process code
      integer ihrd
      integer itopprc
c--event file data
      integer Nunit,NunitOut,NunitIni
      character*512 filename,tmpstr
c total number of input events
      common/AHio/Nunit,NunitOut,NunitIni,filename
c inputs for the matching algorithm
      integer iexc,npfst,nplst,nljets,njstart,njlast
     $     ,ickkw
      double precision etclus,rclus,etaclmax
      common/AHopts/etclus,rclus,etaclmax,iexc,npfst
     $     ,nplst,nljets,njstart,njlast,ickkw
c process and particles parameters
      double precision mc,mb,mt,mw,mz,mh
      double precision ebeam
      integer ndns,ih1,ih2       
      integer nw,nz,nh,nph
      integer ihvy,ihvy2
c pdf set type
      character pdftyp*25
c total number of partons
      integer npart
      common/AHppara/mc,mb,mt,mw,mz,mh,
     & ebeam,ih1,ih2,
     & ihrd,itopprc,
     & nw,nz,nh,nph,
     & ihvy,ihvy2,
     & npart,ndns,pdftyp
c weight information
      real *8 maxwgt,avgwgt,errwgt,totlum
      integer unwev
      common/AHwgts/maxwgt,avgwgt,errwgt,totlum,unwev
c general parameters
      integer nparam
      parameter (nparam=200)
      integer parlen,partyp
      character chpar*8,chpdes*70
      double precision parval
      common/AHpars/parval(nparam),chpar(nparam),chpdes(nparam)
     $     ,parlen(nparam),partyp(nparam)
c global event cuts
      double precision ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +     ptcmin,ptcmax,etacmax,drcmin,
     +     ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +     ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      common/AHcuts/ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +        ptcmin,ptcmax,etacmax,drcmin,
     +        ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +        ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      INTEGER IPVETO
C     CALSIM AND JET VARIABLES                             
      INTEGER NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,PI,ET,DELPHI,CPHCAL,SPHCAL,DELY,
     &     CTHCAL,STHCAL,PCJET,ETJET
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60,PI=3.141593D0)
      COMMON/CALOR/DELY,DELPHI,ET(NCY,NCPHI),
     $     CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN
     $     ,YCMAX
      PARAMETER (NJMAX=500)
      COMMON/GETCOM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(NCY,NCPHI),NCJET
C     
      DOUBLE PRECISION PSERAP
      INTEGER K(NJMAX),KP(NJMAX),KPJ(NJMAX)
C LOCAL VARIABLES
      INTEGER I,J,IHEP,NMATCH,JRMIN
      DOUBLE PRECISION ETAJET,PHIJET,DELR,DPHI,DELRMIN
      DOUBLE PRECISION P(4,10),PT(10),ETA(10),PHI(10)
C HEAVY QUARK MATCHING
      INTEGER IHVQ(10),NHVQ,NMJET,ID
      DOUBLE PRECISION ETAHVQ(10),PHIHVQ(10)
      INTEGER IEND,INORAD
      COMMON/SHVETO/IEND,INORAD(MAXNUP)
      
C DEBUGGING OPTIONS
      INTEGER IDBG
      PARAMETER (IDBG=0)
      DOUBLE PRECISION PTPART,PTJETS,ETAPART,ETAJETS
      INTEGER NMAX
      COMMON/MTCHDBG/PTPART(10),PTJETS(10),ETAPART(10),ETAJETS(10),NMAX
C
      DOUBLE PRECISION ETMIN,ETMAX
      DOUBLE PRECISION TINY
      PARAMETER (TINY=1D-3)
      INTEGER ICOUNT
      DATA ICOUNT/0/
C
      IPVETO=0
      IF(ICKKW.EQ.0) RETURN
      IF(IHRD.EQ.7.OR.IHRD.EQ.8.OR.IHRD.EQ.13) THEN
        WRITE(*,*) 'JET MATCHING FOR HARD PROCESS ',IHRD
     $       ,' NOT IMPLEMENTED, STOP'
        STOP
      ENDIF
      IF(NLJETS.EQ.0.AND.IEXC.EQ.0) RETURN
C CHECK FOR EVENT ERROR OR ZERO WGT
      I=0
C     HERWIG/PYTHIA SPECIFIC
      CALL ALSHER(I)
      IF(I.EQ.1) RETURN
C
C     INITIALIZE DEBUG IF NEEDED
      IF(IDBG.EQ.1) THEN
        WRITE(1,*) ' '
        WRITE(1,*) 'NEW EVENT '
        WRITE(1,*) 'PARTONS'
      ENDIF
      IF(IDBG.EQ.2) THEN
        DO I=1,10
          PTPART(I)=0D0
          ETAPART(I)=0D0
          PTJETS(I)=0D0
          ETAJETS(I)=0D0
        ENDDO
        NMAX=0
      ENDIF
C
C     RECONSTRUCT PARTON-LEVEL EVENT
C     START FROM THE PARTONIC SYSTEM
      DO I=1,NLJETS
        IHEP=I+NJSTART
        DO J=1,4
          P(J,I)=PUP(J,IHEP)
        ENDDO
        PT(I)=SQRT(P(1,I)**2+P(2,I)**2)
        ETA(I)=-LOG(TAN(0.5D0*ATAN2(PT(I)+TINY,P(3,I))))
        PHI(I)=ATAN2(P(2,I),P(1,I))
        IF(IDBG.EQ.1) THEN
          WRITE(1,*) PT(I),ETA(I),PHI(I)
        ENDIF
      ENDDO
      IF(NLJETS.GT.0) CALL ALPSOR(PT,NLJETS,KP,2)  
C     
C     DISPLAY EVENT SEEN BY UPVETO:
      IF(IDBG.EQ.1) THEN
        DO I=1,NHEP
          WRITE(1,111) I,ISTHEP(I),IDHEP(I),JMOHEP(1,I),JMOHEP(2,I)
     $         ,PHEP(1,I),PHEP(2,I),PHEP(3,I)
        ENDDO
 111    FORMAT(5(I4,1X),3(F12.5,1X))
      ENDIF
C     DISPLAY PYTHIA EVENT:
C      CALL PYLIST(7)    ! PYTHIA USER PROCESS EVENT DUMP
C      CALL PYLIST(2)    ! PYTHIA FULL EVENT DUMP
C      CALL PYLIST(5)    ! PYTHIA HEPEVT DUMP
C
C     RECONSTRUCT SHOWERED JETS:
      CALL CALINI
      CALL CALDEL(NPFST,NPLST,NJLAST)
      CALL GETJET(RCLUS,ETCLUS,ETACLMAX)
      IF(NCJET.GT.0) CALL ALPSOR(ETJET,NCJET,K,2)              
      IF(IDBG.EQ.1) THEN
        WRITE(1,*) 'JETS'
        DO I=1,NCJET
          J=K(NCJET+1-I)
          ETAJET=PSERAP(PCJET(1,J))
          PHIJET=ATAN2(PCJET(2,J),PCJET(1,J))
          WRITE(1,*) ETJET(J),ETAJET,PHIJET
        ENDDO
      ENDIF
C     ANALYSE ONLY EVENTS WITH AT LEAST NLJETS-RECONSTRUCTED JETS
      IF(NCJET.LT.NLJETS) GOTO 999
C     ASSOCIATE PARTONS AND JETS, USING MIN(DELR) AS CRITERION
      NMATCH=0
      DO I=1,NCJET
        KPJ(I)=0
      ENDDO
      DO I=1,NLJETS
        DELRMIN=1D5
        DO 110 J=1,NCJET
          IF(KPJ(J).NE.0) GO TO 110
          ETAJET=PSERAP(PCJET(1,J))
          PHIJET=ATAN2(PCJET(2,J),PCJET(1,J))
          DPHI=ABS(PHI(KP(NLJETS-I+1))-PHIJET)
          IF(DPHI.GT.PI) DPHI=2.*PI-DPHI
          DELR=SQRT((ETA(KP(NLJETS-I+1))-ETAJET)**2+(DPHI)**2)
          IF(DELR.LT.DELRMIN) THEN
            DELRMIN=DELR
            JRMIN=J
          ENDIF
 110    CONTINUE
        ETMIN=1D10
        IF(DELRMIN.LT.1.5*RCLUS) THEN
          NMATCH=NMATCH+1
          KPJ(JRMIN)=I
          ETMIN=MIN(ETMIN,ETJET(JRMIN))
C     ASSOCIATE PARTONS AND MATCHED JETS:
          IF(IDBG.EQ.2) THEN
            PTPART(I)=PT(KP(NLJETS-I+1))
            ETAPART(I)=ETA(KP(NLJETS-I+1))
            PTJETS(I)=ETJET(JRMIN)
            ETAJETS(I)=PSERAP(PCJET(1,JRMIN))
            NMAX=NCJET
          ENDIF
C          WRITE(*,*) 'PARTON-JET',I,' BEST MATCH:',K(NCJET+1-JRMIN)
C     $           ,DELRMIN
        ENDIF
      ENDDO
      IF(NMATCH.LT.NLJETS) GOTO 999
C     REJECT EVENTS WITH LARGER JET MULTIPLICITY FROM EXCLUSIVE SAMPLE
      IF(NCJET.GT.NLJETS.AND.IEXC.EQ.1) GOTO 999
C     VETO EVENTS WHERE MATCHED JETS ARE SOFTER THAN NON-MATCHED ONES
      IF(IEXC.NE.1) THEN
        J=NCJET
        DO I=1,NLJETS
          IF(KPJ(K(J)).EQ.0) GOTO 999
          J=J-1
        ENDDO
      ENDIF
C
C     ADDITIONAL TREATMENT FOR HVQ EVENTS: VETO/ACCEPT JETS EMITTED BY HVQ'S
      IF(IHRD.LE.2.OR.IHRD.EQ.6.OR.IHRD.EQ.10.OR.IHRD.EQ.15) THEN
        CALL CALINI
C     RECOSTRUCT POSSIBLE JETS FROM RADIATION OFF THE TOP QUARKS
        CALL CALDEL_HVQ(NPFST,NPLST,NJLAST)
        CALL GETJET(RCLUS,ETCLUS,ETACLMAX)
C     IF NO EXTRA JET: ACCEPT EVENT
        IF(NCJET.EQ.0) RETURN
C     IF EXTRA JETS, REMOVE THOSE LYING WITHIN DRJMIN OF A B/C QUARK, TO
C     ALLOW THE SHOWER TO GOVERN THE DEVELOPMENT OF A B/C JET.
C     START BY FLAGGING VETOED Q AND QBAR OBJECTS:
        NHVQ=0
        DO I=1,NHEP
          ID=IDHEP(I)
          IF(INORAD(I).EQ.1.AND.ABS(ID).LE.5.AND.ABS(ID)
     $         .GE.4) THEN
            NHVQ=NHVQ+1
            IHVQ(NHVQ)=I
            ETAHVQ(NHVQ)=PSERAP(PHEP(1,I))
            PHIHVQ(NHVQ)=ATAN2(PHEP(2,I),PHEP(1,I))
          ENDIF
        ENDDO
        NMJET=NCJET
        DO I=1,NCJET
          ETAJET=PSERAP(PCJET(1,I))
          PHIJET=ATAN2(PCJET(2,I),PCJET(1,I))
          DO J=1,NHVQ
            DPHI=ABS(PHIHVQ(J)-PHIJET)
            IF(DPHI.GT.PI) DPHI=ABS(DPHI-2*PI)
            DELR=SQRT(DPHI**2+(ETAJET-ETAHVQ(J))**2)
            IF(DELR.LT.DRJMIN) THEN
              NMJET=NMJET-1
              ETJET(I)=0D0
            ENDIF
          ENDDO
        ENDDO
C     IF NO UNMATCHED JET: ACCEPT EVENT
        IF(NMJET.EQ.0) RETURN
C     IF JETS AND IEXC=1: REJECT EVENT
        IF(IEXC.EQ.1) GOTO 999
C     IF JETS AND IEXC=0: CHECK THAT JETS ARE SOFTER THAN MATCHED ONES
        ETMAX=0D0
        DO I=1,NCJET
          ETMAX=MAX(ETMAX,ETJET(I))
        ENDDO
        IF(ETMAX.GT.ETMIN) GOTO 999
      ENDIF
      RETURN
C     HERWIG/PYTHIA TERMINATION:
 999  CALL ALSHEN
      IPVETO=1
      END


C----------------------------------------------------------------------
      SUBROUTINE UPINIT
C----------------------------------------------------------------------
C     HERWIG/PYTHIA UNIVERSAL EVENT INITIALITION ROUTINE
C----------------------------------------------------------------------
      IMPLICIT NONE
c process code
      integer ihrd
      integer itopprc
c--event file data
      integer Nunit,NunitOut,NunitIni
      character*512 filename,tmpstr
c total number of input events
      common/AHio/Nunit,NunitOut,NunitIni,filename
c inputs for the matching algorithm
      integer iexc,npfst,nplst,nljets,njstart,njlast
     $     ,ickkw
      double precision etclus,rclus,etaclmax
      common/AHopts/etclus,rclus,etaclmax,iexc,npfst
     $     ,nplst,nljets,njstart,njlast,ickkw
c process and particles parameters
      double precision mc,mb,mt,mw,mz,mh
      double precision ebeam
      integer ndns,ih1,ih2       
      integer nw,nz,nh,nph
      integer ihvy,ihvy2
c pdf set type
      character pdftyp*25
c total number of partons
      integer npart
      common/AHppara/mc,mb,mt,mw,mz,mh,
     & ebeam,ih1,ih2,
     & ihrd,itopprc,
     & nw,nz,nh,nph,
     & ihvy,ihvy2,
     & npart,ndns,pdftyp
c weight information
      real *8 maxwgt,avgwgt,errwgt,totlum
      integer unwev
      common/AHwgts/maxwgt,avgwgt,errwgt,totlum,unwev
c general parameters
      integer nparam
      parameter (nparam=200)
      integer parlen,partyp
      character chpar*8,chpdes*70
      double precision parval
      common/AHpars/parval(nparam),chpar(nparam),chpdes(nparam)
     $     ,parlen(nparam),partyp(nparam)
c global event cuts
      double precision ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +     ptcmin,ptcmax,etacmax,drcmin,
     +     ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +     ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      common/AHcuts/ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +        ptcmin,ptcmax,etacmax,drcmin,
     +        ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +        ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      CHARACTER *3 CSHO
C--   GUP Event common block
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &     IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &     ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &     SPINUP(MAXNUP)
C--   GUP common block
      INTEGER MAXPUP
      PARAMETER(MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON /HEPRUP/ IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &     IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),
     &     XMAXUP(MAXPUP),LPRUP(MAXPUP)
C     CALSIM AND JET VARIABLES                             
      INTEGER NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,PI,ET,DELPHI,CPHCAL,SPHCAL,DELY,
     &     CTHCAL,STHCAL,PCJET,ETJET
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60,PI=3.141593D0)
      COMMON/CALOR/DELY,DELPHI,ET(NCY,NCPHI),
     $     CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN
     $     ,YCMAX
C     
      INTEGER IEND,INORAD
      COMMON/SHVETO/IEND,INORAD(MAXNUP)
c     
C     LOCAL VARIABLES
      CHARACTER*70 STDUMMY
      INTEGER I,NTMP,IHEPMIN
      DOUBLE PRECISION TMP
      DOUBLE PRECISION PBEAM1,PBEAM2
C     USER ACCESS TO MATCHING PARAMETERS
      INTEGER IUSRMAT
      PARAMETER (IUSRMAT=1)
      character*512                           CXpar
      integer         Ixpar 
      double precision            RXpar
      common /EXGPAR/ IXpar(100), RXpar(100), CXpar(100)
C
C USER ACCESS TO INPUT PARAMETERS
c      CHARACTER XXX*500, PROCESS*500
      CHARACTER PROCESS*500
      INTEGER FIREVT 
      WRITE(*,*) 'INPUT NAME OF FILE CONTAINING EVENTS'
      WRITE(*,*) '(FOR "file.unw" ENTER "file")'
      FILENAME = CXpar(1) ! 'alpgen'
      PROCESS = CXpar(2)
      WRITE(*,*) FILENAME   
      FIREVT = IXpar(1)
      WRITE(*,*) ' FIRST EVENT TO READ ', FIREVT
      IF (FIREVT.LT.1) THEN
        WRITE(*,*)'Warning: First event parameter in MY-INPARM.DAT file'
        WRITE(*,*)'         cannot be less than 1. So it was set to 1.'
        FIREVT=1
      ENDIF
      CALL STRCATH(FILENAME,'-pyt.dat',TMPSTR)
      WRITE(*,*) 'NAME OF TXT FILE CONTAINING GENERATION PARAMETERS', 
     +     TMPSTR
      WRITE(*,*) '!!WARNING!!If the file already exists the program' 
      WRITE(*,*) ' will crash, please rename the file and run again'
      OPEN(33,FILE=TMPSTR,STATUS='unknown') ! NEW')
      CALL STRCATH(FILENAME,'.unw',TMPSTR)
      call GETUNIT(NUNIT)
      OPEN(UNIT=NUNIT,FILE=TMPSTR,STATUS='OLD')
      CALL STRCATH(FILENAME,'_unw.par',TMPSTR)
      CALL GETUNIT(NUNITINI)
      OPEN(UNIT=NUNITINI,FILE=TMPSTR,STATUS='OLD')
C     OPEN A LOG FILE
      CALL ALSHCD(CSHO)
C      IF(CSHO.EQ.'HER') THEN
C        CALL STRCATH(FILENAME,'.her-log',TMPSTR)
C      ELSE 
C        CALL STRCATH(FILENAME,'.pyt-log',TMPSTR)
C      ENDIF
C      CALL GETUNIT(NUNITOUT)
C      OPEN(UNIT=NUNITOUT,NAME=TMPSTR,STATUS='UNKNOWN')
C     START READING FILE
      DO I=1,10000
        READ(NUNITINI,'(A)') STDUMMY
        IF(STDUMMY(1:4).EQ.'****') GOTO 10
        WRITE(*,*) STDUMMY
        WRITE(33,*) STDUMMY
      ENDDO
C     
C     READ IN INPUT PARAMETERS
 10   READ(NUNITINI,*) IHRD
C     
c     IF(IHRD.EQ.12) THEN
c     WRITE(*,*) 'HJET PROCESSES NOT AVAILABLE AS YET'
c     STOP
c     ENDIF
      READ(NUNITINI,*) MC,MB,MT,MW,MZ,MH
      DO I=1,1000
        READ(NUNITINI,*,ERR=20) NTMP,TMP
        PARVAL(NTMP)=TMP
      ENDDO
 20   CONTINUE
      READ(NUNITINI,*) AVGWGT,ERRWGT
      READ(NUNITINI,*) UNWEV,TOTLUM
      WRITE(33,*) " "
      WRITE(33,*) "INPUT CROSS SECTION (PB):",AVGWGT," +/-",ERRWGT
      WRITE(33,*) "NUMBER OF INPUT EVENTS:",UNWEV
      WRITE(33,*) "INTEGRATED LUMINOSITY:",TOTLUM
C     WRITE PARAMETER VALUES
      CALL AHSPAR
      PBEAM1=DBLE(EBEAM)
      PBEAM2=DBLE(EBEAM)
      IH1=1
C     CONVERT PDF TYPES
      CALL PDFCONVH(NDNS,NTMP,PDFTYP)
C     DEFINE RANGE FOR PARTONS TO BE USED IN MATCHING
      NLJETS=PARVAL(10)     !  NJETS
      DO I=1,MAXNUP
        INORAD(I)=0
      ENDDO
      CALL ALSHCD(CSHO)
      IF(CSHO.EQ.'HER') THEN
        NPFST=149
        NPLST=149
C     HERWIG: ALL SHOWERS ORIGINATE FROM IHEP=6
        IEND=6
C     HERWIG: HEPEVT EVENT RECORD FOR FINAL STATE STARTS AT 7=6+1
        IHEPMIN=6
      ELSE
        NPFST=1
        NPLST=1
C     PYTHIA: ALL SHOWERS ORIGINATE FROM IHEP=O
        IEND=0
C     PYTHIA: HEPEVT EVENT RECORD FOR FINAL STATE STARTS AT 1=0+1
        IHEPMIN=0
        IDPRUP=661
      ENDIF
      IF(IHRD.LE.2) THEN
C        NLJETS=NPART-6
        NJSTART=4
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE W
        INORAD(IHEPMIN+2+NLJETS+1)=1
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM HEAVY QUARK
C     PAIR
        INORAD(IHEPMIN+1)=1
        INORAD(IHEPMIN+2)=1
      ELSEIF(IHRD.LE.4) THEN
C        NLJETS=NPART-4
        NJSTART=2
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE W
        IF(CSHO.EQ.'HER'.AND.NLJETS.EQ.0) THEN
C     ALLOW RADIATION FROM IHEPMIN+1 IN HERWIG WHEN NJET=0
          INORAD(IHEPMIN+NLJETS+1)=0
        ELSE 
          INORAD(IHEPMIN+NLJETS+1)=1
        ENDIF
      ELSEIF(IHRD.EQ.5) THEN
C        NLJETS=NPART-3*(NW+NZ)-NH-2
        NJSTART=2
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE W
        IF(CSHO.EQ.'HER'.AND.NLJETS.EQ.0.AND.(NW+NZ+NH+NPH).EQ.1) THEN
C     ALLOW RADIATION FROM IHEPMIN+1 IN HERWIG WHEN NJET=0 AND NW+NZ+NH
C     +NPH=1
          INORAD(IHEPMIN+NLJETS+1)=0
        ELSE 
          DO I=1,NW+NZ+NH+NPH
            INORAD(IHEPMIN+NLJETS+I)=1
          ENDDO
        ENDIF
      ELSEIF(IHRD.EQ.6) THEN
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM HEAVY QUARK
C     PAIR
C     NLJETS=NPART-8  (IHVY.EQ.6)     NLJETS=NPART-4  (IHVY.LT.6)
        INORAD(IHEPMIN+1)=1
        INORAD(IHEPMIN+2)=1
        NJSTART=4
        NJLAST=155
      ELSEIF(IHRD.EQ.9) THEN
C     NLJETS=NPART-2
        NJSTART=2
        NJLAST=155
      ELSEIF(IHRD.EQ.10) THEN
C        NLJETS=NPART-4
        NJSTART=3
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE W
        INORAD(IHEPMIN+NLJETS+1+1)=1
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE CHARM
        INORAD(IHEPMIN+1)=1
      ELSEIF(IHRD.EQ.11) THEN
C        NLJETS=NPART-2-NPH
        NJSTART=2
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING THE HARD PHOTONS
        DO I=1,NPH
          INORAD(IHEPMIN+NLJETS+I)=1
        ENDDO
      ELSEIF(IHRD.EQ.12) THEN
C        NLJETS=NPART-2-NH
        NJSTART=2
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING THE HIGGS DECAY PRODUCTS
        IF(NLJETS+NH.GT.1) THEN
          DO I=1,NH
            INORAD(IHEPMIN+NLJETS+I)=1
          ENDDO
        ELSE
          IF(CSHO.EQ.'HER'.AND.NLJETS.EQ.0) THEN
C     ALLOW RADIATION FROM IHEPMIN+1 IN HERWIG WHEN NJET=0
            INORAD(IHEPMIN+1)=0
          ELSE 
            INORAD(IHEPMIN+NLJETS+1)=1
          ENDIF
        ENDIF
      ELSEIF(IHRD.EQ.14) THEN
C        NLJETS=NPART-4-NPH
        NJSTART=2
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE W
        IF(CSHO.EQ.'HER'.AND.NLJETS.EQ.0.AND.NPH.EQ.0) THEN
C     ALLOW RADIATION FROM IHEPMIN+1 IN HERWIG WHEN NJET=0
          INORAD(IHEPMIN+NLJETS+1)=0
        ELSE 
          INORAD(IHEPMIN+NLJETS+1)=1
        ENDIF
      ELSEIF(IHRD.EQ.15) THEN
C        NLJETS=NPART-6-NPH
        NJSTART=2
        NJLAST=155
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM THE W
        INORAD(IHEPMIN+2+NLJETS+NPH+1)=1
C     DO NOT INCLUDE IN MATCHING RADIATION ORIGINATING FROM HEAVY QUARK
C     PAIR
C     NOTICE THAT HERE THE LIGHT JETS PRECEDE THE QQ PAIR
        INORAD(IHEPMIN+NLJETS+1)=1
        INORAD(IHEPMIN+NLJETS+2)=1
      ENDIF
C
C     INPUT JET MATCHING CRITERIA
C
      IF(ICKKW.EQ.1) THEN
        WRITE(*,*) ' '
        WRITE(*,*) 'INPUT 0 FOR INCLUSIVE JET SAMPLE, 1 FOR EXCLUSIVE'
        WRITE(*,*) '(SELECT 0 FOR HIGHEST PARTON MULTIPLICITY SAMPLE)' 
        WRITE(*,*) '(SELECT 1 OTHERWISE)'
C       READ(*,*) IEXC
	IEXC = IXpar(2)
        WRITE(*,*) ' IEXC = ',IEXC
        IF(NLJETS.GT.0) THEN
          WRITE(*,*) 'INPUT ET(CLUS), R(CLUS)'
          ETCLUS=MAX(PTJMIN+5,1.2*PTJMIN)
          RCLUS=DRJMIN
          WRITE(*,*) '(SUGGESTED VALUES:',ETCLUS,RCLUS,')'
C         READ(*,*) ETCLUS,RCLUS,ETACLMAX
          ETCLUS = RXpar(1)
          RCLUS  = RXpar(2)
          ETACLMAX = ETAJMAX
        ELSEIF(NLJETS.EQ.0) THEN
          WRITE(*,*) 'INPUT ET(CLUS), R(CLUS)'
          WRITE(*,*) '(MUST MATCH VALUES USED IN PROCESSING',
     +         ' OF NJET>0 EVENTS; THESE DEFAULT TO:'
          WRITE(*,*)
     +         'ETCLUS=MAX(PTJMIN+5,1.2*PTJMIN) RCLUS=DRJMIN, ',
     +         ' ETACLMAX=ETAJMAX)'
C         READ(*,*) ETCLUS,RCLUS,ETACLMAX
          ETCLUS = RXpar(1)
          RCLUS  = RXpar(2)
          ETACLMAX = ETAJMAX
        ENDIF
        WRITE(*,*) ' '
        WRITE(*,*) 'JET PARAMETERS FOR MATCHING:'
        WRITE(*,*) 'ET >',ETCLUS,' ETACLUS <',ETACLMAX,' R =',RCLUS
        WRITE(*,*) 'DR(PARTON-JET) <',1.5*RCLUS
      ENDIF
C     
C     CALORIMETER ETA RANGE
C     YCMAX=ETAJMAX+DRJMIN    
      YCMAX=ETACLMAX+RCLUS      ! MLM August 24
      YCMIN=-YCMAX
C     FROM NOW ON, PROCESS THE INFORMATION READ IN, TO COMPLETE SETTING
C     UP THE GUP COMMON
C     
C--   SET UP THE BEAMS
C--   ID'S OF BEAM PARTICLES
      IF(IH1.EQ.1) THEN
        IDBMUP(1) = 2212
      ELSEIF(IH1.EQ.-1) THEN
        IDBMUP(1) =-2212
      ELSE
        WRITE(*,*) 'BEAM 1 NOT PROPERLY INITIALISED, STOP'
        STOP
      ENDIF
      IF(IH2.EQ.1) THEN
        IDBMUP(2) = 2212
      ELSEIF(IH2.EQ.-1) THEN
        IDBMUP(2) =-2212
      ELSE
        WRITE(*,*) 'BEAM 2 NOT PROPERLY INITIALISED, STOP'
        STOP
      ENDIF
      EBMUP(1) = ABS(PBEAM1)
      EBMUP(2) = ABS(PBEAM2)
C--   PDF'S FOR THE BEAMS; WILL BE EVALUATED USING THE NDNS VARIABLE
C     READ IN EARLIER
      PDFGUP(1) = -1
      PDFGUP(2) = -1
      PDFSUP(1) = -1
      PDFSUP(2) = -1
C--   WHAT DO DO WITH THE WEIGHTS(WE ARE GENERATING UNWEIGHTED EVENTS)
      IDWTUP = 3
C--   ONLY ONE PROCESS
      NPRUP  = 1
C--   CROSS SECTION
      XSECUP(1) = avgwgt
C--   ERROR ON THE CROSS SECTION
      XERRUP(1) = errwgt
C--   MAXIMUM WEIGHT
      XMAXUP(1) = avgwgt
C--   HERWIG/PYTHIA SPECIFIC PART
      CALL ALSHIN(I)
      LPRUP(1) = I
      END

C     DECK  ID>, UPEVNT.
*     CMZ :-        -13/02/02  07.20.46  by  Peter Richardson
*--   Author :    Michelangelo Mangano
C----------------------------------------------------------------------
      SUBROUTINE UPEVNT
C----------------------------------------------------------------------
c     Puts Alpgen event into GUPI common block HEPEU
c----------------------------------------------------------------------
      implicit none
c process code
      integer ihrd
      integer itopprc
c--event file data
      integer Nunit,NunitOut,NunitIni
      character*512 filename,tmpstr
c total number of input events
      common/AHio/Nunit,NunitOut,NunitIni,filename
c inputs for the matching algorithm
      integer iexc,npfst,nplst,nljets,njstart,njlast
     $     ,ickkw
      double precision etclus,rclus,etaclmax
      common/AHopts/etclus,rclus,etaclmax,iexc,npfst
     $     ,nplst,nljets,njstart,njlast,ickkw
c process and particles parameters
      double precision mc,mb,mt,mw,mz,mh
      double precision ebeam
      integer ndns,ih1,ih2       
      integer nw,nz,nh,nph
      integer ihvy,ihvy2
c pdf set type
      character pdftyp*25
c total number of partons
      integer npart
      common/AHppara/mc,mb,mt,mw,mz,mh,
     & ebeam,ih1,ih2,
     & ihrd,itopprc,
     & nw,nz,nh,nph,
     & ihvy,ihvy2,
     & npart,ndns,pdftyp
c weight information
      real *8 maxwgt,avgwgt,errwgt,totlum
      integer unwev
      common/AHwgts/maxwgt,avgwgt,errwgt,totlum,unwev
c general parameters
      integer nparam
      parameter (nparam=200)
      integer parlen,partyp
      character chpar*8,chpdes*70
      double precision parval
      common/AHpars/parval(nparam),chpar(nparam),chpdes(nparam)
     $     ,parlen(nparam),partyp(nparam)
c global event cuts
      double precision ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +     ptcmin,ptcmax,etacmax,drcmin,
     +     ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +     ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      common/AHcuts/ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +        ptcmin,ptcmax,etacmax,drcmin,
     +        ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +        ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
C--   GUP Event common block
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &     IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &     ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &     SPINUP(MAXNUP)
C--   GUP Run common block
      INTEGER MAXPUP
      PARAMETER(MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON /HEPRUP/ IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &     IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),
     &     XMAXUP(MAXPUP),LPRUP(MAXPUP)
c     
c     local variables
      INTEGER INIT
      DATA INIT/0/
      CHARACTER *3 CSHO
      INTEGER MAXPAR
      PARAMETER (MAXPAR=100)
      INTEGER NEV,IPROC,IFL(MAXPAR)
      REAL SQ,SP(3,MAXPAR),SM(MAXPAR),SWGTRES
      INTEGER I,IUP,IWCH,IST
      REAL *8 TMP,WGTRES
C     LOCAL VARIABLES FOR TOP DECAYS
      INTEGER IT,ITB,IW,IWDEC,IBUP,IWUP
C     LOCAL VARIABLES TOP HIGGS DECAYS
      INTEGER IH
C     LOCAL VARIABLES FOR GAUGE BOSON  DECAYS
      INTEGER IVSTART,IVEND,NVB
C     
C UPDATE MAXIMUM NMBER OF ALLOWED ERRORS
      CALL ALSHER(I)
C
      IST=0
C     INPUT EVENT NUMBER, PROCESS TYPE, N PARTONS, SAMPLE'S AVERAGE
C     WEIGHT AND QSCALE
      READ(NUNIT,2,END=500,ERR=501) NEV,IPROC,NPART,SWGTRES,SQ
 2    FORMAT(I8,1X,I4,1X,I2,2(1X,E12.6))
C     FLAVOUR, COLOUR AND Z-MOMENTUM OF INCOMING PARTONS
      READ(NUNIT,8) IFL(1),ICOLUP(1,1),ICOLUP(2,1),SP(3,1)
      READ(NUNIT,8) IFL(2),ICOLUP(1,2),ICOLUP(2,2),SP(3,2)
C     FLAVOUR, COLOUR, 3-MOMENTUM AND MASS OF OUTGOING PARTONS
      DO I=3,NPART
        READ(NUNIT,9) IFL(I),ICOLUP(1,I),ICOLUP(2,I),SP(1,I),SP(2,I)
     $       ,SP(3,I),SM(I)
      ENDDO
 8    FORMAT(I8,1X,2(I4,1X),F10.3)
 9    FORMAT(I8,1X,2(I4,1X),4(1X,F10.3))
C     
C     START PROCESSING INPUT DATA
C     
C     SCALES AND WEIGHTS
      SCALUP=DBLE(SQ)
      IF(IDWTUP.EQ.3) THEN
	XWGTUP=DBLE(SWGTRES) !AVGWGT
      ELSE
        WRITE(*,*) 'ONLY UNWEIGHTED EVENTS ACCEPTED AS INPUT, STOP'
        STOP
      ENDIF
c     
c---  incoming lines
      do 100 i=1,2
        iup=i
        idup(iup)=ifl(i)
        istup(iup)=-1
        mothup(1,iup)=0
        mothup(2,iup)=0
        pup(1,iup)=0.
        pup(2,iup)=0.
        pup(3,iup)=dble(Sp(3,iup))
        pup(4,iup)=abs(pup(3,iup))
        pup(5,iup)=0d0
 100  continue
c---  outgoing lines
      do 110 i=3,npart
        iup=i
        idup(iup)=ifl(i)
        istup(iup)=1
        mothup(1,iup)=1
        mothup(2,iup)=2
        pup(1,iup)=dble(Sp(1,i))
        pup(2,iup)=dble(Sp(2,i))
        pup(3,iup)=dble(Sp(3,i))
        pup(5,iup)=dble(Sm(i))
        tmp=(pup(5,iup)**2+pup(1,iup)**2+pup(2,iup)**2+pup(3,iup)**2)
        pup(4,iup)=sqrt(tmp)
 110  continue
c     
      nup=npart
c---  set up colour structure labels
      Do iup=1,nup
        if(icolup(1,iup).ne.0) icolup(1,iup)=icolup(1,iup)+500
        if(icolup(2,iup).ne.0) icolup(2,iup)=icolup(2,iup)+500
      Enddo
c     
c     
c     and now consider assignements specific to individual hard
C     processes
c     
c---  W/Z/gamma b bbar + jets, or W/Z + jets
      if (ihrd.le.4.or.ihrd.eq.10.OR.IHRD.EQ.14.OR.IHRD.EQ.15) then
        iwch=0
        do iup=nup-1,nup
          mothup(1,iup)=nup+1
          mothup(2,iup)=0
          if(ihrd.ne.2) iwch=iwch-mod(idup(iup),2)
c     electron+nubar -> 11 + (-12) => -(1)+0 = -1  => W-
c     positron+nu    -> -11+ 12    => -(-1)+0 = -1 => W+
c     u dbar -> 2 -1  => 0 -(-1) = 1 => W+
c     c dbar -> 4 -1  => W+
c etc.
        enddo
        iup=nup+1
        If (iwch.gt.0) then
          idup(iup)=24
        Elseif (iwch.lt.0) then
          idup(iup)=-24
        Else
          idup(iup)=23
        Endif
        istup(iup)=2
        mothup(1,iup)=1
        mothup(2,iup)=2
        tmp=pup(4,iup-2)+pup(4,iup-1)
        pup(4,iup)=tmp
        tmp=tmp**2
        do i=1,3
          pup(i,iup)=pup(i,iup-2)+pup(i,iup-1)
          tmp=tmp-pup(i,iup)**2
        enddo
        pup(5,iup)=sqrt(tmp)
        nup=nup+1
        icolup(1,nup)=0
        icolup(2,nup)=0
c---  nW + mZ + kH + jets
      elseif (ihrd.eq.5) then
c     find first gauge bosons
        ivstart=0
        ivend=0
        do i=1,npart
          if(abs(idup(i)).eq.24.or.idup(i).eq.23) then
            istup(i)=2
            if(ivstart.eq.0) ivstart=i
            ivend=i+1
          endif
        enddo
        nvb=ivend-ivstart
c     decay products pointers, starting from the end
        do i=1,nvb
          mothup(1,npart-2*i+2)=ivend-i
          mothup(1,npart-2*i+1)=ivend-i
          mothup(2,npart-2*i+2)=0
          mothup(2,npart-2*i+1)=0
        enddo
c---  t tbar + jets
c     t tb jets f fbar f fbar W+ b W- bbar
      elseif (ihrd.eq.6.and.abs(ifl(3)).eq.6) then
c     reset top status codes
        istup(3)=2
        istup(4)=2
        if(ifl(3).eq.6) then
          it=3
          itb=4
        else
          it=4
          itb=3
        endif
c     reconstruct W's from decay products
        do iw=1,2
          iwdec=nup-5+2*iw
          iwup=nup+iw
          ibup=iwup+2
          iwch=0
          do iup=iwdec,iwdec+1
            mothup(1,iup)=iwup
            mothup(2,iup)=0
            iwch=iwch-mod(idup(iup),2)
c     electron+nubar -> 11 + (-12) = -1 => W-
c     d + ubar -> 1 + (-2) = -1 => W-
c     positron+nu    -> -11+ 12    =  1 => W+
c     u + dbar -> 2 + (-1) = 1 => W+
          enddo
          If (iwch.gt.0) then
            idup(iwup)=24
            idup(ibup)=5
            mothup(1,iwup)=it
            mothup(2,iwup)=0
            mothup(1,ibup)=it
            mothup(2,ibup)=0
          Elseif (iwch.lt.0) then
            idup(iwup)=-24
            idup(ibup)=-5
            mothup(1,iwup)=itb
            mothup(2,iwup)=0
            mothup(1,ibup)=itb
            mothup(2,ibup)=0
          Endif
          istup(iwup)=2
          istup(ibup)=1
c     reconstruct W momentum
          tmp=pup(4,iwdec)+pup(4,iwdec+1)
          pup(4,iwup)=tmp
          tmp=tmp**2
          do i=1,3
            pup(i,iwup)=pup(i,iwdec)+pup(i,iwdec+1)
            tmp=tmp-pup(i,iwup)**2
          enddo
          pup(5,iwup)=sqrt(tmp)
c     reconstruct b momentum
          tmp=pup(4,mothup(1,iwup))-pup(4,iwup)
          pup(4,ibup)=tmp 
          tmp=tmp**2
          do i=1,3
            pup(i,ibup)=pup(i,mothup(1,iwup))-pup(i,iwup)
            tmp=tmp-pup(i,ibup)**2
          enddo
          pup(5,ibup)=sqrt(tmp)
          icolup(1,iwup)=0
          icolup(2,iwup)=0
          icolup(1,ibup)=icolup(1,mothup(1,iwup))
          icolup(2,ibup)=icolup(2,mothup(1,iwup))
        enddo
c     stop
        nup=nup+4
c---  H t tbar + jets
c     H t tb jets f fbar f fbar W+ b W- bbar
      elseif (ihrd.eq.8.and.abs(ifl(4)).eq.6) then
c     reset top status codes
        istup(4)=2
        istup(5)=2
        if(ifl(4).eq.6) then
          it=4
          itb=5
        else
          it=5
          itb=4
        endif
c     reconstruct W's from decay products
        do iw=1,2
          iwdec=nup-5+2*iw
          iwup=nup+iw
          ibup=iwup+2
          iwch=0
          do iup=iwdec,iwdec+1
            mothup(1,iup)=iwup
            mothup(2,iup)=0
            iwch=iwch-mod(idup(iup),2)
c     electron+nubar -> 11 + (-12) = -1 => W-
c     d + ubar -> 1 + (-2) = -1 => W-
c     positron+nu    -> -11+ 12    =  1 => W+
c     u + dbar -> 2 + (-1) = 1 => W+
          enddo
          If (iwch.gt.0) then
            idup(iwup)=24
            idup(ibup)=5
            mothup(1,iwup)=it
            mothup(2,iwup)=0
            mothup(1,ibup)=it
            mothup(2,ibup)=0
          elseif (iwch.lt.0) then
            idup(iwup)=-24
            idup(ibup)=-5
            mothup(1,iwup)=itb
            mothup(2,iwup)=0
            mothup(1,ibup)=itb
            mothup(2,ibup)=0
          endif
          istup(iwup)=2
          istup(ibup)=1
c     reconstruct W momentum
          tmp=pup(4,iwdec)+pup(4,iwdec+1)
          pup(4,iwup)=tmp
          tmp=tmp**2
          do i=1,3
            pup(i,iwup)=pup(i,iwdec)+pup(i,iwdec+1)
            tmp=tmp-pup(i,iwup)**2
          enddo
          pup(5,iwup)=sqrt(tmp)
c     reconstruct b momentum
          tmp=pup(4,mothup(1,iwup))-pup(4,iwup)
          pup(4,ibup)=tmp 
          tmp=tmp**2
          do i=1,3
            pup(i,ibup)=pup(i,mothup(1,iwup))-pup(i,iwup)
            tmp=tmp-pup(i,ibup)**2
          enddo
          pup(5,ibup)=sqrt(tmp)
          icolup(1,iwup)=0
          icolup(2,iwup)=0
          icolup(1,ibup)=icolup(1,mothup(1,iwup))
          icolup(2,ibup)=icolup(2,mothup(1,iwup))
        enddo
c     stop
        nup=nup+4
c---  SINGLE TOP
c     Input: T  
c     output: jets t b w f fbar t b w f fbar 
      elseif (ihrd.eq.13) then
        nw=1
        if(itopprc.ge.3) nw=2
c     assign mass to the incoming bottom quark, if required
        DO I=1,2
          IF(ABS(IFL(I)).EQ.5) THEN
            IUP=I
            PUP(5,IUP)=mb
            PUP(4,IUP)=SQRT(PUP(3,IUP)**2+PUP(5,IUP)**2)
          ENDIF
        ENDDO
        istup(3)=2
        it=0
        itb=0
        if(ifl(3).eq.6) then
          it=3
        elseif(ifl(3).eq.-6) then
          itb=3
        else
          write(*,*) 'wrong assumption about top position, stop'
          stop
        endif
c
c     TOP DECAY
c     reconstruct W's from decay products
c
c iwdec: 1st W decay product.
        if(nw.eq.1) then
          iwdec=nup-1
        elseif(nw.eq.2) then
          iwdec=nup-3
        endif
c put W and b  at the end
        iwup=nup+1
        ibup=iwup+1
c
        iwch=0
        do iup=iwdec,iwdec+1
          mothup(1,iup)=iwup
          mothup(2,iup)=0
          iwch=iwch-mod(idup(iup),2)
c     electron+nubar -> 11 + (-12) = -1 => W-
c     d + ubar -> 1 + (-2) = -1 => W-
c     positron+nu    -> -11+ 12    =  1 => W+
c     u + dbar -> 2 + (-1) = 1 => W+
        enddo
        If (iwch.gt.0) then
          idup(iwup)=24
          idup(ibup)=5
          mothup(1,iwup)=it
          mothup(2,iwup)=0
          mothup(1,ibup)=it
          mothup(2,ibup)=0
        Elseif (iwch.lt.0) then
          idup(iwup)=-24
          idup(ibup)=-5
          mothup(1,iwup)=itb
          mothup(2,iwup)=0
          mothup(1,ibup)=itb
          mothup(2,ibup)=0
        Endif
        istup(iwup)=2
        istup(ibup)=1
c     reconstruct W momentum
        tmp=pup(4,iwdec)+pup(4,iwdec+1)
        pup(4,iwup)=tmp
        tmp=tmp**2
        do i=1,3
          pup(i,iwup)=pup(i,iwdec)+pup(i,iwdec+1)
          tmp=tmp-pup(i,iwup)**2
        enddo
        pup(5,iwup)=sqrt(tmp)
c     reconstruct b momentum
        tmp=pup(4,mothup(1,iwup))-pup(4,iwup)
        pup(4,ibup)=tmp 
        tmp=tmp**2
        do i=1,3
          pup(i,ibup)=pup(i,mothup(1,iwup))-pup(i,iwup)
          tmp=tmp-pup(i,ibup)**2
        enddo
c        write(*,*) (pup(i,ibup),i=1,4),sqrt((tmp))
        pup(5,ibup)=sqrt(tmp)
        icolup(1,iwup)=0
        icolup(2,iwup)=0
        icolup(1,ibup)=icolup(1,mothup(1,iwup))
        icolup(2,ibup)=icolup(2,mothup(1,iwup))
c     
        nup=nup+2
        if(nw.eq.2) then
c
c     W DECAY
c
c iwdec: 1st W decay product. 
          iwdec=nup-3
c iwup: location of the W in the event record
          iwup=nup-6
          iwch=0
          do iup=iwdec,iwdec+1
            mothup(1,iup)=iwup
            mothup(2,iup)=0
            iwch=iwch-mod(idup(iup),2)
c     electron+nubar -> 11 + (-12) = -1 => W-
c     d + ubar -> 1 + (-2) = -1 => W-
c     positron+nu    -> -11+ 12    =  1 => W+
c     u + dbar -> 2 + (-1) = 1 => W+
          enddo
          istup(iwup)=2
          icolup(1,iwup)=0
          icolup(2,iwup)=0
        endif
      endif
c     herwig debugging:
c      call HWUPUP
      return
c     
c     
c     end of file
 500  ist=1
c     error reading file
 501  if(ist.eq.0) ist=2
C     RESET CROSS-SECTION INFORMATION FOR END OF RUN AND FINALIZE
      IF(IST.GT.0) THEN
        CALL ALSFIN
      ENDIF
      close(Nunit)
      close(NunitOut)
      END

c-------------------------------------------------------------------
      subroutine AHspar
c     set list of parameters types and assign default values
c-------------------------------------------------------------------
      implicit none
c process code
      integer ihrd
      integer itopprc
c--event file data
      integer Nunit,NunitOut,NunitIni
      character*512 filename,tmpstr
c total number of input events
      common/AHio/Nunit,NunitOut,NunitIni,filename
c inputs for the matching algorithm
      integer iexc,npfst,nplst,nljets,njstart,njlast
     $     ,ickkw
      double precision etclus,rclus,etaclmax
      common/AHopts/etclus,rclus,etaclmax,iexc,npfst
     $     ,nplst,nljets,njstart,njlast,ickkw
c process and particles parameters
      double precision mc,mb,mt,mw,mz,mh
      double precision ebeam
      integer ndns,ih1,ih2       
      integer nw,nz,nh,nph
      integer ihvy,ihvy2
c pdf set type
      character pdftyp*25
c total number of partons
      integer npart
      common/AHppara/mc,mb,mt,mw,mz,mh,
     & ebeam,ih1,ih2,
     & ihrd,itopprc,
     & nw,nz,nh,nph,
     & ihvy,ihvy2,
     & npart,ndns,pdftyp
c weight information
      real *8 maxwgt,avgwgt,errwgt,totlum
      integer unwev
      common/AHwgts/maxwgt,avgwgt,errwgt,totlum,unwev
c general parameters
      integer nparam
      parameter (nparam=200)
      integer parlen,partyp
      character chpar*8,chpdes*70
      double precision parval
      common/AHpars/parval(nparam),chpar(nparam),chpdes(nparam)
     $     ,parlen(nparam),partyp(nparam)
c global event cuts
      double precision ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +     ptcmin,ptcmax,etacmax,drcmin,
     +     ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +     ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      common/AHcuts/ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +        ptcmin,ptcmax,etacmax,drcmin,
     +        ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +        ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
c
      ih2=parval(2)
      ebeam=parval(3)
      ndns=parval(4)
      ickkw=parval(7)
      ihvy=parval(11)
      ihvy2=parval(12)
      nw=parval(13)
      nz=parval(14)
      nh=parval(15)
      nph=parval(16)
      ptjmin=parval(30)
      ptbmin=parval(31)
      ptcmin=parval(32)
      ptlmin=parval(33)
      metmin=parval(34)
      ptphmin=parval(35)
      etajmax=parval(40)
      etabmax=parval(41)
      etacmax=parval(42)
      etalmax=parval(43)
      etaphmax=parval(44)
      drjmin=parval(50)
      drbmin=parval(51)
      drcmin=parval(52)
      drlmin=parval(55)
      drphjmin=parval(56)
      drphlmin=parval(57)
      drphmin=parval(58)
      mllmin=parval(61)
      mllmax=parval(62)
      itopprc=parval(102)
c
      end

CDECK  ID>, STRNUM.
*CMZ :-        -13/02/02  07.20.46  by  Peter Richardson
*-- Author :    Michelangelo Mangano
C----------------------------------------------------------------------
      subroutine strnumH(string,num)
C----------------------------------------------------------------------
c- writes the number num on the string string starting at the blank
c- following the last non-blank character
C----------------------------------------------------------------------
      character * (*) string
      character * 20 tmp
      l = len(string)
      write(tmp,'(i15)')num
      j=1
      dowhile(tmp(j:j).eq.' ')
        j=j+1
      enddo
      ipos = istrlH(string)
      ito = ipos+1+(15-j)
      if(ito.gt.l) then
         write(*,*)'error, string too short'
         write(*,*) string
         stop
      endif
      string(ipos+1:ito)=tmp(j:)
      end

      function istrlH(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      istrlH= i
      end
CDECK  ID>, STRCATH.
*CMZ :-        -13/02/02  07.20.46  by  Peter Richardson
*-- Author :    Michelangelo Mangano
C----------------------------------------------------------------------
      subroutine strcatH(str1,str2,str)
C----------------------------------------------------------------------
c concatenates str1 and str2 into str. Ignores trailing blanks of str1,str2
C----------------------------------------------------------------------
      character *(*) str1,str2,str
      l1=istrlH(str1)
      l2=istrlH(str2)
      l =len(str)
      if(l.lt.l1+l2) then
        write(*,*) str1,str2
          write(*,*) 'error: l1+l2>l in strcatH'
          write(*,*) 'l1=',l1,' str1=',str1
          write(*,*) 'l2=',l2,' str2=',str2
          write(*,*) 'l=',l
          stop
      endif
      if(l1.ne.0) str(1:l1)=str1(1:l1)
      if(l2.ne.0) str(l1+1:l1+l2)=str2(1:l2)
      if(l1+l2+1.le.l) str(l1+l2+1:l)= ' '
      end
c-------------------------------------------------------------------
      subroutine pdfconvH(nin,nout,type)
c-------------------------------------------------------------------
c converts ALPHA convention for PDF namings to hvqpdf conventions
      implicit none
      integer nin,nout
      character*25 type
      character*25 pdftyp(20,2)
      data pdftyp/
c cteq sets
     $     'CTEQ4M ','CTEQ4L ','CTEQ4HJ',
     $     'CTEQ5M ','CTEQ5L ','CTEQ5HJ',
     $     'CTEQ6M ','CTEQ6L ',12*' ',
C MRST SETS
     $     'MRST99 ',
     $     'MRST01; as=0.119','MRST01; as=0.117','MRST01; as=0.121'
     $     ,'MRST01J; as=0.121','MRST02LO',14*' '/
      integer pdfmap(20,2)
      data pdfmap/
     $   81,83,88,   101,103, 104,   131,133, 12*0,
     $  111,  185,186,187,188,189,   14*0/
c
      nout=pdfmap(mod(nin ,100),1+nin /100)
      type=pdftyp(mod(nin ,100),1+nin /100)
      
      end

C-----------------------------------------------------------------------
C----Calorimeter simulation obtained from Frank Paige 23 March 1988-----
C
C          USE
C
C     CALL CALINI
C     CALL CALSIM
C
C          THEN TO FIND JETS WITH A SIMPLIFIED VERSION OF THE UA1 JET
C          ALGORITHM WITH JET RADIUS RJET AND MINIMUM SCALAR TRANSVERSE
C          ENERGY EJCUT
C            (RJET=1., EJCUT=5. FOR UA1)
C          USE
C
C     CALL GETJET(RJET,EJCUT)
C
C
C-----------------------------------------------------------------------
C 
C          ADDED BY MIKE SEYMOUR: PARTON-LEVEL CALORIMETER. ALL PARTONS
C          ARE CONSIDERED TO BE HADRONS, SO IN FACT RESEM IS IGNORED
C
C     CALL CALPAR
C
C          HARD PARTICLE CALORIMETER. ONLY USES THOSE PARTICLES WHICH
C          CAME FROM THE HARD PROCESS, AND NOT THE UNDERLYING EVENT
C
C     CALL CALHAR
C
C-----------------------------------------------------------------------
      SUBROUTINE CALINI
C                
C          INITIALIZE CALORIMETER FOR CALSIM AND GETJET.  NOTE THAT
C          BECAUSE THE INITIALIZATION IS SEPARATE, CALSIM CAN BE
C          CALLED MORE THAN ONCE TO SIMULATE PILEUP OF SEVERAL EVENTS.
C
      IMPLICIT NONE
      INTEGER NCY,NCPHI,NJMAX,IPHI,IY,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,PI,ET,DELPHI,PHIX,CPHCAL,SPHCAL,DELY,
     &  YX,THX,CTHCAL,STHCAL,PCJET,ETJET
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60,PI=3.141593D0)
      COMMON/CALOR/DELY,DELPHI,ET(NCY,NCPHI),
     $CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN,YCMAX
      PARAMETER (NJMAX=500)
      COMMON/GETCOM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(NCY,NCPHI),NCJET
      LOGICAL FSTCAL
      DATA FSTCAL/.TRUE./
C
C          INITIALIZE ET ARRAY.
      DO 100 IPHI=1,NCPHI
      DO 100 IY=1,NCY
100   ET(IY,IPHI)=0.
C
      IF (FSTCAL) THEN
C          CALCULATE TRIG. FUNCTIONS.
        DELPHI=2.*PI/FLOAT(NCPHI)
        DO 200 IPHI=1,NCPHI
        PHIX=DELPHI*(IPHI-.5)
        CPHCAL(IPHI)=COS(PHIX)
        SPHCAL(IPHI)=SIN(PHIX)
200     CONTINUE
        DELY=(YCMAX-YCMIN)/FLOAT(NCY)
        DO 300 IY=1,NCY
        YX=DELY*(IY-.5)+YCMIN
        THX=2.*ATAN(EXP(-YX))
        CTHCAL(IY)=COS(THX)
        STHCAL(IY)=SIN(THX)
300     CONTINUE
        FSTCAL=.FALSE.
      ENDIF
      END
C
      SUBROUTINE CALSIM
C                
C          SIMPLE CALORIMETER SIMULATION.  ASSUME UNIFORM Y AND PHI
C          BINS
C...HEPEVT commonblock.
      INTEGER NMXHEP,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
      INTEGER NCY,NCPHI,NJMAX,IHEP,ID,IY,IPHI,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,PI,YIP,PSERAP,
     &  PHIIP,DELY,DELPHI,EIP,ET,STHCAL,CTHCAL,CPHCAL,SPHCAL,
     &  PCJET,ETJET
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60,PI=3.141593D0)
      COMMON/CALOR/DELY,DELPHI,ET(NCY,NCPHI),
     $CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN,YCMAX
      PARAMETER (NJMAX=500)
      COMMON/GETCOM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(NCY,NCPHI),NCJET
C
C          FILL CALORIMETER
C
      DO 200 IHEP=1,NHEP
      IF (ISTHEP(IHEP).EQ.1) THEN
        YIP=PSERAP(PHEP(1,IHEP))
        IF(YIP.LT.YCMIN.OR.YIP.GT.YCMAX) GOTO 200
        ID=ABS(IDHEP(IHEP))
C---EXCLUDE TOP QUARK, LEPTONS, PROMPT PHOTONS
        IF ((ID.GE.11.AND.ID.LE.16).OR.ID.EQ.6.OR.ID.EQ.22) GOTO 200
C
        PHIIP=ATAN2(PHEP(2,IHEP),PHEP(1,IHEP))
        IF(PHIIP.LT.0.) PHIIP=PHIIP+2.*PI
        IY=INT((YIP-YCMIN)/DELY)+1
        IPHI=INT(PHIIP/DELPHI)+1
        EIP=PHEP(4,IHEP)
C            WEIGHT BY SIN(THETA)
        ET(IY,IPHI)=ET(IY,IPHI)+EIP*STHCAL(IY)
      ENDIF
  200 CONTINUE
  999 END
      SUBROUTINE GETJET(RJET,EJCUT,ETAJCUT)
C                
C          SIMPLE JET-FINDING ALGORITHM (SIMILAR TO UA1).
C
C     FIND HIGHEST REMAINING CELL > ETSTOP AND SUM SURROUNDING
C          CELLS WITH--
C            DELTA(Y)**2+DELTA(PHI)**2<RJET**2
C            ET>ECCUT.
C          KEEP JETS WITH ET>EJCUT AND ABS(ETA)<ETAJCUT
C          THE UA1 PARAMETERS ARE RJET=1.0 AND EJCUT=5.0
C                  
      IMPLICIT NONE
      INTEGER NCY,NCPHI,NJMAX,IPHI,IY,JETNO,J,K,NCJET,NPHI1,NPHI2,NY1,
     &  NY2,IPASS,IYMX,IPHIMX,ITLIS,IPHI1,IPHIX,IY1,IYX
      DOUBLE PRECISION YCMIN,YCMAX,PI,ETJET,PCJET,RJET,DELPHI,DELY,
     &  ETMAX,ET,ETSTOP,RR,ECCUT,PX,STHCAL,CPHCAL,SPHCAL,CTHCAL,EJCUT
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60,PI=3.141593D0)
      COMMON/CALOR/DELY,DELPHI,ET(NCY,NCPHI),
     &CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN,YCMAX
      PARAMETER (NJMAX=500)
      COMMON/GETCOM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(NCY,NCPHI),NCJET
      DOUBLE PRECISION ETAJCUT,PSERAP
C
C          PARAMETERS
      DATA ECCUT/0.1D0/
      DATA ETSTOP/1.5D0/
      DATA ITLIS/6/
C
C          INITIALIZE
C
      DO 100 IPHI=1,NCPHI
      DO 100 IY=1,NCY
100   JETNO(IY,IPHI)=0
      DO 110 J=1,NJMAX
      ETJET(J)=0.
      DO 110 K=1,4
110   PCJET(K,J)=0.
      NCJET=0
      NPHI1=RJET/DELPHI
      NPHI2=2*NPHI1+1
      NY1=RJET/DELY
      NY2=2*NY1+1
      IPASS=0
C
C          FIND HIGHEST CELL REMAINING
C
1     ETMAX=0.
      DO 200 IPHI=1,NCPHI
      DO 210 IY=1,NCY
      IF(ET(IY,IPHI).LT.ETMAX) GOTO 210
      IF(JETNO(IY,IPHI).NE.0) GOTO 210
      ETMAX=ET(IY,IPHI)
      IYMX=IY
      IPHIMX=IPHI
210   CONTINUE
200   CONTINUE
      IF(ETMAX.LT.ETSTOP) RETURN
C
C          SUM CELLS
C
      IPASS=IPASS+1
      IF(IPASS.GT.NCY*NCPHI) THEN
        WRITE(ITLIS,8888) IPASS
8888    FORMAT(//' ERROR IN GETJET...IPASS > ',I6)
        RETURN
      ENDIF
      NCJET=NCJET+1
      IF(NCJET.GT.NJMAX) THEN
        WRITE(ITLIS,9999) NCJET
9999    FORMAT(//' ERROR IN GETJET...NCJET > ',I5)
        RETURN
      ENDIF
      DO 300 IPHI1=1,NPHI2
      IPHIX=IPHIMX-NPHI1-1+IPHI1
      IF(IPHIX.LE.0) IPHIX=IPHIX+NCPHI
      IF(IPHIX.GT.NCPHI) IPHIX=IPHIX-NCPHI
      DO 310 IY1=1,NY2
      IYX=IYMX-NY1-1+IY1
      IF(IYX.LE.0) GOTO 310
      IF(IYX.GT.NCY) GOTO 310
      IF(JETNO(IYX,IPHIX).NE.0) GOTO 310
      RR=(DELY*(IY1-NY1-1))**2+(DELPHI*(IPHI1-NPHI1-1))**2
      IF(RR.GT.RJET**2) GOTO 310
      IF(ET(IYX,IPHIX).LT.ECCUT) GOTO 310
      PX=ET(IYX,IPHIX)/STHCAL(IYX)
C          ADD CELL TO JET
      PCJET(1,NCJET)=PCJET(1,NCJET)+PX*STHCAL(IYX)*CPHCAL(IPHIX)
      PCJET(2,NCJET)=PCJET(2,NCJET)+PX*STHCAL(IYX)*SPHCAL(IPHIX)
      PCJET(3,NCJET)=PCJET(3,NCJET)+PX*CTHCAL(IYX)
      PCJET(4,NCJET)=PCJET(4,NCJET)+PX
      ETJET(NCJET)=ETJET(NCJET)+ET(IYX,IPHIX)
      JETNO(IYX,IPHIX)=NCJET
310   CONTINUE
300   CONTINUE
C
C          DISCARD JET IF ET < EJCUT.
C
      IF(ETJET(NCJET).GT.EJCUT.AND.ABS(PSERAP(PCJET(1,NCJET))).LT
     $     .ETAJCUT) GOTO 1
      ETJET(NCJET)=0.
      DO 400 K=1,4
400   PCJET(K,NCJET)=0.
      NCJET=NCJET-1
      GOTO 1
      END
C-----------------------------------------------------------------------
      SUBROUTINE CALDEL(ISTLO,ISTHI,ISTOP)
C     LABEL ALL PARTICLES WITH STATUS BETWEEN ISTLO AND ISTHI (UNTIL A
C     PARTICLE WITH STATUS ISTOP IS FOUND) AS FINAL-STATE, CALL CALSIM
C     AND THEN PUT LABELS BACK TO NORMAL
C
C     THIS VERSION LEAVES OUT PARTICLES THAT POINT BACK TO MOTHERS TO BE
C     LEFT OUT OF MATCHING
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER MAXNUP
      PARAMETER(MAXNUP=500)
      INTEGER IEND,INORAD
      COMMON/SHVETO/IEND,INORAD(MAXNUP)
C...HEPEVT commonblock.
      INTEGER NMXHEP,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
      INTEGER ISTOLD(NMXHEP),IHEP,IST,ISTLO,ISTHI,ISTOP,IMO
      LOGICAL FOUND
c      write(3,*) 'new event',nevhep
      FOUND=.FALSE.
      DO 10 IHEP=1,NHEP
        IST=ISTHEP(IHEP)
        ISTOLD(IHEP)=IST
        IF (IST.EQ.ISTOP) FOUND=.TRUE.
        IF (IST.GE.ISTLO.AND.IST.LE.ISTHI.AND..NOT.FOUND) THEN
C     FOUND A RADIATED PARTON, CHECK MOTHER
          IMO=IHEP
 1        IMO=JMOHEP(1,IMO)
          IF(IMO.EQ.IEND) THEN
C     PARENTHOOD OK
            IST=1
c            write(3,*) ihep,ist
            GOTO 9
          ENDIF
          IF(INORAD(IMO).EQ.1) THEN
C     PARTON COMES FROM A VETOED MOTHER
            IST=0
            GOTO 9
          ELSE
C     CHECK GRANDMOTHER
            GOTO 1
          ENDIF
        ELSE
          IST=0
        ENDIF
 9      ISTHEP(IHEP)=IST
 10   CONTINUE
      CALL CALSIM
      DO 20 IHEP=1,NHEP
        ISTHEP(IHEP)=ISTOLD(IHEP)
 20   CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE CALDEL_HVQ(ISTLO,ISTHI,ISTOP)
C     LABEL ALL PARTICLES WITH STATUS BETWEEN ISTLO AND ISTHI (UNTIL A
C     PARTICLE WITH STATUS ISTOP IS FOUND) AS FINAL-STATE, CALL CALSIM
C     AND THEN PUT LABELS BACK TO NORMAL
C
C     THIS VERSION KEEPS ONLY ALL IST=1 PARTICLES REJECTED BY CALDEL AS
C     DAUGHTERS OF VETOED HEAVY-QUARK MOTHERS: JETS COMPLEMENTARY TO
C     THOSE RECONSTRUCTED BY CALDEL
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER MAXNUP
      PARAMETER(MAXNUP=500)
      INTEGER IEND,INORAD
      COMMON/SHVETO/IEND,INORAD(MAXNUP)
C...HEPEVT commonblock.
      INTEGER NMXHEP,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
      INTEGER ISTOLD(NMXHEP),IHEP,IST,ISTLO,ISTHI,ISTOP,IMO
      INTEGER IDMOTH,IDDAUG
      LOGICAL FOUND
      FOUND=.FALSE.
C TAG W-B FROM TOP DECAYS
      DO IHEP=1,NHEP
        IMO=JMOHEP(1,IHEP)
        IDMOTH=ABS(IDHEP(IMO))
        IDDAUG=ABS(IDHEP(IHEP))
        IF(IDMOTH.EQ.6.AND.(IDDAUG.EQ.5.OR.IDDAUG.EQ.24)) INORAD(IHEP)=2
      ENDDO
C
      DO 10 IHEP=1,NHEP
        IST=ISTHEP(IHEP)
        ISTOLD(IHEP)=IST
        IF (IST.EQ.ISTOP) FOUND=.TRUE.
        IF (IST.GE.ISTLO.AND.IST.LE.ISTHI.AND..NOT.FOUND) THEN
C     FOUND A RADIATED PARTON, CHECK MOTHER
          IMO=IHEP
 1        IMO=JMOHEP(1,IMO)
          IF(IMO.EQ.IEND) THEN
C     PARENTHOOD OK, VETO
            IST=0
            GOTO 9
          ENDIF
          IF(INORAD(IMO).EQ.1) THEN
            IDMOTH=ABS(IDHEP(IMO))
            IDDAUG=ABS(IDHEP(IHEP))
C     VERIFY IT'S A HEAVY QUARK -- LEAVE OUT GAUGE BOSON DECAYS
            IF(IDMOTH.GE.4.AND.IDMOTH.LE.6) THEN
C     PARTON COMES FROM A VETOED MOTHER, KEEP UNLESS IT IS THE EVOLVED
C     MOTHER
              IF(IDMOTH.NE.IDDAUG) THEN
                IST=1
                GOTO 9
              ELSE
                IST=0
              ENDIF
            ELSE
C     NOT A HEAVY QUARK MOTHER, LEAVE OUT OF JET RECONSUTRCTION
              IST=0
            ENDIF
          ELSEIF(INORAD(IMO).EQ.2) THEN
C     IT'S A TOP DECAY PRODUCT
            IST=0
          ELSE
C     GO CHECK GRANDMOTHER
            GOTO 1
          ENDIF
        ELSE
          IST=0
        ENDIF
 9      ISTHEP(IHEP)=IST
 10   CONTINUE
      CALL CALSIM
      DO 20 IHEP=1,NHEP
        ISTHEP(IHEP)=ISTOLD(IHEP)
 20   CONTINUE
      END
C-----------------------------------------------------------------------
      FUNCTION PSERAP(P)
C     PSEUDO-RAPIDITY (-LOG TAN THETA/2)
C-----------------------------------------------------------------------
      DOUBLE PRECISION PSERAP,P(3),PT,PL,TINY,THETA
      PARAMETER (TINY=1D-3)
      PT=SQRT(P(1)**2+P(2)**2)+TINY
      PL=P(3)
      THETA=ATAN2(PT,PL)
      PSERAP=-LOG(TAN(0.5*THETA))
      END
C-----------------------------------------------------------------------
C-----------------------------------------------------------------------
      SUBROUTINE ALPSOR(A,N,K,IOPT)
C-----------------------------------------------------------------------
C     Sort A(N) into ascending order
C     IOPT = 1 : return sorted A and index array K
C     IOPT = 2 : return index array K only
C-----------------------------------------------------------------------
      DOUBLE PRECISION A(N),B(5000)
      INTEGER N,I,J,IOPT,K(N),IL(5000),IR(5000)
      IF (N.GT.5000) then
        write(*,*) 'Too many entries to sort in alpsrt, stop'
        stop
      endif
      if(n.le.0) return
      IL(1)=0
      IR(1)=0
      DO 10 I=2,N
      IL(I)=0
      IR(I)=0
      J=1
   2  IF(A(I).GT.A(J)) GOTO 5
   3  IF(IL(J).EQ.0) GOTO 4
      J=IL(J)
      GOTO 2
   4  IR(I)=-J
      IL(J)=I
      GOTO 10
   5  IF(IR(J).LE.0) GOTO 6
      J=IR(J)
      GOTO 2
   6  IR(I)=IR(J)
      IR(J)=I
  10  CONTINUE
      I=1
      J=1
      GOTO 8
  20  J=IL(J)
   8  IF(IL(J).GT.0) GOTO 20
   9  K(I)=J
      B(I)=A(J)
      I=I+1

C Compatibility with gfortran (CMSSW 3_X)
C      IF(IR(J)) 12,30,13
C  13  J=IR(J)
C      GOTO 8
C  12  J=-IR(J)
C      GOTO 9
C  30  IF(IOPT.EQ.2) RETURN

      IF (IR(J).GT.0) THEN
         J=IR(J)
         GOTO 8
      ELSEIF (IR(J).LT.0) then
         J=-IR(J)
         GOTO 9
      ENDIF
      IF (IOPT.EQ.2) RETURN 
      
      DO 31 I=1,N
  31  A(I)=B(I)
 999  END
C-----------------------------------------------------------------------
      subroutine getunit(n)
      implicit none
      integer n,i
      logical yes
      do i=10,100
        inquire(unit=i,opened=yes)
        if(.not.yes) goto 10
      enddo
      write(*,*) 'no free units to write to available, stop'
      stop
 10   n=i
      end

      SUBROUTINE ALSHCD(CSHO)
      CHARACTER*3 CSHO
      CSHO='PYT'
      END

      SUBROUTINE ALSHEN
C----------------------------------------------------------------------
C     Subroutine called when an event is vetoed.
C----------------------------------------------------------------------      
      END

      SUBROUTINE ALSHER(I)
      INTEGER I
      I=0
      RETURN
      END
C----------------------------------------------------------------------
      SUBROUTINE ALSHIN(I)
C----------------------------------------------------------------------
C     subroutine to initialise the events
C----------------------------------------------------------------------
      IMPLICIT NONE
c process code
      integer ihrd
      integer itopprc
c--event file data
      integer Nunit,NunitOut,NunitIni
      character*512 filename,tmpstr
c total number of input events
      common/AHio/Nunit,NunitOut,NunitIni,filename
c inputs for the matching algorithm
      integer iexc,npfst,nplst,nljets,njstart,njlast
     $     ,ickkw
      double precision etclus,rclus,etaclmax
      common/AHopts/etclus,rclus,etaclmax,iexc,npfst
     $     ,nplst,nljets,njstart,njlast,ickkw
c process and particles parameters
      double precision mc,mb,mt,mw,mz,mh
      double precision ebeam
      integer ndns,ih1,ih2       
      integer nw,nz,nh,nph
      integer ihvy,ihvy2
c pdf set type
      character pdftyp*25
c total number of partons
      integer npart
      common/AHppara/mc,mb,mt,mw,mz,mh,
     & ebeam,ih1,ih2,
     & ihrd,itopprc,
     & nw,nz,nh,nph,
     & ihvy,ihvy2,
     & npart,ndns,pdftyp
c weight information
      real *8 maxwgt,avgwgt,errwgt,totlum
      integer unwev
      common/AHwgts/maxwgt,avgwgt,errwgt,totlum,unwev
c general parameters
      integer nparam
      parameter (nparam=200)
      integer parlen,partyp
      character chpar*8,chpdes*70
      double precision parval
      common/AHpars/parval(nparam),chpar(nparam),chpdes(nparam)
     $     ,parlen(nparam),partyp(nparam)
c global event cuts
      double precision ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +     ptcmin,ptcmax,etacmax,drcmin,
     +     ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +     ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      common/AHcuts/ptjmin,ptjmax,etajmax,drjmin,
     +     ptbmin,ptbmax,etabmax,drbmin,
     +        ptcmin,ptcmax,etacmax,drcmin,
     +        ptphmin,etaphmax,drphjmin,drphmin,drphlmin,
     +        ptlmin,etalmax,drlmin,metmin,mllmin,mllmax
      INTEGER I
C--LOCAL VARIABLES
C     EASY-TEXT
      CHARACTER*10 CGIVE
      CHARACTER*12 CGIV2
      CHARACTER*32 CPASS
C LOCAL VARIABLES
      DOUBLE PRECISION MQ
      INTEGER IQ
C COMMUNICATION CODE
      I=661
C FILL IN PYTHIA SPECIFIC INPUTS
      IF(MC.NE.0) THEN
        IQ=4
        MQ=MC
        WRITE(CGIVE,'(I5)') IQ
        WRITE(CGIV2,'(D12.5)') MQ
        CPASS='PMAS('//CGIVE//',1)='//CGIV2
        CALL PYGIVE(CPASS)
C                PMAS(IQ,1)=MQ
      ENDIF 
      IF(MB.NE.0) THEN
        IQ=5
        MQ=MB
        WRITE(CGIVE,'(I5)') IQ
        WRITE(CGIV2,'(D12.5)') MQ
        CPASS='PMAS('//CGIVE//',1)='//CGIV2
        CALL PYGIVE(CPASS)
C                PMAS(IQ,1)=MQ
      ENDIF 
      IF(MT.NE.0) THEN
        IQ=6
        MQ=MT
        WRITE(CGIVE,'(I5)') IQ
        WRITE(CGIV2,'(D12.5)') MQ
        CPASS='PMAS('//CGIVE//',1)='//CGIV2
        CALL PYGIVE(CPASS)
C                PMAS(IQ,1)=MQ
      ENDIF 
      IF(MW.NE.0) THEN
        WRITE(CGIV2,'(D12.5)') MW
        CPASS='PMAS(24,1)='//CGIV2
        CALL PYGIVE(CPASS)
C                PMAS(24,1)=MW
      ENDIF
      IF(MZ.NE.0) THEN
        WRITE(CGIV2,'(D12.5)') MZ
        CPASS='PMAS(23,1)='//CGIV2
        CALL PYGIVE(CPASS)
C                PMAS(23,1)=MZ
      ENDIF
      IF(MH.NE.0) THEN
        WRITE(CGIV2,'(D12.5)') MH
        CPASS='PMAS(25,1)='//CGIV2
        CALL PYGIVE(CPASS)
C                PMAS(25,1)=MH
      ENDIF
      END

C-----------------------------------------------------------------------
      SUBROUTINE ALSFIN
C-----------------------------------------------------------------------
C     PYTHIA END OF FILE TREATMENT
C-----------------------------------------------------------------------
C--   GUP Event common block
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &     IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &     ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &     SPINUP(MAXNUP)
      NUP=0
      END

*---------------------------
      SUBROUTINE ALPGEN_END 
      implicit none
C...Generation and cross section statistics.
      integer       NGENPD,NGEN
      double precision                   XSEC
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
*
      write(33,*)'          '
      write(33,*)'FRACTION OF EVENTS THAT PASS FRAGMENTATION CUTS', 
     +     DBLE(NGEN(0,3))/MAX(1D0,DBLE(NGEN(0,2)))
      write(33,*)'          '
      write(33,*)'CROSS SECTION ', XSEC(0,3),' mb'
      write(33,*)'          '
      write(33,*)'WEIGHT EVENTS ', XSEC(0,3)/MAX(1D0,DBLE(NGEN(0,2)))
      close(33)
      return
      end
