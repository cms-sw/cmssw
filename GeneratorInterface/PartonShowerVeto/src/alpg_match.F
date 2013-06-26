*-- AUTHOR :    MICHELANGELO MANGANO
C----------------------------------------------------------------------
      SUBROUTINE ALVETO(IPVETO)
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
      character*512 filename
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
C     CALL ALSHER(I)
      IF(I.EQ.1) RETURN

c init uninit variables
      jrmin = 0
      etmin = 0.
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
C     NOTICE THAT THIS INFO COMES FROM THE HEPEUP
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
C     NOW THIS INFO COMES FROM THE HEPEVT
      IF(IDBG.EQ.1) THEN
        DO I=1,NHEP
          WRITE(1,111) I,ISTHEP(I),IDHEP(I),JMOHEP(1,I),JMOHEP(2,I)
     $         ,PHEP(1,I),PHEP(2,I),PHEP(3,I)
        ENDDO
 111    FORMAT(5(I4,1X),3(F12.5,1X))
      ENDIF
      
C     PRINT THE INORAD STATUS OF THE PARTICLES.
C     (UNCOMMENT TO MAKE IT WORK)
C     MUST BE LINKED WITH alpg_debug.F IN AlpgenInterface
C      CALL DBINRD

C     DISPLAY PYTHIA EVENT:
C      CALL PYLIST(7)    ! PYTHIA USER PROCESS EVENT DUMP
C      CALL PYLIST(2)    ! PYTHIA FULL EVENT DUMP
C      CALL PYLIST(5)    ! PYTHIA HEPEVT DUMP
C
C     RECONSTRUCT SHOWERED JETS:
      CALL CALINI_ALPGEN
      CALL CALDEL_ALPGEN(NPFST,NPLST,NJLAST)
      CALL GETJET_ALPGEN(RCLUS,ETCLUS,ETACLMAX)
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
        CALL CALINI_ALPGEN
C     RECOSTRUCT POSSIBLE JETS FROM RADIATION OFF THE TOP QUARKS
        CALL CALDEL_ALPGEN_HVQ(NPFST,NPLST,NJLAST)
        CALL GETJET_ALPGEN(RCLUS,ETCLUS,ETACLMAX)
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
      SUBROUTINE ALSETP
C----------------------------------------------------------------------
C     EVENT INITIALITION ROUTINE THAT STARTS OFF FILLED COMMON BLOCKS
C----------------------------------------------------------------------
      IMPLICIT NONE
c process code
      integer ihrd
      integer itopprc
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
C     CALSIM AND JET VARIABLES                             
      INTEGER NCY,NCPHI
      DOUBLE PRECISION YCMIN,YCMAX,PI,ET,DELPHI,CPHCAL,SPHCAL,DELY,
     &     CTHCAL,STHCAL
      PARAMETER (NCY=100)
      PARAMETER (NCPHI=60,PI=3.141593D0)
      COMMON/CALOR/DELY,DELPHI,ET(NCY,NCPHI),
     $     CTHCAL(NCY),STHCAL(NCY),CPHCAL(NCPHI),SPHCAL(NCPHI),YCMIN
     $     ,YCMAX
C     
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER IEND,INORAD
      COMMON/SHVETO/IEND,INORAD(MAXNUP)
c     
C     LOCAL VARIABLES
      INTEGER I,NTMP,IHEPMIN
C     USER ACCESS TO MATCHING PARAMETERS
      INTEGER IUSRMAT
      PARAMETER (IUSRMAT=1)
C      character*512                           CXpar
C      integer         Ixpar 
C      double precision            RXpar
C      common /EXGPAR/ IXpar(100), RXpar(100), CXpar(100)

C     WRITE PARAMETER VALUES
      CALL AHSPAR
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
C       WRITE(*,*) ' '
C       WRITE(*,*) 'INPUT 0 FOR INCLUSIVE JET SAMPLE, 1 FOR EXCLUSIVE'
C       WRITE(*,*) '(SELECT 0 FOR HIGHEST PARTON MULTIPLICITY SAMPLE)' 
C       WRITE(*,*) '(SELECT 1 OTHERWISE)'
C       READ(*,*) IEXC
C       IEXC = IXpar(2)
        WRITE(*,*) ' IEXC = ',IEXC
        IF(NLJETS.GT.0) THEN
          WRITE(*,*) 'INPUT ET(CLUS), R(CLUS)'
C         ETCLUS=MAX(PTJMIN+5,1.2*PTJMIN)
C         RCLUS=DRJMIN
C         WRITE(*,*) '(SUGGESTED VALUES:',ETCLUS,RCLUS,')'
C         READ(*,*) ETCLUS,RCLUS,ETACLMAX
C         ETCLUS = RXpar(1)
C         RCLUS  = RXpar(2)
          ETACLMAX = ETAJMAX
        ELSEIF(NLJETS.EQ.0) THEN
          WRITE(*,*) 'INPUT ET(CLUS), R(CLUS)'
          WRITE(*,*) '(MUST MATCH VALUES USED IN PROCESSING',
     +         ' OF NJET>0 EVENTS; THESE DEFAULT TO:'
          WRITE(*,*)
     +         'ETCLUS=MAX(PTJMIN+5,1.2*PTJMIN) RCLUS=DRJMIN, ',
     +         ' ETACLMAX=ETAJMAX)'
C         READ(*,*) ETCLUS,RCLUS,ETACLMAX
C         ETCLUS = RXpar(1)
C         RCLUS  = RXpar(2)
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
      character*512 filename
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
      SUBROUTINE CALINI_ALPGEN
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
      SUBROUTINE CALSIM_ALPGEN
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
      END
      SUBROUTINE GETJET_ALPGEN(RJET,EJCUT,ETAJCUT)
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

c fix complains on uninit variables
      iymx = 0
      iphimx = 0
c
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
8888    FORMAT(//' ERROR IN GETJET_ALPGEN...IPASS > ',I6)
        RETURN
      ENDIF
      NCJET=NCJET+1
      IF(NCJET.GT.NJMAX) THEN
        WRITE(ITLIS,9999) NCJET
9999    FORMAT(//' ERROR IN GETJET_ALPGEN...NCJET > ',I5)
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
      SUBROUTINE CALDEL_ALPGEN(ISTLO,ISTHI,ISTOP)
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

C     PRINT PARTICLE STATUS FOR THE CALORIMETER SIMULATION.
C     (UNCOMMENT TO MAKE IT WORK)
C     MUST BE LINKED WITH alpg_debug.F IN AlpgenInterface
C      CALL DBCAL

      CALL CALSIM_ALPGEN
      DO 20 IHEP=1,NHEP
        ISTHEP(IHEP)=ISTOLD(IHEP)
 20   CONTINUE
      END
C-----------------------------------------------------------------------
      SUBROUTINE CALDEL_ALPGEN_HVQ(ISTLO,ISTHI,ISTOP)
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
      CALL CALSIM_ALPGEN
      DO 20 IHEP=1,NHEP
        ISTHEP(IHEP)=ISTOLD(IHEP)
 20   CONTINUE
      END
c$$$C-----------------------------------------------------------------------
c$$$      FUNCTION PSERAP(P)
c$$$C     PSEUDO-RAPIDITY (-LOG TAN THETA/2)
c$$$C-----------------------------------------------------------------------
c$$$      DOUBLE PRECISION PSERAP,P(3),PT,PL,TINY,THETA
c$$$      PARAMETER (TINY=1D-3)
c$$$      PT=SQRT(P(1)**2+P(2)**2)+TINY
c$$$      PL=P(3)
c$$$      THETA=ATAN2(PT,PL)
c$$$      PSERAP=-LOG(TAN(0.5*THETA))
c$$$      END
c$$$C-----------------------------------------------------------------------
c$$$C-----------------------------------------------------------------------
c$$$      SUBROUTINE ALPSOR(A,N,K,IOPT)
c$$$C-----------------------------------------------------------------------
c$$$C     Sort A(N) into ascending order
c$$$C     IOPT = 1 : return sorted A and index array K
c$$$C     IOPT = 2 : return index array K only
c$$$C-----------------------------------------------------------------------
c$$$      DOUBLE PRECISION A(N),B(5000)
c$$$      INTEGER N,I,J,IOPT,K(N),IL(5000),IR(5000)
c$$$      IF (N.GT.5000) then
c$$$        write(*,*) 'Too many entries to sort in alpsrt, stop'
c$$$        stop
c$$$      endif
c$$$      if(n.le.0) return
c$$$      IL(1)=0
c$$$      IR(1)=0
c$$$      DO 10 I=2,N
c$$$      IL(I)=0
c$$$      IR(I)=0
c$$$      J=1
c$$$   2  IF(A(I).GT.A(J)) GOTO 5
c$$$   3  IF(IL(J).EQ.0) GOTO 4
c$$$      J=IL(J)
c$$$      GOTO 2
c$$$   4  IR(I)=-J
c$$$      IL(J)=I
c$$$      GOTO 10
c$$$   5  IF(IR(J).LE.0) GOTO 6
c$$$      J=IR(J)
c$$$      GOTO 2
c$$$   6  IR(I)=IR(J)
c$$$      IR(J)=I
c$$$  10  CONTINUE
c$$$      I=1
c$$$      J=1
c$$$      GOTO 8
c$$$  20  J=IL(J)
c$$$   8  IF(IL(J).GT.0) GOTO 20
c$$$   9  K(I)=J
c$$$      B(I)=A(J)
c$$$      I=I+1
c$$$
c$$$C Compatibility with gfortran (CMSSW 3_X)
c$$$C      IF(IR(J)) 12,30,13
c$$$C  13  J=IR(J)
c$$$C      GOTO 8
c$$$C  12  J=-IR(J)
c$$$C      GOTO 9
c$$$C  30  IF(IOPT.EQ.2) RETURN
c$$$
c$$$      IF (IR(J).GT.0) THEN
c$$$         J=IR(J)
c$$$         GOTO 8
c$$$      ELSEIF (IR(J).LT.0) then
c$$$         J=-IR(J)
c$$$         GOTO 9
c$$$      ENDIF
c$$$      IF (IOPT.EQ.2) RETURN 
c$$$      
c$$$      DO 31 I=1,N
c$$$  31  A(I)=B(I)
c$$$ 999  END

C-----------------------------------------------------------------------
      subroutine getunit(n)
      implicit none
      integer n,i
      logical yes
      WRITE(*,*)'CALLED GETUNIT'
      do i=10,100
        inquire(unit=i,opened=yes)
        if(.not.yes) goto 10
      enddo
      write(*,*) 'no free units to write to available, stop'
      stop
 10   n=i
      end
