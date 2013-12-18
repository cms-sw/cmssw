C*******************************************************************
C*          MadEvent - Pythia interface.                           *
C*                                                                 *
C*                                                                 *
C*   Adapted version from ME2pythia 1.66 (J.Alwall)                *
C*                                                                 *
C*       Simon de Visscher August 2010   sdevissc@cern.ch          *
C*        - Complete upgrade to 1.66                               *
C*        - Addition of showerkt                                   *
C*        - Addition of resonance exclusion for BSM matching       *
C*                                                                 *
C*       Christophe Saout 2009                                     *
C*        - Improvement of JetMatching interface                   *
C*                                                                 *
C*       Dorian Kcira 2008.02.05                                         *
C*        -First implementation of KtMLM in CMSSW                  *
C*       - Improvement of matching routines                        *
C*                                                                 *
C*                                                                 *
C*******************************************************************


C*********************************************************************
C...UPINIT
C...Routine called by PYINIT to set up user-defined processes.
C*********************************************************************      
      SUBROUTINE MGINIT(npara,param,value)

      
      IMPLICIT NONE

      integer npara
      character*20 param(*),value(*)


c      CHARACTER*132 CHAR_READ

C...Pythia parameters.
      INTEGER MSTP,MSTI,MRPY
      DOUBLE PRECISION PARP,PARI,RRPY
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYDATR/MRPY(6),RRPY(100)

C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &   IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &   LPRUP(MAXPUP)

C...Extra commonblock to transfer run info.
      INTEGER LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      COMMON/UPPRIV/LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      DATA LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE/77,6,1,0,0,1/
      SAVE /UPPRIV/

C...Inputs for the matching algorithm
      double precision etcjet,rclmax,etaclmax,qcut,clfact,showerkt
      integer maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres(30)
      integer nqmatch,nexcproc,iexcproc(MAXPUP),iexcval(MAXPUP)
      logical nosingrad,jetprocs
      common/MEMAIN/etcjet,rclmax,etaclmax,qcut,showerkt,clfact,
     $   maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres,
     $   nqmatch,nexcproc,iexcproc,iexcval,nosingrad,jetprocs


C...Parameter arrays (local)
C      integer maxpara
C      parameter (maxpara=1000)
C      integer npara,iseed
C      character*20 param(maxpara),value(maxpara)      

C...Lines to read in assumed never longer than 200 characters. 
C      INTEGER MAXLEN,IBEG,IPR,I
C      PARAMETER (MAXLEN=200)
C      CHARACTER*(MAXLEN) STRING

C...Functions
C      INTEGER iexclusive
C      EXTERNAL iexclusive

C...Format for reading lines.
C      CHARACTER*6 STRFMT
C      STRFMT='(A000)'
C      WRITE(STRFMT(3:5),'(I3)') MAXLEN

C...Extract the model parameter card and read it.
C      CALL MODELPAR(LNHIN)

c...Read the <init> block information

C...Loop until finds line beginning with "<init>" or "<init ". 
C  100 READ(LNHIN,STRFMT,END=130,ERR=130) STRING
C...Pick out random number seed and use for PYR initialization
C      IF(INDEX(STRING,'iseed').NE.0)THEN
C         READ(STRING,*) iseed
C         IF(iseed.gt.0) THEN
C            WRITE(LNHOUT,*) 'Initializing PYR with random seed ',iseed
c            MRPY(1) = iseed
C            MRPY(2) = 0
C         ENDIF
C      ENDIF
C      IBEG=0
C  110 IBEG=IBEG+1
C...Allow indentation.
C      IF(STRING(IBEG:IBEG).EQ.' '.AND.IBEG.LT.MAXLEN-5) GOTO 110 
C      IF(STRING(IBEG:IBEG+5).NE.'<init>'.AND.
C     &STRING(IBEG:IBEG+5).NE.'<init ') GOTO 100

C...Read first line of initialization info.
C      READ(LNHIN,*,END=130,ERR=130) IDBMUP(1),IDBMUP(2),EBMUP(1),
C     &EBMUP(2),PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),IDWTUP,NPRUP

C...Read NPRUP subsequent lines with information on each process.
C      DO 120 IPR=1,NPRUP
C        READ(LNHIN,*,END=130,ERR=130) XSECUP(IPR),XERRUP(IPR),
C     &  XMAXUP(IPR),LPRUP(IPR)
C  120 CONTINUE

C...Set PDFLIB or LHAPDF pdf number for Pythia

C      IF(PDFSUP(1).NE.19070.AND.(PDFSUP(1).NE.0.OR.PDFSUP(2).NE.0))THEN
c     Not CTEQ5L, which is standard in Pythia
C         CALL PYGIVE('MSTP(52)=2')
c     The following works for both PDFLIB and LHAPDF (where PDFGUP(1)=0)
c     But note that the MadEvent output uses the LHAPDF numbering scheme
C        IF(PDFSUP(1).NE.0)THEN
C           MSTP(51)=1000*PDFGUP(1)+PDFSUP(1)
C        ELSE
C           MSTP(51)=1000*PDFGUP(2)+PDFSUP(2)
C        ENDIF
C      ENDIF

C...Initialize widths and partial widths for resonances.
C      CALL PYINRE
        
C...Calculate xsec reduction due to non-decayed resonances
C...based on first event only!

C      CALL BRSUPP

C      REWIND(LNHIN)

C...Extract cuts and matching parameters
C      CALL read_params(LNHIN,npara,param,value,maxpara)

C      call get_integer(npara,param,value," ickkw ",ickkw,0)
C      if(ickkw.eq.1)then
C         call get_integer(npara,param,value," ktscheme ",mektsc,1)
C         write(*,*)'Running matching with ME ktscheme ',mektsc
C      endif
C
C...Set kt clustering scheme (if not already set)
C
      integer i
      call initpydata
      write(*,*)"MGINIT: ickkw is ",ickkw
      write(*,*)"MGINIT: ktscheme is ",mektsc
      write(*,*)"MGINIT: QCut is ",qcut
      write(*,*)"MGINIT: Showerkt is ",showerkt
	  do 10 i = 1, nexcres
         write(*,*) 'EXCRES(', i,')=',EXCRES(i)
  10  continue

      IF(ABS(IDBMUP(1)).EQ.11.AND.ABS(IDBMUP(2)).EQ.11.AND.
     $     IDBMUP(1).EQ.-IDBMUP(2).AND.ktsche.EQ.0)THEN
         ktsche=1
      ELSE IF(ktsche.EQ.0) THEN
         ktsche=4313
      ENDIF

C...Enhance primordial kt
c      CALL PYGIVE('PARP(91)=2.5')
c      CALL PYGIVE('PARP(93)=15')

      IF(ickkw.gt.0) CALL set_matching(npara,param,value)
 
C...For photon initial states from protons: Set proton not to break up
      CALL PYGIVE('MSTP(98)=1')

  
C      IF(ickkw.gt.0.and.(NPRUP.gt.1.or.iexclusive(LPRUP(1)).ne.-1))
C     $     CALL set_matching(LNHIN,npara,param,value)

C...For photon initial states from protons: Set proton not to break up
C      CALL PYGIVE('MSTP(98)=1')

C...Reset event numbering
C      IEVNT=0

      RETURN

C...Error exit: give up if initalization does not work.
C  130 WRITE(*,*) ' Failed to read LHEF initialization information.'
C      WRITE(*,*) ' Event generation will be stopped.'
C      STOP  
      END

C*********************************************************************      
C...UPEVNT
C...Routine called by PYEVNT or PYEVNW to get user process event
C*********************************************************************
      SUBROUTINE MGEVNT

      IMPLICIT NONE

C...Pythia parameters.
      INTEGER MSTP,MSTI
      DOUBLE PRECISION PARP,PARI
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)

C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &   IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &   LPRUP(MAXPUP)
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &   ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &   VTIMUP(MAXNUP),SPINUP(MAXNUP)
C...Pythia common blocks
      INTEGER PYCOMP,KCHG,MINT,NPART,NPARTD,IPART,MAXNUR
      DOUBLE PRECISION PMAS,PARF,VCKM,VINT,PTPART
C...Particle properties + some flavour parameters.
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYINT1/MINT(400),VINT(400)
      PARAMETER (MAXNUR=1000)
      COMMON/PYPART/NPART,NPARTD,IPART(MAXNUR),PTPART(MAXNUR)

C...Extra commonblock to transfer run info.
      INTEGER LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      COMMON/UPPRIV/LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE

C...Inputs for the matching algorithm
      double precision etcjet,rclmax,etaclmax,qcut,clfact,showerkt
      integer maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres(30)
      integer nqmatch,nexcproc,iexcproc(MAXPUP),iexcval(MAXPUP)
      logical nosingrad,jetprocs
      common/MEMAIN/etcjet,rclmax,etaclmax,qcut,showerkt,clfact,
     $   maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres,
     $   nqmatch,nexcproc,iexcproc,iexcval,nosingrad,jetprocs

C...Commonblock to transfer event-by-event matching info
      INTEGER NLJETS,IEXC,Ifile
      DOUBLE PRECISION PTCLUS
      COMMON/MEMAEV/PTCLUS(20),NLJETS,IEXC,Ifile

C...Local variables
      INTEGER I,NEX,KP(MAXNUP),MOTH,NUPREAD,II,iexcl
      DOUBLE PRECISION PSUM,ESUM
C...Lines to read in assumed never longer than 200 characters. 
      INTEGER MAXLEN
      PARAMETER (MAXLEN=200)

C...Functions
      INTEGER iexclusive
      EXTERNAL iexclusive
C...Format for reading lines.
C      CHARACTER*6 STRFMT
C      CHARACTER*1 CDUM

C      STRFMT='(A000)'
C      WRITE(STRFMT(3:5),'(I3)') MAXLEN

C...Loop until finds line beginning with "<event>" or "<event ". 
C  100 READ(LNHIN,STRFMT,END=900,ERR=900) STRING
C      IBEG=0
C  110 IBEG=IBEG+1
C...Allow indentation.
C      IF(STRING(IBEG:IBEG).EQ.' '.AND.IBEG.LT.MAXLEN-6) GOTO 110 
C      IF(STRING(IBEG:IBEG+6).NE.'<event>'.AND.
C     &STRING(IBEG:IBEG+6).NE.'<event ') GOTO 100

C...Read first line of event info.
C      READ(LNHIN,*,END=900,ERR=900) NUPREAD,IDPRUP,XWGTUP,SCALUP,
C     &AQEDUP,AQCDUP

      NUPREAD=NUP

C...Read NUP subsequent lines with information on each particle.
      ESUM=0d0
      PSUM=0d0
      NEX=2
      NUP=1
C      write(*,*)'SCALUP=',SCALUP
      DO 120 I=1,NUPREAD
C      write(*,*)IDUP(NUP),' ',ISTUP(NUP),' ',MOTHUP(1,NUP)
C     &  ,' ',MOTHUP(2,NUP),' ',ICOLUP(1,NUP),' ',ICOLUP(2,NUP)
C     &  ,' ',(PUP(J,NUP),J=1,5),' ',VTIMUP(NUP),' ',SPINUP(NUP)
c        READ(LNHIN,*,END=900,ERR=900) IDUP(NUP),ISTUP(NUP),
c     &  MOTHUP(1,NUP),MOTHUP(2,NUP),ICOLUP(1,NUP),ICOLUP(2,NUP),
c     &  (PUP(J,NUP),J=1,5),VTIMUP(NUP),SPINUP(NUP)
C...Reset resonance momentum to prepare for mass shifts
        IF(ISTUP(NUP).EQ.2) PUP(3,NUP)=0
        IF(ISTUP(NUP).EQ.1)THEN
           NEX=NEX+1
C...Mrenna:  only if 4 < pdgId < 21
           IF(PUP(5,NUP).EQ.0D0.AND.IABS(IDUP(NUP)).GT.3
     $         .AND.IDUP(NUP).LT.21) THEN
C...Set massless particle masses to Pythia default. Adjust z-momentum. 
              PUP(5,NUP)=PMAS(IABS(PYCOMP(IDUP(NUP))),1)
              PUP(3,NUP)=SIGN(SQRT(MAX(0d0,PUP(4,NUP)**2-PUP(5,NUP)**2-
     $           PUP(1,NUP)**2-PUP(2,NUP)**2)),PUP(3,NUP))
           ENDIF
           PSUM=PSUM+PUP(3,NUP)
C...Set mother resonance momenta
           MOTH=MOTHUP(1,NUP)
           DO WHILE (MOTH.GT.2)
             PUP(3,MOTH)=PUP(3,MOTH)+PUP(3,NUP)
             MOTH=MOTHUP(1,MOTH)
           ENDDO
        ENDIF
        NUP=NUP+1
  120 CONTINUE
      NUP=NUP-1

C...Increment event number
C      IEVNT=IEVNT+1

C..Adjust mass of resonances
      DO I=1,NUP
         IF(ISTUP(I).EQ.2)THEN
            PUP(5,I)=SQRT(PUP(4,I)**2-PUP(1,I)**2-PUP(2,I)**2-
     $             PUP(3,I)**2)
         ENDIF
      ENDDO

C...Adjust energy and momentum of incoming particles
C...In massive case need to solve quadratic equation
c      PM1=PUP(5,1)**2
c      PM2=PUP(5,2)**2
c      A1=4d0*(ESUM**2-PSUM**2)
c      A2=ESUM**2-PSUM**2+PM2-PM1
c      A3=2d0*PSUM*A2
c      A4=A3/A1
c      A5=(A2**2-4d0*ESUM**2*PM2)/A1
c
c      PUP(3,2)=A4+SIGN(SQRT(A4**2+A5),PUP(3,2))
c      PUP(3,1)=PSUM-PUP(3,2)
c      PUP(4,1)=SQRT(PUP(3,1)**2+PM1)
c      PUP(4,2)=SQRT(PUP(3,2)**2+PM2)

      ESUM=PUP(4,1)+PUP(4,2)

C...Assuming massless incoming particles - otherwise Pythia adjusts
C...the momenta to make them massless
C     IF(IDBMUP(1).GT.100.AND.IDBMUP(2).GT.100)THEN
C       DO I=1,2
C          PUP(3,I)=0.5d0*(PSUM+SIGN(ESUM,PUP(3,I)))
C          PUP(5,I)=0d0
C        ENDDO
C        PUP(4,1)=ABS(PUP(3,1))
C        PUP(4,2)=ESUM-PUP(4,1)
C      ENDIF
        
C...If you want to use some other scale for parton showering then the 
C...factorisation scale given by MadEvent, please implement the function PYMASC
C...(example function included below) 

      IF(ickkw.eq.0.AND.MSCAL.GT.0) CALL PYMASC(SCALUP)
c      IF(MINT(35).eq.3.AND.ickkw.EQ.1) SCALUP=SQRT(PARP(67))*SCALUP
      
C...Read FSR scale for all FS particles (as comment in event file)
C      IF(ickkw.eq.1)THEN
C        READ(LNHIN,*,END=900,ERR=130) CDUM,(PTPART(I),I=1,NEX)
C 130    CONTINUE
C      ENDIF

      IF(ickkw.gt.0) THEN
c
c   Set up number of jets
c
C         write(*,*)'Setting up the number of jets'
         NLJETS=0
         NPART=0
C         write(*,*)'Cycling on 3->NUP'
         do i=3,NUP
C            write(*,*)'Iteration: i=',i
            if(ISTUP(i).ne.1) cycle
C            write(*,*)'Npart++'
            NPART=NPART+1
            IPART(NPART)=i
            if(iabs(IDUP(i)).gt.nqmatch.and.IDUP(i).ne.21) cycle
            if(MOTHUP(1,i).gt.2) cycle
C     Remove final-state partons that combine to color singlets
            IF((ABS(IDBMUP(1)).NE.11.OR.IDBMUP(1).NE.-IDBMUP(2)).AND.
     $           nosingrad) THEN
               DO II=3,NUP
                  IF(II.NE.i.AND.ISTUP(II).EQ.1)THEN
                     IF((IDUP(II).EQ.-IDUP(i).OR.
     $                    IDUP(i).EQ.21.AND.IDUP(II).EQ.21).AND.
     $                    ICOLUP(1,II).EQ.ICOLUP(2,i).AND.
     $                    ICOLUP(2,II).EQ.ICOLUP(1,i))then
c                        print *,'Found color singlet'
                        CALL PYLIST(7)
                        GOTO 140
                     endif
                  ENDIF
               ENDDO
            ENDIF
            NLJETS=NLJETS+1
C            WRITE(*,*) ' NLJETS=',NLJETS
            PTCLUS(NLJETS)=PTPART(NPART)
c            print *,'   Adding a jet and NLJETS=',NLJETS,' and
c     $        PTCLUS(',NLJETS,')=',PTCLUS(NLJETS)
 140        continue
         enddo
         CALL ALPSOR(PTCLUS,nljets,KP,1)
      
         if(jetprocs) IDPRUP=LPRUP(NLJETS-MINJETS+1)

         IF(ickkw.eq.1) THEN
c   ... and decide whether exclusive or inclusive
            iexcl=iexclusive(IDPRUP)
            if((IEXCFILE.EQ.0.and.NLJETS.eq.MAXJETS.or.
     $           iexcl.eq.0).and.
     $           iexcl.ne.1)then
               IEXC=0
            else if(iexcl.eq.-1)then
               IEXC=-1
            else
               IEXC=1
            endif
         ENDIF
      ENDIF
c      write( *,*)'finishing MGEVNT'
      RETURN

C...Error exit, typically when no more events.
C  900 WRITE(*,*) ' Failed to read LHEF event information,'
C      WRITE(*,*) ' assume end of file has been reached.'
C      NUP=0
C      MINT(51)=2
C      write( *,*)'finishing MGEVNT'
C      RETURN
      END

C*********************************************************************
C...UPVETO
C...Subroutine to implement the MLM jet matching criterion
C*********************************************************************
      SUBROUTINE MGVETO(IPVETO)

      IMPLICIT NONE


     


C...Pythia common blocks
      INTEGER MINT
      DOUBLE PRECISION VINT
      COMMON/PYINT1/MINT(400),VINT(400)
      INTEGER MSTP,MSTI
      DOUBLE PRECISION PARP,PARI
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)

C...GUP Event common block
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &              ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &              SPINUP(MAXNUP)
C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &   IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &   LPRUP(MAXPUP)
C...HEPEVT commonblock.
      INTEGER NMXHEP,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
      INTEGER IPVETO
C...GETJET commonblocks
      INTEGER MNCY,MNCPHI,NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,DELY,DELPHI,ET,STHCAL,CTHCAL,CPHCAL,
     &  SPHCAL,PCJET,ETJET
      PARAMETER (MNCY=200)
      PARAMETER (MNCPHI=200)
      COMMON/CALORM/DELY,DELPHI,ET(MNCY,MNCPHI),
     $CTHCAL(MNCY),STHCAL(MNCY),CPHCAL(MNCPHI),SPHCAL(MNCPHI),
     $YCMIN,YCMAX,NCY,NCPHI
      PARAMETER (NJMAX=500)
      COMMON/GETCOMM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(MNCY,MNCPHI),
     $NCJET
      DOUBLE PRECISION PI
      PARAMETER (PI=3.141593D0)
C     
      DOUBLE PRECISION PSERAP
      INTEGER K(NJMAX),KP(NJMAX),kpj(njmax)

C...Variables for the kT-clustering
      INTEGER NMAX,NN,NSUB,JET,NJETM,IHARD,IP1,IP2
      DOUBLE PRECISION PP,PJET
      DOUBLE PRECISION ECUT,Y,YCUT
      PARAMETER (NMAX=512)
      DIMENSION JET(NMAX),Y(NMAX),PP(4,NMAX),PJET(4,NMAX)
      INTEGER NNM
      DOUBLE PRECISION YM(NMAX),PPM(4,NMAX)

C...kt clustering common block
      INTEGER NMAXKT,NUM,HIST
      PARAMETER (NMAXKT=512)
      DOUBLE PRECISION PPP,KT,ETOT,RSQ,KTP,KTS,KTLAST
      COMMON /KTCOMM/ETOT,RSQ,PPP(9,NMAXKT),KTP(NMAXKT,NMAXKT),
     $   KTS(NMAXKT),KT(NMAXKT),KTLAST(NMAXKT),HIST(NMAXKT),NUM

C...Extra commonblock to transfer run info.
      INTEGER LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      COMMON/UPPRIV/LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE

C...Inputs for the matching algorithm
C   clfact determines how jet-to parton matching is done
C   kt-jets: default=1
C    clfact >= 0: Max mult. if within clfact*max(qcut,Q(partNmax)) from jet, others within clfact*qcut
C    clfact < 0: Max mult. if within |clfact|*Q(jetNmax) from jet, other within |clfact|*qcut
C   cone-jets: default=1.5
C    Matching if within clfact*RCLMAX 

C...Inputs for the matching algorithm
      double precision etcjet,rclmax,etaclmax,qcut,clfact,showerkt
      integer maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres(30)
      integer nqmatch,nexcproc,iexcproc(MAXPUP),iexcval(MAXPUP)
      logical nosingrad,jetprocs
      common/MEMAIN/etcjet,rclmax,etaclmax,qcut,showerkt,clfact,
     $   maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres,
     $   nqmatch,nexcproc,iexcproc,iexcval,nosingrad,jetprocs

C...Commonblock to transfer event-by-event matching info
      INTEGER NLJETS,IEXC,Ifile
      DOUBLE PRECISION PTCLUS
      COMMON/MEMAEV/PTCLUS(20),NLJETS,IEXC,Ifile

      INTEGER nvarev,nvar2
      PARAMETER (nvarev=57,nvar2=6)

      REAL*4 varev(nvarev)
      COMMON/HISTDAT/varev
	  
C...Pythia common blocks
      INTEGER NPART,NPARTD,IPART,MAXNUR
	  DOUBLE PRECISION PTPART
      PARAMETER (MAXNUR=1000)
      COMMON/PYPART/NPART,NPARTD,IPART(MAXNUR),PTPART(MAXNUR)

	  INTEGER flag
	  COMMON/OUTTREE/flag
	  
	  CHARACTER*8 htit(nvarev),htit2(nvar2)
      DATA htit/'Npart','Qjet1','Qjet2','Qjet3','Qjet4',
     $   'Ptcjet1','Ptcjet2','Ptcjet3','Ptcjet4',
     $   'Etacjet1','Etacjet2','Etacjet3','Etacjet4',
     $   'Phicjet1','Phicjet2','Phicjet3','Phicjet4',
     $   'Ptjet1','Ptjet2','Ptjet3','Ptjet4',
     $   'Etajet1','Etajet2','Etajet3','Etajet4',
     $   'Phijet1','Phijet2','Phijet3','Phijet4',
     $   'Idres1','Ptres1','Etares1','Phires1',
     $   'Idres2','Ptres2','Etares2','Phires2',
     $   'Ptlep1','Etmiss','Htjets',
     $   'Ptb','Etab','Ptbbar','Etabbar','Ptbj','Etabj',
     $   'Qpar1','Qpar2','Qpar3','Qpar4',
     $   'Ptpar1','Ptpar2','Ptpar3','Ptpar4',
     $   'Ncjets','Njets','Nfile'/
      DATA htit2/'Npart','Qjet1','Qjet2','Qjet3','Qjet4','Nfile'/
	  




C   local variables
      double precision tiny
      parameter (tiny=1d-3)
      integer idbg
      data idbg/0/

      integer i,j,ihep,nmatch,jrmin,KPT(MAXNUP),nres,ii
      double precision etajet,phijet,delr,dphi,delrmin,ptjet
      double precision p(4,10),pt(10),eta(10),phi(10)
      INTEGER IMO
      logical norad(20)
      REAL*4 var2(nvar2)
      
c      if(NLJETS.GT.0)then
c        idbg=1
c      else
c        idbg=0
c      endif


c      write(*,*)'Entering MGVETO'
c      write(*,*)'qcut is ',qcut,' and showerkt is ',showerkt
      IPVETO=0
c     Return if not MLM matching (or non-matched subprocess)
      
      IF(ICKKW.LE.0.OR.IEXC.eq.-1) RETURN

      IF(NLJETS.LT.MINJETS.OR.NLJETS.GT.MAXJETS)THEN
        if(idbg.eq.1)
     $     WRITE(LNHOUT,*) 'Failed due to NLJETS ',NLJETS,' < ',MINJETS,
     $        ' or > ',MAXJETS
         GOTO 999
      ENDIF

C      write(*,*)'Throw event if it contains an excluded resonance'
      NRES=0
      DO I=1,NUP
c	     write(*,*)'cycling on particles'
        IF(ISTUP(I).EQ.2)THEN
c			write(*,*)'found S2, now comparin with the ',nexcres,' excres'
           DO J=1,nexcres
c			write(*,*)'comparing ',IDUP(I),' and ',EXCRES(J)
              IF(IDUP(I).EQ.EXCRES(J)) NRES=NRES+1
           ENDDO
        ENDIF
      ENDDO
      IF(NRES.GT.0)THEN
c		write(*,*)'Event',IEVNT,
c     &  ' thrown because of ',NRES,'e r'
c     CALL PYLIST(7)
         GOTO 999
      ENDIF

c init uninit variables
      jrmin = 0

C   Set up vetoed mothers
c      DO I=1,MAXNUP
c        INORAD(I)=0
c      ENDDO
c      DO IHEP=1,NUP-2      
c        if(ISTHEP(ihep).gt.1.and.iabs(IDHEP(ihep)).gt.8) then
c        if(iabs(IDHEP(ihep)).gt.5.and.IDHEP(ihep).ne.21) then
c          INORAD(ihep)=1
c        endif
c      ENDDO

c
c     reconstruct parton-level event
c     Set norad for daughters of decayed particles, to not include
c     radiation from these in matched jets
c
      if(idbg.eq.1) then
        write(LNHOUT,*) ' '
        write(LNHOUT,*) 'new event '
c        CALL PYLIST(1)
        CALL PYLIST(7)
        CALL PYLIST(5)
        write(LNHOUT,*) 'PARTONS'
      endif
      i=0
      do ihep=3,nup
         NORAD(ihep)=.false.
        if((ABS(IDBMUP(1)).NE.11.OR.IDBMUP(1).NE.-IDBMUP(2)).AND.
     $        MOTHUP(1,ihep).gt.2) goto 100
        if(ISTUP(ihep).ne.1.or.
     $     (iabs(IDUP(ihep)).gt.nqmatch.and.IDUP(ihep).ne.21)) cycle
c     If quark or gluon making singlet system with other final-state parton
c     remove (since unseen singlet resonance) unless e+e- collision
        IF((ABS(IDBMUP(1)).NE.11.OR.IDBMUP(1).NE.-IDBMUP(2)).AND.
     $       nosingrad)THEN
           DO II=3,NUP
              IF(II.NE.ihep.AND.ISTUP(II).EQ.1)THEN
                 IF((IDUP(II).EQ.-IDUP(ihep).OR.
     $                IDUP(ihep).EQ.21.AND.IDUP(II).EQ.21).AND.
     $                ICOLUP(1,II).EQ.ICOLUP(2,ihep).AND.
     $                ICOLUP(2,II).EQ.ICOLUP(1,ihep))
     $                GOTO 100
              ENDIF
           ENDDO
        ENDIF
        i=i+1
        do j=1,4
          p(j,i)=pup(j,ihep)
        enddo
        pt(i)=sqrt(p(1,i)**2+p(2,i)**2)
        if(i.LE.4) varev(50+i)=pt(i)
        eta(i)=-log(tan(0.5d0*atan2(pt(i)+tiny,p(3,i))))
        phi(i)=atan2(p(2,i),p(1,i))
        if(idbg.eq.1) then
          write(LNHOUT,*) pt(i),eta(i),phi(i)
        endif
        cycle
 100    norad(ihep)=.true.
      enddo
      if(i.ne.NLJETS)then
        print *,'Error in UPVETO: Wrong number of jets found ',i,NLJETS
        CALL PYLIST(7)
        CALL PYLIST(2)
        stop
      endif
C Bubble-sort PTs in descending order
      DO I=1,3
         DO J=4,I+1,-1
            IF(varev(50+J).GT.varev(50+I))THEN
               PTJET=varev(50+J)
               varev(50+J)=varev(50+I)
               varev(50+I)=PTJET
            ENDIF
         ENDDO
      ENDDO
C     Set status for non-clustering partons to 2
      DO ihep=1,NHEP
c         ISTORG(ihep)=ISTHEP(ihep)
         IF(ISTHEP(ihep).EQ.1.AND.iabs(IDHEP(ihep)).GT.5.AND.
     $        IDHEP(ihep).NE.21) THEN
            ISTHEP(ihep)=2
         ELSEIF(ISTHEP(ihep).EQ.1.AND.JMOHEP(1,ihep).GT.0) then
            IMO=JMOHEP(1,ihep)
            DO WHILE(IMO.GT.0)
c           Trace mothers, if non-radiating => daughter is decay - remove
              IF(IMO.le.NUP-2.and.norad(IMO+2)) GOTO 105
              IMO=JMOHEP(1,IMO)
            ENDDO
            cycle
 105        ISTHEP(ihep)=2
         ENDIF
      ENDDO

c      DO ihep=1,NHEP
c            print *,'Part ',ihep,' status=',ISTHEP(ihep),'
c     $   PID=',iabs(IDHEP(ihep)),' mother number=',
c     $  JMOHEP(1,ihep),' status=',ISTHEP(JMOHEP(1,
c     $     ihep)),' PID=',IDHEP(JMOHEP(1,ihep))
c      ENDDO

      DO ihep=1,NHEP

	if ( jmohep(1,ihep) .gt. 0 ) then
c	
c         If valid mother and status is 2 and a mother of 6<PID>nqmatch =>reject from particle list
c 
         IF(ISTHEP(JMOHEP(1,ihep)).EQ.2
     $    .AND.iabs(IDHEP(JMOHEP(1,ihep))).GT.nqmatch.AND.
     $    iabs(IDHEP(JMOHEP(1,ihep))).LT.6) THEN
c            print *,'Have found: part ',ihep,' status=',ISTHEP(ihep),
c     $      'PID=',iabs(IDHEP(ihep)),' mother number=',
c     $      JMOHEP(1,ihep),' status=',ISTHEP(JMOHEP(1,
c     $      ihep)),' PID=',IDHEP(JMOHEP(1,ihep))
          ISTHEP(ihep)=2
         ENDIF
         IF(ISTHEP(ihep).eq.1.AND.iabs(IDHEP(ihep)).GT.
     $     nqmatch.AND.iabs(IDHEP(ihep)).LT.6.AND.
     $     ISTHEP(JMOHEP(1,ihep)).EQ.2.AND.iabs(IDHEP(JMOHEP(1,ihep)))
     $     .EQ.21) goto 999
c
        endif

      ENDDO
c
c      DO ihep=1,NHEP
c          IF(ISTHEP(ihep).EQ.1)print *,'After selection:  Part ',
c     $ ihep,' status=',ISTHEP(ihep),'PID=',iabs(IDHEP(ihep))
c     $   ,' mother number=',JMOHEP(1,ihep),' status=',
c     $   ISTHEP(JMOHEP(1,ihep)),' PID=',IDHEP(JMOHEP(1,ihep))
c       ENDDO



C     Prepare histogram filling
        DO I=1,4
          var2(1+I)=-1
          varev(46+I)=-1
          varev(50+I)=-1
        ENDDO

      I=0
      if(idbg.eq.1) then
        do i=1,nhep
          write(LNHOUT,1000)i,isthep(i),idhep(i),jmohep(1,i),jmohep(2,i)
     $         ,phep(1,i),phep(2,i),phep(3,i)
        enddo
 1000   format(5(i4,1x),3(f12.5,1x))
      endif
      
      IF(ICKKW.EQ.2) GOTO 150
      IF(MSTP(61).eq.0..and.MSTP(71).eq.0)then
c      write(*,*)'No showering - just print out event'
      ELSE IF(qcut.le.0d0)then
c      write(*,*)'qcut<0'

      IF(clfact.EQ.0d0) clfact=1.5d0

c      CALL PYLIST(7)
c      CALL PYLIST(2)
c      CALL PYLIST(5)
c     Start from the partonic system
      IF(NLJETS.GT.0) CALL ALPSOR(pt,nljets,KP,2)  
c     reconstruct showered jets
c     
      YCMAX=ETACLMAX+RCLMAX
      YCMIN=-YCMAX
      CALL CALINIM
      CALL CALDELM(1,1)
      CALL GETJETM(RCLMAX,ETCJET,ETACLMAX)
c     analyse only events with at least nljets-reconstructed jets
      IF(NCJET.GT.0) CALL ALPSOR(ETJET,NCJET,K,2)              
      if(idbg.eq.1) then
        write(LNHOUT,*) 'JETS'
        do i=1,ncjet
          j=k(ncjet+1-i)
          ETAJET=PSERAP(PCJET(1,j))
          PHIJET=ATAN2(PCJET(2,j),PCJET(1,j))
          write(LNHOUT,*) etjet(j),etajet,phijet
        enddo
      endif
      IF(NCJET.LT.NLJETS) THEN
        if(idbg.eq.1)
     $     WRITE(LNHOUT,*) 'Failed due to NCJET ',NCJET,' < ',NLJETS
        GOTO 999
      endif
c     associate partons and jets, using min(delr) as criterion
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
        IF(DELRMIN.LT.clfact*RCLMAX) THEN
          NMATCH=NMATCH+1
          KPJ(JRMIN)=I
        ENDIF
C     WRITE(*,*) 'PARTON-JET',I,' best match:',k(ncjet+1-jrmin)
c     $           ,delrmin
      ENDDO
      IF(NMATCH.LT.NLJETS)  THEN
        if(idbg.eq.1)
     $     WRITE(LNHOUT,*) 'Failed due to NMATCH ',NMATCH,' < ',NLJETS
        GOTO 999
      endif
C REJECT EVENTS WITH LARGER JET MULTIPLICITY FROM EXCLUSIVE SAMPLE
      IF(NCJET.GT.NLJETS.AND.IEXC.EQ.1)  THEN
        if(idbg.eq.1)
     $     WRITE(LNHOUT,*) 'Failed due to NCJET ',NCJET,' > ',NLJETS
        GOTO 999
      endif
C     VETO EVENTS WHERE MATCHED JETS ARE SOFTER THAN NON-MATCHED ONES
      IF(IEXC.NE.1) THEN
        J=NCJET
        DO I=1,NLJETS
          IF(KPJ(K(J)).EQ.0) GOTO 999
          J=J-1
        ENDDO
      ENDIF

      else                      ! qcut.gt.0
      if(showerkt.eq.1.0) then
c      write(*,*)"qcut>=0 and showerkt=1 ==> Veto events where
c     & first shower emission has kt > YCUT"


        IF(NLJETS.EQ.0)THEN
           VINT(358)=0
        ENDIF

        IF(idbg.eq.1) THEN
C           PRINT *,'Using shower emission pt method'
C           write(*,*)'Using shower emission pt method'
C           write(*,*)'qcut, ptclus(1), vint(357),vint(358),vint(360): ',
C     $          qcut,ptclus(1),vint(357),vint(358),vint(360)

C      PRINT *,'qcut, ptclus(1), vint(357),vint(358),vint(360): ',
C     $          qcut,ptclus(1),vint(357),vint(358),vint(360)
        ENDIF
        YCUT=qcut**2

        IF(NLJETS.GT.0.AND.PTCLUS(1)**2.LT.YCUT) THEN
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*) 'Failed due to KT ',
     $       PTCLUS(1),' < ',SQRT(YCUT)
          GOTO 999
        ENDIF
c        PRINT *,'Y,VINT:',SQRT(Y(NLJETS+1)),SQRT(VINT(390))
C        write(*,*)'Y,VINT:',SQRT(Y(NLJETS+1)),SQRT(VINT(390))
C        write(*,*)'mektsc 357, 358: ',mektsc,' ',VINT(357),' ',VINT(358)
        IF(IEXC.EQ.1.AND.
     $       ((mektsc.eq.1.and.MAX(VINT(357),VINT(358)).GT.SQRT(YCUT))
     $       .OR.
     $       (mektsc.eq.2.and.MAX(VINT(360),VINT(358)).GT.SQRT(YCUT))))
     $       THEN
C            write(*,*)'rejection'  
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*),
     $       'Failed due to ',max(VINT(357),VINT(358)),' > ',SQRT(YCUT)
          GOTO 999
        ENDIF
C        PRINT *,NLJETS,IEXC,SQRT(VINT(390)),PTCLUS(1),SQRT(YCUT)
C        write(*,*)'NLJets, iexc, VINT, ptclus(1), sqrt(ycut)',NLJETS
c     &,IEXC,SQRT(VINT(390)),PTCLUS(1),SQRT(YCUT)
c     Highest multiplicity case
        IF(IEXC.EQ.0.AND.NLJETS.GT.0.AND.
     $       ((mektsc.eq.1.and.MAX(VINT(357),VINT(358)).GT.PTCLUS(1))
     $       .OR.
     $       (mektsc.eq.2.and.MAX(VINT(360),VINT(358)).GT.PTCLUS(1))))
     $       THEN
c     $     VINT(390).GT.PTCLUS(1)**2)THEN
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*),
     $       'Failed due to ',max(VINT(357),VINT(358)),' > ',PTCLUS(1)
          GOTO 999
        ENDIF
c     
      else                      ! not shower kt method

        IF(clfact.EQ.0d0) clfact=1d0

C---FIND FINAL STATE COLOURED PARTICLES
        NN=0
        DO IHEP=1,NHEP
          IF (ISTHEP(IHEP).EQ.1
     $       .AND.(ABS(IDHEP(IHEP)).LE.5.OR.IDHEP(IHEP).EQ.21)) THEN
            PTJET=sqrt(PHEP(1,IHEP)**2+PHEP(2,IHEP)**2)
            ETAJET=ABS(LOG(MIN((SQRT(PTJET**2+PHEP(3,IHEP)**2)+
     $       ABS(PHEP(3,IHEP)))/PTJET,1d5)))
            IF(ETAJET.GT.etaclmax) cycle
            NN=NN+1
            IF (NN.GT.NMAX) then
              CALL PYLIST(2)
              PRINT *, 'Too many particles: ', NN
              NN=NN-1
              GOTO 120
            endif
            DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
            ENDDO
          ELSE if(idbg.eq.1)THEN
            PRINT *,'Skipping particle ',IHEP,ISTHEP(IHEP),IDHEP(IHEP)
          ENDIF
        ENDDO


C...Cluster event to find values of Y including jet matching but not veto of too many jets
C...Only used to fill the beforeveto Root tree
 120    ECUT=1
        IF (NN.GT.1) then
          CALL KTCLUS(KTSCHE,PP,NN,ECUT,Y,*999)
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*) 'Clustering values:',
     $       (SQRT(Y(i)),i=1,MIN(NN,3))

C       Print out values in the case where all jets are matched at the
C       value of the NLJETS:th clustering
        var2(1)=NLJETS
        var2(6)= Ifile

        if(NLJETS.GT.MINJETS)then
          YCUT=Y(NLJETS)
          CALL KTRECO(MOD(KTSCHE,10),PP,NN,ECUT,YCUT,YCUT,PJET,JET,
     $       NCJET,NSUB,*999)        

C     Cluster jets with first hard parton
          DO I=1,NLJETS
            DO J=1,4
              PPM(J,I)=PJET(J,I)
            ENDDO
          ENDDO
          
          NJETM=NLJETS
          DO IHARD=1,NLJETS
            NNM=NJETM+1
            DO J=1,4
              PPM(J,NNM)=p(J,IHARD)
            ENDDO
            CALL KTCLUS(KTSCHE,PPM,NNM,ECUT,YM,*999)
            IF(YM(NNM).GT.YCUT) THEN
C       Parton not clustered
              GOTO 130
            ENDIF
            
C       Find jet clustered with parton

            IP1=HIST(NNM)/NMAXKT
            IP2=MOD(HIST(NNM),NMAXKT)
            IF(IP2.NE.NNM.OR.IP1.LE.0)THEN
              GOTO 130
            ENDIF
            DO I=IP1,NJETM-1
              DO J=1,4
                PPM(J,I)=PPM(J,I+1)
              ENDDO
            ENDDO
            NJETM=NJETM-1
          ENDDO                 ! IHARD=1,NLJETS
        endif                   ! NLJETS.GT.MINJETS

        DO I=1,MIN(NN,4)
          var2(1+I)=SQRT(Y(I))
        ENDDO
        WRITE(15,4001) (var2(I),I=1,nvar2)

 130    CONTINUE
C   Now perform jet clustering at the value chosen in qcut

        CALL KTCLUS(KTSCHE,PP,NN,ECUT,Y,*999)

        YCUT=qcut**2
        NCJET=0
          
C     Reconstruct jet momenta
          CALL KTRECO(MOD(KTSCHE,10),PP,NN,ECUT,YCUT,YCUT,PJET,JET,
     $       NCJET,NSUB,*999)        

        ELSE IF (NN.EQ.1) THEN

          Y(1)=PP(1,1)**2+PP(2,1)**2
          IF(Y(1).GT.YCUT)THEN
            NCJET=1
            DO I=1,4
              PJET(I,1)=PP(I,1)
            ENDDO
          ENDIF
        endif

        if(idbg.eq.1) then
          write(LNHOUT,*) 'JETS'
          do i=1,ncjet
            PTJET =SQRT(PJET(1,i)**2+PJET(2,i)**2)
            ETAJET=PSERAP(PJET(1,i))
            PHIJET=ATAN2(PJET(2,i),PJET(1,i))
            write(LNHOUT,*) ptjet,etajet,phijet
          enddo
        endif

        IF(NCJET.LT.NLJETS) THEN
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*) 'Failed due to NCJET ',NCJET,' < ',NLJETS
          GOTO 999
        endif

C...Right number of jets - but the right jets?        
C     For max. multiplicity case, count jets only to the NHARD:th jet
        IF(IEXC.EQ.0)THEN
           IF(NLJETS.GT.0)THEN
              YCUT=Y(NLJETS)
              CALL KTRECO(MOD(KTSCHE,10),PP,NN,ECUT,YCUT,YCUT,PJET,JET,
     $             NCJET,NSUB,*999)
              IF(clfact.GE.0d0) THEN
                 CALL ALPSOR(PTCLUS,nljets,KPT,2)
                 YCUT=MAX(qcut,PTCLUS(KPT(1)))**2
              ENDIF
           ENDIF
        ELSE IF(NCJET.GT.NLJETS) THEN
           if(idbg.eq.1)
     $       WRITE(LNHOUT,*) 'Failed due to NCJET ',NCJET,' > ',NLJETS
           GOTO 999
        ENDIF
C     Cluster jets with hard partons, one at a time
        DO I=1,NLJETS
          DO J=1,4
            PPM(J,I)=PJET(J,I)
          ENDDO
        ENDDO

        NJETM=NLJETS
        IF(clfact.NE.0) YCUT=clfact**2*YCUT
c        YCUT=qcut**2
c        YCUT=(1.5*qcut)**2

        DO 140 IHARD=1,NLJETS
          NN=NJETM+1
          DO J=1,4
            PPM(J,NN)=p(J,IHARD)
          ENDDO
          CALL KTCLUS(KTSCHE,PPM,NN,ECUT,Y,*999)

          IF(Y(NN).GT.YCUT) THEN
C       Parton not clustered
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*) 'Failed due to parton ',IHARD,
     $         ' not clustered: ',Y(NN)
            GOTO 999
          ENDIF
          
C       Find jet clustered with parton

          IP1=HIST(NN)/NMAXKT
          IP2=MOD(HIST(NN),NMAXKT)
          IF(IP2.NE.NN.OR.IP1.LE.0)THEN
          if(idbg.eq.1)
     $       WRITE(LNHOUT,*) 'Failed due to parton ',IHARD,
     $         ' not clustered: ',IP1,IP2,NN,HIST(NN)
            GOTO 999
          ENDIF
C     Remove jet clustered with parton
          DO I=IP1,NJETM-1
            DO J=1,4
              PPM(J,I)=PPM(J,I+1)
            ENDDO
          ENDDO
          NJETM=NJETM-1
 140   CONTINUE

      endif                     ! pt-ordered showers
      endif                     ! qcut.gt.0
C...Cluster particles with |eta| < etaclmax for histograms
 150  NN=0
      DO IHEP=1,NHEP
         IF (ISTHEP(IHEP).EQ.1
     $        .AND.(ABS(IDHEP(IHEP)).LE.5.OR.IDHEP(IHEP).EQ.21)) THEN
            PTJET=sqrt(PHEP(1,IHEP)**2+PHEP(2,IHEP)**2)
            ETAJET=ABS(LOG(MIN((SQRT(PTJET**2+PHEP(3,IHEP)**2)+
     $           ABS(PHEP(3,IHEP)))/PTJET,1d5)))
            IF(ETAJET.GT.etaclmax) cycle
            NN=NN+1
            IF (NN.GT.NMAX) then
               CALL PYLIST(2)
               PRINT *, 'Too many particles: ', NN
               NN=NN-1
               GOTO 160
            ENDIF
            DO I=1,4
               PP(I,NN)=PHEP(I,IHEP)
            ENDDO
         ELSE if(idbg.eq.1)THEN
            PRINT *,'Skipping particle ',IHEP,ISTHEP(IHEP),IDHEP(IHEP)
         ENDIF
      ENDDO
      
 160  ECUT=1
      IF (NN.GT.1) THEN
         CALL KTCLUS(KTSCHE,PP,NN,ECUT,Y,*999)
      ELSE IF(NN.EQ.1) THEN
         Y(1)=PP(1,NN)**2+PP(2,NN)**2
      ENDIF

      DO I=1,MIN(NN,4)
         varev(46+I)=SQRT(Y(I))
      ENDDO

c      write(*,*)' finishing up mgveto, with ipveto= ', ipveto
        OPEN (10, FILE='events.tree')
c       WRITE(10,'(a)') '# File with ntuple events with the variables:'
c      WRITE(10,CGIVE0) (htit(I)(1:len_trim(htit(I))),I=1,nvarev)
c      write (*,'(a)'),(htit(I)(1:len_trim(htit(I))),I=47,50)
c      write (*,4001),(varev(I),I=47,50)
       if (flag.eq.1) then
          varev(1)=NLJETS
          WRITE(10,4001) varev(1),(varev(I),I=47,50)
c          WRITE(*,4001) varev(1),(varev(I),I=47,50)
      endif

      RETURN
 4001 FORMAT(50E15.6)
c HERWIG/PYTHIA TERMINATION:



 999  IPVETO=1
c      write(*,*)' finishing up mgveto, with ipveto= ', ipveto
      END

C*********************************************************************
C   PYMASC
C   Implementation of scale used in Pythia parton showers
C*********************************************************************
      SUBROUTINE PYMASC(scale)
      IMPLICIT NONE

C...Arguments
      REAL*8 scale

C...Functions
      REAL*8 SMDOT5

C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &   IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &   LPRUP(MAXPUP)
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &   ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &   VTIMUP(MAXNUP),SPINUP(MAXNUP)

C...Extra commonblock to transfer run info.
      INTEGER LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      COMMON/UPPRIV/LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE

C...Local variables
      INTEGER ICC1,ICC2,IJ,IDC1,IDC2,IC,IC1,IC2
      REAL*8 QMIN,QTMP

C   Just use the scale read off the event record
      scale=SCALUP

C   Alternatively:

C...  Guesses for the correct scale
C     Assumptions:
C     (1) if the initial state is a color singlet, then
C     use s-hat for the scale
C     
C     (2) if color flow to the final state, use the minimum
C     of the dot products of color connected pairs
C     (times two for consistency with above)

        QMIN=SMDOT5(PUP(1,1),PUP(1,2))
        ICC1=1
        ICC2=2
C     
C     For now, there is no generic way to guarantee the "right"
C     scale choice.  Here, we take the HERWIG pt. of view and
C     choose the dot product of the colored connected "primary"
C     pairs.
C     

        DO 101 IJ=1,NUP
          IF(MOTHUP(2,IJ).GT.2) GOTO 101
          IDC1=ICOLUP(1,IJ)
          IDC2=ICOLUP(2,IJ)
          IF(IDC1.EQ.0) IDC1=-1
          IF(IDC2.EQ.0) IDC2=-2
          
          DO 201 IC=IJ+1,NUP
            IF(MOTHUP(2,IC).GT.2) GOTO 201
            IC1=ICOLUP(1,IC)
            IC2=ICOLUP(2,IC)
            IF(ISTUP(IC)*ISTUP(IJ).GE.1) THEN
              IF(IDC1.EQ.IC2.OR.IDC2.EQ.IC1) THEN
                QTMP=SMDOT5(PUP(1,IJ),PUP(1,IC))
                IF(QTMP.LT.QMIN) THEN
                  QMIN=QTMP
                  ICC1=IJ
                  ICC2=IC
                ENDIF
              ENDIF
            ELSEIF(ISTUP(IC)*ISTUP(IJ).LE.-1) THEN
              IF(IDC1.EQ.IC1.OR.IDC2.EQ.IC2) THEN
                QTMP=SMDOT5(PUP(1,IJ),PUP(1,IC))          
                IF(QTMP.LT.QMIN) THEN
                  QMIN=QTMP
                  ICC1=IJ
                  ICC2=IC
                ENDIF
              ENDIF
            ENDIF
 201      CONTINUE
 101    CONTINUE

        scale=QMIN

      RETURN
      END

C...SMDOT5
C   Helper function

      FUNCTION SMDOT5(V1,V2)
      IMPLICIT NONE
      REAL*8 SMDOT5,TEMP
      REAL*8 V1(5),V2(5)
      INTEGER I

      SMDOT5=0D0
      TEMP=V1(4)*V2(4)
      DO I=1,3
        TEMP=TEMP-V1(I)*V2(I)
      ENDDO

      SMDOT5=SQRT(ABS(TEMP))

      RETURN
      END

C*********************************************************************      
C...set_matching
C...Sets parameters for the matching, i.e. cuts and jet multiplicities
C*********************************************************************      

      SUBROUTINE set_matching(npara,param,value)
      implicit none
c   
c   arguments
c   
      integer npara
      character*20 param(*),value(*)

C...Pythia parameters.
      INTEGER MSTP,MSTI
      DOUBLE PRECISION PARP,PARI
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)

C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &   IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &   LPRUP(MAXPUP)

C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &   ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &   VTIMUP(MAXNUP),SPINUP(MAXNUP)

C...Extra commonblock to transfer run info.
      INTEGER LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      COMMON/UPPRIV/LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE

C...Inputs for the matching algorithm
      double precision etcjet,rclmax,etaclmax,qcut,clfact,showerkt
      integer maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres(30)
      integer nqmatch,nexcproc,iexcproc(MAXPUP),iexcval(MAXPUP)
      logical nosingrad,jetprocs
      common/MEMAIN/etcjet,rclmax,etaclmax,qcut,showerkt,clfact,
     $   maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres,
     $   nqmatch,nexcproc,iexcproc,iexcval,nosingrad,jetprocs

c      DATA ktsche,maxjets,minjets,nexcres/0,-1,-1,0/
c      DATA ktsche,nexcres/0,0/
c      DATA qcut,clfact,showerkt/0d0,0d0,0d0/ 

C...Commonblock to transfer event-by-event matching info
      INTEGER NLJETS,IEXC,Ifile
      DOUBLE PRECISION PTCLUS
      COMMON/MEMAEV/PTCLUS(20),NLJETS,IEXC,Ifile

C...Local variables
      INTEGER I,MAXNJ,NREAD,MINJ,MAXJ
      parameter(MAXNJ=6)
      DOUBLE PRECISION XSTOT(MAXNJ),XSECTOT
      DOUBLE PRECISION ptjmin,etajmax,drjmin,ptbmin,etabmax,xqcut

      integer icount 

C...Functions
      INTEGER iexclusive
      EXTERNAL iexclusive

C...Initialize the icount counter to detect infinite loops
      icount=0

C...Need lower scale for final state radiation in e+e-
      IF(IABS(IDBMUP(1)).EQ.11.AND.IABS(IDBMUP(2)).EQ.11) then
        CALL PYGIVE('PARP(71)=1')
      ENDIF

C...CRUCIAL FOR JET-PARTON MATCHING: CALL UPVETO, ALLOW JET-PARTON MATCHING
C      call pygive('MSTP(143)=1')

C     
C...Check jet multiplicities and set processes
C
      DO I=1,MAXNJ
        XSTOT(I)=0D0
      ENDDO
      MINJ=MAXNJ
      MAXJ=0
      NREAD=0
      NUP=0  
      DO WHILE(.true.)
C	  write(LNHOUT,*)'Launching MGEVNT'
        CALL MGEVNT()
		write(LNHOUT,*)'NLJETS=',NLJETS

	icount = icount+1
        if (icount.gt.10) then
          write (LNHOUT,*) 
     &      'GeneratorInterface/PartonShowerVeto ME2phythia:'
     &      //' Aborting, loop in set_matching above ',icount,' cycles'
          write (LNHOUT,*) 'NUP = ',NUP,' IEXC = ',IEXC 
          stop
        endif  

        IF(NUP.eq.0) goto 20
        IF(IEXC.EQ.-1) cycle

        if(NLJETS.GT.MAXJ) MAXJ=NLJETS
        if(NLJETS.LT.MINJ) MINJ=NLJETS
c        XSTOT(NLJETS+1)=XSTOT(NLJETS+1)+XWGTUP
        XSTOT(NLJETS+1)=XSTOT(NLJETS+1)+1
        NREAD=NREAD+1
      ENDDO

 20   continue
		  
C      REWIND(iunit)

      write(LNHOUT,*) 'Minimum number of jets in file: ',MINJ
      write(LNHOUT,*) 'Maximum number of jets in file: ',MAXJ

      XSECTOT=0d0
      DO I=1,NPRUP
         XSECTOT=XSECTOT+XSECUP(I)
      ENDDO
		write(LNHOUT,*)'NPRUP=',NPRUP
      IF(NPRUP.eq.1.AND.MINJ.lt.MAXJ)THEN
C...If different process ids not set by user, set by jet number

         jetprocs=.true.
         IF(IEXCFILE.eq.0.AND.iexclusive(LPRUP(1)).ne.1) THEN
            nexcproc=1
            IEXCPROC(1)=MAXJ-MINJ
            IEXCVAL(1)=0
         ENDIF
         NPRUP=1+MAXJ-MINJ
         DO I=MINJ,MAXJ
            XSECUP(1+I-MINJ) = XSECTOT*XSTOT(I+1)/NREAD
            XMAXUP(1+I-MINJ) = XMAXUP(1)
            LPRUP(1+I-MINJ)  = I-MINJ
         ENDDO
      ELSE IF(IEXCFILE.EQ.0) THEN
C...Check if any IEXCPROC set, then set IEXCFILE=1
         DO I=1,NPRUP
            IF(iexclusive(LPRUP(I)).EQ.0) IEXCFILE=1
         ENDDO
      ENDIF

      WRITE(LNHOUT,*) ' Number of Events Read:: ',NREAD
      WRITE(LNHOUT,*) ' Total cross section (pb):: ',XSECTOT
      WRITE(LNHOUT,*) ' Process   Cross Section (pb):: '
      DO I=1,NPRUP
        WRITE(LNHOUT,'(I5,E23.5)') I,XSECUP(I)
      ENDDO

      IF(MINJETS.EQ.-1) MINJETS=MINJ
      IF(MAXJETS.EQ.-1) MAXJETS=MAXJ
      write(LNHOUT,*) 'Minimum number of jets allowed: ',MINJETS
      write(LNHOUT,*) 'Maximum number of jets allowed: ',MAXJETS
      write(LNHOUT,*) 'IEXCFILE = ',IEXCFILE
      write(LNHOUT,*) 'jetprocs = ',jetprocs
      DO I=1,NPRUP
         write(LNHOUT,*) 'IEXCPROC(',LPRUP(I),') = ',
     $        iexclusive(LPRUP(I))
      ENDDO

C      CALL FLUSH()

C...Run PYPTFS instead of PYSHOW
c        CALL PYGIVE("MSTJ(41)=12")

c***********************************************************************
c   Read in jet cuts
c***********************************************************************

        call get_real   (npara,param,value," ptj " ,ptjmin,7d3)
        call get_real   (npara,param,value," etaj " ,etajmax,7d3)
        call get_real   (npara,param,value," ptb " ,ptbmin,7d3)
        call get_real   (npara,param,value," etab " ,etabmax,7d3)
        call get_real   (npara,param,value," drjj " ,drjmin,7d3)
        call get_real   (npara,param,value," xqcut " ,xqcut,0d0)

        if(qcut.lt.xqcut) then
           if(showerkt.eq.1) then
              qcut=xqcut
           else
              qcut=max(xqcut*1.2,xqcut+5)
           endif
        endif
        if(xqcut.le.0)then
           write(*,*) 'Warning! ME generation QCUT = 0. QCUT set to 0!'
           qcut=0
        endif

c        etajmax=min(etajmax,etabmax)
c        ptjmin=max(ptjmin,ptbmin)

c      IF(ICKKW.EQ.1) THEN
c        WRITE(*,*) ' '
c        WRITE(*,*) 'INPUT 0 FOR INCLUSIVE JET SAMPLE, 1 FOR EXCLUSIVE'
c        WRITE(*,*) '(SELECT 0 FOR HIGHEST PARTON MULTIPLICITY SAMPLE)' 
c        WRITE(*,*) '(SELECT 1 OTHERWISE)'
c        READ(*,*) IEXCFILE
c      ENDIF
        
C     INPUT PARAMETERS FOR CONE ALGORITHM

        IF(ETCJET.LE.PTJMIN)THEN
           ETCJET=MAX(PTJMIN+5,1.2*PTJMIN)
        ENDIF

        RCLMAX=DRJMIN
        ETACLMAX=ETAJMAX
        IF(qcut.le.0)THEN
          WRITE(*,*) 'JET CONE PARAMETERS FOR MATCHING:'
          WRITE(*,*) 'ET>',ETCJET,' R=',RCLMAX
          WRITE(*,*) 'DR(PARTON-JET)<',1.5*RCLMAX
          WRITE(*,*) 'ETA(JET)<',ETACLMAX
      ELSE IF(ickkw.eq.1) THEN
        WRITE(*,*) 'KT JET PARAMETERS FOR MATCHING:'
        WRITE(*,*) 'QCUT=',qcut
        WRITE(*,*) 'ETA(JET)<',ETACLMAX
        WRITE(*,*) 'Note that in ME generation, qcut = ',xqcut
        write(*,*)'the showerkt param is ',showerkt
        if(showerkt.eq.1.0)THEN
C             WRITE(*,*) 'shower kt is activated'
        endif
        if(showerkt.eq.1.0.and.MSTP(81).LT.20)THEN
          WRITE(*,*)'WARNING: "shower kt" needs pT-ordered showers'
          WRITE(*,*)'         Setting MSTP(81)=',20+MOD(MSTP(81),10)
          MSTP(81)=20+MOD(MSTP(81),10)
       endif
      else if(ickkw.eq.2)then
c     Turn off color coherence suppressions (leave this to ME)
        CALL PYGIVE('MSTP(62)=2')
        CALL PYGIVE('MSTP(67)=0')
        if(MSTP(81).LT.20)THEN
          WRITE(*,*)'WARNING: Must run CKKW with pt-ordered showers'
          WRITE(*,*)'         Setting MSTP(81)=',20+MOD(MSTP(81),10)
          MSTP(81)=20+MOD(MSTP(81),10)
        endif
      endif
      return
      end

      subroutine get_real(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c   finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none

c   
c   arguments
c   
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      real*8 var,def_value
c   
c   local
c   
      logical found
      integer i
c   
c   start
c  
c      write(*,*)'entered get_real subroutine, looking for ',name,
c     &' there are ',npara,' parameters' 
      i=1
      found=.false.
      do while(.not.found.and.i.le.npara)
c         write(*,*)'trying ',param(i)
        found = (index(param(i),name).ne.0)
        if (found) read(value(i),*) var
c     if (found) write (*,*) name,var
        i=i+1
      enddo
      if (.not.found) then
        write (*,*) "Warning: parameter ",name," not found"
        write (*,*) "         setting it to default value ",def_value
        var=def_value
      else
        write(*,*),'Found parameter ',name,var
      endif
      return

      end
c   

      subroutine get_integer(npara,param,value,name,var,def_value)
c----------------------------------------------------------------------------------
c   finds the parameter named "name" in param and associate to "value" in value 
c----------------------------------------------------------------------------------
      implicit none
c   
c   arguments
c   
      integer npara
      character*20 param(*),value(*)
      character*(*)  name
      integer var,def_value
c   
c   local
c   
      logical found
      integer i
c   
c   start
c   
      i=1
      found=.false.
      do while(.not.found.and.i.le.npara)
        found = (index(param(i),name).ne.0)
        if (found) read(value(i),*) var
c     if (found) write (*,*) name,var
        i=i+1
      enddo
      if (.not.found) then
        write (*,*) "Warning: parameter ",name," not found"
        write (*,*) "         setting it to default value ",def_value
        var=def_value
      else
        write(*,*)'Found parameter ',name,var
      endif
      return

      end

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
      IF(IL(J).EQ.0) GOTO 4
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
      IF(IR(J)) 12,30,13
  13  J=IR(J)
      GOTO 8
  12  J=-IR(J)
      GOTO 9
  30  IF(IOPT.EQ.2) RETURN
      DO 31 I=1,N
  31  A(I)=B(I)
      END

C-----------------------------------------------------------------------
C----Calorimeter simulation obtained from Frank Paige 23 March 1988-----
C
C          USE
C
C     CALL CALINIM
C     CALL CALSIMM
C
C          THEN TO FIND JETS WITH A SIMPLIFIED VERSION OF THE UA1 JET
C          ALGORITHM WITH JET RADIUS RJET AND MINIMUM SCALAR TRANSVERSE
C          ENERGY EJCUT
C            (RJET=1., EJCUT=5. FOR UA1)
C          USE
C
C     CALL GETJETM(RJET,EJCUT)
C
C
C-----------------------------------------------------------------------
C 
C          ADDED BY MIKE SEYMOUR: PARTON-LEVEL CALORIMETER. ALL PARTONS
C          ARE CONSIDERED TO BE HADRONS, SO IN FACT RESEM IS IGNORED
C
C     CALL CALPARM
C
C          HARD PARTICLE CALORIMETER. ONLY USES THOSE PARTICLES WHICH
C          CAME FROM THE HARD PROCESS, AND NOT THE UNDERLYING EVENT
C
C     CALL CALHARM
C
C-----------------------------------------------------------------------
      SUBROUTINE CALINIM
C                
C          INITIALIZE CALORIMETER FOR CALSIMM AND GETJETM.  NOTE THAT
C          BECAUSE THE INITIALIZATION IS SEPARATE, CALSIMM CAN BE
C          CALLED MORE THAN ONCE TO SIMULATE PILEUP OF SEVERAL EVENTS.
C
      IMPLICIT NONE
C...GETJET commonblocks
      INTEGER MNCY,MNCPHI,NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,DELY,DELPHI,ET,STHCAL,CTHCAL,CPHCAL,
     &  SPHCAL,PCJET,ETJET
      PARAMETER (MNCY=200)
      PARAMETER (MNCPHI=200)
      COMMON/CALORM/DELY,DELPHI,ET(MNCY,MNCPHI),
     $CTHCAL(MNCY),STHCAL(MNCY),CPHCAL(MNCPHI),SPHCAL(MNCPHI),
     $YCMIN,YCMAX,NCY,NCPHI
      PARAMETER (NJMAX=500)
      COMMON/GETCOMM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(MNCY,MNCPHI),
     $     NCJET

      INTEGER IPHI,IY
      DOUBLE PRECISION PI,PHIX,YX,THX
      PARAMETER (PI=3.141593D0)
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
      SUBROUTINE CALSIMM
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

C...GETJET commonblocks
      INTEGER MNCY,MNCPHI,NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,DELY,DELPHI,ET,STHCAL,CTHCAL,CPHCAL,
     &  SPHCAL,PCJET,ETJET
      PARAMETER (MNCY=200)
      PARAMETER (MNCPHI=200)
      COMMON/CALORM/DELY,DELPHI,ET(MNCY,MNCPHI),
     $CTHCAL(MNCY),STHCAL(MNCY),CPHCAL(MNCPHI),SPHCAL(MNCPHI),
     $YCMIN,YCMAX,NCY,NCPHI
      PARAMETER (NJMAX=500)
      COMMON/GETCOMM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(MNCY,MNCPHI),
     $     NCJET

      INTEGER IHEP,ID,IY,IPHI
      DOUBLE PRECISION PI,YIP,PSERAP,PHIIP,EIP
      PARAMETER (PI=3.141593D0)
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
      SUBROUTINE GETJETM(RJET,EJCUT,ETAJCUT)
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
C...GETJET commonblocks
      INTEGER MNCY,MNCPHI,NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,DELY,DELPHI,ET,STHCAL,CTHCAL,CPHCAL,
     &  SPHCAL,PCJET,ETJET
      PARAMETER (MNCY=200)
      PARAMETER (MNCPHI=200)
      COMMON/CALORM/DELY,DELPHI,ET(MNCY,MNCPHI),
     $CTHCAL(MNCY),STHCAL(MNCY),CPHCAL(MNCPHI),SPHCAL(MNCPHI),
     $YCMIN,YCMAX,NCY,NCPHI
      PARAMETER (NJMAX=500)
      COMMON/GETCOMM/PCJET(4,NJMAX),ETJET(NJMAX),JETNO(MNCY,MNCPHI),
     $     NCJET

      INTEGER IPHI,IY,J,K,NPHI1,NPHI2,NY1,
     &  NY2,IPASS,IYMX,IPHIMX,ITLIS,IPHI1,IPHIX,IY1,IYX
      DOUBLE PRECISION PI,RJET,
     &  ETMAX,ETSTOP,RR,ECCUT,PX,EJCUT
      PARAMETER (PI=3.141593D0)
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
C-ap  initialize these two too to avoid compiler warnings
      iymx = 0
      iphimx = 0
C-ap  end  
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
8888    FORMAT(//' ERROR IN GETJETM...IPASS > ',I6)
        RETURN
      ENDIF
      NCJET=NCJET+1
      IF(NCJET.GT.NJMAX) THEN
        WRITE(ITLIS,9999) NCJET
9999    FORMAT(//' ERROR IN GETJETM...NCJET > ',I5)
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
      SUBROUTINE CALDELM(ISTLO,ISTHI)
C     LABEL ALL PARTICLES WITH STATUS BETWEEN ISTLO AND ISTHI (UNTIL A
C     PARTICLE WITH STATUS ISTOP IS FOUND) AS FINAL-STATE, CALL CALSIMM
C     AND THEN PUT LABELS BACK TO NORMAL
C-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER MAXNUP
      PARAMETER(MAXNUP=500)
C...HEPEVT commonblock.
      INTEGER NMXHEP,NEVHEP,NHEP,ISTHEP,IDHEP,JMOHEP,JDAHEP
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
      SAVE /HEPEVT/
      INTEGER ISTLO,ISTHI

C...Avoid gcc 4.3.4 warnings.
      ISTLO=ISTLO
      ISTHI=ISTHI
      CALL CALSIMM
      END

C****************************************************
C iexclusive returns whether exclusive process or not
C****************************************************

      integer function iexclusive(iproc)
      implicit none
      
      integer iproc, i
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)

C...Inputs for the matching algorithm
      double precision etcjet,rclmax,etaclmax,qcut,clfact,showerkt
      integer maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres(30)
      integer nqmatch,nexcproc,iexcproc(MAXPUP),iexcval(MAXPUP)
      logical nosingrad,jetprocs
      common/MEMAIN/etcjet,rclmax,etaclmax,qcut,showerkt,clfact,
     $   maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres,
     $   nqmatch,nexcproc,iexcproc,iexcval,nosingrad,jetprocs

      
      iexclusive=-2
      do i=1,nexcproc
         if(iproc.eq.iexcproc(i)) then
            iexclusive=iexcval(i)
            return
         endif
      enddo

      return
      end

C*********************************************************************

      subroutine initpydata

      INTEGER KCHG
      DOUBLE PRECISION PMAS,PARF,VCKM
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      INTEGER MSTP,MSTI
      DOUBLE PRECISION PARP,PARI
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)

      if(PMAS(1,1).lt.0.1D0.or.PMAS(1,1).gt.1.) THEN

        PMAS(1,1) = 0.33D0
        PMAS(2,1) = 0.33D0
        PMAS(3,1) = 0.5D0
        PMAS(4,1) = 1.5D0
        PMAS(5,1) = 4.8D0
        PMAS(6,1) = 175D0
        PMas(7,1) = 400D0
        PMas(8,1) = 400D0
        PMas(9,1) = 0D0
        PMas(10,1) = 0D0
        PMas(11,1) = 0.0005D0
        PMas(12,1) = 0D0
        PMas(13,1) = 0.10566D0
        PMas(14,1) = 0D0
        PMas(15,1) = 1.777D0
        PMas(16,1) = 0D0
        PMas(17,1) = 400D0
        PMas(18,1) = 0D0
        PMas(19,1) = 0D0
        PMas(20,1) = 0D0
        PMAS(21,1) = 0D0

        MSTP(61) = 2
        MSTP(71) = 1
        MSTP(183)= 2013

      endif

      return
      end

C***********************************
C Common block initialization block
C***********************************      

      BLOCK DATA MEPYDAT

      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
C...Inputs for the matching algorithm
      double precision etcjet,rclmax,etaclmax,qcut,clfact,showerkt
      integer maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres(30)
      integer nqmatch,nexcproc,iexcproc(MAXPUP),iexcval(MAXPUP)
      logical nosingrad,jetprocs
      common/MEMAIN/etcjet,rclmax,etaclmax,qcut,showerkt,clfact,
     $   maxjets,minjets,iexcfile,ktsche,mektsc,nexcres,excres,
     $   nqmatch,nexcproc,iexcproc,iexcval,nosingrad,jetprocs

C...GETJET commonblocks
      INTEGER MNCY,MNCPHI,NCY,NCPHI,NJMAX,JETNO,NCJET
      DOUBLE PRECISION YCMIN,YCMAX,DELY,DELPHI,ET,STHCAL,CTHCAL,CPHCAL,
     &  SPHCAL,PCJET,ETJET
      PARAMETER (MNCY=200)
      PARAMETER (MNCPHI=200)
      COMMON/CALORM/DELY,DELPHI,ET(MNCY,MNCPHI),
     $CTHCAL(MNCY),STHCAL(MNCY),CPHCAL(MNCPHI),SPHCAL(MNCPHI),
     $YCMIN,YCMAX,NCY,NCPHI

C...Extra commonblock to transfer run info.
      INTEGER LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE
      COMMON/UPPRIV/LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE

C...Initialization statements
      DATA showerkt/0.0/
      DATA qcut,clfact,etcjet/0d0,0d0,0d0/
      DATA ktsche,mektsc,maxjets,minjets,nexcres/0,1,-1,-1,0/
      DATA nqmatch/5/
      DATA nexcproc/0/
      DATA iexcproc/MAXPUP*-1/
      DATA iexcval/MAXPUP*-2/
C      DATA nosingrad,showerkt,jetprocs/.false.,.false.,.false./

      DATA NCY,NCPHI/50,60/

      DATA LNHIN,LNHOUT,MSCAL,IEVNT,ICKKW,ISCALE/77,6,0,0,0,1/

C      nosingrad = .false.
C      showerkt = .false.
C      jetprocs = .false.

      END

