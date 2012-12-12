
c 15.01.2009 Simplified Main program and random number generator for epos

      subroutine crmc_set_f(iEvent,iSeed,pproj,ptarg,
     $     ipart,itarg,imodel,itab,ilheout,lheoutfile,param)

***************************************************************
*
*  interface to epos subroutine
*
*   input: iEvent     - number of events to generate
*          iseed      - random seed
*          pproj      - beam momentum in GeV/c (detector frame)
*          ptarg      - target momentum in GeV/c of nucleon! (detector frame)
*          ipart      - primary particle type
*          itarg      - target particle type
*          imodel     - HE model switch
*          itab       - force tables production or stop if missing
*          ilheout    - output type
*          lheoutfile - output file name
*          param      - param file name
*
***************************************************************
      implicit none
      include "epos.inc"

c     Input values
      integer iSeed,ipart,itarg,imodel,itab,ilheout,iEvent,iout
      double precision pproj, ptarg
      character*1000 param,lheoutfile,output
      common/lheoutput/iout,output

      real   m1,m2
      double precision iecms,e1,e2

      double precision ycm2det
      common/boostvars/ycm2det

      common/producetab/ producetables              !used to link CRMC
      logical producetables                         !with EPOS and QII

c     Set parameters to default value
      call aaset(0)

c     Stop program if missing tables (after aaset)
      producetables=.false.
      if(itab.eq.1)producetables=.true.

c     Set common for crmc_init
      iout=ilheout
      output=lheoutfile

c     Calculations of energy of the center-of-mass in the detector frame
      call idmass(1120,m2)      !target mass = proton
      m1=m2                     !projectile mass
      if(abs(itarg).eq.120)call idmass(120,m2) !if pion as targ or proj
      if(abs(ipart).eq.120)call idmass(120,m1)
      e2=dsqrt(dble(m1)**2+ptarg**2)
      e1=dsqrt(dble(m2)**2+pproj**2)
      iecms=dsqrt((e1+e2)**2-(pproj+ptarg)**2)
c     Later a rapidity boost back into the detector system will be performed
c     ycm2det defines this rapidity
      ycm2det=0
C       if (((e1+e2)-(pproj+ptarg)) .le. 0d0) then
C          ycm2det=1d99
C       elseif (((e1+e2)+(pproj+ptarg)) .le. 0d0) then
C          ycm2det=-1d99
C       else
C          ycm2det=0.5d0*dlog(((e1+e2)+(pproj+ptarg))/
C      +        ((e1+e2)-(pproj+ptarg)))
C       endif
C       if (pproj .le. 0d0) then
C          ycm2det=-ycm2det
C       endif
c     Update some parameters value to run correctly
      call IniEpos(iEvent,iSeed,ipart,itarg,iecms,imodel)
      
c     The parameters can be changed optionnaly by reading a file
c     (example.param) using the following subroutine call
      call EposInput(param)     !(it can be commented)

c     if you put what is in input.optns in example.param, you can even run
c     exactly the same way (coded parameters are overwritten). Don't forget
c     the command : "EndEposInput" at the end of example.param, otherwise it
c     will not run.
      end
      
      subroutine crmc_init_f()
***************************************************************
*
*  init models with values set in crmc_set_f
*
***************************************************************
      implicit none
      integer iout
      character*1000 output
      common/lheoutput/iout,output

c     initialization for the given energy
      call ainit

c     Here the cross section sigineaa is defined

c     LHE type output done by EPOS
      if(iout.eq.1)call EposOutput(output)
      
      end


      subroutine crmc_f(iout,ievent,noutpart,impactpar,outpart,outpx
     +                  ,outpy,outpz,oute,outm,outstat)

***************************************************************
*   output: iout      - output type
*           ievent    - event number
*           noutpart  - number of stable particles produced
*           impactpar - impact parameter in fm
*           outpart   - particle ID (pdg code)
*           outpx     - particle momentum px
*           outpy     - particle momentum py
*           outpz     - particle momentum pz
*           oute      - particle energy
*           outm      - particle mass
*           outstat   - particle status code
*
***************************************************************
      implicit none
      include "epos.inc"
c     Output quantities
      integer noutpart,iout,ievent
      double precision impactpar
      integer outpart(*)
      double precision outpx(*), outpy(*), outpz(*)
      double precision oute(*), outm(*)
      integer outstat(*)

      double precision boostvec1,boostvec2,boostvec3,boostvec4,boostvec5
      double precision ycm2det
      common/boostvars/ycm2det !is this common block needed?

      integer i!,k

c     Calculate an inelastic event
      call aepos(-1)

c     Fix final particles and some event parameters
      call afinal

c     Fill HEP common
      call hepmcstore

c     optional Statistic information (only with debug level ish=1)
      call astati
      if(ish.ge.1) call bfinal

c     Print out (same as the one defined in input.optns)
      if(nhep.gt.nmxhep)then
        print *,'Warning : produced number of particles is too high'
        print *,'          increase nmxhep : ',nhep,' > ',nmxhep
c        stop
      endif
      noutpart=nhep
      impactpar=dble(bimevt)
c     define vec to boost from cm. to cms frame
      boostvec1=0d0
      boostvec2=0d0
      boostvec3=dsinh(dble(ycm2det))
      boostvec4=dcosh(dble(ycm2det))
      boostvec5=1d0
c      write(*,*)nevhep,nhep,boostvec3,boostvec4,ycm2det
      do i=1,nhep
c     boost output to cms frame
         call utlob2(-1,boostvec1,boostvec2,boostvec3
     +        ,boostvec4,boostvec5
     +        ,vhep(1,i),vhep(2,i),vhep(3,i),vhep(4,i),-99)
         call utlob5dbl(-ycm2det
     +        ,phep(1,i), phep(2,i), phep(3,i), phep(4,i), phep(5,i))
         outpart(i)=idhep(i)
         outpx(i)=phep(1,i)
         outpy(i)=phep(2,i)
         outpz(i)=phep(3,i)
         oute(i)=phep(4,i)
         outm(i)=phep(5,i)
         outstat(i)=isthep(i)
c      write(*,'(4x,i6,1x,4(e12.6,1x))')idhep(i),(vhep(k,i),k=1,4)
      enddo

c     Write lhe file
      if(iout.eq.1)call lhesave(ievent)

      end


c-----------------------------------------------------------------------
      subroutine IniEpos(iEvent,iSeed,ipart,itarg,iecms,iModel)
c-----------------------------------------------------------------------
c     Update some parameters and define path to tab files here can be set
c     what particle is stable or not transfer number of event and debug
c     level to main program (to avoid epos.inc in main)
c-----------------------------------------------------------------------
      implicit none
      include "epos.inc"

      double precision iecms
      integer iSeed,ipart,itarg,iModel,iadd,idtrafo,iEvent
      character*4 lhct
      
      iframe=11                 !11 puts it always in nucleon nucleon reference
                                !frame. This is ok because we use ecms
                                !which is calculated in crmc_f.

      model=max(1,iModel)              ! epos = 0,1 / qgsjet01 = 2 / gheisha = 3
                                ! / pythia = 4 / hijing = 5 / sibyll 2.1
                                ! = 6 / qgsjetII.04 = 7 / phojet = 8
                                ! qgsjetII.03 = 11
      if(iModel.eq.0)then
        call LHCparameters      !LHC tune for EPOS
        lhct=".lhc"
        iadd=4
      else
        lhct=""
        iadd=0
      endif

      nfnii=15                  ! epos tab file name lenght
      fnii="tabs/epos.initl"    ! epos tab file name
      nfnid=15
      fnid="tabs/epos.inidi"
      nfnie=15
      fnie="tabs/epos.iniev"
      nfnrj=15+iadd
      fnrj="tabs/epos.inirj"//lhct
      nfncs=15+iadd
      fncs="tabs/epos.inics"//lhct


      seedi=1.d0                !seed for random number generator: at start program
      seedj=iSeed               !seed for random number generator: for first event
      jwseed=0                  !print out seed in see file (1) or not


c     Initialize decay of particles
      nrnody=0                  !number of particle types without decay
                                !(if 0 (default) : all unstable
                                !particles decay (at the end only
                                !(anti)nucleons, (anti)electrons and
                                !muons)
c Particle code is given as
c     id=+/-ijkl
c
c          mesons--
c          i=0, j<=k, +/- is sign for j
c          id=110 for pi0, id=220 for eta, etc.
c
c          baryons--
c          i<=j<=k in general
c          j<i<k for second state antisymmetric in (i,j), eg. l = 2130
c
c          other--
c          id=1,...,6 for quarks
c          id=9 for gluon
c          id=10 for photon
c          id=11,...,16 for leptons
c          i=17 for deuteron
c          i=18 for triton
c          i=19 for alpha
c          id=20 for ks, id=-20 for kl
c
c          i=21...26 for scalar quarks
c          i=29 for gluino
c
c          i=30 for h-dibaryon
c
c          i=31...36 for scalar leptons
c          i=39 for wino
c          i=40 for zino
c
c          id=80 for w+
c          id=81,...,83 for higgs mesons (h0, H0, A0, H+)
c          id=84,...,87 for excited bosons (Z'0, Z''0, W'+)
c          id=90 for z0
c
c          diquarks--
c          id=+/-ij00, i<j for diquark composed of i,j.
c
c Examples : 2130 = lambda, 1330=xi0, 2330=xi-, 3331=omega
c
c Conversion from epos to pdg code can be done using
c      id_pdg=idtrafo('nxs','pdg',id_epos)


c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=120     !pi+
c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=-120    !pi-
c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=130     !K+
c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=-130    !K-
c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=-20     !Kl
c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=-14     !mu+
c$$$      nrnody=nrnody+1
c$$$      nody(nrnody)=14      !mu-

c      nrnody=nrnody+1
c      nody(nrnody)=idtrafo('pdg','nxs',3122)    !lambda using pdg code

      isigma=0                  !do not print out the cross section on screen
      iecho=0                   !"silent" reading mode

c     Debug
      ish=0                     !debug level
      ifch=6                    !debug output (screen)
c     ifch=31    !debug output (file)
c     fnch="epos.debug"
c     nfnch=index(fnch,' ')-1
c     open(ifcx,file=fnch(1:nfnch),status='unknown')

      nevent=iEvent              !number of events
      modsho = 100000000                !printout every modsho events

      ecms=sngl(iecms)          !center of mass energy in GeV/c2
c     pnll=pproj                !beam momentum GeV/c

      infragm=2                 !nuclear fragmentation (realistic)

c     Projecticle definitions
      if (abs(ipart) .eq. 1) then
c     proton
         idprojin = sign(1120,ipart) !proton
         laproj =  sign(1,ipart)   !proj Z
         maproj = 1                !proj A
      elseif (ipart .eq. 12) then
c     carbon
         idprojin=1120
         laproj = 6             !proj Z
         maproj = 12            !proj A
      elseif (ipart .eq. 208) then
c     lead
         idprojin=1120
         laproj = 82            !proj Z
         maproj = 208           !proj A
      elseif (ipart .eq. 120) then
c     pi+
         idprojin = ipart         !pi+
         laproj = -1            !proj Z
         maproj = 1             !proj A
      elseif (ipart .eq. -120) then
c     pi-
         idprojin = ipart         !pi-
         laproj = -1            !proj Z
         maproj = 1             !proj A
c nuclei
      elseif (ipart.gt.10000)then
         idprojin=1120
         maproj=mod(ipart,10000)/10           !proj A
         laproj=mod(ipart,10000000)/10000     !proj Z
c PDG
      else
        idprojin=idtrafo('pdg','nxs',ipart)
        laproj = -1             !proj Z
        maproj = 1              !proj A
      endif

      if(idprojin.eq.99)then
         print *,'Warning : projectile particle not known : ',ipart
         print *,'          id particle must be +/-120(pi+/-)'
         print *,'          1(proton) 12(carbon) 208(lead) or PDG'
         stop
      endif


c     Target definitions : for nucleons, idtarg does not exist
c     Mass number matarg as well as charge, latarg, must be defined

c     idtarg = 1120             !proton
      if ( itarg .eq. 1 ) then
c     proton
         idtargin = sign(1120,itarg)
         latarg = sign(1,itarg) !targ Z
         matarg = 1             !targ A
      elseif ( itarg .eq. 12 ) then
c     carbon
         idtargin=1120
         latarg = 6             !targ Z
         matarg = 12            !targ A
      elseif (ipart .eq. 120) then
c     pi+
         idprojin = ipart         !pi+
         laproj = -1            !proj Z
         maproj = 1             !proj A
      elseif (ipart .eq. -120) then
c     pi-
         idprojin = ipart         !pi-
         laproj = -1            !proj Z
         maproj = 1             !proj A
      elseif ( itarg .eq. 208 ) then
c     lead
         idtargin=1120
         latarg = 82            !targ Z
         matarg = 208           !targ A
c nuclei
      elseif (itarg.gt.10000)then
         idtargin=1120
         matarg=mod(itarg,10000)/10          !targ A
         latarg=mod(itarg,10000000)/10000    !targ Z
c PDG
      elseif (abs(itarg).eq.2112.or.abs(itarg).eq.2212)then
        idtargin=idtrafo('pdg','nxs',itarg)
        latarg = -1             !targ Z
        matarg = 1              !targ A
      else
         print *,'Warning : target particle not known : ',itarg
         print *,'          id particle must be +/-120(pi+/-)'
         print *,'          1(proton) 12(carbon) 208(lead) or PDG'
         stop

      endif

      if ( model.eq.1 ) then   !model variable has no eposLHC
         istmax = 1             !final and mother particles (istmax=1 includes
                                !mother particles)
      else
         istmax=0
      endif

      end

c-----------------------------------------------------------------------
      subroutine lhesave(n)
c-----------------------------------------------------------------------
c     writes the results of a simulation into the file with unit ifdt
c     contains a description of the stored variables.
c     use Les Houches Event File as defined in hep-ph/0109068 for the
c     common block and hep-ph/0609017 for the XML output.
c     some code taken from example from Torbjrn Sjstrand
c     in http://www.thep.lu.se/~torbjorn/lhef
c-----------------------------------------------------------------------
      include 'epos.inc'
 
      integer id
      real taugm
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=nmxhep)  !extend array for file production
c      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
      SAVE /HEPEUP/


C...set event info and get number of particles.
      NUP=nhep             !number of particles
      IDPRUP=nint(typevt)  !type of event (ND,DD,SD)
      XWGTUP=1d0           !weight of event
      SCALUP=-1d0          !scale for PDF (not used)
      AQEDUP=-1d0          !alpha QED (not relevant)
      AQCDUP=-1d0          !alpha QCD (not relevant)

C...Copy event lines, omitting trailing blanks. 
C...Embed in <event> ... </event> block.
      write(ifdt,'(A)') '<event>' 
      write(ifdt,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      DO 220 i=1,nhep

c  store particle variables:
          IDUP(i)=idhep(i)
          if(isthep(i).eq.4)then
          ISTUP(i)=-9      !incoming particle
          else
          ISTUP(i)=isthep(i) !in LHEF:1=final, 2=decayed
          endif
          MOTHUP(1,i)=jmohep(1,i)
          MOTHUP(2,i)=jmohep(2,i)
          ICOLUP(1,i)=0        !color flow
          ICOLUP(2,i)=0        !color flow
          do J=1,5                !particle momentum (GeV/c)
            PUP(J,i)=phep(J,i)
          enddo
          id=idtrafo('pdg','nxs',idhep(i))
          call idtau(id,sngl(phep(4,i)),sngl(phep(5,i)),taugm)
          VTIMUP(i)=dble(taugm*(-alog(rangen())))*1d-12 !life time c*tau in mm
          if(VTIMUP(i).gt.dble(ainfin)
     &   .or.VTIMUP(i).ne.VTIMUP(i))VTIMUP(i)=ainfin
          SPINUP(i)=9           !polarization (not known)
          write(ifdt,*)IDUP(i),ISTUP(i),
     &      MOTHUP(1,i),MOTHUP(2,i),ICOLUP(1,i),ICOLUP(2,i),
     &      (PUP(J,i),J=1,5),VTIMUP(i),SPINUP(i)
  220 CONTINUE

c optional informations
      write(ifdt,*)'#geometry',bimevt,phievt

      write(ifdt,'(A)') '</event>' 

      if(n.eq.nevent)then
C...Successfully reached end of event loop: write closing tag
        write(ifdt,'(A)') '</LesHouchesEvents>' 
        write(ifdt,'(A)') ' ' 
        close(ifdt)
      endif

      return
      end


c-----------------------------------------------------------------------
      subroutine EposOutput(iFile)
c-----------------------------------------------------------------------
c Use EPOS to create lhe file output
c-----------------------------------------------------------------------
      include "epos.inc"
      character*1000 iFile

      istore=4
      nfndt=index(iFile,'.lhe')+4   !'.lhe' extension added in bstora
      fndt(1:nfndt)=iFile(1:nfndt)
      kdtopen=0

      call bstora

c      nopen=0
c      ifop=52
c      open(unit=ifop,file=TRIM(iParam),status='old')
c      call aread
c      close(ifop)
      end

c-----------------------------------------------------------------------
      subroutine EposInput(iParam)
c-----------------------------------------------------------------------
c     Read informations (new options or parameter change) in the file
c     "epos.param". The unit "ifop" is used in aread. If not used, it will
c     use the default value of all parameters.
c-----------------------------------------------------------------------
      include "epos.inc"
      character*1000 iParam

      nopen=0
      ifop=52
      open(unit=ifop,file=TRIM(iParam),status='old')
      call aread
      close(ifop)
      end


c-----------------------------------------------------------------------
      subroutine utLob5dbl(yboost,x1,x2,x3,x4,x5)
c-----------------------------------------------------------------------
c     Same as utlob5 but in double precision
c-----------------------------------------------------------------------
      implicit none
      double precision yboost,y,amt,x1,x2,x3,x4,x5
      amt=dsqrt(x5**2+x1**2+x2**2)
      y=dsign(1D0,x3)*dlog((x4+dabs(x3))/amt)
      y=y-yboost
      x4=amt*dcosh(y)
      x3=amt*dsinh(y)
      return
      end

c-----------------------------------------------------------------------
      subroutine crmc_xsection_f(xsigtot,xsigine,xsigela,xsigdd,xsigsd
     &                          ,xsloela,xsigtotaa,xsigineaa,xsigelaaa)
c-----------------------------------------------------------------------
c     Same as utlob5 but in double precision
c-----------------------------------------------------------------------
      implicit none
      include 'epos.inc'
      double precision xsigtot,xsigine,xsigela,xsigdd,xsigsd
     &                ,xsloela,xsigtotaa,xsigineaa,xsigelaaa

      xsigtot   = dble( sigtot   )
      xsigine   = dble( sigine   )
      xsigela   = dble( sigela   )
      xsigdd    = dble( sigdd    )
      xsigsd    = dble( sigsd    )
      xsloela   = dble( sloela   )
c Nuclear cross section only if needed
      xsigtotaa = 0d0
      xsigineaa = 0d0
      xsigelaaa = 0d0
      if(maproj.gt.1.or.matarg.gt.1)then
        if(model.eq.1)then
          call crseaaEpos(sigtotaa,sigineaa,sigcutaa,sigelaaa)
        else
          call crseaaModel(sigtotaa,sigineaa,sigcutaa,sigelaaa)
        endif
        xsigtotaa = dble( sigtotaa )
        xsigineaa = dble( sigineaa )
        xsigelaaa = dble( sigelaaa )
      endif

      return
      end

C c-----------------------------------------------------------------------
C       function rangen()
C c-----------------------------------------------------------------------
C c     generates a random number
C c-----------------------------------------------------------------------
C       include 'epos.inc'
C       double precision dranf
C  1    rangen=sngl(dranf(dble(irandm)))
C       if(rangen.le.0.)goto 1
C       if(rangen.ge.1.)goto 1
C       if(irandm.eq.1)write(ifch,*)'rangen()= ',rangen

C       return
C       end

C c-----------------------------------------------------------------------
C       double precision function drangen(dummy)
C c-----------------------------------------------------------------------
C c     generates a random number
C c-----------------------------------------------------------------------
C       include 'epos.inc'
C       double precision dummy,dranf
C       drangen=dranf(dummy)
C       if(irandm.eq.1)write(ifch,*)'drangen()= ',drangen

C       return
C       end
C c-----------------------------------------------------------------------
C       function cxrangen(dummy)
C c-----------------------------------------------------------------------
C c     generates a random number
C c-----------------------------------------------------------------------
C       include 'epos.inc'
C       double precision dummy,dranf
C       cxrangen=sngl(dranf(dummy))
C       if(irandm.eq.1)write(ifch,*)'cxrangen()= ',cxrangen

C       return
C       end



c Random number generator from CORSIKA *********************************




C=======================================================================

      DOUBLE PRECISION FUNCTION DRANF(dummy)

C-----------------------------------------------------------------------
C  RAN(DOM  NUMBER) GEN(ERATOR) USED IN EPOS
C  If calling this function within a DO-loop
C  you should use an argument which prevents (dummy) to draw this function
C  outside the loop by an optimizing compiler.
C-----------------------------------------------------------------------
      implicit none
      common/eporansto2/irndmseq
      integer irndmseq
      double precision uni(1),dummy
C-----------------------------------------------------------------------

      call RMMARD( uni,1,irndmseq)

      DRANF = UNI(1)
      UNI(1) = dummy        !to avoid warning

      RETURN
      END


c-----------------------------------------------------------------------
      subroutine ranfgt(seed)
c-----------------------------------------------------------------------
c Initialize seed in EPOS : read seed (output)
c Since original output seed and EPOS seed are different,
c define output seed as : seed=ISEED(3)*1E9+ISEED(2)
c but only for printing. Important values stored in /eporansto/
c Important : to be call before ranfst
c-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER          KSEQ
      PARAMETER        (KSEQ = 2)
      COMMON /CRRANMA3/CD,CINT,CM,TWOM24,TWOM48,MODCNS
      DOUBLE PRECISION CD,CINT,CM,TWOM24,TWOM48
      INTEGER          MODCNS
      COMMON /CRRANMA4/C,U,IJKL,I97,J97,NTOT,NTOT2,JSEQ
      DOUBLE PRECISION C(KSEQ),U(97,KSEQ)
      INTEGER          IJKL(KSEQ),I97(KSEQ),J97(KSEQ),
     *                 NTOT(KSEQ),NTOT2(KSEQ),JSEQ
      common/eporansto/diu0(100),iiseed(3)
      double precision    seed,diu0
      integer iiseed,i

      iiseed(1)=IJKL(1)
      iiseed(2)=NTOT(1)
      iiseed(3)=NTOT2(1)
      seed=dble(iiseed(3))*dble(MODCNS)+dble(iiseed(2))
      diu0(1)=C(1)
      do i=2,98
        diu0(i)=U(i-1,1)
      enddo
      diu0(99)=dble(I97(1))
      diu0(100)=dble(J97(1))
      return
      end

c-----------------------------------------------------------------------
      subroutine ranfst(seed)
c-----------------------------------------------------------------------
c Initialize seed in EPOS :  restore seed (input)
c Since original output seed and EPOS seed are different,
c define output seed as : seed=ISEED(3)*1E9+ISEED(2)
c but only for printing. Important values restored from /eporansto/
c Important : to be call after ranfgt
c-----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER          KSEQ
      PARAMETER        (KSEQ = 2)
      COMMON /CRRANMA3/CD,CINT,CM,TWOM24,TWOM48,MODCNS
      DOUBLE PRECISION CD,CINT,CM,TWOM24,TWOM48
      INTEGER          MODCNS
      COMMON /CRRANMA4/C,U,IJKL,I97,J97,NTOT,NTOT2,JSEQ
      DOUBLE PRECISION C(KSEQ),U(97,KSEQ)
      INTEGER          IJKL(KSEQ),I97(KSEQ),J97(KSEQ),
     *                 NTOT(KSEQ),NTOT2(KSEQ),JSEQ
      common/eporansto/diu0(100),iiseed(3)
      double precision    seedi,seed,diu0
      integer i,iiseed

      seedi=seed
      IJKL(1)=iiseed(1)
      NTOT(1)=iiseed(2)
      NTOT2(1)=iiseed(3)
      C(1)=diu0(1)
      do i=2,98
        U(i-1,1)=diu0(i)
      enddo
      I97(1)=nint(diu0(99))
      J97(1)=nint(diu0(100))
      return
      end

c-----------------------------------------------------------------------
      subroutine ranflim(seed)
c-----------------------------------------------------------------------
      double precision seed
      if(seed.gt.1d9)stop'seed larger than 1e9 not possible !'
      end

c-----------------------------------------------------------------------
      subroutine ranfcv(seed)
c-----------------------------------------------------------------------
c Convert input seed to EPOS random number seed
c Since input seed and EPOS (from Corsika) seed are different,
c define input seed as : seed=ISEED(3)*1E9+ISEED(2)
c-----------------------------------------------------------------------
      IMPLICIT NONE
      COMMON /CRRANMA3/CD,CINT,CM,TWOM24,TWOM48,MODCNS
      DOUBLE PRECISION CD,CINT,CM,TWOM24,TWOM48
      INTEGER          MODCNS
      common/eporansto/diu0(100),iiseed(3)
      double precision    seed,diu0
      integer iiseed

      iiseed(3)=nint(seed/dble(MODCNS))
      iiseed(2)=nint(mod(seed,dble(MODCNS)))

      return
      end

c-----------------------------------------------------------------------
      subroutine ranfini(seed,iseq,iqq)
c-----------------------------------------------------------------------
c Initialize random number sequence iseq with seed
c if iqq=-1, run first ini
c    iqq=0 , set what sequence should be used
c    iqq=1 , initialize sequence for initialization
c    iqq=2 , initialize sequence for first event
c-----------------------------------------------------------------------
      IMPLICIT NONE
      COMMON /CRRANMA3/CD,CINT,CM,TWOM24,TWOM48,MODCNS
      DOUBLE PRECISION CD,CINT,CM,TWOM24,TWOM48
      INTEGER          MODCNS
      common/eporansto/diu0(100),iiseed(3)
      double precision    seed,diu0
      integer iiseed
      common/eporansto2/irndmseq
      integer irndmseq
      integer iseed(3),iseq,iqq,iseqdum

      if(iqq.eq.0)then
        irndmseq=iseq
      elseif(iqq.eq.-1)then
        iseqdum=0
        call RMMAQD(iseed,iseqdum,'R')   !first initialization
      elseif(iqq.eq.2)then
        irndmseq=iseq
        if(seed.ge.dble(MODCNS))then
           write(*,'(a,1p,e8.1)')'seedj larger than',dble(MODCNS)
           stop 'Forbidden !'
        endif
        iiseed(1)=nint(mod(seed,dble(MODCNS)))
c iiseed(2) and iiseed(3) defined in aread
        call RMMAQD(iiseed,iseq,'S') !initialize random number generator
      elseif(iqq.eq.1)then        !dummy sequence for EPOS initialization
        irndmseq=iseq
        if(seed.ge.dble(MODCNS))then
           write(*,'(a,1p,e8.1)')'seedi larger than',dble(MODCNS)
           stop 'Forbidden !'
        endif
        iseed(1)=nint(mod(seed,dble(MODCNS)))
        iseed(2)=0
        iseed(3)=0
        call RMMAQD(iseed,iseq,'S') !initialize random number generator
      endif
      return
      end

C=======================================================================

      SUBROUTINE RMMARD( RVEC,LENV,ISEQ )

C-----------------------------------------------------------------------
C  C(ONE)X
C  R(ANDO)M (NUMBER GENERATOR OF) MAR(SAGLIA TYPE) D(OUBLE PRECISION)
C
C  THESE ROUTINES (RMMARD,RMMAQD) ARE MODIFIED VERSIONS OF ROUTINES
C  FROM THE CERN LIBRARIES. DESCRIPTION OF ALGORITHM SEE:
C               http://consult.cern.ch/shortwrups/v113/top.html
C  IT HAS BEEN CHECKED THAT RESULTS ARE BIT-IDENTICAL WITH CERN
C  DOUBLE PRECISION RANDOM NUMBER GENERATOR RMM48, DESCRIBED IN
C               http://consult.cern.ch/shortwrups/v116/top.html
C  ARGUMENTS:
C   RVEC   = DOUBLE PREC. VECTOR FIELD TO BE FILLED WITH RANDOM NUMBERS
C   LENV   = LENGTH OF VECTOR (# OF RANDNUMBERS TO BE GENERATED)
C   ISEQ   = # OF RANDOM SEQUENCE
C
C  VERSION OF D. HECK FOR DOUBLE PRECISION RANDOM NUMBERS.
C  ADAPTATION  : T. PIEROG    IK  FZK KARLSRUHE FROM D. HECK VERSION
C  DATE     : Feb  17, 2009
C-----------------------------------------------------------------------

      IMPLICIT NONE
      INTEGER          KSEQ
      PARAMETER        (KSEQ = 2)
      COMMON /CRRANMA3/CD,CINT,CM,TWOM24,TWOM48,MODCNS
      DOUBLE PRECISION CD,CINT,CM,TWOM24,TWOM48
      INTEGER          MODCNS
      COMMON /CRRANMA4/C,U,IJKL,I97,J97,NTOT,NTOT2,JSEQ
      DOUBLE PRECISION C(KSEQ),U(97,KSEQ),UNI
      INTEGER          IJKL(KSEQ),I97(KSEQ),J97(KSEQ),
     *                 NTOT(KSEQ),NTOT2(KSEQ),JSEQ

      DOUBLE PRECISION RVEC(*)
      INTEGER          ISEQ,IVEC,LENV
      SAVE

C-----------------------------------------------------------------------

      IF ( ISEQ .GT. 0  .AND.  ISEQ .LE. KSEQ ) JSEQ = ISEQ

      DO   IVEC = 1, LENV
        UNI = U(I97(JSEQ),JSEQ) - U(J97(JSEQ),JSEQ)
        IF ( UNI .LT. 0.D0 ) UNI = UNI + 1.D0
        U(I97(JSEQ),JSEQ) = UNI
        I97(JSEQ)  = I97(JSEQ) - 1
        IF ( I97(JSEQ) .EQ. 0 ) I97(JSEQ) = 97
        J97(JSEQ)  = J97(JSEQ) - 1
        IF ( J97(JSEQ) .EQ. 0 ) J97(JSEQ) = 97
        C(JSEQ)    = C(JSEQ) - CD
        IF ( C(JSEQ) .LT. 0.D0 ) C(JSEQ)  = C(JSEQ) + CM
        UNI        = UNI - C(JSEQ)
        IF ( UNI .LT. 0.D0 ) UNI = UNI + 1.D0
C  AN EXACT ZERO HERE IS VERY UNLIKELY, BUT LET'S BE SAFE.
        IF ( UNI .EQ. 0.D0 ) UNI = TWOM48
        RVEC(IVEC) = UNI
      ENDDO

      NTOT(JSEQ) = NTOT(JSEQ) + LENV
      IF ( NTOT(JSEQ) .GE. MODCNS )  THEN
        NTOT2(JSEQ) = NTOT2(JSEQ) + 1
        NTOT(JSEQ)  = NTOT(JSEQ) - MODCNS
      ENDIF

      RETURN
      END

C=======================================================================

      SUBROUTINE RMMAQD( ISEED, ISEQ, CHOPT )

C-----------------------------------------------------------------------
C  R(ANDO)M (NUMBER GENERATOR OF) MA(RSAGLIA TYPE INITIALIZATION) DOUBLE
C
C  SUBROUTINE FOR INITIALIZATION OF RMMARD
C  THESE ROUTINE RMMAQD IS A MODIFIED VERSION OF ROUTINE RMMAQ FROM
C  THE CERN LIBRARIES. DESCRIPTION OF ALGORITHM SEE:
C               http://consult.cern.ch/shortwrups/v113/top.html
C  FURTHER DETAILS SEE SUBR. RMMARD
C  ARGUMENTS:
C   ISEED  = SEED TO INITIALIZE A SEQUENCE (3 INTEGERS)
C   ISEQ   = # OF RANDOM SEQUENCE
C   CHOPT  = CHARACTER TO STEER INITIALIZE OPTIONS
C
C  CERN PROGLIB# V113    RMMAQ           .VERSION KERNFOR  1.0
C  ORIG. 01/03/89 FCA + FJ
C  ADAPTATION  : T. PIEROG    IK  FZK KARLSRUHE FROM D. HECK VERSION
C  DATE     : Feb  17, 2009
C-----------------------------------------------------------------------

      IMPLICIT NONE
      INTEGER          KSEQ
      PARAMETER        (KSEQ = 2)
      COMMON /CRRANMA3/CD,CINT,CM,TWOM24,TWOM48,MODCNS
      DOUBLE PRECISION CD,CINT,CM,TWOM24,TWOM48
      INTEGER          MODCNS
      COMMON /CRRANMA4/C,U,IJKL,I97,J97,NTOT,NTOT2,JSEQ
      DOUBLE PRECISION C(KSEQ),U(97,KSEQ),UNI
      INTEGER          IJKL(KSEQ),I97(KSEQ),J97(KSEQ),
     *                 NTOT(KSEQ),NTOT2(KSEQ),JSEQ

      DOUBLE PRECISION CC,S,T,UU(97)
      INTEGER          ISEED(3),I,IDUM,II,II97,IJ,IJ97,IORNDM,
     *                 ISEQ,J,JJ,K,KL,L,LOOP2,M,NITER
      CHARACTER        CHOPT*(*), CCHOPT*12
      LOGICAL          FIRST
      SAVE
      DATA             FIRST / .TRUE. /, IORNDM/11/, JSEQ/1/


C-----------------------------------------------------------------------

      IF ( FIRST ) THEN
        TWOM24 = 2.D0**(-24)
        TWOM48 = 2.D0**(-48)
        CD     = 7654321.D0*TWOM24
        CM     = 16777213.D0*TWOM24
        CINT   = 362436.D0*TWOM24
        MODCNS = 1000000000
        FIRST  = .FALSE.
        JSEQ   = 1
      ENDIF
      CCHOPT = CHOPT
      IF ( CCHOPT .EQ. ' ' ) THEN
        ISEED(1) = 54217137
        ISEED(2) = 0
        ISEED(3) = 0
        CCHOPT   = 'S'
        JSEQ     = 1
      ENDIF

      IF     ( INDEX(CCHOPT,'S') .NE. 0 ) THEN
        IF ( ISEQ .GT. 0  .AND.  ISEQ .LE. KSEQ ) JSEQ = ISEQ
        IF ( INDEX(CCHOPT,'V') .NE. 0 ) THEN
          READ(IORNDM,'(3Z8)') IJKL(JSEQ),NTOT(JSEQ),NTOT2(JSEQ)
          READ(IORNDM,'(2Z8,Z16)') I97(JSEQ),J97(JSEQ),C(JSEQ)
          READ(IORNDM,'(24(4Z16,/),Z16)') U
          IJ = IJKL(JSEQ)/30082
          KL = IJKL(JSEQ) - 30082 * IJ
          I  = MOD(IJ/177, 177) + 2
          J  = MOD(IJ, 177)     + 2
          K  = MOD(KL/169, 178) + 1
          L  = MOD(KL, 169)
          CD =  7654321.D0 * TWOM24
          CM = 16777213.D0 * TWOM24
        ELSE
          IJKL(JSEQ)  = ISEED(1)
          NTOT(JSEQ)  = ISEED(2)
          NTOT2(JSEQ) = ISEED(3)
          IJ = IJKL(JSEQ) / 30082
          KL = IJKL(JSEQ) - 30082*IJ
          I  = MOD(IJ/177, 177) + 2
          J  = MOD(IJ, 177)     + 2
          K  = MOD(KL/169, 178) + 1
          L  = MOD(KL, 169)
          DO   II = 1, 97
            S = 0.D0
            T = 0.5D0
            DO   JJ = 1, 48
              M = MOD(MOD(I*J,179)*K, 179)
              I = J
              J = K
              K = M
              L = MOD(53*L+1, 169)
              IF ( MOD(L*M,64) .GE. 32 ) S = S + T
              T = 0.5D0 * T
            ENDDO
            UU(II) = S
          ENDDO
          CC    = CINT
          II97  = 97
          IJ97  = 33
C  COMPLETE INITIALIZATION BY SKIPPING (NTOT2*MODCNS+NTOT) RANDOMNUMBERS
          NITER = MODCNS
          DO   LOOP2 = 1, NTOT2(JSEQ)+1
            IF ( LOOP2 .GT. NTOT2(JSEQ) ) NITER = NTOT(JSEQ)
            DO   IDUM = 1, NITER
              UNI = UU(II97) - UU(IJ97)
              IF ( UNI .LT. 0.D0 ) UNI = UNI + 1.D0
              UU(II97) = UNI
              II97     = II97 - 1
              IF ( II97 .EQ. 0 ) II97 = 97
              IJ97     = IJ97 - 1
              IF ( IJ97 .EQ. 0 ) IJ97 = 97
              CC       = CC - CD
              IF ( CC .LT. 0.D0 ) CC  = CC + CM
            ENDDO
          ENDDO
          I97(JSEQ) = II97
          J97(JSEQ) = IJ97
          C(JSEQ)   = CC
          DO   JJ = 1, 97
            U(JJ,JSEQ) = UU(JJ)
          ENDDO
        ENDIF
      ELSEIF ( INDEX(CCHOPT,'R') .NE. 0 ) THEN
        IF ( ISEQ .GT. 0 ) THEN
          JSEQ = ISEQ
        ELSE
          ISEQ = JSEQ
        ENDIF
        IF ( INDEX(CCHOPT,'V') .NE. 0 ) THEN
          WRITE(IORNDM,'(3Z8)') IJKL(JSEQ),NTOT(JSEQ),NTOT2(JSEQ)
          WRITE(IORNDM,'(2Z8,Z16)') I97(JSEQ),J97(JSEQ),C(JSEQ)
          WRITE(IORNDM,'(24(4Z16,/),Z16)') U
        ELSE
          ISEED(1) = IJKL(JSEQ)
          ISEED(2) = NTOT(JSEQ)
          ISEED(3) = NTOT2(JSEQ)
        ENDIF
      ENDIF

      RETURN
      END
