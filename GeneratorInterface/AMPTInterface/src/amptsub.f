c....................amptsub.f
c.....this file contains 4 sections:
c.....1. ART subroutines;
c.....2. ART functions;
c.....3. ART block data;
c.....4. subprocesses borrowed from other codes.
c.....5. the previous artana.f
c.....6. the previous zpcsub.f
c.....7. subroutine getnp
c.....Note that Parts1-4 are the previous artsub.f
c
c=======================================================================
c.....subroutine to set up ART parameters and analysis files
c.....before looping different events
cms
cms   dlw & gsfs 8/2009 commented out lots of output files
cms
      SUBROUTINE ARTSET
c
      PARAMETER (AMU= 0.9383)
      double precision dpcoal,drcoal,ecritl
      INTEGER ZTA, ZPR
      common  /gg/      dx,dy,dz,dpx,dpy,dpz
clin-10/03/03 
c     "SAVE   " (without argument) is used for most subroutines and functions,
c     this is important for the success when using "f77" to compile:
cc      SAVE /gg/
      common  /zz/      zta,zpr
cc      SAVE /zz/
      COMMON  /RUN/     NUM
cc      SAVE /RUN/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &   IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
      COMMON /INPUT3/ PLAB, ELAB, ZEROPT, B0, BI, BM, DENCUT, CYCBOX
cc      SAVE /INPUT3/
      common /imulst/ iperts
cc      SAVE /imulst/
      common /coal/dpcoal,drcoal,ecritl
      common/anim/nevent,isoft,isflag,izpc
      common /para7/ ioscar,nsmbbbar,nsmmeson
      SAVE   
clin-10/03/03  ecritl: local energy density below which a parton 
c     will freeze out (in GeV/fm^3), for improvements on string melting, 
c     not used in this version of AMPT:
clin-4/2008
c      data ecritl/1.d0/
      ecritl=1.d0
c
c     combine ART initialization into ampt.ini:
c     (Note that the following values are relics from the old ART structure)
c.....input parameter file
c      OPEN(13, FILE = 'art1.ini', STATUS = 'UNKNOWN')
c      READ (13, *) MASSTA, ZTA
      MASSTA=1
      ZTA=1
c      write(12,*) massta, zta, ' massta, zta'
c      READ (13, *) MASSPR, ZPR
      MASSPR=1
      ZPR=1
c      write(12,*) masspr, zpr, ' masspr, zpr'
c      READ (13, *) PLAB, IPLAB
      PLAB=14.6 
      IPLAB=2
c      write(12,*) plab, iplab, ' plab, iplab'
      if(iplab.eq.2)then
         elab=sqrt(plab**2+amu**2)-amu
      else
         elab=plab
      endif
      elab=elab*1000.
c      READ (13, *) ZEROPT
      ZEROPT=0.
c      write(12,*) zeropt, ' zeropt'
clin-10/03/03 ISEED was used as a seed for random number inside ART, 
c     not used in AMPT:
      ISEED=700721
c     0/1: (Normal or Perturbative) multistrange partice production.
c     Perturbative option is disabled for now:
      iperts=0
c      READ (13, *) MANYB, B0, BI, BM
c     2/04/00 MANYB MUST BE SET TO 1 !
c     in order to skip impact parameter setting by ART, then B0 has no effect.
      MANYB=1
      B0=1
      BI=0
      BM=0
c      write(12,*) manyb, b0, bi, bm, ' manyb, b0, bi, bm'
c      READ (13, *) ISEED
c      write(12,*) iseed, ' iseed'
c      READ (13, *) DT
c      write(12,*) dt, ' dt'
c      READ (13, *) NTMAX
c      write(12,*) ntmax, ' ntmax'
c      READ (13, *) ICOLL
      ICOLL=-1
c      write(12,*) icoll, ' icoll'
c      READ (13, *) NUM
c     2/11/03 run events without test particles for now:
      NUM=1
c      write(12,*) num, ' num'
c      READ (13, *) INSYS
      INSYS=1
c      write(12,*) insys, ' insys'
c      READ (13, *) IPOT
      IPOT=3
c      write(12,*) ipot, ' ipot'
c      READ (13, *) MODE
      MODE=0
      IF(ICOLL.EQ.-1)IPOT=0
c      write(12,*) mode, ' mode'
c      READ (13, *) DX, DY, DZ
      DX=2.73
      DY=2.73
      DZ=2.73
c      write(12,*) dx,dy,dz,' dx,dy,dz'
c      READ (13, *) DPX, DPY, DPZ
      DPX=0.6
      DPY=0.6
      DPZ=0.6
c      write(12,*) dpx,dpy,dpz,' dpx,dpy,dpz'
c      READ (13, *) IAVOID
      IAVOID=1
c      write(12,*) iavoid, ' iavoid'
c      READ (13, *) IMOMEN
      IMOMEN=1
c      write(12,*) imomen, ' imomen'
      if(icoll.eq.-1)imomen=3
c      READ (13, *) NFREQ
      NFREQ=10
c      write(12,*) nfreq, ' nfreq'
c      READ (13, *) ICFLOW
      ICFLOW=0
c      write(12,*) ICFLOW, ' ICFLOW'
c      READ (13, *) ICRHO
      ICRHO=0
c      write(12,*) ICRHO, ' ICRHO'
c      READ (13, *) ICOU
      ICOU=0
c      write(12,*)icou, ' icou'
* kaon potential control parameter
* KMUL IS A MULTIPLIER TO THE STANDARD K-N SCATTERING LENGTH
c      READ (13, *) KPOTEN, KMUL
      KPOTEN=0
      KMUL=1
c      write(12,*)kpoten,kmul, ' kpoten, kmul'
* mean field control parameter FOR BARYONS
* no mean filed is used for baryons if their 
* local density is higher than dencut. 
c      READ (13, *) DENCUT
      DENCUT=15
c      write(12,*)dencut, ' dencut'
* test reactions in a box of side-length cycbox
* input cycbox
c      READ (13, *) CYCBOX
      CYCBOX=0
c      write(12,*) cycbox, ' cycbox'
c
clin-5b/2008
c      if(ioscar.eq.2) then
      if(ioscar.eq.2.or.ioscar.eq.3) then
cms      OPEN (92,FILE='ana/parton-initial-afterPropagation.dat',
cms  1        STATUS = 'UNKNOWN')
      endif
      if(ioscar.eq.3) then
clin-6/2009 write out full parton collision history:
cms      OPEN (95,FILE='ana/parton-collisionsHistory.dat',
cms  1        STATUS='UNKNOWN')
clin-6/2009 write out initial minijet information:
cms      OPEN (96,FILE='ana/minijet-initial-beforePropagation.dat',
cms  1        STATUS='UNKNOWN')
clin-6/2009 write out parton info after coalescence:
         if(isoft.eq.4.or.isoft.eq.5) then
cms         OPEN (85,FILE='ana/parton-after-coalescence.dat',
cms  1           STATUS='UNKNOWN')
         endif
      endif
clin-6/2009 write out initial transverse positions of initial nucleons:
cms   OPEN (94,FILE='ana/npart-xy.dat',STATUS='UNKNOWN')

      RETURN
      END

c-----------------------------------------------------------------------

c.....subroutine to initialize cascade.

      SUBROUTINE ARINI

c.....before invoking ARINI:
c.....IAPAR2(1), IAINT2(1) must be set.
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/
      SAVE   

ctest off for resonance (phi, K*) studies:
c      OPEN (89, FILE = 'ana/decay_rec.dat', STATUS = 'UNKNOWN')

      IFLG = IAPAR2(1)
      GOTO (200, 200, 300) IFLG

c.....error choice of initialization
      PRINT *, 'IAPAR2(1) must be 1, 2, or 3'
      STOP

c.....to use default initial conditions generated by the cascade,
c.....or to read in initial conditions.
 200  RETURN

c.....to generate formation time and the position at formation time from 
c.....read-in initial conditions with an averaged formation proper time.
 300  CALL ARINI1
c.....ordering the particle label according to increasing order of 
c.....formation time.
      CALL ARTORD
      RETURN

      END

c-----------------------------------------------------------------------

c.....subroutine to generate formation time and position at formation time
c.....from read-in initial conditions with an averaged formation proper 
c.....time.

      SUBROUTINE ARINI1

c.....before invoking ARINI1:
c.....ARPAR1(1), IAINT2(1) must be set:
      PARAMETER (MAXSTR=150001)
      double precision  smearp,smearh

      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
cc      SAVE /ARPRC/
      COMMON /smearz/smearp,smearh
cc      SAVE /smearz/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
      common /nzpc/nattzp
cc      SAVE /nzpc/
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      common /para8/ idpert,npertd,idxsec
      SAVE   
clin-5/2008 for perturbatively-produced hadrons (currently only deuterons):
cms   OPEN (91, FILE = 'ana/deuteron_processes.dat', 
cms  1     STATUS = 'UNKNOWN')
      if(idpert.eq.1.or.idpert.eq.2) then
cms      OPEN (90, FILE = 'ana/ampt_pert.dat', STATUS = 'UNKNOWN')
      endif
c.....generate formation time and position at formation time.
      TAU0 = ARPAR1(1)
      NP = IAINT2(1)
clin-7/10/01     initial positions already given for hadrons 
c     formed from partons inside ZPC (from string melting):
      if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) then
         if(NP.le.nattzp) return
         do 1001 I = nattzp+1, NP
            IF (ABS(PZAR(I)) .GE. PEAR(I)) THEN
               PRINT *, ' IN ARINI1'
               PRINT *, 'ABS(PZ) .GE. EE for particle ', I
               PRINT *, ' FLAV = ', ITYPAR(I), ' PX = ', PXAR(I), 
     &              ' PY = ', PYAR(I)
               PRINT *, ' PZ = ', PZAR(I), ' EE = ', PEAR(I)
               PRINT *, ' XM = ', XMAR(I)
               RAP = 1000000.0
               GOTO 50
            END IF
            RAP = 0.5 * LOG((PEAR(I) + PZAR(I)) / (PEAR(I) - PZAR(I)))
 50         CONTINUE
            VX = PXAR(I) / PEAR(I)
            VY = PYAR(I) / PEAR(I)
            FTAR(I) = TAU0 * COSH(RAP)
            GXAR(I) = GXAR(I) + VX * FTAR(I)
            GYAR(I) = GYAR(I) + VY * FTAR(I)
            GZAR(I) = TAU0 * SINH(RAP)
clin-5/2009 No formation time for spectator projectile or target nucleons:
            if(PXAR(I).eq.0.and.PYAR(I).eq.0
     1           .and.(PEAR(I)*2/HINT1(1)).gt.0.99
     2           .and.(ITYPAR(I).eq.2112.or.ITYPAR(I).eq.2212)) then
               TAUI=1.E-20
               FTAR(I)=TAUI*COSH(RAP)
               GZAR(I)=TAUI*SINH(RAP)
            endif
 1001    continue
clin-7/10/01-end
clin-3/2009 cleanup of program flow:
      else
         DO 1002 I = 1, NP
            IF (ABS(PZAR(I)) .GE. PEAR(I)) THEN
               PRINT *, ' IN ARINI1'
               PRINT *, 'ABS(PZ) .GE. EE for particle ', I
               PRINT *, ' FLAV = ', ITYPAR(I), ' PX = ', PXAR(I), 
     &              ' PY = ', PYAR(I)
               PRINT *, ' PZ = ', PZAR(I), ' EE = ', PEAR(I)
               PRINT *, ' XM = ', XMAR(I)
               RAP = 1000000.0
               GOTO 100
c               STOP
            END IF
            RAP = 0.5 * LOG((PEAR(I) + PZAR(I)) / (PEAR(I) - PZAR(I)))
 100        CONTINUE
            VX = PXAR(I) / PEAR(I)
            VY = PYAR(I) / PEAR(I)
c.....give initial formation time shift
            TAUI = FTAR(I) + TAU0
            FTAR(I) = TAUI * COSH(RAP)
            GXAR(I) = GXAR(I) + VX * TAU0 * COSH(RAP)
            GYAR(I) = GYAR(I) + VY * TAU0 * COSH(RAP)
c     4/25/03: hadron z-position upon formation determined the same way as x,y:
            GZAR(I) = TAUI * SINH(RAP)
c     the old prescription:
c            GZAR(I) = GZAR(I) + TAU0 * SINH(RAP)
            zsmear=sngl(smearh)*(2.*RANART(NSEED)-1.)
            GZAR(I)=GZAR(I)+zsmear
cbz1/28/99end
c     10/05/01 no formation time for spectator projectile or target nucleons:
            if(PXAR(I).eq.0.and.PYAR(I).eq.0
     1           .and.(PEAR(I)*2/HINT1(1)).gt.0.99
     2           .and.(ITYPAR(I).eq.2112.or.ITYPAR(I).eq.2212)) then
clin-5/2008:
c               TAUI=0.00001
               TAUI=1.E-20
               FTAR(I)=TAUI*COSH(RAP)
               GZAR(I)=TAUI*SINH(RAP)+zsmear
            endif
 1002    CONTINUE
clin-3/2009 cleanup of program flow:
      endif

clin-3/2009 Add initial hadrons before the hadron cascade starts:
      call addhad

      RETURN
      END

c-----------------------------------------------------------------------

c.....subroutine to order particle labels according to increasing 
c.....formation time

      SUBROUTINE ARTORD

c.....before invoking ARTORD:
c.....IAINT2(1) must be set:
      PARAMETER (MAXSTR=150001,MAXR=1)
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
cc      SAVE /ARPRC/
clin-3/2009 Take care of particle weights when user inserts initial hadrons:
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      DIMENSION dptemp(MAXSTR)
c
      DIMENSION ITYP0(MAXSTR), 
     &   GX0(MAXSTR), GY0(MAXSTR), GZ0(MAXSTR), FT0(MAXSTR),
     &   PX0(MAXSTR), PY0(MAXSTR), PZ0(MAXSTR), EE0(MAXSTR),
     &   XM0(MAXSTR)
      DIMENSION INDX(MAXSTR)
      EXTERNAL ARINDX
      SAVE   
c
      NPAR = 0
      NP = IAINT2(1)
      DO 1001 I = 1, NP
         ITYP0(I) = ITYPAR(I)
         GX0(I) = GXAR(I)
         GY0(I) = GYAR(I)
         GZ0(I) = GZAR(I)
         FT0(I) = FTAR(I)
         PX0(I) = PXAR(I)
         PY0(I) = PYAR(I)
         PZ0(I) = PZAR(I)
         EE0(I) = PEAR(I)
         XM0(I) = XMAR(I)
clin-3/2009:
         dptemp(I) = dpertp(I)
 1001 CONTINUE
      CALL ARINDX(MAXSTR, NP, FT0, INDX)
      DO 1002 I = 1, NP
cbz12/3/98
c         IF (ITYP0(INDX(I)) .EQ. 211) THEN
c         IF (ITYP0(INDX(I)) .EQ. 211 .OR. ITYP0(INDX(I)) .EQ. 321) THEN
c         IF (ITYP0(INDX(I)) .EQ. 211 .OR. ITYP0(INDX(I)) .EQ. 2212 .OR.
c     &      ITYP0(INDX(I)) .EQ. 2112 .OR. ITYP0(INDX(I)) .EQ. -211 .OR.
c     &      ITYP0(INDX(I)) .EQ. 111) THEN
c         IF (ITYP0(INDX(I)) .EQ. 211 .OR. ITYP0(INDX(I)) .EQ. 2212 .OR.
c     &      ITYP0(INDX(I)) .EQ. 2112) THEN
         NPAR = NPAR + 1
c         ITYPAR(I) = ITYP0(INDX(I))
c         GXAR(I) = GX0(INDX(I))
c         GYAR(I) = GY0(INDX(I))
c         GZAR(I) = GZ0(INDX(I))
c         FTAR(I) = FT0(INDX(I))
c         PXAR(I) = PX0(INDX(I))
c         PYAR(I) = PY0(INDX(I))
c         PZAR(I) = PZ0(INDX(I))
c         PEAR(I) = EE0(INDX(I))
c         XMAR(I) = XM0(INDX(I))
         ITYPAR(NPAR) = ITYP0(INDX(I))
         GXAR(NPAR) = GX0(INDX(I))
         GYAR(NPAR) = GY0(INDX(I))
         GZAR(NPAR) = GZ0(INDX(I))
         FTAR(NPAR) = FT0(INDX(I))
         PXAR(NPAR) = PX0(INDX(I))
         PYAR(NPAR) = PY0(INDX(I))
         PZAR(NPAR) = PZ0(INDX(I))
         PEAR(NPAR) = EE0(INDX(I))
         XMAR(NPAR) = XM0(INDX(I))
clin-3/2009:
         dpertp(NPAR)=dptemp(INDX(I))
c         END IF
cbz12/3/98end
 1002 CONTINUE
      IAINT2(1) = NPAR
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....subroutine to copy individually generated particle record into
c.....particle record for many test particle runs.

      SUBROUTINE ARINI2(K)

      PARAMETER (MAXSTR=150001,MAXR=1)
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
cc      SAVE /ARPRC/
      COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cc      SAVE /ARPRC1/
      COMMON/tdecay/tfdcy(MAXSTR),tfdpi(MAXSTR,MAXR),tft(MAXSTR)
cc      SAVE /tdecay/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &     IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
      COMMON/RNDF77/NSEED
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
cc      SAVE /RNDF77/
      SAVE   

      MULTI1(K) = IAINT2(1)
      DO 1001 I = 1, MULTI1(K)
         ITYP1(I, K) = ITYPAR(I)
         GX1(I, K) = GXAR(I)
         GY1(I, K) = GYAR(I)
         GZ1(I, K) = GZAR(I)
         FT1(I, K) = FTAR(I)
         PX1(I, K) = PXAR(I)
         PY1(I, K) = PYAR(I)
         PZ1(I, K) = PZAR(I)
         EE1(I, K) = PEAR(I)
         XM1(I, K) = XMAR(I)
clin-3/2009 hadron weights are initialized in addhad():
clin-5/2008 all hadrons not perturbatively-produced have the weight of 1:
c         dpp1(I,K)=1.
         dpp1(I,K)=dpertp(I)
 1001 CONTINUE

c     initialize final time of each particle to ntmax*dt except for 
c     decay daughters, which have values given by tfdcy() and >(ntmax*dt):
      do 1002 ip=1,MAXSTR
         tfdcy(ip)=NTMAX*DT
         tft(ip)=NTMAX*DT
 1002 continue
c
      do 1004 irun=1,MAXR
         do 1003 ip=1,MAXSTR
            tfdpi(ip,irun)=NTMAX*DT
 1003    continue
 1004 continue

      RETURN
      END

c=======================================================================

c.....function to convert PDG flavor code into ART flavor code.

      FUNCTION IARFLV(IPDG)

      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

c.....anti-Delta-
      IF (IPDG .EQ. -1114) THEN
         IARFLV = -6
         RETURN
      END IF

c.....anti-Delta0
      IF (IPDG .EQ. -2114) THEN
         IARFLV = -7
         RETURN
      END IF

c.....anti-Delta+
      IF (IPDG .EQ. -2214) THEN
         IARFLV = -8
         RETURN
      END IF

c.....anti-Delta++
      IF (IPDG .EQ. -2224) THEN
         IARFLV = -9
         RETURN
      END IF

cbzdbg2/23/99
c.....anti-proton
      IF (IPDG .EQ. -2212) THEN
         IARFLV = -1
         RETURN
      END IF

c.....anti-neutron
      IF (IPDG .EQ. -2112) THEN
         IARFLV = -2
         RETURN
      END IF
cbzdbg2/23/99end

c.....eta
      IF (IPDG .EQ. 221) THEN
         IARFLV = 0
         RETURN
      END IF

c.....proton
      IF (IPDG .EQ. 2212) THEN
         IARFLV = 1
         RETURN
      END IF

c.....neutron
      IF (IPDG .EQ. 2112) THEN
         IARFLV = 2
         RETURN
      END IF

c.....pi-
      IF (IPDG .EQ. -211) THEN
         IARFLV = 3
         RETURN
      END IF

c.....pi0
      IF (IPDG .EQ. 111) THEN
         IARFLV = 4
         RETURN
      END IF

c.....pi+
      IF (IPDG .EQ. 211) THEN
         IARFLV = 5
         RETURN
      END IF

c.....Delta-
      IF (IPDG .EQ. 1114) THEN
         IARFLV = 6
         RETURN
      END IF

c.....Delta0
      IF (IPDG .EQ. 2114) THEN
         IARFLV = 7
         RETURN
      END IF

c.....Delta+
      IF (IPDG .EQ. 2214) THEN
         IARFLV = 8
         RETURN
      END IF

c.....Delta++
      IF (IPDG .EQ. 2224) THEN
         IARFLV = 9
         RETURN
      END IF

c.....Lambda
      IF (IPDG .EQ. 3122) THEN
         IARFLV = 14
         RETURN
      END IF

c.....Lambda-bar
      IF (IPDG .EQ. -3122) THEN
         IARFLV = -14
         RETURN
      END IF

c.....Sigma-
      IF (IPDG .EQ. 3112) THEN
         IARFLV = 15
         RETURN
      END IF

c.....Sigma-bar
      IF (IPDG .EQ. -3112) THEN
         IARFLV = -15
         RETURN
      END IF 

c.....Sigma0
      IF (IPDG .EQ. 3212) THEN
         IARFLV = 16
         RETURN
      END IF

c.....Sigma0-bar
      IF (IPDG .EQ. -3212) THEN
         IARFLV = -16
         RETURN
      END IF 

c.....Sigma+
      IF (IPDG .EQ. 3222) THEN
         IARFLV = 17
         RETURN
      END IF

c.....Sigma+ -bar
      IF (IPDG .EQ. -3222) THEN
         IARFLV = -17
         RETURN
      END IF 

c.....K-
      IF (IPDG .EQ. -321) THEN
         IARFLV = 21
         RETURN
      END IF

c.....K+
      IF (IPDG .EQ. 321) THEN
         IARFLV = 23
         RETURN
      END IF

c.....temporary entry for K0
      IF (IPDG .EQ. 311) THEN
         IARFLV = 23
         RETURN
      END IF

c.....temporary entry for K0bar
      IF (IPDG .EQ. -311) THEN
         IARFLV = 21
         RETURN
      END IF

c.....temporary entry for K0S and K0L
      IF (IPDG .EQ. 310 .OR. IPDG .EQ. 130) THEN
         R = RANART(NSEED)
         IF (R .GT. 0.5) THEN
            IARFLV = 23
         ELSE
            IARFLV = 21
         END IF
         RETURN
      END IF

c.....rho-
      IF (IPDG .EQ. -213) THEN
         IARFLV = 25
         RETURN
      END IF

c.....rho0
      IF (IPDG .EQ. 113) THEN
         IARFLV = 26
         RETURN
      END IF

c.....rho+
      IF (IPDG .EQ. 213) THEN
         IARFLV = 27
         RETURN
      END IF

c.....omega
      IF (IPDG .EQ. 223) THEN
         IARFLV = 28
         RETURN
      END IF

c.....phi
      IF (IPDG .EQ. 333) THEN
         IARFLV = 29
         RETURN
      END IF

c.....K*+
      IF (IPDG .EQ. 323) THEN
         IARFLV = 30
         RETURN
      END IF
c.....K*-
      IF (IPDG .EQ. -323) THEN
         IARFLV = -30
         RETURN
      END IF
c.....temporary entry for K*0
      IF (IPDG .EQ. 313) THEN
         IARFLV = 30
         RETURN
      END IF
c.....temporary entry for K*0bar
      IF (IPDG .EQ. -313) THEN
         IARFLV = -30
         RETURN
      END IF

c...... eta-prime
      IF (IPDG .EQ. 331) THEN
         IARFLV = 31
         RETURN
      END IF
 
c...... a1
c     IF (IPDG .EQ. 777) THEN
c        IARFLV = 32
c        RETURN
c     END IF
                                
c... cascade-
      IF (IPDG .EQ. 3312) THEN
         IARFLV = 40
         RETURN
      END IF
 
c... cascade+ (bar)
      IF (IPDG .EQ. -3312) THEN
         IARFLV = -40
         RETURN
      END IF
 
c... cascade0
      IF (IPDG .EQ. 3322) THEN
         IARFLV = 41
         RETURN
      END IF
 
c... cascade0 -bar
      IF (IPDG .EQ. -3322) THEN
         IARFLV = -41
         RETURN
      END IF
 
c... Omega-
      IF (IPDG .EQ. 3334) THEN
         IARFLV = 45
         RETURN
      END IF 

c... Omega+ (bar)
      IF (IPDG .EQ. -3334) THEN
         IARFLV = -45
         RETURN
      END IF

c... Di-Omega
      IF (IPDG .EQ. 6666) THEN
         IARFLV = 44
         RETURN
      END IF
c sp06/05/01 end    

clin-3/2009 keep the same ID numbers in case there are initial deuterons:
      IF (IPDG .EQ. 42 .or. IPDG .EQ. -42) THEN
         IARFLV = IPDG
         RETURN
      END IF

c.....other
      IARFLV = IPDG + 10000

      RETURN
      END

c-----------------------------------------------------------------------

c.....function to convert ART flavor code into PDG flavor code.

      FUNCTION INVFLV(IART)

      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

c.....anti-Delta-
      IF (IART .EQ. -6) THEN
         INVFLV = -1114
         RETURN
      END IF

c.....anti-Delta0
      IF (IART .EQ. -7) THEN
         INVFLV = -2114
         RETURN
      END IF

c.....anti-Delta+
      IF (IART .EQ. -8) THEN
         INVFLV = -2214
         RETURN
      END IF

c.....anti-Delta++
      IF (IART .EQ. -9) THEN
         INVFLV = -2224
         RETURN
      END IF

cbzdbg2/23/99
c.....anti-proton
      IF (IART .EQ. -1) THEN
         INVFLV = -2212
         RETURN
      END IF

c.....anti-neutron
      IF (IART .EQ. -2) THEN
         INVFLV = -2112
         RETURN
      END IF
cbzdbg2/23/99end

c.....eta
      IF (IART .EQ. 0) THEN
         INVFLV = 221
         RETURN
      END IF

c.....proton
      IF (IART .EQ. 1) THEN
         INVFLV = 2212
         RETURN
      END IF

c.....neutron
      IF (IART .EQ. 2) THEN
         INVFLV = 2112
         RETURN
      END IF

c.....pi-
      IF (IART .EQ. 3) THEN
         INVFLV = -211
         RETURN
      END IF

c.....pi0
      IF (IART .EQ. 4) THEN
         INVFLV = 111
         RETURN
      END IF

c.....pi+
      IF (IART .EQ. 5) THEN
         INVFLV = 211
         RETURN
      END IF

c.....Delta-
      IF (IART .EQ. 6) THEN
         INVFLV = 1114
         RETURN
      END IF

c.....Delta0
      IF (IART .EQ. 7) THEN
         INVFLV = 2114
         RETURN
      END IF

c.....Delta+
      IF (IART .EQ. 8) THEN
         INVFLV = 2214
         RETURN
      END IF

c.....Delta++
      IF (IART .EQ. 9) THEN
         INVFLV = 2224
         RETURN
      END IF

cc.....N*(1440), N*(1535) temporary entry
c      IF (IART .GE. 10 .AND. IART .LE.13) THEN
c         INVFLV = 0
c         RETURN
c      END IF

c.....Lambda
      IF (IART .EQ. 14) THEN
         INVFLV = 3122
         RETURN
      END IF
c.....Lambda-bar
      IF (IART .EQ. -14) THEN
         INVFLV = -3122
         RETURN
      END IF 

cbz3/12/99
c.....temporary entry for Sigma's
c      IF (IART .EQ. 15) THEN
c         R = RANART(NSEED)
c         IF (R .GT. 2. / 3.) THEN
c            INVFLV = 3112
c         ELSE IF (R .GT. 1./ 3. .AND. R .LE. 2. / 3.) THEN
c            INVFLV = 3212
c         ELSE
c            INVFLV = 3222
c         END IF
c         RETURN
c      END IF

c.....Sigma-
      IF (IART .EQ. 15) THEN
         INVFLV = 3112
         RETURN
      END IF

c.....Sigma- bar
      IF (IART .EQ. -15) THEN
         INVFLV = -3112
         RETURN
      END IF 

c.....Sigma0
      IF (IART .EQ. 16) THEN
         INVFLV = 3212
         RETURN
      END IF

c.....Sigma0 -bar
      IF (IART .EQ. -16) THEN
         INVFLV = -3212
         RETURN
      END IF

c.....Sigma+
      IF (IART .EQ. 17) THEN
         INVFLV = 3222
         RETURN
      END IF

c.....Sigma+ -bar
      IF (IART .EQ. -17) THEN
         INVFLV = -3222
         RETURN
      END IF 

clin-2/23/03 K0S and K0L are generated at the last timestep:
c.....temporary entry for K- and K0bar
      IF (IART .EQ. 21) THEN
c         R = RANART(NSEED)
c         IF (R .GT. 0.5) THEN
            INVFLV = -321
c         ELSE
c            INVFLV = -311
c            R = RANART(NSEED)
c            IF (R .GT. 0.5) THEN
c               INVFLV = 310
c            ELSE
c               INVFLV = 130
c            END IF
c         END IF
         RETURN
      END IF

c.....temporary entry for K+ and K0
      IF (IART .EQ. 23) THEN
c         R = RANART(NSEED)
c         IF (R .GT. 0.5) THEN
            INVFLV = 321
c         ELSE
c            INVFLV = 311
c            R = RANART(NSEED)
c            IF (R .GT. 0.5) THEN
c               INVFLV = 310
c            ELSE
c               INVFLV = 130
c            END IF
c         END IF
         RETURN
      END IF

c.....K0Long:
      IF (IART .EQ. 22) THEN
         INVFLV = 130
         RETURN
      ENDIF
c.....K0Short:
      IF (IART .EQ. 24) THEN
         INVFLV = 310
         RETURN
      ENDIF

c.....rho-
      IF (IART .EQ. 25) THEN
         INVFLV = -213
         RETURN
      END IF

c.....rho0
      IF (IART .EQ. 26) THEN
         INVFLV = 113
         RETURN
      END IF

c.....rho+
      IF (IART .EQ. 27) THEN
         INVFLV = 213
         RETURN
      END IF

c.....omega
      IF (IART .EQ. 28) THEN
         INVFLV = 223
         RETURN
      END IF

c.....phi
      IF (IART .EQ. 29) THEN
         INVFLV = 333
         RETURN
      END IF

c.....temporary entry for K*+ and K*0
      IF (IART .EQ. 30) THEN
         INVFLV = 323
         IF (RANART(NSEED).GT.0.5) INVFLV = 313
         RETURN
      END IF

c.....temporary entry for K*- and K*0bar
      IF (IART .EQ. -30) THEN
         INVFLV = -323
         IF (RANART(NSEED).GT.0.5) INVFLV = -313
         RETURN
      END IF

c... eta-prime (bar)
      IF (IART .EQ. 31) THEN
         INVFLV = 331
         RETURN
      END IF
 
c... a1
      IF (IART .EQ. 32) THEN
         INVFLV = 777
         RETURN
      END IF
 
c... cascade-
      IF (IART .EQ. 40) THEN
         INVFLV = 3312
         RETURN
      END IF                   

c... cascade+ (bar)
      IF (IART .EQ. -40) THEN
         INVFLV = -3312
         RETURN
      END IF
 
c... cascade0
      IF (IART .EQ. 41) THEN
         INVFLV = 3322
         RETURN
      END IF
 
c... cascade0 -bar
      IF (IART .EQ. -41) THEN
         INVFLV = -3322
         RETURN
      END IF
 
c... Omega-
      IF (IART .EQ. 45) THEN
         INVFLV = 3334
         RETURN
      END IF

c... Omega+ (bar)
      IF (IART .EQ. -45) THEN
         INVFLV = -3334
         RETURN
      END IF

c... Di-Omega
      IF (IART .EQ. 44) THEN
         INVFLV = 6666
         RETURN
      END IF
c sp 12/19/00 end           

clin-5/2008 deuteron ID numbers in ART and ampt.dat:
      IF (IART .EQ. 42) THEN
         INVFLV = 42
         RETURN
      ELSEIF (IART .EQ. -42) THEN         
         INVFLV = -42
         RETURN
      END IF
c
c.....other
      INVFLV = IART - 10000

      RETURN
      END

c=======================================================================

      BLOCK DATA ARDATA

      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/
      SAVE   
      DATA ARPAR1/1.19, 99 * 0.0/
      DATA IAPAR2/3, 49 * 0/
      DATA ARINT1/100 * 0.0/
      DATA IAINT2/50 * 0/

      END

c=======================================================================

c.....Routine borrowed from ZPC.
c.....double precision  is modified to real*4.

cbz1/29/99
c      subroutine index1(n, m, arrin, indx)
      subroutine arindx(n, m, arrin, indx)
cbz1/29/99end
c     indexes the first m elements of ARRIN of length n, i.e., outputs INDX
c     such that ARRIN(INDEX(J)) is in ascending order for J=1,...,m

c      implicit real*4 (a-h, o-z)

      dimension arrin(n), indx(n)
      SAVE   
      do 1001 j = 1, m
         indx(j) = j
 1001 continue
      l = m / 2 + 1
      ir = m
 10   continue
      if (l .gt. 1) then
         l = l - 1
         indxt = indx(l)
         q = arrin(indxt)
      else
         indxt = indx(ir)
         q = arrin(indxt)
         indx(ir) = indx(1)
         ir = ir - 1
         if (ir .eq. 1) then
            indx(1) = indxt
            return
         end if
      end if
      i = l
      j = l + l
 20   if (j .le. ir) then
         if (j .lt. ir) then
            if (arrin(indx(j)) .lt. arrin(indx(j + 1))) j = j + 1
         end if
         if (q .lt. arrin(indx(j))) then
            indx(i) = indx(j)
            i = j
            j = j + j
         else
            j = ir + 1
         end if
      goto 20
      end if
      indx(i) = indxt
      goto 10

      end

c-----------------------------------------------------------------------

c.....extracted from G. Song's ART expasion including K- interactions
c.....file `NEWKAON.FOR'

c     5/01/03 send iblock value into art1f.f, necessary for resonance studies:
c        subroutine newka(icase,irun,iseed,dt,nt,ictrl,i1,i2,
c     &                                   srt,pcx,pcy,pcz)
        subroutine newka(icase,irun,iseed,dt,nt,ictrl,i1,i2,
     &                                   srt,pcx,pcy,pcz,iblock)
      PARAMETER      (MAXSTR=150001,MAXR=1)
      PARAMETER      (AKA=0.498)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /RUN/NUM
cc      SAVE /RUN/
      COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
      COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
      COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
      COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
c
        logical lb1bn, lb2bn,lb1mn,lb2mn
cbz3/7/99 neutralk
c        logical lb1bn1, lb2bayon1, lb1bn0, lb2bn0
        logical lb1bn1, lb2bn1, lb1bn0, lb2bn0
cbz3/7/99 neutralk end
        logical lb1mn0, lb2mn0, lb1mn1, lb2mn1
        logical lb1mn2, lb2mn2
        icase=-1
c        icase: flag for the type of reaction that is going to happen.
c        icase=-1,  no desired reaction, return to main program.
c              1,  NN,ND,DD
c              2,  PI+N, PI+D
c              3,  K(-) absorption.
        nchrg=-100
c        nchrg: Net charges of the two incoming particles.
        ictrl = 1
        lb1=lb(i1)
        lb2=lb(i2)
        em1=e(i1)
        em2=e(i2)
        lb1bn=lb1.eq.1.or.lb1.eq.2.or.(lb1.gt.5.and.lb1.le.13)
        lb2bn=lb2.eq.1.or.lb2.eq.2.or.(lb2.gt.5.and.lb2.le.13)
        lb1bn0=lb1.eq.2.or.lb1.eq.7.or.lb1.eq.10.or.lb1.eq.12
        lb2bn0=lb2.eq.2.or.lb2.eq.7.or.lb2.eq.10.or.lb2.eq.12
        lb1bn1=lb1.eq.1.or.lb1.eq.8.or.lb1.eq.11.or.lb1.eq.13
        lb2bn1=lb2.eq.1.or.lb2.eq.8.or.lb2.eq.11.or.lb2.eq.13
        lb1mn=em1.lt.0.2.or.lb1.eq.0.or.(lb1.ge.25.and.lb1.le.29)
        lb2mn=em2.lt.0.2.or.lb2.eq.0.or.(lb2.ge.25.and.lb2.le.29)
        lb1mn0=lb1.eq.0.or.lb1.eq.4.or.lb1.eq.26.or.
     &                        lb1.eq.28.or.lb1.eq.29
        lb2mn0=lb2.eq.0.or.lb2.eq.4.or.lb2.eq.26.or.
     &                        lb2.eq.28.or.lb2.eq.29
        lb1mn1= lb1.eq.5.or.lb1.eq.27
        lb2mn1= lb2.eq.5.or.lb2.eq.27
        lb1mn2=lb1.eq.3.or.lb1.eq.25
        lb2mn2=lb2.eq.3.or.lb2.eq.25

c        1. consider N+N, N+Resonance, R + R reactions
        if(lb1bn.and.lb2bn) then
c     NN,ND,DD:
           icase=1
c     total cross section
           sig=40.
           if(lb1.eq.9.and.lb2.eq.9) then
                nchrg=4
           endif   
           if((lb1bn1.and.lb2.eq.9)
     &        .or.(lb2bn1.and.lb1.eq.9))then
                nchrg=3
           endif
           if((lb1bn0.and.lb2.eq.9)
     &        .or.(lb2bn0.and.lb1.eq.9)
     &        .or.(lb1bn1.and.lb2bn1)) then
                   nchrg=2
           endif
           if((lb1bn1.and.lb2bn0).or.(lb1.eq.6.and.lb2.eq.9)
     &        .or.(lb2bn1.and.lb1bn0)
     &        .or.(lb2.eq.6.and.lb1.eq.9))then
                   nchrg=1
           endif
           if((lb1bn0.and.lb2bn0).or.(lb1bn1.and.lb2.eq.6)
     &              .or.(lb2bn1.and.lb1.eq.6)) then
                   nchrg=0
           endif
           if((lb1bn0.and.lb2.eq.6)
     &        .or.(lb2bn0.and.lb1.eq.6))then
                nchrg=-1
           endif
           if(lb1.eq.6.and.lb2.eq.6) then
                nchrg=-2
           endif
c     brsig = x2kaon_no_isospin(srt)
           if(nchrg.ge.-1.and.nchrg.le.2) then
c     K,Kbar prduction x sect.
                   brsig = x2kaon(srt)
           else
                   brsig=0.0
c                if(nchrg.eq.-2.or.nchrg.eq.3) then
c                   brsig = x2kaon(srt+0.938-1.232)
c                else
c     nchrg=4
c                   brsig = x2kaon(srt+2.*(0.938-1.232))
c                endif
           endif

cbz3/7/99 neutralk
           BRSIG = 2.0 * BRSIG
cbz3/7/99 neutralk end

        endif

c        2. consider PI(meson:eta,omega,rho,phi) + N(N*,D)
        if((lb1bn.and.lb2mn).OR.(lb2bn.and.lb1mn)) then
c     PN,PD
          icase=2
          sig=20.
          sigma0 = piNsg0(srt)
          brsig=0.0
          if((lb1bn1.and.lb2mn0)
     &       .or.(lb2bn1.and.lb1mn0).
     & or.(lb1bn0.and.lb2mn1).or.(lb2bn0.and.lb1mn1).
     & or.(lb1.eq.9.and.lb2mn2).or.(lb2.eq.9.and.lb1mn2))then
                nchrg=1
cbz3/2/99/song
c                if(lb1bn1.or.lb2bn1) brsig=2.0*sigma0
c                if(lb1bn0.or.lb2bn0) brsig=0.5*sigma0
                if(lb1bn1.or.lb2bn1) brsig=0.5*sigma0
                if(lb1bn0.or.lb2bn0) brsig=2.0*sigma0
cbz3/2/99/song end
c                if(lb1.eq.9.or.lb2.eq.9) brsig=1.5*sigma0
          endif
          if( (lb1bn0.and.lb2mn0 )
     &       .or.(lb2bn0.and.lb1mn0)
     &  .or.(lb1bn1.and.lb2mn2).or.(lb2bn1.and.lb1mn2)
     &  .or.(lb1.eq.6.and.lb2mn1).or.(lb2.eq.6.and.lb1mn1)) then
                nchrg=0
                if(lb1bn1.or.lb2bn1) then
cbz3/2/99/song
c                  brsig=1.5*sigma0
                  brsig=3.0*sigma0
cbz3/2/99/song end
cbz3/11/99/song
c                  ratiok = 1./3.
                  ratiok = 2./3.
cbz3/11/99/song end

c                  ratiok: the ratio of channels: ->nK+k- vs. -> pK0K-
                endif
                if(lb1bn0.or.lb2bn0) then
                  brsig=2.5*sigma0
cbz3/2/99/song
c                  ratiok = 0.8
                  ratiok = 0.2
cbz3/2/99/song end
                endif
c                if(lb1.eq.6.or.lb2.eq.6) then
c     lb=6 : D-
c                  brsig=1.5*sigma0
c                  ratiok = 0.5
c                endif
          endif
          if( (lb1bn0.and.lb2mn2)
     &       .or.(lb2bn0.and.lb1mn2)
     & .or.(lb1.eq.6.and.lb2mn0).or.(lb2.eq.6.and.lb1mn0)) then
                nchrg=-1
                if(lb1bn0.or.lb2bn0) brsig=sigma0
c                if(lb1.eq.6.or.lb2.eq.6) brsig=sigma0
          endif
c          if((lb1.eq.6.and.lb2mn2).or.(lb2.eq.6.and.lb1mn2))then
c                nchrg=-2
c          endif
c          if((lb1bn1.and.lb2mn1).or.(lb2bn1.and.lb1mn1)
c    &           .or.(lb1.eq.9.and.lb2mn0).or.(lb2.eq.9.and.lb1mn0)) then
c                nchrg=2
c          endif

cbz3/11/99 neutralk
          if((lb1.eq.6.and.lb2mn2)
     &       .or.(lb2.eq.6.and.lb1mn2))then
                nchrg=-2
          endif
cbz3/11/99 neutralk
cbz3/8/99 neutralk
          if((lb1bn1.and.lb2mn1)
     &       .or.(lb2bn1.and.lb1mn1)
     & .or.(lb1.eq.9.and.lb2mn0).or.(lb2.eq.9.and.lb1mn0)) then
                nchrg=2
          endif
cbz3/8/99 neutralk end

cbz3/7/99 neutralk
          IF (NCHRG .GE. -2 .AND. NCHRG .LE. 2) THEN
             BRSIG = 3.0 * SIGMA0
          END IF
cbz3/7/99 neutralk end

        endif

c        3. consider K- + N(N*,D) absorption.
c        if((lb1bn.and.lb2.eq.21).OR.(lb2bn.and.lb1.eq.21)) then
        if( (lb1bn.and.(lb2.eq.21.or.lb2.eq.-30)).OR.
     &     (lb2bn.and.(lb1.eq.21.or.lb1.eq.-30)) )then 
c          bmass=em1+em2-aka
          bmass=0.938
          if(srt.le.(bmass+aka)) then
cbz3/2/99
c                write(100,*)'--lb1,lb2,em1,em2,srt',lb1,lb2,em1,em2,srt
cbz3/2/99end
                pkaon=0.
          else
            pkaon=sqrt(((srt**2-(aka**2+bmass**2))/2./bmass)**2-aka**2)
          endif
          sig=0.
          if(lb1.eq.1.or.lb2.eq.1.or.lb1.eq.8.or.lb2.eq.8.or.
     &    lb1.eq.11.or.lb2.eq.11.or.lb1.eq.13.or.lb2.eq.13) then
c          K- + (D+,N*+)p ->
              nchrg=0
              sigela=akPel(pkaon)
              sigsgm=3.*akPsgm(pkaon)
              sig=sigela+sigsgm+akPlam(pkaon)
          endif
          if(lb1.eq.2.or.lb2.eq.2.or.lb1.eq.7.or.lb2.eq.7.or.
     &    lb1.eq.10.or.lb2.eq.10.or.lb1.eq.12.or.lb2.eq.12) then
c          K- + (D0, N*0)n ->
              nchrg=-1
              sigela=akNel(pkaon)
              sigsgm=2.*akNsgm(pkaon)
              sig=sigela+sigsgm+akNlam(pkaon)
          endif
          if(lb1.eq.6.or.lb2.eq.6) then
c     K- + D-
              nchrg=-2
              sigela=akNel(pkaon)
              sigsgm=akNsgm(pkaon)
              sig=sigela+sigsgm
          endif
          if(lb1.eq.9.or.lb2.eq.9) then
c     K- + D++
              nchrg=1
              sigela=akPel(pkaon)
              sigsgm=2.*akPsgm(pkaon)
              sig=sigela+sigsgm+akPlam(pkaon)
          endif

cbz3/8/99 neutralk
          sigela = 0.5 * (AKPEL(PKAON) + AKNEL(PKAON))
          SIGSGM = 1.5 * AKPSGM(PKAON) + AKNSGM(PKAON)
          SIG = sigela + SIGSGM + AKPLAM(PKAON)
cbz3/8/99 neutralk end

          if(sig.gt.1.e-7) then
c     K(-) + N reactions
              icase=3
              brel=sigela/sig
              brsgm=sigsgm/sig
c              branch_lambda=akNlam(pkaon)/sig
              brsig = sig
          endif
        endif

c        4. meson + hyperon -> K- + N
c        if(((lb1.ge.14.and.lb1.le.17).and.lb2mn).OR.
c     &     ((lb2.ge.14.and.lb2.le.17).and.lb1mn)) then
        if(((lb1.ge.14.and.lb1.le.17).and.(lb2.ge.3.and.lb2.le.5)).OR.
     &     ((lb2.ge.14.and.lb2.le.17).and.(lb1.ge.3.and.lb1.le.5)))then
c        first classify the reactions due to total charge.
           nchrg=-100
           if((lb1.eq.15.and.(lb2.eq.3.or.lb2.eq.25)).OR.
     &              (lb2.eq.15.and.(lb1.eq.3.or.lb1.eq.25))) then
                nchrg=-2
c     D-
                  bmass=1.232
           endif
           if((lb1.eq.15.and.lb2mn0).or.(lb2.eq.15.and.lb1mn0).OR.
     &       ((lb1.eq.14.or.lb1.eq.16).and.(lb2.eq.3.or.lb2.eq.25)).OR.
     &       ((lb2.eq.14.or.lb2.eq.16).and.(lb1.eq.3.or.lb1.eq.25)))then
                nchrg=-1
c     n
                 bmass=0.938
           endif
           if((lb1.eq.15.and.(lb2.eq.5.or.lb2.eq.27)).OR.
     &              (lb2.eq.15.and.(lb1.eq.5.or.lb1.eq.27)).or.
     &        (lb1.eq.17.and.(lb2.eq.3.or.lb2.eq.25)).OR.
     &              (lb2.eq.17.and.(lb1.eq.3.or.lb1.eq.25)).or.
     &       ((lb1.eq.14.or.lb1.eq.16).and.lb2mn0).OR.
     &       ((lb2.eq.14.or.lb2.eq.16).and.lb1mn0)) then
                nchrg=0
c     p
                 bmass=0.938
           endif
           if((lb1.eq.17.and.lb2mn0).or.(lb2.eq.17.and.lb1mn0).OR.
     &       ((lb1.eq.14.or.lb1.eq.16).and.(lb2.eq.5.or.lb2.eq.27)).OR.
     &       ((lb2.eq.14.or.lb2.eq.16).and.(lb1.eq.5.or.lb1.eq.27)))then
                nchrg=1
c     D++
                 bmass=1.232
           endif
           sig = 0.
           if(nchrg.ne.-100.and.srt.gt.(aka+bmass)) then
c     PI+sigma or PI + Lambda => Kbar + N reactions
             icase=4
c             pkaon=sqrt(((srt**2-(aka**2+bmass**2))/2./bmass)**2-aka**2)
             pkaon=sqrt(((srt**2-(aka**2+0.938**2))/2./0.938)**2-aka**2)
c     lambda + Pi
             if(lb1.eq.14.or.lb2.eq.14) then
                if(nchrg.ge.0) sigma0=akPlam(pkaon)
                if(nchrg.lt.0) sigma0=akNlam(pkaon)
c     sigma + pi
             else
c     K-p or K-D++
                if(nchrg.ge.0) sigma0=akPsgm(pkaon)
c     K-n or K-D-
                if(nchrg.lt.0) sigma0=akNsgm(pkaon)

cbz3/8/99 neutralk
                SIGMA0 = 1.5 * AKPSGM(PKAON) + AKNSGM(PKAON)
cbz3/8/99 neutralk end

             endif
             sig=(srt**2-(aka+bmass)**2)*(srt**2-(aka-bmass)**2)/
     &         (srt**2-(em1+em2)**2)/(srt**2-(em1-em2)**2)*sigma0
cbz3/8/99 neutralk
c     if(nchrg.eq.-2.or.nchrg.eq.1) sig=2.*sig K-D++, K-D-
c     K0barD++, K-D-
             if(nchrg.eq.-2.or.nchrg.eq.2) sig=2.*sig

cbz3/8/99 neutralk end

c             the factor 2 comes from spin of delta, which is 3/2
c             detailed balance. copy from Page 423 of N.P. A614 1997

cbz3/8/99 neutralk
             IF (LB1 .EQ. 14 .OR. LB2 .EQ. 14) THEN
                SIG = 4.0 / 3.0 * SIG
             ELSE IF (NCHRG .EQ. -2 .OR. NCHRG .EQ. 2) THEN
                SIG = 8.0 / 9.0 * SIG
             ELSE
                SIG = 4.0 / 9.0 * SIG
             END IF
cbz3/8/99 neutralk end
             brsig = sig
             if(sig.lt.1.e-7) sig = 1.e-7
           endif
csp05/07/01
* comment icase=4 statement below if only inelastic
c     PI+L/Si => Kbar + N  OR ELASTIC SCATTERING
           icase=4
           brsig = sig
c     elastic xsecn of 10mb
           sigela = 10.
           sig = sig + sigela
           brel = sigela/sig
cc          brsig = sig
csp05/07/01 end   
        endif
c
c        if(em2.lt.0.2.and.em1.lt.0.2) then
c     PI + PI 
c             icase=5
c     assumed PI PI total x section.
c              sig=50.
c     Mk + Mkbar
c              s0=aka+aka
c              brsig = 0.
c              if(srt.gt.s0) brsig = 2.7*(1.-s0**2/srt**2)**0.76
c              x section for PIPI->KKbar   PRC43 (1991) 1881
c        endif
        if(icase.eq.-1) then
           ictrl = -1
           return
        endif
        px1cm=pcx
        py1cm=pcy
        pz1cm=pcz
        ds=sqrt(sig/31.4)
        dsr=ds+0.1
        ec=(em1+em2+0.02)**2
c        ec=3.59709
c        if((e(i1).ge.1.).and.(e(i2).ge.1.)) ec = 4.75

        call distce(i1,i2,dsr,ds,dt,ec,srt,ic,px1cm,py1cm,pz1cm)
        if(ic.eq.-1) then
c     no anti-kaon production
           ictrl = -1
c           in=in+1
c           write(60,*)'--------------distance-----',in
           return
        endif

clin-10/24/02 set to 0: ik,ik0-3,il,im,im3-4,in,inpion,ipipi, 
c     sgsum,sgsum1,sgsum3:
        ik=0
        ik0=0
        ik1=0
        ik2=0
        ik3=0
        il=0
        im=0
        im3=0
        im4=0
        in=0
        inpion=0
        ipipi=0
        sgsum=0.
        sgsum1=0.
        sgsum3=0.
        if(icase.eq.1) then
           ik=ik+1
           if(srt.gt.2.8639) then
                ik0=ik0+1
                if(em1.lt.1.0.and.em2.lt.1.0) then
                        ik1=ik1+1
                        sgsum1=sgsum1+brsig
c                        ratio_1=sgsum1/ik1/40.
                endif
                if(em1.gt.1.0.and.em2.gt.1.0) then
                        ik3=ik3+1
                        sgsum3=sgsum3+brsig
c                        ratio_3=sgsum3/ik3/40.
                endif
                if(em1.gt.1.0.and.em2.lt.1.0) ik2=ik2+1
                if(em1.lt.1.0.and.em2.gt.1.0) ik2=ik2+1
                sgsum=sgsum+brsig
c                ratio=sgsum/ik0/40.
           endif
        endif
        if(icase.eq.2) inpion=inpion+1
        if(icase.eq.5) ipipi=ipipi+1
c        write(62,*)'ik1,ik2,ik3',ik1,ik2,ik3,ratio_1,ratio_3,ratio
c        write(62,*)'inpion,ipipi',inpion,ipipi
        if(RANART(NSEED).gt.(brsig/sig)) then
c     no anti-kaon production
           ictrl = -1
           return
        endif
        il=il+1
c        kaons could be created now.
        if(icase.eq.1) then
          in=in+1
c          write(60,*)'------in,s2kaon,sig=',in,brsig,sig,lb1,lb2
          call nnkaon(irun,iseed,
     &          ictrl,i1,i2,iblock,srt,pcx,pcy,pcz,nchrg)
        endif
        if(icase.eq.2) then
          im=im+1
c          call npik(irun,iseed,dt,nt,ictrl,i1,i2,srt,
c     &              pcx,pcy,pcz,nchrg,ratiok)
          call npik(irun,iseed,dt,nt,ictrl,i1,i2,srt,
     &              pcx,pcy,pcz,nchrg,ratiok,iblock)
        endif
c
        if(icase.eq.3) then
          im3=im3+1
c          write(63,*)'im3,lb1,lb2,pkaon',im3,lb1,lb2,pkaon
c          write(63,*)'sig,el,sigma',sig,brel,brsgm
c          write(63,*)'srt,pcx,pcy,pcz,em1,em2',srt,pcx,pcy,pcz,em1,em2
          call kaonN(brel,brsgm,irun,iseed,dt,nt,ictrl,
     &                i1,i2,iblock,srt,pcx,pcy,pcz,nchrg)
c         this subroutine format is diff. since three final states are possible
        endif
c

        if(icase.eq.4) then
          im4=im4+1
c          write(64,*)'im4,sigma0,branch,sig=',im4,sigma0,brsig,sig
c          write(64,*)'lb1,lb2,em1,em2,pkaon=',lb1,lb2,em1,em2,pkaon

csp06/07/01
      if(RANART(NSEED).lt.brel) then
         ielstc = 1
      else
         ielstc = 0
      endif                  
c          call Pihypn(ielstc,irun,iseed,dt,nt,ictrl,i1,i2,srt,
c     &                   pcx,pcy,pcz,nchrg)
          call Pihypn(ielstc,irun,iseed,dt,nt,ictrl,i1,i2,srt,
     &                   pcx,pcy,pcz,nchrg,iblock)

csp06/07/01 end
        endif
c        if(icase.eq.5) then
c          im5=im5+1
c          write(65,*)'---im5,s2kaon,sig=',im5,brsig,sig
c          call pipikaon(irun,iseed,dt,nt,ictrl,i1,i2,srt,pcx,pcy,pcz)
c        endif
cbz3/2/99
c        write(101,*)lb1,lb2,lb(i1),lb(i2)
c        write(101,*)em1,em2,e(i1),e(i2),srt
cbz3/2/99end

        return
        end

******************************************
* for pp-->pp + kaon + anti-kaon
c      real*4 function X2kaon(srt)
      real function X2kaon(srt)
      SAVE   
*  This function contains the experimental total pp->pp+K(+)K(-) Xsections    *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*                                                                             *
******************************************
c     minimum c.m.s. energy to create 2 kaon. = 2*(mp+mk)        
        smin = 2.8639
        x2kaon=0.0000001
        if(srt.lt.smin)return
        sigma1 = 2.8
        sigma2 = 7.7
        sigma3 = 3.9
        x = srt**2/smin**2 + 0.0000001
        f1 = (1.+1./sqrt(x))*alog(x) - 4.*(1.-1./sqrt(x))
        f2 = 1. - (1./sqrt(x))*(1.+alog(sqrt(x)))
        f3 = ((x-1.)/x**2)**3.5
        x2kaon = (1.-1./x)**3*(sigma1*f1 + sigma2*f2) + sigma3*f3
        return
        END

        real function piNsg0(srt)
      SAVE   
* cross section in mb for PI- + P -> P + K0 + K-
c     Mn + 2* Mk
        srt0 = 0.938 + 2.*0.498
        if(srt.lt.srt0) then
           piNsg0 = 0.0
           return
        endif
        ratio = srt0**2/srt**2
        piNsg0=1.121*(1.-ratio)**1.86*ratio**2
        return
        end

        real function akNel(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
c        the following data come from PRC 41 (1701)
c        sigma1: K(-) + neutron elastic
        if(pkaon.lt.0.5.or. pkaon.ge.4.0) sigma1=0.
        if(pkaon.ge.0.5.and.pkaon.lt.1.0) sigma1=20.*pkaon**2.74
        if(pkaon.ge.1.0.and.pkaon.lt.4.0) sigma1=20.*pkaon**(-1.8)
        akNel=sigma1
        return
        end

        real function akPel(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
c        the following data come from PRC 41 (1701)
c        sigma2: K(-) + proton elastic
        if(pkaon.lt.0.25.or. pkaon.ge.4.0) sigma2=0.
        if(pkaon.ge.0.25.and.pkaon.lt.4.0) sigma2=13.*pkaon**(-0.9)
        akPel=sigma2
        return
        end

        real function akNsgm(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
c        sigma2: x section for K- + n -> sigma0 + PI-
        if(pkaon.lt.0.5.or. pkaon.ge.6.0) sigma2=0.
        if(pkaon.ge.0.5.and.pkaon.lt.1.0) sigma2=1.2*pkaon**(-1.3)
        if(pkaon.ge.1.0.and.pkaon.lt.6.0) sigma2=1.2*pkaon**(-2.3)
        akNsgm=sigma2
        return
        end

        real function akPsgm(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
c        sigma1: x section for K- + p -> sigma0 + PI0
        if(pkaon.lt.0.2.or. pkaon.ge.1.5) sigma1=0.
        if(pkaon.ge.0.2.and.pkaon.lt.1.5) sigma1=0.6*pkaon**(-1.8)
        akPsgm=sigma1
        return
        end

        real function akPlam(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
c        sigma: x section for K- + p -> lambda + PI0
        p=pkaon
        if(pkaon.lt.0.2.or. pkaon.ge.10.0) sigma=0.
        if(pkaon.ge.0.2.and.pkaon.lt.0.9) sigma=50.*p**2-67.*p+24.
        if(pkaon.ge.0.9.and.pkaon.lt.10.0) sigma=3.0*pkaon**(-2.6)
        akPlam=sigma
        return
        end

        real function akNlam(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
        akNlam=akPlam(pkaon)
        return
        end

* GQ Li parametrization (without resonance)
        real function akNPsg(pkaon)
      SAVE   
*cross section in mb for K- + N reactions.
c       sigma1: x section for K- + p/n -> sigma0 + PI0
         if(pkaon.le.0.345)then
           sigma1=0.624*pkaon**(-1.83)
         else
           sigma1=0.7*pkaon**(-2.09)
         endif
        akNPsg=sigma1
        return
        end   

c-----------------------------------------------------------------------

c.....extracted from G. Song's ART expasion including K- interactions
c.....file `NEWNNK.FOR'

        subroutine nnkaon(irun,iseed,ictrl,i1,i2,iblock,
     &                                   srt,pcx,pcy,pcz,nchrg)
c        <pt>=0.27+0.037*log(srt) was changed to 0.632 + ... on Aug. 14, 1997
c     CANCELED also alpha=1 changed to alpha=3 to decrease the leadng effect.
      PARAMETER      (MAXSTR=150001,MAXR=1)
      PARAMETER      (AKA=0.498)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /RUN/NUM
cc      SAVE /RUN/
      COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
      COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
      COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
      COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
      dimension px(4),py(4),pz(4)
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      SAVE   
c      dm1=e(i1)
c      dm2=e(i2)
      dm3=0.938
      dm4=0.938
c     10/24/02 initialize n to 0:
      n=0

cbz3/11/99 neutralk
c        if(nchrg.eq.-2.or.nchrg.ge.3) dm3=1.232
c        if(nchrg.eq.4) dm4=1.232
        if(nchrg.le.-1.or.nchrg.ge.3) dm3=1.232
        if(nchrg.eq.-2.or.nchrg.eq.4) dm4=1.232
cbz3/11/99 neutralk end
          iblock = 0 
        call fstate(iseed,srt,dm3,dm4,px,py,pz,iflag)
        if(iflag.lt.0) then
c           write(60,*)'------------final state fail-------',n
c     no anti-kaon production
           ictrl = -1
           n=n+1
           return
        endif
        iblock = 12
* Rotate the momenta of particles in the cms of I1 & I2
* px(1), py(1), pz(1): momentum of I1
* px(2), py(2), pz(2): momentum of I2
* px(3), py(3), pz(3): momentum of anti-kaon
* px(4), py(4), pz(4): momentum of kaon


c     10/28/02 get rid of argument usage mismatch in rotate():
        pxrota=px(1)
        pyrota=py(1)
        pzrota=pz(1)
c        call rotate(pcx,pcy,pcz,px(1),py(1),pz(1))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        px(1)=pxrota
        py(1)=pyrota
        pz(1)=pzrota
c
        pxrota=px(2)
        pyrota=py(2)
        pzrota=pz(2)
c        call rotate(pcx,pcy,pcz,px(2),py(2),pz(2))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        px(2)=pxrota
        py(2)=pyrota
        pz(2)=pzrota
c
        pxrota=px(3)
        pyrota=py(3)
        pzrota=pz(3)
c        call rotate(pcx,pcy,pcz,px(3),py(3),pz(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        px(3)=pxrota
        py(3)=pyrota
        pz(3)=pzrota
c
        pxrota=px(4)
        pyrota=py(4)
        pzrota=pz(4)
c        call rotate(pcx,pcy,pcz,px(4),py(4),pz(4))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        px(4)=pxrota
        py(4)=pyrota
        pz(4)=pzrota

        nnn=nnn+2
c     K+
        lpion(nnn,irun)=23
        if(nchrg.eq.-1.or.nchrg.eq.-2) then
c        To keep charge conservation. D-n->nnK0K-, D-D- -> nD-K0K-

cbz3/7/99 neutralk
c           lpion(nnn,irun)=24 ! K0
cbz3/7/99 neutralk end

        endif
c     aka: rest mass of K
        epion(nnn,irun)=aka
c     K-
        lpion(nnn-1,irun)=21
c     aka: rest mass of K
        epion(nnn-1,irun)=aka
* Find the momenta of particles in the final state in the nucleus_nucleus
* cms frame.   Lorentz transformation into lab frame.
        e1cm   = sqrt(dm3**2 + px(1)**2 + py(1)**2 + pz(1)**2)
        p1beta = px(1)*betax + py(1)*betay + pz(1)*betaz
        transf = gamma * ( gamma*p1beta / (gamma+1) + e1cm)
        pt1i1 = betax*transf + px(1)
        pt2i1 = betay*transf + py(1)
        pt3i1 = betaz*transf + pz(1)
        eti1  = dm3
c        lb1   = lb(i1)
        lb1   = 2
        if(nchrg.ge.-2.and.nchrg.le.1) lb1=2

cbz3/7/99 neutralk
        if (nchrg .eq. -2 .or. nchrg .eq. -1) then
           lb1 = 6
        end if
cbz3/7/99 neutralk end

cbz3/11/99 neutralk
c        if(nchrg.eq.2.or.nchrg.eq.3) lb1=1
c        if(nchrg.eq.4) lb1=9
        if(nchrg.eq.1.or.nchrg.eq.2) lb1=1
        if(nchrg.eq.3.or.nchrg.eq.4) lb1=9
cbz3/11/99 neutralk end

* For second nulceon, same
        e2cm   = sqrt(dm4**2 + px(2)**2 + py(2)**2 + pz(2)**2)
        p2beta = px(2)*betax + py(2)*betay + pz(2)*betaz
        transf = gamma * ( gamma*p2beta / (gamma+1) + e2cm)
        pt1i2 = betax*transf + px(2)
        pt2i2 = betay*transf + py(2)
        pt3i2 = betaz*transf + pz(2)
        eti2  = dm4
c        lb2   = lb(i2)
        lb2   = 2

cbz3/11/99 neutralk
c        if(nchrg.eq.-1.or.nchrg.eq.0) lb2=2
c        if(nchrg.eq. 2.or.nchrg.eq.1) lb2=1
c        if(nchrg.eq. 4.or.nchrg.eq.3) lb2=9
c        if(nchrg.eq.-2) lb2=6
        if(nchrg.ge.-1.or.nchrg.le.1) lb2=2
        if(nchrg.eq. 2.or.nchrg.eq.3) lb2=1
        if(nchrg.eq. 4) lb2=9
        if(nchrg.eq.-2) lb2=6
cbz3/11/99 neutralk end

c        if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0.)then
                p(1,i1)=pt1i1
                p(2,i1)=pt2i1
                p(3,i1)=pt3i1
                e(i1)=eti1
                lb(i1)=lb1
                p(1,i2)=pt1i2
                p(2,i2)=pt2i2
                p(3,i2)=pt3i2
                e(i2)=eti2
                lb(i2)=lb2

c                px1 = p(1,i1)
c                py1 = p(2,i1)
c                pz1 = p(3,i1)
c                em1 = e(i1)
c                id(i1) = 2
c                id(i2) = 2
c                id1 = id(i1)
c                iblock = 101  ! K(+)K(-) production
* Get anti-kaons' momenta and coordinates in nucleus-nucleus cms. frame.
        epcmk = sqrt(epion(nnn-1,irun)**2 + px(3)**2+py(3)**2+pz(3)**2)
        betak = px(3)*betax + py(3)*betay + pz(3)*betaz
        transf= gamma*(gamma*betak/(gamma+1.) + epcmk)
        ppion(1,nnn-1,irun)=betax*transf + px(3)
        ppion(2,nnn-1,irun)=betay*transf + py(3)
        ppion(3,nnn-1,irun)=betaz*transf + pz(3)
        rpion(1,nnn-1,irun)=r(1,i1)
        rpion(2,nnn-1,irun)=r(2,i1)
        rpion(3,nnn-1,irun)=r(3,i1)
clin-5/2008:
        dppion(nnn-1,irun)=dpertp(i1)*dpertp(i2)
* Same thing for kaon **************************************
        epcmak = sqrt(epion(nnn,irun)**2 + px(4)**2 +py(4)**2+pz(4)**2)
        betaak = px(4)*betax + py(4)*betay + pz(4)*betaz
        transf= gamma*(gamma*betaak/(gamma+1.) + epcmak)
        ppion(1,nnn,irun)=betax*transf + px(4)
        ppion(2,nnn,irun)=betay*transf + py(4)
        ppion(3,nnn,irun)=betaz*transf + pz(4)
        rpion(1,nnn,irun)=r(1,i2)
        rpion(2,nnn,irun)=r(2,i2)
        rpion(3,nnn,irun)=r(3,i2)
clin-5/2008:
        dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
        return
        end

        subroutine lorntz(ilo,b,pi,pj)
c       It uses to perform Lorentz (or inverse Lorentz) transformation
        dimension pi(4),pj(4),b(3)
      SAVE   
c       dimension db(3)
        bb=b(1)*b(1)+b(2)*b(2)+b(3)*b(3)
        deno3=sqrt(1.-bb)
        if(deno3.eq.0.)deno3=1.e-10
        gam=1./deno3
        ga=gam*gam/(gam+1.)
        if(ilo.eq.1) goto 100
c       Lorentz transformation
        pib=pi(1)*b(1)+pi(2)*b(2)+pi(3)*b(3)
        pjb=pj(1)*b(1)+pj(2)*b(2)+pj(3)*b(3)
c       drb=drd(1)*b(1)+drd(2)*b(2)+drd(3)*b(3)
c       drdb=db(1)*b(1)+db(2)*b(2)+db(3)*b(3)
        do 1001 i=1,3
           pi(i)=pi(i)+b(i)*(ga*pib-gam*pi(4))
           pj(i)=pj(i)+b(i)*(ga*pjb-gam*pj(4))
c       drd(i)=drd(i)+b(i)*ga*drb
c       db(i)=db(i)+b(i)*ga*drdb
 1001   continue
        pi(4)=gam*(pi(4)-pib)
        pj(4)=gam*(pj(4)-pjb)
        return
100     continue
c       inverse Lorentz transformation
        pib=pi(1)*b(1)+pi(2)*b(2)+pi(3)*b(3)
        pjb=pj(1)*b(1)+pj(2)*b(2)+pj(3)*b(3)
        do 1002 i=1,3
           pi(i)=pi(i)+b(i)*(ga*pib+gam*pi(4))
           pj(i)=pj(i)+b(i)*(ga*pjb+gam*pj(4))
 1002   continue
        pi(4)=gam*(pi(4)+pib)
        pj(4)=gam*(pj(4)+pjb)
        return
        end
        
        subroutine fstate(iseed,srt,dm3,dm4,px,py,pz,iflag)
*        function: decide final momentum for N,N,K(+),and K(-)        
        dimension px(4), py(4), pz(4), pe(4)
        COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
        SAVE   

        iseed=iseed
        iflag=-1
c        iflag=-1: fail to find momenta
c             = 1: success
        pio=3.1415926
        aka=0.498
c        v=0.43
c        w=-0.84
c        b=3.78
c        c=0.47
c        d=3.60
c        fmax=1.056
c        gmax=1.+c

        icount=0
        ekmax=(srt-dm3-dm4)/2.
        if(ekmax.le.aka) return
        pkmax=sqrt(ekmax**2-aka**2)

        if(dm3.le.0.0.or.dm4.le.0.0) then
           write(1,*)'error: minus mass!!!'
           return
        endif

c        after we have the momenta for both nucleus, we sample the
c        transverse momentum for K-. 
c        dsigma/dpt**2 = exp(-4.145*pt**2) obtained by fitting data on
c        page 72, fig 23i.
50        continue
        icount=icount+1
        if(icount.gt.10) return
        ptkmi2=-1./4.145*alog(RANART(NSEED))
        ptkm=sqrt(ptkmi2)
3        v1=RANART(NSEED)
        v2=RANART(NSEED)
        rsq=v1**2+v2**2
        if(rsq.ge.1.0.or.rsq.le.0.) goto 3
        fac=sqrt(-2.*alog(rsq)/rsq)
        guass=v1*fac
        if(guass.ge.5.) goto 3
        xstar=guass/5.
        pzkm=pkmax*xstar
        ekm=sqrt(aka**2+pzkm**2+ptkm**2)
        if(RANART(NSEED).gt.aka/ekm) goto 50
        bbb=RANART(NSEED)
        px(3)=ptkm*cos(2.*pio*bbb)
        py(3)=ptkm*sin(2.*pio*bbb)
        if(RANART(NSEED).gt.0.5) pzkm=-1.*pzkm
        pz(3)=pzkm
        pe(3)=ekm
150        ptkpl2=-1./3.68*alog(RANART(NSEED))
        ptkp=sqrt(ptkpl2)
13        v1=RANART(NSEED)
        v2=RANART(NSEED)
        rsq=v1**2+v2**2
        if(rsq.ge.1.0.or.rsq.le.0.) goto 13
        fac=sqrt(-2.*alog(rsq)/rsq)
        guass=v1*fac
        if(guass.ge.3.25) goto 13
        xstar=guass/3.25
        pzkp=pkmax*xstar
        ekp=sqrt(aka**2+pzkp**2+ptkp**2)
        if(RANART(NSEED).gt.aka/ekp) goto 150
        bbb=RANART(NSEED)
        px(4)=ptkp*cos(2.*pio*bbb)
        py(4)=ptkp*sin(2.*pio*bbb)
        if(RANART(NSEED).gt.0.5) pzkp=-1.*pzkp
        pz(4)=pzkp
        pe(4)=ekp

        resten=srt-pe(3)-pe(4)
        restpz=-pz(3)-pz(4)
c     resample
        if(resten.le.abs(restpz)) goto 50
        restms=sqrt(resten**2-restpz**2)
c     resample 
        if(restms.lt.(dm3+dm4)) goto 50
        ptp2=-1./2.76*alog(RANART(NSEED))
        ptp=sqrt(ptp2)
        bbb=RANART(NSEED)
        px(2)=ptp*cos(2.*pio*bbb)
        py(2)=ptp*sin(2.*pio*bbb)
        px(1)=-1.*(px(4)+px(3)+px(2))
        py(1)=-1.*(py(4)+py(3)+py(2))
c     transverse mass for K-
        rmt3=sqrt(dm3**2+px(1)**2+py(1)**2)
c     transverse mass for K+
        rmt4=sqrt(dm4**2+px(2)**2+py(2)**2)
        if(restms.lt.(rmt3+rmt4)) goto 50
c        else: sampling success!
        pzcms=sqrt((restms**2-(rmt3+rmt4)**2)*
     &             (restms**2-(rmt3-rmt4)**2))/2./restms
        if(RANART(NSEED).gt.0.5) then
           pz(1)=pzcms
           pz(2)=-pzcms
        else
           pz(1)=-pzcms
           pz(2)=pzcms
        endif
        beta=restpz/resten        
        gama=1./sqrt(1.-beta**2)
        pz(1)=pz(1)*gama + beta*gama*sqrt(rmt3**2+pz(1)**2)
        pz(2)=pz(2)*gama + beta*gama*sqrt(rmt4**2+pz(2)**2)
        pe(1)=sqrt(rmt3**2+pz(1)**2)
        pe(2)=sqrt(rmt4**2+pz(2)**2)

        iflag=1
        return
        end

c-----------------------------------------------------------------------

c.....extracted from G. Song's ART expasion including K- interactions
c.....file `NPIK.FOR'

****************************************
c        subroutine npik(irun,iseed,dt,nt,ictrl,i1,i2,srt,
c     &                  pcx,pcy,pcz,nchrg,ratiok)
        subroutine npik(irun,iseed,dt,nt,ictrl,i1,i2,srt,
     &                  pcx,pcy,pcz,nchrg,ratiok,iblock)
*
* Process: PI + N -> K(-) + ANYTHING
* 1.  PI- + P -> P + K0 + K-
* 2.  PI+ + N -> P + K+ + K- 
* 3.  PI0 + P -> P + K+ + K-
* 4.  PI0 + N -> P + K0 + K-
* 5.  PI0 + N -> N + K+ + K-
* 6.  PI- + P -> N + K+ + K-
* 7.  PI- + N -> N + K0 + K-
* NOTE: the mass of K is assumed to be same as K0. ie. 0.498 NOT 0.494
****************************************
      PARAMETER      (MAXSTR=150001,MAXR=1,PI=3.1415926)
      PARAMETER      (AKA=0.498)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /RUN/NUM
cc      SAVE /RUN/
      COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
      COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
      COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
      COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
      dimension bb(3),p1(4),p2(4),p3(4),px(4),py(4),pz(4)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      SAVE   
        iseed=iseed
        dt=dt
        nchrg=nchrg
        nt=nt
        ratiok=ratiok
        px(1)=px(1)
        py(1)=py(1)
        pz(1)=pz(1)
        px1cm=pcx
        py1cm=pcy
        pz1cm=pcz
        ictrl = 1
        lb1=lb(i1)
        lb2=lb(i2)
        k1=i1
        k2=i2
c        k1 must be bayron. k2 be meson. If not, exchange.
        if(lb2.eq.1.or.lb2.eq.2.or.(lb2.ge.6.and.lb2.le.13)) then
            k1=i2
            k2=i1
        endif
cbz3/8/99 neutralk
cbz10/12/99
c        LB(I1) = 1 + 2 * RANART(NSEED)
c        LB(I2) = 23
        LB(k1) = 1 + int(2*RANART(NSEED))
        LB(k2) = 23
c       pkmax=sqrt((srt**2-(aka+0.938+aka)**2)*(srt**2-(aka+0.938-aka)**2))
c     &           /2./srt
        pkmax=sqrt((srt**2-(aka+0.938+aka)**2)
     &           *(srt**2-(aka+0.938-aka)**2))/2./srt
        pk = RANART(NSEED)*pkmax
c-----------------------------------------------------
        css=1.-2.*RANART(NSEED)
        sss=sqrt(1.-css**2)
        fai=2*3.1415926*RANART(NSEED)
        p3(1)=pk*sss*cos(fai)
        p3(2)=pk*sss*sin(fai)
        p3(3)=pk*css
        eip = srt - sqrt(aka**2 + pk**2)
        rmnp=sqrt(eip**2-pk**2)
        do 1001 i= 1, 3
           bb(i) = -1.*p3(i)/eip
 1001   continue
c        bb: velocity of the other two particles as a whole.
        pznp=sqrt((rmnp**2-(aka+0.938)**2)
     c  *(rmnp**2-(0.938-aka)**2))/2./rmnp    
c-----------------------------------------------------
        css=1.-2.*RANART(NSEED)
        sss=sqrt(1.-css**2)
        fai=2*3.1415926*RANART(NSEED)
        p1(1)=pznp*sss*cos(fai)
        p1(2)=pznp*sss*sin(fai)
        p1(3)=pznp*css
        p1(4)=sqrt(0.938**2+pznp**2)
        p2(4)=sqrt(aka**2+pznp**2)
        do 1002 i=1,3
           p2(i)=-1.*p1(i)
 1002   continue
c        p1,p2: the momenta of the two particles in their cms
c        p1: momentum of N or P
c        p2: momentum of anti_kaon
c        p3: momentum of K0 or K+
        ilo=1
c        write(61,*)'--------p1,p2',p1,p2
c        write(61,*)'--------bb',bb
        call lorntz(ilo,bb,p1,p2)
c******* Checking *************
c        pxsum = p1(1)+p2(1)+p3(1)
c        pysum = p1(2)+p2(2)+p3(2)
c        pzsum = p1(3)+p2(3)+p3(3)
c        pesum = p1(4)+p2(4)+sqrt(p3(1)**2+p3(2)**2+p3(3)**2+aka**2)-srt
c        write(61,*)'---p1,pxsum',p1,pxsum
c        write(61,*)'---p2,pysum',p2,pysum
c        write(61,*)'---p3,pzsum',p3,pzsum
c        write(61,*)'---pesum',pesum
c***********************************

* Rotate the momenta of particles in the cms of I1 & I2
* px(1), py(1), pz(1): momentum of I1
* px(2), py(2), pz(2): momentum of I2
* px(3), py(3), pz(3): momentum of anti-kaon

c     10/28/02 get rid of argument usage mismatch in rotate():
        pxrota=p1(1)
        pyrota=p1(2)
        pzrota=p1(3)
c        call rotate(pcx,pcy,pcz,p1(1),p1(2),p1(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p1(1)=pxrota
        p1(2)=pyrota
        p1(3)=pzrota
c
        pxrota=p2(1)
        pyrota=p2(2)
        pzrota=p2(3)
c        call rotate(pcx,pcy,pcz,p2(1),p2(2),p2(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p2(1)=pxrota
        p2(2)=pyrota
        p2(3)=pzrota
c
        pxrota=p3(1)
        pyrota=p3(2)
        pzrota=p3(3)
c        call rotate(pcx,pcy,pcz,p3(1),p3(2),p3(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p3(1)=pxrota
        p3(2)=pyrota
        p3(3)=pzrota

        nnn=nnn+1
c     K(-)
        lpion(nnn,irun)=21
c     aka: rest mass of K
        epion(nnn,irun)=aka
* Find the momenta of particles in the final state in the nucleus_nucleus
* cms frame.   Lorentz transformation into lab frame.
        e1cm   = sqrt(0.938**2 + p1(1)**2 + p1(2)**2 + p1(3)**2)
        p1beta = p1(1)*betax + p1(2)*betay + p1(3)*betaz
        transf = gamma * ( gamma*p1beta / (gamma+1) + e1cm)
        pt1i1 = betax*transf + p1(1)
        pt2i1 = betay*transf + p1(2)
        pt3i1 = betaz*transf + p1(3)
        eti1  = 0.938
        lb1   = lb(k1)
         
* For second nulceon, same
        e2cm   = sqrt(aka**2 + p3(1)**2 + p3(2)**2 + p3(3)**2)
        p2beta = p3(1)*betax + p3(2)*betay + p3(3)*betaz
        transf = gamma * ( gamma*p2beta / (gamma+1) + e2cm)
        pt1i2 = betax*transf + p3(1)
        pt2i2 = betay*transf + p3(2)
        pt3i2 = betaz*transf + p3(3)
        eti2  = aka
        lb2   = lb(k2)

c        if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0.)then
*       k1 stand for nucleon, k2 stand for kaon. lpion stand for Kbar.
                p(1,k1)=pt1i1
                p(2,k1)=pt2i1
                p(3,k1)=pt3i1
                e(k1)=eti1
                lb(k1)=lb1
                p(1,k2)=pt1i2
                p(2,k2)=pt2i2
                p(3,k2)=pt3i2
                e(k2)=eti2
                lb(k2)=lb2

c                px1 = p(1,i1)
c                py1 = p(2,i1)
c                pz1 = p(3,i1)
c                em1 = e(i1)
c                id(i1) = 2
c                id(i2) = 2
c                id1 = id(i1)
c     K(+)K(-) production
                iblock = 101
* Get Kaons' momenta and coordinates in nucleus-nucleus cms. frame.
c  p2:  momentum of anti-kaon.
c        epcmk = sqrt(epion(nnn,irun)**2 + p2(1)**2 + p2(2)**2 + p2(3)**2)
        epcmk = sqrt(epion(nnn,irun)**2 + p2(1)**2+p2(2)**2+p2(3)**2)
        betak = p2(1)*betax + p2(2)*betay + p2(3)*betaz
        transf= gamma*(gamma*betak/(gamma+1.) + epcmk)
        ppion(1,nnn,irun)=betax*transf + p2(1)
        ppion(2,nnn,irun)=betay*transf + p2(2)
        ppion(3,nnn,irun)=betaz*transf + p2(3)
clin-5/2008:
        dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
cbz3/2/99
c        write(400,*)'2 ', ppion(1,nnn,irun), ppion(2,nnn,irun),
c     &                    ppion(3,nnn,irun), dt*nt, srt
cbz3/2/99end
c        write(420,*)ppion(1,nnn,irun), ppion(2,nnn,irun),
c     &                    ppion(3,nnn,irun), dt*nt, srt
        k=i2
        if(lb(i1).eq.1.or.lb(i1).eq.2) k=i1
        rpion(1,nnn,irun)=r(1,k)
        rpion(2,nnn,irun)=r(2,k)
        rpion(3,nnn,irun)=r(3,k)
        return
        end

c-----------------------------------------------------------------------

c.....extracted from G. Song's ART expasion including K- interactions
c.....file `PIHYPN.FOR'

******************************************
        subroutine pihypn(ielstc,irun,iseed,dt,nt,ictrl,i1,i2,
     &     srt,pcx,pcy,pcz,nchrg,iblock)
*
* Process: PI + sigma(or Lambda) -> Kbar + N
* NOTE: the mass of K is assumed to be same as K0. ie. 0.498 NOT 0.494
******************************************

c NOTE: for PI + Hyperon: the produced kaons have mass 0.498
      PARAMETER      (MAXSTR=150001,MAXR=1,PI=3.1415926)
      PARAMETER      (AKA=0.498)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /RUN/NUM
cc      SAVE /RUN/
      COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
      COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
      COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
      COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
      dimension p1(4),p2(4)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
        irun=irun
        iseed=iseed
        nt=nt
        dt=dt
        px1cm=pcx
        py1cm=pcy
        pz1cm=pcz
        ictrl = 1
csp06/07/01
        if(ielstc .eq. 1) then
*    L/Si + meson -> L/Si + meson
             k1=i1
             k2=i2
           dm3=e(k1)
           dm4=e(k2)
           iblock = 10
        else
           iblock = 12
csp06/07/01 end  
c        PI + Sigma(or Lambda) -> Kbar + N
        k1=i1
        k2=i2
c        k1 must be bayron! So if I1 is PI, exchange k1 & k2.
        if(lb(i1).lt.14.or.lb(i1).gt.17) then
           k1=i2
           k2=i1
        endif
cbz3/8/99 neutralk
        LB(K1) = 1 + int(2*RANART(NSEED))
        if(nchrg.eq.-2) lb(k1)=6
c     if(nchrg.eq.-1) lb(k1)=2
c     if(nchrg.eq. 0) lb(k1)=1
c     if(nchrg.eq. 1) lb(k1)=9
        IF (NCHRG .EQ. 2) LB(K1) = 9
cbz3/8/99 neutralk end

c     K-
        lb(k2)=21
        dm3=0.938
        if(nchrg.eq.-2.or.nchrg.eq.1) dm3=1.232
        dm4=aka
c        dm3,dm4: the mass of final state particles.
         endif
    
********Now, antikaon will be created.
c        call antikaon_fstate(iseed,srt,dm1,dm2,dm3,dm4,px,py,pz,icou1)
c        pkmax: the maximum momentum of anti-kaon
        pkmax=sqrt((srt**2-(dm3+dm4)**2)*(srt**2-(dm3-dm4)**2))
     &         /2./srt
        pk=pkmax
c-----------------------------------------------------
        css=1.-2.*RANART(NSEED)
        sss=sqrt(1.-css**2)
        fai=2*3.1415926*RANART(NSEED)
        p1(1)=pk*sss*cos(fai)
        p1(2)=pk*sss*sin(fai)
        p1(3)=pk*css
        do 1001 i=1,3
           p2(i)=-1.*p1(i)
 1001   continue
c        p1,p2: the momenta of the two particles in their cms
c        p1: momentum of kaon
c        p2: momentum of Kbar

* Rotate the momenta of particles in the cms of I1 & I2
clin-10/28/02 get rid of argument usage mismatch in rotate():
        pxrota=p1(1)
        pyrota=p1(2)
        pzrota=p1(3)
c        call rotate(pcx,pcy,pcz,p1(1),p1(2),p1(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p1(1)=pxrota
        p1(2)=pyrota
        p1(3)=pzrota
c
        pxrota=p2(1)
        pyrota=p2(2)
        pzrota=p2(3)
c        call rotate(pcx,pcy,pcz,p2(1),p2(2),p2(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p2(1)=pxrota
        p2(2)=pyrota
        p2(3)=pzrota
clin-10/28/02-end

* Find the momenta of particles in the final state in the nucleus_nucleus
* cms frame.   Lorentz transformation into lab frame.
        e1cm   = sqrt(dm3**2 + p1(1)**2 + p1(2)**2 + p1(3)**2)
        p1beta = p1(1)*betax + p1(2)*betay + p1(3)*betaz
        transf = gamma * ( gamma*p1beta / (gamma+1) + e1cm)
        pt1i1 = betax*transf + p1(1)
        pt2i1 = betay*transf + p1(2)
        pt3i1 = betaz*transf + p1(3)
        eti1  = dm3
        lb1   = lb(k1)
         
* For second kaon, same
        e2cm   = sqrt(dm4**2 + p2(1)**2 + p2(2)**2 + p2(3)**2)
        p2beta = p2(1)*betax + p2(2)*betay + p2(3)*betaz
        transf = gamma * ( gamma*p2beta / (gamma+1) + e2cm)
        pt1i2 = betax*transf + p2(1)
        pt2i2 = betay*transf + p2(2)
        pt3i2 = betaz*transf + p2(3)
cbz3/2/99
c        write(400,*)'3 ', pt1i2, pt2i2, pt3i2, dt*nt, srt
cbz3/2/99end
c        write(430,*)pt1i2, pt2i2, pt3i2, dt*nt, srt
        eti2  = dm4
        lb2   = lb(k2)

c        if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0.)then
c        k1=i1
c        k2=i2
*       k1 stand for nucleon, k2 stand for kaon.
                p(1,k1)=pt1i1
                p(2,k1)=pt2i1
                p(3,k1)=pt3i1
                e(k1)=eti1
                lb(k1)=lb1
                p(1,k2)=pt1i2
                p(2,k2)=pt2i2
                p(3,k2)=pt3i2
                e(k2)=eti2
                lb(k2)=lb2

cc                iblock = 101  ! K(+)K(-) production
* Get Kaons' momenta and coordinates in nucleus-nucleus cms. frame.
        return
        end

c-----------------------------------------------------------------------

c.....extracted from G. Song's ART expasion including K- interactions
c.....file `KAONN.FOR'

****************************************
        subroutine kaonN(brel,brsgm,irun,iseed,dt,nt,
     &     ictrl,i1,i2,iblock,srt,pcx,pcy,pcz,nchrg)
*
* Process: PI + sigma(or Lambda) <- Kbar + N
* NOTE: the mass of K is assumed to be same as K0. ie. 0.498 NOT 0.494
****************************************
      PARAMETER      (MAXSTR=150001,MAXR=1,PI=3.1415926)
      PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /RUN/NUM
cc      SAVE /RUN/
      COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
      COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
      COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
      COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
      dimension p1(4),p2(4)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
        dt=dt
        irun=irun
        iseed=iseed
        nchrg=nchrg
        nt=nt
        px1cm=pcx
        py1cm=pcy
        pz1cm=pcz
        ictrl = 1
c        ratio: used for isospin decision.
        k1=i1
        k2=i2
c        k1 must be bayron! So if I1 is Kaon, exchange k1 & k2.
        if(e(i1).lt.0.5.and.e(i1).gt.0.01) then
           k1=i2
           k2=i1
        endif
*** note: for print out only *******************************
c     record kaon's mass
        eee=e(k2)
*** end **************
        rrr=RANART(NSEED)
        if(rrr.lt.brel) then
c       Kbar + N -> Kbar + N
           lb1=lb(k1)
           lb2=lb(k2)
           em1=e(k1)
           em2=e(k2)
           iblock = 10
        else 
           iblock = 12
        if(rrr.lt.(brel+brsgm)) then
c        nchrg: Net charges of the two incoming particles.
c           Kbar + N -> Sigma + PI
           em1=asa
           em2=0.138

cbz3/8/99 neutralk
           LB1 = 15 + int(3*RANART(NSEED))
           LB2 = 3 + int(3*RANART(NSEED))
        else
c           Kbar + N -> Lambda + PI
           em1=ala
           em2=0.138
c     LAmbda
           lb1=14
cbz3/8/99 neutralk
           LB2 = 3 + int(3*RANART(NSEED))
c           if(nchrg.eq.1)  lb2=5  ! K- + D++ -> Lambda + PI+
c           if(nchrg.eq.0)  lb2=4  ! K- + p(D+,N*+) -> Lambda + PI0
c          if(nchrg.eq.-1) lb2=3 ! K- + n(D,N*) -> Lambda + PI-
cbz3/8/99 neutralk

        endif
        endif
        lb(k1)=lb1
        lb(k2)=lb2
    
********Now, antikaon will be created.
c        call antikaon_fstate(iseed,srt,dm1,dm2,dm3,dm4,px,py,pz,icou1)
c        pkmax: the maximum momentum of anti-kaon
c        write(63,*)'srt,em1,em2',srt,em1,em2
c        write(63,*)'-srt,em1,em2',srt,em1,em2
        pkmax=sqrt((srt**2-(em1+em2)**2)*(srt**2-(em1-em2)**2))
     &         /2./srt
        pk=pkmax
c-----------------------------------------------------
        css=1.-2.*RANART(NSEED)
        sss=sqrt(1.-css**2)
        fai=2*3.1415926*RANART(NSEED)
        p1(1)=pk*sss*cos(fai)
        p1(2)=pk*sss*sin(fai)
        p1(3)=pk*css
        do 1001 i=1,3
           p2(i)=-1.*p1(i)
 1001   continue
c        p1,p2: the momenta of the two particles in their cms
c        p1: momentum of kaon
c        p2: momentum of Kbar

* Rotate the momenta of particles in the cms of I1 & I2

clin-10/28/02 get rid of argument usage mismatch in rotate():
        pxrota=p1(1)
        pyrota=p1(2)
        pzrota=p1(3)
c        call rotate(pcx,pcy,pcz,p1(1),p1(2),p1(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p1(1)=pxrota
        p1(2)=pyrota
        p1(3)=pzrota
c
        pxrota=p2(1)
        pyrota=p2(2)
        pzrota=p2(3)
c        call rotate(pcx,pcy,pcz,p2(1),p2(2),p2(3))
        call rotate(pcx,pcy,pcz,pxrota,pyrota,pzrota)
        p2(1)=pxrota
        p2(2)=pyrota
        p2(3)=pzrota
clin-10/28/02-end

* Find the momenta of particles in the final state in the nucleus_nucleus
* cms frame.   Lorentz transformation into lab frame.
        e1cm   = sqrt(em1**2 + p1(1)**2 + p1(2)**2 + p1(3)**2)
        p1beta = p1(1)*betax + p1(2)*betay + p1(3)*betaz
        transf = gamma * ( gamma*p1beta / (gamma+1) + e1cm)
        pt1i1 = betax*transf + p1(1)
        pt2i1 = betay*transf + p1(2)
        pt3i1 = betaz*transf + p1(3)
        eti1  = em1
         
* For second kaon, same
        e2cm   = sqrt(em2**2 + p2(1)**2 + p2(2)**2 + p2(3)**2)
        p2beta = p2(1)*betax + p2(2)*betay + p2(3)*betaz
        transf = gamma * ( gamma*p2beta / (gamma+1) + e2cm)
        pt1i2 = betax*transf + p2(1)
        pt2i2 = betay*transf + p2(2)
        pt3i2 = betaz*transf + p2(3)
        eti2  = em2

c        if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0.)then
c        k1=i1
c        k2=i2
*       k1 stand for bayron, k2 stand for meson.
                p(1,k1)=pt1i1
                p(2,k1)=pt2i1
                p(3,k1)=pt3i1
                e(k1)=eti1
                p(1,k2)=pt1i2
                p(2,k2)=pt2i2
                p(3,k2)=pt3i2
                e(k2)=eti2

cc                iblock = 101  ! K(+)K(-) production
* Get Kaons' momenta and coordinates in nucleus-nucleus cms. frame.
        return
        end

c=======================================================================

clin Below is the previous artana.f:
c=======================================================================

c.....analysis subroutine before the hadronic space-time evolution

      SUBROUTINE ARTAN1
      PARAMETER (MAXSTR=150001, MAXR=1)
c.....y cut for mt spectrum
cbz3/17/99
c      PARAMETER (YMT1 = -0.4, YMT2 = 0.4)
      PARAMETER (YMT1 = -1.0, YMT2 = 1.0)
cbz3/17/99 end
c.....bin width for mt spectrum and y spectrum
clin-9/26/03 no symmetrization in y (or eta) for ana/*.dat:
c      PARAMETER (BMT = 0.05, BY = 0.2)
      PARAMETER (BMT = 0.05, BY = 0.4)
      COMMON /RUN/ NUM
cc      SAVE /RUN/
      COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cbz3/17/99
c     &     dm1k0s(50), DMT1LA(50), DMT1LB(50)
cc      SAVE /ARPRC1/
      COMMON /ARANA1/
     &     dy1ntb(50), dy1ntp(50), DY1HM(50), 
     &     DY1KP(50), DY1KM(50), DY1K0S(50),
     &     DY1LA(50), DY1LB(50), DY1PHI(50),
     &     dm1pip(50), dm1pim(50), DMT1PR(50),
     &     DMT1PB(50), DMT1KP(50), dm1km(50),
     &     dm1k0s(50), DMT1LA(50), DMT1LB(50),
     &     dy1msn(50), DY1PIP(50), DY1PIM(50), 
     &     DY1PI0(50), DY1PR(50), DY1PB(50)
     &     ,DY1NEG(50), DY1CH(50), DE1NEG(50), DE1CH(50)
cc      SAVE /ARANA1/
      SAVE   

cbz3/17/99 end
      DO 1002 J = 1, NUM
         DO 1001 I = 1, MULTI1(J)
            ITYP = ITYP1(I, J)
            PX = PX1(I, J)
            PY = PY1(I, J)
            PZ = PZ1(I, J)
            EE = EE1(I, J)
            XM = XM1(I, J)
c     2/24/03 leptons and photons:
            if(xm.lt.0.01) goto 200
            ptot = sqrt(PX ** 2 + PY ** 2 + pz ** 2)
            eta = 0.5*alog((Ptot+pz+1e-5)/(ptot-pz+1e-5))

            XMT = SQRT(PX ** 2 + PY ** 2 + XM ** 2)
            IF (ABS(PZ) .GE. EE) THEN
               PRINT *, 'IN ARTAN1'
               PRINT *, 'PARTICLE ', I, ' RUN ', J, 'PREC ERR'
cbzdbg2/16/99
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
cbzdbg2/16/99
cbzdbg2/15/99
               PRINT *, ' PZ = ', PZ, ' EE = ', EE
cbzdbg2/16/99
               PRINT *, ' XM = ', XM
cbzdbg2/16/99end
               GOTO 200
c               STOP
cbzdbg2/15/99end
            END IF
            DXMT = XMT - XM
            Y = 0.5 * LOG((EE + PZ) / (EE - PZ))
c.....rapidity cut for the rapidity distribution
            IF (ABS(Y) .GE. 10.0) GOTO 100
clin-9/26/03 no symmetrization in y (or eta) for ana/*.dat:
c            IY = 1 + int(ABS(Y) / BY)
c            Ieta = 1 + int(ABS(eta) / BY)
            IF (ABS(eta) .GE. 10.0) GOTO 100
            IY = 1 + int((Y+10.) / BY)
            Ieta = 1 + int((eta+10.) / BY)

            IF (ITYP .LT. -1000) THEN
               dy1ntb(IY) = dy1ntb(IY) - 1.0
            END IF
            IF (ITYP .GT. 1000) THEN
               dy1ntb(IY) = dy1ntb(IY) + 1.0
            END IF
            IF (ITYP .EQ. -2212) THEN
               dy1ntp(IY) = dy1ntp(IY) - 1.0
            END IF
            IF (ITYP .EQ. 2212) THEN
               dy1ntp(IY) = dy1ntp(IY) + 1.0
            END IF
c            IF (ITYP .EQ. -211 .OR. ITYP .EQ. -321 .OR.
c     &         ITYP .EQ. -2212) THEN
            IF (ITYP .EQ. -2112) THEN
               DY1HM(IY) = DY1HM(IY) + 1.0
            END IF
c
            IF (LUCHGE(ITYP).ne.0) THEN
               DY1CH(IY) = DY1CH(IY) + 1.0
               DE1CH(Ieta) = DE1CH(Ieta) + 1.0
               IF (LUCHGE(ITYP).lt.0) THEN
                  DY1NEG(IY) = DY1NEG(IY) + 1.0
                  DE1NEG(Ieta) = DE1NEG(Ieta) + 1.0
               endif
            END IF

cbz3/17/99
            IF ((ITYP .GE. 100 .AND. ITYP .LT. 1000) .OR. 
     &         (ITYP .GT. -1000 .AND. ITYP .LE. -100)) THEN
               dy1msn(IY) = dy1msn(IY) + 1.0
            END IF
            IF (ITYP .EQ. 211) THEN
               DY1PIP(IY) = DY1PIP(IY) + 1.0
            END IF
            IF (ITYP .EQ. -211) THEN
               DY1PIM(IY) = DY1PIM(IY) + 1.0
            END IF
            IF (ITYP .EQ. 111) THEN
               DY1PI0(IY) = DY1PI0(IY) + 1.0
            END IF
            IF (ITYP .EQ. 2212) THEN
               DY1PR(IY) = DY1PR(IY) + 1.0
            END IF
            IF (ITYP .EQ. -2212) THEN
               DY1PB(IY) = DY1PB(IY) + 1.0
            END IF
cbz3/17/99 end
            IF (ITYP .EQ. 321) THEN
               DY1KP(IY) = DY1KP(IY) + 1.0
            END IF
            IF (ITYP .EQ. -321) THEN
               DY1KM(IY) = DY1KM(IY) + 1.0
            END IF
clin-4/24/03 evaluate K0L instead of K0S, since sometimes we may decay K0S:
c            IF (ITYP .EQ. 310) THEN
            IF (ITYP .EQ. 130) THEN
               DY1K0S(IY) = DY1K0S(IY) + 1.0
            END IF
            IF (ITYP .EQ. 3122) THEN
               DY1LA(IY) = DY1LA(IY) + 1.0
            END IF
            IF (ITYP .EQ. -3122) THEN
               DY1LB(IY) = DY1LB(IY) + 1.0
            END IF
            IF (ITYP .EQ. 333) THEN
               DY1PHI(IY) = DY1PHI(IY) + 1.0
            END IF

c.....insert rapidity cut for mt spectrum here
 100        IF (Y .LT. YMT1 .OR. Y .GT. YMT2) GOTO 200
            IF (DXMT .GE. 50.0 * BMT .OR. DXMT .EQ. 0) GOTO 200
            IMT = 1 + int(DXMT / BMT)
            IF (ITYP .EQ. 211) THEN
               dm1pip(IMT) = dm1pip(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -211) THEN
               dm1pim(IMT) = dm1pim(IMT) + 
     &            1.0 / XMT
            END IF
            IF (ITYP .EQ. 2212) THEN
               DMT1PR(IMT) = DMT1PR(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -2212) THEN
               DMT1PB(IMT) = DMT1PB(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. 321) THEN
               DMT1KP(IMT) = DMT1KP(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -321) THEN
               dm1km(IMT) = dm1km(IMT) + 1.0 / XMT
            END IF
clin-4/24/03:
c            IF (ITYP .EQ. 310) THEN
            IF (ITYP .EQ. 130) THEN
               dm1k0s(IMT) = dm1k0s(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. 3122) THEN
               DMT1LA(IMT) = DMT1LA(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -3122) THEN
               DMT1LB(IMT) = DMT1LB(IMT) + 1.0 / XMT
            END IF

 200        CONTINUE
 1001    CONTINUE
 1002 CONTINUE

      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine after the hadronic space-time evolution

      SUBROUTINE ARTAN2

      PARAMETER (MAXSTR=150001, MAXR=1)
c.....y cut for mt spectrum
cbz3/17/99
c      PARAMETER (YMT1 = -0.4, YMT2 = 0.4)
      PARAMETER (YMT1 = -1.0, YMT2 = 1.0)
cbz3/17/99 end
c.....bin width for mt spectrum and y spectrum
c      PARAMETER (BMT = 0.05, BY = 0.2)
      PARAMETER (BMT = 0.05, BY = 0.4)
      COMMON /RUN/ NUM
cc      SAVE /RUN/
      COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cbz3/17/99
c     &     dm2k0s(50), DMT2LA(50), DMT2LB(50)
cc      SAVE /ARPRC1/
      COMMON /ARANA2/
     &     dy2ntb(50), dy2ntp(50), DY2HM(50), 
     &     DY2KP(50), DY2KM(50), DY2K0S(50),
     &     DY2LA(50), DY2LB(50), DY2PHI(50),
     &     dm2pip(50), dm2pim(50), DMT2PR(50),
     &     DMT2PB(50), DMT2KP(50), dm2km(50),
     &     dm2k0s(50), DMT2LA(50), DMT2LB(50),
     &     dy2msn(50), DY2PIP(50), DY2PIM(50), 
     &     DY2PI0(50), DY2PR(50), DY2PB(50)
     &     ,DY2NEG(50), DY2CH(50), DE2NEG(50), DE2CH(50)
cbz3/17/99 end
cc      SAVE /ARANA2/
      SAVE   

      DO 1002 J = 1, NUM
         DO 1001 I = 1, MULTI1(J)
            ITYP = ITYP1(I, J)
            PX = PX1(I, J)
            PY = PY1(I, J)
            PZ = PZ1(I, J)
            EE = EE1(I, J)
            XM = XM1(I, J)
            XMT = SQRT(PX ** 2 + PY ** 2 + XM ** 2)
c     2/24/03 leptons and photons:
            if(xm.lt.0.01) goto 200
            ptot = sqrt(PX ** 2 + PY ** 2 + pz ** 2)
            eta = 0.5*alog((Ptot+pz+1e-5)/(ptot-pz+1e-5))

            IF (ABS(PZ) .GE. EE) THEN
               PRINT *, 'IN ARTAN2'
               PRINT *, 'PARTICLE ', I, ' RUN ', J, 'PREC ERR'
cbzdbg2/16/99
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
cbzdbg2/16/99
cbzdbg2/15/99
               PRINT *, ' PZ = ', PZ, ' EE = ', EE
cbzdbg2/16/99
               PRINT *, ' XM = ', XM
cbzdbg2/16/99end
               GOTO 200
c               STOP
cbzdbg2/15/99end
            END IF
            DXMT = XMT - XM
            Y = 0.5 * LOG((EE + PZ) / (EE - PZ))
c.....rapidity cut for the rapidity distribution
            IF (ABS(Y) .GE. 10.0) GOTO 100
c            IY = 1 + int(ABS(Y) / BY)
c            Ieta = 1 + int(ABS(eta) / BY)
            IF (ABS(eta) .GE. 10.0) GOTO 100
            IY = 1 + int((Y+10.) / BY)
            Ieta = 1 + int((eta+10.) / BY)

            IF (ITYP .LT. -1000) THEN
               dy2ntb(IY) = dy2ntb(IY) - 1.0
            END IF
            IF (ITYP .GT. 1000) THEN
               dy2ntb(IY) = dy2ntb(IY) + 1.0
            END IF
            IF (ITYP .EQ. -2212) THEN
               dy2ntp(IY) = dy2ntp(IY) - 1.0
            END IF
            IF (ITYP .EQ. 2212) THEN
               dy2ntp(IY) = dy2ntp(IY) + 1.0
            END IF
            IF (ITYP .EQ. -2112) THEN
               DY2HM(IY) = DY2HM(IY) + 1.0
            END IF

            IF (LUCHGE(ITYP).ne.0) THEN
               DY2CH(IY) = DY2CH(IY) + 1.0
               DE2CH(Ieta) = DE2CH(Ieta) + 1.0
               IF (LUCHGE(ITYP).lt.0) THEN
                  DY2NEG(IY) = DY2NEG(IY) + 1.0
                  DE2NEG(Ieta) = DE2NEG(Ieta) + 1.0
               endif
            END IF

cbz3/17/99
            IF ((ITYP .GE. 100 .AND. ITYP .LT. 1000) .OR. 
     &         (ITYP .GT. -1000 .AND. ITYP .LE. -100)) THEN
               dy2msn(IY) = dy2msn(IY) + 1.0
            END IF
            IF (ITYP .EQ. 211) THEN
               DY2PIP(IY) = DY2PIP(IY) + 1.0
            END IF
            IF (ITYP .EQ. -211) THEN
               DY2PIM(IY) = DY2PIM(IY) + 1.0
            END IF
            IF (ITYP .EQ. 111) THEN
               DY2PI0(IY) = DY2PI0(IY) + 1.0
            END IF
            IF (ITYP .EQ. 2212) THEN
               DY2PR(IY) = DY2PR(IY) + 1.0
            END IF
            IF (ITYP .EQ. -2212) THEN
               DY2PB(IY) = DY2PB(IY) + 1.0
            END IF
cbz3/17/99 end
            IF (ITYP .EQ. 321) THEN
               DY2KP(IY) = DY2KP(IY) + 1.0
            END IF
            IF (ITYP .EQ. -321) THEN
               DY2KM(IY) = DY2KM(IY) + 1.0
            END IF
clin-4/24/03:
c            IF (ITYP .EQ. 310) THEN
            IF (ITYP .EQ. 130) THEN
               DY2K0S(IY) = DY2K0S(IY) + 1.0
            END IF
            IF (ITYP .EQ. 3122) THEN
               DY2LA(IY) = DY2LA(IY) + 1.0
            END IF
            IF (ITYP .EQ. -3122) THEN
               DY2LB(IY) = DY2LB(IY) + 1.0
            END IF
            IF (ITYP .EQ. 333) THEN
               DY2PHI(IY) = DY2PHI(IY) + 1.0
            END IF

c.....insert rapidity cut for mt spectrum here
 100        IF (Y .LT. YMT1 .OR. Y .GT. YMT2) GOTO 200
            IF (DXMT .GE. 50.0 * BMT .OR. DXMT .EQ. 0) GOTO 200
            IMT = 1 + int(DXMT / BMT)
            IF (ITYP .EQ. 211) THEN
               dm2pip(IMT) = dm2pip(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -211) THEN
               dm2pim(IMT) = dm2pim(IMT) + 
     &            1.0 / XMT
            END IF
            IF (ITYP .EQ. 2212) THEN
               DMT2PR(IMT) = DMT2PR(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -2212) THEN
               DMT2PB(IMT) = DMT2PB(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. 321) THEN
               DMT2KP(IMT) = DMT2KP(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -321) THEN
               dm2km(IMT) = dm2km(IMT) + 1.0 / XMT
            END IF
clin-4/24/03:
c            IF (ITYP .EQ. 310) THEN
            IF (ITYP .EQ. 130) THEN
               dm2k0s(IMT) = dm2k0s(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. 3122) THEN
               DMT2LA(IMT) = DMT2LA(IMT) + 1.0 / XMT
            END IF
            IF (ITYP .EQ. -3122) THEN
               DMT2LB(IMT) = DMT2LB(IMT) + 1.0 / XMT
            END IF

 200        CONTINUE
 1001    CONTINUE
 1002 CONTINUE

      RETURN
      END

c-----------------------------------------------------------------------

c.....output analysis results at the end of the simulation

      SUBROUTINE ARTOUT(NEVNT)

      PARAMETER (MAXSTR=150001, MAXR=1)
c.....y cut for mt spectrum
cbz3/17/99
c      PARAMETER (YMT1 = -0.4, YMT2 = 0.4)
      PARAMETER (YMT1 = -1.0, YMT2 = 1.0)
cbz3/17/99 end
c.....bin width for mt spectrum and y spectrum
c      PARAMETER (BMT = 0.05, BY = 0.2)
      PARAMETER (BMT = 0.05, BY = 0.4)
      COMMON /RUN/ NUM
cc      SAVE /RUN/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cbz3/17/99
c     &     dm1k0s(50), DMT1LA(50), DMT1LB(50)
cc      SAVE /ARPRC1/
      COMMON /ARANA1/
     &     dy1ntb(50), dy1ntp(50), DY1HM(50), 
     &     DY1KP(50), DY1KM(50), DY1K0S(50),
     &     DY1LA(50), DY1LB(50), DY1PHI(50),
     &     dm1pip(50), dm1pim(50), DMT1PR(50),
     &     DMT1PB(50), DMT1KP(50), dm1km(50),
     &     dm1k0s(50), DMT1LA(50), DMT1LB(50),
     &     dy1msn(50), DY1PIP(50), DY1PIM(50), 
     &     DY1PI0(50), DY1PR(50), DY1PB(50)
     &     ,DY1NEG(50), DY1CH(50), DE1NEG(50), DE1CH(50)
cbz3/17/99 end
cc      SAVE /ARANA1/
cbz3/17/99
c     &     dm2k0s(50), DMT2LA(50), DMT2LB(50)
      COMMON /ARANA2/
     &     dy2ntb(50), dy2ntp(50), DY2HM(50), 
     &     DY2KP(50), DY2KM(50), DY2K0S(50),
     &     DY2LA(50), DY2LB(50), DY2PHI(50),
     &     dm2pip(50), dm2pim(50), DMT2PR(50),
     &     DMT2PB(50), DMT2KP(50), dm2km(50),
     &     dm2k0s(50), DMT2LA(50), DMT2LB(50),
     &     dy2msn(50), DY2PIP(50), DY2PIM(50), 
     &     DY2PI0(50), DY2PR(50), DY2PB(50)
     &     ,DY2NEG(50), DY2CH(50), DE2NEG(50), DE2CH(50)
cc      SAVE /ARANA2/
      SAVE   
cbz3/17/99 end
cms   OPEN (30, FILE = 'ana/dndy_netb.dat', STATUS = 'UNKNOWN')
cms   OPEN (31, FILE = 'ana/dndy_netp.dat', STATUS = 'UNKNOWN')
cms   OPEN (32, FILE = 'ana/dndy_nb.dat', STATUS = 'UNKNOWN')
cms   OPEN (33, FILE = 'ana/dndy_neg.dat', STATUS = 'UNKNOWN')
cms   OPEN (34, FILE = 'ana/dndy_ch.dat', STATUS = 'UNKNOWN')
cms   OPEN (35, FILE = 'ana/dnde_neg.dat', STATUS = 'UNKNOWN')
cms   OPEN (36, FILE = 'ana/dnde_ch.dat', STATUS = 'UNKNOWN')
cms   OPEN (37, FILE = 'ana/dndy_kp.dat', STATUS = 'UNKNOWN')
cms   OPEN (38, FILE = 'ana/dndy_km.dat', STATUS = 'UNKNOWN')
clin-4/24/03
c      OPEN (39, FILE = 'ana/dndy_k0s.dat', STATUS = 'UNKNOWN')
cms   OPEN (39, FILE = 'ana/dndy_k0l.dat', STATUS = 'UNKNOWN')
cms   OPEN (40, FILE = 'ana/dndy_la.dat', STATUS = 'UNKNOWN')
cms   OPEN (41, FILE = 'ana/dndy_lb.dat', STATUS = 'UNKNOWN')
cms   OPEN (42, FILE = 'ana/dndy_phi.dat', STATUS = 'UNKNOWN')
cbz3/17/99
cms   OPEN (43, FILE = 'ana/dndy_meson.dat', STATUS = 'UNKNOWN')
cms   OPEN (44, FILE = 'ana/dndy_pip.dat', STATUS = 'UNKNOWN')
cms   OPEN (45, FILE = 'ana/dndy_pim.dat', STATUS = 'UNKNOWN')
cms   OPEN (46, FILE = 'ana/dndy_pi0.dat', STATUS = 'UNKNOWN')
cms   OPEN (47, FILE = 'ana/dndy_pr.dat', STATUS = 'UNKNOWN')
cms   OPEN (48, FILE = 'ana/dndy_pb.dat', STATUS = 'UNKNOWN')
cbz3/17/99 end

cms   OPEN (50, FILE = 'ana/dndmtdy_pip.dat', STATUS = 'UNKNOWN')
cms   OPEN (51, FILE = 'ana/dndmtdy_0_1_pim.dat', STATUS = 'UNKNOWN')
cms   OPEN (52, FILE = 'ana/dndmtdy_pr.dat', STATUS = 'UNKNOWN')
cms   OPEN (53, FILE = 'ana/dndmtdy_pb.dat', STATUS = 'UNKNOWN')
cms   OPEN (54, FILE = 'ana/dndmtdy_kp.dat', STATUS = 'UNKNOWN')
cms   OPEN (55, FILE = 'ana/dndmtdy_0_5_km.dat', STATUS = 'UNKNOWN')
cms   OPEN (56, FILE = 'ana/dndmtdy_k0s.dat', STATUS = 'UNKNOWN')
cms   OPEN (57, FILE = 'ana/dndmtdy_la.dat', STATUS = 'UNKNOWN')
cms   OPEN (58, FILE = 'ana/dndmtdy_lb.dat', STATUS = 'UNKNOWN')
clin-9/26/03 no symmetrization in y (or eta) for ana/*.dat:
c      SCALE1 = 1. / REAL(NEVNT * NUM) / BY / 2.0
      SCALE1 = 1. / REAL(NEVNT * NUM) / BY
      SCALE2 = 1. / REAL(NEVNT * NUM) / BMT / (YMT2 - YMT1)
c
      DO 1001 I = 1, 50
         ymid=-10.+BY * (I - 0.5)
cms      WRITE (30, 333) ymid, SCALE1 * dy1ntb(I)
cms      WRITE (31, 333) ymid, SCALE1 * dy1ntp(I)
cms      WRITE (32, 333) ymid, SCALE1 * DY1HM(I)
cms      WRITE (37, 333) ymid, SCALE1 * DY1KP(I)
cms      WRITE (38, 333) ymid, SCALE1 * DY1KM(I)
cms      WRITE (39, 333) ymid, SCALE1 * DY1K0S(I)
cms      WRITE (40, 333) ymid, SCALE1 * DY1LA(I)
cms      WRITE (41, 333) ymid, SCALE1 * DY1LB(I)
cms      WRITE (42, 333) ymid, SCALE1 * DY1PHI(I)
cms      WRITE (33, 333) ymid, SCALE1 * DY1NEG(I)
cms      WRITE (34, 333) ymid, SCALE1 * DY1CH(I)
cms      WRITE (35, 333) ymid, SCALE1 * DE1NEG(I)
cms      WRITE (36, 333) ymid, SCALE1 * DE1CH(I)
cms      WRITE (43, 333) ymid, SCALE1 * dy1msn(I)
cms      WRITE (44, 333) ymid, SCALE1 * DY1PIP(I)
cms      WRITE (45, 333) ymid, SCALE1 * DY1PIM(I)
cms      WRITE (46, 333) ymid, SCALE1 * DY1PI0(I)
cms      WRITE (47, 333) ymid, SCALE1 * DY1PR(I)
cms      WRITE (48, 333) ymid, SCALE1 * DY1PB(I)

         IF (dm1pip(I) .NE. 0.0) THEN
cms         WRITE (50, 333) BMT * (I - 0.5), SCALE2 * dm1pip(I)
         END IF
         IF (dm1pim(I) .NE. 0.0) THEN
cms         WRITE (51, 333) BMT * (I - 0.5), SCALE2 * 0.1 * 
cms  &         dm1pim(I)
         END IF
         IF (DMT1PR(I) .NE. 0.0) THEN
cms         WRITE (52, 333) BMT * (I - 0.5), SCALE2 * DMT1PR(I)
         END IF
         IF (DMT1PB(I) .NE. 0.0) THEN
cms         WRITE (53, 333) BMT * (I - 0.5), SCALE2 * DMT1PB(I)
         END IF
         IF (DMT1KP(I) .NE. 0.0) THEN
cms         WRITE (54, 333) BMT * (I - 0.5), SCALE2 * DMT1KP(I)
         END IF
         IF (dm1km(I) .NE. 0.0) THEN
cms         WRITE (55, 333) BMT * (I - 0.5), SCALE2 * 0.5 *
cms  &         dm1km(I)
         END IF
         IF (dm1k0s(I) .NE. 0.0) THEN
cms         WRITE (56, 333) BMT * (I - 0.5), SCALE2 * dm1k0s(I)
         END IF
         IF (DMT1LA(I) .NE. 0.0) THEN
cms         WRITE (57, 333) BMT * (I - 0.5), SCALE2 * DMT1LA(I)
         END IF
         IF (DMT1LB(I) .NE. 0.0) THEN
cms         WRITE (58, 333) BMT * (I - 0.5), SCALE2 * DMT1LB(I)
         END IF
 1001 CONTINUE
c
      DO 1002 I = 30, 48
cms      WRITE (I, *) 'after hadron evolution'
 1002    CONTINUE
      DO 1003 I = 50, 58
cms      WRITE (I, *) 'after hadron evolution'
 1003 CONTINUE

      DO 1004 I = 1, 50
         ymid=-10.+BY * (I - 0.5)
cms      WRITE (30, 333) ymid, SCALE1 * dy2ntb(I)
cms      WRITE (31, 333) ymid, SCALE1 * dy2ntp(I)
cms      WRITE (32, 333) ymid, SCALE1 * DY2HM(I)
cms      WRITE (37, 333) ymid, SCALE1 * DY2KP(I)
cms      WRITE (38, 333) ymid, SCALE1 * DY2KM(I)
cms      WRITE (39, 333) ymid, SCALE1 * DY2K0S(I)
cms      WRITE (40, 333) ymid, SCALE1 * DY2LA(I)
cms      WRITE (41, 333) ymid, SCALE1 * DY2LB(I)
cms      WRITE (42, 333) ymid, SCALE1 * DY2PHI(I)
cms      WRITE (33, 333) ymid, SCALE1 * DY2NEG(I)
cms      WRITE (34, 333) ymid, SCALE1 * DY2CH(I)
cms      WRITE (35, 333) ymid, SCALE1 * DE2NEG(I)
cms      WRITE (36, 333) ymid, SCALE1 * DE2CH(I)
cms      WRITE (43, 333) ymid, SCALE1 * dy2msn(I)
cms      WRITE (44, 333) ymid, SCALE1 * DY2PIP(I)
cms      WRITE (45, 333) ymid, SCALE1 * DY2PIM(I)
cms      WRITE (46, 333) ymid, SCALE1 * DY2PI0(I)
cms      WRITE (47, 333) ymid, SCALE1 * DY2PR(I)
cms      WRITE (48, 333) ymid, SCALE1 * DY2PB(I)
c
         IF (dm2pip(I) .NE. 0.0) THEN
cms         WRITE (50, 333) BMT * (I - 0.5), SCALE2 * dm2pip(I)
         END IF
         IF (dm2pim(I) .NE. 0.0) THEN
cms         WRITE (51, 333) BMT * (I - 0.5), SCALE2 * 0.1 * 
cms  &         dm2pim(I)
         END IF
         IF (DMT2PR(I) .NE. 0.0) THEN
cms         WRITE (52, 333) BMT * (I - 0.5), SCALE2 * DMT2PR(I)
         END IF
         IF (DMT2PB(I) .NE. 0.0) THEN
cms         WRITE (53, 333) BMT * (I - 0.5), SCALE2 * DMT2PB(I)
         END IF
         IF (DMT2KP(I) .NE. 0.0) THEN
cms         WRITE (54, 333) BMT * (I - 0.5), SCALE2 * DMT2KP(I)
         END IF
         IF (dm2km(I) .NE. 0.0) THEN
cms         WRITE (55, 333) BMT * (I - 0.5), SCALE2 * 0.5 * 
cms  &         dm2km(I)
         END IF
         IF (dm2k0s(I) .NE. 0.0) THEN
cms         WRITE (56, 333) BMT * (I - 0.5), SCALE2 * dm2k0s(I)
         END IF
         IF (DMT2LA(I) .NE. 0.0) THEN
cms         WRITE (57, 333) BMT * (I - 0.5), SCALE2 * DMT2LA(I)
         END IF
         IF (DMT2LB(I) .NE. 0.0) THEN
cms         WRITE (58, 333) BMT * (I - 0.5), SCALE2 * DMT2LB(I)
         END IF
 1004 CONTINUE
cms 333  format(2(f12.5,1x))

      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine in HIJING before parton cascade evolution
      SUBROUTINE HJANA1

      PARAMETER (YMAX = 1.0, YMIN = -1.0)
      PARAMETER (DMT = 0.05, DY = 0.2)
      PARAMETER (DR = 0.2)
      PARAMETER (MAXPTN=400001,MAXSTR=150001)
      DIMENSION dyp1(50), DMYP1(200), DEYP1(50)
      DIMENSION dyg1(50), DMYG1(200), DEYG1(50)
      DIMENSION SNYP1(50), SMYP1(200), SEYP1(50)
      DIMENSION SNYG1(50), SMYG1(200), SEYG1(50)
      DIMENSION dnrpj1(50), dnrtg1(50), dnrin1(50),
     &   dnrtt1(50)
      DIMENSION dyg1c(50), dmyg1c(50), deyg1c(50)
      DIMENSION snrpj1(50), snrtg1(50), snrin1(50),
     &   snrtt1(50)
      DIMENSION snyg1c(50), smyg1c(50), seyg1c(50)
      DOUBLE PRECISION  GX0, GY0, GZ0, FT0, PX0, PY0, PZ0, E0, XMASS0

      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
      COMMON/hjcrdn/YP(3,300),YT(3,300)
cc      SAVE /hjcrdn/
      COMMON/HJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &   PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &   PJPM(300,500),NTJ(300),KFTJ(300,500),
     &   PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &   PJTE(300,500),PJTM(300,500)
cc      SAVE /HJJET1/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &   K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &   PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      COMMON /prec1/GX0(MAXPTN),GY0(MAXPTN),GZ0(MAXPTN),FT0(MAXPTN),
     &     PX0(MAXPTN), PY0(MAXPTN), PZ0(MAXPTN), E0(MAXPTN),
     &     XMASS0(MAXPTN), ITYP0(MAXPTN)
cc      SAVE /prec1/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      SAVE   
      DATA IW/0/
      IF (isevt .EQ. IAEVT .AND. isrun .EQ. IARUN) THEN
         DO 1001 I = 1, 200
            DMYP1(I) = SMYP1(I)
            DMYG1(I) = SMYG1(I)
 1001    CONTINUE

         DO 1002 I = 1, 50
            dyp1(I) = SNYP1(I)
            DEYP1(I) = SEYP1(I)
            dyg1(I) = SNYG1(I)
            DEYG1(I) = SEYG1(I)
            dnrpj1(I) = snrpj1(I)
            dnrtg1(I) = snrtg1(I)
            dnrin1(I) = snrin1(I)
            dnrtt1(I) = snrtt1(I)
            dyg1c(I) = snyg1c(I)
            dmyg1c(I) = smyg1c(I)
            deyg1c(I) = seyg1c(I)
 1002    CONTINUE
         nsubp = nsubpS
         nsubg = nsubgS
         NISG = NISGS
      ELSE
         DO 1003 I = 1, 200
            SMYP1(I) = DMYP1(I)
            SMYG1(I) = DMYG1(I)
 1003    CONTINUE

         DO 1004 I = 1, 50
            SNYP1(I) = dyp1(I)
            SEYP1(I) = DEYP1(I)
            SNYG1(I) = dyg1(I)
            SEYG1(I) = DEYG1(I)
            snrpj1(I) = dnrpj1(I)
            snrtg1(I) = dnrtg1(I)
            snrin1(I) = dnrin1(I)
            snrtt1(I) = dnrtt1(I)
            snyg1c(I) = dyg1c(I)
            smyg1c(I) = dmyg1c(I)
            seyg1c(I) = deyg1c(I)
 1004    CONTINUE
         nsubpS = nsubp
         nsubgS = nsubg
         NISGS = NISG
         isevt = IAEVT
         isrun = IARUN
         IW = IW + 1
      END IF
c.....analysis
      DO 1006 I = 1, IHNT2(1)
         DO 1005 J = 1, NPJ(I)
            ITYP = KFPJ(I, J)
            PX = PJPX(I, J)
            PY = PJPY(I, J)
            PZ = PJPZ(I, J)
            PE = PJPE(I, J)
            PM = PJPM(I, J)
            IF (ABS(PZ) .GE. PE) THEN
               PRINT *, ' IN HJANA1, PROJ STR ', I, ' PART ', J
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', PE
               PRINT *, ' XM = ', PM
               GOTO 200
            END IF
            RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
            XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
            DXMT = XMT - PM
            IY = 1 + int(ABS(RAP) / DY)
            IF (IY .GT. 50) GOTO 100
            dyp1(IY) = dyp1(IY) + 1.0
            DEYP1(IY) = DEYP1(IY) + XMT
            IF (ITYP .EQ. 21) THEN
               dyg1(IY) = dyg1(IY) + 1.0
               DEYG1(IY) = DEYG1(IY) + XMT
            END IF
 100        CONTINUE
            IMT = 1 + int(DXMT / DMT)
            IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 200
            IF (IMT .GT. 200) GOTO 200
            DMYP1(IMT) = DMYP1(IMT) + 1.0 / XMT
            IF (ITYP .EQ. 21) THEN
               DMYG1(IMT) = DMYG1(IMT) + 1.0 / XMT
            END IF
 200        CONTINUE
 1005    CONTINUE
 1006 CONTINUE

      DO 1008 I = 1, IHNT2(3)
         DO 1007 J = 1, NTJ(I)
            ITYP = KFTJ(I, J)
            PX = PJTX(I, J)
            PY = PJTY(I, J)
            PZ = PJTZ(I, J)
            PE = PJTE(I, J)
            PM = PJTM(I, J)
            IF (ABS(PZ) .GE. PE) THEN
               PRINT *, ' IN HJANA1, TARG STR ', I, ' PART ', J
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', PE
               PRINT *, ' XM = ', PM
               GOTO 400
            END IF
            RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
            XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
            DXMT = XMT - PM
            IY = 1 + int(ABS(RAP) / DY)
            IF (IY .GT. 50) GOTO 300
            dyp1(IY) = dyp1(IY) + 1.0
            DEYP1(IY) = DEYP1(IY) + XMT
            IF (ITYP .EQ. 21) THEN
               dyg1(IY) = dyg1(IY) + 1.0
               DEYG1(IY) = DEYG1(IY) + XMT
            END IF
 300        CONTINUE
            IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 400
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 200) GOTO 400
            DMYP1(IMT) = DMYP1(IMT) + 1.0 / XMT
            IF (ITYP .EQ. 21) THEN
               DMYG1(IMT) = DMYG1(IMT) + 1.0 / XMT
            END IF
 400        CONTINUE
 1007    CONTINUE
 1008 CONTINUE

      DO 1010 I = 1, NSG
         DO 1009 J = 1, NJSG(I)
            ITYP = K2SG(I, J)
            PX = PXSG(I, J)
            PY = PYSG(I, J)
            PZ = PZSG(I, J)
            PE = PESG(I, J)
            PM = PMSG(I, J)
            IF (ABS(PZ) .GE. PE) THEN
               PRINT *, ' IN HJANA1, INDP STR ', I, ' PART ', J
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', PE
               PRINT *, ' XM = ', PM
               GOTO 600
            END IF
            RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
            XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
            DXMT = XMT - PM
            IY = 1 + int(ABS(RAP) / DY)
            IF (IY .GT. 50) GOTO 500
            dyp1(IY) = dyp1(IY) + 1.0
            DEYP1(IY) = DEYP1(IY) + XMT
            IF (ITYP .EQ. 21) THEN
               dyg1(IY) = dyg1(IY) + 1.0
               DEYG1(IY) = DEYG1(IY) + XMT
            END IF
 500        CONTINUE
            IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 600
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 200) GOTO 600
            DMYP1(IMT) = DMYP1(IMT) + 1.0 / XMT
            IF (ITYP .EQ. 21) THEN
               DMYG1(IMT) = DMYG1(IMT) + 1.0 / XMT
            END IF
 600        CONTINUE
 1009    CONTINUE
 1010 CONTINUE

      DO 1011 I = 1, IHNT2(1)
         YR = SQRT(YP(1, I) ** 2 + YP(2, I) ** 2)
         IR = 1 + int(YR / DR)
clin-4/2008 protect against out-of-bound errors:
c         IF (IR .GT. 50) GOTO 601
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 601
         dnrpj1(IR) = dnrpj1(IR) + 1.0
         dnrtt1(IR) = dnrtt1(IR) + 1.0
 601     CONTINUE
 1011 CONTINUE

      DO 1012 I = 1, IHNT2(3)
         YR = SQRT(YT(1, I) ** 2 + YT(2, I) ** 2)
         IR = 1 + int(YR / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 602
         dnrtg1(IR) = dnrtg1(IR) + 1.0
         dnrtt1(IR) = dnrtt1(IR) + 1.0
 602     CONTINUE
 1012 CONTINUE

      DO 1013 I = 1, NSG
         Y1 = 0.5 * (YP(1, IASG(I, 1)) + YT(1, IASG(I, 2)))
         Y2 = 0.5 * (YP(2, IASG(I, 1)) + YT(2, IASG(I, 2)))
         YR = SQRT(Y1 ** 2 + Y2 ** 2)
         IR = 1 + int(YR / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 603
         dnrin1(IR) = dnrin1(IR) + 1.0
         dnrtt1(IR) = dnrtt1(IR) + 1.0
 603     CONTINUE
 1013 CONTINUE

      DO 1014 I = 1, MUL
         ITYP = ITYP0(I)
         PX = sngl(PX0(I))
         PY = sngl(PY0(I))
         PZ = sngl(PZ0(I))
         PE = sngl(E0(I))
         PM = sngl(XMASS0(I))
         IF (ABS(PZ) .GE. PE) THEN
            PRINT *, ' IN HJANA1, GLUON ', I
            PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
            PRINT *, ' PZ = ', PZ, ' EE = ', PE
            PRINT *, ' XM = ', PM
            GOTO 800
         END IF
         RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
         XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
         DXMT = XMT - PM
         IY = 1 + int(ABS(RAP) / DY)
         IF (IY .GT. 50) GOTO 700
         dyg1c(IY) = dyg1c(IY) + 1.0
         deyg1c(IY) = deyg1c(IY) + XMT
 700     CONTINUE
         IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 800
         IMT = 1 + int(DXMT / DMT)
         IF (IMT .GT. 50) GOTO 800
         dmyg1c(IMT) = dmyg1c(IMT) + 1.0 / XMT
 800     CONTINUE
 1014 CONTINUE
c.....count number of particles
      DO 1016 I = 1, IHNT2(1)
         DO 1015 J = 1, NPJ(I)
            nsubp = nsubp + 1
            IF (KFPJ(I, J) .EQ. 21) nsubg = nsubg + 1
 1015    CONTINUE
 1016 CONTINUE

      DO 1018 I = 1, IHNT2(3)
         DO 1017 J = 1, NTJ(I)
            nsubp = nsubp + 1
            IF (KFTJ(I, J) .EQ. 21) nsubg = nsubg + 1
 1017    CONTINUE
 1018 CONTINUE

      DO 1020 I = 1, NSG
         DO 1019 J = 1, NJSG(I)
            nsubp = nsubp + 1
            IF (K2SG(I, J) .EQ. 21) nsubg = nsubg + 1
 1019    CONTINUE
 1020 CONTINUE
      NISG = NISG + NSG
      IF (IOUT .EQ. 1) THEN
cbzdbg2/16/99
c      PRINT *, ' in HJANA1 '
c      PRINT *, ' total number of partons = ', nsubp
c      PRINT *, ' total number of gluons = ', nsubg, MUL
c      PRINT *, ' number of projectile strings = ', IHNT2(1)
c      PRINT *, ' number of target strings = ', IHNT2(3)
c      PRINT *, ' number of independent strings = ', NSG
      PRINT *, ' in HJANA1 '
      PRINT *, ' total number of partons = ', nsubp / IW
      PRINT *, ' total number of gluons = ', nsubg / IW
c      PRINT *, ' number of projectile strings = ', IHNT2(1)
c      PRINT *, ' number of target strings = ', IHNT2(3)
      PRINT *, ' number of independent strings = ', NISG / IW
cbzdbg2/16/99end
      END IF
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine in ZPC after generation of additional initial
c.....phase space distributions.

      SUBROUTINE HJAN1A
      PARAMETER (MAXPTN=400001)
      PARAMETER (DGX = 0.2, DGY = 0.2, DT = 0.2)
      DIMENSION dgxg1a(50), dgyg1a(50), dtg1a(50)
      DIMENSION sgxg1a(50), sgyg1a(50), stg1a(50)
      DOUBLE PRECISION  GX5, GY5, GZ5, FT5, PX5, PY5, PZ5, E5, XMASS5
      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &   PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &   XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      SAVE   
      DATA IW/0/

      IF (isevt .EQ. IAEVT .AND. isrun .EQ. IARUN) THEN
         DO 1001 I = 1, 50
            dgxg1a(I) = sgxg1a(I)
            dgyg1a(I) = sgyg1a(I)
            dtg1a(I) = stg1a(I)
 1001    CONTINUE
      ELSE
         DO 1002 I = 1, 50
            sgxg1a(I) = dgxg1a(I)
            sgyg1a(I) = dgyg1a(I)
            stg1a(I) = dtg1a(I)
 1002    CONTINUE
         isevt = IAEVT
         isrun = IARUN
         IW = IW + 1
      END IF
c.....analysis
      DO 1003 I = 1, MUL
         IGX = 1 + int(sngl(ABS(GX5(I))) / DGX)
clin-4/2008 protect against out-of-bound errors:
c         IF (IGX .GT. 50) GOTO 100
         IF (IGX .GT. 50 .or. IGX .LT. 1) GOTO 100
         dgxg1a(IGX) = dgxg1a(IGX) + 1.0
 100     CONTINUE
         IGY = 1 + int(sngl(ABS(GY5(I))) / DGY)
         IF (IGY .GT. 50 .or. IGY .LT. 1) GOTO 200
         dgyg1a(IGY) = dgyg1a(IGY) + 1.0
 200     CONTINUE
         IT = 1 + int(sngl(SQRT(FT5(I) ** 2 - GZ5(I) ** 2)) / DT)
         IF (IT .GT. 50 .or. IT .LT. 1) GOTO 300
         dtg1a(IT) = dtg1a(IT) + 1.0
 300     CONTINUE
 1003 CONTINUE
      CALL HJAN1B
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine in HJAN1A

      SUBROUTINE HJAN1B
      PARAMETER (MAXPTN=400001,MAXSTR=150001)
      PARAMETER (DR = 0.2, DT = 0.2)
      DIMENSION DNRG1B(50), dtg1b(50)
      DIMENSION SNRG1B(50), stg1b(50)
      DOUBLE PRECISION  GX5, GY5, GZ5, FT5, PX5, PY5, PZ5, E5, XMASS5
      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &   PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &   XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
      COMMON /ilist8/ LSTRG1(MAXPTN), LPART1(MAXPTN)
cc      SAVE /ilist8/
      COMMON /SREC1/ NSP, NST, NSI
cc      SAVE /SREC1/
      COMMON/hjcrdn/YP(3,300),YT(3,300)
cc      SAVE /hjcrdn/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &   K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &   PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      SAVE   
      DATA IW/0/

      IF (isevt .EQ. IAEVT .AND. isrun .EQ. IARUN) THEN
         DO 1001 I = 1, 50
            DNRG1B(I) = SNRG1B(I)
            dtg1b(I) = stg1b(I)
 1001    CONTINUE
      ELSE
         DO 1002 I = 1, 50
            SNRG1B(I) = DNRG1B(I)
            stg1b(I) = dtg1b(I)
 1002    CONTINUE
         isevt = IAEVT
         isrun = IARUN
         IW = IW + 1
      END IF
c.....analysis
      DO 1003 I = 1, MUL
         J = LSTRG1(I)

         IF (J .LE. NSP) THEN
            K = J
            GX0 = YP(1, J)
            GY0 = YP(2, J)
         ELSE IF (J .LE. NSP + NST) THEN
            K = J - NSP
            GX0 = YT(1, K)
            GY0 = YT(2, K)
         ELSE
            K = J - NSP - NST
            GX0 = 0.5 * (YP(1, IASG(K, 1)) + YT(1, IASG(K, 2)))
            GY0 = 0.5 * (YP(2, IASG(K, 1)) + YT(2, IASG(K, 2)))
         END IF
         R0 = SQRT((sngl(GX5(I)) - GX0)**2 + (sngl(GY5(I)) - GY0)**2)
         IR = 1 + int(R0 / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 100
         DNRG1B(IR) = DNRG1B(IR) + 1.0
 100     CONTINUE
         TAU7 = SQRT(sngl(FT5(I) ** 2 - GZ5(I) ** 2))
         IT = 1 + int(TAU7 / DT)
         IF (IT .GT. 50 .or. IT .LT. 1) GOTO 200
         dtg1b(IT) = dtg1b(IT) + 1.0
 200     CONTINUE
 1003 CONTINUE
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine in HIJING after parton cascade evolution
      SUBROUTINE HJANA2
c
      PARAMETER (YMAX = 1.0, YMIN = -1.0)
      PARAMETER (DMT = 0.05, DY = 0.2)
      PARAMETER (DR = 0.2, DT = 0.2)
      PARAMETER (MAXPTN=400001)
      PARAMETER (MAXSTR=150001)
      DOUBLE PRECISION  PXSGS,PYSGS,PZSGS,PESGS,PMSGS,
     1     GXSGS,GYSGS,GZSGS,FTSGS
      DIMENSION dyp2(50), DMYP2(200), DEYP2(50)
      DIMENSION dyg2(50), DMYG2(200), DEYG2(50)
      DIMENSION SNYP2(50), SMYP2(200), SEYP2(50)
      DIMENSION SNYG2(50), SMYG2(200), SEYG2(50)
      DIMENSION dnrpj2(50), dnrtg2(50), dnrin2(50),
     &   dnrtt2(50)
      DIMENSION dtpj2(50), dttg2(50), dtin2(50),
     &   dttot2(50)
      DIMENSION dyg2c(50), dmyg2c(50), deyg2c(50)
      DIMENSION snrpj2(50), snrtg2(50), snrin2(50),
     &   snrtt2(50)
      DIMENSION stpj2(50), sttg2(50), stin2(50),
     &   sttot2(50)
      DIMENSION snyg2c(50), smyg2c(50), seyg2c(50)
      DOUBLE PRECISION  ATAUI, ZT1, ZT2, ZT3
      DOUBLE PRECISION  GX5, GY5, GZ5, FT5, PX5, PY5, PZ5, E5, XMASS5
      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
      COMMON /SREC2/ATAUI(MAXSTR),ZT1(MAXSTR),ZT2(MAXSTR),ZT3(MAXSTR)
cc      SAVE /SREC2/
      COMMON/HJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &   PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &   PJPM(300,500),NTJ(300),KFTJ(300,500),
     &   PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &   PJTE(300,500),PJTM(300,500)
cc      SAVE /HJJET1/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &   K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &   PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &   PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &   XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
      COMMON/SOFT/PXSGS(MAXSTR,3),PYSGS(MAXSTR,3),PZSGS(MAXSTR,3),
     &     PESGS(MAXSTR,3),PMSGS(MAXSTR,3),GXSGS(MAXSTR,3),
     &     GYSGS(MAXSTR,3),GZSGS(MAXSTR,3),FTSGS(MAXSTR,3),
     &     K1SGS(MAXSTR,3),K2SGS(MAXSTR,3),NJSGS(MAXSTR)
cc      SAVE /SOFT/
      SAVE   
      DATA IW/0/

      IF (isevt .EQ. IAEVT .AND. isrun .EQ. IARUN) THEN
         DO 1001 I = 1, 200
            DMYP2(I) = SMYP2(I)
            DMYG2(I) = SMYG2(I)
 1001    CONTINUE

         DO 1002 I = 1, 50
            dyp2(I) = SNYP2(I)
            DEYP2(I) = SEYP2(I)
            dyg2(I) = SNYG2(I)
            DEYG2(I) = SEYG2(I)
            dnrpj2(I) = snrpj2(I)
            dnrtg2(I) = snrtg2(I)
            dnrin2(I) = snrin2(I)
            dnrtt2(I) = snrtt2(I)
            dtpj2(I) = stpj2(I)
            dttg2(I) = sttg2(I)
            dtin2(I) = stin2(I)
            dttot2(I) = sttot2(I)
            dyg2c(I) = snyg2c(I)
            dmyg2c(I) = smyg2c(I)
            deyg2c(I) = seyg2c(I)
 1002    CONTINUE
         nsubp = nsubpS
         nsubg = nsubgS
         NISG = NISGS
      ELSE
         DO 1003 I = 1, 200
            SMYP2(I) = DMYP2(I)
            SMYG2(I) = DMYG2(I)
 1003    CONTINUE

         DO 1004 I = 1, 50
            SNYP2(I) = dyp2(I)
            SEYP2(I) = DEYP2(I)
            SNYG2(I) = dyg2(I)
            SEYG2(I) = DEYG2(I)
            snrpj2(I) = dnrpj2(I)
            snrtg2(I) = dnrtg2(I)
            snrin2(I) = dnrin2(I)
            snrtt2(I) = dnrtt2(I)
            stpj2(I) = dtpj2(I)
            sttg2(I) = dttg2(I)
            stin2(I) = dtin2(I)
            sttot2(I) = dttot2(I)
            snyg2c(I) = dyg2c(I)
            smyg2c(I) = dmyg2c(I)
            seyg2c(I) = deyg2c(I)
 1004    CONTINUE
         nsubpS = nsubp
         nsubgS = nsubg
         NISGS = NISG
         isevt = IAEVT
         isrun = IARUN
         IW = IW + 1
      END IF

clin-4/28/01:
      if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) goto 510

c.....analysis
      DO 1006 I = 1, IHNT2(1)
         DO 1005 J = 1, NPJ(I)
            ITYP = KFPJ(I, J)
            PX = PJPX(I, J)
            PY = PJPY(I, J)
            PZ = PJPZ(I, J)
            PE = PJPE(I, J)
            PM = PJPM(I, J)
cbzdbg2/16/99
c            IF (ABS(PZ) .GE. PE) GOTO 200
            IF (ABS(PZ) .GE. PE) THEN
               PRINT *, ' IN HJANA2, PROJ STR ', I, ' PART ', J
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', PE
               PRINT *, ' XM = ', PM
               GOTO 200
            END IF
cbzdbg2/16/99end
            RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
            XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
            DXMT = XMT - PM
            IY = 1 + int(ABS(RAP) / DY)
            IF (IY .GT. 50) GOTO 100
            dyp2(IY) = dyp2(IY) + 1.0
            DEYP2(IY) = DEYP2(IY) + XMT
            IF (ITYP .EQ. 21) THEN
               dyg2(IY) = dyg2(IY) + 1.0
               DEYG2(IY) = DEYG2(IY) + XMT
            END IF
 100        CONTINUE
            IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 200
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 200) GOTO 200
            DMYP2(IMT) = DMYP2(IMT) + 1.0 / XMT
            IF (ITYP .EQ. 21) THEN
               DMYG2(IMT) = DMYG2(IMT) + 1.0 / XMT
            END IF
 200        CONTINUE
 1005    CONTINUE
 1006 CONTINUE

      DO 1008 I = 1, IHNT2(3)
         DO 1007 J = 1, NTJ(I)
            ITYP = KFTJ(I, J)
            PX = PJTX(I, J)
            PY = PJTY(I, J)
            PZ = PJTZ(I, J)
            PE = PJTE(I, J)
            PM = PJTM(I, J)
cbzdbg2/16/99
c            IF (ABS(PZ) .GE. PE) GOTO 400
            IF (ABS(PZ) .GE. PE) THEN
               PRINT *, ' IN HJANA2, TARG STR ', I, ' PART ', J
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', PE
               PRINT *, ' XM = ', PM
               GOTO 400
            END IF
cbzdbg2/16/99end
            RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
            XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
            DXMT = XMT - PM
            IY = 1 + int(ABS(RAP) / DY)
            IF (IY .GT. 50) GOTO 300
            dyp2(IY) = dyp2(IY) + 1.0
            DEYP2(IY) = DEYP2(IY) + XMT
            IF (ITYP .EQ. 21) THEN
               dyg2(IY) = dyg2(IY) + 1.0
               DEYG2(IY) = DEYG2(IY) + XMT
            END IF
 300        CONTINUE
            IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 400
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 200) GOTO 400
            DMYP2(IMT) = DMYP2(IMT) + 1.0 / XMT
            IF (ITYP .EQ. 21) THEN
               DMYG2(IMT) = DMYG2(IMT) + 1.0 / XMT
            END IF
 400        CONTINUE
 1007    CONTINUE
 1008 CONTINUE

clin-4/28/01:
 510  continue

      DO 1010 I = 1, NSG
clin-4/25/01 soft3:
c         DO J = 1, NJSG(I)
         NJ=NJSG(I)
         if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) NJ=NJSGS(I)
         DO 1009 J = 1, NJ
clin-4/25/01-end

            ITYP = K2SG(I, J)
            PX = PXSG(I, J)
            PY = PYSG(I, J)
            PZ = PZSG(I, J)
            PE = PESG(I, J)
            PM = PMSG(I, J)
clin-4/25/01 soft3:
            if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) then
               ITYP = K2SGS(I, J)
               PX = sngl(PXSGS(I, J))
               PY = sngl(PYSGS(I, J))
               PZ = sngl(PZSGS(I, J))
               PE = sngl(PESGS(I, J))
               PM = sngl(PMSGS(I, J))
            endif
clin-4/25/01-end

cbzdbg2/16/99
c            IF (ABS(PZ) .GE. PE) GOTO 600
            IF (ABS(PZ) .GE. PE) THEN
               PRINT *, ' IN HJANA2, INDP STR ', I, ' PART ', J
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', PE
               PRINT *, ' XM = ', PM
               GOTO 600
            END IF
cbzdbg2/16/99end
            RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
            XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
            DXMT = XMT - PM
            IY = 1 + int(ABS(RAP) / DY)
            IF (IY .GT. 50) GOTO 500
            dyp2(IY) = dyp2(IY) + 1.0
            DEYP2(IY) = DEYP2(IY) + XMT
            IF (ITYP .EQ. 21) THEN
               dyg2(IY) = dyg2(IY) + 1.0
               DEYG2(IY) = DEYG2(IY) + XMT
            END IF
 500        CONTINUE
            IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 600
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 200) GOTO 600
            DMYP2(IMT) = DMYP2(IMT) + 1.0 / XMT
            IF (ITYP .EQ. 21) THEN
               DMYG2(IMT) = DMYG2(IMT) + 1.0 / XMT
            END IF
 600        CONTINUE
 1009    CONTINUE
 1010 CONTINUE

clin-4/28/01:
      if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) goto 520

      DO 1011 I = 1, IHNT2(1)
         J = I
         YR = SQRT(sngl(ZT1(J) ** 2 + ZT2(J) ** 2))
         IR = 1 + int(YR / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 601
         dnrpj2(IR) = dnrpj2(IR) + 1.0
         dnrtt2(IR) = dnrtt2(IR) + 1.0
 601     CONTINUE
         IT = 1 + int(sngl(ATAUI(J)) / DT)
         IF (IT .GT. 50 .or. IT .LT. 1) GOTO 602
         dtpj2(IT) = dtpj2(IT) + 1.0
         dttot2(IT) = dttot2(IT) + 1.0
 602     CONTINUE
 1011 CONTINUE

      DO 1012 I = 1, IHNT2(3)
         J = I + IHNT2(1)
         YR = SQRT(sngl(ZT1(J) ** 2 + ZT2(J) ** 2))
         IR = 1 + int(YR / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 603
         dnrtg2(IR) = dnrtg2(IR) + 1.0
         dnrtt2(IR) = dnrtt2(IR) + 1.0
 603     CONTINUE
         IT = 1 + int(sngl(ATAUI(J)) / DT)
         IF (IT .GT. 50 .or. IT .LT. 1) GOTO 604
         dttg2(IT) = dttg2(IT) + 1.0
         dttot2(IT) = dttot2(IT) + 1.0
 604     CONTINUE
 1012 CONTINUE

clin-4/28/01:
 520  continue

      DO 1013 I = 1, NSG
         J = I + IHNT2(1) + IHNT2(3)
clin-4/28/01:
         if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) J = I

         YR = SQRT(sngl(ZT1(J) ** 2 + ZT2(J) ** 2))
         IR = 1 + int(YR / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 605
         dnrin2(IR) = dnrin2(IR) + 1.0
         dnrtt2(IR) = dnrtt2(IR) + 1.0
 605     CONTINUE
         IT = 1 + int(sngl(ATAUI(J)) / DT)
         IF (IT .GT. 50 .or. IT .LT. 1) GOTO 606
         dtin2(IT) = dtin2(IT) + 1.0
         dttot2(IT) = dttot2(IT) + 1.0
 606     CONTINUE
 1013 CONTINUE

      DO 1014 I = 1, MUL
         ITYP = ITYP5(I)
         PX = sngl(PX5(I))
         PY = sngl(PY5(I))
         PZ = sngl(PZ5(I))
         PE = sngl(E5(I))
         PM = sngl(XMASS5(I))
cbzdbg2/16/99
c            IF (ABS(PZ) .GE. PE) GOTO 800
         
         IF (ABS(PZ) .GE. PE) THEN
            PRINT *, ' IN HJANA2, GLUON ', I
            PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
            PRINT *, ' PZ = ', PZ, ' EE = ', PE
            PRINT *, ' XM = ', PM
            GOTO 800
         END IF
         
cbzdbg2/16/99end
         RAP = 0.5 * LOG((PE + PZ) / (PE - PZ))
         XMT = SQRT(PX ** 2 + PY ** 2 + PM ** 2)
         DXMT = XMT - PM
         IY = 1 + int(ABS(RAP) / DY)
         IF (IY .GT. 50) GOTO 700
         dyg2c(IY) = dyg2c(IY) + 1.0
         deyg2c(IY) = deyg2c(IY) + XMT
 700     CONTINUE
         IF (RAP .GT. YMAX .OR. RAP .LE. YMIN) GOTO 800
         IMT = 1 + int(DXMT / DMT)
         IF (IMT .GT. 50) GOTO 800
         dmyg2c(IMT) = dmyg2c(IMT) + 1.0 / XMT
 800     CONTINUE
 1014 CONTINUE

clin-4/25/01 soft3:
      if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) goto 530

c.....count number of particles
      DO 1016 I = 1, IHNT2(1)
         DO 1015 J = 1, NPJ(I)
            nsubp = nsubp + 1
            IF (KFPJ(I, J) .EQ. 21) nsubg = nsubg + 1
 1015    CONTINUE
 1016 CONTINUE

      DO 1018 I = 1, IHNT2(3)
         DO 1017 J = 1, NTJ(I)
            nsubp = nsubp + 1
            IF (KFTJ(I, J) .EQ. 21) nsubg = nsubg + 1
 1017    CONTINUE
 1018 CONTINUE

clin-4/25/01 soft3:
 530  continue

      DO 1020 I = 1, NSG
clin-4/25/01 soft3:
c         DO J = 1, NJSG(I)
         NJ=NJSG(I)
         if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) NJ=NJSGS(I)
         DO 1019 J = 1, NJ
clin-4/25/01-end

            nsubp = nsubp + 1

clin-4/25/01
c            IF (K2SG(I, J) .EQ. 21) nsubg = nsubg + 1
            if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) then
               IF(K2SGS(I, J) .EQ. 21) nsubg = nsubg + 1
            else
               IF (K2SG(I, J) .EQ. 21) nsubg = nsubg + 1
            endif
clin-4/25/01-end
 1019    CONTINUE
 1020 CONTINUE
cbzdbg2/16/99
      NISG = NISG + NSG

      IF (IOUT .EQ. 1) THEN
cbzdbg2/16/99end
cbzdbg2/16/99
c      PRINT *, ' in HJANA2 '
c      PRINT *, ' total number of partons = ', nsubp
c      PRINT *, ' total number of gluons = ', nsubg, MUL
c      PRINT *, ' number of projectile strings = ', IHNT2(1)
c      PRINT *, ' number of target strings = ', IHNT2(3)
c      PRINT *, ' number of independent strings = ', NSG
      PRINT *, ' in HJANA2 '
      PRINT *, ' total number of partons = ', nsubp / IW
      PRINT *, ' total number of gluons = ', nsubg / IW
c      PRINT *, ' number of projectile strings = ', IHNT2(1)
c      PRINT *, ' number of target strings = ', IHNT2(3)
      PRINT *, ' number of independent strings = ', NISG / IW
      END IF

      CALL HJAN2A
      CALL HJAN2B

      RETURN
      END

c-----------------------------------------------------------------------

c.....subroutine called by HJANA2
      SUBROUTINE HJAN2A

      PARAMETER (DGX = 0.2, DGY = 0.2, DT = 0.2)
      PARAMETER (MAXPTN=400001,MAXSTR=150001)
      DIMENSION dgxp2a(50), dgyp2a(50), dtp2a(50)
      DIMENSION dgxg2a(50), dgyg2a(50), dtg2a(50)
      DIMENSION sgxp2a(50), sgyp2a(50), stp2a(50)
      DIMENSION sgxg2a(50), sgyg2a(50), stg2a(50)
      DOUBLE PRECISION  GX5, GY5, GZ5, FT5, PX5, PY5, PZ5, E5, XMASS5
      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &   PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &   XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
      COMMON/hjcrdn/YP(3,300),YT(3,300)
cc      SAVE /hjcrdn/
      COMMON/HJJET1/NPJ(300),KFPJ(300,500),PJPX(300,500),
     &   PJPY(300,500),PJPZ(300,500),PJPE(300,500),
     &   PJPM(300,500),NTJ(300),KFTJ(300,500),
     &   PJTX(300,500),PJTY(300,500),PJTZ(300,500),
     &   PJTE(300,500),PJTM(300,500)
cc      SAVE /HJJET1/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &   K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &   PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      SAVE   
      DATA IW/0/

      IF (isevt .EQ. IAEVT .AND. isrun .EQ. IARUN) THEN
         DO 1001 I = 1, 50
            dgxp2a(I) = sgxp2a(I)
            dgyp2a(I) = sgyp2a(I)
            dtp2a(I) = stp2a(I)
            dgxg2a(I) = sgxg2a(I)
            dgyg2a(I) = sgyg2a(I)
            dtg2a(I) = stg2a(I)
 1001    CONTINUE
      ELSE
         DO 1002 I = 1, 50
            sgxp2a(I) = dgxp2a(I)
            sgyp2a(I) = dgyp2a(I)
            stp2a(I) = dtp2a(I)
            sgxg2a(I) = dgxg2a(I)
            sgyg2a(I) = dgyg2a(I)
            stg2a(I) = dtg2a(I)
 1002    CONTINUE
         isevt = IAEVT
         isrun = IARUN
         IW = IW + 1
      END IF
c.....analysis
      DO 1004 I = 1, IHNT2(1)
         DO 1003 J = 1, NPJ(I)
            IF (KFPJ(I, J) .NE. 21) THEN
               IGX = 1 + int(ABS(YP(1, I)) / DGX)
               IF (IGX .GT. 50 .or. IGX .LT. 1) GOTO 100
               dgxp2a(IGX) = dgxp2a(IGX) + 1.0
 100           CONTINUE
               IGY = 1 + int(ABS(YP(2, I)) / DGY)
               IF (IGY .GT. 50 .or. IGY .LT. 1) GOTO 200
               dgyp2a(IGY) = dgyp2a(IGY) + 1.0
 200           CONTINUE
               IT = 1
               dtp2a(IT) = dtp2a(IT) + 1.0
            END IF
 1003    CONTINUE
 1004 CONTINUE

      DO 1006 I = 1, IHNT2(3)
         DO 1005 J = 1, NTJ(I)
            IF (KFTJ(I, J) .NE. 21) THEN
               IGX = 1 + int(ABS(YT(1, I)) / DGX)
               IF (IGX .GT. 50 .or. IGX .LT. 1) GOTO 300
               dgxp2a(IGX) = dgxp2a(IGX) + 1.0
 300           CONTINUE
               IGY = 1 + int(ABS(YT(2, I)) / DGY)
               IF (IGY .GT. 50 .or. IGY .LT. 1) GOTO 400
               dgyp2a(IGY) = dgyp2a(IGY) + 1.0
 400           CONTINUE
               IT = 1
               dtp2a(IT) = dtp2a(IT) + 1.0
            END IF
 1005    CONTINUE
 1006 CONTINUE

      DO 1008 I = 1, NSG
         DO 1007 J = 1, NJSG(I)
            IF (K2SG(I, J) .NE. 21) THEN
               IGX = 1 + int(ABS(0.5 * 
     &            (YP(1, IASG(I, 1)) + YT(1, IASG(I, 2)))) / DGX)
               IF (IGX .GT. 50 .or. IGX .LT. 1) GOTO 500
               dgxp2a(IGX) = dgxp2a(IGX) + 1.0
 500           CONTINUE
               IGY = 1 + int(ABS(0.5 * 
     &            (YP(2, IASG(I, 1)) + YT(2, IASG(I, 2)))) / DGY)
               IF (IGY .GT. 50 .or. IGY .LT. 1) GOTO 600
               dgyp2a(IGY) = dgyp2a(IGY) + 1.0
 600           CONTINUE
               IT = 1
               dtp2a(IT) = dtp2a(IT) + 1.0               
            END IF
 1007    CONTINUE
 1008 CONTINUE

      DO 1009 I = 1, MUL
         IGX = 1 + int(ABS(sngl(GX5(I))) / DGX)
         IF (IGX .GT. 50 .or. IGX .LT. 1) GOTO 700
         dgxg2a(IGX) = dgxg2a(IGX) + 1.0
         dgxp2a(IGX) = dgxp2a(IGX) + 1.0
 700     CONTINUE
         IGY = 1 + int(ABS(sngl(GY5(I))) / DGY)
         IF (IGY .GT. 50 .or. IGY .LT. 1) GOTO 800
         dgyg2a(IGY) = dgyg2a(IGY) + 1.0
         dgyp2a(IGY) = dgyp2a(IGY) + 1.0
 800     CONTINUE
         IT = 1 + int(SQRT(sngl(FT5(I) ** 2 - GZ5(I) ** 2)) / DT)
         IF (IT .GT. 50 .or. IT .LT. 1) GOTO 900
         dtg2a(IT) = dtg2a(IT) + 1.0
         dtp2a(IT) = dtp2a(IT) + 1.0
 900     CONTINUE
 1009 CONTINUE
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine in HJANA2

      SUBROUTINE HJAN2B

      PARAMETER (MAXPTN=400001)
      PARAMETER (MAXSTR=150001)
      PARAMETER (DR = 0.2, DT = 0.2)
      DIMENSION DNRG2B(50), dtg2b(-24:25)
      DIMENSION SNRG2B(50), stg2b(-24:25)
      DOUBLE PRECISION  GX5, GY5, GZ5, FT5, PX5, PY5, PZ5, E5, XMASS5
      DOUBLE PRECISION  ATAUI, ZT1, ZT2, ZT3
      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &   PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &   XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
      COMMON /ilist8/ LSTRG1(MAXPTN), LPART1(MAXPTN)
cc      SAVE /ilist8/
      COMMON /SREC1/ NSP, NST, NSI
cc      SAVE /SREC1/
      COMMON /SREC2/ATAUI(MAXSTR),ZT1(MAXSTR),ZT2(MAXSTR),ZT3(MAXSTR)
cc      SAVE /SREC2/
      COMMON/hjcrdn/YP(3,300),YT(3,300)
cc      SAVE /hjcrdn/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &   K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &   PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
      COMMON /AREVT/ IAEVT, IARUN, MISS
cc      SAVE /AREVT/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      SAVE   
      DATA IW/0/

      IF (isevt .EQ. IAEVT .AND. isrun .EQ. IARUN) THEN
         DO 1001 I = 1, 50
            DNRG2B(I) = SNRG2B(I)
            dtg2b(I - 25) = stg2b(I - 25)
 1001    CONTINUE
      ELSE
         DO 1002 I = 1, 50
            SNRG2B(I) = DNRG2B(I)
            stg2b(I - 25) = dtg2b(I - 25)
 1002    CONTINUE
         isevt = IAEVT
         isrun = IARUN
         IW = IW + 1
      END IF
c.....analysis
      DO 1003 I = 1, MUL
         J = LSTRG1(I)
         GX0 = sngl(ZT1(J))
         GY0 = sngl(ZT2(J))
         R0 = SQRT((sngl(GX5(I)) - GX0)**2 + (sngl(GY5(I)) - GY0)**2)
         IR = 1 + int(R0 / DR)
         IF (IR .GT. 50 .or. IR .LT. 1) GOTO 100
         DNRG2B(IR) = DNRG2B(IR) + 1.0
 100     CONTINUE
         TAU7 = SQRT(sngl(FT5(I) ** 2 - GZ5(I) ** 2))
         DTAU=TAU7 - sngl(ATAUI(J))
         IT = 1 + int(DTAU / DT)
cbzdbg2/21/99
c         IF (ABS(IT) .GT. 25) GOTO 200
         IF (IT .GT. 25 .OR. IT .LT. -24) GOTO 200
cbzdbg2/21/99end
         dtg2b(IT) = dtg2b(IT) + 1.0
 200     CONTINUE
 1003 CONTINUE
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine before ARTMN
      SUBROUTINE HJANA3
c
      PARAMETER (MAXSTR=150001, MAXR=1)
c.....y cut for mt spectrum
      PARAMETER (YMIN = -1.0, YMAX = 1.0)
cbz11/7/99 end
c.....bin width for mt spectrum and y spectrum
      PARAMETER (DMT = 0.05, DY = 0.2)
      DOUBLE PRECISION v2i,eti,xmulti,v2mi,s2mi,xmmult,
     1     v2bi,s2bi,xbmult
      DIMENSION dndyh3(50), DMYH3(50), DEYH3(50)
      COMMON /RUN/ NUM
cc      SAVE /RUN/
      COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cc      SAVE /ARPRC1/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      COMMON/iflow/v2i,eti,xmulti,v2mi,s2mi,xmmult,v2bi,s2bi,xbmult
cc      SAVE /iflow/
      SAVE   
      DATA IW/0/

      IW = IW + 1
      DO 1002 J = 1, NUM
         DO 1001 I = 1, MULTI1(J)
            ITYP = ITYP1(I, J)
            IF (ITYP .GT. -100 .AND. ITYP .LT. 100) GOTO 200
            PX = PX1(I, J)
            PY = PY1(I, J)
            PZ = PZ1(I, J)
            EE = EE1(I, J)
            XM = XM1(I, J)
            XMT = SQRT(PX ** 2 + PY ** 2 + XM ** 2)
            IF (ABS(PZ) .GE. EE) THEN
               PRINT *, 'IN HJANA3'
               PRINT *, ' PARTICLE ', I, ' RUN ', J, 'PREC ERR'
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', EE
               PRINT *, ' XM = ', XM
               GOTO 200
            END IF
            DXMT = XMT - XM
            Y = 0.5 * LOG((EE + PZ) / (EE - PZ))
c.....rapidity cut for the rapidity distribution
c            IY = 1 + int(ABS(Y) / DY)
            IY = 1 + int((Y+10.) / DY)
            IF (IY .GT. 50) GOTO 100
            dndyh3(IY) = dndyh3(IY) + 1.0
            DEYH3(IY) = DEYH3(IY) + XMT
 100        CONTINUE
c.....insert rapidity cut for mt spectrum here
            IF (Y. LT. YMIN .OR. Y .GE. YMAX) GOTO 200
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 50) GOTO 200
            DMYH3(IMT) = DMYH3(IMT) + 1.0 / XMT
 200        CONTINUE
 1001    CONTINUE
 1002 CONTINUE
c
      RETURN
      END

c-----------------------------------------------------------------------

c.....analysis subroutine after ARTMN
      SUBROUTINE HJANA4
      PARAMETER (MAXSTR=150001, MAXR=1)
c.....y cut for mt spectrum
cbz11/7/99
c      PARAMETER (YMIN = -0.5, YMAX = 0.5)
      PARAMETER (YMIN = -1.0, YMAX = 1.0)
cbz11/7/99 end
c.....bin width for mt spectrum and y spectrum
      PARAMETER (DMT = 0.05, DY = 0.2)

      DIMENSION dndyh4(50), DMYH4(50), DEYH4(50)
      COMMON /RUN/ NUM
cc      SAVE /RUN/
      COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cc      SAVE /ARPRC1/
      COMMON /AROUT/ IOUT
cc      SAVE /AROUT/
      COMMON /fflow/ v2f,etf,xmultf,v2fpi,xmulpi
cc      SAVE /fflow/
      SAVE   
      DATA IW/0/

      IW = IW + 1
      DO 1002 J = 1, NUM
         DO 1001 I = 1, MULTI1(J)
            ITYP = ITYP1(I, J)
            IF (ITYP .GT. -100 .AND. ITYP .LT. 100) GOTO 200
            PX = PX1(I, J)
            PY = PY1(I, J)
            PZ = PZ1(I, J)
            EE = EE1(I, J)
            XM = XM1(I, J)
            XMT = SQRT(PX ** 2 + PY ** 2 + XM ** 2)
            IF (ABS(PZ) .GE. EE) THEN
               PRINT *, 'IN HJANA4'
               PRINT *, ' PARTICLE ', I, ' RUN ', J, 'PREC ERR'
               PRINT *, ' FLAV = ', ITYP, ' PX = ', PX, ' PY = ', PY
               PRINT *, ' PZ = ', PZ, ' EE = ', EE
               PRINT *, ' XM = ', XM
               GOTO 200
            END IF
            DXMT = XMT - XM
            Y = 0.5 * LOG((EE + PZ) / (EE - PZ))
c.....rapidity cut for the rapidity distribution
c            IY = 1 + int(ABS(Y) / DY)
            IY = 1 + int((Y+10.) / DY)
            IF (IY .GT. 50) GOTO 100
            dndyh4(IY) = dndyh4(IY) + 1.0
            DEYH4(IY) = DEYH4(IY) + XMT
 100        CONTINUE
c.....insert rapidity cut for mt spectrum here
            IF (Y. LT. YMIN .OR. Y .GE. YMAX) GOTO 200
            IMT = 1 + int(DXMT / DMT)
            IF (IMT .GT. 50) GOTO 200
            DMYH4(IMT) = DMYH4(IMT) + 1.0 / XMT
 200        CONTINUE
 1001    CONTINUE
 1002 CONTINUE
c
      RETURN
      END

c=======================================================================

c.....subroutine to get average values for different strings

      SUBROUTINE zpstrg

      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      PARAMETER (MAXPTN=400001)
      PARAMETER (MAXSTR=150001)
c      REAL*4 YP, YT, PXSG, PYSG, PZSG, PESG, PMSG, HIPR1, HINT1, BB
      REAL YP, YT, PXSG, PYSG, PZSG, PESG, PMSG, HIPR1, HINT1, BB

      COMMON /PARA1/ MUL
cc      SAVE /PARA1/
      COMMON /prec2/GX5(MAXPTN),GY5(MAXPTN),GZ5(MAXPTN),FT5(MAXPTN),
     &   PX5(MAXPTN), PY5(MAXPTN), PZ5(MAXPTN), E5(MAXPTN),
     &   XMASS5(MAXPTN), ITYP5(MAXPTN)
cc      SAVE /prec2/
      COMMON /ilist8/ LSTRG1(MAXPTN), LPART1(MAXPTN)
cc      SAVE /ilist8/
      COMMON /SREC1/ NSP, NST, NSI
cc      SAVE /SREC1/
      COMMON /SREC2/ATAUI(MAXSTR),ZT1(MAXSTR),ZT2(MAXSTR),ZT3(MAXSTR)
cc      SAVE /SREC2/
      COMMON/hjcrdn/YP(3,300),YT(3,300)
cc      SAVE /hjcrdn/
      COMMON/HJJET2/NSG,NJSG(MAXSTR),IASG(MAXSTR,3),K1SG(MAXSTR,100),
     &   K2SG(MAXSTR,100),PXSG(MAXSTR,100),PYSG(MAXSTR,100),
     &   PZSG(MAXSTR,100),PESG(MAXSTR,100),PMSG(MAXSTR,100)
cc      SAVE /HJJET2/
cbz6/28/99 flow1
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
cc      SAVE /HPARNT/
cbz6/28/99 flow1 end
      common/anim/nevent,isoft,isflag,izpc
cc      SAVE /anim/
      common/strg/np(maxstr)
cc      SAVE /strg/
clin-6/06/02 test local freezeout:
      common /frzprc/ 
     &     gxfrz(MAXPTN), gyfrz(MAXPTN), gzfrz(MAXPTN), ftfrz(MAXPTN),
     &     pxfrz(MAXPTN), pyfrz(MAXPTN), pzfrz(MAXPTN), efrz(MAXPTN),
     &     xmfrz(MAXPTN), 
     &     tfrz(302), ifrz(MAXPTN), idfrz(MAXPTN), itlast
cc      SAVE /frzprc/
      SAVE   

clin-6/06/02 test local freezeout for string melting,
c     use space-time values at local freezeout saved in /frzprc/:
      if(isoft.eq.5) then
         do 1001 I = 1, MUL
            ITYP5(i)=idfrz(i)
            GX5(i)=gxfrz(i)
            GY5(i)=gyfrz(i)
            GZ5(i)=gzfrz(i)
            FT5(i)=ftfrz(i)
            PX5(i)=pxfrz(i)
            PY5(i)=pyfrz(i)
            PZ5(i)=pzfrz(i)
            E5(i)=efrz(i)
            XMASS5(i)=xmfrz(i)
 1001    continue
      endif
clin-6/06/02-end

      DO 1002 I = 1, MAXSTR
         ATAUI(I) = 0d0
         ZT1(I) = 0d0
         ZT2(I) = 0d0
clin-4/25/03 add zt3(I) to track longitudinal positions of partons/strings:
         ZT3(I) = 0d0
         NP(I) = 0
 1002 CONTINUE
      DO 1003 I = 1, MUL
         ISTRG = LSTRG1(I)
         TAU7 = SQRT(FT5(I) ** 2 - GZ5(I) ** 2)
         ATAUI(ISTRG) = ATAUI(ISTRG) + TAU7
         ZT1(ISTRG) = ZT1(ISTRG) + GX5(I)
         ZT2(ISTRG) = ZT2(ISTRG) + GY5(I)
         ZT3(ISTRG) = ZT3(ISTRG) + GZ5(I)
         NP(ISTRG) = NP(ISTRG) + 1
 1003 CONTINUE
      
      NSTR = NSP + NST + NSI

clin-7/03/01 correct averaging on transverse coordinates, no shift needed:
      if(isoft.eq.3.or.isoft.eq.4.or.isoft.eq.5) then
         DO 1004 I = 1, NSTR
            IF (NP(I) .NE. 0) THEN
               ATAUI(I) = ATAUI(I) / NP(I)
               ZT1(I) = ZT1(I) / NP(I)
               ZT2(I) = ZT2(I) / NP(I)
               ZT3(I) = ZT3(I) / NP(I)
            ENDIF
 1004    CONTINUE
         return
      endif
clin-7/03/01-end

      DO 1005 I = 1, NSTR
         IF (NP(I) .NE. 0) THEN
            ATAUI(I) = ATAUI(I) / NP(I)
            ZT1(I) = ZT1(I) / NP(I)
            ZT2(I) = ZT2(I) / NP(I)
            ZT3(I) = ZT3(I) / NP(I)
         ELSE
            IF (I .LE. NSP) THEN
               J = I
               ZT1(I) = dble(YP(1, J))
               ZT2(I) = dble(YP(2, J))
               ZT3(I) = 0d0
            ELSE IF (I .GT. NSP .AND. I .LE. NSP + NST) THEN
               J = I - NSP
               ZT1(I) = dble(YT(1, J))
               ZT2(I) = dble(YT(2, J))
               ZT3(I) = 0d0
            ELSE
               J = I - NSP - NST
               ZT1(I) = 0.5d0*
     1              dble((YP(1, IASG(J, 1)) + YT(1, IASG(J, 2))))
               ZT2(I) = 0.5d0 *
     1              dble((YP(2, IASG(J, 1)) + YT(2, IASG(J, 2))))
               ZT3(I) = 0d0
            END IF
         END IF
 1005 CONTINUE

cbz6/28/99 flow1
      BB = HINT1(19)
      DO 1006 I = 1, NSTR
         IF (NP(I).NE.0) THEN
            SHIFT=0d0
         ELSE
            SHIFT=0.5d0*dble(BB)
         END IF
         IF (I .LE. NSP) THEN
            ZT1(I) = ZT1(I) + SHIFT
         ELSE IF (I .GT. NSP .AND. I .LE. NSP + NST) THEN
            ZT1(I) = ZT1(I) - SHIFT
         END IF
 1006 CONTINUE
cbz6/28/99 flow1 end
c
      RETURN
      END

clin-3/2009
c     Initialize hadron weights; 
c     Can add initial hadrons before the hadron cascade starts (but after ZPC).
      subroutine addhad
      PARAMETER (MAXSTR=150001,MAXR=1,xmd=1.8756)
      double precision  smearp,smearh
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
      COMMON /ARPRC/ ITYPAR(MAXSTR),
     &     GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
     &     PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
     &     XMAR(MAXSTR)
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      COMMON /smearz/smearp,smearh
      COMMON/RNDF77/NSEED
      common /para8/ idpert,npertd,idxsec
      SAVE   
c     All hadrons at the start of hadron cascade have the weight of 1
c     except those inserted by the user in this subroutine:
      np0=IAINT2(1)
      DO i=1,np0
         dpertp(I)=1.
      ENDDO
c     Specify number, species, weight, initial x,p,m for inserted hadrons here:
      nadd=0
      tau0=ARPAR1(1)
      DO 100 i=np0+1,np0+nadd
         ITYPAR(I)=42
         dpertp(I)=1d0/dble(nadd)
         GXAR(I)=5.*(1.-2.*RANART(NSEED))
         GYAR(I)=5.*(1.-2.*RANART(NSEED))
         GZAR(I)=2.*(1.-2.*RANART(NSEED))
         FTAR(I)=0.
         PXAR(I)=1.
         PYAR(I)=0.
         PZAR(I)=1.
         XMAR(I)=xmd
c
         PEAR(I)=sqrt(PXAR(I)**2+PYAR(I)**2+PZAR(I)**2+XMAR(I)**2)
         RAP=0.5*alog((PEAR(I)+PZAR(I))/(PEAR(I)-PZAR(I)))
         VX=PXAR(I)/PEAR(I)
         VY=PYAR(I)/PEAR(I)
c.....give initial formation time shift and boost according to rapidity:
         TAUI=FTAR(I)+TAU0
         FTAR(I)=TAUI*COSH(RAP)
         GXAR(I)=GXAR(I)+VX*TAU0*COSH(RAP)
         GYAR(I)=GYAR(I)+VY*TAU0*COSH(RAP)
c     Allow the intial z-position to be different from the Bjorken picture:
         GZAR(I)=TAUI*SINH(RAP)+GZAR(I)
c         GZAR(I)=TAUI*SINH(RAP)
         zsmear=sngl(smearh)*(2.*RANART(NSEED)-1.)
         GZAR(I)=GZAR(I)+zsmear
 100  CONTINUE
      IAINT2(1)=IAINT2(1)+nadd
c
      if(nadd.ge.1.and.idpert.ne.1.and.idpert.ne.2) then
         write(16,*) 'IDPERT must be 1 or 2 to add initial hadrons,
     1 set NPERTD to 0 if you do not need perturbative deuterons'
         stop
      endif
      if(IAINT2(1).gt.MAXSTR) then
         write(16,*) 'Too many initial hadrons, array size is exceeded!'
         stop
      endif
c
      return
      end
