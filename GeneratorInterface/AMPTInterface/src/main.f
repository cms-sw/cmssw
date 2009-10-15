c.....driver program for A Multi-Phase Transport model
      SUBROUTINE AMPT(FRAME0,BMIN,BMAX)
c
      double precision xmp, xmu, alpha, rscut2, cutof2
      double precision smearp,smearh,dpcoal,drcoal,ecritl
cgsfs added following line to match C++ call
      double precision BMIN, BMAX
      integer K
c     CHARACTER*(*) FRAME0
c     CHARACTER FRAME0*8
      CHARACTER*(*) FRAME0
      CHARACTER FRAME*8
cgsfs  added to match specification in AMPTSET
      character*25 amptvn


      COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
      COMMON /HPARNT/HIPR1(100), IHPR2(50), HINT1(100), IHNT2(50)
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
      COMMON /AROUT/ IOUT
      COMMON /AREVT/ IAEVT, IARUN, MISS
      COMMON /smearz/smearp,smearh
      COMMON/RNDF77/NSEED
      common/anim/nevent,isoft,isflag,izpc
c     parton coalescence radii in case of string melting:
      common /coal/dpcoal,drcoal,ecritl
      common/snn/efrm,npart1,npart2
c     initialization value for parton cascade:
      common /para2/ xmp, xmu, alpha, rscut2, cutof2
      common /para7/ ioscar,nsmbbbar,nsmmeson
      common /para8/ idpert,npertd,idxsec
      common /rndm3/ iseedp
c     initialization value for hadron cascade:
      COMMON /RUN/ NUM
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
      COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &   IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
      common/oscar1/iap,izp,iat,izt
      common/oscar2/FRAME,amptvn
      common/resdcy/NSAV,iksdcy
clin-6/2009:
c      common/phidcy/iphidcy
      common/phidcy/iphidcy,pttrig,ntrig,maxmiss
      common/embed/iembed,pxqembd,pyqembd,xembd,yembd

      EXTERNAL HIDATA, PYDATA, LUDATA, ARDATA, PPBDAT, zpcbdt
      SAVE   
c****************

      FRAME=FRAME0
      imiss=0
cgsfs This line should not be here, but the value needs to be set for ARINI2
cgsfs      K=K+1
      K=1

 100  CALL HIJING(FRAME, BMIN, BMAX)
      IAINT2(1) = NATT             


c     evaluate Npart (from primary NN collisions) for both proj and targ:
      call getnp
c     switch for final parton fragmentation:
      IF (IHPR2(20) .EQ. 0) GOTO 2000
c     In the unlikely case of no interaction (even after loop of 20 in HIJING),
c     still repeat the event to get an interaction 
c     (this may have an additional "trigger" effect):
      if(NATT.eq.0) then
         imiss=imiss+1
         if(imiss.le.20) then
            write(6,*) 'repeated event: natt=0,j,imiss=',j,imiss
            goto 100
         else
            write(6,*) 'missed event: natt=0,j=',j
            goto 2000
         endif
      endif
c.....ART initialization and run
      CALL ARINI
      CALL ARINI2(K)
      CALL ARTAN1
      CALL HJANA3
      CALL ARTMN
      CALL HJANA4
      CALL ARTAN2

 2000 CONTINUE
c
c       CALL ARTOUT(NEVNT)
clin-5/2009 ctest off:
c       call flowh0(NEVNT,2)
c       call flowp(2)
c       call iniflw(NEVNT,2)
c       call frztm(NEVNT,2)
c
      RETURN
      END

      SUBROUTINE AMPTSET(EFRM0,FRAME0,PROJ0,TARG0,IAP0,IZP0,IAT0,IZT0)
c
cgsfs added following line to match C++ call
      double precision EFRM0
      double precision xmp, xmu, alpha, rscut2, cutof2
      double precision smearp,smearh,dpcoal,drcoal,ecritl
c     CHARACTER*(*) FRAME0,PROJ0,TARG0
c      CHARACTER FRAME0*8, PROJ0*8, TARG0*8
      CHARACTER*(*) FRAME0,PROJ0,TARG0
      CHARACTER FRAME*8,PROJ*8,TARG*8
      character*25 amptvn
      COMMON/HMAIN1/EATT,JATT,NATT,NT,NP,N0,N01,N10,N11
      COMMON /HPARNT/HIPR1(100), IHPR2(50), HINT1(100), IHNT2(50)
      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
      COMMON /AROUT/ IOUT
      COMMON /AREVT/ IAEVT, IARUN, MISS
      COMMON /smearz/smearp,smearh
      COMMON/RNDF77/NSEED
      common/anim/nevent,isoft,isflag,izpc
c     parton coalescence radii in case of string melting:
      common /coal/dpcoal,drcoal,ecritl
      common/snn/efrm,npart1,npart2
c     initialization value for parton cascade:
      common /para2/ xmp, xmu, alpha, rscut2, cutof2
      common /para7/ ioscar,nsmbbbar,nsmmeson
      common /para8/ idpert,npertd,idxsec
      common /rndm3/ iseedp
c     initialization value for hadron cascade:
      COMMON /RUN/ NUM
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
      COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &   IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
      common/oscar1/iap,izp,iat,izt
      common/oscar2/FRAME,amptvn
      common/resdcy/NSAV,iksdcy
clin-6/2009:
c      common/phidcy/iphidcy
      common/phidcy/iphidcy,pttrig,ntrig,maxmiss
      common/embed/iembed,pxqembd,pyqembd,xembd,yembd

      EXTERNAL HIDATA, PYDATA, LUDATA, ARDATA, PPBDAT, zpcbdt
      SAVE   
c****************
      EFRM=EFRM0
      FRAME=FRAME0
      PROJ=PROJ0
      TARG=TARG0
      IAP=IAP0
      IZP=IZP0
      IAT=IAT0
      IZT=IZT0

c     flag to select default AMPT or string melting:
      isoft=1
c     read initialization value for hadron cascade:
      NTMAX=150
      DT=0.2
c     parj(41) and (42) are a and b parameters in Lund string fragmentation:
      PARJ(41)=2.2
      PARJ(42)=0.5
c     IHPR2(11)=3 (or 2) allows the popcorn mechanism in PYTHIA and 
c     increase the net-baryon stopping in rapidity (value HIJING is 1):
      ipop=1
      if(ipop.eq.1) IHPR2(11)=3
c     PARJ(5) controls the fraction of BMBbar vs BBbar in popcorn:
      PARJ(5)=1.0
c     shadowing flag in HIJING:
      IHPR2(6)=1
c     quenching flag in HIJING:
      IHPR2(4)=0
c     quenching rate when quenching flag is on (=1.0 GeV/fm):
      HIPR1(14)=2.0
c     Minimum pt of hard or semihard scatterings in HIJING: D=2.0 GeV. 
      HIPR1(8)=2.0
c     read initialization value for parton cascade:
      xmu=3.2264d0
      izpc=0
      alpha=0.47140452d0
c     quark coalescence radii in momentum and space for string melting:
      dpcoal=1d6
      drcoal=1d6
c     flag: read in HIJING random # seed at runtime(1) or from input.ampt(D=0):
      ihjsed=0
c     2 seeds for random number generators in HIJING/hadron cascade and ZPC:
      nseed=53153511
      iseedp=8
      iksdcy=0
      iphidcy=1
c     flag for OSCAR output for final partons and hadrons:
      ioscar=0
clin-5/2008     flag for perturbative treatment of deuterons:
      idpert=0
      npertd=1
      idxsec=1
clin-6/2009 To select events that have at least 1 high-Pt minijet parton:
      pttrig=-7.0
      maxmiss=1000
      IHPR2(2)=3
      IHPR2(5)=1
clin-6/2009 To embed a back-to-back q/qbar pair into each event:
      iembed=0
      pxqembd=7.0 
      pyqembd=0.0
      xembd=0.0 
      yembd=0.0

clin-6/2009 ctest off turn on jet triggering:
c      IHPR2(3)=1
c     Trigger Pt of high-pt jets in HIJING:
c      HIPR1(10)=7.
c
      if(isoft.eq.1) then
         amptvn = '1.25 (Default)'
      elseif(isoft.eq.4) then
         amptvn = '2.25 (StringMelting)'
      else
         amptvn = 'Test-Only'
      endif
cgsfs Changed to write(* from write(6
      WRITE(*,50) amptvn
cgsfs      WRITE(12,50) amptvn
 50   FORMAT(' '/
     &11X,'##################################################'/1X,
     &10X,'#      AMPT (A Multi-Phase Transport) model      #'/1X,
     &10X,'#               Version ',a20,             '     #'/1X,
     &10X,'#                06/25/2009                      #'/1X,
     &10X,'##################################################'/1X,
     &10X,' ')
c     when ihjsed=11: use environment variable at run time for HIJING nseed:
      if(ihjsed.eq.11) then
         PRINT *,
     1 '# Read in NSEED in HIJING at run time (e.g. 20030819):'
      endif
c      READ (*, *) nseedr
cgsfs changed from 11 to 12345
      nseedr=12345
      if(ihjsed.eq.11) then
         nseed=nseedr
      endif
c     an odd number is needed for the random number generator:
      if(mod(NSEED,2).eq.0) NSEED=NSEED+1
cgsfs Added following if section back in
      if(ihjsed.eq.11) then      
         PRINT *, '#   read in: ', nseed
c         WRITE(12,*) '# Read in NSEED in HIJING at run time:',nseed
      endif

c     9/26/03 random number generator for f77 compiler:
      CALL SRAND(NSEED)
c
c.....turn on warning messages in nohup.out when an event is repeated:
      IHPR2(10) = 1
c     string formation time:
      ARPAR1(1) = 0.7
c     smearp is the smearing halfwidth on parton z0, 
c     set to 0 for now to avoid overflow in eta.
c     smearh is the smearing halfwidth on string production point z0.
      smearp=0d0
      IAmax=max(iap,iat)
      smearh=1.2d0*IAmax**0.3333d0/(dble(EFRM)/2/0.938d0)
cgsfs  Comment out these debug output files
c     AMPT momentum and space info at freezeout:
c      OPEN (16, FILE = 'ana/ampt.dat', STATUS = 'UNKNOWN')
c      OPEN (14, FILE = 'ana/zpc.dat', STATUS = 'UNKNOWN')
ctest off for resonance (phi, K*) studies:
c      OPEN (17, FILE = 'ana/res-gain.dat', STATUS = 'UNKNOWN')
c      OPEN (18, FILE = 'ana/res-loss.dat', STATUS = 'UNKNOWN')
cgsfs Restored this call which was missing
      CALL HIJSET(EFRM, FRAME, PROJ, TARG, IAP, IZP, IAT, IZT)
      CALL ARTSET
      CALL INIZPC

      RETURN
      END
