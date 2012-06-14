c....................art1f.f
**************************************
*
*                           PROGRAM ART1.0 
*
*        A relativistic transport (ART) model for heavy-ion collisions
*
*   sp/01/04/2002
*   calculates K+K- from phi decay, dimuons from phi decay
*   has finite baryon density & possibilites of varying Kaon 
*   in-medium mass in phiproduction-annhilation channel only.
*
*
* RELEASING DATE: JAN., 1997 
***************************************
* 
* Bao-An Li & Che Ming Ko
* Cyclotron Institute, Texas A&M University.
* Phone: (409) 845-1411
* e-mail: Bali@comp.tamu.edu & Ko@comp.tamu.edu 
* http://wwwcyc.tamu.edu/~bali
***************************************
* Speical notice on the limitation of the code:
* 
* (1) ART is a hadronic transport model
* 
* (2) E_beam/A <= 15 GeV
* 
* (3) The mass of the colliding system is limited by the dimensions of arrays
*    which can be extended purposely. Presently the dimensions are large enough
*     for running Au+Au at 15 GeV/A.
*
* (4) The production and absorption of antiparticles (e.g., ki-, anti-nucleons,
*     etc) are not fully included in this version of the model. They, however, 
*     have essentially no effect on the reaction dynamics and observables 
*     related to nucleons, pions and kaons (K+) at and below AGS energies.
* 
* (5) Bose enhancement for mesons and Pauli blocking for fermions are 
*     turned off.
* 
*********************************
*
* USEFUL REFERENCES ON PHYSICS AND NUMERICS OF NUCLEAR TRANSPORT MODELS:
*     G.F. BERTSCH AND DAS GUPTA, PHYS. REP. 160 (1988) 189.
*     B.A. LI AND W. BAUER, PHYS. REV. C44 (1991) 450.
*     B.A. LI, W. BAUER AND G.F. BERTSCH, PHYS. REV. C44 (1991) 2095.
*     P. DANIELEWICZ AND G.F. BERTSCH, NUCL. PHYS. A533 (1991) 712.
* 
* MAIN REFERENCES ON THIS VERSION OF ART MODEL:
*     B.A. LI AND C.M. KO, PHYS. REV. C52 (1995) 2037; 
*                          NUCL. PHYS. A601 (1996) 457. 
*
**********************************
**********************************
*  VARIABLES IN INPUT-SECTION:                                               * 
*                                                                      *
*  1) TARGET-RELATED QUANTITIES                                        *
*       MASSTA, ZTA -  TARGET MASS IN AMU, TARGET CHARGE  (INTEGER)    *
*                                                                      *
*  2) PROJECTILE-RELATED QUANTITIES                                    *
*       MASSPR, ZPR -  PROJECTILE MASS IN AMU, PROJ. CHARGE(INTEGER)   *
*       ELAB     -  BEAM ENERGY IN [MEV/NUCLEON]               (REAL)  *
*       ZEROPT   -  DISPLACEMENT OF THE SYSTEM IN Z-DIREC. [FM](REAL)  *
*       B        -  IMPACT PARAMETER [FM]                      (REAL)  *
*                                                                      *
*  3) PROGRAM-CONTROL PARAMETERS                                       *
*       ISEED    -  SEED FOR RANDOM NUMBER GENERATOR        (INTEGER)  *
*       DT       -  TIME-STEP-SIZE [FM/C]                      (REAL)  *
*       NTMAX    -  TOTAL NUMBER OF TIMESTEPS               (INTEGER)  *
*       ICOLL    -  (= 1 -> MEAN FIELD ONLY,                           *
*                -   =-1 -> CACADE ONLY, ELSE FULL ART)     (INTEGER)  *
*       NUM      -  NUMBER OF TESTPARTICLES PER NUCLEON     (INTEGER)  *
*       INSYS    -  (=0 -> LAB-SYSTEM, ELSE C.M. SYSTEM)    (INTEGER)  *
*       IPOT     -  1 -> SIGMA=2; 2 -> SIGMA=4/3; 3 -> SIGMA=7/6       *
*                   IN MEAN FIELD POTENTIAL                 (INTEGER)  *
*       MODE     -  (=1 -> interpolation for pauli-blocking,           *
*                    =2 -> local lookup, other -> unblocked)(integer)  *
*       DX,DY,DZ -  widths of cell for paulat in coor. sp. [fm](real)  *
*       DPX,DPY,DPZ-widths of cell for paulat in mom. sp.[GeV/c](real) *
*       IAVOID   -  (=1 -> AVOID FIRST COLL. WITHIN SAME NUCL.         *
*                    =0 -> ALLOW THEM)                      (INTEGER)  *
*       IMOMEN   -  FLAG FOR CHOICE OF INITIAL MOMENTUM DISTRIBUTION   *
*                   (=1 -> WOODS-SAXON DENSITY AND LOCAL THOMAS-FERMI  *
*                    =2 -> NUCLEAR MATTER DEN. AND LOCAL THOMAS-FERMI  *
*                    =3 -> COHERENT BOOST IN Z-DIRECTION)   (INTEGER)  *
*  4) CONTROL-PRINTOUT OPTIONS                                         *
*       NFREQ    -  NUMBER OF TIMSTEPS AFTER WHICH PRINTOUT            *
*                   IS REQUIRED OR ON-LINE ANALYSIS IS PERFORMED       *
*       ICFLOW      =1 PERFORM ON-LINE FLOW ANALYSIS EVERY NFREQ STEPS *
*       ICRHO       =1 PRINT OUT THE BARYON,PION AND ENERGY MATRIX IN  *
*                      THE REACTION PLANE EVERY NFREQ TIME-STEPS       *
*  5)
*       CYCBOX   -  ne.0 => cyclic boundary conditions;boxsize CYCBOX  *
*
**********************************
*               Lables of particles used in this code                     *
**********************************
*         
*         LB(I) IS USED TO LABEL PARTICLE'S CHARGE STATE
*    
*         LB(I)   =
clin-11/07/00:
*                -30 K*-
clin-8/29/00
*                -13 anti-N*(+1)(1535),s_11
*                -12 anti-N*0(1535),s_11
*                 -11 anti-N*(+1)(1440),p_11
*                 -10 anti-N*0(1440), p_11
*                  -9 anti-DELTA+2
*                  -8 anti-DELTA+1
*                  -7 anti-DELTA0
*                  -6 anti-DELTA-1
clin-8/29/00-end

cbali2/7/99 
*                  -2 antineutron 
*                             -1       antiproton
cbali2/7/99 end 
*                   0 eta
*                        1 PROTON
*                   2 NUETRON
*                   3 PION-
*                   4 PION0
*                   5 PION+
*                   6 DELTA-1
*                   7 DELTA0
*                   8 DELTA+1
*                   9 DELTA+2
*                   10 N*0(1440), p_11
*                   11 N*(+1)(1440),p_11
*                  12 N*0(1535),s_11
*                  13 N*(+1)(1535),s_11
*                  14 LAMBDA
*                   15 sigma-, since we used isospin averaged xsection for
*                   16 sigma0  sigma associated K+ production, sigma0 and 
*                   17 sigma+  sigma+ are counted as sigma-
*                   21 kaon-
*                   23 KAON+
*                   24 kaon0
*                   25 rho-
*                         26 rho0
*                   27 rho+
*                   28 omega meson
*                   29 phi
clin-11/07/00:
*                  30 K*+
* sp01/03/01
*                 -14 LAMBDA(bar)
*                  -15 sigma-(bar)
*                  -16 sigma0(bar)
*                  -17 sigma+(bar)
*                   31 eta-prime
*                   40 cascade-
*                  -40 cascade-(bar)
*                   41 cascade0
*                  -41 cascade0(bar)
*                   45 Omega baryon
*                  -45 Omega baryon(bar)
* sp01/03/01 end
clin-5/2008:
*                   42 Deuteron (same in ampt.dat)
*                  -42 anti-Deuteron (same in ampt.dat)
c
*                   ++  ------- SEE BAO-AN LI'S NOTE BOOK
**********************************
cbz11/16/98
c      PROGRAM ART
       SUBROUTINE ARTMN
cbz11/16/98end
**********************************
* PARAMETERS:                                                           *
*  MAXPAR     - MAXIMUM NUMBER OF PARTICLES      PROGRAM CAN HANDLE     *
*  MAXP       - MAXIMUM NUMBER OF CREATED MESONS PROGRAM CAN HANDLE     *
*  MAXR       - MAXIMUM NUMBER OF EVENTS AT EACH IMPACT PARAMETER       *
*  MAXX       - NUMBER OF MESHPOINTS IN X AND Y DIRECTION = 2 MAXX + 1  *
*  MAXZ       - NUMBER OF MESHPOINTS IN Z DIRECTION       = 2 MAXZ + 1  *
*  AMU        - 1 ATOMIC MASS UNIT "GEV/C**2"                           *
*  MX,MY,MZ   - MESH SIZES IN COORDINATE SPACE [FM] FOR PAULI LATTICE   *
*  MPX,MPY,MPZ- MESH SIZES IN MOMENTUM SPACE [GEV/C] FOR PAULI LATTICE  *
*---------------------------------------------------------------------- *
clin      PARAMETER     (maxpar=200000,MAXR=50,AMU= 0.9383,
      PARAMETER     (MAXSTR=150001,MAXR=1,AMU= 0.9383,
     1               AKA=0.498,etaM=0.5475)
      PARAMETER     (MAXX   =   20,  MAXZ  =    24)
      PARAMETER     (ISUM   =   1001,  IGAM  =    1100)
      parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
clin      PARAMETER (MAXP = 14000)
*----------------------------------------------------------------------*
      INTEGER   OUTPAR, zta,zpr
      COMMON  /AA/      R(3,MAXSTR)
cc      SAVE /AA/
      COMMON  /BB/      P(3,MAXSTR)
cc      SAVE /BB/
      COMMON  /CC/      E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON  /HH/  PROPER(MAXSTR)
cc      SAVE /HH/
      common  /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
      common  /gg/      dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
      COMMON  /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
      COMMON  /PP/      PRHO(-20:20,-24:24)
      COMMON  /QQ/      PHRHO(-MAXZ:MAXZ,-24:24)
      COMMON  /RR/      MASSR(0:MAXR)
cc      SAVE /RR/
      common  /ss/      inout(20)
cc      SAVE /ss/
      common  /zz/      zta,zpr
cc      SAVE /zz/
      COMMON  /RUN/     NUM
cc      SAVE /RUN/
clin-4/2008:
c      COMMON  /KKK/     TKAON(7),EKAON(7,0:200)
      COMMON  /KKK/     TKAON(7),EKAON(7,0:2000)
cc      SAVE /KKK/
      COMMON  /KAON/    AK(3,50,36),SPECK(50,36,7),MF
cc      SAVE /KAON/
      COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON  /DDpi/    piRHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DDpi/
      common  /tt/  PEL(-maxx:maxx,-maxx:maxx,-maxz:maxz)
     &,rxy(-maxx:maxx,-maxx:maxx,-maxz:maxz)
cc      SAVE /tt/
clin-4/2008:
c      DIMENSION TEMP(3,MAXSTR),SKAON(7),SEKAON(7,0:200)
      DIMENSION TEMP(3,MAXSTR),SKAON(7),SEKAON(7,0:2000)
cbz12/2/98
      COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &   IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
      COMMON /INPUT3/ PLAB, ELAB, ZEROPT, B0, BI, BM, DENCUT, CYCBOX
cc      SAVE /INPUT3/
cbz12/2/98end
cbz11/16/98
      COMMON /ARPRNT/ ARPAR1(100), IAPAR2(50), ARINT1(100), IAINT2(50)
cc      SAVE /ARPRNT/

c.....note in the below, since a common block in ART is called EE,
c.....the variable EE in /ARPRC/is changed to PEAR.
clin-9/29/03 changed name in order to distinguish from /prec2/
c        COMMON /ARPRC/ ITYPAR(MAXSTR),
c     &       GXAR(MAXSTR), GYAR(MAXSTR), GZAR(MAXSTR), FTAR(MAXSTR),
c     &       PXAR(MAXSTR), PYAR(MAXSTR), PZAR(MAXSTR), PEAR(MAXSTR),
c     &       XMAR(MAXSTR)
cc      SAVE /ARPRC/
clin-9/29/03-end
      COMMON /ARERCP/PRO1(MAXSTR, MAXR)
cc      SAVE /ARERCP/
      COMMON /ARERC1/MULTI1(MAXR)
cc      SAVE /ARERC1/
      COMMON /ARPRC1/ITYP1(MAXSTR, MAXR),
     &     GX1(MAXSTR, MAXR), GY1(MAXSTR, MAXR), GZ1(MAXSTR, MAXR), 
     &     FT1(MAXSTR, MAXR),
     &     PX1(MAXSTR, MAXR), PY1(MAXSTR, MAXR), PZ1(MAXSTR, MAXR),
     &     EE1(MAXSTR, MAXR), XM1(MAXSTR, MAXR)
cc      SAVE /ARPRC1/
c
      DIMENSION NPI(MAXR)
      DIMENSION RT(3, MAXSTR, MAXR), PT(3, MAXSTR, MAXR)
     &     , ET(MAXSTR, MAXR), LT(MAXSTR, MAXR), PROT(MAXSTR, MAXR)

      EXTERNAL IARFLV, INVFLV
cbz11/16/98end
      common /lastt/itimeh,bimp 
cc      SAVE /lastt/
      common/snn/efrm,npart1,npart2
cc      SAVE /snn/
      COMMON/hbt/lblast(MAXSTR),xlast(4,MAXSTR),plast(4,MAXSTR),nlast
cc      SAVE /hbt/
      common/resdcy/NSAV,iksdcy
cc      SAVE /resdcy/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON/FTMAX/ftsv(MAXSTR),ftsvt(MAXSTR, MAXR)
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      COMMON/HPARNT/HIPR1(100),IHPR2(50),HINT1(100),IHNT2(50)
clin-4/2008 zet() expanded to avoid out-of-bound errors:
      real zet(-45:45)
      SAVE   
      data zet /
     4     1.,0.,0.,0.,0.,
     3     1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
     2     -1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
     1     0.,0.,0.,-1.,0.,1.,0.,-1.,0.,-1.,
     s     0.,-2.,-1.,0.,1.,0.,0.,0.,0.,-1.,
     e     0.,
     s     1.,0.,-1.,0.,1.,-1.,0.,1.,2.,0.,
     1     1.,0.,1.,0.,-1.,0.,1.,0.,0.,0.,
     2     -1.,0.,1.,0.,-1.,0.,1.,0.,0.,1.,
     3     0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,
     4     0.,0.,0.,0.,-1./

      nlast=0
      do 1002 i=1,MAXSTR
         ftsv(i)=0.
         do 1101 irun=1,maxr
            ftsvt(i,irun)=0.
 1101    continue
         lblast(i)=999
         do 1001 j=1,4
clin-4/2008 bugs pointed out by Vander Molen & Westfall:
c            xlast(i,j)=0.
c            plast(i,j)=0.
            xlast(j,i)=0.
            plast(j,i)=0.
 1001    continue
 1002 continue

*-------------------------------------------------------------------*
* Input information about the reaction system and contral parameters* 
*-------------------------------------------------------------------*
*              input section starts here                           *
*-------------------------------------------------------------------*

cbz12/2/98
c.....input section is moved to subroutine ARTSET
cbz12/2/98end

*-----------------------------------------------------------------------*
*                   input section ends here                            *
*-----------------------------------------------------------------------*
* read in the table for gengrating the transverse momentum
* IN THE NN-->DDP PROCESS
       call tablem
* several control parameters, keep them fixed in this code. 
       ikaon=1
       nstar=1
       ndirct=0
       dir=0.02
       asy=0.032
       ESBIN=0.04
       MF=36
*----------------------------------------------------------------------*
c      CALL FRONT(12,MASSTA,MASSPR,ELAB)
*----------------------------------------------------------------------*
      RADTA  = 1.124 * FLOAT(MASSTA)**(1./3.)
      RADPR  = 1.124 * FLOAT(MASSPR)**(1./3.)
      ZDIST  = RADTA + RADPR
c      if ( cycbox.ne.0 ) zdist=0
      BMAX   = RADTA + RADPR
      MASS   = MASSTA + MASSPR
      NTOTAL = NUM * MASS
*
      IF (NTOTAL .GT. MAXSTR) THEN
        WRITE(12,'(//10X,''**** FATAL ERROR: TOO MANY TEST PART. ****'//
     & ' '')')
        STOP
      END IF
*
*-----------------------------------------------------------------------
*       RELATIVISTIC KINEMATICS
*
*       1) LABSYSTEM
*
      ETA    = FLOAT(MASSTA) * AMU
      PZTA   = 0.0
      BETATA = 0.0
      GAMMTA = 1.0
*
      EPR    = FLOAT(MASSPR) * (AMU + 0.001 * ELAB)
      PZPR   = SQRT( EPR**2 - (AMU * FLOAT(MASSPR))**2 )
      BETAPR = PZPR / EPR
      GAMMPR = 1.0 / SQRT( 1.0 - BETAPR**2 )
*
* BETAC AND GAMMAC OF THE C.M. OBSERVED IN THE LAB. FRAME
        BETAC=(PZPR+PZTA)/(EPR+ETA)
        GAMMC=1.0 / SQRT(1.-BETAC**2)
*
c      WRITE(12,'(/10x,''****    KINEMATICAL PARAMETERS    ****''/)')
c      WRITE(12,'(10x,''1) LAB-FRAME:        TARGET PROJECTILE'')')
c      WRITE(12,'(10x,''   ETOTAL "GEV" '',2F11.4)') ETA, EPR
c      WRITE(12,'(10x,''   P "GEV/C"    '',2F11.4)') PZTA, PZPR
c      WRITE(12,'(10x,''   BETA         '',2F11.4)') BETATA, BETAPR
c      WRITE(12,'(10x,''   GAMMA        '',2F11.4)') GAMMTA, GAMMPR
      IF (INSYS .NE. 0) THEN
*
*       2) C.M. SYSTEM
*
        S      = (EPR+ETA)**2 - PZPR**2
        xx1=4.*alog(float(massta))
        xx2=4.*alog(float(masspr))
        xx1=exp(xx1)
        xx2=exp(xx2)
        PSQARE = (S**2 + (xx1+ xx2) * AMU**4
     &             - 2.0 * S * AMU**2 * FLOAT(MASSTA**2 + MASSPR**2)
     &             - 2.0 * FLOAT(MASSTA**2 * MASSPR**2) * AMU**4)
     &           / (4.0 * S)
*
        ETA    = SQRT ( PSQARE + (FLOAT(MASSTA) * AMU)**2 )
        PZTA   = - SQRT(PSQARE)
        BETATA = PZTA / ETA
        GAMMTA = 1.0 / SQRT( 1.0 - BETATA**2 )
*
        EPR    = SQRT ( PSQARE + (FLOAT(MASSPR) * AMU)**2 )
        PZPR   = SQRT(PSQARE)
        BETAPR = PZPR/ EPR
        GAMMPR = 1.0 / SQRT( 1.0 - BETAPR**2 )
*
c        WRITE(12,'(10x,''2) C.M.-FRAME:  '')')
c        WRITE(12,'(10x,''   ETOTAL "GEV" '',2F11.4)') ETA, EPR
c        WRITE(12,'(10x,''   P "GEV/C"    '',2F11.4)') PZTA, PZPR
c        WRITE(12,'(10x,''   BETA         '',2F11.4)') BETATA, BETAPR
c        WRITE(12,'(10x,''   GAMMA        '',2F11.4)') GAMMTA, GAMMPR
c        WRITE(12,'(10x,''S "GEV**2"      '',F11.4)')  S
c        WRITE(12,'(10x,''PSQARE "GEV/C"2 '',E14.3)')  PSQARE
c        WRITE(12,'(/10x,''*** CALCULATION DONE IN CM-FRAME ***''/)')
      ELSE
c        WRITE(12,'(/10x,''*** CALCULATION DONE IN LAB-FRAME ***''/)')
      END IF
* MOMENTUM PER PARTICLE
      PZTA = PZTA / FLOAT(MASSTA)
      PZPR = PZPR / FLOAT(MASSPR)
* total initial energy in the N-N cms frame
      ECMS0=ETA+EPR
*-----------------------------------------------------------------------
*
* Start loop over many runs of different impact parameters
* IF MANYB=1, RUN AT A FIXED IMPACT PARAMETER B0, OTHERWISE GENERATE 
* MINIMUM BIAS EVENTS WITHIN THE IMPACT PARAMETER RANGE OF B_MIN AND B_MAX
       DO 50000 IMANY=1,MANYB
*------------------------------------------------------------------------
* Initialize the impact parameter B
       if (manyb. gt.1) then
111       BX=1.0-2.0*RANART(NSEED)
       BY=1.0-2.0*RANART(NSEED)
       B2=BX*BX+BY*BY
       IF(B2.GT.1.0) GO TO 111       
       B=SQRT(B2)*(BM-BI)+BI
       ELSE
       B=B0
       ENDIF
c      WRITE(12,'(///10X,''RUN NUMBER:'',I6)') IMANY       
c      WRITE(12,'(//10X,''IMPACT PARAMETER B FOR THIS RUN:'',
c     &             F9.3,'' FM''/10X,49(''*'')/)') B
*
*-----------------------------------------------------------------------
*       INITIALIZATION
*1 INITIALIZATION IN ISOSPIN SPACE FOR BOTH THE PROJECTILE AND TARGET
      call coulin(masspr,massta,NUM)
*2 INITIALIZATION IN PHASE SPACE FOR THE TARGET
      CALL INIT(1       ,MASSTA   ,NUM     ,RADTA,
     &          B/2.    ,ZEROPT+ZDIST/2.   ,PZTA,
     &          GAMMTA  ,ISEED    ,MASS    ,IMOMEN)
*3.1 INITIALIZATION IN PHASE SPACE FOR THE PROJECTILE
      CALL INIT(1+MASSTA,MASS     ,NUM     ,RADPR,
     &          -B/2.   ,ZEROPT-ZDIST/2.   ,PZPR,
     &          GAMMPR  ,ISEED    ,MASS    ,IMOMEN)
*3.2 OUTPAR IS THE NO. OF ESCAPED PARTICLES
      OUTPAR = 0
*3.3 INITIALIZATION FOR THE NO. OF PARTICLES IN EACH SAMPLE
*    THIS IS NEEDED DUE TO THE FACT THAT PIONS CAN BE PRODUCED OR ABSORBED
      MASSR(0)=0
      DO 1003 IR =1,NUM
      MASSR(IR)=MASS
 1003 CONTINUE
*3.4 INITIALIZation FOR THE KAON SPECTRUM
*      CALL KSPEC0(BETAC,GAMMC)
* calculate the local baryon density matrix
      CALL DENS(IPOT,MASS,NUM,OUTPAR)
*
*-----------------------------------------------------------------------
*       CONTROL PRINTOUT OF INITIAL CONFIGURATION
*
*      WRITE(12,'(''**********  INITIAL CONFIGURATION  **********''/)')
*
c print out the INITIAL density matrix in the reaction plane
c       do ix=-10,10
c       do iz=-10,10
c       write(1053,992)ix,iz,rho(ix,0,iz)/0.168
c       end do
c       end do
*-----------------------------------------------------------------------
*       CALCULATE MOMENTA FOR T = 0.5 * DT 
*       (TO OBTAIN 2ND DEGREE ACCURACY!)
*       "Reference: J. AICHELIN ET AL., PHYS. REV. C31, 1730 (1985)"
*
      IF (ICOLL .NE. -1) THEN
        DO 700 I = 1,NTOTAL
          IX = NINT( R(1,I) )
          IY = NINT( R(2,I) )
          IZ = NINT( R(3,I) )
clin-4/2008 check bounds:
          IF(IX.GE.MAXX.OR.IY.GE.MAXX.OR.IZ.GE.MAXZ
     1         .OR.IX.LE.-MAXX.OR.IY.LE.-MAXX.OR.IZ.LE.-MAXZ) goto 700
          CALL GRADU(IPOT,IX,IY,IZ,GRADX,GRADY,GRADZ)
          P(1,I) = P(1,I) - (0.5 * DT) * GRADX
          P(2,I) = P(2,I) - (0.5 * DT) * GRADY
          P(3,I) = P(3,I) - (0.5 * DT) * GRADZ
  700   CONTINUE
      END IF
*-----------------------------------------------------------------------
*-----------------------------------------------------------------------
*4 INITIALIZATION OF TIME-LOOP VARIABLES
*4.1 COLLISION NUMBER COUNTERS
clin 51      RCNNE  = 0
        RCNNE  = 0
       RDD  = 0
       RPP  = 0
       rppk = 0
       RPN  = 0
       rpd  = 0
       RKN  = 0
       RNNK = 0
       RDDK = 0
       RNDK = 0
      RCNND  = 0
      RCNDN  = 0
      RCOLL  = 0
      RBLOC  = 0
      RDIRT  = 0
      RDECAY = 0
      RRES   = 0
*4.11 KAON PRODUCTION PROBABILITY COUNTER FOR PERTURBATIVE CALCULATIONS ONLY
      DO 1005 KKK=1,5
         SKAON(KKK)  = 0
         DO 1004 IS=1,2000
            SEKAON(KKK,IS)=0
 1004    CONTINUE
 1005 CONTINUE
*4.12 anti-proton and anti-kaon counters
       pr0=0.
       pr1=0.
       ska0=0.
       ska1=0.
*       ============== LOOP OVER ALL TIME STEPS ================       *
*                             STARTS HERE                              *
*       ========================================================       *
cbz11/16/98
      IF (IAPAR2(1) .NE. 1) THEN
         DO 1016 I = 1, MAXSTR
            DO 1015 J = 1, 3
               R(J, I) = 0.
               P(J, I) = 0.
 1015       CONTINUE
            E(I) = 0.
            LB(I) = 0
cbz3/25/00
            ID(I)=0
c     sp 12/19/00
           PROPER(I) = 1.
 1016   CONTINUE
         MASS = 0
cbz12/22/98
c         MASSR(1) = 0
c         NP = 0
c         NPI = 1
         NP = 0
         DO 1017 J = 1, NUM
            MASSR(J) = 0
            NPI(J) = 1
 1017    CONTINUE
         DO 1019 I = 1, MAXR
            DO 1018 J = 1, MAXSTR
               RT(1, J, I) = 0.
               RT(2, J, I) = 0.
               RT(3, J, I) = 0.
               PT(1, J, I) = 0.
               PT(2, J, I) = 0.
               PT(3, J, I) = 0.
               ET(J, I) = 0.
               LT(J, I) = 0
c     sp 12/19/00
               PROT(J, I) = 1.
 1018       CONTINUE
 1019    CONTINUE
cbz12/22/98end
      END IF
cbz11/16/98end
        
      DO 10000 NT = 1,NTMAX

*TEMPORARY PARTICLE COUNTERS
*4.2 PION COUNTERS : LP1,LP2 AND LP3 ARE THE NO. OF P+,P0 AND P-
      LP1=0
      LP2=0
      LP3=0
*4.3 DELTA COUNTERS : LD1,LD2,LD3 AND LD4 ARE THE NO. OF D++,D+,D0 AND D-
      LD1=0
      LD2=0
      LD3=0
      LD4=0
*4.4 N*(1440) COUNTERS : LN1 AND LN2 ARE THE NO. OF N*+ AND N*0
      LN1=0
      LN2=0
*4.5 N*(1535) counters
      LN5=0
*4.6 ETA COUNTERS
      LE=0
*4.7 KAON COUNTERS
      LKAON=0

clin-11/09/00:
* KAON* COUNTERS
      LKAONS=0

*-----------------------------------------------------------------------
        IF (ICOLL .NE. 1) THEN
* STUDYING BINARY COLLISIONS AMONG PARTICLES DURING THIS TIME INTERVAL *
clin-10/25/02 get rid of argument usage mismatch in relcol(.nt.):
           numnt=nt
          CALL RELCOL(LCOLL,LBLOC,LCNNE,LDD,LPP,lppk,
     &    LPN,lpd,LRHO,LOMEGA,LKN,LNNK,LDDK,LNDK,LCNND,
     &    LCNDN,LDIRT,LDECAY,LRES,LDOU,LDDRHO,LNNRHO,
     &    LNNOM,numnt,ntmax,sp,akaon,sk)
c     &    LNNOM,NT,ntmax,sp,akaon,sk)
clin-10/25/02-end
*-----------------------------------------------------------------------

c dilepton production from Dalitz decay
c of pi0 at final time
*      if(nt .eq. ntmax) call dalitz_pi(nt,ntmax)
*                                                                      *
**********************************
*                Lables of collision channels                             *
**********************************
*         LCOLL   - NUMBER OF COLLISIONS              (INTEGER,OUTPUT) *
*         LBLOC   - NUMBER OF PULI-BLOCKED COLLISIONS (INTEGER,OUTPUT) *
*         LCNNE   - NUMBER OF ELASTIC COLLISION       (INTEGER,OUTPUT) *
*         LCNND   - NUMBER OF N+N->N+DELTA REACTION   (INTEGER,OUTPUT) *
*         LCNDN   - NUMBER OF N+DELTA->N+N REACTION   (INTEGER,OUTPUT) *
*         LDD     - NUMBER OF RESONANCE+RESONANCE COLLISIONS
*         LPP     - NUMBER OF PION+PION elastic COLIISIONS
*         lppk    - number of pion(RHO,OMEGA)+pion(RHO,OMEGA)
*                 -->K+K- collisions
*         LPN     - NUMBER OF PION+N-->KAON+X
*         lpd     - number of pion+n-->delta+pion
*         lrho    - number of pion+n-->Delta+rho
*         lomega  - number of pion+n-->Delta+omega
*         LKN     - NUMBER OF KAON RESCATTERINGS
*         LNNK    - NUMBER OF bb-->kAON PROCESS
*         LDDK    - NUMBER OF DD-->KAON PROCESS
*         LNDK    - NUMBER OF ND-->KAON PROCESS
***********************************
* TIME-INTEGRATED COLLISIONS NUMBERS OF VARIOUS PROCESSES
          RCOLL = RCOLL + FLOAT(LCOLL)/num
          RBLOC = RBLOC + FLOAT(LBLOC)/num
          RCNNE = RCNNE + FLOAT(LCNNE)/num
         RDD   = RDD   + FLOAT(LDD)/num
         RPP   = RPP   + FLOAT(LPP)/NUM
         rppk  =rppk   + float(lppk)/num
         RPN   = RPN   + FLOAT(LPN)/NUM
         rpd   =rpd    + float(lpd)/num
         RKN   = RKN   + FLOAT(LKN)/NUM
         RNNK  =RNNK   + FLOAT(LNNK)/NUM
         RDDK  =RDDK   + FLOAT(LDDK)/NUM
         RNDK  =RNDK   + FLOAT(LNDK)/NUM
          RCNND = RCNND + FLOAT(LCNND)/num
          RCNDN = RCNDN + FLOAT(LCNDN)/num
          RDIRT = RDIRT + FLOAT(LDIRT)/num
          RDECAY= RDECAY+ FLOAT(LDECAY)/num
          RRES  = RRES  + FLOAT(LRES)/num
* AVERAGE RATES OF VARIOUS COLLISIONS IN THE CURRENT TIME STEP
          ADIRT=LDIRT/DT/num
          ACOLL=(LCOLL-LBLOC)/DT/num
          ACNND=LCNND/DT/num
          ACNDN=LCNDN/DT/num
          ADECAY=LDECAY/DT/num
          ARES=LRES/DT/num
         ADOU=LDOU/DT/NUM
         ADDRHO=LDDRHO/DT/NUM
         ANNRHO=LNNRHO/DT/NUM
         ANNOM=LNNOM/DT/NUM
         ADD=LDD/DT/num
         APP=LPP/DT/num
         appk=lppk/dt/num
          APN=LPN/DT/num
         apd=lpd/dt/num
         arh=lrho/dt/num
         aom=lomega/dt/num
         AKN=LKN/DT/num
         ANNK=LNNK/DT/num
         ADDK=LDDK/DT/num
         ANDK=LNDK/DT/num
* PRINT OUT THE VARIOUS COLLISION RATES
* (1)N-N COLLISIONS 
c       WRITE(1010,9991)NT*DT,ACNND,ADOU,ADIRT,ADDRHO,ANNRHO+ANNOM
c9991       FORMAT(6(E10.3,2X))
* (2)PION-N COLLISIONS
c       WRITE(1011,'(5(E10.3,2X))')NT*DT,apd,ARH,AOM,APN
* (3)KAON PRODUCTION CHANNELS
c        WRITE(1012,9993)NT*DT,ANNK,ADDK,ANDK,APN,Appk
* (4)D(N*)+D(N*) COLLISION
c       WRITE(1013,'(4(E10.3,2X))')NT*DT,ADDK,ADD,ADD+ADDK
* (5)MESON+MESON
c       WRITE(1014,'(4(E10.3,2X))')NT*DT,APPK,APP,APP+APPK
* (6)DECAY AND RESONANCE
c       WRITE(1016,'(3(E10.3,2X))')NT*DT,ARES,ADECAY
* (7)N+D(N*)
c       WRITE(1017,'(4(E10.3,2X))')NT*DT,ACNDN,ANDK,ACNDN+ANDK
c9992    FORMAT(5(E10.3,2X))
c9993    FORMAT(6(E10.3,2X))
* PRINT OUT TIME-INTEGRATED COLLISION INFORMATION
cbz12/28/98
c        write(1018,'(5(e10.3,2x),/, 4(e10.3,2x))')
c     &           RCNNE,RCNND,RCNDN,RDIRT,rpd,
c     &           RDECAY,RRES,RDD,RPP
c        write(1018,'(6(e10.3,2x),/, 5(e10.3,2x))')
c     &           NT*DT,RCNNE,RCNND,RCNDN,RDIRT,rpd,
c     &           NT*DT,RDECAY,RRES,RDD,RPP
cbz12/18/98end
* PRINT OUT TIME-INTEGRATED KAON MULTIPLICITIES FROM DIFFERENT CHANNELS
c       WRITE(1019,'(7(E10.3,2X))')NT*DT,RNNK,RDDK,RNDK,RPN,Rppk,
c     &                           RNNK+RDDK+RNDK+RPN+Rppk
*                                                                      *

        END IF
*
*       UPDATE BARYON DENSITY
*
        CALL DENS(IPOT,MASS,NUM,OUTPAR)
*
*       UPDATE POSITIONS FOR ALL THE PARTICLES PRESENT AT THIS TIME
*
       sumene=0
        ISO=0
        DO 201 MRUN=1,NUM
        ISO=ISO+MASSR(MRUN-1)
        DO 201 I0=1,MASSR(MRUN)
        I =I0+ISO
        ETOTAL = SQRT( E(I)**2 + P(1,I)**2 + P(2,I)**2 +P(3,I)**2 )
       sumene=sumene+etotal
C for kaons, if there is a potential
C CALCULATE THE ENERGY OF THE KAON ACCORDING TO THE IMPULSE APPROXIMATION
C REFERENCE: B.A. LI AND C.M. KO, PHYS. REV. C 54 (1996) 3283. 
         if(kpoten.ne.0.and.lb(i).eq.23)then
             den=0.
              IX = NINT( R(1,I) )
              IY = NINT( R(2,I) )
              IZ = NINT( R(3,I) )
clin-4/2008:
c       IF (ABS(IX) .LT. MAXX .AND. ABS(IY) .LT. MAXX .AND.
c     & ABS(IZ) .LT. MAXZ) den=rho(ix,iy,iz)
              IF(IX.LT.MAXX.AND.IY.LT.MAXX.AND.IZ.LT.MAXZ
     1             .AND.IX.GT.-MAXX.AND.IY.GT.-MAXX.AND.IZ.GT.-MAXZ)
     2             den=rho(ix,iy,iz)
c         ecor=0.1973**2*0.255*kmul*4*3.14159*(1.+0.4396/0.938)
c         etotal=sqrt(etotal**2+ecor*den)
c** G.Q Li potential form with n_s = n_b and pot(n_0)=29 MeV, m^*=m
c     GeV^2 fm^3
          akg = 0.1727
c     GeV fm^3
          bkg = 0.333
         rnsg = den
         ecor = - akg*rnsg + (bkg*den)**2
         etotal = sqrt(etotal**2 + ecor)
         endif
c
         if(kpoten.ne.0.and.lb(i).eq.21)then
             den=0.
              IX = NINT( R(1,I) )
              IY = NINT( R(2,I) )
              IZ = NINT( R(3,I) )
clin-4/2008:
c       IF (ABS(IX) .LT. MAXX .AND. ABS(IY) .LT. MAXX .AND.
c     & ABS(IZ) .LT. MAXZ) den=rho(ix,iy,iz)
              IF(IX.LT.MAXX.AND.IY.LT.MAXX.AND.IZ.LT.MAXZ
     1             .AND.IX.GT.-MAXX.AND.IY.GT.-MAXX.AND.IZ.GT.-MAXZ)
     2             den=rho(ix,iy,iz)
c* for song potential no effect on position
c** G.Q Li potential form with n_s = n_b and pot(n_0)=29 MeV, m^*=m
c     GeV^2 fm^3
          akg = 0.1727
c     GeV fm^3
          bkg = 0.333
         rnsg = den
         ecor = - akg*rnsg + (bkg*den)**2
         etotal = sqrt(etotal**2 + ecor)
          endif
c
C UPDATE POSITIONS
          R(1,I) = R(1,I) + DT*P(1,I)/ETOTAL
          R(2,I) = R(2,I) + DT*P(2,I)/ETOTAL
          R(3,I) = R(3,I) + DT*P(3,I)/ETOTAL
c use cyclic boundary conitions
            if ( cycbox.ne.0 ) then
              if ( r(1,i).gt. cycbox/2 ) r(1,i)=r(1,i)-cycbox
              if ( r(1,i).le.-cycbox/2 ) r(1,i)=r(1,i)+cycbox
              if ( r(2,i).gt. cycbox/2 ) r(2,i)=r(2,i)-cycbox
              if ( r(2,i).le.-cycbox/2 ) r(2,i)=r(2,i)+cycbox
              if ( r(3,i).gt. cycbox/2 ) r(3,i)=r(3,i)-cycbox
              if ( r(3,i).le.-cycbox/2 ) r(3,i)=r(3,i)+cycbox
            end if
* UPDATE THE DELTA, N* AND PION COUNTERS
          LB1=LB(I)
* 1. FOR DELTA++
        IF(LB1.EQ.9)LD1=LD1+1
* 2. FOR DELTA+
        IF(LB1.EQ.8)LD2=LD2+1
* 3. FOR DELTA0
        IF(LB1.EQ.7)LD3=LD3+1
* 4. FOR DELTA-
        IF(LB1.EQ.6)LD4=LD4+1
* 5. FOR N*+(1440)
        IF(LB1.EQ.11)LN1=LN1+1
* 6. FOR N*0(1440)
        IF(LB1.EQ.10)LN2=LN2+1
* 6.1 FOR N*(1535)
       IF((LB1.EQ.13).OR.(LB1.EQ.12))LN5=LN5+1
* 6.2 FOR ETA
       IF(LB1.EQ.0)LE=LE+1
* 6.3 FOR KAONS
       IF(LB1.EQ.23)LKAON=LKAON+1
clin-11/09/00: FOR KAON*
       IF(LB1.EQ.30)LKAONS=LKAONS+1

* UPDATE PION COUNTER
* 7. FOR PION+
        IF(LB1.EQ.5)LP1=LP1+1
* 8. FOR PION0
        IF(LB1.EQ.4)LP2=LP2+1
* 9. FOR PION-
        IF(LB1.EQ.3)LP3=LP3+1
201     CONTINUE
        LP=LP1+LP2+LP3
        LD=LD1+LD2+LD3+LD4
        LN=LN1+LN2
        ALP=FLOAT(LP)/FLOAT(NUM)
        ALD=FLOAT(LD)/FLOAT(NUM)
        ALN=FLOAT(LN)/FLOAT(NUM)
       ALN5=FLOAT(LN5)/FLOAT(NUM)
        ATOTAL=ALP+ALD+ALN+0.5*ALN5
       ALE=FLOAT(LE)/FLOAT(NUM)
       ALKAON=FLOAT(LKAON)/FLOAT(NUM)
* UPDATE MOMENTUM DUE TO COULOMB INTERACTION 
        if (icou .eq. 1) then
*       with Coulomb interaction
          iso=0
          do 1026 irun = 1,num
            iso=iso+massr(irun-1)
            do 1021 il = 1,massr(irun)
               temp(1,il) = 0.
               temp(2,il) = 0.
               temp(3,il) = 0.
 1021       continue
            do 1023 il = 1, massr(irun)
              i=iso+il
              if (zet(lb(i)).ne.0) then
                do 1022 jl = 1,il-1
                  j=iso+jl
                  if (zet(lb(j)).ne.0) then
                    ddx=r(1,i)-r(1,j)
                    ddy=r(2,i)-r(2,j)
                    ddz=r(3,i)-r(3,j)
                    rdiff = sqrt(ddx**2+ddy**2+ddz**2)
                    if (rdiff .le. 1.) rdiff = 1.
                    grp=zet(lb(i))*zet(lb(j))/rdiff**3
                    ddx=ddx*grp
                    ddy=ddy*grp
                    ddz=ddz*grp
                    temp(1,il)=temp(1,il)+ddx
                    temp(2,il)=temp(2,il)+ddy
                    temp(3,il)=temp(3,il)+ddz
                    temp(1,jl)=temp(1,jl)-ddx
                    temp(2,jl)=temp(2,jl)-ddy
                    temp(3,jl)=temp(3,jl)-ddz
                  end if
 1022          continue
              end if
 1023      continue
            do 1025 il = 1,massr(irun)
              i= iso+il
              if (zet(lb(i)).ne.0) then
                do 1024 idir = 1,3
                  p(idir,i) = p(idir,i) + temp(idir,il)
     &                                    * dt * 0.00144
 1024          continue
              end if
 1025      continue
 1026   continue
        end if
*       In the following, we shall:  
*       (1) UPDATE MOMENTA DUE TO THE MEAN FIELD FOR BARYONS AND KAONS,
*       (2) calculate the thermalization, temperature in a sphere of 
*           radius 2.0 fm AROUND THE CM
*       (3) AND CALCULATE THE NUMBER OF PARTICLES IN THE HIGH DENSITY REGION 
       spt=0
       spz=0
       ncen=0
       ekin=0
          NLOST = 0
          MEAN=0
         nquark=0
         nbaryn=0
csp06/18/01
           rads = 2.
           zras = 0.1
           denst = 0.
           edenst = 0.
csp06/18/01 end
          DO 6000 IRUN = 1,NUM
          MEAN=MEAN+MASSR(IRUN-1)
          DO 5800 J = 1,MASSR(irun)
          I=J+MEAN
c
csp06/18/01
           radut = sqrt(r(1,i)**2+r(2,i)**2)
       if( radut .le. rads )then
        if( abs(r(3,i)) .le. zras*nt*dt )then
c         vols = 3.14159*radut**2*abs(r(3,i))      ! cylinder pi*r^2*l
c     cylinder pi*r^2*l
         vols = 3.14159*rads**2*zras
         engs=sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2+e(i)**2)
         gammas=1.
         if(e(i).ne.0.)gammas=engs/e(i)
c     rho
         denst = denst + 1./gammas/vols
c     energy density
         edenst = edenst + engs/gammas/gammas/vols
        endif
       endif
csp06/18/01 end
c
         drr=sqrt(r(1,i)**2+r(2,i)**2+r(3,i)**2)
         if(drr.le.2.0)then
         spt=spt+p(1,i)**2+p(2,i)**2
         spz=spz+p(3,i)**2
         ncen=ncen+1
         ekin=ekin+sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2+e(i)**2)-e(i)
         endif
              IX = NINT( R(1,I) )
              IY = NINT( R(2,I) )
              IZ = NINT( R(3,I) )
C calculate the No. of particles in the high density region
clin-4/2008:
c              IF (ABS(IX) .LT. MAXX .AND. ABS(IY) .LT. MAXX .AND.
c     & ABS(IZ) .LT. MAXZ) THEN
              IF(IX.LT.MAXX.AND.IY.LT.MAXX.AND.IZ.LT.MAXZ
     1          .AND.IX.GT.-MAXX.AND.IY.GT.-MAXX.AND.IZ.GT.-MAXZ) THEN
       if(rho(ix,iy,iz)/0.168.gt.dencut)go to 5800
       if((rho(ix,iy,iz)/0.168.gt.5.).and.(e(i).gt.0.9))
     &  nbaryn=nbaryn+1
       if(pel(ix,iy,iz).gt.2.0)nquark=nquark+1
       endif
c*
c If there is a kaon potential, propogating kaons 
        if(kpoten.ne.0.and.lb(i).eq.23)then
        den=0.
clin-4/2008:
c       IF (ABS(IX) .LT. MAXX .AND. ABS(IY) .LT. MAXX .AND.
c     & ABS(IZ) .LT. MAXZ)then
        IF(IX.LT.MAXX.AND.IY.LT.MAXX.AND.IZ.LT.MAXZ
     1       .AND.IX.GT.-MAXX.AND.IY.GT.-MAXX.AND.IZ.GT.-MAXZ) THEN
           den=rho(ix,iy,iz)
c        ecor=0.1973**2*0.255*kmul*4*3.14159*(1.+0.4396/0.938)
c       etotal=sqrt(P(1,i)**2+p(2,I)**2+p(3,i)**2+e(i)**2+ecor*den)
c** for G.Q Li potential form with n_s = n_b and pot(n_0)=29 MeV
c     !! GeV^2 fm^3
            akg = 0.1727
c     !! GeV fm^3
            bkg = 0.333
          rnsg = den
          ecor = - akg*rnsg + (bkg*den)**2
          etotal = sqrt(P(1,i)**2+p(2,I)**2+p(3,i)**2+e(i)**2 + ecor)
          ecor = - akg + 2.*bkg**2*den + 2.*bkg*etotal
c** G.Q. Li potential (END)           
        CALL GRADUK(IX,IY,IZ,GRADXk,GRADYk,GRADZk)
        P(1,I) = P(1,I) - DT * GRADXk*ecor/(2.*etotal)
        P(2,I) = P(2,I) - DT * GRADYk*ecor/(2.*etotal)
        P(3,I) = P(3,I) - DT * GRADZk*ecor/(2.*etotal)
        endif
         endif
c
        if(kpoten.ne.0.and.lb(i).eq.21)then
         den=0.
clin-4/2008:
c           IF (ABS(IX) .LT. MAXX .AND. ABS(IY) .LT. MAXX .AND.
c     &        ABS(IZ) .LT. MAXZ)then
         IF(IX.LT.MAXX.AND.IY.LT.MAXX.AND.IZ.LT.MAXZ
     1        .AND.IX.GT.-MAXX.AND.IY.GT.-MAXX.AND.IZ.GT.-MAXZ) THEN
               den=rho(ix,iy,iz)
        CALL GRADUK(IX,IY,IZ,GRADXk,GRADYk,GRADZk)
c        P(1,I) = P(1,I) - DT * GRADXk*(-0.12/0.168)    !! song potential
c        P(2,I) = P(2,I) - DT * GRADYk*(-0.12/0.168)
c        P(3,I) = P(3,I) - DT * GRADZk*(-0.12/0.168)
c** for G.Q Li potential form with n_s = n_b and pot(n_0)=29 MeV
c    !! GeV^2 fm^3
            akg = 0.1727
c     !! GeV fm^3
            bkg = 0.333
          rnsg = den
          ecor = - akg*rnsg + (bkg*den)**2
          etotal = sqrt(P(1,i)**2+p(2,I)**2+p(3,i)**2+e(i)**2 + ecor)
          ecor = - akg + 2.*bkg**2*den - 2.*bkg*etotal
        P(1,I) = P(1,I) - DT * GRADXk*ecor/(2.*etotal)
        P(2,I) = P(2,I) - DT * GRADYk*ecor/(2.*etotal)
        P(3,I) = P(3,I) - DT * GRADZk*ecor/(2.*etotal)
c** G.Q. Li potential (END)           
        endif
         endif
c
c for other mesons, there is no potential
       if(j.gt.mass)go to 5800         
c  with mean field interaction for baryons   (open endif below) !!sp05
**      if( (iabs(lb(i)).eq.1.or.iabs(lb(i)).eq.2) .or.
**    &     (iabs(lb(i)).ge.6.and.iabs(lb(i)).le.17) .or.
**    &      iabs(lb(i)).eq.40.or.iabs(lb(i)).eq.41 )then  
        IF (ICOLL .NE. -1) THEN
* check if the baryon has run off the lattice
*             IX0=NINT(R(1,I)/DX)
*             IY0=NINT(R(2,I)/DY)
*             IZ0=NINT(R(3,I)/DZ)
*             IPX0=NINT(P(1,I)/DPX)
*             IPY0=NINT(P(2,I)/DPY)
*             IPZ0=NINT(P(3,I)/DPZ)
*      if ( (abs(ix0).gt.mx) .or. (abs(iy0).gt.my) .or. (abs(iz0).gt.mz)
*     &  .or. (abs(ipx0).gt.mpx) .or. (abs(ipy0) 
*     &  .or. (ipz0.lt.-mpz) .or. (ipz0.gt.mpzp)) NLOST=NLOST+1
clin-4/2008:
c              IF (ABS(IX) .LT. MAXX .AND. ABS(IY) .LT. MAXX .AND.
c     &                                    ABS(IZ) .LT. MAXZ     ) THEN
           IF(IX.LT.MAXX.AND.IY.LT.MAXX.AND.IZ.LT.MAXZ
     1          .AND.IX.GT.-MAXX.AND.IY.GT.-MAXX.AND.IZ.GT.-MAXZ) THEN
                CALL GRADU(IPOT,IX,IY,IZ,GRADX,GRADY,GRADZ)
              TZ=0.
              GRADXN=0
              GRADYN=0
              GRADZN=0
              GRADXP=0
              GRADYP=0
              GRADZP=0
             IF(ICOU.EQ.1)THEN
                CALL GRADUP(IX,IY,IZ,GRADXP,GRADYP,GRADZP)
                CALL GRADUN(IX,IY,IZ,GRADXN,GRADYN,GRADZN)
               IF(ZET(LB(I)).NE.0)TZ=-1
               IF(ZET(LB(I)).EQ.0)TZ= 1
             END IF
           if(iabs(lb(i)).ge.14.and.iabs(lb(i)).le.17)then
              facl = 2./3.
            elseif(iabs(lb(i)).eq.40.or.iabs(lb(i)).eq.41)then
              facl = 1./3.
            else
              facl = 1.
            endif
        P(1,I) = P(1,I) - facl*DT * (GRADX+asy*(GRADXN-GRADXP)*TZ)
        P(2,I) = P(2,I) - facl*DT * (GRADY+asy*(GRADYN-GRADYP)*TZ)
        P(3,I) = P(3,I) - facl*DT * (GRADZ+asy*(GRADZN-GRADZP)*TZ)
                end if                                                       
              ENDIF
**          endif          !!sp05     
 5800       CONTINUE
 6000       CONTINUE
c print out the average no. of particles in regions where the local 
c baryon density is higher than 5*rho0 
c       write(1072,'(e10.3,2x,e10.3)')nt*dt,float(nbaryn)/float(num)
C print out the average no. of particles in regions where the local 
c energy density is higher than 2 GeV/fm^3. 
c       write(1073,'(e10.3,2x,e10.3)')nt*dt,float(nquark)/float(num)
c print out the no. of particles that have run off the lattice
*          IF (NLOST .NE. 0 .AND. (NT/NFREQ)*NFREQ .EQ. NT) THEN
*            WRITE(12,'(5X,''***'',I7,'' TESTPARTICLES LOST AFTER '',
*     &                   ''TIME STEP NUMBER'',I4)') NLOST, NT
*         END IF
*
*       update phase space density
*        call platin(mode,mass,num,dx,dy,dz,dpx,dpy,dpz,fnorm)
*
*       CONTROL-PRINTOUT OF CONFIGURATION (IF REQUIRED)
*
*        if (inout(5) .eq. 2) CALL ENERGY(NT,IPOT,NUM,MASS,EMIN,EMAX)
*
* 
* print out central baryon density as a function of time
       CDEN=RHO(0,0,0)/0.168
cc        WRITE(1002,990)FLOAT(NT)*DT,CDEN
c        WRITE(1002,1990)FLOAT(NT)*DT,CDEN,denst/real(num)
* print out the central energy density as a function of time
cc        WRITE(1003,990)FLOAT(NT)*DT,PEL(0,0,0)
c        WRITE(1003,1990)FLOAT(NT)*DT,PEL(0,0,0),edenst/real(num)
* print out the no. of pion-like particles as a function of time 
c        WRITE(1004,9999)FLOAT(NT)*DT,ALD,ALN,ALP,ALN5,
c     &               ALD+ALN+ALP+0.5*ALN5
* print out the no. of eta-like particles as a function of time
c        WRITE(1005,991)FLOAT(NT)*DT,ALN5,ALE,ALE+0.5*ALN5
c990       FORMAT(E10.3,2X,E10.3)
c1990       FORMAT(E10.3,2X,E10.3,2X,E10.3)
c991       FORMAT(E10.3,2X,E10.3,2X,E10.3,2X,E10.3)
c9999    FORMAT(e10.3,2X,e10.3,2X,E10.3,2X,E10.3,2X,
c     1  E10.3,2X,E10.3)
C THE FOLLOWING OUTPUTS CAN BE TURNED ON/OFF by setting icflow and icrho=0  
c print out the baryon and meson density matrix in the reaction plane
        IF ((NT/NFREQ)*NFREQ .EQ. NT ) THEN
       if(icflow.eq.1)call flow(nt)
cbz11/18/98
c       if(icrho.ne.1)go to 10000 
c       if (icrho .eq. 1) then 
cbz11/18/98end
c       do ix=-10,10
c       do iz=-10,10
c       write(1053,992)ix,iz,rho(ix,0,iz)/0.168
c       write(1054,992)ix,iz,pirho(ix,0,iz)/0.168
c       write(1055,992)ix,iz,pel(ix,0,iz)
c       end do
c       end do
cbz11/18/98
c        end if
cbz11/18/98end
c992       format(i3,i3,e11.4)
       endif
c print out the ENERGY density matrix in the reaction plane
C CHECK LOCAL MOMENTUM EQUILIBRIUM IN EACH CELL, 
C AND PERFORM ON-LINE FLOW ANALYSIS AT A FREQUENCY OF NFREQ
c        IF ((NT/NFREQ)*NFREQ .EQ. NT ) THEN
c       call flow(nt)
c       call equ(ipot,mass,num,outpar)
c       do ix=-10,10
c       do iz=-10,10
c       write(1055,992)ix,iz,pel(ix,0,iz)
c       write(1056,992)ix,iz,rxy(ix,0,iz)
c       end do
c       end do
c       endif
C calculate the volume of high BARYON AND ENERGY density 
C matter as a function of time
c       vbrho=0.
c       verho=0.
c       do ix=-20,20
c       do iy=-20,20
c       do iz=-20,20
c       if(rho(ix,iy,iz)/0.168.gt.5.)vbrho=vbrho+1.
c       if(pel(ix,iy,iz).gt.2.)verho=verho+1.
c       end do
c       end do
c       end do
c       write(1081,993)dt*nt,vbrho
c       write(1082,993)dt*nt,verho
c993       format(e11.4,2x,e11.4)
*-----------------------------------------------------------------------
cbz11/16/98
c.....for read-in initial conditions produce particles from read-in 
c.....common block.
c.....note that this part is only for cascade with number of test particles
c.....NUM = 1.
      IF (IAPAR2(1) .NE. 1) THEN
         CT = NT * DT
cbz12/22/98
c         NP = MASSR(1)
c         DO WHILE (FTAR(NPI) .GT. CT - DT .AND. FTAR(NPI) .LE. CT)
c            NP = NP + 1
c            R(1, NP) = GXAR(NPI) + PXAR(NPI) / PEAR(NPI) * (CT - FTAR(NPI))
c            R(2, NP) = GYAR(NPI) + PYAR(NPI) / PEAR(NPI) * (CT - FTAR(NPI))
c            R(3, NP) = GZAR(NPI) + PZAR(NPI) / PEAR(NPI) * (CT - FTAR(NPI))
c            P(1, NP) = PXAR(NPI)
c            P(2, NP) = PYAR(NPI)
c            P(3, NP) = PZAR(NPI)
c            E(NP) = XMAR(NPI)
c            LB(NP) = IARFLV(ITYPAR(NPI))
c            NPI = NPI + 1
c         END DO
c         MASSR(1) = NP
         IA = 0
         DO 1028 IRUN = 1, NUM
            DO 1027 IC = 1, MASSR(IRUN)
               IE = IA + IC
               RT(1, IC, IRUN) = R(1, IE)
               RT(2, IC, IRUN) = R(2, IE)
               RT(3, IC, IRUN) = R(3, IE)
               PT(1, IC, IRUN) = P(1, IE)
               PT(2, IC, IRUN) = P(2, IE)
               PT(3, IC, IRUN) = P(3, IE)
               ET(IC, IRUN) = E(IE)
               LT(IC, IRUN) = LB(IE)
c         !! sp 12/19/00
               PROT(IC, IRUN) = PROPER(IE)
clin-5/2008:
               dpertt(IC, IRUN)=dpertp(IE)
 1027       CONTINUE
            NP = MASSR(IRUN)
            NP1 = NPI(IRUN)

cbz10/05/99
c            DO WHILE (FT1(NP1, IRUN) .GT. CT - DT .AND. 
c     &           FT1(NP1, IRUN) .LE. CT)
cbz10/06/99
c            DO WHILE (NPI(IRUN).LE.MULTI1(IRUN).AND.
cbz10/06/99 end
clin-11/13/00 finally read in all unformed particles and do the decays in ART:
c           DO WHILE (NP1.LE.MULTI1(IRUN).AND.
c    &           FT1(NP1, IRUN) .GT. CT - DT .AND. 
c    &           FT1(NP1, IRUN) .LE. CT)
c
               ctlong = ct
             if(nt .eq. (ntmax-1))then
               ctlong = 1.E30
             elseif(nt .eq. ntmax)then
               go to 1111
             endif
            DO WHILE (NP1.LE.MULTI1(IRUN).AND.
     &           FT1(NP1, IRUN) .GT. (CT - DT) .AND. 
     &           FT1(NP1, IRUN) .LE. ctlong)
               NP = NP + 1
               UDT = (CT - FT1(NP1, IRUN)) / EE1(NP1, IRUN)
clin-10/28/03 since all unformed hadrons at time ct are read in at nt=ntmax-1, 
c     their positions should not be propagated to time ct:
               if(nt.eq.(ntmax-1)) then
                  ftsvt(NP,IRUN)=FT1(NP1, IRUN)
                  if(FT1(NP1, IRUN).gt.ct) UDT=0.
               endif
               RT(1, NP, IRUN) = GX1(NP1, IRUN) + 
     &              PX1(NP1, IRUN) * UDT
               RT(2, NP, IRUN) = GY1(NP1, IRUN) + 
     &              PY1(NP1, IRUN) * UDT
               RT(3, NP, IRUN) = GZ1(NP1, IRUN) + 
     &              PZ1(NP1, IRUN) * UDT
               PT(1, NP, IRUN) = PX1(NP1, IRUN)
               PT(2, NP, IRUN) = PY1(NP1, IRUN)
               PT(3, NP, IRUN) = PZ1(NP1, IRUN)
               ET(NP, IRUN) = XM1(NP1, IRUN)
               LT(NP, IRUN) = IARFLV(ITYP1(NP1, IRUN))
clin-5/2008:
               dpertt(NP,IRUN)=dpp1(NP1,IRUN)
clin-4/30/03 ctest off 
c     record initial phi,K*,Lambda(1520) resonances formed during the timestep:
c               if(LT(NP, IRUN).eq.29.or.iabs(LT(NP, IRUN)).eq.30)
c     1              write(17,112) 'formed',LT(NP, IRUN),PX1(NP1, IRUN),
c     2 PY1(NP1, IRUN),PZ1(NP1, IRUN),XM1(NP1, IRUN),nt
c 112           format(a10,1x,I4,4(1x,f9.3),1x,I4)
c
               NP1 = NP1 + 1
c     !! sp 12/19/00
               PROT(NP, IRUN) = 1.
            END DO
*
 1111      continue
            NPI(IRUN) = NP1
            IA = IA + MASSR(IRUN)
            MASSR(IRUN) = NP
 1028    CONTINUE
         IA = 0
         DO 1030 IRUN = 1, NUM
            IA = IA + MASSR(IRUN - 1)
            DO 1029 IC = 1, MASSR(IRUN)
               IE = IA + IC
               R(1, IE) = RT(1, IC, IRUN)
               R(2, IE) = RT(2, IC, IRUN)
               R(3, IE) = RT(3, IC, IRUN)
               P(1, IE) = PT(1, IC, IRUN)
               P(2, IE) = PT(2, IC, IRUN)
               P(3, IE) = PT(3, IC, IRUN)
               E(IE) = ET(IC, IRUN)
               LB(IE) = LT(IC, IRUN)
c     !! sp 12/19/00
               PROPER(IE) = PROT(IC, IRUN)
               if(nt.eq.(ntmax-1)) ftsv(IE)=ftsvt(IC,IRUN)
clin-5/2008:
               dpertp(IE)=dpertt(IC, IRUN)
 1029       CONTINUE
clin-3/2009 Moved here to better take care of freezeout spacetime:
            call hbtout(MASSR(IRUN),nt,ntmax)
 1030    CONTINUE
cbz12/22/98end
      END IF
cbz11/16/98end

clin-5/2009 ctest off:
c      call flowh(ct) 

10000       continue

*                                                                      *
*       ==============  END OF TIME STEP LOOP   ================       *

************************************
*     WRITE OUT particle's MOMENTA ,and/OR COORDINATES ,
*     label and/or their local baryon density in the final state
        iss=0
        do 1032 lrun=1,num
           iss=iss+massr(lrun-1)
           do 1031 l0=1,massr(lrun)
              ipart=iss+l0
 1031      continue
 1032   continue

cbz11/16/98
      IF (IAPAR2(1) .NE. 1) THEN
cbz12/22/98
c        NSH = MASSR(1) - NPI + 1
c        IAINT2(1) = IAINT2(1) + NSH
c.....to shift the unformed particles to the end of the common block
c        IF (NSH .GT. 0) THEN
c           IB = IAINT2(1)
c           IE = MASSR(1) + 1
c           II = -1
c        ELSE IF (NSH .LT. 0) THEN
c           IB = MASSR(1) + 1
c           IE = IAINT2(1)
c           II = 1
c        END IF
c        IF (NSH .NE. 0) THEN
c           DO I = IB, IE, II
c              J = I - NSH
c              ITYPAR(I) = ITYPAR(J)
c              GXAR(I) = GXAR(J)
c              GYAR(I) = GYAR(J)
c              GZAR(I) = GZAR(J)
c              FTAR(I) = FTAR(J)
c              PXAR(I) = PXAR(J)
c              PYAR(I) = PYAR(J)
c              PZAR(I) = PZAR(J)
c              PEAR(I) = PEAR(J)
c              XMAR(I) = XMAR(J)
c           END DO
c        END IF

c.....to copy ART particle info to COMMON /ARPRC/
c        DO I = 1, MASSR(1)
c           ITYPAR(I) = INVFLV(LB(I))
c           GXAR(I) = R(1, I)
c           GYAR(I) = R(2, I)
c           GZAR(I) = R(3, I)
c           FTAR(I) = CT
c           PXAR(I) = P(1, I)
c           PYAR(I) = P(2, I)
c           PZAR(I) = P(3, I)
c           XMAR(I) = E(I)
c           PEAR(I) = SQRT(PXAR(I) ** 2 + PYAR(I) ** 2 + PZAR(I) ** 2
c     &        + XMAR(I) ** 2)
c        END DO
        IA = 0
        DO 1035 IRUN = 1, NUM
           IA = IA + MASSR(IRUN - 1)
           NP1 = NPI(IRUN)
           NSH = MASSR(IRUN) - NP1 + 1
           MULTI1(IRUN) = MULTI1(IRUN) + NSH
c.....to shift the unformed particles to the end of the common block
           IF (NSH .GT. 0) THEN
              IB = MULTI1(IRUN)
              IE = MASSR(IRUN) + 1
              II = -1
           ELSE IF (NSH .LT. 0) THEN
              IB = MASSR(IRUN) + 1
              IE = MULTI1(IRUN)
              II = 1
           END IF
           IF (NSH .NE. 0) THEN
              DO 1033 I = IB, IE, II
                 J = I - NSH
                 ITYP1(I, IRUN) = ITYP1(J, IRUN)
                 GX1(I, IRUN) = GX1(J, IRUN)
                 GY1(I, IRUN) = GY1(J, IRUN)
                 GZ1(I, IRUN) = GZ1(J, IRUN)
                 FT1(I, IRUN) = FT1(J, IRUN)
                 PX1(I, IRUN) = PX1(J, IRUN)
                 PY1(I, IRUN) = PY1(J, IRUN)
                 PZ1(I, IRUN) = PZ1(J, IRUN)
                 EE1(I, IRUN) = EE1(J, IRUN)
                 XM1(I, IRUN) = XM1(J, IRUN)
c     !! sp 12/19/00
                 PRO1(I, IRUN) = PRO1(J, IRUN)
clin-5/2008:
                 dpp1(I,IRUN)=dpp1(J,IRUN)
 1033         CONTINUE
           END IF
           
c.....to copy ART particle info to COMMON /ARPRC1/
           DO 1034 I = 1, MASSR(IRUN)
              IB = IA + I
              ITYP1(I, IRUN) = INVFLV(LB(IB))
              GX1(I, IRUN) = R(1, IB)
              GY1(I, IRUN) = R(2, IB)
              GZ1(I, IRUN) = R(3, IB)
clin-10/28/03:
c since all unformed hadrons at time ct are read in at nt=ntmax-1, 
c their formation time ft1 should be kept to determine their freezeout(x,t):
c              FT1(I, IRUN) = CT
              if(FT1(I, IRUN).lt.CT) FT1(I, IRUN) = CT
              PX1(I, IRUN) = P(1, IB)
              PY1(I, IRUN) = P(2, IB)
              PZ1(I, IRUN) = P(3, IB)
              XM1(I, IRUN) = E(IB)
              EE1(I, IRUN) = SQRT(PX1(I, IRUN) ** 2 + 
     &             PY1(I, IRUN) ** 2 +
     &             PZ1(I, IRUN) ** 2 + 
     &             XM1(I, IRUN) ** 2)
c     !! sp 12/19/00
              PRO1(I, IRUN) = PROPER(IB)
 1034      CONTINUE
 1035   CONTINUE
cbz12/22/98end
      END IF
cbz11/16/98end
c
**********************************
*                                                                      *
*       ======= END OF MANY LOOPS OVER IMPACT PARAMETERS ==========    *
*                                                               *
**********************************
50000   CONTINUE
*
*-----------------------------------------------------------------------
*                       ==== ART COMPLETED ====
*-----------------------------------------------------------------------
cbz11/16/98
c      STOP
      RETURN
cbz11/16/98end
      END
**********************************
      subroutine coulin(masspr,massta,NUM)
*                                                                      *
*     purpose:   initialization of array zet() and lb() for all runs  *
*                lb(i) = 1   =>  proton                               *
*                lb(i) = 2   =>  neutron                              *
**********************************
        integer  zta,zpr
        PARAMETER (MAXSTR=150001)
        common  /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        COMMON  /ZZ/ ZTA,ZPR
cc      SAVE /zz/
      SAVE   
        MASS=MASSTA+MASSPR
        DO 500 IRUN=1,NUM
        do 100 i = 1+(IRUN-1)*MASS,zta+(IRUN-1)*MASS
        LB(i) = 1
  100   continue
        do 200 i = zta+1+(IRUN-1)*MASS,massta+(IRUN-1)*MASS
        LB(i) = 2
  200   continue
        do 300 i = massta+1+(IRUN-1)*MASS,massta+zpr+(IRUN-1)*MASS
        LB(i) = 1
  300   continue
        do 400 i = massta+zpr+1+(IRUN-1)*MASS,
     1  massta+masspr+(IRUN-1)*MASS
        LB(i) = 2
  400   continue
  500   CONTINUE
        return
        end
**********************************
*                                                                      *
      SUBROUTINE RELCOL(LCOLL,LBLOC,LCNNE,LDD,LPP,lppk,
     &LPN,lpd,lrho,lomega,LKN,LNNK,LDDK,LNDK,LCNND,LCNDN,
     &LDIRT,LDECAY,LRES,LDOU,LDDRHO,LNNRHO,LNNOM,
     &NT,ntmax,sp,akaon,sk)
*                                                                      *
*       PURPOSE:    CHECK CONDITIONS AND CALCULATE THE KINEMATICS      * 
*                   FOR BINARY COLLISIONS AMONG PARTICLES              *
*                                 - RELATIVISTIC FORMULA USED          *
*                                                                      *
*       REFERENCES: HAGEDORN, RELATIVISTIC KINEMATICS (1963)           *
*                                                                      *
*       VARIABLES:                                                     *
*         MASSPR  - NUMBER OF NUCLEONS IN PROJECTILE   (INTEGER,INPUT) *
*         MASSTA  - NUMBER OF NUCLEONS IN TARGET       (INTEGER,INPUT) *
*         NUM     - NUMBER OF TESTPARTICLES PER NUCLEON(INTEGER,INPUT) *
*         ISEED   - SEED FOR RANDOM NUMBER GENERATOR   (INTEGER,INPUT) *
*         IAVOID  - (= 1 => AVOID FIRST CLLISIONS WITHIN THE SAME      *
*                   NUCLEUS, ELSE ALL COLLISIONS)      (INTEGER,INPUT) *
*         DELTAR  - MAXIMUM SPATIAL DISTANCE FOR WHICH A COLLISION     *
*                   STILL CAN OCCUR                       (REAL,INPUT) *
*         DT      - TIME STEP SIZE                        (REAL,INPUT) *
*         LCOLL   - NUMBER OF COLLISIONS              (INTEGER,OUTPUT) *
*         LBLOC   - NUMBER OF PULI-BLOCKED COLLISIONS (INTEGER,OUTPUT) *
*         LCNNE   - NUMBER OF ELASTIC COLLISION       (INTEGER,OUTPUT) *
*         LCNND   - NUMBER OF N+N->N+DELTA REACTION   (INTEGER,OUTPUT) *
*         LCNDN   - NUMBER OF N+DELTA->N+N REACTION   (INTEGER,OUTPUT) *
*         LDD     - NUMBER OF RESONANCE+RESONANCE COLLISIONS
*         LPP     - NUMBER OF PION+PION elastic COLIISIONS
*         lppk    - number of pion(RHO,OMEGA)+pion(RHO,OMEGA)
*                   -->K+K- collisions
*         LPN     - NUMBER OF PION+N-->KAON+X
*         lpd     - number of pion+n-->delta+pion
*         lrho    - number of pion+n-->Delta+rho
*         lomega  - number of pion+n-->Delta+omega
*         LKN     - NUMBER OF KAON RESCATTERINGS
*         LNNK    - NUMBER OF bb-->kAON PROCESS
*         LDDK    - NUMBER OF DD-->KAON PROCESS
*         LNDK    - NUMBER OF ND-->KAON PROCESS
*         LB(I) IS USED TO LABEL PARTICLE'S CHARGE STATE
*         LB(I)   = 
cbali2/7/99 
*                 -45 Omega baryon(bar)
*                 -41 cascade0(bar)
*                 -40 cascade-(bar)
clin-11/07/00:
*                 -30 K*-
*                 -17 sigma+(bar)
*                 -16 sigma0(bar)
*                 -15 sigma-(bar)
*                 -14 LAMBDA(bar)
clin-8/29/00
*                 -13 anti-N*(+1)(1535),s_11
*                 -12 anti-N*0(1535),s_11
*                 -11 anti-N*(+1)(1440),p_11
*                 -10 anti-N*0(1440), p_11
*                  -9 anti-DELTA+2
*                  -8 anti-DELTA+1
*                  -7 anti-DELTA0
*                  -6 anti-DELTA-1
*
*                  -2 antineutron 
*                  -1 antiproton
cbali2/7/99end 
*                   0 eta
*                   1 PROTON
*                   2 NUETRON
*                   3 PION-
*                   4 PION0
*                   5 PION+          
*                   6 DELTA-1
*                   7 DELTA0
*                   8 DELTA+1
*                   9 DELTA+2
*                   10 N*0(1440), p_11
*                   11 N*(+1)(1440),p_11
*                  12 N*0(1535),s_11
*                  13 N*(+1)(1535),s_11
*                  14 LAMBDA
*                   15 sigma-
*                   16 sigma0
*                   17 sigma+
*                   21 kaon-
clin-2/23/03        22 Kaon0Long (converted at the last timestep)
*                   23 KAON+
*                   24 Kaon0short (converted at the last timestep then decay)
*                   25 rho-
*                   26 rho0
*                   27 rho+
*                   28 omega meson
*                   29 phi
*                   30 K*+
* sp01/03/01
*                   31 eta-prime
*                   40 cascade-
*                   41 cascade0
*                   45 Omega baryon
* sp01/03/01 end
*                   
*                   ++  ------- SEE NOTE BOOK
*         NSTAR=1 INCLUDING N* RESORANCE
*         ELSE DELTA RESORANCE ONLY
*         NDIRCT=1 INCLUDING DIRECT PROCESS,ELSE NOT
*         DIR - PERCENTAGE OF DIRECT PION PRODUCTION PROCESS
**********************************
      PARAMETER      (MAXSTR=150001,MAXR=1,PI=3.1415926)
      parameter      (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
      PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974,aks=0.895)
      PARAMETER      (AA1=1.26,APHI=1.02,AP1=0.13496)
      parameter            (maxx=20,maxz=24)
      parameter            (rrkk=0.6,prkk=0.3,srhoks=5.,ESBIN=0.04)
      DIMENSION MASSRN(0:MAXR),RT(3,MAXSTR),PT(3,MAXSTR),ET(MAXSTR)
      DIMENSION LT(MAXSTR), PROT(MAXSTR)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /HH/  PROPER(MAXSTR)
cc      SAVE /HH/
      common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
      common   /gg/  dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
      COMMON   /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /RR/  MASSR(0:MAXR)
cc      SAVE /RR/
      common   /ss/  inout(20)
cc      SAVE /ss/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
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
      COMMON   /PE/PROPI(MAXSTR,MAXR)
cc      SAVE /PE/
      COMMON   /KKK/TKAON(7),EKAON(7,0:2000)
cc      SAVE /KKK/
      COMMON  /KAON/    AK(3,50,36),SPECK(50,36,7),MF
cc      SAVE /KAON/
      COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1 px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
      COMMON/tdecay/tfdcy(MAXSTR),tfdpi(MAXSTR,MAXR),tft(MAXSTR)
cc      SAVE /tdecay/
      common /lastt/itimeh,bimp 
cc      SAVE /lastt/
c
      COMMON/ppbmas/niso(15),nstate,ppbm(15,2),thresh(15),weight(15)
cc      SAVE /ppbmas/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/hbt/lblast(MAXSTR),xlast(4,MAXSTR),plast(4,MAXSTR),nlast
cc      SAVE /hbt/
      common/resdcy/NSAV,iksdcy
cc      SAVE /resdcy/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON/FTMAX/ftsv(MAXSTR),ftsvt(MAXSTR, MAXR)
      dimension ftpisv(MAXSTR,MAXR),fttemp(MAXSTR)
      common /dpi/em2,lb2
      common/phidcy/iphidcy,pttrig,ntrig,maxmiss
clin-5/2008:
      DIMENSION dptemp(MAXSTR)
      common /para8/ idpert,npertd,idxsec
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
c
      real zet(-45:45)
      SAVE   
      data zet /
     4     1.,0.,0.,0.,0.,
     3     1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
     2     -1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
     1     0.,0.,0.,-1.,0.,1.,0.,-1.,0.,-1.,
     s     0.,-2.,-1.,0.,1.,0.,0.,0.,0.,-1.,
     e     0.,
     s     1.,0.,-1.,0.,1.,-1.,0.,1.,2.,0.,
     1     1.,0.,1.,0.,-1.,0.,1.,0.,0.,0.,
     2     -1.,0.,1.,0.,-1.,0.,1.,0.,0.,1.,
     3     0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,
     4     0.,0.,0.,0.,-1./

clin-2/19/03 initialize n and nsav for resonance decay at each timestep
c     in order to prevent integer overflow:
      call inidcy

c OFF skip ART collisions to reproduce HJ:      
cc       if(nt.ne.ntmax) return

clin-11/07/00 rrkk is assumed to be 0.6mb(default) for mm->KKbar 
c     with m=rho or omega, estimated from Ko's paper:
c      rrkk=0.6
c prkk: cross section of pi (rho or omega) -> K* Kbar (AND) K*bar K:
c      prkk=0.3
c     cross section in mb for (rho or omega) K* -> pi K:
c      srhoks=5.
clin-11/07/00-end
c      ESBIN=0.04
      RESONA=5.
*-----------------------------------------------------------------------
*     INITIALIZATION OF COUNTING VARIABLES
      NODELT=0
      SUMSRT =0.
      LCOLL  = 0
      LBLOC  = 0
      LCNNE  = 0
      LDD  = 0
      LPP  = 0
      lpd  = 0
      lpdr=0
      lrho = 0
      lrhor=0
      lomega=0
      lomgar=0
      LPN  = 0
      LKN  = 0
      LNNK = 0
      LDDK = 0
      LNDK = 0
      lppk =0
      LCNND  = 0
      LCNDN  = 0
      LDIRT  = 0
      LDECAY = 0
      LRES   = 0
      Ldou   = 0
      LDDRHO = 0
      LNNRHO = 0
      LNNOM  = 0
      MSUM   = 0
      MASSRN(0)=0
* COM: MSUM IS USED TO COUNT THE TOTAL NO. OF PARTICLES 
*      IN PREVIOUS IRUN-1 RUNS
* KAON COUNTERS
      DO 1002 IL=1,5
         TKAON(IL)=0
         DO 1001 IS=1,2000
            EKAON(IL,IS)=0
 1001    CONTINUE
 1002 CONTINUE
c sp 12/19/00
      DO 1004 i =1,NUM
         DO 1003 j =1,MAXSTR
            PROPI(j,i) = 1.
 1003    CONTINUE
 1004 CONTINUE
      
      do 1102 i=1,maxstr
         fttemp(i)=0.
         do 1101 irun=1,maxr
            ftpisv(i,irun)=0.
 1101    continue
 1102 continue

c sp 12/19/00 end
      sp=0
* antikaon counters
      akaon=0
      sk=0
*-----------------------------------------------------------------------
*     LOOP OVER ALL PARALLEL RUNS
cbz11/17/98
c      MASS=MASSPR+MASSTA
      MASS = 0
cbz11/17/98end
      DO 1000 IRUN = 1,NUM
         NNN=0
         MSUM=MSUM+MASSR(IRUN-1)
*     LOOP OVER ALL PSEUDOPARTICLES 1 IN THE SAME RUN
         J10=2
         IF(NT.EQ.NTMAX)J10=1
c
ctest off skips the check of energy conservation after each timestep:
c         enetot=0.
c         do ip=1,MASSR(IRUN)
c            if(e(ip).ne.0.or.lb(ip).eq.10022) enetot=enetot
c     1           +sqrt(p(1,ip)**2+p(2,ip)**2+p(3,ip)**2+e(ip)**2)
c         enddo
c         write(91,*) 'A:',nt,enetot,massr(irun),bimp 

         DO 800 J1 = J10,MASSR(IRUN)
            I1  = J1 + MSUM
* E(I)=0 are for pions having been absorbed or photons which do not enter here:
            IF(E(I1).EQ.0.)GO TO 800

c     To include anti-(Delta,N*1440 and N*1535):
c          IF ((LB(I1) .LT. -13 .OR. LB(I1) .GT. 28)
c     1         .and.iabs(LB(I1)) .ne. 30 ) GOTO 800
            IF (LB(I1) .LT. -45 .OR. LB(I1) .GT. 45) GOTO 800
            X1  = R(1,I1)
            Y1  = R(2,I1)
            Z1  = R(3,I1)
            PX1 = P(1,I1)
            PY1 = P(2,I1)
            PZ1 = P(3,I1)
            EM1 = E(I1)
            am1= em1
            E1  = SQRT( EM1**2 + PX1**2 + PY1**2 + PZ1**2 )
            ID1 = ID(I1)
            LB1 = LB(I1)

c     generate k0short and k0long from K+ and K- at the last timestep:
            if(nt.eq.ntmax.and.(lb1.eq.21.or.lb1.eq.23)) then
               pk0=RANART(NSEED)
               if(pk0.lt.0.25) then
                  LB(I1)=22
               elseif(pk0.lt.0.50) then
                  LB(I1)=24
               endif
               LB1=LB(I1)
            endif
            
clin-8/07/02 these particles don't decay strongly, so skip decay routines:     
c            IF( (lb1.ge.-2.and.lb1.le.5) .OR. lb1.eq.31 .OR.
c     &           (iabs(lb1).ge.14.and.iabs(lb1).le.24) .OR.
c     &           (iabs(lb1).ge.40.and.iabs(lb1).le.45) .or. 
c     &           lb1.eq.31)GO TO 1 
c     only decay K0short when iksdcy=1:
            if(lb1.eq.0.or.lb1.eq.25.or.lb1.eq.26.or.lb1.eq.27
     &           .or.lb1.eq.28.or.lb1.eq.29.or.iabs(lb1).eq.30
     &           .or.(iabs(lb1).ge.6.and.iabs(lb1).le.13)
     &           .or.(iksdcy.eq.1.and.lb1.eq.24)
     &           .or.iabs(lb1).eq.16) then
               continue
            else
               goto 1
            endif
* IF I1 IS A RESONANCE, CHECK WHETHER IT DECAYS DURING THIS TIME STEP
         IF(lb1.ge.25.and.lb1.le.27) then
             wid=0.151
         ELSEIF(lb1.eq.28) then
             wid=0.00841
         ELSEIF(lb1.eq.29) then
             wid=0.00443
          ELSEIF(iabs(LB1).eq.30) then
             WID=0.051
         ELSEIF(lb1.eq.0) then
             wid=1.18e-6
c     to give K0short ct0=2.676cm:
         ELSEIF(iksdcy.eq.1.and.lb1.eq.24) then
             wid=7.36e-15
clin-4/29/03 add Sigma0 decay to Lambda, ct0=2.22E-11m:
         ELSEIF(iabs(lb1).eq.16) then
             wid=8.87e-6
csp-07/25/01 test a1 resonance:
cc          ELSEIF(LB1.EQ.32) then
cc             WID=0.40
          ELSEIF(LB1.EQ.32) then
             call WIDA1(EM1,rhomp,WID,iseed)
          ELSEIF(iabs(LB1).ge.6.and.iabs(LB1).le.9) then
             WID=WIDTH(EM1)
          ELSEIF((iabs(LB1).EQ.10).OR.(iabs(LB1).EQ.11)) then
             WID=W1440(EM1)
          ELSEIF((iabs(LB1).EQ.12).OR.(iabs(LB1).EQ.13)) then
             WID=W1535(EM1)
          ENDIF

* if it is the last time step, FORCE all resonance to strong-decay
* and go out of the loop
          if(nt.eq.ntmax)then
             pdecay=1.1
clin-5b/2008 forbid phi decay at the end of hadronic cascade:
             if(iphidcy.eq.0.and.iabs(LB1).eq.29) pdecay=0.
          else
             T0=0.19733/WID
             GFACTR=E1/EM1
             T0=T0*GFACTR
             IF(T0.GT.0.)THEN
                PDECAY=1.-EXP(-DT/T0)
             ELSE
                PDECAY=0.
             ENDIF
          endif
          XDECAY=RANART(NSEED)

cc dilepton production from rho0, omega, phi decay 
cc        if(lb1.eq.26 .or. lb1.eq.28 .or. lb1.eq.29)
cc     &   call dec_ceres(nt,ntmax,irun,i1)
cc
          IF(XDECAY.LT.PDECAY) THEN
clin-10/25/02 get rid of argument usage mismatch in rhocay():
             idecay=irun
             tfnl=nt*dt
clin-10/28/03 keep formation time of hadrons unformed at nt=ntmax-1:
             if(nt.eq.ntmax.and.ftsv(i1).gt.((ntmax-1)*dt)) 
     1            tfnl=ftsv(i1)
             xfnl=x1
             yfnl=y1
             zfnl=z1
* use PYTHIA to perform decays of eta,rho,omega,phi,K*,(K0s) and Delta:
             if(lb1.eq.0.or.lb1.eq.25.or.lb1.eq.26.or.lb1.eq.27
     &           .or.lb1.eq.28.or.lb1.eq.29.or.iabs(lb1).eq.30
     &           .or.(iabs(lb1).ge.6.and.iabs(lb1).le.9)
     &           .or.(iksdcy.eq.1.and.lb1.eq.24)
     &           .or.iabs(lb1).eq.16) then
c     previous rho decay performed in rhodecay():
c                nnn=nnn+1
c                call rhodecay(idecay,i1,nnn,iseed)
c
ctest off record decays of phi,K*,Lambda(1520) resonances:
c                if(lb1.eq.29.or.iabs(lb1).eq.30) 
c     1               write(18,112) 'decay',lb1,px1,py1,pz1,am1,nt
                call resdec(i1,nt,nnn,wid,idecay)
                p(1,i1)=px1n
                p(2,i1)=py1n
                p(3,i1)=pz1n
clin-5/2008:
                dpertp(i1)=dp1n
c     add decay time to freezeout positions & time at the last timestep:
                if(nt.eq.ntmax) then
                   R(1,i1)=xfnl
                   R(2,i1)=yfnl
                   R(3,i1)=zfnl
                   tfdcy(i1)=tfnl
                endif
c
* decay number for baryon resonance or L/S decay
                if(iabs(lb1).ge.6.and.iabs(lb1).le.9) then
                   LDECAY=LDECAY+1
                endif

* for a1 decay 
c             elseif(lb1.eq.32)then
c                NNN=NNN+1
c                call a1decay(idecay,i1,nnn,iseed,rhomp)

* FOR N*(1440)
             elseif(iabs(LB1).EQ.10.OR.iabs(LB1).EQ.11) THEN
                NNN=NNN+1
                LDECAY=LDECAY+1
                PNSTAR=1.
                IF(E(I1).GT.1.22)PNSTAR=0.6
                IF(RANART(NSEED).LE.PNSTAR)THEN
* (1) DECAY TO SINGLE PION+NUCLEON
                   CALL DECAY(idecay,I1,NNN,ISEED,wid,nt)
                ELSE
* (2) DECAY TO TWO PIONS + NUCLEON
                   CALL DECAY2(idecay,I1,NNN,ISEED,wid,nt)
                   NNN=NNN+1
                ENDIF
c for N*(1535) decay
             elseif(iabs(LB1).eq.12.or.iabs(LB1).eq.13) then
                NNN=NNN+1
                CALL DECAY(idecay,I1,NNN,ISEED,wid,nt)
                LDECAY=LDECAY+1
             endif
c
*COM: AT HIGH ENERGIES WE USE VERY SHORT TIME STEPS,
*     IN ORDER TO TAKE INTO ACCOUNT THE FINITE FORMATIOM TIME, WE
*     DO NOT ALLOW PARTICLES FROM THE DECAY OF RESONANCE TO INTERACT 
*     WITH OTHERS IN THE SAME TIME STEP. CHANGE 9000 TO REVERSE THIS 
*     ASSUMPTION. EFFECTS OF THIS ASSUMPTION CAN BE STUDIED BY CHANGING 
*     THE STATEMENT OF 9000. See notebook for discussions on effects of
*     changing statement 9000.
c
c     kaons from K* decay are converted to k0short (and k0long), 
c     phi decay may produce rho, K0S or eta, N*(1535) decay may produce eta,
c     and these decay daughters need to decay again if at the last timestep:
c     (note: these daughters have been assigned to lb(i1) only, not to lpion)
c             if(nt.eq.ntmax.and.(lb1.eq.29.or.iabs(lb1).eq.30
c     1            .iabs(lb1).eq.12.or.iabs(lb1).eq.13)) then
             if(nt.eq.ntmax) then
                if(lb(i1).eq.25.or.lb(i1).eq.26.or.lb(i1).eq.27) then
                   wid=0.151
                elseif(lb(i1).eq.0) then
                   wid=1.18e-6
                elseif(lb(i1).eq.24.and.iksdcy.eq.1) then
                   wid=7.36e-17
                else
                   goto 9000
                endif
                LB1=LB(I1)
                PX1=P(1,I1)
                PY1=P(2,I1)
                PZ1=P(3,I1)
                EM1=E(I1)
                E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
                call resdec(i1,nt,nnn,wid,idecay)
                p(1,i1)=px1n
                p(2,i1)=py1n
                p(3,i1)=pz1n
                R(1,i1)=xfnl
                R(2,i1)=yfnl
                R(3,i1)=zfnl
                tfdcy(i1)=tfnl
clin-5/2008:
                dpertp(i1)=dp1n
             endif

* negelecting the Pauli blocking at high energies
 9000        go to 800
          ENDIF
* LOOP OVER ALL PSEUDOPARTICLES 2 IN THE SAME RUN
* SAVE ALL THE COORDINATES FOR POSSIBLE CHANGE IN THE FOLLOWING COLLISION
 1        if(nt.eq.ntmax)go to 800
          X1 = R(1,I1)
          Y1 = R(2,I1)
          Z1 = R(3,I1)
c
           DO 600 J2 = 1,J1-1
            I2  = J2 + MSUM
* IF I2 IS A MESON BEING ABSORBED, THEN GO OUT OF THE LOOP
            IF(E(I2).EQ.0.) GO TO 600
clin-5/2008 in case the first particle is already destroyed:
            IF(E(I1).EQ.0.) GO TO 800
            IF (LB(I2) .LT. -45 .OR. LB(I2) .GT. 45) GOTO 600
clin-7/26/03 improve speed
            X2=R(1,I2)
            Y2=R(2,I2)
            Z2=R(3,I2)
            dr0max=5.
clin-9/2008 deuteron+nucleon elastic cross sections could reach ~2810mb:
            ilb1=iabs(LB(I1))
            ilb2=iabs(LB(I2))
            IF(ilb1.EQ.42.or.ilb2.EQ.42) THEN
               if((ILB1.GE.1.AND.ILB1.LE.2)
     1              .or.(ILB1.GE.6.AND.ILB1.LE.13)
     2              .or.(ILB2.GE.1.AND.ILB2.LE.2)
     3              .or.(ILB2.GE.6.AND.ILB2.LE.13)) then
                  if((lb(i1)*lb(i2)).gt.0) dr0max=10.
               endif
            ENDIF
c
            if(((X1-X2)**2+(Y1-Y2)**2+(Z1-Z2)**2).GT.dr0max**2)
     1           GO TO 600
            IF (ID(I1)*ID(I2).EQ.IAVOID) GOTO 400
            ID1=ID(I1)
            ID2 = ID(I2)
c
            ix1= nint(x1/dx)
            iy1= nint(y1/dy)
            iz1= nint(z1/dz)
            PX1=P(1,I1)
            PY1=P(2,I1)
            PZ1=P(3,I1)
            EM1=E(I1)
            AM1=EM1
            LB1=LB(I1)
            E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
            IPX1=NINT(PX1/DPX)
            IPY1=NINT(PY1/DPY)
            IPZ1=NINT(PZ1/DPZ)         
            LB2 = LB(I2)
            PX2 = P(1,I2)
            PY2 = P(2,I2)
            PZ2 = P(3,I2)
            EM2=E(I2)
            AM2=EM2
            lb1i=lb(i1)
            lb2i=lb(i2)
            px1i=P(1,I1)
            py1i=P(2,I1)
            pz1i=P(3,I1)
            em1i=E(I1)
            px2i=P(1,I2)
            py2i=P(2,I2)
            pz2i=P(3,I2)
            em2i=E(I2)
clin-2/26/03 ctest off check energy conservation after each binary search:
            eini=SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
     1           +SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
            pxini=P(1,I1)+P(1,I2)
            pyini=P(2,I1)+P(2,I2)
            pzini=P(3,I1)+P(3,I2)
            nnnini=nnn
c
clin-4/30/03 initialize value:
            iblock=0
c
* TO SAVE COMPUTING TIME we do the following
* (1) make a ROUGH estimate to see whether particle i2 will collide with
* particle I1, and (2) skip the particle pairs for which collisions are 
* not modeled in the code.
* FOR MESON-BARYON AND MESON-MESON COLLISIONS, we use a maximum 
* interaction distance DELTR0=2.6
* for ppbar production from meson (pi rho omega) interactions:
c
            DELTR0=3.
        if( (iabs(lb1).ge.14.and.iabs(lb1).le.17) .or.
     &      (iabs(lb1).ge.30.and.iabs(lb1).le.45) ) DELTR0=5.0
        if( (iabs(lb2).ge.14.and.iabs(lb2).le.17) .or.
     &      (iabs(lb2).ge.30.and.iabs(lb2).le.45) ) DELTR0=5.0

            if(lb1.eq.28.and.lb2.eq.28) DELTR0=4.84
clin-10/08/00 to include pi pi -> rho rho:
            if((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.3.and.lb2.le.5)) then
               E2=SQRT(EM2**2+PX2**2+PY2**2+PZ2**2)
         spipi=(e1+e2)**2-(px1+px2)**2-(py1+py2)**2-(pz1+pz2)**2
               if(spipi.ge.(4*0.77**2)) DELTR0=3.5
            endif

c khyperon
        IF (LB1.EQ.23 .AND. (LB2.GE.14.AND.LB2.LE.17)) GOTO 3699
        IF (LB2.EQ.23 .AND. (LB1.GE.14.AND.LB1.LE.17)) GOTO 3699

* K(K*) + Kbar(K*bar) scattering including 
*     K(K*) + Kbar(K*bar) --> phi + pi(rho,omega) and pi pi(rho,omega)
       if(lb1.eq.21.and.lb2.eq.23)go to 3699
       if(lb2.eq.21.and.lb1.eq.23)go to 3699
       if(lb1.eq.30.and.lb2.eq.21)go to 3699
       if(lb2.eq.30.and.lb1.eq.21)go to 3699
       if(lb1.eq.-30.and.lb2.eq.23)go to 3699
       if(lb2.eq.-30.and.lb1.eq.23)go to 3699
       if(lb1.eq.-30.and.lb2.eq.30)go to 3699
       if(lb2.eq.-30.and.lb1.eq.30)go to 3699
c
clin-12/15/00
c     kaon+rho(omega,eta) collisions:
      if(lb1.eq.21.or.lb1.eq.23) then
         if(lb2.eq.0.or.(lb2.ge.25.and.lb2.le.28)) then
            go to 3699
         endif
      elseif(lb2.eq.21.or.lb2.eq.23) then
         if(lb1.eq.0.or.(lb1.ge.25.and.lb1.le.28)) then
            goto 3699
         endif
      endif

clin-8/14/02 K* (pi, rho, omega, eta) collisions:
      if(iabs(lb1).eq.30 .and.
     1     (lb2.eq.0.or.(lb2.ge.25.and.lb2.le.28)
     2     .or.(lb2.ge.3.and.lb2.le.5))) then
         go to 3699
      elseif(iabs(lb2).eq.30 .and.
     1        (lb1.eq.0.or.(lb1.ge.25.and.lb1.le.28)
     2        .or.(lb1.ge.3.and.lb1.le.5))) then
         goto 3699
clin-8/14/02-end
c K*/K*-bar + baryon/antibaryon collisions:
        elseif( iabs(lb1).eq.30 .and.
     1     (iabs(lb2).eq.1.or.iabs(lb2).eq.2.or.
     2     (iabs(lb2).ge.6.and.iabs(lb2).le.13)) )then
              go to 3699
           endif
         if( iabs(lb2).eq.30 .and.
     1         (iabs(lb1).eq.1.or.iabs(lb1).eq.2.or.
     2         (iabs(lb1).ge.6.and.iabs(lb1).le.13)) )then
                go to 3699
        endif                                                              
* K^+ baryons and antibaryons:
c** K+ + B-bar  --> La(Si)-bar + pi
* K^- and antibaryons, note K^- and baryons are included in newka():
* note that we fail to satisfy charge conjugation for these cross sections:
        if((lb1.eq.23.or.lb1.eq.21).and.
     1       (iabs(lb2).eq.1.or.iabs(lb2).eq.2.or.
     2       (iabs(lb2).ge.6.and.iabs(lb2).le.13))) then
           go to 3699
        elseif((lb2.eq.23.or.lb2.eq.21).and.
     1       (iabs(lb1).eq.1.or.iabs(lb1).eq.2.or.
     2       (iabs(lb1).ge.6.and.iabs(lb1).le.13))) then
           go to 3699
        endif
*
* For anti-nucleons annihilations:
* Assumptions: 
* (1) for collisions involving a p_bar or n_bar,
* we allow only collisions between a p_bar and a baryon or a baryon 
* resonance (as well as a n_bar and a baryon or a baryon resonance),
* we skip all other reactions involving a p_bar or n_bar, 
* such as collisions between p_bar (n_bar) and mesons, 
* and collisions between two p_bar's (n_bar's). 
* (2) we introduce a new parameter rppmax: the maximum interaction 
* distance to make the quick collision check,rppmax=3.57 fm 
* corresponding to a cutoff of annihilation xsection= 400mb which is
* also used consistently in the actual annihilation xsection to be 
* used in the following as given in the subroutine xppbar(srt)
        rppmax=3.57   
* anti-baryon on baryons
        if((lb1.eq.-1.or.lb1.eq.-2.or.(lb1.ge.-13.and.lb1.le.-6))
     1 .and.(lb2.eq.1.or.lb2.eq.2.or.(lb2.ge.6.and.lb2.le.13))) then
            DELTR0 = RPPMAX
            GOTO 2699
       else if((lb2.eq.-1.or.lb2.eq.-2.or.(lb2.ge.-13.and.lb2.le.-6))
     1 .and.(lb1.eq.1.or.lb1.eq.2.or.(lb1.ge.6.and.lb1.le.13))) then
            DELTR0 = RPPMAX
            GOTO 2699
         END IF

c*  ((anti) lambda, cascade, omega  should not be rejected)
        if( (iabs(lb1).ge.14.and.iabs(lb1).le.17) .or.
     &      (iabs(lb2).ge.14.and.iabs(lb2).le.17) )go to 3699
c
clin-9/2008 maximum sigma~2810mb for deuteron+nucleon elastic collisions:
         IF (iabs(LB1).EQ.42.or.iabs(LB2).EQ.42) THEN
            ilb1=iabs(LB1)
            ilb2=iabs(LB2)
            if((ILB1.GE.1.AND.ILB1.LE.2)
     1           .or.(ILB1.GE.6.AND.ILB1.LE.13)
     2           .or.(ILB2.GE.1.AND.ILB2.LE.2)
     3           .or.(ILB2.GE.6.AND.ILB2.LE.13)) then
               if((lb1*lb2).gt.0) deltr0=9.5
            endif
         ENDIF
c
        if( (iabs(lb1).ge.40.and.iabs(lb1).le.45) .or. 
     &      (iabs(lb2).ge.40.and.iabs(lb2).le.45) )go to 3699
c
c* phi channel --> elastic + inelastic scatt.  
         IF( (lb1.eq.29 .and.((lb2.ge.1.and.lb2.le.13).or.  
     &       (lb2.ge.21.and.lb2.le.28).or.iabs(lb2).eq.30)) .OR.
     &     (lb2.eq.29 .and.((lb1.ge.1.and.lb1.le.13).or.
     &       (lb1.ge.21.and.lb1.le.28).or.iabs(lb1).eq.30)) )THEN
             DELTR0=3.0
             go to 3699
        endif
c
c  La/Si, Cas, Om (bar)-meson elastic colln
* pion vs. La & Ca (bar) coll. are treated in resp. subroutines

* SKIP all other K* RESCATTERINGS
        If(iabs(lb1).eq.30.or.iabs(lb2).eq.30) go to 400
* SKIP KAON(+) RESCATTERINGS WITH particles other than pions and baryons 
         If(lb1.eq.23.and.(lb2.lt.1.or.lb2.gt.17))go to 400
         If(lb2.eq.23.and.(lb1.lt.1.or.lb1.gt.17))go to 400
c
c anti-baryon proccess: B-bar+M, N-bar+R-bar, N-bar+N-bar, R-bar+R-bar
c  R = (D,N*)
         if( ((lb1.le.-1.and.lb1.ge.-13)
     &        .and.(lb2.eq.0.or.(lb2.ge.3.and.lb2.le.5)
     &            .or.(lb2.ge.25.and.lb2.le.28))) 
     &      .OR.((lb2.le.-1.and.lb2.ge.-13)
     &         .and.(lb1.eq.0.or.(lb1.ge.3.and.lb1.le.5)
     &              .or.(lb1.ge.25.and.lb1.le.28))) ) then
         elseIF( ((LB1.eq.-1.or.lb1.eq.-2).
     &             and.(LB2.LT.-5.and.lb2.ge.-13))
     &      .OR. ((LB2.eq.-1.or.lb2.eq.-2).
     &             and.(LB1.LT.-5.and.lb1.ge.-13)) )then
         elseIF((LB1.eq.-1.or.lb1.eq.-2)
     &     .AND.(LB2.eq.-1.or.lb2.eq.-2))then
         elseIF((LB1.LT.-5.and.lb1.ge.-13).AND.
     &          (LB2.LT.-5.and.lb2.ge.-13)) then
c        elseif((lb1.lt.0).or.(lb2.lt.0)) then
c         go to 400
       endif               

 2699    CONTINUE
* for baryon-baryon collisions
         IF (LB1 .EQ. 1 .OR. LB1 .EQ. 2 .OR. (LB1 .GE. 6 .AND.
     &        LB1 .LE. 17)) THEN
            IF (LB2 .EQ. 1 .OR. LB2 .EQ. 2 .OR. (LB2 .GE. 6 .AND.
     &           LB2 .LE. 17)) THEN
               DELTR0 = 2.
            END IF
         END IF
c
 3699   RSQARE = (X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2
        IF (RSQARE .GT. DELTR0**2) GO TO 400
*NOW PARTICLES ARE CLOSE ENOUGH TO EACH OTHER !
* KEEP ALL COORDINATES FOR POSSIBLE PHASE SPACE CHANGE
            ix2 = nint(x2/dx)
            iy2 = nint(y2/dy)
            iz2 = nint(z2/dz)
            ipx2 = nint(px2/dpx)
            ipy2 = nint(py2/dpy)
            ipz2 = nint(pz2/dpz)
* FIND MOMENTA OF PARTICLES IN THE CMS OF THE TWO COLLIDING PARTICLES
* AND THE CMS ENERGY SRT
          CALL CMS(I1,I2,PCX,PCY,PCZ,SRT)
clin-7/26/03 improve speed
          drmax=dr0max
          call distc0(drmax,deltr0,DT,
     1         Ifirst,PCX,PCY,PCZ,
     2         x1,y1,z1,px1,py1,pz1,em1,x2,y2,z2,px2,py2,pz2,em2)
          if(Ifirst.eq.-1) goto 400

         ISS=NINT(SRT/ESBIN)
clin-4/2008 use last bin if ISS is out of EKAON's upper bound of 2000:
         if(ISS.gt.2000) ISS=2000
*Sort collisions
c
clin-8/2008 Deuteron+Meson->B+B; 
c     meson=(pi,rho,omega,eta), B=(n,p,Delta,N*1440,N*1535):
         IF (iabs(LB1).EQ.42.or.iabs(LB2).EQ.42) THEN
            ilb1=iabs(LB1)
            ilb2=iabs(LB2)
            if(LB1.eq.0.or.(LB1.GE.3.AND.LB1.LE.5)
     1           .or.(LB1.GE.25.AND.LB1.LE.28)
     2           .or.
     3           LB2.eq.0.or.(LB2.GE.3.AND.LB2.LE.5)
     4           .or.(LB2.GE.25.AND.LB2.LE.28)) then
               GOTO 505
clin-9/2008 Deuteron+Baryon or antiDeuteron+antiBaryon elastic collisions:
            elseif(((ILB1.GE.1.AND.ILB1.LE.2)
     1              .or.(ILB1.GE.6.AND.ILB1.LE.13)
     2              .or.(ILB2.GE.1.AND.ILB2.LE.2)
     3              .or.(ILB2.GE.6.AND.ILB2.LE.13))
     4              .and.(lb1*lb2).gt.0) then
               GOTO 506
            else
               GOTO 400
            endif
         ENDIF
c
* K+ + (N,N*,D)-bar --> L/S-bar + pi
          if( ((lb1.eq.23.or.lb1.eq.30).and.
     &         (lb2.eq.-1.or.lb2.eq.-2.or.(lb2.ge.-13.and.lb2.le.-6))) 
     &         .OR.((lb2.eq.23.or.lb2.eq.30).and.
     &         (lb1.eq.-1.or.lb1.eq.-2.or.(lb1.ge.-13.and.lb1.le.-6))) )
     &         then
             bmass=0.938
             if(srt.le.(bmass+aka)) then
                pkaon=0.
             else
                pkaon=sqrt(((srt**2-(aka**2+bmass**2))
     1               /2./bmass)**2-aka**2)
             endif
clin-10/31/02 cross sections are isospin-averaged, same as those in newka
c     for K- + (N,N*,D) --> L/S + pi:
             sigela = 0.5 * (AKPEL(PKAON) + AKNEL(PKAON))
             SIGSGM = 1.5 * AKPSGM(PKAON) + AKNSGM(PKAON)
             SIG = sigela + SIGSGM + AKPLAM(PKAON)
             if(sig.gt.1.e-7) then
c     ! K+ + N-bar reactions
                icase=3
                brel=sigela/sig
                brsgm=sigsgm/sig
                brsig = sig
                nchrg = 1
                go to 3555
             endif
             go to 400
          endif
c
c
c  meson + hyperon-bar -> K+ + N-bar
          if(((lb1.ge.-17.and.lb1.le.-14).and.(lb2.ge.3.and.lb2.le.5)) 
     &         .OR.((lb2.ge.-17.and.lb2.le.-14)
     &         .and.(lb1.ge.3.and.lb1.le.5)))then
             nchrg=-100
 
C*       first classify the reactions due to total charge.
             if((lb1.eq.-15.and.(lb2.eq.5.or.lb2.eq.27)).OR.
     &            (lb2.eq.-15.and.(lb1.eq.5.or.lb1.eq.27))) then
                nchrg=-2
c     ! D-(bar)
                bmass=1.232
                go to 110
             endif
             if( (lb1.eq.-15.and.(lb2.eq.0.or.lb2.eq.4.or.lb2.eq.26.or.
     &            lb2.eq.28)).OR.(lb2.eq.-15.and.(lb1.eq.0.or.
     &            lb1.eq.4.or.lb1.eq.26.or.lb1.eq.28)).OR.
     &   ((lb1.eq.-14.or.lb1.eq.-16).and.(lb2.eq.5.or.lb2.eq.27)).OR.
     &   ((lb2.eq.-14.or.lb2.eq.-16).and.(lb1.eq.5.or.lb1.eq.27)) )then
                nchrg=-1
c     ! n-bar
                bmass=0.938
                go to 110
             endif
             if(  (lb1.eq.-15.and.(lb2.eq.3.or.lb2.eq.25)).OR.
     &            (lb2.eq.-15.and.(lb1.eq.3.or.lb1.eq.25)).OR.
     &            (lb1.eq.-17.and.(lb2.eq.5.or.lb2.eq.27)).OR.
     &            (lb2.eq.-17.and.(lb1.eq.5.or.lb1.eq.27)).OR.
     &            ((lb1.eq.-14.or.lb1.eq.-16).and.(lb2.eq.0.or.lb2.eq.4
     &            .or.lb2.eq.26.or.lb2.eq.28)).OR.
     &            ((lb2.eq.-14.or.lb2.eq.-16).and.(lb1.eq.0.or.lb1.eq.4
     &            .or.lb1.eq.26.or.lb1.eq.28)) )then
               nchrg=0
c     ! p-bar
                bmass=0.938
                go to 110
             endif
             if( (lb1.eq.-17.and.(lb2.eq.0.or.lb2.eq.4.or.lb2.eq.26.or.
     &            lb2.eq.28)).OR.(lb2.eq.-17.and.(lb1.eq.0.or.
     &            lb1.eq.4.or.lb1.eq.26.or.lb1.eq.28)).OR.
     &  ((lb1.eq.-14.or.lb1.eq.-16).and.(lb2.eq.3.or.lb2.eq.25)).OR.
     &  ((lb2.eq.-14.or.lb2.eq.-16).and.(lb1.eq.3.or.lb1.eq.25)))then
               nchrg=1
c     ! D++(bar)
                bmass=1.232
             endif
c
c 110     if(nchrg.ne.-100.and.srt.ge.(aka+bmass))then !! for elastic
 110         sig = 0.
c !! for elastic
         if(nchrg.ne.-100.and.srt.ge.(aka+bmass))then
cc110        if(nchrg.eq.-100.or.srt.lt.(aka+bmass)) go to 400
c             ! PI + La(Si)-bar => K+ + N-bar reactions
            icase=4
cc       pkaon=sqrt(((srt**2-(aka**2+bmass**2))/2./bmass)**2-aka**2)
            pkaon=sqrt(((srt**2-(aka**2+0.938**2))/2./0.938)**2-aka**2)
c ! lambda-bar + Pi
            if(lb1.eq.-14.or.lb2.eq.-14) then
               if(nchrg.ge.0) sigma0=akPlam(pkaon)
               if(nchrg.lt.0) sigma0=akNlam(pkaon)
c                ! sigma-bar + pi
            else
c !K-p or K-D++
               if(nchrg.ge.0) sigma0=akPsgm(pkaon)
c !K-n or K-D-
               if(nchrg.lt.0) sigma0=akNsgm(pkaon)
               SIGMA0 = 1.5 * AKPSGM(PKAON) + AKNSGM(PKAON)
            endif
            sig=(srt**2-(aka+bmass)**2)*(srt**2-(aka-bmass)**2)/
     &           (srt**2-(em1+em2)**2)/(srt**2-(em1-em2)**2)*sigma0
c ! K0barD++, K-D-
            if(nchrg.eq.-2.or.nchrg.eq.2) sig=2.*sig
C*     the factor 2 comes from spin of delta, which is 3/2
C*     detailed balance. copy from Page 423 of N.P. A614 1997
            IF (LB1 .EQ. -14 .OR. LB2 .EQ. -14) THEN
               SIG = 4.0 / 3.0 * SIG
            ELSE IF (NCHRG .EQ. -2 .OR. NCHRG .EQ. 2) THEN
               SIG = 8.0 / 9.0 * SIG
            ELSE
               SIG = 4.0 / 9.0 * SIG
            END IF
cc        brel=0.
cc        brsgm=0.
cc        brsig = sig
cc          if(sig.lt.1.e-7) go to 400
*-
         endif
c                ! PI + La(Si)-bar => elastic included
         icase=4
         sigela = 10.
         sig = sig + sigela
         brel= sigela/sig
         brsgm=0.
         brsig = sig
*-
         go to 3555
      endif
      
** MULTISTRANGE PARTICLE (Cas,Omega -bar) PRODUCTION - (NON)PERTURBATIVE

* K-/K*0bar + La/Si --> cascade + pi/eta
      if( ((lb1.eq.21.or.lb1.eq.-30).and.(lb2.ge.14.and.lb2.le.17)).OR.
     &  ((lb2.eq.21.or.lb2.eq.-30).and.(lb1.ge.14.and.lb1.le.17)) )then
          kp = 0
          go to 3455
        endif
c K+/K*0 + La/Si(bar) --> cascade-bar + pi/eta
      if( ((lb1.eq.23.or.lb1.eq.30).and.(lb2.le.-14.and.lb2.ge.-17)).OR.
     &  ((lb2.eq.23.or.lb2.eq.30).and.(lb1.le.-14.and.lb1.ge.-17)) )then
          kp = 1
          go to 3455
        endif
* K-/K*0bar + cascade --> omega + pi
       if( ((lb1.eq.21.or.lb1.eq.-30).and.(lb2.eq.40.or.lb2.eq.41)).OR.
     & ((lb2.eq.21.or.lb2.eq.-30).and.(lb1.eq.40.or.lb1.eq.41)) )then
          kp = 0
          go to 3455
        endif
* K+/K*0 + cascade-bar --> omega-bar + pi
       if( ((lb1.eq.23.or.lb1.eq.30).and.(lb2.eq.-40.or.lb2.eq.-41)).OR.
     &  ((lb2.eq.23.or.lb2.eq.30).and.(lb1.eq.-40.or.lb1.eq.-41)) )then
          kp = 1
          go to 3455
        endif
* Omega + Omega --> Di-Omega + photon(eta)
cc        if( lb1.eq.45.and.lb2.eq.45 ) go to 3455

c annhilation of cascade(bar), omega(bar)
         kp = 3
* K- + L/S <-- cascade(bar) + pi/eta
       if( (((lb1.ge.3.and.lb1.le.5).or.lb1.eq.0) 
     &       .and.(iabs(lb2).eq.40.or.iabs(lb2).eq.41))
     & .OR. (((lb2.ge.3.and.lb2.le.5).or.lb2.eq.0) 
     &       .and.(iabs(lb1).eq.40.or.iabs(lb1).eq.41)) )go to 3455
* K- + cascade(bar) <-- omega(bar) + pi
*         if(  (lb1.eq.0.and.iabs(lb2).eq.45)
*    &       .OR. (lb2.eq.0.and.iabs(lb1).eq.45) )go to 3455
        if( ((lb1.ge.3.and.lb1.le.5).and.iabs(lb2).eq.45)
     &  .OR.((lb2.ge.3.and.lb2.le.5).and.iabs(lb1).eq.45) )go to 3455
c

***  MULTISTRANGE PARTICLE PRODUCTION  (END)

c* K+ + La(Si) --> Meson + B
        IF (LB1.EQ.23 .AND. (LB2.GE.14.AND.LB2.LE.17)) GOTO 5699
        IF (LB2.EQ.23 .AND. (LB1.GE.14.AND.LB1.LE.17)) GOTO 5699
c* K- + La(Si)-bar --> Meson + B-bar
       IF (LB1.EQ.21 .AND. (LB2.GE.-17.AND.LB2.LE.-14)) GOTO 5699
       IF (LB2.EQ.21 .AND. (LB1.GE.-17.AND.LB1.LE.-14)) GOTO 5699

c La/Si-bar + B --> pi + K+
       IF( (((LB1.eq.1.or.LB1.eq.2).or.(LB1.ge.6.and.LB1.le.13))
     &       .AND.(LB2.GE.-17.AND.LB2.LE.-14)) .OR.
     &     (((LB2.eq.1.or.LB2.eq.2).or.(LB2.ge.6.and.LB2.le.13))
     &      .AND.(LB1.GE.-17.AND.LB1.LE.-14)) )go to 5999
c La/Si + B-bar --> pi + K-
       IF( (((LB1.eq.-1.or.LB1.eq.-2).or.(LB1.le.-6.and.LB1.ge.-13))
     &       .AND.(LB2.GE.14.AND.LB2.LE.17)) .OR.
     &     (((LB2.eq.-1.or.LB2.eq.-2).or.(LB2.le.-6.and.LB2.ge.-13))
     &       .AND.(LB1.GE.14.AND.LB1.LE.17)) )go to 5999 
*
*
* K(K*) + Kbar(K*bar) --> phi + pi(rho,omega), M + M (M=pi,rho,omega,eta)
       if(lb1.eq.21.and.lb2.eq.23) go to 8699
       if(lb2.eq.21.and.lb1.eq.23) go to 8699
       if(lb1.eq.30.and.lb2.eq.21) go to 8699
       if(lb2.eq.30.and.lb1.eq.21) go to 8699
       if(lb1.eq.-30.and.lb2.eq.23) go to 8699
       if(lb2.eq.-30.and.lb1.eq.23) go to 8699
       if(lb1.eq.-30.and.lb2.eq.30) go to 8699
       if(lb2.eq.-30.and.lb1.eq.30) go to 8699
c* (K,K*)-bar + rho(omega) --> phi +(K,K*)-bar, piK and elastic
       IF( ((lb1.eq.23.or.lb1.eq.21.or.iabs(lb1).eq.30) .and.
     &      (lb2.ge.25.and.lb2.le.28)) .OR.
     &     ((lb2.eq.23.or.lb2.eq.21.or.iabs(lb2).eq.30) .and.
     &      (lb1.ge.25.and.lb1.le.28)) ) go to 8799
c
c* K*(-bar) + pi --> phi + (K,K*)-bar
       IF( (iabs(lb1).eq.30.and.(lb2.ge.3.and.lb2.le.5)) .OR.
     &     (iabs(lb2).eq.30.and.(lb1.ge.3.and.lb1.le.5)) )go to 8799
*
c
c* phi + N --> pi+N(D),  rho+N(D),  K+ +La
c* phi + D --> pi+N(D),  rho+N(D)
       IF( (lb1.eq.29 .and.(lb2.eq.1.or.lb2.eq.2.or.
     &       (lb2.ge.6.and.lb2.le.9))) .OR.
     &     (lb2.eq.29 .and.(lb1.eq.1.or.lb1.eq.2.or.
     &       (lb1.ge.6.and.lb1.le.9))) )go to 7222
c
c* phi + (pi,rho,ome,K,K*-bar) --> K+K, K+K*, K*+K*, (pi,rho,omega)+(K,K*-bar)
       IF( (lb1.eq.29 .and.((lb2.ge.3.and.lb2.le.5).or.
     &      (lb2.ge.21.and.lb2.le.28).or.iabs(lb2).eq.30)) .OR.
     &     (lb2.eq.29 .and.((lb1.ge.3.and.lb1.le.5).or.
     &      (lb1.ge.21.and.lb1.le.28).or.iabs(lb1).eq.30)) )THEN
             go to 7444
      endif
*
c
* La/Si, Cas, Om (bar)-(rho,omega,phi) elastic colln
* pion vs. La, Ca, Omega-(bar) elastic coll. treated in resp. subroutines
      if( ((iabs(lb1).ge.14.and.iabs(lb1).le.17).or.iabs(lb1).ge.40)
     &    .and.((lb2.ge.25.and.lb2.le.29).or.lb2.eq.0) )go to 888
      if( ((iabs(lb2).ge.14.and.iabs(lb2).le.17).or.iabs(lb2).ge.40)
     &    .and.((lb1.ge.25.and.lb1.le.29).or.lb1.eq.0) )go to 888
c
c K+/K* (N,R)  OR   K-/K*- (N,R)-bar  elastic scatt
        if( ((lb1.eq.23.or.lb1.eq.30).and.(lb2.eq.1.or.lb2.eq.2.or.
     &         (lb2.ge.6.and.lb2.le.13))) .OR.
     &      ((lb2.eq.23.or.lb2.eq.30).and.(lb1.eq.1.or.lb1.eq.2.or.
     &         (lb1.ge.6.and.lb1.le.13))) ) go to 888
        if( ((lb1.eq.21.or.lb1.eq.-30).and.(lb2.eq.-1.or.lb2.eq.-2.or.
     &       (lb2.ge.-13.and.lb2.le.-6))) .OR. 
     &      ((lb2.eq.21.or.lb2.eq.-30).and.(lb1.eq.-1.or.lb1.eq.-2.or.
     &       (lb1.ge.-13.and.lb1.le.-6))) ) go to 888
c
* L/S-baryon elastic collision 
       If( ((lb1.ge.14.and.lb1.le.17).and.(lb2.ge.6.and.lb2.le.13))
     & .OR.((lb2.ge.14.and.lb2.le.17).and.(lb1.ge.6.and.lb1.le.13)) )
     &   go to 7799
       If(((lb1.le.-14.and.lb1.ge.-17).and.(lb2.le.-6.and.lb2.ge.-13))
     &.OR.((lb2.le.-14.and.lb2.ge.-17).and.(lb1.le.-6.and.lb1.ge.-13)))
     &   go to 7799
c
c skip other collns with perturbative particles or hyperon-bar
       if( iabs(lb1).ge.40 .or. iabs(lb2).ge.40
     &    .or. (lb1.le.-14.and.lb1.ge.-17) 
     &    .or. (lb2.le.-14.and.lb2.ge.-17) )go to 400
c
c
* anti-baryon on baryon resonaces 
        if((lb1.eq.-1.or.lb1.eq.-2.or.(lb1.ge.-13.and.lb1.le.-6))
     1 .and.(lb2.eq.1.or.lb2.eq.2.or.(lb2.ge.6.and.lb2.le.13))) then
            GOTO 2799
       else if((lb2.eq.-1.or.lb2.eq.-2.or.(lb2.ge.-13.and.lb2.le.-6))
     1 .and.(lb1.eq.1.or.lb1.eq.2.or.(lb1.ge.6.and.lb1.le.13))) then
            GOTO 2799
         END IF
c
clin-10/25/02 get rid of argument usage mismatch in newka():
         inewka=irun
c        call newka(icase,irun,iseed,dt,nt,
clin-5/01/03 set iblock value in art1f.f, necessary for resonance studies:
c        call newka(icase,inewka,iseed,dt,nt,
c     &                  ictrl,i1,i2,srt,pcx,pcy,pcz)
        call newka(icase,inewka,iseed,dt,nt,
     &                  ictrl,i1,i2,srt,pcx,pcy,pcz,iblock)

clin-10/25/02-end
        IF (ICTRL .EQ. 1) GOTO 400
c
* SEPARATE NUCLEON+NUCLEON( BARYON RESONANCE+ BARYON RESONANCE ELASTIC
* COLLISION), BARYON RESONANCE+NUCLEON AND BARYON-PION
* COLLISIONS INTO THREE PARTS TO CHECK IF THEY ARE GOING TO SCATTER,
* WE only allow L/S to COLLIDE elastically with a nucleon and meson
       if((iabs(lb1).ge.14.and.iabs(lb1).le.17).
     &  or.(iabs(lb2).ge.14.and.iabs(lb2).le.17))go to 400
* IF PION+PION COLLISIONS GO TO 777
* if pion+eta, eta+eta to create kaons go to 777 
       IF((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.3.and.lb2.le.5))GO TO 777
       if(lb1.eq.0.and.(lb2.ge.3.and.lb2.le.5)) go to 777
       if(lb2.eq.0.and.(lb1.ge.3.and.lb1.le.5)) go to 777
       if(lb1.eq.0.and.lb2.eq.0)go to 777
* we assume that rho and omega behave the same way as pions in 
* kaon production
* (1) rho(omega)+rho(omega)
       if( (lb1.ge.25.and.lb1.le.28).and.
     &     (lb2.ge.25.and.lb2.le.28) )goto 777
* (2) rho(omega)+pion
      If((lb1.ge.25.and.lb1.le.28).and.(lb2.ge.3.and.lb2.le.5))go to 777
      If((lb2.ge.25.and.lb2.le.28).and.(lb1.ge.3.and.lb1.le.5))go to 777
* (3) rho(omega)+eta
       if((lb1.ge.25.and.lb1.le.28).and.lb2.eq.0)go to 777
       if((lb2.ge.25.and.lb2.le.28).and.lb1.eq.0)go to 777
c
* if kaon+pion collisions go to 889
       if((lb1.eq.23.or.lb1.eq.21).and.(lb2.ge.3.and.lb2.le.5))go to 889
       if((lb2.eq.23.or.lb2.eq.21).and.(lb1.ge.3.and.lb1.le.5))go to 889
c
clin-2/06/03 skip all other (K K* Kbar K*bar) channels:
* SKIP all other K and K* RESCATTERINGS
        If(iabs(lb1).eq.30.or.iabs(lb2).eq.30) go to 400
        If(lb1.eq.21.or.lb2.eq.21) go to 400
        If(lb1.eq.23.or.lb2.eq.23) go to 400
c
* IF PION+baryon COLLISION GO TO 3
           IF( (LB1.ge.3.and.LB1.le.5) .and. 
     &         (iabs(LB2).eq.1.or.iabs(LB2).eq.2.or.
     &          (iabs(LB2).ge.6.and.iabs(LB2).le.13)) )GO TO 3
           IF( (LB2.ge.3.and.LB2.le.5) .and. 
     &         (iabs(LB1).eq.1.or.iabs(LB1).eq.2.or.
     &          (iabs(LB1).ge.6.and.iabs(LB1).le.13)) )GO TO 3
c
* IF rho(omega)+NUCLEON (baryon resonance) COLLISION GO TO 33
           IF( (LB1.ge.25.and.LB1.le.28) .and. 
     &         (iabs(LB2).eq.1.or.iabs(LB2).eq.2.or.
     &          (iabs(LB2).ge.6.and.iabs(LB2).le.13)) )GO TO 33
           IF( (LB2.ge.25.and.LB2.le.28) .and. 
     &         (iabs(LB1).eq.1.or.iabs(LB1).eq.2.or.
     &          (iabs(LB1).ge.6.and.iabs(LB1).le.13)) )GO TO 33
c
* IF ETA+NUCLEON (baryon resonance) COLLISIONS GO TO 547
           IF( LB1.eq.0 .and. 
     &         (iabs(LB2).eq.1.or.iabs(LB2).eq.2.or.
     &          (iabs(LB2).ge.6.and.iabs(LB2).le.13)) )GO TO 547
           IF( LB2.eq.0 .and. 
     &         (iabs(LB1).eq.1.or.iabs(LB1).eq.2.or.
     &          (iabs(LB1).ge.6.and.iabs(LB1).le.13)) )GO TO 547
c
* IF NUCLEON+BARYON RESONANCE COLLISION GO TO 44
            IF((LB1.eq.1.or.lb1.eq.2).
     &        AND.(LB2.GT.5.and.lb2.le.13))GOTO 44
            IF((LB2.eq.1.or.lb2.eq.2).
     &        AND.(LB1.GT.5.and.lb1.le.13))GOTO 44
            IF((LB1.eq.-1.or.lb1.eq.-2).
     &        AND.(LB2.LT.-5.and.lb2.ge.-13))GOTO 44
            IF((LB2.eq.-1.or.lb2.eq.-2).
     &        AND.(LB1.LT.-5.and.lb1.ge.-13))GOTO 44
c
* IF NUCLEON+NUCLEON COLLISION GO TO 4
       IF((LB1.eq.1.or.lb1.eq.2).AND.(LB2.eq.1.or.lb2.eq.2))GOTO 4
       IF((LB1.eq.-1.or.lb1.eq.-2).AND.(LB2.eq.-1.or.lb2.eq.-2))GOTO 4
c
* IF BARYON RESONANCE+BARYON RESONANCE COLLISION GO TO 444
            IF((LB1.GT.5.and.lb1.le.13).AND.
     &         (LB2.GT.5.and.lb2.le.13)) GOTO 444
            IF((LB1.LT.-5.and.lb1.ge.-13).AND.
     &         (LB2.LT.-5.and.lb2.ge.-13)) GOTO 444
c
* if L/S+L/S or L/s+nucleon go to 400
* otherwise, develop a model for their collisions
       if((lb1.lt.3).and.(lb2.ge.14.and.lb2.le.17))goto 400
       if((lb2.lt.3).and.(lb1.ge.14.and.lb1.le.17))goto 400
       if((lb1.ge.14.and.lb1.le.17).and.
     &  (lb2.ge.14.and.lb2.le.17))goto 400
c
* otherwise, go out of the loop
              go to 400
*
*
547           IF(LB1*LB2.EQ.0)THEN
* (1) FOR ETA+NUCLEON SYSTEM, we allow both elastic collision, 
*     i.e. N*(1535) formation and kaon production
*     the total kaon production cross section is
*     ASSUMED to be THE SAME AS PION+NUCLEON COLLISIONS
* (2) for eta+baryon resonance we only allow kaon production
           ece=(em1+em2+0.02)**2
           xkaon0=0.
           if(srt.ge.1.63.AND.SRT.LE.1.7)xkaon0=pnlka(srt)
           IF(SRT.GT.1.7)XKAON0=PNLKA(SRT)+pnska(srt)
cbz3/7/99 neutralk
            XKAON0 = 2.0 * XKAON0
cbz3/7/99 neutralk end

* Here we negelect eta+n inelastic collisions other than the 
* kaon production, therefore the total inelastic cross section
* xkaon equals to the xkaon0 (kaon production cross section)
           xkaon=xkaon0
* note here the xkaon is in unit of fm**2
            XETA=XN1535(I1,I2,0)
        If((iabs(LB(I1)).ge.6.and.iabs(LB(I1)).le.13).or.
     &     (iabs(LB(I2)).ge.6.and.iabs(LB(I2)).le.13)) xeta=0.      
            IF((XETA+xkaon).LE.1.e-06)GO TO 400
            DSE=SQRT((XETA+XKAON)/PI)
           DELTRE=DSE+0.1
        px1cm=pcx
        py1cm=pcy
        pz1cm=pcz
* CHECK IF N*(1535) resonance CAN BE FORMED
         CALL DISTCE(I1,I2,DELTRE,DSE,DT,ECE,SRT,IC,
     1   PCX,PCY,PCZ)
         IF(IC.EQ.-1) GO TO 400
         ekaon(4,iss)=ekaon(4,iss)+1
        IF(XKAON0/(XKAON+XETA).GT.RANART(NSEED))then
* kaon production, USE CREN TO CALCULATE THE MOMENTUM OF L/S K+
        CALL CREN(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)
* kaon production
       IF(IBLOCK.EQ.7) then
          LPN=LPN+1
       elseIF(IBLOCK.EQ.-7) then
       endif
c
       em1=e(i1)
       em2=e(i2)
       GO TO 440
       endif
* N*(1535) FORMATION
        resona=1.
         GO TO 98
         ENDIF
*IF PION+NUCLEON (baryon resonance) COLLISION THEN
3           CONTINUE
           px1cm=pcx
           py1cm=pcy
           pz1cm=pcz
* the total kaon production cross section for pion+baryon (resonance) is
* assumed to be the same as in pion+nucleon
           xkaon0=0.
           if(srt.ge.1.63.AND.SRT.LE.1.7)xkaon0=pnlka(srt)
           IF(SRT.GT.1.7)XKAON0=PNLKA(SRT)+pnska(srt)
            XKAON0 = 2.0 * XKAON0
c
c sp11/21/01  phi production: pi +N(D) -> phi + N(D)
         Xphi = 0.
       if( ( ((lb1.ge.1.and.lb1.le.2).or.
     &        (lb1.ge.6.and.lb1.le.9))
     &   .OR.((lb2.ge.1.and.lb2.le.2).or.
     &        (lb2.ge.6.and.lb2.le.9)) )
     &       .AND. srt.gt.1.958)
     &        call pibphi(srt,lb1,lb2,em1,em2,Xphi,xphin)
c !! in fm^2 above

* if a pion collide with a baryon resonance, 
* we only allow kaon production AND the reabsorption 
* processes: Delta+pion-->N+pion, N*+pion-->N+pion
* Later put in pion+baryon resonance elastic
* cross through forming higher resonances implicitly.
c          If(em1.gt.1.or.em2.gt.1.)go to 31
         If((iabs(LB(I1)).ge.6.and.iabs(LB(I1)).le.13).or.
     &      (iabs(LB(I2)).ge.6.and.iabs(LB(I2)).le.13)) go to 31
* For pion+nucleon collisions: 
* using the experimental pion+nucleon inelastic cross section, we assume it
* is exhausted by the Delta+pion, Delta+rho and Delta+omega production 
* and kaon production. In the following we first check whether 
* inelastic pion+n collision can happen or not, then determine in 
* crpn whether it is through pion production or through kaon production
* note that the xkaon0 is the kaon production cross section
* Note in particular that: 
* xkaon in the following is the total pion+nucleon inelastic cross section
* note here the xkaon is in unit of fm**2, xnpi is also in unit of fm**2
* FOR PION+NUCLEON SYSTEM, THE MINIMUM S IS 1.2056 the minimum srt for 
* elastic scattering, and it is 1.60 for pion production, 1.63 for LAMBDA+kaon 
* production and 1.7 FOR SIGMA+KAON
* (EC = PION MASS+NUCLEON MASS+20MEV)**2
            EC=(em1+em2+0.02)**2
           xkaon=0.
           if(srt.gt.1.23)xkaon=(pionpp(srt)+PIPP1(SRT))/2.
* pion+nucleon elastic cross section is divided into two parts:
* (1) forming D(1232)+N*(1440) +N*(1535)
* (2) cross sections forming higher resonances are calculated as
*     the difference between the total elastic and (1), this part is 
*     treated as direct process since we do not explicitLY include
*     higher resonances.
* the following is the resonance formation cross sections.
*1. PION(+)+PROTON-->DELTA++,PION(-)+NEUTRON-->DELTA(-)
           IF( (LB1*LB2.EQ.5.OR.((LB1*LB2.EQ.6).AND.
     &         (LB1.EQ.3.OR.LB2.EQ.3)))
     &    .OR. (LB1*LB2.EQ.-3.OR.((LB1*LB2.EQ.-10).AND.
     &         (LB1.EQ.5.OR.LB2.EQ.5))) )then    
              XMAX=190.
              xmaxn=0
              xmaxn1=0
              xdirct=dirct1(srt)
               go to 678
           endif
*2. PION(-)+PROTON-->DELTA0,PION(+)+NEUTRON-->DELTA+ 
*   or N*(+)(1440) or N*(+)(1535)
* note the factor 2/3 is from the isospin consideration and
* the factor 0.6 or 0.5 is the branching ratio for the resonance to decay
* into pion+nucleon
            IF( (LB1*LB2.EQ.3.OR.((LB1*LB2.EQ.10).AND.
     &          (LB1.EQ.5.OR.LB2.EQ.5)))
     &     .OR. (LB1*LB2.EQ.-5.OR.((LB1*LB2.EQ.-6).AND.
     &          (LB1.EQ.3.OR.LB2.EQ.3))) )then      
              XMAX=27.
              xmaxn=2./3.*25.*0.6
               xmaxn1=2./3.*40.*0.5
              xdirct=dirct2(srt)
               go to 678
              endif
*3. PION0+PROTON-->DELTA+,PION0+NEUTRON-->DELTA0, or N*(0)(1440) or N*(0)(1535)
            IF((LB1.EQ.4.OR.LB2.EQ.4).AND.
     &         (iabs(LB1*LB2).EQ.4.OR.iabs(LB1*LB2).EQ.8))then
              XMAX=50.
              xmaxn=1./3.*25*0.6
              xmaxn1=1/3.*40.*0.5
              xdirct=dirct3(srt)
                go to 678
              endif
678           xnpin1=0
           xnpin=0
            XNPID=XNPI(I1,I2,1,XMAX)
           if(xmaxn1.ne.0)xnpin1=XNPI(i1,i2,2,XMAXN1)
            if(xmaxn.ne.0)XNPIN=XNPI(I1,I2,0,XMAXN)
* the following 
           xres=xnpid+xnpin+xnpin1
           xnelas=xres+xdirct 
           icheck=1
           go to 34
* For pion + baryon resonance the reabsorption 
* cross section is calculated from the detailed balance
* using reab(i1,i2,srt,ictrl), ictrl=1, 2 and 3
* for pion, rho and omega + baryon resonance
31           ec=(em1+em2+0.02)**2
           xreab=reab(i1,i2,srt,1)

clin-12/02/00 to satisfy detailed balance, forbid N* absorptions:
          if((iabs(lb1).ge.10.and.iabs(lb1).le.13)
     1         .or.(iabs(lb2).ge.10.and.iabs(lb2).le.13)) XREAB = 0.

           xkaon=xkaon0+xreab
* a constant of 10 mb IS USED FOR PION + N* RESONANCE, 
        IF((iabs(LB1).GT.9.AND.iabs(LB1).LE.13) .OR.
     &      (iabs(LB2).GT.9.AND.iabs(LB2).LE.13))THEN
           Xnelas=1.0
        ELSE
           XNELAS=DPION(EM1,EM2,LB1,LB2,SRT)
        ENDIF
           icheck=2
34          IF((Xnelas+xkaon+Xphi).LE.0.000001)GO TO 400
            DS=SQRT((Xnelas+xkaon+Xphi)/PI)
csp09/20/01
c           totcr = xnelas+xkaon
c           if(srt .gt. 3.5)totcr = max1(totcr,3.)
c           DS=SQRT(totcr/PI)
csp09/20/01 end
            
           deltar=ds+0.1
         CALL DISTCE(I1,I2,DELTAR,DS,DT,EC,SRT,IC,
     1   PCX,PCY,PCZ)
         IF(IC.EQ.-1) GO TO 400
       ekaon(4,iss)=ekaon(4,iss)+1
c***
* check what kind of collision has happened
* (1) pion+baryon resonance
* if direct elastic process
        if(icheck.eq.2)then
c  !!sp11/21/01
      if(xnelas/(xnelas+xkaon+Xphi).ge.RANART(NSEED))then
c               call Crdir(PX1CM,PY1CM,PZ1CM,SRT,I1,I2)
               call Crdir(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)
              go to 440
              else
* for inelastic process, go to 96 to check
* kaon production and pion reabsorption : pion+D(N*)-->pion+N
               go to 96
                endif
              endif
*(2) pion+n
* CHECK IF inELASTIC COLLISION IS POSSIBLE FOR PION+N COLLISIONS
clin-8/17/00 typo corrected, many other occurences:
c        IF(XKAON/(XKAON+Xnelas).GT.RANART(NSEED))GO TO 95
       IF((XKAON+Xphi)/(XKAON+Xphi+Xnelas).GT.RANART(NSEED))GO TO 95

* direct process
        if(xdirct/xnelas.ge.RANART(NSEED))then
c               call Crdir(PX1CM,PY1CM,PZ1CM,SRT,I1,I2)
               call Crdir(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)
              go to 440
              endif
* now resonance formation or direct process (higher resonances)
           IF( (LB1*LB2.EQ.5.OR.((LB1*LB2.EQ.6).AND.
     &         (LB1.EQ.3.OR.LB2.EQ.3)))
     &    .OR. (LB1*LB2.EQ.-3.OR.((LB1*LB2.EQ.-10).AND.
     &         (LB1.EQ.5.OR.LB2.EQ.5))) )then    
c
* ONLY DELTA RESONANCE IS POSSIBLE, go to 99
        GO TO 99
       else
* NOW BOTH DELTA AND N* RESORANCE ARE POSSIBLE
* DETERMINE THE RESORANT STATE BY USING THE MONTRE CARLO METHOD
            XX=(XNPIN+xnpin1)/xres
            IF(RANART(NSEED).LT.XX)THEN
* N* RESONANCE IS SELECTED
* decide N*(1440) or N*(1535) formation
        xx0=xnpin/(xnpin+xnpin1)
        if(RANART(NSEED).lt.xx0)then
         RESONA=0.
* N*(1440) formation
         GO TO 97
        else
* N*(1535) formation
        resona=1.
         GO TO 98
        endif
         ELSE
* DELTA RESONANCE IS SELECTED
         GO TO 99
         ENDIF
         ENDIF
97       CONTINUE
            IF(RESONA.EQ.0.)THEN
*N*(1440) IS PRODUCED,WE DETERMINE THE CHARGE STATE OF THE PRODUCED N*
            I=I1
            IF(EM1.LT.0.6)I=I2
* (0.1) n+pion(+)-->N*(+)
           IF( (LB1*LB2.EQ.10.AND.(LB1.EQ.5.OR.LB2.EQ.5))
     &      .OR.(LB1*LB2.EQ.-6.AND.(LB1.EQ.3.OR.LB2.EQ.3)) )THEN
            LB(I)=11
           go to 303
            ENDIF
* (0.2) p+pion(0)-->N*(+)
c            IF(LB(I1)*LB(I2).EQ.4.AND.(LB(I1).EQ.1.OR.LB(I2).EQ.1))THEN
            IF(iabs(LB(I1)*LB(I2)).EQ.4.AND.
     &         (LB(I1).EQ.4.OR.LB(I2).EQ.4))THEN    
            LB(I)=11
           go to 303
            ENDIF
* (0.3) n+pion(0)-->N*(0)
c            IF(LB(I1)*LB(I2).EQ.8.AND.(LB(I1).EQ.2.OR.LB(I2).EQ.2))THEN
            IF(iabs(LB(I1)*LB(I2)).EQ.8.AND.
     &        (LB(I1).EQ.4.OR.LB(I2).EQ.4))THEN    
            LB(I)=10
           go to 303
            ENDIF
* (0.4) p+pion(-)-->N*(0)
c            IF(LB(I1)*LB(I2).EQ.3)THEN
            IF( (LB(I1)*LB(I2).EQ.3)
     &      .OR.(LB(I1)*LB(I2).EQ.-5) )THEN
            LB(I)=10
            ENDIF
303         CALL DRESON(I1,I2)
            if(LB1.lt.0.or.LB2.lt.0) LB(I)=-LB(I)
            lres=lres+1
            GO TO 101
*COM: GO TO 101 TO CHANGE THE PHASE SPACE DENSITY OF THE NUCLEON
            ENDIF
98          IF(RESONA.EQ.1.)THEN
*N*(1535) IS PRODUCED, WE DETERMINE THE CHARGE STATE OF THE PRODUCED N*
            I=I1
            IF(EM1.LT.0.6)I=I2
* note: this condition applies to both eta and pion
* (0.1) n+pion(+)-->N*(+)
c            IF(LB1*LB2.EQ.10.AND.(LB1.EQ.2.OR.LB2.EQ.2))THEN
            IF( (LB1*LB2.EQ.10.AND.(LB1.EQ.5.OR.LB2.EQ.5))
     &      .OR.(LB1*LB2.EQ.-6.AND.(LB1.EQ.3.OR.LB2.EQ.3)) )THEN
            LB(I)=13
           go to 304
            ENDIF
* (0.2) p+pion(0)-->N*(+)
c            IF(LB(I1)*LB(I2).EQ.4.AND.(LB(I1).EQ.1.OR.LB(I2).EQ.1))THEN
            IF(iabs(LB(I1)*LB(I2)).EQ.4.AND.
     &           (LB(I1).EQ.4.OR.LB(I2).EQ.4))THEN 
            LB(I)=13
           go to 304
            ENDIF
* (0.3) n+pion(0)-->N*(0)
c            IF(LB(I1)*LB(I2).EQ.8.AND.(LB(I1).EQ.2.OR.LB(I2).EQ.2))THEN
            IF(iabs(LB(I1)*LB(I2)).EQ.8.AND.
     &           (LB(I1).EQ.4.OR.LB(I2).EQ.4))THEN      
            LB(I)=12
           go to 304
            ENDIF
* (0.4) p+pion(-)-->N*(0)
c            IF(LB(I1)*LB(I2).EQ.3)THEN
            IF( (LB(I1)*LB(I2).EQ.3)
     &      .OR.(LB(I1)*LB(I2).EQ.-5) )THEN
            LB(I)=12
           go to 304
           endif
* (0.5) p+eta-->N*(+)(1535),n+eta-->N*(0)(1535)
           if(lb(i1)*lb(i2).eq.0)then
c            if((lb(i1).eq.1).or.(lb(i2).eq.1))then
            if(iabs(lb(i1)).eq.1.or.iabs(lb(i2)).eq.1)then
           LB(I)=13
           go to 304
           ELSE
           LB(I)=12
           ENDIF
           endif
304         CALL DRESON(I1,I2)
            if(LB1.lt.0.or.LB2.lt.0) LB(I)=-LB(I) 
            lres=lres+1
            GO TO 101
*COM: GO TO 101 TO CHANGE THE PHASE SPACE DENSITY OF THE NUCLEON
            ENDIF
*DELTA IS PRODUCED,IN THE FOLLOWING WE DETERMINE THE
*CHARGE STATE OF THE PRODUCED DELTA
99      LRES=LRES+1
        I=I1
        IF(EM1.LE.0.6)I=I2
* (1) p+pion(+)-->DELTA(++)
c        IF(LB(I1)*LB(I2).EQ.5)THEN
            IF( (LB(I1)*LB(I2).EQ.5)
     &      .OR.(LB(I1)*LB(I2).EQ.-3) )THEN
        LB(I)=9
       go to 305
        ENDIF
* (2) p+pion(0)-->delta(+)
c        IF(LB(I1)*LB(I2).EQ.4.AND.(LB(I1).EQ.1.OR.LB(I2).EQ.1))then
       IF(iabs(LB(I1)*LB(I2)).EQ.4.AND.(LB(I1).EQ.4.OR.LB(I2).EQ.4))then
        LB(I)=8
       go to 305
        ENDIF
* (3) n+pion(+)-->delta(+)
c        IF(LB(I1)*LB(I2).EQ.10.AND.(LB(I1).EQ.2.OR.LB(I2).EQ.2))THEN
       IF( (LB(I1)*LB(I2).EQ.10.AND.(LB(I1).EQ.5.OR.LB(I2).EQ.5))
     & .OR.(LB(I1)*LB(I2).EQ.-6.AND.(LB(I1).EQ.3.OR.LB(I2).EQ.3)) )THEN
        LB(I)=8
       go to 305
        ENDIF
* (4) n+pion(0)-->delta(0)
c        IF(LB(I1)*LB(I2).EQ.8.AND.(LB(I1).EQ.2.OR.LB(I2).EQ.2))THEN
       IF(iabs(LB(I1)*LB(I2)).EQ.8.AND.(LB(I1).EQ.4.OR.LB(I2).EQ.4))THEN
        LB(I)=7
       go to 305
        ENDIF
* (5) p+pion(-)-->delta(0)
c        IF(LB(I1)*LB(I2).EQ.3)THEN
            IF( (LB(I1)*LB(I2).EQ.3)
     &      .OR.(LB(I1)*LB(I2).EQ.-5) )THEN
        LB(I)=7
       go to 305
        ENDIF
* (6) n+pion(-)-->delta(-)
c        IF(LB(I1)*LB(I2).EQ.6.AND.(LB(I1).EQ.2.OR.LB(I2).EQ.2))THEN
       IF( (LB(I1)*LB(I2).EQ.6.AND.(LB(I1).EQ.3.OR.LB(I2).EQ.3))
     & .OR.(LB(I1)*LB(I2).EQ.-10.AND.(LB(I1).EQ.5.OR.LB(I2).EQ.5)) )THEN 
        LB(I)=6
        ENDIF
305     CALL DRESON(I1,I2)
        if(LB1.lt.0.or.LB2.lt.0) LB(I)=-LB(I) 
       GO TO 101

csp-11/08/01 K*
* FOR kaON+pion COLLISIONS, form K* (bar) or
c La/Si-bar + N <-- pi + K+
c La/Si + N-bar <-- pi + K-                                             
c phi + K <-- pi + K                                             
clin (rho,omega) + K* <-- pi + K
889       CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
* the cross section is from C.M. Ko, PRC 23, 2760 (1981).
       spika=60./(1.+4.*(srt-0.895)**2/(0.05)**2)
c
cc       if(lb(i1).eq.23.or.lb(i2).eq.23)then   !! block  K- + pi->La + B-bar

        call Crkpla(PX1CM,PY1CM,PZ1CM,EC,SRT,spika,
     &                  emm1,emm2,lbp1,lbp2,I1,I2,icase,srhoks)
cc
c* only K* or K*bar formation
c       else 
c      DSkn=SQRT(spika/PI/10.)
c      dsknr=dskn+0.1
c      CALL DISTCE(I1,I2,dsknr,DSkn,DT,EC,SRT,IC,
c    1     PX1CM,PY1CM,PZ1CM)
c        IF(IC.EQ.-1) GO TO 400
c       icase = 1
c      endif
c
         if(icase .eq. 0) then
            iblock=0
            go to 400
         endif

       if(icase .eq. 1)then
             call KSRESO(I1,I2)
clin-4/30/03 give non-zero iblock for resonance selections:
             iblock = 171
ctest off for resonance (phi, K*) studies:
c             if(iabs(lb(i1)).eq.30) then
c             write(17,112) 'ks',lb(i1),p(1,i1),p(2,i1),p(3,i1),e(i1),nt
c             elseif(iabs(lb(i2)).eq.30) then
c             write(17,112) 'ks',lb(i2),p(1,i2),p(2,i2),p(3,i2),e(i2),nt
c             endif
c
              lres=lres+1
              go to 101
       elseif(icase .eq. 2)then
             iblock = 71
c
* La/Si (bar) formation                                                   

       elseif(iabs(icase).eq.5)then
             iblock = 88

       else
*
* phi formation
             iblock = 222
       endif
             LB(I1) = lbp1
             LB(I2) = lbp2
             E(I1) = emm1
             E(I2) = emm2
             em1=e(i1)
             em2=e(i2)
             ntag = 0
             go to 440
c             
33       continue
       em1=e(i1)
       em2=e(i2)
* (1) if rho or omega collide with a nucleon we allow both elastic 
*     scattering and kaon production to happen if collision conditions 
*     are satisfied.
* (2) if rho or omega collide with a baryon resonance we allow
*     kaon production, pion reabsorption: rho(omega)+D(N*)-->pion+N
*     and NO elastic scattering to happen
           xelstc=0
            if((lb1.ge.25.and.lb1.le.28).and.
     &    (iabs(lb2).eq.1.or.iabs(lb2).eq.2))
     &      xelstc=ERHON(EM1,EM2,LB1,LB2,SRT)
            if((lb2.ge.25.and.lb2.le.28).and.
     &   (iabs(lb1).eq.1.or.iabs(lb1).eq.2))
     &      xelstc=ERHON(EM1,EM2,LB1,LB2,SRT)
            ec=(em1+em2+0.02)**2
* the kaon production cross section is
           xkaon0=0
           if(srt.ge.1.63.AND.SRT.LE.1.7)xkaon0=pnlka(srt)
           IF(SRT.GT.1.7)XKAON0=PNLKA(SRT)+pnska(srt)
           if(xkaon0.lt.0)xkaon0=0

cbz3/7/99 neutralk
            XKAON0 = 2.0 * XKAON0
cbz3/7/99 neutralk end

* the total inelastic cross section for rho(omega)+N is
           xkaon=xkaon0
           ichann=0
* the total inelastic cross section for rho (omega)+D(N*) is 
* xkaon=xkaon0+reab(**) 

c sp11/21/01  phi production: rho + N(D) -> phi + N(D)
         Xphi = 0.
       if( ( (((lb1.ge.1.and.lb1.le.2).or.
     &         (lb1.ge.6.and.lb1.le.9))
     &         .and.(lb2.ge.25.and.lb2.le.27))
     &   .OR.(((lb2.ge.1.and.lb2.le.2).or.
     &         (lb2.ge.6.and.lb2.le.9))
     &        .and.(lb1.ge.25.and.lb1.le.27)) ).AND. srt.gt.1.958)
     &    call pibphi(srt,lb1,lb2,em1,em2,Xphi,xphin)
c !! in fm^2 above
c
        if((iabs(lb1).ge.6.and.lb2.ge.25).or.
     &    (lb1.ge.25.and.iabs(lb2).ge.6))then
           ichann=1
           ictrl=2
           if(lb1.eq.28.or.lb2.eq.28)ictrl=3
            xreab=reab(i1,i2,srt,ictrl)

clin-12/02/00 to satisfy detailed balance, forbid N* absorptions:
            if((iabs(lb1).ge.10.and.iabs(lb1).le.13)
     1           .or.(iabs(lb2).ge.10.and.iabs(lb2).le.13)) XREAB = 0.

        if(xreab.lt.0)xreab=1.E-06
            xkaon=xkaon0+xreab
          XELSTC=1.0
           endif
            DS=SQRT((XKAON+Xphi+xelstc)/PI)
c
csp09/20/01
c           totcr = xelstc+xkaon
c           if(srt .gt. 3.5)totcr = max1(totcr,3.)
c           DS=SQRT(totcr/PI)
csp09/20/01 end
c
        DELTAR=DS+0.1
       px1cm=pcx
       py1cm=pcy
       pz1cm=pcz
* CHECK IF the collision can happen
         CALL DISTCE(I1,I2,DELTAR,DS,DT,EC,SRT,IC,
     1   PCX,PCY,PCZ)
         IF(IC.EQ.-1) GO TO 400
        ekaon(4,iss)=ekaon(4,iss)+1
c*
* NOW rho(omega)+N or D(N*) COLLISION IS POSSIBLE
* (1) check elastic collision
       if(xelstc/(xelstc+xkaon+Xphi).gt.RANART(NSEED))then
c       call crdir(px1CM,py1CM,pz1CM,srt,I1,i2)
       call crdir(px1CM,py1CM,pz1CM,srt,I1,i2,IBLOCK)
       go to 440
       endif
* (2) check pion absorption or kaon production
        CALL CRRD(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK,xkaon0,xkaon,Xphi,xphin)

* kaon production
csp05/16/01
       IF(IBLOCK.EQ.7) then
          LPN=LPN+1
       elseIF(IBLOCK.EQ.-7) then
       endif
csp05/16/01 end
* rho obsorption
       if(iblock.eq.81) lrhor=lrhor+1
* omega obsorption
       if(iblock.eq.82) lomgar=lomgar+1
       em1=e(i1)
       em2=e(i2)
       GO TO 440
* for pion+n now using the subroutine crpn to change 
* the particle label and set the new momentum of L/S+K final state
95       continue
* NOW PION+N INELASTIC COLLISION IS POSSIBLE
* check pion production or kaon production
        CALL CRPN(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK,xkaon0,xkaon,Xphi,xphin)

* kaon production
csp05/16/01
       IF(IBLOCK.EQ.7) then
          LPN=LPN+1
       elseIF(IBLOCK.EQ.-7) then
       endif
csp05/16/01 end
* pion production
       if(iblock.eq.77) lpd=lpd+1
* rho production
       if(iblock.eq.78) lrho=lrho+1
* omega production
       if(iblock.eq.79) lomega=lomega+1
       em1=e(i1)
       em2=e(i2)
       GO TO 440
* for pion+D(N*) now using the subroutine crpd to 
* (1) check kaon production or pion reabsorption 
* (2) change the particle label and set the new 
*     momentum of L/S+K final state
96       continue
        CALL CRPD(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK,xkaon0,xkaon,Xphi,xphin)

* kaon production
csp05/16/01
       IF(IBLOCK.EQ.7) then
          LPN=LPN+1
       elseIF(IBLOCK.EQ.-7) then
       endif
csp05/16/01 end
* pion obserption
       if(iblock.eq.80) lpdr=lpdr+1
       em1=e(i1)
       em2=e(i2)
       GO TO 440
* CALCULATE KAON PRODUCTION PROBABILITY FROM PION + N COLLISIONS
C        IF(SRT.GT.1.615)THEN
C        CALL PKAON(SRT,XXp,PK)
C        TKAON(7)=TKAON(7)+PK 
C        EKAON(7,ISS)=EKAON(7,ISS)+1
c        CALL KSPEC1(SRT,PK)
C        call LK(3,srt,iseed,pk)
C        ENDIF
* negelecting the pauli blocking at high energies

101       continue
        IF(E(I2).EQ.0.)GO TO 600
        IF(E(I1).EQ.0.)GO TO 800
* IF NUCLEON+BARYON RESONANCE COLLISIONS
44      CONTINUE
* CALCULATE THE TOTAL CROSS SECTION OF NUCLEON+ BARYON RESONANCE COLLISION
* WE ASSUME THAT THE ELASTIC CROSS SECTION IS THE SAME AS NUCLEON+NUCLEON
* COM: WE USE THE PARAMETERISATION BY CUGNON FOR LOW ENERGIES
*      AND THE PARAMETERIZATIONS FROM CERN DATA BOOK FOR HIGHER 
*      ENERGIES. THE CUTOFF FOR THE TOTAL CROSS SECTION IS 55 MB 
       cutoff=em1+em2+0.02
       IF(SRT.LE.CUTOFF)GO TO 400
        IF(SRT.GT.2.245)THEN
       SIGNN=PP2(SRT)
       ELSE
        SIGNN = 35.0 / (1. + (SRT - CUTOFF) * 100.0)  +  20.0
       ENDIF 
        call XND(pcx,pcy,pcz,srt,I1,I2,xinel,
     &               sigk,xsk1,xsk2,xsk3,xsk4,xsk5)
       sig=signn+xinel
* For nucleon+baryon resonance collision, the minimum cms**2 energy is
        EC=(EM1+EM2+0.02)**2
* CHECK THE DISTENCE BETWEEN THE TWO PARTICLES
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ

clin-6/2008 Deuteron production:
        ianti=0
        if(lb(i1).lt.0 .and. lb(i2).lt.0) ianti=1
        call sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
        sig=sig+sdprod
clin-6/2008 perturbative treatment of deuterons:
        ipdflag=0
        if(idpert.eq.1) then
           ipert1=1
           sigr0=sig
           dspert=sqrt(sigr0/pi/10.)
           dsrpert=dspert+0.1
           CALL DISTCE(I1,I2,dsrpert,dspert,DT,EC,SRT,IC,
     1          PX1CM,PY1CM,PZ1CM)
           IF(IC.EQ.-1) GO TO 363
           signn0=0.
           CALL CRND(IRUN,PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     &  IBLOCK,SIGNN0,SIGr0,sigk,xsk1,xsk2,xsk3,xsk4,xsk5,NT,ipert1)
c     &  IBLOCK,SIGNN,SIG,sigk,xsk1,xsk2,xsk3,xsk4,xsk5)
           ipdflag=1
 363       continue
           ipert1=0
        endif
        if(idpert.eq.2) ipert1=1
c
        DS=SQRT(SIG/(10.*PI))
        DELTAR=DS+0.1
        CALL DISTCE(I1,I2,DELTAR,DS,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
c        IF(IC.EQ.-1)GO TO 400
        IF(IC.EQ.-1) then
           if(ipdflag.eq.1) iblock=501
           GO TO 400
        endif

        ekaon(3,iss)=ekaon(3,iss)+1
* CALCULATE KAON PRODUCTION PROBABILITY FROM NUCLEON + BARYON RESONANCE 
* COLLISIONS
        go to 361

* CHECK WHAT KIND OF COLLISION HAS HAPPENED
 361    continue 
        CALL CRND(IRUN,PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     &     IBLOCK,SIGNN,SIG,sigk,xsk1,xsk2,xsk3,xsk4,xsk5,NT,ipert1)
c     &  IBLOCK,SIGNN,SIG,sigk,xsk1,xsk2,xsk3,xsk4,xsk5)
        IF(iblock.eq.0.and.ipdflag.eq.1) iblock=501
        IF(IBLOCK.EQ.11)THEN
           LNDK=LNDK+1
           GO TO 400
c        elseIF(IBLOCK.EQ.-11) then
        elseIF(IBLOCK.EQ.-11.or.iblock.eq.501) then
           GO TO 400
        ENDIF
        if(iblock .eq. 222)then
c    !! sp12/17/01 
           GO TO 400
        ENDIF
        em1=e(i1)
        em2=e(i2)
        GO TO 440
* IF NUCLEON+NUCLEON OR BARYON RESONANCE+BARYON RESONANCE COLLISIONS
4       CONTINUE
* PREPARE THE EALSTIC CROSS SECTION FOR BARYON+BARYON COLLISIONS
* COM: WE USE THE PARAMETERISATION BY CUGNON FOR SRT LEQ 2.0 GEV
*      AND THE PARAMETERIZATIONS FROM CERN DATA BOOK FOR HIGHER 
*      ENERGIES. THE CUTOFF FOR THE TOTAL CROSS SECTION IS 55 MB 
*      WITH LOW-ENERGY-CUTOFF
        CUTOFF=em1+em2+0.14
* AT HIGH ENERGIES THE ISOSPIN DEPENDENCE IS NEGLIGIBLE
* THE TOTAL CROSS SECTION IS TAKEN AS THAT OF THE PP 
* ABOVE E_KIN=800 MEV, WE USE THE ISOSPIN INDEPENDNET XSECTION
        IF(SRT.GT.2.245)THEN
           SIG=ppt(srt)
           SIGNN=SIG-PP1(SRT)
        ELSE
* AT LOW ENERGIES THE ISOSPIN DEPENDENCE FOR NN COLLISION IS STRONG
           SIG=XPP(SRT)
           IF(ZET(LB(I1))*ZET(LB(I2)).LE.0)SIG=XNP(SRT)
           IF(ZET(LB(I1))*ZET(LB(I2)).GT.0)SIG=XPP(SRT)
           IF(ZET(LB(I1)).EQ.0.
     &          AND.ZET(LB(I2)).EQ.0)SIG=XPP(SRT)
           if((lb(i1).eq.-1.and.lb(i2).eq.-2) .or.
     &          (lb(i2).eq.-1.and.lb(i1).eq.-2))sig=xnp(srt)
*     WITH LOW-ENERGY-CUTOFF
           IF (SRT .LT. 1.897) THEN
              SIGNN = SIG
           ELSE 
              SIGNN = 35.0 / (1. + (SRT - 1.897) * 100.0)  +  20.0
           ENDIF
        ENDIF 
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
clin-5/2008 Deuteron production cross sections were not included 
c     in the previous parameterized inelastic cross section of NN collisions  
c     (SIGinel=SIG-SIGNN), so they are added here:
        ianti=0
        if(lb(i1).lt.0 .and. lb(i2).lt.0) ianti=1
        call sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
        sig=sig+sdprod
c
clin-5/2008 perturbative treatment of deuterons:
        ipdflag=0
        if(idpert.eq.1) then
c     For idpert=1: ipert1=1 means we will first treat deuteron perturbatively,
c     then we set ipert1=0 to treat regular NN or NbarNbar collisions including
c     the regular deuteron productions.
c     ipdflag=1 means perturbative deuterons are produced here:
           ipert1=1
           EC=2.012**2
c     Use the same cross section for NN/NNBAR collisions 
c     to trigger perturbative production
           sigr0=sig
c     One can also trigger with X*sbbdm() so the weight will not be too small;
c     but make sure to limit the maximum trigger Xsec:
c           sigr0=sdprod*25.
c           if(sigr0.ge.100.) sigr0=100.
           dspert=sqrt(sigr0/pi/10.)
           dsrpert=dspert+0.1
           CALL DISTCE(I1,I2,dsrpert,dspert,DT,EC,SRT,IC,
     1          PX1CM,PY1CM,PZ1CM)
           IF(IC.EQ.-1) GO TO 365
           signn0=0.
           CALL CRNN(IRUN,PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK,
     1          NTAG,signn0,sigr0,NT,ipert1)
           ipdflag=1
 365       continue
           ipert1=0
        endif
        if(idpert.eq.2) ipert1=1
c
clin-5/2008 in case perturbative deuterons are produced for idpert=1:
c        IF(SIGNN.LE.0)GO TO 400
        IF(SIGNN.LE.0) then
           if(ipdflag.eq.1) iblock=501
           GO TO 400
        endif
c
        EC=3.59709
        ds=sqrt(sig/pi/10.)
        dsr=ds+0.1
        IF((E(I1).GE.1.).AND.(e(I2).GE.1.))EC=4.75
        CALL DISTCE(I1,I2,dsr,ds,DT,EC,SRT,IC,
     1       PX1CM,PY1CM,PZ1CM)
clin-5/2008 in case perturbative deuterons are produced above:
c        IF(IC.EQ.-1) GO TO 400
        IF(IC.EQ.-1) then
           if(ipdflag.eq.1) iblock=501
           GO TO 400
        endif
c
* CALCULATE KAON PRODUCTION PROBABILITY FROM NUCLEON+NUCLEON OR 
* RESONANCE+RESONANCE COLLISIONS
        go to 362

C CHECK WHAT KIND OF COLLISION HAS HAPPENED 
 362    ekaon(1,iss)=ekaon(1,iss)+1
        CALL CRNN(IRUN,PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK,
     1       NTAG,SIGNN,SIG,NT,ipert1)
clin-5/2008 give iblock # in case pert deuterons are produced for idpert=1:
        IF(iblock.eq.0.and.ipdflag.eq.1) iblock=501
clin-5/2008 add iblock # for deuteron formation:
c        IF(IBLOCK.EQ.4.OR.IBLOCK.Eq.9.or.iblock.ge.44.OR.IBLOCK.EQ.-9
c     &       .or.iblock.eq.222)THEN
        IF(IBLOCK.EQ.4.OR.IBLOCK.Eq.9.or.iblock.ge.44.OR.IBLOCK.EQ.-9
     &       .or.iblock.eq.222.or.iblock.eq.501)THEN
c
c     !! sp12/17/01 above
* momentum of the three particles in the final state have been calculated
* in the crnn, go out of the loop
           LCOLL=LCOLL+1
           if(iblock.eq.4)then
              LDIRT=LDIRT+1
           elseif(iblock.eq.44)then
              LDdrho=LDdrho+1
           elseif(iblock.eq.45)then
              Lnnrho=Lnnrho+1
           elseif(iblock.eq.46)then
              Lnnom=Lnnom+1
           elseif(iblock .eq. 222)then
           elseIF(IBLOCK.EQ.9) then
              LNNK=LNNK+1
           elseIF(IBLOCK.EQ.-9) then
           endif
           GO TO 400
        ENDIF

        em1=e(i1)
        em2=e(i2)
        GO TO 440
clin-8/2008 B+B->Deuteron+Meson over
c
clin-8/2008 Deuteron+Meson->B+B collisions:
 505    continue
        ianti=0
        if(lb(i1).lt.0 .or. lb(i2).lt.0) ianti=1
        call sdmbb(SRT,sdm,ianti)
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
c     minimum srt**2, note a 2.012GeV lower cutoff is used in N+N->Deuteron+pi:
        EC=2.012**2
        ds=sqrt(sdm/31.4)
        dsr=ds+0.1
        CALL DISTCE(I1,I2,dsr,ds,DT,EC,SRT,IC,PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
        CALL crdmbb(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK,
     1       NTAG,sdm,NT,ianti)
        LCOLL=LCOLL+1
        GO TO 400
clin-8/2008 Deuteron+Meson->B+B collisions over
c
clin-9/2008 Deuteron+Baryon elastic collisions:
 506    continue
        ianti=0
        if(lb(i1).lt.0 .or. lb(i2).lt.0) ianti=1
        call sdbelastic(SRT,sdb)
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
c     minimum srt**2, note a 2.012GeV lower cutoff is used in N+N->Deuteron+pi:
        EC=2.012**2
        ds=sqrt(sdb/31.4)
        dsr=ds+0.1
        CALL DISTCE(I1,I2,dsr,ds,DT,EC,SRT,IC,PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
        CALL crdbel(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK,
     1       NTAG,sdb,NT,ianti)
        LCOLL=LCOLL+1
        GO TO 400
clin-9/2008 Deuteron+Baryon elastic collisions over
c
* IF BARYON RESONANCE+BARYON RESONANCE COLLISIONS
 444    CONTINUE
* PREPARE THE EALSTIC CROSS SECTION FOR BARYON+BARYON COLLISIONS
       CUTOFF=em1+em2+0.02
* AT HIGH ENERGIES THE ISOSPIN DEPENDENCE IS NEGLIGIBLE
* THE TOTAL CROSS SECTION IS TAKEN AS THAT OF THE PP 
       IF(SRT.LE.CUTOFF)GO TO 400
        IF(SRT.GT.2.245)THEN
       SIGNN=PP2(SRT)
       ELSE
        SIGNN = 35.0 / (1. + (SRT - CUTOFF) * 100.0)  +  20.0
       ENDIF 
       IF(SIGNN.LE.0)GO TO 400
      CALL XDDIN(PCX,PCY,PCZ,SRT,I1,I2,
     &XINEL,SIGK,XSK1,XSK2,XSK3,XSK4,XSK5)
       SIG=SIGNN+XINEL
       EC=(EM1+EM2+0.02)**2
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ

clin-6/2008 Deuteron production:
        ianti=0
        if(lb(i1).lt.0 .and. lb(i2).lt.0) ianti=1
        call sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
        sig=sig+sdprod
clin-6/2008 perturbative treatment of deuterons:
        ipdflag=0
        if(idpert.eq.1) then
           ipert1=1
           sigr0=sig
           dspert=sqrt(sigr0/pi/10.)
           dsrpert=dspert+0.1
           CALL DISTCE(I1,I2,dsrpert,dspert,DT,EC,SRT,IC,
     1          PX1CM,PY1CM,PZ1CM)
           IF(IC.EQ.-1) GO TO 367
           signn0=0.
           CALL CRDD(IRUN,PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1          IBLOCK,NTAG,SIGNN0,SIGr0,NT,ipert1)
c     1          IBLOCK,NTAG,SIGNN,SIG)
           ipdflag=1
 367       continue
           ipert1=0
        endif
        if(idpert.eq.2) ipert1=1
c
        ds=sqrt(sig/31.4)
        dsr=ds+0.1
        CALL DISTCE(I1,I2,dsr,ds,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
c        IF(IC.EQ.-1) GO TO 400
        IF(IC.EQ.-1) then
           if(ipdflag.eq.1) iblock=501
           GO TO 400
        endif

* CALCULATE KAON PRODUCTION PROBABILITY FROM NUCLEON+NUCLEON OR 
* RESONANCE+RESONANCE COLLISIONS
       go to 364

C CHECK WHAT KIND OF COLLISION HAS HAPPENED 
364       ekaon(2,iss)=ekaon(2,iss)+1
* for resonance+resonance
clin-6/2008:
        CALL CRDD(IRUN,PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK,NTAG,SIGNN,SIG,NT,ipert1)
c     1  IBLOCK,NTAG,SIGNN,SIG)
        IF(iblock.eq.0.and.ipdflag.eq.1) iblock=501
c
        IF(iabs(IBLOCK).EQ.10)THEN
* momentum of the three particles in the final state have been calculated
* in the crnn, go out of the loop
           LCOLL=LCOLL+1
           IF(IBLOCK.EQ.10)THEN
              LDDK=LDDK+1
           elseIF(IBLOCK.EQ.-10) then
           endif
           GO TO 400
        ENDIF
clin-6/2008
c        if(iblock .eq. 222)then
        if(iblock .eq. 222.or.iblock.eq.501)then
c    !! sp12/17/01 
           GO TO 400
        ENDIF
        em1=e(i1)
        em2=e(i2)
        GO TO 440
* FOR PION+PION,pion+eta, eta+eta and rho(omega)+pion(rho,omega) or eta 
777       CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
* energy thresh for collisions
       ec0=em1+em2+0.02
       IF(SRT.LE.ec0)GO TO 400
       ec=(em1+em2+0.02)**2
* we negelect the elastic collision between mesons except that betwen
* two pions because of the lack of information about these collisions
* However, we do let them to collide inelastically to produce kaons
clin-8/15/02       ppel=1.e-09
       ppel=20.
        ipp=1
       if(lb1.lt.3.or.lb1.gt.5.or.lb2.lt.3.or.lb2.gt.5)go to 778       
       CALL PPXS(LB1,LB2,SRT,PPSIG,spprho,IPP)
       ppel=ppsig
778       ppink=pipik(srt)

* pi+eta and eta+eta are assumed to be the same as pipik( for pi+pi -> K+K-) 
* estimated from Ko's paper:
        ppink = 2.0 * ppink
       if(lb1.ge.25.and.lb2.ge.25) ppink=rrkk

clin-2/13/03 include omega the same as rho, eta the same as pi:
c        if(((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.25.and.lb2.le.27))
c     1  .or.((lb2.ge.3.and.lb2.le.5).and.(lb1.ge.25.and.lb1.le.27)))
        if( ( (lb1.eq.0.or.(lb1.ge.3.and.lb1.le.5))
     1       .and.(lb2.ge.25.and.lb2.le.28))
     2       .or. ( (lb2.eq.0.or.(lb2.ge.3.and.lb2.le.5))
     3       .and.(lb1.ge.25.and.lb1.le.28))) then
           ppink=0.
           if(srt.ge.(aka+aks)) ppink = prkk
        endif

c pi pi <-> rho rho:
        call spprr(lb1,lb2,srt)
clin-4/03/02 pi pi <-> eta eta:
        call sppee(lb1,lb2,srt)
clin-4/03/02 pi pi <-> pi eta:
        call spppe(lb1,lb2,srt)
clin-4/03/02 rho pi <-> rho eta:
        call srpre(lb1,lb2,srt)
clin-4/03/02 omega pi <-> omega eta:
        call sopoe(lb1,lb2,srt)
clin-4/03/02 rho rho <-> eta eta:
        call srree(lb1,lb2,srt)

        ppinnb=0.
        if(srt.gt.thresh(1)) then
           call getnst(srt)
           if(lb1.ge.3.and.lb1.le.5.and.lb2.ge.3.and.lb2.le.5) then
              ppinnb=ppbbar(srt)
           elseif((lb1.ge.3.and.lb1.le.5.and.lb2.ge.25.and.lb2.le.27)
     1 .or.(lb2.ge.3.and.lb2.le.5.and.lb1.ge.25.and.lb1.le.27)) then
              ppinnb=prbbar(srt)
           elseif(lb1.ge.25.and.lb1.le.27
     1             .and.lb2.ge.25.and.lb2.le.27) then
              ppinnb=rrbbar(srt)
           elseif((lb1.ge.3.and.lb1.le.5.and.lb2.eq.28)
     1             .or.(lb2.ge.3.and.lb2.le.5.and.lb1.eq.28)) then
              ppinnb=pobbar(srt)
           elseif((lb1.ge.25.and.lb1.le.27.and.lb2.eq.28)
     1             .or.(lb2.ge.25.and.lb2.le.27.and.lb1.eq.28)) then
              ppinnb=robbar(srt)
           elseif(lb1.eq.28.and.lb2.eq.28) then
              ppinnb=oobbar(srt)
           else
              if(lb1.ne.0.and.lb2.ne.0) 
     1             write(6,*) 'missed MM lb1,lb2=',lb1,lb2
           endif
        endif
        ppin=ppink+ppinnb+pprr+ppee+pppe+rpre+xopoe+rree

* check if a collision can happen
       if((ppel+ppin).le.0.01)go to 400
       DSPP=SQRT((ppel+ppin)/31.4)
       dsppr=dspp+0.1
        CALL DISTCE(I1,I2,dsppr,DSPP,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
       if(ppel.eq.0)go to 400
* the collision can happen
* check what kind collision has happened
       ekaon(5,iss)=ekaon(5,iss)+1
        CALL CRPP(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK,ppel,ppin,spprho,ipp)

* rho formation, go to 400
c       if(iblock.eq.666)go to 600
       if(iblock.eq.666)go to 555
       if(iblock.eq.6)LPP=LPP+1
       if(iblock.eq.66)then
          LPPk=LPPk+1
       elseif(iblock.eq.366)then
          LPPk=LPPk+1
       elseif(iblock.eq.367)then
          LPPk=LPPk+1
       endif
       em1=e(i1)
       em2=e(i2)
       go to 440

* In this block we treat annihilations of
clin-9/28/00* an anti-nucleon and a baryon or baryon resonance  
* an anti-baryon and a baryon (including resonances)
2799        CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
clin assume the same cross section (as a function of sqrt s) as for PPbar:

clin-ctest annih maximum
c        DSppb=SQRT(amin1(xppbar(srt),30.)/PI/10.)
       DSppb=SQRT(xppbar(srt)/PI/10.)
       dsppbr=dsppb+0.1
        CALL DISTCE(I1,I2,dsppbr,DSppb,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
        CALL Crppba(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK)
       em1=e(i1)
       em2=e(i2)
       go to 440
c
3555    PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
       DSkk=SQRT(SIG/PI/10.)
       dskk0=dskk+0.1
        CALL DISTCE(I1,I2,dskk0,DSkk,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
        CALL Crlaba(PX1CM,PY1CM,PZ1CM,SRT,brel,brsgm,
     &                  I1,I2,nt,IBLOCK,nchrg,icase)
       em1=e(i1)
       em2=e(i2)
       go to 440
*
c perturbative production of cascade and omega
3455    PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        call pertur(PX1CM,PY1CM,PZ1CM,SRT,IRUN,I1,I2,nt,kp,icontp)
        if(icontp .eq. 0)then
c     inelastic collisions:
         em1 = e(i1)
         em2 = e(i2)
         iblock = 727
          go to 440
        endif
c     elastic collisions:
        if (e(i1) .eq. 0.) go to 800
        if (e(i2) .eq. 0.) go to 600
        go to 400
*
c* phi + N --> pi+N(D),  N(D,N*)+N(D,N*),  K+ +La
c* phi + D --> pi+N(D)
7222        CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
        CALL XphiB(LB1, LB2, EM1, EM2, SRT,
     &             XSK1, XSK2, XSK3, XSK4, XSK5, SIGP)
       DSkk=SQRT(SIGP/PI/10.)
       dskk0=dskk+0.1
        CALL DISTCE(I1,I2,dskk0,DSkk,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
        CALL CRPHIB(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     &     XSK1, XSK2, XSK3, XSK4, XSK5, SIGP, IBLOCK)
       em1=e(i1)
       em2=e(i2)
       go to 440
*
c* phi + M --> K+ + K* .....
7444        CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
        CALL PHIMES(I1, I2, SRT, XSK1, XSK2, XSK3, XSK4, XSK5,
     1     XSK6, XSK7, SIGPHI)
       DSkk=SQRT(SIGPHI/PI/10.)
       dskk0=dskk+0.1
        CALL DISTCE(I1,I2,dskk0,DSkk,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
c*---
        PZRT = p(3,i1)+p(3,i2)
        ER1 = sqrt( p(1,i1)**2+p(2,i1)**2+p(3,i1)**2+E(i1)**2 )
        ER2 = sqrt( p(1,i2)**2+p(2,i2)**2+p(3,i2)**2+E(i2)**2 )
        ERT = ER1+ER2
        yy = 0.5*log( (ERT+PZRT)/(ERT-PZRT) )
c*------
        CALL CRPHIM(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     &  XSK1, XSK2, XSK3, XSK4, XSK5, XSK6, SIGPHI, IKKG, IKKL, IBLOCK)
       em1=e(i1)
       em2=e(i2)
       go to 440
c
c lambda-N elastic xsection, Li & Ko, PRC 54(1996)1897.
 7799    CONTINUE
         PX1CM=PCX
         PY1CM=PCY
         PZ1CM=PCZ
         EC=(em1+em2+0.02)**2
         call lambar(i1,i2,srt,siglab)
        DShn=SQRT(siglab/PI/10.)
        dshnr=dshn+0.1
         CALL DISTCE(I1,I2,dshnr,DShn,DT,EC,SRT,IC,
     1    PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
         CALL Crhb(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)
        em1=e(i1)
        em2=e(i2)
        go to 440
c
c* K+ + La(Si) --> Meson + B
c* K- + La(Si)-bar --> Meson + B-bar
5699        CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
        CALL XKHYPE(I1, I2, SRT, XKY1, XKY2, XKY3, XKY4, XKY5,
     &     XKY6, XKY7, XKY8, XKY9, XKY10, XKY11, XKY12, XKY13,
     &     XKY14, XKY15, XKY16, XKY17, SIGK)
       DSkk=SQRT(sigk/PI)
       dskk0=dskk+0.1
        CALL DISTCE(I1,I2,dskk0,DSkk,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
c
       if(lb(i1).eq.23 .or. lb(i2).eq.23)then
             IKMP = 1
        else
             IKMP = -1
        endif
        CALL Crkhyp(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     &     XKY1, XKY2, XKY3, XKY4, XKY5,
     &     XKY6, XKY7, XKY8, XKY9, XKY10, XKY11, XKY12, XKY13,
     &     XKY14, XKY15, XKY16, XKY17, SIGK, IKMP,
     1  IBLOCK)
       em1=e(i1)
       em2=e(i2)
       go to 440
c khyperon end
*
csp11/03/01 La/Si-bar + N --> pi + K+
c  La/Si + N-bar --> pi + K-
5999     CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
        sigkp = 15.
c      if((lb1.ge.14.and.lb1.le.17)
c     &    .or.(lb2.ge.14.and.lb2.le.17))sigkp=10.
        DSkk=SQRT(SIGKP/PI/10.)
        dskk0=dskk+0.1
        CALL DISTCE(I1,I2,dskk0,DSkk,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
c
        CALL CRLAN(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)
        em1=e(i1)
        em2=e(i2)
        go to 440
c
c*
* K(K*) + K(K*) --> phi + pi(rho,omega)
8699     CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
*  CALL CROSSKKPHI(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)  used for KK*->phi+rho

         CALL Crkphi(PX1CM,PY1CM,PZ1CM,EC,SRT,IBLOCK,
     &                  emm1,emm2,lbp1,lbp2,I1,I2,ikk,icase,rrkk,prkk)
         if(icase .eq. 0) then
            iblock=0
            go to 400
         endif

c*---
         if(lbp1.eq.29.or.lbp2.eq.29) then
        PZRT = p(3,i1)+p(3,i2)
        ER1 = sqrt( p(1,i1)**2+p(2,i1)**2+p(3,i1)**2+E(i1)**2 )
        ER2 = sqrt( p(1,i2)**2+p(2,i2)**2+p(3,i2)**2+E(i2)**2 )
        ERT = ER1+ER2
        yy = 0.5*log( (ERT+PZRT)/(ERT-PZRT) )
c*------
             iblock = 222
             ntag = 0
          endif

             LB(I1) = lbp1
             LB(I2) = lbp2
             E(I1) = emm1
             E(I2) = emm2
             em1=e(i1)
             em2=e(i2)
             go to 440
c*
* rho(omega) + K(K*)  --> phi + K(K*)
8799     CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
*  CALL CROSSKKPHI(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,IBLOCK)  used for KK*->phi+rho
         CALL Crksph(PX1CM,PY1CM,PZ1CM,EC,SRT,
     &       emm1,emm2,lbp1,lbp2,I1,I2,ikkg,ikkl,iblock,icase,srhoks)
         if(icase .eq. 0) then
            iblock=0
            go to 400
         endif
c
         if(lbp1.eq.29.or.lbp2.eq.20) then
c*---
        PZRT = p(3,i1)+p(3,i2)
        ER1 = sqrt( p(1,i1)**2+p(2,i1)**2+p(3,i1)**2+E(i1)**2 )
        ER2 = sqrt( p(1,i2)**2+p(2,i2)**2+p(3,i2)**2+E(i2)**2 )
        ERT = ER1+ER2
        yy = 0.5*log( (ERT+PZRT)/(ERT-PZRT) )
          endif

             LB(I1) = lbp1
             LB(I2) = lbp2
             E(I1) = emm1
             E(I2) = emm2
             em1=e(i1)
             em2=e(i2)
             go to 440

* for kaon+baryon scattering, using a constant xsection of 10 mb.
888       CONTINUE
        PX1CM=PCX
        PY1CM=PCY
        PZ1CM=PCZ
        EC=(em1+em2+0.02)**2
         sig = 10.
         if(iabs(lb1).eq.14.or.iabs(lb2).eq.14 .or.
     &      iabs(lb1).eq.30.or.iabs(lb2).eq.30)sig=20.
         if(lb1.eq.29.or.lb2.eq.29)sig=5.0

       DSkn=SQRT(sig/PI/10.)
       dsknr=dskn+0.1
        CALL DISTCE(I1,I2,dsknr,DSkn,DT,EC,SRT,IC,
     1  PX1CM,PY1CM,PZ1CM)
        IF(IC.EQ.-1) GO TO 400
        CALL Crkn(PX1CM,PY1CM,PZ1CM,SRT,I1,I2,
     1  IBLOCK)
       em1=e(i1)
       em2=e(i2)
       go to 440
***

 440    CONTINUE
*                IBLOCK = 0 ; NOTHING HAS HAPPENED
*                IBLOCK = 1 ; ELASTIC N-N COLLISION
*                IBLOCK = 2 ; N + N -> N + DELTA
*                IBLOCK = 3 ; N + DELTA -> N + N
*                IBLOCK = 4 ; N + N -> d + d + PION,DIRECT PROCESS
*               IBLOCK = 5 ; D(N*)+D(N*) COLLISIONS
*                IBLOCK = 6 ; PION+PION COLLISIONS
*                iblock = 7 ; pion+nucleon-->l/s+kaon
*               iblock =77;  pion+nucleon-->delta+pion
*               iblock = 8 ; kaon+baryon rescattering
*                IBLOCK = 9 ; NN-->KAON+X
*                IBLOCK = 10; DD-->KAON+X
*               IBLOCK = 11; ND-->KAON+X
cbali2/1/99
*                
*           iblock   - 1902 annihilation-->pion(+)+pion(-)   (2 pion)
*           iblock   - 1903 annihilation-->pion(+)+rho(-)    (3 pion)
*           iblock   - 1904 annihilation-->rho(+)+rho(-)     (4 pion)
*           iblock   - 1905 annihilation-->rho(0)+omega      (5 pion)
*           iblock   - 1906 annihilation-->omega+omega       (6 pion)
cbali3/5/99
*           iblock   - 1907 K+K- to pi+pi-
cbali3/5/99 end
cbz3/9/99 khyperon
*           iblock   - 1908 K+Y -> piN
cbz3/9/99 khyperon end
cbali2/1/99end

clin-9/28/00 Processes: m(pi rho omega)+m(pi rho omega)
c     to anti-(p n D N*1 N*2)+(p n D N*1 N*2):
*           iblock   - 1801  mm -->pbar p 
*           iblock   - 18021 mm -->pbar n 
*           iblock   - 18022 mm -->nbar p 
*           iblock   - 1803  mm -->nbar n 
*           iblock   - 18041 mm -->pbar Delta 
*           iblock   - 18042 mm -->anti-Delta p
*           iblock   - 18051 mm -->nbar Delta 
*           iblock   - 18052 mm -->anti-Delta n
*           iblock   - 18061 mm -->pbar N*(1400) 
*           iblock   - 18062 mm -->anti-N*(1400) p
*           iblock   - 18071 mm -->nbar N*(1400)
*           iblock   - 18072 mm -->anti-N*(1400) n
*           iblock   - 1808  mm -->anti-Delta Delta 
*           iblock   - 18091 mm -->pbar N*(1535)
*           iblock   - 18092 mm -->anti-N*(1535) p
*           iblock   - 18101 mm -->nbar N*(1535)
*           iblock   - 18102 mm -->anti-N*(1535) n
*           iblock   - 18111 mm -->anti-Delta N*(1440)
*           iblock   - 18112 mm -->anti-N*(1440) Delta
*           iblock   - 18121 mm -->anti-Delta N*(1535)
*           iblock   - 18122 mm -->anti-N*(1535) Delta
*           iblock   - 1813  mm -->anti-N*(1440) N*(1440)
*           iblock   - 18141 mm -->anti-N*(1440) N*(1535)
*           iblock   - 18142 mm -->anti-N*(1535) N*(1440)
*           iblock   - 1815  mm -->anti-N*(1535) N*(1535)
clin-9/28/00-end

clin-10/08/00 Processes: pi pi <-> rho rho
*           iblock   - 1850  pi pi -> rho rho
*           iblock   - 1851  rho rho -> pi pi
clin-10/08/00-end

clin-08/14/02 Processes: pi pi <-> eta eta
*           iblock   - 1860  pi pi -> eta eta
*           iblock   - 1861  eta eta -> pi pi
* Processes: pi pi <-> pi eta
*           iblock   - 1870  pi pi -> pi eta
*           iblock   - 1871  pi eta -> pi pi
* Processes: rho pi <-> rho eta
*           iblock   - 1880  pi pi -> pi eta
*           iblock   - 1881  pi eta -> pi pi
* Processes: omega pi <-> omega eta
*           iblock   - 1890  pi pi -> pi eta
*           iblock   - 1891  pi eta -> pi pi
* Processes: rho rho <-> eta eta
*           iblock   - 1895  rho rho -> eta eta
*           iblock   - 1896  eta eta -> rho rho
clin-08/14/02-end

clin-11/07/00 Processes: 
*           iblock   - 366  pi rho -> K* Kbar or K*bar K
*           iblock   - 466  pi rho <- K* Kbar or K*bar K

clin-9/2008 Deuteron:
*           iblock   - 501  B+B -> Deuteron+Meson
*           iblock   - 502  Deuteron+Meson -> B+B
*           iblock   - 503  Deuteron+Baryon elastic
*           iblock   - 504  Deuteron+Meson elastic
c
                 IF(IBLOCK.EQ.0)        GOTO 400
*COM: FOR DIRECT PROCESS WE HAVE TREATED THE PAULI BLOCKING AND FIND
*     THE MOMENTUM OF PARTICLES IN THE ''LAB'' FRAME. SO GO TO 400
* A COLLISION HAS TAKEN PLACE !!
              LCOLL = LCOLL +1
* WAS COLLISION PAULI-FORBIDEN? IF YES, NTAG = -1
              NTAG = 0
*
*             LORENTZ-TRANSFORMATION INTO CMS FRAME
              E1CM    = SQRT (EM1**2 + PX1CM**2 + PY1CM**2 + PZ1CM**2)
              P1BETA  = PX1CM*BETAX + PY1CM*BETAY + PZ1CM*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1I1 = BETAX * TRANSF + PX1CM
              Pt2I1 = BETAY * TRANSF + PY1CM
              Pt3I1 = BETAZ * TRANSF + PZ1CM
* negelect the pauli blocking at high energies
              go to 90002

clin-10/25/02-comment out following, since there is no path to it:
c*CHECK IF PARTICLE #1 IS PAULI BLOCKED
c              CALL PAULat(I1,occup)
c              if (RANART(NSEED) .lt. occup) then
c                ntag = -1
c              else
c                ntag = 0
c              end if
clin-10/25/02-end

90002              continue
*IF PARTICLE #1 IS NOT PAULI BLOCKED
c              IF (NTAG .NE. -1) THEN
                E2CM    = SQRT (EM2**2 + PX1CM**2 + PY1CM**2 + PZ1CM**2)
                TRANSF  = GAMMA * (-GAMMA*P1BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF - PX1CM
                Pt2I2 = BETAY * TRANSF - PY1CM
                Pt3I2 = BETAZ * TRANSF - PZ1CM
              go to 90003

clin-10/25/02-comment out following, since there is no path to it:
c*CHECK IF PARTICLE #2 IS PAULI BLOCKED
c                CALL PAULat(I2,occup)
c                if (RANART(NSEED) .lt. occup) then
c                  ntag = -1
c                else
c                  ntag = 0
c                end if
cc              END IF
c* IF COLLISION IS BLOCKED,RESTORE THE MOMENTUM,MASSES
c* AND LABELS OF I1 AND I2
cc             IF (NTAG .EQ. -1) THEN
c                LBLOC  = LBLOC + 1
c                P(1,I1) = PX1
c                P(2,I1) = PY1
c                P(3,I1) = PZ1
c                P(1,I2) = PX2
c                P(2,I2) = PY2
c                P(3,I2) = PZ2
c                E(I1)   = EM1
c                E(I2)   = EM2
c                LB(I1)  = LB1
c                LB(I2)  = LB2
cc              ELSE
clin-10/25/02-end

90003           IF(IBLOCK.EQ.1) LCNNE=LCNNE+1
              IF(IBLOCK.EQ.5) LDD=LDD+1
                if(iblock.eq.2) LCNND=LCNND+1
              IF(IBLOCK.EQ.8) LKN=LKN+1
                   if(iblock.eq.43) Ldou=Ldou+1
c                IF(IBLOCK.EQ.2) THEN
* CALCULATE THE AVERAGE SRT FOR N + N---> N + DELTA PROCESS
C                NODELT=NODELT+1
C                SUMSRT=SUMSRT+SRT
c                ENDIF
                IF(IBLOCK.EQ.3) LCNDN=LCNDN+1
* assign final momenta to particles while keep the leadng particle
* behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
              p(1,i1)=pt1i1
              p(2,i1)=pt2i1
              p(3,i1)=pt3i1
              p(1,i2)=pt1i2
              p(2,i2)=pt2i2
              p(3,i2)=pt3i2
C              else
C              p(1,i1)=pt1i2
C              p(2,i1)=pt2i2
C              p(3,i1)=pt3i2
C              p(1,i2)=pt1i1
C              p(2,i2)=pt2i1
C              p(3,i2)=pt3i1
C              endif
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
                EM1     = E(I1)
                EM2     = E(I2)
                LB1     = LB(I1)
                LB2     = LB(I2)
                ID(I1)  = 2
                ID(I2)  = 2
                E1      = SQRT( EM1**2 + PX1**2 + PY1**2 + PZ1**2 )
                ID1     = ID(I1)
              go to 90004
clin-10/25/02-comment out following, since there is no path to it:
c* change phase space density FOR NUCLEONS INVOLVED :
c* NOTE THAT f is the phase space distribution function for nucleons only
c                if ((abs(ix1).le.mx) .and. (abs(iy1).le.my) .and.
c     &              (abs(iz1).le.mz)) then
c                  ipx1p = nint(p(1,i1)/dpx)
c                  ipy1p = nint(p(2,i1)/dpy)
c                  ipz1p = nint(p(3,i1)/dpz)
c                  if ((ipx1p.ne.ipx1) .or. (ipy1p.ne.ipy1) .or.
c     &                (ipz1p.ne.ipz1)) then
c                    if ((abs(ipx1).le.mpx) .and. (abs(ipy1).le.my)
c     &                .and. (ipz1.ge.-mpz) .and. (ipz1.le.mpzp)
c     &                .AND. (AM1.LT.1.))
c     &                f(ix1,iy1,iz1,ipx1,ipy1,ipz1) =
c     &                f(ix1,iy1,iz1,ipx1,ipy1,ipz1) - 1.
c                    if ((abs(ipx1p).le.mpx) .and. (abs(ipy1p).le.my)
c     &                .and. (ipz1p.ge.-mpz).and. (ipz1p.le.mpzp)
c     &                .AND. (EM1.LT.1.))
c     &                f(ix1,iy1,iz1,ipx1p,ipy1p,ipz1p) =
c     &                f(ix1,iy1,iz1,ipx1p,ipy1p,ipz1p) + 1.
c                  end if
c                end if
c                if ((abs(ix2).le.mx) .and. (abs(iy2).le.my) .and.
c     &              (abs(iz2).le.mz)) then
c                  ipx2p = nint(p(1,i2)/dpx)
c                  ipy2p = nint(p(2,i2)/dpy)
c                  ipz2p = nint(p(3,i2)/dpz)
c                  if ((ipx2p.ne.ipx2) .or. (ipy2p.ne.ipy2) .or.
c     &                (ipz2p.ne.ipz2)) then
c                    if ((abs(ipx2).le.mpx) .and. (abs(ipy2).le.my)
c     &                .and. (ipz2.ge.-mpz) .and. (ipz2.le.mpzp)
c     &                .AND. (AM2.LT.1.))
c     &                f(ix2,iy2,iz2,ipx2,ipy2,ipz2) =
c     &                f(ix2,iy2,iz2,ipx2,ipy2,ipz2) - 1.
c                    if ((abs(ipx2p).le.mpx) .and. (abs(ipy2p).le.my)
c     &                .and. (ipz2p.ge.-mpz) .and. (ipz2p.le.mpzp)
c     &                .AND. (EM2.LT.1.))
c     &                f(ix2,iy2,iz2,ipx2p,ipy2p,ipz2p) =
c     &                f(ix2,iy2,iz2,ipx2p,ipy2p,ipz2p) + 1.
c                  end if
c                end if
clin-10/25/02-end

90004              continue
            AM1=EM1
            AM2=EM2
c            END IF


  400       CONTINUE
c
clin-6/10/03 skips the info output on resonance creations:
c            goto 550
cclin-4/30/03 study phi,K*,Lambda(1520) resonances at creation:
cc     note that no decays give these particles, so don't need to consider nnn:
c            if(iblock.ne.0.and.(lb(i1).eq.29.or.iabs(lb(i1)).eq.30
c     1           .or.lb(i2).eq.29.or.iabs(lb(i2)).eq.30
c     2           .or.lb1i.eq.29.or.iabs(lb1i).eq.30
c     3           .or.lb2i.eq.29.or.iabs(lb2i).eq.30)) then
c               lb1now=lb(i1)
c               lb2now=lb(i2)
cc
c               nphi0=0
c               nksp0=0
c               nksm0=0
cc               nlar0=0
cc               nlarbar0=0
c               if(lb1i.eq.29) then
c                  nphi0=nphi0+1
c               elseif(lb1i.eq.30) then
c                  nksp0=nksp0+1
c               elseif(lb1i.eq.-30) then
c                  nksm0=nksm0+1
c               endif
c               if(lb2i.eq.29) then
c                  nphi0=nphi0+1
c               elseif(lb2i.eq.30) then
c                  nksp0=nksp0+1
c               elseif(lb2i.eq.-30) then
c                  nksm0=nksm0+1
c               endif
cc
c               nphi=0
c               nksp=0
c               nksm=0
c               nlar=0
c               nlarbar=0
c               if(lb1now.eq.29) then
c                  nphi=nphi+1
c               elseif(lb1now.eq.30) then
c                  nksp=nksp+1
c               elseif(lb1now.eq.-30) then
c                  nksm=nksm+1
c               endif
c               if(lb2now.eq.29) then
c                  nphi=nphi+1
c               elseif(lb2now.eq.30) then
c                  nksp=nksp+1
c               elseif(lb2now.eq.-30) then
c                  nksm=nksm+1
c               endif
cc     
c               if(nphi.eq.2.or.nksp.eq.2.or.nksm.eq.2) then
c                  write(91,*) '2 same resonances in one reaction!'
c                  write(91,*) nphi,nksp,nksm,iblock
c               endif
c
cc     All reactions create or destroy no more than 1 these resonance,
cc     otherwise file "fort.91" warns us:
c               do 222 ires=1,3
c                  if(ires.eq.1.and.nphi.ne.nphi0) then
c                     idr=29
c                  elseif(ires.eq.2.and.nksp.ne.nksp0) then
c                     idr=30
c                  elseif(ires.eq.3.and.nksm.ne.nksm0) then
c                     idr=-30
c                  else
c                     goto 222
c                  endif
cctest off for resonance (phi, K*) studies:
cc               if(lb1now.eq.idr) then
cc       write(17,112) 'collision',lb1now,P(1,I1),P(2,I1),P(3,I1),e(I1),nt
cc               elseif(lb2now.eq.idr) then
cc       write(17,112) 'collision',lb2now,P(1,I2),P(2,I2),P(3,I2),e(I2),nt
cc               elseif(lb1i.eq.idr) then
cc       write(18,112) 'collision',lb1i,px1i,py1i,pz1i,em1i,nt
cc               elseif(lb2i.eq.idr) then
cc       write(18,112) 'collision',lb2i,px2i,py2i,pz2i,em2i,nt
cc               endif
c 222           continue
c
c            else
c            endif
cc 112        format(a10,I4,4(1x,f9.3),1x,I4)
c
clin-2/26/03 skips the check of energy conservation after each binary search:
c 550        goto 555
c            pxfin=0
c            pyfin=0
c            pzfin=0
c            efin=0
c            if(e(i1).ne.0.or.lb(i1).eq.10022) then
c               efin=efin+SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
c               pxfin=pxfin+P(1,I1)
c               pyfin=pyfin+P(2,I1)
c               pzfin=pzfin+P(3,I1)
c            endif
c            if(e(i2).ne.0.or.lb(i2).eq.10022) then
c               efin=efin+SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
c               pxfin=pxfin+P(1,I2)
c               pyfin=pyfin+P(2,I2)
c               pzfin=pzfin+P(3,I2)
c            endif
c            if((nnn-nnnini).ge.1) then
c               do imore=nnnini+1,nnn
c                  if(EPION(imore,IRUN).ne.0) then
c                     efin=efin+SQRT(EPION(imore,IRUN)**2
c     1                    +PPION(1,imore,IRUN)**2+PPION(2,imore,IRUN)**2
c     2                    +PPION(3,imore,IRUN)**2)
c                     pxfin=pxfin+PPION(1,imore,IRUN)
c                     pyfin=pyfin+PPION(2,imore,IRUN)
c                     pzfin=pzfin+PPION(3,imore,IRUN)
c                  endif
c               enddo
c            endif
c            devio=sqrt((pxfin-pxini)**2+(pyfin-pyini)**2
c     1           +(pzfin-pzini)**2+(efin-eini)**2)
cc
c            if(devio.ge.0.1) then
c               write(92,'a20,5(1x,i6),2(1x,f8.3)') 'iblock,lb,npi=',
c     1              iblock,lb1i,lb2i,lb(i1),lb(i2),e(i1),e(i2)
c               do imore=nnnini+1,nnn
c                  if(EPION(imore,IRUN).ne.0) then
c                     write(92,'a10,2(1x,i6)') 'ipi,lbm=',
c     1                    imore,LPION(imore,IRUN)
c                  endif
c               enddo
c               write(92,'a3,4(1x,f8.3)') 'I:',eini,pxini,pyini,pzini
c               write(92,'a3,5(1x,f8.3)') 
c     1              'F:',efin,pxfin,pyfin,pzfin,devio
c            endif
c
 555        continue
ctest off only one collision for the same 2 particles in the same timestep:
c            if(iblock.ne.0) then
c               goto 800
c            endif
ctest off collisions history:
c            if(iblock.ne.0) then 
c               write(10,*) nt,i1,i2,iblock,x1,z1,x2,z2
c            endif

  600     CONTINUE
  800   CONTINUE
* RELABLE MESONS LEFT IN THIS RUN EXCLUDING THOSE BEING CREATED DURING
* THIS TIME STEP AND COUNT THE TOTAL NO. OF PARTICLES IN THIS RUN
* note that the first mass=mta+mpr particles are baryons
c        write(*,*)'I: NNN,massr ', nnn,massr(irun)
        N0=MASS+MSUM
        DO 1005 N=N0+1,MASSR(IRUN)+MSUM
cbz11/25/98
clin-2/19/03 lb>5000: keep particles with no LB codes in ART(photon,lepton,..):
c        IF(E(N).GT.0.)THEN
        IF(E(N) .GT. 0. .OR. LB(N) .GT. 5000)THEN
cbz11/25/98end
        NNN=NNN+1
        RPION(1,NNN,IRUN)=R(1,N)
        RPION(2,NNN,IRUN)=R(2,N)
        RPION(3,NNN,IRUN)=R(3,N)
clin-10/28/03:
        if(nt.eq.ntmax) then
           ftpisv(NNN,IRUN)=ftsv(N)
           tfdpi(NNN,IRUN)=tfdcy(N)
        endif
c
        PPION(1,NNN,IRUN)=P(1,N)
        PPION(2,NNN,IRUN)=P(2,N)
        PPION(3,NNN,IRUN)=P(3,N)
        EPION(NNN,IRUN)=E(N)
        LPION(NNN,IRUN)=LB(N)
c       !! sp 12/19/00
        PROPI(NNN,IRUN)=PROPER(N)
clin-5/2008:
        dppion(NNN,IRUN)=dpertp(N)
c        if(lb(n) .eq. 45)
c    &   write(*,*)'IN-1  NT,NNN,LB,P ',nt,NNN,lb(n),proper(n)
        ENDIF
 1005 CONTINUE
        MASSRN(IRUN)=NNN+MASS
c        write(*,*)'F: NNN,massrn ', nnn,massrn(irun)
1000   CONTINUE
* CALCULATE THE AVERAGE SRT FOR N + N--->N +DELTA PROCESSES
C        IF(NODELT.NE.0)THEN
C        AVSRT=SUMSRT/FLOAT(NODELT)
C        ELSE
C        AVSRT=0.
C        ENDIF
C        WRITE(1097,'(F8.2,2X,E10.3)')FLOAT(NT)*DT,AVSRT
* RELABLE ALL THE PARTICLES EXISTING AFTER THIS TIME STEP
        IA=0
        IB=0
        DO 10001 IRUN=1,NUM
        IA=IA+MASSR(IRUN-1)
        IB=IB+MASSRN(IRUN-1)
        DO 10001 IC=1,MASSRN(IRUN)
        IE=IA+IC
        IG=IB+IC
        IF(IC.LE.MASS)THEN
        RT(1,IG)=R(1,IE)
        RT(2,IG)=R(2,IE)
        RT(3,IG)=R(3,IE)
clin-10/28/03:
        if(nt.eq.ntmax) then
           fttemp(IG)=ftsv(IE)
           tft(IG)=tfdcy(IE)
        endif
c
        PT(1,IG)=P(1,IE)
        PT(2,IG)=P(2,IE)
        PT(3,IG)=P(3,IE)
        ET(IG)=E(IE)
        LT(IG)=LB(IE)
        PROT(IG)=PROPER(IE)
clin-5/2008:
        dptemp(IG)=dpertp(IE)
        ELSE
        I0=IC-MASS
        RT(1,IG)=RPION(1,I0,IRUN)
        RT(2,IG)=RPION(2,I0,IRUN)
        RT(3,IG)=RPION(3,I0,IRUN)
clin-10/28/03:
        if(nt.eq.ntmax) then
           fttemp(IG)=ftpisv(I0,IRUN)
           tft(IG)=tfdpi(I0,IRUN)
        endif
c
        PT(1,IG)=PPION(1,I0,IRUN)
        PT(2,IG)=PPION(2,I0,IRUN)
        PT(3,IG)=PPION(3,I0,IRUN)
        ET(IG)=EPION(I0,IRUN)
        LT(IG)=LPION(I0,IRUN)
        PROT(IG)=PROPI(I0,IRUN)
clin-5/2008:
        dptemp(IG)=dppion(I0,IRUN)
        ENDIF
10001   CONTINUE
c
        IL=0
clin-10/26/01-hbt:
c        DO 10002 IRUN=1,NUM
        DO 10003 IRUN=1,NUM

        MASSR(IRUN)=MASSRN(IRUN)
        IL=IL+MASSR(IRUN-1)
        DO 10002 IM=1,MASSR(IRUN)
        IN=IL+IM
        R(1,IN)=RT(1,IN)
        R(2,IN)=RT(2,IN)
        R(3,IN)=RT(3,IN)
clin-10/28/03:
        if(nt.eq.ntmax) then
           ftsv(IN)=fttemp(IN)
           tfdcy(IN)=tft(IN)
        endif
        P(1,IN)=PT(1,IN)
        P(2,IN)=PT(2,IN)
        P(3,IN)=PT(3,IN)
        E(IN)=ET(IN)
        LB(IN)=LT(IN)
        PROPER(IN)=PROT(IN)
clin-5/2008:
        dpertp(IN)=dptemp(IN)
       IF(LB(IN).LT.1.OR.LB(IN).GT.2)ID(IN)=0
10002   CONTINUE
clin-ctest off check energy conservation after each timestep
c         enetot=0.
c         do ip=1,MASSR(IRUN)
c            if(e(ip).ne.0.or.lb(ip).eq.10022) enetot=enetot
c     1           +sqrt(p(1,ip)**2+p(2,ip)**2+p(3,ip)**2+e(ip)**2)
c         enddo
c         write(91,*) 'B:',nt,enetot,massr(irun),bimp 
clin-3/2009 move to the end of a timestep to take care of freezeout spacetime:
c        call hbtout(MASSR(IRUN),nt,ntmax)
10003 CONTINUE
c
      RETURN
      END
****************************************
            SUBROUTINE CMS(I1,I2,PX1CM,PY1CM,PZ1CM,SRT)
* PURPOSE : FIND THE MOMENTA OF PARTICLES IN THE CMS OF THE
*          TWO COLLIDING PARTICLES
* VARIABLES :
*****************************************
            PARAMETER (MAXSTR=150001)
            COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
            COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
            COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
            COMMON   /BG/  BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      SAVE   
            PX1=P(1,I1)
            PY1=P(2,I1)
            PZ1=P(3,I1)
            PX2=P(1,I2)
            PY2=P(2,I2)
            PZ2=P(3,I2)
            EM1=E(I1)
            EM2=E(I2)
            E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
            E2=SQRT(EM2**2 + PX2**2 + PY2**2 + PZ2**2 )
            S=(E1+E2)**2-(PX1+PX2)**2-(PY1+PY2)**2-(PZ1+PZ2)**2
            SRT=SQRT(S)
*LORENTZ-TRANSFORMATION IN I1-I2-C.M. SYSTEM
              ETOTAL = E1 + E2
              BETAX  = (PX1+PX2) / ETOTAL
              BETAY  = (PY1+PY2) / ETOTAL
              BETAZ  = (PZ1+PZ2) / ETOTAL
              GAMMA  = 1.0 / SQRT(1.0-BETAX**2-BETAY**2-BETAZ**2)
*TRANSFORMATION OF MOMENTA (PX1CM = - PX2CM)
              P1BETA = PX1*BETAX + PY1*BETAY + PZ1 * BETAZ
              TRANSF = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) - E1 )
              PX1CM  = BETAX * TRANSF + PX1
              PY1CM  = BETAY * TRANSF + PY1
              PZ1CM  = BETAZ * TRANSF + PZ1
              RETURN
              END
***************************************
            SUBROUTINE DISTCE(I1,I2,DELTAR,DS,DT,EC,SRT
     1      ,IC,PX1CM,PY1CM,PZ1CM)
* PURPOSE : CHECK IF THE COLLISION BETWEEN TWO PARTICLES CAN HAPPEN
*           BY CHECKING
*                      (1) IF THE DISTANCE BETWEEN THEM IS SMALLER
*           THAN THE MAXIMUM DISTANCE DETERMINED FROM THE CROSS SECTION.
*                      (2) IF PARTICLE WILL PASS EACH OTHER WITHIN
*           TWO HARD CORE RADIUS.
*                      (3) IF PARTICLES WILL GET CLOSER.
* VARIABLES :
*           IC=1 COLLISION HAPPENED
*           IC=-1 COLLISION CAN NOT HAPPEN
*****************************************
            PARAMETER (MAXSTR=150001)
            COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
            COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
            COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
            COMMON   /BG/  BETAX,BETAY,BETAZ,GAMMA
            COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /BG/
            common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1           px1n,py1n,pz1n,dp1n
            common /dpi/em2,lb2
            SAVE   
            IC=0
            X1=R(1,I1)
            Y1=R(2,I1)
            Z1=R(3,I1)
            PX1=P(1,I1)
            PY1=P(2,I1)
            PZ1=P(3,I1)
            X2=R(1,I2)
            Y2=R(2,I2)
            Z2=R(3,I2)
            PX2=P(1,I2)
            PY2=P(2,I2)
            PZ2=P(3,I2)
            EM1=E(I1)
            EM2=E(I2)
            E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
c            IF (ABS(X1-X2) .GT. DELTAR) GO TO 400
c            IF (ABS(Y1-Y2) .GT. DELTAR) GO TO 400
c            IF (ABS(Z1-Z2) .GT. DELTAR) GO TO 400
            RSQARE = (X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2
            IF (RSQARE .GT. DELTAR**2) GO TO 400
*NOW PARTICLES ARE CLOSE ENOUGH TO EACH OTHER !
              E2     = SQRT ( EM2**2 + PX2**2 + PY2**2 + PZ2**2 )
              S      = SRT*SRT
            IF (S .LT. EC) GO TO 400
*NOW THERE IS ENOUGH ENERGY AVAILABLE !
*LORENTZ-TRANSFORMATION IN I1-I2-C.M. SYSTEM
* BETAX, BETAY, BETAZ AND GAMMA HAVE BEEN GIVEN IN THE SUBROUTINE CMS
*TRANSFORMATION OF MOMENTA (PX1CM = - PX2CM)
              P1BETA = PX1*BETAX + PY1*BETAY + PZ1 * BETAZ
              TRANSF = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) - E1 )
              PRCM   = SQRT (PX1CM**2 + PY1CM**2 + PZ1CM**2)
              IF (PRCM .LE. 0.00001) GO TO 400
*TRANSFORMATION OF SPATIAL DISTANCE
              DRBETA = BETAX*(X1-X2) + BETAY*(Y1-Y2) + BETAZ*(Z1-Z2)
              TRANSF = GAMMA * GAMMA * DRBETA / (GAMMA + 1)
              DXCM   = BETAX * TRANSF + X1 - X2
              DYCM   = BETAY * TRANSF + Y1 - Y2
              DZCM   = BETAZ * TRANSF + Z1 - Z2
*DETERMINING IF THIS IS THE POINT OF CLOSEST APPROACH
              DRCM   = SQRT (DXCM**2  + DYCM**2  + DZCM**2 )
              DZZ    = (PX1CM*DXCM + PY1CM*DYCM + PZ1CM*DZCM) / PRCM
              if ((drcm**2 - dzz**2) .le. 0.) then
                BBB = 0.
              else
                BBB    = SQRT (DRCM**2 - DZZ**2)
              end if
*WILL PARTICLE PASS EACH OTHER WITHIN 2 * HARD CORE RADIUS ?
              IF (BBB .GT. DS) GO TO 400
              RELVEL = PRCM * (1.0/E1 + 1.0/E2)
              DDD    = RELVEL * DT * 0.5
*WILL PARTICLES GET CLOSER ?
              IF (ABS(DDD) .LT. ABS(DZZ)) GO TO 400
              IC=1
              GO TO 500
400           IC=-1
500           CONTINUE
              RETURN
              END
****************************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRNN(IRUN,PX,PY,PZ,SRT,I1,I2,IBLOCK,
     1NTAG,SIGNN,SIG,NT,ipert1)
*     PURPOSE:                                                         *
*             DEALING WITH NUCLEON-NUCLEON COLLISIONS                    *
*     NOTE   :                                                         *
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           NSTAR =1 INCLUDING N* RESORANCE,ELSE NOT                   *
*           NDIRCT=1 INCLUDING DIRECT PION PRODUCTION PROCESS         *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      0-> COLLISION CANNOT HAPPEN                     *
*                      1-> N-N ELASTIC COLLISION                       *
*                      2-> N+N->N+DELTA,OR N+N->N+N* REACTION          *
*                      3-> N+DELTA->N+N OR N+N*->N+N REACTION          *
*                      4-> N+N->D+D+pion reaction
*                     43->N+N->D(N*)+D(N*) reaction
*                     44->N+N->D+D+rho reaction
*                     45->N+N->N+N+rho
*                     46->N+N->N+N+omega
*           N12       - IS USED TO SPECIFY BARYON-BARYON REACTION      *
*                      CHANNELS. M12 IS THE REVERSAL CHANNEL OF N12    *
*                      N12,                                            *
*                      M12=1 FOR p+n-->delta(+)+ n                     *
*                          2     p+n-->delta(0)+ p                     *
*                          3     p+p-->delta(++)+n                     *
*                          4     p+p-->delta(+)+p                      *
*                          5     n+n-->delta(0)+n                      *
*                          6     n+n-->delta(-)+p                      *
*                          7     n+p-->N*(0)(1440)+p                   *
*                          8     n+p-->N*(+)(1440)+n                   *
*                        9     p+p-->N*(+)(1535)+p                     *
*                        10    n+n-->N*(0)(1535)+n                     *
*                         11    n+p-->N*(+)(1535)+n                     *
*                        12    n+p-->N*(0)(1535)+p
*                        13    D(++)+D(-)-->N*(+)(1440)+n
*                         14    D(++)+D(-)-->N*(0)(1440)+p
*                        15    D(+)+D(0)--->N*(+)(1440)+n
*                        16    D(+)+D(0)--->N*(0)(1440)+p
*                        17    D(++)+D(0)-->N*(+)(1535)+p
*                        18    D(++)+D(-)-->N*(0)(1535)+p
*                        19    D(++)+D(-)-->N*(+)(1535)+n
*                        20    D(+)+D(+)-->N*(+)(1535)+p
*                        21    D(+)+D(0)-->N*(+)(1535)+n
*                        22    D(+)+D(0)-->N*(0)(1535)+p
*                        23    D(+)+D(-)-->N*(0)(1535)+n
*                        24    D(0)+D(0)-->N*(0)(1535)+n
*                          25    N*(+)(14)+N*(+)(14)-->N*(+)(15)+p
*                          26    N*(0)(14)+N*(0)(14)-->N*(0)(15)+n
*                          27    N*(+)(14)+N*(0)(14)-->N*(+)(15)+n
*                        28    N*(+)(14)+N*(0)(14)-->N*(0)(15)+p
*                        29    N*(+)(14)+D+-->N*(+)(15)+p
*                        30    N*(+)(14)+D0-->N*(+)(15)+n
*                        31    N*(+)(14)+D(-)-->N*(0)(1535)+n
*                        32    N*(0)(14)+D++--->N*(+)(15)+p
*                        33    N*(0)(14)+D+--->N*(+)(15)+n
*                        34    N*(0)(14)+D+--->N*(0)(15)+p
*                        35    N*(0)(14)+D0-->N*(0)(15)+n
*                        36    N*(+)(14)+D0--->N*(0)(15)+p
*                        ++    see the note book for more listing
*                     
*
*     NOTE ABOUT N*(1440) RESORANCE IN Nucleon+NUCLEON COLLISION:      * 
*     As it has been discussed in VerWest's paper,I= 1(initial isospin)*
*     channel can all be attributed to delta resorance while I= 0      *
*     channel can all be  attribured to N* resorance.Only in n+p       *
*     one can have I=0 channel so is the N*(1440) resonance            *
*                                                                      *
*                             REFERENCES:                            *    
*                    J. CUGNON ET AL., NUCL. PHYS. A352, 505 (1981)    *
*                    Y. KITAZOE ET AL., PHYS. LETT. 166B, 35 (1986)    *
*                    B. VerWest el al., PHYS. PRV. C25 (1982)1979      *
*                    Gy. Wolf  et al, Nucl Phys A517 (1990) 615;       *
*                                     Nucl phys A552 (1993) 349.       *
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,aka=0.498,AP2=0.13957,AM0=1.232,
     2  PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383,APHI=1.020)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        parameter (xmd=1.8756,npdmax=10000)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
        common /gg/ dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
        COMMON /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
        COMMON /NN/NNN
cc      SAVE /NN/
        COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
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
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1 px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      common /dpi/em2,lb2
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      common /para8/ idpert,npertd,idxsec
      dimension ppd(3,npdmax),lbpd(npdmax)
      SAVE   
*-----------------------------------------------------------------------
      n12=0
      m12=0
      IBLOCK=0
      NTAG=0
      EM1=E(I1)
      EM2=E(I2)
      PR=SQRT( PX**2 + PY**2 + PZ**2 )
      C2=PZ / PR
      X1=RANART(NSEED)
      ianti=0
      if(lb(i1).lt.0 .and. lb(i2).lt.0) ianti=1
      call sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
clin-5/2008 Production of perturbative deuterons for idpert=1:
      if(idpert.eq.1.and.ipert1.eq.1) then
         IF (SRT .LT. 2.012) RETURN
         if((iabs(lb(i1)).eq.1.or.iabs(lb(i1)).eq.2)
     1        .and.(iabs(lb(i2)).eq.1.or.iabs(lb(i2)).eq.2)) then
            goto 108
         else
            return
         endif
      endif
c
*-----------------------------------------------------------------------
*COM: TEST FOR ELASTIC SCATTERING (EITHER N-N OR DELTA-DELTA 0R
*      N-DELTA OR N*-N* or N*-Delta)
c      IF (X1 .LE. SIGNN/SIG) THEN
      IF (X1.LE.(SIGNN/SIG)) THEN
*COM:  PARAMETRISATION IS TAKEN FROM THE CUGNON-PAPER
         AS  = ( 3.65 * (SRT - 1.8766) )**6
         A   = 6.0 * AS / (1.0 + AS)
         TA  = -2.0 * PR**2
         X   = RANART(NSEED)
clin-10/24/02        T1  = DLOG( (1-X) * DEXP(dble(A)*dble(TA)) + X )  /  A
         T1  = sngl(DLOG(dble(1.-X)*DEXP(dble(A)*dble(TA))+dble(X)))/  A
         C1  = 1.0 - T1/TA
         T1  = 2.0 * PI * RANART(NSEED)
         IBLOCK=1
         GO TO 107
      ELSE
*COM: TEST FOR INELASTIC SCATTERING
*     IF THE AVAILABLE ENERGY IS LESS THAN THE PION-MASS, NOTHING
*     CAN HAPPEN ANY MORE ==> RETURN (2.012 = 2*AVMASS + PI-MASS)
clin-5/2008: Mdeuteron+Mpi=2.0106 to 2.0152 GeV/c2, so we can still use this:
         IF (SRT .LT. 2.012) RETURN
*     calculate the N*(1535) production cross section in N+N collisions
*     note that the cross sections in this subroutine are in units of mb
*     as only ratios of the cross sections are used to determine the
*     reaction channels
       call N1535(iabs(lb(i1)),iabs(lb(i2)),srt,x1535)
*COM: HERE WE HAVE A PROCESS N+N ==> N+DELTA,OR N+N==>N+N*(144) or N*(1535)
*     OR 
* 3 pi channel : N+N==>d1+d2+PION
       SIG3=3.*(X3pi(SRT)+x33pi(srt))
* 2 pi channel : N+N==>d1+d2+d1*n*+n*n*
       SIG4=4.*X2pi(srt)
* 4 pi channel : N+N==>d1+d2+rho
       s4pi=x4pi(srt)
* N+N-->NN+rho channel
       srho=xrho(srt)
* N+N-->NN+omega
       somega=omega(srt)
* CROSS SECTION FOR KAON PRODUCTION from the four channels
* for NLK channel
       akp=0.498
       ak0=0.498
       ana=0.94
       ada=1.232
       al=1.1157
       as=1.1197
       xsk1=0
       xsk2=0
       xsk3=0
       xsk4=0
       xsk5=0
       t1nlk=ana+al+akp
       if(srt.le.t1nlk)go to 222
       XSK1=1.5*PPLPK(SRT)
* for DLK channel
       t1dlk=ada+al+akp
       t2dlk=ada+al-akp
       if(srt.le.t1dlk)go to 222
       es=srt
       pmdlk2=(es**2-t1dlk**2)*(es**2-t2dlk**2)/(4.*es**2)
       pmdlk=sqrt(pmdlk2)
       XSK3=1.5*PPLPK(srt)
* for NSK channel
       t1nsk=ana+as+akp
       t2nsk=ana+as-akp
       if(srt.le.t1nsk)go to 222
       pmnsk2=(es**2-t1nsk**2)*(es**2-t2nsk**2)/(4.*es**2)
       pmnsk=sqrt(pmnsk2)
       XSK2=1.5*(PPK1(srt)+PPK0(srt))
* for DSK channel
       t1DSk=aDa+aS+akp
       t2DSk=aDa+aS-akp
       if(srt.le.t1dsk)go to 222
       pmDSk2=(es**2-t1DSk**2)*(es**2-t2DSk**2)/(4.*es**2)
       pmDSk=sqrt(pmDSk2)
       XSK4=1.5*(PPK1(srt)+PPK0(srt))
csp11/21/01
c phi production
       if(srt.le.(2.*amn+aphi))go to 222
c  !! mb put the correct form
       xsk5 = 0.0001
csp11/21/01 end
c
* THE TOTAL KAON+ PRODUCTION CROSS SECTION IS THEN
 222   SIGK=XSK1+XSK2+XSK3+XSK4

cbz3/7/99 neutralk
        XSK1 = 2.0 * XSK1
        XSK2 = 2.0 * XSK2
        XSK3 = 2.0 * XSK3
        XSK4 = 2.0 * XSK4
        SIGK = 2.0 * SIGK + xsk5
cbz3/7/99 neutralk end
c
** FOR P+P or L/S+L/S COLLISION:
c       lb1=lb(i1)
c       lb2=lb(i2)
        lb1=iabs(lb(i1))
        lb2=iabs(lb(i2))
        IF((LB(I1)*LB(I2).EQ.1).or.
     &       ((lb1.le.17.and.lb1.ge.14).and.(lb2.le.17.and.lb2.ge.14)).
     &       or.((lb1.le.2).and.(lb2.le.17.and.lb2.ge.14)).
     &       or.((lb2.le.2).and.(lb1.le.17.and.lb1.ge.14)))THEN
clin-8/2008 PP->d+meson here:
           IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
           SIG1=SIGMA(SRT,1,1,0)+0.5*SIGMA(SRT,1,1,1)
           SIG2=1.5*SIGMA(SRT,1,1,1)
           SIGND=SIG1+SIG2+SIG3+SIG4+X1535+SIGK+s4pi+srho+somega
clin-5/2008:
c           IF (X1.GT.(SIGNN+SIGND)/SIG)RETURN
           IF (X1.GT.(SIGNN+SIGND+sdprod)/SIG)RETURN
           DIR=SIG3/SIGND
           IF(RANART(NSEED).LE.DIR)GO TO 106
           IF(RANART(NSEED).LE.SIGK/(SIGK+X1535+SIG4+SIG2+SIG1
     &          +s4pi+srho+somega))GO TO 306
           if(RANART(NSEED).le.s4pi/(x1535+sig4+sig2+sig1
     &          +s4pi+srho+somega))go to 307
           if(RANART(NSEED).le.srho/(x1535+sig4+sig2+sig1
     &          +srho+somega))go to 308
           if(RANART(NSEED).le.somega/(x1535+sig4+sig2+sig1
     &          +somega))go to 309
           if(RANART(NSEED).le.x1535/(sig1+sig2+sig4+x1535))then
* N*(1535) production
              N12=9
           ELSE 
              IF(RANART(NSEED).LE.SIG4/(SIG1+sig2+sig4))THEN
* DOUBLE DELTA PRODUCTION
                 N12=66
                 GO TO 1012
              else
*DELTA PRODUCTION
                 N12=3
                 IF (RANART(NSEED).GT.SIG1/(SIG1+SIG2))N12=4
              ENDIF
           endif
           GO TO 1011
        ENDIF
** FOR N+N COLLISION:
        IF(iabs(LB(I1)).EQ.2.AND.iabs(LB(I2)).EQ.2)THEN
clin-8/2008 NN->d+meson here:
           IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
           SIG1=SIGMA(SRT,1,1,0)+0.5*SIGMA(SRT,1,1,1)
           SIG2=1.5*SIGMA(SRT,1,1,1)
           SIGND=SIG1+SIG2+X1535+SIG3+SIG4+SIGK+s4pi+srho+somega
clin-5/2008:
c           IF (X1.GT.(SIGNN+SIGND)/SIG)RETURN
           IF (X1.GT.(SIGNN+SIGND+sdprod)/SIG)RETURN
           dir=sig3/signd
           IF(RANART(NSEED).LE.DIR)GO TO 106
           IF(RANART(NSEED).LE.SIGK/(SIGK+X1535+SIG4+SIG2+SIG1
     &          +s4pi+srho+somega))GO TO 306
           if(RANART(NSEED).le.s4pi/(x1535+sig4+sig2+sig1
     &          +s4pi+srho+somega))go to 307
           if(RANART(NSEED).le.srho/(x1535+sig4+sig2+sig1
     &          +srho+somega))go to 308
           if(RANART(NSEED).le.somega/(x1535+sig4+sig2+sig1
     &          +somega))go to 309
           IF(RANART(NSEED).LE.X1535/(x1535+sig1+sig2+sig4))THEN
* N*(1535) PRODUCTION
              N12=10
           ELSE 
              if(RANART(NSEED).le.sig4/(sig1+sig2+sig4))then
* double delta production
                 N12=67
                 GO TO 1013
              else
* DELTA PRODUCTION
                 N12=6
                 IF (RANART(NSEED).GT.SIG1/(SIG1+SIG2))N12=5
              ENDIF
           endif
           GO TO 1011
        ENDIF
** FOR N+P COLLISION
        IF(LB(I1)*LB(I2).EQ.2)THEN
clin-5/2008 NP->d+meson here:
           IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
           SIG1=0.5*SIGMA(SRT,1,1,1)+0.25*SIGMA(SRT,1,1,0)
           IF(NSTAR.EQ.1)THEN
              SIG2=(3./4.)*SIGMA(SRT,2,0,1)
           ELSE
              SIG2=0.
           ENDIF
           SIGND=2.*(SIG1+SIG2+X1535)+sig3+sig4+SIGK+s4pi+srho+somega
clin-5/2008:
c           IF (X1.GT.(SIGNN+SIGND)/SIG)RETURN
           IF (X1.GT.(SIGNN+SIGND+sdprod)/SIG)RETURN
           dir=sig3/signd
           IF(RANART(NSEED).LE.DIR)GO TO 106
           IF(RANART(NSEED).LE.SIGK/(SIGND-SIG3))GO TO 306
           if(RANART(NSEED).le.s4pi/(signd-sig3-sigk))go to 307
           if(RANART(NSEED).le.srho/(signd-sig3-sigk-s4pi))go to 308
           if(RANART(NSEED).le.somega/(signd-sig3-sigk-s4pi-srho))
     1          go to 309
           IF(RANART(NSEED).LT.X1535/(SIG1+SIG2+X1535+0.5*sig4))THEN
* N*(1535) PRODUCTION
              N12=11
              IF(RANART(NSEED).LE.0.5)N12=12
           ELSE 
              if(RANART(NSEED).le.sig4/(sig4+2.*(sig1+sig2)))then
* double resonance production
                 N12=68
                 GO TO 1014
              else
                 IF(RANART(NSEED).LE.SIG1/(SIG1+SIG2))THEN
* DELTA PRODUCTION
                    N12=2
                    IF(RANART(NSEED).GE.0.5)N12=1
                 ELSE
* N*(1440) PRODUCTION
                    N12=8
                    IF(RANART(NSEED).GE.0.5)N12=7
                 ENDIF
              ENDIF
           ENDIF
        endif
 1011   iblock=2
        CONTINUE
*PARAMETRIZATION OF THE SHAPE OF THE DELTA RESONANCE ACCORDING
*     TO kitazoe's or J.D.JACKSON'S MASS FORMULA AND BREIT WIGNER
*     FORMULA FOR N* RESORANCE
*     DETERMINE DELTA MASS VIA REJECTION METHOD.
          DMAX = SRT - AVMASS-0.005
          DMAX = SRT - AVMASS-0.005
          DMIN = 1.078
                   IF(N12.LT.7)THEN
* Delta(1232) production
          IF(DMAX.LT.1.232) THEN
          FM=FDE(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FDE():
             xdmass=1.232
c          FM=FDE(1.232,SRT,1.)
          FM=FDE(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY1=0
10        DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
          NTRY1=NTRY1+1
          IF((RANART(NSEED) .GT. FDE(DM,SRT,1.)/FM).AND.
     1    (NTRY1.LE.30)) GOTO 10

clin-2/26/03 limit the Delta mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.1.47) goto 10

              GO TO 13
              ENDIF
                   IF((n12.eq.7).or.(n12.eq.8))THEN
* N*(1440) production
          IF(DMAX.LT.1.44) THEN
          FM=FNS(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FNS():
             xdmass=1.44
c          FM=FNS(1.44,SRT,1.)
          FM=FNS(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY2=0
11        DM=RANART(NSEED)*(DMAX-DMIN)+DMIN
          NTRY2=NTRY2+1
          IF((RANART(NSEED).GT.FNS(DM,SRT,1.)/FM).AND.
     1    (NTRY2.LE.10)) GO TO 11

clin-2/26/03 limit the N* mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.2.14) goto 11

              GO TO 13
              ENDIF
                    IF(n12.ge.17)then
* N*(1535) production
          IF(DMAX.LT.1.535) THEN
          FM=FD5(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FNS():
             xdmass=1.535
c          FM=FD5(1.535,SRT,1.)
          FM=FD5(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY1=0
12        DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
          NTRY1=NTRY1+1
          IF((RANART(NSEED) .GT. FD5(DM,SRT,1.)/FM).AND.
     1    (NTRY1.LE.10)) GOTO 12

clin-2/26/03 limit the N* mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.1.84) goto 12

         GO TO 13
             ENDIF
* CALCULATE THE MASSES OF BARYON RESONANCES IN THE DOUBLE RESONANCE
* PRODUCTION PROCESS AND RELABLE THE PARTICLES
1012       iblock=43
       call Rmasdd(srt,1.232,1.232,1.08,
     &  1.08,ISEED,1,dm1,dm2)
       call Rmasdd(srt,1.232,1.44,1.08,
     &  1.08,ISEED,3,dm1n,dm2n)
       IF(N12.EQ.66)THEN
*(1) PP-->DOUBLE RESONANCES
* DETERMINE THE FINAL STATE
       XFINAL=RANART(NSEED)
       IF(XFINAL.LE.0.25)THEN
* (1.1) D+++D0 
       LB(I1)=9
       LB(I2)=7
       e(i1)=dm1
       e(i2)=dm2
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF((XFINAL.gt.0.25).and.(xfinal.le.0.5))THEN
* (1.2) D++D+
       LB(I1)=8
       LB(I2)=8
       e(i1)=dm1
       e(i2)=dm2
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF((XFINAL.gt.0.5).and.(xfinal.le.0.75))THEN
* (1.3) D+++N*0 
       LB(I1)=9
       LB(I2)=10
       e(i1)=dm1n
       e(i2)=dm2n
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF(XFINAL.gt.0.75)then
* (1.4) D++N*+ 
       LB(I1)=8
       LB(I2)=11
       e(i1)=dm1n
       e(i2)=dm2n
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       ENDIF
1013       iblock=43
       call Rmasdd(srt,1.232,1.232,1.08,
     &  1.08,ISEED,1,dm1,dm2)
       call Rmasdd(srt,1.232,1.44,1.08,
     &  1.08,ISEED,3,dm1n,dm2n)
       IF(N12.EQ.67)THEN
*(2) NN-->DOUBLE RESONANCES
* DETERMINE THE FINAL STATE
       XFINAL=RANART(NSEED)
       IF(XFINAL.LE.0.25)THEN
* (2.1) D0+D0 
       LB(I1)=7
       LB(I2)=7
       e(i1)=dm1
       e(i2)=dm2
       GO TO 200
* go to 200 to set the new momentum
        ENDIF
       IF((XFINAL.gt.0.25).and.(xfinal.le.0.5))THEN
* (2.2) D++D+
       LB(I1)=6
       LB(I2)=8
       e(i1)=dm1
       e(i2)=dm2
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF((XFINAL.gt.0.5).and.(xfinal.le.0.75))THEN
* (2.3) D0+N*0 
       LB(I1)=7
       LB(I2)=10
       e(i1)=dm1n
       e(i2)=dm2n
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF(XFINAL.gt.0.75)then
* (2.4) D++N*+ 
       LB(I1)=8
       LB(I2)=11
       e(i1)=dm1n
       e(i2)=dm2n
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       ENDIF
1014       iblock=43
       call Rmasdd(srt,1.232,1.232,1.08,
     &  1.08,ISEED,1,dm1,dm2)
       call Rmasdd(srt,1.232,1.44,1.08,
     &  1.08,ISEED,3,dm1n,dm2n)
       IF(N12.EQ.68)THEN
*(3) NP-->DOUBLE RESONANCES
* DETERMINE THE FINAL STATE
       XFINAL=RANART(NSEED)
       IF(XFINAL.LE.0.25)THEN
* (3.1) D0+D+ 
       LB(I1)=7
       LB(I2)=8
       e(i1)=dm1
       e(i2)=dm2
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF((XFINAL.gt.0.25).and.(xfinal.le.0.5))THEN
* (3.2) D+++D-
       LB(I1)=9
       LB(I2)=6
       e(i1)=dm1
       e(i2)=dm2
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF((XFINAL.gt.0.5).and.(xfinal.le.0.75))THEN
* (3.3) D0+N*+ 
       LB(I1)=7
       LB(I2)=11
       e(i1)=dm1n
       e(i2)=dm2n
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       IF(XFINAL.gt.0.75)then
* (3.4) D++N*0
       LB(I1)=8
       LB(I2)=10
       e(i1)=dm1n
       e(i2)=dm2n
       GO TO 200
* go to 200 to set the new momentum
       ENDIF
       ENDIF
13       CONTINUE
*-------------------------------------------------------
* RELABLE BARYON I1 AND I2
*1. p+n-->delta(+)+n
          IF(N12.EQ.1)THEN
          IF(iabs(LB(I1)).EQ.1)THEN
          LB(I2)=2
          LB(I1)=8
          E(I1)=DM
          ELSE
          LB(I1)=2
          LB(I2)=8
          E(I2)=DM
          ENDIF
         GO TO 200
          ENDIF
*2 p+n-->delta(0)+p
          IF(N12.EQ.2)THEN
          IF(iabs(LB(I1)).EQ.2)THEN
          LB(I2)=1
          LB(I1)=7
          E(I1)=DM
          ELSE
          LB(I1)=1
          LB(I2)=7
          E(I2)=DM
          ENDIF
         GO TO 200
          ENDIF
*3 p+p-->delta(++)+n
          IF(N12.EQ.3)THEN
          LB(I1)=9
          E(I1)=DM
          LB(I2)=2
          E(I2)=AMN
         GO TO 200
          ENDIF
*4 p+p-->delta(+)+p
          IF(N12.EQ.4)THEN
          LB(I2)=1
          LB(I1)=8
          E(I1)=DM
         GO TO 200
          ENDIF
*5 n+n--> delta(0)+n
          IF(N12.EQ.5)THEN
          LB(I2)=2
          LB(I1)=7
          E(I1)=DM
         GO TO 200
          ENDIF
*6 n+n--> delta(-)+p
          IF(N12.EQ.6)THEN
          LB(I1)=6
          E(I1)=DM
          LB(I2)=1
          E(I2)=AMP
         GO TO 200
          ENDIF
*7 n+p--> N*(0)+p
          IF(N12.EQ.7)THEN
          IF(iabs(LB(I1)).EQ.1)THEN
          LB(I1)=1
          LB(I2)=10
          E(I2)=DM
          ELSE
          LB(I2)=1
          LB(I1)=10
          E(I1)=DM
          ENDIF
         GO TO 200
          ENDIF
*8 n+p--> N*(+)+n
          IF(N12.EQ.8)THEN
          IF(iabs(LB(I1)).EQ.1)THEN
          LB(I2)=2
          LB(I1)=11
          E(I1)=DM
          ELSE
          LB(I1)=2
          LB(I2)=11
          E(I2)=DM
          ENDIF
         GO TO 200
          ENDIF
*9 p+p--> N*(+)(1535)+p
          IF(N12.EQ.9)THEN
          IF(RANART(NSEED).le.0.5)THEN
          LB(I2)=1
          LB(I1)=13
          E(I1)=DM
          ELSE
          LB(I1)=1
          LB(I2)=13
          E(I2)=DM
          ENDIF
         GO TO 200
          ENDIF
*10 n+n--> N*(0)(1535)+n
          IF(N12.EQ.10)THEN
          IF(RANART(NSEED).le.0.5)THEN
          LB(I2)=2
          LB(I1)=12
          E(I1)=DM
          ELSE
          LB(I1)=2
          LB(I2)=12
          E(I2)=DM
          ENDIF
         GO TO 200
          ENDIF
*11 n+p--> N*(+)(1535)+n
          IF(N12.EQ.11)THEN
          IF(iabs(LB(I1)).EQ.2)THEN
          LB(I1)=2
          LB(I2)=13
          E(I2)=DM
          ELSE
          LB(I2)=2
          LB(I1)=13
          E(I1)=DM
          ENDIF
         GO TO 200
          ENDIF
*12 n+p--> N*(0)(1535)+p
          IF(N12.EQ.12)THEN
          IF(iabs(LB(I1)).EQ.1)THEN
          LB(I1)=1
          LB(I2)=12
          E(I2)=DM
          ELSE
          LB(I2)=1
          LB(I1)=12
          E(I1)=DM
          ENDIF
          ENDIF
         endif
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
200       EM1=E(I1)
          EM2=E(I2)
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
              if(srt.le.2.14)C1= 1.0 - 2.0 * RANART(NSEED)
         if(srt.gt.2.14.and.srt.le.2.4)c1=ang(srt,iseed)
         if(srt.gt.2.4)then

clin-10/25/02 get rid of argument usage mismatch in PTR():
             xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
             cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
         endif
          T1   = 2.0 * PI * RANART(NSEED)
       if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
         lb(i1) = -lb(i1)
         lb(i2) = -lb(i2)
       endif
          GO TO 107
*FOR THE NN-->D1+D2+PI PROCESS, FIND MOMENTUM OF THE FINAL TWO
*DELTAS AND PION IN THE NUCLEUS-NUCLEUS CMS.
106     CONTINUE
           NTRY1=0
123        CALL DDP2(SRT,ISEED,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.40))GO TO 123
C       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
                NNN=NNN+1
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
* (1) FOR P+P
              XDIR=RANART(NSEED)
                IF(LB(I1)*LB(I2).EQ.1)THEN
                IF(XDIR.Le.0.2)then
* (1.1)P+P-->D+++D0+PION(0)
                LPION(NNN,IRUN)=4
                EPION(NNN,IRUN)=AP1
              LB(I1)=9
              LB(I2)=7
       GO TO 205
                ENDIF
* (1.2)P+P -->D++D+PION(0)
                IF((XDIR.LE.0.4).AND.(XDIR.GT.0.2))THEN
                LPION(NNN,IRUN)=4
                EPION(NNN,IRUN)=AP1
                LB(I1)=8
                LB(I2)=8
       GO TO 205
              ENDIF 
* (1.3)P+P-->D+++D+PION(-)
                IF((XDIR.LE.0.6).AND.(XDIR.GT.0.4))THEN
                LPION(NNN,IRUN)=3
                EPION(NNN,IRUN)=AP2
                LB(I1)=9
                LB(I2)=8
       GO TO 205
              ENDIF 
                IF((XDIR.LE.0.8).AND.(XDIR.GT.0.6))THEN
                LPION(NNN,IRUN)=5
                EPION(NNN,IRUN)=AP2
                LB(I1)=9
                LB(I2)=6
       GO TO 205
              ENDIF 
                IF(XDIR.GT.0.8)THEN
                LPION(NNN,IRUN)=5
                EPION(NNN,IRUN)=AP2
                LB(I1)=7
                LB(I2)=8
       GO TO 205
              ENDIF 
               ENDIF
* (2)FOR N+N
                IF(iabs(LB(I1)).EQ.2.AND.iabs(LB(I2)).EQ.2)THEN
                IF(XDIR.Le.0.2)then
* (2.1)N+N-->D++D-+PION(0)
                LPION(NNN,IRUN)=4
                EPION(NNN,IRUN)=AP1
              LB(I1)=6
              LB(I2)=7
       GO TO 205
                ENDIF
* (2.2)N+N -->D+++D-+PION(-)
                IF((XDIR.LE.0.4).AND.(XDIR.GT.0.2))THEN
                LPION(NNN,IRUN)=3
                EPION(NNN,IRUN)=AP2
                LB(I1)=6
                LB(I2)=9
       GO TO 205
              ENDIF 
* (2.3)P+P-->D0+D-+PION(+)
                IF((XDIR.GT.0.4).AND.(XDIR.LE.0.6))THEN
                LPION(NNN,IRUN)=5
                EPION(NNN,IRUN)=AP2
                LB(I1)=9
                LB(I2)=8
       GO TO 205
              ENDIF 
* (2.4)P+P-->D0+D0+PION(0)
                IF((XDIR.GT.0.6).AND.(XDIR.LE.0.8))THEN
                LPION(NNN,IRUN)=4
                EPION(NNN,IRUN)=AP1
                LB(I1)=7
                LB(I2)=7
       GO TO 205
              ENDIF 
* (2.5)P+P-->D0+D++PION(-)
                IF(XDIR.GT.0.8)THEN
                LPION(NNN,IRUN)=3
                EPION(NNN,IRUN)=AP2
                LB(I1)=7
                LB(I2)=8
       GO TO 205
              ENDIF 
              ENDIF
* (3)FOR N+P
                IF(LB(I1)*LB(I2).EQ.2)THEN
                IF(XDIR.Le.0.17)then
* (3.1)N+P-->D+++D-+PION(0)
                LPION(NNN,IRUN)=4
                EPION(NNN,IRUN)=AP1
              LB(I1)=6
              LB(I2)=9
       GO TO 205
                ENDIF
* (3.2)N+P -->D+++D0+PION(-)
                IF((XDIR.LE.0.34).AND.(XDIR.GT.0.17))THEN
                LPION(NNN,IRUN)=3
                EPION(NNN,IRUN)=AP2
                LB(I1)=7
                LB(I2)=9
       GO TO 205
              ENDIF 
* (3.3)N+P-->D++D-+PION(+)
                IF((XDIR.GT.0.34).AND.(XDIR.LE.0.51))THEN
                LPION(NNN,IRUN)=5
                EPION(NNN,IRUN)=AP2
                LB(I1)=7
                LB(I2)=8
       GO TO 205
              ENDIF 
* (3.4)N+P-->D++D++PION(-)
                IF((XDIR.GT.0.51).AND.(XDIR.LE.0.68))THEN
                LPION(NNN,IRUN)=3
                EPION(NNN,IRUN)=AP2
                LB(I1)=8
                LB(I2)=8
       GO TO 205
              ENDIF 
* (3.5)N+P-->D0+D++PION(0)
                IF((XDIR.GT.0.68).AND.(XDIR.LE.0.85))THEN
                LPION(NNN,IRUN)=4
                EPION(NNN,IRUN)=AP2
                LB(I1)=7
                LB(I2)=8
       GO TO 205
              ENDIF 
* (3.6)N+P-->D0+D0+PION(+)
                IF(XDIR.GT.0.85)THEN
                LPION(NNN,IRUN)=5
                EPION(NNN,IRUN)=AP2
                LB(I1)=7
                LB(I2)=7
              ENDIF 
                ENDIF
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
205           E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
c
             if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
               lb(i1) = -lb(i1)
               lb(i2) = -lb(i2)
                if(LPION(NNN,IRUN) .eq. 3)then
                  LPION(NNN,IRUN)=5
                elseif(LPION(NNN,IRUN) .eq. 5)then
                  LPION(NNN,IRUN)=3
                endif
               endif
c
             lb1=lb(i1)
* FOR DELTA2
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
              lb2=lb(i2)
* assign delta1 and delta2 to i1 or i2 to keep the leadng particle
* behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
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
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
                IBLOCK=4
* GET PION'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(EPION(NNN,IRUN)**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008:
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008 do not allow smearing in position of produced particles 
c     to avoid immediate reinteraction with the particle I1, I2 or themselves:
c2002        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2002
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                RPION(1,NNN,IRUN)=R(1,I1)
                RPION(2,NNN,IRUN)=R(2,I1)
                RPION(3,NNN,IRUN)=R(3,I1)
c
              go to 90005
clin-5/2008 N+N->Deuteron+pi:
*     FIND MOMENTUM OF THE FINAL PARTICLES IN THE NUCLEUS-NUCLEUS CMS.
 108       CONTINUE
           if(idpert.eq.1.and.ipert1.eq.1.and.npertd.ge.1) then
c     For idpert=1: we produce npertd pert deuterons:
              ndloop=npertd
           elseif(idpert.eq.2.and.npertd.ge.1) then
c     For idpert=2: we first save information for npertd pert deuterons;
c     at the last ndloop we create the regular deuteron+pi 
c     and those pert deuterons:
              ndloop=npertd+1
           else
c     Just create the regular deuteron+pi:
              ndloop=1
           endif
c
           dprob1=sdprod/sig/float(npertd)
           do idloop=1,ndloop
              CALL bbdangle(pxd,pyd,pzd,nt,ipert1,ianti,idloop,pfinal,
     1 dprob1,lbm)
              CALL ROTATE(PX,PY,PZ,PXd,PYd,PZd)
*     LORENTZ-TRANSFORMATION OF THE MOMENTUM OF PARTICLES IN THE FINAL STATE 
*     FROM THE NN CMS FRAME INTO THE GLOBAL CMS FRAME:
*     For the Deuteron:
              xmass=xmd
              E1dCM=SQRT(xmass**2+PXd**2+PYd**2+PZd**2)
              P1dBETA=PXd*BETAX+PYd*BETAY+PZd*BETAZ
              TRANSF=GAMMA*(GAMMA*P1dBETA/(GAMMA+1.)+E1dCM)
              pxi1=BETAX*TRANSF+PXd
              pyi1=BETAY*TRANSF+PYd
              pzi1=BETAZ*TRANSF+PZd
              if(ianti.eq.0)then
                 lbd=42
              else
                 lbd=-42
              endif
              if(idpert.eq.1.and.ipert1.eq.1.and.npertd.ge.1) then
cccc  Perturbative production for idpert=1:
                 nnn=nnn+1
                 PPION(1,NNN,IRUN)=pxi1
                 PPION(2,NNN,IRUN)=pyi1
                 PPION(3,NNN,IRUN)=pzi1
                 EPION(NNN,IRUN)=xmd
                 LPION(NNN,IRUN)=lbd
                 RPION(1,NNN,IRUN)=R(1,I1)
                 RPION(2,NNN,IRUN)=R(2,I1)
                 RPION(3,NNN,IRUN)=R(3,I1)
clin-5/2008 assign the perturbative probability:
                 dppion(NNN,IRUN)=sdprod/sig/float(npertd)
              elseif(idpert.eq.2.and.idloop.le.npertd) then
clin-5/2008 For idpert=2, we produce NPERTD perturbative (anti)deuterons 
c     only when a regular (anti)deuteron+pi is produced in NN collisions.
c     First save the info for the perturbative deuterons:
                 ppd(1,idloop)=pxi1
                 ppd(2,idloop)=pyi1
                 ppd(3,idloop)=pzi1
                 lbpd(idloop)=lbd
              else
cccc  Regular production:
c     For the regular pion: do LORENTZ-TRANSFORMATION:
                 E(i1)=xmm
                 E2piCM=SQRT(xmm**2+PXd**2+PYd**2+PZd**2)
                 P2piBETA=-PXd*BETAX-PYd*BETAY-PZd*BETAZ
                 TRANSF=GAMMA*(GAMMA*P2piBETA/(GAMMA+1.)+E2piCM)
                 pxi2=BETAX*TRANSF-PXd
                 pyi2=BETAY*TRANSF-PYd
                 pzi2=BETAZ*TRANSF-PZd
                 p(1,i1)=pxi2
                 p(2,i1)=pyi2
                 p(3,i1)=pzi2
c     Remove regular pion to check the equivalence 
c     between the perturbative and regular deuteron results:
c                 E(i1)=0.
c
                 LB(I1)=lbm
                 PX1=P(1,I1)
                 PY1=P(2,I1)
                 PZ1=P(3,I1)
                 EM1=E(I1)
                 ID(I1)=2
                 ID1=ID(I1)
                 E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
                 lb1=lb(i1)
c     For the regular deuteron:
                 p(1,i2)=pxi1
                 p(2,i2)=pyi1
                 p(3,i2)=pzi1
                 lb(i2)=lbd
                 lb2=lb(i2)
                 E(i2)=xmd
                 EtI2=E(I2)
                 ID(I2)=2
c     For idpert=2: create the perturbative deuterons:
                 if(idpert.eq.2.and.idloop.eq.ndloop) then
                    do ipertd=1,npertd
                       nnn=nnn+1
                       PPION(1,NNN,IRUN)=ppd(1,ipertd)
                       PPION(2,NNN,IRUN)=ppd(2,ipertd)
                       PPION(3,NNN,IRUN)=ppd(3,ipertd)
                       EPION(NNN,IRUN)=xmd
                       LPION(NNN,IRUN)=lbpd(ipertd)
                       RPION(1,NNN,IRUN)=R(1,I1)
                       RPION(2,NNN,IRUN)=R(2,I1)
                       RPION(3,NNN,IRUN)=R(3,I1)
clin-5/2008 assign the perturbative probability:
                       dppion(NNN,IRUN)=1./float(npertd)
                    enddo
                 endif
              endif
           enddo
           IBLOCK=501
           go to 90005
clin-5/2008 N+N->Deuteron+pi over
* FOR THE NN-->KAON+X PROCESS, FIND MOMENTUM OF THE FINAL PARTICLES IN 
* THE NUCLEUS-NUCLEUS CMS.
306     CONTINUE
csp11/21/01 phi production
              if(XSK5/sigK.gt.RANART(NSEED))then
              pz1=p(3,i1)
              pz2=p(3,i2)
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 1 + int(2 * RANART(NSEED))
              nnn=nnn+1
                LPION(NNN,IRUN)=29
                EPION(NNN,IRUN)=APHI
                iblock = 222
              GO TO 208
               ENDIF
c
                 IBLOCK=9
                 if(ianti .eq. 1)iblock=-9
c
              pz1=p(3,i1)
              pz2=p(3,i2)
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
              nnn=nnn+1
                LPION(NNN,IRUN)=23
                EPION(NNN,IRUN)=Aka
              if(srt.le.2.63)then
* only lambda production is possible
* (1.1)P+P-->p+L+kaon+
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              GO TO 208
                ENDIF
       if(srt.le.2.74.and.srt.gt.2.63)then
* both Lambda and sigma production are possible
              if(XSK1/(XSK1+XSK2).gt.RANART(NSEED))then
* lambda production
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              else
* sigma production
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              ic=2
              endif
              GO TO 208
       endif
       if(srt.le.2.77.and.srt.gt.2.74)then
* then pp-->Delta lamda kaon can happen
              if(xsk1/(xsk1+xsk2+xsk3).
     1          gt.RANART(NSEED))then
* * (1.1)P+P-->p+L+kaon+
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              go to 208
              else
              if(xsk2/(xsk2+xsk3).gt.RANART(NSEED))then
* pp-->psk
              ic=2
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              else
* pp-->D+l+k        
              ic=3
                LB(I1) = 6 + int(4 * RANART(NSEED))
              lb(i2)=14
              endif
              GO TO 208
              endif
       endif
       if(srt.gt.2.77)then
* all four channels are possible
              if(xsk1/(xsk1+xsk2+xsk3+xsk4).gt.RANART(NSEED))then
* p lambda k production
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              go to 208
       else
          if(xsk3/(xsk2+xsk3+xsk4).gt.RANART(NSEED))then
* delta l K production
              ic=3
                LB(I1) = 6 + int(4 * RANART(NSEED))
              lb(i2)=14
              go to 208
          else
              if(xsk2/(xsk2+xsk4).gt.RANART(NSEED))then
* n sigma k production
                   LB(I1) = 1 + int(2 * RANART(NSEED))
                   LB(I2) = 15 + int(3 * RANART(NSEED))
              ic=2
              else
              ic=4
                LB(I1) = 6 + int(4 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              endif
              go to 208
          endif
       endif
       endif
208             continue
         if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
          lb(i1) = - lb(i1)
          lb(i2) = - lb(i2)
          if(LPION(NNN,IRUN) .eq. 23)LPION(NNN,IRUN)=21
         endif
* KEEP ALL COORDINATES OF PARTICLE 2 FOR POSSIBLE PHASE SPACE CHANGE
           NTRY1=0
127        CALL BBKAON(ic,SRT,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.20))GO TO 127
c       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
* (1) for the necleon/delta
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
              E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
             lbi1=lb(i1)
* (2) for the lambda/sigma
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
              lbi2=lb(i2)
* GET the kaon'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(aka**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008
c2003        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2003
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                RPION(1,NNN,IRUN)=R(1,I1)
                RPION(2,NNN,IRUN)=R(2,I1)
                RPION(3,NNN,IRUN)=R(3,I1)
c
* assign the nucleon/delta and lambda/sigma to i1 or i2 to keep the 
* leadng particle behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
              p(1,i1)=pt1i1
              p(2,i1)=pt2i1
              p(3,i1)=pt3i1
              e(i1)=eti1
              lb(i1)=lbi1
              p(1,i2)=pt1i2
              p(2,i2)=pt2i2
              p(3,i2)=pt3i2
              e(i2)=eti2
              lb(i2)=lbi2
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
              go to 90005
* FOR THE NN-->Delta+Delta+rho PROCESS, FIND MOMENTUM OF THE FINAL 
* PARTICLES IN THE NUCLEUS-NUCLEUS CMS.
307     CONTINUE
           NTRY1=0
125        CALL DDrho(SRT,ISEED,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,amrho,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.20))GO TO 125
C       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
                NNN=NNN+1
              arho=amrho
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
* (1) FOR P+P
              XDIR=RANART(NSEED)
                IF(LB(I1)*LB(I2).EQ.1)THEN
                IF(XDIR.Le.0.2)then
* (1.1)P+P-->D+++D0+rho(0)
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
              LB(I1)=9
              LB(I2)=7
       GO TO 2051
                ENDIF
* (1.2)P+P -->D++D+rho(0)
                IF((XDIR.LE.0.4).AND.(XDIR.GT.0.2))THEN
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
                LB(I1)=8
                LB(I2)=8
       GO TO 2051
              ENDIF 
* (1.3)P+P-->D+++D+arho(-)
                IF((XDIR.LE.0.6).AND.(XDIR.GT.0.4))THEN
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=9
                LB(I2)=8
       GO TO 2051
              ENDIF 
                IF((XDIR.LE.0.8).AND.(XDIR.GT.0.6))THEN
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=9
                LB(I2)=6
       GO TO 2051
              ENDIF 
                IF(XDIR.GT.0.8)THEN
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=8
       GO TO 2051
              ENDIF 
               ENDIF
* (2)FOR N+N
                IF(iabs(LB(I1)).EQ.2.AND.iabs(LB(I2)).EQ.2)THEN
                IF(XDIR.Le.0.2)then
* (2.1)N+N-->D++D-+rho(0)
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
              LB(I1)=6
              LB(I2)=7
       GO TO 2051
                ENDIF
* (2.2)N+N -->D+++D-+rho(-)
                IF((XDIR.LE.0.4).AND.(XDIR.GT.0.2))THEN
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=6
                LB(I2)=9
       GO TO 2051
              ENDIF 
* (2.3)P+P-->D0+D-+rho(+)
                IF((XDIR.GT.0.4).AND.(XDIR.LE.0.6))THEN
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=9
                LB(I2)=8
       GO TO 2051
              ENDIF 
* (2.4)P+P-->D0+D0+rho(0)
                IF((XDIR.GT.0.6).AND.(XDIR.LE.0.8))THEN
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=7
       GO TO 2051
              ENDIF 
* (2.5)P+P-->D0+D++rho(-)
                IF(XDIR.GT.0.8)THEN
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=8
       GO TO 2051
              ENDIF 
              ENDIF
* (3)FOR N+P
                IF(LB(I1)*LB(I2).EQ.2)THEN
                IF(XDIR.Le.0.17)then
* (3.1)N+P-->D+++D-+rho(0)
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
              LB(I1)=6
              LB(I2)=9
       GO TO 2051
                ENDIF
* (3.2)N+P -->D+++D0+rho(-)
                IF((XDIR.LE.0.34).AND.(XDIR.GT.0.17))THEN
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=9
       GO TO 2051
              ENDIF 
* (3.3)N+P-->D++D-+rho(+)
                IF((XDIR.GT.0.34).AND.(XDIR.LE.0.51))THEN
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=8
       GO TO 2051
              ENDIF 
* (3.4)N+P-->D++D++rho(-)
                IF((XDIR.GT.0.51).AND.(XDIR.LE.0.68))THEN
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=8
                LB(I2)=8
       GO TO 2051
              ENDIF 
* (3.5)N+P-->D0+D++rho(0)
                IF((XDIR.GT.0.68).AND.(XDIR.LE.0.85))THEN
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=8
       GO TO 2051
              ENDIF 
* (3.6)N+P-->D0+D0+rho(+)
                IF(XDIR.GT.0.85)THEN
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=7
                LB(I2)=7
              ENDIF 
                ENDIF
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
2051          E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
c
             if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
               lb(i1) = -lb(i1)
               lb(i2) = -lb(i2)
                if(LPION(NNN,IRUN) .eq. 25)then
                  LPION(NNN,IRUN)=27
                elseif(LPION(NNN,IRUN) .eq. 27)then
                  LPION(NNN,IRUN)=25
                endif
               endif
c
             lb1=lb(i1)
* FOR DELTA2
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
              lb2=lb(i2)
* assign delta1 and delta2 to i1 or i2 to keep the leadng particle
* behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
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
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
                IBLOCK=44
* GET rho'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(EPION(NNN,IRUN)**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008:
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008:
c2004        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2004
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                RPION(1,NNN,IRUN)=R(1,I1)
                RPION(2,NNN,IRUN)=R(2,I1)
                RPION(3,NNN,IRUN)=R(3,I1)
c
              go to 90005
* FOR THE NN-->N+N+rho PROCESS, FIND MOMENTUM OF THE FINAL 
* PARTICLES IN THE NUCLEUS-NUCLEUS CMS.
308     CONTINUE
           NTRY1=0
126        CALL pprho(SRT,ISEED,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,amrho,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.20))GO TO 126
C       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
                NNN=NNN+1
              arho=amrho
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
* (1) FOR P+P
              XDIR=RANART(NSEED)
                IF(LB(I1)*LB(I2).EQ.1)THEN
                IF(XDIR.Le.0.5)then
* (1.1)P+P-->P+P+rho(0)
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
              LB(I1)=1
              LB(I2)=1
       GO TO 2052
                Else
* (1.2)P+P -->p+n+rho(+)
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=1
                LB(I2)=2
       GO TO 2052
              ENDIF 
              endif
* (2)FOR N+N
                IF(iabs(LB(I1)).EQ.2.AND.iabs(LB(I2)).EQ.2)THEN
                IF(XDIR.Le.0.5)then
* (2.1)N+N-->N+N+rho(0)
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
              LB(I1)=2
              LB(I2)=2
       GO TO 2052
                Else
* (2.2)N+N -->N+P+rho(-)
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=1
                LB(I2)=2
       GO TO 2052
              ENDIF 
              endif
* (3)FOR N+P
                IF(LB(I1)*LB(I2).EQ.2)THEN
                IF(XDIR.Le.0.33)then
* (3.1)N+P-->N+P+rho(0)
                LPION(NNN,IRUN)=26
                EPION(NNN,IRUN)=Arho
              LB(I1)=1
              LB(I2)=2
       GO TO 2052
* (3.2)N+P -->P+P+rho(-)
                else IF((XDIR.LE.0.67).AND.(XDIR.GT.0.34))THEN
                LPION(NNN,IRUN)=25
                EPION(NNN,IRUN)=Arho
                LB(I1)=1
                LB(I2)=1
       GO TO 2052
              Else 
* (3.3)N+P-->N+N+rho(+)
                LPION(NNN,IRUN)=27
                EPION(NNN,IRUN)=Arho
                LB(I1)=2
                LB(I2)=2
       GO TO 2052
              ENDIF 
              endif
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
2052          E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
c
              if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
               lb(i1) = -lb(i1)
               lb(i2) = -lb(i2)
                if(LPION(NNN,IRUN) .eq. 25)then
                  LPION(NNN,IRUN)=27
                elseif(LPION(NNN,IRUN) .eq. 27)then
                  LPION(NNN,IRUN)=25
                endif
               endif
c
             lb1=lb(i1)
* FOR p2
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
              lb2=lb(i2)
* assign p1 and p2 to i1 or i2 to keep the leadng particle
* behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
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
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
                IBLOCK=45
* GET rho'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(EPION(NNN,IRUN)**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008:
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008:
c2005        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2005
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                RPION(1,NNN,IRUN)=R(1,I1)
                RPION(2,NNN,IRUN)=R(2,I1)
                RPION(3,NNN,IRUN)=R(3,I1)
c
              go to 90005
* FOR THE NN-->p+p+omega PROCESS, FIND MOMENTUM OF THE FINAL 
* PARTICLES IN THE NUCLEUS-NUCLEUS CMS.
309     CONTINUE
           NTRY1=0
138        CALL ppomga(SRT,ISEED,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.20))GO TO 138
C       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
                NNN=NNN+1
              aomega=0.782
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
* (1) FOR P+P
                IF(LB(I1)*LB(I2).EQ.1)THEN
* (1.1)P+P-->P+P+omega(0)
                LPION(NNN,IRUN)=28
                EPION(NNN,IRUN)=Aomega
              LB(I1)=1
              LB(I2)=1
       GO TO 2053
                ENDIF
* (2)FOR N+N
                IF(iabs(LB(I1)).EQ.2.AND.iabs(LB(I2)).EQ.2)THEN
* (2.1)N+N-->N+N+omega(0)
                LPION(NNN,IRUN)=28
                EPION(NNN,IRUN)=Aomega
              LB(I1)=2
              LB(I2)=2
       GO TO 2053
                ENDIF
* (3)FOR N+P
                IF(LB(I1)*LB(I2).EQ.2)THEN
* (3.1)N+P-->N+P+omega(0)
                LPION(NNN,IRUN)=28
                EPION(NNN,IRUN)=Aomega
              LB(I1)=1
              LB(I2)=2
       GO TO 2053
                ENDIF
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
2053          E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
              if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
               lb(i1) = -lb(i1)
               lb(i2) = -lb(i2)
               endif
             lb1=lb(i1)
* FOR DELTA2
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
                lb2=lb(i2)
* assign delta1 and delta2 to i1 or i2 to keep the leadng particle
* behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
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
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
                IBLOCK=46
* GET omega'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(EPION(NNN,IRUN)**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008:
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008:
c2006        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2006
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                    RPION(1,NNN,IRUN)=R(1,I1)
                    RPION(2,NNN,IRUN)=R(2,I1)
                    RPION(3,NNN,IRUN)=R(3,I1)
c
              go to 90005
* change phase space density FOR NUCLEONS AFTER THE PROCESS

clin-10/25/02-comment out following, since there is no path to it:
clin-8/16/02 used before set
c     IX1,IY1,IZ1,IPX1,IPY1,IPZ1, IX2,IY2,IZ2,IPX2,IPY2,IPZ2:
c                if ((abs(ix1).le.mx) .and. (abs(iy1).le.my) .and.
c     &              (abs(iz1).le.mz)) then
c                  ipx1p = nint(p(1,i1)/dpx)
c                  ipy1p = nint(p(2,i1)/dpy)
c                  ipz1p = nint(p(3,i1)/dpz)
c                  if ((ipx1p.ne.ipx1) .or. (ipy1p.ne.ipy1) .or.
c     &                (ipz1p.ne.ipz1)) then
c                    if ((abs(ipx1).le.mpx) .and. (abs(ipy1).le.my)
c     &                .and. (ipz1.ge.-mpz) .and. (ipz1.le.mpzp))
c     &                f(ix1,iy1,iz1,ipx1,ipy1,ipz1) =
c     &                f(ix1,iy1,iz1,ipx1,ipy1,ipz1) - 1.
c                    if ((abs(ipx1p).le.mpx) .and. (abs(ipy1p).le.my)
c     &                .and. (ipz1p.ge.-mpz).and. (ipz1p.le.mpzp))
c     &                f(ix1,iy1,iz1,ipx1p,ipy1p,ipz1p) =
c     &                f(ix1,iy1,iz1,ipx1p,ipy1p,ipz1p) + 1.
c                  end if
c                end if
c                if ((abs(ix2).le.mx) .and. (abs(iy2).le.my) .and.
c     &              (abs(iz2).le.mz)) then
c                  ipx2p = nint(p(1,i2)/dpx)
c                  ipy2p = nint(p(2,i2)/dpy)
c                  ipz2p = nint(p(3,i2)/dpz)
c                  if ((ipx2p.ne.ipx2) .or. (ipy2p.ne.ipy2) .or.
c     &                (ipz2p.ne.ipz2)) then
c                    if ((abs(ipx2).le.mpx) .and. (abs(ipy2).le.my)
c     &                .and. (ipz2.ge.-mpz) .and. (ipz2.le.mpzp))
c     &                f(ix2,iy2,iz2,ipx2,ipy2,ipz2) =
c     &                f(ix2,iy2,iz2,ipx2,ipy2,ipz2) - 1.
c                    if ((abs(ipx2p).le.mpx) .and. (abs(ipy2p).le.my)
c     &                .and. (ipz2p.ge.-mpz) .and. (ipz2p.le.mpzp))
c     &                f(ix2,iy2,iz2,ipx2p,ipy2p,ipz2p) =
c     &                f(ix2,iy2,iz2,ipx2p,ipy2p,ipz2p) + 1.
c                  end if
c                end if
clin-10/25/02-end

90005       continue
       RETURN
*-----------------------------------------------------------------------
*COM: SET THE NEW MOMENTUM COORDINATES
107     IF(PX .EQ. 0.0 .AND. PY .EQ. 0.0) THEN
        T2 = 0.0
      ELSE
        T2=ATAN2(PY,PX)
      END IF
      S1   = 1.0 - C1**2 
       IF(S1.LE.0)S1=0
       S1=SQRT(S1)
      S2  =  SQRT( 1.0 - C2**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      CT2  = COS(T2)
      ST2  = SIN(T2)
      PZ   = PR * ( C1*C2 - S1*S2*CT1 )
      SS   = C2 * S1 * CT1  +  S2 * C1
      PX   = PR * ( SS*CT2 - S1*ST1*ST2 )
      PY   = PR * ( SS*ST2 + S1*ST1*CT2 )
      RETURN
      END
clin-5/2008 CRNN over

**********************************
**********************************
*                                                                      *
*                                                                      *
c
      SUBROUTINE CRPP(PX,PY,PZ,SRT,I1,I2,IBLOCK,
     &ppel,ppin,spprho,ipp)
*     PURPOSE:                                                         *
*             DEALING WITH PION-PION COLLISIONS                        *
*     NOTE   :                                                         *
*           VALID ONLY FOR PION-PION-DISTANCES LESS THAN 2.5 FM        *
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     6-> Meson+Meson elastic
*                     66-> Meson+meson-->K+K-
**********************************
      PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1     AMP=0.93828,AP1=0.13496,
     2 AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
      PARAMETER      (AKA=0.498,aks=0.895)
      parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
      COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
      COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
      COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

      lb1i=lb(i1)
      lb2i=lb(i2)

       PX0=PX
       PY0=PY
       PZ0=PZ
        iblock=1
*-----------------------------------------------------------------------
* check Meson+Meson inelastic collisions
clin-9/28/00
c        if((srt.gt.1.).and.(ppin/(ppin+ppel).gt.RANART(NSEED)))then
c        iblock=66
c        e(i1)=0.498
c        e(i2)=0.498
c        lb(i1)=21
c        lb(i2)=23
c        go to 10
clin-11/07/00
c        if(srt.gt.1.and.(ppin/(ppin+ppel)).gt.RANART(NSEED)) then
clin-4/03/02
        if(srt.gt.(2*aka).and.(ppin/(ppin+ppel)).gt.RANART(NSEED)) then
c        if(ppin/(ppin+ppel).gt.RANART(NSEED)) then
clin-10/08/00

           ranpi=RANART(NSEED)
           if((pprr/ppin).ge.ranpi) then

c     1) pi pi <-> rho rho:
              call pi2ro2(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)

clin-4/03/02 eta equilibration:
           elseif((pprr+ppee)/ppin.ge.ranpi) then
c     4) pi pi <-> eta eta:
              call pi2et2(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
           elseif(((pprr+ppee+pppe)/ppin).ge.ranpi) then
c     5) pi pi <-> pi eta:
              call pi3eta(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
           elseif(((pprr+ppee+pppe+rpre)/ppin).ge.ranpi) then
c     6) rho pi <-> pi eta:
              call rpiret(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
           elseif(((pprr+ppee+pppe+rpre+xopoe)/ppin).ge.ranpi) then
c     7) omega pi <-> omega eta:
              call opioet(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
           elseif(((pprr+ppee+pppe+rpre+xopoe+rree)
     1             /ppin).ge.ranpi) then
c     8) rho rho <-> eta eta:
              call ro2et2(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
clin-4/03/02-end

c     2) BBbar production:
           elseif(((pprr+ppee+pppe+rpre+xopoe+rree+ppinnb)/ppin)
     1             .ge.ranpi) then

              call bbarfs(lbb1,lbb2,ei1,ei2,iblock,iseed)
c     3) KKbar production:
           else
              iblock=66
              ei1=aka
              ei2=aka
              lbb1=21
              lbb2=23
clin-11/07/00 pi rho -> K* Kbar and K*bar K productions:
              lb1=lb(i1)
              lb2=lb(i2)
clin-2/13/03 include omega the same as rho, eta the same as pi:
c        if(((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.25.and.lb2.le.27))
c     1  .or.((lb2.ge.3.and.lb2.le.5).and.(lb1.ge.25.and.lb1.le.27)))
        if( ( (lb1.eq.0.or.(lb1.ge.3.and.lb1.le.5))
     1       .and.(lb2.ge.25.and.lb2.le.28))
     2       .or. ( (lb2.eq.0.or.(lb2.ge.3.and.lb2.le.5))
     3       .and.(lb1.ge.25.and.lb1.le.28))) then
           ei1=aks
           ei2=aka
           if(RANART(NSEED).ge.0.5) then
              iblock=366
              lbb1=30
              lbb2=21
           else
              iblock=367
              lbb1=-30
              lbb2=23
           endif
        endif
clin-11/07/00-end
           endif
clin-ppbar-8/25/00
           e(i1)=ei1
           e(i2)=ei2
           lb(i1)=lbb1
           lb(i2)=lbb2
clin-10/08/00-end

       else
cbzdbg10/15/99
c.....for meson+meson elastic srt.le.2Mk, if not pi+pi collision return
         if ((lb(i1).lt.3.or.lb(i1).gt.5).and.
     &        (lb(i2).lt.3.or.lb(i2).gt.5)) return
cbzdbg10/15/99 end

* check Meson+Meson elastic collisions
        IBLOCK=6
* direct process
       if(ipp.eq.1.or.ipp.eq.4.or.ipp.eq.6)go to 10
       if(spprho/ppel.gt.RANART(NSEED))go to 20
       endif
10      NTAG=0
        EM1=E(I1)
        EM2=E(I2)

*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* for isotropic distribution no need to ROTATE THE MOMENTUM

* ROTATE IT 
      CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 

      RETURN
20       continue
       iblock=666
* treat rho formation in pion+pion collisions
* calculate the mass and momentum of rho in the nucleus-nucleus frame
       call rhores(i1,i2)
       if(ipp.eq.2)lb(i1)=27
       if(ipp.eq.3)lb(i1)=26
       if(ipp.eq.5)lb(i1)=25
       return       
      END
**********************************
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRND(IRUN,PX,PY,PZ,SRT,I1,I2,IBLOCK,
     &SIGNN,SIG,sigk,xsk1,xsk2,xsk3,xsk4,xsk5,NT,ipert1)
*     PURPOSE:                                                         *
*             DEALING WITH NUCLEON-BARYON RESONANCE COLLISIONS         *
*     NOTE   :                                                         *
*           VALID ONLY FOR BARYON-BARYON-DISTANCES LESS THAN 1.32 FM   *
*           (1.32 = 2 * HARD-CORE-RADIUS [HRC] )                       *
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           NSTAR =1 INCLUDING N* RESORANCE,ELSE NOT                   *
*           NDIRCT=1 INCLUDING DIRECT PION PRODUCTION PROCESS         *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      0-> COLLISION CANNOT HAPPEN                     *
*                      1-> N-N ELASTIC COLLISION                       *
*                      2-> N+N->N+DELTA,OR N+N->N+N* REACTION          *
*                      3-> N+DELTA->N+N OR N+N*->N+N REACTION          *
*                      4-> N+N->N+N+PION,DIRTCT PROCESS                *
*           N12       - IS USED TO SPECIFY BARYON-BARYON REACTION      *
*                      CHANNELS. M12 IS THE REVERSAL CHANNEL OF N12    *
*                      N12,                                            *
*                      M12=1 FOR p+n-->delta(+)+ n                     *
*                          2     p+n-->delta(0)+ p                     *
*                          3     p+p-->delta(++)+n                     *
*                          4     p+p-->delta(+)+p                      *
*                          5     n+n-->delta(0)+n                      *
*                          6     n+n-->delta(-)+p                      *
*                          7     n+p-->N*(0)(1440)+p                   *
*                          8     n+p-->N*(+)(1440)+n                   *
*                        9     p+p-->N*(+)(1535)+p                     *
*                        10    n+n-->N*(0)(1535)+n                     *
*                         11    n+p-->N*(+)(1535)+n                     *
*                        12    n+p-->N*(0)(1535)+p
*                        13    D(++)+D(-)-->N*(+)(1440)+n
*                         14    D(++)+D(-)-->N*(0)(1440)+p
*                        15    D(+)+D(0)--->N*(+)(1440)+n
*                        16    D(+)+D(0)--->N*(0)(1440)+p
*                        17    D(++)+D(0)-->N*(+)(1535)+p
*                        18    D(++)+D(-)-->N*(0)(1535)+p
*                        19    D(++)+D(-)-->N*(+)(1535)+n
*                        20    D(+)+D(+)-->N*(+)(1535)+p
*                        21    D(+)+D(0)-->N*(+)(1535)+n
*                        22    D(+)+D(0)-->N*(0)(1535)+p
*                        23    D(+)+D(-)-->N*(0)(1535)+n
*                        24    D(0)+D(0)-->N*(0)(1535)+n
*                          25    N*(+)(14)+N*(+)(14)-->N*(+)(15)+p
*                          26    N*(0)(14)+N*(0)(14)-->N*(0)(15)+n
*                          27    N*(+)(14)+N*(0)(14)-->N*(+)(15)+n
*                        28    N*(+)(14)+N*(0)(14)-->N*(0)(15)+p
*                        29    N*(+)(14)+D+-->N*(+)(15)+p
*                        30    N*(+)(14)+D0-->N*(+)(15)+n
*                        31    N*(+)(14)+D(-)-->N*(0)(1535)+n
*                        32    N*(0)(14)+D++--->N*(+)(15)+p
*                        33    N*(0)(14)+D+--->N*(+)(15)+n
*                        34    N*(0)(14)+D+--->N*(0)(15)+p
*                        35    N*(0)(14)+D0-->N*(0)(15)+n
*                        36    N*(+)(14)+D0--->N*(0)(15)+p
*                        ++    see the note book for more listing
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AKA=0.498,APHI=1.020,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        parameter (xmd=1.8756,npdmax=10000)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
        common /gg/ dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
        COMMON /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
        COMMON /NN/NNN
cc      SAVE /NN/
        COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
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
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1 px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      common /dpi/em2,lb2
      common /para8/ idpert,npertd,idxsec
      dimension ppd(3,npdmax),lbpd(npdmax)
      SAVE   
*-----------------------------------------------------------------------
       n12=0
       m12=0
        IBLOCK=0
        NTAG=0
        EM1=E(I1)
        EM2=E(I2)
        PR  = SQRT( PX**2 + PY**2 + PZ**2 )
        C2  = PZ / PR
        X1  = RANART(NSEED)
        ianti=0
        if(lb(i1).lt.0 .and. lb(i2).lt.0)ianti=1

clin-6/2008 Production of perturbative deuterons for idpert=1:
      call sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
      if(idpert.eq.1.and.ipert1.eq.1) then
         IF (SRT .LT. 2.012) RETURN
         if((iabs(lb(i1)).eq.1.or.iabs(lb(i1)).eq.2)
     1        .and.(iabs(lb(i2)).ge.6.and.iabs(lb(i2)).le.13)) then
            goto 108
         elseif((iabs(lb(i2)).eq.1.or.iabs(lb(i2)).eq.2)
     1           .and.(iabs(lb(i1)).ge.6.and.iabs(lb(i1)).le.13)) then
            goto 108
         else
            return
         endif
      endif
*-----------------------------------------------------------------------
*COM: TEST FOR ELASTIC SCATTERING (EITHER N-N OR DELTA-DELTA 0R
*      N-DELTA OR N*-N* or N*-Delta)
      IF (X1 .LE. SIGNN/SIG) THEN
*COM:  PARAMETRISATION IS TAKEN FROM THE CUGNON-PAPER
        AS  = ( 3.65 * (SRT - 1.8766) )**6
        A   = 6.0 * AS / (1.0 + AS)
        TA  = -2.0 * PR**2
        X   = RANART(NSEED)
clin-10/24/02        T1  = ALOG( (1-X) * DEXP(dble(A)*dble(TA)) + X )  /  A
        T1  = sngl(DLOG(dble(1.-X)*DEXP(dble(A)*dble(TA))+dble(X)))/  A
        C1  = 1.0 - T1/TA
        T1  = 2.0 * PI * RANART(NSEED)
        IBLOCK=1
       GO TO 107
      ELSE
*COM: TEST FOR INELASTIC SCATTERING
*     IF THE AVAILABLE ENERGY IS LESS THAN THE PION-MASS, NOTHING
*     CAN HAPPEN ANY MORE ==> RETURN (2.04 = 2*AVMASS + PI-MASS+0.02)
        IF (SRT .LT. 2.04) RETURN
clin-6/2008 add d+meson production for n*N*(0)(1440) and p*N*(+)(1440) channels
c     (they did not have any inelastic reactions before):
        if(((iabs(LB(I1)).EQ.2.or.iabs(LB(I2)).EQ.2).AND.
     1       (LB(I1)*LB(I2)).EQ.20).or.(LB(I1)*LB(I2)).EQ.13) then
           IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
        ENDIF
c
* Resonance absorption or Delta + N-->N*(1440), N*(1535)
* COM: TEST FOR DELTA OR N* ABSORPTION
*      IN THE PROCESS DELTA+N-->NN, N*+N-->NN
        PRF=SQRT(0.25*SRT**2-AVMASS**2)
        IF(EM1.GT.1.)THEN
        DELTAM=EM1
        ELSE
        DELTAM=EM2
        ENDIF
        RENOM=DELTAM*PRF**2/DENOM(SRT,1.)/PR
        RENOMN=DELTAM*PRF**2/DENOM(SRT,2.)/PR
        RENOM1=DELTAM*PRF**2/DENOM(SRT,-1.)/PR
* avoid the inelastic collisions between n+delta- -->N+N 
*       and p+delta++ -->N+N due to charge conservation,
*       but they can scatter to produce kaons 
       if((iabs(lb(i1)).eq.2).and.(iabs(lb(i2)).eq.6)) renom=0.
       if((iabs(lb(i2)).eq.2).and.(iabs(lb(i1)).eq.6)) renom=0.
       if((iabs(lb(i1)).eq.1).and.(iabs(lb(i2)).eq.9)) renom=0.
       if((iabs(lb(i2)).eq.1).and.(iabs(lb(i1)).eq.9)) renom=0.
       Call M1535(iabs(lb(i1)),iabs(lb(i2)),srt,x1535)
        X1440=(3./4.)*SIGMA(SRT,2,0,1)
* CROSS SECTION FOR KAON PRODUCTION from the four channels
* for NLK channel
* avoid the inelastic collisions between n+delta- -->N+N 
*       and p+delta++ -->N+N due to charge conservation,
*       but they can scatter to produce kaons 
       if(((iabs(lb(i1)).eq.2).and.(iabs(lb(i2)).eq.6)).OR. 
     &         ((iabs(lb(i2)).eq.2).and.(iabs(lb(i1)).eq.6)).OR.
     &         ((iabs(lb(i1)).eq.1).and.(iabs(lb(i2)).eq.9)).OR.
     &         ((iabs(lb(i2)).eq.1).and.(iabs(lb(i1)).eq.9)))THEN
clin-6/2008
          IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c          IF((SIGK+SIGNN)/SIG.GE.X1)GO TO 306
          IF((SIGK+SIGNN+sdprod)/SIG.GE.X1)GO TO 306
c
       ENDIF
* WE DETERMINE THE REACTION CHANNELS IN THE FOLLOWING
* FOR n+delta(++)-->p+p or n+delta(++)-->n+N*(+)(1440),n+N*(+)(1535)
* REABSORPTION OR N*(1535) PRODUCTION LIKE IN P+P OR N*(1440) LIKE PN, 
        IF(LB(I1)*LB(I2).EQ.18.AND.
     &  (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))then
        SIGND=SIGMA(SRT,1,1,0)+0.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+X1440+X1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+X1440+X1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+X1440+X1535).GT.RANART(NSEED))GO TO 306
* REABSORPTION:
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+X1440+X1535))THEN
        M12=3
       GO TO 206
       ELSE
* N* PRODUCTION
              IF(RANART(NSEED).LT.X1440/(X1440+X1535))THEN
* N*(1440)
              M12=37
              ELSE
* N*(1535)       M12=38
clin-2/26/03 why is the above commented out? leads to M12=0 but 
c     particle mass is changed after 204 (causes energy violation).
c     replace by elastic process (return):
                   return

              ENDIF
       GO TO 204
       ENDIF
        ENDIF
* FOR p+delta(-)-->n+n or p+delta(-)-->n+N*(0)(1440),n+N*(0)(1535)
* REABSORPTION OR N*(1535) PRODUCTION LIKE IN P+P OR N*(1440) LIKE PN, 
        IF(LB(I1)*LB(I2).EQ.6.AND.
     &   ((iabs(LB(I1)).EQ.1).OR.(iabs(LB(I2)).EQ.1)))then
        SIGND=SIGMA(SRT,1,1,0)+0.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF (X1.GT.(SIGNN+SIGDN+X1440+X1535+SIGK)/SIG)RETURN
        IF (X1.GT.(SIGNN+SIGDN+X1440+X1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+X1440+X1535).GT.RANART(NSEED))GO TO 306
* REABSORPTION:
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+X1440+X1535))THEN
        M12=6
       GO TO 206
       ELSE
* N* PRODUCTION
              IF(RANART(NSEED).LT.X1440/(X1440+X1535))THEN
* N*(1440)
              M12=47
              ELSE
* N*(1535)       M12=48
clin-2/26/03 causes energy violation, replace by elastic process (return):
                   return

              ENDIF
       GO TO 204
       ENDIF
        ENDIF
* FOR p+delta(+)-->p+p, N*(+)(144)+p, N*(+)(1535)+p
        IF(LB(I1)*LB(I2).EQ.8.AND.
     &   (iabs(LB(I1)).EQ.1.OR.iabs(LB(I2)).EQ.1))THEN
        SIGND=1.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+x1440+x1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+x1440+x1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+X1440+X1535).GT.RANART(NSEED))GO TO 306
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+X1440+X1535))THEN
        M12=4
       GO TO 206
       ELSE
              IF(RANART(NSEED).LT.X1440/(X1440+X1535))THEN
* N*(144)
              M12=39
              ELSE
              M12=40
              ENDIF
              GO TO 204
       ENDIF
        ENDIF
* FOR n+delta(0)-->n+n, N*(0)(144)+n, N*(0)(1535)+n
        IF(LB(I1)*LB(I2).EQ.14.AND.
     &   (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))THEN
        SIGND=1.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+x1440+x1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+x1440+x1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+X1440+X1535).GT.RANART(NSEED))GO TO 306
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+X1440+X1535))THEN
        M12=5
       GO TO 206
       ELSE
              IF(RANART(NSEED).LT.X1440/(X1440+X1535))THEN
* N*(144)
              M12=48
              ELSE
              M12=49
              ENDIF
              GO TO 204
       ENDIF
        ENDIF
* FOR n+delta(+)-->n+p, N*(+)(1440)+n,N*(0)(1440)+p,
*                       N*(+)(1535)+n,N*(0)(1535)+p
        IF(LB(I1)*LB(I2).EQ.16.AND.
     &   (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))THEN
        SIGND=0.5*SIGMA(SRT,1,1,1)+0.25*SIGMA(SRT,1,1,0)
        SIGDN=0.5*SIGND*RENOM
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+2.*x1440+2.*x1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+2.*x1440+2.*x1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+2*X1440+2*X1535).GT.RANART(NSEED))GO TO 306
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+2.*X1440+2.*X1535))THEN
        M12=1
       GO TO 206
       ELSE
              IF(RANART(NSEED).LT.X1440/(X1440+X1535))THEN
              M12=41
              IF(RANART(NSEED).LE.0.5)M12=43
              ELSE
              M12=42
              IF(RANART(NSEED).LE.0.5)M12=44
              ENDIF
              GO TO 204
       ENDIF
        ENDIF
* FOR p+delta(0)-->n+p, N*(+)(1440)+n,N*(0)(1440)+p,
*                       N*(+)(1535)+n,N*(0)(1535)+p
        IF(LB(I1)*LB(I2).EQ.7)THEN
        SIGND=0.5*SIGMA(SRT,1,1,1)+0.25*SIGMA(SRT,1,1,0)
        SIGDN=0.5*SIGND*RENOM
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+2.*x1440+2.*x1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+2.*x1440+2.*x1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+2*X1440+2*X1535).GT.RANART(NSEED))GO TO 306
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+2.*X1440+2.*X1535))THEN
        M12=2
       GO TO 206
       ELSE
              IF(RANART(NSEED).LT.X1440/(X1440+X1535))THEN
              M12=50
              IF(RANART(NSEED).LE.0.5)M12=51
              ELSE
              M12=52
              IF(RANART(NSEED).LE.0.5)M12=53
              ENDIF
              GO TO 204
       ENDIF
        ENDIF
* FOR p+N*(0)(14)-->p+n, N*(+)(1535)+n,N*(0)(1535)+p
* OR  P+N*(0)(14)-->D(+)+N, D(0)+P, 
        IF(LB(I1)*LB(I2).EQ.10.AND.
     &  (iabs(LB(I1)).EQ.1.OR.iabs(LB(I2)).EQ.1))then
        SIGND=(3./4.)*SIGMA(SRT,2,0,1)
        SIGDN=SIGND*RENOMN
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+X1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+X1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+X1535).GT.RANART(NSEED))GO TO 306
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+X1535))THEN
        M12=7
        GO TO 206
       ELSE
       M12=54
       IF(RANART(NSEED).LE.0.5)M12=55
       ENDIF
       GO TO 204
        ENDIF
* FOR n+N*(+)-->p+n, N*(+)(1535)+n,N*(0)(1535)+p
        IF(LB(I1)*LB(I2).EQ.22.AND.
     &   (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))then
        SIGND=(3./4.)*SIGMA(SRT,2,0,1)
        SIGDN=SIGND*RENOMN
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+X1535+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+X1535+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN+X1535).GT.RANART(NSEED))GO TO 306
       IF(RANART(NSEED).LT.SIGDN/(SIGDN+X1535))THEN
        M12=8
        GO TO 206
       ELSE
       M12=56
       IF(RANART(NSEED).LE.0.5)M12=57
       ENDIF
       GO TO 204
        ENDIF
* FOR N*(1535)+N-->N+N COLLISIONS
        IF((iabs(LB(I1)).EQ.12).OR.(iabs(LB(I1)).EQ.13).OR.
     1  (iabs(LB(I2)).EQ.12).OR.(iabs(LB(I2)).EQ.13))THEN
        SIGND=X1535
        SIGDN=SIGND*RENOM1
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGDN+SIGK)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGDN+SIGK+sdprod)/SIG)RETURN
c
       IF(SIGK/(SIGK+SIGDN).GT.RANART(NSEED))GO TO 306
        IF(LB(I1)*LB(I2).EQ.24)M12=10
        IF(LB(I1)*LB(I2).EQ.12)M12=12
        IF(LB(I1)*LB(I2).EQ.26)M12=11
       IF(LB(I1)*LB(I2).EQ.13)M12=9
       GO TO 206
        ENDIF
204       CONTINUE
* (1) GENERATE THE MASS FOR THE N*(1440) AND N*(1535)
* (2) CALCULATE THE FINAL MOMENTUM OF THE n+N* SYSTEM
* (3) RELABLE THE FINAL STATE PARTICLES
*PARAMETRIZATION OF THE SHAPE OF THE N* RESONANCE ACCORDING
*     TO kitazoe's or J.D.JACKSON'S MASS FORMULA AND BREIT WIGNER
*     FORMULA FOR N* RESORANCE
*     DETERMINE DELTA MASS VIA REJECTION METHOD.
          DMAX = SRT - AVMASS-0.005
          DMIN = 1.078
          IF((M12.eq.37).or.(M12.eq.39).or.
     1    (M12.eQ.41).OR.(M12.eQ.43).OR.(M12.EQ.46).
     2     OR.(M12.EQ.48).OR.(M12.EQ.50).OR.(M12.EQ.51))then
* N*(1440) production
          IF(DMAX.LT.1.44) THEN
          FM=FNS(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FNS():
             xdmass=1.44
c          FM=FNS(1.44,SRT,1.)
          FM=FNS(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY2=0
11        DM=RANART(NSEED)*(DMAX-DMIN)+DMIN
          NTRY2=NTRY2+1
          IF((RANART(NSEED).GT.FNS(DM,SRT,1.)/FM).AND.
     1    (NTRY2.LE.10)) GO TO 11

clin-2/26/03 limit the N* mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.2.14) goto 11

              GO TO 13
              ELSE
* N*(1535) production
          IF(DMAX.LT.1.535) THEN
          FM=FD5(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FNS():
             xdmass=1.535
c          FM=FD5(1.535,SRT,1.)
          FM=FD5(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY1=0
12        DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
          NTRY1=NTRY1+1
          IF((RANART(NSEED) .GT. FD5(DM,SRT,1.)/FM).AND.
     1    (NTRY1.LE.10)) GOTO 12

clin-2/26/03 limit the N* mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.1.84) goto 12

             ENDIF
13       CONTINUE
* (2) DETERMINE THE FINAL MOMENTUM
       PRF=0.
       PF2=((SRT**2-DM**2+AVMASS**2)/(2.*SRT))**2-AVMASS**2
       IF(PF2.GT.0.)PRF=SQRT(PF2)
* (3) RELABLE FINAL STATE PARTICLES
* 37 D(++)+n-->N*(+)(14)+p
          IF(M12.EQ.37)THEN
          IF(iabs(LB(I1)).EQ.9)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=11
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=11
         E(I1)=DM
          ENDIF
         GO TO 207
          ENDIF
* 38 D(++)+n-->N*(+)(15)+p
          IF(M12.EQ.38)THEN
          IF(iabs(LB(I1)).EQ.9)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=13
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=13
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 39 D(+)+P-->N*(+)(14)+p
          IF(M12.EQ.39)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=11
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=11
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 40 D(+)+P-->N*(+)(15)+p
          IF(M12.EQ.40)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=13
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=13
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 41 D(+)+N-->N*(+)(14)+N
          IF(M12.EQ.41)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=11
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=11
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 42 D(+)+N-->N*(+)(15)+N
          IF(M12.EQ.42)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=13
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=13
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 43 D(+)+N-->N*(0)(14)+P
          IF(M12.EQ.43)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=10
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=10
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 44 D(+)+N-->N*(0)(15)+P
          IF(M12.EQ.44)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=12
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=12
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 46 D(-)+P-->N*(0)(14)+N
          IF(M12.EQ.46)THEN
          IF(iabs(LB(I1)).EQ.6)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=10
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=10
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 47 D(-)+P-->N*(0)(15)+N
          IF(M12.EQ.47)THEN
          IF(iabs(LB(I1)).EQ.6)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=12
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=12
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 48 D(0)+N-->N*(0)(14)+N
          IF(M12.EQ.48)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=11
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=11
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 49 D(0)+N-->N*(0)(15)+N
          IF(M12.EQ.49)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=12
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=12
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 50 D(0)+P-->N*(0)(14)+P
          IF(M12.EQ.50)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=10
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=10
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 51 D(0)+P-->N*(+)(14)+N
          IF(M12.EQ.51)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=11
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=11
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 52 D(0)+P-->N*(0)(15)+P
          IF(M12.EQ.52)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=12
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=12
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 53 D(0)+P-->N*(+)(15)+N
          IF(M12.EQ.53)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=13
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=13
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 54 N*(0)(14)+P-->N*(+)(15)+N
          IF(M12.EQ.54)THEN
          IF(iabs(LB(I1)).EQ.10)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=13
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=13
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 55 N*(0)(14)+P-->N*(0)(15)+P
          IF(M12.EQ.55)THEN
          IF(iabs(LB(I1)).EQ.10)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=12
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=12
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 56 N*(+)(14)+N-->N*(+)(15)+N
          IF(M12.EQ.56)THEN
          IF(iabs(LB(I1)).EQ.11)THEN
          LB(I1)=2
          E(I1)=AMN
         LB(I2)=13
         E(I2)=DM
          ELSE
          LB(I2)=2
          E(I2)=AMN
         LB(I1)=13
         E(I1)=DM
          ENDIF
         GO TO 207
         ENDIF
* 57 N*(+)(14)+N-->N*(0)(15)+P
          IF(M12.EQ.57)THEN
          IF(iabs(LB(I1)).EQ.11)THEN
          LB(I1)=1
          E(I1)=AMP
         LB(I2)=12
         E(I2)=DM
          ELSE
          LB(I2)=1
          E(I2)=AMP
         LB(I1)=12
         E(I1)=DM
          ENDIF
         ENDIF
          GO TO 207
*------------------------------------------------
* RELABLE NUCLEONS AFTER DELTA OR N* BEING ABSORBED
*(1) n+delta(+)-->n+p
206       IF(M12.EQ.1)THEN
          IF(iabs(LB(I1)).EQ.8)THEN
          LB(I2)=2
          LB(I1)=1
          E(I1)=AMP
          ELSE
          LB(I1)=2
          LB(I2)=1
          E(I2)=AMP
          ENDIF
         GO TO 207
          ENDIF
*(2) p+delta(0)-->p+n
          IF(M12.EQ.2)THEN
          IF(iabs(LB(I1)).EQ.7)THEN
          LB(I2)=1
          LB(I1)=2
          E(I1)=AMN
          ELSE
          LB(I1)=1
          LB(I2)=2
          E(I2)=AMN
          ENDIF
         GO TO 207
          ENDIF
*(3) n+delta(++)-->p+p
          IF(M12.EQ.3)THEN
          LB(I1)=1
          LB(I2)=1
          E(I1)=AMP
          E(I2)=AMP
         GO TO 207
          ENDIF
*(4) p+delta(+)-->p+p
          IF(M12.EQ.4)THEN
          LB(I1)=1
          LB(I2)=1
          E(I1)=AMP
          E(I2)=AMP
         GO TO 207
          ENDIF
*(5) n+delta(0)-->n+n
          IF(M12.EQ.5)THEN
          LB(I1)=2
          LB(I2)=2
          E(I1)=AMN
          E(I2)=AMN
         GO TO 207
          ENDIF
*(6) p+delta(-)-->n+n
          IF(M12.EQ.6)THEN
          LB(I1)=2
          LB(I2)=2
          E(I1)=AMN
          E(I2)=AMN
         GO TO 207
          ENDIF
*(7) p+N*(0)-->n+p
          IF(M12.EQ.7)THEN
          IF(iabs(LB(I1)).EQ.1)THEN
          LB(I1)=1
          LB(I2)=2
          E(I1)=AMP
          E(I2)=AMN
          ELSE
          LB(I1)=2
          LB(I2)=1
          E(I1)=AMN
          E(I2)=AMP
          ENDIF
         GO TO 207
          ENDIF
*(8) n+N*(+)-->n+p
          IF(M12.EQ.8)THEN
          IF(iabs(LB(I1)).EQ.2)THEN
          LB(I1)=2
          LB(I2)=1
          E(I1)=AMN
          E(I2)=AMP
          ELSE
          LB(I1)=1
          LB(I2)=2
          E(I1)=AMP
          E(I2)=AMN
          ENDIF
         GO TO 207
          ENDIF
clin-6/2008
c*(9) N*(+)p-->pp
*(9) N*(+)(1535) p-->pp
          IF(M12.EQ.9)THEN
          LB(I1)=1
          LB(I2)=1
          E(I1)=AMP
          E(I2)=AMP
         GO TO 207
         ENDIF
*(12) N*(0)P-->nP
          IF(M12.EQ.12)THEN
          LB(I1)=2
          LB(I2)=1
          E(I1)=AMN
          E(I2)=AMP
         GO TO 207
         ENDIF
*(11) N*(+)n-->nP
          IF(M12.EQ.11)THEN
          LB(I1)=2
          LB(I2)=1
          E(I1)=AMN
          E(I2)=AMP
         GO TO 207
         ENDIF
clin-6/2008
c*(12) N*(0)p-->Np
*(12) N*(0)(1535) p-->Np
          IF(M12.EQ.12)THEN
          LB(I1)=1
          LB(I2)=2
          E(I1)=AMP
          E(I2)=AMN
         ENDIF
*----------------------------------------------
207       PR   = PRF
          C1   = 1.0 - 2.0 * RANART(NSEED)
              if(srt.le.2.14)C1= 1.0 - 2.0 * RANART(NSEED)
         if(srt.gt.2.14.and.srt.le.2.4)c1=ang(srt,iseed)
         if(srt.gt.2.4)then

clin-10/25/02 get rid of argument usage mismatch in PTR():
             xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
         cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
         endif
          T1   = 2.0 * PI * RANART(NSEED)
          IBLOCK=3
      ENDIF
      if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
         lb(i1) = -lb(i1)
         lb(i2) = -lb(i2)
      endif

*-----------------------------------------------------------------------
*COM: SET THE NEW MOMENTUM COORDINATES
 107  IF(PX .EQ. 0.0 .AND. PY .EQ. 0.0) THEN
         T2 = 0.0
      ELSE
         T2=ATAN2(PY,PX)
      END IF
      S1   = SQRT( 1.0 - C1**2 )
      S2  =  SQRT( 1.0 - C2**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      CT2  = COS(T2)
      ST2  = SIN(T2)
      PZ   = PR * ( C1*C2 - S1*S2*CT1 )
      SS   = C2 * S1 * CT1  +  S2 * C1
      PX   = PR * ( SS*CT2 - S1*ST1*ST2 )
      PY   = PR * ( SS*ST2 + S1*ST1*CT2 )
      RETURN
* FOR THE NN-->KAON+X PROCESS, FIND MOMENTUM OF THE FINAL PARTICLES IN 
* THE NUCLEUS-NUCLEUS CMS.
306     CONTINUE
csp11/21/01 phi production
              if(XSK5/sigK.gt.RANART(NSEED))then
              pz1=p(3,i1)
              pz2=p(3,i2)
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 1 + int(2 * RANART(NSEED))
              nnn=nnn+1
                LPION(NNN,IRUN)=29
                EPION(NNN,IRUN)=APHI
                iblock = 222
              GO TO 208
               ENDIF
csp11/21/01 end
                IBLOCK=11
                if(ianti .eq. 1)iblock=-11
c
              pz1=p(3,i1)
              pz2=p(3,i2)
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
              nnn=nnn+1
                LPION(NNN,IRUN)=23
                EPION(NNN,IRUN)=Aka
              if(srt.le.2.63)then
* only lambda production is possible
* (1.1)P+P-->p+L+kaon+
              ic=1

                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              GO TO 208
                ENDIF
       if(srt.le.2.74.and.srt.gt.2.63)then
* both Lambda and sigma production are possible
              if(XSK1/(XSK1+XSK2).gt.RANART(NSEED))then
* lambda production
              ic=1

                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              else
* sigma production

                   LB(I1) = 1 + int(2 * RANART(NSEED))
                   LB(I2) = 15 + int(3 * RANART(NSEED))
              ic=2
              endif
              GO TO 208
       endif
       if(srt.le.2.77.and.srt.gt.2.74)then
* then pp-->Delta lamda kaon can happen
              if(xsk1/(xsk1+xsk2+xsk3).
     1          gt.RANART(NSEED))then
* * (1.1)P+P-->p+L+kaon+
              ic=1

                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              go to 208
              else
              if(xsk2/(xsk2+xsk3).gt.RANART(NSEED))then
* pp-->psk
              ic=2

                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))

              else
* pp-->D+l+k        
              ic=3

                LB(I1) = 6 + int(4 * RANART(NSEED))
              lb(i2)=14
              endif
              GO TO 208
              endif
       endif
       if(srt.gt.2.77)then
* all four channels are possible
              if(xsk1/(xsk1+xsk2+xsk3+xsk4).gt.RANART(NSEED))then
* p lambda k production
              ic=1

                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              go to 208
       else
          if(xsk3/(xsk2+xsk3+xsk4).gt.RANART(NSEED))then
* delta l K production
              ic=3

                LB(I1) = 6 + int(4 * RANART(NSEED))
              lb(i2)=14
              go to 208
          else
              if(xsk2/(xsk2+xsk4).gt.RANART(NSEED))then
* n sigma k production

                   LB(I1) = 1 + int(2 * RANART(NSEED))
                   LB(I2) = 15 + int(3 * RANART(NSEED))

              ic=2
              else
              ic=4

                LB(I1) = 6 + int(4 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))

              endif
              go to 208
          endif
       endif
       endif
208             continue
         if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
          lb(i1) = - lb(i1)
          lb(i2) = - lb(i2)
          if(LPION(NNN,IRUN) .eq. 23)LPION(NNN,IRUN)=21
         endif
       lbi1=lb(i1)
       lbi2=lb(i2)
* KEEP ALL COORDINATES OF PARTICLE 2 FOR POSSIBLE PHASE SPACE CHANGE
           NTRY1=0
128        CALL BBKAON(ic,SRT,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.20))GO TO 128
c       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
* (1) for the necleon/delta
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
              E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
* (2) for the lambda/sigma
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
* GET the kaon'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(aka**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008:
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008:
c2008        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2008
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                    RPION(1,NNN,IRUN)=R(1,I1)
                    RPION(2,NNN,IRUN)=R(2,I1)
                    RPION(3,NNN,IRUN)=R(3,I1)
c
* assign the nucleon/delta and lambda/sigma to i1 or i2 to keep the 
* leadng particle behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
              p(1,i1)=pt1i1
              p(2,i1)=pt2i1
              p(3,i1)=pt3i1
              e(i1)=eti1
              lb(i1)=lbi1
              p(1,i2)=pt1i2
              p(2,i2)=pt2i2
              p(3,i2)=pt3i2
              e(i2)=eti2
              lb(i2)=lbi2
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
                if(LPION(NNN,IRUN) .ne. 29) IBLOCK=11
        LB1=LB(I1)
        LB2=LB(I2)
        AM1=EM1
       am2=em2
        E1= SQRT( EM1**2 + PX1**2 + PY1**2 + PZ1**2 )
       RETURN

clin-6/2008 N+D->Deuteron+pi:
*     FIND MOMENTUM OF THE FINAL PARTICLES IN THE NUCLEUS-NUCLEUS CMS.
 108   CONTINUE
           if(idpert.eq.1.and.ipert1.eq.1.and.npertd.ge.1) then
c     For idpert=1: we produce npertd pert deuterons:
              ndloop=npertd
           elseif(idpert.eq.2.and.npertd.ge.1) then
c     For idpert=2: we first save information for npertd pert deuterons;
c     at the last ndloop we create the regular deuteron+pi 
c     and those pert deuterons:
              ndloop=npertd+1
           else
c     Just create the regular deuteron+pi:
              ndloop=1
           endif
c
           dprob1=sdprod/sig/float(npertd)
           do idloop=1,ndloop
              CALL bbdangle(pxd,pyd,pzd,nt,ipert1,ianti,idloop,pfinal,
     1 dprob1,lbm)
              CALL ROTATE(PX,PY,PZ,PXd,PYd,PZd)
*     LORENTZ-TRANSFORMATION OF THE MOMENTUM OF PARTICLES IN THE FINAL STATE 
*     FROM THE NN CMS FRAME INTO THE GLOBAL CMS FRAME:
*     For the Deuteron:
              xmass=xmd
              E1dCM=SQRT(xmass**2+PXd**2+PYd**2+PZd**2)
              P1dBETA=PXd*BETAX+PYd*BETAY+PZd*BETAZ
              TRANSF=GAMMA*(GAMMA*P1dBETA/(GAMMA+1.)+E1dCM)
              pxi1=BETAX*TRANSF+PXd
              pyi1=BETAY*TRANSF+PYd
              pzi1=BETAZ*TRANSF+PZd
              if(ianti.eq.0)then
                 lbd=42
              else
                 lbd=-42
              endif
              if(idpert.eq.1.and.ipert1.eq.1.and.npertd.ge.1) then
cccc  Perturbative production for idpert=1:
                 nnn=nnn+1
                 PPION(1,NNN,IRUN)=pxi1
                 PPION(2,NNN,IRUN)=pyi1
                 PPION(3,NNN,IRUN)=pzi1
                 EPION(NNN,IRUN)=xmd
                 LPION(NNN,IRUN)=lbd
                 RPION(1,NNN,IRUN)=R(1,I1)
                 RPION(2,NNN,IRUN)=R(2,I1)
                 RPION(3,NNN,IRUN)=R(3,I1)
clin-6/2008 assign the perturbative probability:
                 dppion(NNN,IRUN)=sdprod/sig/float(npertd)
              elseif(idpert.eq.2.and.idloop.le.npertd) then
clin-6/2008 For idpert=2, we produce NPERTD perturbative (anti)deuterons 
c     only when a regular (anti)deuteron+pi is produced in NN collisions.
c     First save the info for the perturbative deuterons:
                 ppd(1,idloop)=pxi1
                 ppd(2,idloop)=pyi1
                 ppd(3,idloop)=pzi1
                 lbpd(idloop)=lbd
              else
cccc  Regular production:
c     For the regular pion: do LORENTZ-TRANSFORMATION:
                 E(i1)=xmm
                 E2piCM=SQRT(xmm**2+PXd**2+PYd**2+PZd**2)
                 P2piBETA=-PXd*BETAX-PYd*BETAY-PZd*BETAZ
                 TRANSF=GAMMA*(GAMMA*P2piBETA/(GAMMA+1.)+E2piCM)
                 pxi2=BETAX*TRANSF-PXd
                 pyi2=BETAY*TRANSF-PYd
                 pzi2=BETAZ*TRANSF-PZd
                 p(1,i1)=pxi2
                 p(2,i1)=pyi2
                 p(3,i1)=pzi2
c     Remove regular pion to check the equivalence 
c     between the perturbative and regular deuteron results:
c                 E(i1)=0.
c
                 LB(I1)=lbm
                 PX1=P(1,I1)
                 PY1=P(2,I1)
                 PZ1=P(3,I1)
                 EM1=E(I1)
                 ID(I1)=2
                 ID1=ID(I1)
                 E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
                 lb1=lb(i1)
c     For the regular deuteron:
                 p(1,i2)=pxi1
                 p(2,i2)=pyi1
                 p(3,i2)=pzi1
                 lb(i2)=lbd
                 lb2=lb(i2)
                 E(i2)=xmd
                 EtI2=E(I2)
                 ID(I2)=2
c     For idpert=2: create the perturbative deuterons:
                 if(idpert.eq.2.and.idloop.eq.ndloop) then
                    do ipertd=1,npertd
                       nnn=nnn+1
                       PPION(1,NNN,IRUN)=ppd(1,ipertd)
                       PPION(2,NNN,IRUN)=ppd(2,ipertd)
                       PPION(3,NNN,IRUN)=ppd(3,ipertd)
                       EPION(NNN,IRUN)=xmd
                       LPION(NNN,IRUN)=lbpd(ipertd)
                       RPION(1,NNN,IRUN)=R(1,I1)
                       RPION(2,NNN,IRUN)=R(2,I1)
                       RPION(3,NNN,IRUN)=R(3,I1)
clin-6/2008 assign the perturbative probability:
                       dppion(NNN,IRUN)=1./float(npertd)
                    enddo
                 endif
              endif
           enddo
           IBLOCK=501
           return
clin-6/2008 N+D->Deuteron+pi over

      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRDD(IRUN,PX,PY,PZ,SRT,I1,I2,IBLOCK,
     1NTAG,SIGNN,SIG,NT,ipert1)
c     1NTAG,SIGNN,SIG)
*     PURPOSE:                                                         *
*             DEALING WITH BARYON RESONANCE-BARYON RESONANCE COLLISIONS*
*     NOTE   :                                                         *
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           NSTAR =1 INCLUDING N* RESORANCE,ELSE NOT                   *
*           NDIRCT=1 INCLUDING DIRECT PION PRODUCTION PROCESS         *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      0-> COLLISION CANNOT HAPPEN                     *
*                      1-> N-N ELASTIC COLLISION                       *
*                      2-> N+N->N+DELTA,OR N+N->N+N* REACTION          *
*                      3-> N+DELTA->N+N OR N+N*->N+N REACTION          *
*                      4-> N+N->N+N+PION,DIRTCT PROCESS                *
*                     5-> DELTA(N*)+DELTA(N*)   TOTAL   COLLISIONS    *
*           N12       - IS USED TO SPECIFY BARYON-BARYON REACTION      *
*                      CHANNELS. M12 IS THE REVERSAL CHANNEL OF N12    *
*                      N12,                                            *
*                      M12=1 FOR p+n-->delta(+)+ n                     *
*                          2     p+n-->delta(0)+ p                     *
*                          3     p+p-->delta(++)+n                     *
*                          4     p+p-->delta(+)+p                      *
*                          5     n+n-->delta(0)+n                      *
*                          6     n+n-->delta(-)+p                      *
*                          7     n+p-->N*(0)(1440)+p                   *
*                          8     n+p-->N*(+)(1440)+n                   *
*                        9     p+p-->N*(+)(1535)+p                     *
*                        10    n+n-->N*(0)(1535)+n                     *
*                         11    n+p-->N*(+)(1535)+n                     *
*                        12    n+p-->N*(0)(1535)+p
*                        13    D(++)+D(-)-->N*(+)(1440)+n
*                         14    D(++)+D(-)-->N*(0)(1440)+p
*                        15    D(+)+D(0)--->N*(+)(1440)+n
*                        16    D(+)+D(0)--->N*(0)(1440)+p
*                        17    D(++)+D(0)-->N*(+)(1535)+p
*                        18    D(++)+D(-)-->N*(0)(1535)+p
*                        19    D(++)+D(-)-->N*(+)(1535)+n
*                        20    D(+)+D(+)-->N*(+)(1535)+p
*                        21    D(+)+D(0)-->N*(+)(1535)+n
*                        22    D(+)+D(0)-->N*(0)(1535)+p
*                        23    D(+)+D(-)-->N*(0)(1535)+n
*                        24    D(0)+D(0)-->N*(0)(1535)+n
*                          25    N*(+)(14)+N*(+)(14)-->N*(+)(15)+p
*                          26    N*(0)(14)+N*(0)(14)-->N*(0)(15)+n
*                          27    N*(+)(14)+N*(0)(14)-->N*(+)(15)+n
*                        28    N*(+)(14)+N*(0)(14)-->N*(0)(15)+p
*                        29    N*(+)(14)+D+-->N*(+)(15)+p
*                        30    N*(+)(14)+D0-->N*(+)(15)+n
*                        31    N*(+)(14)+D(-)-->N*(0)(1535)+n
*                        32    N*(0)(14)+D++--->N*(+)(15)+p
*                        33    N*(0)(14)+D+--->N*(+)(15)+n
*                        34    N*(0)(14)+D+--->N*(0)(15)+p
*                        35    N*(0)(14)+D0-->N*(0)(15)+n
*                        36    N*(+)(14)+D0--->N*(0)(15)+p
*                        +++
*               AND MORE CHANNELS AS LISTED IN THE NOTE BOOK      
*
* NOTE ABOUT N*(1440) RESORANCE:                                       *
*     As it has been discussed in VerWest's paper,I= 1 (initial isospin)
*     channel can all be attributed to delta resorance while I= 0      *
*     channel can all be  attribured to N* resorance.Only in n+p       *
*     one can have I=0 channel so is the N*(1440) resorance            *
* REFERENCES:    J. CUGNON ET AL., NUCL. PHYS. A352, 505 (1981)        *
*                    Y. KITAZOE ET AL., PHYS. LETT. 166B, 35 (1986)    *
*                    B. VerWest el al., PHYS. PRV. C25 (1982)1979      *
*                    Gy. Wolf  et al, Nucl Phys A517 (1990) 615        *
*                    CUTOFF = 2 * AVMASS + 20 MEV                      *
*                                                                      *
*       for N*(1535) we use the parameterization by Gy. Wolf et al     *
*       Nucl phys A552 (1993) 349, added May 18, 1994                  *
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AKA=0.498,APHI=1.020,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        parameter (xmd=1.8756,npdmax=10000)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
        common /gg/ dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
        COMMON /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
        COMMON /NN/NNN
cc      SAVE /NN/
        COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
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
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1 px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      common /dpi/em2,lb2
      common /para8/ idpert,npertd,idxsec
      dimension ppd(3,npdmax),lbpd(npdmax)
      SAVE   
*-----------------------------------------------------------------------
       n12=0
       m12=0
        IBLOCK=0
        NTAG=0
        EM1=E(I1)
        EM2=E(I2)
      PR  = SQRT( PX**2 + PY**2 + PZ**2 )
      C2  = PZ / PR
      IF(PX .EQ. 0.0 .AND. PY .EQ. 0.0) THEN
        T2 = 0.0
      ELSE
        T2=ATAN2(PY,PX)
      END IF
      X1  = RANART(NSEED)
      ianti=0
      if(lb(i1).lt.0 .and. lb(i2).lt.0)ianti=1

clin-6/2008 Production of perturbative deuterons for idpert=1:
      call sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
      if(idpert.eq.1.and.ipert1.eq.1) then
         IF (SRT .LT. 2.012) RETURN
         if((iabs(lb(i1)).ge.6.and.iabs(lb(i1)).le.13)
     1        .and.(iabs(lb(i2)).ge.6.and.iabs(lb(i2)).le.13)) then
            goto 108
         else
            return
         endif
      endif
      
*-----------------------------------------------------------------------
*COM: TEST FOR ELASTIC SCATTERING (EITHER N-N OR DELTA-DELTA 0R
*      N-DELTA OR N*-N* or N*-Delta)
      IF (X1 .LE. SIGNN/SIG) THEN
*COM:  PARAMETRISATION IS TAKEN FROM THE CUGNON-PAPER
        AS  = ( 3.65 * (SRT - 1.8766) )**6
        A   = 6.0 * AS / (1.0 + AS)
        TA  = -2.0 * PR**2
        X   = RANART(NSEED)
clin-10/24/02        T1  = DLOG( (1-X) * DEXP(dble(A)*dble(TA)) + X )  /  A
        T1  = sngl(DLOG(dble(1.-X)*DEXP(dble(A)*dble(TA))+dble(X)))/  A
        C1  = 1.0 - T1/TA
        T1  = 2.0 * PI * RANART(NSEED)
        IBLOCK=20
       GO TO 107
      ELSE
*COM: TEST FOR INELASTIC SCATTERING
*     IF THE AVAILABLE ENERGY IS LESS THAN THE PION-MASS, NOTHING
*     CAN HAPPEN ANY MORE ==> RETURN (2.15 = 2*AVMASS +2*PI-MASS)
        IF (SRT .LT. 2.15) RETURN
*     IF THERE WERE 2 N*(1535) AND THEY DIDN'T SCATT. ELAST., 
*     ALLOW THEM TO PRODUCE KAONS. NO OTHER INELASTIC CHANNELS
*     ARE KNOWN
C       if((lb(i1).ge.12).and.(lb(i2).ge.12))return
*     ALL the inelastic collisions between N*(1535) and Delta as well
*     as N*(1440) TO PRODUCE KAONS, NO OTHER CHANNELS ARE KNOWN
C       if((lb(i1).ge.12).and.(lb(i2).ge.3))return
C       if((lb(i2).ge.12).and.(lb(i1).ge.3))return
*     calculate the N*(1535) production cross section in I1+I2 collisions
       call N1535(iabs(lb(i1)),iabs(lb(i2)),srt,X1535)

* for Delta+Delta-->N*(1440 OR 1535)+N AND N*(1440)+N*(1440)-->N*(1535)+X 
*     AND DELTA+N*(1440)-->N*(1535)+X
* WE ASSUME THEY HAVE THE SAME CROSS SECTIONS as CORRESPONDING N+N COLLISION):
* FOR D++D0, D+D+,D+D-,D0D0,N*+N*+,N*0N*0,N*(+)D+,N*(+)D(-),N*(0)D(0)
* N*(1535) production, kaon production and reabsorption through 
* D(N*)+D(N*)-->NN are ALLOWED.
* CROSS SECTION FOR KAON PRODUCTION from the four channels are
* for NLK channel
       akp=0.498
       ak0=0.498
       ana=0.938
       ada=1.232
       al=1.1157
       as=1.1197
       xsk1=0
       xsk2=0
       xsk3=0
       xsk4=0
       xsk5=0
       t1nlk=ana+al+akp
       if(srt.le.t1nlk)go to 222
       XSK1=1.5*PPLPK(SRT)
* for DLK channel
       t1dlk=ada+al+akp
       t2dlk=ada+al-akp
       if(srt.le.t1dlk)go to 222
       es=srt
       pmdlk2=(es**2-t1dlk**2)*(es**2-t2dlk**2)/(4.*es**2)
       pmdlk=sqrt(pmdlk2)
       XSK3=1.5*PPLPK(srt)
* for NSK channel
       t1nsk=ana+as+akp
       t2nsk=ana+as-akp
       if(srt.le.t1nsk)go to 222
       pmnsk2=(es**2-t1nsk**2)*(es**2-t2nsk**2)/(4.*es**2)
       pmnsk=sqrt(pmnsk2)
       XSK2=1.5*(PPK1(srt)+PPK0(srt))
* for DSK channel
       t1DSk=aDa+aS+akp
       t2DSk=aDa+aS-akp
       if(srt.le.t1dsk)go to 222
       pmDSk2=(es**2-t1DSk**2)*(es**2-t2DSk**2)/(4.*es**2)
       pmDSk=sqrt(pmDSk2)
       XSK4=1.5*(PPK1(srt)+PPK0(srt))
csp11/21/01
c phi production
       if(srt.le.(2.*amn+aphi))go to 222
c  !! mb put the correct form
         xsk5 = 0.0001
csp11/21/01 end
* THE TOTAL KAON+ PRODUCTION CROSS SECTION IS THEN
222       SIGK=XSK1+XSK2+XSK3+XSK4

cbz3/7/99 neutralk
        XSK1 = 2.0 * XSK1
        XSK2 = 2.0 * XSK2
        XSK3 = 2.0 * XSK3
        XSK4 = 2.0 * XSK4
        SIGK = 2.0 * SIGK + xsk5
cbz3/7/99 neutralk end

* The reabsorption cross section for the process
* D(N*)D(N*)-->NN is
       s2d=reab2d(i1,i2,srt)

cbz3/16/99 pion
        S2D = 0.
cbz3/16/99 pion end

*(1) N*(1535)+D(N*(1440)) reactions
*    we allow kaon production and reabsorption only
       if(((iabs(lb(i1)).ge.12).and.(iabs(lb(i2)).ge.12)).OR.
     &       ((iabs(lb(i1)).ge.12).and.(iabs(lb(i2)).ge.6)).OR.
     &       ((iabs(lb(i2)).ge.12).and.(iabs(lb(i1)).ge.6)))THEN
       signd=sigk+s2d
clin-6/2008
       IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c       if(x1.gt.(signd+signn)/sig)return
       if(x1.gt.(signd+signn+sdprod)/sig)return
c
* if kaon production
clin-6/2008
c       IF(SIGK/SIG.GE.RANART(NSEED))GO TO 306
       IF((SIGK+sdprod)/SIG.GE.RANART(NSEED))GO TO 306
c
* if reabsorption
       go to 1012
       ENDIF
       IDD=iabs(LB(I1)*LB(I2))
* channels have the same charge as pp 
        IF((IDD.EQ.63).OR.(IDD.EQ.64).OR.(IDD.EQ.48).
     1  OR.(IDD.EQ.49).OR.(IDD.EQ.11*11).OR.(IDD.EQ.10*10).
     2  OR.(IDD.EQ.88).OR.(IDD.EQ.66).
     3  OR.(IDD.EQ.90).OR.(IDD.EQ.70))THEN
        SIGND=X1535+SIGK+s2d
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF (X1.GT.(SIGNN+SIGND)/SIG)RETURN
        IF (X1.GT.(SIGNN+SIGND+sdprod)/SIG)RETURN
c
* if kaon production
       IF(SIGK/SIGND.GT.RANART(NSEED))GO TO 306
* if reabsorption
       if(s2d/(x1535+s2d).gt.RANART(NSEED))go to 1012
* if N*(1535) production
       IF(IDD.EQ.63)N12=17
       IF(IDD.EQ.64)N12=20
       IF(IDD.EQ.48)N12=23
       IF(IDD.EQ.49)N12=24
       IF(IDD.EQ.121)N12=25
       IF(IDD.EQ.100)N12=26
       IF(IDD.EQ.88)N12=29
       IF(IDD.EQ.66)N12=31
       IF(IDD.EQ.90)N12=32
       IF(IDD.EQ.70)N12=35
       GO TO 1011
        ENDIF
* IN DELTA+N*(1440) and N*(1440)+N*(1440) COLLISIONS, 
* N*(1535), kaon production and reabsorption are ALLOWED
* IN N*(1440)+N*(1440) COLLISIONS, ONLY N*(1535) IS ALLOWED
       IF((IDD.EQ.110).OR.(IDD.EQ.77).OR.(IDD.EQ.80))THEN
clin-6/2008
          IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c       IF(X1.GT.(SIGNN+X1535+SIGK+s2d)/SIG)RETURN
          IF(X1.GT.(SIGNN+X1535+SIGK+s2d+sdprod)/SIG)RETURN
c
       IF(SIGK/(X1535+SIGK+s2d).GT.RANART(NSEED))GO TO 306
       if(s2d/(x1535+s2d).gt.RANART(NSEED))go to 1012
       IF(IDD.EQ.77)N12=30
       IF((IDD.EQ.77).AND.(RANART(NSEED).LE.0.5))N12=36
       IF(IDD.EQ.80)N12=34
       IF((IDD.EQ.80).AND.(RANART(NSEED).LE.0.5))N12=35
       IF(IDD.EQ.110)N12=27
       IF((IDD.EQ.110).AND.(RANART(NSEED).LE.0.5))N12=28
       GO TO 1011
        ENDIF
       IF((IDD.EQ.54).OR.(IDD.EQ.56))THEN
* LIKE FOR N+P COLLISION, 
* IN DELTA+DELTA COLLISIONS BOTH N*(1440) AND N*(1535) CAN BE PRODUCED
        SIG2=(3./4.)*SIGMA(SRT,2,0,1)
        SIGND=2.*(SIG2+X1535)+SIGK+s2d
clin-6/2008
        IF(X1.LE.((SIGNN+sdprod)/SIG)) GO TO 108
c        IF(X1.GT.(SIGNN+SIGND)/SIG)RETURN
        IF(X1.GT.(SIGNN+SIGND+sdprod)/SIG)RETURN
c
       IF(SIGK/SIGND.GT.RANART(NSEED))GO TO 306
       if(s2d/(2.*(sig2+x1535)+s2d).gt.RANART(NSEED))go to 1012
       IF(RANART(NSEED).LT.X1535/(SIG2+X1535))THEN
* N*(1535) PRODUCTION
       IF(IDD.EQ.54)N12=18
       IF((IDD.EQ.54).AND.(RANART(NSEED).LE.0.5))N12=19
       IF(IDD.EQ.56)N12=21
       IF((IDD.EQ.56).AND.(RANART(NSEED).LE.0.5))N12=22
               ELSE 
* N*(144) PRODUCTION
       IF(IDD.EQ.54)N12=13
       IF((IDD.EQ.54).AND.(RANART(NSEED).LE.0.5))N12=14
       IF(IDD.EQ.56)N12=15
       IF((IDD.EQ.56).AND.(RANART(NSEED).LE.0.5))N12=16
              ENDIF
       ENDIF
1011       CONTINUE
       iblock=5
*PARAMETRIZATION OF THE SHAPE OF THE N*(1440) AND N*(1535) 
* RESONANCE ACCORDING
*     TO kitazoe's or J.D.JACKSON'S MASS FORMULA AND BREIT WIGNER
*     FORMULA FOR N* RESORANCE
*     DETERMINE DELTA MASS VIA REJECTION METHOD.
          DMAX = SRT - AVMASS-0.005
          DMIN = 1.078
          IF((n12.ge.13).and.(n12.le.16))then
* N*(1440) production
          IF(DMAX.LT.1.44) THEN
          FM=FNS(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FNS():
             xdmass=1.44
c          FM=FNS(1.44,SRT,1.)
          FM=FNS(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY2=0
11        DM=RANART(NSEED)*(DMAX-DMIN)+DMIN
          NTRY2=NTRY2+1
          IF((RANART(NSEED).GT.FNS(DM,SRT,1.)/FM).AND.
     1    (NTRY2.LE.10)) GO TO 11

clin-2/26/03 limit the N* mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.2.14) goto 11

              GO TO 13
              ENDIF
                    IF((n12.ge.17).AND.(N12.LE.36))then
* N*(1535) production
          IF(DMAX.LT.1.535) THEN
          FM=FD5(DMAX,SRT,0.)
          ELSE

clin-10/25/02 get rid of argument usage mismatch in FNS():
             xdmass=1.535
c          FM=FD5(1.535,SRT,1.)
          FM=FD5(xdmass,SRT,1.)
clin-10/25/02-end

          ENDIF
          IF(FM.EQ.0.)FM=1.E-09
          NTRY1=0
12        DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
          NTRY1=NTRY1+1
          IF((RANART(NSEED) .GT. FD5(DM,SRT,1.)/FM).AND.
     1    (NTRY1.LE.10)) GOTO 12

clin-2/26/03 limit the N* mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.1.84) goto 12

             ENDIF
13       CONTINUE
*-------------------------------------------------------
* RELABLE BARYON I1 AND I2
*13 D(++)+D(-)--> N*(+)(14)+n
          IF(N12.EQ.13)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=11
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=11
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*14 D(++)+D(-)--> N*(0)(14)+P
          IF(N12.EQ.14)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=10
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=10
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*15 D(+)+D(0)--> N*(+)(14)+n
          IF(N12.EQ.15)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=11
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=11
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*16 D(+)+D(0)--> N*(0)(14)+P
          IF(N12.EQ.16)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=10
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=10
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*17 D(++)+D(0)--> N*(+)(14)+P
          IF(N12.EQ.17)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
         go to 200
          ENDIF
*18 D(++)+D(-)--> N*(0)(15)+P
          IF(N12.EQ.18)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*19 D(++)+D(-)--> N*(+)(15)+N
          IF(N12.EQ.19)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*20 D(+)+D(+)--> N*(+)(15)+P
          IF(N12.EQ.20)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*21 D(+)+D(0)--> N*(+)(15)+N
          IF(N12.EQ.21)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*22 D(+)+D(0)--> N*(0)(15)+P
          IF(N12.EQ.22)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*23 D(+)+D(-)--> N*(0)(15)+N
          IF(N12.EQ.23)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*24 D(0)+D(0)--> N*(0)(15)+N
          IF(N12.EQ.24)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
         go to 200
          ENDIF
*25 N*(+)+N*(+)--> N*(0)(15)+P
          IF(N12.EQ.25)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
         go to 200
          ENDIF
*26 N*(0)+N*(0)--> N*(0)(15)+N
          IF(N12.EQ.26)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
         go to 200
          ENDIF
*27 N*(+)+N*(0)--> N*(+)(15)+N
          IF(N12.EQ.27)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*28 N*(+)+N*(0)--> N*(0)(15)+P
          IF(N12.EQ.28)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*27 N*(+)+N*(0)--> N*(+)(15)+N
          IF(N12.EQ.27)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*29 N*(+)+D(+)--> N*(+)(15)+P
          IF(N12.EQ.29)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*30 N*(+)+D(0)--> N*(+)(15)+N
          IF(N12.EQ.30)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*31 N*(+)+D(-)--> N*(0)(15)+N
          IF(N12.EQ.31)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*32 N*(0)+D(++)--> N*(+)(15)+P
          IF(N12.EQ.32)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*33 N*(0)+D(+)--> N*(+)(15)+N
          IF(N12.EQ.33)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=13
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=13
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*34 N*(0)+D(+)--> N*(0)(15)+P
          IF(N12.EQ.34)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
*35 N*(0)+D(0)--> N*(0)(15)+N
          IF(N12.EQ.35)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*36 N*(+)+D(0)--> N*(0)(15)+P
          IF(N12.EQ.36)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=12
          E(I2)=DM
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=12
          E(I1)=DM
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
1012         continue
         iblock=55
         lb1=lb(i1)
         lb2=lb(i2)
         ich=iabs(lb1*lb2)
*-------------------------------------------------------
* RELABLE BARYON I1 AND I2 in the reabsorption processes
*37 D(++)+D(-)--> n+p
          IF(ich.EQ.9*6)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=1
          E(I1)=amp
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*38 D(+)+D(0)--> n+p
          IF(ich.EQ.8*7)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=1
          E(I1)=amp
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*39 D(++)+D(0)--> p+p
          IF(ich.EQ.9*7)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=1
         E(I1)=AMP
         go to 200
          ENDIF
*40 D(+)+D(+)--> p+p
          IF(ich.EQ.8*8)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=1
         E(I1)=AMP
          go to 200
          ENDIF
*41 D(+)+D(-)--> n+n
          IF(ich.EQ.8*6)THEN
          LB(I2)=2
          E(I2)=amn
         LB(I1)=2
         E(I1)=AMN
          go to 200
          ENDIF
*42 D(0)+D(0)--> n+n
          IF(ich.EQ.6*6)THEN
          LB(I2)=2
          E(I2)=amn
         LB(I1)=2
         E(I1)=AMN
         go to 200
          ENDIF
*43 N*(+)+N*(+)--> p+p
          IF(ich.EQ.11*11.or.ich.eq.13*13.or.ich.eq.11*13)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=1
         E(I1)=AMP
         go to 200
          ENDIF
*44 N*(0)(1440)+N*(0)--> n+n
          IF(ich.EQ.10*10.or.ich.eq.12*12.or.ich.eq.10*12)THEN
          LB(I2)=2
          E(I2)=amn
         LB(I1)=2
         E(I1)=AMN
         go to 200
          ENDIF
*45 N*(+)+N*(0)--> n+p
          IF(ich.EQ.10*11.or.ich.eq.12*13.or.ich.
     &    eq.10*13.or.ich.eq.11*12)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=1
          E(I1)=amp
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*46 N*(+)+D(+)--> p+p
          IF(ich.eq.11*8.or.ich.eq.13*8)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=1
         E(I1)=AMP
          go to 200
          ENDIF
*47 N*(+)+D(0)--> n+p
          IF(ich.EQ.11*7.or.ich.eq.13*7)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=2
         E(I1)=AMN
          ELSE
          LB(I1)=1
          E(I1)=amp
         LB(I2)=2
         E(I2)=AMN
          ENDIF
         go to 200
          ENDIF
*48 N*(+)+D(-)--> n+n
          IF(ich.EQ.11*6.or.ich.eq.13*6)THEN
          LB(I2)=2
          E(I2)=amn
         LB(I1)=2
         E(I1)=AMN
          go to 200
          ENDIF
*49 N*(0)+D(++)--> p+p
          IF(ich.EQ.10*9.or.ich.eq.12*9)THEN
          LB(I2)=1
          E(I2)=amp
         LB(I1)=1
         E(I1)=AMP
         go to 200
          ENDIF
*50 N*(0)+D(0)--> n+n
          IF(ich.EQ.10*7.or.ich.eq.12*7)THEN
          LB(I2)=2
          E(I2)=amn
         LB(I1)=2
         E(I1)=AMN
          go to 200
          ENDIF
*51 N*(0)+D(+)--> n+p
          IF(ich.EQ.10*8.or.ich.eq.12*8)THEN
          IF(RANART(NSEED).LE.0.5)THEN
          LB(I2)=2
          E(I2)=amn
         LB(I1)=1
         E(I1)=AMP
          ELSE
          LB(I1)=2
          E(I1)=amn
         LB(I2)=1
         E(I2)=AMP
          ENDIF
         go to 200
          ENDIF
         lb(i1)=1
         e(i1)=amp
         lb(i2)=2
         e(i2)=amn
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
* resonance production or absorption in resonance+resonance collisions is
* assumed to have the same pt distribution as pp
200       EM1=E(I1)
          EM2=E(I2)
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
             if(srt.le.2.14)C1= 1.0 - 2.0 * RANART(NSEED)
         if(srt.gt.2.14.and.srt.le.2.4)c1=ang(srt,iseed)       
         if(srt.gt.2.4)then

clin-10/25/02 get rid of argument usage mismatch in PTR():
             xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
         cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
         endif
          T1   = 2.0 * PI * RANART(NSEED)
       if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
         lb(i1) = -lb(i1)
         lb(i2) = -lb(i2)
       endif
         ENDIF
*COM: SET THE NEW MOMENTUM COORDINATES
107   S1   = SQRT( 1.0 - C1**2 )
      S2  =  SQRT( 1.0 - C2**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      CT2  = COS(T2)
      ST2  = SIN(T2)
      PZ   = PR * ( C1*C2 - S1*S2*CT1 )
      SS   = C2 * S1 * CT1  +  S2 * C1
      PX   = PR * ( SS*CT2 - S1*ST1*ST2 )
      PY   = PR * ( SS*ST2 + S1*ST1*CT2 )
      RETURN
* FOR THE DD-->KAON+X PROCESS, FIND MOMENTUM OF THE FINAL PARTICLES IN 
* THE NUCLEUS-NUCLEUS CMS.
306     CONTINUE
csp11/21/01 phi production
              if(XSK5/sigK.gt.RANART(NSEED))then
              pz1=p(3,i1)
              pz2=p(3,i2)
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 1 + int(2 * RANART(NSEED))
              nnn=nnn+1
                LPION(NNN,IRUN)=29
                EPION(NNN,IRUN)=APHI
                iblock = 222
              GO TO 208
               ENDIF
              iblock=10
                if(ianti .eq. 1)iblock=-10
              pz1=p(3,i1)
              pz2=p(3,i2)
* DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
              nnn=nnn+1
                LPION(NNN,IRUN)=23
                EPION(NNN,IRUN)=Aka
              if(srt.le.2.63)then
* only lambda production is possible
* (1.1)P+P-->p+L+kaon+
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              GO TO 208
                ENDIF
       if(srt.le.2.74.and.srt.gt.2.63)then
* both Lambda and sigma production are possible
              if(XSK1/(XSK1+XSK2).gt.RANART(NSEED))then
* lambda production
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              else
* sigma production
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              ic=2
              endif
              GO TO 208
       endif
       if(srt.le.2.77.and.srt.gt.2.74)then
* then pp-->Delta lamda kaon can happen
              if(xsk1/(xsk1+xsk2+xsk3).gt.RANART(NSEED))then
* * (1.1)P+P-->p+L+kaon+
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              go to 208
              else
              if(xsk2/(xsk2+xsk3).gt.RANART(NSEED))then
* pp-->psk
              ic=2
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              else
* pp-->D+l+k        
              ic=3
                LB(I1) = 6 + int(4 * RANART(NSEED))
              lb(i2)=14
              endif
              GO TO 208
              endif
       endif
       if(srt.gt.2.77)then
* all four channels are possible
              if(xsk1/(xsk1+xsk2+xsk3+xsk4).gt.RANART(NSEED))then
* p lambda k production
              ic=1
                LB(I1) = 1 + int(2 * RANART(NSEED))
              LB(I2)=14
              go to 208
       else
          if(xsk3/(xsk2+xsk3+xsk4).gt.RANART(NSEED))then
* delta l K production
              ic=3
                LB(I1) = 6 + int(4 * RANART(NSEED))
              lb(i2)=14
              go to 208
          else
              if(xsk2/(xsk2+xsk4).gt.RANART(NSEED))then
* n sigma k production
                LB(I1) = 1 + int(2 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              ic=2
              else
* D sigma K
              ic=4
                LB(I1) = 6 + int(4 * RANART(NSEED))
                LB(I2) = 15 + int(3 * RANART(NSEED))
              endif
              go to 208
          endif
       endif
       endif
208             continue
         if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
          lb(i1) = - lb(i1)
          lb(i2) = - lb(i2)
          if(LPION(NNN,IRUN) .eq. 23)LPION(NNN,IRUN)=21
         endif
       lbi1=lb(i1)
       lbi2=lb(i2)
* KEEP ALL COORDINATES OF PARTICLE 2 FOR POSSIBLE PHASE SPACE CHANGE
           NTRY1=0
129        CALL BBKAON(ic,SRT,PX3,PY3,PZ3,DM3,PX4,PY4,PZ4,DM4,
     &  PPX,PPY,PPZ,icou1)
       NTRY1=NTRY1+1
       if((icou1.lt.0).AND.(NTRY1.LE.20))GO TO 129
c       if(icou1.lt.0)return
* ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
       CALL ROTATE(PX,PY,PZ,PX3,PY3,PZ3)
       CALL ROTATE(PX,PY,PZ,PX4,PY4,PZ4)
       CALL ROTATE(PX,PY,PZ,PPX,PPY,PPZ)
* FIND THE MOMENTUM OF PARTICLES IN THE FINAL STATE IN THE NUCLEUS-
* NUCLEUS CMS. FRAME 
* (1) for the necleon/delta
*             LORENTZ-TRANSFORMATION INTO LAB FRAME FOR DELTA1
              E1CM    = SQRT (dm3**2 + PX3**2 + PY3**2 + PZ3**2)
              P1BETA  = PX3*BETAX + PY3*BETAY + PZ3*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Pt1i1 = BETAX * TRANSF + PX3
              Pt2i1 = BETAY * TRANSF + PY3
              Pt3i1 = BETAZ * TRANSF + PZ3
             Eti1   = DM3
* (2) for the lambda/sigma
                E2CM    = SQRT (dm4**2 + PX4**2 + PY4**2 + PZ4**2)
                P2BETA  = PX4*BETAX+PY4*BETAY+PZ4*BETAZ
                TRANSF  = GAMMA * (GAMMA*P2BETA / (GAMMA + 1.) + E2CM)
                Pt1I2 = BETAX * TRANSF + PX4
                Pt2I2 = BETAY * TRANSF + PY4
                Pt3I2 = BETAZ * TRANSF + PZ4
              EtI2   = DM4
* GET the kaon'S MOMENTUM AND COORDINATES IN NUCLEUS-NUCLEUS CMS. FRAME
                EPCM=SQRT(aka**2+PPX**2+PPY**2+PPZ**2)
                PPBETA=PPX*BETAX+PPY*BETAY+PPZ*BETAZ
                TRANSF=GAMMA*(GAMMA*PPBETA/(GAMMA+1.)+EPCM)
                PPION(1,NNN,IRUN)=BETAX*TRANSF+PPX
                PPION(2,NNN,IRUN)=BETAY*TRANSF+PPY
                PPION(3,NNN,IRUN)=BETAZ*TRANSF+PPZ
clin-5/2008:
                dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
clin-5/2008:
c2007        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2007
c                RPION(1,NNN,IRUN)=R(1,I1)+0.5*x01
c                RPION(2,NNN,IRUN)=R(2,I1)+0.5*y01
c                RPION(3,NNN,IRUN)=R(3,I1)+0.5*z01
                    RPION(1,NNN,IRUN)=R(1,I1)
                    RPION(2,NNN,IRUN)=R(2,I1)
                    RPION(3,NNN,IRUN)=R(3,I1)
c
* assign the nucleon/delta and lambda/sigma to i1 or i2 to keep the 
* leadng particle behaviour
C              if((pt1i1*px1+pt2i1*py1+pt3i1*pz1).gt.0)then
              p(1,i1)=pt1i1
              p(2,i1)=pt2i1
              p(3,i1)=pt3i1
              e(i1)=eti1
              lb(i1)=lbi1
              p(1,i2)=pt1i2
              p(2,i2)=pt2i2
              p(3,i2)=pt3i2
              e(i2)=eti2
              lb(i2)=lbi2
                PX1     = P(1,I1)
                PY1     = P(2,I1)
                PZ1     = P(3,I1)
              EM1       = E(I1)
                ID(I1)  = 2
                ID(I2)  = 2
                ID1     = ID(I1)
        LB1=LB(I1)
        LB2=LB(I2)
        AM1=EM1
       am2=em2
        E1= SQRT( EM1**2 + PX1**2 + PY1**2 + PZ1**2 )
       RETURN

clin-6/2008 D+D->Deuteron+pi:
*     FIND MOMENTUM OF THE FINAL PARTICLES IN THE NUCLEUS-NUCLEUS CMS.
 108   CONTINUE
           if(idpert.eq.1.and.ipert1.eq.1.and.npertd.ge.1) then
c     For idpert=1: we produce npertd pert deuterons:
              ndloop=npertd
           elseif(idpert.eq.2.and.npertd.ge.1) then
c     For idpert=2: we first save information for npertd pert deuterons;
c     at the last ndloop we create the regular deuteron+pi 
c     and those pert deuterons:
              ndloop=npertd+1
           else
c     Just create the regular deuteron+pi:
              ndloop=1
           endif
c
           dprob1=sdprod/sig/float(npertd)
           do idloop=1,ndloop
              CALL bbdangle(pxd,pyd,pzd,nt,ipert1,ianti,idloop,pfinal,
     1 dprob1,lbm)
              CALL ROTATE(PX,PY,PZ,PXd,PYd,PZd)
*     LORENTZ-TRANSFORMATION OF THE MOMENTUM OF PARTICLES IN THE FINAL STATE 
*     FROM THE NN CMS FRAME INTO THE GLOBAL CMS FRAME:
*     For the Deuteron:
              xmass=xmd
              E1dCM=SQRT(xmass**2+PXd**2+PYd**2+PZd**2)
              P1dBETA=PXd*BETAX+PYd*BETAY+PZd*BETAZ
              TRANSF=GAMMA*(GAMMA*P1dBETA/(GAMMA+1.)+E1dCM)
              pxi1=BETAX*TRANSF+PXd
              pyi1=BETAY*TRANSF+PYd
              pzi1=BETAZ*TRANSF+PZd
              if(ianti.eq.0)then
                 lbd=42
              else
                 lbd=-42
              endif
              if(idpert.eq.1.and.ipert1.eq.1.and.npertd.ge.1) then
cccc  Perturbative production for idpert=1:
                 nnn=nnn+1
                 PPION(1,NNN,IRUN)=pxi1
                 PPION(2,NNN,IRUN)=pyi1
                 PPION(3,NNN,IRUN)=pzi1
                 EPION(NNN,IRUN)=xmd
                 LPION(NNN,IRUN)=lbd
                 RPION(1,NNN,IRUN)=R(1,I1)
                 RPION(2,NNN,IRUN)=R(2,I1)
                 RPION(3,NNN,IRUN)=R(3,I1)
clin-6/2008 assign the perturbative probability:
                 dppion(NNN,IRUN)=sdprod/sig/float(npertd)
              elseif(idpert.eq.2.and.idloop.le.npertd) then
clin-6/2008 For idpert=2, we produce NPERTD perturbative (anti)deuterons 
c     only when a regular (anti)deuteron+pi is produced in NN collisions.
c     First save the info for the perturbative deuterons:
                 ppd(1,idloop)=pxi1
                 ppd(2,idloop)=pyi1
                 ppd(3,idloop)=pzi1
                 lbpd(idloop)=lbd
              else
cccc  Regular production:
c     For the regular pion: do LORENTZ-TRANSFORMATION:
                 E(i1)=xmm
                 E2piCM=SQRT(xmm**2+PXd**2+PYd**2+PZd**2)
                 P2piBETA=-PXd*BETAX-PYd*BETAY-PZd*BETAZ
                 TRANSF=GAMMA*(GAMMA*P2piBETA/(GAMMA+1.)+E2piCM)
                 pxi2=BETAX*TRANSF-PXd
                 pyi2=BETAY*TRANSF-PYd
                 pzi2=BETAZ*TRANSF-PZd
                 p(1,i1)=pxi2
                 p(2,i1)=pyi2
                 p(3,i1)=pzi2
c     Remove regular pion to check the equivalence 
c     between the perturbative and regular deuteron results:
c                 E(i1)=0.
c
                 LB(I1)=lbm
                 PX1=P(1,I1)
                 PY1=P(2,I1)
                 PZ1=P(3,I1)
                 EM1=E(I1)
                 ID(I1)=2
                 ID1=ID(I1)
                 E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
                 lb1=lb(i1)
c     For the regular deuteron:
                 p(1,i2)=pxi1
                 p(2,i2)=pyi1
                 p(3,i2)=pzi1
                 lb(i2)=lbd
                 lb2=lb(i2)
                 E(i2)=xmd
                 EtI2=E(I2)
                 ID(I2)=2
c     For idpert=2: create the perturbative deuterons:
                 if(idpert.eq.2.and.idloop.eq.ndloop) then
                    do ipertd=1,npertd
                       nnn=nnn+1
                       PPION(1,NNN,IRUN)=ppd(1,ipertd)
                       PPION(2,NNN,IRUN)=ppd(2,ipertd)
                       PPION(3,NNN,IRUN)=ppd(3,ipertd)
                       EPION(NNN,IRUN)=xmd
                       LPION(NNN,IRUN)=lbpd(ipertd)
                       RPION(1,NNN,IRUN)=R(1,I1)
                       RPION(2,NNN,IRUN)=R(2,I1)
                       RPION(3,NNN,IRUN)=R(3,I1)
clin-6/2008 assign the perturbative probability:
                       dppion(NNN,IRUN)=1./float(npertd)
                    enddo
                 endif
              endif
           enddo
           IBLOCK=501
           return
clin-6/2008 D+D->Deuteron+pi over

        END
**********************************
**********************************
*                                                                      *
      SUBROUTINE INIT(MINNUM,MAXNUM,NUM,RADIUS,X0,Z0,P0,
     &                GAMMA,ISEED,MASS,IOPT)
*                                                                      *
*       PURPOSE:     PROVIDING INITIAL CONDITIONS FOR PHASE-SPACE      *
*                    DISTRIBUTION OF TESTPARTICLES                     *
*       VARIABLES:   (ALL INPUT)                                       *
*         MINNUM  - FIRST TESTPARTICLE TREATED IN ONE RUN    (INTEGER) *
*         MAXNUM  - LAST TESTPARTICLE TREATED IN ONE RUN     (INTEGER) *
*         NUM     - NUMBER OF TESTPARTICLES PER NUCLEON      (INTEGER) *
*         RADIUS  - RADIUS OF NUCLEUS "FM"                      (REAL) *
*         X0,Z0   - DISPLACEMENT OF CENTER OF NUCLEUS IN X,Z-          *
*                   DIRECTION "FM"                              (REAL) *
*         P0      - MOMENTUM-BOOST IN C.M. FRAME "GEV/C"        (REAL) *
*         GAMMA   - RELATIVISTIC GAMMA-FACTOR                   (REAL) *
*         ISEED   - SEED FOR RANDOM-NUMBER GENERATOR         (INTEGER) *
*         MASS    - TOTAL MASS OF THE SYSTEM                 (INTEGER) *
*         IOPT    - OPTION FOR DIFFERENT OCCUPATION OF MOMENTUM        *
*                   SPACE                                    (INTEGER) *
*                                                                      *
**********************************
      PARAMETER     (MAXSTR=150001,  AMU   = 0.9383)
      PARAMETER     (MAXX   =   20,  MAXZ  =    24)
      PARAMETER     (PI=3.1415926)
*
      REAL              PTOT(3)
      COMMON  /AA/      R(3,MAXSTR)
cc      SAVE /AA/
      COMMON  /BB/      P(3,MAXSTR)
cc      SAVE /BB/
      COMMON  /CC/      E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common  /ss/      inout(20)
cc      SAVE /ss/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
*----------------------------------------------------------------------
*     PREPARATION FOR LORENTZ-TRANSFORMATIONS
*
      ISEED=ISEED
      IF (P0 .NE. 0.) THEN
        SIGN = P0 / ABS(P0)
      ELSE
        SIGN = 0.
      END IF
      BETA = SIGN * SQRT(GAMMA**2-1.)/GAMMA
*-----------------------------------------------------------------------
*     TARGET-ID = 1 AND PROJECTILE-ID = -1
*
      IF (MINNUM .EQ. 1) THEN
        IDNUM = 1
      ELSE
        IDNUM = -1
      END IF
*-----------------------------------------------------------------------
*     IDENTIFICATION OF TESTPARTICLES AND ASSIGMENT OF RESTMASS
*
*     LOOP OVER ALL PARALLEL RUNS:
      DO 400 IRUN = 1,NUM
        DO 100 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
          ID(I) = IDNUM
          E(I)  = AMU
  100   CONTINUE
*-----------------------------------------------------------------------
*       OCCUPATION OF COORDINATE-SPACE
*
        DO 300 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
  200     CONTINUE
            X = 1.0 - 2.0 * RANART(NSEED)
            Y = 1.0 - 2.0 * RANART(NSEED)
            Z = 1.0 - 2.0 * RANART(NSEED)
          IF ((X*X+Y*Y+Z*Z) .GT. 1.0) GOTO 200
          R(1,I) = X * RADIUS
          R(2,I) = Y * RADIUS
          R(3,I) = Z * RADIUS
  300   CONTINUE
  400 CONTINUE
*=======================================================================
      IF (IOPT .NE. 3) THEN
*-----
*     OPTION 1: USE WOODS-SAXON PARAMETRIZATION FOR DENSITY AND
*-----          CALCULATE LOCAL FERMI-MOMENTUM
*
        RHOW0  = 0.168
        DO 1000 IRUN = 1,NUM
          DO 600 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
  500       CONTINUE
              PX = 1.0 - 2.0 * RANART(NSEED)
              PY = 1.0 - 2.0 * RANART(NSEED)
              PZ = 1.0 - 2.0 * RANART(NSEED)
            IF (PX*PX+PY*PY+PZ*PZ .GT. 1.0) GOTO 500
            RDIST  = SQRT( R(1,I)**2 + R(2,I)**2 + R(3,I)**2 )
            RHOWS  = RHOW0 / (  1.0 + EXP( (RDIST-RADIUS) / 0.55 )  )
            PFERMI = 0.197 * (1.5 * PI*PI * RHOWS)**(1./3.)
*-----
*     OPTION 2: NUCLEAR MATTER CASE
            IF(IOPT.EQ.2) PFERMI=0.27
           if(iopt.eq.4) pfermi=0.
*-----
            P(1,I) = PFERMI * PX
            P(2,I) = PFERMI * PY
            P(3,I) = PFERMI * PZ
  600     CONTINUE
*
*         SET TOTAL MOMENTUM TO 0 IN REST FRAME AND BOOST
*
          DO 700 IDIR = 1,3
            PTOT(IDIR) = 0.0
  700     CONTINUE
          NPART = 0
          DO 900 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
            NPART = NPART + 1
            DO 800 IDIR = 1,3
              PTOT(IDIR) = PTOT(IDIR) + P(IDIR,I)
  800       CONTINUE
  900     CONTINUE
          DO 950 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
            DO 925 IDIR = 1,3
              P(IDIR,I) = P(IDIR,I) - PTOT(IDIR) / FLOAT(NPART)
  925       CONTINUE
*           BOOST
            IF ((IOPT .EQ. 1).or.(iopt.eq.2)) THEN
              EPART = SQRT(P(1,I)**2+P(2,I)**2+P(3,I)**2+AMU**2)
              P(3,I) = GAMMA*(P(3,I) + BETA*EPART)
            ELSE
              P(3,I) = P(3,I) + P0
            END IF
  950     CONTINUE
 1000   CONTINUE
*-----
      ELSE
*-----
*     OPTION 3: GIVE ALL NUCLEONS JUST A Z-MOMENTUM ACCORDING TO
*               THE BOOST OF THE NUCLEI
*
        DO 1200 IRUN = 1,NUM
          DO 1100 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
            P(1,I) = 0.0
            P(2,I) = 0.0
            P(3,I) = P0
 1100     CONTINUE
 1200   CONTINUE
*-----
      END IF
*=======================================================================
*     PUT PARTICLES IN THEIR POSITION IN COORDINATE-SPACE
*     (SHIFT AND RELATIVISTIC CONTRACTION)
*
      DO 1400 IRUN = 1,NUM
        DO 1300 I = MINNUM+(IRUN-1)*MASS,MAXNUM+(IRUN-1)*MASS
          R(1,I) = R(1,I) + X0
* two nuclei in touch after contraction
          R(3,I) = (R(3,I)+Z0)/ GAMMA 
* two nuclei in touch before contraction
c          R(3,I) = R(3,I) / GAMMA + Z0
 1300   CONTINUE
 1400 CONTINUE
*
      RETURN
      END
**********************************
*                                                                      *
      SUBROUTINE DENS(IPOT,MASS,NUM,NESC)
*                                                                      *
*       PURPOSE:     CALCULATION OF LOCAL BARYON, MESON AND ENERGY     * 
*                    DENSITY FROM SPATIAL DISTRIBUTION OF TESTPARTICLES*
*                                                                      *
*       VARIABLES (ALL INPUT, ALL INTEGER)                             *
*         MASS    -  MASS NUMBER OF THE SYSTEM                         *
*         NUM     -  NUMBER OF TESTPARTICLES PER NUCLEON               *
*                                                                      *
*         NESC    -  NUMBER OF ESCAPED PARTICLES      (INTEGER,OUTPUT) *
*                                                                      *
**********************************
      PARAMETER     (MAXSTR= 150001,MAXR=1)
      PARAMETER     (MAXX   =    20,  MAXZ  =    24)
*
      dimension pxl(-maxx:maxx,-maxx:maxx,-maxz:maxz),
     1          pyl(-maxx:maxx,-maxx:maxx,-maxz:maxz),
     2          pzl(-maxx:maxx,-maxx:maxx,-maxz:maxz)
      COMMON  /AA/      R(3,MAXSTR)
cc      SAVE /AA/
      COMMON  /BB/      P(3,MAXSTR)
cc      SAVE /BB/
      COMMON  /CC/      E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      COMMON  /DDpi/    piRHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DDpi/
      COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common  /ss/  inout(20)
cc      SAVE /ss/
      COMMON  /RR/  MASSR(0:MAXR)
cc      SAVE /RR/
      common  /tt/  PEL(-maxx:maxx,-maxx:maxx,-maxz:maxz)
     &,rxy(-maxx:maxx,-maxx:maxx,-maxz:maxz)
cc      SAVE /tt/
      common  /bbb/ bxx(-maxx:maxx,-maxx:maxx,-maxz:maxz),
     &byy(-maxx:maxx,-maxx:maxx,-maxz:maxz),
     &bzz(-maxx:maxx,-maxx:maxx,-maxz:maxz)
*
      real zet(-45:45)
      SAVE   
      data zet /
     4     1.,0.,0.,0.,0.,
     3     1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
     2     -1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
     1     0.,0.,0.,-1.,0.,1.,0.,-1.,0.,-1.,
     s     0.,-2.,-1.,0.,1.,0.,0.,0.,0.,-1.,
     e     0.,
     s     1.,0.,-1.,0.,1.,-1.,0.,1.,2.,0.,
     1     1.,0.,1.,0.,-1.,0.,1.,0.,0.,0.,
     2     -1.,0.,1.,0.,-1.,0.,1.,0.,0.,1.,
     3     0.,0.,0.,0.,0.,0.,0.,0.,0.,-1.,
     4     0.,0.,0.,0.,-1./

      DO 300 IZ = -MAXZ,MAXZ
        DO 200 IY = -MAXX,MAXX
          DO 100 IX = -MAXX,MAXX
            RHO(IX,IY,IZ) = 0.0
            RHOn(IX,IY,IZ) = 0.0
            RHOp(IX,IY,IZ) = 0.0
            piRHO(IX,IY,IZ) = 0.0
           pxl(ix,iy,iz) = 0.0
           pyl(ix,iy,iz) = 0.0
           pzl(ix,iy,iz) = 0.0
           pel(ix,iy,iz) = 0.0
           bxx(ix,iy,iz) = 0.0
           byy(ix,iy,iz) = 0.0
           bzz(ix,iy,iz) = 0.0
  100     CONTINUE
  200   CONTINUE
  300 CONTINUE
*
      NESC  = 0
      BIG   = 1.0 / ( 3.0 * FLOAT(NUM) )
      SMALL = 1.0 / ( 9.0 * FLOAT(NUM) )
*
      MSUM=0
      DO 400 IRUN = 1,NUM
      MSUM=MSUM+MASSR(IRUN-1)
      DO 400 J=1,MASSr(irun)
      I=J+MSUM
        IX = NINT( R(1,I) )
        IY = NINT( R(2,I) )
        IZ = NINT( R(3,I) )
        IF( IX .LE. -MAXX .OR. IX .GE. MAXX .OR.
     &      IY .LE. -MAXX .OR. IY .GE. MAXX .OR.
     &      IZ .LE. -MAXZ .OR. IZ .GE. MAXZ )    THEN
          NESC = NESC + 1
        ELSE
c
csp01/04/02 include baryon density
          if(j.gt.mass)go to 30
c         if( (lb(i).eq.1.or.lb(i).eq.2) .or.
c    &    (lb(i).ge.6.and.lb(i).le.17) )then                       
* (1) baryon density
          RHO(IX,  IY,  IZ  ) = RHO(IX,  IY,  IZ  ) + BIG
          RHO(IX+1,IY,  IZ  ) = RHO(IX+1,IY,  IZ  ) + SMALL
          RHO(IX-1,IY,  IZ  ) = RHO(IX-1,IY,  IZ  ) + SMALL
          RHO(IX,  IY+1,IZ  ) = RHO(IX,  IY+1,IZ  ) + SMALL
          RHO(IX,  IY-1,IZ  ) = RHO(IX,  IY-1,IZ  ) + SMALL
          RHO(IX,  IY,  IZ+1) = RHO(IX,  IY,  IZ+1) + SMALL
          RHO(IX,  IY,  IZ-1) = RHO(IX,  IY,  IZ-1) + SMALL
* (2) CALCULATE THE PROTON DENSITY
         IF(ZET(LB(I)).NE.0)THEN
          RHOP(IX,  IY,  IZ  ) = RHOP(IX,  IY,  IZ  ) + BIG
          RHOP(IX+1,IY,  IZ  ) = RHOP(IX+1,IY,  IZ  ) + SMALL
          RHOP(IX-1,IY,  IZ  ) = RHOP(IX-1,IY,  IZ  ) + SMALL
          RHOP(IX,  IY+1,IZ  ) = RHOP(IX,  IY+1,IZ  ) + SMALL
          RHOP(IX,  IY-1,IZ  ) = RHOP(IX,  IY-1,IZ  ) + SMALL
          RHOP(IX,  IY,  IZ+1) = RHOP(IX,  IY,  IZ+1) + SMALL
          RHOP(IX,  IY,  IZ-1) = RHOP(IX,  IY,  IZ-1) + SMALL
         go to 40
         ENDIF
* (3) CALCULATE THE NEUTRON DENSITY
         IF(ZET(LB(I)).EQ.0)THEN
          RHON(IX,  IY,  IZ  ) = RHON(IX,  IY,  IZ  ) + BIG
          RHON(IX+1,IY,  IZ  ) = RHON(IX+1,IY,  IZ  ) + SMALL
          RHON(IX-1,IY,  IZ  ) = RHON(IX-1,IY,  IZ  ) + SMALL
          RHON(IX,  IY+1,IZ  ) = RHON(IX,  IY+1,IZ  ) + SMALL
          RHON(IX,  IY-1,IZ  ) = RHON(IX,  IY-1,IZ  ) + SMALL
          RHON(IX,  IY,  IZ+1) = RHON(IX,  IY,  IZ+1) + SMALL
          RHON(IX,  IY,  IZ-1) = RHON(IX,  IY,  IZ-1) + SMALL
         go to 40
          END IF
c           else    !! sp01/04/02
* (4) meson density       
30              piRHO(IX,  IY,  IZ  ) = piRHO(IX,  IY,  IZ  ) + BIG
          piRHO(IX+1,IY,  IZ  ) = piRHO(IX+1,IY,  IZ  ) + SMALL
          piRHO(IX-1,IY,  IZ  ) = piRHO(IX-1,IY,  IZ  ) + SMALL
          piRHO(IX,  IY+1,IZ  ) = piRHO(IX,  IY+1,IZ  ) + SMALL
          piRHO(IX,  IY-1,IZ  ) = piRHO(IX,  IY-1,IZ  ) + SMALL
          piRHO(IX,  IY,  IZ+1) = piRHO(IX,  IY,  IZ+1) + SMALL
          piRHO(IX,  IY,  IZ-1) = piRHO(IX,  IY,  IZ-1) + SMALL
c           endif    !! sp01/04/02
* to calculate the Gamma factor in each cell
*(1) PX
40       pxl(ix,iy,iz)=pxl(ix,iy,iz)+p(1,I)*BIG
       pxl(ix+1,iy,iz)=pxl(ix+1,iy,iz)+p(1,I)*SMALL
       pxl(ix-1,iy,iz)=pxl(ix-1,iy,iz)+p(1,I)*SMALL
       pxl(ix,iy+1,iz)=pxl(ix,iy+1,iz)+p(1,I)*SMALL
       pxl(ix,iy-1,iz)=pxl(ix,iy-1,iz)+p(1,I)*SMALL
       pxl(ix,iy,iz+1)=pxl(ix,iy,iz+1)+p(1,I)*SMALL
       pxl(ix,iy,iz-1)=pxl(ix,iy,iz-1)+p(1,I)*SMALL
*(2) PY
       pYl(ix,iy,iz)=pYl(ix,iy,iz)+p(2,I)*BIG
       pYl(ix+1,iy,iz)=pYl(ix+1,iy,iz)+p(2,I)*SMALL
       pYl(ix-1,iy,iz)=pYl(ix-1,iy,iz)+p(2,I)*SMALL
       pYl(ix,iy+1,iz)=pYl(ix,iy+1,iz)+p(2,I)*SMALL
       pYl(ix,iy-1,iz)=pYl(ix,iy-1,iz)+p(2,I)*SMALL
       pYl(ix,iy,iz+1)=pYl(ix,iy,iz+1)+p(2,I)*SMALL
       pYl(ix,iy,iz-1)=pYl(ix,iy,iz-1)+p(2,I)*SMALL
* (3) PZ
       pZl(ix,iy,iz)=pZl(ix,iy,iz)+p(3,I)*BIG
       pZl(ix+1,iy,iz)=pZl(ix+1,iy,iz)+p(3,I)*SMALL
       pZl(ix-1,iy,iz)=pZl(ix-1,iy,iz)+p(3,I)*SMALL
       pZl(ix,iy+1,iz)=pZl(ix,iy+1,iz)+p(3,I)*SMALL
       pZl(ix,iy-1,iz)=pZl(ix,iy-1,iz)+p(3,I)*SMALL
       pZl(ix,iy,iz+1)=pZl(ix,iy,iz+1)+p(3,I)*SMALL
       pZl(ix,iy,iz-1)=pZl(ix,iy,iz-1)+p(3,I)*SMALL
* (4) ENERGY
       pel(ix,iy,iz)=pel(ix,iy,iz)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*BIG
       pel(ix+1,iy,iz)=pel(ix+1,iy,iz)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*SMALL
       pel(ix-1,iy,iz)=pel(ix-1,iy,iz)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*SMALL
       pel(ix,iy+1,iz)=pel(ix,iy+1,iz)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*SMALL
       pel(ix,iy-1,iz)=pel(ix,iy-1,iz)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*SMALL
       pel(ix,iy,iz+1)=pel(ix,iy,iz+1)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*SMALL
       pel(ix,iy,iz-1)=pel(ix,iy,iz-1)
     1     +sqrt(e(I)**2+p(1,i)**2+p(2,I)**2+p(3,I)**2)*SMALL
        END IF
  400 CONTINUE
*
      DO 301 IZ = -MAXZ,MAXZ
        DO 201 IY = -MAXX,MAXX
          DO 101 IX = -MAXX,MAXX
      IF((RHO(IX,IY,IZ).EQ.0).OR.(PEL(IX,IY,IZ).EQ.0))
     1GO TO 101
      SMASS2=PEL(IX,IY,IZ)**2-PXL(IX,IY,IZ)**2
     1-PYL(IX,IY,IZ)**2-PZL(IX,IY,IZ)**2
       IF(SMASS2.LE.0)SMASS2=1.E-06
       SMASS=SQRT(SMASS2)
           IF(SMASS.EQ.0.)SMASS=1.e-06
           GAMMA=PEL(IX,IY,IZ)/SMASS
           if(gamma.eq.0)go to 101
       bxx(ix,iy,iz)=pxl(ix,iy,iz)/pel(ix,iy,iz)                  
       byy(ix,iy,iz)=pyl(ix,iy,iz)/pel(ix,iy,iz)       
       bzz(ix,iy,iz)=pzl(ix,iy,iz)/pel(ix,iy,iz)                  
            RHO(IX,IY,IZ) = RHO(IX,IY,IZ)/GAMMA
            RHOn(IX,IY,IZ) = RHOn(IX,IY,IZ)/GAMMA
            RHOp(IX,IY,IZ) = RHOp(IX,IY,IZ)/GAMMA
            piRHO(IX,IY,IZ) = piRHO(IX,IY,IZ)/GAMMA
            pEL(IX,IY,IZ) = pEL(IX,IY,IZ)/(GAMMA**2)
           rho0=0.163
           IF(IPOT.EQ.0)THEN
           U=0
           GO TO 70
           ENDIF
           IF(IPOT.EQ.1.or.ipot.eq.6)THEN
           A=-0.1236
           B=0.0704
           S=2
           GO TO 60
           ENDIF
           IF(IPOT.EQ.2.or.ipot.eq.7)THEN
           A=-0.218
           B=0.164
           S=4./3.
           ENDIF
           IF(IPOT.EQ.3)THEN
           a=-0.3581
           b=0.3048
           S=1.167
           GO TO 60
           ENDIF
           IF(IPOT.EQ.4)THEN
           denr=rho(ix,iy,iz)/rho0         
           b=0.3048
           S=1.167
           if(denr.le.4.or.denr.gt.7)then
           a=-0.3581
           else
           a=-b*denr**(1./6.)-2.*0.036/3.*denr**(-0.333)
           endif
           GO TO 60
           ENDIF
60           U = 0.5*A*RHO(IX,IY,IZ)**2/RHO0 
     1        + B/(1+S) * (RHO(IX,IY,IZ)/RHO0)**S*RHO(IX,IY,IZ)  
70           PEL(IX,IY,IZ)=PEL(IX,IY,IZ)+U
  101     CONTINUE
  201   CONTINUE
  301 CONTINUE
      RETURN
      END

**********************************
*                                                                      *
      SUBROUTINE GRADU(IOPT,IX,IY,IZ,GRADX,GRADY,GRADZ)
*                                                                      *
*       PURPOSE:     DETERMINE GRAD(U(RHO(X,Y,Z)))                     *
*       VARIABLES:                                                     *
*         IOPT                - METHOD FOR EVALUATING THE GRADIENT     *
*                                                      (INTEGER,INPUT) *
*         IX, IY, IZ          - COORDINATES OF POINT   (INTEGER,INPUT) *
*         GRADX, GRADY, GRADZ - GRADIENT OF U            (REAL,OUTPUT) *
*                                                                      *
**********************************
      PARAMETER         (MAXX =    20,  MAXZ =   24)
      PARAMETER         (RHO0 = 0.167)
*
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                  RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                  RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      common  /ss/      inout(20)
cc      SAVE /ss/
      common  /tt/  PEL(-maxx:maxx,-maxx:maxx,-maxz:maxz)
     &,rxy(-maxx:maxx,-maxx:maxx,-maxz:maxz)
cc      SAVE /tt/
      SAVE   
*
      RXPLUS   = RHO(IX+1,IY,  IZ  ) / RHO0
      RXMINS   = RHO(IX-1,IY,  IZ  ) / RHO0
      RYPLUS   = RHO(IX,  IY+1,IZ  ) / RHO0
      RYMINS   = RHO(IX,  IY-1,IZ  ) / RHO0
      RZPLUS   = RHO(IX,  IY,  IZ+1) / RHO0
      RZMINS   = RHO(IX,  IY,  IZ-1) / RHO0
      den0     = RHO(IX,  IY,  IZ) / RHO0
      ene0     = pel(IX,  IY,  IZ) 
*-----------------------------------------------------------------------
      GOTO (1,2,3,4,5) IOPT
       if(iopt.eq.6)go to 6
       if(iopt.eq.7)go to 7
*
    1 CONTINUE
*       POTENTIAL USED IN 1) (STIFF):
*       U = -.124 * RHO/RHO0 + .0705 (RHO/RHO0)**2 GEV
*
           GRADX  = -0.062 * (RXPLUS - RXMINS) + 0.03525 * (RXPLUS**2 -
     &                                                      RXMINS**2)
           GRADY  = -0.062 * (RYPLUS - RYMINS) + 0.03525 * (RYPLUS**2 -
     &                                                      RYMINS**2)
           GRADZ  = -0.062 * (RZPLUS - RZMINS) + 0.03525 * (RZPLUS**2 -
     &                                                      RZMINS**2)
           RETURN
*
    2 CONTINUE
*       POTENTIAL USED IN 2):
*       U = -.218 * RHO/RHO0 + .164 (RHO/RHO0)**(4/3) GEV
*
           EXPNT = 1.3333333
           GRADX = -0.109 * (RXPLUS - RXMINS) 
     &     + 0.082 * (RXPLUS**EXPNT-RXMINS**EXPNT)
           GRADY = -0.109 * (RYPLUS - RYMINS) 
     &     + 0.082 * (RYPLUS**EXPNT-RYMINS**EXPNT)
           GRADZ = -0.109 * (RZPLUS - RZMINS) 
     &     + 0.082 * (RZPLUS**EXPNT-RZMINS**EXPNT)
           RETURN
*
    3 CONTINUE
*       POTENTIAL USED IN 3) (SOFT):
*       U = -.356 * RHO/RHO0 + .303 * (RHO/RHO0)**(7/6)  GEV
*
           EXPNT = 1.1666667
          acoef = 0.178
           GRADX = -acoef * (RXPLUS - RXMINS) 
     &     + 0.1515 * (RXPLUS**EXPNT-RXMINS**EXPNT)
           GRADY = -acoef * (RYPLUS - RYMINS) 
     &     + 0.1515 * (RYPLUS**EXPNT-RYMINS**EXPNT)
           GRADZ = -acoef * (RZPLUS - RZMINS) 
     &     + 0.1515 * (RZPLUS**EXPNT-RZMINS**EXPNT)
                 RETURN
*
*
    4   CONTINUE
*       POTENTIAL USED IN 4) (super-soft in the mixed phase of 4 < rho/rho <7):
*       U1 = -.356 * RHO/RHO0 + .303 * (RHO/RHO0)**(7/6)  GEV
*       normal phase, soft eos of iopt=3
*       U2 = -.02 * (RHO/RHO0)**(2/3) -0.0253 * (RHO/RHO0)**(7/6)  GEV
*
       eh=4.
       eqgp=7.
           acoef=0.178
           EXPNT = 1.1666667
       denr=rho(ix,iy,iz)/rho0
       if(denr.le.eh.or.denr.ge.eqgp)then
           GRADX = -acoef * (RXPLUS - RXMINS) 
     &     + 0.1515 * (RXPLUS**EXPNT-RXMINS**EXPNT)
           GRADY = -acoef * (RYPLUS - RYMINS) 
     &     + 0.1515 * (RYPLUS**EXPNT-RYMINS**EXPNT)
           GRADZ = -acoef * (RZPLUS - RZMINS) 
     &     + 0.1515 * (RZPLUS**EXPNT-RZMINS**EXPNT)
       else
          acoef1=0.178
          acoef2=0.0
          expnt2=2./3.
           GRADX =-acoef1* (RXPLUS**EXPNT-RXMINS**EXPNT)
     &                 -acoef2* (RXPLUS**expnt2 - RXMINS**expnt2) 
           GRADy =-acoef1* (RyPLUS**EXPNT-RyMINS**EXPNT)
     &                 -acoef2* (RyPLUS**expnt2 - RyMINS**expnt2) 
           GRADz =-acoef1* (RzPLUS**EXPNT-RzMINS**EXPNT)
     &                 -acoef2* (RzPLUS**expnt2 - RzMINS**expnt2) 
       endif
       return
*     
    5   CONTINUE
*       POTENTIAL USED IN 5) (SUPER STIFF):
*       U = -.10322 * RHO/RHO0 + .04956 * (RHO/RHO0)**(2.77)  GEV
*
           EXPNT = 2.77
           GRADX = -0.0516 * (RXPLUS - RXMINS) 
     &     + 0.02498 * (RXPLUS**EXPNT-RXMINS**EXPNT)
           GRADY = -0.0516 * (RYPLUS - RYMINS) 
     &     + 0.02498 * (RYPLUS**EXPNT-RYMINS**EXPNT)
           GRADZ = -0.0516 * (RZPLUS - RZMINS) 
     &     + 0.02498 * (RZPLUS**EXPNT-RZMINS**EXPNT)
           RETURN
*
    6 CONTINUE
*       POTENTIAL USED IN 6) (STIFF-qgp):
*       U = -.124 * RHO/RHO0 + .0705 (RHO/RHO0)**2 GEV
*
       if(ene0.le.0.5)then
           GRADX  = -0.062 * (RXPLUS - RXMINS) + 0.03525 * (RXPLUS**2 -
     &                                                      RXMINS**2)
           GRADY  = -0.062 * (RYPLUS - RYMINS) + 0.03525 * (RYPLUS**2 -
     &                                                      RYMINS**2)
           GRADZ  = -0.062 * (RZPLUS - RZMINS) + 0.03525 * (RZPLUS**2 -
     &                                                      RZMINS**2)
           RETURN
       endif
       if(ene0.gt.0.5.and.ene0.le.1.5)then
*       U=c1-ef*rho/rho0**2/3
       ef=36./1000.
           GRADX  = -0.5*ef* (RXPLUS**0.67-RXMINS**0.67)
           GRADy  = -0.5*ef* (RyPLUS**0.67-RyMINS**0.67)
           GRADz  = -0.5*ef* (RzPLUS**0.67-RzMINS**0.67)
           RETURN
       endif
       if(ene0.gt.1.5)then
* U=800*(rho/rho0)**1/3.-Ef*(rho/rho0)**2/3.-c2
       ef=36./1000.
       cf0=0.8
        GRADX  =0.5*cf0*(rxplus**0.333-rxmins**0.333) 
     &         -0.5*ef* (RXPLUS**0.67-RXMINS**0.67)
        GRADy  =0.5*cf0*(ryplus**0.333-rymins**0.333) 
     &         -0.5*ef* (RyPLUS**0.67-RyMINS**0.67)
        GRADz  =0.5*cf0*(rzplus**0.333-rzmins**0.333) 
     &         -0.5*ef* (RzPLUS**0.67-RzMINS**0.67)
           RETURN
       endif
*
    7 CONTINUE
*       POTENTIAL USED IN 7) (Soft-qgp):
       if(den0.le.4.5)then
*       POTENTIAL USED is the same as IN 3) (SOFT):
*       U = -.356 * RHO/RHO0 + .303 * (RHO/RHO0)**(7/6)  GEV
*
           EXPNT = 1.1666667
          acoef = 0.178
           GRADX = -acoef * (RXPLUS - RXMINS) 
     &     + 0.1515 * (RXPLUS**EXPNT-RXMINS**EXPNT)
           GRADY = -acoef * (RYPLUS - RYMINS) 
     &     + 0.1515 * (RYPLUS**EXPNT-RYMINS**EXPNT)
           GRADZ = -acoef * (RZPLUS - RZMINS) 
     &     + 0.1515 * (RZPLUS**EXPNT-RZMINS**EXPNT)
       return
       endif
       if(den0.gt.4.5.and.den0.le.5.1)then
*       U=c1-ef*rho/rho0**2/3
       ef=36./1000.
           GRADX  = -0.5*ef* (RXPLUS**0.67-RXMINS**0.67)
           GRADy  = -0.5*ef* (RyPLUS**0.67-RyMINS**0.67)
           GRADz  = -0.5*ef* (RzPLUS**0.67-RzMINS**0.67)
           RETURN
       endif
       if(den0.gt.5.1)then
* U=800*(rho/rho0)**1/3.-Ef*(rho/rho0)**2/3.-c2
       ef=36./1000.
       cf0=0.8
        GRADX  =0.5*cf0*(rxplus**0.333-rxmins**0.333) 
     &         -0.5*ef* (RXPLUS**0.67-RXMINS**0.67)
        GRADy  =0.5*cf0*(ryplus**0.333-rymins**0.333) 
     &         -0.5*ef* (RyPLUS**0.67-RyMINS**0.67)
        GRADz  =0.5*cf0*(rzplus**0.333-rzmins**0.333) 
     &         -0.5*ef* (RzPLUS**0.67-RzMINS**0.67)
           RETURN
       endif
        END
**********************************
*                                                                      *
      SUBROUTINE GRADUK(IX,IY,IZ,GRADXk,GRADYk,GRADZk)
*                                                                      *
*       PURPOSE:     DETERMINE the baryon density gradient for         *
*                    proporgating kaons in a mean field caused by      *
*                    surrounding baryons                               * 
*       VARIABLES:                                                     *
*         IX, IY, IZ          - COORDINATES OF POINT   (INTEGER,INPUT) *
*         GRADXk, GRADYk, GRADZk                       (REAL,OUTPUT)   *
*                                                                      *
**********************************
      PARAMETER         (MAXX =    20,  MAXZ =   24)
      PARAMETER         (RHO0 = 0.168)
*
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                  RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                  RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      common  /ss/      inout(20)
cc      SAVE /ss/
      SAVE   
*
      RXPLUS   = RHO(IX+1,IY,  IZ  ) 
      RXMINS   = RHO(IX-1,IY,  IZ  ) 
      RYPLUS   = RHO(IX,  IY+1,IZ  ) 
      RYMINS   = RHO(IX,  IY-1,IZ  ) 
      RZPLUS   = RHO(IX,  IY,  IZ+1) 
      RZMINS   = RHO(IX,  IY,  IZ-1) 
           GRADXk  = (RXPLUS - RXMINS)/2. 
           GRADYk  = (RYPLUS - RYMINS)/2.
           GRADZk  = (RZPLUS - RZMINS)/2.
           RETURN
           END
*-----------------------------------------------------------------------
      SUBROUTINE GRADUP(IX,IY,IZ,GRADXP,GRADYP,GRADZP)
*                                                                      *
*       PURPOSE:     DETERMINE THE GRADIENT OF THE PROTON DENSITY      *
*       VARIABLES:                                                     *
*                                                                           *
*         IX, IY, IZ          - COORDINATES OF POINT   (INTEGER,INPUT) *
*         GRADXP, GRADYP, GRADZP - GRADIENT OF THE PROTON              *
*                                  DENSITY(REAL,OUTPUT)                *
*                                                                      *
**********************************
      PARAMETER         (MAXX =    20,  MAXZ =   24)
      PARAMETER         (RHO0 = 0.168)
*
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      common  /ss/      inout(20)
cc      SAVE /ss/
      SAVE   
*
      RXPLUS   = RHOP(IX+1,IY,  IZ  ) / RHO0
      RXMINS   = RHOP(IX-1,IY,  IZ  ) / RHO0
      RYPLUS   = RHOP(IX,  IY+1,IZ  ) / RHO0
      RYMINS   = RHOP(IX,  IY-1,IZ  ) / RHO0
      RZPLUS   = RHOP(IX,  IY,  IZ+1) / RHO0
      RZMINS   = RHOP(IX,  IY,  IZ-1) / RHO0
*-----------------------------------------------------------------------
*
           GRADXP  = (RXPLUS - RXMINS)/2. 
           GRADYP  = (RYPLUS - RYMINS)/2.
           GRADZP  = (RZPLUS - RZMINS)/2.
           RETURN
      END
*-----------------------------------------------------------------------
      SUBROUTINE GRADUN(IX,IY,IZ,GRADXN,GRADYN,GRADZN)
*                                                                      *
*       PURPOSE:     DETERMINE THE GRADIENT OF THE NEUTRON DENSITY     *
*       VARIABLES:                                                     *
*                                                                           *
*         IX, IY, IZ          - COORDINATES OF POINT   (INTEGER,INPUT) *
*         GRADXN, GRADYN, GRADZN - GRADIENT OF THE NEUTRON             *
*                                  DENSITY(REAL,OUTPUT)                *
*                                                                      *
**********************************
      PARAMETER         (MAXX =    20,  MAXZ =   24)
      PARAMETER         (RHO0 = 0.168)
*
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      common  /ss/      inout(20)
cc      SAVE /ss/
      SAVE   
*
      RXPLUS   = RHON(IX+1,IY,  IZ  ) / RHO0
      RXMINS   = RHON(IX-1,IY,  IZ  ) / RHO0
      RYPLUS   = RHON(IX,  IY+1,IZ  ) / RHO0
      RYMINS   = RHON(IX,  IY-1,IZ  ) / RHO0
      RZPLUS   = RHON(IX,  IY,  IZ+1) / RHO0
      RZMINS   = RHON(IX,  IY,  IZ-1) / RHO0
*-----------------------------------------------------------------------
*
           GRADXN  = (RXPLUS - RXMINS)/2. 
           GRADYN  = (RYPLUS - RYMINS)/2.
           GRADZN  = (RZPLUS - RZMINS)/2.
           RETURN
      END

*-----------------------------------------------------------------------------
*FUNCTION FDE(DMASS) GIVES DELTA MASS DISTRIBUTION BY USING OF
*KITAZOE'S FORMULA
        REAL FUNCTION FDE(DMASS,SRT,CON)
      SAVE   
        AMN=0.938869
        AVPI=0.13803333
        AM0=1.232
        FD=4.*(AM0**2)*WIDTH(DMASS)/((DMASS**2-1.232**2)**2
     1  +AM0**2*WIDTH(DMASS)**2)
        IF(CON.EQ.1.)THEN
        P11=(SRT**2+DMASS**2-AMN**2)**2
     1  /(4.*SRT**2)-DMASS**2
       if(p11.le.0)p11=1.E-06
       p1=sqrt(p11)
        ELSE
        DMASS=AMN+AVPI
        P11=(SRT**2+DMASS**2-AMN**2)**2
     1  /(4.*SRT**2)-DMASS**2
       if(p11.le.0)p11=1.E-06
       p1=sqrt(p11)
        ENDIF
        FDE=FD*P1*DMASS
        RETURN
        END
*-------------------------------------------------------------
*FUNCTION FDE(DMASS) GIVES N*(1535) MASS DISTRIBUTION BY USING OF
*KITAZOE'S FORMULA
        REAL FUNCTION FD5(DMASS,SRT,CON)
      SAVE   
        AMN=0.938869
        AVPI=0.13803333
        AM0=1.535
        FD=4.*(AM0**2)*W1535(DMASS)/((DMASS**2-1.535**2)**2
     1  +AM0**2*W1535(DMASS)**2)
        IF(CON.EQ.1.)THEN
        P1=SQRT((SRT**2+DMASS**2-AMN**2)**2
     1  /(4.*SRT**2)-DMASS**2)
        ELSE
        DMASS=AMN+AVPI
        P1=SQRT((SRT**2+DMASS**2-AMN**2)**2
     1  /(4.*SRT**2)-DMASS**2)
        ENDIF
        FD5=FD*P1*DMASS
        RETURN
        END
*--------------------------------------------------------------------------
*FUNCTION FNS(DMASS) GIVES N* MASS DISTRIBUTION 
c     BY USING OF BREIT-WIGNER FORMULA
        REAL FUNCTION FNS(DMASS,SRT,CON)
      SAVE   
        WIDTH=0.2
        AMN=0.938869
        AVPI=0.13803333
        AN0=1.43
        FN=4.*(AN0**2)*WIDTH/((DMASS**2-1.44**2)**2+AN0**2*WIDTH**2)
        IF(CON.EQ.1.)THEN
        P1=SQRT((SRT**2+DMASS**2-AMN**2)**2
     1  /(4.*SRT**2)-DMASS**2)
        ELSE
        DMASS=AMN+AVPI
        P1=SQRT((SRT**2+DMASS**2-AMN**2)**2
     1  /(4.*SRT**2)-DMASS**2)
        ENDIF
        FNS=FN*P1*DMASS
        RETURN
        END
*-----------------------------------------------------------------------------
*-----------------------------------------------------------------------------
* PURPOSE:1. SORT N*(1440) and N*(1535) 2-body DECAY PRODUCTS
*         2. DETERMINE THE MOMENTUM AND COORDINATES OF NUCLEON AND PION
*            AFTER THE DELTA OR N* DECAYING
* DATE   : JAN. 24,1990, MODIFIED ON MAY 17, 1994 TO INCLUDE ETA 
        SUBROUTINE DECAY(IRUN,I,NNN,ISEED,wid,nt)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,ETAM=0.5475,AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
        COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &       IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
      COMMON/RNDF77/NSEED
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
cc      SAVE /RNDF77/
      SAVE   
        lbanti=LB(I)
c
        DM=E(I)
*1. FOR N*+(1440) DECAY
        IF(iabs(LB(I)).EQ.11)THEN
           X3=RANART(NSEED)
           IF(X3.GT.(1./3.))THEN
              LB(I)=2
              NLAB=2
              LPION(NNN,IRUN)=5
              EPION(NNN,IRUN)=AP2
           ELSE
              LB(I)=1
              NLAB=1
              LPION(NNN,IRUN)=4
              EPION(NNN,IRUN)=AP1
           ENDIF
*2. FOR N*0(1440) DECAY
        ELSEIF(iabs(LB(I)).EQ.10)THEN
           X4=RANART(NSEED)
           IF(X4.GT.(1./3.))THEN
              LB(I)=1
              NLAB=1
              LPION(NNN,IRUN)=3
              EPION(NNN,IRUN)=AP2
           ELSE
              LB(I)=2
              NALB=2
              LPION(NNN,IRUN)=4
              EPION(NNN,IRUN)=AP1
           ENDIF
* N*(1535) CAN DECAY TO A PION OR AN ETA IF DM > 1.49 GeV
*3 N*(0)(1535) DECAY
        ELSEIF(iabs(LB(I)).EQ.12)THEN
           CTRL=0.65
           IF(DM.lE.1.49)ctrl=-1.
           X5=RANART(NSEED)
           IF(X5.GE.ctrl)THEN
* DECAY TO PION+NUCLEON
              X6=RANART(NSEED)
              IF(X6.GT.(1./3.))THEN
                 LB(I)=1
                 NLAB=1
                 LPION(NNN,IRUN)=3
                 EPION(NNN,IRUN)=AP2
              ELSE
                 LB(I)=2
                 NALB=2
                 LPION(NNN,IRUN)=4
                 EPION(NNN,IRUN)=AP1
              ENDIF
           ELSE
* DECAY TO ETA+NEUTRON
              LB(I)=2
              NLAB=2
              LPION(NNN,IRUN)=0
              EPION(NNN,IRUN)=ETAM
           ENDIF
*4. FOR N*+(1535) DECAY
        ELSEIF(iabs(LB(I)).EQ.13)THEN
           CTRL=0.65
           IF(DM.lE.1.49)ctrl=-1.
           X5=RANART(NSEED)
           IF(X5.GE.ctrl)THEN
* DECAY TO PION+NUCLEON
              X8=RANART(NSEED)
              IF(X8.GT.(1./3.))THEN
                 LB(I)=2
                 NLAB=2
                 LPION(NNN,IRUN)=5
                 EPION(NNN,IRUN)=AP2
              ELSE
                 LB(I)=1
                 NLAB=1
                 LPION(NNN,IRUN)=4
                 EPION(NNN,IRUN)=AP1
              ENDIF
           ELSE
* DECAY TO ETA+NUCLEON
              LB(I)=1
              NLAB=1
              LPION(NNN,IRUN)=0
              EPION(NNN,IRUN)=ETAM
           ENDIF
        ENDIF
c
        CALL DKINE(IRUN,I,NNN,NLAB,ISEED,wid,nt)
c
c     anti-particle ID for anti-N* decays:
        if(lbanti.lt.0) then
           lbi=LB(I)
           if(lbi.eq.1.or.lbi.eq.2) then
              lbi=-lbi
           elseif(lbi.eq.3) then
              lbi=5
           elseif(lbi.eq.5) then
              lbi=3
           endif
           LB(I)=lbi
c
           lbi=LPION(NNN,IRUN)
           if(lbi.eq.3) then
              lbi=5
           elseif(lbi.eq.5) then
              lbi=3
           elseif(lbi.eq.1.or.lbi.eq.2) then
              lbi=-lbi
           endif
           LPION(NNN,IRUN)=lbi
        endif
c
        if(nt.eq.ntmax) then
c     at the last timestep, assign rho or eta (decay daughter) 
c     to lb(i1) only (not to lpion) in order to decay them again:
           lbm=LPION(NNN,IRUN)
           if(lbm.eq.0.or.lbm.eq.25
     1          .or.lbm.eq.26.or.lbm.eq.27) then
c     switch rho or eta with baryon, positions are the same (no change needed):
              lbsave=lbm
              xmsave=EPION(NNN,IRUN)
              pxsave=PPION(1,NNN,IRUN)
              pysave=PPION(2,NNN,IRUN)
              pzsave=PPION(3,NNN,IRUN)
clin-5/2008:
              dpsave=dppion(NNN,IRUN)
              LPION(NNN,IRUN)=LB(I)
              EPION(NNN,IRUN)=E(I)
              PPION(1,NNN,IRUN)=P(1,I)
              PPION(2,NNN,IRUN)=P(2,I)
              PPION(3,NNN,IRUN)=P(3,I)
clin-5/2008:
              dppion(NNN,IRUN)=dpertp(I)
              LB(I)=lbsave
              E(I)=xmsave
              P(1,I)=pxsave
              P(2,I)=pysave
              P(3,I)=pzsave
clin-5/2008:
              dpertp(I)=dpsave
           endif
        endif

       RETURN
       END

*-------------------------------------------------------------------
*-------------------------------------------------------------------
* PURPOSE:
*         CALCULATE THE MOMENTUM OF NUCLEON AND PION (OR ETA) 
*         IN THE LAB. FRAME AFTER DELTA OR N* DECAY
* DATE   : JAN. 24,1990, MODIFIED ON MAY 17, 1994 TO INCLUDE ETA PRODUCTION
        SUBROUTINE DKINE(IRUN,I,NNN,NLAB,ISEED,wid,nt)
        PARAMETER (hbarc=0.19733)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,ETAM=0.5475,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1 px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
        COMMON/tdecay/tfdcy(MAXSTR),tfdpi(MAXSTR,MAXR),tft(MAXSTR)
cc      SAVE /tdecay/
        COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &       IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
        EXTERNAL IARFLV, INVFLV
      SAVE   
        ISEED=ISEED
* READ IN THE COORDINATES OF DELTA OR N* UNDERGOING DECAY
        PX=P(1,I)
        PY=P(2,I)
        PZ=P(3,I)
        RX=R(1,I)
        RY=R(2,I)
        RZ=R(3,I)
        DM=E(I)
        EDELTA=SQRT(DM**2+PX**2+PY**2+PZ**2)
        PM=EPION(NNN,IRUN)
        AM=AMP
        IF(NLAB.EQ.2)AM=AMN
* FIND OUT THE MOMENTUM AND ENERGY OF PION AND NUCLEON IN DELTA REST FRAME
* THE MAGNITUDE OF MOMENTUM IS DETERMINED BY ENERGY CONSERVATION ,THE FORMULA
* CAN BE FOUND ON PAGE 716,W BAUER P.R.C40,1989
* THE DIRECTION OF THE MOMENTUM IS ASSUMED ISOTROPIC. NOTE THAT P(PION)=-P(N)
        Q2=((DM**2-AM**2+PM**2)/(2.*DM))**2-PM**2
        IF(Q2.LE.0.)Q2=1.e-09
        Q=SQRT(Q2)
11      QX=1.-2.*RANART(NSEED)
        QY=1.-2.*RANART(NSEED)
        QZ=1.-2.*RANART(NSEED)
        QS=QX**2+QY**2+QZ**2
        IF(QS.GT.1.) GO TO 11
        PXP=Q*QX/SQRT(QS)
        PYP=Q*QY/SQRT(QS)
        PZP=Q*QZ/SQRT(QS)
        EP=SQRT(Q**2+PM**2)
        PXN=-PXP
        PYN=-PYP
        PZN=-PZP
        EN=SQRT(Q**2+AM**2)
* TRANSFORM INTO THE LAB. FRAME. THE GENERAL LORENTZ TRANSFORMATION CAN
* BE FOUND ON PAGE 34 OF R. HAGEDORN " RELATIVISTIC KINEMATICS"
        GD=EDELTA/DM
        FGD=GD/(1.+GD)
        BDX=PX/EDELTA
        BDY=PY/EDELTA
        BDZ=PZ/EDELTA
        BPP=BDX*PXP+BDY*PYP+BDZ*PZP
        BPN=BDX*PXN+BDY*PYN+BDZ*PZN
        P(1,I)=PXN+BDX*GD*(FGD*BPN+EN)
        P(2,I)=PYN+BDY*GD*(FGD*BPN+EN)
        P(3,I)=PZN+BDZ*GD*(FGD*BPN+EN)
        E(I)=AM
* WE ASSUME THAT THE SPACIAL COORDINATE OF THE NUCLEON
* IS THAT OF THE DELTA
        PPION(1,NNN,IRUN)=PXP+BDX*GD*(FGD*BPP+EP)
        PPION(2,NNN,IRUN)=PYP+BDY*GD*(FGD*BPP+EP)
        PPION(3,NNN,IRUN)=PZP+BDZ*GD*(FGD*BPP+EP)
clin-5/2008:
        dppion(NNN,IRUN)=dpertp(I)
* WE ASSUME THE PION OR ETA COMING FROM DELTA DECAY IS LOCATED ON THE SPHERE
* OF RADIUS 0.5FM AROUND DELTA, THIS POINT NEED TO BE CHECKED 
* AND OTHER CRIERTION MAY BE TRIED
clin-2/20/03 no additional smearing for position of decay daughters:
c200         X0 = 1.0 - 2.0 * RANART(NSEED)
c            Y0 = 1.0 - 2.0 * RANART(NSEED)
c            Z0 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X0*X0+Y0*Y0+Z0*Z0) .GT. 1.0) GOTO 200
c        RPION(1,NNN,IRUN)=R(1,I)+0.5*x0
c        RPION(2,NNN,IRUN)=R(2,I)+0.5*y0
c        RPION(3,NNN,IRUN)=R(3,I)+0.5*z0
        RPION(1,NNN,IRUN)=R(1,I)
        RPION(2,NNN,IRUN)=R(2,I)
        RPION(3,NNN,IRUN)=R(3,I)
c
        devio=SQRT(EPION(NNN,IRUN)**2+PPION(1,NNN,IRUN)**2
     1       +PPION(2,NNN,IRUN)**2+PPION(3,NNN,IRUN)**2)
     2       +SQRT(E(I)**2+P(1,I)**2+P(2,I)**2+P(3,I)**2)-e1
c        if(abs(devio).gt.0.02) write(93,*) 'decay(): nt=',nt,devio,lb1

c     add decay time to daughter's formation time at the last timestep:
        if(nt.eq.ntmax) then
           tau0=hbarc/wid
           taudcy=tau0*(-1.)*alog(1.-RANART(NSEED))
c     lorentz boost:
           taudcy=taudcy*e1/em1
           tfnl=tfnl+taudcy
           xfnl=xfnl+px1/e1*taudcy
           yfnl=yfnl+py1/e1*taudcy
           zfnl=zfnl+pz1/e1*taudcy
           R(1,I)=xfnl
           R(2,I)=yfnl
           R(3,I)=zfnl
           tfdcy(I)=tfnl
           RPION(1,NNN,IRUN)=xfnl
           RPION(2,NNN,IRUN)=yfnl
           RPION(3,NNN,IRUN)=zfnl
           tfdpi(NNN,IRUN)=tfnl
        endif

cc 200    format(a30,2(1x,e10.4))
cc 210    format(i6,5(1x,f8.3))
cc 220    format(a2,i5,5(1x,f8.3))

        RETURN
        END

*-----------------------------------------------------------------------------
*-----------------------------------------------------------------------------
* PURPOSE:1. N*-->N+PION+PION  DECAY PRODUCTS
*         2. DETERMINE THE MOMENTUM AND COORDINATES OF NUCLEON AND PION
*            AFTER THE DELTA OR N* DECAYING
* DATE   : NOV.7,1994
*----------------------------------------------------------------------------
        SUBROUTINE DECAY2(IRUN,I,NNN,ISEED,wid,nt)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,ETAM=0.5475,AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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

        lbanti=LB(I)
c
        DM=E(I)
* DETERMINE THE DECAY PRODUCTS
* FOR N*+(1440) DECAY
        IF(iabs(LB(I)).EQ.11)THEN
           X3=RANART(NSEED)
           IF(X3.LT.(1./3))THEN
              LB(I)=2
              NLAB=2
              LPION(NNN,IRUN)=5
              EPION(NNN,IRUN)=AP2
              LPION(NNN+1,IRUN)=4
              EPION(NNN+1,IRUN)=AP1
           ELSEIF(X3.LT.2./3.AND.X3.GT.1./3.)THEN
              LB(I)=1
              NLAB=1
              LPION(NNN,IRUN)=5
              EPION(NNN,IRUN)=AP2
              LPION(NNN+1,IRUN)=3
              EPION(NNN+1,IRUN)=AP2
           ELSE
              LB(I)=1
              NLAB=1
              LPION(NNN,IRUN)=4
              EPION(NNN,IRUN)=AP1
              LPION(NNN+1,IRUN)=4
              EPION(NNN+1,IRUN)=AP1
           ENDIF
* FOR N*0(1440) DECAY
        ELSEIF(iabs(LB(I)).EQ.10)THEN
           X3=RANART(NSEED)
           IF(X3.LT.(1./3))THEN
              LB(I)=2
              NLAB=2
              LPION(NNN,IRUN)=4
              EPION(NNN,IRUN)=AP1
              LPION(NNN+1,IRUN)=4
              EPION(NNN+1,IRUN)=AP1
           ELSEIF(X3.LT.2./3.AND.X3.GT.1./3.)THEN
              LB(I)=1
              NLAB=1
              LPION(NNN,IRUN)=3
              EPION(NNN,IRUN)=AP2
              LPION(NNN+1,IRUN)=4
              EPION(NNN+1,IRUN)=AP1
           ELSE
              LB(I)=2
              NLAB=2
              LPION(NNN,IRUN)=5
              EPION(NNN,IRUN)=AP2
              LPION(NNN+1,IRUN)=3
              EPION(NNN+1,IRUN)=AP2
           ENDIF
        ENDIF

        CALL DKINE2(IRUN,I,NNN,NLAB,ISEED,wid,nt)
c
c     anti-particle ID for anti-N* decays:
        if(lbanti.lt.0) then
           lbi=LB(I)
           if(lbi.eq.1.or.lbi.eq.2) then
              lbi=-lbi
           elseif(lbi.eq.3) then
              lbi=5
           elseif(lbi.eq.5) then
              lbi=3
           endif
           LB(I)=lbi
c
           lbi=LPION(NNN,IRUN)
           if(lbi.eq.3) then
              lbi=5
           elseif(lbi.eq.5) then
              lbi=3
           elseif(lbi.eq.1.or.lbi.eq.2) then
              lbi=-lbi
           endif
           LPION(NNN,IRUN)=lbi
c
           lbi=LPION(NNN+1,IRUN)
           if(lbi.eq.3) then
              lbi=5
           elseif(lbi.eq.5) then
              lbi=3
           elseif(lbi.eq.1.or.lbi.eq.2) then
              lbi=-lbi
           endif
           LPION(NNN+1,IRUN)=lbi
        endif
c
       RETURN
       END
*-------------------------------------------------------------------
*--------------------------------------------------------------------------
*         CALCULATE THE MOMENTUM OF NUCLEON AND PION (OR ETA) 
*         IN THE LAB. FRAME AFTER DELTA OR N* DECAY
* DATE   : JAN. 24,1990, MODIFIED ON MAY 17, 1994 TO INCLUDE ETA PRODUCTION
*--------------------------------------------------------------------------
        SUBROUTINE DKINE2(IRUN,I,NNN,NLAB,ISEED,wid,nt)
        PARAMETER (hbarc=0.19733)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,ETAM=0.5475,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1 px1n,py1n,pz1n,dp1n
cc      SAVE /leadng/
        COMMON/tdecay/tfdcy(MAXSTR),tfdpi(MAXSTR,MAXR),tft(MAXSTR)
cc      SAVE /tdecay/
        COMMON /INPUT2/ ILAB, MANYB, NTMAX, ICOLL, INSYS, IPOT, MODE, 
     &       IMOMEN, NFREQ, ICFLOW, ICRHO, ICOU, KPOTEN, KMUL
cc      SAVE /INPUT2/
        EXTERNAL IARFLV, INVFLV
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      SAVE   
 
        ISEED=ISEED
* READ IN THE COORDINATES OF THE N*(1440) UNDERGOING DECAY
        PX=P(1,I)
        PY=P(2,I)
        PZ=P(3,I)
        RX=R(1,I)
        RY=R(2,I)
        RZ=R(3,I)
        DM=E(I)
        EDELTA=SQRT(DM**2+PX**2+PY**2+PZ**2)
        PM1=EPION(NNN,IRUN)
        PM2=EPION(NNN+1,IRUN)
        AM=AMN
       IF(NLAB.EQ.1)AM=AMP
* THE MAXIMUM MOMENTUM OF THE NUCLEON FROM THE DECAY OF A N*
       PMAX2=(DM**2-(AM+PM1+PM2)**2)*(DM**2-(AM-PM1-PM2)**2)/4/DM**2
       PMAX=SQRT(PMAX2)
* GENERATE THE MOMENTUM OF THE NUCLEON IN THE N* REST FRAME
       CSS=1.-2.*RANART(NSEED)
       SSS=SQRT(1-CSS**2)
       FAI=2*PI*RANART(NSEED)
       PX0=PMAX*SSS*COS(FAI)
       PY0=PMAX*SSS*SIN(FAI)
       PZ0=PMAX*CSS
       EP0=SQRT(PX0**2+PY0**2+PZ0**2+AM**2)
clin-5/23/01 bug: P0 for pion0 is equal to PMAX, leaving pion+ and pion- 
c     without no relative momentum, thus producing them with equal momenta, 
* BETA AND GAMMA OF THE CMS OF PION+-PION-
       BETAX=-PX0/(DM-EP0)
       BETAY=-PY0/(DM-EP0)
       BETAZ=-PZ0/(DM-EP0)
       GD1=1./SQRT(1-BETAX**2-BETAY**2-BETAZ**2)
       FGD1=GD1/(1+GD1)
* GENERATE THE MOMENTA OF PIONS IN THE CMS OF PION+PION-
        Q2=((DM-EP0)/(2.*GD1))**2-PM1**2
        IF(Q2.LE.0.)Q2=1.E-09
        Q=SQRT(Q2)
11      QX=1.-2.*RANART(NSEED)
        QY=1.-2.*RANART(NSEED)
        QZ=1.-2.*RANART(NSEED)
        QS=QX**2+QY**2+QZ**2
        IF(QS.GT.1.) GO TO 11
        PXP=Q*QX/SQRT(QS)
        PYP=Q*QY/SQRT(QS)
        PZP=Q*QZ/SQRT(QS)
        EP=SQRT(Q**2+PM1**2)
        PXN=-PXP
        PYN=-PYP
        PZN=-PZP
        EN=SQRT(Q**2+PM2**2)
* TRANSFORM THE MOMENTA OF PION+PION- INTO THE N* REST FRAME
        BPP1=BETAX*PXP+BETAY*PYP+BETAZ*PZP
        BPN1=BETAX*PXN+BETAY*PYN+BETAZ*PZN
* FOR PION-
        P1M=PXN+BETAX*GD1*(FGD1*BPN1+EN)
        P2M=PYN+BETAY*GD1*(FGD1*BPN1+EN)
        P3M=PZN+BETAZ*GD1*(FGD1*BPN1+EN)
       EPN=SQRT(P1M**2+P2M**2+P3M**2+PM2**2)
* FOR PION+
        P1P=PXP+BETAX*GD1*(FGD1*BPP1+EP)
        P2P=PYP+BETAY*GD1*(FGD1*BPP1+EP)
        P3P=PZP+BETAZ*GD1*(FGD1*BPP1+EP)
       EPP=SQRT(P1P**2+P2P**2+P3P**2+PM1**2)
* TRANSFORM MOMENTA OF THE THREE PIONS INTO THE 
* THE NUCLEUS-NUCLEUS CENTER OF MASS  FRAME. 
* THE GENERAL LORENTZ TRANSFORMATION CAN
* BE FOUND ON PAGE 34 OF R. HAGEDORN " RELATIVISTIC KINEMATICS"
        GD=EDELTA/DM
        FGD=GD/(1.+GD)
        BDX=PX/EDELTA
        BDY=PY/EDELTA
        BDZ=PZ/EDELTA
       BP0=BDX*PX0+BDY*PY0+BDZ*PZ0
        BPP=BDX*P1P+BDY*P2P+BDZ*P3P
        BPN=BDX*P1M+BDY*P2M+BDZ*P3M
* FOR THE NUCLEON
        P(1,I)=PX0+BDX*GD*(FGD*BP0+EP0)
        P(2,I)=PY0+BDY*GD*(FGD*BP0+EP0)
        P(3,I)=PZ0+BDZ*GD*(FGD*BP0+EP0)
       E(I)=am
       ID(I)=0
       enucl=sqrt(p(1,i)**2+p(2,i)**2+p(3,i)**2+e(i)**2)
* WE ASSUME THAT THE SPACIAL COORDINATE OF THE PION0
* IS in a sphere of radius 0.5 fm around N*
* FOR PION+
        PPION(1,NNN,IRUN)=P1P+BDX*GD*(FGD*BPP+EPP)
        PPION(2,NNN,IRUN)=P2P+BDY*GD*(FGD*BPP+EPP)
        PPION(3,NNN,IRUN)=P3P+BDZ*GD*(FGD*BPP+EPP)
       epion1=sqrt(ppion(1,nnn,irun)**2
     &  +ppion(2,nnn,irun)**2+ppion(3,nnn,irun)**2
     &  +epion(nnn,irun)**2)
clin-2/20/03 no additional smearing for position of decay daughters:
c200         X0 = 1.0 - 2.0 * RANART(NSEED)
c            Y0 = 1.0 - 2.0 * RANART(NSEED)
c            Z0 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X0*X0+Y0*Y0+Z0*Z0) .GT. 1.0) GOTO 200
c        RPION(1,NNN,IRUN)=R(1,I)+0.5*x0
c        RPION(2,NNN,IRUN)=R(2,I)+0.5*y0
c        RPION(3,NNN,IRUN)=R(3,I)+0.5*z0
        RPION(1,NNN,IRUN)=R(1,I)
        RPION(2,NNN,IRUN)=R(2,I)
        RPION(3,NNN,IRUN)=R(3,I)
* FOR PION-
        PPION(1,NNN+1,IRUN)=P1M+BDX*GD*(FGD*BPN+EPN)
        PPION(2,NNN+1,IRUN)=P2M+BDY*GD*(FGD*BPN+EPN)
        PPION(3,NNN+1,IRUN)=P3M+BDZ*GD*(FGD*BPN+EPN)
clin-5/2008:
        dppion(NNN,IRUN)=dpertp(I)
        dppion(NNN+1,IRUN)=dpertp(I)
c
       epion2=sqrt(ppion(1,nnn+1,irun)**2
     &  +ppion(2,nnn+1,irun)**2+ppion(3,nnn+1,irun)**2
     &  +epion(nnn+1,irun)**2)
clin-2/20/03 no additional smearing for position of decay daughters:
c300         X0 = 1.0 - 2.0 * RANART(NSEED)
c            Y0 = 1.0 - 2.0 * RANART(NSEED)
c            Z0 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X0*X0+Y0*Y0+Z0*Z0) .GT. 1.0) GOTO 300
c        RPION(1,NNN+1,IRUN)=R(1,I)+0.5*x0
c        RPION(2,NNN+1,IRUN)=R(2,I)+0.5*y0
c        RPION(3,NNN+1,IRUN)=R(3,I)+0.5*z0
        RPION(1,NNN+1,IRUN)=R(1,I)
        RPION(2,NNN+1,IRUN)=R(2,I)
        RPION(3,NNN+1,IRUN)=R(3,I)
c
* check energy conservation in the decay
c       efinal=enucl+epion1+epion2
c       DEEE=(EDELTA-EFINAL)/EDELTA
c       IF(ABS(DEEE).GE.1.E-03)write(6,*)1,edelta,efinal

        devio=SQRT(EPION(NNN,IRUN)**2+PPION(1,NNN,IRUN)**2
     1       +PPION(2,NNN,IRUN)**2+PPION(3,NNN,IRUN)**2)
     2       +SQRT(E(I)**2+P(1,I)**2+P(2,I)**2+P(3,I)**2)
     3       +SQRT(EPION(NNN+1,IRUN)**2+PPION(1,NNN+1,IRUN)**2
     4       +PPION(2,NNN+1,IRUN)**2+PPION(3,NNN+1,IRUN)**2)-e1
c        if(abs(devio).gt.0.02) write(93,*) 'decay2(): nt=',nt,devio,lb1

c     add decay time to daughter's formation time at the last timestep:
        if(nt.eq.ntmax) then
           tau0=hbarc/wid
           taudcy=tau0*(-1.)*alog(1.-RANART(NSEED))
c     lorentz boost:
           taudcy=taudcy*e1/em1
           tfnl=tfnl+taudcy
           xfnl=xfnl+px1/e1*taudcy
           yfnl=yfnl+py1/e1*taudcy
           zfnl=zfnl+pz1/e1*taudcy
           R(1,I)=xfnl
           R(2,I)=yfnl
           R(3,I)=zfnl
           tfdcy(I)=tfnl
           RPION(1,NNN,IRUN)=xfnl
           RPION(2,NNN,IRUN)=yfnl
           RPION(3,NNN,IRUN)=zfnl
           tfdpi(NNN,IRUN)=tfnl
           RPION(1,NNN+1,IRUN)=xfnl
           RPION(2,NNN+1,IRUN)=yfnl
           RPION(3,NNN+1,IRUN)=zfnl
           tfdpi(NNN+1,IRUN)=tfnl
        endif

cc 200    format(a30,2(1x,e10.4))
cc 210    format(i6,5(1x,f8.3))
cc 220    format(a2,i5,5(1x,f8.3))

        RETURN
        END
*---------------------------------------------------------------------------
*---------------------------------------------------------------------------
* PURPOSE : CALCULATE THE MASS AND MOMENTUM OF BARYON RESONANCE 
*           AFTER PION OR ETA BEING ABSORBED BY A NUCLEON
* NOTE    : 
*           
* DATE    : JAN.29,1990
        SUBROUTINE DRESON(I1,I2)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      SAVE   
* 1. DETERMINE THE MOMENTUM COMPONENT OF DELTA/N* IN THE LAB. FRAME
        E10=SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
        E20=SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
        IF(iabs(LB(I2)) .EQ. 1 .OR. iabs(LB(I2)) .EQ. 2 .OR.
     &     (iabs(LB(I2)) .GE. 6 .AND. iabs(LB(I2)) .LE. 17)) THEN
        E(I1)=0.
        I=I2
        ELSE
        E(I2)=0.
        I=I1
        ENDIF
        P(1,I)=P(1,I1)+P(1,I2)
        P(2,I)=P(2,I1)+P(2,I2)
        P(3,I)=P(3,I1)+P(3,I2)
* 2. DETERMINE THE MASS OF DELTA/N* BY USING THE REACTION KINEMATICS
        DM=SQRT((E10+E20)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2)
        E(I)=DM
        RETURN
        END
*---------------------------------------------------------------------------
* PURPOSE : CALCULATE THE MASS AND MOMENTUM OF RHO RESONANCE 
*           AFTER PION + PION COLLISION
* DATE    : NOV. 30,1994
        SUBROUTINE RHORES(I1,I2)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      SAVE   
* 1. DETERMINE THE MOMENTUM COMPONENT OF THE RHO IN THE CMS OF NN FRAME
*    WE LET I1 TO BE THE RHO AND ABSORB I2
        E10=SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
        E20=SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
        P(1,I1)=P(1,I1)+P(1,I2)
        P(2,I1)=P(2,I1)+P(2,I2)
        P(3,I1)=P(3,I1)+P(3,I2)
* 2. DETERMINE THE MASS OF THE RHO BY USING THE REACTION KINEMATICS
        DM=SQRT((E10+E20)**2-P(1,I1)**2-P(2,I1)**2-P(3,I1)**2)
        E(I1)=DM
       E(I2)=0
        RETURN
        END
*---------------------------------------------------------------------------
* PURPOSE : CALCULATE THE PION+NUCLEON CROSS SECTION ACCORDING TO THE
*           BREIT-WIGNER FORMULA/(p*)**2
* VARIABLE : LA = 1 FOR DELTA RESONANCE
*            LA = 0 FOR N*(1440) RESONANCE
*            LA = 2 FRO N*(1535) RESONANCE
* DATE    : JAN.29,1990
        REAL FUNCTION XNPI(I1,I2,LA,XMAX)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      SAVE   
        AVMASS=0.5*(AMN+AMP)
        AVPI=(2.*AP2+AP1)/3.
* 1. DETERMINE THE MOMENTUM COMPONENT OF DELTA IN THE LAB. FRAME
        E10=SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
        E20=SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
        P1=P(1,I1)+P(1,I2)
        P2=P(2,I1)+P(2,I2)
        P3=P(3,I1)+P(3,I2)
* 2. DETERMINE THE MASS OF DELTA BY USING OF THE REACTION KINEMATICS
        DM=SQRT((E10+E20)**2-P1**2-P2**2-P3**2)
        IF(DM.LE.1.1) THEN
        XNPI=1.e-09
        RETURN
        ENDIF
* 3. DETERMINE THE PION+NUCLEON CROSS SECTION ACCORDING TO THE
*    BREIT-WIGNER FORMULA IN UNIT OF FM**2
        IF(LA.EQ.1)THEN
        GAM=WIDTH(DM)
        F1=0.25*GAM**2/(0.25*GAM**2+(DM-1.232)**2)
        PDELT2=0.051622
        GO TO 10
       ENDIF
       IF(LA.EQ.0)THEN
        GAM=W1440(DM)
        F1=0.25*GAM**2/(0.25*GAM**2+(DM-1.440)**2)
        PDELT2=0.157897
       GO TO 10
        ENDIF
       IF(LA.EQ.2)THEN
        GAM=W1535(DM)
        F1=0.25*GAM**2/(0.25*GAM**2+(DM-1.535)**2)
        PDELT2=0.2181
        ENDIF
10      PSTAR2=((DM**2-AVMASS**2+AVPI**2)/(2.*DM))**2-AVPI**2
        IF(PSTAR2.LE.0.)THEN
        XNPI=1.e-09
        ELSE
* give the cross section in unit of fm**2
        XNPI=F1*(PDELT2/PSTAR2)*XMAX/10.
        ENDIF
        RETURN
        END
*------------------------------------------------------------------------------
*****************************************
        REAL FUNCTION SIGMA(SRT,ID,IOI,IOF)
*PURPOSE : THIS IS THE PROGRAM TO CALCULATE THE ISOSPIN DECOMPOSED CROSS
*       SECTION BY USING OF B.J.VerWEST AND R.A.ARNDT'S PARAMETERIZATION
*REFERENCE: PHYS. REV. C25(1982)1979
*QUANTITIES: IOI -- INITIAL ISOSPIN OF THE TWO NUCLEON SYSTEM
*            IOF -- FINAL   ISOSPIN -------------------------
*            ID -- =1 FOR DELTA RESORANCE
*                  =2 FOR N*    RESORANCE
*DATE : MAY 15,1990
*****************************************
        PARAMETER (AMU=0.9383,AMP=0.1384,PI=3.1415926,HC=0.19733)
      SAVE   
        IF(ID.EQ.1)THEN
        AMASS0=1.22
        T0 =0.12
        ELSE
        AMASS0=1.43
        T0 =0.2
        ENDIF
        IF((IOI.EQ.1).AND.(IOF.EQ.1))THEN
        ALFA=3.772
        BETA=1.262
        AM0=1.188
        T=0.09902
        ENDIF
        IF((IOI.EQ.1).AND.(IOF.EQ.0))THEN
        ALFA=15.28
        BETA=0.
        AM0=1.245
        T=0.1374
        ENDIF
        IF((IOI.EQ.0).AND.(IOF.EQ.1))THEN
        ALFA=146.3
        BETA=0.
        AM0=1.472
        T=0.02649
        ENDIF
        ZPLUS=(SRT-AMU-AMASS0)*2./T0
        ZMINUS=(AMU+AMP-AMASS0)*2./T0
        deln=ATAN(ZPLUS)-ATAN(ZMINUS)
       if(deln.eq.0)deln=1.E-06
        AMASS=AMASS0+(T0/4.)*ALOG((1.+ZPLUS**2)/(1.+ZMINUS**2))
     1  /deln
        S=SRT**2
        P2=S/4.-AMU**2
        S0=(AMU+AM0)**2
        P02=S0/4.-AMU**2
        P0=SQRT(P02)
        PR2=(S-(AMU-AMASS)**2)*(S-(AMU+AMASS)**2)/(4.*S)
        IF(PR2.GT.1.E-06)THEN
        PR=SQRT(PR2)
        ELSE
        PR=0.
        SIGMA=1.E-06
        RETURN
        ENDIF
        SS=AMASS**2
        Q2=(SS-(AMU-AMP)**2)*(SS-(AMU+AMP)**2)/(4.*SS)
        IF(Q2.GT.1.E-06)THEN
        Q=SQRT(Q2)
        ELSE
        Q=0.
        SIGMA=1.E-06
        RETURN
        ENDIF
        SS0=AM0**2
        Q02=(SS0-(AMU-AMP)**2)*(SS0-(AMU+AMP)**2)/(4.*SS0)
        Q0=SQRT(Q02)
        SIGMA=PI*(HC)**2/(2.*P2)*ALFA*(PR/P0)**BETA*AM0**2*T**2
     1  *(Q/Q0)**3/((SS-AM0**2)**2+AM0**2*T**2)
        SIGMA=SIGMA*10.
       IF(SIGMA.EQ.0)SIGMA=1.E-06
        RETURN
        END

*****************************
        REAL FUNCTION DENOM(SRT,CON)
* NOTE: CON=1 FOR DELTA RESONANCE, CON=2 FOR N*(1440) RESONANCE
*       con=-1 for N*(1535)
* PURPOSE : CALCULATE THE INTEGRAL IN THE DETAILED BALANCE
*
* DATE : NOV. 15, 1991
*******************************
        PARAMETER (AP1=0.13496,
     1  AP2=0.13957,PI=3.1415926,AVMASS=0.9383)
      SAVE   
        AVPI=(AP1+2.*AP2)/3.
        AM0=1.232
        AMN=AVMASS
        AMP=AVPI
        AMAX=SRT-AVMASS
        AMIN=AVMASS+AVPI
        NMAX=200
        DMASS=(AMAX-AMIN)/FLOAT(NMAX)
        SUM=0.
        DO 10 I=1,NMAX+1
        DM=AMIN+FLOAT(I-1)*DMASS
        IF(CON.EQ.1.)THEN
        Q2=((DM**2-AMN**2+AMP**2)/(2.*DM))**2-AMP**2
           IF(Q2.GT.0.)THEN
           Q=SQRT(Q2)
           ELSE
           Q=1.E-06
           ENDIF
        TQ=0.47*(Q**3)/(AMP**2*(1.+0.6*(Q/AMP)**2))
        ELSE if(con.eq.2)then
        TQ=0.2
        AM0=1.44
       else if(con.eq.-1.)then
       tq=0.1
       am0=1.535
        ENDIF
        A1=4.*TQ*AM0**2/(AM0**2*TQ**2+(DM**2-AM0**2)**2)
        S=SRT**2
        P0=(S+DM**2-AMN**2)**2/(4.*S)-DM**2
        IF(P0.LE.0.)THEN
        P1=1.E-06
        ELSE
        P1=SQRT(P0)
        ENDIF
        F=DM*A1*P1
        IF((I.EQ.1).OR.(I.EQ.(NMAX+1)))THEN
        SUM=SUM+F*0.5
        ELSE
        SUM=SUM+F
        ENDIF
10      CONTINUE
        DENOM=SUM*DMASS/(2.*PI)
        RETURN
        END
**********************************
* subroutine : ang.FOR
* PURPOSE : Calculate the angular distribution of Delta production process 
* DATE    : Nov. 19, 1992
* REFERENCE: G. WOLF ET. AL., NUCL. PHYS. A517 (1990) 615
* Note: this function applies when srt is larger than 2.14 GeV,
* for less energetic reactions, we assume the angular distribution
* is isotropic.
***********************************
       real function ang(srt,iseed)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
      ISEED=ISEED 
c        if(srt.le.2.14)then
c       b1s=0.5
c       b2s=0.
c      endif
      if((srt.gt.2.14).and.(srt.le.2.4))then
       b1s=29.03-23.75*srt+4.865*srt**2
         b2s=-30.33+25.53*srt-5.301*srt**2
      endif
      if(srt.gt.2.4)then
       b1s=0.06
         b2s=0.4
      endif
        x=RANART(NSEED)
       p=b1s/b2s
       q=(2.*x-1.)*(b1s+b2s)/b2s
       IF((-q/2.+sqrt((q/2.)**2+(p/3.)**3)).GE.0.)THEN
       ang1=(-q/2.+sqrt((q/2.)**2+(p/3.)**3))**(1./3.)
       ELSE
       ang1=-(q/2.-sqrt((q/2.)**2+(p/3.)**3))**(1./3.)
       ENDIF
       IF((-q/2.-sqrt((q/2.)**2+(p/3.)**3).GE.0.))THEN
       ang2=(-q/2.-sqrt((q/2.)**2+(p/3.)**3))**(1./3.)
       ELSE
       ang2=-(q/2.+sqrt((q/2.)**2+(p/3.)**3))**(1./3.)
       ENDIF
       ANG=ANG1+ANG2
       return
       end
*--------------------------------------------------------------------------
*****subprogram * kaon production from pi+B collisions *******************
      real function PNLKA(srt)
      SAVE   
* units: fm**2
***********************************C
      ala=1.116
      aka=0.498
      ana=0.939
      t1=ala+aka      
      if(srt.le.t1) THEN
      Pnlka=0
      Else
      IF(SRT.LT.1.7)sbbk=(0.9/0.091)*(SRT-T1)
      IF(SRT.GE.1.7)sbbk=0.09/(SRT-1.6)
      Pnlka=0.25*sbbk
* give the cross section in units of fm**2
       pnlka=pnlka/10.
      endif     
      return
      end
*-------------------------------------------------------------------------
*****subprogram * kaon production from pi+B collisions *******************
      real function PNSKA(srt)
      SAVE   
***********************************
       if(srt.gt.3.0)then
       pnska=0
       return
       endif
      ala=1.116
      aka=0.498
      ana=0.939
      asa=1.197
      t1=asa+aka      
      if(srt.le.t1) THEN
      Pnska=0
       return
      Endif
      IF(SRT.LT.1.9)SBB1=(0.7/0.218)*(SRT-T1)
      IF(SRT.GE.1.9)SBB1=0.14/(SRT-1.7)
      sbb2=0.
       if(srt.gT.1.682)sbb2=0.5*(1.-0.75*(srt-1.682))
       pnska=0.25*(sbb1+sbb2)
* give the cross section in fm**2
       pnska=pnska/10.
      return
      end

********************************
*
*       Kaon momentum distribution in baryon-baryon-->N lamda K process
*
*       NOTE: dsima/dp is prototional to (1-p/p_max)(p/p_max)^2
*              we use rejection method to generate kaon momentum
*
*       Variables: Fkaon = F(p)/F_max
*                 srt   = cms energy of the colliding pair, 
*                          used to calculate the P_max
*       Date: Feb. 8, 1994
*
*       Reference: C. M. Ko et al.  
******************************** 
       Real function fkaon(p,pmax)
      SAVE   
       fmax=0.148
       if(pmax.eq.0.)pmax=0.000001
       fkaon=(1.-p/pmax)*(p/pmax)**2
       if(fkaon.gt.fmax)fkaon=fmax
       fkaon=fkaon/fmax
       return
       end

*************************
* cross section for N*(1535) production in ND OR NN* collisions
* VARIABLES:
* LB1,LB2 ARE THE LABLES OF THE TWO COLLIDING PARTICLES
* SRT IS THE CMS ENERGY
* X1535 IS THE N*(1535) PRODUCTION CROSS SECTION
* NOTE THAT THE N*(1535) PRODUCTION CROSS SECTION IS 2 TIMES THE ETA 
* PRODUCTION CROSS SECTION
* DATE: MAY 18, 1994
* ***********************
       Subroutine M1535(LB1,LB2,SRT,X1535)
      SAVE   
       S0=2.424
       x1535=0.
       IF(SRT.LE.S0)RETURN
       SIGMA=2.*0.102*(SRT-S0)/(0.058+(SRT-S0)**2)
* I N*(1535) PRODUCTION IN NUCLEON-DELTA COLLISIONS
*(1) nD(++)->pN*(+)(1535), pD(-)->nN*(0)(1535),pD(+)-->N*(+)p
cbz11/25/98
c       IF((LB1*LB2.EQ.18).OR.(LB1*LB2.EQ.6).
c     1  or.(lb1*lb2).eq.8)then
       IF((LB1*LB2.EQ.18.AND.(LB1.EQ.2.OR.LB2.EQ.2)).OR.
     &     (LB1*LB2.EQ.6.AND.(LB1.EQ.1.OR.LB2.EQ.1)).or.
     &     (lb1*lb2.eq.8.AND.(LB1.EQ.1.OR.LB2.EQ.1)))then
cbz11/25/98end
       X1535=SIGMA
       return
       ENDIF
*(2) pD(0)->pN*(0)(1535),pD(0)->nN*(+)(1535)
       IF(LB1*LB2.EQ.7)THEN
       X1535=3.*SIGMA
       RETURN
       ENDIF 
* II N*(1535) PRODUCTION IN N*(1440)+NUCLEON REACTIONS
*(3) N*(+)(1440)p->N*(0+)(1535)p, N*(0)(1440)n->N*(0)(1535)
cbz11/25/98
c       IF((LB1*LB2.EQ.11).OR.(LB1*LB2.EQ.20))THEN
       IF((LB1*LB2.EQ.11).OR.
     &     (LB1*LB2.EQ.20.AND.(LB1.EQ.2.OR.LB2.EQ.2)))THEN
cbz11/25/98end
       X1535=SIGMA
       RETURN
       ENDIF
*(4) N*(0)(1440)p->N*(0+) or N*(+)(1440)n->N*(0+)(1535)
cbz11/25/98
c       IF((LB1*LB2.EQ.10).OR.(LB1*LB2.EQ.22))X1535=3.*SIGMA
       IF((LB1*LB2.EQ.10.AND.(LB1.EQ.1.OR.LB2.EQ.1)).OR.
     &     (LB1*LB2.EQ.22.AND.(LB1.EQ.2.OR.LB2.EQ.2)))
     &     X1535=3.*SIGMA
cbz11/25/98end
       RETURN
       END
*************************
* cross section for N*(1535) production in NN collisions
* VARIABLES:
* LB1,LB2 ARE THE LABLES OF THE TWO COLLIDING PARTICLES
* SRT IS THE CMS ENERGY
* X1535 IS THE N*(1535) PRODUCTION CROSS SECTION
* NOTE THAT THE N*(1535) PRODUCTION CROSS SECTION IS 2 TIMES THE ETA 
* PRODUCTION CROSS SECTION
* DATE: MAY 18, 1994
* ***********************
       Subroutine N1535(LB1,LB2,SRT,X1535)
      SAVE   
       S0=2.424
       x1535=0.
       IF(SRT.LE.S0)RETURN
       SIGMA=2.*0.102*(SRT-S0)/(0.058+(SRT-S0)**2)
* I N*(1535) PRODUCTION IN NUCLEON-NUCLEON COLLISIONS
*(1) pp->pN*(+)(1535), nn->nN*(0)(1535)
cbdbg11/25/98
c       IF((LB1*LB2.EQ.1).OR.(LB1*LB2.EQ.4))then
       IF((LB1*LB2.EQ.1).OR.
     &     (LB1.EQ.2.AND.LB2.EQ.2))then
cbz11/25/98end
       X1535=SIGMA
       return
       endif
*(2) pn->pN*(0)(1535),pn->nN*(+)(1535)
       IF(LB1*LB2.EQ.2)then
       X1535=3.*SIGMA
       return
       endif 
* III N*(1535) PRODUCTION IN DELTA+DELTA REACTIONS
* (5) D(++)+D(0), D(+)+D(+),D(+)+D(-),D(0)+D(0)
cbz11/25/98
c       IF((LB1*LB2.EQ.63).OR.(LB1*LB2.EQ.64).OR.(LB1*LB2.EQ.48).
c     1  OR.(LB1*LB2.EQ.49))then
       IF((LB1*LB2.EQ.63.AND.(LB1.EQ.7.OR.LB2.EQ.7)).OR.
     &     (LB1*LB2.EQ.64.AND.(LB1.EQ.8.OR.LB2.EQ.8)).OR.
     &     (LB1*LB2.EQ.48.AND.(LB1.EQ.6.OR.LB2.EQ.6)).OR.
     &     (LB1*LB2.EQ.49.AND.(LB1.EQ.7.OR.LB2.EQ.7)))then
cbz11/25/98end
       X1535=SIGMA
       return
       endif
* (6) D(++)+D(-),D(+)+D(0)
cbz11/25/98
c       IF((LB1*LB2.EQ.54).OR.(LB1*LB2.EQ.56))then
       IF((LB1*LB2.EQ.54.AND.(LB1.EQ.6.OR.LB2.EQ.6)).OR.
     &     (LB1*LB2.EQ.56.AND.(LB1.EQ.7.OR.LB2.EQ.7)))then
cbz11/25/98end
       X1535=3.*SIGMA
       return
       endif
* IV N*(1535) PRODUCTION IN N*(1440)+N*(1440) REACTIONS
cbz11/25/98
c       IF((LB1*LB2.EQ.100).OR.(LB1*LB2.EQ.11*11))X1535=SIGMA
       IF((LB1.EQ.10.AND.LB2.EQ.10).OR.
     &     (LB1.EQ.11.AND.LB2.EQ.11))X1535=SIGMA
c       IF(LB1*LB2.EQ.110)X1535=3.*SIGMA
       IF(LB1*LB2.EQ.110.AND.(LB1.EQ.10.OR.LB2.EQ.10))X1535=3.*SIGMA
cbdbg11/25/98end
       RETURN
       END
************************************       
* FUNCTION WA1(DMASS) GIVES THE A1 DECAY WIDTH

        subroutine WIDA1(DMASS,rhomp,wa1,iseed)
      SAVE   
c
        PIMASS=0.137265
        coupa = 14.8
c
       RHOMAX = DMASS-PIMASS-0.02
       IF(RHOMAX.LE.0)then
         rhomp=0.
c   !! no decay
         wa1=-10.
        endif
        icount = 0
711       rhomp=RHOMAS(RHOMAX,ISEED)
      icount=icount+1
      if(dmass.le.(pimass+rhomp)) then
       if(icount.le.100) then
        goto 711
       else
         rhomp=0.
c   !! no decay
         wa1=-10.
        return
       endif
      endif
      qqp2=(dmass**2-(rhomp+pimass)**2)*(dmass**2-(rhomp-pimass)**2)
      qqp=sqrt(qqp2)/(2.0*dmass)
      epi=sqrt(pimass**2+qqp**2)
      erho=sqrt(rhomp**2+qqp**2)
      epirho=2.0*(epi*erho+qqp**2)**2+rhomp**2*epi**2
      wa1=coupa**2*qqp*epirho/(24.0*3.1416*dmass**2)
       return
       end
************************************       
* FUNCTION W1535(DMASS) GIVES THE N*(1535) DECAY WIDTH 
c     FOR A GIVEN N*(1535) MASS
* HERE THE FORMULA GIVEN BY KITAZOE IS USED
        REAL FUNCTION W1535(DMASS)
      SAVE   
        AVMASS=0.938868
        PIMASS=0.137265
           AUX = 0.25*(DMASS**2-AVMASS**2-PIMASS**2)**2
     &           -(AVMASS*PIMASS)**2
            IF (AUX .GT. 0.) THEN
              QAVAIL = SQRT(AUX / DMASS**2)
            ELSE
              QAVAIL = 1.E-06
            END IF
            W1535 = 0.15* QAVAIL/0.467
c       W1535=0.15
        RETURN
        END
************************************       
* FUNCTION W1440(DMASS) GIVES THE N*(1440) DECAY WIDTH 
c     FOR A GIVEN N*(1535) MASS
* HERE THE FORMULA GIVEN BY KITAZOE IS USED
        REAL FUNCTION W1440(DMASS)
      SAVE   
        AVMASS=0.938868
        PIMASS=0.137265
           AUX = 0.25*(DMASS**2-AVMASS**2-PIMASS**2)**2
     &           -(AVMASS*PIMASS)**2
            IF (AUX .GT. 0.) THEN
              QAVAIL = SQRT(AUX)/DMASS
            ELSE
              QAVAIL = 1.E-06
            END IF
c              w1440=0.2 
           W1440 = 0.2* (QAVAIL/0.397)**3
        RETURN
        END
****************
* PURPOSE : CALCULATE THE PION(ETA)+NUCLEON CROSS SECTION 
*           ACCORDING TO THE BREIT-WIGNER FORMULA, 
*           NOTE THAT N*(1535) IS S_11
* VARIABLE : LA = 1 FOR PI+N
*            LA = 0 FOR ETA+N
* DATE    : MAY 16, 1994
****************
        REAL FUNCTION XN1535(I1,I2,LA)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,ETAM=0.5475,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      SAVE   
        AVMASS=0.5*(AMN+AMP)
        AVPI=(2.*AP2+AP1)/3.
* 1. DETERMINE THE MOMENTUM COMPONENT OF N*(1535) IN THE LAB. FRAME
        E10=SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
        E20=SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
        P1=P(1,I1)+P(1,I2)
        P2=P(2,I1)+P(2,I2)
        P3=P(3,I1)+P(3,I2)
* 2. DETERMINE THE MASS OF DELTA BY USING OF THE REACTION KINEMATICS
        DM=SQRT((E10+E20)**2-P1**2-P2**2-P3**2)
        IF(DM.LE.1.1) THEN
        XN1535=1.E-06
        RETURN
        ENDIF
* 3. DETERMINE THE PION(ETA)+NUCLEON->N*(1535) CROSS SECTION ACCORDING TO THE
*    BREIT-WIGNER FORMULA IN UNIT OF FM**2
        GAM=W1535(DM)
       GAM0=0.15
        F1=0.25*GAM0**2/(0.25*GAM**2+(DM-1.535)**2)
        IF(LA.EQ.1)THEN
       XMAX=11.3
        ELSE
       XMAX=74.
        ENDIF
        XN1535=F1*XMAX/10.
        RETURN
        END
***************************8
*FUNCTION FDE(DMASS) GIVES DELTA MASS DISTRIBUTION BY USING OF
*KITAZOE'S FORMULA
        REAL FUNCTION FDELTA(DMASS)
      SAVE   
        AMN=0.938869
        AVPI=0.13803333
        AM0=1.232
        FD=0.25*WIDTH(DMASS)**2/((DMASS-1.232)**2
     1  +0.25*WIDTH(DMASS)**2)
        FDELTA=FD
        RETURN
        END
* FUNCTION WIDTH(DMASS) GIVES THE DELTA DECAY WIDTH FOR A GIVEN DELTA MASS
* HERE THE FORMULA GIVEN BY KITAZOE IS USED
        REAL FUNCTION WIDTH(DMASS)
      SAVE   
        AVMASS=0.938868
        PIMASS=0.137265
           AUX = 0.25*(DMASS**2-AVMASS**2-PIMASS**2)**2
     &           -(AVMASS*PIMASS)**2
            IF (AUX .GT. 0.) THEN
              QAVAIL = SQRT(AUX / DMASS**2)
            ELSE
              QAVAIL = 1.E-06
            END IF
            WIDTH = 0.47 * QAVAIL**3 /
     &              (PIMASS**2 * (1.+0.6*(QAVAIL/PIMASS)**2))
c       width=0.115
        RETURN
        END
************************************       
        SUBROUTINE ddp2(SRT,ISEED,PX,PY,PZ,DM1,PNX,
     &  PNY,PNZ,DM2,PPX,PPY,PPZ,icou1)
* PURPOSE : CALCULATE MOMENTUM OF PARTICLES IN THE FINAL SATAT FROM
* THE PROCESS N+N--->D1+D2+PION
*       DATE : July 25, 1994
* Generate the masses and momentum for particles in the NN-->DDpi process
* for a given center of mass energy srt, the momenta are given in the center
* of mass of the NN
*****************************************
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
       icou1=0
       pi=3.1415926
        AMN=938.925/1000.
        AMP=137.265/1000.
* (1) GENGRATE THE MASS OF DELTA1 AND DELTA2 USING
       srt1=srt-amp-0.02
       ntrym=0
8       call Rmasdd(srt1,1.232,1.232,1.08,
     &  1.08,ISEED,1,dm1,dm2)
       ntrym=ntrym+1
* CONSTANTS FOR GENERATING THE LONGITUDINAL MOMENTUM 
* FOR ONE OF THE RESONANCES
       V=0.43
       W=-0.84
* (2) Generate the transverse momentum
*     OF DELTA1
* (2.1) estimate the maximum transverse momentum
       PTMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2
       if(ptmax2.le.0)go to 8
       PTMAX=SQRT(PTMAX2)*1./3.
7       PT=PTR(PTMAX,ISEED)       
* (3.1) THE MAXIMUM LONGITUDINAL MOMENTUM IS
       PZMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2-PT**2
       IF((PZMAX2.LT.0.).and.ntrym.le.100)then 
       go to 7
       else
       pzmax2=1.E-09
       endif
       PZMAX=SQRT(PZMAX2)
       XMAX=2.*PZMAX/SRT
* (3.2) THE GENERATED X IS
* THE DSTRIBUTION HAS A MAXIMUM AT X0=-V/(2*w), f(X0)=1.056
       ntryx=0
       fmax00=1.056
       x00=0.26
       if(abs(xmax).gt.0.26)then
       f00=fmax00
       else
       f00=1.+v*abs(xmax)+w*xmax**2
       endif
9       X=XMAX*(1.-2.*RANART(NSEED))
       ntryx=ntryx+1
       xratio=(1.+V*ABS(X)+W*X**2)/f00       
clin-8/17/00       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
* (3.5) THE PZ IS
       PZ=0.5*SRT*X
* The x and y components of the deltA1
       fai=2.*pi*RANART(NSEED)
       Px=pt*cos(fai)
       Py=pt*sin(fai)
* find the momentum of delta2 and pion
* the energy of the delta1
       ek=sqrt(dm1**2+PT**2+Pz**2)
* (1) Generate the momentum of the delta2 in the cms of delta2 and pion
*     the energy of the cms of DP
        eln=srt-ek
       IF(ELN.lE.0)then
       icou1=-1
       return
       endif
* beta and gamma of the cms of delta2+pion
       bx=-Px/eln
       by=-Py/eln
       bz=-Pz/eln
       ga=1./sqrt(1.-bx**2-by**2-bz**2)
* the momentum of delta2 and pion in their cms frame
       elnc=eln/ga 
       pn2=((elnc**2+dm2**2-amp**2)/(2.*elnc))**2-dm2**2
       if(pn2.le.0)then
       icou1=-1
       return
       endif
       pn=sqrt(pn2)

clin-10/25/02 get rid of argument usage mismatch in PTR():
        xptr=0.33*PN
c       PNT=PTR(0.33*PN,ISEED)
       PNT=PTR(xptr,ISEED)
clin-10/25/02-end

       fain=2.*pi*RANART(NSEED)
       pnx=pnT*cos(fain)
       pny=pnT*sin(fain)
       SIG=1
       IF(X.GT.0)SIG=-1
       pnz=SIG*SQRT(pn**2-PNT**2)
       en=sqrt(dm2**2+pnx**2+pny**2+pnz**2)
* (2) the momentum for the pion
       ppx=-pnx
       ppy=-pny
       ppz=-pnz
       ep=sqrt(amp**2+ppx**2+ppy**2+ppz**2)
* (3) for the delta2, LORENTZ-TRANSFORMATION INTO nn cms FRAME
        PBETA  = PnX*BX + PnY*By+ PnZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + En )
              Pnx = BX * TRANS0 + PnX
              Pny = BY * TRANS0 + PnY
              Pnz = BZ * TRANS0 + PnZ
* (4) for the pion, LORENTZ-TRANSFORMATION INTO nn cms FRAME
             if(ep.eq.0.)ep=1.E-09
              PBETA  = PPX*BX + PPY*By+ PPZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + EP )
              PPx = BX * TRANS0 + PPX
              PPy = BY * TRANS0 + PPY
              PPz = BZ * TRANS0 + PPZ
       return
       end
****************************************
        SUBROUTINE ddrho(SRT,ISEED,PX,PY,PZ,DM1,PNX,
     &  PNY,PNZ,DM2,PPX,PPY,PPZ,amp,icou1)
* PURPOSE : CALCULATE MOMENTUM OF PARTICLES IN THE FINAL SATAT FROM
* THE PROCESS N+N--->D1+D2+rho
*       DATE : Nov.5, 1994
* Generate the masses and momentum for particles in the NN-->DDrho process
* for a given center of mass energy srt, the momenta are given in the center
* of mass of the NN
*****************************************
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
       icou1=0
       pi=3.1415926
        AMN=938.925/1000.
        AMP=770./1000.
* (1) GENGRATE THE MASS OF DELTA1 AND DELTA2 USING
       srt1=srt-amp-0.02
       ntrym=0
8       call Rmasdd(srt1,1.232,1.232,1.08,
     &  1.08,ISEED,1,dm1,dm2)
       ntrym=ntrym+1
* GENERATE THE MASS FOR THE RHO
       RHOMAX = SRT-DM1-DM2-0.02
       IF(RHOMAX.LE.0.and.ntrym.le.20)go to 8
       AMP=RHOMAS(RHOMAX,ISEED)
* CONSTANTS FOR GENERATING THE LONGITUDINAL MOMENTUM 
* FOR ONE OF THE RESONANCES
       V=0.43
       W=-0.84
* (2) Generate the transverse momentum
*     OF DELTA1
* (2.1) estimate the maximum transverse momentum
       PTMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2
       PTMAX=SQRT(PTMAX2)*1./3.
7       PT=PTR(PTMAX,ISEED)
* (3) GENGRATE THE LONGITUDINAL MOMENTUM FOR DM1
*     USING THE GIVEN DISTRIBUTION
* (3.1) THE MAXIMUM LONGITUDINAL MOMENTUM IS
       PZMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2-PT**2
       IF((PZMAX2.LT.0.).and.ntrym.le.100)then 
       go to 7
       else
       pzmax2=1.E-06
       endif
       PZMAX=SQRT(PZMAX2)
       XMAX=2.*PZMAX/SRT
* (3.2) THE GENERATED X IS
* THE DSTRIBUTION HAS A MAXIMUM AT X0=-V/(2*w), f(X0)=1.056
       ntryx=0
       fmax00=1.056
       x00=0.26
       if(abs(xmax).gt.0.26)then
       f00=fmax00
       else
       f00=1.+v*abs(xmax)+w*xmax**2
       endif
9       X=XMAX*(1.-2.*RANART(NSEED))
       ntryx=ntryx+1
       xratio=(1.+V*ABS(X)+W*X**2)/f00       
clin-8/17/00       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
* (3.5) THE PZ IS
       PZ=0.5*SRT*X
* The x and y components of the delta1
       fai=2.*pi*RANART(NSEED)
       Px=pt*cos(fai)
       Py=pt*sin(fai)
* find the momentum of delta2 and rho
* the energy of the delta1
       ek=sqrt(dm1**2+PT**2+Pz**2)
* (1) Generate the momentum of the delta2 in the cms of delta2 and rho
*     the energy of the cms of Drho
        eln=srt-ek
       IF(ELN.lE.0)then
       icou1=-1
       return
       endif
* beta and gamma of the cms of delta2 and rho
       bx=-Px/eln
       by=-Py/eln
       bz=-Pz/eln
       ga=1./sqrt(1.-bx**2-by**2-bz**2)
       elnc=eln/ga
       pn2=((elnc**2+dm2**2-amp**2)/(2.*elnc))**2-dm2**2
       if(pn2.le.0)then
       icou1=-1
       return
       endif
       pn=sqrt(pn2)

clin-10/25/02 get rid of argument usage mismatch in PTR():
        xptr=0.33*PN
c       PNT=PTR(0.33*PN,ISEED)
       PNT=PTR(xptr,ISEED)
clin-10/25/02-end

       fain=2.*pi*RANART(NSEED)
       pnx=pnT*cos(fain)
       pny=pnT*sin(fain)
       SIG=1
       IF(X.GT.0)SIG=-1
       pnz=SIG*SQRT(pn**2-PNT**2)
       en=sqrt(dm2**2+pnx**2+pny**2+pnz**2)
* (2) the momentum for the rho
       ppx=-pnx
       ppy=-pny
       ppz=-pnz
       ep=sqrt(amp**2+ppx**2+ppy**2+ppz**2)
* (3) for the delta2, LORENTZ-TRANSFORMATION INTO nn cms FRAME
        PBETA  = PnX*BX + PnY*By+ PnZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + En )
              Pnx = BX * TRANS0 + PnX
              Pny = BY * TRANS0 + PnY
              Pnz = BZ * TRANS0 + PnZ
* (4) for the rho, LORENTZ-TRANSFORMATION INTO nn cms FRAME
             if(ep.eq.0.)ep=1.e-09
              PBETA  = PPX*BX + PPY*By+ PPZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + EP )
              PPx = BX * TRANS0 + PPX
              PPy = BY * TRANS0 + PPY
              PPz = BZ * TRANS0 + PPZ
       return
       end
****************************************
        SUBROUTINE pprho(SRT,ISEED,PX,PY,PZ,DM1,PNX,
     &  PNY,PNZ,DM2,PPX,PPY,PPZ,amp,icou1)
* PURPOSE : CALCULATE MOMENTUM OF PARTICLES IN THE FINAL SATAT FROM
* THE PROCESS N+N--->N1+N2+rho
*       DATE : Nov.5, 1994
* Generate the masses and momentum for particles in the NN--> process
* for a given center of mass energy srt, the momenta are given in the center
* of mass of the NN
*****************************************
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
        ntrym=0
       icou1=0
       pi=3.1415926
        AMN=938.925/1000.
*        AMP=770./1000.
       DM1=amn
       DM2=amn
* GENERATE THE MASS FOR THE RHO
       RHOMAX=SRT-DM1-DM2-0.02
       IF(RHOMAX.LE.0)THEN
       ICOU=-1
       RETURN
       ENDIF
       AMP=RHOMAS(RHOMAX,ISEED)
* CONSTANTS FOR GENERATING THE LONGITUDINAL MOMENTUM 
* FOR ONE OF THE nucleons
       V=0.43
       W=-0.84
* (2) Generate the transverse momentum
*     OF p1
* (2.1) estimate the maximum transverse momentum
       PTMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2
       PTMAX=SQRT(PTMAX2)*1./3.
7       PT=PTR(PTMAX,ISEED)
* (3) GENGRATE THE LONGITUDINAL MOMENTUM FOR DM1
*     USING THE GIVEN DISTRIBUTION
* (3.1) THE MAXIMUM LONGITUDINAL MOMENTUM IS
       PZMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2-PT**2
       NTRYM=NTRYM+1
       IF((PZMAX2.LT.0.).and.ntrym.le.100)then 
       go to 7
       else
       pzmax2=1.E-06
       endif
       PZMAX=SQRT(PZMAX2)
       XMAX=2.*PZMAX/SRT
* (3.2) THE GENERATED X IS
* THE DSTRIBUTION HAS A MAXIMUM AT X0=-V/(2*w), f(X0)=1.056
       ntryx=0
       fmax00=1.056
       x00=0.26
       if(abs(xmax).gt.0.26)then
       f00=fmax00
       else
       f00=1.+v*abs(xmax)+w*xmax**2
       endif
9       X=XMAX*(1.-2.*RANART(NSEED))
       ntryx=ntryx+1
       xratio=(1.+V*ABS(X)+W*X**2)/f00       
clin-8/17/00       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
* (3.5) THE PZ IS
       PZ=0.5*SRT*X
* The x and y components of the delta1
       fai=2.*pi*RANART(NSEED)
       Px=pt*cos(fai)
       Py=pt*sin(fai)
* find the momentum of delta2 and rho
* the energy of the delta1
       ek=sqrt(dm1**2+PT**2+Pz**2)
* (1) Generate the momentum of the delta2 in the cms of delta2 and rho
*     the energy of the cms of Drho
        eln=srt-ek
       IF(ELN.lE.0)then
       icou1=-1
       return
       endif
* beta and gamma of the cms of the two partciles
       bx=-Px/eln
       by=-Py/eln
       bz=-Pz/eln
       ga=1./sqrt(1.-bx**2-by**2-bz**2)
        elnc=eln/ga
       pn2=((elnc**2+dm2**2-amp**2)/(2.*elnc))**2-dm2**2
       if(pn2.le.0)then
       icou1=-1
       return
       endif
       pn=sqrt(pn2)

clin-10/25/02 get rid of argument usage mismatch in PTR():
        xptr=0.33*PN
c       PNT=PTR(0.33*PN,ISEED)
       PNT=PTR(xptr,ISEED)
clin-10/25/02-end

       fain=2.*pi*RANART(NSEED)
       pnx=pnT*cos(fain)
       pny=pnT*sin(fain)
       SIG=1
       IF(X.GT.0)SIG=-1
       pnz=SIG*SQRT(pn**2-PNT**2)
       en=sqrt(dm2**2+pnx**2+pny**2+pnz**2)
* (2) the momentum for the rho
       ppx=-pnx
       ppy=-pny
       ppz=-pnz
       ep=sqrt(amp**2+ppx**2+ppy**2+ppz**2)
* (3) for the delta2, LORENTZ-TRANSFORMATION INTO nn cms FRAME
        PBETA  = PnX*BX + PnY*By+ PnZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + En )
              Pnx = BX * TRANS0 + PnX
              Pny = BY * TRANS0 + PnY
              Pnz = BZ * TRANS0 + PnZ
* (4) for the rho, LORENTZ-TRANSFORMATION INTO nn cms FRAME
             if(ep.eq.0.)ep=1.e-09
              PBETA  = PPX*BX + PPY*By+ PPZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + EP )
              PPx = BX * TRANS0 + PPX
              PPy = BY * TRANS0 + PPY
              PPz = BZ * TRANS0 + PPZ
       return
       end
***************************8
****************************************
        SUBROUTINE ppomga(SRT,ISEED,PX,PY,PZ,DM1,PNX,
     &  PNY,PNZ,DM2,PPX,PPY,PPZ,icou1)
* PURPOSE : CALCULATE MOMENTUM OF PARTICLES IN THE FINAL SATAT FROM
* THE PROCESS N+N--->N1+N2+OMEGA
*       DATE : Nov.5, 1994
* Generate the masses and momentum for particles in the NN--> process
* for a given center of mass energy srt, the momenta are given in the center
* of mass of the NN
*****************************************
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
        ntrym=0
       icou1=0
       pi=3.1415926
        AMN=938.925/1000.
        AMP=782./1000.
       DM1=amn
       DM2=amn
* CONSTANTS FOR GENERATING THE LONGITUDINAL MOMENTUM 
* FOR ONE OF THE nucleons
       V=0.43
       W=-0.84
* (2) Generate the transverse momentum
*     OF p1
* (2.1) estimate the maximum transverse momentum
       PTMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2
       PTMAX=SQRT(PTMAX2)*1./3.
7       PT=PTR(PTMAX,ISEED)
* (3) GENGRATE THE LONGITUDINAL MOMENTUM FOR DM1
*     USING THE GIVEN DISTRIBUTION
* (3.1) THE MAXIMUM LONGITUDINAL MOMENTUM IS
       PZMAX2=(SRT**2-(DM1+DM2+AMP)**2)*
     1  (SRT**2-(DM1-AMP-DM2)**2)/4./SRT**2-PT**2
       NTRYM=NTRYM+1
       IF((PZMAX2.LT.0.).and.ntrym.le.100)then 
       go to 7
       else
       pzmax2=1.E-09
       endif
       PZMAX=SQRT(PZMAX2)
       XMAX=2.*PZMAX/SRT
* (3.2) THE GENERATED X IS
* THE DSTRIBUTION HAS A MAXIMUM AT X0=-V/(2*w), f(X0)=1.056
       ntryx=0
       fmax00=1.056
       x00=0.26
       if(abs(xmax).gt.0.26)then
       f00=fmax00
       else
       f00=1.+v*abs(xmax)+w*xmax**2
       endif
9       X=XMAX*(1.-2.*RANART(NSEED))
       ntryx=ntryx+1
       xratio=(1.+V*ABS(X)+W*X**2)/f00       
clin-8/17/00       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
       IF(xratio.LT.RANART(NSEED).and.ntryx.le.50)GO TO 9       
* (3.5) THE PZ IS
       PZ=0.5*SRT*X
* The x and y components of the delta1
       fai=2.*pi*RANART(NSEED)
       Px=pt*cos(fai)
       Py=pt*sin(fai)
* find the momentum of delta2 and rho
* the energy of the delta1
       ek=sqrt(dm1**2+PT**2+Pz**2)
* (1) Generate the momentum of the delta2 in the cms of delta2 and rho
*     the energy of the cms of Drho
        eln=srt-ek
       IF(ELN.lE.0)then
       icou1=-1
       return
       endif
       bx=-Px/eln
       by=-Py/eln
       bz=-Pz/eln
       ga=1./sqrt(1.-bx**2-by**2-bz**2)
       elnc=eln/ga
       pn2=((elnc**2+dm2**2-amp**2)/(2.*elnc))**2-dm2**2
       if(pn2.le.0)then
       icou1=-1
       return
       endif
       pn=sqrt(pn2)

clin-10/25/02 get rid of argument usage mismatch in PTR():
        xptr=0.33*PN
c       PNT=PTR(0.33*PN,ISEED)
       PNT=PTR(xptr,ISEED)
clin-10/25/02-end

       fain=2.*pi*RANART(NSEED)
       pnx=pnT*cos(fain)
       pny=pnT*sin(fain)
       SIG=1
       IF(X.GT.0)SIG=-1
       pnz=SIG*SQRT(pn**2-PNT**2)
       en=sqrt(dm2**2+pnx**2+pny**2+pnz**2)
* (2) the momentum for the rho
       ppx=-pnx
       ppy=-pny
       ppz=-pnz
       ep=sqrt(amp**2+ppx**2+ppy**2+ppz**2)
* (3) for the delta2, LORENTZ-TRANSFORMATION INTO nn cms FRAME
        PBETA  = PnX*BX + PnY*By+ PnZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + En )
              Pnx = BX * TRANS0 + PnX
              Pny = BY * TRANS0 + PnY
              Pnz = BZ * TRANS0 + PnZ
* (4) for the rho, LORENTZ-TRANSFORMATION INTO nn cms FRAME
             if(ep.eq.0.)ep=1.E-09
              PBETA  = PPX*BX + PPY*By+ PPZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + EP )
              PPx = BX * TRANS0 + PPX
              PPy = BY * TRANS0 + PPY
              PPz = BZ * TRANS0 + PPZ
       return
       end
***************************8
***************************8
*   DELTA MASS GENERATOR
       REAL FUNCTION RMASS(DMAX,ISEED)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
          ISEED=ISEED 
* THE MINIMUM MASS FOR DELTA
          DMIN = 1.078
* Delta(1232) production
          IF(DMAX.LT.1.232) THEN
          FM=FDELTA(DMAX)
          ELSE
          FM=1.
          ENDIF
          IF(FM.EQ.0.)FM=1.E-06
          NTRY1=0
10        DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
          NTRY1=NTRY1+1
          IF((RANART(NSEED) .GT. FDELTA(DM)/FM).AND.
     1    (NTRY1.LE.10)) GOTO 10
clin-2/26/03 sometimes Delta mass can reach very high values (e.g. 15.GeV),
c     thus violating the thresh of the collision which produces it 
c     and leads to large violation of energy conservation. 
c     To limit the above, limit the Delta mass below a certain value 
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.1.47) goto 10

       RMASS=DM
       RETURN
       END

*------------------------------------------------------------------
* THE Breit Wigner FORMULA
        REAL FUNCTION FRHO(DMASS)
      SAVE   
        AM0=0.77
       WID=0.153
        FD=0.25*wid**2/((DMASS-AM0)**2+0.25*WID**2)
        FRHO=FD
        RETURN
        END
***************************8
*   RHO MASS GENERATOR
       REAL FUNCTION RHOMAS(DMAX,ISEED)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
          ISEED=ISEED
* THE MINIMUM MASS FOR DELTA
          DMIN = 0.28
* RHO(770) production
          IF(DMAX.LT.0.77) THEN
          FM=FRHO(DMAX)
          ELSE
          FM=1.
          ENDIF
          IF(FM.EQ.0.)FM=1.E-06
          NTRY1=0
10        DM = RANART(NSEED) * (DMAX-DMIN) + DMIN
          NTRY1=NTRY1+1
          IF((RANART(NSEED) .GT. FRHO(DM)/FM).AND.
     1    (NTRY1.LE.10)) GOTO 10
clin-2/26/03 limit the rho mass below a certain value
c     (here taken as its central value + 2* B-W fullwidth):
          if(dm.gt.1.07) goto 10

       RHOMAS=DM
       RETURN
       END
******************************************
* for pp-->pp+2pi
c      real*4 function X2pi(srt)
      real function X2pi(srt)
*  This function contains the experimental 
c     total pp-pp+pi(+)pi(-) Xsections    *
*  srt    = DSQRT(s) in GeV                                                  *
*  xsec   = production cross section in mb                                   *
*  earray = EXPerimental table with proton momentum in GeV/c                 *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye)*
*                                                                            *
******************************************
c      real*4   xarray(15), earray(15)
      real   xarray(15), earray(15)
      SAVE   
      data earray /2.23,2.81,3.67,4.0,4.95,5.52,5.97,6.04,
     &6.6,6.9,7.87,8.11,10.01,16.0,19./
      data xarray /1.22,2.51,2.67,2.95,2.96,2.84,2.8,3.2,
     &2.7,3.0,2.54,2.46,2.4,1.66,1.5/

           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
       x2pi=0.000001
       if(srt.le.2.2)return
      plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
      if (plab .lt. earray(1)) then
        x2pi = xarray(1)
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,15
        if (earray(ie) .eq. plab) then
          x2pi= xarray(ie)
          return
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          X2pi = exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &    /(xmax-xmin) )
          return
        end if
 1001 continue
      return
        END
******************************************
* for pp-->pn+pi(+)pi(+)pi(-)
c      real*4 function X3pi(srt)
      real function X3pi(srt)
*  This function contains the experimental pp->pp+3pi cross sections          *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
c      real*4   xarray(12), earray(12)
      real   xarray(12), earray(12)
      SAVE   
      data xarray /0.02,0.4,1.15,1.60,2.19,2.85,2.30,
     &3.10,2.47,2.60,2.40,1.70/
      data earray /2.23,2.81,3.67,4.00,4.95,5.52,5.97,
     &6.04,6.60,6.90,10.01,19./

           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
       x3pi=1.E-06
       if(srt.le.2.3)return
      plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
      if (plab .lt. earray(1)) then
        x3pi = xarray(1)
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,12
        if (earray(ie) .eq. plab) then
          x3pi= xarray(ie)
          return
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          X3pi= exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &                                            /(xmax-xmin) )
          return
        end if
 1001 continue
      return
        END
******************************************
******************************************
* for pp-->pp+pi(+)pi(-)pi(0)
c      real*4 function X33pi(srt)
      real function X33pi(srt)
*  This function contains the experimental pp->pp+3pi cross sections          *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
c      real*4   xarray(12), earray(12)
      real   xarray(12), earray(12)
      SAVE   
      data xarray /0.02,0.22,0.74,1.10,1.76,1.84,2.20,
     &2.40,2.15,2.60,2.30,1.70/
      data earray /2.23,2.81,3.67,4.00,4.95,5.52,5.97,
     &6.04,6.60,6.90,10.01,19./

           pmass=0.9383 
       x33pi=1.E-06
       if(srt.le.2.3)return
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
      plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
      if (plab .lt. earray(1)) then
        x33pi = xarray(1)
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,12
        if (earray(ie) .eq. plab) then
          x33pi= xarray(ie)
          return
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          x33pi= exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &    /(xmax-xmin))
          return
        end if
 1001   continue
        return
        END
******************************************
c       REAL*4 FUNCTION X4pi(SRT)
      REAL FUNCTION X4pi(SRT)
      SAVE   
*       CROSS SECTION FOR NN-->DD+rho PROCESS
* *****************************
       akp=0.498
       ak0=0.498
       ana=0.94
       ada=1.232
       al=1.1157
       as=1.1197
       pmass=0.9383
       ES=SRT
       IF(ES.LE.4)THEN
       X4pi=0.
       ELSE
* cross section for two resonance pp-->DD+DN*+N*N*
       xpp2pi=4.*x2pi(es)
* cross section for pp-->pp+spi
       xpp3pi=3.*(x3pi(es)+x33pi(es))
* cross section for pp-->pD+ and nD++
       pps1=sigma(es,1,1,0)+0.5*sigma(es,1,1,1)
       pps2=1.5*sigma(es,1,1,1)
       ppsngl=pps1+pps2+s1535(es)
* CROSS SECTION FOR KAON PRODUCTION from the four channels
* for NLK channel
       xk1=0
       xk2=0
       xk3=0
       xk4=0
       t1nlk=ana+al+akp
       t2nlk=ana+al-akp
       if(es.le.t1nlk)go to 333
       pmnlk2=(es**2-t1nlk**2)*(es**2-t2nlk**2)/(4.*es**2)
       pmnlk=sqrt(pmnlk2)
       xk1=pplpk(es)
* for DLK channel
       t1dlk=ada+al+akp
       t2dlk=ada+al-akp
       if(es.le.t1dlk)go to 333
       pmdlk2=(es**2-t1dlk**2)*(es**2-t2dlk**2)/(4.*es**2)
       pmdlk=sqrt(pmdlk2)
       xk3=pplpk(es)
* for NSK channel
       t1nsk=ana+as+akp
       t2nsk=ana+as-akp
       if(es.le.t1nsk)go to 333
       pmnsk2=(es**2-t1nsk**2)*(es**2-t2nsk**2)/(4.*es**2)
       pmnsk=sqrt(pmnsk2)
       xk2=ppk1(es)+ppk0(es)
* for DSK channel
       t1DSk=aDa+aS+akp
       t2DSk=aDa+aS-akp
       if(es.le.t1dsk)go to 333
       pmDSk2=(es**2-t1DSk**2)*(es**2-t2DSk**2)/(4.*es**2)
       pmDSk=sqrt(pmDSk2)
       xk4=ppk1(es)+ppk0(es)
* THE TOTAL KAON+ AND KAON0 PRODUCTION CROSS SECTION IS THEN
333       XKAON=3.*(xk1+xk2+xk3+xk4)
* cross section for pp-->DD+rho
       x4pi=pp1(es)-ppsngl-xpp2pi-xpp3pi-XKAON
       if(x4pi.le.0)x4pi=1.E-06
       ENDIF
       RETURN
       END
******************************************
* for pp-->inelastic
c      real*4 function pp1(srt)
      real function pp1(srt)
      SAVE   
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
           pmass=0.9383 
       PP1=0.
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
      plab2=((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2
       IF(PLAB2.LE.0)RETURN
      plab=sqrt(PLAB2)
       pmin=0.968
       pmax=2080
      if ((plab .lt. pmin).or.(plab.gt.pmax)) then
        pp1 = 0.
        return
      end if
c* fit parameters
       a=30.9
       b=-28.9
       c=0.192
       d=-0.835
       an=-2.46
        pp1 = a+b*(plab**an)+c*(alog(plab))**2
       if(pp1.le.0)pp1=0.0
        return
        END
******************************************
* for pp-->elastic
c      real*4 function pp2(srt)
      real function pp2(srt)
      SAVE   
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
      plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
       pmin=2.
       pmax=2050
       if(plab.gt.pmax)then
       pp2=8.
       return
       endif
        if(plab .lt. pmin)then
        pp2 = 25.
        return
        end if
c* fit parameters
       a=11.2
       b=25.5
       c=0.151
       d=-1.62
       an=-1.12
        pp2 = a+b*(plab**an)+c*(alog(plab))**2+d*alog(plab)
       if(pp2.le.0)pp2=0
        return
        END

******************************************
* for pp-->total
c      real*4 function ppt(srt)
      real function ppt(srt)
      SAVE   
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
      plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
       pmin=3. 
       pmax=2100
      if ((plab .lt. pmin).or.(plab.gt.pmax)) then
        ppt = 55.
        return
      end if
c* fit parameters
       a=45.6
       b=219.0
       c=0.410
       d=-3.41
       an=-4.23
        ppt = a+b*(plab**an)+c*(alog(plab))**2+d*alog(plab)
       if(ppt.le.0)ppt=0.0
        return
        END

*************************
* cross section for N*(1535) production in PP collisions
* VARIABLES:
* LB1,LB2 ARE THE LABLES OF THE TWO COLLIDING PARTICLES
* SRT IS THE CMS ENERGY
* X1535 IS THE N*(1535) PRODUCTION CROSS SECTION
* NOTE THAT THE N*(1535) PRODUCTION CROSS SECTION IS 2 TIMES THE ETA 
* PRODUCTION CROSS SECTION
* DATE: Aug. 1 , 1994
* ********************************
       real function s1535(SRT)
      SAVE   
       S0=2.424
       s1535=0.
       IF(SRT.LE.S0)RETURN
       S1535=2.*0.102*(SRT-S0)/(0.058+(SRT-S0)**2)
       return
       end
****************************************
* generate a table for pt distribution for
       subroutine tablem
* THE PROCESS N+N--->N+N+PION
*       DATE : July 11, 1994
*****************************************
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      SAVE   
       ptmax=2.01
       anorm=ptdis(ptmax)
       do 10 L=0,200
       x=0.01*float(L+1)
       rr=ptdis(x)/anorm
       earray(l)=rr
       xarray(l)=x
10       continue
       RETURN
       end
*********************************
       real function ptdis(x)
      SAVE   
* NUCLEON TRANSVERSE MOMENTUM DISTRIBUTION AT HIGH ENERGIES
* DATE: Aug. 11, 1994
*********************************
       b=3.78
       c=0.47
       d=3.60
c       b=b*3
c       d=d*3
       ptdis=1./(2.*b)*(1.-exp(-b*x**2))-c/d*x*exp(-d*x)
     1     -c/D**2*(exp(-d*x)-1.)
       return
       end
*****************************
       subroutine ppxS(lb1,lb2,srt,ppsig,spprho,ipp)
* purpose: this subroutine gives the cross section for pion+pion 
*          elastic collision
* variables: 
*       input: lb1,lb2 and srt are the labels and srt for I1 and I2
*       output: ppsig: pp xsection
*               ipp: label for the pion+pion channel
*               Ipp=0 NOTHING HAPPEND 
*                  1 for Pi(+)+PI(+) DIRECT
*                   2     PI(+)+PI(0) FORMING RHO(+)
*                  3     PI(+)+PI(-) FORMING RHO(0)
*                   4     PI(0)+PI(O) DIRECT
*                  5     PI(0)+PI(-) FORMING RHO(-)
*                  6     PI(-)+PI(-) DIRECT
* reference: G.F. Bertsch, Phys. Rev. D37 (1988) 1202.
* date : Aug 29, 1994
*****************************
       parameter (amp=0.14,pi=3.1415926)
      SAVE   
       PPSIG=0.0

cbzdbg10/15/99
        spprho=0.0
cbzdbg10/15/99 end

       IPP=0
       IF(SRT.LE.0.3)RETURN
       q=sqrt((srt/2)**2-amp**2)
       esigma=5.8*amp
       tsigma=2.06*q
       erho=0.77
       trho=0.095*q*(q/amp/(1.+(q/erho)**2))**2
       esi=esigma-srt
       if(esi.eq.0)then
       d00=pi/2.
       go to 10
       endif
       d00=atan(tsigma/2./esi)
10       erh=erho-srt
       if(erh.eq.0.)then
       d11=pi/2.
       go to 20
       endif
       d11=atan(trho/2./erh)
20       d20=-0.12*q/amp
       s0=8.*pi*sin(d00)**2/q**2
       s1=8*pi*3*sin(d11)**2/q**2
       s2=8*pi*5*sin(d20)**2/q**2
c    !! GeV^-2 to mb
        s0=s0*0.197**2*10.
        s1=s1*0.197**2*10.
        s2=s2*0.197**2*10.
C       ppXS=s0/9.+s1/3.+s2*0.56
C       if(ppxs.le.0)ppxs=0.00001
       spprho=s1/2.
* (1) PI(+)+PI(+)
       IF(LB1.EQ.5.AND.LB2.EQ.5)THEN
       IPP=1
       PPSIG=S2
       RETURN
       ENDIF
* (2) PI(+)+PI(0)
       IF((LB1.EQ.5.AND.LB2.EQ.4).OR.(LB1.EQ.4.AND.LB2.EQ.5))THEN
       IPP=2
       PPSIG=S2/2.+S1/2.
       RETURN
       ENDIF
* (3) PI(+)+PI(-)
       IF((LB1.EQ.5.AND.LB2.EQ.3).OR.(LB1.EQ.3.AND.LB2.EQ.5))THEN
       IPP=3
       PPSIG=S2/6.+S1/2.+S0/3.
       RETURN
       ENDIF
* (4) PI(0)+PI(0)
       IF(LB1.EQ.4.AND.LB2.EQ.4)THEN
       IPP=4
       PPSIG=2*S2/3.+S0/3.
       RETURN
       ENDIF
* (5) PI(0)+PI(-)
       IF((LB1.EQ.4.AND.LB2.EQ.3).OR.(LB1.EQ.3.AND.LB2.EQ.4))THEN
       IPP=5
       PPSIG=S2/2.+S1/2.
       RETURN
       ENDIF
* (6) PI(-)+PI(-)
       IF(LB1.EQ.3.AND.LB2.EQ.3)THEN
       IPP=6
       PPSIG=S2
       ENDIF
       return
       end
**********************************
* elementary kaon production cross sections
*  from the CERN data book
*  date: Sept.2, 1994
*  for pp-->pLK+
c      real*4 function pplpk(srt)
      real function pplpk(srt)
      SAVE   
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
*   find the center of mass energy corresponding to the given pm as
*   if Lambda+N+K are produced
       pplpk=0.
        plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
       pmin=2.82
       pmax=25.0
       if(plab.gt.pmax)then
       pplpk=0.036
       return
       endif
        if(plab .lt. pmin)then
        pplpk = 0.
        return
        end if
c* fit parameters
       a=0.0654
       b=-3.16
       c=-0.0029
       an=-4.14
        pplpk = a+b*(plab**an)+c*(alog(plab))**2
       if(pplpk.le.0)pplpk=0
        return
        END

******************************************
* for pp-->pSigma+K0
c      real*4 function ppk0(srt)
      real function ppk0(srt)
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*                                                                             *
******************************************
c      real*4   xarray(7), earray(7)
      real   xarray(7), earray(7)
      SAVE   
      data xarray /0.030,0.025,0.025,0.026,0.02,0.014,0.06/
      data earray /3.67,4.95,5.52,6.05,6.92,7.87,10./

           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
       ppk0=0
       if(srt.le.2.63)return
       if(srt.gt.4.54)then
       ppk0=0.037
       return
       endif
        plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
        if (plab .lt. earray(1)) then
        ppk0 = xarray(1)
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,7
        if (earray(ie) .eq. plab) then
          ppk0 = xarray(ie)
          go to 10
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          ppk0 = exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &/(xmax-xmin) )
          go to 10
        end if
 1001 continue
10       continue
      return
        END
******************************************
* for pp-->pSigma0K+
c      real*4 function ppk1(srt)
      real function ppk1(srt)
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*                                                                             *
******************************************
c      real*4   xarray(7), earray(7)
      real   xarray(7), earray(7)
      SAVE   
      data xarray /0.013,0.025,0.016,0.012,0.017,0.029,0.025/
      data earray /3.67,4.95,5.52,5.97,6.05,6.92,7.87/

           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
       ppk1=0.
       if(srt.le.2.63)return
       if(srt.gt.4.08)then
       ppk1=0.025
       return
       endif
        plab=sqrt(((srt**2-2.*pmass**2)/(2.*pmass))**2-pmass**2)
        if (plab .lt. earray(1)) then
        ppk1 =xarray(1)
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,7
        if (earray(ie) .eq. plab) then
          ppk1 = xarray(ie)
          go to 10
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          ppk1 = exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &/(xmax-xmin) )
          go to 10
        end if
 1001 continue
10       continue
      return
        END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRPN(PX,PY,PZ,SRT,I1,I2,
     & IBLOCK,xkaon0,xkaon,Xphi,xphin)
*     PURPOSE:                                                         *
*           DEALING WITH PION+N-->L/S+KAON PROCESS AND PION PRODUCTION *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     7  PION+N-->L/S+KAON
*           iblock   - 77 pion+N-->Delta+pion
*           iblock   - 78 pion+N-->Delta+RHO
*           iblock   - 79 pion+N-->Delta+OMEGA
*           iblock   - 222 pion+N-->Phi 
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,APHI=1.020,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

      PX0=PX
      PY0=PY
      PZ0=PZ
      iblock=1
      x1=RANART(NSEED)
      ianti=0
      if(lb(i1).lt.0 .or. lb(i2).lt.0) ianti=1
      if(xkaon0/(xkaon+Xphi).ge.x1)then
* kaon production
*-----------------------------------------------------------------------
        IBLOCK=7
        if(ianti .eq. 1)iblock=-7
        NTAG=0
* RELABLE PARTICLES FOR THE PROCESS PION+n-->LAMBDA K OR SIGMA k
* DECIDE LAMBDA OR SIGMA PRODUCTION, AND TO CALCULATE THE NEW
* MOMENTA FOR PARTICLES IN THE FINAL STATE.
       KAONC=0
       IF(PNLKA(SRT)/(PNLKA(SRT)
     &       +PNSKA(SRT)).GT.RANART(NSEED))KAONC=1
       IF(E(I1).LE.0.2)THEN
           LB(I1)=23
           E(I1)=AKA
           IF(KAONC.EQ.1)THEN
              LB(I2)=14
              E(I2)=ALA
           ELSE
              LB(I2) = 15 + int(3 * RANART(NSEED))
              E(I2)=ASA       
           ENDIF
           if(ianti .eq. 1)then
              lb(i1) = 21
              lb(i2) = -lb(i2)
           endif
       ELSE
           LB(I2)=23
           E(I2)=AKA
           IF(KAONC.EQ.1)THEN
              LB(I1)=14
              E(I1)=ALA
           ELSE
              LB(I1) = 15 + int(3 * RANART(NSEED))
              E(I1)=ASA       
           ENDIF
           if(ianti .eq. 1)then
              lb(i2) = 21
              lb(i1) = -lb(i1)
           endif
       ENDIF
        EM1=E(I1)
        EM2=E(I2)
        go to 50
* to gererate the momentum for the kaon and L/S
      elseif(Xphi/(xkaon+Xphi).ge.x1)then
          iblock=222
         if(xphin/Xphi .ge. RANART(NSEED))then
          LB(I1)= 1+int(2*RANART(NSEED))
           E(I1)=AMN
         else
          LB(I1)= 6+int(4*RANART(NSEED))
           E(I1)=AM0
         endif
c  !! at present only baryon
         if(ianti .eq. 1)lb(i1)=-lb(i1)
          LB(I2)= 29
           E(I2)=APHI
        EM1=E(I1)
        EM2=E(I2)
       go to 50
         else
* CHECK WHAT KIND OF PION PRODUCTION PROCESS HAS HAPPENED
       IF(RANART(NSEED).LE.TWOPI(SRT)/
     &  (TWOPI(SRT)+THREPI(SRT)+FOURPI(SRT)))THEN
       iblock=77
       ELSE 
        IF(THREPI(SRT)/(THREPI(SRT)+FOURPI(SRT)).
     &  GT.RANART(NSEED))THEN
       IBLOCK=78
       ELSE
       IBLOCK=79
       ENDIF
       endif
       ntag=0
* pion production (Delta+pion/rho/omega in the final state)
* generate the mass of the delta resonance
       X2=RANART(NSEED)
* relable the particles
       if(iblock.eq.77)then
* GENERATE THE DELTA MASS
       dmax=srt-ap1-0.02
       dm=rmass(dmax,iseed)
* pion+baryon-->pion+delta
* Relable particles, I1 is assigned to the Delta and I2 is assigned to the
* meson
*(1) for pi(+)+p-->D(+)+P(+) OR D(++)+p(0)
       if( ((lb(i1).eq.1.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.1))
     &       .OR. ((lb(i1).eq.-1.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-1)) )then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=8
       e(i1)=dm
       lb(i2)=5
       e(i2)=ap1
       go to 40
       ELSE
       lb(i1)=9
       e(i1)=dm
       lb(i2)=4
        ipi = 4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=8
       e(i2)=dm
       lb(i1)=5
       e(i1)=ap1
       go to 40
       ELSE
       lb(i2)=9
       e(i2)=dm
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(2) for pi(-)+p-->D(0)+P(0) OR D(+)+p(-),or D(-)+p(+)
       if( ((lb(i1).eq.1.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.1))
     &        .OR. ((lb(i1).eq.-1.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-1)) )then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=6
       e(i1)=dm
       lb(i2)=5
       e(i2)=ap1
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i1)=7
       e(i1)=dm
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i1)=8
       e(i1)=dm
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=6
       e(i2)=dm
       lb(i1)=5
       e(i1)=ap1
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i2)=7
       e(i2)=dm
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i2)=8
       e(i2)=dm
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(3) for pi(+)+n-->D(+)+Pi(0) OR D(++)+p(-) or D(0)+pi(+)
       if( ((lb(i1).eq.2.and.lb(i2).eq.5).
     &   or.(lb(i1).eq.5.and.lb(i2).eq.2))
     & .OR. ((lb(i1).eq.-2.and.lb(i2).eq.3).
     &   or.(lb(i1).eq.3.and.lb(i2).eq.-2)) )then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=8
       e(i1)=dm
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i1)=7
       e(i1)=dm
       lb(i2)=5
       e(i2)=ap1
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i1)=9
       e(i1)=dm
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=8
       e(i2)=dm
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i2)=7
       e(i2)=dm
       lb(i1)=5
       e(i1)=ap1
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i2)=9
       e(i2)=dm
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(4) for pi(0)+p-->D(+)+Pi(0) OR D(++)+p(-) or D(0)+pi(+)
       if((iabs(lb(i1)).eq.1.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.1))then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=8
       e(i1)=dm
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i1)=7
       e(i1)=dm
       lb(i2)=5
       e(i2)=ap1
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i1)=9
       e(i1)=dm
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=8
       e(i2)=dm
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i2)=7
       e(i2)=dm
       lb(i1)=5
       e(i1)=ap1
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i2)=9
       e(i2)=dm
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif 
*(5) for pi(-)+n-->D(-)+P(0) OR D(0)+p(-)
       if( ((lb(i1).eq.2.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.2))
     &         .OR. ((lb(i1).eq.-2.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-2)) )then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=6
       e(i1)=dm
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ELSE
       lb(i1)=7
       e(i1)=dm
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=6
       e(i2)=dm
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ELSE
       lb(i2)=7
       e(i2)=dm
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(6) for pi(0)+n-->D(0)+P(0), D(-)+p(+) or D(+)+p(-)
       if((iabs(lb(i1)).eq.2.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.2))then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=7
       e(i1)=dm
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Endif
       IF(X2.LE.0.67.AND.X2.GT.0.33)THEN       
       lb(i1)=6
       e(i1)=dm
       lb(i2)=5
       e(i2)=ap1
       go to 40
       endif
       IF(X2.GT.0.67)THEN
       LB(I1)=8
       E(I1)=DM
       LB(I2)=3
       E(I2)=AP1
       GO TO 40
       ENDIF
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=7
       e(i2)=dm
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ENDIF
       IF(X2.LE.0.67.AND.X2.GT.0.33)THEN       
       lb(i2)=6
       e(i2)=dm
       lb(i1)=5
       e(i1)=ap1
       go to 40
       endif
       IF(X2.GT.0.67)THEN
       LB(I2)=8
       E(I2)=DM
       LB(I1)=3
       E(I1)=AP1
       GO TO 40
       ENDIF
              endif
       endif
                     ENDIF
       if(iblock.eq.78)then
       call Rmasdd(srt,1.232,0.77,1.08,
     &  0.28,ISEED,4,dm,ameson)
       arho=AMESON
* pion+baryon-->Rho+delta
*(1) for pi(+)+p-->D(+)+rho(+) OR D(++)+rho(0)
       if( ((lb(i1).eq.1.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.1))
     &        .OR. ((lb(i1).eq.-1.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-1)) )then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=8
       e(i1)=dm
       lb(i2)=27
       e(i2)=arho
       go to 40
       ELSE
       lb(i1)=9
       e(i1)=dm
       lb(i2)=26
       e(i2)=arho
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=8
       e(i2)=dm
       lb(i1)=27
       e(i1)=arho
       go to 40
       ELSE
       lb(i2)=9
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
       endif
              endif
       endif
*(2) for pi(-)+p-->D(+)+rho(-) OR D(0)+rho(0) or D(-)+rho(+)
       if( ((lb(i1).eq.1.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.1))
     &        .OR. ((lb(i1).eq.-1.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-1)) )then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=6
       e(i1)=dm
       lb(i2)=27
       e(i2)=arho
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i1)=7
       e(i1)=dm
       lb(i2)=26
       e(i2)=arho
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i1)=8
       e(i1)=dm
       lb(i2)=25
       e(i2)=arho
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=6
       e(i2)=dm
       lb(i1)=27
       e(i1)=arho
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i2)=7
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i2)=8
       e(i2)=dm
       lb(i1)=25
       e(i1)=arho
       go to 40
       endif
              endif
       endif
*(3) for pi(+)+n-->D(+)+rho(0) OR D(++)+rho(-) or D(0)+rho(+)
       if( ((lb(i1).eq.2.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.2))
     &       .OR.((lb(i1).eq.-2.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-2)) )then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=8
       e(i1)=dm
       lb(i2)=26
       e(i2)=arho
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i1)=7
       e(i1)=dm
       lb(i2)=27
       e(i2)=arho
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i1)=9
       e(i1)=dm
       lb(i2)=25
       e(i2)=arho
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=8
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i2)=7
       e(i2)=dm
       lb(i1)=27
       e(i1)=arho
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i2)=9
       e(i2)=dm
       lb(i1)=25
       e(i1)=arho
       go to 40
       endif
              endif
       endif
*(4) for pi(0)+p-->D(+)+rho(0) OR D(++)+rho(-) or D(0)+rho(+)
       if((iabs(lb(i1)).eq.1.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.1))then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=7
       e(i1)=dm
       lb(i2)=27
       e(i2)=arho
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i1)=8
       e(i1)=dm
       lb(i2)=26
       e(i2)=arho
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i1)=9
       e(i1)=dm
       lb(i2)=25
       e(i2)=arho
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=7
       e(i2)=dm
       lb(i1)=27
       e(i1)=arho
       go to 40
       ENDIF
       if(X2.gt.0.33.and.X2.le.0.67)then
       lb(i2)=8
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
       endif
       if(X2.gt.0.67)then
       lb(i2)=9
       e(i2)=dm
       lb(i1)=25
       e(i1)=arho
       go to 40
       endif
              endif
       endif 
*(5) for pi(-)+n-->D(-)+rho(0) OR D(0)+rho(-)
       if( ((lb(i1).eq.2.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.2))
     &        .OR. ((lb(i1).eq.-2.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-2)) )then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=6
       e(i1)=dm
       lb(i2)=26
       e(i2)=arho
       go to 40
       ELSE
       lb(i1)=7
       e(i1)=dm
       lb(i2)=25
       e(i2)=arho
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=6
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
       ELSE
       lb(i2)=7
       e(i2)=dm
       lb(i1)=25
       e(i1)=arho
       go to 40
       endif
              endif
       ENDIF
*(6) for pi(0)+n-->D(0)+rho(0), D(-)+rho(+) and D(+)+rho(-)
       if((iabs(lb(i1)).eq.2.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.2))then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       IF(X2.LE.0.33)THEN
       lb(i1)=7
       e(i1)=dm
       lb(i2)=26
       e(i2)=arho
       go to 40
       endif
       if(x2.gt.0.33.and.x2.le.0.67)then       
       lb(i1)=6
       e(i1)=dm
       lb(i2)=27
       e(i2)=arho
       go to 40
       endif
       if(x2.gt.0.67)then
       lb(i1)=8
       e(i1)=dm
       lb(i2)=25
       e(i2)=arho
       endif
              else
        ii = i2
       IF(X2.LE.0.33)THEN
       lb(i2)=7
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
       endif
       if(x2.le.0.67.and.x2.gt.0.33)then       
       lb(i2)=6
       e(i2)=dm
       lb(i1)=27
       e(i1)=arho
       go to 40
       endif
       if(x2.gt.0.67)then
       lb(i2)=8
       e(i2)=dm
       lb(i1)=25
       e(i1)=arho
       endif
              endif
       endif
                     Endif
       if(iblock.eq.79)then
       aomega=0.782
* GENERATE THE DELTA MASS
       dmax=srt-0.782-0.02
       dm=rmass(dmax,iseed)
* pion+baryon-->omega+delta
*(1) for pi(+)+p-->D(++)+omega(0)
       if( ((lb(i1).eq.1.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.1))
     &  .OR.((lb(i1).eq.-1.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-1)) )then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       lb(i1)=9
       e(i1)=dm
       lb(i2)=28
       e(i2)=aomega
       go to 40
              else
        ii = i2
       lb(i2)=9
       e(i2)=dm
       lb(i1)=28
       e(i1)=aomega
       go to 40
              endif
       endif
*(2) for pi(-)+p-->D(0)+omega(0) 
       if( ((lb(i1).eq.1.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.1))
     &        .OR. ((lb(i1).eq.-1.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-1)) )then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       lb(i1)=7
       e(i1)=dm
       lb(i2)=28
       e(i2)=aomega
       go to 40
              else
        ii = i2
       lb(i2)=7
       e(i2)=dm
       lb(i1)=28
       e(i1)=aomega
       go to 40
              endif
       endif
*(3) for pi(+)+n-->D(+)+omega(0) 
       if( ((lb(i1).eq.2.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.2))
     &       .OR. ((lb(i1).eq.-2.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-2)) )then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       lb(i1)=8
       e(i1)=dm
       lb(i2)=28
       e(i2)=aomega
       go to 40
              else
        ii = i2
       lb(i2)=8
       e(i2)=dm
       lb(i1)=28
       e(i1)=aomega
       go to 40
              endif
       endif
*(4) for pi(0)+p-->D(+)+omega(0) 
       if((iabs(lb(i1)).eq.1.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.1))then
              if(iabs(lb(i1)).eq.1)then
        ii = i1
       lb(i1)=8
       e(i1)=dm
       lb(i2)=28
       e(i2)=aomega
       go to 40
              else
        ii = i2
       lb(i2)=8
       e(i2)=dm
       lb(i1)=28
       e(i1)=aomega
       go to 40
              endif
       endif 
*(5) for pi(-)+n-->D(-)+omega(0) 
       if( ((lb(i1).eq.2.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.2))
     &        .OR. ((lb(i1).eq.-2.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-2)) )then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       lb(i1)=6
       e(i1)=dm
       lb(i2)=28
       e(i2)=aomega
       go to 40
              ELSE
        ii = i2
       lb(i2)=6
       e(i2)=dm
       lb(i1)=28
       e(i1)=aomega
              endif
       ENDIF
*(6) for pi(0)+n-->D(0)+omega(0) 
       if((iabs(lb(i1)).eq.2.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.2))then
              if(iabs(lb(i1)).eq.2)then
        ii = i1
       lb(i1)=7
       e(i1)=dm
       lb(i2)=28
       e(i2)=aomega
       go to 40
              else
        ii = i2
       lb(i2)=7
       e(i2)=dm
       lb(i1)=26
       e(i1)=arho
       go to 40
              endif
       endif
                     Endif
40       em1=e(i1)
       em2=e(i2)
       if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
         lb(ii) = -lb(ii)
           jj = i2
          if(ii .eq. i2)jj = i1
         if(iblock .eq. 77)then
          if(lb(jj).eq.3)then
           lb(jj) = 5
          elseif(lb(jj).eq.5)then
           lb(jj) = 3
          endif
         elseif(iblock .eq. 78)then
          if(lb(jj).eq.25)then
           lb(jj) = 27
          elseif(lb(jj).eq.27)then
           lb(jj) = 25
          endif
         endif
       endif
           endif
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
50          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=0.00000001
          PR=SQRT(PR2)/(2.*SRT)
* here we use the same transverse momentum distribution as for
* pp collisions, it might be necessary to use a different distribution

clin-10/25/02 get rid of argument usage mismatch in PTR():
          xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
         cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
*          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CREN(PX,PY,PZ,SRT,I1,I2,IBLOCK)
*     PURPOSE:                                                         *
*             DEALING WITH ETA+N-->L/S+KAON PROCESS                   *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     7  ETA+N-->L/S+KAON
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
        NTAG=0
        IBLOCK=7
        ianti=0
        if(lb(i1).lt.0 .or. lb(i2).lt.0)then
          ianti=1
          iblock=-7
        endif
* RELABLE PARTICLES FOR THE PROCESS eta+n-->LAMBDA K OR SIGMA k
* DECIDE LAMBDA OR SIGMA PRODUCTION, AND TO CALCULATE THE NEW
* MOMENTA FOR PARTICLES IN THE FINAL STATE.
       KAONC=0
       IF(PNLKA(SRT)/(PNLKA(SRT)
     & +PNSKA(SRT)).GT.RANART(NSEED))KAONC=1
       IF(E(I1).LE.0.6)THEN
       LB(I1)=23
       E(I1)=AKA
        IF(KAONC.EQ.1)THEN
       LB(I2)=14
       E(I2)=ALA
        ELSE
        LB(I2) = 15 + int(3 * RANART(NSEED))
       E(I2)=ASA       
        ENDIF
          if(ianti .eq. 1)then
            lb(i1)=21
            lb(i2)=-lb(i2)
          endif
       ELSE
       LB(I2)=23
       E(I2)=AKA
        IF(KAONC.EQ.1)THEN
       LB(I1)=14
       E(I1)=ALA
        ELSE
         LB(I1) = 15 + int(3 * RANART(NSEED))
       E(I1)=ASA       
        ENDIF
          if(ianti .eq. 1)then
            lb(i2)=21
            lb(i1)=-lb(i1)
          endif
       ENDIF
        EM1=E(I1)
        EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
        PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* FOR THE ISOTROPIC DISTRIBUTION THERE IS NO NEED TO ROTATE
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
c      SUBROUTINE Crdir(PX,PY,PZ,SRT,I1,I2)
      SUBROUTINE Crdir(PX,PY,PZ,SRT,I1,I2,IBLOCK)
*     PURPOSE:                                                         *
*             DEALING WITH pion+N-->pion+N PROCESS                   *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                    
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
        IBLOCK=999
        NTAG=0
        EM1=E(I1)
        EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
        PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)

clin-10/25/02 get rid of argument usage mismatch in PTR():
          xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
         cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
           T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE the momentum
      call rotate(px0,py0,pz0,px,py,pz)
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRPD(PX,PY,PZ,SRT,I1,I2,
     & IBLOCK,xkaon0,xkaon,Xphi,xphin)
*     PURPOSE:                                                         *
*     DEALING WITH PION+D(N*)-->PION +N OR 
*                                             L/S+KAON PROCESS         *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     7  PION+D(N*)-->L/S+KAON
*           iblock   - 80 pion+D(N*)-->pion+N
*           iblock   - 81 RHO+D(N*)-->PION+N
*           iblock   - 82 OMEGA+D(N*)-->PION+N
*                     222  PION+D --> PHI
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,APHI=1.020,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
        IBLOCK=1
       x1=RANART(NSEED)
        ianti=0
        if(lb(i1).lt.0 .or. lb(i2).lt.0)ianti=1
       if(xkaon0/(xkaon+Xphi).ge.x1)then
* kaon production
*-----------------------------------------------------------------------
        IBLOCK=7
        if(ianti .eq. 1)iblock=-7
        NTAG=0
* RELABLE PARTICLES FOR THE PROCESS PION+n-->LAMBDA K OR SIGMA k
* DECIDE LAMBDA OR SIGMA PRODUCTION, AND TO CALCULATE THE NEW
* MOMENTA FOR PARTICLES IN THE FINAL STATE.
       KAONC=0
       IF(PNLKA(SRT)/(PNLKA(SRT)
     &       +PNSKA(SRT)).GT.RANART(NSEED))KAONC=1
clin-8/17/00     & +PNSKA(SRT)).GT.RANART(NSEED))KAONC=1
       IF(E(I1).LE.0.2)THEN
           LB(I1)=23
           E(I1)=AKA
           IF(KAONC.EQ.1)THEN
              LB(I2)=14
              E(I2)=ALA
           ELSE
              LB(I2) = 15 + int(3 * RANART(NSEED))
              E(I2)=ASA       
           ENDIF
           if(ianti .eq. 1)then
              lb(i1)=21
              lb(i2)=-lb(i2)
           endif
       ELSE
           LB(I2)=23
           E(I2)=AKA
           IF(KAONC.EQ.1)THEN
              LB(I1)=14
              E(I1)=ALA
           ELSE
              LB(I1) = 15 + int(3 * RANART(NSEED))
              E(I1)=ASA       
           ENDIF
           if(ianti .eq. 1)then
              lb(i2)=21
              lb(i1)=-lb(i1)
           endif
       ENDIF
        EM1=E(I1)
        EM2=E(I2)
       go to 50
* to gererate the momentum for the kaon and L/S
c
c* Phi production
       elseif(Xphi/(xkaon+Xphi).ge.x1)then
          iblock=222
         if(xphin/Xphi .ge. RANART(NSEED))then
          LB(I1)= 1+int(2*RANART(NSEED))
           E(I1)=AMN
         else
          LB(I1)= 6+int(4*RANART(NSEED))
           E(I1)=AM0
         endif
c   !! at present only baryon
          if(ianti .eq. 1)lb(i1)=-lb(i1)
          LB(I2)= 29
           E(I2)=APHI
        EM1=E(I1)
        EM2=E(I2)
       go to 50
         else
* PION REABSORPTION HAS HAPPENED
       X2=RANART(NSEED)
       IBLOCK=80
       ntag=0
* Relable particles, I1 is assigned to the nucleon
* and I2 is assigned to the pion
* for the reverse of the following process
*(1) for D(+)+P(+)-->p+pion(+)
        if( ((lb(i1).eq.8.and.lb(i2).eq.5).
     &       or.(lb(i1).eq.5.and.lb(i2).eq.8))
     &       .OR.((lb(i1).eq.-8.and.lb(i2).eq.3).
     &       or.(lb(i1).eq.3.and.lb(i2).eq.-8)) )then
           if(iabs(lb(i1)).eq.8)then
              ii = i1
              lb(i1)=1
              e(i1)=amn
              lb(i2)=5
              e(i2)=ap1
              go to 40
           else
              ii = i2
              lb(i2)=1
              e(i2)=amn
              lb(i1)=5
              e(i1)=ap1
              go to 40
           endif
       endif
c
*(2) for D(0)+P(0)-->n+pi(0) or p+pi(-) 
       if((iabs(lb(i1)).eq.7.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.7))then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(3) for D(+)+Pi(0)-->pi(+)+n or pi(0)+p 
       if((iabs(lb(i1)).eq.8.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.8))then
              if(iabs(lb(i1)).eq.8)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(4) for D(-)+Pi(0)-->n+pi(-) 
       if((iabs(lb(i1)).eq.6.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.6))then
              if(iabs(lb(i1)).eq.6)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(5) for D(+)+Pi(-)-->pi(0)+n or pi(-)+p
       if( ((lb(i1).eq.8.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.8))
     &        .OR.((lb(i1).eq.-8.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-8)) )then
              if(iabs(lb(i1)).eq.8)then
        ii = i1
        IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ELSE
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
        IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ELSE
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(6) D(0)+P(+)-->n+pi(+) or p+pi(0)
       if( ((lb(i1).eq.7.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.7))
     &        .OR.((lb(i1).eq.-7.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-7)) )then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
         IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
         IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(7) for D(0)+Pi(-)-->n+pi(-) 
       if( ((lb(i1).eq.7.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.7))
     &        .OR.((lb(i1).eq.-7.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-7)) )then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(8) D(-)+P(+)-->n+pi(0) or p+pi(-)
       if( ((lb(i1).eq.6.and.lb(i2).eq.5)
     &      .or.(lb(i1).eq.5.and.lb(i2).eq.6))
     &   .OR.((lb(i1).eq.-6.and.lb(i2).eq.3).
     &      or.(lb(i1).eq.3.and.lb(i2).eq.-6)) )then
              if(iabs(lb(i1)).eq.6)then
         ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
         ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
c
*(9) D(++)+P(-)-->n+pi(+) or p+pi(0)
       if( ((lb(i1).eq.9.and.lb(i2).eq.3)
     &   .or.(lb(i1).eq.3.and.lb(i2).eq.9))
     &       .OR. ((lb(i1).eq.-9.and.lb(i2).eq.5)
     &   .or.(lb(i1).eq.5.and.lb(i2).eq.-9)) )then
              if(iabs(lb(i1)).eq.9)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(10) for D(++)+Pi(0)-->p+pi(+) 
       if((iabs(lb(i1)).eq.9.and.lb(i2).eq.4)
     &    .or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.9))then
           if(iabs(lb(i1)).eq.9)then
        ii = i1
       lb(i1)=1
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=1
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(11) for N*(1440)(+)or N*(1535)(+)+P(+)-->p+pion(+)
       if( ((lb(i1).eq.11.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.11).
     &  or.(lb(i1).eq.13.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.13))
     &        .OR.((lb(i1).eq.-11.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-11).
     &  or.(lb(i1).eq.-13.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-13)) )then
              if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
       lb(i1)=1
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=1
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
              endif
       endif
*(12) for N*(1440) or N*(1535)(0)+P(0)-->n+pi(0) or p+pi(-) 
       if((iabs(lb(i1)).eq.10.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.10).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.12).
     &  or.(lb(i2).eq.4.and.iabs(lb(i1)).eq.12))then
              if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(13) for N*(1440) or N*(1535)(+)+Pi(0)-->pi(+)+n or pi(0)+p 
       if((iabs(lb(i1)).eq.11.and.lb(i2).eq.4).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.11).
     &  or.(lb(i1).eq.4.and.iabs(lb(i2)).eq.13).
     &  or.(lb(i2).eq.4.and.iabs(lb(i1)).eq.13))then
              if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(14) for N*(1440) or N*(1535)(+)+Pi(-)-->pi(0)+n or pi(-)+p
       if( ((lb(i1).eq.11.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.11).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.13).
     &  or.(lb(i2).eq.3.and.lb(i1).eq.13))
     &        .OR.((lb(i1).eq.-11.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-11).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-13).
     &  or.(lb(i2).eq.5.and.lb(i1).eq.-13)) )then
       if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
         IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ELSE
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
         IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ELSE
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(15) N*(1440) or N*(1535)(0)+P(+)-->n+pi(+) or p+pi(0)
       if( ((lb(i1).eq.10.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.10).
     &  or.(lb(i1).eq.12.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.12))
     &        .OR.((lb(i1).eq.-10.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-10).
     &  or.(lb(i1).eq.-12.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.-12)) )then
       if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
        IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
        IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(16) for N*(1440) or N*(1535) (0)+Pi(-)-->n+pi(-) 
       if( ((lb(i1).eq.10.and.lb(i2).eq.3).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.10).
     &  or.(lb(i1).eq.3.and.lb(i2).eq.12).
     &  or.(lb(i1).eq.12.and.lb(i2).eq.3))
     &        .OR.((lb(i1).eq.-10.and.lb(i2).eq.5).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-10).
     &  or.(lb(i1).eq.5.and.lb(i2).eq.-12).
     &  or.(lb(i1).eq.-12.and.lb(i2).eq.5)) )then
           if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
40       em1=e(i1)
       em2=e(i2)
       if(ianti.eq.1 .and.  lb(i1).ge.1 .and. lb(i2).ge.1)then
         lb(ii) = -lb(ii)
           jj = i2
          if(ii .eq. i2)jj = i1
          if(lb(jj).eq.3)then
           lb(jj) = 5
          elseif(lb(jj).eq.5)then
           lb(jj) = 3
          endif
         endif
          endif
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
50          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.E-09
          PR=SQRT(PR2)/(2.*SRT)

clin-10/25/02 get rid of argument usage mismatch in PTR():
          xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
         cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
c         C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1 
* rotate the momentum
       call rotate(px0,py0,pz0,px,py,pz)
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRRD(PX,PY,PZ,SRT,I1,I2,
     & IBLOCK,xkaon0,xkaon,Xphi,xphin)
*     PURPOSE:                                                         *
*     DEALING WITH rho(omega)+N or D(N*)-->PION +N OR 
*                                             L/S+KAON PROCESS         *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     7  rho(omega)+N or D(N*)-->L/S+KAON
*           iblock   - 80 pion+D(N*)-->pion+N
*           iblock   - 81 RHO+D(N*)-->PION+N
*           iblock   - 82 OMEGA+D(N*)-->PION+N
*           iblock   - 222 pion+N-->Phi 
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER     (AKA=0.498,ALA=1.1157,ASA=1.1974,APHI=1.02)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
       IBLOCK=1
       ianti=0
       if(lb(i1).lt.0 .or. lb(i2).lt.0) ianti=1
       x1=RANART(NSEED)
       if(xkaon0/(xkaon+Xphi).ge.x1)then
* kaon production
*-----------------------------------------------------------------------
        IBLOCK=7
        if(ianti .eq. 1)iblock=-7
        NTAG=0
* RELABLE PARTICLES FOR THE PROCESS PION+n-->LAMBDA K OR SIGMA k
* DECIDE LAMBDA OR SIGMA PRODUCTION, AND TO CALCULATE THE NEW
* MOMENTA FOR PARTICLES IN THE FINAL STATE.
       KAONC=0
       IF(PNLKA(SRT)/(PNLKA(SRT)
     & +PNSKA(SRT)).GT.RANART(NSEED))KAONC=1
clin-8/17/00     & +PNSKA(SRT)).GT.RANART(NSEED))KAONC=1
       IF(E(I1).LE.0.92)THEN
       LB(I1)=23
       E(I1)=AKA
              IF(KAONC.EQ.1)THEN
       LB(I2)=14
       E(I2)=ALA
              ELSE
        LB(I2) = 15 + int(3 * RANART(NSEED))
       E(I2)=ASA       
              ENDIF
         if(ianti .eq. 1)then
          lb(i1) = 21
          lb(i2) = -lb(i2)
         endif
       ELSE
       LB(I2)=23
       E(I2)=AKA
              IF(KAONC.EQ.1)THEN
       LB(I1)=14
       E(I1)=ALA
              ELSE
         LB(I1) = 15 + int(3 * RANART(NSEED))
       E(I1)=ASA       
              ENDIF
         if(ianti .eq. 1)then
          lb(i2) = 21
          lb(i1) = -lb(i1)
         endif
       ENDIF
        EM1=E(I1)
        EM2=E(I2)
       go to 50
* to gererate the momentum for the kaon and L/S
c
c* Phi production
       elseif(Xphi/(xkaon+Xphi).ge.x1)then
          iblock=222
         if(xphin/Xphi .ge. RANART(NSEED))then
          LB(I1)= 1+int(2*RANART(NSEED))
           E(I1)=AMN
         else
          LB(I1)= 6+int(4*RANART(NSEED))
           E(I1)=AM0
         endif
c   !! at present only baryon
         if(ianti .eq. 1)lb(i1)=-lb(i1)
          LB(I2)= 29
           E(I2)=APHI
        EM1=E(I1)
        EM2=E(I2)
       go to 50
         else
* rho(omega) REABSORPTION HAS HAPPENED
       X2=RANART(NSEED)
       IBLOCK=81
       ntag=0
       if(lb(i1).eq.28.or.lb(i2).eq.28)go to 60
* we treat Rho reabsorption in the following 
* Relable particles, I1 is assigned to the Delta 
* and I2 is assigned to the meson
* for the reverse of the following process
*(1) for D(+)+rho(+)-->p+pion(+)
       if( ((lb(i1).eq.8.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.8))
     &        .OR. ((lb(i1).eq.-8.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-8)) )then
              if(iabs(lb(i1)).eq.8)then
        ii = i1
       lb(i1)=1
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=1
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
              endif
       endif
*(2) for D(0)+rho(0)-->n+pi(0) or p+pi(-) 
       if((iabs(lb(i1)).eq.7.and.lb(i2).eq.26).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.7))then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(3) for D(+)+rho(0)-->pi(+)+n or pi(0)+p 
       if((iabs(lb(i1)).eq.8.and.lb(i2).eq.26).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.8))then
              if(iabs(lb(i1)).eq.8)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(4) for D(-)+rho(0)-->n+pi(-) 
       if((iabs(lb(i1)).eq.6.and.lb(i2).eq.26).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.6))then
              if(iabs(lb(i1)).eq.6)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(5) for D(+)+rho(-)-->pi(0)+n or pi(-)+p
       if( ((lb(i1).eq.8.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.8))
     &        .OR. ((lb(i1).eq.-8.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-8)) )then
              if(iabs(lb(i1)).eq.8)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ELSE
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ELSE
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(6) D(0)+rho(+)-->n+pi(+) or p+pi(0)
       if( ((lb(i1).eq.7.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.7))
     &       .OR.((lb(i1).eq.-7.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-7)) )then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(7) for D(0)+rho(-)-->n+pi(-) 
       if( ((lb(i1).eq.7.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.7))
     &       .OR.((lb(i1).eq.-7.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-7)) )then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(8) D(-)+rho(+)-->n+pi(0) or p+pi(-)
       if( ((lb(i1).eq.6.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.6))
     &        .OR. ((lb(i1).eq.-6.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-6)) )then
              if(iabs(lb(i1)).eq.6)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(9) D(++)+rho(-)-->n+pi(+) or p+pi(0)
       if( ((lb(i1).eq.9.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.9))
     &        .OR.((lb(i1).eq.-9.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-9)) )then
              if(iabs(lb(i1)).eq.9)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(10) for D(++)+rho(0)-->p+pi(+) 
       if((iabs(lb(i1)).eq.9.and.lb(i2).eq.26).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.9))then
              if(iabs(lb(i1)).eq.9)then
        ii = i1
       lb(i1)=1
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=1
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(11) for N*(1440)(+)or N*(1535)(+)+rho(+)-->p+pion(+)
       if( ((lb(i1).eq.11.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.11).
     &  or.(lb(i1).eq.13.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.13))
     &        .OR. ((lb(i1).eq.-11.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-11).
     &  or.(lb(i1).eq.-13.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-13)) )then
              if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
       lb(i1)=1
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=1
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
              endif
       endif
*(12) for N*(1440) or N*(1535)(0)+rho(0)-->n+pi(0) or p+pi(-) 
       if((iabs(lb(i1)).eq.10.and.lb(i2).eq.26).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.10).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.12).
     &  or.(lb(i2).eq.26.and.iabs(lb(i1)).eq.12))then
              if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(13) for N*(1440) or N*(1535)(+)+rho(0)-->pi(+)+n or pi(0)+p 
       if((iabs(lb(i1)).eq.11.and.lb(i2).eq.26).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.11).
     &  or.(lb(i1).eq.26.and.iabs(lb(i2)).eq.13).
     &  or.(lb(i2).eq.26.and.iabs(lb(i1)).eq.13))then
              if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(14) for N*(1440) or N*(1535)(+)+rho(-)-->pi(0)+n or pi(-)+p
       if( ((lb(i1).eq.11.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.11).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.13).
     &  or.(lb(i2).eq.25.and.lb(i1).eq.13))
     &        .OR.((lb(i1).eq.-11.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-11).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-13).
     &  or.(lb(i2).eq.27.and.lb(i1).eq.-13)) )then
       if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       ELSE
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       ELSE
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(15) N*(1440) or N*(1535)(0)+rho(+)-->n+pi(+) or p+pi(0)
       if( ((lb(i1).eq.10.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.10).
     &  or.(lb(i1).eq.12.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.12))
     &         .OR.((lb(i1).eq.-10.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-10).
     &  or.(lb(i1).eq.-12.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.-12)) )then
       if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       ENDIF
*(16) for N*(1440) or N*(1535) (0)+rho(-)-->n+pi(-) 
       if( ((lb(i1).eq.10.and.lb(i2).eq.25).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.10).
     &  or.(lb(i1).eq.25.and.lb(i2).eq.12).
     &  or.(lb(i1).eq.12.and.lb(i2).eq.25))
     &       .OR. ((lb(i1).eq.-10.and.lb(i2).eq.27).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-10).
     &  or.(lb(i1).eq.27.and.lb(i2).eq.-12).
     &  or.(lb(i1).eq.-12.and.lb(i2).eq.27)) )then
       if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
60       IBLOCK=82
* FOR OMEGA REABSORPTION
* Relable particles, I1 is assigned to the Delta 
* and I2 is assigned to the meson
* for the reverse of the following process
*(1) for D(0)+OMEGA(0)-->n+pi(0) or p+pi(-) 
       if((iabs(lb(i1)).eq.7.and.lb(i2).eq.28).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.7))then
              if(iabs(lb(i1)).eq.7)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(2) for D(+)+OMEGA(0)-->pi(+)+n or pi(0)+p 
       if((iabs(lb(i1)).eq.8.and.lb(i2).eq.28).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.8))then
              if(iabs(lb(i1)).eq.8)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(3) for D(-)+OMEGA(0)-->n+pi(-) 
       if((iabs(lb(i1)).eq.6.and.lb(i2).eq.28).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.6))then
              if(iabs(lb(i1)).eq.6)then
        ii = i1
       lb(i1)=2
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=2
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(4) for D(++)+OMEGA(0)-->p+pi(+) 
       if((iabs(lb(i1)).eq.9.and.lb(i2).eq.28).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.9))then
              if(iabs(lb(i1)).eq.9)then
        ii = i1
       lb(i1)=1
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       else
        ii = i2
       lb(i2)=1
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       ENDIF
       endif
*(5) for N*(1440) or N*(1535)(0)+omega(0)-->n+pi(0) or p+pi(-) 
       if((iabs(lb(i1)).eq.10.and.lb(i2).eq.28).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.10).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.12).
     &  or.(lb(i2).eq.28.and.iabs(lb(i1)).eq.12))then
              if(iabs(lb(i1)).eq.10.or.iabs(lb(i1)).eq.12)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=3
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=3
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
*(6) for N*(1440) or N*(1535)(+)+omega(0)-->pi(+)+n or pi(0)+p 
       if((iabs(lb(i1)).eq.11.and.lb(i2).eq.28).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.11).
     &  or.(lb(i1).eq.28.and.iabs(lb(i2)).eq.13).
     &  or.(lb(i2).eq.28.and.iabs(lb(i1)).eq.13))then
              if(iabs(lb(i1)).eq.11.or.iabs(lb(i1)).eq.13)then
        ii = i1
       IF(X2.LE.0.5)THEN
       lb(i1)=2
       e(i1)=amn
       lb(i2)=5
       e(i2)=ap1
       go to 40
       Else
       lb(i1)=1
       e(i1)=amn
       lb(i2)=4
       e(i2)=ap1
       go to 40
       endif
              else
        ii = i2
       IF(X2.LE.0.5)THEN
       lb(i2)=2
       e(i2)=amn
       lb(i1)=5
       e(i1)=ap1
       go to 40
       Else
       lb(i2)=1
       e(i2)=amn
       lb(i1)=4
       e(i1)=ap1
       go to 40
       endif
              endif
       endif
40       em1=e(i1)
       em2=e(i2)
       if(ianti.eq.1 .and. lb(i1).ge.1 .and. lb(i2).ge.1)then
         lb(ii) = -lb(ii)
           jj = i2
          if(ii .eq. i2)jj = i1
          if(lb(jj).eq.3)then
           lb(jj) = 5
          elseif(lb(jj).eq.5)then
           lb(jj) = 3
          endif
         endif
       endif
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
50          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.E-09
          PR=SQRT(PR2)/(2.*SRT)
*          C1   = 1.0 - 2.0 * RANART(NSEED)

clin-10/25/02 get rid of argument usage mismatch in PTR():
          xptr=0.33*pr
c         cc1=ptr(0.33*pr,iseed)
         cc1=ptr(xptr,iseed)
clin-10/25/02-end

         c1=sqrt(pr**2-cc1**2)/pr
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1 
* ROTATE THE MOMENTUM
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ)
      RETURN
      END
**********************************
* sp 03/19/01                                                          *
*                                                                      *
        SUBROUTINE Crlaba(PX,PY,PZ,SRT,brel,brsgm,
     &                        I1,I2,nt,IBLOCK,nchrg,icase)
*     PURPOSE:                                                         *
*            DEALING WITH   K+ + N(D,N*)-bar <-->  La(Si)-bar + pi     *
*     NOTE   :                                                         *
*                                                                      *
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     8-> elastic scatt                               *
*                     100-> K+ + N-bar -> Sigma-bar + PI
*                     102-> PI + Sigma(Lambda)-bar -> K+ + N-bar
**********************************
        PARAMETER (MAXSTR=150001, MAXR=1, AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER  (AKA=0.498,ALA=1.1157,ASA=1.1974)
        PARAMETER  (ETAM=0.5475, AOMEGA=0.782, ARHO=0.77)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
      NT=NT
c
      PX0=PX
      PY0=PY
      PZ0=PZ
c
      if(icase .eq. 3)then
         rrr=RANART(NSEED)
         if(rrr.lt.brel) then
c            !! elastic scat.  (avoid in reverse process)
            IBLOCK=8
        else 
            IBLOCK=100
            if(rrr.lt.(brel+brsgm)) then
c*    K+ + N-bar -> Sigma-bar + PI
               LB(i1) = -15 - int(3 * RANART(NSEED))

               e(i1)=asa
            else
c*    K+ + N-bar -> Lambda-bar + PI
               LB(i1)= -14  
               e(i1)=ala
            endif
            LB(i2) = 3 + int(3 * RANART(NSEED))
            e(i2)=0.138
        endif
      endif
c
c
      if(icase .eq. 4)then
         rrr=RANART(NSEED)
         if(rrr.lt.brel) then
c            !! elastic scat.
            IBLOCK=8
         else    
            IBLOCK=102
c    PI + Sigma(Lambda)-bar -> K+ + N-bar
c         ! K+
            LB(i1) = 23
            LB(i2) = -1 - int(2 * RANART(NSEED))
            if(nchrg.eq.-2) LB(i2) = -6
            if(nchrg.eq. 1) LB(i2) = -9
            e(i1) = aka
            e(i2) = 0.938
            if(nchrg.eq.-2.or.nchrg.eq.1) e(i2)=1.232
         endif
      endif
c
      EM1=E(I1)
      EM2=E(I2)
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
      PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1     - 4.0 * (EM1*EM2)**2
      IF(PR2.LE.0.)PR2=1.e-09
      PR=SQRT(PR2)/(2.*SRT)
      C1   = 1.0 - 2.0 * RANART(NSEED)
      T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
      CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE Crkn(PX,PY,PZ,SRT,I1,I2,IBLOCK)
*     PURPOSE:                                                         *
*             DEALING WITH kaON+N/pi-->KAON +N/pi elastic PROCESS      *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     8-> PION+N-->L/S+KAON
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
*-----------------------------------------------------------------------
        IBLOCK=8
        NTAG=0
        EM1=E(I1)
        EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE Crppba(PX,PY,PZ,SRT,I1,I2,IBLOCK)
*     PURPOSE:                                                         *

clin-8/29/00*             DEALING WITH anti-nucleon annihilation with 
*             DEALING WITH anti-baryon annihilation with 

*             nucleons or baryon resonances
*             Determine:                                               *
*             (1) no. of pions in the final state
*             (2) relable particles in the final state
*             (3) new momenta of final state particles                 *
*                  
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - INFORMATION about the reaction channel          *
*                
*           iblock   - 1902 annihilation-->pion(+)+pion(-)   (2 pion)
*           iblock   - 1903 annihilation-->pion(+)+rho(-)    (3 pion)
*           iblock   - 1904 annihilation-->rho(+)+rho(-)     (4 pion)
*           iblock   - 1905 annihilation-->rho(0)+omega      (5 pion)
*           iblock   - 1906 annihilation-->omega+omega       (6 pion)
*       charge conservation is enforced in relabling particles 
*       in the final state (note: at the momentum we don't check the
*       initial charges while dealing with annihilation, since some
*       annihilation channels between antinucleons and nucleons (baryon
*       resonances) might be forbiden by charge conservation, this effect
*       should be small, but keep it in mind.
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AMRHO=0.769,AMOMGA=0.782,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
* determine the no. of pions in the final state using a 
* statistical model
       call pbarfs(srt,npion,iseed)
* find the masses of the final state particles before calculate 
* their momenta, and relable them. The masses of rho and omega 
* will be generated according to the Breit Wigner formula       (NOTE!!!
* NOT DONE YET, AT THE MOMENT LET US USE FIXED RHO AND OMEGA MAEES)
cbali2/22/99
* Here we generate two stes of integer random numbers (3,4,5)
* one or both of them are used directly as the lables of pions
* similarly, 22+nchrg1 and 22+nchrg2 are used directly 
* to label rhos  
       nchrg1=3+int(3*RANART(NSEED))
       nchrg2=3+int(3*RANART(NSEED))
* the corresponding masses of pions
      pmass1=ap1
       pmass2=ap1
       if(nchrg1.eq.3.or.nchrg1.eq.5)pmass1=ap2
       if(nchrg2.eq.3.or.nchrg2.eq.5)pmass2=ap2
* (1) for 2 pion production
       IF(NPION.EQ.2)THEN 
       IBLOCK=1902
* randomly generate the charges of final state particles,
       LB(I1)=nchrg1
       E(I1)=pmass1
       LB(I2)=nchrg2
       E(I2)=pmass2
* TO CALCULATE THE FINAL MOMENTA
       GO TO 50
       ENDIF
* (2) FOR 3 PION PRODUCTION
       IF(NPION.EQ.3)THEN 
       IBLOCK=1903
       LB(I1)=nchrg1
       E(I1)=pmass1
       LB(I2)=22+nchrg2
            E(I2)=AMRHO
       GO TO 50
       ENDIF
* (3) FOR 4 PION PRODUCTION
* we allow both rho+rho and pi+omega with 50-50% probability
        IF(NPION.EQ.4)THEN 
       IBLOCK=1904
* determine rho+rho or pi+omega
       if(RANART(NSEED).ge.0.5)then
* rho+rho  
       LB(I1)=22+nchrg1
       E(I1)=AMRHO
       LB(I2)=22+nchrg2
            E(I2)=AMRHO
       else
* pion+omega
       LB(I1)=nchrg1
       E(I1)=pmass1
       LB(I2)=28
            E(I2)=AMOMGA
       endif
       GO TO 50
       ENDIF
* (4) FOR 5 PION PRODUCTION
        IF(NPION.EQ.5)THEN 
       IBLOCK=1905
* RHO AND OMEGA
        LB(I1)=22+nchrg1
       E(I1)=AMRHO
       LB(I2)=28
       E(I2)=AMOMGA
       GO TO 50
       ENDIF
* (5) FOR 6 PION PRODUCTION
         IF(NPION.EQ.6)THEN 
       IBLOCK=1906
* OMEGA AND OMEGA
        LB(I1)=28
       E(I1)=AMOMGA
       LB(I2)=28
          E(I2)=AMOMGA
       ENDIF
cbali2/22/99
50    EM1=E(I1)
      EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.E-08
          PR=SQRT(PR2)/(2.*SRT)
* WE ASSUME AN ISOTROPIC ANGULAR DISTRIBUTION IN THE CMS 
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
      RETURN
      END
cbali2/7/99end
cbali3/5/99
**********************************
*     PURPOSE:                                                         *
*     assign final states for K+K- --> light mesons
*
      SUBROUTINE crkkpi(I1,I2,XSK1, XSK2, XSK3, XSK4,
     &             XSK5, XSK6, XSK7, XSK8, XSK9, XSK10, XSK11, SIGK,
     &             IBLOCK,lbp1,lbp2,emm1,emm2)
*
*     QUANTITIES:                                                     *
*           IBLOCK   - INFORMATION about the reaction channel          *
*                
*             iblock   - 1907
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AMRHO=0.769,AMOMGA=0.782,
     &  AMETA = 0.5473,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
 
        XSK11=XSK11
        IBLOCK=1907
        X1 = RANART(NSEED) * SIGK
        XSK2 = XSK1 + XSK2
        XSK3 = XSK2 + XSK3
        XSK4 = XSK3 + XSK4
        XSK5 = XSK4 + XSK5
        XSK6 = XSK5 + XSK6
        XSK7 = XSK6 + XSK7
        XSK8 = XSK7 + XSK8
        XSK9 = XSK8 + XSK9
        XSK10 = XSK9 + XSK10
        IF (X1 .LE. XSK1) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 3 + int(3 * RANART(NSEED))
           E(I1) = AP2
           E(I2) = AP2
           GOTO 100
        ELSE IF (X1 .LE. XSK2) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 25 + int(3 * RANART(NSEED))
           E(I1) = AP2
           E(I2) = AMRHO
           GOTO 100
        ELSE IF (X1 .LE. XSK3) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 28
           E(I1) = AP2
           E(I2) = AMOMGA
           GOTO 100
        ELSE IF (X1 .LE. XSK4) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 0
           E(I1) = AP2
           E(I2) = AMETA
           GOTO 100
        ELSE IF (X1 .LE. XSK5) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 25 + int(3 * RANART(NSEED))
           E(I1) = AMRHO
           E(I2) = AMRHO
           GOTO 100
        ELSE IF (X1 .LE. XSK6) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 28
           E(I1) = AMRHO
           E(I2) = AMOMGA
           GOTO 100
        ELSE IF (X1 .LE. XSK7) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 0
           E(I1) = AMRHO
           E(I2) = AMETA
           GOTO 100
        ELSE IF (X1 .LE. XSK8) THEN
           LB(I1) = 28
           LB(I2) = 28
           E(I1) = AMOMGA
           E(I2) = AMOMGA
           GOTO 100
        ELSE IF (X1 .LE. XSK9) THEN
           LB(I1) = 28
           LB(I2) = 0
           E(I1) = AMOMGA
           E(I2) = AMETA
           GOTO 100
        ELSE IF (X1 .LE. XSK10) THEN
           LB(I1) = 0
           LB(I2) = 0
           E(I1) = AMETA
           E(I2) = AMETA
        ELSE
          iblock = 222
          call rhores(i1,i2)
c     !! phi
          lb(i1) = 29
c          return
          e(i2)=0.
        END IF

 100    CONTINUE
        lbp1=lb(i1)
        lbp2=lb(i2)
        emm1=e(i1)
        emm2=e(i2)

      RETURN
      END
**********************************
*     PURPOSE:                                                         *
*             DEALING WITH K+Y -> piN scattering
*
      SUBROUTINE Crkhyp(PX,PY,PZ,SRT,I1,I2,
     &     XKY1, XKY2, XKY3, XKY4, XKY5,
     &     XKY6, XKY7, XKY8, XKY9, XKY10, XKY11, XKY12, XKY13,
     &     XKY14, XKY15, XKY16, XKY17, SIGK, IKMP,
     &     IBLOCK)
*
*             Determine:                                               *
*             (1) relable particles in the final state                 *
*             (2) new momenta of final state particles                 *
*                                                                        *
*     QUANTITIES:                                                    *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - INFORMATION about the reaction channel          *
*                                                                     *
*             iblock   - 1908                                          *
*             iblock   - 222   !! phi                                  *
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AMRHO=0.769,AMOMGA=0.782,APHI=1.02,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
          parameter (pimass=0.140, AMETA = 0.5473, aka=0.498,
     &     aml=1.116,ams=1.193, AM1440 = 1.44, AM1535 = 1.535)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       XKY17=XKY17
       PX0=PX
       PY0=PY
       PZ0=PZ
       IBLOCK=1908
c
        X1 = RANART(NSEED) * SIGK
        XKY2 = XKY1 + XKY2
        XKY3 = XKY2 + XKY3
        XKY4 = XKY3 + XKY4
        XKY5 = XKY4 + XKY5
        XKY6 = XKY5 + XKY6
        XKY7 = XKY6 + XKY7
        XKY8 = XKY7 + XKY8
        XKY9 = XKY8 + XKY9
        XKY10 = XKY9 + XKY10
        XKY11 = XKY10 + XKY11
        XKY12 = XKY11 + XKY12
        XKY13 = XKY12 + XKY13
        XKY14 = XKY13 + XKY14
        XKY15 = XKY14 + XKY15
        XKY16 = XKY15 + XKY16
        IF (X1 .LE. XKY1) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = PIMASS
           E(I2) = AMP
           GOTO 100
        ELSE IF (X1 .LE. XKY2) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 6 + int(4 * RANART(NSEED))
           E(I1) = PIMASS
           E(I2) = AM0
           GOTO 100
        ELSE IF (X1 .LE. XKY3) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 10 + int(2 * RANART(NSEED))
           E(I1) = PIMASS
           E(I2) = AM1440
           GOTO 100
        ELSE IF (X1 .LE. XKY4) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 12 + int(2 * RANART(NSEED))
           E(I1) = PIMASS
           E(I2) = AM1535
           GOTO 100
        ELSE IF (X1 .LE. XKY5) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = AMRHO
           E(I2) = AMP
           GOTO 100
        ELSE IF (X1 .LE. XKY6) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 6 + int(4 * RANART(NSEED))
           E(I1) = AMRHO
           E(I2) = AM0
           GOTO 100
        ELSE IF (X1 .LE. XKY7) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 10 + int(2 * RANART(NSEED))
           E(I1) = AMRHO
           E(I2) = AM1440
           GOTO 100
        ELSE IF (X1 .LE. XKY8) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 12 + int(2 * RANART(NSEED))
           E(I1) = AMRHO
           E(I2) = AM1535
           GOTO 100
        ELSE IF (X1 .LE. XKY9) THEN
           LB(I1) = 28
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = AMOMGA
           E(I2) = AMP
           GOTO 100
        ELSE IF (X1 .LE. XKY10) THEN
           LB(I1) = 28
           LB(I2) = 6 + int(4 * RANART(NSEED))
           E(I1) = AMOMGA
           E(I2) = AM0
           GOTO 100
        ELSE IF (X1 .LE. XKY11) THEN
           LB(I1) = 28
           LB(I2) = 10 + int(2 * RANART(NSEED))
           E(I1) = AMOMGA
           E(I2) = AM1440
           GOTO 100
        ELSE IF (X1 .LE. XKY12) THEN
           LB(I1) = 28
           LB(I2) = 12 + int(2 * RANART(NSEED))
           E(I1) = AMOMGA
           E(I2) = AM1535
           GOTO 100
        ELSE IF (X1 .LE. XKY13) THEN
           LB(I1) = 0
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = AMETA
           E(I2) = AMP
           GOTO 100
        ELSE IF (X1 .LE. XKY14) THEN
           LB(I1) = 0
           LB(I2) = 6 + int(4 * RANART(NSEED))
           E(I1) = AMETA
           E(I2) = AM0
           GOTO 100
        ELSE IF (X1 .LE. XKY15) THEN
           LB(I1) = 0
           LB(I2) = 10 + int(2 * RANART(NSEED))
           E(I1) = AMETA
           E(I2) = AM1440
           GOTO 100
        ELSE IF (X1 .LE. XKY16) THEN
           LB(I1) = 0
           LB(I2) = 12 + int(2 * RANART(NSEED))
           E(I1) = AMETA
           E(I2) = AM1535
           GOTO 100
        ELSE
           LB(I1) = 29
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = APHI
           E(I2) = AMN
          IBLOCK=222
           GOTO 100
        END IF

 100    CONTINUE
         if(IKMP .eq. -1) LB(I2) = -LB(I2)

      EM1=E(I1)
      EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.E-08
          PR=SQRT(PR2)/(2.*SRT)
* WE ASSUME AN ISOTROPIC ANGULAR DISTRIBUTION IN THE CMS 
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
      RETURN
      END
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE CRLAN(PX,PY,PZ,SRT,I1,I2,IBLOCK)
*     PURPOSE:                                                         *
*      DEALING WITH La/Si-bar + N --> K+ + pi PROCESS                  *
*                   La/Si + N-bar --> K- + pi                          *
*     NOTE   :                                                         *
*
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      71
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

        PX0=PX
        PY0=PY                                                          
        PZ0=PZ
        IBLOCK=71
        NTAG=0
       if( (lb(i1).ge.14.and.lb(i1).le.17) .OR.
     &     (lb(i2).ge.14.and.lb(i2).le.17) )then
        LB(I1)=21
       else
        LB(I1)=23
       endif
        LB(I2)= 3 + int(3 * RANART(NSEED))
        E(I1)=AKA
        E(I2)=0.138
        EM1=E(I1)
        EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
        PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1
      PY   = PR * S1*ST1
* FOR THE ISOTROPIC DISTRIBUTION THERE IS NO NEED TO ROTATE
      RETURN
      END
csp11/03/01 end
********************************** 
**********************************
*                                                                      *
*                                                                      *
        SUBROUTINE Crkpla(PX,PY,PZ,EC,SRT,spika,
     &                  emm1,emm2,lbp1,lbp2,I1,I2,icase,srhoks)
 
*     PURPOSE:                                                         *
*     DEALING WITH  K+ + Pi ---> La/Si-bar + B, phi+K, phi+K* OR  K* *
*                   K- + Pi ---> La/Si + B-bar  OR   K*-bar          *
 
*     NOTE   :                                                         *
*
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      71
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AP2=0.13957,AMRHO=0.769,AMOMGA=0.782,
     2  AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER (AKA=0.498,AKS=0.895,ALA=1.1157,ASA=1.1974
     1 ,APHI=1.02)
        PARAMETER (AM1440 = 1.44, AM1535 = 1.535)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

          emm1=0.
          emm2=0.
          lbp1=0
          lbp2=0
           XKP0 = spika
           XKP1 = 0.
           XKP2 = 0.
           XKP3 = 0.
           XKP4 = 0.
           XKP5 = 0.
           XKP6 = 0.
           XKP7 = 0.
           XKP8 = 0.
           XKP9 = 0.
           XKP10 = 0.
           sigm = 15.
c         if(lb(i1).eq.21.or.lb(i2).eq.21)sigm=10.
        pdd = (srt**2-(aka+ap1)**2)*(srt**2-(aka-ap1)**2)
c
         if(srt .lt. (ala+amn))go to 70
        XKP1 = sigm*(4./3.)*(srt**2-(ala+amn)**2)*
     &           (srt**2-(ala-amn)**2)/pdd
         if(srt .gt. (ala+am0))then
        XKP2 = sigm*(16./3.)*(srt**2-(ala+am0)**2)*
     &           (srt**2-(ala-am0)**2)/pdd
         endif
         if(srt .gt. (ala+am1440))then
        XKP3 = sigm*(4./3.)*(srt**2-(ala+am1440)**2)*
     &           (srt**2-(ala-am1440)**2)/pdd
         endif
         if(srt .gt. (ala+am1535))then
        XKP4 = sigm*(4./3.)*(srt**2-(ala+am1535)**2)*
     &           (srt**2-(ala-am1535)**2)/pdd
         endif
c
         if(srt .gt. (asa+amn))then
        XKP5 = sigm*4.*(srt**2-(asa+amn)**2)*
     &           (srt**2-(asa-amn)**2)/pdd
         endif
         if(srt .gt. (asa+am0))then
        XKP6 = sigm*16.*(srt**2-(asa+am0)**2)*
     &           (srt**2-(asa-am0)**2)/pdd
         endif
         if(srt .gt. (asa+am1440))then
        XKP7 = sigm*4.*(srt**2-(asa+am1440)**2)*
     &           (srt**2-(asa-am1440)**2)/pdd
         endif
         if(srt .gt. (asa+am1535))then
        XKP8 = sigm*4.*(srt**2-(asa+am1535)**2)*
     &           (srt**2-(asa-am1535)**2)/pdd
         endif
70     continue
          sig1 = 195.639
          sig2 = 372.378
       if(srt .gt. aphi+aka)then
        pff = sqrt((srt**2-(aphi+aka)**2)*(srt**2-(aphi-aka)**2))
         XKP9 = sig1*pff/sqrt(pdd)*1./32./pi/srt**2
        if(srt .gt. aphi+aks)then
        pff = sqrt((srt**2-(aphi+aks)**2)*(srt**2-(aphi-aks)**2))
         XKP10 = sig2*pff/sqrt(pdd)*3./32./pi/srt**2
       endif
        endif

clin-8/15/02 K pi -> K* (rho omega), from detailed balance, 
c neglect rho and omega mass difference for now:
        sigpik=0.
        if(srt.gt.(amrho+aks)) then
           sigpik=srhoks*9.
     1          *(srt**2-(0.77-aks)**2)*(srt**2-(0.77+aks)**2)/4
     2          /srt**2/(px**2+py**2+pz**2)
           if(srt.gt.(amomga+aks)) sigpik=sigpik*12./9.
        endif

c
         sigkp = XKP0 + XKP1 + XKP2 + XKP3 + XKP4
     &         + XKP5 + XKP6 + XKP7 + XKP8 + XKP9 + XKP10 +sigpik
           icase = 0 
         DSkn=SQRT(sigkp/PI/10.)
        dsknr=dskn+0.1
        CALL DISTCE(I1,I2,dsknr,DSkn,DT,EC,SRT,IC,
     1  PX,PY,PZ)
        IF(IC.EQ.-1)return
c
        randu = RANART(NSEED)*sigkp
        XKP1 = XKP0 + XKP1
        XKP2 = XKP1 + XKP2
        XKP3 = XKP2 + XKP3
        XKP4 = XKP3 + XKP4
        XKP5 = XKP4 + XKP5
        XKP6 = XKP5 + XKP6
        XKP7 = XKP6 + XKP7
        XKP8 = XKP7 + XKP8
        XKP9 = XKP8 + XKP9

        XKP10 = XKP9 + XKP10
c
c   !! K* formation
         if(randu .le. XKP0)then
           icase = 1
            return
         else
* La/Si-bar + B formation
           icase = 2
         if( randu .le. XKP1 )then
             lbp1 = -14
             lbp2 = 1 + int(2*RANART(NSEED))
             emm1 = ala
             emm2 = amn
             go to 60
         elseif( randu .le. XKP2 )then
             lbp1 = -14
             lbp2 = 6 + int(4*RANART(NSEED))
             emm1 = ala
             emm2 = am0
             go to 60
         elseif( randu .le. XKP3 )then
             lbp1 = -14
             lbp2 = 10 + int(2*RANART(NSEED))
             emm1 = ala
             emm2 = am1440
             go to 60
         elseif( randu .le. XKP4 )then
             lbp1 = -14
             lbp2 = 12 + int(2*RANART(NSEED))
             emm1 = ala
             emm2 = am1535
             go to 60
         elseif( randu .le. XKP5 )then
             lbp1 = -15 - int(3*RANART(NSEED))
             lbp2 = 1 + int(2*RANART(NSEED))
             emm1 = asa
             emm2 = amn
             go to 60
         elseif( randu .le. XKP6 )then
             lbp1 = -15 - int(3*RANART(NSEED))
             lbp2 = 6 + int(4*RANART(NSEED))
             emm1 = asa
             emm2 = am0
             go to 60
          elseif( randu .lt. XKP7 )then
             lbp1 = -15 - int(3*RANART(NSEED))
             lbp2 = 10 + int(2*RANART(NSEED))
             emm1 = asa
             emm2 = am1440
             go to 60
          elseif( randu .lt. XKP8 )then
             lbp1 = -15 - int(3*RANART(NSEED))
             lbp2 = 12 + int(2*RANART(NSEED))
             emm1 = asa
             emm2 = am1535
             go to 60
          elseif( randu .lt. XKP9 )then
c       !! phi +K  formation (iblock=224)
            icase = 3
             lbp1 = 29
             lbp2 = 23
             emm1 = aphi
             emm2 = aka
           if(lb(i1).eq.21.or.lb(i2).eq.21)then
c         !! phi +K-bar  formation (iblock=124)
             lbp2 = 21
             icase = -3
           endif
             go to 60
          elseif( randu .lt. XKP10 )then
c       !! phi +K* formation (iblock=226)
            icase = 4
             lbp1 = 29
             lbp2 = 30
             emm1 = aphi
             emm2 = aks
           if(lb(i1).eq.21.or.lb(i2).eq.21)then
             lbp2 = -30
             icase = -4
           endif
           go to 60

          else
c       !! (rho,omega) +K* formation (iblock=88)
            icase=5
            lbp1=25+int(3*RANART(NSEED))
            lbp2=30
            emm1=amrho
            emm2=aks
            if(srt.gt.(amomga+aks).and.RANART(NSEED).lt.0.25) then
               lbp1=28
               emm1=amomga
            endif
            if(lb(i1).eq.21.or.lb(i2).eq.21)then
               lbp2=-30
               icase=-5
            endif

          endif
          endif
c
60       if( icase.eq.2 .and. (lb(i1).eq.21.or.lb(i2).eq.21) )then
            lbp1 = -lbp1
            lbp2 = -lbp2
         endif
        PX0=PX
        PY0=PY
        PZ0=PZ
*-----------------------------------------------------------------------       
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
           PR2   = (SRT**2 - EMM1**2 - EMM2**2)**2
     1                - 4.0 * (EMM1*EMM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1
      PY   = PR * S1*ST1
* FOR THE ISOTROPIC DISTRIBUTION THERE IS NO NEED TO ROTATE
      RETURN
      END
**********************************       
*                                                                      *
*                                                                      *
        SUBROUTINE Crkphi(PX,PY,PZ,EC,SRT,IBLOCK,
     &                  emm1,emm2,lbp1,lbp2,I1,I2,ikk,icase,rrkk,prkk)
 
*     PURPOSE:                                                         *
*     DEALING WITH   KKbar, KK*bar, KbarK*, K*K*bar --> Phi + pi(rho,omega)
*     and KKbar --> (pi eta) (pi eta), (rho omega) (rho omega)
*     and KK*bar or Kbar K* --> (pi eta) (rho omega)
*
*     NOTE   :                                                         *
*
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      222
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AP2=0.13957,APHI=1.02,
     2  AM0=1.232,AMNS=1.52,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974,ACAS=1.3213)
        PARAMETER      (AKS=0.895,AOMEGA=0.7819, ARHO=0.77)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

        lb1 = lb(i1) 
        lb2 = lb(i2) 
        icase = 0

c        if(srt .lt. aphi+ap1)return
cc        if(srt .lt. aphi+ap1) then
        if(srt .lt. (aphi+ap1)) then
           sig1 = 0.
           sig2 = 0.
           sig3 = 0.
        else
c
         if((lb1.eq.23.and.lb2.eq.21).or.(lb2.eq.23.and.lb1.eq.21))then
            dnr =  4.
            ikk = 2
          elseif((lb1.eq.21.and.lb2.eq.30).or.(lb2.eq.21.and.lb1.eq.30)
     & .or.(lb1.eq.23.and.lb2.eq.-30).or.(lb2.eq.23.and.lb1.eq.-30))then
             dnr = 12.
             ikk = 1
          else
             dnr = 36.
             ikk = 0
          endif
              
          sig1 = 0.
          sig2 = 0.
          sig3 = 0.
          srri = E(i1)+E(i2)
          srr1 = aphi+ap1
          srr2 = aphi+aomega
          srr3 = aphi+arho
c
          pii = (srt**2-(e(i1)+e(i2))**2)*(srt**2-(e(i1)-e(i2))**2)
          srrt = srt - amax1(srri,srr1)
cc   to avoid divergent/negative values at small srrt:
c          if(srrt .lt. 0.3)then
          if(srrt .lt. 0.3 .and. srrt .gt. 0.01)then
          sig = 1.69/(srrt**0.141 - 0.407)
         else
          sig = 3.74 + 0.008*srrt**1.9
         endif                 
          sig1=sig*(9./dnr)*(srt**2-(aphi+ap1)**2)*
     &           (srt**2-(aphi-ap1)**2)/pii
          if(srt .gt. aphi+aomega)then
          srrt = srt - amax1(srri,srr2)
cc         if(srrt .lt. 0.3)then
          if(srrt .lt. 0.3 .and. srrt .gt. 0.01)then
          sig = 1.69/(srrt**0.141 - 0.407)
         else
          sig = 3.74 + 0.008*srrt**1.9
         endif                 
          sig2=sig*(9./dnr)*(srt**2-(aphi+aomega)**2)*
     &           (srt**2-(aphi-aomega)**2)/pii
           endif
         if(srt .gt. aphi+arho)then
          srrt = srt - amax1(srri,srr3)
cc         if(srrt .lt. 0.3)then
          if(srrt .lt. 0.3 .and. srrt .gt. 0.01)then
          sig = 1.69/(srrt**0.141 - 0.407)
         else
          sig = 3.74 + 0.008*srrt**1.9
         endif                 
          sig3=sig*(27./dnr)*(srt**2-(aphi+arho)**2)*
     &           (srt**2-(aphi-arho)**2)/pii
         endif                 
c         sig1 = amin1(20.,sig1)
c         sig2 = amin1(20.,sig2)
c         sig3 = amin1(20.,sig3)
        endif

        rrkk0=rrkk
        prkk0=prkk
        SIGM=0.
        if((lb1.eq.23.and.lb2.eq.21).or.(lb2.eq.23.and.lb1.eq.21))then
           CALL XKKANN(SRT, XSK1, XSK2, XSK3, XSK4, XSK5,
     &          XSK6, XSK7, XSK8, XSK9, XSK10, XSK11, SIGM, rrkk0)
        elseif((lb1.eq.21.and.lb2.eq.30).or.(lb2.eq.21.and.lb1.eq.30)
     & .or.(lb1.eq.23.and.lb2.eq.-30).or.(lb2.eq.23.and.lb1.eq.-30))then
           CALL XKKSAN(i1,i2,SRT,SIGKS1,SIGKS2,SIGKS3,SIGKS4,SIGM,prkk0)
        else
        endif
c
c         sigks = sig1 + sig2 + sig3
        sigm0=sigm
        sigks = sig1 + sig2 + sig3 + SIGM
        DSkn=SQRT(sigks/PI/10.)
        dsknr=dskn+0.1
        CALL DISTCE(I1,I2,dsknr,DSkn,DT,EC,SRT,IC,
     1  PX,PY,PZ)
        IF(IC.EQ.-1)return
        icase = 1
        ranx = RANART(NSEED) 

        lbp1 = 29
        emm1 = aphi
        if(ranx .le. sig1/sigks)then 
           lbp2 = 3 + int(3*RANART(NSEED))
           emm2 = ap1
        elseif(ranx .le. (sig1+sig2)/sigks)then
           lbp2 = 28
           emm2 = aomega
        elseif(ranx .le. (sig1+sig2+sig3)/sigks)then
           lbp2 = 25 + int(3*RANART(NSEED))
           emm2 = arho
        else
           if((lb1.eq.23.and.lb2.eq.21)
     &          .or.(lb2.eq.23.and.lb1.eq.21))then
              CALL crkkpi(I1,I2,XSK1, XSK2, XSK3, XSK4,
     &             XSK5, XSK6, XSK7, XSK8, XSK9, XSK10, XSK11, SIGM0,
     &             IBLOCK,lbp1,lbp2,emm1,emm2)
           elseif((lb1.eq.21.and.lb2.eq.30)
     &             .or.(lb2.eq.21.and.lb1.eq.30)
     &             .or.(lb1.eq.23.and.lb2.eq.-30)
     &             .or.(lb2.eq.23.and.lb1.eq.-30))then
              CALL crkspi(I1,I2,SIGKS1, SIGKS2, SIGKS3, SIGKS4,
     &             SIGM0,IBLOCK,lbp1,lbp2,emm1,emm2)
           else
           endif
        endif
*
        PX0=PX
        PY0=PY
        PZ0=PZ
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
           PR2   = (SRT**2 - EMM1**2 - EMM2**2)**2
     1                - 4.0 * (EMM1*EMM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1
      PY   = PR * S1*ST1
* FOR THE ISOTROPIC DISTRIBUTION THERE IS NO NEED TO ROTATE
      RETURN
      END
csp11/21/01 end
**********************************
*                                                                      *
*                                                                      *
        SUBROUTINE Crksph(PX,PY,PZ,EC,SRT,
     &     emm1,emm2,lbp1,lbp2,I1,I2,ikkg,ikkl,iblock,
     &     icase,srhoks)
 
*     PURPOSE:                                                         *
*     DEALING WITH   K + rho(omega) or K* + pi(rho,omega) 
*                    --> Phi + K(K*), pi + K* or pi + K, and elastic 
*     NOTE   :                                                         *
*
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      222
*                      223 --> phi + pi(rho,omega)
*                      224 --> phi + K <-> K + pi(rho,omega)
*                      225 --> phi + K <-> K* + pi(rho,omega)
*                      226 --> phi + K* <-> K + pi(rho,omega)
*                      227 --> phi + K* <-> K* + pi(rho,omega)
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AP2=0.13957,APHI=1.02,
     2  AM0=1.232,AMNS=1.52,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974,ACAS=1.3213)
        PARAMETER      (AKS=0.895,AOMEGA=0.7819, ARHO=0.77)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

        lb1 = lb(i1) 
        lb2 = lb(i2) 
        icase = 0
        sigela=10.
        sigkm=0.
c     K(K*) + rho(omega) -> pi K*(K)
        if((lb1.ge.25.and.lb1.le.28).or.(lb2.ge.25.and.lb2.le.28)) then
           if(iabs(lb1).eq.30.or.iabs(lb2).eq.30) then
              sigkm=srhoks
clin-2/26/03 check whether (rho K) is above the (pi K*) thresh:
           elseif((lb1.eq.23.or.lb1.eq.21.or.lb2.eq.23.or.lb2.eq.21)
     1             .and.srt.gt.(ap2+aks)) then
              sigkm=srhoks
           endif
        endif

c        if(srt .lt. aphi+aka)return
        if(srt .lt. (aphi+aka)) then
           sig11=0.
           sig22=0.
        else

c K*-bar +pi --> phi + (K,K*)-bar
         if( (iabs(lb1).eq.30.and.(lb2.ge.3.and.lb2.le.5)) .or.
     &       (iabs(lb2).eq.30.and.(lb1.ge.3.and.lb1.le.5)) )then
              dnr =  18.
              ikkl = 0
              IBLOCK = 225
c               sig1 = 15.0  
c               sig2 = 30.0  
clin-2/06/03 these large values reduces to ~10 mb for sig11 or sig22
c     due to the factors of ~1/(32*pi*s)~1/200:
               sig1 = 2047.042  
               sig2 = 1496.692
c K(-bar)+rho --> phi + (K,K*)-bar
       elseif((lb1.eq.23.or.lb1.eq.21.and.(lb2.ge.25.and.lb2.le.27)).or.
     &      (lb2.eq.23.or.lb2.eq.21.and.(lb1.ge.25.and.lb1.le.27)) )then
              dnr =  18.
              ikkl = 1
              IBLOCK = 224
c               sig1 = 3.5  
c               sig2 = 9.0  
               sig1 = 526.702
               sig2 = 1313.960
c K*(-bar) +rho
         elseif( (iabs(lb1).eq.30.and.(lb2.ge.25.and.lb2.le.27)) .or.
     &           (iabs(lb2).eq.30.and.(lb1.ge.25.and.lb1.le.27)) )then
              dnr =  54.
              ikkl = 0
              IBLOCK = 225
c               sig1 = 3.5  
c               sig2 = 9.0  
               sig1 = 1371.257
               sig2 = 6999.840
c K(-bar) + omega
         elseif( ((lb1.eq.23.or.lb1.eq.21) .and. lb2.eq.28).or.
     &           ((lb2.eq.23.or.lb2.eq.21) .and. lb1.eq.28) )then
              dnr = 6.
              ikkl = 1
              IBLOCK = 224
c               sig1 = 3.5  
c               sig2 = 6.5  
               sig1 = 355.429
               sig2 = 440.558
c K*(-bar) +omega
          else
              dnr = 18.
              ikkl = 0
              IBLOCK = 225
c               sig1 = 3.5  
c               sig2 = 15.0  
               sig1 = 482.292
               sig2 = 1698.903
          endif

            sig11 = 0.
            sig22 = 0.
c         sig11=sig1*(6./dnr)*(srt**2-(aphi+aka)**2)*
c    &           (srt**2-(aphi-aka)**2)/(srt**2-(e(i1)+e(i2))**2)/
c    &           (srt**2-(e(i1)-e(i2))**2)
        pii = sqrt((srt**2-(e(i1)+e(i2))**2)*(srt**2-(e(i1)-e(i2))**2))
        pff = sqrt((srt**2-(aphi+aka)**2)*(srt**2-(aphi-aka)**2))
          sig11 = sig1*pff/pii*6./dnr/32./pi/srt**2
c
          if(srt .gt. aphi+aks)then
c         sig22=sig2*(18./dnr)*(srt**2-(aphi+aks)**2)*
c    &           (srt**2-(aphi-aks)**2)/(srt**2-(e(i1)+e(i2))**2)/
c    &           (srt**2-(e(i1)-e(i2))**2)
        pff = sqrt((srt**2-(aphi+aks)**2)*(srt**2-(aphi-aks)**2))
          sig22 = sig2*pff/pii*18./dnr/32./pi/srt**2
           endif
c         sig11 = amin1(20.,sig11)
c         sig22 = amin1(20.,sig22)
c
        endif

c         sigks = sig11 + sig22
         sigks=sig11+sig22+sigela+sigkm
c
        DSkn=SQRT(sigks/PI/10.)
        dsknr=dskn+0.1
        CALL DISTCE(I1,I2,dsknr,DSkn,DT,EC,SRT,IC,
     1  PX,PY,PZ)
        IF(IC.EQ.-1)return
        icase = 1
        ranx = RANART(NSEED) 

         if(ranx .le. (sigela/sigks))then 
            lbp1=lb1
            emm1=e(i1)
            lbp2=lb2
            emm2=e(i2)
            iblock=111
         elseif(ranx .le. ((sigela+sigkm)/sigks))then 
            lbp1=3+int(3*RANART(NSEED))
            emm1=0.14
            if(lb1.eq.23.or.lb2.eq.23) then
               lbp2=30
               emm2=aks
            elseif(lb1.eq.21.or.lb2.eq.21) then
               lbp2=-30
               emm2=aks
            elseif(lb1.eq.30.or.lb2.eq.30) then
               lbp2=23
               emm2=aka
            else
               lbp2=21
               emm2=aka
            endif
            iblock=112
         elseif(ranx .le. ((sigela+sigkm+sig11)/sigks))then 
            lbp2 = 23
            emm2 = aka
            ikkg = 1
            if(lb1.eq.21.or.lb2.eq.21.or.lb1.eq.-30.or.lb2.eq.-30)then
               lbp2=21
               iblock=iblock-100
            endif
            lbp1 = 29
            emm1 = aphi
         else
            lbp2 = 30
            emm2 = aks
            ikkg = 0
            IBLOCK=IBLOCK+2
            if(lb1.eq.21.or.lb2.eq.21.or.lb1.eq.-30.or.lb2.eq.-30)then
               lbp2=-30
               iblock=iblock-100
            endif
            lbp1 = 29
            emm1 = aphi
         endif
*
        PX0=PX
        PY0=PY
        PZ0=PZ
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
           PR2   = (SRT**2 - EMM1**2 - EMM2**2)**2
     1                - 4.0 * (EMM1*EMM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1
      PY   = PR * S1*ST1
* FOR THE ISOTROPIC DISTRIBUTION THERE IS NO NEED TO ROTATE
      RETURN
      END
csp11/21/01 end
**********************************
********************************** 
        SUBROUTINE bbkaon(ic,SRT,PX,PY,PZ,ana,PlX,
     &  PlY,PlZ,ala,pkX,PkY,PkZ,icou1)
* purpose: generate the momenta for kaon,lambda/sigma and nucleon/delta
*          in the BB-->nlk process
* date: Sept. 9, 1994
c
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PI=3.1415962
       icou1=0
       aka=0.498
        ala=1.116
       if(ic.eq.2.or.ic.eq.4)ala=1.197
       ana=0.939
* generate the mass of the delta
       if(ic.gt.2)then
       dmax=srt-aka-ala-0.02
        DM1=RMASS(DMAX,ISEED)
       ana=dm1
       endif
       t1=aka+ana+ala
       t2=ana+ala-aka
       if(srt.le.t1)then
       icou1=-1
       return
       endif
       pmax=sqrt((srt**2-t1**2)*(srt**2-t2**2))/(2.*srt)
       if(pmax.eq.0.)pmax=1.e-09
* (1) Generate the momentum of the kaon according to the distribution Fkaon
*     and assume that the angular distribution is isotropic       
*     in the cms of the colliding pair
       ntry=0
1       pk=pmax*RANART(NSEED)
       ntry=ntry+1
       prob=fkaon(pk,pmax)
       if((prob.lt.RANART(NSEED)).and.(ntry.le.40))go to 1
       cs=1.-2.*RANART(NSEED)
       ss=sqrt(1.-cs**2)
       fai=2.*3.14*RANART(NSEED)
       pkx=pk*ss*cos(fai)
       pky=pk*ss*sin(fai)
       pkz=pk*cs
* the energy of the kaon
       ek=sqrt(aka**2+pk**2)
* (2) Generate the momentum of the nucleon/delta in the cms of N/delta 
*     and lamda/sigma 
*  the energy of the cms of NL
        eln=srt-ek
       if(eln.le.0)then
       icou1=-1
       return
       endif
* beta and gamma of the cms of L/S+N
       bx=-pkx/eln
       by=-pky/eln
       bz=-pkz/eln
       ga=1./sqrt(1.-bx**2-by**2-bz**2)
        elnc=eln/ga
       pn2=((elnc**2+ana**2-ala**2)/(2.*elnc))**2-ana**2
       if(pn2.le.0.)pn2=1.e-09
       pn=sqrt(pn2)
       csn=1.-2.*RANART(NSEED)
       ssn=sqrt(1.-csn**2)
       fain=2.*3.14*RANART(NSEED)
       px=pn*ssn*cos(fain)
       py=pn*ssn*sin(fain)
       pz=pn*csn
       en=sqrt(ana**2+pn2)
* the momentum of the lambda/sigma in the n-l cms frame is
       plx=-px
       ply=-py
       plz=-pz
* (3) LORENTZ-TRANSFORMATION INTO nn cms FRAME for the neutron/delta
        PBETA  = PX*BX + PY*By+ PZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + En )
              Px = BX * TRANS0 + PX
              Py = BY * TRANS0 + PY
              Pz = BZ * TRANS0 + PZ
* (4) Lorentz-transformation for the lambda/sigma
       el=sqrt(ala**2+plx**2+ply**2+plz**2)
        PBETA  = PlX*BX + PlY*By+ PlZ*Bz
              TRANS0  = GA * ( GA * PBETA / (GA + 1.) + El )
              Plx = BX * TRANS0 + PlX
              Ply = BY * TRANS0 + PlY
              Plz = BZ * TRANS0 + PlZ
             return
             end
******************************************
* for pion+pion-->K+K-
c      real*4 function pipik(srt)
      real function pipik(srt)
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  NOTE: DEVIDE THE CROSS SECTION TO OBTAIN K+ PRODUCTION                     *
******************************************
c      real*4   xarray(5), earray(5)
      real   xarray(5), earray(5)
      SAVE   
      data xarray /0.001, 0.7,1.5,1.7,2.0/
      data earray /1.,1.2,1.6,2.0,2.4/

           pmass=0.9383 
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
       pipik=0.
       if(srt.le.1.)return
       if(srt.gt.2.4)then
           pipik=2.0/2.
           return
       endif
        if (srt .lt. earray(1)) then
           pipik =xarray(1)/2.
           return
        end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,5
        if (earray(ie) .eq. srt) then
          pipik = xarray(ie)
          go to 10
        else if (earray(ie) .gt. srt) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          pipik = exp(ymin + (alog(srt)-xmin)*(ymax-ymin)
     &/(xmax-xmin) )
          go to 10
        end if
 1001 continue
10       PIPIK=PIPIK/2.
       continue
      return
        END
**********************************
* TOTAL PION-P INELASTIC CROSS SECTION 
*  from the CERN data book
*  date: Sept.2, 1994
*  for pion++p-->Delta+pion
c      real*4 function pionpp(srt)
      real function pionpp(srt)
      SAVE   
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in fm**2                                 *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
           pmass=0.14 
       pmass1=0.938
       PIONPP=0.00001
       IF(SRT.LE.1.22)RETURN
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
        plab=sqrt(((srt**2-pmass**2-pmass1**2)/(2.*pmass1))**2-pmass**2)
       pmin=0.3
       pmax=25.0
       if(plab.gt.pmax)then
       pionpp=20./10.
       return
       endif
        if(plab .lt. pmin)then
        pionpp = 0.
        return
        end if
c* fit parameters
       a=24.3
       b=-12.3
       c=0.324
       an=-1.91
       d=-2.44
        pionpp = a+b*(plab**an)+c*(alog(plab))**2+d*alog(plab)
       if(pionpp.le.0)pionpp=0
       pionpp=pionpp/10.
        return
        END
**********************************
* elementary cross sections
*  from the CERN data book
*  date: Sept.2, 1994
*  for pion-+p-->INELASTIC
c      real*4 function pipp1(srt)
      real function pipp1(srt)
      SAVE   
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in fm**2                                 *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*  UNITS: FM**2
******************************************
           pmass=0.14 
       pmass1=0.938
       PIPP1=0.0001
       IF(SRT.LE.1.22)RETURN
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
c      ekin = 2.*pmass*((srt/(2.*pmass))**2 - 1.)
        plab=sqrt(((srt**2-pmass**2-pmass1**2)/(2.*pmass1))**2-pmass**2)
       pmin=0.3
       pmax=25.0
       if(plab.gt.pmax)then
       pipp1=20./10.
       return
       endif
        if(plab .lt. pmin)then
        pipp1 = 0.
        return
        end if
c* fit parameters
       a=26.6
       b=-7.18
       c=0.327
       an=-1.86
       d=-2.81
        pipp1 = a+b*(plab**an)+c*(alog(plab))**2+d*alog(plab)
       if(pipp1.le.0)pipp1=0
       PIPP1=PIPP1/10.
        return
        END
* *****************************
c       real*4 function xrho(srt)
      real function xrho(srt)
      SAVE   
*       xsection for pp-->pp+rho
* *****************************
       pmass=0.9383
       rmass=0.77
       trho=0.151
       xrho=0.000000001
       if(srt.le.2.67)return
       ESMIN=2.*0.9383+rmass-trho/2.
       ES=srt
* the cross section for tho0 production is
       xrho0=0.24*(es-esmin)/(1.4+(es-esmin)**2)
       xrho=3.*Xrho0
       return
       end
* *****************************
c       real*4 function omega(srt)
      real function omega(srt)
      SAVE   
*       xsection for pp-->pp+omega
* *****************************
       pmass=0.9383
       omass=0.782
       tomega=0.0084
       omega=0.00000001
       if(srt.le.2.68)return
       ESMIN=2.*0.9383+omass-tomega/2.
       es=srt
       omega=0.36*(es-esmin)/(1.25+(es-esmin)**2)
       return
       end
******************************************
* for ppi(+)-->DELTA+pi
c      real*4 function TWOPI(srt)
      real function TWOPI(srt)
*  This function contains the experimental pi+p-->DELTA+PION cross sections   *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
c      real*4   xarray(19), earray(19)
      real   xarray(19), earray(19)
      SAVE   
      data xarray /0.300E-05,0.187E+01,0.110E+02,0.149E+02,0.935E+01,
     &0.765E+01,0.462E+01,0.345E+01,0.241E+01,0.185E+01,0.165E+01,
     &0.150E+01,0.132E+01,0.117E+01,0.116E+01,0.100E+01,0.856E+00,
     &0.745E+00,0.300E-05/
      data earray /0.122E+01, 0.147E+01, 0.172E+01, 0.197E+01,
     &0.222E+01, 0.247E+01, 0.272E+01, 0.297E+01, 0.322E+01,
     &0.347E+01, 0.372E+01, 0.397E+01, 0.422E+01, 0.447E+01,
     &0.472E+01, 0.497E+01, 0.522E+01, 0.547E+01, 0.572E+01/

           pmass=0.14 
       pmass1=0.938
       TWOPI=0.000001
       if(srt.le.1.22)return
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
        plab=SRT
      if (plab .lt. earray(1)) then
        TWOPI= 0.00001
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,19
        if (earray(ie) .eq. plab) then
          TWOPI= xarray(ie)
          return
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          TWOPI= exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &    /(xmax-xmin) )
          return
        end if
 1001   continue
      return
        END
******************************************
******************************************
* for ppi(+)-->DELTA+RHO
c      real*4 function THREPI(srt)
      real function THREPI(srt)
*  This function contains the experimental pi+p-->DELTA + rho cross sections  *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
c      real*4   xarray(15), earray(15)
      real   xarray(15), earray(15)
      SAVE   
      data xarray /8.0000000E-06,6.1999999E-05,1.881940,5.025690,    
     &11.80154,13.92114,15.07308,11.79571,11.53772,10.01197,9.792673,    
     &9.465264,8.970490,7.944254,6.886320/    
      data earray /0.122E+01, 0.147E+01, 0.172E+01, 0.197E+01,
     &0.222E+01, 0.247E+01, 0.272E+01, 0.297E+01, 0.322E+01,
     &0.347E+01, 0.372E+01, 0.397E+01, 0.422E+01, 0.447E+01,
     &0.472E+01/

           pmass=0.14 
       pmass1=0.938
       THREPI=0.000001
       if(srt.le.1.36)return
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
        plab=SRT
      if (plab .lt. earray(1)) then
        THREPI = 0.00001
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,15
        if (earray(ie) .eq. plab) then
          THREPI= xarray(ie)
          return
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          THREPI = exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &    /(xmax-xmin) )
          return
        end if
 1001   continue
      return
        END
******************************************
******************************************
* for ppi(+)-->DELTA+omega
c      real*4 function FOURPI(srt)
      real function FOURPI(srt)
*  This function contains the experimental pi+p-->DELTA+PION cross sections   *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*                                                                             *
******************************************
c      real*4   xarray(10), earray(10)
      real   xarray(10), earray(10)
      SAVE   
      data xarray /0.0001,1.986597,6.411932,7.636956,    
     &9.598362,9.889740,10.24317,10.80138,11.86988,12.83925/    
      data earray /2.468,2.718,2.968,0.322E+01,
     &0.347E+01, 0.372E+01, 0.397E+01, 0.422E+01, 0.447E+01,
     &0.472E+01/

           pmass=0.14 
       pmass1=0.938
       FOURPI=0.000001
       if(srt.le.1.52)return
* 1.Calculate p(lab)  from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
        plab=SRT
      if (plab .lt. earray(1)) then
        FOURPI= 0.00001
        return
      end if
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,10
        if (earray(ie) .eq. plab) then
          FOURPI= xarray(ie)
          return
        else if (earray(ie) .gt. plab) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          FOURPI= exp(ymin + (alog(plab)-xmin)*(ymax-ymin)
     &    /(xmax-xmin) )
          return
        end if
 1001   continue
      return
        END
******************************************
******************************************
* for pion (rho or omega)+baryon resonance collisions
c      real*4 function reab(i1,i2,srt,ictrl)
      real function reab(i1,i2,srt,ictrl)
*  This function calculates the cross section for 
*  pi+Delta(N*)-->N+PION process                                              *
*  srt    = DSQRT(s) in GeV                                                   *
*  reab   = cross section in fm**2                                            *
*  ictrl=1,2,3 for pion, rho and omega+D(N*)    
****************************************
      PARAMETER (MAXSTR=150001,MAXR=1,PI=3.1415926)
      parameter      (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
      PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
      parameter      (amn=0.938,ap1=0.14,arho=0.77,aomega=0.782)
       parameter       (maxx=20,maxz=24)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      SAVE   
       LB1=LB(I1)
       LB2=LB(I2)
       reab=0
       if(ictrl.eq.1.and.srt.le.(amn+2.*ap1+0.02))return
       if(ictrl.eq.3.and.srt.le.(amn+ap1+aomega+0.02))return
       pin2=((srt**2+ap1**2-amn**2)/(2.*srt))**2-ap1**2
       if(pin2.le.0)return
* for pion+D(N*)-->pion+N
       if(ictrl.eq.1)then
       if(e(i1).gt.1)then 
       ed=e(i1)       
       else
       ed=e(i2)
       endif       
       pout2=((srt**2+ap1**2-ed**2)/(2.*srt))**2-ap1**2
       if(pout2.le.0)return
       xpro=twopi(srt)/10.
       factor=1/3.
       if( ((lb1.eq.8.and.lb2.eq.5).or.
     &    (lb1.eq.5.and.lb2.eq.8))
     &        .OR.((lb1.eq.-8.and.lb2.eq.3).or.
     &    (lb1.eq.3.and.lb2.eq.-8)) )factor=1/4.
       if((iabs(lb1).ge.10.and.iabs(lb1).le.13).
     &  or.(iabs(lb2).ge.10.and.iabs(lb2).le.13))factor=1.
       reab=factor*pin2/pout2*xpro
       return
       endif
* for rho reabsorption
       if(ictrl.eq.2)then
       if(lb(i2).ge.25)then 
       ed=e(i1)
       arho1=e(i2)       
       else
       ed=e(i2)
       arho1=e(i1)
       endif       
       if(srt.le.(amn+ap1+arho1+0.02))return
       pout2=((srt**2+arho1**2-ed**2)/(2.*srt))**2-arho1**2
       if(pout2.le.0)return
       xpro=threpi(srt)/10.
       factor=1/3.
       if( ((lb1.eq.8.and.lb2.eq.27).or.
     &       (lb1.eq.27.and.lb2.eq.8))
     & .OR. ((lb1.eq.-8.and.lb2.eq.25).or.
     &       (lb1.eq.25.and.lb2.eq.-8)) )factor=1/4.
       if((iabs(lb1).ge.10.and.iabs(lb1).le.13).
     &  or.(iabs(lb2).ge.10.and.iabs(lb2).le.13))factor=1.
       reab=factor*pin2/pout2*xpro
       return
       endif
* for omega reabsorption
       if(ictrl.eq.3)then
       if(e(i1).gt.1)ed=e(i1)       
       if(e(i2).gt.1)ed=e(i2)       
       pout2=((srt**2+aomega**2-ed**2)/(2.*srt))**2-aomega**2
       if(pout2.le.0)return
       xpro=fourpi(srt)/10.
       factor=1/6.
       if((iabs(lb1).ge.10.and.iabs(lb1).le.13).
     &  or.(iabs(lb2).ge.10.and.iabs(lb2).le.13))factor=1./3.
       reab=factor*pin2/pout2*xpro
       endif
      return
        END
******************************************
* for the reabsorption of two resonances
* This function calculates the cross section for 
* DD-->NN, N*N*-->NN and DN*-->NN
c      real*4 function reab2d(i1,i2,srt)
      real function reab2d(i1,i2,srt)
*  srt    = DSQRT(s) in GeV                                                   *
*  reab   = cross section in mb
****************************************
      PARAMETER      (MAXSTR=150001,MAXR=1,PI=3.1415926)
      parameter      (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
      PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
      parameter      (amn=0.938,ap1=0.14,arho=0.77,aomega=0.782)
       parameter       (maxx=20,maxz=24)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      SAVE   
       reab2d=0
       LB1=iabs(LB(I1))
       LB2=iabs(LB(I2))
       ed1=e(i1)       
       ed2=e(i2)       
       pin2=(srt/2.)**2-amn**2
       pout2=((srt**2+ed1**2-ed2**2)/(2.*srt))**2-ed1**2
       if(pout2.le.0)return
       xpro=x2pi(srt)
       factor=1/4.
       if((lb1.ge.10.and.lb1.le.13).and.
     &    (lb2.ge.10.and.lb2.le.13))factor=1.
       if((lb1.ge.6.and.lb1.le.9).and.
     &    (lb2.gt.10.and.lb2.le.13))factor=1/2.
       if((lb2.ge.6.and.lb2.le.9).and.
     &    (lb1.gt.10.and.lb1.le.13))factor=1/2.
       reab2d=factor*pin2/pout2*xpro
       return
       end
***************************************
      SUBROUTINE rotate(PX0,PY0,PZ0,px,py,pz)
      SAVE   
* purpose: rotate the momentum of a particle in the CMS of p1+p2 such that 
* the x' y' and z' in the cms of p1+p2 is the same as the fixed x y and z
* quantities:
*            px0,py0 and pz0 are the cms momentum of the incoming colliding
*            particles
*            px, py and pz are the cms momentum of any one of the particles 
*            after the collision to be rotated
***************************************
* the momentum, polar and azimuthal angles of the incoming momentm
      PR0  = SQRT( PX0**2 + PY0**2 + PZ0**2 )
      IF(PR0.EQ.0)PR0=0.00000001
      C2  = PZ0 / PR0
      IF(PX0 .EQ. 0.0 .AND. PY0 .EQ. 0.0) THEN
        T2 = 0.0
      ELSE
        T2=ATAN2(PY0,PX0)
      END IF
      S2  =  SQRT( 1.0 - C2**2 )
      CT2  = COS(T2)
      ST2  = SIN(T2)
* the momentum, polar and azimuthal angles of the momentum to be rotated
      PR=SQRT(PX**2+PY**2+PZ**2)
      IF(PR.EQ.0)PR=0.0000001
      C1=PZ/PR
      IF(PX.EQ.0.AND.PY.EQ.0)THEN
      T1=0.
      ELSE
      T1=ATAN2(PY,PX)
      ENDIF
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      SS   = C2 * S1 * CT1  +  S2 * C1
* THE MOMENTUM AFTER ROTATION
      PX   = PR * ( SS*CT2 - S1*ST1*ST2 )
      PY   = PR * ( SS*ST2 + S1*ST1*CT2 )
      PZ   = PR * ( C1*C2 - S1*S2*CT1 )
      RETURN
      END
******************************************
c      real*4 function Xpp(srt)
      real function Xpp(srt)
*  This function contains the experimental total n-p cross sections           *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*  WITH A CUTOFF AT 55MB                                                      *
******************************************
c      real*4   xarray(14), earray(14)
      real   xarray(14), earray(14)
      SAVE   
      data earray /20.,30.,40.,60.,80.,100.,
     &170.,250.,310.,
     &350.,460.,560.,660.,800./
      data xarray /150.,90.,80.6,48.0,36.6,
     &31.6,25.9,24.0,23.1,
     &24.0,28.3,33.6,41.5,47/

      xpp=0.
       pmass=0.9383 
* 1.Calculate E_kin(lab) [MeV] from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
      ekin = 2000.*pmass*((srt/(2.*pmass))**2 - 1.)
      if (ekin .lt. earray(1)) then
        xpp = xarray(1)
       IF(XPP.GT.55)XPP=55
        return
      end if
       IF(EKIN.GT.EARRAY(14))THEN
       XPP=XARRAY(14)
       RETURN
       ENDIF
*
*
* 2.Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,14
        if (earray(ie) .eq. ekin) then
          xPP= xarray(ie)
       if(xpp.gt.55)xpp=55.
          return
       endif
        if (earray(ie) .gt. ekin) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          XPP = exp(ymin + (alog(ekin)-xmin)
     &          *(ymax-ymin)/(xmax-xmin) )
       IF(XPP.GT.55)XPP=55.
       go to 50
        end if
 1001 continue
50       continue
        return
        END
******************************************
      real function Xnp(srt)
*  This function contains the experimental total n-p cross sections           *
*  srt    = DSQRT(s) in GeV                                                   *
*  xsec   = production cross section in mb                                    *
*  earray = EXPerimental table with proton energies in MeV                    *
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
*  WITH  A CUTOFF AT 55MB                                                *
******************************************
c      real*4   xarray(11), earray(11)
      real   xarray(11), earray(11)
      SAVE   
      data   earray /20.,30.,40.,60.,90.,135.0,200.,
     &300.,400.,600.,800./
      data  xarray / 410.,270.,214.5,130.,78.,53.5,
     &41.6,35.9,34.2,34.3,34.9/

       xnp=0.
       pmass=0.9383
* 1.Calculate E_kin(lab) [MeV] from srt [GeV]
*   Formula used:   DSQRT(s) = 2 m DSQRT(E_kin/(2m) + 1)
      ekin = 2000.*pmass*((srt/(2.*pmass))**2 - 1.)
      if (ekin .lt. earray(1)) then
        xnp = xarray(1)
       IF(XNP.GT.55)XNP=55
        return
      end if
       IF(EKIN.GT.EARRAY(11))THEN
       XNP=XARRAY(11)
       RETURN
       ENDIF
*
*Interpolate double logarithmically to find sigma(srt)
*
      do 1001 ie = 1,11
        if (earray(ie) .eq. ekin) then
          xNP = xarray(ie)
         if(xnp.gt.55)xnp=55.
          return
       endif
        if (earray(ie) .gt. ekin) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          xNP = exp(ymin + (alog(ekin)-xmin)
     &          *(ymax-ymin)/(xmax-xmin) )
       IF(XNP.GT.55)XNP=55
       go to 50
        end if
 1001 continue
50       continue
        return
        END
*******************************
       function ptr(ptmax,iseed)
* (2) Generate the transverse momentum
*     OF nucleons
*******************************
        COMMON/TABLE/ xarray(0:1000),earray(0:1000)
cc      SAVE /TABLE/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
       ISEED=ISEED
       ptr=0.
       if(ptmax.le.1.e-02)then
       ptr=ptmax
       return
       endif
       if(ptmax.gt.2.01)ptmax=2.01
       tryial=ptdis(ptmax)/ptdis(2.01)
       XT=RANART(NSEED)*tryial
* look up the table and
*Interpolate double logarithmically to find pt
        do 50 ie = 1,200
        if (earray(ie) .eq. xT) then
          ptr = xarray(ie)
       return
       end if
          if(xarray(ie-1).le.0.00001)go to 50
          if(xarray(ie).le.0.00001)go to 50
          if(earray(ie-1).le.0.00001)go to 50
          if(earray(ie).le.0.00001)go to 50
        if (earray(ie) .gt. xT) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          ptr= exp(ymin + (alog(xT)-xmin)*(ymax-ymin)
     &    /(xmax-xmin) )
       if(ptr.gt.ptmax)ptr=ptmax
       return
       endif
50      continue
       return
       end

**********************************
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE XND(px,py,pz,srt,I1,I2,xinel,
     &               sigk,xsk1,xsk2,xsk3,xsk4,xsk5)
*     PURPOSE:                                                         *
*             calculate NUCLEON-BARYON RESONANCE inelatic Xsection     *
*     NOTE   :                                                         *
*     QUANTITIES:                                                 *
*                      CHANNELS. M12 IS THE REVERSAL CHANNEL OF N12    *
*                      N12,                                            *
*                      M12=1 FOR p+n-->delta(+)+ n                     *
*                          2     p+n-->delta(0)+ p                     *
*                          3     p+p-->delta(++)+n                     *
*                          4     p+p-->delta(+)+p                      *
*                          5     n+n-->delta(0)+n                      *
*                          6     n+n-->delta(-)+p                      *
*                          7     n+p-->N*(0)(1440)+p                   *
*                          8     n+p-->N*(+)(1440)+n                   *
*                        9     p+p-->N*(+)(1535)+p                     *
*                        10    n+n-->N*(0)(1535)+n                     *
*                         11    n+p-->N*(+)(1535)+n                     *
*                        12    n+p-->N*(0)(1535)+p
*                        13    D(++)+D(-)-->N*(+)(1440)+n
*                         14    D(++)+D(-)-->N*(0)(1440)+p
*                        15    D(+)+D(0)--->N*(+)(1440)+n
*                        16    D(+)+D(0)--->N*(0)(1440)+p
*                        17    D(++)+D(0)-->N*(+)(1535)+p
*                        18    D(++)+D(-)-->N*(0)(1535)+p
*                        19    D(++)+D(-)-->N*(+)(1535)+n
*                        20    D(+)+D(+)-->N*(+)(1535)+p
*                        21    D(+)+D(0)-->N*(+)(1535)+n
*                        22    D(+)+D(0)-->N*(0)(1535)+p
*                        23    D(+)+D(-)-->N*(0)(1535)+n
*                        24    D(0)+D(0)-->N*(0)(1535)+n
*                          25    N*(+)(14)+N*(+)(14)-->N*(+)(15)+p
*                          26    N*(0)(14)+N*(0)(14)-->N*(0)(15)+n
*                          27    N*(+)(14)+N*(0)(14)-->N*(+)(15)+n
*                        28    N*(+)(14)+N*(0)(14)-->N*(0)(15)+p
*                        29    N*(+)(14)+D+-->N*(+)(15)+p
*                        30    N*(+)(14)+D0-->N*(+)(15)+n
*                        31    N*(+)(14)+D(-)-->N*(0)(1535)+n
*                        32    N*(0)(14)+D++--->N*(+)(15)+p
*                        33    N*(0)(14)+D+--->N*(+)(15)+n
*                        34    N*(0)(14)+D+--->N*(0)(15)+p
*                        35    N*(0)(14)+D0-->N*(0)(15)+n
*                        36    N*(+)(14)+D0--->N*(0)(15)+p
*                            and more
***********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AKA=0.498,APHI=1.020,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
        common /gg/ dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
        COMMON /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
        COMMON /NN/NNN
cc      SAVE /NN/
        COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
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
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      SAVE   

*-----------------------------------------------------------------------
       xinel=0.
       sigk=0
       xsk1=0
       xsk2=0
       xsk3=0
       xsk4=0
       xsk5=0
        EM1=E(I1)
        EM2=E(I2)
      PR  = SQRT( PX**2 + PY**2 + PZ**2 )
*     CAN HAPPEN ANY MORE ==> RETURN (2.04 = 2*AVMASS + PI-MASS+0.02)
        IF (SRT .LT. 2.04) RETURN
* Resonance absorption or Delta + N-->N*(1440), N*(1535)
* COM: TEST FOR DELTA OR N* ABSORPTION
*      IN THE PROCESS DELTA+N-->NN, N*+N-->NN
        PRF=SQRT(0.25*SRT**2-AVMASS**2)
        IF(EM1.GT.1.)THEN
        DELTAM=EM1
        ELSE
        DELTAM=EM2
        ENDIF
        RENOM=DELTAM*PRF**2/DENOM(SRT,1.)/PR
        RENOMN=DELTAM*PRF**2/DENOM(SRT,2.)/PR
        RENOM1=DELTAM*PRF**2/DENOM(SRT,-1.)/PR
* avoid the inelastic collisions between n+delta- -->N+N 
*       and p+delta++ -->N+N due to charge conservation,
*       but they can scatter to produce kaons 
       if((iabs(lb(i1)).eq.2).and.(iabs(lb(i2)).eq.6)) renom=0.
       if((iabs(lb(i2)).eq.2).and.(iabs(lb(i1)).eq.6)) renom=0.
       if((iabs(lb(i1)).eq.1).and.(iabs(lb(i2)).eq.9)) renom=0.
       if((iabs(lb(i2)).eq.1).and.(iabs(lb(i1)).eq.9)) renom=0.
       Call M1535(iabs(lb(i1)),iabs(lb(i2)),srt,x1535)
        X1440=(3./4.)*SIGMA(SRT,2,0,1)
* CROSS SECTION FOR KAON PRODUCTION from the four channels
* for NLK channel
       akp=0.498
       ak0=0.498
       ana=0.94
       ada=1.232
       al=1.1157
       as=1.1197
       xsk1=0
       xsk2=0
       xsk3=0
       xsk4=0
c      !! phi production
       xsk5=0
       t1nlk=ana+al+akp
       if(srt.le.t1nlk)go to 222
       XSK1=1.5*PPLPK(SRT)
* for DLK channel
       t1dlk=ada+al+akp
       t2dlk=ada+al-akp
       if(srt.le.t1dlk)go to 222
       es=srt
       pmdlk2=(es**2-t1dlk**2)*(es**2-t2dlk**2)/(4.*es**2)
       pmdlk=sqrt(pmdlk2)
       XSK3=1.5*PPLPK(srt)
* for NSK channel
       t1nsk=ana+as+akp
       t2nsk=ana+as-akp
       if(srt.le.t1nsk)go to 222
       pmnsk2=(es**2-t1nsk**2)*(es**2-t2nsk**2)/(4.*es**2)
       pmnsk=sqrt(pmnsk2)
       XSK2=1.5*(PPK1(srt)+PPK0(srt))
* for DSK channel
       t1DSk=aDa+aS+akp
       t2DSk=aDa+aS-akp
       if(srt.le.t1dsk)go to 222
       pmDSk2=(es**2-t1DSk**2)*(es**2-t2DSk**2)/(4.*es**2)
       pmDSk=sqrt(pmDSk2)
       XSK4=1.5*(PPK1(srt)+PPK0(srt))
csp11/21/01
c phi production
       if(srt.le.(2.*amn+aphi))go to 222
c  !! mb put the correct form
         xsk5 = 0.0001
csp11/21/01 end

* THE TOTAL KAON+ PRODUCTION CROSS SECTION IS THEN
222       SIGK=XSK1+XSK2+XSK3+XSK4

cbz3/7/99 neutralk
        XSK1 = 2.0 * XSK1
        XSK2 = 2.0 * XSK2
        XSK3 = 2.0 * XSK3
        XSK4 = 2.0 * XSK4
        SIGK = 2.0 * SIGK + xsk5
cbz3/7/99 neutralk end

* avoid the inelastic collisions between n+delta- -->N+N 
*       and p+delta++ -->N+N due to charge conservation,
*       but they can scatter to produce kaons 
       if(((iabs(lb(i1)).eq.2).and.(iabs(lb(i2)).eq.6)).OR. 
     &         ((iabs(lb(i2)).eq.2).and.(iabs(lb(i1)).eq.6)).OR.
     &         ((iabs(lb(i1)).eq.1).and.(iabs(lb(i2)).eq.9)).OR.
     &         ((iabs(lb(i2)).eq.1).and.(iabs(lb(i1)).eq.9)))THEN
       xinel=sigk
       return
       ENDIF
* WE DETERMINE THE REACTION CHANNELS IN THE FOLLOWING
* FOR n+delta(++)-->p+p or n+delta(++)-->n+N*(+)(1440),n+N*(+)(1535)
* REABSORPTION OR N*(1535) PRODUCTION LIKE IN P+P OR N*(1440) LIKE PN, 
        IF(LB(I1)*LB(I2).EQ.18.AND.
     &    (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))then
        SIGND=SIGMA(SRT,1,1,0)+0.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
        xinel=SIGDN+X1440+X1535+SIGK
       RETURN
       endif
* FOR p+delta(-)-->n+n or p+delta(-)-->n+N*(0)(1440),n+N*(0)(1535)
* REABSORPTION OR N*(1535) PRODUCTION LIKE IN P+P OR N*(1440) LIKE PN, 
        IF(LB(I1)*LB(I2).EQ.6.AND.
     &    (iabs(LB(I1)).EQ.1.OR.iabs(LB(I2)).EQ.1))THEN
        SIGND=SIGMA(SRT,1,1,0)+0.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
        xinel=SIGDN+X1440+X1535+SIGK
       RETURN
       endif
* FOR p+delta(+)-->p+p, N*(+)(144)+p, N*(+)(1535)+p
cbz11/25/98
        IF(LB(I1)*LB(I2).EQ.8.AND.
     &    (iabs(LB(I1)).EQ.1.OR.iabs(LB(I2)).EQ.1))THEN
        SIGND=1.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
        xinel=SIGDN+x1440+x1535+SIGK
       RETURN
       endif
* FOR n+delta(0)-->n+n, N*(0)(144)+n, N*(0)(1535)+n
        IF(LB(I1)*LB(I2).EQ.14.AND.
     &   (iabs(LB(I1)).EQ.2.AND.iabs(LB(I2)).EQ.2))THEN
        SIGND=1.5*SIGMA(SRT,1,1,1)
        SIGDN=0.25*SIGND*RENOM
        xinel=SIGDN+x1440+x1535+SIGK
       RETURN
       endif
* FOR n+delta(+)-->n+p, N*(+)(1440)+n,N*(0)(1440)+p,
*                       N*(+)(1535)+n,N*(0)(1535)+p
        IF(LB(I1)*LB(I2).EQ.16.AND.
     &     (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))THEN
        SIGND=0.5*SIGMA(SRT,1,1,1)+0.25*SIGMA(SRT,1,1,0)
        SIGDN=0.5*SIGND*RENOM
        xinel=SIGDN+2.*x1440+2.*x1535+SIGK
       RETURN
       endif
* FOR p+delta(0)-->n+p, N*(+)(1440)+n,N*(0)(1440)+p,
*                       N*(+)(1535)+n,N*(0)(1535)+p
        IF(LB(I1)*LB(I2).EQ.7)THEN
        SIGND=0.5*SIGMA(SRT,1,1,1)+0.25*SIGMA(SRT,1,1,0)
        SIGDN=0.5*SIGND*RENOM
        xinel=SIGDN+2.*x1440+2.*x1535+SIGK
       RETURN
       endif
* FOR p+N*(0)(14)-->p+n, N*(+)(1535)+n,N*(0)(1535)+p
* OR  P+N*(0)(14)-->D(+)+N, D(0)+P, 
        IF(LB(I1)*LB(I2).EQ.10.AND.
     &   (iabs(LB(I1)).EQ.1.OR.iabs(LB(I2)).EQ.1))then
        SIGND=(3./4.)*SIGMA(SRT,2,0,1)
        SIGDN=SIGND*RENOMN
        xinel=SIGDN+X1535+SIGK
       RETURN
       endif
* FOR n+N*(+)-->p+n, N*(+)(1535)+n,N*(0)(1535)+p
        IF(LB(I1)*LB(I2).EQ.22.AND.
     &   (iabs(LB(I1)).EQ.2.OR.iabs(LB(I2)).EQ.2))then
        SIGND=(3./4.)*SIGMA(SRT,2,0,1)
        SIGDN=SIGND*RENOMN
        xinel=SIGDN+X1535+SIGK
       RETURN
       endif
* FOR N*(1535)+N-->N+N COLLISIONS
        IF((iabs(LB(I1)).EQ.12).OR.(iabs(LB(I1)).EQ.13).OR.
     1  (iabs(LB(I2)).EQ.12).OR.(iabs(LB(I2)).EQ.13))THEN
        SIGND=X1535
        SIGDN=SIGND*RENOM1
        xinel=SIGDN+SIGK
       RETURN
       endif
        RETURN
       end
**********************************
*                                                                      *
*                                                                      *
      SUBROUTINE XDDIN(PX,PY,PZ,SRT,I1,I2,
     &XINEL,SIGK,XSK1,XSK2,XSK3,XSK4,XSK5)
*     PURPOSE:                                                         *
*             DEALING WITH BARYON RESONANCE-BARYON RESONANCE COLLISIONS*
*     NOTE   :                                                         *
*           VALID ONLY FOR BARYON-BARYON-DISTANCES LESS THAN 1.32 FM   *
*           (1.32 = 2 * HARD-CORE-RADIUS [HRC] )                       *
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           NSTAR =1 INCLUDING N* RESORANCE,ELSE NOT                   *
*           NDIRCT=1 INCLUDING DIRECT PION PRODUCTION PROCESS         *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      0-> COLLISION CANNOT HAPPEN                     *
*                      1-> N-N ELASTIC COLLISION                       *
*                      2-> N+N->N+DELTA,OR N+N->N+N* REACTION          *
*                      3-> N+DELTA->N+N OR N+N*->N+N REACTION          *
*                      4-> N+N->N+N+PION,DIRTCT PROCESS                *
*                     5-> DELTA(N*)+DELTA(N*)   TOTAL   COLLISIONS    *
*           N12       - IS USED TO SPECIFY BARYON-BARYON REACTION      *
*                      CHANNELS. M12 IS THE REVERSAL CHANNEL OF N12    *
*                      N12,                                            *
*                      M12=1 FOR p+n-->delta(+)+ n                     *
*                          2     p+n-->delta(0)+ p                     *
*                          3     p+p-->delta(++)+n                     *
*                          4     p+p-->delta(+)+p                      *
*                          5     n+n-->delta(0)+n                      *
*                          6     n+n-->delta(-)+p                      *
*                          7     n+p-->N*(0)(1440)+p                   *
*                          8     n+p-->N*(+)(1440)+n                   *
*                        9     p+p-->N*(+)(1535)+p                     *
*                        10    n+n-->N*(0)(1535)+n                     *
*                         11    n+p-->N*(+)(1535)+n                     *
*                        12    n+p-->N*(0)(1535)+p
*                        13    D(++)+D(-)-->N*(+)(1440)+n
*                         14    D(++)+D(-)-->N*(0)(1440)+p
*                        15    D(+)+D(0)--->N*(+)(1440)+n
*                        16    D(+)+D(0)--->N*(0)(1440)+p
*                        17    D(++)+D(0)-->N*(+)(1535)+p
*                        18    D(++)+D(-)-->N*(0)(1535)+p
*                        19    D(++)+D(-)-->N*(+)(1535)+n
*                        20    D(+)+D(+)-->N*(+)(1535)+p
*                        21    D(+)+D(0)-->N*(+)(1535)+n
*                        22    D(+)+D(0)-->N*(0)(1535)+p
*                        23    D(+)+D(-)-->N*(0)(1535)+n
*                        24    D(0)+D(0)-->N*(0)(1535)+n
*                          25    N*(+)(14)+N*(+)(14)-->N*(+)(15)+p
*                          26    N*(0)(14)+N*(0)(14)-->N*(0)(15)+n
*                          27    N*(+)(14)+N*(0)(14)-->N*(+)(15)+n
*                        28    N*(+)(14)+N*(0)(14)-->N*(0)(15)+p
*                        29    N*(+)(14)+D+-->N*(+)(15)+p
*                        30    N*(+)(14)+D0-->N*(+)(15)+n
*                        31    N*(+)(14)+D(-)-->N*(0)(1535)+n
*                        32    N*(0)(14)+D++--->N*(+)(15)+p
*                        33    N*(0)(14)+D+--->N*(+)(15)+n
*                        34    N*(0)(14)+D+--->N*(0)(15)+p
*                        35    N*(0)(14)+D0-->N*(0)(15)+n
*                        36    N*(+)(14)+D0--->N*(0)(15)+p
*                        +++
*               AND MORE CHANNELS AS LISTED IN THE NOTE BOOK      
*
* NOTE ABOUT N*(1440) RESORANCE:                                       *
*     As it has been discussed in VerWest's paper,I= 1 (initial isospin)
*     channel can all be attributed to delta resorance while I= 0      *
*     channel can all be  attribured to N* resorance.Only in n+p       *
*     one can have I=0 channel so is the N*(1440) resorance            *
* REFERENCES:    J. CUGNON ET AL., NUCL. PHYS. A352, 505 (1981)        *
*                    Y. KITAZOE ET AL., PHYS. LETT. 166B, 35 (1986)    *
*                    B. VerWest el al., PHYS. PRV. C25 (1982)1979      *
*                    Gy. Wolf  et al, Nucl Phys A517 (1990) 615        *
*                    CUTOFF = 2 * AVMASS + 20 MEV                      *
*                                                                      *
*       for N*(1535) we use the parameterization by Gy. Wolf et al     *
*       Nucl phys A552 (1993) 349, added May 18, 1994                  *
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AKA=0.498,APHI=1.020,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
        common /gg/ dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
        COMMON /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
        COMMON /NN/NNN
cc      SAVE /NN/
        COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
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
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      SAVE   
*-----------------------------------------------------------------------
       XINEL=0
       SIGK=0
       XSK1=0
       XSK2=0
       XSK3=0
       XSK4=0
       XSK5=0
        EM1=E(I1)
        EM2=E(I2)
      PR  = SQRT( PX**2 + PY**2 + PZ**2 )
*     IF THERE WERE 2 N*(1535) AND THEY DIDN'T SCATT. ELAST., 
*     ALLOW THEM TO PRODUCE KAONS. NO OTHER INELASTIC CHANNELS
*     ARE KNOWN
C       if((lb(i1).ge.12).and.(lb(i2).ge.12))return
*     ALL the inelastic collisions between N*(1535) and Delta as well
*     as N*(1440) TO PRODUCE KAONS, NO OTHER CHANNELS ARE KNOWN
C       if((lb(i1).ge.12).and.(lb(i2).ge.3))return
C       if((lb(i2).ge.12).and.(lb(i1).ge.3))return
*     calculate the N*(1535) production cross section in I1+I2 collisions
       call N1535(iabs(lb(i1)),iabs(lb(i2)),srt,X1535)
c
* for Delta+Delta-->N*(1440 OR 1535)+N AND N*(1440)+N*(1440)-->N*(1535)+X 
*     AND DELTA+N*(1440)-->N*(1535)+X
* WE ASSUME THEY HAVE THE SAME CROSS SECTIONS as CORRESPONDING N+N COLLISION):
* FOR D++D0, D+D+,D+D-,D0D0,N*+N*+,N*0N*0,N*(+)D+,N*(+)D(-),N*(0)D(0)
* N*(1535) production, kaon production and reabsorption through 
* D(N*)+D(N*)-->NN are ALLOWED.
* CROSS SECTION FOR KAON PRODUCTION from the four channels are
* for NLK channel
       akp=0.498
       ak0=0.498
       ana=0.94
       ada=1.232
       al=1.1157
       as=1.1197
       xsk1=0
       xsk2=0
       xsk3=0
       xsk4=0
       t1nlk=ana+al+akp
       if(srt.le.t1nlk)go to 222
       XSK1=1.5*PPLPK(SRT)
* for DLK channel
       t1dlk=ada+al+akp
       t2dlk=ada+al-akp
       if(srt.le.t1dlk)go to 222
       es=srt
       pmdlk2=(es**2-t1dlk**2)*(es**2-t2dlk**2)/(4.*es**2)
       pmdlk=sqrt(pmdlk2)
       XSK3=1.5*PPLPK(srt)
* for NSK channel
       t1nsk=ana+as+akp
       t2nsk=ana+as-akp
       if(srt.le.t1nsk)go to 222
       pmnsk2=(es**2-t1nsk**2)*(es**2-t2nsk**2)/(4.*es**2)
       pmnsk=sqrt(pmnsk2)
       XSK2=1.5*(PPK1(srt)+PPK0(srt))
* for DSK channel
       t1DSk=aDa+aS+akp
       t2DSk=aDa+aS-akp
       if(srt.le.t1dsk)go to 222
       pmDSk2=(es**2-t1DSk**2)*(es**2-t2DSk**2)/(4.*es**2)
       pmDSk=sqrt(pmDSk2)
       XSK4=1.5*(PPK1(srt)+PPK0(srt))
csp11/21/01
c phi production
       if(srt.le.(2.*amn+aphi))go to 222
c  !! mb put the correct form
         xsk5 = 0.0001
csp11/21/01 end
* THE TOTAL KAON+ PRODUCTION CROSS SECTION IS THEN
222       SIGK=XSK1+XSK2+XSK3+XSK4

cbz3/7/99 neutralk
        XSK1 = 2.0 * XSK1
        XSK2 = 2.0 * XSK2
        XSK3 = 2.0 * XSK3
        XSK4 = 2.0 * XSK4
        SIGK = 2.0 * SIGK + xsk5
cbz3/7/99 neutralk end

        IDD=iabs(LB(I1)*LB(I2))
* The reabsorption cross section for the process
* D(N*)D(N*)-->NN is
       s2d=reab2d(i1,i2,srt)

cbz3/16/99 pion
        S2D = 0.
cbz3/16/99 pion end

*(1) N*(1535)+D(N*(1440)) reactions
*    we allow kaon production and reabsorption only
       if(((iabs(lb(i1)).ge.12).and.(iabs(lb(i2)).ge.12)).OR.
     &       ((iabs(lb(i1)).ge.12).and.(iabs(lb(i2)).ge.6)).OR.
     &       ((iabs(lb(i2)).ge.12).and.(iabs(lb(i1)).ge.6)))THEN
       XINEL=sigk+s2d
       RETURN
       ENDIF
* channels have the same charge as pp 
        IF((IDD.EQ.63).OR.(IDD.EQ.64).OR.(IDD.EQ.48).
     1  OR.(IDD.EQ.49).OR.(IDD.EQ.11*11).OR.(IDD.EQ.10*10).
     2  OR.(IDD.EQ.88).OR.(IDD.EQ.66).
     3  OR.(IDD.EQ.90).OR.(IDD.EQ.70))THEN
        XINEL=X1535+SIGK+s2d
       RETURN
        ENDIF
* IN DELTA+N*(1440) and N*(1440)+N*(1440) COLLISIONS, 
* N*(1535), kaon production and reabsorption are ALLOWED
* IN N*(1440)+N*(1440) COLLISIONS, ONLY N*(1535) IS ALLOWED
       IF((IDD.EQ.110).OR.(IDD.EQ.77).OR.(IDD.EQ.80))THEN
       XINEL=X1535+SIGK+s2d
       RETURN
       ENDIF       
       IF((IDD.EQ.54).OR.(IDD.EQ.56))THEN
* LIKE FOR N+P COLLISION, 
* IN DELTA+DELTA COLLISIONS BOTH N*(1440) AND N*(1535) CAN BE PRODUCED
        SIG2=(3./4.)*SIGMA(SRT,2,0,1)
        XINEL=2.*(SIG2+X1535)+SIGK+s2d
       RETURN
       ENDIF
       RETURN
       END
******************************************
      real function dirct1(srt)
*  This function contains the experimental, direct pion(+) + p cross sections *
*  srt    = DSQRT(s) in GeV                                                   *
*  dirct1  = cross section in fm**2                                     *
*  earray = EXPerimental table with the srt            
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
******************************************
c      real*4   xarray(122), earray(122)
      real   xarray(122), earray(122)
      SAVE   
      data   earray /
     &1.568300,1.578300,1.588300,1.598300,1.608300,1.618300,1.628300,    
     &1.638300,1.648300,1.658300,1.668300,1.678300,1.688300,1.698300,    
     &1.708300,1.718300,1.728300,1.738300,1.748300,1.758300,1.768300,    
     &1.778300,1.788300,1.798300,1.808300,1.818300,1.828300,1.838300,    
     &1.848300,1.858300,1.868300,1.878300,1.888300,1.898300,1.908300,    
     &1.918300,1.928300,1.938300,1.948300,1.958300,1.968300,1.978300,    
     &1.988300,1.998300,2.008300,2.018300,2.028300,2.038300,2.048300,    
     &2.058300,2.068300,2.078300,2.088300,2.098300,2.108300,2.118300,    
     &2.128300,2.138300,2.148300,2.158300,2.168300,2.178300,2.188300,    
     &2.198300,2.208300,2.218300,2.228300,2.238300,2.248300,2.258300,    
     &2.268300,2.278300,2.288300,2.298300,2.308300,2.318300,2.328300,    
     &2.338300,2.348300,2.358300,2.368300,2.378300,2.388300,2.398300,    
     &2.408300,2.418300,2.428300,2.438300,2.448300,2.458300,2.468300,    
     &2.478300,2.488300,2.498300,2.508300,2.518300,2.528300,2.538300,    
     &2.548300,2.558300,2.568300,2.578300,2.588300,2.598300,2.608300,    
     &2.618300,2.628300,2.638300,2.648300,2.658300,2.668300,2.678300,
     &2.688300,2.698300,2.708300,2.718300,2.728300,2.738300,2.748300,    
     &2.758300,2.768300,2.778300/
      data xarray/
     &1.7764091E-02,0.5643668,0.8150568,1.045565,2.133695,3.327922,
     &4.206488,3.471242,4.486876,5.542213,6.800052,7.192446,6.829848,    
     &6.580306,6.868410,8.527946,10.15720,9.716511,9.298335,8.901310,    
     &10.31213,10.52185,11.17630,11.61639,12.05577,12.71596,13.46036,    
     &14.22060,14.65449,14.94775,14.93310,15.32907,16.56481,16.29422,    
     &15.18548,14.12658,13.72544,13.24488,13.31003,14.42680,12.84423,    
     &12.49025,12.14858,11.81870,11.18993,11.35816,11.09447,10.83873,    
     &10.61592,10.53754,9.425521,8.195912,9.661075,9.696192,9.200142,    
     &8.953734,8.715461,8.484999,8.320765,8.255512,8.190969,8.127125,    
     &8.079508,8.073004,8.010611,7.948909,7.887895,7.761005,7.626290,    
     &7.494696,7.366132,7.530178,8.392097,9.046881,8.962544,8.879403,    
     &8.797427,8.716601,8.636904,8.558312,8.404368,8.328978,8.254617,    
     &8.181265,8.108907,8.037527,7.967100,7.897617,7.829057,7.761405,    
     &7.694647,7.628764,7.563742,7.499570,7.387562,7.273281,7.161334,    
     &6.973375,6.529592,6.280323,6.293136,6.305725,6.318097,6.330258,    
     &6.342214,6.353968,6.365528,6.376895,6.388079,6.399081,6.409906,    
     &6.420560,6.431045,6.441367,6.451529,6.461533,6.471386,6.481091,    
     &6.490650,6.476413,6.297259,6.097826/

      dirct1=0
      if (srt .lt. earray(1)) then
        dirct1 = 0.00001
        return
      end if
      if (srt .gt. earray(122)) then
        dirct1 = xarray(122)
       dirct1=dirct1/10.
        return
      end if
*
*Interpolate double logarithmically to find xdirct2(srt)
*
      do 1001 ie = 1,122
        if (earray(ie) .eq. srt) then
          dirct1= xarray(ie)
         dirct1=dirct1/10.
          return
       endif
        if (earray(ie) .gt. srt) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          dirct1= exp(ymin + (alog(srt)-xmin)
     &          *(ymax-ymin)/(xmax-xmin) )
       dirct1=dirct1/10.
       go to 50
        end if
 1001 continue
50       continue
        return
        END
*******************************
******************************************
      real function dirct2(srt)
*  This function contains the experimental, direct pion(-) + p cross sections *
*  srt    = DSQRT(s) in GeV                                                   *
*  dirct2 = cross section in fm**2
*  earray = EXPerimental table with the srt            
*  xarray = EXPerimental table with cross sections in mb (curve to guide eye) *
******************************************
c      real*4   xarray(122), earray(122)
      real   xarray(122), earray(122)
      SAVE   
      data   earray /
     &1.568300,1.578300,1.588300,1.598300,1.608300,1.618300,1.628300,    
     &1.638300,1.648300,1.658300,1.668300,1.678300,1.688300,1.698300,    
     &1.708300,1.718300,1.728300,1.738300,1.748300,1.758300,1.768300,    
     &1.778300,1.788300,1.798300,1.808300,1.818300,1.828300,1.838300,    
     &1.848300,1.858300,1.868300,1.878300,1.888300,1.898300,1.908300,    
     &1.918300,1.928300,1.938300,1.948300,1.958300,1.968300,1.978300,    
     &1.988300,1.998300,2.008300,2.018300,2.028300,2.038300,2.048300,    
     &2.058300,2.068300,2.078300,2.088300,2.098300,2.108300,2.118300,    
     &2.128300,2.138300,2.148300,2.158300,2.168300,2.178300,2.188300,    
     &2.198300,2.208300,2.218300,2.228300,2.238300,2.248300,2.258300,    
     &2.268300,2.278300,2.288300,2.298300,2.308300,2.318300,2.328300,    
     &2.338300,2.348300,2.358300,2.368300,2.378300,2.388300,2.398300,    
     &2.408300,2.418300,2.428300,2.438300,2.448300,2.458300,2.468300,    
     &2.478300,2.488300,2.498300,2.508300,2.518300,2.528300,2.538300,    
     &2.548300,2.558300,2.568300,2.578300,2.588300,2.598300,2.608300,    
     &2.618300,2.628300,2.638300,2.648300,2.658300,2.668300,2.678300,
     &2.688300,2.698300,2.708300,2.718300,2.728300,2.738300,2.748300,    
     &2.758300,2.768300,2.778300/
      data xarray/0.5773182,1.404156,2.578629,3.832013,4.906011,
     &9.076963,13.10492,10.65975,15.31156,19.77611,19.92874,18.68979,    
     &19.80114,18.39536,14.34269,13.35353,13.58822,14.57031,10.24686,    
     &11.23386,9.764803,10.35652,10.53539,10.07524,9.582198,9.596469,    
     &9.818489,9.012848,9.378012,9.529244,9.529698,8.835624,6.671396,    
     &8.797758,8.133437,7.866227,7.823946,7.808504,7.791755,7.502062,    
     &7.417275,7.592349,7.752028,7.910585,8.068122,8.224736,8.075289,    
     &7.895902,7.721359,7.551512,7.386224,7.225343,7.068739,6.916284,    
     &6.767842,6.623294,6.482520,6.345404,6.211833,7.339510,7.531462,    
     &7.724824,7.919620,7.848021,7.639856,7.571083,7.508881,7.447474,    
     &7.386855,7.327011,7.164454,7.001266,6.842526,6.688094,6.537823,    
     &6.391583,6.249249,6.110689,5.975790,5.894200,5.959503,6.024602,    
     &6.089505,6.154224,6.218760,6.283128,6.347331,6.297411,6.120248,    
     &5.948606,6.494864,6.357106,6.222824,6.091910,5.964267,5.839795,    
     &5.718402,5.599994,5.499146,5.451325,5.404156,5.357625,5.311721,    
     &5.266435,5.301964,5.343963,5.385833,5.427577,5.469200,5.510702,    
     &5.552088,5.593359,5.634520,5.675570,5.716515,5.757356,5.798093,    
     &5.838732,5.879272,5.919717,5.960068,5.980941/

      dirct2=0.
      if (srt .lt. earray(1)) then
        dirct2 = 0.00001
        return
      end if
      if (srt .gt. earray(122)) then
        dirct2 = xarray(122)
       dirct2=dirct2/10.
        return
      end if
*
*Interpolate double logarithmically to find xdirct2(srt)
*
      do 1001 ie = 1,122
        if (earray(ie) .eq. srt) then
          dirct2= xarray(ie)
         dirct2=dirct2/10.
          return
       endif
        if (earray(ie) .gt. srt) then
          ymin = alog(xarray(ie-1))
          ymax = alog(xarray(ie))
          xmin = alog(earray(ie-1))
          xmax = alog(earray(ie))
          dirct2= exp(ymin + (alog(srt)-xmin)
     &          *(ymax-ymin)/(xmax-xmin) )
       dirct2=dirct2/10.
       go to 50
        end if
 1001 continue
50       continue
        return
        END
*******************************
******************************
* this program calculates the elastic cross section for rho+nucleon
* through higher resonances
c       real*4 function ErhoN(em1,em2,lb1,lb2,srt)
       real function ErhoN(em1,em2,lb1,lb2,srt)
* date : Dec. 19, 1994
* ****************************
c       implicit real*4 (a-h,o-z)
      dimension   arrayj(19),arrayl(19),arraym(19),
     &arrayw(19),arrayb(19)
      SAVE   
      data arrayj /0.5,1.5,0.5,0.5,2.5,2.5,1.5,0.5,1.5,3.5,
     &1.5,0.5,1.5,0.5,2.5,0.5,1.5,2.5,3.5/
      data arrayl/1,2,0,0,2,3,2,1,1,3,
     &1,0,2,0,3,1,1,2,3/
      data arraym /1.44,1.52,1.535,1.65,1.675,1.68,1.70,1.71,
     &1.72,1.99,1.60,1.62,1.70,1.90,1.905,1.910,
     &1.86,1.93,1.95/
      data arrayw/0.2,0.125,0.15,0.15,0.155,0.125,0.1,0.11,
     &0.2,0.29,0.25,0.16,0.28,0.15,0.3,0.22,0.25,
     &0.25,0.24/
      data arrayb/0.15,0.20,0.05,0.175,0.025,0.125,0.1,0.20,
     &0.53,0.34,0.05,0.07,0.15,0.45,0.45,0.058,
     &0.08,0.12,0.08/

* the minimum energy for pion+delta collision
       pi=3.1415926
       xs=0
* include contribution from each resonance
       do 1001 ir=1,19
cbz11/25/98
       IF(IR.LE.8)THEN
c       if(lb1*lb2.eq.27.OR.LB1*LB2.EQ.25*2)branch=0.
c       if(lb1*lb2.eq.26.OR.LB1*LB2.EQ.26*2)branch=1./3.
c       if(lb1*lb2.eq.27*2.OR.LB1*LB2.EQ.25)branch=2./3.
c       ELSE
c       if(lb1*lb2.eq.27.OR.LB1*LB2.EQ.25*2)branch=1.
c       if(lb1*lb2.eq.26.OR.LB1*LB2.EQ.26*2)branch=2./3.
c       if(lb1*lb2.eq.27*2.OR.LB1*LB2.EQ.25)branch=1./3.
c       ENDIF
       if( ((lb1*lb2.eq.27.AND.(LB1.EQ.1.OR.LB2.EQ.1)).OR.
     &     (LB1*LB2.EQ.25*2.AND.(LB1.EQ.2.OR.LB2.EQ.2)))
     &       .OR.((lb1*lb2.eq.-25.AND.(LB1.EQ.-1.OR.LB2.EQ.-1)).OR.
     &     (LB1*LB2.EQ.-27*2.AND.(LB1.EQ.-2.OR.LB2.EQ.-2))) )
     &     branch=0.
        if((iabs(lb1*lb2).eq.26.AND.(iabs(LB1).EQ.1.OR.iabs(LB2).EQ.1))
     &   .OR.(iabs(LB1*LB2).EQ.26*2
     &   .AND.(iabs(LB1).EQ.2.OR.iabs(LB2).EQ.2)))
     &     branch=1./3.
       if( ((lb1*lb2.eq.27*2.AND.(LB1.EQ.2.OR.LB2.EQ.2)).OR.
     &     (LB1*LB2.EQ.25.AND.(LB1.EQ.1.OR.LB2.EQ.1)))
     &  .OR.((lb1*lb2.eq.-25*2.AND.(LB1.EQ.-2.OR.LB2.EQ.-2)).OR.
     &     (LB1*LB2.EQ.-27.AND.(LB1.EQ.-1.OR.LB2.EQ.-1))) )
     &     branch=2./3.
       ELSE
       if( ((lb1*lb2.eq.27.AND.(LB1.EQ.1.OR.LB2.EQ.1)).OR.
     &     (LB1*LB2.EQ.25*2.AND.(LB1.EQ.2.OR.LB2.EQ.2)))
     &       .OR.((lb1*lb2.eq.-25.AND.(LB1.EQ.-1.OR.LB2.EQ.-1)).OR.
     &     (LB1*LB2.EQ.-27*2.AND.(LB1.EQ.-2.OR.LB2.EQ.-2))) )
     &     branch=1.
        if((iabs(lb1*lb2).eq.26.AND.(iabs(LB1).EQ.1.OR.iabs(LB2).EQ.1))
     &   .OR.(iabs(LB1*LB2).EQ.26*2
     &   .AND.(iabs(LB1).EQ.2.OR.iabs(LB2).EQ.2)))
     &     branch=2./3.
       if( ((lb1*lb2.eq.27*2.AND.(LB1.EQ.2.OR.LB2.EQ.2)).OR.
     &     (LB1*LB2.EQ.25.AND.(LB1.EQ.1.OR.LB2.EQ.1)))
     &  .OR.((lb1*lb2.eq.-25*2.AND.(LB1.EQ.-2.OR.LB2.EQ.-2)).OR.
     &     (LB1*LB2.EQ.-27.AND.(LB1.EQ.-1.OR.LB2.EQ.-1))) )
     &     branch=1./3.
       ENDIF
cbz11/25/98end
       xs0=fdR(arraym(ir),arrayj(ir),arrayl(ir),
     &arrayw(ir),arrayb(ir),srt,EM1,EM2)
       xs=xs+1.3*pi*branch*xs0*(0.1973)**2
 1001 continue
       Erhon=xs
       return
       end
***************************8
*FUNCTION FDE(DMASS) GIVES DELTA MASS DISTRIBUTION BY USING OF
*KITAZOE'S FORMULA
c        REAL*4 FUNCTION FDR(DMASS,aj,al,width,widb0,srt,em1,em2)
      REAL FUNCTION FDR(DMASS,aj,al,width,widb0,srt,em1,em2)
      SAVE   
        AMd=em1
        AmP=em2
           Ak02= 0.25*(DMASS**2-amd**2-amp**2)**2
     &           -(Amp*amd)**2
            IF (ak02 .GT. 0.) THEN
              Q0 = SQRT(ak02/DMASS)
            ELSE
              Q0= 0.0
             fdR=0
           return
            END IF
           Ak2= 0.25*(srt**2-amd**2-amp**2)**2
     &           -(Amp*amd)**2
            IF (ak2 .GT. 0.) THEN
              Q = SQRT(ak2/DMASS)
            ELSE
              Q= 0.00
             fdR=0
             return
            END IF
       b=widb0*1.2*dmass/srt*(q/q0)**(2.*al+1)
     &  /(1.+0.2*(q/q0)**(2*al))
        FDR=(2.*aj+1)*WIDTH**2*b/((srt-dmass)**2
     1  +0.25*WIDTH**2)/(6.*q**2)
        RETURN
        END
******************************
* this program calculates the elastic cross section for pion+delta
* through higher resonances
c       REAL*4 FUNCTION DIRCT3(SRT)
      REAL FUNCTION DIRCT3(SRT)
* date : Dec. 19, 1994
* ****************************
c     implicit real*4 (a-h,o-z)
      dimension   arrayj(17),arrayl(17),arraym(17),
     &arrayw(17),arrayb(17)
      SAVE   
      data arrayj /1.5,0.5,2.5,2.5,1.5,0.5,1.5,3.5,
     &1.5,0.5,1.5,0.5,2.5,0.5,1.5,2.5,3.5/
      data arrayl/2,0,2,3,2,1,1,3,
     &1,0,2,0,3,1,1,2,3/
      data arraym /1.52,1.65,1.675,1.68,1.70,1.71,
     &1.72,1.99,1.60,1.62,1.70,1.90,1.905,1.910,
     &1.86,1.93,1.95/
      data arrayw/0.125,0.15,0.155,0.125,0.1,0.11,
     &0.2,0.29,0.25,0.16,0.28,0.15,0.3,0.22,0.25,
     &0.25,0.24/
      data arrayb/0.55,0.6,0.375,0.6,0.1,0.15,
     &0.15,0.05,0.35,0.3,0.15,0.1,0.1,0.22,
     &0.2,0.09,0.4/

* the minimum energy for pion+delta collision
       pi=3.1415926
       amn=0.938
       amp=0.138
       xs=0
* include contribution from each resonance
       branch=1./3.
       do 1001 ir=1,17
       if(ir.gt.8)branch=2./3.
       xs0=fd1(arraym(ir),arrayj(ir),arrayl(ir),
     &arrayw(ir),arrayb(ir),srt)
       xs=xs+1.3*pi*branch*xs0*(0.1973)**2
 1001   continue
       DIRCT3=XS
       RETURN
       end
***************************8
*FUNCTION FDE(DMASS) GIVES DELTA MASS DISTRIBUTION BY USING OF
*KITAZOE'S FORMULA
c        REAL*4 FUNCTION FD1(DMASS,aj,al,width,widb0,srt)
      REAL FUNCTION FD1(DMASS,aj,al,width,widb0,srt)
      SAVE   
        AMN=0.938
        AmP=0.138
       amd=amn
           Ak02= 0.25*(DMASS**2-amd**2-amp**2)**2
     &           -(Amp*amd)**2
            IF (ak02 .GT. 0.) THEN
              Q0 = SQRT(ak02/DMASS)
            ELSE
              Q0= 0.0
             fd1=0
           return
            END IF
           Ak2= 0.25*(srt**2-amd**2-amp**2)**2
     &           -(Amp*amd)**2
            IF (ak2 .GT. 0.) THEN
              Q = SQRT(ak2/DMASS)
            ELSE
              Q= 0.00
             fd1=0
             return
            END IF
       b=widb0*1.2*dmass/srt*(q/q0)**(2.*al+1)
     &  /(1.+0.2*(q/q0)**(2*al))
        FD1=(2.*aj+1)*WIDTH**2*b/((srt-dmass)**2
     1  +0.25*WIDTH**2)/(2.*q**2)
        RETURN
        END
******************************
* this program calculates the elastic cross section for pion+delta
* through higher resonances
c       REAL*4 FUNCTION DPION(EM1,EM2,LB1,LB2,SRT)
      REAL FUNCTION DPION(EM1,EM2,LB1,LB2,SRT)
* date : Dec. 19, 1994
* ****************************
c     implicit real*4 (a-h,o-z)
      dimension   arrayj(19),arrayl(19),arraym(19),
     &arrayw(19),arrayb(19)
      SAVE   
      data arrayj /0.5,1.5,0.5,0.5,2.5,2.5,1.5,0.5,1.5,3.5,
     &1.5,0.5,1.5,0.5,2.5,0.5,1.5,2.5,3.5/
      data arrayl/1,2,0,0,2,3,2,1,1,3,
     &1,0,2,0,3,1,1,2,3/
      data arraym /1.44,1.52,1.535,1.65,1.675,1.68,1.70,1.71,
     &1.72,1.99,1.60,1.62,1.70,1.90,1.905,1.910,
     &1.86,1.93,1.95/
      data arrayw/0.2,0.125,0.15,0.15,0.155,0.125,0.1,0.11,
     &0.2,0.29,0.25,0.16,0.28,0.15,0.3,0.22,0.25,
     &0.25,0.24/
      data arrayb/0.15,0.25,0.,0.05,0.575,0.125,0.379,0.10,
     &0.10,0.062,0.45,0.60,0.6984,0.05,0.25,0.089,
     &0.19,0.2,0.13/

* the minimum energy for pion+delta collision
       pi=3.1415926
       amn=0.94
       amp=0.14
       xs=0
* include contribution from each resonance
       do 1001 ir=1,19
       BRANCH=0.
cbz11/25/98
       if(ir.LE.8)THEN
c       IF(LB1*LB2.EQ.5*7.OR.LB1*LB2.EQ.3*8)branch=1./6.
c       IF(LB1*LB2.EQ.4*7.OR.LB1*LB2.EQ.4*8)branch=1./3.
c       IF(LB1*LB2.EQ.5*6.OR.LB1*LB2.EQ.3*9)branch=1./2.
c       ELSE
c       IF(LB1*LB2.EQ.5*8.OR.LB1*LB2.EQ.5*6)branch=2./5.
c       IF(LB1*LB2.EQ.3*9.OR.LB1*LB2.EQ.3*7)branch=2./5.
c       IF(LB1*LB2.EQ.5*7.OR.LB1*LB2.EQ.3*8)branch=8./15.
c       IF(LB1*LB2.EQ.4*7.OR.LB1*LB2.EQ.4*8)branch=1./15.
c       IF(LB1*LB2.EQ.4*9.OR.LB1*LB2.EQ.4*6)branch=3./5.
c       ENDIF
       IF( ((LB1*LB2.EQ.5*7.AND.(LB1.EQ.5.OR.LB2.EQ.5)).OR.
     &     (LB1*LB2.EQ.3*8.AND.(LB1.EQ.3.OR.LB2.EQ.3)))
     &       .OR.((LB1*LB2.EQ.-3*7.AND.(LB1.EQ.3.OR.LB2.EQ.3)).OR.
     &     (LB1*LB2.EQ.-5*8.AND.(LB1.EQ.5.OR.LB2.EQ.5))) )
     &     branch=1./6.
       IF((iabs(LB1*LB2).EQ.4*7.AND.(LB1.EQ.4.OR.LB2.EQ.4)).OR.
     &     (iabs(LB1*LB2).EQ.4*8.AND.(LB1.EQ.4.OR.LB2.EQ.4)))
     &     branch=1./3.
       IF( ((LB1*LB2.EQ.5*6.AND.(LB1.EQ.5.OR.LB2.EQ.5)).OR.
     &     (LB1*LB2.EQ.3*9.AND.(LB1.EQ.3.OR.LB2.EQ.3)))
     &       .OR.((LB1*LB2.EQ.-3*6.AND.(LB1.EQ.3.OR.LB2.EQ.3)).OR.
     &     (LB1*LB2.EQ.-5*9.AND.(LB1.EQ.5.OR.LB2.EQ.5))) )
     &     branch=1./2.
       ELSE
       IF( ((LB1*LB2.EQ.5*8.AND.(LB1.EQ.5.OR.LB2.EQ.5)).OR.
     &     (LB1*LB2.EQ.5*6.AND.(LB1.EQ.5.OR.LB2.EQ.5)))
     &        .OR.((LB1*LB2.EQ.-3*8.AND.(LB1.EQ.3.OR.LB2.EQ.3)).OR.
     &     (LB1*LB2.EQ.-3*6.AND.(LB1.EQ.3.OR.LB2.EQ.3))) )
     &     branch=2./5.
       IF( ((LB1*LB2.EQ.3*9.AND.(LB1.EQ.3.OR.LB2.EQ.3)).OR.
     &     (LB1*LB2.EQ.3*7.AND.(LB1.EQ.3.OR.LB2.EQ.3)))
     &        .OR. ((LB1*LB2.EQ.-5*9.AND.(LB1.EQ.5.OR.LB2.EQ.5)).OR.
     &     (LB1*LB2.EQ.-5*7.AND.(LB1.EQ.5.OR.LB2.EQ.5))) )
     &     branch=2./5.
       IF( ((LB1*LB2.EQ.5*7.AND.(LB1.EQ.5.OR.LB2.EQ.5)).OR.
     &     (LB1*LB2.EQ.3*8.AND.(LB1.EQ.3.OR.LB2.EQ.3)))
     &        .OR.((LB1*LB2.EQ.-3*7.AND.(LB1.EQ.3.OR.LB2.EQ.3)).OR.
     &     (LB1*LB2.EQ.-5*8.AND.(LB1.EQ.5.OR.LB2.EQ.5))) )
     &     branch=8./15.
       IF((iabs(LB1*LB2).EQ.4*7.AND.(LB1.EQ.4.OR.LB2.EQ.4)).OR.
     &     (iabs(LB1*LB2).EQ.4*8.AND.(LB1.EQ.4.OR.LB2.EQ.4)))
     &     branch=1./15.
       IF((iabs(LB1*LB2).EQ.4*9.AND.(LB1.EQ.4.OR.LB2.EQ.4)).OR.
     &     (iabs(LB1*LB2).EQ.4*6.AND.(LB1.EQ.4.OR.LB2.EQ.4)))
     &     branch=3./5.
       ENDIF
cbz11/25/98end
       xs0=fd2(arraym(ir),arrayj(ir),arrayl(ir),
     &arrayw(ir),arrayb(ir),EM1,EM2,srt)
       xs=xs+1.3*pi*branch*xs0*(0.1973)**2
 1001   continue
       DPION=XS
       RETURN
       end
***************************8
*FUNCTION FDE(DMASS) GIVES DELTA MASS DISTRIBUTION BY USING OF
*KITAZOE'S FORMULA
c        REAL*4 FUNCTION FD2(DMASS,aj,al,width,widb0,EM1,EM2,srt)
      REAL FUNCTION FD2(DMASS,aj,al,width,widb0,EM1,EM2,srt)
      SAVE   
        AmP=EM1
       amd=EM2
           Ak02= 0.25*(DMASS**2-amd**2-amp**2)**2
     &           -(Amp*amd)**2
            IF (ak02 .GT. 0.) THEN
              Q0 = SQRT(ak02/DMASS)
            ELSE
              Q0= 0.0
             fd2=0
           return
            END IF
           Ak2= 0.25*(srt**2-amd**2-amp**2)**2
     &           -(Amp*amd)**2
            IF (ak2 .GT. 0.) THEN
              Q = SQRT(ak2/DMASS)
            ELSE
              Q= 0.00
             fd2=0
             return
            END IF
       b=widb0*1.2*dmass/srt*(q/q0)**(2.*al+1)
     &  /(1.+0.2*(q/q0)**(2*al))
        FD2=(2.*aj+1)*WIDTH**2*b/((srt-dmass)**2
     1  +0.25*WIDTH**2)/(4.*q**2)
        RETURN
        END
***************************8
*   MASS GENERATOR for two resonances simultaneously
       subroutine Rmasdd(srt,am10,am20,
     &dmin1,dmin2,ISEED,ic,dm1,dm2)
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
       ISEED=ISEED
       amn=0.94
       amp=0.14
* the maximum mass for resonance 1
         dmax1=srt-dmin2
* generate the mass for the first resonance
 5        NTRY1=0
         ntry2=0
         ntry=0
         ictrl=0
10        DM1 = RANART(NSEED) * (DMAX1-DMIN1) + DMIN1
          NTRY1=NTRY1+1
* the maximum mass for resonance 2 
         if(ictrl.eq.0)dmax2=srt-dm1
* generate the mass for the second resonance
20         dm2=RANART(NSEED)*(dmax2-dmin2)+dmin2
          NTRY2=NTRY2+1
* check the energy-momentum conservation with two masses
* q2 in the following is q**2*4*srt**2
         q2=((srt**2-dm1**2-dm2**2)**2-4.*dm1**2*dm2**2)
         if(q2.le.0)then
         dmax2=dm2-0.01
c         dmax1=dm1-0.01
         ictrl=1
         go to 20
         endif
* determine the weight of the mass pair         
          IF(DMAX1.LT.am10) THEN
          if(ic.eq.1)FM1=Fmassd(DMAX1)
          if(ic.eq.2)FM1=Fmassn(DMAX1)
          if(ic.eq.3)FM1=Fmassd(DMAX1)
          if(ic.eq.4)FM1=Fmassd(DMAX1)
          ELSE
          if(ic.eq.1)FM1=Fmassd(am10)
          if(ic.eq.2)FM1=Fmassn(am10)
          if(ic.eq.3)FM1=Fmassd(am10)
          if(ic.eq.4)FM1=Fmassd(am10)
          ENDIF
          IF(DMAX2.LT.am20) THEN
          if(ic.eq.1)FM2=Fmassd(DMAX2)
          if(ic.eq.2)FM2=Fmassn(DMAX2)
          if(ic.eq.3)FM2=Fmassn(DMAX2)
          if(ic.eq.4)FM2=Fmassr(DMAX2)
          ELSE
          if(ic.eq.1)FM2=Fmassd(am20)
          if(ic.eq.2)FM2=Fmassn(am20)
          if(ic.eq.3)FM2=Fmassn(am20)
          if(ic.eq.4)FM2=Fmassr(am20)
          ENDIF
          IF(FM1.EQ.0.)FM1=1.e-04
          IF(FM2.EQ.0.)FM2=1.e-04
         prob0=fm1*fm2
          if(ic.eq.1)prob=Fmassd(dm1)*fmassd(dm2)
          if(ic.eq.2)prob=Fmassn(dm1)*fmassn(dm2)
          if(ic.eq.3)prob=Fmassd(dm1)*fmassn(dm2)
          if(ic.eq.4)prob=Fmassd(dm1)*fmassr(dm2)
         if(prob.le.1.e-06)prob=1.e-06
         fff=prob/prob0
         ntry=ntry+1 
          IF(RANART(NSEED).GT.fff.AND.
     1    NTRY.LE.20) GO TO 10

clin-2/26/03 limit the mass of (rho,Delta,N*1440) below a certain value
c     (here taken as its central value + 2* B-W fullwidth):
          if((abs(am10-0.77).le.0.01.and.dm1.gt.1.07)
     1         .or.(abs(am10-1.232).le.0.01.and.dm1.gt.1.47)
     2         .or.(abs(am10-1.44).le.0.01.and.dm1.gt.2.14)) goto 5
          if((abs(am20-0.77).le.0.01.and.dm2.gt.1.07)
     1         .or.(abs(am20-1.232).le.0.01.and.dm2.gt.1.47)
     2         .or.(abs(am20-1.44).le.0.01.and.dm2.gt.2.14)) goto 5

       RETURN
       END
*FUNCTION Fmassd(DMASS) GIVES the delta MASS DISTRIBUTION 
        REAL FUNCTION Fmassd(DMASS)
      SAVE   
        AM0=1.232
        Fmassd=am0*WIDTH(DMASS)/((DMASS**2-am0**2)**2
     1  +am0**2*WIDTH(DMASS)**2)
        RETURN
        END
*FUNCTION Fmassn(DMASS) GIVES the N* MASS DISTRIBUTION 
        REAL FUNCTION Fmassn(DMASS)
      SAVE   
        AM0=1.44
        Fmassn=am0*W1440(DMASS)/((DMASS**2-am0**2)**2
     1  +am0**2*W1440(DMASS)**2)
        RETURN
        END
*FUNCTION Fmassr(DMASS) GIVES the rho MASS DISTRIBUTION 
        REAL FUNCTION Fmassr(DMASS)
      SAVE   
        AM0=0.77
       wid=0.153
        Fmassr=am0*Wid/((DMASS**2-am0**2)**2
     1  +am0**2*Wid**2)
        RETURN
        END
**********************************
* PURPOSE : flow analysis  
* DATE : Feb. 1, 1995
***********************************
       subroutine flow(nt)
c       IMPLICIT REAL*4 (A-H,O-Z)
       PARAMETER ( PI=3.1415926,APion=0.13957,aka=0.498)
        PARAMETER   (MAXSTR=150001,MAXR=1,AMU= 0.9383,etaM=0.5475)
       DIMENSION ypion(-80:80),ypr(-80:80),ykaon(-80:80)
       dimension pxpion(-80:80),pxpro(-80:80),pxkaon(-80:80)
*----------------------------------------------------------------------*
      COMMON  /AA/      R(3,MAXSTR)
cc      SAVE /AA/
      COMMON  /BB/      P(3,MAXSTR)
cc      SAVE /BB/
      COMMON  /CC/      E(MAXSTR)
cc      SAVE /CC/
      COMMON  /EE/      ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON  /RR/      MASSR(0:MAXR)
cc      SAVE /RR/
      COMMON  /RUN/     NUM
cc      SAVE /RUN/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      SAVE   
*----------------------------------------------------------------------*
       NT=NT
       ycut1=-2.6
       ycut2=2.6
       DY=0.2
       LY=NINT((YCUT2-YCUT1)/DY)
***********************************
C initialize the transverse momentum counters 
       do 11 kk=-80,80
       pxpion(kk)=0
       pxpro(kk)=0
       pxkaon(kk)=0
11       continue
       DO 701 J=-LY,LY
       ypion(j)=0
       ykaon(j)=0
       ypr(j)=0
  701   CONTINUE
       nkaon=0
       npr=0
       npion=0
          IS=0
          DO 20 NRUN=1,NUM
          IS=IS+MASSR(NRUN-1)
          DO 20 J=1,MASSR(NRUN)
          I=J+IS
* for protons go to 200 to calculate its rapidity and transvese momentum
* distributions
       e00=sqrt(P(1,I)**2+P(2,i)**2+P(3,i)**2+e(I)**2)
       y00=0.5*alog((e00+p(3,i))/(e00-p(3,i)))
       if(abs(y00).ge.ycut2)go to 20
       iy=nint(y00/DY)
       if(abs(iy).ge.80)go to 20
       if(e(i).eq.0)go to 20
       if(lb(i).ge.25)go to 20
       if((lb(i).le.5).and.(lb(i).ge.3))go to 50
       if(lb(i).eq.1.or.lb(i).eq.2)go to 200
cbz3/10/99
c       if(lb(i).ge.6.and.lb(i).le.15)go to 200
       if(lb(i).ge.6.and.lb(i).le.17)go to 200
cbz3/10/99 end
       if(lb(i).eq.23)go to 400
       go to 20
* calculate rapidity and transverse momentum distribution for pions
50       npion=npion+1
* (2) rapidity distribution in the cms frame
        ypion(iy)=ypion(iy)+1
       pxpion(iy)=pxpion(iy)+p(1,i)/e(I)
       go TO 20
* calculate rapidity and transverse energy distribution for baryons
200      npr=npr+1  
                pxpro(iy)=pxpro(iy)+p(1,I)/E(I)
                 ypr(iy)=ypr(iy)+1.
        go to 20
400     nkaon=nkaon+1  
                 ykaon(iy)=ykaon(iy)+1.
                pxkaon(iy)=pxkaon(iy)+p(1,i)/E(i)
20      CONTINUE
C PRINT OUT NUCLEON'S TRANSVERSE MOMENTUM distribution
c       write(1041,*)Nt
c       write(1042,*)Nt
c       write(1043,*)Nt
c       write(1090,*)Nt
c       write(1091,*)Nt
c       write(1092,*)Nt
       do 3 npt=-10,10
       IF(ypr(npt).eq.0) go to 101
       pxpro(NPT)=-Pxpro(NPT)/ypr(NPT)
       DNUC=Pxpro(NPT)/SQRT(ypr(NPT))
c       WRITE(1041,*)NPT*DY,Pxpro(NPT),DNUC
c print pion's transverse momentum distribution
101       IF(ypion(npt).eq.0) go to 102
       pxpion(NPT)=-pxpion(NPT)/ypion(NPT)
       DNUCp=pxpion(NPT)/SQRT(ypion(NPT))
c       WRITE(1042,*)NPT*DY,Pxpion(NPT),DNUCp
c kaons
102       IF(ykaon(npt).eq.0) go to 3
       pxkaon(NPT)=-pxkaon(NPT)/ykaon(NPT)
       DNUCk=pxkaon(NPT)/SQRT(ykaon(NPT))
c       WRITE(1043,*)NPT*DY,Pxkaon(NPT),DNUCk
3       CONTINUE
********************************
* OUTPUT PION AND PROTON RAPIDITY DISTRIBUTIONS
       DO 1001 M=-LY,LY
* PROTONS
       DYPR=0
       IF(YPR(M).NE.0)DYPR=SQRT(YPR(M))/FLOAT(NRUN)/DY
       YPR(M)=YPR(M)/FLOAT(NRUN)/DY
c       WRITE(1090,'(E11.3,2X,E11.3,2X,E11.3)')m*DY,YPR(M),DYPR
* PIONS
       DYPION=0
       IF(YPION(M).NE.0)DYPION=SQRT(YPION(M))/FLOAT(NRUN)/DY
       YPION(M)=YPION(M)/FLOAT(NRUN)/DY
c       WRITE(1091,'(E11.3,2X,E11.3,2X,E11.3)')m*DY,YPION(M),DYPION
* KAONS
       DYKAON=0
       IF(YKAON(M).NE.0)DYKAON=SQRT(YKAON(M))/FLOAT(NRUN)/DY
       YKAON(M)=YKAON(M)/FLOAT(NRUN)/DY
c       WRITE(1092,'(E11.3,2X,E11.3,2X,E11.3)')m*DY,YKAON(M),DYKAON
 1001 CONTINUE
       return
       end
cbali1/16/99
********************************************
* Purpose: pp_bar annihilation cross section as a functon of their cms energy
c      real*4 function xppbar(srt)
      real function xppbar(srt)
*  srt    = DSQRT(s) in GeV                                                   *
*  xppbar = pp_bar annihilation cross section in mb                           *
*                                                    
*  Reference: G.J. Wang, R. Bellwied, C. Pruneau and G. Welke
*             Proc. of the 14th Winter Workshop on Nuclear Dynamics, 
*             Snowbird, Utah 31, Eds. W. Bauer and H.G. Ritter 
*             (Plenum Publishing, 1998)                             *
*
******************************************
       Parameter (pmass=0.9383,xmax=400.)
      SAVE   
* Note:
* (1) we introduce a new parameter xmax=400 mb:
*     the maximum annihilation xsection 
* there are shadowing effects in pp_bar annihilation, with this parameter
* we can probably look at these effects  
* (2) Calculate p(lab) from srt [GeV], since the formular in the 
* reference applies only to the case of a p_bar on a proton at rest
* Formula used: srt**2=2.*pmass*(pmass+sqrt(pmass**2+plab**2))
       xppbar=1.e-06
       plab2=(srt**2/(2.*pmass)-pmass)**2-pmass**2
       if(plab2.gt.0)then
           plab=sqrt(plab2)
       xppbar=67./(plab**0.7)
       if(xppbar.gt.xmax)xppbar=xmax
       endif
         return
      END
cbali1/16/99 end
**********************************
cbali2/6/99
********************************************
* Purpose: To generate randomly the no. of pions in the final 
*          state of pp_bar annihilation according to a statistical 
*          model by using of the rejection method.  
cbz2/25/99
c      real*4 function pbarfs(srt,npion,iseed)
      subroutine pbarfs(srt,npion,iseed)
cbz2/25/99end
* Quantities: 
*  srt: DSQRT(s) in GeV                                                    *
*  npion: No. of pions produced in the annihilation of ppbar at srt        *
*  nmax=6, cutoff of the maximum no. of n the code can handle     
*                                             
*  Reference: C.M. Ko and R. Yuan, Phys. Lett. B192 (1987) 31      *
*
******************************************
       parameter (pimass=0.140,pi=3.1415926) 
       Dimension factor(6),pnpi(6) 
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
       ISEED=ISEED 
C the factorial coefficients in the pion no. distribution 
* from n=2 to 6 calculated use the formula in the reference
       factor(2)=1.
       factor(3)=1.17e-01
       factor(4)=3.27e-03
       factor(5)=3.58e-05
       factor(6)=1.93e-07
       ene=(srt/pimass)**3/(6.*pi**2)
c the relative probability from n=2 to 6
       do 1001 n=2,6 
           pnpi(n)=ene**n*factor(n)
 1001   continue
c find the maximum of the probabilities, I checked a 
c Fortan manual: max() returns the maximum value of 
c the same type as in the argument list
       pmax=max(pnpi(2),pnpi(3),pnpi(4),pnpi(5),pnpi(6))
c randomly generate n between 2 and 6
       ntry=0
 10    npion=2+int(5*RANART(NSEED))
clin-4/2008 check bounds:
       if(npion.gt.6) goto 10
       thisp=pnpi(npion)/pmax  
       ntry=ntry+1 
c decide whether to take this npion according to the distribution
c using rejection method.
       if((thisp.lt.RANART(NSEED)).and.(ntry.le.20)) go to 10
c now take the last generated npion and return
       return
       END
**********************************
cbali2/6/99 end
cbz3/9/99 kkbar
cbali3/5/99
******************************************
* purpose: Xsection for K+ K- to pi+ pi-
c      real*4 function xkkpi(srt)
*  srt    = DSQRT(s) in GeV                                  *
*  xkkpi   = xsection in mb obtained from
*           the detailed balance                             *
* ******************************************
c          parameter (pimass=0.140,aka=0.498)
c       xkkpi=1.e-08 
c       ppi2=(srt/2)**2-pimass**2
c       pk2=(srt/2)**2-aka**2
c       if(ppi2.le.0.or.pk2.le.0)return
cbz3/9/99 kkbar
c       xkkpi=ppi2/pk2*pipik(srt)
c       xkkpi=9.0 / 4.0 * ppi2/pk2*pipik(srt)
c        xkkpi = 2.0 * xkkpi
cbz3/9/99 kkbar end

cbz3/9/99 kkbar
c       end
c       return
c        END
cbz3/9/99 kkbar end

cbali3/5/99 end
cbz3/9/99 kkbar end

cbz3/9/99 kkbar
*****************************
* purpose: Xsection for K+ K- to pi+ pi-
      SUBROUTINE XKKANN(SRT, XSK1, XSK2, XSK3, XSK4, XSK5,
     &     XSK6, XSK7, XSK8, XSK9, XSK10, XSK11, SIGK, rrkk)
*  srt    = DSQRT(s) in GeV                                       *
*  xsk1   = annihilation into pi pi                               *
*  xsk2   = annihilation into pi rho (shifted to XKKSAN)         *
*  xsk3   = annihilation into pi omega (shifted to XKKSAN)       *
*  xsk4   = annihilation into pi eta                              *
*  xsk5   = annihilation into rho rho                             *
*  xsk6   = annihilation into rho omega                           *
*  xsk7   = annihilation into rho eta (shifted to XKKSAN)        *
*  xsk8   = annihilation into omega omega                         *
*  xsk9   = annihilation into omega eta (shifted to XKKSAN)      *
*  xsk10  = annihilation into eta eta                             *
*  sigk   = xsection in mb obtained from                          *
*           the detailed balance                                  *
* ***************************
      PARAMETER  (MAXSTR=150001, MAXX=20,  MAXZ=24)
          PARAMETER (AKA=0.498, PIMASS=0.140, RHOM = 0.770, 
     &     OMEGAM = 0.7819, ETAM = 0.5473, APHI=1.02)
      COMMON  /AA/ R(3,MAXSTR)
cc      SAVE /AA/
      COMMON /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
      SAVE   

        S = SRT ** 2
       SIGK = 1.E-08
        XSK1 = 0.0
        XSK2 = 0.0
        XSK3 = 0.0
        XSK4 = 0.0
        XSK5 = 0.0
        XSK6 = 0.0
        XSK7 = 0.0
        XSK8 = 0.0
        XSK9 = 0.0
        XSK10 = 0.0
        XSK11 = 0.0

        XPION0 = PIPIK(SRT)
c.....take into account both K+ and K0
        XPION0 = 2.0 * XPION0
        PI2 = S * (S - 4.0 * AKA ** 2)
         if(PI2 .le. 0.0)return

        XM1 = PIMASS
        XM2 = PIMASS
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XSK1 = 9.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

clin-8/28/00 (pi eta) eta -> K+K- is assumed the same as pi pi -> K+K-:
        XM1 = PIMASS
        XM2 = ETAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XSK4 = 3.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        XM1 = ETAM
        XM2 = ETAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XSK10 = 1.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        XPION0 = rrkk

clin-11/07/00: (pi eta) (rho omega) -> K* Kbar (or K*bar K) instead to K Kbar:
c        XM1 = PIMASS
c        XM2 = RHOM
c        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
c        IF (PF2 .GT. 0.0) THEN
c           XSK2 = 27.0 / 4.0 * PF2 / PI2 * XPION0
c        END IF

c        XM1 = PIMASS
c        XM2 = OMEGAM
c        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
c        IF (PF2 .GT. 0.0) THEN
c           XSK3 = 9.0 / 4.0 * PF2 / PI2 * XPION0
c        END IF

        XM1 = RHOM
        XM2 = RHOM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XSK5 = 81.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        XM1 = RHOM
        XM2 = OMEGAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XSK6 = 27.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

c        XM1 = RHOM
c        XM2 = ETAM
c        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
c        IF (PF2 .GT. 0.0) THEN
c           XSK7 = 9.0 / 4.0 * PF2 / PI2 * XPION0
c        END IF

        XM1 = OMEGAM
        XM2 = OMEGAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XSK8 = 9.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

c        XM1 = OMEGAM
c        XM2 = ETAM
c        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
c        IF (PF2 .GT. 0.0) THEN
c           XSK9 = 3.0 / 4.0 * PF2 / PI2 * XPION0
c        END IF

c* K+ + K- --> phi
          fwdp = 1.68*(aphi**2-4.*aka**2)**1.5/6./aphi/aphi     
          pkaon=0.5*sqrt(srt**2-4.0*aka**2)
          XSK11 = 30.*3.14159*0.1973**2*(aphi*fwdp)**2/
     &             ((srt**2-aphi**2)**2+(aphi*fwdp)**2)/pkaon**2
c
        SIGK = XSK1 + XSK2 + XSK3 + XSK4 + XSK5 + 
     &     XSK6 + XSK7 + XSK8 + XSK9 + XSK10 + XSK11

       RETURN
        END
cbz3/9/99 kkbar end

*****************************
* purpose: Xsection for Phi + B 
       SUBROUTINE XphiB(LB1, LB2, EM1, EM2, SRT,
     &                  XSK1, XSK2, XSK3, XSK4, XSK5, SIGP)
c
* ***************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
          PARAMETER (AKA=0.498, ALA = 1.1157, PIMASS=0.140, APHI=1.02)
        parameter (arho=0.77)
      SAVE   

       SIGP = 1.E-08
        XSK1 = 0.0
        XSK2 = 0.0
        XSK3 = 0.0
        XSK4 = 0.0
        XSK5 = 0.0
        XSK6 = 0.0
          srrt = srt - (em1+em2)

c* phi + N(D) -> elastic scattering
c            XSK1 = 0.56  !! mb
c  !! mb  (photo-production xsecn used)
            XSK1 = 8.00
c
c* phi + N(D) -> pi + N
        IF (srt  .GT. (ap1+amn)) THEN
             XSK2 = 0.0235*srrt**(-0.519) 
        END IF
c
c* phi + N(D) -> pi + D
        IF (srt  .GT. (ap1+am0)) THEN
            if(srrt .lt. 0.7)then
             XSK3 = 0.0119*srrt**(-0.534)
            else
             XSK3 = 0.0130*srrt**(-0.304)
            endif      
        END IF
c
c* phi + N(D) -> rho + N
        IF (srt  .GT. (arho+amn)) THEN
           if(srrt .lt. 0.7)then
             XSK4 = 0.0166*srrt**(-0.786)
            else
             XSK4 = 0.0189*srrt**(-0.277)
            endif
        END IF
c
c* phi + N(D) -> rho + D   (same as pi + D)
        IF (srt  .GT. (arho+am0)) THEN
            if(srrt .lt. 0.7)then
             XSK5 = 0.0119*srrt**(-0.534)
            else
             XSK5 = 0.0130*srrt**(-0.304)
            endif      
        END IF
c
c* phi + N -> K+ + La
       IF( (lb1.ge.1.and.lb1.le.2) .or. (lb2.ge.1.and.lb2.le.2) )THEN
        IF (srt  .GT. (aka+ala)) THEN
           XSK6 = 1.715/((srrt+3.508)**2-12.138)  
        END IF
       END IF
        SIGP = XSK1 + XSK2 + XSK3 + XSK4 + XSK5 + XSK6
       RETURN
        END
c
**********************************
*
        SUBROUTINE CRPHIB(PX,PY,PZ,SRT,I1,I2,
     &     XSK1, XSK2, XSK3, XSK4, XSK5, SIGP, IBLOCK)
*
*     PURPOSE:                                                         *
*             DEALING WITH PHI + N(D) --> pi+N(D), rho+N(D),  K+ + La
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - INFORMATION about the reaction channel          *
*                
*             iblock   - 20  elastic
*             iblock   - 221  K+ formation
*             iblock   - 223  others
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AMRHO=0.769,AMOMGA=0.782,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974,ARHO=0.77)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
c
       PX0=PX
       PY0=PY
       PZ0=PZ
       IBLOCK=223
c
        X1 = RANART(NSEED) * SIGP
        XSK2 = XSK1 + XSK2
        XSK3 = XSK2 + XSK3
        XSK4 = XSK3 + XSK4
        XSK5 = XSK4 + XSK5
c
c  !! elastic scatt.
        IF (X1 .LE. XSK1) THEN
           iblock=20
           GOTO 100
        ELSE IF (X1 .LE. XSK2) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = AP1
           E(I2) = AMN
           GOTO 100
        ELSE IF (X1 .LE. XSK3) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 6 + int(4 * RANART(NSEED))
           E(I1) = AP1
           E(I2) = AM0
           GOTO 100
        ELSE IF (X1 .LE. XSK4) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 1 + int(2 * RANART(NSEED))
           E(I1) = ARHO
           E(I2) = AMN
           GOTO 100
        ELSE IF (X1 .LE. XSK5) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = 6 + int(4 * RANART(NSEED))
           E(I1) = ARHO
           E(I2) = AM0
           GOTO 100
        ELSE 
           LB(I1) = 23
           LB(I2) = 14
           E(I1) = AKA
           E(I2) = ALA
          IBLOCK=221
         ENDIF
 100    CONTINUE
      EM1=E(I1)
      EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.E-08
          PR=SQRT(PR2)/(2.*SRT)
* WE ASSUME AN ISOTROPIC ANGULAR DISTRIBUTION IN THE CMS 
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
      RETURN
      END
c
*****************************
* purpose: Xsection for Phi + B 
c!! in fm^2
      SUBROUTINE pibphi(srt,lb1,lb2,em1,em2,Xphi,xphin) 
c
*      phi + N(D) <- pi + N
*      phi + N(D) <- pi + D
*      phi + N(D) <- rho + N
*      phi + N(D) <- rho + D   (same as pi + D)
c
* ***************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
          PARAMETER (AKA=0.498, ALA = 1.1157, PIMASS=0.140, APHI=1.02)
        parameter (arho=0.77)
      SAVE   

       Xphi = 0.0
       xphin = 0.0
       xphid = 0.0
c
       if( (lb1.ge.3.and.lb1.le.5) .or.
     &     (lb2.ge.3.and.lb2.le.5) )then
c
       if( (iabs(lb1).ge.1.and.iabs(lb1).le.2) .or.
     &     (iabs(lb2).ge.1.and.iabs(lb2).le.2) )then
c* phi + N <- pi + N
        IF (srt  .GT. (aphi+amn)) THEN
             srrt = srt - (aphi+amn)
             sig = 0.0235*srrt**(-0.519) 
          xphin=sig*1.*(srt**2-(aphi+amn)**2)*
     &           (srt**2-(aphi-amn)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
c* phi + D <- pi + N
        IF (srt  .GT. (aphi+am0)) THEN
             srrt = srt - (aphi+am0)
             sig = 0.0235*srrt**(-0.519) 
          xphid=sig*4.*(srt**2-(aphi+am0)**2)*
     &           (srt**2-(aphi-am0)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
       else
c* phi + N <- pi + D
        IF (srt  .GT. (aphi+amn)) THEN
             srrt = srt - (aphi+amn)
            if(srrt .lt. 0.7)then
             sig = 0.0119*srrt**(-0.534)
            else
             sig = 0.0130*srrt**(-0.304)
            endif      
          xphin=sig*(1./4.)*(srt**2-(aphi+amn)**2)*
     &           (srt**2-(aphi-amn)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
c* phi + D <- pi + D
        IF (srt  .GT. (aphi+am0)) THEN
             srrt = srt - (aphi+am0)
             if(srrt .lt. 0.7)then
             sig = 0.0119*srrt**(-0.534)
            else
             sig = 0.0130*srrt**(-0.304)
            endif      
          xphid=sig*1.*(srt**2-(aphi+am0)**2)*
     &           (srt**2-(aphi-am0)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
       endif
c
c
C** for rho + N(D) colln
c
       else
c
       if( (iabs(lb1).ge.1.and.iabs(lb1).le.2) .or.
     &     (iabs(lb2).ge.1.and.iabs(lb2).le.2) )then
c
c* phi + N <- rho + N
        IF (srt  .GT. (aphi+amn)) THEN
             srrt = srt - (aphi+amn)
           if(srrt .lt. 0.7)then
             sig = 0.0166*srrt**(-0.786)
            else
             sig = 0.0189*srrt**(-0.277)
            endif
          xphin=sig*(1./3.)*(srt**2-(aphi+amn)**2)*
     &           (srt**2-(aphi-amn)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
c* phi + D <- rho + N
        IF (srt  .GT. (aphi+am0)) THEN
             srrt = srt - (aphi+am0)
           if(srrt .lt. 0.7)then
             sig = 0.0166*srrt**(-0.786)
            else
             sig = 0.0189*srrt**(-0.277)
            endif
          xphid=sig*(4./3.)*(srt**2-(aphi+am0)**2)*
     &           (srt**2-(aphi-am0)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
       else
c* phi + N <- rho + D  (same as pi+D->phi+N)
        IF (srt  .GT. (aphi+amn)) THEN
             srrt = srt - (aphi+amn)
            if(srrt .lt. 0.7)then
             sig = 0.0119*srrt**(-0.534)
            else
             sig = 0.0130*srrt**(-0.304)
            endif      
          xphin=sig*(1./12.)*(srt**2-(aphi+amn)**2)*
     &           (srt**2-(aphi-amn)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
c* phi + D <- rho + D  (same as pi+D->phi+D)
        IF (srt  .GT. (aphi+am0)) THEN
             srrt = srt - (aphi+am0)
             if(srrt .lt. 0.7)then
             sig = 0.0119*srrt**(-0.534)
            else
             sig = 0.0130*srrt**(-0.304)
            endif      
          xphid=sig*(1./3.)*(srt**2-(aphi+am0)**2)*
     &           (srt**2-(aphi-am0)**2)/(srt**2-(em1+em2)**2)/
     &           (srt**2-(em1-em2)**2)
        END IF
       endif
        END IF
c   !! in fm^2
         xphin = xphin/10.
c   !! in fm^2
         xphid = xphid/10.
         Xphi = xphin + xphid

       RETURN
        END
c
*****************************
* purpose: Xsection for phi +M to K+K etc
      SUBROUTINE PHIMES(I1, I2, SRT, XSK1, XSK2, XSK3, XSK4, XSK5,
     1     XSK6, XSK7, SIGPHI)

*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      223 --> phi destruction
*                      20 -->  elastic
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER  (AKA=0.498, AKS=0.895, AOMEGA=0.7819,
     3               ARHO=0.77, APHI=1.02)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        PARAMETER  (MAXX=20,  MAXZ=24)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
      COMMON  /DD/      RHO(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHOP(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ),
     &                     RHON(-MAXX:MAXX,-MAXX:MAXX,-MAXZ:MAXZ)
cc      SAVE /DD/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      SAVE   

        S = SRT ** 2
       SIGPHI = 1.E-08
        XSK1 = 0.0
        XSK2 = 0.0
        XSK3 = 0.0
        XSK4 = 0.0
        XSK5 = 0.0
        XSK6 = 0.0
        XSK7 = 0.0
         em1 = E(i1)
         em2 = E(i2)
         LB1 = LB(i1)
         LB2 = LB(i2)
         akap = aka
c******
c
c   !! mb, elastic
         XSK1 = 5.0
         
           pii = sqrt((S-(em1+em2)**2)*(S-(em1-em2)**2))
* phi + K(-bar) channel
       if( lb1.eq.23.or.lb2.eq.23 .or. lb1.eq.21.or.lb2.eq.21 )then
          if(srt .gt. (ap1+akap))then
c             XSK2 = 2.5  
           pff = sqrt((S-(ap1+akap)**2)*(S-(ap1-akap)**2))
           XSK2 = 195.639*pff/pii/32./pi/S 
          endif
          if(srt .gt. (arho+akap))then
c              XSK3 = 3.5  
           pff = sqrt((S-(arho+akap)**2)*(S-(arho-akap)**2))
           XSK3 = 526.702*pff/pii/32./pi/S 
          endif
          if(srt .gt. (aomega+akap))then
c               XSK4 = 3.5 
           pff = sqrt((S-(aomega+akap)**2)*(S-(aomega-akap)**2))
           XSK4 = 355.429*pff/pii/32./pi/S 
          endif
          if(srt .gt. (ap1+aks))then
c           XSK5 = 15.0  
           pff = sqrt((S-(ap1+aks)**2)*(S-(ap1-aks)**2))
           XSK5 = 2047.042*pff/pii/32./pi/S 
          endif
          if(srt .gt. (arho+aks))then
c            XSK6 = 3.5 
           pff = sqrt((S-(arho+aks)**2)*(S-(arho-aks)**2))
           XSK6 = 1371.257*pff/pii/32./pi/S 
          endif
          if(srt .gt. (aomega+aks))then
c            XSK7 = 3.5 
           pff = sqrt((S-(aomega+aks)**2)*(S-(aomega-aks)**2))
           XSK7 = 482.292*pff/pii/32./pi/S 
          endif
c
       elseif( iabs(lb1).eq.30.or.iabs(lb2).eq.30 )then
* phi + K*(-bar) channel
c
          if(srt .gt. (ap1+akap))then
c             XSK2 = 3.5  
           pff = sqrt((S-(ap1+akap)**2)*(S-(ap1-akap)**2))
           XSK2 = 372.378*pff/pii/32./pi/S 
          endif
          if(srt .gt. (arho+akap))then
c              XSK3 = 9.0  
           pff = sqrt((S-(arho+akap)**2)*(S-(arho-akap)**2))
           XSK3 = 1313.960*pff/pii/32./pi/S 
          endif
          if(srt .gt. (aomega+akap))then
c               XSK4 = 6.5 
           pff = sqrt((S-(aomega+akap)**2)*(S-(aomega-akap)**2))
           XSK4 = 440.558*pff/pii/32./pi/S 
          endif
          if(srt .gt. (ap1+aks))then
c           XSK5 = 30.0 !wrong  
           pff = sqrt((S-(ap1+aks)**2)*(S-(ap1-aks)**2))
           XSK5 = 1496.692*pff/pii/32./pi/S 
          endif
          if(srt .gt. (arho+aks))then
c            XSK6 = 9.0 
           pff = sqrt((S-(arho+aks)**2)*(S-(arho-aks)**2))
           XSK6 = 6999.840*pff/pii/32./pi/S 
          endif
          if(srt .gt. (aomega+aks))then
c            XSK7 = 15.0 
           pff = sqrt((S-(aomega+aks)**2)*(S-(aomega-aks)**2))
           XSK7 = 1698.903*pff/pii/32./pi/S 
          endif
       else
c
* phi + rho(pi,omega) channel
c
           srr1 = em1+em2
         if(srt .gt. (akap+akap))then
          srrt = srt - srr1
cc          if(srrt .lt. 0.3)then
          if(srrt .lt. 0.3 .and. srrt .gt. 0.01)then
          XSK2 = 1.69/(srrt**0.141 - 0.407)
          else
          XSK2 = 3.74 + 0.008*srrt**1.9
          endif                 
         endif
         if(srt .gt. (akap+aks))then
          srr2 = akap+aks
          srr = amax1(srr1,srr2)
          srrt = srt - srr
cc          if(srrt .lt. 0.3)then
          if(srrt .lt. 0.3 .and. srrt .gt. 0.01)then
          XSK3 = 1.69/(srrt**0.141 - 0.407)
          else
          XSK3 = 3.74 + 0.008*srrt**1.9
          endif
         endif
         if(srt .gt. (aks+aks))then
          srr2 = aks+aks
          srr = amax1(srr1,srr2)
          srrt = srt - srr
cc          if(srrt .lt. 0.3)then
          if(srrt .lt. 0.3 .and. srrt .gt. 0.01)then
          XSK4 = 1.69/(srrt**0.141 - 0.407)
          else
          XSK4 = 3.74 + 0.008*srrt**1.9
          endif
         endif
c          xsk2 = amin1(20.,xsk2)
c          xsk3 = amin1(20.,xsk3)
c          xsk4 = amin1(20.,xsk4)
      endif

        SIGPHI = XSK1 + XSK2 + XSK3 + XSK4 + XSK5 + XSK6 + XSK7

       RETURN
       END

**********************************
*     PURPOSE:                                                         *
*             DEALING WITH phi+M  scatt.
*
       SUBROUTINE CRPHIM(PX,PY,PZ,SRT,I1,I2,
     &  XSK1, XSK2, XSK3, XSK4, XSK5, XSK6, SIGPHI, IKKG, IKKL, IBLOCK)
*
*     QUANTITIES:                                                      *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                      20 -->  elastic
*                      223 --> phi + pi(rho,omega)
*                      224 --> phi + K -> K + pi(rho,omega)
*                      225 --> phi + K -> K* + pi(rho,omega)
*                      226 --> phi + K* -> K + pi(rho,omega)
*                      227 --> phi + K* -> K* + pi(rho,omega)
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,ARHO=0.77,AOMEGA=0.7819,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER    (AKA=0.498,AKS=0.895)
        parameter   (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
c
       PX0=PX
       PY0=PY
       PZ0=PZ
         LB1 = LB(i1)
         LB2 = LB(i2)

        X1 = RANART(NSEED) * SIGPHI
        XSK2 = XSK1 + XSK2
        XSK3 = XSK2 + XSK3
        XSK4 = XSK3 + XSK4
        XSK5 = XSK4 + XSK5
        XSK6 = XSK5 + XSK6
        IF (X1 .LE. XSK1) THEN
c        !! elastic scatt
           IBLOCK=20
           GOTO 100
        ELSE
c
*phi + (K,K*)-bar
       if( lb1.eq.23.or.lb1.eq.21.or.iabs(lb1).eq.30 .OR.
     &     lb2.eq.23.or.lb2.eq.21.or.iabs(lb2).eq.30 )then
c
             if(lb1.eq.23.or.lb2.eq.23)then
               IKKL=1
               IBLOCK=224
               iad1 = 23
               iad2 = 30
              elseif(lb1.eq.30.or.lb2.eq.30)then
               IKKL=0
               IBLOCK=226
               iad1 = 23
               iad2 = 30
             elseif(lb1.eq.21.or.lb2.eq.21)then
               IKKL=1
               IBLOCK=124
               iad1 = 21
               iad2 = -30
c         !! -30
             else
               IKKL=0
               IBLOCK=126
               iad1 = 21
               iad2 = -30
              endif
         IF (X1 .LE. XSK2) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = iad1
           E(I1) = AP1
           E(I2) = AKA
           IKKG = 1
           GOTO 100
        ELSE IF (X1 .LE. XSK3) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = iad1
           E(I1) = ARHO
           E(I2) = AKA
           IKKG = 1
           GOTO 100
        ELSE IF (X1 .LE. XSK4) THEN
           LB(I1) = 28
           LB(I2) = iad1
           E(I1) = AOMEGA
           E(I2) = AKA
           IKKG = 1
           GOTO 100
        ELSE IF (X1 .LE. XSK5) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = iad2
           E(I1) = AP1
           E(I2) = AKS
           IKKG = 0
           IBLOCK=IBLOCK+1
           GOTO 100
        ELSE IF (X1 .LE. XSK6) THEN
           LB(I1) = 25 + int(3 * RANART(NSEED))
           LB(I2) = iad2
           E(I1) = ARHO
           E(I2) = AKS
           IKKG = 0
           IBLOCK=IBLOCK+1
           GOTO 100
        ELSE 
           LB(I1) = 28
           LB(I2) = iad2
           E(I1) = AOMEGA
           E(I2) = AKS
           IKKG = 0
           IBLOCK=IBLOCK+1
           GOTO 100
         ENDIF
       else
c      !! phi destruction via (pi,rho,omega)
          IBLOCK=223
*phi + pi(rho,omega)
         IF (X1 .LE. XSK2) THEN
           LB(I1) = 23
           LB(I2) = 21
           E(I1) = AKA
           E(I2) = AKA
           IKKG = 2
           IKKL = 0
           GOTO 100
        ELSE IF (X1 .LE. XSK3) THEN
           LB(I1) = 23
c           LB(I2) = 30
           LB(I2) = -30
clin-2/10/03 currently take XSK3 to be the sum of KK*bar & KbarK*:
           if(RANART(NSEED).le.0.5) then
              LB(I1) = 21
              LB(I2) = 30
           endif
              
           E(I1) = AKA
           E(I2) = AKS
           IKKG = 1
           IKKL = 0
           GOTO 100
        ELSE IF (X1 .LE. XSK4) THEN
           LB(I1) = 30
c           LB(I2) = 30
           LB(I2) = -30
           E(I1) = AKS
           E(I2) = AKS
           IKKG = 0
           IKKL = 0
           GOTO 100
         ENDIF
       endif
         ENDIF
*
100    CONTINUE
       EM1=E(I1)
       EM2=E(I2)

*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.E-08
          PR=SQRT(PR2)/(2.*SRT)
* WE ASSUME AN ISOTROPIC ANGULAR DISTRIBUTION IN THE CMS 
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
      RETURN
      END
**********************************
**********************************
cbz3/9/99 khyperon
*************************************
* purpose: Xsection for K+Y ->  piN                                       *
*          Xsection for K+Y-bar ->  piN-bar   !! sp03/29/01               *
*
        SUBROUTINE XKHYPE(I1, I2, SRT, XKY1, XKY2, XKY3, XKY4, XKY5,
     &     XKY6, XKY7, XKY8, XKY9, XKY10, XKY11, XKY12, XKY13,
     &     XKY14, XKY15, XKY16, XKY17, SIGK)
c      subroutine xkhype(i1, i2, srt, sigk)
*  srt    = DSQRT(s) in GeV                                               *
*  xkkpi   = xsection in mb obtained from                                 *
*           the detailed balance                                          *
* ***********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,AMRHO=0.769,AMOMGA=0.782,APHI=1.02,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
          parameter (pimass=0.140, AMETA = 0.5473, aka=0.498,
     &     aml=1.116,ams=1.193, AM1440 = 1.44, AM1535 = 1.535)
        COMMON  /EE/ID(MAXSTR), LB(MAXSTR)
cc      SAVE /EE/
      SAVE   

        S = SRT ** 2
       SIGK=1.E-08 
        XKY1 = 0.0
        XKY2 = 0.0
        XKY3 = 0.0
        XKY4 = 0.0
        XKY5 = 0.0
        XKY6 = 0.0
        XKY7 = 0.0
        XKY8 = 0.0
        XKY9 = 0.0
        XKY10 = 0.0
        XKY11 = 0.0
        XKY12 = 0.0
        XKY13 = 0.0
        XKY14 = 0.0
        XKY15 = 0.0
        XKY16 = 0.0
        XKY17 = 0.0

        LB1 = LB(I1)
        LB2 = LB(I2)
        IF (iabs(LB1) .EQ. 14 .OR. iabs(LB2) .EQ. 14) THEN
           XKAON0 = PNLKA(SRT)
           XKAON0 = 2.0 * XKAON0
           PI2 = (S - (AML + AKA) ** 2) * (S - (AML - AKA) ** 2)
        ELSE
           XKAON0 = PNSKA(SRT)
           XKAON0 = 2.0 * XKAON0
           PI2 = (S - (AMS + AKA) ** 2) * (S - (AMS - AKA) ** 2)
        END IF
          if(PI2 .le. 0.0)return

        XM1 = PIMASS
        XM2 = AMP
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY1 = 3.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = PIMASS
        XM2 = AM0
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY2 = 12.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = PIMASS
        XM2 = AM1440
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY3 = 3.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = PIMASS
        XM2 = AM1535
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY4 = 3.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMRHO
        XM2 = AMP
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY5 = 9.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMRHO
        XM2 = AM0
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY6 = 36.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMRHO
        XM2 = AM1440
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY7 = 9.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMRHO
        XM2 = AM1535
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY8 = 9.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMOMGA
        XM2 = AMP
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY9 = 3.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMOMGA
        XM2 = AM0
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY10 = 12.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMOMGA
        XM2 = AM1440
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY11 = 3.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMOMGA
        XM2 = AM1535
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY12 = 3.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMETA
        XM2 = AMP
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY13 = 1.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMETA
        XM2 = AM0
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY14 = 4.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMETA
        XM2 = AM1440
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY15 = 1.0 * PF2 / PI2 * XKAON0
        END IF
        
        XM1 = AMETA
        XM2 = AM1535
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           XKY16 = 1.0 * PF2 / PI2 * XKAON0
        END IF

csp11/21/01  K+ + La --> phi + N 
        if(lb1.eq.14 .or. lb2.eq.14)then
         if(srt .gt. (aphi+amn))then
           srrt = srt - (aphi+amn)
           sig = 1.715/((srrt+3.508)**2-12.138)
         XM1 = AMN
         XM2 = APHI
         PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
c     ! fm^-1
         XKY17 = 3.0 * PF2 / PI2 * SIG/10.
        endif
       endif
csp11/21/01  end 
c

       IF ((iabs(LB1) .GE. 15 .AND. iabs(LB1) .LE. 17) .OR. 
     &     (iabs(LB2) .GE. 15 .AND. iabs(LB2) .LE. 17)) THEN
           DDF = 3.0
           XKY1 = XKY1 / DDF
           XKY2 = XKY2 / DDF
           XKY3 = XKY3 / DDF
           XKY4 = XKY4 / DDF
           XKY5 = XKY5 / DDF
           XKY6 = XKY6 / DDF
           XKY7 = XKY7 / DDF
           XKY8 = XKY8 / DDF
           XKY9 = XKY9 / DDF
           XKY10 = XKY10/ DDF
           XKY11 = XKY11 / DDF
           XKY12 = XKY12 / DDF
           XKY13 = XKY13 / DDF
           XKY14 = XKY14 / DDF
           XKY15 = XKY15 / DDF
           XKY16 = XKY16 / DDF
        END IF
        
        SIGK = XKY1 + XKY2 + XKY3 + XKY4 +
     &       XKY5 + XKY6 + XKY7 + XKY8 +
     &       XKY9 + XKY10 + XKY11 + XKY12 +
     &       XKY13 + XKY14 + XKY15 + XKY16 + XKY17

       RETURN
       END

C*******************************  
      BLOCK DATA PPBDAT 
    
      parameter (AMP=0.93828,AMN=0.939457,
     1     AM0=1.232,AM1440 = 1.44, AM1535 = 1.535)

c     to give default values to parameters for BbarB production from mesons
      COMMON/ppbmas/niso(15),nstate,ppbm(15,2),thresh(15),weight(15)
cc      SAVE /ppbmas/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   
c     thresh(i) gives the mass thresh for final channel i:
      DATA thresh/1.87656,1.877737,1.878914,2.17028,
     1     2.171457,2.37828,2.379457,2.464,2.47328,2.474457,
     2     2.672,2.767,2.88,2.975,3.07/
c     ppbm(i,j=1,2) gives masses for the two final baryons of channel i,
c     with j=1 for the lighter baryon:
      DATA (ppbm(i,1),i=1,15)/amp,amp,amn,amp,amn,amp,amn,
     1     am0,amp,amn,am0,am0,am1440,am1440,am1535/
      DATA (ppbm(i,2),i=1,15)/amp,amn,amn,am0,am0,am1440,am1440,
     1     am0,am1535,am1535,am1440,am1535,am1440,am1535,am1535/
c     factr2(i) gives weights for producing i pions from ppbar annihilation:
      DATA factr2/0,1,1.17e-01,3.27e-03,3.58e-05,1.93e-07/
c     niso(i) gives the degeneracy factor for final channel i:
      DATA niso/1,2,1,16,16,4,4,64,4,4,32,32,4,8,4/

      END   


*****************************************
* get the number of BbarB states available for mm collisions of energy srt 
      subroutine getnst(srt)
*  srt    = DSQRT(s) in GeV                                                   *
*****************************************
      parameter (pimass=0.140,pi=3.1415926)
      COMMON/ppbmas/niso(15),nstate,ppbm(15,2),thresh(15),weight(15)
cc      SAVE /ppbmas/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s=srt**2
      nstate=0
      wtot=0.
      if(srt.le.thresh(1)) return
      do 1001 i=1,15
         weight(i)=0.
         if(srt.gt.thresh(i)) nstate=i
 1001 continue
      do 1002 i=1,nstate
         pf2=(s-(ppbm(i,1)+ppbm(i,2))**2)
     1        *(s-(ppbm(i,1)-ppbm(i,2))**2)/4/s
         weight(i)=pf2*niso(i)
         wtot=wtot+weight(i)
 1002 continue
      ene=(srt/pimass)**3/(6.*pi**2)
      fsum=factr2(2)+factr2(3)*ene+factr2(4)*ene**2
     1     +factr2(5)*ene**3+factr2(6)*ene**4

      return
      END

*****************************************
* for pion+pion-->Bbar B                                                      *
c      real*4 function ppbbar(srt)
      real function ppbbar(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      sppb2p=xppbar(srt)*factr2(2)/fsum
      pi2=(s-4*pimass**2)/4
      ppbbar=4./9.*sppb2p/pi2*wtot

      return
      END

*****************************************
* for pion+rho-->Bbar B                                                      *
c      real*4 function prbbar(srt)
      real function prbbar(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      sppb3p=xppbar(srt)*factr2(3)*ene/fsum
      pi2=(s-(pimass+arho)**2)*(s-(pimass-arho)**2)/4/s
      prbbar=4./27.*sppb3p/pi2*wtot

      return
      END

*****************************************
* for rho+rho-->Bbar B                                                      *
c      real*4 function rrbbar(srt)
      real function rrbbar(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      sppb4p=xppbar(srt)*factr2(4)*ene**2/fsum
      pi2=(s-4*arho**2)/4
      rrbbar=4./81.*(sppb4p/2)/pi2*wtot

      return
      END

*****************************************
* for pi+omega-->Bbar B                                                      *
c      real*4 function pobbar(srt)
      real function pobbar(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      sppb4p=xppbar(srt)*factr2(4)*ene**2/fsum
      pi2=(s-(pimass+aomega)**2)*(s-(pimass-aomega)**2)/4/s
      pobbar=4./9.*(sppb4p/2)/pi2*wtot

      return
      END

*****************************************
* for rho+omega-->Bbar B                                                      *
c      real*4 function robbar(srt)
      real function robbar(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      sppb5p=xppbar(srt)*factr2(5)*ene**3/fsum
      pi2=(s-(arho+aomega)**2)*(s-(arho-aomega)**2)/4/s
      robbar=4./27.*sppb5p/pi2*wtot

      return
      END

*****************************************
* for omega+omega-->Bbar B                                                    *
c      real*4 function oobbar(srt)
      real function oobbar(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      sppb6p=xppbar(srt)*factr2(6)*ene**4/fsum
      pi2=(s-4*aomega**2)/4
      oobbar=4./9.*sppb6p/pi2*wtot

      return
      END

*****************************************
* Generate final states for mm-->Bbar B                                       *
      SUBROUTINE bbarfs(lbb1,lbb2,ei1,ei2,iblock,iseed)
*****************************************
      COMMON/ppbmas/niso(15),nstate,ppbm(15,2),thresh(15),weight(15)
cc      SAVE /ppbmas/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
      ISEED=ISEED
c     determine which final BbarB channel occurs:
      rd=RANART(NSEED)
      wsum=0.
      do 1001 i=1,nstate
         wsum=wsum+weight(i)
         if(rd.le.(wsum/wtot)) then
            ifs=i
            ei1=ppbm(i,1)
            ei2=ppbm(i,2)
            goto 10
         endif
 1001 continue
 10   continue

c1    pbar p
      if(ifs.eq.1) then
         iblock=1801
         lbb1=-1
         lbb2=1
      elseif(ifs.eq.2) then
c2    pbar n
         if(RANART(NSEED).le.0.5) then
            iblock=18021
            lbb1=-1
            lbb2=2
c2    nbar p
         else
            iblock=18022
            lbb1=1
            lbb2=-2
         endif
c3    nbar n
      elseif(ifs.eq.3) then
         iblock=1803
         lbb1=-2
         lbb2=2
c4&5  (pbar nbar) Delta, (p n) anti-Delta
      elseif(ifs.eq.4.or.ifs.eq.5) then
         rd=RANART(NSEED)
         if(rd.le.0.5) then
c     (pbar nbar) Delta
            if(ifs.eq.4) then
               iblock=18041
               lbb1=-1
            else
               iblock=18051
               lbb1=-2
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.25) then
               lbb2=6
            elseif(rd2.le.0.5) then
               lbb2=7
            elseif(rd2.le.0.75) then
               lbb2=8
            else
               lbb2=9
            endif
         else
c     (p n) anti-Delta
            if(ifs.eq.4) then
               iblock=18042
               lbb1=1
            else
               iblock=18052
               lbb1=2
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.25) then
               lbb2=-6
            elseif(rd2.le.0.5) then
               lbb2=-7
            elseif(rd2.le.0.75) then
               lbb2=-8
            else
               lbb2=-9
            endif
         endif
c6&7  (pbar nbar) N*(1440), (p n) anti-N*(1440)
      elseif(ifs.eq.6.or.ifs.eq.7) then
         rd=RANART(NSEED)
         if(rd.le.0.5) then
c     (pbar nbar) N*(1440)
            if(ifs.eq.6) then
               iblock=18061
               lbb1=-1
            else
               iblock=18071
               lbb1=-2
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.5) then
               lbb2=10
            else
               lbb2=11
            endif
         else
c     (p n) anti-N*(1440)
            if(ifs.eq.6) then
               iblock=18062
               lbb1=1
            else
               iblock=18072
               lbb1=2
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.5) then
               lbb2=-10
            else
               lbb2=-11
            endif
         endif
c8    Delta anti-Delta
      elseif(ifs.eq.8) then
         iblock=1808
         rd1=RANART(NSEED)
         if(rd1.le.0.25) then
            lbb1=6
         elseif(rd1.le.0.5) then
            lbb1=7
         elseif(rd1.le.0.75) then
            lbb1=8
         else
            lbb1=9
         endif
         rd2=RANART(NSEED)
         if(rd2.le.0.25) then
            lbb2=-6
         elseif(rd2.le.0.5) then
            lbb2=-7
         elseif(rd2.le.0.75) then
            lbb2=-8
         else
            lbb2=-9
         endif
c9&10 (pbar nbar) N*(1535), (p n) anti-N*(1535)
      elseif(ifs.eq.9.or.ifs.eq.10) then
         rd=RANART(NSEED)
         if(rd.le.0.5) then
c     (pbar nbar) N*(1440)
            if(ifs.eq.9) then
               iblock=18091
               lbb1=-1
            else
               iblock=18101
               lbb1=-2
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.5) then
               lbb2=12
            else
               lbb2=13
            endif
         else
c     (p n) anti-N*(1535)
            if(ifs.eq.9) then
               iblock=18092
               lbb1=1
            else
               iblock=18102
               lbb1=2
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.5) then
               lbb2=-12
            else
               lbb2=-13
            endif
         endif
c11&12 anti-Delta N*, Delta anti-N*
      elseif(ifs.eq.11.or.ifs.eq.12) then
         rd=RANART(NSEED)
         if(rd.le.0.5) then
c     anti-Delta N*
            rd1=RANART(NSEED)
            if(rd1.le.0.25) then
               lbb1=-6
            elseif(rd1.le.0.5) then
               lbb1=-7
            elseif(rd1.le.0.75) then
               lbb1=-8
            else
               lbb1=-9
            endif
            if(ifs.eq.11) then
               iblock=18111
               rd2=RANART(NSEED)
               if(rd2.le.0.5) then
                  lbb2=10
               else
                  lbb2=11
               endif
            else
               iblock=18121
               rd2=RANART(NSEED)
               if(rd2.le.0.5) then
                  lbb2=12
               else
                  lbb2=13
               endif
            endif
         else
c     Delta anti-N*
            rd1=RANART(NSEED)
            if(rd1.le.0.25) then
               lbb1=6
            elseif(rd1.le.0.5) then
               lbb1=7
            elseif(rd1.le.0.75) then
               lbb1=8
            else
               lbb1=9
            endif
            if(ifs.eq.11) then
               iblock=18112
               rd2=RANART(NSEED)
               if(rd2.le.0.5) then
                  lbb2=-10
               else
                  lbb2=-11
               endif
            else
               iblock=18122
               rd2=RANART(NSEED)
               if(rd2.le.0.5) then
                  lbb2=-12
               else
                  lbb2=-13
               endif
            endif
         endif
c13   N*(1440) anti-N*(1440)
      elseif(ifs.eq.13) then
         iblock=1813
         rd1=RANART(NSEED)
         if(rd1.le.0.5) then
            lbb1=10
         else
            lbb1=11
         endif
         rd2=RANART(NSEED)
         if(rd2.le.0.5) then
            lbb2=-10
         else
            lbb2=-11
         endif
c14   anti-N*(1440) N*(1535), N*(1440) anti-N*(1535)
      elseif(ifs.eq.14) then
         rd=RANART(NSEED)
         if(rd.le.0.5) then
c     anti-N*(1440) N*(1535)
            iblock=18141
            rd1=RANART(NSEED)
            if(rd1.le.0.5) then
               lbb1=-10
            else
               lbb1=-11
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.5) then
               lbb2=12
            else
               lbb2=13
            endif
         else
c     N*(1440) anti-N*(1535)
            iblock=18142
            rd1=RANART(NSEED)
            if(rd1.le.0.5) then
               lbb1=10
            else
               lbb1=11
            endif
            rd2=RANART(NSEED)
            if(rd2.le.0.5) then
               lbb2=-12
            else
               lbb2=-13
            endif
         endif
c15   N*(1535) anti-N*(1535)
      elseif(ifs.eq.15) then
         iblock=1815
         rd1=RANART(NSEED)
         if(rd1.le.0.5) then
            lbb1=12
         else
            lbb1=13
         endif
         rd2=RANART(NSEED)
         if(rd2.le.0.5) then
            lbb2=-12
         else
            lbb2=-13
         endif
      else
      endif

      RETURN
      END

*****************************************
* for pi pi <-> rho rho cross sections
        SUBROUTINE spprr(lb1,lb2,srt)
        parameter (arho=0.77)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

        pprr=0.
        if((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.3.and.lb2.le.5)) then
c     for now, rho mass taken to be the central value in these two processes
           if(srt.gt.(2*arho)) pprr=ptor(srt)
        elseif((lb1.ge.25.and.lb1.le.27).and.(lb2.ge.25.and.lb2.le.27)) 
     1          then
           pprr=rtop(srt)
        endif
c
        return
        END

*****************************************
* for pi pi -> rho rho, determined from detailed balance
      real function ptor(srt)
*****************************************
      parameter (pimass=0.140,arho=0.77)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s2=srt**2
      ptor=9*(s2-4*arho**2)/(s2-4*pimass**2)*rtop(srt)

      return
      END

*****************************************
* for rho rho -> pi pi, assumed a constant cross section (in mb)
      real function rtop(srt)
*****************************************
      srt=srt
      rtop=5.
      return
      END

*****************************************
* for pi pi <-> rho rho final states
      SUBROUTINE pi2ro2(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
      PARAMETER (MAXSTR=150001)
      PARAMETER (AP1=0.13496,AP2=0.13957)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
      iseed=iseed
      if((lb(i1).ge.3.and.lb(i1).le.5)
     1     .and.(lb(i2).ge.3.and.lb(i2).le.5)) then
         iblock=1850
         ei1=0.77
         ei2=0.77
c     for now, we don't check isospin states(allowing pi+pi+ & pi0pi0 -> 2rho)
c     thus the cross sections used are considered as the isospin-averaged ones.
         lbb1=25+int(3*RANART(NSEED))
         lbb2=25+int(3*RANART(NSEED))
      elseif((lb(i1).ge.25.and.lb(i1).le.27)
     1     .and.(lb(i2).ge.25.and.lb(i2).le.27)) then
         iblock=1851
         lbb1=3+int(3*RANART(NSEED))
         lbb2=3+int(3*RANART(NSEED))
         ei1=ap2
         ei2=ap2
         if(lbb1.eq.4) ei1=ap1
         if(lbb2.eq.4) ei2=ap1
      endif

      return
      END

*****************************************
* for pi pi <-> eta eta cross sections
        SUBROUTINE sppee(lb1,lb2,srt)
        parameter (ETAM=0.5475)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

        ppee=0.
        if((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.3.and.lb2.le.5)) then
           if(srt.gt.(2*ETAM)) ppee=ptoe(srt)
        elseif(lb1.eq.0.and.lb2.eq.0) then
           ppee=etop(srt)
        endif

        return
        END

*****************************************
* for pi pi -> eta eta, determined from detailed balance, spin-isospin averaged
      real function ptoe(srt)
*****************************************
      parameter (pimass=0.140,ETAM=0.5475)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s2=srt**2
      ptoe=1./9.*(s2-4*etam**2)/(s2-4*pimass**2)*etop(srt)

      return
      END
*****************************************
* for eta eta -> pi pi, assumed a constant cross section (in mb)
      real function etop(srt)
*****************************************
      srt=srt
c     eta equilibration:
c     most important channel is found to be pi pi <-> pi eta, then
c     rho pi <-> rho eta.
      etop=5.
      return
      END

*****************************************
* for pi pi <-> eta eta final states
      SUBROUTINE pi2et2(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
      PARAMETER (MAXSTR=150001)
      PARAMETER (AP1=0.13496,AP2=0.13957,ETAM=0.5475)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

      iseed=iseed
      if((lb(i1).ge.3.and.lb(i1).le.5)
     1     .and.(lb(i2).ge.3.and.lb(i2).le.5)) then
         iblock=1860
         ei1=etam
         ei2=etam
c     for now, we don't check isospin states(allowing pi+pi+ & pi0pi0 -> 2rho)
c     thus the cross sections used are considered as the isospin-averaged ones.
         lbb1=0
         lbb2=0
      elseif(lb(i1).eq.0.and.lb(i2).eq.0) then
         iblock=1861
         lbb1=3+int(3*RANART(NSEED))
         lbb2=3+int(3*RANART(NSEED))
         ei1=ap2
         ei2=ap2
         if(lbb1.eq.4) ei1=ap1
         if(lbb2.eq.4) ei2=ap1
      endif

      return
      END

*****************************************
* for pi pi <-> pi eta cross sections
        SUBROUTINE spppe(lb1,lb2,srt)
        parameter (pimass=0.140,ETAM=0.5475)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

        pppe=0.
        if((lb1.ge.3.and.lb1.le.5).and.(lb2.ge.3.and.lb2.le.5)) then
           if(srt.gt.(ETAM+pimass)) pppe=pptope(srt)
        elseif((lb1.ge.3.and.lb1.le.5).and.lb2.eq.0) then
           pppe=petopp(srt)
        elseif((lb2.ge.3.and.lb2.le.5).and.lb1.eq.0) then
           pppe=petopp(srt)
        endif

        return
        END

*****************************************
* for pi pi -> pi eta, determined from detailed balance, spin-isospin averaged
      real function pptope(srt)
*****************************************
      parameter (pimass=0.140,ETAM=0.5475)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s2=srt**2
      pf2=(s2-(pimass+ETAM)**2)*(s2-(pimass-ETAM)**2)/2/sqrt(s2)
      pi2=(s2-4*pimass**2)*s2/2/sqrt(s2)
      pptope=1./3.*pf2/pi2*petopp(srt)

      return
      END
*****************************************
* for pi eta -> pi pi, assumed a constant cross section (in mb)
      real function petopp(srt)
*****************************************
      srt=srt
c     eta equilibration:
      petopp=5.
      return
      END

*****************************************
* for pi pi <-> pi eta final states
      SUBROUTINE pi3eta(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
      PARAMETER (MAXSTR=150001)
      PARAMETER (AP1=0.13496,AP2=0.13957,ETAM=0.5475)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

      ISEED=ISEED
      if((lb(i1).ge.3.and.lb(i1).le.5)
     1     .and.(lb(i2).ge.3.and.lb(i2).le.5)) then
         iblock=1870
         ei1=ap2
         ei2=etam
c     for now, we don't check isospin states(allowing pi+pi+ & pi0pi0 -> 2rho)
c     thus the cross sections used are considered as the isospin-averaged ones.
         lbb1=3+int(3*RANART(NSEED))
         if(lbb1.eq.4) ei1=ap1
         lbb2=0
      elseif((lb(i1).ge.3.and.lb(i1).le.5.and.lb(i2).eq.0).or.
     1        (lb(i2).ge.3.and.lb(i2).le.5.and.lb(i1).eq.0)) then
         iblock=1871
         lbb1=3+int(3*RANART(NSEED))
         lbb2=3+int(3*RANART(NSEED))
         ei1=ap2
         ei2=ap2
         if(lbb1.eq.4) ei1=ap1
         if(lbb2.eq.4) ei2=ap1
      endif

      return
      END

*****************************************
* for rho pi <-> rho eta cross sections
        SUBROUTINE srpre(lb1,lb2,srt)
        parameter (pimass=0.140,ETAM=0.5475,arho=0.77)
        common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
        common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

        rpre=0.
        if(lb1.ge.25.and.lb1.le.27.and.lb2.ge.3.and.lb2.le.5) then
           if(srt.gt.(ETAM+arho)) rpre=rptore(srt)
        elseif(lb2.ge.25.and.lb2.le.27.and.lb1.ge.3.and.lb1.le.5) then
           if(srt.gt.(ETAM+arho)) rpre=rptore(srt)
        elseif(lb1.ge.25.and.lb1.le.27.and.lb2.eq.0) then
           if(srt.gt.(pimass+arho)) rpre=retorp(srt)
        elseif(lb2.ge.25.and.lb2.le.27.and.lb1.eq.0) then
           if(srt.gt.(pimass+arho)) rpre=retorp(srt)
        endif

        return
        END

*****************************************
* for rho pi->rho eta, determined from detailed balance, spin-isospin averaged
      real function rptore(srt)
*****************************************
      parameter (pimass=0.140,ETAM=0.5475,arho=0.77)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s2=srt**2
      pf2=(s2-(arho+ETAM)**2)*(s2-(arho-ETAM)**2)/2/sqrt(s2)
      pi2=(s2-(arho+pimass)**2)*(s2-(arho-pimass)**2)/2/sqrt(s2)
      rptore=1./3.*pf2/pi2*retorp(srt)

      return
      END
*****************************************
* for rho eta -> rho pi, assumed a constant cross section (in mb)
      real function retorp(srt)
*****************************************
      srt=srt
c     eta equilibration:
      retorp=5.
      return
      END

*****************************************
* for rho pi <-> rho eta final states
      SUBROUTINE rpiret(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
      PARAMETER (MAXSTR=150001)
      PARAMETER (AP1=0.13496,AP2=0.13957,ETAM=0.5475,arho=0.77)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   
      ISEED=ISEED
      if((lb(i1).ge.25.and.lb(i1).le.27
     1     .and.lb(i2).ge.3.and.lb(i2).le.5).or.
     2     (lb(i1).ge.3.and.lb(i1).le.5
     3     .and.lb(i2).ge.25.and.lb(i2).le.27)) then
         iblock=1880
         ei1=arho
         ei2=etam
c     for now, we don't check isospin states(allowing pi+pi+ & pi0pi0 -> 2rho)
c     thus the cross sections used are considered as the isospin-averaged ones.
         lbb1=25+int(3*RANART(NSEED))
         lbb2=0
      elseif((lb(i1).ge.25.and.lb(i1).le.27.and.lb(i2).eq.0).or.
     1        (lb(i2).ge.25.and.lb(i2).le.27.and.lb(i1).eq.0)) then
         iblock=1881
         lbb1=25+int(3*RANART(NSEED))
         lbb2=3+int(3*RANART(NSEED))
         ei1=arho
         ei2=ap2
         if(lbb2.eq.4) ei2=ap1
      endif

      return
      END

*****************************************
* for omega pi <-> omega eta cross sections
        SUBROUTINE sopoe(lb1,lb2,srt)
        parameter (ETAM=0.5475,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

        xopoe=0.
        if((lb1.eq.28.and.lb2.ge.3.and.lb2.le.5).or.
     1       (lb2.eq.28.and.lb1.ge.3.and.lb1.le.5)) then
           if(srt.gt.(aomega+ETAM)) xopoe=xop2oe(srt)
        elseif((lb1.eq.28.and.lb2.eq.0).or.
     1          (lb1.eq.0.and.lb2.eq.28)) then
           if(srt.gt.(aomega+ETAM)) xopoe=xoe2op(srt)
        endif

        return
        END

*****************************************
* for omega pi -> omega eta, 
c     determined from detailed balance, spin-isospin averaged
      real function xop2oe(srt)
*****************************************
      parameter (pimass=0.140,ETAM=0.5475,aomega=0.782)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s2=srt**2
      pf2=(s2-(aomega+ETAM)**2)*(s2-(aomega-ETAM)**2)/2/sqrt(s2)
      pi2=(s2-(aomega+pimass)**2)*(s2-(aomega-pimass)**2)/2/sqrt(s2)
      xop2oe=1./3.*pf2/pi2*xoe2op(srt)

      return
      END
*****************************************
* for omega eta -> omega pi, assumed a constant cross section (in mb)
      real function xoe2op(srt)
*****************************************
      srt=srt
c     eta equilibration:
      xoe2op=5.
      return
      END

*****************************************
* for omega pi <-> omega eta final states
      SUBROUTINE opioet(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
      PARAMETER (MAXSTR=150001)
      PARAMETER (AP1=0.13496,AP2=0.13957,ETAM=0.5475,aomega=0.782)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

      iseed=iseed
      if((lb(i1).ge.3.and.lb(i1).le.5.and.lb(i2).eq.28).or.
     1     (lb(i2).ge.3.and.lb(i2).le.5.and.lb(i1).eq.28)) then
         iblock=1890
         ei1=aomega
         ei2=etam
c     for now, we don't check isospin states(allowing pi+pi+ & pi0pi0 -> 2rho)
c     thus the cross sections used are considered as the isospin-averaged ones.
         lbb1=28
         lbb2=0
      elseif((lb(i1).eq.28.and.lb(i2).eq.0).or.
     1        (lb(i1).eq.0.and.lb(i2).eq.28)) then
         iblock=1891
         lbb1=28
         lbb2=3+int(3*RANART(NSEED))
         ei1=aomega
         ei2=ap2
         if(lbb2.eq.4) ei2=ap1
      endif

      return
      END

*****************************************
* for rho rho <-> eta eta cross sections
        SUBROUTINE srree(lb1,lb2,srt)
        parameter (ETAM=0.5475,arho=0.77)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

        rree=0.
        if(lb1.ge.25.and.lb1.le.27.and.
     1       lb2.ge.25.and.lb2.le.27) then
           if(srt.gt.(2*ETAM)) rree=rrtoee(srt)
        elseif(lb1.eq.0.and.lb2.eq.0) then
           if(srt.gt.(2*arho)) rree=eetorr(srt)
        endif

        return
        END

*****************************************
* for eta eta -> rho rho
c     determined from detailed balance, spin-isospin averaged
      real function eetorr(srt)
*****************************************
      parameter (ETAM=0.5475,arho=0.77)
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      SAVE   

      s2=srt**2
      eetorr=81.*(s2-4*arho**2)/(s2-4*etam**2)*rrtoee(srt)

      return
      END
*****************************************
* for rho rho -> eta eta, assumed a constant cross section (in mb)
      real function rrtoee(srt)
*****************************************
      srt=srt
c     eta equilibration:
      rrtoee=5.
      return
      END

*****************************************
* for rho rho <-> eta eta final states
      SUBROUTINE ro2et2(i1,i2,lbb1,lbb2,ei1,ei2,iblock,iseed)
      PARAMETER (MAXSTR=150001)
      parameter (ETAM=0.5475,arho=0.77)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      common/ppb1/ene,factr2(6),fsum,ppinnb,s,wtot
cc      SAVE /ppb1/
      common/ppmm/pprr,ppee,pppe,rpre,xopoe,rree
cc      SAVE /ppmm/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

      ISEED=ISEED
      if(lb(i1).ge.25.and.lb(i1).le.27.and.
     1     lb(i2).ge.25.and.lb(i2).le.27) then
         iblock=1895
         ei1=etam
         ei2=etam
c     for now, we don't check isospin states(allowing pi+pi+ & pi0pi0 -> 2rho)
c     thus the cross sections used are considered as the isospin-averaged ones.
         lbb1=0
         lbb2=0
      elseif(lb(i1).eq.0.and.lb(i2).eq.0) then
         iblock=1896
         lbb1=25+int(3*RANART(NSEED))
         lbb2=25+int(3*RANART(NSEED))
         ei1=arho
         ei2=arho
      endif

      return
      END

*****************************
* purpose: Xsection for K* Kbar or K*bar K to pi(eta) rho(omega)
      SUBROUTINE XKKSAN(i1,i2,SRT,SIGKS1,SIGKS2,SIGKS3,SIGKS4,SIGK,prkk)
*  srt    = DSQRT(s) in GeV                                       *
*  sigk   = xsection in mb obtained from                          *
*           the detailed balance                                  *
* ***************************
          PARAMETER (AKA=0.498, PIMASS=0.140, RHOM = 0.770,aks=0.895,
     & OMEGAM = 0.7819, ETAM = 0.5473)
      PARAMETER (MAXSTR=150001)
      COMMON  /CC/      E(MAXSTR)
cc      SAVE /CC/
      SAVE   

        S = SRT ** 2
       SIGKS1 = 1.E-08
       SIGKS2 = 1.E-08
       SIGKS3 = 1.E-08
       SIGKS4 = 1.E-08

        XPION0 = prkk
clin note that prkk is for pi (rho omega) -> K* Kbar (AND!) K*bar K:
        XPION0 = XPION0/2

cc
c        PI2 = (S - (aks + AKA) ** 2) * (S - (aks - AKA) ** 2)
        PI2 = (S - (e(i1) + e(i2)) ** 2) * (S - (e(i1) - e(i2)) ** 2)
        SIGK = 1.E-08
        if(PI2 .le. 0.0) return

        XM1 = PIMASS
        XM2 = RHOM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PI2 .GT. 0.0 .AND. PF2 .GT. 0.0) THEN
           SIGKS1 = 27.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        XM1 = PIMASS
        XM2 = OMEGAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PI2 .GT. 0.0 .AND. PF2 .GT. 0.0) THEN
           SIGKS2 = 9.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        XM1 = RHOM
        XM2 = ETAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           SIGKS3 = 9.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        XM1 = OMEGAM
        XM2 = ETAM
        PF2 = (S - (XM1 + XM2) ** 2) * (S - (XM1 - XM2) ** 2)
        IF (PF2 .GT. 0.0) THEN
           SIGKS4 = 3.0 / 4.0 * PF2 / PI2 * XPION0
        END IF

        SIGK=SIGKS1+SIGKS2+SIGKS3+SIGKS4

       RETURN
        END

**********************************
*     PURPOSE:                                                         *
*     assign final states for KK*bar or K*Kbar --> light mesons
*
c      SUBROUTINE Crkspi(PX,PY,PZ,SRT,I1,I2,IBLOCK)
      SUBROUTINE crkspi(I1,I2,XSK1, XSK2, XSK3, XSK4, SIGK,
     & IBLOCK,lbp1,lbp2,emm1,emm2)
*             iblock   - 466
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1)
          PARAMETER (AP1=0.13496,AP2=0.13957,RHOM = 0.770,PI=3.1415926)
        PARAMETER (AETA=0.548,AMOMGA=0.782)
        parameter (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       IBLOCK=466
* charges of final state mesons:

        X1 = RANART(NSEED) * SIGK
        XSK2 = XSK1 + XSK2
        XSK3 = XSK2 + XSK3
        XSK4 = XSK3 + XSK4
        IF (X1 .LE. XSK1) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 25 + int(3 * RANART(NSEED))
           E(I1) = AP2
           E(I2) = rhom
        ELSE IF (X1 .LE. XSK2) THEN
           LB(I1) = 3 + int(3 * RANART(NSEED))
           LB(I2) = 28
           E(I1) = AP2
           E(I2) = AMOMGA
        ELSE IF (X1 .LE. XSK3) THEN
           LB(I1) = 0
           LB(I2) = 25 + int(3 * RANART(NSEED))
           E(I1) = AETA
           E(I2) = rhom
        ELSE
           LB(I1) = 0
           LB(I2) = 28
           E(I1) = AETA
           E(I2) = AMOMGA
        ENDIF

        if(lb(i1).eq.4) E(I1) = AP1
        lbp1=lb(i1)
        lbp2=lb(i2)
        emm1=e(i1)
        emm2=e(i2)

      RETURN
      END

*---------------------------------------------------------------------------
* PURPOSE : CALCULATE THE MASS AND MOMENTUM OF K* RESONANCE 
*           AFTER PION + KAON COLLISION
*clin only here the K* mass may be different from aks=0.895
        SUBROUTINE KSRESO(I1,I2)
        PARAMETER (MAXSTR=150001,MAXR=1,
     1  AMN=0.939457,AMP=0.93828,
     2  AP1=0.13496,AP2=0.13957,AM0=1.232,PI=3.1415926)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
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
      SAVE   
* 1. DETERMINE THE MOMENTUM COMPONENT OF THE K* IN THE CMS OF PI-K FRAME
*    WE LET I1 TO BE THE K* AND ABSORB I2
        E10=SQRT(E(I1)**2+P(1,I1)**2+P(2,I1)**2+P(3,I1)**2)
        E20=SQRT(E(I2)**2+P(1,I2)**2+P(2,I2)**2+P(3,I2)**2)
        IF(LB(I2) .EQ. 21 .OR. LB(I2) .EQ. 23) THEN
        E(I1)=0.
        I=I2
        ELSE
        E(I2)=0.
        I=I1
        ENDIF
        if(LB(I).eq.23) then
           LB(I)=30
        else if(LB(I).eq.21) then
           LB(I)=-30
        endif
        P(1,I)=P(1,I1)+P(1,I2)
        P(2,I)=P(2,I1)+P(2,I2)
        P(3,I)=P(3,I1)+P(3,I2)
* 2. DETERMINE THE MASS OF K* BY USING THE REACTION KINEMATICS
        DM=SQRT((E10+E20)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2)
        E(I)=DM
        RETURN
        END

c--------------------------------------------------------
*************************************
*                                                                         *
      SUBROUTINE pertur(PX,PY,PZ,SRT,IRUN,I1,I2,nt,kp,icont)
*                                                                         *
*       PURPOSE:   TO PRODUCE CASCADE AND OMEGA PERTURBATIVELY            *
c sp 01/03/01
*                   40 cascade-
*                  -40 cascade-(bar)
*                   41 cascade0
*                  -41 cascade0(bar)
*                   45 Omega baryon
*                  -45 Omega baryon(bar)
*                   44 Di-Omega
**********************************
      PARAMETER      (MAXSTR=150001,MAXR=1,PI=3.1415926)
      parameter      (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
      PARAMETER (AMN=0.939457,AMP=0.93828,AP1=0.13496,AP2=0.13957)
      PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974,aks=0.895)
      PARAMETER      (ACAS=1.3213,AOME=1.6724,AMRHO=0.769,AMOMGA=0.782)
      PARAMETER      (AETA=0.548,ADIOMG=3.2288)
      parameter            (maxx=20,maxz=24)
      COMMON   /AA/  R(3,MAXSTR)
cc      SAVE /AA/
      COMMON   /BB/  P(3,MAXSTR)
cc      SAVE /BB/
      COMMON   /CC/  E(MAXSTR)
cc      SAVE /CC/
      COMMON   /EE/  ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      COMMON   /HH/  PROPER(MAXSTR)
cc      SAVE /HH/
      common /ff/f(-mx:mx,-my:my,-mz:mz,-mpx:mpx,-mpy:mpy,-mpz:mpzp)
cc      SAVE /ff/
      common   /gg/  dx,dy,dz,dpx,dpy,dpz
cc      SAVE /gg/
      COMMON   /INPUT/ NSTAR,NDIRCT,DIR
cc      SAVE /INPUT/
      COMMON   /NN/NNN
cc      SAVE /NN/
      COMMON   /PA/RPION(3,MAXSTR,MAXR)
cc      SAVE /PA/
      COMMON   /PB/PPION(3,MAXSTR,MAXR)
cc      SAVE /PB/
      COMMON   /PC/EPION(MAXSTR,MAXR)
cc      SAVE /PC/
      COMMON   /PD/LPION(MAXSTR,MAXR)
cc      SAVE /PD/
      COMMON   /PE/PROPI(MAXSTR,MAXR)
cc      SAVE /PE/
      COMMON   /RR/  MASSR(0:MAXR)
cc      SAVE /RR/
      COMMON   /BG/BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
c     perturbative method is disabled:
c      common /imulst/ iperts
c
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      SAVE   
      kp=kp
      nt=nt

      px0 = px
      py0 = py
      pz0 = pz
      LB1 = LB(I1)
      EM1 = E(I1)
      X1  = R(1,I1)
      Y1  = R(2,I1)
      Z1  = R(3,I1)
      prob1 = PROPER(I1)
c     
      LB2 = LB(I2)
      EM2 = E(I2)
      X2  = R(1,I2)
      Y2  = R(2,I2)
      Z2  = R(3,I2)
      prob2 = PROPER(I2)
c
c                 !! flag for real 2-body process (1/0=no/yes)
      icont = 1
c                !! flag for elastic scatt only (-1=no)
      icsbel = -1

* K-/K*0bar + La/Si --> cascade + pi
* K+/K*0 + La/Si (bar) --> cascade-bar + pi
       if( (lb1.eq.21.or.lb1.eq.23.or.iabs(lb1).eq.30) .and.
     &     (iabs(lb2).ge.14.and.iabs(lb2).le.17) )go to 60
       if( (lb2.eq.21.or.lb2.eq.23.or.iabs(lb2).eq.30) .and.
     &     (iabs(lb1).ge.14.and.iabs(lb1).le.17) )go to 60
* K-/K*0bar + cascade --> omega + pi
* K+/K*0 + cascade-bar --> omega-bar + pi
        if( (lb1.eq.21.or.lb1.eq.23.or.iabs(lb1).eq.30) .and.
     &      (iabs(lb2).eq.40.or.iabs(lb2).eq.41) )go to 70
        if( (lb2.eq.21.or.lb2.eq.23.or.iabs(lb2).eq.30) .and.
     &      (iabs(lb1).eq.40.or.iabs(lb1).eq.41) )go to 70
c
c annhilation of cascade,cascade-bar, omega,omega-bar
c
* K- + La/Si <-- cascade + pi(eta,rho,omega)
* K+ + La/Si(bar) <-- cascade-bar + pi(eta,rho,omega)
       if( (((lb1.ge.3.and.lb1.le.5).or.lb1.eq.0) 
     &        .and.(iabs(lb2).eq.40.or.iabs(lb2).eq.41))
     & .OR. (((lb2.ge.3.and.lb2.le.5).or.lb2.eq.0) 
     &        .and.(iabs(lb1).eq.40.or.iabs(lb1).eq.41)) )go to 90
* K- + cascade <-- omega + pi
* K+ + cascade-bar <-- omega-bar + pi
c         if( (lb1.eq.0.and.iabs(lb2).eq.45)
c    &    .OR. (lb2.eq.0.and.iabs(lb1).eq.45) ) go to 110
       if( ((lb1.ge.3.and.lb1.le.5).and.iabs(lb2).eq.45)
     & .OR.((lb2.ge.3.and.lb2.le.5).and.iabs(lb1).eq.45) )go to 110
c

c----------------------------------------------------
*  for process:  K-bar + L(S) --> Ca + pi 
*
C...Fix compiler warnings
C60         if(iabs(lb1).ge.14 .and. iabs(lb1).le.17)then 
60         if(lb1a.ge.14 .and. lb1a.le.17)then 
             asap = e(i1)
             akap = e(i2)
             idp = i1
           else
             asap = e(i2)
             akap = e(i1)
             idp = i2
           endif
          app = 0.138
         if(srt .lt. (acas+app))return
          srrt = srt - (acas+app) + (amn+akap)
          pkaon = sqrt(((srrt**2-(amn**2+akap**2))/2./amn)**2 - akap**2)
          sigca = 1.5*( akNPsg(pkaon)+akNPsg(pkaon) )
clin pii & pff should be each divided by (4*srt**2), 
c     but these two factors cancel out in the ratio pii/pff:
          pii = sqrt((srt**2-(amn+akap)**2)*(srt**2-(amn-akap)**2))
          pff = sqrt((srt**2-(asap+app)**2)*(srt**2-(asap-app)**2))
         cmat = sigca*pii/pff
         sigpi = cmat*
     &            sqrt((srt**2-(acas+app)**2)*(srt**2-(acas-app)**2))/
     &            sqrt((srt**2-(asap+akap)**2)*(srt**2-(asap-akap)**2))
c 
         sigeta = 0.
        if(srt .gt. (acas+aeta))then
           srrt = srt - (acas+aeta) + (amn+akap)
         pkaon = sqrt(((srrt**2-(amn**2+akap**2))/2./amn)**2 - akap**2)
            sigca = 1.5*( akNPsg(pkaon)+akNPsg(pkaon) )
         cmat = sigca*pii/pff
         sigeta = cmat*
     &            sqrt((srt**2-(acas+aeta)**2)*(srt**2-(acas-aeta)**2))/
     &            sqrt((srt**2-(asap+akap)**2)*(srt**2-(asap-akap)**2))
        endif
c
         sigca = sigpi + sigeta
         sigpe = 0.
clin-2/25/03 disable the perturb option:
c        if(iperts .eq. 1) sigpe = 40.   !! perturbative xsecn
           sig = amax1(sigpe,sigca)     
         ds = sqrt(sig/31.4)
         dsr = ds + 0.1
         ec = (em1+em2+0.02)**2
         call distce(i1,i2,dsr,ds,dt,ec,srt,ic,px,py,pz)
           if(ic .eq. -1)return
          brpp = sigca/sig
c
c else particle production
          if( (lb1.ge.14.and.lb1.le.17) .or.
     &          (lb2.ge.14.and.lb2.le.17) )then
c   !! cascade- or cascde0
            lbpp1 = 40 + int(2*RANART(NSEED))
          else
* elseif(lb1 .eq. -14 .or. lb2 .eq. -14)
c     !! cascade-bar- or cascde0 -bar
            lbpp1 = -40 - int(2*RANART(NSEED))
          endif
              empp1 = acas
           if(RANART(NSEED) .lt. sigpi/sigca)then
c    !! pion
            lbpp2 = 3 + int(3*RANART(NSEED))
            empp2 = 0.138
           else
c    !! eta
            lbpp2 = 0
            empp2 = aeta
           endif        
c* check real process of cascade(bar) and pion formation
          if(RANART(NSEED) .lt. brpp)then
c       !! real process flag
            icont = 0
            lb(i1) = lbpp1
            e(i1) = empp1
c  !! cascade formed with prob Gam
            proper(i1) = brpp
            lb(i2) = lbpp2
            e(i2) = empp2
c         !! pion/eta formed with prob 1.
            proper(i2) = 1.
           endif
c else only cascade(bar) formed perturbatively
             go to 700
            
c----------------------------------------------------
*  for process:  Cas(bar) + K_bar(K) --> Om(bar) + pi  !! eta
*
C...fix to compiler warning  
C70         if(iabs(lb1).eq.40 .or. iabs(lb1).eq.41)then 
           lb1a=iabs(lb1)
70         if(lb1a.eq.40 .or. lb1a.eq.41)then 
             acap = e(i1)
             akap = e(i2)
             idp = i1
           else
             acap = e(i2)
             akap = e(i1)
             idp = i2
           endif
           app = 0.138
*         ames = aeta
c  !! only pion
           ames = 0.138
         if(srt .lt. (aome+ames))return 
          srrt = srt - (aome+ames) + (amn+akap)
         pkaon = sqrt(((srrt**2-(amn**2+akap**2))/2./amn)**2 - akap**2)
c use K(bar) + Ca --> Om + eta  xsecn same as  K(bar) + N --> Si + Pi
*  as Omega have no resonances
c** using same matrix elements as K-bar + N -> Si + pi
         sigomm = 1.5*( akNPsg(pkaon)+akNPsg(pkaon) )
         cmat = sigomm*
     &          sqrt((srt**2-(amn+akap)**2)*(srt**2-(amn-akap)**2))/
     &          sqrt((srt**2-(asa+app)**2)*(srt**2-(asa-app)**2))
        sigom = cmat*
     &           sqrt((srt**2-(aome+ames)**2)*(srt**2-(aome-ames)**2))/
     &           sqrt((srt**2-(acap+akap)**2)*(srt**2-(acap-akap)**2))
          sigpe = 0.
clin-2/25/03 disable the perturb option:
c         if(iperts .eq. 1) sigpe = 40.   !! perturbative xsecn
          sig = amax1(sigpe,sigom)     
         ds = sqrt(sig/31.4)
         dsr = ds + 0.1
         ec = (em1+em2+0.02)**2
         call distce(i1,i2,dsr,ds,dt,ec,srt,ic,px,py,pz)
           if(ic .eq. -1)return
           brpp = sigom/sig
c
c else particle production
           if( (lb1.ge.40.and.lb1.le.41) .or.
     &           (lb2.ge.40.and.lb2.le.41) )then
c    !! omega
            lbpp1 = 45
           else
* elseif(lb1 .eq. -40 .or. lb2 .eq. -40)
c    !! omega-bar
            lbpp1 = -45
           endif
           empp1 = aome
*           lbpp2 = 0    !! eta
c    !! pion
           lbpp2 = 3 + int(3*RANART(NSEED))
           empp2 = ames
c
c* check real process of omega(bar) and pion formation
           xrand=RANART(NSEED)
         if(xrand .lt. (proper(idp)*brpp))then
c       !! real process flag
            icont = 0
            lb(i1) = lbpp1
            e(i1) = empp1
c  !! P_Om = P_Cas*Gam
            proper(i1) = proper(idp)*brpp
            lb(i2) = lbpp2
            e(i2) = empp2
c   !! pion formed with prob 1.
            proper(i2) = 1.
          elseif(xrand.lt.brpp) then
c else omega(bar) formed perturbatively and cascade destroyed
             e(idp) = 0.
          endif
             go to 700
            
c-----------------------------------------------------------
*  for process:  Ca + pi/eta --> K-bar + L(S)
*
C...Fxi compiler warning
C90         if(iabs(lb1).eq.40 .or. iabs(lb1).eq.41)then 
90         if(lb1a.eq.40 .or. lb1a.eq.41)then 
             acap = e(i1)
             app = e(i2)
             idp = i1
             idn = i2
           else
             acap = e(i2)
             app = e(i1)
             idp = i2
             idn = i1
           endif
c            akal = (aka+aks)/2.  !! average of K and K* taken
c  !! using K only
            akal = aka
c
         alas = ala
       if(srt .le. (alas+aka))return
           srrt = srt - (acap+app) + (amn+aka)
         pkaon = sqrt(((srrt**2-(amn**2+aka**2))/2./amn)**2 - aka**2)
c** using same matrix elements as K-bar + N -> La/Si + pi
         sigca = 1.5*( akNPsg(pkaon)+akNPsg(pkaon) )
         cmat = sigca*
     &          sqrt((srt**2-(amn+aka)**2)*(srt**2-(amn-aka)**2))/
     &          sqrt((srt**2-(alas+0.138)**2)*(srt**2-(alas-0.138)**2))
         sigca = cmat*
     &            sqrt((srt**2-(acap+app)**2)*(srt**2-(acap-app)**2))/
     &            sqrt((srt**2-(alas+aka)**2)*(srt**2-(alas-aka)**2))
c    !! pi
            dfr = 1./3.
c       !! eta
           if(lb(idn).eq.0)dfr = 1.
        sigcal = sigca*dfr*(srt**2-(alas+aka)**2)*
     &           (srt**2-(alas-aka)**2)/(srt**2-(acap+app)**2)/
     &           (srt**2-(acap-app)**2)
c
          alas = ASA
       if(srt .le. (alas+aka))then
         sigcas = 0.
       else
           srrt = srt - (acap+app) + (amn+aka)
        pkaon = sqrt(((srrt**2-(amn**2+aka**2))/2./amn)**2 - aka**2)
c use K(bar) + La/Si --> Ca + Pi  xsecn same as  K(bar) + N --> Si + Pi
c** using same matrix elements as K-bar + N -> La/Si + pi
          sigca = 1.5*( akNPsg(pkaon)+akNPsg(pkaon) )
         cmat = sigca*
     &          sqrt((srt**2-(amn+aka)**2)*(srt**2-(amn-aka)**2))/
     &          sqrt((srt**2-(alas+0.138)**2)*(srt**2-(alas-0.138)**2))
         sigca = cmat*
     &            sqrt((srt**2-(acap+app)**2)*(srt**2-(acap-app)**2))/
     &            sqrt((srt**2-(alas+aka)**2)*(srt**2-(alas-aka)**2))
c    !! pi
            dfr = 1.
c    !! eta
           if(lb(idn).eq.0)dfr = 3.
        sigcas = sigca*dfr*(srt**2-(alas+aka)**2)*
     &           (srt**2-(alas-aka)**2)/(srt**2-(acap+app)**2)/
     &           (srt**2-(acap-app)**2)
       endif
c
         sig = sigcal + sigcas
         brpp = 1.                                                   
         ds = sqrt(sig/31.4)
         dsr = ds + 0.1
         ec = (em1+em2+0.02)**2
         call distce(i1,i2,dsr,ds,dt,ec,srt,ic,px,py,pz)
c
clin-2/25/03: checking elastic scatt after failure of inelastic scatt gives 
c     conditional probability (in general incorrect), tell Pal to correct:
       if(ic .eq. -1)then
c check for elastic scatt, no particle annhilation
c  !! elastic cross section of 20 mb
         ds = sqrt(20.0/31.4)
         dsr = ds + 0.1
         call distce(i1,i2,dsr,ds,dt,ec,srt,icsbel,px,py,pz)
           if(icsbel .eq. -1)return
            empp1 = EM1
            empp2 = EM2
             go to 700
       endif
c
c else pert. produced cascade(bar) is annhilated OR real process
c
* DECIDE LAMBDA OR SIGMA PRODUCTION
c
       IF(sigcal/sig .GT. RANART(NSEED))THEN  
          if(lb1.eq.40.or.lb1.eq.41.or.lb2.eq.40.or.lb2.eq.41)then
          lbpp1 = 21
           lbpp2 = 14
          else
           lbpp1 = 23
           lbpp2 = -14
          endif
         alas = ala
       ELSE
          if(lb1.eq.40.or.lb1.eq.41.or.lb2.eq.40.or.lb2.eq.41)then
           lbpp1 = 21
            lbpp2 = 15 + int(3 * RANART(NSEED))
          else
            lbpp1 = 23
            lbpp2 = -15 - int(3 * RANART(NSEED))
          endif
         alas = ASA       
        ENDIF
             empp1 = aka  
             empp2 = alas 
c
c check for real process for L/S(bar) and K(bar) formation
          if(RANART(NSEED) .lt. proper(idp))then
* real process
c       !! real process flag
            icont = 0
            lb(i1) = lbpp1
            e(i1) = empp1
c   !! K(bar) formed with prob 1.
            proper(i1) = 1.
            lb(i2) = lbpp2
            e(i2) = empp2
c   !! L/S(bar) formed with prob 1.
            proper(i2) = 1.
             go to 700
           else
c else only cascade(bar) annhilation & go out
            e(idp) = 0.
           endif
          return
c
c----------------------------------------------------
*  for process:  Om(bar) + pi --> Cas(bar) + K_bar(K)
*
110         if(lb1 .eq. 45 .or. lb1 .eq. -45)then 
             aomp = e(i1)
             app = e(i2)
             idp = i1
             idn = i2
           else
             aomp = e(i2)
             app = e(i1)
             idp = i2
             idn = i1
           endif
c            akal = (aka+aks)/2.  !! average of K and K* taken 
c  !! using K only
            akal = aka
       if(srt .le. (acas+aka))return
           srrt = srt - (aome+app) + (amn+aka)
         pkaon = sqrt(((srrt**2-(amn**2+aka**2))/2./amn)**2 - aka**2)
c use K(bar) + Ca --> Om + eta  xsecn same as  K(bar) + N --> Si + Pi
c** using same matrix elements as K-bar + N -> La/Si + pi
           sigca = 1.5*( akNPsg(pkaon)+akNPsg(pkaon) )
         cmat = sigca*
     &          sqrt((srt**2-(amn+aka)**2)*(srt**2-(amn-aka)**2))/
     &          sqrt((srt**2-(asa+0.138)**2)*(srt**2-(asa-0.138)**2))
         sigom = cmat*
     &            sqrt((srt**2-(aomp+app)**2)*(srt**2-(aomp-app)**2))/
     &            sqrt((srt**2-(acas+aka)**2)*(srt**2-(acas-aka)**2))
c            dfr = 2.    !! eta
c    !! pion
           dfr = 2./3.
        sigom = sigom*dfr*(srt**2-(acas+aka)**2)*
     &           (srt**2-(acas-aka)**2)/(srt**2-(aomp+app)**2)/
     &           (srt**2-(aomp-app)**2)
c                                                                         
         brpp = 1.
         ds = sqrt(sigom/31.4)
         dsr = ds + 0.1
         ec = (em1+em2+0.02)**2
         call distce(i1,i2,dsr,ds,dt,ec,srt,ic,px,py,pz)
c
clin-2/25/03: checking elastic scatt after failure of inelastic scatt gives 
c     conditional probability (in general incorrect), tell Pal to correct:
       if(ic .eq. -1)then
c check for elastic scatt, no particle annhilation
c  !! elastic cross section of 20 mb
         ds = sqrt(20.0/31.4)
         dsr = ds + 0.1
         call distce(i1,i2,dsr,ds,dt,ec,srt,icsbel,px,py,pz)
           if(icsbel .eq. -1)return
            empp1 = EM1
            empp2 = EM2
             go to 700
       endif
c
c else pert. produced omega(bar) annhilated  OR real process
c annhilate only pert. omega, rest from hijing go out WITHOUT annhil.
           if(lb1.eq.45 .or. lb2.eq.45)then
c  !! Ca
             lbpp1 = 40 + int(2*RANART(NSEED))
c   !! K-
             lbpp2 = 21
            else
* elseif(lb1 .eq. -45 .or. lb2 .eq. -45)
c    !! Ca-bar
            lbpp1 = -40 - int(2*RANART(NSEED))
c      !! K+
            lbpp2 = 23
           endif
             empp1 = acas
             empp2 = aka  
c
c check for real process for Cas(bar) and K(bar) formation
          if(RANART(NSEED) .lt. proper(idp))then
c       !! real process flag
            icont = 0
            lb(i1) = lbpp1
            e(i1) = empp1
c   !! P_Cas(bar) = P_Om(bar)
            proper(i1) = proper(idp)
            lb(i2) = lbpp2
            e(i2) = empp2
c   !! K(bar) formed with prob 1.
            proper(i2) = 1.
c
           else
c else Cascade(bar)  produced and Omega(bar) annhilated
            e(idp) = 0.
           endif
c   !! for produced particles
             go to 700
c
c-----------------------------------------------------------
700    continue
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EMpp1**2 - EMpp2**2)**2
     &                - 4.0 * (EMpp1*EMpp2)**2
          IF(PR2.LE.0.)PR2=0.00000001
          PR=SQRT(PR2)/(2.*SRT)
* using isotropic
      C1   = 1.0 - 2.0 * RANART(NSEED)
      T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
* ROTATE IT 
       CALL ROTATE(PX0,PY0,PZ0,PX,PY,PZ) 
       if(icont .eq. 0)return
c
* LORENTZ-TRANSFORMATION INTO CMS FRAME
              E1CM    = SQRT (EMpp1**2 + PX**2 + PY**2 + PZ**2)
              P1BETA  = PX*BETAX + PY*BETAY + PZ*BETAZ
              TRANSF  = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) + E1CM )
              Ppt11 = BETAX * TRANSF + PX
              Ppt12 = BETAY * TRANSF + PY
              Ppt13 = BETAZ * TRANSF + PZ
c
cc** for elastic scattering update the momentum of pertb particles
         if(icsbel .ne. -1)then
c            if(EMpp1 .gt. 0.9)then
              p(1,i1) = Ppt11
              p(2,i1) = Ppt12
              p(3,i1) = Ppt13
c            else
              E2CM    = SQRT (EMpp2**2 + PX**2 + PY**2 + PZ**2)
              TRANSF  = GAMMA * ( -GAMMA * P1BETA / (GAMMA + 1) + E2CM )
              Ppt21 = BETAX * TRANSF - PX
              Ppt22 = BETAY * TRANSF - PY
              Ppt23 = BETAZ * TRANSF - PZ
              p(1,i2) = Ppt21
              p(2,i2) = Ppt22
              p(3,i2) = Ppt23
c            endif
             return
          endif
clin-5/2008:
c2008        X01 = 1.0 - 2.0 * RANART(NSEED)
c            Y01 = 1.0 - 2.0 * RANART(NSEED)
c            Z01 = 1.0 - 2.0 * RANART(NSEED)
c        IF ((X01*X01+Y01*Y01+Z01*Z01) .GT. 1.0) GOTO 2008
c                Xpt=X1+0.5*x01
c                Ypt=Y1+0.5*y01
c                Zpt=Z1+0.5*z01
                Xpt=X1
                Ypt=Y1
                Zpt=Z1
c
c
c          if(lbpp1 .eq. 45)then
c           write(*,*)'II lb1,lb2,lbpp1,empp1,proper(idp),brpp'
c           write(*,*)lb1,lb2,lbpp1,empp1,proper(idp),brpp
c          endif
c
               NNN=NNN+1
               PROPI(NNN,IRUN)= proper(idp)*brpp
               LPION(NNN,IRUN)= lbpp1
               EPION(NNN,IRUN)= empp1
                RPION(1,NNN,IRUN)=Xpt
                RPION(2,NNN,IRUN)=Ypt
                RPION(3,NNN,IRUN)=Zpt
               PPION(1,NNN,IRUN)=Ppt11
               PPION(2,NNN,IRUN)=Ppt12
               PPION(3,NNN,IRUN)=Ppt13
clin-5/2008:
               dppion(nnn,irun)=dpertp(i1)*dpertp(i2)
            RETURN
            END
**********************************
*  sp 12/08/00                                                         *
      SUBROUTINE Crhb(PX,PY,PZ,SRT,I1,I2,IBLOCK)
*     PURPOSE:                                                         *
*        DEALING WITH hyperon+N(D,N*)->hyp+N(D,N*) elastic PROCESS     *
*     NOTE   :                                                         *
*          
*     QUANTITIES:                                                 *
*           PX,PY,PZ - MOMENTUM COORDINATES OF ONE PARTICLE IN CM FRAME*
*           SRT      - SQRT OF S                                       *
*           IBLOCK   - THE INFORMATION BACK                            *
*                     144-> hyp+N(D,N*)->hyp+N(D,N*)
**********************************
        PARAMETER (MAXSTR=150001,MAXR=1,AMN=0.939457,
     1  AMP=0.93828,AP1=0.13496,
     2  AP2=0.13957,AM0=1.232,PI=3.1415926,CUTOFF=1.8966,AVMASS=0.9383)
        PARAMETER      (AKA=0.498,ALA=1.1157,ASA=1.1974)
        parameter     (MX=4,MY=4,MZ=8,MPX=4,MPY=4,mpz=10,mpzp=10)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
        common/input1/ MASSPR,MASSTA,ISEED,IAVOID,DT
cc      SAVE /input1/
      COMMON/RNDF77/NSEED
cc      SAVE /RNDF77/
      SAVE   

       PX0=PX
       PY0=PY
       PZ0=PZ
*-----------------------------------------------------------------------
        IBLOCK=144
        NTAG=0
        EM1=E(I1)
        EM2=E(I2)
*-----------------------------------------------------------------------
* CALCULATE THE MAGNITUDE OF THE FINAL MOMENTUM THROUGH
* ENERGY CONSERVATION
          PR2   = (SRT**2 - EM1**2 - EM2**2)**2
     1                - 4.0 * (EM1*EM2)**2
          IF(PR2.LE.0.)PR2=1.e-09
          PR=SQRT(PR2)/(2.*SRT)
          C1   = 1.0 - 2.0 * RANART(NSEED)
          T1   = 2.0 * PI * RANART(NSEED)
      S1   = SQRT( 1.0 - C1**2 )
      CT1  = COS(T1)
      ST1  = SIN(T1)
      PZ   = PR * C1
      PX   = PR * S1*CT1 
      PY   = PR * S1*ST1
      RETURN
      END
****************************************
c sp 04/05/01
* Purpose: lambda-baryon elastic xsection as a functon of their cms energy
         subroutine lambar(i1,i2,srt,siglab)
*  srt    = DSQRT(s) in GeV                                               *
*  siglab = lambda-nuclar elastic cross section in mb 
*         = 12 + 0.43/p_lab**3.3 (mb)  
*                                                    
* (2) Calculate p(lab) from srt [GeV], since the formular in the 
* reference applies only to the case of a p_bar on a proton at rest
* Formula used: srt**2=2.*pmass*(pmass+sqrt(pmass**2+plab**2))
*****************************
        PARAMETER (MAXSTR=150001)
        COMMON /AA/ R(3,MAXSTR)
cc      SAVE /AA/
        COMMON /BB/ P(3,MAXSTR)
cc      SAVE /BB/
        COMMON /CC/ E(MAXSTR)
cc      SAVE /CC/
        COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
cc      SAVE /EE/
      SAVE   

          siglab=1.e-06
        if( iabs(lb(i1)).ge.14.and.iabs(lb(i1)).le.17 )then
          eml = e(i1)
          emb = e(i2)
         else
          eml = e(i2)
          emb = e(i1)
        endif
       pthr = srt**2-eml**2-emb**2
        if(pthr .gt. 0.)then
       plab2=(pthr/2./emb)**2-eml**2
       if(plab2.gt.0)then
         plab=sqrt(plab2)
         siglab=12. + 0.43/(plab**3.3)
       if(siglab.gt.200.)siglab=200.
       endif
       endif
         return
      END
C------------------------------------------------------------------
clin-7/26/03 improve speed
***************************************
            SUBROUTINE distc0(drmax,deltr0,DT,
     1     Ifirst,PX1CM,PY1CM,PZ1CM,
     2     x1,y1,z1,px1,py1,pz1,em1,x2,y2,z2,px2,py2,pz2,em2)
* PURPOSE : CHECK IF THE COLLISION BETWEEN TWO PARTICLES CAN HAPPEN
*           BY CHECKING
*                      (2) IF PARTICLE WILL PASS EACH OTHER WITHIN
*           TWO HARD CORE RADIUS.
*                      (3) IF PARTICLES WILL GET CLOSER.
* VARIABLES :
*           Ifirst=1 COLLISION may HAPPENED
*           Ifirst=-1 COLLISION CAN NOT HAPPEN
*****************************************
            COMMON   /BG/  BETAX,BETAY,BETAZ,GAMMA
cc      SAVE /BG/
      SAVE   
            deltr0=deltr0 
            Ifirst=-1
            E1=SQRT(EM1**2+PX1**2+PY1**2+PZ1**2)
*NOW PARTICLES ARE CLOSE ENOUGH TO EACH OTHER !
            E2     = SQRT ( EM2**2 + PX2**2 + PY2**2 + PZ2**2 )
*NOW THERE IS ENOUGH ENERGY AVAILABLE !
*LORENTZ-TRANSFORMATION IN I1-I2-C.M. SYSTEM
* BETAX, BETAY, BETAZ AND GAMMA HAVE BEEN GIVEN IN THE SUBROUTINE CMS
*TRANSFORMATION OF MOMENTA (PX1CM = - PX2CM)
              P1BETA = PX1*BETAX + PY1*BETAY + PZ1 * BETAZ
              TRANSF = GAMMA * ( GAMMA * P1BETA / (GAMMA + 1) - E1 )
              PRCM   = SQRT (PX1CM**2 + PY1CM**2 + PZ1CM**2)
              IF (PRCM .LE. 0.00001) return
*TRANSFORMATION OF SPATIAL DISTANCE
              DRBETA = BETAX*(X1-X2) + BETAY*(Y1-Y2) + BETAZ*(Z1-Z2)
              TRANSF = GAMMA * GAMMA * DRBETA / (GAMMA + 1)
              DXCM   = BETAX * TRANSF + X1 - X2
              DYCM   = BETAY * TRANSF + Y1 - Y2
              DZCM   = BETAZ * TRANSF + Z1 - Z2
*DETERMINING IF THIS IS THE POINT OF CLOSEST APPROACH
              DRCM   = SQRT (DXCM**2  + DYCM**2  + DZCM**2 )
              DZZ    = (PX1CM*DXCM + PY1CM*DYCM + PZ1CM*DZCM) / PRCM
              if ((drcm**2 - dzz**2) .le. 0.) then
                BBB = 0.
              else
                BBB    = SQRT (DRCM**2 - DZZ**2)
              end if
*WILL PARTICLE PASS EACH OTHER WITHIN 2 * HARD CORE RADIUS ?
              IF (BBB .GT. drmax) return
              RELVEL = PRCM * (1.0/E1 + 1.0/E2)
              DDD    = RELVEL * DT * 0.5
*WILL PARTICLES GET CLOSER ?
              IF (ABS(DDD) .LT. ABS(DZZ)) return
              Ifirst=1
              RETURN
              END
*---------------------------------------------------------------------------
c
clin-8/2008 B+B->Deuteron+Meson cross section in mb:
      subroutine sbbdm(srt,sdprod,ianti,lbm,xmm,pfinal)
      PARAMETER (xmd=1.8756,AP1=0.13496,AP2=0.13957,
     1     xmrho=0.770,xmomega=0.782,xmeta=0.548,srt0=2.012)
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      common /para8/ idpert,npertd,idxsec
      COMMON/RNDF77/NSEED
      SAVE   
c
      sdprod=0.
      sbbdpi=0.
      sbbdrho=0.
      sbbdomega=0.
      sbbdeta=0.
      if(srt.le.(em1+em2)) return
c
      ilb1=iabs(lb1)
      ilb2=iabs(lb2)
ctest off check Xsec using fixed mass for resonances:
c      if(ilb1.ge.6.and.ilb1.le.9) then
c         em1=1.232
c      elseif(ilb1.ge.10.and.ilb1.le.11) then
c         em1=1.44
c      elseif(ilb1.ge.12.and.ilb1.le.13) then
c         em1=1.535
c      endif
c      if(ilb2.ge.6.and.ilb2.le.9) then
c         em2=1.232
c      elseif(ilb2.ge.10.and.ilb2.le.11) then
c         em2=1.44
c      elseif(ilb2.ge.12.and.ilb2.le.13) then
c         em2=1.535
c      endif
c
      s=srt**2
      pinitial=sqrt((s-(em1+em2)**2)*(s-(em1-em2)**2))/2./srt
      fs=fnndpi(s)
c     Determine isospin and spin factors for the ratio between 
c     BB->Deuteron+Meson and Deuteron+Meson->BB cross sections:
      if(idxsec.eq.1.or.idxsec.eq.2) then
c     Assume B+B -> d+Meson has the same cross sections as N+N -> d+pi:
      else
c     Assume d+Meson -> B+B has the same cross sections as d+pi -> N+N, 
c     then determine B+B -> d+Meson cross sections:
         if(ilb1.ge.1.and.ilb1.le.2.and.
     1        ilb2.ge.1.and.ilb2.le.2) then
            pifactor=9./8.
         elseif((ilb1.ge.1.and.ilb1.le.2.and.
     1           ilb2.ge.6.and.ilb2.le.9).or.
     2           (ilb2.ge.1.and.ilb2.le.2.and.
     1           ilb1.ge.6.and.ilb1.le.9)) then
            pifactor=9./64.
         elseif((ilb1.ge.1.and.ilb1.le.2.and.
     1           ilb2.ge.10.and.ilb2.le.13).or.
     2           (ilb2.ge.1.and.ilb2.le.2.and.
     1           ilb1.ge.10.and.ilb1.le.13)) then
            pifactor=9./16.
         elseif(ilb1.ge.6.and.ilb1.le.9.and.
     1           ilb2.ge.6.and.ilb2.le.9) then
            pifactor=9./128.
         elseif((ilb1.ge.6.and.ilb1.le.9.and.
     1           ilb2.ge.10.and.ilb2.le.13).or.
     2           (ilb2.ge.6.and.ilb2.le.9.and.
     1           ilb1.ge.10.and.ilb1.le.13)) then
            pifactor=9./64.
         elseif((ilb1.ge.10.and.ilb1.le.11.and.
     1           ilb2.ge.10.and.ilb2.le.11).or.
     2           (ilb2.ge.12.and.ilb2.le.13.and.
     1           ilb1.ge.12.and.ilb1.le.13)) then
            pifactor=9./8.
         elseif((ilb1.ge.10.and.ilb1.le.11.and.
     1           ilb2.ge.12.and.ilb2.le.13).or.
     2           (ilb2.ge.10.and.ilb2.le.11.and.
     1           ilb1.ge.12.and.ilb1.le.13)) then
            pifactor=9./16.
         endif
      endif
c     d pi: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
*     (1) FOR P+P->Deuteron+pi+:
      IF((ilb1*ilb2).EQ.1)THEN
         lbm=5
         if(ianti.eq.1) lbm=3
         xmm=ap2
*     (2)FOR N+N->Deuteron+pi-:
      ELSEIF(ilb1.EQ.2.AND.ilb2.EQ.2)THEN
         lbm=3
         if(ianti.eq.1) lbm=5
         xmm=ap2
*     (3)FOR N+P->Deuteron+pi0:
      ELSEIF((ilb1*ilb2).EQ.2)THEN
         lbm=4
         xmm=ap1
      ELSE
c     For baryon resonances, use isospin-averaged cross sections:
         lbm=3+int(3 * RANART(NSEED))
         if(lbm.eq.4) then
            xmm=ap1
         else
            xmm=ap2
         endif
      ENDIF
c
      if(srt.ge.(xmd+xmm)) then
         pfinal=sqrt((s-(xmd+xmm)**2)*(s-(xmd-xmm)**2))/2./srt
         if((ilb1.eq.1.and.ilb2.eq.1).or.
     1        (ilb1.eq.2.and.ilb2.eq.2)) then
c     for pp or nn initial states:
            sbbdpi=fs*pfinal/pinitial/4.
         elseif((ilb1.eq.1.and.ilb2.eq.2).or.
     1           (ilb1.eq.2.and.ilb2.eq.1)) then
c     factor of 1/2 for pn or np initial states:
            sbbdpi=fs*pfinal/pinitial/4./2.
         else
c     for other BB initial states (spin- and isospin averaged):
            if(idxsec.eq.1) then
c     1: assume the same |matrix element|**2 (after averaging over initial 
c     spins and isospins) for B+B -> deuteron+meson at the same sqrt(s);
               sbbdpi=fs*pfinal/pinitial*3./16.
            elseif(idxsec.eq.2.or.idxsec.eq.4) then
               threshold=amax1(xmd+xmm,em1+em2)
               snew=(srt-threshold+srt0)**2
               if(idxsec.eq.2) then
c     2: assume the same |matrix element|**2 for B+B -> deuteron+meson 
c     at the same sqrt(s)-threshold:
                  sbbdpi=fnndpi(snew)*pfinal/pinitial*3./16.
               elseif(idxsec.eq.4) then
c     4: assume the same |matrix element|**2 for B+B <- deuteron+meson 
c     at the same sqrt(s)-threshold:
                  sbbdpi=fnndpi(snew)*pfinal/pinitial/6.*pifactor
               endif
            elseif(idxsec.eq.3) then
c     3: assume the same |matrix element|**2 for B+B <- deuteron+meson 
c     at the same sqrt(s):
               sbbdpi=fs*pfinal/pinitial/6.*pifactor
            endif
c
         endif
      endif
c     
*     d rho: DETERMINE THE CROSS SECTION TO THIS FINAL STATE:
      if(srt.gt.(xmd+xmrho)) then
         pfinal=sqrt((s-(xmd+xmrho)**2)*(s-(xmd-xmrho)**2))/2./srt
         if(idxsec.eq.1) then
            sbbdrho=fs*pfinal/pinitial*3./16.
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmd+xmrho,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sbbdrho=fnndpi(snew)*pfinal/pinitial*3./16.
            elseif(idxsec.eq.4) then
c     The spin- and isospin-averaged factor is 3-times larger for rho:
               sbbdrho=fnndpi(snew)*pfinal/pinitial/6.*(pifactor*3.)
            endif
         elseif(idxsec.eq.3) then
            sbbdrho=fs*pfinal/pinitial/6.*(pifactor*3.)
         endif
      endif
c
*     d omega: DETERMINE THE CROSS SECTION TO THIS FINAL STATE:
      if(srt.gt.(xmd+xmomega)) then
         pfinal=sqrt((s-(xmd+xmomega)**2)*(s-(xmd-xmomega)**2))/2./srt
         if(idxsec.eq.1) then
            sbbdomega=fs*pfinal/pinitial*3./16.
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmd+xmomega,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sbbdomega=fnndpi(snew)*pfinal/pinitial*3./16.
            elseif(idxsec.eq.4) then
               sbbdomega=fnndpi(snew)*pfinal/pinitial/6.*pifactor
            endif
         elseif(idxsec.eq.3) then
            sbbdomega=fs*pfinal/pinitial/6.*pifactor
         endif
      endif
c
*     d eta: DETERMINE THE CROSS SECTION TO THIS FINAL STATE:
      if(srt.gt.(xmd+xmeta)) then
         pfinal=sqrt((s-(xmd+xmeta)**2)*(s-(xmd-xmeta)**2))/2./srt
         if(idxsec.eq.1) then
            sbbdeta=fs*pfinal/pinitial*3./16.
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmd+xmeta,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sbbdeta=fnndpi(snew)*pfinal/pinitial*3./16.
            elseif(idxsec.eq.4) then
               sbbdeta=fnndpi(snew)*pfinal/pinitial/6.*(pifactor/3.)
            endif
         elseif(idxsec.eq.3) then
            sbbdeta=fs*pfinal/pinitial/6.*(pifactor/3.)
         endif
      endif
c
      sdprod=sbbdpi+sbbdrho+sbbdomega+sbbdeta
ctest off
c      write(99,111) srt,sbbdpi,sbbdrho,sbbdomega,sbbdeta,sdprod
c 111  format(6(f8.2,1x))
c
      if(sdprod.le.0) return
c
c     choose final state and assign masses here:
      x1=RANART(NSEED)
      if(x1.le.sbbdpi/sdprod) then
c     use the above-determined lbm and xmm.
      elseif(x1.le.(sbbdpi+sbbdrho)/sdprod) then
         lbm=25+int(3*RANART(NSEED))
         xmm=xmrho
      elseif(x1.le.(sbbdpi+sbbdrho+sbbdomega)/sdprod) then
         lbm=28
         xmm=xmomega
      else
         lbm=0
         xmm=xmeta
      endif
c
      return
      end
c
c     Generate angular distribution of Deuteron in the CMS frame:
      subroutine bbdangle(pxd,pyd,pzd,nt,ipert1,ianti,idloop,pfinal,
     1 dprob1,lbm)
      PARAMETER (PI=3.1415926)
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      COMMON/RNDF77/NSEED
      common /para8/ idpert,npertd,idxsec
      COMMON /AREVT/ IAEVT, IARUN, MISS
      SAVE   
c     take isotropic distribution for now:
      C1=1.0-2.0*RANART(NSEED)
      T1=2.0*PI*RANART(NSEED)
      S1=SQRT(1.0-C1**2)
      CT1=COS(T1)
      ST1=SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      PZd=pfinal*C1
      PXd=pfinal*S1*CT1 
      PYd=pfinal*S1*ST1
clin-5/2008 track the number of produced deuterons:
      if(idpert.eq.1.and.npertd.ge.1) then
         dprob=dprob1
      elseif(idpert.eq.2.and.npertd.ge.1) then
         dprob=1./float(npertd)
      endif
      if(ianti.eq.0) then
         if(idpert.eq.0.or.(idpert.eq.1.and.ipert1.eq.0).or.
     1        (idpert.eq.2.and.idloop.eq.(npertd+1))) then
            write (91,*) lb1,' *',lb2,' ->d+',lbm,' (regular d prodn) 
     1 @evt#',iaevt,' @nt=',nt
         elseif((idpert.eq.1.or.idpert.eq.2).and.idloop.eq.npertd) then
            write (91,*) lb1,' *',lb2,' ->d+',lbm,' (pert d prodn) 
     1 @evt#',iaevt,' @nt=',nt,' @prob=',dprob
         endif
      else
         if(idpert.eq.0.or.(idpert.eq.1.and.ipert1.eq.0).or.
     1        (idpert.eq.2.and.idloop.eq.(npertd+1))) then
            write (91,*) lb1,' *',lb2,' ->d+',lbm,' (regular dbar prodn) 
     1 @evt#',iaevt,' @nt=',nt
         elseif((idpert.eq.1.or.idpert.eq.2).and.idloop.eq.npertd) then
            write (91,*) lb1,' *',lb2,' ->d+',lbm,' (pert dbar prodn) 
     1 @evt#',iaevt,' @nt=',nt,' @prob=',dprob
         endif
      endif
c
      return
      end
c
c     Deuteron+Meson->B+B cross section (in mb)
      subroutine sdmbb(SRT,sdm,ianti)
      PARAMETER (AMN=0.939457,AMP=0.93828,
     1     AM0=1.232,AM1440=1.44,AM1535=1.535,srt0=2.012)
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      common /dpifsl/lbnn1,lbnn2,lbnd1,lbnd2,lbns1,lbns2,lbnp1,lbnp2,
     1     lbdd1,lbdd2,lbds1,lbds2,lbdp1,lbdp2,lbss1,lbss2,
     2     lbsp1,lbsp2,lbpp1,lbpp2
      common /dpifsm/xmnn1,xmnn2,xmnd1,xmnd2,xmns1,xmns2,xmnp1,xmnp2,
     1     xmdd1,xmdd2,xmds1,xmds2,xmdp1,xmdp2,xmss1,xmss2,
     2     xmsp1,xmsp2,xmpp1,xmpp2
      common /dpisig/sdmel,sdmnn,sdmnd,sdmns,sdmnp,sdmdd,sdmds,sdmdp,
     1     sdmss,sdmsp,sdmpp
      common /para8/ idpert,npertd,idxsec
      COMMON/RNDF77/NSEED
      SAVE   
c
      sdm=0.
      sdmel=0.
      sdmnn=0.
      sdmnd=0.
      sdmns=0.
      sdmnp=0.
      sdmdd=0.
      sdmds=0.
      sdmdp=0.
      sdmss=0.
      sdmsp=0.
      sdmpp=0.
ctest off check Xsec using fixed mass for resonances:
c      if(lb1.ge.25.and.lb1.le.27) then
c         em1=0.776
c      elseif(lb1.eq.28) then
c         em1=0.783
c      elseif(lb1.eq.0) then
c         em1=0.548
c      endif
c      if(lb2.ge.25.and.lb2.le.27) then
c         em2=0.776
c      elseif(lb2.eq.28) then
c         em2=0.783
c      elseif(lb2.eq.0) then
c         em2=0.548
c      endif
c
      if(srt.le.(em1+em2)) return
      s=srt**2
      pinitial=sqrt((s-(em1+em2)**2)*(s-(em1-em2)**2))/2./srt
      fs=fnndpi(s)
c     Determine isospin and spin factors for the ratio between 
c     Deuteron+Meson->BB and BB->Deuteron+Meson cross sections:
      if(idxsec.eq.1.or.idxsec.eq.2) then
c     Assume B+B -> d+Meson has the same cross sections as N+N -> d+pi, 
c     then determine d+Meson -> B+B cross sections:
         if((lb1.ge.3.and.lb1.le.5).or.
     1        (lb2.ge.3.and.lb2.le.5)) then
            xnnfactor=8./9.
         elseif((lb1.ge.25.and.lb1.le.27).or.
     1           (lb2.ge.25.and.lb2.le.27)) then
            xnnfactor=8./27.
         elseif(lb1.eq.28.or.lb2.eq.28) then
            xnnfactor=8./9.
         elseif(lb1.eq.0.or.lb2.eq.0) then
            xnnfactor=8./3.
         endif
      else
c     Assume d+Meson -> B+B has the same cross sections as d+pi -> N+N:
      endif
clin-9/2008 For elastic collisions:
      if(idxsec.eq.1.or.idxsec.eq.3) then
c     1/3: assume the same |matrix element|**2 (after averaging over initial 
c     spins and isospins) for d+Meson elastic at the same sqrt(s);
         sdmel=fdpiel(s)
      elseif(idxsec.eq.2.or.idxsec.eq.4) then
c     2/4: assume the same |matrix element|**2 (after averaging over initial 
c     spins and isospins) for d+Meson elastic at the same sqrt(s)-threshold:
         threshold=em1+em2
         snew=(srt-threshold+srt0)**2
         sdmel=fdpiel(snew)
      endif
c
*     NN: DETERMINE THE CHARGE STATES OF PARTICLESIN THE FINAL STATE
      IF(((lb1.eq.5.or.lb2.eq.5.or.lb1.eq.27.or.lb2.eq.27)
     1     .and.ianti.eq.0).or.
     2     ((lb1.eq.3.or.lb2.eq.3.or.lb1.eq.25.or.lb2.eq.25)
     3     .and.ianti.eq.1))THEN
*     (1) FOR Deuteron+(pi+,rho+) -> P+P or DeuteronBar+(pi-,rho-)-> PBar+PBar:
         lbnn1=1
         lbnn2=1
         xmnn1=amp
         xmnn2=amp
      ELSEIF(lb1.eq.3.or.lb2.eq.3.or.lb1.eq.26.or.lb2.eq.26
     1        .or.lb1.eq.28.or.lb2.eq.28.or.lb1.eq.0.or.lb2.eq.0)THEN
*     (2) FOR Deuteron+(pi0,rho0,omega,eta) -> N+P 
*     or DeuteronBar+(pi0,rho0,omega,eta) ->NBar+PBar:
         lbnn1=2
         lbnn2=1
         xmnn1=amn
         xmnn2=amp
      ELSE
*     (3) FOR Deuteron+(pi-,rho-) -> N+N or DeuteronBar+(pi+,rho+)-> NBar+NBar:
         lbnn1=2
         lbnn2=2
         xmnn1=amn
         xmnn2=amn
      ENDIF
      if(srt.gt.(xmnn1+xmnn2)) then
         pfinal=sqrt((s-(xmnn1+xmnn2)**2)*(s-(xmnn1-xmnn2)**2))/2./srt
         if(idxsec.eq.1) then
c     1: assume the same |matrix element|**2 (after averaging over initial 
c     spins and isospins) for B+B -> deuteron+meson at the same sqrt(s);
            sdmnn=fs*pfinal/pinitial*3./16.*xnnfactor
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmnn1+xmnn2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
c     2: assume the same |matrix element|**2 for B+B -> deuteron+meson 
c     at the same sqrt(s)-threshold:
               sdmnn=fnndpi(snew)*pfinal/pinitial*3./16.*xnnfactor
            elseif(idxsec.eq.4) then
c     4: assume the same |matrix element|**2 for B+B <- deuteron+meson 
c     at the same sqrt(s)-threshold:
               sdmnn=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
c     3: assume the same |matrix element|**2 for B+B <- deuteron+meson 
c     at the same sqrt(s):
            sdmnn=fs*pfinal/pinitial/6.
         endif
      endif
c     
*     ND: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbnd1=1+int(2*RANART(NSEED))
      lbnd2=6+int(4*RANART(NSEED))
      if(lbnd1.eq.1) then
         xmnd1=amp
      elseif(lbnd1.eq.2) then
         xmnd1=amn
      endif
      xmnd2=am0
      if(srt.gt.(xmnd1+xmnd2)) then
         pfinal=sqrt((s-(xmnd1+xmnd2)**2)*(s-(xmnd1-xmnd2)**2))/2./srt
         if(idxsec.eq.1) then
c     The spin- and isospin-averaged factor is 8-times larger for ND:
            sdmnd=fs*pfinal/pinitial*3./16.*(xnnfactor*8.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmnd1+xmnd2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmnd=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*8.)
            elseif(idxsec.eq.4) then
               sdmnd=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmnd=fs*pfinal/pinitial/6.
         endif
      endif
c
*     NS: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbns1=1+int(2*RANART(NSEED))
      lbns2=10+int(2*RANART(NSEED))
      if(lbns1.eq.1) then
         xmns1=amp
      elseif(lbns1.eq.2) then
         xmns1=amn
      endif
      xmns2=am1440
      if(srt.gt.(xmns1+xmns2)) then
         pfinal=sqrt((s-(xmns1+xmns2)**2)*(s-(xmns1-xmns2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmns=fs*pfinal/pinitial*3./16.*(xnnfactor*2.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmns1+xmns2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmns=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*2.)
            elseif(idxsec.eq.4) then
               sdmns=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmns=fs*pfinal/pinitial/6.
         endif
      endif
c
*     NP: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbnp1=1+int(2*RANART(NSEED))
      lbnp2=12+int(2*RANART(NSEED))
      if(lbnp1.eq.1) then
         xmnp1=amp
      elseif(lbnp1.eq.2) then
         xmnp1=amn
      endif
      xmnp2=am1535
      if(srt.gt.(xmnp1+xmnp2)) then
         pfinal=sqrt((s-(xmnp1+xmnp2)**2)*(s-(xmnp1-xmnp2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmnp=fs*pfinal/pinitial*3./16.*(xnnfactor*2.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmnp1+xmnp2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmnp=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*2.)
            elseif(idxsec.eq.4) then
               sdmnp=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmnp=fs*pfinal/pinitial/6.
         endif
      endif
c
*     DD: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbdd1=6+int(4*RANART(NSEED))
      lbdd2=6+int(4*RANART(NSEED))
      xmdd1=am0
      xmdd2=am0
      if(srt.gt.(xmdd1+xmdd2)) then
         pfinal=sqrt((s-(xmdd1+xmdd2)**2)*(s-(xmdd1-xmdd2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmdd=fs*pfinal/pinitial*3./16.*(xnnfactor*16.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmdd1+xmdd2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmdd=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*16.)
            elseif(idxsec.eq.4) then
               sdmdd=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmdd=fs*pfinal/pinitial/6.
         endif
      endif
c
*     DS: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbds1=6+int(4*RANART(NSEED))
      lbds2=10+int(2*RANART(NSEED))
      xmds1=am0
      xmds2=am1440
      if(srt.gt.(xmds1+xmds2)) then
         pfinal=sqrt((s-(xmds1+xmds2)**2)*(s-(xmds1-xmds2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmds=fs*pfinal/pinitial*3./16.*(xnnfactor*8.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmds1+xmds2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmds=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*8.)
            elseif(idxsec.eq.4) then
               sdmds=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmds=fs*pfinal/pinitial/6.
         endif
      endif
c
*     DP: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbdp1=6+int(4*RANART(NSEED))
      lbdp2=12+int(2*RANART(NSEED))
      xmdp1=am0
      xmdp2=am1535
      if(srt.gt.(xmdp1+xmdp2)) then
         pfinal=sqrt((s-(xmdp1+xmdp2)**2)*(s-(xmdp1-xmdp2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmdp=fs*pfinal/pinitial*3./16.*(xnnfactor*8.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmdp1+xmdp2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmdp=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*8.)
            elseif(idxsec.eq.4) then
               sdmdp=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmdp=fs*pfinal/pinitial/6.
         endif
      endif
c
*     SS: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbss1=10+int(2*RANART(NSEED))
      lbss2=10+int(2*RANART(NSEED))
      xmss1=am1440
      xmss2=am1440
      if(srt.gt.(xmss1+xmss2)) then
         pfinal=sqrt((s-(xmss1+xmss2)**2)*(s-(xmss1-xmss2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmss=fs*pfinal/pinitial*3./16.*xnnfactor
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmss1+xmss2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmss=fnndpi(snew)*pfinal/pinitial*3./16.*xnnfactor
            elseif(idxsec.eq.4) then
               sdmss=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmns=fs*pfinal/pinitial/6.
         endif
      endif
c
*     SP: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbsp1=10+int(2*RANART(NSEED))
      lbsp2=12+int(2*RANART(NSEED))
      xmsp1=am1440
      xmsp2=am1535
      if(srt.gt.(xmsp1+xmsp2)) then
         pfinal=sqrt((s-(xmsp1+xmsp2)**2)*(s-(xmsp1-xmsp2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmsp=fs*pfinal/pinitial*3./16.*(xnnfactor*2.)
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmsp1+xmsp2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmsp=fnndpi(snew)*pfinal/pinitial*3./16.*(xnnfactor*2.)
            elseif(idxsec.eq.4) then
               sdmsp=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmsp=fs*pfinal/pinitial/6.
         endif
      endif
c
*     PP: DETERMINE THE CHARGE STATES OF PARTICLES IN THE FINAL STATE
      lbpp1=12+int(2*RANART(NSEED))
      lbpp2=12+int(2*RANART(NSEED))
      xmpp1=am1535
      xmpp2=am1535
      if(srt.gt.(xmpp1+xmpp2)) then
         pfinal=sqrt((s-(xmpp1+xmpp2)**2)*(s-(xmpp1-xmpp2)**2))/2./srt
         if(idxsec.eq.1) then
            sdmpp=fs*pfinal/pinitial*3./16.*xnnfactor
         elseif(idxsec.eq.2.or.idxsec.eq.4) then
            threshold=amax1(xmpp1+xmpp2,em1+em2)
            snew=(srt-threshold+srt0)**2
            if(idxsec.eq.2) then
               sdmpp=fnndpi(snew)*pfinal/pinitial*3./16.*xnnfactor
            elseif(idxsec.eq.4) then
               sdmpp=fnndpi(snew)*pfinal/pinitial/6.
            endif
         elseif(idxsec.eq.3) then
            sdmpp=fs*pfinal/pinitial/6.
         endif
      endif
c
      sdm=sdmel+sdmnn+sdmnd+sdmns+sdmnp+sdmdd+sdmds+sdmdp
     1     +sdmss+sdmsp+sdmpp
      if(ianti.eq.1) then
         lbnn1=-lbnn1
         lbnn2=-lbnn2
         lbnd1=-lbnd1
         lbnd2=-lbnd2
         lbns1=-lbns1
         lbns2=-lbns2
         lbnp1=-lbnp1
         lbnp2=-lbnp2
         lbdd1=-lbdd1
         lbdd2=-lbdd2
         lbds1=-lbds1
         lbds2=-lbds2
         lbdp1=-lbdp1
         lbdp2=-lbdp2
         lbss1=-lbss1
         lbss2=-lbss2
         lbsp1=-lbsp1
         lbsp2=-lbsp2
         lbpp1=-lbpp1
         lbpp2=-lbpp2
      endif
ctest off
c      write(98,100) srt,sdmnn,sdmnd,sdmns,sdmnp,sdmdd,sdmds,sdmdp,
c     1     sdmss,sdmsp,sdmpp,sdm
c 100  format(f5.2,11(1x,f5.1))
c
      return
      end
c
clin-9/2008 Deuteron+Meson ->B+B and elastic collisions
      SUBROUTINE crdmbb(PX,PY,PZ,SRT,I1,I2,IBLOCK,
     1     NTAG,sig,NT,ianti)
      PARAMETER (MAXSTR=150001,MAXR=1)
      COMMON /AA/R(3,MAXSTR)
      COMMON /BB/ P(3,MAXSTR)
      COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
      COMMON /CC/ E(MAXSTR)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
      COMMON /AREVT/ IAEVT, IARUN, MISS
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      common /para8/ idpert,npertd,idxsec
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      common /dpifsl/lbnn1,lbnn2,lbnd1,lbnd2,lbns1,lbns2,lbnp1,lbnp2,
     1     lbdd1,lbdd2,lbds1,lbds2,lbdp1,lbdp2,lbss1,lbss2,
     2     lbsp1,lbsp2,lbpp1,lbpp2
      common /dpifsm/xmnn1,xmnn2,xmnd1,xmnd2,xmns1,xmns2,xmnp1,xmnp2,
     1     xmdd1,xmdd2,xmds1,xmds2,xmdp1,xmdp2,xmss1,xmss2,
     2     xmsp1,xmsp2,xmpp1,xmpp2
      common /dpisig/sdmel,sdmnn,sdmnd,sdmns,sdmnp,sdmdd,sdmds,sdmdp,
     1     sdmss,sdmsp,sdmpp
      COMMON/RNDF77/NSEED
      SAVE   
*-----------------------------------------------------------------------
      IBLOCK=0
      NTAG=0
      EM1=E(I1)
      EM2=E(I2)
      s=srt**2
      if(sig.le.0) return
c
      if(iabs(lb1).eq.42) then
         ideut=i1
         lbm=lb2
         idm=i2
      else
         ideut=i2
         lbm=lb1
         idm=i1
      endif
cccc  Elastic collision or destruction of perturbatively-produced deuterons:
      if((idpert.eq.1.or.idpert.eq.2).and.dpertp(ideut).ne.1.) then
c     choose reaction channels:
         x1=RANART(NSEED)
         if(x1.le.sdmel/sig)then
c     Elastic collisions:
            if(ianti.eq.0) then
               write(91,*) '  d+',lbm,' (pert d M elastic) @nt=',nt
     1              ,' @prob=',dpertp(ideut)
            else
               write(91,*) '  d+',lbm,' (pert dbar M elastic) @nt=',nt
     1              ,' @prob=',dpertp(ideut)
            endif
            pfinal=sqrt((s-(em1+em2)**2)*(s-(em1-em2)**2))/2./srt
            CALL dmelangle(pxn,pyn,pzn,pfinal)
            CALL ROTATE(PX,PY,PZ,Pxn,Pyn,Pzn)
            EdCM=SQRT(E(ideut)**2+Pxn**2+Pyn**2+Pzn**2)
            PdBETA=Pxn*BETAX+Pyn*BETAY+Pzn*BETAZ
            TRANSF=GAMMA*(GAMMA*PdBETA/(GAMMA+1.)+EdCM)
            Pt1d=BETAX*TRANSF+Pxn
            Pt2d=BETAY*TRANSF+Pyn
            Pt3d=BETAZ*TRANSF+Pzn
            p(1,ideut)=pt1d
            p(2,ideut)=pt2d
            p(3,ideut)=pt3d
            IBLOCK=504
            PX1=P(1,I1)
            PY1=P(2,I1)
            PZ1=P(3,I1)
            ID(I1)=2
            ID(I2)=2
c     Change the position of the perturbative deuteron to that of 
c     the meson to avoid consecutive collisions between them:
            R(1,ideut)=R(1,idm)
            R(2,ideut)=R(2,idm)
            R(3,ideut)=R(3,idm)
         else
c     Destruction of deuterons:
            if(ianti.eq.0) then
               write(91,*) '  d+',lbm,' ->BB (pert d destrn) @nt=',nt
     1              ,' @prob=',dpertp(ideut)
            else
               write(91,*) '  d+',lbm,' ->BB (pert dbar destrn) @nt=',nt
     1              ,' @prob=',dpertp(ideut)
            endif
            e(ideut)=0.
            IBLOCK=502
         endif
         return
      endif
c
cccc  Destruction of regularly-produced deuterons:
      IBLOCK=502
c     choose final state and assign masses here:
      x1=RANART(NSEED)
      if(x1.le.sdmnn/sig)then
         lbb1=lbnn1
         lbb2=lbnn2
         xmb1=xmnn1
         xmb2=xmnn2
      elseif(x1.le.(sdmnn+sdmnd)/sig)then
         lbb1=lbnd1
         lbb2=lbnd2
         xmb1=xmnd1
         xmb2=xmnd2
      elseif(x1.le.(sdmnn+sdmnd+sdmns)/sig)then
         lbb1=lbns1
         lbb2=lbns2
         xmb1=xmns1
         xmb2=xmns2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp)/sig)then
         lbb1=lbnp1
         lbb2=lbnp2
         xmb1=xmnp1
         xmb2=xmnp2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp+sdmdd)/sig)then
         lbb1=lbdd1
         lbb2=lbdd2
         xmb1=xmdd1
         xmb2=xmdd2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp+sdmdd+sdmds)/sig)then
         lbb1=lbds1
         lbb2=lbds2
         xmb1=xmds1
         xmb2=xmds2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp+sdmdd+sdmds+sdmdp)/sig)then
         lbb1=lbdp1
         lbb2=lbdp2
         xmb1=xmdp1
         xmb2=xmdp2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp+sdmdd+sdmds+sdmdp
     1        +sdmss)/sig)then
         lbb1=lbss1
         lbb2=lbss2
         xmb1=xmss1
         xmb2=xmss2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp+sdmdd+sdmds+sdmdp
     1        +sdmss+sdmsp)/sig)then
         lbb1=lbsp1
         lbb2=lbsp2
         xmb1=xmsp1
         xmb2=xmsp2
      elseif(x1.le.(sdmnn+sdmnd+sdmns+sdmnp+sdmdd+sdmds+sdmdp
     1        +sdmss+sdmsp+sdmpp)/sig)then
         lbb1=lbpp1
         lbb2=lbpp2
         xmb1=xmpp1
         xmb2=xmpp2
      else
c     Elastic collision:
         lbb1=lb1
         lbb2=lb2
         xmb1=em1
         xmb2=em2
         IBLOCK=504
      endif
      LB(I1)=lbb1
      E(i1)=xmb1
      LB(I2)=lbb2
      E(I2)=xmb2
      lb1=lb(i1)
      lb2=lb(i2)
      pfinal=sqrt((s-(xmb1+xmb2)**2)*(s-(xmb1-xmb2)**2))/2./srt
c
      if(iblock.eq.502) then
         CALL dmangle(pxn,pyn,pzn,nt,ianti,pfinal,lbm)
      elseif(iblock.eq.504) then
         if(ianti.eq.0) then
            write (91,*) ' d+',lbm,' (regular d M elastic) @evt#',
     1           iaevt,' @nt=',nt,' lb1,2=',lb1,lb2
         else
            write (91,*) ' d+',lbm,' (regular dbar M elastic) @evt#',
     1           iaevt,' @nt=',nt,' lb1,2=',lb1,lb2
         endif
         CALL dmelangle(pxn,pyn,pzn,pfinal)
      else
         print *, 'Wrong iblock number in crdmbb()'
         stop
      endif
*     ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
c     (This is not needed for isotropic distributions)
      CALL ROTATE(PX,PY,PZ,Pxn,Pyn,Pzn)
*     LORENTZ-TRANSFORMATION OF THE MOMENTUM OF PARTICLES IN THE FINAL STATE 
*     FROM THE NUCLEUS-NUCLEUS CMS. FRAME INTO LAB FRAME:
*     For the 1st baryon:
      E1CM=SQRT(E(I1)**2+Pxn**2+Pyn**2+Pzn**2)
      P1BETA=Pxn*BETAX+Pyn*BETAY+Pzn*BETAZ
      TRANSF=GAMMA*(GAMMA*P1BETA/(GAMMA+1.)+E1CM)
      Pt1i1=BETAX*TRANSF+Pxn
      Pt2i1=BETAY*TRANSF+Pyn
      Pt3i1=BETAZ*TRANSF+Pzn
c
      p(1,i1)=pt1i1
      p(2,i1)=pt2i1
      p(3,i1)=pt3i1
*     For the 2nd baryon:
      E2CM=SQRT(E(I2)**2+Pxn**2+Pyn**2+Pzn**2)
      P2BETA=-Pxn*BETAX-Pyn*BETAY-Pzn*BETAZ
      TRANSF=GAMMA*(GAMMA*P2BETA/(GAMMA+1.)+E2CM)
      Pt1I2=BETAX*TRANSF-Pxn
      Pt2I2=BETAY*TRANSF-Pyn
      Pt3I2=BETAZ*TRANSF-Pzn
c     
      p(1,i2)=pt1i2
      p(2,i2)=pt2i2
      p(3,i2)=pt3i2
c
      PX1=P(1,I1)
      PY1=P(2,I1)
      PZ1=P(3,I1)
      EM1=E(I1)
      EM2=E(I2)
      ID(I1)=2
      ID(I2)=2
      RETURN
      END
c
c     Generate angular distribution of BB from d+meson in the CMS frame:
      subroutine dmangle(pxn,pyn,pzn,nt,ianti,pfinal,lbm)
      PARAMETER (PI=3.1415926)
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      COMMON /AREVT/ IAEVT, IARUN, MISS
      COMMON/RNDF77/NSEED
      SAVE   
c     take isotropic distribution for now:
      C1=1.0-2.0*RANART(NSEED)
      T1=2.0*PI*RANART(NSEED)
      S1=SQRT(1.0-C1**2)
      CT1=COS(T1)
      ST1=SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      Pzn=pfinal*C1
      Pxn=pfinal*S1*CT1 
      Pyn=pfinal*S1*ST1
clin-5/2008 track the number of regularly-destructed deuterons:
      if(ianti.eq.0) then
         write (91,*) ' d+',lbm,' ->BB (regular d destrn) @evt#',
     1        iaevt,' @nt=',nt,' lb1,2=',lb1,lb2
      else
         write (91,*) ' d+',lbm,' ->BB (regular dbar destrn) @evt#',
     1        iaevt,' @nt=',nt,' lb1,2=',lb1,lb2
      endif
c
      return
      end
c
c     Angular distribution of d+meson elastic collisions in the CMS frame:
      subroutine dmelangle(pxn,pyn,pzn,pfinal)
      PARAMETER (PI=3.1415926)
      COMMON/RNDF77/NSEED
      SAVE   
c     take isotropic distribution for now:
      C1=1.0-2.0*RANART(NSEED)
      T1=2.0*PI*RANART(NSEED)
      S1=SQRT(1.0-C1**2)
      CT1=COS(T1)
      ST1=SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      Pzn=pfinal*C1
      Pxn=pfinal*S1*CT1 
      Pyn=pfinal*S1*ST1
      return
      end
c
clin-9/2008 Deuteron+Baryon elastic cross section (in mb)
      subroutine sdbelastic(SRT,sdb)
      PARAMETER (srt0=2.012)
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      common /para8/ idpert,npertd,idxsec
      SAVE   
c
      sdb=0.
      sdbel=0.
      if(srt.le.(em1+em2)) return
      s=srt**2
c     For elastic collisions:
      if(idxsec.eq.1.or.idxsec.eq.3) then
c     1/3: assume the same |matrix element|**2 (after averaging over initial 
c     spins and isospins) for d+Baryon elastic at the same sqrt(s);
         sdbel=fdbel(s)
      elseif(idxsec.eq.2.or.idxsec.eq.4) then
c     2/4: assume the same |matrix element|**2 (after averaging over initial 
c     spins and isospins) for d+Baryon elastic at the same sqrt(s)-threshold:
         threshold=em1+em2
         snew=(srt-threshold+srt0)**2
         sdbel=fdbel(snew)
      endif
      sdb=sdbel
      return
      end
clin-9/2008 Deuteron+Baryon elastic collisions
      SUBROUTINE crdbel(PX,PY,PZ,SRT,I1,I2,IBLOCK,
     1     NTAG,sig,NT,ianti)
      PARAMETER (MAXSTR=150001,MAXR=1)
      COMMON /AA/R(3,MAXSTR)
      COMMON /BB/ P(3,MAXSTR)
      COMMON /BG/BETAX,BETAY,BETAZ,GAMMA
      COMMON /CC/ E(MAXSTR)
      COMMON /EE/ ID(MAXSTR),LB(MAXSTR)
      COMMON /AREVT/ IAEVT, IARUN, MISS
      common/leadng/lb1,px1,py1,pz1,em1,e1,xfnl,yfnl,zfnl,tfnl,
     1     px1n,py1n,pz1n,dp1n
      common /dpi/em2,lb2
      common /para8/ idpert,npertd,idxsec
      COMMON /dpert/dpertt(MAXSTR,MAXR),dpertp(MAXSTR),dplast(MAXSTR),
     1     dpdcy(MAXSTR),dpdpi(MAXSTR,MAXR),dpt(MAXSTR, MAXR),
     2     dpp1(MAXSTR,MAXR),dppion(MAXSTR,MAXR)
      SAVE   
*-----------------------------------------------------------------------
      IBLOCK=0
      NTAG=0
      EM1=E(I1)
      EM2=E(I2)
      s=srt**2
      if(sig.le.0) return
      IBLOCK=503
c
      if(iabs(lb1).eq.42) then
         ideut=i1
         lbb=lb2
         idb=i2
      else
         ideut=i2
         lbb=lb1
         idb=i1
      endif
cccc  Elastic collision of perturbatively-produced deuterons:
      if((idpert.eq.1.or.idpert.eq.2).and.dpertp(ideut).ne.1.) then
         if(ianti.eq.0) then
            write(91,*) '  d+',lbb,' (pert d B elastic) @nt=',nt
     1           ,' @prob=',dpertp(ideut),p(1,idb),p(2,idb)
     2           ,p(1,ideut),p(2,ideut)
         else
            write(91,*) '  d+',lbb,' (pert dbar Bbar elastic) @nt=',nt
     1           ,' @prob=',dpertp(ideut),p(1,idb),p(2,idb)
     2           ,p(1,ideut),p(2,ideut)
         endif
         pfinal=sqrt((s-(em1+em2)**2)*(s-(em1-em2)**2))/2./srt
         CALL dbelangle(pxn,pyn,pzn,pfinal)
         CALL ROTATE(PX,PY,PZ,Pxn,Pyn,Pzn)
         EdCM=SQRT(E(ideut)**2+Pxn**2+Pyn**2+Pzn**2)
         PdBETA=Pxn*BETAX+Pyn*BETAY+Pzn*BETAZ
         TRANSF=GAMMA*(GAMMA*PdBETA/(GAMMA+1.)+EdCM)
         Pt1d=BETAX*TRANSF+Pxn
         Pt2d=BETAY*TRANSF+Pyn
         Pt3d=BETAZ*TRANSF+Pzn
         p(1,ideut)=pt1d
         p(2,ideut)=pt2d
         p(3,ideut)=pt3d
         PX1=P(1,I1)
         PY1=P(2,I1)
         PZ1=P(3,I1)
         ID(I1)=2
         ID(I2)=2
c     Change the position of the perturbative deuteron to that of 
c     the baryon to avoid consecutive collisions between them:
         R(1,ideut)=R(1,idb)
         R(2,ideut)=R(2,idb)
         R(3,ideut)=R(3,idb)
         return
      endif
c
c     Elastic collision of regularly-produced deuterons:
      if(ianti.eq.0) then
         write (91,*) ' d+',lbb,' (regular d B elastic) @evt#',
     1        iaevt,' @nt=',nt,' lb1,2=',lb1,lb2
      else
         write (91,*) ' d+',lbb,' (regular dbar Bbar elastic) @evt#',
     1        iaevt,' @nt=',nt,' lb1,2=',lb1,lb2
      endif
      pfinal=sqrt((s-(em1+em2)**2)*(s-(em1-em2)**2))/2./srt
      CALL dbelangle(pxn,pyn,pzn,pfinal)
*     ROTATE THE MOMENTA OF PARTICLES IN THE CMS OF P1+P2
c     (This is not needed for isotropic distributions)
      CALL ROTATE(PX,PY,PZ,Pxn,Pyn,Pzn)
*     LORENTZ-TRANSFORMATION OF THE MOMENTUM OF PARTICLES IN THE FINAL STATE 
*     FROM THE NUCLEUS-NUCLEUS CMS. FRAME INTO LAB FRAME:
*     For the 1st baryon:
      E1CM=SQRT(E(I1)**2+Pxn**2+Pyn**2+Pzn**2)
      P1BETA=Pxn*BETAX+Pyn*BETAY+Pzn*BETAZ
      TRANSF=GAMMA*(GAMMA*P1BETA/(GAMMA+1.)+E1CM)
      Pt1i1=BETAX*TRANSF+Pxn
      Pt2i1=BETAY*TRANSF+Pyn
      Pt3i1=BETAZ*TRANSF+Pzn
c
      p(1,i1)=pt1i1
      p(2,i1)=pt2i1
      p(3,i1)=pt3i1
*     For the 2nd baryon:
      E2CM=SQRT(E(I2)**2+Pxn**2+Pyn**2+Pzn**2)
      P2BETA=-Pxn*BETAX-Pyn*BETAY-Pzn*BETAZ
      TRANSF=GAMMA*(GAMMA*P2BETA/(GAMMA+1.)+E2CM)
      Pt1I2=BETAX*TRANSF-Pxn
      Pt2I2=BETAY*TRANSF-Pyn
      Pt3I2=BETAZ*TRANSF-Pzn
c     
      p(1,i2)=pt1i2
      p(2,i2)=pt2i2
      p(3,i2)=pt3i2
c
      PX1=P(1,I1)
      PY1=P(2,I1)
      PZ1=P(3,I1)
      EM1=E(I1)
      EM2=E(I2)
      ID(I1)=2
      ID(I2)=2
      RETURN
      END
c
c     Part of the cross section function of NN->Deuteron+Pi (in mb):
      function fnndpi(s)
      parameter(srt0=2.012)
      if(s.le.srt0**2) then
         fnndpi=0.
      else
         fnndpi=26.*exp(-(s-4.65)**2/0.1)+4.*exp(-(s-4.65)**2/2.)
     1        +0.28*exp(-(s-6.)**2/10.)
      endif
      return
      end
c
c     Angular distribution of d+baryon elastic collisions in the CMS frame:
      subroutine dbelangle(pxn,pyn,pzn,pfinal)
      PARAMETER (PI=3.1415926)
      COMMON/RNDF77/NSEED
      SAVE   
c     take isotropic distribution for now:
      C1=1.0-2.0*RANART(NSEED)
      T1=2.0*PI*RANART(NSEED)
      S1=SQRT(1.0-C1**2)
      CT1=COS(T1)
      ST1=SIN(T1)
* THE MOMENTUM IN THE CMS IN THE FINAL STATE
      Pzn=pfinal*C1
      Pxn=pfinal*S1*CT1 
      Pyn=pfinal*S1*ST1
      return
      end
c
c     Cross section of Deuteron+Pi elastic (in mb):
      function fdpiel(s)
      parameter(srt0=2.012)
      if(s.le.srt0**2) then
         fdpiel=0.
      else
         fdpiel=63.*exp(-(s-4.67)**2/0.15)+15.*exp(-(s-6.25)**2/0.3)
      endif
      return
      end
c
c     Cross section of Deuteron+N elastic (in mb):
      function fdbel(s)
      parameter(srt0=2.012)
      if(s.le.srt0**2) then
         fdbel=0.
      else
         fdbel=2500.*exp(-(s-7.93)**2/0.003)
     1        +300.*exp(-(s-7.93)**2/0.1)+10.
      endif
      return
      end
