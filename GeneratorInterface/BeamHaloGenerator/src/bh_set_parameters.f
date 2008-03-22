
      subroutine bhsetparam(ival, fval, cval)

      integer ival(8)
      real fval(4)
      character *(*) cval

      INTEGER    GENMOD        !          "rate only" mode (0)
                               !  or   "MC generator" mode (1)
                               !  or "read from file" mode (2)
      INTEGER    LHC_B1        !  LHC beam 1 is  (off/on = 0/1)
      INTEGER    LHC_B2        !  LHC beam 1 is  (off/on = 0/1)
      INTEGER    IW_MUO        !  I want muons   (no/yes = 0/1)
      INTEGER    IW_HAD        !  I want hadrons (no/yes = 0/1)
      REAL       EG_MIN        !  minimum energy [GeV]
      REAL       EG_MAX        !  maximum energy [GeV]
      INTEGER    NEVENT        !
      INTEGER    OFFSET        !
      INTEGER    idx_shift_bx    ! e.g. -2, -1 for previous bunch-crossing
      REAL       BXNS            ! time between 2 bx's, in ns
      REAL       W0            !  external per second normalization
      
      CHARACTER*100 G3FNAME    ! Genmod = 3 file name

      COMMON /BHGCTRL/ GENMOD,LHC_B1,LHC_B2,IW_MUO,IW_HAD,
     +                 NEVENT,OFFSET,idx_shift_bx,
     +                 EG_MIN,EG_MAX,BXNS,
     +                 W0,G3FNAME
c
      GENMOD = ival(1)
      LHC_B1 = ival(2)
      LHC_B2 = ival(3)
      IW_MUO = ival(4)
      IW_HAD = ival(5)
      NEVENT = ival(6)
      OFFSET = ival(7)
      idx_shift_bx = ival(8)

      EG_MIN = fval(1)
      EG_MAX = fval(2)
      W0     = fval(4)
      BXNS = fval(3)

      G3FNAME = cval

c
      write(6,*) 'GENMOD is ', GENMOD
      write(6,*) 'LHC_B1 is ', LHC_B1
      write(6,*) 'LHC_B2 is ', LHC_B2
      write(6,*) 'IW_MUO is ',IW_MUO
      write(6,*) 'IW_HAD is ',IW_HAD
      write(6,*) 'NEVENT is ',NEVENT
      write(6,*) 'OFFSET is ',OFFSET
      write(6,*) 'EG_MIN is ',EG_MIN
      write(6,*) 'EG_MAX is ',EG_MAX
      write(6,*) 'idx_shift_bx is ',idx_shift_bx
      write(6,*) 'BXNS  is ',BXNS
      write(6,*) 'W0 is     ',W0
      write(6,*) 'G3FNAME is ',G3FNAME


      return
      end

