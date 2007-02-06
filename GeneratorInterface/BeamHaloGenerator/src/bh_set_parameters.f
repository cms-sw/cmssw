
      subroutine bhsetparam(ival, fval)

      integer ival(5)
      real fval(2)

      INTEGER    GENMOD        !          "rate only" mode (0)
                               !  or   "MC generator" mode (1)
                               !  or "read from file" mode (2)
      INTEGER    LHC_B1        !  LHC beam 1 is  (off/on = 0/1)
      INTEGER    LHC_B2        !  LHC beam 1 is  (off/on = 0/1)
      INTEGER    IW_MUO        !  I want muons   (no/yes = 0/1)
      INTEGER    IW_HAD        !  I want hadrons (no/yes = 0/1)
      REAL       EG_MIN        !  minimum energy [GeV]
      REAL       EG_MAX        !  maximum energy [GeV]

      COMMON /BHGCTRL/ GENMOD,LHC_B1,LHC_B2,IW_MUO,IW_HAD,EG_MIN,EG_MAX
c
      GENMOD = ival(1)
      LHC_B1 = ival(2)
      LHC_B2 = ival(3)
      IW_MUO = ival(4)
      IW_HAD = ival(5)

      EG_MIN = fval(1)
      EG_MAX = fval(2)
c
      write(6,*) 'GENMOD is ', GENMOD
      write(6,*) 'LHC_B1 is ', LHC_B1
      write(6,*) 'LHC_B2 is ', LHC_B2
      write(6,*) 'IW_MUO is ',IW_MUO
      write(6,*) 'IW_HAD is ',IW_HAD
      write(6,*) 'EG_MIN is ',EG_MIN
      write(6,*) 'EG_MAX is ',EG_MAX


      return
      end

