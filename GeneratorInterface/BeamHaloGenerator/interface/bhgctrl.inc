*
* $Id: bhgctrl.inc
*
*****************************************************************
*                                                               *
*     DESCRIPTION: this include file contains the control       *
*                  parameters for beam halo                     *                                                               *
*                                                               *
*****************************************************************
*
      INTEGER    GENMOD        !          "rate only" mode (0)
                               !  or   "MC generator" mode (1)
                               !  or "read from file" mode (2)
      INTEGER    LHC_B1        !  LHC beam 1 is  (off/on = 0/1)
      INTEGER    LHC_B2        !  LHC beam 1 is  (off/on = 0/1)
      INTEGER    IW_MUO        !  I want muons   (no/yes = 0/1)
      INTEGER    IW_HAD        !  I want hadrons (no/yes = 0/1)
      REAL       EG_MIN        !  minimum energy [GeV]
      REAL       EG_MAX        !  maximum energy [GeV]

      INTEGER    idx_shift_bx    ! e.g. -2, -1 for previous bunch-crossing
      REAL       BXNS            ! time between 2 bx's, in ns
                                                                                                                         
                                                                                                                         
      COMMON /BHGCTRL/ GENMOD,LHC_B1,LHC_B2,IW_MUO,IW_HAD,idx_shift_bx,
     +                 EG_MIN,EG_MAX,BXNS

cc      COMMON /BHGCTRL/ GENMOD,LHC_B1,LHC_B2,IW_MUO,IW_HAD,EG_MIN,EG_MAX
