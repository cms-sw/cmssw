*****************************************************************
*                                                               *
*     DESCRIPTION: this include file contains the control       *
*                  parameters and most relevant constants       *
*                  and common block for event counting          *
*                                                               *
*     last update (18/12/2004) by droll                         *
*                                                               *
*****************************************************************
      ! particle masses
      REAL       M_MUON
      PARAMETER (M_MUON = 0.105658357)    ! mu+- mass [GeV/c^2]
      REAL       M_PION
      PARAMETER (M_PION = 0.13957018)     ! pi+- mass [GeV/c^2]
      REAL       M_KAON
      PARAMETER (M_KAON = 0.493677)       !  K+- mass [GeV/c^2]
      REAL       M_PROT
      PARAMETER (M_PROT = 0.93827203)     !  p+  mass [GeV/c^2]
      REAL       M_NEUT
      PARAMETER (M_NEUT = 0.93956536)     !  n0  mass [GeV/c^2]
      REAL       SP_O_L
      PARAMETER (SP_O_L = 299.792458)     ! c [mm/ns]
      ! LHC simulation parameters
      REAL       E_BEAM
      PARAMETER (E_BEAM = 7000.)          ! energy per LHC beam  [GeV]
      REAL       Z_ZERO
      PARAMETER (Z_ZERO = 23000.)         ! scoring plane z [mm]
      REAL       T_ZERO
      PARAMETER (T_ZERO = -Z_ZERO/SP_O_L) ! starting time [ns]

      ! for "read from file" mode #2
      ! name and number of events of file1
      CHARACTER*100 F1NAME
      PARAMETER    (F1NAME = 'input.txt')
      INTEGER       F1N_EV
      PARAMETER    (F1N_EV = 8200000)

      ! event counting
      INTEGER NILAST,MAXLINE
      DOUBLE PRECISION NP_GEN,N_READ
      COMMON/BHGNEV/   NP_GEN,N_READ,NILAST,MAXLINE
