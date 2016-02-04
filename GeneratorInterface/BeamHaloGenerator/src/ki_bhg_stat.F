************************************************************************
*                                                                      *
      SUBROUTINE KI_BHG_STAT(iret)      !      by droll (18/12/04)           *
*                                                                      *
*                print statistical output and prepare termination      *
*                                                                      *
************************************************************************

      IMPLICIT NONE
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgcons.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgp_info.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgctrl.inc"
      REAL RT_MUP,RT_PIP,RT_KAP,RT_PRO,RT_NEU
     &    ,RT_MUM,RT_PIM,RT_KAM
      REAL SUM_BS,RATE_S,TIME_I,F_FRAC

      integer iret
      iret = 0

      PRINT*,'*********************************************************'
      PRINT*,'*********************************************************'
      PRINT*,'***                                                   ***'
      PRINT*,'***     B E A M    H A L O    S T A T I S T I C S     ***'
      PRINT*,'***                                                   ***'
      PRINT*,'*********************************************************'
      PRINT*,'*********************************************************'
      PRINT*,' '
      PRINT*,'        number of generated events:',NINT(NP_GEN)
      IF (GENMOD.EQ.0) PRINT*,'        (no events are generated)'
      IF (GENMOD.EQ.1) PRINT*,'        (events are unweighted)'
      IF (GENMOD.EQ.2) PRINT*,'        (events are weighted)'
      IF (GENMOD.EQ.3) PRINT*,'        (events weighted, mars)' 
      IF (GENMOD.EQ.2) THEN
        F_FRAC = REAL(N_READ)/REAL(F1N_EV)*100.
        PRINT*,'        processed',F_FRAC,'% of all events'
        IF (NINT(N_READ).NE.F1N_EV)
     &  PRINT*,'        WARNING: rates and statistics are too low'
      ENDIF
      PRINT*,' '
      IF (LHC_B1.EQ.1) PRINT*,'        LHC beam 1 is switched ON'
      IF (LHC_B1.EQ.0) PRINT*,'        LHC beam 1 is switched OFF'
      IF (LHC_B2.EQ.1) PRINT*,'        LHC beam 2 is switched ON'
      IF (LHC_B2.EQ.0) PRINT*,'        LHC beam 2 is switched OFF'
      PRINT*,' '
      PRINT*,'        minimal energy generated:',EG_MIN,' GeV'
      PRINT*,'        maximal energy generated:',EG_MAX,' GeV'
      PRINT*,' '
      SUM_BS = REAL(LHC_B1+LHC_B2) ! sum of beams
      if( genmod .eq. 3 ) sum_bs = sum_bs/nprime
      RT_MUP = REAL(R_MU_P)*SUM_BS ! mu+ rate [Hz]
      PRINT*,'        rate of generated mu+:',RT_MUP,' Hz'
      RT_MUM = REAL(R_MU_M)*SUM_BS ! mu- rate [Hz]
      PRINT*,'        rate of generated mu-:',RT_MUM,' Hz'
      RT_PIP = REAL(R_PI_P)*SUM_BS ! pi+ rate [Hz]
      PRINT*,'        rate of generated pi+:',RT_PIP,' Hz'
      RT_PIM = REAL(R_PI_M)*SUM_BS ! pi- rate [Hz]
      PRINT*,'        rate of generated pi-:',RT_PIM,' Hz'
      RT_KAP = REAL(R_KA_P)*SUM_BS ! K+ rate [Hz]
      PRINT*,'        rate of generated  K+:',RT_KAP,' Hz'
      RT_KAM = REAL(R_KA_M)*SUM_BS ! K- rate [Hz]
      PRINT*,'        rate of generated  K-:',RT_KAM,' Hz'
      RT_PRO = REAL(R_PROT)*SUM_BS ! p+ rate [Hz]
      PRINT*,'        rate of generated  p+:',RT_PRO,' Hz'
      RT_NEU = REAL(R_NEUT)*SUM_BS ! n0 rate [Hz]
      PRINT*,'        rate of generated  n0:',RT_NEU,' Hz'
      RATE_S = RT_MUP+RT_PIP+RT_KAP+RT_PRO+RT_NEU
     &       + RT_MUM+RT_PIM+RT_KAM ! particle rate [Hz]
      TIME_I = REAL(NP_GEN)/RATE_S  ! time interval for all particles in ntuple
      PRINT*,'        -> rate:',RATE_S,' particles per second'
      IF (GENMOD.EQ.1) THEN
        PRINT*,'        -> # of events in your ntuple corresponds'
        PRINT*,'           to a time interval of',TIME_I,' s'
      ENDIF
      PRINT*,' '
      PRINT*,'*********************************************************'
      PRINT*,'*********************************************************'

      IF (GENMOD.EQ.2) CLOSE(22)
      IF (GENMOD.EQ.3) CLOSE(22)
      RETURN
      
      END

