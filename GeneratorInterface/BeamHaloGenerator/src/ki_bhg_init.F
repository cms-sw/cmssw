************************************************************************
*                                                                      *
      SUBROUTINE KI_BHG_INIT(seed)      !      by droll (30/05/05)           *
*                                                                      *
*                initialize beam halo generator                        *
*                                                                      *
************************************************************************

      IMPLICIT NONE
cc #include "kine/kicdes/pythia.inc"
cc #include "kine/kicdes/bhgrhists.inc"
cc #include "kine/kicdes/bhgcons.inc"
cc #include "kine/kicdes/bhgctrl.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgcons.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgctrl.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgrhists.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgp_info.inc"
      INTEGER SEED,K1,K2

      INTEGER NI,M15_ID,IORIG,KORIG,COUNT,ID_PDG
      REAL    EF,W,X,Y,DCX,DCY,TOFF,PRIMEHITZ,
     &        ZORIG,XORIG,YORIG,EORIG,WORIG

      INTEGER IDCONVERT(40)
      DATA IDCONVERT/2212,2112,211,-211,321,-321,-13,13,22,11,-11,
     &              -2212,111,1,6,-104,-102,14,-14,12,-12,130,310,
     &               311,-311,3122,-3122,3222,3212,3112,-2112,3322,
     &               3312,3334,5112,5212,5222,-3322,-5132,-5332/


      PRINT*,' '
      PRINT*,'*********************************************************'
      PRINT*,'*********************************************************'
      PRINT*,'***                                                   ***'
      PRINT*,'***  B E A M   H A L O   G E N E R A T O R  (vers.2)  ***'
      PRINT*,'***                                                   ***'
      PRINT*,'*********************************************************'
      PRINT*,'*********************************************************'
      PRINT*,' '

      ! check some input parameters
      IF (GENMOD.LT.0.OR.GENMOD.GT.3) THEN
        PRINT*,'bad choice of parameter GENMOD'
        STOP
      ENDIF
      IF (LHC_B1.NE.0.AND.LHC_B1.NE.1) THEN
        PRINT*,'bad choice of parameter LHC_B1'
        STOP
      ENDIF
      IF (LHC_B2.NE.0.AND.LHC_B2.NE.1) THEN
        PRINT*,'bad choice of parameter LHC_B2'
        STOP
      ENDIF
      IF (IW_MUO.NE.1.AND.IW_HAD.NE.1) THEN
        PRINT*,'bad choice of parameters IW_MUO and IW_HAD'
        STOP
      ENDIF

      ! initialize random packages
c      SEED = MRPY(1)
      k1 = 0
      k2 = 0
      CALL RLUXGO(4,SEED,K1,K2)
      CALL RM48IN(SEED,0,0)

      IF (GENMOD.EQ.1) THEN
        CALL F66MUP  ! mu+ histos
        CALL RNHPRE(H01MUP,N_BINS)
        CALL RNHPRE(H11MUP,N_BINS)
        CALL RNHPRE(H12MUP,N_BINS)
        CALL RNHPRE(H13MUP,N_BINS)
        CALL RNHPRE(H14MUP,N_BINS)
        CALL RNHPRE(H15MUP,N_BINS)
        CALL RNHPRE(H16MUP,N_BINS)
        CALL RNHPRE(H17MUP,N_BINS)
        CALL RNHPRE(H18MUP,N_BINS)
        CALL RNHPRE(H21MUP,N_BINS)
        CALL RNHPRE(H22MUP,N_BINS)
        CALL RNHPRE(H23MUP,N_BINS)
        CALL RNHPRE(H24MUP,N_BINS)
        CALL RNHPRE(H25MUP,N_BINS)
        CALL RNHPRE(H26MUP,N_BINS)
        CALL RNHPRE(H27MUP,N_BINS)
        CALL RNHPRE(H28MUP,N_BINS)
        CALL RNHPRE(H31MUP,N_BINS)
        CALL RNHPRE(H32MUP,N_BINS)
        CALL RNHPRE(H33MUP,N_BINS)
        CALL RNHPRE(H34MUP,N_BINS)
        CALL RNHPRE(H35MUP,N_BINS)
        CALL RNHPRE(H36MUP,N_BINS)
        CALL RNHPRE(H37MUP,N_BINS)
        CALL RNHPRE(H38MUP,N_BINS)
        CALL RNHPRE(H41MUP,N_BINS)
        CALL RNHPRE(H42MUP,N_BINS)
        CALL RNHPRE(H43MUP,N_BINS)
        CALL RNHPRE(H44MUP,N_BINS)
        CALL RNHPRE(H45MUP,N_BINS)
        CALL RNHPRE(H46MUP,N_BINS)
        CALL RNHPRE(H47MUP,N_BINS)
        CALL RNHPRE(H48MUP,N_BINS)
        CALL F66MUM  ! mu- histos
        CALL RNHPRE(H01MUM,N_BINS)
        CALL RNHPRE(H11MUM,N_BINS)
        CALL RNHPRE(H12MUM,N_BINS)
        CALL RNHPRE(H13MUM,N_BINS)
        CALL RNHPRE(H14MUM,N_BINS)
        CALL RNHPRE(H15MUM,N_BINS)
        CALL RNHPRE(H16MUM,N_BINS)
        CALL RNHPRE(H17MUM,N_BINS)
        CALL RNHPRE(H18MUM,N_BINS)
        CALL RNHPRE(H21MUM,N_BINS)
        CALL RNHPRE(H22MUM,N_BINS)
        CALL RNHPRE(H23MUM,N_BINS)
        CALL RNHPRE(H24MUM,N_BINS)
        CALL RNHPRE(H25MUM,N_BINS)
        CALL RNHPRE(H26MUM,N_BINS)
        CALL RNHPRE(H27MUM,N_BINS)
        CALL RNHPRE(H28MUM,N_BINS)
        CALL RNHPRE(H31MUM,N_BINS)
        CALL RNHPRE(H32MUM,N_BINS)
        CALL RNHPRE(H33MUM,N_BINS)
        CALL RNHPRE(H34MUM,N_BINS)
        CALL RNHPRE(H35MUM,N_BINS)
        CALL RNHPRE(H36MUM,N_BINS)
        CALL RNHPRE(H37MUM,N_BINS)
        CALL RNHPRE(H38MUM,N_BINS)
        CALL RNHPRE(H41MUM,N_BINS)
        CALL RNHPRE(H42MUM,N_BINS)
        CALL RNHPRE(H43MUM,N_BINS)
        CALL RNHPRE(H44MUM,N_BINS)
        CALL RNHPRE(H45MUM,N_BINS)
        CALL RNHPRE(H46MUM,N_BINS)
        CALL RNHPRE(H47MUM,N_BINS)
        CALL RNHPRE(H48MUM,N_BINS)
        CALL F66PIP  ! pi+ histos
        CALL RNHPRE(H01PIP,N_BINS)
        CALL RNHPRE(H11PIP,N_BINS)
        CALL RNHPRE(H12PIP,N_BINS)
        CALL RNHPRE(H13PIP,N_BINS)
        CALL RNHPRE(H14PIP,N_BINS)
        CALL RNHPRE(H15PIP,N_BINS)
        CALL RNHPRE(H16PIP,N_BINS)
        CALL RNHPRE(H17PIP,N_BINS)
        CALL RNHPRE(H18PIP,N_BINS)
        CALL RNHPRE(H21PIP,N_BINS)
        CALL RNHPRE(H22PIP,N_BINS)
        CALL RNHPRE(H23PIP,N_BINS)
        CALL RNHPRE(H24PIP,N_BINS)
        CALL RNHPRE(H25PIP,N_BINS)
        CALL RNHPRE(H26PIP,N_BINS)
        CALL RNHPRE(H27PIP,N_BINS)
        CALL RNHPRE(H28PIP,N_BINS)
        CALL RNHPRE(H31PIP,N_BINS)
        CALL RNHPRE(H32PIP,N_BINS)
        CALL RNHPRE(H33PIP,N_BINS)
        CALL RNHPRE(H34PIP,N_BINS)
        CALL RNHPRE(H35PIP,N_BINS)
        CALL RNHPRE(H36PIP,N_BINS)
        CALL RNHPRE(H37PIP,N_BINS)
        CALL RNHPRE(H38PIP,N_BINS)
        CALL RNHPRE(H41PIP,N_BINS)
        CALL RNHPRE(H42PIP,N_BINS)
        CALL RNHPRE(H43PIP,N_BINS)
        CALL RNHPRE(H44PIP,N_BINS)
        CALL RNHPRE(H45PIP,N_BINS)
        CALL RNHPRE(H46PIP,N_BINS)
        CALL RNHPRE(H47PIP,N_BINS)
        CALL RNHPRE(H48PIP,N_BINS)
        CALL F66PIM  ! pi- histos
        CALL RNHPRE(H01PIM,N_BINS)
        CALL RNHPRE(H11PIM,N_BINS)
        CALL RNHPRE(H12PIM,N_BINS)
        CALL RNHPRE(H13PIM,N_BINS)
        CALL RNHPRE(H14PIM,N_BINS)
        CALL RNHPRE(H15PIM,N_BINS)
        CALL RNHPRE(H16PIM,N_BINS)
        CALL RNHPRE(H17PIM,N_BINS)
        CALL RNHPRE(H18PIM,N_BINS)
        CALL RNHPRE(H21PIM,N_BINS)
        CALL RNHPRE(H22PIM,N_BINS)
        CALL RNHPRE(H23PIM,N_BINS)
        CALL RNHPRE(H24PIM,N_BINS)
        CALL RNHPRE(H25PIM,N_BINS)
        CALL RNHPRE(H26PIM,N_BINS)
        CALL RNHPRE(H27PIM,N_BINS)
        CALL RNHPRE(H28PIM,N_BINS)
        CALL RNHPRE(H31PIM,N_BINS)
        CALL RNHPRE(H32PIM,N_BINS)
        CALL RNHPRE(H33PIM,N_BINS)
        CALL RNHPRE(H34PIM,N_BINS)
        CALL RNHPRE(H35PIM,N_BINS)
        CALL RNHPRE(H36PIM,N_BINS)
        CALL RNHPRE(H37PIM,N_BINS)
        CALL RNHPRE(H38PIM,N_BINS)
        CALL RNHPRE(H41PIM,N_BINS)
        CALL RNHPRE(H42PIM,N_BINS)
        CALL RNHPRE(H43PIM,N_BINS)
        CALL RNHPRE(H44PIM,N_BINS)
        CALL RNHPRE(H45PIM,N_BINS)
        CALL RNHPRE(H46PIM,N_BINS)
        CALL RNHPRE(H47PIM,N_BINS)
        CALL RNHPRE(H48PIM,N_BINS)
        CALL F66KAP  ! K+ histos
        CALL RNHPRE(H01KAP,N_BINS)
        CALL RNHPRE(H11KAP,N_BINS)
        CALL RNHPRE(H12KAP,N_BINS)
        CALL RNHPRE(H13KAP,N_BINS)
        CALL RNHPRE(H14KAP,N_BINS)
        CALL RNHPRE(H15KAP,N_BINS)
        CALL RNHPRE(H16KAP,N_BINS)
        CALL RNHPRE(H17KAP,N_BINS)
        CALL RNHPRE(H18KAP,N_BINS)
        CALL RNHPRE(H21KAP,N_BINS)
        CALL RNHPRE(H22KAP,N_BINS)
        CALL RNHPRE(H23KAP,N_BINS)
        CALL RNHPRE(H24KAP,N_BINS)
        CALL RNHPRE(H25KAP,N_BINS)
        CALL RNHPRE(H26KAP,N_BINS)
        CALL RNHPRE(H27KAP,N_BINS)
        CALL RNHPRE(H28KAP,N_BINS)
        CALL RNHPRE(H31KAP,N_BINS)
        CALL RNHPRE(H32KAP,N_BINS)
        CALL RNHPRE(H33KAP,N_BINS)
        CALL RNHPRE(H34KAP,N_BINS)
        CALL RNHPRE(H35KAP,N_BINS)
        CALL RNHPRE(H36KAP,N_BINS)
        CALL RNHPRE(H37KAP,N_BINS)
        CALL RNHPRE(H38KAP,N_BINS)
        CALL RNHPRE(H41KAP,N_BINS)
        CALL RNHPRE(H42KAP,N_BINS)
        CALL RNHPRE(H43KAP,N_BINS)
        CALL RNHPRE(H44KAP,N_BINS)
        CALL RNHPRE(H45KAP,N_BINS)
        CALL RNHPRE(H46KAP,N_BINS)
        CALL RNHPRE(H47KAP,N_BINS)
        CALL RNHPRE(H48KAP,N_BINS)
        CALL F66KAM  ! K- histos
        CALL RNHPRE(H01KAM,N_BINS)
        CALL RNHPRE(H11KAM,N_BINS)
        CALL RNHPRE(H12KAM,N_BINS)
        CALL RNHPRE(H13KAM,N_BINS)
        CALL RNHPRE(H14KAM,N_BINS)
        CALL RNHPRE(H15KAM,N_BINS)
        CALL RNHPRE(H16KAM,N_BINS)
        CALL RNHPRE(H17KAM,N_BINS)
        CALL RNHPRE(H18KAM,N_BINS)
        CALL RNHPRE(H21KAM,N_BINS)
        CALL RNHPRE(H22KAM,N_BINS)
        CALL RNHPRE(H23KAM,N_BINS)
        CALL RNHPRE(H24KAM,N_BINS)
        CALL RNHPRE(H25KAM,N_BINS)
        CALL RNHPRE(H26KAM,N_BINS)
        CALL RNHPRE(H27KAM,N_BINS)
        CALL RNHPRE(H28KAM,N_BINS)
        CALL RNHPRE(H31KAM,N_BINS)
        CALL RNHPRE(H32KAM,N_BINS)
        CALL RNHPRE(H33KAM,N_BINS)
        CALL RNHPRE(H34KAM,N_BINS)
        CALL RNHPRE(H35KAM,N_BINS)
        CALL RNHPRE(H36KAM,N_BINS)
        CALL RNHPRE(H37KAM,N_BINS)
        CALL RNHPRE(H38KAM,N_BINS)
        CALL RNHPRE(H41KAM,N_BINS)
        CALL RNHPRE(H42KAM,N_BINS)
        CALL RNHPRE(H43KAM,N_BINS)
        CALL RNHPRE(H44KAM,N_BINS)
        CALL RNHPRE(H45KAM,N_BINS)
        CALL RNHPRE(H46KAM,N_BINS)
        CALL RNHPRE(H47KAM,N_BINS)
        CALL RNHPRE(H48KAM,N_BINS)
        CALL F66PRO  ! p+ histos
        CALL RNHPRE(H01PRO,N_BINS)
        CALL RNHPRE(H11PRO,N_BINS)
        CALL RNHPRE(H12PRO,N_BINS)
        CALL RNHPRE(H13PRO,N_BINS)
        CALL RNHPRE(H14PRO,N_BINS)
        CALL RNHPRE(H15PRO,N_BINS)
        CALL RNHPRE(H16PRO,N_BINS)
        CALL RNHPRE(H17PRO,N_BINS)
        CALL RNHPRE(H18PRO,N_BINS)
        CALL RNHPRE(H21PRO,N_BINS)
        CALL RNHPRE(H22PRO,N_BINS)
        CALL RNHPRE(H23PRO,N_BINS)
        CALL RNHPRE(H24PRO,N_BINS)
        CALL RNHPRE(H25PRO,N_BINS)
        CALL RNHPRE(H26PRO,N_BINS)
        CALL RNHPRE(H27PRO,N_BINS)
        CALL RNHPRE(H28PRO,N_BINS)
        CALL RNHPRE(H31PRO,N_BINS)
        CALL RNHPRE(H32PRO,N_BINS)
        CALL RNHPRE(H33PRO,N_BINS)
        CALL RNHPRE(H34PRO,N_BINS)
        CALL RNHPRE(H35PRO,N_BINS)
        CALL RNHPRE(H36PRO,N_BINS)
        CALL RNHPRE(H37PRO,N_BINS)
        CALL RNHPRE(H38PRO,N_BINS)
        CALL RNHPRE(H41PRO,N_BINS)
        CALL RNHPRE(H42PRO,N_BINS)
        CALL RNHPRE(H43PRO,N_BINS)
        CALL RNHPRE(H44PRO,N_BINS)
        CALL RNHPRE(H45PRO,N_BINS)
        CALL RNHPRE(H46PRO,N_BINS)
        CALL RNHPRE(H47PRO,N_BINS)
        CALL RNHPRE(H48PRO,N_BINS)
        CALL F66NEU  ! n0 histos
        CALL RNHPRE(H01NEU,N_BINS)
        CALL RNHPRE(H11NEU,N_BINS)
        CALL RNHPRE(H12NEU,N_BINS)
        CALL RNHPRE(H13NEU,N_BINS)
        CALL RNHPRE(H14NEU,N_BINS)
        CALL RNHPRE(H15NEU,N_BINS)
        CALL RNHPRE(H16NEU,N_BINS)
        CALL RNHPRE(H17NEU,N_BINS)
        CALL RNHPRE(H18NEU,N_BINS)
        CALL RNHPRE(H21NEU,N_BINS)
        CALL RNHPRE(H22NEU,N_BINS)
        CALL RNHPRE(H23NEU,N_BINS)
        CALL RNHPRE(H24NEU,N_BINS)
        CALL RNHPRE(H25NEU,N_BINS)
        CALL RNHPRE(H26NEU,N_BINS)
        CALL RNHPRE(H27NEU,N_BINS)
        CALL RNHPRE(H28NEU,N_BINS)
        CALL RNHPRE(H31NEU,N_BINS)
        CALL RNHPRE(H32NEU,N_BINS)
        CALL RNHPRE(H33NEU,N_BINS)
        CALL RNHPRE(H34NEU,N_BINS)
        CALL RNHPRE(H35NEU,N_BINS)
        CALL RNHPRE(H36NEU,N_BINS)
        CALL RNHPRE(H37NEU,N_BINS)
        CALL RNHPRE(H38NEU,N_BINS)
        CALL RNHPRE(H41NEU,N_BINS)
        CALL RNHPRE(H42NEU,N_BINS)
        CALL RNHPRE(H43NEU,N_BINS)
        CALL RNHPRE(H44NEU,N_BINS)
        CALL RNHPRE(H45NEU,N_BINS)
        CALL RNHPRE(H46NEU,N_BINS)
        CALL RNHPRE(H47NEU,N_BINS)
        CALL RNHPRE(H48NEU,N_BINS)
      ENDIF

      ! initialize counters, limits, rates, ratios
      NP_GEN = 0.
      N_READ = 0.
      CALL INILIM
      CALL INIRES
      CALL INIROS

      ! open input file
      IF (GENMOD.EQ.2) OPEN(22,FILE=F1NAME,STATUS='UNKNOWN')

      IF (GENMOD.EQ.3) THEN
         MAXLINE=0
         OPEN(22,FILE=G3FNAME,STATUS='UNKNOWN')
         READ(22,*) MAXLINE
         NPRIME = 0.0
         NSTACK = 0.0
         COUNT = 0
         DO WHILE( COUNT .LT. OFFSET )
            READ(22,*) NI,M15_ID,EF,W,X,Y,DCX,DCY,TOFF,PRIMEHITZ,
     &                 ZORIG,XORIG,YORIG,EORIG,WORIG,IORIG,KORIG

            ID_PDG = IDCONVERT(M15_ID)
            IF (IW_MUO.EQ.1) THEN
               IF (EF.GT.EMUMIN.AND.EF.LT.EMUMAX) THEN
                  IF (ID_PDG.EQ. -13) COUNT = COUNT + 1 ! accept mu+
                  IF (ID_PDG.EQ.  13) COUNT = COUNT + 1 ! accept mu-
               ENDIF
            ENDIF
            IF (IW_HAD.EQ.1) THEN
               IF (EF.GT.EPIMIN.AND.EF.LT.EPIMAX) THEN
                  IF (ID_PDG.EQ. 211) COUNT = COUNT + 1 ! accept pi+
                  IF (ID_PDG.EQ.-211) COUNT = COUNT + 1 ! accept pi-
               ENDIF
               IF (EF.GT.EKAMIN.AND.EF.LT.EKAMAX) THEN
                  IF (ID_PDG.EQ. 321) COUNT = COUNT + 1 ! accept K+
                  IF (ID_PDG.EQ.-321) COUNT = COUNT + 1 ! accept K-
               ENDIF
               IF (EF.GT.EPRMIN.AND.EF.LT.EPRMAX) THEN
                  IF (ID_PDG.EQ.2212) COUNT = COUNT + 1 ! accept p+
               ENDIF
               IF (EF.GT.ENEMIN.AND.EF.LT.ENEMAX) THEN
                  IF (ID_PDG.EQ.2112) COUNT = COUNT + 1 ! accept n0
               ENDIF
            ENDIF
         ENDDO
         
      ENDIF


      RETURN

      END                                                                     

************************************************************************
      SUBROUTINE INILIM      !      by droll (01/06/05)                *
*                calculate limits for kinetic E for each particle type *
************************************************************************

      IMPLICIT NONE
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgcons.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgp_info.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgctrl.inc"

      ! ranges of kinetic energies
      EMUMIN = MAX(EG_MIN-M_MUON,1.)
      EMUMAX = MIN(EG_MAX-M_MUON,E_BEAM-M_MUON)
      EPIMIN = MAX(EG_MIN-M_PION,1.)
      EPIMAX = MIN(EG_MAX-M_PION,E_BEAM-M_PION)
      EKAMIN = MAX(EG_MIN-M_KAON,1.)
      EKAMAX = MIN(EG_MAX-M_KAON,E_BEAM-M_KAON)
      EPRMIN = MAX(EG_MIN-M_PROT,1.)
      EPRMAX = MIN(EG_MAX-M_PROT,E_BEAM-M_PROT)
      ENEMIN = MAX(EG_MIN-M_NEUT,1.)
      ENEMAX = MIN(EG_MAX-M_NEUT,E_BEAM-M_NEUT)

      ! for max.E >= E_BEAM allow slighly higher energies (e.g. off-beam protons)
      IF ((E_BEAM/EG_MAX).LT.1.0002) THEN
        EMUMAX = 1.0002*E_BEAM - M_MUON
        EPIMAX = 1.0002*E_BEAM - M_PION
        EKAMAX = 1.0002*E_BEAM - M_KAON
        EPRMAX = 1.0002*E_BEAM - M_PROT
        ENEMAX = 1.0002*E_BEAM - M_NEUT
      ENDIF

      RETURN
      END

************************************************************************
      SUBROUTINE INIRES      !      by droll (01/06/05)                *
*                initialize/calculate rates for each particle type     *
************************************************************************

      IMPLICIT NONE
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgp_info.inc"
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgctrl.inc"
      REAL GR_MUP,GR_MUM,GR_PIP,GR_PIM,GR_KAP,GR_KAM,GR_PRO,GR_NEU

      ! initialize rates for each particle type
      R_MU_P = 0.
      R_MU_M = 0.
      R_PI_P = 0.
      R_PI_M = 0.
      R_KA_P = 0.
      R_KA_M = 0.
      R_PROT = 0.
      R_NEUT = 0.
      IF (GENMOD.LE.1.AND.IW_MUO.EQ.1) THEN
        R_MU_P = GR_MUP()
        R_MU_M = GR_MUM()
      ENDIF
      IF (GENMOD.LE.1.AND.IW_HAD.EQ.1) THEN
        R_PI_P = GR_PIP()
        R_PI_M = GR_PIM()
        R_KA_P = GR_KAP()
        R_KA_M = GR_KAM()
        R_PROT = GR_PRO()
        R_NEUT = GR_NEU()
      ENDIF

      RETURN
      END

************************************************************************
      SUBROUTINE INIROS      !      by droll (01/06/05)                *
*                calculate ratios of rates for each particle type      *
************************************************************************

      IMPLICIT NONE
#include "GeneratorInterface/BeamHaloGenerator/interface/bhgp_info.inc"
      DOUBLE PRECISION SRATES

      ! calculate rate fractions for each particle type
      SRATES = R_MU_P+R_PI_P+R_KA_P+R_PROT+R_NEUT
     &       + R_MU_M+R_PI_M+R_KA_M
      FR_MUP = R_MU_P/SRATES
      FR_MUM = R_MU_M/SRATES
      FR_PIP = R_PI_P/SRATES
      FR_PIM = R_PI_M/SRATES
      FR_KAP = R_KA_P/SRATES
      FR_KAM = R_KA_M/SRATES
      FR_PRO = R_PROT/SRATES
      FR_NEU = R_NEUT/SRATES

      RETURN
      END
