*****************************************************************
*                                                               *
*     DESCRIPTION: this include file contains common blocks     *
*                  with kinetic energy ranges and               *
*                  with particle rate counters and              *
*                  with particle rate fractions and             *
*                  with random function infos                   *
*                                                               *
*     last update (31/05/2005) by droll                         *
*                                                               *
*****************************************************************

      ! allowed kinetic energy ranges for event generation
      !              mu+-   pi+-   K+-    p+     n0
      REAL          EMUMIN,EPIMIN,EKAMIN,EPRMIN,ENEMIN ! minimum
     &             ,EMUMAX,EPIMAX,EKAMAX,EPRMAX,ENEMAX ! maximum
      COMMON/KINENE/EMUMIN,EPIMIN,EKAMIN,EPRMIN,ENEMIN
     &             ,EMUMAX,EPIMAX,EKAMAX,EPRMAX,ENEMAX

      ! particle rate counters
      !                 mu+    pi+    K+     p+     n0
      !                 mu-    pi-    K-
      DOUBLE PRECISION R_MU_P,R_PI_P,R_KA_P,R_PROT,R_NEUT
     &                ,R_MU_M,R_PI_M,R_KA_M,NPRIME,NSTACK
      COMMON/PRATES/   R_MU_P,R_PI_P,R_KA_P,R_PROT,R_NEUT
     &                ,R_MU_M,R_PI_M,R_KA_M,NPRIME,NSTACK

      ! particle rate fractions
      !                 mu+    pi+    K+     p+     n0
      !                 mu-    pi-    K-
      REAL          FR_MUP,FR_PIP,FR_KAP,FR_PRO,FR_NEU
     &             ,FR_MUM,FR_PIM,FR_KAM
      COMMON/PARFRC/FR_MUP,FR_PIP,FR_KAP,FR_PRO,FR_NEU
     &             ,FR_MUM,FR_PIM,FR_KAM
