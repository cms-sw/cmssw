## Important note!
## This file has been modified by hand to give the gluino and the 
## stop_1 a very narrow width, such that it can be used to try out
## the R-hadron machinery. It is not a realistic SUSY scenario.
##
##******************************************************************
##                      MadGraph/MadEvent                          *
##******************************************************************
##                                                                 *
##  param_card corresponding the SPS point 1a (by SoftSusy 2.0.5)  *
##                                                                 *
##******************************************************************
## Les Houches friendly file for the (MS)SM parameters of MadGraph *
##      SM parameter set and decay widths produced by MSSMCalc     *
##******************************************************************
##*Please note the following IMPORTANT issues:                     *
##                                                                 *
##0. REFRAIN from editing this file by hand! Some of the parame-   *
##   ters are not independent. Always use a calculator.            *
##                                                                 *
##1. alpha_S(MZ) has been used in the calculation of the parameters*
##   This value is KEPT by madgraph when no pdf are used lpp(i)=0, *
##   but, for consistency, it will be reset by madgraph to the     *
##   value expected IF the pdfs for collisions with hadrons are    *
##   used.                                                         *
##                                                                 *
##2. Values of the charm and bottom kinematic (pole) masses are    *
##   those used in the matrix elements and phase space UNLESS they *
##   are set to ZERO from the start in the model (particles.dat)   *
##   This happens, for example,  when using 5-flavor QCD where     *
##   charm and bottom are treated as partons in the initial state  *
##   and a zero mass might be hardwired in the model definition.   *
##                                                                 *
##       The SUSY decays have calculated using SDECAY 1.1a         *
##                                                                 *
##******************************************************************
#
BLOCK DCINFO  # Decay Program information
     1   SDECAY      # decay calculator
     2   1.1a        # version number
#
BLOCK SPINFO  # Spectrum calculator information
     1   SOFTSUSY    # spectrum calculator                 
     2   2.0.5         # version number                    
#
BLOCK MODSEL  # Model selection
     1     1   sugra                                             
#
BLOCK SMINPUTS  # Standard Model inputs
     1     1.27934000E+02   # alpha_em^-1(M_Z)^MSbar
     2     1.16637000E-05   # G_F [GeV^-2]
     3     1.18000000E-01   # alpha_S(M_Z)^MSbar
     4     9.11876000E+01   # M_Z pole mass
     5     4.25000000E+00   # mb(mb)^MSbar
     6     1.75000000E+02   # mt pole mass
     7     1.77700000E+00   # mtau pole mass
#
BLOCK MINPAR  # Input parameters - minimal models
     1     1.00000000E+02   # m0                  
     2     2.50000000E+02   # m12                 
     3     1.00000000E+01   # tanb                
     4     1.00000000E+00   # sign(mu)            
     5    -1.00000000E+02   # A0                  
#
BLOCK MASS  # Mass Spectrum
# PDG code           mass       particle
         5     4.88991651E+00   # b-quark pole mass calculated from mb(mb)_Msbar
         6     1.75000000E+02   # mt pole mass (not read by ME)
        24     7.98290131E+01   # W+
        25     1.10899057E+02   # h
        35     3.99960116E+05   # H
        36     3.99583917E+05   # A
        37     4.07879012E+05   # H+
   1000001     5.68441109E+05   # ~d_L
   2000001     5.45228462E+05   # ~d_R
   1000002     5.61119014E+05   # ~u_L
   2000002     5.49259265E+05   # ~u_R
   1000003     5.68441109E+05   # ~s_L
   2000003     5.45228462E+05   # ~s_R
   1000004     5.61119014E+05   # ~c_L
   2000004     5.49259265E+05   # ~c_R
   1000005     5.13065179E+05   # ~b_1
   2000005     5.43726676E+05   # ~b_2
   1000006     3.0E+05   # ~t_1
   2000006     5.85785818E+05   # ~t_2
   1000011     2.02915690E+05   # ~e_L
   2000011     1.44102799E+05   # ~e_R
   1000012     1.85258326E+05   # ~nu_eL
   1000013     2.02915690E+05   # ~mu_L
   2000013     1.44102799E+05   # ~mu_R
   1000014     1.85258326E+05   # ~nu_muL
   1000015     1.34490864E+05   # ~tau_1
   2000015     2.06867805E+05   # ~tau_2
   1000016     1.84708464E+05   # ~nu_tauL
   1000021     1500.00   # ~g
   1000022     9.66880686E+05   # ~chi_10
   1000023     1.81088157E+05   # ~chi_20
   1000025    -3.63756027E+05   # ~chi_30
   1000035     3.81729382E+05   # ~chi_40
   1000024     1.81696474E+05   # ~chi_1+
   1000037     3.79939320E+05   # ~chi_2+
#
BLOCK NMIX  # Neutralino Mixing Matrix
  1  1     9.86364430E-01   # N_11
  1  2    -5.31103553E-02   # N_12
  1  3     1.46433995E-01   # N_13
  1  4    -5.31186117E-02   # N_14
  2  1     9.93505358E-02   # N_21
  2  2     9.44949299E-01   # N_22
  2  3    -2.69846720E-01   # N_23
  2  4     1.56150698E-01   # N_24
  3  1    -6.03388002E-02   # N_31
  3  2     8.77004854E-02   # N_32
  3  3     6.95877493E-01   # N_33
  3  4     7.10226984E-01   # N_34
  4  1    -1.16507132E-01   # N_41
  4  2     3.10739017E-01   # N_42
  4  3     6.49225960E-01   # N_43
  4  4    -6.84377823E-01   # N_44
#
BLOCK UMIX  # Chargino Mixing Matrix U
  1  1     9.16834859E-01   # U_11
  1  2    -3.99266629E-01   # U_12
  2  1     3.99266629E-01   # U_21
  2  2     9.16834859E-01   # U_22
#
BLOCK VMIX  # Chargino Mixing Matrix V
  1  1     9.72557835E-01   # V_11
  1  2    -2.32661249E-01   # V_12
  2  1     2.32661249E-01   # V_21
  2  2     9.72557835E-01   # V_22
#
BLOCK STOPMIX  # Stop Mixing Matrix
  1  1     5.53644960E-01   # O_{11}
  1  2     8.32752820E-01   # O_{12}
  2  1     8.32752820E-01   # O_{21}
  2  2    -5.53644960E-01   # O_{22}
#
BLOCK SBOTMIX  # Sbottom Mixing Matrix
  1  1     9.38737896E-01   # O_{11}
  1  2     3.44631925E-01   # O_{12}
  2  1    -3.44631925E-01   # O_{21}
  2  2     9.38737896E-01   # O_{22}
#
BLOCK STAUMIX  # Stau Mixing Matrix
  1  1     2.82487190E-01   # O_{11}
  1  2     9.59271071E-01   # O_{12}
  2  1     9.59271071E-01   # O_{21}
  2  2    -2.82487190E-01   # O_{22}
#
BLOCK ALPHA  # Higgs mixing
          -1.13825210E-01   # Mixing angle in the neutral Higgs boson sector
#
BLOCK HMIX Q=  4.67034192E+02  # DRbar Higgs Parameters
     1     3.57680977E+02   # mu(Q)MSSM DRbar     
     2     9.74862403E+00   # tan beta(Q)MSSM DRba
     3     2.44894549E+02   # higgs vev(Q)MSSM DRb
     4     1.66439065E+05   # mA^2(Q)MSSM DRbar   
#
BLOCK GAUGE Q=  4.67034192E+02  # The gauge couplings
     3     1.10178679E+00   # g3(Q) MSbar
#
BLOCK AU Q=  4.67034192E+02  # The trilinear couplings
  1  1     0.00000000E+00   # A_u(Q) DRbar
  2  2     0.00000000E+00   # A_c(Q) DRbar
  3  3    -4.98129778E+02   # A_t(Q) DRbar
#
BLOCK AD Q=  4.67034192E+02  # The trilinear couplings
  1  1     0.00000000E+00   # A_d(Q) DRbar
  2  2     0.00000000E+00   # A_s(Q) DRbar
  3  3    -7.97274397E+02   # A_b(Q) DRbar
#
BLOCK AE Q=  4.67034192E+02  # The trilinear couplings
  1  1     0.00000000E+00   # A_e(Q) DRbar
  2  2     0.00000000E+00   # A_mu(Q) DRbar
  3  3    -2.51776873E+02   # A_tau(Q) DRbar
#
BLOCK YU Q=  4.67034192E+02  # The Yukawa couplings
  3  3     8.92844550E-01   # y_t(Q) DRbar
#
BLOCK YD Q=  4.67034192E+02  # The Yukawa couplings
  3  3     1.38840206E-01   # y_b(Q) DRbar
#
BLOCK YE Q=  4.67034192E+02  # The Yukawa couplings
  3  3     1.00890810E-01   # y_tau(Q) DRbar
#
BLOCK MSOFT Q=  4.67034192E+02  # The soft SUSY breaking masses at the scale Q
     1     1.01396534E+02   # M_1(Q)              
     2     1.91504241E+02   # M_2(Q)              
     3     5.88263031E+02   # M_3(Q)              
    21     3.23374943E+04   # mH1^2(Q)            
    22    -1.28800134E+05   # mH2^2(Q)            
    31     1.95334764E+02   # meL(Q)              
    32     1.95334764E+02   # mmuL(Q)             
    33     1.94495956E+02   # mtauL(Q)            
    34     1.36494061E+02   # meR(Q)              
    35     1.36494061E+02   # mmuR(Q)             
    36     1.34043428E+02   # mtauR(Q)            
    41     5.47573466E+02   # mqL1(Q)             
    42     5.47573466E+02   # mqL2(Q)             
    43     4.98763839E+02   # mqL3(Q)             
    44     5.29511195E+02   # muR(Q)              
    45     5.29511195E+02   # mcR(Q)              
    46     4.23245877E+02   # mtR(Q)              
    47     5.23148807E+02   # mdR(Q)              
    48     5.23148807E+02   # msR(Q)              
    49     5.19867261E+02   # mbR(Q)              
#
#
#
#                             =================
#                             |The decay table|
#                             =================
#
# - The multi-body decays for the inos, stops and sbottoms are included.
#
#         PDG            Width
DECAY        25     1.98610799E-03   # h decays
#          BR         NDA      ID1       ID2
     1.45642955E-01    2          15       -15   # BR(H1 -> tau- tau+)
     8.19070713E-01    2           5        -5   # BR(H1 -> b bb)
     3.36338173E-02    2          24       -24   # BR(H1 -> W+ W-)
     1.65251528E-03    2          23        23   # BR(H1 -> Z Z)
#
#         PDG            Width
DECAY        35     5.74801389E-01   # H decays
#          BR         NDA      ID1       ID2
     1.39072676E-01    2          15       -15   # BR(H -> tau- tau+)
     4.84110879E-02    2           6        -6   # BR(H -> t tb)
     7.89500067E-01    2           5        -5   # BR(H -> b bb)
     3.87681171E-03    2          24       -24   # BR(H -> W+ W-)
     1.80454752E-03    2          23        23   # BR(H -> Z Z)
     0.00000000E+00    2          24       -37   # BR(H -> W+ H-)
     0.00000000E+00    2         -24        37   # BR(H -> W- H+)
     0.00000000E+00    2          37       -37   # BR(H -> H+ H-)
     1.73348101E-02    2          25        25   # BR(H -> h h)
     0.00000000E+00    2          36        36   # BR(H -> A A)
#
#         PDG            Width
DECAY        36     6.32178488E-01   # A decays
#          BR         NDA      ID1       ID2
     1.26659725E-01    2          15       -15   # BR(A -> tau- tau+)
     1.51081526E-01    2           6        -6   # BR(A -> t tb)
     7.19406137E-01    2           5        -5   # BR(A -> b bb)
     2.85261228E-03    2          23        25   # BR(A -> Z h)
     0.00000000E+00    2          23        35   # BR(A -> Z H)
     0.00000000E+00    2          24       -37   # BR(A -> W+ H-)
     0.00000000E+00    2         -24        37   # BR(A -> W- H+)
#
#         PDG            Width
DECAY        37     5.46962813E-01   # H+ decays
#          BR         NDA      ID1       ID2
     1.49435135E-01    2         -15        16   # BR(H+ -> tau+ nu_tau)
     8.46811711E-01    2           6        -5   # BR(H+ -> t bb)
     3.75315387E-03    2          24        25   # BR(H+ -> W+ h)
     0.00000000E+00    2          24        35   # BR(H+ -> W+ H)
     0.00000000E+00    2          24        36   # BR(H+ -> W+ A)
#
#         PDG            Width
DECAY   1000021     0.00E+00   # gluino decays
#          BR         NDA      ID1       ID2
     2.08454202E-02    2     1000001        -1   # BR(~g -> ~d_L  db)
     2.08454202E-02    2    -1000001         1   # BR(~g -> ~d_L* d )
     5.07075274E-02    2     2000001        -1   # BR(~g -> ~d_R  db)
     5.07075274E-02    2    -2000001         1   # BR(~g -> ~d_R* d )
     2.89787767E-02    2     1000002        -2   # BR(~g -> ~u_L  ub)
     2.89787767E-02    2    -1000002         2   # BR(~g -> ~u_L* u )
     4.46872773E-02    2     2000002        -2   # BR(~g -> ~u_R  ub)
     4.46872773E-02    2    -2000002         2   # BR(~g -> ~u_R* u )
     2.08454202E-02    2     1000003        -3   # BR(~g -> ~s_L  sb)
     2.08454202E-02    2    -1000003         3   # BR(~g -> ~s_L* s )
     5.07075274E-02    2     2000003        -3   # BR(~g -> ~s_R  sb)
     5.07075274E-02    2    -2000003         3   # BR(~g -> ~s_R* s )
     2.89787767E-02    2     1000004        -4   # BR(~g -> ~c_L  cb)
     2.89787767E-02    2    -1000004         4   # BR(~g -> ~c_L* c )
     4.46872773E-02    2     2000004        -4   # BR(~g -> ~c_R  cb)
     4.46872773E-02    2    -2000004         4   # BR(~g -> ~c_R* c )
     1.05840237E-01    2     1000005        -5   # BR(~g -> ~b_1  bb)
     1.05840237E-01    2    -1000005         5   # BR(~g -> ~b_1* b )
     5.56574805E-02    2     2000005        -5   # BR(~g -> ~b_2  bb)
     5.56574805E-02    2    -2000005         5   # BR(~g -> ~b_2* b )
     4.80642793E-02    2     1000006        -6   # BR(~g -> ~t_1  tb)
     4.80642793E-02    2    -1000006         6   # BR(~g -> ~t_1* t )
     0.00000000E+00    2     2000006        -6   # BR(~g -> ~t_2  tb)
     0.00000000E+00    2    -2000006         6   # BR(~g -> ~t_2* t )
#
#         PDG            Width
DECAY   1000006     0.0E+00   # stop1 decays
#          BR         NDA      ID1       ID2
     1.92947616E-01    2     1000022         6   # BR(~t_1 -> ~chi_10 t )
     1.17469211E-01    2     1000023         6   # BR(~t_1 -> ~chi_20 t )
     0.00000000E+00    2     1000025         6   # BR(~t_1 -> ~chi_30 t )
     0.00000000E+00    2     1000035         6   # BR(~t_1 -> ~chi_40 t )
     6.75747693E-01    2     1000024         5   # BR(~t_1 -> ~chi_1+ b )
     1.38354802E-02    2     1000037         5   # BR(~t_1 -> ~chi_2+ b )
     0.00000000E+00    2     1000021         6   # BR(~t_1 -> ~g      t )
     0.00000000E+00    2     1000005        37   # BR(~t_1 -> ~b_1    H+)
     0.00000000E+00    2     2000005        37   # BR(~t_1 -> ~b_2    H+)
     0.00000000E+00    2     1000005        24   # BR(~t_1 -> ~b_1    W+)
     0.00000000E+00    2     2000005        24   # BR(~t_1 -> ~b_2    W+)
#
#         PDG            Width
DECAY   2000006     7.37313275E+00   # stop2 decays
#          BR         NDA      ID1       ID2
     2.96825635E-02    2     1000022         6   # BR(~t_2 -> ~chi_10 t )
     8.68035358E-02    2     1000023         6   # BR(~t_2 -> ~chi_20 t )
     4.18408351E-02    2     1000025         6   # BR(~t_2 -> ~chi_30 t )
     1.93281647E-01    2     1000035         6   # BR(~t_2 -> ~chi_40 t )
     2.19632356E-01    2     1000024         5   # BR(~t_2 -> ~chi_1+ b )
     2.02206148E-01    2     1000037         5   # BR(~t_2 -> ~chi_2+ b )
     0.00000000E+00    2     1000021         6   # BR(~t_2 -> ~g      t )
     3.66397706E-02    2     1000006        25   # BR(~t_2 -> ~t_1    h )
     0.00000000E+00    2     1000006        35   # BR(~t_2 -> ~t_1    H )
     0.00000000E+00    2     1000006        36   # BR(~t_2 -> ~t_1    A )
     0.00000000E+00    2     1000005        37   # BR(~t_2 -> ~b_1    H+)
     0.00000000E+00    2     2000005        37   # BR(~t_2 -> ~b_2    H+)
     1.89913144E-01    2     1000006        23   # BR(~t_2 -> ~t_1    Z )
     0.00000000E+00    2     1000005        24   # BR(~t_2 -> ~b_1    W+)
     0.00000000E+00    2     2000005        24   # BR(~t_2 -> ~b_2    W+)
#
#         PDG            Width
DECAY   1000005     3.73627601E+00   # sbottom1 decays
#          BR         NDA      ID1       ID2
     4.43307074E-02    2     1000022         5   # BR(~b_1 -> ~chi_10 b )
     3.56319904E-01    2     1000023         5   # BR(~b_1 -> ~chi_20 b )
     5.16083795E-03    2     1000025         5   # BR(~b_1 -> ~chi_30 b )
     1.04105080E-02    2     1000035         5   # BR(~b_1 -> ~chi_40 b )
     4.45830064E-01    2    -1000024         6   # BR(~b_1 -> ~chi_1- t )
     0.00000000E+00    2    -1000037         6   # BR(~b_1 -> ~chi_2- t )
     0.00000000E+00    2     1000021         5   # BR(~b_1 -> ~g      b )
     0.00000000E+00    2     1000006       -37   # BR(~b_1 -> ~t_1    H-)
     0.00000000E+00    2     2000006       -37   # BR(~b_1 -> ~t_2    H-)
     1.37947979E-01    2     1000006       -24   # BR(~b_1 -> ~t_1    W-)
     0.00000000E+00    2     2000006       -24   # BR(~b_1 -> ~t_2    W-)
#
#         PDG            Width
DECAY   2000005     8.01566294E-01   # sbottom2 decays
#          BR         NDA      ID1       ID2
     2.86200590E-01    2     1000022         5   # BR(~b_2 -> ~chi_10 b )
     1.40315912E-01    2     1000023         5   # BR(~b_2 -> ~chi_20 b )
     5.32635592E-02    2     1000025         5   # BR(~b_2 -> ~chi_30 b )
     7.48748121E-02    2     1000035         5   # BR(~b_2 -> ~chi_40 b )
     1.79734294E-01    2    -1000024         6   # BR(~b_2 -> ~chi_1- t )
     0.00000000E+00    2    -1000037         6   # BR(~b_2 -> ~chi_2- t )
     0.00000000E+00    2     1000021         5   # BR(~b_2 -> ~g      b )
     0.00000000E+00    2     1000005        25   # BR(~b_2 -> ~b_1    h )
     0.00000000E+00    2     1000005        35   # BR(~b_2 -> ~b_1    H )
     0.00000000E+00    2     1000005        36   # BR(~b_2 -> ~b_1    A )
     0.00000000E+00    2     1000006       -37   # BR(~b_2 -> ~t_1    H-)
     0.00000000E+00    2     2000006       -37   # BR(~b_2 -> ~t_2    H-)
     0.00000000E+00    2     1000005        23   # BR(~b_2 -> ~b_1    Z )
     2.65610832E-01    2     1000006       -24   # BR(~b_2 -> ~t_1    W-)
     0.00000000E+00    2     2000006       -24   # BR(~b_2 -> ~t_2    W-)
#
#         PDG            Width
DECAY   1000002     5.47719539E+00   # sup_L decays
#          BR         NDA      ID1       ID2
     6.65240987E-03    2     1000022         2   # BR(~u_L -> ~chi_10 u)
     3.19051458E-01    2     1000023         2   # BR(~u_L -> ~chi_20 u)
     8.44929059E-04    2     1000025         2   # BR(~u_L -> ~chi_30 u)
     1.03485173E-02    2     1000035         2   # BR(~u_L -> ~chi_40 u)
     6.49499518E-01    2     1000024         1   # BR(~u_L -> ~chi_1+ d)
     1.36031676E-02    2     1000037         1   # BR(~u_L -> ~chi_2+ d)
     0.00000000E+00    2     1000021         2   # BR(~u_L -> ~g      u)
#
#         PDG            Width
DECAY   2000002     1.15297292E+00   # sup_R decays
#          BR         NDA      ID1       ID2
     9.86377420E-01    2     1000022         2   # BR(~u_R -> ~chi_10 u)
     8.46640647E-03    2     1000023         2   # BR(~u_R -> ~chi_20 u)
     1.23894695E-03    2     1000025         2   # BR(~u_R -> ~chi_30 u)
     3.91722611E-03    2     1000035         2   # BR(~u_R -> ~chi_40 u)
     0.00000000E+00    2     1000024         1   # BR(~u_R -> ~chi_1+ d)
     0.00000000E+00    2     1000037         1   # BR(~u_R -> ~chi_2+ d)
     0.00000000E+00    2     1000021         2   # BR(~u_R -> ~g      u)
#
#         PDG            Width
DECAY   1000001     5.31278772E+00   # sdown_L decays
#          BR         NDA      ID1       ID2
     2.32317969E-02    2     1000022         1   # BR(~d_L -> ~chi_10 d)
     3.10235077E-01    2     1000023         1   # BR(~d_L -> ~chi_20 d)
     1.52334771E-03    2     1000025         1   # BR(~d_L -> ~chi_30 d)
     1.48849798E-02    2     1000035         1   # BR(~d_L -> ~chi_40 d)
     6.06452481E-01    2    -1000024         2   # BR(~d_L -> ~chi_1- u)
     4.36723179E-02    2    -1000037         2   # BR(~d_L -> ~chi_2- u)
     0.00000000E+00    2     1000021         1   # BR(~d_L -> ~g      d)
#
#         PDG            Width
DECAY   2000001     2.85812308E-01   # sdown_R decays
#          BR         NDA      ID1       ID2
     9.86529614E-01    2     1000022         1   # BR(~d_R -> ~chi_10 d)
     8.44510350E-03    2     1000023         1   # BR(~d_R -> ~chi_20 d)
     1.21172119E-03    2     1000025         1   # BR(~d_R -> ~chi_30 d)
     3.81356102E-03    2     1000035         1   # BR(~d_R -> ~chi_40 d)
     0.00000000E+00    2    -1000024         2   # BR(~d_R -> ~chi_1- u)
     0.00000000E+00    2    -1000037         2   # BR(~d_R -> ~chi_2- u)
     0.00000000E+00    2     1000021         1   # BR(~d_R -> ~g      d)
#
#         PDG            Width
DECAY   1000004     5.47719539E+00   # scharm_L decays
#          BR         NDA      ID1       ID2
     6.65240987E-03    2     1000022         4   # BR(~c_L -> ~chi_10 c)
     3.19051458E-01    2     1000023         4   # BR(~c_L -> ~chi_20 c)
     8.44929059E-04    2     1000025         4   # BR(~c_L -> ~chi_30 c)
     1.03485173E-02    2     1000035         4   # BR(~c_L -> ~chi_40 c)
     6.49499518E-01    2     1000024         3   # BR(~c_L -> ~chi_1+ s)
     1.36031676E-02    2     1000037         3   # BR(~c_L -> ~chi_2+ s)
     0.00000000E+00    2     1000021         4   # BR(~c_L -> ~g      c)
#
#         PDG            Width
DECAY   2000004     1.15297292E+00   # scharm_R decays
#          BR         NDA      ID1       ID2
     9.86377420E-01    2     1000022         4   # BR(~c_R -> ~chi_10 c)
     8.46640647E-03    2     1000023         4   # BR(~c_R -> ~chi_20 c)
     1.23894695E-03    2     1000025         4   # BR(~c_R -> ~chi_30 c)
     3.91722611E-03    2     1000035         4   # BR(~c_R -> ~chi_40 c)
     0.00000000E+00    2     1000024         3   # BR(~c_R -> ~chi_1+ s)
     0.00000000E+00    2     1000037         3   # BR(~c_R -> ~chi_2+ s)
     0.00000000E+00    2     1000021         4   # BR(~c_R -> ~g      c)
#
#         PDG            Width
DECAY   1000003     5.31278772E+00   # sstrange_L decays
#          BR         NDA      ID1       ID2
     2.32317969E-02    2     1000022         3   # BR(~s_L -> ~chi_10 s)
     3.10235077E-01    2     1000023         3   # BR(~s_L -> ~chi_20 s)
     1.52334771E-03    2     1000025         3   # BR(~s_L -> ~chi_30 s)
     1.48849798E-02    2     1000035         3   # BR(~s_L -> ~chi_40 s)
     6.06452481E-01    2    -1000024         4   # BR(~s_L -> ~chi_1- c)
     4.36723179E-02    2    -1000037         4   # BR(~s_L -> ~chi_2- c)
     0.00000000E+00    2     1000021         3   # BR(~s_L -> ~g      s)
#
#         PDG            Width
DECAY   2000003     2.85812308E-01   # sstrange_R decays
#          BR         NDA      ID1       ID2
     9.86529614E-01    2     1000022         3   # BR(~s_R -> ~chi_10 s)
     8.44510350E-03    2     1000023         3   # BR(~s_R -> ~chi_20 s)
     1.21172119E-03    2     1000025         3   # BR(~s_R -> ~chi_30 s)
     3.81356102E-03    2     1000035         3   # BR(~s_R -> ~chi_40 s)
     0.00000000E+00    2    -1000024         4   # BR(~s_R -> ~chi_1- c)
     0.00000000E+00    2    -1000037         4   # BR(~s_R -> ~chi_2- c)
     0.00000000E+00    2     1000021         3   # BR(~s_R -> ~g      s)
#
#         PDG            Width
DECAY   1000011     2.13682161E-01   # selectron_L decays
#          BR         NDA      ID1       ID2
     5.73155386E-01    2     1000022        11   # BR(~e_L -> ~chi_10 e-)
     1.64522579E-01    2     1000023        11   # BR(~e_L -> ~chi_20 e-)
     0.00000000E+00    2     1000025        11   # BR(~e_L -> ~chi_30 e-)
     0.00000000E+00    2     1000035        11   # BR(~e_L -> ~chi_40 e-)
     2.62322035E-01    2    -1000024        12   # BR(~e_L -> ~chi_1- nu_e)
     0.00000000E+00    2    -1000037        12   # BR(~e_L -> ~chi_2- nu_e)
#
#         PDG            Width
DECAY   2000011     2.16121626E-01   # selectron_R decays
#          BR         NDA      ID1       ID2
     1.00000000E+00    2     1000022        11   # BR(~e_R -> ~chi_10 e-)
     0.00000000E+00    2     1000023        11   # BR(~e_R -> ~chi_20 e-)
     0.00000000E+00    2     1000025        11   # BR(~e_R -> ~chi_30 e-)
     0.00000000E+00    2     1000035        11   # BR(~e_R -> ~chi_40 e-)
     0.00000000E+00    2    -1000024        12   # BR(~e_R -> ~chi_1- nu_e)
     0.00000000E+00    2    -1000037        12   # BR(~e_R -> ~chi_2- nu_e)
#
#         PDG            Width
DECAY   1000013     2.13682161E-01   # smuon_L decays
#          BR         NDA      ID1       ID2
     5.73155386E-01    2     1000022        13   # BR(~mu_L -> ~chi_10 mu-)
     1.64522579E-01    2     1000023        13   # BR(~mu_L -> ~chi_20 mu-)
     0.00000000E+00    2     1000025        13   # BR(~mu_L -> ~chi_30 mu-)
     0.00000000E+00    2     1000035        13   # BR(~mu_L -> ~chi_40 mu-)
     2.62322035E-01    2    -1000024        14   # BR(~mu_L -> ~chi_1- nu_mu)
     0.00000000E+00    2    -1000037        14   # BR(~mu_L -> ~chi_2- nu_mu)
#
#         PDG            Width
DECAY   2000013     2.16121626E-01   # smuon_R decays
#          BR         NDA      ID1       ID2
     1.00000000E+00    2     1000022        13   # BR(~mu_R -> ~chi_10 mu-)
     0.00000000E+00    2     1000023        13   # BR(~mu_R -> ~chi_20 mu-)
     0.00000000E+00    2     1000025        13   # BR(~mu_R -> ~chi_30 mu-)
     0.00000000E+00    2     1000035        13   # BR(~mu_R -> ~chi_40 mu-)
     0.00000000E+00    2    -1000024        14   # BR(~mu_R -> ~chi_1- nu_mu)
     0.00000000E+00    2    -1000037        14   # BR(~mu_R -> ~chi_2- nu_mu)
#
#         PDG            Width
DECAY   1000015     1.48327268E-01   # stau_1 decays
#          BR         NDA      ID1       ID2
     1.00000000E+00    2     1000022        15   # BR(~tau_1 -> ~chi_10  tau-)
     0.00000000E+00    2     1000023        15   # BR(~tau_1 -> ~chi_20  tau-)
     0.00000000E+00    2     1000025        15   # BR(~tau_1 -> ~chi_30  tau-)
     0.00000000E+00    2     1000035        15   # BR(~tau_1 -> ~chi_40  tau-)
     0.00000000E+00    2    -1000024        16   # BR(~tau_1 -> ~chi_1-  nu_tau)
     0.00000000E+00    2    -1000037        16   # BR(~tau_1 -> ~chi_2-  nu_tau)
     0.00000000E+00    2     1000016       -37   # BR(~tau_1 -> ~nu_tauL H-)
     0.00000000E+00    2     1000016       -24   # BR(~tau_1 -> ~nu_tauL W-)
#
#         PDG            Width
DECAY   2000015     2.69906096E-01   # stau_2 decays
#          BR         NDA      ID1       ID2
     5.96653046E-01    2     1000022        15   # BR(~tau_2 -> ~chi_10  tau-)
     1.54536760E-01    2     1000023        15   # BR(~tau_2 -> ~chi_20  tau-)
     0.00000000E+00    2     1000025        15   # BR(~tau_2 -> ~chi_30  tau-)
     0.00000000E+00    2     1000035        15   # BR(~tau_2 -> ~chi_40  tau-)
     2.48810195E-01    2    -1000024        16   # BR(~tau_2 -> ~chi_1-  nu_tau)
     0.00000000E+00    2    -1000037        16   # BR(~tau_2 -> ~chi_2-  nu_tau)
     0.00000000E+00    2     1000016       -37   # BR(~tau_2 -> ~nu_tauL H-)
     0.00000000E+00    2     1000016       -24   # BR(~tau_2 -> ~nu_tauL W-)
     0.00000000E+00    2     1000015        25   # BR(~tau_2 -> ~tau_1 h)
     0.00000000E+00    2     1000015        35   # BR(~tau_2 -> ~tau_1 H)
     0.00000000E+00    2     1000015        36   # BR(~tau_2 -> ~tau_1 A)
     0.00000000E+00    2     1000015        23   # BR(~tau_2 -> ~tau_1 Z)
#
#         PDG            Width
DECAY   1000012     1.49881634E-01   # snu_eL decays
#          BR         NDA      ID1       ID2
     9.77700764E-01    2     1000022        12   # BR(~nu_eL -> ~chi_10 nu_e)
     8.11554922E-03    2     1000023        12   # BR(~nu_eL -> ~chi_20 nu_e)
     0.00000000E+00    2     1000025        12   # BR(~nu_eL -> ~chi_30 nu_e)
     0.00000000E+00    2     1000035        12   # BR(~nu_eL -> ~chi_40 nu_e)
     1.41836867E-02    2     1000024        11   # BR(~nu_eL -> ~chi_1+ e-)
     0.00000000E+00    2     1000037        11   # BR(~nu_eL -> ~chi_2+ e-)
#
#         PDG            Width
DECAY   1000014     1.49881634E-01   # snu_muL decays
#          BR         NDA      ID1       ID2
     9.77700764E-01    2     1000022        14   # BR(~nu_muL -> ~chi_10 nu_mu)
     8.11554922E-03    2     1000023        14   # BR(~nu_muL -> ~chi_20 nu_mu)
     0.00000000E+00    2     1000025        14   # BR(~nu_muL -> ~chi_30 nu_mu)
     0.00000000E+00    2     1000035        14   # BR(~nu_muL -> ~chi_40 nu_mu)
     1.41836867E-02    2     1000024        13   # BR(~nu_muL -> ~chi_1+ mu-)
     0.00000000E+00    2     1000037        13   # BR(~nu_muL -> ~chi_2+ mu-)
#
#         PDG            Width
DECAY   1000016     1.47518977E-01   # snu_tauL decays
#          BR         NDA      ID1       ID2
     9.85994529E-01    2     1000022        16   # BR(~nu_tauL -> ~chi_10 nu_tau)
     6.25129612E-03    2     1000023        16   # BR(~nu_tauL -> ~chi_20 nu_tau)
     0.00000000E+00    2     1000025        16   # BR(~nu_tauL -> ~chi_30 nu_tau)
     0.00000000E+00    2     1000035        16   # BR(~nu_tauL -> ~chi_40 nu_tau)
     7.75417479E-03    2     1000024        15   # BR(~nu_tauL -> ~chi_1+ tau-)
     0.00000000E+00    2     1000037        15   # BR(~nu_tauL -> ~chi_2+ tau-)
     0.00000000E+00    2    -1000015       -37   # BR(~nu_tauL -> ~tau_1+ H-)
     0.00000000E+00    2    -2000015       -37   # BR(~nu_tauL -> ~tau_2+ H-)
     0.00000000E+00    2    -1000015       -24   # BR(~nu_tauL -> ~tau_1+ W-)
     0.00000000E+00    2    -2000015       -24   # BR(~nu_tauL -> ~tau_2+ W-)
#
#         PDG            Width
DECAY   1000024     1.70414503E-02   # chargino1+ decays
#          BR         NDA      ID1       ID2
     0.00000000E+00    2     1000002        -1   # BR(~chi_1+ -> ~u_L   db)
     0.00000000E+00    2     2000002        -1   # BR(~chi_1+ -> ~u_R   db)
     0.00000000E+00    2    -1000001         2   # BR(~chi_1+ -> ~d_L*  u )
     0.00000000E+00    2    -2000001         2   # BR(~chi_1+ -> ~d_R*  u )
     0.00000000E+00    2     1000004        -3   # BR(~chi_1+ -> ~c_L   sb)
     0.00000000E+00    2     2000004        -3   # BR(~chi_1+ -> ~c_R   sb)
     0.00000000E+00    2    -1000003         4   # BR(~chi_1+ -> ~s_L*  c )
     0.00000000E+00    2    -2000003         4   # BR(~chi_1+ -> ~s_R*  c )
     0.00000000E+00    2     1000006        -5   # BR(~chi_1+ -> ~t_1   bb)
     0.00000000E+00    2     2000006        -5   # BR(~chi_1+ -> ~t_2   bb)
     0.00000000E+00    2    -1000005         6   # BR(~chi_1+ -> ~b_1*  t )
     0.00000000E+00    2    -2000005         6   # BR(~chi_1+ -> ~b_2*  t )
     0.00000000E+00    2     1000012       -11   # BR(~chi_1+ -> ~nu_eL  e+  )
     0.00000000E+00    2     1000014       -13   # BR(~chi_1+ -> ~nu_muL  mu+ )
     0.00000000E+00    2     1000016       -15   # BR(~chi_1+ -> ~nu_tau1 tau+)
     0.00000000E+00    2    -1000011        12   # BR(~chi_1+ -> ~e_L+    nu_e)
     0.00000000E+00    2    -2000011        12   # BR(~chi_1+ -> ~e_R+    nu_e)
     0.00000000E+00    2    -1000013        14   # BR(~chi_1+ -> ~mu_L+   nu_mu)
     0.00000000E+00    2    -2000013        14   # BR(~chi_1+ -> ~mu_R+   nu_mu)
     9.25161117E-01    2    -1000015        16   # BR(~chi_1+ -> ~tau_1+  nu_tau)
     0.00000000E+00    2    -2000015        16   # BR(~chi_1+ -> ~tau_2+  nu_tau)
     7.48388828E-02    2     1000022        24   # BR(~chi_1+ -> ~chi_10  W+)
     0.00000000E+00    2     1000023        24   # BR(~chi_1+ -> ~chi_20  W+)
     0.00000000E+00    2     1000025        24   # BR(~chi_1+ -> ~chi_30  W+)
     0.00000000E+00    2     1000035        24   # BR(~chi_1+ -> ~chi_40  W+)
     0.00000000E+00    2     1000022        37   # BR(~chi_1+ -> ~chi_10  H+)
     0.00000000E+00    2     1000023        37   # BR(~chi_1+ -> ~chi_20  H+)
     0.00000000E+00    2     1000025        37   # BR(~chi_1+ -> ~chi_30  H+)
     0.00000000E+00    2     1000035        37   # BR(~chi_1+ -> ~chi_40  H+)
#
#         PDG            Width
DECAY   1000037     2.48689510E+00   # chargino2+ decays
#          BR         NDA      ID1       ID2
     0.00000000E+00    2     1000002        -1   # BR(~chi_2+ -> ~u_L   db)
     0.00000000E+00    2     2000002        -1   # BR(~chi_2+ -> ~u_R   db)
     0.00000000E+00    2    -1000001         2   # BR(~chi_2+ -> ~d_L*  u )
     0.00000000E+00    2    -2000001         2   # BR(~chi_2+ -> ~d_R*  u )
     0.00000000E+00    2     1000004        -3   # BR(~chi_2+ -> ~c_L   sb)
     0.00000000E+00    2     2000004        -3   # BR(~chi_2+ -> ~c_R   sb)
     0.00000000E+00    2    -1000003         4   # BR(~chi_2+ -> ~s_L*  c )
     0.00000000E+00    2    -2000003         4   # BR(~chi_2+ -> ~s_R*  c )
     0.00000000E+00    2     1000006        -5   # BR(~chi_2+ -> ~t_1   bb)
     0.00000000E+00    2     2000006        -5   # BR(~chi_2+ -> ~t_2   bb)
     0.00000000E+00    2    -1000005         6   # BR(~chi_2+ -> ~b_1*  t )
     0.00000000E+00    2    -2000005         6   # BR(~chi_2+ -> ~b_2*  t )
     2.00968837E-02    2     1000012       -11   # BR(~chi_2+ -> ~nu_eL  e+  )
     2.00968837E-02    2     1000014       -13   # BR(~chi_2+ -> ~nu_muL  mu+ )
     2.74507395E-02    2     1000016       -15   # BR(~chi_2+ -> ~nu_tau1 tau+)
     5.20406111E-02    2    -1000011        12   # BR(~chi_2+ -> ~e_L+    nu_e)
     0.00000000E+00    2    -2000011        12   # BR(~chi_2+ -> ~e_R+    nu_e)
     5.20406111E-02    2    -1000013        14   # BR(~chi_2+ -> ~mu_L+   nu_mu)
     0.00000000E+00    2    -2000013        14   # BR(~chi_2+ -> ~mu_R+   nu_mu)
     2.82859898E-04    2    -1000015        16   # BR(~chi_2+ -> ~tau_1+  nu_tau)
     5.66729336E-02    2    -2000015        16   # BR(~chi_2+ -> ~tau_2+  nu_tau)
     2.31513269E-01    2     1000024        23   # BR(~chi_2+ -> ~chi_1+  Z )
     6.76715120E-02    2     1000022        24   # BR(~chi_2+ -> ~chi_10  W+)
     2.93654849E-01    2     1000023        24   # BR(~chi_2+ -> ~chi_20  W+)
     0.00000000E+00    2     1000025        24   # BR(~chi_2+ -> ~chi_30  W+)
     0.00000000E+00    2     1000035        24   # BR(~chi_2+ -> ~chi_40  W+)
     1.78478848E-01    2     1000024        25   # BR(~chi_2+ -> ~chi_1+  h )
     0.00000000E+00    2     1000024        35   # BR(~chi_2+ -> ~chi_1+  H )
     0.00000000E+00    2     1000024        36   # BR(~chi_2+ -> ~chi_1+  A )
     0.00000000E+00    2     1000022        37   # BR(~chi_2+ -> ~chi_10  H+)
     0.00000000E+00    2     1000023        37   # BR(~chi_2+ -> ~chi_20  H+)
     0.00000000E+00    2     1000025        37   # BR(~chi_2+ -> ~chi_30  H+)
     0.00000000E+00    2     1000035        37   # BR(~chi_2+ -> ~chi_40  H+)
#
#         PDG            Width
DECAY   1000022     0.00000000E+00   # neutralino1 decays
#
#         PDG            Width
DECAY   1000023     2.07770048E-02   # neutralino2 decays
#          BR         NDA      ID1       ID2
     0.00000000E+00    2     1000022        23   # BR(~chi_20 -> ~chi_10   Z )
     0.00000000E+00    2     1000024       -24   # BR(~chi_20 -> ~chi_1+   W-)
     0.00000000E+00    2    -1000024        24   # BR(~chi_20 -> ~chi_1-   W+)
     0.00000000E+00    2     1000037       -24   # BR(~chi_20 -> ~chi_2+   W-)
     0.00000000E+00    2    -1000037        24   # BR(~chi_20 -> ~chi_2-   W+)
     0.00000000E+00    2     1000022        25   # BR(~chi_20 -> ~chi_10   h )
     0.00000000E+00    2     1000022        35   # BR(~chi_20 -> ~chi_10   H )
     0.00000000E+00    2     1000022        36   # BR(~chi_20 -> ~chi_10   A )
     0.00000000E+00    2     1000024       -37   # BR(~chi_20 -> ~chi_1+   H-)
     0.00000000E+00    2    -1000024        37   # BR(~chi_20 -> ~chi_1-   H+)
     0.00000000E+00    2     1000037       -37   # BR(~chi_20 -> ~chi_2+   H-)
     0.00000000E+00    2    -1000037        37   # BR(~chi_20 -> ~chi_2-   H+)
     0.00000000E+00    2     1000002        -2   # BR(~chi_20 -> ~u_L      ub)
     0.00000000E+00    2    -1000002         2   # BR(~chi_20 -> ~u_L*     u )
     0.00000000E+00    2     2000002        -2   # BR(~chi_20 -> ~u_R      ub)
     0.00000000E+00    2    -2000002         2   # BR(~chi_20 -> ~u_R*     u )
     0.00000000E+00    2     1000001        -1   # BR(~chi_20 -> ~d_L      db)
     0.00000000E+00    2    -1000001         1   # BR(~chi_20 -> ~d_L*     d )
     0.00000000E+00    2     2000001        -1   # BR(~chi_20 -> ~d_R      db)
     0.00000000E+00    2    -2000001         1   # BR(~chi_20 -> ~d_R*     d )
     0.00000000E+00    2     1000004        -4   # BR(~chi_20 -> ~c_L      cb)
     0.00000000E+00    2    -1000004         4   # BR(~chi_20 -> ~c_L*     c )
     0.00000000E+00    2     2000004        -4   # BR(~chi_20 -> ~c_R      cb)
     0.00000000E+00    2    -2000004         4   # BR(~chi_20 -> ~c_R*     c )
     0.00000000E+00    2     1000003        -3   # BR(~chi_20 -> ~s_L      sb)
     0.00000000E+00    2    -1000003         3   # BR(~chi_20 -> ~s_L*     s )
     0.00000000E+00    2     2000003        -3   # BR(~chi_20 -> ~s_R      sb)
     0.00000000E+00    2    -2000003         3   # BR(~chi_20 -> ~s_R*     s )
     0.00000000E+00    2     1000006        -6   # BR(~chi_20 -> ~t_1      tb)
     0.00000000E+00    2    -1000006         6   # BR(~chi_20 -> ~t_1*     t )
     0.00000000E+00    2     2000006        -6   # BR(~chi_20 -> ~t_2      tb)
     0.00000000E+00    2    -2000006         6   # BR(~chi_20 -> ~t_2*     t )
     0.00000000E+00    2     1000005        -5   # BR(~chi_20 -> ~b_1      bb)
     0.00000000E+00    2    -1000005         5   # BR(~chi_20 -> ~b_1*     b )
     0.00000000E+00    2     2000005        -5   # BR(~chi_20 -> ~b_2      bb)
     0.00000000E+00    2    -2000005         5   # BR(~chi_20 -> ~b_2*     b )
     0.00000000E+00    2     1000011       -11   # BR(~chi_20 -> ~e_L-     e+)
     0.00000000E+00    2    -1000011        11   # BR(~chi_20 -> ~e_L+     e-)
     2.95071995E-02    2     2000011       -11   # BR(~chi_20 -> ~e_R-     e+)
     2.95071995E-02    2    -2000011        11   # BR(~chi_20 -> ~e_R+     e-)
     0.00000000E+00    2     1000013       -13   # BR(~chi_20 -> ~mu_L-    mu+)
     0.00000000E+00    2    -1000013        13   # BR(~chi_20 -> ~mu_L+    mu-)
     2.95071995E-02    2     2000013       -13   # BR(~chi_20 -> ~mu_R-    mu+)
     2.95071995E-02    2    -2000013        13   # BR(~chi_20 -> ~mu_R+    mu-)
     4.40985601E-01    2     1000015       -15   # BR(~chi_20 -> ~tau_1-   tau+)
     4.40985601E-01    2    -1000015        15   # BR(~chi_20 -> ~tau_1+   tau-)
     0.00000000E+00    2     2000015       -15   # BR(~chi_20 -> ~tau_2-   tau+)
     0.00000000E+00    2    -2000015        15   # BR(~chi_20 -> ~tau_2+   tau-)
     0.00000000E+00    2     1000012       -12   # BR(~chi_20 -> ~nu_eL    nu_eb)
     0.00000000E+00    2    -1000012        12   # BR(~chi_20 -> ~nu_eL*   nu_e )
     0.00000000E+00    2     1000014       -14   # BR(~chi_20 -> ~nu_muL   nu_mub)
     0.00000000E+00    2    -1000014        14   # BR(~chi_20 -> ~nu_muL*  nu_mu )
     0.00000000E+00    2     1000016       -16   # BR(~chi_20 -> ~nu_tau1  nu_taub)
     0.00000000E+00    2    -1000016        16   # BR(~chi_20 -> ~nu_tau1* nu_tau )
#
#         PDG            Width
DECAY   1000025     1.91598495E+00   # neutralino3 decays
#          BR         NDA      ID1       ID2
     1.13226601E-01    2     1000022        23   # BR(~chi_30 -> ~chi_10   Z )
     2.11969194E-01    2     1000023        23   # BR(~chi_30 -> ~chi_20   Z )
     2.95329778E-01    2     1000024       -24   # BR(~chi_30 -> ~chi_1+   W-)
     2.95329778E-01    2    -1000024        24   # BR(~chi_30 -> ~chi_1-   W+)
     0.00000000E+00    2     1000037       -24   # BR(~chi_30 -> ~chi_2+   W-)
     0.00000000E+00    2    -1000037        24   # BR(~chi_30 -> ~chi_2-   W+)
     2.13076490E-02    2     1000022        25   # BR(~chi_30 -> ~chi_10   h )
     0.00000000E+00    2     1000022        35   # BR(~chi_30 -> ~chi_10   H )
     0.00000000E+00    2     1000022        36   # BR(~chi_30 -> ~chi_10   A )
     1.24538329E-02    2     1000023        25   # BR(~chi_30 -> ~chi_20   h )
     0.00000000E+00    2     1000023        35   # BR(~chi_30 -> ~chi_20   H )
     0.00000000E+00    2     1000023        36   # BR(~chi_30 -> ~chi_20   A )
     0.00000000E+00    2     1000024       -37   # BR(~chi_30 -> ~chi_1+   H-)
     0.00000000E+00    2    -1000024        37   # BR(~chi_30 -> ~chi_1-   H+)
     0.00000000E+00    2     1000037       -37   # BR(~chi_30 -> ~chi_2+   H-)
     0.00000000E+00    2    -1000037        37   # BR(~chi_30 -> ~chi_2-   H+)
     0.00000000E+00    2     1000002        -2   # BR(~chi_30 -> ~u_L      ub)
     0.00000000E+00    2    -1000002         2   # BR(~chi_30 -> ~u_L*     u )
     0.00000000E+00    2     2000002        -2   # BR(~chi_30 -> ~u_R      ub)
     0.00000000E+00    2    -2000002         2   # BR(~chi_30 -> ~u_R*     u )
     0.00000000E+00    2     1000001        -1   # BR(~chi_30 -> ~d_L      db)
     0.00000000E+00    2    -1000001         1   # BR(~chi_30 -> ~d_L*     d )
     0.00000000E+00    2     2000001        -1   # BR(~chi_30 -> ~d_R      db)
     0.00000000E+00    2    -2000001         1   # BR(~chi_30 -> ~d_R*     d )
     0.00000000E+00    2     1000004        -4   # BR(~chi_30 -> ~c_L      cb)
     0.00000000E+00    2    -1000004         4   # BR(~chi_30 -> ~c_L*     c )
     0.00000000E+00    2     2000004        -4   # BR(~chi_30 -> ~c_R      cb)
     0.00000000E+00    2    -2000004         4   # BR(~chi_30 -> ~c_R*     c )
     0.00000000E+00    2     1000003        -3   # BR(~chi_30 -> ~s_L      sb)
     0.00000000E+00    2    -1000003         3   # BR(~chi_30 -> ~s_L*     s )
     0.00000000E+00    2     2000003        -3   # BR(~chi_30 -> ~s_R      sb)
     0.00000000E+00    2    -2000003         3   # BR(~chi_30 -> ~s_R*     s )
     0.00000000E+00    2     1000006        -6   # BR(~chi_30 -> ~t_1      tb)
     0.00000000E+00    2    -1000006         6   # BR(~chi_30 -> ~t_1*     t )
     0.00000000E+00    2     2000006        -6   # BR(~chi_30 -> ~t_2      tb)
     0.00000000E+00    2    -2000006         6   # BR(~chi_30 -> ~t_2*     t )
     0.00000000E+00    2     1000005        -5   # BR(~chi_30 -> ~b_1      bb)
     0.00000000E+00    2    -1000005         5   # BR(~chi_30 -> ~b_1*     b )
     0.00000000E+00    2     2000005        -5   # BR(~chi_30 -> ~b_2      bb)
     0.00000000E+00    2    -2000005         5   # BR(~chi_30 -> ~b_2*     b )
     5.57220455E-04    2     1000011       -11   # BR(~chi_30 -> ~e_L-     e+)
     5.57220455E-04    2    -1000011        11   # BR(~chi_30 -> ~e_L+     e-)
     1.25266782E-03    2     2000011       -11   # BR(~chi_30 -> ~e_R-     e+)
     1.25266782E-03    2    -2000011        11   # BR(~chi_30 -> ~e_R+     e-)
     5.57220455E-04    2     1000013       -13   # BR(~chi_30 -> ~mu_L-    mu+)
     5.57220455E-04    2    -1000013        13   # BR(~chi_30 -> ~mu_L+    mu-)
     1.25266782E-03    2     2000013       -13   # BR(~chi_30 -> ~mu_R-    mu+)
     1.25266782E-03    2    -2000013        13   # BR(~chi_30 -> ~mu_R+    mu-)
     5.26279239E-03    2     1000015       -15   # BR(~chi_30 -> ~tau_1-   tau+)
     5.26279239E-03    2    -1000015        15   # BR(~chi_30 -> ~tau_1+   tau-)
     6.72814564E-03    2     2000015       -15   # BR(~chi_30 -> ~tau_2-   tau+)
     6.72814564E-03    2    -2000015        15   # BR(~chi_30 -> ~tau_2+   tau-)
     3.18920485E-03    2     1000012       -12   # BR(~chi_30 -> ~nu_eL    nu_eb)
     3.18920485E-03    2    -1000012        12   # BR(~chi_30 -> ~nu_eL*   nu_e )
     3.18920485E-03    2     1000014       -14   # BR(~chi_30 -> ~nu_muL   nu_mub)
     3.18920485E-03    2    -1000014        14   # BR(~chi_30 -> ~nu_muL*  nu_mu )
     3.20245934E-03    2     1000016       -16   # BR(~chi_30 -> ~nu_tau1  nu_taub)
     3.20245934E-03    2    -1000016        16   # BR(~chi_30 -> ~nu_tau1* nu_tau )
#
#         PDG            Width
DECAY   1000035     2.58585079E+00   # neutralino4 decays
#          BR         NDA      ID1       ID2
     2.15369294E-02    2     1000022        23   # BR(~chi_40 -> ~chi_10   Z )
     1.85499971E-02    2     1000023        23   # BR(~chi_40 -> ~chi_20   Z )
     0.00000000E+00    2     1000025        23   # BR(~chi_40 -> ~chi_30   Z )
     2.49541430E-01    2     1000024       -24   # BR(~chi_40 -> ~chi_1+   W-)
     2.49541430E-01    2    -1000024        24   # BR(~chi_40 -> ~chi_1-   W+)
     0.00000000E+00    2     1000037       -24   # BR(~chi_40 -> ~chi_2+   W-)
     0.00000000E+00    2    -1000037        24   # BR(~chi_40 -> ~chi_2-   W+)
     6.93213268E-02    2     1000022        25   # BR(~chi_40 -> ~chi_10   h )
     0.00000000E+00    2     1000022        35   # BR(~chi_40 -> ~chi_10   H )
     0.00000000E+00    2     1000022        36   # BR(~chi_40 -> ~chi_10   A )
     1.47602336E-01    2     1000023        25   # BR(~chi_40 -> ~chi_20   h )
     0.00000000E+00    2     1000023        35   # BR(~chi_40 -> ~chi_20   H )
     0.00000000E+00    2     1000023        36   # BR(~chi_40 -> ~chi_20   A )
     0.00000000E+00    2     1000025        25   # BR(~chi_40 -> ~chi_30   h )
     0.00000000E+00    2     1000025        35   # BR(~chi_40 -> ~chi_30   H )
     0.00000000E+00    2     1000025        36   # BR(~chi_40 -> ~chi_30   A )
     0.00000000E+00    2     1000024       -37   # BR(~chi_40 -> ~chi_1+   H-)
     0.00000000E+00    2    -1000024        37   # BR(~chi_40 -> ~chi_1-   H+)
     0.00000000E+00    2     1000037       -37   # BR(~chi_40 -> ~chi_2+   H-)
     0.00000000E+00    2    -1000037        37   # BR(~chi_40 -> ~chi_2-   H+)
     0.00000000E+00    2     1000002        -2   # BR(~chi_40 -> ~u_L      ub)
     0.00000000E+00    2    -1000002         2   # BR(~chi_40 -> ~u_L*     u )
     0.00000000E+00    2     2000002        -2   # BR(~chi_40 -> ~u_R      ub)
     0.00000000E+00    2    -2000002         2   # BR(~chi_40 -> ~u_R*     u )
     0.00000000E+00    2     1000001        -1   # BR(~chi_40 -> ~d_L      db)
     0.00000000E+00    2    -1000001         1   # BR(~chi_40 -> ~d_L*     d )
     0.00000000E+00    2     2000001        -1   # BR(~chi_40 -> ~d_R      db)
     0.00000000E+00    2    -2000001         1   # BR(~chi_40 -> ~d_R*     d )
     0.00000000E+00    2     1000004        -4   # BR(~chi_40 -> ~c_L      cb)
     0.00000000E+00    2    -1000004         4   # BR(~chi_40 -> ~c_L*     c )
     0.00000000E+00    2     2000004        -4   # BR(~chi_40 -> ~c_R      cb)
     0.00000000E+00    2    -2000004         4   # BR(~chi_40 -> ~c_R*     c )
     0.00000000E+00    2     1000003        -3   # BR(~chi_40 -> ~s_L      sb)
     0.00000000E+00    2    -1000003         3   # BR(~chi_40 -> ~s_L*     s )
     0.00000000E+00    2     2000003        -3   # BR(~chi_40 -> ~s_R      sb)
     0.00000000E+00    2    -2000003         3   # BR(~chi_40 -> ~s_R*     s )
     0.00000000E+00    2     1000006        -6   # BR(~chi_40 -> ~t_1      tb)
     0.00000000E+00    2    -1000006         6   # BR(~chi_40 -> ~t_1*     t )
     0.00000000E+00    2     2000006        -6   # BR(~chi_40 -> ~t_2      tb)
     0.00000000E+00    2    -2000006         6   # BR(~chi_40 -> ~t_2*     t )
     0.00000000E+00    2     1000005        -5   # BR(~chi_40 -> ~b_1      bb)
     0.00000000E+00    2    -1000005         5   # BR(~chi_40 -> ~b_1*     b )
     0.00000000E+00    2     2000005        -5   # BR(~chi_40 -> ~b_2      bb)
     0.00000000E+00    2    -2000005         5   # BR(~chi_40 -> ~b_2*     b )
     9.64835418E-03    2     1000011       -11   # BR(~chi_40 -> ~e_L-     e+)
     9.64835418E-03    2    -1000011        11   # BR(~chi_40 -> ~e_L+     e-)
     3.75684470E-03    2     2000011       -11   # BR(~chi_40 -> ~e_R-     e+)
     3.75684470E-03    2    -2000011        11   # BR(~chi_40 -> ~e_R+     e-)
     9.64835418E-03    2     1000013       -13   # BR(~chi_40 -> ~mu_L-    mu+)
     9.64835418E-03    2    -1000013        13   # BR(~chi_40 -> ~mu_L+    mu-)
     3.75684470E-03    2     2000013       -13   # BR(~chi_40 -> ~mu_R-    mu+)
     3.75684470E-03    2    -2000013        13   # BR(~chi_40 -> ~mu_R+    mu-)
     2.68215241E-03    2     1000015       -15   # BR(~chi_40 -> ~tau_1-   tau+)
     2.68215241E-03    2    -1000015        15   # BR(~chi_40 -> ~tau_1+   tau-)
     1.62289809E-02    2     2000015       -15   # BR(~chi_40 -> ~tau_2-   tau+)
     1.62289809E-02    2    -2000015        15   # BR(~chi_40 -> ~tau_2+   tau-)
     2.53796547E-02    2     1000012       -12   # BR(~chi_40 -> ~nu_eL    nu_eb)
     2.53796547E-02    2    -1000012        12   # BR(~chi_40 -> ~nu_eL*   nu_e )
     2.53796547E-02    2     1000014       -14   # BR(~chi_40 -> ~nu_muL   nu_mub)
     2.53796547E-02    2    -1000014        14   # BR(~chi_40 -> ~nu_muL*  nu_mu )
     2.54724352E-02    2     1000016       -16   # BR(~chi_40 -> ~nu_tau1  nu_taub)
     2.54724352E-02    2    -1000016        16   # BR(~chi_40 -> ~nu_tau1* nu_tau )
