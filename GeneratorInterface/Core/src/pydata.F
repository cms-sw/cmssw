C*********************************************************************
C*********************************************************************
C*                                                                  **
C*                                                      June 2010   **
C*                                                                  **
C*                       The Lund Monte Carlo                       **
C*                                                                  **
C*                        PYTHIA version 6.4                        **
C*                                                                  **
C*                        Torbjorn Sjostrand                        **
C*                 Department of Theoretical Physics                **
C*                         Lund University                          **
C*               Solvegatan 14A, S-223 62 Lund, Sweden              **
C*                    E-mail torbjorn@thep.lu.se                    **
C*                                                                  **
C*                  SUSY and Technicolor parts by                   **
C*                         Stephen Mrenna                           **
C*                       Computing Division                         ** 
C*            Generators and Detector Simulation Group              **
C*              Fermi National Accelerator Laboratory               **
C*                 MS 234, Batavia, IL  60510, USA                  **
C*                   phone + 1 - 630 - 840 - 2556                   **
C*                      E-mail mrenna@fnal.gov                      **
C*                                                                  **
C*         New multiple interactions and more SUSY parts by         **
C*                          Peter Skands                            **
C*               CERN/PH, CH-1211 Geneva, Switzerland               **
C*                    phone +41 - 22 - 767 2447                     **
C*                   E-mail peter.skands@cern.ch                    **
C*                                                                  **
C*         Several parts are written by Hans-Uno Bengtsson          **
C*          PYSHOW is written together with Mats Bengtsson          **
C*               PYMAEL is written by Emanuel Norrbin               **
C*     advanced popcorn baryon production written by Patrik Eden    **
C*    code for virtual photons mainly written by Christer Friberg   **
C*    code for low-mass strings mainly written by Emanuel Norrbin   **
C*        Bose-Einstein code mainly written by Leif Lonnblad        **
C*      CTEQ  parton distributions are by the CTEQ collaboration    **
C*      GRV 94 parton distributions are by Glueck, Reya and Vogt    **
C*   SaS photon parton distributions together with Gerhard Schuler  **
C*     g + g and q + qbar -> t + tbar + H code by Zoltan Kunszt     **
C*         MSSM Higgs mass calculation code by M. Carena,           **
C*           J.R. Espinosa, M. Quiros and C.E.M. Wagner             **
C*  UED implementation by M. Elkacimi, D. Goujdami, H. Przysiezniak **
C*         PYGAUS adapted from CERN library (K.S. Kolbig)           **
C*        NRQCD/colour octet production of onium by S. Wolf         **
C*                                                                  **
C*   The latest program version and documentation is found on WWW   **
C*            http://www.thep.lu.se/~torbjorn/Pythia.html           **
C*                                                                  **
C*              Copyright Torbjorn Sjostrand, Lund 2010             **
C*                                                                  **
C*********************************************************************
C*********************************************************************
C                                                                    *
C  List of subprograms in order of appearance, with main purpose     *
C  (S = subroutine, F = function, B = block data)                    *
C                                                                    *
C  B   PYDATA   to contain all default values                        *
C  S   PYCKBD   to check that BLOCK DATA has been correctly loaded   *
C  S   PYTEST   to test the proper functioning of the package        *
C  S   PYHEPC   to convert between /PYJETS/ and /HEPEVT/ records     *
C                                                                    *
C  S   PYINIT   to administer the initialization procedure           *
C  S   PYEVNT   to administer the generation of an event             *
C  S   PYEVNW   ditto, for new multiple interactions scenario        *
C  S   PYSTAT   to print cross-section and other information         *
C  S   PYUPEV   to administer the generation of an LHA hard process  *
C  S   PYUPIN   to provide initialization needed for LHA input       *
C  S   PYLHEF   to produce a Les Houches Event File from run         *
C  S   PYINRE   to initialize treatment of resonances                *
C  S   PYINBM   to read in beam, target and frame choices            *
C  S   PYINKI   to initialize kinematics of incoming particles       *
C  S   PYINPR   to set up the selection of included processes        *
C  S   PYXTOT   to give total, elastic and diffractive cross-sect.   *
C  S   PYMAXI   to find differential cross-section maxima            *
C  S   PYPILE   to select multiplicity of pileup events              *
C  S   PYSAVE   to save alternatives for gamma-p and gamma-gamma     *
C  S   PYGAGA   to handle lepton -> lepton + gamma branchings        *
C  S   PYRAND   to select subprocess and kinematics for event        *
C  S   PYSCAT   to set up kinematics and colour flow of event        *
C  S   PYEVOL   handler for pT-ordered ISR and multiple interactions *
C  S   PYSSPA   to simulate initial state spacelike showers          *
C  S   PYPTIS   to do pT-ordered initial state spacelike showers     *
C  S   PYMEMX   auxiliary to PYSSPA/PYPTIS for ME correction maximum *
C  S   PYMEWT   auxiliary to PYSSPA/.. for matrix element correction *
C  S   PYPTMI   to do pT-ordered multiple interactions               *
C  F   PYFCMP   to give companion quark x*f distribution             *
C  F   PYPCMP   to calculate momentum integral for companion quarks  *
C  S   PYUPRE   to rearranges contents of the HEPEUP commonblock     *
C  S   PYADSH   to administrate sequential final-state showers       *
C  S   PYVETO   to allow the generation of an event to be aborted    *
C  S   PYRESD   to perform resonance decays                          *
C  S   PYMULT   to generate multiple interactions - old scheme       *
C  S   PYREMN   to add on target remnants - old scheme               *
C  S   PYMIGN   to generate multiple interactions - new scheme       *
C  S   PYMIHK   to connect colours in mult. int. - new scheme        *
C  S   PYCTTR   to translate PYTHIA colour information to LHA1 tags  *
C  S   PYMIHG   to collapse two pairs of LHA1 colour tags.           *
C  S   PYMIRM   to add on target remnants in mult. int.- new scheme  *
C  S   PYFSCR   to perform final state colour reconnections - -"-    *
C  S   PYDIFF   to set up kinematics for diffractive events          *
C  S   PYDISG   to set up kinematics, remnant and showers for DIS    *
C  S   PYDOCU   to compute cross-sections and handle documentation   *
C  S   PYFRAM   to perform boosts between different frames           *
C  S   PYWIDT   to calculate full and partial widths of resonances   *
C  S   PYOFSH   to calculate partial width into off-shell channels   *
C  S   PYRECO   to handle colour reconnection in W+W- events         *
C  S   PYKLIM   to calculate borders of allowed kinematical region   *
C  S   PYKMAP   to construct value of kinematical variable           *
C  S   PYSIGH   to calculate differential cross-sections             *
C  S   PYSGQC   auxiliary to PYSIGH for QCD processes                *
C  S   PYSGHF   auxiliary to PYSIGH for heavy flavour processes      *
C  S   PYSGWZ   auxiliary to PYSIGH for W and Z processes            *
C  S   PYSGHG   auxiliary to PYSIGH for Higgs processes              *
C  S   PYSGSU   auxiliary to PYSIGH for supersymmetry processes      *
C  S   PYSGTC   auxiliary to PYSIGH for technicolor processes        *
C  S   PYSGEX   auxiliary to PYSIGH for various exotic processes     *
C  S   PYPDFU   to evaluate parton distributions                     *
C  S   PYPDFL   to evaluate parton distributions at low x and Q^2    *
C  S   PYPDEL   to evaluate electron parton distributions            *
C  S   PYPDGA   to evaluate photon parton distributions (generic)    *
C  S   PYGGAM   to evaluate photon parton distributions (SaS sets)   *
C  S   PYGVMD   to evaluate VMD part of photon parton distributions  *
C  S   PYGANO   to evaluate anomalous part of photon PDFs            *
C  S   PYGBEH   to evaluate Bethe-Heitler part of photon PDFs        *
C  S   PYGDIR   to evaluate direct contribution to photon PDFs       *
C  S   PYPDPI   to evaluate pion parton distributions                *
C  S   PYPDPR   to evaluate proton parton distributions              *
C  F   PYCTEQ   to evaluate the CTEQ 3 proton parton distributions   *
C  S   PYGRVL   to evaluate the GRV 94L proton parton distributions  *
C  S   PYGRVM   to evaluate the GRV 94M proton parton distributions  *
C  S   PYGRVD   to evaluate the GRV 94D proton parton distributions  *
C  F   PYGRVV   auxiliary to the PYGRV* routines                     *
C  F   PYGRVW   auxiliary to the PYGRV* routines                     *
C  F   PYGRVS   auxiliary to the PYGRV* routines                     *
C  F   PYCT5L   to evaluate the CTEQ 5L proton parton distributions  *
C  F   PYCT5M   to evaluate the CTEQ 5M1 proton parton distributions *
C  S   PYPDPO   to evaluate old proton parton distributions          *
C  F   PYHFTH   to evaluate threshold factor for heavy flavour       *
C  S   PYSPLI   to find flavours left in hadron when one removed     *
C  F   PYGAMM   to evaluate ordinary Gamma function Gamma(x)         *
C  S   PYWAUX   to evaluate auxiliary functions W1(s) and W2(s)      *
C  S   PYI3AU   to evaluate auxiliary function I3(s,t,u,v)           *
C  F   PYSPEN   to evaluate Spence (dilogarithm) function Sp(x)      *
C  S   PYQQBH   to evaluate matrix element for g + g -> Q + Qbar + H *
C  S   PYSTBH   to evaluate matrix element for t + b + H processes   *
C  S   PYTBHB   auxiliary to PYSTBH                                  *
C  S   PYTBHG   auxiliary to PYSTBH                                  *
C  S   PYTBHQ   auxiliary to PYSTBH                                  *
C  F   PYTBHS   auxiliary to PYSTBH                                  *
C                                                                    *
C  S   PYMSIN   to initialize the supersymmetry simulation           *
C  S   PYSLHA   to interface to SUSY spectrum and decay calculators  *
C  S   PYAPPS   to determine MSSM parameters from SUGRA input        *
C  S   PYSUGI   to determine MSSM parameters using ISASUSY           *
C  S   PYFEYN   to determine MSSM Higgs parameters using FEYNHIGGS   *
C  F   PYRNMQ   to determine running squark masses                   *
C  S   PYTHRG   to calculate sfermion third-gen. mass eigenstates    *
C  S   PYINOM   to calculate neutralino/chargino mass eigenstates    *
C  F   PYRNM3   to determine running M3, gluino mass                 *
C  S   PYEIG4   to calculate eigenvalues and -vectors in 4*4 matrix  *
C  S   PYHGGM   to determine Higgs mass spectrum                     *
C  S   PYSUBH   to determine Higgs masses in the MSSM                *
C  S   PYPOLE   to determine Higgs masses in the MSSM                *
C  S   PYRGHM   auxiliary to PYPOLE                                  *
C  S   PYGFXX   auxiliary to PYRGHM                                  *
C  F   PYFINT   auxiliary to PYPOLE                                  *
C  F   PYFISB   auxiliary to PYFINT                                  *
C  S   PYSFDC   to calculate sfermion decay partial widths           *
C  S   PYGLUI   to calculate gluino decay partial widths             *
C  S   PYTBBN   to calculate 3-body decay of gluino to neutralino    *
C  S   PYTBBC   to calculate 3-body decay of gluino to chargino      *
C  S   PYNJDC   to calculate neutralino decay partial widths         *
C  S   PYCJDC   to calculate chargino decay partial widths           *
C  F   PYXXZ6   auxiliary for ino 3-body decays                      *
C  F   PYXXGA   auxiliary for ino -> ino + gamma decay               *
C  F   PYX2XG   auxiliary for ino -> ino + gauge boson decay         *
C  F   PYX2XH   auxiliary for ino -> ino + Higgs decay               *
C  S   PYHEXT   to calculate non-SM Higgs decay partial widths       *
C  F   PYH2XX   auxiliary for H -> ino + ino decay                   *
C  F   PYGAUS   to perform Gaussian integration                      *
C  F   PYGAU2   copy of PYGAUS to allow two-dimensional integration  *
C  F   PYSIMP   to perform Simpson integration                       *
C  F   PYLAMF   to evaluate the lambda kinematics function           *
C  S   PYTBDY   to perform 3-body decay of gauginos                  *
C  S   PYTECM   to calculate techni_rho/omega masses                 *
C  S   PYXDIN   to initialize Universal Extra Dimensions             *
C  S   PYUEDC   to compute UED mass radiative corrections            *
C  S   PYXUED   to compute UED cross sections                        *
C  S   PYGRAM   to generate UED G* (excited graviton) mass spectrum  *
C  F   PYGRAW   to compute UED partial widths to G*                  *
C  F   PYWDKK   to compute UED differential partial widths to G*     *
C  S   PYEICG   to calculate eigenvalues of a 4*4 complex matrix     *
C  S   PYCMQR   auxiliary to PYEICG                                  *
C  S   PYCMQ2   auxiliary to PYEICG                                  *
C  S   PYCDIV   auxiliary to PYCMQR                                  *
C  S   PYCSRT   auxiliary to PYCMQR                                  *
C  S   PYTHAG   auxiliary to PYCMQR                                  *
C  S   PYCBAL   auxiliary to PYEICG                                  *
C  S   PYCBA2   auxiliary to PYEICG                                  *
C  S   PYCRTH   auxiliary to PYEICG                                  *
C  S   PYLDCM   auxiliary to PYSIGH, for technicolor in QCD 2 -> 2   *
C  S   PYBKSB   auxiliary to PYSIGH, for technicolor in QCD 2 -> 2   *
C  S   PYWIDX   to calculate decay widths from within PYWIDT         *
C  S   PYRVSF   to calculate R-violating sfermion decay widths       *
C  S   PYRVNE   to calculate R-violating neutralino decay widths     *
C  S   PYRVCH   to calculate R-violating chargino decay widths       *
C  S   PYRVGL   to calculate R-violating gluino decay widths         *
C  F   PYRVSB   auxiliary to PYRVSF                                  *
C  S   PYRVGW   to calculate R-Violating 3-body widths               *
C  F   PYRVI1   auxiliary to PYRVGW, to do PS integration for res.   *
C  F   PYRVI2   auxiliary to PYRVGW, to do PS integration for LR-int.*
C  F   PYRVI3   auxiliary to PYRVGW, to do PS X integral for int.    *
C  F   PYRVG1   auxiliary to PYRVI1, general matrix element, res.    *
C  F   PYRVG2   auxiliary to PYRVI2, general matrix element, LR-int. *
C  F   PYRVG3   auxiliary to PYRVI3, to do PS Y integral for int.    *
C  F   PYRVG4   auxiliary to PYRVG3, general matrix element, int.    *
C  F   PYRVR    auxiliary to PYRVG1, Breit-Wigner                    *
C  F   PYRVS    auxiliary to PYRVG2 & PYRVG4                         *
C                                                                    *
C  S   PY1ENT   to fill one entry (= parton or particle)             *
C  S   PY2ENT   to fill two entries                                  *
C  S   PY3ENT   to fill three entries                                *
C  S   PY4ENT   to fill four entries                                 *
C  S   PY2FRM   to interface to generic two-fermion generator        *
C  S   PY4FRM   to interface to generic four-fermion generator       *
C  S   PY6FRM   to interface to generic six-fermion generator        *
C  S   PY4JET   to generate a shower from a given 4-parton config    *
C  S   PY4JTW   to evaluate the weight od a shower history for above *
C  S   PY4JTS   to set up the parton configuration for above         *
C  S   PYJOIN   to connect entries with colour flow information      *
C  S   PYGIVE   to fill (or query) commonblock variables             *
C  S   PYONOF   to allow easy control of particle decay modes        *
C  S   PYTUNE   to select a predefined 'tune' for min-bias and UE    *
C  S   PYEXEC   to administrate fragmentation and decay chain        *
C  S   PYPREP   to rearrange showered partons along strings          *
C  S   PYSTRF   to do string fragmentation of jet system             *
C  S   PYJURF   to find boost to string junction rest frame          *
C  S   PYINDF   to do independent fragmentation of one or many jets  *
C  S   PYDECY   to do the decay of a particle                        *
C  S   PYDCYK   to select parton and hadron flavours in decays       *
C  S   PYKFDI   to select parton and hadron flavours in fragm        *
C  S   PYNMES   to select number of popcorn mesons                   *
C  S   PYKFIN   to calculate falvour prod. ratios from input params. *
C  S   PYPTDI   to select transverse momenta in fragm                *
C  S   PYZDIS   to select longitudinal scaling variable in fragm     *
C  S   PYSHOW   to do m-ordered timelike parton shower evolution     *
C  S   PYPTFS   to do pT-ordered timelike parton shower evolution    *
C  F   PYMAEL   auxiliary to PYSHOW & PYPTFS: gluon emission ME's    *
C  S   PYBOEI   to include Bose-Einstein effects (crudely)           *
C  S   PYBESQ   auxiliary to PYBOEI                                  *
C  F   PYMASS   to give the mass of a particle or parton             *
C  F   PYMRUN   to give the running MSbar mass of a quark            *
C  S   PYNAME   to give the name of a particle or parton             *
C  F   PYCHGE   to give three times the electric charge              *
C  F   PYCOMP   to compress standard KF flavour code to internal KC  *
C  S   PYERRM   to write error messages and abort faulty run         *
C  F   PYALEM   to give the alpha_electromagnetic value              *
C  F   PYALPS   to give the alpha_strong value                       *
C  F   PYANGL   to give the angle from known x and y components      *
C  F   PYR      to provide a random number generator                 *
C  S   PYRGET   to save the state of the random number generator     *
C  S   PYRSET   to set the state of the random number generator      *
C  S   PYROBO   to rotate and/or boost an event                      *
C  S   PYEDIT   to remove unwanted entries from record               *
C  S   PYLIST   to list event record or particle data                *
C  S   PYLOGO   to write a logo                                      *
C  S   PYUPDA   to update particle data                              *
C  F   PYK      to provide integer-valued event information          *
C  F   PYP      to provide real-valued event information             *
C  S   PYSPHE   to perform sphericity analysis                       *
C  S   PYTHRU   to perform thrust analysis                           *
C  S   PYCLUS   to perform three-dimensional cluster analysis        *
C  S   PYCELL   to perform cluster analysis in (eta, phi, E_T)       *
C  S   PYJMAS   to give high and low jet mass of event               *
C  S   PYFOWO   to give Fox-Wolfram moments                          *
C  S   PYTABU   to analyze events, with tabular output               *
C                                                                    *
C  S   PYEEVT   to administrate the generation of an e+e- event      *
C  S   PYXTEE   to give the total cross-section at given CM energy   *
C  S   PYRADK   to generate initial state photon radiation           *
C  S   PYXKFL   to select flavour of primary qqbar pair              *
C  S   PYXJET   to select (matrix element) jet multiplicity          *
C  S   PYX3JT   to select kinematics of three-jet event              *
C  S   PYX4JT   to select kinematics of four-jet event               *
C  S   PYXDIF   to select angular orientation of event               *
C  S   PYONIA   to perform generation of onium decay to gluons       *
C                                                                    *
C  S   PYBOOK   to book a histogram                                  *
C  S   PYFILL   to fill an entry in a histogram                      *
C  S   PYFACT   to multiply histogram contents by a factor           *
C  S   PYOPER   to perform operations between histograms             *
C  S   PYHIST   to print and reset all histograms                    *
C  S   PYPLOT   to print a single histogram                          *
C  S   PYNULL   to reset contents of a single histogram              *
C  S   PYDUMP   to dump histogram contents onto a file               *
C                                                                    *
C  S   PYSTOP   routine to handle Fortran STOP condition             *
C                                                                    *
C  S   PYKCUT   dummy routine for user kinematical cuts              *
C  S   PYEVWT   dummy routine for weighting events                   *
C  S   UPINIT   dummy routine to initialize user processes           *
C  S   UPEVNT   dummy routine to generate a user process event       *
C  S   UPVETO   dummy routine to abort event at parton level         *
C  S   PDFSET   dummy routine to be removed when using PDFLIB        *
C  S   STRUCTM  dummy routine to be removed when using PDFLIB        *
C  S   STRUCTP  dummy routine to be removed when using PDFLIB        *
C  S   SUGRA    dummy routine to be removed when linking with ISAJET *
C  F   VISAJE   dummy functn. to be removed when linking with ISAJET *
C  S   SSMSSM   dummy routine to be removed when linking with ISAJET *
C  S   FHSETFLAGS  dummy routine          -"-              FEYNHIGGS *
C  S   FHSETPARA   dummy routine          -"-              FEYNHIGGS *
C  S   FHHIGGSCORR dummy routine          -"-              FEYNHIGGS *
C  S   PYTAUD   dummy routine for interface to tau decay libraries   *
C  S   PYTIME   dummy routine for giving date and time               *
C                                                                    *
C*********************************************************************
 
C...PYDATA
C...Default values for switches and parameters,
C...and particle, decay and process data.
 
      BLOCK DATA PYDATA
 
C...Double precision and integer declarations.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
      IMPLICIT INTEGER(I-N)
      INTEGER PYK,PYCHGE,PYCOMP
C...Commonblocks.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
      COMMON/PYDAT4/CHAF(500,2)
      CHARACTER CHAF*16
      COMMON/PYDATR/MRPY(6),RRPY(100)
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
      COMMON/PYINT1/MINT(400),VINT(400)
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)
      COMMON/PYINT4/MWID(500),WIDS(500,5)
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
      COMMON/PYINT6/PROC(0:500)
      CHARACTER PROC*28
      COMMON/PYINT7/SIGT(0:6,0:6,0:5)
      COMMON/PYMSSM/IMSS(0:99),RMSS(0:99)
      COMMON/PYSSMT/ZMIX(4,4),UMIX(2,2),VMIX(2,2),SMZ(4),SMW(2),
     &SFMIX(16,4),ZMIXI(4,4),UMIXI(2,2),VMIXI(2,2)
      COMMON/PYMSRV/RVLAM(3,3,3), RVLAMP(3,3,3), RVLAMB(3,3,3)
      COMMON/PYTCSM/ITCM(0:99),RTCM(0:99)
      COMMON/PYPUED/IUED(0:99),RUED(0:99)
      COMMON/PYBINS/IHIST(4),INDX(1000),BIN(20000)
      COMMON/PYLH3P/MODSEL(200),PARMIN(100),PAREXT(200),RMSOFT(0:100),
     &     AU(3,3),AD(3,3),AE(3,3)
      COMMON/PYLH3C/CPRO(2),CVER(2)
      CHARACTER CPRO*12,CVER*12
      SAVE /PYDAT1/,/PYDAT2/,/PYDAT3/,/PYDAT4/,/PYDATR/,/PYSUBS/,
     &/PYPARS/,/PYINT1/,/PYINT2/,/PYINT3/,/PYINT4/,/PYINT5/,
     &/PYINT6/,/PYINT7/,/PYMSSM/,/PYSSMT/,/PYMSRV/,/PYTCSM/,/PYPUED/,
     &/PYBINS/,/PYLH3P/,/PYLH3C/
 
C...PYDAT1, containing status codes and most parameters.
      DATA MSTU/
     &   0,    0,    0, 4000,10000,  500, 8000,    0,    0,    2,
     1   6,    0,    1,    0,    0,    1,    0,    0,    0,    0,
     2   2,   10,    0,    0,    1,   10,    0,    0,    0,    0,
     3   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     4   2,    2,    1,    4,    2,    1,    1,    0,    0,    0,
     5  25,   24,    0,    1,    0,    0,    0,    0,    0,    0,
     6   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     7  30*0,
     1   1,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     2   1,    5,    3,    5,    0,    0,    0,    0,    0,    0,
     &  80*0/
      DATA (PARU(I),I=1,100)/
     &  3.141592653589793D0, 6.283185307179586D0,
     &  0.197327D0, 5.06773D0, 0.389380D0, 2.56819D0,  4*0D0,
     1  0.001D0, 0.09D0, 0.01D0, 2D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0,
     2  0D0,   0D0,   0D0,   0D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0,
     3  0D0,   0D0,   0D0,   0D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0,
     4  2.0D0,  1.0D0, 0.25D0,  2.5D0, 0.05D0,
     4  0D0,   0D0, 0.0001D0, 0D0,   0D0,
     5  2.5D0,1.5D0,7.0D0,1.0D0,0.5D0,2.0D0,3.2D0, 0D0, 0D0, 0D0,
     6  40*0D0/
      DATA (PARU(I),I=101,200)/
     &  0.00729735D0, 0.232D0, 0.007764D0, 1.0D0, 1.16639D-5,
     &  0D0, 0D0, 0D0, 0D0,  0D0,
     1  0.20D0, 0.25D0, 1.0D0, 4.0D0, 10D0, 0D0, 0D0,  0D0, 0D0, 0D0,
     2 -0.693D0, -1.0D0, 0.387D0, 1.0D0, -0.08D0,
     2 -1.0D0,  1.0D0,  1.0D0,  1.0D0,  0D0,
     3  1.0D0,-1.0D0, 1.0D0,-1.0D0, 1.0D0,  0D0,  0D0, 0D0, 0D0, 0D0,
     4  5.0D0, 1.0D0, 1.0D0,  0D0, 1.0D0, 1.0D0,  0D0, 0D0, 0D0, 0D0,
     5  1.0D0,   0D0,   0D0,   0D0,   0D0,   0D0, 0D0, 0D0, 0D0, 0D0,
     6  1.0D0, 1.0D0, 1.0D0, 1.0D0, 1.0D0,  0D0,  0D0, 0D0, 0D0, 0D0,
     7  1.0D0, 1.0D0, 1.0D0, 1.0D0, 1.0D0, 1.0D0, 1.0D0, 0D0,0D0,0D0,
     8  1.0D0, 1.0D0, 1.0D0, 0.0D0, 0.0D0, 1.0D0, 1.0D0, 0D0,0D0,0D0,
     9  0D0,  0D0,  0D0,  0D0, 1.0D0,  0D0,  0D0, 0D0, 0D0, 0D0/
      DATA MSTJ/
     &  1,    3,    0,    0,    0,    0,    0,    0,    0,    0,
     1  4,    2,    0,    1,    0,    2,    2,   20,    0,    0,
     2  2,    1,    1,    2,    1,    2,    2,    0,    0,    0,
     3  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     4  2,    2,    4,    2,    5,    3,    3,    0,    0,    3,
     5  0,    3,    0,    2,    0,    0,    1,    0,    0,    0,
     6  40*0,
     &  5,    2,    7,    5,    1,    1,    0,    2,    0,    2,
     1  0,    0,    0,    0,    1,    1,    0,    0,    0,    0,
     2  80*0/
      DATA PARJ/
     &  0.10D0, 0.30D0, 0.40D0, 0.05D0, 0.50D0,
     &  0.50D0, 0.50D0,   0.6D0,   1.2D0,   0.6D0,
     1  0.50D0,0.60D0,0.75D0, 0D0, 0D0, 0D0, 0D0, 1.0D0, 1.0D0, 0D0,
     2  0.36D0, 1.0D0,0.01D0, 2.0D0,1.0D0,0.4D0, 0D0, 0D0, 0D0, 0D0,
     3  0.10D0, 1.0D0, 0.8D0, 1.5D0,0D0,2.0D0,0.2D0, 0D0,0.08D0,1D0,
     4  0.3D0, 0.58D0, 0.5D0, 0.9D0,0.5D0,1.0D0,1.0D0,1.5D0,1D0,10D0,
     5  0.77D0, 0.77D0, 0.77D0, -0.05D0, -0.005D0,
     5  0D0, 0D0, 0D0, 1.0D0, 0D0,
     6  4.5D0, 0.7D0, 0D0,0.003D0, 0.5D0, 0.5D0, 0D0, 0D0, 0D0, 0D0,
     7  10D0, 1000D0, 100D0, 1000D0, 0D0, 0.7D0,10D0, 0D0,0D0,0.5D0,
     8  0.29D0, 1.0D0, 1.0D0,  0D0,  10D0, 10D0, 0D0, 0D0, 0D0,1D-4,
     9  0.02D0, 1.0D0, 0.2D0,  0D0,  0D0,  0D0,  0D0, 0D0, 0D0, 0D0,
     &  0D0,  0D0,  0D0,  0D0,   0D0,   0D0,  0D0,  0D0,  0D0,  0D0,
     1  0D0,  0D0,  0D0,  0D0,   0D0,   0D0,  0D0,  0D0,  0D0,  0D0,
     2  1.0D0, 0.25D0,91.187D0,2.489D0, 0.01D0,
     2  2.0D0,  1.0D0, 0.25D0,0.002D0,   0D0,
     3  0D0, 0D0, 0D0, 0D0, 0.01D0, 0.99D0, 0D0, 0D0,  0.2D0,   0D0,
     4  10*0D0,
     5  10*0D0,
     6  10*0D0,
     7  0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, -0.693D0,
     8 -1.0D0, 0.387D0, 1.0D0, -0.08D0, -1.0D0,
     8  1.0D0,  1.0D0, -0.693D0, -1.0D0, 0.387D0,
     9  1.0D0, -0.08D0, -1.0D0,   1.0D0, 1.0D0,
     9  5*0D0/
 
C...PYDAT2, with particle data and flavour treatment parameters.
      DATA (KCHG(I,1),I=   1, 500)/-1,2,-1,2,-1,2,-1,2,2*0,-3,0,-3,0,   
     &-3,0,-3,6*0,3,9*0,3,2*0,3,4*0,-1,41*0,2,-1,20*0,3*3,7*0,3*3,3*0,  
     &3*3,3*0,3*3,6*0,3*3,3*0,3*3,4*0,-2,-3,2*1,2*0,4,2*3,6,2*-2,2*-3,  
     &0,2*1,2*0,2*3,-2,2*-3,2*0,-3,2*1,2*0,3,0,2*4,2*3,2*6,3,2*1,2*0,   
     &2*3,2*0,4,2*3,2*6,2*3,6,2*-2,2*-3,0,-3,0,2*1,2*0,2*3,0,3,2*-2,    
     &2*-3,2*0,2*-3,0,2*1,2*0,2*3,2*0,2*3,-2,2*-3,2*0,2*-3,2*0,-3,2*0,  
     &2*3,4*0,2*3,2*0,2*3,2*0,2*3,4*0,2*3,2*0,2*3,3*0,3,2*0,3,0,3,0,3,  
     &2*0,3,0,3,3*0,-1,2,-1,2,-1,2,-3,0,-3,0,-3,4*0,3,2*0,3,0,-1,2,-1,  
     &2,-1,2,-3,0,-3,0,-3,2*0,3,3*0,3,8*0,-1,2,-3,6*0,3,2*6,0,3,4*0,3,  
     &7*0,3,
C...UED singlet and doublet quarks, leptons, and KK g, gamma, Z, and W
     &81*0,-1,2,-1,2,-1,2,-1,2,-1,2,-1,2, 
     &3*-3,0,-3,0,-3,0,-3,
     &3*0,3, 
     &25*0/
      DATA (KCHG(I,2),I=   1, 500)/8*1,12*0,2,20*0,1,107*0,-1,0,2*-1,   
     &2*0,-1,3*0,2*-1,3*0,2*-1,4*0,-1,5*0,2*-1,4*0,2*-1,5*0,2*-1,6*0,   
     &-1,7*0,2*-1,5*0,2*-1,6*0,2*-1,7*0,2*-1,8*0,-1,56*0,6*1,6*0,2,7*0, 
     &6*1,9*0,2,3*0,2,0,5*2,2*1,17*0,6*2,
     &83*0,12*1,9*0,2,3*0,25*0/
      DATA (KCHG(I,3),I=   1, 500)/8*1,2*0,8*1,5*0,1,9*0,1,2*0,1,3*0,   
     &2*1,39*0,1,0,2*1,20*0,3*1,4*0,6*1,3*0,9*1,3*0,12*1,4*0,100*1,2*0, 
     &2*1,2*0,4*1,2*0,6*1,2*0,8*1,3*0,1,0,2*1,0,3*1,0,4*1,3*0,12*1,3*0, 
     &1,2*0,1,0,12*1,0,1,3*0,1,8*0,4*1,5*0,3*1,0,1,3*0,2*1,7*0,1,
     &81*0,21*1,3*0,1,25*0/
      DATA (KCHG(I,4),I=   1, 290)/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 
     &16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,   
     &37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,   
     &58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,   
     &79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,   
     &100,110,111,113,115,130,211,213,215,221,223,225,310,311,313,315,  
     &321,323,325,331,333,335,411,413,415,421,423,425,431,433,435,441,  
     &443,445,511,513,515,521,523,525,531,533,535,541,543,545,551,553,  
     &555,990,1103,1114,2101,2103,2112,2114,2203,2212,2214,2224,3101,   
     &3103,3112,3114,3122,3201,3203,3212,3214,3222,3224,3303,3312,3314, 
     &3322,3324,3334,4101,4103,4112,4114,4122,4132,4201,4203,4212,4214, 
     &4222,4224,4232,4301,4303,4312,4314,4322,4324,4332,4334,4403,4412, 
     &4414,4422,4424,4432,4434,4444,5101,5103,5112,5114,5122,5132,5142, 
     &5201,5203,5212,5214,5222,5224,5232,5242,5301,5303,5312,5314,5322, 
     &5324,5332,5334,5342,5401,5403,5412,5414,5422,5424,5432,5434,5442, 
     &5444,5503,5512,5514,5522,5524,5532,5534,5542,5544,5554,10111,     
     &10113,10211,10213,10221,10223,10311,10313,10321,10323,10331,      
     &10333,10411,10413,10421,10423,10431,10433,10441,10443,10511,      
     &10513,10521,10523,10531,10533,10541,10543,10551,10553,20113,      
     &20213,20223,20313,20323,20333,20413,20423,20433,20443,20513/      
      DATA (KCHG(I,4),I= 291, 500)/20523,20533,20543,20553,100443,      
     &100553,1000001,1000002,1000003,1000004,1000005,1000006,1000011,   
     &1000012,1000013,1000014,1000015,1000016,1000021,1000022,1000023,  
     &1000024,1000025,1000035,1000037,1000039,2000001,2000002,2000003,  
     &2000004,2000005,2000006,2000011,2000012,2000013,2000014,2000015,  
     &2000016,3000111,3000211,3000221,3000331,3000113,3000213,3000223,  
     &3100021,3100111,3200111,3100113,3200113,3300113,3400113,4000001,  
     &4000002,4000011,4000012,5000039,9900012,9900014,9900016,9900023,  
     &9900024,9900041,9900042,9900110,9900210,9900220,9900330,9900440,  
     &9902110,9902210,9900443,9900441,9910441,9900553,9900551,9910551,  
     &3000115,3000215,
     &81*0,
C...UED singlet and doublet quarks and leptons, and KK g, gamma, Z, and W.
     &6100001,6100002,6100003,6100004,6100005,6100006, 
     &5100001,5100002,5100003,5100004,5100005,5100006, 
     &6100011,6100013,6100015,
     &5100012,5100011,5100014,5100013,5100016,5100015, 
     &5100021,5100022,5100023,5100024,
     &25*0/ 
      DATA (PMAS(I,1),I=   1, 217)/2*0.33D0,0.5D0,1.5D0,4.8D0,175D0,    
     &2*400D0,2*0D0,0.00051D0,0D0,0.10566D0,0D0,1.777D0,0D0,400D0,      
     &5*0D0,91.188D0,80.45D0,115D0,6*0D0,500D0,900D0,500D0,3*300D0,     
     &3*0D0,5000D0,200D0,40*0D0,1D0,2D0,5D0,16*0D0,0.13498D0,0.7685D0,  
     &1.318D0,0.49767D0,0.13957D0,0.7669D0,1.318D0,0.54745D0,0.78194D0, 
     &1.275D0,2*0.49767D0,0.8961D0,1.432D0,0.4936D0,0.8916D0,1.425D0,   
     &0.95777D0,1.0194D0,1.525D0,1.8693D0,2.01D0,2.46D0,1.8645D0,       
     &2.0067D0,2.46D0,1.9685D0,2.1124D0,2.5735D0,2.9798D0,3.09688D0,    
     &3.5562D0,5.2792D0,5.3248D0,5.83D0,5.2789D0,5.3248D0,5.83D0,       
     &5.3693D0,5.4163D0,6.07D0,6.594D0,6.602D0,7.35D0,9.4D0,9.4603D0,   
     &9.9132D0,0D0,0.77133D0,1.234D0,0.57933D0,0.77133D0,0.93957D0,     
     &1.233D0,0.77133D0,0.93827D0,1.232D0,1.231D0,0.80473D0,0.92953D0,  
     &1.19744D0,1.3872D0,1.11568D0,0.80473D0,0.92953D0,1.19255D0,       
     &1.3837D0,1.18937D0,1.3828D0,1.09361D0,1.3213D0,1.535D0,1.3149D0,  
     &1.5318D0,1.67245D0,1.96908D0,2.00808D0,2.4521D0,2.5D0,2.2849D0,   
     &2.4703D0,1.96908D0,2.00808D0,2.4535D0,2.5D0,2.4529D0,2.5D0,       
     &2.4656D0,2.15432D0,2.17967D0,2.55D0,2.63D0,2.55D0,2.63D0,2.704D0, 
     &2.8D0,3.27531D0,3.59798D0,3.65648D0,3.59798D0,3.65648D0,          
     &3.78663D0,3.82466D0,4.91594D0,5.38897D0,5.40145D0,5.8D0,5.81D0,   
     &5.641D0,5.84D0,7.00575D0,5.38897D0,5.40145D0,5.8D0,5.81D0,5.8D0/  
      DATA (PMAS(I,1),I= 218, 500)/5.81D0,5.84D0,7.00575D0,5.56725D0,   
     &5.57536D0,5.96D0,5.97D0,5.96D0,5.97D0,6.12D0,6.13D0,7.19099D0,    
     &6.67143D0,6.67397D0,7.03724D0,7.0485D0,7.03724D0,7.0485D0,        
     &7.21101D0,7.219D0,8.30945D0,8.31325D0,10.07354D0,10.42272D0,      
     &10.44144D0,10.42272D0,10.44144D0,10.60209D0,10.61426D0,           
     &11.70767D0,11.71147D0,15.11061D0,0.9835D0,1.231D0,0.9835D0,       
     &1.231D0,1D0,1.17D0,1.429D0,1.29D0,1.429D0,1.29D0,2*1.4D0,2.272D0, 
     &2.424D0,2.272D0,2.424D0,2.5D0,2.536D0,3.4151D0,3.46D0,5.68D0,     
     &5.73D0,5.68D0,5.73D0,5.92D0,5.97D0,7.25D0,7.3D0,9.8598D0,9.875D0, 
     &2*1.23D0,1.282D0,2*1.402D0,1.427D0,2*2.372D0,2.56D0,3.5106D0,     
     &2*5.78D0,6.02D0,7.3D0,9.8919D0,3.686D0,10.0233D0,32*500D0,        
     &3*110D0,350D0,3*210D0,500D0,125D0,250D0,400D0,2*350D0,300D0,      
     &4*400D0,1000D0,3*500D0,1200D0,750D0,2*200D0,7*0D0,3*3.1D0,        
     &3*9.5D0,2*250D0,
     &81*0,
C...UED
     &586.,588.,586.,588.,586.,586.,6*598.,
     &3*505.,6*516.,640.,501.,536.,536.,25*0.D0/
      DATA (PMAS(I,2),I=   1, 500)/5*0D0,1.39816D0,16*0D0,2.47813D0,    
     &2.07115D0,0.00367D0,6*0D0,14.54029D0,0D0,16.66099D0,8.38842D0,    
     &3.3752D0,4.17669D0,3*0D0,417.29147D0,0.39162D0,60*0D0,0.151D0,   
     &0.107D0,2*0D0,0.149D0,0.107D0,0D0,0.00843D0,0.185D0,2*0D0,        
     &0.0505D0,0.109D0,0D0,0.0498D0,0.098D0,0.0002D0,0.00443D0,0.076D0, 
     &2*0D0,0.023D0,2*0D0,0.023D0,2*0D0,0.015D0,0.0013D0,0D0,0.002D0,   
     &2*0D0,0.02D0,2*0D0,0.02D0,2*0D0,0.02D0,2*0D0,0.02D0,5*0D0,0.12D0, 
     &3*0D0,0.12D0,2*0D0,2*0.12D0,3*0D0,0.0394D0,4*0D0,0.036D0,0D0,     
     &0.0358D0,2*0D0,0.0099D0,0D0,0.0091D0,74*0D0,0.06D0,0.142D0,       
     &0.06D0,0.142D0,0D0,0.36D0,0.287D0,0.09D0,0.287D0,0.09D0,0.25D0,   
     &0.08D0,0.05D0,0.02D0,0.05D0,0.02D0,0.05D0,0D0,0.014D0,0.01D0,     
     &8*0.05D0,0D0,0.01D0,2*0.4D0,0.025D0,2*0.174D0,0.053D0,3*0.05D0,   
     &0.0009D0,4*0.05D0,3*0D0,19*1D0,0D0,7*1D0,0D0,1D0,0D0,1D0,0D0,     
     &0.0208D0,0.01195D0,0.03705D0,0.09511D0,1.89978D0,1.60746D0,       
     &0.13396D0,200.47294D0,0.02296D0,0.18886D0,94.66794D0,6.08718D0,   
     &0D0,2.17482D0,2.59359D0,2.59687D0,0.42896D0,0.41912D0,0.14153D0,  
     &2*0.00098D0,0.00097D0,26.7245D0,21.74916D0,0.88159D0,0.88001D0,   
     &7*0D0,6*0.01D0,0.25499D0,0.28446D0,131*0D0/                       
      DATA (PMAS(I,3),I=   1, 500)/5*0D0,13.98156D0,16*0D0,24.78129D0,  
     &20.71149D0,0.03669D0,6*0D0,145.40294D0,0D0,166.60993D0,           
     &83.88423D0,33.75195D0,41.76694D0,3*0D0,4172.91467D0,3.91621D0,    
     &60*0D0,0.4D0,0.25D0,2*0D0,0.4D0,0.25D0,0D0,0.1D0,0.17D0,2*0D0,    
     &0.2D0,0.12D0,0D0,0.2D0,0.12D0,0.002D0,0.015D0,0.2D0,2*0D0,0.12D0, 
     &2*0D0,0.12D0,2*0D0,0.05D0,0.005D0,0D0,0.01D0,2*0D0,0.05D0,2*0D0,  
     &0.05D0,2*0D0,0.05D0,2*0D0,0.05D0,5*0D0,0.14D0,3*0D0,0.14D0,2*0D0, 
     &2*0.14D0,3*0D0,0.04D0,4*0D0,0.035D0,0D0,0.035D0,2*0D0,0.05D0,0D0, 
     &0.05D0,74*0D0,0.05D0,0.25D0,0.05D0,0.25D0,0D0,0.2D0,0.4D0,        
     &0.005D0,0.4D0,0.01D0,0.35D0,0.001D0,0.1D0,0.08D0,0.1D0,0.08D0,    
     &0.1D0,0D0,0.05D0,0.02D0,6*0.1D0,0.05D0,0.1D0,0D0,0.02D0,2*0.3D0,  
     &0.05D0,2*0.3D0,0.02D0,2*0.1D0,0.03D0,0.001D0,4*0.1D0,3*0D0,       
     &19*10D0,0.00001D0,7*10D0,0.00001D0,10D0,0.00001D0,10D0,0.00001D0, 
     &0.20797D0,0.11949D0,0.37048D0,0.95114D0,18.99785D0,16.07463D0,    
     &1.33964D0,450D0,0.22959D0,1.88863D0,360D0,60.8718D0,0D0,          
     &21.74824D0,25.93594D0,25.96873D0,4.28961D0,4.19124D0,1.41528D0,   
     &0.00977D0,0.00976D0,0.00973D0,267.24501D0,217.49162D0,8.81592D0,  
     &8.80013D0,13*0D0,2.54987D0,2.84456D0,
     &81*0,
C...UED
     &12*0.2D0,9*0.1D0,0.2,10.,0.07,0.3,25*0.D0/
      DATA (PMAS(I,4),I=   1, 500)/12*0D0,658654D0,0D0,0.0872D0,68*0D0, 
     &0.1D0,0.387D0,16*0D0,0.00003D0,2*0D0,15500D0,7804.5D0,5*0D0,      
     &26.762D0,3*0D0,3709D0,5*0D0,0.317D0,2*0D0,0.1244D0,2*0D0,0.14D0,  
     &5*0D0,0.468D0,2*0D0,0.462D0,2*0D0,0.483D0,2*0D0,0.15D0,18*0D0,    
     &44.34D0,0D0,78.88D0,4*0D0,23.96D0,2*0D0,49.1D0,0D0,87.1D0,0D0,    
     &24.6D0,4*0D0,0.0618D0,0.029D0,6*0D0,0.106D0,6*0D0,0.019D0,2*0D0,  
     &7*0.1D0,4*0D0,0.342D0,2*0.387D0,6*0D0,2*0.387D0,6*0D0,0.387D0,    
     &0D0,0.387D0,2*0D0,8*0.387D0,0D0,9*0.387D0,120*0D0,131*0D0/        

      DATA PARF/
     &  0.5D0,0.25D0, 0.5D0,0.25D0, 1D0, 0.5D0,  0D0,  0D0,  0D0, 0D0,
     1  0.5D0,  0D0, 0.5D0,  0D0,  1D0,  1D0,  0D0,  0D0,  0D0, 0D0,
     2  0.5D0,  0D0, 0.5D0,  0D0,  1D0,  1D0,  0D0,  0D0,  0D0, 0D0,
     3  0.5D0,  0D0, 0.5D0,  0D0,  1D0,  1D0,  0D0,  0D0,  0D0, 0D0,
     4  0.5D0,  0D0, 0.5D0,  0D0,  1D0,  1D0,  0D0,  0D0,  0D0, 0D0,
     5  0.5D0,  0D0, 0.5D0,  0D0,  1D0,  1D0,  0D0,  0D0,  0D0, 0D0,
     6  0.75D0, 0.5D0, 0D0,0.1667D0,0.0833D0,0.1667D0,0D0,0D0,0D0, 0D0,
     7  0D0,  0D0,  1D0,0.3333D0,0.6667D0,0.3333D0,0D0,0D0,0D0, 0D0,
     8  0D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0, 0D0,
     9  0.0099D0, 0.0056D0, 0.199D0, 1.23D0, 4.17D0, 165D0,  4*0D0,
     & 0.325D0,0.325D0,0.5D0,1.6D0, 5.0D0,  0D0,  0D0,  0D0,  0D0, 0D0,
     1 0D0,0.11D0,0.16D0,0.048D0,0.50D0,0.45D0,0.55D0,0.60D0,0D0,0D0,
     2 0.2D0, 0.1D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0,  0D0, 0D0,
     3 60*0D0,
     4 0.2D0,  0.5D0,  8*0D0,
     5 1800*0D0/
      DATA ((VCKM(I,J),J=1,4),I=1,4)/
     &  0.95113D0,  0.04884D0,  0.00003D0,  0.00000D0,
     &  0.04884D0,  0.94940D0,  0.00176D0,  0.00000D0,
     &  0.00003D0,  0.00176D0,  0.99821D0,  0.00000D0,
     &  0.00000D0,  0.00000D0,  0.00000D0,  1.00000D0/
 
C...PYDAT3, with particle decay parameters and data.
      DATA (MDCY(I,1),I=   1, 500)/5*0,3*1,6*0,1,0,1,5*0,3*1,6*0,1,0,   
     &4*1,3*0,2*1,40*0,3*1,16*0,3*1,2*0,9*1,0,32*1,2*0,1,3*0,1,2*0,2*1, 
     &2*0,3*1,2*0,4*1,0,5*1,2*0,4*1,2*0,5*1,2*0,6*1,0,7*1,2*0,5*1,2*0,  
     &6*1,2*0,7*1,2*0,8*1,0,75*1,0,7*1,0,1,0,1,0,26*1,7*0,8*1,
     &81*0,
C...UED
     &5*1,0,5*1,0,13*1,25*0/
      DATA (MDCY(I,2),I=   1, 351)/1,9,17,25,33,41,56,66,2*0,76,80,82,  
     &87,89,143,145,150,2*0,153,162,174,190,210,6*0,289,0,311,334,420,  
     &503,3*0,530,539,40*0,540,541,545,16*0,554,556,561,570,579,581,    
     &583,590,598,604,613,615,617,620,630,636,639,650,656,667,673,736,  
     &739,747,808,810,818,851,853,857,858,861,863,899,900,908,944,945,  
     &953,992,993,997,1028,1029,1033,1034,1043,2*0,1045,3*0,1046,2*0,   
     &1049,1052,2*0,1053,1055,1058,2*0,1062,1063,1066,1069,0,1072,1077, 
     &1079,1082,1084,2*0,1088,1089,1090,1166,2*0,1170,1171,1172,1173,   
     &1174,2*0,1178,1179,1181,1182,1184,1188,0,1189,1193,1197,1201,     
     &1205,1209,1213,2*0,1217,1218,1219,1236,1245,2*0,1254,1255,1256,   
     &1257,1258,1267,2*0,1276,1277,1278,1279,1280,1289,1290,2*0,1299,   
     &1308,1317,1326,1335,1344,1353,1362,0,1371,1380,1389,1398,1407,    
     &1416,1425,1434,1443,1452,1453,1454,1455,1456,1461,1464,1466,1471, 
     &1473,1478,1485,1489,1491,1493,1495,1497,1499,1501,1503,1504,1506, 
     &1508,1510,1512,1514,1516,1518,1520,1522,1523,1525,1527,1541,1543, 
     &1545,1549,1551,1553,1555,1557,1559,1561,1563,1565,1567,1578,1592, 
     &1637,1661,1706,1730,1775,1802,1833,1859,1891,1917,1949,1975,2162, 
     &2331,2595,2826,3106,3402,0,3657,3706,3734,3783,3811,3860,3888,0,  
     &3924,0,3960,0,3996,4004,4012,4020,4217,4243,4270,4023,4029,4036,  
     &4043,4050,4056,4062,4071,4075,4079,4082,4084,4104,4126,4148,4170/ 
      DATA (MDCY(I,2),I= 352, 500)/4185,4197,4204,7*0,4211,4212,4213,   
     &4214,4215,4216,4296,4322,
     &81*0,
C...UED
     %5001,5003,5005,5007,5009,5011,5013,5016,5019,5022,5025,5028,
     &5031,5032,5033,
     &5034,5035,5036,5037,5038,5039,5040,5064,5065,5083,
     &25*0/
      DATA (MDCY(I,3),I=   1, 500)/5*8,15,2*10,2*0,4,2,5,2,54,2,5,3,    
     &2*0,9,12,16,20,79,6*0,22,0,23,86,83,27,3*0,9,1,40*0,1,4,9,16*0,2, 
     &5,2*9,2*2,7,8,6,9,2*2,3,10,6,3,11,6,11,6,63,3,8,61,2,8,33,2,4,1,  
     &3,2,36,1,8,36,1,8,39,1,4,31,1,4,1,9,2,2*0,1,3*0,3,2*0,3,1,2*0,2,  
     &3,4,2*0,1,3*3,0,5,2,3,2,4,2*0,2*1,76,4,2*0,4*1,4,2*0,1,2,1,2,4,1, 
     &0,7*4,2*0,2*1,17,2*9,2*0,4*1,2*9,2*0,4*1,9,1,9,2*0,8*9,0,9*9,4*1, 
     &5,3,2,5,2,5,7,4,7*2,1,9*2,1,2*2,14,2*2,4,9*2,11,14,45,24,45,24,   
     &45,27,31,26,32,26,32,26,187,169,264,231,280,296,255,0,49,28,49,   
     &28,49,28,36,0,36,0,36,0,3*8,3,26,27,26,6,3*7,2*6,9,2*4,3,2,20,    
     &3*22,15,12,2*7,7*0,6*1,26,30,
     &81*0,
C...UED
     &6*2,6*3,9*1,24,1,18,6,25*0/                                 
      DATA (MDME(I,1),I=   1,8000)/6*1,-1,7*1,-1,7*1,-1,7*1,-1,7*1,-1,  
     &7*1,-1,1,7*-1,8*1,2*-1,8*1,2*-1,73*1,-1,2*1,-1,5*1,0,2*-1,6*1,0,  
     &2*-1,3*1,-1,6*1,2*-1,6*1,2*-1,3*1,-1,3*1,-1,3*1,5*-1,3*1,-1,6*1,  
     &2*-1,3*1,-1,5*1,62*1,6*1,2*-1,6*1,8*-1,3*1,-1,3*1,-1,3*1,5*-1,   
     &3*1,4*-1,6*1,2*-1,3*1,-1,12*1,62*1,6*1,2*-1,3*1,-1,9*1,62*1,    
     &3*1,-1,3*1,-1,1,18*1,4*1,2*-1,2*1,-1,1249*1,2*-1,377*1,2*-1,     
     &1921*1,2*-1,6*1,2*-1,133*1,2*-1,6*1,2*-1,10*1,-1,3*1,-1,3*1,5*-1, 
     &3*1,-1,16*1,2*-1,6*1,2*-1,16*1,2*-1,6*1,2*-1,13*1,-1,3*1,-1,3*1,  
     &5*-1,3*1,-1,
     &649*0,
C...UED
     &10*1,2*0,15*1,3*0,9*1,5*1,0,5*1,0,5*1,0,5*1,0,
     &1,24*1,2912*0/
      DATA (MDME(I,2),I=   1,8000)/43*102,4*0,102,0,6*53,3*102,4*0,102, 
     &2*0,3*102,4*0,102,2*0,6*102,42,6*102,2*42,2*0,8*41,2*0,36*41,     
     &8*102,0,102,0,102,2*0,21*102,8*32,8*0,16*32,4*0,8*32,9*0,62*53,   
     &8*32,14*0,16*32,7*0,8*32,16*0,62*53,8*32,13*0,62*53,4*32,5*0,     
     &18*53,6*32,4*0,12,2*42,2*11,9*42,0,2,3,15*0,4*42,5*0,3,12*0,2,    
     &3*0,1,0,3,16*0,2*3,15*0,2*42,2*3,18*0,2*3,3*0,1,11*0,22*42,41*0,  
     &2*3,9*0,16*42,45*0,3,10*0,10*42,20*0,2*13,6*0,12,2*0,12,0,12,     
     &14*42,16*0,48,3*13,2*42,9*0,14*42,16*0,48,3*13,2*42,9*0,14*42,    
     &19*0,48,3*13,2*42,6*0,2*11,28*42,5*0,32,3*0,4*32,2*4,0,32,45*0,   
     &14*42,52*0,10*13,2*42,2*11,4*0,2*42,2*11,6*0,2*42,2*11,0,2*42,    
     &2*11,2*42,2*11,2*42,2*11,2*42,2*11,2*42,2*11,2*42,2*11,2*42,2*11, 
     &2*0,3*42,8*0,48,3*13,20*42,4*0,18*42,4*0,9*42,0,162*42,50*0,2*12, 
     &17*0,2*32,33*0,12,9*0,32,2*0,12,11*0,4*32,2*4,5*0,2404*53,4*32,   
     &3*0,6*32,3*0,4*32,3*0,50*32,3*53,12*0,8*32,12*0,66*51,6*32,9*0,   
     &9*32,17*0,6*51,10*0,8*32,15*0,16*32,14*0,8*32,18*0,8*32,18*0,     
     &16*32,
C...UED
     &653*0,30*0,9*0,12*0,37*0,2912*0/
      DATA (BRAT(I)  ,I=   1, 348)/43*0D0,0.00003D0,0.001765D0,         
     &0.998205D0,35*0D0,1D0,6*0D0,0.1783D0,0.1735D0,0.1131D0,0.2494D0,  
     &0.003D0,0.09D0,0.0027D0,0.01D0,0.0014D0,0.0012D0,2*0.00025D0,     
     &0.0071D0,0.012D0,0.0004D0,0.00075D0,0.00006D0,2*0.00078D0,        
     &0.0034D0,0.08D0,0.011D0,0.0191D0,0.00006D0,0.005D0,0.0133D0,      
     &0.0067D0,0.0005D0,0.0035D0,0.0006D0,0.0015D0,0.00021D0,0.0002D0,  
     &0.00075D0,0.0001D0,0.0002D0,0.0011D0,3*0.0002D0,0.00022D0,        
     &0.0004D0,0.0001D0,2*0.00205D0,2*0.00069D0,0.00025D0,0.00051D0,    
     &0.00025D0,35*0D0,0.153995D0,0.11942D0,0.153984D0,0.119259D0,      
     &0.152272D0,3*0D0,0.033576D0,0.066806D0,0.033576D0,0.066806D0,     
     &0.0335D0,0.066806D0,2*0D0,0.321369D0,0.016494D0,2*0D0,0.016502D0, 
     &0.320615D0,2*0D0,0.00001D0,0.000591D0,6*0D0,2*0.108166D0,         
     &0.108087D0,0D0,0.000001D0,0D0,0.000353D0,0.04359D0,0.795274D0,    
     &4*0D0,0.000339D0,0.095746D0,0D0,0.060724D0,0.003054D0,0.000919D0, 
     &64*0D0,0.145835D0,0.113276D0,0.145835D0,0.113271D0,0.145781D0,    
     &0.049002D0,2*0D0,0.032025D0,0.063642D0,0.032025D0,0.063642D0,     
     &0.032022D0,0.063642D0,8*0D0,0.251225D0,0.0129D0,0.000006D0,0D0,   
     &0.0129D0,0.250764D0,0.00038D0,0D0,0.000008D0,0.000465D0,          
     &0.215418D0,5*0D0,2*0.085312D0,0.08531D0,7*0D0,0.000029D0,         
     &0.000536D0,5*0D0,0.000074D0,0D0,0.000417D0,0.000015D0,0.000061D0/ 
      DATA (BRAT(I)  ,I= 349, 655)/0.306789D0,0.689189D0,0D0,0.00289D0, 
     &69*0D0,0.000001D0,0.000072D0,0.001333D0,4*0D0,0.000001D0,         
     &0.000184D0,0D0,0.003108D0,0.000015D0,0.000003D0,2*0D0,0.995284D0, 
     &66*0D0,0.000014D0,0.082234D0,2*0D0,0.000013D0,0.003746D0,0D0,     
     &0.913992D0,18*0D0,3*0.215119D0,0.214724D0,2*0D0,0.06996D0,        
     &0.069959D0,0D0,2*1D0,2*0.08D0,0.76D0,0.08D0,2*0.105D0,0.04D0,     
     &0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,0.988D0,0.012D0,       
     &0.998739D0,0.00079D0,0.00038D0,0.000046D0,0.000045D0,2*0.34725D0, 
     &0.144D0,0.104D0,0.0245D0,2*0.01225D0,0.0028D0,0.0057D0,0.2112D0,  
     &0.1256D0,2*0.1939D0,2*0.1359D0,0.002D0,0.001D0,0.0006D0,          
     &0.999877D0,0.000123D0,0.99955D0,0.00045D0,2*0.34725D0,0.144D0,    
     &0.104D0,0.049D0,0.0028D0,0.0057D0,0.3923D0,0.321D0,0.2317D0,      
     &0.0478D0,0.0049D0,0.0013D0,0.0003D0,0.0007D0,0.89D0,0.08693D0,    
     &0.0221D0,0.00083D0,2*0.00007D0,0.564D0,0.282D0,0.072D0,0.028D0,   
     &0.023D0,2*0.0115D0,0.005D0,0.003D0,0.6861D0,0.3139D0,2*0.5D0,     
     &0.665D0,0.333D0,0.002D0,0.333D0,0.166D0,0.168D0,0.084D0,0.087D0,  
     &0.043D0,0.059D0,2*0.029D0,0.002D0,0.6352D0,0.2116D0,0.0559D0,     
     &0.0173D0,0.0482D0,0.0318D0,0.666D0,0.333D0,0.001D0,0.332D0,       
     &0.166D0,0.168D0,0.084D0,0.086D0,0.043D0,0.059D0,2*0.029D0,        
     &2*0.002D0,0.437D0,0.208D0,0.302D0,0.0302D0,0.0212D0,0.0016D0/     
      DATA (BRAT(I)  ,I= 656, 831)/0.48947D0,0.34D0,3*0.043D0,0.027D0,  
     &0.0126D0,0.0013D0,0.0003D0,0.00025D0,0.00008D0,0.444D0,2*0.222D0, 
     &0.104D0,2*0.004D0,0.07D0,0.065D0,2*0.005D0,2*0.011D0,5*0.001D0,   
     &0.07D0,0.065D0,2*0.005D0,2*0.011D0,5*0.001D0,0.026D0,0.019D0,     
     &0.066D0,0.041D0,0.045D0,0.076D0,0.0073D0,2*0.0047D0,0.026D0,      
     &0.001D0,0.0006D0,0.0066D0,0.005D0,2*0.003D0,2*0.0006D0,2*0.001D0, 
     &0.006D0,0.005D0,0.012D0,0.0057D0,0.067D0,0.008D0,0.0022D0,        
     &0.027D0,0.004D0,0.019D0,0.012D0,0.002D0,0.009D0,0.0218D0,0.001D0, 
     &0.022D0,0.087D0,0.001D0,0.0019D0,0.0015D0,0.0028D0,0.683D0,       
     &0.306D0,0.011D0,0.3D0,0.15D0,0.16D0,0.08D0,0.13D0,0.06D0,0.08D0,  
     &0.04D0,0.034D0,0.027D0,2*0.002D0,2*0.004D0,2*0.002D0,0.034D0,     
     &0.027D0,2*0.002D0,2*0.004D0,2*0.002D0,0.0365D0,0.045D0,0.073D0,   
     &0.062D0,3*0.021D0,0.0061D0,0.015D0,0.025D0,0.0088D0,0.074D0,      
     &0.0109D0,0.0041D0,0.002D0,0.0035D0,0.0011D0,0.001D0,0.0027D0,     
     &2*0.0016D0,0.0018D0,0.011D0,0.0063D0,0.0052D0,0.018D0,0.016D0,    
     &0.0034D0,0.0036D0,0.0009D0,0.0006D0,0.015D0,0.0923D0,0.018D0,     
     &0.022D0,0.0077D0,0.009D0,0.0075D0,0.024D0,0.0085D0,0.067D0,       
     &0.0511D0,0.017D0,0.0004D0,0.0028D0,0.619D0,0.381D0,0.3D0,0.15D0,  
     &0.16D0,0.08D0,0.13D0,0.06D0,0.08D0,0.04D0,0.01D0,2*0.02D0,0.03D0, 
     &2*0.005D0,2*0.02D0,0.03D0,2*0.005D0,0.015D0,0.037D0,0.028D0/      
      DATA (BRAT(I)  ,I= 832, 997)/0.079D0,0.095D0,0.052D0,0.0078D0,    
     &4*0.001D0,0.028D0,0.033D0,0.026D0,0.05D0,0.01D0,4*0.005D0,0.25D0, 
     &0.0952D0,0.94D0,0.06D0,2*0.4D0,2*0.1D0,1D0,0.0602D0,0.0601D0,     
     &0.8797D0,0.135D0,0.865D0,0.02D0,0.055D0,2*0.005D0,0.008D0,        
     &0.012D0,0.02D0,0.055D0,2*0.005D0,0.008D0,0.012D0,0.01D0,0.03D0,   
     &0.0035D0,0.011D0,0.0055D0,0.0042D0,0.009D0,0.018D0,0.015D0,       
     &0.0185D0,0.0135D0,0.025D0,0.0004D0,0.0007D0,0.0008D0,0.0014D0,    
     &0.0019D0,0.0025D0,0.4291D0,0.08D0,0.07D0,0.02D0,0.015D0,0.005D0,  
     &1D0,0.3D0,0.15D0,0.16D0,0.08D0,0.13D0,0.06D0,0.08D0,0.04D0,       
     &0.02D0,0.055D0,2*0.005D0,0.008D0,0.012D0,0.02D0,0.055D0,          
     &2*0.005D0,0.008D0,0.012D0,0.01D0,0.03D0,0.0035D0,0.011D0,         
     &0.0055D0,0.0042D0,0.009D0,0.018D0,0.015D0,0.0185D0,0.0135D0,      
     &0.025D0,0.0004D0,0.0007D0,0.0008D0,0.0014D0,0.0019D0,0.0025D0,    
     &0.4291D0,0.08D0,0.07D0,0.02D0,0.015D0,0.005D0,1D0,0.3D0,0.15D0,   
     &0.16D0,0.08D0,0.13D0,0.06D0,0.08D0,0.04D0,0.02D0,0.055D0,         
     &2*0.005D0,0.008D0,0.012D0,0.02D0,0.055D0,2*0.005D0,0.008D0,       
     &0.012D0,0.01D0,0.03D0,0.0035D0,0.011D0,0.0055D0,0.0042D0,0.009D0, 
     &0.018D0,0.015D0,0.0185D0,0.0135D0,0.025D0,2*0.0002D0,0.0007D0,    
     &2*0.0004D0,0.0014D0,0.001D0,0.0009D0,0.0025D0,0.4291D0,0.08D0,    
     &0.07D0,0.02D0,0.015D0,0.005D0,1D0,2*0.3D0,2*0.2D0,0.047D0/        
      DATA (BRAT(I)  ,I= 998,1188)/0.122D0,0.006D0,0.012D0,0.035D0,     
     &0.012D0,0.035D0,0.003D0,0.007D0,0.15D0,0.037D0,0.008D0,0.002D0,   
     &0.05D0,0.015D0,0.003D0,0.001D0,0.014D0,0.042D0,0.014D0,0.042D0,   
     &0.24D0,0.065D0,0.012D0,0.003D0,0.001D0,0.002D0,0.001D0,0.002D0,   
     &0.014D0,0.003D0,1D0,2*0.3D0,2*0.2D0,1D0,0.0252D0,0.0248D0,        
     &0.0267D0,0.015D0,0.045D0,0.015D0,0.045D0,0.7743D0,0.029D0,0.22D0, 
     &0.78D0,1D0,0.331D0,0.663D0,0.006D0,0.663D0,0.331D0,0.006D0,1D0,   
     &0.999D0,0.001D0,0.88D0,2*0.06D0,0.639D0,0.358D0,0.002D0,0.001D0,  
     &1D0,0.88D0,2*0.06D0,0.516D0,0.483D0,0.001D0,0.88D0,2*0.06D0,      
     &0.9988D0,0.0001D0,0.0006D0,0.0004D0,0.0001D0,0.667D0,0.333D0,     
     &0.9954D0,0.0011D0,0.0035D0,0.333D0,0.667D0,0.676D0,0.234D0,       
     &0.085D0,0.005D0,2*1D0,0.018D0,2*0.005D0,0.003D0,0.002D0,          
     &2*0.006D0,0.018D0,2*0.005D0,0.003D0,0.002D0,2*0.006D0,0.0066D0,   
     &0.025D0,0.016D0,0.0088D0,2*0.005D0,0.0058D0,0.005D0,0.0055D0,     
     &4*0.004D0,2*0.002D0,2*0.004D0,0.003D0,0.002D0,2*0.003D0,          
     &3*0.002D0,2*0.001D0,0.002D0,2*0.001D0,2*0.002D0,0.0013D0,         
     &0.0018D0,5*0.001D0,4*0.003D0,2*0.005D0,2*0.002D0,2*0.001D0,       
     &2*0.002D0,2*0.001D0,0.2432D0,0.057D0,2*0.035D0,0.15D0,2*0.075D0,  
     &0.03D0,2*0.015D0,2*0.08D0,0.76D0,0.08D0,4*1D0,2*0.08D0,0.76D0,    
     &0.08D0,1D0,2*0.5D0,1D0,2*0.5D0,2*0.08D0,0.76D0,0.08D0,1D0/        
      DATA (BRAT(I)  ,I=1189,1381)/2*0.08D0,0.76D0,3*0.08D0,0.76D0,     
     &3*0.08D0,0.76D0,3*0.08D0,0.76D0,3*0.08D0,0.76D0,3*0.08D0,0.76D0,  
     &3*0.08D0,0.76D0,0.08D0,2*1D0,2*0.105D0,0.04D0,0.0077D0,0.02D0,    
     &0.0235D0,0.0285D0,0.0435D0,0.0011D0,0.0022D0,0.0044D0,0.4291D0,   
     &0.08D0,0.07D0,0.02D0,0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,      
     &0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,      
     &0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,4*1D0,2*0.105D0,0.04D0,      
     &0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,0.04D0,      
     &0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,4*1D0,2*0.105D0,       
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,1D0,2*0.105D0,  
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0,      
     &0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,0.015D0,0.005D0,2*0.105D0/      
      DATA (BRAT(I)  ,I=1382,1582)/0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,   
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,2*0.105D0,0.04D0,0.5D0,0.08D0,0.14D0,0.01D0,      
     &0.015D0,0.005D0,4*1D0,0.52D0,0.26D0,0.11D0,2*0.055D0,0.333D0,     
     &0.334D0,0.333D0,0.667D0,0.333D0,0.28D0,0.14D0,0.313D0,0.157D0,    
     &0.11D0,0.667D0,0.333D0,0.28D0,0.14D0,0.313D0,0.157D0,0.11D0,      
     &0.36D0,0.18D0,0.03D0,2*0.015D0,2*0.2D0,4*0.25D0,0.667D0,0.333D0,  
     &0.667D0,0.333D0,0.667D0,0.333D0,0.667D0,0.333D0,4*0.5D0,0.007D0,  
     &0.993D0,1D0,0.667D0,0.333D0,0.667D0,0.333D0,0.667D0,0.333D0,      
     &0.667D0,0.333D0,8*0.5D0,0.02D0,0.98D0,1D0,4*0.5D0,3*0.146D0,      
     &3*0.05D0,0.15D0,2*0.05D0,4*0.024D0,0.066D0,0.667D0,0.333D0,       
     &0.667D0,0.333D0,4*0.25D0,0.667D0,0.333D0,0.667D0,0.333D0,2*0.5D0, 
     &0.273D0,0.727D0,0.667D0,0.333D0,0.667D0,0.333D0,4*0.5D0,0.35D0,   
     &0.65D0,2*0.0083D0,0.1866D0,0.324D0,0.184D0,0.027D0,0.001D0,       
     &0.093D0,0.087D0,0.078D0,0.0028D0,3*0.014D0,0.008D0,0.024D0/       
      DATA (BRAT(I)  ,I=1583,4150)/0.008D0,0.024D0,0.425D0,0.02D0,      
     &0.185D0,0.088D0,0.043D0,0.067D0,0.066D0,2404*0D0,0.024396D0,      
     &0.045285D0,0.83119D0,2*0D0,0.000349D0,0.09878D0,0D0,0.019884D0,   
     &0.02341D0,0.362776D0,0.550787D0,2*0D0,0.000152D0,0.042991D0,      
     &0.013695D0,0.025421D0,0.466595D0,2*0D0,0.000196D0,0.055451D0,     
     &0.438642D0,0.445781D0,0D0,0.554219D0,4*0.00335D0,0.522257D0,      
     &0.464343D0,6*0D0,1D0,6*0D0,1D0,4*0.013853D0,0.562703D0,           
     &0.376702D0,0.00518D0,4*0.006254D0,0.974985D0,7*0D0,4*0.148299D0,  
     &0.015351D0,0D0,0.182109D0,0.167099D0,0.042247D0,0.850973D0,       
     &0.005411D0,0.045025D0,0.098591D0,0.849898D0,0.021617D0,           
     &0.030018D0,0.098466D0,0.294448D0,0.10945D0,0.596102D0,0.389906D0, 
     &0.610094D0,3*0.0633D0,0.063299D0,0.063295D0,0.056281D0,2*0D0,     
     &6*0.020495D0,2*0D0,0.327919D0,0.04099D0,0.045236D0,0.090112D0,    
     &0.19874D0,0.010204D0,0.000003D0,0.010205D0,0.198356D0,0.000151D0, 
     &0.000006D0,0.000367D0,0.081967D0,0.19874D0,0.010204D0,0.000003D0, 
     &0.010205D0,0.198356D0,0.000151D0,0.000006D0,0.000367D0,           
     &0.081967D0,4*0D0,0.198776D0,0.010206D0,0.000003D0,0.010207D0,     
     &0.19839D0,0.000151D0,0.000006D0,0.000367D0,0.081893D0,0.198776D0, 
     &0.010206D0,0.000003D0,0.010207D0,0.19839D0,0.000151D0,0.000006D0, 
     &0.000367D0,0.081893D0,4*0D0,0.199344D0,0.010234D0,0.000003D0/     
      DATA (BRAT(I)  ,I=4151,4281)/0.010236D0,0.198928D0,0.000149D0,    
     &0.000006D0,0.000368D0,0.080733D0,0.199344D0,0.010234D0,           
     &0.000003D0,0.010236D0,0.198928D0,0.000149D0,0.000006D0,           
     &0.000368D0,0.080733D0,4*0D0,0.184738D0,0.104588D0,0.184738D0,     
     &0.104587D0,0.184731D0,0.09582D0,0.022902D0,0.008429D0,0.015602D0, 
     &0.022902D0,0.008429D0,0.015602D0,0.022902D0,0.008429D0,           
     &0.015602D0,0.28959D0,0.01487D0,0.000008D0,0.01487D0,0.289061D0,   
     &0.000492D0,0.000009D0,0.000536D0,0.27911D0,2*0.037151D0,          
     &0.03715D0,0.090266D0,2*0.001805D0,0.090266D0,0.001805D0,          
     &0.812263D0,0.00179D0,0.090428D0,0.001809D0,0.001808D0,0.090428D0, 
     &0.001808D0,0.81372D0,0D0,6*1D0,0.095602D0,2*0.338272D0,           
     &0.156896D0,0.019193D0,0.017993D0,0.001168D0,0.001462D0,           
     &0.009608D0,0.003306D0,0.002132D0,0.003127D0,0.002132D0,           
     &0.003127D0,0.00213D0,3*0D0,0.001411D0,0.00045D0,0.001411D0,       
     &0.00045D0,0.001411D0,0.00045D0,2*0D0,0.097996D0,0.399787D0,       
     &0.262464D0,0.185427D0,0.022683D0,0.007648D0,0.004259D0,           
     &0.005925D0,0.000304D0,2*0D0,0.000304D0,0.005914D0,0.000002D0,     
     &2*0D0,0.000011D0,0.001258D0,5*0D0,3*0.002005D0,0D0,0.272178D0,    
     &0.022112D0,0.255165D0,0.015534D0,2*0.108965D0,0.031557D0,         
     &0.005562D0,0.044965D0,0.004674D0,0.007637D0,0.020597D0/           
      DATA (BRAT(I)  ,I=4282,8000)/0.007636D0,0.020595D0,0.007616D0,    
     &3*0D0,0.017298D0,0.004782D0,0.017298D0,0.004782D0,0.017297D0,     
     &0.004782D0,2*0D0,0.055332D0,2*0.319757D0,0.121576D0,2*0.001556D0, 
     &4*0D0,0.0277D0,0.021481D0,0.027699D0,0.021477D0,0.027658D0,3*0D0, 
     &0.006071D0,0.01208D0,0.006071D0,0.01208D0,0.006069D0,0.01208D0,   
     &2*0D0,0.035891D0,0.209476D0,0.129084D0,0.286631D0,0.10742D0,      
     &0.109486D0,4*0D0,0.035282D0,0.001812D0,2*0D0,0.001812D0,          
     &0.035215D0,0.000021D0,0D0,0.000001D0,0.000065D0,0.011965D0,5*0D0, 
     &2*0.011947D0,0.011946D0,0D0,
     &649*0.D0,
C....UED
     &0.001D0,0.999D0,0.001D0,0.999D0,0.001D0,0.999D0,
     &0.001D0,0.999D0,0.001D0,0.999D0,0.001D0,0.999D0, 
     &0.33D0,0.66D0,0.01D0,0.33D0,0.66D0,0.01D0,0.33D0,0.66D0,0.01D0,
     &0.33D0,0.66D0,0.01D0,0.98D0,0.D0,0.02D0,0.33D0,0.66D0,0.01D0,
     &9*1.D0,              
     &24*0.0416667,        
     &1.,                  
     &3*0.D0,6*0.08333D0, 
     &3*0.D0,6*0.08333D0,
     &6*0.166667D0,        
     &2912*0.D0/
      DATA (KFDP(I,1),I=   1, 377)/21,22,23,4*-24,25,21,22,23,4*24,25,  
     &21,22,23,4*-24,25,21,22,23,4*24,25,21,22,23,4*-24,25,21,22,23,    
     &4*24,25,37,1000022,1000023,1000025,1000035,1000021,1000039,21,22, 
     &23,4*-24,25,2*-37,21,22,23,4*24,25,2*37,22,23,-24,25,23,24,-12,   
     &22,23,-24,25,23,24,-12,-14,48*16,22,23,-24,25,23,24,22,23,-24,25, 
     &-37,23,24,37,1,2,3,4,5,6,7,8,21,1,2,3,4,5,6,7,8,11,13,15,17,1,2,  
     &3,4,5,6,7,8,11,12,13,14,15,16,17,18,4*-1,4*-3,4*-5,4*-7,-11,-13,  
     &-15,-17,1,2,3,4,5,6,7,8,11,13,15,17,21,2*22,23,24,1000022,        
     &2*1000023,3*1000025,4*1000035,2*1000024,2*1000037,1000001,        
     &2000001,1000001,-1000001,1000002,2000002,1000002,-1000002,        
     &1000003,2000003,1000003,-1000003,1000004,2000004,1000004,         
     &-1000004,1000005,2000005,1000005,-1000005,1000006,2000006,        
     &1000006,-1000006,1000011,2000011,1000011,-1000011,1000012,        
     &2000012,1000012,-1000012,1000013,2000013,1000013,-1000013,        
     &1000014,2000014,1000014,-1000014,1000015,2000015,1000015,         
     &-1000015,1000016,2000016,1000016,-1000016,1,2,3,4,5,6,7,8,11,12,  
     &13,14,15,16,17,18,24,37,2*23,25,35,4*-1,4*-3,4*-5,4*-7,-11,-13,   
     &-15,-17,3*24,1,2,3,4,5,6,7,8,11,13,15,17,21,2*22,23,24,23,25,24,  
     &37,23,25,36,1000022,2*1000023,3*1000025,4*1000035,2*1000024,      
     &2*1000037,1000001,2000001,1000001,-1000001,1000002,2000002/       
      DATA (KFDP(I,1),I= 378, 580)/1000002,-1000002,1000003,2000003,    
     &1000003,-1000003,1000004,2000004,1000004,-1000004,1000005,        
     &2000005,1000005,-1000005,1000006,2000006,1000006,-1000006,        
     &1000011,2000011,1000011,-1000011,1000012,2000012,1000012,         
     &-1000012,1000013,2000013,1000013,-1000013,1000014,2000014,        
     &1000014,-1000014,1000015,2000015,1000015,-1000015,1000016,        
     &2000016,1000016,-1000016,1,2,3,4,5,6,7,8,11,13,15,17,21,2*22,23,  
     &24,23,25,24,37,1000022,2*1000023,3*1000025,4*1000035,2*1000024,   
     &2*1000037,1000001,2000001,1000001,-1000001,1000002,2000002,       
     &1000002,-1000002,1000003,2000003,1000003,-1000003,1000004,        
     &2000004,1000004,-1000004,1000005,2000005,1000005,-1000005,        
     &1000006,2000006,1000006,-1000006,1000011,2000011,1000011,         
     &-1000011,1000012,2000012,1000012,-1000012,1000013,2000013,        
     &1000013,-1000013,1000014,2000014,1000014,-1000014,1000015,        
     &2000015,1000015,-1000015,1000016,2000016,1000016,-1000016,-1,-3,  
     &-5,-7,-11,-13,-15,-17,24,2*1000022,2*1000023,2*1000025,2*1000035, 
     &1000006,2000006,1000006,2000006,-1000001,-1000003,-1000011,       
     &-1000013,-1000015,-2000015,1,2,3,4,5,6,11,13,15,2,82,-11,-13,2*2, 
     &-12,-14,-16,2*-2,2*-4,-2,-4,2*22,211,111,221,13,11,213,-213,221,  
     &223,321,130,310,111,331,111,211,-12,12,-14,14,211,111,22,-13,-11/ 
      DATA (KFDP(I,1),I= 581, 992)/2*211,213,113,221,223,321,211,331,   
     &22,111,211,2*22,211,22,111,211,22,211,221,111,11,211,111,2*211,   
     &321,130,310,221,111,211,111,130,310,321,2*311,321,311,323,313,    
     &323,313,321,3*311,-13,3*211,12,14,311,2*321,311,321,313,323,313,  
     &323,311,4*321,211,111,3*22,111,321,130,-213,113,213,211,22,111,   
     &11,13,211,321,130,310,221,211,111,11*-11,11*-13,-311,-313,-311,   
     &-313,-20313,2*-311,-313,-311,-313,2*111,2*221,2*331,2*113,2*223,  
     &2*333,-311,-313,2*-321,211,-311,-321,333,-311,-313,-321,211,      
     &2*-321,2*-311,-321,211,113,421,2*411,421,411,423,413,423,413,421, 
     &411,8*-11,8*-13,-321,-323,-321,-323,-311,2*-313,-311,-313,2*-311, 
     &-321,-10323,-321,-323,-321,-311,2*-313,211,111,333,3*-321,-311,   
     &-313,-321,-313,310,333,211,2*-321,-311,-313,-311,211,-321,3*-311, 
     &211,113,321,2*421,411,421,413,423,413,423,411,421,-15,5*-11,      
     &5*-13,221,331,333,221,331,333,10221,211,213,211,213,321,323,321,  
     &323,2212,221,331,333,221,2*2,2*431,421,411,423,413,82,11,13,82,   
     &443,82,6*12,6*14,2*16,3*-411,3*-413,2*-411,2*-413,2*441,2*443,    
     &2*20443,2*2,2*4,2,4,511,521,511,523,513,523,513,521,511,6*12,     
     &6*14,2*16,3*-421,3*-423,2*-421,2*-423,2*441,2*443,2*20443,2*2,    
     &2*4,2,4,521,511,521,513,523,513,523,511,521,6*12,6*14,2*16,       
     &3*-431,3*-433,2*-431,2*-433,3*441,3*443,3*20443,2*2,2*4,2,4,531/  
      DATA (KFDP(I,1),I= 993,1402)/521,511,523,513,16,2*4,2*12,2*14,    
     &2*16,4*2,4*4,2*-11,2*-13,2*-1,2*-3,2*-11,2*-13,2*-1,541,511,521,  
     &513,523,21,11,13,15,1,2,3,4,21,22,553,21,2112,2212,2*2112,2212,   
     &2112,2*2212,2112,-12,3122,3212,3112,2212,2*2112,-12,2*3122,3222,  
     &3112,2212,2112,2212,3122,3222,3212,3122,3112,-12,-14,-12,3322,    
     &3312,2*3122,3212,3322,3312,3122,3322,3312,-12,2*4122,7*-11,7*-13, 
     &2*2224,2*2212,2*2214,2*3122,2*3212,2*3214,5*3222,4*3224,2*3322,   
     &3324,2*2224,7*2212,5*2214,2*2112,2*2114,2*3122,2*3212,2*3214,     
     &2*3222,2*3224,4*2,3,2*2,1,2*2,-11,-13,2*2,4*4122,-11,-13,2*2,     
     &3*4132,3*4232,-11,-13,2*2,4332,-11,-13,2*2,-11,-13,2*2,-11,-13,   
     &2*2,-11,-13,2*2,-11,-13,2*2,-11,-13,2*2,-11,-13,2*2,2*5122,-12,   
     &-14,-16,5*4122,441,443,20443,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,    
     &2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,4*5122,-12,-14,-16,2*-2,   
     &2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,2*5132,2*5232,-12,-14,-16, 
     &2*-2,2*-4,-2,-4,5332,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,     
     &2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,     
     &2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,  
     &-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,   
     &-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,  
     &2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2/     
      DATA (KFDP(I,1),I=1403,1713)/2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2, 
     &-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,-12,   
     &-14,-16,2*-2,2*-4,-2,-4,-12,-14,-16,2*-2,2*-4,-2,-4,221,223,221,  
     &223,211,111,321,130,310,213,113,-213,321,311,321,311,323,313,     
     &2*311,321,311,321,313,323,321,211,111,321,130,310,2*211,313,-313, 
     &323,-323,421,411,423,413,411,421,413,423,411,421,423,413,443,     
     &2*82,521,511,523,513,511,521,513,523,521,511,523,513,511,521,513, 
     &523,553,2*21,213,-213,113,213,10211,10111,-10211,2*221,213,2*113, 
     &-213,2*321,2*311,113,323,2*313,323,313,-313,323,-323,423,2*413,   
     &2*423,413,443,82,523,2*513,2*523,2*513,523,553,21,11,13,82,4*443, 
     &10441,20443,445,441,11,13,15,1,2,3,4,21,22,2*553,10551,20553,555, 
     &1000039,-1000024,-1000037,1000022,1000023,1000025,1000035,        
     &1000002,2000002,1000002,2000002,1000021,3*-12,3*-14,3*-16,12,11,  
     &12,11,12,11,14,13,14,13,14,13,16,15,16,15,16,15,2*-2,2*-4,2*-6,   
     &1000039,1000024,1000037,1000022,1000023,1000025,1000035,1000001,  
     &2000001,1000001,2000001,1000021,3*-11,3*-13,3*-15,2*-1,-3,        
     &1000039,-1000024,-1000037,1000022,1000023,1000025,1000035,        
     &1000004,2000004,1000004,2000004,1000021,3*-12,3*-14,3*-16,12,11,  
     &12,11,12,11,14,13,14,13,14,13,16,15,16,15,16,15,2*-2,2*-4,2*-6,   
     &1000039,1000024,1000037,1000022,1000023,1000025,1000035,1000003/  
      DATA (KFDP(I,1),I=1714,1984)/2000003,1000003,2000003,1000021,     
     &3*-11,3*-13,3*-15,2*-1,-3,1000039,-1000024,-1000037,1000022,      
     &1000023,1000025,1000035,1000006,2000006,1000006,2000006,1000021,  
     &3*-12,3*-14,3*-16,12,11,12,11,12,11,14,13,14,13,14,13,16,15,16,   
     &15,16,15,2*-2,2*-4,2*-6,1000039,1000024,1000037,1000022,1000023,  
     &1000025,1000035,1000005,2000005,1000005,2000005,1000021,1000022,  
     &1000016,-1000015,3*-11,3*-13,3*-15,2*-1,-3,1000039,-1000024,      
     &-1000037,1000022,1000023,1000025,1000035,1000012,2000012,1000012, 
     &2*12,2*14,2*16,3*-14,3*-16,3*-2,3*-4,3*-6,1000039,1000024,        
     &1000037,1000022,1000023,1000025,1000035,1000011,2000011,1000011,  
     &2000011,3*-13,3*-15,3*-1,3*-3,3*-5,1000039,-1000024,-1000037,     
     &1000022,1000023,1000025,1000035,1000014,2000014,1000014,2000014,  
     &2*12,2*14,2*16,3*-12,3*-16,3*-2,3*-4,3*-6,1000039,1000024,        
     &1000037,1000022,1000023,1000025,1000035,1000013,2000013,1000013,  
     &2000013,3*-11,3*-15,3*-1,3*-3,3*-5,1000039,-1000024,-1000037,     
     &1000022,1000023,1000025,1000035,1000016,2000016,1000016,2000016,  
     &2*12,2*14,2*16,3*-12,3*-14,3*-2,3*-4,3*-6,1000039,1000024,        
     &1000037,1000022,1000023,1000025,1000035,1000015,2000015,1000015,  
     &2000015,3*-11,3*-13,3*-1,3*-3,3*-5,1000039,1000001,-1000001,      
     &2000001,-2000001,1000002,-1000002,2000002,-2000002,1000003/       
      DATA (KFDP(I,1),I=1985,2321)/-1000003,2000003,-2000003,1000004,   
     &-1000004,2000004,-2000004,1000005,-1000005,2000005,-2000005,      
     &1000006,-1000006,2000006,-2000006,6*1000022,6*1000023,6*1000025,  
     &6*1000035,1000024,-1000024,1000024,-1000024,1000024,-1000024,     
     &1000037,-1000037,1000037,-1000037,1000037,-1000037,-12,12,-11,11, 
     &-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,   
     &-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-14,14,-13,13,   
     &-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,   
     &-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-16,16,-15,15,   
     &-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,   
     &-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-2,2,-2,2,-2,2,  
     &-4,4,-4,4,-4,4,-6,6,-6,6,-6,6,5*1000039,4,1,-12,12,-12,12,-12,12, 
     &-12,12,-12,12,-12,12,-14,14,-14,14,-14,14,-14,14,-14,14,-14,14,   
     &-16,16,-16,16,-16,16,-16,16,-16,16,-16,16,-12,12,-11,11,-12,12,   
     &-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,   
     &-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-14,14,-13,13,-14,14,   
     &-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,   
     &-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-16,16,-15,15,-16,16,   
     &-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,   
     &-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-2,2,-2,2,-2,2,-4,4,-4/ 
      DATA (KFDP(I,1),I=2322,2573)/4,-4,4,-6,6,-6,6,-6,6,5*1000039,     
     &16*1000022,1000024,-1000024,1000024,-1000024,1000024,-1000024,    
     &1000024,-1000024,1000024,-1000024,1000024,-1000024,1000037,       
     &-1000037,1000037,-1000037,1000037,-1000037,1000037,-1000037,      
     &1000037,-1000037,1000037,-1000037,1000024,-1000024,1000037,       
     &-1000037,1000001,-1000001,2000001,-2000001,1000002,-1000002,      
     &2000002,-2000002,1000003,-1000003,2000003,-2000003,1000004,       
     &-1000004,2000004,-2000004,1000005,-1000005,2000005,-2000005,      
     &1000006,-1000006,2000006,-2000006,1000011,-1000011,2000011,       
     &-2000011,1000012,-1000012,2000012,-2000012,1000013,-1000013,      
     &2000013,-2000013,1000014,-1000014,2000014,-2000014,1000015,       
     &-1000015,2000015,-2000015,1000016,-1000016,2000016,-2000016,      
     &5*1000021,-12,12,-12,12,-12,12,-12,12,-12,12,-12,12,-14,14,-14,   
     &14,-14,14,-14,14,-14,14,-14,14,-16,16,-16,16,-16,16,-16,16,-16,   
     &16,-16,16,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,   
     &11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,   
     &12,-11,11,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,   
     &13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,   
     &14,-13,13,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,   
     &15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16/   
      DATA (KFDP(I,1),I=2574,2892)/16,-15,15,-2,2,-2,2,-2,2,-4,4,-4,4,  
     &-4,4,-6,6,-6,6,-6,6,2*1000039,6*1000022,6*1000023,6*1000025,      
     &6*1000035,1000022,1000023,1000025,1000035,1000002,2000002,        
     &-1000001,-2000001,1000004,2000004,-1000003,-2000003,1000006,      
     &2000006,-1000005,-2000005,1000012,2000012,-1000011,-2000011,      
     &1000014,2000014,-1000013,-2000013,1000016,2000016,-1000015,       
     &-2000015,2*1000021,-12,12,-11,-12,12,-11,-12,12,-11,-12,12,-11,   
     &-12,12,-11,-12,12,-11,-14,-13,-14,-13,-14,-13,-14,14,-13,-14,14,  
     &-13,-14,14,-13,-16,-15,-16,-15,-16,-15,-16,-15,-16,-15,-16,-15,   
     &-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-12,2*-11,12, 
     &-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-14,2*-13,14, 
     &-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-14,2*-13,14, 
     &-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-16,2*-15,16,-16,2*-15,16, 
     &-16,2*-15,16,-16,2*-15,16,-16,2*-15,16,-16,2*-15,16,-16,2*-15,16, 
     &-16,2*-15,16,-16,2*-15,16,2,-1,2,-1,2*2,-1,2,-1,3*2,-1,2*4,-3,    
     &3*4,-3,2*6,5*1000039,16*1000022,16*1000023,1000024,-1000024,      
     &1000024,-1000024,1000024,-1000024,1000024,-1000024,1000024,       
     &-1000024,1000024,-1000024,1000037,-1000037,1000037,-1000037,      
     &1000037,-1000037,1000037,-1000037,1000037,-1000037,1000037,       
     &-1000037,1000024,-1000024,1000037,-1000037,1000001,-1000001/      
      DATA (KFDP(I,1),I=2893,3182)/2000001,-2000001,1000002,-1000002,   
     &2000002,-2000002,1000003,-1000003,2000003,-2000003,1000004,       
     &-1000004,2000004,-2000004,1000005,-1000005,2000005,-2000005,      
     &1000006,-1000006,2000006,-2000006,1000011,-1000011,2000011,       
     &-2000011,1000012,-1000012,2000012,-2000012,1000013,-1000013,      
     &2000013,-2000013,1000014,-1000014,2000014,-2000014,1000015,       
     &-1000015,2000015,-2000015,1000016,-1000016,2000016,-2000016,      
     &5*1000021,-12,12,-12,12,-12,12,-12,12,-12,12,-12,12,-14,14,-14,   
     &14,-14,14,-14,14,-14,14,-14,14,-16,16,-16,16,-16,16,-16,16,-16,   
     &16,-16,16,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,   
     &11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,   
     &12,-11,11,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,   
     &13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,   
     &14,-13,13,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,   
     &15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,   
     &16,-15,15,-2,2,-2,2,-2,2,-4,4,-4,4,-4,4,-6,6,-6,6,-6,6,5*1000039, 
     &16*1000022,16*1000023,16*1000025,1000024,-1000024,1000024,        
     &-1000024,1000024,-1000024,1000024,-1000024,1000024,-1000024,      
     &1000024,-1000024,1000037,-1000037,1000037,-1000037,1000037,       
     &-1000037,1000037,-1000037,1000037,-1000037,1000037,-1000037/      
      DATA (KFDP(I,1),I=3183,3459)/1000024,-1000024,1000037,-1000037,   
     &1000001,-1000001,2000001,-2000001,1000002,-1000002,2000002,       
     &-2000002,1000003,-1000003,2000003,-2000003,1000004,-1000004,      
     &2000004,-2000004,1000005,-1000005,2000005,-2000005,1000006,       
     &-1000006,2000006,-2000006,1000011,-1000011,2000011,-2000011,      
     &1000012,-1000012,2000012,-2000012,1000013,-1000013,2000013,       
     &-2000013,1000014,-1000014,2000014,-2000014,1000015,-1000015,      
     &2000015,-2000015,1000016,-1000016,2000016,-2000016,5*1000021,-12, 
     &12,-12,12,-12,12,-12,12,-12,12,-12,12,-14,14,-14,14,-14,14,-14,   
     &14,-14,14,-14,14,-16,16,-16,16,-16,16,-16,16,-16,16,-16,16,-12,   
     &12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,   
     &11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-12,12,-11,11,-14,   
     &14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,   
     &13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-14,14,-13,13,-16,   
     &16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,   
     &15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-16,16,-15,15,-2,2,  
     &-2,2,-2,2,-4,4,-4,4,-4,4,-6,6,-6,6,-6,6,2*1000039,15*1000024,     
     &6*1000022,6*1000023,6*1000025,6*1000035,1000022,1000023,1000025,  
     &1000035,1000002,2000002,-1000001,-2000001,1000004,2000004,        
     &-1000003,-2000003,1000006,2000006,-1000005,-2000005,1000012/      
      DATA (KFDP(I,1),I=3460,3782)/2000012,-1000011,-2000011,1000014,   
     &2000014,-1000013,-2000013,1000016,2000016,-1000015,-2000015,      
     &2*1000021,-12,12,-11,-12,12,-11,-12,12,-11,-12,12,-11,-12,12,-11, 
     &-12,12,-11,-14,14,-13,-14,14,-13,-14,14,-13,-14,14,-13,-14,14,    
     &-13,-14,14,-13,-16,16,-15,-16,16,-15,-16,16,-15,-16,16,-15,-16,   
     &16,-15,-16,16,-15,-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-12,     
     &2*-11,12,-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-12,2*-11,12,-12, 
     &2*-11,12,-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-14, 
     &2*-13,14,-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-14,2*-13,14,-16, 
     &2*-15,16,-16,2*-15,16,-16,2*-15,16,-16,2*-15,16,-16,2*-15,16,-16, 
     &2*-15,16,-16,2*-15,16,-16,2*-15,16,-16,2*-15,16,2,-1,2,-1,2*2,-1, 
     &2,-1,3*2,-1,2*4,-3,3*4,-3,2*6,1000039,-1000024,-1000037,1000022,  
     &1000023,1000025,1000035,4*1000001,1000002,2000002,1000002,        
     &2000002,1000021,3*-12,3*-14,3*-16,12,11,12,11,12,11,14,13,14,13,  
     &14,13,16,15,16,15,16,15,2*-2,2*-4,2*-6,1000039,1000024,1000037,   
     &1000022,1000023,1000025,1000035,4*1000002,1000001,2000001,        
     &1000001,2000001,1000021,3*-11,3*-13,3*-15,2*-1,-3,1000039,        
     &-1000024,-1000037,1000022,1000023,1000025,1000035,4*1000003,      
     &1000004,2000004,1000004,2000004,1000021,3*-12,3*-14,3*-16,12,11,  
     &12,11,12,11,14,13,14,13,14,13,16,15,16,15,16,15,2*-2,2*-4,2*-6/   
      DATA (KFDP(I,1),I=3783,4156)/1000039,1000024,1000037,1000022,     
     &1000023,1000025,1000035,4*1000004,1000003,2000003,1000003,        
     &2000003,1000021,3*-11,3*-13,3*-15,2*-1,-3,1000039,-1000024,       
     &-1000037,1000022,1000023,1000025,1000035,4*1000005,1000006,       
     &2000006,1000006,2000006,1000021,3*-12,3*-14,3*-16,12,11,12,11,12, 
     &11,14,13,14,13,14,13,16,15,16,15,16,15,2*-2,2*-4,2*-6,1000039,    
     &1000024,1000037,1000022,1000023,1000025,1000035,4*1000006,        
     &1000005,2000005,1000005,2000005,1000021,3*-11,3*-13,3*-15,2*-1,   
     &-3,1000039,-1000024,-1000037,1000022,1000023,1000025,1000035,     
     &4*1000011,1000012,2000012,1000012,2000012,2*12,2*14,2*16,3*-14,   
     &3*-16,3*-2,3*-4,3*-6,1000039,-1000024,-1000037,1000022,1000023,   
     &1000025,1000035,4*1000013,1000014,2000014,1000014,2000014,2*12,   
     &2*14,2*16,3*-12,3*-16,3*-2,3*-4,3*-6,1000039,-1000024,-1000037,   
     &1000022,1000023,1000025,1000035,4*1000015,1000016,2000016,        
     &1000016,2000016,2*12,2*14,2*16,3*-12,3*-14,3*-2,3*-4,3*-6,3,4,5,  
     &6,11,13,15,21,2*4,2,4,24,-11,-13,-15,3,4,5,6,11,13,15,21,5,6,21,  
     &1,2,3,4,5,6,1,2,3,4,5,6,21,1,2,3,4,5,6,21,1,2,3,4,5,6,21,1,2,3,4, 
     &5,6,1,2,3,4,5,6,1,2,3,4,5,6,21,3100111,3200111,21,22,23,-24,21,   
     &22,23,24,22,23,-24,23,24,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18, 
     &21,22,23,24,9*11,9*-11,11,-11,11,-11,9*13,9*-13,13,-13,13,-13,
     &9*15/     
      DATA (KFDP(I,1),I=4157,8000)/9*-15,15,-15,15,-15,1,2,3,4,5,6,11,
     &12,9900012,13,14,9900014,15,16,9900016,3*-1,3*-3,3*-5,-11,-13,-15,   
     &3*-11,2*-13,-15,24,3*-11,2*-13,-15,9900024,3*443,3*553,2*24,      
     &2*3000211,2*22,2*23,22,23,1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,   
     &18,2*24,3*3000211,2*24,4*-1,4*-3,4*-5,4*-7,-11,-13,-15,-17,22,23, 
     &22,23,24,3000211,24,3000211,22,23,1,2,3,4,5,6,7,8,11,12,13,14,15, 
     &16,17,18,2*24,-24,23,2*22,24,-24,2*23,1,2,3,4,5,6,7,8,11,12,13,   
     &14,15,16,17,18,2*22,23,2*24,23,22,2*24,23,4*-1,4*-3,4*-5,4*-7,    
     &-11,-13,-15,-17,
     &649*0,
C...UED
     &5100023,5100022,5100023,5100022,5100023,5100022,
     &5100023,5100022,5100023,5100022,5100023,5100022, 
     &5100023,-5100024,5100022,5100023,5100024,5100022,
     &5100023,-5100024,5100022,5100023,5100024,5100022,
     &5100023,-5100024,5100022,5100023,5100024,5100022, 
     &9*5100022, 
     &6100001,6100002,6100003,6100004,6100005,6100006,
     &5100001,5100002,5100003,5100004,5100005,5100006,
     &-6100001,-6100002,-6100003,-6100004,-6100005,-6100006,
     &-5100001,-5100002,-5100003,-5100004,-5100005,-5100006, 
     &39, 
     &6100011,6100013,6100015,
     &5100011,5100013,5100015,
     %5100012,5100014,5100016,
     &-6100011,-6100013,-6100015,
     &-5100011,-5100013,-5100015,
     %-5100012,-5100014,-5100016,
     &-5100011,-5100013,-5100015,
     &5100012,5100014,5100016,
     &2912*0/
      DATA (KFDP(I,2),I=   1, 339)/3*1,2,4,6,8,1,3*2,1,3,5,7,2,3*3,2,4, 
     &6,8,3,3*4,1,3,5,7,4,3*5,2,4,6,8,5,3*6,1,3,5,7,6,5,6*1000006,3*7,  
     &2,4,6,8,7,4,6,3*8,1,3,5,7,8,5,7,2*11,12,11,12,2*11,2*13,14,13,14, 
     &13,11,13,-211,-213,-211,-213,-211,-213,-211,-213,2*-211,-321,     
     &-323,-321,2*-323,3*-321,4*-211,-213,-211,-213,-211,-213,-211,     
     &-213,-211,-213,3*-211,-213,4*-211,-323,-321,2*-211,2*-321,3*-211, 
     &2*15,16,15,16,15,2*17,18,17,2*18,2*17,-1,-2,-3,-4,-5,-6,-7,-8,21, 
     &-1,-2,-3,-4,-5,-6,-7,-8,-11,-13,-15,-17,-1,-2,-3,-4,-5,-6,-7,-8,  
     &-11,-12,-13,-14,-15,-16,-17,-18,2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8,  
     &12,14,16,18,-1,-2,-3,-4,-5,-6,-7,-8,-11,-13,-15,-17,21,22,2*23,   
     &-24,2*1000022,1000023,1000022,1000023,1000025,1000022,1000023,    
     &1000025,1000035,-1000024,-1000037,-1000024,-1000037,-1000001,     
     &2*-2000001,2000001,-1000002,2*-2000002,2000002,-1000003,          
     &2*-2000003,2000003,-1000004,2*-2000004,2000004,-1000005,          
     &2*-2000005,2000005,-1000006,2*-2000006,2000006,-1000011,          
     &2*-2000011,2000011,-1000012,2*-2000012,2000012,-1000013,          
     &2*-2000013,2000013,-1000014,2*-2000014,2000014,-1000015,          
     &2*-2000015,2000015,-1000016,2*-2000016,2000016,-1,-2,-3,-4,-5,-6, 
     &-7,-8,-11,-12,-13,-14,-15,-16,-17,-18,-24,-37,22,25,2*36,2,4,6,8, 
     &2,4,6,8,2,4,6,8,2,4,6,8,12,14,16,18,23,22,25,-1,-2,-3,-4,-5,-6/   
      DATA (KFDP(I,2),I= 340, 533)/-7,-8,-11,-13,-15,-17,21,22,2*23,    
     &-24,2*25,-37,-24,3*36,2*1000022,1000023,1000022,1000023,1000025,  
     &1000022,1000023,1000025,1000035,-1000024,-1000037,-1000024,       
     &-1000037,-1000001,2*-2000001,2000001,-1000002,2*-2000002,2000002, 
     &-1000003,2*-2000003,2000003,-1000004,2*-2000004,2000004,-1000005, 
     &2*-2000005,2000005,-1000006,2*-2000006,2000006,-1000011,          
     &2*-2000011,2000011,-1000012,2*-2000012,2000012,-1000013,          
     &2*-2000013,2000013,-1000014,2*-2000014,2000014,-1000015,          
     &2*-2000015,2000015,-1000016,2*-2000016,2000016,-1,-2,-3,-4,-5,-6, 
     &-7,-8,-11,-13,-15,-17,21,22,2*23,-24,2*25,-37,-24,2*1000022,      
     &1000023,1000022,1000023,1000025,1000022,1000023,1000025,1000035,  
     &-1000024,-1000037,-1000024,-1000037,-1000001,2*-2000001,2000001,  
     &-1000002,2*-2000002,2000002,-1000003,2*-2000003,2000003,-1000004, 
     &2*-2000004,2000004,-1000005,2*-2000005,2000005,-1000006,          
     &2*-2000006,2000006,-1000011,2*-2000011,2000011,-1000012,          
     &2*-2000012,2000012,-1000013,2*-2000013,2000013,-1000014,          
     &2*-2000014,2000014,-1000015,2*-2000015,2000015,-1000016,          
     &2*-2000016,2000016,2,4,6,8,12,14,16,18,25,1000024,1000037,        
     &1000024,1000037,1000024,1000037,1000024,1000037,2*-1000005,       
     &2*-2000005,1000002,1000004,1000012,1000014,2*1000016,-3,-4,-5,-6/ 
      DATA (KFDP(I,2),I= 534, 938)/-7,-8,-13,-15,-17,11,-82,12,14,-1,   
     &-3,11,13,15,1,4,3,4,1,3,22,11,-211,2*22,-13,-11,-211,211,111,211, 
     &-321,130,310,22,2*111,-211,11,-11,13,-13,-211,111,22,14,12,111,   
     &22,111,3*211,-311,22,211,22,111,-211,211,11,-211,13,22,-211,111,  
     &-211,22,111,-11,-211,111,2*-211,-321,130,310,221,111,-211,111,    
     &2*0,-211,111,22,-211,111,-211,111,-211,211,-213,113,223,221,14,   
     &111,211,111,-11,-13,211,111,22,211,111,211,111,2*211,213,113,223, 
     &221,22,-211,111,113,223,22,111,-321,310,211,111,2*-211,221,22,    
     &-11,-13,-211,-321,130,310,221,-211,111,11*12,11*14,2*211,2*213,   
     &211,20213,2*321,2*323,211,213,211,213,211,213,211,213,211,213,    
     &211,213,3*211,213,211,2*321,8*211,2*113,3*211,111,22,211,111,211, 
     &111,4*211,8*12,8*14,2*211,2*213,2*111,221,2*113,223,333,20213,    
     &211,2*321,323,2*311,313,-211,111,113,2*211,321,2*211,311,321,310, 
     &211,-211,4*211,321,4*211,113,2*211,-321,111,22,-211,111,-211,111, 
     &-211,211,-211,211,16,5*12,5*14,3*211,3*213,211,2*111,2*113,       
     &2*-311,2*-313,-2112,3*321,323,2*-1,22,111,321,311,321,311,-82,    
     &-11,-13,-82,22,-82,6*-11,6*-13,2*-15,211,213,20213,211,213,20213, 
     &431,433,431,433,311,313,311,313,311,313,-1,-4,-3,-4,-1,-3,22,     
     &-211,111,-211,111,-211,211,-211,211,6*-11,6*-13,2*-15,211,213,    
     &20213,211,213,20213,431,433,431,433,321,323,321,323,321,323,-1/   
      DATA (KFDP(I,2),I= 939,1352)/-4,-3,-4,-1,-3,22,211,111,211,111,   
     &4*211,6*-11,6*-13,2*-15,211,213,20213,211,213,20213,431,433,431,  
     &433,221,331,333,221,331,333,221,331,333,-1,-4,-3,-4,-1,-3,22,     
     &-321,-311,-321,-311,-15,-3,-1,2*-11,2*-13,2*-15,-1,-4,-3,-4,-3,   
     &-4,-1,-4,2*12,2*14,2,3,2,3,2*12,2*14,2,1,22,411,421,411,421,21,   
     &-11,-13,-15,-1,-2,-3,-4,2*21,22,21,2*-211,111,22,111,211,22,211,  
     &-211,11,2*-211,111,-211,111,22,11,22,111,-211,211,111,211,22,211, 
     &111,211,-211,22,11,13,11,-211,2*111,2*22,111,211,-321,-211,111,   
     &11,2*-211,7*12,7*14,-321,-323,-311,-313,-311,-313,211,213,211,    
     &213,211,213,111,221,331,113,223,111,221,113,223,321,323,321,-211, 
     &-213,111,221,331,113,223,333,10221,111,221,331,113,223,211,213,   
     &211,213,321,323,321,323,321,323,311,313,311,313,2*-1,-3,-1,2203,  
     &3201,3203,2203,2101,2103,12,14,-1,-3,2*111,2*211,12,14,-1,-3,22,  
     &111,2*22,111,22,12,14,-1,-3,22,12,14,-1,-3,12,14,-1,-3,12,14,-1,  
     &-3,12,14,-1,-3,12,14,-1,-3,12,14,-1,-3,12,14,-1,-3,2*-211,11,13,  
     &15,-211,-213,-20213,-431,-433,3*3122,1,4,3,4,1,3,11,13,15,1,4,3,  
     &4,1,3,11,13,15,1,4,3,4,1,3,2*111,2*211,11,13,15,1,4,3,4,1,3,11,   
     &13,15,1,4,3,4,1,3,4*22,11,13,15,1,4,3,4,1,3,22,11,13,15,1,4,3,4,  
     &1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1, 
     &3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3/ 
      DATA (KFDP(I,2),I=1353,1815)/11,13,15,1,4,3,4,1,3,11,13,15,1,4,3, 
     &4,1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4, 
     &1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1, 
     &3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3,11,13,15,1,4,3,4,1,3, 
     &2*111,2*211,-211,111,-321,130,310,-211,111,211,-211,111,-213,113, 
     &-211,111,223,211,111,213,113,211,111,223,-211,111,-321,130,310,   
     &2*-211,-311,311,-321,321,211,111,211,111,-211,111,-211,111,311,   
     &2*321,311,22,2*-82,-211,111,-211,111,211,111,211,111,-321,-311,   
     &-321,-311,411,421,411,421,22,2*21,-211,2*211,111,-211,111,2*211,  
     &111,-211,211,111,211,-321,2*-311,-321,22,-211,111,211,111,-311,   
     &311,-321,321,211,111,-211,111,321,311,22,-82,-211,111,211,111,    
     &-321,-311,411,421,22,21,-11,-13,-82,211,111,221,111,4*22,-11,-13, 
     &-15,-1,-2,-3,-4,2*21,211,111,3*22,1,2*2,4*1,2*-24,2*-37,2*1,3,5,  
     &1,3,5,1,3,5,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,-3,-5,-3,-5,-3,   
     &-5,2,2*1,4*2,2*24,2*37,2,1,3,5,1,3,5,1,3,5,-3,2*-5,3,2*4,4*3,     
     &2*-24,2*-37,3,1,3,5,1,3,5,1,3,5,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,  
     &5,6,-1,-5,-1,-5,-1,-5,4,2*3,4*4,2*24,2*37,4,1,3,5,1,3,5,1,3,5,-3, 
     &2*-5,5,2*6,4*5,2*-24,2*-37,5,1,3,5,1,3,5,1,3,5,1,2,3,4,5,6,1,2,3, 
     &4,5,6,1,2,3,4,5,6,-1,-3,-1,-3,-1,-3,6,2*5,4*6,2*24,2*37,6,4,-15,  
     &16,1,3,5,1,3,5,1,3,5,-3,2*-5,11,2*12,4*11,2*-24,-37,13,15,11,15/  
      DATA (KFDP(I,2),I=1816,2317)/11,13,11,13,15,11,13,15,1,3,5,1,3,5, 
     &1,3,5,12,2*11,4*12,2*24,2*37,11,13,15,11,13,15,1,3,5,1,3,5,1,3,5, 
     &13,2*14,4*13,2*-24,2*-37,13,15,11,15,11,13,11,13,15,11,13,15,1,3, 
     &5,1,3,5,1,3,5,14,2*13,4*14,2*24,2*37,11,13,15,11,13,15,1,3,5,1,3, 
     &5,1,3,5,15,2*16,4*15,2*-24,2*-37,13,15,11,15,11,13,11,13,15,11,   
     &13,15,1,3,5,1,3,5,1,3,5,16,2*15,4*16,2*24,2*37,11,13,15,11,13,15, 
     &1,3,5,1,3,5,1,3,5,21,-1,1,-1,1,-2,2,-2,2,-3,3,-3,3,-4,4,-4,4,-5,  
     &5,-5,5,-6,6,-6,6,1,3,5,2,4,6,1,3,5,2,4,6,1,3,5,2,4,6,1,3,5,2,4,6, 
     &1,-1,3,-3,5,-5,1,-1,3,-3,5,-5,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3, 
     &-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-2,2, 
     &-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5, 
     &-6,6,-5,5,-6,6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4, 
     &-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-1,1,-3,3,-1,1,-1,1, 
     &-3,3,-1,1,-1,1,-3,3,22,23,25,35,36,-1,-3,-13,13,-13,13,-13,13,    
     &-15,15,-15,15,-15,15,-11,11,-11,11,-11,11,-15,15,-15,15,-15,15,   
     &-11,11,-11,11,-11,11,-13,13,-13,13,-13,13,-1,1,-2,2,-1,1,-2,2,-1, 
     &1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6, 
     &6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5, 
     &5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4, 
     &4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-1,1,-3/ 
      DATA (KFDP(I,2),I=2318,2770)/3,-1,1,-1,1,-3,3,-1,1,-1,1,-3,3,22,  
     &23,25,35,36,22,23,11,13,15,12,14,16,1,3,5,2,4,25,35,36,-24,24,11, 
     &-11,13,-13,15,-15,1,-1,3,-3,-24,24,11,-11,13,-13,15,-15,1,-1,3,   
     &-3,-37,37,-37,37,-1,1,-1,1,-2,2,-2,2,-3,3,-3,3,-4,4,-4,4,-5,5,-5, 
     &5,-6,6,-6,6,-11,11,-11,11,-12,12,-12,12,-13,13,-13,13,-14,14,-14, 
     &14,-15,15,-15,15,-16,16,-16,16,1,3,5,2,4,-13,13,-13,13,-13,13,    
     &-15,15,-15,15,-15,15,-11,11,-11,11,-11,11,-15,15,-15,15,-15,15,   
     &-11,11,-11,11,-11,11,-13,13,-13,13,-13,13,-1,1,-2,2,-1,1,-2,2,-1, 
     &1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6, 
     &6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5, 
     &5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4, 
     &4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-1,1,-3, 
     &3,-1,1,-1,1,-3,3,-1,1,-1,1,-3,3,24,37,24,-11,-13,-15,-1,-3,24,    
     &-11,-13,-15,-1,-3,24,-11,-13,-15,-1,-3,24,-11,-13,-15,-1,-3,4*37, 
     &2*-1,2*2,2*-3,2*4,2*-5,2*6,2*-11,2*12,2*-13,2*14,2*-15,2*16,-1,   
     &-3,-13,14,2*-13,14,2*-13,14,-13,-15,16,2*-15,16,2*-15,16,-15,     
     &6*-11,-15,16,2*-15,16,2*-15,16,-15,6*-11,6*-13,-1,-2,-1,2,-1,-2,  
     &-1,2,-1,-2,-1,2,-3,-4,-3,4,-3,-4,-3,4,-3,-4,-3,4,-5,-6,-5,6,-5,   
     &-6,-5,6,-5,-6,-5,6,-1,-2,-1,2,-1,-2,-1,2,-1,-2,-1,2,-3,-4,-3,4,   
     &-3,-4,-3,4,-3,-4,-3,4,-5,-6,-5,6,-5,-6,-5,6,-5,-6,-5,6,-1,-2,-1/  
      DATA (KFDP(I,2),I=2771,3221)/2,-1,-2,-1,2,-1,-2,-1,2,-3,-4,-3,4,  
     &-3,-4,-3,4,-3,-4,-3,4,-5,-6,-5,6,-5,-6,-5,6,-5,-6,-5,6,2,-1,2,-1, 
     &2*4,-3,4,-3,3*6,-5,2*4,-3,3*6,-5,2*6,22,23,25,35,36,22,23,11,13,  
     &15,12,14,16,1,3,5,2,4,25,35,36,22,23,11,13,15,12,14,16,1,3,5,2,4, 
     &25,35,36,-24,24,11,-11,13,-13,15,-15,1,-1,3,-3,-24,24,11,-11,13,  
     &-13,15,-15,1,-1,3,-3,-37,37,-37,37,-1,1,-1,1,-2,2,-2,2,-3,3,-3,3, 
     &-4,4,-4,4,-5,5,-5,5,-6,6,-6,6,-11,11,-11,11,-12,12,-12,12,-13,13, 
     &-13,13,-14,14,-14,14,-15,15,-15,15,-16,16,-16,16,1,3,5,2,4,-13,   
     &13,-13,13,-13,13,-15,15,-15,15,-15,15,-11,11,-11,11,-11,11,-15,   
     &15,-15,15,-15,15,-11,11,-11,11,-11,11,-13,13,-13,13,-13,13,-1,1,  
     &-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6, 
     &-5,5,-6,6,-5,5,-6,6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3, 
     &-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-2,2,-1,1,-2,2, 
     &-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5, 
     &-6,6,-1,1,-1,1,-3,3,-1,1,-1,1,-3,3,-1,1,-1,1,-3,3,22,23,25,35,36, 
     &22,23,11,13,15,12,14,16,1,3,5,2,4,25,35,36,22,23,11,13,15,12,14,  
     &16,1,3,5,2,4,25,35,36,22,23,11,13,15,12,14,16,1,3,5,2,4,25,35,36, 
     &-24,24,11,-11,13,-13,15,-15,1,-1,3,-3,-24,24,11,-11,13,-13,15,    
     &-15,1,-1,3,-3,-37,37,-37,37,-1,1,-1,1,-2,2,-2,2,-3,3,-3,3,-4,4,   
     &-4,4,-5,5,-5,5,-6,6,-6,6,-11,11,-11,11,-12,12,-12,12,-13,13,-13/  
      DATA (KFDP(I,2),I=3222,3669)/13,-14,14,-14,14,-15,15,-15,15,-16,  
     &16,-16,16,1,3,5,2,4,-13,13,-13,13,-13,13,-15,15,-15,15,-15,15,    
     &-11,11,-11,11,-11,11,-15,15,-15,15,-15,15,-11,11,-11,11,-11,11,   
     &-13,13,-13,13,-13,13,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,  
     &3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-2,2,-1,1,-2, 
     &2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4,4,-5,5,-6,6,-5,5,-6,6,-5, 
     &5,-6,6,-1,1,-2,2,-1,1,-2,2,-1,1,-2,2,-3,3,-4,4,-3,3,-4,4,-3,3,-4, 
     &4,-5,5,-6,6,-5,5,-6,6,-5,5,-6,6,-1,1,-1,1,-3,3,-1,1,-1,1,-3,3,-1, 
     &1,-1,1,-3,3,24,37,23,11,13,15,12,14,16,1,3,5,2,4,25,35,36,24,-11, 
     &-13,-15,-1,-3,24,-11,-13,-15,-1,-3,24,-11,-13,-15,-1,-3,24,-11,   
     &-13,-15,-1,-3,4*37,2*-1,2*2,2*-3,2*4,2*-5,2*6,2*-11,2*12,2*-13,   
     &2*14,2*-15,2*16,-1,-3,-13,14,2*-13,14,2*-13,14,-13,-15,16,2*-15,  
     &16,2*-15,16,-15,-11,12,2*-11,12,2*-11,12,-11,-15,16,2*-15,16,     
     &2*-15,16,-15,-11,12,2*-11,12,2*-11,12,-11,-13,14,2*-13,14,2*-13,  
     &14,-13,-1,-2,-1,2,-1,-2,-1,2,-1,-2,-1,2,-3,-4,-3,4,-3,-4,-3,4,-3, 
     &-4,-3,4,-5,-6,-5,6,-5,-6,-5,6,-5,-6,-5,6,-1,-2,-1,2,-1,-2,-1,2,   
     &-1,-2,-1,2,-3,-4,-3,4,-3,-4,-3,4,-3,-4,-3,4,-5,-6,-5,6,-5,-6,-5,  
     &6,-5,-6,-5,6,-1,-2,-1,2,-1,-2,-1,2,-1,-2,-1,2,-3,-4,-3,4,-3,-4,   
     &-3,4,-3,-4,-3,4,-5,-6,-5,6,-5,-6,-5,6,-5,-6,-5,6,2,-1,2,-1,2*4,   
     &-3,4,-3,3*6,-5,2*4,-3,3*6,-5,2*6,1,2*2,4*1,23,25,35,36,2*-24/     
      DATA (KFDP(I,2),I=3670,4183)/2*-37,2*1,3,5,1,3,5,1,3,5,1,2,3,4,5, 
     &6,1,2,3,4,5,6,1,2,3,4,5,6,-3,-5,-3,-5,-3,-5,2,2*1,4*2,23,25,35,   
     &36,2*24,2*37,2,1,3,5,1,3,5,1,3,5,-3,2*-5,3,2*4,4*3,23,25,35,36,   
     &2*-24,2*-37,3,1,3,5,1,3,5,1,3,5,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,  
     &5,6,-1,-5,-1,-5,-1,-5,4,2*3,4*4,23,25,35,36,2*24,2*37,4,1,3,5,1,  
     &3,5,1,3,5,-3,2*-5,5,2*6,4*5,23,25,35,36,2*-24,2*-37,5,1,3,5,1,3,  
     &5,1,3,5,1,2,3,4,5,6,1,2,3,4,5,6,1,2,3,4,5,6,-1,-3,-1,-3,-1,-3,6,  
     &2*5,4*6,23,25,35,36,2*24,2*37,6,1,3,5,1,3,5,1,3,5,-3,2*-5,11,     
     &2*12,4*11,23,25,35,36,2*-24,2*-37,13,15,11,15,11,13,11,13,15,11,  
     &13,15,1,3,5,1,3,5,1,3,5,13,2*14,4*13,23,25,35,36,2*-24,2*-37,13,  
     &15,11,15,11,13,11,13,15,11,13,15,1,3,5,1,3,5,1,3,5,15,2*16,4*15,  
     &23,25,35,36,2*-24,2*-37,13,15,11,15,11,13,11,13,15,11,13,15,1,3,  
     &5,1,3,5,1,3,5,-3,-4,-5,-6,-11,-13,-15,21,-1,-3,2*-5,5,12,14,16,   
     &-3,-4,-5,-6,-11,-13,-15,21,-5,-6,21,-1,-2,-3,-4,-5,-6,-1,-2,-3,   
     &-4,-5,-6,21,-1,-2,-3,-4,-5,-6,21,-1,-2,-3,-4,-5,-6,21,-1,-2,-3,   
     &-4,-5,-6,-1,-2,-3,-4,-5,-6,-1,-2,-3,-4,-5,-6,3*21,3*1,4*2,1,2*11, 
     &2*12,11,-1,-2,-3,-4,-5,-6,-7,-8,-11,-12,-13,-14,-15,-16,-17,-18,  
     &21,22,23,-24,3*-1,3*-3,3*-5,3*1,3*3,3*5,-13,13,-15,15,3*-1,3*-3,     
     &3*-5,3*1,3*3,3*5,-11,11,-15,15,3*-1,3*-3,3*-5,3*1,3*3,3*5,-11,11,     
     &-13,13,-1,-2,-3,-4,-5,-6,-11,-12,9900012,-13,-14,9900014,-15,-16/   
      DATA (KFDP(I,2),I=4184,8000)/9900016,2,4,6,2,4,6,2,4,6,9900012,   
     &9900014,9900016,-11,-13,-15,-13,2*-15,24,-11,-13,-15,-13,2*-15,   
     &9900024,6*21,-24,-3000211,-24,-3000211,3000111,3000221,3000111,   
     &3000221,2*23,-1,-2,-3,-4,-5,-6,-7,-8,-11,-12,-13,-14,-15,-16,-17, 
     &-18,23,3000111,23,3000111,22,3000221,22,2,4,6,8,2,4,6,8,2,4,6,8,  
     &2,4,6,8,12,14,16,18,2*3000111,2*3000221,-3000211,2*-24,-3000211,  
     &2*23,-1,-2,-3,-4,-5,-6,-7,-8,-11,-12,-13,-14,-15,-16,-17,-18,-24, 
     &-3000211,3000211,3000221,3000113,3000223,-3000213,3000213,        
     &3000113,3000223,-1,-2,-3,-4,-5,-6,-7,-8,-11,-12,-13,-14,-15,-16,  
     &-17,-18,24,3000211,24,3000111,3000221,3000211,3000213,3000113,    
     &3000223,3000213,2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8,12,14,16,18,      
     &649*0,
C...UED     
     &1,1,2,2,3,3,4,4,5,5,6,6, 
     &1,2,1,2,1,2,3,4,3,4,3,4,5,6,5,6,5,6,
     &11,13,15,12,11,14,13,16,15, 
     &-1,-2,-3,-4,-5,-6,-1,-2,-3,-4,-5,-6,
     &1,2,3,4,5,6,1,2,3,4,5,6, 
     &22, 
     &-11,-13,-15,-11,-13,-15,-12,-14,-16,
     &11,13,15,11,13,15,12,14,16,
     &12,14,16,-11,-13,-15, 
     &2912*0/
      DATA (KFDP(I,3),I=   1,1021)/81*0,14,6*0,2*16,2*0,6*111,310,130,  
     &2*0,3*111,310,130,321,113,211,223,221,2*113,2*211,2*223,2*221,    
     &2*113,221,2*113,2*213,-213,113,2*111,310,130,310,130,2*310,130,   
     &402*0,4*3,4*4,1,4,3,2*2,0,-11,8*0,-211,5*0,2*111,211,-211,211,    
     &-211,10*0,111,4*0,2*111,-211,-11,11,-13,22,111,3*0,22,3*0,111,    
     &211,4*0,111,11*0,111,-211,6*0,-211,3*111,7*0,111,-211,5*0,2*221,  
     &3*0,111,5*0,111,11*0,-311,-313,-311,-321,-313,-323,111,221,331,   
     &113,223,-311,-313,-311,-321,-313,-323,111,221,331,113,223,22*0,   
     &111,113,2*211,-211,-311,211,111,3*211,-211,7*211,7*0,111,-211,    
     &111,-211,-321,-323,-311,-321,-313,-323,-211,-213,-321,-323,-311,  
     &-321,-313,-323,-211,-213,22*0,111,113,-311,2*-211,211,-211,310,   
     &-211,2*111,211,2*-211,-321,-211,2*211,-211,111,-211,2*211,6*0,    
     &111,-211,111,-211,0,221,331,333,321,311,221,331,333,321,311,20*0, 
     &3,13*0,-411,-413,-10413,-10411,-20413,-415,-411,-413,-10413,      
     &-10411,-20413,-415,-411,-413,16*0,-4,-1,-4,-3,2*-2,5*0,111,-211,  
     &111,-211,-421,-423,-10423,-10421,-20423,-425,-421,-423,-10423,    
     &-10421,-20423,-425,-421,-423,16*0,-4,-1,-4,-3,2*-2,5*0,111,-211,  
     &111,-211,-431,-433,-10433,-10431,-20433,-435,-431,-433,-10433,    
     &-10431,-20433,-435,-431,-433,19*0,-4,-1,-4,-3,2*-2,8*0,441,443,   
     &441,443,441,443,-4,-1,-4,-3,-4,-3,-4,-1,531,533,531,533,3,2,3,2/  
      DATA (KFDP(I,3),I=1022,2223)/511,513,511,513,1,2,13*0,2*21,11*0,  
     &2112,6*0,2212,12*0,2*3122,3212,10*0,3322,2*0,3122,3212,3214,2112, 
     &2114,2212,2112,3122,3212,3214,2112,2114,2212,2112,52*0,3*3,1,6*0, 
     &4*3,4*0,4*3,6*0,4*3,0,28*3,2*0,3*4122,8*0,4,1,4,3,2*2,4*4,1,4,3,  
     &2*2,4*4,1,4,3,2*2,4*0,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*0,4*4,1,4,3,  
     &2*2,0,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,    
     &4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,  
     &3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,    
     &4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,3,2*2,4*4,1,4,  
     &3,2*2,31*0,211,111,45*0,-211,2*111,-211,3*111,-211,111,211,30*0,  
     &-211,111,13*0,2*21,-211,111,199*0,2*5,210*0,-1,-3,-5,-2,-4,-6,-1, 
     &-3,-5,-2,-4,-6,-1,-3,-5,-2,-4,-6,-1,-3,-5,-2,-4,-6,-2,2,-4,4,-6,  
     &6,-2,2,-4,4,-6,6,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,  
     &-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5, 
     &-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5, 
     &-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1, 
     &-1,1,-1,3,-3,3,-3,5,-5,5,-5,-3,3,-5,5,-5,5,-3,3,-5,5,-5,5,-3,3,   
     &-5,5,-5,5,5*0,11,12,11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,11, 
     &-11,13,-13,15,-15,11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,11,   
     &-11,13,-13,15,-15,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3/ 
      DATA (KFDP(I,3),I=2224,2783)/-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,  
     &-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5, 
     &-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1, 
     &-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,-3,3,   
     &-5,5,-5,5,-3,3,-5,5,-5,5,-3,3,-5,5,-5,5,7*0,-11,-13,-15,-12,-14,  
     &-16,-1,-3,-5,-2,-4,5*0,-12,12,-14,14,-16,16,-2,2,-4,4,2*0,-12,12, 
     &-14,14,-16,16,-2,2,-4,4,52*0,-1,-3,-5,-2,-4,11,-11,13,-13,15,-15, 
     &11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,   
     &11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,1,-1,1,-1,3,-3,3,-3,5,  
     &-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5, 
     &-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1, 
     &-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1, 
     &-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,-3,3,-5,5,   
     &-5,5,-3,3,-5,5,-5,5,-3,3,-5,5,-5,5,3*0,12,14,16,2,4,0,12,14,16,2, 
     &4,0,12,14,16,2,4,0,12,14,16,2,4,28*0,2,4,12,-11,11,14,-13,13,16,  
     &-15,15,12,-11,11,14,-13,13,16,-15,15,12,11,14,13,16,15,12,-11,11, 
     &14,-13,13,16,-15,15,12,11,14,13,16,15,12,11,14,13,16,15,2*2,1,-1, 
     &2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,   
     &2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,   
     &2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1/   
      DATA (KFDP(I,3),I=2784,3354)/2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3, 
     &2*6,5,-5,3,-3,5,-5,1,3,-3,5,-5,1,3,5,-5,1,5,-5,1,3,5,-5,1,3,7*0,  
     &-11,-13,-15,-12,-14,-16,-1,-3,-5,-2,-4,5*0,-11,-13,-15,-12,-14,   
     &-16,-1,-3,-5,-2,-4,5*0,-12,12,-14,14,-16,16,-2,2,-4,4,2*0,-12,12, 
     &-14,14,-16,16,-2,2,-4,4,52*0,-1,-3,-5,-2,-4,11,-11,13,-13,15,-15, 
     &11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,   
     &11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,1,-1,1,-1,3,-3,3,-3,5,  
     &-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5, 
     &-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1, 
     &-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1, 
     &-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,-3,3,-5,5,   
     &-5,5,-3,3,-5,5,-5,5,-3,3,-5,5,-5,5,7*0,-11,-13,-15,-12,-14,-16,   
     &-1,-3,-5,-2,-4,5*0,-11,-13,-15,-12,-14,-16,-1,-3,-5,-2,-4,5*0,    
     &-11,-13,-15,-12,-14,-16,-1,-3,-5,-2,-4,5*0,-12,12,-14,14,-16,16,  
     &-2,2,-4,4,2*0,-12,12,-14,14,-16,16,-2,2,-4,4,52*0,-1,-3,-5,-2,-4, 
     &11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,   
     &11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,11,-11,13,-13,15,-15,1, 
     &-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1, 
     &-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3, 
     &-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,1,-1,1,-1,3,-3,3/ 
      DATA (KFDP(I,3),I=3355,8000)/-3,5,-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,  
     &-5,5,-5,1,-1,1,-1,3,-3,3,-3,5,-5,5,-5,-3,3,-5,5,-5,5,-3,3,-5,5,   
     &-5,5,-3,3,-5,5,-5,5,3*0,-11,-13,-15,-12,-14,-16,-1,-3,-5,-2,-4,   
     &4*0,12,14,16,2,4,0,12,14,16,2,4,0,12,14,16,2,4,0,12,14,16,2,4,    
     &28*0,2,4,12,-11,11,14,-13,13,16,-15,15,12,-11,11,14,-13,13,16,    
     &-15,15,12,-11,11,14,-13,13,16,-15,15,12,-11,11,14,-13,13,16,-15,  
     &15,12,-11,11,14,-13,13,16,-15,15,12,-11,11,14,-13,13,16,-15,15,   
     &2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,   
     &2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,   
     &2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,   
     &2*2,1,-1,2*4,3,-3,2*6,5,-5,2*2,1,-1,2*4,3,-3,2*6,5,-5,3,-3,5,-5,  
     &1,3,-3,5,-5,1,3,5,-5,1,5,-5,1,3,5,-5,1,3,351*0,-5,95*0,2,4,6,2,4, 
     &6,2,4,6,-2,-4,-6,-2,-4,-6,-2,-4,-6,2*9900014,2*9900016,2,4,6,2,4, 
     &6,2,4,6,-2,-4,-6,-2,-4,-6,-2,-4,-6,2*9900012,2*9900016,2,4,6,2,4, 
     &6,2,4,6,-2,-4,-6,-2,-4,-6,-2,-4,-6,2*9900012,2*9900014,3831*0/    
      DATA (KFDP(I,4),I=   1,8000)/94*0,4*111,6*0,111,2*0,-211,0,-211,  
     &3*0,111,2*-211,0,111,0,2*111,113,221,2*111,-213,-211,211,113,     
     &6*111,310,2*130,402*0,13*81,41*0,-11,10*0,111,-211,4*0,111,62*0,  
     &111,211,111,211,7*0,111,211,111,211,35*0,2*-211,2*111,211,111,    
     &-211,2*211,2*-211,13*0,-211,111,-211,111,4*0,-211,111,-211,111,   
     &34*0,111,-211,3*111,3*-211,2*111,3*-211,14*0,-321,-311,3*0,-321,  
     &-311,20*0,-3,43*0,6*1,39*0,6*2,42*0,6*3,14*0,8*4,4*0,4*-5,4*0,    
     &2*-5,67*0,-211,111,5*0,-211,111,52*0,2101,2103,2*2101,6*0,4*81,   
     &4*0,4*81,6*0,4*81,0,28*81,13*0,6*2101,18*81,4*0,18*81,4*0,9*81,0, 
     &162*81,31*0,-211,111,6516*0/                                      
      DATA (KFDP(I,5),I=   1,8000)/96*0,2*111,17*0,111,7*0,2*111,0,     
     &3*111,0,111,597*0,-211,2*111,-211,111,-211,111,65*0,111,-211,     
     &3*111,-211,111,7193*0/                                            
 
C...PYDAT4, with particle names (character strings).
      DATA (CHAF(I,1),I=   1, 202)/'d','u','s','c','b','t','b''','t''', 
     &2*' ','e-','nu_e','mu-','nu_mu','tau-','nu_tau','tau''-',         
     &'nu''_tau',2*' ','g','gamma','Z0','W+','h0',6*' ','Z''0','Z"0',   
     &'W''+','H0','A0','H+',' ','Graviton',' ','R0','LQ_ue',38*' ',     
     &'specflav','rndmflav','phasespa','c-hadron','b-hadron',2*' ',     
     &'junction',' ','system','cluster','string','indep.','CMshower',   
     &'SPHEaxis','THRUaxis','CLUSjet','CELLjet','table',' ','reggeon',  
     &'pi0','rho0','a_20','K_L0','pi+','rho+','a_2+','eta','omega',     
     &'f_2','K_S0','K0','K*0','K*_20','K+','K*+','K*_2+','eta''','phi', 
     &'f''_2','D+','D*+','D*_2+','D0','D*0','D*_20','D_s+','D*_s+',     
     &'D*_2s+','eta_c','J/psi','chi_2c','B0','B*0','B*_20','B+','B*+',  
     &'B*_2+','B_s0','B*_s0','B*_2s0','B_c+','B*_c+','B*_2c+','eta_b',  
     &'Upsilon','chi_2b','pomeron','dd_1','Delta-','ud_0','ud_1','n0',  
     &'Delta0','uu_1','p+','Delta+','Delta++','sd_0','sd_1','Sigma-',   
     &'Sigma*-','Lambda0','su_0','su_1','Sigma0','Sigma*0','Sigma+',    
     &'Sigma*+','ss_1','Xi-','Xi*-','Xi0','Xi*0','Omega-','cd_0',       
     &'cd_1','Sigma_c0','Sigma*_c0','Lambda_c+','Xi_c0','cu_0','cu_1',  
     &'Sigma_c+','Sigma*_c+','Sigma_c++','Sigma*_c++','Xi_c+','cs_0',   
     &'cs_1','Xi''_c0','Xi*_c0','Xi''_c+','Xi*_c+','Omega_c0',          
     &'Omega*_c0','cc_1','Xi_cc+','Xi*_cc+','Xi_cc++','Xi*_cc++'/       
      DATA (CHAF(I,1),I= 203, 332)/'Omega_cc+','Omega*_cc+',            
     &'Omega*_ccc++','bd_0','bd_1','Sigma_b-','Sigma*_b-','Lambda_b0',  
     &'Xi_b-','Xi_bc0','bu_0','bu_1','Sigma_b0','Sigma*_b0','Sigma_b+', 
     &'Sigma*_b+','Xi_b0','Xi_bc+','bs_0','bs_1','Xi''_b-','Xi*_b-',    
     &'Xi''_b0','Xi*_b0','Omega_b-','Omega*_b-','Omega_bc0','bc_0',     
     &'bc_1','Xi''_bc0','Xi*_bc0','Xi''_bc+','Xi*_bc+','Omega''_bc0',   
     &'Omega*_bc0','Omega_bcc+','Omega*_bcc+','bb_1','Xi_bb-',          
     &'Xi*_bb-','Xi_bb0','Xi*_bb0','Omega_bb-','Omega*_bb-',            
     &'Omega_bbc0','Omega*_bbc0','Omega*_bbb-','a_00','b_10','a_0+',    
     &'b_1+','f_0','h_1','K*_00','K_10','K*_0+','K_1+','f''_0','h''_1', 
     &'D*_0+','D_1+','D*_00','D_10','D*_0s+','D_1s+','chi_0c','h_1c',   
     &'B*_00','B_10','B*_0+','B_1+','B*_0s0','B_1s0','B*_0c+','B_1c+',  
     &'chi_0b','h_1b','a_10','a_1+','f_1','K*_10','K*_1+','f''_1',      
     &'D*_1+','D*_10','D*_1s+','chi_1c','B*_10','B*_1+','B*_1s0',       
     &'B*_1c+','chi_1b','psi''','Upsilon''','~d_L','~u_L','~s_L',       
     &'~c_L','~b_1','~t_1','~e_L-','~nu_eL','~mu_L-','~nu_muL',         
     &'~tau_1-','~nu_tauL','~g','~chi_10','~chi_20','~chi_1+',          
     &'~chi_30','~chi_40','~chi_2+','~Gravitino','~d_R','~u_R','~s_R',  
     &'~c_R','~b_2','~t_2','~e_R-','~nu_eR','~mu_R-','~nu_muR',         
     &'~tau_2-','~nu_tauR','pi_tc0','pi_tc+','pi''_tc0','eta_tc0'/      
      DATA (CHAF(I,1),I= 333, 500)/'rho_tc0','rho_tc+','omega_tc',      
     &'V8_tc','pi_22_1_tc','pi_22_8_tc','rho_11_tc','rho_12_tc',        
     &'rho_21_tc','rho_22_tc','d*','u*','e*-','nu*_e0','Graviton*',     
     &'nu_Re','nu_Rmu','nu_Rtau','Z_R0','W_R+','H_L++','H_R++',         
     &'rho_diff0','pi_diffr+','omega_di','phi_diff','J/psi_di',         
     &'n_diffr0','p_diffr+','cc~[3S18]','cc~[1S08]','cc~[3P08]',        
     &'bb~[3S18]','bb~[1S08]','bb~[3P08]','a_tc0','a_tc+',
     &81*' ',
C...UED    
     &'d*_S','u*_S','s*_S','c*_S','b*_S','t*_S',
     &'d*_D','u*_D','s*_D','c*_D','b*_D','t*_D',
     &'e*_S-','mu*_S-','tau*_S-',
     &'nu*_eD','e*_D-','nu*_muD','mu*_D-','nu*_tauD','tau*_D-',
     &'g*','gamma*','Z*0','W*+',25*' '/               
      DATA (CHAF(I,2),I=   1, 205)/'dbar','ubar','sbar','cbar','bbar',  
     &'tbar','b''bar','t''bar',2*' ','e+','nu_ebar','mu+','nu_mubar',   
     &'tau+','nu_taubar','tau''+','nu''_taubar',5*' ','W-',9*' ',       
     &'W''-',2*' ','H-',3*' ','Rbar0','LQ_uebar',39*' ','rndmflavbar',  
     &' ','c-hadronbar','b-hadronbar',20*' ','pi-','rho-','a_2-',4*' ', 
     &'Kbar0','K*bar0','K*_2bar0','K-','K*-','K*_2-',3*' ','D-','D*-',  
     &'D*_2-','Dbar0','D*bar0','D*_2bar0','D_s-','D*_s-','D*_2s-',      
     &3*' ','Bbar0','B*bar0','B*_2bar0','B-','B*-','B*_2-','B_sbar0',   
     &'B*_sbar0','B*_2sbar0','B_c-','B*_c-','B*_2c-',4*' ','dd_1bar',   
     &'Deltabar+','ud_0bar','ud_1bar','nbar0','Deltabar0','uu_1bar',    
     &'pbar-','Deltabar-','Deltabar--','sd_0bar','sd_1bar','Sigmabar+', 
     &'Sigma*bar+','Lambdabar0','su_0bar','su_1bar','Sigmabar0',        
     &'Sigma*bar0','Sigmabar-','Sigma*bar-','ss_1bar','Xibar+',         
     &'Xi*bar+','Xibar0','Xi*bar0','Omegabar+','cd_0bar','cd_1bar',     
     &'Sigma_cbar0','Sigma*_cbar0','Lambda_cbar-','Xi_cbar0','cu_0bar', 
     &'cu_1bar','Sigma_cbar-','Sigma*_cbar-','Sigma_cbar--',            
     &'Sigma*_cbar--','Xi_cbar-','cs_0bar','cs_1bar','Xi''_cbar0',      
     &'Xi*_cbar0','Xi''_cbar-','Xi*_cbar-','Omega_cbar0',               
     &'Omega*_cbar0','cc_1bar','Xi_ccbar-','Xi*_ccbar-','Xi_ccbar--',   
     &'Xi*_ccbar--','Omega_ccbar-','Omega*_ccbar-','Omega*_cccbar-'/    
      DATA (CHAF(I,2),I= 206, 325)/'bd_0bar','bd_1bar','Sigma_bbar+',   
     &'Sigma*_bbar+','Lambda_bbar0','Xi_bbar+','Xi_bcbar0','bu_0bar',   
     &'bu_1bar','Sigma_bbar0','Sigma*_bbar0','Sigma_bbar-',             
     &'Sigma*_bbar-','Xi_bbar0','Xi_bcbar-','bs_0bar','bs_1bar',        
     &'Xi''_bbar+','Xi*_bbar+','Xi''_bbar0','Xi*_bbar0','Omega_bbar+',  
     &'Omega*_bbar+','Omega_bcbar0','bc_0bar','bc_1bar','Xi''_bcbar0',  
     &'Xi*_bcbar0','Xi''_bcbar-','Xi*_bcbar-','Omega''_bcba',           
     &'Omega*_bcbar0','Omega_bccbar-','Omega*_bccbar-','bb_1bar',       
     &'Xi_bbbar+','Xi*_bbbar+','Xi_bbbar0','Xi*_bbbar0','Omega_bbbar+', 
     &'Omega*_bbbar+','Omega_bbcbar0','Omega*_bbcbar0',                 
     &'Omega*_bbbbar+',2*' ','a_0-','b_1-',2*' ','K*_0bar0','K_1bar0',  
     &'K*_0-','K_1-',2*' ','D*_0-','D_1-','D*_0bar0','D_1bar0',         
     &'D*_0s-','D_1s-',2*' ','B*_0bar0','B_1bar0','B*_0-','B_1-',       
     &'B*_0sbar0','B_1sbar0','B*_0c-','B_1c-',3*' ','a_1-',' ',         
     &'K*_1bar0','K*_1-',' ','D*_1-','D*_1bar0','D*_1s-',' ',           
     &'B*_1bar0','B*_1-','B*_1sbar0','B*_1c-',3*' ','~d_Lbar',          
     &'~u_Lbar','~s_Lbar','~c_Lbar','~b_1bar','~t_1bar','~e_L+',        
     &'~nu_eLbar','~mu_L+','~nu_muLbar','~tau_1+','~nu_tauLbar',3*' ',  
     &'~chi_1-',2*' ','~chi_2-',' ','~d_Rbar','~u_Rbar','~s_Rbar',      
     &'~c_Rbar','~b_2bar','~t_2bar','~e_R+','~nu_eRbar','~mu_R+'/       
      DATA (CHAF(I,2),I= 326, 500)/'~nu_muRbar','~tau_2+',              
     &'~nu_tauRbar',' ','pi_tc-',3*' ','rho_tc-',8*' ','d*bar','u*bar', 
     &'e*bar+','nu*_ebar0',5*' ','W_R-','H_L--','H_R--',' ',            
     &'pi_diffr-',3*' ','n_diffrbar0','p_diffrbar-',7*' ','a_tc-',     
     &81*' ',
C...UED
     &'d*_Sbar','u*_Sbar','s*_Sbar','c*_Sbar','b*_Sbar','t*_Sbar',
     &'d*_Dbar','u*_Dbar','s*_Dbar','c*_Dbar','b*_Dbar','t*_Dbar',
     &'e*_Sbar+','mu*_Sbar+','tau*_Sbar+',
     &'nu*_eDbar','e*_Dbar+',
     &'nu*_muDbar','mu*_Dbar+',
     &'nu*_tauDbar','tau*_Dbar+',
     &'g*','gamma*','Z*0','W*-',25*' '/            
 
C...PYDATR, with initial values for the random number generator.
      DATA MRPY/19780503,0,0,97,33,0/
 
C...Default values for allowed processes and kinematics constraints.
      DATA MSEL/1/
      DATA MSUB/500*0/
      DATA ((KFIN(I,J),J=-40,40),I=1,2)/16*0,4*1,4*0,6*1,5*0,5*1,0,
     &5*1,5*0,6*1,4*0,4*1,16*0,16*0,4*1,4*0,6*1,5*0,5*1,0,5*1,5*0,
     &6*1,4*0,4*1,16*0/
      DATA CKIN/
     &  2.0D0, -1.0D0,  0.0D0, -1.0D0,  1.0D0,
     &  1.0D0,  -10D0,   10D0,  -40D0,   40D0,
     1  -40D0,   40D0,  -40D0,   40D0,  -40D0,
     1   40D0, -1.0D0,  1.0D0, -1.0D0,  1.0D0,
     2  0.0D0,  1.0D0,  0.0D0,  1.0D0, -1.0D0,
     2  1.0D0, -1.0D0,  1.0D0,    0D0,    0D0,
     3  2.0D0, -1.0D0,    0D0,    0D0,  0.0D0,
     3 -1.0D0,  0.0D0, -1.0D0,  4.0D0, -1.0D0,
     4 12.0D0, -1.0D0, 12.0D0, -1.0D0, 12.0D0,
     4 -1.0D0, 12.0D0, -1.0D0,    0D0,    0D0,
     5  0.0D0, -1.0D0,  0.0D0, -1.0D0,  0.0D0,
     5 -1.0D0,    0D0,    0D0,    0D0,    0D0,
     6 0.0001D0, 0.99D0, 0.0001D0, 0.99D0,    0D0,
     6   -1D0,    0D0,   -1D0,    0D0,   -1D0,
     7    0D0,   -1D0, 0.0001D0, 0.99D0, 0.0001D0,
     7 0.99D0,    2D0,   -1D0,    0D0,    0D0,
     8  120*0D0/
 
C...Default values for main switches and parameters. Reset information.
      DATA (MSTP(I),I=1,100)/
     &  3,    1,    2,    0,    0,    0,    0,    0,    0,    0,
     1  1,    0,    1,   30,    0,    1,    4,    3,    4,    3,
     2  1,    0,    1,    0,    0,    0,    0,    0,    0,    1,
     3  1,    8,    0,    1,    0,    2,    1,    5,    2,    0,
     4  2,    1,    3,    7,    3,    1,    1,    0,    1,    0,
     5  7,    1,    3,    1,    5,    1,    1,    5,    1,    7,
     6  2,    3,    2,    2,    1,    5,    2,    3,    0,    0,
     7  1,    1,    0,    0,    0,    0,    0,    0,    0,    0,
     8  1,    4,  100,    1,    1,    2,    4,    1,    1,    0,
     9  1,    3,    1,    3,    1,    0,    0,    0,    0,    0/
      DATA (MSTP(I),I=101,200)/
     &  3,    1,    0,    0,    0,    0,    0,    0,    0,    0,
     1  1,    1,    1,    0,    0,    0,    0,    0,    0,    0,
     2  0,    1,    2,    1,    1,  100,    0,    0,   10,    0,
     3  0,    4,    0,    1,    0,    0,    0,    0,    0,    0,
     4  0,    0,    0,    0,    0,    1,    0,    0,    0,    0,
     5  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     6  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     7  0,    2,    0,    0,    0,    0,    0,    0,    0,    0,
     8  6,  424, 2010,   10,   21,    0,    0,    0,    0,    0,
     9  0,    0,    0,    0,    0,    0,    0,    0,    0,    0/
      DATA (PARP(I),I=1,100)/
     &  0.25D0,  10D0, 8*0D0,
     1  0D0, 0D0, 1.0D0, 0.01D0, 0.5D0, 1.0D0, 1.0D0, 0.4D0, 2*0D0,
     2  10*0D0,
     3  1.5D0,2.0D0,0.075D0,1.0D0,0.2D0,0D0,1.0D0,0.70D0,0.006D0,0D0,
     4  0.02D0,2.0D0,0.10D0,1000D0,2054D0,123D0,246D0,50D0,0D0,0.054D0,
     5  10*0D0,
     6  0.25D0, 1.0D0,0.25D0, 1.0D0, 2.0D0,1D-3, 4.0D0,1D-3,2*0D0,
     7  4.0D0, 0.25D0, 5*0D0, 0.025D0, 2.0D0, 0.1D0,
     8  1.90D0, 2.0D0, 0.5D0, 0.4D0, 0.90D0,
     8  0.95D0, 0.7D0, 0.5D0, 1800D0, 0.25D0,
     9  2.0D0,0.40D0,5.0D0,1.0D0,0.0D0,3.0D0,1.0D0,0.75D0,1.0D0,5.0D0/
      DATA (PARP(I),I=101,200)/
     &  0.5D0, 0.28D0,  1.0D0, 0.8D0, 0D0, 0D0, 0D0, 0D0, 0D0, 1D0,
     1  2.0D0, 3*0D0, 1.5D0, 0.5D0, 0.6D0, 2.5D0, 2.0D0, 1.0D0,
     2  1.0D0,  0.4D0, 8*0D0,
     3  0.01D0, 9*0D0,
     4  1.16D0, 0.0119D0, 0.01D0, 0.01D0, 0.05D0, 
     4  9.28D0, 0.15D0, 0.02D0, 0.48D0, 0.09D0,
     5  0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0, 0D0,
     6  2.20D0, 23.6D0, 18.4D0, 11.5D0, 0.5D0, 0D0, 0D0, 0D0, 2*0D0,
     7  0D0,   0D0,   0D0,  1.0D0, 6*0D0,
     8  0.1D0, 0.01D0, 0.01D0, 0.01D0, 0.1D0, 0.01D0, 0.01D0, 0.01D0,
     8  0.3D0, 0.64D0,
     9  0.64D0, 5.0D0, 1.0D4, 1.0D4, 6*0D0/
      DATA MSTI/200*0/
      DATA PARI/200*0D0/
      DATA MINT/400*0/
      DATA VINT/400*0D0/
 
C...Constants for the generation of the various processes.
      DATA (ISET(I),I=1,100)/
     &  1,    1,    1,   -1,    3,   -1,   -1,    3,   -2,    2,
     1  2,    2,    2,    2,    2,    2,   -1,    2,    2,    2,
     2 -1,    2,    2,    2,    2,    2,   -1,    2,    2,    2,
     3  2,    2,    2,    2,    2,    2,   -1,   -1,   -1,   -1,
     4 -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     5 -1,   -1,    2,    2,   -1,   -1,   -1,    2,   -1,   -1,
     6 -1,   -1,   -1,   -1,   -1,   -1,   -1,    2,    2,    2,
     7  4,    4,    4,   -1,   -1,    4,    4,   -1,   -1,    2,
     8  2,    2,    2,    2,    2,    2,    2,    2,    2,   -2,
     9  0,    0,    0,    0,    0,    9,   -2,   -2,    8,   -2/
      DATA (ISET(I),I=101,200)/
     & -1,    1,    1,    1,    1,    2,    2,    2,   -2,    2,
     1  2,    2,    2,    2,    2,   -1,   -1,   -1,   -2,   -2,
     2  5,    5,    5,    5,   -2,   -2,   -2,   -2,   -2,   -2,
     3  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     4  1,    1,    1,    1,    1,    1,    1,    1,    1,   -2,
     5  1,    1,    1,   -2,   -2,    1,    1,    1,   -2,   -2,
     6  2,    2,    2,    2,    2,    2,    2,    2,    2,   -2,
     7  2,    2,    5,    5,   -2,    2,    2,    5,    5,   -2,
     8  5,    5,    2,    2,    2,    5,    5,    2,    2,    2,
     9  1,    1,    1,    2,    2,   -2,   -2,   -2,   -2,   -2/
      DATA (ISET(I),I=201,300)/
     &  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     1  2,    2,    2,    2,   -2,    2,    2,    2,    2,    2,
     2  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     3  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     4  2,    2,    2,    2,   -1,    2,    2,    2,    2,    2,
     5  2,    2,    2,    2,   -1,    2,   -1,    2,    2,   -2,
     6  2,    2,    2,    2,    2,   -1,   -1,   -1,   -1,   -1,
     7  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     8  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     9  2,    2,    2,    2,    2,    2,    2,    2,    2,    2/
      DATA (ISET(I),I=301,500)/
     &  2, 9*-2, 9*2, 21*-2,
     4  1,    1,    2,    2,    2,    2,    2,    2,    2,    2,
     5  5,    5,    1,    1,   -1,   -1,   -1,   -1,   -1,   -1,
     6  2,    2,    2,    2,    2,    2,    2,    2,   -1,    2,
     7  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     8  2,    2,    2,    2,    2,    2,    2,    2,   -2,   -2,
     9  1,    1,    2,    2,    2, 5*-2,
     &  5,    5, 18*-2,
     2  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     3  2,    2,    2,    2,    2,    2,    2,    2,    2, 21*-2,
     6  2,    2,    2,    2,    2,    2,    2,    2,    2,    2,
     7  2,    2,    2,    2,    2,    2,    2,    2,    2,   -2,
     8  2,    2,  18*-2/
      DATA ((KFPR(I,J),J=1,2),I=1,50)/
     &  23,    0,   24,    0,   25,    0,   24,    0,   25,    0,
     &  24,    0,   23,    0,   25,    0,    0,    0,    0,    0,
     1   0,    0,    0,    0,   21,   21,   21,   22,   21,   23,
     1  21,   24,   21,   25,   22,   22,   22,   23,   22,   24,
     2  22,   25,   23,   23,   23,   24,   23,   25,   24,   24,
     2  24,   25,   25,   25,    0,   21,    0,   22,    0,   23,
     3   0,   24,    0,   25,    0,   21,    0,   22,    0,   23,
     3   0,   24,    0,   25,    0,   21,    0,   22,    0,   23,
     4   0,   24,    0,   25,    0,   21,    0,   22,    0,   23,
     4   0,   24,    0,   25,    0,   21,    0,   22,    0,   23/
      DATA ((KFPR(I,J),J=1,2),I=51,100)/
     5   0,   24,    0,   25,    0,    0,    0,    0,    0,    0,
     5   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     6   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     6   0,    0,    0,    0,   21,   21,   24,   24,   23,   24,
     7  23,   23,   24,   24,   23,   24,   23,   25,   22,   22,
     7  23,   23,   24,   24,   24,   25,   25,   25,    0,  211,
     8   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     8 443,   21,10441,   21,20443,   21,  445,   21,    0,    0,
     9   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     9   0,    0,    0,    0,    0,    0,    0,    0,    0,    0/
      DATA ((KFPR(I,J),J=1,2),I=101,150)/
     &  23,    0,   25,    0,   25,    0,10441,    0,  445,    0,
     & 443,   22,  443,   21,  443,   22,    0,    0,   22,   25,
     1  21,   25,    0,   25,   21,   25,   22,   22,   21,   22,
     1  22,   23,   23,   23,   24,   24,    0,    0,    0,    0,
     2  25,    6,   25,    6,   25,    0,   25,    0,    0,    0,
     2   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     3   0,   21,    0,   21,    0,   22,    0,   22,    0,    0,
     3   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
     4  32,    0,   34,    0,   37,    0,   41,    0,   42,    0,
     4 4000011, 0, 4000001, 0, 4000002, 0, 3000331, 0,   0,    0/
      DATA ((KFPR(I,J),J=1,2),I=151,200)/
     5  35,    0,   35,    0,   35,    0,    0,    0,    0,    0,
     5  36,    0,   36,    0,   36,    0,    0,    0,    0,    0,
     6   6,   37,   42,    0,   42,   42,   42,   42,   11,    0,
     6  11,    0, 0, 4000001, 0, 4000002, 0, 4000011,    0,    0,
     7  23,   35,   24,   35,   35,    0,   35,    0,    0,    0,
     7  23,   36,   24,   36,   36,    0,   36,    0,    0,    0,
     8  35,    6,   35,    6,   21,   35,    0,   35,   21,   35,
     8  36,    6,   36,    6,   21,   36,    0,   36,   21,   36,
     9  3000113, 0, 3000213, 0, 3000223, 0, 11,    0,   11,    0,
     9   0,    0,    0,    0,    0,    0,    0,    0,    0,    0/
      DATA ((KFPR(I,J),J=1,2),I=201,240)/
     &  1000011,   1000011,   2000011,   2000011,   1000011,
     &  2000011,   1000013,   1000013,   2000013,   2000013,
     &  1000013,   2000013,   1000015,   1000015,   2000015,
     &  2000015,   1000015,   2000015,   1000011,   1000012,
     1  1000015,   1000016,   2000015,   1000016,   1000012,
     1  1000012,   1000016,   1000016,         0,         0,
     1  1000022,   1000022,   1000023,   1000023,   1000025,
     1  1000025,   1000035,   1000035,   1000022,   1000023,
     2  1000022,   1000025,   1000022,   1000035,   1000023,
     2  1000025,   1000023,   1000035,   1000025,   1000035,
     2  1000024,   1000024,   1000037,   1000037,   1000024,
     2  1000037,   1000022,   1000024,   1000023,   1000024,
     3  1000025,   1000024,   1000035,   1000024,   1000022,
     3  1000037,   1000023,   1000037,   1000025,   1000037,
     3  1000035,   1000037,   1000021,   1000022,   1000021,
     3  1000023,   1000021,   1000025,   1000021,   1000035/
      DATA ((KFPR(I,J),J=1,2),I=241,280)/
     4  1000021,   1000024,   1000021,   1000037,   1000021,
     4  1000021,   1000021,   1000021,         0,         0,
     4  1000002,   1000022,   2000002,   1000022,   1000002,
     4  1000023,   2000002,   1000023,   1000002,   1000025,
     5  2000002,   1000025,   1000002,   1000035,   2000002,
     5  1000035,   1000001,   1000024,   2000005,   1000024,
     5  1000001,   1000037,   2000005,   1000037,   1000002,
     5  1000021,   2000002,   1000021,         0,         0,
     6  1000006,   1000006,   2000006,   2000006,   1000006,
     6  2000006,   1000006,   1000006,   2000006,   2000006,
     6        0,         0,         0,         0,         0,
     6        0,         0,         0,         0,         0,
     7  1000002,   1000002,   2000002,   2000002,   1000002,
     7  2000002,   1000002,   1000002,   2000002,   2000002,
     7  1000002,   2000002,   1000002,   1000002,   2000002,
     7  2000002,   1000002,   1000002,   2000002,   2000002/
      DATA ((KFPR(I,J),J=1,2),I=281,350)/
     8  1000005,   1000002,   2000005,   2000002,   1000005,
     8  2000002,   1000005,   1000002,   2000005,   2000002,
     8  1000005,   2000002,   1000005,   1000005,   2000005,
     8  2000005,   1000005,   1000005,   2000005,   2000005,
     9  1000005,   1000005,   2000005,   2000005,   1000005,
     9  2000005,   1000005,   1000021,   2000005,   1000021,
     9  1000005,   2000005,        37,        25,        37,
     9       35,        36,        25,        36,        35,
     &       37,        37,      18*0,
C...UED: 311-319
     &  5100021,   5100021, 
     &  5100002,   5100021, 
     &  5100002,   5100001,
     &  5100002,  -5100002, 
     &  5100002,  -5100002,
     &  5100002,  -6100001,
     &  5100002,  -5100001,
     &  5100002,   6100001,
     &  5100001,  -5100001,
     &  42*0,
     4  9900041,         0,   9900042,         0,   9900041,
     4       11,   9900042,        11,   9900041,        13,
     4  9900042,        13,   9900041,        15,   9900042,
     4       15,   9900041,   9900041,   9900042,   9900042/
      DATA ((KFPR(I,J),J=1,2),I=351,400)/
     5  9900041,         0,   9900042,         0,   9900023,
     5        0,   9900024,         0,         0,         0,
     5        0,         0,         0,         0,         0,
     5        0,         0,         0,         0,         0,
     6       24,        24,        24,   3000211,   3000211,
     6  3000211,        22,   3000111,        22,   3000221,
     6       23,   3000111,        23,   3000221,        24,
     6  3000211,         0,         0,        24,        23,
     7       24,   3000111,   3000211,        23,   3000211,
     7  3000111,        22,   3000211,        23,   3000211,
     7       24,   3000111,        24,   3000221,        22,
     7       24,        22,        23,        23,        23,
     8   0,    0,    0,    0,   21,   21,    0,   21,    0,    0,
     8  21,   21,    0,    0,    0,    0,    0,    0,    0,    0,
     9  5000039,         0,   5000039,         0,        21,
     9  5000039,         0,   5000039,        21,   5000039,
     9     10*0/
      DATA ((KFPR(I,J),J=1,2),I=401,500)/
     &  37,    6,   37,    6,    36*0,
     2      443,        21,   9900443,        21,   9900441,
     2       21,   9910441,        21,         0,   9900443,
     2        0,   9900441,         0,   9910441,        21,
     2  9900443,        21,   9900441,        21,   9910441,
     3 10441, 21, 20443,  21,  445,   21,    0, 10441,   0, 20443,
     3   0,  445,   21, 10441,  21, 20443,  21,  445,  42*0,
     6      553,        21,   9900553,        21,   9900551,
     6       21,   9910551,        21,         0,   9900553,
     6        0,   9900551,         0,   9910551,        21,
     6  9900553,        21,   9900551,        21,   9910551,
     7 10551, 21, 20553,  21,  555,   21,    0, 10551,   0, 20553,
     7   0,  555,   21, 10551,  21, 20553,  21,  555, 42*0/
      DATA COEF/10000*0D0/
      DATA (((ICOL(I,J,K),K=1,2),J=1,4),I=1,40)/
     &4,0,3,0,2,0,1,0,3,0,4,0,1,0,2,0,2,0,0,1,4,0,0,3,3,0,0,4,1,0,0,2,
     &3,0,0,4,1,4,3,2,4,0,0,3,4,2,1,3,2,0,4,1,4,0,2,3,4,0,3,4,2,0,1,2,
     &3,2,1,0,1,4,3,0,4,3,3,0,2,1,1,0,3,2,1,4,1,0,0,2,2,4,3,1,2,0,0,1,
     &3,2,1,4,1,4,3,2,4,2,1,3,4,2,1,3,3,4,4,3,1,2,2,1,2,0,3,1,2,0,0,0,
     &4,2,1,0,0,0,1,0,3,0,0,3,1,2,0,0,4,0,0,4,0,0,1,2,2,0,0,1,4,4,3,3,
     &2,2,1,1,4,4,3,3,3,3,4,4,1,1,2,2,3,2,1,3,1,2,0,0,4,2,1,4,0,0,1,2,
     &4,0,0,0,4,0,1,3,0,0,3,0,2,4,3,0,3,4,0,0,1,0,0,1,0,0,3,4,2,0,0,2,
     &3,0,0,0,1,0,0,0,0,0,3,0,2,0,0,0,2,0,3,1,2,0,0,0,3,2,1,0,1,0,0,0,
     &4,4,3,3,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
     &0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0/
 
C...Treatment of resonances.
      DATA (MWID(I)  ,I=   1, 500)/5*0,3*1,8*0,1,5*0,3*1,6*0,1,0,4*1,   
     &3*0,2*1,254*0,19*2,0,7*2,0,2,0,2,0,26*1,7*0,6*2,2*1,
     &81*0,21*1,4*1,25*0/
 
C...Character constants: name of processes.
      DATA PROC(0)/                    'All included subprocesses   '/
      DATA (PROC(I),I=1,20)/
     &'f + fbar -> gamma*/Z0       ',  'f + fbar'' -> W+/-           ',
     &'f + fbar -> h0              ',  'gamma + W+/- -> W+/-        ',
     &'Z0 + Z0 -> h0               ',  'Z0 + W+/- -> W+/-           ',
     &'                            ',  'W+ + W- -> h0               ',
     &'                            ',  'f + f'' -> f + f'' (QFD)      ',
     1'f + f'' -> f + f'' (QCD)      ','f + fbar -> f'' + fbar''      ',
     1'f + fbar -> g + g           ',  'f + fbar -> g + gamma       ',
     1'f + fbar -> g + Z0          ',  'f + fbar'' -> g + W+/-       ',
     1'f + fbar -> g + h0          ',  'f + fbar -> gamma + gamma   ',
     1'f + fbar -> gamma + Z0      ',  'f + fbar'' -> gamma + W+/-   '/
      DATA (PROC(I),I=21,40)/
     2'f + fbar -> gamma + h0      ',  'f + fbar -> Z0 + Z0         ',
     2'f + fbar'' -> Z0 + W+/-      ', 'f + fbar -> Z0 + h0         ',
     2'f + fbar -> W+ + W-         ',  'f + fbar'' -> W+/- + h0      ',
     2'f + fbar -> h0 + h0         ',  'f + g -> f + g              ',
     2'f + g -> f + gamma          ',  'f + g -> f + Z0             ',
     3'f + g -> f'' + W+/-          ', 'f + g -> f + h0             ',
     3'f + gamma -> f + g          ',  'f + gamma -> f + gamma      ',
     3'f + gamma -> f + Z0         ',  'f + gamma -> f'' + W+/-      ',
     3'f + gamma -> f + h0         ',  'f + Z0 -> f + g             ',
     3'f + Z0 -> f + gamma         ',  'f + Z0 -> f + Z0            '/
      DATA (PROC(I),I=41,60)/
     4'f + Z0 -> f'' + W+/-         ', 'f + Z0 -> f + h0            ',
     4'f + W+/- -> f'' + g          ', 'f + W+/- -> f'' + gamma      ',
     4'f + W+/- -> f'' + Z0         ', 'f + W+/- -> f'' + W+/-       ',
     4'f + W+/- -> f'' + h0         ', 'f + h0 -> f + g             ',
     4'f + h0 -> f + gamma         ',  'f + h0 -> f + Z0            ',
     5'f + h0 -> f'' + W+/-         ', 'f + h0 -> f + h0            ',
     5'g + g -> f + fbar           ',  'g + gamma -> f + fbar       ',
     5'g + Z0 -> f + fbar          ',  'g + W+/- -> f + fbar''       ',
     5'g + h0 -> f + fbar          ',  'gamma + gamma -> f + fbar   ',
     5'gamma + Z0 -> f + fbar      ',  'gamma + W+/- -> f + fbar''   '/
      DATA (PROC(I),I=61,80)/
     6'gamma + h0 -> f + fbar      ',  'Z0 + Z0 -> f + fbar         ',
     6'Z0 + W+/- -> f + fbar''      ', 'Z0 + h0 -> f + fbar         ',
     6'W+ + W- -> f + fbar         ',  'W+/- + h0 -> f + fbar''      ',
     6'h0 + h0 -> f + fbar         ',  'g + g -> g + g              ',
     6'gamma + gamma -> W+ + W-    ',  'gamma + W+/- -> Z0 + W+/-   ',
     7'Z0 + Z0 -> Z0 + Z0          ',  'Z0 + Z0 -> W+ + W-          ',
     7'Z0 + W+/- -> Z0 + W+/-      ',  'Z0 + Z0 -> Z0 + h0          ',
     7'W+ + W- -> gamma + gamma    ',  'W+ + W- -> Z0 + Z0          ',
     7'W+/- + W+/- -> W+/- + W+/-  ',  'W+/- + h0 -> W+/- + h0      ',
     7'h0 + h0 -> h0 + h0          ',  'q + gamma -> q'' + pi+/-     '/
      DATA (PROC(I),I=81,100)/
     8'q + qbar -> Q + Qbar, mass  ',  'g + g -> Q + Qbar, massive  ',
     8'f + q -> f'' + Q, massive    ', 'g + gamma -> Q + Qbar, mass ',
     8'gamma + gamma -> F + Fbar, m',  'g + g -> J/Psi + g          ',
     8'g + g -> chi_0c + g         ',  'g + g -> chi_1c + g         ',
     8'g + g -> chi_2c + g         ',  '                            ',
     9'Elastic scattering          ',  'Single diffractive (XB)     ',
     9'Single diffractive (AX)     ',  'Double  diffractive         ',
     9'Low-pT scattering           ',  'Semihard QCD 2 -> 2         ',
     9'                            ',  '                            ',
     9'q + gamma* -> q             ',  '                            '/
      DATA (PROC(I),I=101,120)/
     &'g + g -> gamma*/Z0          ',  'g + g -> h0                 ',
     &'gamma + gamma -> h0         ',  'g + g -> chi_0c             ',
     &'g + g -> chi_2c             ',  'g + g -> J/Psi + gamma      ',
     &'gamma + g -> J/Psi + g      ',  'gamma+gamma -> J/Psi + gamma',
     &'                            ',  'f + fbar -> gamma + h0      ',
     1'q + qbar -> g + h0          ',  'q + g -> q + h0             ',
     1'g + g -> g + h0             ',  'g + g -> gamma + gamma      ',
     1'g + g -> g + gamma          ',  'g + g -> gamma + Z0         ',
     1'g + g -> Z0 + Z0            ',  'g + g -> W+ + W-            ',
     1'                            ',  '                            '/
      DATA (PROC(I),I=121,140)/
     2'g + g -> Q + Qbar + h0      ',  'q + qbar -> Q + Qbar + h0   ',
     2'f + f'' -> f + f'' + h0       ',
     2'f + f'' -> f" + f"'' + h0     ',
     2'                            ',  '                            ',
     2'                            ',  '                            ',
     2'                            ',  '                            ',
     3'f + gamma*_T -> f + g       ',  'f + gamma*_L -> f + g       ',
     3'f + gamma*_T -> f + gamma   ',  'f + gamma*_L -> f + gamma   ',
     3'g + gamma*_T -> f + fbar    ',  'g + gamma*_L -> f + fbar    ',
     3'gamma*_T+gamma*_T -> f+fbar ',  'gamma*_T+gamma*_L -> f+fbar ',
     3'gamma*_L+gamma*_T -> f+fbar ',  'gamma*_L+gamma*_L -> f+fbar '/
      DATA (PROC(I),I=141,160)/
     4'f + fbar -> gamma*/Z0/Z''0   ', 'f + fbar'' -> W''+/-          ',
     4'f + fbar'' -> H+/-           ', 'f + fbar'' -> R              ',
     4'q + l -> LQ                 ',  'e + gamma -> e*             ',
     4'd + g -> d*                 ',  'u + g -> u*                 ',
     4'g + g -> eta_tc             ',  '                            ',
     5'f + fbar -> H0              ',  'g + g -> H0                 ',
     5'gamma + gamma -> H0         ',  '                            ',
     5'                            ',  'f + fbar -> A0              ',
     5'g + g -> A0                 ',  'gamma + gamma -> A0         ',
     5'                            ',  '                            '/
      DATA (PROC(I),I=161,180)/
     6'f + g -> f'' + H+/-          ', 'q + g -> LQ + lbar          ',
     6'g + g -> LQ + LQbar         ',  'q + qbar -> LQ + LQbar      ',
     6'f + fbar -> f'' + fbar'' (g/Z)',
     6'f +fbar'' -> f" + fbar"'' (W) ',
     6'q + q'' -> q" + d*           ',  'q + q'' -> q" + u*           ',
     6'q + qbar -> e + e*          ',  '                            ',
     7'f + fbar -> Z0 + H0         ', 'f + fbar'' -> W+/- + H0      ',
     7'f + f'' -> f + f'' + H0       ',
     7'f + f'' -> f" + f"'' + H0     ',
     7'                            ',  'f + fbar -> Z0 + A0         ',
     7'f + fbar'' -> W+/- + A0      ',
     7'f + f'' -> f + f'' + A0       ',
     7'f + f'' -> f" + f"'' + A0     ',
     7'                            '/
      DATA (PROC(I),I=181,200)/
     8'g + g -> Q + Qbar + H0      ',  'q + qbar -> Q + Qbar + H0   ',
     8'q + qbar -> g + H0          ',  'q + g -> q + H0             ',
     8'g + g -> g + H0             ',  'g + g -> Q + Qbar + A0      ',
     8'q + qbar -> Q + Qbar + A0   ',  'q + qbar -> g + A0          ',
     8'q + g -> q + A0             ',  'g + g -> g + A0             ',
     9'f + fbar -> rho_tc0         ',  'f + f'' -> rho_tc+/-         ',
     9'f + fbar -> omega_tc0      ',  'f+fbar -> f''+fbar'' (ETC)  ',
     9'f+fbar'' -> f"+fbar"'' (ETC)','                          ',
     9'                            ',  '                            ',
     9'                            ',  '                            '/
      DATA (PROC(I),I=201,220)/
     &'f + fbar -> ~e_L + ~e_Lbar  ',  'f + fbar -> ~e_R + ~e_Rbar  ',
     &'f + fbar -> ~e_R + ~e_Lbar  ',  'f + fbar -> ~mu_L + ~mu_Lbar',
     &'f + fbar -> ~mu_R + ~mu_Rbar',  'f + fbar -> ~mu_L + ~mu_Rbar',
     &'f+fbar -> ~tau_1 + ~tau_1bar',  'f+fbar -> ~tau_2 + ~tau_2bar',
     &'f+fbar -> ~tau_1 + ~tau_2bar',  'q + qbar'' -> ~l_L + ~nulbar ',
     1'q+qbar''-> ~tau_1 + ~nutaubar', 'q+qbar''-> ~tau_2 + ~nutaubar',
     1'f + fbar -> ~nul + ~nulbar  ',  'f+fbar -> ~nutau + ~nutaubar',
     1'                            ',  'f + fbar -> ~chi1 + ~chi1   ',
     1'f + fbar -> ~chi2 + ~chi2   ',  'f + fbar -> ~chi3 + ~chi3   ',
     1'f + fbar -> ~chi4 + ~chi4   ',  'f + fbar -> ~chi1 + ~chi2   '/
      DATA (PROC(I),I=221,240)/
     2'f + fbar -> ~chi1 + ~chi3   ',  'f + fbar -> ~chi1 + ~chi4   ',
     2'f + fbar -> ~chi2 + ~chi3   ',  'f + fbar -> ~chi2 + ~chi4   ',
     2'f + fbar -> ~chi3 + ~chi4   ',  'f+fbar -> ~chi+-1 + ~chi-+1 ',
     2'f+fbar -> ~chi+-2 + ~chi-+2 ',  'f+fbar -> ~chi+-1 + ~chi-+2 ',
     2'q + qbar'' -> ~chi1 + ~chi+-1', 'q + qbar'' -> ~chi2 + ~chi+-1',
     3'q + qbar'' -> ~chi3 + ~chi+-1', 'q + qbar'' -> ~chi4 + ~chi+-1',
     3'q + qbar'' -> ~chi1 + ~chi+-2', 'q + qbar'' -> ~chi2 + ~chi+-2',
     3'q + qbar'' -> ~chi3 + ~chi+-2', 'q + qbar'' -> ~chi4 + ~chi+-2',
     3'q + qbar -> ~chi1 + ~g      ',  'q + qbar -> ~chi2 + ~g      ',
     3'q + qbar -> ~chi3 + ~g      ',  'q + qbar -> ~chi4 + ~g      '/
      DATA (PROC(I),I=241,260)/
     4'q + qbar'' -> ~chi+-1 + ~g   ', 'q + qbar'' -> ~chi+-2 + ~g  ',
     4'q + qbar -> ~g + ~g         ',  'g + g -> ~g + ~g            ',
     4'                            ',  'qj + g -> ~qj_L + ~chi1     ',
     4'qj + g -> ~qj_R + ~chi1     ',  'qj + g -> ~qj_L + ~chi2     ',
     4'qj + g -> ~qj_R + ~chi2     ',  'qj + g -> ~qj_L + ~chi3     ',
     5'qj + g -> ~qj_R + ~chi3     ',  'qj + g -> ~qj_L + ~chi4     ',
     5'qj + g -> ~qj_R + ~chi4     ',  'qj + g -> ~qk_L + ~chi+-1   ',
     5'qj + g -> ~qk_R + ~chi+-1   ',  'qj + g -> ~qk_L + ~chi+-2   ',
     5'qj + g -> ~qk_R + ~chi+-2   ',  'qj + g -> ~qj_L + ~g        ',
     5'qj + g -> ~qj_R + ~g        ',  '                            '/
      DATA (PROC(I),I=261,300)/
     6'f + fbar -> ~t_1 + ~t_1bar  ',  'f + fbar -> ~t_2 + ~t_2bar  ',
     6'f + fbar -> ~t_1 + ~t_2bar  ',  'g + g -> ~t_1 + ~t_1bar     ',
     6'g + g -> ~t_2 + ~t_2bar     ',  '                            ',
     6'                            ',  '                            ',
     6'                            ',  '                            ',
     7'qi + qj -> ~qi_L + ~qj_L    ',  'qi + qj -> ~qi_R + ~qj_R    ',
     7'qi + qj -> ~qi_L + ~qj_R    ',  'qi+qjbar -> ~qi_L + ~qj_Lbar',
     7'qi+qjbar -> ~qi_R + ~qj_Rbar',  'qi+qjbar -> ~qi_L + ~qj_Rbar',
     7'f + fbar -> ~qi_L + ~qi_Lbar',  'f + fbar -> ~qi_R + ~qi_Rbar',
     7'g + g -> ~qi_L + ~qi_Lbar   ',  'g + g -> ~qi_R + ~qi_Rbar   ',
     8'b + qj -> ~b_1 + ~qj_L      ',  'b + qj -> ~b_2 + ~qj_R      ',
     8'b + qj -> ~b_1 + ~qj_R      ',  'b + qjbar -> ~b_1 + ~qj_Lbar',
     8'b + qjbar -> ~b_2 + ~qj_Rbar',  'b + qjbar -> ~b_1 + ~qj_Rbar',
     8'f + fbar -> ~b_1 + ~b_1bar  ',  'f + fbar -> ~b_2 + ~b_2bar  ',
     8'g + g -> ~b_1 + ~b_1bar     ',  'g + g -> ~b_2 + ~b_2bar     ',
     9'b + b -> ~b_1 + ~b_1        ',  'b + b -> ~b_2 + ~b_2        ',
     9'b + b -> ~b_1 + ~b_2        ',  'b + g -> ~b_1 + ~g          ',
     9'b + g -> ~b_2 + ~g          ',  'b + bbar -> ~b_1 + ~b_2bar  ',
     9'f + fbar'' -> H+/- + h0     ',  'f + fbar -> H+/- + H0       ',
     9'f + fbar -> A0 + h0         ',  'f + fbar -> A0 + H0         '/
      DATA (PROC(I),I=301,340)/
     &'f + fbar -> H+ + H-         ',
     &9*'                          ',  'g + g -> g* + g*            ',
     &'q + g -> q*_D + g*          ',  'qi + qj -> q*_Di + q*_Dj    ',
     &'g + g -> q*_D + q*_Dbar     ',  'q  + qbar -> q*_D + q*_Dbar ',
     &'qi + qbarj -> q*Di + q*Sbarj',  'qi + qjbar -> q*Di + q*Dbarj',
     &'qi + qj -> q*_Di + q*_Sj    ',  'qi + qibar -> q*Dj + q*Dbarj',
     &21*'                          '/
      DATA (PROC(I),I=341,380)/
     4'l + l -> H_L++/--           ',  'l + l -> H_R++/--           ',
     4'l + gamma -> H_L++/-- e-/+  ',  'l + gamma -> H_R++/-- e-/+  ',
     4'l + gamma -> H_L++/-- mu-/+ ',  'l + gamma -> H_R++/-- mu-/+ ',
     4'l + gamma -> H_L++/-- tau-/+',  'l + gamma -> H_R++/-- tau-/+',
     4'f + fbar -> H_L++ + H_L--   ',  'f + fbar -> H_R++ + H_R--   ',
     5'f + f -> f'' + f'' + H_L++/-- ',
     5'f + f -> f'' + f'' + H_R++/-- ','f + fbar -> Z_R0            ',
     5'f + fbar'' -> W_R+/-         ',5*'                            ',
     6'                            ',  'f + fbar -> W_L+ W_L-       ',
     6'f + fbar -> W_L+/- pi_T-/+  ',  'f + fbar -> pi_T+ pi_T-     ',
     6'f + fbar -> gamma pi_T0     ',  'f + fbar -> gamma pi_T0''    ',
     6'f + fbar -> Z0 pi_T0        ',  'f + fbar -> Z0 pi_T0''       ',
     6'f + fbar -> W+/- pi_T-/+    ',  '                            ',
     7'f + fbar'' -> W_L+/- Z_L0    ', 'f + fbar'' -> W_L+/- pi_T0   ',
     7'f + fbar'' -> pi_T+/- Z_L0   ', 'f + fbar'' -> pi_T+/- pi_T0  ',
     7'f + fbar'' -> gamma pi_T+/-  ', 'f + fbar'' -> Z0 pi_T+/-     ',
     7'f + fbar'' -> W+/- pi_T0     ',
     7'f + fbar'' -> W+/- pi_T0''    ',
     7'f + fbar'' -> gamma W+/-(ETC)','f + fbar -> gamma Z0 (ETC)',
     7'f + fbar -> Z0 Z0 (ETC)     '/
      DATA (PROC(I),I=381,420)/
     8'f + f'' -> f + f'' (ETC)      ','f + fbar -> f'' + fbar'' (ETC)',
     8'f + fbar -> g + g (ETC)     ',  'f + g -> f + g (ETC)        ',
     8'g + g -> f + fbar (ETC)     ',  'g + g -> g + g (ETC)        ',
     8'q + qbar -> Q + Qbar (ETC)  ',  'g + g -> Q + Qbar (ETC)     ',
     8'                            ',  '                            ',
     9'f + fbar -> G*              ',  'g + g -> G*                 ',
     9'q + qbar -> g + G*          ',  'q + g -> q + G*             ',
     9'g + g -> g + G*             ',  '                            ',
     9 4*'                         ',
     &'g + g -> t + b + H+/-       ',  'q + qbar -> t + b + H+/-    ',
     & 18*'                            '/
      DATA (PROC(I),I=421,460)/
     2'g + g  -> cc~[3S1(1)] + g   ',  'g + g  -> cc~[3S1(8)] + g   ',
     2'g + g  -> cc~[1S0(8)] + g   ',  'g + g  -> cc~[3PJ(8)] + g   ',
     2'g + q  -> q + cc~[3S1(8)]   ',  'g + q  -> q + cc~[1S0(8)]   ',
     2'g + q  -> q + cc~[3PJ(8)]   ',  'q + q~ -> g + cc~[3S1(8)]   ',
     2'q + q~ -> g + cc~[1S0(8)]   ',  'q + q~ -> g + cc~[3PJ(8)]   ',
     3'g + g  -> cc~[3P0(1)] + g   ',  'g + g  -> cc~[3P1(1)] + g   ',
     3'g + g  -> cc~[3P2(1)] + g   ',  'q + g  -> q + cc~[3P0(1)]   ',
     3'q + g  -> q + cc~[3P1(1)]   ',  'q + g  -> q + cc~[3P2(1)]   ',
     3'q + q~ -> g + cc~[3P0(1)]   ',  'q + q~ -> g + cc~[3P1(1)]   ',
     3'q + q~ -> g + cc~[3P2(1)]   ',
     3     21 *'                            '/
      DATA (PROC(I),I=461,500)/
     6'g + g  -> bb~[3S1(1)] + g   ',  'g + g  -> bb~[3S1(8)] + g   ',
     6'g + g  -> bb~[1S0(8)] + g   ',  'g + g  -> bb~[3PJ(8)] + g   ',
     6'g + q  -> q + bb~[3S1(8)]   ',  'g + q  -> q + bb~[1S0(8)]   ',
     6'g + q  -> q + bb~[3PJ(8)]   ',  'q + q~ -> g + bb~[3S1(8)]   ',
     6'q + q~ -> g + bb~[1S0(8)]   ',  'q + q~ -> g + bb~[3PJ(8)]   ',
     7'g + g  -> bb~[3P0(1)] + g   ',  'g + g  -> bb~[3P1(1)] + g   ',
     7'g + g  -> bb~[3P2(1)] + g   ',  'q + g  -> q + bb~[3P0(1)]   ',
     7'q + g  -> q + bb~[3P1(1)]   ',  'q + g  -> q + bb~[3P2(1)]   ',
     7'q + q~ -> g + bb~[3P0(1)]   ',  'q + q~ -> g + bb~[3P1(1)]   ',
     7'q + q~ -> g + bb~[3P2(1)]   ',
     7     21 *'                            '/
 
C...Cross sections and slope offsets.
      DATA SIGT/294*0D0/
 
C...Supersymmetry switches and parameters.
      DATA IMSS/0,
     &  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,
     1  89*0/
      DATA RMSS/0D0,
     &  80D0,160D0,500D0,800D0,2D0,250D0,200D0,800D0,700D0,800D0,
     1  700D0,500D0,250D0,200D0,800D0,400D0,0D0,0.1D0,850D0,0.041D0,
     2   1D0,800D0,1D4,1D4,1D4,0D0,0D0,0D0,24D17,0D0,
     3  10*0D0,  
     4  0D0,1D0,8*0D0,  
     5  49*0D0/
C...Initial values for R-violating SUSY couplings.
C...Should not be changed here. See PYMSIN.
      DATA RVLAM/27*0D0/
      DATA RVLAMP/27*0D0/
      DATA RVLAMB/27*0D0/
 
C...Technicolor switches and parameters
      DATA ITCM/0,
     &  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     1  89*0/
      DATA RTCM/0D0,
     &  82D0,1.333D0,.333D0,0.408D0,1D0,1D0,.0182D0,1D0,0D0,1.333D0,
     1  .05D0,200D0,200D0,0D0,0D0,0D0,0D0,0D0,0D0,0D0,
     2  .283D0,.707D0,0D0,0D0,0D0,1.667D0,250D0,250D0,.707D0,0D0,
     3  .707D0,0D0,1D0,0D0,0D0,0D0,0D0,0D0,0D0,0D0,
     4  1000D0, 1D0, 1D0, 1D0, 1D0, 0D0, 1D0, 3*200D0,
     4  200D0, 48*0D0/
 
C...UED switches and parameters.
C... IUED(0) empty IUED vector element
C... IUED(1) UED ON(=1)/OFF(=0) switch
C... IUED(2) ON(=1)/OFF(=0) switch for gravity mediated decays
C... IUED(3) NFLAVOURS Number of KK excitation quark flavours
C... IUED(4) N the number of large extra dimensions
C... IUED(5) Selects whether the code takes Lambda (=0)
C...         or Lambda*R (=1) as input.
C... IUED(6) With radiative corrections to the masses (=1)
C...         or without (=0)
C...
C... RUED(0) empty RUED vector element
C... RUED(1) RINV (1/R) the curvature of the extra dimension
C... RUED(2) XMD the (4+N)-dimensional Planck scale
C... RUED(3) LAMUED (Lambda cutoff scale)
C... RUED(4) LAMUED/RINV (feasible values are order of 10-20)
C...
      DATA IUED/0,0,0,5,6,0,1,93*0/
      DATA RUED/0.D0,1000D0,5000D0,20000.,20.,95*0D0/

C...Data for histogramming routines.
      DATA IHIST/1000,20000,55,1/
      DATA INDX/1000*0/

C...Data for SUSY Les Houches Accord.
      DATA CPRO/'PYTHIA      ','PYTHIA      '/
      DATA CVER/'6.4         ','6.4         '/
      DATA MODSEL/200*0/
      DATA PARMIN/100*0D0/
      DATA RMSOFT/101*0D0/
      DATA AU/9*0D0/
      DATA AD/9*0D0/
      DATA AE/9*0D0/
 
      END
