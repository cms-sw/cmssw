#ifndef HERWIG_INC
#define HERWIG_INC

//-------------------------HERWIG common block -------------------------------------
// COMMON block stuff, that doesn't come with the HerwigWrapper6_4.h ....

/*C Arrays for particle properties (NMXRES = max no of particles defined)
      PARAMETER(NMXRES=500)
      COMMON/HWPROP/RLTIM(0:NMXRES),RMASS(0:NMXRES),RSPIN(0:NMXRES),
     & ICHRG(0:NMXRES),IDPDG(0:NMXRES),IFLAV(0:NMXRES),NRES,
     & VTOCDK(0:NMXRES),VTORDK(0:NMXRES),
     & QORQQB(0:NMXRES),QBORQQ(0:NMXRES) */

const int nmxres = 500+1; // we need NMXRES+1 entries ...
extern "C" {
  extern struct {
    double RLTIM[nmxres], RMASS[nmxres], RSPIN[nmxres];
    int ICHRG[nmxres], IDPDG[nmxres],IFLAV[nmxres], NRES;
    int VTOCDK[nmxres], VTORDK[nmxres], QORQQB[nmxres], QBORQQ[nmxres];    
  } hwprop_;
}
#define hwprop hwprop_

/*C Parameters for Sudakov form factors
C (NMXSUD= max no of entries in lookup table)
      PARAMETER (NMXSUD=1024)
      COMMON/HWUSUD/ACCUR,QEV(NMXSUD,6),SUD(NMXSUD,6),INTER,NQEV,NSUD,
      & SUDORD*/

const int nmxsud = 1024;
extern "C" {
  extern struct {
    double ACCUR, QEV[6][nmxsud],SUD[6][nmxsud];
    int INTER, NQEV, NSUD, SUDORD;
  } hwusud_;
}
#define hwusud hwusud_

/*C  New parameters for version 6.203
      DOUBLE PRECISION ABWGT,ABWSUM,AVABW
      INTEGER NNEGWT,NNEGEV
      LOGICAL NEGWTS
      COMMON/HW6203/ABWGT,ABWSUM,AVABW,NNEGWT,NNEGEV,NEGWTS */

extern"C" {
  extern struct {
    double ABWGT, ABWSUM, AVABW;
    int NNEGWT,NNEGEV,NEGWTS;
  } hw6203_;
}
#define hw6203 hw6203_


/*CHARACTER*20
     & AUTPDF
     COMMON/HWPRCH/AUTPDF(2),BDECAY   */

extern "C" {
  extern struct {
    char AUTPDF[2][20],BDECAY[4];
  } hwprch_;
}
#define hwprch hwprch_

/*C Parameters for minimum bias/soft underlying event
      COMMON/HWMINB/
      & PMBN1,PMBN2,PMBN3,PMBK1,PMBK2,PMBM1,PMBM2,PMBP1,PMBP2,PMBP3  */

extern "C" {
  extern struct {
    double PMBN1,PMBN2,PMBN3,PMBK1,PMBK2,PMBM1,PMBM2,PMBP1,PMBP2,PMBP3;
  } hwminb_;
}
#define hwminb hwminb_

/*C Variables controling mixing and vertex information
C--VTXPIP should have been a 5-vector, problems with NAG compiler
      COMMON/HWDIST/EXAG,GEV2MM,HBAR,PLTCUT,VMIN2,VTXPIP(5),XMIX(2),
      & XMRCT(2),YMIX(2),YMRCT(2),IOPDKL,MAXDKL,MIXING,PIPSMR */

extern "C" {
  extern struct {
    double EXAG,GEV2MM,HBAR,PLTCUT,VMIN2,VTXPIP[5],XMIX[2],XMRCT[2],YMIX[2],YMRCT[2];
    int IOPDKL,MAXDKL,MIXING,PIPSMR;
  } hwdist_;
}
#define hwdist hwdist_

/*      PARAMETER(NMXCDK=4000)
      COMMON/HWUCLU/CLDKWT(NMXCDK),CTHRPW(12,12),PRECO,RESN(12,12),
      & RMIN(12,12),LOCN(12,12),NCLDK(NMXCDK),NRECO,CLRECO  */

const int nmxcdk=4000;
extern "C" {
  extern struct {
    double CLDKWT[nmxcdk],CTHRPW[12][12],PRECO,RESN[12][12], RMIN[12][12];
    int LOCN[12][12],NCLDK[nmxcdk], NRECO,CLRECO;
  } hwuclu_;
}
#define hwuclu hwuclu_ 

/*C Weights used in cluster decays
      COMMON/HWUWTS/REPWT(0:3,0:4,0:4),SNGWT,DECWT,QWT(3),PWT(12),
      & SWTEF(NMXRES)  */

extern "C" {
  extern struct {
    double REPWT[5][5][4],SNGWT,DECWT,QWT[3],PWT[12],SWTEF[nmxres];
  } hwuwts_;
}
#define hwuwts hwuwts_

/*C  Other new parameters for version 6.2
      DOUBLE PRECISION VIPWID,DXRCYL,DXZMAX,DXRSPH
      LOGICAL WZRFR,FIX4JT
      INTEGER IMSSM,IHIGGS,PARITY,LRSUSY
      COMMON/HW6202/VIPWID(3),DXRCYL,DXZMAX,DXRSPH,WZRFR,FIX4JT,
      & IMSSM,IHIGGS,PARITY,LRSUSY   */

extern "C" {
  extern struct {
    double VIPWID[3], DXRCYL,DXZMAX,DXRSPH;
    int WZRFR,FIX4JT,IMSSM,IHIGGS,PARITY,LRSUSY;
  } hw6202_;
}
#define hw6202 hw6202_

/*      PARAMETER (MODMAX=50)
      COMMON/HWBOSC/ALPFAC,BRHIG(12),ENHANC(12),GAMMAX,RHOHEP(3,NMXHEP),
      & IOPHIG,MODBOS(MODMAX)  */

const int hepevt_size = 4000; // check in HerwigWrapper
const int modmax = 50;
extern "C" {
  extern struct {
    double ALPFAC, BRHIG[12], ENHANC[12], GAMMAX, RHOHEP[hepevt_size][3];
    int IOPHIG, MODBOS[modmax];
  } hwbosc_;
}
#define hwbosc hwbosc_

/*      COMMON/HWHARD/ASFIXD,CLQ(7,6),COSS,COSTH,CTMAX,DISF(13,2),EMLST,
     & EMMAX,EMMIN,EMPOW,EMSCA,EPOLN(3),GCOEF(7),GPOLN,OMEGA0,PHOMAS,
     & PPOLN(3),PTMAX,PTMIN,PTPOW,Q2MAX,Q2MIN,Q2POW,Q2WWMN,Q2WWMX,QLIM,
     & SINS,THMAX,Y4JT,TMNISR,TQWT,XX(2),XLMIN,XXMIN,YBMAX,YBMIN,YJMAX,
     & YJMIN,YWWMAX,YWWMIN,WHMIN,ZJMAX,ZMXISR,IAPHIG,IBRN(2),IBSH,
     & ICO(10),IDCMF,IDN(10),IFLMAX,IFLMIN,IHPRO,IPRO,MAPQ(6),MAXFL,
     & BGSHAT,COLISR,FSTEVT,FSTWGT,GENEV,HVFCEN,TPOL,DURHAM   */

extern "C" {
  extern struct {
    double ASFIXD,CLQ[6][7],COSS,COSTH,CTMAX,DISF[2][13],EMLST, EMMAX,EMMIN,EMPOW,EMSCA,EPOLN[3],GCOEF[7],GPOLN,OMEGA0,PHOMAS, PPOLN[3],PTMAX,PTMIN,PTPOW,Q2MAX,Q2MIN,Q2POW,Q2WWMN,Q2WWMX,QLIM, SINS,THMAX,Y4JT,TMNISR,TQWT,XX[2],XLMIN,XXMIN,YBMAX,YBMIN,YJMAX,YJMIN,YWWMAX,YWWMIN,WHMIN,ZJMAX,ZMXISR;
    int IAPHIG,IBRN[2],IBSH, ICO[10],IDCMF,IDN[10],IFLMAX,IFLMIN,IHPRO,IPRO,MAPQ[6],MAXFL,BGSHAT,COLISR,FSTEVT,FSTWGT,GENEV,HVFCEN,TPOL,DURHAM;
  } hwhard_;
}
#define hwhard hwhard_

/*C other HERWIG branching, event and hard subprocess common blocks
  COMMON/HWBRCH/ANOMSC(2,2),HARDST,PTINT(3,2),XFACT,INHAD,JNHAD,
  & NSPAC(7),ISLENT,BREIT,FROST,USECMF */

extern "C" {
  extern struct {
    double ANOMSC[2][2],HARDST,PTINT[2][3],XFACT;
    int INHAD,JNHAD,NSPAC[7],ISLENT,BREIT,FROST,USECMF;
  } hwbrch_;
}
#define hwbrch hwbrch_

/*      LOGICAL PRESPL
	COMMON /HW6500/ PRESPL   */

extern "C" {
  extern struct {
    int PRESPL;
  } hw6500_;
}
#define hw6500 hw6500_

/*C R-Parity violating parameters and colours
      COMMON /HWRPAR/ LAMDA1(3,3,3),LAMDA2(3,3,3),
      &                LAMDA3(3,3,3),HRDCOL(2,5),RPARTY,COLUPD   */

extern "C" {
  extern struct {
    double LAMDA1[3][3][3],LAMDA2[3][3][3],LAMDA3[3][3][3];
    int HRDCOL[5][2],RPARTY,COLUPD;
  } hwrpar_;
}
#define hwrpar hwrpar_

/*C SUSY parameters
      COMMON/HWSUSY/
     & TANB,ALPHAH,COSBPA,SINBPA,COSBMA,SINBMA,COSA,SINA,COSB,SINB,COTB,
     & ZMIXSS(4,4),ZMXNSS(4,4),ZSGNSS(4), LFCH(16),RFCH(16),
     & SLFCH(16,4),SRFCH(16,4), WMXUSS(2,2),WMXVSS(2,2), WSGNSS(2),
     & QMIXSS(6,2,2),LMIXSS(6,2,2),
     & THETAT,THETAB,THETAL,ATSS,ABSS,ALSS,MUSS,FACTSS,
     & GHWWSS(3),GHZZSS(3),GHDDSS(4),GHUUSS(4),GHWHSS(3),
     & GHSQSS(4,6,2,2),XLMNSS,RMMNSS,DMSSM,SENHNC(24),SSPARITY,SUSYIN  */

extern "C" {
  extern struct {
    double TANB,ALPHAH,COSBPA,SINBPA,COSBMA,SINBMA,COSA,SINA,COSB,SINB,COTB,ZMIXSS[4][4],ZMXNSS[4][4],ZSGNSS[4], LFCH[16],RFCH[16],SLFCH[4][16],SRFCH[4][16], WMXUSS[2][2],WMXVSS[2][2], WSGNSS[2],QMIXSS[2][2][6],LMIXSS[2][2][6],THETAT,THETAB,THETAL,ATSS,ABSS,ALSS,MUSS,FACTSS,GHWWSS[3],GHZZSS[3],GHDDSS[4],GHUUSS[4],GHWHSS[3],GHSQSS[2][2][6][4],XLMNSS,RMMNSS,DMSSM,SENHNC[24],SSPARITY;
    int SUSYIN;
  } hwsusy_;
}
#define hwsusy hwsusy_

/*INTEGER NDECSY,NSEARCH,LRDEC,LWDEC
      LOGICAL SYSPIN,THREEB,FOURB
      CHARACTER *6 TAUDEC
      COMMON /HWDSPN/NDECSY,NSEARCH,LRDEC,LWDEC,SYSPIN,THREEB,
      &	FOURB,TAUDEC */

extern "C" {
  extern struct {
    int NDECSY,NSEARCH,LRDEC,LWDEC,SYSPIN,THREEB,FOURB;
    char TAUDEC[6];
  } hwdspn_;
}

#define hwdspn hwdspn_

/*C--common block for Les Houches interface to store information we need
C
      INTEGER MAXHRP
      PARAMETER (MAXHRP=100)
      DOUBLE PRECISION LHWGT(MAXHRP),LHWGTS(MAXHRP),LHMXSM,
     &     LHXSCT(MAXHRP),LHXERR(MAXHRP),LHXMAX(MAXHRP)
      INTEGER LHIWGT(MAXHRP),ITYPLH,LHNEVT(MAXHRP)
      LOGICAL LHSOFT,LHGLSF	
      COMMON /HWGUPR/LHWGT,LHWGTS,LHXSCT,LHXERR,LHXMAX,LHMXSM,LHIWGT,
      &     LHNEVT,ITYPLH,LHSOFT,LHGLSF  */

const int maxhrp = 100;
extern "C" {
  extern struct {
    double LHWGT[maxhrp],LHWGTS[maxhrp],LHMXSM,LHXSCT[maxhrp],LHXERR[maxhrp],LHXMAX[maxhrp];
    int LHIWGT,LHNEVT,ITYPLH,LHSOFT,LHGLSF;
  } hwgupr_;
}
#define hwgupr hwgupr_

/*C  New parameters for version 6.3
      INTEGER IMAXCH,IMAXOP
      PARAMETER (IMAXCH=20,IMAXOP=40)
      DOUBLE PRECISION MJJMIN,CHNPRB(IMAXCH)
      INTEGER IOPSTP,IOPSH
      LOGICAL OPTM,CHON(IMAXCH)
      COMMON/HW6300/MJJMIN,CHNPRB,IOPSTP,IOPSH,OPTM,CHON   */

const int imaxch = 20;
extern "C" {
  extern struct {
    double MJJMIN,CHNPRB[imaxch];
    int IOPSTP,IOPSH,OPTM,CHON[imaxch];
  } hw6300_;
}
#define hw6300 hw6300_

//-------------------------- JIMMY COMMON BLOCK -------------------------------
/*
      DOUBLE PRECISION YGAMMA, JMZMIN, JMRAD, PTJIM
      DOUBLE PRECISION PHAD, JMU2, JMV2, SMALL, JMARRY
c     JMARRY is the array storing gamma-p xsec at various z, & 
c	max weight for each z
      DOUBLE PRECISION TOTSCAT, NLOST

      INTEGER MAXMS, NPSIMP, MSFLAG, JMPTYP, JCMVAR, NPROC
      LOGICAL ANOMOFF

      PARAMETER( NPROC = 117 )
      PARAMETER( MAXMS  = 100  )  ! Maximum multiple scatters
      PARAMETER( NPSIMP = 16 )  ! No. of Simpson rule (YBJ)
C                                 intervals (must be even)
      PARAMETER( SMALL  = 1.0D-20  )
      INTEGER JMOUT, JMBUG, FN_TYPE, NSCAT, JMUEO, MAXMSTRY
      PARAMETER(JMOUT = 6)
      COMMON / JMPARM /  PTJIM, YGAMMA, JMZMIN, JMRAD(264)
     &     ,PHAD, JMU2, JMV2, JMARRY( 6+MAXMS,0:NPSIMP )
     &     ,NLOST, TOTSCAT, ANOMOFF, JCMVAR, JMUEO
     &     ,JMPTYP(NPROC), JMBUG, FN_TYPE, MSFLAG, MAXMSTRY
      DOUBLE PRECISION JMPROC, JMVETO
      COMMON / JMEVNT/ JMPROC(NPROC)
     &,        JMVETO(2,13), NSCAT
*/

const int NPROC = 117;
const int MAXMS = 100;
const int NPSIMP = 16;
const double SMALL = 0.00000000000000000001;

extern "C" {
  extern struct {
    double PTJIM,YGAMMA,JMZMIN,JMRAD[264],PHAD,JMU2,JMV2,JMARRY[NPSIMP+1][6+MAXMS],
      NLOST,TOTSCAT;
    int ANAMOFF,JCMVAR,JMUEO,JMPTYP[NPROC],JMBUG,FN_TYPE,MSFLAG,MAXMSTRY;
  } jmparm_;
}
#define jmparm jmparm_

extern "C" {
  extern struct {
    double JMPROC[NPROC],JMVETO[13][2];
    int NSCAT;
  } jmevnt_;
}
#define jmevnt jmevnt_

//------------------------------ JIMMY functions -------------------------------------------------
extern"C" {
  void jimmin_(void);
  void jminit_(void);
  double hwmsct_dummy_(double);
  void jmefin_(void);
}

#define jimmin jimmin_
#define jminit jminit_
#define hwmsct_dummy hwmsct_dummy_
#define jmefin jmefin_


//------------------------------ LHAPDF functions -------------------------------------------------


//------------------------------ POMWIG functions -------------------------------------------------

// Subroutine inside H1QCD
#define qcd_1994 qcd_1994_
extern "C" {
    void qcd_1994(double&,double&,double*,int);
}
// HWABEG to initialize H1 pomeron structure function
#define hwabeg hwabeg_
extern "C" {
    void hwabeg();
}
#endif
