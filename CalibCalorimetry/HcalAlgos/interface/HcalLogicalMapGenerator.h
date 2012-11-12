#ifndef HcalLogicalMapGenerator_h
#define HcalLogicalMapGenerator_h

#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
class HcalTopology;

class HcalLogicalMapGenerator {
  public:
         
    HcalLogicalMapGenerator( );
    ~HcalLogicalMapGenerator( );
    HcalLogicalMap createMap(const HcalTopology* topo, unsigned int mapIOV = 4);

  private:

    void buildHBEFTMap(const HcalTopology* topo,
		       std::vector <HBHEHFLogicalMapEntry>&,std::vector <HTLogicalMapEntry>&,std::vector <uint32_t>&,
		       std::vector <uint32_t>&,
		       std::vector <uint32_t>&,
		       std::vector <uint32_t>&,
		       std::vector <uint32_t>&);
    void buildHOXMap  (const HcalTopology* topo,
		       std::vector <HOHXLogicalMapEntry>&  ,std::vector <uint32_t>&,
		       std::vector <uint32_t>&,
		       std::vector <uint32_t>&);
    void buildCALIBMap(const HcalTopology* topo,
		       std::vector <CALIBLogicalMapEntry>& ,std::vector <uint32_t>&,
		       std::vector <uint32_t>&);
    void buildZDCMap  (const HcalTopology* topo,
		       std::vector <ZDCLogicalMapEntry>&   ,std::vector <uint32_t>&,
		       std::vector <uint32_t>&);
    void ConstructTriggerTower(const HcalTopology* topo,
			       std::vector <HTLogicalMapEntry>&,
			       int i_side, int i_eta, int i_phi, int i_dphi, int i_depth, std::string i_det, int i_wedge, int i_rm,
			       int i_pixel, int i_qie, int i_adc, int i_rm_fi, int i_fi_ch, int i_crate, int i_htr, std::string i_fpga,
			       int i_htr_fi, int i_spigot, int i_slb, std::string i_slbin, std::string i_slbin2,
			       std::string i_slnam, int i_rctcra, int i_rctcar, int i_rctcon, std::string i_rctnam, int i_fed
			       );

    unsigned int mapIOV_;

/*****************************Constants**************************/

    static const int NHBHECR    = 9;        //HB HE crates
    static const int NHTRS      = 3;          //Htrs 0, 1, 2 for HB HE
    static const int NHSETS     = 4;         //4 sets of HB/HE htrs

    static const int NHFCR      = 3;          //HF crates

    static const int NHOCR      = 4;          //HO crates
    static const int NHTRSHO    = 4;        //Htrs 0-3 for HO
    static const int NHSETSHO   = 3;       //3 sets of H0 htrs
    static const int NHOETA     = 16;
    static const int NHOPHI     = 72;  

    static const int NCALIBCR   = 13;        //CALIB crates

    static const int NZDCCAB    = 9;         // ZDC cables

    static const int NFBR       = 8;           //Fibers for htrs
    static const int NFCH       = 3;          //3 Fiber channels ranging 0-2
    static const int NTOPBOT    = 2;        //2 FPGAs (top/bottom)
    static const int NRMFIBR    = 6;        //6 rm fibers ranging 2-7
    static const int NRMSLOT    = 4;        //4 rm slots ranging 1-4



    //Global iterator variables
    int i, j;


    //Variables that need to be printed
    int irm,     irm_fi,  iwedge,  ipixel,  iqie,    iadc,  isector;
    int islb,    irctcra, irctcar, irctcon, irctnam, idphi;
    int iside,   ieta,    iphi,    idepth,  icrate,  ihtr;
    int ihtr_fi, ifi_ch,  ispigot, idcc,    idcc_sl, ifed;
    // indexing number
    int inum;
    //CALIB channel function
    int ich_type;
    std::string subdet;
    //ZDC channel function
    int ix, iy, idx, idet_ch, icab;

    std::string rbx, slbin, slbin2, slnam;
    std::string det,  fpga, rctnam, letter;
    char tempbuff[30];
  
    int hbhecrate[NHBHECR];
    int hfcrate[NHFCR];
    int hocrate[NHOCR];
    int fedhbhenum[NHBHECR][2];
    int fedhfnum[NHFCR][2];
    int fedhonum[NHOCR][2];
    int ihslot[NHSETS];
    int ihslotho[NHSETSHO][NHTRSHO];
    int ihbhephis[NHBHECR];
    int ihfphis[NHFCR];
    int ihophis[NHOCR];
    int ihbheetadepth[NHTRS][NTOPBOT][NFBR][NFCH][2];
    int ihfetadepth[NTOPBOT][NFBR][NFCH][2];
    int irm_rmfiHBHE[NHTRS][NTOPBOT][NFBR][2];

    int irm_rmfiHF[NHTRS][NTOPBOT][NFBR][2];
    int ipixelHB[NRMFIBR][NFCH][NRMSLOT];
                                     
    int ipixelHE[NRMFIBR][NFCH][NRMSLOT];
    int iadcquiHBHE[NRMFIBR][NFCH][2];
 
    const char* S_slbin_odd[8];
    const char* S_slbin_even[8];
    const char* rct_rackHBHE[18];
    const char* S_slbin_7[4];
    const char* S_slbin_3[4];
    const char* rct_rackHF[18];
 
    int slb_table[29];
    int HO_RM_table[24][16][2];
    int HO_RM_fi_eta5to15[11];

    int HO_RM_fi_eta16;

    int HO_htr_fi_450eta5to15[2][11];
    int HO_htr_fi_450eta16M[4];
    int HO_htr_fi_450eta16P[4][6];
    int HO_htr_fi_eta4[2][6];
    int HO_htr_fi_eta123[2][6];
    //static const int HO_Block3_LC[][6];
    int HO_RM_fi_eta1to4[72][2][2];
    int ipixelHO[NRMFIBR][NFCH][2];
    std::string letterHO[NRMFIBR][NFCH][2];
    int rmspecialeta[6][6];
    std::string rmspeciallet_code[6][6];  
    std::string rmspecialdet[6][6];
  
  
    //Associated variables
    int hfphi; 
    char sidesign, S_side;
  
    //For slb and rct
    int phi, phideg, etaslb, oddcard, eta2, eta3, phimod8, ietamod;
    int crazy;

    //For HO
    int phmod24, phmod6, sidear, iph, iet, isid, ring, sector;
    bool phi1458, phi271011, phir0v1, phir0v2, phir0v3, phir0v4;
    bool phmod6e450, phmod6e123;

    //HO Patch panel

    int ic,is,ih,itb,ifb,ifc,ifwtb;

    int calibcrate[NCALIBCR];
    int fedcalibnum[NCALIBCR][2];
    int icalibphis[NCALIBCR];
    int icalibsector_min[NCALIBCR];
    int icalibsector_max[NCALIBCR];
    std::string detIDCALIB[NCALIBCR][NTOPBOT];
    int nfbr_max[NCALIBCR][NTOPBOT];
    int calibHOinfo[NHOCR][NTOPBOT][NFBR][2];

};

#endif
