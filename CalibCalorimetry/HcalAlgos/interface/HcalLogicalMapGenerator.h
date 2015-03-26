#ifndef HcalLogicalMapGenerator_h
#define HcalLogicalMapGenerator_h

#include <tuple>
#include <algorithm>
#include <vector>
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

class HcalTopology;

//Here we define the class for HO ReMap, hua.wei@cern.ch
class HOReMap 
{   
 public:

  // Constructor + destructor
  HOReMap ();
  ~HOReMap();

  // Build the HOReMap
  void addFile(const char* file_path, const int & crate_number);

  // Print the hash 
  void printHash();
  
  // Getter functions
  int getHTR     ( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  int getHTR     ( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib );
  
  int getHTRFiber( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  int getHTRFiber( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib );

  int getFPGA    ( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  int getFPGA    ( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib );

  bool isHOX     ( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  bool isHOX     ( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib );

  int getBlock   ( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  int getBlock   ( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib ); 
 
  int getCoupler ( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  int getCoupler ( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib );

  int getCrate ( const int &         ring, const int &         sector, const int & rm, const int & rm_fib );
  int getCrate ( const std::string & ring, const std::string & sector, const int & rm, const int & rm_fib );

 private:

  class box 
  {
   public:
    // Define sides
    enum box_side { NONE, TOP, BOTTOM };

    // Constructors
    box();
    box( const std::vector<std::vector<std::string> > & data,
         const box_side & side, const int & row , const int  & column );
    // Getter functions 
    int      getRing()      { return m_int_ring;       }
    int      getSector()    { return m_int_sector;     }
    int      getRM()        { return m_int_rm;         }
    int      getRMFib()     { return m_int_rm_fib;     }
    box_side getSide()      { return m_side;           }
    int      getSlot()      { return m_int_slot;       }
    int      getHTRFiber()  { return m_int_htr_fiber;  }
    bool     isHOX()        { return m_is_hox;         }
    int      getCoupler()   { return m_int_coupler;    }
    int      getBlock()     { return m_int_block;      }
    int      getCrate()     { return m_int_crate;      }

    // Setter functions
    void     setBlock(int b){ m_int_block = b;         }
    void     setCrate(int c){ m_int_crate = c;         }

    // Hash
    int getHash();

   protected:
    //Raw integers from constructor
    box_side m_side;
    int      m_row;
    int      m_column;

    // Strings taken from raw data
    std::string m_str_ring;
    std::string m_str_sector;
    std::string m_str_rm_rm_fib;

    // Processed integers derived from strings
    int m_int_slot;
    int m_int_ring;
    int m_int_sector;
    int m_int_rm;
    int m_int_rm_fib;
    int m_int_sign;
    int m_int_htr_fiber;
    int m_int_crate;
    int m_int_block;
    int m_int_coupler;

    // Is this HOX?
    bool m_is_hox;
  };


  class ho_box : public box 
  {
   public:
    // constructors
    ho_box ( const std::vector<std::vector<std::string> > & data,
             const box_side & side, const int& row  , const int & column );
    ho_box ();
    // Setter functions
    void setSlot(const int& s){ m_int_slot = s; }

   protected:

    // Strings taken from raw data
    std::string m_str_phi;
    std::string m_str_htr_fiber;

    // Processed integers derived from strings
    int m_int_phi_min;
    int m_int_phi_max;
  };

  class hox_box : public box 
  {
   public:
    // constructor 
    hox_box ( const std::vector<std::vector<std::string> > & data,
              const box_side & side, const int& row  , const int & column );
    //Processed integers derived from strings
    int m_int_crate;

   protected:
    // Strings taken from raw data
    const std::string m_str_crate_slot_htr_fib;
    const std::string m_str_crate;
    const std::string m_str_slot;
    const std::string m_str_htr_fib;

  };

  // Private functions
  void processRawData(const std::vector<std::vector<std::string> > & raw_data, const int & crate_number);
  
  // Hash table info
  const int m_max_hash;
  std::vector<box> * m_hash_table;
  
  // Raw data info
  const size_t m_raw_data_n_columns;
};

namespace tools 
{
  // Split a string into a vector of strings that are split by a delimiter
  void split ( const std::string & content,
               std::vector<std::string> & output,
               const char & delimiter );
  // Clean white space from end of line
  void rtrim (std::string &content);
  // Is this string a number?
  bool is_number(const std::string & content);
};

namespace box_tools 
{
  int getHash( const int & ring,
               const int & sector,
               const int & rm,
               const int & rm_fib );

  int getHash( const std::string & ring,
               const std::string & sector,
               const int & rm,
               const int & rm_fib );
};

//####################//
class HcalLogicalMapGenerator 
{
  public:
         
    HcalLogicalMapGenerator( );
    ~HcalLogicalMapGenerator( );
    HcalLogicalMap createMap(const HcalTopology* topo, unsigned int mapIOV = 5);

  private:

    //function for build qie information, added by hua.wei@cern.ch
    void buildHCALQIEInfo(std::vector <HBHEHFQIEInfo>&,
                          std::vector <HOR0QIEInfo>&,
                          std::vector <HOR12QIEInfo>&);
  
    void buildQIECardNormal(std::vector <QIECardNormal>&);
    void buildQIECardCalib(std::vector <QIECardCalib>&);

    void averageQIEFNAL(const std::vector <HBHEHFQIEInfo>&,
                        const std::vector <HOR0QIEInfo>&,
                        const std::vector <HOR12QIEInfo>&,
                        const std::vector <QIECardNormal>&,
                        float (&fnal_hb_offsets_slopes_ave)[6][32],
                        float (&fnal_he_offsets_slopes_ave)[6][32],
                        float (&fnal_hf_offsets_slopes_ave)[6][32],
                        float (&fnal_ho_offsets_slopes_ave)[6][32],
                        float (&fnal_hb_fcs_ave)[6][16][32],
                        float (&fnal_he_fcs_ave)[6][16][32],
                        float (&fnal_hf_fcs_ave)[6][16][32],
                        float (&fnal_ho_fcs_ave)[6][16][32]
                        );

    int matchNormalInfo(int qieid, int adc, char det[],
                        const std::vector <QIECardNormal>&,
                        const float (&fnal_hb_offsets_slopes_ave)[6][32],
                        const float (&fnal_he_offsets_slopes_ave)[6][32],
                        const float (&fnal_hf_offsets_slopes_ave)[6][32],
                        const float (&fnal_ho_offsets_slopes_ave)[6][32],
                        float (&offsets_slopes)[32]);

    void buildHBEFTMap (const HcalTopology* topo,
		        std::vector <HBHEHFLogicalMapEntry>&,
                        std::vector <HTLogicalMapEntry>&,
                        std::vector <HBHEHFQIEInfo>&,
                        const std::vector <QIECardNormal>&,
                        std::vector <OfflineDB>&,
                        const float (&fnal_hb_offsets_slopes_ave)[6][32],
                        const float (&fnal_he_offsets_slopes_ave)[6][32],
                        const float (&fnal_hf_offsets_slopes_ave)[6][32],
                        const float (&fnal_ho_offsets_slopes_ave)[6][32],
                        std::vector <QIEMap>&,
                        std::vector <uint32_t>&,
                        std::vector <uint32_t>&,
                        std::vector <uint32_t>&,
                        std::vector <uint32_t>&,
                        std::vector <uint32_t>&);

    void buildHOXMap   (const HcalTopology* topo,
		        std::vector <HOHXLogicalMapEntry>&,
                        std::vector <HOR0QIEInfo>&,
                        std::vector <HOR12QIEInfo>&,
                        const std::vector <QIECardNormal>&,
                        std::vector <OfflineDB>&,
                        const float (&fnal_hb_offsets_slopes_ave)[6][32],
                        const float (&fnal_he_offsets_slopes_ave)[6][32],
                        const float (&fnal_hf_offsets_slopes_ave)[6][32],
                        const float (&fnal_ho_offsets_slopes_ave)[6][32],
                        std::vector <QIEMap>&,
                        HOReMap &,
                        std::vector <uint32_t>&,
                        std::vector <uint32_t>&,
                        std::vector <uint32_t>&);
    //this is a patch function for buildHOXMap, plugin at the end of the buildHOXMap;
    //FE swap happens in HO1P06 rm3 and rm4, let_code ABCDEF, we should swap iphi 28 <-> 29; added by hua.wei@cern.ch
    int HO1P06_FE_SWAP (std::string rbx,
                        int rm,
                        std::string let_code,
                        int iphi); 

    void buildCALIBMap (const HcalTopology* topo,
		        std::vector <CALIBLogicalMapEntry>&,
                        std::vector <uint32_t>&,
		        std::vector <uint32_t>&);
    void buildZDCMap   (const HcalTopology* topo,
		        std::vector <ZDCLogicalMapEntry>&,
                        std::vector <uint32_t>&,
		        std::vector <uint32_t>&);

    void ConstructTriggerTower(const HcalTopology* topo,
			       std::vector <HTLogicalMapEntry>&,
			       int i_side, int i_eta, int i_phi, int i_dphi, int i_depth, std::string i_det, int i_wedge, int i_rm,
			       int i_pixel, int i_qie, int i_adc, int i_rm_fi, int i_fi_ch, int i_crate, int i_htr, std::string i_fpga,
			       int i_htr_fi, int i_spigot, int i_slb, std::string i_slbin, std::string i_slbin2,
			       std::string i_slnam, int i_rctcra, int i_rctcar, int i_rctcon, std::string i_rctnam, int i_fed,
                               int i_qieid);

    int matchHBHEHFQIE(int subdet, int side, int rbxno, int rm, int qie,
                       const std::vector <HBHEHFQIEInfo>&);
    int matchHOR0QIE(int rbxno, int rm, int qie, 
                     const std::vector <HOR0QIEInfo>&);
    int matchHOR12QIE(int side, int ring, int rbxno, int rm, int qie, 
                      const std::vector <HOR12QIEInfo>&);
    
    //algorithm to put the offlineDB in correct order
    void offlinedb_order(std::vector <OfflineDB>&);

    //int QIECardNormalsClassify(int qieid, int adc, char det[],
                               //const std::vector <QIECardNormal>&);

    //function to build HOReMap information, added by hua.wei@cern.ch
    void buildHOReMapInformation( HOReMap &myHOReMap );

    unsigned int mapIOV_;

    //###########################Constants###########################//

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
    int iqieid;
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
    int HO_RM_fi_eta1to4[24][2][2];
    int HO_RM_fi_eta1to4_sipm[24][4][2];
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

