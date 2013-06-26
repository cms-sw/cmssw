#ifndef big_h
#define big_h 1
#include <vector>
#include <string>
#include "CondFormats/Calibration/interface/TensorIndex.h"
class big{
 public:
  big():id_current(-1),index_id(-1),cota_current(0.),cotb_current(0.),fpix_current(false){
    //constructor
    tVector_.reserve(1000);
    thVector_.reserve(1000);
    sVector_.reserve(1000);
  }
  void fill(size_t tVectorSize,size_t thVectorSize,size_t sVectorSize,
	    const std::string& atitle);
  
  ///inline class bigEntry
  class bigEntry{
  public:
    //constructor
    bigEntry(){
      par.reserve(parIDX::SIZE);
      ytemp.reserve(ytempIDX::SIZE);
      xtemp.reserve(xtempIDX::SIZE);
      avg.reserve(avgIDX::SIZE);
      aqfl.reserve(aqflIDX::SIZE);
      chi2.reserve(chi2IDX::SIZE);
      spare.reserve(spareIDX::SIZE);  
    }
    void fill(int runnum,float seed);
  public:  
    int runnum;
    float alpha;
    float cotalpha;
    float beta;
    float cotbeta ;
    float costrk[3];
    float qavg;
    float symax;  
    float dyone;  
    float syone;
    float sxmax;
    float dxone;   
    float sxone; 
    float dytwo;  
    float sytwo;      
    float dxtwo; 
    float sxtwo;     
    float qmin;   
    //projected pixel uncertainty parameterization, first dimension x,y; 
    typedef TensorIndex<2,2,5> parIDX;
    std::vector<float> par; 
    //templates for y-reconstruction (binned over 1 central pixel)
    typedef TensorIndex<9,21> ytempIDX;
    std::vector<float> ytemp;
    //templates for x-reconstruction (binned over 1 central pixel)
    typedef TensorIndex<9,7> xtempIDX;
    std::vector<float> xtemp;
    //average parameters (binned in 4 charge bins ), first dimention x,y; second dimention bias,rms,g0,sigma; 
    typedef TensorIndex<2,4,4> avgIDX;
    std::vector<float> avg;
    //Aqfl-parameterized x,y-correction (in 4 charge bins), first dimension x,y
    typedef TensorIndex<2,4,6> aqflIDX;
    std::vector<float> aqfl;
    //chi^2 (in 4 charge bins), first dimension x,y; second dimension average, minimum; 
    typedef TensorIndex<2,2,4> chi2IDX;
    std::vector<float> chi2;
    //spare entries, first dimension x,y
    typedef TensorIndex<2,10> spareIDX;
    std::vector<float> spare;  
  };//inline class bigEntry
  
  //inline class bigHeader
  class bigHeader{
  public:
    bigHeader():title(""){}
    void fill (const std::string& atitle);
    /// data members
    std::string title;      //!< template title 
    int ID;                 //!< template ID number 
    int NBy;                //!< number of Barrel y entries 
    int NByx;               //!< number of Barrel y-slices of x entries 
    int NBxx;               //!< number of Barrel x entries in each slice
    int NFy;                //!< number of FPix y entries 
    int NFyx;               //!< number of FPix y-slices of x entries 
    int NFxx;               //!< number of FPix x entries in each slice
    float vbias;            //!< detector bias potential in Volts 
    float temperature;      //!< detector temperature in deg K 
    float fluence;          //!< radiation fluence in n_eq/cm^2 
    float qscale;           //!< Charge scaling to match cmssw and pixelav 
    float s50;              //!< 1/2 of the readout threshold in ADC units 
    int templ_version;      //!< Version number of the template to ensure code compatibility 

  };//end inline class bigHeader

  //inline class bigStore
  class bigStore{
  public:
    //constructor
    bigStore(){
      entby.reserve(entbyIDX::SIZE);
      entbx.reserve(entbxIDX::SIZE);
      entfy.reserve(entfyIDX::SIZE);
      entfx.reserve(entfxIDX::SIZE);
    }
    //dummy filler
    void fill( const std::string& atitle  );
    //data members
    bigHeader head;
    typedef  TensorIndex<60> entbyIDX;
    std::vector<bigEntry> entby;
    typedef  TensorIndex<5,9> entbxIDX;
    std::vector<bigEntry> entbx;
    typedef  TensorIndex<5> entfyIDX;
    std::vector<bigEntry> entfy;
    typedef  TensorIndex<2,9> entfxIDX;
    std::vector<bigEntry> entfx;
  };//end inline class bigStore


  typedef std::vector<bigEntry> entryVector;
  typedef std::vector<bigHeader> headVector;
  typedef std::vector<bigStore> storeVector;
 private:
  entryVector tVector_;
  headVector thVector_;
  storeVector sVector_;
  int id_current;           //!< current id
  int index_id;             //!< current index
  float cota_current;       //!< current cot alpha
  float cotb_current;       //!< current cot beta
  float abs_cotb;           //!< absolute value of cot beta
  bool fpix_current;        //!< current pix detector (false for BPix, true for FPix)
};//end big
#endif
