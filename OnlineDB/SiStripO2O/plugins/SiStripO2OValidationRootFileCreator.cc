// -*- C++ -*-
//
// Package:    SiStripO2OValidationRootFile
// Class:      SiStripO2OValidationRootFile
// 
/**\class SiStripO2OValidationRootFile SiStripO2OValidationRootFile.cc O2OValidation/Pedestal/src/SiStripO2OValidationRootFile.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  
//         Created:  Mon Jan  4 13:22:08 CET 2010
// $Id$
//
//


// system include files
#include <memory>
#include <FWCore/Version/interface/GetReleaseVersion.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

//data containers
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"

//ES data records
// #include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
// #include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
// #include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
// #include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
// #include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
// #include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
// #include "DPGAnalysis/SiStripTools/src/APVLatencyRcd.cc"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"



//root includes 
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "TBranch.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TIterator.h"
#include "TKey.h"
#include "TLegend.h"
#include "TObject.h"
#include "TObjString.h"
#include "TString.h"
#include "TTree.h"


//
// class decleration
//

class SiStripO2OValidationRootFile : public edm::EDAnalyzer {
public:
  explicit SiStripO2OValidationRootFile(const edm::ParameterSet&);
  ~SiStripO2OValidationRootFile();
  
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void FEDCablingValidation(const edm::EventSetup& iSetup, int treenr);
  void NoiseValidation(const edm::EventSetup& iSetup, int treenr);
  void PedestalValidation(const edm::EventSetup& iSetup, int treenr);
  void ThresholdValidation(const edm::EventSetup& iSetup, int treenr);
  void QualityValidation(const edm::EventSetup& iSetup, int treenr);
  void APVTimingValidation(const edm::EventSetup& iSetup, int treenr);
  void APVLatencyValidation(const edm::EventSetup& iSetup, int treenr);
  
  //Tree variables
  int nroftrees;
  TFile *val_file;
  TTree *val_tree[7];
  TrackerMap *val_map[100];
  uint32_t tree_detid;
  //Noise
  float integrated_noise;
  //Threshold
  float inththreshold;
  float intlthreshold;
  float intcthreshold;
  //Pedestal
  float intpedestal;
  //FED Cabling
    int	tree_fedcr;
    int	tree_fedsl;
    int	tree_fedid;
    int   tree_feunit;
    int   tree_fechan;
    int   tree_fedchan;
    int   tree_feccr;
    int   tree_fecsl;
    int   tree_fecring;
    int	tree_ccuaddr;
    int	tree_module;
    int   tree_apv1;
    int	tree_apv2;
    int	tree_apvpair;
    int   tree_nrapvpairs;
    int   tree_dcuid;
  //Bad Strips
  int tree_nrbadstrips;
  //APV Tick Height
  float tree_tickheight;
  int tree_apvnr;
  //APV Latency
  int tree_apvmode;
  int tree_apvlatency;

  //config parameters
  bool cfg_FEDCabling; int treenrfed;
  bool cfg_Threshold;  int treenrthr;
  bool cfg_Quality;    int treenrqua;
  bool cfg_Noise;      int treenrnoi;
  bool cfg_Pedestal;   int treenrped;
  bool cfg_APVLatency; int treenrlat;
  bool cfg_APVTiming;  int treenrtim;
  TString cfg_rootfile;
  bool cfg_debug;     
  
  //Names and Titles
  TString cmssw_version;
  TString tree_name;
  
  //Struct needed to speed up fed cabling validation
  struct fedcabling{
    bool operator() (fedcabling i, fedcabling j) { return (i.tree_detid<j.tree_detid);}
    uint32_t tree_detid;
    int	tree_fedcr;
    int	tree_fedsl;
    int	tree_fedid;
    int   tree_feunit;
    int   tree_fechan;
    int   tree_fedchan;
    int   tree_feccr;
    int   tree_fecsl;
    int   tree_fecring;
    int	tree_ccuaddr;
    int	tree_module;
    int   tree_apv1;
    int	tree_apv2;
    int	tree_apvpair;
    int   tree_nrapvpairs;
    int   tree_dcuid;
  };
  
  fedcabling trackercabling;
  std::vector<fedcabling> v_trackercabling;

    struct noise{
    bool operator() (noise i, noise j) { return (i.tree_detid<j.tree_detid);}
      uint32_t tree_detid;
      float integrated_noise;
    };
  
  noise trackernoise;
  std::vector<noise> v_trackernoise;

    struct pedestal{
    bool operator() (pedestal i, pedestal j) { return (i.tree_detid<j.tree_detid);}
      uint32_t tree_detid;
      float intpedestal;
    };
  
  pedestal trackerpedestal;
  std::vector<pedestal> v_trackerpedestal;

    struct threshold{
    bool operator() (threshold i, threshold j) { return (i.tree_detid<j.tree_detid);}
      uint32_t tree_detid;
      float inththreshold;
      float intlthreshold;
      float intcthreshold;
    };
  threshold trackerthreshold;
  std::vector<threshold> v_trackerthreshold;

 
  struct timing{
    bool operator() (timing i, timing j) { return (i.tree_detid<j.tree_detid);}
    uint32_t tree_detid;
    float tree_tickheight;
    int tree_apvnr;
  };
  
  timing trackertiming;
  std::vector<timing> v_trackertiming;


  
  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiStripO2OValidationRootFile::SiStripO2OValidationRootFile(const edm::ParameterSet& iConfig):
  cfg_FEDCabling(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateFEDCabling",false))),
  cfg_Threshold(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateThreshold",false))),
  cfg_Quality(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateQuality",false))),
  cfg_Noise(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateNoise",false))),
  cfg_Pedestal(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidatePedestal",false))),
  cfg_APVLatency(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateAPVLatency",false))),
  cfg_APVTiming(static_cast<bool>(iConfig.getUntrackedParameter<bool>("ValidateAPVTiming",false))),
  cfg_rootfile(static_cast<TString>(iConfig.getUntrackedParameter<std::string>("RootFile","SiStripO2OValidation.root"))),
 cfg_debug(static_cast<bool>(iConfig.getUntrackedParameter<bool>("DebugMode",false)))
{
  using namespace edm;
   nroftrees=0;
    //Objects
   cmssw_version=edm::getReleaseVersion();
   std::cout<< cmssw_version<<std::endl;
   TObjArray *array  = cmssw_version.Tokenize("\"");
   TObjString *tmpstr[100];
   
   std::cout << "array->GetSize():"<<array->GetSize() << std::endl;
   for (int i=0; i<(array->GetSize()-1);i++){
     tmpstr[i]=((TObjString*)(array->At(i)));
     std::cout << "i["<<i<<"]"<<": " << tmpstr[i]->GetString()<< std::endl;
     if((tmpstr[i]->GetString()).Contains("CMSSW")){
       cmssw_version=tmpstr[i]->GetString();
       break;
     }
   }
   val_file=new TFile(cfg_rootfile,"update");
 }


SiStripO2OValidationRootFile::~SiStripO2OValidationRootFile()
{
  std::cout << "[SiStripO2OValidationRootFile::~SiStripO2OValidationRootFile()]" << std::endl;
  if( cfg_FEDCabling )   val_tree[treenrfed]->Write();
  if( cfg_Pedestal )     val_tree[treenrped]->Write();
  if( cfg_Noise )        val_tree[treenrnoi]->Write();
  if( cfg_Quality )      val_tree[treenrqua]->Write();
  if( cfg_Threshold )    val_tree[treenrthr]->Write();
  if( cfg_APVLatency )   val_tree[treenrlat]->Write();
  if( cfg_APVTiming )    val_tree[treenrtim]->Write();
  val_file->Close();
}

//////////////////////////////////
void SiStripO2OValidationRootFile::FEDCablingValidation(const edm::EventSetup& iSetup, int treenr){
  std::cout << "SiStripO2OValidationRootFile::FEDCablingValidation: treenr: " << treenr << std::endl;
  //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_FEDCabling_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);

  //set branches
  val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
  val_tree[treenr]->Branch("DCUID",&tree_dcuid, "DCUID/I");
  val_tree[treenr]->Branch("FEDCR",&tree_fedcr, "FEDCR/I");
  val_tree[treenr]->Branch("FEDSL",&tree_fedsl, "FEDSL/I");
  val_tree[treenr]->Branch("FEDID",&tree_fedid, "FEDID/I");
  val_tree[treenr]->Branch("FEUNIT",&tree_feunit, "FEUNIT/I");
  val_tree[treenr]->Branch("FECHAN",&tree_fechan, "FECHAN/I");
  val_tree[treenr]->Branch("FEDCHAN",&tree_fedchan, "FEDCHAN/I");
  val_tree[treenr]->Branch("FECCR",&tree_feccr, "FECCR/I");
  val_tree[treenr]->Branch("FECSL",&tree_fecsl, "FECSL/I");
  val_tree[treenr]->Branch("FECRING",&tree_fecring, "FECRING/I");
  val_tree[treenr]->Branch("CCUADDR",&tree_ccuaddr, "CCUADDR/I");
  val_tree[treenr]->Branch("MODULE",&tree_module, "MODULE/I");
  val_tree[treenr]->Branch("APV1",&tree_apv1, "APV1/I");
  val_tree[treenr]->Branch("APV2",&tree_apv2, "APV2/I");
  val_tree[treenr]->Branch("APVPAIR",&tree_apvpair, "APVPAIR/I");
  val_tree[treenr]->Branch("NRAPVPAIRS",&tree_nrapvpairs, "NRAPVPAIRS/I");

  edm::ESHandle<SiStripFedCabling> fed;
  iSetup.get<SiStripFedCablingRcd>().get( fed ); 
 
  std::stringstream ss;
 
  const std::vector<uint16_t>& fed_ids = fed->feds();
  std::vector<uint16_t>::const_iterator ifed = fed_ids.begin(); 
  for ( ; ifed != fed_ids.end(); ifed++ ) {
    const std::vector<FedChannelConnection>& conns = fed->connections(*ifed);
    uint16_t connected = 0;
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) { 
      if ( iconn->fedId() < sistrip::valid_ ) { 
	connected++; 
	//iconn->terse(ss); 
        //ss<< std::endl;
	trackercabling.tree_fedcr=iconn->fedCrate();
	trackercabling.tree_fedsl=iconn->fedSlot();
	trackercabling.tree_fedid=iconn->fedId();
        trackercabling.tree_feunit= SiStripFedKey::feUnit( iconn->fedCh() );
        trackercabling.tree_fechan=SiStripFedKey::feChan( iconn->fedCh() );
        trackercabling.tree_fedchan=iconn->fedCh();
        trackercabling.tree_feccr=iconn->fecCrate();
        trackercabling.tree_fecsl=iconn->fecSlot();
        trackercabling.tree_fecring=iconn->fecRing();
	trackercabling.tree_ccuaddr=iconn->ccuAddr();
	trackercabling.tree_module=iconn->ccuChan();
        trackercabling.tree_apv1=iconn->i2cAddr(0);
	trackercabling.tree_apv2=iconn->i2cAddr(1);
	trackercabling.tree_apvpair=iconn->apvPairNumber()+1;
        trackercabling.tree_nrapvpairs=iconn->nApvPairs();
        trackercabling.tree_dcuid=iconn->dcuId();
        trackercabling.tree_detid= iconn->detId();
	v_trackercabling.push_back(trackercabling);
	//val_tree[treenr]->Fill();
      } 
    }
  }
  std::sort(v_trackercabling.begin(), v_trackercabling.end(), trackercabling);
  for(std::vector<fedcabling>::iterator iter=v_trackercabling.begin(); iter!=v_trackercabling.end();iter++){
    tree_detid=iter->tree_detid;
    tree_fedcr=iter->tree_fedcr;
    tree_fedsl=iter->tree_fedsl;
    tree_fedid=iter->tree_fedid;
    tree_feunit=iter->tree_feunit;
    tree_fechan=iter->tree_fechan;
    tree_fedchan=iter->tree_fedchan;
    tree_feccr=iter->tree_feccr;
    tree_fecsl=iter->tree_fecsl;
    tree_fecring=iter->tree_fecring;
    tree_ccuaddr=iter->tree_ccuaddr;
    tree_module=iter->tree_module;
    tree_apv1=iter->tree_apv1;
    tree_apv2=iter->tree_apv2;
    tree_apvpair=iter->tree_apvpair;
    tree_nrapvpairs=iter->tree_nrapvpairs;
    tree_dcuid=iter->tree_dcuid;
    val_tree[treenr]->Fill();
  }
}

//////////////////////////////////
void SiStripO2OValidationRootFile::QualityValidation(const edm::EventSetup& iSetup, int treenr){
 std::cout << "SiStripO2OValidationRootFile::QualityValidation: treenr: " << treenr << std::endl;
  //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_BadStrip_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);
 
    //set branches
  val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
  val_tree[treenr]->Branch("NRBADSTRIPS",&tree_nrbadstrips, "NRBADSTRIPS/I");

  //  edm::ESHandle<SiStripQuality> SiStripQuality_;
  //iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);
  edm::ESHandle<SiStripBadStrip> SiStripBadStrip_;
  iSetup.get<SiStripBadStripRcd>().get(SiStripBadStrip_);
  std::stringstream ss;
  std::vector<uint32_t> DetIds_;
  SiStripBadStrip_->getDetIds(DetIds_);
  if(DetIds_.begin()==DetIds_.end()) std::cout << "BadStrips Objects empty!!!" << std::endl;
  for(std::vector<uint32_t>::iterator it=DetIds_.begin(); it!=DetIds_.end(); it++){
    tree_nrbadstrips=0;
    tree_detid=*it;
    SiStripBadStrip::Range range(SiStripBadStrip_->getRange(*it));
    for( std::vector<unsigned int>::const_iterator badStrip = range.first;
	 badStrip != range.second; ++badStrip ) {
      ss << "strip: " << *badStrip << std::endl;
      tree_nrbadstrips++;
    }
    val_tree[treenr]->Fill();
  }
 }

//////////////////////////////////
void SiStripO2OValidationRootFile::APVLatencyValidation(const edm::EventSetup& iSetup, int treenr){
  std::cout << "SiStripO2OValidationRootFile::APVLatencyValidation: treenr: " << treenr << std::endl;
  //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_APVLatency_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);
   
  //set branches
   val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
   val_tree[treenr]->Branch("APVNR",&tree_apvnr, "APVNR/I");
   val_tree[treenr]->Branch("APVLATENCY",&tree_apvlatency, "APVLATENCY/I");
   val_tree[treenr]->Branch("APVMODE",&tree_apvmode, "APVMODE/I");

   edm::ESHandle<SiStripLatency> SiStripLatency_;
   iSetup.get<SiStripLatencyRcd>().get(SiStripLatency_);

   std::stringstream ss;
   std::vector<SiStripLatency::Latency> v_latencies;
   v_latencies=SiStripLatency_->allLatencyAndModes();
   
  for( SiStripLatency::latConstIt it = v_latencies.begin(); it != v_latencies.end(); ++it ) {
    tree_detid = it->detIdAndApv >> 3;
    tree_apvnr = it->detIdAndApv & 7; // 7 is 0...0111
    tree_apvlatency=static_cast<int>(it->latency);
    tree_apvmode=static_cast<int>(it->mode);
  }
  val_tree[treenr]->Fill();
}

//////////////////////////////////
void SiStripO2OValidationRootFile::APVTimingValidation(const edm::EventSetup& iSetup, int treenr){
   std::cout << "SiStripO2OValidationRootFile::APVTimingValidation: treenr: " << treenr << std::endl;
  //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_APVTiming_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);
   
  //set branches
  val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
  val_tree[treenr]->Branch("APVNR",&tree_apvnr, "APVNR/I");
  val_tree[treenr]->Branch("TICKHEIGHT",&tree_tickheight, "TICKHEIGHT/F");

  edm::ESHandle<SiStripApvGain> SiStripApvGain_;
  iSetup.get<SiStripApvGainRcd>().get(SiStripApvGain_);
 
  std::vector<uint32_t> v_detids;
  SiStripApvGain_->getDetIds(v_detids);
   edm::LogInfo("[SiStripO2OValidationRootFile::APVGainValidation] Number of detids ")  << v_detids.size() << std::endl;
 
     for (size_t id=0;id<v_detids.size();id++)
       {
         SiStripApvGain::Range range=SiStripApvGain_->getRange(v_detids[id]);
	 int apv=0;
         for(int it=0;it<range.second-range.first;it++){
	              edm::LogInfo("SiStripApvGainReader")  << "detid " << v_detids[id] << " \t"
                                              << " apv " << apv << " \t"
                                              << SiStripApvGain_->getApvGain(it,range)     << " \t" 
                                              << std::endl;          
	   trackertiming.tree_detid=v_detids[id];
	   trackertiming.tree_apvnr=apv++;
	   trackertiming.tree_tickheight=SiStripApvGain_->getApvGain(it,range);
           v_trackertiming.push_back(trackertiming);
         } 
       }

std::sort(v_trackertiming.begin(), v_trackertiming.end(), trackertiming); 
for(std::vector<timing>::iterator iter=v_trackertiming.begin(); iter!=v_trackertiming.end(); ++iter){
     tree_detid=iter->tree_detid;
     tree_apvnr=iter->tree_apvnr;
     tree_tickheight=iter->tree_tickheight;
     val_tree[treenr]->Fill();
   }

}

//////////////////////////////////
void SiStripO2OValidationRootFile::PedestalValidation(const edm::EventSetup& iSetup, int treenr){
  std::cout << "SiStripO2OValidationRootFile::PedestalValidation: treenr: " << treenr << std::endl;
  //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_Pedestal_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);
   
  //set branches
  val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
  val_tree[treenr]->Branch("INTPEDESTAL",&intpedestal, "INTPEDESTAL/F");

  edm::ESHandle<SiStripPedestals> SiStripPedestals_;
  iSetup.get<SiStripPedestalsRcd>().get(SiStripPedestals_);
   
  std::vector<uint32_t> detid;
  SiStripPedestals_->getDetIds(detid);
  std::cout << "detid.size(): "<<detid.size() << std::endl;
  for (size_t id=0;id<detid.size();id++)
      {
	SiStripPedestals::Range range=SiStripPedestals_->getRange(detid[id]);
	trackerpedestal.tree_detid=detid[id];
	for(int it=0;it<(range.second-range.first)*8/10;it++){
	  intpedestal+=SiStripPedestals_->getPed(it,range);
	}
	trackerpedestal.intpedestal=intpedestal;
	v_trackerpedestal.push_back(trackerpedestal);
	intpedestal=0.;
      }

 std::sort(v_trackerpedestal.begin(), v_trackerpedestal.end(), trackerpedestal); 
 for(std::vector<pedestal>::iterator iter=v_trackerpedestal.begin(); iter!=v_trackerpedestal.end(); ++iter){
     tree_detid=iter->tree_detid;
     intpedestal=iter->intpedestal;
     val_tree[treenr]->Fill();
   }

  // std::stringstream ss;
  // SiStripPedestals_->printSummary(ss);
  // std::cout << ss.str() << std::endl;
}

//////////////////////////////////
void SiStripO2OValidationRootFile::NoiseValidation(const edm::EventSetup& iSetup, int treenr){
  std::cout << "SiStripO2OValidationRootFile::NoiseValidation: treenr: " << treenr << std::endl;
   using namespace edm; 
   val_map[0]=new TrackerMap("SiStripO2OValidationMap");
   
   //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_Noise_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);
   

  //set branches
  val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
  val_tree[treenr]->Branch("INTEGRATEDNOISE",&integrated_noise, "INTEGRATEDNOISE/F");

  std::cout << "about to get noise " << std::endl;
   edm::ESHandle<SiStripNoises> SiStripNoises_;
   iSetup.get<SiStripNoisesRcd>().get(SiStripNoises_);
  std::cout << "got noise from dbfile " << std::endl;
    

   std::vector<uint32_t> detid;
   SiStripNoises_->getDetIds(detid);

   
   for (size_t id=0;id<detid.size();id++)
       {
	 SiStripNoises::Range range=SiStripNoises_->getRange(detid[id]);
	 
	 //	 int strip=0;
	 for(int it=0;it<(range.second-range.first)*8/9;it++){
	   integrated_noise+=SiStripNoises_->getNoise(it,range);
	 }
	 trackernoise.tree_detid=detid[id];
	 trackernoise.integrated_noise=integrated_noise;
	 v_trackernoise.push_back(trackernoise);
	 integrated_noise=0.;
       }
   std::sort(v_trackernoise.begin(), v_trackernoise.end(), trackernoise);
   for(std::vector<noise>::iterator iter=v_trackernoise.begin(); iter!=v_trackernoise.end(); ++iter){
     tree_detid=iter->tree_detid;
     integrated_noise=iter->integrated_noise;
     val_tree[treenr]->Fill();
   }
   
 }

//////////////////////////////////
void SiStripO2OValidationRootFile::ThresholdValidation(const edm::EventSetup& iSetup, int treenr){
  std::cout << "SiStripO2OValidationRootFile::ThresholdValidation: treenr: " << treenr << std::endl;
  //Create TTree
   tree_name="ValidationTree";
   std::cout<< tree_name<<std::endl;
   tree_name=tree_name+"_Threshold_"+cmssw_version;
   std::cout<< tree_name<<std::endl;
   val_tree[treenr]=new TTree(tree_name,tree_name);
   

  //set branches
  val_tree[treenr]->Branch("DETID",&tree_detid, "DETID/I");
  val_tree[treenr]->Branch("INTHTHRESHOLD",&inththreshold, "INTHTHRESHOLD/F");
  val_tree[treenr]->Branch("INTLTHRESHOLD",&intlthreshold, "INTLTHRESHOLD/F");
  val_tree[treenr]->Branch("INTCTHRESHOLD",&intcthreshold, "INTCTHRESHOLD/F");

  edm::ESHandle<SiStripThreshold> SiStripThreshold_;
  iSetup.get<SiStripThresholdRcd>().get(SiStripThreshold_);
  
  std::vector<uint32_t> detid;
  SiStripThreshold_->getDetIds(detid);
    for (size_t id=0;id<detid.size();id++)
      {
	SiStripThreshold::Range range=SiStripThreshold_->getRange(detid[id]);
	trackerthreshold.tree_detid=detid[id];
	for(int it=0;it<768;it++){
	  uint16_t estrip=(it & sistrip::FirstThStripMask_)<<sistrip::FirstThStripShift_ | (63 & sistrip::HighThStripMask_);
	  SiStripThreshold::ContainerIterator p = std::upper_bound(range.first,range.second,estrip,SiStripThreshold::dataStrictWeakOrdering());
	  if (p!=range.first){
	     SiStripThreshold::Data data=(*(--p));
	     intlthreshold+=data.getLth();
	     inththreshold+=data.getHth();
	     intcthreshold+=data.getClusth();
	  }
	}
	trackerthreshold.intlthreshold=intlthreshold;
	trackerthreshold.inththreshold=inththreshold;
	trackerthreshold.intcthreshold=intcthreshold;
	v_trackerthreshold.push_back(trackerthreshold);
	intlthreshold=0.;
	inththreshold=0.;
	intcthreshold=0.;
      } 
    
                                                                
std::sort(v_trackerthreshold.begin(), v_trackerthreshold.end(), trackerthreshold);
for(std::vector<threshold>::iterator iter=v_trackerthreshold.begin(); iter!=v_trackerthreshold.end(); ++iter){
     tree_detid=iter->tree_detid;
     intlthreshold=iter->intlthreshold;
     inththreshold=iter->inththreshold;
     intcthreshold=iter->intcthreshold;
     val_tree[treenr]->Fill();
   }

}
//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
SiStripO2OValidationRootFile::beginJob()
{

}

// ------------ method called to for each event  ------------
void
SiStripO2OValidationRootFile::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  if( cfg_FEDCabling ){ FEDCablingValidation(iSetup, nroftrees);  treenrfed=nroftrees; ++nroftrees;} 
  if( cfg_Pedestal )  { PedestalValidation(iSetup, nroftrees);    treenrped=nroftrees; ++nroftrees;}
  if( cfg_Noise )     { NoiseValidation(iSetup, nroftrees);       treenrnoi=nroftrees; ++nroftrees;}
  if( cfg_Quality )   { QualityValidation(iSetup, nroftrees);     treenrqua=nroftrees; ++nroftrees;}
  if( cfg_Threshold ) { ThresholdValidation(iSetup, nroftrees);   treenrthr=nroftrees; ++nroftrees;}
  if( cfg_APVLatency ){ APVLatencyValidation(iSetup, nroftrees);  treenrlat=nroftrees; ++nroftrees;}
  if( cfg_APVTiming ) { APVTimingValidation(iSetup, nroftrees);   treenrtim=nroftrees; ++nroftrees;}
}



// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripO2OValidationRootFile::endJob() {

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripO2OValidationRootFile);
