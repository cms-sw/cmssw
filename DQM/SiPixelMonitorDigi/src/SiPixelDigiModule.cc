#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
/// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <cstdio>

// Data Formats
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//
// Constructors
//
SiPixelDigiModule::SiPixelDigiModule() : id_(0),
					 ncols_(416),
					 nrows_(160) 
{
}
///
SiPixelDigiModule::SiPixelDigiModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 
}
///
SiPixelDigiModule::SiPixelDigiModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 
}
//
// Destructor
//
SiPixelDigiModule::~SiPixelDigiModule() {}
//
// Book histograms
//
void SiPixelDigiModule::book(const edm::ParameterSet& iConfig, const edm::EventSetup& iSetup, DQMStore::IBooker & iBooker, int type, bool twoD, bool hiRes, bool reducedSet, bool additInfo, bool isUpgrade) {

  //isUpgrade = iConfig.getUntrackedParameter<bool>("isUpgrade");
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *pTT = tTopoHandle.product();
    
  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  if(barrel){
    isHalfModule = PixelBarrelName(DetId(id_),pTT,isUpgrade).isHalfModule();
  }

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  
  int nbinx=ncols_/2, nbiny=nrows_/2;
  std::string twodtitle           = "Number of Digis (1bin=four pixels)"; 
  std::string pxtitle             = "Number of Digis (1bin=two columns)";
  std::string pytitle             = "Number of Digis (1bin=two rows)";
  std::string twodroctitle        = "ROC Occupancy (1bin=one ROC)";
  std::string twodzeroOccroctitle = "Zero Occupancy ROC Map (1bin=one ROC) for ";
  if(hiRes){
    nbinx = ncols_;
    nbiny = nrows_;
    twodtitle    = "Number of Digis (1bin=one pixel)";
    pxtitle = "Number of Digis (1bin=one column)";
    pytitle = "Number of Digis (1bin=one row)";
  }
  if(type==0){
    SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
    // Number of digis
    hid = theHistogramId->setHistoId("ndigis",id_);
    meNDigis_ = iBooker.book1D(hid,"Number of Digis",25,0.,25.);
    meNDigis_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    hid = theHistogramId->setHistoId("adc",id_);
    meADC_ = iBooker.book1D(hid,"Digi charge",128,0.,256.);
    meADC_->setAxisTitle("ADC counts",1);
	if(!reducedSet)
	{
    if(twoD){
      if(additInfo){
	// 2D hit map
	hid = theHistogramId->setHistoId("hitmap",id_);
	mePixDigis_ = iBooker.book2D(hid,twodtitle,nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
	mePixDigis_->setAxisTitle("Columns",1);
	mePixDigis_->setAxisTitle("Rows",2);
	//std::cout << "During booking: type is "<< type << ", ID is "<< id_ << ", pwd for booking is " << theDMBE->pwd() << ", Plot name: " << hid << std::endl;
      }
    }
    else{
      // projections of 2D hit map
      hid = theHistogramId->setHistoId("hitmap",id_);
      mePixDigis_px_ = iBooker.book1D(hid+"_px",pxtitle,nbinx,0.,float(ncols_));
      mePixDigis_py_ = iBooker.book1D(hid+"_py",pytitle,nbiny,0.,float(nrows_));
      mePixDigis_px_->setAxisTitle("Columns",1);
      mePixDigis_py_->setAxisTitle("Rows",1);
    }
	}
    delete theHistogramId;

  }
  
  if(type==1 && barrel){
    uint32_t DBladder;
    DBladder = PixelBarrelName(DetId(id_),pTT,isUpgrade).ladderName();
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    hid = src.label() + "_" + sladder;
    if(isHalfModule) hid += "H";
    else hid += "F";
    // Number of digis
    meNDigisLad_ = iBooker.book1D("ndigis_"+hid,"Number of Digis",25,0.,25.);
    meNDigisLad_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCLad_ = iBooker.book1D("adc_" + hid,"Digi charge",128,0.,256.);
    meADCLad_->setAxisTitle("ADC counts",1);
	if(!reducedSet)
	{
    if(twoD){
      // 2D hit map
      mePixDigisLad_ = iBooker.book2D("hitmap_"+hid,twodtitle,nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
      mePixDigisLad_->setAxisTitle("Columns",1);
      mePixDigisLad_->setAxisTitle("Rows",2);
      //std::cout << "During booking: type is "<< type << ", ID is "<< id_ << ", pwd for booking is " << theDMBE->pwd() << ", Plot name: " << hid << std::endl;
    }
    else{
      // projections of 2D hit map
      mePixDigisLad_px_ = iBooker.book1D("hitmap_"+hid+"_px",pxtitle,nbinx,0.,float(ncols_));
      mePixDigisLad_py_ = iBooker.book1D("hitmap_"+hid+"_py",pytitle,nbiny,0.,float(nrows_));
      mePixDigisLad_px_->setAxisTitle("Columns",1);
      mePixDigisLad_py_->setAxisTitle("Rows",1);	
    }
	}
  }
  if(type==2 && barrel){
    uint32_t DBlayer;
    DBlayer = PixelBarrelName(DetId(id_),pTT,isUpgrade).layerName();
    char slayer[80]; sprintf(slayer,"Layer_%i",DBlayer);
    hid = src.label() + "_" + slayer;
    if(!additInfo){
      // Number of digis
      meNDigisLay_ = iBooker.book1D("ndigis_"+hid,"Number of Digis",25,0.,25.);
      meNDigisLay_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
      meADCLay_ = iBooker.book1D("adc_" + hid,"Digi charge",128,0.,256.);
      meADCLay_->setAxisTitle("ADC counts",1);
    }
    if(!reducedSet){
      if(twoD || additInfo){
	// 2D hit map
	if(isHalfModule){
	  mePixDigisLay_ = iBooker.book2D("hitmap_"+hid,twodtitle,nbinx,0.,float(ncols_),2*nbiny,0.,float(2*nrows_));
	}
	else{
	  mePixDigisLay_ = iBooker.book2D("hitmap_"+hid,twodtitle,nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));

	}
	mePixDigisLay_->setAxisTitle("Columns",1);
	mePixDigisLay_->setAxisTitle("Rows",2);
	
	//std::cout << "During booking: type is "<< type << ", ID is "<< id_ << ", pwd for booking is " << theDMBE->pwd() << ", Plot name: " << hid << std::endl;
	int yROCbins[3] = {18,30,42};
	mePixRocsLay_ = iBooker.book2D("rocmap_"+hid,twodroctitle,32,0.,32.,yROCbins[DBlayer-1],1.5,1.5+float(yROCbins[DBlayer-1]/2));
	mePixRocsLay_->setAxisTitle("ROCs per Module",1);
	mePixRocsLay_->setAxisTitle("ROCs per 1/2 Ladder",2);
	meZeroOccRocsLay_ = iBooker.book2D("zeroOccROC_map",twodzeroOccroctitle+hid,32,0.,32.,yROCbins[DBlayer-1],1.5,1.5+float(yROCbins[DBlayer-1]/2));
	meZeroOccRocsLay_->setAxisTitle("ROCs per Module",1);
	meZeroOccRocsLay_->setAxisTitle("ROCs per 1/2 Ladder",2);
      }
      if(!twoD && !additInfo){
	// projections of 2D hit map
	mePixDigisLay_px_ = iBooker.book1D("hitmap_"+hid+"_px",pxtitle,nbinx,0.,float(ncols_));
	if(isHalfModule){
	  mePixDigisLay_py_ = iBooker.book1D("hitmap_"+hid+"_py",pytitle,2*nbiny,0.,float(2*nrows_));
	}
	else{
	  mePixDigisLay_py_ = iBooker.book1D("hitmap_"+hid+"_py",pytitle,nbiny,0.,float(nrows_));
	}
	mePixDigisLay_px_->setAxisTitle("Columns",1);
	mePixDigisLay_py_->setAxisTitle("Rows",1);
      }
    }
  }
  if(type==3 && barrel){
    uint32_t DBmodule;
    DBmodule = PixelBarrelName(DetId(id_),pTT,isUpgrade).moduleName();
    char smodule[80]; sprintf(smodule,"Ring_%i",DBmodule);
    hid = src.label() + "_" + smodule;
    // Number of digis
    meNDigisPhi_ = iBooker.book1D("ndigis_"+hid,"Number of Digis",25,0.,25.);
    meNDigisPhi_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCPhi_ = iBooker.book1D("adc_" + hid,"Digi charge",128,0.,256.);
    meADCPhi_->setAxisTitle("ADC counts",1);
    if(!reducedSet)
      {
	if(twoD){
	  
	  // 2D hit map
	  if(isHalfModule){
	    mePixDigisPhi_ = iBooker.book2D("hitmap_"+hid,twodtitle,nbinx,0.,float(ncols_),2*nbiny,0.,float(2*nrows_));
	  }
	  else {
	    mePixDigisPhi_ = iBooker.book2D("hitmap_"+hid,twodtitle,nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
	  }
	  mePixDigisPhi_->setAxisTitle("Columns",1);
	  mePixDigisPhi_->setAxisTitle("Rows",2);
	  //std::cout << "During booking: type is "<< type << ", ID is "<< id_ << ", pwd for booking is " << theDMBE->pwd() << ", Plot name: " << hid << std::endl;
	}
	else{
	  // projections of 2D hit map
	  mePixDigisPhi_px_ = iBooker.book1D("hitmap_"+hid+"_px",pxtitle,nbinx,0.,float(ncols_));
	  if(isHalfModule){
	    mePixDigisPhi_py_ = iBooker.book1D("hitmap_"+hid+"_py",pytitle,2*nbiny,0.,float(2*nrows_));
	  }
	  else{
	    mePixDigisPhi_py_ = iBooker.book1D("hitmap_"+hid+"_py",pytitle,nbiny,0.,float(nrows_));
	  }
	  mePixDigisPhi_px_->setAxisTitle("Columns",1);
	  mePixDigisPhi_py_->setAxisTitle("Rows",1);
	}
      }
  }
  if(type==4 && endcap){
    uint32_t blade;
    blade= PixelEndcapName(DetId(id_),pTT,isUpgrade).bladeName();
    
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    hid = src.label() + "_" + sblade;
    // Number of digis
    meNDigisBlade_ = iBooker.book1D("ndigis_"+hid,"Number of Digis",25,0.,25.);
    meNDigisBlade_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCBlade_ = iBooker.book1D("adc_" + hid,"Digi charge",128,0.,256.);
    meADCBlade_->setAxisTitle("ADC counts",1);
  }
  if(type==5 && endcap){
    uint32_t disk;
    disk = PixelEndcapName(DetId(id_),pTT,isUpgrade).diskName();
    
    char sdisk[80]; sprintf(sdisk, "Disk_%i",disk);
    hid = src.label() + "_" + sdisk;
    if(!additInfo){
      // Number of digis
      meNDigisDisk_ = iBooker.book1D("ndigis_"+hid,"Number of Digis",25,0.,25.);
      meNDigisDisk_->setAxisTitle("Number of digis",1);
      // Charge in ADC counts
      meADCDisk_ = iBooker.book1D("adc_" + hid,"Digi charge",128,0.,256.);
      meADCDisk_->setAxisTitle("ADC counts",1);
    }
    if(additInfo){
      mePixDigisDisk_ = iBooker.book2D("hitmap_"+hid,twodtitle,260,0.,260.,160,0.,160.);
      mePixDigisDisk_->setAxisTitle("Columns",1);
      mePixDigisDisk_->setAxisTitle("Rows",2);
      //ROC information in disks
      mePixRocsDisk_  = iBooker.book2D("rocmap_"+hid,twodroctitle,26,0.,26.,24,1.,13.);
      mePixRocsDisk_ ->setAxisTitle("ROCs per Module (2 Panels)",1);
      mePixRocsDisk_ ->setAxisTitle("Blade Number",2);
      meZeroOccRocsDisk_  = iBooker.book2D("zeroOccROC_map",twodzeroOccroctitle+hid,26,0.,26.,24,1.,13.);
      meZeroOccRocsDisk_ ->setAxisTitle("Zero-Occupancy ROCs per Module (2 Panels)",1);
      meZeroOccRocsDisk_ ->setAxisTitle("Blade Number",2);
    }
  }
  if(type==6 && endcap){
    uint32_t panel;
    uint32_t module;
    panel= PixelEndcapName(DetId(id_),pTT,isUpgrade).pannelName();
    module= PixelEndcapName(DetId(id_),pTT,isUpgrade).plaquetteName();
    
    char slab[80]; sprintf(slab, "Panel_%i_Ring_%i",panel, module);
    hid = src.label() + "_" + slab;
    // Number of digis
    meNDigisRing_ = iBooker.book1D("ndigis_"+hid,"Number of Digis",25,0.,25.);
    meNDigisRing_->setAxisTitle("Number of digis",1);
    // Charge in ADC counts
    meADCRing_ = iBooker.book1D("adc_" + hid,"Digi charge",128,0.,256.);
    meADCRing_->setAxisTitle("ADC counts",1);
	if(!reducedSet)
	{
    if(twoD){
      // 2D hit map
      mePixDigisRing_ = iBooker.book2D("hitmap_"+hid,twodtitle,nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
      mePixDigisRing_->setAxisTitle("Columns",1);
      mePixDigisRing_->setAxisTitle("Rows",2);
      //std::cout << "During booking: type is "<< type << ", ID is "<< id_ << ", pwd for booking is " << theDMBE->pwd() << ", Plot name: " << hid << std::endl;
    }
    else{
      // projections of 2D hit map
      mePixDigisRing_px_ = iBooker.book1D("hitmap_"+hid+"_px",pxtitle,nbinx,0.,float(ncols_));
      mePixDigisRing_py_ = iBooker.book1D("hitmap_"+hid+"_py",pytitle,nbiny,0.,float(nrows_));
      mePixDigisRing_px_->setAxisTitle("Columns",1);
      mePixDigisRing_py_->setAxisTitle("Rows",1);
    }
	}
  }
}


//
// Fill histograms
//
int SiPixelDigiModule::fill(const edm::DetSetVector<PixelDigi>& input, const edm::EventSetup& iSetup,
             MonitorElement* combBarrel, MonitorElement* chanBarrel, std::vector<MonitorElement*>& chanBarrelL, MonitorElement* combEndcap,
			    bool modon, bool ladon, bool layon, bool phion, 
			    bool bladeon, bool diskon, bool ringon, 
			    bool twoD, bool reducedSet, bool twoDimModOn, bool twoDimOnlyLayDisk,
			    int &nDigisA, int &nDigisB, bool isUpgrade) {
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *pTT = tTopoHandle.product();

  bool barrel = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
  bool endcap = DetId(id_).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
  bool isHalfModule = false;
  uint32_t DBladder = 0;
  if(barrel){
    isHalfModule = PixelBarrelName(DetId(id_),pTT,isUpgrade).isHalfModule();
    DBladder = PixelBarrelName(DetId(id_),pTT,isUpgrade).ladderName();
  }

  edm::DetSetVector<PixelDigi>::const_iterator isearch = input.find(id_); // search  digis of detid
  
  unsigned int numberOfDigisMod = 0;
  int msize;
  if (isUpgrade) {msize=10;} else {msize=8;}
  int numberOfDigis[msize]; for(int i=0; i!=msize; i++) numberOfDigis[i]=0; 
  nDigisA=0; nDigisB=0;
  if( isearch != input.end() ) {  // Not an empty iterator
    
    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      int adc = di->adc();    // charge
      int col = di->column(); // column 
      int row = di->row();    // row
      numberOfDigisMod++;
      
      int DBlayer = 0;
      int DBmodule =0;
      
      if (!isUpgrade) {
      PixelBarrelName::Shell DBshell = PixelBarrelName(DetId(id_),pTT,isUpgrade).shell();
        DBlayer  = PixelBarrelName(DetId(id_),pTT,isUpgrade).layerName();
        DBmodule = PixelBarrelName(DetId(id_),pTT,isUpgrade).moduleName();
      if(barrel){
        if(isHalfModule){
          if(DBshell==PixelBarrelName::pI||DBshell==PixelBarrelName::pO){
	    numberOfDigis[0]++; nDigisA++;
	    if(DBlayer==1) numberOfDigis[2]++;
	    if(DBlayer==2) numberOfDigis[3]++;
	    if(DBlayer==3) numberOfDigis[4]++;
	  }
          if(DBshell==PixelBarrelName::mI||DBshell==PixelBarrelName::mO){
	    numberOfDigis[1]++; nDigisB++;
	    if(DBlayer==1) numberOfDigis[5]++;
	    if(DBlayer==2) numberOfDigis[6]++;
	    if(DBlayer==3) numberOfDigis[7]++;
	  }
        }else{
          if(row<80){
	    numberOfDigis[0]++; nDigisA++;
	    if(DBlayer==1) numberOfDigis[2]++;
	    if(DBlayer==2) numberOfDigis[3]++;
	    if(DBlayer==3) numberOfDigis[4]++;
	  }else{ 
	    numberOfDigis[1]++; nDigisB++;
	    if(DBlayer==1) numberOfDigis[5]++;
	    if(DBlayer==2) numberOfDigis[6]++;
	    if(DBlayer==3) numberOfDigis[7]++;
	  }
        }
      }
      } else if (isUpgrade) {
        DBlayer  = PixelBarrelName(DetId(id_),pTT,isUpgrade).layerName();
        DBmodule = PixelBarrelName(DetId(id_),pTT,isUpgrade).moduleName();
	if(barrel){
	  if(row<80){
	    numberOfDigis[0]++; nDigisA++;
	    if(DBlayer==1) numberOfDigis[2]++;
	    if(DBlayer==2) numberOfDigis[3]++;
	    if(DBlayer==3) numberOfDigis[4]++;
	    if(DBlayer==4) numberOfDigis[5]++;
	  }else{ 
	    numberOfDigis[1]++; nDigisB++;
	    if(DBlayer==1) numberOfDigis[6]++;
	    if(DBlayer==2) numberOfDigis[7]++;
	    if(DBlayer==3) numberOfDigis[8]++;
	    if(DBlayer==4) numberOfDigis[9]++;
	  }
	}
      }
      
      if(modon){
	if(!reducedSet){
	  if(twoD) {
	    if(twoDimModOn) (mePixDigis_)->Fill((float)col,(float)row);
	  }
	  else {
	    (mePixDigis_px_)->Fill((float)col);
	    (mePixDigis_py_)->Fill((float)row);
	  }
	}
	(meADC_)->Fill((float)adc);
      }
      if(ladon && barrel){
	(meADCLad_)->Fill((float)adc);
	if(!reducedSet){
	  if(twoD) (mePixDigisLad_)->Fill((float)col,(float)row);
	  else {
	  (mePixDigisLad_px_)->Fill((float)col);
	  (mePixDigisLad_py_)->Fill((float)row);
	  }
	}
      }
      if((layon || twoDimOnlyLayDisk) && barrel){
	if(!twoDimOnlyLayDisk) (meADCLay_)->Fill((float)adc);
	if(!reducedSet){
	  if((layon && twoD) || twoDimOnlyLayDisk){
	    //ROC histos...
	    float rocx = (float)col/52. + 8.0*float(DBmodule-1);
	    float rocy = (float)row/160.+float(DBladder);
	    //Shift 1st ladder (half modules) up by 1 bin
	    if(DBladder==1) rocy = rocy + 0.5;
	    mePixRocsLay_->Fill(rocx,rocy);

	    if(isHalfModule && DBladder==1){
	      (mePixDigisLay_)->Fill((float)col,(float)row+80);
	    }
	    else (mePixDigisLay_)->Fill((float)col,(float)row);
	  }
	  if((layon && !twoD) && !twoDimOnlyLayDisk){
	    (mePixDigisLay_px_)->Fill((float)col);
	    if(isHalfModule && DBladder==1) {
	      (mePixDigisLay_py_)->Fill((float)row+80);
	    }
	    else (mePixDigisLay_py_)->Fill((float)row);
	  }
	}
      }
      if(phion && barrel){
	(meADCPhi_)->Fill((float)adc);
	if(!reducedSet)
	{
	if(twoD){
	  if(isHalfModule && DBladder==1){
	    (mePixDigisPhi_)->Fill((float)col,(float)row+80);
	  }
	  else (mePixDigisPhi_)->Fill((float)col,(float)row);
	}
	else {
	  (mePixDigisPhi_px_)->Fill((float)col);
	  if(isHalfModule && DBladder==1) {
	    (mePixDigisPhi_py_)->Fill((float)row+80);
	  }
	  else (mePixDigisPhi_py_)->Fill((float)row);
	}
	}
      }
      if(bladeon && endcap){
	(meADCBlade_)->Fill((float)adc);
      }

      if((diskon || twoDimOnlyLayDisk) && endcap){
	if(!twoDimOnlyLayDisk) (meADCDisk_)->Fill((float)adc);
	if(twoDimOnlyLayDisk){
	  (mePixDigisDisk_)->Fill((float)col,(float)row);
	  //ROC monitoring
	  int DBpanel;
	  int DBblade;
     DBpanel= PixelEndcapName(DetId(id_),pTT,isUpgrade).pannelName();
     DBblade= PixelEndcapName(DetId(id_),pTT,isUpgrade).bladeName();
	  float offx = 0.;
	  //This crazy offset takes into account the roc and module fpix configuration
	  for (int i = DBpanel; i < DBmodule; ++i) {offx = offx + float(5+DBpanel-i);}
	  float rocx = (float)col/52. + offx + 14.0*float(DBpanel-1);
	  float rocy = (float)row/160.+float(DBblade);
	  mePixRocsDisk_->Fill(rocx,rocy);
	}
      }
      if(ringon && endcap){
	(meADCRing_)->Fill((float)adc);
	if(!reducedSet)
	{
	if(twoD) (mePixDigisRing_)->Fill((float)col,(float)row);
	else {
	  (mePixDigisRing_px_)->Fill((float)col);
	  (mePixDigisRing_py_)->Fill((float)row);
	}
	}
      }
    }
    if(modon) (meNDigis_)->Fill((float)numberOfDigisMod);
    if(ladon && barrel) (meNDigisLad_)->Fill((float)numberOfDigisMod);
    if(layon && barrel && !twoDimOnlyLayDisk) (meNDigisLay_)->Fill((float)numberOfDigisMod);
    if(phion && barrel) (meNDigisPhi_)->Fill((float)numberOfDigisMod);
    if(bladeon && endcap) (meNDigisBlade_)->Fill((float)numberOfDigisMod);
    if(diskon && endcap && !twoDimOnlyLayDisk) (meNDigisDisk_)->Fill((float)numberOfDigisMod);
    if(ringon && endcap) (meNDigisRing_)->Fill((float)numberOfDigisMod);
    if(barrel){ 
      if(combBarrel) combBarrel->Fill((float)numberOfDigisMod);
      if(chanBarrel){ if(numberOfDigis[0]>0) chanBarrel->Fill((float)numberOfDigis[0]); if(numberOfDigis[1]>0) chanBarrel->Fill((float)numberOfDigis[1]); }
      int j = 2;
      for (std::vector<MonitorElement*>::iterator i = chanBarrelL.begin(); i != chanBarrelL.end(); i++)
      {
         if(numberOfDigis[j]>0) (*i)->Fill((float)numberOfDigis[j]);
         j++;
      }
    }else if(endcap){
      if(combEndcap) combEndcap->Fill((float)numberOfDigisMod);
    }
  }
  
  //std::cout<<"numberOfDigis for this module: "<<numberOfDigis<<std::endl;
  return numberOfDigisMod;
}

// This was done in the Source file, but is moved to the Module for thread safety reasons. Using ME that is booked here.
void SiPixelDigiModule::resetRocMap(){
  if (mePixRocsDisk_) mePixRocsDisk_->Reset();
  if (mePixRocsLay_) mePixRocsLay_->Reset();
}

//Moved from source. Gets the zero and low eff ROCs from each module. Called in source for each module.
std::pair<int,int> SiPixelDigiModule::getZeroLoEffROCs(){
  int nZeroROC = 0;
  int nLoEffROC = 0;
  float SF = 1.0;
  if (mePixRocsDisk_ && meZeroOccRocsDisk_){
    if (mePixRocsDisk_->getEntries() > 0) SF = float(mePixRocsDisk_->getNbinsX()*mePixRocsDisk_->getNbinsY()/mePixRocsDisk_->getEntries());
    for (int i = 1; i < mePixRocsDisk_->getNbinsX(); ++i){
      for (int j = 1; j < mePixRocsDisk_->getNbinsY(); ++j){
	float localX = float(i) - 0.5;
	float localY = float(j)/2.0 + 0.75;
	if (mePixRocsDisk_->getBinContent(i,j)    <  1 ) {nZeroROC++; meZeroOccRocsDisk_->Fill(localX,localY);}
	if (mePixRocsDisk_->getBinContent(i,j)*SF < 0.25){nLoEffROC++;}
      }
    }
    return std::pair<int,int>(nZeroROC,nLoEffROC);
  }
  if (mePixRocsLay_ && meZeroOccRocsLay_){
    if (mePixRocsLay_->getEntries() > 0) SF = float(mePixRocsLay_->getNbinsX()*mePixRocsLay_->getNbinsY()/mePixRocsLay_->getEntries());
    for (int i = 1; i < mePixRocsLay_->getNbinsX(); ++i){
      for (int j = 1; j < mePixRocsLay_->getNbinsY(); ++j){
	float localX = float(i) - 0.5;
	float localY = float(j)/2.0 + 1.25;
	if (mePixRocsLay_->getBinContent(i,j)    <  1 ) {nZeroROC++; meZeroOccRocsLay_->Fill(localX,localY);}
	if (mePixRocsLay_->getBinContent(i,j)*SF < 0.25){nLoEffROC++;}
      }
    }
    return std::pair<int,int>(nZeroROC,nLoEffROC);
  }
  return std::pair<int,int>(0,0);
}
