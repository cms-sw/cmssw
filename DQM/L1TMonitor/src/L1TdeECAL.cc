#include "DQM/L1TMonitor/interface/L1TdeECAL.h"
#include <bitset>

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
//#include "DQM/EcalCommon/interface/Numbers.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace dedefs;

L1TdeECAL::L1TdeECAL(const edm::ParameterSet& iConfig) {

  verbose_ = iConfig.getUntrackedParameter<int>("VerboseFlag",0);

  if(verbose())
    std::cout << "L1TdeECAL::L1TdeECAL()...\n" << std::flush;
  
  DEsource_ = iConfig.getParameter<edm::InputTag>("DataEmulCompareSource");
  histFolder_ = iConfig.getUntrackedParameter<std::string>("HistFolder", "L1TEMU/xpert/Ecal/");
  
  dbe = NULL;
  if (iConfig.getUntrackedParameter<bool>("DQMStore", false)) { 
    dbe = edm::Service<DQMStore>().operator->();
    dbe->setVerbose(0);
  }
  
  histFile_ = iConfig.getUntrackedParameter<std::string>("HistFile", "");
  if(iConfig.getUntrackedParameter<bool> ("disableROOToutput", true))
    histFile_ = "";

  if(dbe!=NULL)
    dbe->setCurrentFolder(histFolder_);
  
  if(verbose())
    std::cout << "L1TdeECAL::L1TdeECAL()...done.\n" << std::flush;
}

L1TdeECAL::~L1TdeECAL() {}

void 
L1TdeECAL::beginJob(const edm::EventSetup&) {

  if(verbose())
    std::cout << "L1TdeECAL::beginJob()  start\n";

  DQMStore* dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();
  if(dbe) {
    dbe->setCurrentFolder(histFolder_);
    dbe->rmdir(histFolder_);
  }

  if(dbe) {
    dbe->setCurrentFolder(histFolder_);

    etmapData.reserve(nSM);
    etmapEmul.reserve(nSM);
    etmapDiff.reserve(nSM);
    etmapData.resize( nSM, static_cast<MonitorElement*>(0) );
    etmapEmul.resize( nSM, static_cast<MonitorElement*>(0) );
    etmapDiff.resize( nSM, static_cast<MonitorElement*>(0) );
    
    
    std::string lbl("");
    char tmp[100];
    for(int j=0; j<nSM; j++) {
      lbl.clear();
      sprintf(tmp, "etmapDataSM%d", j+1);
      lbl+=tmp;
      etmapData[j] = dbe->book3D(lbl.c_str(),lbl.c_str(),
				 nTTEta, 0, nTTEta,
				 nTTPhi, 0, nTTPhi,
				 256, 0, 256.);
      sprintf(tmp, "etmapEmulSM%d", j+1);
      lbl.clear(); lbl+=tmp;
      etmapEmul[j] = dbe->book3D(lbl.c_str(),lbl.c_str(),
				 nTTEta, 0, nTTEta,
				 nTTPhi, 0, nTTPhi,
				 256, 0, 256.);
      sprintf(tmp, "etmapDiffSM%d", j+1);
      lbl.clear(); lbl+=tmp;
      etmapDiff[j] = dbe->book3D(lbl.c_str(),lbl.c_str(),
				 nTTEta, 0, nTTEta,
				 nTTPhi, 0, nTTPhi,
				 256, 0, 256.);
    }   
    lbl= "EcalEtMapDiff" ;
    EcalEtMapDiff = dbe->bookProfile2D(lbl.c_str(),lbl.c_str(),
				       35, -17.5, 17.5,
				       72, -10., 350.,
				       256, 0, 256.);
    lbl= "EcalFGMapDiff" ;
    EcalFGMapDiff = dbe->bookProfile2D(lbl.c_str(),lbl.c_str(),
				       35, -17.5, 17.5,
				       72, -10., 350.,
				       2, 0, 2.);
  }
  
  if(verbose())
    std::cout << "L1TdeECAL::beginJob()  end.\n" << std::flush;
}

void 
L1TdeECAL::endJob() {
  if(verbose())
    std::cout << "L1TdeECAL::endJob()...\n" << std::flush;
  if(histFile_.size()!=0  && dbe) 
    dbe->save(histFile_);
  //tbd delete emap;
}

 
// ------------ method called to for each event  ------------
void
  L1TdeECAL::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
  if(verbose())
    std::cout << "L1TdeECAL::analyze()  start\n" << std::flush;

  /// get the comparison results
  edm::Handle<L1DataEmulRecord> deRecord;
  iEvent.getByLabel(DEsource_, deRecord);
  
  bool isComp = deRecord->get_isComp(ETP);
  if(!isComp) {
    if(verbose()) 
      std::cout << "[L1TdeECAL] Ecal information not generated in de-record."
		<< " Skiping event!\n";
    return;
  }

  int DEncand[2];
  for(int j=0; j<2; j++) 
    DEncand[j] = deRecord->getNCand(ETP,j);
  
  if(verbose()) 
    std::cout << "[L1TdeECAL] ncands" 
	      << " data: " << DEncand[0]
	      << " emul: " << DEncand[1]
	      << std::endl;

  
  /// get the de candidates
  L1DEDigiCollection deColl;
  deColl = deRecord->getColl();

  if(verbose()) {
    std::cout << "[L1TdeECAL] digis: \n";
    for(L1DEDigiCollection::const_iterator it=deColl.begin(); it!=deColl.end(); it++)
      if(it->sid()==ETP)
	std::cout << "\t" << *it << std::endl;
  }
  

  /// --- Fill histograms(me) ---

  EcalElectronicsMapping emap;

  // d|e candidate loop
  for(L1DEDigiCollection::const_iterator it=deColl.begin(); 
      it!=deColl.end(); it++) {
    
    int    sid = it->sid();
    int    cid = it->cid();
    
    if(sid!=ETP)
      continue;

    if(it->empty())
      continue;

    assert(cid==ECALtp);
    
    if(verbose()) 
      std::cout << "[L1TdeECAL] processing digi: \n\t"
		<< *it << "\n" << std::flush;
    
    //get (global) tt coordinates
    int iphi = (int)it->x1();
    int ieta = (int)it->x2();
    
    //get sm id
    int ism = iEtaiPhiToSMid(ieta,iphi); //1..36

    //get local indices
    int zside = (ieta>0?1:-1);
    int iet = abs(ieta);
    int ipt = iphi;
    ipt = ipt + 2;
    if ( ipt > 72 ) ipt = ipt - 72;
    ipt = (ipt-1)%4 + 1;
    if ( zside > 0 ) ipt = 5 - ipt;


    /// Alternatively:
    EcalTrigTowerDetId idt(zside, EcalBarrel, abs(ieta), iphi);
    // ... from EcalElectronicsMapping (needs addt'l lib)
    //int itt = map.iTT  (idt); //1..68
    int itcc = emap.TCCid(idt); //37..54(eb+) 55..72(eb-)
    // need here converter tcc->sm id
    int smid = TCCidToSMid(itcc);
    // ... or as in EBTriggerTowerTask (needs addt'l lib)
    //int ismt = Numbers::iSM( idt );

    if(verbose())
      std::cout << "L1TdeECAL \t"
		<< " smid:" << smid 
		<< " ism:"  << ism 
		<< " itcc:" << itcc 
		<< " local phi:" << ipt << " eta:" << iet 
		<< "\n" << std::flush
		<< *it
		<< "\n" << std::flush;
    if(ism!=smid)
      LogDebug("L1TdeECAL") << "consistency check failure\n\t"
			    << " smid:" << smid 
			    << " ism:"  << ism 
			    << " itcc:" << itcc 
			    << std::endl;
    
    float xiet = iet+0.5;
    float xipt = ipt+0.5;
    
    //get energy values
    float rankarr[2]; 
    it->rank(rankarr);
    // get FG values
    unsigned int raw[2] ;
    it->data(raw) ;
    int FG[2] = { (raw[0] & 0x1000000)!=0, (raw[1] & 0x1000000)!=0 } ;

    int type = it->type(); //3 data only, 4 emul only
    if(type!=4 && etmapData[ism-1])
      etmapData[ism-1]->Fill(xiet-1, xipt-1, rankarr[0]);
    if(type!=3 && etmapEmul[ism-1])
      etmapEmul[ism-1]->Fill(xiet-1, xipt-1, rankarr[1]);
    if(type<2 && etmapDiff[ism-1]) {
      float diff = fabs(rankarr[0]-rankarr[1]);
      etmapDiff[ism-1]->Fill(xiet-1, xipt-1, diff);
      float phi = iphi ;
      if (phi>70) phi -= 73 ;
      phi *= 5 ;
      if (phi>0) phi -= 5 ;
      EcalEtMapDiff->Fill(ieta, phi, diff) ;
      diff = fabs(FG[0]-FG[1]);
      EcalFGMapDiff->Fill(ieta, phi, diff) ;
    }
  }//close loop over dedigi-cands


  if(verbose())
    std::cout << "L1TdeECAL::analyze() end.\n" << std::flush;

}

// in  :: iphi    1..72 [71,72,1..70]
//        ieta  -17..-1 (z-)   1..17 (z+)
// out :: SMid   19..36 (z-)   1..18 (z+)
int L1TdeECAL::iEtaiPhiToSMid(int ieta, int iphi) {
  // barrel
  int iz = (ieta<0)?-1:1;
  iphi += 2;
  if (iphi > 72) iphi -= 72;
  const int kEBTowersInPhi = 4; //EBDetId::kTowersInPhi
  int sm = ( iphi - 1 ) / kEBTowersInPhi;
  if ( iz < 0 ) 
    sm += 19;
  else
    sm += 1;
  return sm;
}

// in  :: TCCid  37..54 (z-) 55..72 (z+)
// out :: SMid   19..36 (z-)  1..18 (z+)
int L1TdeECAL::TCCidToSMid(int tccid) {
  // barrel
  if      ( tccid>37-1 && tccid<54+1) return tccid-37+19;
  else if ( tccid>55-1 && tccid<72+1) return tccid-55+ 1;
  else return 999;
}
