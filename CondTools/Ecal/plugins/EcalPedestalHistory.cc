/**\class EcalPedestalHistory

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EcalEcalPedestalHistory.cc,v 0.0 2016/05/02 jean fay Exp $
//
//

#include "EcalPedestalHistory.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

//#include<fstream>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TProfile.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>

using namespace edm;
using namespace cms;
using namespace std;

//
// constants, enums and typedefs
//
//const Int_t kSample=10;
//
// static data member definitions
//
//int gainValues[3] = {12, 6, 1};

//
// constructors and destructor
//

//====================================================================
EcalPedestalHistory::EcalPedestalHistory(const edm::ParameterSet& iConfig) {
  //====================================================================
  //now do what ever initialization is needed
  //  EBDigiCollection_          = iConfig.getParameter<edm::InputTag>("EBDigiCollection");
  runnumber_ = iConfig.getUntrackedParameter<int>("runnumber", -1);
  ECALType_ = iConfig.getParameter<std::string>("ECALType");
  runType_ = iConfig.getParameter<std::string>("runType");
  startevent_ = iConfig.getUntrackedParameter<unsigned int>("startevent", 1);

  std::cout << "EcalPedestals Source handler constructor\n" << std::endl;
  m_firstRun = static_cast<unsigned int>(atoi(iConfig.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned int>(atoi(iConfig.getParameter<std::string>("lastRun").c_str()));
  m_sid = iConfig.getParameter<std::string>("OnlineDBSID");
  m_user = iConfig.getParameter<std::string>("OnlineDBUser");
  m_pass = iConfig.getParameter<std::string>("OnlineDBPassword");
  m_locationsource = iConfig.getParameter<std::string>("LocationSource");
  m_location = iConfig.getParameter<std::string>("Location");
  m_gentag = iConfig.getParameter<std::string>("GenTag");
  std::cout << m_sid << "/" << m_user << "/" << m_pass << "/" << m_location << "/" << m_gentag << std::endl;

  vector<int> listDefaults;
  listDefaults.push_back(-1);

  cnt_evt_ = 0;
  //  cout << "Exiting constructor" << endl;
}  //constructor

//========================================================================
EcalPedestalHistory::~EcalPedestalHistory() {
  //========================================================================
  cout << "ANALYSIS FINISHED" << endl;
}  //destructor

//========================================================================
void EcalPedestalHistory::beginRun(edm::Run const&, edm::EventSetup const& c) {
  ///========================================================================

  cout << "Entering beginRun" << endl;
  /*     do not use any tag...
  edm::ESHandle<EcalChannelStatus> pChannelStatus;
  c.get<EcalChannelStatusRcd>().get(pChannelStatus);
  const EcalChannelStatus* chStatus = pChannelStatus.product();  
  EcalChannelStatusMap::const_iterator chit;
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    EBDetId id = EBDetId::unhashIndex(iChannel);
    chit = chStatus->getMap().find(id.rawId());
    if( chit != chStatus->getMap().end() ) {
      EcalChannelStatusCode ch_code = (*chit);
      uint16_t statusCode = ch_code.getStatusCode() & 31;
      if(statusCode == 1 || (statusCode > 7 && statusCode < 12))
	maskedChannels_.push_back(iChannel);
    }
  }
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    EEDetId id = EEDetId::unhashIndex(iChannel);
    chit = chStatus->getMap().find(id.rawId());
    if( chit != chStatus->getMap().end() ) {
      EcalChannelStatusCode ch_code = (*chit);
      uint16_t statusCode = ch_code.getStatusCode() & 31;
      if(statusCode == 1 || (statusCode > 7 && statusCode < 12))
	maskedEEChannels_.push_back(iChannel);
    }
  }
  */
  TH1F** hMean = new TH1F*[15];
  TH1F** hRMS = new TH1F*[15];
  TFile f("PedHist.root", "RECREATE");

  typedef struct {
    int iChannel;
    int ix;
    int iy;
    int iz;
  } Chan_t;
  Chan_t Chan;
  Chan.iChannel = -1;
  Chan.ix = -1;
  Chan.iy = -1;
  Chan.iz = -1;

  TTree* tPedChan = new TTree("PedChan", "Channels");  // Output tree for channels
  tPedChan->Branch("Channels", &Chan.iChannel, "iChannel/I");
  tPedChan->Branch("x", &Chan.ix, "ix/I");
  tPedChan->Branch("y", &Chan.iy, "iy/I");
  tPedChan->Branch("z", &Chan.iz, "iz/I");
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    Chan.iChannel = iChannel;
    EBDetId myEBDetId = EBDetId::unhashIndex(iChannel);
    Chan.ix = myEBDetId.ieta();  // -85:-1,1:85
    Chan.iy = myEBDetId.iphi();  // 1:360
    Chan.iz = 0;
    if (iChannel % 10000 == 0)
      cout << " EB channel " << iChannel << " eta " << Chan.ix << " phi " << Chan.iy << endl;
    tPedChan->Fill();
  }
  for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
    Chan.iChannel = iChannel;
    EEDetId myEEDetId = EEDetId::unhashIndex(iChannel);
    Chan.ix = myEEDetId.ix();
    Chan.iy = myEEDetId.iy();
    Chan.iz = myEEDetId.zside();
    if (iChannel % 1000 == 0)
      cout << " EE channel " << iChannel << " x " << Chan.ix << " y " << Chan.iy << " z " << Chan.iz << endl;
    tPedChan->Fill();
  }
  tPedChan->Write();
  tPedChan->Print();

  typedef struct {
    int Run;
    double Mean[kChannels];
    double RMS[kChannels];
  } Ped_t;
  Ped_t PedVal;
  PedVal.Run = -1;  // initialization
  for (int iChannel = 0; iChannel < kChannels; iChannel++) {
    PedVal.Mean[iChannel] = -1.;
    PedVal.RMS[iChannel] = -1.;
  }
  TTree* tPedHist = new TTree("PedHist", "Pedestal History");  // Output tree for pedestal mean/rms
  tPedHist->Branch("Pedestals", &PedVal.Run, "Run/I");
  tPedHist->Branch("Mean", PedVal.Mean, "Mean[75848]/D");
  tPedHist->Branch("RMS", PedVal.RMS, "RMS[75848]/D");

  // here we retrieve all the runs after the last from online DB
  std::cout << "Retrieving run list from ONLINE DB ... " << std::endl;
  econn = new EcalCondDBInterface(m_sid, m_user, m_pass);
  std::cout << "Connection done" << std::endl;
  if (!econn) {
    std::cout << " Problem with OMDS: connection parameters " << m_sid << "/" << m_user << "/" << m_pass << std::endl;
    throw cms::Exception("OMDS not available");
  }

  // these are the online conditions DB classes
  RunList my_runlist;
  RunTag my_runtag;
  LocationDef my_locdef;
  RunTypeDef my_rundef;

  my_locdef.setLocation(m_location);
  my_rundef.setRunType("PEDESTAL");
  my_runtag.setLocationDef(my_locdef);
  my_runtag.setRunTypeDef(my_rundef);
  my_runtag.setGeneralTag(m_gentag);

  // here we retrieve the Monitoring run records
  MonVersionDef monverdef;
  monverdef.setMonitoringVersion("test01");
  MonRunTag mon_tag;
  //	mon_tag.setGeneralTag("CMSSW");
  mon_tag.setGeneralTag("CMSSW-offline-private");
  mon_tag.setMonVersionDef(monverdef);
  MonRunList mon_list;
  mon_list.setMonRunTag(mon_tag);
  mon_list.setRunTag(my_runtag);
  //    mon_list=econn->fetchMonRunList(my_runtag, mon_tag);
  unsigned int min_run = 0, max_since = 0;
  if (m_firstRun < max_since) {
    min_run = max_since + 1;  // we have to add 1 to the last transferred one
  } else {
    min_run = m_firstRun;
  }

  unsigned int max_run = m_lastRun;
  mon_list = econn->fetchMonRunList(my_runtag, mon_tag, min_run, max_run);

  std::vector<MonRunIOV> mon_run_vec = mon_list.getRuns();
  int mon_runs = mon_run_vec.size();
  std::cout << "number of Mon runs is : " << mon_runs << std::endl;

  if (mon_runs > 0) {
    int NbChan = 0;
    for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
      if (iChannel % 10000 == 1) {
        hMean[NbChan] = new TH1F(Form("Mean_%i", NbChan), Form("Mean EB %i", iChannel), mon_runs, 0., mon_runs);
        hRMS[NbChan] = new TH1F(Form("RMS_%i", NbChan), Form("RMS EB %i", iChannel), mon_runs, 0., mon_runs);
        NbChan++;
      }
    }
    for (int iChannel = 0; iChannel < kEEChannels; iChannel++) {
      if (iChannel % 2000 == 1) {
        hMean[NbChan] = new TH1F(Form("Mean_%i", NbChan), Form("Mean EE %i", iChannel), mon_runs, 0., mon_runs);
        hRMS[NbChan] = new TH1F(Form("RMS_%i", NbChan), Form("RMS EE %i", iChannel), mon_runs, 0., mon_runs);
        NbChan++;
      }
    }

    //    int krmax = std::min(mon_runs, 30);
    int krmax = mon_runs;
    for (int kr = 0; kr < krmax; kr++) {
      std::cout << "-kr------:  " << kr << std::endl;

      unsigned int irun = static_cast<unsigned int>(mon_run_vec[kr].getRunIOV().getRunNumber());
      std::cout << "retrieve the data for run number: " << irun << std::endl;
      if (mon_run_vec[kr].getSubRunNumber() <= 1) {
        // retrieve the data for a given run
        RunIOV runiov_prime = mon_run_vec[kr].getRunIOV();
        // retrieve the pedestals from OMDS for this run
        std::map<EcalLogicID, MonPedestalsDat> dataset_mon;
        econn->fetchDataSet(&dataset_mon, &mon_run_vec[kr]);
        std::cout << "OMDS record for run " << irun << " is made of " << dataset_mon.size() << std::endl;
        int nEB = 0, nEE = 0;
        typedef std::map<EcalLogicID, MonPedestalsDat>::const_iterator CImon;
        EcalLogicID ecid_xt;
        MonPedestalsDat rd_ped;

        // this to validate ...
        int nbad = 0;
        for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
          ecid_xt = p->first;
          rd_ped = p->second;
          int sm_num = ecid_xt.getID1();
          int xt_num = ecid_xt.getID2();
          int yt_num = ecid_xt.getID3();

          //checkPedestal
          bool result = true;
          if (rd_ped.getPedRMSG12() > 3 || rd_ped.getPedRMSG12() <= 0 || rd_ped.getPedRMSG6() > 2 ||
              rd_ped.getPedRMSG12() <= 0 || rd_ped.getPedRMSG1() > 1 || rd_ped.getPedRMSG1() <= 0 ||
              rd_ped.getPedMeanG12() > 300 || rd_ped.getPedMeanG12() <= 100 || rd_ped.getPedMeanG6() > 300 ||
              rd_ped.getPedMeanG6() <= 100 || rd_ped.getPedMeanG1() > 300 || rd_ped.getPedMeanG6() <= 100)
            result = false;

          // here we check and count how many bad channels we have
          if (!result) {
            nbad++;
            if (nbad < 10)
              std::cout << "BAD LIST: channel " << sm_num << "/" << xt_num << "/" << yt_num << "ped/rms "
                        << rd_ped.getPedMeanG12() << "/" << rd_ped.getPedRMSG12() << std::endl;
          }
          if (ecid_xt.getName() == "EB_crystal_number") {
            nEB++;
          } else {
            nEE++;
          }
        }  // end loop over pedestal data
        // ok or bad? A bad run is for more than 5% bad channels

        //	      if(nbad<(dataset_mon.size()*0.1)){
        if (nbad < (dataset_mon.size() * 0.05) &&
            (nEB > 10200 || nEE > 2460)) {  // this is good run, fill histo and tree
          PedVal.Run = irun;
          int NbChan = 0;
          for (CImon p = dataset_mon.begin(); p != dataset_mon.end(); p++) {
            ecid_xt = p->first;
            rd_ped = p->second;
            int sm_num = ecid_xt.getID1();
            int xt_num = ecid_xt.getID2();
            int yt_num = ecid_xt.getID3();

            if (ecid_xt.getName() == "EB_crystal_number") {  // Barrel
              EBDetId ebdetid(sm_num, xt_num, EBDetId::SMCRYSTALMODE);
              int iChannel = ebdetid.hashedIndex();
              if (iChannel < 0 || iChannel > 61200)
                cout << " SM " << sm_num << " Chan in SM " << xt_num << " IChannel " << iChannel << endl;
              if (iChannel % 10000 == 1) {
                hMean[NbChan]->Fill(kr, rd_ped.getPedMeanG12());
                hRMS[NbChan]->Fill(kr, rd_ped.getPedRMSG12());
                NbChan++;
              }
              PedVal.Mean[iChannel] = rd_ped.getPedMeanG12();
              PedVal.RMS[iChannel] = rd_ped.getPedRMSG12();
              if (iChannel % 10000 == 0)
                cout << " channel " << iChannel << " mean " << PedVal.Mean[iChannel] << " RMS " << PedVal.RMS[iChannel]
                     << endl;
            } else {  // Endcaps
              if (EEDetId::validDetId(xt_num, yt_num, sm_num)) {
                EEDetId eedetid(xt_num, yt_num, sm_num);
                int iChannel = eedetid.hashedIndex();
                if (iChannel < 0 || iChannel > 14648)
                  cout << " x " << sm_num << " y " << xt_num << " z " << yt_num << " IChannel " << iChannel << endl;
                if (iChannel % 2000 == 1) {
                  hMean[NbChan]->Fill(kr, rd_ped.getPedMeanG12());
                  hRMS[NbChan]->Fill(kr, rd_ped.getPedRMSG12());
                  NbChan++;
                }
                int iChanEE = kEBChannels + iChannel;
                //		cout << " channel EE " << iChanEE << endl;
                PedVal.Mean[iChanEE] = rd_ped.getPedMeanG12();
                PedVal.RMS[iChanEE] = rd_ped.getPedRMSG12();
              }  // valid ee Id
            }    // Endcaps
          }      // loop over channels
          tPedHist->Fill();
          cout << " We got a good run " << irun << endl;
        }  // good run
      }    // mon_run_vec
    }      // loop over all runs
  }        // number of runs > 0
  cout << "Exiting beginRun" << endl;
  for (int NbChan = 0; NbChan < 15; NbChan++) {
    if (hMean[NbChan]->GetEntries() > 0.) {  // save only when filled!
      hMean[NbChan]->Write();
      hRMS[NbChan]->Write();
    }
  }
  tPedHist->Write();
  tPedHist->Print();
  f.Close();

}  //beginRun

//========================================================================
void EcalPedestalHistory::endRun(edm::Run const&, edm::EventSetup const& c) {
  //========================================================================

}  //endRun

//========================================================================
void EcalPedestalHistory::beginJob() {
  ///========================================================================

}  //beginJob

//========================================================================
void EcalPedestalHistory::endJob() {
  //========================================================================

}  //endJob

//
// member functions
//

//========================================================================
void EcalPedestalHistory::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //========================================================================

  if (cnt_evt_ == 0) {
    if (ECALType_ == "EB" || ECALType_ == "EA") {
      cout << " Barrel data : nb channels " << kEBChannels << endl;
    } else if (ECALType_ == "EE" || ECALType_ == "EA") {
      cout << " End cap data : nb channels " << kEEChannels << endl;
    } else {
      cout << " strange ECALtype : " << ECALType_ << " abort " << endl;
      return;
    }
    /*
    int NbOfmaskedChannels =  maskedChannels_.size();
    cout << " Nb masked EB channels " << NbOfmaskedChannels << endl;
    for (vector<int>::iterator iter = maskedChannels_.begin(); iter != maskedChannels_.end(); ++iter)
      cout<< " : masked channel " << *(iter) << endl;
    NbOfmaskedChannels =  maskedEEChannels_.size();
    cout << " Nb masked EE channels " << NbOfmaskedChannels << endl;
    for (vector<int>::iterator iter = maskedEEChannels_.begin(); iter != maskedEEChannels_.end(); ++iter)
      cout<< " : masked channel " << *(iter) << endl;
    */
  }
  cnt_evt_++;

}  //analyze

//define this as a plug-in
DEFINE_FWK_MODULE(EcalPedestalHistory);
