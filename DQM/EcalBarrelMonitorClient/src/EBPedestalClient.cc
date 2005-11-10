/*
 * \file EBPedestalClient.cc
 * 
 * $Date: 2005/11/10 09:55:15 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPedestalClient.h>

EBPedestalClient::EBPedestalClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

}

EBPedestalClient::~EBPedestalClient(){

  this->unsubscribe();

}

void EBPedestalClient::beginJob(const edm::EventSetup& c){

  cout << "EBPedestalClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

  this->subscribe();

}

void EBPedestalClient::beginRun(const edm::EventSetup& c){

  cout << "EBPedestalClient: beginRun" << endl;

  jevt_ = 0;

  this->subscribe();

}

void EBPedestalClient::endJob(void) {

  cout << "EBPedestalClient: endJob, ievt = " << ievt_ << endl;

}

void EBPedestalClient::endRun(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  cout << "EBPedestalClient: endRun, jevt = " << jevt_ << endl;

  EcalLogicID ecid;
  MonPedestalsDat p;
  map<EcalLogicID, MonPedestalsDat> dataset;

  cout << "Writing MonPedestalsDatObjects to database ..." << endl;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03;
    float mean01, mean02, mean03;
    float rms01, rms02, rms03;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        num01  = num02  = num03  = -1.;
        mean01 = mean02 = mean03 = -1.;
        rms01  = rms02  = rms03  = -1.;

        bool update_channel = false;

        MonitorElementT<TNamed>* ob;

        TProfile2D* h01 = 0;
        TProfile2D* h02 = 0;
        TProfile2D* h03 = 0;

        if ( me01[ism-1] ) {
          ob = dynamic_cast<MonitorElementT<TNamed>*> (me01[ism-1]);
          if ( ob ) h01 = dynamic_cast<TProfile2D*> (ob->operator->());
        }

        if ( me02[ism-1] ) {
          ob = dynamic_cast<MonitorElementT<TNamed>*> (me02[ism-1]);
          if ( ob ) h02 = dynamic_cast<TProfile2D*> (ob->operator->());
        }

        if ( me03[ism-1] ) {
          ob = dynamic_cast<MonitorElementT<TNamed>*> (me03[ism-1]);
          if ( ob ) h03 = dynamic_cast<TProfile2D*> (ob->operator->());
        }

        float n_min_tot = 1000.;
        float n_min_bin = 50.;

        if ( h01 && h01->GetEntries() >= n_min_tot ) {
          num01 = h01->GetBinEntries(h01->GetBin(ie, ip));
          if ( num01 >= n_min_bin ) {
            mean01 = h01->GetBinContent(h01->GetBin(ie, ip));
            rms01  = h01->GetBinError(h01->GetBin(ie, ip));
            update_channel = true;
          }
        }
  
        if ( h02 && h02->GetEntries() >= n_min_tot ) {
          num02 = h02->GetBinEntries(h02->GetBin(ie, ip));
          if ( num02 >= n_min_bin ) {
            mean02 = h02->GetBinContent(h02->GetBin(ie, ip));
            rms02  = h02->GetBinError(h02->GetBin(ie, ip));
            update_channel = true;
          }
        }
  
        if ( h03 && h03->GetEntries() >= n_min_tot ) {
          num03 = h03->GetBinEntries(h03->GetBin(ie, ip));
          if ( num03 >= n_min_bin ) {
            mean03 = h03->GetBinContent(h03->GetBin(ie, ip));
            rms03  = h03->GetBinError(h03->GetBin(ie, ip));
            update_channel = true;
          }
        }

        if ( update_channel ) {

          if ( ie == 1 && ip == 1 ) {

            cout << "Inserting dataset for SM=" << ism << endl;

            cout << "G01 (" << ie << "," << ip << ") " << num01  << " "
                                                       << mean01 << " "
                                                       << rms01  << endl;
            cout << "G06 (" << ie << "," << ip << ") " << num02  << " "
                                                       << mean02 << " "
                                                       << rms02  << endl;
            cout << "G12 (" << ie << "," << ip << ") " << num03  << " "
                                                       << mean03 << " "
                                                       << rms03  << endl;

          }

          p.setPedMeanG1(mean01);
          p.setPedRMSG1(rms01);

          p.setPedMeanG6(mean02);
          p.setPedRMSG6(rms02);

          p.setPedMeanG12(mean03);
          p.setPedRMSG12(rms03);

          p.setTaskStatus(1);

          try {
            ecid = econn->getEcalLogicID("EB_crystal_index", ism, ie-1, ip-1);
            dataset[ecid] = p;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }

        }

      }
    }

  }

  try {
    cout << "Inserting dataset in DB." << endl;
    if ( econn ) econn->insertDataSet(&dataset, runiov, runtag );
  } catch (runtime_error &e) {
    cerr << e.what() << endl;
  }

}

void EBPedestalClient::subscribe(void){

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBPedestalClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBPedestalClient::unsubscribe(void){

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM*");

}

void EBPedestalClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  cout << "EBPedestalClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;

  this->subscribeNew();

  Char_t histo[150];

  for ( int ism = 1; ism <= 36; ism++ ) {
    me01[ism-1] = me02[ism-1] = me03[ism-1] = 0;
  }

  for ( int ism = 1; ism <= 36; ism++ ) {

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain01/EBPT pedestal SM%02d G01", ism);
    me01[ism-1] = mui_->get(histo);
    if ( me01[ism-1] ) {
      cout << "Found '" << histo << "'" << endl;
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain06/EBPT pedestal SM%02d G06", ism);
    me02[ism-1] = mui_->get(histo);
    if ( me02[ism-1] ) {
      cout << "Found '" << histo << "'" << endl;
    }

    sprintf(histo, "Collector/FU0/EcalBarrel/EBPedestalTask/Gain12/EBPT pedestal SM%02d G12", ism);
    me03[ism-1] = mui_->get(histo);
    if ( me03[ism-1] ) {
      cout << "Found '" << histo << "'" << endl;
    }

  }

}

