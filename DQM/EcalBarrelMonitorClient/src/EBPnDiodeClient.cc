/*
 * \file EBPnDiodeClient.cc
 *
 * $Date: 2005/12/30 11:19:36 $
 * $Revision: 1.28 $
 * \author G. Della Ricca
 * \author F. Cossutti
 *
*/

#include <DQM/EcalBarrelMonitorClient/interface/EBPnDiodeClient.h>

EBPnDiodeClient::EBPnDiodeClient(const edm::ParameterSet& ps, MonitorUserInterface* mui){

  mui_ = mui;

  for ( int ism = 1; ism <= 36; ism++ ) {

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;

  }

  // collateSources switch
  collateSources_ = ps.getUntrackedParameter<bool>("collateSources", false);

  // verbosity switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);

}

EBPnDiodeClient::~EBPnDiodeClient(){

  this->cleanup();

}

void EBPnDiodeClient::beginJob(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBPnDiodeClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBPnDiodeClient::beginRun(const edm::EventSetup& c){

  if ( verbose_ ) cout << "EBPnDiodeClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

  this->subscribe();

}

void EBPnDiodeClient::endJob(void) {

  if ( verbose_ ) cout << "EBPnDiodeClient: endJob, ievt = " << ievt_ << endl;

}

void EBPnDiodeClient::endRun(void) {

  if ( verbose_ ) cout << "EBPnDiodeClient: endRun, jevt = " << jevt_ << endl;

  this->unsubscribe();

  this->cleanup();

}

void EBPnDiodeClient::setup(void) {

}

void EBPnDiodeClient::cleanup(void) {

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( h01_[ism-1] ) delete h01_[ism-1];
    h01_[ism-1] = 0;
    if ( h02_[ism-1] ) delete h02_[ism-1];
    h02_[ism-1] = 0;
    if ( h03_[ism-1] ) delete h03_[ism-1];
    h03_[ism-1] = 0;
    if ( h04_[ism-1] ) delete h04_[ism-1];
    h04_[ism-1] = 0;

  }

}

void EBPnDiodeClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, RunTag* runtag) {

  EcalLogicID ecid;
  MonPNDat p;
  map<EcalLogicID, MonPNDat> dataset;

  cout << "Creating MonPnDatObjects to database ..." << endl;

  const float n_min_tot = 1000.;
  const float n_min_bin = 50.;

  for ( int ism = 1; ism <= 36; ism++ ) {

    float num01, num02, num03, num04;
    float mean01, mean02, mean03, mean04;
    float rms01, rms02, rms03, rms04;

    for ( int i = 1; i <= 10; i++ ) {

      num01  = num02  = num03  = num04  = -1.;
      mean01 = mean02 = mean03 = mean04 = -1.;
      rms01  = rms02  = rms03  = rms04  = -1.;

      bool update_channel = false;

      if ( h01_[ism-1] && h01_[ism-1]->GetEntries() >= n_min_tot ) {
        num01 = h01_[ism-1]->GetBinEntries(h01_[ism-1]->GetBin(1, i));
        if ( num01 >= n_min_bin ) {
          mean01 = h01_[ism-1]->GetBinContent(h01_[ism-1]->GetBin(1, i));
          rms01  = h01_[ism-1]->GetBinError(h01_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( h02_[ism-1] && h02_[ism-1]->GetEntries() >= n_min_tot ) {
        num02 = h02_[ism-1]->GetBinEntries(h02_[ism-1]->GetBin(1, i));
        if ( num02 >= n_min_bin ) {
          mean02 = h02_[ism-1]->GetBinContent(h02_[ism-1]->GetBin(1, i));
          rms02  = h02_[ism-1]->GetBinError(h02_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( h03_[ism-1] && h03_[ism-1]->GetEntries() >= n_min_tot ) {
        num03 = h03_[ism-1]->GetBinEntries(h03_[ism-1]->GetBin(i));
        if ( num03 >= n_min_bin ) {
          mean03 = h03_[ism-1]->GetBinContent(h03_[ism-1]->GetBin(1, i));
          rms03  = h03_[ism-1]->GetBinError(h03_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( h04_[ism-1] && h04_[ism-1]->GetEntries() >= n_min_tot ) {
        num04 = h04_[ism-1]->GetBinEntries(h04_[ism-1]->GetBin(1, i));
        if ( num04 >= n_min_bin ) {
          mean04 = h04_[ism-1]->GetBinContent(h04_[ism-1]->GetBin(1, i));
          rms04  = h04_[ism-1]->GetBinError(h04_[ism-1]->GetBin(1, i));
          update_channel = true;
        }
      }

      if ( update_channel ) {

        if ( i == 1 ) {

          cout << "Preparing dataset for SM=" << ism << endl;

          cout << "PNs (" << i << ") L1 " << num01  << " " << mean01 << " " << rms01  << endl;
          cout << "PNs (" << i << ") L2 " << num02  << " " << mean02 << " " << rms02  << endl;
          cout << "PNs (" << i << ") L3 " << num03  << " " << mean03 << " " << rms03  << endl;
          cout << "PNs (" << i << ") L4 " << num04  << " " << mean04 << " " << rms04  << endl;

        }

        p.setADCMean(mean01);
        p.setADCRMS(rms01);

        p.setTaskStatus(true);

        if ( econn ) {
          try {
            ecid = econn->getEcalLogicID("EB_LM_PN", ism, i-1);
            dataset[ecid] = p;
          } catch (runtime_error &e) {
            cerr << e.what() << endl;
          }
        }

      }

    }

  }

  if ( econn ) {
    try {
      cout << "Inserting dataset ... " << flush;
      econn->insertDataSet(&dataset, runiov, runtag);
      cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

}

void EBPnDiodeClient::subscribe(void){

  if ( verbose_ ) cout << "EBPnDiodeClient: subscribe" << endl;

  // subscribe to all monitorable matching pattern
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/EBPDT PNs SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/EBPDT PNs SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/EBPDT PNs SM*");
  mui_->subscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/EBPDT PNs SM*");

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPnDiodeClient: collate" << endl;

    Char_t histo[80];

    for ( int ism = 1; ism <= 36; ism++ ) {

      sprintf(histo, "EBPDT PNs SM%02d L1", ism);
      me_h01_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser1/EBPDT PNs SM%02d L1", ism);
      mui_->add(me_h01_[ism-1], histo);

      sprintf(histo, "EBPDT PNs SM%02d L2", ism);
      me_h02_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser2/EBPDT PNs SM%02d L2", ism);
      mui_->add(me_h02_[ism-1], histo);

      sprintf(histo, "EBPDT PNs SM%02d L3", ism);
      me_h03_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser3/EBPDT PNs SM%02d L3", ism);
      mui_->add(me_h03_[ism-1], histo);

      sprintf(histo, "EBPDT PNs SM%02d L4", ism);
      me_h04_[ism-1] = mui_->collateProf2D(histo, histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4");
      sprintf(histo, "*/EcalBarrel/EBPnDiodeTask/Laser4/EBPDT PNs SM%02d L4", ism);
      mui_->add(me_h04_[ism-1], histo);

    }

  }

}

void EBPnDiodeClient::subscribeNew(void){

  // subscribe to new monitorable matching pattern
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser1/EBPDT PNs SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser2/EBPDT PNs SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser3/EBPDT PNs SM*");
  mui_->subscribeNew("*/EcalBarrel/EBPnDiodeTask/Laser4/EBPDT PNs SM*");

}

void EBPnDiodeClient::unsubscribe(void){

  if ( verbose_ ) cout << "EBPnDiodeClient: unsubscribe" << endl;

  if ( collateSources_ ) {

    if ( verbose_ ) cout << "EBPnDiodeClient: uncollate" << endl;

    DaqMonitorBEInterface* bei = mui_->getBEInterface();

    if ( bei ) {

      Char_t histo[80];

      for ( int ism = 1; ism <= 36; ism++ ) {

        sprintf(histo, "EBPDT PNs SM%02d L1", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBPnDiodeTask/Laser1");
        bei->removeElement(histo);

        sprintf(histo, "EBPDT PNs SM%02d L2", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBPnDiodeTask/Laser2");
        bei->removeElement(histo);

        sprintf(histo, "EBPDT PNs SM%02d L3", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBPnDiodeTask/Laser3");
        bei->removeElement(histo);

        sprintf(histo, "EBPDT PNs SM%02d L4", ism);
        bei->setCurrentFolder("EcalBarrel/Sums/EBPnDiodeTask/Laser4");
        bei->removeElement(histo);

      }

    }

  }

  // unsubscribe to all monitorable matching pattern
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser1/EBPDT PNs SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser2/EBPDT PNs SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser3/EBPDT PNs SM*");
  mui_->unsubscribe("*/EcalBarrel/EBPnDiodeTask/Laser4/EBPDT PNs SM*");

}

void EBPnDiodeClient::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( verbose_ ) cout << "EBPnDiodeClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  Char_t histo[150];

  MonitorElement* me;
  MonitorElementT<TNamed>* ob;

  for ( int ism = 1; ism <= 36; ism++ ) {

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser1/EBPDT PNs SM%02d L1", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser1/EBPDT PNs SM%02d L1", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h01_[ism-1] ) delete h01_[ism-1];
        sprintf(histo, "ME EBPDT PNs SM%02d L1", ism);
        h01_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h01_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser2/EBPDT PNs SM%02d L2", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser2/EBPDT PNs SM%02d L2", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h02_[ism-1] ) delete h02_[ism-1];
        sprintf(histo, "ME EBPDT PNs SM%02d L2", ism);
        h02_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h02_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser3/EBPDT PNs SM%02d L3", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser3/EBPDT PNs SM%02d L3", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h03_[ism-1] ) delete h03_[ism-1];
        sprintf(histo, "ME EBPDT PNs SM%02d L3", ism);
        h03_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h03_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

    if ( collateSources_ ) {
      sprintf(histo, "EcalBarrel/Sums/EBPnDiodeTask/Laser4/EBPDT PNs SM%02d L4", ism);
    } else {
      sprintf(histo, "Collector/FU0/EcalBarrel/EBPnDiodeTask/Laser4/EBPDT PNs SM%02d L4", ism);
    }
    me = mui_->get(histo);
    if ( me ) {
      if ( verbose_ ) cout << "Found '" << histo << "'" << endl;
      ob = dynamic_cast<MonitorElementT<TNamed>*> (me);
      if ( ob ) {
        if ( h04_[ism-1] ) delete h04_[ism-1];
        sprintf(histo, "ME EBPDT PNs SM%02d L4", ism);
        h04_[ism-1] = dynamic_cast<TProfile2D*> ((ob->operator->())->Clone(histo));
//        h04_[ism-1] = dynamic_cast<TProfile2D*> (ob->operator->());
      }
    }

  }

}

void EBPnDiodeClient::htmlOutput(int run, string htmlDir, string htmlName){

  cout << "Preparing EBPnDiodeClient html output ..." << endl;

  ofstream htmlFile;

  htmlFile.open((htmlDir + htmlName).c_str());

  // html page header
  htmlFile << "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">  " << endl;
  htmlFile << "<html>  " << endl;
  htmlFile << "<head>  " << endl;
  htmlFile << "  <meta content=\"text/html; charset=ISO-8859-1\"  " << endl;
  htmlFile << " http-equiv=\"content-type\">  " << endl;
  htmlFile << "  <title>Monitor:PnDiodeTask output</title> " << endl;
  htmlFile << "</head>  " << endl;
  htmlFile << "<style type=\"text/css\"> td { font-weight: bold } </style>" << endl;
  htmlFile << "<body>  " << endl;
  htmlFile << "<br>  " << endl;
  htmlFile << "<h2>Run:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" << endl;
  htmlFile << "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">" << run << "</span></h2>" << endl;
  htmlFile << "<h2>Monitoring task:&nbsp;&nbsp;&nbsp;&nbsp; <span " << endl;
  htmlFile << " style=\"color: rgb(0, 0, 153);\">PNDIODE</span></h2> " << endl;
  htmlFile << "<hr>" << endl;
//  htmlFile << "<table border=1><tr><td bgcolor=red>channel has problems in this task</td>" << endl;
//  htmlFile << "<td bgcolor=lime>channel has NO problems</td>" << endl;
//  htmlFile << "<td bgcolor=yellow>channel is missing</td></table>" << endl;
//  htmlFile << "<hr>" << endl;

  // Produce the plots to be shown as .png files from existing histograms

  int csize = 250;

//  double histMax = 1.e15;

  string imgNameME[2], imgName, meName;

  TCanvas* cAmp = new TCanvas("cAmp", "Temp", csize, csize);

  TH1D* obj1d;

  // Loop on barrel supermodules

  for ( int ism = 1 ; ism <= 36 ; ism++ ) {

    // Loop on wavelength

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      // Monitoring elements plots

      imgNameME[iCanvas-1] = "";

      obj1d = 0;
      switch ( iCanvas ) {
        case 1:
          if ( h01_[ism-1] ) obj1d = h01_[ism-1]->ProjectionY("_py", 1, 10, "e");
          break;
        case 2:
          if ( h02_[ism-1] ) obj1d = h02_[ism-1]->ProjectionY("_py", 1, 10, "e");
          break;
        case 3:
          if ( h03_[ism-1] ) obj1d = h03_[ism-1]->ProjectionY("_py", 1, 10, "e");
          break;
        case 4:
          if ( h04_[ism-1] ) obj1d = h04_[ism-1]->ProjectionY("_py", 1, 10, "e");
          break;
        default:
          break;
      }

      if ( obj1d ) {

        meName = obj1d->GetName();

        for ( unsigned int i = 0; i < meName.size(); i++ ) {
          if ( meName.substr(i, 1) == " " )  {
            meName.replace(i, 1 ,"_" );
          }
        }
        imgNameME[iCanvas-1] = meName + ".png";
        imgName = htmlDir + imgNameME[iCanvas-1];

        cAmp->cd();
        gStyle->SetOptStat("euomr");
        obj1d->SetStats(kTRUE);
//        if ( obj1d->GetMaximum(histMax) > 0. ) {
//          gPad->SetLogy(1);
//        } else {
//          gPad->SetLogy(0);
//        }
        obj1d->Draw();
        cAmp->Update();
        cAmp->SaveAs(imgName.c_str());
        gPad->SetLogy(0);

        delete obj1d;

      }

    }

    htmlFile << "<h3><strong>Supermodule&nbsp;&nbsp;" << ism << "</strong></h3>" << endl;
    htmlFile << "<table border=\"0\" cellspacing=\"0\" " << endl;
    htmlFile << "cellpadding=\"10\" align=\"center\"> " << endl;
    htmlFile << "<tr align=\"center\">" << endl;

    for ( int iCanvas = 1 ; iCanvas <= 2 ; iCanvas++ ) {

      if ( imgNameME[iCanvas-1].size() != 0 )
        htmlFile << "<td colspan=\"2\"><img src=\"" << imgNameME[iCanvas-1] << "\"></td>" << endl;
      else
        htmlFile << "<td><img src=\"" << " " << "\"></td>" << endl;

    }

    htmlFile << "</tr>" << endl;
    htmlFile << "</table>" << endl;
    htmlFile << "<br>" << endl;

  }

  delete cAmp;

  // html page footer
  htmlFile << "</body> " << endl;
  htmlFile << "</html> " << endl;

  htmlFile.close();

}

