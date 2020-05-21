/*!
  \file MonitorElementsDb.cc
  \brief Generate a Monitor Element from DB data
  \author B. Gobbo
*/

#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cmath>
#include <fstream>
#include <iostream>

#include "DQMServices/Core/interface/DQMStore.h"

#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ITransaction.h"

#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeList.h"

#include "TCanvas.h"
#include "TStyle.h"

#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

#include "DQM/EcalMonitorDbModule/interface/MonitorElementsDb.h"

MonitorElementsDb::MonitorElementsDb(const edm::ParameterSet &ps, std::string &xmlFile) {
  xmlFile_ = xmlFile;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  if (dqmStore_) {
    dqmStore_->setCurrentFolder(prefixME_);

    parser_ = new MonitorXMLParser(xmlFile_);
    try {
      parser_->load();
    } catch (std::runtime_error const &e) {
      delete parser_;
      parser_ = nullptr;
      std::cerr << "Error loading parser: " << e.what() << std::endl;
    }

    if (parser_)
      MEinfo_ = parser_->getDB_ME();

    for (auto &i : MEinfo_) {
      MonitorElement *tmp;
      tmp = nullptr;
      if (strcmp(i.type.c_str(), "th1d") == 0) {
        tmp = dqmStore_->book1D(i.title, i.title, i.xbins, i.xfrom, i.xto);
      } else if (strcmp(i.type.c_str(), "th2d") == 0) {
        tmp = dqmStore_->book2D(i.title, i.title, i.xbins, i.xfrom, i.xto, i.ybins, i.yfrom, i.yto);
      } else if (strcmp(i.type.c_str(), "tprofile") == 0) {
        tmp = dqmStore_->bookProfile(i.title, i.title, i.xbins, i.xfrom, i.xto, i.ybins, i.yfrom, i.yto);
      } else if (strcmp(i.type.c_str(), "tprofile2d") == 0) {
        tmp = dqmStore_->bookProfile2D(
            i.title, i.title, i.xbins, i.xfrom, i.xto, i.ybins, i.yfrom, i.yto, i.zbins, i.zfrom, i.zto);
      }

      MEs_.push_back(tmp);
    }
  }

  ievt_ = 0;
}

MonitorElementsDb::~MonitorElementsDb() { delete parser_; }

void MonitorElementsDb::beginJob(void) { ievt_ = 0; }

void MonitorElementsDb::endJob(void) { std::cout << "MonitorElementsDb: analyzed " << ievt_ << " events" << std::endl; }

void MonitorElementsDb::analyze(const edm::Event &e, const edm::EventSetup &c, coral::ISessionProxy *session) {
  ievt_++;

  bool atLeastAQuery;
  atLeastAQuery = false;

  std::vector<std::string> vars;

  if (session) {
    for (unsigned int i = 0; i < MEinfo_.size(); i++) {
      // i-th ME...

      if (MEs_[i] != nullptr && (ievt_ % MEinfo_[i].ncycle) == 0) {
        MEs_[i]->Reset();

        vars.clear();

        try {
          atLeastAQuery = true;

          session->transaction().start(true);

          coral::ISchema &schema = session->nominalSchema();

          coral::IQuery *query = schema.newQuery();

          for (auto &querie : MEinfo_[i].queries) {
            if (strcmp(querie.query.c_str(), "addToTableList") == 0) {
              query->addToTableList(querie.arg);
            } else if (strcmp(querie.query.c_str(), "addToOutputList") == 0) {
              query->addToOutputList(querie.arg, querie.alias);
              vars.push_back(querie.alias);
            } else if (strcmp(querie.query.c_str(), "setCondition") == 0) {
              query->setCondition(querie.arg, coral::AttributeList());
            } else if (strcmp(querie.query.c_str(), "addToOrderList") == 0) {
              query->addToOrderList(querie.arg);
            }
          }

          coral::ICursor &cursor = query->execute();

          unsigned int k = 0;

          while (cursor.next() && k < MEinfo_[i].loop) {
            // while ( cursor.next() ) {

            const coral::AttributeList &row = cursor.currentRow();

            std::vector<float> vvars;
            vvars.clear();
            for (auto &var : vars) {
              if (!var.empty()) {
                vvars.push_back(row[var].data<float>());
              }
            }
            if (vvars.size() == 2) {
              // std::cout << k << " -- " << vvars[0] << " -- " << vvars[1] <<
              // std::endl;
              MEs_[i]->Fill(vvars[0], vvars[1]);
            } else if (vvars.size() == 3) {
              // std::cout << k << " -- " << vvars[0] << " -- " << vvars[1] << "
              // -- " << vvars[2] << std::endl;
              MEs_[i]->Fill(vvars[0], vvars[1], vvars[2]);
            } else if (vvars.size() == 4) {
              // std::cout << k << " -- " << vvars[0] << " -- " << vvars[1] << "
              // -- " << vvars[2] << " -- " << vvars[3] << std::endl;
              MEs_[i]->Fill(vvars[0], vvars[1], vvars[2], vvars[3]);
            } else {
              std::cerr << "Too many variables to plot..." << std::endl;
              exit(1);
            }

            k++;
          }

          delete query;

        } catch (coral::Exception &e) {
          std::cerr << "CORAL Exception : " << e.what() << std::endl;
        } catch (std::exception &e) {
          std::cerr << "Standard C++ exception : " << e.what() << std::endl;
        }
      }
    }

    if (atLeastAQuery)
      session->transaction().commit();
  }
}

void MonitorElementsDb::htmlOutput(std::string &htmlDir) {
  gStyle->SetOptStat(0);
  gStyle->SetOptFit();
  gStyle->SetPalette(1, nullptr);

  for (unsigned int i = 0; i < MEinfo_.size(); i++) {
    if (MEs_[i] != nullptr && (ievt_ % MEinfo_[i].ncycle) == 0) {
      TCanvas *c1;
      int n = MEinfo_[i].xbins > MEinfo_[i].ybins ? int(round(float(MEinfo_[i].xbins) / float(MEinfo_[i].ybins)))
                                                  : int(round(float(MEinfo_[i].ybins) / float(MEinfo_[i].xbins)));
      if (MEinfo_[i].xbins > MEinfo_[i].ybins) {
        c1 = new TCanvas("c1", "dummy", 400 * n, 400);
      } else {
        c1 = new TCanvas("c1", "dummy", 400, 400 * n);
      }
      c1->SetGrid();
      c1->cd();

      const double histMax = 1.e15;

      TObject *ob = const_cast<MonitorElement *>(MEs_[i])->getRootObject();
      if (ob) {
        if (dynamic_cast<TH1F *>(ob)) {
          TH1F *h = dynamic_cast<TH1F *>(ob);
          h->Draw();
        } else if (dynamic_cast<TH2F *>(ob)) {
          TH2F *h = dynamic_cast<TH2F *>(ob);
          if (h->GetMaximum(histMax) > 1.e4) {
            gPad->SetLogz(1);
          } else {
            gPad->SetLogz(0);
          }
          h->Draw("colz");
        } else if (dynamic_cast<TProfile *>(ob)) {
          TProfile *h = dynamic_cast<TProfile *>(ob);
          if (h->GetMaximum(histMax) > 1.e4) {
            gPad->SetLogz(1);
          } else {
            gPad->SetLogz(0);
          }
          h->Draw("colz");
        }
      }

      c1->Update();
      std::string name = htmlDir + "/" + MEinfo_[i].title + ".png";
      c1->SaveAs(name.c_str());

      delete c1;
    }
  }
}
