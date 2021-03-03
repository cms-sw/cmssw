// Original Author:  Steven Won
// Original Author:  Alan Campbell for castor
//         Created:  Fri May  2 15:34:43 CEST 2008
// Written to replace the combination of HcalPedestalAnalyzer and HcalPedestalAnalysis
// This code runs 1000x faster and produces all outputs from a single run
// (ADC, fC in .txt plus an .xml file)
//
#include <memory>
#include "CalibCalorimetry/CastorCalib/interface/CastorPedestalsAnalysis.h"

CastorPedestalsAnalysis::CastorPedestalsAnalysis(const edm::ParameterSet& ps)
    : castorDigiCollectionTag(ps.getParameter<edm::InputTag>("castorDigiCollectionTag")) {
  hiSaveFlag = ps.getUntrackedParameter<bool>("hiSaveFlag", false);
  dumpXML = ps.getUntrackedParameter<bool>("dumpXML", false);
  verboseflag = ps.getUntrackedParameter<bool>("verbose", false);
  firstTS = ps.getUntrackedParameter<int>("firstTS", 0);
  lastTS = ps.getUntrackedParameter<int>("lastTS", 9);
  firsttime = true;

  tok_cond_ = esConsumes<CastorDbService, CastorDbRecord>();
  tok_map_ = esConsumes<CastorElectronicsMap, CastorElectronicsMapRcd>();
}

CastorPedestalsAnalysis::~CastorPedestalsAnalysis() {
  CastorPedestals* rawPedsItem = new CastorPedestals(true);
  CastorPedestalWidths* rawWidthsItem = new CastorPedestalWidths(true);
  CastorPedestals* rawPedsItemfc = new CastorPedestals(false);
  CastorPedestalWidths* rawWidthsItemfc = new CastorPedestalWidths(false);

  //Calculate pedestal constants
  std::cout << "Calculating Pedestal constants...\n";
  std::vector<NewPedBunch>::iterator bunch_it;
  for (bunch_it = Bunches.begin(); bunch_it != Bunches.end(); ++bunch_it) {
    if (bunch_it->usedflag) {
      if (verboseflag)
        std::cout << "Analyzing channel sector= " << bunch_it->detid.sector()
                  << " module = " << bunch_it->detid.module() << std::endl;
      //pedestal constant is the mean
      bunch_it->cap[0] /= bunch_it->num[0][0];
      bunch_it->cap[1] /= bunch_it->num[1][1];
      bunch_it->cap[2] /= bunch_it->num[2][2];
      bunch_it->cap[3] /= bunch_it->num[3][3];
      bunch_it->capfc[0] /= bunch_it->num[0][0];
      bunch_it->capfc[1] /= bunch_it->num[1][1];
      bunch_it->capfc[2] /= bunch_it->num[2][2];
      bunch_it->capfc[3] /= bunch_it->num[3][3];
      //widths are the covariance matrix--assumed symmetric
      bunch_it->sig[0][0] = (bunch_it->prod[0][0] / bunch_it->num[0][0]) - (bunch_it->cap[0] * bunch_it->cap[0]);
      bunch_it->sig[0][1] = (bunch_it->prod[0][1] / bunch_it->num[0][1]) - (bunch_it->cap[0] * bunch_it->cap[1]);
      bunch_it->sig[0][2] = (bunch_it->prod[0][2] / bunch_it->num[0][2]) - (bunch_it->cap[0] * bunch_it->cap[2]);
      bunch_it->sig[0][3] = (bunch_it->prod[0][3] / bunch_it->num[0][3]) - (bunch_it->cap[0] * bunch_it->cap[3]);
      bunch_it->sig[1][0] = (bunch_it->prod[1][0] / bunch_it->num[1][0]) - (bunch_it->cap[1] * bunch_it->cap[0]);
      bunch_it->sig[1][1] = (bunch_it->prod[1][1] / bunch_it->num[1][1]) - (bunch_it->cap[1] * bunch_it->cap[1]);
      bunch_it->sig[1][2] = (bunch_it->prod[1][2] / bunch_it->num[1][2]) - (bunch_it->cap[1] * bunch_it->cap[2]);
      bunch_it->sig[1][3] = (bunch_it->prod[1][3] / bunch_it->num[1][3]) - (bunch_it->cap[1] * bunch_it->cap[3]);
      bunch_it->sig[2][0] = (bunch_it->prod[2][0] / bunch_it->num[2][0]) - (bunch_it->cap[2] * bunch_it->cap[0]);
      bunch_it->sig[2][1] = (bunch_it->prod[2][1] / bunch_it->num[2][1]) - (bunch_it->cap[2] * bunch_it->cap[1]);
      bunch_it->sig[2][2] = (bunch_it->prod[2][2] / bunch_it->num[2][2]) - (bunch_it->cap[2] * bunch_it->cap[2]);
      bunch_it->sig[2][3] = (bunch_it->prod[2][3] / bunch_it->num[2][3]) - (bunch_it->cap[2] * bunch_it->cap[3]);
      bunch_it->sig[3][0] = (bunch_it->prod[3][0] / bunch_it->num[3][0]) - (bunch_it->cap[3] * bunch_it->cap[0]);
      bunch_it->sig[3][1] = (bunch_it->prod[3][1] / bunch_it->num[3][1]) - (bunch_it->cap[3] * bunch_it->cap[1]);
      bunch_it->sig[3][2] = (bunch_it->prod[3][2] / bunch_it->num[3][2]) - (bunch_it->cap[3] * bunch_it->cap[2]);
      bunch_it->sig[3][3] = (bunch_it->prod[3][3] / bunch_it->num[3][3]) - (bunch_it->cap[3] * bunch_it->cap[3]);

      bunch_it->sigfc[0][0] =
          (bunch_it->prodfc[0][0] / bunch_it->num[0][0]) - (bunch_it->capfc[0] * bunch_it->capfc[0]);
      bunch_it->sigfc[0][1] =
          (bunch_it->prodfc[0][1] / bunch_it->num[0][1]) - (bunch_it->capfc[0] * bunch_it->capfc[1]);
      bunch_it->sigfc[0][2] =
          (bunch_it->prodfc[0][2] / bunch_it->num[0][2]) - (bunch_it->capfc[0] * bunch_it->capfc[2]);
      bunch_it->sigfc[0][3] =
          (bunch_it->prodfc[0][3] / bunch_it->num[0][3]) - (bunch_it->capfc[0] * bunch_it->capfc[3]);
      bunch_it->sigfc[1][0] =
          (bunch_it->prodfc[1][0] / bunch_it->num[1][0]) - (bunch_it->capfc[1] * bunch_it->capfc[0]);
      bunch_it->sigfc[1][1] =
          (bunch_it->prodfc[1][1] / bunch_it->num[1][1]) - (bunch_it->capfc[1] * bunch_it->capfc[1]);
      bunch_it->sigfc[1][2] =
          (bunch_it->prodfc[1][2] / bunch_it->num[1][2]) - (bunch_it->capfc[1] * bunch_it->capfc[2]);
      bunch_it->sigfc[1][3] =
          (bunch_it->prodfc[1][3] / bunch_it->num[1][3]) - (bunch_it->capfc[1] * bunch_it->capfc[3]);
      bunch_it->sigfc[2][0] =
          (bunch_it->prodfc[2][0] / bunch_it->num[2][0]) - (bunch_it->capfc[2] * bunch_it->capfc[0]);
      bunch_it->sigfc[2][1] =
          (bunch_it->prodfc[2][1] / bunch_it->num[2][1]) - (bunch_it->capfc[2] * bunch_it->capfc[1]);
      bunch_it->sigfc[2][2] =
          (bunch_it->prodfc[2][2] / bunch_it->num[2][2]) - (bunch_it->capfc[2] * bunch_it->capfc[2]);
      bunch_it->sigfc[2][3] =
          (bunch_it->prodfc[2][3] / bunch_it->num[2][3]) - (bunch_it->capfc[2] * bunch_it->capfc[3]);
      bunch_it->sigfc[3][0] =
          (bunch_it->prodfc[3][0] / bunch_it->num[3][0]) - (bunch_it->capfc[3] * bunch_it->capfc[0]);
      bunch_it->sigfc[3][1] =
          (bunch_it->prodfc[3][1] / bunch_it->num[3][1]) - (bunch_it->capfc[3] * bunch_it->capfc[1]);
      bunch_it->sigfc[3][2] =
          (bunch_it->prodfc[3][2] / bunch_it->num[3][2]) - (bunch_it->capfc[3] * bunch_it->capfc[2]);
      bunch_it->sigfc[3][3] =
          (bunch_it->prodfc[3][3] / bunch_it->num[3][3]) - (bunch_it->capfc[3] * bunch_it->capfc[3]);

      for (int i = 0; i != 3; i++) {
        CASTORMeans->Fill(bunch_it->cap[i]);
        CASTORWidths->Fill(bunch_it->sig[i][i]);
      }

      //if(bunch_it->detid.subdet() == 1){

      int fillphi = bunch_it->detid.sector();
      //if (bunch_it->detid.depth()==4) fillphi++;

      //    dephist[bunch_it->detid.module()-1]->Fill(bunch_it->detid.ieta(),fillphi,
      //             (bunch_it->cap[0]+bunch_it->cap[1]+bunch_it->cap[2]+bunch_it->cap[3])/4);
      dephist->Fill(bunch_it->detid.module(),
                    fillphi,
                    (bunch_it->cap[0] + bunch_it->cap[1] + bunch_it->cap[2] + bunch_it->cap[3]) / 4);

      const CastorPedestal item(bunch_it->detid,
                                bunch_it->cap[0],
                                bunch_it->cap[1],
                                bunch_it->cap[2],
                                bunch_it->cap[3],
                                bunch_it->sig[0][0],
                                bunch_it->sig[1][1],
                                bunch_it->sig[2][2],
                                bunch_it->sig[3][3]);
      rawPedsItem->addValues(item);
      CastorPedestalWidth widthsp(bunch_it->detid);
      widthsp.setSigma(0, 0, bunch_it->sig[0][0]);
      widthsp.setSigma(0, 1, bunch_it->sig[0][1]);
      widthsp.setSigma(0, 2, bunch_it->sig[0][2]);
      widthsp.setSigma(0, 3, bunch_it->sig[0][3]);
      widthsp.setSigma(1, 0, bunch_it->sig[1][0]);
      widthsp.setSigma(1, 1, bunch_it->sig[1][1]);
      widthsp.setSigma(1, 2, bunch_it->sig[1][2]);
      widthsp.setSigma(1, 3, bunch_it->sig[1][3]);
      widthsp.setSigma(2, 0, bunch_it->sig[2][0]);
      widthsp.setSigma(2, 1, bunch_it->sig[2][1]);
      widthsp.setSigma(2, 2, bunch_it->sig[2][2]);
      widthsp.setSigma(2, 3, bunch_it->sig[2][3]);
      widthsp.setSigma(3, 0, bunch_it->sig[3][0]);
      widthsp.setSigma(3, 1, bunch_it->sig[3][1]);
      widthsp.setSigma(3, 2, bunch_it->sig[3][2]);
      widthsp.setSigma(3, 3, bunch_it->sig[3][3]);
      rawWidthsItem->addValues(widthsp);

      const CastorPedestal itemfc(bunch_it->detid,
                                  bunch_it->capfc[0],
                                  bunch_it->capfc[1],
                                  bunch_it->capfc[2],
                                  bunch_it->capfc[3],
                                  bunch_it->sigfc[0][0],
                                  bunch_it->sigfc[1][1],
                                  bunch_it->sigfc[2][2],
                                  bunch_it->sigfc[3][3]);
      rawPedsItemfc->addValues(itemfc);
      CastorPedestalWidth widthspfc(bunch_it->detid);
      widthspfc.setSigma(0, 0, bunch_it->sigfc[0][0]);
      widthspfc.setSigma(0, 1, bunch_it->sigfc[0][1]);
      widthspfc.setSigma(0, 2, bunch_it->sigfc[0][2]);
      widthspfc.setSigma(0, 3, bunch_it->sigfc[0][3]);
      widthspfc.setSigma(1, 0, bunch_it->sigfc[1][0]);
      widthspfc.setSigma(1, 1, bunch_it->sigfc[1][1]);
      widthspfc.setSigma(1, 2, bunch_it->sigfc[1][2]);
      widthspfc.setSigma(1, 3, bunch_it->sigfc[1][3]);
      widthspfc.setSigma(2, 0, bunch_it->sigfc[2][0]);
      widthspfc.setSigma(2, 1, bunch_it->sigfc[2][1]);
      widthspfc.setSigma(2, 2, bunch_it->sigfc[2][2]);
      widthspfc.setSigma(2, 3, bunch_it->sigfc[2][3]);
      widthspfc.setSigma(3, 0, bunch_it->sigfc[3][0]);
      widthspfc.setSigma(3, 1, bunch_it->sigfc[3][1]);
      widthspfc.setSigma(3, 2, bunch_it->sigfc[3][2]);
      widthspfc.setSigma(3, 3, bunch_it->sigfc[3][3]);
      rawWidthsItemfc->addValues(widthspfc);
    }
  }

  // dump the resulting list of pedestals into a file
  std::ofstream outStream1(pedsADCfilename.c_str());
  CastorDbASCIIIO::dumpObject(outStream1, (*rawPedsItem));
  std::ofstream outStream2(widthsADCfilename.c_str());
  CastorDbASCIIIO::dumpObject(outStream2, (*rawWidthsItem));

  std::ofstream outStream3(pedsfCfilename.c_str());
  CastorDbASCIIIO::dumpObject(outStream3, (*rawPedsItemfc));
  std::ofstream outStream4(widthsfCfilename.c_str());
  CastorDbASCIIIO::dumpObject(outStream4, (*rawWidthsItemfc));

  if (dumpXML) {
    std::ofstream outStream5(XMLfilename.c_str());
    //  CastorCondXML::dumpObject (outStream5, runnum, runnum, runnum, XMLtag, 1, (*rawPedsItem), (*rawWidthsItem));
  }

  if (hiSaveFlag) {
    theFile->Write();
  } else {
    theFile->cd();
    theFile->cd("CASTOR");
    CASTORMeans->Write();
    CASTORWidths->Write();
  }
  theFile->cd();
  dephist->Write();
  dephist->SetDrawOption("colz");
  dephist->GetXaxis()->SetTitle("module");
  dephist->GetYaxis()->SetTitle("sector");

  //for (int n=0; n!= 4; n++)
  //{
  //dephist[n]->Write();
  //dephist[n]->SetDrawOption("colz");
  //dephist[n]->GetXaxis()->SetTitle("i#eta");
  //dephist[n]->GetYaxis()->SetTitle("i#phi");
  //}

  std::stringstream tempstringout;
  tempstringout << runnum;
  std::string name1 = tempstringout.str() + "_pedplots_1d.png";
  std::string name2 = tempstringout.str() + "_pedplots_2d.png";

  TStyle* theStyle = new TStyle("style", "null");
  theStyle->SetPalette(1, nullptr);
  theStyle->SetCanvasDefH(1200);  //Height of canvas
  theStyle->SetCanvasDefW(1600);  //Width of canvas

  gStyle = theStyle;
  /*
    TCanvas * c1 = new TCanvas("c1","graph",1);
    c1->Divide(2,2);
    c1->cd(1);
    CASTORMeans->Draw();
    c1->SaveAs(name1.c_str());   

    theStyle->SetOptStat("n");
    gStyle = theStyle;

    TCanvas * c2 = new TCanvas("c2","graph",1);
 //   c2->Divide(2,2);
    c2->cd(1);
    dephist->Draw();
    dephist->SetDrawOption("colz");
    //c2->cd(2);
    //dephist[1]->Draw();
    //dephist[1]->SetDrawOption("colz");
    //c2->cd(3);
    //dephist[2]->Draw();
    //dephist[2]->SetDrawOption("colz");
    //c2->cd(4);
    //dephist[3]->Draw();
    //dephist[3]->SetDrawOption("colz");
    c2->SaveAs(name2.c_str());
*/
  std::cout << "Writing ROOT file... ";
  theFile->Close();
  std::cout << "ROOT file closed.\n";
}

// ------------ method called to for each event  ------------
void CastorPedestalsAnalysis::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::Handle<CastorDigiCollection> castor;
  e.getByLabel(castorDigiCollectionTag, castor);

  auto conditions = &iSetup.getData(tok_cond_);
  const CastorQIEShape* shape = conditions->getCastorShape();

  if (firsttime) {
    runnum = e.id().run();
    std::string runnum_string;
    std::stringstream tempstringout;
    tempstringout << runnum;
    runnum_string = tempstringout.str();
    ROOTfilename = runnum_string + "-peds_ADC.root";
    pedsADCfilename = runnum_string + "-peds_ADC.txt";
    pedsfCfilename = runnum_string + "-peds_fC.txt";
    widthsADCfilename = runnum_string + "-widths_ADC.txt";
    widthsfCfilename = runnum_string + "-widths_fC.txt";
    XMLfilename = runnum_string + "-peds_ADC_complete.xml";
    XMLtag = "Castor_pedestals_" + runnum_string;

    theFile = new TFile(ROOTfilename.c_str(), "RECREATE");
    theFile->cd();
    // Create sub-directories
    theFile->mkdir("CASTOR");
    theFile->cd();

    CASTORMeans = new TH1F("All Ped Means CASTOR", "All Ped Means CASTOR", 100, 0, 9);
    CASTORWidths = new TH1F("All Ped Widths CASTOR", "All Ped Widths CASTOR", 100, 0, 3);

    dephist = new TH2F("Pedestals (ADC)", "All Castor", 14, 0., 14.5, 16, .5, 16.5);
    // dephist[0] = new TH2F("Pedestals (ADC)","Depth 1",89, -44, 44, 72, .5, 72.5);
    // dephist[1] = new TH2F("Pedestals (ADC)","Depth 2",89, -44, 44, 72, .5, 72.5);
    // dephist[2] = new TH2F("Pedestals (ADC)","Depth 3",89, -44, 44, 72, .5, 72.5);
    // dephist[3] = new TH2F("Pedestals (ADC)","Depth 4",89, -44, 44, 72, .5, 72.5);

    const CastorElectronicsMap* myRefEMap = &iSetup.getData(tok_map_);
    std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
    for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); ++it) {
      HcalGenericDetId mygenid(it->rawId());
      if (mygenid.isHcalCastorDetId()) {
        NewPedBunch a;
        HcalCastorDetId chanid(mygenid.rawId());
        a.detid = chanid;
        a.usedflag = false;
        std::string type = "CASTOR";
        for (int i = 0; i != 4; i++) {
          a.cap[i] = 0;
          a.capfc[i] = 0;
          for (int j = 0; j != 4; j++) {
            a.sig[i][j] = 0;
            a.sigfc[i][j] = 0;
            a.prod[i][j] = 0;
            a.prodfc[i][j] = 0;
            a.num[i][j] = 0;
          }
        }
        Bunches.push_back(a);
      }
    }
    firsttime = false;
  }

  std::vector<NewPedBunch>::iterator bunch_it;

  for (CastorDigiCollection::const_iterator j = castor->begin(); j != castor->end(); ++j) {
    const CastorDataFrame digi = (const CastorDataFrame)(*j);
    for (bunch_it = Bunches.begin(); bunch_it != Bunches.end(); ++bunch_it)
      if (bunch_it->detid.rawId() == digi.id().rawId())
        break;
    bunch_it->usedflag = true;
    for (int ts = firstTS; ts != lastTS + 1; ts++) {
      const CastorQIECoder* coder = conditions->getCastorCoder(digi.id().rawId());
      bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
      bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
      double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
      bunch_it->capfc[digi.sample(ts).capid()] += charge1;
      bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] +=
          (digi.sample(ts).adc() * digi.sample(ts).adc());
      bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
      if ((ts + 1 < digi.size()) && (ts + 1 < lastTS)) {
        bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts + 1).capid()] +=
            digi.sample(ts).adc() * digi.sample(ts + 1).adc();
        double charge2 = coder->charge(*shape, digi.sample(ts + 1).adc(), digi.sample(ts + 1).capid());
        bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts + 1).capid()] += charge1 * charge2;
        bunch_it->num[digi.sample(ts).capid()][digi.sample(ts + 1).capid()] += 1;
      }
      if ((ts + 2 < digi.size()) && (ts + 2 < lastTS)) {
        bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts + 2).capid()] +=
            digi.sample(ts).adc() * digi.sample(ts + 2).adc();
        double charge2 = coder->charge(*shape, digi.sample(ts + 2).adc(), digi.sample(ts + 2).capid());
        bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts + 2).capid()] += charge1 * charge2;
        bunch_it->num[digi.sample(ts).capid()][digi.sample(ts + 2).capid()] += 1;
      }
      if ((ts + 3 < digi.size()) && (ts + 3 < lastTS)) {
        bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts + 3).capid()] +=
            digi.sample(ts).adc() * digi.sample(ts + 3).adc();
        double charge2 = coder->charge(*shape, digi.sample(ts + 3).adc(), digi.sample(ts + 3).capid());
        bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts + 3).capid()] += charge1 * charge2;
        bunch_it->num[digi.sample(ts).capid()][digi.sample(ts + 3).capid()] += 1;
      }
    }
  }

  //this is the last brace
}

//define this as a plug-in
DEFINE_FWK_MODULE(CastorPedestalsAnalysis);
