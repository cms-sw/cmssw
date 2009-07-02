// Original Author:  Steven Won
//         Created:  Fri May  2 15:34:43 CEST 2008
// Written to replace the combination of HcalPedestalAnalyzer and HcalPedestalAnalysis 
// This code runs 1000x faster and produces all outputs from a single run
// (ADC, fC in .txt plus an .xml file)
//
#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalsAnalysis.h"

HcalPedestalsAnalysis::HcalPedestalsAnalysis(const edm::ParameterSet& ps)
{
   hiSaveFlag = ps.getUntrackedParameter<bool>("hiSaveFlag", false);
   dumpXML = ps.getUntrackedParameter<bool>("dumpXML", true);
   verboseflag = ps.getUntrackedParameter<bool>("verbose", false);
   firstTS = ps.getUntrackedParameter<int>("firstTS", 0);
   lastTS = ps.getUntrackedParameter<int>("lastTS", 9);   
   firsttime = true;
}


HcalPedestalsAnalysis::~HcalPedestalsAnalysis()
{
   HcalPedestals* rawPedsItem = new HcalPedestals(true);
   HcalPedestalWidths* rawWidthsItem = new HcalPedestalWidths(true);
   HcalPedestals* rawPedsItemfc = new HcalPedestals(false);
   HcalPedestalWidths* rawWidthsItemfc = new HcalPedestalWidths(false);

   //Calculate pedestal constants
   std::cout << "Calculating Pedestal constants...\n";
   std::vector<NewPedBunch>::iterator bunch_it;
   for(bunch_it=Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
   {
      if(bunch_it->usedflag){

      if(verboseflag) std::cout << "Analyzing channel phi= " << bunch_it->detid.iphi() 
        << " eta = " << bunch_it->detid.ieta() << " depth = " << bunch_it->detid.depth()
        << std::endl;
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
      bunch_it->sig[0][0] = (bunch_it->prod[0][0]/bunch_it->num[0][0])-(bunch_it->cap[0])*(bunch_it->cap[0]);
      bunch_it->sig[1][1] = (bunch_it->prod[1][1]/bunch_it->num[1][1])-(bunch_it->cap[1])*(bunch_it->cap[1]);
      bunch_it->sig[2][2] = (bunch_it->prod[2][2]/bunch_it->num[2][2])-(bunch_it->cap[2])*(bunch_it->cap[2]);
      bunch_it->sig[3][3] = (bunch_it->prod[3][3]/bunch_it->num[3][3])-(bunch_it->cap[3])*(bunch_it->cap[3]);
      bunch_it->sig[0][1] = (bunch_it->prod[0][1])/(bunch_it->num[0][1])-(bunch_it->cap[0]*bunch_it->cap[1]);
      bunch_it->sig[0][2] = (bunch_it->prod[0][2])/(bunch_it->num[0][2])-(bunch_it->cap[0]*bunch_it->cap[2]);
      bunch_it->sig[0][3] = (bunch_it->prod[0][3])/(bunch_it->num[0][3])-(bunch_it->cap[0]*bunch_it->cap[3]);
      bunch_it->sig[1][2] = (bunch_it->prod[1][2])/(bunch_it->num[1][2])-(bunch_it->cap[1]*bunch_it->cap[2]);
      bunch_it->sig[1][3] = (bunch_it->prod[1][3])/(bunch_it->num[1][3])-(bunch_it->cap[1]*bunch_it->cap[3]);
      bunch_it->sig[2][3] = (bunch_it->prod[2][3])/(bunch_it->num[2][3])-(bunch_it->cap[2]*bunch_it->cap[3]);

      bunch_it->sigfc[0][0] = (bunch_it->prodfc[0][0]/bunch_it->num[0][0])-(bunch_it->capfc[0])*(bunch_it->capfc[0]);
      bunch_it->sigfc[1][1] = (bunch_it->prodfc[1][1]/bunch_it->num[1][1])-(bunch_it->capfc[1])*(bunch_it->capfc[1]);
      bunch_it->sigfc[2][2] = (bunch_it->prodfc[2][2]/bunch_it->num[2][2])-(bunch_it->capfc[2])*(bunch_it->capfc[2]);
      bunch_it->sigfc[3][3] = (bunch_it->prodfc[3][3]/bunch_it->num[3][3])-(bunch_it->capfc[3])*(bunch_it->capfc[3]);
      bunch_it->sigfc[0][1] = (bunch_it->prodfc[0][1]/(bunch_it->num[0][1]))-(bunch_it->capfc[0]*bunch_it->capfc[1]);
      bunch_it->sigfc[0][2] = (bunch_it->prodfc[0][2]/(bunch_it->num[0][2]))-(bunch_it->capfc[0]*bunch_it->capfc[2]);
      bunch_it->sigfc[0][3] = (bunch_it->prodfc[0][3]/(bunch_it->num[0][3]))-(bunch_it->capfc[0]*bunch_it->capfc[3]);
      bunch_it->sigfc[1][2] = (bunch_it->prodfc[1][2]/(bunch_it->num[1][2]))-(bunch_it->capfc[1]*bunch_it->capfc[2]);
      bunch_it->sigfc[1][3] = (bunch_it->prodfc[1][3]/(bunch_it->num[1][3]))-(bunch_it->capfc[1]*bunch_it->capfc[3]);
      bunch_it->sigfc[2][3] = (bunch_it->prodfc[2][3]/(bunch_it->num[2][3]))-(bunch_it->capfc[2]*bunch_it->capfc[3]);

      if(bunch_it->detid.subdet() == 1){
         for(int i = 0; i != 3; i++){
            HBMeans->Fill(bunch_it->cap[i]);
            HBWidths->Fill(bunch_it->sig[i][i]);
         }
      }
      if(bunch_it->detid.subdet() == 2){
         for(int i = 0; i != 3; i++){
            HEMeans->Fill(bunch_it->cap[i]);
            HEWidths->Fill(bunch_it->sig[i][i]);
         }
      }
      if(bunch_it->detid.subdet() == 3){
         for(int i = 0; i != 3; i++){
            HOMeans->Fill(bunch_it->cap[i]);
            HOWidths->Fill(bunch_it->sig[i][i]);
         }
      }
      if(bunch_it->detid.subdet() == 4){
         for(int i = 0; i != 3; i++){
            HFMeans->Fill(bunch_it->cap[i]);
            HFWidths->Fill(bunch_it->sig[i][i]);
         }
      }

      dephist[bunch_it->detid.depth()-1]->Fill(bunch_it->detid.ieta(),bunch_it->detid.iphi(),
                (bunch_it->cap[0]+bunch_it->cap[1]+bunch_it->cap[2]+bunch_it->cap[3])/4);

      const HcalPedestal item(bunch_it->detid, bunch_it->cap[0], bunch_it->cap[1], bunch_it->cap[2], bunch_it->cap[3],
                              bunch_it->sig[0][0], bunch_it->sig[1][1], bunch_it->sig[2][2], bunch_it->sig[3][3]);
      rawPedsItem->addValues(item);
      HcalPedestalWidth widthsp(bunch_it->detid);
      widthsp.setSigma(0,0,bunch_it->sig[0][0]);
      widthsp.setSigma(0,1,bunch_it->sig[0][1]);
      widthsp.setSigma(0,2,bunch_it->sig[0][2]);
      widthsp.setSigma(0,3,bunch_it->sig[0][3]);
      widthsp.setSigma(1,1,bunch_it->sig[1][1]);
      widthsp.setSigma(1,2,bunch_it->sig[1][2]);
      widthsp.setSigma(1,3,bunch_it->sig[1][3]);
      widthsp.setSigma(2,2,bunch_it->sig[2][2]);
      widthsp.setSigma(2,3,bunch_it->sig[2][3]);
      widthsp.setSigma(3,3,bunch_it->sig[3][3]);
      rawWidthsItem->addValues(widthsp);

      const HcalPedestal itemfc(bunch_it->detid, bunch_it->capfc[0], bunch_it->capfc[1], bunch_it->capfc[2], bunch_it->capfc[3],
                              bunch_it->sigfc[0][0], bunch_it->sigfc[1][1], bunch_it->sigfc[2][2], bunch_it->sigfc[3][3]);
      rawPedsItemfc->addValues(itemfc);
      HcalPedestalWidth widthspfc(bunch_it->detid);
      widthspfc.setSigma(0,0,bunch_it->sigfc[0][0]);
      widthspfc.setSigma(0,1,bunch_it->sigfc[0][1]);
      widthspfc.setSigma(0,2,bunch_it->sigfc[0][2]);
      widthspfc.setSigma(0,3,bunch_it->sigfc[0][3]);
      widthspfc.setSigma(1,1,bunch_it->sigfc[1][1]);
      widthspfc.setSigma(1,2,bunch_it->sigfc[1][2]);
      widthspfc.setSigma(1,3,bunch_it->sigfc[1][3]);
      widthspfc.setSigma(2,2,bunch_it->sigfc[2][2]);
      widthspfc.setSigma(2,3,bunch_it->sigfc[2][3]);
      widthspfc.setSigma(3,3,bunch_it->sigfc[3][3]);
      rawWidthsItemfc->addValues(widthspfc);

      }
   }

    // dump the resulting list of pedestals into a file
    std::ofstream outStream1(pedsADCfilename.c_str());
    HcalDbASCIIIO::dumpObject (outStream1, (*rawPedsItem) );
    std::ofstream outStream2(widthsADCfilename.c_str());
    HcalDbASCIIIO::dumpObject (outStream2, (*rawWidthsItem) );

    std::ofstream outStream3(pedsfCfilename.c_str());
    HcalDbASCIIIO::dumpObject (outStream3, (*rawPedsItemfc) );
    std::ofstream outStream4(widthsfCfilename.c_str());
    HcalDbASCIIIO::dumpObject (outStream4, (*rawWidthsItemfc) );

    if(dumpXML){
       std::ofstream outStream5(XMLfilename.c_str());
       HcalCondXML::dumpObject (outStream5, runnum, runnum, runnum, XMLtag, 1, (*rawPedsItem), (*rawWidthsItem)); 
    }

    if(hiSaveFlag){
       theFile->Write();
    }else{
       theFile->cd();
       theFile->cd("HB");
       HBMeans->Write();
       HBWidths->Write();
       theFile->cd();
       theFile->cd("HF");
       HFMeans->Write();
       HFWidths->Write();
       theFile->cd();
       theFile->cd("HE");
       HEMeans->Write();
       HEWidths->Write();
       theFile->cd();
       theFile->cd("HO");
       HOMeans->Write();
       HOWidths->Write();
    }
    theFile->cd();
    for (int n=0; n!= 4; n++) 
    {
         dephist[n]->Write();
         dephist[n]->SetDrawOption("colz");
         dephist[n]->GetXaxis()->SetTitle("i#eta");
         dephist[n]->GetYaxis()->SetTitle("i#phi");
    }

    std::stringstream tempstringout;
    tempstringout << runnum;
    std::string name1 = tempstringout.str() + "_pedplots_1d.png";
    std::string name2 = tempstringout.str() + "_pedplots_2d.png";

    TStyle *theStyle = new TStyle("style","null");
    theStyle->SetPalette(1,0);
    theStyle->SetCanvasDefH(1200); //Height of canvas
    theStyle->SetCanvasDefW(1600); //Width of canvas

    gStyle = theStyle;

    TCanvas * c1 = new TCanvas("c1","graph",1);
    c1->Divide(2,2);
    c1->cd(1);
    HBMeans->Draw();
    c1->cd(2);
    HEMeans->Draw();
    c1->cd(3);
    HOMeans->Draw();
    c1->cd(4);
    HFMeans->Draw();
    c1->SaveAs(name1.c_str());   

    theStyle->SetOptStat("n");
    gStyle = theStyle;

    TCanvas * c2 = new TCanvas("c2","graph",1);
    c2->Divide(2,2);
    c2->cd(1);
    dephist[0]->Draw();
    dephist[0]->SetDrawOption("colz");
    c2->cd(2);
    dephist[1]->Draw();
    dephist[1]->SetDrawOption("colz");
    c2->cd(3);
    dephist[2]->Draw();
    dephist[2]->SetDrawOption("colz");
    c2->cd(4);
    dephist[3]->Draw();
    dephist[3]->SetDrawOption("colz");
    c2->SaveAs(name2.c_str());

    std::cout << "Writing ROOT file... ";
    theFile->Close();
    std::cout << "ROOT file closed.\n";
}

// ------------ method called to for each event  ------------
void
HcalPedestalsAnalysis::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   edm::Handle<HBHEDigiCollection> hbhe;              e.getByType(hbhe);
   edm::Handle<HODigiCollection> ho;                  e.getByType(ho);
   edm::Handle<HFDigiCollection> hf;                  e.getByType(hf);
   edm::ESHandle<HcalDbService> conditions;
   iSetup.get<HcalDbRecord>().get(conditions);

   const HcalQIEShape* shape = conditions->getHcalShape();

   if(firsttime)
   {
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
      XMLtag = "Hcal_pedestals_" + runnum_string;

      theFile = new TFile(ROOTfilename.c_str(), "RECREATE");
      theFile->cd();
      // Create sub-directories
      theFile->mkdir("HB"); 
      theFile->mkdir("HE");
      theFile->mkdir("HF");
      theFile->mkdir("HO");
      theFile->cd();

      HBMeans = new TH1F("All Ped Means HB","All Ped Means HB", 100, 0, 9);
      HBWidths = new TH1F("All Ped Widths HB","All Ped Widths HB", 100, 0, 3);
      HEMeans = new TH1F("All Ped Means HE","All Ped Means HE", 100, 0, 9);
      HEWidths = new TH1F("All Ped Widths HE","All Ped Widths HE", 100, 0, 3);
      HFMeans = new TH1F("All Ped Means HF","All Ped Means HF", 100, 0, 9);
      HFWidths = new TH1F("All Ped Widths HF","All Ped Widths HF", 100, 0, 3);
      HOMeans = new TH1F("All Ped Means HO","All Ped Means HO", 100, 0, 9);
      HOWidths = new TH1F("All Ped Widths HO","All Ped Widths HO", 100, 0, 3);

      dephist[0] = new TH2F("Pedestals (ADC)","Depth 1",89, -44, 44, 72, .5, 72.5);
      dephist[1] = new TH2F("Pedestals (ADC)","Depth 2",89, -44, 44, 72, .5, 72.5);
      dephist[2] = new TH2F("Pedestals (ADC)","Depth 3",89, -44, 44, 72, .5, 72.5);
      dephist[3] = new TH2F("Pedestals (ADC)","Depth 4",89, -44, 44, 72, .5, 72.5);

      edm::ESHandle<HcalElectronicsMap> refEMap;
      iSetup.get<HcalElectronicsMapRcd>().get(refEMap);
      const HcalElectronicsMap* myRefEMap = refEMap.product();
      std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
      for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
      {     
         HcalGenericDetId mygenid(it->rawId());
         if(mygenid.isHcalDetId())
         {
            NewPedBunch a;
            HcalDetId chanid(mygenid.rawId());
            a.detid = chanid;
            a.usedflag = false;
            string type;
            if(chanid.subdet() == 1) type = "HB";
            if(chanid.subdet() == 2) type = "HE";
            if(chanid.subdet() == 3) type = "HO";
            if(chanid.subdet() == 4) type = "HF";
            for(int i = 0; i != 4; i++)
            {
               a.cap[i] = 0;
               a.capfc[i] = 0;
               for(int j = 0; j != 4; j++)
               {
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

   for(HBHEDigiCollection::const_iterator j = hbhe->begin(); j != hbhe->end(); j++)
   {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts != lastTS+1; ts++)
      {
         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+3).adc(), digi.sample(ts+3).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += 1;
         }
      }
   }

   for(HODigiCollection::const_iterator j = ho->begin(); j != ho->end(); j++)
   {
      const HODataFrame digi = (const HODataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts <= lastTS; ts++)
      {
         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+3).adc(), digi.sample(ts+3).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += 1;
         }
      }
   }

   for(HFDigiCollection::const_iterator j = hf->begin(); j != hf->end(); j++)
   {
      const HFDataFrame digi = (const HFDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts <= lastTS; ts++)
      {
         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+3).adc(), digi.sample(ts+3).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += 1;
         }
      }

/* Once I figure out how to unpack Calib digis they go here
      const HFDataFrame digi = (const HFDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts <= lastTS; ts++)
      {
//         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
//         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
//         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
//         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
//            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
//            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
//            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
//            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
//            double charge2 = coder->charge(*shape, digi.sample(ts+3).adc(), digi.sample(ts+3).capid());
//            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += 1;
         }
*/

   }
//this is the last brace
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalsAnalysis);
