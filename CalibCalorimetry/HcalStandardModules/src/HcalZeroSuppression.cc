// Original Author:  Steven Won
//         Created:  Tues Aug 5 2008
//
#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalZeroSuppression.h"

HcalZeroSuppression::HcalZeroSuppression(const edm::ParameterSet& ps)
{
   verboseflag = ps.getUntrackedParameter<bool>("verbose", false);
   firsttime = true;
}


HcalZeroSuppression::~HcalZeroSuppression()
{
   HcalZSThresholds* ZSItem = new HcalZSThresholds();

   //Calculate ZS Thresholds
   std::cout << "Calculating ZS Thresholds...\n";
   std::vector<ZSBunch>::iterator bunch_it;
   int threshold;
   for(bunch_it=Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
   {
      if(bunch_it->usedflag){

      if(verboseflag) std::cout << "Analyzing channel phi= " << bunch_it->detid.iphi() 
        << " eta = " << bunch_it->detid.ieta() << " depth = " << bunch_it->detid.depth()
        << std::endl;

      bunch_it->cap[0] /= bunch_it->num[0][0];
      bunch_it->cap[1] /= bunch_it->num[1][1];
      bunch_it->cap[2] /= bunch_it->num[2][2];
      bunch_it->cap[3] /= bunch_it->num[3][3];
      bunch_it->sig[0][0] = (bunch_it->prod[0][0]/bunch_it->num[0][0])-(bunch_it->cap[0])*(bunch_it->cap[0]);
      bunch_it->sig[1][1] = (bunch_it->prod[1][1]/bunch_it->num[1][1])-(bunch_it->cap[1])*(bunch_it->cap[1]);
      bunch_it->sig[2][2] = (bunch_it->prod[2][2]/bunch_it->num[2][2])-(bunch_it->cap[2])*(bunch_it->cap[2]);
      bunch_it->sig[3][3] = (bunch_it->prod[3][3]/bunch_it->num[3][3])-(bunch_it->cap[3])*(bunch_it->cap[3]);
/*      bunch_it->sig[0][1] = (bunch_it->prod[0][1])/(bunch_it->num[0][1])-(bunch_it->cap[0]*bunch_it->cap[1]);
      bunch_it->sig[0][2] = (bunch_it->prod[0][2])/(bunch_it->num[0][2])-(bunch_it->cap[0]*bunch_it->cap[2]);
      bunch_it->sig[0][3] = (bunch_it->prod[0][3])/(bunch_it->num[0][3])-(bunch_it->cap[0]*bunch_it->cap[3]);
      bunch_it->sig[1][2] = (bunch_it->prod[1][2])/(bunch_it->num[1][2])-(bunch_it->cap[1]*bunch_it->cap[2]);
      bunch_it->sig[1][3] = (bunch_it->prod[1][3])/(bunch_it->num[1][3])-(bunch_it->cap[1]*bunch_it->cap[3]);
      bunch_it->sig[2][3] = (bunch_it->prod[2][3])/(bunch_it->num[2][3])-(bunch_it->cap[2]*bunch_it->cap[3]);
Off diagonal terms not needed for ZS */

      threshold = (int)floor(
                                (1.5 * sqrt(bunch_it->sig[0][0]) + bunch_it->cap[0] +
                                 1.5 * sqrt(bunch_it->sig[1][1]) + bunch_it->cap[1] +
                                 1.5 * sqrt(bunch_it->sig[2][2]) + bunch_it->cap[2] +
                                 1.5 * sqrt(bunch_it->sig[3][3]) + bunch_it->cap[3]) / 4 * 2); // 2 timeslices used for ZS
      }else{
      threshold = 8;
      }
      const HcalZSThreshold zerosup(bunch_it->detid, threshold);
      ZSItem->addValues(zerosup);
      ZSHist[bunch_it->detid.subdet()-1]->Fill(threshold);
      depthhist[bunch_it->detid.depth()-1]->Fill(bunch_it->detid.ieta(), bunch_it->detid.iphi(), threshold); 
   }

    std::string asciiout = ZSfilename + ".txt";
    std::string xmlout = ZSfilename + ".xml";
    std::string name1 = ZSfilename + "_1dplots.png";
    std::string name2 = ZSfilename + "_2dplots.png";

    std::ofstream outStream8(asciiout.c_str());
    HcalDbASCIIIO::dumpObject (outStream8, (*ZSItem) );

    std::ofstream outStream9(xmlout.c_str());
    HcalCondXML::dumpObject(outStream9, runnum, 1, -1, tag, 1, *ZSItem);

    theFile->cd();
    for(int n = 0; n != 4; n++)
    {
       ZSHist[n]->Write();
       depthhist[n]->Write();
    }

    TStyle *theStyle = new TStyle("style","null");
    theStyle->SetPalette(1,0);
    theStyle->SetCanvasDefH(1200); //Height of canvas
    theStyle->SetCanvasDefW(1600); //Width of canvas
   
    gStyle = theStyle;
   
    TCanvas * c1 = new TCanvas("c1","graph",1);
    c1->Divide(2,2);
    c1->cd(1);
    ZSHist[0]->Draw();
    c1->cd(2);
    ZSHist[1]->Draw();
    c1->cd(3);
    ZSHist[2]->Draw();
    c1->cd(4);
    ZSHist[3]->Draw();
    c1->SaveAs(name1.c_str());   

    theStyle->SetOptStat("n");
    gStyle = theStyle;

    TCanvas * c2 = new TCanvas("c2","graph",1);
    c2->Divide(2,2);
    c2->cd(1);
    depthhist[0]->Draw();
    depthhist[0]->SetDrawOption("colz");
    c2->cd(2);
    depthhist[1]->Draw();
    depthhist[1]->SetDrawOption("colz");
    c2->cd(3);
    depthhist[2]->Draw();
    depthhist[2]->SetDrawOption("colz");
    c2->cd(4);
    depthhist[3]->Draw();
    depthhist[3]->SetDrawOption("colz");
    c2->SaveAs(name2.c_str());


    std::cout << "Writing ROOT file... ";
    theFile->Close();
    std::cout << "ROOT file closed.\n";
}

// ------------ method called to for each event  ------------
void
HcalZeroSuppression::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   edm::Handle<HBHEDigiCollection> hbhe;              e.getByType(hbhe);
   edm::Handle<HODigiCollection> ho;                  e.getByType(ho);
   edm::Handle<HFDigiCollection> hf;                  e.getByType(hf);
   edm::ESHandle<HcalDbService> conditions;
   iSetup.get<HcalDbRecord>().get(conditions);

   if(firsttime)
   {
      runnum = e.id().run();
      std::string runnum_string;
      std::stringstream tempstringout;
      tempstringout << runnum;
      runnum_string = tempstringout.str();
      ROOTfilename = runnum_string + "-ZSThresholds.root";
      ZSfilename = runnum_string + "-ZSThresholds";
      tag = "ZS_INDIV_" + runnum_string;

      firstTS = 0;
      lastTS = 7;

      theFile = new TFile(ROOTfilename.c_str(), "RECREATE");
      theFile->cd();
      // Create sub-directories
      
      ZSHist[0] = new TH1F("ZS Thresholds HB","HB", 21, -.5, 20.5);
      ZSHist[1] = new TH1F("ZS Thresholds HE","HE", 21, -.5, 20.5);
      ZSHist[2] = new TH1F("ZS Thresholds HO","HO", 21, -.5, 20.5);
      ZSHist[3] = new TH1F("ZS Thresholds HF","HF", 21, -.5, 20.5);

      depthhist[0] = new TH2F("Thresholds depth 1","Depth 1",89, -44, 44, 72, .5, 72.5);
      depthhist[1] = new TH2F("Thresholds depth 2","Depth 2",89, -44, 44, 72, .5, 72.5);
      depthhist[2] = new TH2F("Thresholds depth 3","Depth 3",89, -44, 44, 72, .5, 72.5);
      depthhist[3] = new TH2F("Thresholds depth 4","Depth 4",89, -44, 44, 72, .5, 72.5);

      edm::ESHandle<HcalElectronicsMap> refEMap;
      iSetup.get<HcalElectronicsMapRcd>().get(refEMap);
      const HcalElectronicsMap* myRefEMap = refEMap.product();
      std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
      for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
      {     
         HcalGenericDetId mygenid(it->rawId());
         if(mygenid.isHcalDetId())
         {
            ZSBunch a;
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
               for(int j = 0; j != 4; j++)
               {
                  a.sig[i][j] = 0;
                  a.prod[i][j] = 0;
                  a.num[i][j] = 0;
               }
            }
            Bunches.push_back(a);
         }
      }
      firsttime = false;
   }

   std::vector<ZSBunch>::iterator bunch_it;

   for(HBHEDigiCollection::const_iterator j = hbhe->begin(); j != hbhe->end(); j++)
   {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts != lastTS+1; ts++)
      {
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
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
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
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
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += 1;
         }
      }

   }
//this is the last brace
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalZeroSuppression);
