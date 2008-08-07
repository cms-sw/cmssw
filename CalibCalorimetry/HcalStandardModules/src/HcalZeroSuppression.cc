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
   std::vector<NewPedBunch>::iterator bunch_it;
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

      int threshold = (int)floor(
                                (3 * sqrt(bunch_it->sig[0][0]) + bunch_it->cap[0] +
                                 3 * sqrt(bunch_it->sig[1][1]) + bunch_it->cap[1] +
                                 3 * sqrt(bunch_it->sig[2][2]) + bunch_it->cap[2] +
                                 3 * sqrt(bunch_it->sig[3][3]) + bunch_it->cap[3]) / 4);
      const HcalZSThreshold zerosup(bunch_it->detid, threshold);
      ZSItem->addValues(zerosup);
      }
   }

    std::ofstream outStream8(ZSfilename.c_str());
    HcalDbASCIIIO::dumpObject (outStream8, (*ZSItem) );

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
      ZSfilename = runnum_string + "-ZSThresholds.txt";

      firstTS = 0;
      lastTS = 1;

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

   std::vector<NewPedBunch>::iterator bunch_it;

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
