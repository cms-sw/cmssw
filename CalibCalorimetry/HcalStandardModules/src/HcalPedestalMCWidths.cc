// Original Author:  Steven Won
//         Created:  Fri May  2 15:34:43 CEST 2008
// Written to replace the combination of HcalPedestalAnalyzer and HcalPedestalAnalysis 
// This code runs 1000x faster and produces all outputs from a single run
// (ADC, fC in .txt plus an .xml file)
//
#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalMCWidths.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/Utilities/interface/isFinite.h"

HcalPedestalMCWidths::HcalPedestalMCWidths(const edm::ParameterSet& ps) :
   hbheDigiCollectionTag_(ps.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
   hoDigiCollectionTag_(ps.getParameter<edm::InputTag>("hoDigiCollectionTag")),
   hfDigiCollectionTag_(ps.getParameter<edm::InputTag>("hfDigiCollectionTag")) {

   firsttime = true;
   histflag = ps.getUntrackedParameter<bool>("saveHists",true);
}


HcalPedestalMCWidths::~HcalPedestalMCWidths()
{
   HcalCovarianceMatrices * rawCovItem = new HcalCovarianceMatrices(theTopology);
   std::ofstream outfile(widthsfilename.c_str());
   std::vector<MCWidthsBunch>::iterator bunch_it;
   for(bunch_it=Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
   {
      if(!bunch_it->usedflag) continue;

      HcalCovarianceMatrix item(bunch_it->detid.rawId());
      for(int i = 0; i != 4; i++){
         for(int j = 0; j != 10; j++){
            for(int k = 0; k != 10; k++){            
               bunch_it->sig[i][j][k] = (bunch_it->prod[i][j][k]/bunch_it->num[i][j][k]);//-(bunch_it->cap[i]*bunch_it->cap[(i+j)%4]);
               if(!edm::isNotFinite(bunch_it->sig[i][j][k]))
               {
                  item.setValue(i,j,k,bunch_it->sig[i][j][k]);
               }else{
                  item.setValue(i,j,k,0.0);
               }
            }
         }
      }

      rawCovItem->addValues(item);

   if(histflag)
   {
     if(bunch_it->detid.subdet() == 1){
         for(int i = 0; i != 4; i++){
            for(int j = 0; j != 10; j++){
               for(int k = j; k != 10; k++)
               if(!edm::isNotFinite(bunch_it->sig[i][j][k])) HBMeans[j]->Fill(k-j,bunch_it->sig[i][j][k]);
            }
         }
      }
      if(bunch_it->detid.subdet() == 2){
         for(int i = 0; i != 4; i++){
            for(int j = 0; j != 10; j++){
               for(int k = j; k != 10; k++)
               if(!edm::isNotFinite(bunch_it->sig[i][j][k])) HEMeans[j]->Fill(k-j,bunch_it->sig[i][j][k]);
            }
         }
      }
      if(bunch_it->detid.subdet() == 3){
         for(int i = 0; i != 4; i++){
            for(int j = 0; j != 10; j++){
               for(int k = j; k != 10; k++)
               if(!edm::isNotFinite(bunch_it->sig[i][j][k])) HFMeans[j]->Fill(k-j,bunch_it->sig[i][j][k]);
            }
         }
      }
      if(bunch_it->detid.subdet() == 4){
         for(int i = 0; i != 4; i++){
            for(int j = 0; j != 10; j++){
               for(int k = j; k != 10; k++)
               if(!edm::isNotFinite(bunch_it->sig[i][j][k])) HOMeans[j]->Fill(k-j,bunch_it->sig[i][j][k]);
            }
         }
      }

         
    }
    }
    HcalDbASCIIIO::dumpObject (outfile, (*rawCovItem) );    

    if(histflag)
    {
    theFile->cd();
    std::cout << "Writing histograms..." << std::endl;
    for(int i = 0; i != 10; i++){
       HBMeans[i]->Write();
       HFMeans[i]->Write();
       HEMeans[i]->Write();
       HOMeans[i]->Write();
    }
    }
    theFile->Close();
    std::cout << "ROOT file closed.\n";
}

// ------------ method called to for each event  ------------
void
HcalPedestalMCWidths::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   edm::Handle<HBHEDigiCollection> hbhe;              e.getByLabel(hbheDigiCollectionTag_, hbhe);
   edm::Handle<HODigiCollection> ho;                  e.getByLabel(hoDigiCollectionTag_, ho);
   edm::Handle<HFDigiCollection> hf;                  e.getByLabel(hfDigiCollectionTag_, hf);
   edm::ESHandle<HcalDbService> conditions;
   iSetup.get<HcalDbRecord>().get(conditions);

   edm::ESHandle<HcalPedestals> refPeds;
   iSetup.get<HcalPedestalsRcd>().get(refPeds);
   const HcalPedestals* myRefPeds = refPeds.product();

   if(firsttime)
   {
      theTopology=new HcalTopology(*(myRefPeds->topo()));
      runnum = e.id().run();
      std::string runnum_string;
      std::stringstream tempstringout;
      tempstringout << runnum;
      runnum_string = tempstringout.str();
      widthsfilename = runnum_string + "-MCwidths.txt";
      std::string rootfilename = runnum_string + "-MCwidths.root";
      theFile = new TFile(rootfilename.c_str(), "RECREATE");

      for(int i = 0; i!= 10; i++)
      {
         tempstringout.str("");
         tempstringout << i;
         std::string histname3 = tempstringout.str();
         std::string histname2 = "Mean covariance TS ";
         std::string histname1 = "HB ";
         std::string histname = histname1 + histname2 + histname3;
         HBMeans[i] = new TProfile(histname.c_str(),histname1.c_str(), 10, -0.50, 9.5);
         histname1 = "HE ";
         histname = histname1 + histname2 + histname3;
         HEMeans[i] = new TProfile(histname.c_str(),histname1.c_str(), 10, -0.50, 9.5);
         histname1 = "HF ";
         histname = histname1 + histname2 + histname3;
         HFMeans[i] = new TProfile(histname.c_str(),histname1.c_str(), 10, -0.50, 9.5);
         histname1 = "HO ";
         histname = histname1 + histname2 + histname3;
         HOMeans[i] = new TProfile(histname.c_str(),histname1.c_str(), 10, -0.50, 9.5);
      }

      edm::ESHandle<HcalElectronicsMap> refEMap;
      iSetup.get<HcalElectronicsMapRcd>().get(refEMap);
      const HcalElectronicsMap* myRefEMap = refEMap.product();
      std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
      for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
      {     
         HcalGenericDetId mygenid(it->rawId());
         if(mygenid.isHcalDetId())
         {
            MCWidthsBunch a;
            HcalDetId chanid(mygenid.rawId());
            a.detid = chanid;
            a.usedflag = false;
            for(int i = 0; i != 4; i++)
            {
               for(int j = 0; j != 10; j++){
               for(int k = 0; k != 10; k++)
               {
                  a.sig[i][j][k] = 0;
                  a.prod[i][j][k] = 0;
                  a.num[i][j][k] = 0;
               }}
            }
            Bunches.push_back(a);
         }
      }
      firsttime = false;
   }

   std::vector<MCWidthsBunch>::iterator bunch_it;

   for(HBHEDigiCollection::const_iterator j = hbhe->begin(); j != hbhe->end(); j++)
   {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      DetId searchid(digi.id().rawId());
      const HcalPedestal* peds = (myRefPeds->getValues( searchid ));
      bunch_it->usedflag = true;
      int firstcapid = digi.sample(0).capid();
      for(int ts = 0; ts != 10; ts++)
      {
         for(int j = ts; j != 10; j++){
         bunch_it->num[firstcapid][ts][j] += 1;
         bunch_it->prod[firstcapid][ts][j] += 
                                           ((digi.sample(j).adc()-peds->getValue(firstcapid))
                                           *(digi.sample(ts).adc()-peds->getValue(digi.sample(ts).capid())));
         }
      }
   }

   for(HODigiCollection::const_iterator j = ho->begin(); j != ho->end(); j++)
   {
      const HODataFrame digi = (const HODataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      DetId searchid(digi.id().rawId());
      const HcalPedestal* peds = (myRefPeds->getValues( searchid ));
      bunch_it->usedflag = true;
      int firstcapid = digi.sample(0).capid();
      for(int ts = 0; ts != 10; ts++)
      {
         for(int j = ts; j != 10; j++){
         bunch_it->num[firstcapid][ts][j] += 1;
         bunch_it->prod[firstcapid][ts][j] += 
                                           ((digi.sample(j).adc()-peds->getValue(firstcapid))
                                           *(digi.sample(ts).adc()-peds->getValue(digi.sample(ts).capid())));
         }
      }
   }

   for(HFDigiCollection::const_iterator j = hf->begin(); j != hf->end(); j++)
   {
      const HFDataFrame digi = (const HFDataFrame)(*j);
      for(bunch_it = Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      DetId searchid(digi.id().rawId());
      const HcalPedestal* peds = (myRefPeds->getValues( searchid ));
      bunch_it->usedflag = true;
      int firstcapid = digi.sample(0).capid();
      for(int ts = 0; ts != 10; ts++)
      {
         for(int j = ts; j != 10; j++){
         bunch_it->num[firstcapid][ts][j] += 1;
         bunch_it->prod[firstcapid][ts][j] += 
                                           ((digi.sample(j).adc()-peds->getValue(firstcapid))
                                           *(digi.sample(ts).adc()-peds->getValue(digi.sample(ts).capid())));
         }
      }

   }
//this is the last brace
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalMCWidths);
