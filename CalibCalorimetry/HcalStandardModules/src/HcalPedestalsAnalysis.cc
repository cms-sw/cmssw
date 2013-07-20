// Original Author:  Steven Won
//         Created:  Fri May  2 15:34:43 CEST 2008
// Written to replace the combination of HcalPedestalAnalyzer and HcalPedestalAnalysis 
// This code runs 1000x faster and produces all outputs from a single run
// (ADC, fC in .txt plus an .xml file)
//
// $Id: HcalPedestalsAnalysis.cc,v 1.25 2012/11/13 03:30:20 dlange Exp $

#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalsAnalysis.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

HcalPedestalsAnalysis::HcalPedestalsAnalysis(const edm::ParameterSet& ps) :
   hbheDigiCollectionTag_(ps.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
   hoDigiCollectionTag_(ps.getParameter<edm::InputTag>("hoDigiCollectionTag")),
   hfDigiCollectionTag_(ps.getParameter<edm::InputTag>("hfDigiCollectionTag")) {

   std::cout << "Code version 10.6\n";
   hiSaveFlag = ps.getUntrackedParameter<bool>("hiSaveFlag", false);
   dumpXML = ps.getUntrackedParameter<bool>("dumpXML", true);
   verboseflag = ps.getUntrackedParameter<bool>("verbose", false);
   firstTS = ps.getUntrackedParameter<int>("firstTS", 0);
   lastTS = ps.getUntrackedParameter<int>("lastTS", 9);   
   firsttime = true;
   ievt = 0;

   rawPedsItem = 0;
   rawWidthsItem = 0;
   rawPedsItemfc = 0;
   rawWidthsItemfc = 0;
}


HcalPedestalsAnalysis::~HcalPedestalsAnalysis()
{}

void HcalPedestalsAnalysis::endJob()
{
//   std::cout << "NEvents " << ievt << std::endl;
   if(ievt < 1000) return;
   std::string ievt_string;
   std::stringstream tempstringout;
   tempstringout << ievt;
   ievt_string = tempstringout.str();
   pedsADCfilename += ievt_string;
   pedsADCfilename += ".txt";
   //Calculate pedestal constants
   std::cout << "Calculating Pedestal constants...\n";
   std::vector<NewPedBunch>::iterator bunch_it;
   for(bunch_it=Bunches.begin(); bunch_it != Bunches.end(); bunch_it++)
   {
      if(bunch_it->usedflag){

      if(verboseflag) std::cout << "Analyzing channel " << bunch_it->detid << std::endl;
      //pedestal constant is the mean
      if(bunch_it->num[0][0]!=0) bunch_it->cap[0] /= bunch_it->num[0][0];
      if(bunch_it->num[1][1]!=0) bunch_it->cap[1] /= bunch_it->num[1][1];
      if(bunch_it->num[2][2]!=0) bunch_it->cap[2] /= bunch_it->num[2][2];
      if(bunch_it->num[3][3]!=0) bunch_it->cap[3] /= bunch_it->num[3][3];
      if(bunch_it->num[0][0]!=0) bunch_it->capfc[0] /= bunch_it->num[0][0];
      if(bunch_it->num[1][1]!=0) bunch_it->capfc[1] /= bunch_it->num[1][1];
      if(bunch_it->num[2][2]!=0) bunch_it->capfc[2] /= bunch_it->num[2][2];
      if(bunch_it->num[3][3]!=0) bunch_it->capfc[3] /= bunch_it->num[3][3];
      bunch_it->sig[0][0] = (bunch_it->prod[0][0]/bunch_it->num[0][0])-(bunch_it->cap[0]*bunch_it->cap[0]);
      bunch_it->sig[0][1] = (bunch_it->prod[0][1]/bunch_it->num[0][1])-(bunch_it->cap[0]*bunch_it->cap[1]);
      bunch_it->sig[0][2] = (bunch_it->prod[0][2]/bunch_it->num[0][2])-(bunch_it->cap[0]*bunch_it->cap[2]);
      bunch_it->sig[0][3] = (bunch_it->prod[0][3]/bunch_it->num[0][3])-(bunch_it->cap[0]*bunch_it->cap[3]);
      bunch_it->sig[1][0] = (bunch_it->prod[1][0]/bunch_it->num[1][0])-(bunch_it->cap[1]*bunch_it->cap[0]);
      bunch_it->sig[1][1] = (bunch_it->prod[1][1]/bunch_it->num[1][1])-(bunch_it->cap[1]*bunch_it->cap[1]);
      bunch_it->sig[1][2] = (bunch_it->prod[1][2]/bunch_it->num[1][2])-(bunch_it->cap[1]*bunch_it->cap[2]);
      bunch_it->sig[1][3] = (bunch_it->prod[1][3]/bunch_it->num[1][3])-(bunch_it->cap[1]*bunch_it->cap[3]);
      bunch_it->sig[2][0] = (bunch_it->prod[2][0]/bunch_it->num[2][0])-(bunch_it->cap[2]*bunch_it->cap[0]);
      bunch_it->sig[2][1] = (bunch_it->prod[2][1]/bunch_it->num[2][1])-(bunch_it->cap[2]*bunch_it->cap[1]);
      bunch_it->sig[2][2] = (bunch_it->prod[2][2]/bunch_it->num[2][2])-(bunch_it->cap[2]*bunch_it->cap[2]);
      bunch_it->sig[2][3] = (bunch_it->prod[2][3]/bunch_it->num[2][3])-(bunch_it->cap[2]*bunch_it->cap[3]);
      bunch_it->sig[3][0] = (bunch_it->prod[3][0]/bunch_it->num[3][0])-(bunch_it->cap[3]*bunch_it->cap[0]);
      bunch_it->sig[3][1] = (bunch_it->prod[3][1]/bunch_it->num[3][1])-(bunch_it->cap[3]*bunch_it->cap[1]);
      bunch_it->sig[3][2] = (bunch_it->prod[3][2]/bunch_it->num[3][2])-(bunch_it->cap[3]*bunch_it->cap[2]);
      bunch_it->sig[3][3] = (bunch_it->prod[3][3]/bunch_it->num[3][3])-(bunch_it->cap[3]*bunch_it->cap[3]);

      bunch_it->sigfc[0][0] = (bunch_it->prodfc[0][0]/bunch_it->num[0][0])-(bunch_it->capfc[0]*bunch_it->capfc[0]);
      bunch_it->sigfc[0][1] = (bunch_it->prodfc[0][1]/bunch_it->num[0][1])-(bunch_it->capfc[0]*bunch_it->capfc[1]);
      bunch_it->sigfc[0][2] = (bunch_it->prodfc[0][2]/bunch_it->num[0][2])-(bunch_it->capfc[0]*bunch_it->capfc[2]);
      bunch_it->sigfc[0][3] = (bunch_it->prodfc[0][3]/bunch_it->num[0][3])-(bunch_it->capfc[0]*bunch_it->capfc[3]);
      bunch_it->sigfc[1][0] = (bunch_it->prodfc[1][0]/bunch_it->num[1][0])-(bunch_it->capfc[1]*bunch_it->capfc[0]);
      bunch_it->sigfc[1][1] = (bunch_it->prodfc[1][1]/bunch_it->num[1][1])-(bunch_it->capfc[1]*bunch_it->capfc[1]);
      bunch_it->sigfc[1][2] = (bunch_it->prodfc[1][2]/bunch_it->num[1][2])-(bunch_it->capfc[1]*bunch_it->capfc[2]);
      bunch_it->sigfc[1][3] = (bunch_it->prodfc[1][3]/bunch_it->num[1][3])-(bunch_it->capfc[1]*bunch_it->capfc[3]);
      bunch_it->sigfc[2][0] = (bunch_it->prodfc[2][0]/bunch_it->num[2][0])-(bunch_it->capfc[2]*bunch_it->capfc[0]);
      bunch_it->sigfc[2][1] = (bunch_it->prodfc[2][1]/bunch_it->num[2][1])-(bunch_it->capfc[2]*bunch_it->capfc[1]);
      bunch_it->sigfc[2][2] = (bunch_it->prodfc[2][2]/bunch_it->num[2][2])-(bunch_it->capfc[2]*bunch_it->capfc[2]);
      bunch_it->sigfc[2][3] = (bunch_it->prodfc[2][3]/bunch_it->num[2][3])-(bunch_it->capfc[2]*bunch_it->capfc[3]);
      bunch_it->sigfc[3][0] = (bunch_it->prodfc[3][0]/bunch_it->num[3][0])-(bunch_it->capfc[3]*bunch_it->capfc[0]);
      bunch_it->sigfc[3][1] = (bunch_it->prodfc[3][1]/bunch_it->num[3][1])-(bunch_it->capfc[3]*bunch_it->capfc[1]);
      bunch_it->sigfc[3][2] = (bunch_it->prodfc[3][2]/bunch_it->num[3][2])-(bunch_it->capfc[3]*bunch_it->capfc[2]);
      bunch_it->sigfc[3][3] = (bunch_it->prodfc[3][3]/bunch_it->num[3][3])-(bunch_it->capfc[3]*bunch_it->capfc[3]);

      if(bunch_it->detid.subdet() == 1){
         for(int i = 0; i != 4; i++){
            HBMeans->Fill(bunch_it->cap[i]);
            HBWidths->Fill(bunch_it->sig[i][i]);
         }
      }
      if(bunch_it->detid.subdet() == 2){
         for(int i = 0; i != 4; i++){
            HEMeans->Fill(bunch_it->cap[i]);
            HEWidths->Fill(bunch_it->sig[i][i]);
         }
      }
      if(bunch_it->detid.subdet() == 3){
         for(int i = 0; i != 4; i++){
            HOMeans->Fill(bunch_it->cap[i]);
            HOWidths->Fill(bunch_it->sig[i][i]);
         }
      }
      if(bunch_it->detid.subdet() == 4){
         for(int i = 0; i != 4; i++){
            HFMeans->Fill(bunch_it->cap[i]);
            HFWidths->Fill(bunch_it->sig[i][i]);
         }
      }

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
//    std::ofstream outStream2(widthsADCfilename.c_str());
//    HcalDbASCIIIO::dumpObject (outStream2, (*rawWidthsItem) );
//
//    std::ofstream outStream3(pedsfCfilename.c_str());
//    HcalDbASCIIIO::dumpObject (outStream3, (*rawPedsItemfc) );
//    std::ofstream outStream4(widthsfCfilename.c_str());
//    HcalDbASCIIIO::dumpObject (outStream4, (*rawWidthsItemfc) );
//
//    if(dumpXML){
//       std::ofstream outStream5(XMLfilename.c_str());
//       HcalDbXml::dumpObject (outStream5, runnum, 0, 2147483647, XMLtag, 1, (*rawPedsItem), (*rawWidthsItem)); 
//    }
//
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

    std::cout << "Writing ROOT file... ";
    theFile->Close();
    std::cout << "ROOT file closed.\n";
    
    delete rawPedsItem;
    delete rawWidthsItem;
    delete rawPedsItemfc;
    delete rawWidthsItemfc;   
}

// ------------ method called to for each event  ------------
void
HcalPedestalsAnalysis::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   edm::Handle<HBHEDigiCollection> hbhe;              e.getByLabel(hbheDigiCollectionTag_, hbhe);
   edm::Handle<HODigiCollection> ho;                  e.getByLabel(hoDigiCollectionTag_, ho);
   edm::Handle<HFDigiCollection> hf;                  e.getByLabel(hfDigiCollectionTag_, hf);
   edm::ESHandle<HcalDbService> conditions;
   iSetup.get<HcalDbRecord>().get(conditions);


   if(firsttime)
   {

     edm::ESHandle<HcalTopology> topology;
     iSetup.get<IdealGeometryRecord>().get( topology );
     theTopology=new HcalTopology(*topology);

     rawPedsItem = new HcalPedestals(theTopology,true);
     rawWidthsItem = new HcalPedestalWidths(theTopology,true);
     rawPedsItemfc = new HcalPedestals(theTopology,false);
     rawWidthsItemfc = new HcalPedestalWidths(theTopology,false);

      runnum = e.id().run();
      std::string runnum_string;
      std::stringstream tempstringout;
      tempstringout << runnum;
      runnum_string = tempstringout.str();
      ROOTfilename = runnum_string + "-peds_ADC.root";
      pedsADCfilename = runnum_string + "-peds_ADC_";
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
	 const HcalQIEShape* shape = conditions->getHcalShape(coder);
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
	 const HcalQIEShape* shape = conditions->getHcalShape(coder);
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
	 const HcalQIEShape* shape = conditions->getHcalShape(coder);
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

//this is the last brace
ievt++;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalsAnalysis);
