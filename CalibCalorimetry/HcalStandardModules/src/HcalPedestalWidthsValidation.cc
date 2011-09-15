// Original Author:  Steven Won
//         Created:  Fri May  2 15:34:43 CEST 2008
// Written to replace the combination of HcalPedestalAnalyzer and HcalPedestalAnalysis 
// This code runs 1000x faster and produces all outputs from a single run
// (ADC, fC in .txt plus an .xml file)
//
#include <memory>
#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalWidthsValidation.h"

HcalPedestalWidthsValidation::HcalPedestalWidthsValidation(const edm::ParameterSet& ps)
{
   firstTS = ps.getUntrackedParameter<int>("firstTS", 0);
   lastTS = ps.getUntrackedParameter<int>("lastTS", 9);   
   firsttime = true;

   rawPedsItem = new HcalPedestals();
   rawWidthsItem = new HcalPedestalWidths();
   rawPedsItemfc = new HcalPedestals();
   rawWidthsItemfc = new HcalPedestalWidths();
}


HcalPedestalWidthsValidation::~HcalPedestalWidthsValidation()
{
   //Calculate pedestal constants
   std::cout << "Calculating Pedestal constants...\n";
   std::vector<NewPedBunchVal>::iterator bunch_it;
   for(bunch_it=BunchVales.begin(); bunch_it != BunchVales.end(); bunch_it++)
   {
      if(bunch_it->usedflag){

      bunch_it->cap[0] /= bunch_it->num[0][0];
      bunch_it->cap[1] /= bunch_it->num[1][1];
      bunch_it->cap[2] /= bunch_it->num[2][2];
      bunch_it->cap[3] /= bunch_it->num[3][3];
      bunch_it->capfc[0] /= bunch_it->num[0][0];
      bunch_it->capfc[1] /= bunch_it->num[1][1];
      bunch_it->capfc[2] /= bunch_it->num[2][2];
      bunch_it->capfc[3] /= bunch_it->num[3][3];
      //widths are the covariance matrix--NOT assumed symmetric here
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

      bunch_it->sig[1][0] = (bunch_it->prod[1][0]/bunch_it->num[1][0])-(bunch_it->cap[1])*(bunch_it->cap[0]);
      bunch_it->sig[2][0] = (bunch_it->prod[2][0]/bunch_it->num[2][0])-(bunch_it->cap[2])*(bunch_it->cap[0]);
      bunch_it->sig[2][1] = (bunch_it->prod[2][1]/bunch_it->num[2][1])-(bunch_it->cap[2])*(bunch_it->cap[1]);
      bunch_it->sig[3][0] = (bunch_it->prod[3][0]/bunch_it->num[3][0])-(bunch_it->cap[3])*(bunch_it->cap[0]);
      bunch_it->sig[3][1] = (bunch_it->prod[3][1]/bunch_it->num[3][1])-(bunch_it->cap[3])*(bunch_it->cap[1]);
      bunch_it->sig[3][2] = (bunch_it->prod[3][2]/bunch_it->num[3][2])-(bunch_it->cap[3])*(bunch_it->cap[2]);
      
      

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

      bunch_it->sigfc[1][0] = (bunch_it->prodfc[1][0]/(bunch_it->num[1][0]))-(bunch_it->capfc[1]*bunch_it->capfc[0]);
      bunch_it->sigfc[2][0] = (bunch_it->prodfc[2][0]/(bunch_it->num[2][0]))-(bunch_it->capfc[2]*bunch_it->capfc[0]);
      bunch_it->sigfc[2][1] = (bunch_it->prodfc[2][1]/(bunch_it->num[2][1]))-(bunch_it->capfc[2]*bunch_it->capfc[1]);
      bunch_it->sigfc[3][0] = (bunch_it->prodfc[3][0]/(bunch_it->num[3][0]))-(bunch_it->capfc[3]*bunch_it->capfc[0]);
      bunch_it->sigfc[3][1] = (bunch_it->prodfc[3][1]/(bunch_it->num[3][1]))-(bunch_it->capfc[3]*bunch_it->capfc[1]);
      bunch_it->sigfc[3][2] = (bunch_it->prodfc[3][2]/(bunch_it->num[3][2]))-(bunch_it->capfc[3]*bunch_it->capfc[2]);

      for(int i = 0; i != 4; i++){
         for(int j = 0; j != 4; j++){
	    bunch_it->covarhistADC->Fill(i,j,bunch_it->sig[i][j]);
	    bunch_it->covarhistfC->Fill(i,j,bunch_it->sigfc[i][j]);
	 }
      }

      if(bunch_it->detid.subdet() == 1){
         theFile->cd("HB");
         bunch_it->covarhistADC->Write();
         bunch_it->covarhistfC->Write();
         for(int i = 0; i != 4; i++){
            bunch_it->hist[i]->Write();
            HBMeans->Fill(bunch_it->cap[i]);
            HBWidths->Fill(sqrt(bunch_it->sig[i][i]));
         }
      }
      if(bunch_it->detid.subdet() == 2){
         theFile->cd("HE");
         bunch_it->covarhistADC->Write();
         bunch_it->covarhistfC->Write();
         for(int i = 0; i != 4; i++){
            bunch_it->hist[i]->Write();
            HEMeans->Fill(bunch_it->cap[i]);
            HEWidths->Fill(sqrt(bunch_it->sig[i][i]));
         }
      }
      if(bunch_it->detid.subdet() == 3){
         theFile->cd("HO");
         bunch_it->covarhistADC->Write();
         bunch_it->covarhistfC->Write();
         for(int i = 0; i != 4; i++){
            bunch_it->hist[i]->Write();
            HOMeans->Fill(bunch_it->cap[i]);
            HOWidths->Fill(sqrt(bunch_it->sig[i][i]));
         }
      }
      if(bunch_it->detid.subdet() == 4){
         theFile->cd("HF");
         bunch_it->covarhistADC->Write();
         bunch_it->covarhistfC->Write();
         for(int i = 0; i != 4; i++){
            bunch_it->hist[i]->Write();
            HFMeans->Fill(bunch_it->cap[i]);
            HFWidths->Fill(sqrt(bunch_it->sig[i][i]));
         }
      }

      theFile->cd();

      const HcalPedestal item(bunch_it->detid, bunch_it->cap[0], bunch_it->cap[1], bunch_it->cap[2], bunch_it->cap[3]);
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

      const HcalPedestal itemfc(bunch_it->detid, bunch_it->capfc[0], bunch_it->capfc[1],
                                   bunch_it->capfc[2], bunch_it->capfc[3]);
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
    
    delete rawPedsItem;
    delete rawWidthsItem;
    delete rawPedsItemfc;
    delete rawWidthsItemfc;   
}

// ------------ method called to for each event  ------------
void
HcalPedestalWidthsValidation::analyze(const edm::Event& e, const edm::EventSetup& iSetup)
{

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
      ROOTfilename = runnum_string + "-val_peds_ADC.root";
      pedsADCfilename = runnum_string + "-val_peds_ADC.txt";
      pedsfCfilename = runnum_string + "-val_peds_fC.txt";
      widthsADCfilename = runnum_string + "-val_widths_ADC.txt";
      widthsfCfilename = runnum_string + "-val_widths_fC.txt";

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
            NewPedBunchVal a;
            HcalDetId chanid(mygenid.rawId());
            a.detid = chanid;
            a.usedflag = false;
	    std::ostringstream s1;
            s1 << mygenid;
            std::string histname = s1.str();
	    histname += " Covariance Matrix";
	    a.covarhistADC = new TH2F(histname.c_str(), histname.c_str(), 4, -.5, 3.5, 4, -.5, 3.5);
	    a.covarhistADC->SetOption("TEXT");
	    a.covarhistADC->SetStats(0);
	    histname += " fC";
	    a.covarhistfC = new TH2F(histname.c_str(), histname.c_str(), 4, -.5, 3.5, 4, -.5, 3.5);
	    a.covarhistfC->SetOption("TEXT");
	    a.covarhistfC->SetStats(0);
            for(int i = 0; i != 4; i++)
            {
               a.cap[i] = 0;
               a.capfc[i] = 0;
	       std::ostringstream s3;
               s3 << mygenid;
	       std::string histname = s3.str() + " Cap ";
	       std::ostringstream s4;
	       s4 << i;
	       histname += s4.str();
	       std::ostringstream s5;
	       s5 << std::hex << (mygenid.rawId()) << std::dec;
	       std::string histnamedetid = s5.str();
               a.hist[i] = new TH1F(histname.c_str(), histnamedetid.c_str(), 16, -.5, 15.5);
               for(int j = 0; j != 4; j++)
               {
                  a.sig[i][j] = 0;
                  a.sigfc[i][j] = 0;
                  a.prod[i][j] = 0;
                  a.prodfc[i][j] = 0;
                  a.num[i][j] = 0;
               }
            }
            BunchVales.push_back(a);
//	    theFile->cd();
         }
      }
      firsttime = false;
   }

   std::vector<NewPedBunchVal>::iterator bunch_it;

   for(HBHEDigiCollection::const_iterator j = hbhe->begin(); j != hbhe->end(); j++)
   {
      const HBHEDataFrame digi = (const HBHEDataFrame)(*j);
      for(bunch_it = BunchVales.begin(); bunch_it != BunchVales.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts != lastTS+1; ts++)
      {
         if(digi.sample(ts).adc() > 15) continue;
         bunch_it->hist[digi.sample(ts).capid()]->Fill(digi.sample(ts).adc());
         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            if(digi.sample(ts+1).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            if(digi.sample(ts+2).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            if(digi.sample(ts+3).adc() > 15) continue;
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
      for(bunch_it = BunchVales.begin(); bunch_it != BunchVales.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts <= lastTS; ts++)
      {
         if(digi.sample(ts).adc() > 15) continue;
         bunch_it->hist[digi.sample(ts).capid()]->Fill(digi.sample(ts).adc());
         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            if(digi.sample(ts+1).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            if(digi.sample(ts+2).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            if(digi.sample(ts+3).adc() > 15) continue;
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
      for(bunch_it = BunchVales.begin(); bunch_it != BunchVales.end(); bunch_it++)
         if(bunch_it->detid.rawId() == digi.id().rawId()) break;
      bunch_it->usedflag = true;
      for(int ts = firstTS; ts <= lastTS; ts++)
      {
         if(digi.sample(ts).adc() > 15) continue;
         bunch_it->hist[digi.sample(ts).capid()]->Fill(digi.sample(ts).adc());
         const HcalQIECoder* coder = conditions->getHcalCoder(digi.id().rawId());
         bunch_it->num[digi.sample(ts).capid()][digi.sample(ts).capid()] += 1;
         bunch_it->cap[digi.sample(ts).capid()] += digi.sample(ts).adc();
         double charge1 = coder->charge(*shape, digi.sample(ts).adc(), digi.sample(ts).capid());
         bunch_it->capfc[digi.sample(ts).capid()] += charge1;
         bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts).capid()] += (digi.sample(ts).adc() * digi.sample(ts).adc());
         bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts).capid()] += charge1 * charge1;
         if((ts+1 < digi.size()) && (ts+1 < lastTS)){
            if(digi.sample(ts+1).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += digi.sample(ts).adc()*digi.sample(ts+1).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+1).adc(), digi.sample(ts+1).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+1).capid()] += 1;
         }
         if((ts+2 < digi.size()) && (ts+2 < lastTS)){
            if(digi.sample(ts+2).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += digi.sample(ts).adc()*digi.sample(ts+2).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+2).adc(), digi.sample(ts+2).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+2).capid()] += 1;
         }
         if((ts+3 < digi.size()) && (ts+3 < lastTS)){
            if(digi.sample(ts+3).adc() > 15) continue;
            bunch_it->prod[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += digi.sample(ts).adc()*digi.sample(ts+3).adc();
            double charge2 = coder->charge(*shape, digi.sample(ts+3).adc(), digi.sample(ts+3).capid());
            bunch_it->prodfc[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += charge1*charge2;
            bunch_it->num[digi.sample(ts).capid()][digi.sample(ts+3).capid()] += 1;
         }
      }
   }

//this is the last brace
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalPedestalWidthsValidation);
