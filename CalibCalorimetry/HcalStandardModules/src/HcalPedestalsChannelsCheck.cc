// $Id: HcalPedestalsChannelsCheck.cc,v 1.7 2011/03/16 17:56:26 andrey Exp $

#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalsChannelsCheck.h"
#include <ios>

HcalPedestalsChannelsCheck::HcalPedestalsChannelsCheck(edm::ParameterSet const& ps)
{
  epsilon = ps.getUntrackedParameter<double>("epsilon",0.2);
  //epsilon = 0.002;
  runnum = ps.getUntrackedParameter<int>("runNumber",0);
  difhist[0] = new TH1F("Difference in pedestals HB","Each CapId (HB)",100,0,1.5);
  difhist[1] = new TH1F("Difference in pedestals HE","Each CapId (HE)",100,0,1.5);
  difhist[2] = new TH1F("Difference in pedestals HO","Each CapId (HO)",100,0,1.5);
  difhist[3] = new TH1F("Difference in pedestals HF","Each CapId (HF)",100,0,1.5);
  etaphi[0] = new TH2F("Average difference per channel d1","Depth 1",89, -44, 44, 72, .5, 72.5);
  etaphi[1] = new TH2F("Average difference per channel d2","Depth 2",89, -44, 44, 72, .5, 72.5);
  etaphi[2] = new TH2F("Average difference per channel d3","Depth 3",89, -44, 44, 72, .5, 72.5);
  etaphi[3] = new TH2F("Average difference per channel d4","Depth 4",89, -44, 44, 72, .5, 72.5);
}

HcalPedestalsChannelsCheck::~HcalPedestalsChannelsCheck()
{
    std::stringstream tempstringout;
    tempstringout << runnum;
    std::string name1 = tempstringout.str() + "_peddifplots_1d.png";
    std::string name2 = tempstringout.str() + "_peddifplots_2d.png";
    std::string name3 = tempstringout.str() + "_peddifs.root";
    TFile * theFile = new TFile(name3.c_str(),"RECREATE");
    for(int n = 0; n != 4; n++) {etaphi[n]->Write(); difhist[n]->Write();}

    TStyle *theStyle = new TStyle("style","null");
//  theStyle->SetPalette(1,0); Replaced with new palette to set cells with no entry = white
//  big thanks to Jeff Temple!
    const Int_t NRGBs= 6;
    const Int_t NCont = 255; //??  Maybe smaller number?
    Double_t stops[NRGBs]={0.00, 0.49,0.499,0.501,0.51,1.00};
    Double_t blue[NRGBs]={1.0, 0.0, 1.0, 1.0,0.0,0.0};
    Double_t green[NRGBs]={0.0,1.0,1.0,1.0,1.0,0.0};
    Double_t red[NRGBs]={0.0,0.0,1.0,1.0,0.0,1.0};
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
   // You need to do the TColor command twice, or else the existing color scheme doesn't get replaced with your new table
    theStyle->SetNumberContours(NCont);

    theStyle->SetCanvasDefH(1200); //Height of canvas
    theStyle->SetCanvasDefW(1600); //Width of canvas
    gStyle = theStyle;

    TCanvas * c1 = new TCanvas("c1","graph",1);
    c1->Divide(2,2);
    c1->cd(1);
    difhist[0]->Draw();
    c1->cd(2);
    difhist[1]->Draw();
    c1->cd(3);
    difhist[2]->Draw();
    c1->cd(4);
    difhist[3]->Draw();
    c1->SaveAs(name1.c_str());   

    theStyle->SetOptStat("n");
    gStyle = theStyle;

    TCanvas * c2 = new TCanvas("c2","graph",1);
    c2->Divide(2,2);
    c2->cd(1);
    etaphi[0]->SetMinimum(-1);
    etaphi[0]->SetMaximum(1);
    etaphi[0]->Draw();
    etaphi[0]->SetDrawOption("colz");
    c2->cd(2);
    etaphi[1]->SetMinimum(-1);
    etaphi[1]->SetMaximum(1);
    etaphi[1]->Draw();
    etaphi[1]->SetDrawOption("colz");
    c2->cd(3);
    etaphi[2]->SetMinimum(-1);
    etaphi[2]->SetMaximum(1);
    etaphi[2]->Draw();
    etaphi[2]->SetDrawOption("colz");
    c2->cd(4);
    etaphi[3]->SetMinimum(-1);
    etaphi[3]->SetMaximum(1);
    etaphi[3]->Draw();
    etaphi[3]->SetDrawOption("colz");
    c2->SaveAs(name2.c_str());

   theFile->Close();
}

void HcalPedestalsChannelsCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
   using namespace edm::eventsetup;
   // get pedestals from file ("new pedestals")
   edm::ESHandle<HcalPedestals> newPeds;
   es.get<HcalPedestalsRcd>().get("update",newPeds);
   const HcalPedestals* myNewPeds = newPeds.product();

   // get older pedestals ("reference") 
   edm::ESHandle<HcalPedestals> refPeds;
   es.get<HcalPedestalsRcd>().get("reference",refPeds);
   const HcalPedestals* myRefPeds = refPeds.product();
 
   // get e-map from reference
   edm::ESHandle<HcalElectronicsMap> refEMap;
   es.get<HcalElectronicsMapRcd>().get("reference",refEMap);
   const HcalElectronicsMap* myRefEMap = refEMap.product();
 
   std::vector<DetId> listNewChan = myNewPeds->getAllChannels();
   std::vector<DetId> listRefChan = myRefPeds->getAllChannels();
   std::vector<DetId>::iterator cell;
   bool failflag = false;

   //std::cout<<"Size of listNewChan: "<<listNewChan.size()<<std::endl;
   //std::cout<<"Size of listRefChan: "<<listRefChan.size()<<std::endl;

   if(myRefPeds->isADC() != myNewPeds->isADC()) throw cms::Exception("Peds not in same units!");

   // store channels which have changed by more that epsilon

   HcalPedestals *changedchannels = new HcalPedestals(myRefPeds->isADC());
   for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
         DetId mydetid = *it;
 	 HcalGenericDetId mygenid(it->rawId());
         if(!mygenid.isHcalDetId()) continue;
         HcalDetId hocheck(mydetid);
         double thresh = epsilon;
         if(hocheck.subdet()==3)
	 {
	    if((hocheck.iphi() >= 47) && (hocheck.iphi() <= 58))
	    {
	       if((hocheck.ieta() >= 5) && (hocheck.ieta() <= 10)) thresh = epsilon * 2;
	    }
	    if((hocheck.iphi() >= 59) && (hocheck.iphi() <= 70))
	    {
	       if((hocheck.ieta() >= 11) && (hocheck.ieta() <= 15)) thresh = epsilon *2;
	    }
         }

         cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
         if (cell == listNewChan.end())
            {
               continue;
            }
         else
            { 
               const float* values = (myNewPeds->getValues( mydetid ))->getValues();
               const float* oldvalue = (myRefPeds->getValues( mydetid ))->getValues();
               double avgchange = 0.25 * ( (*oldvalue-*values)
                                   +(*(oldvalue+1)-*(values+1)) 
                                   +(*(oldvalue+2)-*(values+2))
                                   +(*(oldvalue+3)-*(values+3)) );
	       /* debug
	       if(hocheck.iphi()==2 && hocheck.ieta()>-9 && hocheck.ieta()<-5 )
		 {
		   std::cout<<std::dec << HcalGenericDetId(mydetid.rawId())<<" "<<*oldvalue<<"  "<<*values<<std::endl;
		   std::cout<<std::dec << HcalGenericDetId(mydetid.rawId())<<" "<<*(oldvalue+1)<<"  "<<*(values+1)<<std::endl;
		 }
	       */

               if(fabs(avgchange) > thresh){
               etaphi[hocheck.depth()-1]->Fill(hocheck.ieta(),hocheck.iphi(),avgchange);
               difhist[hocheck.subdet()-1]->Fill(fabs(avgchange));
//               if(hocheck.subdet()==3) continue;
// 	       throw cms::Exception("DataDoesNotMatch") << "Values differ by more than deltaP";
               std::cout << std::dec << HcalGenericDetId(mydetid.rawId()) <<"  "<< std::hex  << std::uppercase << mydetid.rawId() <<"\t  has changed by "<< avgchange * -1 << " \t  threshold: "<<thresh<< std::endl;
               failflag = true;
               const HcalPedestal* item = myNewPeds->getValues(mydetid);
               changedchannels->addValues(*item);
             }
             listNewChan.erase(cell);  // fix 25.02.08
           }
        
       } 
     // first get the list of all channels from the update
     std::vector<DetId> listChangedChan = changedchannels->getAllChannels();
 
     HcalPedestals *resultPeds = new HcalPedestals( myRefPeds->isADC() );
     for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
       {
         DetId mydetid = *it;
         cell = std::find(listChangedChan.begin(), listChangedChan.end(), mydetid);
         if (cell == listChangedChan.end()) // not present in new list, take old pedestals
           {
             //   bool addValue (DetId fId, const float fValues [4]);
 	    const HcalPedestal* item = myRefPeds->getValues(mydetid);
//             std::cout << "o";
             resultPeds->addValues(*item);
           }
         else // present in new list, take new pedestals
           {
             const HcalPedestal* item = myNewPeds->getValues(mydetid);
//             std::cout << "n";
             resultPeds->addValues(*item);
             // compare the values of the pedestals for valid channels between update and reference
             listChangedChan.erase(cell);  // fix 25.02.08
           }
       }

     std::cout << std::endl;
 
     std::vector<DetId> listResult = resultPeds->getAllChannels();
     // get the e-map list of channels
     std::vector<HcalGenericDetId> listEMap = myRefEMap->allPrecisionId();
     // look up if emap channels are all present in pedestals, if not then cerr
     if(1)//checkemapflag)
     {
       for (std::vector<HcalGenericDetId>::const_iterator it = listEMap.begin(); it != listEMap.end(); it++)
       {
   	  DetId mydetid = DetId(it->rawId());
 	HcalGenericDetId mygenid(it->rawId());
            if(!mygenid.isHcalDetId()) continue;
 	//	std::cout << "id = " << mygenid << ", hashed id = " << mygenid.hashedId() << std::endl;
 	if (std::find(listResult.begin(), listResult.end(), mydetid ) == listResult.end())
 	  {
//            if(!mygenid.isHcalId()) continue;
 	    std::cout << "Conditions not found for DetId = " << HcalGenericDetId(it->rawId()) << std::endl;
 	  }
       }
     }
 
     // dump the resulting list of pedestals into a file
     if(failflag)
     {
        std::ofstream outStream3("dump.txt");//outfile.c_str());
//        std::cout << "--- Dumping Pedestals - the merged ones ---" << std::endl;
        HcalDbASCIIIO::dumpObject (outStream3, (*resultPeds) );
     }
  
}


DEFINE_FWK_MODULE(HcalPedestalsChannelsCheck);
