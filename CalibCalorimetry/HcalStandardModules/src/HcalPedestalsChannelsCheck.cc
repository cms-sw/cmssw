#include "CalibCalorimetry/HcalStandardModules/interface/HcalPedestalsChannelsCheck.h"

HcalPedestalsChannelsCheck::HcalPedestalsChannelsCheck(edm::ParameterSet const& ps)
{
   epsilon = .1;
   runnum = ps.getUntrackedParameter<int>("runNumber",0);
   difhist[0] = new TH1F("Difference in pedestals HB","Each CapId (HB)",100,-1.5,1.5);
   difhist[1] = new TH1F("Difference in pedestals HE","Each CapId (HE)",100,-1.5,1.5);
   difhist[2] = new TH1F("Difference in pedestals HO","Each CapId (HO)",100,-1.5,1.5);
   difhist[3] = new TH1F("Difference in pedestals HF","Each CapId (HF)",100,-1.5,1.5);
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
    theStyle->SetPalette(1,0);
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
    etaphi[0]->Draw();
    etaphi[0]->SetDrawOption("colz");
    c2->cd(2);
    etaphi[1]->Draw();
    etaphi[1]->SetDrawOption("colz");
    c2->cd(3);
    etaphi[2]->Draw();
    etaphi[2]->SetDrawOption("colz");
    c2->cd(4);
    etaphi[3]->Draw();
    etaphi[3]->SetDrawOption("colz");
    c2->SaveAs(name2.c_str());

   theFile->Close();
}

void HcalPedestalsChannelsCheck::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
   using namespace edm::eventsetup;
   // get fake pedestals from file ("new pedestals")
   edm::ESHandle<HcalPedestals> newPeds;
   es.get<HcalPedestalsRcd>().get("update",newPeds);
   const HcalPedestals* myNewPeds = newPeds.product();
 
   // get DB pedestals from Frontier/OrcoX ("reference")
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

   // store channels which have changed by more that epsilon
   HcalPedestals *changedchannels = new HcalPedestals();
   for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
      {
         DetId mydetid = *it;
         HcalDetId hocheck(mydetid);
//         if(hocheck.subdet()==3) continue;
         cell = std::find(listNewChan.begin(), listNewChan.end(), mydetid);
         if (cell == listNewChan.end())
            {
               continue;
            }
         else
            { 
               const float* values = (myNewPeds->getValues( mydetid ))->getValues();
               const float* oldvalue = (myRefPeds->getValues( mydetid ))->getValues();
               difhist[hocheck.subdet()-1]->Fill((*oldvalue-*values));
               difhist[hocheck.subdet()-1]->Fill((*oldvalue+1)-(*values+1));
               difhist[hocheck.subdet()-1]->Fill((*oldvalue+2)-(*values+2));
               difhist[hocheck.subdet()-1]->Fill((*oldvalue+3)-(*values+3));
               double avgchange = ( (*oldvalue-*values)
                                   +((*oldvalue+1)-(*values+1)) 
                                   +((*oldvalue+2)-(*values+2))
                                   +((*oldvalue+3)-(*values+3)) ) / 4;
               etaphi[hocheck.depth()-1]->Fill(hocheck.ieta(),hocheck.iphi(),avgchange);
               if(hocheck.subdet()==3) continue;
               if( (fabs(*oldvalue-*values)>epsilon) || (fabs(*(oldvalue+1)-*(values+1))>epsilon) || (fabs(*(oldvalue+2)-*(values+2))>epsilon) || (fabs(*(oldvalue+3)-*(values+3))>epsilon) ){
// 	       throw cms::Exception("DataDoesNotMatch") << "Values differ by more than deltaP";
               std::cout << HcalGenericDetId(mydetid.rawId()) << std::endl;
               failflag = true;
               const HcalPedestal* item = myNewPeds->getValues(mydetid);
               changedchannels->addValues(*item);
             }
             listNewChan.erase(cell);  // fix 25.02.08
           }
        
       } 
     // first get the list of all channels from the update
     std::vector<DetId> listChangedChan = changedchannels->getAllChannels();
 
     HcalPedestals *resultPeds = new HcalPedestals(); //myRefPeds->isADC() );
     for (std::vector<DetId>::iterator it = listRefChan.begin(); it != listRefChan.end(); it++)
       {
         DetId mydetid = *it;
         cell = std::find(listChangedChan.begin(), listChangedChan.end(), mydetid);
         if (cell == listChangedChan.end()) // not present in new list, take old pedestals
           {
             //   bool addValue (DetId fId, const float fValues [4]);
 	    const HcalPedestal* item = myRefPeds->getValues(mydetid);
             std::cout << "o";
             resultPeds->addValues(*item);
           }
         else // present in new list, take new pedestals
           {
             const HcalPedestal* item = myNewPeds->getValues(mydetid);
             std::cout << "n";
             resultPeds->addValues(*item);
             // compare the values of the pedestals for valid channels between update and reference
             listChangedChan.erase(cell);  // fix 25.02.08
           }
       }
 
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
 	//	std::cout << "id = " << mygenid << ", hashed id = " << mygenid.hashedId() << std::endl;
 	if (std::find(listResult.begin(), listResult.end(), mydetid ) == listResult.end())
 	  {
 	    std::cout << "Conditions not found for DetId = " << HcalGenericDetId(it->rawId()) << std::endl;
 	  }
       }
     }
 
     // dump the resulting list of pedestals into a file
     if(failflag)
     {
        std::ofstream outStream3("dump.txt");//outfile.c_str());
        std::cout << "--- Dumping Pedestals - thei merged ones ---" << std::endl;
        HcalDbASCIIIO::dumpObject (outStream3, (*resultPeds) );
     }
  
}


DEFINE_FWK_MODULE(HcalPedestalsChannelsCheck);
