#ifndef HcalObjRepresent_h
#define HcalObjRepresent_h

#include "CondFormats/HcalObjects/interface/HcalGains.h"

//#include "CondCore/Utilities/interface/PayLoadInspector.h"
//#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"
#include "CondFormats/HcalObjects/interface/HcalDetIdRelationship.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include "TH1F.h"
#include "TH2F.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TROOT.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TColor.h"
#include "TLine.h"
#include "TLatex.h"
#include "TProfile.h"
#include "TPaveLabel.h"

//functions for correct representation of data in summary and plot
namespace HcalObjRepresent{

       //used to produce all display objects for payload inspector
       template<class Items, class Item>
       class HcalDataContainer {
       public : 

           HcalDataContainer(std::shared_ptr<Items> payload, unsigned int run) : payload_(payload), run_(run) {
                PlotMode_ = "Map";
           }

           virtual ~HcalDataContainer(){};
           // For easier channel mapping
           typedef std::tuple<int, int, int> Coord;
           typedef std::map< Coord, Item > tHcalValCont;
           // mapping of pair of subdetector name (e.g, "HE") and depth number (e.g. 3) to Histogram of data for that subdetector/depth pair
           typedef std::map< std::pair< std::string, int >, TH2F* > DepthMap;






           ///////////////// public Get functions  /////////////////
           unsigned int GetRun() {return run_;}

           std::string GetTopoMode() {return TopoMode_;}

           std::map<std::string,int> GetSubDetDepths() {return subDetDepths_;}

           DepthMap GetDepths() {
             fillValConts();
             //std::cout << "Got depths with run number = " << std::to_string(GetRun()) << std::endl;
             return depths_;
           }


           ///////////////// setting object fields /////////////////

           // Fills a tHcalValCont for each subdetector, setting Topology Mode along the way
           void fillValConts(){
               if(!depths_.empty()) return;
               int iphi,ieta,depth;
               HcalDetId detId;
           
               std::string subDetName;
               std::vector<Item> itemsVec;
               std::pair<std::string,int> depthKey;
               const char* histLabel;
               for(std::pair< std::string, std::vector<Item> > cont : (*payload_).getAllContainers()){
                   subDetName = std::get<0>(cont);
                   itemsVec = std::get<1>(cont);
           
                   auto valContainer = getContFromString(subDetName);
                   
                   for(Item item : itemsVec){
                     detId = HcalDetId(item.rawId());
                     iphi = detId.iphi();
                     ieta = detId.ieta();
                     depth = detId.depth();  
                     Coord coord = std::make_tuple(depth,ieta,iphi);
                     //Add hist if it's not there, AND if it is one we care about: HO,HB,HE,HF, AND it's not an empty entry (depth = 0 in HO)
                     if(subDetName[0] == 'H' && depth != 0){
                     valContainer->insert( std::pair<std::tuple<int,int,int>,Item>(coord,item));


                     depthKey = std::make_pair(subDetName, depth);

                     auto depthVal = depths_.find(depthKey);
                       if(depthVal == depths_.end()){
                         histLabel = ("run" + std::to_string(run_) + "_" + subDetName + "_d" + std::to_string(depth)).c_str();
                         depths_.insert(std::make_pair(std::make_pair(subDetName, depth), new TH2F(histLabel,histLabel, 83, -42.5, 41.5, 71, 0.5, 71.5)));
                       }    
                       depths_[depthKey]->Fill(ieta, iphi, getValue(&item));
                     }
                   }
               }

               //Still need to know which hists to take when done; decide now
               setTopoModeFromValConts();
           }
    
           ////NOTE to be implemented in PayloadInspector classes
           virtual float getValue(Item* item){throw cms::Exception ("Value definition not found") << "getValue definition not found for " << payload_->myname();};



           //Gets Hcal Object at given coordinate
           //Currently unused but remains as a potentially useful function
           Item* getItemFromValCont(std::string subDetName, int depth, int ieta, int iphi, bool throwOnFail) {
           
             Item* cell = nullptr;
             Coord coord = std::make_tuple(depth,ieta,iphi);
             tHcalValCont* valContainer = getContFromString(subDetName);
           
             auto it = valContainer->find(coord);
             if (it != valContainer->end()) cell = &it->second;
            
             if ((!cell)) {
               if (throwOnFail) {
                 throw cms::Exception ("Conditions not found") 
           	<< "Unavailable Conditions of type " << payload_->myname() << " for cell " << subDetName << " (" << depth << ", " << ieta << ", " << iphi << ")";
               } 
             } 
             return cell;
           
           }


    



           /////////////////// Building Graphics //////////////////////

           //TODO: remove zero entries from doing divide and subtract

           // To generate Ratios of two IOVs        
           void Divide(HcalDataContainer* dataCont2) {

             //TODO: Do something like looping over this and that depths setting every empty bin (which I think means content=0) to -999. Then replacing the divide call with another manual loop that sets all bins with content -999 to 0

             PlotMode_ = "Ratio";
             DepthMap::iterator depth1;
             std::pair<std::string,int> key;
             DepthMap myDepths = this->GetDepths();
             DepthMap theirDepths = dataCont2->GetDepths();
             for(depth1 = myDepths.begin(); depth1 != myDepths.end(); depth1++){
               key = depth1->first;
               if(theirDepths.count(key) != 0) {
                 myDepths.at(key)->Divide((const TH1*)theirDepths.at(key));
               } else {
                 throw cms::Exception ("Unaligned Conditions") << "trying to plot conditions for " << payload_->myname() << "; found value for " << std::get<0>(key) << " depth " << std::to_string(std::get<1>(key)) << " in run " << GetRun() << " but not in run " << dataCont2->GetRun();
               }
             }
           }



           // To generate Diffs of two IOVs
           void Subtract(HcalDataContainer* dataCont2) {
             PlotMode_ = "Diff";
             DepthMap::iterator depth1;
             std::pair<std::string,int> key;
             DepthMap myDepths = this->GetDepths();
             DepthMap theirDepths = dataCont2->GetDepths();
             for(depth1 = myDepths.begin(); depth1 != myDepths.end(); depth1++){
               key = depth1->first;
               if(theirDepths.count(key) != 0) {
                 myDepths.at(key)->Add(myDepths.at(key), theirDepths.at(key), 1, -1);
               } else {
                 throw cms::Exception ("Unaligned Conditions") << "trying to plot conditions for " << payload_->myname() << "; found value for " << std::get<0>(key) << " depth " << std::to_string(std::get<1>(key)) << " in run " << GetRun() << " but not in run " << dataCont2->GetRun();
               }
             }
           }


           // wisely determines what range to set histogram axis to
           std::pair< float, float > GetRange(TH1* hist) {

               if(PlotMode_ == "Ratio") {
                 float amp;
                 Double_t adjustMin = 1; Double_t tempMin;
                 int nBinsX = hist->GetXaxis()->GetNbins(); int nBinsY = hist->GetYaxis()->GetNbins();
                 for(int i = 0; i < nBinsX; i++) {
                   for(int j = 0; j < nBinsY; j++) {
                     tempMin = hist->GetBinContent(i,j);
                     if((tempMin != 0) && (tempMin < adjustMin)) adjustMin = tempMin;
                   }
                 }
                 amp = std::max((1 - adjustMin),(hist->GetMaximum() - 1) );
                 //amp = std::max((1 - hist->GetMinimum()),(hist->GetMaximum() - 1) );
                 return std::make_pair( 1 - amp, 1 + amp);
               } 
               else if(PlotMode_ == "Diff") {
                 float amp;
                 amp = std::max((0 - hist->GetMinimum()),hist->GetMaximum() );
                 return std::make_pair( (-1 * amp), amp);
               } 
               else {
               Double_t adjustMin = 10000; Double_t tempMin;
                 int nBinsX = hist->GetXaxis()->GetNbins(); int nBinsY = hist->GetYaxis()->GetNbins();
                 for(int i = 0; i < nBinsX; i++) {
                   for(int j = 0; j < nBinsY; j++) {
                     tempMin = hist->GetBinContent(i,j);
                     if((tempMin != 0) && (tempMin < adjustMin)) adjustMin = tempMin;
                   }
                 }
               return std::make_pair(((adjustMin==10000) ? hist->GetMinimum() : adjustMin), hist->GetMaximum());
               }
           }

           // set style
           void initGraphics() {

             gStyle->SetOptStat(0);
             gStyle->SetPalette(1);
             gStyle->SetOptFit(0);
             gStyle->SetLabelFont(42);
             gStyle->SetLabelFont(42);
             gStyle->SetTitleFont(42);
             gStyle->SetTitleFont(42);
             gStyle->SetMarkerSize(0);
             gStyle->SetTitleOffset(1.3,"Y");
             gStyle->SetTitleOffset(1.0,"X");
             gStyle->SetNdivisions(510);
             gStyle->SetStatH(0.11);
             gStyle->SetStatW(0.33);
             gStyle->SetTitleW(0.4);
             gStyle->SetTitleX(0.13);
             gStyle->SetPadTickX(1);
             gStyle->SetPadTickY(1);
          }


          TH1D* GetProjection(TH2F* hist, std::string plotType, const char* newName, std::string subDetName, int depth) {

            //TODO: Also want average for standard projection of 2DHist (not ratio or diff)?
            //if (PlotMode_ != "Ratio") return (plotType=="EtaProfile") ? ((TH2F*)(hist->Clone("temp")))->ProjectionX(newName) : ((TH2F*)(hist->Clone("temp")))->ProjectionY(newName);

            //TH1D* projection = ((TH2F*)(depths_[std::make_pair(subDetName,depth)]->Clone("temp")))->ProjectionX(newName);



            int xBins = (plotType=="EtaProfile") ? 83 : 71;
            int etaMin = -42, etaMax = 42, phiMin = 1, phiMax = 72;
            int xMin = (plotType=="EtaProfile") ? etaMin : phiMin;
            int xMax = (plotType=="EtaProfile") ? etaMax : phiMax;
            int otherMin = (plotType=="EtaProfile") ? phiMin : etaMin;
            int otherMax = (plotType=="EtaProfile") ? phiMax : etaMax;
            TH1D* retHist = new TH1D(newName,newName, xBins,xMin,xMax);
            int numChannels;
            Double_t sumVal;
            Double_t channelVal;
            int ieta, iphi;
            int bin = 0;
            for(int i = xMin; i <= xMax; i++ && bin++) { 
              numChannels = 0; sumVal = 0;
              for(int j = otherMin; j <= otherMax; j++) {
                ieta = (plotType=="EtaProfile") ? i : j; ieta += 42;
                iphi = (plotType=="EtaProfile") ? j : i; iphi += -1;
                channelVal = hist->GetBinContent(ieta,iphi);
                //std::cout << "(ieta, iphi)    :       (" << std::to_string(ieta) << ", " << std::to_string(iphi) << ")" << std::endl;
                if(channelVal != 0 ) {
                  sumVal += channelVal; 
                  numChannels++;
                }
              }
              //if(sumVal !=0) projection->SetBinContent(i, sumVal/((Double_t)numChannels));//retHist->Fill(i,sumVal/((Double_t)numChannels));
              if(sumVal !=0) retHist->Fill(i,sumVal/((Double_t)numChannels));
            }
            return retHist; 
            //return projection;
          }


          // fills a canvas with given subdetector information, plotting all depths
          void FillCanv(TCanvas* canvas, std::string subDetName, int startDepth=1, int startCanv=1, std::string plotForm= "2DHist") {


             const char* newName;
             std::pair< float, float> range;
             int padNum;
             int maxDepth = (subDetName=="HO")?4:subDetDepths_[subDetName];
             TH1D* projection;
             TLatex label;
             for(int i = startDepth; i <= maxDepth ; i++){
               //skip if data not obtained; TODO: Maybe add text on plot saying data not found?
               if(depths_.count(std::make_pair(subDetName,i)) == 0) {
                 return; 
               }

               padNum = i+startCanv-1;
               if(subDetName=="HO") padNum = padNum - 3;
               canvas->cd(padNum);
               canvas->GetPad(padNum)->SetGridx(1);
               canvas->GetPad(padNum)->SetGridy(1);



               if(plotForm == "2DHist"){
                 canvas->GetPad(padNum)->SetRightMargin(0.13);
                 range = GetRange( depths_[std::make_pair(subDetName,i)]);
                 depths_[std::make_pair(subDetName,i)]->Draw("colz");
                 depths_[std::make_pair(subDetName,i)]->SetContour(100);
                 depths_[std::make_pair(subDetName,i)]->GetXaxis()->SetTitle("ieta");
                 depths_[std::make_pair(subDetName,i)]->GetYaxis()->SetTitle("iphi");
                 depths_[std::make_pair(subDetName,i)]->GetXaxis()->CenterTitle();
                 depths_[std::make_pair(subDetName,i)]->GetYaxis()->CenterTitle();
                 depths_[std::make_pair(subDetName,i)]->GetZaxis()->SetRangeUser(std::get<0>(range), std::get<1>(range));
                 depths_[std::make_pair(subDetName,i)]->GetYaxis()->SetTitleSize(0.06);
                 depths_[std::make_pair(subDetName,i)]->GetYaxis()->SetTitleOffset(0.80);
                 depths_[std::make_pair(subDetName,i)]->GetXaxis()->SetTitleSize(0.06);
                 depths_[std::make_pair(subDetName,i)]->GetXaxis()->SetTitleOffset(0.80);
                 depths_[std::make_pair(subDetName,i)]->GetYaxis()->SetLabelSize(0.055);
                 depths_[std::make_pair(subDetName,i)]->GetXaxis()->SetLabelSize(0.055);
               } else {
                 canvas->GetPad(padNum)->SetLeftMargin(0.152);
                 canvas->GetPad(padNum)->SetRightMargin(0.02);
                 //gStyle->SetTitleOffset(1.6,"Y");
                 newName = ("run_" + std::to_string(run_) + "_" + subDetName + "_d" + std::to_string(i) + "_" + (plotForm=="EtaProfile" ? "ieta" : "iphi")).c_str();
                 //projection = ((TH2F*)(depths_[std::make_pair(subDetName,i)]->Clone("temp")))->ProjectionX(newName);
                 projection = GetProjection(depths_[std::make_pair(subDetName,i)],plotForm, newName, subDetName, i);
                 range = GetRange(projection);
                 projection->Draw("hist");
                 projection->GetXaxis()->SetTitle((plotForm=="EtaProfile" ? "ieta" : "iphi"));
                 projection->GetXaxis()->CenterTitle();
                 projection->GetYaxis()->SetTitle((payload_->myname() + " " + (PlotMode_ == "Map" ? "" : PlotMode_) + " " + GetUnit(payload_->myname())).c_str());
	         label.SetNDC();
	         label.SetTextAlign(26);
	         label.SetTextSize(0.05);
	         label.DrawLatex(0.3, 0.95, ("run_" + std::to_string(run_) + "_" + subDetName + "_d" + std::to_string(i)).c_str());
                 projection->GetYaxis()->CenterTitle();
                 projection->GetXaxis()->SetTitleSize(0.06);
                 projection->GetYaxis()->SetTitleSize(0.06);
                 projection->GetXaxis()->SetTitleOffset(0.80);
                 projection->GetXaxis()->SetLabelSize(0.055);
                 projection->GetYaxis()->SetTitleOffset(1.34);
                 projection->GetYaxis()->SetLabelSize(0.055);
               }

             }

 
          }


           //// functions called in Payload Inspector classes to import final canvases to be plotted

           // profile = "EtaProfile" || "PhiProfile"
           TCanvas* getCanvasAll(std::string profile="2DHist") {
             fillValConts();
             initGraphics();
             TCanvas *HAll = new TCanvas("HAll", "HAll", 1680, (GetTopoMode()=="2015/2016")?1680:2500);
             HAll->Divide(3, (GetTopoMode()=="2015/2016")?3:6, 0.02,0.01);
             FillCanv(HAll,"HB",1,1,profile);
             FillCanv(HAll,"HO",4,3,profile);
             FillCanv(HAll,"HF",1,4,profile);
             FillCanv(HAll,"HE",1,(GetTopoMode()=="2015/2016")?7:10,profile);
             return HAll;
           }           
           
           TCanvas* getCanvasHF() {
           
             fillValConts();
             initGraphics();
             TCanvas *HF = new TCanvas("HF", "HF", 1600, 1000);
             HF->Divide(3, 2,0.02,0.01);
             FillCanv(HF,"HF");
             return HF;
           }           
           TCanvas* getCanvasHE() {
           
             fillValConts();
             initGraphics();
             TCanvas *HE = new TCanvas("HE", "HE", 1680, 1680);
             HE->Divide(3, 3,0.02,0.01);
             FillCanv(HE,"HE");
             return HE;
           }           
           TCanvas* getCanvasHBHO() {
             fillValConts();
             initGraphics();
             TCanvas *HBHO = new TCanvas("HBHO", "HBHO", 1680, 1680);
             HBHO->Divide(3, 3,0.02,0.01);
             FillCanv(HBHO,"HB");
             FillCanv(HBHO,"HO",4,3);
             FillCanv(HBHO,"HB",1,4,"EtaProfile");
             FillCanv(HBHO,"HO",4,6,"EtaProfile");
             FillCanv(HBHO,"HB",1,7,"PhiProfile");
             FillCanv(HBHO,"HO",4,9,"PhiProfile");
             return HBHO;
           }           


           std::string GetUnit(std::string type) { 
             std::string unit =  units_[type];
             if(unit.empty()) return "";
             else return "("+unit+")";
           }
           

       private:
           DepthMap depths_;
           std::shared_ptr<Items> payload_;
           unsigned int run_;
           std::string TopoMode_;
           // "Map", "Ratio", or "Diff"
           std::string PlotMode_;
       
           tHcalValCont HBvalContainer;
           tHcalValCont HEvalContainer;
           tHcalValCont HOvalContainer;
           tHcalValCont HFvalContainer;
           tHcalValCont HTvalContainer;
           tHcalValCont ZDCvalContainer;
           tHcalValCont CALIBvalContainer;
           tHcalValCont CASTORvalContainer;
           std::map<std::string,int> subDetDepths_;
           std::map<std::string, std::string> units_ = {
             { "HcalPedestals", "ADC" },
             { "HcalGains",  ""},//dimensionless TODO: verify
             { "HcalL1TriggerObjects", "" },//dimensionless TODO: Verify
             { "HcalPedestalWidths", "ADC" },
             { "HcalRespCorrs", "" },//dimensionless TODO: verify
             { "Dark Current", "" },
             { "fcByPE", "" },
             { "crossTalk", "" },
             { "parLin", "" }
           };

           tHcalValCont* getContFromString(std::string subDetString) {
            
             if(subDetString == "HB") return &HBvalContainer;
             else if(subDetString == "HE") return &HEvalContainer;
             else if(subDetString == "HF") return &HFvalContainer;
             else if(subDetString == "HO") return &HOvalContainer;
             else if(subDetString == "HT") return &HTvalContainer;
             else if(subDetString == "CALIB") return &CALIBvalContainer;
             else if(subDetString == "CASTOR") return &CASTORvalContainer;
             //else return &ZDCvalContainer;
             else if(subDetString == "ZDC_EM" || subDetString == "ZDC" || subDetString == "ZDC_HAD" || subDetString == "ZDC_LUM") return &ZDCvalContainer;
             else throw cms::Exception ("subDetString "+subDetString+" not found in Item");
             
           }

           void setTopoModeFromValConts(bool throwOnFail=false) {
                

             // Check HEP17 alternate channel for 2017, just by checking if the 7th depth is there, or if HF has 4 depths
             if(depths_.count(std::make_pair("HF",4))!=0 || depths_.count(std::make_pair("HE",7))!=0) TopoMode_ = "2017";
             // Check endcap depth unique to 2018
             else if(HEvalContainer.count(std::make_tuple(7,-26,63))!=0) TopoMode_ = "2018";
             // if not 2017 or 2018, 2015 and 2016 are the same
             else TopoMode_ = "2015/2016"; 
             //NOTE: HO's one depth is labeled depth 4
             if(TopoMode_ == "2018" || TopoMode_ == "2017") {
               subDetDepths_.insert(std::pair<std::string,int>("HB",2));
               subDetDepths_.insert(std::pair<std::string,int>("HE",7));
               subDetDepths_.insert(std::pair<std::string,int>("HF",4));
               subDetDepths_.insert(std::pair<std::string,int>("HO",1));
             } else if(TopoMode_ == "2015/2016") {
               subDetDepths_.insert(std::pair<std::string,int>("HB",2));
               subDetDepths_.insert(std::pair<std::string,int>("HE",3));
               subDetDepths_.insert(std::pair<std::string,int>("HF",2));
               subDetDepths_.insert(std::pair<std::string,int>("HO",1));
             }
    
          }
       };






        void drawTable(int nbRows, int nbColumns){
              TLine* l = new TLine;
              l->SetLineWidth(1);
              for (int i = 1; i < nbRows; i++) {
                double y = (double) i;
                l = new TLine(0., y, nbColumns, y);
                l->Draw();
              }
      
              for (int i = 1; i < nbColumns; i++) {
                double x = (double) i;
                double y = (double) nbRows;
                l = new TLine(x, 0., x, y);
                l->Draw();
              }
        }




        std::string SciNotatStr(float num){

            // Create an output string stream
            std::ostringstream streamObj2;

            if(num==-1) return "NOT FOUND";
 
            //Add double to stream
            streamObj2 << num;
            // Get string from output string stream
            std::string strObj2 = streamObj2.str();
 
            return strObj2;
        }




	inline std::string IntToBinary(unsigned int number) {
		std::stringstream ss;
		unsigned int mask = 1<<31;
		for (unsigned short int i = 0; i < 32; ++i){
			//if (!(i % 4))
			//	ss << "_";
			if (mask & number)
				ss << "1";
			else 
				ss << "0";
			mask = mask >> 1;
		}
		return ss.str();
	}


	const bool isBitSet(unsigned int bitnumber, unsigned int status)
	{ 
		unsigned int statadd = 0x1<<(bitnumber);
		return (status&statadd)?(true):(false);
	}


	std::string getBitsSummary(uint32_t bits, std::string  statusBitArray[], short unsigned int bitMap[]  ){
		std::stringstream ss;
		for (unsigned int i = 0; i < 9; ++i){
			if (isBitSet(bitMap[i], bits)){
				ss << "[" <<bitMap[i]<< "]" << statusBitArray[bitMap[i]] << "; ";
			}
		}
		ss << std::endl;
		return ss.str();
	}


	//functions for making plot:
	void setBinLabels(std::vector<TH2F> &depth)
	{
		// Set labels for all depth histograms
		for (unsigned int i=0;i<depth.size();++i)
		{
			depth[i].SetXTitle("i#eta");
			depth[i].SetYTitle("i#phi");
		}

		std::stringstream label;

		// set label on every other bin
		for (int i=-41;i<=-29;i=i+2)
		{
			label<<i;
			depth[0].GetXaxis()->SetBinLabel(i+42,label.str().c_str());
			depth[1].GetXaxis()->SetBinLabel(i+42,label.str().c_str());
			label.str("");
		}
		depth[0].GetXaxis()->SetBinLabel(14,"-29HE");
		depth[1].GetXaxis()->SetBinLabel(14,"-29HE");

		// offset by one for HE
		for (int i=-27;i<=27;i=i+2)
		{
			label<<i;
			depth[0].GetXaxis()->SetBinLabel(i+43,label.str().c_str());
			label.str("");
		}
		depth[0].GetXaxis()->SetBinLabel(72,"29HE");
		for (int i=29;i<=41;i=i+2)
		{
			label<<i;
			depth[0].GetXaxis()->SetBinLabel(i+44,label.str().c_str());
			label.str("");
		}
		for (int i=16;i<=28;i=i+2)
		{
			label<<i-43;
			depth[1].GetXaxis()->SetBinLabel(i,label.str().c_str());
			label.str("");
		}
		depth[1].GetXaxis()->SetBinLabel(29,"NULL");
		for (int i=15;i<=27;i=i+2)
		{
			label<<i;
			depth[1].GetXaxis()->SetBinLabel(i+15,label.str().c_str());
			label.str("");
		}

		depth[1].GetXaxis()->SetBinLabel(44,"29HE");
		for (int i=29;i<=41;i=i+2)
		{
			label<<i;
			depth[1].GetXaxis()->SetBinLabel(i+16,label.str().c_str());
			label.str("");
		}

		// HE depth 3 labels;
		depth[2].GetXaxis()->SetBinLabel(1,"-28");
		depth[2].GetXaxis()->SetBinLabel(2,"-27");
		depth[2].GetXaxis()->SetBinLabel(3,"Null");
		depth[2].GetXaxis()->SetBinLabel(4,"-16");
		depth[2].GetXaxis()->SetBinLabel(5,"Null");
		depth[2].GetXaxis()->SetBinLabel(6,"16");
		depth[2].GetXaxis()->SetBinLabel(7,"Null");
		depth[2].GetXaxis()->SetBinLabel(8,"27");
		depth[2].GetXaxis()->SetBinLabel(9,"28");
	}
	// Now define functions that can be used in conjunction with EtaPhi histograms

	// This arrays the eta binning for depth 2 histograms (with a gap between -15 -> +15)
	const int binmapd2[]={-42,-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,
		-29,-28,-27,-26,-25,-24,-23,-22,-21,-20,-19,-18,-17,
		-16,-15,-9999, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
		30,31,32,33,34,35,36,37,38,39,40,41,42};

	// This stores eta binning in depth 3 (where HE is only present at a few ieta values)

	const int binmapd3[]={-28,-27,-9999,-16,-9999,16,-9999,27,28};

	inline int CalcEtaBin(int subdet, int ieta, int depth)
	{
		// This takes the eta value from a subdetector and return an eta counter value as used by eta-phi array
		// (ieta=-41 corresponds to bin 0, +41 to bin 85 -- there are two offsets to deal with the overlap at |ieta|=29).
		// For HO, ieta = -15 corresponds to bin 0, and ieta=15 is bin 30
		// For HE depth 3, things are more complicated, but feeding the ieta value will give back the corresponding counter eta value

		// The CalcEtaBin value is the value as used within our array counters, and thus starts at 0.
		// If you are using it with getBinContent or setBinContent, you will need to add +1 to the result of this function

		int etabin=-9999; // default invalid value

		if (depth==1)
		{
			// Depth 1 is fairly straightforward -- just shift HF-, HF+ by -/+1
			etabin=ieta+42;
			if (subdet==HcalForward)
			{
				ieta < 0 ? etabin-- : etabin++;
			}
		}

		else if (depth==2)
		{
			// Depth 2 is more complicated, given that there are no cells in the range |ieta|<15
			if (ieta<-14)
			{
				etabin=ieta+42;
				if (subdet==HcalForward) etabin--;
			}
			else if (ieta>14)
			{
				etabin=ieta+14;
				if (subdet==HcalForward) etabin++;
			}

		}
		// HO is also straightforward; a simple offset to the ieta value is applied
		else if (subdet==HcalOuter && abs(ieta)<16)
			etabin=ieta+15;
		else if (subdet==HcalEndcap)
		{
			// HE depth 3 has spotty coverage; hard-code the bin response
			if (depth==3)
			{
				if (ieta==-28) etabin=0;
				else if (ieta==-27) etabin=1;
				else if (ieta==-16) etabin=3;
				else if (ieta==16)  etabin=5;
				else if (ieta==27)  etabin=7;
				else if (ieta==28)  etabin=8;
			}
		}
		return etabin;
	}

	inline int CalcIeta(int subdet, int eta, int depth)
	{
		// This function returns the 'true' ieta value given subdet, eta, and depth
		// Here 'eta' is the index from our arrays (it starts at 0);
		// remember that histogram bins start with bin 1, so there's an offset of 1
		// to consider if using getBinContent(eta,phi)

		// eta runs from 0...X  (X depends on depth)
		int ieta=-9999; // default value is nonsensical
		if (subdet==HcalBarrel)
		{
			if (depth==1) 
			{
				ieta=eta-42;
				if (ieta==0) return -9999;
				return ieta;
			}
			else if (depth==2)
			{
				ieta=binmapd2[eta];
				if (ieta==0) return -9999;
				if (ieta==17 || ieta == -17) 
					return -9999; // no depth 2 cells at |ieta| = 17
				return ieta;
			}
			else
				return -9999; // non-physical value
		}
		else if (subdet==HcalForward)
		{
			if (depth==1)
			{
				ieta=eta-42;
				if (eta<13) ieta++;
				else if (eta>71) ieta--;
				else return -9999; // if outside forward range, return dummy
				return ieta;
			}
			else if (depth==2)
			{
				ieta=binmapd2[eta]; // special map for depth 2
				if (ieta<=-30) ieta++;
				else if (ieta>=30) ieta--;
				else return -9999;
				return ieta;
			}
			else return -9999;
		}

		else if (subdet==HcalEndcap)
		{
			if (depth==1) 
				ieta=eta-42;
			else if (depth==2) 
			{
				ieta=binmapd2[eta];
				if (abs(ieta)>29 || abs(ieta)<18) return -9999; // outside HE
				if (ieta==0) return -9999;
				return ieta;
			}
			else if (depth==3)
			{
				if (eta<0 || eta>8) return -9999;
				else
					ieta=binmapd3[eta]; // special map for depth 3
				if (ieta==0) return -9999;
				return ieta;
			}
			else return -9999;
		} // HcalEndcap
		else if ( subdet==HcalOuter)
		{
			if (depth!=4)
				return -9999;
			else
			{
				ieta= eta-15;  // bin 0 is ieta=-15, all bins increment normally from there
				if (abs(ieta)>15) return -9999;
				if (ieta==0) return -9999;
				return ieta;
			}
		} // HcalOuter
		if (ieta==0) return -9999;
		return ieta;
	}

	inline int CalcIeta(int eta, int depth)
	{
		// This version of CalcIeta does the same as the function above,
		// but does not require that 'subdet' be specified.

		// returns ieta value give an eta counter.
		// eta runs from 0...X  (X depends on depth)
		int ieta=-9999;
		if (eta<0) return ieta;
		if (depth==1)
		{
			ieta=eta-42; // default shift: bin 0 corresponds to a histogram ieta of -42 (which is offset by 1 from true HF value of -41)
			if (eta<13) ieta++;
			else if (eta>71) ieta--;
			if (ieta==0) ieta=-9999;
			return ieta;
		}
		else if (depth==2)
		{
			if (eta>57) return -9999;
			else
			{
				ieta=binmapd2[eta];
				if (ieta==-9999) return ieta;
				if (ieta==0) return -9999;
				if (ieta==17 || ieta == -17) return -9999; // no depth 2 cells at |ieta| = 17
				else if (ieta<=-30) ieta++;
				else if (ieta>=30) ieta--;
				return ieta;
			}
		}
		else if (depth==3)
		{
			if (eta>8) return -9999;
			else
				ieta=binmapd3[eta];
			if (ieta==0) return -9999;
			return ieta;
		}
		else if (depth==4)
		{
			ieta= eta-15;  // bin 0 is ieta=-15, all bins increment normally from there
			if (abs(ieta)>15) return -9999;
			if (ieta==0) return -9999;
			return ieta;
		}
		return ieta; // avoids compilation warning
	}


	// Functions to check whether a given (eta,depth) value is valid for a given subdetector

	inline std::vector<std::string> HcalEtaPhiHistNames()
	{
		std::vector<std::string> name;
		name.push_back("HB HE HF Depth 1 ");
		name.push_back("HB HE HF Depth 2 ");
		name.push_back("HE Depth 3 ");
		name.push_back("HO Depth 4 ");
		return name;
	}


	inline bool isHB(int etabin, int depth)
	{
		if (depth>2) return false;
		else if (depth<1) return false;
		else
		{
			int ieta=CalcIeta(etabin,depth);
			if (ieta==-9999) return false;
			if (depth==1)
			{
				if (abs(ieta)<=16 ) return true;
				else return false;
			}
			else if (depth==2)
			{
				if (abs(ieta)==15 || abs(ieta)==16) return true;
				else return false;
			}
		}
		return false;
	}

	inline bool isHE(int etabin, int depth)
	{
		if (depth>3) return false;
		else if (depth<1) return false;
		else
		{
			int ieta=CalcIeta(etabin,depth);
			if (ieta==-9999) return false;
			if (depth==1)
			{
				if (abs(ieta)>=17 && abs(ieta)<=28 ) return true;
				if (ieta==-29 && etabin==13) return true; // HE -29
				if (ieta==29 && etabin == 71) return true; // HE +29
			}
			else if (depth==2)
			{
				if (abs(ieta)>=17 && abs(ieta)<=28 ) return true;
				if (ieta==-29 && etabin==13) return true; // HE -29
				if (ieta==29 && etabin == 43) return true; // HE +29
			}
			else if (depth==3)
				return true;
		}
		return false;
	}

	inline bool isHF(int etabin, int depth)
	{
		if (depth>2) return false;
		else if (depth<1) return false;
		else
		{
			int ieta=CalcIeta(etabin,depth);
			if (ieta==-9999) return false;
			if (depth==1)
			{
				if (ieta==-29 && etabin==13) return false; // HE -29
				else if (ieta==29 && etabin == 71) return false; // HE +29
				else if (abs(ieta)>=29 ) return true;
			}
			else if (depth==2)
			{
				if (ieta==-29 && etabin==13) return false; // HE -29
				else if (ieta==29 && etabin==43) return false; // HE +29
				else if (abs(ieta)>=29 ) return true;
			}
		}
		return false;
	}

	inline bool isHO(int etabin, int depth)
	{
		if (depth!=4) return false;
		int ieta=CalcIeta(etabin,depth);
		if (ieta!=-9999) return true;
		return false;
	}

	// Checks whether HO region contains SiPM

	inline bool isSiPM(int ieta, int iphi, int depth)
	{
		if (depth!=4) return false;
		// HOP1
		if (ieta>=5 && ieta <=10 && iphi>=47 && iphi<=58) return true;  
		// HOP2
		if (ieta>=11 && ieta<=15 && iphi>=59 && iphi<=70) return true;
		return false;
	}  // bool isSiPM


	// Checks whether (subdet, ieta, iphi, depth) value is a valid Hcal cell

	inline bool validDetId(HcalSubdetector sd, int ies, int ip, int dp)
	{
		// inputs are (subdetector, ieta, iphi, depth)
		// stolen from latest version of DataFormats/HcalDetId/src/HcalDetId.cc (not yet available in CMSSW_2_1_9)

		const int ie ( abs( ies ) ) ;

		return ( ( ip >=  1         ) &&
			( ip <= 72         ) &&
			( dp >=  1         ) &&
			( ie >=  1         ) &&
			( ( ( sd == HcalBarrel ) &&
			( ( ( ie <= 14         ) &&
			( dp ==  1         )    ) ||
			( ( ( ie == 15 ) || ( ie == 16 ) ) && 
			( dp <= 2          )                ) ) ) ||
			(  ( sd == HcalEndcap ) &&
			( ( ( ie == 16 ) &&
			( dp ==  3 )          ) ||
			( ( ie == 17 ) &&
			( dp ==  1 )          ) ||
			( ( ie >= 18 ) &&
			( ie <= 20 ) &&
			( dp <=  2 )          ) ||
			( ( ie >= 21 ) &&
			( ie <= 26 ) &&
			( dp <=  2 ) &&
			( ip%2 == 1 )         ) ||
			( ( ie >= 27 ) &&
			( ie <= 28 ) &&
			( dp <=  3 ) &&
			( ip%2 == 1 )         ) ||
			( ( ie == 29 ) &&
			( dp <=  2 ) &&
			( ip%2 == 1 )         )          )      ) ||
			(  ( sd == HcalOuter ) &&
			( ie <= 15 ) &&
			( dp ==  4 )           ) ||
			(  ( sd == HcalForward ) &&
			( dp <=  2 )          &&
			( ( ( ie >= 29 ) &&
			( ie <= 39 ) &&
			( ip%2 == 1 )    ) ||
			( ( ie >= 40 ) &&
			( ie <= 41 ) &&
			( ip%4 == 3 )         )  ) ) ) ) ;



	} // bool validDetId(HcalSubdetector sd, int ies, int ip, int dp)


	// Sets eta, phi labels for 'summary' eta-phi plots (identical to Depth 1 Eta-Phi labelling)

	inline void SetEtaPhiLabels(TH2F &h)
	{
		std::stringstream label;
		for (int i=-41;i<=-29;i=i+2)
		{
			label<<i;
			h.GetXaxis()->SetBinLabel(i+42,label.str().c_str());
			label.str("");
		}
		h.GetXaxis()->SetBinLabel(14,"-29HE");

		// offset by one for HE
		for (int i=-27;i<=27;i=i+2)
		{
			label<<i;
			h.GetXaxis()->SetBinLabel(i+43,label.str().c_str());
			label.str("");
		}
		h.GetXaxis()->SetBinLabel(72,"29HE");
		for (int i=29;i<=41;i=i+2)
		{
			label<<i;
			h.GetXaxis()->SetBinLabel(i+44,label.str().c_str());
			label.str("");
		}
		return;
	}


	// Fill Unphysical bins in histograms
	inline void FillUnphysicalHEHFBins(std::vector<TH2F> &hh)
	{
		int ieta=0;
		int iphi=0;
		// First 2 depths have 5-10-20 degree corrections
		for (unsigned int d=0;d<3;++d)
		{
			//BUG CAN BE HERE:
			//if (hh[d] != 0) continue;

			for (int eta=0;eta<hh[d].GetNbinsX();++eta)
			{
				ieta=CalcIeta(eta,d+1);
				if (ieta==-9999 || abs(ieta)<21) continue;
				for (int phi=0;phi <hh[d].GetNbinsY();++phi)
				{
					iphi=phi+1;
					if (iphi%2==1 && abs(ieta)<40 && iphi<73)
					{
						hh[d].SetBinContent(eta+1,iphi+1,hh[d].GetBinContent(eta+1,iphi));
					}
					// last two eta strips span 20 degrees in phi
					// Fill the phi cell above iphi, and the 2 below it
					else  if (abs(ieta)>39 && iphi%4==3 && iphi<73)
					{
						//ieta=40, iphi=3 covers iphi 3,4,5,6
						hh[d].SetBinContent(eta+1,(iphi)%72+1, hh[d].GetBinContent(eta+1,iphi));
						hh[d].SetBinContent(eta+1,(iphi+1)%72+1, hh[d].GetBinContent(eta+1,iphi));
						hh[d].SetBinContent(eta+1,(iphi+2)%72+1, hh[d].GetBinContent(eta+1,iphi));
					}
				} // for (int phi...)
			} // for (int eta...)
		} // for (int d=0;...)
		// no corrections needed for HO (depth 4)
		return;
	} // FillUnphysicalHEHFBins(MonitorElement* hh)


	//Fill unphysical bins for single ME
	inline void FillUnphysicalHEHFBins(TH2F &hh)
	{
		// Fills unphysical HE/HF bins for Summary Histogram
		// Summary Histogram is binned with the same binning as the Depth 1 EtaPhiHists

		//CAN BE BUG HERE:
		//if (hh==0) return;

		int ieta=0;
		int iphi=0;
		int etabins = hh.GetNbinsX();
		int phibins = hh.GetNbinsY();
		float binval=0;
		for (int eta=0;eta<etabins;++eta) // loop over eta bins
		{
			ieta=CalcIeta(eta,1);
			if (ieta==-9999 || abs(ieta)<21) continue;  // ignore etas that don't exist, or that have 5 degree phi binning

			for (int phi=0;phi<phibins;++phi)
			{
				iphi=phi+1;
				if (iphi%2==1 && abs(ieta)<40 && iphi<73) // 10 degree phi binning condition
				{
					binval=hh.GetBinContent(eta+1,iphi);
					hh.SetBinContent(eta+1,iphi+1,binval);
				} // if (iphi%2==1...) 
				else if (abs(ieta)>39 && iphi%4==3 && iphi<73) // 20 degree phi binning condition
				{
					// Set last two eta strips where each cell spans 20 degrees in phi
					// Set next phi cell above iphi, and 2 cells below the actual cell 
					hh.SetBinContent(eta+1, (iphi)%72+1, hh.GetBinContent(eta+1,iphi));
					hh.SetBinContent(eta+1, (iphi+1)%72+1, hh.GetBinContent(eta+1,iphi));
					hh.SetBinContent(eta+1, (iphi+2)%72+1, hh.GetBinContent(eta+1,iphi));
				} // else if (abs(ieta)>39 ...)
			} // for (int phi=0;phi<72;++phi)

		} // for (int eta=0; eta< (etaBins_-2);++eta)

		return;
	} // FillUnphysicalHEHFBins(std::vector<MonitorElement*> &hh)



	// special fill call based on detid -- eventually will need special treatment
	void Fill(HcalDetId& id, double val /*=1*/, std::vector<TH2F> &depth)
	{ 
		// If in HF, need to shift by 1 bin (-1 bin lower in -HF, +1 bin higher in +HF)
		if (id.subdet()==HcalForward)
			depth[id.depth()-1].Fill(id.ieta()<0 ? id.ieta()-1 : id.ieta()+1, id.iphi(), val);
		else 
			depth[id.depth()-1].Fill(id.ieta(),id.iphi(),val);
	}

	void Reset(std::vector<TH2F> &depth) 
	{
		for (unsigned int d=0;d<depth.size();d++)
			//BUG CAN BE HERE:
			//if(depth[d]) 
			depth[d].Reset();
	} // void Reset(void)

	void setup(std::vector<TH2F> &depth, std::string name, std::string units=""){
		std::string unittitle, unitname;
		if (units.empty())
		{
			unitname = units;
			unittitle = "No Units";
		}
		else
		{
			unitname = " " + units;
			unittitle = units;
		}

		// Push back depth plots
		////////Create 4 plots:
		//1. create first plot	
		depth.push_back(TH2F(("HB HE HF Depth 1 "+name+unitname).c_str(),
			(name+" Depth 1 -- HB HE HF ("+unittitle+")").c_str(),
			85,-42.5,42.5,
			72,0.5,72.5));

		//2.1 prepare second plot	
		float ybins[73];
		for (int i=0;i<=72;i++) ybins[i]=(float)(i+0.5);
		float xbinsd2[]={-42.5,-41.5,-40.5,-39.5,-38.5,-37.5,-36.5,-35.5,-34.5,-33.5,-32.5,-31.5,-30.5,-29.5,
			-28.5,-27.5,-26.5,-25.5,-24.5,-23.5,-22.5,-21.5,-20.5,-19.5,-18.5,-17.5,-16.5,
			-15.5,-14.5,
			14.5, 15.5,
			16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,
			31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.5,39.5,40.5,41.5,42.5};

		//2.2 create second plot	
		depth.push_back(TH2F(("HB HE HF Depth 2 "+name+unitname).c_str(),
			(name+" Depth 2 -- HB HE HF ("+unittitle+")").c_str(),
			57, xbinsd2, 72, ybins));

		//3.1 Set up variable-sized bins for HE depth 3 (MonitorElement also requires phi bins to be entered in array format)
		float xbins[]={-28.5,-27.5,-26.5,-16.5,-15.5,
			15.5,16.5,26.5,27.5,28.5};
		//3.2
		depth.push_back(TH2F(("HE Depth 3 "+name+unitname).c_str(),
			(name+" Depth 3 -- HE ("+unittitle+")").c_str(),
			// Use variable-sized eta bins 
			9, xbins, 72, ybins));

		//4.1 HO bins are fixed width, but cover a smaller eta range (-15 -> 15)
		depth.push_back(TH2F(("HO Depth 4 "+name+unitname).c_str(),
			(name+" Depth 4 -- HO ("+unittitle+")").c_str(),
			31,-15.5,15.5,
			72,0.5,72.5));

		for (unsigned int i=0;i<depth.size();++i)
			depth[i].Draw("colz");

		setBinLabels(depth); // set axis titles, special bins		
	}

	void fillOneGain(std::vector<TH2F> &graphData, HcalGains::tAllContWithNames &allContainers, std::string name, int id, std::string units=""){
		setup(graphData, name); 

		std::stringstream x;
		// Change the titles of each individual histogram
		for (unsigned int d=0;d < graphData.size();++d){
			graphData[d].Reset();
			x << "Gain "<< id  << " for HCAL depth " << d+1;

			//BUG CAN BE HERE:
			//if (ChannelStatus->depth[d]) 
			graphData[d].SetTitle(x.str().c_str());  // replace "setTitle" with "SetTitle", since you are using TH2F objects instead of MonitorElements
			x.str("");
		}

		HcalDetId hcal_id;
		int ieta, depth, iphi;

		//main loop
		// get all containers with names
		//HcalGains::tAllContWithNames allContainers = object().getAllContainers();

		//ITERATORS AND VALUES:
		HcalGains::tAllContWithNames::const_iterator iter;
		std::vector<HcalGain>::const_iterator contIter;
		float gain = 0.0;

		//Run trough given id gain:
		int i = id;

		//run trough all pair containers
		for (iter = allContainers.begin(); iter != allContainers.end(); ++iter){
			//Run trough all values:
			for (contIter = (*iter).second.begin(); contIter != (*iter).second.end(); ++contIter){
				hcal_id = HcalDetId((uint32_t)(*contIter).rawId());

				depth = hcal_id.depth();
				if (depth<1 || depth>4) 
					continue;

				ieta=hcal_id.ieta();
				iphi=hcal_id.iphi();

				if (hcal_id.subdet() == HcalForward)
					ieta>0 ? ++ieta : --ieta;

				//GET VALUE:
				gain = (*contIter).getValue(i);
				//logstatus = log2(1.*channelBits)+1;

				//FILLING GOES HERE:
				graphData[depth-1].Fill(ieta,iphi, gain);

				//FOR DEBUGGING:
				//std::cout << "ieta: " << ieta << "; iphi: " << iphi << "; logstatus: " << logstatus << "; channelBits: " << channelBits<< std::endl;
			}

		}	
	}


	//FillUnphysicalHEHFBins(graphData);
	//return ("kasdasd");

	class ADataRepr //Sample base class for c++ inheritance tutorial
	{
	public:
		ADataRepr(unsigned int d):m_total(d){};
		virtual ~ADataRepr(){};
		unsigned int nr, id;
		std::stringstream filename, rootname, plotname;

		void fillOneGain(std::vector<TH2F> &graphData, std::string units=""){
			std::stringstream ss("");

			if (m_total == 1)
				ss << rootname.str() << " for HCAL depth ";
			else
				ss << rootname.str() << nr << " for HCAL depth ";

			setup(graphData, ss.str()); 

			// Change the titles of each individual histogram
			for (unsigned int d=0;d < graphData.size();++d){
				graphData[d].Reset();

				ss.str("");
				if (m_total == 1)
					ss << plotname.str() << " for HCAL depth " << d+1;
				else
					ss << plotname.str() << nr << " for HCAL depth " << d+1;


				//BUG CAN BE HERE:
				//if (ChannelStatus->depth[d]) 
				graphData[d].SetTitle(ss.str().c_str());  // replace "setTitle" with "SetTitle", since you are using TH2F objects instead of MonitorElements
				ss.str("");
			}
			//overload this function:
			doFillIn(graphData);

			FillUnphysicalHEHFBins(graphData);
			
			ss.str("");
			if (m_total == 1)
				ss << filename.str() << ".png";
			else
				ss << filename.str() << nr << ".png";
			draw(graphData, ss.str());
			//FOR DEBUGGING:
			//std::cout << "ieta: " << ieta << "; iphi: " << iphi << "; logstatus: " << logstatus << "; channelBits: " << channelBits<< std::endl;
		}

	protected:
		unsigned int m_total;
		HcalDetId hcal_id;
		int ieta, depth, iphi;


	        virtual void doFillIn(std::vector<TH2F> &graphData) = 0;
		
	private:

	void draw(std::vector<TH2F> &graphData, std::string filename) {
		//Drawing...
		// use David's palette
		gStyle->SetPalette(1);
		const Int_t NCont = 999;
		gStyle->SetNumberContours(NCont);
		TCanvas canvas("CC map","CC map",840,369*4);

		TPad pad1("pad1","pad1", 0.0, 0.75, 1.0, 1.0);
		pad1.Draw();
		TPad pad2("pad2","pad2", 0.0, 0.5, 1.0, 0.75);
		pad2.Draw();
		TPad pad3("pad3","pad3", 0.0, 0.25, 1.0, 0.5);
		pad3.Draw();
		TPad pad4("pad4","pad4", 0.0, 0.0, 1.0, 0.25);
		pad4.Draw();


		pad1.cd();
		graphData[0].SetStats(false);
		graphData[0].Draw("colz");

		pad2.cd();
		graphData[1].SetStats(false);
		graphData[1].Draw("colz");

		pad3.cd();
		graphData[2].SetStats(false);
		graphData[2].Draw("colz");

		pad4.cd();
		graphData[3].SetStats(false);
		graphData[3].Draw("colz");

		canvas.SaveAs(filename.c_str());
	}

		void setup(std::vector<TH2F> &depth, std::string name, std::string units=""){
			std::string unittitle, unitname;
			if (units.empty())
			{
				unitname = units;
				unittitle = "No Units";
			}
			else
			{
				unitname = " " + units;
				unittitle = units;
			}

			// Push back depth plots
			////////Create 4 plots:
			//1. create first plot	
			depth.push_back(TH2F(("HB HE HF Depth 1 "+name+unitname).c_str(),
				(name+" Depth 1 -- HB HE HF ("+unittitle+")").c_str(),
				85,-42.5,42.5,
				72,0.5,72.5));

			//2.1 prepare second plot	
			float ybins[73];
			for (int i=0;i<=72;i++) ybins[i]=(float)(i+0.5);
			float xbinsd2[]={-42.5,-41.5,-40.5,-39.5,-38.5,-37.5,-36.5,-35.5,-34.5,-33.5,-32.5,-31.5,-30.5,-29.5,
				-28.5,-27.5,-26.5,-25.5,-24.5,-23.5,-22.5,-21.5,-20.5,-19.5,-18.5,-17.5,-16.5,
				-15.5,-14.5,
				14.5, 15.5,
				16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,
				31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.5,39.5,40.5,41.5,42.5};

			//2.2 create second plot	
			depth.push_back(TH2F(("HB HE HF Depth 2 "+name+unitname).c_str(),
				(name+" Depth 2 -- HB HE HF ("+unittitle+")").c_str(),
				57, xbinsd2, 72, ybins));

			//3.1 Set up variable-sized bins for HE depth 3 (MonitorElement also requires phi bins to be entered in array format)
			float xbins[]={-28.5,-27.5,-26.5,-16.5,-15.5,
				15.5,16.5,26.5,27.5,28.5};
			//3.2
			depth.push_back(TH2F(("HE Depth 3 "+name+unitname).c_str(),
				(name+" Depth 3 -- HE ("+unittitle+")").c_str(),
				// Use variable-sized eta bins 
				9, xbins, 72, ybins));

			//4.1 HO bins are fixed width, but cover a smaller eta range (-15 -> 15)
			depth.push_back(TH2F(("HO Depth 4 "+name+unitname).c_str(),
				(name+" Depth 4 -- HO ("+unittitle+")").c_str(),
				31,-15.5,15.5,
				72,0.5,72.5));

			for (unsigned int i=0;i<depth.size();++i)
				depth[i].Draw("colz");

			setBinLabels(depth); // set axis titles, special bins		
		}

		//functions for making plot:
		void setBinLabels(std::vector<TH2F> &depth)
		{
			// Set labels for all depth histograms
			for (unsigned int i=0;i<depth.size();++i)
			{
				depth[i].SetXTitle("i#eta");
				depth[i].SetYTitle("i#phi");
			}

			std::stringstream label;

			// set label on every other bin
			for (int i=-41;i<=-29;i=i+2)
			{
				label<<i;
				depth[0].GetXaxis()->SetBinLabel(i+42,label.str().c_str());
				depth[1].GetXaxis()->SetBinLabel(i+42,label.str().c_str());
				label.str("");
			}
			depth[0].GetXaxis()->SetBinLabel(14,"-29HE");
			depth[1].GetXaxis()->SetBinLabel(14,"-29HE");

			// offset by one for HE
			for (int i=-27;i<=27;i=i+2)
			{
				label<<i;
				depth[0].GetXaxis()->SetBinLabel(i+43,label.str().c_str());
				label.str("");
			}
			depth[0].GetXaxis()->SetBinLabel(72,"29HE");
			for (int i=29;i<=41;i=i+2)
			{
				label<<i;
				depth[0].GetXaxis()->SetBinLabel(i+44,label.str().c_str());
				label.str("");
			}
			for (int i=16;i<=28;i=i+2)
			{
				label<<i-43;
				depth[1].GetXaxis()->SetBinLabel(i,label.str().c_str());
				label.str("");
			}
			depth[1].GetXaxis()->SetBinLabel(29,"NULL");
			for (int i=15;i<=27;i=i+2)
			{
				label<<i;
				depth[1].GetXaxis()->SetBinLabel(i+15,label.str().c_str());
				label.str("");
			}

			depth[1].GetXaxis()->SetBinLabel(44,"29HE");
			for (int i=29;i<=41;i=i+2)
			{
				label<<i;
				depth[1].GetXaxis()->SetBinLabel(i+16,label.str().c_str());
				label.str("");
			}

			// HE depth 3 labels;
			depth[2].GetXaxis()->SetBinLabel(1,"-28");
			depth[2].GetXaxis()->SetBinLabel(2,"-27");
			depth[2].GetXaxis()->SetBinLabel(3,"Null");
			depth[2].GetXaxis()->SetBinLabel(4,"-16");
			depth[2].GetXaxis()->SetBinLabel(5,"Null");
			depth[2].GetXaxis()->SetBinLabel(6,"16");
			depth[2].GetXaxis()->SetBinLabel(7,"Null");
			depth[2].GetXaxis()->SetBinLabel(8,"27");
			depth[2].GetXaxis()->SetBinLabel(9,"28");
		}

	};
}
#endif
