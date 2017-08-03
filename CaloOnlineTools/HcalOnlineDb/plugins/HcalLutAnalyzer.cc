// -*- C++ -*-
//
// Package:    Test/HcalLutAnalyzer
// Class:      HcalLutAnalyzer
// 
/**\class HcalLutAnalyzer HcalLutAnalyzer.cc Test/HcalLutAnalyzer/plugins/HcalLutAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Aleko Khukhunaishvili
//         Created:  Fri, 21 Jul 2017 08:42:05 GMT
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream> 

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/LutXml.h"
#include "PhysicsTools/FWLite/interface/CommandLineParser.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "TString.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"

class HcalLutAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit HcalLutAnalyzer(const edm::ParameterSet&);
      ~HcalLutAnalyzer(){};
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      std::string inputDir;
      std::string plotsDir;
      std::vector<std::string> tags_;
      std::vector<std::string> quality_;
      std::vector<std::string> pedestals_;
      std::vector<std::string> gains_;
      std::vector<std::string> respcorrs_;

      double Zmin;
      double Zmax;
      double Ymin;
      double Ymax;
      double Pmin;
      double Pmax;
};

HcalLutAnalyzer::HcalLutAnalyzer(const edm::ParameterSet& iConfig)
{
    inputDir	= iConfig.getParameter<std::string>("inputDir");
    plotsDir	= iConfig.getParameter<std::string>("plotsDir");
    tags_	= iConfig.getParameter<std::vector<std::string> >("tags");
    quality_	= iConfig.getParameter<std::vector<std::string> >("quality");
    pedestals_	= iConfig.getParameter<std::vector<std::string> >("pedestals");
    gains_	= iConfig.getParameter<std::vector<std::string> >("gains");
    respcorrs_	= iConfig.getParameter<std::vector<std::string> >("respcorrs");

    Zmin = iConfig.getParameter<double>("Zmin");
    Zmax = iConfig.getParameter<double>("Zmax");
    Ymin = iConfig.getParameter<double>("Ymin");
    Ymax = iConfig.getParameter<double>("Ymax");
    Pmin = iConfig.getParameter<double>("Pmin");
    Pmax = iConfig.getParameter<double>("Pmax");
}


void 
HcalLutAnalyzer::analyze(const edm::Event&, const edm::EventSetup& iSetup)
{
    using namespace std;

    edm::ESHandle<HcalTopology> topology ;
    iSetup.get<HcalRecNumberingRecord>().get( topology );

    typedef std::vector<std::string> vstring;
    typedef std::map<unsigned long int, float> LUTINPUT;

    static const int NVAR=5; //variables
    static const int NDET=2; //detectors
    static const int NDEP=7; //depths
    static const int NLEV=3; //old,new,ratio

    const bool  doRatio[NVAR] = {false, true, true, false, true};
    const char* titleVar[NVAR]= {"Pedestals", "RespCorrs", "Gains", "Threshold", "LUTs"};
    const char* titleHisR[NLEV]= {"Old", "New", "Ratio"};
    const char* titleHisD[NLEV]= {"Old", "New", "Difference"};
    const char* titleDet[4]= {"HBHE", "HF", "HEP17", "HO"};
    const int   DEP[NDET]={7,4};
    const char* titleDep[NDEP]= {"depth1", "depth2", "depth3", "depth4", "depth5","depth6","depth7"};


    TH2D     *r[NVAR][NDET];
    TProfile *p[NVAR][NDET];

    TH2D *h[NVAR][NLEV][NDET][NDEP];
    TH2D *hlut[4][NLEV];
    TH2D *hslope[2];
    TH2D *houtput[4][2];

    for(int d=0; d<4; ++d){
	for(int i=0; i<3; ++i){
	    hlut[d][i] = new TH2D(Form("Lut_%s_%s", titleDet[d], titleHisR[i]), Form("Input LUT, %s", (i==2?"Ratio":tags_[i].c_str())), 260, 0, 260, 240, 0, i==NLEV-1?3:2400);
	    hlut[d][i]->SetMarkerColor(d==0?kBlue:d==1? kGreen+2 : d==2?kRed : kCyan);
	    hlut[d][i]->SetXTitle("raw adc");
	    hlut[d][i]->SetYTitle("lin adc");
	}
    }

    for(int d=0; d<NDET; ++d){
	hslope[d] = new TH2D(Form("GainLutScatter_%s", titleDet[d]), Form("Gain-Lutslope scatter, %s",titleDet[d]), 200, 0, 2, 200, 0, 2);
	hslope[d]->SetXTitle("Gain x RespCorr ratio");
	hslope[d]->SetYTitle("Lut ratio");

	for(int j=0; j<NVAR; ++j){
	    double rmin=doRatio[j]?Ymin:-6;
	    double rmax=doRatio[j]?Ymax: 6;
	    r[j][d] = new TH2D(Form("r%s_%s",  titleVar[j], titleDet[d]), Form("%s, %s",titleVar[j], titleDet[d]), 83,-41.5, 41.5, 250, rmin, rmax); 
	    r[j][d]->SetXTitle("iEta");
	    r[j][d]->SetYTitle(doRatio[j]?"New / Old":"New - Old");
	    p[j][d] = new TProfile(Form("p%s_%s",  titleVar[j], titleDet[d]), Form("%s, %s",titleVar[j], titleDet[d]), 83,-41.5, 41.5); 
	    p[j][d]->SetXTitle("iEta");
	    p[j][d]->SetYTitle(doRatio[j]?"New / Old":"New - Old");
	    p[j][d]->SetMarkerStyle(20);
	    p[j][d]->SetMarkerSize(0.9);
	    p[j][d]->SetMarkerColor(kBlue);

	    for(int p=0; p<DEP[d]; ++p){
		for(int i=0; i<NLEV; ++i){
		    const char *titHist=doRatio[j]?titleHisR[i]:titleHisD[i];
		    h[j][i][d][p] = new TH2D(Form("h%s_%s_%s_%s",  titleVar[j], titHist,titleDet[d],titleDep[p]), 
			                     Form("%s, %s, %s, %s",titleVar[j], titHist,titleDet[d],titleDep[p]), 83, -41.5,  41.5, 72, 0.5, 72.5);
		    h[j][i][d][p]->SetXTitle("iEta");
		    h[j][i][d][p]->SetYTitle("iPhi");
		}
	    }
	}
    }

    for(int i=0; i<4; ++i){
	int color=i==0?kBlue : i==1? kViolet : i==2 ? kGreen+2 : kRed;
	houtput[i][0] = new TH2D(Form("houtlut0_%d",i), Form("Output LUT, %s", tags_[0].c_str()), 2100,0,2100,260,0,260);
	houtput[i][1] = new TH2D(Form("houtlut1_%d",i), Form("Output LUT, %s", tags_[1].c_str()), 2100,0,2100,260,0,260);
	for(int j=0; j<2; ++j) {
	    houtput[i][j]->SetMarkerColor(color);
	    houtput[i][j]->SetLineColor(color);
	}
    }

    //FILL LUT INPUT DATA
    LUTINPUT lutgain[2];
    LUTINPUT lutresp[2];
    LUTINPUT lutpede[2];

    assert(tags_.size()==2);
    assert(gains_.size()==2);
    assert(respcorrs_.size()==2);
    assert(pedestals_.size()==2);

    unsigned long int iraw;
    int ieta, iphi, idep;
    string det, base;
    float val1, val2, val3, val4;
    float wid1, wid2, wid3, wid4;
    char buffer[1024];

    std::vector<HcalDetId> BadChans[2];
    std::vector<HcalDetId> ZeroLuts[2];

    //Read input condition files
    for(int ii=0; ii<2; ++ii){
	//Gains
	std::ifstream infile(edm::FileInPath(Form("%s/Gains/Gains_Run%s.txt", inputDir.c_str(), gains_[ii].c_str())).fullPath().c_str());
	assert(!infile.fail());
	while(!infile.eof()){
	    infile.getline(buffer, 1024);
	    if(buffer[0]=='#') continue;
	    std::istringstream(buffer) >> ieta >> iphi >> idep >> det >> val1 >> val2 >> val3 >> val4 >> iraw ; 
	    if(det!="HB" && det!="HE" && det!="HF") continue;

	    float theval = (val1+val2+val3+val4)/4.0;

	    HcalSubdetector subdet = det=="HB" ? HcalBarrel :
		                     det=="HE" ? HcalEndcap :
				     det=="HF" ? HcalForward:
				                 HcalOther;

	    HcalDetId id(subdet, ieta, iphi, idep);
	    lutgain[ii].insert(LUTINPUT::value_type(id.rawId(), theval));
	}


	//Pedestals
	std::ifstream infped(edm::FileInPath(Form("%s/Pedestals/Pedestals_Run%s.txt", inputDir.c_str(), pedestals_[ii].c_str())).fullPath().c_str());
	assert(!infped.fail());
	while(!infped.eof()){
	    infped.getline(buffer, 1024);
	    if(buffer[0]=='#') continue;
	    std::istringstream(buffer) >> ieta >> iphi >> idep >> det >> val1 >> val2 >> val3 >> val4 >> wid1 >> wid2 >> wid3 >> wid4 >> iraw ; 
	    if(det!="HB" && det!="HE" && det!="HF") continue;

	    float theval = (val1+val2+val3+val4)/4.0;

	    HcalSubdetector subdet = det=="HB" ? HcalBarrel :
		                     det=="HE" ? HcalEndcap :
				     det=="HF" ? HcalForward:
				                 HcalOther;

	    HcalDetId id(subdet, ieta, iphi, idep);
	    lutpede[ii].insert(LUTINPUT::value_type(id.rawId(), theval));
	}



	//Response corrections
	std::ifstream inresp(edm::FileInPath(Form("%s/RespCorrs/RespCorrs_Run%s.txt", inputDir.c_str(), respcorrs_[ii].c_str())).fullPath().c_str());
	assert(!inresp.fail());
	while(!inresp.eof()){
	    inresp.getline(buffer, 1024);
	    if(buffer[0]=='#') continue;
	    std::istringstream(buffer) >> ieta >> iphi >> idep >> det >> val1 >> iraw ; 
	    if(det!="HB" && det!="HE" && det!="HF") continue;

	    float theval = val1;

	    HcalSubdetector subdet = det=="HB" ? HcalBarrel :
		                     det=="HE" ? HcalEndcap :
				     det=="HF" ? HcalForward:
				                 HcalOther;

	    HcalDetId id(subdet, ieta, iphi, idep);
	    lutresp[ii].insert(LUTINPUT::value_type(id.rawId(), theval));
	}


	//ChannelQuality
	std::ifstream inchan(edm::FileInPath(Form("%s/ChannelQuality/ChannelQuality_Run%s.txt", inputDir.c_str(), quality_[ii].c_str())).fullPath().c_str());
	assert(!inchan.fail());
	while(!inchan.eof()){
	    inchan.getline(buffer, 1024);
	    if(buffer[0]=='#') continue;
	    std::istringstream(buffer) >> ieta >> iphi >> idep >> det >> base >> val1 >> iraw ; 

	    float theval = val1;

	    HcalSubdetector subdet = det=="HB" ? HcalBarrel :
		                     det=="HE" ? HcalEndcap :
				     det=="HF" ? HcalForward:
				     det=="HO" ? HcalOuter:
				                 HcalOther;

	    HcalDetId id(subdet, ieta, iphi, idep);
	    if(theval!=0) BadChans[ii].push_back(id);
	}
    }


    LutXml xmls1(edm::FileInPath(Form("%s/%s/%s.xml", inputDir.c_str(), tags_[0].c_str(), tags_[0].c_str())).fullPath()); 
    LutXml xmls2(edm::FileInPath(Form("%s/%s/%s.xml", inputDir.c_str(), tags_[1].c_str(), tags_[1].c_str())).fullPath()); 

    xmls1.create_lut_map();
    xmls2.create_lut_map();

    for (const auto& xml2 : xmls2){
	HcalGenericDetId detid(xml2.first);

	if(detid.genericSubdet()==HcalGenericDetId::HcalGenTriggerTower){
	    HcalTrigTowerDetId tid(detid.rawId());
	    if(!topology->validHT(tid)) continue;
	    const auto& lut2 = xml2.second;

	    int D=abs(tid.ieta())<29 ? (lut2.size()==1024 ? 0 : 3) : 
		      tid.version()==0? 1: 2;
	    for(size_t i=0; i<lut2.size(); ++i){
		if(int(i)%4==D)
		houtput[D][1]->Fill(i,lut2[i]);
	    }
	}
	else if(topology->valid(detid)){
	    HcalDetId id(detid);
	    HcalSubdetector subdet=id.subdet();
	    int idet = int(subdet);
	    const auto& lut2 = xml2.second;
	    int hbhe = idet==HcalForward ? 1 : 
		       idet==HcalOuter   ? 3 : 
		       lut2.size()==128 ? 0 : 2;
	    for(size_t i=0; i<lut2.size(); ++i) {
		hlut[hbhe][1]->Fill(i, lut2[i]);
		if(hbhe==2) hlut[hbhe][1]->Fill(i, lut2[i]&0x3FF); 
	    }
	}
    }

    for (const auto& xml1 : xmls1){

	HcalGenericDetId detid(xml1.first);
	const auto& lut1 = xml1.second;

	if(detid.genericSubdet()==HcalGenericDetId::HcalGenTriggerTower){
	    HcalTrigTowerDetId tid(detid.rawId());
	    if(!topology->validHT(tid)) continue;
	    int D=abs(tid.ieta())<29 ? (lut1.size()==1024 ? 0 : 3) : 
		      tid.version()==0? 1: 2;
	    for(size_t i=0; i<lut1.size(); ++i){
		if(int(i)%4==D)
		houtput[D][0]->Fill(i,lut1[i]);
	    }
	}else if(topology->valid(detid)){
	    HcalDetId id(detid);
	    HcalSubdetector subdet=id.subdet();
	    int idet = int(subdet);
	    const auto& lut1 = xml1.second;
	    int hbhe = idet==HcalForward ? 1 : 
		       idet==HcalOuter   ? 3 : 
		       lut1.size()==128 ? 0 : 2;
	    for(size_t i=0; i<lut1.size(); ++i) {
		hlut[hbhe][0]->Fill(i, lut1[i]);
		if(hbhe==2) hlut[hbhe][0]->Fill(i, lut1[i]&0x3FF); 
	    }
	}

	auto xml2 =xmls2.find(detid.rawId());
	if(xml2==xmls2.end()) continue;

	if(detid.genericSubdet()==HcalGenericDetId::HcalGenTriggerTower) continue;

	HcalDetId id(detid);

	HcalSubdetector subdet=id.subdet();
	int idet = int(subdet);
	int ieta = id.ieta();
	int iphi = id.iphi();
	int idep = id.depth()-1;
	unsigned long int iraw = id.rawId();

	if(!topology->valid(detid)) continue;

	int hbhe = idet==HcalForward;

	const auto& lut2 = xml2->second;


	size_t size = lut1.size();
	if(size != lut2.size()) continue;

	std::vector<unsigned int> llut1(size);
	std::vector<unsigned int> llut2(size);
	for(size_t i=0; i<size; ++i){
	    llut1[i]=hbhe==0? lut1[i]&0x3FF: lut1[i] ;
	    llut2[i]=hbhe==0? lut2[i]&0x3FF: lut2[i] ;
	}

	int threshold[2]={0, 0};
	//Thresholds
	for(size_t i=0; i<size; ++i){
	    if(llut1[i]>0){
		threshold[0]=i;
		break;
	    }
	    if(i==size-1){
		ZeroLuts[0].push_back(id);
	    }
	}
	for(size_t i=0; i<size; ++i){
	    if(llut2[i]>0){
		threshold[1]=i;
		break;
	    }
	    if(i==size-1){
		ZeroLuts[1].push_back(id);
	    }
	}

	if(subdet!=HcalBarrel && subdet!=HcalEndcap && subdet!=HcalForward) continue;

	//fill conditions

	double xfill=0;
	h[0][0][hbhe][idep]->Fill(ieta, iphi,lutpede[0][iraw]);
	h[0][1][hbhe][idep]->Fill(ieta, iphi,lutpede[1][iraw]);
	xfill=lutpede[1][iraw]-lutpede[0][iraw];
	h[0][2][hbhe][idep]->Fill(ieta, iphi, xfill);
	r[0][hbhe]->Fill(ieta, xfill);
	p[0][hbhe]->Fill(ieta, xfill);

	h[1][0][hbhe][idep]->Fill(ieta, iphi,lutresp[0][iraw]);
	h[1][1][hbhe][idep]->Fill(ieta, iphi,lutresp[1][iraw]);
	xfill=lutresp[1][iraw]/lutresp[0][iraw];
	h[1][2][hbhe][idep]->Fill(ieta, iphi, xfill);
	r[1][hbhe]->Fill(ieta, xfill);
	p[1][hbhe]->Fill(ieta, xfill);

	h[2][0][hbhe][idep]->Fill(ieta, iphi,lutgain[0][iraw]);
	h[2][1][hbhe][idep]->Fill(ieta, iphi,lutgain[1][iraw]);
	xfill=lutgain[1][iraw]/lutgain[0][iraw];
	h[2][2][hbhe][idep]->Fill(ieta, iphi, xfill);
	r[2][hbhe]->Fill(ieta, xfill);
	p[2][hbhe]->Fill(ieta, xfill);

	h[3][0][hbhe][idep]->Fill(ieta, iphi, threshold[0]);
	h[3][1][hbhe][idep]->Fill(ieta, iphi, threshold[1]);
	xfill=threshold[1]-threshold[0];
	h[3][2][hbhe][idep]->Fill(ieta, iphi, xfill);
	r[3][hbhe]->Fill(ieta, xfill);
	p[3][hbhe]->Fill(ieta, xfill);

	size_t maxvalue=hbhe==0?1023:2047;

	//LutDifference
	for(size_t i=0; i<size; ++i){

	    hlut[hbhe][2]->Fill(i, llut1[i]==0?0:(double)llut2[i]/llut1[i]);

	    if(i==size-1 || (llut1[i]==maxvalue || llut2[i]==maxvalue)){ //Fill with only the last one before the maximum 
		if(llut1[i-1]==0 || llut2[i-1]==0) { 
		    break; 		
		}
		double condratio=lutgain[1][iraw]/lutgain[0][iraw] * lutresp[1][iraw]/lutresp[0][iraw]; 
		xfill= (double)llut2[i-1]/llut1[i-1];
		hslope[hbhe]->Fill(condratio, xfill);

		h[4][0][hbhe][idep]->Fill(ieta, iphi, (double)llut1[i-1]/(i-1));
		h[4][1][hbhe][idep]->Fill(ieta, iphi, (double)llut2[i-1]/(i-1));
		h[4][2][hbhe][idep]->Fill(ieta, iphi, xfill);
		r[4][hbhe]->Fill(ieta, xfill);
		p[4][hbhe]->Fill(ieta, xfill);

		break;
	    }
	}
    }

    gROOT->SetStyle("Plain");
    gStyle->SetPalette(1);
    gStyle->SetStatW(0.2);
    gStyle->SetStatH(0.1);
    gStyle->SetStatY(1.0);
    gStyle->SetStatX(0.9);
    gStyle->SetOptStat(110010);
    gStyle->SetOptFit(111111);

    //Draw and Print
    TCanvas *cc = new TCanvas("cc", "cc", 0, 0, 1600, 1200);
    cc->SetGridy();
    for(int j=0; j<NVAR; ++j){
	gSystem->mkdir(TString(plotsDir)+"/_"+titleVar[j]);
	for(int d=0; d<NDET; ++d){
	    cc->Clear(); 
	    r[j][d]->Draw("colz"); 
	    cc->Print(TString(plotsDir)+"/_"+titleVar[j]+"/"+TString(r[j][d]->GetName())+".pdf");

	    cc->Clear(); 
	    p[j][d]->Draw(); 
	    if(doRatio[j]){
		p[j][d]->SetMinimum(Pmin);
		p[j][d]->SetMaximum(Pmax);
	    }
	    cc->Print(TString(plotsDir)+"/_"+titleVar[j]+"/"+TString(p[j][d]->GetName())+".pdf");

	    for(int i=0; i<NLEV; ++i){
		for(int p=0; p<DEP[d]; ++p){
		    cc->Clear(); 
		    h[j][i][d][p]->Draw("colz"); 

		    if(i==NLEV-1){
			if(doRatio[j]){
			    h[j][2][d][p]->SetMinimum(Zmin);
			    h[j][2][d][p]->SetMaximum(Zmax);
			}
			else{
			    h[j][2][d][p]->SetMinimum(-3);
			    h[j][2][d][p]->SetMaximum( 3);
			}
		    }
		    cc->Print(TString(plotsDir)+"/_"+titleVar[j]+"/"+TString(h[j][i][d][p]->GetName())+".pdf");
		}
	    }
	}
    }

    for(int i=0; i<NLEV; ++i){
	cc->Clear(); 
	hlut[0][i]->Draw(); 
	hlut[1][i]->Draw("sames"); 
	hlut[2][i]->Draw("sames"); 
	hlut[3][i]->Draw("sames"); 
	cc->Print(TString(plotsDir)+Form("LUT_%d.gif",i));
    }
    cc->Clear(); hslope[0]->Draw("colz"); cc->SetGridx(); cc->SetGridy(); cc->Print(TString(plotsDir)+"GainLutScatterHBHE.pdf");
    cc->Clear(); hslope[1]->Draw("colz"); cc->SetGridx(); cc->SetGridy(); cc->Print(TString(plotsDir)+"GainLutScatterLutHF.pdf");

    for(int i=0; i<2; ++i){
	cc->Clear(); 
	houtput[0][i]->Draw("box"); 
	houtput[1][i]->Draw("samebox"); 
	houtput[2][i]->Draw("samebox"); 
	houtput[3][i]->Draw("samebox"); 
	cc->Print(TString(plotsDir)+Form("OUT_%d.gif",i));
    }
}





void
HcalLutAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(HcalLutAnalyzer);
