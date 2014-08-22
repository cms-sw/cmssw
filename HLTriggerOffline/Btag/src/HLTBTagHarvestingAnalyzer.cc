#include "HLTriggerOffline/Btag/interface/HLTBTagHarvestingAnalyzer.h"
#include "TCutG.h"
#include <cassert>


//
// constructors and destructor
//
HLTBTagHarvestingAnalyzer::HLTBTagHarvestingAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
   hltPathNames_        = iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");

   // DQMStore services   
   dqm = edm::Service<DQMStore>().operator->();
  edm::ParameterSet mc = iConfig.getParameter<edm::ParameterSet>("mcFlavours");
  m_mcLabels = mc.getParameterNamesForType<std::vector<unsigned int> >();
  minTags=iConfig.getParameter< std::vector<double> >("minTags");
  maxTag=iConfig.getParameter<double>("maxTag");
}


HLTBTagHarvestingAnalyzer::~HLTBTagHarvestingAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTBTagHarvestingAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
HLTBTagHarvestingAnalyzer::beginJob()
{   
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTBTagHarvestingAnalyzer::endJob() 
{
    using namespace edm;

	std::cout<<"HLTBTagHarvestingAnalyzer::endJob"<<std::endl;

    
   assert(hltPathNames_.size()== minTags.size());



  	Exception excp(errors::LogicError);


	if (! dqm) {  excp << "DQM is not ready";            excp.raise(); }

   std::string dqmFolder_hist;
   for (unsigned int ind=0; ind<hltPathNames_.size();ind++) {



	  dqmFolder_hist = Form("HLT/BTag/%s",hltPathNames_[ind].c_str());

      std::string effDir = Form("HLT/BTag/%s/efficiency",hltPathNames_[ind].c_str());
      dqm->setCurrentFolder(effDir);
//      cout<<"Eff dir "<<effDir<<endl;


/*    MonitorElement *denME = NULL;
    MonitorElement *numME = NULL;
*/ 	

    TH1 *den =NULL;
    TH1 *num =NULL; 

		
/// 1D and 2D for L25 discriminator
	map<TString,TProfile*> efficsL25;
	map<TString,bool> efficsL25OK;

	  for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
			 bool isOK=false;
		     TString label="JetTag_L25_";
		     TString flavour= m_mcLabels[i].c_str();
		     label+=flavour;


		//  std::cout<<"Calculate Effl25 for "<<label<<std::endl;
		    isOK=GetNumDenumerators ((TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,0,minTags[ind],maxTag);
		    if (isOK)   efficsL25[flavour]=calculateEfficiency1D(num,den,(label+"_efficiency_vs_disc").Data()); 
            efficsL25OK[flavour] = isOK;

   
		    label="JetTag_L25_";
		    label+=flavour+TString("_disc_pT");
    
		    isOK=GetNumDenumerators ((TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,1,minTags[ind],maxTag);
			if (isOK) {
				    TProfile * eff=calculateEfficiency1D(num,den,(label+"_efficiency_vs_pT").Data());
				    delete eff;
			}
		} /// for mc labels

///save mistagrate vs b-eff plots

	if (efficsL25OK["b"] && efficsL25OK["c"]) mistagrate(efficsL25["b"], efficsL25["c"], "L25_b_c_mistagrate" ); 
	if (efficsL25OK["b"] && efficsL25OK["light"]) mistagrate(efficsL25["b"], efficsL25["light"], "L25_b_light_mistagrate" ); 
	if (efficsL25OK["b"] && efficsL25OK["g"]) mistagrate(efficsL25["b"], efficsL25["g"], "L25_b_g_mistagrate" ); 

/// 1D && 2D for L3 discriminator

	map<TString,TProfile*> efficsL3;
    map<TString,bool> efficsL3OK;

	  for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
             bool isOK=false;
		     TString label="JetTag_L3_";
		     TString flavour= m_mcLabels[i].c_str();
		     label+=flavour;


			   isOK=GetNumDenumerators ((TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,0,minTags[ind],maxTag);
		       if (isOK) efficsL3[flavour]=calculateEfficiency1D(num,den,(label+"_efficiency_vs_disc").Data());
               efficsL3OK[flavour]=isOK;

			   label="JetTag_L3_";
			   label+=flavour+TString("_disc_pT");

			   isOK=GetNumDenumerators ((TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,1,minTags[ind],maxTag);
				if (isOK) {
	    		   TProfile * eff=calculateEfficiency1D(num,den,(label+"_efficiency_vs_pT").Data());
				    delete eff;
				}
    } /// for mc labels

///save mistagrate vs b-eff plots
    if (efficsL3OK["b"] && efficsL3OK["c"])      mistagrate(efficsL3["b"], efficsL3["c"], "L3_b_c_mistagrate" );
    if (efficsL3OK["b"] && efficsL3OK["light"])  mistagrate(efficsL3["b"], efficsL3["light"], "L3_b_light_mistagrate" );
    if (efficsL3OK["b"] && efficsL3OK["g"])      mistagrate(efficsL3["b"], efficsL3["g"], "L3_b_g_mistagrate" );


   } /// for triggers
 }

/*
		possible types: 

			type =0 for eff_vs_discriminator
			type =1 for eff_vs_pT ot eff_vs_Eta
*/
bool HLTBTagHarvestingAnalyzer::GetNumDenumerators(string num,string den,TH1 * & ptrnum,TH1* & ptrden,int type,double minTag, double maxTag)
{

    MonitorElement *denME = NULL;
    MonitorElement *numME = NULL;
       denME = dqm->get(den);
       numME = dqm->get(num);
if(denME==0 || numME==0){
    cout << "Could not find MEs: "<<den<<endl;
    cout << "Could not find MEs: "<<num<<endl;
    return false;
  } else {
    cout << "found MEs: "<<den<<endl;

	}

if (type==1)
{
	TH2F* numH2 = numME->getTH2F();
	TH2F* denH2 = denME->getTH2F();

///numerator preparing
TCutG * cutg_num= new TCutG("cutg_num",4);
cutg_num->SetPoint(0,minTag,0.);
cutg_num->SetPoint(1,minTag,350.);
cutg_num->SetPoint(2,maxTag,350.);
cutg_num->SetPoint(3,maxTag,0.);
ptrnum = numH2->ProjectionY("numerator",0,-1,"[cutg_num]");

///denominator preparing
TCutG * cutg_den= new TCutG("cutg_den",4);
cutg_den->SetPoint(0,-1000.,20);
cutg_den->SetPoint(1,-1000.,550);
cutg_den->SetPoint(2,1000.,550);
cutg_den->SetPoint(3,1000.,20);
ptrden = denH2->ProjectionY("denumerator",0,-1,"[cutg_den]");

delete cutg_num;
delete cutg_den;

}


if (type==0)
{

	TH1* numH1 = numME->getTH1();
	TH1* denH1 = denME->getTH1();

//    cout<<"numH1="<<numH1<<endl;
    ptrden=(TH1*)denH1->Clone("denominator");
    ptrnum=(TH1*)numH1->Clone("numerator");

    ptrnum->SetBinContent(1,numH1->Integral());
    ptrden->SetBinContent(1,numH1->Integral());
    for  (int j=2;j<=numH1->GetNbinsX();j++) {
//	std::cout<<"1Bin #"<<j<<" content "<<numH1->Integral()-numH1->Integral(1,j-1)<<std::endl;
//	std::cout<<"2Bin #"<<j<<" content "<<numH1->Integral()<<std::endl;
         ptrnum->SetBinContent(j,numH1->Integral()-numH1->Integral(1,j-1));
         ptrden->SetBinContent(j,numH1->Integral());
        }


}
return true;
}


void HLTBTagHarvestingAnalyzer::mistagrate( TProfile* num, TProfile* den, string effName ){

  TProfile* eff;

    eff = new TProfile(effName.c_str(),effName.c_str(),100,0,1);


  eff->SetTitle(effName.c_str());
  eff->SetXTitle("b-effficiency");
  eff->SetYTitle("mistag rate");
  eff->SetOption("PE");
  eff->SetLineColor(2);
  eff->SetLineWidth(2);
  eff->SetMarkerStyle(20);
  eff->SetMarkerSize(0.8);
  eff->GetYaxis()->SetRangeUser(0.001,1.001);
  eff->GetXaxis()->SetRangeUser(-0.001,1.001);
  eff->SetStats(kFALSE);
  for(int i=1;i<=num->GetNbinsX();i++){
  double beff=num->GetBinContent(i);
//  double beffErr=num->GetBinError(i);
  double miseff=den->GetBinContent(i);
  double miseffErr=den->GetBinError(i);
  int binX = eff->GetXaxis()->FindBin(beff);
  if (eff->GetBinEntries(binX)!=0) continue;
  eff->SetBinContent(binX,miseff);
  eff->SetBinError(binX,miseffErr);
  eff->SetBinEntries( binX, 1 );


 }
  dqm->bookProfile(effName,eff);
  delete eff;

return;

}

TProfile*  HLTBTagHarvestingAnalyzer::calculateEfficiency1D( TH1* num, TH1* den, string effName ){
  TProfile* eff;

 std::cout<<"Efficiency name: "<<effName<<std::endl;

  if(num->GetXaxis()->GetXbins()->GetSize()==0){
    eff = new TProfile(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXmin(),num->GetXaxis()->GetXmax());
  }else{
    eff = new TProfile(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXbins()->GetArray());
  }
 
  eff->SetTitle(effName.c_str());
  eff->SetXTitle( num->GetXaxis()->GetTitle() );
  eff->SetYTitle("Efficiency");
  eff->SetOption("PE");
  eff->SetLineColor(2);
  eff->SetLineWidth(2);
  eff->SetMarkerStyle(20);
  eff->SetMarkerSize(0.8);
  eff->GetYaxis()->SetRangeUser(-0.001,1.001);
  eff->SetStats(kFALSE);

  for(int i=1;i<=num->GetNbinsX();i++){
    double e, low, high;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
    if (int(den->GetBinContent(i))>0.) e= double(num->GetBinContent(i))/double(den->GetBinContent(i));
    else e=0.;
//    cout<<"bin "<<i<<" eff "<<e<<endl;
    low=TEfficiency::Wilson((double)den->GetBinContent(i),(double)num->GetBinContent(i),0.683,false);
    high=TEfficiency::Wilson((double)den->GetBinContent(i),(double)num->GetBinContent(i),0.683,true);
#else    
    Efficiency( (double)num->GetBinContent(i), (double)den->GetBinContent(i), 0.683, e, low, high );
#endif
    
    double err = e-low>high-e ? e-low : high-e;
    //here is the trick to store info in TProfile:
    eff->SetBinContent( i, e );
    eff->SetBinEntries( i, 1 );
    eff->SetBinError( i, sqrt(e*e+err*err) );


  }
  dqm->bookProfile(effName,eff);
//  delete eff;
return eff;
}



// ------------ method called when starting to processes a run  ------------
void 
HLTBTagHarvestingAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& )
{
}

// ------------ method called when ending the processing of a run  ------------
void 
HLTBTagHarvestingAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
HLTBTagHarvestingAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const & , edm::EventSetup const & )
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
HLTBTagHarvestingAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HLTBTagHarvestingAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTBTagHarvestingAnalyzer);
