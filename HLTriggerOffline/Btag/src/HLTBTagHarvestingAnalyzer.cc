#include "HLTriggerOffline/Btag/interface/HLTBTagHarvestingAnalyzer.h"

HLTBTagHarvestingAnalyzer::HLTBTagHarvestingAnalyzer(const edm::ParameterSet& iConfig)
{
	//getParameter
	hltPathNames_			= iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");
	edm::ParameterSet mc	= iConfig.getParameter<edm::ParameterSet>("mcFlavours");
	m_mcLabels				= mc.getParameterNamesForType<std::vector<unsigned int> >();
	m_histoName				= iConfig.getParameter<std::vector<std::string> >("histoName");
	m_minTag				= iConfig.getParameter<double>("minTag");
}

HLTBTagHarvestingAnalyzer::~HLTBTagHarvestingAnalyzer()
{
}

// ------------ method called once each job just after ending the event loop  ------------
	void 
HLTBTagHarvestingAnalyzer::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter)
{
	using namespace edm;
	Exception excp(errors::LogicError);
	std::string dqmFolder_hist;

	//for each hltPath and for each flavour, do the "b-tag efficiency vs jet pt" and "b-tag efficiency vs mistag rate" plots
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) 
	{
		dqmFolder_hist = Form("HLT/BTag/Discriminator/%s",hltPathNames_[ind].c_str());
		std::string effDir = Form("HLT/BTag/Discriminator/%s/efficiency",hltPathNames_[ind].c_str());
		ibooker.setCurrentFolder(effDir);
		TH1 *den =NULL;
		TH1 *num =NULL; 
		std::map<TString,TH1F> effics;
		std::map<TString,bool> efficsOK;
		for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
			bool isOK=false;
			TString label= m_histoName.at(ind) + std::string("__"); //"JetTag__";
			TString flavour= m_mcLabels[i].c_str();
			label+=flavour;
			isOK=GetNumDenumerators(ibooker,igetter,(TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,0);
			if (isOK){
			
				//do the 'b-tag efficiency vs discr' plot
				effics[flavour]=calculateEfficiency1D(ibooker,igetter,*num,*den,(label+"_efficiency_vs_disc").Data());
				efficsOK[flavour]=isOK;
			}
			label= m_histoName.at(ind)+std::string("___");
			label+=flavour+TString("_disc_pT");
			isOK=GetNumDenumerators (ibooker,igetter,(TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,1);
			if (isOK) {
			
				//do the 'b-tag efficiency vs pT' plot
				TH1F eff=calculateEfficiency1D(ibooker,igetter,*num,*den,(label+"_efficiency_vs_pT").Data());
			}
		} /// for mc labels
		
		///save mistagrate vs b-eff plots
		if (efficsOK["b"] && efficsOK["c"])      mistagrate(ibooker,igetter,&effics["b"], &effics["c"], m_histoName.at(ind)+"_b_c_mistagrate" );
		if (efficsOK["b"] && efficsOK["light"])  mistagrate(ibooker,igetter,&effics["b"], &effics["light"], m_histoName.at(ind)+"_b_light_mistagrate" );
		if (efficsOK["b"] && efficsOK["g"])      mistagrate(ibooker,igetter,&effics["b"], &effics["g"], m_histoName.at(ind)+"_b_g_mistagrate" );
	} /// for triggers
}

bool HLTBTagHarvestingAnalyzer::GetNumDenumerators(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, std::string num,std::string den,TH1 * & ptrnum,TH1* & ptrden,int type)
{
        using namespace edm;
/*
   possible types: 
   type =0 for eff_vs_discriminator
   type =1 for eff_vs_pT ot eff_vs_Eta
 */
	MonitorElement *denME = NULL;
	MonitorElement *numME = NULL;
	denME = igetter.get(den);
	numME = igetter.get(num);
	Exception excp(errors::LogicError);
	
	if ( denME == NULL || numME == NULL ) 
	{
		excp << "Plots not found:\n";
		if(denME == NULL) excp << den << "\n";
		if(numME == NULL) excp << num << "\n";
		excp.raise();
	}
	
	if (type==0) //efficiency_vs_discr: fill "ptrnum" with the cumulative function of the DQM plots contained in "num" and "ptrden" with a flat function
	{
		TH1* numH1 = numME->getTH1();
		TH1* denH1 = denME->getTH1();
		ptrden=(TH1*)denH1->Clone("denominator");
		ptrnum=(TH1*)numH1->Clone("numerator");

		ptrnum->SetBinContent(1,numH1->Integral());
		ptrden->SetBinContent(1,numH1->Integral());
		for  (int j=2;j<=numH1->GetNbinsX();j++) {
			ptrnum->SetBinContent(j,numH1->Integral()-numH1->Integral(1,j-1));
			ptrden->SetBinContent(j,numH1->Integral());
		}
	}
	
	if (type==1) //efficiency_vs_pT: fill "ptrden" with projection of the plots contained in "den" and fill "ptrnum" with projection of the plots contained in "num", having btag>m_minTag
	{
		TH2F* numH2 = numME->getTH2F();
		TH2F* denH2 = denME->getTH2F();

		///numerator preparing
		TCutG * cutg_num= new TCutG("cutg_num",4);
		cutg_num->SetPoint(0,m_minTag,0);
		cutg_num->SetPoint(1,m_minTag,9999);
		cutg_num->SetPoint(2,1.1,9999);
		cutg_num->SetPoint(3,1.1,0);
		ptrnum = numH2->ProjectionY("numerator",0,-1,"[cutg_num]");

		///denominator preparing
		TCutG * cutg_den= new TCutG("cutg_den",4);
		cutg_den->SetPoint(0,-10.1,0);
		cutg_den->SetPoint(1,-10.1,9999);
		cutg_den->SetPoint(2,1.1,9999);
		cutg_den->SetPoint(3,1.1,0);
		ptrden = denH2->ProjectionY("denumerator",0,-1,"[cutg_den]");
		delete cutg_num;
		delete cutg_den;
	}
	return true;
}


void HLTBTagHarvestingAnalyzer::mistagrate(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, TH1F* num, TH1F* den, std::string effName ){
	//do the efficiency_vs_mistag_rate plot
	TH1F* eff;
	eff = new TH1F(effName.c_str(),effName.c_str(),100,0,1);
	eff->SetTitle(effName.c_str());
	eff->SetXTitle("b-effficiency");
	eff->SetYTitle("mistag rate");
	eff->SetOption("E");
	eff->SetLineColor(2);
	eff->SetLineWidth(2);
	eff->SetMarkerStyle(20);
	eff->SetMarkerSize(0.8);
	eff->GetYaxis()->SetRangeUser(0.001,1.001);
	eff->GetXaxis()->SetRangeUser(-0.001,1.001);
	eff->SetStats(kFALSE);
	
	//for each bin in the discr -> find efficiency and mistagrate -> put them in a plot
	for(int i=1;i<=num->GetNbinsX();i++){
		double beff=num->GetBinContent(i);
		double miseff=den->GetBinContent(i);
		double miseffErr= den->GetBinError(i);
		int binX = eff->GetXaxis()->FindBin(beff);
		if (eff->GetBinContent(binX)!=0) continue;
		eff->SetBinContent(binX,miseff);
		eff->SetBinError(binX,miseffErr);
	}
	MonitorElement *me;
	me = ibooker.book1D(effName.c_str(),eff);
	me->setEfficiencyFlag();

	delete eff;
	return;
}

TH1F  HLTBTagHarvestingAnalyzer::calculateEfficiency1D(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, TH1 & num , TH1 & den, std::string effName ){
	//calculate the efficiency as num/den ratio
	TH1F eff;
	if(num.GetXaxis()->GetXbins()->GetSize()==0){
		 eff = TH1F(effName.c_str(),effName.c_str(),num.GetXaxis()->GetNbins(),num.GetXaxis()->GetXmin(),num.GetXaxis()->GetXmax());
	}else{
		eff = TH1F(effName.c_str(),effName.c_str(),num.GetXaxis()->GetNbins(),num.GetXaxis()->GetXbins()->GetArray());
	} 
	eff.SetTitle(effName.c_str());
	eff.SetXTitle( num.GetXaxis()->GetTitle() );
	eff.SetYTitle("Efficiency");
	eff.SetOption("PE");
	eff.SetLineColor(2);
	eff.SetLineWidth(2);
	eff.SetMarkerStyle(20);
	eff.SetMarkerSize(0.8);
	eff.GetYaxis()->SetRangeUser(-0.001,1.001);
	for(int i=1;i<=num.GetNbinsX();i++){
		double d, n,err;
		d= den.GetBinContent(i);
		n= num.GetBinContent(i);
		double e;
		if(d!=0){
			e=n/d;
			err =  sqrt(e*(1-e)/d); //from binomial standard deviation
		}
		else{
			e=0;
			err=0;
		}		
		eff.SetBinContent( i, e );
		eff.SetBinError( i, err );
	}
	
	MonitorElement *me;
	me = ibooker.book1D(effName,&eff);
	me->setEfficiencyFlag();
	
	return eff;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTBTagHarvestingAnalyzer);
