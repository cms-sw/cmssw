#include "HLTriggerOffline/Btag/interface/HLTBTagHarvestingAnalyzer.h"

HLTBTagHarvestingAnalyzer::HLTBTagHarvestingAnalyzer(const edm::ParameterSet& iConfig)
{
	//getParameter
	hltPathNames_			= iConfig.getParameter< std::vector<std::string> > ("HLTPathNames");
	edm::ParameterSet mc	= iConfig.getParameter<edm::ParameterSet>("mcFlavours");
	m_mcLabels				= mc.getParameterNamesForType<std::vector<unsigned int> >();
	m_histoName				= iConfig.getParameter<std::vector<std::string> >("histoName");
	m_minTag				= iConfig.getParameter<double>("minTag");

	// DQMStore services
	dqm = edm::Service<DQMStore>().operator->();
}

HLTBTagHarvestingAnalyzer::~HLTBTagHarvestingAnalyzer()
{
	// do anything here that needs to be done at desctruction time
	// (e.g. close files, deallocate resources etc.)
}

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
	Exception excp(errors::LogicError);
	if (! dqm) {
		excp << "DQM is not ready"; 
		excp.raise();
	}
	std::string dqmFolder_hist;

	//for each hltPath and for each flavour, do the "b-tag efficiency vs jet pt" and "b-tag efficiency vs mistag rate" plots
	for (unsigned int ind=0; ind<hltPathNames_.size();ind++) 
	{
		dqmFolder_hist = Form("HLT/BTag/%s",hltPathNames_[ind].c_str());
		std::string effDir = Form("HLT/BTag/%s/efficiency",hltPathNames_[ind].c_str());
		dqm->setCurrentFolder(effDir);
		TH1 *den =NULL;
		TH1 *num =NULL; 
		map<TString,TH1F*> effics;
		map<TString,bool> efficsOK;
		for (unsigned int i = 0; i < m_mcLabels.size(); ++i)
		{
			bool isOK=false;
			TString label= m_histoName.at(ind) + string("__"); //"JetTag__";
			TString flavour= m_mcLabels[i].c_str();
			label+=flavour;
			isOK=GetNumDenumerators((TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,0);
			if (isOK){
			
				//do the 'b-tag efficiency vs discr' plot
				effics[flavour]=calculateEfficiency1D(num,den,(label+"_efficiency_vs_disc").Data());
				efficsOK[flavour]=isOK;
			}
			label= m_histoName.at(ind)+string("___");
			label+=flavour+TString("_disc_pT");
			isOK=GetNumDenumerators ((TString(dqmFolder_hist)+"/"+label).Data(),(TString(dqmFolder_hist)+"/"+label).Data(),num,den,1);
			if (isOK) {
			
				//do the 'b-tag efficiency vs pT' plot
				TH1F * eff=calculateEfficiency1D(num,den,(label+"_efficiency_vs_pT").Data());
				delete eff;
			}
		} /// for mc labels
		
		///save mistagrate vs b-eff plots
		if (efficsOK["b"] && efficsOK["c"])      mistagrate(effics["b"], effics["c"], m_histoName.at(ind)+"_b_c_mistagrate" );
		if (efficsOK["b"] && efficsOK["light"])  mistagrate(effics["b"], effics["light"], m_histoName.at(ind)+"_b_light_mistagrate" );
		if (efficsOK["b"] && efficsOK["g"])      mistagrate(effics["b"], effics["g"], m_histoName.at(ind)+"_b_g_mistagrate" );
	} /// for triggers
}

bool HLTBTagHarvestingAnalyzer::GetNumDenumerators(string num,string den,TH1 * & ptrnum,TH1* & ptrden,int type)
{
/*
   possible types: 
   type =0 for eff_vs_discriminator
   type =1 for eff_vs_pT ot eff_vs_Eta
 */
	MonitorElement *denME = NULL;
	MonitorElement *numME = NULL;
	denME = dqm->get(den);
	numME = dqm->get(num);
	if(denME==0 || numME==0){
		cout << "Could not find MEs: "<<den<<endl;
		return false;
	} else {
		cout << "found MEs: "<<den<<endl;
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


void HLTBTagHarvestingAnalyzer::mistagrate( TH1F* num, TH1F* den, string effName ){
	//do the efficiency_vs_mistag_rate plot
	TH1F* eff;
	eff = new TH1F(effName.c_str(),effName.c_str(),1000,0,1);
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
	dqm->book1D(effName.c_str(),eff);
	delete eff;

	return;
}

TH1F*  HLTBTagHarvestingAnalyzer::calculateEfficiency1D( TH1* num, TH1* den, string effName ){
	//calculate the efficiency as num/den ratio
	TH1F* eff;
	std::cout<<"Efficiency name: "<<effName<<std::endl;
	if(num->GetXaxis()->GetXbins()->GetSize()==0){
		eff = new TH1F(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXmin(),num->GetXaxis()->GetXmax());
	}else{
		eff = new TH1F(effName.c_str(),effName.c_str(),num->GetXaxis()->GetNbins(),num->GetXaxis()->GetXbins()->GetArray());
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
	for(int i=1;i<=num->GetNbinsX();i++){
		double d, n;
		d= den->GetBinContent(i);
		n= num->GetBinContent(i);
		double e;
		if(d!=0)	e=n/d;
		else		e=0;
		double err = sqrt(e*(1-e)/d); //from binomial standard deviation
		eff->SetBinContent( i, e );
		eff->SetBinError( i, err );
	}
	dqm->book1D(effName,eff);
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


