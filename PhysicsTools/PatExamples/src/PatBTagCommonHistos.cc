// -*- C++ -*-
//
// Package:    PatBTag
// Class:      PatBTagCommonHistos
// 
/**\class PatBTagCommonHistos PatBTagCommonHistos.cc

 Description: <Define and Fill common set of histograms depending on flavor>

 Implementation:
 
 Create a container of histograms. 
*/
//
// Original Author:  J. E. Ramirez
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatExamples/interface/PatBTagCommonHistos.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

//
// constructors and destructor
//
PatBTagCommonHistos::PatBTagCommonHistos(const edm::ParameterSet& iConfig):
  histocontainer_()
,BTagger(iConfig.getParameter< edm::ParameterSet >("BJetOperatingPoints"))
   //now do what ever initialization is needed
{
  edm::ParameterSet PatBjet_(iConfig.getParameter< edm::ParameterSet >("BjetTag"));
  BTagdisccut_      = PatBjet_.getUntrackedParameter<double>("mindiscriminatorcut",5.0);
  BTagdiscriminator_= PatBjet_.getParameter<std::string>("discriminator");
  BTagpurity_       = PatBjet_.getParameter<std::string>("purity");
}


PatBTagCommonHistos::~PatBTagCommonHistos()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PatBTagCommonHistos::Fill( edm::View<pat::Jet>::const_iterator& jet_iter, std::string flavor)
{

float isb    =jet_iter->bDiscriminator(BTagdiscriminator_);
//
// no pt cut histograms
//
if(
	fabs(jet_iter->eta()) < 2.4
	&& fabs(jet_iter->eta()) > 0
){
	//
	//Fill histos for Loose(defined in configuration file) cut
	//tagged using auxiliar function (Loose)
	//
	if(BTagger.IsbTag(*jet_iter,BTagpurity_,BTagdiscriminator_)){
		histocontainer_["jet_pt_uncorr_"+flavor+"_tagged"]->Fill(jet_iter->correctedJet("raw").pt());
		histocontainer_["jet_pt_"+flavor+"_tagged"]->Fill(jet_iter->pt());
		histocontainer_["jet_eta_"+flavor+"_tagged"]->Fill(jet_iter->eta());
	}
	//
	// Fill histos 
	//    tagged minimum discriminant cut criteria
	//
	if( isb > float(BTagdisccut_) ){
		histocontainer_["jet_pt_uncorr_"+flavor+"_taggedmincut"]->Fill(jet_iter->correctedJet("raw").pt());
		histocontainer_["jet_pt_"+flavor+"_taggedmincut"]->Fill(jet_iter->pt());
		histocontainer_["jet_eta_"+flavor+"_taggedmincut"]->Fill(jet_iter->eta());
	}
	//
	//fill taggable jets histograms (tracks in jet > 0,1,2) 
        std::map<int,std::string> tagbl;
        tagbl[0]="0";tagbl[1]="1";tagbl[2]="2";
	for (size_t i=0; i < tagbl.size(); i++)
		if( jet_iter->associatedTracks().size() > i ){
			histocontainer_["jet_pt_uncorr_"+flavor+"_taggable"+tagbl[i]]->Fill(jet_iter->correctedJet("raw").pt());
			histocontainer_["jet_pt_"+flavor+"_taggable"+tagbl[i]]->Fill(jet_iter->pt());
			histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]]->Fill(jet_iter->eta());
			if (jet_iter->pt() <  30 ){
				histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]+"_030"]->Fill(jet_iter->eta());
			}
			if (jet_iter->pt() >  30 && jet_iter->pt() < 50){
				histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]+"_3050"]->Fill(jet_iter->eta());
			}
			if (jet_iter->pt() >  50 ){
				histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]+"_50"]->Fill(jet_iter->eta());
			}
		}
	//
	// Fill histos needed for normalization
	// uncorrected pt distributions
	// corrected pt distributions
	// eta distributions
	// scatter plots for pts 
	histocontainer_["jet_pt_uncorr_"+flavor]->Fill(jet_iter->correctedJet("raw").pt());
	histocontainer_["jet_pt_"+flavor]->Fill(jet_iter->pt());
	histocontainer_["jet_eta_"+flavor]->Fill(jet_iter->eta());
	histocontainer_["tracks_in_jet_"+flavor]->Fill(jet_iter->associatedTracks().size());
	h2_["jet_scatter_pt_"+flavor]->Fill(jet_iter->pt(),jet_iter->correctedJet("raw").pt());
	for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
		histocontainer_["pt_tracks_in_jet_"+flavor]->Fill(jet_iter->associatedTracks()[itrack]->pt());
	}
	if (jet_iter->pt() <  30 ){
		histocontainer_["jet_eta_"+flavor+"030"]->Fill(jet_iter->eta());
		histocontainer_["tracks_in_jet_"+flavor+"030"]->Fill(jet_iter->associatedTracks().size());
		for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
			histocontainer_["pt_tracks_in_jet_"+flavor+"030"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
		}
		if (fabs(jet_iter->eta()) <  1.5 ){
			for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
				histocontainer_["pt_tracks_in_jet_"+flavor+"_center030"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
			}
		}
		if (fabs(jet_iter->eta()) >  1.5 && fabs(jet_iter->eta()) <  2.4){
			for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
				histocontainer_["pt_tracks_in_jet_"+flavor+"_sides030"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
			}
		}
	}
	if (jet_iter->pt() >  30 && jet_iter->pt() < 50){
		histocontainer_["jet_eta_"+flavor+"3050"]->Fill(jet_iter->eta());
		histocontainer_["tracks_in_jet_"+flavor+"3050"]->Fill(jet_iter->associatedTracks().size());
		for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
			histocontainer_["pt_tracks_in_jet_"+flavor+"3050"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
		}
		if (fabs(jet_iter->eta()) <  1.5 ){
			for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
				histocontainer_["pt_tracks_in_jet_"+flavor+"_center3050"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
			}
		}
		if (fabs(jet_iter->eta()) >  1.5 && fabs(jet_iter->eta()) <  2.4){
			for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
				histocontainer_["pt_tracks_in_jet_"+flavor+"_sides3050"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
			}
		}
	}
	if (jet_iter->pt() >  50 ){
		histocontainer_["jet_eta_"+flavor+"50"]->Fill(jet_iter->eta());
		histocontainer_["tracks_in_jet_"+flavor+"50"]->Fill(jet_iter->associatedTracks().size());
		for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
			histocontainer_["pt_tracks_in_jet_"+flavor+"50"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
		}
		if (fabs(jet_iter->eta()) <  1.5 ){
			for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
				histocontainer_["pt_tracks_in_jet_"+flavor+"_center50"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
			}
		}
		if (fabs(jet_iter->eta()) >  1.5 && fabs(jet_iter->eta()) <  2.4){
			for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
				histocontainer_["pt_tracks_in_jet_"+flavor+"_sides50"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
			}
		}
	}
	if (fabs(jet_iter->eta()) <  1.5 ){
		for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
			histocontainer_["pt_tracks_in_jet_"+flavor+"_center"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
		}
	}
	if (fabs(jet_iter->eta()) >  1.5 && fabs(jet_iter->eta()) <  2.4){
		for(size_t itrack=0;itrack < jet_iter->associatedTracks().size() ;++itrack){
			histocontainer_["pt_tracks_in_jet_"+flavor+"_sides"]->Fill(jet_iter->associatedTracks()[itrack]->pt());
		}
	}

 }//endif
}//end function
// ------------ method called once each job just before starting event loop  ------------
// ------------  This function is needed to set a group of histogram  -------------------
void 
PatBTagCommonHistos::Set(std::string flavor)
{

  const int ntptarray = 23;
  const int njptarray = 14;
  const int netaarray = 19;
  Double_t jetetabins[netaarray] = {-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5};
  Double_t jetptxbins[njptarray] = {0.,10.,20.,30., 40., 50., 60., 70., 80, 90., 100., 120., 140., 230.};
  Double_t jetpttbins[ntptarray] = {0.,1.,2.,4.,6.,8.,10.,15.,20.,25.,30., 35.,40., 45., 50., 60., 70., 80, 90., 100., 120., 140., 230.};
  edm::Service<TFileService> fs;
  std::string histoid;
  std::string histotitle;
  
  //Define histograms
  histoid = "jet_pt_"+flavor;        histotitle = flavor+" jet p_{T} [GeV/c]";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),njptarray-1,jetptxbins);
  histoid = "jet_pt_uncorr_"+flavor; histotitle = flavor+" jet uncorrected p_{T} [GeV/c]";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),njptarray-1,jetptxbins);
  histoid = "jet_eta_"+flavor;       
  histocontainer_["jet_eta_"+flavor]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
  histoid = "jet_eta_"+flavor+"030";
  histocontainer_["jet_eta_"+flavor+"030"]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
  histoid = "jet_eta_"+flavor+"3050";
  histocontainer_["jet_eta_"+flavor+"3050"]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
  histoid = "jet_eta_"+flavor+"50";
  histocontainer_["jet_eta_"+flavor+"50"]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
  histoid = "jet_pt_"+flavor+"_tagged";
  histocontainer_["jet_pt_"+flavor+"_tagged"]=fs->make<TH1D>(histoid.c_str(),"jet_tagged p_{T} [GeV/c]",njptarray-1,jetptxbins);
  histoid = "jet_pt_uncorr_"+flavor+"_tagged";
  histocontainer_["jet_pt_uncorr_"+flavor+"_tagged"]=fs->make<TH1D>(histoid.c_str(),"jet_tagged p_{T} [GeV/c]",njptarray-1,jetptxbins);
  histoid = "jet_eta_"+flavor+"_tagged";
  histocontainer_["jet_eta_"+flavor+"_tagged"]=fs->make<TH1D>(histoid.c_str(),"jet_tagged #eta",netaarray-1,jetetabins);

  //tagged min cut
  histoid = "jet_pt_"+flavor+"_taggedmincut";
  histocontainer_["jet_pt_"+flavor+"_taggedmincut"]=fs->make<TH1D>(histoid.c_str(),"jet_tagged p_{T} [GeV/c]",njptarray-1,jetptxbins);
  histoid = "jet_pt_uncorr_"+flavor+"_taggedmincut";
  histocontainer_["jet_pt_uncorr_"+flavor+"_taggedmincut"]=fs->make<TH1D>(histoid.c_str(),"jet_tagged p_{T} [GeV/c]",njptarray-1,jetptxbins);
  histoid = "jet_eta_"+flavor+"_taggedmincut";
  histocontainer_["jet_eta_"+flavor+"_taggedmincut"]=fs->make<TH1D>(histoid.c_str(),"jet_tagged #eta",netaarray-1,jetetabins);

  //#tracks per jet
  histoid = "tracks_in_jet_"+flavor;  histotitle = "traks per jet "+flavor;
  histocontainer_["tracks_in_jet_"+flavor]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),31,-0.5,30.5);
  histoid = "tracks_in_jet_"+flavor+"030"; histotitle = "traks per jet "+flavor+ "pt_{T} < 30[GeV/c]";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),31,-0.5,30.5);
  histoid = "tracks_in_jet_"+flavor+"3050"; histotitle = "traks per jet "+flavor+ "30 < pt_{T} < 50[GeV/c]";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),31,-0.5,30.5);
  histoid = "tracks_in_jet_"+flavor+"50"; histotitle = "traks per jet "+flavor+ "pt_{T} > 50[GeV/c]";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),31,-0.5,30.5);

  // pt of tracks in bins of jet pt 0-30,30-50,50
  histoid= "pt_tracks_in_jet_"+flavor; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"030"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"3050"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"50"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);

  // pt of tracks in bins of jet eta abs(eta)<1.5, 1.5<abs(eta)<2.4
  // combined with bins of jet pt
  histoid= "pt_tracks_in_jet_"+flavor+"_center"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_center030"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_center3050"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_center50"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_sides"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_sides030"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_sides3050"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);
  histoid= "pt_tracks_in_jet_"+flavor+"_sides50"; histotitle = "track p_{T} [GeV/c] "+ flavor+" jets";
  histocontainer_[histoid]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),ntptarray-1,jetpttbins);

  //taggable 0,1,2
  std::map<int,std::string> tagbl;
  tagbl[0]="0";tagbl[1]="1";tagbl[2]="2";
  for (size_t i=0; i < tagbl.size(); i++){
	histoid = "jet_pt_"+flavor+"_taggable"+tagbl[i];        histotitle = flavor+" jet_taggable p_{T} [GeV/c]";
	histocontainer_["jet_pt_"+flavor+"_taggable"+tagbl[i]]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),njptarray-1,jetptxbins);
	histoid = "jet_pt_uncorr_"+flavor+"_taggable"+tagbl[i]; histotitle = flavor+" jet_taggable p_{T} uncorr [GeV/c]";
	histocontainer_["jet_pt_uncorr_"+flavor+"_taggable"+tagbl[i]]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),njptarray-1,jetptxbins);
	histoid = "jet_eta_"+flavor+"_taggable"+tagbl[i];       histotitle = flavor+" jet_taggable #eta";
	histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]]=fs->make<TH1D>(histoid.c_str(),histotitle.c_str(),netaarray-1,jetetabins);
	histoid = "jet_eta_"+flavor+"_taggable"+tagbl[i]+"_030";
	histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]+"_030"]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
	histoid = "jet_eta_"+flavor+"_taggable"+tagbl[i]+"_3050";
	histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]+"_3050"]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
	histoid = "jet_eta_"+flavor+"_taggable"+tagbl[i]+"_50";
	histocontainer_["jet_eta_"+flavor+"_taggable"+tagbl[i]+"_50"]=fs->make<TH1D>(histoid.c_str(),"jet #eta",netaarray-1,jetetabins);
  }

  histoid = "jet_scatter_pt_"+flavor;
  h2_["jet_scatter_pt_"+flavor]=fs->make<TH2D>(histoid.c_str(),"jet p_{T} uncorr(y) vs corr(x) [GeV/c]",
                                                njptarray-1,jetptxbins,njptarray-1,jetptxbins);

}//end function Set

// ------------ method called once each job just before starting event loop  ------------
// ------------              after setting histograms                --------------------
// ------------  This function is needed to save histogram errors -----------------------
void 
PatBTagCommonHistos::Sumw2()
{
  for (std::map<std::string,TH1D*>::const_iterator ih=histocontainer_.begin();
         ih!= histocontainer_.end(); ++ih) {
      TH1D *htemp = (TH1D*) ih->second;
      htemp->Sumw2();
  }
}//end function Sumw2

