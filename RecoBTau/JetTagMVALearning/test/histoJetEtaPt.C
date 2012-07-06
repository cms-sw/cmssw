
#include <TFile.h>
#include <TH2D.h>
#include <vector>
#include <TString.h>
#include <TChain.h>
#include <TDirectory.h>

#include <stdio.h>
#include <iostream> // for std::cout and std::endl

void paint(TString a, TString b)
{
	TChain *t = new TChain(a);
	t->Add(a + "_" + b +".root");

	std::cout << "painting the histograms for: " << a + "_" + b +".root" << std::endl;

	TDirectory *dir = new TDirectory("dir", "dir");
	dir->cd();

//	histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 60, 10, 610);
//	histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 40, 4.7004803657924166, 6.404803657924166);
//	TH2D * histo = new TH2D("jets", "jets", 25, -2.5, 2.5, 20, 4.0943445622221004, 6.1943445622221004); 
	TH2D * histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 40, 4.0943445622221004, 6.1943445622221004);//original
//	TH2D * histo = new TH2D("jets", "jets", 50, -2.5, 2.5, 40, 4.0943445622221004, 7.8);
	histo->SetDirectory(dir);

	//the varexp part of the draw syntax means: draw log(jetPt+50) versus jetEta and append the existing ("+" -> avoid recreation) histogram called "jets"
	//selection is an expression with a combination of the Tree variables -> no selection applied in this case ""
	//option is the drawing option -> if option contains the string "goff", no graphics is generated.
	//fourth and fifth arguments are: Int_t nevents, Int_t firstevent
	t->Draw("log(jetPt+50):jetEta >> +jets", "", "Lego goff");
	//std::cout <<"jetPt " << log(jetPt+50) << " and jetEta " << jetEta << std::endl;


	std::cout << "saving the histograms: " << a + "_" + b +"_histo.root" << std::endl;
	TFile g(a + "_" + b +"_histo.root", "RECREATE");
	histo->SetDirectory(&g);
	delete dir;

	g.cd();
	histo->Write();
	g.Close();
}

void histoJetEtaPt()
{
  std::vector<TString> cat;
  cat.push_back(TString("CombinedSVRecoVertex"));
  cat.push_back("CombinedSVPseudoVertex");
  cat.push_back("CombinedSVNoVertex");
  std::vector<TString> types;
  types.push_back("DUSG");
  types.push_back("B_DUSG");
  types.push_back("B_C");
  types.push_back("C");
  types.push_back("B");
  for(size_t i=0;i< cat.size(); i++){
   for(size_t j=0;j< types.size(); j++){
	  paint(cat[i],types[j]);
   }
 }
}
