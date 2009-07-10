{

TString DBS_SAMPLE = gSystem->Getenv("DBS_SAMPLE") ;

TString val_ref_file_name = gSystem->Getenv("VAL_REF_FILE") ;
TString val_new_file_name = gSystem->Getenv("VAL_NEW_FILE") ;
TString val_ref_release = gSystem->Getenv("VAL_REF_RELEASE") ;
TString val_new_release = gSystem->Getenv("VAL_NEW_RELEASE") ;
TString val_analyzer = gSystem->Getenv("VAL_ANALYZER") ;

// style:
TStyle *eleStyle = new TStyle("eleStyle","Style for electron validation");
eleStyle->SetCanvasBorderMode(0); 
eleStyle->SetCanvasColor(kWhite);
eleStyle->SetCanvasDefH(600); 
eleStyle->SetCanvasDefW(800); 
eleStyle->SetCanvasDefX(0);  
eleStyle->SetCanvasDefY(0);
eleStyle->SetPadBorderMode(0); 
eleStyle->SetPadColor(kWhite);
eleStyle->SetPadGridX(false);
eleStyle->SetPadGridY(false);
eleStyle->SetGridColor(0);
eleStyle->SetGridStyle(3);
eleStyle->SetGridWidth(1);
eleStyle->SetOptStat(1);
eleStyle->SetPadTickX(1); 
eleStyle->SetPadTickY(1);
eleStyle->SetHistLineColor(1);
eleStyle->SetHistLineStyle(0);
eleStyle->SetHistLineWidth(2);
eleStyle->SetEndErrorSize(2);
eleStyle->SetErrorX(0.);
eleStyle->SetOptStat(1);
eleStyle->SetTitleColor(1, "XYZ");
eleStyle->SetTitleFont(42, "XYZ");
eleStyle->SetTitleXOffset(1.0);
eleStyle->SetTitleYOffset(1.0);
eleStyle->SetLabelOffset(0.005, "XYZ");
eleStyle->SetTitleSize(0.05, "XYZ");
eleStyle->SetTitleFont(22,"X");
eleStyle->SetTitleFont(22,"Y");
eleStyle->SetHistLineWidth(2);
eleStyle->SetPadBottomMargin(0.13);
eleStyle->SetPadLeftMargin(0.15);
eleStyle->SetMarkerStyle(21);
eleStyle->SetMarkerSize(0.8);

eleStyle->cd(); 
  
gROOT->ForceStyle();

TFile * file_old = 0 ;
if ( val_ref_file_name != "" )
 { file_old = TFile::Open(val_ref_file_name) ; }
TFile * file_new = 0 ;
if ( val_new_file_name != "" )
 { file_new = TFile::Open(val_new_file_name) ; }

TCanvas * canvas ;
TH1 * histo_old, * histo_new ;
Double_t nold, nnew ;

std::ofstream web_page("validation.html") ;
web_page
  <<"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
  <<"<html>\n"
  <<"<head>\n"
  <<"<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n"
  <<"<title>"<<val_new_release<<" vs "<<val_ref_release<<"<br>"<<DBS_SAMPLE<<"</title>\n"
  <<"</head>\n"
  <<"<h1>"<<val_new_release<<" vs "<<val_ref_release<<"<br>"<<DBS_SAMPLE<<"</h1>\n"
  <<"<p>The following plots were made using analyzer "
  <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/RecoEgamma/Examples/plugins/"<<val_analyzer<<".cc\">"
  <<"RecoEgamma/Examples/plugins/"<<val_analyzer<<".cc"
  <<"</a>\n"
  <<"and configuration "
  <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/RecoEgamma/Examples/test/"<<val_analyzer<<"_cfg.py\">"
  <<"RecoEgamma/Examples/test/"<<val_analyzer<<"_cfg.py"
  <<"</a>, with dataset "<<DBS_SAMPLE<<" as input.\n"
  <<"<p>The script used to make the plots is "
  <<"<a href=\"newvalidation.C\">here</a>.\n"
  <<"<p>The list of histograms is <a href=\"histos.txt\">here</a>.\n" ;

if (file_old==0)
 {
  web_page
	  <<"<p>In all plots below, "
		<<" there was no "<<val_ref_release<<" histograms to compare with, "
		<<" and the "<<val_new_release<<" histograms are in red.</p>\n" ;
 }
else
 {
  web_page
	  <<"<p>In all plots below, "
		<<" there "<<val_ref_release<<" histograms are in blue, "
		<<" and the "<<val_new_release<<" histograms are in red.</p>\n" ;
 }

web_page<<"<br><br><hr>" ;
std::ifstream histo_file("histos.txt") ;
std::string histo_name, gif_name, canvas_name ;
int scaled, log, err ;
int divide;
std::string num, denom;
while (histo_file>>histo_name>>scaled>>log>>err>>divide>>num>>denom)
 {
  gif_name = std::string("gifs/")+histo_name+".gif" ;
  canvas_name = std::string("c")+histo_name ;
  canvas = new TCanvas(canvas_name.c_str()) ;
  canvas->SetFillColor(10) ;
  web_page<<"<br><br><p>" ;

  if ( file_old != 0 )
   {
    histo_old = (TH1 *)file_old->Get(histo_name.c_str()) ;
    if (histo_old!=0)
     {
      histo_old->SetLineColor(4) ;
      histo_old->SetLineWidth(3) ;
      if (divide==0) {
        histo_old->Draw("hist") ;
      } else {
	// special for efficiencies
	TH1F *h_num = (TH1F *)file_old->Get(num.c_str()) ;
        TH1F *h_res = (TH1F*)h_num->Clone("res");
        h_res->Reset();
	TH1F *h_denom = (TH1F *)file_old->Get(denom.c_str()) ;
        std::cout << "DIVIDING "<< num.c_str() << " by " << denom.c_str() << std::endl;
	h_res->Divide(h_num,h_denom,1,1,"b");
        h_res->GetXaxis()->SetTitle(h_num->GetXaxis()->GetTitle());
        h_res->GetYaxis()->SetTitle(h_num->GetYaxis()->GetTitle());
        h_res->SetLineColor(4) ;
        h_res->SetLineWidth(3) ;
        h_res ->Draw("hist") ;    
      }
     }
    else
     {
      web_page<<"No <b>"<<histo_name<<"</b> for "<<val_ref_release<<".<br>" ;
     }
   }

  histo_new = (TH1 *)file_new->Get(histo_name.c_str()) ;
  if (histo_new!=0)
   {
    if (log==1)canvas->SetLogy(1);
    histo_new->SetLineColor(2) ;
    histo_new->SetMarkerColor(2) ;
    histo_new->SetLineWidth(3) ;
    if ((scaled==1)&&(file_old!=0)&&(histo_old!=0)&&(histo_new->GetEntries()!=0))
     { if (histo_old!=0) histo_new->Scale(histo_old->GetEntries()/histo_new->GetEntries()) ; }   
    if (divide==0) {
    if (err==1) {
      if (histo_old!=0) histo_new->Draw("same E1 P") ;
      else histo_new->Draw("E1 P") ; 
    } else {
      if (histo_old!=0) histo_new->Draw("same hist") ;
      else histo_new->Draw("hist") ;
    }  
    } else {
      // special for efficiencies
      TH1F *h_num = (TH1 *)file_new->Get(num.c_str()) ;
      TH1F *h_res = (TH1F*)h_num->Clone("res");
      TH1F *h_denom = (TH1 *)file_new->Get(denom.c_str()) ;
      h_res->Divide(h_num,h_denom,1,1,"b");
      h_res->GetXaxis()->SetTitle(h_num->GetXaxis()->GetTitle());
      h_res->GetYaxis()->SetTitle(h_num->GetYaxis()->GetTitle());
      h_res->SetLineColor(2) ;
      h_res->SetMarkerColor(2) ;
      h_res->SetLineWidth(3) ;
      h_res ->Draw("same E1 P") ;      
    }
    std::cout<<histo_name
      <<" has "<<histo_new->GetEffectiveEntries()<<" entries"
      <<" of mean value "<<histo_new->GetMean()
      <<std::endl ;
    canvas->SaveAs(gif_name.c_str()) ;
	web_page<<"<img class=\"image\" width=\"500\" src=\""<<gif_name<<"\"><br>" ;
   }
  else if ((file_old!=0)&&(histo_old!=0))
   {
    std::cout<<histo_name<<" NOT FOUND"<<std::endl ;
    web_page<<"<br>(no such histo for "<<val_new_release<<")" ;
    canvas->SaveAs(gif_name.c_str()) ;
	web_page<<"<img class=\"image\" width=\"500\" src=\""<<gif_name<<"\"><br>" ;
   }
  else
   {
    web_page<<"No <b>"<<histo_name<<"</b> for "<<val_new_release<<".<br>" ;
   }
  web_page<<"</p>\n" ;
 }

// cumulated efficiencies

web_page<<"\n</html>"<<std::endl ;
web_page.close() ;

}
