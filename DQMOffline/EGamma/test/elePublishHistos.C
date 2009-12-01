
//======================================================================
// This script generate a web page from a ROOT file containing histograms.
// It expects the histograms being split among one level of subdirectories.
// It is configured through those environment variables:
//
// $PUB_INPUT_FILE : name of the input ROOT file.
// $PUB_INPUT_FOLDER : the top directory within the ROOT file.
// $PUB_OUTPUT_DIR : name of the output directory.
// $PUB_TITLE : web page title.
// $PUB_COMMENT : text to be inserted at the top of the web page.
//======================================================================

int elePublishHistos()
 {
  TString pub_input_file = gSystem->Getenv("PUB_INPUT_FILE") ;
  TString pub_input_folder = gSystem->Getenv("PUB_INPUT_FOLDER") ;
  TString pub_output_dir = gSystem->Getenv("PUB_OUTPUT_DIR") ;
  TString pub_title = gSystem->Getenv("PUB_TITLE") ;
  TString pub_comment = gSystem->Getenv("PUB_COMMENT") ;
  
  // prepare unix output directories
  if (gSystem->mkdir(pub_output_dir,kTRUE)<0)
   { std::cerr<<"Failed to create "<<pub_output_dir<<std::endl ; exit(1) ; }
  else
   { std::cout<<"Outputdir: "<<pub_output_dir<<std::endl ; }

  // style
  TStyle *eleStyle = new TStyle("eleStyle","Style for electron dqm offline");
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
  
  // web page header
  std::ofstream web_page((pub_output_dir+"/index.html").Data()) ;
  web_page
    <<"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
    <<"<html>\n"
    <<"<head>\n"
    <<"<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n"
    <<"<title>"<<pub_title<<"</title>\n"
    <<"</head>\n"
    <<"<h1><a href=\"../\">"<<pub_title<<"</h1>\n" ;
  web_page<<"<p>"<<pub_comment ;
  web_page
    <<" Plots below were made using configuration "
    <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/DQMOffline/EGamma/test/ElectronAnalyzer_cfg.py\">"
    <<"DQMOffline/EGamma/test/ElectronAnalyzer_cfg.py</a> and "
    <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/DQMOffline/EGamma/test/ElectronOfflineClient_cfg.py\">"
    <<"DQMOffline/EGamma/test/ElectronOfflineClient_cfg.py</a>." ;
  web_page<<"</p>\n" ;

  web_page<<"\n</html>"<<std::endl ;
  web_page.close() ;
  exit(0) ;

TString val_new_file_url ;
TString file_new_dir = internal_path  ;
TFile * file_new = 0 ;
if ( val_new_file_name != "" )
 {
  file_new = TFile::Open(val_new_file_name) ;
  if (file_new!=0)
   {
    std::cout<<"open "<<val_new_file_name<<std::endl ;
    if (val_new_file_name.BeginsWith(val_web)==kTRUE)
     {
      val_new_file_url = val_new_file_name ;
      val_new_file_url.Remove(0,val_web.Length()) ;
      val_new_file_url.Prepend(val_web_url) ;
     }
    if (file_new->cd(internal_path)!=kTRUE)
     {
      std::cerr<<"Failed move to: "<<internal_path<<std::endl ;
      file_new_dir = "" ;
     }
    else
     {
      std::cerr<<"cd "<<internal_path<<std::endl ;
      file_new->cd() ;
     }
   }
  else
   { std::cerr<<"Failed to open: "<<val_new_file_name<<std::endl ; }
 }

TCanvas * canvas ;
TH1 * histo_ref, * histo_new ;


std::string histo_path, canvas_name ;
TString histo_name, gif_name, gif_path, short_histo_name, num_ref, denom_ref ;
int scaled, log, err ;
int divide;
std::string num, denom, cat ;
int eol ; // end of line
int eoc ; // enf of category

std::ifstream histo_file1(histos_path.c_str()) ;
web_page
  <<"<br><table border=\"1\" cellpadding=\"5\" width=\"100%\">"
  <<"<tr valign=\"top\"><td width=\"20%\">\n" ;
int cat_num = 0 ;

cat = "" ;
do
 {
  std::getline(histo_file1,cat) ;
 } while (cat.empty()) ;

web_page<<"<b>"<<cat<<"</b><br><br>" ;

while (histo_file1>>histo_path>>scaled>>log>>err>>divide>>num>>denom>>eol>>eoc)
 {
  histo_name = histo_path ;
  Ssiz_t pos = histo_name.Last('/') ;
  if (pos!=kNPOS) histo_name.Remove(0,pos+1) ;
  short_histo_name = histo_name ;
  short_histo_name.Remove(0,2) ;
  web_page<<"<a href=\"#"<<short_histo_name<<"\">"<<short_histo_name<<"</a><br>\n" ;
  if (eoc)
   {
    cat_num++ ;
    if ((cat_num%5)==0)
     { web_page<<"<br></td></tr>\n<tr valign=\"top\"><td width=\"20%\">" ; }
    else
     { web_page<<"<br></td><td width=\"20%\">\n" ; }
    cat = "" ;
    do
     {
      std::getline(histo_file1,cat) ;
     } while (cat.empty()) ;
    web_page<<"<b>"<<cat<<"</b><br><br>" ;
   }
 }
web_page<<"<br></td></tr></table>\n" ;
histo_file1.close() ;

web_page<<"<br><br><table cellpadding=\"5\"><tr valign=\"top\"><td>\n" ;
std::ifstream histo_file2(histos_path.c_str()) ;

cat = "" ;
do
 {
  std::getline(histo_file2,cat) ;
 } while (cat.empty()) ;

while (histo_file2>>histo_path>>scaled>>log>>err>>divide>>num>>denom>>eol>>eoc)
 {
  histo_name = histo_path ;
  Ssiz_t pos = histo_name.Last('/') ;
  if (pos!=kNPOS) histo_name.Remove(0,pos+1) ;
  short_histo_name = histo_name ;
  short_histo_name.Remove(0,2) ;
  
  gif_name = "gifs/" ;
  gif_name += histo_name ;
  gif_name += ".gif" ;
  gif_path = val_web_path ;
  gif_path += "/" ;
  gif_path += gif_name ;
  canvas_name = std::string("c")+histo_name.Data() ;
  canvas = new TCanvas(canvas_name.c_str()) ;
  canvas->SetFillColor(10) ;
  
  web_page<<"<a id=\""<<short_histo_name<<"\" name=\""<<short_histo_name<<"\"></a>" ;

  if ( file_ref != 0 )
   {
    if (file_ref_dir.IsNull()) histo_ref = (TH1 *)file_ref->Get(histo_name) ;
    else histo_ref = (TH1 *)file_ref->Get(file_ref_dir+histo_path.c_str()) ;
	
    if (histo_ref!=0)
     {
      histo_ref->SetLineColor(4) ;
      histo_ref->SetLineWidth(3) ;
      if (divide==0)
       { histo_ref->Draw("hist") ; }
      else
       {
  	    num_ref = num ;
  		  denom_ref = denom ;
  	    if (file_ref_dir.IsNull())
  	     {
    		  pos = num_ref.Last('/') ;
    		  if (pos!=kNPOS) num_ref.Remove(0,pos+1) ;
    		  pos = denom_ref.Last('/') ;
    		  if (pos!=kNPOS) denom_ref.Remove(0,pos+1) ;
    		 }
		 
        // special for efficiencies
        TH1F *h_num = (TH1F *)file_ref->Get(file_ref_dir+num_ref) ;
        TH1F *h_res = (TH1F*)h_num->Clone("res");
        //h_res->Reset();
        TH1F *h_denom = (TH1F *)file_ref->Get(file_ref_dir+denom_ref) ;
        std::cout << "DIVIDING OLD "<< num_ref << " by " << denom_ref << std::endl;
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
      web_page<<"No <b>"<<histo_path<<"</b> for "<<val_ref_release<<".<br>" ;
     }
   }

  gErrorIgnoreLevel = kWarning ;

  histo_new = (TH1 *)file_new->Get(file_new_dir+histo_path.c_str()) ;
  if (histo_new!=0)
   {
    if (log==1) canvas->SetLogy(1);
    histo_new->SetLineColor(2) ;
    histo_new->SetMarkerColor(2) ;
    histo_new->SetLineWidth(3) ;
    if ((scaled==1)&&(file_ref!=0)&&(histo_ref!=0)&&(histo_new->GetEntries()!=0))
     { histo_new->Scale(histo_ref->GetEntries()/histo_new->GetEntries()) ; }
    if (divide==0)
     {
      if (err==1)
       {
        if (histo_ref!=0) histo_new->Draw("same E1 P") ;
        else histo_new->Draw("E1 P") ;
       }
      else
       {
        if (histo_ref!=0) histo_new->Draw("same hist") ;
        else histo_new->Draw("hist") ;
       }
     }
    else
     {
      // special for efficiencies
      TH1F *h_num = (TH1 *)file_new->Get(file_new_dir+num.c_str()) ;
      TH1F *h_res = (TH1F*)h_num->Clone("res");
      TH1F *h_denom = (TH1 *)file_new->Get(file_new_dir+denom.c_str()) ;
      std::cout << "DIVIDING NEW "<< num.c_str() << " by " << denom.c_str() << std::endl;
      h_res->Divide(h_num,h_denom,1,1,"b");
      h_res->GetXaxis()->SetTitle(h_num->GetXaxis()->GetTitle());
      h_res->GetYaxis()->SetTitle(h_num->GetYaxis()->GetTitle());
      h_res->SetLineColor(2) ;
      h_res->SetMarkerColor(2) ;
      h_res->SetLineWidth(3) ;
      if (err==1) h_res ->Draw("same E1 P") ;
      else  h_res ->Draw("same hist") ;
     }
    std::cout<<histo_name
      <<" has "<<histo_new->GetEffectiveEntries()<<" entries"
      <<" of mean value "<<histo_new->GetMean()
      <<std::endl ;
    canvas->SaveAs(gif_path.Data()) ;
	web_page<<"<a href=\""<<gif_name<<"\"><img border=\"0\" class=\"image\" width=\"500\" src=\""<<gif_name<<"\"></a><br>" ;
   }
  else if ((file_ref!=0)&&(histo_ref!=0))
   {
    std::cout<<histo_path<<" NOT FOUND"<<std::endl ;
    web_page<<"<br>(no such histo for "<<val_new_release<<")" ;
    canvas->SaveAs(gif_path.Data()) ;
	web_page<<"<a href=\""<<gif_name<<"\"><img border=\"0\" class=\"image\" width=\"500\" src=\""<<gif_name<<"\"></a><br>" ;
   }
  else
   {
    web_page<<"No <b>"<<histo_path<<"</b> for "<<val_new_release<<".<br>" ;
   }
  if (eol)
   { web_page<<"</td></tr>\n<tr valign=\"top\"><td>" ; }
  else
   { web_page<<"</td><td>" ; }
  if (eoc)
   {
	cat = "" ;
    do
     {
      std::getline(histo_file2,cat) ;
     } while (cat.empty()) ;
   }
 }
histo_file2.close() ;
web_page<<"</td></tr></table>\n" ;

// cumulated efficiencies

web_page<<"\n</html>"<<std::endl ;
web_page.close() ;

}
