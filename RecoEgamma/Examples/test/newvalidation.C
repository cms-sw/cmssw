{

TString DBS_SAMPLE = gSystem->Getenv("DBS_SAMPLE") ;

TString val_ref_file_name = gSystem->Getenv("VAL_REF_FILE") ;
TString val_new_file_name = gSystem->Getenv("VAL_NEW_FILE") ;
TString val_ref_release = gSystem->Getenv("VAL_REF_RELEASE") ;
TString val_new_release = gSystem->Getenv("VAL_NEW_RELEASE") ;
TString val_analyzer = gSystem->Getenv("VAL_ANALYZER") ;

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
		<<" and the "<<val_new_release<<" histograms are in red.\n" ;
 }
else
 {
  web_page
	  <<"<p>In all plots below, "
		<<" there "<<val_ref_release<<" histograms are in blue, "
		<<" and the "<<val_new_release<<" histograms are in red.\n" ;
 }

std::ifstream histo_file("histos.txt") ;
std::string histo_name, gif_name, canvas_name ;
int scaled ;
while (histo_file>>histo_name>>scaled)
 {
  gif_name = std::string("gifs/")+histo_name+".gif" ;
  canvas_name = std::string("c")+histo_name ;
  canvas = new TCanvas(canvas_name.c_str()) ;
  canvas->SetFillColor(10) ;
	web_page<<"<br><h3>"<<histo_name<<"</h3>" ;
	
	if ( file_old != 0 )
	 {
  	histo_old = (TH1 *)file_old->Get(histo_name.c_str()) ;
		if (histo_old!=0)
		 {
  	  histo_old->SetLineColor(4) ;
      histo_old->SetLineWidth(3) ;
      histo_old->Draw() ;
		 }
		else
		 {
	    web_page<<"<br>(no such histo for "<<val_ref_release<<")" ;
		 }
	 }
	
	histo_new = (TH1 *)file_new->Get(histo_name.c_str()) ;
  histo_new->SetLineColor(2) ;
  histo_new->SetLineWidth(3) ;
	if ((scaled==1)&&(file_old!=0)&&(histo_old!=0))
   { histo_new->Scale(histo_old->GetEntries()/histo_new->GetEntries()) ; }
  histo_new->Draw("same") ;

  std::cout<<histo_name
	  <<" has "<<histo_new->GetEffectiveEntries()<<" entries"
	  <<" of mean value "<<histo_new->GetMean()
		<<std::endl ;

  canvas->SaveAs(gif_name.c_str()) ;
	web_page<<"<br><p><img class=\"image\" width=\"500\" src=\""<<gif_name<<"\">\n" ;
 }

web_page<<"\n</html>"<<std::endl ;
web_page.close() ;
 
}
