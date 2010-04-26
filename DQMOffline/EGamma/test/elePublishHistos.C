
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

  // prepare unix output directory
  pub_output_dir = gSystem->ExpandPathName(pub_output_dir.Data()) ;
  if (gSystem->AccessPathName(pub_output_dir.Data())==kFALSE)
   { std::cout<<"Output directory is "<<pub_output_dir<<std::endl ; }
  else if (gSystem->mkdir(pub_output_dir,kTRUE)<0)
   { std::cerr<<"Failed to create "<<pub_output_dir<<std::endl ; exit(1) ; }
  else
   { std::cout<<"Creating "<<pub_output_dir<<std::endl ; }

  // open input file
  if (gSystem->CopyFile(pub_input_file.Data(),(pub_output_dir+"/"+pub_input_file).Data(),kTRUE)<0)
   { std::cerr<<"Failed to copy "<<pub_input_file<<std::endl ; exit(2) ; }
  else
   { std::cout<<"Input file is "<<pub_input_file<<std::endl ; }
  TFile * file = TFile::Open(pub_input_file) ;
  if (file!=0)
   {
    std::cout<<"Opening "<<pub_input_file<<std::endl ;
    if (file->cd(pub_input_folder)!=kTRUE)
     { std::cerr<<"Do not find "<<pub_input_folder<<std::endl ; exit(4) ; }
    else
     { std::cout<<"Input folder is "<<pub_input_folder<<std::endl ; }
   }
  else
   { std::cerr<<"Failed to open "<<pub_input_file<<std::endl ; exit(3) ; }


  // web page header
  std::ofstream web_page((pub_output_dir+"/index.html").Data()) ;
  web_page
    <<"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 3.2 Final//EN\">\n"
    <<"<html>\n"
    <<"<head>\n"
    <<"<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\" />\n"
    <<"<title>"<<pub_title<<"</title>\n"
    <<"</head>\n"
    <<"<h1><a href=\"../\">"<<pub_title<<"</a></h1>\n" ;
  web_page<<"<p>"<<pub_comment ;
  web_page
    <<" They were made using configurations "
    <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/DQMOffline/EGamma/test/ElectronAnalyzer_cfg.py\">"
    <<"DQMOffline/EGamma/test/ElectronAnalyzer_cfg.py</a> and "
    <<"<a href=\"http://cmslxr.fnal.gov/lxr/source/DQMOffline/EGamma/test/ElectronOfflineClient_cfg.py\">"
    <<"DQMOffline/EGamma/test/ElectronOfflineClient_cfg.py</a>." ;
  web_page
    <<" One can download the full <a href=\""<<pub_input_file<<"\"> histograms file</a>." ;
  web_page<<"</p>\n" ;

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
  eleStyle->SetPadTickX(1);
  eleStyle->SetPadTickY(1);
  eleStyle->SetHistLineColor(1);
  eleStyle->SetHistLineStyle(0);
  eleStyle->SetHistLineWidth(2);
  eleStyle->SetEndErrorSize(2);
  eleStyle->SetErrorX(0.);
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

  // variables for the next loops
  int cat_num ;
  TList * keys1, * keys2 ;
  TKey * key1, * key2 ;
  TObject * obj1, * obj2 ;
  TDirectory * dir ;
  TH1 * histo ;
  TString short_histo_name, anchor_name, histo_option ;
  file->cd(pub_input_folder) ;
  keys1 = gDirectory->GetListOfKeys() ;
  TIter * nextKey1, * nextKey2 ;

  // top table
  std::cout<<"Writing top table"<<std::endl ;
  web_page
    <<"<br><table border=\"1\" cellpadding=\"5\" width=\"100%\">"
    <<"<tr valign=\"top\">\n" ;
  cat_num = 0 ;
  file->cd(pub_input_folder) ;
  keys1 = gDirectory->GetListOfKeys() ;
  nextKey1 = new TIter(keys1) ;
  while (key1 = (TKey *)(*nextKey1)())
   {
    obj1 = key1->ReadObj() ;
    if (obj1->IsA()->InheritsFrom("TDirectory")==kFALSE)
     { std::cout<<"Ignoring object "<<obj1->GetName()<<std::endl ; continue ; }
    else
     { std::cout<<"Processing folder "<<obj1->GetName()<<std::endl ; }
    dir = (TDirectory *)obj1 ;
    web_page<<"<td width=\"20%\"><b>"<<dir->GetName()<<"</b><br><br>\n" ;
    keys2 = dir->GetListOfKeys() ;
    nextKey2 = new TIter(keys2) ;
    while (key2 = (TKey *)(*nextKey2)())
     {
      obj2 = key2->ReadObj() ;
      if (obj2->IsA()->InheritsFrom("TH1")==kFALSE)
       { std::cout<<"Ignoring object "<<obj2->GetName()<<std::endl ; continue ; }
      short_histo_name = obj2->GetName() ;
      //short_histo_name.Remove(0,3) ;
      anchor_name = dir->GetName() ;
      anchor_name += "_" ;
      anchor_name += short_histo_name ;
      web_page<<"<a href=\"#"<<anchor_name<<"\">"<<short_histo_name<<"</a><br>\n" ;
     }
    web_page<<"<br></td>\n" ;
    cat_num++ ;
    if ((cat_num%5)==0)
     { web_page<<"</tr>\n<tr valign=\"top\">" ; }
   }
  web_page<<"</tr></table>\n" ;

  // histograms
  std::cout<<"Plotting histograms"<<std::endl ;
  gErrorIgnoreLevel = kWarning ;
  web_page<<"<br><br><table cellpadding=\"5\"><tr valign=\"top\"><td>\n" ;
  TCanvas * canvas ;
  TString left_histo_name, histo_name, gif_name, gif_path, canvas_name ;
  cat_num = 0 ;
  file->cd(pub_input_folder) ;
  keys1 = gDirectory->GetListOfKeys() ;
  nextKey1 = new TIter(keys1) ;
  while (key1 = (TKey *)(*nextKey1)())
   {
    obj1 = key1->ReadObj() ;
    if (obj1->IsA()->InheritsFrom("TDirectory")==kFALSE)
     { continue ; }
    dir = (TDirectory *)obj1 ;
    keys2 = dir->GetListOfKeys() ;
    nextKey2 = new TIter(keys2) ;
    while (key2 = (TKey *)(*nextKey2)())
     {
      obj2 = key2->ReadObj() ;
      if (obj2->IsA()->InheritsFrom("TH1")==kFALSE)
       { std::cout<<"Ignoring object "<<obj2->GetName()<<std::endl ; continue ; }
      histo = (TH1 *)obj2 ;

      std::cout
        <<dir->GetName()<<"/"<<histo->GetName()<<";"<<key2->GetCycle()
        <<" has "<<histo->GetEntries()<<" entries"
        <<" (~"<<histo->GetEffectiveEntries()<<")"
        <<" of mean value "<<histo->GetMean()
        <<std::endl ;

      histo_name = histo->GetName() ;
      if (left_histo_name.IsNull()==kFALSE)
       {
        if (histo_name.Index(left_histo_name)==0)
         { web_page<<"</td><td>" ; }
        else
         {
          left_histo_name = histo_name ;
          web_page<<"</td></tr>\n<tr valign=\"top\"><td>" ;
         }
       }
      else
       { left_histo_name = histo_name ; }

      short_histo_name = histo_name ;
      //short_histo_name.Remove(0,3) ;
      anchor_name = dir->GetName() ;
      anchor_name += "_" ;
      anchor_name += short_histo_name ;
      gif_name = anchor_name+".gif" ;
      gif_path = pub_output_dir+"/"+gif_name ;
      canvas_name = "c_" ;
      canvas_name += anchor_name ;
      canvas = new TCanvas(canvas_name) ;
      canvas->SetFillColor(10) ;

      histo->SetLineColor(2) ;
      histo->SetMarkerColor(2) ;
      histo->SetLineWidth(3) ;

      histo_option = histo->GetOption() ;
      if ((histo_option.Contains("ELE_LOGY")==kTRUE)&&(histo->GetEntries()>0)&&(histo->GetMaximum()>0))
       { canvas->SetLogy(1) ; }

      if (histo->IsA()->InheritsFrom("TH2")==kTRUE)
       {
        gStyle->SetPalette(1) ;
        gStyle->SetOptStat(111) ;
        histo->Draw(/*"COLZ"*/) ;
       }
      else if (histo->IsA()->InheritsFrom("TProfile")==kTRUE)
       {
        gStyle->SetOptStat(111) ;
        histo->Draw(/*"E1 P"*/) ;
       }
      else
       {
        gStyle->SetOptStat(111111) ;
        histo->Draw(/*"E1 P"*/) ;
       }
      canvas->SaveAs(gif_path.Data()) ;

      web_page
        <<"<a id=\""<<anchor_name<<"\" name=\""<<anchor_name<<"\"></a>"
        <<"<a href=\""<<gif_name<<"\"><img border=\"0\" class=\"image\" width=\"500\" src=\""<<gif_name<<"\"></a><br>\n" ;
     }
   }
  web_page<<"</td></tr></table>\n" ;

  // the end
  web_page<<"\n</html>"<<std::endl ;
  web_page.close() ;
  std::cout<<"Histos written to "<<pub_output_dir<<std::endl ;
  return 0 ;
 }
