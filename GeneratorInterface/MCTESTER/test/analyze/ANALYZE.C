TObjArray results1;
TObjArray results2;

TList tmp_results1;
TList tmp_results2;

TObjArray user_histos1;
TObjArray user_histos2;

TObjArray gen1_only;
TObjArray gen2_only;

GenerationDescription *gen1=0;
GenerationDescription *gen2=0;


// this is for debuging purposes: if weights weren't stored correctly,
// we may suppress usage of weights and "force" usage of "num of entries"
// instead (it works by resetting weight sums to num of entries

bool SuppressUsageOfWeights=false; //default
//bool SuppressUsageOfWeights=true;

void ANALYZE() //default
{
  ANALYZE(".","./prod1/mc-tester.root","./prod2/mc-tester.root");
}

void ANALYZE(TString work_dir,TString path1, TString path2 )
{

  TH1::AddDirectory(kFALSE); 
    
  Setup::stage=0;
    

  TFile *f1=new TFile(path1,"READ");


  TFile *f2=new TFile(path2,"READ");


  // Now combine data from GenerationDescription objects to SETUP,
  // and finaly source user-modifications from SETUP.C

  if (!gInterpreter->ExecuteMacro(work_dir+"/SETUP.C")){  //probably no SETUP.C in working dir
    cout << "Using default SETUP.C" << endl;
    gInterpreter->ExecuteMacro("SETUP.C"); //use SETUP.C in this dir.
  }

  if (Setup::user_analysis==0) {
    printf("\n\n\n*************************************************\n");
    printf("ERROR!\n Histogram Analysis function is not set! \n ");
    printf("Please use SETUP.C file\n");
    printf("*************************************************\n\n\n");
    exit(-10);
  }

  ReadResults(f1, tmp_results1,user_histos1,1);
  ReadResults(f2, tmp_results2,user_histos2,2);

  if (gen1) {
    if (strlen(gen1->gen_desc_1)
	&& strcmp(Setup::gen1_desc_1,"Description of generator (1) not specified.")==0)
      Setup::gen1_desc_1=gen1->gen_desc_1;
    if (strlen(gen1->gen_desc_2)
	&& strcmp(Setup::gen1_desc_2,"Please change this text using SETUP.C file!")==0)
      Setup::gen1_desc_2=gen1->gen_desc_2;
    if (strlen(gen1->gen_desc_3) && strcmp(Setup::gen1_desc_3,"")==0)
      Setup::gen1_desc_3=gen1->gen_desc_3;

    if (strlen(gen1->gen_path)  ) Setup::gen1_path  =gen1->gen_path;
    Setup::decay_particle = gen1->decay_particle;
  }

  if (gen2) {
    if (strlen(gen2->gen_desc_1)
	&& strcmp(Setup::gen2_desc_1,"Description of generator (2) not specified..")==0)
      Setup::gen2_desc_1=gen2->gen_desc_1;
    if (strlen(gen2->gen_desc_2)
	&& strcmp(Setup::gen2_desc_2,"Please change this text using SETUP.C file!")==0)
      Setup::gen2_desc_2=gen2->gen_desc_2;
    if (strlen(gen2->gen_desc_3) && strcmp(Setup::gen2_desc_3,"")==0)
      Setup::gen2_desc_3=gen2->gen_desc_3;

    if (strlen(gen2->gen_path)  ) Setup::gen2_path  =gen2->gen_path;
  }

  // now reassign channels from Gen2 to match the precedence
  // of ones from Gen1...

  TIter next(&tmp_results1);
  TDecayMode *m=0;
  while (m=(TDecayMode*)next()) {
    
    TDecayMode *m2 = (TDecayMode*) tmp_results2.FindObject(m->GetName());
    if (m2) {
      results1.Add(m);
      results2.Add(m2);
          
    } else {
      gen1_only.Add(m);

    }
    
  }


  TIter next2(&tmp_results2);
  TDecayMode *m=0;
  while (m=(TDecayMode*)next2()) {
    
    TDecayMode *m2 = (TDecayMode*) tmp_results1.FindObject(m->GetName());
    if (!m2) {
      gen2_only.Add(m);
    } else {
    }
    
  }       


  TFile *fResults=new TFile(work_dir+"/mc-results.root","RECREATE");

  printf("processing...\n");
  ProcessModes(results1,results2,fResults);
  //If there are used defined histograms then process them
  if(user_histos1.GetEntries()>0||user_histos2.GetEntries()>0)    
    ProcessUserHistos(user_histos1,user_histos2,fResults);
  printf("generating LaTeX output...\n");
  LaTeXOut(work_dir);
  printf("writing...\n");
  fResults->Write();


}

void ReadResults(TFile *f, TList &a,TObjArray &user_histos,int generator)
{

  f->cd();
  printf("Opening file : %s\n\n", f->GetName());

  TList *directories=f->GetListOfKeys();
  directories->Print();
  TIter nextdir(directories);
  TKey *key_dir=0;
  // loop over directories
  while (key_dir=(TKey*) nextdir() ) {
    if (strcmp(key_dir->GetName(),"USER_HISTOGRAMS")==0) {
      // we need to read user histograms...
      TDirectory *subdir=(TDirectory*) f->Get(key_dir->GetName());
      TList   *objects=subdir->GetListOfKeys();

      // now we want to find TDecayMode object...
      TIter    nextobj(objects);
      TKey    *key_obj=0;
      TDecayMode *dm=0;
      int cycle=-1;
      // within subdirectory, loop over all objects:
      while (key_obj=(TKey*)nextobj()) {
	if (strcmp(key_obj->GetClassName(),"TH1D")==0 ) {
	  TH1D *h=(TH1D*) key_obj->ReadObj();
	  cycle=key_obj->GetCycle();
	  user_histos.AddLast(h);
	}
      }

    } else if (strcmp(key_dir->GetClassName(),"TDirectoryFile")==0 ||
	       strcmp(key_dir->GetClassName(),"TDirectory")==0 ) { //Added by ND
          
      TDirectory *subdir=(TDirectory*) f->Get(key_dir->GetName());
          
      cout<<"We are in "<<subdir->GetName()<<"\n";

      TList   *objects=subdir->GetListOfKeys();

      // now we want to find TDecayMode object...
      TIter    nextobj(objects);
      TKey    *key_obj=0;
      TDecayMode *dm=0;
      int cycle=-1;


      // within subdirectory, loop over all objects:
      while (key_obj=(TKey*)nextobj()) {
	//key_obj->ls();
	/*if (strcmp(key_obj->GetClassName(),"TDecayMode")==0 &&
	  key_obj->GetCycle() > cycle ) {*/
	if (strcmp(key_obj->GetClassName(),"TDecayMode")==0) {
	  dm=(TDecayMode*) key_obj->ReadObj();
	  cycle=key_obj->GetCycle();
	  printf("DM: %s nentries %i\n",dm->GetName(),dm->GetNEntries());

	}
      }

      if (!dm) {
	printf("ERROR: no TDecayMode object found in directory %s\n",subdir->GetName());
	subdir->ls();
	exit(-1);    
      }
          
      printf("Read decay mode: %s, nentries %i/%i\n",dm->GetName(),dm->GetNEntries(),dm->NFills);

      //now insert to the list in such a way, that list is sorted.
      TDecayMode *ml1=0;
      TDecayMode *ml2=0;
      TIter next(&a);
      while (ml1=(TDecayMode*)next()) {
    
	//int nent1=ml1->GetNEntries();
	//int nent=dm->GetNEntries();
	double nent1=ml1->GetSumw();
	double nent=dm->GetSumw();
	// we look for the last one that is bigger than us...
	if (nent1>nent) ml2=ml1;
	else
	  break;
      }
      a.AddAfter(ml2,dm);
          
      //a.Add(dm);

      dm->histograms->Clear();
          
      // now loop once again to restore all histograms.
      // this is more tricky, as there may be more than one
      // cycle for each histogram. We take ones with highest
      // cycle numbers.
          
      TIter  nexthist(objects);
      TKey  *key_hist=0;
      char name[129];
      while (key_hist=(TKey*)nexthist()) {
	if (strcmp(key_hist->GetClassName(),"TH1D")==0) {
	      
	  //printf(" FOUND KEY: %s CYCLE %i\n",key_hist->GetName(),key_hist->GetCycle());
	  TH1D  *h=0;

	  int cycleh=key_hist->GetCycle();
	  if (cycleh<cycle) {
	    //printf("skipping hist %s with cycle %i\n",key_hist->GetName(),cycleh);
	    continue;
	  }
	      
	  if (cycle<cycleh) cycle=cycleh;
	      
	  // firstly: check if we already have a histogram
	  // with this name on the list...

	  sprintf (name,"%s_GEN%1i",key_hist->GetName(),generator);

	  TObject *hist1=dm->histograms->FindObject(name);
	      
	  if (!hist1) {
	    h=(TH1D*)key_hist->ReadObj();
	    if (!h) {
	      printf("cannot read object %s\n!",key_hist->GetName());
	      exit(-2);
	    }
	    // we make a copy of the histogram, because the one we
	    // have now will disappear when we close the file!!!
	    TH1D *h1=new TH1D(*h);
	    sprintf (name,"%s_GEN%1i",h->GetName(),generator);
	    h1->SetName(name);
	    sprintf (name,"%s from Generator %1i",h->GetTitle(),generator);
	    h1->SetTitle(name);
	    dm->histograms->Add(h1);
	    
	    //NEW! Rebin here!
	    if (Setup::rebin_factor>1) h1->Rebin(Setup::rebin_factor);
	    
	    // Ah! And we normalize them as well...
	    //Normalization moved to ProcessModes();
	    //if (h1->Integral() > 0.0)
	    //    h1->Scale(1.0/h1->Integral());
	    
	  } else {
	    //printf("HIST already on list: %i :",hist1);
	    //hist1->ls();
	  }
	      
	}    
      } // end of histograms reading...

    } else {
      
      if (generator==1)
	gen1=(GenerationDescription*)key_dir->ReadObj();
      else
	gen2=(GenerationDescription*)key_dir->ReadObj();
    }
  }; // end of loop over directories
  


  printf("==============================\n");

}



void ProcessModes(TObjArray &modes1, TObjArray &modes2 , TFile *outfile)
{
  outfile->cd();

  TIter next1(&modes1);
  TIter next2(&modes2);

  TDecayMode *m1=0;
  TDecayMode *m2=0;

  while ( (m1=(TDecayMode*)next1()) && (m2=(TDecayMode*)next2()) ) {

    printf("   processing mode: %s\n",m1->GetName());
    
    if (   strcmp( m1->GetName(), m2->GetName() ) != 0) {
      printf ("INCONSISTENCY IN DATA FILES!!!\n");
      printf (" tried to compare %s with %s\n",m1->GetTitle(), m2->GetTitle());
      exit(-10);
    }
    
    outfile->cd();
    TDirectory *subdir=outfile->mkdir(m1->GetName(),m1->GetLaTeXName());

    TDirectory *histdir=subdir->mkdir("histograms");
    
    TIter nexthist1(m1->histograms);
    TIter nexthist2(m2->histograms);

    TH1D *h1=0;
    TH1D *h2=0;

    double max_fit_param=0.0;
    while ( (h1=(TH1D*)nexthist1())  && (h2=(TH1D*)nexthist2()) ) {

      if(h1->GetNbinsX()!=h2->GetNbinsX())
	{
	  printf("Histogram Division Error!\n");
	  printf("Cannot divide histograms with different number of bins!\n");
	  exit(-1);
	}
      char name[128];
      char title[128];
      char *hname=h1->GetName();
      char *htitle=h1->GetTitle();
      sprintf(name,"%s",hname);
      sprintf(title,"Comparison of %s",htitle);

      
      int cutlen=strlen(name);
      cutlen-=5;
      name[cutlen]=0;
          
      cutlen=strlen(title);
      cutlen-=17;
      title[cutlen]=0;
          
      strcat(title," in channel ");
      strcat(title,m1->GetName());
          
      TDecayResult *dr=new TDecayResult();
      dr->SetName(name+1); // without leading "h"
      dr->SetTitle( title);


      strcat(name,"_DIFF");

      histdir->cd();

      //    double max1=h1->GetMaximum();
      //    double max2=h2->GetMaximum();

      //    if (max1 > 0) h1->Scale(1.0/max1);
      //    if (max2>0) h2->Scale(1.0/max2);
          

          
      dr->h1=new TH1D(*h1);
      dr->h2=new TH1D(*h2);
          
          

      TH1D *hdiff=new TH1D(*h1);
      hdiff->SetName(name);
      hdiff->SetTitle(title);


      hdiff->SetMinimum(-1);

      double int1=h1->Integral();
      double int2=h2->Integral();
      if (int1<=0) int1=1.0;
      if (int2<=0) int2=1.0;
      hdiff->Divide(h1,h2,1.0/int1,1.0/int2);

      if (hdiff->GetMinimum() < 0) {
	// this means: out of range division!

	hdiff->SetMinimum(0);
	hdiff->SetMaximum(3);

      } else {
	if (hdiff->GetMaximum() > 3.0) hdiff->SetMaximum(3.0);
      }
          
      //    for (int ii=0;ii<hdiff->GetNbinsX();ii++)
      //hdiff->SetBinError(i,0.0);
          
      dr->hdiff=hdiff;

      //    h1->Sumw2();
      //    h2->Sumw2();
          

      double (*HistogramAnalysis)(TH1D*,TH1D*) = Setup::user_analysis;

      dr->fit_parameter= HistogramAnalysis(h1,h2);

      if (dr->fit_parameter > max_fit_param)
	max_fit_param=dr->fit_parameter;

      subdir->cd();
      subdir->Append(dr);

    }
    
    m1->fit_parameter=max_fit_param;
    m2->fit_parameter=max_fit_param;
    

  }

}


void ProcessUserHistos(TObjArray &histos1, TObjArray &histos2 , TFile *outfile)
{
  outfile->cd();
  TDirectory *subdir=outfile->mkdir("USER_HISTOGRAMS","USER_HISTOGRAMS");
  TDirectory *histdir=subdir->mkdir("histograms");
    
  for (int i=0; i<histos1.GetEntries();i++) {
    TH1D *h1=histos1[i];
    TH1D *h2=histos2.FindObject(h1->GetName());
    
    if (!h2) {
      printf("histo2 not found...\n");
      break;
    } 
    
    char name[256];
    char title[256];
    sprintf(name,h1->GetName());
    sprintf(title,h1->GetTitle());
    
    TDecayResult *dr=new TDecayResult();
    dr->SetName(name+1); // without leading "h"
    dr->SetTitle( title);


    strcat(name,"_DIFF");

    histdir->cd();
    //h1->Rebin(20);
    //h2->Rebin(20);
        
    dr->h1=new TH1D(*h1);
    dr->h2=new TH1D(*h2);

    TH1D *hdiff=new TH1D(*h1);
    hdiff->SetName(name);
    hdiff->SetTitle(title);
    hdiff->SetMinimum(-1);

    double int1=h1->Integral();
    double int2=h2->Integral();
    if (int1<=0) int1=1.0;
    if (int2<=0) int2=1.0;
    hdiff->Divide(h1,h2,1.0/int1,1.0/int2);

    if (hdiff->GetMinimum() < 0) {
      // this means: out of range division!

      hdiff->SetMinimum(0);
      hdiff->SetMaximum(3);

    } else {
      if (hdiff->GetMaximum() > 3.0) hdiff->SetMaximum(3.0);
    }
    
    dr->hdiff=hdiff;
    
    histdir->Append(dr->h1);
    histdir->Append(dr->h2);
    histdir->Append(dr->hdiff);
    
    subdir->cd();
    subdir->Append(dr);
    
  }

}


void LaTeXOut(TString work_dir)
{

  long int total1=0;
  long int total2=0;
  double totalw1=0;
  double totalw2=0;


  TIter next1(&tmp_results1);
  TIter next2(&tmp_results2);

  TDecayMode *m1=0;
  TDecayMode *m2=0;

  while ( (m1=(TDecayMode*)next1()) ) {

    total1+=m1->GetNEntries();


    if (SuppressUsageOfWeights) {
      //!!! DEBUG MODE WHERE WEIGHTS NOT STORED PROPERLY !!!
      printf("M1: Forcing totalw1 from NEntries: %li\n",m1->GetNEntries());
      totalw1+=m1->GetNEntries();
    } else {
    
      totalw1+=m1->GetSumw();
          
    }
  }

  while ( (m2=(TDecayMode*)next2()) ) {
    total2+=m2->GetNEntries();

    if (SuppressUsageOfWeights) {
      //!!! DEBUG MODE WHERE WEIGHTS NOT STORED PROPERLY !!!
      printf("M2: Forcing totalw1 from NEntries: %li\n",m2->GetNEntries());
      totalw2+=m2->GetNEntries();
    } else {
    
      totalw2+=m2->GetSumw();
          
    }
  }

  printf("total=%f total2=%f totalw1=%f totalw2=%f\n",total1,total2,totalw1,totalw2);

  FILE *fRES= fopen((const char *)(work_dir+"/mc-results.tex"),"w");

  fprintf(fRES,"%%Generated by MC-TESTER:ANALYZE.C.\n");
  fprintf(fRES,"%% to be included in tester.tex\n\n");

  fprintf(fRES,"\n");
  fprintf(fRES,"\\author{Nadia Davidson  \\and Piotr Golonka \\and Tomasz Pierzchala  \\and Tomasz Przedzinski \\and Zbigniew Was \\and } \n");
  fprintf(fRES,"\\title{MC-TESTER v1.24.4 - results for decays of particle $%s$ (PDG code %i).}\n",
	  HEPParticle::GetLaTeXName(Setup::decay_particle), Setup::decay_particle);
  fprintf(fRES,"\\maketitle\n");
  fprintf(fRES,"\n");

  

  fprintf(fRES,"\\section*{Results from \\textcolor{red}{generator 1}.}\n");
  fprintf(fRES,"\\vspace{0.3cm} \n");
  fprintf(fRES,"{\\centering \\begin{tabular}{|c|} \n");
  fprintf(fRES,"\\hline \n");
  fprintf(fRES," \\\\ \n");
  fprintf(fRES,"%s \\\\ \n", Setup::gen1_desc_1);
  fprintf(fRES,"%s \\\\ \n", Setup::gen1_desc_2);
  fprintf(fRES,"%s \\\\ \n", Setup::gen1_desc_3);
  fprintf(fRES," \\\\ \n");
  fprintf(fRES,"\\hline \n");
  fprintf(fRES,"\\end{tabular}\\par} \n");
  fprintf(fRES,"\\vspace{0.3cm} \n");

  fprintf(fRES,"\\begin{itemize} \n");
  
  char ver1[4096];
  TString resv=Setup::ResolvePath(Setup::gen1_path,ver1);
  if (resv) fprintf(fRES,"\\item{From directory:} \\\\ {\\tt %s}\n",(const char *)resv.ReplaceAll("_","\\_"));
  strcat(ver1,"/version");
  FILE *fVer1=fopen(ver1,"r");
  if (fVer1) {
    fgets(ver1,4095,fVer1);
    if (ver1[strlen(ver1)-1]=='\n' ) ver1[strlen(ver1)-1]=0;
    fprintf(fRES,"\\item{Code version (from {\\tt version} file):} %s\n",ver1);
    fclose(fVer1);
  }

  fprintf(fRES,"\\item Total number of analyzed decays: %i\n",total1);
  
  int only_1=gen1_only.GetEntriesFast();
  if(only_1>0)
    fprintf(fRES,"\\item Number of decay channels found: %i + %i\n",results1.GetEntriesFast(), only_1);
  else
    fprintf(fRES,"\\item Number of decay channels found: %i\n",results1.GetEntriesFast());

  fprintf(fRES,"\\end{itemize} \n");
  



  fprintf(fRES,"\\section*{Results from \\textcolor{green}{generator 2}.}\n");

  fprintf(fRES,"\\vspace{0.3cm} \n");
  fprintf(fRES,"{\\centering \\begin{tabular}{|c|} \n");
  fprintf(fRES,"\\hline \n");
  fprintf(fRES," \\\\ \n");
  fprintf(fRES,"%s \\\\ \n", Setup::gen2_desc_1);
  fprintf(fRES,"%s \\\\ \n", Setup::gen2_desc_2);
  fprintf(fRES,"%s \\\\ \n", Setup::gen2_desc_3);
  fprintf(fRES," \\\\ \n");
  fprintf(fRES,"\\hline \n");
  fprintf(fRES,"\\end{tabular}\\par} \n");
  fprintf(fRES,"\\vspace{0.3cm} \n");

  fprintf(fRES,"\\begin{itemize} \n");


  char ver2[4096];
  TString resv=Setup::ResolvePath(Setup::gen2_path,ver2);   //fix for dir with "_" character
  if (resv) fprintf(fRES,"\\item{From directory:} \\\\ {\\tt %s}\n",(const char *)resv.ReplaceAll("_","\\_"));
  strcat(ver2,"/version");
  FILE *fVer2=fopen(ver2,"r");
  if (fVer2) {
    fgets(ver2,4095,fVer2);
    if (ver2[strlen(ver2)-1]=='\n' ) ver2[strlen(ver2)-1]=0;

    fprintf(fRES,"\\item{Code version (from{ \\tt version} file):} %s\n",ver2);
    fclose(fVer2);
  }


  fprintf(fRES,"\\item Total number of analyzed decays: %i\n",total2);

  int only_2=gen2_only.GetEntriesFast();
  if(only_2>0)
    fprintf(fRES,"\\item Number of decay channels found: %i + %i\n",results2.GetEntriesFast(), only_2);
  else
    fprintf(fRES,"\\item Number of decay channels found: %i\n",results2.GetEntriesFast());
  fprintf(fRES,"\\end{itemize} \n");






  // Now we check if histograms from both channels are compatible, and print warning if the aren't:

  int max_decay_multiplicity=20;
  int OK=1;

  const char *warning_header="\n\\section*{WARNINGS (not all entries are used):} \n{\\centering \\begin{longtable}{|c|c|c|c|c|} \n\\hline \nWhat & n-body decay & n-body histogram & GEN1 & GEN2 \\\\\n\\hline \n";
  for (int i=1; i<max_decay_multiplicity;i++) {
    for (int j=1; j<=i;j++) {
      
      if (gen1->nbins[i][j] != gen2->nbins[i][j]) {

	if (OK) { fprintf(fRES,warning_header); OK=0; }
	
	fprintf(fRES,"\\hline Number of  bins & %i & %i &%li & %li \\\\ \n",
		i,j,gen1->nbins[i][j],gen2->nbins[i][j]);
      }
          
      if (gen1->bin_min[i][j] != gen2->bin_min[i][j]) {

	if (OK) { fprintf(fRES,warning_header); OK=0; }

	fprintf(fRES,"\\hline Minimal bin & %i & %i & %f & %f \\\\ \n",
		i,j,gen1->bin_min[i][j],gen2->bin_min[i][j]);
      }
          
      if (gen1->bin_max[i][j] != gen2->bin_max[i][j]) {

	if (OK) { fprintf(fRES,warning_header); OK=0; }

	fprintf(fRES,"\\hline Maximal bin & %i & %i &%f & %f \\\\ \n",
		i,j,gen1->bin_max[i][j],gen2->bin_max[i][j]);
      }
          
    }
  }

  if (!OK) {
    fprintf(fRES,"\\hline\n");
    fprintf(fRES,"\\end{longtable}\\par} \n");
    fprintf(fRES,"\\vspace{0.3cm} \n");
  }


  // Print a table with decay modes
  // and collect statistics for similarity coefficients T1 and T2
  double T1=0.0;
  double T2=0.0;
  double BR_kappa=3;

  fprintf(fRES,"\n\n\\newpage\n\n");

  fprintf(fRES,"\\section*{Found decay modes:}\n");
  fprintf(fRES,"\n");

  fprintf(fRES,"\\vspace{0.3cm} \n");
  fprintf(fRES,"{\\centering \\begin{longtable}{|c|c|c|c|} \n");


  fprintf(fRES,"\\hline \n");
  fprintf(fRES,"Decay channel &\\multicolumn{2}{|c|}{ Branching Ratio $\\pm$ Rough Errors} & Max. shape\\\\ \n");
  fprintf(fRES,"      & \\textcolor{red}{Generator \\#1} & \\textcolor{green}{Generator \\#2} & dif. param.\\\\ \n");
  fprintf(fRES,"\\hline \n");




  TIter next1(&results1);
  TIter next2(&results2);
  TString mode_label;

  while ( (m1=(TDecayMode*)next1()) && (m2=(TDecayMode*)next2()) ) {

    double error1=0.0;
    double error2=0.0;

    if (m1->GetNEntries()>0)
      error1=100.0*(m1->GetNEntries())/total1 / sqrt(m1->GetNEntries());

    if (m2->GetNEntries()>0)
      error2=fabs(100.0*(m2->GetNEntries())/total2 / sqrt(m2->GetNEntries()));

    double m1sumw=m1->GetSumw();
    double m2sumw=m2->GetSumw();

    if (SuppressUsageOfWeights) {
      //!!! DEBUG MODE when weight are not set correct !!!//
      m1sumw=m1->GetNEntries();
      m2sumw=m2->GetNEntries();
    }
    mode_label = m1->GetName();
    fprintf(fRES,"\\hline \n \\hyperref[%s]{$%s$} &",(const char *)mode_label.ReplaceAll("~","tilda") ,m1->GetLaTeXName());
    fprintf(fRES," \\textcolor{red}{%7.4f $\\pm$ %7.4f\\%%} &\\textcolor{green}{  %7.4f $\\pm$ %7.4f\\%%} & %6.5f \\\\ \n",
	    //100.0*m1->GetNEntries()/total1, error1,
	    //100.0*m2->GetNEntries()/total2, error2,
	    100.0*m1sumw/totalw1, error1,
	    100.0*m2sumw/totalw2, error2,
	    m1->fit_parameter ); 
    fprintf(fRES,"\\hline \n");

    
    double sigma1=100.0/m1->GetNEntries();
    double sigma2=100.0/m2->GetNEntries();
    double BR_sigma =  sqrt( (sigma1*sigma1 + sigma2*sigma2) );
    double BR_diff=fabs(100.0*m1->GetNEntries()/total1 - 100.0*m2->GetNEntries()/total2) ;// - BR_kappa * BR_sigma;
    double BR_avg=(100*m1->GetNEntries()/total1 + 100.0*m2->GetNEntries()/total2)/2.0;
    double BR_diff_corrected=0;
    if (BR_diff > BR_kappa*BR_sigma) BR_diff_corrected=BR_diff-BR_kappa*BR_sigma;
    
    T1+=BR_diff_corrected;
    T2+=(m1->fit_parameter*BR_avg);
    printf("n1=%f n2=%f sigma1=%f sigma2=%f Br_sigma=%f, BR_diff=%f T1+=%f T2+=%f\n",100.0*m1->GetNEntries()/total1,
	   100.0*m2->GetNEntries()/total2,
	   sigma1,sigma2,
	   BR_sigma, BR_diff, BR_diff_corrected,
	   (m1->fit_parameter*BR_avg));

  }

  if (only_1 > 0) {
    
    fprintf(fRES,"\\hline \\multicolumn{4}{|c|}{ Channels From First Generator Only:} \\\\ \n");


    TIter next1(&gen1_only);    
    while ( (m1=(TDecayMode*)next1()) ) {
    
      double error1=0.0;
      double m1sumw=m1->GetSumw();

      if (SuppressUsageOfWeights) {
	//!!! DEBUG MODE when weight are not set correct !!!//
	m1sumw=m1->GetNEntries();
      }

      if (m1->GetNEntries()>0)
	error1=100.0*(m1->GetNEntries()) /total1 / sqrt(m1->GetNEntries());

      fprintf(fRES,"\\hline \n $%s$ &",m1->GetLaTeXName());
      fprintf(fRES," \\textcolor{red}{%7.4f  $\\pm$  %7.4f \\%% }&  - & - \\\\ \n",
	      100.0*m1sumw/totalw1,error1);

      double sigma1=100.0/m1->GetNEntries();
      double BR_sigma =  sigma1;
      double BR_diff=fabs(100.0*m1->GetNEntries()/total1) ;// - BR_kappa * BR_sigma;

      double BR_diff_corrected=0;
      if (BR_diff > BR_kappa*BR_sigma) BR_diff_corrected=BR_diff-BR_kappa*BR_sigma;
    
      T1+=BR_diff_corrected;
      printf("n1=%f sigma1=%f Br_sigma=%f, BR_diff=%f T1+=%f\n",100.0*m1->GetNEntries()/total1,
	     sigma1,
	     BR_sigma, BR_diff, BR_diff_corrected);

    }


    fprintf(fRES,"\\hline \n");
  }
    
  if (only_2 > 0) {
    
    
    fprintf(fRES,"\\hline \\multicolumn{4}{|c|}{ Channels From Second Generator Only:} \\\\ \n");
             

    TIter next2(&gen2_only);    
    while ( (m2=(TDecayMode*)next2()) ) {


      double error2=0.0;
      double m2sumw=m2->GetSumw();

      if (SuppressUsageOfWeights) {
	//!!! DEBUG MODE when weight are not set correct !!!//
	m2sumw=m2->GetNEntries();
      }

      if (m2->GetNEntries()>0)
	error2=100.0*(m2->GetNEntries())/total2 / sqrt(m2->GetNEntries());
    
      mode_label = m2->GetName();
      fprintf(fRES,"\\hline \n $%s$ &",m2->GetLaTeXName());
      fprintf(fRES," -   & \\ \\textcolor{green}{%7.4f  $\\pm$ %6.5f \\%%} & - \\\\ \n",
	      100.0*m2sumw/totalw2,error2 );

      double sigma2=100.0/m2->GetNEntries();
      double BR_sigma =  sigma2;
      double BR_diff=fabs(100.0*m2->GetNEntries()/total2) ;// - BR_kappa * BR_sigma;

      double BR_diff_corrected=0;
      if (BR_diff > BR_kappa*BR_sigma) BR_diff_corrected=BR_diff-BR_kappa*BR_sigma;
    
      T1+=BR_diff_corrected;
      printf("n2=%f sigma2=%f Br_sigma=%f, BR_diff=%f T1+=%f\n",100.0*m2->GetNEntries()/total2,
	     sigma2,
	     BR_sigma, BR_diff, BR_diff_corrected);
    }

    fprintf(fRES,"\\hline \n");
    
  }
    
    
  fprintf(fRES,"\\end{longtable}\\par} \n");
  fprintf(fRES,"\\vspace{0.3cm} \n");

  printf("T1  == %f\n",T1);
  printf("T2  == %f\n",T2);

  fprintf(fRES,"Similarity coefficients: T1=%f \\%%, T2=%f \\%% \n",T1, T2);
  if(results1.GetEntriesFast()!=0) fprintf(fRES,"\n\\newpage\n\\tableofcontents{}\n");
  fclose(fRES);

  // print it also to the file ...
  FILE *fOUT=fopen((const char *)(work_dir+"/MC-TESTER.DAT"),"a");
  fprintf(fOUT,"%s, %s, %f, %f\n",ver1,ver2,T1, T2);
  fclose(fOUT);

}

//  LocalWords:  GetSumw

