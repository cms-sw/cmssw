TObject *genDesc  =NULL;
TObject *userHisto=NULL;
TString defaultPattern("mc-tester_*.root");

// Load libraries
void loadLibs()
{
  gSystem->Load("libHist.so");
  gSystem->Load("libGpad.so");
  gSystem.Load("libHEPEvent.so");
  gSystem.Load("libHepMCEvent.so");
  gSystem.Load("libMCTester.so");
  gROOT->SetStyle("Plain");
}

void MERGE()
{
  cout<<"Usage: merge( <output> , <input directory or first file> )\n";
  cout<<"       merge( <output> , <input directory or first file> , <pattern> )\n\n";
}

void MERGE(TString outfile, TString firstfile, TString pat=defaultPattern) { merge(outfile,firstfile,pat); }
void merge(TString outfile, TString firstfile, TString pat=defaultPattern)
{
  TList   decayModes;
  bool    save=false;
  TString dir=".";
  TString first="";
  long    flags=0,id=0,size=0,modtime=0;
  if(firstfile.Contains("~"))
    {
      cout<<"ERROR: input path cannot contain '~' character."<<endl;
      return;
    }
  // Check pattern
  TRegexp pattern(pat,true);
  if(pattern.Status()!=0)
    {
      cout<<"ERROR: Wrong regular expression."<<endl;
      return;
    }
  // Load libraries
  loadLibs();
  gSystem->GetPathInfo(firstfile, &id, &size, &flags, &modtime);
  bool isDirectory = flags&2;
  if(isDirectory)
    {
      dir=firstfile;
      first="";
    }
  else
    {
      //Separate directory from filename
      int separator=firstfile.Last('/');
      if(separator!=-1) dir=firstfile(0,separator+1);
      first=firstfile(separator+1,firstfile.Length());
    }
  cout<<"Output file: "<<outfile<<endl;
  cout<<"Input  dir:  "<<dir<<endl;
  if(!isDirectory) cout<<"First  file: "<<first<<endl;
  cout<<endl;
  // Get file list and add first file if present
  TList *files = getFileList(first,dir,pattern);
  if(!isDirectory)
    {
      TFile *ff = new TFile(firstfile,"READ");
      files->AddBefore(files->First(),ff);
    }
  // Merge files
  TIter next(files);
  TFile *file;
  while(file = (TFile*)next())
    {
      if(!file->IsOpen()) continue;
      ReadFile(file,decayModes);
      cout<<"=============================="<<endl;
      save=true;
    }
  // Save output
  if(save)
    {
      cout<<"Saving..."<<endl;
      SaveOutput(outfile,decayModes);
      cout<<"Output saved to "<<outfile<<endl<<endl;
    }
  else    cout<<"Nothing to save."<<endl<<endl;
  // Closing files
  cout<<"Closing files..."<<endl;
  TIter cl(files);
  while(file = (TFile*)cl()) { file->Close(); delete file; }
  delete files;
  gSystem->Exit(0);
}

// Search for files matching the pattern and different from the first file
TList *getFileList(TString first, TString dirname, TRegexp pattern)
{
  void *dir = gSystem->OpenDirectory(dirname);
  if(!dir) return 0;
  const char *file = 0;
  TList *list  = new TList();
  while(file = gSystem->GetDirEntry(dir))
    {
      TString f(file);
      if(!f.Contains(pattern)) continue;
      if(!f.CompareTo(first)) continue;
      cout<<"Added  file: "<<f<<endl;
      list->Add(new TFile(dirname+"/"+f,"READ"));
    }
  gSystem->FreeDirectory(dir);
  cout<<endl;
  return list;
}

// Save to .root file
void SaveOutput(TString name,TList &list)
{
  TFile *out = new TFile(name,"RECREATE");
  TDirectory *pwd = gDirectory;
  for(int i=0;i<list.GetSize();i++)
    {
      TDecayMode *dm = list.At(i);
      TDirectory *subdir = out->mkdir(dm->GetName());
      subdir->cd();
      subdir->Append(dm);
      subdir->Append(dm->histograms);
      pwd->cd();
    }
  if(!genDesc) cout<<"WARNING! No Generator description in files!"<<endl;
  else out->Append(genDesc);
  if(userHisto)
    {
      
      TDecayMode *uh = (TDecayMode*) userHisto;
      cout<<"INFO: Appending user histograms"<<endl;
      TDirectoryFile *histos = out->mkdir("USER_HISTOGRAMS");
      TIter  nexthist(uh->histograms);
      TKey  *key_hist=0;
      TH1D *h;
      while(h=(TH1D*)nexthist()) histos->Append(h);
    }
  out->Write();
  out->Close();
  delete out;
}

// Read decay modes from single file
void ReadFile(TFile *f, TList &list)
{
  f->cd();
  cout<<"Reading file: "<<f->GetName()<<endl<<endl;
  TList *dir=f->GetListOfKeys();
  TIter nextdir(dir);
  TKey *key_dir=0;
  while(key_dir=(TKey*)nextdir())
    {
      if(strcmp(key_dir->GetClassName(),"TDirectoryFile")==0 ||
	 strcmp(key_dir->GetClassName(),"TDirectory")==0 )
	{
	  TDirectory *subdir=(TDirectory*)f->Get(key_dir->GetName());
	  TList *keys=subdir->GetListOfKeys();
	  TIter  nextKey(keys);
	  TKey  *key=0;
	  // User Histograms
	  if(strcmp(key_dir->GetName(),"USER_HISTOGRAMS")==0)
	    {
	      TDecayMode *dm = new TDecayMode();
	      dm->histograms = new TObjArray();
	      while(key=(TKey*)nextKey())
		{
		  int cycle=-10;
		  if(strcmp(key->GetClassName(),"TH1D")==0)
		    {
		      TH1D *h=0;
		      int cycleh=key->GetCycle();
		      if(cycleh<cycle) { cout<<"Skipping..."<<endl; continue; }
		      if(cycle<cycleh) cycle=cycleh;
		      h=(TH1D*)key->ReadObj();
		      if(!h)
			{
			  cout<<"Cannot read: "<<key->GetName()<<endl;
			  exit(-2);
			}
		      TH1D *h1=new TH1D(*h);
		      dm->histograms->Add(h1);
		    }
		}
	      if(userHisto==NULL) userHisto = (TDecayMode*) new TDecayMode(*dm);
	      else                dmMerge(userHisto,dm,keys);
	      continue;
	    }
	  TDecayMode *file_dm=0;
	  TDecayMode *dm=0;
	  // Regular decay mode
	  while(key=(TKey*)nextKey())
	    {
	      if(strcmp(key->GetClassName(),"TDecayMode")==0)
		{
		  file_dm=(TDecayMode*)key->ReadObj();
		  dm = new TDecayMode(*file_dm);
		}
	    }
	  if(!dm)
	    {
	      cout<<"ERROR: no TDecayMode object found in directory: "<<subdir->GetName()<<endl;
	      subdir->ls();
	      continue;
	    }
	  // Merge with existing decay mode or add as new one
	  TObject *existingDM=list.FindObject(dm->GetName());
	  if(existingDM)
	    {
	      int ret = dmMerge(existingDM,dm,keys);
	      if(ret==-1)
		{
		  printf("WARNING: Histograms with different ranges or number of bins. File skipped.\n");
		  return;
		}
	    }
	  else
	    {
	      dmRestoreHistograms(dm,keys);
	      list.Add(dm);
	    }
	}
      else
	{
	  //If not directory, it's generation description
	  if(genDesc || strcmp(key_dir->GetClassName(),"GenerationDescription")!=0) continue;
	  GenerationDescription *file_gd = (GenerationDescription*) key_dir->ReadObj();
	  GenerationDescription *gd = new GenerationDescription(*file_gd);
	  genDesc=gd;
	}
    }
}

// Merge same modes
int dmMerge(TObject *a1, TObject *b1,TList *keys)
{
  TDecayMode *a=(TDecayMode*)a1;
  TDecayMode *b=(TDecayMode*)b1;
  TIter  nexthist(keys);
  TKey  *key_hist=0;
  while(key_hist=(TKey*)nexthist())
    {
      int cycle=-10;
      if(strcmp(key_hist->GetClassName(),"TH1D")==0)
	{
	  TH1D  *h=0;
	  int cycleh=key_hist->GetCycle();
	  if(cycleh<cycle) { cout<<"Skipping..."<<endl; continue; }
	  if(cycle<cycleh) cycle=cycleh;
	  h=(TH1D*)key_hist->ReadObj();
	  if(!h)
	    {
	      cout<<"Cannot read: "<<key_hist->GetName()<<endl;
	      exit(-2);
	    }
	  TH1D *eh = (TH1D*) a->histograms->FindObject(h->GetName());
	  if(!eh) continue;
	  if(eh->GetNbinsX()!=h->GetNbinsX() ||
	     eh->GetXaxis()->GetXmax()!=h->GetXaxis()->GetXmax())
	    return -1;
	  eh->Add(h);
	}
    }
  a->SetNEntries(a->GetNEntries()+b->GetNEntries());
  a->SetSumw(a->GetSumw()+b->GetSumw());
  a->SetSumw2(a->GetSumw2()+b->GetSumw2());
  return 0;
}

// Reload histograms
int dmRestoreHistograms(TObject *tdm,TList *keys)
{
  TDecayMode *dm = (TDecayMode*)tdm;
  dm->histograms = new TObjArray();
  TIter  nexthist(keys);
  TKey  *key_hist=0;
  char name[129];
  while(key_hist=(TKey*)nexthist())
    {
      int cycle=-10;
      if(strcmp(key_hist->GetClassName(),"TH1D")==0)
	{
	  TH1D  *h=0;
	  int cycleh=key_hist->GetCycle();
	  if(cycleh<cycle) { cout<<"Skipping..."<<endl; continue; }
	  if(cycle<cycleh) cycle=cycleh;
	  h=(TH1D*)key_hist->ReadObj();
	  if(!h)
	    {
	      cout<<"Cannot read: "<<key_hist->GetName()<<endl;
	      exit(-2);
	    }
	  TH1D *h1=new TH1D(*h);
	  dm->histograms->Add(h1);
	}
    }
  return 0;
}
