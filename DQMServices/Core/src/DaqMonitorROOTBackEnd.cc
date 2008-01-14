/** \file
 *  $Date: 2008/01/11 15:47:48 $
 *  $Revision: 1.1 $
 *  $Author: lat $
 */
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Core/interface/Tokenizer.h"

#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/Core/interface/MonitorElementRootT.h"
#include "DQMServices/Core/interface/DQMTagHelper.h"

#include "DQMServices/Core/interface/QCriterionRoot.h"

#include "FWCore/ParameterSet/interface/types.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"

#include <TROOT.h>
#include <TFile.h>
#include <TKey.h>

using namespace dqm::me_util;
using namespace dqm::qtests;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector; using std::set;

DaqMonitorROOTBackEnd * DaqMonitorROOTBackEnd::theinstance = 0;

DaqMonitorBEInterface * DaqMonitorROOTBackEnd::instance()
{
  if(theinstance==0)
    {
      try
	{
	  edm::ParameterSet pset;
	  theinstance = new DaqMonitorROOTBackEnd(pset);
	}
      catch(cms::Exception e)
	{
	  cout << e.what() << endl;
	  exit (-1);
	}
      
    }
  return (DaqMonitorBEInterface *) theinstance;
}

DaqMonitorROOTBackEnd::DaqMonitorROOTBackEnd(edm::ParameterSet const&pset) : 
  DaqMonitorBEInterface(pset)
{

  // set steerable parameters
  DQM_VERBOSE = pset.getUntrackedParameter<int>("verbose",1);
  cout << " DaqMonitorROOTBackEnd: verbose parameter set to " << 
            DQM_VERBOSE << endl;
  
  string subsystemname = 
                pset.getUntrackedParameter<string>("subSystemName","");
 
  string referencefilename = 
                pset.getUntrackedParameter<string>("referenceFileName","");
  cout << " DaqMonitorROOTBackEnd: reference file name set to " <<
            referencefilename << endl;		

  // 	    
  first_time_onRoot = true;
  overwriteFromFile = false;
  readOnlyDirectory = "";

  // initialize fCurrentFolder to "root directory": 
  TFolder * root_temp = gROOT->GetRootFolder();
  Own.top = new MonitorElementRootFolder(root_temp, root_temp->GetName());
  Own.top->pathname_ = ROOT_PATHNAME;
  setCurrentFolder(ROOT_PATHNAME);

  if ( referencefilename != "" ) this->readReferenceME(referencefilename);
  // fixme: check that file has root extension
  // or rather check that file has same name as subsystem
  // maybe the subsystem name could be centralized here 

  c1 = 0;
  last_objname = last_pathname = "";
  theinstance = this;
  tagHelper = new DQMTagHelper(this);

  availableAlgorithms.insert(Comp2RefChi2ROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefKolmogorovROOT::getAlgoName());
  availableAlgorithms.insert(ContentsXRangeROOT::getAlgoName());
  availableAlgorithms.insert(ContentsYRangeROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefEqualStringROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefEqualIntROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefEqualFloatROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefEqualH1ROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefEqualH2ROOT::getAlgoName());
  availableAlgorithms.insert(Comp2RefEqualH3ROOT::getAlgoName());
  availableAlgorithms.insert(MeanWithinExpectedROOT::getAlgoName());
  availableAlgorithms.insert(DeadChannelROOT::getAlgoName());
  availableAlgorithms.insert(NoisyChannelROOT::getAlgoName());
  availableAlgorithms.insert(MostProbableLandauROOT::getAlgoName());

  availableAlgorithms.insert(ContentsTH2FWithinRangeROOT::getAlgoName());
  availableAlgorithms.insert(ContentsProfWithinRangeROOT::getAlgoName());
  availableAlgorithms.insert(ContentsProf2DWithinRangeROOT::getAlgoName());

  tagHelper = new DQMTagHelper(this);

}

// true if pathname exists
bool DaqMonitorROOTBackEnd::pathExists(string inpath, const rootDir & Dir) const
{
  if(isTopFolder(inpath)) return true;
  return(getRootFolder(Dir)->pathExists(inpath));
}

// set the last directory in fullpath as the current directory (create if needed);
// to be invoked by user to specify directories for monitoring objects b4 booking;
// commands book1D (etc) and removeElement(name) imply elements in this directory!
void DaqMonitorROOTBackEnd::setCurrentFolder(string fullpath)
{
  fCurrentFolder = DaqMonitorBEInterface::makeDirectory(fullpath);
}

DaqMonitorROOTBackEnd::~DaqMonitorROOTBackEnd()
{
  // turn messages off
  setVerbose(0);

  // first, remove subscribers
  for(sdir_it Dir = Subscribers.begin(); Dir != Subscribers.end(); ++Dir)
    // remove subscriber directory
    rmdir(ROOT_PATHNAME, Dir->second);

  // then empty map
  Subscribers.clear();
    
  // then, remove tags
  for(tdir_it tag = Tags.begin(); tag != Tags.end(); ++tag)
    // remove tag directory
    rmdir(ROOT_PATHNAME, tag->second);

  // then empty map
  Tags.clear();

  // now remove everything else
  Own.top->setVerbose(0);
  Own.top->clear(); Own.top = 0;
  // does this clear everything under "Own.top"? Need to check for memory leaks...
  if(c1)
    {
      delete c1; c1 = 0;
    }

  theinstance = 0;
  delete tagHelper;
}

// clone/copy received TH1F object from <source> to ME in <dir>
MonitorElement * 
DaqMonitorROOTBackEnd::book1D(std::string name, TH1F* source,
                              MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " book1D: MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;	    
  }
  
  TH1F *h1p = reinterpret_cast<TH1F*>(source->Clone());
  h1p->SetName(name.c_str());
  // h1p->Reset();
  // remove histogram from gDirectory so we can release memory with remove method
  h1p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH1(h1p, name);
  addElement(me, dir->getPathname(), "TH1F");
  return me;
}

MonitorElement * 
DaqMonitorROOTBackEnd::book1D(string name, string title, int nchX, 
			      double lowX, double highX, 
			      MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " book1D: MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;	    
  }
  
  TH1F *h1p = new TH1F(name.c_str(),title.c_str(),nchX,lowX,highX);
  // remove histogram from gDirectory so we can release memory with remove method
  h1p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH1(h1p, name);
  addElement(me, dir->getPathname(), "TH1F");
  return me;
}

MonitorElement *
DaqMonitorROOTBackEnd::book1D(string name, string title, int nchX,
                              float *xbinsize,
                              MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " book1D: MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  TH1F *h1p = new TH1F(name.c_str(),title.c_str(),nchX,xbinsize);
  // remove histogram from gDirectory so we can release memory with remove method
  h1p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH1(h1p, name);
  addElement(me, dir->getPathname(), "TH1F");
  return me;
}

// clone/copy received TH2F object from <source> to ME in <dir>
MonitorElement *
DaqMonitorROOTBackEnd::book2D(std::string name, TH2F* source,
                              MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }

  TH2F *h2p = reinterpret_cast<TH2F*>(source->Clone());
  h2p->SetName(name.c_str());
  //h2p->Reset();
  // remove histogram from gDirectory so we can release memory with remove method
  h2p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH2(h2p,name);
  addElement(me, dir->getPathname(), "TH2F");
  return me;
}

MonitorElement * 
DaqMonitorROOTBackEnd::book2D(string name, string title, int nchX, 
			      double lowX, double highX, int nchY,double lowY,
			      double highY, MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  TH2F *h2p = new TH2F(name.c_str(),title.c_str(),nchX,lowX,highX,
		       nchY, lowY, highY);
  // remove histogram from gDirectory so we can release memory with remove method
  h2p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH2(h2p,name);
  addElement(me, dir->getPathname(), "TH2F");
  return me;
}

// clone/copy received TH2F object from <source> to ME in <dir>
MonitorElement *
DaqMonitorROOTBackEnd::book3D(std::string name, TH3F* source,
                              MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }

  TH3F *h3p = reinterpret_cast<TH3F*>(source->Clone());
  h3p->SetName(name.c_str());
  //h3p->Reset();
  // remove histogram from gDirectory so we can release memory with remove method
  h3p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH3(h3p,name);
  addElement(me, dir->getPathname(), "TH3F");
  return me;
}

MonitorElement * 
DaqMonitorROOTBackEnd::book3D(string name, string title, int nchX, 
			      double lowX, double highX, int nchY, 
			      double lowY, double highY, int nchZ,
			      double lowZ, double highZ, 
			      MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  TH3F *h3p = new TH3F(name.c_str(),title.c_str(),nchX,lowX,highX,
		       nchY, lowY, highY,
		       nchZ, lowZ, highZ);
 // remove histogram from gDirectory so we can release memory with remove method
  h3p->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootH3(h3p,name);
  addElement(me, dir->getPathname(), "TH3F");
  return me;
}

// clone/copy received TProfile object from <source> to ME in <dir>
MonitorElement *
DaqMonitorROOTBackEnd::bookProfile(std::string name, TProfile* source,
                              MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }

  TProfile *hp = reinterpret_cast<TProfile*>(source->Clone());
  hp->SetName(name.c_str());
  //hp->Reset();
  // remove histogram from gDirectory so we can release memory with remove method
  hp->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootProf(hp,name);
  addElement(me, dir->getPathname(), "TProfile");
  return me;
}

MonitorElement *
DaqMonitorROOTBackEnd::bookProfile(string name, string title, 
				   int nchX, double lowX, double highX, 
				   int nchY, double lowY, double highY,
				   MonitorElementRootFolder * dir, 
				   char * option)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  TProfile *hpp = new TProfile(name.c_str(),title.c_str(),nchX,lowX,highX,
			       lowY, highY, option);
  // remove profile from gDirectory so we can release memory with remove method
  hpp->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootProf(hpp,name);
  addElement(me, dir->getPathname(), "TProfile");
  return me;
}

// clone/copy received TProfile2D object from <source> to ME in <dir>
MonitorElement *
DaqMonitorROOTBackEnd::bookProfile2D(std::string name, TProfile2D* source,
                              MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }

  TProfile2D *hp2d = reinterpret_cast<TProfile2D*>(source->Clone());
  hp2d->SetName(name.c_str());
  //hp2d->Reset();
  // remove histogram from gDirectory so we can release memory with remove method
  hp2d->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootProf2D(hp2d,name);
  addElement(me, dir->getPathname(), "TProfile2D");
  return me;
}

MonitorElement *
DaqMonitorROOTBackEnd::bookProfile2D(string name, string title, 
				     int nchX, double lowX, double highX, 
				     int nchY, double lowY, double highY,
				     int nchZ, double lowZ, double highZ,
				     MonitorElementRootFolder * dir, 
				     char * option)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  TProfile2D *hpp = new TProfile2D(name.c_str(),title.c_str(),
				   nchX,lowX,highX,
				   nchY, lowY, highY, 
				   lowZ, highZ, option);
  // remove profile from gDirectory so we can release memory with remove method
  hpp->SetDirectory(0);
  MonitorElement *me = new MonitorElementRootProf2D(hpp,name);
  addElement(me, dir->getPathname(), "TProfile2D");
  return me;
}

MonitorElement * DaqMonitorROOTBackEnd::bookFloat(string name, 
				       MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;

  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  float *val = new float(0);
  MonitorElement *me =  new MonitorElementRootFloat(val, name);
  addElement(me, dir->getPathname(), "float");
  return me;
}

MonitorElement * DaqMonitorROOTBackEnd::bookInt(string name, 
					MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;

  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  }
  
  int *val = new int(0);
  MonitorElement *me =  new MonitorElementRootInt(val, name);
  addElement(me, dir->getPathname(), "integer");
  return me;
}

MonitorElement * 
DaqMonitorROOTBackEnd::bookString(string name, string value,
				  MonitorElementRootFolder * dir)
{
  if(!dir)return (MonitorElement *) 0;
  MonitorElement *existingme = findObject(name,dir->getPathname());
  if (existingme) { 
    cout << " MonitorElement " << existingme->getFullname() << 
            " already existing, returning pointer " << endl;
    return existingme;
  } 
  
  string *val = new string(value);
  MonitorElement *me =  new MonitorElementRootString(val, name);
  addElement(me, dir->getPathname(), "string");
  return me;
}

void DaqMonitorROOTBackEnd::showDirStructure(void) const
{
  vector<string> contents; getContents(contents);
  string longline =" ------------------------------------------------------------";
  cout << longline << endl;
  cout <<"                    Directory structure:                     " << endl;
  cout << longline << endl;
  for(vIt it = contents.begin(); it != contents.end(); ++it)
    cout << *it << endl;
  cout << longline << endl;
}

// true if fCurrentFolder is the root folder (gROOT->GetRootFolder() )
bool DaqMonitorROOTBackEnd::isRootFolder(void)
{
  return (((TFolder*) fCurrentFolder->operator->()) == 
	  gROOT->GetRootFolder());
}

// add monitoring element to current folder
void DaqMonitorROOTBackEnd::addElement(MonitorElement * me, string pathname, 
				       string type)
{
  if(pathname == ROOT_PATHNAME && first_time_onRoot)
    {
      cout << " DaqMonitorROOTBackEnd info: no directory has been " << 
	"\n specified to store monitoring objects; using root..." << endl;
      first_time_onRoot = false;
    }

  lock();
  MonitorElementRootFolder * folder = makeDirectory(pathname, Own);

  if(type != "" && DQM_VERBOSE)
    {
      cout << " Adding new " << type << " object " << me->getName() 
	   << " to";
      if(pathname == ROOT_PATHNAME)
	cout << " top directory " << endl;
      else
	cout << " directory " << pathname << endl;
    }

  folder->addElement(me);

 // add to addedContents
  monit_it It = addedContents.find(pathname);
  if(It == addedContents.end())
    {
      set<string> temp; temp.insert(me->getName());
      addedContents[pathname] = temp;
    }
  else
    It->second.insert(me->getName());

  unlock();
}

// get all contents;
// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
// if showContents = false, change form to <dir pathname>:
// (useful for subscription requests; meant to imply "all contents")
void DaqMonitorROOTBackEnd::getContents(std::vector<string> & put_here,
					bool showContents) const
{
  get(put_here, false, showContents);
}

// get monitorable;
// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
// if showContents = false, change form to <dir pathname>:
// (useful for subscription requests; meant to imply "all contents")
void DaqMonitorROOTBackEnd::getMonitorable(std::vector<string> & put_here,
					   bool showContents) const
{
  get(put_here, true, showContents);
}


// to be called by getContents (flag = false) or getMonitorable (flag = true)
// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>
// if showContents = false, change form to <dir pathname>:
// (useful for subscription requests; meant to imply "all contents")
void DaqMonitorROOTBackEnd::get(std::vector<string> & put_here, bool monit_flag,
				bool showContents) const
{
  put_here.clear();
  for(cdir_it fold = Own.paths.begin(); fold != Own.paths.end(); 
      ++fold){
    MonitorElementRootFolder * folder = fold->second;
    // skip empty directories (ie. w/o any null, or, non-null ME)
    if(folder->objects_.empty())
      continue;
    
    // skip null MEs if called by getContents, do not if called by getMonitorable
    bool skipNull = !monit_flag;
    string contents = folder->getChildren(skipNull);
    // skip empty directories
    if(contents.empty())
      continue;

    if(showContents)
      contents = folder->pathname_ + ":" + contents;
    else
      contents = folder->pathname_ + ":";
    
    put_here.push_back(contents);
  }

}


// *** Use this for saving monitoring objects in ROOT files with dir structure ***
// cd into directory (create first if it doesn't exist)
// returns success flag 
bool DaqMonitorROOTBackEnd::cdInto(string inpath) const
{
  if(isTopFolder(inpath)) return gDirectory->cd("/");
  
  // input path
  dqm::Tokenizer path("/",inpath);
  // get top directory --> *subf
  dqm::Tokenizer::const_iterator subf = path.begin();

  // loop: look at all (sub)directories specified by "path"
  while(subf != path.end() && (*subf) != "") { 
    TObject * dir = gDirectory->Get(subf->c_str());
    if(dir) {
      if(dir->IsA() != TDirectory::Class() ){
	 cerr << " *** Error! " << *subf << " is not a directory " << endl;
	 return false;
      }
    }
    else gDirectory->mkdir(subf->c_str());

    if(!gDirectory->cd(subf->c_str())) {
	cerr << " *** Error! Cannot cd into " << *subf << endl;
	return false;
    }

    // go down one (sub)directory
    ++subf;
  } // loop: look at all (sub)directories specified by "inpath"

  return true;
}

// update directory structure maps for folder
void DaqMonitorROOTBackEnd::updateMaps(MonitorElementRootFolder * dir, 
				       rootDir & rDir)
{
  rDir.paths[dir->getPathname()] = dir;
}

// save dir_fullpath with monitoring objects into root file <filename>;
// include quality test results with status >= minimum_status 
// (defined in Core/interface/QTestStatus.h);
// if dir_fullpath="", save full monitoring structure
void DaqMonitorROOTBackEnd::save(string filename, string dir_fullpath,
				 int minimum_status)
{
  lock();
  TFile f(filename.c_str(), "RECREATE");
  TObjString version = TObjString((TString)edm::getReleaseVersion());
  version.Write(); // write CMSSW version to output file
  TObjString DQMpatchversion = TObjString((TString)getDQMPatchVersion());
  DQMpatchversion.Write(); // write DQM patch version to output file

  if(f.IsZombie()) {
      cerr << " *** Error! Failed creating filename " << filename << endl;
      unlock();
      return;
  }
  f.cd();
    
  for(dir_it path = Own.paths.begin(); path != Own.paths.end(); ++path) { 
    // loop over directory structure
    // gDirectory->cd("/"); 

    // consider only subdirectories of <dir_fullpath>
    if(dir_fullpath != "" && !isSubdirectory(dir_fullpath, path->first)) continue;

    MonitorElementRootFolder * folder = path->second;

    vME children;
    folder->getContents(children);
    // skip directories w/o any monitoring elements
    if (children.empty()) continue;

    // loop over monitoring elements in directory
    for(vMEIt me = children.begin(); me != children.end(); ++me) { 

      if (!(*me)) continue ;
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*> (*me);

      // do not store reference histograms (unless there is a quality test attached)
      if ((path->first).find(referenceDirName+"/")==(unsigned int)0) {
         //// look for qtest result for corresponding me
	 string refmename = (*me)->getFullname();
	 string mename = "";
	 unsigned length_to_chop = referenceDirName.size();
	 if (refmename.substr(0, length_to_chop) == referenceDirName) {
	     // remove another character ("/") if necessary
	     if (refmename != referenceDirName) ++length_to_chop; 
	     mename  = refmename.substr(length_to_chop, refmename.size());
         }
	 MonitorElement* bme = get(mename) ;
	 if (bme) {
       	    if ((bme->getQReports()).size() == 0 ) {
	       if (DQM_VERBOSE>1) cout << "DaqMonitorROOTBackEnd::save : skipping me \"" 
	                         << path->first << "\"" << endl;
 	    continue;
	    }
	 }
	 else { 
	    if (DQM_VERBOSE>1) cout << "DaqMonitorROOTBackEnd::save : skipping me \"" 
	                         << path->first << "\"" << endl;
            continue;
	 }
      }

      if (DQM_VERBOSE) cout << "DaqMonitorROOTBackEnd::save : saving me \"" 
                            << (*me)->getFullname() << "\"" << endl;

      // create directory
      gDirectory->cd("/");
      if (isTopFolder(path->first)) cdInto ( monitorDirName );
      else cdInto (monitorDirName + "/" + path->first);

      // save me
      if (ob) ob->operator->()->Write(); // <-- works ok (default)
      else {
	FoldableMonitor * fm = dynamic_cast<FoldableMonitor *> (*me);
	if((path->first).find(referenceDirName+"/")==(unsigned int)0) continue;
	if (fm) fm->getRootObject()->Write();
      }

      if ((path->first).find(referenceDirName+"/")==(unsigned int)0) continue;

      // save quality tests' results
      QR_map qreports = (*me)->getQReports();
      // loop over ME's qreports
      for (qr_it it = qreports.begin(); it !=  qreports.end(); ++it) { 
	MERootQReport * mqr = (MERootQReport *) it->second;
	if(!mqr) continue;
	FoldableMonitor * qr = dynamic_cast<FoldableMonitor *> (mqr);
	if (qr) qr->getRootObject()->Write();
	else cerr << " *** Failed to save quality test result for " << mqr->getName() << endl;
      } // loop over ME's qreports

    } // loop over monitoring elements in directory
  } // loop over directory structure

  f.Close();
  unlock();
  cout << " Saved DQM file " << filename  << endl;
}

void DaqMonitorROOTBackEnd::readReferenceME(std::string filename) {
  open(filename,false,"",referenceDirName); 
}

std::string DaqMonitorROOTBackEnd::getFileReleaseVersion(std::string filename) 
{
  TFile f(filename.c_str());
  if(f.IsZombie())
    {
      cerr << " *** Error! Failed opening filename " << filename << endl;
      f.Close();
      return "";
    }
  TIter nextkey( gDirectory->GetListOfKeys() );

  string name = "none" ;  
  TKey *key;
  while ( (key = (TKey*)nextkey())) {
    TObject *obj = key->ReadObj();
    name = (string)(obj->GetName());
    if (name.find("CMSSW")==1) break;
  }
  f.Close() ;
  return name ;
}

std::string DaqMonitorROOTBackEnd::getFileDQMPatchVersion(std::string filename)
{
  TFile f(filename.c_str());
  if(f.IsZombie())
    {
      cerr << " *** Error! Failed opening filename " << filename << endl;
      f.Close();
      return "";
    }
  TIter nextkey( gDirectory->GetListOfKeys() );

  string name = "none" ;  
  TKey *key;
  while ( (key = (TKey*)nextkey())) {
    TObject *obj = key->ReadObj();
    name = (string)(obj->GetName());
    if (name.find("DQMPATCH")==0) break;
  }
  f.Close() ;
  return name ;
}


/// open/read root file <filename>, and copy MonitorElements;
/// if flag=true, overwrite identical MonitorElements (default: false);
/// if directory != "", read only selected directory
void DaqMonitorROOTBackEnd::open(string filename, bool overwrite, 
				 string directory, string prepend)
{
  cout << " DaqMonitorROOTBackEnd::open : opening ME input file " 
       << filename << " ... " << endl; 
  TFile f(filename.c_str());
  if(f.IsZombie())
    {
      cerr << " *** Error! Failed opening filename " << filename << endl;
      return;
    }

  overwriteFromFile = overwrite;
  readOnlyDirectory = directory;
  unsigned int N = readDirectory(&f, ROOT_PATHNAME, prepend);
  f.Close();
  if(DQM_VERBOSE) {
    cout << " Successfully read " << N << " monitoring objects from file " << filename ;
    if (prepend != "") cout << " and prepended \"" << prepend << "/\" to the path "
	 << endl; 
    } 
  cout << " DaqMonitorROOTBackEnd::open : file " << filename << " closed " << endl;
}


// read ROOT objects from file <file> in directory <orig_pathname>;
// return total # of ROOT objects read
unsigned int 
DaqMonitorROOTBackEnd::readDirectory(TFile* file, string orig_pathname, 
                                     string prepend)
{
  unsigned int tot_count = 0; unsigned int count = 0;
  // ROOT_PATHNAME corresponds to empty string (ie. top of file structure)
  if(orig_pathname == ROOT_PATHNAME) orig_pathname = "";

  if(!file->cd(orig_pathname.c_str())) {
      cout << " *** Failed to unpack directory " << orig_pathname << endl;
      return 0;
  }
  
  // chop-off monitorDirName from beginning of orig_pathname when booking MEs
  string core_pathname = orig_pathname;
  unsigned length_to_chop = monitorDirName.size();
  unsigned orig_length = orig_pathname.size();
  
  if(orig_pathname.substr(0, length_to_chop) == monitorDirName) {
      if(orig_pathname != monitorDirName) ++length_to_chop; // remove another character ("/") if necessary
      core_pathname = orig_pathname.substr(length_to_chop, orig_length);
  }

  // Take chopped pathname apart:
  unsigned n = core_pathname.find_first_of("/");
  string head_pathname = core_pathname.substr(0,n);
  string tail_pathname = core_pathname.substr(n+1,core_pathname.length());

    /*cout << " pathnames: " << prepend << ":" << head_pathname << ":" 
                         << tail_pathname << ":" 
  			 << prepend.find("Run") << ":" 
  			 << prepend.find_first_of("Run") << ":"
  			 << prepend.find("0") << endl;
    */  
  
  string prep_pathname = core_pathname ;
  if ( prepend == "Collate" || prepend == referenceDirName ) { 
     if ( head_pathname != referenceDirName ) {
          prep_pathname = prepend + "/" + core_pathname ;
     }
     if ( tail_pathname == "EventInfo" ) return 0;
  } 
//  else if ( prepend.find("Run") == 0 && prepend.find("0") == 3 ) {
//     prep_pathname = prepend + "/" + head_pathname + "/RunSummary/" + tail_pathname ;            
//     // FIXME do sth about Lumi here and use Run from ME contents in file
//  } 
  else if ( prepend != "" ) {
     prep_pathname = prepend + "/" + core_pathname ;
  }
  
  //cout << " " << prepend << " " << core_pathname << " " << prep_pathname << endl;
  //cout << monitorDirName << endl;

  // skip directory if readOnlyDirectory != empty and core_pathname is not subdir
  bool skipDirectory = false;
  if(!readOnlyDirectory.empty() && 
     !isSubdirectory(readOnlyDirectory, core_pathname))
    skipDirectory = true;
  
  // loop over all keys in this directory
  TIter nextkey( gDirectory->GetListOfKeys() );
  
  TKey *key;
  while ( (key = (TKey*)nextkey())) {
    // loop over all ROOT objects
    
    // read object from file
    TObject *obj = key->ReadObj();
    
    if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) 
      { // this is a subdirectory
	
	string subdir = obj->GetName();
	if(!orig_pathname.empty())
	  subdir = orig_pathname + "/" + subdir;
	
	tot_count += readDirectory(file, subdir, prepend);
      }
    else
      {
	if(skipDirectory) continue;

	if(DQM_VERBOSE > 1) cout << " Found object " << obj->GetName() << " of type " 
	       << obj->IsA()->GetName() << endl;
	
	MonitorElementRootFolder * dir = 
	  DaqMonitorBEInterface::makeDirectory(prep_pathname);
	const bool fromRemoteNode = false; // objects read locally
	bool success = extractObject(obj, dir,fromRemoteNode);
	if(success) ++count;
      }
    
  } // loop over all objects

  if(DQM_VERBOSE && count)
    cout << " Read " << count << " ROOT objects from directory " 
	       << orig_pathname << endl;	

  tot_count += count;
  return tot_count;
}

// get folder corresponding to inpath wrt to root (create subdirs if necessary)
MonitorElementRootFolder * 
DaqMonitorROOTBackEnd::makeDirectory(string inpath, rootDir & Dir)
{
  MonitorElementRootFolder * fret = getDirectory(inpath, Dir);
  if(!fret)
    fret = getRootFolder(Dir)->makeDirectory(inpath);

  updateMaps(fret, Dir);
  return fret;
}

// get (pointer to) last directory in inpath (null if inpath does not exist)
MonitorElementRootFolder * 
DaqMonitorROOTBackEnd::getDirectory(string inpath, const rootDir & Dir) const
{
  if (isTopFolder(inpath)) return getRootFolder(Dir);
  else return getRootFolder(Dir)->getDirectory(inpath);
}


// remove monitoring element from directory; 
// if warning = true, print message if element does not exist
void DaqMonitorROOTBackEnd::removeElement(MonitorElementRootFolder * dir, 
					  string name, bool warning)
{
  if(!dir) {
      cerr << " Cannot remove object " << name 
	   << " from null folder " << endl;
      return;
  }

  lock();

  dir->setVerbose(DQM_VERBOSE);

  // first remove monitoring element
  bool wasThere = dir->removeElement(name, warning);
  bool releaseMemory = dir->ownsChildren();
  if(releaseMemory && wasThere) {
      // then add name to "removedContents"
      add2RemovedContents(dir, name);
    }
  unlock();

  if(releaseMemory) {
      // remove <name> from all subscribers' and tags' directories
      removeCopies(dir->getPathname(), name);
      // remove all references in addedTags, removedTags
      tagHelper->removeTags(dir->getPathname(), name);
    }
}

// remove all monitoring elements from directory; 
// if warning = true, print message if element does not exist
void DaqMonitorROOTBackEnd::removeContents(MonitorElementRootFolder * dir)
{
  if(!dir) {
      cerr << " Cannot remove contents of null folder " << endl;
      return;
    }

  lock();

  bool releaseMemory = dir->ownsChildren();
  // first, add name to "removedContents"
  if(releaseMemory)
    add2RemovedContents(dir);
  // then, remove monitoring object
  dir->setVerbose(DQM_VERBOSE);
  dir->removeContents();

  unlock();

  if(releaseMemory)
    // remove contents from all subscribers' and tags' directories
    removeCopies(dir->getPathname());

}


// equivalent to "cd .."
void DaqMonitorROOTBackEnd::goUp(void)
{
  if(fCurrentFolder->parent_)
    setCurrentFolder(fCurrentFolder->parent_->pathname_);
}

// cd to subdirectory (if there)
void DaqMonitorROOTBackEnd::cd(string subdir_path)
{
  string fullpath = subdir_path;
  if(!isRootFolder())
    fullpath = fCurrentFolder->pathname_ + "/" + fullpath;

  if(pathExists(fullpath, Own))
    setCurrentFolder(fullpath);
  else
    cerr << " *** Error! Cannot cd into " << subdir_path << endl;    
}

// delete directory (all contents + subfolders)
void DaqMonitorROOTBackEnd::rmdir(const string & pathname, rootDir & rDir)
{
  MonitorElementRootFolder * dir = getDirectory(pathname, rDir);
  if(!dir)
    {
      cerr << " *** Command rmdir failed " << endl;
      return;
    }

  if(getRootFolder(Own) == dir)
    {
      cerr << " *** Error! Removal of root folder is not allowed! " << endl;
      return;
    }

  // go to parent directory if current-folder is subdirectory of dir to be removed
  if(getRootFolder(rDir) == getRootFolder(Own) && 
     isSubdirectory(pathname, pwd()) )
    {
      if(dir->parent_)
	fCurrentFolder = dir->parent_;
      else
	{
	  cerr << " *** Cannot cd to parent folder of directory "
	       << pathname << endl;
	  cd();
	}
    }

  // first remove from internal structure
  removeReferences(pathname, rDir);

  dir->setVerbose(DQM_VERBOSE);
  // then delete directory
  delete dir;
}

// remove directory (wrt root)
void DaqMonitorROOTBackEnd::rmdir(string inpath)
{
  if(!pathExists(inpath, Own))
    {
      cerr << " *** Directory " << inpath << " does not exist " << endl;
      cerr << " *** Command rmdir failed " << endl;
      return;
    }
  rmdir(inpath, Own);
}

// remove all references for directories starting w/ pathname;
// put contents of directories in removeContents (if applicable)
void DaqMonitorROOTBackEnd::removeReferences(string pathname, rootDir & rDir)
{
  // this is a little tricky: we have to erase not just the entry "pathname"
  // in map "rDir.paths", but also every entry that begins
  // with the same pathname (ie. subdirectories of that folder)
  
  vector<string> to_remove;
  for(dir_it it = rDir.paths.begin(); it != rDir.paths.end(); ++it)
    { // loop over all pathnames
      if(isSubdirectory(pathname, it->first))
	{
	  // this ensures that we won't remove eg. dir_2 
	  // when we want to remove just dir...
	  to_remove.push_back(it->first);
	  // empty contents of that directory
	  MonitorElementRootFolder * dir = getDirectory(it->first, rDir);
	  removeContents(dir);
	}
      
    } // loop over all pathnames
  
  for(vIt it = to_remove.begin(); it != to_remove.end(); ++it)
    rDir.paths.erase(*it);
}

// copy monitoring elements from source to destination
void DaqMonitorROOTBackEnd::copy(const MonitorElementRootFolder * const source, 
				 MonitorElementRootFolder * const dest, 
				 const vector<string> & contents)
{
  if(!source || !dest)
    {
      cerr << " Command copy failed " << endl;
      return;
    }

  // now look for monitoring objects
  for(cvIt it = contents.begin(); it != contents.end(); ++it)
    {
      // get object from source directory
      MonitorElement * me = source->findObject(*it);
      if(!me)
	{
	  cerr << " *** Object " << (*it) << " does not exist in directory " 
	       << source->getPathname() << endl;
	  cerr << " Ignored copy command " << endl;
	  continue;
	}

      // copy object
      dest->addElement(me);
    }
  
}

// remove subscribed monitoring elements;
// if warning = true, printout error messages when problems
void 
DaqMonitorROOTBackEnd::removeSubsc(MonitorElementRootFolder * const dir, 
				   const vector<string> & contents, 
				   bool warning)
{
  if(!dir)
    {
      cerr << " Command removeSubsc failed " << endl;
      return;
    }

  // now look for monitoring objects
  for(cvIt it = contents.begin(); it != contents.end(); ++it)
    {

      // get object from directory
      MonitorElement * me = dir->findObject(*it);
      if(!me)
	{
	  if(warning)
	    {
	      cerr << " *** Object " << (*it) 
		   << " does not exist in directory " 
		   << dir->getPathname() << endl;
	      cerr << " Ignored unsubscribe command " << endl;
	    }
	  continue;
	}

      // remove object from directory
      removeElement(dir, *it, warning);
    }
  
  // if contents: empty, remove all contents of directory
  if(contents.empty())
    removeContents(dir);
  
}

// add <name> of folder to private method removedContents
void 
DaqMonitorROOTBackEnd::add2RemovedContents(const MonitorElementRootFolder * subdir,
					   string name)
{
  string pathname = subdir->getPathname();
  monit_it It = removedContents.find(pathname);
  if(It == removedContents.end())
    {
      set<string> temp; temp.insert(name);
      removedContents[pathname] = temp;
    }
  else
    It->second.insert(name);
}

// add contents of folder to private method removedContents
void 
DaqMonitorROOTBackEnd::add2RemovedContents(const MonitorElementRootFolder * subdir)
{
  for(cME_it it = subdir->objects_.begin(); it != subdir->objects_.end(); 
      ++it)
    add2RemovedContents(subdir, it->first);
}

// get first non-null ME found starting in path; null if failure
MonitorElement * DaqMonitorROOTBackEnd::getMEfromFolder(cdir_it & path) const
{
  // get starting value for path;
  cdir_it start_path = path;
  // if at end of folder list, go to beginning
  if(path == Own.paths.end())
    path = Own.paths.begin();
  // if still at the end, list is empty; quit
  if(path == Own.paths.end())
    return (MonitorElement * )0;
  
  // find 1st non-null ME in 1st non-empty folder
  do{
    MonitorElementRootFolder * folder = path->second;
    for(cME_it it = folder->objects_.begin(); 
	it != folder->objects_.end(); ++it)
      {
	// try to find 1st non-null ME in directory
	if(it->second)
	  return it->second;
      }

    // did not find non-null ME; go to next one
    ++path; 
    // if we've reached the end, move to beginning (unless we started here)
    if(path == Own.paths.end() && path != start_path)
      path = Own.paths.begin();

  }
  // stay here till we reach the starting value
  while(path != start_path);
  
  return (MonitorElement *) 0;
}

// get first ME found following obj_name of "path"; null if failure
MonitorElement * 
DaqMonitorROOTBackEnd::getMEfromFolder(cdir_it & path, string obj_name) const
{
  MonitorElementRootFolder * folder = path->second;
  // look for name in ME_map
  ME_it obj = folder->objects_.find(obj_name);
  if(obj == folder->objects_.end())
    {
      // name does not exist (ie. has been removed); move on to next folder
      ++path;
      return getMEfromFolder(path);
    }
  else
    {
      do
	{
	  ++obj;
	  if(obj == folder->objects_.end())
	    {
	      // name exists but is last one in folder; move on to next folder
	      ++path;
	      return getMEfromFolder(path);
	    }
	}
      // add a non-null ptr check to account for non-subscribed objects
      while(!obj->second);

      // got next ME in folder
      return obj->second;
    }

}

// add null monitoring element to folder <pathname> (can NOT be folder);
// used for registering monitorables before user has subscribed to <name>
void DaqMonitorROOTBackEnd::addElement(std::string name, std::string pathname)
{
  lock();
  MonitorElementRootFolder * folder = 
    DaqMonitorBEInterface::makeDirectory(pathname);
  folder->addElement(name);
  unlock();
}



// look for object <name> in directory <pathname>
MonitorElement * DaqMonitorROOTBackEnd::findObject(string name, string pathname) 
  const
{
  MonitorElementRootFolder * folder = getDirectory(pathname, Own);
  if(!folder)
    return (MonitorElement *) 0;

  return folder->findObject(name);
}

// get list of subdirectories of current directory
vector<string> DaqMonitorROOTBackEnd::getSubdirs(void) const
{
  vector<string> ret; ret.clear();
  for(cdir_it it = fCurrentFolder->subfolds_.begin(); 
      it != fCurrentFolder->subfolds_.end(); ++it)
    ret.push_back(it->first);
  
  return ret;    
}

// get list of (non-dir) MEs of current directory
vector<string> DaqMonitorROOTBackEnd::getMEs(void) const
{
  vector<string> ret; ret.clear();
  for(cME_it it = fCurrentFolder->objects_.begin(); 
      it != fCurrentFolder->objects_.end(); ++it)
    ret.push_back(it->first);
  
  return ret;    
}

// true if directory (or any subfolder at any level below it) contains
// at least one valid (i.e. non-null) monitoring element
bool DaqMonitorROOTBackEnd::containsAnyMEs(string pathname) const
{
  MonitorElementRootFolder * folder = getDirectory(pathname, Own);
  if(!folder)
    {
      cerr << " *** Directory " << pathname << " does not exist! " << endl;
      cerr << " Cmd containsAnyMEs failed..." << endl;
      return false;
    }

  return folder->containsAnyMEs();
}

// true if directory (or any subfolder at any level below it) contains
// at least one monitorable element
bool DaqMonitorROOTBackEnd::containsAnyMonitorable(string pathname) const
{
  MonitorElementRootFolder * folder = getDirectory(pathname, Own);
  if(!folder)
    {
      cerr << " *** Directory " << pathname << " does not exist! " << endl;
      cerr << " Cmd containsAnyMonitorable failed..." << endl;
      return false;
    }
  
  return folder->containsAnyMonitorable();
}

// true if Monitoring Element <me> is needed by any subscriber
bool DaqMonitorROOTBackEnd::isNeeded(string pathname, string me) const
{
  // we will loop over Subscribers
  // and check if directory <pathname> exists
  for(csdir_it subs = Subscribers.begin();subs != Subscribers.end(); ++subs)
    { // loop over all subscribers
      MonitorElementRootFolder * dir = getDirectory(pathname, subs->second);
      // skip subscriber if no such pathname
      if(!dir)continue;
      if(dir->findObject(me)) return true;
    } // loop over all subscribers

  // if here, there are no subscribers that need this monitoring element
  return false;

  // NOTE: I may want to look in (private member) request2add for object <me>;
  // if it is there, this method should return true!
  // I need to think 1. if it's possible to have a request to unsubscribe and a
  // request to subscribe (from two different clients) simultaneously, and
  // 2. how I'd implement this...
}

// add quality report (to be called when test is to run locally)
QReport * DaqMonitorROOTBackEnd::addQReport(MonitorElement * me, QCriterion * qc)
  const
{
  if(!qc)
    {
      cerr << " *** Cannot add QReport with null QCriterion for " 
	   << " MonitorElement " << me->getName() << endl;
      return (QReport *) 0;
    }

  return addQReport(me, qc->getName(), qc);
}

// add quality report (to be called by ReceiverBase)
QReport * DaqMonitorROOTBackEnd::addQReport(MonitorElement * me, string qtname, 
					    QCriterion * qc) const
{
  if(!me)
    {
      cerr << " *** Cannot add QReport " << qtname << " to null MonitorElement!"
	   << endl;
      return (QReport *) 0;
    }

  // check if qtest already exists for this ME
  if(qreportExists(me, qtname))
    return (QReport *) 0;
  
  string * val = new string(QReport::getNullMessage());
  QReport * qr = 0;
  if(qc)
    qr = new MERootQReport(val, me->getName(), qtname, qc);
  else
    qr = new MERootQReport(val, me->getName(), qtname);

  DaqMonitorBEInterface::addQReport(me, qr);
  return qr;
}

// create quality test with unique name <qtname> (analogous to ME name);
// quality test can then be attached to ME with useQTest method
// (<algo_name> must match one of known algorithms)
QCriterion * DaqMonitorROOTBackEnd::createQTest(string algo_name, string qtname)
{
  bool canCreate = true;

  if(availableAlgorithms.find(algo_name) == availableAlgorithms.end())
    {
      cerr << " *** Do not know how to create algorithm with name " 
	   << algo_name << " ! " << endl;
      canCreate = false;
    }

  if(getQCriterion(qtname))
    {
      cerr << " *** Quality test " << qtname << " has already been defined! "
	   << endl;
      canCreate = false;
    }

  if(!canCreate)
    {
      cerr << " Creation of quality test " << qtname << " has failed "<< endl;
      return (QCriterion *) 0;
    }

  QCriterion * qc = 0;
  if(algo_name == Comp2RefChi2ROOT::getAlgoName())
    qc = new MEComp2RefChi2ROOT(qtname);
  else if(algo_name == Comp2RefKolmogorovROOT::getAlgoName())
    qc = new MEComp2RefKolmogorovROOT(qtname);
  else if(algo_name == ContentsXRangeROOT::getAlgoName())
    qc = new MEContentsXRangeROOT(qtname);
  else if(algo_name == ContentsYRangeROOT::getAlgoName())
    qc = new MEContentsYRangeROOT(qtname);
  else if(algo_name == ContentsYRangeROOT::getAlgoName())
    qc = new MEContentsYRangeROOT(qtname);
  else if(algo_name == Comp2RefEqualStringROOT::getAlgoName())
    qc = new MEComp2RefEqualStringROOT(qtname);
  else if(algo_name == Comp2RefEqualIntROOT::getAlgoName())
    qc = new MEComp2RefEqualIntROOT(qtname);
  else if(algo_name == Comp2RefEqualFloatROOT::getAlgoName())
    qc = new MEComp2RefEqualFloatROOT(qtname);
  else if(algo_name == Comp2RefEqualH1ROOT::getAlgoName())
    qc = new MEComp2RefEqualH1ROOT(qtname);
  else if(algo_name == Comp2RefEqualH2ROOT::getAlgoName())
    qc = new MEComp2RefEqualH2ROOT(qtname);
  else if(algo_name == Comp2RefEqualH3ROOT::getAlgoName())
    qc = new MEComp2RefEqualH3ROOT(qtname);
  else if(algo_name == MeanWithinExpectedROOT::getAlgoName())
    qc = new MEMeanWithinExpectedROOT(qtname);
  else if(algo_name == DeadChannelROOT::getAlgoName())
    qc = new MEDeadChannelROOT(qtname);
  else if(algo_name == NoisyChannelROOT::getAlgoName())
    qc = new MENoisyChannelROOT(qtname);
  else if(algo_name == MostProbableLandauROOT::getAlgoName())
    qc = new MEMostProbableLandauROOT(qtname);
    
  else if(algo_name == ContentsTH2FWithinRangeROOT::getAlgoName())
    qc = new MEContentsTH2FWithinRangeROOT(qtname);
  else if(algo_name == ContentsProfWithinRangeROOT::getAlgoName())
    qc = new MEContentsProfWithinRangeROOT(qtname);
  else if(algo_name == ContentsProf2DWithinRangeROOT::getAlgoName())
    qc = new MEContentsProf2DWithinRangeROOT(qtname);

  qtests_[qtname] = qc;
  return qc;
}

// get "global" folder <inpath> status (one of: STATUS_OK, WARNING, ERROR, OTHER);
// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
// see Core/interface/QTestStatus.h for details on "OTHER" 
int DaqMonitorROOTBackEnd::getStatus(std::string inpath) const
{
  if(isTopFolder(inpath))
    return getRootFolder(Own)->getStatus();

  MonitorElementRootFolder * folder = getDirectory(inpath, Own);
  if(!folder)
    {
      cerr << " *** Cannot determine status for unknown directory = " 
	   << inpath << endl;
      return -1;
    }

  return folder->getStatus();
}

// same as above for tag;
int DaqMonitorROOTBackEnd::getStatus(unsigned int tag) const
{
  MonitorElementRootFolder * folder = 0;
  tdir_it tg;
  if(tagHelper->getTag(tag, tg, false)) // do not create
    folder = getDirectory(ROOT_PATHNAME, tg->second);

  if(!folder)
    {
      cerr << " *** Cannot determine status for unknown tag = " 
	   << tag << endl;
      return -1;
    }
  return folder->getStatus();
}

// same as scanContents in base class but for one path only
void DaqMonitorROOTBackEnd::scanContents
(const string & search_string, const MonitorElementRootFolder * folder,
 vector<MonitorElement *> & put_here) const
{
  string pathname = folder->getPathname();
  for(cME_it it = folder->objects_.begin(); 
      it != folder->objects_.end(); ++it)
    { // loop over files of folder
	
      // skip null MonitorElements
      if(!it->second)
	continue;
      
      string fullname = getUnixName(pathname, it->first);
      
      if(matchit(fullname, search_string))
	// this is a match!
	put_here.push_back(it->second);
    } // loop over files of folder

}

// run quality tests (also finds updated contents in last monitoring cycle,
// including newly added content) <-- to be called only by runQTests
void DaqMonitorROOTBackEnd::runQualityTests(void)
{
  for(cdir_it path = Own.paths.begin(); path != Own.paths.end(); ++path)
    { // loop over all pathnames 
      
      MonitorElementRootFolder * folder = path->second;
      if(!folder) throw folder;
      
      string pathname = folder->getPathname();
      if ((pathname).find(referenceDirName+"/")==(unsigned int)0) continue;

      for(cME_it it = folder->objects_.begin(); 
	  it != folder->objects_.end(); ++it)
	{ // loop over monitoring objects in current folder
	  
	  // skip MEs that appear only on monitorable
	  if(!it->second) 
	    continue;
	  
	  if(it->second->wasUpdated())
	    add2UpdatedContents(it->first, pathname);
	  
	  // quality tests should be run if (a) ME has been modified, or
	  // (b) algorithm has been modified; 
	  // this is done in MonitorElement::runQTests()
	  it->second->runQTests();
	  
	} // loop over monitoring objects in current folder

    } // loop over all pathnames 

}

// get ME from full pathname (e.g. "my/long/dir/my_histo")
MonitorElement * DaqMonitorROOTBackEnd::get(string fullpath) const
{
  string path, filename;
  StringUtil::unpack(fullpath, path, filename); 	 
  MonitorElementRootFolder * dir = getDirectory(path, Own); 
  if(!dir) return (MonitorElement *)0; 	 
  return dir->findObject(filename);
}

MonitorElement * DaqMonitorROOTBackEnd::getReferenceME(MonitorElement* me) const
{
  if (me) return findObject(me->getName(),referenceDirName+"/"+me->getPathname());
  else {
    cerr << " MonitorElement " << me->getPathname() << "/" << 
              me->getName() << " does not exist! " << endl;
    return (MonitorElement*) 0 ;
  }
}

bool DaqMonitorROOTBackEnd::isReferenceME(MonitorElement* me) const
{
  if (me && (me->getPathname().find(referenceDirName+"/")==0)) return true; 
  return false;
}

bool DaqMonitorROOTBackEnd::isCollateME(MonitorElement* me) const
{
  if (me && (me->getPathname().find(collateDirName+"/")==0))  // check that collate histos themselves 
	                                                      // are not picked up for additional 
							      // collation
	    return true; 
  return false;
}

bool DaqMonitorROOTBackEnd::makeReferenceME(MonitorElement* me)
{
    string refpathname = referenceDirName+"/"+me->getPathname();

    MonitorElementRootFolder * refdir = 
          DaqMonitorBEInterface::makeDirectory(refpathname);

    MonitorElementT<TNamed>* ob = 
          dynamic_cast<MonitorElementT<TNamed>*> (me);
    if(ob)
      {
      TObject * tobj = dynamic_cast<TObject *> (ob->operator->());
      const bool fromRemoteNode = false; // objects read locally
      bool success = extractObject(tobj, refdir,fromRemoteNode);
      return success ;
      }
    return false;
}

void DaqMonitorROOTBackEnd::deleteME(MonitorElement* me)
{
    string pathname = me->getPathname();
    MonitorElementRootFolder * dir = 
          DaqMonitorBEInterface::makeDirectory(pathname);
    removeElement(dir,me->getName());
}

// get all MonitorElements tagged as <tag>
vector<MonitorElement *> DaqMonitorROOTBackEnd::get(unsigned int tag) const
{
  vector<MonitorElement *> ret;
  ctdir_it tg = Tags.find(tag);
  if(tg == Tags.end())
    return ret;
  
  get(tg->second.paths, ret);
  return ret;
}

// add all (tagged) MEs to put_here
void DaqMonitorROOTBackEnd::get(const dir_map & Dir, 
				vector<MonitorElement *> & put_here) const
{
  // loop over all pathnames
  for(cdir_it it = Dir.begin(); it != Dir.end(); ++it)
    {
      MonitorElementRootFolder * dir = it->second;
      // no reason to call getAllContents, since all subdirectories are listed
      // explicitly in tag structure
      dir->getContents(put_here);
    }
}



// get vector with all children of folder
// (does NOT include contents of subfolders)
vector<MonitorElement *> DaqMonitorROOTBackEnd::getContents(string pathname) 
  const
{
  vector<MonitorElement *> ret;
  getContents(pathname, Own, ret);
  return ret;
}

// same as above for tagged MonitorElements
vector<MonitorElement *> 
DaqMonitorROOTBackEnd::getContents(string pathname, unsigned int tag) const
{
  vector<MonitorElement *> ret;
  ctdir_it tg = Tags.find(tag);
  if(tg != Tags.end())
    getContents(pathname, tg->second, ret);

  return ret;
}

// get vector with all children of folder in <rDir>
// (does NOT include contents of subfolders)
void DaqMonitorROOTBackEnd::getContents
(string & pathname, const rootDir & rDir,
 vector<MonitorElement *> & put_here) const
{
  chopLastSlash(pathname);

  MonitorElementRootFolder * dir = getDirectory(pathname, rDir); 
  if(dir)
    dir->getContents(put_here);
}

// get vector with children of folder, including all subfolders + their children;
// exact pathname: FAST
// pathname including wildcards (*, ?): SLOW!
vector<MonitorElement*> DaqMonitorROOTBackEnd::getAllContents
(string pathname) const
{
  vector<MonitorElement *> ret;
  getAllContents(pathname, Own, ret);
  return ret;
}

// same as above for tagged MonitorElements
vector<MonitorElement*> DaqMonitorROOTBackEnd::getAllContents
(string pathname,  unsigned int tag) const
{
  vector<MonitorElement *> ret;
  ctdir_it tg = Tags.find(tag);
  if(tg == Tags.end()) return ret;
  getAllContents(pathname, tg->second, ret);
  return ret;
}
 
// get vector with all children of folder and all subfolders of <rDir>;
// pathname may include wildcards (*, ?) ==> SLOW!
void DaqMonitorROOTBackEnd::getAllContents
(string & pathname, const rootDir & rDir, vector<MonitorElement*> & put_here) 
  const
{
  // simple case: no wildcards ==> single directory
  if(!hasWildCards(pathname))
    {
      chopLastSlash(pathname);
      MonitorElementRootFolder * dir = getDirectory(pathname, rDir); 
      if(dir)
	dir->getAllContents(put_here);
    }
  else
    // case of search-string with wildcards
    DaqMonitorBEInterface::scanContents(pathname, rDir, put_here);

}

// make new directory structure for Subscribers, Tags and CMEs
void DaqMonitorROOTBackEnd::makeDirStructure(rootDir & Dir, string name)
{
  string ftitle = name + "_folder";
  TFolder * ff = new TFolder(name.c_str(), ftitle.c_str());
  Dir.top = new MonitorElementRootFolder(ff, ff->GetName());
  Dir.top->pathname_ = ROOT_PATHNAME;	  
  Dir.top->ownsChildren_ = false;
  Dir.top->setRootName();
}

// tag ME as <myTag>
void DaqMonitorROOTBackEnd::tag(MonitorElement * me, unsigned int myTag)
{
  tagHelper->tag(me, myTag);
}

// opposite action of tag method
void DaqMonitorROOTBackEnd::untag(MonitorElement * me, unsigned int myTag)
{
  tagHelper->untag(me, myTag);
}

// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
void DaqMonitorROOTBackEnd::tag(string fullpathname, unsigned int myTag)
{
  tagHelper->tag(fullpathname, myTag);
}

// opposite action of tag method
void DaqMonitorROOTBackEnd::untag(string fullpathname, unsigned int myTag)
{
  tagHelper->untag(fullpathname, myTag);
}

// tag all children of folder (does NOT include subfolders)
void DaqMonitorROOTBackEnd::tagContents(string pathname, unsigned int myTag)
{
  tagHelper->tagContents(pathname, myTag);
}

// opposite action of tagContents method
void DaqMonitorROOTBackEnd::untagContents
(string pathname, unsigned int myTag)
{
  tagHelper->untagContents(pathname, myTag);
}

// tag all children of folder, including all subfolders and their children;
// exact pathname: FAST
// pathname including wildcards (*, ?): SLOW!
void DaqMonitorROOTBackEnd::tagAllContents
(string pathname, unsigned int myTag)
{
  tagHelper->tagAllContents(pathname, myTag);
}

// opposite action of tagAllContents method
void DaqMonitorROOTBackEnd::untagAllContents
(string pathname, unsigned int myTag)
{
  tagHelper->untagAllContents(pathname, myTag);
}


/* get all tags, return vector with strings of the form
   <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */
void DaqMonitorROOTBackEnd::getAllTags(vector<string> & put_here) const
{
  tagHelper->getTags(allTags, put_here);
}

void DaqMonitorROOTBackEnd::getAddedTags(vector<string> & put_here) const
{
  tagHelper->getTags(addedTags, put_here);
}

void DaqMonitorROOTBackEnd::getRemovedTags(vector<string> & put_here) const
{
  tagHelper->getTags(removedTags, put_here);
}

// check if added contents belong to folder 
// (use flag to specify if subfolders should be included)
void DaqMonitorROOTBackEnd::checkAddedFolder
(cmonit_it & added_path, const rootDir & Dir, 
 vector<MonitorElement*>& put_here) const
{
  MonitorElementRootFolder * dir = getDirectory(added_path->first, Dir);
  if(!dir)return;
  
  for(csIt it = added_path->second.begin(); 
      it!= added_path->second.end(); ++it)
    {// loop over all added MEs
      MonitorElement * me =  dir->findObject(*it);
      if(me) put_here.push_back(me);
    } // loop over all added MEs

}

// check if added contents match search paths
void DaqMonitorROOTBackEnd::checkAddedSearchPaths
(const vector<string>& search_path, const rootDir & Dir, 
 vector<MonitorElement*>& put_here) const
{
  for(cvIt sp = search_path.begin(); sp != search_path.end(); ++sp)
    { // loop over search-paths
      
      if(!hasWildCards(*sp))
	{
	  string look4path, look4filename;
	  StringUtil::unpack(*sp, look4path, look4filename);
	  
	  cmonit_it added_path = addedContents.find(look4path);
	  if(added_path == addedContents.end())
	    continue;
	  
	  csIt added_name = added_path->second.find(look4filename);
	  if(added_name == added_path->second.end())
	    continue;
	  
	  MonitorElementRootFolder* dir=getDirectory(added_path->first,Dir);
	  if(!dir)continue;
	  
	  MonitorElement * me =  dir->findObject(look4filename);
	  if(me) put_here.push_back(me);
	}
      
      else
	{
	  cmonit_it start, end, parent_dir;
	  getSubRange<monit_map>(*sp, addedContents,start,end,parent_dir);
	  
	  for(cmonit_it path = start; path != end; ++path)
	    // loop over all pathnames of added contents
	    checkAddedContents(*sp, path, Dir, put_here);
	  
	  if(parent_dir != addedContents.end())
	    checkAddedContents(*sp, parent_dir, Dir, put_here);
	  
	}
      
    } // loop over search-paths
}

// same as in base class for given search_string and path; put matches into put_here
void DaqMonitorROOTBackEnd::checkAddedContents
(const string & search_string, cmonit_it & added_path, const rootDir & Dir,
 vector<MonitorElement*> & put_here) const
{
  MonitorElementRootFolder * dir = getDirectory(added_path->first, Dir);
  if(!dir)return;

  for(csIt it = added_path->second.begin(); 
      it!= added_path->second.end(); ++it){
    // loop over all added MEs
    
    // get unix-like filename
    string fullname = getUnixName(added_path->first, *it);

    if(matchit(fullname, search_string))
      {
	// this is a match!
	MonitorElement* me = dir->findObject(*it);
	if(me)
	  put_here.push_back(me);
	
      } // this is a match!
		      
  }  // loop over all added MEs

}

// make copies for <me> for all tags it comes with
// (to be called the first time <me> is received or read from a file)
void DaqMonitorROOTBackEnd::makeTagCopies(MonitorElement * me)
{
  cdirt_it dir = allTags.find(me->getPathname());
  if(dir != allTags.end())
    { // found pathname in allTags

      ctags_it ME_name = (dir->second).find(me->getName());
      if(ME_name != (dir->second).end())
	{ // found ME name in directory
	  const set<unsigned int> & tags = ME_name->second;
	  
	  for(set<unsigned int>::const_iterator t = tags.begin();
	      t != tags.end(); ++t)
	    { // loop over tags for particular ME
	      tdir_it tg;
	      // create if not there
	      if(!tagHelper->getTag(*t, tg, true)) 
		continue;
	      // add shortcut to original ME
	      tagHelper->add(me, tg->second);
	    } // loop over tags for particular ME
	} // found ME name in directory
    } // found pathname in allTags

}

// extract TH1F object from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractTH1F
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TH1F *h1 = dynamic_cast<TH1F*>(to); string nm = h1->GetName();
  MonitorElement * me = dir->findObject(nm);
  if (!wantME(me, dir, nm, fromRemoteNode)) return;
  h1->SetName("extracted");

  if(!me)
    {
      me = book1D ( nm, h1, dir);

    /* fixme cleanup
      me = book1D(nm, h1->GetTitle(), h1->GetNbinsX(), 
		  h1->GetXaxis()->GetXmin(), 
		  h1->GetXaxis()->GetXmax(), dir);
      // set alphanumeric labels, if needed
      if(h1->GetXaxis()->GetLabels())
	{
	  for(int i = 1; i <= h1->GetNbinsX(); ++i)
	    me->setBinLabel(i,h1->GetXaxis()->GetBinLabel(i));
	  MonitorElementRootH1 * local = 
	    dynamic_cast<MonitorElementRootH1 *> (me);
	  ((TH1F*) local->operator->())->GetXaxis()->LabelsOption("v");
	}
      // set axis titles
      me->setAxisTitle(h1->GetXaxis()->GetTitle(), 1);
      me->setAxisTitle(h1->GetYaxis()->GetTitle(), 2);
    */    
      // set canDelete flag if ME arrived from remote node
      if(fromRemoteNode)
	// cannot delete objects arrived from different node
	dir->canDeleteFromMenu[nm] = false;

      makeTagCopies(me);
    } 
  else if (isCollateME(me)) 
    {
        //((TH1F*)((MonitorElementRootH1*)me)->getMonitorable())->Add(h1);
        MonitorElementRootH1* local = dynamic_cast<MonitorElementRootH1 *> (me);
        ((TH1F*) local->operator->())->Add(h1);
        cout << " collated TH1F: " << dir->getPathname() << "/" << nm <<endl;
        return;
    }  
  
  MonitorElementRootH1 * put_here = dynamic_cast<MonitorElementRootH1 *> (me);
  if(put_here) put_here->copyFrom(h1);
}

// extract TH2F object from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractTH2F
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TH2F *h2 = dynamic_cast<TH2F*>(to); string nm = h2->GetName();
  MonitorElement * me = dir->findObject(nm);
  if(!wantME(me, dir, nm, fromRemoteNode))return;
  h2->SetName("extracted");
  
  if(!me)
    {
      me = book2D ( nm, h2, dir);
      /*  me = book2D(nm, h2->GetTitle(), h2->GetNbinsX(), 
		  h2->GetXaxis()->GetXmin(), 
		  h2->GetXaxis()->GetXmax(), h2->GetNbinsY(), 
		  h2->GetYaxis()->GetXmin(), 
		  h2->GetYaxis()->GetXmax(), dir);
      // set alphanumeric labels, if needed
      if(h2->GetXaxis()->GetLabels())
	{
	  for(int i = 1; i <= h2->GetNbinsX(); ++i)
	    me->setBinLabel(i,h2->GetXaxis()->GetBinLabel(i),1);
	  MonitorElementRootH2 * local = 
	    dynamic_cast<MonitorElementRootH2 *> (me);
	  ((TH2F*) local->operator->())->GetXaxis()->LabelsOption("v");
	}
      if(h2->GetYaxis()->GetLabels())
	{
	  for(int i = 1; i <= h2->GetNbinsY(); ++i)
	    me->setBinLabel(i,h2->GetYaxis()->GetBinLabel(i),2);
	}
      // set axis titles
      me->setAxisTitle(h2->GetXaxis()->GetTitle(), 1);
      me->setAxisTitle(h2->GetYaxis()->GetTitle(), 2);
      */
      // set canDelete flag if ME arrived from remote node
      if(fromRemoteNode)
	// cannot delete objects arrived from different node
	dir->canDeleteFromMenu[nm] = false;
      
      makeTagCopies(me);
    }
  else if (isCollateME(me)) 
    {
        MonitorElementRootH2* local = dynamic_cast<MonitorElementRootH2 *> (me);
        ((TH2F*) local->operator->())->Add(h2);
        cout << " collated TH2F: " << dir->getPathname() << "/" << nm <<endl;
        return;
    }  
  
  MonitorElementRootH2 * put_here = dynamic_cast<MonitorElementRootH2 *> (me);
  if(put_here) put_here->copyFrom(h2);
  
}

// extract TProfile object from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractTProf
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TProfile *hp = dynamic_cast<TProfile*>(to); string nm = hp->GetName();
  MonitorElement * me = dir->findObject(nm);
  if(!wantME(me, dir, nm, fromRemoteNode))return;
  hp->SetName("extracted");

  if(!me)
    {
      me = bookProfile ( nm, hp, dir);
      /* fixme cleanup
      me = bookProfile(nm, hp->GetTitle(), hp->GetNbinsX(), 
		       hp->GetXaxis()->GetXmin(), 
		       hp->GetXaxis()->GetXmax(), 
		       hp->GetNbinsY(), 
		       hp->GetYaxis()->GetXmin(),
		       hp->GetYaxis()->GetXmax(), dir,
		       (char *)hp->GetErrorOption());
      // set alphanumeric labels, if needed
      if(hp->GetXaxis()->GetLabels())
	{
	  for(int i = 1; i <= hp->GetNbinsX(); ++i)
	    me->setBinLabel(i,hp->GetXaxis()->GetBinLabel(i),1);
	  MonitorElementRootProf * local = 
	    dynamic_cast<MonitorElementRootProf *> (me);
	  ((TProfile*) local->operator->())->GetXaxis()->LabelsOption("v");
	}
      if(hp->GetYaxis()->GetLabels())
	{
	  for(int i = 1; i <= hp->GetNbinsY(); ++i)
	    me->setBinLabel(i,hp->GetYaxis()->GetBinLabel(i),2);
	}
      // set axis titles
      me->setAxisTitle(hp->GetXaxis()->GetTitle(), 1);
      me->setAxisTitle(hp->GetYaxis()->GetTitle(), 2);
      */
      // set canDelete flag if ME arrived from remote node
      if(fromRemoteNode)
	// cannot delete objects arrived from different node
	dir->canDeleteFromMenu[nm] = false;
      
      makeTagCopies(me);
    }
  else if (isCollateME(me)) 
    {
        MonitorElementRootProf* local = dynamic_cast<MonitorElementRootProf *> (me);
        TProfile* hp2=(TProfile*) local->operator->();
	local->addProfiles((TProfile*)hp,(TProfile*)hp2,(TProfile*)hp2,(float)1.,(float)1.);
        cout << " collated TProfile: " << dir->getPathname() << "/" << nm <<endl;
        return;
    }  
  
  MonitorElementRootProf* put_here=dynamic_cast<MonitorElementRootProf *>(me);
  if(put_here)
  put_here->copyFrom(hp);
}

// extract TProfile2D object from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractTProf2D
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TProfile2D *hp = dynamic_cast<TProfile2D*>(to); string nm = hp->GetName();
  MonitorElement * me = dir->findObject(nm);
  if(!wantME(me, dir, nm, fromRemoteNode))return;
  hp->SetName("extracted");

  if(!me)
    {
      me = bookProfile2D ( nm, hp, dir);
      /* fixme cleanup
      me = bookProfile2D(nm, hp->GetTitle(), hp->GetNbinsX(), 
			 hp->GetXaxis()->GetXmin(), 
			 hp->GetXaxis()->GetXmax(), 
			 hp->GetNbinsY(), 
			 hp->GetYaxis()->GetXmin(),
			 hp->GetYaxis()->GetXmax(),
			 hp->GetNbinsZ(), 
			 hp->GetZaxis()->GetXmin(),
			 hp->GetZaxis()->GetXmax(), dir,
			 (char *)hp->GetErrorOption());
      // set alphanumeric labels, if needed
      if(hp->GetXaxis()->GetLabels())
	{
	  for(int i = 1; i <= hp->GetNbinsX(); ++i)
	    me->setBinLabel(i,hp->GetXaxis()->GetBinLabel(i),1);
	  MonitorElementRootProf2D * local = 
	    dynamic_cast<MonitorElementRootProf2D *> (me);
	  ((TProfile2D*) local->operator->())->GetXaxis()->LabelsOption("v");
	  
	}
      if(hp->GetYaxis()->GetLabels())
	{
	  for(int i = 1; i <= hp->GetNbinsY(); ++i)
	    me->setBinLabel(i,hp->GetYaxis()->GetBinLabel(i),2);
	}
      if(hp->GetZaxis()->GetLabels())
	{
	  for(int i = 1; i <= hp->GetNbinsZ(); ++i)
	    me->setBinLabel(i,hp->GetZaxis()->GetBinLabel(i),3);
	}
      // set axis titles
      me->setAxisTitle(hp->GetXaxis()->GetTitle(), 1);
      me->setAxisTitle(hp->GetYaxis()->GetTitle(), 2);
      me->setAxisTitle(hp->GetZaxis()->GetTitle(), 3);
      */
      // set canDelete flag if ME arrived from remote node
      if(fromRemoteNode)
	// cannot delete objects arrived from different node
	dir->canDeleteFromMenu[nm] = false;
      
      makeTagCopies(me);
    }
  else if (isCollateME(me))
    {
      MonitorElementRootProf2D* local = dynamic_cast<MonitorElementRootProf2D *> (me);
      TProfile2D* hp2=(TProfile2D*) local->operator->();
      local->addProfiles((TProfile2D*)hp,(TProfile2D*)hp2,(TProfile2D*)hp2,(float)1.,(float)1.);
      cout << " collated TProfile2D: " << dir->getPathname() << "/" << nm <<endl;
      return;
    }

  MonitorElementRootProf2D * put_here = dynamic_cast<MonitorElementRootProf2D *>(me);
  if (put_here) put_here->copyFrom(hp);
}

// extract TH3F object from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractTH3F
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TH3F * h3 = dynamic_cast<TH3F*>(to); string nm = h3->GetName();
  MonitorElement * me = dir->findObject(nm);
  if(!wantME(me, dir, nm, fromRemoteNode))return;
  h3->SetName("extracted");

  if(!me)
    {
      me = book3D ( nm, h3, dir);
      /*
      me = book3D(nm, h3->GetTitle(), h3->GetNbinsX(), 
		  h3->GetXaxis()->GetXmin(), 
		  h3->GetXaxis()->GetXmax(), h3->GetNbinsY(), 
		  h3->GetYaxis()->GetXmin(), 
		  h3->GetYaxis()->GetXmax(), h3->GetNbinsZ(), 
		  h3->GetZaxis()->GetXmin(), 
		  h3->GetZaxis()->GetXmax(), dir);
	
      // set alphanumeric labels, if needed
      if(h3->GetXaxis()->GetLabels())
	{
	  for(int i = 1; i <= h3->GetNbinsX(); ++i)
	    me->setBinLabel(i,h3->GetXaxis()->GetBinLabel(i),1);
	  MonitorElementRootH3 * local = 
	    dynamic_cast<MonitorElementRootH3 *> (me);
	  ((TH3F*) local->operator->())->GetXaxis()->LabelsOption("v");
	}
      if(h3->GetYaxis()->GetLabels())
	{
	  for(int i = 1; i <= h3->GetNbinsY(); ++i)
	    me->setBinLabel(i,h3->GetYaxis()->GetBinLabel(i),2);
	}
      if(h3->GetZaxis()->GetLabels())
	{
	  for(int i = 1; i <= h3->GetNbinsZ(); ++i)
	    me->setBinLabel(i,h3->GetZaxis()->GetBinLabel(i),3);
	}
      // set axis titles
      me->setAxisTitle(h3->GetXaxis()->GetTitle(), 1);
      me->setAxisTitle(h3->GetYaxis()->GetTitle(), 2);
      me->setAxisTitle(h3->GetZaxis()->GetTitle(), 3);
      */
      // set canDelete flag if ME arrived from remote node
      if(fromRemoteNode)
	// cannot delete objects arrived from different node
	dir->canDeleteFromMenu[nm] = false;
      
      makeTagCopies(me);
    }
    else if (isCollateME(me)) 
    {
        MonitorElementRootH3* local = dynamic_cast<MonitorElementRootH3 *> (me);
        ((TH3F*) local->operator->())->Add(h3);
        cout << " collated TH3F: " << dir->getPathname() << "/" << nm <<endl;
        return;
    }  

  MonitorElementRootH3 * put_here = dynamic_cast<MonitorElementRootH3 *> (me);
  if (put_here) put_here->copyFrom(h3);
}

// extract object (either TObjString or TNamed depending on flag);
// return success
bool DaqMonitorROOTBackEnd::extractObject(TObject * to, bool fromRemoteNode,
					  string & name, string & value)
{
  if(fromRemoteNode) {
      TObjString * tn = dynamic_cast<TObjString *> (to); 
      return unpack(tn, name, value);
  }
  else  {
      TNamed * tn = dynamic_cast<TNamed *> (to);
      return unpack(tn, name, value);
  }
}

// extract integer from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractInt
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  string name; string value;
  bool success = extractObject(to, fromRemoteNode, name, value);
  if(success) {
      MonitorElement * me = dir->findObject(name);
      if(!wantME(me, dir, name, fromRemoteNode))return;
      if(!me) {
	  me = bookInt(name, dir);
	  // set canDelete flag if ME arrived from remote node
	  if(fromRemoteNode)
	    // cannot delete objects arrived from different node
	    dir->canDeleteFromMenu[name] = false;

	  makeTagCopies(me);
      }
      
      MonitorElementRootInt * put_here = 
	dynamic_cast<MonitorElementRootInt *> (me);
      if(put_here) put_here->Fill(atoi(value.c_str()));
      else cerr << " *** Failed to update integer " << name << endl;
  }
}

// extract float from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorROOTBackEnd::extractFloat
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  string name; string value;
  bool success = extractObject(to, fromRemoteNode, name, value);
  if(success)
    {
      MonitorElement * me = dir->findObject(name);
      if(!wantME(me, dir, name, fromRemoteNode))return;
      if(!me)
	{
	  me = bookFloat(name, dir);
	  // set canDelete flag if ME arrived from remote node
	  if(fromRemoteNode)
	    // cannot delete objects arrived from different node
	    dir->canDeleteFromMenu[name] = false;

	  makeTagCopies(me);
	}
      
      MonitorElementRootFloat * put_here = 
	dynamic_cast<MonitorElementRootFloat *> (me);
      if(put_here)
	put_here->Fill(atof(value.c_str()));
      else
	cerr << " *** Failed to update float " << name << endl;
    } 
}

// extract string from <to> into <me> in <dir>; 
// Do not know how to handle this if me != 0
void DaqMonitorROOTBackEnd::extractString
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  string name; string value;
  bool success = extractObject(to, fromRemoteNode, name, value);
  if(success)
    {
      MonitorElement * me = dir->findObject(name);
      if(!wantME(me, dir, name, fromRemoteNode))return;

      if(me)
	{ // string already exists; do we need it?
	  if(fromRemoteNode)
	    {
	      // do we want to support a "Fill(string s)" method for strings?
	      cerr << " *** Do not know how to update string objects! "
		   << endl;
	      cerr << " String " << name 
		   << " in directory " << dir->getPathname()
		   << " already defined... " << endl;
	      return;
	    }
	  else
	    // remove string, so we can replace it w/ new value
	    removeElement(dir, name);
	}
	  
      // book string here
      me = bookString(name, value, dir);
      // set canDelete flag if ME arrived from remote node
      if(fromRemoteNode)
	// cannot delete objects arrived from different node
	dir->canDeleteFromMenu[name] = false;
      makeTagCopies(me);

    } // unpack
}

// extract QReport
void DaqMonitorROOTBackEnd::extractQReport
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  string name; string value;
  bool success = extractObject(to, fromRemoteNode, name, value);
  if(!success)
    return;

  string ME_name; string qtest_name; int status; string message;
  bool all_ok =  
    StringUtil::unpackQReport(name, value, ME_name, qtest_name, 
			      status, message);

  if(!all_ok)
    return;

  try{

    MonitorElement * me = dir->findObject(ME_name);
    if(!me)
      throw me;
    
    QReport * qr = getQReport(me, qtest_name);
    if(!qr)
      qr = addQReport(me, qtest_name);

    // normally, we need not check (again) for a null qr here;
    // keep it for now...
    if(!qr)
      throw qr;    

    qr->setStatus(status);
    qr->setMessage(message);
    qr->updateReport(); // calling derived class method

    return; // if here, everything is good
  }

  catch (MonitorElement * me)
    {
      cerr << " *** MonitorElement " << ME_name << " for quality test " 
	   << qtest_name << " does not exist! " << endl;
    }
  catch (QReport * qr)
    {
      cerr << " *** Unexpected null QReport for quality test " << qtest_name
	   << endl;
    }

  cerr << " QReport unpacking has failed... " << endl;

}

/// true if ME should be extracted from object;
/// for remoteNode: true if replacing old ME or ME is desired
/// for local ME: true if ME does not exist or we can overwrite
bool DaqMonitorROOTBackEnd::wantME
(MonitorElement * me, MonitorElementRootFolder * dir, 
 const string & nm, bool fromRemoteNode) const
{
  if(fromRemoteNode)
    // if object arrived from upstream node, copy if
    // me != 0 (ie. already there) or if it is desired
    return me || isDesired(dir, nm, true); // show warning
  else
    // if object is read from ROOT file, copy if 
    // me = 0 (does not exist), or overwrite = true;
    return (!me) || overwriteFromFile;
}

// extract object (TH1F, TH2F, ...) from <to>; return success flag;
// flag fromRemoteNode indicating if ME arrived from different node

bool DaqMonitorROOTBackEnd::extractObject
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  string nm = to->GetName();
  
  if(isTH1F(to)) extractTH1F(to, dir, fromRemoteNode);
  else if(isTH2F(to)) extractTH2F(to, dir, fromRemoteNode);
  else if(isTProf2D(to)) extractTProf2D(to, dir, fromRemoteNode);
  else if(isTProf(to)) extractTProf(to, dir, fromRemoteNode);
  else if(isTH3F(to)) extractTH3F(to, dir, fromRemoteNode);
  else if(isInt(to)) extractInt(to, dir, fromRemoteNode);
  else if(isFloat(to)) extractFloat(to, dir, fromRemoteNode);
  else if(isString(to)) extractString(to, dir, fromRemoteNode);
  else if(isQReport(to)) extractQReport(to, dir, fromRemoteNode);
  else if (nm.find("CMSSW")==1) cout << " ME input file version: " << nm << endl;
  else if (nm.find("DQMPATCH")==0) cout << " DQM patch version: " << nm << endl;
  else {
     cout << " *** Failed to extract object " << nm
          << " of type " << to->IsA()->GetName() 
          << " (title: " << to->GetTitle() << ") " << endl;
     return false;
  }
  return true;

}

// unpack TNamed into name <nm> and value <value>; return success
// for reading from files: TNamed *getRootObject
bool DaqMonitorROOTBackEnd::unpack(TNamed * tn, string & nm, string & value)
  const
{
  // expect TNamed title in the format: X=value
  // where X = i, f, s, qr
  if(tn)
    {
      nm = tn->GetName();
      string title = tn->GetTitle();
      unsigned _begin = title.find("=");
      if(_begin != string::npos)
	value = title.substr(_begin+1, title.size()-_begin-1);
      else
	{
	  cerr << " *** Value extraction for object " << nm 
	       << " from title " << title << " has failed" << endl;
	  return false;
	}
    }
  else // dynamic_cast failed
    {
      cerr << " *** Dynamic_cast for TNamed has failed! " << endl;
      return false;
    }

  // if here, everything is ok
  return true;


}
// unpack TObjString into name <nm> and value <value>; return success;
// (for remote nodes: TObjString * getTagObject)
bool DaqMonitorROOTBackEnd::unpack(TObjString * tn, string & nm, string & value)
  const
{

  // expect string in the format: <name>X=value</name>
  // where X = i, f, s, qr
  if(tn)
    {
      string orig_name = tn->GetName();
      unsigned _begin = orig_name.find("<");
      unsigned _end = orig_name.find(">");
      if(_begin != string::npos && _end != string::npos)
	nm = orig_name.substr(_begin+1, _end-_begin-1);
      else
	{
	  cerr << " *** Unpacking of TObjString = " << tn->GetName()
	       << " has failed " << endl;
	  return false;
	}
      
      _begin = orig_name.find("=");
      _end = orig_name.rfind("<");
      if(_begin != string::npos && _end != string::npos)
	value = orig_name.substr(_begin+1, _end-_begin-1);
      else
	{
	  cerr << " *** Value extraction for object " << nm << " has failed"
	       << endl;
	  return false;
	}
    }
  else // dynamic_cast failed
    {
      cerr << " *** Dynamic_cast for TObjString has failed! " << endl;
      return false;
    }

  // if here, everything is ok
  return true;

}

// true if Monitoring Element <me> in directory <folder> has isDesired = true;
// if warning = true and <me> does not exist, show warning
bool DaqMonitorROOTBackEnd::isDesired(MonitorElementRootFolder * folder, 
				      string me, bool warning) const
{
  if(warning && !folder->hasMonitorable(me))
    {
      cerr << " *** Object " << me << " does not exist in " 
	   << folder->getPathname() << endl;
      cerr << " Error in isDesired method... " << endl;
      return false;
    }

  // if no entry, assume ME is not desired
  if(folder->isDesired.find(me) == folder->isDesired.end())
    return false;

  // if here, entry exists
  return folder->isDesired[me];

}

