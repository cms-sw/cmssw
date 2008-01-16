#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/ParameterSet/interface/types.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/DQMPatchVersion.h"

#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Core/interface/Tokenizer.h"

#include "DQMServices/Core/interface/DQMTagHelper.h"
#include "DQMServices/Core/interface/QCriterionRoot.h"

#include <iostream>

#include <TKey.h>
#include <TROOT.h>

using namespace dqm::me_util;
using namespace dqm::qtests;

using std::cout; using std::endl; using std::cerr;
using std::vector; using std::string; using std::set;

DaqMonitorBEInterface * DaqMonitorBEInterface::theinstance = 0;

DaqMonitorBEInterface * DaqMonitorBEInterface::instance()
{
  if(theinstance==0)
    {
      try
	{
	  edm::ParameterSet pset;
	  theinstance = new DaqMonitorBEInterface(pset);
	}
      catch(cms::Exception e)
	{
	  cout << e.what() << endl;
	  exit (-1);
	}
      
    }
  return (DaqMonitorBEInterface *) theinstance;
}

DaqMonitorBEInterface::DaqMonitorBEInterface(edm::ParameterSet const &pset){

  theinstance = this;

  dqm_locker = 0;
  DQM_VERBOSE = 1; resetMonitoringDiff(); resetMonitorableDiff();

  // set steerable parameters
  DQM_VERBOSE = pset.getUntrackedParameter<int>("verbose",1);
  cout << " DaqMonitorBEInterface: verbose parameter set to " << 
            DQM_VERBOSE << endl;
  
  string subsystemname = 
                pset.getUntrackedParameter<string>("subSystemName","");
 
  string referencefilename = 
                pset.getUntrackedParameter<string>("referenceFileName","");
  cout << " DaqMonitorBEInterface: reference file name set to " <<
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



// clone/copy received TH1F object from <source> to ME in <dir>
MonitorElement * 
DaqMonitorBEInterface::book1D(std::string name, TH1F* source,
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
DaqMonitorBEInterface::book1D(string name, string title, int nchX, 
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
DaqMonitorBEInterface::book1D(string name, string title, int nchX,
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
DaqMonitorBEInterface::book2D(std::string name, TH2F* source,
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
DaqMonitorBEInterface::book2D(string name, string title, int nchX, 
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
DaqMonitorBEInterface::book3D(std::string name, TH3F* source,
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
DaqMonitorBEInterface::book3D(string name, string title, int nchX, 
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
DaqMonitorBEInterface::bookProfile(std::string name, TProfile* source,
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
DaqMonitorBEInterface::bookProfile(string name, string title, 
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
DaqMonitorBEInterface::bookProfile2D(std::string name, TProfile2D* source,
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
DaqMonitorBEInterface::bookProfile2D(string name, string title, 
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

MonitorElement * DaqMonitorBEInterface::bookFloat(string name, 
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

MonitorElement * DaqMonitorBEInterface::bookInt(string name, 
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
DaqMonitorBEInterface::bookString(string name, string value,
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

// extract TH1F object from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorBEInterface::extractTH1F
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TH1F *h1 = dynamic_cast<TH1F*>(to); string nm = h1->GetName();
  MonitorElement * me = dir->findObject(nm);
  if (!wantME(me, dir, nm, fromRemoteNode)) return;
  h1->SetName("extracted");

  if(!me)
    {
      me = book1D ( nm, h1, dir);
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
void DaqMonitorBEInterface::extractTH2F
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
void DaqMonitorBEInterface::extractTProf
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
void DaqMonitorBEInterface::extractTProf2D
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TProfile2D *hp = dynamic_cast<TProfile2D*>(to); string nm = hp->GetName();
  MonitorElement * me = dir->findObject(nm);
  if(!wantME(me, dir, nm, fromRemoteNode))return;
  hp->SetName("extracted");

  if(!me)
    {
      me = bookProfile2D ( nm, hp, dir);
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
void DaqMonitorBEInterface::extractTH3F
(TObject * to, MonitorElementRootFolder * dir, bool fromRemoteNode)
{
  TH3F * h3 = dynamic_cast<TH3F*>(to); string nm = h3->GetName();
  MonitorElement * me = dir->findObject(nm);
  if(!wantME(me, dir, nm, fromRemoteNode))return;
  h3->SetName("extracted");

  if(!me)
    {
      me = book3D ( nm, h3, dir);
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
bool DaqMonitorBEInterface::extractObject(TObject * to, bool fromRemoteNode,
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
// extract object (TH1F, TH2F, ...) from <to>; return success flag;
// flag fromRemoteNode indicating if ME arrived from different node

bool DaqMonitorBEInterface::extractObject
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

// extract integer from <to> into <me> in <dir>; 
// if me != 0, will overwrite object
void DaqMonitorBEInterface::extractInt
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
void DaqMonitorBEInterface::extractFloat
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
void DaqMonitorBEInterface::extractString
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
void DaqMonitorBEInterface::extractQReport
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
bool DaqMonitorBEInterface::wantME
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

// true if Monitoring Element <me> in directory <folder> has isDesired = true;
// if warning = true and <me> does not exist, show warning
bool DaqMonitorBEInterface::isDesired(MonitorElementRootFolder * folder, 
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

// true if Monitoring Element <me> is needed by any subscriber
bool DaqMonitorBEInterface::isNeeded(string pathname, string me) const
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

// check against null objects
bool DaqMonitorBEInterface::checkElement(const MonitorElement * const me) const
{
  if(!me)
    {
      cerr << " *** Error! Null monitoring element " << endl;
      return false;
    }
  return true;
}

// remove monitoring element from directory; 
// if warning = true, print message if element does not exist
void DaqMonitorBEInterface::removeElement(MonitorElementRootFolder * dir, 
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
void DaqMonitorBEInterface::removeContents(MonitorElementRootFolder * dir)
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

// remove all references for directories starting w/ pathname;
// put contents of directories in removeContents (if applicable)
void DaqMonitorBEInterface::removeReferences(string pathname, rootDir & rDir)
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

// get added contents (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getAddedContents(vector<string> & put_here) const
{
  convert(put_here, addedContents);
}

// get removed contents (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getRemovedContents(vector<string> & put_here) const
{
  convert(put_here, removedContents);
}

// get updated contents (since last cycle)
// COMPLEMENTARY to addedContents, removedContents
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getUpdatedContents(vector<string> & put_here) const
{
  convert(put_here, updatedContents);
}

// convert dqm::me_util::monit_map into 
// vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
// to be invoked by getAddedContents, getRemovedContents, getUpdatedContents
void 
DaqMonitorBEInterface::convert(vector<string> & put_here, const monit_map & in) 
  const
{
  put_here.clear();
  for(cmonit_it path = in.begin(); path != in.end(); ++path)
    { // loop over all pathnames/directories
      string new_entry;
      const set<string> & input = path->second;
      
      for(csIt it = input.begin(); it != input.end(); ++it)
	{ // loop over all monitoring elements names

	  if(it != input.begin())
	    new_entry += ",";
	  new_entry += *it;
	} // loop over all monitoring elements names

      if(!new_entry.empty())
	{ // directory contains at least one monitoring element
	  new_entry = path->first + ":" + new_entry;
	  put_here.push_back(new_entry);
	}

    } // loop over all pathnames/directories
}

// to be called by getContents (flag = false) or getMonitorable (flag = true)
// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>
// if showContents = false, change form to <dir pathname>:
// (useful for subscription requests; meant to imply "all contents")
void DaqMonitorBEInterface::get(std::vector<string> & put_here, bool monit_flag,
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

// get ME from full pathname (e.g. "my/long/dir/my_histo")
MonitorElement * DaqMonitorBEInterface::get(string fullpath) const
{
  string path, filename;
  StringUtil::unpack(fullpath, path, filename); 	 
  MonitorElementRootFolder * dir = getDirectory(path, Own); 
  if(!dir) return (MonitorElement *)0; 	 
  return dir->findObject(filename);
}

MonitorElement * DaqMonitorBEInterface::getReferenceME(MonitorElement* me) const
{
  if (me) return findObject(me->getName(),referenceDirName+"/"+me->getPathname());
  else {
    cerr << " MonitorElement " << me->getPathname() << "/" << 
              me->getName() << " does not exist! " << endl;
    return (MonitorElement*) 0 ;
  }
}

bool DaqMonitorBEInterface::isReferenceME(MonitorElement* me) const
{
  if (me && (me->getPathname().find(referenceDirName+"/")==0)) return true; 
  return false;
}

bool DaqMonitorBEInterface::isCollateME(MonitorElement* me) const
{
  if (me && (me->getPathname().find(collateDirName+"/")==0))  // check that collate histos themselves 
	                                                      // are not picked up for additional 
							      // collation
	    return true; 
  return false;
}

bool DaqMonitorBEInterface::makeReferenceME(MonitorElement* me)
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

void DaqMonitorBEInterface::deleteME(MonitorElement* me)
{
    string pathname = me->getPathname();
    MonitorElementRootFolder * dir = 
          DaqMonitorBEInterface::makeDirectory(pathname);
    removeElement(dir,me->getName());
}

// get all MonitorElements tagged as <tag>
vector<MonitorElement *> DaqMonitorBEInterface::get(unsigned int tag) const
{
  vector<MonitorElement *> ret;
  ctdir_it tg = Tags.find(tag);
  if(tg == Tags.end())
    return ret;
  
  get(tg->second.paths, ret);
  return ret;
}

// add all (tagged) MEs to put_here
void DaqMonitorBEInterface::get(const dir_map & Dir, 
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
vector<MonitorElement *> DaqMonitorBEInterface::getContents(string pathname) 
  const
{
  vector<MonitorElement *> ret;
  getContents(pathname, Own, ret);
  return ret;
}

// same as above for tagged MonitorElements
vector<MonitorElement *> 
DaqMonitorBEInterface::getContents(string pathname, unsigned int tag) const
{
  vector<MonitorElement *> ret;
  ctdir_it tg = Tags.find(tag);
  if(tg != Tags.end())
    getContents(pathname, tg->second, ret);

  return ret;
}

// get vector with all children of folder in <rDir>
// (does NOT include contents of subfolders)
void DaqMonitorBEInterface::getContents
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
vector<MonitorElement*> DaqMonitorBEInterface::getAllContents
(string pathname) const
{
  vector<MonitorElement *> ret;
  getAllContents(pathname, Own, ret);
  return ret;
}

// same as above for tagged MonitorElements
vector<MonitorElement*> DaqMonitorBEInterface::getAllContents
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
void DaqMonitorBEInterface::getAllContents
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

void DaqMonitorBEInterface::showDirStructure(void) const
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
bool DaqMonitorBEInterface::isRootFolder(void)
{
  return (((TFolder*) fCurrentFolder->operator->()) == 
	  gROOT->GetRootFolder());
}

// add monitoring element to current folder
void DaqMonitorBEInterface::addElement(MonitorElement * me, string pathname, 
				       string type)
{
  if(pathname == ROOT_PATHNAME && first_time_onRoot)
    {
      cout << " DaqMonitorBEInterface info: no directory has been " << 
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
void DaqMonitorBEInterface::getContents(std::vector<string> & put_here,
					bool showContents) const
{
  get(put_here, false, showContents);
}

// get monitorable;
// return vector<string> of the form <dir pathname>:<obj1>,<obj2>,<obj3>;
// if showContents = false, change form to <dir pathname>:
// (useful for subscription requests; meant to imply "all contents")
void DaqMonitorBEInterface::getMonitorable(std::vector<string> & put_here,
					   bool showContents) const
{
  get(put_here, true, showContents);
}



// get first non-null ME found starting in path; null if failure
MonitorElement * DaqMonitorBEInterface::getMEfromFolder(cdir_it & path) const
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
DaqMonitorBEInterface::getMEfromFolder(cdir_it & path, string obj_name) const
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
// get list of subdirectories of current directory
vector<string> DaqMonitorBEInterface::getSubdirs(void) const
{
  vector<string> ret; ret.clear();
  for(cdir_it it = fCurrentFolder->subfolds_.begin(); 
      it != fCurrentFolder->subfolds_.end(); ++it)
    ret.push_back(it->first);
  
  return ret;    
}

// get list of (non-dir) MEs of current directory
vector<string> DaqMonitorBEInterface::getMEs(void) const
{
  vector<string> ret; ret.clear();
  for(cME_it it = fCurrentFolder->objects_.begin(); 
      it != fCurrentFolder->objects_.end(); ++it)
    ret.push_back(it->first);
  
  return ret;    
}

// get added monitorable (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getAddedMonitorable(vector<string> & put_here) const
{
  put_here = addedMonitorable;
}

// get removed monitorable (since last cycle)
// return vector<string> of the form: <dir pathname>:<obj1>,<obj2>,...
void DaqMonitorBEInterface::getRemovedMonitorable(vector<string> & put_here) const
{
  put_here = removedMonitorable;
}

// reset modifications to monitorable since last cycle 
// and sets of added/removed contents
void DaqMonitorBEInterface::resetMonitorableDiff()
{
  // reset added, removed monitorable
  addedMonitorable.clear();
  removedMonitorable.clear();
  // reset added, removed contents;
  addedContents.clear();
  removedContents.clear();
  // reset modified tags
  addedTags.clear();
  removedTags.clear();

  rMonitorableDiffWasCalled = true;
}

// reset updated contents and updated QReports
void DaqMonitorBEInterface::resetMonitoringDiff()
{
  // reset updated contents
  updatedContents.clear();
  // reset updated QReports
  updatedQReports.clear();

  rMonitoringDiffWasCalled = true;
}

/* come here after sending monitorable to all receivers;
   if callResetDiff = true, call resetMonitorableDiff
   (typical behaviour: Sources & Collector have callResetDiff = true, whereas
   clients have callResetDiff = false, so GUI/WebInterface can access the 
   modifications in monitorable & monitoring) */
void 
DaqMonitorBEInterface::doneSendingMonitorable(bool callResetDiff)
{
  // if flag=true, reset list of modified monitoring
  if(callResetDiff)resetMonitorableDiff();
}

/* come here after sending monitoring to all receivers;
   (a) call resetUpdate for modified contents:
   
   if resetMEs=true, reset MEs that were updated (and have resetMe = true);
   [flag resetMe is typically set by sources (false by default)];
   [Clients in standalone mode should also have resetMEs = true] 
   
   (b) if callResetDiff = true, call resetMonitoringDiff
   (typical behaviour: Sources & Collector have callResetDiff = true, whereas
   clients have callResetDiff = false, so GUI/WebInterface can access the 
   modifications in monitorable & monitoring) */
void 
DaqMonitorBEInterface::doneSendingMonitoring(bool resetMEs, bool callResetDiff)
{

  // reset "update" flag for monitoring objects that have been updated/added
  for(monit_it path = updatedContents.begin(); 
      path != updatedContents.end(); ++path)
    { // loop over all pathnames/directories
      
      string pathname = path->first;
      
      for(sIt it = path->second.begin(); it != path->second.end(); ++it)
	{ // loop over all ME names
	  
	  MonitorElement * me = findObject(*it, pathname);
	  if(me)
	    {
	      // if reset, reset (ie. clear contents of) monitoring element
	      if(resetMEs && me->resetMe())me->Reset();
	      me->resetUpdate();
	    }
	  
	} // loop over all ME names
      
    }  // loop over all pathnames/directories
  
  // reset "update" flag for QReports that have been updated/added
  for(set<QReport *>::iterator it = updatedQReports.begin(); 
      it != updatedQReports.end(); ++it)
    {      
      if(*it)
	(*it)->resetUpdate();
    }
  
  // if flag=true, reset list of modified monitoring
  if(callResetDiff)resetMonitoringDiff();

}

// get "global" folder <inpath> status (one of: STATUS_OK, WARNING, ERROR, OTHER);
// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
// see Core/interface/QTestStatus.h for details on "OTHER" 
int DaqMonitorBEInterface::getStatus(std::string inpath) const
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
int DaqMonitorBEInterface::getStatus(unsigned int tag) const
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

// reset ME contents 
void DaqMonitorBEInterface::softReset(MonitorElement * me)
{
  if(!checkElement(me))
    return;

  me->softReset();
}

// reverts action of softReset
void DaqMonitorBEInterface::disableSoftReset(MonitorElement * me)
{
  if(!checkElement(me))
    return;

  me->disableSoftReset();
}

// if true, will accumulate ME contents (over many periods)
// until method is called with flag = false again
void DaqMonitorBEInterface::setAccumulate(MonitorElement * me, bool flag)
{
  if(!checkElement(me))
    return;

  me->setAccumulate(flag);
}

// update directory structure maps for folder
void DaqMonitorBEInterface::updateMaps(MonitorElementRootFolder * dir, 
				       rootDir & rDir)
{
  rDir.paths[dir->getPathname()] = dir;
}
// add <name> to back-end interface's updatedContents
void DaqMonitorBEInterface::add2UpdatedContents(string name, string pathname)
{
  monit_it It = updatedContents.find(pathname);
  if(It == updatedContents.end())
    {
      set<string> temp; temp.insert(name);
      updatedContents[pathname] = temp;
    }
  else
    It->second.insert(name);
}

// run quality tests (also finds updated contents in last monitoring cycle,
// including newly added content) 
void DaqMonitorBEInterface::runQTests(void)
{
  if(!wasResetCalled())
    {
      cout << " *** Warning! Need to call MonitorUserInterface::doMonitoring\n"
	   << " before calling MonitorUserInterface::runQTests again! " << endl;
    }

  // keep track here of modified algorithm since last time runQTests ran
  vector<QCriterion *> modifiedAlgos;
  for(qc_it qc = qtests_.begin(); qc != qtests_.end(); ++qc)
    { // loop over quality tests
      if( (qc->second)->wasModified() )
	modifiedAlgos.push_back(qc->second);

    } // loop over quality tests

  try
    {      
      // first, check if qtests_ should be attached to any of the added elements
      if(!addedContents.empty())
	checkAddedElements();

      // now run the quality tests for real!
      runQualityTests();
      
    } // try-block

  catch(ME_map * m)
    {
      nullError("ME_map");
    }
  catch(MonitorElement * m)
    {
      nullError("MonitorElement");
    }
  catch (...)
    {
      cerr << " *** Unknown error returned by DaqMonitorBEInterface::runQTests " 
	   << endl;
    }

  // reset "wasModified" flag for quality-test algorithms
  for(vqc_it it = modifiedAlgos.begin(); it != modifiedAlgos.end(); ++it)
    (*it)->wasModified_ = false;

  rMonitoringDiffWasCalled = rMonitorableDiffWasCalled = false;
}

// loop over quality tests & addedContents: look for MEs that 
// match QCriterion::rules; upon a match, add QReport to ME(s)
void DaqMonitorBEInterface::checkAddedElements(void)
{
  for(cqc_it qc = qtests_.begin(); qc != qtests_.end(); ++qc)
    { // loop over quality tests
      vME allMEs;
      checkAddedContents(qc->second->rules, allMEs);
      addQReport(allMEs, qc->second);
    } // loop over quality tests
}

// get QCriterion corresponding to <qtname> 
// (null pointer if QCriterion does not exist)
QCriterion * DaqMonitorBEInterface::getQCriterion(string qtname) const
{
  cqc_it it = qtests_.find(qtname);
  if(it == qtests_.end())
    return (QCriterion *) 0;
  else
    return it->second;
}

// get QReport from ME (null pointer if no such QReport)
QReport * 
DaqMonitorBEInterface::getQReport(MonitorElement * me, string qtname)
{
  QReport * ret = 0;
  if(me)
    {
      qr_it it = me->qreports_.find(qtname);
      if(it != me->qreports_.end())
	ret = it->second;
    }

  return ret;
}

// add quality report (to be called when test is to run locally)
QReport * DaqMonitorBEInterface::addQReport(MonitorElement * me, QCriterion * qc)
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
QReport * DaqMonitorBEInterface::addQReport(MonitorElement * me, string qtname, 
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



// scan structure <rDir>, looking for all MEs matching <search_string>;
// put results in <put_here>
void DaqMonitorBEInterface::scanContents
(const string & search_string, const rootDir & rDir,  
 vector<MonitorElement *> & put_here) const
{
  put_here.clear();
  
  if(!hasWildCards(search_string))
    {
      MonitorElement * me = get(search_string);
      if(me)
	put_here.push_back(me);
    }
  else
    {

      cdir_it start, end, parent_dir;
      getSubRange<dir_map>(search_string, rDir.paths,start,end,parent_dir);
      
      // do parent directory first
      if(parent_dir != Own.paths.end())
	scanContents(search_string, parent_dir->second, put_here);
      
      for(cdir_it path = start; path != end; ++path)
	// loop over pathnames in directory structure
	scanContents(search_string, path->second, put_here);
    }
}

// same as scanContents in base class but for one path only
void DaqMonitorBEInterface::scanContents
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

DaqMonitorBEInterface::~DaqMonitorBEInterface(void)
{
  // remove all quality tests
  for(qc_it it = qtests_.begin(); it != qtests_.end(); ++it)
    {
      if(it->second)
	delete it->second;
    }
  qtests_.clear();

  if(dqm_locker)delete dqm_locker;

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

  theinstance = 0;
  delete tagHelper;
}


// acquire lock
void DaqMonitorBEInterface::lock()
{
  // cout << " Called lock " << endl;
  // mutex is not released till previous lock has been deleted in unlock()
  dqm_locker = 
    new boost::mutex::scoped_lock(*edm::rootfix::getGlobalMutex());
}

// release lock
void DaqMonitorBEInterface::unlock()
{
  //  cout << " Called unlock " << endl;
  if(dqm_locker)
    {
      // use local stack pointer to release memory, so we do not set
      // dqm_locker to zero AFTER lock has been released
      // (dangerous, as another thread may have acquired lock in the meantime)
      boost::mutex::scoped_lock * tmp_lock = dqm_locker;
      dqm_locker = 0;
      delete tmp_lock;
    }
  //  
}

// get rootDir corresponding to tag 
// (Own for tag=0, or null for non-existing tag)
const rootDir * DaqMonitorBEInterface::getRootDir(unsigned int tag) const
{
  const rootDir * ret = 0;
  if(tag)
    {
      ctdir_it tg = Tags.find(tag);
      if(tg != Tags.end())
	ret = &(tg->second);
    }
  else // this corresponds to Own
    ret = &Own;

  return ret;
}

// check if added contents match rules; put matches in put_here
void DaqMonitorBEInterface::checkAddedContents
(const searchCriteria & rules, vector<MonitorElement *> & put_here) const
{
  for(csMapIt sc = rules.search.begin(); sc != rules.search.end(); ++sc)
    {
      const rootDir * Dir = getRootDir(sc->first);
      if(!Dir)continue;
      
      checkAddedSearchPaths(sc->second.search_path, *Dir, put_here);
      checkAddedFolders(sc->second.folders, *Dir, false, put_here);
      checkAddedFolders(sc->second.foldersFull, *Dir, true, put_here);
    }
  
  
  for(vector<unsigned int>::const_iterator tg = rules.tags.begin(); 
      tg != rules.tags.end(); ++tg)
    {
      const rootDir * Dir = getRootDir(*tg);
      if(!Dir)continue;
      checkAddedTags(*Dir, put_here);
    }
}

// same as in base class for given search_string and path; put matches into put_here
void DaqMonitorBEInterface::checkAddedContents
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


// check if added contents belong to folders 
// (use flag to specify if subfolders should be included)
void DaqMonitorBEInterface::checkAddedFolders
(const vector<string> & folders, const rootDir & Dir, bool useSubfolders,
 vector<MonitorElement*>& put_here) const
{
  for(cvIt f = folders.begin(); f != folders.end(); ++f)
    { // loop over folders to be watched
      
      if(useSubfolders)
	{ // will consider all subdirectories of *f
	  
	  for(cmonit_it added_path = addedContents.begin(); added_path != 
		addedContents.end(); ++added_path)
	    {
	      if(isSubdirectory(*f, added_path->first))
		checkAddedFolder(added_path, Dir, put_here);
	    }
	  
	}
      else
	{ // will only consider directory *f
	  cmonit_it added_path = addedContents.find(*f);
	  if(added_path != addedContents.end())
	    checkAddedFolder(added_path, Dir, put_here);
	}
      
    } // loop over folders to be watched
}

// check if added contents belong to folder 
// (use flag to specify if subfolders should be included)
void DaqMonitorBEInterface::checkAddedFolder
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


// check if added contents are tagged
void DaqMonitorBEInterface::checkAddedTags
(const rootDir & Dir, vector<MonitorElement*>& put_here) const
{
  for(cmonit_it added_path = addedContents.begin(); 
      added_path != addedContents.end(); ++added_path)
    checkAddedFolder(added_path, Dir, put_here);
}

// remove all contents from <pathname> from all subscribers and tags
void DaqMonitorBEInterface::removeCopies(const string & pathname)
{
  // we will loop over Subscribers and Tags
  // and remove contents from all directories <pathname>
  for(sdir_it subs = Subscribers.begin(); subs!= Subscribers.end(); ++subs)
    { // loop over all subscribers
       MonitorElementRootFolder * dir = getDirectory(pathname, subs->second);
      // skip subscriber if no such pathname
      if(!dir)continue;
       removeContents(dir);
    } // loop over all subscribers
    

  for(tdir_it tag = Tags.begin(); tag != Tags.end(); ++tag)
    { // loop over all tags
      MonitorElementRootFolder * dir = getDirectory(pathname, tag->second);
      // skip tag if no such pathname
      if(!dir)continue;
      removeContents(dir);
   } // loop over all tags

}

// check if added contents match search paths
void DaqMonitorBEInterface::checkAddedSearchPaths
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

// remove Monitor Element <name> from all subscribers, tags and CME directories
void DaqMonitorBEInterface::removeCopies(const string & pathname, 
					 const string & name)
{
  // we will loop over Subscribers and Tags
  // and remove <name> from all directories <pathname>
  
  for(sdir_it subs= Subscribers.begin(); subs != Subscribers.end(); ++subs)
    // loop over all subscribers
    remove(pathname, name, subs->second);

  for(tdir_it tag = Tags.begin(); tag != Tags.end(); ++tag)
    // loop over all tags
    remove(pathname, name, tag->second);

}

// remove Monitor Element <name> from <pathname> in <Dir>
void DaqMonitorBEInterface::remove(const string & pathname, 
				   const string & name, rootDir & Dir)
{
  MonitorElementRootFolder * dir = getDirectory(pathname, Dir);
  // skip subscriber if no such pathname
  if(!dir)return;
  
  removeElement(dir, name, false); // no warnings
}

// attach quality test <qc> to all ME matching <search_string>;
// if tag != 0, this applies to tagged contents
// <search_string> could : (a) be exact pathname (e.g. A/B/C/histo): FAST
// (b) include wildcards (e.g. A/?/C/histo, A/B/*/histo or A/B/*): SLOW
void DaqMonitorBEInterface::useQTest
(unsigned int tag, string search_string, const rootDir & Dir, QCriterion * qc) 
  const
{
  assert(qc);
  qc->add2search_path(search_string, tag);
  vME allMEs;
  scanContents(search_string, Dir, allMEs);
  addQReport(allMEs, qc); 
}

// attach quality test <qc> to directory contents ==> FAST
// if tag != 0, this applies to tagged contents
// (need exact pathname without wildcards, e.g. A/B/C);
// use flag to specify whether subfolders (and their contents) should be included;
void DaqMonitorBEInterface::useQTest(unsigned int tag, string pathname, 
				     bool useSubfolds, const rootDir & Dir, 
				     QCriterion * qc) const
{
  assert(qc);
  qc->add2folders(pathname, useSubfolds, tag);
  vME allMEs;
  if(useSubfolds)
    getAllContents(pathname, Dir, allMEs);
  else
    getContents(pathname, Dir, allMEs);
  addQReport(allMEs, qc);
}

// attach quality test <qtname> to tagged MEs ==> FAST
void DaqMonitorBEInterface::useQTest(unsigned int tag, const rootDir & Dir,
				     QCriterion * qc) const
{
  assert(qc);
  qc->add2tags(tag);
  vME allMEs;
  get(Dir.paths, allMEs);
  addQReport(allMEs, qc);
}


void DaqMonitorBEInterface::useQTest(string search_string, string qtname) const
{
   useQTest(0, search_string, qtname); // "0" means no tag
}   

void DaqMonitorBEInterface::useQTest(unsigned int tag, string search_string,
				    string qtname) const
{
  if(search_string.empty())
    return;

  QCriterion * qc = getQCriterion(qtname);
  if(!qc)
    {
      cerr << " *** Quality test " << qtname << " does not exist! " << endl;
      return;
    }

  if(tag == 0) // "0" means no tag
    useQTest(0, search_string, this->Own, qc);
  else
    {
      ctdir_it tg = Tags.find(tag);
      if(tg != Tags.end())
	useQTest(tag, search_string, tg->second, qc);
      else
	qc->add2search_path(search_string, tag);
    }
   
}

// attach quality test <qtname> to tagged MEs ==> FAST
// this action applies to all MEs already available or future ones
void DaqMonitorBEInterface::useQTest(unsigned int tag, string qtname) const
{
  QCriterion * qc = getQCriterion(qtname);
  if(!qc)
    {
      cerr << " *** Quality test " << qtname << " does not exist! " << endl;
      return;
    }
  if(tag == 0)
    {
      cerr << " *** Tag must be positive number! \n";
      return;
    }
  
  ctdir_it tg = Tags.find(tag);
  if(tg != Tags.end())
    useQTest(tag, tg->second, qc);
  else
    qc->add2tags(tag); 
}

// add quality reports to all MEs
void DaqMonitorBEInterface::addQReport(vector<MonitorElement *> & allMEs, 
				       QCriterion * qc) const
{
  assert(qc);
  string qr_name = qc->getName();
  for(vMEIt me = allMEs.begin(); me != allMEs.end(); ++me)
    {
      /* I need to double-check that qreport is not already added to ME;
	 This is because there is a chance that users may
	 1. define a ME after resetMonitoringDiff has been called
	 2. call MonitorUserInterface::useQTest
	 3. and then call MonitorUserInterface::runQTests, which
	 eventually calls this function
	 In this case ME appears in addedContents and this call would
	 give an error... (not sure of a better way right now)
      */
      if(!(*me)->getQReport(qr_name))
	    addQReport(*me, qc);

    }
}

// create quality test with unique name <qtname> (analogous to ME name);
// quality test can then be attached to ME with useQTest method
// (<algo_name> must match one of known algorithms)
QCriterion * DaqMonitorBEInterface::createQTest(string algo_name, string qtname)
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

// run quality tests (also finds updated contents in last monitoring cycle,
// including newly added content) <-- to be called only by runQTests
void DaqMonitorBEInterface::runQualityTests(void)
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

// get "global" status (one of: STATUS_OK, WARNING, ERROR, OTHER) for group of MEs;
// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
// see Core/interface/QTestStatus.h for details on "OTHER" 
int DaqMonitorBEInterface::getStatus(const vector<MonitorElement *> & ME_group) 
  const
{
  if(hasError(ME_group))
    return dqm::qstatus::ERROR;
  else if(hasWarning(ME_group))
    return dqm::qstatus::WARNING;
  else if(hasOtherReport(ME_group))
    return dqm::qstatus::OTHER;
  else
    return dqm::qstatus::STATUS_OK;  
}

// true if at least one ME gave hasError = true
bool DaqMonitorBEInterface::hasError(const vector<MonitorElement *> & ME_group) 
  const
{
  for(vMEcIt me = ME_group.begin(); me != ME_group.end(); ++me)
    if(*me && (*me)->hasError())return true;
  // if here, no ME with an error has been found
  return false;

}

// true if at least one ME gave hasWarning = true
bool DaqMonitorBEInterface::hasWarning(const vector<MonitorElement *> & ME_group)
  const
{
  for(vMEcIt me = ME_group.begin(); me != ME_group.end(); ++me)
    if(*me && (*me)->hasWarning())return true;
  // if here, no ME with a warning has been found
  return false;
}

// true if at least one ME gave hasOtherReport = true
bool DaqMonitorBEInterface::hasOtherReport(const vector<MonitorElement *> & 
					   ME_group) const
{
  for(vMEcIt me = ME_group.begin(); me != ME_group.end(); ++me)
    if(*me && (*me)->hasOtherReport())return true;
  // if here, no ME with another (non-ok) status has been found
  return false;
}

// unpack TNamed into name <nm> and value <value>; return success
// for reading from files: TNamed *getRootObject
bool DaqMonitorBEInterface::unpack(TNamed * tn, string & nm, string & value)
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
bool DaqMonitorBEInterface::unpack(TObjString * tn, string & nm, string & value)
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

/// open/read root file <filename>, and copy MonitorElements;
/// if flag=true, overwrite identical MonitorElements (default: false);
/// if directory != "", read only selected directory
void DaqMonitorBEInterface::open(string filename, bool overwrite, 
				 string directory, string prepend)
{
  cout << " DaqMonitorBEInterface::open : opening ME input file " 
       << filename << " ... " << endl; 
  TFile f(filename.c_str());
  if(f.IsZombie())
    {
      cerr << " *** Error! Failed opening filename " << filename << endl;
      return;
    }

  readOnlyDirectory = directory;
  unsigned int N = readDirectory(&f, ROOT_PATHNAME, prepend);
  f.Close();
  if(DQM_VERBOSE) {
    cout << " Successfully read " << N << " monitoring objects from file " << filename ;
    if (prepend != "") cout << " and prepended \"" << prepend << "/\" to the path "
	 << endl; 
    } 
  cout << " DaqMonitorBEInterface::open : file " << filename << " closed " << endl;
}

// read ROOT objects from file <file> in directory <orig_pathname>;
// return total # of ROOT objects read
unsigned int 
DaqMonitorBEInterface::readDirectory(TFile* file, string orig_pathname, 
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

// equivalent to "cd .."
void DaqMonitorBEInterface::goUp(void)
{
  if(fCurrentFolder->parent_)
    setCurrentFolder(fCurrentFolder->parent_->pathname_);
}

// cd to subdirectory (if there)
void DaqMonitorBEInterface::cd(string subdir_path)
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
void DaqMonitorBEInterface::rmdir(const string & pathname, rootDir & rDir)
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
void DaqMonitorBEInterface::rmdir(string inpath)
{
  if(!pathExists(inpath, Own))
    {
      cerr << " *** Directory " << inpath << " does not exist " << endl;
      cerr << " *** Command rmdir failed " << endl;
      return;
    }
  rmdir(inpath, Own);
}


// *** Use this for saving monitoring objects in ROOT files with dir structure ***
// cd into directory (create first if it doesn't exist)
// returns success flag 
bool DaqMonitorBEInterface::cdInto(string inpath) const
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

// get folder corresponding to inpath wrt to root (create subdirs if necessary)
MonitorElementRootFolder * 
DaqMonitorBEInterface::makeDirectory(string inpath, rootDir & Dir)
{
  MonitorElementRootFolder * fret = getDirectory(inpath, Dir);
  if(!fret)
    fret = getRootFolder(Dir)->makeDirectory(inpath);

  updateMaps(fret, Dir);
  return fret;
}

// get (pointer to) last directory in inpath (null if inpath does not exist)
MonitorElementRootFolder * 
DaqMonitorBEInterface::getDirectory(string inpath, const rootDir & Dir) const
{
  if (isTopFolder(inpath)) return getRootFolder(Dir);
  else return getRootFolder(Dir)->getDirectory(inpath);
}

// save dir_fullpath with monitoring objects into root file <filename>;
// include quality test results with status >= minimum_status 
// (defined in Core/interface/QTestStatus.h);
// if dir_fullpath="", save full monitoring structure
void DaqMonitorBEInterface::save(string filename, string dir_fullpath,
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
	       if (DQM_VERBOSE>1) cout << "DaqMonitorBEInterface::save : skipping me \"" 
	                         << path->first << "\"" << endl;
 	    continue;
	    }
	 }
	 else { 
	    if (DQM_VERBOSE>1) cout << "DaqMonitorBEInterface::save : skipping me \"" 
	                         << path->first << "\"" << endl;
            continue;
	 }
      }

      if (DQM_VERBOSE) cout << "DaqMonitorBEInterface::save : saving me \"" 
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

void DaqMonitorBEInterface::readReferenceME(std::string filename) {
  open(filename,false,"",referenceDirName); 
}

std::string DaqMonitorBEInterface::getFileReleaseVersion(std::string filename) 
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

std::string DaqMonitorBEInterface::getFileDQMPatchVersion(std::string filename)
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

// copy monitoring elements from source to destination
void DaqMonitorBEInterface::copy(const MonitorElementRootFolder * const source, 
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
DaqMonitorBEInterface::removeSubsc(MonitorElementRootFolder * const dir, 
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
DaqMonitorBEInterface::add2RemovedContents(const MonitorElementRootFolder * subdir,
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
DaqMonitorBEInterface::add2RemovedContents(const MonitorElementRootFolder * subdir)
{
  for(cME_it it = subdir->objects_.begin(); it != subdir->objects_.end(); 
      ++it)
    add2RemovedContents(subdir, it->first);
}


// add null monitoring element to folder <pathname> (can NOT be folder);
// used for registering monitorables before user has subscribed to <name>
void DaqMonitorBEInterface::addElement(std::string name, std::string pathname)
{
  lock();
  MonitorElementRootFolder * folder = 
    DaqMonitorBEInterface::makeDirectory(pathname);
  folder->addElement(name);
  unlock();
}



// look for object <name> in directory <pathname>
MonitorElement * DaqMonitorBEInterface::findObject(string name, string pathname) 
  const
{
  MonitorElementRootFolder * folder = getDirectory(pathname, Own);
  if(!folder)
    return (MonitorElement *) 0;

  return folder->findObject(name);
}


// true if directory (or any subfolder at any level below it) contains
// at least one valid (i.e. non-null) monitoring element
bool DaqMonitorBEInterface::containsAnyMEs(string pathname) const
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
bool DaqMonitorBEInterface::containsAnyMonitorable(string pathname) const
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

// make new directory structure for Subscribers, Tags and CMEs
void DaqMonitorBEInterface::makeDirStructure(rootDir & Dir, string name)
{
  string ftitle = name + "_folder";
  TFolder * ff = new TFolder(name.c_str(), ftitle.c_str());
  Dir.top = new MonitorElementRootFolder(ff, ff->GetName());
  Dir.top->pathname_ = ROOT_PATHNAME;	  
  Dir.top->ownsChildren_ = false;
  Dir.top->setRootName();
}

// tag ME as <myTag>
void DaqMonitorBEInterface::tag(MonitorElement * me, unsigned int myTag)
{
  tagHelper->tag(me, myTag);
}

// opposite action of tag method
void DaqMonitorBEInterface::untag(MonitorElement * me, unsigned int myTag)
{
  tagHelper->untag(me, myTag);
}

// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
void DaqMonitorBEInterface::tag(string fullpathname, unsigned int myTag)
{
  tagHelper->tag(fullpathname, myTag);
}

// opposite action of tag method
void DaqMonitorBEInterface::untag(string fullpathname, unsigned int myTag)
{
  tagHelper->untag(fullpathname, myTag);
}

// tag all children of folder (does NOT include subfolders)
void DaqMonitorBEInterface::tagContents(string pathname, unsigned int myTag)
{
  tagHelper->tagContents(pathname, myTag);
}

// opposite action of tagContents method
void DaqMonitorBEInterface::untagContents
(string pathname, unsigned int myTag)
{
  tagHelper->untagContents(pathname, myTag);
}

// tag all children of folder, including all subfolders and their children;
// exact pathname: FAST
// pathname including wildcards (*, ?): SLOW!
void DaqMonitorBEInterface::tagAllContents
(string pathname, unsigned int myTag)
{
  tagHelper->tagAllContents(pathname, myTag);
}

// opposite action of tagAllContents method
void DaqMonitorBEInterface::untagAllContents
(string pathname, unsigned int myTag)
{
  tagHelper->untagAllContents(pathname, myTag);
}


/* get all tags, return vector with strings of the form
   <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */
void DaqMonitorBEInterface::getAllTags(vector<string> & put_here) const
{
  tagHelper->getTags(allTags, put_here);
}

void DaqMonitorBEInterface::getAddedTags(vector<string> & put_here) const
{
  tagHelper->getTags(addedTags, put_here);
}

void DaqMonitorBEInterface::getRemovedTags(vector<string> & put_here) const
{
  tagHelper->getTags(removedTags, put_here);
}



// make copies for <me> for all tags it comes with
// (to be called the first time <me> is received or read from a file)
void DaqMonitorBEInterface::makeTagCopies(MonitorElement * me)
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

// true if pathname exists
bool DaqMonitorBEInterface::pathExists(string inpath, const rootDir & Dir) const
{
  if(isTopFolder(inpath)) return true;
  return(getRootFolder(Dir)->pathExists(inpath));
}

// set the last directory in fullpath as the current directory (create if needed);
// to be invoked by user to specify directories for monitoring objects b4 booking;
// commands book1D (etc) and removeElement(name) imply elements in this directory!
void DaqMonitorBEInterface::setCurrentFolder(string fullpath)
{
  fCurrentFolder = DaqMonitorBEInterface::makeDirectory(fullpath);
}

const string DaqMonitorBEInterface::monitorDirName = "DQMData";
const string DaqMonitorBEInterface::referenceDirName = "Reference";
const string DaqMonitorBEInterface::collateDirName = "Collate";
const string DaqMonitorBEInterface::dqmPatchVersion = dqm::DQMPatchVersion ;
