#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/Core/interface/Tokenizer.h"

#include "DQMServices/Core/interface/MonitorElementRootT.h"

#include "TF1.h"
#include "TMath.h"
#include "TClass.h"

using namespace dqm::me_util;

using std::cout; using std::endl; using std::cerr;
using std::string; using std::vector;

MonitorElementRootFolder::MonitorElementRootFolder(TFolder *f, const string name) 
  : MonitorElementRootObject(f,name)
{
  man.folder_flag = true; 
  // use setParent method to set parent
  parent_ = 0; 
  ownsChildren_ = true; // default value
  verbose_ = 1;
}

// (a) cleanup of folder's subfolder list 
// (this calls the clear method of folder's subfolders as well)
// (b) deletion of folder's objects 
// (c) removes "this" entry from parent's list of subfolders
void MonitorElementRootFolder::clear()
{
  // first do the cleanup for the subfolders

  // loop over subfolders, get subfolder names to be fed into findFolder
  // (this is necessary since subf->clear deletes entry from subfolds_)
  
  vector<string> subfolders;
  for(dir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    subfolders.push_back(it->first);

  for(cvIt it = subfolders.begin(); it != subfolders.end(); ++it)
    {
      MonitorElementRootFolder * subf = findFolder(*it);
      subf->setVerbose(verbose_);
      // kill the subfolder
      delete subf;
    }

  // then empty subfolder map
  subfolds_.clear();

  // then remove the objects of this folder
  removeContents();

  // last, remove "this" from parent's list of subfolders
  if(parent_)
    parent_->subfolds_.erase(getName());

}

// set name of root folder
void MonitorElementRootFolder::setRootName()
{
  MonitorElementRootFolder * curr = this;
  // find top folder (has parent_ = 0)
  while(curr->parent_)
    curr = curr->parent_;
  
  rootFolderName_ = curr->getName();
}

MonitorElementRootFolder::~MonitorElementRootFolder(void)
{
  // do a cleanup before dying...
  clear();

  if(verbose_)
    {
      // for Subscribers and Tags, find and print name of top folder
      string addName;
      if(!ownsChildren())
	addName = " (" + getRootName() + ")";

      if(getPathname() == ROOT_PATHNAME)
	cout << " Removed top directory" << addName << endl;
      else
	cout << " Removed directory " << getPathname() << addName << endl;
    }
}

// remove all monitoring elements in directory (not including subfolders)
void MonitorElementRootFolder::removeContents()
{
  vector<string> all_objects;
  for(ME_it it = objects_.begin(); it != objects_.end(); ++it)
    {
      // save list of ME names to be fed into removeMEName
      all_objects.push_back(it->first);
      // remove/unsubscribe MonitorElement
      if(it->second)
	removeElement(it->second, it->first);
    }

  // remove all ME names
  for(cvIt it = all_objects.begin(); it != all_objects.end(); ++it)
    removeMEName(*it);
}


// remove monitoring element from contents; return success flag
// if warning = true, print message if element does not exist
bool MonitorElementRootFolder::removeElement(const string & name, bool warning)
{
  ME_it it = objects_.find(name);
  if(it == objects_.end())
    {
      if(warning)
	{
	  cerr << " *** Error! Cannot remove object " << name 
	       << " from directory " << getPathname() << endl;
	  cerr << " Object does not belong to folder contents " << endl;
	}
      return false;
    }

  if(it->second)removeElement(it->second, name);
  removeMEName(name);
  
  return true;
}

// same as above with pointer instead of name
void MonitorElementRootFolder::removeElement(MonitorElement * me, 
					     const string & name)
{
  assert(me);

 if(verbose_)
    {
      cout << " Removing";
      if(!ownsChildren())
	cout << " shortcut to";
      // use "name" instead of "me->getName()" in case ME is shortcut
      // of already deleted object...
      cout << " object " << name << " in directory " << getPathname();
      if(!ownsChildren())
	cout << " (" << getRootName() << ")";
      cout << endl;
    }
 
  // release memory, if necessary
  if(this->ownsChildren())
    delete me;
}

// remove MonitorElement name from neverSent, isDesired, objects_ (if applicable)
void MonitorElementRootFolder::removeMEName(const string & name)
{
  // erase "neverSent" bool parameter (if it's there)
  neverSent.erase(name);
  
  // erase "isDesired" bool parameter (if it's there)
  isDesired.erase(name);
  
  // check if ME can be removed from menu (assume yes for now)
  bool removeFromMenu = true;

  // check if canDelete bool parameter exists
  bIt canD = canDeleteFromMenu.find(name);
  if(canD != canDeleteFromMenu.end())
    { // yes, it does exist
      if(canDeleteFromMenu[name])
	canDeleteFromMenu.erase(name);
      else
	// CANNOT remove object from map if flag = false
	removeFromMenu = false;
    }
  
  if(removeFromMenu)
    objects_.erase(name);
  else
    objects_[name] = (MonitorElement *) 0;
}

// add monitoring element to contents (could be folder)
void MonitorElementRootFolder::addElement(MonitorElement * obj)
{
  if(obj->isFolder())
    {
      MonitorElementRootFolder * child = (MonitorElementRootFolder *) obj;
      child->setParent(this);
      subfolds_[obj->getName()] = child;
      // subfolder inherits memory management rules from parent folder
      child->ownsChildren_ = ownsChildren();
    }
  else
    {
      objects_[obj->getName()] = obj;
      // have folder own <obj> only if folder owns children
      if(ownsChildren())obj->setParent(this);
    }
}

// true if at least one of the folder objects has been updated
bool MonitorElementRootFolder::wasUpdated() const
{
  cME_it it = objects_.begin();
  while(it != objects_.end())
    {
      if(it->second->wasUpdated()) return true;
      ++it;
    }
  return false;
}

// *** ??? I should merge findFolder and findObject ***

// look for subfolder "name" in current dir; return pointer (or null if not found)
MonitorElementRootFolder * 
MonitorElementRootFolder::findFolder(string name) const
{
  cdir_it it = subfolds_.find(name);
  if(it  == subfolds_.end())
    return (MonitorElementRootFolder *) 0; // name not found
  
  return (it->second);
}

// look for monitor element "name" in current dir; return pointer 
// (or null if not found)
MonitorElement * MonitorElementRootFolder::findObject(string name) const
{
  cME_it it = objects_.find(name);
  if(it  == objects_.end())
    return (MonitorElement *) 0; // name not found
  
  return it->second;
}

// get (pointer to) last directory in inpath (create necessary subdirs)
MonitorElementRootFolder * MonitorElementRootFolder::makeDirectory(string inpath)
{  
  // input path
  dqm::Tokenizer path("/",inpath);
  // get top directory --> *subf
  dqm::Tokenizer::const_iterator subf = path.begin();
  // initialize "running" folder to current folder
  MonitorElementRootFolder * folder = this;
  
  // added protection against empty strings 
  // (caused by pathnames ending with a slash: "dir1/dir2/")
  while(subf != path.end() && (*subf) != "")
    { // loop: look at all (sub)directories specified by "inpath"
      MonitorElementRootFolder *fOneDown = folder->findFolder(*subf);      
      if(!fOneDown)
	{ // if here: subf directory does not belong 
	  // to "contents" of parent directory: add it
	  fOneDown = makeDir(folder, *subf);
	  
	}  // subf directory was not found
      
      // set sub-folder as "parent" directory
      folder = fOneDown;
      // go down one (sub)directory
      ++subf;
      
    } // loop: look at all (sub)directories specified by "inpath"
  
  return folder;

}


// true if inpath exists
bool MonitorElementRootFolder::pathExists(string inpath) const
{  
  return (getDirectory(inpath) != 0);
}
 
// get (pointer to) last directory in inpath (null if inpath does not exist)
MonitorElementRootFolder * MonitorElementRootFolder::getDirectory(string inpath) 
  const
{
  // input path
  dqm::Tokenizer path("/",inpath);
  // get top directory --> *subf
  dqm::Tokenizer::const_iterator subf = path.begin();

  // initialize "running" folder to current folder
  MonitorElementRootFolder * folder = const_cast<MonitorElementRootFolder *> 
    // is there a way of not using const_cast here?
    (this);
  
  // added protection against empty strings 
  // (caused by pathnames ending with a slash: "dir1/dir2/")
  while(subf != path.end() && (*subf) != "")
    { // loop: look at all (sub)directories specified by "inpath"
      
      MonitorElementRootFolder *fOneDown = folder->findFolder(*subf);      
      if(!fOneDown)
	return (MonitorElementRootFolder *) 0;// subf directory was not found
      
      // set sub-folder as "parent" directory
      folder = fOneDown;
      // go down one (sub)directory
      ++subf;
      
    } // loop: look at all (sub)directories specified by "inpath"
  
  // if we made it down here, then inpath exists
  return folder;
}

// get folder children in string of the form <obj1>,<obj2>,<obj3>;
// (empty string for folder containing no objects)
// if skipNull = true, skip null monitoring elements
string MonitorElementRootFolder::getChildren(bool skipNull) const
{
  string contents;
  bool add_comma = false; // add comma between objects
  for(cME_it it = objects_.begin(); it != objects_.end(); ++it)
    { // loop over objects
      if(skipNull && !it->second)
	continue;

      if(add_comma)contents += ",";
      // add monitoring element to objects_
      contents += it->first;
      add_comma = true;	
    } // loop over objects
  
  return contents;
}

// true if folder has no objects
bool MonitorElementRootFolder::empty(void) const
{
  return objects_.empty();
}

// make new directory <name> and attach to directory "parent"
MonitorElementRootFolder * 
MonitorElementRootFolder::makeDir(MonitorElementRootFolder * parent, 
				  const string & name)
{
  string ftitle = name + "_folder";

  // I don't need this line except if I decide 
  // to add TFolder child to TFolder parent

  //  TFolder * top_dir = ((TFolder*) (parent->operator->()));
  //	  TFolder * ff = top_dir->AddFolder(name.c_str(), ftitle.c_str());
  TFolder * ff = new TFolder(name.c_str(), ftitle.c_str());
  MonitorElementRootFolder * child = new MonitorElementRootFolder(ff, name);
  string prefix = "";
  if(parent->pathname_ != ROOT_PATHNAME) prefix = parent->pathname_ + "/";

  // pathname of new directory is pathname of parent + / + directory name
  child->pathname_ = prefix + name;
  parent->addElement( (MonitorElement *)child );
  return child;
}

// collect all subfolders of "this"
void MonitorElementRootFolder::getSubfolders(vector<MonitorElementRootFolder*> & 
					     put_here)
{
  // loop over subfolders
  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = it->second; 
      subf->getSubfolders(put_here);
      put_here.push_back(subf);
    }
}

// true if "this" (or any subfolder at any level below "this") contains
// at least one valid (i.e. non-null) monitoring element
bool MonitorElementRootFolder::containsAnyMEs(void) const
{
  for(cME_it it = objects_.begin(); it != objects_.end(); ++it)
    {
      // check if "this" contains a non-null ME
      if(it->second)
	return true;
    }

  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = (it->second);
      // check if subfolder contains a non-null ME
      if(subf->containsAnyMEs())
	return true;
    }

  // at this point, we have found no valid ME
  return false;
}

// true if "this" (or any subfolder at any level below "this") contains
// at least one monitorable element
bool MonitorElementRootFolder::containsAnyMonitorable(void) const
{
  // if at least one ME listed return true
  if(!objects_.empty())
    return true;

  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = (it->second); 
      // check if subfolder contains any monitorable
      if(subf->containsAnyMonitorable())
	return true;
    }

  // at this point, we have found no contained monitorable
  return false;
  
}

// true if at least of one of contents or subfolders gave hasWarning = true
bool MonitorElementRootFolder::hasWarning(void) const
{

  // loop over contents
  for(cME_it it = objects_.begin(); it != objects_.end(); ++it)
    if(it->second && it->second->hasWarning())return true;
  
      
  // loop over subfolders
  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = (it->second); 
      if(subf->hasWarning())return true;
    }
  // if here no ME with a warning has been found
  return false;

}

// true if at least of one of contents or subfolders gave hasError = true
bool MonitorElementRootFolder::hasError(void) const
{
  // loop over contents
  for(cME_it it = objects_.begin(); it != objects_.end(); ++it)
    if(it->second && it->second->hasError())return true;
  
      
  // loop over subfolders
  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = (it->second); 
      if(subf->hasError())return true;
    }
  // if here no ME with an error has been found
  return false;

}

// true if at least of one of contents or subfolders gave hasOtherReport = true
bool MonitorElementRootFolder::hasOtherReport(void) const
{
  // loop over contents
  for(cME_it it = objects_.begin(); it != objects_.end(); ++it)
    if(it->second && it->second->hasOtherReport())return true;
  
      
  // loop over subfolders
  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = (it->second); 
      if(subf->hasOtherReport())return true;
    }
  // if here no ME with another (non-ok) status has been found
  return false;

}

// get status of folder (one of: STATUS_OK, WARNING, ERROR, OTHER);
// returns most sever error, where ERROR > WARNING > OTHER > STATUS_OK;
// see Core/interface/QTestStatus.h for details on "OTHER" 
int MonitorElementRootFolder::getStatus(void) const
{
  if(hasError())
    return dqm::qstatus::ERROR;
  else if(hasWarning())
    return dqm::qstatus::WARNING;
  else if(hasOtherReport())
    return dqm::qstatus::OTHER;
  else
    return dqm::qstatus::STATUS_OK;
}

// update vector with all children of folder
// (does NOT include contents of subfolders)
void MonitorElementRootFolder::getContents(vector<MonitorElement*> & put_here) 
  const
{
  for(cME_it it = objects_.begin(); it != objects_.end(); ++it)
    {
      if(it->second)
	put_here.push_back(it->second);
    }
}

// update vector with all children of folder and all subfolders
void MonitorElementRootFolder::getAllContents(vector<MonitorElement*> & put_here)
  const
{
  // add own children first
  getContents(put_here);
  // then loop over subfolders; call method recursively
  for(cdir_it it = subfolds_.begin(); it != subfolds_.end(); ++it)
    {
      MonitorElementRootFolder * subf = it->second;
      subf->getAllContents(put_here);
    }
}

// ====================================================================
// for description: see DQMServices/Core/interface/MonitorElement.h
void MonitorElementRootH1::softReset(void)
{

  TH1F * orig = (TH1F*)operator->();

  if(!reference_)
    {
      // first time softReset is called: create reference_ object
      string ref_title = orig->GetName(); ref_title += "_ref";
      reference_ = new TH1F(ref_title.c_str(), orig->GetTitle(),
			    orig->GetNbinsX(), 
			    orig->GetXaxis()->GetXmin(), 
			    orig->GetXaxis()->GetXmax());
      ((TH1F *) reference_)->SetDirectory(0);
      ((TH1F *) reference_)->Reset();
    }

  /* set reference_: 
     if this is the first time softReset is called: reference_ = orig,
     otherwise (since orig has been reduced by reference_ already):
     (new value of) reference_ = orig + (old value of) reference_  */
  ((TH1F *)reference_)->Add(orig);
 
  // now we can reset the ME
  orig->Reset();

}

// for description: see DQMServices/Core/interface/MonitorElement.h
void MonitorElementRootH2::softReset(void)
{

  TH2F * orig = (TH2F*)operator->();

  if(!reference_)
    {
      // first time softReset is called: create reference_ object
      string ref_title = orig->GetName(); ref_title += "_ref";
      reference_ = new TH2F(ref_title.c_str(), orig->GetTitle(),
			    orig->GetNbinsX(), 
			    orig->GetXaxis()->GetXmin(), 
			    orig->GetXaxis()->GetXmax(),
			    orig->GetNbinsY(), 
			    orig->GetYaxis()->GetXmin(), 
			    orig->GetYaxis()->GetXmax());
      ((TH2F *) reference_)->SetDirectory(0);
      ((TH2F *) reference_)->Reset();
    }

  /* set reference_: 
     if this is the first time softReset is called: reference_ = orig,
     otherwise (since orig has been reduced by reference_ already):
     (new value of) reference_ = orig + (old value of) reference_  */
  ((TH2F *)reference_)->Add(orig);
 
  // now we can reset the ME
  orig->Reset();

}

// for description: see DQMServices/Core/interface/MonitorElement.h
void MonitorElementRootProf::softReset(void)
{

  TProfile * orig = (TProfile*)operator->();

  if(!reference_)
    {
      // first time softReset is called: create reference_ object
      string ref_title = orig->GetName(); ref_title += "_ref";
      reference_ = new TProfile(ref_title.c_str(), orig->GetTitle(),
				orig->GetNbinsX(), 
				orig->GetXaxis()->GetXmin(), 
				orig->GetXaxis()->GetXmax(),
				orig->GetYaxis()->GetXmin(), 
				orig->GetYaxis()->GetXmax(),
				orig->GetErrorOption());
      ((TProfile *) reference_)->SetDirectory(0);
      ((TProfile *) reference_)->Reset();
    }

  /* set reference_: 
     if this is the first time softReset is called: reference_ = orig,
     otherwise (since orig has been reduced by reference_ already):
     (new value of) reference_ = orig + (old value of) reference_  */
  addProfiles((TProfile *)reference_, orig, (TProfile *)reference_, 1, 1);
 
  // now we can reset the ME
  orig->Reset();

}

// for description: see DQMServices/Core/interface/MonitorElement.h
void MonitorElementRootProf2D::softReset(void)
{

  TProfile2D * orig = (TProfile2D*)operator->();

  if(!reference_)
    {
      // first time softReset is called: create reference_ object
      string ref_title = orig->GetName(); ref_title += "_ref";
      reference_ = new TProfile2D(ref_title.c_str(), orig->GetTitle(),
				  orig->GetNbinsX(), 
				  orig->GetXaxis()->GetXmin(), 
				  orig->GetXaxis()->GetXmax(),
				  orig->GetNbinsY(), 
				  orig->GetYaxis()->GetXmin(), 
				  orig->GetYaxis()->GetXmax(),
				  orig->GetZaxis()->GetXmin(), 
				  orig->GetZaxis()->GetXmax(),
				  orig->GetErrorOption());
      ((TProfile2D *) reference_)->SetDirectory(0);
      ((TProfile2D *) reference_)->Reset();
    }

  /* set reference_: 
     if this is the first time softReset is called: reference_ = orig,
     otherwise (since orig has been reduced by reference_ already):
     (new value of) reference_ = orig + (old value of) reference_  */
  addProfiles((TProfile2D *)reference_, orig, (TProfile2D *)reference_,1,1);
  
  // now we can reset the ME
  orig->Reset();
}

// for description: see DQMServices/Core/interface/MonitorElement.h
void MonitorElementRootH3::softReset(void)
{

  TH3F * orig = (TH3F*)operator->();

  if(!reference_)
    {
      // first time softReset is called: create reference_ object
      string ref_title = orig->GetName(); ref_title += "_ref";
      reference_ = new TH3F(ref_title.c_str(), orig->GetTitle(),
			    orig->GetNbinsX(), 
			    orig->GetXaxis()->GetXmin(), 
			    orig->GetXaxis()->GetXmax(),
			    orig->GetNbinsY(), 
			    orig->GetYaxis()->GetXmin(), 
			    orig->GetYaxis()->GetXmax(),
			    orig->GetNbinsZ(), 
			    orig->GetZaxis()->GetXmin(), 
			    orig->GetZaxis()->GetXmax());
      ((TH3F *) reference_)->SetDirectory(0);
      ((TH3F *) reference_)->Reset();
    }
  
  /* set reference_: 
     if this is the first time softReset is called: reference_ = orig,
     otherwise (since orig has been reduced by reference_ already):
     (new value of) reference_ = orig + (old value of) reference_  */
  ((TH3F *)reference_)->Add(orig);
 
  // now we can reset the ME
  orig->Reset();

}

void MonitorElementRootObject::copyFunctions(TH1 * copyFromThis, TH1 * copyToThis)
{
  // will copy functions only if local-copy and original-object are equal
  // (ie. no soft-resetting or accumulating is enabled)
  if(isSoftResetEnabled() || isAccumulateEnabled())return;

  int N_func = copyFromThis->GetListOfFunctions()->GetSize();
  for(int i = 0; i != N_func; ++i)
    {
      TObject * obj = copyFromThis->GetListOfFunctions()->At(i);
     // not interested in statistics
      if(string(obj->IsA()->GetName()) == "TPaveStats")continue;

      TF1 * fn = dynamic_cast<TF1 *> (obj);
      if(fn)
	copyToThis->GetListOfFunctions()->Add(new TF1(*fn));
      else
	{
	  cout << " *** Cannot extract function " << obj->GetName() 
	       << " of type " << obj->IsA()->GetName()
	       << " for MonitorElement " << getName() << endl;
	}
    }
}

void MonitorElementRootH1::copyFrom(TH1F * just_in)
{
  TH1F * orig = (TH1F*)operator->();
  if(orig->GetTitle() != just_in->GetTitle())
    this->setTitle(just_in->GetTitle());

  if(!accumulate_on)orig->Reset();
  if(isSoftResetEnabled())
    // subtract "reference" from just_in
    orig->Add(just_in, (TH1F*)reference_, 1, -1);
  else
    orig->Add(just_in);

  MonitorElementRootObject::copyFunctions(just_in, orig);
}

void MonitorElementRootH2::copyFrom(TH2F * just_in)
{
  TH2F * orig = (TH2F*)operator->();
  if(orig->GetTitle() != just_in->GetTitle())
    this->setTitle(just_in->GetTitle());

  if(!accumulate_on)orig->Reset();
  if(isSoftResetEnabled())
    // subtract "reference" from just_in
    orig->Add(just_in, (TH2F*)reference_, 1, -1);
  else
    orig->Add(just_in);

  MonitorElementRootObject::copyFunctions(just_in, orig);
}

void MonitorElementRootProf::copyFrom(TProfile * just_in)
{
  TProfile * orig = (TProfile*)operator->();
  if(orig->GetTitle() != just_in->GetTitle())
    this->setTitle(just_in->GetTitle());

  if(!accumulate_on)orig->Reset();
  if(isSoftResetEnabled())
    // subtract "reference" from just_in
    addProfiles(just_in, (TProfile *) reference_, orig, 1, -1);
  else
    orig->Add(just_in);

  MonitorElementRootObject::copyFunctions(just_in, orig);
}

// implementation: Giuseppe.Della-Ricca@ts.infn.it
// Can be called with sum = h1 or sum = h2
void MonitorElementRootProf::addProfiles(TProfile * h1, TProfile * h2, 
					 TProfile * sum, float c1, float c2)
{
  if(!h1 || !h2 || !sum)
    {
      cerr << " *** MonitorElementRootProf: Cannot add null profiles! " << endl;
      return;
    }

  const Int_t kNstat = 6;
  
  Double_t stats1[kNstat];
  Double_t stats2[kNstat];
  Double_t stats3[kNstat];
  
  for (Int_t i = 0; i < kNstat; ++i) {
    stats1[i] = stats2[i] = stats3[i] = 0.;
  }

  h1->GetStats(stats1);
  h2->GetStats(stats2);

  for ( Int_t i = 0; i < kNstat; i++ ) {
    stats3[i] = c1*stats1[i] + c2*stats2[i];
  }
  stats3[1] = c1*TMath::Abs(c1)*stats1[1] + c2*TMath::Abs(c2)*stats2[1];

  Double_t tot_entries = c1*h1->GetEntries() + c2* h2->GetEntries();

  TArrayD* h1sumw2 = h1->GetSumw2();
  TArrayD* h2sumw2 = h2->GetSumw2();
  
  for(Int_t bin = 0; bin <= sum->GetNbinsX()+1; ++bin) 
    {
      
      Double_t entries = c1*h1->GetBinEntries(bin)+
	c2*h2->GetBinEntries(bin);
      
      Double_t content = c1*h1->GetBinEntries(bin)*h1->GetBinContent(bin)+
	c2*h2->GetBinEntries(bin)*h2->GetBinContent(bin);
      
      Double_t error=TMath::Sqrt(c1*TMath::Abs(c1)*h1sumw2->fArray[bin]+
				 c2*TMath::Abs(c2)*h2sumw2->fArray[bin]);

      sum->SetBinContent(bin, content);
      sum->SetBinError(bin, error);
      sum->SetBinEntries(bin, entries);
    }
  
  sum->SetEntries(tot_entries);
  sum->PutStats(stats3);

}

void MonitorElementRootProf2D::copyFrom(TProfile2D * just_in)
{
  TProfile2D * orig = (TProfile2D*)operator->();
  if(orig->GetTitle() != just_in->GetTitle())
    this->setTitle(just_in->GetTitle());

  if(!accumulate_on)orig->Reset();
  if(isSoftResetEnabled())
    // subtract "reference" from just_in
    addProfiles(just_in, (TProfile2D *) reference_, orig, 1, -1);
  else
    orig->Add(just_in);

  MonitorElementRootObject::copyFunctions(just_in, orig);
}

// implementation: Giuseppe.Della-Ricca@ts.infn.it
// Can be called with sum = h1 or sum = h2
void 
MonitorElementRootProf2D::addProfiles(TProfile2D * h1, TProfile2D * h2, 
				      TProfile2D * sum, float c1, float c2)
{
  if(!h1 || !h2 || !sum)
    {
      cerr << " *** MonitorElementRootProf2D: Cannot add null profiles! "<<endl;
      return;
    }

  const Int_t kNstat = 9;

  Double_t stats1[kNstat];
  Double_t stats2[kNstat];
  Double_t stats3[kNstat];

  for ( Int_t i = 0; i < kNstat; i++ ) {
    stats1[i] = stats2[i] = stats3[i] = 0.;
  }

  h1->GetStats(stats1);
  h2->GetStats(stats2);

  for ( Int_t i = 0; i < kNstat; i++ ) {
    stats3[i] = c1*stats1[i] + c2*stats2[i];
  }
  stats3[1] = c1*TMath::Abs(c1)*stats1[1] + c2*TMath::Abs(c2)*stats2[1];

  Double_t tot_entries = c1*h1->GetEntries() + c2*h2->GetEntries();

  TArrayD* h1sumw2 = h1->GetSumw2();
  TArrayD* h2sumw2 = h2->GetSumw2();
  
  for ( Int_t binx = 0; binx <= sum->GetNbinsX()+1; binx++ ) {
    for ( Int_t biny = 0; biny <= sum->GetNbinsY()+1; biny++ ) {
      
      Int_t bin = sum->GetBin(binx, biny);

      Double_t entries = c1*h1->GetBinEntries(bin)+
                         c2*h2->GetBinEntries(bin);

      Double_t content = c1*h1->GetBinEntries(bin)*h1->GetBinContent(bin)+
                         c2*h2->GetBinEntries(bin)*h2->GetBinContent(bin);

      Double_t error=TMath::Sqrt(c1*TMath::Abs(c1)*h1sumw2->fArray[bin]+
				 c2*TMath::Abs(c2)*h2sumw2->fArray[bin]);

      sum->SetBinContent(bin, content);
      sum->SetBinError(bin, error);
      sum->SetBinEntries(bin, entries);

    }
  }

  sum->SetEntries(tot_entries);
  sum->PutStats(stats3);

}

void MonitorElementRootH3::copyFrom(TH3F * just_in)
{
  TH3F * orig = (TH3F*)operator->();
  if(orig->GetTitle() != just_in->GetTitle())
    this->setTitle(just_in->GetTitle());

  if(!accumulate_on)orig->Reset();
  if(isSoftResetEnabled())
    // subtract "reference" from just_in
    orig->Add(just_in, (TH3F*)reference_, 1, -1);
  else
    orig->Add(just_in);

  MonitorElementRootObject::copyFunctions(just_in, orig);
}

// adds reference_ back into val_ contents (ie. reverts action of softReset)
void MonitorElementRootH1::unresetContents(void)
{
  TH1F * orig = (TH1F*)operator->(); 
  orig->Add((TH1F *)reference_);
}

// adds reference_ back into val_ contents (ie. reverts action of softReset)
void MonitorElementRootH2::unresetContents(void)
{
  TH2F * orig = (TH2F*)operator->(); 
  orig->Add((TH2F *)reference_);
}

// adds reference_ back into val_ contents (ie. reverts action of softReset)
void MonitorElementRootProf::unresetContents(void)
{
  TProfile * orig = (TProfile*)operator->(); 
  addProfiles(orig, (TProfile*)reference_, orig, 1, 1);
}

// adds reference_ back into val_ contents (ie. reverts action of softReset)
void MonitorElementRootProf2D::unresetContents(void)
{
  TProfile2D * orig = (TProfile2D*)operator->(); 
  addProfiles(orig, (TProfile2D*)reference_, orig, 1, 1);
}

// adds reference_ back into val_ contents (ie. reverts action of softReset)
void MonitorElementRootH3::unresetContents(void)
{
  TH3F * orig = (TH3F*)operator->(); 
  orig->Add((TH1F *)reference_);
}

float MonitorElementRootH1::getMean(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH1F*)const_ptr())->GetMean(axis);
}

float MonitorElementRootH1::getMeanError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH1F*)const_ptr())->GetMeanError(axis);
}

float MonitorElementRootH1::getRMS(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH1F*)const_ptr())->GetRMS(axis);
}

float MonitorElementRootH1::getRMSError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH1F*)const_ptr())->GetRMSError(axis);
}

float MonitorElementRootH1::getBinContent(int binx) const
{
  return ((const TH1F*)const_ptr())->GetBinContent(binx);
}

float MonitorElementRootH1::getBinError(int binx) const
{
  return ((const TH1F*)const_ptr())->GetBinError(binx);
}

float MonitorElementRootH1::getEntries(void) const
{
  return ((const TH1F*)const_ptr())->GetEntries();
}

int MonitorElementRootH1::getNbinsX() const
{
  return ((const TH1F*)const_ptr())->GetNbinsX();
}

int MonitorElementRootH1::getNbinsY() const
{
  return ((const TH1F*)const_ptr())->GetNbinsY();
}

int MonitorElementRootH1::getNbinsZ() const
{
  return ((const TH1F*)const_ptr())->GetNbinsZ();
}

float MonitorElementRootH2::getMean(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH2F*)const_ptr())->GetMean(axis);
}

float MonitorElementRootH2::getMeanError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH2F*)const_ptr())->GetMeanError(axis);
}

float MonitorElementRootH2::getRMS(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH2F*)const_ptr())->GetRMS(axis);
}

float MonitorElementRootH2::getRMSError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH2F*)const_ptr())->GetRMSError(axis);
}

float MonitorElementRootH2::getBinContent(int binx, int biny) const
{
  return ((const TH2F*)const_ptr())->GetBinContent(binx, biny);
}

float MonitorElementRootH2::getBinError(int binx, int biny) const
{
  return ((const TH2F*)const_ptr())->GetBinError(binx,biny);
}

float MonitorElementRootH2::getEntries(void) const
{
  return ((const TH2F*)const_ptr())->GetEntries();
}

int MonitorElementRootH2::getNbinsX() const
{
  return ((const TH2F*)const_ptr())->GetNbinsX();
}

int MonitorElementRootH2::getNbinsY() const
{
  return ((const TH2F*)const_ptr())->GetNbinsY();
}

int MonitorElementRootH2::getNbinsZ() const
{
  return ((const TH2F*)const_ptr())->GetNbinsZ();
}

float MonitorElementRootH3::getMean(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH3F*)const_ptr())->GetMean(axis);
}

float MonitorElementRootH3::getMeanError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH3F*)const_ptr())->GetMeanError(axis);
}

float MonitorElementRootH3::getRMS(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH3F*)const_ptr())->GetRMS(axis);
}

float MonitorElementRootH3::getRMSError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TH3F*)const_ptr())->GetRMSError(axis);
}

float MonitorElementRootH3::getBinContent(int binx,int biny,int binz) const
{
  return ((const TH3F*)const_ptr())->GetBinContent(binx,biny,binz);
}

float MonitorElementRootH3::getBinError(int binx,int biny,int binz) const
{
  return ((const TH3F*)const_ptr())->GetBinError(binx,biny,binz);
}

float MonitorElementRootH3::getEntries(void) const
{
  return ((const TH3F*)const_ptr())->GetEntries();
}

int MonitorElementRootH3::getNbinsX() const
{
  return ((const TH3F*)const_ptr())->GetNbinsX();
}

int MonitorElementRootH3::getNbinsY() const
{
  return ((const TH3F*)const_ptr())->GetNbinsY();
}

int MonitorElementRootH3::getNbinsZ() const
{
  return ((const TH3F*)const_ptr())->GetNbinsZ();
}

float MonitorElementRootProf::getMean(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile*)const_ptr())->GetMean(axis);
}

float MonitorElementRootProf::getMeanError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile*)const_ptr())->GetMeanError(axis);
}

float MonitorElementRootProf::getRMS(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile*)const_ptr())->GetRMS(axis);
}

float MonitorElementRootProf::getRMSError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile*)const_ptr())->GetRMSError(axis);
}

float MonitorElementRootProf::getBinContent(int binx) const
{
  return ((const TProfile*)const_ptr())->GetBinContent(binx);
}

float MonitorElementRootProf::getBinError(int binx) const
{
  return ((const TProfile*)const_ptr())->GetBinError(binx);
}

float MonitorElementRootProf::getEntries(void) const
{
  return ((const TProfile*)const_ptr())->GetEntries();
}

float MonitorElementRootProf::getBinEntries(int bin) const
{
  return ((const TProfile*)const_ptr())->GetBinEntries(bin);
}

float MonitorElementRootProf::getYmin() const
{
  return ((const TProfile*)const_ptr())->GetYmin();
}

float MonitorElementRootProf::getYmax() const
{
  return ((const TProfile*)const_ptr())->GetYmax();
}

int MonitorElementRootProf::getNbinsX() const
{
  return ((const TProfile*)const_ptr())->GetNbinsX();
}

int MonitorElementRootProf::getNbinsY() const
{
  return ((const TProfile*)const_ptr())->GetNbinsY();
}

int MonitorElementRootProf::getNbinsZ() const
{
  return ((const TProfile*)const_ptr())->GetNbinsZ();
}

float MonitorElementRootProf2D::getMean(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile2D*)const_ptr())->GetMean(axis);
}

float MonitorElementRootProf2D::getMeanError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile2D*)const_ptr())->GetMeanError(axis);
}

float MonitorElementRootProf2D::getRMS(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile2D*)const_ptr())->GetRMS(axis);
}

float MonitorElementRootProf2D::getRMSError(int axis) const
{
  if(!checkAxis(axis))return 0;
  return ((const TProfile2D*)const_ptr())->GetRMSError(axis);
}

float MonitorElementRootProf2D::getBinContent(int binx, int biny) const
{
  return ((const TProfile2D*)const_ptr())->GetBinContent(binx, biny);
}

float MonitorElementRootProf2D::getBinError(int binx, int biny) const
{
  return ((const TProfile2D*)const_ptr())->GetBinError(binx,biny);
}

float MonitorElementRootProf2D::getEntries(void) const
{
  return ((const TProfile2D*)const_ptr())->GetEntries();
}

float MonitorElementRootProf2D::getBinEntries(int bin) const
{
  return ((const TProfile2D*)const_ptr())->GetBinEntries(bin);
}
int MonitorElementRootProf2D::getNbinsX() const
{
  return ((const TProfile2D*)const_ptr())->GetNbinsX();
}

int MonitorElementRootProf2D::getNbinsY() const
{
  return ((const TProfile2D*)const_ptr())->GetNbinsY();
}

int MonitorElementRootProf2D::getNbinsZ() const
{
  return ((const TProfile2D*)const_ptr())->GetNbinsZ();
}

void MonitorElementRootH1::setBinContent(int binx, float content)
{
  ( (TH1F *) this->operator->() )->SetBinContent(binx, content);
}

void MonitorElementRootH1::setBinError(int binx, float error)
{
  ( (TH1F *) this->operator->() )->SetBinError(binx, error);
}

void MonitorElementRootH1::setEntries(float nentries)
{
  ( (TH1F *) this->operator->() )->SetEntries(nentries);
}

void MonitorElementRootH1::setBinLabel(int bin, std::string label, int axis)
{
  if(!label.empty())
    {
      TAxis * ax = getAxis(axis);
      if(ax)
	ax->SetBinLabel(bin, label.c_str());
    }
}

void MonitorElementRootH1::setAxisRange(float xmin, float xmax, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetRangeUser(xmin, xmax);
}

void MonitorElementRootH1::setAxisTimeDisplay(int value, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeDisplay(value);
}

void MonitorElementRootH1::setAxisTimeFormat(const char *format, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeFormat(format);
}

void MonitorElementRootH1::setAxisTimeOffset(double toffset, const char *option, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeOffset(toffset, option);
}

void MonitorElementRootH1::setAxisTitle(string axis_title, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTitle(axis_title.c_str());
}


void MonitorElementRootH1::setTitle(string new_title)
{
  ( (TH1F *) this->operator->() )->SetTitle(new_title.c_str());
}

void MonitorElementRootH2::setBinContent(int binx, int biny, float content)
{
  ( (TH2F *) this->operator->() )->SetBinContent(binx, biny, content);
}

void MonitorElementRootH2::setBinError(int binx, int biny, float error)
{
  ( (TH2F *) this->operator->() )->SetBinError(binx, biny, error);
}

void MonitorElementRootH2::setEntries(float nentries)
{
  ( (TH2F *) this->operator->() )->SetEntries(nentries);
}

void MonitorElementRootH2::setBinLabel(int bin, std::string label, int axis)
{
  if(!label.empty())
    {
      TAxis * ax = getAxis(axis);
      if(ax)
	ax->SetBinLabel(bin, label.c_str());
    }
}

void MonitorElementRootH2::setAxisRange(float xmin, float xmax, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetRangeUser(xmin, xmax);
}

void MonitorElementRootH2::setAxisTitle(string axis_title, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTitle(axis_title.c_str());
}

void MonitorElementRootH2::setAxisTimeDisplay(int value, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeDisplay(value);
}

void MonitorElementRootH2::setAxisTimeFormat(const char *format, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeFormat(format);
}

void MonitorElementRootH2::setAxisTimeOffset(double toffset, const char *option, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeOffset(toffset, option);
}

void MonitorElementRootH2::setTitle(string new_title)
{
  ( (TH2F *) this->operator->() )->SetTitle(new_title.c_str());
}

void MonitorElementRootProf::setBinContent(int binx, float content)
{
  ( (TProfile *) this->operator->() )->SetBinContent(binx, content);
}

void MonitorElementRootProf::setBinError(int binx, float error)
{
  ( (TProfile *) this->operator->() )->SetBinError(binx, error);
}

void MonitorElementRootProf::setEntries(float nentries)
{
  ( (TProfile *) this->operator->() )->SetEntries(nentries);
}

void MonitorElementRootProf::setBinLabel(int bin, std::string label, int axis)
{
  if(!label.empty())
    {
      TAxis * ax = getAxis(axis);
      if(ax)
	ax->SetBinLabel(bin, label.c_str());
    }
}

void MonitorElementRootProf::setAxisRange(float xmin, float xmax, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetRangeUser(xmin, xmax);
}

void MonitorElementRootProf::setAxisTitle(string axis_title, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTitle(axis_title.c_str());
}

void MonitorElementRootProf::setAxisTimeDisplay(int value, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeDisplay(value);
}

void MonitorElementRootProf::setAxisTimeFormat(const char *format, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeFormat(format);
}

void MonitorElementRootProf::setAxisTimeOffset(double toffset, const char *option, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeOffset(toffset, option);
}

void MonitorElementRootProf::setTitle(string new_title)
{
  ( (TProfile *) this->operator->() )->SetTitle(new_title.c_str());
}


void MonitorElementRootProf2D::setBinContent(int binx, int biny, float content)
{
  ( (TProfile2D *) this->operator->() )->SetBinContent(binx, biny, content);
}

void MonitorElementRootProf2D::setBinError(int binx, int biny, float error)
{
  ( (TProfile2D *) this->operator->() )->SetBinError(binx, biny, error);
}

void MonitorElementRootProf2D::setEntries(float nentries)
{
  ( (TProfile *) this->operator->() )->SetEntries(nentries);
}

void MonitorElementRootProf2D::setBinLabel(int bin, std::string label, int axis)
{
  if(!label.empty())
    {
      TAxis * ax = getAxis(axis);
      if(ax)
	ax->SetBinLabel(bin, label.c_str());
    }
}

void MonitorElementRootProf2D::setAxisRange(float xmin, float xmax, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetRangeUser(xmin, xmax);
}

void MonitorElementRootProf2D::setAxisTitle(string axis_title, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTitle(axis_title.c_str());
}

void MonitorElementRootProf2D::setAxisTimeDisplay(int value, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeDisplay(value);
}

void MonitorElementRootProf2D::setAxisTimeFormat(const char *format, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeFormat(format);
}

void MonitorElementRootProf2D::setAxisTimeOffset(double toffset, const char *option, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeOffset(toffset, option);
}

void MonitorElementRootProf2D::setTitle(string new_title)
{
  ( (TProfile2D *) this->operator->() )->SetTitle(new_title.c_str());
}


void MonitorElementRootH3::setBinContent(int binx,int biny,int binz,float content)
{
  ( (TH3F *) this->operator->() )->SetBinContent(binx, biny, binz, content);
}

void MonitorElementRootH3::setBinError(int binx, int biny, int binz, float error)
{
  ( (TH3F *) this->operator->() )->SetBinError(binx, biny, binz, error);
}

void MonitorElementRootH3::setEntries(float nentries)
{
  ( (TH3F *) this->operator->() )->SetEntries(nentries);
}

void MonitorElementRootH3::setBinLabel(int bin, std::string label, int axis)
{
  if(!label.empty())
    {
      TAxis * ax = getAxis(axis);
      if(ax)
	ax->SetBinLabel(bin, label.c_str());
    }
}

void MonitorElementRootH3::setAxisRange(float xmin, float xmax, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetRangeUser(xmin, xmax);
}

void MonitorElementRootH3::setAxisTitle(string axis_title, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTitle(axis_title.c_str());
}

void MonitorElementRootH3::setAxisTimeDisplay(int value, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeDisplay(value);
}

void MonitorElementRootH3::setAxisTimeFormat(const char *format, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeFormat(format);
}

void MonitorElementRootH3::setAxisTimeOffset(double toffset, const char *option, int axis)
{
  TAxis * ax = getAxis(axis);
  if(ax)
    ax->SetTimeOffset(toffset, option);
}

void MonitorElementRootH3::setTitle(string new_title)
{
  ( (TH3F *) this->operator->() )->SetTitle(new_title.c_str());
}

std::string MonitorElementRootH1::getAxisTitle(int axis) const
{
  const TAxis * ax = getAxis(axis);
  if(ax)
    return std::string(ax->GetTitle());
   return std::string("");
}
 
std::string MonitorElementRootH1::getTitle() const
{
  return std::string(((const TH1F*)const_ptr())->GetTitle());
}
 
std::string MonitorElementRootH2::getAxisTitle(int axis) const
{
  TAxis * ax = getAxis(axis);
  if(ax)
    return std::string(ax->GetTitle());
  return std::string("");
}
 
std::string MonitorElementRootH2::getTitle() const
{
  return std::string(((const TH2F*)const_ptr())->GetTitle());
}
 
std::string MonitorElementRootH3::getAxisTitle(int axis) const
{
  TAxis * ax = getAxis(axis);
  if(ax)
    return std::string(ax->GetTitle());
  return std::string("");
}

std::string MonitorElementRootH3::getTitle() const
{
  return std::string(((const TH3F*)const_ptr())->GetTitle());
}
 
void MonitorElementRootProf::setBinEntries(int bin, float nentries)
{
  ( (TProfile*) this->operator->() )->SetBinEntries(bin,nentries);
}
 
std::string MonitorElementRootProf::getAxisTitle(int axis) const
{
  TAxis * ax = getAxis(axis);
  if(ax)
    return std::string(ax->GetTitle());
   return std::string("");
}
 
std::string MonitorElementRootProf::getTitle() const
{
  return std::string(((const TProfile*)const_ptr())->GetTitle());
}
 
void MonitorElementRootProf2D::setBinEntries(int bin, float nentries)
{
  ( (TProfile2D*) this->operator->() )->SetBinEntries(bin,nentries);
}
 
std::string MonitorElementRootProf2D::getAxisTitle(int axis) const
{
  TAxis * ax = getAxis(axis);
  if(ax)
    return std::string(ax->GetTitle());
  return std::string("");
}
 
std::string MonitorElementRootProf2D::getTitle() const
{
  return std::string(((const TProfile2D*)const_ptr())->GetTitle());
}

