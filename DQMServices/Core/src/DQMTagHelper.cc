#include "DQMServices/Core/interface/DQMTagHelper.h"

#include <iostream>
#include <vector>

using namespace dqm::me_util;
using std::cerr; using std::endl; 
using std::map; using std::set; using std::vector;
using std::string;

// tag ME as <myTag>
void DQMTagHelper::tag(MonitorElement * me, unsigned int myTag)
{
  if(!me)return;
  if(!myTag)
    {
      cerr << " *** Tag must be positive number! \n";
      cerr << " Tagging of MonitorElement " << me->getName() 
	   << " in " << me->getPathname() << " has failed... " << endl;
    }
  tdir_it tg;
  if(getTag(myTag, tg, true)) // create if not there
    add(me, tg->second);
  addTag(me->getPathname(), me->getName(), myTag);
}

// opposite action of tag method
void DQMTagHelper::untag(MonitorElement * me, unsigned int myTag)
{
  if(!me)return;
  string pathname = me->getPathname(); string ME_name = me->getName();

  tdir_it tg;
  if(getTag(myTag, tg, false)) // do not create
    remove(pathname, ME_name, tg->second);
  removeTag(pathname, ME_name, myTag);
}

// new tag for ME_name (to be called by ReceiverBase when receiving new tag)
void DQMTagHelper::tag(const string & pathname, const string & ME_name,
		       unsigned int myTag)
{
  tdir_it tg;
  if(getTag(myTag, tg, true)) // create if not there
    add(pathname, ME_name, tg->second);
  addTag(pathname, ME_name, myTag);
}

// remove tag from ME_name (to be called by ReceiverBase when notified)
void DQMTagHelper::untag(const string & pathname, const string & ME_name,
		       unsigned int myTag)
{
  tdir_it tg;
  if(getTag(myTag, tg, false)) // do not create
    remove(pathname, ME_name, tg->second);
  removeTag(pathname, ME_name, myTag);
}

// remove all instances of ME in pathname in addedTags, removedTags 
// (to be called when ME is removed)
void DQMTagHelper::removeTags(const string & pathname, const string & ME_name)
{
  bei->lock();

  // we do not want to erase from allTags, because even when ME is removed
  // tag property is preserved!

  dirt_it dir = bei->addedTags.find(pathname);
  if(dir != bei->addedTags.end())
    (dir->second).erase(ME_name);
  
  dir = bei->removedTags.find(pathname);
  if(dir != bei->removedTags.end())
    (dir->second).erase(ME_name);  

  bei->unlock();
}

// update allTags, addedTags, removedTags with addition of <myTag>
void DQMTagHelper::addTag(const string & pathname, const string & ME_name,
			  unsigned int myTag)
{
  bei->lock();
  modifyTag(pathname, ME_name, myTag, true, bei->allTags); // add
  modifyTag(pathname, ME_name, myTag, true, bei->addedTags); // add
  modifyTag(pathname, ME_name, myTag, false, bei->removedTags); // remove
  bei->unlock();
}

// update allTags, addedTags, removedTags with removal of <myTag>
void DQMTagHelper::removeTag(const string & pathname, const string & ME_name,
			     unsigned int myTag)
{

  bei->lock();
  modifyTag(pathname, ME_name, myTag, false, bei->allTags); // remove
  modifyTag(pathname, ME_name, myTag, false, bei->addedTags); // remove
  modifyTag(pathname, ME_name, myTag, true, bei->removedTags); // add
  bei->unlock();
}

// if add=true, add tag to map <Tags>, else remove;
// to be called for allTags, addedTags, removedTags
void DQMTagHelper::modifyTag
(const string& pathname, const string & ME_name, unsigned int myTag, bool addTag,
 dir_tags & tags)
{
  dirt_it dir = tags.find(pathname);

  // pathname does not exist in map <tags>
  if(dir == tags.end())
    {
      // if we are removing the tag, there is nothing left to do
      if(!addTag)return;

      std::pair<dirt_it, bool> newEntry;
      map<string, set<unsigned int> > tmp;
      newEntry = tags.insert(dir_tags::value_type(pathname, tmp) );
      if(newEntry.second) // <bool flag> = true for success
	dir = newEntry.first;
      else
	{
	  cerr << " *** Failed to add tag " << myTag << " for ME " << ME_name
	       << " in directory " << pathname << endl;
	  return;
	}
    }

  // look for ME in dir->second set
  tags_it me = (dir->second).find(ME_name);
  // ME name does not exist in directory
  if(me == dir->second.end())
    {
      // if we are removing the tag, there is nothing left to do
      if(!addTag)return;

      set<unsigned int> tmp_tag; tmp_tag.insert(myTag);
      (dir->second)[ME_name] = tmp_tag;
    }
  else
    {
      // ME name does exist in directory: add or remove according to flag
      if(addTag)
	me->second.insert(myTag);
      else
	me->second.erase(myTag);
    }
    
}

// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
void DQMTagHelper::tag(const string & fullpathname, unsigned int myTag)
{
  MonitorElement * me = bei->get(fullpathname);
  if(!me)
    {
      cerr << " *** Cannot find MonitorElement " << fullpathname << endl;
      return;
    }

  tag(me, myTag);
}

// opposite action of tag method
void DQMTagHelper::untag(const string& fullpathname, unsigned int myTag)
{
  MonitorElement * me = bei->get(fullpathname);
  if(!me)
    {
      cerr << " *** Cannot find MonitorElement " << fullpathname << endl;
      return;
    }

  untag(me, myTag);
}

// tag all children of folder (does NOT include subfolders)
void DQMTagHelper::tagContents(string & pathname, unsigned int myTag)
{
  vME contents;
  bei->getContents(pathname, bei->Own, contents);
  if(contents.empty())
    return;

  tdir_it tg;
  if(!getTag(myTag, tg, true)) // create if not there
    return;

  for(vME::iterator it = contents.begin(); it != contents.end(); ++it)
    {
      add(*it, tg->second);
      addTag((*it)->getPathname(), (*it)->getName(), myTag);
    }
}

// opposite action of tagContents method
void DQMTagHelper::untagContents(string & pathname, unsigned int myTag)
{
  vME contents;
  bei->getContents(pathname, bei->Own, contents);
  if(contents.empty())
    return;

  tdir_it tg;
  if(!getTag(myTag, tg, false)) // do not create
    return;
  
  for(vME::iterator it = contents.begin(); it != contents.end(); ++it)
    {
      string ME_name = (*it)->getName();
      remove(pathname, ME_name, tg->second);
      removeTag(pathname, ME_name, myTag);
    }
}

// tag all children of folder, including all subfolders and their children;
// exact pathname: FAST
// pathname including wildcards (*, ?): SLOW!
void DQMTagHelper::tagAllContents(string & pathname, unsigned int myTag)
{
  vME contents;
  bei->getAllContents(pathname, bei->Own, contents);
  if(contents.empty())
    return;

  tdir_it tg;
  if(!getTag(myTag, tg, true)) // create if not there
    return;
  
  for(vME::iterator it = contents.begin(); it != contents.end(); ++it)
    {
      add(*it, tg->second);  
      addTag((*it)->getPathname(), (*it)->getName(), myTag);
    }
}

// opposite action of tagAllContents method
void DQMTagHelper::untagAllContents(string & pathname, unsigned int myTag)
{
  vME contents;
  bei->getAllContents(pathname, bei->Own, contents);
  if(contents.empty())
    return;
  
  tdir_it tg;
  if(!getTag(myTag, tg, false)) // do not create
    return;
  
  for(vME::iterator it = contents.begin(); it != contents.end(); ++it)
    {
      string pathname = (*it)->getPathname(); 
      string ME_name = (*it)->getName();
      remove(pathname, ME_name, tg->second);
      removeTag(pathname, ME_name, myTag);
    }
}

// get iterator in Tags corresponding to myTag; 
// if flag=true, create if necessary; return success
bool DQMTagHelper::getTag(unsigned int myTag, tdir_it & tg, bool shouldCreate)
{
  tg = bei->Tags.find(myTag);

  if(tg == bei->Tags.end())
    {

      if(!shouldCreate)return false;

      // myTag not appearing in Tags; must construct root folder
      std::pair<tdir_it, bool> newEntry;
      newEntry = bei->Tags.insert(tag_map::value_type(myTag, rootDir()) );
      
      if(newEntry.second) // <bool flag> = true for success
	tg = newEntry.first;
      else
	{
	  cerr << " *** Failed to construct root directory for Tag "
	       << myTag << endl;
	  return false;
	}     
    }

  if(!tg->second.top)
    {
      // root folder for tag does not exist; create now
      std::ostringstream tagName; tagName << "Tag: " << myTag;
      bei->makeDirStructure(tg->second, tagName.str());
    }

  return true;
}

/* get all tags for <tags> map, return vector with strings of the form
   <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc. */
void DQMTagHelper::getTags
(const dir_tags & tags, vector<string> & put_here) const
{
  put_here.clear();
  for(cdirt_it dir = tags.begin(); dir != tags.end(); ++dir)
    { // loop over pathnames
      
      bool add_comma = false; // add comma between objects
      string entry;

      for(ctags_it me = dir->second.begin(); me != dir->second.end(); ++me)
	{ // loop over MEs

	  if(add_comma)entry += ",";
	  entry += me->first; // add ME name

	  for(set<unsigned int>::const_iterator tg = me->second.begin();
	      tg != me->second.end(); ++tg)
	    { // loop over tags for ME
	      std::ostringstream tagNo; tagNo << (*tg);
	      entry += "/" + tagNo.str(); // add tag #
	      
	      add_comma = true;
	    } // loop over tags for ME
	  
	} // loop over MEs
      
      if(!entry.empty())
	{
	  entry = dir->first + ":" + entry; // add pathname
	  put_here.push_back(entry);
	}
    } // loop over pathnames
}

// add ME to Dir structure (to be called by user-action)
// (no bei-locking needed; this is just for Tags directory)
void DQMTagHelper::add(MonitorElement * me, rootDir & Dir)
{
  assert(me);
  MonitorElementRootFolder* folder=bei->makeDirectory(me->getPathname(),Dir);
  folder->addElement(me);
}

// add ME to Dir structure (to be invoked when ReceiverBase receives new tag)
// (no bei-locking needed; this is just for Tags directory)
void DQMTagHelper::add(const string& pathname, const string& ME_name,
		       rootDir & Dir)
{
  MonitorElementRootFolder * folder = bei->makeDirectory(pathname, Dir);
  if(!folder->findObject(ME_name))
    folder->addElement(ME_name);
}

// remove ME from Dir structure
// (no bei-locking needed; this is just for Tags directory)
void DQMTagHelper::remove(const string& pathname, const string& ME_name,
				   rootDir & Dir)
{
  MonitorElementRootFolder * folder = bei->getDirectory(pathname, Dir);
  if(folder)
    folder->removeElement(ME_name, false); // no warning
}
