#ifndef DQM_Tag_Helper_h
#define DQM_Tag_Helper_h

#include "DQMServices/Core/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/Core/interface/DQMDefinitions.h"

#include <string>

/** Helper class where the dealing with ME-tagging has been outsourced
 */

class DQMTagHelper
{
 public:
 private:
  DaqMonitorROOTBackEnd * bei;

  DQMTagHelper(DaqMonitorROOTBackEnd * useThis){bei = useThis;}

  ~DQMTagHelper(){}

  /// tag ME as <myTag>
  void tag(MonitorElement * me, unsigned int myTag);
  /// opposite action of tag method
  void untag(MonitorElement * me, unsigned int myTag);
  /// tag ME specified by full pathname (e.g. "my/long/dir/my_histo")
  void tag(const std::string & fullpathname, unsigned int myTag);
  /// opposite action of tag method
  void untag(const std::string& fullpathname, unsigned int myTag);
  /// tag all children of folder (does NOT include subfolders)
  void tagContents(std::string & pathname,unsigned int myTag);
  /// opposite action of tagContents method
  void untagContents(std::string& pathname,unsigned int myTag);
  /// tag all children of folder, including all subfolders and their children;
  /// exact pathname: FAST
  /// pathname including wildcards (*, ?): SLOW!
  void tagAllContents(std::string & pathname, unsigned int myTag);
  /// opposite action of tagAllContents method
  void untagAllContents(std::string & pathname, unsigned int myTag);

  ///
  /// add ME to Dir structure (to be called by user-action)
  void add(MonitorElement * me, dqm::me_util::rootDir & Dir);
  /// add ME to Dir structure (to be invoked when ReceiverBase receives new tag)
  void add(const std::string& pathname, const std::string& ME_name,
	   dqm::me_util::rootDir & Dir);
  /// remove ME from Dir structure
  void remove(const std::string& pathname, const std::string& ME_name,
	      dqm::me_util::rootDir & Dir);

  /// new tag for ME_name (to be called by ReceiverBase when receiving new tag)
  void tag(const std::string & pathname, const std::string & ME_name,
	   unsigned int myTag);
  /// remove tag from ME_name (to be called by ReceiverBase when notified)
  void untag(const std::string & pathname, const std::string & ME_name,
	     unsigned int myTag);

  /// get iterator in Tags corresponding to myTag;
  /// if flag=true, create if necessary; return success
  bool getTag(unsigned int myTag, dqm::me_util::tdir_it & tg, 
	      bool shouldCreate);
  /// update allTags, addedTags, removedTags with addition of <myTag>
  void addTag(const std::string & pathname, const std::string & ME_name,
	      unsigned int myTag);
  /// update allTags, addedTags, removedTags with removal of <myTag>
  void  removeTag(const std::string & pathname, const std::string & ME_name,
		  unsigned int myTag);
  /// if add=true, add tag to map <Tags>, else remove;
  /// to be called for allTags, addedTags, removedTags
  void modifyTag(const std::string& pathname, const std::string & ME_name, 
		 unsigned int myTag, bool addTag, 
		 dqm::me_util::dir_tags & tags);
  /// remove all instances of ME in pathname in addedTags, removedTags 
  /// (to be called when ME is removed)
  void removeTags(const std::string & pathname, const std::string & ME_name);

  /// get all tags for <tags> map, return vector with strings of the form
  /// <dir pathname>:<obj1>/<tag1>/<tag2>,<obj2>/<tag1>/<tag3>, etc.
  void getTags(const dqm::me_util::dir_tags & tags, 
	       std::vector<std::string> & put_here) const;

  friend class DaqMonitorROOTBackEnd;
  friend class ReceiverBase;
  friend class SenderBase;

};

#endif // #define DQM_Tag_Helper_h
