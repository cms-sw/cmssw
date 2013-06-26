#ifndef RecoLuminosity_LumiProducer_RevisionDML_H 
#define RecoLuminosity_LumiProducer_RevisionDML_H
#include <string>
#include <vector>
namespace coral{
  class ISchema;
}
namespace lumi{
  class RevisionDML{
  public:
    //class Revision{
    //public:
    //  Revision():revision_id(0),data_id(0){}
    //  unsigned long long revision_id;
    //  unsigned long long data_id;
    //};
    class DataID{
    public:
      DataID():lumi_id(0),trg_id(0),hlt_id(0){}
    public:
      unsigned long long lumi_id;	
      unsigned long long trg_id;
      unsigned long long hlt_id;
    };
    class Entry{
    public:
      Entry():revision_id(0),entry_id(0),data_id(0),entry_name(""){}
      unsigned long long revision_id;
      unsigned long long entry_id;
      unsigned long long data_id;
      std::string entry_name;
    };
    class LumiEntry : public Entry{
    public:
      LumiEntry():source(""),runnumber(0),bgev(0.0){}
      std::string source;
      unsigned int runnumber;
      float bgev;
      unsigned int ncollidingbunches;
    };
    class TrgEntry : public Entry{
      public:
      TrgEntry():source(""),runnumber(0),bitzeroname(""){}
      std::string source;
      unsigned int runnumber;
      std::string bitzeroname;
      std::string bitnames;
    };
    class HltEntry : public Entry{
    public:
      HltEntry():source(""),runnumber(0),npath(0){}
      std::string source;
      unsigned int runnumber;
      unsigned int npath;
      std::string pathnames;
    };

    /**
       select revision_id from revisions where name=:branchName
    **/    
    unsigned long long branchIdByName(coral::ISchema& schema,const std::string& branchName);
    
    /**
       select e.entry_id from entrytabl e,revisiontable r where r.revision_id=e.revision_id and e.name=:entryname and r.branch_name=:branchname
    **/
    unsigned long long getEntryInBranchByName(coral::ISchema& schema,
					      const std::string& datatableName,
					      const std::string& entryname,
					      const std::string& branchname);
    /**
       allocate new revision_id,entry_id,data_id
     **/
    void bookNewEntry(coral::ISchema& schema,
		      const std::string& datatableName,
		      Entry& entry);
    /**
       allocate new revision_id,data_id
    **/
    void bookNewRevision(coral::ISchema& schema,
		      const std::string& datatableName,
		      Entry& revision);
    /**
       1. allocate and insert a new revision in the revisions table
       2. allocate and insert a new entry into the entry table with the new revision
       3. insert into data_rev table new data_id,revision_id mapping
       insert into revisions(revision_id,branch_id,branch_name,comment,ctime) values()
       insert into datatablename_entries (entry_id,revision_id) values()
       insert into datatablename_rev(data_id,revision_id) values()
    **/
    void addEntry(coral::ISchema& schema,const std::string& datatableName,const Entry& entry,unsigned long long branch_id,const std::string& branchname );
    /**
       1.insert a new revision into the revisions table
       2.insert into data_id, revision_id pair to  datatable_revmap
       insert into revisions(revision_id,branch_id,branch_name,ctime) values()
       insert into datatable_rev(data_id,revision_id) values())
    **/
    void addRevision(coral::ISchema& schema,const std::string& datatableName,const Entry& revision,unsigned long long branch_id,std::string& branchname );
    void insertLumiRunData(coral::ISchema& schema,const LumiEntry& lumientry);
    void insertTrgRunData(coral::ISchema& schema,const TrgEntry& trgentry);
    void insertHltRunData(coral::ISchema& schema,const HltEntry& hltentry);
    
    unsigned long long currentHFDataTagId(coral::ISchema& schema);
    unsigned long long HFDataTagIdByName(coral::ISchema& schema,
					 const std::string& datatagname);
    unsigned long long addRunToCurrentHFDataTag(coral::ISchema& schema,
				  unsigned int runnum,
				  unsigned long long lumiid,
				  unsigned long long trgid,
				  unsigned long long hltid,
				  const std::string& patchcomment);
    DataID dataIDForRun(coral::ISchema& schema,
			unsigned int runnum,
			unsigned long long tagid);
    
  };
}//ns lumi
#endif
