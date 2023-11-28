#ifndef RecoLuminosity_LumiProducer_HLTV32DB_H
#define RecoLuminosity_LumiProducer_HLTV32DB_H
#include "CoralBase/AttributeList.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/Blob.h"
#include "CoralBase/Exception.h"
#include "RelationalAccess/ConnectionService.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ITransaction.h"
#include "RelationalAccess/ITypeConverter.h"
#include "RelationalAccess/IQuery.h"
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/IView.h"
#include "RelationalAccess/ITable.h"
#include "RelationalAccess/ITableDataEditor.h"
#include "RelationalAccess/IBulkOperation.h"

#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
#include "RecoLuminosity/LumiProducer/interface/LumiNames.h"
#include "RecoLuminosity/LumiProducer/interface/idDealer.h"
#include "RecoLuminosity/LumiProducer/interface/Exception.h"
#include "RecoLuminosity/LumiProducer/interface/DBConfig.h"
#include "RecoLuminosity/LumiProducer/interface/ConstantDef.h"
#include "RecoLuminosity/LumiProducer/interface/RevisionDML.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include "RecoLuminosity/LumiProducer/interface/DataPipe.h"
namespace lumi {
  class HLTV32DB : public DataPipe {
  public:
    const static unsigned int COMMITINTERVAL = 200;    //commit interval in LS,totalrow=nls*(~200)
    const static unsigned int COMMITLSINTERVAL = 500;  //commit interval in LS

    explicit HLTV32DB(const std::string& dest);
    unsigned long long retrieveData(unsigned int) override;
    const std::string dataType() const override;
    const std::string sourceType() const override;
    ~HLTV32DB() override;
    struct hltinfo {
      unsigned int cmsluminr;
      std::string pathname;
      unsigned int hltinput;
      unsigned int hltaccept;
      unsigned int prescale;
    };
    typedef std::map<unsigned int, std::string, std::less<unsigned int> > HltPathMap;  //order by hltpathid
    typedef std::vector<std::map<unsigned int, HLTV32DB::hltinfo, std::less<unsigned int> > > HltResult;
    void writeHltData(coral::ISessionProxy* lumisession,
                      unsigned int irunnumber,
                      const std::string& source,
                      unsigned int npath,
                      HltResult::iterator hltBeg,
                      HltResult::iterator hltEnd,
                      unsigned int commitintv);
    unsigned long long writeHltDataToSchema2(coral::ISessionProxy* lumisession,
                                             unsigned int irunnumber,
                                             const std::string& source,
                                             unsigned int npath,
                                             HltResult::iterator hltBeg,
                                             HltResult::iterator hltEnd,
                                             HltPathMap& hltpathmap,
                                             unsigned int commitintv);
  };  //cl HLTV32DB
  //
  //implementation
  //
  HLTV32DB::HLTV32DB(const std::string& dest) : DataPipe(dest) {}
  unsigned long long HLTV32DB::retrieveData(unsigned int runnumber) {
    std::string confdbschema("CMS_HLT");
    std::string hltschema("CMS_RUNINFO");
    std::string gtschema("CMS_GT_MON");
    std::string confdbpathtabname("PATHS");
    std::string triggerpathtabname("HLT_SUPERVISOR_TRIGGERPATHS");
    std::string maptabname("HLT_SUPERVISOR_SCALAR_MAP_V2");
    std::string gttabname("LUMI_SECTIONS");
    coral::ConnectionService* svc = new coral::ConnectionService;
    lumi::DBConfig dbconf(*svc);
    if (!m_authpath.empty()) {
      dbconf.setAuthentication(m_authpath);
    }
    /**retrieve hlt info with 3 queries from runinfo
       1. select prescale_index,lumi_section from cms_gt_mon.lumi_sections where run_number=:runnum; => map{ls:psindex}
       2. select name from cms_hlt.paths where pathid=:pathid
    **/
    std::string::size_type cutpos = m_source.find(';');
    std::string dbsource = m_source;
    std::string csvsource("");
    if (cutpos != std::string::npos) {
      dbsource = m_source.substr(0, cutpos);
      csvsource = m_source.substr(cutpos + 1);
    }
    //std::cout<<"dbsource: "<<dbsource<<" , csvsource: "<<csvsource<<std::endl;
    coral::ISessionProxy* srcsession = svc->connect(dbsource, coral::ReadOnly);
    coral::ITypeConverter& tpc = srcsession->typeConverter();
    tpc.setCppTypeForSqlType("unsigned int", "NUMBER(11)");
    srcsession->transaction().start(true);
    coral::ISchema& gtSchemaHandle = srcsession->schema(gtschema);
    coral::ISchema& hltSchemaHandle = srcsession->schema(hltschema);
    coral::ISchema& confdbSchemaHandle = srcsession->schema(confdbschema);
    if (!hltSchemaHandle.existsTable(triggerpathtabname) || !hltSchemaHandle.existsTable(maptabname)) {
      throw lumi::Exception("missing hlt tables", "retrieveData", "HLTV32DB");
    }
    /** select prescale_index,lumi_section from cms_gt_mon.lumi_sections where run_number=:runnum; **/
    //this is trg lumi section number
    std::vector<std::pair<unsigned int, unsigned int> > psindexmap;
    coral::AttributeList psindexVariableList;
    psindexVariableList.extend("runnumber", typeid(unsigned int));
    psindexVariableList["runnumber"].data<unsigned int>() = runnumber;
    coral::IQuery* qPsindex = gtSchemaHandle.tableHandle(gttabname).newQuery();
    coral::AttributeList psindexOutput;
    psindexOutput.extend("PRESCALE_INDEX", typeid(unsigned int));
    psindexOutput.extend("LUMI_SECTION", typeid(unsigned int));
    qPsindex->addToOutputList("PRESCALE_INDEX");
    qPsindex->addToOutputList("LUMI_SECTION");
    qPsindex->setCondition("RUN_NUMBER=:runnumber", psindexVariableList);
    qPsindex->defineOutput(psindexOutput);
    coral::ICursor& psindexCursor = qPsindex->execute();
    unsigned int lsmin = 4294967295;
    unsigned int lsmax = 0;
    while (psindexCursor.next()) {
      if (!psindexCursor.currentRow()["PRESCALE_INDEX"].isNull()) {
        unsigned int psindx = psindexCursor.currentRow()["PRESCALE_INDEX"].data<unsigned int>();
        unsigned int pslsnum = psindexCursor.currentRow()["LUMI_SECTION"].data<unsigned int>();
        if (pslsnum > lsmax) {
          lsmax = pslsnum;
        }
        if (pslsnum < lsmin) {
          lsmin = pslsnum;
        }
        psindexmap.push_back(std::make_pair(pslsnum, psindx));
      }
    }
    delete qPsindex;
    if (psindexmap.empty()) {
      srcsession->transaction().commit();
      delete srcsession;
      throw lumi::Exception("no psindex data found", "retrieveData", "HLTV32DB");
    }

    /** select distinct ( PATHID ) from HLT_SUPERVISOR_TRIGGERPATHS where runnumber=:runnumber;**/
    /** initialize hltpathmap **/
    HltPathMap hltpathmap;
    coral::AttributeList bindVariableList;
    bindVariableList.extend("runnumber", typeid(unsigned int));
    bindVariableList["runnumber"].data<unsigned int>() = runnumber;
    coral::IQuery* q1 = hltSchemaHandle.tableHandle(triggerpathtabname).newQuery();
    coral::AttributeList hltpathid;
    hltpathid.extend("hltpathid", typeid(unsigned int));
    q1->addToOutputList("distinct(PATHID)", "hltpathid");
    q1->setCondition("RUNNUMBER=:runnumber", bindVariableList);
    q1->defineOutput(hltpathid);
    coral::ICursor& c = q1->execute();
    while (c.next()) {
      unsigned int hid = c.currentRow()["hltpathid"].data<unsigned int>();
      hltpathmap.insert(std::make_pair(hid, ""));
    }
    delete q1;
    /** select name from cms_hlt.paths where pathid=:pathid **/
    /** fill up hltpathmap **/
    HltPathMap::iterator mpit;
    HltPathMap::iterator mpitBeg = hltpathmap.begin();
    HltPathMap::iterator mpitEnd = hltpathmap.end();
    for (mpit = mpitBeg; mpit != mpitEnd; ++mpit) {  //loop over paths
      coral::IQuery* mq = confdbSchemaHandle.newQuery();
      coral::AttributeList mqbindVariableList;
      mqbindVariableList.extend("pathid", typeid(unsigned int));
      mqbindVariableList["pathid"].data<unsigned int>() = mpit->first;
      mq->addToTableList(confdbpathtabname);
      mq->addToOutputList("NAME", "hltpathname");
      mq->setCondition("PATHID=:pathid", mqbindVariableList);
      coral::ICursor& mqcursor = mq->execute();
      while (mqcursor.next()) {
        std::string pathname = mqcursor.currentRow()["hltpathname"].data<std::string>();
        hltpathmap[mpit->first] = pathname;
      }
      delete mq;
    }

    /**initialize hltresult**/
    HltResult hltresult;
    unsigned int nls = lsmax - lsmin + 1;
    //std::cout<<"nls "<<nls<<std::endl;
    hltresult.reserve(nls);  //
    //fix all size
    for (unsigned int i = lsmin; i <= lsmax; ++i) {
      if (i == 0)
        continue;  //skip ls=0
      std::map<unsigned int, HLTV32DB::hltinfo> allpaths;
      HltPathMap::iterator aIt;
      HltPathMap::iterator aItBeg = hltpathmap.begin();
      HltPathMap::iterator aItEnd = hltpathmap.end();
      for (aIt = aItBeg; aIt != aItEnd; ++aIt) {
        HLTV32DB::hltinfo ct;
        ct.cmsluminr = i;
        ct.pathname = aIt->second;
        ct.hltinput = 0;
        ct.hltaccept = 0;
        ct.prescale = 0;
        allpaths.insert(std::make_pair(aIt->first, ct));
      }
      hltresult.push_back(allpaths);
    }

    bool lscountfromzero = false;

    /**select t.l1pass,t.paccept,t.pathid,m.psvalue from cms_runinfo.hlt_supervisor_triggerpaths t, cms_runinfo.hlt_supervisor_scalar_map_v2 m where m.pathid=t.pathid and m.runnumber=t.runnumber and m.runnumber=:runnum and m.psindex=:0 and t.lsnumber=:ls **/
    for (std::vector<std::pair<unsigned int, unsigned int> >::iterator it = psindexmap.begin(); it != psindexmap.end();
         ++it) {
      //loop over ls
      unsigned int lsnum = it->first;
      unsigned int psindex = it->second;
      coral::AttributeList hltdataVariableList;
      hltdataVariableList.extend("runnumber", typeid(unsigned int));
      hltdataVariableList.extend("lsnum", typeid(unsigned int));
      hltdataVariableList.extend("psindex", typeid(unsigned int));
      hltdataVariableList["runnumber"].data<unsigned int>() = runnumber;
      hltdataVariableList["lsnum"].data<unsigned int>() = lsnum;
      hltdataVariableList["psindex"].data<unsigned int>() = psindex;
      coral::IQuery* qHltData = hltSchemaHandle.newQuery();
      qHltData->addToTableList(triggerpathtabname, "t");
      qHltData->addToTableList(maptabname, "m");
      coral::AttributeList hltdataOutput;
      hltdataOutput.extend("L1PASS", typeid(unsigned int));
      hltdataOutput.extend("PACCEPT", typeid(unsigned int));
      hltdataOutput.extend("PATHID", typeid(unsigned int));
      hltdataOutput.extend("PSVALUE", typeid(unsigned int));

      qHltData->addToOutputList("t.L1PASS", "l1pass");
      qHltData->addToOutputList("t.PACCEPT", "paccept");
      qHltData->addToOutputList("t.PATHID", "pathid");
      qHltData->addToOutputList("m.PSVALUE", "psvalue");
      qHltData->setCondition(
          "m.PATHID=t.PATHID and m.RUNNUMBER=t.RUNNUMBER and m.RUNNUMBER=:runnumber AND m.PSINDEX=:psindex AND "
          "t.LSNUMBER=:lsnum",
          hltdataVariableList);
      qHltData->defineOutput(hltdataOutput);
      coral::ICursor& hltdataCursor = qHltData->execute();
      while (hltdataCursor.next()) {
        const coral::AttributeList& row = hltdataCursor.currentRow();
        if (lsnum == 0) {
          lscountfromzero = true;
          if (lscountfromzero) {
            std::cout << "hlt ls count from 0 , we skip/dodge/parry it!" << std::endl;
          }
        } else {
          unsigned int pathid = row["PATHID"].data<unsigned int>();
          std::map<unsigned int, hltinfo>& allpathinfo = hltresult.at(lsnum - 1);
          hltinfo& pathcontent = allpathinfo[pathid];
          pathcontent.hltinput = row["L1PASS"].data<unsigned int>();
          pathcontent.hltaccept = row["PACCEPT"].data<unsigned int>();
          pathcontent.prescale = row["PSVALUE"].data<unsigned int>();
        }
      }
      delete qHltData;
    }
    srcsession->transaction().commit();
    delete srcsession;
    //
    // Write into DB
    //
    unsigned int npath = hltpathmap.size();
    coral::ISessionProxy* destsession = svc->connect(m_dest, coral::Update);
    coral::ITypeConverter& lumitpc = destsession->typeConverter();
    lumitpc.setCppTypeForSqlType("unsigned int", "NUMBER(7)");
    lumitpc.setCppTypeForSqlType("unsigned int", "NUMBER(10)");
    lumitpc.setCppTypeForSqlType("unsigned long long", "NUMBER(20)");

    //for(hltIt=hltItBeg;hltIt!=hltItEnd;++hltIt){
    //  std::map<unsigned int,HLTV32DB::hltinfo>::iterator pathIt;
    //  std::map<unsigned int,HLTV32DB::hltinfo>::iterator pathItBeg=hltIt->begin();
    //  std::map<unsigned int,HLTV32DB::hltinfo>::iterator pathItEnd=hltIt->end();
    //  for(pathIt=pathItBeg;pathIt!=pathItEnd;++pathIt){
    //	std::cout<<"cmslsnr "<<pathIt->second.cmsluminr<<" "<<pathIt->second.pathname<<" "<<pathIt->second.hltinput<<" "<<pathIt->second.hltaccept<<" "<<pathIt->second.prescale<<std::endl;
    //  }
    //}
    unsigned int totalcmsls = hltresult.size();
    std::cout << "inserting totalhltls " << totalcmsls << " total path " << npath << std::endl;
    //HltResult::iterator hltItBeg=hltresult.begin();
    //HltResult::iterator hltItEnd=hltresult.end();
    unsigned long long hltdataid = 0;
    try {
      if (m_mode == "loadoldschema") {
        std::cout << "writing hlt data to old hlt table" << std::endl;
        writeHltData(destsession, runnumber, dbsource, npath, hltresult.begin(), hltresult.end(), COMMITINTERVAL);
        std::cout << "done" << std::endl;
      }
      std::cout << "writing hlt data to new lshlt table" << std::endl;
      //std::cout<<"npath "<<npath<<std::endl;
      /**for(HltResult::iterator hltIt=hltresult.begin();hltIt!=hltresult.end();++hltIt){
	  std::map<unsigned int,HLTV32DB::hltinfo>::const_iterator pathIt;
	  std::map<unsigned int,HLTV32DB::hltinfo>::const_iterator pathBeg=hltIt->begin();
	  std::map<unsigned int,HLTV32DB::hltinfo>::const_iterator pathEnd=hltIt->end();
	  for(pathIt=pathBeg;pathIt!=pathEnd;++pathIt){
	  unsigned int cmslsnum = pathIt->second.cmsluminr;
	  std::string pathname = pathIt->second.pathname;
	  unsigned int inputcount = pathIt->second.hltinput;
	  unsigned int acceptcount = pathIt->second.hltaccept;
	  unsigned int prescale = pathIt->second.prescale;
	  std::cout<<"cmslsnum "<<cmslsnum<<" pathname "<<pathname<<" inputcount "<<inputcount<<" acceptcount "<<acceptcount<<" prescale "<<prescale<<std::endl;
	  }
	  }
       **/
      hltdataid = writeHltDataToSchema2(
          destsession, runnumber, dbsource, npath, hltresult.begin(), hltresult.end(), hltpathmap, COMMITLSINTERVAL);
      std::cout << "done" << std::endl;
      delete destsession;
      delete svc;
    } catch (const coral::Exception& er) {
      std::cout << "database problem " << er.what() << std::endl;
      destsession->transaction().rollback();
      delete destsession;
      delete svc;
      throw er;
    }
    return hltdataid;
  }
  void HLTV32DB::writeHltData(coral::ISessionProxy* lumisession,
                              unsigned int irunnumber,
                              const std::string& source,
                              unsigned int npath,
                              HltResult::iterator hltItBeg,
                              HltResult::iterator hltItEnd,
                              unsigned int commitintv) {
    std::map<unsigned long long, std::vector<unsigned long long> > idallocationtable;
    unsigned int hltlscount = 0;
    unsigned int totalcmsls = std::distance(hltItBeg, hltItEnd);
    std::cout << "\t allocating total ids " << totalcmsls * npath << std::endl;
    lumisession->transaction().start(false);
    lumi::idDealer idg(lumisession->nominalSchema());
    unsigned long long hltID =
        idg.generateNextIDForTable(LumiNames::hltTableName(), totalcmsls * npath) - totalcmsls * npath;
    for (HltResult::iterator hltIt = hltItBeg; hltIt != hltItEnd; ++hltIt, ++hltlscount) {
      std::vector<unsigned long long> pathvec;
      pathvec.reserve(npath);
      for (unsigned int i = 0; i < npath; ++i, ++hltID) {
        pathvec.push_back(hltID);
      }
      idallocationtable.insert(std::make_pair(hltlscount, pathvec));
    }
    std::cout << "\t all ids allocated" << std::endl;

    coral::AttributeList hltData;
    hltData.extend("HLT_ID", typeid(unsigned long long));
    hltData.extend("RUNNUM", typeid(unsigned int));
    hltData.extend("CMSLSNUM", typeid(unsigned int));
    hltData.extend("PATHNAME", typeid(std::string));
    hltData.extend("INPUTCOUNT", typeid(unsigned int));
    hltData.extend("ACCEPTCOUNT", typeid(unsigned int));
    hltData.extend("PRESCALE", typeid(unsigned int));

    //loop over lumi LS
    unsigned long long& hlt_id = hltData["HLT_ID"].data<unsigned long long>();
    unsigned int& hltrunnum = hltData["RUNNUM"].data<unsigned int>();
    unsigned int& cmslsnum = hltData["CMSLSNUM"].data<unsigned int>();
    std::string& pathname = hltData["PATHNAME"].data<std::string>();
    unsigned int& inputcount = hltData["INPUTCOUNT"].data<unsigned int>();
    unsigned int& acceptcount = hltData["ACCEPTCOUNT"].data<unsigned int>();
    unsigned int& prescale = hltData["PRESCALE"].data<unsigned int>();
    hltlscount = 0;
    coral::IBulkOperation* hltInserter = nullptr;
    unsigned int comittedls = 0;
    for (HltResult::iterator hltIt = hltItBeg; hltIt != hltItEnd; ++hltIt, ++hltlscount) {
      std::map<unsigned int, HLTV32DB::hltinfo>::const_iterator pathIt;
      std::map<unsigned int, HLTV32DB::hltinfo>::const_iterator pathBeg = hltIt->begin();
      std::map<unsigned int, HLTV32DB::hltinfo>::const_iterator pathEnd = hltIt->end();
      if (!lumisession->transaction().isActive()) {
        lumisession->transaction().start(false);
        coral::ITable& hlttable = lumisession->nominalSchema().tableHandle(LumiNames::hltTableName());
        hltInserter = hlttable.dataEditor().bulkInsert(hltData, npath);
      } else {
        if (hltIt == hltItBeg) {
          coral::ITable& hlttable = lumisession->nominalSchema().tableHandle(LumiNames::hltTableName());
          hltInserter = hlttable.dataEditor().bulkInsert(hltData, npath);
        }
      }
      unsigned int hltpathcount = 0;
      for (pathIt = pathBeg; pathIt != pathEnd; ++pathIt, ++hltpathcount) {
        hlt_id = idallocationtable[hltlscount].at(hltpathcount);
        hltrunnum = irunnumber;
        cmslsnum = pathIt->second.cmsluminr;
        pathname = pathIt->second.pathname;
        inputcount = pathIt->second.hltinput;
        acceptcount = pathIt->second.hltaccept;
        prescale = pathIt->second.prescale;
        hltInserter->processNextIteration();
      }
      hltInserter->flush();
      ++comittedls;
      if (comittedls == commitintv) {
        std::cout << "\t committing in LS chunck " << comittedls << std::endl;
        delete hltInserter;
        hltInserter = nullptr;
        lumisession->transaction().commit();
        comittedls = 0;
        std::cout << "\t committed " << std::endl;
      } else if (hltlscount == (totalcmsls - 1)) {
        std::cout << "\t committing at the end" << std::endl;
        delete hltInserter;
        hltInserter = nullptr;
        lumisession->transaction().commit();
        std::cout << "\t done" << std::endl;
      }
    }
  }
  unsigned long long HLTV32DB::writeHltDataToSchema2(coral::ISessionProxy* lumisession,
                                                     unsigned int irunnumber,
                                                     const std::string& source,
                                                     unsigned int npath,
                                                     HltResult::iterator hltItBeg,
                                                     HltResult::iterator hltItEnd,
                                                     HltPathMap& hltpathmap,
                                                     unsigned int commitintv) {
    unsigned int totalcmsls = std::distance(hltItBeg, hltItEnd);
    std::cout << "inserting totalcmsls " << totalcmsls << std::endl;
    coral::AttributeList lshltData;
    lshltData.extend("DATA_ID", typeid(unsigned long long));
    lshltData.extend("RUNNUM", typeid(unsigned int));
    lshltData.extend("CMSLSNUM", typeid(unsigned int));
    lshltData.extend("PRESCALEBLOB", typeid(coral::Blob));
    lshltData.extend("HLTCOUNTBLOB", typeid(coral::Blob));
    lshltData.extend("HLTACCEPTBLOB", typeid(coral::Blob));
    unsigned long long& data_id = lshltData["DATA_ID"].data<unsigned long long>();
    unsigned int& hltrunnum = lshltData["RUNNUM"].data<unsigned int>();
    unsigned int& cmslsnum = lshltData["CMSLSNUM"].data<unsigned int>();
    coral::Blob& prescaleblob = lshltData["PRESCALEBLOB"].data<coral::Blob>();
    coral::Blob& hltcountblob = lshltData["HLTCOUNTBLOB"].data<coral::Blob>();
    coral::Blob& hltacceptblob = lshltData["HLTACCEPTBLOB"].data<coral::Blob>();

    unsigned long long branch_id = 3;
    std::string branch_name("DATA");
    lumi::RevisionDML revisionDML;
    lumi::RevisionDML::HltEntry hltrundata;
    std::stringstream op;
    op << irunnumber;
    std::string runnumberStr = op.str();
    lumisession->transaction().start(false);
    hltrundata.entry_name = runnumberStr;
    hltrundata.source = source;
    hltrundata.runnumber = irunnumber;
    hltrundata.npath = npath;
    std::string pathnames;
    HltPathMap::iterator hltpathmapIt;
    HltPathMap::iterator hltpathmapItBeg = hltpathmap.begin();
    HltPathMap::iterator hltpathmapItEnd = hltpathmap.end();
    for (hltpathmapIt = hltpathmapItBeg; hltpathmapIt != hltpathmapItEnd; ++hltpathmapIt) {
      if (hltpathmapIt != hltpathmapItBeg) {
        pathnames += std::string(",");
      }
      pathnames += hltpathmapIt->second;
    }
    std::cout << "\tpathnames " << pathnames << std::endl;
    hltrundata.pathnames = pathnames;
    hltrundata.entry_id = revisionDML.getEntryInBranchByName(
        lumisession->nominalSchema(), lumi::LumiNames::hltdataTableName(), runnumberStr, branch_name);
    if (hltrundata.entry_id == 0) {
      revisionDML.bookNewEntry(lumisession->nominalSchema(), LumiNames::hltdataTableName(), hltrundata);
      std::cout << "hltrundata revision_id " << hltrundata.revision_id << " entry_id " << hltrundata.entry_id
                << " data_id " << hltrundata.data_id << std::endl;
      revisionDML.addEntry(
          lumisession->nominalSchema(), LumiNames::hltdataTableName(), hltrundata, branch_id, branch_name);
    } else {
      revisionDML.bookNewRevision(lumisession->nominalSchema(), LumiNames::hltdataTableName(), hltrundata);
      std::cout << "hltrundata revision_id " << hltrundata.revision_id << " entry_id " << hltrundata.entry_id
                << " data_id " << hltrundata.data_id << std::endl;
      revisionDML.addRevision(
          lumisession->nominalSchema(), LumiNames::hltdataTableName(), hltrundata, branch_id, branch_name);
    }
    std::cout << "inserting hltrundata" << std::endl;
    revisionDML.insertHltRunData(lumisession->nominalSchema(), hltrundata);
    std::cout << "inserting lshlt data" << std::endl;

    unsigned int hltlscount = 0;
    coral::IBulkOperation* hltInserter = nullptr;
    unsigned int comittedls = 0;
    for (HltResult::iterator hltIt = hltItBeg; hltIt != hltItEnd; ++hltIt, ++hltlscount) {
      unsigned int cmslscount = hltlscount + 1;
      std::map<unsigned int, HLTV32DB::hltinfo, std::less<unsigned int> >::const_iterator pathIt;
      std::map<unsigned int, HLTV32DB::hltinfo, std::less<unsigned int> >::const_iterator pathBeg = hltIt->begin();
      std::map<unsigned int, HLTV32DB::hltinfo, std::less<unsigned int> >::const_iterator pathEnd = hltIt->end();
      if (!lumisession->transaction().isActive()) {
        lumisession->transaction().start(false);
        coral::ITable& hlttable = lumisession->nominalSchema().tableHandle(LumiNames::lshltTableName());
        hltInserter = hlttable.dataEditor().bulkInsert(lshltData, npath);
      } else {
        if (hltIt == hltItBeg) {
          coral::ITable& hlttable = lumisession->nominalSchema().tableHandle(LumiNames::lshltTableName());
          hltInserter = hlttable.dataEditor().bulkInsert(lshltData, npath);
        }
      }
      data_id = hltrundata.data_id;
      hltrunnum = irunnumber;
      cmslsnum = cmslscount;
      std::vector<unsigned int> prescales;
      prescales.reserve(npath);
      std::vector<unsigned int> hltcounts;
      hltcounts.reserve(npath);
      std::vector<unsigned int> hltaccepts;
      hltaccepts.reserve(npath);

      for (pathIt = pathBeg; pathIt != pathEnd; ++pathIt) {
        unsigned int hltcount = pathIt->second.hltinput;
        //std::cout<<"hltcount "<<hltcount<<std::endl;
        hltcounts.push_back(hltcount);
        unsigned int hltaccept = pathIt->second.hltaccept;
        //std::cout<<"hltaccept "<<hltaccept<<std::endl;
        hltaccepts.push_back(hltaccept);
        unsigned int prescale = pathIt->second.prescale;
        //std::cout<<"prescale "<<prescale<<std::endl;
        prescales.push_back(prescale);
      }
      prescaleblob.resize(sizeof(unsigned int) * npath);
      void* prescaleblob_StartAddress = prescaleblob.startingAddress();
      std::memmove(prescaleblob_StartAddress, &prescales[0], sizeof(unsigned int) * npath);
      hltcountblob.resize(sizeof(unsigned int) * npath);
      void* hltcountblob_StartAddress = hltcountblob.startingAddress();
      std::memmove(hltcountblob_StartAddress, &hltcounts[0], sizeof(unsigned int) * npath);
      hltacceptblob.resize(sizeof(unsigned int) * npath);
      void* hltacceptblob_StartAddress = hltacceptblob.startingAddress();
      std::memmove(hltacceptblob_StartAddress, &hltaccepts[0], sizeof(unsigned int) * npath);

      hltInserter->processNextIteration();
      hltInserter->flush();
      ++comittedls;
      if (comittedls == commitintv) {
        std::cout << "\t committing in LS chunck " << comittedls << std::endl;
        delete hltInserter;
        hltInserter = nullptr;
        lumisession->transaction().commit();
        comittedls = 0;
        std::cout << "\t committed " << std::endl;
      } else if (hltlscount == (totalcmsls - 1)) {
        std::cout << "\t committing at the end" << std::endl;
        delete hltInserter;
        hltInserter = nullptr;
        lumisession->transaction().commit();
        std::cout << "\t done" << std::endl;
      }
    }
    return hltrundata.data_id;
  }
  const std::string HLTV32DB::dataType() const { return "HLTV3"; }
  const std::string HLTV32DB::sourceType() const { return "DB"; }
  HLTV32DB::~HLTV32DB() {}
}  // namespace lumi
#include "RecoLuminosity/LumiProducer/interface/DataPipeFactory.h"
DEFINE_EDM_PLUGIN(lumi::DataPipeFactory, lumi::HLTV32DB, "HLTV32DB");
#endif
