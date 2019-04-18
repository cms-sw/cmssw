#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define LOCAL_DEBUG

MuonDDDNumbering::MuonDDDNumbering( const MuonDDDConstants& muonConstants ){
  //  MuonDDDConstants muonConstants;
  theLevelPart=muonConstants.getValue("level");
  theSuperPart=muonConstants.getValue("super");
  theBasePart=muonConstants.getValue("base");
  theStartCopyNo=muonConstants.getValue("xml_starts_with_copyno");

  // some consistency checks

  if (theBasePart!=1) {
    edm::LogWarning("Geometry") 
      << "MuonDDDNumbering finds unusual base constant:" << theBasePart;
  }
  if (theSuperPart<100) {
    edm::LogWarning("Geometry") 
      << "MuonDDDNumbering finds unusual super constant:" << theSuperPart;
  }
  if (theLevelPart<10*theSuperPart) {
    edm::LogWarning("Geometry") 
      << "MuonDDDNumbering finds unusual level constant:" << theLevelPart;
  }
  if ((theStartCopyNo!=0)&&(theStartCopyNo!=1)) {
    edm::LogWarning("Geometry") 
      << "MuonDDDNumbering finds unusual start value for copy numbers:"
      << theStartCopyNo;
  }

#ifdef LOCAL_DEBUG
  edm::LogVerbatim("Geometry")
    << "MuonDDDNumbering configured with"
    << " Level = " << theLevelPart << " Super = " << theSuperPart
    << " Base = " << theBasePart << " StartCopyNo = " << theStartCopyNo;
#endif

}

MuonBaseNumber MuonDDDNumbering::geoHistoryToBaseNumber(const DDGeoHistory & history){
  MuonBaseNumber num;

#ifdef LOCAL_DEBUG
  edm::LogVerbatim("Geometry") 
    << "MuonDDDNumbering create MuonBaseNumber for " << history;
#endif

  //loop over all parents and check
  DDGeoHistory::const_iterator cur=history.begin();
  DDGeoHistory::const_iterator end=history.end();
  while (cur!=end) {
    const DDLogicalPart & ddlp = cur->logicalPart();
    const int tag=getInt("CopyNoTag",ddlp)/theLevelPart;
    if (tag>0) {
      const int offset=getInt("CopyNoOffset",ddlp);
      const int copyno=(cur->copyno())+offset%theSuperPart;
      const int super=offset/theSuperPart;
      num.addBase(tag,super,copyno-theStartCopyNo);
    }
    cur++;
  }

#ifdef LOCAL_DEBUG
  edm::LogVerbatim("Geometry") << "MuonDDDNumbering::" <<  num.getLevels();
  for (int i=1;i<=num.getLevels();i++) {
    edm::LogVerbatim("Geometry") 
      << "[" << i << "] " << num.getSuperNo(i) << " " << num.getBaseNo(i);
  }
#endif
 
  return num;
}

int MuonDDDNumbering::getInt(const std::string & s, const DDLogicalPart & part)
{
    DDValue val(s);
    std::vector<const DDsvalues_type *> result = part.specifics();
    std::vector<const DDsvalues_type *>::iterator it = result.begin();
    bool foundIt = false;
    for (; it != result.end(); ++it)
    {
	foundIt = DDfetch(*it,val);
	if (foundIt) break;
    }    
    if (foundIt)
    { 
      std::vector<double> temp = val.doubles();
      if (temp.size() != 1)
      {
	edm::LogError("Geometry") 
	  << "MuonDDDNumbering:: ERROR: I need only 1 " << s 
	  << " in DDLogicalPart " << part.name();
	 abort();
      }      
      return int(temp[0]);
    }
    else return 0;
}

