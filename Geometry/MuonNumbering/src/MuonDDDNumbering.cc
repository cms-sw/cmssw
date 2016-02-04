#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <iostream>

//#define LOCAL_DEBUG

MuonDDDNumbering::MuonDDDNumbering( const MuonDDDConstants& muonConstants ){
  //  MuonDDDConstants muonConstants;
  theLevelPart=muonConstants.getValue("level");
  theSuperPart=muonConstants.getValue("super");
  theBasePart=muonConstants.getValue("base");
  theStartCopyNo=muonConstants.getValue("xml_starts_with_copyno");

  // some consistency checks

  if (theBasePart!=1) {
    std::cout << "MuonDDDNumbering finds unusual base constant:"
	 <<theBasePart<<std::endl;
  }
  if (theSuperPart<100) {
    std::cout << "MuonDDDNumbering finds unusual super constant:"
	 <<theSuperPart<<std::endl;
  }
  if (theLevelPart<10*theSuperPart) {
    std::cout << "MuonDDDNumbering finds unusual level constant:"
	 <<theLevelPart<<std::endl;
  }
  if ((theStartCopyNo!=0)&&(theStartCopyNo!=1)) {
    std::cout << "MuonDDDNumbering finds unusual start value for copy numbers:"
	 <<theStartCopyNo<<std::endl;
  }

#ifdef LOCAL_DEBUG
  std::cout << "MuonDDDNumbering configured with"<<std::endl;
  std::cout << "Level = "<<theLevelPart<<" ";
  std::cout << "Super = "<<theSuperPart<<" ";
  std::cout << "Base = "<<theBasePart<<" ";
  std::cout << "StartCopyNo = "<<theStartCopyNo<<std::endl;
#endif

}

MuonBaseNumber MuonDDDNumbering::geoHistoryToBaseNumber(const DDGeoHistory & history){
  MuonBaseNumber num;

#ifdef LOCAL_DEBUG
  std::cout << "MuonDDDNumbering create MuonBaseNumber for"<<std::endl;
  std::cout << history <<std::endl;
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
  std::cout << num.getLevels() <<std::endl;
  for (int i=1;i<=num.getLevels();i++) {
    std::cout << num.getSuperNo(i)<<" "<<num.getBaseNo(i)<<std::endl;
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
	std::cout << " ERROR: I need only 1 " << s << " in DDLogicalPart " << part.name() << std::endl;
	 abort();
      }      
      return int(temp[0]);
    }
    else return 0;
}

