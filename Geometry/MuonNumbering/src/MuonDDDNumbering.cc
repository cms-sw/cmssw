#include "Geometry/MuonNumbering/interface/MuonDDDNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

MuonDDDNumbering::MuonDDDNumbering( const MuonDDDConstants& muonConstants )
{
  theLevelPart=muonConstants.getValue("level");
  theSuperPart=muonConstants.getValue("super");
  theBasePart=muonConstants.getValue("base");
  theStartCopyNo=muonConstants.getValue("xml_starts_with_copyno");

  // some consistency checks

  if( theBasePart != 1 )
  {
      LogWarning( "MuonNumbering" )
	  << "MuonDDDNumbering finds unusual base constant: "
	  << theBasePart;
  }
  if( theSuperPart < 100 )
  {
      LogWarning( "MuonNumbering" )
	  << "MuonDDDNumbering finds unusual super constant: "
	  << theSuperPart;
  }
  if( theLevelPart < 10 * theSuperPart )
  {
      LogWarning( "MuonNumbering" )
	  << "MuonDDDNumbering finds unusual level constant: "
	  << theLevelPart;
  }
  if(( theStartCopyNo != 0 ) && ( theStartCopyNo != 1 ))
  {
      LogWarning( "MuonNumbering" )
	  << "MuonDDDNumbering finds unusual start value for copy numbers: "
	  << theStartCopyNo;
  }

  LogDebug( "MuonNumbering" ) 
      << "MuonDDDNumbering configured with\n"
      << "Level = " << theLevelPart
      << " Super = " << theSuperPart
      << " Base = " << theBasePart
      << " StartCopyNo = " << theStartCopyNo;
}

MuonBaseNumber
MuonDDDNumbering::geoHistoryToBaseNumber(const DDGeoHistory & history)
{
  MuonBaseNumber num;

  LogDebug( "MuonNumbering" )
      << "MuonDDDNumbering create MuonBaseNumber for\n"
      << history;

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

  LogDebug( "MuonNumbering" )
      << num.getLevels();
  for( int i = 1; i <= num.getLevels(); i++ )
  {
      LogDebug( "MuonNumbering" ) << num.getSuperNo(i) << " " << num.getBaseNo(i);
  }

  return num;
}

int
MuonDDDNumbering::getInt(const std::string & s, const DDLogicalPart & part)
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
	 LogError( "MuonNumbering" ) << " ERROR: I need only 1 " << s << " in DDLogicalPart " << part.name();
	 throw cms::Exception("GeometryBuildFailure", "MuonDDDNumbering needs only one " + s );
      }      
      return int(temp[0]);
    }
    else return 0;
}

