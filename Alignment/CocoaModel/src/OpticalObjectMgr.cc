#include "Alignment/CocoaModel/interface/OpticalObjectMgr.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include <cstdlib>

OpticalObjectMgr* OpticalObjectMgr::theInstance = nullptr;
  
//-----------------------------------------------------------------------
OpticalObjectMgr* OpticalObjectMgr::getInstance()
{
  if( !theInstance ) {
    theInstance = new OpticalObjectMgr;
    theInstance->theLastCmsSwID = 1;
    //    theInstance->verbose = ALIUtils::verbosity();
  }
  return theInstance;
}


//-----------------------------------------------------------------------
OpticalObject* OpticalObjectMgr::findOptO( const ALIstring& longName, bool exists ) const
{
  OpticalObject* opto = nullptr;
  msopto::const_iterator cite = theOptODict.find( longName );
  if( cite == theOptODict.end() ) {
    if( exists ) {
      std::cerr << "!!!! EXITING: OptO not found: " << longName << std::endl;
      exit(1);
    } else {
      std::cerr << "!! WARNING: OptO not found: " << longName << std::endl;
    }
  } else {
    opto = (*cite).second;
  }

  return opto;

}  


//-----------------------------------------------------------------------
std::vector<OpticalObject*> OpticalObjectMgr::findOptOs( const ALIstring& name, bool exists ) const
{
  std::vector<OpticalObject*> vopto;
  msopto::const_iterator cite;
  //----- Look for OptO's that contains 'name' in its longName as the last word (after the last '/')
  for( cite = theOptODict.begin(); cite != theOptODict.end(); ++cite ) {
    ALIstring oname = (*cite).first;
    int nf = oname.rfind( name );
    int sf = oname.rfind( '/' );
    if( nf != -1 && sf <= nf ) {
      vopto.push_back( (*cite).second );
    }
  }

  if( vopto.empty() ) {
    if( exists ) {
      std::cerr << "!!!! EXITING: OptO not found: " << name << std::endl;
      exit(1);
    } else {
      std::cerr << "!! WARNING: OptO not found: " << name << std::endl;
    }
  }

  return vopto;

}


//-----------------------------------------------------------------------
void OpticalObjectMgr::dumpOptOs( std::ostream& out ) const
{
  std::cout << "OPTICALOBJECT list size " << theOptODict.size() << std::endl;
  std::vector< OpticalObject* >::const_iterator vocite;
  for( vocite = Model::OptOList().begin(); vocite != Model::OptOList().end(); ++vocite ) {
    ALIstring name = (*vocite)->name();
    ALIUtils::dump3v( (*vocite)->centreGlobal(), name + " CENTRE GLOBAL: " );
    if( (*vocite)->parent() != nullptr ) ALIUtils::dump3v( (*vocite)->centreLocal(),  name + "  CENTRE LOCAL: " ); //not for the 'system'
  }

}


//-----------------------------------------------------------------------
ALIuint OpticalObjectMgr::buildCmsSwID()
{
  return theLastCmsSwID++;
}
