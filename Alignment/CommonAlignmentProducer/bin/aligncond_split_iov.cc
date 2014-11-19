#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

#include "CondCore/DBCommon/interface/IOVInfo.h"
#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/UserLogInfo.h"
#include "CondCore/DBCommon/interface/TagInfo.h"
#include "CondCore/Utilities/interface/Utilities.h"

#include "CondCore/ORA/interface/Object.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"

#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"

#include <iterator>
#include <iostream>
#include <sstream>

namespace cond {
  class AlignSplitIOV : public Utilities {
  public:
    AlignSplitIOV();
    ~AlignSplitIOV();
    int execute() override;

    template<class T>
    std::string processPayloadContainer(cond::DbSession &sourcedb,
					cond::DbSession &destdb, 
					const std::string &token,
					const std::string &containerName);
  };
}

cond::AlignSplitIOV::AlignSplitIOV()
 :Utilities("aligncond_split_iov")
{
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addOption<std::string>("sourceTag","i","tag to export( default = destination tag)");
  addOption<std::string>("destTag","t","destination tag (required)");
  addAuthenticationOptions();
  addOption<bool>("verbose","v","verbose");
}

cond::AlignSplitIOV::~AlignSplitIOV()
{

}

int cond::AlignSplitIOV::execute()
{
  initializePluginManager();
 
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  std::string destConnect = getOptionValue<std::string>("destConnect");
  
  std::string destTag = getOptionValue<std::string>("destTag");
  std::string sourceTag(destTag);
  if (hasOptionValue("sourceTag"))
    sourceTag = getOptionValue<std::string>("sourceTag");
  bool verbose = hasOptionValue("verbose");
  
  cond::Time_t since = std::numeric_limits<cond::Time_t>::min();
  cond::Time_t till = std::numeric_limits<cond::Time_t>::max();

  std::string sourceiovtoken;
  cond::TimeType sourceiovtype;

  std::string destiovtoken;

  cond::DbSession sourcedb = openDbSession("sourceConnect", false);
  cond::DbSession destdb = openDbSession("destConnect");

  sourcedb.transaction().start(false);
  cond::MetaData sourceMetadata(sourcedb);
  sourceiovtoken = sourceMetadata.getToken(sourceTag);
  if (sourceiovtoken.empty()) 
    throw std::runtime_error(std::string("tag ") + sourceTag + std::string(" not found"));

  if (verbose)
    std::cout << "source iov token: " << sourceiovtoken << std::endl;

  cond::IOVProxy iov(sourcedb, sourceiovtoken);
  sourceiovtype = iov.timetype();
  if (verbose)
    std::cout << "source iov type " << sourceiovtype << std::endl;

  since = std::max(since, cond::timeTypeSpecs[sourceiovtype].beginValue);
  till  = std::min(till,  cond::timeTypeSpecs[sourceiovtype].endValue);
  
  unsigned int counter = 0;
  for (cond::IOVProxy::const_iterator ioviterator = iov.begin();
       ioviterator != iov.end();
       ++ioviterator) {

    std::stringstream newTag;
    newTag << destTag << "_" << counter;
    
    std::cout << "iov " << counter << ":\t" 
	      << ioviterator->since() << " \t "
	      << ioviterator->till() << std::endl;
    
    if (verbose)
      std::cout << "\t" << ioviterator->token() << std::endl;

    cond::DbScopedTransaction transaction(destdb);
    transaction.start(false);
    std::string payloadContainerName = sourcedb.classNameForItem(ioviterator->token());
    std::string objToken;
    if (payloadContainerName=="Alignments")
      objToken = processPayloadContainer<Alignments>(sourcedb, destdb, 
						     ioviterator->token(), payloadContainerName);
    else if (payloadContainerName=="AlignmentErrorsExtended")
      objToken = processPayloadContainer<AlignmentErrorsExtended>(sourcedb, destdb,
							  ioviterator->token(), payloadContainerName);
    else if (payloadContainerName=="AlignmentSurfaceDeformations")
      objToken = processPayloadContainer<AlignmentSurfaceDeformations>(sourcedb, destdb,
								       ioviterator->token(), payloadContainerName);
    else if (payloadContainerName=="SiPixelLorentzAngle")
      objToken = processPayloadContainer<SiPixelLorentzAngle>(sourcedb, destdb,
								       ioviterator->token(), payloadContainerName);
    else if (payloadContainerName=="SiStripLorentzAngle")
      objToken = processPayloadContainer<SiStripLorentzAngle>(sourcedb, destdb,
								       ioviterator->token(), payloadContainerName);
    else if (payloadContainerName=="SiStripBackPlaneCorrection")
      objToken = processPayloadContainer<SiStripBackPlaneCorrection>(sourcedb, destdb,
								       ioviterator->token(), payloadContainerName);
    else {
      return 1;
    }

    cond::IOVEditor editor(destdb);
    editor.create(iov.timetype(), cond::timeTypeSpecs[sourceiovtype].endValue);
    editor.append(cond::timeTypeSpecs[sourceiovtype].beginValue, objToken);
    std::string iovToken = editor.token();
    editor.stamp(cond::userInfo(),false);
 
    cond::MetaData metadata(destdb);
    metadata.addMapping(newTag.str(), iovToken, sourceiovtype);
    transaction.commit();
    
    ::sleep(1);

    ++counter;
  }
  
  std::cout << "Total # of payload objects: " << counter << std::endl;
  
  return 0;
}

template<class T>
std::string cond::AlignSplitIOV::processPayloadContainer(cond::DbSession &sourcedb,
							 cond::DbSession &destdb, 
							 const std::string &token,
							 const std::string &containerName)
{
  boost::shared_ptr<T> object = sourcedb.getTypedObject<T>(token);
  destdb.createDatabase();
  return destdb.storeObject(object.get(), containerName);
}

int main( int argc, char** argv )
{
  cond::AlignSplitIOV utilities;
  return utilities.run(argc,argv);
}
