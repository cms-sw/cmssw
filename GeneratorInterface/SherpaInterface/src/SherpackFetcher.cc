/* Based on the cmsGetFnConnect and fn-fileget implementation by Dave Dykstra 
 * CMSSW/ FWCore/ Services/ bin/ cmsGetFnConnect.cc
 * http://cdcvs.fnal.gov/cgi-bin/public-cvs/cvsweb-public.cgi/~checkout~/frontier/client/fn-fileget.c?rev=1.1&content-type=text/plain
*/


//~ #include "SherpackFetcher.h"
#include "GeneratorInterface/SherpaInterface/interface/SherpackFetcher.h"

#include <memory>

//~ #include <libtar.h>
namespace spf {

SherpackFetcher::SherpackFetcher(edm::ParameterSet const& pset)
{
		if (!pset.exists("SherpaProcess")) SherpaProcess="";
		  else SherpaProcess=pset.getParameter<std::string>("SherpaProcess");
		if (!pset.exists("SherpackLocation")) SherpackLocation="";
		  else SherpackLocation=pset.getParameter<std::string>("SherpackLocation");
		if (!pset.exists("SherpackChecksum")) SherpackChecksum="";
		  else SherpackChecksum=pset.getParameter<std::string>("SherpackChecksum");
		if (!pset.exists("FetchSherpack")) FetchSherpack=false;
		  else FetchSherpack=pset.getParameter<bool>("FetchSherpack");
		if (!pset.exists("SherpaPath")) SherpaPath="";
		  else SherpaPath=pset.getParameter<std::string>("SherpaPath");

}

int SherpackFetcher::Fetch()
{

  std::string option  = "-c";
  std::string constr = "`cmsGetFnConnect frontier://smallfiles`";
  std::string sherpack = "sherpa_" + SherpaProcess + "_MASTER.tgz";
  std::string sherpackunzip = "sherpa_" + SherpaProcess + "_MASTER.tar";
  std::string path = SherpackLocation + "/" + sherpack;
  
  //create the command line
   
  //~ //fn-fileget -c "`cmsGetFnConnect frontier://smallfiles`" slc5_ia32_gcc434/sherpa/1.2.2-cms3/8TeV/EWK/sherpa_8TeV_ewk_Zleptons5jetsincl_50_mll_8000_MASTER.tgz 
  if(FetchSherpack == true){
		std::cout << "SherpackFetcher: Trying to fetch the Sherpack " << sherpack << std::endl;
		int res =FnFileGet(path);
		if (res!=1)  {
				std::cout <<"SherpackFetcher: Fetching of Sherpack did not succeed, terminating" << std::endl;
				return -1;
		}
		std::cout << "SherpackFetcher: Fetching successful" << std::endl;
  }
  
  std::ifstream my_file(sherpack.c_str());
	if (!my_file.good())
	{
      std::cout << "SherpackFetcher: No Sherpack found" << std::endl;
      return -2;
	}
  my_file.close();
  std::cout << "SherpackFetcher: Sherpack found" << std::endl;
  
  if(SherpackChecksum!=""){
	  char md5checksum[33]; 
	  spu::md5_File(sherpack, md5checksum);
	  for (int k=0; k<33; k++){
		if (md5checksum[k]!= SherpackChecksum[k]) {
				std::cout << "SherpackFetcher: failure, calculated and specified checksums differ!" << std::endl;
				return -3;
		}
	  }
	  std::cout << "SherpackFetcher: Calculated checksum of the Sherpack is " << md5checksum << " and matches" << std::endl;
  } else {
	  std::cout << "SherpackFetcher: Ignoring Checksum" << std::endl;
  }
  
  
  std::cout << "SherpackFetcher: Trying to unzip the Sherpack" << std::endl;
  int res=spu::Unzip(sherpack,sherpackunzip);
  if (res!=0) {
		std::cout << "SherpackFetcher: Decompressing failed " << std::endl;
		return -4;
  }
  std::cout << "SherpackFetcher: Decompressing successful " << std::endl;
  
  FILE *file = fopen(const_cast<char*>(sherpackunzip.c_str()),"r");
  if( file ) {
		std::cout << "SherpackFetcher: Decompressed Sherpack exists with name " << sherpackunzip << " starting to untar it"<<std::endl;			
		spu::Untar(file,SherpaPath.c_str());
  } else {
		std::cout << "SherpackFetcher: Could not open decompressed Sherpack" << std::endl;
		return -5;
  }
  fclose(file);
  return 0;
}

int SherpackFetcher::FnFileGet(std::string pathstring)
{
  int ec;
  unsigned long channel;
  FrontierConfig *config;
  
  std::string connectstr="";
  try {
	  std::auto_ptr<edm::SiteLocalConfig> slcptr(new edm::service::SiteLocalConfigService(edm::ParameterSet()));
	  auto slc = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::SiteLocalConfig> >(slcptr);
	  edm::ServiceToken slcToken = edm::ServiceRegistry::createContaining(slc);
	  edm::ServiceRegistry::Operate operate(slcToken);

	  edm::Service<edm::SiteLocalConfig> localconfservice;
      localconfservice->lookupCalibConnect("frontier://smallfiles");
      connectstr=localconfservice->lookupCalibConnect("frontier://smallfiles");
  } catch(cms::Exception const& e) {
	std::cerr << e.explainSelf() << std::endl;
    return 2;
  }

  if(frontier_init(malloc,free)!=0)
   {
    fprintf(stderr,"Error initializing frontier client: %s\n",frontier_getErrorMsg());
    return 2;
   }
  ec=FRONTIER_OK;
  //~ config=frontierConfig_get(connectstr,"",&ec); 
  config=frontierConfig_get(connectstr.c_str(),"",&ec); 
  if(ec!=FRONTIER_OK)
   {
    fprintf(stderr,"Error getting frontierConfig object: %s\n",frontier_getErrorMsg());
    return 2;
   }
  channel=frontier_createChannel2(config,&ec);
  if(ec!=FRONTIER_OK)
   {
    fprintf(stderr,"Error creating frontier channel: %s\n",frontier_getErrorMsg());
    return 2;
   }

 
    char uribuf[4096];
    FrontierRSBlob *frsb;
    int fd;
    int n;
    char *p;
    const char *localname;
    
	const char *path=pathstring.c_str();
    snprintf(uribuf,sizeof(uribuf)-1,    	"Frontier/type=frontier_file:1:DEFAULT&encoding=BLOB&p1=%s",path);
    ec=frontier_getRawData(channel,uribuf);
    if(ec!=FRONTIER_OK)
     {
      fprintf(stderr,"Error getting data for %s: %s\n",path,frontier_getErrorMsg());
      return 3;
     }
    frsb=frontierRSBlob_open(channel,0,1,&ec);
    if(ec!=FRONTIER_OK)
     {
      fprintf(stderr,"Error opening result blob for %s: %s\n",path,frontier_getErrorMsg());
      return 3;
     }
    // ignore the result type, will always be an array
    (void)frontierRSBlob_getByte(frsb,&ec);
    if(ec!=FRONTIER_OK)
     {
      fprintf(stderr,"Error getting result type for %s: %s\n",path,frontier_getErrorMsg());
      return 3;
     }
    n=frontierRSBlob_getInt(frsb,&ec);
    if(ec!=FRONTIER_OK)
     {
      fprintf(stderr,"Error getting result size for %s: %s\n",path,frontier_getErrorMsg());
      return 3;
     }
    p=frontierRSBlob_getByteArray(frsb,n,&ec);
    if(ec!=FRONTIER_OK)
     {
      fprintf(stderr,"Error getting result data for %s: %s\n",path,frontier_getErrorMsg());
      return 3;
     }
    localname=strrchr(path,'/');
    if(localname==NULL)
		localname=pathstring.c_str();
    else
      localname++;
    fd=open(localname,O_CREAT|O_TRUNC|O_WRONLY,0666);
    if(fd==-1)
     {
      fprintf(stderr,"Error creating %s: %s\n",localname,strerror(errno));
      ec=-1;
      return 3;
     }
    if(write(fd,p,n)<0)
     {
      fprintf(stderr,"Error writing to %s: %s\n",localname,strerror(errno));
      ec=-1;
      close(fd);
      return 3;
     }
    close(fd);
    printf("%d bytes written to %s\n",n,localname);
    frontierRSBlob_close(frsb,&ec);
   

  frontier_closeChannel(channel);

  return (ec==FRONTIER_OK);
}

SherpackFetcher::~SherpackFetcher()
{
}



} // end of namespace definition

//~ using spf::SherpackFetcher;
//~ DEFINE_FWK_MODULE(SherpackFetcher);
