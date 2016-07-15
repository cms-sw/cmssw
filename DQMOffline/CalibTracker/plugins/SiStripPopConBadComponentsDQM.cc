#include "DQMOffline/CalibTracker/plugins/SiStripPopConSourceHandler.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "DQMOffline/CalibTracker/plugins/SiStripDQMStoreReader.h"

/**
  @class SiStripBadComponentsDQMService
  @author M. De Mattia, S. Dutta, D. Giordano

  @popcon::PopConSourceHandler to read modules flagged by the DQM as bad and write in the database.
*/
class SiStripPopConBadComponentsHandlerFromDQM : public SiStripPopConSourceHandler<SiStripBadStrip>, private SiStripDQMStoreReader
{
public:
  explicit SiStripPopConBadComponentsHandlerFromDQM(const edm::ParameterSet& iConfig);
  virtual ~SiStripPopConBadComponentsHandlerFromDQM();
  // interface methods: implemented in template
  void initialize() {}
  SiStripBadStrip* getObj();
protected:
  std::string getMetaDataString();
private:
  edm::FileInPath fp_;
};

#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

SiStripPopConBadComponentsHandlerFromDQM::SiStripPopConBadComponentsHandlerFromDQM(const edm::ParameterSet& iConfig)
  : SiStripPopConSourceHandler<SiStripBadStrip>(iConfig)
  , SiStripDQMStoreReader(iConfig)
  , fp_{iConfig.getUntrackedParameter<edm::FileInPath>("file", edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))}
{
  edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::SiStripBadComponentsDQMService]";
}

SiStripPopConBadComponentsHandlerFromDQM::~SiStripPopConBadComponentsHandlerFromDQM()
{
  edm::LogInfo("SiStripBadComponentsDQMService") <<  "[SiStripBadComponentsDQMService::~SiStripBadComponentsDQMService]";
}

std::string SiStripPopConBadComponentsHandlerFromDQM::getMetaDataString()
{
  std::stringstream ss;
  ss << SiStripPopConSourceHandler<SiStripBadStrip>::getMetaDataString();
  getObj()->printSummary(ss);
  return ss.str();
}

SiStripBadStrip* SiStripPopConBadComponentsHandlerFromDQM::getObj()
{
  //*LOOP OVER THE LIST OF SUMMARY OBJECTS TO INSERT IN DB*//

  openRequestedFile();

  std::cout << "[readBadComponents]: opened requested file" << std::endl;

  std::unique_ptr<SiStripBadStrip> obj{new SiStripBadStrip{}};

  SiStripDetInfoFileReader reader(fp_.fullPath());

  dqmStore_->cd();

  std::string mdir = "MechanicalView";
  if (!goToDir(mdir)) return obj.release();
  std::string mechanicalview_dir = dqmStore_->pwd();

  std::vector<std::string> subdet_folder;
  subdet_folder.push_back("TIB");
  subdet_folder.push_back("TOB");
  subdet_folder.push_back("TEC/side_1");
  subdet_folder.push_back("TEC/side_2");
  subdet_folder.push_back("TID/side_1");
  subdet_folder.push_back("TID/side_2");

  int nDetsTotal = 0;
  int nDetsWithErrorTotal = 0;
  for( std::vector<std::string>::const_iterator im = subdet_folder.begin(); im != subdet_folder.end(); ++im ) {
    std::string dname = mechanicalview_dir + "/" + (*im);
    if (!dqmStore_->dirExists(dname)) continue;

    dqmStore_->cd(dname);
    std::vector<std::string> module_folders;
    getModuleFolderList(module_folders);
    int nDets = module_folders.size();

    int nDetsWithError = 0;
    std::string bad_module_folder = dname + "/" + "BadModuleList";
    if (dqmStore_->dirExists(bad_module_folder)) {
      std::vector<MonitorElement *> meVec = dqmStore_->getContents(bad_module_folder);
      for( std::vector<MonitorElement *>::const_iterator it = meVec.begin(); it != meVec.end(); ++it ) {
        nDetsWithError++;
        std::cout << (*it)->getName() <<  " " << (*it)->getIntValue() << std::endl;
        uint32_t detId = boost::lexical_cast<uint32_t>((*it)->getName());
        short flag = (*it)->getIntValue();

        std::vector<unsigned int> theSiStripVector;

        unsigned short firstBadStrip=0, NconsecutiveBadStrips=0;
        unsigned int theBadStripRange;

        // for(std::vector<uint32_t>::const_iterator is=BadApvList_.begin(); is!=BadApvList_.end(); ++is){

        //   firstBadStrip=(*is)*128;
        NconsecutiveBadStrips=reader.getNumberOfApvsAndStripLength(detId).first*128;

        theBadStripRange = obj->encode(firstBadStrip,NconsecutiveBadStrips,flag);

        LogDebug("SiStripBadComponentsDQMService") << "detid " << detId << " \t"
                                                   << ", flag " << flag
                                                   << std::endl;

        theSiStripVector.push_back(theBadStripRange);
        // }

        SiStripBadStrip::Range range(theSiStripVector.begin(),theSiStripVector.end());
        if ( !obj->put(detId,range) ) {
          edm::LogError("SiStripBadFiberBuilder")<<"[SiStripBadFiberBuilder::analyze] detid already exists"<<std::endl;
        }
      }
    }
    nDetsTotal += nDets;
    nDetsWithErrorTotal += nDetsWithError;        
  }
  dqmStore_->cd();

  return obj.release();
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
using SiStripPopConBadComponentsDQM = popcon::PopConAnalyzer<SiStripPopConBadComponentsHandlerFromDQM>;
DEFINE_FWK_MODULE(SiStripPopConBadComponentsDQM);
