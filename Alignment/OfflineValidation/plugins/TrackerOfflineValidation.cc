// -*- C++ -*-
//
// Package:    TrackerOfflineValidation
// Class:      TrackerOfflineValidation
// 
/**\class TrackerOfflineValidation TrackerOfflineValidation.cc Alignment/Validator/src/TrackerOfflineValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Erik Butz
//         Created:  Tue Dec 11 14:03:05 CET 2007
// $Id: TrackerOfflineValidation.cc,v 1.56 2013/01/07 20:46:23 wmtan Exp $
//
//

// system include files
#include <memory>
#include <map>
#include <sstream>
#include <math.h>
#include <utility>
#include <vector>
#include <iostream>

// ROOT includes
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TMath.h"

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

//
// class declaration
//
class TrackerOfflineValidation : public edm::EDAnalyzer {
public:
  explicit TrackerOfflineValidation(const edm::ParameterSet&);
  ~TrackerOfflineValidation();
  
  enum HistogrammType { XResidual, NormXResidual, 
			YResidual, /*NormYResidual, */
			XprimeResidual, NormXprimeResidual, 
			YprimeResidual, NormYprimeResidual,
			XResidualProfile, YResidualProfile };
  
private:

  struct ModuleHistos{
    ModuleHistos() :  ResHisto(), NormResHisto(), ResYHisto(), /*NormResYHisto(),*/
		      ResXprimeHisto(), NormResXprimeHisto(), 
		      ResYprimeHisto(), NormResYprimeHisto(),
                      ResXvsXProfile(), ResXvsYProfile(),
                      ResYvsXProfile(), ResYvsYProfile(),
                      LocalX(), LocalY() {} 
    TH1* ResHisto;
    TH1* NormResHisto;
    TH1* ResYHisto;
    /* TH1* NormResYHisto; */
    TH1* ResXprimeHisto;
    TH1* NormResXprimeHisto;
    TH1* ResYprimeHisto;
    TH1* NormResYprimeHisto;

    TProfile* ResXvsXProfile;
    TProfile* ResXvsYProfile;
    TProfile* ResYvsXProfile;
    TProfile* ResYvsYProfile;

    TH1* LocalX;
    TH1* LocalY;
  };
  
  // container struct to organize collection of histogramms during endJob
  struct SummaryContainer{
    SummaryContainer() : sumXResiduals_(), summaryXResiduals_(), 
			 sumNormXResiduals_(), summaryNormXResiduals_(),
			 sumYResiduals_(), summaryYResiduals_() ,
			 sumNormYResiduals_(), summaryNormYResiduals_() {}
    
    TH1* sumXResiduals_;
    TH1* summaryXResiduals_;
    TH1* sumNormXResiduals_;
    TH1* summaryNormXResiduals_;
    TH1* sumYResiduals_;
    TH1* summaryYResiduals_;
    TH1* sumNormYResiduals_;
    TH1* summaryNormYResiduals_;
  };
  
  
  struct DirectoryWrapper{
    DirectoryWrapper(const DirectoryWrapper& upDir,const std::string& newDir,
		     const std::string& basedir,bool useDqmMode)
      : tfd(0),
	dqmMode(useDqmMode),
	theDbe(0) {
      if (newDir.length()!=0){
        if(upDir.directoryString.length()!=0)directoryString=upDir.directoryString+"/"+newDir;
	else directoryString = newDir;
      }
      else
	directoryString=upDir.directoryString;

      if (!dqmMode){
	if (newDir.length()==0) tfd.reset(&(*upDir.tfd));
	else
	  tfd.reset(new TFileDirectory(upDir.tfd->mkdir(newDir)));
      }
      else {
	theDbe=edm::Service<DQMStore>().operator->();
      }
    }
    
    DirectoryWrapper(const std::string& newDir,const std::string& basedir,bool useDqmMode)
      : tfd(0),
	dqmMode(useDqmMode),
	theDbe(0) {
      if (!dqmMode){
	edm::Service<TFileService> fs;
	if (newDir.length()==0){
	  tfd.reset(new TFileDirectory(static_cast<TFileDirectory&>(*fs)));
	}
	else {
	  tfd.reset(new TFileDirectory(fs->mkdir(newDir)));
	  directoryString=newDir;
	}
      }
      else {
	if (newDir.length()!=0){
	  if(basedir.length()!=0)directoryString=basedir+"/"+newDir;
	  else directoryString = newDir;
	}
	else directoryString=basedir;
	theDbe=edm::Service<DQMStore>().operator->();
      }
    }
    // Generalization of Histogram Booking; allows switch between TFileService and DQMStore
    template <typename T> TH1* make(const char* name,const char* title,int nBinX,double minBinX,double maxBinX);
    template <typename T> TH1* make(const char* name,const char* title,int nBinX,double *xBins);//variable bin size in x for profile histo 
    template <typename T> TH1* make(const char* name,const char* title,int nBinX,double minBinX,double maxBinX,int nBinY,double minBinY,double maxBinY);
    template <typename T> TH1* make(const char* name,const char* title,int nBinX,double minBinX,double maxBinX,double minBinY,double maxBinY);  // at present not used
    
    std::auto_ptr<TFileDirectory> tfd;
    std::string directoryString;
    const bool dqmMode;
    DQMStore* theDbe;
  };
  
  
  // 
  // ------------- private member function -------------
  // 
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  virtual void checkBookHists(const edm::EventSetup& setup);

  void bookGlobalHists(DirectoryWrapper& tfd);
  void bookDirHists(DirectoryWrapper& tfd, const Alignable& ali, const TrackerTopology* tTopo);
  void bookHists(DirectoryWrapper& tfd, const Alignable& ali, const TrackerTopology* tTopo, align::StructureType type, int i); 
 
  void collateSummaryHists( DirectoryWrapper& tfd, const Alignable& ali, int i, 
			    std::vector<TrackerOfflineValidation::SummaryContainer>& vLevelProfiles);
  
  void fillTree(TTree& tree, const std::map<int, TrackerOfflineValidation::ModuleHistos>& moduleHist_, 
		TkOffTreeVariables& treeMem, const TrackerGeometry& tkgeom, const TrackerTopology* tTopo);
  
  TrackerOfflineValidation::SummaryContainer bookSummaryHists(DirectoryWrapper& tfd, 
							      const Alignable& ali, 
							      align::StructureType type, int i); 

  ModuleHistos& getHistStructFromMap(const DetId& detid); 

  bool isBarrel(uint32_t subDetId);
  bool isEndCap(uint32_t subDetId);
  bool isPixel(uint32_t subDetId);
  bool isDetOrDetUnit(align::StructureType type);

  TH1* bookTH1F(bool isTransient, DirectoryWrapper& tfd, const char* histName, const char* histTitle, 
		int nBinsX, double lowX, double highX);

  TProfile* bookTProfile(bool isTransient, DirectoryWrapper& tfd, const char* histName, const char* histTitle, 
			 int nBinsX, double lowX, double highX);

  TProfile* bookTProfile(bool isTransient, DirectoryWrapper& tfd, const char* histName, const char* histTitle, 
			 int nBinsX, double lowX, double highX, double lowY, double highY);

  void getBinning(uint32_t subDetId, TrackerOfflineValidation::HistogrammType residualtype, 
		  int& nBinsX, double& lowerBoundX, double& upperBoundX);

  void summarizeBinInContainer(int bin, SummaryContainer& targetContainer, 
			       SummaryContainer& sourceContainer);

  void summarizeBinInContainer(int bin, uint32_t subDetId, SummaryContainer& targetContainer, 
			       ModuleHistos& sourceContainer);

  void setSummaryBin(int bin, TH1* targetHist, TH1* sourceHist);
    
  float Fwhm(const TH1* hist) const;
  std::pair<float,float> fitResiduals(TH1* hist) const; //, float meantmp, float rmstmp);
  float getMedian( const TH1* hist) const; 
  
  // From MillePedeAlignmentMonitor: Get Index for Arbitary vector<class> by name
  template <class OBJECT_TYPE> int GetIndex(const std::vector<OBJECT_TYPE*>& vec, const TString& name);
  
  
  // ---------- member data ---------------------------

  const edm::ParameterSet parSet_;
  edm::ESHandle<TrackerGeometry> tkGeom_;
  const TrackerGeometry *bareTkGeomPtr_; // ugly hack to book hists only once, but check 

  // parameters from cfg to steer
  const bool lCoorHistOn_;
  const bool moduleLevelHistsTransient_;
  const bool moduleLevelProfiles_;
  const bool stripYResiduals_;
  const bool useFwhm_;
  const bool useFit_;
  const bool useOverflowForRMS_;
  const bool dqmMode_;
  const std::string moduleDirectory_;
  
  // a vector to keep track which pointers should be deleted at the very end
  std::vector<TH1*> vDeleteObjects_;

  std::vector<TH1*> vTrackHistos_;
  std::vector<TH1*> vTrackProfiles_;
  std::vector<TH1*> vTrack2DHistos_;
  
  std::map<int,TrackerOfflineValidation::ModuleHistos> mPxbResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mPxeResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTibResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTidResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTobResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTecResiduals_;

  const edm::EventSetup* lastSetup_;
};


//
// constants, enums and typedefs
//

//
// static data member definitions
//

template <class OBJECT_TYPE>  
int TrackerOfflineValidation::GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name)
{
  int result = 0;
  for (typename std::vector<OBJECT_TYPE*>::const_iterator iter = vec.begin(), iterEnd = vec.end();
       iter != iterEnd; ++iter, ++result) {
    if (*iter && (*iter)->GetName() == name) return result;
  }
  edm::LogError("Alignment") << "@SUB=TrackerOfflineValidation::GetIndex" << " could not find " << name;
  return -1;
}


template <> TH1* TrackerOfflineValidation::DirectoryWrapper::make<TH1F>(const char* name,const char* title,int nBinX,double minBinX,double maxBinX){
  if(dqmMode){theDbe->setCurrentFolder(directoryString); return theDbe->book1D(name,title,nBinX,minBinX,maxBinX)->getTH1();}
  else{return tfd->make<TH1F>(name,title,nBinX,minBinX,maxBinX);}
}

template <> TH1* TrackerOfflineValidation::DirectoryWrapper::make<TProfile>(const char* name,const char* title,int nBinX,double *xBins){
  if(dqmMode){
    theDbe->setCurrentFolder(directoryString);
    //DQM profile requires y-bins for construction... using TProfile creator by hand...
    TProfile *tmpProfile=new TProfile(name,title,nBinX,xBins);
    tmpProfile->SetDirectory(0);
    return theDbe->bookProfile(name,tmpProfile)->getTH1();
  }
  else{return tfd->make<TProfile>(name,title,nBinX,xBins);}
}

template <> TH1* TrackerOfflineValidation::DirectoryWrapper::make<TProfile>(const char* name,const char* title,int nBinX,double minBinX,double maxBinX){
  if(dqmMode){
    theDbe->setCurrentFolder(directoryString);
    //DQM profile requires y-bins for construction... using TProfile creator by hand...
    TProfile *tmpProfile=new TProfile(name,title,nBinX,minBinX,maxBinX);
    tmpProfile->SetDirectory(0);
    return theDbe->bookProfile(name,tmpProfile)->getTH1();
  }
  else{return tfd->make<TProfile>(name,title,nBinX,minBinX,maxBinX);}
}

template <> TH1* TrackerOfflineValidation::DirectoryWrapper::make<TProfile>(const char* name ,const char* title,int nbinX,double minX ,double maxX,double minY,double maxY){
  if(dqmMode){
    theDbe->setCurrentFolder(directoryString);
    int dummy(0); // DQMProfile wants Y channels... does not use them!
    return (theDbe->bookProfile(name,title,nbinX,minX,maxX,dummy,minY,maxY)->getTH1());
  }
  else{
    return tfd->make<TProfile>(name,title,nbinX,minX,maxX,minY,maxY);
  }
}

template <> TH1* TrackerOfflineValidation::DirectoryWrapper::make<TH2F>(const char* name,const char* title,int nBinX,double minBinX,double maxBinX,int nBinY,double minBinY,double maxBinY){
  if(dqmMode){theDbe->setCurrentFolder(directoryString); return theDbe->book2D(name,title,nBinX,minBinX,maxBinX,nBinY,minBinY,maxBinY)->getTH1();}
  else{return tfd->make<TH2F>(name,title,nBinX,minBinX,maxBinX,nBinY,minBinY,maxBinY);}
}


//
// constructors and destructor
//
TrackerOfflineValidation::TrackerOfflineValidation(const edm::ParameterSet& iConfig)
  : parSet_(iConfig), bareTkGeomPtr_(0), lCoorHistOn_(parSet_.getParameter<bool>("localCoorHistosOn")),
    moduleLevelHistsTransient_(parSet_.getParameter<bool>("moduleLevelHistsTransient")),
    moduleLevelProfiles_(parSet_.getParameter<bool>("moduleLevelProfiles")),
    stripYResiduals_(parSet_.getParameter<bool>("stripYResiduals")), 
    useFwhm_(parSet_.getParameter<bool>("useFwhm")),
    useFit_(parSet_.getParameter<bool>("useFit")),
    useOverflowForRMS_(parSet_.getParameter<bool>("useOverflowForRMS")),
    dqmMode_(parSet_.getParameter<bool>("useInDqmMode")),
    moduleDirectory_(parSet_.getParameter<std::string>("moduleDirectoryInOutput")),
    lastSetup_(nullptr)
{
}


TrackerOfflineValidation::~TrackerOfflineValidation()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  for( std::vector<TH1*>::const_iterator it = vDeleteObjects_.begin(), itEnd = vDeleteObjects_.end(); 
       it != itEnd;
       ++it) delete *it;
}


//
// member functions
//


// ------------ method called once each job just before starting event loop  ------------
void
TrackerOfflineValidation::checkBookHists(const edm::EventSetup& es)
{
  es.get<TrackerDigiGeometryRecord>().get( tkGeom_ );
  const TrackerGeometry *newBareTkGeomPtr = &(*tkGeom_);
  if (newBareTkGeomPtr == bareTkGeomPtr_) return; // already booked hists, nothing changed

  if (!bareTkGeomPtr_) { // pointer not yet set: called the first time => book hists

    //Retrieve tracker topology from geometry
    edm::ESHandle<TrackerTopology> tTopoHandle;
    es.get<IdealGeometryRecord>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();

    // construct alignable tracker to get access to alignable hierarchy 
    AlignableTracker aliTracker(&(*tkGeom_), tTopo);
    
    edm::LogInfo("TrackerOfflineValidation") << "There are " << newBareTkGeomPtr->detIds().size()
					     << " dets in the Geometry record.\n"
					     << "Out of these "<<newBareTkGeomPtr->detUnitIds().size()
					     <<" are detUnits";
    
    // Book Histogramms for global track quantities
    std::string globDir("GlobalTrackVariables");
    DirectoryWrapper trackglobal(globDir,moduleDirectory_,dqmMode_);
    this->bookGlobalHists(trackglobal);
    
    // recursively book histogramms on lowest level
    DirectoryWrapper tfdw("",moduleDirectory_,dqmMode_);
    this->bookDirHists(tfdw, aliTracker, tTopo);
  }
  else { // histograms booked, but changed TrackerGeometry?
    edm::LogWarning("GeometryChange") << "@SUB=checkBookHists"
				      << "TrackerGeometry changed, but will not re-book hists!";
  }
  bareTkGeomPtr_ = newBareTkGeomPtr;
}


void 
TrackerOfflineValidation::bookGlobalHists(DirectoryWrapper& tfd )
{

  vTrackHistos_.push_back(tfd.make<TH1F>("h_tracketa",
					 "Track #eta;#eta_{Track};Number of Tracks",
					 90,-3.,3.));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_trackphi",
					 "Track #phi;#phi_{Track};Number of Tracks",
					 90,-3.15,3.15));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_trackNumberOfValidHits",
					 "Track # of valid hits;# of valid hits _{Track};Number of Tracks",
					40,0.,40.));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_trackNumberOfLostHits",
					 "Track # of lost hits;# of lost hits _{Track};Number of Tracks",
					10,0.,10.));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_curvature",
					 "Curvature #kappa;#kappa_{Track};Number of Tracks",
					 100,-.05,.05));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_curvature_pos",
					 "Curvature |#kappa| Positive Tracks;|#kappa_{pos Track}|;Number of Tracks",
					 100,.0,.05));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_curvature_neg",
					 "Curvature |#kappa| Negative Tracks;|#kappa_{neg Track}|;Number of Tracks",
					 100,.0,.05));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_diff_curvature",
					 "Curvature |#kappa| Tracks Difference;|#kappa_{Track}|;# Pos Tracks - # Neg Tracks",
					 100,.0,.05));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_chi2",
					 "#chi^{2};#chi^{2}_{Track};Number of Tracks",
					 500,-0.01,500.));
  vTrackHistos_.push_back(tfd.make<TH1F>("h_chi2Prob",
					 "#chi^{2} probability;#chi^{2}prob_{Track};Number of Tracks",
					100,0.0,1.));	       
  vTrackHistos_.push_back(tfd.make<TH1F>("h_normchi2",
					 "#chi^{2}/ndof;#chi^{2}/ndof;Number of Tracks",
					 100,-0.01,10.));     
  vTrackHistos_.push_back(tfd.make<TH1F>("h_pt",
					 "p_{T}^{track};p_{T}^{track} [GeV];Number of Tracks",
					 250,0.,250));           
  vTrackHistos_.push_back(tfd.make<TH1F>("h_ptResolution",
					 "#delta{p_{T}/p_{T}^{track}};#delta_{p_{T}/p_{T}^{track}};Number of Tracks",
					 100,0.,0.5));           

  vTrackProfiles_.push_back(tfd.make<TProfile>("p_d0_vs_phi",
					       "Transverse Impact Parameter vs. #phi;#phi_{Track};#LT d_{0} #GT [cm]",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_dz_vs_phi",
					       "Longitudinal Impact Parameter vs. #phi;#phi_{Track};#LT d_{z} #GT [cm]",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_d0_vs_eta",
					       "Transverse Impact Parameter vs. #eta;#eta_{Track};#LT d_{0} #GT [cm]",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_dz_vs_eta",
					       "Longitudinal Impact Parameter vs. #eta;#eta_{Track};#LT d_{z} #GT [cm]",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_chi2_vs_phi",
					       "#chi^{2} vs. #phi;#phi_{Track};#LT #chi^{2} #GT",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_chi2Prob_vs_phi",
					       "#chi^{2} probablility vs. #phi;#phi_{Track};#LT #chi^{2} probability#GT",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_chi2Prob_vs_d0",
                                               "#chi^{2} probablility vs. |d_{0}|;|d_{0}|[cm];#LT #chi^{2} probability#GT",
                                               100,0,80));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_normchi2_vs_phi",
					       "#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_chi2_vs_eta",
					       "#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT",
					       100,-3.15,3.15));
  //variable binning for chi2/ndof vs. pT
  double xBins[19]={0.,0.15,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,7.,10.,15.,25.,40.,100.,200.};
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_normchi2_vs_pt",
					       "norm #chi^{2} vs. p_{T}_{Track}; p_{T}_{Track};#LT #chi^{2}/ndof #GT",
					       18,xBins));

  vTrackProfiles_.push_back(tfd.make<TProfile>("p_normchi2_vs_p",
					       "#chi^{2}/ndof vs. p_{Track};p_{Track};#LT #chi^{2}/ndof #GT",
					        18,xBins));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_chi2Prob_vs_eta",
					       "#chi^{2} probability vs. #eta;#eta_{Track};#LT #chi^{2} probability #GT",
					       100,-3.15,3.15));   
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_normchi2_vs_eta",
					       "#chi^{2}/ndof vs. #eta;#eta_{Track};#LT #chi^{2}/ndof #GT",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_kappa_vs_phi",
					       "#kappa vs. #phi;#phi_{Track};#kappa",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_kappa_vs_eta",
					       "#kappa vs. #eta;#eta_{Track};#kappa",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_ptResolution_vs_phi",
					       "#delta_{p_{T}}/p_{T}^{track};#phi^{track};#delta_{p_{T}}/p_{T}^{track}",
					       100, -3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_ptResolution_vs_eta",
					       "#delta_{p_{T}}/p_{T}^{track};#eta^{track};#delta_{p_{T}}/p_{T}^{track}",
					       100, -3.15,3.15));

  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_d0_vs_phi",
					   "Transverse Impact Parameter vs. #phi;#phi_{Track};d_{0} [cm]",
					   100, -3.15, 3.15, 100,-1.,1.) );
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_dz_vs_phi",
					   "Longitudinal Impact Parameter vs. #phi;#phi_{Track};d_{z} [cm]",
					   100, -3.15, 3.15, 100,-100.,100.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_d0_vs_eta",
					   "Transverse Impact Parameter vs. #eta;#eta_{Track};d_{0} [cm]",
					   100, -3.15, 3.15, 100,-1.,1.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_dz_vs_eta",
					   "Longitudinal Impact Parameter vs. #eta;#eta_{Track};d_{z} [cm]",
					   100, -3.15, 3.15, 100,-100.,100.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_chi2_vs_phi",
					   "#chi^{2} vs. #phi;#phi_{Track};#chi^{2}",
					   100, -3.15, 3.15, 500, 0., 500.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_chi2Prob_vs_phi",
					   "#chi^{2} probability vs. #phi;#phi_{Track};#chi^{2} probability",
					   100, -3.15, 3.15, 100, 0., 1.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_chi2Prob_vs_d0",
                                           "#chi^{2} probability vs. |d_{0}|;|d_{0}| [cm];#chi^{2} probability",
					   100, 0, 80, 100, 0., 1.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_normchi2_vs_phi",
					   "#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof",
					   100, -3.15, 3.15, 100, 0., 10.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_chi2_vs_eta",
					   "#chi^{2} vs. #eta;#eta_{Track};#chi^{2}",
					   100, -3.15, 3.15, 500, 0., 500.));  
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_chi2Prob_vs_eta",
					   "#chi^{2} probaility vs. #eta;#eta_{Track};#chi^{2} probability",
					   100, -3.15, 3.15, 100, 0., 1.));  
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_normchi2_vs_eta",
					   "#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof",
					   100,-3.15,3.15, 100, 0., 10.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_kappa_vs_phi",
					   "#kappa vs. #phi;#phi_{Track};#kappa",
					   100,-3.15,3.15, 100, .0,.05));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_kappa_vs_eta",
					   "#kappa vs. #eta;#eta_{Track};#kappa",
					   100,-3.15,3.15, 100, .0,.05));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_normchi2_vs_kappa",
					   "#kappa vs. #chi^{2}/ndof;#chi^{2}/ndof;#kappa",
					   100,0.,10, 100,-.03,.03));

  /****************** Definition of 2-D Histos of ResX vs momenta ****************************/
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resXprime_pixB",
					   "#momentum vs. #resX in pixB;#momentum;#resX",
					   15,0.,15., 200, -0.1,0.1));			   
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resXprime_pixE",
					   "#momentum vs. #resX in pixE;#momentum;#resX",
					   15,0.,15., 200, -0.1,0.1)); 
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resXprime_TIB",
					   "#momentum vs. #resX in TIB;#momentum;#resX",
					   15,0.,15., 200, -0.1,0.1)); 
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resXprime_TID",
					   "#momentum vs. #resX in TID;#momentum;#resX",
					   15,0.,15., 200, -0.1,0.1)); 
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resXprime_TOB",
					   "#momentum vs. #resX in TOB;#momentum;#resX",
					   15,0.,15., 200, -0.1,0.1));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resXprime_TEC",
					   "#momentum vs. #resX in TEC;#momentum;#resX",
					   15,0.,15., 200, -0.1,0.1)); 	

  /****************** Definition of 2-D Histos of ResY vs momenta ****************************/
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resYprime_pixB",
					   "#momentum vs. #resY in pixB;#momentum;#resY",
					   15,0.,15., 200, -0.1,0.1));			   
  vTrack2DHistos_.push_back(tfd.make<TH2F>("p_vs_resYprime_pixE",
					   "#momentum vs. #resY in pixE;#momentum;#resY",
					   15,0.,15., 200, -0.1,0.1)); 

}


void
TrackerOfflineValidation::bookDirHists(DirectoryWrapper& tfd, const Alignable& ali, const TrackerTopology* tTopo)
{
  std::vector<Alignable*> alivec(ali.components());
  for(int i=0, iEnd = ali.components().size();i < iEnd; ++i) {
    std::string structurename  = AlignableObjectId::idToString((alivec)[i]->alignableObjectId());
    LogDebug("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    dirname << structurename;
    // add no suffix counter to Strip and Pixel, just aesthetics
    if (structurename != "Strip" && structurename != "Pixel") dirname << "_" << i+1;

    if (structurename.find("Endcap",0) != std::string::npos ) {
      DirectoryWrapper f(tfd,dirname.str(),moduleDirectory_,dqmMode_);
      bookHists(f, *(alivec)[i], tTopo, ali.alignableObjectId() , i);
      bookDirHists( f, *(alivec)[i], tTopo);
    } else if( !(this->isDetOrDetUnit( (alivec)[i]->alignableObjectId()) )
	      || alivec[i]->components().size() > 1) {      
      DirectoryWrapper f(tfd,dirname.str(),moduleDirectory_,dqmMode_);
      bookHists(tfd, *(alivec)[i], tTopo, ali.alignableObjectId() , i);
      bookDirHists( f, *(alivec)[i], tTopo);
    } else {
      bookHists(tfd, *(alivec)[i], tTopo, ali.alignableObjectId() , i);
    }
  }
}


void 
TrackerOfflineValidation::bookHists(DirectoryWrapper& tfd, const Alignable& ali, const TrackerTopology* tTopo, align::StructureType type, int i)
{
  TrackerAlignableId aliid;
  const DetId id = ali.id();

  // comparing subdetandlayer to subdetIds gives a warning at compile time
  // -> subdetandlayer could also be pair<uint,uint> but this has to be adapted
  // in AlignableObjId 
  std::pair<int,int> subdetandlayer = aliid.typeAndLayerFromDetId(id, tTopo);

  align::StructureType subtype = align::invalid;
  
  // are we on or just above det, detunit level respectively?
  if (type == align::AlignableDetUnit )subtype = type;
  else if( this->isDetOrDetUnit(ali.alignableObjectId()) ) subtype = ali.alignableObjectId();
  
  // construct histogramm title and name
  std::stringstream histoname, histotitle, normhistoname, normhistotitle, 
    yhistoname, yhistotitle,
    xprimehistoname, xprimehistotitle, normxprimehistoname, normxprimehistotitle,
    yprimehistoname, yprimehistotitle, normyprimehistoname, normyprimehistotitle,
    localxname, localxtitle, localyname, localytitle,
    resxvsxprofilename, resxvsxprofiletitle, resyvsxprofilename, resyvsxprofiletitle,
    resxvsyprofilename, resxvsyprofiletitle, resyvsyprofilename, resyvsyprofiletitle; 
  
  std::string wheel_or_layer;

  if( this->isEndCap(static_cast<uint32_t>(subdetandlayer.first)) ) wheel_or_layer = "_wheel_";
  else if ( this->isBarrel(static_cast<uint32_t>(subdetandlayer.first)) ) wheel_or_layer = "_layer_";
  else edm::LogWarning("TrackerOfflineValidation") << "@SUB=TrackerOfflineValidation::bookHists" 
						   << "Unknown subdetid: " <<  subdetandlayer.first;     
  
  histoname << "h_residuals_subdet_" << subdetandlayer.first 
	    << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  yhistoname << "h_y_residuals_subdet_" << subdetandlayer.first 
	     << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  xprimehistoname << "h_xprime_residuals_subdet_" << subdetandlayer.first 
		  << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  yprimehistoname << "h_yprime_residuals_subdet_" << subdetandlayer.first 
		  << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  normhistoname << "h_normresiduals_subdet_" << subdetandlayer.first 
		<< wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  normxprimehistoname << "h_normxprimeresiduals_subdet_" << subdetandlayer.first 
		      << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  normyprimehistoname << "h_normyprimeresiduals_subdet_" << subdetandlayer.first 
		      << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
  histotitle << "X Residual for module " << id.rawId() << ";x_{tr} - x_{hit} [cm]";
  yhistotitle << "Y Residual for module " << id.rawId() << ";y_{tr} - y_{hit} [cm]";
  normhistotitle << "Normalized Residual for module " << id.rawId() << ";x_{tr} - x_{hit}/#sigma";
  xprimehistotitle << "X' Residual for module " << id.rawId() << ";(x_{tr} - x_{hit})' [cm]";
  normxprimehistotitle << "Normalized X' Residual for module " << id.rawId() << ";(x_{tr} - x_{hit})'/#sigma";
  yprimehistotitle << "Y' Residual for module " << id.rawId() << ";(y_{tr} - y_{hit})' [cm]";
  normyprimehistotitle << "Normalized Y' Residual for module " << id.rawId() << ";(y_{tr} - y_{hit})'/#sigma";
  
  if ( moduleLevelProfiles_ ) {
    localxname << "h_localx_subdet_" << subdetandlayer.first 
	       << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
    localyname << "h_localy_subdet_" << subdetandlayer.first 
	       << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
    localxtitle << "u local for module " << id.rawId() << "; u_{tr,r}";
    localytitle << "v local for module " << id.rawId() << "; v_{tr,r}";

    resxvsxprofilename << "p_residuals_x_vs_x_subdet_" << subdetandlayer.first 
		       << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
    resyvsxprofilename << "p_residuals_y_vs_x_subdet_" << subdetandlayer.first 
		       << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
    resxvsyprofilename << "p_residuals_x_vs_y_subdet_" << subdetandlayer.first 
		       << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
    resyvsyprofilename << "p_residuals_y_vs_y_subdet_" << subdetandlayer.first 
		       << wheel_or_layer << subdetandlayer.second << "_module_" << id.rawId();
    resxvsxprofiletitle << "U Residual vs u for module " << id.rawId() << "; u_{tr,r} ;(u_{tr} - u_{hit})/tan#alpha [cm]";
    resyvsxprofiletitle << "V Residual vs u for module " << id.rawId() << "; u_{tr,r} ;(v_{tr} - v_{hit})/tan#beta  [cm]";
    resxvsyprofiletitle << "U Residual vs v for module " << id.rawId() << "; v_{tr,r} ;(u_{tr} - u_{hit})/tan#alpha [cm]";
    resyvsyprofiletitle << "V Residual vs v for module " << id.rawId() << "; v_{tr,r} ;(v_{tr} - v_{hit})/tan#beta  [cm]";
  }
  
  if( this->isDetOrDetUnit( subtype ) ) {
    ModuleHistos &histStruct = this->getHistStructFromMap(id);
    int nbins = 0;
    double xmin = 0., xmax = 0.;
    double ymin = -0.1, ymax = 0.1;

    // do not allow transient hists in DQM mode
    bool moduleLevelHistsTransient(moduleLevelHistsTransient_);
    if (dqmMode_) moduleLevelHistsTransient = false;
    
    // decide via cfg if hists in local coordinates should be booked 
    if(lCoorHistOn_) {
      this->getBinning(id.subdetId(), XResidual, nbins, xmin, xmax);
      histStruct.ResHisto = this->bookTH1F(moduleLevelHistsTransient, tfd, 
					   histoname.str().c_str(),histotitle.str().c_str(),		     
					   nbins, xmin, xmax);
      this->getBinning(id.subdetId(), NormXResidual, nbins, xmin, xmax);
      histStruct.NormResHisto = this->bookTH1F(moduleLevelHistsTransient, tfd,
					       normhistoname.str().c_str(),normhistotitle.str().c_str(),
					       nbins, xmin, xmax);
    } 
    this->getBinning(id.subdetId(), XprimeResidual, nbins, xmin, xmax);
    histStruct.ResXprimeHisto = this->bookTH1F(moduleLevelHistsTransient, tfd, 
					       xprimehistoname.str().c_str(),xprimehistotitle.str().c_str(),
					       nbins, xmin, xmax);
    this->getBinning(id.subdetId(), NormXprimeResidual, nbins, xmin, xmax);
    histStruct.NormResXprimeHisto = this->bookTH1F(moduleLevelHistsTransient, tfd, 
						   normxprimehistoname.str().c_str(),normxprimehistotitle.str().c_str(),
						   nbins, xmin, xmax);

    if ( moduleLevelProfiles_ ) {
      this->getBinning(id.subdetId(), XResidualProfile, nbins, xmin, xmax);      

      histStruct.LocalX = this->bookTH1F(moduleLevelHistsTransient, tfd,
					 localxname.str().c_str(),localxtitle.str().c_str(),
					 nbins, xmin, xmax);
      histStruct.LocalY = this->bookTH1F(moduleLevelHistsTransient, tfd,
					 localyname.str().c_str(),localytitle.str().c_str(),
					 nbins, xmin, xmax);
      histStruct.ResXvsXProfile = this->bookTProfile(moduleLevelHistsTransient, tfd,
						     resxvsxprofilename.str().c_str(),resxvsxprofiletitle.str().c_str(),
						     nbins, xmin, xmax, ymin, ymax);
      histStruct.ResXvsXProfile->Sumw2(); // to be filled with weights, so uncertainties need sum of square of weights
      histStruct.ResXvsYProfile = this->bookTProfile(moduleLevelHistsTransient, tfd,
						     resxvsyprofilename.str().c_str(),resxvsyprofiletitle.str().c_str(),
						     nbins, xmin, xmax, ymin, ymax);
      histStruct.ResXvsYProfile->Sumw2(); // to be filled with weights, so uncertainties need sum of square of weights
    }

    if( this->isPixel(subdetandlayer.first) || stripYResiduals_ ) {
      this->getBinning(id.subdetId(), YprimeResidual, nbins, xmin, xmax);
      histStruct.ResYprimeHisto = this->bookTH1F(moduleLevelHistsTransient, tfd,
						 yprimehistoname.str().c_str(),yprimehistotitle.str().c_str(),
						 nbins, xmin, xmax);
      if (lCoorHistOn_) { // un-primed y-residual
	this->getBinning(id.subdetId(), YResidual, nbins, xmin, xmax);
	histStruct.ResYHisto = this->bookTH1F(moduleLevelHistsTransient, tfd, 
					      yhistoname.str().c_str(), yhistotitle.str().c_str(),
					      nbins, xmin, xmax);
      }
      this->getBinning(id.subdetId(), NormYprimeResidual, nbins, xmin, xmax);
      histStruct.NormResYprimeHisto = this->bookTH1F(moduleLevelHistsTransient, tfd, 
						     normyprimehistoname.str().c_str(),normyprimehistotitle.str().c_str(),
						     nbins, xmin, xmax);
      // Here we could add un-primed normalised y-residuals if(lCoorHistOn_)...
      if ( moduleLevelProfiles_ ) {
	this->getBinning(id.subdetId(), YResidualProfile, nbins, xmin, xmax);      
	
	histStruct.ResYvsXProfile = this->bookTProfile(moduleLevelHistsTransient, tfd,
						       resyvsxprofilename.str().c_str(),resyvsxprofiletitle.str().c_str(),
						       nbins, xmin, xmax, ymin, ymax);
	histStruct.ResYvsXProfile->Sumw2(); // to be filled with weights, so uncertainties need sum of square of weights
	histStruct.ResYvsYProfile = this->bookTProfile(moduleLevelHistsTransient, tfd,
						       resyvsyprofilename.str().c_str(),resyvsyprofiletitle.str().c_str(),
						       nbins, xmin, xmax, ymin, ymax);
	histStruct.ResYvsYProfile->Sumw2(); // to be filled with weights, so uncertainties need sum of square of weights
      }
    }
  }
}


TH1* TrackerOfflineValidation::bookTH1F(bool isTransient, DirectoryWrapper& tfd, const char* histName, const char* histTitle, 
					int nBinsX, double lowX, double highX)
{
  if (isTransient) {
    vDeleteObjects_.push_back(new TH1F(histName, histTitle, nBinsX, lowX, highX));
    return vDeleteObjects_.back(); // return last element of vector
  } 
  else
    return tfd.make<TH1F>(histName, histTitle, nBinsX, lowX, highX);
}

TProfile* TrackerOfflineValidation::bookTProfile(bool isTransient, DirectoryWrapper& tfd, const char* histName, const char* histTitle, 
						 int nBinsX, double lowX, double highX)
{
  if (isTransient) {
    TProfile * profile = new TProfile(histName, histTitle, nBinsX, lowX, highX);
    vDeleteObjects_.push_back(profile);
    return profile;
  }
  else
    return (TProfile*)tfd.make<TProfile>(histName, histTitle, nBinsX, lowX, highX);
}


TProfile* TrackerOfflineValidation::bookTProfile(bool isTransient, DirectoryWrapper& tfd, const char* histName, const char* histTitle, 
						 int nBinsX, double lowX, double highX, double lowY, double highY)
{
  if (isTransient) {
    TProfile * profile = new TProfile(histName, histTitle, nBinsX, lowX, highX, lowY, highY);
    vDeleteObjects_.push_back(profile);
    return profile;
  }
  else
    return (TProfile*)tfd.make<TProfile>(histName, histTitle, nBinsX, lowX, highX, lowY, highY);
}

bool TrackerOfflineValidation::isBarrel(uint32_t subDetId)
{
  return (subDetId == StripSubdetector::TIB ||
	  subDetId == StripSubdetector::TOB ||
	  subDetId == PixelSubdetector::PixelBarrel );
}


bool TrackerOfflineValidation::isEndCap(uint32_t subDetId)
{
  return ( subDetId == StripSubdetector::TID ||
	   subDetId == StripSubdetector::TEC ||
	   subDetId == PixelSubdetector::PixelEndcap );
}


bool TrackerOfflineValidation::isPixel(uint32_t subDetId)
{
  return (subDetId == PixelSubdetector::PixelBarrel ||
	  subDetId == PixelSubdetector::PixelEndcap );
}


bool TrackerOfflineValidation::isDetOrDetUnit(align::StructureType type)
{
  return ( type == align::AlignableDet ||
	   type == align::AlignableDetUnit );
}


void 
TrackerOfflineValidation::getBinning(uint32_t subDetId, 
				     TrackerOfflineValidation::HistogrammType residualType, 
				     int& nBinsX, double& lowerBoundX, double& upperBoundX)
{
  // determine if 
  const bool isPixel = this->isPixel(subDetId);
  
  edm::ParameterSet binningPSet;
  
  switch(residualType) 
    {
    case XResidual :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TH1XResPixelModules");                
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TH1XResStripModules");                
      break;
    case NormXResidual : 
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TH1NormXResPixelModules");             
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TH1NormXResStripModules");                
      break;
    case XprimeResidual :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TH1XprimeResPixelModules");                
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TH1XprimeResStripModules");                
      break;
    case NormXprimeResidual :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TH1NormXprimeResPixelModules");
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TH1NormXprimeResStripModules");
      break;
    case YResidual : // borrow y-residual binning from yprime
    case YprimeResidual :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TH1YResPixelModules");                
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TH1YResStripModules");                
      break; 
      /* case NormYResidual :*/
    case NormYprimeResidual :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TH1NormYResPixelModules");             
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TH1NormYResStripModules");  
      break;
    case XResidualProfile :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TProfileXResPixelModules");                
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TProfileXResStripModules");                
      break;
    case YResidualProfile :
      if(isPixel) binningPSet = parSet_.getParameter<edm::ParameterSet>("TProfileYResPixelModules");                
      else binningPSet        = parSet_.getParameter<edm::ParameterSet>("TProfileYResStripModules");                
      break;
    }
  nBinsX      = binningPSet.getParameter<int32_t>("Nbinx");		       
  lowerBoundX = binningPSet.getParameter<double>("xmin");		       
  upperBoundX = binningPSet.getParameter<double>("xmax");     
}


void 
TrackerOfflineValidation::setSummaryBin(int bin, TH1* targetHist, TH1* sourceHist)
{
  if(targetHist && sourceHist) {
    targetHist->SetBinContent(bin, sourceHist->GetMean(1));
    if(useFwhm_) targetHist->SetBinError(bin, Fwhm(sourceHist)/2.);
    else targetHist->SetBinError(bin, sourceHist->GetRMS(1) );
  }
  else return;
}


void 
TrackerOfflineValidation::summarizeBinInContainer( int bin, SummaryContainer& targetContainer, 
						   SummaryContainer& sourceContainer)
{
  this->setSummaryBin(bin, targetContainer.summaryXResiduals_, sourceContainer.sumXResiduals_);
  this->setSummaryBin(bin, targetContainer.summaryNormXResiduals_, sourceContainer.sumNormXResiduals_);
  // If no y-residual hists, just returns:
  this->setSummaryBin(bin, targetContainer.summaryYResiduals_, sourceContainer.sumYResiduals_);
  this->setSummaryBin(bin, targetContainer.summaryNormYResiduals_, sourceContainer.sumNormYResiduals_);
}


void 
TrackerOfflineValidation::summarizeBinInContainer( int bin, uint32_t subDetId, 
						   SummaryContainer& targetContainer, 
						   ModuleHistos& sourceContainer)
{
  // takes two summary Containers and sets summaryBins for all histograms
  this->setSummaryBin(bin, targetContainer.summaryXResiduals_, sourceContainer.ResXprimeHisto);
  this->setSummaryBin(bin, targetContainer.summaryNormXResiduals_, sourceContainer.NormResXprimeHisto);
  if( this->isPixel(subDetId) || stripYResiduals_ ) {
    this->setSummaryBin(bin, targetContainer.summaryYResiduals_, sourceContainer.ResYprimeHisto);
    this->setSummaryBin(bin, targetContainer.summaryNormYResiduals_, sourceContainer.NormResYprimeHisto);
  }
}


TrackerOfflineValidation::ModuleHistos& 
TrackerOfflineValidation::getHistStructFromMap(const DetId& detid)
{
  // get a struct with histograms from the respective map
  // if no object exist, the reference is automatically created by the map
  // throw exception if non-tracker id is passed
  uint subdetid = detid.subdetId();
  if(subdetid == PixelSubdetector::PixelBarrel ) {
    return mPxbResiduals_[detid.rawId()];
  } else if (subdetid == PixelSubdetector::PixelEndcap) {
    return mPxeResiduals_[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TIB) {
    return mTibResiduals_[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TID) {
    return mTidResiduals_[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TOB) {
    return mTobResiduals_[detid.rawId()];
  } else if(subdetid  == StripSubdetector::TEC) {
    return mTecResiduals_[detid.rawId()];
  } else {
    throw cms::Exception("Geometry Error")
      << "[TrackerOfflineValidation] Error, tried to get reference for non-tracker subdet " << subdetid 
      << " from detector " << detid.det();
    return mPxbResiduals_[0];
  }
}


// ------------ method called to for each event  ------------
void
TrackerOfflineValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if (useOverflowForRMS_)TH1::StatOverflows(kTRUE);
  this->checkBookHists(iSetup); // check whether hists are booked and do so if not yet done
  
  TrackerValidationVariables avalidator_(iSetup,parSet_);
    
  std::vector<TrackerValidationVariables::AVTrackStruct> vTrackstruct;
  avalidator_.fillTrackQuantities(iEvent, vTrackstruct);
  
  for (std::vector<TrackerValidationVariables::AVTrackStruct>::const_iterator itT = vTrackstruct.begin();	 
       itT != vTrackstruct.end();
       ++itT) {
    
    // Fill 1D track histos
    static const int etaindex = this->GetIndex(vTrackHistos_,"h_tracketa");
    vTrackHistos_[etaindex]->Fill(itT->eta);
    static const int phiindex = this->GetIndex(vTrackHistos_,"h_trackphi");
    vTrackHistos_[phiindex]->Fill(itT->phi);
    static const int numOfValidHitsindex = this->GetIndex(vTrackHistos_,"h_trackNumberOfValidHits");
    vTrackHistos_[numOfValidHitsindex]->Fill(itT->numberOfValidHits);
    static const int numOfLostHitsindex = this->GetIndex(vTrackHistos_,"h_trackNumberOfLostHits");
    vTrackHistos_[numOfLostHitsindex]->Fill(itT->numberOfLostHits);
    static const int kappaindex = this->GetIndex(vTrackHistos_,"h_curvature");
    vTrackHistos_[kappaindex]->Fill(itT->kappa);
    static const int kappaposindex = this->GetIndex(vTrackHistos_,"h_curvature_pos");
    if (itT->charge > 0)
      vTrackHistos_[kappaposindex]->Fill(fabs(itT->kappa));
    static const int kappanegindex = this->GetIndex(vTrackHistos_,"h_curvature_neg");
    if (itT->charge < 0)
      vTrackHistos_[kappanegindex]->Fill(fabs(itT->kappa));
    static const int normchi2index = this->GetIndex(vTrackHistos_,"h_normchi2");
    vTrackHistos_[normchi2index]->Fill(itT->normchi2);
    static const int chi2index = this->GetIndex(vTrackHistos_,"h_chi2");
    vTrackHistos_[chi2index]->Fill(itT->chi2);
    static const int chi2Probindex = this->GetIndex(vTrackHistos_,"h_chi2Prob");
    vTrackHistos_[chi2Probindex]->Fill(itT->chi2Prob);
    static const int ptindex = this->GetIndex(vTrackHistos_,"h_pt");
    vTrackHistos_[ptindex]->Fill(itT->pt);
    if (itT->ptError != 0.) {
      static const int ptResolutionindex = this->GetIndex(vTrackHistos_,"h_ptResolution");
      vTrackHistos_[ptResolutionindex]->Fill(itT->ptError/itT->pt);
    }
    // Fill track profiles
    static const int d0phiindex = this->GetIndex(vTrackProfiles_,"p_d0_vs_phi");
    vTrackProfiles_[d0phiindex]->Fill(itT->phi,itT->d0);
    static const int dzphiindex = this->GetIndex(vTrackProfiles_,"p_dz_vs_phi");
    vTrackProfiles_[dzphiindex]->Fill(itT->phi,itT->dz);
    static const int d0etaindex = this->GetIndex(vTrackProfiles_,"p_d0_vs_eta");
    vTrackProfiles_[d0etaindex]->Fill(itT->eta,itT->d0);
    static const int dzetaindex = this->GetIndex(vTrackProfiles_,"p_dz_vs_eta");
    vTrackProfiles_[dzetaindex]->Fill(itT->eta,itT->dz);
    static const int chiphiindex = this->GetIndex(vTrackProfiles_,"p_chi2_vs_phi");
    vTrackProfiles_[chiphiindex]->Fill(itT->phi,itT->chi2);
    static const int chiProbphiindex = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_phi");
    vTrackProfiles_[chiProbphiindex]->Fill(itT->phi,itT->chi2Prob);
    static const int chiProbabsd0index = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_d0");
    vTrackProfiles_[chiProbabsd0index]->Fill(fabs(itT->d0),itT->chi2Prob);
    static const int normchiphiindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_phi");
    vTrackProfiles_[normchiphiindex]->Fill(itT->phi,itT->normchi2);
    static const int chietaindex = this->GetIndex(vTrackProfiles_,"p_chi2_vs_eta");
    vTrackProfiles_[chietaindex]->Fill(itT->eta,itT->chi2);
    static const int normchiptindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_pt");
    vTrackProfiles_[normchiptindex]->Fill(itT->pt,itT->normchi2);
    static const int normchipindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_p");
    vTrackProfiles_[normchipindex]->Fill(itT->p,itT->normchi2);
    static const int chiProbetaindex = this->GetIndex(vTrackProfiles_,"p_chi2Prob_vs_eta");
    vTrackProfiles_[chiProbetaindex]->Fill(itT->eta,itT->chi2Prob);
    static const int normchietaindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_eta");
    vTrackProfiles_[normchietaindex]->Fill(itT->eta,itT->normchi2);
    static const int kappaphiindex = this->GetIndex(vTrackProfiles_,"p_kappa_vs_phi");
    vTrackProfiles_[kappaphiindex]->Fill(itT->phi,itT->kappa);
    static const int kappaetaindex = this->GetIndex(vTrackProfiles_,"p_kappa_vs_eta");
    vTrackProfiles_[kappaetaindex]->Fill(itT->eta,itT->kappa);
    static const int ptResphiindex = this->GetIndex(vTrackProfiles_,"p_ptResolution_vs_phi");
    vTrackProfiles_[ptResphiindex]->Fill(itT->phi,itT->ptError/itT->pt);
    static const int ptResetaindex = this->GetIndex(vTrackProfiles_,"p_ptResolution_vs_eta");
    vTrackProfiles_[ptResetaindex]->Fill(itT->eta,itT->ptError/itT->pt);

    // Fill 2D track histos
    static const int d0phiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_d0_vs_phi");
    vTrack2DHistos_[d0phiindex_2d]->Fill(itT->phi,itT->d0);
    static const int dzphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_dz_vs_phi");
    vTrack2DHistos_[dzphiindex_2d]->Fill(itT->phi,itT->dz);
    static const int d0etaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_d0_vs_eta");
    vTrack2DHistos_[d0etaindex_2d]->Fill(itT->eta,itT->d0);
    static const int dzetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_dz_vs_eta");
    vTrack2DHistos_[dzetaindex_2d]->Fill(itT->eta,itT->dz);
    static const int chiphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2_vs_phi");
    vTrack2DHistos_[chiphiindex_2d]->Fill(itT->phi,itT->chi2);
    static const int chiProbphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2Prob_vs_phi");
    vTrack2DHistos_[chiProbphiindex_2d]->Fill(itT->phi,itT->chi2Prob);
    static const int chiProbabsd0index_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2Prob_vs_d0");
    vTrack2DHistos_[chiProbabsd0index_2d]->Fill(fabs(itT->d0),itT->chi2Prob);
    static const int normchiphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_phi");
    vTrack2DHistos_[normchiphiindex_2d]->Fill(itT->phi,itT->normchi2);
    static const int chietaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2_vs_eta");
    vTrack2DHistos_[chietaindex_2d]->Fill(itT->eta,itT->chi2);
    static const int chiProbetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2Prob_vs_eta");
    vTrack2DHistos_[chiProbetaindex_2d]->Fill(itT->eta,itT->chi2Prob);
    static const int normchietaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_eta");
    vTrack2DHistos_[normchietaindex_2d]->Fill(itT->eta,itT->normchi2);
    static const int kappaphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_kappa_vs_phi");
    vTrack2DHistos_[kappaphiindex_2d]->Fill(itT->phi,itT->kappa);
    static const int kappaetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_kappa_vs_eta");
    vTrack2DHistos_[kappaetaindex_2d]->Fill(itT->eta,itT->kappa);
    static const int normchi2kappa_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_kappa");
    vTrack2DHistos_[normchi2kappa_2d]->Fill(itT->normchi2,itT->kappa);

    // hit quantities: residuals, normalized residuals
    for (std::vector<TrackerValidationVariables::AVHitStruct>::const_iterator itH = itT->hits.begin();
	 itH != itT->hits.end();
	 ++itH) {
      
      DetId detid(itH->rawDetId);
      ModuleHistos &histStruct = this->getHistStructFromMap(detid);

      // fill histos in local coordinates if set in cf
      if (lCoorHistOn_) {
	histStruct.ResHisto->Fill(itH->resX);
	if(itH->resErrX != 0) histStruct.NormResHisto->Fill(itH->resX/itH->resErrX);
	if (this->isPixel(detid.subdetId()) || stripYResiduals_ ) {
	  histStruct.ResYHisto->Fill(itH->resY);
	  // here add un-primed normalised y-residuals if wanted
	}
      }
      if (itH->resXprime != -999.) {
	histStruct.ResXprimeHisto->Fill(itH->resXprime);

	/******************************* Fill 2-D histo ResX vs momenta *****************************/
        if (detid.subdetId() == PixelSubdetector::PixelBarrel) { 
          static const int resXvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resXprime_pixB");
          vTrack2DHistos_[resXvsPindex_2d]->Fill(itT->p,itH->resXprime);
        }	
        if (detid.subdetId() == PixelSubdetector::PixelEndcap) { 
          static const int resXvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resXprime_pixE");
          vTrack2DHistos_[resXvsPindex_2d]->Fill(itT->p,itH->resXprime);
        }
        if (detid.subdetId() == StripSubdetector::TIB) { 
	  static const int resXvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resXprime_TIB");
          vTrack2DHistos_[resXvsPindex_2d]->Fill(itT->p,itH->resXprime);
        }
	if (detid.subdetId() == StripSubdetector::TID) { 
          static const int resXvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resXprime_TID");
          vTrack2DHistos_[resXvsPindex_2d]->Fill(itT->p,itH->resXprime);
        }
        if (detid.subdetId() == StripSubdetector::TOB) { 
	  static const int resXvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resXprime_TOB");
          vTrack2DHistos_[resXvsPindex_2d]->Fill(itT->p,itH->resXprime);
        }
        if (detid.subdetId() == StripSubdetector::TEC) {   
          static const int resXvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resXprime_TEC");
          vTrack2DHistos_[resXvsPindex_2d]->Fill(itT->p,itH->resXprime);
        }
	/******************************************/

	if ( moduleLevelProfiles_ && itH->inside ) {
	  float tgalpha = tan(itH->localAlpha);
	  if ( fabs(tgalpha)!=0 ){
	    histStruct.LocalX->Fill(itH->localXnorm, tgalpha*tgalpha); 
	    histStruct.LocalY->Fill(itH->localYnorm, tgalpha*tgalpha); 
/*           if (this->isEndCap(detid.subdetId()) && !this->isPixel(detid.subdetId())) {
             if((itH->resX)*(itH->resXprime)>0){
            histStruct.ResXvsXProfile->Fill(itH->localXnorm, itH->resXatTrkY/tgalpha, tgalpha*tgalpha);
            histStruct.ResXvsYProfile->Fill(itH->localYnorm, itH->resXatTrkY/tgalpha, tgalpha*tgalpha);
             } else {
            histStruct.ResXvsXProfile->Fill(itH->localXnorm, (-1)*itH->resXatTrkY/tgalpha, tgalpha*tgalpha);
            histStruct.ResXvsYProfile->Fill(itH->localYnorm, (-1)*itH->resXatTrkY/tgalpha, tgalpha*tgalpha);
               }

          }else {  
*/
	    histStruct.ResXvsXProfile->Fill(itH->localXnorm, itH->resXatTrkY/tgalpha, tgalpha*tgalpha); 
	    histStruct.ResXvsYProfile->Fill(itH->localYnorm, itH->resXatTrkY/tgalpha, tgalpha*tgalpha); 

//          }

	  }
	}

	if(itH->resXprimeErr != 0 && itH->resXprimeErr != -999 ) {	
	  histStruct.NormResXprimeHisto->Fill(itH->resXprime/itH->resXprimeErr);
	}
      }     
      
      if (itH->resYprime != -999.) {
	if (this->isPixel(detid.subdetId()) || stripYResiduals_ ) {
	  histStruct.ResYprimeHisto->Fill(itH->resYprime);

	  /******************************* Fill 2-D histo ResY vs momenta *****************************/
	  if (detid.subdetId() == PixelSubdetector::PixelBarrel) { 
	    static const int resYvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resYprime_pixB");
	    vTrack2DHistos_[resYvsPindex_2d]->Fill(itT->p,itH->resYprime);
	  }	
	  if (detid.subdetId() == PixelSubdetector::PixelEndcap) { 
	    static const int resYvsPindex_2d = this->GetIndex(vTrack2DHistos_,"p_vs_resYprime_pixE");
	    vTrack2DHistos_[resYvsPindex_2d]->Fill(itT->p,itH->resYprime);
	  }
	  /******************************************/

	  if ( moduleLevelProfiles_ && itH->inside ) {
	    float tgbeta = tan(itH->localBeta);
	    if ( fabs(tgbeta)!=0 ){
/*           if (this->isEndCap(detid.subdetId()) && !this->isPixel(detid.subdetId())) {
            
            if((itH->resY)*(itH->resYprime)>0){
            histStruct.ResYvsXProfile->Fill(itH->localXnorm, itH->resYprime/tgbeta, tgbeta*tgbeta);
            histStruct.ResYvsYProfile->Fill(itH->localYnorm, itH->resYprime/tgbeta, tgbeta*tgbeta);
             } else {
            histStruct.ResYvsXProfile->Fill(itH->localXnorm, (-1)*itH->resYprime/tgbeta, tgbeta*tgbeta);
            histStruct.ResYvsYProfile->Fill(itH->localYnorm, (-1)*itH->resYprime/tgbeta, tgbeta*tgbeta);
               }

          }else {  
*/
	      histStruct.ResYvsXProfile->Fill(itH->localXnorm, itH->resY/tgbeta, tgbeta*tgbeta); 
	      histStruct.ResYvsYProfile->Fill(itH->localYnorm, itH->resY/tgbeta, tgbeta*tgbeta); 
//              }
	    }
	  }

	  if (itH->resYprimeErr != 0 && itH->resYprimeErr != -999. ) {	
	    histStruct.NormResYprimeHisto->Fill(itH->resYprime/itH->resYprimeErr);
	  } 
	}
      }
      
    } // finish loop over hit quantities
  } // finish loop over track quantities

  if (useOverflowForRMS_) TH1::StatOverflows(kFALSE);  
}


// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerOfflineValidation::endJob()
{

  if (!tkGeom_.product()) return;

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  lastSetup_->get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  AlignableTracker aliTracker(&(*tkGeom_), tTopo);
  
  static const int kappadiffindex = this->GetIndex(vTrackHistos_,"h_diff_curvature");
  vTrackHistos_[kappadiffindex]->Add(vTrackHistos_[this->GetIndex(vTrackHistos_,"h_curvature_neg")],
				     vTrackHistos_[this->GetIndex(vTrackHistos_,"h_curvature_pos")],-1,1);
 
  // Collate Information for Subdetectors
  // create summary histogramms recursively
  std::vector<TrackerOfflineValidation::SummaryContainer> vTrackerprofiles;
  DirectoryWrapper f("",moduleDirectory_,dqmMode_);
  this->collateSummaryHists(f,(aliTracker), 0, vTrackerprofiles);
  
  if (dqmMode_) return;
  // Should be excluded in dqmMode, since TTree is not usable
  // In dqmMode tree operations are are sourced out to the additional module TrackerOfflineValidationSummary
  
  edm::Service<TFileService> fs;
  TTree *tree = fs->make<TTree>("TkOffVal","TkOffVal");
 
  TkOffTreeVariables *treeMemPtr = new TkOffTreeVariables;
  // We create branches for all members of 'TkOffTreeVariables' (even if not needed).
  // This works because we have a dictionary for 'TkOffTreeVariables'
  // (see src/classes_def.xml and src/classes.h):
  tree->Branch("TkOffTreeVariables", &treeMemPtr); // address of pointer!
 
  this->fillTree(*tree, mPxbResiduals_, *treeMemPtr, *tkGeom_, tTopo);
  this->fillTree(*tree, mPxeResiduals_, *treeMemPtr, *tkGeom_, tTopo);
  this->fillTree(*tree, mTibResiduals_, *treeMemPtr, *tkGeom_, tTopo);
  this->fillTree(*tree, mTidResiduals_, *treeMemPtr, *tkGeom_, tTopo);
  this->fillTree(*tree, mTobResiduals_, *treeMemPtr, *tkGeom_, tTopo);
  this->fillTree(*tree, mTecResiduals_, *treeMemPtr, *tkGeom_, tTopo);

  delete treeMemPtr; treeMemPtr = 0;
}


void
TrackerOfflineValidation::collateSummaryHists( DirectoryWrapper& tfd, const Alignable& ali, int i, 
					       std::vector<TrackerOfflineValidation::SummaryContainer>& vLevelProfiles)
{
  std::vector<Alignable*> alivec(ali.components());
  if( this->isDetOrDetUnit((alivec)[0]->alignableObjectId()) ) return;
  
  for(int iComp=0, iCompEnd = ali.components().size();iComp < iCompEnd; ++iComp) {
    std::vector< TrackerOfflineValidation::SummaryContainer > vProfiles;        
    std::string structurename  = AlignableObjectId::idToString((alivec)[iComp]->alignableObjectId());
    
    LogDebug("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    dirname << structurename;
    
    // add no suffix counter to strip and pixel -> just aesthetics
    if (structurename != "Strip" && structurename != "Pixel") dirname << "_" << iComp+1;
    
    if(  !(this->isDetOrDetUnit( (alivec)[iComp]->alignableObjectId()) )
	 || (alivec)[0]->components().size() > 1 ) {
      DirectoryWrapper f(tfd,dirname.str(),moduleDirectory_,dqmMode_);
      this->collateSummaryHists( f, *(alivec)[iComp], i, vProfiles);
      vLevelProfiles.push_back(this->bookSummaryHists(tfd, *(alivec[iComp]), ali.alignableObjectId(), iComp+1));
      TH1 *hY = vLevelProfiles[iComp].sumYResiduals_;
      TH1 *hNormY = vLevelProfiles[iComp].sumNormYResiduals_;
      for(uint n = 0; n < vProfiles.size(); ++n) {
	this->summarizeBinInContainer(n+1, vLevelProfiles[iComp], vProfiles[n] );
	vLevelProfiles[iComp].sumXResiduals_->Add(vProfiles[n].sumXResiduals_);
	vLevelProfiles[iComp].sumNormXResiduals_->Add(vProfiles[n].sumNormXResiduals_);
	if (hY)     hY->Add(vProfiles[n].sumYResiduals_);         // only if existing
	if (hNormY) hNormY->Add(vProfiles[n].sumNormYResiduals_); // dito (pxl, stripYResiduals_)
      }
      if(dqmMode_)continue;  // No fits in dqmMode
      //add fit values to stat box
      this->fitResiduals(vLevelProfiles[iComp].sumXResiduals_);
      this->fitResiduals(vLevelProfiles[iComp].sumNormXResiduals_);
      if (hY)     this->fitResiduals(hY);     // only if existing (pixel or stripYResiduals_)
      if (hNormY) this->fitResiduals(hNormY); // dito
    } else {
      // nothing to be done for det or detunits
      continue;
    }
  }
}


TrackerOfflineValidation::SummaryContainer 
TrackerOfflineValidation::bookSummaryHists(DirectoryWrapper& tfd, const Alignable& ali, 
					   align::StructureType type, int i) 
{
  const uint aliSize = ali.components().size();
  const align::StructureType alitype = ali.alignableObjectId();
  const align::StructureType subtype = ali.components()[0]->alignableObjectId();
  const char *aliTypeName = AlignableObjectId::idToString(alitype); // lifetime of char* OK
  const char *aliSubtypeName = AlignableObjectId::idToString(subtype);
  const char *typeName = AlignableObjectId::idToString(type);

  const DetId aliDetId = ali.id(); 
  // y residuals only if pixel or specially requested for strip:
  const bool bookResidY = this->isPixel(aliDetId.subdetId()) || stripYResiduals_;

  SummaryContainer sumContainer;
  
  // Book summary hists with one bin per component, 
  // but special case for Det with two DetUnit that we want to summarize one level up 
  // (e.g. in TOBRods with 12 bins for 6 stereo and 6 rphi DetUnit.)
  //    component of ali is not Det or Det with just one components
  const uint subcompSize = ali.components()[0]->components().size();
  if (subtype != align::AlignableDet || subcompSize == 1) { // Det with 1 comp. should not exist anymore...
    const TString title(Form("Summary for substructures in %s %d;%s;",aliTypeName,i,aliSubtypeName));
    
    sumContainer.summaryXResiduals_ = tfd.make<TH1F>(Form("h_summaryX%s_%d",aliTypeName,i), 
						     title + "#LT #Delta x' #GT",
						     aliSize, 0.5, aliSize+0.5);
    sumContainer.summaryNormXResiduals_ = tfd.make<TH1F>(Form("h_summaryNormX%s_%d",aliTypeName,i), 
							 title + "#LT #Delta x'/#sigma #GT",
							 aliSize,0.5,aliSize+0.5);
    
    if (bookResidY) {
      sumContainer.summaryYResiduals_ = tfd.make<TH1F>(Form("h_summaryY%s_%d",aliTypeName,i), 
						       title + "#LT #Delta y' #GT",
						       aliSize, 0.5, aliSize+0.5);
      sumContainer.summaryNormYResiduals_ = tfd.make<TH1F>(Form("h_summaryNormY%s_%d",aliTypeName,i), 
							   title + "#LT #Delta y'/#sigma #GT",
							   aliSize,0.5,aliSize+0.5);
    }
    
  } else if (subtype == align::AlignableDet && subcompSize > 1) { // fixed: was aliSize before
    if (subcompSize != 2) { // strange... expect only 2 DetUnits in DS layers
      // this 2 is hardcoded factor 2 in binning below and also assumed later on
      edm::LogError("Alignment") << "@SUB=bookSummaryHists"
				 << "Det with " << subcompSize << " components";
    }
    // title contains x-title
    const TString title(Form("Summary for substructures in %s %d;%s;", aliTypeName, i,
			     AlignableObjectId::idToString(ali.components()[0]->components()[0]->alignableObjectId())));
    
    sumContainer.summaryXResiduals_ 
      = tfd.make<TH1F>(Form("h_summaryX%s_%d", aliTypeName, i), 
		       title + "#LT #Delta x' #GT", (2*aliSize), 0.5, 2*aliSize+0.5);
    sumContainer.summaryNormXResiduals_ 
      = tfd.make<TH1F>(Form("h_summaryNormX%s_%d", aliTypeName, i), 
		       title + "#LT #Delta x'/#sigma #GT", (2*aliSize), 0.5, 2*aliSize+0.5);

    if (bookResidY) {
      sumContainer.summaryYResiduals_ 
	= tfd.make<TH1F>(Form("h_summaryY%s_%d", aliTypeName, i), 
			 title + "#LT #Delta y' #GT", (2*aliSize), 0.5, 2*aliSize+0.5);
      sumContainer.summaryNormYResiduals_ 
	= tfd.make<TH1F>(Form("h_summaryNormY%s_%d", aliTypeName, i), 
			 title + "#LT #Delta y'/#sigma #GT", (2*aliSize), 0.5, 2*aliSize+0.5);
    }

  } else {
    edm::LogError("TrackerOfflineValidation") << "@SUB=TrackerOfflineValidation::bookSummaryHists" 
					      << "No summary histogramm for hierarchy level " 
					      << aliTypeName << " in subdet " << aliDetId.subdetId();
  }

  // Now book hists that just sum up the residual histograms from lower levels.
  // Axis title is copied from lowest level module of structure.
  // Should be safe that y-hists are only touched if non-null pointers...
  int nbins = 0;
  double xmin = 0., xmax = 0.;
  const TString sumTitle(Form("Residual for %s %d in %s;", aliTypeName, i, typeName));
  const ModuleHistos &xTitHists = this->getHistStructFromMap(aliDetId); // for x-axis titles
  this->getBinning(aliDetId.subdetId(), XprimeResidual, nbins, xmin, xmax);
  
  sumContainer.sumXResiduals_ = tfd.make<TH1F>(Form("h_Xprime_%s_%d", aliTypeName, i),
					       sumTitle + xTitHists.ResXprimeHisto->GetXaxis()->GetTitle(),
					       nbins, xmin, xmax);
  
  this->getBinning(aliDetId.subdetId(), NormXprimeResidual, nbins, xmin, xmax);
  sumContainer.sumNormXResiduals_ = tfd.make<TH1F>(Form("h_NormXprime_%s_%d",aliTypeName,i), 
						   sumTitle + xTitHists.NormResXprimeHisto->GetXaxis()->GetTitle(),
						   nbins, xmin, xmax);
  if (bookResidY) {
    this->getBinning(aliDetId.subdetId(), YprimeResidual, nbins, xmin, xmax);
    sumContainer.sumYResiduals_ = tfd.make<TH1F>(Form("h_Yprime_%s_%d",aliTypeName,i), 
						 sumTitle + xTitHists.ResYprimeHisto->GetXaxis()->GetTitle(),
						 nbins, xmin, xmax);
    
    this->getBinning(aliDetId.subdetId(), NormYprimeResidual, nbins, xmin, xmax);
    sumContainer.sumNormYResiduals_ = tfd.make<TH1F>(Form("h_NormYprime_%s_%d",aliTypeName,i), 
						     sumTitle + xTitHists.NormResYprimeHisto->GetXaxis()->GetTitle(),
						     nbins, xmin, xmax);
  }
  
  // If we are at the lowest level, we already sum up and fill the summary.

  // special case I: For DetUnits and Dets with only one subcomponent start filling summary histos
  if( ( subtype == align::AlignableDet && subcompSize == 1) || subtype  == align::AlignableDetUnit ) {
    for(uint k = 0; k < aliSize; ++k) {
      DetId detid = ali.components()[k]->id();
      ModuleHistos &histStruct = this->getHistStructFromMap(detid);
      this->summarizeBinInContainer(k+1, detid.subdetId() ,sumContainer, histStruct );
      sumContainer.sumXResiduals_->Add(histStruct.ResXprimeHisto);
      sumContainer.sumNormXResiduals_->Add(histStruct.NormResXprimeHisto);
      if( this->isPixel(detid.subdetId()) || stripYResiduals_ ) {
      	sumContainer.sumYResiduals_->Add(histStruct.ResYprimeHisto);
      	sumContainer.sumNormYResiduals_->Add(histStruct.NormResYprimeHisto);
      }
    }
  } else if( subtype == align::AlignableDet && subcompSize > 1) { // fixed: was aliSize before
    // special case II: Fill summary histos for dets with two detunits 
    for(uint k = 0; k < aliSize; ++k) {
      for(uint j = 0; j < subcompSize; ++j) { // assumes all have same size (as binning does)
	DetId detid = ali.components()[k]->components()[j]->id();
	ModuleHistos &histStruct = this->getHistStructFromMap(detid);	
	this->summarizeBinInContainer(2*k+j+1, detid.subdetId() ,sumContainer, histStruct );
	sumContainer.sumXResiduals_->Add( histStruct.ResXprimeHisto);
	sumContainer.sumNormXResiduals_->Add( histStruct.NormResXprimeHisto);
	if( this->isPixel(detid.subdetId()) || stripYResiduals_ ) {
	  sumContainer.sumYResiduals_->Add( histStruct.ResYprimeHisto);
	  sumContainer.sumNormYResiduals_->Add( histStruct.NormResYprimeHisto);
	}
      }
    }
  }
  return sumContainer;
}


float 
TrackerOfflineValidation::Fwhm (const TH1* hist) const
{
  float max = hist->GetMaximum();
  int left = -1, right = -1;
  for(unsigned int i = 1, iEnd = hist->GetNbinsX(); i <= iEnd; ++i) {
    if(hist->GetBinContent(i) < max/2. && hist->GetBinContent(i+1) > max/2. && left == -1) {
      if(max/2. - hist->GetBinContent(i) < hist->GetBinContent(i+1) - max/2.) {
	left = i;
	++i;
      } else {
	left = i+1;
	++i;
      }
    }
    if(left != -1 && right == -1) {
      if(hist->GetBinContent(i) > max/2. && hist->GetBinContent(i+1) < max/2.) {
	if( hist->GetBinContent(i) - max/2. < max/2. - hist->GetBinContent(i+1)) {
	  right = i;
	} else {
	  right = i+1;
	}
	
      }
    }
  }
  return hist->GetXaxis()->GetBinCenter(right) - hist->GetXaxis()->GetBinCenter(left);
}


void 
TrackerOfflineValidation::fillTree(TTree& tree,
				   const std::map<int, TrackerOfflineValidation::ModuleHistos>& moduleHist_,
				   TkOffTreeVariables &treeMem, const TrackerGeometry& tkgeom, const TrackerTopology* tTopo)
{
 
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator it = moduleHist_.begin(), 
	itEnd= moduleHist_.end(); it != itEnd;++it ) { 
    treeMem.clear(); // make empty/default
    
    //variables concerning the tracker components/hierarchy levels
    DetId detId_ = it->first;
    treeMem.moduleId = detId_;
    treeMem.subDetId = detId_.subdetId();
    treeMem.isDoubleSide =0;

    if(treeMem.subDetId == PixelSubdetector::PixelBarrel){
      unsigned int whichHalfBarrel(1), rawId(detId_.rawId());  //DetId does not know about halfBarrels is PXB ...
      if( (rawId>=302056964 && rawId<302059300) || (rawId>=302123268 && rawId<302127140) ||
	  (rawId>=302189572 && rawId<302194980) ) whichHalfBarrel=2;
      treeMem.layer = tTopo->pxbLayer(detId_); 
      treeMem.half = whichHalfBarrel;
      treeMem.rod = tTopo->pxbLadder(detId_);     // ... so, ladder is not per halfBarrel-Layer, but per barrel-layer!
      treeMem.module = tTopo->pxbModule(detId_);
    } else if(treeMem.subDetId == PixelSubdetector::PixelEndcap){
      unsigned int whichHalfCylinder(1), rawId(detId_.rawId());  //DetId does not kmow about halfCylinders in PXF
      if( (rawId>=352394500 && rawId<352406032) || (rawId>=352460036 && rawId<352471568) ||
	  (rawId>=344005892 && rawId<344017424) || (rawId>=344071428 && rawId<344082960) ) whichHalfCylinder=2;
      treeMem.layer = tTopo->pxfDisk(detId_); 
      treeMem.side = tTopo->pxfSide(detId_);
      treeMem.half = whichHalfCylinder;
      treeMem.blade = tTopo->pxfBlade(detId_); 
      treeMem.panel = tTopo->pxfPanel(detId_);
      treeMem.module = tTopo->pxfModule(detId_);
    } else if(treeMem.subDetId == StripSubdetector::TIB){
      unsigned int whichHalfShell(1), rawId(detId_.rawId());  //DetId does not kmow about halfShells in TIB
       if ( (rawId>=369120484 && rawId<369120688) || (rawId>=369121540 && rawId<369121776) ||
	    (rawId>=369136932 && rawId<369137200) || (rawId>=369137988 && rawId<369138288) ||
            (rawId>=369153396 && rawId<369153744) || (rawId>=369154436 && rawId<369154800) ||
	    (rawId>=369169844 && rawId<369170256) || (rawId>=369170900 && rawId<369171344) ||
	    (rawId>=369124580 && rawId<369124784) || (rawId>=369125636 && rawId<369125872) ||
	    (rawId>=369141028 && rawId<369141296) || (rawId>=369142084 && rawId<369142384) ||
	    (rawId>=369157492 && rawId<369157840) || (rawId>=369158532 && rawId<369158896) ||
	    (rawId>=369173940 && rawId<369174352) || (rawId>=369174996 && rawId<369175440) ) whichHalfShell=2;
      treeMem.layer = tTopo->tibLayer(detId_); 
      treeMem.side = tTopo->tibStringInfo(detId_)[0];
      treeMem.half = whichHalfShell;
      treeMem.rod = tTopo->tibStringInfo(detId_)[2]; 
      treeMem.outerInner = tTopo->tibStringInfo(detId_)[1]; 
      treeMem.module = tTopo->tibModule(detId_);
      treeMem.isStereo = tTopo->tibStereo(detId_);
      treeMem.isDoubleSide = tTopo->tibIsDoubleSide(detId_);
    } else if(treeMem.subDetId == StripSubdetector::TID){
      treeMem.layer = tTopo->tidWheel(detId_); 
      treeMem.side = tTopo->tidSide(detId_);
      treeMem.ring = tTopo->tidRing(detId_); 
      treeMem.outerInner = tTopo->tidModuleInfo(detId_)[0]; 
      treeMem.module = tTopo->tidModuleInfo(detId_)[1];
      treeMem.isStereo = tTopo->tidStereo(detId_);
      treeMem.isDoubleSide = tTopo->tidIsDoubleSide(detId_);
    } else if(treeMem.subDetId == StripSubdetector::TOB){
      treeMem.layer = tTopo->tobLayer(detId_); 
      treeMem.side = tTopo->tobRodInfo(detId_)[0];
      treeMem.rod = tTopo->tobRodInfo(detId_)[1]; 
      treeMem.module = tTopo->tobModule(detId_);
      treeMem.isStereo = tTopo->tobStereo(detId_);
      treeMem.isDoubleSide = tTopo->tobIsDoubleSide(detId_);
    } else if(treeMem.subDetId == StripSubdetector::TEC) {
      treeMem.layer = tTopo->tecWheel(detId_); 
      treeMem.side  = tTopo->tecSide(detId_);
      treeMem.ring  = tTopo->tecRing(detId_); 
      treeMem.petal = tTopo->tecPetalInfo(detId_)[1]; 
      treeMem.outerInner = tTopo->tecPetalInfo(detId_)[0];
      treeMem.module = tTopo->tecModule(detId_);
      treeMem.isStereo = tTopo->tecStereo(detId_);
      treeMem.isDoubleSide = tTopo->tecIsDoubleSide(detId_); 
    }
    
    //variables concerning the tracker geometry
    const Surface::PositionType &gPModule = tkgeom.idToDet(detId_)->position();
    treeMem.posPhi = gPModule.phi();
    treeMem.posEta = gPModule.eta();
    treeMem.posR   = gPModule.perp();
    treeMem.posX   = gPModule.x();
    treeMem.posY   = gPModule.y();
    treeMem.posZ   = gPModule.z();
 
    const Surface& surface =  tkgeom.idToDet(detId_)->surface();
    
    //global Orientation of local coordinate system of dets/detUnits   
    LocalPoint  lUDirection(1.,0.,0.), lVDirection(0.,1.,0.), lWDirection(0.,0.,1.);
    GlobalPoint gUDirection = surface.toGlobal(lUDirection),
                gVDirection = surface.toGlobal(lVDirection),
		gWDirection = surface.toGlobal(lWDirection);
    double dR(999.), dPhi(999.), dZ(999.);
    if(treeMem.subDetId==PixelSubdetector::PixelBarrel || treeMem.subDetId==StripSubdetector::TIB || treeMem.subDetId==StripSubdetector::TOB){
      dR = gWDirection.perp() - gPModule.perp();
      dPhi = deltaPhi(gUDirection.phi(),gPModule.phi());
      dZ = gVDirection.z() - gPModule.z();
      if(dZ>=0.)treeMem.rOrZDirection = 1; else treeMem.rOrZDirection = -1;
    }else if(treeMem.subDetId==PixelSubdetector::PixelEndcap){
      dR = gUDirection.perp() - gPModule.perp();
      dPhi = deltaPhi(gVDirection.phi(),gPModule.phi());
      dZ = gWDirection.z() - gPModule.z();
      if(dR>=0.)treeMem.rOrZDirection = 1; else treeMem.rOrZDirection = -1;
    }else if(treeMem.subDetId==StripSubdetector::TID || treeMem.subDetId==StripSubdetector::TEC){
      dR = gVDirection.perp() - gPModule.perp();
      dPhi = deltaPhi(gUDirection.phi(),gPModule.phi());
      dZ = gWDirection.z() - gPModule.z();
      if(dR>=0.)treeMem.rOrZDirection = 1; else treeMem.rOrZDirection = -1;
    }
    if(dR>=0.)treeMem.rDirection = 1; else treeMem.rDirection = -1;
    if(dPhi>=0.)treeMem.phiDirection = 1; else treeMem.phiDirection = -1;
    if(dZ>=0.)treeMem.zDirection = 1; else treeMem.zDirection = -1;
    
    //mean and RMS values (extracted from histograms(Xprime on module level)
    treeMem.entries = static_cast<UInt_t>(it->second.ResXprimeHisto->GetEntries());
    treeMem.meanX = it->second.ResXprimeHisto->GetMean();
    treeMem.rmsX  = it->second.ResXprimeHisto->GetRMS();
    //treeMem.sigmaX = Fwhm(it->second.ResXprimeHisto)/2.355;
    
    if (useFit_) {
      //call fit function which returns mean and sigma from the fit
      //for absolute residuals
      std::pair<float,float> fitResult1 = this->fitResiduals(it->second.ResXprimeHisto);
      treeMem.fitMeanX = fitResult1.first;
      treeMem.fitSigmaX = fitResult1.second;
      //for normalized residuals
      std::pair<float,float> fitResult2 = this->fitResiduals(it->second.NormResXprimeHisto);
      treeMem.fitMeanNormX = fitResult2.first;
      treeMem.fitSigmaNormX = fitResult2.second;
    }
    
    //get median for absolute residuals
    treeMem.medianX   = this->getMedian(it->second.ResXprimeHisto);

    int numberOfBins=it->second.ResXprimeHisto->GetNbinsX();
    treeMem.numberOfUnderflows = it->second.ResXprimeHisto->GetBinContent(0);
    treeMem.numberOfOverflows = it->second.ResXprimeHisto->GetBinContent(numberOfBins+1);
    treeMem.numberOfOutliers =  it->second.ResXprimeHisto->GetBinContent(0)+it->second.ResXprimeHisto->GetBinContent(numberOfBins+1);
    
    //mean and RMS values (extracted from histograms(normalized Xprime on module level)
    treeMem.meanNormX = it->second.NormResXprimeHisto->GetMean();
    treeMem.rmsNormX = it->second.NormResXprimeHisto->GetRMS();

    double stats[20];
    it->second.NormResXprimeHisto->GetStats(stats);
    // GF  treeMem.chi2PerDofX = stats[3]/(stats[0]-1);
    if (stats[0]) treeMem.chi2PerDofX = stats[3]/stats[0];
    
    treeMem.sigmaNormX = Fwhm(it->second.NormResXprimeHisto)/2.355;
    treeMem.histNameX = it->second.ResXprimeHisto->GetName();
    treeMem.histNameNormX = it->second.NormResXprimeHisto->GetName();
    
    // fill tree variables in local coordinates if set in cfg
    if(lCoorHistOn_) {
      treeMem.meanLocalX = it->second.ResHisto->GetMean();
      treeMem.rmsLocalX = it->second.ResHisto->GetRMS();
      treeMem.meanNormLocalX = it->second.NormResHisto->GetMean();
      treeMem.rmsNormLocalX = it->second.NormResHisto->GetRMS();

      treeMem.histNameLocalX = it->second.ResHisto->GetName();
      treeMem.histNameNormLocalX = it->second.NormResHisto->GetName();
      if (it->second.ResYHisto) treeMem.histNameLocalY = it->second.ResYHisto->GetName();
    }

    // mean and RMS values in local y (extracted from histograms(normalized Yprime on module level)
    // might exist in pixel only
    if (it->second.ResYprimeHisto) {//(stripYResiduals_)
      TH1 *h = it->second.ResYprimeHisto;
      treeMem.meanY = h->GetMean();
      treeMem.rmsY  = h->GetRMS();
      
      if (useFit_) { // fit function which returns mean and sigma from the fit
	std::pair<float,float> fitMeanSigma = this->fitResiduals(h);
	treeMem.fitMeanY  = fitMeanSigma.first;
	treeMem.fitSigmaY = fitMeanSigma.second;
      }
      
      //get median for absolute residuals
      treeMem.medianY   = this->getMedian(h);

      treeMem.histNameY = h->GetName();
    }
    if (it->second.NormResYprimeHisto) {
      TH1 *h = it->second.NormResYprimeHisto;
      treeMem.meanNormY = h->GetMean();
      treeMem.rmsNormY  = h->GetRMS();
      h->GetStats(stats); // stats buffer defined above
      if (stats[0]) treeMem.chi2PerDofY = stats[3]/stats[0];

      if (useFit_) { // fit function which returns mean and sigma from the fit
	std::pair<float,float> fitMeanSigma = this->fitResiduals(h);
	treeMem.fitMeanNormY  = fitMeanSigma.first;
	treeMem.fitSigmaNormY = fitMeanSigma.second;
      }
      treeMem.histNameNormY = h->GetName();
    }

    if (moduleLevelProfiles_) {
      if (it->second.ResXvsXProfile) {
	TH1 *h = it->second.ResXvsXProfile;
	treeMem.meanResXvsX = h->GetMean();
	treeMem.rmsResXvsX  = h->GetRMS();
	treeMem.profileNameResXvsX = h->GetName();
      } 
      if (it->second.ResXvsYProfile) {
	TH1 *h = it->second.ResXvsYProfile;
	treeMem.meanResXvsY = h->GetMean();
	treeMem.rmsResXvsY  = h->GetRMS();
	treeMem.profileNameResXvsY = h->GetName();
      } 
      if (it->second.ResYvsXProfile) {
	TH1 *h = it->second.ResYvsXProfile;
	treeMem.meanResYvsX = h->GetMean();
	treeMem.rmsResYvsX  = h->GetRMS();
	treeMem.profileNameResYvsX = h->GetName();
      } 
      if (it->second.ResYvsYProfile) {
	TH1 *h = it->second.ResYvsYProfile;
	treeMem.meanResYvsY = h->GetMean();
	treeMem.rmsResYvsY  = h->GetRMS();
	treeMem.profileNameResYvsY = h->GetName();
      }
    }

    tree.Fill();
  }
}


std::pair<float,float> 
TrackerOfflineValidation::fitResiduals(TH1* hist) const
{
  std::pair<float,float> fitResult(9999., 9999.);
  if (!hist || hist->GetEntries() < 20) return fitResult;

  float mean  = hist->GetMean();
  float sigma = hist->GetRMS();

  try { // for < CMSSW_2_2_0 since ROOT warnings from fit are converted to exceptions
    // Remove the try/catch for more recent CMSSW!
    // first fit: two RMS around mean
    TF1 func("tmp", "gaus", mean - 2.*sigma, mean + 2.*sigma); 
    if (0 == hist->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
      mean  = func.GetParameter(1);
      sigma = func.GetParameter(2);
      // second fit: three sigma of first fit around mean of first fit
      func.SetRange(mean - 3.*sigma, mean + 3.*sigma);
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
      if (0 == hist->Fit(&func, "Q0LR")) {
	if (hist->GetFunction(func.GetName())) { // Take care that it is later on drawn:
	  hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
	}
	fitResult.first = func.GetParameter(1);
	fitResult.second = func.GetParameter(2);
      }
    }
  } catch (cms::Exception const & e) {
    edm::LogWarning("Alignment") << "@SUB=TrackerOfflineValidation::fitResiduals"
				 << "Caught this exception during ROOT fit: "
				 << e.what();
  }
  return fitResult;
}


float 
TrackerOfflineValidation::getMedian(const TH1* histo) const
{
  float median = 999;
  int nbins = histo->GetNbinsX();

  //extract median from histogram
  double *x = new double[nbins];
  double *y = new double[nbins];
  for (int j = 0; j < nbins; j++) {
    x[j] = histo->GetBinCenter(j+1);
    y[j] = histo->GetBinContent(j+1);
  }
  median = TMath::Median(nbins, x, y);
  
  delete[] x; x = 0;
  delete [] y; y = 0;  

  return median;

}
//define this as a plug-in
DEFINE_FWK_MODULE(TrackerOfflineValidation);
