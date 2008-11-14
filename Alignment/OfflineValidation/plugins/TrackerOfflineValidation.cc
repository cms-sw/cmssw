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
// $Id: TrackerOfflineValidation.cc,v 1.17 2008/09/20 13:40:40 flucke Exp $
//
//


// system include files
#include <memory>
#include <map>
#include <sstream>
#include <math.h>
#include <utility>

// ROOT includes
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TStyle.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

//
// class decleration
//

class TrackerOfflineValidation : public edm::EDAnalyzer {
public:
  explicit TrackerOfflineValidation(const edm::ParameterSet&);
  ~TrackerOfflineValidation();
  
  enum HistogrammType { XResidual, NormXResidual, 
			XprimeResidual, NormXprimeResidual, 
			YprimeResidual, NormYprimeResidual};
  
private:

  
  struct ModuleHistos{
    ModuleHistos() :  ResHisto(), NormResHisto(), ResXprimeHisto(), NormResXprimeHisto(), 
		      ResYprimeHisto(), NormResYprimeHisto() {} 
    TH1* ResHisto;
    TH1* NormResHisto;
    TH1* ResXprimeHisto;
    TH1* NormResXprimeHisto;
    TH1* ResYprimeHisto;
    TH1* NormResYprimeHisto;
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
  
  struct TreeVariables{
    TreeVariables(): meanLocalX_(), meanNormLocalX_(), meanX_(), meanNormX_(),
		     meanY_(), meanNormY_(),chi2PerDof_(),
		     rmsLocalX_(), rmsNormLocalX_(), rmsX_(), rmsNormX_(), 
		     rmsY_(), rmsNormY_(), sigmaX_(),sigmaNormX_(),
		     fitMeanX_(),  fitSigmaX_(),fitMeanNormX_(),fitSigmaNormX_(),
		     posR_(), posPhi_(), posEta_(),
		     posX_(), posY_(), posZ_(),
		      numberOfUnderflows_(), numberOfOverflows_(),numberOfOutliers_(),
		     entries_(), moduleId_(), subDetId_(),
		     layer_(), side_(), rod_(),ring_(), 
		     petal_(),blade_(), panel_(), outerInner_(),
		     isDoubleSide_(),
		     histNameLocalX_(), histNameNormLocalX_(), histNameX_(), histNameNormX_(), 
                     histNameY_(), histNameNormY_() {} 
    void clear() { *this = TreeVariables(); }
    Float_t meanLocalX_, meanNormLocalX_, meanX_,meanNormX_,    //mean value read out from modul histograms
      meanY_,meanNormY_, chi2PerDof_,
      rmsLocalX_, rmsNormLocalX_, rmsX_, rmsNormX_,      //rms value read out from modul histograms
      rmsY_, rmsNormY_,sigmaX_,sigmaNormX_,
      fitMeanX_,  fitSigmaX_,fitMeanNormX_,fitSigmaNormX_,
      posR_, posPhi_, posEta_,                     //global coordiantes    
      posX_, posY_, posZ_,             //global coordiantes 
      numberOfUnderflows_, numberOfOverflows_,numberOfOutliers_;
    UInt_t  entries_, moduleId_, subDetId_,          //number of entries for each modul //modul Id = detId and subdetector Id
      layer_, side_, rod_, 
      ring_, petal_, 
      blade_, panel_, 
      outerInner_; //orientation of modules in TIB:1/2= int/ext string, TID:1/2=back/front ring, TEC 1/2=back/front petal 
    Bool_t isDoubleSide_;
    std::string histNameLocalX_, histNameNormLocalX_, histNameX_, histNameNormX_,
       histNameY_, histNameNormY_;    
  };


  // 
  // ------------- private member function -------------
  // 
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void checkBookHists(const edm::EventSetup &setup);



  void bookGlobalHists(TFileDirectory &tfd);
  void bookDirHists(TFileDirectory &tfd, const Alignable& ali, const AlignableObjectId &aliobjid);
  void bookHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, 
		 const AlignableObjectId &aliobjid);
 
  void collateSummaryHists( TFileDirectory &tfd, const Alignable& ali, int i, 
			    const AlignableObjectId &aliobjid, 
			    std::vector<TrackerOfflineValidation::SummaryContainer > &v_levelProfiles);
  
  void bookTree(TTree &tree, struct TrackerOfflineValidation::TreeVariables &treeMem);
  
  void fillTree(TTree &tree, const std::map<int, TrackerOfflineValidation::ModuleHistos> &moduleHist_, 
		struct TrackerOfflineValidation::TreeVariables &treeMem, const TrackerGeometry &tkgeom );
  
  TrackerOfflineValidation::SummaryContainer bookSummaryHists(TFileDirectory &tfd, 
							      const Alignable& ali, 
							      align::StructureType type, int i, 
							      const AlignableObjectId &aliobjid); 

  ModuleHistos& getHistStructFromMap(const DetId& detid); 

  bool isBarrel(uint32_t subDetId);
  bool isEndCap(uint32_t subDetId);
  bool isPixel(uint32_t subDetId);
  bool isDetOrDetUnit(align::StructureType type);

  TH1* bookTH1F(bool isTransient, TFileDirectory& tfd, const char* histName, const char* histTitle, 
		int nBinsX, double lowX, double highX);

  void getBinning(uint32_t subDetId, TrackerOfflineValidation::HistogrammType residualtype, 
		  int &nBinsX, double &lowerBoundX, double &upperBoundX);

  void summarizeBinInContainer(int bin, SummaryContainer &targetContainer, 
			       SummaryContainer &sourceContainer);

  void summarizeBinInContainer(int bin, uint32_t subDetId, SummaryContainer &targetContainer, 
			       ModuleHistos &sourceContainer);

  void setSummaryBin(int bin, TH1* targetHist, TH1* sourceHist);
    
  float Fwhm(const TH1* hist);
  std::pair<float,float> fitResiduals(const TH1 *hist,float meantmp,float rmstmp);
void fitSumResiduals(const TH1 *hist);
  // From MillePedeAlignmentMonitor: Get Index for Arbitary vector<class> by name
  template <class OBJECT_TYPE>  
  int GetIndex(const std::vector<OBJECT_TYPE*> &vec, const TString &name);

  // ---------- member data ---------------------------


  const edm::ParameterSet parset_;
  edm::ESHandle<TrackerGeometry> tkGeom_;
  const TrackerGeometry *bareTkGeomPtr_; // ugly hack to book hists only once, but check 

  
  // parameters from cfg to steer
  bool lCoorHistOn_;
  bool moduleLevelHistsTransient_;
  bool overlappOn_;
  bool stripYResiduals_;
  bool useFwhm_;
  bool useFit_;
  bool useOverflowForRMS_;
  std::map< std::pair<uint32_t, uint32_t >, TH1*> hOverlappResidual;

  // a vector to keep track which pointers should be deleted at the very end
  std::vector<TH1*> vDeleteObjects_;

  // 
  std::vector<TH1*> vTrackHistos_;
  std::vector<TProfile*> vTrackProfiles_;
  std::vector<TH2*> vTrack2DHistos_;
  
  std::map<int,TrackerOfflineValidation::ModuleHistos> mPxbResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mPxeResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTibResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTidResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTobResiduals_;
  std::map<int,TrackerOfflineValidation::ModuleHistos> mTecResiduals_;



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
//
// constructors and destructor
//
TrackerOfflineValidation::TrackerOfflineValidation(const edm::ParameterSet& iConfig)
  : parset_(iConfig), bareTkGeomPtr_(0), lCoorHistOn_(parset_.getParameter<bool>("localCoorHistosOn")),
    moduleLevelHistsTransient_(parset_.getParameter<bool>("moduleLevelHistsTransient")),
    overlappOn_(parset_.getParameter<bool>("overlappOn")), 
    stripYResiduals_(parset_.getParameter<bool>("stripYResiduals")), 
    useFwhm_(parset_.getParameter<bool>("useFwhm")),
    useFit_(parset_.getParameter<bool>("useFit")),
    useOverflowForRMS_(parset_.getParameter<bool>("useOverflowForRMS"))
  
{
   //now do what ever initialization is needed
}


TrackerOfflineValidation::~TrackerOfflineValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  for( std::vector<TH1*>::const_iterator it = vDeleteObjects_.begin(), itEnd = vDeleteObjects_.end(); 
       it != itEnd; ++it) delete *it;
    
}


//
// member functions
//


// ------------ method called once each job just before starting event loop  ------------
void
TrackerOfflineValidation::checkBookHists(const edm::EventSetup &es)
{
  es.get<TrackerDigiGeometryRecord>().get( tkGeom_ );
  const TrackerGeometry *newBareTkGeomPtr = &(*tkGeom_);
  if (newBareTkGeomPtr == bareTkGeomPtr_) return; // already booked hists, nothing changed

  if (!bareTkGeomPtr_) { // pointer not yet set: called the first time => book hists
    edm::Service<TFileService> fs;    
    AlignableObjectId aliobjid;
    
    // construct alignable tracker to get access to alignable hierarchy 
    AlignableTracker aliTracker(&(*tkGeom_));
    
    edm::LogInfo("TrackerOfflineValidation") << "There are " << newBareTkGeomPtr->detIds().size()
					     << " detUnits in the Geometry record";
    
    //
    // Book Histogramms for global track quantities
    TFileDirectory trackglobal = fs->mkdir("GlobalTrackVariables");  
    this->bookGlobalHists(trackglobal);
    
    // recursively book histogramms on lowest level
    this->bookDirHists(static_cast<TFileDirectory&>(*fs), aliTracker, aliobjid);  
  } else { // histograms booked, but changed TrackerGeometry?
    edm::LogWarning("GeometryChange") << "@SUB=checkBookHists"
				      << "TrackerGeometry changed, but will not re-book hists!";
  }

  bareTkGeomPtr_ = newBareTkGeomPtr;
}


void 
TrackerOfflineValidation::bookGlobalHists(TFileDirectory &tfd )
{

  vTrackHistos_.push_back(tfd.make<TH1F>("h_tracketa",
					 "Track #eta;#eta_{Track};Number of Tracks",
					 90,-3.,3.));
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
  vTrackHistos_.push_back(tfd.make<TH1F>("h_normchi2",
					 "#chi^{2}/ndof;#chi^{2}/ndof;Number of Tracks",
					 100,-0.01,10.));     
  vTrackHistos_.push_back(tfd.make<TH1F>("h_pt",
					 "p_{T}^{track};p_{T}^{track} [GeV];Number of Tracks",
					 100,0.,2500));           
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
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_normchi2_vs_phi",
					       "#chi^{2}/ndof vs. #phi;#phi_{Track};#LT #chi^{2}/ndof #GT",
					       100,-3.15,3.15));
  vTrackProfiles_.push_back(tfd.make<TProfile>("p_chi2_vs_eta",
					       "#chi^{2} vs. #eta;#eta_{Track};#LT #chi^{2} #GT",
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
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_normchi2_vs_phi",
					   "#chi^{2}/ndof vs. #phi;#phi_{Track};#chi^{2}/ndof",
					   100, -3.15, 3.15, 100, 0., 10.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_chi2_vs_eta",
					   "#chi^{2} vs. #eta;#eta_{Track};#chi^{2}",
					   100, -3.15, 3.15, 500, 0., 500.));  
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_normchi2_vs_eta",
					   "#chi^{2}/ndof vs. #eta;#eta_{Track};#chi^{2}/ndof",
					   100,-3.15,3.15, 100, 0., 10.));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_kappa_vs_phi",
					   "#kappa vs. #phi;#phi_{Track};#kappa",
					   100,-3.15,3.15, 100, .0,.05));
  vTrack2DHistos_.push_back(tfd.make<TH2F>("h2_kappa_vs_eta",
					   "#kappa vs. #eta;#eta_{Track};#kappa",
					   100,-3.15,3.15, 100, .0,.05));
 

}


void
TrackerOfflineValidation::bookDirHists( TFileDirectory &tfd, const Alignable& ali, const AlignableObjectId &aliobjid)
{
  std::vector<Alignable*> alivec(ali.components());
  for(int i=0, iEnd = ali.components().size();i < iEnd; ++i) {
    std::string structurename  = aliobjid.typeToName((alivec)[i]->alignableObjectId());
    LogDebug("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    
    // add no suffix counter to Strip and Pixel
    // just aesthetics
    if(structurename != "Strip" && structurename != "Pixel") {
      dirname << structurename << "_" << i+1;
    } else {
      dirname << structurename;
    }
    if (structurename.find("Endcap",0) != std::string::npos )   {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      bookHists(f, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
      bookDirHists( f, *(alivec)[i], aliobjid);
    } else if( !(this->isDetOrDetUnit( (alivec)[i]->alignableObjectId()) )
	      || alivec[i]->components().size() > 1) {      
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      bookHists(tfd, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
      bookDirHists( f, *(alivec)[i], aliobjid);
    } else {
      bookHists(tfd, *(alivec)[i], ali.alignableObjectId() , i, aliobjid);
    }
  }
}




void 
TrackerOfflineValidation::bookHists(TFileDirectory &tfd, const Alignable& ali, align::StructureType type, int i, const AlignableObjectId &aliobjid)
{

  TrackerAlignableId aliid;
  const DetId id = ali.id();

  // comparing subdetandlayer to subdetIds gives a warning at compile time
  // -> subdetandlayer could also be pair<uint,uint> but this has to be adapted
  // in AlignableObjId 
  std::pair<int,int> subdetandlayer = aliid.typeAndLayerFromDetId(id);

  align::StructureType subtype = align::invalid;
  
  // are we on or just above det, detunit level respectively?
  if (type == align::AlignableDetUnit )subtype = type;
  else if( this->isDetOrDetUnit(ali.alignableObjectId()) ) subtype = ali.alignableObjectId();
  
  // construct histogramm title and name
  std::stringstream histoname, histotitle, normhistoname, normhistotitle, 
    xprimehistoname, xprimehistotitle, normxprimehistoname, normxprimehistotitle,
    yprimehistoname, yprimehistotitle, normyprimehistoname, normyprimehistotitle;
  
  std::string wheel_or_layer;

  if( this->isEndCap(static_cast<uint32_t>(subdetandlayer.first)) ) wheel_or_layer = "_wheel_";
  else if ( this->isBarrel(static_cast<uint32_t>(subdetandlayer.first)) ) wheel_or_layer = "_layer_";
  else edm::LogWarning("TrackerOfflineValidation") << "@SUB=TrackerOfflineValidation::bookHists" 
						   << "Unknown subdetid: " <<  subdetandlayer.first;     
  
  
  histoname << "h_residuals_subdet_" << subdetandlayer.first 
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
  histotitle << "Residual for module " << id.rawId() << ";x_{pred} - x_{rec} [cm]";
  normhistotitle << "Normalized Residual for module " << id.rawId() << ";x_{pred} - x_{rec}/#sigma";
  xprimehistotitle << "X' Residual for module " << id.rawId() << ";x_{pred} - x_{rec} [cm]";
  normxprimehistotitle << "Normalized X' Residual for module " << id.rawId() << ";x_{pred} - x_{rec}/#sigma";
  yprimehistotitle << "Y' Residual for module " << id.rawId() << ";y_{pred} - y_{rec} [cm]";
  normyprimehistotitle << "Normalized Y' Residual for module " << id.rawId() << ";y_{pred} - y_{rec}/#sigma";
  
  
  if( this->isDetOrDetUnit( subtype ) ) {
    ModuleHistos &histStruct = this->getHistStructFromMap(id);
    int nbins = 0;
    double xmin = 0., xmax = 0.;

    // decide via cfg if hists in local coordinates should be booked 
    if(lCoorHistOn_) {
      this->getBinning(id.subdetId(), XResidual, nbins, xmin, xmax);
      histStruct.ResHisto = this->bookTH1F(moduleLevelHistsTransient_, tfd, 
					   histoname.str().c_str(),histotitle.str().c_str(),		     
					   nbins, xmin, xmax);
      this->getBinning(id.subdetId(), NormXResidual, nbins, xmin, xmax);
      histStruct.NormResHisto = this->bookTH1F(moduleLevelHistsTransient_, tfd,
					       normhistoname.str().c_str(),normhistotitle.str().c_str(),
					       nbins, xmin, xmax);
    } 
    this->getBinning(id.subdetId(), XprimeResidual, nbins, xmin, xmax);
    histStruct.ResXprimeHisto = this->bookTH1F(moduleLevelHistsTransient_, tfd, 
					       xprimehistoname.str().c_str(),xprimehistotitle.str().c_str(),
					       nbins, xmin, xmax);
    this->getBinning(id.subdetId(), NormXprimeResidual, nbins, xmin, xmax);
    histStruct.NormResXprimeHisto = this->bookTH1F(moduleLevelHistsTransient_, tfd, 
						   normxprimehistoname.str().c_str(),normxprimehistotitle.str().c_str(),
						   nbins, xmin, xmax);

    if( this->isPixel(subdetandlayer.first) || stripYResiduals_ ) {
      this->getBinning(id.subdetId(), YprimeResidual, nbins, xmin, xmax);
      histStruct.ResYprimeHisto = this->bookTH1F(moduleLevelHistsTransient_, tfd,
						 yprimehistoname.str().c_str(),yprimehistotitle.str().c_str(),
						 nbins, xmin, xmax);
      this->getBinning(id.subdetId(), NormYprimeResidual, nbins, xmin, xmax);
      histStruct.NormResYprimeHisto = this->bookTH1F(moduleLevelHistsTransient_, tfd, 
						     normyprimehistoname.str().c_str(),normyprimehistotitle.str().c_str(),
						     nbins, xmin, xmax);
    }

  }
  
}


TH1* TrackerOfflineValidation::bookTH1F(bool isTransient, TFileDirectory& tfd, const char* histName, const char* histTitle, 
		int nBinsX, double lowX, double highX)
{
  if(isTransient) {
    vDeleteObjects_.push_back(new TH1F(histName, histTitle, nBinsX, lowX, highX));
    return vDeleteObjects_.back(); // return last element of vector
  }
  else
    return tfd.make<TH1F>(histName, histTitle, nBinsX, lowX, highX);


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
	   subDetId == PixelSubdetector::PixelEndcap);
}

bool TrackerOfflineValidation::isPixel(uint32_t subDetId)
{
  return (subDetId == PixelSubdetector::PixelBarrel || subDetId == PixelSubdetector::PixelEndcap);
}


bool TrackerOfflineValidation::isDetOrDetUnit(align::StructureType type)
{
  return ( type == align::AlignableDet || type == align::AlignableDetUnit);
}

void 
TrackerOfflineValidation::getBinning(uint32_t subDetId, 
				     TrackerOfflineValidation::HistogrammType residualType, 
				     int &nBinsX, double &lowerBoundX, double &upperBoundX)
{
  // determine if 
  const bool isPixel = this->isPixel(subDetId);
  
  edm::ParameterSet binningPSet;
  
  switch(residualType) 
    {
    case XResidual :
      if(isPixel) binningPSet = parset_.getParameter<edm::ParameterSet>("TH1XResPixelModules");                
      else binningPSet        = parset_.getParameter<edm::ParameterSet>("TH1XResStripModules");                
      break;
    case NormXResidual : 
      if(isPixel) binningPSet = parset_.getParameter<edm::ParameterSet>("TH1NormXResPixelModules");             
      else binningPSet        = parset_.getParameter<edm::ParameterSet>("TH1NormXResStripModules");                
      break;
    case XprimeResidual :
      if(isPixel) binningPSet = parset_.getParameter<edm::ParameterSet>("TH1XprimeResPixelModules");                
      else binningPSet        = parset_.getParameter<edm::ParameterSet>("TH1XprimeResStripModules");                
      break;
    case NormXprimeResidual :
      if(isPixel) binningPSet = parset_.getParameter<edm::ParameterSet>("TH1NormXprimeResPixelModules");                
      else binningPSet        = parset_.getParameter<edm::ParameterSet>("TH1NormXprimeResStripModules");                
      break;
    case YprimeResidual :
      if(isPixel) binningPSet = parset_.getParameter<edm::ParameterSet>("TH1YResPixelModules");                
      else binningPSet        = parset_.getParameter<edm::ParameterSet>("TH1YResStripModules");                
      break; 
    case NormYprimeResidual :
      if(isPixel) binningPSet = parset_.getParameter<edm::ParameterSet>("TH1NormYResPixelModules");             
      else binningPSet        = parset_.getParameter<edm::ParameterSet>("TH1NormYResStripModules");  
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
  } else {
    return;
  }

}


void 
TrackerOfflineValidation::summarizeBinInContainer( int bin, SummaryContainer &targetContainer, 
						   SummaryContainer &sourceContainer)
{
  
  
  this->setSummaryBin(bin, targetContainer.summaryXResiduals_, sourceContainer.sumXResiduals_);
  this->setSummaryBin(bin, targetContainer.summaryNormXResiduals_, sourceContainer.sumNormXResiduals_);
  this->setSummaryBin(bin, targetContainer.summaryYResiduals_, sourceContainer.sumYResiduals_);
  this->setSummaryBin(bin, targetContainer.summaryNormYResiduals_, sourceContainer.sumNormYResiduals_);

}

void 
TrackerOfflineValidation::summarizeBinInContainer( int bin, uint32_t subDetId, 
						   SummaryContainer &targetContainer, 
						   ModuleHistos &sourceContainer)
{

  // takes two summary Containers and sets summaryBins for all histogramms
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

  // get a struct with histogramms from the respective map
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
  this->checkBookHists(iSetup); // check whether hists are are booked and do so if not yet done
  
  //using namespace edm;
  TrackerValidationVariables avalidator_(iSetup,parset_);
  edm::Service<TFileService> fs;
    
  std::vector<TrackerValidationVariables::AVTrackStruct> vTrackstruct;
  avalidator_.fillTrackQuantities(iEvent,vTrackstruct);
  std::vector<TrackerValidationVariables::AVHitStruct> v_hitstruct;
  avalidator_.fillHitQuantities(iEvent,v_hitstruct);
  
  
  for (std::vector<TrackerValidationVariables::AVTrackStruct>::const_iterator it = vTrackstruct.begin(),
	 itEnd = vTrackstruct.end(); it != itEnd; ++it) {
    
    // Fill 1D track histos
    static const int etaindex = this->GetIndex(vTrackHistos_,"h_tracketa");
    vTrackHistos_[etaindex]->Fill(it->eta);
    static const int kappaindex = this->GetIndex(vTrackHistos_,"h_curvature");
    vTrackHistos_[kappaindex]->Fill(it->kappa);
    static const int kappaposindex = this->GetIndex(vTrackHistos_,"h_curvature_pos");
    if(it->charge > 0)
      vTrackHistos_[kappaposindex]->Fill(fabs(it->kappa));
    static const int kappanegindex = this->GetIndex(vTrackHistos_,"h_curvature_neg");
    if(it->charge < 0)
      vTrackHistos_[kappanegindex]->Fill(fabs(it->kappa));
    static const int normchi2index = this->GetIndex(vTrackHistos_,"h_normchi2");
    vTrackHistos_[normchi2index]->Fill(it->normchi2);
    static const int chi2index = this->GetIndex(vTrackHistos_,"h_chi2");
    vTrackHistos_[chi2index]->Fill(it->chi2);
    static const int ptindex = this->GetIndex(vTrackHistos_,"h_pt");
    vTrackHistos_[ptindex]->Fill(it->pt);
    if(it->ptError != 0.) {
      static const int ptResolutionindex = this->GetIndex(vTrackHistos_,"h_ptResolution");
      vTrackHistos_[ptResolutionindex]->Fill(it->ptError/it->pt);
    }
    // Fill track profiles
    static const int d0phiindex = this->GetIndex(vTrackProfiles_,"p_d0_vs_phi");
    vTrackProfiles_[d0phiindex]->Fill(it->phi,it->d0);
    static const int dzphiindex = this->GetIndex(vTrackProfiles_,"p_dz_vs_phi");
    vTrackProfiles_[dzphiindex]->Fill(it->phi,it->dz);
    static const int d0etaindex = this->GetIndex(vTrackProfiles_,"p_d0_vs_eta");
    vTrackProfiles_[d0etaindex]->Fill(it->eta,it->d0);
    static const int dzetaindex = this->GetIndex(vTrackProfiles_,"p_dz_vs_eta");
    vTrackProfiles_[dzetaindex]->Fill(it->eta,it->dz);
    static const int chiphiindex = this->GetIndex(vTrackProfiles_,"p_chi2_vs_phi");
    vTrackProfiles_[chiphiindex]->Fill(it->phi,it->chi2);
    static const int normchiphiindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_phi");
    vTrackProfiles_[normchiphiindex]->Fill(it->phi,it->normchi2);
    static const int chietaindex = this->GetIndex(vTrackProfiles_,"p_chi2_vs_eta");
    vTrackProfiles_[chietaindex]->Fill(it->eta,it->chi2);
    static const int normchietaindex = this->GetIndex(vTrackProfiles_,"p_normchi2_vs_eta");
    vTrackProfiles_[normchietaindex]->Fill(it->eta,it->normchi2);
    static const int kappaphiindex = this->GetIndex(vTrackProfiles_,"p_kappa_vs_phi");
    vTrackProfiles_[kappaphiindex]->Fill(it->phi,it->kappa);
    static const int kappaetaindex = this->GetIndex(vTrackProfiles_,"p_kappa_vs_eta");
    vTrackProfiles_[kappaetaindex]->Fill(it->eta,it->kappa);
    static const int ptResphiindex = this->GetIndex(vTrackProfiles_,"p_ptResolution_vs_phi");
    vTrackProfiles_[ptResphiindex]->Fill(it->phi,it->ptError/it->pt);
    static const int ptResetaindex = this->GetIndex(vTrackProfiles_,"p_ptResolution_vs_eta");
    vTrackProfiles_[ptResetaindex]->Fill(it->eta,it->ptError/it->pt);

    // Fill 2D track histos
    static const int d0phiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_d0_vs_phi");
    vTrack2DHistos_[d0phiindex_2d]->Fill(it->phi,it->d0);
    static const int dzphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_dz_vs_phi");
    vTrack2DHistos_[dzphiindex_2d]->Fill(it->phi,it->dz);
    static const int d0etaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_d0_vs_eta");
    vTrack2DHistos_[d0etaindex_2d]->Fill(it->eta,it->d0);
    static const int dzetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_dz_vs_eta");
    vTrack2DHistos_[dzetaindex_2d]->Fill(it->eta,it->dz);
    static const int chiphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2_vs_phi");
    vTrack2DHistos_[chiphiindex_2d]->Fill(it->phi,it->chi2);
    static const int normchiphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_phi");
    vTrack2DHistos_[normchiphiindex_2d]->Fill(it->phi,it->normchi2);
    static const int chietaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_chi2_vs_eta");
    vTrack2DHistos_[chietaindex_2d]->Fill(it->eta,it->chi2);
    static const int normchietaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_normchi2_vs_eta");
    vTrack2DHistos_[normchietaindex_2d]->Fill(it->eta,it->normchi2);
    static const int kappaphiindex_2d = this->GetIndex(vTrack2DHistos_,"h2_kappa_vs_phi");
    vTrack2DHistos_[kappaphiindex_2d]->Fill(it->phi,it->kappa);
    static const int kappaetaindex_2d = this->GetIndex(vTrack2DHistos_,"h2_kappa_vs_eta");
    vTrack2DHistos_[kappaetaindex_2d]->Fill(it->eta,it->kappa);
     
  } // finish loop over track quantities


  // hit quantities: residuals, normalized residuals
  for (std::vector<TrackerValidationVariables::AVHitStruct>::const_iterator it = v_hitstruct.begin(),
  	 itEnd = v_hitstruct.end(); it != itEnd; ++it) {
    DetId detid(it->rawDetId);
    ModuleHistos &histStruct = this->getHistStructFromMap(detid);
    
    // fill histos in local coordinates if set in cf
    if(lCoorHistOn_) {
      histStruct.ResHisto->Fill(it->resX);
      if(it->resErrX != 0) histStruct.NormResHisto->Fill(it->resX/it->resErrX);
    }
    if(it->resXprime != -999.) {
      histStruct.ResXprimeHisto->Fill(it->resXprime);
      if(it->resXprimeErr != 0 && it->resXprimeErr != -999 ) {	
	histStruct.NormResXprimeHisto->Fill(it->resXprime/it->resXprimeErr);
      } 
    }
    if(it->resYprime != -999.) {
      if( this->isPixel(detid.subdetId())  || stripYResiduals_ ) {
	histStruct.ResYprimeHisto->Fill(it->resYprime);
	if(it->resYprimeErr != 0 && it->resYprimeErr != -999. ) {	
	  histStruct.NormResYprimeHisto->Fill(it->resYprime/it->resYprimeErr);
	} 
      }
    }

    
    if(overlappOn_) {
      std::pair<uint32_t,uint32_t> tmp_pair(std::make_pair(it->rawDetId, it->overlapres.first));
      if(it->overlapres.first != 0 ) {
	if( hOverlappResidual[tmp_pair] ) {
	  hOverlappResidual[tmp_pair]->Fill(it->overlapres.second);
	} else if( hOverlappResidual[std::make_pair( it->overlapres.first, it->rawDetId) ]) {
	  hOverlappResidual[std::make_pair( it->overlapres.first, it->rawDetId) ]->Fill(it->overlapres.second);
	} else {
	  TFileDirectory tfd = fs->mkdir("OverlappResiduals");
	  hOverlappResidual[tmp_pair] = tfd.make<TH1F>(Form("hOverlappResidual_%d_%d",tmp_pair.first,tmp_pair.second),
						       "Overlapp Residuals",100,-50,50);
	  hOverlappResidual[tmp_pair]->Fill(it->overlapres.second);
	}
      }
    } // end overlappOn

  }
  if (useOverflowForRMS_) TH1::StatOverflows(kFALSE);  
}



// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerOfflineValidation::endJob()
{
  AlignableTracker aliTracker(&(*tkGeom_));
  edm::Service<TFileService> fs;   
  AlignableObjectId aliobjid;

  TTree *tree = fs->make<TTree>("TkOffVal","TkOffVal");
  TreeVariables treeMem;
  this->bookTree(*tree, treeMem);

  this->fillTree(*tree, mPxbResiduals_ ,treeMem, *tkGeom_);
  this->fillTree(*tree, mPxeResiduals_ ,treeMem, *tkGeom_);
  this->fillTree(*tree, mTibResiduals_ ,treeMem, *tkGeom_);
  this->fillTree(*tree, mTidResiduals_ ,treeMem, *tkGeom_);
  this->fillTree(*tree, mTobResiduals_ ,treeMem, *tkGeom_);
  this->fillTree(*tree, mTecResiduals_ ,treeMem, *tkGeom_);
  static const int kappadiffindex = this->GetIndex(vTrackHistos_,"h_diff_curvature");
  vTrackHistos_[kappadiffindex]->Add(vTrackHistos_[this->GetIndex(vTrackHistos_,"h_curvature_neg")],
				     vTrackHistos_[this->GetIndex(vTrackHistos_,"h_curvature_pos")],-1,1);

  // Collate Information for Subdetectors
  // create summary histogramms recursively
 
  std::vector<TrackerOfflineValidation::SummaryContainer > vTrackerprofiles;
  this->collateSummaryHists((*fs),(aliTracker), 0, aliobjid, vTrackerprofiles);
   
}


void
TrackerOfflineValidation::collateSummaryHists( TFileDirectory &tfd, const Alignable& ali, int i, 
					       const AlignableObjectId &aliobjid, 
					       std::vector< TrackerOfflineValidation::SummaryContainer > &v_levelProfiles)
{
  
  std::vector<Alignable*> alivec(ali.components());
  if( this->isDetOrDetUnit((alivec)[0]->alignableObjectId()) ) return;

  for(int iComp=0, iCompEnd = ali.components().size();iComp < iCompEnd; ++iComp) {
    std::vector< TrackerOfflineValidation::SummaryContainer > v_profiles;        
    std::string structurename  = aliobjid.typeToName((alivec)[iComp]->alignableObjectId());
 
    LogDebug("TrackerOfflineValidation") << "StructureName = " << structurename;
    std::stringstream dirname;
    
    // add no suffix counter to strip and pixel -> just aesthetics
    if(structurename != "Strip" && structurename != "Pixel") {
      dirname << structurename << "_" << iComp+1;
    } else {
      dirname << structurename;
    }
    
    if(  !(this->isDetOrDetUnit( (alivec)[iComp]->alignableObjectId()) )
	 || (alivec)[0]->components().size() > 1 ) {
      TFileDirectory f = tfd.mkdir((dirname.str()).c_str());
      this->collateSummaryHists( f, *(alivec)[iComp], i, aliobjid, v_profiles);
      v_levelProfiles.push_back(this->bookSummaryHists(tfd, *(alivec[iComp]), ali.alignableObjectId(), iComp, aliobjid));
      for(uint n = 0; n < v_profiles.size(); ++n) {
	this->summarizeBinInContainer(n+1, v_levelProfiles[iComp], v_profiles[n] );
	v_levelProfiles[iComp].sumXResiduals_->Add(v_profiles[n].sumXResiduals_);
	v_levelProfiles[iComp].sumNormXResiduals_->Add(v_profiles[n].sumNormXResiduals_);
	v_levelProfiles[iComp].sumYResiduals_->Add(v_profiles[n].sumYResiduals_);
	v_levelProfiles[iComp].sumNormYResiduals_->Add(v_profiles[n].sumNormYResiduals_);
      }
      //add fit values to stat box
      fitSumResiduals(v_levelProfiles[iComp].sumXResiduals_);
      fitSumResiduals(v_levelProfiles[iComp].sumNormXResiduals_);
      fitSumResiduals(v_levelProfiles[iComp].sumYResiduals_);
      fitSumResiduals(v_levelProfiles[iComp].sumNormYResiduals_);
    } else {
      // nothing to be done for det or detunits
      continue;
    }

  }

}

TrackerOfflineValidation::SummaryContainer 
TrackerOfflineValidation::bookSummaryHists(TFileDirectory &tfd, const Alignable& ali, 
					   align::StructureType type, int i, 
					   const AlignableObjectId &aliobjid)
{

  uint subsize = ali.components().size();
  align::StructureType alitype = ali.alignableObjectId();
  align::StructureType subtype = ali.components()[0]->alignableObjectId();
  SummaryContainer sumContainer;
  
  if( subtype  != align::AlignableDet || (subtype  == align::AlignableDet && ali.components()[0]->components().size() == 1)
      ) {
    sumContainer.summaryXResiduals_ = tfd.make<TH1F>(Form("h_summaryX%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (X' - coordinate);%s ;#LT #Delta x #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				subsize,0.5,subsize+0.5)  ;

    sumContainer.summaryNormXResiduals_ = tfd.make<TH1F>(Form("h_summaryNormX%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (normalized X' - coordinate);%s ;#LT #Delta x #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				subsize,0.5,subsize+0.5)  ;
    
    sumContainer.summaryYResiduals_ = tfd.make<TH1F>(Form("h_summaryY%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (Y' - coordinate);%s;#LT #Delta y #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				subsize,0.5,subsize+0.5)  ;

    sumContainer.summaryNormYResiduals_ = tfd.make<TH1F>(Form("h_summaryNormY%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (normalized Y' - coordinate);%s ;#LT #Delta y #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				subsize,0.5,subsize+0.5)  ;


  } else if( subtype == align::AlignableDet && subsize > 1) {
    sumContainer.summaryXResiduals_ = tfd.make<TH1F>(Form("h_summaryX%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (X' - coordinate);%s;#LT #Delta x #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				(2*subsize),0.5,2*subsize+0.5)  ;  
    
    sumContainer.summaryNormXResiduals_ = tfd.make<TH1F>(Form("h_summaryNormX%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (normalized X' - coordinate);%s;#LT #Delta x #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				(2*subsize),0.5,2*subsize+0.5)  ;  
    
    sumContainer.summaryYResiduals_ = tfd.make<TH1F>(Form("h_summaryY%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (Y' - coordinate);%s;#LT #Delta y #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				(2*subsize),0.5,2*subsize+0.5)  ;  

    sumContainer.summaryNormYResiduals_ = tfd.make<TH1F>(Form("h_summaryNormY%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Summary for substructures in %s %d (normalized Y' - coordinate);%s;#LT #Delta y #GT",
				     aliobjid.typeToName(alitype).c_str(),i,aliobjid.typeToName(subtype).c_str()),
				(2*subsize),0.5,2*subsize+0.5)  ;  


  } else {
    edm::LogWarning("TrackerOfflineValidation") << "@SUB=TrackerOfflineValidation::bookSummaryHists" 
						<< "No summary histogramm for hierarchy level " 
						<< aliobjid.typeToName(subtype);      
  }
  DetId aliDetId = ali.id(); 
  int nbins = 0;
  double xmin = 0., xmax = 0.;
  this->getBinning(aliDetId.subdetId(), XprimeResidual, nbins, xmin, xmax);
  sumContainer.sumXResiduals_ = tfd.make<TH1F>(Form("h_Xprime_%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("X' Residual for %s %d in %s ",aliobjid.typeToName(alitype).c_str(),i,
				     aliobjid.typeToName(type).c_str(),aliobjid.typeToName(subtype).c_str()),
					       nbins, xmin, xmax);
  
  this->getBinning(aliDetId.subdetId(), NormXprimeResidual, nbins, xmin, xmax);
  sumContainer.sumNormXResiduals_ = tfd.make<TH1F>(Form("h_NormXprime_%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
						   Form("Normalized X' Residual for %s %d in %s ",
							aliobjid.typeToName(alitype).c_str(),i,
							aliobjid.typeToName(type).c_str(),
							aliobjid.typeToName(subtype).c_str()),
						   nbins, xmin, xmax);

  this->getBinning(aliDetId.subdetId(), YprimeResidual, nbins, xmin, xmax);
  sumContainer.sumYResiduals_ = tfd.make<TH1F>(Form("h_Yprime_%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
				Form("Y' Residual for %s %d in %s ",aliobjid.typeToName(alitype).c_str(),i,
				     aliobjid.typeToName(type).c_str(),aliobjid.typeToName(subtype).c_str()),
					       nbins, xmin, xmax);

  this->getBinning(aliDetId.subdetId(), NormYprimeResidual, nbins, xmin, xmax);
  sumContainer.sumNormYResiduals_ = tfd.make<TH1F>(Form("h_NormYprime_%s_%d",aliobjid.typeToName(alitype).c_str(),i), 
						   Form("Normalized Y' Residual for %s %d in %s ",
							aliobjid.typeToName(alitype).c_str(),i,
							aliobjid.typeToName(type).c_str(),
							aliobjid.typeToName(subtype).c_str()),
						   nbins, xmin, xmax);

  
  // special case I: For DetUnits and Detwith  only one subcomponent start filling summary histos
  if( (  subtype == align::AlignableDet && ali.components()[0]->components().size() == 1) || 
      subtype  == align::AlignableDetUnit  
      ) {
    for(uint k=0;k<subsize;++k) {
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
  }
  // special case II: Fill summary histos for dets with two detunits 
  else if( subtype == align::AlignableDet && subsize > 1) {
    for(uint k = 0; k < subsize; ++k) { 
      uint jEnd = ali.components()[0]->components().size();
      for(uint j = 0; j <  jEnd; ++j) {
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
TrackerOfflineValidation::Fwhm (const TH1* hist) 
{
  float fwhm = 0.;
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
  fwhm = hist->GetXaxis()->GetBinCenter(right) - hist->GetXaxis()->GetBinCenter(left);
  return fwhm;
}

void 
TrackerOfflineValidation::bookTree(TTree &tree, struct TrackerOfflineValidation::TreeVariables &treeMem)
{
  //variables concerning the tracker components/hierarchy levels
  tree.Branch("moduleId",&treeMem.moduleId_,"modulId/i");
  tree.Branch("subDetId",&treeMem.subDetId_,"subDetId/i");
  tree.Branch("layer",&treeMem.layer_,"layer/i");
  tree.Branch("side",&treeMem.side_,"side/i");
  tree.Branch("rod",&treeMem.rod_,"rod/i");
  tree.Branch("ring",&treeMem.ring_,"ring/i");
  tree.Branch("petal",&treeMem.petal_,"petal/i");
  tree.Branch("blade",&treeMem.blade_,"blade/i");
  tree.Branch("panel",&treeMem.panel_,"panel/i");
  tree.Branch("outerInner",&treeMem.outerInner_,"outerInner/i");
  tree.Branch("isDoubleSide",&treeMem.isDoubleSide_,"isDoubleSide/O");
  
  //variables concerning the tracker geometry
  tree.Branch("posPhi",&treeMem.posPhi_,"gobalPhi/F");
  tree.Branch("posEta",&treeMem.posEta_,"posEta/F");
  tree.Branch("posR",&treeMem.posR_,"posR/F");
  tree.Branch("posX",&treeMem.posX_,"gobalX/F");
  tree.Branch("posY",&treeMem.posY_,"posY/F");
  tree.Branch("posZ",&treeMem.posZ_,"posZ/F");
  
  //mean and RMS values (extracted from histograms(Xprime) on module level)
  tree.Branch("entries",&treeMem.entries_,"entries/i");
  tree.Branch("meanX",&treeMem.meanX_,"meanX/F");
  tree.Branch("rmsX",&treeMem.rmsX_,"rmsX/F");
  tree.Branch("sigmaX",&treeMem.sigmaX_,"sigmaX/F");
  tree.Branch("sigmaNormX",&treeMem.sigmaNormX_,"sigmaNormX/F"); 
  tree.Branch("meanNormX",&treeMem.meanNormX_,"meanNormX/F");
  tree.Branch("rmsNormX",&treeMem.rmsNormX_,"rmsNormX/F");
  if (useFit_) {
  tree.Branch("fitMeanX",&treeMem.fitMeanX_,"fitMeanX/F"); 
  tree.Branch("fitSigmaX",&treeMem.fitSigmaX_,"fitSigmaX/F");
  tree.Branch("fitMeanNormX",&treeMem.fitMeanNormX_,"fitMeanNormX/F"); 
  tree.Branch("fitSigmaNormX",&treeMem.fitSigmaNormX_,"fitSigmaNormX/F");
  }
  tree.Branch("numberOfUnderflows",&treeMem.numberOfUnderflows_,"numberOfUnderflows/I");
  tree.Branch("numberOfOverflows",&treeMem.numberOfOverflows_,"numberOfOverflows/I");
  tree.Branch("numberOfOutliers",&treeMem.numberOfOutliers_,"numberOfOutliers/I");
  tree.Branch("chi2PerDof",&treeMem.chi2PerDof_,"chi2PerDof/F");
  tree.Branch("meanY",&treeMem.meanY_,"meanY/F");
  tree.Branch("rmsY",&treeMem.rmsY_,"rmsY/F");
  // if (stripYResiduals_==false), these stay empty for strip, but we need it in tree for pixel
  tree.Branch("meanNormY",&treeMem.meanNormY_,"meanNormY/F");
  tree.Branch("rmsNormY",&treeMem.rmsNormY_,"rmsNormY/F");
  
  //histogram names 
  tree.Branch("histNameX",&treeMem.histNameX_,"histNameX/b");
  tree.Branch("histNameNormX",&treeMem.histNameNormX_,"histNameNormX/b");
  // if (stripYResiduals_==false), these stay empty for strip, but we need it in tree for pixel
  tree.Branch("histNameY",&treeMem.histNameY_,"histNameY/b");
  tree.Branch("histNameNormY",&treeMem.histNameNormY_,"histNameNormY/b");

  // book tree variables in local coordinates if set in cf
  if(lCoorHistOn_) {
    //mean and RMS values (extracted from histograms(X) on module level)
    tree.Branch("meanLocalX",&treeMem.meanLocalX_,"meanLocalX/F");
    tree.Branch("rmsLocalX",&treeMem.rmsLocalX_,"rmsLocalX/F");
    tree.Branch("meanNormLocalX",&treeMem.meanNormLocalX_,"meanNormLocalX/F");
    tree.Branch("rmsNormLocalX",&treeMem.rmsNormLocalX_,"rmsNormLocalX/F");
    tree.Branch("histNameLocalX",&treeMem.histNameLocalX_,"histNameLocalX/b");
    tree.Branch("histNameNormLocalX",&treeMem.histNameNormLocalX_,"histNameNormLocalX/b");
  }

}

void 
TrackerOfflineValidation::fillTree(TTree &tree,const std::map<int, TrackerOfflineValidation::ModuleHistos> &moduleHist_, struct TrackerOfflineValidation::TreeVariables &treeMem ,const TrackerGeometry &tkgeom )
{
 
  for(std::map<int, TrackerOfflineValidation::ModuleHistos>::const_iterator it = moduleHist_.begin(), 
	itEnd= moduleHist_.end(); it != itEnd;++it ) { 
    treeMem.clear(); // make empty/default
    //variables concerning the tracker components/hierarchy levels
    DetId detId_ = it->first;
    treeMem.moduleId_ = detId_;
    treeMem.subDetId_ = detId_.subdetId();
    treeMem.isDoubleSide_ =0;

    if(treeMem.subDetId_== PixelSubdetector::PixelBarrel){
      PXBDetId pxbId(detId_); 
      treeMem.layer_ = pxbId.layer(); 
      treeMem.rod_ = pxbId.ladder();
  
    } else if(treeMem.subDetId_ == PixelSubdetector::PixelEndcap){
      PXFDetId pxfId(detId_); 
      treeMem.layer_ = pxfId.disk(); 
      treeMem.side_ = pxfId.side();
      treeMem.blade_ = pxfId.blade(); 
      treeMem.panel_ = pxfId.panel();

    } else if(treeMem.subDetId_ == StripSubdetector::TIB){
      TIBDetId tibId(detId_); 
      treeMem.layer_ = tibId.layer(); 
      treeMem.side_ = tibId.string()[0];
      treeMem.rod_ = tibId.string()[2]; 
      treeMem.outerInner_ = tibId.string()[1]; 
      if (tibId.isDoubleSide())  treeMem.isDoubleSide_ = 1;
    } else if(treeMem.subDetId_ == StripSubdetector::TID){
      TIDDetId tidId(detId_); 
      treeMem.layer_ = tidId.wheel(); 
      treeMem.side_ = tidId.side();
      treeMem.ring_ = tidId.ring(); 
      treeMem.outerInner_ = tidId.module()[0]; 
      if (tidId.isDoubleSide())  treeMem.isDoubleSide_ = 1;
    } else if(treeMem.subDetId_ == StripSubdetector::TOB){
      TOBDetId tobId(detId_); 
      treeMem.layer_ = tobId.layer(); 
      treeMem.side_ = tobId.rod()[0];
      treeMem.rod_ = tobId.rod()[1]; 
      if (tobId.isDoubleSide())  treeMem.isDoubleSide_ = 1;
    } else if(treeMem.subDetId_ == StripSubdetector::TEC) {
      TECDetId tecId(detId_); 
      treeMem.layer_ = tecId.wheel(); 
      treeMem.side_ = tecId.side();
      treeMem.ring_ = tecId.ring(); 
      treeMem.petal_ = tecId.petal()[1]; 
      treeMem.outerInner_ = tecId.petal()[0];
      if (tecId.isDoubleSide())  treeMem.isDoubleSide_ = 1; 
    }
    
    //variables concerning the tracker geometry
    
    const Surface::PositionType &gPModule = tkgeom.idToDet(detId_)->position();
    treeMem.posPhi_ = gPModule.phi();
    treeMem.posEta_ = gPModule.eta();
    treeMem.posR_   = gPModule.perp();
    treeMem.posX_   = gPModule.x();
    treeMem.posY_   = gPModule.y();
    treeMem.posZ_   = gPModule.z();

    //mean and RMS values (extracted from histograms(Xprime on module level)
    treeMem.entries_ = static_cast<UInt_t>(it->second.ResXprimeHisto->GetEntries());
    treeMem.meanX_ = it->second.ResXprimeHisto->GetMean();
    treeMem.rmsX_ = it->second.ResXprimeHisto->GetRMS();
    //treeMem.sigmaX_ = Fwhm(it->second.ResXprimeHisto)/2.355;
    if (useFit_) {
      
      //call fit function which returns mean and sigma from the fit
      //for absolute residuals
       std::pair<float,float> fitResult1 = fitResiduals(it->second.ResXprimeHisto, it->second.ResXprimeHisto->GetMean(), it->second.ResXprimeHisto->GetRMS());
       treeMem.fitMeanX_=fitResult1.first;
       treeMem.fitSigmaX_=fitResult1.second;
       //for normalized residuals
       std::pair<float,float> fitResult2 = fitResiduals(it->second.NormResXprimeHisto, it->second.NormResXprimeHisto->GetMean(), it->second.NormResXprimeHisto->GetRMS());
       treeMem.fitMeanNormX_=fitResult2.first;
       treeMem.fitSigmaNormX_=fitResult2.second;

    }

    int numberOfBins=it->second.ResXprimeHisto->GetNbinsX();
    treeMem.numberOfUnderflows_ = it->second.ResXprimeHisto->GetBinContent(0);
    treeMem.numberOfOverflows_ = it->second.ResXprimeHisto->GetBinContent(numberOfBins+1);
    treeMem.numberOfOutliers_ =  it->second.ResXprimeHisto->GetBinContent(0)+it->second.ResXprimeHisto->GetBinContent(numberOfBins+1);
    //mean and RMS values (extracted from histograms(normalized Xprime on module level)
    treeMem.meanNormX_ = it->second.NormResXprimeHisto->GetMean();
    treeMem.rmsNormX_ = it->second.NormResXprimeHisto->GetRMS();
    double stats[20];
    if(it->second.NormResXprimeHisto->GetEntries()>0){
      it->second.NormResXprimeHisto->GetStats(stats);
      treeMem.chi2PerDof_ = stats[3]/(stats[0]-1);
    }
    treeMem.sigmaNormX_ = Fwhm(it->second.NormResXprimeHisto)/2.355;
    treeMem.histNameX_= it->second.ResXprimeHisto->GetName();
    treeMem.histNameNormX_= it->second.NormResXprimeHisto->GetName();
    

    // fill tree variables in local coordinates if set in cfg
    if(lCoorHistOn_) {
      treeMem.meanLocalX_ = it->second.ResHisto->GetMean();
      treeMem.rmsLocalX_ = it->second.ResHisto->GetRMS();
      treeMem.meanNormLocalX_ = it->second.NormResHisto->GetMean();
      treeMem.rmsNormLocalX_ = it->second.NormResHisto->GetRMS();

      treeMem.histNameLocalX_ = it->second.ResHisto->GetName();
      treeMem.histNameNormLocalX_=it->second.NormResHisto->GetName();
    }

    // mean and RMS values in local y (extracted from histograms(normalized Yprime on module level)
    // might exist in pixel only
    if (it->second.ResYprimeHisto) {//(stripYResiduals_){
      treeMem.meanY_ = it->second.ResYprimeHisto->GetMean();
      treeMem.rmsY_ = it->second.ResYprimeHisto->GetRMS();
      treeMem.histNameY_= it->second.ResYprimeHisto->GetName();
    }
    if (it->second.NormResYprimeHisto) {
      treeMem.meanNormY_ = it->second.NormResYprimeHisto->GetMean();
      treeMem.rmsNormY_ = it->second.NormResYprimeHisto->GetRMS();
      treeMem.histNameNormY_= it->second.ResYprimeHisto->GetName();
    }

    tree.Fill();
  }
}

std::pair<float,float> 
TrackerOfflineValidation::fitResiduals(const TH1 *h,float meantmp,float rmstmp)
{
  std::pair<float,float> fitResult;
  try{
    TH1*hist=0;
    hist = const_cast<TH1*>(h);
    TF1 *ftmp1= new TF1("ftmp1","gaus",meantmp-2*rmstmp, meantmp+2*rmstmp); 
    hist->Fit("ftmp1","Q0LR");
    float mean = ftmp1->GetParameter(1);
    float sigma = ftmp1->GetParameter(2);
    delete ftmp1;
    TF1 *ftmp2= new TF1("ftmp2","gaus",mean-3*sigma,mean+3*sigma); 
    hist->Fit("ftmp2","Q0LR");
    fitResult.first = ftmp2->GetParameter(1);
    fitResult.second = ftmp2->GetParameter(2);
    delete ftmp2;
  }catch (cms::Exception const & e) {
    std::cout << e.what() << std::endl;
    std::cout <<"set values of fit to 9999" << std::endl;
    fitResult.first = 9999.;
    fitResult.second = 9999.;
  }
   return fitResult;
}

void 
TrackerOfflineValidation::fitSumResiduals(const TH1 *h)
{
  
  try{
    TH1*hist=0;
    hist = const_cast<TH1*>(h);
    float meantmp = hist->GetMean();
    float rmstmp = hist->GetRMS();
    TF1 *ftmp1= new TF1("ftmp1","gaus",meantmp-2*rmstmp, meantmp+2*rmstmp); 
    hist->Fit("ftmp1","Q0LR");
    float mean = ftmp1->GetParameter(1);
    float sigma = ftmp1->GetParameter(2);
    delete ftmp1;
    TF1 *ftmp2= new TF1("ftmp2","gaus",mean-3*sigma,mean+3*sigma); 
    hist->Fit("ftmp2","Q0LR");
    delete ftmp2;
    
  }catch (cms::Exception const & e) {
    std::cout << e.what() << std::endl;
    std::cout <<"set values of fit to 9999" << std::endl;
  }
 }
//define this as a plug-in
DEFINE_FWK_MODULE(TrackerOfflineValidation);
