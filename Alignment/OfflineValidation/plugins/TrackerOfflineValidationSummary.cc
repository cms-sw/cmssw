// -*- C++ -*-
//
// Package:    TrackerOfflineValidationSummary
// Class:      TrackerOfflineValidationSummary
// 
/**\class TrackerOfflineValidationSummary TrackerOfflineValidationSummary.cc Alignment/TrackerOfflineValidationSummary/src/TrackerOfflineValidationSummary.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Johannes Hauk
//         Created:  Sat Aug 22 10:31:34 CEST 2009
// $Id: TrackerOfflineValidationSummary.cc,v 1.1 2009/10/09 14:07:29 hauk Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"


#include "TTree.h"
//#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "TH1.h"
#include "TMath.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//
// class decleration
//

class TrackerOfflineValidationSummary : public edm::EDAnalyzer {
   public:
      explicit TrackerOfflineValidationSummary(const edm::ParameterSet&);
      ~TrackerOfflineValidationSummary();


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
      
      virtual void beginJob(const edm::EventSetup& es) ;
      virtual void analyze(const edm::Event& evt, const edm::EventSetup&){};
      virtual void endJob() ;
      
      void fillTree(TTree& tree, std::map<int, TrackerOfflineValidationSummary::ModuleHistos>& moduleHist, 
		TkOffTreeVariables& treeMem, const TrackerGeometry& tkgeom );
      
      std::pair<float,float> fitResiduals(TH1* hist)const;
      float getMedian(const TH1* hist)const;
      
      void associateModuleHistsWithTree(const TkOffTreeVariables& treeMem, TrackerOfflineValidationSummary::ModuleHistos& moduleHists);
      
      // ----------member data ---------------------------
      
      const edm::ParameterSet parSet_;
      edm::ESHandle<TrackerGeometry> tkGeom_;
      
      // parameters from cfg to steer
      const std::string moduleDirectory_;
      const bool useFit_;
      
      DQMStore* dbe_;
      
      std::map<int,TrackerOfflineValidationSummary::ModuleHistos> mPxbResiduals_;
      std::map<int,TrackerOfflineValidationSummary::ModuleHistos> mPxeResiduals_;
      std::map<int,TrackerOfflineValidationSummary::ModuleHistos> mTibResiduals_;
      std::map<int,TrackerOfflineValidationSummary::ModuleHistos> mTidResiduals_;
      std::map<int,TrackerOfflineValidationSummary::ModuleHistos> mTobResiduals_;
      std::map<int,TrackerOfflineValidationSummary::ModuleHistos> mTecResiduals_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackerOfflineValidationSummary::TrackerOfflineValidationSummary(const edm::ParameterSet& iConfig):
   parSet_(iConfig), moduleDirectory_(parSet_.getParameter<std::string>("moduleDirectoryInOutput")),
   useFit_(parSet_.getParameter<bool>("useFit")), dbe_(0)
{
  //now do what ever initialization is needed
  dbe_ = edm::Service<DQMStore>().operator->();
}


TrackerOfflineValidationSummary::~TrackerOfflineValidationSummary()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
//void
//TrackerOfflineValidationSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
//{
//}


// ------------ method called once each job just before starting event loop  ------------
void 
TrackerOfflineValidationSummary::beginJob(const edm::EventSetup& es)
{
  es.get<TrackerDigiGeometryRecord>().get( tkGeom_ );
  const TrackerGeometry* bareTkGeomPtr = &(*tkGeom_);
  const TrackingGeometry::DetIdContainer& detIdContainer = bareTkGeomPtr->detIds();
  std::vector<DetId>::const_iterator iDet;
  for(iDet = detIdContainer.begin(); iDet != detIdContainer.end(); ++iDet){
    const DetId& detId = *iDet;
    const uint32_t rawId = detId.rawId();
    const unsigned int subdetId = detId.subdetId();
    if     (subdetId == PixelSubdetector::PixelBarrel) mPxbResiduals_[rawId];
    else if(subdetId == PixelSubdetector::PixelEndcap) mPxeResiduals_[rawId];
    else if(subdetId  == StripSubdetector::TIB) mTibResiduals_[rawId];
    else if(subdetId  == StripSubdetector::TID) mTidResiduals_[rawId];
    else if(subdetId  == StripSubdetector::TOB) mTobResiduals_[rawId];
    else if(subdetId  == StripSubdetector::TEC) mTecResiduals_[rawId];
    else {
      throw cms::Exception("Geometry Error")
        << "[TrackerOfflineValidationSummary] Error, tried to get reference for non-tracker subdet " << subdetId 
        << " from detector " << detId.det();
      mPxbResiduals_[0];
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackerOfflineValidationSummary::endJob()
{
  AlignableTracker aliTracker(&(*tkGeom_));
  AlignableObjectId aliobjid;
  
  TTree* tree = new TTree("TkOffVal","TkOffVal");
  
  TkOffTreeVariables *treeMemPtr = new TkOffTreeVariables;
  // We create branches for all members of 'TkOffTreeVariables' (even if not needed).
  // This works because we have a dictionary for 'TkOffTreeVariables'
  // (see src/classes_def.xml and src/classes.h):
  tree->Branch("TkOffTreeVariables", &treeMemPtr); // address of pointer!
  
  this->fillTree(*tree, mPxbResiduals_, *treeMemPtr, *tkGeom_);
  this->fillTree(*tree, mPxeResiduals_, *treeMemPtr, *tkGeom_);
  this->fillTree(*tree, mTibResiduals_, *treeMemPtr, *tkGeom_);
  this->fillTree(*tree, mTidResiduals_, *treeMemPtr, *tkGeom_);
  this->fillTree(*tree, mTobResiduals_, *treeMemPtr, *tkGeom_);
  this->fillTree(*tree, mTecResiduals_, *treeMemPtr, *tkGeom_);
  
  //dbe_->showDirStructure();
  //dbe_->save("dqmOut.root");
  
  // Put here the method for filling histograms which show summarized values (mean, rms, median ...)
  // of the module-based histograms from TrackerOfflineValidation
  
  delete tree; tree = 0;
  delete treeMemPtr; treeMemPtr = 0;
}


void 
TrackerOfflineValidationSummary::fillTree(TTree& tree,
				   std::map<int, TrackerOfflineValidationSummary::ModuleHistos>& moduleHist,
				   TkOffTreeVariables& treeMem, const TrackerGeometry& tkgeom)
{
  for(std::map<int, TrackerOfflineValidationSummary::ModuleHistos>::iterator it = moduleHist.begin(), 
	itEnd= moduleHist.end(); it != itEnd;++it ) { 
    treeMem.clear(); // make empty/default
    
    //variables concerning the tracker components/hierarchy levels
    DetId detId_ = it->first;
    treeMem.moduleId = detId_;
    treeMem.subDetId = detId_.subdetId();
    treeMem.isDoubleSide =0;

    if(treeMem.subDetId == PixelSubdetector::PixelBarrel){
      PXBDetId pxbId(detId_);
      unsigned int whichHalfBarrel(1), rawId(detId_.rawId());  //DetId does not know about halfBarrels is PXB ...
      if( (rawId>=302056964 && rawId<302059300) || (rawId>=302123268 && rawId<302127140) || (rawId>=302189572 && rawId<302194980) )whichHalfBarrel=2;
      treeMem.layer = pxbId.layer();
      treeMem.half = whichHalfBarrel; 
      treeMem.rod = pxbId.ladder();     // ... so, ladder is not per halfBarrel-Layer, but per barrel-layer! Needs complicated calculation in associateModuleHistsWithTree()
      treeMem.module = pxbId.module();
    } else if(treeMem.subDetId == PixelSubdetector::PixelEndcap){
      PXFDetId pxfId(detId_); 
      unsigned int whichHalfCylinder(1), rawId(detId_.rawId());  //DetId does not kmow about halfCylinders in PXF
      if( (rawId>=352394500 && rawId<352406032) || (rawId>=352460036 && rawId<352471568) || (rawId>=344005892 && rawId<344017424) || (rawId>=344071428 && rawId<344082960) )whichHalfCylinder=2;
      treeMem.layer = pxfId.disk(); 
      treeMem.side = pxfId.side();
      treeMem.half = whichHalfCylinder;
      treeMem.blade = pxfId.blade(); 
      treeMem.panel = pxfId.panel();
      treeMem.module = pxfId.module();
    } else if(treeMem.subDetId == StripSubdetector::TIB){
      TIBDetId tibId(detId_); 
      unsigned int whichHalfShell(1), rawId(detId_.rawId());  //DetId does not kmow about halfShells in TIB
       if ( (rawId>=369120484 && rawId<369120688) || (rawId>=369121540 && rawId<369121776) || (rawId>=369136932 && rawId<369137200) || (rawId>=369137988 && rawId<369138288) ||
            (rawId>=369153396 && rawId<369153744) || (rawId>=369154436 && rawId<369154800) || (rawId>=369169844 && rawId<369170256) || (rawId>=369170900 && rawId<369171344) ||
	    (rawId>=369124580 && rawId<369124784) || (rawId>=369125636 && rawId<369125872) || (rawId>=369141028 && rawId<369141296) || (rawId>=369142084 && rawId<369142384) ||
	    (rawId>=369157492 && rawId<369157840) || (rawId>=369158532 && rawId<369158896) || (rawId>=369173940 && rawId<369174352) || (rawId>=369174996 && rawId<369175440) ) whichHalfShell=2;
      treeMem.layer = tibId.layer(); 
      treeMem.side = tibId.string()[0];
      treeMem.half = whichHalfShell;
      treeMem.rod = tibId.string()[2]; 
      treeMem.outerInner = tibId.string()[1]; 
      treeMem.module = tibId.module();
      treeMem.isStereo = tibId.stereo();
      treeMem.isDoubleSide = tibId.isDoubleSide();
    } else if(treeMem.subDetId == StripSubdetector::TID){
      TIDDetId tidId(detId_); 
      treeMem.layer = tidId.wheel(); 
      treeMem.side = tidId.side();
      treeMem.ring = tidId.ring(); 
      treeMem.outerInner = tidId.module()[0]; 
      treeMem.module = tidId.module()[1];
      treeMem.isStereo = tidId.stereo();
      treeMem.isDoubleSide = tidId.isDoubleSide();
    } else if(treeMem.subDetId == StripSubdetector::TOB){
      TOBDetId tobId(detId_); 
      treeMem.layer = tobId.layer(); 
      treeMem.side = tobId.rod()[0];
      treeMem.rod = tobId.rod()[1]; 
      treeMem.module = tobId.module();
      treeMem.isStereo = tobId.stereo();
      treeMem.isDoubleSide = tobId.isDoubleSide();
    } else if(treeMem.subDetId == StripSubdetector::TEC) {
      TECDetId tecId(detId_); 
      treeMem.layer = tecId.wheel(); 
      treeMem.side  = tecId.side();
      treeMem.ring  = tecId.ring(); 
      treeMem.petal = tecId.petal()[1]; 
      treeMem.outerInner = tecId.petal()[0];
      treeMem.module = tecId.module();
      treeMem.isStereo = tecId.stereo();
      treeMem.isDoubleSide = tecId.isDoubleSide();
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
    
    
    // Assign histos from first step (TrackerOfflineValidation) to the module's entry in the TTree for retrieving mean, rms, median ...
    associateModuleHistsWithTree(treeMem, it->second);
    
    
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
    
    //treeMem.sigmaNormX = Fwhm(it->second.NormResXprimeHisto)/2.355;
    treeMem.histNameX = it->second.ResXprimeHisto->GetName();
    treeMem.histNameNormX = it->second.NormResXprimeHisto->GetName();
    

    // fill tree variables in local coordinates if set in cfg of TrackerOfllineValidation
    if(it->second.ResHisto && it->second.NormResHisto){ // if(lCoorHistOn_) {
      treeMem.meanLocalX = it->second.ResHisto->GetMean();
      treeMem.rmsLocalX = it->second.ResHisto->GetRMS();
      treeMem.meanNormLocalX = it->second.NormResHisto->GetMean();
      treeMem.rmsNormLocalX = it->second.NormResHisto->GetRMS();
      treeMem.histNameLocalX = it->second.ResHisto->GetName();
      treeMem.histNameNormLocalX = it->second.NormResHisto->GetName();
    }

    // mean and RMS values in local y (extracted from histograms(normalized Yprime on module level)
    // might exist in pixel only
    if (it->second.ResYprimeHisto) { //(stripYResiduals_){
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
    tree.Fill();
  }
}


void
TrackerOfflineValidationSummary::associateModuleHistsWithTree(const TkOffTreeVariables& treeMem, TrackerOfflineValidationSummary::ModuleHistos& moduleHists){
   std::stringstream histDir;
   if(moduleDirectory_.length() != 0)histDir<<moduleDirectory_<<"/";
   std::string wheelOrLayer("_layer_");
   if(treeMem.subDetId == PixelSubdetector::PixelBarrel){
     unsigned int half(treeMem.half), layer(treeMem.layer), ladder(0);
     if(layer==1){
       if(half==2)ladder = treeMem.rod -5;
       else if(treeMem.rod>15)ladder = treeMem.rod -10;
       else ladder = treeMem.rod;
     }else if(layer==2){
       if(half==2)ladder = treeMem.rod -8;
       else if(treeMem.rod>24)ladder = treeMem.rod -16;
       else ladder = treeMem.rod;
     }else if(layer==3){
       if(half==2)ladder = treeMem.rod -11;
       else if(treeMem.rod>33)ladder = treeMem.rod -22;
       else ladder = treeMem.rod;
     }
     histDir<<"Pixel/TPBBarrel_1/TPBHalfBarrel_"<<treeMem.half<<"/TPBLayer_"<<treeMem.layer<<"/TPBLadder_"<<ladder;
   }else if(treeMem.subDetId == PixelSubdetector::PixelEndcap){
     unsigned int side(treeMem.side), half(treeMem.half), blade(0);
     if(side==1)side=3;
     if(half==2)blade = treeMem.blade -6;
     else if(treeMem.blade>18)blade = treeMem.blade -12;
     else blade = treeMem.blade;
     histDir<<"Pixel/TPEEndcap_"<<side<<"/TPEHalfCylinder_"<<treeMem.half<<"/TPEHalfDisk_"<<treeMem.layer<<"/TPEBlade_"<<blade<<"/TPEPanel_"<<treeMem.panel;
     wheelOrLayer = "_wheel_";
   }else if(treeMem.subDetId == StripSubdetector::TIB){
     unsigned int half(treeMem.half), layer(treeMem.layer), surface(treeMem.outerInner), string(0);
     if(half==2){
       if(layer==1){
         if(surface==1)string = treeMem.rod -13;
	 else if(surface==2)string = treeMem.rod -15;
       }
       if(layer==2){
         if(surface==1)string = treeMem.rod -17;
	 else if(surface==2)string = treeMem.rod -19;
       }
       if(layer==3){
         if(surface==1)string = treeMem.rod -22;
	 else if(surface==2)string = treeMem.rod -23;
       }
       if(layer==4){
         if(surface==1)string = treeMem.rod -26;
	 else if(surface==2)string = treeMem.rod -28;
       }
     }
     else string = treeMem.rod;
     std::stringstream detString;
     if(treeMem.layer<3 && !treeMem.isDoubleSide)detString<<"/Det_"<<treeMem.module;
     else detString<<"";
     histDir<<"Strip/TIBBarrel_1/TIBHalfBarrel_"<<treeMem.side<<"/TIBLayer_"<<treeMem.layer<<"/TIBHalfShell_"<<treeMem.half<<"/TIBSurface_"<<treeMem.outerInner<<"/TIBString_"<<string<<detString.str();
   }else if(treeMem.subDetId == StripSubdetector::TID){
     unsigned int side(treeMem.side), outerInner(0);
     if(side==1)side=3;
     if(treeMem.outerInner==1)outerInner=2;
     else if(treeMem.outerInner==2)outerInner=1;
     std::stringstream detString;
     if(treeMem.ring<3 && !treeMem.isDoubleSide)detString<<"/Det_"<<treeMem.module;
     else detString<<"";
     histDir<<"Strip/TIDEndcap_"<<side<<"/TIDDisk_"<<treeMem.layer<<"/TIDRing_"<<treeMem.ring<<"/TIDSide_"<<outerInner<<detString.str();
     wheelOrLayer = "_wheel_";
   }else if(treeMem.subDetId == StripSubdetector::TOB){
     std::stringstream detString;
     if(treeMem.layer<3 && !treeMem.isDoubleSide)detString<<"/Det_"<<treeMem.module;
     else detString<<"";
     histDir<<"Strip/TOBBarrel_4/TOBHalfBarrel_"<<treeMem.side<<"/TOBLayer_"<<treeMem.layer<<"/TOBRod_"<<treeMem.rod<<detString.str();
   }else if(treeMem.subDetId == StripSubdetector::TEC) {
     unsigned int side(0), outerInner(0), ring(0);
     if(treeMem.side==1)side=6;
     else if(treeMem.side==2)side=5;
     if(treeMem.outerInner==1)outerInner = 2;
     else if(treeMem.outerInner==2)outerInner=1;
     if(treeMem.layer>3 && treeMem.layer<7)ring = treeMem.ring -1;
     else if(treeMem.layer==7 || treeMem.layer==8)ring = treeMem.ring -2;
     else if(treeMem.layer==9)ring = treeMem.ring -3;
     else ring = treeMem.ring;
     std::stringstream detString;
     if((treeMem.ring<3 || treeMem.ring==5) && !treeMem.isDoubleSide)detString<<"/Det_"<<treeMem.module;
     else detString<<"";
     histDir<<"Strip/TECEndcap_"<<side<<"/TECDisk_"<<treeMem.layer<<"/TECSide_"<<outerInner<<"/TECPetal_"<<treeMem.petal<<"/TECRing_"<<ring<<detString.str();
     wheelOrLayer = "_wheel_";
   }
   std::stringstream histName;
   histName<<"residuals_subdet_"<<treeMem.subDetId<<wheelOrLayer<<treeMem.layer<<"_module_"<<treeMem.moduleId;
   
   std::string fullPath;
   
   
   fullPath = histDir.str()+"/h_xprime_"+histName.str();
   if(dbe_->get(fullPath)) moduleHists.ResXprimeHisto = dbe_->get(fullPath)->getTH1();
   else{edm::LogError("TrackerOfflineValidationSummary")<<"Problem with names in input file produced in TrackerOfflineValidation ...\n"
                                                        <<"This histogram should exist in every configuration, "
							<<"but no histogram with name "<<fullPath<<" is found!";
     return;
   }
   fullPath = histDir.str()+"/h_normxprime"+histName.str();
   if(dbe_->get(fullPath)) moduleHists.NormResXprimeHisto = dbe_->get(fullPath)->getTH1();
   fullPath = histDir.str()+"/h_yprime_"+histName.str();
   if(dbe_->get(fullPath)) moduleHists.ResYprimeHisto = dbe_->get(fullPath)->getTH1();
   fullPath = histDir.str()+"/h_normyprime"+histName.str();
   if(dbe_->get(fullPath)) moduleHists.NormResYprimeHisto = dbe_->get(fullPath)->getTH1();
   fullPath = histDir.str()+"/h_"+histName.str();
   if(dbe_->get(fullPath)) moduleHists.ResHisto = dbe_->get(fullPath)->getTH1();
   fullPath = histDir.str()+"/h_norm"+histName.str();
   if(dbe_->get(fullPath)) moduleHists.NormResHisto = dbe_->get(fullPath)->getTH1();
}


std::pair<float,float> 
TrackerOfflineValidationSummary::fitResiduals(TH1* hist) const
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
TrackerOfflineValidationSummary::getMedian(const TH1 *histo) const
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
DEFINE_FWK_MODULE(TrackerOfflineValidationSummary);
