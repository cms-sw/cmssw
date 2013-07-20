// -*- C++ -*-
// Package:    SiStripChannelGain
// Class:      SiStripGainCosmicCalculator
// Original Author:  G. Bruno, D. Kcira
//         Created:  Mon May 20 10:04:31 CET 2007
// $Id: SiStripGainCosmicCalculator.cc,v 1.13 2013/01/11 05:51:19 wmtan Exp $
#include "CalibTracker/SiStripChannelGain/plugins/SiStripGainCosmicCalculator.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"

//---------------------------------------------------------------------------------------------------------
SiStripGainCosmicCalculator::SiStripGainCosmicCalculator(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>(iConfig){
  edm::LogInfo("SiStripGainCosmicCalculator::SiStripGainCosmicCalculator");
  ExpectedChargeDeposition = 200.;
  edm::LogInfo("SiStripApvGainCalculator::SiStripApvGainCalculator")<<"ExpectedChargeDeposition="<<ExpectedChargeDeposition;

  TrackProducer =  iConfig.getParameter<std::string>("TrackProducer");
  TrackLabel    =  iConfig.getParameter<std::string>("TrackLabel");

  detModulesToBeExcluded.clear(); detModulesToBeExcluded = iConfig.getParameter< std::vector<unsigned> >("detModulesToBeExcluded");
  MinNrEntries = iConfig.getUntrackedParameter<unsigned>("minNrEntries", 20);
  MaxChi2OverNDF = iConfig.getUntrackedParameter<double>("maxChi2OverNDF", 5.);

  outputHistogramsInRootFile = iConfig.getParameter<bool>("OutputHistogramsInRootFile");
  outputFileName = iConfig.getParameter<std::string>("OutputFileName");

  edm::LogInfo("SiStripApvGainCalculator")<<"Clusters from "<<detModulesToBeExcluded.size()<<" modules will be ignored in the calibration:";
  edm::LogInfo("SiStripApvGainCalculator")<<"The calibration for these DetIds will be set to a default value";
  for( std::vector<uint32_t>::const_iterator imod = detModulesToBeExcluded.begin(); imod != detModulesToBeExcluded.end(); imod++){
    edm::LogInfo("SiStripApvGainCalculator")<<"exclude detid = "<< *imod;
  }

  printdebug_ = iConfig.getUntrackedParameter<bool>("printDebug", false);
  tTopo = nullptr;
}


SiStripGainCosmicCalculator::~SiStripGainCosmicCalculator(){
  edm::LogInfo("SiStripGainCosmicCalculator::~SiStripGainCosmicCalculator");
}

void SiStripGainCosmicCalculator::algoEndJob(){
}

void SiStripGainCosmicCalculator::algoBeginJob(const edm::EventSetup& iSetup)
{
   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<IdealGeometryRecord>().get(tTopoHandle);
   tTopo = tTopoHandle.product();
 
   eventSetupCopy_ = &iSetup;
   std::cout<<"SiStripGainCosmicCalculator::algoBeginJob called"<<std::endl;
   total_nr_of_events = 0;
   HlistAPVPairs = new TObjArray(); HlistOtherHistos = new TObjArray();
   //
   HlistOtherHistos->Add(new TH1F( Form("APVPairCorrections"), Form("APVPairCorrections"), 50,-1.,4.));
   HlistOtherHistos->Add(new TH1F(Form("APVPairCorrectionsTIB1mono"),Form("APVPairCorrectionsTIB1mono"),50,-1.,4.));
   HlistOtherHistos->Add(new TH1F(Form("APVPairCorrectionsTIB1stereo"),Form("APVPairCorrectionsTIB1stereo"),50,-1.,4.));
   HlistOtherHistos->Add(new TH1F(Form("APVPairCorrectionsTIB2"),Form("APVPairCorrectionsTIB2"),50,-1.,4.));
   HlistOtherHistos->Add(new TH1F(Form("APVPairCorrectionsTOB1"),Form("APVPairCorrectionsTOB1"),50,-1.,4.));
   HlistOtherHistos->Add(new TH1F(Form("APVPairCorrectionsTOB2"),Form("APVPairCorrectionsTOB2"),50,-1.,4.));
   HlistOtherHistos->Add(new TH1F(Form("LocalAngle"),Form("LocalAngle"),70,-0.1,3.4));
   HlistOtherHistos->Add(new TH1F(Form("LocalAngleAbsoluteCosine"),Form("LocalAngleAbsoluteCosine"),48,-0.1,1.1));
   HlistOtherHistos->Add(new TH1F(Form("LocalPosition_cm"),Form("LocalPosition_cm"),100,-5.,5.));
   HlistOtherHistos->Add(new TH1F(Form("LocalPosition_normalized"),Form("LocalPosition_normalized"),100,-1.1,1.1));
   TH1F* local_histo = new TH1F(Form("SiStripRecHitType"),Form("SiStripRecHitType"),2,0.5,2.5); HlistOtherHistos->Add(local_histo);
   local_histo->GetXaxis()->SetBinLabel(1,"simple"); local_histo->GetXaxis()->SetBinLabel(2,"matched");

   // get cabling and find out list of active detectors
   edm::ESHandle<SiStripDetCabling> siStripDetCabling; iSetup.get<SiStripDetCablingRcd>().get(siStripDetCabling);
   std::vector<uint32_t> activeDets; activeDets.clear();
   SelectedDetIds.clear();
   siStripDetCabling->addActiveDetectorsRawIds(activeDets);
//    SelectedDetIds = activeDets; // all active detector modules
   // use SiStripSubStructure for selecting certain regions
   SiStripSubStructure substructure;
   substructure.getTIBDetectors(activeDets, SelectedDetIds, 0, 0, 0, 0); // this adds rawDetIds to SelectedDetIds
   substructure.getTOBDetectors(activeDets, SelectedDetIds, 0, 0, 0);    // this adds rawDetIds to SelectedDetIds
   // get tracker geometry and find nr. of apv pairs for each active detector 
   edm::ESHandle<TrackerGeometry> tkGeom; iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );     
   for(TrackerGeometry::DetContainer::const_iterator it = tkGeom->dets().begin(); it != tkGeom->dets().end(); it++){ // loop over detector modules
     if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){
       uint32_t detid=((*it)->geographicalId()).rawId();
       // get thickness for all detector modules, not just for active, this is strange 
       double module_thickness = (*it)->surface().bounds().thickness(); // get thickness of detector from GeomDet (DetContainer == vector<GeomDet*>)
       thickness_map.insert(std::make_pair(detid,module_thickness));
       //
       bool is_active_detector = false;
       for(std::vector<uint32_t>::iterator iactive = SelectedDetIds.begin(); iactive != SelectedDetIds.end(); iactive++){
         if( *iactive == detid ){
           is_active_detector = true;
           break; // leave for loop if found matching detid
         }
       }
       //
       bool exclude_this_detid = false;
       for( std::vector<uint32_t>::const_iterator imod = detModulesToBeExcluded.begin(); imod != detModulesToBeExcluded.end(); imod++ ){
           if(*imod == detid) exclude_this_detid = true; // found in exclusion list
           break;
       }
       //
       if(is_active_detector && (!exclude_this_detid)){ // check whether is active detector and that should not be excluded
	 const StripTopology& p = dynamic_cast<StripGeomDetUnit*>((*it))->specificTopology();
	 unsigned short NAPVPairs = p.nstrips()/256;
         if( NAPVPairs<2 || NAPVPairs>3 ) {
           edm::LogError("SiStripGainCosmicCalculator")<<"Problem with Number of strips in detector: "<<p.nstrips()<<" Exiting program";
           exit(1);
         }
         for(int iapp = 0; iapp<NAPVPairs; iapp++){
           TString hid = Form("ChargeAPVPair_%i_%i",detid,iapp);
           HlistAPVPairs->Add(new TH1F(hid,hid,45,0.,1350.)); // multiply by 3 to take into account division by width
         }
       }
     }
   }
}

//---------------------------------------------------------------------------------------------------------
void SiStripGainCosmicCalculator::algoAnalyze(const edm::Event & iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  total_nr_of_events++;

  //TO BE RESTORED
  //  anglefinder_->init(event,iSetup);


  // get seeds
//  edm::Handle<TrajectorySeedCollection> seedcoll;
//  event.getByType(seedcoll);
  // get tracks
  Handle<reco::TrackCollection> trackCollection; iEvent.getByLabel(TrackProducer, TrackLabel, trackCollection);
  const reco::TrackCollection *tracks=trackCollection.product();

//  // get magnetic field
//  edm::ESHandle<MagneticField> esmagfield;
//  es.get<IdealMagneticFieldRecord>().get(esmagfield);
//  magfield=&(*esmagfield);
  // loop over tracks
  for(reco::TrackCollection::const_iterator itr = tracks->begin(); itr != tracks->end(); itr++){ // looping over tracks

    //TO BE RESTORED
    //    std::vector<std::pair<const TrackingRecHit *,float> >hitangle =anglefinder_->findtrackangle((*(*seedcoll).begin()),*itr);
    std::vector<std::pair<const TrackingRecHit *,float> >hitangle;// =anglefinder_->findtrackangle((*(*seedcoll).begin()),*itr);

    for(std::vector<std::pair<const TrackingRecHit *,float> >::const_iterator hitangle_iter=hitangle.begin();hitangle_iter!=hitangle.end();hitangle_iter++){
      const TrackingRecHit * trechit = hitangle_iter->first;
      float local_angle=hitangle_iter->second;
      LocalPoint local_position= trechit->localPosition();
      const SiStripRecHit2D* sistripsimplehit=dynamic_cast<const SiStripRecHit2D*>(trechit);
      const SiStripMatchedRecHit2D* sistripmatchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(trechit);
//      std::cout<<" hit/matched "<<std::ios::hex<<sistripsimplehit<<" "<<sistripmatchedhit<<std::endl;
      ((TH1F*) HlistOtherHistos->FindObject("LocalAngle"))->Fill(local_angle);
      ((TH1F*) HlistOtherHistos->FindObject("LocalAngleAbsoluteCosine"))->Fill(fabs(cos(local_angle)));
      if(sistripsimplehit){
        ((TH1F*) HlistOtherHistos->FindObject("SiStripRecHitType"))->Fill(1.);
        const SiStripRecHit2D::ClusterRef & cluster=sistripsimplehit->cluster();
        const std::vector<uint8_t>& ampls = cluster->amplitudes();
//        const std::vector<uint16_t>& ampls = cluster->amplitudes();
        uint32_t thedetid  = cluster->geographicalId();
        double module_width = moduleWidth(thedetid, &iSetup);
        ((TH1F*) HlistOtherHistos->FindObject("LocalPosition_cm"))->Fill(local_position.x());
        ((TH1F*) HlistOtherHistos->FindObject("LocalPosition_normalized"))->Fill(local_position.x()/module_width);
        double module_thickness = moduleThickness(thedetid, &iSetup);
        int ifirststrip= cluster->firstStrip();
        int theapvpairid = int(float(ifirststrip)/256.);
        TH1F* histopointer = (TH1F*) HlistAPVPairs->FindObject(Form("ChargeAPVPair_%i_%i",thedetid,theapvpairid));
        if( histopointer ){
          short cCharge = 0;
          for(unsigned int iampl = 0; iampl<ampls.size(); iampl++){
          cCharge += ampls[iampl];
          }
          double cluster_charge_over_path = ((double)cCharge) * fabs(cos(local_angle)) / ( 10. * module_thickness);
          histopointer->Fill(cluster_charge_over_path);
        }
      }else{
           if(sistripmatchedhit) ((TH1F*) HlistOtherHistos->FindObject("SiStripRecHitType"))->Fill(2.);
      }
    }
  }
}


//---------------------------------------------------------------------------------------------------------
std::pair<double,double> SiStripGainCosmicCalculator::getPeakOfLandau( TH1F * inputHisto){ // automated fitting with finding of the appropriate nr. of ADCs
  // set some default dummy value and return if no entries
  double adcs = -0.5; double error = 0.; double nr_of_entries = inputHisto->GetEntries();
  if(nr_of_entries < MinNrEntries){
    return std::make_pair(adcs,error);
  }
//
//  // fit with initial setting of  parameter values
//  double rms_of_histogram = inputHisto->GetRMS();
//  TF1 *landaufit = new TF1("landaufit","landau",0.,450.);
//  landaufit->SetParameters(nr_of_entries,mean_of_histogram,rms_of_histogram);
//  inputHisto->Fit("landaufit","0Q+");
//  delete landaufit;
//
  // perform fit with standard landau
  inputHisto->Fit("landau","0Q");
  TF1 * fitfunction = (TF1*) inputHisto->GetListOfFunctions()->First();
  adcs = fitfunction->GetParameter("MPV");
  error = fitfunction->GetParError(1); // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
  double chi2 = fitfunction->GetChisquare();
  double ndf = fitfunction->GetNDF();
  double chi2overndf = chi2 / ndf;
  // in case things went wrong, try to refit in smaller range
  if(adcs< 2. || (error/adcs)>1.8 ){
     inputHisto->Fit("landau","0Q",0,0.,400.);
     TF1 * fitfunction2 = (TF1*) inputHisto->GetListOfFunctions()->First();
     std::cout<<"refitting landau for histogram "<<inputHisto->GetTitle()<<std::endl;
     std::cout<<"initial error/adcs ="<<error<<" / "<<adcs<<std::endl;
     std::cout<<"new     error/adcs ="<<fitfunction2->GetParError(1)<<" / "<<fitfunction2->GetParameter("MPV")<<std::endl;
     adcs = fitfunction2->GetParameter("MPV");
     error = fitfunction2->GetParError(1); // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
     chi2 = fitfunction2->GetChisquare();
     ndf = fitfunction2->GetNDF();
     chi2overndf = chi2 / ndf;
   }
   // if still wrong, give up
   if(adcs<2. || chi2overndf>MaxChi2OverNDF){
     adcs = -0.5; error = 0.;
   }
  return std::make_pair(adcs,error);
}

//---------------------------------------------------------------------------------------------------------
double SiStripGainCosmicCalculator::moduleWidth(const uint32_t detid, const edm::EventSetup* iSetup) // get width of the module detid
{ //dk: copied from A. Giammanco and hacked,  module_width values : 10.49 12.03 6.144 7.14 9.3696
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );     
  double module_width=0.;
  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
    std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
  }else{
    module_width = it->surface().bounds().width();
  }
  return module_width;
}

//---------------------------------------------------------------------------------------------------------
double SiStripGainCosmicCalculator::moduleThickness(const uint32_t detid, const edm::EventSetup* iSetup) // get thickness of the module detid
{ //dk: copied from A. Giammanco and hacked
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );
  double module_thickness=0.;
  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
    std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
  }else{
    module_thickness = it->surface().bounds().thickness();
  }
  return module_thickness;
}

//---------------------------------------------------------------------------------------------------------
SiStripApvGain * SiStripGainCosmicCalculator::getNewObject() {
  std::cout<<"SiStripGainCosmicCalculator::getNewObject called"<<std::endl;

  std::cout<<"total_nr_of_events="<<total_nr_of_events<<std::endl;
  // book some more histograms
  TH1F *ChargeOfEachAPVPair = new TH1F("ChargeOfEachAPVPair","ChargeOfEachAPVPair",1,0,1); ChargeOfEachAPVPair->SetBit(TH1::kCanRebin);
  TH1F *EntriesApvPairs = new TH1F("EntriesApvPairs","EntriesApvPairs",1,0,1); EntriesApvPairs->SetBit(TH1::kCanRebin);
  TH1F * NrOfEntries = new TH1F("NrOfEntries","NrOfEntries",351,-0.5,350.5);// NrOfEntries->SetBit(TH1::kCanRebin);
  TH1F * ModuleThickness = new TH1F("ModuleThickness","ModuleThickness",2,0.5,2.5); HlistOtherHistos->Add(ModuleThickness);
  ModuleThickness->GetXaxis()->SetBinLabel(1,"320mu"); ModuleThickness->GetXaxis()->SetBinLabel(2,"500mu"); ModuleThickness->SetYTitle("Nr APVPairs");
  TH1F * ModuleWidth = new TH1F("ModuleWidth","ModuleWidth",5,0.5,5.5); HlistOtherHistos->Add(ModuleWidth);
  ModuleWidth->GetXaxis()->SetBinLabel(1,"6.144cm"); ModuleWidth->GetXaxis()->SetBinLabel(2,"7.14cm");
  ModuleWidth->GetXaxis()->SetBinLabel(3,"9.3696cm"); ModuleWidth->GetXaxis()->SetBinLabel(4,"10.49cm");
  ModuleWidth->GetXaxis()->SetBinLabel(5,"12.03cm");
  ModuleWidth->SetYTitle("Nr APVPairs");
  // loop over single histograms and extract peak value of charge
  HlistAPVPairs->Sort(); // sort alfabetically
  TIter hiterator(HlistAPVPairs);
  double MeanCharge = 0.;
  double NrOfApvPairs = 0.;
  TH1F *MyHisto = (TH1F*)hiterator();
  while( MyHisto ){
    TString histo_title = MyHisto->GetTitle();
    if(histo_title.Contains("ChargeAPVPair_")){
      std::pair<double,double> two_values = getPeakOfLandau(MyHisto);
      double local_nrofadcs = two_values.first;
      double local_sigma = two_values.second;
      ChargeOfEachAPVPair->Fill(histo_title, local_nrofadcs);
      int ichbin  = ChargeOfEachAPVPair->GetXaxis()->FindBin(histo_title.Data());
      ChargeOfEachAPVPair->SetBinError(ichbin,local_sigma);
      EntriesApvPairs->Fill(histo_title, MyHisto->GetEntries());
      NrOfEntries->Fill(MyHisto->GetEntries());
      if(local_nrofadcs > 0){ // if nr of adcs is negative, the fitting routine could not extract meaningfull numbers
       MeanCharge += local_nrofadcs;
       NrOfApvPairs += 1.; // count nr of apv pairs since do not know whether nr of bins of histogram is the same
      }
    }
    MyHisto = (TH1F*)hiterator();
  }
  ChargeOfEachAPVPair->LabelsDeflate("X"); EntriesApvPairs->LabelsDeflate("X"); // trim nr. of bins to match active labels
  HlistOtherHistos->Add(ChargeOfEachAPVPair);
  HlistOtherHistos->Add(EntriesApvPairs);
  HlistOtherHistos->Add(NrOfEntries); 
  MeanCharge = MeanCharge / NrOfApvPairs;
  // calculate correction
  TH1F* CorrectionOfEachAPVPair = (TH1F*) ChargeOfEachAPVPair->Clone("CorrectionOfEachAPVPair");
  TH1F *ChargeOfEachAPVPairControlView = new TH1F("ChargeOfEachAPVPairControlView","ChargeOfEachAPVPairControlView",1,0,1); ChargeOfEachAPVPairControlView->SetBit(TH1::kCanRebin);
TH1F *CorrectionOfEachAPVPairControlView = new TH1F("CorrectionOfEachAPVPairControlView","CorrectionOfEachAPVPairControlView",1,0,1); CorrectionOfEachAPVPairControlView->SetBit(TH1::kCanRebin);
  std::ofstream APVPairTextOutput("apvpair_corrections.txt");
  APVPairTextOutput<<"# MeanCharge = "<<MeanCharge<<std::endl;
  APVPairTextOutput<<"# Nr. of APVPairs = "<<NrOfApvPairs<<std::endl;
  for(int ibin=1; ibin <= ChargeOfEachAPVPair->GetNbinsX(); ibin++){
     TString local_bin_label = ChargeOfEachAPVPair->GetXaxis()->GetBinLabel(ibin);
     double local_charge_over_path = ChargeOfEachAPVPair->GetBinContent(ibin);
     if(local_bin_label.Contains("ChargeAPVPair_") && local_charge_over_path > 0.0000001){ // calculate correction only for meaningful numbers
       uint32_t extracted_detid; std::istringstream read_label((local_bin_label(14,9)).Data()); read_label >> extracted_detid; 
       unsigned short extracted_apvpairid; std::istringstream read_apvpair((local_bin_label(24,1)).Data()); read_apvpair >> extracted_apvpairid; 
       double local_error_of_charge = ChargeOfEachAPVPair->GetBinError(ibin);
       double local_correction = -0.5;
       double local_error_correction = 0.;
       local_correction = MeanCharge / local_charge_over_path; // later use ExpectedChargeDeposition instead of MeanCharge
       local_error_correction = local_correction * local_error_of_charge / local_charge_over_path;
       if(local_error_correction>1.8){ // understand why error too large sometimes
         std::cout<<"too large error "<<local_error_correction<<" for histogram "<<local_bin_label<<std::endl;
       }
       double nr_of_entries = EntriesApvPairs->GetBinContent(ibin);
       APVPairTextOutput<<local_bin_label<<" "<<local_correction<<" "<<local_charge_over_path<<" "<<nr_of_entries<<std::endl;
       CorrectionOfEachAPVPair->SetBinContent(ibin, local_correction);
       CorrectionOfEachAPVPair->SetBinError(ibin, local_error_correction);
       ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrections"))->Fill(local_correction);
       DetId thedetId = DetId(extracted_detid);
       unsigned int generalized_layer = 0;
       // calculate generalized_layer:  31,32 = TIB1, 33 = TIB2, 33 = TIB3, 51 = TOB1, 52 = TOB2, 60 = TEC
       if(thedetId.subdetId()==StripSubdetector::TIB){
          
          generalized_layer = 10*thedetId.subdetId() + tTopo->tibLayer(thedetId.rawId()) + tTopo->tibStereo(thedetId.rawId());
  	  if(tTopo->tibLayer(thedetId.rawId())==2){
  	    generalized_layer++;
  	    if (tTopo->tibGlued(thedetId.rawId())) edm::LogError("ClusterMTCCFilter")<<"WRONGGGG"<<std::endl;
  	  }
        }else{
          generalized_layer = 10*thedetId.subdetId();
  	  if(thedetId.subdetId()==StripSubdetector::TOB){
  	    
  	    generalized_layer += tTopo->tobLayer(thedetId.rawId());
  	  }
        }
       if(generalized_layer==31){
         ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrectionsTIB1mono"))->Fill(local_correction);
       }
       if(generalized_layer==32){
         ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrectionsTIB1stereo"))->Fill(local_correction);
       }
       if(generalized_layer==33){
        ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrectionsTIB2"))->Fill(local_correction);
       }
       if(generalized_layer==51){
        ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrectionsTOB1"))->Fill(local_correction);
       }
       if(generalized_layer==52){
        ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrectionsTOB2"))->Fill(local_correction);
       }
       // control view
       edm::ESHandle<SiStripDetCabling> siStripDetCabling; eventSetupCopy_->get<SiStripDetCablingRcd>().get(siStripDetCabling);
       const FedChannelConnection& fedchannelconnection = siStripDetCabling->getConnection( extracted_detid, extracted_apvpairid );
       std::ostringstream local_key;
       // in S. Mersi's analysis the APVPair id seems to be used instead of the lldChannel, hence use the same here
       local_key<<"fecCrate"<<fedchannelconnection.fecCrate()<<"_fecSlot"<<fedchannelconnection.fecSlot()<<"_fecRing"<<fedchannelconnection.fecRing()<<"_ccuAddr"<<fedchannelconnection.ccuAddr()<<"_ccuChan"<<fedchannelconnection.ccuChan()<<"_apvPair"<<extracted_apvpairid;
       TString control_key = local_key.str();
       ChargeOfEachAPVPairControlView->Fill(control_key,local_charge_over_path);
       int ibin1  = ChargeOfEachAPVPairControlView->GetXaxis()->FindBin(control_key);
       ChargeOfEachAPVPairControlView->SetBinError(ibin1,local_error_of_charge);
       CorrectionOfEachAPVPairControlView->Fill(control_key, local_correction);
       int ibin2  = CorrectionOfEachAPVPairControlView->GetXaxis()->FindBin(control_key);
       CorrectionOfEachAPVPairControlView->SetBinError(ibin2, local_error_correction);
       // thickness of each module
       double module_thickness = moduleThickness(extracted_detid, eventSetupCopy_);
       if( fabs(module_thickness - 0.032)<0.001 ) ModuleThickness->Fill(1);
       if( fabs(module_thickness - 0.05)<0.001 )  ModuleThickness->Fill(2);
       // width of each module     
       double module_width = moduleWidth(extracted_detid, eventSetupCopy_);
       if(fabs(module_width-6.144)<0.01) ModuleWidth->Fill(1);
       if(fabs(module_width-7.14)<0.01) ModuleWidth->Fill(2);
       if(fabs(module_width-9.3696)<0.01) ModuleWidth->Fill(3);
       if(fabs(module_width-10.49)<0.01) ModuleWidth->Fill(4);
       if(fabs(module_width-12.03)<0.01) ModuleWidth->Fill(5);
     }
  }
  HlistOtherHistos->Add(CorrectionOfEachAPVPair);
  ChargeOfEachAPVPairControlView->LabelsDeflate("X");
  CorrectionOfEachAPVPairControlView->LabelsDeflate("X");
  HlistOtherHistos->Add(ChargeOfEachAPVPairControlView);
  HlistOtherHistos->Add(CorrectionOfEachAPVPairControlView);
  // output histograms to file


  if(outputHistogramsInRootFile){
    TFile *outputfile = new TFile(outputFileName,"RECREATE");
    HlistAPVPairs->Write();
    HlistOtherHistos->Write();
    outputfile->Close();
  }

  SiStripApvGain * obj = new SiStripApvGain();

//   for(std::map<uint32_t,OptoScanAnalysis*>::const_iterator it = analyses.begin(); it != analyses.end(); it++){
//     //Generate Gain for det detid
//     std::vector<float> theSiStripVector;
//     for(unsigned short j=0; j<it->second; j++){
//       float gain;

//       //      if(sigmaGain_/meanGain_ < 0.00001) gain = meanGain_;
//       //      else{
//       gain = CLHEP::RandGauss::shoot(meanGain_, sigmaGain_);
//       if(gain<=minimumPosValue_) gain=minimumPosValue_;
//       //      }

//       if (printdebug_)
// 	edm::LogInfo("SiStripGainCalculator") << "detid " << it->first << " \t"
// 					      << " apv " << j << " \t"
// 					      << gain    << " \t" 
// 					      << std::endl; 	    
//       theSiStripVector.push_back(gain);
//     }
//     SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
//     if ( ! obj->put(it->first,range) )
//       edm::LogError("SiStripGainCalculator")<<"[SiStripGainCalculator::beginJob] detid already exists"<<std::endl;
//   }
  
  return obj;
}

