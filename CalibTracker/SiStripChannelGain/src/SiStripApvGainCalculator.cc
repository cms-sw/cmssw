// -*- C++ -*-
// Package:    SiStripChannelGain
// Class:      SiStripApvGainCalculator
// Original Author:  Dorian Kcira, Pierre Rodeghiero
//         Created:  Mon Nov 20 10:04:31 CET 2006
// $Id$

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibTracker/SiStripChannelGain/interface/SiStripApvGainCalculator.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

//#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"

#include "TFile.h"
#include "TF1.h"
#include "TString.h"
#include <fstream>
#include <sstream>

SiStripApvGainCalculator::SiStripApvGainCalculator(const edm::ParameterSet& iConfig)
{
   conf_ =  iConfig;
   anglefinder_=new  TrackLocalAngle(iConfig);
   ExpectedChargeDeposition = 200.;
   edm::LogInfo("SiStripApvGainCalculator::SiStripApvGainCalculator")<<"ExpectedChargeDeposition="<<ExpectedChargeDeposition;
}


SiStripApvGainCalculator::~SiStripApvGainCalculator()
{
  delete anglefinder_;
}


// ------------ method called to for each event  ------------
void SiStripApvGainCalculator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  total_nr_of_events++;
  anglefinder_->init(iEvent,iSetup);
  // get seeds
  edm::Handle<TrajectorySeedCollection> seedcoll;
  iEvent.getByType(seedcoll);
  // get tracks
  Handle<reco::TrackCollection> trackCollection; iEvent.getByLabel(TrackProducer, TrackLabel, trackCollection);
  const reco::TrackCollection *tracks=trackCollection.product();
//  // get magnetic field
//  edm::ESHandle<MagneticField> esmagfield;
//  es.get<IdealMagneticFieldRecord>().get(esmagfield);
//  magfield=&(*esmagfield);
  // loop over tracks
  for(reco::TrackCollection::const_iterator itr = tracks->begin(); itr != tracks->end(); itr++){ // looping over tracks
    std::vector<std::pair<const TrackingRecHit *,float> >hitangle =anglefinder_->findtrackangle((*(*seedcoll).begin()),*itr);
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
        const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=sistripsimplehit->cluster();
        const std::vector<short>& ampls = cluster->amplitudes();
        uint32_t thedetid  = cluster->geographicalId();
        double module_width = moduleWidth(thedetid, &iSetup);
        ((TH1F*) HlistOtherHistos->FindObject("LocalPosition_cm"))->Fill(local_position.x());
        ((TH1F*) HlistOtherHistos->FindObject("LocalPosition_normalized"))->Fill(local_position.x()/module_width);
        double module_thickness = moduleThickness(thedetid, &iSetup);
        int ifirststrip= cluster->firstStrip();
        int theapvpairid = int(float(ifirststrip)/256.);
        std::ostringstream oshistoid; oshistoid.str(""); oshistoid << "ChargeAPVPair_" << thedetid << "_" << theapvpairid; TString histoid = oshistoid.str();
        TH1F* histopointer = (TH1F*) HlistAPVPairs->FindObject(histoid);
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

//     for(TrackingRecHitRefVector::iterator irech = itr->recHitsBegin(); irech != itr->recHitsEnd(); irech++){ // looping over TrackingRecHits of a track
//       const SiStripRecHit2D* sistripsimplehit=dynamic_cast<const SiStripRecHit2D*>(&**irech);
//       const SiStripMatchedRecHit2D* sistripmatchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&**irech);
//       std::cout<<" hit/matched "<<std::ios::hex<<sistripsimplehit<<" "<<sistripmatchedhit<<std::endl;
//     }

// ------------ method called once each job just before starting event loop  ------------
void SiStripApvGainCalculator::beginJob(const edm::EventSetup& iSetup)
{
   eventSetupCopy_ = &iSetup;
   std::cout<<"SiStripApvGainCalculator::beginJob called"<<std::endl;
   total_nr_of_events = 0;
   //
   detModulesToBeExcluded.clear();
   detModulesToBeExcluded = conf_.getParameter< std::vector<unsigned> >("detModulesToBeExcluded");
   edm::LogInfo("SiStripApvGainCalculator")<<"Clusters from "<<detModulesToBeExcluded.size()<<" modules will be ignored in the calibration:";
   edm::LogInfo("SiStripApvGainCalculator")<<"The calibration for these DetIds will be set to a default value";
   for( std::vector<uint32_t>::const_iterator imod = detModulesToBeExcluded.begin(); imod != detModulesToBeExcluded.end(); imod++){
     edm::LogInfo("SiStripApvGainCalculator")<<"exclude detid = "<< *imod;
   }
   //
   HlistAPVPairs = new TObjArray();
   HlistOtherHistos = new TObjArray();
   //
   std::ostringstream oshistoid; TString histoid;
   oshistoid.str(""); oshistoid << "APVPairCorrections";        histoid=oshistoid.str(); HlistOtherHistos->Add(new TH1F(histoid,histoid,200,-1.,4.));
   oshistoid.str(""); oshistoid << "LocalAngle";        histoid=oshistoid.str(); HlistOtherHistos->Add(new TH1F(histoid,histoid,70,-0.1,3.4));
   oshistoid.str(""); oshistoid << "LocalAngleAbsoluteCosine";   histoid=oshistoid.str(); HlistOtherHistos->Add(new TH1F(histoid,histoid,48,-0.1,1.1));
   oshistoid.str(""); oshistoid << "LocalPosition_cm";     histoid=oshistoid.str(); HlistOtherHistos->Add(new TH1F(histoid,histoid,100,-5.,5.));
   oshistoid.str(""); oshistoid << "LocalPosition_normalized";     histoid=oshistoid.str(); HlistOtherHistos->Add(new TH1F(histoid,histoid,100,-1.1,1.1));
   //
   oshistoid.str(""); oshistoid << "SiStripRecHitType"; histoid=oshistoid.str();
   TH1F * local_histo = new TH1F(histoid,histoid,2,0.5,2.5); HlistOtherHistos->Add(local_histo);
   local_histo->GetXaxis()->SetBinLabel(1,"simple"); local_histo->GetXaxis()->SetBinLabel(2,"matched");
   //
   TrackProducer =  conf_.getParameter<std::string>("TrackProducer");
   TrackLabel    =  conf_.getParameter<std::string>("TrackLabel");
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
           edm::LogError("SiStripApvGainCalculator")<<"Problem with Number of strips in detector: "<<p.nstrips()<<" Exiting program";
           exit(1);
         }
         for(int iapp = 0; iapp<NAPVPairs; iapp++){
           oshistoid.str(""); oshistoid << "ChargeAPVPair_" << detid << "_" << iapp; histoid = oshistoid.str();
//           HlistAPVPairs->Add(new TH1F(histoid,histoid,45,0.,450.));
           HlistAPVPairs->Add(new TH1F(histoid,histoid,45,0.,1350.)); // multiply by 3 to take into account division by width
         }
       }
     }
   }
}


// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripApvGainCalculator::endJob() {
  std::cout<<"SiStripApvGainCalculator::endJob called"<<std::endl;
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
  ModuleWidth->GetXaxis()->SetBinLabel(5,"12.03");
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
      double local_nrofadcs = getPeakOfLandau( MyHisto );
      ChargeOfEachAPVPair->Fill(histo_title, local_nrofadcs);
      EntriesApvPairs->Fill(histo_title, MyHisto->GetEntries());
      NrOfEntries->Fill(MyHisto->GetEntries());
      MeanCharge += local_nrofadcs;
      NrOfApvPairs += 1.; // count nr of apv pairs since do not know whether nr of bins of histogram is the same
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
  std::ofstream APVPairTextOutput("apvpair_corrections.txt");
  APVPairTextOutput<<"# MeanCharge = "<<MeanCharge<<std::endl;
  APVPairTextOutput<<"# Nr. of APVPairs = "<<NrOfApvPairs<<std::endl;
  for(int ibin=1; ibin <= ChargeOfEachAPVPair->GetNbinsX(); ibin++){
     TString local_bin_label = ChargeOfEachAPVPair->GetXaxis()->GetBinLabel(ibin);
     if(local_bin_label.Contains("ChargeAPVPair_")){
       uint32_t extracted_detid; std::istringstream read_label((local_bin_label(14,9)).Data()); read_label >> extracted_detid; 
       unsigned short extracted_apvpairid; std::istringstream read_apvpair((local_bin_label(24,1)).Data()); read_apvpair >> extracted_apvpairid; 
       double local_charge_over_path = ChargeOfEachAPVPair->GetBinContent(ibin);
       double local_correction = -0.5;
       if(local_charge_over_path > 0.0000001){
          local_correction = MeanCharge / local_charge_over_path; // later use ExpectedChargeDeposition instead of MeanCharge
       }
       double nr_of_entries = EntriesApvPairs->GetBinContent(ibin);
       APVPairTextOutput<<local_bin_label<<" "<<local_correction<<" "<<local_charge_over_path<<" "<<nr_of_entries<<std::endl;
       CorrectionOfEachAPVPair->SetBinContent(ibin, local_correction);
       ((TH1F*) HlistOtherHistos->FindObject("APVPairCorrections"))->Fill(local_correction);
       // control view
       edm::ESHandle<SiStripDetCabling> siStripDetCabling; eventSetupCopy_->get<SiStripDetCablingRcd>().get(siStripDetCabling);
       const FedChannelConnection& fedchannelconnection = siStripDetCabling->getConnection( extracted_detid, extracted_apvpairid );

/*
      // this is the method to be used for newer CMSSW releases, at least from CMSSW_1_2_x on:w
       uint32_t n_controlkey = SiStripFecKey::key(
                               fedchannelconnection.fecCrate(),
        		       fedchannelconnection.fecSlot(),
        		       fedchannelconnection.fecRing(),
        		       fedchannelconnection.ccuAddr(),
        		       fedchannelconnection.ccuChan(),
        		       fedchannelconnection.lldChannel()
                               );
*/

/*
      // this is the method to be used for CMSSW_1_0_4
      uint32_t n_controlkey = SiStripGenerateKey::controlKey(
                                fedchannelconnection.fecCrate(),
                                fedchannelconnection.fecSlot(),
                                fedchannelconnection.fecRing(),
                                fedchannelconnection.ccuAddr(),
                                fedchannelconnection.ccuChan(),
        		       fedchannelconnection.lldChannel()
                                );
       std::ostringstream local_key;  local_key << n_controlkey; TString control_key= local_key.str();
       std::cout<<"control_key="<<control_key<<"  detid="<<extracted_detid<<" apvpair="<<extracted_apvpairid<<std::endl;
*/

       std::cout<<"   fecCrate="<<fedchannelconnection.fecCrate()<<"  fecSlot="<<fedchannelconnection.fecSlot()<<" fecRing="<<fedchannelconnection.fecRing()<<" ccuAddr="<<fedchannelconnection.ccuAddr()<<" ccuChan="<<fedchannelconnection.ccuChan()<<" lldChannel="<<fedchannelconnection.lldChannel()<<std::endl;
       std::ostringstream local_key;
       // in S. Mersi's analysis the APVPair id seems to be used instead of the lldChannel, hence use the same here
       local_key<<"fecCrate"<<fedchannelconnection.fecCrate()<<"_fecSlot"<<fedchannelconnection.fecSlot()<<"_fecRing"<<fedchannelconnection.fecRing()<<"_ccuAddr"<<fedchannelconnection.ccuAddr()<<"_ccuChan"<<fedchannelconnection.ccuChan()<<"_apvPair"<<extracted_apvpairid;
       TString control_key = local_key.str();
       ChargeOfEachAPVPairControlView->Fill(control_key,local_charge_over_path);
       // thickness of each module
       double module_thickness = moduleThickness(extracted_detid, eventSetupCopy_);
       if( abs(module_thickness - 0.032)<0.001 ) ModuleThickness->Fill(1);
       if( abs(module_thickness - 0.05)<0.001 )  ModuleThickness->Fill(2);
       // width of each module     
       double module_width = moduleWidth(extracted_detid, eventSetupCopy_);
       if(abs(module_width-6.144)<0.01) ModuleWidth->Fill(1);
       if(abs(module_width-7.14)<0.01) ModuleWidth->Fill(2);
       if(abs(module_width-9.3696)<0.01) ModuleWidth->Fill(3);
       if(abs(module_width-10.49)<0.01) ModuleWidth->Fill(4);
       if(abs(module_width-12.03)<0.01) ModuleWidth->Fill(5);
     }
  }
  HlistOtherHistos->Add(CorrectionOfEachAPVPair);
  ChargeOfEachAPVPairControlView->LabelsDeflate("X");
  HlistOtherHistos->Add(ChargeOfEachAPVPairControlView);
  // output histograms to file
  bool outputHistogramsInRootFile = conf_.getParameter<bool>("OutputHistogramsInRootFile");
  TString outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputHistogramsInRootFile){
    TFile *outputfile = new TFile(outputFileName,"RECREATE");
    HlistAPVPairs->Write();
    HlistOtherHistos->Write();
    outputfile->Close();
  }
}

//-------- automated fitting with finding of the appropriate nr. of ADCs
double SiStripApvGainCalculator::getPeakOfLandau( TH1F * inputHisto ){
  double adcs = 0.;
  double nr_of_entries = inputHisto->GetEntries();
  // set some default value and return if no entries
  if(nr_of_entries == 0){
    adcs = -0.5;  // default dummy value
    return adcs;
  }
  double mean_of_histogram = inputHisto->GetMean();
  if( nr_of_entries < 20.){ // get mean if less than 20 entries
     adcs = mean_of_histogram;
  } else {
//    // fit with initial setting of  parameter values
//    double rms_of_histogram = inputHisto->GetRMS();
//    TF1 *landaufit = new TF1("landaufit","landau",0.,450.);
//    landaufit->SetParameters(nr_of_entries,mean_of_histogram,rms_of_histogram);
//    inputHisto->Fit("landaufit","0Q+");
//    delete landaufit;
    // perform fit with standard landau
    inputHisto->Fit("landau","0Q+");
    TF1 * fitfunction = (TF1*) inputHisto->GetListOfFunctions()->First();
    adcs = fitfunction->GetParameter("MPV");
    if(adcs< 2 ) adcs = mean_of_histogram;
  }
  return adcs;
}

//-------- get width of the module detid
double SiStripApvGainCalculator::moduleWidth(const uint32_t detid, const edm::EventSetup* iSetup)
{ //dk: copied from A. Giammanco and hacked
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );     
  double module_width=0.;
  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
    std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
  }else{
    module_width = it->surface().bounds().width();
  }
  return module_width;
/*
     module_width=10.49
     module_width=12.03
     module_width=6.144
     module_width=7.14
     module_width=9.3696
*/
}

//-------- get thickness of the module detid
double SiStripApvGainCalculator::moduleThickness(const uint32_t detid, const edm::EventSetup* iSetup)
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


//define this as a plug-in
DEFINE_FWK_MODULE(SiStripApvGainCalculator)
