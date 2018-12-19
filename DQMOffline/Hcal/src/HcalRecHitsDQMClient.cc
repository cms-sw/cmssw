#include "DQMOffline/Hcal/interface/HcalRecHitsDQMClient.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"

HcalRecHitsDQMClient::HcalRecHitsDQMClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{

  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile", "myfile.root");
  debug_ = false;
  verbose_ = false;
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
}


HcalRecHitsDQMClient::~HcalRecHitsDQMClient()
{ 
  
}

void HcalRecHitsDQMClient::beginJob()
{
 

}


void HcalRecHitsDQMClient::beginRun(const edm::Run& run, const edm::EventSetup& es){
   
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  es.get<HcalRecNumberingRecord>().get( pHRNDC );
  hcons = &(*pHRNDC);
  maxDepthHB_ = hcons->getMaxDepth(0);
  maxDepthHE_ = hcons->getMaxDepth(1);
  maxDepthHF_ = hcons->getMaxDepth(2);
  maxDepthHO_ = hcons->getMaxDepth(3);
 
  edm::ESHandle<CaloGeometry> geometry;

  es.get<CaloGeometryRecord > ().get(geometry);
 
  const std::vector<DetId>& hbCells = geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  const std::vector<DetId>& heCells = geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  const std::vector<DetId>& hoCells = geometry->getValidDetIds(DetId::Hcal, HcalOuter);
  const std::vector<DetId>& hfCells = geometry->getValidDetIds(DetId::Hcal, HcalForward);
 
  nChannels_[1] = hbCells.size(); 
  nChannels_[2] = heCells.size(); 
  nChannels_[3] = hoCells.size(); 
  nChannels_[4] = hfCells.size();
  nChannels_[0] = nChannels_[1] + nChannels_[2] + nChannels_[3] + nChannels_[4];
  //avoid divide by zero
  for(unsigned i = 0; i < 5; ++i){
    if(nChannels_[i]==0) nChannels_[i] = 1;
  }
 
  //std::cout << "Channels HB:" << nChannels_[1] << " HE:" << nChannels_[2] << " HO:" << nChannels_[3] << " HF:" << nChannels_[4] << std::endl;
 
  //We hardcode the HF depths because in the dual readout configuration, rechits are not defined for depths 3&4
  maxDepthHF_ = (maxDepthHF_ > 2 ? 2 : maxDepthHF_); //We reatin the dynamic possibility that HF might have 0 or 1 depths
 
  maxDepthAll_ = ( maxDepthHB_ + maxDepthHO_ > maxDepthHE_ ? maxDepthHB_ + maxDepthHO_ : maxDepthHE_ );
  maxDepthAll_ = ( maxDepthAll_ > maxDepthHF_ ? maxDepthAll_ : maxDepthHF_ );
 
}

void HcalRecHitsDQMClient::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter)
{
  igetter.setCurrentFolder(dirName_);

  if (verbose_) std::cout << "\nrunClient" << std::endl; 

  std::vector<MonitorElement*> hcalMEs;

  // Since out folders are fixed to three, we can just go over these three folders
  // i.e., CaloTowersD/CaloTowersTask, HcalRecHitsD/HcalRecHitTask, NoiseRatesV/NoiseRatesTask.
  std::vector<std::string> fullPathHLTFolders = igetter.getSubdirs();
  for(unsigned int i=0;i<fullPathHLTFolders.size();i++) {

    if (verbose_) std::cout <<"\nfullPath: "<< fullPathHLTFolders[i] << std::endl;
    igetter.setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = igetter.getSubdirs();
    for(unsigned int j=0;j<fullSubPathHLTFolders.size();j++) {

      if (verbose_) std::cout <<"fullSub: "<<fullSubPathHLTFolders[j] << std::endl;

      if( strcmp(fullSubPathHLTFolders[j].c_str(), "HcalRecHitsD/HcalRecHitTask") ==0  ){
         hcalMEs = igetter.getContents(fullSubPathHLTFolders[j]);
         if (verbose_) std::cout <<"hltMES size : "<<hcalMEs.size()<<std::endl;
         if( !HcalRecHitsEndjob(hcalMEs) ) std::cout<<"\nError in HcalRecHitsEndjob!"<<std::endl<<std::endl;
      }

    }    

  }

}


// called after entering the HcalRecHitsD/HcalRecHitTask directory
// hcalMEs are within that directory
int HcalRecHitsDQMClient::HcalRecHitsEndjob(const std::vector<MonitorElement*> &hcalMEs){

   MonitorElement* Nhf=nullptr;

   //Search for emap histograms, and collect them into this vector
   //All subdtectors are plotted together in these histograms. We only need to look for different depths
   std::vector<MonitorElement*> emap_depths;

   //This vector is filled occupancy_maps identified by both subdetector and depth
   std::vector<MonitorElement*> occupancy_maps;
   std::vector<std::string> occupancyID;

   //This vector is filled with emean_vs_ieta histograms, they are divided by both subdetector and depth
   std::vector<MonitorElement*> emean_vs_ieta;

   //These are the only histograms filled in this module; however, the histograms are created empty in HcalRecHitsAnalyzer
   //occupancy_vs_ieta, divided by both subdetector and depth
   std::vector<MonitorElement*> occupancy_vs_ieta;
   std::vector<std::string> occupancy_vs_ietaID;

   //RecHit_StatusWord & RecHit_Aux_StatusWord
   //Divided by subdectector
   std::vector<MonitorElement*> RecHit_StatusWord;
   std::vector<float>           RecHit_StatusWord_Channels;
   std::vector<MonitorElement*> RecHit_Aux_StatusWord;
   std::vector<float>           RecHit_Aux_StatusWord_Channels;

   for(unsigned int ih=0; ih<hcalMEs.size(); ih++){

      //N_HF is not special, it is just convient to get the total number of events
      //The number of entries in N_HF is equal to the number of events
      if( hcalMEs[ih]->getName() == "N_HF" ){ 
         Nhf= hcalMEs[ih];
         continue;
      }

      // ***********************
      // * We fill the various MonitorElement vectors by searching for a matching substring
      // * The methods that are used are agnostic to the ordering of vectors
      // ***********************     

      if( hcalMEs[ih]->getName().find("emap_depth") != std::string::npos ){
         emap_depths.push_back(hcalMEs[ih]);
         continue;  
      } 

      if( hcalMEs[ih]->getName().find("occupancy_map_H") != std::string::npos ){
         occupancy_maps.push_back(hcalMEs[ih]);

         // Use occupancyID to save the subdetector and depth information
         // This will help preserve both indifference to vector ordering and specific details of the detector topology
         // The position in occupancyID must correspond to the histogram position in occupancy_maps

         // Save the string after "occupancy_map_"
         
         std::string prefix = "occupancy_map_";

         occupancyID.push_back( hcalMEs[ih]->getName().substr(prefix.size()) );

         continue;
      }

      if( hcalMEs[ih]->getName().find("emean_vs_ieta_H") != std::string::npos ){
         emean_vs_ieta.push_back(hcalMEs[ih]);
         continue;
      }

      if( hcalMEs[ih]->getName().find("occupancy_vs_ieta_H") != std::string::npos ){
         occupancy_vs_ieta.push_back(hcalMEs[ih]);

         // Use occupancy_vs_ietaID to save the subdetector and depth information
         // This will help preserve both indifference to vector ordering and specific details of the detector topology
         // The position in occupancyID must correspond to the histogram position in occupancy_vs_ieta

         // Save the string after "occupancy_vs_ieta_"
         
         std::string prefix = "occupancy_vs_ieta_";

         occupancy_vs_ietaID.push_back( hcalMEs[ih]->getName().substr(prefix.size()) );

         continue;
      }

      if( hcalMEs[ih]->getName().find("HcalRecHitTask_RecHit_StatusWord_H") != std::string::npos ){
         RecHit_StatusWord.push_back(hcalMEs[ih]);

         if(hcalMEs[ih]->getName().find("HB") != std::string::npos ){
            RecHit_StatusWord_Channels.push_back((float)nChannels_[1]);
         }else if(hcalMEs[ih]->getName().find("HE") != std::string::npos ){
            RecHit_StatusWord_Channels.push_back((float)nChannels_[2]);
         }else if(hcalMEs[ih]->getName().find("H0") != std::string::npos ){
            RecHit_StatusWord_Channels.push_back((float)nChannels_[3]);
         }else if(hcalMEs[ih]->getName().find("HF") != std::string::npos ){
            RecHit_StatusWord_Channels.push_back((float)nChannels_[4]);
         } else {
            RecHit_StatusWord_Channels.push_back(1.);
         }

         continue;
      }

      if( hcalMEs[ih]->getName().find("HcalRecHitTask_RecHit_Aux_StatusWord_H") != std::string::npos ){
         RecHit_Aux_StatusWord.push_back(hcalMEs[ih]);

         if(hcalMEs[ih]->getName().find("HB") != std::string::npos ){
            RecHit_Aux_StatusWord_Channels.push_back((float)nChannels_[1]);
         }else if(hcalMEs[ih]->getName().find("HE") != std::string::npos ){
            RecHit_Aux_StatusWord_Channels.push_back((float)nChannels_[2]);
         }else if(hcalMEs[ih]->getName().find("H0") != std::string::npos ){
            RecHit_Aux_StatusWord_Channels.push_back((float)nChannels_[3]);
         }else if(hcalMEs[ih]->getName().find("HF") != std::string::npos ){
            RecHit_Aux_StatusWord_Channels.push_back((float)nChannels_[4]);
         } else {
            RecHit_Aux_StatusWord_Channels.push_back(1.);
         }

         continue;
      }

   } 

   // mean energies and occupancies evaluation

   double nevtot = Nhf->getEntries(); // Use the number of entries in the Nhf histogram to give the total number of events

   if(verbose_) std::cout<<"nevtot : "<<nevtot<<std::endl;

   // emap histograms are scaled by the number of events
   float  fev           = float (nevtot);
   double scaleBynevtot = 1 / fev;

   // In this and the following histogram vectors, recognize that the for-loop index
   // does not have to correspond to any particular depth
   for(unsigned int depthIdx = 0; depthIdx < emap_depths.size(); depthIdx++){

      int nx = emap_depths[depthIdx]->getNbinsX();
      int ny = emap_depths[depthIdx]->getNbinsY();

      float cnorm;
      float enorm;

      for (int i = 1; i <= nx; i++) {      
         for (int j = 1; j <= ny; j++) {
	    cnorm = emap_depths[depthIdx]->getBinContent(i,j) * scaleBynevtot;
	    enorm = emap_depths[depthIdx]->getBinError(i,j) * scaleBynevtot;
            emap_depths[depthIdx]->setBinContent(i,j,cnorm);
            emap_depths[depthIdx]->setBinError(i,j,enorm);

         }
      }
   }

   // occupancy_maps & matched occupancy_vs_ieta

   bool omatched = false;

   for(unsigned int occupancyIdx = 0; occupancyIdx < occupancy_maps.size(); occupancyIdx++){

      int nx = occupancy_maps[occupancyIdx]->getNbinsX();
      int ny = occupancy_maps[occupancyIdx]->getNbinsY();

      float cnorm;
      float enorm;

      unsigned int vsIetaIdx = occupancy_vs_ieta.size();
      omatched = false;

      for(vsIetaIdx = 0; vsIetaIdx < occupancy_vs_ieta.size(); vsIetaIdx++){
         if(occupancyID[occupancyIdx] == occupancy_vs_ietaID[vsIetaIdx]){
            omatched = true;
            break;
         }
      }// match occupancy_vs_ieta histogram

      for (int i = 1; i <= nx; i++) {      
         for (int j = 1; j <= ny; j++) {
	    cnorm = occupancy_maps[occupancyIdx]->getBinContent(i,j) * scaleBynevtot;
	    enorm = occupancy_maps[occupancyIdx]->getBinError(i,j) * scaleBynevtot;
            occupancy_maps[occupancyIdx]->setBinContent(i,j,cnorm);
            occupancy_maps[occupancyIdx]->setBinError(i,j,enorm);

         }
      }

      //Fill occupancy_vs_ieta

      if(omatched){

         //We run over all of the ieta values
         for (int ieta = -41; ieta <= 41; ieta++) {
            float phi_factor = 1.;
            float sumphi = 0.;
	    float sumphie = 0.;

            if(ieta == 0) continue; //ieta=0 is not defined

            phi_factor = phifactor(ieta);

            //the rechits occupancy map defines iphi as 0..71
            for (int iphi = 0; iphi <= 71; iphi++) {
               int binIeta = occupancy_maps[occupancyIdx]->getTH2F()->GetXaxis()->FindBin(float(ieta));
               int binIphi = occupancy_maps[occupancyIdx]->getTH2F()->GetYaxis()->FindBin(float(iphi));

               float content = occupancy_maps[occupancyIdx]->getBinContent(binIeta,binIphi);
	       float econtent = occupancy_maps[occupancyIdx]->getBinError(binIeta,binIphi);

               sumphi += content;
	       sumphie += econtent*econtent;
            }//for loop over phi

	    int ietabin = occupancy_vs_ieta[vsIetaIdx]->getTH1F()->GetXaxis()->FindBin(float(ieta));

	    

            // fill occupancies vs ieta
            cnorm = sumphi / phi_factor;
	    enorm = sqrt(sumphie) / phi_factor;
            occupancy_vs_ieta[vsIetaIdx]->setBinContent(ietabin, cnorm);
	    occupancy_vs_ieta[vsIetaIdx]->setBinError(ietabin,enorm);

         }//Fill occupancy_vs_ieta
      }//if omatched
   }

   // Status Word
   // Normalized by number of events and by number of channels per subdetector as well

   for(unsigned int StatusWordIdx = 0; StatusWordIdx < RecHit_StatusWord.size(); StatusWordIdx++){
         
      int nx = RecHit_StatusWord[StatusWordIdx]->getNbinsX();

      float cnorm;
      float enorm;

      for (int i = 1; i <= nx; i++) {      
         cnorm = RecHit_StatusWord[StatusWordIdx]->getBinContent(i) * scaleBynevtot / RecHit_StatusWord_Channels[StatusWordIdx];
         enorm = RecHit_StatusWord[StatusWordIdx]->getBinError(i) * scaleBynevtot / RecHit_StatusWord_Channels[StatusWordIdx];
         RecHit_StatusWord[StatusWordIdx]->setBinContent(i,cnorm);
         RecHit_StatusWord[StatusWordIdx]->setBinError(i,enorm);

      }

   }

   for(unsigned int AuxStatusWordIdx = 0; AuxStatusWordIdx < RecHit_Aux_StatusWord.size(); AuxStatusWordIdx++){
         
      int nx = RecHit_Aux_StatusWord[AuxStatusWordIdx]->getNbinsX();

      float cnorm;
      float enorm;

      for (int i = 1; i <= nx; i++) {      
         cnorm = RecHit_Aux_StatusWord[AuxStatusWordIdx]->getBinContent(i) * scaleBynevtot / RecHit_Aux_StatusWord_Channels[AuxStatusWordIdx];
         enorm = RecHit_Aux_StatusWord[AuxStatusWordIdx]->getBinError(i) * scaleBynevtot / RecHit_Aux_StatusWord_Channels[AuxStatusWordIdx];
         RecHit_Aux_StatusWord[AuxStatusWordIdx]->setBinContent(i,cnorm);
         RecHit_Aux_StatusWord[AuxStatusWordIdx]->setBinError(i,enorm);

      }


   }

   return 1;
}

float HcalRecHitsDQMClient::phifactor(int ieta){

   float phi_factor_;

   if(ieta >= -20 && ieta <= 20 ){
      phi_factor_ = 72.;
   } else {
      if(ieta >= 40 || ieta <= -40 ){
         phi_factor_ = 18.;
      } else {
         phi_factor_ = 36.;
      }
   }

   return phi_factor_;

}

DEFINE_FWK_MODULE(HcalRecHitsDQMClient);
