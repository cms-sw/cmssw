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

   MonitorElement* Nhf=0;

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
   std::vector<MonitorElement*> RecHit_Aux_StatusWord;

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
         continue;
      }

      if( hcalMEs[ih]->getName().find("HcalRecHitTask_RecHit_Aux_StatusWord_H") != std::string::npos ){
         RecHit_Aux_StatusWord.push_back(hcalMEs[ih]);
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

      for (int i = 1; i <= nx; i++) {      
         for (int j = 1; j <= ny; j++) {
	    cnorm = emap_depths[depthIdx]->getBinContent(i,j) * scaleBynevtot;
            emap_depths[depthIdx]->setBinContent(i,j,cnorm);

         }
      }
   }

   // occupancy_maps & matched occupancy_vs_ieta

   bool oMatched = false;

   for(unsigned int occupancyIdx = 0; occupancyIdx < occupancy_maps.size(); occupancyIdx++){

      int nx = occupancy_maps[occupancyIdx]->getNbinsX();
      int ny = occupancy_maps[occupancyIdx]->getNbinsY();

      float cnorm;

      for (int i = 1; i <= nx; i++) {      
         for (int j = 1; j <= ny; j++) {
	    cnorm = occupancy_maps[occupancyIdx]->getBinContent(i,j) * scaleBynevtot;
            occupancy_maps[occupancyIdx]->setBinContent(i,j,cnorm);

         }
      }

   }

   // Status Word
   // Previously these histograms were normalized by number of channels per subdetector as well

   for(unsigned int StatusWordIdx = 0; StatusWordIdx < RecHit_StatusWord.size(); StatusWordIdx++){
         
      int nx = RecHit_StatusWord[StatusWordIdx]->getNbinsX();

      float cnorm;

      for (int i = 1; i <= nx; i++) {      
         cnorm = RecHit_StatusWord[StatusWordIdx]->getBinContent(i) * scaleBynevtot;
         RecHit_StatusWord[StatusWordIdx]->setBinContent(i,cnorm);

      }

   }

   for(unsigned int AuxStatusWordIdx = 0; AuxStatusWordIdx < RecHit_Aux_StatusWord.size(); AuxStatusWordIdx++){
         
      int nx = RecHit_Aux_StatusWord[AuxStatusWordIdx]->getNbinsX();

      float cnorm;

      for (int i = 1; i <= nx; i++) {      
         cnorm = RecHit_Aux_StatusWord[AuxStatusWordIdx]->getBinContent(i) * scaleBynevtot;
         RecHit_Aux_StatusWord[AuxStatusWordIdx]->setBinContent(i,cnorm);

      }


   }

   return 1;
}

float HcalRecHitsDQMClient::phifactor(float ieta){

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
