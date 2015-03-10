// -*- C++ -*-
//
// 
/**\class EcalCreateTTAvgTimes

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Wed Sep 30 16:29:33 CEST 2009
// $Id: EcalCreateTTAvgTimes.cc,v 1.2 2011/04/29 13:40:37 scooper Exp $
//
//

#include "CalibCalorimetry/EcalTiming/interface/EcalCreateTTAvgTimes.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalCreateTTAvgTimes::EcalCreateTTAvgTimes(const edm::ParameterSet& iConfig) :
  inputFile_ (iConfig.getUntrackedParameter<std::string>("timingCalibFile","")),
  subtractTowerAvgForOfflineCalibs_ (iConfig.getUntrackedParameter<bool>("subtractTowerAvgForOfflineCalibs",true))

{
   //now do what ever initialization is needed
  if(!subtractTowerAvgForOfflineCalibs_)
    edm::LogWarning("EcalCreateTTAvgTimes") << "Not subtracting TT averages from crystals for offline calibrations.";

}


EcalCreateTTAvgTimes::~EcalCreateTTAvgTimes()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalCreateTTAvgTimes::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   // Input is text file in the format "EE/EB"<tab>hash<tab>calib<tab>calib_error

   LogInfo("EcalCreateTTAvgTimes") << "Reading channel statuses from file.";
   ifstream calibFile(inputFile_.c_str());
   if(!calibFile.good())
   {
     LogError("EcalCreateTTAvgTimes") << "*** Problems opening file: " << inputFile_;
     throw cms::Exception ("Cannot open timing calib file");
   }

   double cryCalibsRawEB[61200];
   double cryCalibErrorsRawEB[61200];
   double cryCalibsRawEE[14648];
   double cryCalibErrorsRawEE[14648];
   for(int i=0; i<61200; ++i)
   {
     cryCalibsRawEB[i] = -999;
     cryCalibErrorsRawEB[i] = -999;
   }
   for(int i=0;i<14648;++i)
   {
     cryCalibsRawEE[i] = -999;
     cryCalibErrorsRawEE[i] = -999;
   }

   //double ttSums[54][68];
   //double ttNumEntries[54][68];
   vector<float> times[54][68]; // Each FE has the times for the crys inside
   vector<float> timeErrors[54][68]; // Likewise for the time errors
   int FEcalibs[54][68]; // Keep track of the calculated FE calibs
   int ttEtaCoords[54][68];
   int ttPhiCoords[54][68];
   // Initialize
   for(int i=0;i<54;++i)
   {
     for(int j=0;j<68;++j)
     {
       //ttSums[i][j]=0;
       //ttNumEntries[i][j]=0;
       ttEtaCoords[i][j]=0;
       ttPhiCoords[i][j]=0;
       FEcalibs[i][j]=0;
     }
   }

   std::string EcalSubDet;
   std::string str;
   int hashedIndex(0);
   float calib(0);
   float calibError(0);
   bool makeCalibsEB = false;
   bool makeCalibsEE = false;

   while (!calibFile.eof()) 
   {
     calibFile >> EcalSubDet;
     if(EcalSubDet!=std::string("EB") && EcalSubDet!=std::string("EE"))
     {
       std::getline(calibFile,str);
       continue;
     }
     else
     {
       calibFile >> hashedIndex >> calib >> calibError;
     }
  
     if(EcalSubDet==std::string("EB"))
     {
       EBDetId ebid = EBDetId::unhashIndex(hashedIndex);
       if(ebid==EBDetId())
         LogError("EcalCreateTTAvgTimes") << "Crystal with hashedIndex " << hashedIndex
           << " is not a real crystal.";
       EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(ebid);
       //SIC CHANGE OCT 14 2009
       times[elecId.dccId()-1][elecId.towerId()-1].push_back(calib);
       timeErrors[elecId.dccId()-1][elecId.towerId()-1].push_back(calibError);
       //ttSums[elecId.dccId()-1][elecId.towerId()-1]+=calib;
       cryCalibsRawEB[hashedIndex] = calib;
       cryCalibErrorsRawEB[hashedIndex] = calibError;
       //++ttNumEntries[elecId.dccId()-1][elecId.towerId()-1];
       makeCalibsEB = true;
       //DEBUG
       if(fabs(calib) > 5)
         LogInfo("EcalCreateTTAvgTimes") << "!!!Strange calib (" << calib << ") found for cry: " << ebid;

       if(ttEtaCoords[elecId.dccId()-1][elecId.towerId()-1]==0)
       {
         ttEtaCoords[elecId.dccId()-1][elecId.towerId()-1]=ebid.ieta();
         ttPhiCoords[elecId.dccId()-1][elecId.towerId()-1]=ebid.iphi();
       }
     }
     else if(EcalSubDet==std::string("EE"))
     {
       EEDetId eedetid = EEDetId::unhashIndex(hashedIndex);
       if(eedetid==EEDetId())
         LogError("EcalCreateTTAvgTimes") << "Crystal with hashedIndex " << hashedIndex
           << " is not a real crystal.";
       EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(eedetid);
       //SIC CHANGE OCT 14 2009
       times[elecId.dccId()-1][elecId.towerId()-1].push_back(calib);
       timeErrors[elecId.dccId()-1][elecId.towerId()-1].push_back(calibError);
       //ttSums[elecId.dccId()-1][elecId.towerId()-1]+=calib;
       cryCalibsRawEE[hashedIndex] = calib;
       cryCalibErrorsRawEE[hashedIndex] = calibError;
       //++ttNumEntries[elecId.dccId()-1][elecId.towerId()-1];
       makeCalibsEE = true;
       if(ttEtaCoords[elecId.dccId()-1][elecId.towerId()-1]==0)
       {
         ttEtaCoords[elecId.dccId()-1][elecId.towerId()-1]=eedetid.ix();
         ttPhiCoords[elecId.dccId()-1][elecId.towerId()-1]=eedetid.iy();
       }
       LogInfo("EcalCreateTTAvgTimes") << "done";
     }
     else
     {
       LogWarning("EcalCreateTTAvgTimes") << " *** " << EcalSubDet << " is neither EB nor EE";
     }
   }
   calibFile.close();

    LogInfo("EcalCreateTTAvgTimes") << "Making root file...";
   
    TFile* rootfile;
    if(makeCalibsEB && makeCalibsEE)
      rootfile = new TFile("myFedAvgTimingTreeAll.root","recreate");
    else if(makeCalibsEB)
      rootfile = new TFile("myFedAvgTimingTreeEB.root","recreate");
    else if(makeCalibsEE)
      rootfile = new TFile("myFedAvgTimingTreeEE.root","recreate");
    else
    {
      LogError("EcalCreateTTAvgTimes") << "Neither EE nor EB calibs found; returning.";
      return;
    }

   rootfile->cd();

   int DCCid = 0;
   int towerId = 0;
   float averageTime = 0;
   int ieta = 0;
   int iphi = 0;

   TH1F* calibSWHistEB = new TH1F("calibSWHistEB","Offline calibrations EB;ns",2000,-10,10);
   TH1F* calibSWHistEEM = new TH1F("calibSWHistEEM","Offline calibrations EEM;ns",2000,-10,10);
   TH1F* calibSWHistEEP = new TH1F("calibSWHistEEP","Offline calibrations EEP;ns",2000,-10,10);
   TH2F* calibSWMapEB = new TH2F("calibSWMapEB","time calib map EB [ns];i#phi;i#eta",360,1.,361.,172,-86,86);
   TH2F* calibSWMapEEP = new TH2F("calibSWMapEEP","time calib map EEP [ns];ix;iy",100,1,101,100,1,101);
   TH2F* calibSWMapEEM = new TH2F("calibSWMapEEM","time calib map EEM [ns[;ix;iy",100,1,101,100,1,101);

   TH1F* calibSWErrorHistEB = new TH1F("calibSWErrorHistEB","Offline calibration uncertainty EB;ns",50,0,1);
   TH1F* calibSWErrorHistEEM = new TH1F("calibSWErrorHistEEM","Offline calibration uncertainty EEM;ns",50,0,1);
   TH1F* calibSWErrorHistEEP = new TH1F("calibSWErrorHistEEP","Offline calibration uncertainty EEP;ns",50,0,1);

   //XXX SIC OCT 14 2009
   // FILTER
   for(int i=0; i<54;++i)
   {
     for(int j=0;j<68;++j)
     {
       //if(i==40 && j==55)
       //  verbose=true;
       while(times[i][j].size() > 19) // Must have at least 20 crys left?
       {
         LogInfo("EcalCreateTTAvgTimes") << "FILTER: consider DCC " << i+1 << " tower " << j+1;
         //std::pair<double,double> meanSig = computeWeightedMeanAndSigma(times[i][j],timeErrors[i][j]);
         std::pair<double,double> meanSig = computeUnweightedMeanAndSigma(times[i][j]);
         double mean = meanSig.first;
         //double sigma = meanSig.second;
         //LogInfo("EcalCreateTTAvgTimes") << "Mean of crystal calibs: " << mean << " sigma of crystal calibs: "
         //  << sigma << " mean+1*sigma=" << mean+sigma;
         double maxChi2 = -111;
         int indexOfMaxChi2 = -1;
         for(unsigned int k=0;k < times[i][j].size(); ++k)
         {
           double singleChi = (times[i][j][k]-mean)/timeErrors[i][j][k];
           if(singleChi*singleChi > maxChi2)
           {
             maxChi2 = singleChi*singleChi;
             indexOfMaxChi2 = k;
           }
           //LogInfo("EcalCreateTTAvgTimes") << "crystal: " << k << " chi2: " << singleChi*singleChi << " time: "
           //  << times[i][j][k] << " timeError: " << timeErrors[i][j][k];
         }
         float oldTime = times[i][j][indexOfMaxChi2];
         float oldTimeError = timeErrors[i][j][indexOfMaxChi2];
         //LogInfo("EcalCreateTTAvgTimes") << "Max chi2 found: " << maxChi2 << " with time: " << oldTime
         //  << " and error: " << oldTimeError;
         times[i][j].erase(times[i][j].begin()+indexOfMaxChi2);
         timeErrors[i][j].erase(timeErrors[i][j].begin()+indexOfMaxChi2);
         std::pair<double,double> newMeanSig = computeUnweightedMeanAndSigma(times[i][j]);
         double newMean = newMeanSig.first;
         double newSigma = newMeanSig.second;
         //LogInfo("EcalCreateTTAvgTimes") << "New Mean: " << newMean << " new sigma: " << newSigma
         //    << " fabs(mean-newMean)=" << fabs(mean-newMean) << " 0.5*newSigma=" << 0.5*newSigma;
         if(fabs(mean-newMean) < 0.5*newSigma) // change was minimal
         {
           times[i][j].push_back(oldTime);
           timeErrors[i][j].push_back(oldTimeError);
           break;
         }
           LogInfo("EcalCreateTTAvgTimes") << "Max chi2 cry was rejected.  Loop again. ";
       }
     }
   }

   TTree* mytree = new TTree("fedAvgTree","fedAvgTree");
   mytree->Branch("DCCid",&DCCid,"DCCid/I");
   mytree->Branch("towerId",&towerId,"towerId/I");
   mytree->Branch("averageTime",&averageTime,"averageTime/F");
   mytree->Branch("ieta",&ieta,"ieta/I");
   mytree->Branch("iphi",&iphi,"iphi/I");

   // Print out the averages
   for(int i=0;i<54;++i)
   {
     if(!makeCalibsEE && (i+1 > 45 || i+1 < 10))
       continue;
     if(!makeCalibsEB && i+1 <= 45 && i+1 >= 10)
       continue;
     
     DCCid = i+1;
     ofstream calibFEOutStream;
     string filename = "calibs_FE_dcc_";
     filename+=intToString(DCCid);
     filename+=".txt";
     calibFEOutStream.open(filename.c_str());
     if(!calibFEOutStream.good() || !calibFEOutStream.is_open())
     {
       LogError("EcalCreateTTAvgTimes") << "Couldn't open FE calib output file; exiting.";
       return;
     }
     for(int j=0;j<68;++j)
     {
       if(times[i][j].size() != 0)
       {
         towerId = j+1;
         //averageTime = ttSums[i][j]/ttNumEntries[i][j];
         //averageTime = computeWeightedMeanAndSigma(times[i][j],timeErrors[i][j]).first;
         //XXX SIC NOV 4 2009 UNWEIGHT AVG
         averageTime = computeUnweightedMeanAndSigma(times[i][j]).first;
         ieta = ttEtaCoords[i][j];
         iphi = ttPhiCoords[i][j];
         mytree->Fill();
         //calibFEOutStream << DCCid << "\t" << towerId << "\t" << 
         calibFEOutStream << towerId << "\t" << 
           round(averageTime) << endl;
         FEcalibs[i][j] = (int)round(averageTime);
           //round(ttSums[i][j]/ttNumEntries[i][j]) << endl;
         LogInfo("EcalCreateTTAvgTimes") << "For DCC: " << i+1 << " tower: " << j+1 << " average is: "
           << averageTime << " rounded to: " << round(averageTime);
       }
     }
     calibFEOutStream.close();
   }

   // Make the new SW calib files
   if(makeCalibsEB)
   {
     ofstream calibOutEBStream;
     calibOutEBStream.open("calibsEB_SW.txt");
     if(!calibOutEBStream.good() || !calibOutEBStream.is_open())
     {
       LogError("EcalCreateTTAvgTimes") << "Couldn't open EB SW calib output file; exiting.";
       return;
     }
     //SIC NOV 18 The problem stream is not needed at the moment.
     //ofstream calibOutEBProblemStream;
     //calibOutEBProblemStream.open("calibsEB_SW_problems.txt");
     //if(!calibOutEBProblemStream.good() || !calibOutEBProblemStream.is_open())
     //{
     //  cout << "Couldn't open EB SW calib problems output file; exiting." << endl;
     //  return;
     //}
     for(int i=0; i<61200; ++i)
     {
       EBDetId ebid = EBDetId::unhashIndex(i);
       if(ebid==EBDetId())
         LogError("EcalCreateTTAvgTimes") << "Crystal with EB hashedIndex " << i
           << " is not a real crystal.";
       EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(ebid);
       //if(ttNumEntries[elecId.dccId()-1][elecId.towerId()-1] > 0 && cryCalibsRawEB[i] != -999)
       if(times[elecId.dccId()-1][elecId.towerId()-1].size() > 0 && cryCalibsRawEB[i] != -999)
       {
         //double avgTTtime = round(ttSums[elecId.dccId()-1][elecId.towerId()-1]/ttNumEntries[elecId.dccId()-1][elecId.towerId()-1]);
         //double avgTTtime = computeWeightedMeanAndSigma(times[elecId.dccId()-1][elecId.towerId()-1],timeErrors[elecId.dccId()-1][elecId.towerId()-1]).first;
         //double avgTTtime = computeUnweightedMeanAndSigma(times[elecId.dccId()-1][elecId.towerId()-1]).first;

         int ttTime = FEcalibs[elecId.dccId()-1][elecId.towerId()-1];

         if(subtractTowerAvgForOfflineCalibs_)
         {
           calibOutEBStream << "EB" << "\t" << i << "\t" << cryCalibsRawEB[i]-ttTime
             << "\t\t" << cryCalibErrorsRawEB[i] << endl;
           calibSWHistEB->Fill(cryCalibsRawEB[i]-ttTime);
           calibSWMapEB->Fill(ebid.iphi(),ebid.ieta(),cryCalibsRawEB[i]-ttTime);
         }
         else
         {
           calibOutEBStream << "EB" << "\t" << i << "\t" << cryCalibsRawEB[i]
             << "\t\t" << cryCalibErrorsRawEB[i] << endl;
           calibSWHistEB->Fill(cryCalibsRawEB[i]);
           calibSWMapEB->Fill(ebid.iphi(),ebid.ieta(),cryCalibsRawEB[i]);
         }

         calibSWErrorHistEB->Fill(cryCalibErrorsRawEB[i]);
         // Not needed
         //if(fabs(cryCalibsRawEB[i]-avgTTtime) > 3.5)
         //  calibOutEBProblemStream << "EB" << "\t" << i << "\t" << "ic: " << ebid.ic() << "\t\tIsm: " << ebid.ism() <<
         //    "\t\tCalib: " << cryCalibsRawEB[i]-avgTTtime << endl;

       }
     }
     calibOutEBStream.close();
     //calibOutEBProblemStream.close();
   }
   if(makeCalibsEE)
   {
     ofstream calibOutEEStream;
     calibOutEEStream.open("calibsEE_SW.txt");
     if(!calibOutEEStream.good() || !calibOutEEStream.is_open())
     {
       LogError("EcalCreateTTAvgTimes") << "Couldn't open EE SW calib output file; exiting.";
       return;
     }
     //SIC NOV 18 The problem stream is not needed at the moment.
     //ofstream calibOutEEProblemStream;
     //calibOutEEProblemStream.open("calibsEE_SW_problems.txt");
     //if(!calibOutEEProblemStream.good() || !calibOutEEProblemStream.is_open())
     //{
     //  cout << "Couldn't open EE SW calib output file; exiting." << endl;
     //  return;
     //}
     for(int i=0; i<14648; ++i)
     {
       EEDetId eeid = EEDetId::unhashIndex(i);
       if(eeid==EEDetId())
         LogError("EcalCreateTTAvgTimes") << "Crystal with EE hashedIndex " << i
           << " is not a real crystal.";
       EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(eeid);
       if(times[elecId.dccId()-1][elecId.towerId()-1].size() > 0 && cryCalibsRawEE[i] != -999)
       {
         //double avgTTtime = round(ttSums[elecId.dccId()-1][elecId.towerId()-1]/ttNumEntries[elecId.dccId()-1][elecId.towerId()-1]);
         //double avgTTtime = computeWeightedMeanAndSigma(times[elecId.dccId()-1][elecId.towerId()-1],timeErrors[elecId.dccId()-1][elecId.towerId()-1]).first;
         
         int ttTime = FEcalibs[elecId.dccId()-1][elecId.towerId()-1];
         if(subtractTowerAvgForOfflineCalibs_)
         {
           calibOutEEStream << "EE" << "\t" << i << "\t" << cryCalibsRawEE[i]-ttTime
           << "\t\t" << cryCalibErrorsRawEE[i] << endl;
           if(eeid.zside()>0)
           {
             calibSWMapEEP->Fill(eeid.ix(),eeid.iy(),cryCalibsRawEE[i]-ttTime);
             calibSWHistEEP->Fill(cryCalibsRawEE[i]-ttTime);
           }
           else
           {
             calibSWMapEEM->Fill(eeid.ix(),eeid.iy(),cryCalibsRawEE[i]-ttTime);
             calibSWHistEEM->Fill(cryCalibsRawEE[i]-ttTime);
           }
         }
         else
         {
           calibOutEEStream << "EE" << "\t" << i << "\t" << cryCalibsRawEE[i]
           << "\t\t" << cryCalibErrorsRawEE[i] << endl;
           if(eeid.zside()>0)
           {
             calibSWMapEEP->Fill(eeid.ix(),eeid.iy(),cryCalibsRawEE[i]);
             calibSWHistEEP->Fill(cryCalibsRawEE[i]);
           }
           else
           {
             calibSWMapEEM->Fill(eeid.ix(),eeid.iy(),cryCalibsRawEE[i]);
             calibSWHistEEM->Fill(cryCalibsRawEE[i]);
           }
         }

         if(eeid.zside()>0)
           calibSWErrorHistEEP->Fill(cryCalibErrorsRawEE[i]);
         else
           calibSWErrorHistEEM->Fill(cryCalibErrorsRawEE[i]);

         // Not needed
         //if(fabs(cryCalibsRawEE[i]-avgTTtime) > 3.5)
         //  calibOutEEProblemStream << "EE" << "\t" << i << "\t" << "ix: " << eeid.ix() << "\tiy: "<< eeid.iy() <<
         //    "\t\tCalib: " << cryCalibsRawEE[i]-avgTTtime << endl;
       }
     }
     calibOutEEStream.close();
     //calibOutEEProblemStream.close();
   }
   
   calibSWHistEB->Write();
   calibSWHistEEP->Write();
   calibSWHistEEM->Write();
   calibSWMapEB->Write();
   calibSWMapEEP->Write();
   calibSWMapEEM->Write();
   calibSWErrorHistEB->Write();
   calibSWErrorHistEEM->Write();
   calibSWErrorHistEEP->Write();
   mytree->Write();
   rootfile->Close();
   delete rootfile;

}

// ------------ method to compute unweighted mean and sigma  ----------------------------
const std::pair<double,double> EcalCreateTTAvgTimes::computeUnweightedMeanAndSigma(std::vector<float>& times)
{
  double mean = 0;
  double sigma2 = 0;  //XXX: here defined as "error on the mean"
  for(std::vector<float>::const_iterator timeItr = times.begin(); timeItr != times.end(); ++timeItr)
  {
    float time = *timeItr;
    mean+=time;
  }
  if(mean != 0)
    mean/=times.size();
  else
    return std::make_pair<double,double>(0.,-100.);
    
  for(std::vector<float>::const_iterator timeItr = times.begin(); timeItr != times.end(); ++timeItr)
  {
    float time = *timeItr;
    sigma2+=(time-mean)*(time-mean);

  }
  if(sigma2 > 0)
  {
    sigma2/=times.size();
    double sigma=sqrt(sigma2);
    return std::pair<double,double>(mean,sigma);
  }

  return std::pair<double,double>(0.,-100.);
}

// ------------ method to compute weighted mean and sigma  ------------------------------
const std::pair<double,double> EcalCreateTTAvgTimes::computeWeightedMeanAndSigma(std::vector<float>& times, std::vector<float>& timeErrors)
{
  double mean = 0;
  double sigma = 0;  //XXX: here defined as "error on the mean"
  std::vector<float>::const_iterator timeErrItr = timeErrors.begin();
  for(std::vector<float>::const_iterator timeItr = times.begin(); timeItr != times.end(); ++timeItr)
  {
    float time = *timeItr;
    float error = *timeErrItr;

    mean+=time/(error*error);
    sigma+=1/(error*error);
    ++timeErrItr;
  }
  if(sigma > 0)
  {
    mean/=sigma;  // mean = SUM(t/s^2)/SUM(1/s^2)
    sigma=sqrt(1/sigma);
    return std::pair<double,double>(mean,sigma);
  }

  return std::pair<double,double>(0.,-100.);
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalCreateTTAvgTimes::beginJob(const edm::EventSetup& c)
{
  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  ecalElectronicsMap_ = handle.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalCreateTTAvgTimes::endJob() {
}

// ------------ intToString method  ------------------------------------------------------
std::string EcalCreateTTAvgTimes::intToString(int num)
{ 
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
} 


//define this as a plug-in
DEFINE_FWK_MODULE(EcalCreateTTAvgTimes);
