#include <iostream>
//

#include "DQMOffline/EGamma/plugins/PhotonOfflineClient.h"


//#define TWOPI 6.283185308
//

/** \class PhotonOfflineClient
 **
 **
 **  $Id: PhotonOfflineClient
 **  authors:
 **   Nancy Marinelli, U. of Notre Dame, US
 **   Jamie Antonelli, U. of Notre Dame, US
 **
 ***/



using namespace std;
using std::cout;

PhotonOfflineClient::PhotonOfflineClient(const edm::ParameterSet& pset)
{

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  parameters_ = pset;

  analyzerName_    = pset.getParameter<string>("analyzerName");
  cutStep_       = pset.getParameter<double>("cutStep");
  numberOfSteps_ = pset.getParameter<int>("numberOfSteps");

  etMin  = pset.getParameter<double>("etMin");
  etMax  = pset.getParameter<double>("etMax");
  etBin  = pset.getParameter<int>("etBin");
  etaMin = pset.getParameter<double>("etaMin");
  etaMax = pset.getParameter<double>("etaMax");
  etaBin = pset.getParameter<int>("etaBin");
  phiMin = pset.getParameter<double>("phiMin");
  phiMax = pset.getParameter<double>("phiMax");
  phiBin = pset.getParameter<int>("phiBin");

  standAlone_ = pset.getParameter<bool>("standAlone");
  batch_ = pset.getParameter<bool>("batch");
  excludeBkgHistos_ =  pset.getParameter<bool>("excludeBkgHistos");

  outputFileName_ = pset.getParameter<string>("OutputFileName");
  inputFileName_  = pset.getUntrackedParameter<string>("InputFileName");

  histo_index_photons_ = 0;
  histo_index_conversions_ = 0;
  histo_index_efficiency_ = 0;
  histo_index_invMass_ = 0;

  types_.push_back("All");
  types_.push_back("GoodCandidate");
  if (!excludeBkgHistos_)  types_.push_back("Background");

  parts_.push_back("AllEcal");
  parts_.push_back("Barrel");
  parts_.push_back("Endcaps");

}

PhotonOfflineClient::~PhotonOfflineClient()
{}
void PhotonOfflineClient::beginJob()
{}
void PhotonOfflineClient::analyze(const edm::Event& e, const edm::EventSetup& esup)
{}
void PhotonOfflineClient::endLuminosityBlock( const edm::LuminosityBlock& , const edm::EventSetup& setup)
{}

void PhotonOfflineClient::endJob()
{
  if(standAlone_) runClient();
}
void PhotonOfflineClient::endRun(const edm::Run& run, const edm::EventSetup& setup)
{
  if(!standAlone_) runClient();
}

void PhotonOfflineClient::runClient()
{
  if(!dbe_) return;

  if(batch_)  dbe_->open(inputFileName_);

  //std::cout << " PostProcessing analyzer name " << analyzerName_ << std::endl;
  if(!dbe_->dirExists("Egamma/"+analyzerName_)){
    std::cout << "Folder Egamma/"+analyzerName_+" does not exist - Abort the efficiency calculation "  << std::endl;
    return;
  }

  //find out how many histograms are in the various folders
  histo_index_photons_     = dbe_->get("Egamma/"+analyzerName_+"/numberOfHistogramsInPhotonsFolder")->getIntValue();
  histo_index_conversions_ = dbe_->get("Egamma/"+analyzerName_+"/numberOfHistogramsInConversionsFolder")->getIntValue();
  histo_index_efficiency_  = dbe_->get("Egamma/"+analyzerName_+"/numberOfHistogramsInEfficiencyFolder")->getIntValue();
  histo_index_invMass_     = dbe_->get("Egamma/"+analyzerName_+"/numberOfHistogramsInInvMassFolder")->getIntValue();

  dbe_->setCurrentFolder("Egamma/"+analyzerName_+"/");
  dbe_->removeElement("numberOfHistogramsInPhotonsFolder");
  dbe_->removeElement("numberOfHistogramsInConversionsFolder");
  dbe_->removeElement("numberOfHistogramsInEfficiencyFolder");
  dbe_->removeElement("numberOfHistogramsInInvMassFolder");

  string AllPath    = "Egamma/"+analyzerName_+"/AllPhotons/";
  string IsoPath    = "Egamma/"+analyzerName_+"/GoodCandidatePhotons/";
  string NonisoPath = "Egamma/"+analyzerName_+"/BackgroundPhotons/";
  string EffPath    = "Egamma/"+analyzerName_+"/Efficiencies/";

  //booking efficiency histograms
  dbe_->setCurrentFolder(EffPath);

  p_efficiencyVsEtaLoose_ = bookHisto("EfficiencyVsEtaLoose","Fraction of Photons passing Loose Isolation vs #eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtLoose_  = bookHisto("EfficiencyVsEtLoose", "Fraction of Photons passing Loose Isolation vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);
  p_efficiencyVsEtaTight_ = bookHisto("EfficiencyVsEtaTight","Fraction of Photons passing Tight Isolation vs #eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtTight_  = bookHisto("EfficiencyVsEtTight", "Fraction of Photons passing Tight Isolation vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);

  p_efficiencyVsEtaHLT_ = bookHisto("EfficiencyVsEtaHLT","Fraction of Photons firing HLT vs #eta;#eta",etaBin,etaMin, etaMax);
  p_efficiencyVsEtHLT_  = bookHisto("EfficiencyVsEtHLT", "Fraction of Photons firing HLT vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);

  p_convFractionVsEtaLoose_ = bookHisto("ConvFractionVsEtaLoose","Fraction of Loosely Isolated Photons which are matched to two tracks vs #eta;#eta",etaBin,etaMin, etaMax);
  p_convFractionVsEtLoose_  = bookHisto("ConvFractionVsEtLoose", "Fraction of Loosely Isolated Photons which are matched to two tracks vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);
  p_convFractionVsEtaTight_ = bookHisto("ConvFractionVsEtaTight","Fraction of Tightly Isolated Photons which are matched to two tracks vs #eta;#eta",etaBin,etaMin, etaMax);
  p_convFractionVsEtTight_  = bookHisto("ConvFractionVsEtTight", "Fraction of Tightly Isolated Photons which are matched to two tracks vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);

  p_vertexReconstructionEfficiencyVsEta_ = bookHisto("VertexReconstructionEfficiencyVsEta","Fraction of Converted Photons which have a valid vertex vs #eta;#eta",etaBin,etaMin, etaMax);

  //booking conversion fraction histograms
  dbe_->setCurrentFolder(AllPath+"Et above 20 GeV/Conversions");
  p_convFractionVsEt_  = book2DHistoVector("1D","convFractionVsEt", "Fraction of Converted Photons vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);
  p_convFractionVsPhi_ = book3DHistoVector("1D","convFractionVsPhi","Fraction of Converted Photons vs #phi;#phi",phiBin,phiMin,phiMax);
  p_convFractionVsEta_ = book2DHistoVector("1D","convFractionVsEta","Fraction of Converted Photons vs #eta;#eta",etaBin,etaMin,etaMax);

  //booking bad channel fraction histograms
  dbe_->setCurrentFolder(AllPath+"Et above 20 GeV/");
  p_badChannelsFractionVsPhi_ = book2DHistoVector("1D","badChannelsFractionVsPhi","Fraction of Photons which have at least one bad channel vs #phi;#phi",phiBin,phiMin,phiMax);
  p_badChannelsFractionVsEta_ = book2DHistoVector("1D","badChannelsFractionVsEta","Fraction of Photons which have at least one bad channel vs #eta;#eta",etaBin,etaMin, etaMax);
  p_badChannelsFractionVsEt_  = book2DHistoVector("1D","badChannelsFractionVsEt", "Fraction of Photons which have at least one bad channel vs E_{T};E_{T} (GeV)",etBin,etMin,etMax);

  //making efficiency plots
  MonitorElement * dividend;
  MonitorElement * numerator;
  MonitorElement * denominator;

  currentFolder_.str("");
  currentFolder_ << AllPath << "Et above 20 GeV/";

  //HLT efficiency plots
  dividend    = retrieveHisto(EffPath,"EfficiencyVsEtaHLT");
  numerator   = retrieveHisto(EffPath,"phoEtaPostHLT");
  denominator = retrieveHisto(EffPath,"phoEtaPreHLT");
  dividePlots(dividend,numerator,denominator);

  dividend    = retrieveHisto(EffPath,"EfficiencyVsEtHLT");
  numerator   = retrieveHisto(EffPath,"phoEtPostHLT");
  denominator = retrieveHisto(EffPath,"phoEtPreHLT");
  dividePlots(dividend,numerator,denominator);

  //efficiencies vs Eta
  denominator = retrieveHisto(currentFolder_.str(),"phoEta");

  dividend    = retrieveHisto(EffPath,"EfficiencyVsEtaLoose");
  numerator   = retrieveHisto(EffPath,"phoEtaLoose");
  dividePlots(dividend,numerator,denominator);

  dividend    = retrieveHisto(EffPath,"EfficiencyVsEtaTight");
  numerator   = retrieveHisto(EffPath,"phoEtaTight");
  dividePlots(dividend,numerator,denominator);

  //efficiencies vs Et
  denominator = retrieveHisto(currentFolder_.str(),"phoEtAllEcal");

  dividend    = retrieveHisto(EffPath,"EfficiencyVsEtLoose");
  numerator   = retrieveHisto(EffPath,"phoEtLoose");
  dividePlots(dividend,numerator,denominator);

  dividend    = retrieveHisto(EffPath,"EfficiencyVsEtTight");
  numerator   = retrieveHisto(EffPath,"phoEtTight");
  dividePlots(dividend,numerator,denominator);

  //conversion fractions vs Eta
  dividend    = retrieveHisto(EffPath,"ConvFractionVsEtaLoose");
  numerator   = retrieveHisto(EffPath,"convEtaLoose");
  denominator = retrieveHisto(EffPath,"phoEtaLoose");
  dividePlots(dividend,numerator,denominator);

  dividend    = retrieveHisto(EffPath,"ConvFractionVsEtaTight");
  numerator   = retrieveHisto(EffPath,"convEtaTight");
  denominator = retrieveHisto(EffPath,"phoEtaTight");
  dividePlots(dividend,numerator,denominator);

  //conversion fractions vs Et
  dividend    = retrieveHisto(EffPath,"ConvFractionVsEtLoose");
  numerator   = retrieveHisto(EffPath,"convEtLoose");
  denominator = retrieveHisto(EffPath,"phoEtLoose");
  dividePlots(dividend,numerator,denominator);

  dividend    = retrieveHisto(EffPath,"ConvFractionVsEtTight");
  numerator   = retrieveHisto(EffPath,"convEtTight");
  denominator = retrieveHisto(EffPath,"phoEtTight");
  dividePlots(dividend,numerator,denominator);

  //conversion vertex recontruction efficiency
  dividend    = retrieveHisto(EffPath,"VertexReconstructionEfficiencyVsEta");
  numerator   = retrieveHisto(currentFolder_.str()+"Conversions/","phoConvEta");
  denominator = retrieveHisto(EffPath,"phoEtaVertex");
  dividePlots(dividend,numerator,denominator);

  dbe_->setCurrentFolder(EffPath);

  dbe_->removeElement("phoEtaPreHLT");
  dbe_->removeElement("phoEtPreHLT");
  dbe_->removeElement("phoEtaPostHLT");
  dbe_->removeElement("phoEtPostHLT");
  dbe_->removeElement("phoEtaLoose");
  dbe_->removeElement("phoEtaTight");
  dbe_->removeElement("phoEtLoose");
  dbe_->removeElement("phoEtTight");
  dbe_->removeElement("phoEtaVertex");
  dbe_->removeElement("convEtaLoose");
  dbe_->removeElement("convEtaTight");
  dbe_->removeElement("convEtLoose");
  dbe_->removeElement("convEtTight");

  for(uint type=0;type!=types_.size();++type){

    for (int cut = 0; cut !=numberOfSteps_; ++cut) {

      currentFolder_.str("");
      currentFolder_ << "Egamma/"+analyzerName_+"/" << types_[type] << "Photons/Et above " << (cut+1)*cutStep_ << " GeV/";

      //making bad channel histograms

      //vs Et
      dividend    = retrieveHisto(currentFolder_.str(),"badChannelsFractionVsEt");
      numerator   = retrieveHisto(currentFolder_.str(),"phoEtBadChannels");
      denominator = retrieveHisto(currentFolder_.str(),"phoEtAllEcal");
      dividePlots(dividend,numerator,denominator);

      //vs eta
      dividend    = retrieveHisto(currentFolder_.str(),"badChannelsFractionVsEta");
      numerator   = retrieveHisto(currentFolder_.str(),"phoEtaBadChannels");
      denominator = retrieveHisto(currentFolder_.str(),"phoEta");
      dividePlots(dividend,numerator,denominator);

      //vs phi
      dividend    = retrieveHisto(currentFolder_.str(),"badChannelsFractionVsPhi");
      numerator   = retrieveHisto(currentFolder_.str(),"phoPhiBadChannels");
      denominator = retrieveHisto(currentFolder_.str(),"phoPhiAllEcal");
      dividePlots(dividend,numerator,denominator);

      //making conversion fraction histograms

      //vs Et
      dividend    = retrieveHisto(currentFolder_.str()+"Conversions/","convFractionVsEt");
      numerator   = retrieveHisto(currentFolder_.str()+"Conversions/","phoConvEtAllEcal");
      denominator = retrieveHisto(currentFolder_.str(),"phoEtAllEcal");
      dividePlots(dividend,numerator,denominator);

      //vs eta
      dividend    = retrieveHisto(currentFolder_.str()+"Conversions/","convFractionVsEta");
      numerator   = retrieveHisto(currentFolder_.str()+"Conversions/","phoConvEtaForEfficiency");
      denominator = retrieveHisto(currentFolder_.str(),"phoEta");
      dividePlots(dividend,numerator,denominator);

      //vs phi
      dividend    = retrieveHisto(currentFolder_.str()+"Conversions/","convFractionVsPhiAllEcal");
      numerator   = retrieveHisto(currentFolder_.str()+"Conversions/","phoConvPhiForEfficiencyAllEcal");
      denominator = retrieveHisto(currentFolder_.str(),"phoPhiAllEcal");
      dividePlots(dividend,numerator,denominator);
      dividend    = retrieveHisto(currentFolder_.str()+"Conversions/","convFractionVsPhiBarrel");
      numerator   = retrieveHisto(currentFolder_.str()+"Conversions/","phoConvPhiForEfficiencyBarrel");
      denominator = retrieveHisto(currentFolder_.str(),"phoPhiBarrel");
      dividePlots(dividend,numerator,denominator);
      dividend    = retrieveHisto(currentFolder_.str()+"Conversions/","convFractionVsPhiEndcaps");
      numerator   = retrieveHisto(currentFolder_.str()+"Conversions/","phoConvPhiForEfficiencyEndcaps");
      denominator = retrieveHisto(currentFolder_.str(),"phoPhiEndcaps");
      dividePlots(dividend,numerator,denominator);


      dbe_->setCurrentFolder(currentFolder_.str()+"Conversions/");
      dbe_->removeElement("phoConvEtaForEfficiency");
      dbe_->removeElement("phoConvPhiForEfficiencyAllEcal");
      dbe_->removeElement("phoConvPhiForEfficiencyBarrel");
      dbe_->removeElement("phoConvPhiForEfficiencyEndcaps");
    }
  }  
  
  if(standAlone_) dbe_->save(outputFileName_);
  else if(batch_) dbe_->save(inputFileName_);

}

void  PhotonOfflineClient::dividePlots(MonitorElement* dividend, MonitorElement* numerator, MonitorElement* denominator){
  double value,err;

  dividend->setEfficiencyFlag(); 
  if(denominator->getEntries()==0) return;

  for (int j=1; j<=numerator->getNbinsX(); j++){
    if (denominator->getBinContent(j)!=0){
      value = ((double) numerator->getBinContent(j))/((double) denominator->getBinContent(j));
      err = sqrt( value*(1-value) / ((double) denominator->getBinContent(j)) );
      dividend->setBinContent(j, value);
      dividend->setBinError(j,err);
    }
    else {
      dividend->setBinContent(j, 0);
      dividend->setBinError(j,0);
    }
    dividend->setEntries(numerator->getEntries());
  }
}


void  PhotonOfflineClient::dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator){
  double value,err;

  dividend->setEfficiencyFlag(); 
  for (int j=1; j<=numerator->getNbinsX(); j++){
    if (denominator!=0){
      value = ((double) numerator->getBinContent(j))/denominator;
      err = sqrt( value*(1-value) / denominator);
      dividend->setBinContent(j, value);
      dividend->setBinError(j,err);
    }
    else {
      dividend->setBinContent(j, 0);
    }
  }

}

MonitorElement* PhotonOfflineClient::bookHisto(string histoName, string title, int bin, double min, double max)
{

  int histo_index = 0;
  stringstream histo_number_stream;


  //determining which folder we're in
  if(dbe_->pwd().find( "InvMass" ) != string::npos){
    histo_index_invMass_++;
    histo_index = histo_index_invMass_;
  }
  if(dbe_->pwd().find( "Efficiencies" ) != string::npos){
    histo_index_efficiency_++;
    histo_index = histo_index_efficiency_;
  }
  histo_number_stream << "h_";
  if(histo_index<10)   histo_number_stream << "0";
  histo_number_stream << histo_index;

  return dbe_->book1D(histo_number_stream.str()+"_"+histoName,title,bin,min,max);

}

vector<vector<MonitorElement*> > PhotonOfflineClient::book2DHistoVector(string histoType, string histoName, string title,
									     int xbin, double xmin,double xmax,
									     int ybin, double ymin, double ymax)
{
  int histo_index = 0;

  vector<MonitorElement*> temp1DVector;
  vector<vector<MonitorElement*> > temp2DVector;

  //determining which folder we're in
  bool conversionPlot = false;
  if(dbe_->pwd().find( "Conversions" ) != string::npos) conversionPlot = true;


  if(conversionPlot){
    histo_index_conversions_++;
    histo_index = histo_index_conversions_;
  }
  else{
    histo_index_photons_++;
    histo_index = histo_index_photons_;
  }


  stringstream histo_number_stream;
  histo_number_stream << "h_";
  if(histo_index<10)   histo_number_stream << "0";
  histo_number_stream << histo_index << "_";


  for(int cut = 0; cut != numberOfSteps_; ++cut){ //looping over Et cut values

    for(uint type=0;type!=types_.size();++type){  //looping over isolation type

      currentFolder_.str("");
      currentFolder_ << "Egamma/"+analyzerName_+"/" << types_[type] << "Photons/Et above " << (cut+1)*cutStep_ << " GeV";
      if(conversionPlot) currentFolder_ << "/Conversions";

      dbe_->setCurrentFolder(currentFolder_.str());

      string kind;
      if(conversionPlot) kind = " Conversions: ";
      else kind = " Photons: ";

      if(histoType=="1D")           temp1DVector.push_back(dbe_->book1D(histo_number_stream.str()+histoName,types_[type]+kind+title,xbin,xmin,xmax));
      else if(histoType=="2D")      temp1DVector.push_back(dbe_->book2D(histo_number_stream.str()+histoName,types_[type]+kind+title,xbin,xmin,xmax,ybin,ymin,ymax));
      else if(histoType=="Profile") temp1DVector.push_back(dbe_->bookProfile(histo_number_stream.str()+histoName,types_[type]+kind+title,xbin,xmin,xmax,ybin,ymin,ymax,""));
      //else cout << "bad histoType\n";
    }

    temp2DVector.push_back(temp1DVector);
    temp1DVector.clear();
  }

  return temp2DVector;

}



vector<vector<vector<MonitorElement*> > > PhotonOfflineClient::book3DHistoVector(string histoType, string histoName, string title,
									     int xbin, double xmin,double xmax,
									     int ybin, double ymin, double ymax)
{
  int histo_index = 0;

  vector<MonitorElement*> temp1DVector;
  vector<vector<MonitorElement*> > temp2DVector;
  vector<vector<vector<MonitorElement*> > > temp3DVector;


  //determining which folder we're in
  bool conversionPlot = false;
  if(dbe_->pwd().find( "Conversions" ) != string::npos) conversionPlot = true;


  if(conversionPlot){
    histo_index_conversions_++;
    histo_index = histo_index_conversions_;
  }
  else{
    histo_index_photons_++;
    histo_index = histo_index_photons_;
  }

  stringstream histo_number_stream;
  histo_number_stream << "h_";
  if(histo_index<10)   histo_number_stream << "0";
  histo_number_stream << histo_index << "_";


  for(int cut = 0; cut != numberOfSteps_; ++cut){     //looping over Et cut values

    for(uint type=0;type!=types_.size();++type){      //looping over isolation type

      for(uint part=0;part!=parts_.size();++part){    //looping over different parts of the ecal

	currentFolder_.str("");
	currentFolder_ << "Egamma/"+analyzerName_+"/" << types_[type] << "Photons/Et above " << (cut+1)*cutStep_ << " GeV";
	if(conversionPlot) currentFolder_ << "/Conversions";

	dbe_->setCurrentFolder(currentFolder_.str());

	string kind;
	if(conversionPlot) kind = " Conversions: ";
	else kind = " Photons: ";

	if(histoType=="1D")           temp1DVector.push_back(dbe_->book1D(      histo_number_stream.str()+histoName+parts_[part],types_[type]+kind+parts_[part]+": "+title,xbin,xmin,xmax));
	else if(histoType=="2D")      temp1DVector.push_back(dbe_->book2D(      histo_number_stream.str()+histoName+parts_[part],types_[type]+kind+parts_[part]+": "+title,xbin,xmin,xmax,ybin,ymin,ymax));
	else if(histoType=="Profile") temp1DVector.push_back(dbe_->bookProfile( histo_number_stream.str()+histoName+parts_[part],types_[type]+kind+parts_[part]+": "+title,xbin,xmin,xmax,ybin,ymin,ymax,""));
	//else cout << "bad histoType\n";


      }

      temp2DVector.push_back(temp1DVector);
      temp1DVector.clear();
    }

    temp3DVector.push_back(temp2DVector);
    temp2DVector.clear();
  }

  return temp3DVector;
}


MonitorElement* PhotonOfflineClient::retrieveHisto(string dir, string name){
  //cout << "dir = " << dir << endl;
  //cout << "name = " << name << endl;
  vector<MonitorElement*> histoVector;
  uint indexOfRelevantHistogram=0;
  string fullMEName = "";
  histoVector = dbe_->getContents(dir);
  for(uint index=0;index!=histoVector.size();index++){
    string MEName = histoVector[index]->getName();
    if(MEName.find( name ) != string::npos){
      indexOfRelevantHistogram = index;
      break;
    }
  }
  return histoVector[indexOfRelevantHistogram];
}

