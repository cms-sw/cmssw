#include "GeneratorInterface/RivetInterface/interface/RivetHarvesting.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/Common/interface/Handle.h"

#include "Rivet/AnalysisHandler.hh"
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "LWH/AIManagedObject.h"
#include "FWCore/Utilities/interface/tinyxml.h"

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace Rivet;
using namespace edm;
using namespace std;

RivetHarvesting::RivetHarvesting(const edm::ParameterSet& pset) : 
_analysisHandler(),
_fileNames(pset.getParameter<std::vector<std::string> >("FilesToHarvest")),
_sumOfWeights(pset.getParameter<std::vector<double> >("VSumOfWeights")),
_crossSections(pset.getParameter<std::vector<double> >("VCrossSections")),
_outFileName(pset.getParameter<std::string>("OutputFile")),
_isFirstEvent(true),
_hepmcCollection(pset.getParameter<edm::InputTag>("HepMCCollection")),
_analysisNames(pset.getParameter<std::vector<std::string> >("AnalysisNames"))
{

  if (_sumOfWeights.size() != _fileNames.size() ||
      _sumOfWeights.size() != _crossSections.size() ||
      _fileNames.size()    != _crossSections.size()){
    throw cms::Exception("RivetHarvesting") << "Mismatch in vector sizes: FilesToHarvest: " << _sumOfWeights.size() << ", VSumOfWeights: " << _sumOfWeights.size() << ", VCrossSections: " << _crossSections.size();  
  }    
        
  
  //get the analyses
  _analysisHandler.addAnalyses(_analysisNames);

  //go through the analyses and check those that need the cross section
  const std::set< AnaHandle, AnaHandleLess > & analyses = _analysisHandler.analyses();

  std::set< AnaHandle, AnaHandleLess >::const_iterator ibeg = analyses.begin();
  std::set< AnaHandle, AnaHandleLess >::const_iterator iend = analyses.end();
  std::set< AnaHandle, AnaHandleLess >::const_iterator iana; 
  double xsection = -1.;
  xsection = pset.getParameter<double>("CrossSection");
  for (iana = ibeg; iana != iend; ++iana){
    if ((*iana)->needsCrossSection())
      (*iana)->setCrossSection(xsection);
  }

  double totalSumOfWeights = _sumOfWeights[0];
  _lumis.push_back(_sumOfWeights[0]/_crossSections[0]); 
  for (unsigned int i = 1; i < _sumOfWeights.size(); ++i){
    _lumis.push_back(_sumOfWeights[i]/_crossSections[i]);
    totalSumOfWeights += _sumOfWeights[i]*_lumis[0]/_lumis[i];
  }
  _analysisHandler.setSumOfWeights(totalSumOfWeights);  


}

RivetHarvesting::~RivetHarvesting(){
}

void RivetHarvesting::beginJob(){
  //set the environment, very ugly but rivet is monolithic when it comes to paths
  char * cmsswbase    = getenv("CMSSW_BASE");
  char * cmsswrelease = getenv("CMSSW_RELEASE_BASE");
  std::string rivetref, rivetinfo;
  rivetref = "RIVET_REF_PATH=" + string(cmsswbase) + "/src/GeneratorInterface/RivetInterface/data:" + string(cmsswrelease) + "/src/GeneratorInterface/RivetInterface/data";
  rivetinfo = "RIVET_INFO_PATH=" + string(cmsswbase) + "/src/GeneratorInterface/RivetInterface/data:" + string(cmsswrelease) + "/src/GeneratorInterface/RivetInterface/data";
  putenv(strdup(rivetref.c_str()));
  putenv(strdup(rivetinfo.c_str()));
}

void RivetHarvesting::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  return;
}

void RivetHarvesting::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup){
  if (!_isFirstEvent)
    return;

  //initialize the analysis handles, all histograms are booked
  //we need at least one event to get the handler initialized
  edm::Handle<HepMCProduct> evt;
  iEvent.getByLabel(_hepmcCollection, evt);

  // get HepMC GenEvent
  const HepMC::GenEvent *myGenEvent = evt->GetEvent();
  _analysisHandler.init(*myGenEvent);
  //gain access to the histogram factory and change the histograms
  AIDA::ITree & tree = _analysisHandler.tree();
  tree.ls(".", true);

  /*
  map<string, vector<DPSXYPoint> > existingHistos;
  vector<string>::const_iterator iFile;
  vector<string>::const_iterator iFileBeg = _fileNames.begin(); 
  vector<string>::const_iterator iFileEnd = _fileNames.end();
  for (iFile = iFileBeg; iFile != iFileEnd; ++iFile) {
    map<string, vector<DPSXYPoint> > thisFileHistos = getDPSXYValsErrs(*iFile);
    map<string, vector<DPSXYPoint> >::const_iterator iMap;
    map<string, vector<DPSXYPoint> >::const_iterator iMapBeg = thisFileHistos.begin();
    map<string, vector<DPSXYPoint> >::const_iterator iMapEnd = thisFileHistos.end();
    for (iMap = iMapBeg; iMap != iMapEnd; ++iMap){
      std::cout << iMap->first << " found in the original file " << *iFile << std::endl;
      existingHistos[iMap->first] = iMap->second;
    }
    //existingHistos.insert(existingHistos.end(), thisFileHistos.begin(), thisFileHistos.end());
  }
  */

  for (std::vector<std::string>::const_iterator iAna = _analysisNames.begin(); iAna != _analysisNames.end(); ++iAna){
    std::vector<std::string> listOfNames = tree.listObjectNames("./"+(*iAna), true);
    std::vector<std::string>::const_iterator iNameBeg = listOfNames.begin();
    std::vector<std::string>::const_iterator iNameEnd = listOfNames.end();
    for (std::vector<std::string>::const_iterator iName = iNameBeg; iName != iNameEnd; ++iName ){
      AIDA::IManagedObject * iObj = tree.find(*iName);
      if (!iObj){
        std::cout << *iName << " not found; SKIPPING!" << std::endl;
        continue;
      } 
      
      std::cout << *iName << " FOUND!" << std::endl;
      vector<string>::const_iterator iFile;
      vector<string>::const_iterator iFileBeg = _fileNames.begin(); 
      vector<string>::const_iterator iFileEnd = _fileNames.end();
      AIDA::IHistogram1D* histo = dynamic_cast<AIDA::IHistogram1D*>(iObj); 
      AIDA::IProfile1D*   prof  = dynamic_cast<AIDA::IProfile1D*>(iObj);
      string tmpdir = "/tmpdir";
      tree.mkdir(tmpdir);
      unsigned int ifc = 0;
      for (iFile = iFileBeg; iFile != iFileEnd; ++iFile) {
        std::cout << "opening file " << *iFile << std::endl;
        string name = *iName;
        string tostrip = *iAna+'/';
        name.replace(name.find(tostrip),tostrip.length(),"");
        name.replace(name.find("/"),1,"");
        cout << name << endl;
        vector<DPSXYPoint> original = getDPSXYValsErrs(*iFile, *iAna, name);
        if (histo){
          const string tmppath = tmpdir + "/" + name;
          cout << tmppath << endl;
          IHistogram1D* tmphisto = _analysisHandler.histogramFactory().createCopy(tmppath, *histo);
          tmphisto->reset();
          for (unsigned int i = 0; i < original.size(); ++i){
            tmphisto->fill(original[i].xval, original[i].yval);
          }
          tmphisto->scale(_lumis[ifc]);
          histo->add(*tmphisto);
          //iObj = new AIDA::IHistogram1D(*(_analysisHandler.histogramFactory().add(*iName, *histo, *tmphisto)));
          tree.rm(tmppath);
          //delete tmphisto;
        } else if (prof) {
          std::cout << *iName << "is a profile, doing nothing " << std::endl;
        } else {
          std::cout << *iName << " is neither a IHistogram1D neither a IProfile1D. Doing nothing with it." << std::endl;
        }
        ++ifc;
      }
      cout << iObj << endl;
    }
  }

  tree.ls(".", true);

  _isFirstEvent = false;
}


void RivetHarvesting::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  return;
}

void RivetHarvesting::endJob(){
  _analysisHandler.finalize();   
  _analysisHandler.writeData(_outFileName);
}

vector<DPSXYPoint> RivetHarvesting::getDPSXYValsErrs(std::string filename, std::string path, std::string name) {
    // Open AIDA XML file
    TiXmlDocument doc(filename);
    doc.LoadFile();
    if (doc.Error()) {
      string err = "Error in " + string(doc.Value());
      err += ": " + string(doc.ErrorDesc());
      cerr << err << endl;
      throw cms::Exception("RivetHarvesting") << "Cannot open " << filename;
    }

    // Return value, to be populated
    vector<DPSXYPoint> rtn;

    try {
      // Walk down tree to get to the <paper> element
      const TiXmlNode* aidaN = doc.FirstChild("aida");
      if (!aidaN) throw cms::Exception("RivetHarvesting") << "Couldn't get <aida> root element";
      for (const TiXmlNode* dpsN = aidaN->FirstChild("dataPointSet"); dpsN; dpsN = dpsN->NextSibling()) {
        const TiXmlElement* dpsE = dpsN->ToElement();
        const string plotname = dpsE->Attribute("name");
        const string plotpath = dpsE->Attribute("path");
        if (plotpath != path && plotname != name)
          continue;
        /// Check path to make sure that this is a reference histogram.
        //if (plotpath.find("/REF") != 0) {
        //  cerr << "Skipping non-reference histogram " << plotname << endl;
        //  continue;
        //}

        /// @todo Check that "path" matches filename
        vector<DPSXYPoint> points;
        for (const TiXmlNode* dpN = dpsN->FirstChild("dataPoint"); dpN; dpN = dpN->NextSibling()) {
          const TiXmlNode* xMeasN = dpN->FirstChild("measurement");
          const TiXmlNode* yMeasN = xMeasN->NextSibling();
          if (xMeasN && yMeasN)  {
            const TiXmlElement* xMeasE = xMeasN->ToElement();
            const TiXmlElement* yMeasE = yMeasN->ToElement();
            const string xcentreStr   = xMeasE->Attribute("value");
            const string xerrplusStr  = xMeasE->Attribute("errorPlus");
            const string xerrminusStr = xMeasE->Attribute("errorMinus");
            const string ycentreStr   = yMeasE->Attribute("value");
            const string yerrplusStr  = yMeasE->Attribute("errorPlus");
            const string yerrminusStr = yMeasE->Attribute("errorMinus");
            //if (!centreStr) throw Error("Couldn't get a valid bin centre");
            //if (!errplusStr) throw Error("Couldn't get a valid bin err+");
            //if (!errminusStr) throw Error("Couldn't get a valid bin err-");
            istringstream xssC(xcentreStr);
            istringstream xssP(xerrplusStr);
            istringstream xssM(xerrminusStr);
            istringstream yssC(ycentreStr);
            istringstream yssP(yerrplusStr);
            istringstream yssM(yerrminusStr);
            double xcentre, xerrplus, xerrminus, ycentre, yerrplus, yerrminus;
            xssC >> xcentre; xssP >> xerrplus; xssM >> xerrminus;
            yssC >> ycentre; yssP >> yerrplus; yssM >> yerrminus;
            //cout << "  " << centre << " + " << errplus << " - " << errminus << endl;
            DPSXYPoint pt(xcentre, xerrminus, xerrplus, ycentre, yerrminus, yerrplus);
            points.push_back(pt);
          } else {
            cerr << "Couldn't get <measurement> tag" << endl;
            /// @todo Throw an exception here?
          }
        }

        return points;
      }

    }
    // Write out the error
    /// @todo Rethrow as a general XML failure.
    catch (std::exception& e) {
      cerr << e.what() << endl;
      throw;
    }

    throw cms::Exception("RivetHarvesting") << "could not find " << path << "/" << name << " in file " << filename;
    return rtn;
}

DEFINE_FWK_MODULE(RivetHarvesting);
